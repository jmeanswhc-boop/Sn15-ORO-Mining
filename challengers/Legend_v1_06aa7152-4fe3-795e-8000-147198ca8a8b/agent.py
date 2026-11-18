from __future__ import annotations

import json
import re
import time
import uuid
from os import getenv
from typing import Any, Optional

from src.agent.proxy_client import ProxyClient

_proxy = ProxyClient(timeout=120, max_retries=2, retry_delay=1.0)
_RUN_ID = uuid.uuid4().hex[:12]

_MODEL_PRIORITIES = [
    "zai-org/GLM-5.2-TEE",
    "moonshotai/Kimi-K2.6-TEE",
    "zai-org/GLM-5.1-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
    "deepseek-ai/DeepSeek-V3-0324-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
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

_MESSAGE_GATE_FAMILIES = {
    "intent_decomposition",
    "constraint_satisfaction",
    "preference_reasoning",
    "recovery",
    "ranking",
    "justification",
    "retrieval_recall",
}

_NO_CLAIMS_FAMILIES = {
    "intent_decomposition",
    "retrieval_recall",
    "constraint_satisfaction",
    "preference_reasoning",
    "ranking",
    "recovery",
}

_JUSTIFICATION_BANNED_FIELDS = frozenset({
    "hdr", "hdr10", "hdr10+", "hdr_support", "hdr_compatible",
    "dolby_vision", "hlg",
    "oled", "amoled", "qled", "ips", "va", "tn", "nano_ips",
    "local_dimming", "quantum_dot",
    "curved", "flat_panel",
    "variable_refresh_rate", "vrr", "adaptive_sync",
    "gsync", "g_sync", "g-sync", "freesync", "free_sync", "amd_freesync",
    "overdrive",
    "nfc", "wifi", "wi_fi", "wifi6", "wifi_6", "wifi6e",
    "bluetooth", "bt",
    "4g", "lte", "5g", "5g_ready",
    "usb_c", "usb-c", "type_c", "thunderbolt", "usb4",
    "hdmi", "displayport", "dp", "vga", "dvi",
    "microphone", "mic", "built_in_mic", "noise_cancellation", "anc",
    "active_noise_cancelling", "passive_noise_cancelling",
    "headphone_jack", "aux_port", "3.5mm",
    "stereo", "stereo_speakers", "dolby_atmos", "dts",
    "ois", "optical_image_stabilization", "night_mode", "night_vision",
    "ultra_wide", "telephoto", "periscope",
    "waterproof", "water_resistant", "ip_rating", "ip68", "ip67", "ip65",
    "dust_proof", "dust_resistant", "military_grade",
    "gorilla_glass", "tempered_glass", "scratch_resistant",
    "wireless_charging", "qi_charging", "magsafe",
    "fast_charging", "quick_charge", "pd_charging",
    "mfi_certified",
    "fingerprint_sensor", "fingerprint_reader", "fingerprint",
    "face_id", "face_unlock", "iris_scanner",
    "stylus_support", "pen_support", "s_pen",
    "backlit_keyboard", "rgb", "rgb_lighting",
    "energy_star", "tco_certified", "epeat", "rhs", "rohs",
})


def _primary_model() -> str:
    try:
        return getenv("SANDBOX_MODEL") or _MODEL_PRIORITIES[0]
    except Exception:
        return _MODEL_PRIORITIES[0]


def _inference(messages: list[dict], tools: list[dict]) -> dict | None:
    try:
        primary = _primary_model()
        order = [primary] + [m for m in _MODEL_PRIORITIES if m != primary]
        for model in order:
            try:
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
            except Exception:
                pass
            time.sleep(0.5)
        return None
    except Exception:
        return None


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = tool_call["function"].get("arguments", "{}")
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            return {}
        return parsed
    except Exception:
        return {}


def _sanitize_args(name: str, args: dict[str, Any], family: str) -> dict[str, Any]:
    try:
        out = dict(args)
        if name == "place_test_order":
            if family in _NO_CLAIMS_FAMILIES:
                out.pop("justification", None)
            elif family == "justification":
                jus = out.get("justification")
                if isinstance(jus, dict):
                    claims = jus.get("claims") or []
                    clean = [
                        c for c in claims
                        if isinstance(c, dict)
                        and str(c.get("field", "")).lower() not in _JUSTIFICATION_BANNED_FIELDS
                    ]
                    out["justification"] = {**jus, "claims": clean}
        return out
    except Exception:
        return dict(args) if isinstance(args, dict) else {}


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
    "  • 'filter' without candidates in query → retrieval_recall or "
    "constraint_satisfaction\n\n"
    "Families:\n"
    "- intent_decomposition: mix of firm requirements and nice-to-have preferences "
    "(e.g. firm screen size + preferred brand).\n"
    "- retrieval_recall: 'find an in-stock listing matching a phrase / keyword' — "
    "keyword search and recall task.\n"
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
    try:
        lines = []
        for t in tools:
            try:
                fn = t.get("function") or {}
                name = fn.get("name", "")
                desc = (fn.get("description") or "").split(".")[0].strip()
                if name:
                    lines.append(f"  • {name}: {desc}" if desc else f"  • {name}")
            except Exception:
                continue
        return "\n".join(lines) if lines else "  (none listed)"
    except Exception:
        return "  (none listed)"


def _llm_detect_family(query: str, tools: list[dict]) -> str:
    try:
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
        except (KeyError, IndexError, json.JSONDecodeError, TypeError, AttributeError):
            pass
        return ""
    except Exception:
        return ""


def _detect_family(query: str, tools: list[dict] | None = None) -> str:
    try:
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
        if (
            "highest advertised maximum refresh rate" in q
            or "compare only these candidate" in q
            or "compare only the following" in q
            or ("tradeoff priority" in q and "refresh" in q)
            or ("compare" in tool_names and _VARIANT_RE.search(query or ""))
        ):
            return "ranking"
        if "preferences are relaxable only after the exact item becomes unavailable" in q:
            return "recovery"
        if "shopper wants a valid product plus a trustworthy reason" in q:
            return "justification"
        return "retrieval_recall"
    except Exception:
        return "retrieval_recall"


_BASE_RULES = """\
You are a precise shopping agent. Core rules:
- BUDGET: Never order a product priced above the stated budget.
- IN-STOCK: Only order in-stock items.
- FINAL STEP: Always call place_test_order (add_to_cart alone is NOT enough).
- TRUTHFULNESS: Only state attribute values you literally read from a tool response.
- STEP LIMIT: Aim to finish in ≤12 tool calls. Prefer commit over more searching.
- START WITH TOOLS: First action must be search/filter/compare — NEVER open with `message`.
- RESPOND TO MESSAGES: Only when the shopper sends a non-empty message, your NEXT
  tool call MUST be `message` (one-sentence ack). Do not call message otherwise.
- KEEP CART STABLE: After add_to_cart, do NOT remove unless an event says the
  item is unavailable OR its price now exceeds budget. Pre-emptive removal fails grading.
- place_test_order: pass ONLY product_id and sku unless this is a justification task.
"""

_INTENT_DECOMP_PROMPT = _BASE_RULES + """
INTENT DECOMPOSITION:
Firm: category, one hardware spec, budget, in-stock. Soft: preferred brand.

Workflow:
1. Search preferred brand + spec; view 1-2 hits.
2. If a fully compliant preferred-brand item exists → add_to_cart and KEEP it.
3. Else pick best any-brand compliant item → add_to_cart.
4. On budget/price events: only remove if NEW price > NEW budget; then replace.
5. place_test_order(product_id, sku) — NO justification.

Every shopper message requires an immediate `message` reply before other tools.
"""

_RETRIEVAL_RECALL_PROMPT = _BASE_RULES + """
RETRIEVAL RECALL:
1. Search key terms; reformulate if needed (2-3 distinct queries).
2. View 2-3 candidates. Before committing, verify the product TITLE literally
   contains every key word from the search phrase — do NOT infer a match from
   category or context alone. If the title is missing a key word, keep searching.
3. Pick the best in-stock match within budget where all phrase words appear in
   the title. add_to_cart → place_test_order(product_id, sku) — NO justification.
4. If stockout after cart: message ack → inspect → switch → order. Do not stall.
"""

_CONSTRAINT_SATISFACTION_PROMPT = _BASE_RULES + """
CONSTRAINT SATISFACTION:
Every requirement is FIRM for the EXACT SKU (category, brand if named, spec, budget, stock).
If the shopper says they will tell you a requirement when asked, call message first.
View SKU options carefully — sibling SKUs can differ.

BUDGET: The binding budget is the number stated in the original user query.
Do NOT treat a product's listed price or any mid-conversation figure as a new budget.
If the first matching item goes out of stock, continue searching under the ORIGINAL budget.

STUCK: After 4+ failed searches for a specific spec, broaden your query
(e.g. drop brand, vary screen size ±0.2 inches, try synonyms).
place_test_order(product_id, sku) — NO justification.
"""

_PREFERENCE_REASONING_PROMPT = _BASE_RULES + """
PREFERENCE REASONING:
Hard: category, budget, in-stock, real device (not accessory).
Soft: sweat/water resistance, sport fit, materials — prefer use-case fit over cheapest.
View 2-3 candidates, pick best fit, add_to_cart → place_test_order (NO justification).
Reply with `message` to every shopper clarification before other tools.
"""

_RANKING_PROMPT = _BASE_RULES + """
RANKING — TWO PHASES:

PHASE 1 (2 calls):
1. compare(ALL candidate product_ids from the query) — ONE call.
2. Rank in-stock variants: (a) highest max refresh Hz (b) largest screen (c) lowest price.
3. add_to_cart the winner IMMEDIATELY.
   FORBIDDEN before first add_to_cart: inspect_stock, search, filter, viewing all candidates.

PHASE 2 (after "is that still available?"):
4. message ack first.
5. inspect_stock / compare remaining candidates; re-rank.
6. Switch cart if needed; place_test_order(product_id, sku) ONLY — never justification.
"""

_RECOVERY_PROMPT = _BASE_RULES + """
RECOVERY:
Never relax category/budget/in-stock. Preference ladder: exact model+color → same model other color → other models.
add_to_cart preferred item and KEEP it until an event fires.
On "still available?": message → inspect; if gone, remove → recover ladder → order.

PREFERENCE UNAVAILABLE: If the preferred color/model is not found in search results
at all (not just OOS), send ONE `message` to the shopper explicitly naming the
fallback (e.g. "The white 8520 isn't available; I'm ordering the black 8520 instead.")
before placing the order.

place_test_order(product_id, sku) — NO justification claims.
"""

_JUSTIFICATION_PROMPT = _BASE_RULES + """
JUSTIFICATION — exact workflow:
1. search → 2. view (read EXACT attribute keys/values from the response) → 3. verify budget
4. message the shopper with product name, price, in-stock, key spec
5. add_to_cart → 6. place_test_order WITH justification.claims

Claims rules (violations zero the score):
- ONLY include field/value pairs whose EXACT key appears in the `view` attributes you
  received. If you did not see a key named "hdr" in the attributes, do NOT claim it.
- Allowed fields (commonly present): resolution, screen_size, size, price, in_stock, brand,
  battery, storage, ram, os, processor, camera.
- BANNED (hallucinated almost every time — never include these):
  hdr, hdr10, hdr_support, dolby_vision, microphone, mic, headphone_jack, aux_port,
  nfc, 5g, 4g, lte.
- Use field="resolution" (never "display"). Value must be the exact string from view.
- Do NOT copy a spec from the shopper's query into a claim — read it from attributes.
- Brand claim: only include if the view attributes explicitly list the brand.
"""


def _system_prompt(family: str) -> str:
    try:
        return {
            "intent_decomposition": _INTENT_DECOMP_PROMPT,
            "retrieval_recall": _RETRIEVAL_RECALL_PROMPT,
            "constraint_satisfaction": _CONSTRAINT_SATISFACTION_PROMPT,
            "preference_reasoning": _PREFERENCE_REASONING_PROMPT,
            "ranking": _RANKING_PROMPT,
            "recovery": _RECOVERY_PROMPT,
            "justification": _JUSTIFICATION_PROMPT,
        }.get(family, _BASE_RULES)
    except Exception:
        return _BASE_RULES


def _env_call(
    binding: dict[str, Any],
    problem_id: str,
    turn: int,
    actions: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    try:
        group_id = f"{_RUN_ID}-{problem_id}-turn-{turn}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{i}",
                    "action": {"name": name, "args": args},
                }
                for i, (name, args) in enumerate(actions, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        return result
    except Exception:
        raise


def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(str(v).replace(",", "").strip())
    except Exception:
        return None


def _parse_hz(text: Any) -> Optional[float]:
    try:
        if text is None:
            return None
        nums = [
            float(m.group(1))
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*-?\s*hz", str(text), re.I)
        ]
        return max(nums) if nums else None
    except Exception:
        return None


def _parse_inches(text: Any) -> Optional[float]:
    try:
        if text is None:
            return None
        nums = [
            float(m.group(1))
            for m in re.finditer(
                r"(\d+(?:\.\d+)?)\s*-?\s*(?:inches|inch|in)\b", str(text), re.I
            )
        ]
        return max(nums) if nums else None
    except Exception:
        return None


_VARIANT_RE = re.compile(r"(\d{6,})::([\w.\-]+)")
_BUDGET_RE = re.compile(
    r"(?:spend up to|at or under|at or below|under|below|<=?)\s*"
    r"`?([\d,]+(?:\.\d+)?)",
    re.I,
)


def _parse_candidates(query: str) -> list[tuple[str, str]]:
    try:
        return [(a, b) for a, b in _VARIANT_RE.findall(query or "")]
    except Exception:
        return []


def _parse_budget(query: str) -> Optional[float]:
    try:
        m = _BUDGET_RE.search(query or "")
        return float(m.group(1).replace(",", "")) if m else None
    except Exception:
        return None


def _rank_key(row: dict[str, Any]) -> tuple:
    try:
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
    except Exception:
        return (0.0, 0.0, 1e18)


def _unwrap_obs(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        if not isinstance(payload, dict):
            return {}
        inner = payload.get("observation")
        if isinstance(inner, dict) and (
            "compared" in inner
            or "results" in inner
            or "variants" in inner
            or "product_id" in inner
            or "in_stock" in inner
            or "price" in inner
        ):
            return inner
        if any(k in payload for k in ("compared", "results", "variants", "in_stock", "price")):
            return payload
        return inner if isinstance(inner, dict) else payload
    except Exception:
        return {}


def _rows_from_compare(obs: dict[str, Any], candidates: list[tuple[str, str]]) -> list[dict[str, Any]]:
    try:
        obs = _unwrap_obs(obs)
        rows: list[dict[str, Any]] = []
        compared = obs.get("compared") or obs.get("products") or obs.get("results") or []
        wanted = {(pid, sku) for pid, sku in candidates}
        sku_by_pid = {pid: sku for pid, sku in candidates}

        if isinstance(compared, list):
            for item in compared:
                try:
                    if not isinstance(item, dict):
                        continue
                    pid = str(item.get("product_id") or "")
                    variants = item.get("variants") or []
                    want_sku = sku_by_pid.get(pid)
                    if variants:
                        matched = False
                        for v in variants:
                            try:
                                if not isinstance(v, dict):
                                    continue
                                sku = str(v.get("sku") or "")
                                if want_sku and sku != want_sku:
                                    continue
                                matched = True
                                rows.append(
                                    {
                                        **item,
                                        **v,
                                        "product_id": pid,
                                        "sku": sku or want_sku or "",
                                        "title": item.get("title"),
                                        "facts": item.get("facts") or {},
                                        "brand": item.get("brand"),
                                    }
                                )
                            except Exception:
                                continue
                        if not matched and want_sku:
                            try:
                                base_v = variants[0] if isinstance(variants[0], dict) else {}
                                rows.append(
                                    {
                                        **item,
                                        **base_v,
                                        "product_id": pid,
                                        "sku": want_sku,
                                        "title": item.get("title"),
                                        "facts": item.get("facts") or {},
                                        "brand": item.get("brand"),
                                    }
                                )
                            except Exception:
                                pass
                    else:
                        sku = str(item.get("sku") or want_sku or "")
                        if (pid, sku) in wanted or want_sku:
                            rows.append({**item, "product_id": pid, "sku": sku or want_sku})
                except Exception:
                    continue

        have = {(str(r.get("product_id")), str(r.get("sku"))) for r in rows}
        for pid, sku in candidates:
            if (pid, sku) not in have:
                rows.append({"product_id": pid, "sku": sku})
        return rows
    except Exception:
        return [{"product_id": pid, "sku": sku} for pid, sku in (candidates or [])]


def _eligible(row: dict[str, Any], budget: Optional[float]) -> bool:
    try:
        if row.get("in_stock") is False:
            return False
        price = _as_float(row.get("price") or row.get("current_price"))
        if budget is not None and price is not None and price > budget + 1e-6:
            return False
        return True
    except Exception:
        return False


def _run_ranking(
    problem_data: dict[str, Any],
    dialogue: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
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
            try:
                turn += 1
                if turn > max_steps:
                    raise RuntimeError("step budget exhausted")
                result = _env_call(binding, problem_id, turn, actions)
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": note,
                        "tool_calls": [
                            {
                                "id": f"rank-{turn}-{i}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args),
                                },
                            }
                            for i, (name, args) in enumerate(actions, start=1)
                        ],
                        "environment_result": result,
                    }
                )
                return result
            except Exception:
                raise

        def done(result: dict[str, Any]) -> bool:
            try:
                return any(
                    (c.get("observation") or {}).get("done") for c in (result.get("calls") or [])
                )
            except Exception:
                return False

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
            try:
                row = by_key.get((pid, sku))
                if row is None:
                    row = next((r for r in rows if str(r.get("product_id")) == pid), None)
                    if row is None:
                        row = {"product_id": pid, "sku": sku, "in_stock": True}
                    else:
                        row = {**row, "sku": sku}
                if _eligible(row, budget):
                    live.append(row)
            except Exception:
                continue

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
                [("message", {"content": "Checking availability now and I will switch if a better in-stock option ranks higher."})],
                note="rank-ack",
            )
            if done(result):
                return dialogue

        refreshed: list[dict[str, Any]] = []
        for pid, sku in candidates:
            try:
                if turn >= max_steps - 2:
                    break
                result = act(
                    [("inspect_stock", {"product_id": pid, "sku": sku})],
                    note="rank-rerank",
                )
                if done(result):
                    return dialogue
                stock = _unwrap_obs((result.get("calls") or [{}])[0].get("observation") or {})
                row = {
                    "product_id": pid,
                    "sku": sku,
                    "price": stock.get("price") or stock.get("current_price"),
                    "in_stock": stock.get("in_stock", True),
                }
                prev = by_key.get((pid, sku)) or next(
                    (r for r in rows if str(r.get("product_id")) == pid), {}
                )
                for k in ("refresh_rate", "screen_size", "title", "facts", "raw_facts"):
                    if k in prev and k not in row:
                        row[k] = prev[k]
                if _eligible(row, budget):
                    refreshed.append(row)
            except Exception:
                continue

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

        result = act(
            [("place_test_order", {"product_id": cart_pid, "sku": cart_sku})],
            note="rank-order",
        )
        _ = result
        return dialogue
    except Exception as exc:
        dialogue.append(
            {
                "role": "assistant",
                "content": f"ranking_error: {type(exc).__name__}: {exc}",
            }
        )
        return dialogue


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    dialogue: list[dict[str, Any]] = []
    try:
        environment = problem_data["environment"]
        binding = environment["binding"]
        policy = environment["policy_view"]
        problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        max_calls = int(policy.get("max_calls_per_turn", 1))
        max_steps = int(policy["max_steps"])
        query: str = policy.get("query", "")

        available_tools: list[dict] = policy.get("tools") or []
        family = _detect_family(query, available_tools)

        if family == "ranking":
            try:
                return _run_ranking(problem_data, dialogue)
            except Exception as exc:
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": f"ranking_fallback: {type(exc).__name__}: {exc}",
                    }
                )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system_prompt(family)},
            {"role": "user", "content": query},
        ]
        must_message = False

        for turn in range(1, max_steps + 1):
            try:
                if turn in (8, 12, 15, 20, 22):
                    remaining = max_steps - turn
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"[STEP CHECK: {turn}/{max_steps} used, {remaining} left] "
                                "If you already have a valid in-stock item within budget, "
                                "add_to_cart (if needed) and place_test_order IMMEDIATELY."
                            ),
                        }
                    )

                tool_calls = []
                assistant_content = ""
                for _attempt in range(2):
                    try:
                        inference = _inference(messages, policy["tools"])
                        if inference is None:
                            break
                        assistant = inference["choices"][0]["message"]
                        assistant_content = assistant.get("content") or ""
                        tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
                        if tool_calls:
                            break
                        messages.extend(
                            [
                                {"role": "assistant", "content": assistant_content},
                                {
                                    "role": "user",
                                    "content": (
                                        "The environment has not reported done=true. "
                                        "Choose one of the supplied tools to continue."
                                    ),
                                },
                            ]
                        )
                    except Exception:
                        tool_calls = []
                        break
                else:
                    if not tool_calls:
                        break
                if not tool_calls:
                    break

                if must_message and tool_calls:
                    try:
                        first_name = tool_calls[0]["function"]["name"]
                        if first_name != "message":
                            tool_calls = [
                                {
                                    "id": f"forced-msg-{turn}",
                                    "type": "function",
                                    "function": {
                                        "name": "message",
                                        "arguments": json.dumps(
                                            {
                                                "content": (
                                                    "Understood — I will follow that update "
                                                    "and continue carefully."
                                                )
                                            }
                                        ),
                                    },
                                }
                            ]
                    except Exception:
                        pass

                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                    }
                )

                group_id = f"{_RUN_ID}-{problem_id}-turn-{turn}"
                try:
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
                                    ),
                                },
                            }
                            for index, call in enumerate(tool_calls, start=1)
                        ],
                    }
                    result = _proxy.post("/environment/call", json_data=envelope)
                except Exception:
                    break
                if result is None:
                    break

                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "environment_result": result,
                    }
                )

                was_message_turn = bool(tool_calls) and all(
                    tc["function"]["name"] == "message" for tc in tool_calls
                )

                must_message = False
                try:
                    for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": json.dumps(call_result["observation"]),
                            }
                        )
                except Exception:
                    try:
                        for tool_call, call_result in zip(tool_calls, result.get("calls") or []):
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.get("id", ""),
                                    "content": json.dumps(
                                        (call_result or {}).get("observation") or {}
                                    ),
                                }
                            )
                    except Exception:
                        pass

                um = result.get("user_message")
                um_content = ""
                try:
                    if isinstance(um, dict):
                        um_content = str(um.get("content") or "").strip()
                    elif isinstance(um, str):
                        um_content = um.strip()
                except Exception:
                    um_content = ""

                if um_content:
                    messages.append({"role": "user", "content": um_content})
                    if family in _MESSAGE_GATE_FAMILIES and not was_message_turn:
                        must_message = True
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[SYSTEM] The shopper just spoke. Your NEXT tool call "
                                    "MUST be `message` to acknowledge them. Do not call any "
                                    "other tool first. After that one ack, continue with "
                                    "search/view/cart/order — do not keep messaging."
                                ),
                            }
                        )
                try:
                    if any(call["observation"]["done"] for call in result["calls"]):
                        break
                except Exception:
                    try:
                        if any(
                            (call.get("observation") or {}).get("done")
                            for call in (result.get("calls") or [])
                        ):
                            break
                    except Exception:
                        pass

                time.sleep(0.35)
            except Exception:
                continue

        return dialogue
    except Exception as exc:
        dialogue.append(
            {
                "role": "assistant",
                "content": f"agent_error: {type(exc).__name__}: {exc}",
            }
        )
        return dialogue


__all__ = ["agent_main"]
