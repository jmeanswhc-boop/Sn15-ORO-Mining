"""Lovelystar EnvPack v1 — LS shop-loop behaviors on ORO Bench tools.

Donor DNA: Lovelystar_tfDonor_r106_8a4a844b (ShoppingBench R106 #1 shop loop).
Chassis: 0828_envPack_closedRank_v3 EnvPack contract (binding, gates, closed set).

Ports LS *behaviors* (not ShoppingBench tools / P-S-V routers):
  TF1 — mutually exclusive sub-query decomp + shopper intervention reply
  TF2 — broad then narrow multi-query retrieval (≥ several distinct product_ids)
  TF3 — literal constraint / category_path checks after view
  TF4 — prefer cheaper among fully valid; soft prefs after hard constraints
  TF5 — closed candidate ranking when shopper lists product_id::sku
  TF6 — staged recovery (re-search → reformulate → cart fix); cap thrash
  TF7 — final requirements checklist; schema-safe justification claims only

Do not memorize suite IDs. Obey this episode's tools + Task rules only.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_OPENROUTER_MODELS = [
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "deepseek/deepseek-chat-v3.1",
]
_CHUTES_MODELS = [
    "moonshotai/Kimi-K2.6-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
]

_DEFAULT_CLAIM_FIELDS = frozenset(
    {
        "brand",
        "category",
        "in_stock",
        "price",
        "refresh_rate",
        "resolution",
        "screen_size",
    }
)
_FIELD_ALIASES = {
    "display": "resolution",
    "screen": "screen_size",
    "size": "screen_size",
    "screen size": "screen_size",
    "refresh": "refresh_rate",
    "hz": "refresh_rate",
    "category_path": "category",
}

_CAND_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9][A-Za-z0-9._-]*)")

_proxy = ProxyClient(timeout=120, max_retries=2)

_SYSTEM = """You are an ORO Bench shopping agent using Lovelystar-style shopping discipline
on EnvPack tools only.

Obey this episode's shopper text + Task rules. Race tasks are hidden — never memorize
product IDs or suite ranks. Use ONLY the tools supplied in this request. Continue until
an observation has done=true (usually after place_test_order).

## Phase A — Intent (TF1)  [Lovelystar decomp]
Before heavy retrieval, silently decompose the ask into mutually exclusive sub-goals
(one product / one decision each, max 4). Extract per sub-goal: keywords, firm price /
budget, firm specs, soft prefs. Do not invent products from filler words.
If the shopper asks a question, changes budget/spec, or asks which item you will buy:
  → NEXT action MUST be `message` with a short direct answer, then resume.
Empty shopper pings are not new questions — do not loop on empty acknowledgements.

## Phase B — Retrieval (TF2)  [broad → narrow]
Open-catalog path (no closed candidate list):
  1) search with core title tokens (complete tokens, any order).
  2) filter for brand / category / max_price / in_stock when helpful.
  3) Survey several distinct product_ids (≥5 when Task rules require) before buying.
  4) If hits are weak/empty: reformulate (drop soft brand, keep firm specs) and search again.
Never thrash: after ~3 failed reformulations, switch strategy (broader keywords or filter-first).

## Phase C — Constraints (TF3)
view (and compare) candidates. Treat category_path / SKU options / named hardware /
budget / in-stock as NON-NEGOTIABLE. Titles alone do not prove category. Match brands,
models, sizes, and codes LITERALLY — no close substitutes for firm specs.
Reject accessories when a full device is required.

## Phase D — Preferences & ranking (TF4 / TF5)
Among fully valid options: honor stated soft prefs; if still tied, prefer less expensive
(Lovelystar "prefer less expensive among equals") unless Task rules say otherwise.
If the shopper lists explicit variant keys like `product_id::sku` ("compare only these"):
  - Those keys ARE the entire eligible set. Do NOT open-catalog search for substitutes.
  - view each listed product_id (compare if helpful); read refresh/size/price/stock for the sku.
  - Rank by stated priority (often highest refresh → largest screen → lowest price).
  - First add_to_cart and final order MUST use a listed product_id::sku.

## Phase E — Recovery (TF6)  [shop-examination style]
When stock/price fails or cart is wrong:
  1) message if the shopper asked about availability.
  2) inspect_stock / inspect_cart; remove_from_cart if switching.
  3) Prefer same model other color → same color other model → then other models
     (do not jump to unrelated models while a same-model option remains).
  4) Re-search/filter with narrower criteria only after views prove the miss.
Do not place_test_order on a "promising" pick that failed constraint checks.

## Phase F — Finish / justification (TF7)
Before place_test_order: re-check EVERY firm requirement against viewed/compare facts
(Lovelystar final verify). Path:
  view/compare → inspect_stock → (message if asked) → add_to_cart → place_test_order.
Justification.claims ONLY if the shopper asked why/reason. Use schema fields only
(brand, category, in_stock, price, refresh_rate, resolution, screen_size). Omit otherwise.
Never invent fields (display, size, hdr, category_path, summary).
"""


def _provider() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw in ("chutes", "openrouter"):
        return raw
    if getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "chutes"


def _model_list() -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    return list(_OPENROUTER_MODELS if _provider() == "openrouter" else _CHUTES_MODELS)


def _shopper_text(query: str) -> str:
    return query.split("Task rules:", 1)[0].strip()


def _parse_candidate_variants(query: str) -> list[tuple[str, str]]:
    head = _shopper_text(query)
    low = head.lower()
    pairs = _CAND_KEY_RE.findall(head)
    if not any(
        s in low
        for s in (
            "compare only",
            "candidate variant",
            "candidate variants",
            "these candidate",
            "only these",
        )
    ):
        if len(pairs) < 2:
            return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pid, sku in pairs:
        sku = sku.rstrip(".,;:)")
        key = f"{pid}::{sku}"
        if key in seen:
            continue
        seen.add(key)
        out.append((pid, sku))
    return out


def _justification_required(query: str) -> bool:
    head = _shopper_text(query).lower()
    needles = (
        "why",
        "reason",
        "walk me through",
        "tell me why",
        "justif",
        "good pick",
        "reason i can trust",
    )
    return any(n in head for n in needles)


def _claim_field_allowlist(tools: list[dict[str, Any]]) -> frozenset[str]:
    for tool in tools:
        fn = tool.get("function") or tool
        if (fn.get("name") or "") != "place_test_order":
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        claims = (
            ((props.get("justification") or {}).get("properties") or {})
            .get("claims", {})
            .get("items", {})
            .get("properties", {})
            .get("field", {})
        )
        enum = claims.get("enum")
        if isinstance(enum, list) and enum:
            return frozenset(str(x) for x in enum)
    return _DEFAULT_CLAIM_FIELDS


def _normalize_claim_field(field: str, allowed: frozenset[str]) -> str | None:
    raw = str(field or "").strip()
    if not raw:
        return None
    key = raw.lower()
    if raw in allowed:
        return raw
    if key in allowed:
        return key
    mapped = _FIELD_ALIASES.get(key)
    if mapped and mapped in allowed:
        return mapped
    return None


def _normalize_claim_value(field: str, value: Any) -> str:
    text = str(value).strip()
    if field == "in_stock":
        low = text.lower()
        if low in ("true", "1", "yes", "in stock", "instock"):
            return "true"
        if low in ("false", "0", "no", "out of stock"):
            return "false"
    if field == "price":
        cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
        if cleaned:
            try:
                return str(float(cleaned))
            except ValueError:
                return text
    if field == "refresh_rate":
        m = re.search(r"(\d+(?:\.\d+)?)\s*hz", text, flags=re.I)
        if m:
            return f"{m.group(1)}Hz"
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            return f"{text}Hz"
    if field == "screen_size":
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:inch|in|\"|'')?", text, flags=re.I)
        if m:
            return f"{m.group(1)} inches"
    return text


def _sanitize_place_order_args(
    args: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
) -> dict[str, Any]:
    out = dict(args)
    if not _justification_required(query):
        out.pop("justification", None)
        return out
    just = out.get("justification")
    if not isinstance(just, dict):
        return out
    claims_in = just.get("claims")
    if not isinstance(claims_in, list):
        out.pop("justification", None)
        return out
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    for claim in claims_in:
        if not isinstance(claim, dict):
            continue
        field = _normalize_claim_field(str(claim.get("field", "")), allowed_fields)
        if field is None or field in seen:
            continue
        value = _normalize_claim_value(field, claim.get("value", ""))
        if value == "":
            continue
        cleaned.append({"field": field, "value": value})
        seen.add(field)
    if cleaned:
        out["justification"] = {"claims": cleaned}
    else:
        out.pop("justification", None)
    return out


def _arguments(
    tool_call: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    if (tool_call["function"].get("name") or "") == "place_test_order":
        return _sanitize_place_order_args(
            parsed, query=query, allowed_fields=allowed_fields
        )
    return parsed


def _looks_like_shopper_intervention(text: str) -> bool:
    low = (text or "").lower().strip()
    if len(low) < 8:
        return False
    cues = (
        "?",
        "which one",
        "before you buy",
        "hold on",
        "wait,",
        "still available",
        "i can only go",
        "non-negotiable",
        "re-check",
        "tell you if you ask",
        "firm requirement",
        "please keep to",
        "compare the current",
    )
    return any(c in low for c in cues)


def _chat(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    last_err: Exception | None = None
    for model in _model_list():
        try:
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if inference is None:
                last_err = RuntimeError(f"inference failed for {model}")
                continue
            return inference
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(f"inference unavailable: {last_err}")


def _env_call(envelope: dict[str, Any]) -> dict[str, Any]:
    base_id = str(envelope.get("call_id") or "call")
    for attempt in range(3):
        payload = dict(envelope)
        if attempt:
            suffix = f"r{attempt}-{int(time.time() * 1000) % 1_000_000}"
            payload["call_id"] = f"{base_id}-{suffix}"
            payload["idempotency_key"] = payload["call_id"]
            calls = []
            for index, call in enumerate(payload.get("calls") or [], start=1):
                c = dict(call)
                c["call_id"] = f"{payload['call_id']}-{index}"
                calls.append(c)
            payload["calls"] = calls
        result = _proxy.post("/environment/call", json_data=payload)
        if result is not None:
            return result
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"environment call failed after retries ({base_id})")


def _tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [str(c.get("function", {}).get("name") or "") for c in tool_calls]


def _closed_set_violation(
    tool_calls: list[dict[str, Any]],
    *,
    candidates: list[tuple[str, str]],
    viewed: set[str],
) -> str | None:
    if not candidates:
        return None
    cand_pids = {pid for pid, _ in candidates}
    cand_keys = {f"{pid}::{sku}" for pid, sku in candidates}
    names = _tool_names(tool_calls)
    if any(n in ("search", "filter") for n in names) and len(viewed & cand_pids) < min(
        2, len(cand_pids)
    ):
        return (
            "Closed candidate set: do not search/filter. "
            f"Call view on listed product_ids first: {sorted(cand_pids)}"
        )
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        raw = call["function"].get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        if name == "view":
            pid = str(args.get("product_id") or "")
            if pid and pid not in cand_pids:
                return (
                    f"view product_id {pid} is outside the listed candidates. "
                    f"View only: {sorted(cand_pids)}"
                )
        if name in ("add_to_cart", "inspect_stock", "place_test_order"):
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and pid not in cand_pids:
                return (
                    f"{name} product_id {pid} is outside listed candidates "
                    f"{sorted(cand_pids)}"
                )
            if pid and sku and f"{pid}::{sku}" not in cand_keys:
                if name in ("add_to_cart", "place_test_order"):
                    return (
                        f"{name} {pid}::{sku} is not a listed variant. "
                        f"Use one of: {sorted(cand_keys)}"
                    )
    return None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """EnvPack loop: Lovelystar shop phases + closed-set / claim / message gates."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    tools = policy["tools"]
    query = str(policy.get("query") or "")
    allowed_fields = _claim_field_allowlist(tools)
    candidates = _parse_candidate_variants(query)
    cand_pids = {pid for pid, _ in candidates}

    boot = (
        f"{query}\n\n"
        "Lovelystar procedure this episode:\n"
        "1) Decompose into mutually exclusive sub-goals (silent).\n"
        "2) Retrieve broad→narrow OR view closed candidates if listed.\n"
        "3) Constraint-check with view/compare; recover staged if needed.\n"
        "4) Final verify all firm requirements, then inspect_stock → cart → place_test_order.\n"
        "Reply to real shopper questions with message once, then continue."
    )
    if candidates:
        listed = ", ".join(f"{pid}::{sku}" for pid, sku in candidates)
        boot += (
            "\n\nCLOSED CANDIDATE SET (mandatory): "
            f"{listed}. "
            "view each of these product_ids (compare if helpful). Rank by the shopper's "
            "stated priority. Do not search the open catalog for other products. "
            "First cart and final order must use a listed product_id::sku."
        )
    else:
        boot += (
            "\n\nOpen-catalog path: search/filter until several distinct product_ids "
            "are surveyed; then view/compare; prefer less expensive among fully valid."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": boot},
    ]
    dialogue: list[dict[str, Any]] = []
    must_message = False
    viewed: set[str] = set()
    searchish = 0

    for turn in range(1, int(policy["max_steps"]) + 1):
        if must_message:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "REQUIRED this turn: call `message` once to answer the shopper, "
                        "then next turns resume shopping/ranking."
                    ),
                }
            )
        elif candidates and len(viewed & cand_pids) < len(cand_pids):
            missing = sorted(cand_pids - viewed)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Still missing views for closed-set product_ids: "
                        f"{missing}. Call view on the next missing id (or compare several)."
                    ),
                }
            )
        elif not candidates and searchish >= 3 and len(viewed) < 2:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Retrieval thrash guard (Lovelystar TF6): stop repeating the same "
                        "search. Reformulate keywords OR view the best hits so far, then "
                        "constraint-check before more search."
                    ),
                }
            )

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for _attempt in range(4):
            inference = _chat(messages=messages, tools=tools)
            assistant = inference["choices"][0]["message"]
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if not tool_calls:
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {
                            "role": "user",
                            "content": "No tool call. Call a supplied tool now.",
                        },
                    ]
                )
                continue
            if must_message and "message" not in _tool_names(tool_calls):
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": "Rejected. Call only `message` to answer the shopper.",
                        },
                    ]
                )
                continue
            if not must_message:
                violation = _closed_set_violation(
                    tool_calls, candidates=candidates, viewed=viewed
                )
                if violation:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": f"Rejected: {violation}"},
                        ]
                    )
                    continue
            break
        else:
            if candidates and len(viewed & cand_pids) < len(cand_pids):
                miss = sorted(cand_pids - viewed)[0]
                tool_calls = [
                    {
                        "id": f"forced-view-{turn}",
                        "type": "function",
                        "function": {
                            "name": "view",
                            "arguments": json.dumps({"product_id": miss}),
                        },
                    }
                ]
                assistant_content = f"Viewing closed-set candidate {miss}."
            else:
                raise RuntimeError("model returned no valid tool call before completion")

        if must_message:
            msg_only = [
                c
                for c in tool_calls
                if c.get("function", {}).get("name") == "message"
            ]
            if msg_only:
                tool_calls = msg_only[:1]

        names = _tool_names(tool_calls)
        if any(n in ("search", "filter") for n in names):
            searchish += 1

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
                    "action": {
                        "name": call["function"]["name"],
                        "args": _arguments(
                            call, query=query, allowed_fields=allowed_fields
                        ),
                    },
                }
                for index, call in enumerate(tool_calls, start=1)
            ],
        }
        result = _env_call(envelope)
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for call in tool_calls:
            if (call.get("function", {}).get("name") or "") == "view":
                raw = call["function"].get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = {}
                if isinstance(args, dict) and args.get("product_id"):
                    viewed.add(str(args["product_id"]))

        if "message" in names:
            must_message = False

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(call_result["observation"]),
                }
            )
        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            messages.append({"role": "user", "content": content})
            if _looks_like_shopper_intervention(str(content)):
                must_message = True
        if any(call["observation"].get("done") for call in result["calls"]):
            break

        if not must_message and not candidates:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Episode not done. Lovelystar finish: re-verify firm constraints "
                        "against viewed facts; inspect_stock; add_to_cart; place_test_order. "
                        "Omit justification unless the shopper asked for a reason. "
                        "If constraints fail, recover (TF6) — do not order a weak pick."
                    ),
                }
            )
        elif not must_message and candidates and len(viewed & cand_pids) >= len(cand_pids):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "All listed candidates viewed. Compare/rank by stated priority, "
                        "prefer less expensive among equals if tied, inspect_stock the winner, "
                        "add_to_cart that listed sku, place_test_order (no open-catalog substitutes)."
                    ),
                }
            )

    return dialogue


__all__ = ["agent_main"]
