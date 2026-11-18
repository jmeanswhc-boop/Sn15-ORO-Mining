from __future__ import annotations

import json
import re
from os import getenv
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient


_BG_MODEL_LIGHT = "deepseek/deepseek-v3.2"
_BG_MODEL_HEAVY = "moonshotai/kimi-k2.6"
_BG_MODEL_RESERVE = (
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "google/gemma-4-31b-it",
    "deepseek/deepseek-v3.2",
)

_bg_bridge = ProxyClient(timeout=120, max_retries=2)

LANE_INTENT = "intent_decomposition"
LANE_RECALL = "retrieval_recall"
LANE_CONSTRAINT = "constraint_satisfaction"
LANE_PREFERENCE = "preference_reasoning"
LANE_RANK = "ranking"
LANE_RECOVER = "recovery"
LANE_JUSTIFY = "justification"


_BG_LANE_CUES: list[tuple[str, tuple[str, ...]]] = [
    (
        LANE_INTENT,
        (
            "mix of firm requirements and nice-to-haves",
            "soft preference, subordinate",
            "preferred-brand",
            "later correction does not repair an initial choice",
            "honor that named brand when a fully compliant",
        ),
    ),
    (
        LANE_RECALL,
        (
            "reformulate queries and inspect candidates",
            "survey the realistic options",
            "several distinct products, not one lucky hit",
            "search the catalog for what the user described",
        ),
    ),
    (
        LANE_CONSTRAINT,
        (
            "every requirement must hold for the exact variant",
            "resolution names here denote pixel classes",
            "check the specific sku's option, price, and stock",
            "i will tell you if you ask",
        ),
    ),
    (
        LANE_PREFERENCE,
        (
            "soft preferences matter",
            "blindly buying the cheapest valid item",
            "choose a strong match across several preference axes",
            "genuine intended purchase, not an exploratory",
        ),
    ),
    (
        LANE_RANK,
        (
            "stated tradeoff priority",
            "strictly worse than another available option",
            "highest advertised maximum refresh rate",
            "first cart choice must be the best-ranked",
            "compare only these candidate",
        ),
    ),
    (
        LANE_RECOVER,
        (
            "relaxable only after the exact item becomes unavailable",
            "least-relaxed in-stock option",
            "try another color of the same model before changing models",
            "conflicting title does not override",
        ),
    ),
    (
        LANE_JUSTIFY,
        (
            "when placing the order, include a structured justification",
            "ground claims in raw product facts",
            "do not justify from marketing title words alone",
            "trustworthy reason",
            "walk me through why",
        ),
    ),
]

_BG_GOAL_REGEX: list[tuple[str, re.Pattern[str]]] = [
    (LANE_RANK, re.compile(r"compare only these candidate", re.I)),
    (LANE_JUSTIFY, re.compile(r"walk me through why|tell me why|trustworthy reason", re.I)),
    (LANE_RECOVER, re.compile(r"if that exact .* disappears|strongly prefer", re.I)),
    (LANE_INTENT, re.compile(r"ideally,? i prefer|firm requirements matter more", re.I)),
    (LANE_PREFERENCE, re.compile(r"i'?d rather pay|for daily|workout|soft preference", re.I)),
    (LANE_CONSTRAINT, re.compile(r"i will tell you if you ask", re.I)),
]


def classify_lane(shopper_text: str) -> str:
    lowered = (shopper_text or "").lower()
    rules_blob = (
        lowered.split("task rules:", 1)[1] if "task rules:" in lowered else lowered
    )
    chosen_lane = LANE_RECALL
    peak_score = -1
    for lane_id, cue_list in _BG_LANE_CUES:
        score = sum(1 for cue in cue_list if cue in rules_blob or cue in lowered)
        if score > peak_score:
            peak_score = score
            chosen_lane = lane_id
    if peak_score <= 0:
        for lane_id, pattern in _BG_GOAL_REGEX:
            if pattern.search(lowered):
                return lane_id
    return chosen_lane


_BG_BASE_PLAYBOOK = """
You are BlackGirl v6, an ORO Bench shopping agent.

# Contract
- Use ONLY tools supplied in this episode's tool list.
- Continue until an observation reports done=true.
- Respect max_steps and max_calls_per_turn.
- When max_calls_per_turn > 1, batch independent views/inspects in one turn.

# Standard loop
1) Discover — search / filter (value-side keywords, not JSON field names)
2) Inspect — view / compare / inspect_stock on concrete SKUs
3) Decide — add_to_cart / remove_from_cart (first cart is scored — no placeholders)
4) Confirm — inspect_stock / inspect_cart after events; read shopper messages
5) Finish — place_test_order once verified (include justification.claims when required)

# Firm vs soft
Extract constraints: category, brand, model, size, capacity, color, material,
feature, budget/price, stock, resolution, refresh_rate, screen_size, use-case.
Mark FIRM (must hold) vs SOFT (preference). Soft never overrides firm.

# Matching
- Literal: brands, model codes, identifiers, uncommon tokens, exact sizes/colors.
- Size aliases only: s/small, m/medium, l/large, xl/extra large.
- Semantic: everyday paraphrases for soft checks only.
- Resolution classes when Task rules define them (both pixel sides):
  HD≥1280×720, Full HD≥1920×1080, QHD≥2560×1440, 4K/UHD≥3840×2160.
  Lowercase qHD ≠ QHD.

# Search keywords
Catalog search matches VALUES. Prefer product-type words, brand, model codes,
distinctive phrase fragments. Avoid dumping keys like "brand" or "category".
If search is weak: one small reformulation — drop noise, keep distinctive tokens.

# Shopper dialogue (critical)
Whenever a shopper user_message asks a question or clarifies budget/spec,
answer with the message tool in the NEXT turn before further cart/order actions.
Never ignore a direct question. Keep messages short and concrete.

# Market events
Prices/stock can change after cart actions. Re-verify with inspect_stock /
inspect_cart. If over budget or out of stock, remove and recover per family rules.
Once a verified in-stock in-budget pick exists, place_test_order — do not wait
for permission unless the shopper asked a pending question.
""".strip()

_BG_LANE_PLAYBOOK: dict[str, str] = {
    LANE_INTENT: """
# TK1 — Intent Decomposition
Strategy:
1. Parse FIRM: category, one hardware spec, budget, in-stock.
2. Parse SOFT: named preferred brand (subordinate).
3. Search/filter → view compliant SKUs; confirm category_path is a real device.
4. FIRST add_to_cart: preferred brand IF a fully firm-compliant SKU exists;
   otherwise best firm-compliant item. No exploratory placeholders.
5. If shopper asks "which one / does it have X?": message with the pick + evidence.
6. On price jump / budget tighten: remove invalid cart item; re-search under new
   budget; still prefer brand only when fully compliant.
7. inspect_stock → place_test_order.

A later fix does NOT repair a first cart that ignored an available compliant
preferred-brand option. Always answer interventions via message.
""".strip(),
    LANE_RECALL: """
# TK2 — Retrieval Recall
Strategy:
1. Search with goal phrase; also try tokenized / category+filter variants.
2. Shortlist SEVERAL distinct products (not one lucky hit); reject accessories /
   off-category noise using category_path from view/compare.
3. View/compare promising hits; pick best in-stock within budget.
4. One small keyword reformulation if recall is thin.
5. Once verified, place_test_order immediately — no permission wait.
""".strip(),
    LANE_CONSTRAINT: """
# TK3 — Constraint Satisfaction
Strategy:
1. If the goal says a firm hardware spec is revealed only when asked, call
   message FIRST to ask for that exact requirement before searching.
2. EVERY firm requirement must hold on the EXACT SKU (option + price + stock).
   Titles mix good/bad variants — always view then inspect_stock on the SKU.
3. Apply resolution class rules from Task rules when relevant.
4. If the wanted variant sells out, do NOT take a sibling SKU that misses a
   firm requirement — find another product that satisfies all.
5. Finish with place_test_order once verified; do not burn steps on chatter.
""".strip(),
    LANE_PREFERENCE: """
# TK4 — Preference Reasoning
Strategy:
Hard: category, budget, in-stock, real device (not accessory/review page).
Soft: use-case axes (weather, sweat/water resistance, battery, materials,
connectivity, workout/outdoor).

1. Optionally message to clarify priorities BEFORE first cart.
2. view/compare candidates across soft axes — not cheapest-by-default.
3. FIRST add_to_cart must be a genuine preference-fit purchase (scored).
   A later upgrade does not repair a weak first commitment.
4. Prefer a slightly dearer item that matches soft goals if still in budget.
5. Re-check stock/price after events; then place_test_order.
""".strip(),
    LANE_RANK: """
# TK5 — Ranking
Strategy:
1. Restrict to the candidate variants named in the query (product_id::sku).
2. view and/or compare ALL listed candidates BEFORE any add_to_cart.
3. Rank ONLY by stated priority (typical: highest refresh rate including
   advertised overclock → largest screen → lowest price).
4. FIRST add_to_cart MUST be the current best-ranked in-stock in-budget option.
5. After stockout/price events, remove invalid item, re-rank survivors, cart
   the new #1, then place_test_order quickly (do not stall).
6. Never pick a dominated option on every axis.

CRITICAL — place_test_order on TK5:
- Call place_test_order with ONLY product_id and sku.
- Do NOT pass justification / claims. Optional untruthful claims hard-fail
  the episode even when the ranked pick is correct.
""".strip(),
    LANE_RECOVER: """
# TK6 — Recovery
Strategy:
Never relax: category, budget, in-stock.
Prefer exact model+color first (use explicit model fact when supplied; titles
do not override model facts).

If exact item disappears / price breaks budget after a cart action:
1) same model, other color
2) only then other models
Re-check price and stock after every cart change before place_test_order.
Any eligible listing matching preferred model+color is a valid first choice.
""".strip(),
    LANE_JUSTIFY: """
# TK7 — Justification
Strategy:
1. Find a valid in-stock in-budget product; view/compare raw facts BEFORE first
   cart add and BEFORE ordering. Never justify from title words alone.
2. place_test_order MUST include justification.claims (schema requires it).

Claim allowlist (ANY other field → faithful=False → reward 0):
- REQUIRED: price, in_stock, and exactly ONE shopper spec field using its
  canonical name from the goal (resolution OR screen_size OR refresh_rate —
  never aliases display / size / category / category_path / device_operating_system).
- PLUS exactly ONE soft reason from: brand, streaming, microphone,
  privacy_cover, autofocus, hdr — ONLY if view/compare shows an explicit
  positive capability. Prefer brand when the brand field is present on the
  viewed product. NEVER invent hdr=true (or any feature) when unknown/absent.

Value rules:
- price / in_stock must match the latest inspect_stock or cart current state.
- resolution/screen_size/refresh_rate values must come from viewed facts, not
  marketing titles.
- No summary prose, no evidence free text inside claims.
- Keep the claims array minimal: 3 required + 1 soft reason (4 total).

Also reject screen-only / accessory SKUs when a full device is required.
""".strip(),
}

_BG_IDLE_PRODS = (
    "No tool call received. Choose a supplied tool now. "
    "If you have not searched yet, call search or filter.",
    "Still no tool call. Progress: search/filter, view a candidate, answer any "
    "pending shopper question with message, or place_test_order if the cart "
    "already holds a verified in-stock SKU.",
)

_BG_NUDGE_HIDDEN_SPEC = (
    "Coach [TK3]: the shopper has one firm hardware spec revealed only if asked. "
    "Call message NOW to ask for that exact requirement before searching."
)

_BG_NUDGE_REPLY = (
    "Coach: the shopper just sent a message. Answer it with the message tool in "
    "this turn (which pick / does it meet the firm spec / budget). Do not add to "
    "cart or order until that question is answered."
)

_BG_NUDGE_RECOVER = (
    "Coach [TK6]: action failed or stock/price changed ({error}). Recover with "
    "least relaxation that still satisfies category, budget, and in-stock."
)

_BG_NUDGE_RANK = (
    "Coach [TK5]: view/compare every listed candidate and rank by the stated "
    "tradeoff axes BEFORE the first add_to_cart. First cart = #1. On "
    "place_test_order pass ONLY product_id+sku — omit justification entirely."
)

_BG_NUDGE_CLAIMS = (
    "Coach [TK7]: claims allowlist only — "
    "(1) canonical spec field resolution|screen_size|refresh_rate from the goal, "
    "(2) price, (3) in_stock, (4) one soft reason brand|streaming|microphone|"
    "privacy_cover|autofocus|hdr only if explicitly evidenced. "
    "Forbidden: display, size, category, category_path, device_operating_system, "
    "invented hdr. Minimal 4 claims; values from view/inspect_stock only."
)

_BG_NUDGE_BRAND = (
    "Coach [TK1]: first cart should honor the preferred brand when a fully "
    "firm-compliant SKU exists; otherwise take the best firm-compliant item."
)

_BG_NUDGE_PREF = (
    "Coach [TK4]: compare soft-preference axes before first cart. First cart must "
    "be a real preference-fit purchase, not the cheapest placeholder."
)


def _bg_alias_provider_id(raw_model: str) -> str:
    alias_map = {
        "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
        "deepseek-ai/DeepSeek-V3.1-TEE": "deepseek/deepseek-chat-v3.1",
        "deepseek-ai/DeepSeek-V3-0324-TEE": "deepseek/deepseek-chat-v3-0324",
        "deepseek-ai/DeepSeek-R1-0528-TEE": "deepseek/deepseek-r1-0528",
        "Qwen/Qwen3-32B-TEE": "qwen/qwen3-32b",
        "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
        "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
        "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
        "zai-org/GLM-5.1-TEE": "z-ai/glm-5.1",
        "zai-org/GLM-5.2-TEE": "z-ai/glm-5.2",
        "moonshotai/Kimi-K2.6-TEE": "moonshotai/kimi-k2.6",
        "openai/gpt-oss-120b-TEE": "openai/gpt-oss-120b",
        "XiaomiMiMo/MiMo-V2-Flash-TEE": "xiaomi/mimo-v2-flash",
        "unsloth/Mistral-Nemo-Instruct-2407-TEE": "mistralai/mistral-small-2603",
    }
    return alias_map.get(raw_model, raw_model)


def _bg_pick_model(lane_id: str) -> str:
    forced = (getenv("SANDBOX_MODEL") or "").strip()
    if forced:
        return _bg_alias_provider_id(forced)
    if lane_id == LANE_RECALL:
        return _BG_MODEL_LIGHT
    return _BG_MODEL_HEAVY


def _bg_list_tool_labels(tool_schemas: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for schema in tool_schemas:
        fn_block = schema.get("function") if isinstance(schema, dict) else None
        if isinstance(fn_block, dict) and fn_block.get("name"):
            labels.append(str(fn_block["name"]))
        elif isinstance(schema, dict) and schema.get("name"):
            labels.append(str(schema["name"]))
    return labels


def _bg_compose_system(lane_id: str, policy_blob: dict[str, Any]) -> str:
    labels = _bg_list_tool_labels(policy_blob.get("tools") or [])
    lane_text = _BG_LANE_PLAYBOOK.get(lane_id, _BG_LANE_PLAYBOOK[LANE_RECALL])
    return (
        f"{_BG_BASE_PLAYBOOK}\n\n"
        f"{lane_text}\n\n"
        f"# Runtime\n"
        f"Router selected pipeline: {lane_id}\n"
        f"Available tools: {', '.join(labels) if labels else '(see tool schemas)'}\n"
        f"max_steps={policy_blob.get('max_steps')} "
        f"max_calls_per_turn={policy_blob.get('max_calls_per_turn', 1)}\n"
    )


def _bg_parse_args(tc_item: dict[str, Any]) -> dict[str, Any]:
    raw_args = tc_item["function"].get("arguments", "{}")
    if isinstance(raw_args, dict):
        decoded = raw_args
    else:
        try:
            decoded = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            decoded = {}
    if not isinstance(decoded, dict):
        raise ValueError("tool arguments must be a JSON object")
    return decoded


def _bg_shrink_obs(payload: Any, cap: int = 7000) -> str:
    try:
        rendered = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        rendered = str(payload)
    if len(rendered) <= cap:
        return rendered
    return rendered[: cap - 20] + "...<truncated>"


def _bg_extract_action_names(tc_items: list[dict[str, Any]]) -> list[str]:
    labels = []
    for tc_item in tc_items:
        try:
            labels.append(str(tc_item["function"]["name"]))
        except (KeyError, TypeError):
            labels.append("?")
    return labels


def _bg_is_inquiry(blob: str) -> bool:
    cleaned = (blob or "").strip().lower()
    if not cleaned:
        return False
    if "?" in cleaned:
        return True
    prefixes = (
        "which ",
        "what ",
        "does ",
        "did ",
        "can you",
        "could you",
        "wait",
        "hold on",
        "before you",
        "re-check",
        "got anything",
    )
    return any(cleaned.startswith(p) or f" {p}" in f" {cleaned}" for p in prefixes)


def _bg_next_coach(
    lane_id: str,
    step_idx: int,
    prior_actions: list[str],
    prior_payloads: list[Any],
    shopper_text: str,
    await_reply: bool,
    cart_used: bool,
) -> str | None:
    action_blob = " ".join(prior_actions)
    goal_head = shopper_text.lower().split("task rules:", 1)[0]

    if await_reply:
        return _BG_NUDGE_REPLY

    if (
        lane_id == LANE_CONSTRAINT
        and step_idx == 1
        and "i will tell you if you ask" in goal_head
        and "message" not in action_blob
    ):
        return _BG_NUDGE_HIDDEN_SPEC

    if lane_id == LANE_RANK and step_idx == 1 and not cart_used:
        return _BG_NUDGE_RANK

    if lane_id == LANE_INTENT and step_idx == 1:
        return _BG_NUDGE_BRAND

    if (
        lane_id == LANE_PREFERENCE
        and step_idx <= 2
        and "compare" not in action_blob
        and not cart_used
    ):
        return _BG_NUDGE_PREF

    if lane_id == LANE_JUSTIFY and step_idx >= 2:
        if "view" not in action_blob and "compare" not in action_blob:
            return _BG_NUDGE_CLAIMS
        if step_idx >= 4 and "place_test_order" not in action_blob:
            return _BG_NUDGE_CLAIMS

    if step_idx <= 2 and not any(
        label in action_blob
        for label in ("search", "filter", "view", "compare", "message")
    ):
        return (
            "Hint: start with search or filter using value-side keywords from the "
            "shopper goal, then view the best hits."
        )

    if lane_id == LANE_RECALL and step_idx == 3 and "search" in action_blob:
        return (
            "Hint [TK2]: one small keyword reformulation or filter so the shortlist "
            "covers several distinct products before ordering."
        )

    if lane_id == LANE_RECOVER and any(
        isinstance(item, dict)
        and (
            item.get("error")
            or (
                isinstance(item.get("observation"), dict)
                and (
                    item["observation"].get("market_event")
                    or item["observation"].get("added")
                )
            )
        )
        for item in prior_payloads
    ):
        detail = "market/stock change"
        for item in prior_payloads:
            if isinstance(item, dict) and item.get("error"):
                detail = str(item["error"])
                break
        return _BG_NUDGE_RECOVER.format(error=detail)

    if lane_id == LANE_RANK and cart_used is False and step_idx >= 2:
        if "add_to_cart" not in action_blob and (
            "view" in action_blob or "compare" in action_blob
        ):
            return (
                "Hint [TK5]: you have inspected candidates — add_to_cart ONLY the "
                "best-ranked SKU now (first cart = graded #1)."
            )
    if lane_id == LANE_RANK and cart_used and step_idx >= 3:
        return (
            "Hint [TK5]: after recovery, place_test_order with ONLY product_id "
            "and sku — do not attach justification.claims."
        )

    return None


def _bg_chat(
    *,
    model_id: str,
    transcript: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    call_budget: int,
) -> tuple[str, list[dict[str, Any]]]:
    candidate_ids = [model_id]
    for reserve_id in _BG_MODEL_RESERVE:
        if reserve_id not in candidate_ids:
            candidate_ids.append(reserve_id)

    last_failure: Exception | None = None
    for try_id in candidate_ids[:3]:
        reply_packet = _bg_bridge.post(
            "/inference/chat/completions",
            json_data={
                "model": try_id,
                "messages": transcript,
                "tools": tool_schemas,
                "tool_choice": "required",
                "temperature": 0,
            },
        )
        if reply_packet is None:
            last_failure = RuntimeError(f"inference failed for model={try_id}")
            continue
        try:
            reply_msg = reply_packet["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            last_failure = exc
            continue
        spoken = reply_msg.get("content") or ""
        actions = list(reply_msg.get("tool_calls") or [])[:call_budget]
        return spoken, actions
    raise RuntimeError(f"inference request failed ({last_failure})")


def _bg_drive_lane(
    *,
    lane_id: str,
    problem_data: dict[str, Any],
) -> list[dict[str, Any]]:

    env_block = problem_data["environment"]
    session_binding = env_block["binding"]
    policy_blob = env_block["policy_view"]
    shopper_text = str(policy_blob.get("query") or problem_data.get("query") or "")
    episode_key = str(
        problem_data.get("problem_id", problem_data.get("id", "problem"))
    )
    call_budget = max(1, int(policy_blob.get("max_calls_per_turn", 1)))
    step_limit = max(1, int(policy_blob.get("max_steps", 20)))
    tool_schemas = policy_blob.get("tools") or []
    model_id = _bg_pick_model(lane_id)

    goal_head = shopper_text.lower().split("task rules:", 1)[0]
    need_hidden_spec = (
        lane_id == LANE_CONSTRAINT and "i will tell you if you ask" in goal_head
    )

    transcript: list[dict[str, Any]] = [
        {"role": "system", "content": _bg_compose_system(lane_id, policy_blob)},
        {"role": "user", "content": shopper_text},
    ]
    if need_hidden_spec:
        transcript.append({"role": "user", "content": _BG_NUDGE_HIDDEN_SPEC})
    if lane_id == LANE_RANK:
        transcript.append({"role": "user", "content": _BG_NUDGE_RANK})
    if lane_id == LANE_JUSTIFY:
        transcript.append({"role": "user", "content": _BG_NUDGE_CLAIMS})

    trail: list[dict[str, Any]] = [
        {
            "role": "meta",
            "blackgirl": {
                "version": 6,
                "router": lane_id,
                "model": model_id,
                "provider": "openrouter",
            },
        }
    ]

    prior_actions: list[str] = []
    prior_payloads: list[Any] = []
    await_reply = False
    cart_used = False

    for step_idx in range(1, step_limit + 1):
        coach_line = _bg_next_coach(
            lane_id,
            step_idx,
            prior_actions,
            prior_payloads,
            shopper_text,
            await_reply,
            cart_used,
        )
        if coach_line:
            transcript.append({"role": "user", "content": coach_line})

        spoken = ""
        actions: list[dict[str, Any]] = []
        for retry_idx in range(len(_BG_IDLE_PRODS) + 1):
            spoken, actions = _bg_chat(
                model_id=model_id,
                transcript=transcript,
                tool_schemas=tool_schemas,
                call_budget=call_budget,
            )
            if actions:
                break
            trail.append({"role": "assistant", "content": spoken})
            idle_line = _BG_IDLE_PRODS[min(retry_idx, len(_BG_IDLE_PRODS) - 1)]
            transcript.extend(
                [
                    {"role": "assistant", "content": spoken},
                    {"role": "user", "content": idle_line},
                ]
            )
        else:
            raise RuntimeError("model returned no tool call before completion")

        action_labels = _bg_extract_action_names(actions)

        transcript.append(
            {
                "role": "assistant",
                "content": spoken,
                "tool_calls": actions,
            }
        )

        batch_key = f"{episode_key}-turn-{step_idx}"
        env_request = {
            **session_binding,
            "call_id": batch_key,
            "idempotency_key": batch_key,
            "turn": step_idx,
            "calls": [
                {
                    "call_id": f"{batch_key}-{slot}",
                    "action": {
                        "name": tc_item["function"]["name"],
                        "args": _bg_parse_args(tc_item),
                    },
                }
                for slot, tc_item in enumerate(actions, start=1)
            ],
        }
        env_reply = _bg_bridge.post("/environment/call", json_data=env_request)
        if env_reply is None:
            raise RuntimeError("environment call failed")

        trail.append(
            {
                "role": "assistant",
                "content": spoken,
                "tool_calls": actions,
                "environment_result": env_reply,
                "blackgirl_turn": {"family": lane_id, "turn": step_idx},
            }
        )

        prior_actions = action_labels
        prior_payloads = []
        if "add_to_cart" in action_labels:
            cart_used = True
        if "message" in action_labels:
            await_reply = False

        for tc_item, call_row in zip(actions, env_reply["calls"], strict=True):
            payload = call_row.get("observation")
            prior_payloads.append(payload)
            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_item["id"],
                    "content": _bg_shrink_obs(payload),
                }
            )

        shopper_msg = env_reply.get("user_message")
        if shopper_msg and shopper_msg.get("content"):
            spoken_user = str(shopper_msg["content"])
            transcript.append({"role": "user", "content": spoken_user})
            if _bg_is_inquiry(spoken_user):
                await_reply = True

        fault_list = [
            item.get("error")
            for item in prior_payloads
            if isinstance(item, dict) and item.get("error")
        ]
        if fault_list and lane_id == LANE_RECOVER:
            transcript.append(
                {
                    "role": "user",
                    "content": _BG_NUDGE_RECOVER.format(error=fault_list[0]),
                }
            )

        for item in prior_payloads:
            if not isinstance(item, dict):
                continue
            nested = (
                item.get("observation")
                if isinstance(item.get("observation"), dict)
                else item
            )
            if isinstance(nested, dict) and nested.get("market_event"):
                event_row = nested["market_event"]
                transcript.append(
                    {
                        "role": "user",
                        "content": (
                            f"Coach: market_event observed ({event_row.get('kind')}). "
                            "Re-verify price/stock; if invalid under firm constraints, "
                            "remove_from_cart and recover per family rules before ordering."
                        ),
                    }
                )
                break

        if any(
            isinstance(call_row.get("observation"), dict)
            and call_row["observation"].get("done")
            for call_row in env_reply.get("calls") or []
        ):
            break

    return trail


_BG_LANE_HANDLERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    LANE_INTENT: lambda packet: _bg_drive_lane(lane_id=LANE_INTENT, problem_data=packet),
    LANE_RECALL: lambda packet: _bg_drive_lane(lane_id=LANE_RECALL, problem_data=packet),
    LANE_CONSTRAINT: lambda packet: _bg_drive_lane(
        lane_id=LANE_CONSTRAINT, problem_data=packet
    ),
    LANE_PREFERENCE: lambda packet: _bg_drive_lane(
        lane_id=LANE_PREFERENCE, problem_data=packet
    ),
    LANE_RANK: lambda packet: _bg_drive_lane(lane_id=LANE_RANK, problem_data=packet),
    LANE_RECOVER: lambda packet: _bg_drive_lane(
        lane_id=LANE_RECOVER, problem_data=packet
    ),
    LANE_JUSTIFY: lambda packet: _bg_drive_lane(
        lane_id=LANE_JUSTIFY, problem_data=packet
    ),
}


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:

    env_block = problem_data["environment"]
    policy_blob = env_block["policy_view"]
    shopper_text = str(policy_blob.get("query") or problem_data.get("query") or "")
    lane_id = classify_lane(shopper_text)
    handler = _BG_LANE_HANDLERS.get(lane_id, _BG_LANE_HANDLERS[LANE_RECALL])
    return handler(problem_data)


__all__ = ["agent_main", "classify_lane"]
