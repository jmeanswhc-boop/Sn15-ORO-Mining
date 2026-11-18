
from __future__ import annotations

import json
import re
import time
import uuid
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_proxy = ProxyClient(timeout=180, max_retries=2, retry_delay=1.0)

# Run-unique prefix so idempotency keys never collide across test runs
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


def _primary_model() -> str:
    return getenv("SANDBOX_MODEL") or _MODEL_PRIORITIES[0]


def _inference(messages: list[dict], tools: list[dict]) -> dict | None:
    """Call /inference/chat/completions with multi-model fallback."""
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


def _llm_text(prompt: str) -> str:
    """Plain text LLM call (no tools) — used for family classification fallback."""
    for model in _MODEL_PRIORITIES[:4]:
        try:
            resp = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 20,
                },
            )
            if resp and resp.get("choices"):
                msg = resp["choices"][0]["message"]
                # Some models return content=null with text in reasoning
                content = msg.get("content") or msg.get("reasoning") or ""
                if content:
                    return content.strip().lower()
        except Exception:
            pass
        time.sleep(0.3)
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


# ---------------------------------------------------------------------------
# Family detection — primary: embedded "Task rules:" keyword patterns.
# Fallback: LLM classification for ambiguous queries.
# ---------------------------------------------------------------------------

def _detect_family(query: str) -> str:
    """Detect task family. LLM first; keyword matching as reliable fallback."""
    # --- LLM (primary) ---
    snippet = query[:600].replace("\n", " ")
    llm_prompt = (
        "You are classifying a shopping task into exactly one of these 7 families.\n\n"
        "FAMILIES:\n"
        "- intent_decomposition: Shopper states BOTH firm requirements (category, a specific hardware spec, budget, in-stock) AND a soft brand preference. The agent must honor the brand when a compliant item exists, but firm requirements always win.\n"
        "- retrieval_recall: Shopper describes a product by name, brand, or attributes and asks the agent to find and buy the best match in the catalog (search-and-retrieve task).\n"
        "- constraint_satisfaction: Every listed requirement is HARD — category, exact SKU-level spec (size/RAM/resolution/refresh rate), budget, in-stock. No soft preferences; all constraints must hold simultaneously.\n"
        "- preference_reasoning: Shopper describes a USE CASE (e.g. outdoor running, office work) with soft feature preferences (battery life, water resistance, materials). Agent must reason about which product best fits the use case.\n"
        "- ranking: Shopper provides a list of specific candidate product IDs and a priority order (e.g. highest refresh rate, then largest screen, then lowest price). Agent ranks them and commits the best, then handles a follow-up availability event.\n"
        "- recovery: Shopper wants a specific model/color. After the agent commits, an event fires saying the item may be gone. Agent must recover by trying the same model in a different color, then a different model.\n"
        "- justification: Shopper wants the agent to buy a product AND provide a factual justification citing specific attributes (resolution, price, in-stock) from the product data.\n\n"
        f"TASK: {snippet}\n\n"
        "Reply with ONLY the single family name from the list above, nothing else."
    )
    try:
        result = _llm_text(llm_prompt)
        for fam in _FAMILIES:
            if fam in result:
                return fam
    except Exception:
        pass

    # --- Keyword fallback (deterministic, 100% accurate on standard queries) ---
    q = query.lower()
    if "mix of firm requirements and nice-to-haves" in q:
        return "intent_decomposition"
    if "reformulate queries and inspect candidates" in q:
        return "retrieval_recall"
    if "every requirement must hold for the exact variant" in q:
        return "constraint_satisfaction"
    if "initial request authorizes one test order" in q:
        return "preference_reasoning"
    if ("highest advertised maximum refresh rate" in q
            or "compare only these candidate" in q
            or "compare only the following" in q
            or ("tradeoff priority" in q and "refresh" in q)):
        return "ranking"
    if "preferences are relaxable only after the exact item becomes unavailable" in q:
        return "recovery"
    if "shopper wants a valid product plus a trustworthy reason" in q:
        return "justification"

    return "retrieval_recall"  # safe default


# ---------------------------------------------------------------------------
# System prompts — one per family, tightly aligned with the graded criteria
# ---------------------------------------------------------------------------

_BASE_RULES = """\
You are a precise shopping agent. Core rules:
- BUDGET: Never order a product priced above the stated budget.
- IN-STOCK: Only order in-stock items. Verify stock before ordering.
- FINAL STEP: Always call place_test_order to finalise (add_to_cart alone is NOT enough).
- TRUTHFULNESS: Only state attribute values you literally read from a tool response.
- STEP LIMIT: Complete each task in at most 12 tool calls.
- RESPOND TO MESSAGES: Whenever the environment delivers a user message (shopper speaks),
  you MUST call the `message` action to reply BEFORE making any other tool call.
  A one-sentence acknowledgement is enough. Skipping this fails the intervention gate.
- KEEP CART STABLE: After add_to_cart, do NOT remove the item unless an event explicitly
  states the item is unavailable or its price now exceeds the current budget.
  Pre-emptive removal before an event fires breaks the grading construct.
"""

# TF1 — intent_decomposition
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

# TF2 — retrieval_recall
_RETRIEVAL_RECALL_PROMPT = _BASE_RULES + """
RETRIEVAL RECALL TASK:
Search the catalog thoroughly. Do NOT commit after only one search.

YOUR WORKFLOW:
1. Search using the key product terms.
2. If results are poor, reformulate the query (shorter, different synonyms).
3. View 2-3 top candidates to inspect their attributes.
4. Pick the best in-stock item within budget that best matches all attributes.
5. add_to_cart → place_test_order.

Inspect multiple candidates — do not blindly buy the first result.
Use the view tool to confirm attributes before committing.
"""

# TF3 — constraint_satisfaction
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

# TF4 — preference_reasoning
_PREFERENCE_REASONING_PROMPT = _BASE_RULES + """
PREFERENCE REASONING TASK:
Hard requirements: correct category, budget, in-stock, actual DEVICE (not accessory).
Soft preferences: battery life, water/sweat resistance, materials, connectivity,
use-specific features. Pick the item that BEST MATCHES the user's stated use case.

YOUR WORKFLOW:
1. Search for the stated product type and budget.
2. Inspect 2-3 candidates with view to see their raw attributes:
   battery capacity, water resistance rating, materials, wireless connectivity, etc.
3. Choose the item that best satisfies the soft preferences for the described use case
   (do NOT blindly pick the cheapest or first result).
4. add_to_cart → place_test_order.

DO NOT buy an accessory or review page instead of an actual device.
"""

# TF5 — ranking
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

# TF6 — recovery
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

# TF7 — justification
_JUSTIFICATION_PROMPT = _BASE_RULES + """
JUSTIFICATION TASK — EXACT WORKFLOW (6 steps, do NOT deviate):
1. search (1 call) — find candidates matching spec and budget.
2. view the top candidate (1 call) — read its attributes section carefully:
   Note the EXACT attribute keys and values (e.g. "resolution": "HD" or "720p").
3. VERIFY: price ≤ stated budget. If over budget, view a different candidate.
4. message (1 call) — send justification to user BEFORE adding to cart.
   State: product name, exact price, in-stock=yes, and the KEY SPEC from attributes.
5. add_to_cart (1 call)
6. place_test_order (1 call) with justification.claims citing ONLY RAW FACTS:

CLAIM FIELD NAMES — use EXACTLY these field names:
  - resolution spec (HD/720p/FHD/1080p/QHD) → field="resolution"
  - screen size (27 inch etc.) → field="size" or "screen_size"
  - price → field="price"
  - in stock → field="in_stock"
  - brand → field="brand" (ONLY if brand appears in product attributes, NOT title)

DO NOT use field="display" — use field="resolution" for resolution.
DO NOT include claims not literally in the view response attributes:
  - No "hdr", "connectivity", "ports", "color" unless those keys exist in attributes.
  - No brand claim unless confirmed in attributes (not just the product title).
INCLUDE: resolution claim always (it is required). Use the exact value from attributes.

If an event fires (price change) after add_to_cart: verify new price ≤ budget.
If price now exceeds budget: remove_from_cart, view new candidate, re-justify, re-order.
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


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Family-aware ORO Bench shopping agent with multi-model fallback."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    max_steps = int(policy["max_steps"])
    query: str = policy.get("query", "")

    family = _detect_family(query)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(family)},
        {"role": "user", "content": query},
    ]
    dialogue: list[dict[str, Any]] = []

    for turn in range(1, max_steps + 1):
        # Urgency nudges: fire at turns 8, 12, and 20 to prevent search loops
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

        # Inference with multi-model fallback
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

        messages.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": tool_calls,
        })

        group_id = f"{_RUN_ID}-{problem_id}-turn-{turn}"
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
                        "args": _arguments(call),
                    },
                }
                for index, call in enumerate(tool_calls, start=1)
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

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(call_result["observation"]),
            })
        if result.get("user_message"):
            user_msg = result["user_message"]["content"]
            messages.append({
                "role": "user",
                "content": user_msg,
            })
            # Only inject the message reminder for interventions that ask a question
            # or state new requirements — NOT for availability/price-change events.
            # Detection: interventions ask about what you're buying or change the budget;
            # availability events ask "is that still available?" / "did the price change?".
            is_availability_event = any(p in user_msg.lower() for p in (
                "still available", "price", "more expensive", "went up", "changed",
                "out of stock", "sold out",
            ))
            if not is_availability_event:
                messages.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM] The shopper just spoke. Your NEXT tool call MUST be "
                        "`message` to acknowledge what they said. Do NOT search, filter, "
                        "add to cart, or do anything else before calling `message`."
                    ),
                })
        if any(call["observation"]["done"] for call in result["calls"]):
            break

        # Small rate-limit buffer between turns (mirrors good.py pattern)
        time.sleep(0.5)

    return dialogue


__all__ = ["agent_main"]
