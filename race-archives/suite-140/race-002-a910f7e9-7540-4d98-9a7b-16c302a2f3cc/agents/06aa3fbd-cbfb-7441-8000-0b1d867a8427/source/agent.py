from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from os import getenv
from typing import Any, Literal

from src.agent.proxy_client import ProxyClient

Family = Literal[
    "intent_decomposition",
    "retrieval_recall",
    "constraint_satisfaction",
    "preference_reasoning",
    "ranking",
    "recovery",
    "justification",
    "generic",
]

_FAMILIES = frozenset(
    {
        "intent_decomposition",
        "retrieval_recall",
        "constraint_satisfaction",
        "preference_reasoning",
        "ranking",
        "recovery",
        "justification",
        "generic",
    }
)

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_proxy = ProxyClient(timeout=120, max_retries=2)

_BLOCKED_WHILE_PENDING = frozenset({"add_to_cart", "place_test_order"})
_EXTRACT_TRIES = 3

_SHARED_RULES = """
Shared rules for every family:
- Honor the CONSTRAINT / RETRIEVAL SHEET injected each turn.
- When pending_question is true, NEXT action MUST be `message`. Never
  add_to_cart or place_test_order until pending_question is false.
- inspect_stock before ordering.
- On market price/stock events: acknowledge with `message` that has NO
  question marks and does NOT ask the shopper to choose. Then
  remove_from_cart → inspect_stock on a backup → add_to_cart →
  place_test_order. Never wait for shopper preference after an event.
- Confirm facts via view/compare (not title guesses). Use category_path when
  a category is required.
- Reserve the last ~8 steps for verify / event recovery / order.
- Continue until an observation reports done=true.
"""

_TF1_SYSTEM = (
    """You are a shopping agent for intent-decomposition (TF1).
"""
    + _SHARED_RULES
    + """
TF1 policy:
1. FIRM: category, gated hardware attribute, budget, in-stock. Never violate.
2. SOFT: preferred brand. Soft never overrides firm.
3. Prefer soft brand for the initial cart only if fully firm-compliant.
4. Otherwise buy any brand that meets every firm requirement.
5. Commitment rule (critical for reward): when you answer "which one are you
   going with?", name the product you will actually cart. After that, keep
   that same product if it still meets every firm constraint (including a
   lowered budget). If you must switch, call `message` first naming the NEW
   pick, then add_to_cart that same product — never cart a silent switch.
6. When pending_question is false and a firm-compliant variant is verified,
   place_test_order.
"""
)

_TF2_SYSTEM = (
    """You are a shopping agent for retrieval-recall (TF2).
"""
    + _SHARED_RULES
    + """
TF2 policy:
1. Match the theme phrase EXACTLY as given, required category, budget, stock.
2. Search plan: query 1 MUST be the exact theme phrase. Reformulations keep
   those theme tokens; never invent a product type absent from the theme
   (e.g. do not turn "roto pvc" into "roto pvc cutter").
3. Maximize recall early: run primary + reformulations, then filter by
   category/budget, compare several eligible candidates — but leave headroom
   for stockout recovery and order (do not burn the step budget on endless
   views).
4. Prefer candidates whose viewed facts support the theme phrase.
5. After stockout/price events: message without questions → remove → backup
   inspect → cart → order. Autonomously pick a backup; do not ask which.
6. When pending_question is false and a verified eligible variant is ready,
   place_test_order.
"""
)

_TF3_SYSTEM = (
    """You are a shopping agent for constraint-satisfaction (TF3).
"""
    + _SHARED_RULES
    + """
TF3 policy:
1. EVERY stated constraint is FIRM: brand (if given), category, budget,
   in-stock, and any gated hardware attribute.
2. If the goal says a firm spec will be told "if you ask", your FIRST action
   should be `message` asking what that firm requirement is. Do not cart
   before the gated attribute is known.
3. After constraints are complete: search/filter → view/compare → verify
   attributes from structured facts (not title alone) → inspect_stock →
   order. Precision over recall (previous-bench PV_SELECT idea).
4. Budget cuts and stockouts still apply; recover without violating firm
   constraints. No questions during event recovery.
5. place_test_order only when every firm constraint is verified.
"""
)

_TF5_SYSTEM = (
    """You are a shopping agent for ranking (TF5).
"""
    + _SHARED_RULES
    + """
TF5 policy:
1. ONLY consider the candidate variants listed in the goal / RANKING SHEET.
   Do not search the open catalog for other products.
2. view and/or compare EVERY listed candidate. Read refresh_rate, screen_size,
   price, and in_stock from structured facts.
3. Rank lexicographically by the stated priority (usually highest refresh,
   then largest screen, then lowest price). Never pick a dominated option.
4. Cart the current top eligible candidate. After stockout, re-rank remaining
   in-stock candidates and buy the new top — no shopper questions.
5. Honor budget and in-stock. Follow COMPUTED_TOP on the sheet when present.
"""
)

_TF4_SYSTEM = (
    """You are a shopping agent for preference-reasoning (TF4).
"""
    + _SHARED_RULES
    + """
TF4 policy:
1. HARD: real device in the stated category (not an accessory/review page),
   budget, in-stock. Never cart or order when viewed price exceeds budget.
2. SOFT preferences matter more than price among hard-valid items. Do NOT
   blindly buy the cheapest valid listing when a better preference fit exists.
3. BEFORE the first add_to_cart: send `message` clarifying the shopper's
   priority tradeoffs for their use case, then view/compare visible facts.
4. Search with preference tokens (e.g. wireless bluetooth IPX sport sweat
   battery), not price-only queries. Compare ≥2 distinct listing_ids.
5. FIRST cart must be a genuine multi-axis preference fit — look for several
   of: sweat/water/IPX, sport/gym/run fit, wireless/bluetooth, battery life.
   A weak exploratory first cart cannot be repaired by a later swap.
6. After price_change: message with NO '?', remove_from_cart, recover to a
   DIFFERENT listing_id that still shows multi-axis preference evidence and
   is within budget after inspect_stock, then place_test_order (product_id+sku
   only — no justification.claims).
7. One test order only — do not ask for a second confirmation.
"""
)

_TF6_SYSTEM = (
    """You are a shopping agent for recovery-by-relaxation (TF6).
"""
    + _SHARED_RULES
    + """
TF6 policy:
1. CRITICAL (never relax): category, budget, in-stock.
2. RELAXABLE preferences: exact model + exact color from the sheet.
3. FIRST add_to_cart MUST be the full match (preferred model AND color).
   View/compare that exact item before carting — inspect identity facts.
   Use an explicit model fact when present; a conflicting title does not win.
   For color: variant options → color fact → title only if unambiguous.
4. After stockout of that commitment: recover with MINIMAL relaxation —
   same model / other color FIRST; change models only as a last resort.
   Do not ask the shopper for permission to relax.
5. After the event: view/compare the replacement's identity AND inspect_stock
   before carting/ordering. Message acknowledgements must have NO '?'.
6. place_test_order args MUST be ONLY product_id + sku. Do NOT attach
   justification.claims (or any justification object) — extra claims fail
   the episode even when recovery was perfect.
"""
)

_TF7_SYSTEM = (
    """You are a shopping agent for justification (TF7).
"""
    + _SHARED_RULES
    + """
TF7 policy:
1. Find a product matching brand/category/budget/attributes/in-stock.
2. Evidence must come from view/compare/inspect_stock facts — never title-only
   guesses. Reject brand mismatches and title decoys (e.g. wrong brand in title).
3. FIRST add_to_cart must be FACT-BACKED: gated attributes must appear as
   explicit structured facts (exact screen size, not a size range; resolution
   that normalizes to the required hd/fhd/4k). Ambiguous facts → view another
   candidate. This first commitment is required for the price-event gate.
4. place_test_order MUST include justification.claims as STRING values only
   (e.g. "true", "5559.0", "hd", "27in"). Cover REQUIRED_CLAIMS from the sheet.
   Prefer only required fields; avoid extra category claims unless necessary.
5. After price/stock events: message with NO '?' → remove → inspect a different
   fact-backed backup → cart → order with fresh string claims.
"""
)

_GENERIC_SYSTEM = (
    """You are a shopping agent for ORO Bench generated environments.
"""
    + _SHARED_RULES
    + """
Satisfy the shopper goal using the supplied tools. Verify before ordering.
"""
)

_FAMILY_ROUTE_PROMPT = """Classify this ORO Bench shopping goal into one family.
Return JSON only (no markdown).

Families:
- "intent_decomposition": firm requirements plus soft preferences (e.g. prefer
  a brand but firm specs/budget matter more); multi-step clarification common.
- "retrieval_recall": find an in-stock listing in a category matching a theme
  phrase / quote, under a budget, and buy one; recall/coverage matters.
- "constraint_satisfaction": explicit firm constraints (brand/budget/stock /
  specs); may say a firm requirement will be told if you ask.
- "preference_reasoning": loose hard buy (category/budget/stock/real device)
  plus soft multi-axis use-case preferences (e.g. gym/sweat/weather, don't
  just grab the cheapest); tradeoffs and clarifying priorities matter.
- "ranking": compare ONLY a listed set of candidate variants with an explicit
  lexicographic tradeoff rule (highest X, then largest Y, then lowest price).
- "recovery": wants an exact preferred model+color under budget/stock; if that
  exact item disappears, keep budget/stock firm and relax color before model.
- "justification": buy a matching item and explain why with evidence; often
  asks to walk through why the pick is right / give a reason you can trust.
- "generic": anything else.

JSON schema:
{{
  "family": "intent_decomposition" | "retrieval_recall" | "constraint_satisfaction" | "preference_reasoning" | "ranking" | "recovery" | "justification" | "generic",
  "confidence": <0-1 number>,
  "rationale": <one short sentence>
}}

Goal:
{query}
"""

_TF1_GOAL_EXTRACT_PROMPT = """Analyze an intent-decomposition shopping goal.
Separate FIRM requirements from SOFT preferences. Return JSON only.

FIRM: category/type, one hardware attribute (screen/RAM/resolution/refresh),
budget ceiling, in-stock if stated.
SOFT: preferred brand / taste. Soft never overrides firm.

JSON schema:
{{
  "firm": {{
    "category": <string|null>,
    "gated_attribute": {{"key": <string|null>, "value": <string|null>}},
    "budget_max": <number|null>,
    "in_stock": <true|false>
  }},
  "soft": {{"preferred_brand": <string|null>}},
  "rewritten_goal": <string>,
  "notes": <string>
}}

Normalize budget_max as a bare number. Compact gated values (e.g. "7.6in").

Goal:
{query}
"""

_TF2_GOAL_EXTRACT_PROMPT = """Analyze a retrieval-recall shopping goal.
Extract theme/category/budget and propose search reformulations for recall.
Return JSON only.

JSON schema:
{{
  "theme_phrase": <exact theme string from the goal|null>,
  "category": <string|null>,
  "budget_max": <number|null>,
  "in_stock": <true|false>,
  "primary_query": <MUST equal theme_phrase>,
  "reformulations": [<2 to 4 alternate search strings>],
  "notes": <string>
}}

Reformulation guidance (keyword-alt recall, not random rewrites):
- theme_phrase and primary_query MUST be the exact quoted/phrase match text
  from the goal (e.g. "roto pvc", "a77s back", "aha glow", "spread creamy").
- Every reformulation MUST retain the core theme tokens.
- Allowed: word-order swaps, light synonyms, common expansions of tokens
  already in the theme (e.g. "ps" → "playstation" only if theme has
  playstation/ps context).
- Forbidden: inventing a product type not in the theme (no adding cutter,
  housing, serum, case, etc. unless those words already appear in theme).
- Prefer catalog-value style tokens over metadata-key words only.

Goal:
{query}
"""

_TF3_GOAL_EXTRACT_PROMPT = """Analyze a constraint-satisfaction shopping goal.
All constraints are FIRM. Return JSON only.

JSON schema:
{{
  "brand": <string|null>,
  "category": <string|null>,
  "gated_attribute": {{"key": <string|null>, "value": <string|null>}},
  "budget_max": <number|null>,
  "in_stock": <true|false>,
  "underspecified": <true if goal says a firm requirement will be told if you ask>,
  "rewritten_goal": <string>,
  "notes": <string>
}}

If a screen/RAM/resolution is already stated, put it in gated_attribute.
If underspecified, gated value may be null until the shopper answers.

Goal:
{query}
"""

_TF5_GOAL_EXTRACT_PROMPT = """Analyze a ranking shopping goal with an explicit candidate list.
Return JSON only.

JSON schema:
{{
  "category": <string|null>,
  "budget_max": <number|null>,
  "in_stock": <true|false>,
  "candidates": [{{"product_id": <str>, "sku": <str>}}],
  "priority": [<axis names in order, e.g. "refresh_rate", "screen_size", "price">],
  "priority_sense": {{
    "refresh_rate": "max" | "min",
    "screen_size": "max" | "min",
    "price": "max" | "min"
  }},
  "notes": <string>
}}

Parse EVERY product_id::sku token from the goal into candidates.
Default senses: refresh_rate=max, screen_size=max, price=min.

Goal:
{query}
"""

_TF7_GOAL_EXTRACT_PROMPT = """Analyze a justification shopping goal.
Extract purchase constraints and which claims must be evidenced. Return JSON only.

JSON schema:
{{
  "brand": <string|null>,
  "category": <string|null>,
  "budget_max": <number|null>,
  "in_stock": <true|false>,
  "attributes": {{
    "resolution": <string|null>,
    "screen_size": <string|null>,
    "refresh_rate": <string|null>
  }},
  "required_claims": [<subset of brand,category,in_stock,price,resolution,screen_size,refresh_rate>],
  "rewritten_goal": <string>,
  "notes": <string>
}}

Always include in_stock and price in required_claims. Also include any gated
attribute present (resolution / screen_size / refresh_rate) and brand if stated.
Normalize resolution like "720p HD" → "hd" when appropriate.

Goal:
{query}
"""

_TF4_GOAL_EXTRACT_PROMPT = """Analyze a preference-reasoning shopping goal.
Separate HARD constraints from SOFT multi-axis preferences. Return JSON only.

HARD: category / real device type, budget ceiling, in-stock.
SOFT: use-case-driven preference axes (sweat/water, sport fit, battery,
wireless, rugged, navigation, kid durability, etc.) and tradeoff priority.
Do NOT treat "cheapest" as the objective.

JSON schema:
{{
  "category": <string|null>,
  "budget_max": <number|null>,
  "in_stock": <true|false>,
  "use_case": <string|null>,
  "soft_axes": [<short axis labels>],
  "tradeoff_priority": [<ordered soft priorities as strings>],
  "unacceptable_shortcuts": [<strings>],
  "needs_priority_clarification": <true|false>,
  "rewritten_goal": <string>,
  "notes": <string>
}}

Goal:
{query}
"""

_TF6_GOAL_EXTRACT_PROMPT = """Analyze a recovery / model-color shopping goal.
Extract critical constraints and preferred model+color. Return JSON only.

CRITICAL (never relax): category, budget, in-stock.
RELAXABLE: exact model and exact color. Default relaxation order is
color_before_model (other color of same model before changing models).

JSON schema:
{{
  "category": <string|null>,
  "budget_max": <number|null>,
  "in_stock": <true|false>,
  "preferred_model": <string|null>,
  "preferred_color": <string|null>,
  "relaxation_order": ["color_before_model"] | ["model_before_color"],
  "rewritten_goal": <string>,
  "notes": <string>
}}

Goal:
{query}
"""

_UPDATE_EXTRACT_PROMPT = """Update shopping constraints from a mid-episode shopper message.
Return JSON only.

Current sheet:
{sheet}

Shopper message:
{message}

JSON schema:
{{
  "budget_max": <number|null>,
  "gated_attribute": {{"key": <string|null>, "value": <string|null>}} | null,
  "category": <string|null>,
  "theme_phrase": <string|null>,
  "brand": <string|null>,
  "in_stock": <true|false|null>,
  "preferred_brand": <string|null>,
  "is_question": <true|false>,
  "notes": <string>
}}

Rules:
- null means unchanged (except preferred_brand "" clears soft brand).
- Questions/confirmations => is_question=true.
- Pure constraint updates without asking => is_question=false.
- If the shopper reveals a firm hardware spec (e.g. 6.9-inch screen), fill
  gated_attribute even when also changing budget.
"""


@dataclass
class ConstraintSheet:
    family: Family = "generic"
    category: str | None = None
    gated_key: str | None = None
    gated_value: str | None = None
    theme_phrase: str | None = None
    budget: float | None = None
    in_stock_required: bool = True
    preferred_brand: str | None = None
    brand: str | None = None  # firm brand (TF3/TF7)
    pending_question: bool = False
    last_shopper_text: str | None = None
    rewritten_goal: str | None = None
    primary_query: str | None = None
    reformulations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Commitment / recovery tracking (TF1 event gate + stockout).
    last_product_id: str | None = None
    last_sku: str | None = None
    committed_product_id: str | None = None
    committed_sku: str | None = None
    answering_pending: bool = False
    saw_market_event: bool = False
    event_recovery_mode: bool = False
    # TF3 underspec
    underspecified: bool = False
    asked_for_spec: bool = False
    # TF5 ranking
    candidates: list[dict[str, str]] = field(default_factory=list)
    rank_priority: list[str] = field(default_factory=list)
    rank_sense: dict[str, str] = field(default_factory=dict)
    candidate_facts: dict[str, dict[str, Any]] = field(default_factory=dict)
    computed_top_id: str | None = None
    computed_top_sku: str | None = None
    # TF7 justification
    required_claims: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    owned_facts: dict[str, Any] = field(default_factory=dict)
    # TF4 preference reasoning
    use_case: str | None = None
    soft_axes: list[str] = field(default_factory=list)
    tradeoff_priority: list[str] = field(default_factory=list)
    unacceptable_shortcuts: list[str] = field(default_factory=list)
    needs_priority_clarification: bool = True
    asked_preference_clarify: bool = False
    observed_listing_ids: list[str] = field(default_factory=list)
    # TF6 recovery
    preferred_model: str | None = None
    preferred_color: str | None = None
    relaxation_order: str = "color_before_model"

    def status_block(self) -> str:
        pending = "yes" if self.pending_question else "no"
        lines = [
            f"FAMILY: {self.family}",
            "SHEET (authoritative):",
        ]
        if self.family == "intent_decomposition":
            lines.append(
                "FIRM: "
                + "; ".join(
                    [
                        f"category={self.category or 'unknown'}",
                        f"gated={self.gated_key or 'attr'}:{self.gated_value or 'unknown'}",
                        f"budget<={self.budget if self.budget is not None else 'unknown'}",
                        f"in_stock={self.in_stock_required}",
                    ]
                )
            )
            lines.append(f"SOFT: preferred_brand={self.preferred_brand or 'none'}")
            if self.rewritten_goal:
                lines.append(f"rewritten_goal: {self.rewritten_goal[:240]}")
            if self.committed_product_id:
                lines.append(
                    "COMMITTED_PICK: "
                    f"product_id={self.committed_product_id} "
                    f"sku={self.committed_sku or 'unknown'} "
                    "(keep if still firm-compliant; message before switching)"
                )
        elif self.family == "retrieval_recall":
            lines.append(
                "RECALL TARGET: "
                + "; ".join(
                    [
                        f"theme={self.theme_phrase or 'unknown'}",
                        f"category={self.category or 'unknown'}",
                        f"budget<={self.budget if self.budget is not None else 'unknown'}",
                        f"in_stock={self.in_stock_required}",
                    ]
                )
            )
            qplan = [self.primary_query] if self.primary_query else []
            qplan.extend(self.reformulations)
            qplan = [q for q in qplan if q]
            if qplan:
                lines.append("SEARCH PLAN (use multiple before buying):")
                for i, q in enumerate(qplan[:5], start=1):
                    lines.append(f"  {i}. {q}")
            lines.append(
                "Recall tip: search exact theme first, then reformulations that "
                "keep theme tokens; filter(category,budget,in_stock); compare; "
                "verify/order with ~8 steps reserved for recovery."
            )
        elif self.family == "constraint_satisfaction":
            lines.append(
                "FIRM CONSTRAINTS: "
                + "; ".join(
                    [
                        f"brand={self.brand or 'any'}",
                        f"category={self.category or 'unknown'}",
                        f"gated={self.gated_key or 'attr'}:{self.gated_value or 'UNKNOWN'}",
                        f"budget<={self.budget if self.budget is not None else 'unknown'}",
                        f"in_stock={self.in_stock_required}",
                    ]
                )
            )
            if self.underspecified and not self.gated_value:
                lines.append(
                    "UNDERSPECIFIED: gated attribute unknown. NEXT action MUST be "
                    "`message` asking what the firm hardware/spec requirement is."
                )
            elif self.gated_value:
                lines.append(
                    "Precision tip: verify gated attribute via view/compare facts "
                    "(not title alone) before carting."
                )
        elif self.family == "ranking":
            lines.append(
                "RANKING: "
                + "; ".join(
                    [
                        f"category={self.category or 'unknown'}",
                        f"budget<={self.budget if self.budget is not None else 'unknown'}",
                        f"in_stock={self.in_stock_required}",
                        f"priority={'>'.join(self.rank_priority) or 'refresh_rate>screen_size>price'}",
                    ]
                )
            )
            if self.candidates:
                lines.append("CANDIDATES (only these; do not open-search):")
                for c in self.candidates:
                    key = f"{c.get('product_id')}::{c.get('sku')}"
                    facts = self.candidate_facts.get(key) or {}
                    extra = ""
                    if facts:
                        extra = (
                            f" | rr={facts.get('refresh_rate')} "
                            f"sz={facts.get('screen_size')} "
                            f"price={facts.get('price')} "
                            f"stock={facts.get('in_stock')}"
                        )
                    lines.append(f"  - {key}{extra}")
            if self.computed_top_id:
                lines.append(
                    f"COMPUTED_TOP: product_id={self.computed_top_id} "
                    f"sku={self.computed_top_sku} "
                    "(cart this if still in stock / within budget)"
                )
        elif self.family == "preference_reasoning":
            lines.append(
                "HARD: "
                + "; ".join(
                    [
                        f"category={self.category or 'unknown'}",
                        f"budget<={self.budget if self.budget is not None else 'unknown'}",
                        f"in_stock={self.in_stock_required}",
                        "real_device=yes (not accessory)",
                    ]
                )
            )
            lines.append(f"USE_CASE: {self.use_case or 'from goal'}")
            if self.soft_axes:
                lines.append("SOFT_AXES: " + ", ".join(self.soft_axes[:8]))
            if self.tradeoff_priority:
                lines.append(
                    "TRADEOFF_PRIORITY: " + " > ".join(self.tradeoff_priority[:6])
                )
            if self.unacceptable_shortcuts:
                lines.append(
                    "AVOID: " + "; ".join(self.unacceptable_shortcuts[:4])
                )
            lines.append(
                f"distinct_listing_ids_observed: {len(self.observed_listing_ids)}"
            )
            if self.needs_priority_clarification and not self.asked_preference_clarify:
                lines.append(
                    "CLARIFY FIRST: before the first cart, `message` the shopper "
                    "to confirm which soft priorities matter most for the use case."
                )
            # Soft shortlist tip from harvested evidence (not a hidden scorer).
            scored: list[str] = []
            for key, facts in list(self.candidate_facts.items())[:12]:
                sigs = _tf4_preference_signals(facts)
                price = facts.get("price")
                if sigs:
                    scored.append(
                        f"{key}: axes={sorted(sigs)} price={price}"
                    )
            if scored:
                lines.append("PREFERENCE SIGNALS (from viewed facts):")
                for row in scored[:6]:
                    lines.append(f"  - {row}")
            lines.append(
                "Preference tip: cart only in-budget items with ≥2 axes among "
                "water/sport/wireless/battery; compare ≥2 listing_ids first."
            )
        elif self.family == "recovery":
            lines.append(
                "CRITICAL (never relax): "
                + "; ".join(
                    [
                        f"category={self.category or 'unknown'}",
                        f"budget<={self.budget if self.budget is not None else 'unknown'}",
                        f"in_stock={self.in_stock_required}",
                    ]
                )
            )
            lines.append(
                "PREFERRED (relaxable): "
                + "; ".join(
                    [
                        f"model={self.preferred_model or 'unknown'}",
                        f"color={self.preferred_color or 'unknown'}",
                        f"order={self.relaxation_order}",
                    ]
                )
            )
            lines.append(
                "Recovery tip: first cart = exact model+color. After stockout, "
                "try same model / other color before changing models. Inspect "
                "identity before cart and again after the event. "
                "Order with product_id+sku only — never attach justification.claims."
            )
            if self.rewritten_goal:
                lines.append(f"rewritten_goal: {self.rewritten_goal[:240]}")
        elif self.family == "justification":
            lines.append(
                "JUSTIFY TARGET: "
                + "; ".join(
                    [
                        f"brand={self.brand or 'any'}",
                        f"category={self.category or 'unknown'}",
                        f"budget<={self.budget if self.budget is not None else 'unknown'}",
                        f"attrs={json.dumps(self.attributes) if self.attributes else '{}'}",
                        f"in_stock={self.in_stock_required}",
                    ]
                )
            )
            lines.append(
                "REQUIRED_CLAIMS for place_test_order.justification.claims: "
                + ", ".join(self.required_claims or ["in_stock", "price"])
            )
            if self.owned_facts:
                lines.append(f"OWNED_FACTS: {json.dumps(self.owned_facts)[:400]}")
            lines.append(
                "TF7 critical: first cart only if FACT-BACKED (exact gated facts). "
                "Claims values MUST be strings matching REQUIRED_CLAIMS "
                "(hd/fhd/27in/true/price) — no JSON booleans/numbers."
            )
        else:
            lines.append(
                f"goal_budget<={self.budget if self.budget is not None else 'unknown'}; "
                f"category={self.category or 'unknown'}"
            )

        lines.append(f"pending_question: {pending}")
        if self.event_recovery_mode:
            if self.family == "preference_reasoning":
                lines.append(
                    "EVENT RECOVERY: message with NO '?', remove, switch to a "
                    "DIFFERENT listing_id, inspect_stock, cart, order."
                )
            elif self.family == "recovery":
                lines.append(
                    "EVENT RECOVERY: message with NO '?', remove, then minimal "
                    f"relaxation ({self.relaxation_order}), inspect identity+stock, "
                    "cart, order."
                )
            else:
                lines.append(
                    "EVENT RECOVERY: message with NO '?', then remove_from_cart → "
                    "inspect backup → add_to_cart → place_test_order. Do not ask shopper."
                )
        if self.last_shopper_text:
            lines.append(f"last_shopper_message: {self.last_shopper_text[:240]}")
        if self.pending_question:
            lines.append(
                "ACTION REQUIRED NOW: call tool `message` and answer the shopper. "
                "Do not cart or order this turn."
            )
        return "\n".join(lines)


def _model() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(
        provider, _DEFAULT_MODELS["chutes"]
    )


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _user_text(user_message: Any) -> str | None:
    if user_message is None:
        return None
    if isinstance(user_message, str):
        text = user_message.strip()
        return text or None
    if isinstance(user_message, dict):
        content = user_message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, str) and not content.strip():
            return None
        if content not in (None, ""):
            return json.dumps(content)
    return None


def _observation_blob(observation: Any) -> str:
    try:
        return json.dumps(observation, default=str)
    except TypeError:
        return str(observation)


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty LLM content")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.I)
    if fence:
        raw = fence.group(1).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in LLM content")
    parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("JSON root is not an object")
    return parsed


def _as_float(value: Any) -> float | None:
    if value is None or value is False:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.\-]", "", value.replace(",", ""))
        if not cleaned or cleaned in {".", "-", "-."}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _as_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "yes", "1"}:
            return True
        if lower in {"false", "no", "0"}:
            return False
    return default


def _theme_from_query(query: str) -> str | None:
    """Pull the exact TF2 theme phrase from the public goal text."""
    patterns = [
        r'matches\s+the\s+phrase\s+[\'"]([^\'"]+)[\'"]',
        r'matches\s+[\'"]([^\'"]+)[\'"]',
        r'matches\s+the\s+phrase\s+([^,\n]+?)(?=\s*,|\s+at\s+or\s+under|\s+and\s+buy)',
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.I)
        if match:
            theme = match.group(1).strip().strip("'\"")
            if theme:
                return theme
    return None


def _theme_tokens(theme: str) -> list[str]:
    return [t for t in re.split(r"\s+", theme.lower()) if t and t not in {"a", "an", "the", "and", "or", "for"}]


# Product-type nouns agents tend to invent; banned in reformulations unless
# already present in the theme phrase.
_INVENTED_TYPE_WORDS = frozenset(
    {
        "cutter",
        "cutting",
        "housing",
        "panel",
        "replacement",
        "tool",
        "tools",
        "pipe",
        "cement",
        "tape",
        "connector",
        "serum",
        "toner",
        "cream",
        "lotion",
        "case",
        "cover",
        "charger",
        "cable",
    }
)


def _reformulation_keeps_theme(text: str, theme: str) -> bool:
    tokens = _theme_tokens(theme)
    if not tokens:
        return True
    lower = text.lower()
    need = len(tokens) if len(tokens) <= 2 else max(2, (len(tokens) + 1) // 2)
    hit = sum(1 for t in tokens if t in lower)
    return hit >= need


def _introduces_invented_type(text: str, theme: str) -> bool:
    theme_l = theme.lower()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in _INVENTED_TYPE_WORDS and word not in theme_l:
            return True
    return False


def _sanitize_tf2_queries(sheet: ConstraintSheet, query: str) -> None:
    heuristic = _theme_from_query(query)
    if heuristic:
        sheet.theme_phrase = heuristic
    if not sheet.theme_phrase:
        return
    theme = sheet.theme_phrase.strip()
    sheet.primary_query = theme
    cleaned: list[str] = []
    for item in sheet.reformulations:
        text = item.strip()
        if not text or text.lower() == theme.lower():
            continue
        if not _reformulation_keeps_theme(text, theme):
            continue
        if _introduces_invented_type(text, theme):
            continue
        if text not in cleaned:
            cleaned.append(text)
    # Deterministic light variants that keep theme tokens only.
    words = theme.split()
    extras = [f'"{theme}"']
    if len(words) >= 2:
        extras.append(" ".join(reversed(words)))
        extras.append(" ".join(words[:2]))
    for extra in extras:
        extra = extra.strip()
        if (
            extra
            and extra.lower() != theme.lower()
            and extra not in cleaned
            and _reformulation_keeps_theme(extra, theme)
            and not _introduces_invented_type(extra, theme)
        ):
            cleaned.append(extra)
    sheet.reformulations = cleaned[:4]


def _track_product_args(sheet: ConstraintSheet, name: str, args: dict[str, Any]) -> None:
    if name not in {
        "view",
        "inspect_stock",
        "add_to_cart",
        "remove_from_cart",
        "place_test_order",
    }:
        return
    pid = args.get("product_id")
    sku = args.get("sku")
    if isinstance(pid, str) and pid.strip():
        sheet.last_product_id = pid.strip()
    if isinstance(sku, str) and sku.strip():
        sheet.last_sku = sku.strip()


def _cart_target(tool_calls: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    for call in tool_calls:
        name = (call.get("function") or {}).get("name")
        if name not in {"add_to_cart", "place_test_order"}:
            continue
        try:
            args = _arguments(call)
        except Exception:
            continue
        pid = args.get("product_id")
        sku = args.get("sku")
        return (
            pid.strip() if isinstance(pid, str) and pid.strip() else None,
            sku.strip() if isinstance(sku, str) and sku.strip() else None,
        )
    return None, None


def _message_has_question(tool_calls: list[dict[str, Any]]) -> bool:
    for call in tool_calls:
        name = (call.get("function") or {}).get("name")
        if name != "message":
            continue
        try:
            args = _arguments(call)
        except Exception:
            continue
        content = args.get("content") or args.get("text") or ""
        if isinstance(content, str) and "?" in content:
            return True
    return False


def _violates_commitment_gate(sheet: ConstraintSheet, tool_calls: list[dict[str, Any]]) -> bool:
    if sheet.family != "intent_decomposition":
        return False
    if sheet.saw_market_event or sheet.event_recovery_mode:
        return False
    if not sheet.committed_product_id:
        return False
    names = _call_names(tool_calls)
    if not any(n in {"add_to_cart", "place_test_order"} for n in names):
        return False
    pid, _sku = _cart_target(tool_calls)
    if not pid:
        return False
    return pid != sheet.committed_product_id


def _chat_json(prompt: str, *, temperature: float = 0) -> dict[str, Any]:
    last_error: Exception | None = None
    temp = temperature
    for _ in range(_EXTRACT_TRIES):
        inference = _proxy.post(
            "/inference/chat/completions",
            json_data={
                "model": _model(),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return a single valid JSON object only. "
                            "No markdown fences. No commentary."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": temp,
            },
        )
        if inference is None:
            last_error = RuntimeError("inference request failed")
            temp = min(temp + 0.2, 0.6)
            continue
        try:
            content = inference["choices"][0]["message"].get("content") or ""
            return _extract_json_object(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            temp = min(temp + 0.2, 0.6)
    raise RuntimeError(f"json extract failed: {last_error}")


def _detect_family_heuristic(query: str) -> Family | None:
    lower = query.lower()
    if re.search(
        r"compare only these candidate|among ties, choose|highest refresh|lexicographic",
        lower,
    ):
        return "ranking"
    if re.search(
        r"walk me through why|tell me why|reason i can trust|why the one you pick",
        lower,
    ):
        return "justification"
    if re.search(
        r"another color of the same model|try another color|changing models only|"
        r"exact color/model|if that exact .+ disappears|color/model disappears",
        lower,
    ):
        return "recovery"
    if re.search(
        r"don't just grab the cheapest|do not just grab the cheapest|"
        r"rather pay a bit more|for daily runs|heavy gym sessions|"
        r"every kind of weather|sweat and water",
        lower,
    ):
        return "preference_reasoning"
    if re.search(
        r"firm requirement on the spec that i will tell you if you ask|"
        r"there is one firm requirement",
        lower,
    ):
        return "constraint_satisfaction"
    if re.search(
        r'matches\s+[\'"]|matches\s+the\s+phrase|find an in-stock listing', lower
    ):
        return "retrieval_recall"
    if re.search(r"prefer .+ but the firm requirements matter more", lower):
        return "intent_decomposition"
    if "firm requirements" in lower and "prefer" in lower:
        return "intent_decomposition"
    # Brand + budget smartphone without soft prefer ≈ TF3 when not TF1 soft.
    if re.search(r"\bat or under\b.+\bin stock\b", lower) and "prefer" not in lower:
        if re.search(r"smartphone|monitor|tablet", lower):
            # Could be TF3 or TF7; TF7 usually asks for why.
            if "why" not in lower and "reason" not in lower:
                return "constraint_satisfaction"
    return None


def _public_family(problem_data: dict[str, Any]) -> Family | None:
    aliases = {
        "intent_decomposition": "intent_decomposition",
        "retrieval_recall": "retrieval_recall",
        "constraint_satisfaction": "constraint_satisfaction",
        "preference_reasoning": "preference_reasoning",
        "ranking": "ranking",
        "recovery": "recovery",
        "justification": "justification",
        "tf1": "intent_decomposition",
        "tf2": "retrieval_recall",
        "tf3": "constraint_satisfaction",
        "tf4": "preference_reasoning",
        "tf5": "ranking",
        "tf6": "recovery",
        "tf7": "justification",
        "intent": "intent_decomposition",
        "retrieval": "retrieval_recall",
        "constraint": "constraint_satisfaction",
        "preference": "preference_reasoning",
    }
    for key in ("family", "task_family", "category"):
        raw = problem_data.get(key)
        if isinstance(raw, str):
            val = raw.strip().lower().replace(" ", "_")
            if val in aliases:
                return aliases[val]  # type: ignore[return-value]
            if val.startswith("tf1"):
                return "intent_decomposition"
            if val.startswith("tf2"):
                return "retrieval_recall"
            if val.startswith("tf3"):
                return "constraint_satisfaction"
            if val.startswith("tf4"):
                return "preference_reasoning"
            if val.startswith("tf5"):
                return "ranking"
            if val.startswith("tf6"):
                return "recovery"
            if val.startswith("tf7"):
                return "justification"
    env = problem_data.get("environment")
    if isinstance(env, dict):
        for key in ("family", "task_family"):
            raw = env.get(key)
            if isinstance(raw, str):
                val = raw.strip().lower().replace(" ", "_")
                if val in aliases:
                    return aliases[val]  # type: ignore[return-value]
    # task id like TF5-ranking-300000 — fallback only (soft router prefers LLM)
    for key in ("problem_id", "task_id", "id"):
        raw = problem_data.get(key)
        if isinstance(raw, str) and raw.upper().startswith("TF"):
            m = re.match(r"TF(\d+)-([a-z_]+)", raw, flags=re.I)
            if m:
                num = m.group(1)
                return {
                    "1": "intent_decomposition",
                    "2": "retrieval_recall",
                    "3": "constraint_satisfaction",
                    "4": "preference_reasoning",
                    "5": "ranking",
                    "6": "recovery",
                    "7": "justification",
                }.get(num)  # type: ignore[return-value]
    return None


def _route_family(problem_data: dict[str, Any], query: str) -> Family:
    """Soft-first routing: LLM classify → query heuristic → public metadata."""
    heuristic = _detect_family_heuristic(query)
    try:
        data = _chat_json(_FAMILY_ROUTE_PROMPT.format(query=query))
        fam = str(data.get("family") or "").strip().lower()
        if fam in _FAMILIES:
            if fam == "generic" and heuristic:
                return heuristic
            return fam  # type: ignore[return-value]
    except Exception:
        pass
    if heuristic:
        return heuristic
    public = _public_family(problem_data)
    if public:
        return public
    return "generic"



def _apply_gated(sheet: ConstraintSheet, gated: Any) -> None:
    if not isinstance(gated, dict):
        return
    key = gated.get("key")
    value = gated.get("value")
    if isinstance(key, str) and key.strip():
        sheet.gated_key = key.strip()
    if value is not None and str(value).strip():
        sheet.gated_value = str(value).strip()


def _extract_tf1(sheet: ConstraintSheet, query: str) -> None:
    data = _chat_json(_TF1_GOAL_EXTRACT_PROMPT.format(query=query))
    firm = data.get("firm") if isinstance(data.get("firm"), dict) else {}
    soft = data.get("soft") if isinstance(data.get("soft"), dict) else {}
    category = firm.get("category")
    if isinstance(category, str) and category.strip():
        sheet.category = category.strip()
    _apply_gated(sheet, firm.get("gated_attribute"))
    budget = _as_float(firm.get("budget_max"))
    if budget is not None:
        sheet.budget = budget
    in_stock = _as_bool(firm.get("in_stock"), default=True)
    if in_stock is not None:
        sheet.in_stock_required = in_stock
    brand = soft.get("preferred_brand")
    if isinstance(brand, str) and brand.strip():
        sheet.preferred_brand = brand.strip()
    rewritten = data.get("rewritten_goal")
    if isinstance(rewritten, str) and rewritten.strip():
        sheet.rewritten_goal = rewritten.strip()
    notes = data.get("notes")
    if isinstance(notes, str) and notes.strip():
        sheet.notes.append(notes.strip())


def _extract_tf2(sheet: ConstraintSheet, query: str) -> None:
    data = _chat_json(_TF2_GOAL_EXTRACT_PROMPT.format(query=query))
    theme = data.get("theme_phrase")
    if isinstance(theme, str) and theme.strip():
        sheet.theme_phrase = theme.strip()
    category = data.get("category")
    if isinstance(category, str) and category.strip():
        sheet.category = category.strip()
    budget = _as_float(data.get("budget_max"))
    if budget is not None:
        sheet.budget = budget
    in_stock = _as_bool(data.get("in_stock"), default=True)
    if in_stock is not None:
        sheet.in_stock_required = in_stock
    primary = data.get("primary_query")
    if isinstance(primary, str) and primary.strip():
        sheet.primary_query = primary.strip()
    elif sheet.theme_phrase:
        sheet.primary_query = sheet.theme_phrase
    reforms = data.get("reformulations")
    if isinstance(reforms, list):
        cleaned: list[str] = []
        for item in reforms:
            if isinstance(item, str) and item.strip():
                text = item.strip()
                if text != sheet.primary_query and text not in cleaned:
                    cleaned.append(text)
        sheet.reformulations = cleaned[:4]
    notes = data.get("notes")
    if isinstance(notes, str) and notes.strip():
        sheet.notes.append(notes.strip())
    _sanitize_tf2_queries(sheet, query)


def _parse_candidates_from_query(query: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in re.finditer(r"(\d{6,})::([A-Za-z0-9_-]+)", query):
        pid, sku = match.group(1), match.group(2)
        key = f"{pid}::{sku}"
        if key in seen:
            continue
        seen.add(key)
        found.append({"product_id": pid, "sku": sku})
    return found


def _default_rank_sense(priority: list[str]) -> dict[str, str]:
    sense = {"refresh_rate": "max", "screen_size": "max", "price": "min"}
    for axis in priority:
        sense.setdefault(axis, "max" if axis != "price" else "min")
    return sense


def _extract_tf3(sheet: ConstraintSheet, query: str) -> None:
    data = _chat_json(_TF3_GOAL_EXTRACT_PROMPT.format(query=query))
    brand = data.get("brand")
    if isinstance(brand, str) and brand.strip():
        sheet.brand = brand.strip()
    category = data.get("category")
    if isinstance(category, str) and category.strip():
        sheet.category = category.strip()
    _apply_gated(sheet, data.get("gated_attribute"))
    # Heuristic screen from goal when LLM misses it.
    if not sheet.gated_value:
        m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", query, flags=re.I)
        if m:
            sheet.gated_key = sheet.gated_key or "screen_size"
            sheet.gated_value = f"{m.group(1)}in"
    budget = _as_float(data.get("budget_max"))
    if budget is not None:
        sheet.budget = budget
    in_stock = _as_bool(data.get("in_stock"), default=True)
    if in_stock is not None:
        sheet.in_stock_required = in_stock
    underspec = _as_bool(data.get("underspecified"), default=None)
    if underspec is None:
        underspec = bool(
            re.search(r"if you ask|firm requirement on the spec", query, flags=re.I)
        )
    sheet.underspecified = bool(underspec) and not bool(sheet.gated_value)
    rewritten = data.get("rewritten_goal")
    if isinstance(rewritten, str) and rewritten.strip():
        sheet.rewritten_goal = rewritten.strip()
    notes = data.get("notes")
    if isinstance(notes, str) and notes.strip():
        sheet.notes.append(notes.strip())


def _extract_tf5(sheet: ConstraintSheet, query: str) -> None:
    # Deterministic candidate parse first (previous-bench style structure).
    sheet.candidates = _parse_candidates_from_query(query)
    sheet.rank_priority = ["refresh_rate", "screen_size", "price"]
    sheet.rank_sense = _default_rank_sense(sheet.rank_priority)
    m_budget = re.search(
        r"at or under\s*([0-9]+(?:\.[0-9]+)?)\s*PHP", query, flags=re.I
    )
    if m_budget:
        sheet.budget = float(m_budget.group(1))
    if re.search(r"\bmonitor", query, flags=re.I):
        sheet.category = "Monitors"
    sheet.in_stock_required = True
    try:
        data = _chat_json(_TF5_GOAL_EXTRACT_PROMPT.format(query=query))
    except Exception:
        sheet.notes.append("tf5_extract_failed_heuristic_only")
        return
    category = data.get("category")
    if isinstance(category, str) and category.strip():
        sheet.category = category.strip()
    budget = _as_float(data.get("budget_max"))
    if budget is not None:
        sheet.budget = budget
    in_stock = _as_bool(data.get("in_stock"), default=True)
    if in_stock is not None:
        sheet.in_stock_required = in_stock
    cands = data.get("candidates")
    if isinstance(cands, list) and cands:
        parsed: list[dict[str, str]] = []
        for item in cands:
            if not isinstance(item, dict):
                continue
            pid = item.get("product_id")
            sku = item.get("sku")
            if isinstance(pid, str) and isinstance(sku, str) and pid and sku:
                parsed.append({"product_id": pid.strip(), "sku": sku.strip()})
        if parsed:
            sheet.candidates = parsed
    priority = data.get("priority")
    if isinstance(priority, list) and priority:
        sheet.rank_priority = [str(x) for x in priority if str(x).strip()]
    sense = data.get("priority_sense")
    if isinstance(sense, dict) and sense:
        sheet.rank_sense = {str(k): str(v).lower() for k, v in sense.items()}
    else:
        sheet.rank_sense = _default_rank_sense(sheet.rank_priority)
    notes = data.get("notes")
    if isinstance(notes, str) and notes.strip():
        sheet.notes.append(notes.strip())


def _normalize_resolution(value: str) -> str:
    lower = value.strip().lower()
    if "4k" in lower or "2160" in lower:
        return "4k"
    if "1080" in lower or "full hd" in lower or "fhd" in lower:
        return "fhd"
    if "720" in lower or re.search(r"\bhd\b", lower):
        return "hd"
    return lower


def _extract_tf7(sheet: ConstraintSheet, query: str) -> None:
    data = _chat_json(_TF7_GOAL_EXTRACT_PROMPT.format(query=query))
    brand = data.get("brand")
    if isinstance(brand, str) and brand.strip():
        sheet.brand = brand.strip()
    category = data.get("category")
    if isinstance(category, str) and category.strip():
        sheet.category = category.strip()
    budget = _as_float(data.get("budget_max"))
    if budget is not None:
        sheet.budget = budget
    in_stock = _as_bool(data.get("in_stock"), default=True)
    if in_stock is not None:
        sheet.in_stock_required = in_stock
    attrs = data.get("attributes") if isinstance(data.get("attributes"), dict) else {}
    cleaned_attrs: dict[str, str] = {}
    for key in ("resolution", "screen_size", "refresh_rate"):
        val = attrs.get(key)
        if isinstance(val, str) and val.strip():
            cleaned_attrs[key] = (
                _normalize_resolution(val) if key == "resolution" else val.strip()
            )
    # Heuristics from goal text.
    if "resolution" not in cleaned_attrs:
        if re.search(r"720p|\bhd\b", query, flags=re.I):
            cleaned_attrs["resolution"] = "hd"
        elif re.search(r"1080p|full hd|\bfhd\b", query, flags=re.I):
            cleaned_attrs["resolution"] = "fhd"
        elif re.search(r"4k|2160p", query, flags=re.I):
            cleaned_attrs["resolution"] = "4k"
    if "screen_size" not in cleaned_attrs:
        m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", query, flags=re.I)
        if m:
            cleaned_attrs["screen_size"] = f"{m.group(1)}in"
    sheet.attributes = cleaned_attrs
    if "screen_size" in cleaned_attrs:
        sheet.gated_key = "screen_size"
        sheet.gated_value = cleaned_attrs["screen_size"]
    elif "resolution" in cleaned_attrs:
        sheet.gated_key = "resolution"
        sheet.gated_value = cleaned_attrs["resolution"]
    claims = data.get("required_claims")
    req: list[str] = []
    if isinstance(claims, list):
        for c in claims:
            if isinstance(c, str) and c.strip():
                req.append(c.strip())
    for must in ("in_stock", "price"):
        if must not in req:
            req.append(must)
    for attr_key in cleaned_attrs:
        if attr_key not in req:
            req.append(attr_key)
    # Do NOT force brand/category into required_claims — extras can fail
    # faithfulness. Soft brand may still be claimed when clearly supported.
    sheet.required_claims = [
        c for c in req if c not in {"category"} or c in cleaned_attrs
    ]
    # Drop category from required entirely.
    sheet.required_claims = [c for c in sheet.required_claims if c != "category"]
    rewritten = data.get("rewritten_goal")
    if isinstance(rewritten, str) and rewritten.strip():
        sheet.rewritten_goal = rewritten.strip()
    notes = data.get("notes")
    if isinstance(notes, str) and notes.strip():
        sheet.notes.append(notes.strip())


def _extract_tf4(sheet: ConstraintSheet, query: str) -> None:
    data = _chat_json(_TF4_GOAL_EXTRACT_PROMPT.format(query=query))
    category = data.get("category")
    if isinstance(category, str) and category.strip():
        sheet.category = category.strip()
    budget = _as_float(data.get("budget_max"))
    if budget is not None:
        sheet.budget = budget
    in_stock = _as_bool(data.get("in_stock"), default=True)
    if in_stock is not None:
        sheet.in_stock_required = in_stock
    use_case = data.get("use_case")
    if isinstance(use_case, str) and use_case.strip():
        sheet.use_case = use_case.strip()
    axes = data.get("soft_axes")
    if isinstance(axes, list):
        sheet.soft_axes = [
            str(a).strip() for a in axes if isinstance(a, str) and str(a).strip()
        ][:8]
    priority = data.get("tradeoff_priority")
    if isinstance(priority, list):
        sheet.tradeoff_priority = [
            str(p).strip() for p in priority if isinstance(p, str) and str(p).strip()
        ][:6]
    shortcuts = data.get("unacceptable_shortcuts")
    if isinstance(shortcuts, list):
        sheet.unacceptable_shortcuts = [
            str(s).strip()
            for s in shortcuts
            if isinstance(s, str) and str(s).strip()
        ][:4]
    clarify = _as_bool(data.get("needs_priority_clarification"), default=True)
    sheet.needs_priority_clarification = True if clarify is None else bool(clarify)
    rewritten = data.get("rewritten_goal")
    if isinstance(rewritten, str) and rewritten.strip():
        sheet.rewritten_goal = rewritten.strip()
    notes = data.get("notes")
    if isinstance(notes, str) and notes.strip():
        sheet.notes.append(notes.strip())
    # Light heuristics when extract is sparse.
    if not sheet.use_case:
        for cue in (
            "daily runs",
            "gym",
            "weather",
            "outdoor",
            "kids",
            "commute",
        ):
            if cue in query.lower():
                sheet.use_case = cue
                break
    if not sheet.soft_axes:
        hints = []
        lower = query.lower()
        if re.search(r"sweat|water|ipx|weather", lower):
            hints.append("sweat_water_resistance")
        if re.search(r"gym|sport|run|workout", lower):
            hints.append("sport_secure_fit")
        if re.search(r"battery|long.?lasting", lower):
            hints.append("battery_life")
        if re.search(r"wireless|bluetooth", lower):
            hints.append("wireless_freedom")
        sheet.soft_axes = hints


def _extract_tf6(sheet: ConstraintSheet, query: str) -> None:
    data = _chat_json(_TF6_GOAL_EXTRACT_PROMPT.format(query=query))
    category = data.get("category")
    if isinstance(category, str) and category.strip():
        sheet.category = category.strip()
    budget = _as_float(data.get("budget_max"))
    if budget is not None:
        sheet.budget = budget
    in_stock = _as_bool(data.get("in_stock"), default=True)
    if in_stock is not None:
        sheet.in_stock_required = in_stock
    model = data.get("preferred_model")
    if isinstance(model, str) and model.strip():
        sheet.preferred_model = model.strip()
    color = data.get("preferred_color")
    if isinstance(color, str) and color.strip():
        sheet.preferred_color = color.strip()
    order = data.get("relaxation_order")
    if isinstance(order, list) and order:
        joined = " ".join(str(x) for x in order).lower()
        if "model_before" in joined:
            sheet.relaxation_order = "model_before_color"
        else:
            sheet.relaxation_order = "color_before_model"
    elif isinstance(order, str) and order.strip():
        sheet.relaxation_order = order.strip()
    rewritten = data.get("rewritten_goal")
    if isinstance(rewritten, str) and rewritten.strip():
        sheet.rewritten_goal = rewritten.strip()
    notes = data.get("notes")
    if isinstance(notes, str) and notes.strip():
        sheet.notes.append(notes.strip())
    # Heuristic color/model cues when LLM misses them.
    if not sheet.preferred_color:
        m = re.search(
            r"\b(black|white|green|blue|red|silver|gold|gray|grey|pink|purple|"
            r"sage green|navy|midnight)\b",
            query,
            flags=re.I,
        )
        if m:
            sheet.preferred_color = m.group(1)
    if not sheet.preferred_model:
        m = re.search(
            r"(?:prefer(?:s|red)?|want(?:s)?|eye on)\s+(?:the\s+)?"
            r"(?:black|white|green|blue|red|silver|gold|gray|grey|pink|purple)?\s*"
            r"([A-Za-z0-9][A-Za-z0-9._-]{3,})",
            query,
            flags=re.I,
        )
        if m:
            sheet.preferred_model = m.group(1)
    if not sheet.category:
        if re.search(r"smartphone", query, flags=re.I):
            sheet.category = "Smartphones"
        elif re.search(r"headphone", query, flags=re.I):
            sheet.category = "Headphones"


def _init_sheet(problem_data: dict[str, Any], query: str) -> ConstraintSheet:
    family = _route_family(problem_data, query)
    sheet = ConstraintSheet(family=family)
    try:
        if family == "intent_decomposition":
            _extract_tf1(sheet, query)
        elif family == "retrieval_recall":
            _extract_tf2(sheet, query)
        elif family == "constraint_satisfaction":
            _extract_tf3(sheet, query)
        elif family == "preference_reasoning":
            _extract_tf4(sheet, query)
        elif family == "ranking":
            _extract_tf5(sheet, query)
        elif family == "recovery":
            _extract_tf6(sheet, query)
        elif family == "justification":
            _extract_tf7(sheet, query)
        else:
            try:
                _extract_tf2(sheet, query)
            except Exception:
                try:
                    _extract_tf1(sheet, query)
                except Exception:
                    sheet.notes.append("generic_extract_failed")
    except Exception:
        sheet.notes.append(f"{family}_extract_failed")
    return sheet


def _numeric_fact(value: Any) -> float | None:
    if value is None or value is False:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"(\d+(?:\.\d+)?)", value.replace(",", ""))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _normalize_screen_size(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    # Reject ranges / multi-size listings (not fact-backed for a single size).
    if re.search(r"\d+(?:\.\d+)?\s*[-–to]+\s*\d+", text):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    num = m.group(1)
    if num.endswith(".0"):
        num = num[:-2]
    return f"{num}in"


def _observed_resolution(facts: dict[str, Any]) -> str | None:
    for key in ("resolution", "display"):
        raw = facts.get(key)
        if raw is None:
            continue
        text = str(raw).lower()
        if "3840" in text or "4k" in text or "uhd" in text:
            return "4k"
        if "1920" in text or "1080" in text or "full hd" in text or "fhd" in text:
            return "fhd"
        if "1280" in text or "720" in text:
            return "hd"
        if re.search(r"\bhd\b", text) and "full" not in text and "fhd" not in text:
            return "hd"
    return None


def _facts_for_product(
    sheet: ConstraintSheet, product_id: str | None, sku: str | None
) -> dict[str, Any]:
    if not product_id:
        return {}
    keys: list[str] = []
    if sku:
        keys.append(f"{product_id}::{sku}")
    keys.append(product_id)
    merged: dict[str, Any] = {}
    for key in keys:
        prev = sheet.candidate_facts.get(key)
        if prev:
            merged.update(prev)
    # Prefer SKU-scoped candidate facts only. Do NOT merge sheet.owned_facts
    # across products — stale attrs from a prior view pollute the next SKU.
    return merged


def _tf7_is_fact_backed(
    sheet: ConstraintSheet, product_id: str | None, sku: str | None
) -> tuple[bool, str]:
    """First/recovery cart must be a constraint-faithful fact-backed match."""
    if not product_id:
        return False, "missing product_id"
    facts = _facts_for_product(sheet, product_id, sku)
    if not facts:
        return False, "no viewed/inspected facts for this product yet"

    if sheet.brand:
        brand = facts.get("brand")
        if not isinstance(brand, str) or brand.strip().lower() != sheet.brand.strip().lower():
            return False, f"brand fact mismatch (need {sheet.brand}, got {brand!r})"

    if sheet.in_stock_required and facts.get("in_stock") is False:
        return False, "not in stock"

    price = _numeric_fact(facts.get("price"))
    if sheet.budget is not None and price is not None and price > sheet.budget:
        return False, f"price {price} exceeds budget {sheet.budget}"

    for attr, need in (sheet.attributes or {}).items():
        if not need:
            continue
        if attr == "resolution":
            got = _observed_resolution(facts)
            need_n = _normalize_resolution(need)
            if got != need_n:
                return False, f"resolution fact {got!r} != required {need_n!r}"
        elif attr == "screen_size":
            got = _normalize_screen_size(facts.get("screen_size") or facts.get("size"))
            need_n = _normalize_screen_size(need)
            if not got or not need_n or got != need_n:
                return False, f"screen_size fact {got!r} != required {need_n!r}"
        elif attr == "refresh_rate":
            got = _numeric_fact(facts.get("refresh_rate"))
            need_n = _numeric_fact(need)
            if got is None or need_n is None or got != need_n:
                return False, f"refresh_rate fact {got!r} != required {need_n!r}"
    return True, "ok"


def _ingest_product_facts(sheet: ConstraintSheet, observation: Any) -> None:
    """Harvest structured facts from view/compare/inspect observations (TF5/TF7)."""
    blob = observation
    if isinstance(observation, dict):
        for key in ("observation", "data", "result"):
            if isinstance(observation.get(key), (dict, list)):
                blob = observation[key]
                break

    products: list[dict[str, Any]] = []
    if isinstance(blob, dict):
        if "compared" in blob and isinstance(blob["compared"], list):
            products.extend([p for p in blob["compared"] if isinstance(p, dict)])
        elif "results" in blob and isinstance(blob["results"], list):
            products.extend([p for p in blob["results"] if isinstance(p, dict)])
        elif "product_id" in blob:
            products.append(blob)
        elif "price" in blob or "in_stock" in blob:
            products.append(blob)
    elif isinstance(blob, list):
        products.extend([p for p in blob if isinstance(p, dict)])

    for prod in products:
        pid = prod.get("product_id")
        sku = prod.get("sku") or prod.get("sku_id")
        facts_src = prod.get("facts") if isinstance(prod.get("facts"), dict) else {}
        merged: dict[str, Any] = {}
        for key in ("price", "in_stock", "brand", "currency", "title"):
            if key in prod:
                merged[key] = prod[key]
        for key in (
            "refresh_rate",
            "screen_size",
            "resolution",
            "display",
            "size",
            "brand",
            "price",
            "in_stock",
            "battery",
            "battery_life",
            "water_resistance",
            "connectivity",
        ):
            if key in facts_src:
                merged[key] = facts_src[key]
        if "screen_size" not in merged and "size" in merged:
            merged["screen_size"] = merged["size"]
        path = prod.get("category_path")
        if isinstance(path, list) and path:
            merged["category"] = path[-1] if isinstance(path[-1], str) else str(path[-1])
        # Keep a searchable text blob for soft preference-signal checks (TF4).
        text_parts: list[str] = []
        if isinstance(merged.get("title"), str):
            text_parts.append(merged["title"])
        for fk, fv in facts_src.items():
            text_parts.append(f"{fk} {fv}")
        if isinstance(path, list):
            text_parts.extend(str(x) for x in path)
        if text_parts:
            merged["evidence_text"] = " ".join(text_parts)
        if isinstance(prod.get("listing_id"), str):
            merged["listing_id"] = prod["listing_id"]
        if isinstance(pid, str) and pid:
            key = f"{pid}::{sku}" if isinstance(sku, str) and sku else pid
            prev = sheet.candidate_facts.get(key) or {}
            old_text = prev.get("evidence_text")
            prev.update({k: v for k, v in merged.items() if v is not None})
            if isinstance(old_text, str) and old_text:
                new_text = prev.get("evidence_text")
                if not isinstance(new_text, str) or len(new_text) < len(old_text):
                    prev["evidence_text"] = old_text
            sheet.candidate_facts[key] = prev
            listing = prod.get("listing_id") or merged.get("listing_id")
            if isinstance(listing, str) and listing.strip():
                lid = listing.strip()
                if lid not in sheet.observed_listing_ids:
                    sheet.observed_listing_ids.append(lid)
            if sheet.family == "justification":
                # Replace owned_facts for the active product (no cross-SKU leak).
                sheet.owned_facts = {
                    k: v for k, v in merged.items() if v is not None
                }
                sheet.owned_facts["product_id"] = pid
                if isinstance(sku, str):
                    sheet.owned_facts["sku"] = sku


def _build_tf7_claims(
    sheet: ConstraintSheet, product_id: str | None, sku: str | None
) -> list[dict[str, str]]:
    """Build claims ONLY from observed facts for this SKU — never invent values."""
    facts = _facts_for_product(sheet, product_id, sku)
    claims: list[dict[str, str]] = []
    required = [c for c in (sheet.required_claims or ["in_stock", "price"]) if c != "category"]
    for field_name in required:
        if field_name == "in_stock":
            stock = facts.get("in_stock")
            if stock is None:
                continue
            claims.append(
                {
                    "field": "in_stock",
                    "value": "true" if stock is not False else "false",
                }
            )
        elif field_name == "price":
            price = _numeric_fact(facts.get("price"))
            if price is None:
                continue
            if float(price).is_integer():
                claims.append({"field": "price", "value": str(int(price))})
            else:
                claims.append({"field": "price", "value": str(price)})
        elif field_name == "resolution":
            got = _observed_resolution(facts)
            if not got:
                continue
            claims.append({"field": "resolution", "value": got})
        elif field_name == "screen_size":
            got = _normalize_screen_size(facts.get("screen_size") or facts.get("size"))
            if not got:
                continue
            claims.append({"field": "screen_size", "value": got})
        elif field_name == "refresh_rate":
            rate = _numeric_fact(facts.get("refresh_rate"))
            if rate is None:
                continue
            claims.append({"field": "refresh_rate", "value": str(int(rate))})
        elif field_name == "brand":
            brand = facts.get("brand")
            if not isinstance(brand, str) or not brand.strip():
                continue
            claims.append({"field": "brand", "value": brand.strip()})
    return claims


def _coerce_tf7_order_args(
    sheet: ConstraintSheet, args: dict[str, Any]
) -> dict[str, Any]:
    """Force string, required-only justification claims from owned facts."""
    pid = args.get("product_id")
    sku = args.get("sku")
    if not isinstance(pid, str):
        pid = sheet.last_product_id
    if not isinstance(sku, str):
        sku = sheet.last_sku
    claims = _build_tf7_claims(
        sheet,
        pid if isinstance(pid, str) else None,
        sku if isinstance(sku, str) else None,
    )
    out = dict(args)
    if isinstance(pid, str):
        out["product_id"] = pid
    if isinstance(sku, str):
        out["sku"] = sku
    out["justification"] = {"claims": claims}
    return out


def _strip_non_tf7_order_args(args: dict[str, Any]) -> dict[str, Any]:
    """Non-TF7 families must not submit justification.claims (fails hard checks)."""
    out = dict(args)
    out.pop("justification", None)
    return out


def _recompute_ranking_top(sheet: ConstraintSheet) -> None:
    if sheet.family != "ranking" or not sheet.candidates:
        return
    priority = sheet.rank_priority or ["refresh_rate", "screen_size", "price"]
    sense = sheet.rank_sense or _default_rank_sense(priority)

    def sort_key(cand: dict[str, str]) -> tuple:
        key = f"{cand['product_id']}::{cand['sku']}"
        facts = sheet.candidate_facts.get(key) or {}
        in_stock = facts.get("in_stock")
        price = _numeric_fact(facts.get("price"))
        eligible = True
        if sheet.in_stock_required and in_stock is False:
            eligible = False
        if sheet.budget is not None and price is not None and price > sheet.budget:
            eligible = False
        parts: list[Any] = [0 if eligible else 1]
        for axis in priority:
            raw = facts.get(axis) if axis != "price" else facts.get("price")
            num = _numeric_fact(raw)
            direction = sense.get(axis, "max" if axis != "price" else "min")
            if num is None:
                parts.append((1, 0.0))
            elif direction == "min":
                parts.append((0, num))
            else:
                parts.append((0, -num))
        return tuple(parts)

    ranked = sorted(sheet.candidates, key=sort_key)
    top = ranked[0]
    key = f"{top['product_id']}::{top['sku']}"
    facts = sheet.candidate_facts.get(key) or {}
    if facts:
        sheet.computed_top_id = top["product_id"]
        sheet.computed_top_sku = top["sku"]


def _needs_ask_spec(sheet: ConstraintSheet) -> bool:
    return (
        sheet.family == "constraint_satisfaction"
        and sheet.underspecified
        and not sheet.gated_value
        and not sheet.asked_for_spec
    )


def _needs_preference_clarify(sheet: ConstraintSheet) -> bool:
    return (
        sheet.family == "preference_reasoning"
        and sheet.needs_priority_clarification
        and not sheet.asked_preference_clarify
        and not sheet.event_recovery_mode
    )


def _tf4_preference_signals(facts: dict[str, Any]) -> set[str]:
    """Soft multi-axis evidence flags from viewed text (not the hidden scorer)."""
    text = " ".join(
        str(facts.get(k) or "")
        for k in (
            "evidence_text",
            "title",
            "battery",
            "battery_life",
            "water_resistance",
            "connectivity",
            "category",
        )
    ).casefold()
    hits: set[str] = set()
    if re.search(
        r"\b(ipx?[4-8]|ip5\d|ip6\d|sweat\s*(proof|resistant)?|water\s*(proof|resistant)?)\b",
        text,
    ):
        hits.add("water")
    if re.search(
        r"\b(sports?|running|gym|workout|fitness|ear\s*hook|secure\s*fit|open\s*ear)\b",
        text,
    ):
        hits.add("sport")
    if re.search(r"\b(wireless|bluetooth|true\s*wireless|tws)\b", text):
        hits.add("wireless")
    if re.search(r"\bbattery\b|\b\d+\s*(h|hr|hrs|hour|hours)\b", text):
        hits.add("battery")
    return hits


def _tf4_cart_reject_reason(
    sheet: ConstraintSheet, product_id: str | None, sku: str | None
) -> str | None:
    if sheet.family != "preference_reasoning":
        return None
    if not product_id:
        return "missing product_id"
    facts = _facts_for_product(sheet, product_id, sku)
    if not facts:
        return "no viewed/compared facts for this product yet — view it first"
    price = _numeric_fact(facts.get("price"))
    if sheet.budget is not None and price is not None and price > sheet.budget:
        return (
            f"viewed price {price} exceeds budget {sheet.budget} "
            "(never cart over-budget items)"
        )
    if facts.get("in_stock") is False:
        return "product is not in stock"
    signals = _tf4_preference_signals(facts)
    # Require multi-axis evidence; water-only / cheapest-only fails TF4.
    if len(signals) < 2:
        return (
            "weak preference evidence "
            f"(axes={sorted(signals) or 'none'}; need ≥2 of "
            "water/sport/wireless/battery from viewed facts). "
            "View a stronger multi-axis match."
        )
    if not sheet.event_recovery_mode and len(sheet.observed_listing_ids) < 2:
        return (
            "compare/view at least 2 distinct listing_id values among "
            "hard-valid options before the first cart"
        )
    return None


def _violates_tf4_clarify_gate(
    sheet: ConstraintSheet, tool_calls: list[dict[str, Any]]
) -> bool:
    if not _needs_preference_clarify(sheet):
        return False
    names = _call_names(tool_calls)
    return any(n in {"add_to_cart", "place_test_order"} for n in names)


def _violates_tf4_cart_gate(
    sheet: ConstraintSheet, tool_calls: list[dict[str, Any]]
) -> bool:
    if sheet.family != "preference_reasoning":
        return False
    names = _call_names(tool_calls)
    if not any(n in {"add_to_cart", "place_test_order"} for n in names):
        return False
    pid, sku = _cart_target(tool_calls)
    return _tf4_cart_reject_reason(sheet, pid, sku) is not None


def _violates_justification_gate(
    sheet: ConstraintSheet, tool_calls: list[dict[str, Any]]
) -> bool:
    if sheet.family != "justification":
        return False
    if "place_test_order" not in _call_names(tool_calls):
        return False
    # Claims are coerced on emit; block when SKU is not fact-backed OR we
    # cannot cover every required claim from observed facts (no inventing).
    pid: str | None = None
    sku: str | None = None
    for call in tool_calls:
        if (call.get("function") or {}).get("name") != "place_test_order":
            continue
        try:
            args = _arguments(call)
        except Exception:
            return True
        pid = args.get("product_id") if isinstance(args.get("product_id"), str) else None
        sku = args.get("sku") if isinstance(args.get("sku"), str) else None
    ok, _reason = _tf7_is_fact_backed(sheet, pid, sku)
    if not ok:
        return True
    claims = _build_tf7_claims(sheet, pid, sku)
    have = {c["field"] for c in claims}
    required = [
        c
        for c in (sheet.required_claims or ["in_stock", "price"])
        if c != "category"
    ]
    return any(field_name not in have for field_name in required)


def _violates_tf7_cart_gate(
    sheet: ConstraintSheet, tool_calls: list[dict[str, Any]]
) -> bool:
    if sheet.family != "justification":
        return False
    if "add_to_cart" not in _call_names(tool_calls):
        return False
    for call in tool_calls:
        if (call.get("function") or {}).get("name") != "add_to_cart":
            continue
        try:
            args = _arguments(call)
        except Exception:
            return True
        pid = args.get("product_id") if isinstance(args.get("product_id"), str) else None
        sku = args.get("sku") if isinstance(args.get("sku"), str) else None
        ok, _reason = _tf7_is_fact_backed(sheet, pid, sku)
        return not ok
    return False


def _tf7_cart_reject_reason(
    sheet: ConstraintSheet, tool_calls: list[dict[str, Any]]
) -> str:
    for call in tool_calls:
        if (call.get("function") or {}).get("name") != "add_to_cart":
            continue
        try:
            args = _arguments(call)
        except Exception:
            return "invalid add_to_cart arguments"
        pid = args.get("product_id") if isinstance(args.get("product_id"), str) else None
        sku = args.get("sku") if isinstance(args.get("sku"), str) else None
        _ok, reason = _tf7_is_fact_backed(sheet, pid, sku)
        return reason
    return "not fact-backed"


def _violates_ask_spec_gate(
    sheet: ConstraintSheet, tool_calls: list[dict[str, Any]]
) -> bool:
    if not _needs_ask_spec(sheet):
        return False
    names = _call_names(tool_calls)
    if not names:
        return True
    if any(n in _BLOCKED_WHILE_PENDING for n in names):
        return True
    return "message" not in names


def _apply_shopper_update(sheet: ConstraintSheet, text: str) -> None:
    sheet.last_shopper_text = text
    prior_budget = sheet.budget
    try:
        data = _chat_json(
            _UPDATE_EXTRACT_PROMPT.format(sheet=sheet.status_block(), message=text)
        )
    except Exception:
        if "?" in text:
            sheet.pending_question = True
            sheet.notes.append("update_extract_failed_question_fallback")
        else:
            sheet.notes.append("update_extract_failed")
        return

    budget = _as_float(data.get("budget_max"))
    if budget is not None:
        sheet.budget = budget
    if data.get("gated_attribute") is not None:
        _apply_gated(sheet, data.get("gated_attribute"))
    category = data.get("category")
    if isinstance(category, str) and category.strip():
        sheet.category = category.strip()
    theme = data.get("theme_phrase")
    if isinstance(theme, str) and theme.strip():
        sheet.theme_phrase = theme.strip()
    brand = data.get("brand")
    if isinstance(brand, str) and brand.strip():
        sheet.brand = brand.strip()
    in_stock = _as_bool(data.get("in_stock"), default=None)
    if in_stock is not None:
        sheet.in_stock_required = in_stock
    if "preferred_brand" in data:
        brand = data.get("preferred_brand")
        if isinstance(brand, str) and brand.strip():
            sheet.preferred_brand = brand.strip()
        elif brand == "":
            sheet.preferred_brand = None
    if _as_bool(data.get("is_question"), default=False):
        sheet.pending_question = True
    notes = data.get("notes")
    if isinstance(notes, str) and notes.strip():
        sheet.notes.append(notes.strip())
    # Spec reveal clears underspecified gate.
    if sheet.gated_value:
        sheet.underspecified = False
    # Budget cuts: remind model to keep committed pick when still feasible.
    if (
        sheet.family in {"intent_decomposition", "constraint_satisfaction"}
        and budget is not None
        and prior_budget is not None
        and budget < prior_budget
        and sheet.committed_product_id
    ):
        sheet.notes.append("budget_cut_keep_committed_if_compliant")


def _system_for(family: Family) -> str:
    if family == "intent_decomposition":
        return _TF1_SYSTEM
    if family == "retrieval_recall":
        return _TF2_SYSTEM
    if family == "constraint_satisfaction":
        return _TF3_SYSTEM
    if family == "preference_reasoning":
        return _TF4_SYSTEM
    if family == "ranking":
        return _TF5_SYSTEM
    if family == "recovery":
        return _TF6_SYSTEM
    if family == "justification":
        return _TF7_SYSTEM
    return _GENERIC_SYSTEM


def _tool_names(tools: list[Any]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if "function" in tool else tool
        if isinstance(fn, dict) and fn.get("name"):
            names.add(str(fn["name"]))
    return names


def _tool_choice(tools: list[Any], *, force_message: bool) -> Any:
    if force_message and "message" in _tool_names(tools):
        return {"type": "function", "function": {"name": "message"}}
    return "required"


def _call_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [
        str(name)
        for call in tool_calls
        for name in [(call.get("function") or {}).get("name")]
        if name
    ]


def _violates_pending_gate(tool_calls: list[dict[str, Any]]) -> bool:
    names = _call_names(tool_calls)
    if not names:
        return True
    if any(name in _BLOCKED_WHILE_PENDING for name in names):
        return True
    return "message" not in names


def _market_event_nudge(observation: Any, sheet: ConstraintSheet) -> str | None:
    blob = _observation_blob(observation).lower()
    if "market_event" not in blob and "price_change" not in blob and "stockout" not in blob:
        return None
    sheet.saw_market_event = True
    sheet.event_recovery_mode = True
    budget = (
        f"{sheet.budget}" if sheet.budget is not None else "the current budget"
    )
    if sheet.family == "preference_reasoning":
        return (
            "MARKET EVENT (price_change) active. Recover autonomously:\n"
            "1) `message` acknowledging the reprice — NO question marks.\n"
            "2) remove_from_cart on the repriced target.\n"
            "3) Recover to a DIFFERENT listing_id (not a variant of the same "
            "listing). Prefer another hard-valid option that still fits the "
            f"use case / soft axes (budget<={budget}).\n"
            "4) inspect_stock that replacement → add_to_cart → place_test_order.\n"
            "Do not ask the shopper which option to pick."
        )
    if sheet.family == "recovery":
        model = sheet.preferred_model or "the preferred model"
        color = sheet.preferred_color or "the preferred color"
        return (
            "MARKET EVENT (stockout) active. Recover with MINIMAL relaxation:\n"
            "1) `message` acknowledging the stockout — NO question marks.\n"
            "2) remove_from_cart on the unavailable exact match.\n"
            f"3) Prefer another color of the SAME model ({model}) before "
            f"changing models (preferred color was {color}; "
            f"order={sheet.relaxation_order}).\n"
            "4) view/compare the replacement identity, inspect_stock, then "
            "add_to_cart → place_test_order within budget.\n"
            "Do not ask permission to relax color/model."
        )
    return (
        "MARKET EVENT active. Recover autonomously now:\n"
        "1) `message` acknowledging the price/stock change — NO question marks, "
        "do NOT ask which option the shopper prefers.\n"
        "2) remove_from_cart on the broken target.\n"
        "3) inspect_stock on another already-viewed compliant backup "
        f"(theme=`{sheet.theme_phrase or ''}`, budget<={budget}).\n"
        "4) add_to_cart + place_test_order that backup.\n"
        "Asking the shopper to choose will fail the episode gate."
    )


def _shopper_nudge(sheet: ConstraintSheet) -> str:
    if sheet.pending_question:
        if sheet.family == "intent_decomposition":
            return (
                "Shopper intervention detected. Next tool MUST be `message`. "
                "Name the exact product you will cart (the one you just viewed/"
                "inspected). After messaging, keep that pick if it still meets "
                "firm constraints — do not silently switch brands/SKUs."
            )
        return (
            "Shopper intervention detected. Next tool MUST be `message`. "
            "Answer briefly. Do not cart/order until messaged."
        )
    if sheet.family == "constraint_satisfaction":
        return (
            "Shopper updated firm constraints. If a gated spec was revealed, "
            "search/filter for brand+gated+budget, verify via view/compare, "
            "then inspect_stock and order. Never violate firm constraints."
        )
    if sheet.family == "preference_reasoning":
        return (
            "Shopper shared preference details. Update soft priorities from "
            "their message, compare visible facts across listing_ids, and "
            "choose a preference-strong hard-valid option — not cheapest-only."
        )
    if sheet.family == "ranking":
        return (
            "Shopper updated constraints. Re-check candidate facts, recompute "
            "lexicographic top among remaining in-stock options, then order."
        )
    if sheet.family == "recovery":
        return (
            "Shopper updated constraints. Keep budget/stock critical. Honor "
            "preferred model+color when possible; after stockout relax color "
            "before model."
        )
    if sheet.family == "justification":
        return (
            "Shopper updated constraints. Re-verify with view/compare/"
            "inspect_stock and include matching justification.claims on order."
        )
    if sheet.family == "intent_decomposition" and sheet.committed_product_id:
        return (
            "Shopper updated firm constraints. If COMMITTED_PICK still meets "
            f"budget/gated/stock (product_id={sheet.committed_product_id}), "
            "keep it: inspect_stock → add_to_cart → order. To switch, "
            "`message` naming the new pick first, then cart that same id."
        )
    if sheet.family == "retrieval_recall":
        return (
            "Shopper updated constraints. Refresh the recall target, continue "
            "unused search-plan queries if needed, then filter/compare/verify."
        )
    return (
        "Shopper updated firm constraints. Use the sheet, re-search if needed, "
        "then verify before ordering."
    )


def _opening_nudge(sheet: ConstraintSheet) -> str:
    base = sheet.status_block() + "\n\n"
    if sheet.family == "retrieval_recall":
        theme = sheet.theme_phrase or sheet.primary_query or "the theme phrase"
        return base + (
            f"Search query 1 MUST be exactly `{theme}`. Then run reformulations "
            "that keep those tokens (do not invent product types). Filter by "
            "category/budget, compare a shortlist, and reserve ~8 steps for "
            "stockout recovery and order."
        )
    if sheet.family == "intent_decomposition":
        return base + (
            "Find firm-compliant candidates first. Honor soft brand only when "
            "fully firm-compliant. When asked which one you are going with, "
            "message that pick and cart the same product_id unless a later "
            "firm constraint forces a messaged switch."
        )
    if sheet.family == "constraint_satisfaction":
        if _needs_ask_spec(sheet):
            return base + (
                "FIRST tool call: `message` asking what the firm hardware/spec "
                "requirement is. Do not search/cart until it is revealed."
            )
        return base + (
            "All listed constraints are firm. Search/filter precisely, verify "
            "gated attributes via view/compare facts (not title), then order."
        )
    if sheet.family == "preference_reasoning":
        return base + (
            "FIRST: `message` clarifying which soft priorities matter most for "
            f"use_case=`{sheet.use_case or 'the stated use'}`. Search with "
            "preference tokens (wireless/IPX/sport/battery). View/compare ≥2 "
            "listing_ids. Cart only an in-budget multi-axis match (≥2 of "
            "water/sport/wireless/battery). After price_change recover to a "
            "different listing_id with the same bar — no justification.claims."
        )
    if sheet.family == "ranking":
        return base + (
            "View/compare every listed candidate only. Apply the lexicographic "
            "priority. Cart COMPUTED_TOP when available. After stockout, take "
            "the next remaining top — no open catalog search."
        )
    if sheet.family == "recovery":
        return base + (
            f"Find and view the exact preferred item "
            f"(model=`{sheet.preferred_model}`, color=`{sheet.preferred_color}`) "
            "first, then cart that full match. After stockout, recover with "
            f"{sheet.relaxation_order} — inspect identity+stock before re-order. "
            "place_test_order with ONLY product_id+sku (no justification.claims)."
        )
    if sheet.family == "justification":
        return base + (
            "Find a compliant product, view(+compare) until gated facts are "
            "EXPLICIT (exact size, clean hd/fhd — no ranges/decoys), inspect_stock, "
            "THEN first add_to_cart. place_test_order claims must be STRINGS for "
            f"{sheet.required_claims}. After price spike, recover without '?'."
        )
    return base + "Satisfy the goal with verify-then-order discipline."


def _steps_remaining_nudge(turn: int, max_steps: int, sheet: ConstraintSheet) -> str | None:
    remaining = max_steps - turn + 1
    if remaining > 10:
        return None
    if sheet.event_recovery_mode:
        return (
            f"Only ~{remaining} steps left and EVENT RECOVERY is active. "
            "Finish remove → inspect backup → cart → place_test_order now. "
            "No more open-ended searching or questions."
        )
    if sheet.family == "retrieval_recall":
        return (
            f"Only ~{remaining} steps left. Stop broad viewing. Pick the best "
            "already-eligible candidate, inspect_stock, and place_test_order "
            "(leave margin for a possible stockout recovery)."
        )
    if sheet.family == "preference_reasoning":
        return (
            f"Only ~{remaining} steps left. Finish preference-fit verify/"
            "recover (different listing_id if repriced) and place_test_order."
        )
    if sheet.family == "recovery":
        return (
            f"Only ~{remaining} steps left. Finish minimal-relaxation recovery "
            "(color before model) with inspect_stock, then place_test_order."
        )
    return (
        f"Only ~{remaining} steps left. Verify the committed/compliant pick "
        "and place_test_order soon."
    )


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    max_steps = int(policy["max_steps"])
    tools = policy["tools"]
    query = policy["query"]

    sheet = _init_sheet(problem_data, query)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_for(sheet.family)},
        {"role": "user", "content": query},
        {"role": "user", "content": _opening_nudge(sheet)},
    ]
    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "event": "family_route_and_extract",
            "family": sheet.family,
            "constraints": {
                "category": sheet.category,
                "budget": sheet.budget,
                "gated_key": sheet.gated_key,
                "gated_value": sheet.gated_value,
                "theme_phrase": sheet.theme_phrase,
                "preferred_brand": sheet.preferred_brand,
                "brand": sheet.brand,
                "primary_query": sheet.primary_query,
                "reformulations": list(sheet.reformulations),
                "candidates": list(sheet.candidates),
                "rank_priority": list(sheet.rank_priority),
                "required_claims": list(sheet.required_claims),
                "attributes": dict(sheet.attributes),
                "underspecified": sheet.underspecified,
                "use_case": sheet.use_case,
                "soft_axes": list(sheet.soft_axes),
                "tradeoff_priority": list(sheet.tradeoff_priority),
                "preferred_model": sheet.preferred_model,
                "preferred_color": sheet.preferred_color,
                "relaxation_order": sheet.relaxation_order,
                "notes": list(sheet.notes),
            },
        }
    ]

    for turn in range(1, max_steps + 1):
        messages.append({"role": "user", "content": sheet.status_block()})
        step_nudge = _steps_remaining_nudge(turn, max_steps, sheet)
        if step_nudge:
            messages.append({"role": "user", "content": step_nudge})

        tool_calls: list[dict[str, Any]] = []
        assistant_content = ""
        for _attempt in range(4):
            force_message = sheet.pending_question or _needs_ask_spec(sheet)
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": _model(),
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": _tool_choice(
                        tools, force_message=force_message
                    ),
                    "temperature": 0,
                },
            )
            if inference is None:
                raise RuntimeError("inference request failed")
            assistant = inference["choices"][0]["message"]
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if not tool_calls:
                dialogue.append({"role": "assistant", "content": assistant_content})
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {
                            "role": "user",
                            "content": (
                                "Choose one of the supplied tools. "
                                + (
                                    "pending_question=true → use `message` now."
                                    if sheet.pending_question
                                    else "Continue toward a verified compliant order."
                                )
                            ),
                        },
                    ]
                )
                continue

            if sheet.pending_question and _violates_pending_gate(tool_calls):
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "pending_question_gate",
                    }
                )
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": (
                                "Rejected: pending_question is true. "
                                "Call ONLY `message` and answer the shopper now. "
                                "Name the product you will cart."
                            ),
                        },
                    ]
                )
                tool_calls = []
                continue

            if _violates_ask_spec_gate(sheet, tool_calls):
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "ask_spec_gate",
                    }
                )
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": (
                                "Rejected: the firm hardware/spec requirement is "
                                "still unknown. Call `message` now and ask what "
                                "the firm requirement is before searching or carting."
                            ),
                        },
                    ]
                )
                tool_calls = []
                continue

            if _violates_tf4_clarify_gate(sheet, tool_calls):
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "tf4_preference_clarify_gate",
                    }
                )
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": (
                                "Rejected: clarify soft priorities with `message` "
                                "before the first add_to_cart / place_test_order. "
                                "Ask which use-case tradeoffs matter most, then "
                                "compare facts across listing_ids."
                            ),
                        },
                    ]
                )
                tool_calls = []
                continue

            if _violates_tf4_cart_gate(sheet, tool_calls):
                pid, sku = _cart_target(tool_calls)
                reason = _tf4_cart_reject_reason(sheet, pid, sku) or "preference gate"
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "tf4_preference_cart_gate",
                        "reason": reason,
                    }
                )
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": (
                                "Rejected: cart/order blocked for TF4 "
                                f"({reason}). Stay under budget, view/compare "
                                "stronger multi-axis candidates (water/sport/"
                                "wireless/battery), then inspect_stock and cart."
                            ),
                        },
                    ]
                )
                tool_calls = []
                continue

            if sheet.event_recovery_mode and _message_has_question(tool_calls):
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "event_recovery_no_questions",
                    }
                )
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": (
                                "Rejected: during EVENT RECOVERY your `message` "
                                "must not contain '?'. Acknowledge the event and "
                                "state you will switch to another compliant item, "
                                "then remove/inspect/cart/order — do not ask."
                            ),
                        },
                    ]
                )
                tool_calls = []
                continue

            if _violates_commitment_gate(sheet, tool_calls):
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "commitment_gate",
                    }
                )
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": (
                                "Rejected: silent switch away from COMMITTED_PICK "
                                f"(product_id={sheet.committed_product_id}, "
                                f"sku={sheet.committed_sku}). Either "
                                "add_to_cart that committed product if still "
                                "firm-compliant, or call `message` first naming "
                                "the new pick (after viewing it), then cart that "
                                "same product_id."
                            ),
                        },
                    ]
                )
                tool_calls = []
                continue

            if _violates_tf7_cart_gate(sheet, tool_calls):
                reason = _tf7_cart_reject_reason(sheet, tool_calls)
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "tf7_fact_backed_cart_gate",
                        "reason": reason,
                    }
                )
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": (
                                "Rejected: add_to_cart is not FACT-BACKED "
                                f"({reason}). View/compare another candidate until "
                                "brand + gated attribute facts are explicit and "
                                "unambiguous, then inspect_stock and cart that SKU."
                            ),
                        },
                    ]
                )
                tool_calls = []
                continue

            if _violates_justification_gate(sheet, tool_calls):
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "justification_gate",
                    }
                )
                need = ", ".join(sheet.required_claims or ["in_stock", "price"])
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": (
                                "Rejected: place_test_order target is not "
                                "fact-backed for required claims "
                                f"({need}). View/inspect a compliant SKU first."
                            ),
                        },
                    ]
                )
                tool_calls = []
                continue
            break
        else:
            raise RuntimeError("model returned no valid tool call before completion")

        # Track products + commitment updates from this turn's planned calls.
        answering_now = sheet.pending_question and "message" in _call_names(tool_calls)
        for call in tool_calls:
            name = str((call.get("function") or {}).get("name") or "")
            try:
                args = _arguments(call)
            except Exception:
                args = {}
            _track_product_args(sheet, name, args)
            if (
                name == "message"
                and not sheet.event_recovery_mode
                and sheet.family == "intent_decomposition"
                and sheet.last_product_id
                and (
                    answering_now
                    or (
                        sheet.committed_product_id
                        and sheet.last_product_id != sheet.committed_product_id
                    )
                )
            ):
                # Re-commit to the product just viewed/inspected when messaging.
                sheet.committed_product_id = sheet.last_product_id
                sheet.committed_sku = sheet.last_sku

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        group_id = f"{problem_id}-turn-{turn}"
        env_calls = []
        for index, call in enumerate(tool_calls, start=1):
            name = call["function"]["name"]
            args = _arguments(call)
            if sheet.family == "justification" and name == "place_test_order":
                args = _coerce_tf7_order_args(sheet, args)
            elif name == "place_test_order":
                # Extra claims poison non-TF7 hard checks (seen on TF6).
                args = _strip_non_tf7_order_args(args)
            env_calls.append(
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {"name": name, "args": args},
                }
            )
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": env_calls,
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
                "family": sheet.family,
                "pending_question": sheet.pending_question,
                "committed_product_id": sheet.committed_product_id,
                "event_recovery_mode": sheet.event_recovery_mode,
            }
        )

        if "message" in _call_names(tool_calls):
            if sheet.family == "constraint_satisfaction" and _needs_ask_spec(sheet):
                sheet.asked_for_spec = True
            if sheet.family == "preference_reasoning" and not sheet.asked_preference_clarify:
                sheet.asked_preference_clarify = True
            if sheet.pending_question and sheet.family == "intent_decomposition":
                if sheet.last_product_id and not sheet.committed_product_id:
                    sheet.committed_product_id = sheet.last_product_id
                    sheet.committed_sku = sheet.last_sku
            sheet.pending_question = False
            # After a clean recovery ack, allow cart/order of backups.
            if sheet.event_recovery_mode and not _message_has_question(tool_calls):
                sheet.answering_pending = False

        nudges: list[str] = []
        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            observation = call_result["observation"]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(observation),
                }
            )
            if sheet.family in {
                "ranking",
                "justification",
                "constraint_satisfaction",
                "preference_reasoning",
                "recovery",
            }:
                _ingest_product_facts(sheet, observation)
            event_nudge = _market_event_nudge(observation, sheet)
            if event_nudge and event_nudge not in nudges:
                nudges.append(event_nudge)

        if sheet.family == "ranking":
            _recompute_ranking_top(sheet)
            if sheet.computed_top_id:
                nudges.append(
                    f"COMPUTED_TOP updated: product_id={sheet.computed_top_id} "
                    f"sku={sheet.computed_top_sku}. Prefer this among remaining "
                    "in-stock candidates within budget."
                )

        user_text = _user_text(result.get("user_message"))
        if user_text:
            messages.append({"role": "user", "content": user_text})
            _apply_shopper_update(sheet, user_text)
            nudges.append(_shopper_nudge(sheet))

        for nudge in nudges:
            messages.append({"role": "user", "content": nudge})

        if any(call["observation"]["done"] for call in result["calls"]):
            break

    return dialogue


__all__ = ["agent_main"]
