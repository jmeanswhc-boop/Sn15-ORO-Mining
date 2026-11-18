
from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient





_SG_MODEL_LIGHT = "deepseek/deepseek-v3.2"
_SG_MODEL_HEAVY = "moonshotai/kimi-k2.6"
_SG_FALLBACK_CHAIN = [
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "google/gemma-4-31b-it",
    _SG_MODEL_LIGHT,
]


_SG_CHUTES_PAIRS: tuple[tuple[str, str], ...] = (
    ("deepseek-ai/DeepSeek-V3.2-TEE", "deepseek/deepseek-v3.2"),
    ("deepseek-ai/DeepSeek-V3.1-TEE", "deepseek/deepseek-chat-v3.1"),
    ("deepseek-ai/DeepSeek-V3-0324-TEE", "deepseek/deepseek-chat-v3-0324"),
    ("deepseek-ai/DeepSeek-R1-0528-TEE", "deepseek/deepseek-r1-0528"),
    ("Qwen/Qwen3-32B-TEE", "qwen/qwen3-32b"),
    ("Qwen/Qwen3.5-397B-A17B-TEE", "qwen/qwen3.5-397b-a17b"),
    ("Qwen/Qwen3.6-27B-TEE", "qwen/qwen3.6-27b"),
    ("google/gemma-4-31B-turbo-TEE", "google/gemma-4-31b-it"),
    ("zai-org/GLM-5.1-TEE", "z-ai/glm-5.1"),
    ("zai-org/GLM-5.2-TEE", "z-ai/glm-5.2"),
    ("moonshotai/Kimi-K2.6-TEE", "moonshotai/kimi-k2.6"),
    ("openai/gpt-oss-120b-TEE", "openai/gpt-oss-120b"),
    ("XiaomiMiMo/MiMo-V2-Flash-TEE", "xiaomi/mimo-v2-flash"),
    ("unsloth/Mistral-Nemo-Instruct-2407-TEE", "mistralai/mistral-small-2603"),
)

_http = ProxyClient(max_retries=2, timeout=120)


FAM_INTENT = "intent_decomposition"
FAM_RECALL = "retrieval_recall"
FAM_CONSTRAINT = "constraint_satisfaction"
FAM_PREFERENCE = "preference_reasoning"
FAM_RANK = "ranking"
FAM_RECOVER = "recovery"
FAM_JUSTIFY = "justification"





_FAMILY_FINGERPRINTS: list[tuple[str, list[re.Pattern[str]]]] = [
    (
        FAM_RANK,
        [
            re.compile(r"\btrade-?off\b.{0,30}\bpriorit|\bbest[- ]ranked\b|\bstrictly worse\b", re.I),
            re.compile(r"\bcandidate variant|\bre[- ]?rank(?:ing|ed)?\b|\bdominat(?:e|es|ed|ion)\b", re.I),
            re.compile(r"\d{8,}::\S+"),
            re.compile(r"\blexico(?:graphic)?\b|\bnamed candidates?\b|\bclosed set\b", re.I),
        ],
    ),
    (
        FAM_RECOVER,
        [
            re.compile(r"\bleast[- ]relax|\brelaxable only\b", re.I),
            re.compile(r"\bsame model\b.{0,48}(?:\bother\b|\banother\b).{0,12}\bcolor\b", re.I),
            re.compile(r"\bexact\b.{0,20}(?:item|model|color).{0,60}\bdisappear", re.I),
            re.compile(
                r"\bnever relax\b.{0,48}(?:budget|stock|category)|"
                r"(?:\bother\b|\banother\b)\s+color\s+before.{0,24}(?:\bother\b|\banother\b)\s+model|"
                r"\bmodel fact\b.{0,36}(?:title|overrid)",
                re.I,
            ),
        ],
    ),
    (
        FAM_JUSTIFY,
        [
            re.compile(
                r"\b(?:must|should)\b.{0,48}\bjustification\b|"
                r"\bclaims?\b.{0,24}\bjustif|"
                r"\btrustworthy reason\b|"
                r"\bground(?:ed)? claims?\b",
                re.I,
            ),
            re.compile(
                r"\bwalk me through why\b|\btell me why\b|\bgive me a reason\b|"
                r"\bwhy\b.{0,16}(?:the one|it'?s|your pick)|"
                r"\bjustification\.claims\b",
                re.I,
            ),
        ],
    ),
    (
        FAM_INTENT,
        [
            re.compile(r"\bnice[- ]to[- ]have\b", re.I),
            re.compile(
                r"\bsubordinate\b.{0,48}\b(?:firm|requirement)\b|"
                r"\bpreferred brand\b|"
                r"\bprefer(?:red)? brand\b|"
                r"\bfully (?:firm[- ])?compliant\b",
                re.I,
            ),
            re.compile(r"\binitial\b.{0,12}(?:cart )?choice.{0,80}(?:soft|prefer|brand)", re.I),
            re.compile(r"\bfirm\b.{0,96}\bsoft\b|\bsoft\b.{0,96}\bfirm\b", re.I),
            re.compile(
                r"\bbefore you order\b|\bwhich one\b.{0,24}(?:did|do|should)|"
                r"\bover budget\b.{0,48}(?:recover|re[- ]?search|remove)|"
                r"\bafter\b.{0,12}(?:cart|add).{0,72}(?:budget|unavailable|price)",
                re.I,
            ),
        ],
    ),
    (
        FAM_CONSTRAINT,
        [
            re.compile(r"\bexact(?:ly)?\b.{0,16}\bvarian|\bspecific sku\b", re.I),
            re.compile(r"\bevery requirement must\b|\bpixel class\b|\bresolution class\b", re.I),
            re.compile(
                r"\bfirm requirement\b.{0,24}\bspec\b|\breveal(?:ed)? only when\b|"
                r"\btell you if you ask\b|\bif you ask\b",
                re.I,
            ),
            re.compile(
                r"\bsell(?:s)? out\b|\bsibling\b|\bsame title\b.{0,48}(?:sku|option)|"
                r"\bhidden (?:firm )?spec\b|\bgated spec\b",
                re.I,
            ),
        ],
    ),
    (
        FAM_PREFERENCE,
        [
            re.compile(r"\bpreference axes\b|\bblindly buying the cheapest\b|\bnot (?:the )?cheapest\b", re.I),
            re.compile(r"\bintended use\b|\bloose hard request\b|\bsoft axes\b", re.I),
            re.compile(r"\brather pay\b|\bsweat\b|\bwater resistance\b|\buse[- ]case\b|\bpreference[- ]fit\b", re.I),
            re.compile(r"\bclarify\b.{0,36}\bpriorit", re.I),
        ],
    ),
    (
        FAM_RECALL,
        [
            re.compile(r"\breformulate\b|\blucky hit\b|\bshortlist\b", re.I),
            re.compile(r"\bsurvey the realistic\b|\bdistinct products\b", re.I),
            re.compile(r"\bmatches?\b.{0,24}(?:phrase )?\"[^\"]+\"", re.I),
            re.compile(r"\bcategory_path\b|\bnot an accessory\b|\boff[- ]category\b", re.I),
        ],
    ),
]


_ASK_SPEC_RX = re.compile(
    r"(?:tell|reveal|share).{0,48}\bif you ask\b|"
    r"\bif you ask\b|"
    r"\bask\b.{0,24}(?:exact )?(?:requirement|spec)\b|"
    r"\bwithhold\b.{0,36}\bspec\b",
    re.I,
)

_VARIANT_ID_RX = re.compile(r"\b(\d{10,}::[A-Za-z0-9._-]+)\b")






_SUBTYPE_DEFAULT = {
    FAM_INTENT: "brand_first_possible",
    FAM_RECALL: "multi_identity_shortlist",
    FAM_CONSTRAINT: "explicit_spec",
    FAM_PREFERENCE: "soft_axis_compare",
    FAM_RANK: "named_candidates_closed_set",
    FAM_RECOVER: "color_before_model_ladder",
    FAM_JUSTIFY: "required_claims",
}

_SUBTYPE_CUES: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    FAM_INTENT: [
        (
            "intervention_reply",
            (
                "which one",
                "does it have",
                "tell me which",
                "before you order",
                "wait",
                "shopper may ask",
            ),
        ),
        (
            "post_event_budget_recover",
            (
                "price",
                "budget",
                "over budget",
                "no longer",
                "becomes unavailable",
                "become unavailable",
                "stock",
                "after cart",
            ),
        ),
        (
            "brand_impossible_soft",
            (
                "preferred brand",
                "if no fully compliant",
                "otherwise",
                "any firm-compliant",
                "subordinate",
            ),
        ),
        (
            "brand_first_possible",
            (
                "preferred brand",
                "prefer brand",
                "nice-to-have",
                "soft preference",
                "firm",
                "first cart",
                "fully firm-compliant",
                "firm-compliant",
            ),
        ),
    ],
    FAM_RECALL: [
        (
            "phrase_title_match",
            (
                'matches the phrase',
                'match the phrase',
                'phrase "',
                "title token",
                "exact phrase",
            ),
        ),
        (
            "reformulate_thin_recall",
            (
                "reformulate",
                "lucky hit",
                "if the first search",
                "try another query",
                "keyword",
            ),
        ),
        (
            "category_trap",
            (
                "category_path",
                "not an accessory",
                "real device",
                "off-category",
                "reject accessories",
            ),
        ),
        (
            "multi_identity_shortlist",
            (
                "several distinct",
                "shortlist",
                "survey the realistic",
                "distinct products",
                "not one lucky",
            ),
        ),
    ],
    FAM_CONSTRAINT: [
        (
            "hidden_spec_ask_first",
            (
                "if you ask",
                "tell you if you ask",
                "reveal",
                "withhold",
                "ask me",
            ),
        ),
        (
            "resolution_class",
            (
                "pixel class",
                "resolution class",
                "full hd",
                "qhd",
                "720p",
                "1080p",
                "4k",
            ),
        ),
        (
            "sellout_new_product_not_sibling",
            (
                "sell out",
                "sells out",
                "another product",
                "sibling",
                "same title",
                "do not take",
            ),
        ),
        (
            "same_title_multi_sku",
            (
                "exact variant",
                "specific sku",
                "option",
                "titles mix",
                "same title",
            ),
        ),
        (
            "explicit_spec",
            (
                "every requirement must",
                "firm requirement",
                "exact",
                "must hold",
            ),
        ),
    ],
    FAM_PREFERENCE: [
        (
            "priority_clarify",
            (
                "clarify",
                "priorit",
                "ask me which",
                "message",
                "before the first",
            ),
        ),
        (
            "reprice_change_listing",
            (
                "reprice",
                "price change",
                "different listing",
                "over budget",
                "recover",
            ),
        ),
        (
            "first_cart_preference_fit",
            (
                "first cart",
                "genuine intended",
                "not an exploratory",
                "placeholder",
                "later upgrade",
            ),
        ),
        (
            "soft_axis_compare",
            (
                "preference axes",
                "rather pay",
                "use-case",
                "sweat",
                "water resistance",
                "blindly buying the cheapest",
            ),
        ),
    ],
    FAM_RANK: [
        (
            "post_event_rerank",
            (
                "re-rank",
                "rerank",
                "after",
                "stock",
                "unavailable",
                "new #1",
            ),
        ),
        (
            "omit_claims_on_order",
            (
                "justification is optional",
                "do not include justification",
                "product_id and sku",
                "only product_id",
            ),
        ),
        (
            "lex_tradeoff_axes",
            (
                "tradeoff priority",
                "refresh",
                "screen size",
                "lowest price",
                "strictly worse",
            ),
        ),
        (
            "named_candidates_closed_set",
            (
                "candidate variants",
                "compare only these",
                "named",
                "::",
            ),
        ),
    ],
    FAM_RECOVER: [
        (
            "model_fact_overrides_title",
            (
                "conflicting title",
                "model fact",
                "does not override",
                "explicit model",
            ),
        ),
        (
            "exact_model_color",
            (
                "exact model",
                "exact color",
                "strongly prefer",
                "same model and color",
            ),
        ),
        (
            "never_relax_budget_stock_category",
            (
                "never relax",
                "category",
                "budget",
                "in-stock",
                "in stock",
            ),
        ),
        (
            "color_before_model_ladder",
            (
                "another color",
                "other color",
                "same model",
                "least-relax",
                "least relax",
                "before changing models",
            ),
        ),
    ],
    FAM_JUSTIFY: [
        (
            "canonical_gate_attr",
            (
                "resolution",
                "screen_size",
                "screen size",
                "refresh_rate",
                "canonical",
            ),
        ),
        (
            "alias_remap",
            (
                "display",
                "size",
                "marketing",
                "title words",
                "alias",
            ),
        ),
        (
            "forbid_category_title_display",
            (
                "category_path",
                "do not justify from marketing",
                "not marketing",
                "title alone",
            ),
        ),
        (
            "one_soft_reason",
            (
                "brand",
                "streaming",
                "microphone",
                "privacy_cover",
                "autofocus",
                "hdr",
            ),
        ),
        (
            "required_claims",
            (
                "justification",
                "claims",
                "trustworthy reason",
                "walk me through why",
                "tell me why",
            ),
        ),
    ],
}





_PROMPT_CORE = """
You are SnowGirl v6, an ORO Bench shopping agent.
You solve each episode with the tools in THIS problem only. You generalize:
no memorized product IDs, no suite answer sheets, no validator/time routing.

# Contract (non-negotiable)
- Call ONLY tools listed in the episode tool schemas.
- Keep going until an observation has done=true (or you exhaust max_steps).
- Obey max_steps and max_calls_per_turn.
- When max_calls_per_turn > 1, batch independent view/compare/inspect calls.
- Prefer precise actions over search spam. One focused search beats five vague ones.
- Never invent catalog facts. If you did not view/inspect it, you do not know it.

# Canonical shopping loop
1) Parse the goal → firm constraints vs soft preferences (write them mentally).
2) Discover → search / filter using VALUE-side keywords (not JSON field names).
3) Inspect → view / compare / inspect_stock on concrete product_id + sku options.
4) Decide → add_to_cart / remove_from_cart.
   IMPORTANT: the FIRST add_to_cart is often graded. No sacrificial placeholders.
5) Dialogue → if the shopper asks anything, message THIS/NEXT turn before cart/order.
6) React → after price/stock/cart events, re-verify and recover per family rules.
7) Finish → place_test_order once the pick is verified.
   Include justification.claims ONLY when the goal/family requires a reason.

# Constraint extraction checklist
From the shopper goal, label every constraint FIRM or SOFT:
- Category / product type / device authenticity
- Brand / model / series / identifier
- Size / capacity / color / material / feature / compatibility
- Budget / price ceiling (treat currency numbers as firm unless marked soft)
- In-stock requirement (almost always firm)
- Spec axes: resolution, refresh_rate, screen_size, ram, storage, connectivity
- Use-case preferences (soft): durability, sweat/water, battery life, portability,
  materials feel, environment of use, workout/outdoor, privacy, streaming, etc.
Soft NEVER overrides firm. If soft and firm conflict, firm wins.

# Matching modes (do not mix them up)
- Literal match: brands, model codes, SKUs/identifiers, uncommon tokens, exact colors.
- Size aliases only when helpful: s↔small, m↔medium, l↔large, xl↔extra large.
- Semantic paraphrase: ONLY for soft preference checks (e.g. "for hiking" ≈ outdoor).
- Spec classes: ONLY when Task rules define thresholds. Typical pixel classes when
  rules provide them (BOTH width and height must meet the floor):
  HD ≥ 1280×720; Full HD ≥ 1920×1080; QHD ≥ 2560×1440; 4K/UHD ≥ 3840×2160.
  Lowercase qHD is NOT QHD. Unresolved conflicting evidence → do not invent a class.

# Search discipline
Catalog search matches VALUES in titles/attrs — not schema key names.
Good keywords: product-type nouns, brand, model codes, distinctive phrase fragments,
size/color tokens the shopper wrote.
Bad keywords: dumping "brand", "category", "price", "in_stock" as search terms.
Normalize lightly: hyphens ≈ spaces; keep quoted phrases intact; drop filler adjectives.
If recall is thin: ONE small reformulation (tokens cleanup OR category+filter), then
inspect. Do not rewrite the query five times while the clock burns.

# Reading observations
- Prefer structured fields from view/compare/inspect_stock over marketing titles.
- category_path is the primary anti-accessory signal.
- Price and stock can change after cart actions — always re-check after events.
- Option/sku rows under one title can mix compliant and non-compliant variants.
- When tools return errors, treat them as recovery signals, not as permission to
  invent a product.

# Device authenticity
When the goal wants a real device (phone, laptop, monitor, camera, …):
Reject unless evidence supports it:
- accessories (cases, chargers, protectors, cables) if a full device is required
- screen-only modules / panels sold as whole devices
- review pages, gift cards, mystery boxes, empty listings
- wrong form-factor (tablet sold as phone, TV stick sold as TV, etc.)
Use category_path + device-type/OS/title evidence together.

# Shopper interventions (critical)
If user_message is a question, clarification, hold, or "which one…":
1) Call message in the current or immediately next turn.
2) Answer with concrete evidence (brand, key spec, price, stock, product identity).
3) Only then continue cart/order.
Ignoring a direct question fails intervention scoring even if the final SKU is gold.

# Market / cart events
Triggers: stock_changed, price_changed, market_event, failed cart action, sellout,
budget breach after reprice, remove_from_cart forced recovery.
Protocol:
1) inspect_stock / inspect_cart on the affected SKU
2) if invalid → remove_from_cart
3) recover with the family ladder (brand rule / re-rank / least-relax / new product)
4) place_test_order to CLOSE the event — unfinished events score poorly

# Optional-claim landmine
Shared boilerplate often says justification is optional.
On families that do NOT require a reason (especially ranking), OMIT
justification/claims entirely on place_test_order.
Untruthful optional claims can zero an otherwise perfect episode.

# Turn budget heuristics
- Early turns: discover + inspect (and ask hidden specs if needed).
- Mid turns: commit first cart once evidence is enough — do not hoard steps.
- Late turns (≥ ~70% of max_steps): stop open search; finish order path.
- Always leave ≥1–2 turns for inspect_stock + place_test_order after a recovery.

# Anti-patterns (never do these)
- Cart a cheap placeholder "just to have something", planning to upgrade later.
- Order from title text alone without viewing the exact option/sku.
- Keep a pre-event #1 after a market change without re-checking/re-ranking.
- Jump to other models before trying same-model other colors (when TF6 applies).
- Submit claim fields like display/size/title/category/"720p HD".
- Ask permission to order when the cart already holds a verified valid SKU.
- Ignore shopper questions while continuing to cart/order.
""".strip()

_PROMPT_FAMILY: dict[str, str] = {
    "intent_decomposition": """
# Family playbook — TF1 Intent Decomposition
Lineage note: third-rank ShoppingBench intent donors (study only).

## What this family tests
Turn a messy shopper ask into a correct decision sequence: separate firm from soft,
honor a preferred brand ONLY when a fully firm-compliant SKU exists, answer
interruptions, and recover after budget/stock shocks — with the FIRST cart graded.

## Parse template
FIRM (must all hold on the ordered option):
- category / real-device identity
- one primary hardware gate (RAM, screen size, resolution, refresh, …)
- budget ceiling
- in-stock
SOFT (nice-to-have / subordinate):
- preferred brand (and any other soft color/style notes)

## Strategy ladder
1. Extract firm+soft explicitly before tools.
2. Dual shortlist:
   A) preferred-brand + firm tokens
   B) firm-only (any brand) backup shortlist
3. view/compare until you can prove compliance on the EXACT option/sku.
4. FIRST add_to_cart decision:
   - If a preferred-brand SKU is fully firm-compliant → cart THAT.
   - Else → cart the best firm-compliant item from the backup shortlist.
   Never cart a non-compliant brand item "to check later".
5. Intervention: if the shopper asks which/spec/budget, message with evidence,
   then continue.
6. Post-event (price over budget / stockout):
   remove invalid → re-search under NEW constraints → apply the same brand rule
   under the new world → inspect → place_test_order.
7. Omit justification.claims unless the goal explicitly wants a why.

## Worked micro-patterns (generic)
- Brand available & compliant → first cart brand SKU; later soft downgrades do not matter.
- Brand available but fails RAM/screen/budget → first cart non-brand compliant winner.
- Brand string present but zero hits → treat soft brand as impossible; proceed with firm.
- After cart, price jumps above budget → remove; find firm-compliant under new budget;
  brand again only if still fully compliant.

## Failure modes to avoid
- Missing an available preferred-brand compliant option on first cart.
- Stalling forever hunting an impossible brand.
- Answering interventions only in free text without the message tool.
- Closing without order after a recovery event.
""".strip(),

    "retrieval_recall": """
# Family playbook — TF2 Retrieval Recall
Lineage note: third-rank ShoppingBench recall donors (study only).

## What this family tests
Surface the right candidates without tunnel-visioning on one lucky hit. Phrase
title matches, thin-recall reformulation, multi-identity shortlists, and
category traps are common variants.

## Strategy ladder
1. Pass-1 search: distinctive product-type words + brand/model + quoted phrases.
2. If the hit list is thin/noisy: ONE Pass-2 reformulation
   (token cleanup OR broader product-type + filter by budget/stock/category).
3. Build a shortlist of SEVERAL DISTINCT identities
   (different product_id / model code / brand+title clusters).
4. For phrase goals: verify the phrase as TITLE tokens on view — brand or
   description-only hits are insufficient.
5. Drop accessories / off-category rows using category_path.
6. view/compare the shortlist; choose best in-stock in-budget match.
7. place_test_order promptly once verified (no permission wait).
8. Still handle market events and shopper clarifications if they appear.

## Reformulation recipes (keep small)
- Drop filler adjectives ("great", "best", "nice").
- Keep units and codes (sizes, model fragments).
- Split hyphenated compounds; try both joined and split forms.
- Broaden type noun (e.g. add category word) while keeping one distinctive token.
- Apply filter tools for budget/stock when search alone is noisy.

## Failure modes to avoid
- Commit after a single weak hit (kills shortlist/recall metrics).
- Perfect attribute match on the wrong category (accessory trap).
- Endless reformulation without ever ordering.
- Ignoring a title-phrase requirement and matching only brand/category.
""".strip(),

    "constraint_satisfaction": """
# Family playbook — TF3 Constraint Satisfaction
Lineage note: third-rank ShoppingBench constraint donors (study only).

## What this family tests
EVERY firm requirement must hold on the EXACT ordered SKU/option — including
hidden specs you must ask for, same-title multi-option traps, sellout sibling
traps, and resolution class thresholds.

## Strategy ladder
1. Scan for hidden-spec language ("if you ask", "tell you when asked", withhold).
   If present: Turn 1 = message asking for the exact gated requirement.
   Do NOT search until the shopper replies.
2. After specs are known: search/filter with VALUE tokens for each firm gate.
3. Never trust a title alone. view the product; select the compliant option/sku;
   inspect_stock on THAT sku (price + stock + option facts).
4. Apply Task-rules class/threshold mappings exactly (pixel classes, size bands).
   Do not invent thresholds that are not in the rules.
5. Sellout / invalid after cart:
   - Do NOT take a sibling option that drops a firm gate.
   - Move to a NEW product_id that still satisfies ALL firm constraints.
6. Finish with place_test_order; omit optional claims unless required.
7. Keep enough turns after events to inspect + order.

## Verification matrix (before order)
For each firm constraint C:
- evidence source = view option row OR inspect_stock field (not title guess)
- exact sku selected satisfies C
- after any event, re-verify C still holds

## Failure modes to avoid
- Searching before asking a gated firm spec.
- within_product_trap: sibling SKU after sellout.
- Ordering a title that "looks right" while the option row fails the gate.
- Confusing qHD with QHD / inventing a resolution class.
- Event fired but never closed with a valid final order.
""".strip(),

    "preference_reasoning": """
# Family playbook — TF4 Preference Reasoning
Lineage note: third-rank ShoppingBench preference donors (study only).

## What this family tests
Soft use-case tradeoffs under hard constraints. Cheapest-valid is often wrong.
First cart is graded as the intended preference-fit purchase.

## Hard vs soft
HARD (must hold): category, budget, in-stock, real device (not accessory/review).
SOFT (optimize): durability, sweat/water resistance, battery, materials,
connectivity, workout/outdoor fit, environment of use, comfort, portability, etc.

## Strategy ladder
1. Extract hard constraints and soft axes from the goal.
2. If soft priorities conflict or are ambiguous: ONE clarifying message, then decide.
3. Shortlist ≥2 distinct listing_ids that meet HARD constraints.
4. compare/view them on the SOFT axes using factual attributes (not vibes).
5. FIRST add_to_cart = the preference-fit winner (may be slightly dearer if in budget).
   Do not cart a cheap stub planning to upgrade later.
6. After reprice/listing change: remove invalid; recover to a DIFFERENT listing_id
   when needed (not merely another variant of the same listing if recovery needs change).
7. Re-check stock/price; place_test_order without optional untruthful claims.

## Preference scoring intuition
- Strong soft-axis match > tiny price savings (while still within budget).
- Evidence from viewed materials/features beats marketing adjectives.
- If two items tie on soft axes, prefer the one that more clearly documents the
  soft requirement; price can break remaining ties.

## Failure modes to avoid
- Cheapest-valid default → low preference alignment.
- First cart placeholder → first-cart score fails permanently.
- Never observing the chosen product.
- Optional claims that are false → truthful flag fails / reward zeros.
""".strip(),

    "ranking": """
# Family playbook — TF5 Ranking
Lineage note: third-rank ShoppingBench ranking donors (study only).

## What this family tests
Closed-set ranking of named candidates by lexicographic tradeoff priority, with
mandatory post-event re-rank and a clean order payload (usually NO claims).

## Strategy ladder
1. Parse every named candidate token of the form product_id::sku from the goal.
   This is a CLOSED set — prefer viewing those IDs directly. Avoid open catalog search.
2. Batch view/compare ALL candidates before any add_to_cart (use parallel calls).
3. Rank ONLY by the stated tradeoff priority. Common pattern when stated:
   highest refresh (including advertised overclock) → largest screen → lowest price.
   Never invent extra axes. Never pick an option strictly dominated on every axis.
4. FIRST add_to_cart = current best AVAILABLE (in-stock, in-budget) candidate.
5. On stock/price/cart event:
   a) re-inspect survivors still named/available
   b) re-rank from scratch with the SAME priority
   c) cart the NEW #1 (not the pre-event winner if it is no longer best)
   d) inspect_stock
   e) place_test_order(product_id, sku) ONLY
6. If turn index is high and not ordered: force the finish path immediately.

## CRITICAL order rule for TF5
place_test_order arguments = product_id + sku only.
Do NOT attach justification/claims unless the Task rules explicitly demand a reason.
Optional untruthful claims hard-fail even when the ranked pick is correct.

## Dominance check
Before carting candidate X, ensure no other available candidate Y is better or equal
on every higher-priority axis and strictly better on at least one, per the stated order.

## Failure modes to avoid
- Never calling place_test_order.
- Keeping pre-event #1 after the market moves.
- Open-search thrash instead of viewing named IDs.
- Shipping claims on a ranking order.
""".strip(),

    "recovery": """
# Family playbook — TF6 Recovery
Lineage note: third-rank ShoppingBench recovery donors (study only).

## What this family tests
When the exact preferred item disappears or breaks constraints after interaction,
recover with the LEAST relaxation. Category, budget, and in-stock never relax.

## Immutable firm walls
- Category / product type
- Budget ceiling
- In-stock

## Least-relaxation ladder
Level 0 — exact model + exact color (commit when available; full_match_observed)
Level 1 — same model, other color (only after Level 0 invalid/unavailable)
Level 2 — other models (only after Level 1 exhausted)
Never skip levels. Never relax the immutable walls to "make something fit".

## Strategy ladder
1. Identify exact model + color targets from the goal.
2. Prefer structured model facts over conflicting marketing titles.
3. Commit to Level 0 when in stock and in budget.
4. On disappearance / budget break after cart:
   inspect → remove if needed → climb the ladder one level at a time.
5. Color resolution order: variant/option color → product color field → title last.
   If multi-color is unresolved, do not invent a color.
6. Re-check price+stock after every cart change; then place_test_order.
7. Omit optional untruthful claims.

## Failure modes to avoid
- Jumping to other models before trying other colors of the same model.
- Skipping a commit when the preferred exact item was available.
- Relaxing budget/category/stock to force a match.
- Trusting a title that contradicts an explicit model fact.
""".strip(),

    "justification": """
# Family playbook — TF7 Justification
Lineage note: EnvPack claim discipline (no ShoppingBench primary TF7 donor).

## What this family tests
A valid purchase PLUS structured, fact-backed justification.claims on order.
Wrong claim shape zeros reward even when the product is correct.

## Strategy ladder
1. Find a valid in-stock in-budget product that meets firm gates.
2. view/compare (and inspect_stock) so every claim value is grounded BEFORE order.
3. Build a MINIMAL claim set, then place_test_order with justification.claims.

## Allowlisted claim fields
REQUIRED:
- price (current)
- in_stock (current)
- exactly ONE firm-spec field using the canonical name implied by the goal:
  resolution | screen_size | refresh_rate | ram
PLUS exactly ONE soft reason IF evidenced:
  brand | streaming | microphone | privacy_cover | autofocus | hdr
Prefer brand when present. NEVER invent hdr/streaming/etc. when unknown/absent.

## Forbidden fields / shapes
category, category_path, device_operating_system / os, title, summary, evidence,
reason, display, size, and marketing phrases used AS field names
(e.g. field="720p HD", field="Full HD display").

## Value rules
- Remap aliases before submit: display→resolution, size/screen→screen_size,
  stock/availability→in_stock.
- Spec values from viewed facts / normalized measurements, not title poetry.
- price/in_stock must match the latest inspect_stock or cart state after events.
- No prose blobs inside claims. Keep ~3 required + 1 soft.

## Post-event claims
If price/stock/spec evidence changes after cart, re-view/re-inspect and rebuild
claims from fresh facts before ordering.

## Failure modes to avoid
- field=display/size/"720p HD" → canonical faithfulness fails.
- Invented features.
- Stale price/stock vs claims.
- Correct-looking claims on the wrong gold product.
""".strip(),
}

_PROMPT_SUBTYPE: dict[str, str] = {
    "brand_first_possible": """
# Subtype playbook — brand_first_possible (TF1)

Signal: preferred brand can be fully firm-compliant.
Play:
1. Search brand + firm tokens first; keep a firm-only backup shortlist in parallel.
2. Prove the brand option on exact sku (spec + budget + stock + real device).
3. FIRST cart MUST be that brand SKU.
4. Message interventions with brand+spec+price evidence.
5. After events, re-apply the same rule under new constraints.
Trap: fixing brand later does not repair a non-brand first cart when brand was available.
""".strip(),

    "brand_impossible_soft": """
# Subtype playbook — brand_impossible_soft (TF1)

Signal: soft brand is subordinate; may have ZERO fully compliant SKUs.
Play:
1. Brief brand attempt (do not burn the whole budget).
2. As soon as brand options fail any firm gate, switch to best firm-compliant any-brand.
3. First cart = firm winner; do not stall.
4. Still answer interventions; still recover after events with firm walls intact.
Trap: endless brand hunting while a valid non-brand item was available.
""".strip(),

    "intervention_reply": """
# Subtype playbook — intervention_reply (TF1)

Signal: shopper may ask which/spec/budget mid-episode.
Play:
1. On user_message question/hold → message tool immediately.
2. Answer with concrete SKU identity + firm evidence + price/stock.
3. Then resume inspect → order.
4. Never cart/order in the same breath while ignoring the question.
Trap: correct final SKU but interventions_answered fails.
""".strip(),

    "post_event_budget_recover": """
# Subtype playbook — post_event_budget_recover (TF1)

Signal: after cart, price/stock may breach budget or vanish.
Play:
1. Detect invalid cart via inspect_stock/cart / market signals.
2. remove_from_cart the invalid item.
3. Re-search under the NEW budget/stock reality.
4. Brand preference still only if fully firm-compliant under the new world.
5. Close with a valid place_test_order (event must be finished).
Trap: leaving an over-budget item in cart or never re-ordering after recovery.
""".strip(),

    "phrase_title_match": """
# Subtype playbook — phrase_title_match (TF2)

Signal: goal requires a phrase to appear in the product title.
Play:
1. Keep distinctive phrase tokens intact in Pass-1 search (respect quotes).
2. On view, verify the phrase as complete title tokens (not stemmed inventions).
3. Reject hits that only match brand or category without the title phrase.
4. Then apply stock/budget and authenticity checks.
Trap: ordering a brand-right, phrase-wrong title.
""".strip(),

    "reformulate_thin_recall": """
# Subtype playbook — reformulate_thin_recall (TF2)

Signal: first search may be weak/noisy.
Play:
1. Evaluate first page quickly (noise vs signal).
2. Perform ONE small reformulation: drop filler, keep codes/units, or switch to
   category+filter.
3. Inspect several distinct identities from the improved shortlist.
4. Stop rewriting once a usable shortlist appears; proceed to order.
Trap: five near-identical rewrites that never change recall; or never ordering.
""".strip(),

    "multi_identity_shortlist": """
# Subtype playbook — multi_identity_shortlist (TF2)

Signal: must survey several distinct products before buying.
Play:
1. Collect multiple product_ids / model identities (not duplicate titles of one listing).
2. view/compare across the shortlist.
3. Deduplicate near-clones (same id, or same brand+model+price cluster).
4. Pick best in-stock in-budget after the survey.
Trap: counting raw rows as diversity when they are one identity repeated.
""".strip(),

    "category_trap": """
# Subtype playbook — category_trap (TF2)

Signal: catalog noise includes accessories and off-category items.
Play:
1. Always read category_path on view/compare.
2. Reject chargers, cases, protectors, cables, review pages when a full device
   (or the stated category) is required.
3. Confirm device-type/OS/title evidence for phones/laptops/etc.
4. Only then score stock/budget among authentic candidates.
Trap: high lexical recall on an accessory that fails final_in_gold.
""".strip(),

    "hidden_spec_ask_first": """
# Subtype playbook — hidden_spec_ask_first (TF3)

Signal: firm hardware spec is revealed only when asked.
Play:
1. Turn 1 MUST be message requesting the exact gated requirement.
2. Wait for user_message reply; do not search/filter yet.
3. After the value arrives, search with that value + other firm tokens.
4. Verify the revealed gate on the exact option/sku before cart/order.
Trap: searching first and locking onto a SKU that misses the hidden gate.
""".strip(),

    "explicit_spec": """
# Subtype playbook — explicit_spec (TF3)

Signal: firm specs are already written in the goal.
Play:
1. Extract every firm gate immediately (no ask needed).
2. Search with value-side tokens for those gates.
3. view + inspect_stock the exact option; check ALL gates.
4. Order only after the matrix is green.
Trap: satisfying 3 of 4 firm gates and hoping.
""".strip(),

    "same_title_multi_sku": """
# Subtype playbook — same_title_multi_sku (TF3)

Signal: one title hides multiple options (good and bad).
Play:
1. view the product and read option/sku rows.
2. Select the compliant option explicitly.
3. inspect_stock on THAT sku (not a sibling).
4. Cart/order only the verified option.
Trap: title looks perfect; option row fails the firm gate.
""".strip(),

    "sellout_new_product_not_sibling": """
# Subtype playbook — sellout_new_product_not_sibling (TF3)

Signal: wanted variant may sell out after cart.
Play:
1. On sellout/invalid: remove if needed.
2. Search for a DIFFERENT product_id that still meets every firm constraint.
3. Do not "fix" via a sibling sku that drops a firm requirement.
4. Re-verify and order the new product.
Trap: within_product_trap / sibling trap after sellout.
""".strip(),

    "resolution_class": """
# Subtype playbook — resolution_class (TF3)

Signal: resolution names denote pixel classes per Task rules.
Play:
1. Read the rule thresholds (both pixel sides).
2. Map evidence to HD / Full HD / QHD / 4K only with those floors.
3. Treat qHD as NOT QHD.
4. If evidence conflicts or is incomplete, do not invent a class — inspect more
   or pick a candidate with clear numeric resolution.
Trap: marketing "HD" wording without numeric proof against the class floor.
""".strip(),

    "soft_axis_compare": """
# Subtype playbook — soft_axis_compare (TF4)

Signal: shopper states soft use-case preferences.
Play:
1. Identify the soft axes explicitly.
2. Shortlist ≥2 hard-compliant candidates.
3. compare/view on those axes with factual attributes.
4. Prefer the stronger soft match even if slightly dearer (still in budget).
5. First cart = that winner.
Trap: price-only ranking while soft axes are ignored.
""".strip(),

    "priority_clarify": """
# Subtype playbook — priority_clarify (TF4)

Signal: soft priorities are ambiguous or competing.
Play:
1. Send ONE short clarifying message about which soft axis matters most.
2. After the reply (or if none), decide with best available evidence.
3. Inspect and first-cart a real preference-fit item.
Trap: endless clarifying chatter without ever carting; or carting before clarify
when the goal clearly invites a priority ask.
""".strip(),

    "first_cart_preference_fit": """
# Subtype playbook — first_cart_preference_fit (TF4)

Signal: first cart is graded as the intended purchase.
Play:
1. Do the soft-axis compare BEFORE any add_to_cart.
2. First add_to_cart = the real preference-aligned choice.
3. Later upgrades do not repair a weak first commitment.
Trap: cheap placeholder first cart.
""".strip(),

    "reprice_change_listing": """
# Subtype playbook — reprice_change_listing (TF4)

Signal: after reprice, recovery may need a different listing.
Play:
1. inspect after the event; remove invalid cart item.
2. Choose another listing_id that still meets hard + soft goals.
3. Do not merely flip variants of the same listing if recovery requires change.
4. Re-verify and order.
Trap: clinging to the same listing when it no longer fits after reprice.
""".strip(),

    "named_candidates_closed_set": """
# Subtype playbook — named_candidates_closed_set (TF5)

Signal: candidates listed as product_id::sku in the goal.
Play:
1. Extract the full closed set.
2. view/compare those IDs directly (batch if allowed).
3. Avoid open-ended catalog search that burns steps.
4. Rank and first-cart within this set only (unless an event removes some).
Trap: ignoring named IDs and searching the open catalog.
""".strip(),

    "lex_tradeoff_axes": """
# Subtype playbook — lex_tradeoff_axes (TF5)

Signal: rank by stated lexicographic priority only.
Play:
1. Write the axis order exactly as stated in the goal.
2. Compare candidates axis-by-axis in that order.
3. Typical when stated: refresh (incl. overclock ads) → screen size → price.
4. Never invent axes; never pick a dominated option.
Trap: mixing in unstated axes (brand popularity, random features).
""".strip(),

    "post_event_rerank": """
# Subtype playbook — post_event_rerank (TF5)

Signal: after stock/price/cart events, #1 may change.
Play:
1. Re-inspect remaining named candidates.
2. Re-rank from scratch with the same lexicographic priority.
3. Cart the NEW #1 (not the pre-event pick if demoted).
4. inspect_stock → place_test_order(product_id, sku) only.
Trap: market_rerank_after_event=false because you kept the old #1.
""".strip(),

    "omit_claims_on_order": """
# Subtype playbook — omit_claims_on_order (TF5)

Signal: justification is optional / not required for ranking.
Play:
1. place_test_order with ONLY product_id and sku.
2. Strip any justification/claims the model tries to attach.
3. Do not "helpfully" add why-text on a ranking task.
Trap: correct ranked pick zeroed by untruthful optional claims.
""".strip(),

    "exact_model_color": """
# Subtype playbook — exact_model_color (TF6)

Signal: strongly prefer exact model + color.
Play:
1. Search/commit Level 0 (exact model+color) first when in stock & in budget.
2. Record the full match via view/inspect before climbing the ladder.
3. Only leave Level 0 after it becomes unavailable/invalid.
Trap: browsing other models while the exact preferred item was available.
""".strip(),

    "color_before_model_ladder": """
# Subtype playbook — color_before_model_ladder (TF6)

Signal: least-relaxation requires other colors before other models.
Play:
1. When Level 0 dies, search same model / other color (Level 1).
2. Verify stock+budget on the alternate color.
3. Only then consider other models (Level 2).
4. Never relax category/budget/in-stock while climbing.
Trap: skipping Level 1 and jumping to another model.
""".strip(),

    "model_fact_overrides_title": """
# Subtype playbook — model_fact_overrides_title (TF6)

Signal: explicit model facts beat conflicting marketing titles.
Play:
1. Prefer structured model fields / stated model facts.
2. If title disagrees, trust the model fact for matching/recovery.
3. Still verify stock/price on the concrete sku you cart.
Trap: letting a flashy title override the shopper's explicit model requirement.
""".strip(),

    "never_relax_budget_stock_category": """
# Subtype playbook — never_relax_budget_stock_category (TF6)

Signal: category, budget, in-stock are immutable.
Play:
1. Every recovery candidate must still satisfy all three.
2. If a candidate breaks any wall, discard it and keep climbing legally.
3. Re-check walls after every cart/event.
Trap: "close enough" over-budget or wrong-category recoveries.
""".strip(),

    "required_claims": """
# Subtype playbook — required_claims (TF7)

Signal: order must include structured justification.claims.
Play:
1. Ground facts with view/inspect_stock first.
2. Build minimal allowlisted claims.
3. Attach them on place_test_order.
4. After events, refresh values before submit.
Trap: ordering with empty/missing claims, or prose instead of field/value pairs.
""".strip(),

    "canonical_gate_attr": """
# Subtype playbook — canonical_gate_attr (TF7)

Signal: claim the firm-spec attribute the shopper actually asked for.
Play:
1. Detect which gate the goal cares about: resolution / screen_size /
   refresh_rate / ram.
2. Use that canonical field name (not an alias).
3. Put the measured/normalized value in value.
Trap: claiming a different spec than the gated one.
""".strip(),

    "alias_remap": """
# Subtype playbook — alias_remap (TF7)

Signal: models often emit display/size/marketing field names.
Play:
1. Remap display→resolution, size/screen→screen_size, stock→in_stock.
2. If the field is a marketing phrase (e.g. "720p HD"), move content into value
   and set field to the canonical gate attr.
3. Drop fields that cannot be remapped safely.
Trap: submitting field="display" or field="720p HD".
""".strip(),

    "forbid_category_title_display": """
# Subtype playbook — forbid_category_title_display (TF7)

Signal: several fields are hard-forbidden.
Play:
1. Never claim category / category_path / title / summary / evidence / display /
   size / OS.
2. Ground every surviving claim in view/inspect_stock/cart state.
3. Prefer fewer true claims over more decorative ones.
Trap: "explaining" with title/category fields that fail faithfulness.
""".strip(),

    "one_soft_reason": """
# Subtype playbook — one_soft_reason (TF7)

Signal: exactly one soft reason if evidenced.
Play:
1. Prefer brand when brand is visible and relevant.
2. Otherwise one of streaming/microphone/privacy_cover/autofocus/hdr — only if
   explicitly true in viewed facts AND aligned with the goal.
3. Do not stack multiple soft feature reasons.
4. Never invent hdr/streaming when unknown.
Trap: multiple soft reasons or invented feature flags.
""".strip(),
}





_CASEBOOK_SHARED = """
# Shared edge-case casebook (all families)

## A. First-cart irrevocability
Several families score the FIRST add_to_cart separately from later fixes.
Treat first cart as the intended commitment. No sacrificial placeholders.
If you are unsure, inspect one more fact — do not cart garbage "for now".

## B. Intervention protocol
On shopper user_message that is a question, hold, or clarification:
1) message tool this turn (or immediately next)
2) concrete answer with identity + key evidence
3) only then continue cart/order
Ignoring interventions fails interventions_answered even with a gold final SKU.

## C. Market-event protocol
When stock_changed / price_changed / market_event / remove-driven recovery:
1) inspect_stock / inspect_cart on the affected SKU
2) if invalid: remove_from_cart
3) recover per family ladder (brand / re-rank / least-relax / new product)
4) place_test_order to close the event
Unfinished events are a common silent failure.

## D. Optional justification landmine
If the family does not require a reason, omit justification entirely.
Untruthful optional claims can zero paid_reward even when the product is gold.
Ranking episodes are especially sensitive.

## E. Device authenticity
When a full device is required, reject:
- screen-only modules
- accessories (cases, chargers, protectors, cables) unless that is the goal
- review / gift-card / mystery listings
- wrong form-factor substitutions
Use category_path + title/device/OS evidence together.

## F. Budget discipline
Always re-check price after cart events. Over-budget finals fail even if the
attribute match is perfect. Currency parsing: compare numeric magnitudes after
normalizing separators; do not ignore tax/fee fields if tools expose them as the
payable price.

## G. Evidence hierarchy
Highest trust → lowest trust:
inspect_stock/cart structured fields → view option/sku rows → compare tables →
product structured attrs → title tokens → marketing blurb.
Never let a lower-trust source override a higher-trust contradiction.

## H. Parallelism
When max_calls_per_turn allows, batch independent views of shortlist candidates
in one turn. Do not serialize obvious independent inspections.

## I. Finish discipline
Once a verified in-stock in-budget pick exists and no shopper question is pending,
place_test_order. Do not stall until max_steps "just in case".
""".strip()

_CASEBOOK_BY_FAMILY: dict[str, str] = {
    "intent_decomposition": """
# TF1 casebook — Intent Decomposition

## Case grid
1) Brand available + fully compliant → first cart MUST be that brand.
2) Brand available but fails firm spec → first cart = best other firm-compliant.
3) Brand string present but zero catalog hits → impossible soft; proceed with firm.
4) Shopper asks mid-episode "which one did you pick?" → message with brand+spec+price+stock.
5) Price jumps over budget after cart → remove, re-find firm-compliant under new budget;
   brand again only if still fully compliant.
6) Soft brand becomes available later after a bad first cart → too late for first-cart
   brand score; still finish correctly for final validity.
7) Accessory brand hit → reject via category_path; keep searching for real device.
8) Two brand SKUs, only one meets RAM/screen → cart the compliant brand SKU, not the other.

## Micro-checklist before first cart
- [ ] firm category / authenticity ok
- [ ] firm hardware gate ok on exact option
- [ ] budget ok (current price)
- [ ] in stock
- [ ] preferred brand honored IFF all above hold for that brand
- [ ] no pending unanswered shopper question

## Micro-checklist before final order
- [ ] cart item still valid after any event
- [ ] interventions answered
- [ ] no optional untruthful claims attached
""".strip(),

    "retrieval_recall": """
# TF2 casebook — Retrieval Recall

## Case grid
1) Exact phrase in title required → keep phrase tokens; verify on view.
2) Thin first page → reformulate once (tokens / category+filter).
3) Many near-duplicates → count distinct identities, not raw row count.
4) Accessory pollution → drop via category_path.
5) Good shortlist found → order without waiting for permission.
6) Hyphen/space variants → try both if first form misses.
7) Budget filter available → use it when open search returns mostly over-budget noise.
8) Shopper clarifies mid-search → message, then adjust keywords.

## Micro-checklist before order
- [ ] ≥1 reformulation OR strong first hit covering distinctive tokens
- [ ] several distinct identities inspected (when shortlist is required)
- [ ] chosen title/category match the goal (phrase/authenticity)
- [ ] in stock + in budget
- [ ] not an accessory trap
""".strip(),

    "constraint_satisfaction": """
# TF3 casebook — Constraint Satisfaction

## Case grid
1) Hidden firm spec → message first; block search until answered.
2) Explicit multi-constraint → every constraint on exact SKU.
3) Same title, mixed options → view options; pick compliant sku; inspect_stock.
4) Sellout → new product_id, not non-compliant sibling.
5) Resolution class → apply Task-rules thresholds; qHD ≠ QHD.
6) Refresh/RAM gates → verify on option row, not marketing adjectives.
7) After ask, shopper gives the gated value → immediately encode it into search tokens.
8) Event after cart → re-verify ALL firm gates before final order.

## Micro-checklist before order
- [ ] asked hidden spec if required
- [ ] exact sku option matches every firm constraint
- [ ] price + stock verified after any event
- [ ] no within-product / sibling trap after sellout
- [ ] class thresholds applied correctly when relevant
""".strip(),

    "preference_reasoning": """
# TF4 casebook — Preference Reasoning

## Case grid
1) Soft axes listed → compare ≥2 listing_ids on those axes.
2) Ambiguous priority → one clarifying message, then decide.
3) First cart must be preference-fit (not cheapest stub).
4) Reprice → recover on a different listing when needed.
5) Optional claims present by model mistake → strip them.
6) Soft axis only weakly evidenced → prefer the candidate with clearer factual support.
7) Hard constraint failure on a soft favorite → discard; soft cannot override hard.
8) Two soft favorites tie → use secondary soft axis then price, still in budget.

## Micro-checklist before first cart
- [ ] hard constraints satisfied
- [ ] soft axes compared on real facts (≥2 candidates when possible)
- [ ] first cart is the intended preference pick
- [ ] no pending clarification that you chose to ask but did not wait for
""".strip(),

    "ranking": """
# TF5 casebook — Ranking

## Case grid
1) Closed candidate set → view all named product_id::sku.
2) Lexicographic axes → rank only by stated priority.
3) Dominated option → never select.
4) Post-event → re-rank survivors; cart new #1; order.
5) Claims on order → omit unless required.
6) One candidate goes out of stock → rank remaining only; do not invent outsiders.
7) Overclock/advertised refresh mentioned in axes → use the advertised value the
   goal says to trust when rules allow.
8) Late turns → force finish; do not re-open search.

## Micro-checklist before order
- [ ] all named candidates considered (or remaining after event)
- [ ] current #1 is in cart
- [ ] dominance check passed
- [ ] place_test_order has only product_id+sku (usual case)
""".strip(),

    "recovery": """
# TF6 casebook — Recovery

## Case grid
1) Exact model+color available → commit first (Level 0).
2) Exact disappears → same model other color next (Level 1).
3) No other color → other models last (Level 2).
4) Model fact vs title conflict → trust model fact.
5) Never relax category/budget/stock.
6) Multi-color unresolved → do not invent; gather option evidence.
7) Price breaks budget after cart → treat as Level invalid; climb legally.
8) Optional claims → omit unless required; never invent features during recovery.

## Micro-checklist before order
- [ ] least-relaxation level chosen correctly (no skipped levels)
- [ ] category/budget/stock still hold
- [ ] facts grounded in view/inspect
- [ ] color source trusted in the evidence hierarchy
""".strip(),

    "justification": """
# TF7 casebook — Justification

## Case grid
1) Required claims → allowlisted fields only.
2) Gate attr from goal → resolution/screen_size/refresh_rate/ram.
3) Alias field names → remap or drop.
4) Soft reason → exactly one evidenced reason (prefer brand).
5) Post-event → re-verify facts then re-justify.
6) Forbidden fields appear → strip before submit.
7) Spec phrase used as field → move to value; set canonical field.
8) Missing price/in_stock → pull from latest inspect_stock before order.

## Micro-checklist before order
- [ ] viewed/inspect facts for every claim value
- [ ] no forbidden fields
- [ ] price + in_stock + canonical gate attr present
- [ ] at most one soft reason, evidenced
- [ ] aliases remapped
""".strip(),
}





_RECIPE_SEARCH_REFORM = """
# Recipe — search reformulation (TF2 / thin recall)

Pass 1 — Phrase / distinctive tokens
- Product-type noun + brand/model fragments + quoted phrases intact.
- Do NOT dump schema keys (brand/category/price) as keywords.
- Keep uncommon tokens (codes, materials, exact color words).

Pass 2 — Token cleanup (one small edit)
- Drop filler adjectives and shopper hedging words.
- Keep units and codes; normalize size words only when helpful
  (xl → extra large).
- Try hyphen↔space variants once.

Pass 3 — Category + filter
- Broaden product-type token slightly.
- Apply budget/stock/category filters if tools allow.
- Inspect several distinct hits before committing.

Stop rule: once a usable shortlist appears, stop rewriting and move to
view/compare/order. Reformulation is a scalpel, not a loop.
""".strip()

_RECIPE_RECOVERY_LADDER = """
# Recipe — least-relaxation ladder (TF6)

Immutable walls: category, budget, in-stock.

Level 0 — Exact model + exact color
- Commit when available; view/inspect to lock facts.

Level 1 — Exact model + other color
- Only after Level 0 unavailable/invalid.
- Resolve color via option → product color → title last.
- Re-verify stock and price.

Level 2 — Other models
- Only after Level 1 exhausted.
- Still inside immutable walls.

Model-fact rule: explicit model fact > conflicting marketing title.
Do not invent colors when multi-color evidence is unresolved.
After every climb: inspect → cart → inspect → place_test_order.
""".strip()

_RECIPE_RANK_CYCLE = """
# Recipe — ranking cycle (TF5)

1. Parse named product_id::sku candidates (closed set).
2. Batch view/compare all of them.
3. Rank by stated lexicographic axes only; drop dominated options.
4. add_to_cart current #1 (first cart graded).
5. On market/cart event:
   a) inspect survivors still available
   b) re-rank from scratch (same axes)
   c) cart NEW #1
   d) inspect_stock
   e) place_test_order(product_id, sku) with NO justification
6. If turn budget is high (≥8) and not ordered: force finish path now.
7. Never open-search the wider catalog unless named IDs are impossible to view.
""".strip()

_RECIPE_HIDDEN_SPEC = """
# Recipe — hidden firm spec (TF3)

Turn 1: message asking for the withheld firm hardware requirement
(refresh rate / RAM / screen / resolution / … — whatever the goal withholds).
Wait for shopper user_message.
Only then: search/filter using the revealed value + other firm tokens.
Verify the exact SKU option against ALL firm constraints (matrix).
If sellout: new product_id that still satisfies the gated spec — no sibling trap.
Keep turns for post-event re-verify + order.
""".strip()

_RECIPE_BRAND_FIRST = """
# Recipe — preferred-brand first cart (TF1)

1. Extract firm constraints + preferred brand soft.
2. Dual retrieve: (brand+firm) and (firm-only backup).
3. Prove compliance on exact options via view/inspect.
4. If a brand SKU is fully firm-compliant → first cart that SKU.
5. Else → first cart best firm-compliant non-brand item.
6. On intervention: message with evidence, then continue.
7. On budget/stock event: remove → recover under new constraints with same brand rule.
8. Omit optional claims unless required.
""".strip()

_RECIPE_CLAIMS = """
# Recipe — justification claims (TF7)

After view/inspect_stock on the final SKU, build claims:
1. canonical firm-spec field from the goal
   (resolution | screen_size | refresh_rate | ram)
2. price (current, post-event)
3. in_stock (current, post-event)
4. exactly one soft reason (prefer brand) if evidenced

Remap aliases before submit (display→resolution, size→screen_size, …).
Drop forbidden fields (category/title/display/size/OS/summary/evidence).
Never invent features. Prefer minimal true claims over decorative ones.
""".strip()

_RECIPE_PREFERENCE = """
# Recipe — preference / soft-axis compare (TF4)

1. Extract firm constraints + soft use-case axes from the goal.
2. Shortlist several firm-compliant SKUs; do not default to cheapest.
3. Optionally one clarifying message if priorities conflict.
4. compare/view against soft axes (fit, features, form, environment) before first cart.
5. First cart = preference-fit winner under firm constraints.
6. On reprice / listing change: re-evaluate survivors; change listing_id when needed.
7. Re-verify stock/price; place_test_order without optional untruthful claims.
""".strip()

_RECIPES_BY_FAMILY: dict[str, str] = {
    "intent_decomposition": _RECIPE_BRAND_FIRST,
    "retrieval_recall": _RECIPE_SEARCH_REFORM,
    "constraint_satisfaction": _RECIPE_HIDDEN_SPEC,
    "preference_reasoning": _RECIPE_PREFERENCE,
    "ranking": _RECIPE_RANK_CYCLE,
    "recovery": _RECIPE_RECOVERY_LADDER,
    "justification": _RECIPE_CLAIMS,
}





_EMPTY_TOOL_NUDGES = (
    "SnowGirl: no tool call received. You must call a supplied tool now. "
    "If you have no shortlist yet, start with search or filter using value-side "
    "keywords from the shopper goal (not JSON field names).",
    "SnowGirl: still no tool call. Advance the episode deliberately: "
    "search/filter → view/compare concrete SKUs → message if the shopper asked "
    "anything → add_to_cart only a verified pick → place_test_order once stock "
    "and budget are confirmed. Do not answer in plain text only.",
)

_HINT_ASK_SPEC = (
    "SnowGirl coach · TF3 hidden spec: a firm hardware requirement is withheld "
    "until you ask. Call message THIS turn requesting that exact requirement. "
    "Do not search/filter until the shopper replies with the gated value."
)

_HINT_REPLY = (
    "SnowGirl coach · intervention: the shopper just spoke. Call message THIS "
    "turn with a concrete answer (identity + key firm evidence + price/stock). "
    "Do not cart or order until that reply is sent."
)

_HINT_RECOVERY = (
    "SnowGirl coach · TF6 recovery: failure or market shift ({error}). Climb the "
    "least-relaxation ladder without skipping levels — exact model+color → same "
    "model other color → other models last. Never relax category, budget, or "
    "in-stock. Re-inspect after each cart change."
)

_HINT_RANK_OPEN = (
    "SnowGirl coach · TF5 open: extract every named product_id::sku, batch "
    "view/compare the closed set, rank ONLY on the stated lexicographic axes, "
    "then first-cart the current #1. place_test_order takes product_id+sku only "
    "— omit justification/claims."
)

_HINT_RANK_RERANK = (
    "SnowGirl coach · TF5 re-rank: catalog/cart state moved. Re-inspect surviving "
    "named candidates, rebuild the ranking from scratch on the same axes, cart "
    "the NEW #1 (not the pre-event winner if demoted), inspect_stock, then "
    "place_test_order(product_id, sku) only."
)

_HINT_RANK_FINISH = (
    "SnowGirl coach · TF5 finish: stop open search now. Re-rank in-stock named "
    "candidates, ensure the current #1 is carted, and place_test_order "
    "immediately with product_id+sku only."
)

_HINT_CLAIMS = (
    "SnowGirl coach · TF7 claims: justification.claims allowlist only — "
    "(1) canonical gate attr resolution|screen_size|refresh_rate|ram from the "
    "goal, (2) price, (3) in_stock, (4) at most one evidenced soft reason "
    "(prefer brand; streaming/microphone/privacy_cover/autofocus/hdr only when "
    "explicitly true). Forbidden: category, title, display, size, OS, summary, "
    "evidence. Remap aliases. Values from view/inspect_stock only."
)

_HINT_BRAND = (
    "SnowGirl coach · TF1 brand/first-cart: maintain a brand+firm shortlist and "
    "a firm-only backup. If a preferred-brand SKU is fully firm-compliant, it "
    "MUST win the first cart; otherwise cart the best firm-compliant any-brand "
    "item. Answer shopper questions with message before further cart/order."
)

_HINT_PREF = (
    "SnowGirl coach · TF4 preference: compare ≥2 hard-compliant candidates on "
    "the stated soft use-case axes before any add_to_cart. First cart = the "
    "preference-fit winner (may be slightly dearer if still in budget), never a "
    "cheap placeholder."
)

_HINT_RECALL = (
    "SnowGirl coach · TF2 recall: run ONE small reformulation (token cleanup or "
    "category+filter) so several distinct product identities appear, verify "
    "title/category authenticity, then proceed to order — do not keep rewriting."
)

_HINT_MARKET = (
    "SnowGirl coach · market ({kind}): re-check price/stock on the carted SKU "
    "via inspect_stock/cart. If invalid, remove_from_cart and recover per the "
    "active family ladder, then finish with a valid place_test_order so the "
    "event is closed."
)





_SG_FIELD_REMAP = (
    ("display", "resolution"),
    ("resolution class", "resolution"),
    ("pixel class", "resolution"),
    ("screen", "screen_size"),
    ("size", "screen_size"),
    ("screen size", "screen_size"),
    ("stock", "in_stock"),
    ("availability", "in_stock"),
)
_SG_FIELD_REMAP_MAP = dict(_SG_FIELD_REMAP)

_SG_CLAIM_DENY = frozenset(
    {
        "category",
        "category_path",
        "device_operating_system",
        "device os",
        "os",
        "title",
        "summary",
        "evidence",
        "reason",
        "display",
        "size",
    }
)

_SG_CLAIM_ALLOW = frozenset(
    {
        "resolution",
        "screen_size",
        "refresh_rate",
        "ram",
        "brand",
        "price",
        "in_stock",
        "streaming",
        "microphone",
        "privacy_cover",
        "autofocus",
        "hdr",
    }
)

_SG_SOFT_FEATURES = frozenset(
    {"streaming", "microphone", "privacy_cover", "autofocus", "hdr"}
)
_SG_SOFT_ALL = frozenset({"brand"}) | _SG_SOFT_FEATURES

_SG_SPEC_AS_FIELD = re.compile(
    r"(?i)^(?:\d{3,4}\s*[pP](?:\s*HD)?|full\s*hd|qhd|uhd|4k|"
    r"\d+(?:\.\d+)?\s*-?\s*inch(?:es)?)$"
)

_SG_FAMILY_PRIORITY = (
    (FAM_RANK, 7),
    (FAM_RECOVER, 6),
    (FAM_JUSTIFY, 5),
    (FAM_CONSTRAINT, 4),
    (FAM_INTENT, 3),
    (FAM_PREFERENCE, 2),
    (FAM_RECALL, 1),
)





def _shopper_goal_text(query: str) -> str:
    text = query or ""
    marker = "Task rules:"
    cut = text.find(marker)
    return text if cut < 0 else text[:cut]

def _classify_family(query: str) -> str:
    blob = query or ""
    tally: dict[str, int] = {}
    for family_id, patterns in _FAMILY_FINGERPRINTS:
        n = 0
        for rx in patterns:
            if rx.search(blob):
                n += 1
        if n:
            tally[family_id] = n

    low = blob.lower()
    if "justification is optional" in low and FAM_JUSTIFY in tally:
        head = _shopper_goal_text(blob)
        if not re.search(r"\bwhy\b|reason|justify", head, re.I):
            tally.pop(FAM_JUSTIFY, None)

    if not tally:
        return FAM_RECALL

    rank = dict(_SG_FAMILY_PRIORITY)
    winner = max(tally.items(), key=lambda item: (item[1], rank.get(item[0], 0)))
    return winner[0]

def _classify_subtype(family: str, query: str) -> str:
    lowered = (query or "").lower()
    rules = lowered
    if "task rules:" in lowered:
        rules = lowered.split("task rules:", 1)[1]

    pick = _SUBTYPE_DEFAULT.get(family, "default")
    top = 0
    head = _shopper_goal_text(query)
    for subtype_id, cues in _SUBTYPE_CUES.get(family) or []:
        hits = 0
        for cue in cues:
            if cue in rules or cue in lowered:
                hits += 1
        if subtype_id == "hidden_spec_ask_first" and _ASK_SPEC_RX.search(head):
            hits += 3
        if subtype_id == "named_candidates_closed_set" and _VARIANT_ID_RX.search(
            query or ""
        ):
            hits += 2
        if hits > top:
            top = hits
            pick = subtype_id
    return pick

def _episode_tags(family: str, subtype: str, query: str) -> dict[str, bool]:
    head = _shopper_goal_text(query)
    justify_goal = bool(
        re.search(r"\bwhy\b|trustworthy reason|structured justification", head, re.I)
    )
    non_justify = family != FAM_JUSTIFY
    return {
        "needs_ask_spec": family == FAM_CONSTRAINT
        and (
            subtype == "hidden_spec_ask_first" or bool(_ASK_SPEC_RX.search(head))
        ),
        "has_named_variants": bool(_VARIANT_ID_RX.search(query or "")),
        "needs_justification": family == FAM_JUSTIFY or justify_goal,
        "omit_optional_claims": non_justify,
    }

def _goal_needs_ask(query: str) -> bool:
    return bool(_ASK_SPEC_RX.search(_shopper_goal_text(query)))

def _extract_named_skus(query: str) -> list[str]:
    found = _VARIANT_ID_RX.findall(query or "")
    ordered: list[str] = []
    seen: set[str] = set()
    for token in found:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered

def _resolve_openrouter_id(raw: str) -> str:
    key = (raw or "").strip()
    if not key:
        return _SG_MODEL_HEAVY
    for chutes_id, openrouter_id in _SG_CHUTES_PAIRS:
        if key == chutes_id or key.endswith(chutes_id.split("/", 1)[-1]):
            return openrouter_id
    return key

def _pick_model(family: str, subtype: str) -> str:
    forced = (getenv("SANDBOX_MODEL") or "").strip()
    if forced:
        return _resolve_openrouter_id(forced)
    light_ok = family == FAM_RECALL and subtype in {
        "multi_identity_shortlist",
        "phrase_title_match",
        "reformulate_thin_recall",
        "category_trap",
    }
    return _SG_MODEL_LIGHT if light_ok else _SG_MODEL_HEAVY

def _gate_attr_from_goal(query: str) -> str | None:
    head = _shopper_goal_text(query).lower()
    checks = (
        (r"\b\d+(?:\.\d+)?\s*-?\s*inch|\bscreen size\b", "screen_size"),
        (r"\b(?:\d{3,4}\s*p|720p|1080p|full\s*hd|qhd|uhd|4k|resolution)\b", "resolution"),
        (r"\brefresh\b|\bhz\b", "refresh_rate"),
        (r"\bram\b|\bgb\b", "ram"),
    )
    for pattern, field in checks:
        if re.search(pattern, head):
            return field
    return None

def _canon_claim_field(field: str, query: str) -> str:
    raw = str(field or "").strip()
    key = raw.lower()
    if key in _SG_CLAIM_DENY and key not in _SG_FIELD_REMAP_MAP:
        return ""
    remapped = _SG_FIELD_REMAP_MAP.get(key)
    if remapped:
        return remapped
    if _SG_SPEC_AS_FIELD.match(raw):
        return _gate_attr_from_goal(query) or "resolution"
    return raw

def _canon_claim_value(field: str, value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if field == "screen_size":
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        return f"{m.group(1)} inches" if m else text
    if field == "resolution":
        compact = text.lower().replace(" ", "")
        if compact in {"720p", "720phd", "hd"} or ("1280" in compact and "720" in compact):
            return "hd"
        if compact in {"1080p", "1080pfullhd", "fullhd", "fhd"} or (
            "1920" in compact and "1080" in compact
        ):
            if "1920" in text and "1080" in text:
                return re.sub(r"\s+", " ", text)
            return "1920 x 1080"
        if compact in {"1440p", "qhd"} or ("2560" in compact and "1440" in compact):
            if "2560" in text and "1440" in text:
                return re.sub(r"\s+", " ", text)
            return "2560 x 1440"
    if field == "in_stock":
        return "true" if text.lower() in {"true", "1", "yes", "in_stock"} else text
    return text

def _scrub_order_args(
    tool_name: str,
    args: dict[str, Any],
    *,
    family: str,
    query: str,
    tags: dict[str, bool],
) -> dict[str, Any]:
    payload = dict(args)
    if tool_name != "place_test_order":
        return payload

    head = _shopper_goal_text(query)
    wants_why = bool(re.search(r"\bwhy\b|justif|reason|tell me why", head, re.I))

    if family == FAM_RANK and not wants_why:
        payload.pop("justification", None)
        return payload

    if family != FAM_JUSTIFY:
        if tags.get("omit_optional_claims") and not wants_why:
            payload.pop("justification", None)
        return payload

    block = payload.get("justification")
    if not isinstance(block, dict):
        return payload
    raw_claims = block.get("claims")
    if not isinstance(raw_claims, list):
        return payload

    kept: list[dict[str, str]] = []
    used: set[str] = set()
    soft_used = False
    head_l = head.lower()
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        field = _canon_claim_field(str(item.get("field") or ""), query)
        value = _canon_claim_value(field, item.get("value"))
        if not field or value == "":
            continue
        fl = field.lower()
        if fl in _SG_CLAIM_DENY or fl not in _SG_CLAIM_ALLOW:
            continue
        if fl in _SG_SOFT_FEATURES and fl not in head_l:
            continue
        if fl in _SG_SOFT_ALL:
            if soft_used:
                continue
            soft_used = True
        if fl in used:
            continue
        used.add(fl)
        kept.append({"field": field, "value": value})

    payload["justification"] = {"claims": kept}
    return payload

def _shopper_needs_reply(text: str) -> bool:
    blob = (text or "").strip().lower()
    if not blob:
        return False
    if "?" in blob:
        return True

    cues = (
        r"^\s*which\b",
        r"^\s*what\b",
        r"^\s*does\b",
        r"^\s*did\b",
        r"\bcan you\b",
        r"\bcould you\b",
        r"^\s*wait\b",
        r"\bhold on\b",
        r"\bbefore you\b",
        r"\bre-?check\b",
        r"\bgot anything\b",
    )
    return any(re.search(pat, blob) for pat in cues)

def _read_market_kind(obs: Any) -> str:
    if not isinstance(obs, dict):
        return ""
    if obs.get("stock_changed"):
        return "stock_changed"
    if obs.get("price_changed"):
        return "price_changed"
    if obs.get("event"):
        return str(obs.get("event"))
    for key in ("market_event",):
        node = obs.get(key)
        if isinstance(node, dict):
            return str(node.get("kind") or key)
        if node:
            return key
    nested = obs.get("observation")
    if isinstance(nested, dict):
        node = nested.get("market_event")
        if isinstance(node, dict):
            return str(node.get("kind") or "market_event")
        if node:
            return "market_event"
    return ""

def _list_tool_names(tools: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for entry in tools:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            out.append(str(fn["name"]))
        elif entry.get("name"):
            out.append(str(entry["name"]))
    return out

def _compose_system(
    family: str,
    subtype: str,
    policy: dict[str, Any],
    query: str,
    tags: dict[str, bool],
) -> str:
    tool_names = _list_tool_names(policy.get("tools") or [])
    fam_text = _PROMPT_FAMILY.get(family, _PROMPT_FAMILY[FAM_RECALL])
    sub_text = _PROMPT_SUBTYPE.get(subtype, "")
    named = _extract_named_skus(query)
    named_block = ""
    if family == FAM_RANK and named:
        named_block = (
            "Closed candidate set (inspect each; re-rank after events):\n"
            + ", ".join(named)
            + "\n"
        )
    tag_bits = (
        f"needs_ask_spec={tags.get('needs_ask_spec')} "
        f"has_named_variants={tags.get('has_named_variants')} "
        f"needs_justification={tags.get('needs_justification')} "
        f"omit_optional_claims={tags.get('omit_optional_claims')}"
    )
    parts = [
        _PROMPT_CORE,
        fam_text,
        sub_text,
        _CASEBOOK_SHARED,
        _CASEBOOK_BY_FAMILY.get(family, ""),
        _RECIPES_BY_FAMILY.get(family, ""),
        named_block,
        "# Runtime",
        f"Detected task family: {family}",
        f"Detected subtype: {subtype}",
        f"Tags: {tag_bits}",
        f"Available tools: {', '.join(tool_names) if tool_names else '(see tool schemas)'}",
        f"max_steps={policy.get('max_steps')} "
        f"max_calls_per_turn={policy.get('max_calls_per_turn', 1)}",
    ]
    return "\n\n".join(p for p in parts if p)

def _parse_tool_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    fn = tool_call.get("function") if isinstance(tool_call, dict) else None
    raw: Any = None
    if isinstance(fn, dict):
        raw = fn.get("arguments")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        raise ValueError("tool arguments must be a JSON object")
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        parsed = {}
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("tool arguments must be a JSON object")

def _obs_to_text(obs: Any, *, cap: int = 6500) -> str:
    try:
        encoded = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        encoded = str(obs)
    if len(encoded) <= cap:
        return encoded
    return f"{encoded[: cap - 16]}...<truncated>"

def _names_from_calls(tool_calls: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for call in tool_calls:
        try:
            out.append(str(call["function"]["name"]))
        except (KeyError, TypeError, AttributeError):
            out.append("?")
    return out

def _fn_name(call: dict[str, Any]) -> str:
    try:
        return str(call["function"]["name"])
    except (KeyError, TypeError, AttributeError):
        return ""

def _fn_id(call: dict[str, Any], fallback: str) -> str:
    try:
        cid = call.get("id")
        return str(cid) if cid else fallback
    except (AttributeError, TypeError):
        return fallback

def _turn_coach(
    *,
    family: str,
    subtype: str,
    turn: int,
    last_tool_names: list[str],
    last_obs: list[Any],
    tags: dict[str, bool],
    await_reply: bool,
    saw_market_event: bool,
    ordered: bool,
    first_cart_done: bool,
    asked_hidden_spec: bool,
    search_count: int,
    viewed_named: bool,
) -> str | None:
    used = " ".join(last_tool_names)

    if await_reply:
        return _HINT_REPLY

    if (
        tags.get("needs_ask_spec")
        and turn == 1
        and not asked_hidden_spec
        and "message" not in used
    ):
        return _HINT_ASK_SPEC

    if family == FAM_RANK and not ordered:
        if saw_market_event and turn >= 2 and "place_test_order" not in used:
            return _HINT_RANK_RERANK
        if turn >= 8:
            return _HINT_RANK_FINISH
        if turn == 1:
            return _HINT_RANK_OPEN
        if turn >= 2 and not first_cart_done and (
            viewed_named or "view" in used or "compare" in used
        ):
            return (
                "SnowGirl coach · TF5: candidates inspected — add_to_cart only "
                "the current #1 (first cart is graded)."
            )
        if first_cart_done and turn >= 3:
            return (
                "SnowGirl coach · TF5: finish with place_test_order(product_id, "
                "sku) only — leave justification off the payload."
            )

    if family == FAM_INTENT and turn == 1:
        return _HINT_BRAND

    if (
        family == FAM_PREFERENCE
        and turn <= 2
        and "compare" not in used
        and not first_cart_done
    ):
        return _HINT_PREF

    if family == FAM_JUSTIFY and not ordered:
        if turn in (2, 5) or (
            turn >= 2 and "view" not in used and "compare" not in used
        ):
            return _HINT_CLAIMS

    bootstrap_tools = ("search", "filter", "view", "compare", "message")
    if turn <= 2 and not any(name in used for name in bootstrap_tools):
        return (
            "SnowGirl coach: open with search/filter on value-side keywords from "
            "the shopper goal, then view the strongest hits."
        )

    if family == FAM_RECALL and turn == 3 and search_count >= 1:
        return _HINT_RECALL

    for obs in last_obs:
        if isinstance(obs, dict) and obs.get("error") and family == FAM_RECOVER:
            return _HINT_RECOVERY.format(error=str(obs.get("error")))

    if saw_market_event and not ordered and turn >= 2:
        return _HINT_MARKET.format(kind="stock/price/cart")

    if (
        family == FAM_CONSTRAINT
        and subtype == "sellout_new_product_not_sibling"
        and saw_market_event
        and not ordered
    ):
        return (
            "SnowGirl coach · TF3 sellout: switch to a NEW product_id that still "
            "meets every firm constraint — sibling SKUs that drop a firm spec "
            "are invalid."
        )

    if (
        family == FAM_RECOVER
        and subtype == "color_before_model_ladder"
        and saw_market_event
    ):
        return (
            "SnowGirl coach · TF6 ladder: same model / other color before other "
            "models. Never relax category, budget, or in-stock."
        )

    return None

def _chat_with_tools(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_calls: int,
) -> tuple[str, list[dict[str, Any]]]:
    queue: list[str] = [model]
    for candidate in _SG_FALLBACK_CHAIN:
        if candidate not in queue:
            queue.append(candidate)

    failure: Exception | None = None
    for model_id in queue[:3]:
        payload = {
            "model": model_id,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "temperature": 0,
        }
        response = _http.post("/inference/chat/completions", json_data=payload)
        if response is None:
            failure = RuntimeError(f"empty inference response model={model_id}")
            continue
        try:
            msg = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            failure = exc
            continue
        content = msg.get("content") or ""
        calls = list(msg.get("tool_calls") or [])[:max_calls]
        return content, calls
    raise RuntimeError(f"SnowGirl inference exhausted fallbacks ({failure})")

def _opening_coaches(
    family: str, tags: dict[str, bool]
) -> list[dict[str, Any]]:
    extras: list[dict[str, Any]] = []
    if tags.get("needs_ask_spec"):
        extras.append({"role": "user", "content": _HINT_ASK_SPEC})
    if family == FAM_RANK:
        extras.append({"role": "user", "content": _HINT_RANK_OPEN})
    if family == FAM_JUSTIFY:
        extras.append({"role": "user", "content": _HINT_CLAIMS})
    if family == FAM_INTENT:
        extras.append({"role": "user", "content": _HINT_BRAND})
    return extras

def _note_named_view(args: dict[str, Any], named: set[str]) -> bool:
    if not named:
        return False
    pid = str(args.get("product_id") or "")
    sku = str(args.get("sku") or "")
    if pid and sku and f"{pid}::{sku}" in named:
        return True
    if pid and any(token.startswith(pid) for token in named):
        return True
    return bool(pid)

def _episode_done(result_calls: list[Any]) -> bool:
    for call in result_calls:
        if not isinstance(call, dict):
            continue
        obs = call.get("observation")
        if isinstance(obs, dict) and obs.get("done"):
            return True
    return False

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:

    env = problem_data["environment"]
    binding = env["binding"]
    policy = env["policy_view"]
    query = str(policy.get("query") or problem_data.get("query") or "")
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
    max_steps = max(1, int(policy.get("max_steps", 20)))
    tools = policy.get("tools") or []

    family = _classify_family(query)
    subtype = _classify_subtype(family, query)
    tags = _episode_tags(family, subtype, query)
    model = _pick_model(family, subtype)

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _compose_system(family, subtype, policy, query, tags),
        },
        {"role": "user", "content": query},
        *_opening_coaches(family, tags),
    ]

    transcript: list[dict[str, Any]] = [
        {
            "role": "meta",
            "snowgirl": {
                "version": 6,
                "family": family,
                "subtype": subtype,
                "tags": tags,
                "model": model,
                "provider": "openrouter",
                "refs": "third_rank_per_tf",
                "shell": "snowgirl_native",
            },
        }
    ]

    state = {
        "await_reply": False,
        "saw_market_event": False,
        "ordered": False,
        "first_cart_done": False,
        "asked_hidden_spec": False,
        "search_count": 0,
        "viewed_named": False,
        "last_tools": [],
        "last_obs": [],
    }
    named = set(_extract_named_skus(query))

    for turn in range(1, max_steps + 1):
        coach = _turn_coach(
            family=family,
            subtype=subtype,
            turn=turn,
            last_tool_names=state["last_tools"],
            last_obs=state["last_obs"],
            tags=tags,
            await_reply=state["await_reply"],
            saw_market_event=state["saw_market_event"],
            ordered=state["ordered"],
            first_cart_done=state["first_cart_done"],
            asked_hidden_spec=state["asked_hidden_spec"],
            search_count=state["search_count"],
            viewed_named=state["viewed_named"],
        )
        if coach:
            messages.append({"role": "user", "content": coach})

        content = ""
        tool_calls: list[dict[str, Any]] = []
        for attempt in range(len(_EMPTY_TOOL_NUDGES) + 1):
            content, tool_calls = _chat_with_tools(
                model=model,
                messages=messages,
                tools=tools,
                max_calls=max_calls,
            )
            if tool_calls:
                break
            transcript.append({"role": "assistant", "content": content})
            nudge = _EMPTY_TOOL_NUDGES[min(attempt, len(_EMPTY_TOOL_NUDGES) - 1)]
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": nudge})
        else:
            raise RuntimeError("SnowGirl: model never produced a tool call")

        messages.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls}
        )

        batch_id = f"{problem_id}-sg-{turn}"
        calls_out: list[dict[str, Any]] = []
        for idx, call in enumerate(tool_calls, start=1):
            name = _fn_name(call)
            if not name:
                raise RuntimeError("SnowGirl: tool call missing function name")
            args = _scrub_order_args(
                name,
                _parse_tool_args(call),
                family=family,
                query=query,
                tags=tags,
            )
            fn = call.get("function")
            if isinstance(fn, dict):
                fn["arguments"] = json.dumps(args)
            calls_out.append(
                {
                    "call_id": f"{batch_id}-{idx}",
                    "action": {"name": name, "args": args},
                }
            )
            if name in ("search", "filter"):
                state["search_count"] += 1
            elif name == "message":
                state["await_reply"] = False
                state["asked_hidden_spec"] = True
            elif name == "add_to_cart":
                state["first_cart_done"] = True
            elif name == "remove_from_cart":
                state["saw_market_event"] = True
            elif name == "place_test_order":
                state["ordered"] = True
            elif name in ("view", "compare"):
                if _note_named_view(args, named):
                    state["viewed_named"] = True

        env_body = {
            **binding,
            "call_id": batch_id,
            "idempotency_key": batch_id,
            "turn": turn,
            "calls": calls_out,
        }
        result = _http.post("/environment/call", json_data=env_body)
        if result is None:
            raise RuntimeError("SnowGirl: /environment/call returned empty")

        transcript.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
                "environment_result": result,
                "snowgirl_turn": {
                    "family": family,
                    "subtype": subtype,
                    "turn": turn,
                    "state": {
                        k: state[k]
                        for k in (
                            "await_reply",
                            "saw_market_event",
                            "ordered",
                            "first_cart_done",
                            "asked_hidden_spec",
                            "search_count",
                            "viewed_named",
                        )
                    },
                },
            }
        )

        state["last_tools"] = _names_from_calls(tool_calls)
        state["last_obs"] = []
        result_calls = result.get("calls") or []
        for idx, (tool_call, call_result) in enumerate(
            zip(tool_calls, result_calls), start=1
        ):
            obs = (
                call_result.get("observation")
                if isinstance(call_result, dict)
                else None
            )
            state["last_obs"].append(obs)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _fn_id(tool_call, f"{batch_id}-{idx}"),
                    "content": _obs_to_text(obs),
                }
            )
            kind = _read_market_kind(obs)
            if kind:
                state["saw_market_event"] = True
                if not state["ordered"]:
                    messages.append(
                        {
                            "role": "user",
                            "content": _HINT_MARKET.format(kind=kind),
                        }
                    )

        spoken_msg = result.get("user_message")
        if spoken_msg and spoken_msg.get("content"):
            spoken = str(spoken_msg["content"])
            messages.append({"role": "user", "content": spoken})
            if _shopper_needs_reply(spoken):
                state["await_reply"] = True

        errs = [
            obs.get("error")
            for obs in state["last_obs"]
            if isinstance(obs, dict) and obs.get("error")
        ]
        if errs and family == FAM_RECOVER:
            messages.append(
                {
                    "role": "user",
                    "content": _HINT_RECOVERY.format(error=errs[0]),
                }
            )
        marketish = bool(errs) or any(
            _read_market_kind(obs) for obs in state["last_obs"]
        )
        if (
            family == FAM_RANK
            and state["saw_market_event"]
            and not state["ordered"]
            and marketish
        ):
            messages.append({"role": "user", "content": _HINT_RANK_RERANK})

        if _episode_done(result_calls):
            break

    return transcript

__all__ = ["agent_main"]
