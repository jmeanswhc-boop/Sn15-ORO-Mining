from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# =============================================================================
# OpenRouter routing — Dragon desk lane
# =============================================================================

_DRAGON_LIGHT = "deepseek/deepseek-v3.2"
_DRAGON_HARD = "moonshotai/kimi-k2.6"
_DRAGON_FALLBACKS: list[str] = [
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "google/gemma-4-31b-it",
]

# Chutes TEE → OpenRouter id pairs (tuple list; Dragon naming).
_TEE_TO_OPENROUTER: tuple[tuple[str, str], ...] = (
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

_proxy = ProxyClient(timeout=120, max_retries=2)

# Family identifiers
F_INTENT = "intent_decomposition"
F_RECALL = "retrieval_recall"
F_CONSTRAINT = "constraint_satisfaction"
F_PREFERENCE = "preference_reasoning"
F_RANKING = "ranking"
F_RECOVERY = "recovery"
F_JUSTIFY = "justification"

_OBS_CAP = 6500

# =============================================================================
# Capability fingerprints — generic themes (not suite query copies)
# =============================================================================

_FINGERPRINTS: list[tuple[str, list[re.Pattern[str]]]] = [
    (
        F_RANKING,
        [
            re.compile(
                r"\btrade-?off\b.{0,36}\bpriorit|"
                r"\blexico(?:graphic)?\b|"
                r"\bstrictly (?:worse|dominated)\b|"
                r"\bbest[- ]ranked\b",
                re.I,
            ),
            re.compile(
                r"\bnamed candidates?\b|\bclosed[- ]set\b|"
                r"\bre[- ]?rank(?:ing|ed)?\b|\bdominat(?:e|es|ed|ion)\b|"
                r"\bcandidate variant",
                re.I,
            ),
            re.compile(r"\d{8,}::\S+"),
        ],
    ),
    (
        F_RECOVERY,
        [
            re.compile(r"\bleast[- ]relax|\brelaxable only\b", re.I),
            re.compile(
                r"\bsame model\b.{0,48}(?:\bother\b|\banother\b).{0,12}\bcolor\b",
                re.I,
            ),
            re.compile(
                r"\bexact\b.{0,20}(?:item|model|color).{0,60}\bdisappear|"
                r"\bnever relax\b.{0,48}(?:budget|stock|category)|"
                r"(?:\bother\b|\banother\b)\s+color\s+before.{0,24}"
                r"(?:\bother\b|\banother\b)\s+model|"
                r"\bmodel fact\b.{0,36}(?:title|overrid)",
                re.I,
            ),
        ],
    ),
    (
        F_JUSTIFY,
        [
            re.compile(
                r"\b(?:must|should)\b.{0,48}\bjustification\b|"
                r"\bclaims?\b.{0,24}\bjustif|"
                r"\btrustworthy reason\b|"
                r"\bground(?:ed)? claims?\b|"
                r"\bjustification\.claims\b",
                re.I,
            ),
            re.compile(
                r"\bwalk me through why\b|\btell me why\b|"
                r"\bgive me a reason\b|"
                r"\bwhy\b.{0,16}(?:the one|it'?s|your pick)",
                re.I,
            ),
        ],
    ),
    (
        F_INTENT,
        [
            re.compile(r"\bnice[- ]to[- ]have\b", re.I),
            re.compile(
                r"\bpreferred brand\b|\bprefer(?:red)? brand\b|"
                r"\bfully (?:firm[- ])?compliant\b|"
                r"\bsubordinate\b.{0,48}\b(?:firm|requirement)\b",
                re.I,
            ),
            re.compile(
                r"\bfirm\b.{0,96}\bsoft\b|\bsoft\b.{0,96}\bfirm\b|"
                r"\binitial\b.{0,12}(?:cart )?choice.{0,80}(?:soft|prefer|brand)",
                re.I,
            ),
            re.compile(
                r"\bbefore you order\b|\bwhich one\b.{0,24}(?:did|do|should)|"
                r"\bover budget\b.{0,48}(?:recover|re[- ]?search|remove)|"
                r"\bafter\b.{0,12}(?:cart|add).{0,72}(?:budget|unavailable|price)",
                re.I,
            ),
        ],
    ),
    (
        F_CONSTRAINT,
        [
            re.compile(r"\bexact(?:ly)?\b.{0,16}\bvarian|\bspecific sku\b", re.I),
            re.compile(
                r"\bevery requirement must\b|\bpixel class\b|\bresolution class\b",
                re.I,
            ),
            re.compile(
                r"\bfirm requirement\b.{0,24}\bspec\b|"
                r"\breveal(?:ed)? only when\b|"
                r"\btell you if you ask\b|\bif you ask\b|"
                r"\bhidden (?:firm )?spec\b|\bgated spec\b",
                re.I,
            ),
            re.compile(
                r"\bsell(?:s)? out\b|\bsibling\b|"
                r"\bsame title\b.{0,48}(?:sku|option)",
                re.I,
            ),
        ],
    ),
    (
        F_PREFERENCE,
        [
            re.compile(
                r"\bpreference axes\b|\bsoft axes\b|"
                r"\bblindly buying the cheapest\b|"
                r"\bnot (?:the )?cheapest\b|"
                r"\bpreference[- ]fit\b",
                re.I,
            ),
            re.compile(
                r"\bintended use\b|\buse[- ]case\b|"
                r"\brather pay\b|\bsweat\b|\bwater resistance\b|"
                r"\bloose hard request\b",
                re.I,
            ),
            re.compile(r"\bclarify\b.{0,36}\bpriorit", re.I),
        ],
    ),
    (
        F_RECALL,
        [
            re.compile(r"\breformulate\b|\blucky hit\b|\bshortlist\b", re.I),
            re.compile(
                r"\bsurvey the realistic\b|\bdistinct products\b|"
                r"\bmatches?\b.{0,24}(?:phrase )?\"[^\"]+\"",
                re.I,
            ),
            re.compile(
                r"\bcategory_path\b|\bnot an accessory\b|\boff[- ]category\b",
                re.I,
            ),
        ],
    ),
]

_ASK_GATE_RX = re.compile(
    r"(?:tell|reveal|share).{0,48}\bif you ask\b|"
    r"\bif you ask\b|"
    r"\bask\b.{0,24}(?:exact )?(?:requirement|spec)\b|"
    r"\bwithhold\b.{0,36}\bspec\b",
    re.I,
)

_NAMED_SKU_RX = re.compile(r"\b(\d{10,}::[A-Za-z0-9._-]+)\b")

_FAMILY_TIEBREAK: dict[str, int] = {
    F_RANKING: 7,
    F_RECOVERY: 6,
    F_JUSTIFY: 5,
    F_CONSTRAINT: 4,
    F_INTENT: 3,
    F_PREFERENCE: 2,
    F_RECALL: 1,
}

# Soft mode tags (lighter than a full subtype catalog)
MODE_ASK_SPEC = "ask_spec"
MODE_CLOSED_RANK = "closed_rank"
MODE_BRAND_SOFT = "brand_soft"
MODE_THIN_RECALL = "thin_recall"
MODE_SOFT_COMPARE = "soft_compare"
MODE_LEAST_RELAX = "least_relax"
MODE_CLAIM_BUILD = "claim_build"
MODE_INTERVENE = "intervene"
MODE_MARKET = "market_recover"
MODE_PHRASE = "phrase_title"
MODE_DEFAULT = "desk_default"

# =============================================================================
# CORE doctrine — Dragon Strategy Desk
# =============================================================================

CORE_DOCTRINE = """
# Dragon Strategy Desk — CORE doctrine

You are the Dragon Strategy Desk operator for an ORO Bench EnvPack shopping
episode. Tools are dynamic; you must use the supplied schemas only. Decisions
come from the public goal, Task rules, and live observations — never from
memorized product ids or suite answers.

## Contract
- Satisfy the shopper under firm walls (budget, stock, category / real device,
  explicit specs) while optimizing soft preferences where they exist.
- First `add_to_cart` is often graded — never park a cheap placeholder.
- `place_test_order` closes the episode. Attach `justification.claims` ONLY when
  the family / goal truly requires a why; otherwise omit claims entirely.
- Answer shopper questions with the `message` tool before cart/order progress.

## Canonical loop
1. Parse goal → tag every constraint FIRM or SOFT.
2. Discover → search / filter with VALUE-SIDE keywords (never dump schema key
   names like "brand", "category", "price", "in_stock" as search tokens).
3. Inspect → `view` / `compare` / `inspect_stock` on concrete product_id + sku.
4. Decide → cart the verified pick. First cart must be intentional.
5. Dialogue → if the shopper asks anything, `message` this or next turn first.
6. React → after price/stock/cart events, re-verify and recover on the family
   ladder, then close with a valid order.
7. Finish → `place_test_order` once verified.

## Matching modes (do not mix casually)
- Literal: brands, model codes, SKUs, uncommon tokens, exact colors.
- Size aliases: s↔small, m↔medium, l↔large, xl↔extra large — only when helpful.
- Semantic paraphrase: soft preferences only ("for hiking" ≈ outdoor).
- Spec classes: ONLY thresholds stated in Task rules (HD / Full HD / QHD / 4K).
  `qHD` is NOT `QHD`. Never invent class floors.

## Search discipline
Good keywords: product-type nouns, brand, model fragments, distinctive phrase
tokens the shopper wrote, size/color literals.
Bad keywords: service/currency filler in refine loops; schema key dumps; five
synonymous rewrites that never change tokens.
Thin recall: ONE small reformulation (token cleanup OR broader type + filter),
then inspect. Do not thrash.

## Interventions
If `user_message` is a question / clarification / "which one…":
1. Call `message` this or next turn.
2. Answer with concrete evidence (brand, key spec, price, stock, identity).
3. Only then continue cart / order.
Ignoring a direct question fails intervention scoring even if the SKU is gold.

## Market / cart events
Triggers: stock_changed, price_changed, market_event, failed cart, sellout,
budget breach after reprice.
Protocol: inspect affected SKU → remove if invalid → recover with the ACTIVE
family ladder → `place_test_order` to CLOSE the event. Unfinished events score
poorly.

## Optional-claims landmine
On non-justification families (especially ranking), omit justification/claims
on `place_test_order`. Optional untruthful claims can zero an otherwise perfect
run.

## Turn budget
- Early: discover + inspect (and ask hidden specs if gated).
- Mid: commit first cart once evidence is enough — do not hoard steps.
- Late (≥ ~70% of max_steps): stop open search; finish the order path.
- Always reserve ≥1–2 turns for inspect_stock + place_test_order after recovery.

## Evidence hierarchy
Structured option / attribute rows and inspect_stock fields beat marketing
titles. Model facts override conflicting SEO titles. Phrase goals need title
tokens, not brand-only hits.

## Device authenticity
Prefer real devices over accessories / off-category noise. Use category_path
when available to reject kits, cases, and fillers that look similar in title.

## Anti-patterns (never)
- Cart a stub "just to have something," planning to upgrade later.
- Order from title alone without viewing the exact option/sku.
- Keep a pre-event #1 after the market moves without re-checking.
- Jump to other models before trying same-model other colors (recovery).
- Submit claim fields like display / size / title / category / "720p HD".
- Ask permission to order when the cart already holds a verified valid SKU.
- Ignore shopper questions while continuing to cart/order.

## Desk operating notes
- Prefer fewer, higher-signal tool calls over parallel thrash.
- When max_calls_per_turn > 1, batch compatible inspects (view/compare) but
  never batch a cart with an unanswered intervention.
- Observations may be long; extract the sku row you need and ignore SEO fluff.
- If a tool fails or returns empty, reformulate once, then switch to inspect
  on whatever candidates remain — do not loop forever.
- Authenticity: kits, cases, mounts, and "compatible with X" accessories are
  not substitutes for the real device unless the goal asks for the accessory.
""".strip()

# =============================================================================
# FAMILY STRATEGY PACKS — doctrine / donor tactics / case grid / recipe
# =============================================================================

PACK_INTENT = {
    "title": "TF1 · Intent Decomposition",
    "doctrine": """
## What this family tests
Turn a messy shopper ask into a correct decision sequence: separate firm walls
from soft preferences, honor preferred brand ONLY when a fully firm-compliant
SKU exists, answer interruptions via `message`, and recover after budget/stock
shocks without relaxing firm walls. First cart is graded.

The desk treats this as a sequencing problem more than a search problem: wrong
order of operations (cart before brand check, ignore intervention, stall on
impossible brand) fails even when a gold SKU exists in the catalog.

## Firm vs soft extraction
- FIRM: budget ceiling, in-stock, category / real device, explicit numeric or
  coded specs the shopper stated as must-haves.
- SOFT: preferred brand, nice-to-have features, aesthetic leanings, "would like".
- Brand is a SCORING BONUS, never a hard wall that overrides firm compliance.
- One mention of a budget/spec binds the nearest product only — do not smear
  constraints across unrelated items in multi-item goals.
- When the goal mixes "must" and "prefer", map must→firm and prefer→soft even
  if the prose is informal.

## Strategy ladder (8 steps)
1. Extract firm + soft explicitly before any tool call.
2. Dual shortlist: (A) preferred-brand + firm tokens; (B) firm-only any-brand backup.
3. Search A and B with value-side keywords; keep identities distinct.
4. `view` / `compare` until compliance is proven on the EXACT option/sku.
5. First `add_to_cart`: brand SKU if fully firm-compliant; else best firm backup.
   Never cart a non-compliant brand item "to check later."
6. Intervention: if shopper asks which/spec/budget → `message` with evidence, then continue.
7. Post-event (over budget / stockout): remove invalid → re-search under NEW
   constraints → same brand rule → inspect → `place_test_order`.
8. Omit justification.claims unless the goal explicitly wants a why.
""".strip(),
    "donor_tactics": """
## Donor tactics (Dragon desk names)
- Dual shortlist: maintain preferred-brand lane and firm-any-brand lane in parallel.
- Brand soft gate: brand-first IFF the brand SKU clears every firm wall.
- Constraint-scope discipline: do not apply one item's budget/spec to siblings.
- Same-shop preference: when tools expose shop/seller ids, prefer coherent carts;
  after failed same-shop preference, accept any firm-compliant set.
- Hard-check gate: before order, re-verify only non-negotiables on the exact sku.
- Market recover+close: remove → re-find under new world → order (do not stall).
- Near-miss refine: if brand lane is thin, one short reformulation from literal
  firm tokens — no service/price words in the refine string.
- Voucher-aware total (when voucher tools exist): estimate payable total before
  first commit; do not cart something that only works pre-discount fantasy.
""".strip(),
    "case_grid": """
## Case grid (micro-patterns)
1) Brand available + firm OK → first cart MUST be that brand SKU.
2) Brand exists but fails a firm gate → cart firm backup; do not stall hunting brand.
3) Brand impossible → brief brand attempt then switch; never infinite brand loop.
4) Shopper asks "which one?" mid-flow → message with brand/spec/price/stock, then cart.
5) After cart, price jumps over budget → remove → re-search under new budget → brand rule again.
6) Stockout of brand pick → remove → dual shortlist again under remaining firm walls.
7) Multi-item goal → bind each constraint to its product; do not treat kit fillers as separate gold.
8) Late turn with verified cart → order promptly; skip optional claims.
""".strip(),
    "failure_modes": """
## Failure modes / anti-patterns
- Missing an available preferred-brand compliant option on first cart.
- Stalling forever hunting an impossible brand.
- Answering interventions only in free text without the `message` tool.
- Closing without order after a recovery event.
- Treating filler/kit sub-items as separate products.
- Carting non-compliant brand "as a placeholder."
- Re-applying the old budget after a market reprice without re-checking.
""".strip(),
    "checklists": """
## Pre-cart checklist
- [ ] Firm list written; soft list written; brand tagged soft.
- [ ] Dual shortlist inspected on exact sku rows.
- [ ] Preferred-brand lane checked for full firm compliance.
- [ ] No pending unanswered shopper question.

## Pre-order checklist
- [ ] Cart holds the intentional first-cart winner (or post-event replacement).
- [ ] inspect_stock current on the ordered sku.
- [ ] After any event: firm walls still hold.
- [ ] place_test_order without optional claims (usual case).
""".strip(),
    "recipe": """
## Operational recipe
Parse firm/soft → dual shortlist search → prove compliance on exact sku →
first-cart brand-iff-compliant else firm backup → message interruptions →
on event remove+re-find under new constraints with same brand rule →
inspect_stock → place_test_order (no optional claims).
""".strip(),
}

PACK_RECALL = {
    "title": "TF2 · Retrieval Recall",
    "doctrine": """
## What this family tests
Surface the right candidates without tunnel-visioning on one lucky hit: phrase /
title matches, thin-recall reformulation, multi-identity shortlists, and
category / accessory traps.

The desk grades whether you SURVEYED realistic alternatives, not whether you
luckily clicked the first SEO title that vaguely matched.

## Firm vs soft extraction
- FIRM: distinctive phrase tokens, product type, budget, in-stock, explicit codes.
- SOFT: filler adjectives ("great", "best", "nice") — drop them from search.
- Phrase goals: the phrase must appear as TITLE tokens on view — brand-only or
  description-only hits are insufficient.
- Numbers/codes/variants are literal: `100ml` ≠ `10–150ml`; `green 2` ≠ `green`.
- Hyphenated compounds: try both joined and split forms once, not endlessly.

## Strategy ladder (8 steps)
1. Pass-1 search: distinctive product-type words + brand/model + quoted phrases.
2. If hit list is thin/noisy: ONE Pass-2 reformulation (token cleanup OR broader
   type + filter). Do not rewrite five times.
3. Shortlist several DISTINCT identities (different product_id / model / brand+title).
4. Phrase goals: verify phrase as title tokens on `view`.
5. Drop accessories / off-category rows via category_path when available.
6. `view` / `compare` shortlist; choose best in-stock in-budget match.
7. `place_test_order` promptly once verified.
8. Still handle market events and clarifications if they appear.
""".strip(),
    "donor_tactics": """
## Donor tactics (Dragon desk names)
- Near-miss refine: after candidates fail full verify, propose ≤N short (2–6 word)
  searches from literal non-negotiable tokens × distinctive missing constraint;
  no price/service words; must differ from baseline.
- Least-widen fallback: empty/thin → drop services → widen price slightly →
  truncate to first 2 distinctive tokens.
- Keyword noise strip: remove "looking for / show me / find me" prefixes.
- Dual shortlist: keep ~many candidates then filter; dedupe by product_id /
  brand+model+price — duplicates are NOT recall diversity.
- Exact SKU phrase bonus: boost when query phrase hits sku_options / title tokens.
- Evidence rank: fewest extraneous tokens → dedicated key over compound →
  structured attrs over SEO title.
- Reformulation recipes (keep small): drop filler adjectives; keep units/codes;
  split hyphenated compounds; broaden type noun while keeping one distinctive token;
  apply filter tools for budget/stock when search alone is noisy.
""".strip(),
    "case_grid": """
## Case grid (micro-patterns)
1) Distinctive quoted phrase → keep phrase tokens; verify on title before cart.
2) Thin first page → one near-miss refine from literal tokens; then inspect.
3) Many near-duplicates → dedupe identities; shortlist distinct product_ids.
4) Accessory with perfect attrs → reject via category_path / real-device check.
5) Code/size literal mismatch → discard even if brand matches.
6) Good shortlist in stock → cart best match; order promptly (no hoarding).
7) Market event after cart → re-verify; if invalid, near-miss refine again then order.
8) Shopper clarification mid-search → message, then resume shortlist path.
""".strip(),
    "failure_modes": """
## Failure modes / anti-patterns
- Commit after a single weak hit (kills shortlist/recall metrics).
- Endless synonymous rewrites that don't change tokens.
- Perfect attribute match on the wrong category (accessory trap).
- Ignoring a title-phrase requirement and matching only brand/category.
- Counting duplicate rows as recall diversity.
""".strip(),
    "checklists": """
## Pre-cart checklist
- [ ] Pass-1 done; Pass-2 only if thin/noisy (at most one).
- [ ] ≥2 distinct identities inspected when available.
- [ ] Phrase/title verified if required; category trap checked.
- [ ] Chosen sku in stock and in budget.

## Pre-order checklist
- [ ] Exact viewed option/sku, not title guess.
- [ ] No pending shopper question.
- [ ] place_test_order promptly; omit optional claims.
""".strip(),
    "recipe": """
## Operational recipe
Pass-1 distinctive tokens → optional one near-miss refine → multi-id shortlist →
phrase + category gates → view/compare → cart best verified match →
inspect_stock → place_test_order.
""".strip(),
}

PACK_CONSTRAINT = {
    "title": "TF3 · Constraint Satisfaction",
    "doctrine": """
## What this family tests
Every firm requirement must hold on the EXACT ordered SKU/option. Includes hidden
specs you must ask for before search, same-title multi-option traps, sellout
sibling traps, and resolution/class thresholds from Task rules.

Verification matrix (before order): for each firm constraint C, evidence must
come from the view option row or inspect_stock field — not a title guess — and
the exact sku selected must satisfy C. After any event, re-verify C still holds.

## Firm vs soft extraction
- ALL stated requirements are REQUIRED unless clearly marked soft.
- Hidden/gated specs: language like "if you ask" / "tell you when asked" →
  Turn 1 MUST be `message` asking for the gated requirement; do NOT search first.
- Soft adjectives are out of scope for the hard-check gate.
- Spec classes: apply ONLY mappings stated in Task rules; never invent floors.
  `qHD` ≠ `QHD`.
- Do not invent extra pass/fail criteria mid-select that reject valid gold.

## Strategy ladder (8 steps)
1. Scan for hidden-spec language. If present: Turn 1 = `message`; wait for reply.
2. After specs known: search/filter with VALUE tokens for each firm gate.
3. Never trust a title alone. `view` the product; select the compliant option/sku;
   `inspect_stock` on THAT sku.
4. Apply Task-rules class/threshold mappings exactly.
5. Sellout / invalid after cart: do NOT take a sibling option that drops a firm
   gate; move to a NEW product_id that still satisfies ALL firm constraints.
6. Hard-check gate: final re-check of non-negotiables on the exact sku row.
7. Finish with `place_test_order`; omit optional claims unless required.
8. Keep enough turns after events to inspect + order.
""".strip(),
    "donor_tactics": """
## Donor tactics (Dragon desk names)
- Ask-first gate: message the hidden firm spec before any search.
- Hard-check gate: pre-submit re-check of literal numbers/codes/variants only.
- Combo gate + swap: hard-check fails → swap next compatible candidate / new product_id.
- SKU coverage bonus: prefer options that hit multiple firm gates on the SAME sku row.
- Unit/value keyword variants: expand `100 ml` ↔ `100ml` in search without softening
  literal match at select time.
- Sellout new-product rule: never within-product sibling trap after sellout.
- Pre-extracted constraint list: select/filter must use the decomposed list; do
  not invent extra gates after seeing attractive titles.
""".strip(),
    "case_grid": """
## Case grid (micro-patterns)
1) "If you ask…" present → message first; block search until reply lands.
2) Same title, multiple options → pick compliant option row; never title-only order.
3) Resolution class stated → map via Task rules only; reject invented floors.
4) Sellout of chosen sku → new product_id that still clears ALL firm gates.
5) Sibling option available but drops a firm gate → reject; find new product.
6) Event after cart → re-verify EVERY firm constraint on the replacement.
7) Extra invented constraint rejects gold → do not invent pass/fail mid-select.
8) Late turns after event → force inspect + place_test_order path.
""".strip(),
    "failure_modes": """
## Failure modes / anti-patterns
- Searching before asking a gated firm spec.
- within_product_trap: sibling SKU after sellout.
- Ordering a title that "looks right" while the option row fails the gate.
- Confusing qHD with QHD / inventing a resolution class.
- Event fired but never closed with a valid final order.
- Inventing extra constraints that reject valid gold.
""".strip(),
    "checklists": """
## Pre-cart checklist
- [ ] Hidden specs asked and answered (if gated).
- [ ] Every firm C has evidence from view option row or inspect_stock.
- [ ] Exact sku selected satisfies every firm C.
- [ ] Not relying on title poetry.

## Pre-order checklist
- [ ] Hard-check gate passed on non-negotiables.
- [ ] After any event: all firm C still hold on ordered sku.
- [ ] No sibling trap; product_id is intentional.
- [ ] place_test_order; omit optional claims unless required.
""".strip(),
    "recipe": """
## Operational recipe
Ask-first if gated → wait → value-token search → view exact option →
inspect_stock → hard-check all firm gates → on sellout swap to NEW product_id →
re-verify → place_test_order.
""".strip(),
}

PACK_PREFERENCE = {
    "title": "TF4 · Preference Reasoning",
    "doctrine": """
## What this family tests
Soft use-case tradeoffs under hard constraints. Cheapest-valid is often WRONG.
First cart is graded as the intended preference-fit purchase.

The desk rewards evidence-backed soft-axis alignment among hard-compliant
candidates. A slightly dearer item that clearly documents the soft need beats a
cheaper item that only "looks fine" in marketing copy.

## Firm vs soft extraction
HARD (must hold): category / real device, budget ceiling, in-stock.
SOFT (optimize): durability, sweat/water, battery, materials, workout/outdoor,
comfort, portability, connectivity.
Soft NEVER overrides hard. If soft priorities conflict or are ambiguous: ONE
clarifying `message`, then decide — do not endless-clarify.
Facet admissibility: only attribute keys that match the shopper's facet count
(e.g. frame color ≠ lens color).

## Strategy ladder (8 steps)
1. Extract hard constraints and soft axes from the goal.
2. If soft priorities conflict: one clarifying `message`, then decide.
3. Shortlist ≥2 distinct listing/product identities that meet HARD constraints.
4. `compare` / `view` them on the SOFT axes using factual attributes (not vibes).
5. First `add_to_cart` = preference-fit winner (may be slightly dearer if in budget).
   Do not cart a cheap stub planning to upgrade later.
6. After reprice/listing change: remove invalid; recover to a DIFFERENT listing when needed.
7. Re-check stock/price; `place_test_order` without optional untruthful claims.
8. If two items tie on soft axes, prefer clearer documentation of the soft
   requirement; price breaks remaining ties.
""".strip(),
    "donor_tactics": """
## Donor tactics (Dragon desk names)
- Dual shortlist / multi-batch compare: force ≥2 hard-compliant candidates before first cart.
- Soft-axis compare: score only on axes the shopper named, using viewed facts.
- Facet-admissible evidence: if query says "grey FRAME", only frame-color-like keys
  count — not lens_color or unrelated fields.
- Evidence quality over pennies: structured soft-axis evidence beats tiny price savings.
- First-cart preference-fit: never cheapest stub then upgrade.
- Claim strip: non-justification families omit claims on order.
- Reprice recover: when listing/price changes, re-evaluate preference winner under
  still-hard walls; do not cling to a now-invalid favorite.
""".strip(),
    "case_grid": """
## Case grid (micro-patterns)
1) Two hard-OK items, one better on named soft axis → cart the soft winner.
2) Soft favorite fails hard budget → discard; soft never overrides hard.
3) Ambiguous soft priority → one clarify message, then compare ≥2.
4) Cheapest hard-OK is weak on soft evidence → do NOT default to cheapest.
5) Reprice pushes winner over budget → remove; pick next preference-fit listing.
6) Listing change invalidates soft evidence → re-view and re-compare.
7) Tie on soft axes → clearer documentation wins; else lower price.
8) First cart already preference-fit → order without optional claims.
""".strip(),
    "failure_modes": """
## Failure modes / anti-patterns
- Cheapest-valid default → low preference alignment.
- First cart placeholder → first-cart score fails permanently.
- Soft favorite that fails hard constraints.
- Never observing / comparing soft axes.
- Optional claims that are false → reward zeros.
- Endless clarify without carting.
""".strip(),
    "checklists": """
## Pre-cart checklist
- [ ] Hard walls listed; soft axes listed.
- [ ] ≥2 hard-compliant candidates compared on soft axes.
- [ ] Winner chosen by soft evidence, still within hard walls.
- [ ] No unanswered clarification pending (or one clarify already done).

## Pre-order checklist
- [ ] First cart is the preference-fit winner (or post-event replacement).
- [ ] Stock/price re-checked after any change.
- [ ] place_test_order without optional untruthful claims.
""".strip(),
    "recipe": """
## Operational recipe
Extract hard/soft → optional one clarify → shortlist ≥2 hard-OK →
compare on soft axes with facts → first-cart preference winner →
on reprice recover different listing → inspect → place_test_order (no claims).
""".strip(),
}

PACK_RANKING = {
    "title": "TF5 · Ranking",
    "doctrine": """
## What this family tests
Closed-set ranking of named candidates by lexicographic tradeoff priority, with
mandatory post-event re-rank and a clean order payload (usually NO claims).

This family fails spectacularly on two landmines: (1) keeping a pre-event winner
after the market moves, and (2) attaching optional justification.claims that are
wrong or even unnecessary.

## Firm vs soft extraction
- Candidates are a CLOSED SET of named `product_id::sku` tokens from the goal.
- Rank ONLY by the stated tradeoff priority (common when stated: highest refresh
  incl. advertised overclock → largest screen → lowest price).
- Never invent extra axes. Never pick an option strictly dominated on every axis.
- Soft marketing copy is irrelevant; closed-set facts win.
- Budget/stock still apply as availability filters on the ranked set.

## Strategy ladder (8 steps)
1. Parse every named candidate token product_id::sku from the goal.
2. Prefer viewing those IDs directly — avoid open catalog thrash.
3. Batch `view` / `compare` ALL candidates before any `add_to_cart`.
4. Rank only by stated lexicographic axes; apply dominance check.
5. First `add_to_cart` = current best AVAILABLE (in-stock, in-budget) candidate.
6. On stock/price/cart event: re-inspect survivors → re-rank from scratch with
   the SAME priority → cart the NEW #1 → inspect_stock → place_test_order.
7. If turn index is high and not ordered: force the finish path immediately.
8. place_test_order arguments = product_id + sku ONLY (no justification/claims
   unless Task rules explicitly demand a reason).
""".strip(),
    "donor_tactics": """
## Donor tactics (Dragon desk names)
- Closed-set lex rank: extract named IDs; skip open search when IDs are given.
- Dominance check: before carting X, ensure no available Y beats X on the stated order.
- Hard-check gate: firm gates still apply on the ranked pick.
- Post-event re-rank: never keep pre-event #1 if it is no longer best.
- Finish discipline: late-turn force order; do not reopen the catalog.
- Claim strip: strip justification on ranking orders (critical landmine).
- Among equals: prefer clearer structured evidence over earliest appearance in the list.
""".strip(),
    "case_grid": """
## Case grid (micro-patterns)
1) Named candidates present → view all; never open-search thrash.
2) Lex axes stated → rank strictly; ignore unstated soft axes.
3) Dominated candidate → never cart it while a better available option exists.
4) Pre-event #1 still in cart but no longer best → swap to new #1 then order.
5) Stockout of #1 → re-rank remaining named set; cart new #1.
6) Price change alters lex order → re-rank from scratch.
7) Late turn, not ordered → force inspect + place_test_order(product_id, sku).
8) Model attaches claims → sanitizer/policy: omit claims on ranking order.
""".strip(),
    "failure_modes": """
## Failure modes / anti-patterns
- Never calling place_test_order.
- Keeping pre-event #1 after the market moves.
- Open-search thrash instead of viewing named IDs.
- Shipping claims on a ranking order.
- Inventing extra ranking axes not in the goal.
""".strip(),
    "checklists": """
## Pre-cart checklist
- [ ] All named candidates parsed and viewed/compared.
- [ ] Lex order applied; dominance checked.
- [ ] Current #1 is available (stock + budget).

## Pre-order checklist
- [ ] After any event: re-ranked from scratch; cart matches new #1.
- [ ] place_test_order has ONLY product_id + sku (usual case).
- [ ] No optional justification.claims.
""".strip(),
    "recipe": """
## Operational recipe
Parse named product_id::sku set → batch view/compare → lex rank + dominance →
cart current #1 → on event re-rank survivors → cart new #1 →
inspect_stock → place_test_order(product_id, sku) with NO claims.
""".strip(),
}

PACK_RECOVERY = {
    "title": "TF6 · Recovery",
    "doctrine": """
## What this family tests
When the exact preferred item disappears or breaks constraints after interaction,
recover with the LEAST relaxation. Category, budget, and in-stock NEVER relax.

Skipping a ladder level (jumping to other models while other colors of the same
model remain) is a characteristic failure. So is relaxing a wall "just this once."

## Firm vs soft extraction
Immutable firm walls: category / product type, budget ceiling, in-stock.
Never relax these to "make something fit."
Color and model follow the least-relax ladder; they are ordered preferences,
not walls you may shatter.
Color evidence order: variant/option color → product color field → title last.

## Least-relaxation ladder
Level 0: Exact model + exact color (commit when available).
Level 1: Same model, other color (only after Level 0 invalid/unavailable).
Level 2: Other models (only after Level 1 exhausted).
Never skip levels.

## Strategy ladder (8 steps)
1. Identify exact model + color targets from the goal.
2. Prefer structured model facts over conflicting marketing titles.
3. Commit to Level 0 when in stock and in budget.
4. On disappearance / budget break after cart: inspect → remove if needed →
   climb the ladder ONE level at a time.
5. Color evidence order: variant/option color → product color field → title last.
   If multi-color is unresolved, do not invent a color.
6. Re-check price+stock after every cart change; then `place_test_order`.
7. Omit optional untruthful claims.
8. If L1/L2 search is thin: one near-miss refine keeping brand/color/model tokens.
""".strip(),
    "donor_tactics": """
## Donor tactics (Dragon desk names)
- Least-relax ladder: climb 0 → 1 → 2 without skipping.
- Hard walls immutable: never relax budget / stock / category.
- Model-fact override: structured model beats marketing title.
- Near-miss refine: preferred item gone but near-misses remain — short refine from sticky tokens.
- Thin recovery fallback: drop service → widen price slightly → shorten query.
- Recover+close: after climb, inspect → cart → inspect → place_test_order.
- Sticky ID keys: keep brand/color/model/id tokens while climbing; do not
  replace the whole query with vague synonyms.
""".strip(),
    "case_grid": """
## Case grid (micro-patterns)
1) Exact model+color available → commit Level 0; do not "keep looking."
2) Exact color gone, same model other color exists → Level 1 before other models.
3) Same model all colors gone → Level 2 other models under firm walls.
4) Title says color A but option row says color B → trust option/model fact.
5) Budget break after reprice → remove; climb ladder under NEW budget (walls still hold).
6) Jumping to other models while other colors remain → anti-pattern; undo.
7) Thin L1/L2 results → one near-miss refine; then inspect.
8) Event recovered but no order → force place_test_order to close.
""".strip(),
    "failure_modes": """
## Failure modes / anti-patterns
- Jumping to other models before trying other colors of the same model.
- Skipping a commit when the preferred exact item was available.
- Relaxing budget/category/stock to force a match.
- Trusting a title that contradicts an explicit model fact.
- Event recovery without final order.
""".strip(),
    "checklists": """
## Pre-cart checklist
- [ ] Exact model + color identified.
- [ ] Level 0 attempted before Level 1/2.
- [ ] Model facts checked against title conflicts.
- [ ] Firm walls (category/budget/stock) intact.

## Pre-order checklist
- [ ] Ladder level justified by prior unavailability.
- [ ] Price+stock re-checked after cart changes.
- [ ] place_test_order closes the event; omit optional claims.
""".strip(),
    "recipe": """
## Operational recipe
Lock exact model+color → commit Level 0 if valid → on break inspect/remove →
climb least-relax one level → optional near-miss refine if thin →
re-check walls → place_test_order (no claims).
""".strip(),
}

PACK_JUSTIFY = {
    "title": "TF7 · Justification",
    "doctrine": """
## What this family tests
A valid purchase PLUS structured, fact-backed justification.claims on order.
Wrong claim shape zeros reward even when the product is correct.

ShoppingBench rarely labels justification as primary; EnvPack does. Treat claim
shape as a first-class deliverable equal to the SKU choice.

## Firm vs soft extraction
- Purchase must be valid: in-stock, in-budget, firm gates met.
- Claims must be minimal, allowlisted, and grounded in view / inspect_stock.
- Soft feature claims only if mentioned in the goal; at most ONE soft reason.
- After market events: rebuild claims from FRESH facts before ordering.
- Forbidden shapes include marketing phrases used AS field names.

## Strategy ladder (7+ steps)
1. Find a valid in-stock in-budget product that meets firm gates.
2. `view` / `compare` / `inspect_stock` so every claim value is grounded BEFORE order.
3. Build a MINIMAL claim set:
   - price (current)
   - in_stock (current)
   - exactly ONE firm-spec field using canonical name: resolution | screen_size |
     refresh_rate | ram (whichever the goal implies)
   - plus at most ONE soft reason if evidenced (prefer brand; else streaming /
     microphone / privacy_cover / autofocus / hdr when mentioned)
4. Remap aliases before submit: display→resolution, size/screen→screen_size,
   stock/availability→in_stock.
5. Deny forbidden fields: category, title, display, size, OS, summary, evidence, reason.
6. `place_test_order` with justification.claims.
7. If price/stock/spec evidence changes after cart: re-view / re-inspect and
   rebuild claims from fresh facts.
8. Never invent soft features when unknown/absent; prefer fewer true claims.
""".strip(),
    "donor_tactics": """
## Donor tactics (Dragon desk names)
- Claim allowlist: only resolution, screen_size, refresh_rate, ram, brand, price,
  in_stock, streaming, microphone, privacy_cover, autofocus, hdr.
- Alias remap: display→resolution; size/screen→screen_size; stock→in_stock.
- Soft-feature gate: soft keys only if the goal mentions them; at most one soft.
- Fresh-evidence rebuild: after events, re-inspect before attaching claims.
- Hard-check gate: product validity first; claims second — both required.
- Non-TF7 landmine awareness: on other families, STRIP claims instead.
- Value hygiene: no prose blobs inside claims; normalize screen_size / resolution
  to compact factual forms when possible.
""".strip(),
    "case_grid": """
## Case grid (micro-patterns)
1) Valid product + grounded price/in_stock/spec → minimal claims → order.
2) Model emits field=display → remap to resolution (or drop if unmappable).
3) Model emits field=size → remap to screen_size.
4) Model emits "720p HD" as a field name → reject; use canonical resolution.
5) Soft feature not in goal → drop soft claim; keep required trio.
6) Two soft reasons → keep only one (prefer brand).
7) Price changed after cart → re-inspect; refresh price/in_stock claims.
8) Correct claims on wrong product → fix product first; claims cannot save it.
""".strip(),
    "failure_modes": """
## Failure modes / anti-patterns
- field=display / size / "720p HD" → canonical faithfulness fails.
- Invented features.
- Stale price/stock vs claims.
- Correct-looking claims on the wrong gold product.
- Decorative claim stacks (prefer fewer true claims).
""".strip(),
    "checklists": """
## Pre-cart checklist
- [ ] Firm gates satisfied on exact sku.
- [ ] View/inspect evidence collected for every planned claim.

## Pre-order checklist
- [ ] Claims allowlisted and remapped.
- [ ] price + in_stock match latest inspect/cart state.
- [ ] Exactly one firm-spec field; ≤1 soft reason; soft mentioned in goal.
- [ ] No denied fields (category/title/display/size/os/summary/evidence).
""".strip(),
    "recipe": """
## Operational recipe
Find valid product → ground facts via view/inspect → build minimal allowlisted
claims with alias remap → place_test_order with justification.claims →
on event rebuild claims from fresh facts then order.
""".strip(),
}

STRATEGY_PACKS: dict[str, dict[str, str]] = {
    F_INTENT: PACK_INTENT,
    F_RECALL: PACK_RECALL,
    F_CONSTRAINT: PACK_CONSTRAINT,
    F_PREFERENCE: PACK_PREFERENCE,
    F_RANKING: PACK_RANKING,
    F_RECOVERY: PACK_RECOVERY,
    F_JUSTIFY: PACK_JUSTIFY,
}

# Mode tag legend (injected into runtime block for the model)
_MODE_LEGEND = {
    MODE_ASK_SPEC: "Ask the gated firm spec via message before any search.",
    MODE_CLOSED_RANK: "Closed-set lex rank over named product_id::sku tokens.",
    MODE_BRAND_SOFT: "Preferred brand is soft — brand-first iff fully firm-compliant.",
    MODE_THIN_RECALL: "Expect thin recall; allow one near-miss refine then shortlist.",
    MODE_SOFT_COMPARE: "Compare ≥2 hard-OK candidates on named soft axes before cart.",
    MODE_LEAST_RELAX: "Climb exact→other-color→other-model; never relax walls.",
    MODE_CLAIM_BUILD: "Attach minimal allowlisted justification.claims on order.",
    MODE_INTERVENE: "Expect shopper questions; message before cart/order.",
    MODE_MARKET: "Expect market/cart shocks; recover then close with order.",
    MODE_PHRASE: "Verify distinctive phrase as title tokens on view.",
    MODE_DEFAULT: "Follow the active family operational recipe.",
}

# =============================================================================
# Dragon desk coaches (runtime nudges — not sibling "SnowGirl coach" wording)
# =============================================================================

DESK_COACH_ASK_SPEC = (
    "Dragon desk · Hidden-spec gate: the shopper withholds a firm requirement "
    "until asked. Call `message` NOW requesting that gated spec. Do not search, "
    "view, cart, or order until the reply arrives."
)

DESK_COACH_INTERVENTION = (
    "Dragon desk · Intervention: the shopper asked a direct question. Call "
    "`message` this turn with concrete evidence (brand, key spec, price, stock, "
    "identity). Do not advance cart/order until you answer."
)

DESK_COACH_BRAND_DUAL = (
    "Dragon desk · TF1 dual-shortlist: keep (A) preferred-brand + firm and "
    "(B) firm-any-brand lanes. First cart = brand SKU only if fully firm-"
    "compliant; otherwise cart the firm backup. Never cart non-compliant brand "
    "as a stub."
)

DESK_COACH_ONE_REFINE = (
    "Dragon desk · TF2 one-refine shortlist: if recall is thin, run ONE short "
    "near-miss refine from literal tokens (no service/price words), then inspect "
    "several DISTINCT product identities before committing."
)

DESK_COACH_SOFT_COMPARE = (
    "Dragon desk · TF4 soft-axis compare: shortlist ≥2 hard-compliant candidates, "
    "compare them on the named soft axes with viewed facts, then first-cart the "
    "preference-fit winner — not the cheapest stub."
)

DESK_COACH_RANK_OPEN = (
    "Dragon desk · TF5 closed-set: extract every product_id::sku, batch view/"
    "compare the named set, lex-rank by stated axes only, cart current #1. "
    "Skip open catalog thrash."
)

DESK_COACH_RANK_RERANK = (
    "Dragon desk · TF5 re-rank: market moved. Re-inspect named survivors, re-rank "
    "from scratch with the same lex priority, cart the NEW #1 (not the pre-event "
    "winner if it lost), then prepare to order."
)

DESK_COACH_RANK_FINISH = (
    "Dragon desk · TF5 finish: stop searching. Ensure current #1 is carted and "
    "call place_test_order(product_id, sku) ONLY — omit justification/claims."
)

DESK_COACH_LEAST_RELAX = (
    "Dragon desk · TF6 least-relax ladder: Level 0 exact model+color → Level 1 "
    "same model other color → Level 2 other models. Never skip levels. Never "
    "relax budget, stock, or category. Close with place_test_order."
)

DESK_COACH_CLAIMS = (
    "Dragon desk · TF7 claims allowlist: on place_test_order attach minimal "
    "justification.claims using canonical keys (resolution/screen_size/"
    "refresh_rate/ram + price + in_stock + at most one soft). Remap display→"
    "resolution, size/screen→screen_size, stock→in_stock. Deny category/title/"
    "display/size/os/summary/evidence."
)

DESK_COACH_MARKET = (
    "Dragon desk · Market event: inspect affected sku → remove if invalid → "
    "recover on the active family ladder → place_test_order to CLOSE the event. "
    "Unfinished events score poorly."
)

DESK_COACH_EMPTY_TOOL = (
    "Dragon desk · Empty tool_calls: the environment is not done. Choose one of "
    "the supplied tools and continue the active family recipe."
)

DESK_COACH_OMIT_CLAIMS = (
    "Dragon desk · Claims landmine: this family does not require a why. On "
    "place_test_order send product_id + sku only — strip justification/claims."
)

DESK_COACH_LATE_FINISH = (
    "Dragon desk · Turn budget: you are in the late window. Stop open search. "
    "Re-verify stock/price on the intended sku and place_test_order. Leave no "
    "unfinished market event."
)

DESK_COACH_AUTHENTICITY = (
    "Dragon desk · Authenticity: reject accessories / off-category noise that "
    "only match via SEO title. Prefer real devices with matching category_path."
)

# =============================================================================
# Claim desk constants
# =============================================================================

_FIELD_REMAP: dict[str, str] = {
    "display": "resolution",
    "resolution class": "resolution",
    "pixel class": "resolution",
    "screen": "screen_size",
    "size": "screen_size",
    "screen size": "screen_size",
    "stock": "in_stock",
    "availability": "in_stock",
}

_CLAIM_DENY = frozenset(
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

_CLAIM_ALLOW = frozenset(
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

_SOFT_FEATURE_KEYS = frozenset(
    {"streaming", "microphone", "privacy_cover", "autofocus", "hdr"}
)
_SOFT_ALL_KEYS = frozenset({"brand"}) | _SOFT_FEATURE_KEYS

_SPEC_AS_FIELD_RX = re.compile(
    r"(?i)^(?:\d{3,4}\s*[pP](?:\s*HD)?|full\s*hd|qhd|uhd|4k|"
    r"\d+(?:\.\d+)?\s*-?\s*inch(?:es)?)$"
)

# =============================================================================
# Classification + prompt assembly
# =============================================================================


def _goal_head(query: str) -> str:
    text = query or ""
    marker = "Task rules:"
    cut = text.find(marker)
    return text if cut < 0 else text[:cut]


def detect_family(query: str) -> str:
    """Score capability fingerprints; tie-break with Dragon priority order."""
    blob = query or ""
    tally: dict[str, int] = {}
    for family_id, patterns in _FINGERPRINTS:
        hits = sum(1 for rx in patterns if rx.search(blob))
        if hits:
            tally[family_id] = hits

    low = blob.lower()
    if "justification is optional" in low and F_JUSTIFY in tally:
        head = _goal_head(blob)
        if not re.search(r"\bwhy\b|reason|justify", head, re.I):
            tally.pop(F_JUSTIFY, None)

    if not tally:
        return F_RECALL

    return max(
        tally.items(),
        key=lambda item: (item[1], _FAMILY_TIEBREAK.get(item[0], 0)),
    )[0]


def detect_mode(family: str, query: str) -> list[str]:
    """Lighter mode tags (not a full subtype catalog)."""
    head = _goal_head(query)
    low = (query or "").lower()
    head_l = head.lower()
    tags: list[str] = []

    if family == F_CONSTRAINT and (
        _ASK_GATE_RX.search(head) or _ASK_GATE_RX.search(query or "")
    ):
        tags.append(MODE_ASK_SPEC)
    if family == F_RANKING or _NAMED_SKU_RX.search(query or ""):
        if family == F_RANKING:
            tags.append(MODE_CLOSED_RANK)
    if family == F_INTENT and re.search(
        r"prefer(?:red)? brand|nice[- ]to[- ]have|fully (?:firm[- ])?compliant",
        low,
        re.I,
    ):
        tags.append(MODE_BRAND_SOFT)
    if family == F_INTENT and re.search(
        r"which one|before you order|shopper may ask|\bwait\b", low, re.I
    ):
        tags.append(MODE_INTERVENE)
    if family == F_RECALL and re.search(
        r"reformulat|lucky hit|thin|shortlist|keyword", low, re.I
    ):
        tags.append(MODE_THIN_RECALL)
    if family == F_RECALL and re.search(
        r"phrase|title token|matches?\s+the\s+phrase", low, re.I
    ):
        tags.append(MODE_PHRASE)
    if family == F_PREFERENCE:
        tags.append(MODE_SOFT_COMPARE)
    if family == F_RECOVERY:
        tags.append(MODE_LEAST_RELAX)
    if family == F_JUSTIFY or re.search(
        r"\bwhy\b|trustworthy reason|justification\.claims", head_l, re.I
    ):
        if family == F_JUSTIFY:
            tags.append(MODE_CLAIM_BUILD)
    if re.search(
        r"price_changed|stock_changed|market|over budget|sell(?:s)? out|"
        r"becomes? unavailable|after (?:cart|add)",
        low,
        re.I,
    ):
        tags.append(MODE_MARKET)

    if not tags:
        tags.append(MODE_DEFAULT)
    # de-dupe preserve order
    seen: set[str] = set()
    ordered: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _tee_to_openrouter(raw: str) -> str:
    key = (raw or "").strip()
    if not key:
        return _DRAGON_HARD
    for tee_id, or_id in _TEE_TO_OPENROUTER:
        if key == tee_id or key.endswith(tee_id.split("/", 1)[-1]):
            return or_id
    return key


def _select_model(family: str, modes: list[str]) -> str:
    forced = (getenv("SANDBOX_MODEL") or "").strip()
    if forced:
        return _tee_to_openrouter(forced)
    # Light lane: simple recall modes; hard for everything else.
    if family == F_RECALL and MODE_ASK_SPEC not in modes:
        return _DRAGON_LIGHT
    return _DRAGON_HARD


def _fallback_chain(primary: str) -> list[str]:
    chain = [primary]
    for mid in _DRAGON_FALLBACKS:
        if mid not in chain:
            chain.append(mid)
    if _DRAGON_LIGHT not in chain:
        chain.append(_DRAGON_LIGHT)
    return chain[:4]  # primary + up to 3 fallbacks explored in inference loop


def _tool_catalog(tools: list[dict[str, Any]]) -> str:
    names: list[str] = []
    for entry in tools or []:
        fn = entry.get("function") if isinstance(entry, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
        elif isinstance(entry, dict) and entry.get("name"):
            names.append(str(entry["name"]))
    return ", ".join(names) if names else "(dynamic tools from policy)"


def assemble_system_prompt(
    *,
    family: str,
    modes: list[str],
    query: str,
    tools: list[dict[str, Any]],
    max_steps: int,
    max_calls: int,
) -> str:
    pack = STRATEGY_PACKS.get(family) or STRATEGY_PACKS[F_RECALL]
    named = _NAMED_SKU_RX.findall(query or "")
    named_line = (
        ", ".join(list(dict.fromkeys(named))[:12]) if named else "(none parsed)"
    )
    omit = family != F_JUSTIFY
    mode_lines = "\n".join(
        f"  - {m}: {_MODE_LEGEND.get(m, 'follow family recipe')}" for m in modes
    )
    runtime = f"""
## Runtime block (Dragon Strategy Desk)
- Active family: {family}
- Mode tags:
{mode_lines}
- Available tools: {_tool_catalog(tools)}
- max_steps: {max_steps}
- max_calls_per_turn: {max_calls}
- Named candidate tokens parsed: {named_line}
- Ask-first hidden spec: {"YES — message before search" if MODE_ASK_SPEC in modes else "no"}
- Optional claims policy: {"OMIT justification/claims on order" if omit else "ATTACH minimal allowlisted claims on order"}
- Always tool_choice=required; temperature=0; respect max_calls_per_turn.
""".strip()

    sections = [
        CORE_DOCTRINE,
        f"# Family pack — {pack['title']}",
        pack["doctrine"],
        pack["donor_tactics"],
        pack["case_grid"],
        pack["failure_modes"],
        pack["checklists"],
        pack["recipe"],
        runtime,
    ]
    return "\n\n".join(sections)


# =============================================================================
# Claim / order sanitizer
# =============================================================================


def _infer_spec_field(query: str) -> str | None:
    head = _goal_head(query).lower()
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


def _canon_field(field: str, query: str) -> str:
    raw = str(field or "").strip()
    key = raw.lower()
    if key in _CLAIM_DENY and key not in _FIELD_REMAP:
        return ""
    if key in _FIELD_REMAP:
        return _FIELD_REMAP[key]
    if _SPEC_AS_FIELD_RX.match(raw):
        return _infer_spec_field(query) or "resolution"
    return raw


def _canon_value(field: str, value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if field == "screen_size":
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        return f"{m.group(1)} inches" if m else text
    if field == "resolution":
        compact = text.lower().replace(" ", "")
        if compact in {"720p", "720phd", "hd"} or (
            "1280" in compact and "720" in compact
        ):
            return "hd"
        if compact in {"1080p", "1080pfullhd", "fullhd", "fhd"} or (
            "1920" in compact and "1080" in compact
        ):
            if "1920" in text and "1080" in text:
                return re.sub(r"\s+", " ", text)
            return "1920 x 1080"
        if compact in {"1440p", "qhd"} or (
            "2560" in compact and "1440" in compact
        ):
            if "2560" in text and "1440" in text:
                return re.sub(r"\s+", " ", text)
            return "2560 x 1440"
    if field == "in_stock":
        return "true" if text.lower() in {"true", "1", "yes", "in_stock"} else text
    return text


def sanitize_order_args(
    tool_name: str,
    args: dict[str, Any],
    *,
    family: str,
    query: str,
) -> dict[str, Any]:
    """Outbound sanitizer: strip claims on TF5/non-justify; allowlist+remap on TF7."""
    payload = dict(args)
    if tool_name != "place_test_order":
        return payload

    head = _goal_head(query)
    wants_why = bool(re.search(r"\bwhy\b|justif|reason|tell me why", head, re.I))

    if family == F_RANKING and not wants_why:
        payload.pop("justification", None)
        return payload

    if family != F_JUSTIFY:
        if not wants_why:
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
        field = _canon_field(str(item.get("field") or ""), query)
        value = _canon_value(field, item.get("value"))
        if not field or value == "":
            continue
        fl = field.lower()
        if fl in _CLAIM_DENY or fl not in _CLAIM_ALLOW:
            continue
        if fl in _SOFT_FEATURE_KEYS and fl not in head_l:
            continue
        if fl in _SOFT_ALL_KEYS:
            if soft_used:
                continue
            soft_used = True
        if fl in used:
            continue
        used.add(fl)
        kept.append({"field": field, "value": value})

    payload["justification"] = {"claims": kept}
    return payload


# =============================================================================
# Episode helpers
# =============================================================================


def _parse_tool_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("function", {}).get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _truncate_obs(obs: Any, cap: int = _OBS_CAP) -> str:
    text = json.dumps(obs, ensure_ascii=False) if not isinstance(obs, str) else obs
    if len(text) <= cap:
        return text
    return text[: cap - 20] + "…[truncated]"


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


def _market_kind(obs: Any) -> str:
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


def _obs_done(obs: Any) -> bool:
    if isinstance(obs, dict):
        if obs.get("done") is True:
            return True
        nested = obs.get("observation")
        if isinstance(nested, dict) and nested.get("done") is True:
            return True
    return False


class EpisodeState:
    """Mutable desk state for one EnvPack episode."""

    __slots__ = (
        "await_reply",
        "saw_market",
        "ordered",
        "first_cart",
        "asked_spec",
        "search_count",
        "viewed_named",
    )

    def __init__(self) -> None:
        self.await_reply = False
        self.saw_market = False
        self.ordered = False
        self.first_cart = False
        self.asked_spec = False
        self.search_count = 0
        self.viewed_named = 0


def _desk_coaches(
    *,
    family: str,
    modes: list[str],
    state: EpisodeState,
    turn: int,
    max_steps: int,
) -> list[str]:
    notes: list[str] = []
    late = turn >= max(2, int(max_steps * 0.7))

    if MODE_ASK_SPEC in modes and not state.asked_spec and turn <= 2:
        notes.append(DESK_COACH_ASK_SPEC)

    if state.await_reply:
        notes.append(DESK_COACH_INTERVENTION)

    if family == F_INTENT and turn <= 4 and not state.first_cart:
        notes.append(DESK_COACH_BRAND_DUAL)

    if family == F_RECALL and state.search_count >= 1 and not state.first_cart:
        notes.append(DESK_COACH_ONE_REFINE)

    if family == F_PREFERENCE and not state.first_cart:
        notes.append(DESK_COACH_SOFT_COMPARE)

    if family == F_RANKING:
        if not state.first_cart and turn <= 5:
            notes.append(DESK_COACH_RANK_OPEN)
        if state.saw_market and not state.ordered:
            notes.append(DESK_COACH_RANK_RERANK)
        if late or (state.saw_market and turn >= 2):
            notes.append(DESK_COACH_RANK_FINISH)
        notes.append(DESK_COACH_OMIT_CLAIMS)

    if family == F_RECOVERY:
        notes.append(DESK_COACH_LEAST_RELAX)

    if family == F_JUSTIFY and (state.first_cart or late):
        notes.append(DESK_COACH_CLAIMS)

    if state.saw_market and not state.ordered:
        notes.append(DESK_COACH_MARKET)

    if family not in {F_JUSTIFY, F_RANKING} and late:
        notes.append(DESK_COACH_OMIT_CLAIMS)

    if late and not state.ordered:
        notes.append(DESK_COACH_LATE_FINISH)

    if family in {F_RECALL, F_CONSTRAINT, F_PREFERENCE} and not state.first_cart:
        notes.append(DESK_COACH_AUTHENTICITY)

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for n in notes:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:5]


def _inference_chat(
    *,
    models: list[str],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Try primary then fallbacks (up to 3 models total attempts beyond empty)."""
    last_err: str | None = None
    for model_id in models[:3]:
        try:
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": model_id,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
        except Exception as exc:  # noqa: BLE001 — sandbox may raise variously
            last_err = str(exc)
            continue
        if inference is None:
            last_err = "inference request returned None"
            continue
        choices = inference.get("choices") or []
        if not choices:
            last_err = "empty choices"
            continue
        message = choices[0].get("message") or {}
        return message, model_id
    raise RuntimeError(f"inference failed across models: {last_err}")


# =============================================================================
# Entry
# =============================================================================


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Dragon Strategy Desk EnvPack loop. Returns dialogue transcript."""
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    query = str(policy.get("query") or problem_data.get("query") or "")
    tools = list(policy.get("tools") or [])
    max_steps = int(policy.get("max_steps") or 20)
    max_calls = int(policy.get("max_calls_per_turn") or 1)
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))

    family = detect_family(query)
    modes = detect_mode(family, query)
    primary_model = _select_model(family, modes)
    model_chain = _fallback_chain(primary_model)

    system_prompt = assemble_system_prompt(
        family=family,
        modes=modes,
        query=query,
        tools=tools,
        max_steps=max_steps,
        max_calls=max_calls,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "dragon": {
                "version": 1,
                "family": family,
                "mode": modes,
                "model": primary_model,
                "provider": "openrouter",
                "lineage": "rank3_strategy_desk",
            },
        }
    ]

    state = EpisodeState()
    if MODE_ASK_SPEC in modes:
        # Desk expects ask-first; keep await until we see a shopper reply after message.
        pass

    for turn in range(1, max_steps + 1):
        coaches = _desk_coaches(
            family=family,
            modes=modes,
            state=state,
            turn=turn,
            max_steps=max_steps,
        )
        turn_messages = list(messages)
        if coaches:
            turn_messages = turn_messages + [
                {
                    "role": "user",
                    "content": "Dragon desk coach notes:\n- " + "\n- ".join(coaches),
                }
            ]

        assistant: dict[str, Any] | None = None
        used_model = primary_model
        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []

        for nudge_round in range(2):
            attempt_messages = turn_messages
            if nudge_round == 1:
                attempt_messages = turn_messages + [
                    {"role": "user", "content": DESK_COACH_EMPTY_TOOL}
                ]
            assistant, used_model = _inference_chat(
                models=model_chain,
                messages=attempt_messages,
                tools=tools,
            )
            assistant_content = assistant.get("content") or ""
            tool_calls = list(assistant.get("tool_calls") or [])[:max_calls]
            if tool_calls:
                break
            dialogue.append({"role": "assistant", "content": assistant_content})
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": DESK_COACH_EMPTY_TOOL},
                ]
            )
            turn_messages = list(messages)
        else:
            raise RuntimeError("model returned no tool call before completion")

        # Track outbound intents from tool names / args
        sanitized_calls: list[dict[str, Any]] = []
        env_actions: list[dict[str, Any]] = []
        for index, call in enumerate(tool_calls, start=1):
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            args = _parse_tool_args(call)
            args = sanitize_order_args(name, args, family=family, query=query)

            if name in {"search", "find_product", "search_products"}:
                state.search_count += 1
            if name == "message":
                state.asked_spec = True
                state.await_reply = False
            if name == "add_to_cart":
                state.first_cart = True
            if name == "place_test_order":
                state.ordered = True
            if name in {"view", "compare"} and _NAMED_SKU_RX.search(
                json.dumps(args, ensure_ascii=False)
            ):
                state.viewed_named += 1

            # Rewrite tool_call arguments for transcript consistency
            new_call = {
                "id": call.get("id"),
                "type": call.get("type", "function"),
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
            sanitized_calls.append(new_call)
            env_actions.append(
                {
                    "call_id": f"{problem_id}-turn-{turn}-{index}",
                    "action": {"name": name, "args": args},
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": sanitized_calls,
            }
        )

        group_id = f"{problem_id}-turn-{turn}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": env_actions,
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")

        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": sanitized_calls,
                "environment_result": result,
                "dragon_model": used_model,
            }
        )

        done_any = False
        for tool_call, call_result in zip(
            sanitized_calls, result.get("calls") or [], strict=False
        ):
            obs = call_result.get("observation")
            if _market_kind(obs):
                state.saw_market = True
            if _obs_done(obs):
                done_any = True
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": _truncate_obs(obs),
                }
            )

        user_msg = result.get("user_message")
        if isinstance(user_msg, dict):
            content = str(user_msg.get("content") or "")
            if content:
                messages.append({"role": "user", "content": content})
                if _shopper_needs_reply(content):
                    state.await_reply = True
                # After ask-spec, a reply unlocks search
                if state.asked_spec and MODE_ASK_SPEC in modes:
                    pass

        if done_any or any(
            _obs_done(c.get("observation")) for c in (result.get("calls") or [])
        ):
            break

    return dialogue


__all__ = ["agent_main"]
