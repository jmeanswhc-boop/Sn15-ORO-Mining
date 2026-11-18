from __future__ import annotations
import json
import re
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
_SG_MODEL_LIGHT = 'deepseek/deepseek-v3.2'
_SG_MODEL_HEAVY = 'qwen/qwen3.5-397b-a17b'
_SG_FALLBACK_CHAIN = ['google/gemma-4-31b-it', 'openai/gpt-oss-120b', 'deepseek/deepseek-chat-v3.1', _SG_MODEL_LIGHT]
_SG_CHUTES_PAIRS: tuple[tuple[str, str], ...] = (('deepseek-ai/DeepSeek-V3.2-TEE', 'deepseek/deepseek-v3.2'), ('deepseek-ai/DeepSeek-V3.1-TEE', 'deepseek/deepseek-chat-v3.1'), ('deepseek-ai/DeepSeek-V3-0324-TEE', 'deepseek/deepseek-chat-v3-0324'), ('deepseek-ai/DeepSeek-R1-0528-TEE', 'deepseek/deepseek-r1-0528'), ('Qwen/Qwen3-32B-TEE', 'qwen/qwen3-32b'), ('Qwen/Qwen3.5-397B-A17B-TEE', 'qwen/qwen3.5-397b-a17b'), ('Qwen/Qwen3.6-27B-TEE', 'qwen/qwen3.6-27b'), ('google/gemma-4-31B-turbo-TEE', 'google/gemma-4-31b-it'), ('openai/gpt-oss-120b-TEE', 'openai/gpt-oss-120b'), ('XiaomiMiMo/MiMo-V2-Flash-TEE', 'xiaomi/mimo-v2-flash'), ('unsloth/Mistral-Nemo-Instruct-2407-TEE', 'mistralai/mistral-small-2603'), ('zai-org/GLM-5.1-TEE', 'qwen/qwen3.5-397b-a17b'), ('zai-org/GLM-5.2-TEE', 'qwen/qwen3.5-397b-a17b'), ('moonshotai/Kimi-K2.6-TEE', 'qwen/qwen3.5-397b-a17b'), ('moonshotai/Kimi-K2.5-TEE', 'qwen/qwen3.5-397b-a17b'))
_http = ProxyClient(max_retries=2, timeout=120)
FAM_INTENT = 'intent_decomposition'
FAM_RECALL = 'retrieval_recall'
FAM_CONSTRAINT = 'constraint_satisfaction'
FAM_PREFERENCE = 'preference_reasoning'
FAM_RANK = 'ranking'
FAM_RECOVER = 'recovery'
FAM_JUSTIFY = 'justification'
_FAMILY_FINGERPRINTS: list[tuple[str, list[re.Pattern[str]]]] = [(FAM_RANK, [re.compile('\\btrade-?off\\b.{0,30}\\bpriorit|\\bbest[- ]ranked\\b|\\bstrictly worse\\b', re.I), re.compile('\\bcandidate variant|\\bre[- ]?rank(?:ing|ed)?\\b|\\bdominat(?:e|es|ed|ion)\\b', re.I), re.compile('\\d{8,}::\\S+'), re.compile('\\blexico(?:graphic)?\\b|\\bnamed candidates?\\b|\\bclosed set\\b', re.I)]), (FAM_RECOVER, [re.compile('\\bleast[- ]relax|\\brecover\\b.{0,24}\\bunavailable', re.I), re.compile('\\bsame model\\b.{0,48}(?:\\bother\\b|\\banother\\b).{0,12}\\bcolor\\b', re.I), re.compile('\\bexact\\b.{0,20}(?:item|model|color).{0,60}\\bdisappear', re.I), re.compile('\\bnever relax\\b.{0,48}(?:budget|stock|category)|(?:\\bother\\b|\\banother\\b)\\s+color\\s+before.{0,24}(?:\\bother\\b|\\banother\\b)\\s+model|\\bmodel fact\\b.{0,36}(?:title|overrid)', re.I)]), (FAM_JUSTIFY, [re.compile('\\b(?:must|should)\\b.{0,48}\\bjustification\\b|\\bclaims?\\b.{0,24}\\bjustif|\\bground(?:ed)? claims?\\b', re.I), re.compile("\\btell me why\\b|\\bgive me a reason\\b|\\bwhy\\b.{0,16}(?:the one|it'?s|your pick)|\\bstructured justification\\b", re.I)]), (FAM_INTENT, [re.compile('\\bnice[- ]to[- ]have\\b', re.I), re.compile('\\bsubordinate\\b.{0,48}\\b(?:firm|requirement)\\b|\\bpreferred brand\\b|\\bprefer(?:red)? brand\\b', re.I), re.compile('\\binitial\\b.{0,12}(?:cart )?choice.{0,80}(?:soft|prefer|brand)', re.I), re.compile('\\bfirm\\b.{0,96}\\bsoft\\b|\\bsoft\\b.{0,96}\\bfirm\\b', re.I), re.compile('\\bbefore you order\\b|\\bwhich one\\b.{0,24}(?:did|do|should)|\\bover budget\\b.{0,48}(?:recover|re[- ]?search|remove)|\\bafter\\b.{0,12}(?:cart|add).{0,72}(?:budget|unavailable|price)', re.I)]), (FAM_CONSTRAINT, [re.compile('\\bexact(?:ly)?\\b.{0,16}\\bvarian|\\bspecific sku\\b', re.I), re.compile('\\bfirm (?:requirement|constraint)s?\\b|\\bgated (?:spec|requirement)\\b', re.I), re.compile('\\bfirm requirement\\b.{0,24}\\bspec\\b|\\breveal(?:ed)? only when\\b|\\btell you if you ask\\b|\\bif you ask\\b', re.I), re.compile('\\bsell(?:s)? out\\b|\\bsibling\\b|\\bsame title\\b.{0,48}(?:sku|option)|\\bhidden (?:firm )?spec\\b|\\bgated spec\\b', re.I)]), (FAM_PREFERENCE, [re.compile('\\bnot (?:the )?cheapest\\b|\\bsoft (?:axes|preferences?)\\b', re.I), re.compile('\\bintended use\\b|\\bsoft axes\\b', re.I), re.compile('\\brather pay\\b|\\buse[- ]case\\b', re.I), re.compile('\\bclarify\\b.{0,36}\\bpriorit', re.I)]), (FAM_RECALL, [re.compile('\\breformulate\\b|\\bshortlist\\b', re.I), re.compile('\\bdistinct (?:products|identities)\\b', re.I), re.compile('\\bmatches?\\b.{0,24}(?:phrase )?\\"[^\\"]+\\"', re.I), re.compile('\\bcategory_path\\b|\\boff[- ]category\\b', re.I)])]
_ASK_SPEC_RX = re.compile('(?:tell|reveal|share).{0,48}\\bif you ask\\b|\\bif you ask\\b|\\bask\\b.{0,24}(?:exact )?(?:requirement|spec)\\b|\\bwithhold\\b.{0,36}\\bspec\\b', re.I)
_VARIANT_ID_RX = re.compile('\\b(\\d{10,}::[A-Za-z0-9._-]+)\\b')
_PROMPT_CORE = '\nYou are SnowGirl v7, an ORO Bench shopping agent.\nYou solve each episode with the tools in THIS problem only. You generalize:\nno memorized product IDs, no suite answer sheets, no validator/time routing.\n\n# Contract (non-negotiable)\n- Call ONLY tools listed in the episode tool schemas.\n- Keep going until an observation has done=true (or you exhaust max_steps).\n- Obey max_steps and max_calls_per_turn.\n- When max_calls_per_turn > 1, batch independent view/compare/inspect calls.\n- Prefer precise actions over search spam. One focused search beats five vague ones.\n- Never invent catalog facts. If you did not view/inspect it, you do not know it.\n\n# Canonical shopping loop\n1) Parse the goal → firm constraints vs soft preferences (write them mentally).\n2) Discover → search / filter using VALUE-side keywords (not JSON field names).\n3) Inspect → view / compare / inspect_stock on concrete product_id + sku options.\n4) Decide → add_to_cart / remove_from_cart.\n   IMPORTANT: the FIRST add_to_cart is often graded. No sacrificial placeholders.\n5) Dialogue → if the shopper asks anything, message THIS/NEXT turn before cart/order.\n6) React → after price/stock/cart events, re-verify and recover per family rules.\n7) Finish → place_test_order once the pick is verified.\n   Include justification.claims ONLY when the goal/family requires a reason.\n\n# Constraint extraction checklist\nFrom the shopper goal, label every constraint FIRM or SOFT:\n- Category / product type / device authenticity\n- Brand / model / series / identifier\n- Size / capacity / color / material / feature / compatibility\n- Budget / price ceiling (treat stated amounts as firm unless marked soft)\n- In-stock requirement (almost always firm)\n- Any named spec/attribute gate in the goal (compare observed values to what was asked)\n- Use-case preferences (soft): whatever the shopper actually named\nSoft NEVER overrides firm. If soft and firm conflict, firm wins.\n\n# Matching modes (do not mix them up)\n- Literal match: brands, model codes, SKUs/identifiers, uncommon tokens, exact colors.\n- Size/unit wording: keep the shopper\'s own tokens; do not invent conversions.\n- Semantic paraphrase: ONLY for soft preference checks against named use-cases.\n- Spec classes: If Task rules define class floors, follow those; otherwise compare\n  numeric evidence only; never invent thresholds.\n  Unresolved conflicting evidence → do not invent a class.\n\n# Search discipline\nCatalog search matches VALUES in titles/attrs — not schema key names.\nGood keywords: product-type nouns, brand, model codes, distinctive phrase fragments,\nsize/color tokens the shopper wrote.\nBad keywords: dumping "brand", "category", "price", "in_stock" as search terms.\nNormalize lightly: hyphens ≈ spaces; keep quoted phrases intact; drop filler adjectives.\nIf recall is thin: ONE small reformulation (tokens cleanup OR category+filter), then\ninspect. Do not rewrite the query five times while the clock burns.\n\n# Reading observations\n- Prefer structured fields from view/compare/inspect_stock over marketing titles.\n- category_path is the primary anti-accessory signal.\n- Price and stock can change after cart actions — always re-check after events.\n- Option/sku rows under one title can mix compliant and non-compliant variants.\n- When tools return errors, treat them as recovery signals, not as permission to\n  invent a product.\n\n# Device authenticity\nWhen the goal wants a complete product of the asked type:\nReject unless evidence supports it:\n- accessories / spare parts if a full item is required\n- components sold as the whole product\n- review pages, gift cards, mystery boxes, empty listings\n- wrong form-factor substitutions\nUse category_path + title/type evidence together.\n\n# Shopper interventions (critical)\nIf user_message is a question, clarification, hold, or "which one…":\n1) Call message in the current or immediately next turn.\n2) Answer with concrete evidence (brand, key spec, price, stock, product identity).\n3) Only then continue cart/order.\nIgnoring a direct question fails intervention scoring even if the final SKU is gold.\n\n# Market / cart events\nTriggers: stock_changed, price_changed, market_event, failed cart action, sellout,\nbudget breach after reprice, remove_from_cart forced recovery.\nProtocol:\n1) inspect_stock / inspect_cart on the affected SKU\n2) if invalid → remove_from_cart\n3) recover with the family ladder (brand rule / re-rank / least-relax / new product)\n4) place_test_order to CLOSE the event — unfinished events score poorly\n\n# Optional-claim landmine\nShared boilerplate often says justification is optional.\nOn families that do NOT require a reason (especially ranking), OMIT\njustification/claims entirely on place_test_order.\nUntruthful optional claims can zero an otherwise perfect episode.\n\n# Turn budget heuristics\n- Early turns: discover + inspect (and ask hidden specs if needed).\n- Mid turns: commit first cart once evidence is enough — do not hoard steps.\n- Late turns (≥ ~70% of max_steps): stop open search; finish order path.\n- Always leave ≥1–2 turns for inspect_stock + place_test_order after a recovery.\n\n# Anti-patterns (never do these)\n- Cart a cheap placeholder "just to have something", planning to upgrade later.\n- Order from title text alone without viewing the exact option/sku.\n- Keep a pre-event #1 after a market change without re-checking/re-ranking.\n- Jump to other models before trying same-model other colors (when TF6 applies).\n- Submit claim fields that are marketing phrases or meta labels (category/title)\n  instead of catalog keys from the goal and viewed facts.\n- Ask permission to order when the cart already holds a verified valid SKU.\n- Ignore shopper questions while continuing to cart/order.\n'.strip()
_PROMPT_FAMILY: dict[str, str] = {'intent_decomposition': '\n# Family playbook — TF1 Intent Decomposition\nLineage note: third-rank ShoppingBench intent donors (study only).\n\n## What this family tests\nTurn a messy shopper ask into a correct decision sequence: separate firm from soft,\nhonor a preferred brand ONLY when a fully firm-compliant SKU exists, answer\ninterruptions, and recover after budget/stock shocks — with the FIRST cart graded.\n\n## Parse template\nFIRM (must all hold on the ordered option):\n- category / real-device identity\n- the primary spec/attribute gate named in the goal\n- budget ceiling\n- in-stock\nSOFT (nice-to-have / subordinate):\n- preferred brand (and any other soft color/style notes)\n\n## Strategy ladder\n1. Extract firm+soft explicitly before tools.\n2. Dual shortlist:\n   A) preferred-brand + firm tokens\n   B) firm-only (any brand) backup shortlist\n3. view/compare until you can prove compliance on the EXACT option/sku.\n4. FIRST add_to_cart decision:\n   - If a preferred-brand SKU is fully firm-compliant → cart THAT.\n   - Else → cart the best firm-compliant item from the backup shortlist.\n   Never cart a non-compliant brand item "to check later".\n5. Intervention: if the shopper asks which/spec/budget, message with evidence,\n   then continue.\n6. Post-event (price over budget / stockout):\n   remove invalid → re-search under NEW constraints → apply the same brand rule\n   under the new world → inspect → place_test_order.\n7. Omit justification.claims unless the goal explicitly wants a why.\n\n## Worked micro-patterns (generic)\n- Brand available & compliant → first cart brand SKU; later soft downgrades do not matter.\n- Brand available but fails a firm spec/budget → first cart non-brand compliant winner.\n- Brand string present but zero hits → treat soft brand as impossible; proceed with firm.\n- After cart, price jumps above budget → remove; find firm-compliant under new budget;\n  brand again only if still fully compliant.\n\n## Failure modes to avoid\n- Missing an available preferred-brand compliant option on first cart.\n- Stalling forever hunting an impossible brand.\n- Answering interventions only in free text without the message tool.\n- Closing without order after a recovery event.\n'.strip(), 'retrieval_recall': '\n# Family playbook — TF2 Retrieval Recall\nLineage note: third-rank ShoppingBench recall donors (study only).\n\n## What this family tests\nSurface the right candidates without committing after a single weak hit. Phrase\ntitle matches, one reformulation when recall is thin, several distinct\nidentities, and category traps are common variants.\n\n## Strategy ladder\n1. Pass-1 search: distinctive product-type words + brand/model + quoted phrases.\n2. If the hit list is thin/noisy: ONE Pass-2 reformulation\n   (token cleanup OR broader product-type + filter by budget/stock/category).\n3. Build a shortlist of SEVERAL DISTINCT identities\n   (different product_id / model code / brand+title clusters).\n4. For phrase goals: verify the phrase as TITLE tokens on view — brand or\n   description-only hits are insufficient.\n5. Drop accessories / off-category rows using category_path.\n6. view/compare the shortlist; choose best in-stock in-budget match.\n7. place_test_order promptly once verified (no permission wait).\n8. Still handle market events and shopper clarifications if they appear.\n\n## Reformulation recipes (keep small)\n- Drop empty praise adjectives.\n- Keep units and codes (sizes, model fragments).\n- Split hyphenated compounds; try both joined and split forms.\n- Broaden type noun (e.g. add category word) while keeping one distinctive token.\n- Apply filter tools for budget/stock when search alone is noisy.\n\n## Failure modes to avoid\n- Commit after a single weak hit (kills shortlist/recall metrics).\n- Perfect attribute match on the wrong category (accessory trap).\n- Endless reformulation without ever ordering.\n- Ignoring a title-phrase requirement and matching only brand/category.\n'.strip(), 'constraint_satisfaction': '\n# Family playbook — TF3 Constraint Satisfaction\nLineage note: third-rank ShoppingBench constraint donors (study only).\n\n## What this family tests\nEVERY firm requirement must hold on the EXACT ordered SKU/option — including\nhidden specs you must ask for, same-title multi-option traps, and sellout\nsibling traps. Follow Task-rules class floors only when present.\n\n## Strategy ladder\n1. Scan for hidden-spec language ("if you ask", "tell you when asked", withhold).\n   If present: Turn 1 = message asking for the exact gated requirement.\n   Do NOT search until the shopper replies.\n2. After specs are known: search/filter with VALUE tokens for each firm gate.\n3. Never trust a title alone. view the product; select the compliant option/sku;\n   inspect_stock on THAT sku (price + stock + option facts).\n4. If Task rules define class/threshold mappings, follow them exactly.\n   Otherwise compare numeric evidence only; never invent thresholds.\n5. Sellout / invalid after cart:\n   - Do NOT take a sibling option that drops a firm gate.\n   - Move to a NEW product_id that still satisfies ALL firm constraints.\n6. Finish with place_test_order; omit optional claims unless required.\n7. Keep enough turns after events to inspect + order.\n\n## Verification matrix (before order)\nFor each firm constraint C:\n- evidence source = view option row OR inspect_stock field (not title guess)\n- exact sku selected satisfies C\n- after any event, re-verify C still holds\n\n## Failure modes to avoid\n- Searching before asking a gated firm spec.\n- Sibling SKU after sellout (must switch product).\n- Ordering a title that "looks right" while the option row fails the gate.\n- Inventing class floors or thresholds not stated in Task rules.\n- Event fired but never closed with a valid final order.\n'.strip(), 'preference_reasoning': '\n# Family playbook — TF4 Preference Reasoning\nLineage note: third-rank ShoppingBench preference donors (study only).\n\n## What this family tests\nSoft use-case tradeoffs under hard constraints. Cheapest-valid is often wrong.\nFirst cart is graded as the intended preference-aligned purchase.\n\n## Hard vs soft\nHARD (must hold): category, budget, in-stock, real device (not accessory/review).\nSOFT (optimize): whatever use-case axes the shopper named. Do not invent extra axes.\n\n## Strategy ladder\n1. Extract hard constraints and soft axes from the goal.\n2. If soft priorities conflict or are ambiguous: ONE clarifying message, then decide.\n3. Shortlist ≥2 distinct listing_ids that meet HARD constraints.\n4. compare/view them on the SOFT axes using factual attributes (not vibes).\n5. FIRST add_to_cart = the preference-aligned winner (may be slightly dearer if in budget).\n   Do not cart a cheap stub planning to upgrade later.\n6. After reprice/listing change: remove invalid; recover to a DIFFERENT listing_id\n   when needed (not merely another variant of the same listing if recovery needs change).\n7. Re-check stock/price; place_test_order without optional untruthful claims.\n\n## Preference scoring intuition\n- Strong soft-axis match > tiny price savings (while still within budget).\n- Evidence from viewed materials/features beats marketing adjectives.\n- If two items tie on soft axes, prefer the one that more clearly documents the\n  soft requirement; price can break remaining ties.\n\n## Failure modes to avoid\n- Cheapest-valid default → low preference alignment.\n- First cart placeholder → first-cart score fails permanently.\n- Never observing the chosen product.\n- Optional claims that are false → truthful flag fails / reward zeros.\n'.strip(), 'ranking': '\n# Family playbook — TF5 Ranking\nLineage note: third-rank ShoppingBench ranking donors (study only).\n\n## What this family tests\nClosed-set ranking of named candidates by lexicographic tradeoff priority, with\nmandatory post-event re-rank and a clean order payload (usually NO claims).\n\n## Strategy ladder\n1. Parse every named candidate token of the form product_id::sku from the goal.\n   This is a CLOSED set — prefer viewing those IDs directly. Avoid open catalog search.\n2. Batch view/compare ALL candidates before any add_to_cart (use parallel calls).\n3. Rank ONLY by the tradeoff priority the goal states. Never invent extra axes.\n   Never pick an option strictly dominated on every axis.\n4. FIRST add_to_cart = current best AVAILABLE (in-stock, in-budget) candidate.\n5. On stock/price/cart event:\n   a) re-inspect survivors still named/available\n   b) re-rank from scratch with the SAME priority\n   c) cart the NEW #1 (not the pre-event winner if it is no longer best)\n   d) inspect_stock\n   e) place_test_order(product_id, sku) ONLY\n6. If turn index is high and not ordered: force the finish path immediately.\n\n## CRITICAL order rule for TF5\nplace_test_order arguments = product_id + sku only.\nDo NOT attach justification/claims unless the Task rules explicitly demand a reason.\nOptional untruthful claims hard-fail even when the ranked pick is correct.\n\n## Dominance check\nBefore carting candidate X, ensure no other available candidate Y is better or equal\non every higher-priority axis and strictly better on at least one, per the stated order.\n\n## Failure modes to avoid\n- Never calling place_test_order.\n- Keeping pre-event #1 after the market moves.\n- Open-search thrash instead of viewing named IDs.\n- Shipping claims on a ranking order.\n'.strip(), 'recovery': '\n# Family playbook — TF6 Recovery\nLineage note: third-rank ShoppingBench recovery donors (study only).\n\n## What this family tests\nWhen the exact preferred item disappears or breaks constraints after interaction,\nrecover with the LEAST relaxation. Category, budget, and in-stock never relax.\n\n## Immutable firm walls\n- Category / product type\n- Budget ceiling\n- In-stock\n\n## Least-relaxation ladder\nLevel 0 — exact model + exact color (commit when available)\nLevel 1 — same model, other color (only after Level 0 invalid/unavailable)\nLevel 2 — other models (only after Level 1 exhausted)\nNever skip levels. Never relax the immutable walls to "make something fit".\n\n## Strategy ladder\n1. Identify exact model + color targets from the goal.\n2. Prefer structured model facts over conflicting marketing titles.\n3. Commit to Level 0 when in stock and in budget.\n4. On disappearance / budget break after cart:\n   inspect → remove if needed → climb the ladder one level at a time.\n5. Color resolution order: variant/option color → product color field → title last.\n   If multi-color is unresolved, do not invent a color.\n6. Re-check price+stock after every cart change; then place_test_order.\n7. Omit optional untruthful claims.\n\n## Failure modes to avoid\n- Jumping to other models before trying other colors of the same model.\n- Skipping a commit when the preferred exact item was available.\n- Relaxing budget/category/stock to force a match.\n- Trusting a title that contradicts an explicit model fact.\n'.strip(), 'justification': '\n# Family playbook — TF7 Justification\nLineage note: EnvPack claim discipline (no ShoppingBench primary TF7 donor).\n\n## What this family tests\nA valid purchase PLUS structured, fact-backed justification.claims on order.\nWrong claim shape zeros reward even when the product is correct.\n\n## Strategy ladder\n1. Find a valid in-stock in-budget product that meets firm gates.\n2. view/compare (and inspect_stock) so every claim value is grounded BEFORE order.\n3. Build a MINIMAL claim set, then place_test_order with justification.claims.\n\n## Claim shape\nUse catalog attribute keys implied by the goal and by what you viewed.\nWhen those facts exist, include:\n- current price\n- current in_stock\n- the firm spec the shopper actually asked about (that catalog key)\n- at most one extra evidenced reason (prefer brand if present)\n\nDo not invent feature flags. Do not use marketing phrases as field names.\nDo not use meta labels (category, title, summary, evidence, OS) as claims.\n\n## Value rules\n- Claim values must equal observed catalog text (view/inspect).\n- Do not rewrite measurements into a different unit or layout.\n- price/in_stock must match the latest inspect_stock or cart state after events.\n- Keep claims short and factual.\n\n## Post-event claims\nIf price/stock/spec evidence changes after cart, re-view/re-inspect and rebuild\nclaims from fresh facts before ordering.\n\n## Failure modes to avoid\n- Marketing-phrase field names.\n- Invented features.\n- Stale price/stock vs claims.\n- Correct-looking claims on the wrong product.\n'.strip()}
_CASEBOOK_SHARED = '\n# Shared edge-case casebook (all families)\n\n## A. First-cart irrevocability\nSeveral families score the FIRST add_to_cart separately from later fixes.\nTreat first cart as the intended commitment. No sacrificial placeholders.\nIf you are unsure, inspect one more fact — do not cart garbage "for now".\n\n## B. Intervention protocol\nOn shopper user_message that is a question, hold, or clarification:\n1) message tool this turn (or immediately next)\n2) concrete answer with identity + key evidence\n3) only then continue cart/order\nIgnoring a shopper question fails even if the final SKU is otherwise valid.\n\n## C. Market-event protocol\nWhen stock_changed / price_changed / market_event / remove-driven recovery:\n1) inspect_stock / inspect_cart on the affected SKU\n2) if invalid: remove_from_cart\n3) recover per family ladder (brand / re-rank / least-relax / new product)\n4) place_test_order to close the event\nUnfinished events are a common silent failure.\n\n## D. Optional justification landmine\nIf the family does not require a reason, omit justification entirely.\nUntruthful optional claims can zero an otherwise valid episode.\nRanking episodes are especially sensitive.\n\n## E. Device authenticity\nWhen a complete product of the asked type is required, reject:\n- components sold as the whole item\n- accessories / spare parts unless that is the goal\n- review / gift-card / mystery listings\n- wrong form-factor substitutions\nUse category_path + title/type evidence together.\n\n## F. Budget discipline\nAlways re-check price after cart events. Over-budget finals fail even if the\nattribute match is perfect. Compare numeric magnitudes after normalizing\nseparators; do not ignore tax/fee fields if tools expose them as the payable\nprice.\n\n## G. Evidence hierarchy\nHighest trust → lowest trust:\ninspect_stock/cart structured fields → view option/sku rows → compare tables →\nproduct structured attrs → title tokens → marketing blurb.\nNever let a lower-trust source override a higher-trust contradiction.\n\n## H. Parallelism\nWhen max_calls_per_turn allows, batch independent views of shortlist candidates\nin one turn. Do not serialize obvious independent inspections.\n\n## I. Finish discipline\nOnce a verified in-stock in-budget pick exists and no shopper question is pending,\nplace_test_order. Do not stall until max_steps "just in case".\n'.strip()
_CASEBOOK_BY_FAMILY: dict[str, str] = {'intent_decomposition': '\n# TF1 casebook — Intent Decomposition\n\n## Case grid\n1) Brand available + fully compliant → first cart MUST be that brand.\n2) Brand available but fails firm spec → first cart = best other firm-compliant.\n3) Brand string present but zero catalog hits → impossible soft; proceed with firm.\n4) Shopper asks mid-episode "which one did you pick?" → message with brand+spec+price+stock.\n5) Price jumps over budget after cart → remove, re-find firm-compliant under new budget;\n   brand again only if still fully compliant.\n6) Soft brand becomes available later after a bad first cart → too late for first-cart\n   brand score; still finish correctly for final validity.\n7) Accessory brand hit → reject via category_path; keep searching for real device.\n8) Two brand SKUs, only one meets the firm spec → cart the compliant brand SKU, not the other.\n\n## Micro-checklist before first cart\n- [ ] firm category / authenticity ok\n- [ ] firm hardware gate ok on exact option\n- [ ] budget ok (current price)\n- [ ] in stock\n- [ ] preferred brand honored IFF all above hold for that brand\n- [ ] no pending unanswered shopper question\n\n## Micro-checklist before final order\n- [ ] cart item still valid after any event\n- [ ] interventions answered\n- [ ] no optional untruthful claims attached\n'.strip(), 'retrieval_recall': '\n# TF2 casebook — Retrieval Recall\n\n## Case grid\n1) Exact phrase in title required → keep phrase tokens; verify on view.\n2) Thin first page → reformulate once (tokens / category+filter).\n3) Many near-duplicates → count distinct identities, not raw row count.\n4) Accessory pollution → drop via category_path.\n5) Good shortlist found → order without waiting for permission.\n6) Hyphen/space variants → try both if first form misses.\n7) Budget filter available → use it when open search returns mostly over-budget noise.\n8) Shopper clarifies mid-search → message, then adjust keywords.\n\n## Micro-checklist before order\n- [ ] ≥1 reformulation OR strong first hit covering distinctive tokens\n- [ ] several distinct identities inspected (when shortlist is required)\n- [ ] chosen title/category match the goal (phrase/authenticity)\n- [ ] in stock + in budget\n- [ ] not an accessory trap\n'.strip(), 'constraint_satisfaction': '\n# TF3 casebook — Constraint Satisfaction\n\n## Case grid\n1) Hidden firm spec → message first; block search until answered.\n2) Explicit multi-constraint → every constraint on exact SKU.\n3) Same title, mixed options → view options; pick compliant sku; inspect_stock.\n4) Sellout → new product_id, not non-compliant sibling.\n5) Spec class floors → follow Task rules if present; else compare numeric evidence only.\n6) Spec gates → verify on option row, not marketing adjectives.\n7) After ask, shopper gives the gated value → immediately encode it into search tokens.\n8) Event after cart → re-verify ALL firm gates before final order.\n\n## Micro-checklist before order\n- [ ] asked hidden spec if required\n- [ ] exact sku option matches every firm constraint\n- [ ] price + stock verified after any event\n- [ ] no within-product / sibling trap after sellout\n- [ ] Task-rules class floors followed when present (never invent)\n'.strip(), 'preference_reasoning': '\n# TF4 casebook — Preference Reasoning\n\n## Case grid\n1) Soft axes listed → compare ≥2 listing_ids on those axes.\n2) Ambiguous priority → one clarifying message, then decide.\n3) First cart must be preference-aligned (not cheapest stub).\n4) Reprice → recover on a different listing when needed.\n5) Optional claims present by model mistake → strip them.\n6) Soft axis only weakly evidenced → prefer the candidate with clearer factual support.\n7) Hard constraint failure on a soft favorite → discard; soft cannot override hard.\n8) Two soft favorites tie → use secondary soft axis then price, still in budget.\n\n## Micro-checklist before first cart\n- [ ] hard constraints satisfied\n- [ ] soft axes compared on real facts (≥2 candidates when possible)\n- [ ] first cart is the intended preference pick\n- [ ] no pending clarification that you chose to ask but did not wait for\n'.strip(), 'ranking': '\n# TF5 casebook — Ranking\n\n## Case grid\n1) Closed candidate set → view all named product_id::sku.\n2) Lexicographic axes → rank only by stated priority.\n3) Dominated option → never select.\n4) Post-event → re-rank survivors; cart new #1; order.\n5) Claims on order → omit unless required.\n6) One candidate goes out of stock → rank remaining only; do not invent outsiders.\n7) If the goal says to trust an advertised spec value, use that advertised value.\n8) Late turns → force finish; do not re-open search.\n\n## Micro-checklist before order\n- [ ] all named candidates considered (or remaining after event)\n- [ ] current #1 is in cart\n- [ ] dominance check passed\n- [ ] place_test_order has only product_id+sku (usual case)\n'.strip(), 'recovery': '\n# TF6 casebook — Recovery\n\n## Case grid\n1) Exact model+color available → commit first (Level 0).\n2) Exact disappears → same model other color next (Level 1).\n3) No other color → other models last (Level 2).\n4) Model fact vs title conflict → trust model fact.\n5) Never relax category/budget/stock.\n6) Multi-color unresolved → do not invent; gather option evidence.\n7) Price breaks budget after cart → treat as Level invalid; climb legally.\n8) Optional claims → omit unless required; never invent features during recovery.\n\n## Micro-checklist before order\n- [ ] least-relaxation level chosen correctly (no skipped levels)\n- [ ] category/budget/stock still hold\n- [ ] facts grounded in view/inspect\n- [ ] color source trusted in the evidence hierarchy\n'.strip(), 'justification': '\n# TF7 casebook — Justification\n\n## Case grid\n1) Required claims → catalog keys from the goal and viewed facts.\n2) Firm spec field → whatever attribute the shopper actually asked about.\n3) Meta/junk field names → drop (do not invent a replacement key).\n4) Extra reason → at most one evidenced reason (prefer brand).\n5) Post-event → re-verify facts then re-justify.\n6) Forbidden meta fields appear → strip before submit.\n7) Marketing phrase used as field → DROP.\n8) Missing price/in_stock → pull from latest inspect_stock before order.\n\n## Micro-checklist before order\n- [ ] viewed/inspect facts for every claim value (values = catalog text)\n- [ ] no meta/junk field names\n- [ ] price + in_stock + the asked spec present when those facts exist\n- [ ] at most one extra reason, evidenced\n- [ ] no rewritten units or invented features\n'.strip()}
_RECIPE_SEARCH_REFORM = '\n# Recipe — search reformulation (TF2 / thin recall)\n\nPass 1 — Phrase / distinctive tokens\n- Product-type noun + brand/model fragments + quoted phrases intact.\n- Do NOT dump schema keys (brand/category/price) as keywords.\n- Keep uncommon tokens (codes, materials, exact color words).\n\nPass 2 — Token cleanup (one small edit)\n- Drop empty praise adjectives and shopper hedging words.\n- Keep units and codes as the shopper wrote them.\n- Try hyphen↔space variants once.\n\nPass 3 — Category + filter\n- Broaden product-type token slightly.\n- Apply budget/stock/category filters if tools allow.\n- Inspect several distinct hits before committing.\n\nStop rule: once a usable shortlist appears, stop rewriting and move to\nview/compare/order. Reformulation is a scalpel, not a loop.\n'.strip()
_RECIPE_RECOVERY_LADDER = '\n# Recipe — least-relaxation ladder (TF6)\n\nImmutable walls: category, budget, in-stock.\n\nLevel 0 — Exact model + exact color\n- Commit when available; view/inspect to lock facts.\n\nLevel 1 — Exact model + other color\n- Only after Level 0 unavailable/invalid.\n- Resolve color via option → product color → title last.\n- Re-verify stock and price.\n\nLevel 2 — Other models\n- Only after Level 1 exhausted.\n- Still inside immutable walls.\n\nModel-fact rule: explicit model fact > conflicting marketing title.\nDo not invent colors when multi-color evidence is unresolved.\nAfter every climb: inspect → cart → inspect → place_test_order.\n'.strip()
_RECIPE_RANK_CYCLE = '\n# Recipe — ranking cycle (TF5)\n\n1. Parse named product_id::sku candidates (closed set).\n2. Batch view/compare all of them.\n3. Rank by stated lexicographic axes only; drop dominated options.\n4. add_to_cart current #1 (first cart graded).\n5. On market/cart event:\n   a) inspect survivors still available\n   b) re-rank from scratch (same axes)\n   c) cart NEW #1\n   d) inspect_stock\n   e) place_test_order(product_id, sku) with NO justification\n6. If turn budget is high (≥8) and not ordered: force finish path now.\n7. Never open-search the wider catalog unless named IDs are impossible to view.\n'.strip()
_RECIPE_HIDDEN_SPEC = '\n# Recipe — hidden firm spec (TF3)\n\nTurn 1: message asking for the withheld firm requirement\n(whatever attribute the goal withholds — do not guess the value).\nWait for shopper user_message.\nOnly then: search/filter using the revealed value + other firm tokens.\nVerify the exact SKU option against ALL firm constraints (matrix).\nIf sellout: new product_id that still satisfies the gated spec — no sibling trap.\nKeep turns for post-event re-verify + order.\n'.strip()
_RECIPE_BRAND_FIRST = '\n# Recipe — preferred-brand first cart (TF1)\n\n1. Extract firm constraints + preferred brand soft.\n2. Dual retrieve: (brand+firm) and (firm-only backup).\n3. Prove compliance on exact options via view/inspect.\n4. If a brand SKU is fully firm-compliant → first cart that SKU.\n5. Else → first cart best firm-compliant non-brand item.\n6. On intervention: message with evidence, then continue.\n7. On budget/stock event: remove → recover under new constraints with same brand rule.\n8. Omit optional claims unless required.\n'.strip()
_RECIPE_CLAIMS = '\n# Recipe — justification claims (TF7)\n\nAfter view/inspect_stock on the final SKU, build claims from observed facts:\n1. the firm spec key the goal actually named\n2. price (current, post-event)\n3. in_stock (current, post-event)\n4. at most one extra evidenced reason (prefer brand)\n\nDrop meta/junk field names. Values must equal observed catalog text.\nNever invent features or rewrite measurements.\n'.strip()
_RECIPE_PREFERENCE = '\n# Recipe — preference / soft-axis compare (TF4)\n\n1. Extract firm constraints + soft use-case axes from the goal.\n2. Shortlist several firm-compliant SKUs; do not default to cheapest.\n3. Optionally one clarifying message if priorities conflict.\n4. compare/view against soft axes (fit, features, form, environment) before first cart.\n5. First cart = preference-aligned winner under firm constraints.\n6. On reprice / listing change: re-evaluate survivors; change listing_id when needed.\n7. Re-verify stock/price; place_test_order without optional untruthful claims.\n'.strip()
_RECIPES_BY_FAMILY: dict[str, str] = {'intent_decomposition': _RECIPE_BRAND_FIRST, 'retrieval_recall': _RECIPE_SEARCH_REFORM, 'constraint_satisfaction': _RECIPE_HIDDEN_SPEC, 'preference_reasoning': _RECIPE_PREFERENCE, 'ranking': _RECIPE_RANK_CYCLE, 'recovery': _RECIPE_RECOVERY_LADDER, 'justification': _RECIPE_CLAIMS}
_EMPTY_TOOL_NUDGES = ('SnowGirl: no tool call received. You must call a supplied tool now. If you have no shortlist yet, start with search or filter using value-side keywords from the shopper goal (not JSON field names).', 'SnowGirl: still no tool call. Advance the episode deliberately: search/filter → view/compare concrete SKUs → message if the shopper asked anything → add_to_cart only a verified pick → place_test_order once stock and budget are confirmed. Do not answer in plain text only.')
_HINT_ASK_SPEC = 'SnowGirl coach · TF3 hidden spec: a firm hardware requirement is withheld until you ask. Call message THIS turn requesting that exact requirement. Do not search/filter until the shopper replies with the gated value.'
_HINT_REPLY = 'SnowGirl coach · intervention: the shopper just spoke. Call message THIS turn with a concrete answer (identity + key firm evidence + price/stock). Do not cart or order until that reply is sent.'
_HINT_RECOVERY = 'SnowGirl coach · TF6 recovery: failure or market shift ({error}). Climb the least-relaxation ladder without skipping levels — exact model+color → same model other color → other models last. Never relax category, budget, or in-stock. Re-inspect after each cart change.'
_HINT_RANK_OPEN = 'SnowGirl coach · TF5 open: extract every named product_id::sku, batch view/compare the closed set, rank ONLY on the stated lexicographic axes, then first-cart the current #1. place_test_order takes product_id+sku only — omit justification/claims.'
_HINT_RANK_RERANK = 'SnowGirl coach · TF5 re-rank: catalog/cart state moved. Re-inspect surviving named candidates, rebuild the ranking from scratch on the same axes, cart the NEW #1 (not the pre-event winner if demoted), inspect_stock, then place_test_order(product_id, sku) only.'
_HINT_RANK_FINISH = 'SnowGirl coach · TF5 finish: stop open search now. Re-rank in-stock named candidates, ensure the current #1 is carted, and place_test_order immediately with product_id+sku only.'
_HINT_CLAIMS = 'SnowGirl coach · TF7 claims: justification.claims use catalog attribute keys from the goal and from viewed facts — price, in_stock, the asked spec, and at most one evidenced extra reason (prefer brand). Forbidden: category, title, OS, summary, evidence. Values must equal observed catalog text. Do not invent features or rewrite measurements.'
_HINT_BRAND = 'SnowGirl coach · TF1 brand/first-cart: maintain a brand+firm shortlist and a firm-only backup. If a preferred-brand SKU is fully firm-compliant, it MUST win the first cart; otherwise cart the best firm-compliant any-brand item. Answer shopper questions with message before further cart/order.'
_HINT_PREF = 'SnowGirl coach · TF4 preference: compare ≥2 hard-compliant candidates on the stated soft use-case axes before any add_to_cart. First cart = the preference-aligned winner (may be slightly dearer if still in budget), never a cheap placeholder.'
_HINT_RECALL = 'SnowGirl coach · TF2 recall: run ONE small reformulation (token cleanup or category+filter) so several distinct product identities appear, verify title/category authenticity, then proceed to order — do not keep rewriting.'
_HINT_MARKET = 'SnowGirl coach · market ({kind}): re-check price/stock on the carted SKU via inspect_stock/cart. If invalid, remove_from_cart and recover per the active family ladder, then finish with a valid place_test_order so the event is closed.'
_SG_CLAIM_DENY = frozenset({'category', 'category_path', 'device_operating_system', 'device os', 'os', 'title', 'summary', 'evidence', 'reason'})
_SG_FAMILY_PRIORITY = ((FAM_RANK, 7), (FAM_RECOVER, 6), (FAM_JUSTIFY, 5), (FAM_CONSTRAINT, 4), (FAM_INTENT, 3), (FAM_PREFERENCE, 2), (FAM_RECALL, 1))

def _shopper_goal_text(query: str) -> str:
    text = query or ''
    marker = 'Task rules:'
    cut = text.find(marker)
    return text if cut < 0 else text[:cut]

def _classify_family(query: str) -> str:
    blob = query or ''
    tally: dict[str, int] = {}
    for (family_id, patterns) in _FAMILY_FINGERPRINTS:
        n = 0
        for rx in patterns:
            if rx.search(blob):
                n += 1
        if n:
            tally[family_id] = n
    low = blob.lower()
    if 'justification is optional' in low and FAM_JUSTIFY in tally:
        head = _shopper_goal_text(blob)
        if not re.search('\\bwhy\\b|reason|justify', head, re.I):
            tally.pop(FAM_JUSTIFY, None)
    if not tally:
        return FAM_RECALL
    rank = dict(_SG_FAMILY_PRIORITY)
    winner = max(tally.items(), key=lambda item: (item[1], rank.get(item[0], 0)))
    return winner[0]

def _episode_tags(family: str, query: str) -> dict[str, bool]:
    head = _shopper_goal_text(query)
    justify_goal = bool(re.search('\\bwhy\\b|structured justification', head, re.I))
    non_justify = family != FAM_JUSTIFY
    return {'needs_ask_spec': family == FAM_CONSTRAINT and bool(_ASK_SPEC_RX.search(head)), 'has_named_variants': bool(_VARIANT_ID_RX.search(query or '')), 'needs_justification': family == FAM_JUSTIFY or justify_goal, 'omit_optional_claims': non_justify}

def _goal_needs_ask(query: str) -> bool:
    return bool(_ASK_SPEC_RX.search(_shopper_goal_text(query)))

def _extract_named_skus(query: str) -> list[str]:
    found = _VARIANT_ID_RX.findall(query or '')
    ordered: list[str] = []
    seen: set[str] = set()
    for token in found:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered

def _resolve_openrouter_id(raw: str) -> str:
    key = (raw or '').strip()
    if not key:
        return _SG_MODEL_HEAVY
    for (chutes_id, openrouter_id) in _SG_CHUTES_PAIRS:
        if key == chutes_id or key.endswith(chutes_id.split('/', 1)[-1]):
            key = openrouter_id
            break
    low = key.lower()
    if 'kimi' in low or 'glm' in low or low.startswith('z-ai/') or low.startswith('moonshotai/'):
        return _SG_MODEL_HEAVY
    return key

def _pick_model(family: str) -> str:
    forced = (getenv('SANDBOX_MODEL') or '').strip()
    if forced:
        return _resolve_openrouter_id(forced)
    return _SG_MODEL_LIGHT if family == FAM_RECALL else _SG_MODEL_HEAVY

def _canon_claim_field(field: str, query: str) -> str:
    del query
    raw = str(field or '').strip()
    if raw.lower() in _SG_CLAIM_DENY:
        return ''
    return raw

def _canon_claim_value(field: str, value: Any) -> str:
    text = '' if value is None else str(value).strip()
    if field == 'in_stock':
        return 'true' if text.lower() in {'true', '1', 'yes', 'in_stock'} else text
    return text

def _scrub_order_args(tool_name: str, args: dict[str, Any], *, family: str, query: str, tags: dict[str, bool]) -> dict[str, Any]:
    payload = dict(args)
    if tool_name != 'place_test_order':
        return payload
    head = _shopper_goal_text(query)
    wants_why = bool(re.search('\\bwhy\\b|justif|reason|tell me why', head, re.I))
    if family == FAM_RANK and (not wants_why):
        payload.pop('justification', None)
        return payload
    if family != FAM_JUSTIFY:
        if tags.get('omit_optional_claims') and (not wants_why):
            payload.pop('justification', None)
        return payload
    block = payload.get('justification')
    if not isinstance(block, dict):
        return payload
    raw_claims = block.get('claims')
    if not isinstance(raw_claims, list):
        return payload
    kept: list[dict[str, str]] = []
    used: set[str] = set()
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        field = _canon_claim_field(str(item.get('field') or ''), query)
        value = _canon_claim_value(field, item.get('value'))
        if not field or value == '':
            continue
        fl = field.lower()
        if fl in _SG_CLAIM_DENY:
            continue
        if fl in used:
            continue
        used.add(fl)
        kept.append({'field': field, 'value': value})
    payload['justification'] = {'claims': kept}
    return payload

def _shopper_needs_reply(text: str) -> bool:
    blob = (text or '').strip().lower()
    if not blob:
        return False
    if '?' in blob:
        return True
    cues = ('^\\s*which\\b', '^\\s*what\\b', '^\\s*does\\b', '^\\s*did\\b', '\\bcan you\\b', '\\bcould you\\b', '^\\s*wait\\b', '\\bhold on\\b', '\\bbefore you\\b', '\\bre-?check\\b', '\\bgot anything\\b')
    return any((re.search(pat, blob) for pat in cues))

def _read_market_kind(obs: Any) -> str:
    if not isinstance(obs, dict):
        return ''
    if obs.get('stock_changed'):
        return 'stock_changed'
    if obs.get('price_changed'):
        return 'price_changed'
    if obs.get('event'):
        return str(obs.get('event'))
    for key in ('market_event',):
        node = obs.get(key)
        if isinstance(node, dict):
            return str(node.get('kind') or key)
        if node:
            return key
    nested = obs.get('observation')
    if isinstance(nested, dict):
        node = nested.get('market_event')
        if isinstance(node, dict):
            return str(node.get('kind') or 'market_event')
        if node:
            return 'market_event'
    return ''

def _list_tool_names(tools: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for entry in tools:
        if not isinstance(entry, dict):
            continue
        fn = entry.get('function')
        if isinstance(fn, dict) and fn.get('name'):
            out.append(str(fn['name']))
        elif entry.get('name'):
            out.append(str(entry['name']))
    return out

def _compose_system(family: str, policy: dict[str, Any], query: str, tags: dict[str, bool]) -> str:
    tool_names = _list_tool_names(policy.get('tools') or [])
    fam_text = _PROMPT_FAMILY.get(family, _PROMPT_FAMILY[FAM_RECALL])
    named = _extract_named_skus(query)
    named_block = ''
    if family == FAM_RANK and named:
        named_block = 'Closed candidate set (inspect each; re-rank after events):\n' + ', '.join(named) + '\n'
    tag_bits = f"needs_ask_spec={tags.get('needs_ask_spec')} has_named_variants={tags.get('has_named_variants')} needs_justification={tags.get('needs_justification')} omit_optional_claims={tags.get('omit_optional_claims')}"
    parts = [_PROMPT_CORE, fam_text, _CASEBOOK_SHARED, _CASEBOOK_BY_FAMILY.get(family, ''), _RECIPES_BY_FAMILY.get(family, ''), named_block, '# Runtime', f'Detected task family: {family}', f'Tags: {tag_bits}', f"Available tools: {(', '.join(tool_names) if tool_names else '(see tool schemas)')}", f"max_steps={policy.get('max_steps')} max_calls_per_turn={policy.get('max_calls_per_turn', 1)}"]
    return '\n\n'.join((p for p in parts if p))

def _parse_tool_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    fn = tool_call.get('function') if isinstance(tool_call, dict) else None
    raw: Any = None
    if isinstance(fn, dict):
        raw = fn.get('arguments')
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        raise ValueError('tool arguments must be a JSON object')
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        parsed = {}
    if isinstance(parsed, dict):
        return parsed
    raise ValueError('tool arguments must be a JSON object')

def _obs_to_text(obs: Any, *, cap: int=6500) -> str:
    try:
        encoded = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        encoded = str(obs)
    if len(encoded) <= cap:
        return encoded
    return f'{encoded[:cap - 16]}...<truncated>'

def _names_from_calls(tool_calls: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for call in tool_calls:
        try:
            out.append(str(call['function']['name']))
        except (KeyError, TypeError, AttributeError):
            out.append('?')
    return out

def _fn_name(call: dict[str, Any]) -> str:
    try:
        return str(call['function']['name'])
    except (KeyError, TypeError, AttributeError):
        return ''

def _fn_id(call: dict[str, Any], fallback: str) -> str:
    try:
        cid = call.get('id')
        return str(cid) if cid else fallback
    except (AttributeError, TypeError):
        return fallback

def _turn_coach(*, family: str, turn: int, last_tool_names: list[str], last_obs: list[Any], tags: dict[str, bool], await_reply: bool, saw_market_event: bool, ordered: bool, first_cart_done: bool, asked_hidden_spec: bool, search_count: int, viewed_named: bool) -> str | None:
    used = ' '.join(last_tool_names)
    if await_reply:
        return _HINT_REPLY
    if tags.get('needs_ask_spec') and turn == 1 and (not asked_hidden_spec) and ('message' not in used):
        return _HINT_ASK_SPEC
    if family == FAM_RANK and (not ordered):
        if saw_market_event and turn >= 2 and ('place_test_order' not in used):
            return _HINT_RANK_RERANK
        if turn >= 8:
            return _HINT_RANK_FINISH
        if turn == 1:
            return _HINT_RANK_OPEN
        if turn >= 2 and (not first_cart_done) and (viewed_named or 'view' in used or 'compare' in used):
            return 'SnowGirl coach · TF5: candidates inspected — add_to_cart only the current #1 (first cart is graded).'
        if first_cart_done and turn >= 3:
            return 'SnowGirl coach · TF5: finish with place_test_order(product_id, sku) only — leave justification off the payload.'
    if family == FAM_INTENT and turn == 1:
        return _HINT_BRAND
    if family == FAM_PREFERENCE and turn <= 2 and ('compare' not in used) and (not first_cart_done):
        return _HINT_PREF
    if family == FAM_JUSTIFY and (not ordered):
        if turn in (2, 5) or (turn >= 2 and 'view' not in used and ('compare' not in used)):
            return _HINT_CLAIMS
    bootstrap_tools = ('search', 'filter', 'view', 'compare', 'message')
    if turn <= 2 and (not any((name in used for name in bootstrap_tools))):
        return 'SnowGirl coach: open with search/filter on value-side keywords from the shopper goal, then view the strongest hits.'
    if family == FAM_RECALL and turn == 3 and (search_count >= 1):
        return _HINT_RECALL
    for obs in last_obs:
        if isinstance(obs, dict) and obs.get('error') and (family == FAM_RECOVER):
            return _HINT_RECOVERY.format(error=str(obs.get('error')))
    if family == FAM_CONSTRAINT and saw_market_event and (not ordered):
        return 'SnowGirl coach · TF3 sellout: switch to a NEW product_id that still meets every firm constraint — sibling SKUs that drop a firm spec are invalid.'
    if family == FAM_RECOVER and saw_market_event:
        return 'SnowGirl coach · TF6 ladder: same model / other color before other models. Never relax category, budget, or in-stock.'
    if saw_market_event and (not ordered) and (turn >= 2):
        return _HINT_MARKET.format(kind='stock/price/cart')
    return None

def _chat_with_tools(*, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_calls: int) -> tuple[str, list[dict[str, Any]]]:
    queue: list[str] = [model]
    for candidate in _SG_FALLBACK_CHAIN:
        if candidate not in queue:
            queue.append(candidate)
    failure: Exception | None = None
    for model_id in queue[:3]:
        payload = {'model': model_id, 'messages': messages, 'tools': tools, 'tool_choice': 'required', 'temperature': 0}
        response = _http.post('/inference/chat/completions', json_data=payload)
        if response is None:
            failure = RuntimeError(f'empty inference response model={model_id}')
            continue
        try:
            msg = response['choices'][0]['message']
        except (KeyError, IndexError, TypeError) as exc:
            failure = exc
            continue
        content = msg.get('content') or ''
        calls = list(msg.get('tool_calls') or [])[:max_calls]
        return (content, calls)
    raise RuntimeError(f'SnowGirl inference exhausted fallbacks ({failure})')

def _opening_coaches(family: str, tags: dict[str, bool]) -> list[dict[str, Any]]:
    extras: list[dict[str, Any]] = []
    if tags.get('needs_ask_spec'):
        extras.append({'role': 'user', 'content': _HINT_ASK_SPEC})
    if family == FAM_RANK:
        extras.append({'role': 'user', 'content': _HINT_RANK_OPEN})
    if family == FAM_JUSTIFY:
        extras.append({'role': 'user', 'content': _HINT_CLAIMS})
    if family == FAM_INTENT:
        extras.append({'role': 'user', 'content': _HINT_BRAND})
    return extras

def _note_named_view(args: dict[str, Any], named: set[str]) -> bool:
    if not named:
        return False
    pid = str(args.get('product_id') or '')
    sku = str(args.get('sku') or '')
    if pid and sku and (f'{pid}::{sku}' in named):
        return True
    if pid and any((token.startswith(pid) for token in named)):
        return True
    return bool(pid)

def _episode_done(result_calls: list[Any]) -> bool:
    for call in result_calls:
        if not isinstance(call, dict):
            continue
        obs = call.get('observation')
        if isinstance(obs, dict) and obs.get('done'):
            return True
    return False

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    env = problem_data['environment']
    binding = env['binding']
    policy = env['policy_view']
    query = str(policy.get('query') or problem_data.get('query') or '')
    problem_id = str(problem_data.get('problem_id', problem_data.get('id', 'problem')))
    max_calls = max(1, int(policy.get('max_calls_per_turn', 1)))
    max_steps = max(1, int(policy.get('max_steps', 20)))
    tools = policy.get('tools') or []
    family = _classify_family(query)
    tags = _episode_tags(family, query)
    model = _pick_model(family)
    messages: list[dict[str, Any]] = [{'role': 'system', 'content': _compose_system(family, policy, query, tags)}, {'role': 'user', 'content': query}, *_opening_coaches(family, tags)]
    transcript: list[dict[str, Any]] = [{'role': 'meta', 'snowgirl': {'version': 7, 'family': family, 'tags': tags, 'model': model, 'provider': 'openrouter', 'refs': 'third_rank_per_tf', 'shell': 'snowgirl_native'}}]
    state = {'await_reply': False, 'saw_market_event': False, 'ordered': False, 'first_cart_done': False, 'asked_hidden_spec': False, 'search_count': 0, 'viewed_named': False, 'last_tools': [], 'last_obs': []}
    named = set(_extract_named_skus(query))
    for turn in range(1, max_steps + 1):
        coach = _turn_coach(family=family, turn=turn, last_tool_names=state['last_tools'], last_obs=state['last_obs'], tags=tags, await_reply=state['await_reply'], saw_market_event=state['saw_market_event'], ordered=state['ordered'], first_cart_done=state['first_cart_done'], asked_hidden_spec=state['asked_hidden_spec'], search_count=state['search_count'], viewed_named=state['viewed_named'])
        if coach:
            messages.append({'role': 'user', 'content': coach})
        content = ''
        tool_calls: list[dict[str, Any]] = []
        for attempt in range(len(_EMPTY_TOOL_NUDGES) + 1):
            (content, tool_calls) = _chat_with_tools(model=model, messages=messages, tools=tools, max_calls=max_calls)
            if tool_calls:
                break
            transcript.append({'role': 'assistant', 'content': content})
            nudge = _EMPTY_TOOL_NUDGES[min(attempt, len(_EMPTY_TOOL_NUDGES) - 1)]
            messages.append({'role': 'assistant', 'content': content})
            messages.append({'role': 'user', 'content': nudge})
        else:
            raise RuntimeError('SnowGirl: model never produced a tool call')
        messages.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls})
        batch_id = f'{problem_id}-sg-{turn}'
        calls_out: list[dict[str, Any]] = []
        for (idx, call) in enumerate(tool_calls, start=1):
            name = _fn_name(call)
            if not name:
                raise RuntimeError('SnowGirl: tool call missing function name')
            args = _scrub_order_args(name, _parse_tool_args(call), family=family, query=query, tags=tags)
            fn = call.get('function')
            if isinstance(fn, dict):
                fn['arguments'] = json.dumps(args)
            calls_out.append({'call_id': f'{batch_id}-{idx}', 'action': {'name': name, 'args': args}})
            if name in ('search', 'filter'):
                state['search_count'] += 1
            elif name == 'message':
                state['await_reply'] = False
                state['asked_hidden_spec'] = True
            elif name == 'add_to_cart':
                state['first_cart_done'] = True
            elif name == 'remove_from_cart':
                state['saw_market_event'] = True
            elif name == 'place_test_order':
                state['ordered'] = True
            elif name in ('view', 'compare'):
                if _note_named_view(args, named):
                    state['viewed_named'] = True
        env_body = {**binding, 'call_id': batch_id, 'idempotency_key': batch_id, 'turn': turn, 'calls': calls_out}
        result = _http.post('/environment/call', json_data=env_body)
        if result is None:
            raise RuntimeError('SnowGirl: /environment/call returned empty')
        transcript.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls, 'environment_result': result, 'snowgirl_turn': {'family': family, 'turn': turn, 'state': {k: state[k] for k in ('await_reply', 'saw_market_event', 'ordered', 'first_cart_done', 'asked_hidden_spec', 'search_count', 'viewed_named')}}})
        state['last_tools'] = _names_from_calls(tool_calls)
        state['last_obs'] = []
        result_calls = result.get('calls') or []
        for (idx, (tool_call, call_result)) in enumerate(zip(tool_calls, result_calls), start=1):
            obs = call_result.get('observation') if isinstance(call_result, dict) else None
            state['last_obs'].append(obs)
            messages.append({'role': 'tool', 'tool_call_id': _fn_id(tool_call, f'{batch_id}-{idx}'), 'content': _obs_to_text(obs)})
            kind = _read_market_kind(obs)
            if kind:
                state['saw_market_event'] = True
                if not state['ordered']:
                    messages.append({'role': 'user', 'content': _HINT_MARKET.format(kind=kind)})
        spoken_msg = result.get('user_message')
        if spoken_msg and spoken_msg.get('content'):
            spoken = str(spoken_msg['content'])
            messages.append({'role': 'user', 'content': spoken})
            if _shopper_needs_reply(spoken):
                state['await_reply'] = True
        errs = [obs.get('error') for obs in state['last_obs'] if isinstance(obs, dict) and obs.get('error')]
        if errs and family == FAM_RECOVER:
            messages.append({'role': 'user', 'content': _HINT_RECOVERY.format(error=errs[0])})
        marketish = bool(errs) or any((_read_market_kind(obs) for obs in state['last_obs']))
        if family == FAM_RANK and state['saw_market_event'] and (not state['ordered']) and marketish:
            messages.append({'role': 'user', 'content': _HINT_RANK_RERANK})
        if _episode_done(result_calls):
            break
    return transcript
__all__ = ['agent_main']
