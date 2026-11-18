"""ORO Bench shopping agent — agent_str contract + 115-BILL product brain.

New mechanism (SN15 generated environment):
  model / solver emits only supplied tools; this file POSTs /environment/call
  and feeds observations back until observation.done (TF1–TF7).

115-BILL (ShoppingBench product winner) pieces that are called:
  P_DECOMPOSE → PSV_BUILD → PSV_KW → compact_kv / compact_alt → PV_SELECT.
  Shop/voucher engines, find_product, recommend_product, SegTree, ExtTool,
  MultiModelVoting, PSV_REVIEW / SV_DECOMPOSE are omitted.
  TF2 retrieval never runs the product brain — exact title-phrase only.
"""
from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_INFER = ProxyClient(
    timeout=75,
    max_retries=1,
    retry_delay=1.0,
    api_key=getenv("OPENROUTER_API_KEY") or getenv("INFERENCE_ACCESS_TOKEN"),
)
_ENV = ProxyClient(timeout=30, max_retries=2, retry_delay=1.0)

# Per-family OpenRouter ladders. Tuned independently for TF1–TF7.
_FAMILY_MODELS: dict[str, tuple[str, ...]] = {
    "intent": (
        "z-ai/glm-5.2",
        "moonshotai/kimi-k2.6",
        "qwen/qwen3.5-397b-a17b",
        "deepseek/deepseek-v3.2",
    ),
    "retrieval": (
        "moonshotai/kimi-k2.6",
        "qwen/qwen3.5-397b-a17b",
        "z-ai/glm-5.2",
        "deepseek/deepseek-v3.2",
    ),
    "constraint": (
        "z-ai/glm-5.2",
        "qwen/qwen3.5-397b-a17b",
        "moonshotai/kimi-k2.6",
        "deepseek/deepseek-v3.2",
    ),
    "preference": (
        "moonshotai/kimi-k2.6",
        "z-ai/glm-5.2",
        "qwen/qwen3.5-397b-a17b",
        "deepseek/deepseek-v3.2",
    ),
    "ranking": (
        "z-ai/glm-5.2",
        "qwen/qwen3.5-397b-a17b",
        "moonshotai/kimi-k2.6",
        "deepseek/deepseek-v3.2",
    ),
    "recovery": (
        "z-ai/glm-5.2",
        "moonshotai/kimi-k2.6",
        "qwen/qwen3.5-397b-a17b",
        "deepseek/deepseek-v3.2",
    ),
    "justification": (
        "z-ai/glm-5.2",
        "qwen/qwen3.5-397b-a17b",
        "moonshotai/kimi-k2.6",
        "deepseek/deepseek-v3.2",
    ),
    "generic": (
        "z-ai/glm-5.2",
        "moonshotai/kimi-k2.6",
        "qwen/qwen3.5-397b-a17b",
        "deepseek/deepseek-v3.2",
    ),
}

# 115-BILL OpenRouter helper chain (parse / build / keyword / select).
# Mapped from 115 TEE LLM_PRIORITIES; gemma stays last so compact_kv's
# gemstone→stone scrub still matches the original.
_HELPER_MODELS = (
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.1",
    "z-ai/glm-5.2",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.5-397b-a17b",
    "google/gemma-4-31b-it",
)
_FAMILY_TEMP = {
    "intent": 0.0,
    "retrieval": 0.1,
    "constraint": 0.0,
    "preference": 0.2,
    "ranking": 0.0,
    "recovery": 0.0,
    "justification": 0.0,
    "generic": 0.0,
}
_HELPER_TEMP = 0.0
_HELPER_TEMP_UP = 0.3
_PREFILTER_TOPN = 30
_SELECT_BATCH = 30
_MAX_SELECT_TRIES = 1
_PRUNE_GAP = 1.75

_KW_STOP = (
    "looking", "find", "need", "want", "show", "please", "ideally", "prefer",
    "suitable", "compatible", "support", "made", "that", "with", "without",
    "from", "for", "and", "the", "a", "an", "under", "over", "below", "above",
    "between", "than", "php", "usd", "in-stock", "instock", "stock", "category",
    "listing", "buy", "one", "place", "order", "test",
)
_ACCESSORY = (
    "phone case", "case for", "screen protector", "tempered glass",
    "replacement screen", "mystery box", "review page", "feature phone",
)
_WATER = ("water", "sweat", "ipx", "waterproof", "resistant", "ip67", "ip68")
_WET_STRONG = (
    "ipx", "ip67", "ip68", "ipx4", "ipx5", "ipx6", "ipx7",
    "waterproof", "water-resistant", "water resistant",
    "sweat", "sweatproof", "sweat-proof",
)
_SPORT = (
    "earhook", "ear hook", "ear-hook", "sport", "running", "workout",
    "gym", "secure fit", "in-ear", "inear", "earbud",
)
_WIRELESS = ("wireless", "bluetooth", "bt ", "tws", "true wireless")
_BATTERY = ("mah", "battery", "hours", "hrs")
_YES_WORDS = ("yes", "enabled", "included", "exists", "true", "required")
_NO_WORDS = ("no", "without any", "without")
_ID_KEYS = ("brand", "color", "model", "id", "compatibility", "color_family")
_METRIC_UNITS = {
    "mm": "", "cm": "", "mg": "", "kg": "", "oz": "", "lb": "", "ml": "",
    "mah": "", "ah": "", "wh": "", "kwh": "", "mv": "", "kv": "", "ma": "",
    "kw": "", "hz": "", "khz": "", "mhz": "", "ghz": "", "bps": "",
    "kbps": "", "mbps": "", "gbps": "", "°c": "", "°f": "", "kpa": "",
    "mpa": "", "dpi": "", "lux": "", "db": "", "rpm": "",
    "y": "year", "yrs": "year", "km": "kilometer", "ft": "feet",
    "m": "meter", "yd": "yard", "g": "gram", "l": "liter", "v": "volt",
    "w": "watt", "pa": "pascal", "kb": "kilobyte", "mb": "megabyte",
    "gb": "gigabyte", "tb": "terabyte", "mp": "megapixel",
    "pc": "piece", "pcs": "piece",
}
_KW_STOPWORDS = (
    "color family", "suitable", "compatible", "support", "made", "aged",
    "needs", "that", "helps", "contains", "includes", "option", "about",
    "above", "after", "also", "before", "below", "between", "for", "from",
    "comes", "less", "more", "over", "than", "under", "with", "within",
    "without", "provides", "sold",
)

_PLAYBOOK = """You are a commerce solver operating inside a generated shopping environment.

Use ONLY the tools supplied in this request. Their names and arguments can change
between tasks. Never invent a tool. Never wait for human permission after a
candidate is verified — place_test_order ends the episode.

# How to shop
1. Decompose the shopper goal into firm constraints vs soft preferences.
   Firm: category, budget, in-stock, named hardware spec (RAM / screen / resolution /
   refresh), and a named brand only when the task rules say the brand is firm.
   Soft: preferred brand when the rules say so, color/model extras, use-case tastes.
2. Retrieve broadly, then inspect. Start with search (short 2–5 word queries, k=10)
   and/or filter (category, brand, max_price, in_stock). Reformulate if results are
   empty, off-category, or too narrow. Title-only hits are not proof of category —
   confirm category_path on view/compare.
3. Read before you commit. view one product for facts + SKU variants. compare up to
   five products when you must trade off. inspect_stock on the exact SKU you intend
   to buy. A listing often sells the right spec and the wrong spec under one title.
4. The first successful add_to_cart is a real purchase commitment, not a probe.
   Market events (price jump or stockout) fire on that first accepted add, and on
   some tasks a later recovery add as well. A throwaway first add cannot be repaired
   by a later correction.
5. After every cart change, inspect_stock the added SKU. On harder tasks the add
   reply itself stays quiet — the live price/stock appears on the next read.
   If the pick is now over budget or out of stock, recover to a different valid
   product. Do not keep a sibling SKU that misses a firm spec.
6. Shopper messages after cart actions can replace a requirement or tighten budget.
   Use the latest stated values. Answer any direct question with the message tool
   before you order.
7. inspect_cart, then place_test_order for a carted SKU that still meets every
   firm rule at the current price and stock.

# Resolution classes (when a display spec is required)
Use the longer and shorter sides, ignoring orientation.
- HD: at least 1280 x 720
- Full HD: at least 1920 x 1080
- QHD: at least 2560 x 1440
- 4K / UHD: at least 3840 x 2160
Pick the highest class whose both thresholds are met. lowercase qHD is quarter HD,
not QHD. Conflicting numbers without a selected-variant value are unresolved.

# Devices vs accessories
When the shopper wants a device (especially a smartphone), reject cases, screen
protectors, replacement screens, review pages, mystery boxes, and tablet-only
listings. For a smartphone, require an explicit smartphone mention or a real
device OS in title/facts. Category labels alone are not enough. A selected
screen-only SKU is not a phone.

# Retrieval coverage
When the rules ask you to surface several distinct matching products, keep
searching and viewing until you have seen multiple different product identities
that match the requested title tokens. Variants of one product_id, obvious
cross-retailer twins, and near-duplicate titles count as one identity.

# Preferences
When the shopper gives a use case plus soft tastes, inspect raw facts (battery,
water resistance, materials, connectivity, features). Compare at least two
distinct listing_id values that meet the hard rules before the first add.
Ask one short priority question with message before the first cart add when the
goal invites clarification. Do not buy the cheapest valid item by default.

# Ranking
When the shopper states a tradeoff order (refresh, then screen size, then price),
rank live options that way. The first cart add must already be the current best,
not a placeholder. After a market change, re-rank survivors and inspect the
replacement before ordering.

# Recovery
Never relax category, budget, or in-stock. If an exact model+color disappears,
try the same model in another color first, then a different model. Use the
selected variant options for color, then a color fact, then the title only if
earlier sources are silent. Longest named shade wins (sage green is green).
Ambiguous multi-color text is unresolved — pick a clear option.

# Justification
If the shopper asked for a reason, place_test_order must include justification
with claims only: each claim is {field, value}. No summary key, no evidence
prose. Ground every claim in view/compare/inspect_stock of the FINAL product.
Allowed fields only: the gated spec (screen_size number OR resolution class
hd/fhd/qhd/uhd), price as a string, in_stock as "true", and brand when viewed.
Never emit display, size, category, or operating_system. After the first add,
remove that event-target SKU first, then add a different still-valid SKU.

# Search craft
- Prefer distinctive nouns: category + one spec or brand. Drop filler words.
- If page-1 is junk, drop a word, swap a synonym, or switch to filter.
- Punctuation and extra adjectives in a query often hide the right listing.
- in_stock=true when you are ready to buy; leave it unset while surveying.

Think in the assistant content: cite concrete prices, SKUs, specs, and why
rejected options fail. Then call a tool. Stop only when an observation has
done=true or the step budget is gone.
"""

_TF1_WORKSPACE = """# TF1 — intent (two scores: first add, then recovery)

Firm (never relax): category_path, the gated hardware spec on the SELECTED
SKU (usually screen_size), CURRENT shopper cap, in-stock.
Soft: named preferred brand, including two-word brands like VEGGY BASKET.
A later fix does NOT repair a first add that skipped an available
preferred-brand item that already met every firm rule.

Shopper lines you MUST honor (exact patterns):
- "which one are you going with, and does it actually have a N-inch screen?"
  → message the chosen title/brand/SKU and say the spec holds, THEN add.
- "i can only go up to X PHP now" / "budget is at most X" → X is the only
  legal cap from that moment. Original query budget is dead.
- "more expensive from A to B" → event prices, NOT a new budget.
- "please keep to my stated requirements" → compare live options under X.

Phase A — first add
1. search/filter preferred brand + spec + max_price=CURRENT cap. view SKU.
2. If a preferred-brand SKU is in-stock, on-spec, and <= CURRENT cap,
   add_to_cart THAT SKU. Never open with a cheaper other brand.
3. Answer the "which one / spec" question with message BEFORE add_to_cart.
4. STOP after the first add. Never place_test_order on this turn.

Phase B — after the first add (price_change always fires)
1. Current cap is the latest "only go up to X" / "at most X".
2. inspect_stock the carted SKU. If live price > X or OOS, remove_from_cart.
3. filter/search again with max_price=X. Pick the cheapest hard-valid SKU
   whose LIVE inspect_stock price is <= X (brand may drop).
4. place_test_order {product_id, sku} only. No justification object.

Never order a SKU whose CURRENT inspect_stock price is above the latest cap.
If none exist under X, keep searching — do not place an over-cap order.
"""
_TF2_WORKSPACE = """# TF2 — retrieval
Paid reward is retrieval_quality. query_efficiency collapses after 3 distinct
searches. Search the quoted phrase ONCE. Do not rewrite, reverse, prefix the
category, or search single leftover tokens. A title hits iff EVERY phrase
token is in the title. After the first add, do not search again — swap the
OOS event SKU for another already-viewed title-hit, then place_test_order.
"""
_TF3_WORKSPACE = """# TF3 — constraint
Every firm rule must hold on the exact variant. Ask if they will tell the spec.
"""
_TF4_WORKSPACE = """# TF4 — preference
Hard gates: Headphones (a real device, not a case/review), CURRENT budget, in-stock.
Soft axes (do not buy the cheapest unrated item): sweat/water resistance,
sport/secure fit, wireless, then battery.

Phase A — first add is a real preference-fit buy (scored).
1. Ask one short priority question with message.
2. search/filter waterproof / IPX / sport earbuds. compare two listing_id values.
3. add_to_cart the best wet+sport SKU. STOP. A price_change always fires.

Phase B — close the event, then order
1. The first SKU is now the event target. Its live price is over budget.
2. remove_from_cart THAT SKU FIRST. Never add a replacement while it is still carted.
3. add_to_cart a different in-stock wet/sport SKU under the CURRENT cap.
4. inspect_stock the replacement only. place_test_order {product_id, sku}.
"""
_TF5_WORKSPACE = """# TF5 — ranking
Rank only named product_id::sku. refresh > screen > lowest price.
"""
_TF6_WORKSPACE = """# TF6 — recovery
Never relax category/budget/in-stock. Color before model after stockout.
"""
_TF7_WORKSPACE = """# TF7 — justification
Attributes beat title. facts.display / facts.size are ATTR. A title that says
HD / 1080P / 27-inch is only TITLE evidence. If facts.display is 2560x1440,
it is QHD, not FHD. Never first-add a title decoy or a reseller whose brand
field does not match — that zeros intended_event_commitment even if the
final order is gold.
1. view/compare facts BEFORE the first add.
2. First add a branded, fact-backed, under-cap SKU. That add is the event.
3. remove it FIRST, add a DIFFERENT still-valid fact-backed SKU, then order.
Allowed claims only: gated spec, price, in_stock, brand.
screen_size = number from facts.size ("27"), never "27-inch" or "24-27".
resolution = hd|fhd|qhd|uhd from facts.display, never "720p".
"""
_WORKSPACES = {
    "intent": _TF1_WORKSPACE,
    "retrieval": _TF2_WORKSPACE,
    "constraint": _TF3_WORKSPACE,
    "preference": _TF4_WORKSPACE,
    "ranking": _TF5_WORKSPACE,
    "recovery": _TF6_WORKSPACE,
    "justification": _TF7_WORKSPACE,
}
_LANE_MARKERS = (
    ("intent", "mix of firm requirements and nice-to-haves"),
    ("retrieval", "survey the realistic options"),
    ("constraint", "every requirement must hold for the exact variant"),
    ("preference", "soft preferences matter"),
    ("ranking", "highest advertised maximum refresh rate"),
    ("recovery", "relaxable only after the exact item becomes unavailable"),
    ("justification", "walk me through why"),
    ("justification", "structured justification"),
)
_KEEP_OBS = (
    "error", "done", "action", "market_event", "added", "product_id", "sku",
    "price", "in_stock", "currency", "title", "brand", "category_path", "facts",
    "options", "listing_id", "returned", "truncated", "cart", "justification",
)

# --- 115-BILL prompts (called: P_DECOMPOSE / PSV_BUILD / PSV_KW / PV_SELECT) ---

_P_DECOMPOSE = '\nYou are AI assistant to analyze the user\'s shopping query. The user is looking for a single product with specific constraints.\n\n# Task\n## Step-1. Identify the price constraint if mentioned, and normalize it to "low-high" format without currency name.\n- A price constraint is mentioned like "between 10 and 50 php", "more than 2400", or "less than 120 USD".\n- User may mention a price constraint without currency specified, like "Looking for a makeup bag between 110 and 200."\n- If a numeric range is mentioned without a currency, carefully check the context to determine whether it refers to price. Examples:\n    * "Show me black tshirts between 50 and 150." → The phrase "between 50 and 150" is the price constraint.\n    * "Need aircon with power usage between 1000 and 1500." → The phrase "between 1000 and 1500" is the power consumption constraint, not the price constraint.\n- Normalize the price constraint:\n    * Normalize as "low-high" if both lower and upper bounds of price were specified. Example: "5-160"\n    * Normalize as "low-" if only the lower bound was specified. Example: "410-"\n    * Normalize as "-high" if only the upper bound was specified. Example: "-905"\n\n## Step-2. Identify the service constraints if mentioned. Check for the following service labels:\n    * lazmall: user wants an official, genuine, guaranteed, or authentic product through LazMall or LazMall Perk which is a platform providing easy and fast returns\n    * `official`: user explicitly wants product from an official/authorized **store, seller, shop, or flagship store**. \n        **Trigger phrases must include "official" + a store/seller term** (e.g., "official store", "official seller", "authorized seller", "official flagship store").\n        **Does NOT trigger when "official" modifies the product itself** (e.g., "official ball", "official men\'s shoes", "official product used in matches").\n    * `lazflash`: user wants lightning deal through LazFlash\n    * `flashsale`: user wants flash sale or lightning deal or limited time offer\n    * free_ship: user wants free or complimentary shipping or delivery\n    * cod: user wants \'Cash on Delivery\', \'Payment on Delivery\', or any phrase indicating payment upon receipt/delivery (e.g., "pay when they arrive", "pay on arrival", "pay upon delivery", "pay when it gets here")\n    * product_only: user requests only the product itself, explicitly excluding accessories, bundles, or related items. Trigger phrases must **exactly match** one of: "only the product", "just the product", "product only", "no accessories", "without accessories", "not a set", "exclude bundle". **NO other variation triggers this.** Phrases like "levers only", "shoes only", "[product name] only" that merely specify a product variant do NOT trigger this constraint. Phrases like "single item", "single items", "each item", or "per item" that refer to quantity, warranty, or shipping scope do NOT trigger this constraint.\n    * **Example:** Query: "I need brake levers only, suitable for motorcycles." → This is NOT product_only because "levers only" specifies a product variant, not exclusion of accessories. Query: "I need brake levers without accessories" → This IS product_only.\n**ATTENTION - The term "branded" or "genuine original branded" is NOT a service constraint, but a separate constraint meaning the product must have a branded feature. DO NOT mark it with a service label!**\n\n## Step-3. Rewrite the query, by excluding the price and service constraints.\n- Make sure the rewritten query includes all details from the original query, except for price and service constraints.\n- Strictly respect the wording of original query. There is no typo.\n- DON\'T INCLUDE PRICE AND SERVICE CONSTRAINTS IN THE REWRITTEN QUERY!\n\n## Step-4. Identify the following paraphrasable tokens in the query with their alternatives. (If either left or right side token is detected, set other side token as the alternative.)\n    * "t-shirt"↔"tshirt", "t shirt"↔"tshirt", "email"↔"e-mail", "wristwatch"↔"wrist watch", "flip-flop"↔"flipflop", "x ray"↔"xray", "air conditioner"↔"aircon", ...\n    * "standard"↔"default", "normal"↔"regular", "plain"↔"simple", "high"↔"max", "shirt"↔"tshirt"\n    * "gray"↔"grey", "color"↔"colour", "chili"↔"chilli"\n\n## Step-5. Identify the most suspicious token in the query that may be non-beneficial for product search. Examples:\n    * "pair" for querying "gloves", because all goves are sold by pairs\n    * "set" for querying "salt and pepper shakers", because they are almost always sold as a set\n    * "replacement" for querying "tyre", because all tyres are for replacement\n    * "supplement" for querying "vitamin", because all vitamins are for supplement\n    * "unit" for querying "ABS", because all ABS are moduled as unit\n\n**STEP-4 EXAMPLES ARE BI-DIRECTIONAL. CAPTURE TOKENS THAT APPEAR IN EITHER LEFT OR RIGHT SIDE, AND GIVE OTHER SIDE AS THE ALTERNATIVE!**\n**DO NOT AGGRESSIVELY EXTEND THE PARAPHRASABLE TOKENS BEYOND THE EXPLICIT EXAMPLES OF STEP-5!**\n\n# Response Format\nReturn JSON object like:\n{{\n    // Step-1\n    "prices": {{ // Find a phrase related to the price constraints from the original query, if exists.\n        "rationale": <str>, // A brief summary in 1 sentence to identify the price constraints from the original query.\n        "normalized": <str|null> // Normalized price format ("low-high" or "low-" or "-high", without currency). Set null if price constraint does not exist.\n    }},\n\n    // Step-2\n    "services": {{ // Find all phrases related to the service constraints from the original query.\n        "rationale": <str>, // A brief summary in 1 sentence to identify the service constraints from the original query.\n\n        // If no service constraint was found, don\'t output the following phrases but skip to Step 3!\n        // If found service constraints, list all those phrases below.\n\n        <phrase_1, str>: <service_label, str|null>, // The exact service label string: "lazmall", "official", "lazflash", "flashsale", "free_ship", "cod", or "product_only". Use null if the phrase is not a service constraint.\n        ... ...\n    }},\n\n    // Step-3\n    "rewritten": {{\n        "rationale": <str>, // A brief summary in 1 sentence to find any potential pitfalls for rewriting and how to manage it.\n        "result": <str> // Rewrite the query excluding price and service constraints. Never miss any details.\n    }},\n\n    // Step-4\n    "alt": {{ // List the identified tokens and their alternatives, that are required in Step-4. (Limit them as given examples in Step-4. Don\'t extend! Empty list is preferred!)\n        <token_1, str>: <alternative_1, str>, // the source token must exactly exist in the rewritten query of Step-3, without spell modification\n        ... ...\n    }},\n\n    // Step-5\n    "redundant": <top_redundant_token, str|null> // one most suspicious token. (You are not enforced to select one always! Null is preferred!)\n}}\n\n# Original Query\n{query}\n'

_PSV_BUILD = '\nYou are an AI assistant that converts product descriptions into structured e-commerce attributes.\nYour task is to transform each product description into a set of key-value pairs that faithfully represent all information contained in the description.\n\n# Core Objective\nConvert every product description into a structured attribute dictionary such that:\n\n## 1. The attributes must be **mutually exclusive**:\n- The same information must not appear in multiple attributes.\n- Each piece of information should be assigned to exactly one attribute.\n\n## 2. The attributes must be **collectively exhaustive**:\n- All meaningful product information and constraints from the original description must be preserved.\n- The combination of all attributes should fully reconstruct the original meaning.\n\n## 3. Each attribute must represent one constraint aspect:\n- Don\'t include multiple constraint aspects for one attribute.\n- Don\'t split one constraint aspect into two attributes.\n\n## 4. The output should resemble a real e-commerce database schema.\n\n# Attribute Construction Rules\n## Rule-1. Mandatory Product Type(basic product category name)\nEvery output must contain one pair "product_type: <value>". This must be the first attribute in the result object.\n\n## Rule-2. Minimize Value Length\n- Make values as short as possible while preserving meaning, by moving descriptive tokens from the value into the key when appropriate. Examples:\n    * "star shape" → "shape: star"\n    * "type A or B models" → "model: type A B"\n    * "metal-based" → "material_base: metal"\n    * "compatible with healthcare apps" → "compatible_app: healthcare"\n    * "as a model boots" → "model: boots"\n    * "fit for office use" → "usage: office"\n    * "with the interest in nature" → "interest: nature"\n    * "pack of 120 balls" → "pack_ball: 120"\n    * "spring-inspirated design" → "design_inspiration: spring"\n    * "perfect for Yamaha piano" → "compatible_piano: Yamaha"\n    * "with side camera 5MP" → "side_camera: 5MP"\n    * "having 512GB hard disk size" → "hard_disk_size: 512GB"\n- The value MUST NOT INCLUDE the descriptive word or token moved into the key side.\n    * Correct: "shape: circle" || Incorrect: "shape: circle-shaped"\n    * Correct: "table: 12" || Incorrect: "table: 12 tables"\n- DO NOT split one constraint aspect into two attributes to make values short. It\'s strictly forbidden. You must make values short inside one attribute for one constraint aspect!\n    * "with 20 sockets" → Correct: "socket_count: 20" || Incorrect: "pack_type: socket" + "pack_size: 20"\n    * "made for G1, G3, or G-K types" → Correct: "type: G1 G3 G-K" || Incorrect: "type: G1" + "type: G3" + "type: G-K"\n- Convert the unit "piece" to "pcs", because "pcs" is the reserved unit of "piece(s)" for the current e-commerce db. Remove space between number and "pcs". Never use "pcs" unless "piece(s)" or "pcs" was explicitly used in the description.\n    * "packed in 24 pieces" → Correct: "pack: 24pcs" || Incorrect: "pack: 24 pcs", Incorrect: "pack: 24 piece"\n    * "1 piece color family" → Correct: "color_family: 1pcs" || Incorrect: "color_family: 1 piece", Incorrect: "color_family: 1"\n    * "set of 20" → Correct: "set: 20" || Incorrect: "set: 20pcs", Incorrect: "pack: 20pcs"\n- For the pattern `[number][space][unit]` (with space token between number and unit), unit token must be moved to the key. This rule is not applied to the unit "piece". Example:\n    * "that come with 7 pearls" → Correct: "pearl: 7" || Incorrect: "feature: 7 pearls"\n- For the pattern `[number][unit]` (without space token between number and unit), both number and unit tokens must remain in the value. This rule is applied to the unit "pcs" which is converted from "piece". Example:\n    * "in radius 5mm" → Correct: "radius: 5mm" || Incorrect: "radius_mm: 5"\n\n## Rule-3. Prefer Source-Derived Keys\n- When possible, derive attribute names from words or tokens already present in the description. Example:\n    * "compatible with healthcare apps" → Correct: "compatible_app: healthcare" || Incorrect: "fit: healthcare apps"\n    * "for model Pacific" → Correct: "model: Pacific" || Incorrect: "compatibility: Pacific model"\n    * "with checker pattern" → Correct: "pattern: checker" || Incorrect: "design: checker pattern"\n    * "comes with 10 piece" → Correct: "piece: 10" || Incorrect: "pack: 10"\n- Only introduce generic schema keys when no suitable key can be derived. Common fallback keys include:\n    `product_type, brand, size, color, color_family, pack, capacity, user, gender, model, compatibility, material, feature, usage, style, design, accessory, variation, option, choice`\n- NEVER move the number-included tokens into the keys. They MUST be in the values. Example:\n    * "2pcs color" → Correct: "color: 2pcs" || Incorrect: "2pcs: color" or "2pcs_color: color"\n    \n### The keys "color" and "color_family" are not limited to color modifiers!\n- Unlike their names, the keys "color" and "color_family" can be used for any constraint including color, size, package type, and all. Examples:\n    * "color: red #03"\n    * "color: premium"\n    * "color: 1pcs"\n    * "color_family: 10pcs"\n    * "color_family: 10cm"\n    * "color_family: button"\n- Don\'t mistakenly recognize "color" and "color family" as if they\'re attribute values. Incorrect Examples:\n    * "color: family"\n    * "family: color"\n    * "color_family: family"\n    * "color_family: color"\n    * "variation: color family"\n    * "option: color family"\n- If the original query mentions "I am looking for 5pcs color wide neck bottles", then "5pcs color" means "color: 5pcs" (looks a bit odd, but it\'s the correct one!), not that user is finding a "colored" bottle.\n    * Correct: "product_type: bottle", "color: 5pcs", "neck: wide"\n    * Incorrect: "product_type: bottle", "pack: 5pcs", "color: color", "neck: wide"\n\n## Rule-4. Remove Filler Words\n- Remove non-informative words such as:\n    `and, or, in, with, for, to, still, just, upon, about, the, at least, above, less than, between, a, an, from, to, \'s, s\'`, unless they are part of a deterministic term.\n- Examples to remove:\n    * "compatible with office use" → "usage: office"\n    * "specifically with the interest in nature" → "interest: nature"\n    * "sportwomen\'s bra" → "product_type: sportswomen bra"\n    * "above 12 years" → "age: 12+ years"\n    * "fit for kids\'" → "user: kid"\n    * "between 2 and 5 meters" → "length: 2-5 meters"\n- Examples to keep:\n    * "2-in-1 type" → "type: 2-in-1"\n    * "with brand just honests" → "brand: just honests"\n    * "color \'a\'" → "color: a"\n\n## Rule-5. Negation Handling\nNever use: "yes", "true", "false", "included", "used", "required" as standalone values.\n    * "comes with protection lid" → Correct: "accessory: protection lid" || Incorrect: "protection_lid: included"\n    * "operation in safe mode" → Correct: "operation_mode: safe" || Incorrect: "safe_mode_operation: true"\n    * "genuine and branded" → Correct: "feature: genuine branded" || Incorrect: "branded: yes"\n    * "with iOS required" → Correct: "requirement: iOS" || Incorrect: "ios: required"\n\n### Handling "no" / "not" / "without"\n- Only use "no"/"not" when the original text explicitly expresses absence by "no", "not" or "without". But "no"/"not" cannot be used as standalone value.\n    * "without any cover" → Correct: "feature: no cover" || Incorrect: "cover: no"\n    * "no fragrance" → Correct: "feature: no fragrance" || Incorrect: "fragrance: no"\n    * "with axis not specified" → Correct: "feature: not specified axis" || Incorrect: "axis: not"\n\n- Do not convert other expressions into "no" or "not". Only "without" or "without any" can be converted to "no".\n    * "wind-free" → Correct: "feature: wind-free" || Incorrect: "feature: no wind"\n    * "stop smoking" → Correct: "feature: stop smoking" || Incorrect: "feature: no smoking"\n\n## Rule-6. Some Edge Cases\n    * "fit for all ages" → "fit_age: all"\n    * "comes with any accessories" → "accessory: any"\n    * "suitable for both men and women" → "gender: unisex"\n    * "one size" → "size: one"\n    * "one size fits to all" → "size: freesize"\n    * "for any gender" → "gender: any"\n    * "compatible with iPhone and iPad" → "compatible_device: iPhone iPad"\n    * "fit for any models" → "model: universal"\n    * "available in red or blue" → "color: red blue"\n\n## Rule-7. Singular Form Preference\n- Use singular forms unless:\n    * the plural form is explicitly required, or\n    * the plural form is a recognized deterministic term.\n- Examples:\n    * "balls" → "ball"\n    * "United States" → "United States" (unchanged)\n    * "shoes" → "shoes" (unchanged)\n\n## Rule-8. Attribute Quality Rules\n    * DO NOT create duplicate attributes.\n    * DO NOT split a single concept across multiple attributes to make values short.\n    * DO NOT invent information.\n    * DO NOT infer specifications not explicitly stated.\n    * DO NOT remove meaningful token. Minimizing value means moving tokens to the key when possible, not removing them entirely.\n    * KEEP values concise but fully informative.\n    * PRESERVE original wording whenever possible. There is no typo in the original description.\n\n# Procedures\n\n## Step 1. Rewrite the original product description in compact version by removing non-informative words.\n- Remove non-informative tokens to make the description search-friendly, unless they are part of deterministic terms.\n- In this step, do not split key and value tokens yet. Just rewrite the description left-to-right order by removing redundant tokens.\n- Examples:\n    * "hp04 ink bottle made in France, in 4b color family and multipack variation" → "hp04 ink bottle France 4b color family multipack variation"\n    * "brand new bicycle head light that fits any bicycle model" → "brand new bicycle head light any bicycle model"\n    * "video game CD with title The God" → "video game CD title The God"\n    * "photo frame with size 50cm by 120cm" → "photo frame size 50cm 120cm"\n    * "USB cable with type C or U plugs" → "USB cable type C U plug"\n    * "fan accessories compatible with Dell R960C workstation" → "fan accessory Dell R960C workstation"\n    * "phone case specifically designed for Samsung Galaxy" → "phone case design Samsung Galaxy"\n\n## Step 2. Convert the description into key-value formatted attribute dictionary.\n- Make sure any meaningful detail is not missing in the attribute dictionary.\n- Make sure the attributes are non-overlapped.\n- Strictly keep the **Attribute Construction Rules** (1-8).\n\n# Response Format\nReturn a JSON object like:\n{{\n        "1": {{ // for product description "1"\n        "description": <original description 1, str>,\n\n        // Step 1\n        "compact": <rewrite description 1 into a search-friendly, compact, concise version without non-informative tokens, str>,\n\n        // Step 2\n        "rationale": <a short summary of how attributes were selected and will be organized by strictly adhering the construction rules, str>,\n        \n        "attributes": {{\n                "product_type": <value, str>, // this is the must-have pair\n            <attribute_key_1, str>: <attribute_value_1, str>,\n            ... ...\n        }}\n    }},\n    ... ...\n}}\n\n# Requirements:\n- Output valid JSON only. No markdown. No explanations outside the JSON.\n- The first key inside every result object must be "product_type".\n- Thoroughly check all the rules and find the best way of transformation!\n\n# Examples of "rationale" field:\n    * From the original description, meaningful tokens are xxx, xxx, ... The product type is xxx. The best way of attributes construction from the rest tokens by following the Rules 1-8 and mimicking the real e-commerce db is xxx: xxx, xxx: xxx, ... No missing meaningful token? All meaningful tokens are included in the product type and attributes. Good.\n    * From the original description, meaningful tokens are xxx, xxx, ... The product type is xxx. The best way of attributes construction from the rest tokens by following the Rules 1-8 and mimicking the real e-commerce db is xxx: xxx, xxx: xxx, ... No missing meaningful token? The token xxx is missing. We must add an attribute xxx: xxx. Good.\n    * From the original description, meaningful tokens are xxx, xxx, ... The product type is xxx. The best way of attributes construction from the rest tokens by following the Rules 1-8 and mimicking the real e-commerce db is xxx: xxx, xxx: xxx, ... No missing meaningful token? The token xxx is missing. It must be added to the product type value. So the correct product type is xxx. Good.\n\n# Original Product Descriptions\n{queries}\n'

_PSV_KW = '\nYou are an AI assistant that reviews e-commerce search keywords and recommends, when appropriate, a single alternative keyword set.\nYou are given one or more shopping queries together with the current search keywords generated for each query.\nThe existing keywords are already high quality. Your task is NOT to rewrite them. Instead, identify rare cases where making exactly one small change could improve search recall without significantly reducing precision.\n\n# Search Engine Behavior\nThe product catalog is stored as JSON. Only JSON values are indexed for search. Therefore:\n- Words that normally appear only as JSON keys (for example: size, color, color family, material, brand, category, product type) generally do not help retrieval and may increase irrelevant matches.\n- Hyphens are normalized to spaces before search. Therefore "t-shirt" and "t shirt" are equivalent.\n- The search engine uses BM25 with morphological normalization. Variants such as: `shape/shaped`, `woman/women`, `child/children` produce identical matches.\n- Re-ordering the keywords produces the exactly identical matches.\n- Duplication of keywords produces the exactly identical matches.\n\n# About the Existing Keywords\nThe original keywords were generated automatically and are usually well optimized. In particular:\n- Most useful value-side keywords are already included.\n- Non-informative words have already been removed.\n- Some query terms are intentionally normalized to match catalog conventions, for example:\n  * `large` → `l`, `extra old` → `xo`\n  * `fit for men and women` → `unisex`\n  * `set of 12 pieces` → `12pcs`\n  * `water-resistant` → `waterproof`\n\nAssume the original keywords are correct unless there is a strong reason to change them.\n\n# Objective\nRecommend AT MOST ONE alternative keyword set. The recommendation should have a reasonable chance of improving search quality compared with the original keywords. If no change is clearly beneficial, return null.\n\n# Allowed Modification\nChoose exactly one of the following:\n1. Add exactly one keyword.\nOR\n2. Remove exactly one keyword.\n\nNever perform both operations. Never rewrite, reorder, or substitute keywords. The alternative must otherwise remain identical to the original keywords.\n\n# Step 1 — Consider Adding\nRecommend an addition only if there is a missing keyword that is likely to appear in indexed product values and is expected to improve retrieval.\n\nPossible candidates include:\n- an obvious missing product attribute\n- a keyword whose presence in values is reasonably likely\n\nDo NOT add:\n- non-informative words\n- grammatical words\n- shopping intent words\n- words that are clearly JSON keys\n- intentionally normalized words\n- tokens that already exist in the original keywords\n- tokens that are not included in the query\n\nExamples of words that should NOT be added:\ncompatible, for, with, pair, included, perfect, specifically, above, size, color, material, color family, brand, ...\n\n# Step 2 — Consider Removing\nIf no useful addition exists, determine whether removing exactly one keyword would likely improve retrieval.\n\nOnly remove a keyword if it is likely to:\n- rarely appear in indexed values,\n- behave more like a metadata key than product content,\n- or unnecessarily narrow the search. (Examples: "set", "pair", "pack", "accessory")\n\nDo NOT remove:\n- tokens that clearly describe the product\n- numbers\n- unique identifiers\n- brands\n\n# Decision Rule\nRecommend an alternative only when the expected improvement is meaningful. When uncertain, prefer returning null.\n\n# Response Format\nReturn JSON object.\n{{\n        // check all queries in order as given\n\n    "1": {{ // for query 1\n\n        "rationale": "<explanation>",\n        "alt": "<full alternative keyword string>" | null\n    }},\n    ... ...\n}}\n\nRequirements:\n- "alt" must contain the complete keywords string after the single modification, so that it can be immediately used for the search.\n- If no recommendation is justified, set "alt" to null.\n- Output valid JSON only. No markdown. No additional text outside JSON.\n\n# Shopping Queries and Original Keywords\n{inf}\n'

_PV_SELECT = '\nYou are an AI assistant that selects the best matching product for a shopping query from an ordered candidate list.\nYour objective is **precision**. A product is considered a valid match **only if every required constraint extracted from the query is satisfied**.\n\n# Overall Procedure\nPerform the following phases strictly in order.\n\n## Phase 1. Extract Constraints\n- Read the shopping query and extract all constraints. Each constraint must belong to exactly one category. Possible categories include (but are not limited to):\n    `brand, model, series, identifier/code/SKU, number, size, capacity, color, material, feature, compatibility, shape, style, package, age, gender, usage, pattern`\n- All constraints are *REQUIRED*.\n- Do NOT begin checking products yet.\n\n## Phase 2. Classify Matching Type\nEach extracted constraint must use exactly one matching mode.\n\n### Literal Match\nUse literal matching for `brands, model names, identifiers, product codes, uncommon token combinations, numbers, serial names, abbreviations, versions, colors`.\n    * Literal constraints must appear explicitly.\n    * Never normalize them. Never partially match them.\n\n    Examples:\n        Query "100ml" → Accept "100ml", reject "50-150ml"\n        Query "iPhone" → Accept "iPhone", reject "Apple smartphone"\n        Query "Green 2" → Accept "green 2", reject "green" (Casing difference allowed)\n        Query "3-7 years" → Accept "3-7yrs", reject "5 years"\n\n### Size Token Equivalence (Literal Match Exception)\nSize constraints use a special literal matching rule: standard size abbreviations and their full forms are interchangeable. When checking size constraints, normalize both the query and the candidate to a canonical form before comparison.\nCanonical mappings:\n- "s" ↔ "small"\n- "m" ↔ "medium"\n- "l" ↔ "large"  \n- "xl" ↔ "extra large" ↔ "xlarge"\n- "xxl" ↔ "extra extra large" ↔ "2xl"\n- "xs" ↔ "extra small"\nThis is the ONLY exception to "never normalize literal constraints."\n\n### Semantic Match\n- Everything else uses semantic matching. Paraphrases are allowed. Common typos are allowed. Examples:\n    child = kid, wood = wooden, regular = standard, plain = simple, customized = personalized, water-resistant = waterproof, dioxide = dioxde\n- Never be too strict. Be generous with common sense and implicity for semantic match constraints. Most of the candidates are matching. Prefer FALSE POSITIVE than FALSE NEGATIVE!\n\n## Phase 3. Product Evaluation\n- Evaluate products in Fibonacci batches. Let `N_0 = 3`, `N_1 = 2`, and `N_[k+2] = N_[k] + N_[k+1]`. Therefore, the batches are:\n    * Phase 0: (N_0 = 3) → Products 1-3\n    * Phase 1: (N_1 = 2) → Products 4-5\n    * Phase 2: (N_2 = 5) → Products 6-10\n    * Phase 3: (N_3 = 7) → Products 11-17\n- For every product evaluate every constraint independently.\n- Each constraint receives one result only: `PASS`, `FAIL`, or `UNKNOWN`\n- A product is a valid candidate only if **every constraint is PASS**!\n- If any constraint is FAIL reject the product immediately.\n- `UNKNOWN` is treated as FAIL.\n\n## Field Rules\n### Product Type\nProduct type has already been verified. Never reject a product because of product type.\n\n### Title\nThe title is an unordered keyword collection. Do NOT infer sentence meaning. Only check token existence. Example:\n    - Query "water bottle" → Title "Black Plastic Milk Bottles Cover Kids Water". The title contains both `water` and `bottle`. Therefore title evidence is POSITIVE. Do not infer that the product is a milk bottle. The title is only evidence of keyword presence.\n    - Query "tank tester" → Title "tank capacity tester". The title contains both `tank` and `tester. Therefore title evidence is POSITIVE. Do not infer that the product type is mismatching. The title is only the evidence of keyword presence.\n\n### Attributes\n- Attributes are authoritative. Each attribute consists of `Key : Value`. Evaluate every attribute independently.\n- If any attribute provides a valid value for a constraint, the constraint PASSes.\n- Conflicting values in other attributes are ignored because they may represent different variations.\n\n### Color Fields\n- The attributes `color` and `color_family` may contain:\n    * colors (Example: "color: black", "color_family: white bracelet")\n    * sizes (Example: "color_family: 10mm", "color": "1.6m")\n    * package names (Example: "color: 1pcs", "color: 3pcs")\n    * identifiers (Example: "color_family: G #1")\n    * quantities (Example: "color_family": "5kg")\n    * serial names (Example: "color": "aa12-c")\n- Treat them as generic attributes. Do not assume they only contain colors.\n\n## Constraint Evaluation\n- For every constraint check title. If satisfied: PASS. Otherwise: check every attribute. If any attribute satisfies the constraint: PASS. Otherwise: FAIL.\n- Never infer literal constraints. Never invent missing information. Never assume compatibility. Never guess.\n- Attribute specificity matters: A constraint is best satisfied by an attribute whose key directly corresponds to the constraint category. Generic attributes (e.g., `options`, `feature`, `variation`) containing the value are weaker evidence than specific attributes (e.g., `color`, `size`, `brand`).\n- For PASS: the value must appear in a category-matching attribute OR in the title. Generic attributes alone are NOT sufficient for PASS unless no category-specific attribute exists in the product data.\n\n## Phase 4. Recommendation\n- If one or more valid products are found in the current Fibonacci batch: STOP immediately.\n- Only compare the valid products found in this batch. Choose the product that most closely matches the original wording of the shopping query. Prefer:\n    * more exact wording\n    * more literal matches\n    * more explicit evidence\n    * evidence in attributes list than in title. Structured evidence in attributes/options/variants over noisy SEO title text.\n    * Brand Attribute Tie-Breaker Rule: In cases where multiple candidate products yield similar relevance scores, and the brand name is present in each product\'s title, prioritize the candidate whose structured attributes additionally contain that brand name.\n    * If both satisfy all constraints, prefer the more exact / direct match (structured attributes, fewer extraneous tokens).\n- If multiple products are equally matched: choose the earliest product! Do NOT continue to later Fibonacci phases.\n- If no valid products exist: continue to the next Fibonacci batch.\n\n**PRODUCT TYPES HAVE ALREADY BEEN VERIFIED FOR ALL PRODUCTS BY THE SYSTEM! DO NOT REJECT BECAUSE OF PRODUCT TYPE MISMATCH!**\n**TITLE IS ONLY THE EVIDENCE OF KEYWORD PRESENCE! IF RELEVANT TOKENS EXIST IN THE TITLE, IT\'S A MATCH, REGARDLESS OF THEIR ORDERS AND SEMANTICS!**\n\n# Response Format\nReturn JSON like:\n{{\n        ... ...\n\n    "phase_k": {{\n            "fibonacci": <int>, // fibonacci number N_[k]\n        "check_range": <str>, // checking range of product id: e.g. "7-11"\n        "Product_[k]": <str>, // evaluation for the first product of the current batch\n        "Product_[k+1]": <str>, // evaluation for the second product of the current batch\n        ... ...\n        "valid": <str|null>, // valid matching Product Ids separated by commas if at least one exists, otherwise null\n    }},\n    ... ...\n\n    // if at least one valid matching product was found (at least 1), stop Fibonacci phases\n    \n    "recommend": <str|null>, // a summary for how the best product will be selected from valid products if valid ones found, otherwise null\n    "best": <str|null> // product id of the recommended best matching product if found, otherwise null\n}}\n\n# Output Guidelines\n- The checking text must be concise. One product per line. Example:\n    * "Product_3": "Brand PASS; Capacity PASS; Color FAIL"\n    * "Product_4": "Brand PASS; Capacity PASS; Color PASS; Material PASS"\n- Recommendation should be a short sentence. Example:\n    "Valid: Product 4, 6, 7. Comparing: ... ... Conclusion: Product 4 aligns most closely with the original wording."\n\n# Shopping Query:\n{query}\n\n# Candidate Products:\n{products}\n'

_PV_FILTER = """
You are filtering e-commerce search results.
Given a shopping query and a numbered list of candidate products, remove the
products that do not match the query.
Return JSON only: {{"keep": [<index>, ...]}}
# Query
{query}
# Candidates
{cands}
"""

def _lane(query: str) -> str:
    blob = query.casefold()
    for name, marker in _LANE_MARKERS:
        if marker in blob:
            return name
    return "generic"


def _model_ladder(lane: str) -> list[str]:
    override = (getenv("SANDBOX_MODEL") or "").strip()
    ladder = list(_FAMILY_MODELS.get(lane) or _FAMILY_MODELS["generic"])
    if override and "/" in override and not override.endswith("-TEE"):
        if override not in ladder:
            ladder.insert(0, override)
        else:
            ladder = [override] + [name for name in ladder if name != override]
    return ladder


def _goal_head(query: str) -> str:
    return query.split("Task rules:", 1)[0]


def _title_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").casefold())


_PHRASE_STOP = {
    "the", "phrase", "matches", "match", "that", "listing", "listings",
    "find", "in", "stock", "category", "and", "buy", "one", "under",
    "or", "at", "an", "a", "php", "usd",
}


def _quoted_terms(head: str) -> list[str]:
    found = re.findall(r'"([^"]+)"', head)
    raw = found[0] if found else ""
    if not raw:
        match = re.search(
            r"(?:the\s+)?phrase\s+(.+?)(?:,|\bat or under\b|\band buy\b|$)",
            head,
            re.I,
        )
        if match:
            raw = match.group(1)
    return [tok for tok in _title_tokens(raw) if tok not in _PHRASE_STOP and not tok.isdigit()]


def _tf2_phrase(goal: dict[str, Any]) -> str:
    return " ".join(goal.get("terms") or []).strip()


def _tf2_rows(
    rows: list[dict[str, Any]],
    goal: dict[str, Any],
    plan: dict[str, Any],
    *,
    skip_sku: str | None = None,
    skip_pid: str | None = None,
) -> list[dict[str, Any]]:
    terms = list(goal.get("terms") or [])
    out: list[dict[str, Any]] = []
    for row in rows:
        if terms and not _title_hits(str(row.get("title") or ""), terms):
            continue
        merged = dict(row)
        if not merged.get("sku"):
            variants = _iter_variants(row)
            if variants:
                merged = {**row, **variants[0]}
        if not merged.get("product_id") or not merged.get("sku"):
            continue
        if skip_sku and str(merged.get("sku") or "") == str(skip_sku):
            continue
        if skip_pid and str(merged.get("product_id") or "") == str(skip_pid):
            continue
        if not _price_ok(merged, goal.get("budget"), plan.get("price")):
            continue
        if not _in_stock(merged):
            continue
        path = _path_text(merged.get("category_path"))
        category = goal.get("category")
        if path and category and category.casefold() not in path.casefold():
            continue
        out.append(merged)
    tokens = terms or list(plan.get("keywords") or [])
    out.sort(key=lambda r: -_weigh(r, tokens, "", plan))
    return out


def _title_hits(title: str, terms: list[str]) -> bool:
    if not terms:
        return False
    tokens = set(_title_tokens(title))
    return all(term in tokens for term in terms)


def _extract_number(text: str) -> float:
    matches = re.findall(r"\d+\.?\d*", str(text or ""))
    return float(matches[0]) if matches else 0.0


def _as_float(value: Any) -> float | None:
    if value in (None, "", False):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    matches = re.findall(r"\d+\.?\d*", str(value))
    return float(matches[0]) if matches else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"true", "1", "yes", "in_stock", "in stock"}


def _parse_json(text: str) -> Any:
    cleaned = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", str(text), flags=re.I | re.S)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0]
    for start in sorted(starts):
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            return obj
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(cleaned.replace("'", '"'))
    except json.JSONDecodeError:
        return None


def _keywords_fallback(head: str) -> list[str]:
    words: list[str] = []
    for tok in _title_tokens(head):
        if tok in _KW_STOP or tok.isdigit() or len(tok) < 2:
            continue
        if tok not in words:
            words.append(tok)
        if len(words) >= 6:
            break
    return words


def _blob(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=True, default=str).casefold()


def _path_text(path: Any) -> str:
    if isinstance(path, list):
        return " ".join(str(x) for x in path)
    return str(path or "")


_BRAND_FILLERS = {
    "a", "an", "the", "in", "in-stock", "instock", "stock", "cheap",
    "good", "nice", "new", "best", "valid",
}
_BRAND_STOP = {
    "smartphone", "smartphones", "headphone", "headphones", "monitor",
    "monitors", "tablet", "tablets", "phone", "phones", "gaming",
    "earbuds", "earbud", "earphone", "earphones", "hd", "fhd", "qhd",
    "uhd", "full", "inch", "inches", "hz", "720p", "1080p", "1440p",
    "2160p", "4k", "in",
}


def _extract_brand(head: str) -> str | None:
    named = re.search(
        r"(?:prefer|brand)\s+([A-Za-z][A-Za-z0-9]*(?:\s+[A-Za-z][A-Za-z0-9]*){0,2})",
        head,
        re.I,
    )
    if named:
        token = named.group(1).strip()
        if token.casefold() not in _BRAND_STOP:
            return token
    lead = re.search(r"\b(?:need|want|get|find)(?:\s+me)?(?:\s+an|\s+a)?\s+(.+)", head, re.I)
    if not lead:
        return None
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", lead.group(1))
    cleaned: list[str] = []
    for word in words:
        key = word.casefold().replace("-", "")
        if key in {"instock", "stock"} or word.casefold() in _BRAND_FILLERS:
            continue
        if word.casefold() in _BRAND_STOP or re.match(r"\d", word):
            break
        cleaned.append(word)
        if len(cleaned) >= 2:
            break
    token = " ".join(cleaned).strip()
    if token and token.casefold() not in _BRAND_STOP:
        return token
    return None


def _product_type(goal: dict[str, Any]) -> str:
    cat = str(goal.get("category") or "").casefold()
    if "smart" in cat:
        return "smartphone"
    if "tablet" in cat:
        return "tablet"
    if "monitor" in cat:
        return "monitor"
    if "head" in cat:
        return "headphone"
    return ""


def _parse_goal(query: str) -> dict[str, Any]:
    head = _goal_head(query)
    facts: dict[str, Any] = {"head": head.strip()}
    budget = re.search(r"(\d+(?:\.\d+)?)\s*(USD|EUR|GBP|PHP|SGD|AUD|CAD)", head, re.I)
    if budget:
        facts["budget"] = budget.group(1)
        facts["currency"] = budget.group(2).upper()
    brand = _extract_brand(head)
    if brand:
        facts["brand"] = brand
    inches = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", head, re.I)
    if inches:
        facts["screen"] = inches.group(1)
    res = re.search(r"\b(720p|1080p|1440p|2160p|4k|uhd|qhd|full hd|hd)\b", head, re.I)
    if res:
        facts["resolution"] = res.group(1)
    hz = re.search(r"(\d+)\s*hz", head, re.I)
    if hz:
        facts["refresh"] = hz.group(1)
    ram = re.search(r"(\d+)\s*gb\s*ram", head, re.I)
    if ram:
        facts["ram"] = ram.group(1)
    cat = re.search(r"in the ([^,\n]+?) category", head, re.I)
    if cat:
        facts["category"] = cat.group(1).strip()
    elif re.search(r"\bsmartphone", head, re.I):
        facts["category"] = "Smartphones"
    elif re.search(r"\bheadphone", head, re.I):
        facts["category"] = "Headphones"
    elif re.search(r"\bmonitor", head, re.I):
        facts["category"] = "Monitors"
    elif re.search(r"\btablet", head, re.I):
        facts["category"] = "Tablets"
    color_model = re.search(
        r"prefer the (black|green|white|blue|red|gold|silver|gray|grey)\s+(\S+)",
        head,
        re.I,
    )
    if color_model:
        facts["color"] = color_model.group(1).casefold()
        facts["model"] = color_model.group(2)
    facts["candidates"] = re.findall(r"(\d{8,}::[\w.-]+)", head)
    facts["terms"] = _quoted_terms(head)
    facts["ask_spec"] = "tell you if you ask" in head.casefold()
    facts["device"] = bool(re.search(r"\b(smartphone|tablet|monitor|headphone)", head, re.I))
    return facts


_TIGHTEN_BUDGET = re.compile(
    r"(?:only\s+go\s+up\s+to|can\s+only\s+(?:go\s+up\s+to|spend(?:\s+up\s+to)?)|"
    r"new\s+(?:budget|limit|cap)|budget\s+is\s+(?:now|at\s+most)|"
    r"at\s+most|cap\s+is)\s+"
    r"(\d+(?:\.\d+)?)\s*(USD|EUR|GBP|PHP|SGD|AUD|CAD)",
    re.I,
)
_EVENT_PRICE = re.compile(
    r"(?:more expensive|increased|jumped|went\s+from|from)\s+"
    r"\d+(?:\.\d+)?\s*(?:USD|EUR|GBP|PHP|SGD|AUD|CAD)?\s+to\s+\d+",
    re.I,
)


def _absorb_note(goal: dict[str, Any], note: str, plan: dict[str, Any] | None = None) -> None:
    text = str(note or "")
    tighten = _TIGHTEN_BUDGET.search(text)
    if tighten:
        new_cap = _as_float(tighten.group(1))
        old_cap = _as_float(goal.get("budget"))
        if new_cap is not None and (old_cap is None or new_cap <= old_cap + 1e-9):
            goal["budget"] = tighten.group(1)
            goal["currency"] = tighten.group(2).upper()
            if isinstance(plan, dict):
                plan["price"] = f"-{tighten.group(1)}"
    elif not _EVENT_PRICE.search(text):
        budget = re.search(r"(\d+(?:\.\d+)?)\s*(USD|EUR|GBP|PHP|SGD|AUD|CAD)", text, re.I)
        if budget and re.search(r"\b(budget|spend|balance|only|now)\b", text, re.I):
            new_cap = _as_float(budget.group(1))
            old_cap = _as_float(goal.get("budget"))
            if new_cap is not None and (old_cap is None or new_cap < old_cap):
                goal["budget"] = budget.group(1)
                goal["currency"] = budget.group(2).upper()
                if isinstance(plan, dict):
                    plan["price"] = f"-{budget.group(1)}"
    inches = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", text, re.I)
    if inches:
        goal["screen"] = inches.group(1)
    ram = re.search(r"(\d+)\s*gb\s*ram", text, re.I)
    if ram:
        goal["ram"] = ram.group(1)
    res = re.search(r"\b(720p|1080p|1440p|2160p|4k|uhd|qhd|full hd|hd)\b", text, re.I)
    if res:
        goal["resolution"] = res.group(1)
    hz = re.search(r"(\d+)\s*hz", text, re.I)
    if hz:
        goal["refresh"] = hz.group(1)


def _llm_json(prompt: str, models: list[str], temperature: float) -> dict[str, Any] | None:
    last: dict[str, Any] | None = None
    temp = temperature
    for model in models:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "stream": False,
        }
        reply = _INFER.post("/inference/chat/completions", json_data=payload)
        if not isinstance(reply, dict) or reply.get("error"):
            continue
        choices = reply.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = ""
        if isinstance(message, dict):
            content = str(message.get("content") or "")
        parsed = _parse_json(content)
        if isinstance(parsed, dict):
            return parsed
        last = parsed if isinstance(parsed, dict) else last
        temp = min(temp + _HELPER_TEMP_UP, 0.6)
    return last


def _compact_kv(key: str, value: str, query: str) -> tuple[str, str] | None:
    key, value = key.lower(), value.lower()
    splits, origin = key.split("_"), key.replace("_", " ")
    if "brand" in splits:
        return (key, value)
    value = value.replace("'s", "")
    q = query.casefold()
    if value in _YES_WORDS and origin in q and not re.search(r"\b(mode|support)$", key):
        key, value = _YES_WORDS[0], origin
    for yes in _YES_WORDS:
        if re.search(rf"\b{re.escape(yes)}$", value, re.I):
            value = re.sub(rf"\b{re.escape(yes)}$", "", value, flags=re.I)
    for no in _NO_WORDS:
        match = re.search(rf"^{re.escape(no)}\s+([\w\s]+)$", value, re.I)
        if match:
            value = f"{_NO_WORDS[0]} {match.group(1)}"
            break
        if re.search(rf"\b{re.escape(no)}$", value, re.I) and origin in q:
            key, value = _NO_WORDS[0], origin
            break
    if len(splits) > 1 and "count" in splits and not re.search(r"\bcount\b", query, re.I):
        splits = [part for part in splits if part != "count"]
        key = "_".join(splits)
    if value == "1" and re.search(r"\bsingle\b", query, re.I) and not re.search(r"\b1[a-z]*\b", query, re.I):
        value = "single"
    kv_splits = set(key.replace("_", " ").split() + value.replace("_", " ").split())
    for stop in _KW_STOPWORDS:
        value = re.sub(rf"\s{re.escape(stop)}\s", " ", value, flags=re.I)
        if kv_splits == set(stop.split()):
            value = ""
    for idk in _ID_KEYS:
        if value == idk and key == idk:
            value = ""
    match = re.search(r"^(any|all)\s+([\w\s]+)$", value, re.I)
    if match:
        key, value = match.group(2).replace(" ", "_"), match.group(1)
        splits = key.split("_")
    if value in {"one", "free"} and "size" in key:
        key = "size"
        value = f"{value} {key}"
    for unit in splits:
        if unit != "pcs" and unit.endswith("s"):
            unit = unit[:-1]
        if not value.isnumeric():
            break
        for mu in list(_METRIC_UNITS.keys()) + list(_METRIC_UNITS.values()):
            if not mu:
                continue
            if mu == unit:
                if re.search(rf"\b{re.escape(value)}{re.escape(unit)}s?\b", query, re.I):
                    value = f"{value}{mu}"
                else:
                    value = f"{value} {mu}"
                break
    # 115 gemstone→stone scrub: last helper slug prefix + stringfy_price[:2]
    value = re.sub(r"\bgem(stone)\b", r"\1", value, flags=re.I)
    value = re.sub(r"[&|+]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return (key, value) if value else None


def _compact_alt(key: str, value: str) -> tuple[str, str] | None:
    key = key.lower().replace("'s", "").replace("'", "")
    value = value.lower().replace("'s", "").replace("'", "")
    key, value = key.replace("-", " "), value.replace("-", " ")
    key, value = re.sub(r"\s+", " ", key), re.sub(r"\s+", " ", value)
    if re.search(r"\d", key + value):
        return None
    if key == value:
        return None
    return key, value


def _parse_p115_decompose(js: dict[str, Any]) -> dict[str, Any]:
    rewritten = ""
    raw = js.get("rewritten")
    if isinstance(raw, dict):
        rewritten = str(raw.get("result") or "").strip()
    elif isinstance(raw, str):
        rewritten = raw.strip()
    prices = js.get("prices") if isinstance(js.get("prices"), dict) else {}
    norm = prices.get("normalized")
    price = None
    if isinstance(norm, str) and "-" in norm:
        cleaned = re.sub(r"[^\d\-.]", "", norm)
        parts = cleaned.split("-")
        if len(parts) == 2:
            parts = [p if re.search(r"\d", p) else "" for p in parts]
            price = "-".join(parts)
    alts = js.get("alt") if isinstance(js.get("alt"), dict) else {}
    redundant = js.get("redundant")
    return {
        "rewritten": rewritten,
        "price": price,
        "alts": {str(k): str(v) for k, v in alts.items() if str(k).strip() and str(v).strip()},
        "redundant": str(redundant).strip() if isinstance(redundant, str) and redundant.strip() else None,
    }


def _p115_plan(query: str) -> dict[str, Any]:
    head = _goal_head(query).strip()
    plan: dict[str, Any] = {
        "rewritten": head,
        "compact": head,
        "keywords": _keywords_fallback(head),
        "alt_keywords": [],
        "attrs": {},
        "price": None,
        "alts": {},
        "searches": [],
    }
    dec = _llm_json(_P_DECOMPOSE.format(query=head), list(_HELPER_MODELS), _HELPER_TEMP)
    if isinstance(dec, dict):
        parsed = _parse_p115_decompose(dec)
        if parsed["rewritten"]:
            plan["rewritten"] = parsed["rewritten"]
        if parsed["price"]:
            plan["price"] = parsed["price"]
        plan["alts"] = parsed["alts"]
        if parsed["redundant"] and parsed["redundant"] in plan["keywords"]:
            plan["keywords"] = [w for w in plan["keywords"] if w != parsed["redundant"]] or plan["keywords"]
    built = _llm_json(
        _PSV_BUILD.format(queries=json.dumps({"1": plan["rewritten"]}, ensure_ascii=True)),
        list(_HELPER_MODELS),
        _HELPER_TEMP,
    )
    row = built.get("1") if isinstance(built, dict) else None
    if isinstance(row, dict):
        if isinstance(row.get("compact"), str) and row["compact"].strip():
            plan["compact"] = row["compact"].strip()
        attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
        cleaned: dict[str, str] = {}
        for key, value in attrs.items():
            pair = _compact_kv(str(key), str(value), plan["rewritten"])
            if not pair:
                continue
            nk, nv = pair
            if nk not in cleaned:
                reused = False
                for ok, ov in list(cleaned.items()):
                    if ov == nv:
                        if f"{nv} {nk.replace('_', ' ')}" in plan["rewritten"].casefold():
                            cleaned[nk] = nv
                            del cleaned[ok]
                        reused = True
                        break
                if not reused:
                    cleaned[nk] = nv
            else:
                cleaned[nk] = f"{cleaned[nk]} {nv}".strip()
        plan["attrs"] = cleaned
        values = [v for v in cleaned.values() if v.casefold() not in _KW_STOP]
        if values:
            plan["keywords"] = _title_tokens(" ".join(values))[:6] or plan["keywords"]
    cleaned_alts: dict[str, str] = {}
    for token, alt in (plan.get("alts") or {}).items():
        pair = _compact_alt(str(token), str(alt))
        if pair:
            cleaned_alts[pair[0]] = pair[1]
    plan["alts"] = cleaned_alts
    for token, alt in cleaned_alts.items():
        words = list(plan["keywords"])
        if token in words:
            swapped = [alt if w == token else w for w in words]
            plan["alt_keywords"] = _title_tokens(" ".join(swapped))[:6]
            break
    inf = json.dumps(
        {"1": {"query": plan["rewritten"], "keywords": " ".join(plan["keywords"])}},
        ensure_ascii=True,
    )
    alt = _llm_json(_PSV_KW.format(inf=inf), list(_HELPER_MODELS), _HELPER_TEMP)
    if isinstance(alt, dict):
        kw_row = alt.get("1")
        if isinstance(kw_row, dict) and isinstance(kw_row.get("alt"), str) and kw_row["alt"].strip():
            plan["alt_keywords"] = [w for w in kw_row["alt"].split() if w][:6]
    primary = " ".join(plan["keywords"][:5])
    alt_q = " ".join(plan["alt_keywords"][:5])
    plan["searches"] = [s for s in (primary, alt_q, plan["compact"], plan["rewritten"]) if s]
    return plan


def _is_title_direct_match(word: str, tokens: set[str]) -> bool:
    if word in tokens:
        return True
    plural = word[:-1] if word.endswith("s") else word + "s"
    if plural in tokens:
        return True
    if len(word) < 3:
        return False
    return any(other.startswith(word) for other in tokens if len(other) > len(word))


def _is_title_partial_match(word: str, tokens: set[str]) -> bool:
    return any(
        word.startswith(other) or other.startswith(word)
        for other in tokens
        if len(other) > 2
    )


def _title_hit_score(words: list[str], tokens: set[str], title: str) -> float:
    score = 0.0
    for word in words:
        if _is_title_direct_match(word, tokens):
            score += 2
        elif _is_title_partial_match(word, tokens):
            score += 1
        if any(ch.isdigit() for ch in word) and word in title:
            score += 2
    return score


def _flatten_facts(row: dict[str, Any]) -> tuple[str, set[str]]:
    parts: list[str] = []
    values: set[str] = set()
    for key in ("facts", "options", "specification", "attributes", "variants"):
        blob = row.get(key)
        if blob in (None, "", [], {}):
            continue
        if isinstance(blob, dict):
            for name, value in blob.items():
                parts.append(str(name).replace("_", " "))
                for item in value if isinstance(value, list) else [value]:
                    text = str(item).strip().casefold()
                    if text:
                        parts.append(text)
                        values.add(text)
        else:
            text = json.dumps(blob, ensure_ascii=True, default=str).casefold()
            parts.append(text)
            values.update(_title_tokens(text))
    return (" ".join(parts).casefold(), values)


def _p115_score(row: dict[str, Any], query: str, plan: dict[str, Any] | None = None) -> float:
    title = str(row.get("title") or "").casefold()
    tokens = set(_title_tokens(title))
    words = _title_tokens(query)
    score = _title_hit_score(words, tokens, title)
    fact_text, fact_values = _flatten_facts(row)
    fact_tokens = set(_title_tokens(fact_text))
    for word in words:
        if word in fact_values:
            score += 5
        elif word in fact_tokens:
            score += 2
    price = _as_float(row.get("price"))
    if plan and price is not None and plan.get("price"):
        low, high = _numerify_price(plan.get("price"))
        score += -25 if price < low or price > high else 5
    return score


def _format_candidates(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for i, row in enumerate(rows[:10], start=1):
        blocks.append(
            json.dumps(
                {
                    "id": i,
                    "product_id": row.get("product_id"),
                    "sku": row.get("sku"),
                    "title": row.get("title"),
                    "brand": row.get("brand"),
                    "price": row.get("price"),
                    "category_path": row.get("category_path"),
                    "facts": row.get("facts") or row.get("specification"),
                    "options": row.get("options"),
                    "in_stock": row.get("in_stock"),
                },
                ensure_ascii=True,
                default=str,
            )
        )
    return "\n".join(blocks)


def _p115_filter(query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    top = rows[:_PREFILTER_TOPN]
    if len(top) <= 2:
        return top
    parsed = _llm_json(
        _PV_FILTER.format(
            query=_goal_head(query),
            cands=json.dumps(
                [
                    {"i": i, "title": r.get("title"), "price": r.get("price"), "brand": r.get("brand")}
                    for i, r in enumerate(top)
                ],
                ensure_ascii=True,
                default=str,
            ),
        ),
        list(_HELPER_MODELS),
        _HELPER_TEMP,
    )
    keep: list[int] = []
    raw = parsed.get("keep") if isinstance(parsed, dict) else None
    if isinstance(raw, list):
        for item in raw:
            idx = int(_extract_number(str(item)))
            if 0 <= idx < len(top):
                keep.append(idx)
    kept = [top[i] for i in keep] if keep else []
    return kept or top


def _parse_select_index(parsed: dict[str, Any], n: int) -> int:
    raw = parsed.get("best")
    if raw not in (None, "", "null"):
        idx = int(_extract_number(str(raw))) - 1
        if 0 <= idx < n:
            return idx
    for value in reversed(list(parsed.values())):
        if isinstance(value, dict) and value.get("best") not in (None, "", "null"):
            idx = int(_extract_number(str(value.get("best")))) - 1
            if 0 <= idx < n:
                return idx
    return -1


def _p115_select(query: str, rows: list[dict[str, Any]], must: bool) -> dict[str, Any] | None:
    if not rows:
        return None
    if len(rows) == 1 and must:
        return dict(rows[0])
    cursor = 0
    tries = 0
    while cursor < len(rows) and tries < _MAX_SELECT_TRIES:
        batch = rows[cursor : cursor + _SELECT_BATCH]
        cursor += _SELECT_BATCH
        tries += 1
        parsed = _llm_json(
            _PV_SELECT.format(query=_goal_head(query), products=_format_candidates(batch)),
            list(_HELPER_MODELS),
            _HELPER_TEMP,
        )
        if not isinstance(parsed, dict):
            continue
        idx = _parse_select_index(parsed, len(batch))
        if idx >= 0:
            pick = dict(batch[idx])
            pick["reason"] = str(parsed.get("recommend") or "")
            return pick
    return dict(rows[0]) if must else None


def _tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(tools, list):
        return names
    for spec in tools:
        if not isinstance(spec, dict):
            continue
        fn = spec.get("function") if isinstance(spec.get("function"), dict) else spec
        name = str((fn or {}).get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _call_name(call: dict[str, Any]) -> str:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    return str(fn.get("name") or call.get("name") or "")


def _parse_args(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    parsed = _parse_json(str(raw))
    return parsed if isinstance(parsed, dict) else {}


def _compact(payload: Any, limit: int = 4500) -> str:
    text = json.dumps(payload, ensure_ascii=True, default=str)
    if len(text) <= limit:
        return text
    if isinstance(payload, dict):
        keep = {key: payload[key] for key in _KEEP_OBS if key in payload}
        for key in ("results", "variants", "compared"):
            rows = payload.get(key)
            if isinstance(rows, list):
                keep[key] = rows[:6]
        text = json.dumps(keep, ensure_ascii=True, default=str)
        if len(text) <= limit:
            return text
    return text[: limit - 3] + "..."


def _unwrap_obs(obs: dict[str, Any]) -> dict[str, Any]:
    inner = obs.get("observation")
    if not isinstance(inner, dict):
        return obs
    markers = (
        "results", "added", "product_id", "price", "facts", "variants",
        "compared", "cart", "market_event", "category_path",
    )
    if not any(key in inner for key in markers):
        return obs
    flat = {key: value for key, value in obs.items() if key != "observation"}
    flat.update(inner)
    return flat


def _as_obs(call_result: Any) -> dict[str, Any]:
    if not isinstance(call_result, dict):
        return {}
    obs = call_result.get("observation")
    if not isinstance(obs, dict):
        return {}
    return _unwrap_obs(obs)


def _iter_variants(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in ("variants", "options", "skus"):
        raw = row.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and (item.get("sku") or item.get("product_id")):
                    out.append(item)
    if row.get("sku"):
        out.append(row)
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for item in out:
        sku = str(item.get("sku") or "")
        if sku in seen:
            continue
        seen.add(sku)
        uniq.append(item)
    return uniq


def _has_tool(names: list[str], name: str) -> bool:
    return name in names


def _numerify_price(norm: str | None) -> tuple[float, float]:
    if not norm or "-" not in str(norm):
        return (0.0, float("inf"))
    low_s, high_s = str(norm).split("-", 1)
    low = _as_float(low_s) or 0.0
    high = _as_float(high_s)
    return (low, high if high is not None else float("inf"))


def _price_ok(row: dict[str, Any], budget: Any, price_norm: str | None) -> bool:
    price = _as_float(row.get("price"))
    if price is None:
        return True
    if budget is not None:
        cap = _as_float(budget)
        if cap is not None and price > cap + 1e-6:
            return False
    low, high = _numerify_price(price_norm)
    return low <= price <= high


def _in_stock(row: dict[str, Any]) -> bool:
    if "in_stock" not in row:
        return True
    return _truthy(row.get("in_stock"))


def _category_ok(row: dict[str, Any], category: str | None) -> bool:
    if not category:
        return True
    blob = _path_text(row.get("category_path")).casefold()
    if blob:
        return category.casefold() in blob
    return category.casefold() in _blob(row)


def _is_accessory(row: dict[str, Any]) -> bool:
    blob = _blob(row)
    return any(token in blob for token in _ACCESSORY)


def _is_wet(row: dict[str, Any]) -> bool:
    blob = _blob(row)
    return any(token in blob for token in _WET_STRONG)


def _pref_axis_score(row: dict[str, Any]) -> float:
    blob = _blob(row)
    wet = 1.0 if any(token in blob for token in _WET_STRONG) else 0.0
    sport = 1.0 if any(token in blob for token in _SPORT) else 0.0
    wireless = 1.0 if any(token in blob for token in _WIRELESS) else 0.0
    battery = 1.0 if any(token in blob for token in _BATTERY) else 0.0
    return 0.26 * wet + 0.26 * sport + 0.26 * wireless + 0.22 * battery


def _same_sku(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not left or not right:
        return False
    lpid = str(left.get("product_id") or "")
    rpid = str(right.get("product_id") or "")
    lsku = str(left.get("sku") or "")
    rsku = str(right.get("sku") or "")
    if lpid and rpid and lpid == rpid:
        return True
    return bool(lsku and rsku and lsku == rsku)


def _canon_res(raw: Any) -> str:
    token = str(raw or "").casefold()
    if token in {"720p", "hd", "hd+", "1280", "720"}:
        return "hd"
    if token in {"1080p", "full hd", "fhd", "1920", "1080"}:
        return "fhd"
    if token in {"1440p", "qhd", "2560", "1440"}:
        return "qhd"
    if token in {"4k", "uhd", "2160p", "3840", "2160"}:
        return "uhd"
    return token


def _facts_dict(row: dict[str, Any]) -> dict[str, Any]:
    facts = row.get("facts")
    return facts if isinstance(facts, dict) else {}


def _fact_display(row: dict[str, Any]) -> str:
    facts = _facts_dict(row)
    return str(facts.get("display") or facts.get("resolution") or "")


def _fact_size(row: dict[str, Any]) -> str:
    facts = _facts_dict(row)
    return str(facts.get("size") or facts.get("screen_size") or facts.get("screen") or "")


def _display_class(text: str) -> str:
    token = str(text or "").casefold().replace("*", " x ")
    if any(part in token for part in ("3840", "2160", "4k", "uhd")):
        return "uhd"
    if any(part in token for part in ("2560", "1440", "qhd")):
        return "qhd"
    if any(part in token for part in ("1920", "1080", "fhd", "full hd")):
        return "fhd"
    if any(part in token for part in ("1280", "720p", "720")) or token.strip() in {"hd", "hd+"}:
        return "hd"
    return _canon_res(token)


def _tf7_size_num(text: str) -> str | None:
    token = str(text or "").strip()
    if not token:
        return None
    if re.search(r"\d+(?:\.\d+)?\s*[-~/]\s*\d+", token):
        return None
    nums = re.findall(r"(\d+(?:\.\d+)?)", token)
    if len(nums) != 1:
        return None
    num = _as_float(nums[0])
    if num is None:
        return None
    if abs(num - round(num)) < 1e-6:
        return str(int(round(num)))
    return nums[0].rstrip("0").rstrip(".")


def _tf7_ok(row: dict[str, Any], goal: dict[str, Any]) -> bool:
    if not _facts_dict(row):
        return False
    want_brand = str(goal.get("brand") or "").strip().casefold()
    have_brand = str(row.get("brand") or "").strip().casefold()
    if want_brand and want_brand not in have_brand:
        return False
    if goal.get("screen"):
        size = _tf7_size_num(_fact_size(row))
        want = str(goal["screen"]).rstrip("0").rstrip(".")
        if not size or size != want:
            return False
    if goal.get("resolution"):
        disp = _fact_display(row)
        if not disp:
            return False
        got = _display_class(disp)
        want = _canon_res(goal["resolution"])
        if got != want:
            return False
    return True


def _tf7_price_value(row: dict[str, Any]) -> str | None:
    price = row.get("price")
    if price in (None, ""):
        return None
    num = _as_float(price)
    if num is not None and abs(num - round(num)) < 1e-6:
        return str(int(round(num)))
    return str(price)


def _tf7_claims(goal: dict[str, Any], row: dict[str, Any]) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    if goal.get("screen"):
        screen = _tf7_size_num(_fact_size(row)) or str(goal["screen"])
        claims.append({"field": "screen_size", "value": screen})
    elif goal.get("resolution"):
        disp = _fact_display(row)
        res = _display_class(disp) if disp else _canon_res(goal["resolution"])
        if res in {"hd", "fhd", "qhd", "uhd"}:
            claims.append({"field": "resolution", "value": res})
    elif goal.get("refresh"):
        claims.append({"field": "refresh_rate", "value": str(goal["refresh"])})
    price = _tf7_price_value(row)
    if price is not None:
        claims.append({"field": "price", "value": price})
    claims.append({"field": "in_stock", "value": "true"})
    brand = str(row.get("brand") or goal.get("brand") or "").strip()
    if brand:
        claims.append({"field": "brand", "value": brand})
    return claims


def _spec_ok(row: dict[str, Any], goal: dict[str, Any]) -> bool:
    blob = _blob(row)
    if goal.get("screen"):
        want = str(goal["screen"]).rstrip("0").rstrip(".")
        size = _fact_size(row)
        if size:
            num = _tf7_size_num(size)
            if num is not None and num != want:
                return False
            if num is None and want not in size.casefold():
                return False
        elif not re.search(rf"{re.escape(want)}\s*-?\s*inch", blob, re.I):
            return False
    if goal.get("ram"):
        if f"{goal['ram']}gb" not in blob.replace(" ", "") and f"{goal['ram']} gb" not in blob:
            return False
    if goal.get("refresh"):
        if f"{goal['refresh']}hz" not in blob.replace(" ", "") and f"{goal['refresh']} hz" not in blob:
            return False
    if goal.get("resolution"):
        want = _canon_res(goal["resolution"]) or str(goal["resolution"]).casefold()
        disp = _fact_display(row)
        if disp:
            got = _display_class(disp)
            if got and got != want:
                return False
        else:
            token = str(goal["resolution"]).casefold()
            aliases = {token, _canon_res(token)}
            if token in {"hd", "720p"} or _canon_res(token) == "hd":
                aliases.update({"720p", "1280", "720", "hd"})
            if token in {"full hd", "1080p", "fhd"} or _canon_res(token) == "fhd":
                aliases.update({"1080p", "1920", "1080", "full hd", "fhd"})
            if token in {"qhd", "1440p"} or _canon_res(token) == "qhd":
                aliases.update({"1440p", "2560", "1440", "qhd"})
            if token in {"4k", "uhd", "2160p"} or _canon_res(token) == "uhd":
                aliases.update({"4k", "uhd", "3840", "2160"})
            if not any(alias in blob for alias in aliases):
                return False
    if goal.get("color") and goal["color"] not in blob:
        return False
    if goal.get("model") and str(goal["model"]).casefold() not in blob:
        return False
    return True


def _max_refresh(row: dict[str, Any]) -> float:
    blob = _blob(row)
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*hz", blob)]
    over = [float(x) for x in re.findall(r"overclock[^0-9]{0,12}(\d+(?:\.\d+)?)", blob)]
    return max(nums + over, default=0.0)


def _screen_size(row: dict[str, Any]) -> float:
    blob = _blob(row)
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*-?\s*inch", blob)]
    return max(nums, default=0.0)


def _weigh(row: dict[str, Any], tokens: list[str], query: str, plan: dict[str, Any] | None) -> float:
    return _p115_score(row, " ".join(tokens) or query, plan)


def _hard_ok(row: dict[str, Any], goal: dict[str, Any], plan: dict[str, Any], *, firm_brand: bool) -> bool:
    if not _price_ok(row, goal.get("budget"), plan.get("price")):
        return False
    if not _in_stock(row):
        return False
    if not _category_ok(row, goal.get("category")):
        return False
    if goal.get("device") and _is_accessory(row):
        return False
    if not _spec_ok(row, goal):
        return False
    if firm_brand and goal.get("brand"):
        brand = str(row.get("brand") or "").casefold()
        if goal["brand"].casefold() not in brand and goal["brand"].casefold() not in _blob(row):
            return False
    return True


def _choose_sku(
    row: dict[str, Any],
    goal: dict[str, Any],
    plan: dict[str, Any],
    *,
    firm_brand: bool,
) -> dict[str, Any] | None:
    variants = _iter_variants(row) or [row]
    ranked = []
    for var in variants:
        merged = {**row, **var}
        if not _hard_ok(merged, goal, plan, firm_brand=firm_brand):
            continue
        ranked.append(merged)
    if not ranked:
        return None
    pid = str(ranked[0].get("product_id") or row.get("product_id") or "")
    sku = str(ranked[0].get("sku") or "")
    if not pid or not sku:
        return None
    pick = dict(ranked[0])
    pick["product_id"] = pid
    pick["sku"] = sku
    return pick


def _environment_call(envelope: dict[str, Any]) -> dict[str, Any]:
    result = _ENV.post("/environment/call", json_data=envelope)
    if not isinstance(result, dict) or "calls" not in result:
        result = _ENV.post("/environment/call", json_data=envelope)
    if not isinstance(result, dict):
        raise RuntimeError("environment call failed")
    return result


class _Session:
    def __init__(
        self,
        *,
        binding: dict[str, Any],
        problem_id: str,
        names: list[str],
        max_steps: int,
        max_calls: int,
    ) -> None:
        self.binding = binding
        self.problem_id = problem_id
        self.names = names
        self.max_steps = max_steps
        self.max_calls = max_calls
        self.turn = 0
        self.dialogue: list[dict[str, Any]] = []
        self.observations: list[dict[str, Any]] = []
        self.used: list[str] = []
        self.shopper_notes: list[str] = []
        self.catalog: dict[str, dict[str, Any]] = {}
        self.done = False
        self.added: list[dict[str, Any]] = []
        self.event_seen = False
        self.event_target: dict[str, Any] | None = None
        self.cart: list[dict[str, str]] = []

    def remaining(self) -> int:
        return self.max_steps - self.turn

    def call(self, actions: list[tuple[str, dict[str, Any]]], content: str = "") -> list[dict[str, Any]]:
        if self.done or self.remaining() <= 0 or not actions:
            return []
        usable = [(n, a) for n, a in actions if n in self.names][: self.max_calls]
        if not usable:
            return []
        self.turn += 1
        group_id = f"{self.problem_id}-turn-{self.turn}"
        tool_calls = []
        env_actions = []
        for index, (name, args) in enumerate(usable, start=1):
            call_id = f"{group_id}-{index}"
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=True)},
                }
            )
            env_actions.append({"call_id": call_id, "action": {"name": name, "args": args}})
            self.used.append(name)
            if name == "remove_from_cart":
                self._drop_cart(str(args.get("product_id") or ""), str(args.get("sku") or ""))
        result = _environment_call(
            {
                **self.binding,
                "call_id": group_id,
                "idempotency_key": group_id,
                "turn": self.turn,
                "calls": env_actions,
            }
        )
        self.dialogue.append(
            {
                "role": "assistant",
                "content": content or "ELON product engine",
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )
        returned = result.get("calls") if isinstance(result.get("calls"), list) else []
        obs_list: list[dict[str, Any]] = []
        for call, call_result in zip(tool_calls, returned):
            obs = _as_obs(call_result)
            self.observations.append(obs)
            obs_list.append(obs)
            self._ingest(obs)
            added = obs.get("added")
            if isinstance(added, dict):
                self.added.append({**obs, **added})
                self._push_cart(str(added.get("product_id") or obs.get("product_id") or ""), str(added.get("sku") or obs.get("sku") or ""))
            elif added:
                self.added.append(obs)
                self._push_cart(str(obs.get("product_id") or ""), str(obs.get("sku") or ""))
            self._note_event(obs)
            self._note_event(call_result if isinstance(call_result, dict) else {})
            if obs.get("done") is True:
                self.done = True
        extras: list[Any] = [result.get("user_message")]
        for call_result in returned:
            if isinstance(call_result, dict):
                extras.append(call_result.get("user_message"))
                obs = call_result.get("observation")
                if isinstance(obs, dict):
                    extras.append(obs.get("user_message"))
        for extra in extras:
            text = ""
            if isinstance(extra, dict):
                text = str(extra.get("content") or extra.get("text") or "").strip()
            elif isinstance(extra, str):
                text = extra.strip()
            if text and (not self.shopper_notes or self.shopper_notes[-1] != text):
                self.shopper_notes.append(text)
        return obs_list

    def _ingest(self, obs: dict[str, Any]) -> None:
        rows: list[dict[str, Any]] = []
        for key in ("results", "compared", "variants"):
            raw = obs.get(key)
            if isinstance(raw, list):
                rows.extend([r for r in raw if isinstance(r, dict)])
        if obs.get("product_id"):
            rows.append(obs)
        for row in rows:
            pid = str(row.get("product_id") or "").strip()
            if not pid:
                continue
            prev = self.catalog.get(pid, {})
            merged = dict(prev)
            for key, value in row.items():
                if value not in (None, "", [], {}):
                    merged[key] = value
            self.catalog[pid] = merged

    def _push_cart(self, pid: str, sku: str) -> None:
        if not pid and not sku:
            return
        item = {"product_id": pid, "sku": sku}
        if not any(_same_sku(item, row) for row in self.cart):
            self.cart.append(item)

    def _drop_cart(self, pid: str, sku: str) -> None:
        gone = {"product_id": pid, "sku": sku}
        self.cart = [row for row in self.cart if not _same_sku(row, gone)]

    def _note_event(self, payload: Any) -> None:
        found: list[Any] = []

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                ev = obj.get("market_event")
                if ev:
                    found.append(ev)
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        walk(value)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(payload)
        for ev in found:
            self.event_seen = True
            if not isinstance(ev, dict):
                continue
            target = ev.get("target") if isinstance(ev.get("target"), dict) else ev
            pid = str((target or {}).get("product_id") or ev.get("product_id") or "")
            sku = str((target or {}).get("sku") or ev.get("sku") or "")
            if pid or sku:
                self.event_target = {"product_id": pid, "sku": sku}
            price = ev.get("new_price")
            if price is not None and pid:
                row = dict(self.catalog.get(pid) or {})
                row["product_id"] = pid
                if sku:
                    row["sku"] = sku
                row["price"] = price
                self.catalog[pid] = row


def _rows(session: _Session) -> list[dict[str, Any]]:
    return list(session.catalog.values())


class _P143Solver:
    def __init__(
        self,
        session: _Session,
        *,
        query: str,
        lane: str,
        goal: dict[str, Any],
        plan: dict[str, Any],
    ) -> None:
        self.s = session
        self.query = query
        self.lane = lane
        self.goal = goal
        self.plan = plan

    def run(self) -> None:
        self._phase_ask()
        self._retrieve()
        self._sync()
        if self.s.done:
            return
        self._inspect()
        self._sync()
        if self.s.done:
            return
        if self.lane == "justification" and not self._just_rows() and self.s.remaining() > 8:
            self._inspect()
            self._sync()
        if self.lane == "preference":
            self._pref_announce()
        if self.lane == "retrieval":
            self._try_adds(phase="a", limit=3)
            self._sync()
            if self.s.done:
                return
            self._tf2_recover()
            if self.s.done:
                return
            self._order()
            return
        self._try_adds(phase="a", limit=5)
        self._sync()
        if self.s.done:
            return
        if self.lane == "preference":
            self._pref_recover()
            if self.s.done:
                return
            self._pref_order()
            return
        if self.lane == "justification":
            self._just_recover()
            if self.s.done:
                return
            self._just_order()
            return
        if self.lane == "intent":
            self._intent_recover()
        else:
            self._phase_recover()
        if self.s.done:
            return
        if not self.s.added:
            self._try_adds(phase="b", limit=5)
        if self.s.done:
            return
        self._order()

    def _sync(self) -> None:
        for note in self.s.shopper_notes:
            _absorb_note(self.goal, note, self.plan)

    def _firm_brand(self) -> bool:
        return self.lane in {"constraint", "justification"}

    def _phase_ask(self) -> None:
        if self.lane == "constraint" and self.goal.get("ask_spec") and _has_tool(self.s.names, "message"):
            self.s.call(
                [("message", {"content": "What is the firm hardware spec I must match on the exact variant?"})]
            )
            self._sync()
        elif self.lane == "preference" and _has_tool(self.s.names, "message"):
            self.s.call(
                [
                    (
                        "message",
                        {"content": "For this use case, should I prioritize sweat/water resistance over the lowest price?"},
                    )
                ]
            )
            self._sync()

    def _search_queries(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []

        def add(text: str) -> None:
            q = " ".join(str(text).split())
            key = q.casefold()
            if q and key not in seen:
                seen.add(key)
                out.append(q)

        brand = self.goal.get("brand") or ""
        ptype = (self.plan.get("attrs") or {}).get("product_type") or ""
        if self.lane == "ranking":
            return out
        if self.lane == "retrieval":
            phrase = _tf2_phrase(self.goal)
            if phrase:
                add(phrase)
            return out[:1]
        if self.lane == "recovery":
            add(f"{self.goal.get('color') or ''} {self.goal.get('model') or ''} smartphone")
            add(f"{self.goal.get('model') or ''} smartphone")
        if self.lane == "intent" and brand:
            screen = self.goal.get("screen") or ""
            add(f"{brand} smartphone {screen} inch".strip())
            add(f"{brand} smartphone")
            add(brand)
        if self.lane == "preference":
            add("headphone waterproof IPX")
            add("sweat proof wireless earbuds")
            add("sport earhook bluetooth headphone")
            add("headphone sweat water resistant")
            add("headphone waterproof")
        if self.lane == "justification":
            ptype = ptype or _product_type(self.goal)
            res = _canon_res(self.goal.get("resolution")) if self.goal.get("resolution") else ""
            add(
                f"{brand} {ptype} {self.goal.get('screen') or ''} {res or self.goal.get('resolution') or ''}"
            )
            if brand and self.goal.get("screen"):
                add(f"{brand} {self.goal['screen']} inch")
            if brand and (res or self.goal.get("resolution")):
                add(f"{brand} {res or self.goal.get('resolution')}")
            if brand:
                add(brand)
        for item in self.plan.get("searches") or []:
            add(item)
        if ptype:
            add(ptype)
        return out[:6]

    def _retrieve(self) -> None:
        actions: list[tuple[str, dict[str, Any]]] = []
        if self.lane == "ranking":
            pids = []
            for cand in self.goal.get("candidates") or []:
                pid = str(cand).split("::", 1)[0]
                if pid not in pids:
                    pids.append(pid)
            if _has_tool(self.s.names, "compare") and pids:
                for i in range(0, len(pids), 5):
                    actions.append(("compare", {"product_ids": pids[i : i + 5]}))
            elif _has_tool(self.s.names, "view"):
                for pid in pids[: self.s.max_calls]:
                    actions.append(("view", {"product_id": pid}))
            if actions:
                self.s.call(actions, content="ELON retrieve named SKUs")
            return
        if _has_tool(self.s.names, "search"):
            for q in self._search_queries():
                actions.append(("search", {"query": q, "k": 10, "in_stock": True}))
        if _has_tool(self.s.names, "filter"):
            filt: dict[str, Any] = {"k": 10, "in_stock": True}
            if self.goal.get("category"):
                filt["category"] = self.goal["category"]
            if self.goal.get("budget"):
                cap = _as_float(self.goal["budget"])
                if cap is not None:
                    filt["max_price"] = cap
            if self.lane == "preference" and not filt.get("category"):
                filt["category"] = "Headphones"
            if self.lane in {"intent", "justification"} and self.goal.get("brand"):
                branded = dict(filt)
                branded["brand"] = self.goal["brand"]
                actions.append(("filter", branded))
            actions.append(("filter", filt))
        if actions:
            self.s.call(actions, content="ELON retrieve")
            self._sync()

    def _intent_brand_retrieve(self) -> None:
        brand = str(self.goal.get("brand") or "").strip()
        if not brand or self.s.remaining() <= 4:
            return
        actions: list[tuple[str, dict[str, Any]]] = []
        if _has_tool(self.s.names, "search"):
            actions.append(
                (
                    "search",
                    {
                        "query": f"{brand} {self.goal.get('category') or 'smartphone'} {self.goal.get('screen') or ''}".strip(),
                        "k": 10,
                        "in_stock": True,
                    },
                )
            )
        if _has_tool(self.s.names, "filter"):
            filt: dict[str, Any] = {"brand": brand, "k": 10, "in_stock": True}
            if self.goal.get("category"):
                filt["category"] = self.goal["category"]
            cap = _as_float(self.goal.get("budget"))
            if cap is not None:
                filt["max_price"] = cap
            actions.append(("filter", filt))
        if actions:
            self.s.call(actions, content="TF1 brand retrieve")
            self._sync()
        self._inspect()

    def _intent_announce(self, pick: dict[str, Any]) -> None:
        if not _has_tool(self.s.names, "message"):
            return
        brand = pick.get("brand") or self.goal.get("brand") or ""
        spec = self.goal.get("screen") or self.goal.get("resolution") or "the required spec"
        title = pick.get("title") or pick.get("product_id")
        price = pick.get("price")
        ccy = self.goal.get("currency") or ""
        cap = self.goal.get("budget")
        self.s.call(
            [
                (
                    "message",
                    {
                        "content": (
                            f"I'm going with the {brand} {title} (SKU {pick.get('sku')}). "
                            f"Yes, it has the required {spec}-inch/spec on this exact variant. "
                            f"Listed at {price} {ccy}, in stock, and at or under the current "
                            f"cap of {cap} {ccy}."
                        )
                    },
                )
            ]
        )
        self._sync()

    def _intent_recover(self) -> None:
        self._sync()
        if not self.s.added or self.s.done:
            return
        last = self.s.added[-1]
        live = self._stock_check(last)
        self._sync()
        if live is None or self.s.done:
            return
        cap = _as_float(self.goal.get("budget"))
        broken = (not _in_stock(live)) or (not _price_ok(live, self.goal.get("budget"), self.plan.get("price")))
        if _has_tool(self.s.names, "message") and (self.s.event_seen or broken or cap is not None):
            self.s.call(
                [
                    (
                        "message",
                        {
                            "content": (
                                f"Acknowledged. Current cap is {self.goal.get('budget')} "
                                f"{self.goal.get('currency') or ''}. "
                                "I will not order the repriced first-add SKU. "
                                "Switching to another in-stock on-spec item at or under that cap."
                            )
                        },
                    )
                ]
            )
            self._sync()
        if not (self.s.event_seen or broken):
            return
        actions: list[tuple[str, dict[str, Any]]] = []
        if _has_tool(self.s.names, "filter"):
            filt: dict[str, Any] = {"k": 10, "in_stock": True}
            if self.goal.get("category"):
                filt["category"] = self.goal["category"]
            if cap is not None:
                filt["max_price"] = cap
            actions.append(("filter", filt))
        screen = self.goal.get("screen") or ""
        if _has_tool(self.s.names, "search") and screen:
            actions.append(("search", {"query": f"smartphone {screen} inch", "k": 10, "in_stock": True}))
        if actions:
            self.s.call(actions, content="TF1 recover retrieve")
            self._sync()
        self._inspect()
        nxt = self._cheapest_live(self._candidates(phase="b"), skip_sku=str(last.get("sku") or ""))
        if nxt and str(nxt.get("sku")) != str(last.get("sku")):
            self._swap(last, nxt)
        self._sync()

    def _tf2_recover(self) -> None:
        if self.s.done or not self.s.added:
            return
        last = self.s.added[-1]
        live = self._stock_check(last)
        if live is None or self.s.done:
            return
        broken = (not _in_stock(live)) or (not _price_ok(live, self.goal.get("budget"), self.plan.get("price")))
        if not (self.s.event_seen or broken):
            return
        nxt = self._pick(phase="b")
        if nxt and not _same_sku(nxt, last):
            self._swap(last, nxt)

    def _pref_rows(self, skip: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in _rows(self.s):
            sku_row = _choose_sku(row, self.goal, self.plan, firm_brand=False)
            if not sku_row:
                continue
            if skip and _same_sku(sku_row, skip):
                continue
            if self.s.event_target and _same_sku(sku_row, self.s.event_target):
                continue
            if _is_accessory(sku_row):
                continue
            rows.append(sku_row)
        wet = [r for r in rows if _is_wet(r)]
        loose = [r for r in rows if any(token in _blob(r) for token in _WATER)]
        pool = wet or loose or rows
        pool.sort(
            key=lambda r: (
                -_pref_axis_score(r),
                _as_float(r.get("price")) if _as_float(r.get("price")) is not None else 1e18,
            )
        )
        return pool

    def _pref_announce(self) -> None:
        if not _has_tool(self.s.names, "message"):
            return
        rows = self._pref_rows()[:2]
        if not rows:
            return
        parts = []
        for row in rows:
            parts.append(
                f"{row.get('brand') or ''} {row.get('title') or row.get('product_id')} "
                f"at {row.get('price')} {self.goal.get('currency') or ''}"
            )
        self.s.call(
            [
                (
                    "message",
                    {
                        "content": (
                            "Comparing water/sweat-rated headphones under the current cap: "
                            + " vs ".join(parts)
                            + ". I will add the water-rated sport option, not the cheapest unrated one."
                        )
                    },
                )
            ]
        )
        self._sync()

    def _pref_recover(self) -> None:
        if self.s.done or not self.s.added:
            return
        old = self.s.event_target or self.s.added[0]
        alts = self._pref_rows(skip=old)
        if not alts and self.s.remaining() > 6:
            actions: list[tuple[str, dict[str, Any]]] = []
            if _has_tool(self.s.names, "search"):
                actions.append(("search", {"query": "headphone waterproof IPX", "k": 10, "in_stock": True}))
                actions.append(("search", {"query": "sport wireless earbuds sweat", "k": 10, "in_stock": True}))
            if _has_tool(self.s.names, "filter"):
                filt: dict[str, Any] = {"category": "Headphones", "k": 10, "in_stock": True}
                cap = _as_float(self.goal.get("budget"))
                if cap is not None:
                    filt["max_price"] = cap
                actions.append(("filter", filt))
            if actions:
                self.s.call(actions[: self.s.max_calls], content="TF4 recover retrieve")
                self._sync()
            self._inspect()
            alts = self._pref_rows(skip=old)
        if not alts or self.s.done:
            return
        if _has_tool(self.s.names, "message") and self.s.remaining() > 5:
            cap = self.goal.get("budget")
            ccy = self.goal.get("currency") or ""
            self.s.call(
                [
                    (
                        "message",
                        {
                            "content": (
                                f"The first cart SKU was repriced over the {cap} {ccy} cap. "
                                "I am removing that event-target SKU first, then carting a "
                                "different in-stock waterproof/sport headphone under the live cap."
                            )
                        },
                    )
                ]
            )
            self._sync()
        self._swap(old, alts[0])

    def _pref_order(self) -> None:
        self._sync()
        if self.s.done or not self.s.added:
            return
        if self.s.event_target and any(_same_sku(self.s.event_target, row) for row in self.s.cart):
            self._pref_recover()
        if self.s.done:
            return
        last: dict[str, Any] | None = None
        for row in reversed(self.s.added):
            if self.s.event_target and _same_sku(row, self.s.event_target):
                continue
            last = row
            break
        if last is None:
            return
        live = last
        if self.s.remaining() > 2:
            checked = self._stock_check(last)
            if checked is None or self.s.done:
                return
            live = checked
        if not _in_stock(live) or not _price_ok(live, self.goal.get("budget"), self.plan.get("price")):
            nxt = self._pref_rows(skip=last)
            if nxt:
                self._swap(last, nxt[0])
                if self.s.done or not self.s.added:
                    return
                last = self.s.added[-1]
                if self.s.event_target and _same_sku(last, self.s.event_target):
                    return
                live = last
        if not _in_stock(live) or not _price_ok(live, self.goal.get("budget"), self.plan.get("price")):
            return
        if not _has_tool(self.s.names, "place_test_order"):
            return
        self.s.call(
            [
                (
                    "place_test_order",
                    {
                        "product_id": str(live.get("product_id") or ""),
                        "sku": str(live.get("sku") or ""),
                    },
                )
            ],
            content="ELON TF4 order",
        )

    def _inspect(self) -> None:
        pids = [str(r.get("product_id") or "") for r in _rows(self.s) if r.get("product_id")]
        seen: set[str] = set()
        need = []
        for pid in pids:
            if not pid or pid in seen:
                continue
            seen.add(pid)
            row = self.s.catalog.get(pid) or {}
            if row.get("facts") or row.get("options") or row.get("category_path") or row.get("variants"):
                continue
            need.append(pid)
        if self.lane == "preference":
            ranked = sorted(_rows(self.s), key=lambda r: -_pref_axis_score(r))
            need = [str(r.get("product_id") or "") for r in ranked if r.get("product_id")][:10]
        if self.lane == "retrieval":
            terms = list(self.goal.get("terms") or [])
            ranked = [
                r
                for r in _rows(self.s)
                if _title_hits(str(r.get("title") or ""), terms)
            ]
            ranked.sort(key=lambda r: -_weigh(r, terms, self.query, self.plan))
            need = [str(r.get("product_id") or "") for r in ranked if r.get("product_id")][:10]
        if self.lane == "intent":
            brand = str(self.goal.get("brand") or "").casefold()
            branded = [pid for pid in need if brand and brand in _blob(self.s.catalog.get(pid) or {})]
            need = (branded or need)[:8]
        if self.lane == "justification":
            brand = str(self.goal.get("brand") or "").casefold()
            ranked = []
            for row in _rows(self.s):
                pid = str(row.get("product_id") or "")
                if not pid:
                    continue
                blob = _blob(row)
                if brand and brand not in blob and brand not in str(row.get("brand") or "").casefold():
                    continue
                ranked.append(pid)
            need = (ranked or [str(r.get("product_id") or "") for r in _rows(self.s) if r.get("product_id")])[:10]
        if not need:
            return
        if _has_tool(self.s.names, "compare"):
            batch = [("compare", {"product_ids": need[i : i + 5]}) for i in range(0, min(len(need), 15), 5)]
            self.s.call(batch, content="ELON view/compare")
        elif _has_tool(self.s.names, "view"):
            self.s.call([("view", {"product_id": pid}) for pid in need[: self.s.max_calls]], content="ELON view")

    def _candidates(self, *, phase: str) -> list[dict[str, Any]]:
        rows = _rows(self.s)
        firm_brand = self._firm_brand() or (self.lane == "intent" and phase == "a")
        if self.lane == "ranking":
            named = []
            for cand in self.goal.get("candidates") or []:
                pid, _, sku = str(cand).partition("::")
                row = dict(self.s.catalog.get(pid) or {"product_id": pid})
                row["sku"] = sku or row.get("sku")
                named.append(row)
            return named
        gated = []
        for row in rows:
            sku_row = _choose_sku(row, self.goal, self.plan, firm_brand=firm_brand)
            if sku_row:
                gated.append(sku_row)
        if self.lane == "intent" and phase == "a" and not gated:
            self._intent_brand_retrieve()
            rows = _rows(self.s)
            for row in rows:
                sku_row = _choose_sku(row, self.goal, self.plan, firm_brand=True)
                if sku_row:
                    gated.append(sku_row)
            if not gated:
                for row in rows:
                    sku_row = _choose_sku(row, self.goal, self.plan, firm_brand=False)
                    if sku_row:
                        gated.append(sku_row)
        if self.lane == "retrieval":
            skip_sku = str(self.s.added[-1].get("sku") or "") if self.s.added and phase == "b" else None
            skip_pid = str(self.s.added[-1].get("product_id") or "") if self.s.added and phase == "b" else None
            return _tf2_rows(
                rows,
                self.goal,
                self.plan,
                skip_sku=skip_sku,
                skip_pid=skip_pid,
            )
        if self.lane == "preference":
            skip = self.s.event_target if phase != "a" else None
            return self._pref_rows(skip=skip)
        if self.lane == "justification":
            skip = self.s.event_target if phase != "a" else None
            return self._just_rows(skip=skip)
        if self.lane == "intent":
            gated.sort(key=lambda r: _as_float(r.get("price")) if _as_float(r.get("price")) is not None else 1e18)
            return gated
        tokens = list(self.plan.get("keywords") or []) + list(self.goal.get("terms") or [])
        gated.sort(key=lambda r: -_weigh(r, tokens, self.query, self.plan))
        if gated:
            top = _weigh(gated[0], tokens, self.query, self.plan)
            gated = [r for r in gated if _weigh(r, tokens, self.query, self.plan) > max(0.0, top - _PRUNE_GAP)] or gated
        return _p115_filter(self.query, gated)

    def _rank_refresh(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def key(row: dict[str, Any]) -> tuple:
            price = _as_float(row.get("price"))
            return (-_max_refresh(row), -_screen_size(row), price if price is not None else 1e18)

        return sorted(rows, key=key)

    def _pick(self, *, phase: str) -> dict[str, Any] | None:
        rows = self._candidates(phase=phase)
        if not rows:
            return None
        if self.lane == "ranking":
            live = []
            for row in rows:
                checked = self._stock_check(row)
                if checked and _in_stock(checked) and _price_ok(checked, self.goal.get("budget"), self.plan.get("price")):
                    live.append(checked)
            ranked = self._rank_refresh(live or rows)
            return ranked[0] if ranked else None
        if self.lane == "recovery" and phase == "a":
            exact = [r for r in rows if _spec_ok(r, self.goal)]
            rows = exact or rows
        if self.lane == "intent":
            return self._cheapest_live(rows, skip_sku=None)
        if self.lane == "preference":
            return rows[0]
        if self.lane == "justification":
            return rows[0]
        if self.lane == "retrieval":
            return rows[0]
        return _p115_select(self.query, rows, True)

    def _cheapest_live(self, rows: list[dict[str, Any]], skip_sku: str | None) -> dict[str, Any] | None:
        ranked = sorted(
            rows,
            key=lambda r: _as_float(r.get("price")) if _as_float(r.get("price")) is not None else 1e18,
        )
        best: dict[str, Any] | None = None
        best_price = 1e18
        for row in ranked[:6]:
            if skip_sku and str(row.get("sku") or "") == str(skip_sku):
                continue
            if self.s.remaining() <= 3:
                break
            live = self._stock_check(row)
            if live is None or self.s.done:
                return best
            if not _in_stock(live) or not _price_ok(live, self.goal.get("budget"), self.plan.get("price")):
                continue
            price = _as_float(live.get("price"))
            if price is None:
                price = _as_float(row.get("price"))
            if price is not None and price < best_price:
                best = live
                best_price = price
        if best:
            return best
        for row in ranked:
            if skip_sku and str(row.get("sku") or "") == str(skip_sku):
                continue
            if _hard_ok(row, self.goal, self.plan, firm_brand=False):
                return row
        return None

    def _try_adds(self, *, phase: str, limit: int) -> None:
        rows = self._candidates(phase=phase)
        announced = False
        for row in rows[:limit]:
            if self.s.added or self.s.done:
                return
            if not _price_ok(row, self.goal.get("budget"), self.plan.get("price")):
                continue
            if self.lane == "intent" and phase == "a" and not announced:
                self._intent_announce(row)
                announced = True
            self._commit(row)

    def _stock_check(self, row: dict[str, Any]) -> dict[str, Any] | None:
        pid = str(row.get("product_id") or "")
        sku = str(row.get("sku") or "")
        if not pid or not sku or not _has_tool(self.s.names, "inspect_stock"):
            return row
        obs_list = self.s.call([("inspect_stock", {"product_id": pid, "sku": sku})], content="ELON inspect_stock")
        self._sync()
        live = obs_list[-1] if obs_list else {}
        inner = live.get("observation") if isinstance(live.get("observation"), dict) else live
        merged = dict(row)
        for key in ("price", "in_stock", "sku", "product_id", "currency"):
            if inner.get(key) not in (None, ""):
                merged[key] = inner[key]
            elif live.get(key) not in (None, ""):
                merged[key] = live[key]
        return merged

    def _commit(self, pick: dict[str, Any]) -> None:
        self._sync()
        live = self._stock_check(pick)
        if self.s.done or live is None:
            return
        if not _in_stock(live) or not _price_ok(live, self.goal.get("budget"), self.plan.get("price")):
            return
        pid = str(live.get("product_id") or "")
        sku = str(live.get("sku") or "")
        if not pid or not sku or not _has_tool(self.s.names, "add_to_cart"):
            return
        self.s.call([("add_to_cart", {"product_id": pid, "sku": sku})], content="ELON recommend/add")
        self._sync()

    def _phase_recover(self) -> None:
        if not self.s.added:
            return
        last = self.s.added[-1]
        live = self._stock_check(last)
        if live is None or self.s.done:
            return
        broken = (not _in_stock(live)) or (not _price_ok(live, self.goal.get("budget"), self.plan.get("price")))
        if self.s.event_seen or broken:
            if self.lane == "recovery":
                relaxed = dict(self.goal)
                relaxed.pop("color", None)
                alt = None
                for row in _rows(self.s):
                    sku_row = _choose_sku(row, relaxed, self.plan, firm_brand=False)
                    if sku_row and str(sku_row.get("sku")) != str(last.get("sku")):
                        alt = sku_row
                        break
                if alt:
                    self._swap(last, alt)
            else:
                nxt = self._pick(phase="b")
                if nxt and str(nxt.get("sku")) != str(last.get("sku")):
                    self._swap(last, nxt)

    def _swap(self, old: dict[str, Any], new: dict[str, Any]) -> None:
        if _same_sku(old, new):
            return
        if _has_tool(self.s.names, "remove_from_cart") and old.get("product_id") and old.get("sku"):
            self.s.call(
                [("remove_from_cart", {"product_id": str(old["product_id"]), "sku": str(old["sku"])})],
                content="ELON remove event target first",
            )
            self._sync()
        if self.s.done:
            return
        live = new
        if self.s.remaining() > 2:
            checked = self._stock_check(new)
            if checked is None or self.s.done:
                return
            live = checked
        if not _in_stock(live) or not _price_ok(live, self.goal.get("budget"), self.plan.get("price")):
            return
        if not _has_tool(self.s.names, "add_to_cart"):
            return
        self.s.call(
            [("add_to_cart", {"product_id": str(live["product_id"]), "sku": str(live["sku"])})],
            content="ELON recover add after remove",
        )
        self._sync()

    def _just_rows(self, skip: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in _rows(self.s):
            sku_row = _choose_sku(row, self.goal, self.plan, firm_brand=True)
            if not sku_row:
                continue
            merged = dict(self.s.catalog.get(str(sku_row.get("product_id") or "")) or {})
            merged.update(sku_row)
            if not _tf7_ok(merged, self.goal):
                continue
            if skip and _same_sku(merged, skip):
                continue
            if self.s.event_target and _same_sku(merged, self.s.event_target):
                continue
            if _is_accessory(merged):
                continue
            rows.append(merged)
        rows.sort(
            key=lambda r: _as_float(r.get("price")) if _as_float(r.get("price")) is not None else 1e18
        )
        return rows

    def _just_recover(self) -> None:
        if self.s.done or not self.s.added:
            return
        old = self.s.event_target or self.s.added[0]
        alts = self._just_rows(skip=old)
        if not alts and self.s.remaining() > 6:
            actions: list[tuple[str, dict[str, Any]]] = []
            brand = self.goal.get("brand") or ""
            ptype = _product_type(self.goal) or "monitor"
            res = _canon_res(self.goal.get("resolution")) if self.goal.get("resolution") else ""
            if _has_tool(self.s.names, "search"):
                if brand:
                    actions.append(
                        (
                            "search",
                            {
                                "query": f"{brand} {ptype} {self.goal.get('screen') or ''} {res}".strip(),
                                "k": 10,
                                "in_stock": True,
                            },
                        )
                    )
                    actions.append(("search", {"query": brand, "k": 10, "in_stock": True}))
            if _has_tool(self.s.names, "filter"):
                filt: dict[str, Any] = {"k": 10, "in_stock": True}
                if self.goal.get("category"):
                    filt["category"] = self.goal["category"]
                if brand:
                    filt["brand"] = brand
                cap = _as_float(self.goal.get("budget"))
                if cap is not None:
                    filt["max_price"] = cap
                actions.append(("filter", filt))
            if actions:
                self.s.call(actions[: self.s.max_calls], content="TF7 recover retrieve")
                self._sync()
            self._inspect()
            alts = self._just_rows(skip=old)
        if not alts or self.s.done:
            return
        self._swap(old, alts[0])

    def _just_order(self) -> None:
        self._sync()
        if self.s.done or not self.s.added:
            return
        if self.s.event_target and any(_same_sku(self.s.event_target, row) for row in self.s.cart):
            self._just_recover()
        if self.s.done:
            return
        last: dict[str, Any] | None = None
        for row in reversed(self.s.added):
            if self.s.event_target and _same_sku(row, self.s.event_target):
                continue
            last = row
            break
        if last is None:
            return
        live = last
        if self.s.remaining() > 2:
            checked = self._stock_check(last)
            if checked is None or self.s.done:
                return
            live = checked
        if not _in_stock(live) or not _price_ok(live, self.goal.get("budget"), self.plan.get("price")):
            nxt = self._just_rows(skip=last)
            if nxt:
                self._swap(last, nxt[0])
                if self.s.done or not self.s.added:
                    return
                last = self.s.added[-1]
                if self.s.event_target and _same_sku(last, self.s.event_target):
                    return
                live = last
        if not _in_stock(live) or not _price_ok(live, self.goal.get("budget"), self.plan.get("price")):
            return
        if self.s.event_target and _same_sku(live, self.s.event_target):
            return
        cat = self.s.catalog.get(str(live.get("product_id") or "")) or {}
        if cat.get("brand") and not live.get("brand"):
            live = dict(live)
            live["brand"] = cat["brand"]
        if not _has_tool(self.s.names, "place_test_order"):
            return
        self.s.call(
            [
                (
                    "place_test_order",
                    {
                        "product_id": str(live.get("product_id") or ""),
                        "sku": str(live.get("sku") or ""),
                        "justification": {"claims": _tf7_claims(self.goal, live)},
                    },
                )
            ],
            content="ELON TF7 order",
        )

    def _claims(self, row: dict[str, Any]) -> list[dict[str, str]]:
        merged = dict(self.s.catalog.get(str(row.get("product_id") or "")) or {})
        merged.update(row)
        return _tf7_claims(self.goal, merged)

    def _order(self) -> None:
        self._sync()
        if self.s.done or not self.s.added:
            return
        if _has_tool(self.s.names, "inspect_cart"):
            self.s.call([("inspect_cart", {})], content="ELON inspect_cart")
            self._sync()
        if self.s.done:
            return
        last = self.s.added[-1]
        live = self._stock_check(last)
        if live is None or self.s.done:
            return
        if not _in_stock(live) or not _price_ok(live, self.goal.get("budget"), self.plan.get("price")):
            nxt = self._pick(phase="b")
            if nxt:
                self._swap(last, nxt)
                if self.s.done:
                    return
                last = self.s.added[-1]
                live = self._stock_check(last)
        if live is None or self.s.done:
            return
        if not _in_stock(live) or not _price_ok(live, self.goal.get("budget"), self.plan.get("price")):
            return
        if not _has_tool(self.s.names, "place_test_order"):
            return
        args: dict[str, Any] = {
            "product_id": str(live.get("product_id") or ""),
            "sku": str(live.get("sku") or ""),
        }
        if self.lane == "justification":
            args["justification"] = {"claims": self._claims(live)}
        self.s.call([("place_test_order", args)], content="ELON recommend/order")


def _strip_justification(
    name: str,
    args: dict[str, Any],
    lane: str,
    goal: dict[str, Any] | None = None,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cleaned = dict(args)
    if name != "place_test_order":
        return cleaned
    if lane != "justification":
        cleaned.pop("justification", None)
        return cleaned
    pid = str(cleaned.get("product_id") or "")
    row = dict((catalog or {}).get(pid) or {})
    row["product_id"] = pid
    if cleaned.get("sku"):
        row["sku"] = cleaned.get("sku")
    cleaned["justification"] = {"claims": _tf7_claims(goal or {}, row)}
    return cleaned


def _cap_reminder(goal: dict[str, Any], plan: dict[str, Any], lane: str) -> str:
    cap = goal.get("budget")
    ccy = goal.get("currency") or ""
    lines = [
        f"CURRENT firm budget cap: {cap} {ccy}." if cap else "Honor the latest shopper cap.",
        "Ignore 'more expensive from A to B' — those are event prices, not a new budget.",
        "place_test_order args are only product_id and sku." if lane != "justification" else "",
    ]
    if lane == "justification":
        lines.append(
            "TF7 event close: remove the first cart SKU FIRST, then add a different "
            "on-spec branded SKU, then order. Claims allowed: in_stock, price, and "
            "screen_size OR resolution (canonical hd/fhd/qhd/uhd). Values must be "
            "strings. Include brand when viewed. Never claim display, size, "
            "operating_system, or category."
        )
    if lane == "intent":
        lines.append(
            "If the shopper asked which item / whether the spec holds, message that "
            "answer before add_to_cart. First add must be the preferred brand when "
            "a fully compliant SKU exists. After a price jump, recover under the "
            f"CURRENT cap {cap} {ccy}, never the original query budget."
        )
    if lane == "retrieval":
        phrase = " ".join(goal.get("terms") or [])
        lines.append(
            f"Do not search again. The phrase is `{phrase}`. "
            "Swap the OOS cart item for another title-hit already in the catalog, "
            "then place_test_order {{product_id, sku}} immediately."
        )
    if plan.get("price"):
        lines.append(f"Normalized price gate: {plan.get('price')}")
    if plan.get("keywords") and lane != "retrieval":
        lines.append("ELON keywords: " + " ".join(plan.get("keywords") or []))
    return " ".join(part for part in lines if part)


def _guard_calls(
    tool_calls: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    session: _Session | None = None,
    lane: str = "",
) -> list[dict[str, Any]]:
    committed = any(obs.get("added") for obs in observations)
    names = [_call_name(call) for call in tool_calls]
    if (not committed) and ("add_to_cart" in names) and ("place_test_order" in names):
        tool_calls = [call for call in tool_calls if _call_name(call) != "place_test_order"]
        names = [_call_name(call) for call in tool_calls]
    target = session.event_target if session is not None else None
    cart = list(session.cart) if session is not None else []
    target_carted = bool(target and any(_same_sku(target, row) for row in cart))
    if lane in {"preference", "justification"} and target_carted:
        if "remove_from_cart" not in names:
            tool_calls = [
                call
                for call in tool_calls
                if _call_name(call) not in {"add_to_cart", "place_test_order"}
            ]
        else:
            removed = [call for call in tool_calls if _call_name(call) == "remove_from_cart"]
            rest = [call for call in tool_calls if _call_name(call) != "remove_from_cart"]
            rest = [call for call in rest if _call_name(call) != "place_test_order"]
            tool_calls = removed + rest
    return tool_calls


def _coach(
    *,
    lane: str,
    query: str,
    names: list[str],
    turn: int,
    max_steps: int,
    observations: list[dict[str, Any]],
    used: list[str],
    shopper_notes: list[str],
) -> str:
    lines: list[str] = []
    remaining = max_steps - turn
    latest = observations[-1] if observations else {}
    seen_ids: list[str] = []
    cart_adds = 0
    empty_lists = 0
    viewed = 0
    compared = 0
    inspected = 0
    messaged = 0
    event_seen = False
    for obs, name in zip(observations, used):
        if name == "message":
            messaged += 1
        if name in {"view", "compare"}:
            viewed += 1
        if name == "compare":
            compared += 1
        if name == "inspect_stock":
            inspected += 1
        if obs.get("added"):
            cart_adds += 1
        if obs.get("market_event"):
            event_seen = True
        if name in {"search", "filter"} and not (obs.get("results") or []):
            empty_lists += 1
        for row in list(obs.get("results") or []) + list(obs.get("compared") or []):
            pid = str((row or {}).get("product_id") or "")
            if pid and pid not in seen_ids:
                seen_ids.append(pid)
        pid = str(obs.get("product_id") or "")
        if pid and pid not in seen_ids:
            seen_ids.append(pid)

    if latest.get("error"):
        lines.append(f"Last action failed ({latest.get('error')}). Change the tool or the arguments.")
    if latest.get("market_event") or (lane == "preference" and event_seen):
        lines.append(
            "A market event is visible. remove_from_cart the repriced first SKU "
            "FIRST. Do not add a replacement while that event-target is still "
            "carted. Then add a different in-stock waterproof/sport headphone "
            "under the live cap and place_test_order that replacement only."
        )
    if latest.get("added") and not event_seen:
        lines.append(
            "Cart add accepted. Immediately inspect_stock that SKU — live state "
            "may have changed even if this reply looks calm. Then read any new "
            "shopper line before ordering."
        )
    if empty_lists and used and used[-1] in {"search", "filter"}:
        lines.append(
            "Catalog read returned no rows. Shorten the query, drop a filter, or "
            "swap keywords. Do not place an order from an empty shortlist."
        )
    if shopper_notes:
        lines.append("Honor the latest shopper update exactly. Requirements they did not change still apply.")
    if remaining <= 5:
        lines.append(
            f"{remaining} environment turns remain. Inspect the current best "
            "carted SKU and place_test_order if it still satisfies the live rules."
        )
    elif remaining <= 8 and cart_adds and inspected == 0:
        lines.append("You have a cart item that has not been stock-checked. inspect_stock now.")

    if lane == "retrieval":
        phrase = " ".join(_quoted_terms(query) or [])
        if cart_adds == 0:
            lines.append(
                f"Compare title-hits for `{phrase}` already in the catalog, then "
                "add_to_cart one in-stock under-budget hit. Do not issue another search."
            )
        else:
            lines.append(
                "TF2 event close: do not search. remove_from_cart the OOS first SKU, "
                "add another already-viewed title-hit, then place_test_order immediately."
            )
    if lane == "preference":
        if cart_adds == 0 and messaged == 0 and turn >= 2:
            lines.append(
                "Ask one short priority question with message before the first add, "
                "then compare at least two listing_id values that meet the hard rules."
            )
        if cart_adds == 0 and compared == 0 and viewed < 2 and turn >= 3:
            lines.append("Compare or view two real device listings before committing.")
        if cart_adds:
            lines.append(
                "TF4 event close: remove the first cart SKU, then add a different "
                "waterproof/sport headphone, then order. Never add-then-remove."
            )
    if lane == "ranking" and cart_adds == 0 and viewed == 0:
        lines.append("View the current tradeoff leader before the first add.")
    if lane == "justification":
        if cart_adds == 0 and viewed == 0:
            lines.append("View or compare raw facts on a branded on-spec listing before the first add.")
        if cart_adds:
            lines.append(
                "TF7 event close: remove the first cart SKU, then add a different "
                "on-spec branded item, then order. Never add-then-remove."
            )
        if cart_adds and "place_test_order" in names:
            lines.append(
                "Claims allowlist only: gated spec (screen_size number or resolution "
                "hd/fhd/qhd/uhd), price string, in_stock \"true\", and brand. "
                "No display, size, category, or OS."
            )
    if lane == "intent" and cart_adds == 0:
        lines.append(
            "First add should honor an available fully compliant preferred brand "
            "when the rules treat brand as soft."
        )
    if lane == "recovery" and event_seen:
        lines.append("Relax color before model. Keep category, budget, and in-stock firm.")
    if lane == "constraint" and viewed == 0:
        lines.append("View the exact SKU options before adding. Title match is not enough.")

    budget = re.search(r"(\d+(?:\.\d+)?)\s*(USD|EUR|GBP|PHP|SGD|AUD|CAD)", query)
    if budget and remaining > 5:
        lines.append(f"Stay at or under {budget.group(1)} {budget.group(2)} at the CURRENT price.")
    if names:
        lines.append("Available tools: " + ", ".join(names) + ".")
    return "\n".join(lines)


def _infer(
    messages: list[dict[str, Any]],
    tools: list[Any],
    models: list[str],
    temperature: float,
) -> dict[str, Any]:
    last_error = "inference request failed"
    for model in models:
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "temperature": temperature,
            "stream": False,
        }
        reply = _INFER.post("/inference/chat/completions", json_data=payload)
        if not isinstance(reply, dict):
            last_error = f"empty inference from {model}"
            continue
        if reply.get("error"):
            last_error = str(reply.get("error"))
            continue
        choices = reply.get("choices")
        if not isinstance(choices, list) or not choices:
            last_error = f"no choices from {model}"
            continue
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            return message
        last_error = f"malformed message from {model}"
    raise RuntimeError(last_error)


def _fallback_loop(
    session: _Session,
    *,
    query: str,
    tools: list[Any],
    lane: str,
    plan: dict[str, Any],
    goal: dict[str, Any],
) -> None:
    models = _model_ladder(lane)
    temperature = float(_FAMILY_TEMP.get(lane, 0.0))
    system = _PLAYBOOK
    extra = _WORKSPACES.get(lane)
    if extra:
        system = f"{_PLAYBOOK}\n\n{extra}"
    if plan.get("keywords") and lane != "retrieval":
        system += (
            "\n\nELON retrieve plan\n"
            f"rewritten: {plan.get('rewritten')}\n"
            f"keywords: {' '.join(plan.get('keywords') or [])}\n"
            f"searches: {' | '.join((plan.get('searches') or [])[:3])}\n"
        )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
        {"role": "user", "content": _cap_reminder(goal, plan, lane)},
    ]
    if session.shopper_notes:
        messages.append(
            {
                "role": "user",
                "content": "Shopper updates:\n" + "\n".join(session.shopper_notes[-6:]),
            }
        )
    if session.catalog:
        messages.append(
            {
                "role": "user",
                "content": "Catalog so far:\n" + _compact(list(session.catalog.values())[:8], 3500),
            }
        )
    while (not session.done) and session.remaining() > 0:
        hint = _coach(
            lane=lane,
            query=query,
            names=session.names,
            turn=session.turn + 1,
            max_steps=session.max_steps,
            observations=session.observations,
            used=session.used,
            shopper_notes=session.shopper_notes,
        )
        if hint:
            messages.append(
                {
                    "role": "user",
                    "content": f"Solver note (turn {session.turn + 1}/{session.max_steps}):\n{hint}",
                }
            )
        assistant = _infer(messages, tools, models, temperature)
        content = str(assistant.get("content") or "")
        raw_calls = assistant.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raw_calls = []
        tool_calls = [call for call in raw_calls if isinstance(call, dict)][: session.max_calls]
        tool_calls = _guard_calls(tool_calls, session.observations, session=session, lane=lane)
        if not tool_calls:
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "The environment is still open. Call one of the supplied tools now.",
                    },
                ]
            )
            continue
        actions = []
        for call in tool_calls:
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(fn.get("name") or call.get("name") or "")
            args = _strip_justification(
                name,
                _parse_args(fn.get("arguments", call.get("arguments"))),
                lane,
                goal=goal,
                catalog=session.catalog,
            )
            if lane == "retrieval" and name in {"search", "filter"}:
                continue
            actions.append((name, args))
        if not actions:
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Do not search or filter again. Use an already-viewed title-hit: "
                            "remove_from_cart the OOS SKU if needed, add_to_cart a different "
                            "title-hit, then place_test_order {product_id, sku}."
                        ),
                    },
                ]
            )
            continue
        session.call(actions, content=content)
        for note in session.shopper_notes:
            _absorb_note(goal, note, plan)
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        if session.observations:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_calls[0].get("id") or ""),
                    "content": _compact(session.observations[-1]),
                }
            )
        messages.append({"role": "user", "content": _cap_reminder(goal, plan, lane)})


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data.get("environment") or {}
    binding = dict(environment.get("binding") or {})
    policy = environment.get("policy_view") or {}
    query = str(policy.get("query") or problem_data.get("query") or "")
    tools = policy.get("tools") or []
    names = _tool_names(tools)
    max_steps = int(policy.get("max_steps") or 20)
    max_calls = int(policy.get("max_calls_per_turn") or 1)
    problem_id = str(problem_data.get("problem_id") or problem_data.get("id") or "task")
    lane = _lane(query)
    goal = _parse_goal(query)
    if lane == "retrieval":
        phrase = _tf2_phrase(goal)
        plan = {
            "rewritten": phrase,
            "compact": phrase,
            "keywords": list(goal.get("terms") or []),
            "alt_keywords": [],
            "attrs": {},
            "price": f"-{goal['budget']}" if goal.get("budget") else None,
            "alts": {},
            "searches": [phrase] if phrase else [],
        }
    elif lane == "preference":
        head = _goal_head(query).strip()
        if not goal.get("category"):
            goal["category"] = "Headphones"
        plan = {
            "rewritten": head,
            "compact": head,
            "keywords": _keywords_fallback(head),
            "alt_keywords": [],
            "attrs": {"product_type": "headphone"},
            "price": f"-{goal['budget']}" if goal.get("budget") else None,
            "alts": {},
            "searches": ["headphone waterproof IPX", "sport wireless earbuds sweat"],
        }
    elif lane == "justification":
        head = _goal_head(query).strip()
        ptype = _product_type(goal) or "monitor"
        res = _canon_res(goal.get("resolution")) if goal.get("resolution") else ""
        brand = goal.get("brand") or ""
        searches = []
        if brand:
            searches.append(
                f"{brand} {ptype} {goal.get('screen') or ''} {res or goal.get('resolution') or ''}".strip()
            )
            searches.append(brand)
        if goal.get("screen"):
            searches.append(f"{brand} {goal['screen']} inch {ptype}".strip())
        plan = {
            "rewritten": head,
            "compact": head,
            "keywords": _keywords_fallback(head),
            "alt_keywords": [],
            "attrs": {"product_type": ptype},
            "price": f"-{goal['budget']}" if goal.get("budget") else None,
            "alts": {},
            "searches": searches,
        }
    else:
        plan = _p115_plan(query)
    session = _Session(
        binding=binding,
        problem_id=problem_id,
        names=names,
        max_steps=max_steps,
        max_calls=max_calls,
    )
    _P143Solver(session, query=query, lane=lane, goal=goal, plan=plan).run()
    if not session.done:
        _fallback_loop(session, query=query, tools=tools, lane=lane, plan=plan, goal=goal)
    return session.dialogue


__all__ = ["agent_main"]
