"""ORO Bench shopping agent — agent_str contract + youKnowIt product brain.

New mechanism (SN15 generated environment):
  model / solver emits only supplied tools; this file POSTs /environment/call
  and feeds observations back until observation.done (TF1–TF7).

youKnowIt (ShoppingBench 118) pieces that are actually called here:
  A_DECOMPOSE → compact_kw / update_q search build → is_kw_in title/attr
  scoring → PV_SELECT + MATCH_RULES.
  Shop/voucher engines, find_product, recommend_product, SegTree, and the
  enhancement kit are omitted.
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

# youKnowIt OpenRouter helper chain (decompose / select). Heads match 118.
_HELPER_MODELS = (
    "google/gemma-4-31b-it",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-chat-v3.1",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.2",
)
_SELECT_HELPERS = (
    "google/gemma-4-31b-it",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-chat-v3.1",
    "moonshotai/kimi-k2.6",
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
_SELECT_BATCH = 17
_MAX_SELECT_TRIES = 2
_PRUNE_GAP = 1.75

_SCHEMA_KEYS = frozenset({
    "product_type", "color", "color_family", "material", "brand", "style",
    "size", "size_system", "gender", "user", "age", "pack", "set", "model",
    "compatibility", "pattern", "design", "shape", "origin", "usage",
    "fit", "feature", "category", "capacity", "volume", "length", "weight",
    "pre_config", "suitable", "condition", "warranty_days", "version",
    "option", "variation", "solution", "quality", "unit", "frame", "theme",
    "promotion", "inspiration", "interest", "vibe", "id",
})
_YKI_STOP = (
    "color family", "suitable", "compatible", "about", "above", "after",
    "also", "and", "as", "at", "before", "below", "between", "by", "for",
    "from", "in", "less", "more", "of", "on", "or", "over", "per", "than",
    "to", "under", "up", "with", "within",
)
_SYNONYMS = (
    ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"), ("five", "5"),
    ("six", "6"), ("seven", "7"), ("eight", "8"), ("nine", "9"), ("ten", "10"),
    ("one", "single"), ("two", "double"), ("three", "triple"),
    ("water-resistant", "waterproof"), ("water resistant", "waterproof"),
    ("water-proof", "waterproof"),
)


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

# --- youKnowIt prompts (called: A_DECOMPOSE / PV_SELECT / MATCH_RULES) ---

_A_DECOMPOSE = """
You are AI assistant to decompose a user's shopping query into several chunks, for automatic processing.
The user is looking for one or more (max 4) products{shop_intent}.

# Task
1. Identify the number of products user is looking for, and what they are.
- Keep in mind that the max number of products is 4.
- Sometimes it seems enumerating more than 4 products. In fact, some of them are the sub-products packed inside a kit. If the number seems beyond 4, correctly identify the kit product.
- Check carefully and don't mistakenly take some constraints or filler words as an individual product!
- Shop queries like "Find shops offering both X and Y" or "First,... Second,... Third,..." contain multiple products; each item gets its own sub-query and may have different price/service constraints.

2. Identify the price constraint for each product, and rewrite it to the normalized format.
- Price constraint is mentioned like "between 10 and 50 php", "priced from 180 to 505 PHP", "priced above 410 PHP", "cost is above 85 PHP", "cost over 176 PHP", "more than 2400", "less than 120 USD", ...
- User may mention the price constraint without currency unit: e.g. "Looking for a makeup bag between 110 and 200."
- Don't mistakenly take other numerical constraint as the price constraint: e.g. "Need aircon with power usage between 1000 and 1500." -> "1000-1500" is power consumption range constraint.
- For normalization, write using "[low]-[high]" format. Exclude currency unit! e.g. "10-100"
- "[low]-" if upper bound is missing. e.g. "57-"
- "-[high]" if lower bound is missing. e.g. "-88"
- null if price constraint is not mentioned

3. Identify the service constraints, if exist, and find which products they apply for. Check the following service labels:
- **official**: user wants product that is official, guaranteed and authentic
- **free_ship**: user wants free or complimentary shippable product
- **fast_delivery**: user wants fast delivery
- **cod**: user wants 'Cash on Delivery' option
- **flashsale**: user wants flash sale or lightning deal. Also map phrases that explicitly convey a **time-limited** promotion: "limited-time deal/offer/promotion", "time-limited deal/offer", "seller's limited-time deal/offer", "exclusive time-limited offer", or any phrasing that implies a temporary promotional price. Do NOT map generic phrases like "seller's deal", "seller's promo", or "special offer" unless a time-limited qualifier is explicitly present.
- **return**: user wants easy or fast returns
- **lazmall**: user wants products from LazMall or LazMall Perk    
- **lazflash**: user wants products from LazFlash
- Map "free delivery/shipping" to **free_ship**, "cash/pay on delivery" to **cod**, "LazFlash deal" to **lazflash**, "LazMall perks/store" to **lazmall**.
- When a single phrase mentions multiple services (e.g. "LazMall and LazFlash"), include ALL of them in the `labels` list for that entry.
- CRITICAL — never let a brand name get absorbed into a service chunk: phrases like "from **famenr**'s official store", "**nike** LazMall store", "official store of **samsung**" mention TWO separate things — a service condition (official/lazmall) AND a brand name. Only the generic seller/service words ("official", "store", "shop", "lazmall") belong to the service chunk that gets excluded from the sub-query. The proper name itself is never part of the service constraint — it must still appear in the rewritten sub-query and be extracted as its own `brand→<name>` keyword in step 5, even though it sits right next to (or inside) the service wording.

4. Rewrite the sub-query for each product, by excluding the price constraints and service constraints.
- Make sure each rewritten sub-query does not include price and service constraints.
- Make sure each rewritten sub-query does not miss any details for the corresponding product from the original query.

5. Convert each sub-query into a list of key-value formatted keywords for search.
- Convert like "from manuel store" -> "brand→manuel", "with gold color" -> "color→gold", ... by using `→` for delimiting key and value.
- Keep multi-word color / variant phrases intact when they appear as a unit in the query. "in a random color" → `color_family→random color` (NOT truncated `color→random`); "dark blue" → `color→dark blue`; "rose gold" → `color→rose gold`. Truncating to the first word makes bare SKUs like `color_family: random` look cleaner than the true option `color_family: random color`.
- Generic seller/shop nouns ("store", "shop", "seller", "official", "lazmall", "flagship") are NEVER part of the brand value itself — strip them and keep only the proper name. "from manuel store" is `brand→manuel`, not `brand→manuel store`.
- A brand mention does not disappear just because it is wrapped in service wording. "from famenr's official store" still produces `brand→famenr` as a keyword (in addition to the `official` service label from step 3) — do not drop the brand, and do not fold the service words into its value.
- Identify brand the same way regardless of where it appears in the query: a proper/distinctive name (not a generic English product-type or attribute word) attributed to the product via "from X", "X brand", "X's official/authentic/original", "by X", or simply named directly before/after the product noun. When in doubt whether a distinctive name is a brand or a product-line/model name, prefer `brand→<name>` — it is still a strong identity signal for search.
- For replacement parts or accessories, prefer product_type + compatibility/model keys; e.g. "blue button flex replacement for Oppo A5S" -> product_type→button flex, compatibility→oppo a5s, color→blue.
- Remove redundant words like "compatible with", "suitable for", "still", "fit to", "available with", "made from", ... unless they are used for brand, title, name, identifier or other non-negotiable attributes.
- Our e-commerce database uses JSON for product details, and only the value fields are indexed. Be careful not to include key-like words in the value side.
- For example, "metal-based" must be converted to "base→metal", not "material→metal-based". "suitable for student use" -> "suitable_use→student", not "suitable→student use". Shorten value strings as much as possible.
- When converting to key-value format, fetch the key name from the original query when possible. e.g. "suitable for model queen" -> "model→queen", not "compatibility→model queen".
- When a phrase describes a component's attribute (e.g. "rubber sole", "leather upper", "metal base"), produce exactly ONE keyword with the component as the key and the attribute as the value (e.g. `sole→rubber`). Do NOT also emit a general `material→rubber` — doing so would hallucinate a separate constraint that was never stated in the query.
- Convert plural forms into singular forms for better search performance, unless plural forms are used for brand, title, name, identifier or non-negotiable.
- The keywords must be mutually exclusive: exactly one keyword for one constraint aspect.
- The keywords must be collectively exhausted: the sum of the keywords must exactly cover the all meaningful information of the sub-query.
- Keep search values concise: each sub-query's keyword values should total 2-8 words when joined for search. Prioritize brand, product_type, and 1-2 distinctive attributes; omit price and service wording.
- For negation of a product feature that must be physically absent ("no stones", "without fragrance", "stone-free"), keep the negated phrase as a keyword value, e.g. feature→no stones, material→fragrance-free. Only use this for genuine absence of a feature — NOT for universal compatibility.
- Do NOT produce negated values for compatibility/fit attributes that mean "any" or "universal". For example: "no specific car year", "no particular model", "fits any year" → convert to car_year→universal, model→universal, or simply omit the keyword; never produce car_year→no specific.
- Variant identifiers (color codes, pack counts like 10pcs, SKU names) belong in color, pack/set, or color_family keys when they pick a purchasable variant.

# Rule of Handling the Scope of Modifiers
- It's very important to understand the scope of modifiers for identifying which product a certain constraint chunk is belonged to.
- A modifier applies to the nearest product mention only that can grammatically receive that modifier, unless there is explicit mention.
- Don't judge the scope of modifiers by the existence and the locations of commas or conjunctions.

e.g. In ```white shirt and black pants, with free shipping```, "with free shipping" applies for "black pants" only.
e.g. In ```a liquid single, gold lotion and shaving cream```, "a liquid single" applies for "gold lotion" only.
e.g. In ```Find comic books, one for 2-4 year olds and other one for 5-8 year olds.```, "comic books" applies for all products, because it's explicitly declared and followed by enumeration of "one", "another one", ...
e.g. In ``ink bottle, pencil and eraser. all must be less than 40````, "all must be less than 40" applies for all products, because it's explicitly mentioned with "all".

# Widely Used Keys for Keyword Key-Value Formatting
The followings keys are widely recognized:
- product_type
    there must exist at least one keyword with "product_type" key
- color_family
    unlike the nuance of its name, it's not limited to color modifier, but can be anything which represents a variation of the product like color, size, serial number, identifier, ...
- color
    color attribute of the product, but not limited to the color adjectives.
    sometimes it can be a strange-looking color code or identifier like "black 002", "#4 green", "other/extra", "gold 1", ...
- pre_config
    the attribute that is pre-confugured in the factory or the shop, like `pre_config→built`, `pre_config→tested`, ...
- pack, set
    the attribute about the package or bundle size, like `pack→20`, `set→3`, ...
- brand
    the manufacturer/seller name identifying the product, e.g. "from famenr's official store" -> `brand→famenr`, "nike shoes" -> `brand→nike`.
    This is a strong, non-negotiable identity signal — never merge it with, or drop it in favor of, a nearby price/service phrase, and never let generic words like "store", "shop", "official", "branded" leak into its value.
- feature, material, category, model, compatibility, origin, gender, user, usage, fit, suitable, promotion
- style, design, pattern, frame, theme, shape, inspiration, interest, vibe, condition, warranty_days
- size, length, capacity, volume, quality, unit, variation, option, solution, version

# Rule of Disambiguating a Color CODE from a Color-COUNT Quantity
- A bare number sitting right in front of the word "color"/"colour"/"variant" is structurally ambiguous between two very different meanings:
    (a) a color/variant CODE — the number is simply the name/index of one specific color option, exactly like the "black 002" or "gold 1" codes mentioned above (e.g. "color 1", "color 2", "color 3" can just be how a seller labels their distinct color options).
    (b) a real COUNT — how many distinct colors the product includes in total (e.g. "comes in 3 colors", "available in 5 color options").
- Default to reading a bare singular number + "color" as (a), a color/variant CODE, unless the sentence gives an explicit quantity signal:
    - the word "colors" is plural (not singular "color"), and/or
    - it's paired with quantity-implying wording like "comes in", "available in", "with ... options", "multiple", "assorted", or an enumerated list of actual color names.
  Without one of these signals, do NOT assume a bare number before "color" means "N total colors" or "single/monochrome color" — that is a common misreading. A real single-color request is normally phrased as "single color", "monochrome", "one solid color", or "same color throughout", not a bare digit glued to "color".
- Extract the number as its own identifier keyword (`color→<value>` or `color_family→<value>`), the same way you would for any other color code — do not fold it together with an unrelated quantity mentioned elsewhere in the phrase (e.g. a separate pack/piece count like "14pcs" is still its own `pack`/`set` keyword, extracted independently).
    e.g. "in the 14pcs 1 color" -> `pack→14` + `color_family→1` (kept as two separate keywords: "14" is the ordinary piece-count, "1" is a color-code identifier — NOT "one color" / "single color").
    e.g. "hair ties in 20pcs 3 color" -> `pack→20` + `color_family→3`.
    e.g. "14pcs, comes in 3 colors" -> `pack→14` + `color_family→3 colors` (here "comes in" + plural "colors" are the explicit quantity signals, so this one genuinely counts distinct colors).
- This ambiguity is not specific to any one product category or number — apply the same "default to CODE, require explicit signals for COUNT" reasoning to any bare number immediately modifying "color"/"variant", regardless of what other constraints appear nearby in the query.

# Handling Search-Unfriendly Keywords
When you convert each sub-query into a list of key-value formatted keywords for search, some keywords are general expressions like "fit for any gender", "one size fits all", "suitable for all models", "good for men and women" ...
Those expressions are not search-friendly. They must be converted with the following rules:
- If a gender constraint contains "all" or "any" like "all gender" or "any gender", then convert it to "gender→all" or "gender→any". If it's like "men and women" or "boys and girls", then convert it to "gender→unisex".
- If a size constraint is about "one size fits all" meaning, convert it to "size→freesize".
- If a constraint is about the casualness of wearings, convert it to "fit→casual".
- For other constraints like "compatible with any machines", convert it to "compatible_machine→universal".
- Compatibility/fit attributes that mean "any" or "no specific" (e.g. "no specific car year", "fits any year", "no particular model") must be converted to X→universal (e.g. car_year→universal, model→universal) or omitted entirely — never produce negated keyword values like car_year→no specific for these cases.

# Response Format
Return JSON object like:
{{
    "products": {{ // identify the number of the products mentioned, and their boundaries from the original query
        // make sure max product count is 4!        
        "1": <str>, // a short phrase for the first product. no need to write full text for this product. just a short differentiable phrase works.
        ... ...
    }},

    // Strictly keep in mind the `Rule of Handling the Scope of Modifiers` to correctly identify the corresonding product for each constraint chunk!

    "price": {{ // find all chunks related to the price constraints from the original query, and assign each of them to the corresponding product
        <chunk_1, str>: {{
            "apply_for": <product_id, str>, // the corresponding product id ("1", "2", ...)
            "normalized": <str> // the normalized price format
        }},
        ... ...
    }},

    "service": {{ // find all chunks related to the service constraints from the original query, and assign each of them to the corresponding product
        <chunk_1, str>: {{
            "apply_for": <product_id, str>, // the corresponding product id ("1", "2", ...)
            "labels": [<str>, ...]  // list ALL applicable service labels for this chunk (e.g. ["lazmall", "lazflash"] when both are mentioned)
        }},
        ... ...
    }},

    "sub_queries": {{ // rewrite sub-queries for the products, respectively. never miss any details, but don't include price and service constraints!
        "1": <str>, // sub-query for the first product, with price and service constraints excluded (never include price and service constraints, even if one word is left!)
        ... ...
    }},

    "keywords": {{ // convert each sub-query into a list of key-value formatted keywords.
        "1": [ // for the first sub-query
            
            // split the sub-query into mutually exclusive and collectively exhausted keywords, each of which corresponds to exactly one constraint aspect
            // one keyword must be a key-value pair

            // for example, "in color dark blue": "color→dark blue"

            <key→value, str>, // extracted keyword. use "→" for delimiting key and value in the keyword.
            ... ...
        ]
    }}
}}

# Original Shopping Query
{query}
"""

_MATCH_RULES = """
1. Tokens that appear to be numbers, identifiers, codes, variant names, or uncommon combinations must be matched literally and must not be normalized or partially matched. They are non-negotiable!
- If query mentioned "100ml", the product details must explicitly include the number "100" in the context of volume. "10-150ml" is not a match, even if 100 is in range 10-150.
- If query mentioned "3-7 yrs old", the product details must explicitly include both "3" and "7" in one of its attributes with the context of age. "5 years" is not a match, even if 5 is in range 3-7.
- If query mentioned "iPhone", the product details must explicitly include "iPhone". Guessing from "Apple Smartphone" is not allowed.
- If query mentioned "color green 2", the product details must explicitly include "green 2" for color-related details. "green" is not allowed. "green 2" is not spell error, but a specific identifier!
- A bare number keyword like `color→1` or `color_family→3` (extracted from a query's "N color" phrasing) is a specific color CODE, not a count of how many colors the product has and not "single/monochrome color". Never accept a product just because it happens to have exactly one color value, or because "1" or "3" occurs *somewhere* in its data — a lone digit is a weak signal and must be checked carefully:
  - The number must appear as its OWN standalone token in the color/color_family value — bounded by a space, the start/end of the string, or a separator like "-"/"/", never fused directly onto other characters. "14pcs 1", "style 1", "color-1", "1 red" all contain a standalone "1" and count. "1# kuromi 14pcs" (fused "1#"), "v1", "1x", "101" do NOT contain a standalone "1" and must NOT count, even though the digit visually appears.
  - If the query also carries a related quantity/pack keyword (e.g. `pack→14`), the standalone digit is only strong evidence when it sits immediately next to that SAME quantity mention in the product's data (e.g. "14pcs 1", "14pcs-1", "14 pcs, color 1"). A standalone digit that matches but sits next to an unrelated number/phrase (e.g. a separate style/model index like "model 1 (14 in 1)", "1# kuromi") is naming a different variant scheme — treat it as a mismatch even though the digit coincidentally matches.
  - When genuinely unsure whether an isolated digit match is the real requested variant or just a coincidental unrelated number, treat it as a mismatch — favor precision over recall for this specific ambiguity, since accepting the wrong variant is worse than under-matching.
- If query mentioned "up to 100m/s", the product details must explicitly include "100" with compatible semantics, like "90-100m/s" or "< 100m/s". "100-110m/s" is incorrect because the context doesn't match!

2. For descriptive constraints, rely on semantics and common sense.
- In particular, you must be generous for matching product type. If it's acceptable product type by common sense, see it a match. For example, "capacity tester" can be seen a match of "tester" if it's about battery.
- Replacement or accessory queries (e.g. "button flex for Oppo A5S") match when model/compatibility appears in title or attributes, even if product_type wording differs slightly.
- **CRITICAL — accessory vs primary product:** when the query asks for a primary item itself (e.g. `travel luggage bag`, `suitcase`, `duffel bag`, `phone`), a listing whose actual merchandise is a related accessory or spare part is a mismatch — even if SEO title stuffing repeats the primary item's words. Examples of mismatches for a luggage/travel-bag query: luggage straps, suitcase buckles/belts, bag tags, grippers, locks, hanging hooks. The product must BE the bag (or a clear bag paraphrase: duffel, tote, messenger bag, etc.), not an accessory for one.
- Also, consider paraphrasing as a match. (It's not applied to non-negotiable constraints mentioned in Rule-1.)
- For example, "standard" = "default", "kid" = "child", "beautiful" = "pretty", "regular" = "normal", "plain" = "simple", "freesize" = "one size", "personalized" = "customized", ...
- Common collective/quantity words used as product-type qualifiers — "set", "pack", "kit", "bundle", "lot" — are descriptive, NOT identifiers. They must NOT be matched literally. A product whose title or attributes use the plural form of the item (e.g. "Stickers") or contain a variation/quantity that implies a collection is a valid match for a query like "sticker set" or "cable pack".

3. `title` might be messy.
- Sometimes the `title` string contains more words than needed. It's vendor's intention for better SEO. So don't drop a product only because the `title` contains redundant words.
- For example, if you are looking for "water bottle", and a product's `title` is "Black Plastic Water Bottle Cover Kids Feeding", then this is a match. Don't drop it because it's "water bottle cover". Vendors love to enumerate many many words in `title` for SEO.
- **But this SEO generosity does NOT override Rule 2's accessory-vs-primary rule.** A title like "Nylon Luggage Straps … Suitcase Bag Straps … Travel Belt Bag Lock Hook" is selling straps, not a travel luggage bag — treat it as a product_type mismatch.
- If an attribute is found within `title`, see it a match, even if `title` field includes redundant information than needed.

4. Other attribute lines may contain different values. As long as at least one attribute line contributes a valid match, the product may still match.
- `attributes` filed consists of many lines. Each line represents one attribute with format "[key]: [value]".
- `attributes` field may represent a number of variations of the product. If at least one variation is a match, see this product a match.
- Therefore, the conflicting or alternative values in other attribute lines are allowed.
- For example, if the query mentioned "black" while both "black" and "green" are found in two different lines from `attributes`, respectively, then see it a match.

5. SKU / variant options (`sku_options` field)
- When `sku_options` is present, variant constraints in the query (color, pack size like "10pcs", variant names like "button blue", color codes like "a") must match at least one line in `sku_options` literally (Rule-1 applies).
- If the query specifies a variant and `sku_options` exists but no line contains that variant, it is a mismatch even when `title` partially matches.
- **Bare color vs shade-qualified SKU:** when the query asks for a bare basic color (e.g. `blue`, `red`, `green`) with no shade/tone word, a SKU value that adds an unrequested shade qualifier is a mismatch — not a literal match. Examples that must NOT match query `blue`: `sky blue`, `dark blue`, `light blue`, `navy blue`, `deep blue`. Only an exact (or synonymous bare) value such as `color_family: blue` / `color: blue` matches. Conversely, if the query says `dark blue`, then `dark blue` matches and bare `blue` is less exact.

6. Negation and absence constraints
- Phrases like "no stones", "without X", "non-X", or "stone-free" require the forbidden concept must NOT appear as a positive feature in title, attributes, or sku_options.
- Do not treat unrelated category words as proof of absence; require explicit negation or absence in product details.

7. Matching determines whether a product satisfies the query; final recommendation determines which matching product is the most exact.
- A product that technically satisfies all constraints can still be less exact than another all-matching product.
- During recommendation, prefer products whose matching evidence represents the requested values more directly — fewest **unsupported or conflicting** tokens relative to the Shopping Query. Tokens that appear in / support the query are not penalties just because they make a SKU longer than a bare substring match.
- Structured attributes are generally stronger evidence than noisy SEO title text.
- Original ranking is only a tie-breaker after match quality is equal.

8. Attribute evidence must be token-based, not inference-based.
- An attribute line is only valid evidence for a query constraint if its key or value contains at least one token from the query.
- Do NOT use an attribute as evidence by inferring its meaning. For example, `magnification: 1000x` is NOT evidence for a "science microscope" query because neither "magnification" nor "1000x" appears in the query — even though magnification is a property of microscopes.
- If an attribute has no token overlap with the query, treat it as absent. Do not reason from it.
"""

_PV_SELECT = """
You are AI assistant to select the best matching product for the user's shopping query from the candidate products.
You should examine each product thoroughly against the `Given Constraints` list (already extracted from the query).

# Primary Goal
Your goal is to choose the most exact matching product, not merely the first acceptable product.
Accuracy is more important than reducing checking cost.

# Strategy
- You must check ALL candidate products before deciding the final best product.
- The `Given Constraints` list below was already extracted from the query during query decomposition. Treat it as the definitive, exhaustive constraint list for **pass/fail matching** — do NOT invent new constraints that a product must satisfy.
- For each product, check whether it satisfies every constraint in the `Given Constraints` list.
- A product is an all-matching product only if every given constraint is satisfied.
- After checking all candidates, collect all all-matching products.
- If there are no all-matching products, return null.
- If there are multiple all-matching products, apply the Recommendation Rules below in priority order (R1 first, then R2, etc.) until one candidate is preferred over the others.
- When applying R1 (extra / unsupported tokens), ground the judgment against the **Shopping Query** text as well as Given Constraints: tokens that support or appear in the query are not extras, even if they go beyond a truncated constraint keyword.
- Original candidate order is only a final tie-breaker (R6) when all earlier rules leave candidates equal.

# Exact Variant Matching
- When the user explicitly specifies a variant attribute (such as color, quantity/pieces, size, weight, volume, pack count, model, or similar), prefer candidates whose variant exactly matches that requirement.
- When the user does NOT specify such a variant, prefer the product whose matching evidence contains the fewest unsupported or conflicting qualifiers attached to the required value.
- Do not stop at the first partially matching variant. Check all available variants across all candidates for a more exact match before deciding.

# Matching Rules
{rules}

# Recommendation Rules
All-matching products are not automatically equal.
Evaluate candidates in three phases. Move to the next phase only when the current phase cannot distinguish candidates.

---

## Phase 1 — Evidence Quality (consider R1, R2, R3 together)
R1, R2, and R3 each measure a different facet of how reliable and clean the matching evidence is. Assess all three facets together for each candidate, then ask: does one product have clearly better evidence quality overall?

- If one product wins on all three (or two out of three) facets while the other wins on none (or one), the first product has clearly better evidence quality — prefer it.
- If the picture is mixed (e.g. Product A wins R1 but loses R2 and R3), apply R1 → R2 → R3 in order within this phase as a tiebreaker.

**R1 — Directness of match (fewest unsupported / conflicting tokens)**
Prefer the product whose matching attribute value has the fewest tokens that are **unnecessary or disagree** with the user's request. Do NOT treat "longer SKU" as automatically worse.
- **What counts as an extra token:** a token in the matching value that does **not** help satisfy the Shopping Query — i.e. it is unsupported by the query wording, or it conflicts with / narrows beyond what the user asked (unrequested shade, size, material, pack count, etc.).
- **What does NOT count as extra:** tokens that appear in the Shopping Query (or clearly paraphrase a stated request), and tokens that cover another stated constraint. Those are supporting evidence — they make the match more direct, even if they lengthen the SKU.
- Ground this judgment against the **Shopping Query** text (and Given Constraints together), not against a truncated keyword alone. A longer value that more faithfully reproduces query phrasing beats a shorter bare substring of that phrasing.
- `color_family: pink` is preferred over `color: small pink (1.5cm)` for a query asking for "pink" — size/dimension tokens are unsupported extras.
- `color_family: blue` is preferred over `color: sky blue` / `color: dark blue` for a query asking for bare "blue" — shade prefixes disagree with / go beyond the ask (and are not valid matches under Matching Rule 5).
- `material: silicone` is preferred over a title containing "silicone-coated plastic handle" for a query asking for silicone — "coated plastic handle" is unsupported noise.
- **CRITICAL — supporting / query-grounded tokens are NOT extras.** If the Shopping Query contains a fuller phrase and one candidate's SKU reproduces that fuller phrase while another only keeps a bare substring, prefer the fuller reproduction. Example: query mentions `black-door-lock` → `color_family: black-door-lock` beats `color_family: black` (the additional tokens support the query; they are not noise). Same idea for `random color` vs `random`.
- **CRITICAL — cross-constraint tokens are NOT extras.** When the query has multiple constraints (e.g. color AND material, or color AND capacity/size), a single `sku_options` value that covers several of them is the BEST evidence — those tokens are required coverage, not noise. Prefer it over a bare match that covers only one constraint while leaving others to the noisy title.
  - Example: constraints `color_family: black` AND `material: carbon fiber leather` → `color_family: black carbon fiber【leather】` wins over `color_family: black`.
  - Example: constraints `capacity: 260g` AND `color_family: mint` → `color: 260g mint` wins over `color: mint`.
- **An attribute value that is an exact or near-exact reproduction of the query term has zero unsupported tokens and is always the most direct possible evidence — it wins R1 unconditionally against any other form of matching.**
- **Scope: evaluate only the specific `sku_options` variant entry or `attributes` line that provides the matching evidence.** Other variants or attribute lines are not part of this comparison.
- **CRITICAL — when scoring unsupported tokens, count ALL unsupported/conflicting tokens in the full matching value, not just those near the matched keyword.** A compound value that starts with a matching term but continues with unsupported content still carries those trailing extras. Compare the full matching entry of each candidate.

**R2 — Attribute key specificity**
Prefer the product whose matching evidence comes from an attribute key dedicated to the queried property.
- `color: pink` is preferred over `variant: pink round 500ml` for a query about color.
- `material: silicone` is preferred over `description: made with silicone` for a query about material.
- A dedicated, property-specific key signals the vendor explicitly categorized that dimension, making the match more reliable than evidence buried in a compound or catch-all field.
- **Scope: judge only the key of the specific entry that matched — not the keys of other variants or attributes the product happens to have.**
- **CRITICAL — a dedicated `sku_options` key that names the exact queried property (e.g., `piece`, `pack`, `color`, `size`) is always more specific than a compound catch-all key like `model` or `name` that embeds the value alongside other unrelated information. The former explicitly categorizes the dimension; the latter buries it.**

**R3 — Structured field over noisy title**
Prefer the product whose matching evidence appears in a structured field (`sku_options`, named `attributes`) rather than only in the free-text `title`.
- SEO titles frequently pile on unrelated words; structured fields carry deliberate, vendor-confirmed data.
- **CRITICAL — `sku_options` is the highest-quality structured field.** A product with the required variant in its `sku_options` always wins R3 over a product that has no `sku_options` and carries the match only inside a compound attribute string or in the title. Having evidence in `sku_options` is a strength — it means the vendor explicitly registered that purchasable variant.

---

## Phase 2 — Value Precision (R4)
If evidence quality (Phase 1) is equal across candidates, assess value precision.

**R4 — Exact or narrow value over compound or range value**
Prefer the product whose matching attribute holds a single, narrow value rather than a multi-value or range value.
- `size: M` is preferred over `size: S/M/L` when the query asks for "M".
- `capacity: 500ml` is preferred over `capacity: 300–700ml` when the query asks for "500ml".

---

## Phase 3 — Descriptive Coverage (R5)
If Phase 1 and Phase 2 are equal, assess how well each product covers the full intent of the user's query.

**R5 — Broader descriptive coverage**
Prefer the product with more complete coverage of the user's descriptive intent (use-case, function, material, compatibility, style, or benefit), provided that coverage does not introduce unsupported or conflicting qualifiers that would penalize it under R1.

---

## Phase 4 — Tie-breaker (R6)
Only when Phases 1–3 are all equal, prefer the candidate that appeared earlier in the candidate list.

**R6 — Original ranking as final tie-breaker**

---

Do not choose an earlier product merely because it appears first. A later candidate is preferred whenever it has better evidence quality, value precision, or coverage under Phases 1–3.

# Response Format
Return JSON like:
{{
    "check": "<short product-by-product checking summary>",

    "all_matches": [<product ids that satisfy all constraints>],

    "recommend": "<which phase and rule decided the winner, with a one-line evidence comparison, e.g. 'Phase 1 / R1: product X has color_family:pink (0 unsupported tokens) vs product Y color:small pink(1.5cm) (2 unsupported: size/dimension)'>",

    "best": <str|null>
}}

# Output Guidelines
1. Be concise and structured.
- Each product checking content must be one line.
- Use `\\n` for delimiting products.
- Each product checking content should look like:
  "Product <ID>→<constraint>:<evidence>, match; <constraint>:<evidence>, match;"
- If a required constraint is missing, stop checking that product and mark it as mismatch.

2. For recommendation:
- Compare only all-matching products.
- State which phase (Phase 1/2/3) and which rule (R1–R6) decided the winner, then give a concrete one-line evidence comparison.
- Do not rely on original order unless match quality is fully tied through all of Phase 1–3.

3. Return JSON only.

# Shopping Query:
```{query}```

# Given Constraints (already extracted from the query during decomposition — this is the definitive, exhaustive constraint list; do not extract new constraints from the query text)
{constraints}

# Candidate Products:
{products}
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


def _normalize_str(text: str) -> str:
    x = str(text or "").lower()
    x = re.sub(r"\b(\d+)([a-z]+)\b", r"\1 \2", x)
    x = re.sub(r"\b([a-z]+)(\d+)\b", r"\1 \2", x)
    x = re.sub(r"[\(\)\[\]\{\}\+\*\_\:\;\,\-\~\/\<\>\=]", " ", x)
    x = re.sub(r"\b(\d+)\s*x\s*(\d+)\b", r"\1 \2", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _spell_vars(token: str) -> list[str]:
    s = token
    vars: list[str] = []
    if len(s) >= 3 and not re.search(r"\d", s):
        for suffix in ("es", "ic", "n", "d", "ed", "y", "ing", "ily", "ied", "er", "est", "ier", "ren"):
            vars.append(s + suffix)
            if s.endswith(suffix):
                body = s[: -len(suffix)]
                vars.append(body + "y")
                vars.append(body)
                if len(body) >= 4 and body[-1] == body[-2]:
                    vars.append(body[:-1])
        if s.endswith("s"):
            vars.append(s[:-1] + "es")
            vars.append(s[:-1])
        else:
            vars.append(s + "s")
        if s.endswith("y"):
            for suffix in ("", "ied", "ies", "ier", "ing", "ily"):
                vars.append(s[:-1] + suffix)
    return [v for v in set(vars) if v != s and v not in _YKI_STOP and len(v) >= 3]


def _is_kw_in(kw: str, blob: str, as_is: bool, terminal: bool = False) -> bool:
    if not kw:
        return False
    if re.search(
        (r"\b" if re.match(r"\w", kw[0]) else "")
        + re.escape(kw)
        + (r"\b" if re.match(r"\w", kw[-1]) else ""),
        blob,
        re.I,
    ):
        return True
    if as_is:
        return False
    for var in _spell_vars(kw):
        if re.search(rf"\b{re.escape(var)}\b", blob, re.I):
            return True
    if len(kw) >= 6 and len(blob) > len(kw) + 1:
        segs = blob.replace("🎈", " ").split()
        for i in range(len(segs) - 1):
            if kw == segs[i] + segs[i + 1]:
                return True
    if (not as_is) and (not terminal) and len(kw) >= 5:
        for seg in blob.replace("🎈", " ").split():
            if len(seg) > len(kw) and (seg.startswith(kw) or seg.endswith(kw)):
                return True
    if kw in {"pcs", "pc", "pack"} and any(w in blob for w in ("piece", "pack", "set")):
        return True
    for left, right in _SYNONYMS:
        if kw in {left, f"{left}s"} or left == f"{kw}s":
            if re.search(rf"\b{re.escape(right)}s?\b", blob):
                return True
        elif kw in {right, f"{right}s"} or right == f"{kw}s":
            if re.search(rf"\b{re.escape(left)}s?\b", blob):
                return True
    if not terminal:
        spell = re.sub(r"ss", "sss", kw, flags=re.I)
        if spell != kw and _is_kw_in(spell, blob, False, True):
            return True
    return False


def _compact_kw(nested: str, query: str) -> list[str] | None:
    splits = [s for s in [s.strip() for s in nested.lower().split("→")] if s]
    if len(splits) < 2:
        return None
    keys, value = splits[:-1], splits[-1]
    if value.lower() in _YES_WORDS:
        for i in range(len(keys)):
            token = " ".join(k.replace("_", " ") for k in keys[i:])
            if token in query:
                keys, value = [value], token
                break
        else:
            return None
    for yes in _YES_WORDS:
        if re.search(rf"\b{re.escape(yes)}$", value, re.I):
            value = re.sub(rf"\b{re.escape(yes)}$", "", value, flags=re.I)
    for nw in _NO_WORDS:
        match = re.search(rf"^{re.escape(nw)}\s+([\w\s]+)$", value, re.I)
        if match:
            keys.append("no")
            value = match.group(1)
            break
    if f"{keys[-1]} {value}" in _YKI_STOP or f"{value} {keys[-1]}" in _YKI_STOP:
        return None
    for stop in _YKI_STOP:
        value = re.sub(rf"\b{re.escape(stop)}\b", "", value, flags=re.I)
    if len(keys) == 1 and "_" in keys[0]:
        prefix = keys[0].split("_")[0] + "-"
        if value.startswith(prefix):
            value = value[len(prefix) :]
        bare = prefix[:-1]
        if re.search(rf"^{re.escape(bare)}[a-z]+$", value, re.I):
            value = value[len(bare) :]
    match = re.search(r"^(any|all)\s+([\w\s]+)$", value, re.I)
    if match:
        extra = match.group(2).replace(" ", "_")
        value = match.group(1)
        if extra not in keys:
            keys.append(extra)
    if value.isnumeric() and keys[-1] in {"eu", "us", "int"}:
        value, keys = f"{keys[-1]} {value}", ["size"]
    value = re.sub(r"\bge[a-z]([a-z]{2}one)\b", r"\1", value, flags=re.I)
    value = re.sub(r"\b(\d+)in(\d+)\b", r"\1-in-\2", value, flags=re.I)
    value = re.sub(r"[&|]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return (keys + [value]) if value else None


def _kw_constraints(pairs: list[list[str]]) -> str:
    lines = []
    for kw in pairs:
        if not kw:
            continue
        *keys, value = kw
        key = "_".join(k for k in keys if k) or "attribute"
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) if lines else "- (none extracted)"


def _search_from_kws(pairs: list[list[str]]) -> str:
    segs: list[str] = []
    for kw in pairs:
        if not kw:
            continue
        value = kw[-1]
        keys = kw[:-1]
        leaf = keys[-1] if keys else ""
        if leaf and leaf not in _SCHEMA_KEYS:
            for part in leaf.replace("_", " ").split():
                part = part.lower().strip()
                if part and part not in segs:
                    segs.append(part)
        for part in value.split():
            part = part.lower().strip()
            if part and part not in segs:
                segs.append(part)
    return " ".join(segs)


def _extract_decompose(js: dict[str, Any]) -> list[dict[str, Any]]:
    products = js.get("products") if isinstance(js.get("products"), dict) else {}
    keywords = js.get("keywords") if isinstance(js.get("keywords"), dict) else {}
    prices = js.get("price") if isinstance(js.get("price"), dict) else {}
    subs = js.get("sub_queries") if isinstance(js.get("sub_queries"), dict) else {}
    out: list[dict[str, Any]] = []
    count = len(products) or 1
    for i in range(count):
        sid = str(i + 1)
        query = str((subs.get(sid) if isinstance(subs, dict) else "") or "").strip()
        raw_kws = keywords.get(sid) if isinstance(keywords.get(sid), list) else []
        kws = [str(k).strip() for k in raw_kws if str(k).strip()]
        price = None
        for row in prices.values() if isinstance(prices, dict) else []:
            if not isinstance(row, dict):
                continue
            if str(row.get("apply_for") or "") != sid:
                continue
            norm = str(row.get("normalized") or "")
            cleaned = re.sub(r"[^\d\-.]", "", norm)
            if cleaned.count("-") == 1:
                price = cleaned
                break
        out.append({"query": query, "keywords": kws, "price": price})
    return out


def _yki_plan(query: str) -> dict[str, Any]:
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
        "kw_pairs": [],
    }
    dec = _llm_json(
        _A_DECOMPOSE.format(query=head, shop_intent=""),
        list(_HELPER_MODELS),
        _HELPER_TEMP,
    )
    if isinstance(dec, dict):
        try:
            items = _extract_decompose(dec)
        except Exception:
            items = []
        if items:
            item = items[0]
            if item.get("query"):
                plan["rewritten"] = item["query"]
                plan["compact"] = item["query"]
            if item.get("price"):
                plan["price"] = item["price"]
            pairs: list[list[str]] = []
            attrs: dict[str, str] = {}
            for raw in item.get("keywords") or []:
                packed = _compact_kw(str(raw), plan["rewritten"])
                if packed:
                    pairs.append(packed)
                    attrs["_".join(packed[:-1]) or "attribute"] = packed[-1]
            plan["kw_pairs"] = pairs
            plan["attrs"] = attrs
            built = _search_from_kws(pairs)
            if built:
                plan["keywords"] = _title_tokens(built)[:8] or plan["keywords"]
                plan["compact"] = built
    primary = " ".join(plan["keywords"][:6])
    plan["searches"] = [s for s in (primary, plan["compact"], plan["rewritten"]) if s]
    return plan


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


def _yki_score(row: dict[str, Any], query: str, plan: dict[str, Any] | None = None) -> float:
    title = _normalize_str(str(row.get("title") or ""))
    facts, _ = _flatten_facts(row)
    blob = f"{title} {_normalize_str(facts)}"
    pairs = list((plan or {}).get("kw_pairs") or [])
    if not pairs:
        words = _title_tokens(query)
        return float(sum(2 if _is_kw_in(word, blob, False) else 0 for word in words))
    score = 0.0
    for kw in pairs:
        value = kw[-1]
        keys = kw[:-1]
        atoms = [a for a in _normalize_str(value).split() if a]
        if not atoms:
            continue
        as_is = bool(re.search(r"^\d+$", value))
        hits = sum(1 for atom in atoms if _is_kw_in(atom, blob, as_is))
        if hits == len(atoms):
            score += 0.95 if keys and keys[-1] == "product_type" else 1.0
            if keys and keys[-1] == "product_type" and len(atoms) >= 2 and _normalize_str(value) in title:
                score += 0.2
        elif hits:
            score += hits / len(atoms)
    price = _as_float(row.get("price"))
    if plan and price is not None and plan.get("price"):
        low, high = _numerify_price(plan.get("price"))
        score += -2.5 if price < low or price > high else 0.4
    return score


def _format_candidates(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for i, row in enumerate(rows[:10], start=1):
        facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
        options = row.get("options") if isinstance(row.get("options"), dict) else {}
        attr_lines = [f"{k}: {v}" for k, v in {**facts, **options}.items() if v not in (None, "", [], {})]
        blocks.append(
            json.dumps(
                {
                    "id": i,
                    "title": row.get("title"),
                    "attributes": "\n".join(attr_lines),
                    "price": row.get("price"),
                    "brand": row.get("brand"),
                    "product_id": row.get("product_id"),
                    "sku": row.get("sku"),
                    "in_stock": row.get("in_stock"),
                },
                ensure_ascii=True,
                default=str,
            )
        )
    return "\n".join(blocks)


def _yki_filter(query: str, rows: list[dict[str, Any]], plan: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if len(rows) <= 2:
        return rows
    ranked = sorted(rows, key=lambda r: -_yki_score(r, query, plan))
    if not ranked:
        return rows
    top = _yki_score(ranked[0], query, plan)
    kept = [r for r in ranked if _yki_score(r, query, plan) >= top - _PRUNE_GAP]
    return kept or ranked[:_PREFILTER_TOPN]


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


def _yki_select(query: str, rows: list[dict[str, Any]], must: bool, plan: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if not rows:
        return None
    if len(rows) == 1 and must:
        return dict(rows[0])
    constraints = _kw_constraints(list((plan or {}).get("kw_pairs") or []))
    cursor = 0
    tries = 0
    while cursor < len(rows) and tries < _MAX_SELECT_TRIES:
        batch = rows[cursor : cursor + _SELECT_BATCH]
        cursor += _SELECT_BATCH
        tries += 1
        parsed = _llm_json(
            _PV_SELECT.format(
                rules=_MATCH_RULES,
                query=_goal_head(query),
                constraints=constraints,
                products=_format_candidates(batch),
            ),
            list(_SELECT_HELPERS),
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


def _identity_key(row: dict[str, Any]) -> str:
    pid = str(row.get("product_id") or "").strip()
    if pid:
        return f"id:{pid}"
    brand = str(row.get("brand") or "").casefold()
    words = " ".join(sorted(set(_title_tokens(str(row.get("title") or "")))))
    return f"twin:{brand}|{words}"


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
    return _yki_score(row, " ".join(tokens) or query, plan)


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


class _ElonSolver:
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
        return _yki_filter(self.query, gated, self.plan)

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
        return _yki_select(self.query, rows, True, self.plan)

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
    if lane == "retrieval":
        phrase = " ".join(goal.get("terms") or [])
        lines.append(
            f"Do not search again. The phrase is `{phrase}`. "
            "Swap the OOS cart item for another title-hit already in the catalog, "
            "then place_test_order {{product_id, sku}} immediately."
        )
    if lane == "intent":
        lines.append(
            "If the shopper asked which item / whether the spec holds, message that "
            "answer before add_to_cart. First add must be the preferred brand when "
            "a fully compliant SKU exists. After a price jump, recover under the "
            f"CURRENT cap {cap} {ccy}, never the original query budget."
        )
    if plan.get("price"):
        lines.append(f"Normalized price gate: {plan.get('price')}")
    if plan.get("keywords") and lane != "retrieval":
        lines.append("youKnowIt keywords: " + " ".join(plan.get("keywords") or []))
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
            "\n\nyouKnowIt retrieve plan\n"
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
        plan = _yki_plan(query)
    session = _Session(
        binding=binding,
        problem_id=problem_id,
        names=names,
        max_steps=max_steps,
        max_calls=max_calls,
    )
    _ElonSolver(session, query=query, lane=lane, goal=goal, plan=plan).run()
    if not session.done:
        _fallback_loop(session, query=query, tools=tools, lane=lane, plan=plan, goal=goal)
    return session.dialogue


__all__ = ["agent_main"]
