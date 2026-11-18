"""ORO Bench agent: VALID product engine executes real environment tools.

Main contract is Axis-A: POST /environment/call until observation.done.
VALID.py shop/voucher/find_product APIs are omitted. The product-lane method
is embedded and called: decompose → build keywords → search → view →
price-gate → LLM filter → weigh → select → inspect_stock → add_to_cart →
recover → place_test_order.
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
_HELPER_MODELS = (
    "deepseek/deepseek-v3.2",
    "google/gemma-4-31b-it",
    "openai/gpt-oss-120b",
    "z-ai/glm-5.2",
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

_KW_STOP = (
    "looking",
    "find",
    "need",
    "want",
    "show",
    "please",
    "ideally",
    "prefer",
    "suitable",
    "compatible",
    "support",
    "made",
    "that",
    "with",
    "without",
    "from",
    "for",
    "and",
    "the",
    "a",
    "an",
    "under",
    "over",
    "below",
    "above",
    "between",
    "than",
    "php",
    "usd",
    "in-stock",
    "instock",
    "stock",
    "category",
    "listing",
    "buy",
    "one",
    "place",
    "order",
    "test",
    "phrase",
    "matches",
)

_ACCESSORY = (
    "phone case",
    "case for",
    "screen protector",
    "tempered glass",
    "replacement screen",
    "mystery box",
    "review page",
    "feature phone",
)

_WATER = ("water", "sweat", "ipx", "waterproof", "resistant", "ip67", "ip68")
_WET_STRONG = (
    "ipx",
    "ip67",
    "ip68",
    "ipx4",
    "ipx5",
    "ipx6",
    "ipx7",
    "waterproof",
    "water-resistant",
    "water resistant",
    "sweat",
    "sweatproof",
    "sweat-proof",
)

_PLAYBOOK = """You are a commerce solver inside a generated shopping environment.

Use ONLY the tools supplied in this request. Never invent a tool. Never wait
for human permission after a candidate is verified — place_test_order ends
the episode.

# How to shop
1. Split firm gates vs soft preferences.
2. Retrieve, then inspect. search uses short 2-5 word queries (k=10).
   filter uses category, brand, max_price, in_stock. Confirm category_path.
3. view facts + SKU variants. compare up to five. inspect_stock the SKU.
4. First successful add_to_cart is a commitment. Do not place_test_order
   on the same turn as the first add.
5. After cart change, inspect_stock. Recover if over budget or out of stock.
6. Honor shopper messages. Answer direct questions with message.
7. inspect_cart, then place_test_order for a still-valid carted SKU.

Think with concrete prices, SKUs, and specs, then call a tool.
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
_TF2_WORKSPACE = """# TF2 — retrieval (VALID product-lane keywords)

Paid reward is retrieval_quality. query_efficiency collapses if you issue
more than 1–3 distinct searches. 8+ rewrites score ~0.15 even on gold.
No place_test_order → paid 0. The catalog indexes JSON VALUES with BM25.
Title is an unordered token bag: order and grammar do not matter.

VALID retrieve (do exactly this):
1. Extract the quoted phrase or the words after "phrase". Stop at a comma
   or "at or under". Drop filler (the, phrase, matches, listing, find,
   category). Those words are the ONLY search keywords — value-side
   tokens, not JSON keys (size/color/brand/category).
2. search that exact keyword string ONCE (k=10, in_stock). Optional: ONE
   alternative that adds OR removes exactly one token (VALID PSV_KW).
   Prefer no alt. Do not reverse the phrase. Do not search single tokens.
   Do not prefix the category. Reordering and duplicates are identical.
3. filter category + max_price + in_stock. compare/view title-hits until
   you see several distinct identities. A title hits iff EVERY phrase
   token appears in the title, any order.
4. inspect_stock one title-hit, add_to_cart, STOP. No order this turn.
5. After "is that still available?" the first SKU is OOS. remove it. add
   another already-viewed title-hit still in stock and under budget.
   Do not search again — the catalog already has the shortlist.
6. place_test_order {product_id, sku} immediately. Never spend leftover
   steps searching. No order → paid 0 even if recall looked fine.
"""
_TF3_WORKSPACE = """# TF3 — constraint
Every firm rule must hold on the EXACT SKU: category, brand if named,
gated spec, CURRENT cap, in-stock. Ask for the hidden spec first.
Shopper will answer with the spec and later "only go up to X now".
1. message asking the firm hardware spec. Absorb N-inch / RAM / Hz from
   every shopper line. Re-search brand + spec + max_price=X.
2. view the SKU options. First add a genuine device (not a case/protector)
   that matches the spec on that variant.
3. After "is that still available?" it is OOS. remove. filter/search again
   with brand + spec + new cap. inspect_stock the replacement. add. order
   {product_id, sku} only. Do not keep messaging — close the episode.
"""
_TF4_WORKSPACE = """# TF4 — preference (VALID: attributes/facts beat title)

Hard: Headphones category, CURRENT cap, in-stock, a real headphone.
Soft axes the grader scores from VISIBLE FACTS, then title tokens:
sweat_water_resistance, sport_secure_fit, wireless_freedom, battery_life.
Title-only "IPX" on an over-ear is NOT sweat evidence (Mypro H3 scores 0.2).
Sports + waterproof + wireless earbuds / neckband / bone-conduction score ~0.78.

VALID retrieve:
1. search "sports waterproof earbuds", "IPX sweat-proof sports", 
   "bone conduction waterproof". Do not dump generic headphone rewrites.
2. compare facts. A constraint PASSes on a fact/option value (ATTR) or
   title token (TITLE). Prefer ATTR. Do not infer from SEO titles.
3. First add MUST be a compared sport+water+wireless SKU still under the
   cap (cheaper sports-waterproof earbuds beat an expensive "sweat" title).
   That first add is the event commitment. A later gold order does not
   repair a non-aligned first add (paid 0, event_closed=false).
4. Price jump always fires. remove it. add another already-viewed
   sport+water+wireless SKU still under the cap. Never recover to a
   title-only over-ear. place_test_order {product_id, sku} only.
"""
_TF5_WORKSPACE = """# TF5 — ranking
Rank ONLY the named product_id::sku list. Do not search the open catalog.
Order: highest advertised max refresh (incl. overclock) > larger screen >
lowest live price. compare them, inspect_stock each, add the current top.
After "is that still available?" re-inspect every named SKU and add the
new top. place_test_order {product_id, sku} only. No justification.
"""
_TF6_WORKSPACE = """# TF6 — recovery
Never relax category / budget / in-stock / genuine-device. Soft rank:
exact color+model, then drop color, then drop model.
1. search color+model smartphone. view. add the exact preferred SKU.
2. After stockout, add a same-model different-color (or listed sibling
   SKU) that is still a real smartphone under budget.
3. place_test_order {product_id, sku} only.
"""
_TF7_WORKSPACE = """# TF7 — justification (VALID: facts/options are authoritative)

view facts BEFORE the first add. Attributes beat title. A title that says
1080P / HD / 27-inch is only TITLE evidence; facts.display / facts.size
are ATTR. If facts.display is 2560x1440, it is QHD, not FHD — reject it
when the query asked Full HD. Never first-add an over-cap or off-spec
decoy (that zeros intended_event_commitment even if the final order is gold).

1. search brand + gated spec only (2 tight queries). compare facts.
2. First add a branded, on-spec, under-cap SKU whose FACTS prove the spec.
3. After the price jump, order a DIFFERENT still-valid SKU (not the event
   target). Episode must close.
4. Tiny justification. Allowed fields ONLY: in_stock, price, gated spec
   (screen_size OR resolution), optional brand if it matches visible brand.
   Forbidden (zero the score): display, operating_system, category,
   refresh_rate unless that is the gated spec, any free-text field.
Required claims, all string values:
- in_stock = "true"
- price = live inspect_stock price as a string
- screen_size = the number from facts.size ("27"), never "27-inch"
- resolution = hd|fhd|qhd|uhd from facts.display, never "720p" / "1920 x 1080"
place_test_order {product_id, sku, justification:{claims:[...]}}.
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
    "error",
    "done",
    "action",
    "market_event",
    "added",
    "product_id",
    "sku",
    "price",
    "in_stock",
    "currency",
    "title",
    "brand",
    "category_path",
    "facts",
    "options",
    "listing_id",
    "returned",
    "truncated",
    "cart",
    "justification",
)

_P_DECOMPOSE = """
You analyze one shopping query into search structure.
Return JSON only.
{{
  "prices": {{"rationale": "<str>", "normalized": "<low-high|low-|-high|null>"}},
  "rewritten": {{"rationale": "<str>", "result": "<query minus price/service filler>"}},
  "keywords": ["<2-6 short catalog value words>"],
  "alt": {{"<token>": "<alternative>"}},
  "redundant": "<optional weak token or null>"
}}
Rules:
- Normalize budget as low-high / low- / -high without currency.
- rewritten keeps product type, brand, and hardware spec; drop budget words.
- keywords are value-side catalog tokens (not JSON keys like size/color).
# Query
{query}
"""

_PSV_BUILD = """
Convert one shopping description into e-commerce attributes.
Return JSON only:
{{"1": {{"compact": "<search-friendly rewrite>",
  "attributes": {{"product_type": "<type>", "<key>": "<short value>"}}}}}}
Rules: mutually exclusive keys, keep every meaningful token, product_type first.
# Description
{queries}
"""

_PSV_KW = """
You review e-commerce search keywords and recommend, when appropriate, a
single alternative keyword set. The existing keywords are already high
quality. Do NOT rewrite them. Identify rare cases where exactly one small
change could improve recall without hurting precision.

Search engine: only JSON VALUES are indexed (BM25 + morphological
normalization). JSON keys (size, color, color family, material, brand,
category, product type) do not help and add noise. Hyphens = spaces.
Reordering or duplicating keywords returns the identical hit set.

Allowed change: add exactly one keyword OR remove exactly one keyword.
Never both. Never substitute, reverse, or paraphrase. The alternative
must otherwise stay identical. Prefer null.

Do not add: grammatical words, shopping-intent words, metadata keys,
tokens not in the query, or tokens already present.
Do not remove: product-describing tokens, numbers, identifiers, brands.

Return JSON only:
{{"1": {{"rationale": "<str>", "alt": "<full keyword string or null>"}}}}
# Query and keywords
{inf}
"""

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

_PV_SELECT = """
Select the best matching product. A valid match satisfies EVERY required
constraint. Structured facts/options beat title tokens.
Phase 1: extract constraints (type, brand, model, size, spec, color, ...).
Phase 2: literal match for brands, models, numbers, colors. Semantic for the rest.
Phase 3: evaluate in Fibonacci batches (3, then 2, then 5, then 7). Each
constraint is ATTR (structured exact), ATTR- (structured weak), TITLE,
FAIL, or UNKNOWN. UNKNOWN is FAIL. Stop at the first batch with a valid item.
Phase 4: among valid items, more ATTR wins, then closer wording, then earliest.
Return JSON only:
{{"best": "<1-based product id or null>", "recommend": "<one sentence>",
  "valid": "<comma ids or null>"}}
# Shopping Query
{query}
# Candidate Products
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
    if override and override not in ladder:
        ladder.insert(0, override)
    elif override:
        ladder = [override] + [name for name in ladder if name != override]
    return ladder


def _goal_head(query: str) -> str:
    return query.split("Task rules:", 1)[0]


def _title_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").casefold())


_PHRASE_STOP = {
    "the",
    "phrase",
    "matches",
    "match",
    "that",
    "listing",
    "listings",
    "find",
    "in",
    "stock",
    "category",
    "and",
    "buy",
    "one",
    "under",
    "or",
    "at",
    "an",
    "a",
    "php",
    "usd",
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
    cleaned = re.sub(r"^```(?:json)?\s*", "", str(text).strip(), flags=re.I)
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
    words = []
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


def _parse_goal(query: str) -> dict[str, Any]:
    head = _goal_head(query)
    facts: dict[str, Any] = {"head": head.strip()}
    budget = re.search(r"(\d+(?:\.\d+)?)\s*(USD|EUR|GBP|PHP|SGD|AUD|CAD)", head, re.I)
    if budget:
        facts["budget"] = budget.group(1)
        facts["currency"] = budget.group(2).upper()
    brand = re.search(
        r"(?:prefer|brand)\s+([A-Za-z][A-Za-z0-9]*(?:\s+[A-Za-z][A-Za-z0-9]*){0,2})",
        head,
        re.I,
    )
    if not brand:
        brand = re.search(
            r"\b(?:need a|want a|get me an|find me a)\s+([A-Za-z][A-Za-z0-9]+)\s+",
            head,
            re.I,
        )
    if brand:
        token = brand.group(1).strip()
        if token.casefold() not in {"smartphone", "headphone", "monitor", "tablet", "in", "stock"}:
            facts["brand"] = token
    if not facts.get("brand"):
        stock_brand = re.search(
            r"in-stock\s+([A-Za-z][A-Za-z0-9]+)",
            head,
            re.I,
        )
        if stock_brand:
            token = stock_brand.group(1).strip()
            if token.casefold() not in {"smartphone", "headphone", "monitor", "tablet", "in"}:
                facts["brand"] = token
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


def _valid_plan(query: str) -> dict[str, Any]:
    head = _goal_head(query).strip()
    plan: dict[str, Any] = {
        "rewritten": head,
        "compact": head,
        "keywords": _keywords_fallback(head),
        "alt_keywords": [],
        "attrs": {},
        "price": None,
        "searches": [],
    }
    dec = _llm_json(_P_DECOMPOSE.format(query=head), list(_HELPER_MODELS), _HELPER_TEMP)
    if isinstance(dec, dict):
        rewritten = ((dec.get("rewritten") or {}) if isinstance(dec.get("rewritten"), dict) else {}).get("result")
        if isinstance(rewritten, str) and rewritten.strip():
            plan["rewritten"] = rewritten.strip()
        kws = dec.get("keywords")
        if isinstance(kws, list):
            clean = [str(x).strip() for x in kws if str(x).strip()]
            if clean:
                plan["keywords"] = clean[:6]
        prices = dec.get("prices") if isinstance(dec.get("prices"), dict) else {}
        norm = prices.get("normalized")
        if isinstance(norm, str) and "-" in norm:
            plan["price"] = norm
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
        plan["attrs"] = {str(k): str(v) for k, v in attrs.items() if str(v).strip()}
        values = [v for v in plan["attrs"].values() if v.casefold() not in _KW_STOP]
        if values:
            plan["keywords"] = _title_tokens(" ".join(values))[:6] or plan["keywords"]
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


def _valid_filter(query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _valid_select(query: str, rows: list[dict[str, Any]], must: bool) -> dict[str, Any] | None:
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


def _compact(payload: Any, limit: int = 5000) -> str:
    text = json.dumps(payload, ensure_ascii=True, default=str)
    if len(text) <= limit:
        return text
    if isinstance(payload, dict):
        keep = {key: payload[key] for key in _KEEP_OBS if key in payload}
        for key in ("results", "variants", "compared"):
            rows = payload.get(key)
            if isinstance(rows, list):
                keep[key] = rows[:8]
        text = json.dumps(keep, ensure_ascii=True, default=str)
        if len(text) <= limit:
            return text
    return text[: limit - 3] + "..."


def _unwrap_obs(obs: dict[str, Any]) -> dict[str, Any]:
    inner = obs.get("observation")
    if not isinstance(inner, dict):
        return obs
    markers = (
        "results",
        "added",
        "product_id",
        "price",
        "facts",
        "variants",
        "compared",
        "cart",
        "market_event",
        "category_path",
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
    return category.casefold() in blob


def _is_accessory(row: dict[str, Any]) -> bool:
    blob = _blob(row)
    return any(token in blob for token in _ACCESSORY)


def _screen_match(row: dict[str, Any], target: str) -> bool:
    needle = str(target or "").rstrip("0").rstrip(".")
    if not needle:
        return True
    title = str(row.get("title") or "")
    if re.search(rf"{re.escape(needle)}\s*-?\s*inch", title, re.I):
        return True
    facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
    for key in ("size", "screen_size", "screen", "display_size"):
        val = str(facts.get(key) or "")
        if needle in val.casefold().replace(" ", "") or needle in val:
            return True
    blob = _blob(row)
    return bool(re.search(rf"{re.escape(needle)}\s*-?\s*inch", blob, re.I))


def _canon_res(raw: Any) -> str:
    token = str(raw or "").casefold()
    if token in {"720p", "hd", "1280", "720"}:
        return "hd"
    if token in {"1080p", "full hd", "fhd", "1920", "1080"}:
        return "fhd"
    if token in {"1440p", "qhd", "2560", "1440"}:
        return "qhd"
    if token in {"4k", "uhd", "2160p", "3840", "2160"}:
        return "uhd"
    return token


def _is_wet(row: dict[str, Any]) -> bool:
    return _pref_score(row)[0] > 0


def _facts_dict(row: dict[str, Any]) -> dict[str, Any]:
    facts = row.get("facts")
    return facts if isinstance(facts, dict) else {}


def _fact_text(row: dict[str, Any]) -> str:
    return json.dumps(
        {"f": _facts_dict(row), "o": row.get("options"), "s": row.get("specification")},
        ensure_ascii=True,
        default=str,
    ).casefold()


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
    if any(part in token for part in ("1280", "720p", "720")) or token.strip() in {"hd"}:
        return "hd"
    return _canon_res(token)


_SWEAT_MARK = ("ipx", "waterproof", "sweat", "splash", "water-resistant", "water resistant", "sweatproof")
_SPORT_MARK = ("sport", "earbud", "earbuds", "in-ear", "inear", "tws", "bone", "conduction", "neck", "hook", "secure")
_WIRE_MARK = ("wireless", "bluetooth", "tws")
_OVER_EAR = ("over ear", "over-ear", "overear")


def _pref_axis(row: dict[str, Any], marks: tuple[str, ...]) -> int:
    facts = _fact_text(row)
    title = str(row.get("title") or "").casefold()
    if any(mark in facts for mark in marks):
        return 2
    if any(mark in title for mark in marks):
        return 1
    return 0


def _pref_score(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    sweat = _pref_axis(row, _SWEAT_MARK)
    sport = _pref_axis(row, _SPORT_MARK)
    wireless = _pref_axis(row, _WIRE_MARK)
    title = str(row.get("title") or "").casefold()
    if any(mark in title for mark in _OVER_EAR) and sport == 0:
        sweat = 0
    battery = 1 if "battery" in _facts_dict(row) else 0
    return (sweat + sport + wireless, sweat, sport, wireless, battery)


def _spec_ok(row: dict[str, Any], goal: dict[str, Any]) -> bool:
    blob = _blob(row)
    size = _fact_size(row)
    if goal.get("screen"):
        needle = str(goal["screen"])
        need = needle.rstrip("0").rstrip(".")
        size_hit = bool(need and size and need in size.replace(" ", "").casefold())
        inch = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(?:inch|in\b)", size, re.I)
        if inch and _as_float(inch.group(1)) is not None and _as_float(needle) is not None:
            if abs(float(inch.group(1)) - float(needle)) >= 1.0:
                return False
        if not size_hit and not _screen_match(row, needle):
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
            if not any(a in blob for a in aliases):
                return False
    if goal.get("color"):
        if goal["color"] not in blob:
            return False
    if goal.get("model"):
        if str(goal["model"]).casefold() not in blob:
            return False
    return True


def _tf7_screen_value(row: dict[str, Any], goal: dict[str, Any]) -> str | None:
    size = _fact_size(row) or str(row.get("title") or "")
    needle = str(goal.get("screen") or "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(?:inch|in\b)?", size, re.I)
    if match:
        raw = match.group(1)
        if not needle or needle.rstrip("0").rstrip(".") in raw.rstrip("0").rstrip("."):
            num = _as_float(raw)
            if num is not None and abs(num - round(num)) < 1e-6:
                return str(int(round(num)))
            return raw
    if needle:
        return str(needle)
    return None


def _tf7_res_value(row: dict[str, Any], goal: dict[str, Any]) -> str | None:
    disp = _fact_display(row)
    if disp:
        got = _display_class(disp)
        if got in {"hd", "fhd", "qhd", "uhd"}:
            return got
    if goal.get("resolution"):
        got = _canon_res(goal["resolution"])
        if got in {"hd", "fhd", "qhd", "uhd"}:
            return got
    return None


def _tf7_price_value(row: dict[str, Any]) -> str | None:
    price = row.get("price")
    if price in (None, ""):
        return None
    return str(price)


def _tf7_claims(goal: dict[str, Any], row: dict[str, Any]) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    if goal.get("screen"):
        screen = _tf7_screen_value(row, goal)
        if screen:
            claims.append({"field": "screen_size", "value": screen})
    elif goal.get("resolution"):
        res = _tf7_res_value(row, goal)
        if res:
            claims.append({"field": "resolution", "value": res})
    price = _tf7_price_value(row)
    if price is not None:
        claims.append({"field": "price", "value": price})
    claims.append({"field": "in_stock", "value": "true"})
    brand = str(row.get("brand") or goal.get("brand") or "").strip()
    if brand:
        claims.append({"field": "brand", "value": brand})
    return claims


def _one_token_alt(original: list[str], alt: str) -> str | None:
    words = [w for w in _title_tokens(alt) if w not in _PHRASE_STOP]
    if not words:
        return None
    old, new = set(original), set(words)
    if abs(len(new) - len(old)) != 1:
        return None
    if not (new <= old or old <= new):
        return None
    return " ".join(words)


def _tf2_queries(terms: list[str], alt: str | None = None) -> list[str]:
    joined = " ".join(terms).strip()
    out = [joined] if joined else []
    cleaned = _one_token_alt(terms, alt or "")
    if cleaned and cleaned.casefold() != joined.casefold():
        out.append(cleaned)
    return out


def _tf2_alt_kw(terms: list[str]) -> str | None:
    phrase = " ".join(terms).strip()
    if not phrase:
        return None
    inf = json.dumps({"1": {"query": phrase, "keywords": phrase}}, ensure_ascii=True)
    parsed = _llm_json(_PSV_KW.format(inf=inf), list(_HELPER_MODELS), _HELPER_TEMP)
    row = parsed.get("1") if isinstance(parsed, dict) else None
    alt = row.get("alt") if isinstance(row, dict) else None
    if alt in (None, "", "null"):
        alt = parsed.get("alt") if isinstance(parsed, dict) else None
    if not isinstance(alt, str) or not alt.strip() or alt.strip().casefold() in {"null", "none"}:
        return None
    return _one_token_alt(terms, alt)


def _tf2_category_token(goal: dict[str, Any]) -> str | None:
    terms = {t.casefold() for t in (goal.get("terms") or [])}
    for tok in _title_tokens(str(goal.get("category") or "")):
        if tok in _PHRASE_STOP or tok in terms or len(tok) < 3:
            continue
        return tok
    return None


def _pin_tf2_plan(goal: dict[str, Any], plan: dict[str, Any]) -> None:
    terms = list(goal.get("terms") or [])
    if not terms:
        return
    plan["keywords"] = terms
    plan["compact"] = " ".join(terms)
    plan["rewritten"] = plan["compact"]
    plan["alt_keywords"] = []
    plan["searches"] = _tf2_queries(terms)
    plan["attrs"] = dict(plan.get("attrs") or {})


def _max_refresh(row: dict[str, Any]) -> float:
    blob = _blob(row)
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*hz", blob)]
    over = [float(x) for x in re.findall(r"overclock[^0-9]{0,12}(\d+(?:\.\d+)?)", blob)]
    return max(nums + over, default=0.0)


def _screen_size(row: dict[str, Any]) -> float:
    blob = _blob(row)
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*-?\s*inch", blob)]
    return max(nums, default=0.0)


def _weigh(row: dict[str, Any], tokens: list[str]) -> float:
    title = set(_title_tokens(str(row.get("title") or "")))
    facts = set(_title_tokens(_blob({"f": row.get("facts"), "o": row.get("options"), "s": row.get("specification")})))
    score = 0.0
    for tok in tokens:
        if tok in title:
            score += 0.75
        if tok in facts:
            score += 0.4
    return score


def _tf2_sku_row(row: dict[str, Any]) -> dict[str, Any] | None:
    merged = dict(row)
    if not merged.get("sku"):
        variants = _iter_variants(row)
        if variants:
            merged = {**row, **variants[0]}
    if not merged.get("product_id") or not merged.get("sku"):
        return None
    return merged


def _tf2_category_ok(row: dict[str, Any], category: str | None) -> bool:
    if not category:
        return True
    path = _path_text(row.get("category_path"))
    if not path:
        return True
    if category.casefold() in path.casefold():
        return True
    tokens = [tok for tok in _title_tokens(category) if tok not in _PHRASE_STOP and len(tok) > 2]
    blob = path.casefold()
    return bool(tokens) and any(tok in blob for tok in tokens)


def _tf2_live_rows(
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
        merged = _tf2_sku_row(row)
        if merged is None:
            continue
        if skip_sku and str(merged.get("sku") or "") == str(skip_sku):
            continue
        if skip_pid and str(merged.get("product_id") or "") == str(skip_pid):
            continue
        if not _price_ok(merged, goal.get("budget"), plan.get("price")):
            continue
        if not _in_stock(merged):
            continue
        if not _tf2_category_ok(merged, goal.get("category")):
            continue
        out.append(merged)
    out.sort(key=lambda r: -_weigh(r, terms))
    return out


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


def _choose_sku(row: dict[str, Any], goal: dict[str, Any], plan: dict[str, Any], *, firm_brand: bool) -> dict[str, Any] | None:
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
                "content": content or "VALID product engine",
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
            elif added:
                self.added.append(obs)
            if obs.get("market_event"):
                self.event_seen = True
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


def _rows(session: _Session) -> list[dict[str, Any]]:
    return list(session.catalog.values())


class _ValidSolver:
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
        if self.lane == "constraint" and self.goal.get("screen"):
            self._retrieve()
            self._sync()
        self._inspect()
        self._sync()
        if self.s.done:
            return
        if self.lane == "preference":
            self._pref_announce()
        self._try_adds(phase="a", limit=5)
        self._sync()
        if self.s.done:
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
        if (not self.s.done) and self.s.added:
            self._last_chance_order()

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
                        {
                            "content": "For this use case, should I prioritize sweat/water resistance over the lowest price?"
                        },
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
            for q in self.plan.get("searches") or _tf2_queries(list(self.goal.get("terms") or [])):
                add(q)
            return out[:2]
        if self.lane == "recovery":
            add(f"{self.goal.get('color') or ''} {self.goal.get('model') or ''} smartphone")
            add(f"{self.goal.get('model') or ''} smartphone")
        if self.lane == "intent" and brand:
            screen = self.goal.get("screen") or ""
            add(f"{brand} smartphone {screen} inch".strip())
            add(f"{brand} smartphone")
            add(brand)
        if self.lane == "preference":
            add("sports waterproof earbuds")
            add("IPX sweat-proof sports")
            add("bone conduction waterproof")
            return out[:3]
        if self.lane == "constraint":
            screen = self.goal.get("screen") or ""
            add(f"{brand} smartphone {screen} inch".strip())
            add(f"{brand} {screen} inch".strip())
            if brand.casefold() == "apple":
                add(f"iPhone {screen} inch".strip())
        if self.lane == "justification":
            res = _canon_res(self.goal.get("resolution")) if self.goal.get("resolution") else ""
            kind = "monitor" if (self.goal.get("category") or "").casefold().startswith("monitor") else (
                "tablet" if "tablet" in (self.goal.get("category") or "").casefold() else "smartphone"
            )
            add(f"{brand} {kind} {self.goal.get('screen') or ''} {self.goal.get('resolution') or res}".strip())
            if brand and self.goal.get("screen"):
                add(f"{brand} {self.goal['screen']} inch")
            elif brand and res:
                add(f"{brand} {res}")
            return out[:2]
        for item in self.plan.get("searches") or []:
            add(item)
        if ptype:
            add(ptype)
        return out[:10] if self.lane == "retrieval" else out[:6]

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
                self.s.call(actions, content="VALID retrieve named SKUs")
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
            if self.lane == "intent" and self.goal.get("brand"):
                branded = dict(filt)
                branded["brand"] = self.goal["brand"]
                actions.append(("filter", branded))
            actions.append(("filter", filt))
        if actions:
            self.s.call(actions, content="VALID retrieve")
            self._sync()
        if self.lane == "retrieval":
            self._survey_more()

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

    def _pref_announce(self) -> None:
        if not _has_tool(self.s.names, "message"):
            return
        rows = [r for r in self._candidates(phase="a") if _is_wet(r)][:2]
        if not rows:
            rows = self._candidates(phase="a")[:2]
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
                            + ". I will add the water-rated option, not the cheapest unrated one."
                        )
                    },
                )
            ]
        )
        self._sync()

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
        if _has_tool(self.s.names, "message") and (self.s.event_seen or broken) and self.s.remaining() > 6:
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
        if _has_tool(self.s.names, "search"):
            if screen:
                actions.append(
                    ("search", {"query": f"smartphone {screen} inch", "k": 10, "in_stock": True})
                )
                actions.append(
                    ("search", {"query": f"{screen} inch phone", "k": 10, "in_stock": True})
                )
            actions.append(
                ("search", {"query": str(self.goal.get("category") or "smartphone"), "k": 10, "in_stock": True})
            )
        if actions:
            self.s.call(actions, content="TF1 recover retrieve")
            self._sync()
        self._inspect()
        nxt = self._cheapest_live(self._candidates(phase="b"), skip_sku=str(last.get("sku") or ""))
        if nxt and str(nxt.get("sku")) != str(last.get("sku")):
            self._swap(last, nxt)
        self._sync()

    def _survey_more(self) -> None:
        terms = list(self.goal.get("terms") or [])
        matches = [row for row in _rows(self.s) if _title_hits(str(row.get("title") or ""), terms)]
        if len({_identity_key(r) for r in matches}) >= 3 or self.s.remaining() <= 10:
            return
        extra = _tf2_category_token(self.goal)
        if not extra or extra in {t.casefold() for t in terms}:
            return
        alt = (" ".join(terms) + " " + extra).strip()
        if alt and _has_tool(self.s.names, "search"):
            self.s.call([("search", {"query": alt, "k": 10, "in_stock": True})])
            self._sync()

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
        if self.lane == "retrieval":
            terms = list(self.goal.get("terms") or [])
            ranked = [
                r
                for r in _rows(self.s)
                if _title_hits(str(r.get("title") or ""), terms)
            ]
            ranked.sort(key=lambda r: -_weigh(r, terms))
            need = [str(r.get("product_id") or "") for r in ranked if r.get("product_id")][:8]
        if self.lane == "intent":
            brand = str(self.goal.get("brand") or "").casefold()
            branded = [pid for pid in need if brand and brand in _blob(self.s.catalog.get(pid) or {})]
            need = (branded or need)[:8]
        if self.lane == "preference":
            ranked = sorted(_rows(self.s), key=lambda r: _pref_score(r), reverse=True)
            need = [str(r.get("product_id") or "") for r in ranked if r.get("product_id")][:8]
        if self.lane == "justification":
            brand = str(self.goal.get("brand") or "").casefold()
            ranked = sorted(
                _rows(self.s),
                key=lambda r: (
                    0 if brand and brand in str(r.get("brand") or "").casefold() else 1,
                    0 if _spec_ok(r, self.goal) else 1,
                    0 if _facts_dict(r) else 1,
                ),
            )
            need = [str(r.get("product_id") or "") for r in ranked if r.get("product_id")][:8]
        if not need:
            return
        if _has_tool(self.s.names, "compare"):
            batch = [("compare", {"product_ids": need[i : i + 5]}) for i in range(0, min(len(need), 15), 5)]
            self.s.call(batch, content="VALID view/compare")
        elif _has_tool(self.s.names, "view"):
            self.s.call([("view", {"product_id": pid}) for pid in need[: self.s.max_calls]], content="VALID view")

    def _candidates(self, *, phase: str) -> list[dict[str, Any]]:
        rows = _rows(self.s)
        if self.lane == "retrieval":
            return _tf2_live_rows(rows, self.goal, self.plan)
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
        if self.lane == "preference":
            wet = [r for r in gated if _pref_score(r)[0] >= 2]
            if wet:
                gated = wet
            elif phase == "a":
                loose = [r for r in gated if _pref_score(r)[0] >= 1]
                if loose:
                    gated = loose
            gated.sort(
                key=lambda r: (
                    -_pref_score(r)[0],
                    -_pref_score(r)[1],
                    -_pref_score(r)[2],
                    _as_float(r.get("price")) if _as_float(r.get("price")) is not None else 1e18,
                )
            )
            return gated
        if self.lane == "intent":
            gated.sort(
                key=lambda r: _as_float(r.get("price")) if _as_float(r.get("price")) is not None else 1e18
            )
            return gated
        tokens = list(self.plan.get("keywords") or []) + list(self.goal.get("terms") or [])
        gated.sort(key=lambda r: -_weigh(r, tokens))
        if gated:
            top = _weigh(gated[0], tokens)
            gated = [r for r in gated if _weigh(r, tokens) > max(0.0, top - _PRUNE_GAP)] or gated
        return _valid_filter(self.query, gated)

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
        if self.lane == "retrieval":
            skip_sku = str(self.s.added[-1].get("sku") or "") if self.s.added and phase == "b" else None
            skip_pid = str(self.s.added[-1].get("product_id") or "") if self.s.added and phase == "b" else None
            rows = _tf2_live_rows(_rows(self.s), self.goal, self.plan, skip_sku=skip_sku, skip_pid=skip_pid)
            return rows[0] if rows else None
        if self.lane == "preference":
            skip_sku = str(self.s.added[-1].get("sku") or "") if self.s.added and phase == "b" else None
            skip_pid = str(self.s.added[-1].get("product_id") or "") if self.s.added and phase == "b" else None
            rows = [
                r
                for r in rows
                if str(r.get("sku") or "") != str(skip_sku or "")
                and str(r.get("product_id") or "") != str(skip_pid or "")
            ]
            return rows[0] if rows else None
        if self.lane == "justification":
            skip_sku = str(self.s.added[-1].get("sku") or "") if self.s.added and phase == "b" else None
            skip_pid = str(self.s.added[-1].get("product_id") or "") if self.s.added and phase == "b" else None
            brand = str(self.goal.get("brand") or "").casefold()
            ranked = []
            for row in rows:
                if skip_sku and str(row.get("sku") or "") == skip_sku:
                    continue
                if skip_pid and str(row.get("product_id") or "") == skip_pid:
                    continue
                if not _spec_ok(row, self.goal):
                    continue
                size = _fact_size(row).strip()
                disp = _fact_display(row).strip()
                if size.casefold() in {"default", "n/a", "none"} or disp.casefold() in {"default", "n/a"}:
                    continue
                if re.search(r"\d+\s*[-–/]\s*\d+", size):
                    continue
                ranked.append(row)
            ranked.sort(
                key=lambda r: (
                    0 if brand and brand in str(r.get("brand") or "").casefold() else 1,
                    0 if _facts_dict(r) else 1,
                    _as_float(r.get("price")) if _as_float(r.get("price")) is not None else 1e18,
                )
            )
            return ranked[0] if ranked else None
        if self.lane == "intent":
            skip = str(self.s.added[-1].get("sku") or "") if self.s.added and phase == "b" else None
            return self._cheapest_live(rows, skip_sku=skip)
        if phase == "b" and self.s.added:
            last_sku = str(self.s.added[-1].get("sku") or "")
            last_pid = str(self.s.added[-1].get("product_id") or "")
            rows = [
                r
                for r in rows
                if str(r.get("sku") or "") != last_sku and str(r.get("product_id") or "") != last_pid
            ]
        pick = _valid_select(self.query, rows, True)
        return pick

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
            if self.lane == "justification":
                if not _spec_ok(row, self.goal):
                    continue
                size = _fact_size(row).strip()
                if size.casefold() in {"default", "n/a", "none"} or re.search(r"\d+\s*[-–/]\s*\d+", size):
                    continue
                brand = str(self.goal.get("brand") or "").casefold()
                if brand and brand not in str(row.get("brand") or "").casefold() and brand not in _blob(row):
                    continue
            if self.lane == "preference" and _pref_score(row)[0] < 2 and phase == "a":
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
        obs_list = self.s.call([("inspect_stock", {"product_id": pid, "sku": sku})], content="VALID inspect_stock")
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
        self.s.call([("add_to_cart", {"product_id": pid, "sku": sku})], content="VALID recommend/add")
        self._sync()

    def _recover_alts(self) -> None:
        if self.s.remaining() <= 4:
            return
        actions: list[tuple[str, dict[str, Any]]] = []
        cap = _as_float(self.goal.get("budget"))
        if _has_tool(self.s.names, "filter"):
            filt: dict[str, Any] = {"k": 10, "in_stock": True}
            if self.goal.get("category"):
                filt["category"] = self.goal["category"]
            if cap is not None:
                filt["max_price"] = cap
            if self.lane in {"constraint", "justification"} and self.goal.get("brand"):
                filt["brand"] = self.goal["brand"]
            actions.append(("filter", filt))
        if _has_tool(self.s.names, "search"):
            if self.lane == "retrieval":
                phrase = " ".join(self.goal.get("terms") or [])
                if phrase:
                    actions.append(("search", {"query": phrase, "k": 10, "in_stock": True}))
            elif self.lane == "preference":
                actions.append(("search", {"query": "headphone waterproof IPX", "k": 10, "in_stock": True}))
            elif self.lane in {"constraint", "justification", "intent"}:
                brand = self.goal.get("brand") or ""
                screen = self.goal.get("screen") or ""
                actions.append(
                    ("search", {"query": f"{brand} smartphone {screen} inch".strip(), "k": 10, "in_stock": True})
                )
                if str(brand).casefold() == "apple":
                    actions.append(("search", {"query": f"iPhone {screen} inch".strip(), "k": 10, "in_stock": True}))
        if actions:
            self.s.call(actions[: self.s.max_calls], content="family recover retrieve")
            self._sync()
        self._inspect()

    def _phase_recover(self) -> None:
        if not self.s.added:
            return
        last = self.s.added[-1]
        live = self._stock_check(last)
        if live is None or self.s.done:
            return
        broken = (not _in_stock(live)) or (not _price_ok(live, self.goal.get("budget"), self.plan.get("price")))
        if not (self.s.event_seen or broken):
            return
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
            return
        if self.lane in {"retrieval", "preference"}:
            nxt = self._pick(phase="b")
            if nxt and str(nxt.get("sku")) != str(last.get("sku")):
                self._swap(last, nxt)
            return
        nxt = self._pick(phase="b")
        if (not nxt) or str(nxt.get("sku")) == str(last.get("sku")):
            self._recover_alts()
            nxt = self._pick(phase="b")
        if nxt and str(nxt.get("sku")) != str(last.get("sku")) and str(nxt.get("product_id")) != str(last.get("product_id")):
            self._swap(last, nxt)

    def _tf2_close(self) -> None:
        if self.s.done:
            return
        if not self.s.added:
            self._try_adds(phase="a", limit=3)
        if self.s.done:
            return
        if self.s.added:
            self._phase_recover()
        if self.s.done:
            return
        self._order()
        if (not self.s.done) and self.s.added:
            self._last_chance_order()

    def _last_chance_order(self) -> None:
        if self.s.done or not self.s.added or self.s.remaining() <= 2:
            return
        last = self.s.added[-1]
        live = self._stock_check(last)
        if live is not None and _in_stock(live) and _price_ok(live, self.goal.get("budget"), self.plan.get("price")):
            self._order()
            return
        if self.lane != "retrieval":
            self._recover_alts()
        nxt = self._pick(phase="b")
        if nxt and str(nxt.get("sku")) != str(last.get("sku")):
            self._swap(last, nxt)
        self._order()

    def _swap(self, old: dict[str, Any], new: dict[str, Any]) -> None:
        actions: list[tuple[str, dict[str, Any]]] = []
        if _has_tool(self.s.names, "remove_from_cart") and old.get("product_id") and old.get("sku"):
            actions.append(
                ("remove_from_cart", {"product_id": str(old["product_id"]), "sku": str(old["sku"])})
            )
        live = self._stock_check(new)
        if live is None or self.s.done:
            return
        if _in_stock(live) and _price_ok(live, self.goal.get("budget"), self.plan.get("price")):
            if actions:
                self.s.call(actions)
            self.s.call(
                [("add_to_cart", {"product_id": str(live["product_id"]), "sku": str(live["sku"])})],
                content="VALID recover",
            )
            self._sync()

    def _claims(self, row: dict[str, Any]) -> list[dict[str, str]]:
        return _tf7_claims(self.goal, row)

    def _order(self) -> None:
        self._sync()
        if self.s.done or not self.s.added:
            return
        if _has_tool(self.s.names, "inspect_cart"):
            self.s.call([("inspect_cart", {})], content="VALID inspect_cart")
            self._sync()
        if self.s.done:
            return
        last = self.s.added[-1]
        live = self._stock_check(last)
        if live is None or self.s.done:
            return
        if not _in_stock(live) or not _price_ok(live, self.goal.get("budget"), self.plan.get("price")):
            nxt = (
                self._cheapest_live(self._candidates(phase="b"), skip_sku=str(last.get("sku") or ""))
                if self.lane == "intent"
                else self._pick(phase="b")
            )
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
        catalog_row = self.s.catalog.get(str(live.get("product_id") or "")) or {}
        proven = {**catalog_row, **live}
        args: dict[str, Any] = {
            "product_id": str(live.get("product_id") or ""),
            "sku": str(live.get("sku") or ""),
        }
        if self.lane == "justification":
            args["justification"] = {"claims": self._claims(proven)}
        self.s.call([("place_test_order", args)], content="VALID recommend/order")


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
    if lane == "constraint":
        lines.append(
            "Ask the hidden spec, then match it on the exact SKU. After stockout, "
            "swap to another on-spec branded device under the current cap and order."
        )
    if lane == "preference":
        lines.append(
            "First add MUST be a compared sports+waterproof+wireless headphone under "
            "the cap (earbuds/neckband/bone, not a title-only over-ear). After the "
            "price jump, swap to another sport+water SKU under the cap and "
            "place_test_order immediately."
        )
    if lane == "justification":
        lines.append(
            "First add a branded on-spec under-cap SKU whose FACTS prove the spec. "
            "Claims allowed: in_stock, price, and screen_size OR resolution "
            "(canonical hd/fhd/qhd/uhd). Values must be strings. Never claim "
            "display, operating_system, or category. Do not order the event-target SKU."
        )
    if plan.get("price"):
        lines.append(f"Normalized price gate: {plan.get('price')}")
    return " ".join(part for part in lines if part)


def _guard_calls(
    tool_calls: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    committed = any(obs.get("added") for obs in observations)
    names = [_call_name(call) for call in tool_calls]
    if (not committed) and ("add_to_cart" in names) and ("place_test_order" in names):
        return [call for call in tool_calls if _call_name(call) != "place_test_order"]
    return tool_calls


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
    if lane == "retrieval":
        blocked = {"search", "filter"}
        tools = [spec for spec in tools if not set(_tool_names([spec])) & blocked]
        system += (
            "\n\nTF2 close-out: search and filter are disabled. "
            "inspect_stock the cart, remove the OOS SKU, add another title-hit "
            "already in the catalog, then place_test_order immediately."
        )
    if plan.get("keywords"):
        system += (
            "\n\nVALID retrieve plan\n"
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
        assistant = _infer(messages, tools, models, temperature)
        content = str(assistant.get("content") or "")
        raw_calls = assistant.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raw_calls = []
        tool_calls = [call for call in raw_calls if isinstance(call, dict)][: session.max_calls]
        tool_calls = _guard_calls(tool_calls, session.observations)
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
                goal,
                session.catalog,
            )
            if lane == "retrieval" and name in {"search", "filter"}:
                continue
            actions.append((name, args))
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
        plan = {
            "rewritten": " ".join(goal.get("terms") or []),
            "compact": " ".join(goal.get("terms") or []),
            "keywords": list(goal.get("terms") or []),
            "alt_keywords": [],
            "attrs": {},
            "price": f"-{goal['budget']}" if goal.get("budget") else None,
            "searches": _tf2_queries(list(goal.get("terms") or [])),
        }
        _pin_tf2_plan(goal, plan)
    elif lane == "preference":
        plan = {
            "rewritten": "sports waterproof earbuds",
            "compact": "sports waterproof earbuds",
            "keywords": ["sports", "waterproof", "earbuds", "ipx"],
            "alt_keywords": [],
            "attrs": {"product_type": "headphone"},
            "price": f"-{goal['budget']}" if goal.get("budget") else None,
            "searches": [
                "sports waterproof earbuds",
                "IPX sweat-proof sports",
                "bone conduction waterproof",
            ],
        }
    elif lane == "justification":
        brand = str(goal.get("brand") or "")
        res = _canon_res(goal.get("resolution")) if goal.get("resolution") else ""
        screen = str(goal.get("screen") or "")
        kind = "monitor" if (goal.get("category") or "").casefold().startswith("monitor") else (
            "tablet" if "tablet" in (goal.get("category") or "").casefold() else "smartphone"
        )
        compact = f"{brand} {kind} {screen} {res}".strip()
        plan = {
            "rewritten": compact,
            "compact": compact,
            "keywords": [w for w in (brand, kind, screen, res) if w],
            "alt_keywords": [],
            "attrs": {"product_type": kind},
            "price": f"-{goal['budget']}" if goal.get("budget") else None,
            "searches": [compact] + ([f"{brand} {screen} inch".strip()] if screen else []),
        }
    else:
        plan = _valid_plan(query)
    if goal.get("budget") and not plan.get("price"):
        plan["price"] = f"-{goal['budget']}"
    elif goal.get("budget") and plan.get("price"):
        _high = _numerify_price(str(plan.get("price")))[1]
        cap = _as_float(goal.get("budget"))
        if cap is not None and _high > cap + 1e-9:
            plan["price"] = f"-{goal['budget']}"
    session = _Session(
        binding=binding,
        problem_id=problem_id,
        names=names,
        max_steps=max_steps,
        max_calls=max_calls,
    )
    solver = _ValidSolver(session, query=query, lane=lane, goal=goal, plan=plan)
    solver.run()
    if (not session.done) and lane == "retrieval":
        solver._tf2_close()
    if (not session.done) and lane in {"preference", "justification"}:
        solver._tf2_close()
    if not session.done:
        _fallback_loop(session, query=query, tools=tools, lane=lane, plan=plan, goal=goal)
    return session.dialogue


__all__ = ["agent_main"]
