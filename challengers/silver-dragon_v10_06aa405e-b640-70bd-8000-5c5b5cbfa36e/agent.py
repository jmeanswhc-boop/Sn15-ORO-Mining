from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# =============================================================================
# OpenRouter only. TEE ids in SANDBOX_MODEL map to OpenRouter slugs.
# =============================================================================

_FAST = "deepseek/deepseek-v3.2"
_RECOVER = "z-ai/glm-5.2"
_BACKUP = "qwen/qwen3.5-397b-a17b"
_TO_OPENROUTER: dict[str, str] = {
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

_proxy = ProxyClient(timeout=45, max_retries=1)
_STARTED = 0.0
_BUDGET_S = 265.0
_OBS_CAP = 2400

F_INTENT = "intent_decomposition"
F_RECALL = "retrieval_recall"
F_CONSTRAINT = "constraint_satisfaction"
F_PREFERENCE = "preference_reasoning"
F_RANKING = "ranking"
F_RECOVERY = "recovery"
F_JUSTIFY = "justification"

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

_NAMED_SKU_RX = re.compile(r"\b(\d{6,})::([A-Za-z0-9._-]+)\b")
_NUM_RX = re.compile(r"(\d+(?:\.\d+)?)")
_BUDGET_RX = re.compile(
    r"(?:at or under|at or below|under|below|up to|upto|budget|<=)\s*"
    r"(\d+(?:\.\d+)?)",
    re.I,
)
_ASK_GATE_RX = re.compile(
    r"(?:tell|reveal|share).{0,48}\bif you ask\b|"
    r"\bif you ask\b|"
    r"\bask\b.{0,24}(?:exact )?(?:requirement|spec)\b|"
    r"\bwithhold\b.{0,36}\bspec\b",
    re.I,
)

_SEARCH_NAMES = ("search", "find_product", "search_products", "search_catalog")
_FILTER_NAMES = ("filter", "filter_catalog", "catalog_filter")
_VIEW_NAMES = ("view", "view_product", "view_product_information", "get_product")
_COMPARE_NAMES = ("compare", "compare_products")
_INSPECT_NAMES = ("inspect_stock", "check_stock", "get_stock")
_CART_NAMES = ("add_to_cart", "add_item", "cart_add")
_REMOVE_NAMES = ("remove_from_cart", "cart_remove", "remove_item")
_ORDER_NAMES = ("place_test_order", "place_order", "checkout")
_MESSAGE_NAMES = ("message", "talk", "respond", "say", "reply")
_MECH_HINTS = (
    "inspect_stock",
    "check_stock",
    "add_to_cart",
    "remove_from_cart",
    "place_test_order",
    "place_order",
    "checkout",
    "inspect_cart",
    "view_cart",
    "message",
    "talk",
    "respond",
)

_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "with",
        "from",
        "that",
        "this",
        "need",
        "want",
        "looking",
        "find",
        "show",
        "please",
        "would",
        "like",
        "under",
        "below",
        "budget",
        "task",
        "rules",
        "shopper",
    }
)

# Routing fingerprints (not sent to the model).
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
            re.compile(r"\d{6,}::\S+"),
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
            re.compile(r"\bevery requirement must\b|\bspec class\b", re.I),
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
                r"\brather pay\b|\bloose hard request\b",
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

_FAMILY_TIEBREAK: dict[str, int] = {
    F_RANKING: 7,
    F_RECOVERY: 6,
    F_JUSTIFY: 5,
    F_CONSTRAINT: 4,
    F_INTENT: 3,
    F_PREFERENCE: 2,
    F_RECALL: 1,
}

# =============================================================================
# General doctrine — no suite spec tokens, class names, or claim recipes
# =============================================================================

CORE_DOCTRINE = """
# Dragon Strategy Desk

You are a shopping agent in a generated environment. Use ONLY the tools
supplied this episode. Decisions come from the public goal, Task rules, and
live observations — never from memorized product ids.

## Contract
- Satisfy firm walls (budget, stock, category / real device, stated specs)
  while optimizing soft preferences when they exist.
- First add_to_cart is graded — never cart a placeholder.
- Never add_to_cart and place_test_order in the same turn.
- Continue until an observation reports done=true.

## Loop
1. Parse the shopper text: tag each constraint FIRM or SOFT.
2. Discover with the shopper's own words (values, not schema key names).
3. Inspect the exact option/sku (view / compare / inspect_stock).
4. Cart the verified pick. First cart must be intentional.
5. If the shopper asks a question, call message before further cart/order.
6. After price/stock/cart events, re-verify, recover, then close.
7. Finish with the order tool once the live winner is verified.

## Matching
- Literal for brands, model codes, SKUs, uncommon tokens, exact colors.
- Size aliases only when helpful.
- Semantic paraphrase for soft preferences only.
- Spec mappings: apply ONLY what Task rules and viewed facts state.
  Never invent class floors or extra pass/fail gates.

## Search
Good: product-type nouns, brand, model fragments, distinctive phrases.
Bad: schema key dumps; five synonymous rewrites that never change tokens.
Thin recall: one small reformulation, then inspect. Do not thrash.

## Dialogue
If user_message is a question, call message this or next turn with concrete
evidence, then continue. Ignoring a question fails even if the SKU is gold.

## Market / cart events
Inspect the affected sku → remove if invalid → recover on the active family
ladder → order to CLOSE the event. Never order the event target.

## Orders
If the shopper asked for a why, attach justification.claims from viewed facts
using only fields this episode's order tool allows. Otherwise pass ONLY
product_id and sku. Never invent attributes or copy title decoys.

## Turn budget
Early: discover + inspect (ask gated specs first if the goal says so).
Mid: commit first cart once evidence is enough.
Late: stop open search; finish inspect + order. Reserve turns to close events.

## Evidence
Structured option / attribute rows and inspect_stock beat marketing titles.
Model facts override conflicting titles. Phrase goals need title tokens.
Reject accessories / off-category noise unless the goal asks for them.

## Anti-patterns
- Placeholder first cart.
- Order from title alone without viewing the exact option.
- Keep a pre-event #1 after the market moves.
- Skip a recovery ladder level.
- Use disallowed or marketing strings as claim field names.
- Ask permission to order when the cart already holds a verified SKU.
""".strip()

PACK_INTENT = """
# Intent decomposition
Separate FIRM walls from SOFT preferences. Preferred brand is a scoring bonus:
first cart that brand IFF a fully firm-compliant SKU exists; otherwise cart
the best firm backup. Do not stall hunting an impossible brand.
Answer shopper questions with message before cart/order.
After budget/stock shocks: remove → re-find under the new world with the same
brand rule → order. Omit optional claims unless the shopper asked why.
""".strip()

PACK_RECALL = """
# Retrieval recall
Search distinctive tokens, then at most one short reformulation.
Shortlist several DISTINCT identities. Phrase goals: the phrase must appear
as title tokens on view. Drop accessories / wrong kinds via category when
available. Cart the best verified in-stock in-budget match and order promptly.
""".strip()

PACK_CONSTRAINT = """
# Constraint satisfaction
If a firm spec is revealed only when asked, call message BEFORE any search.
Every firm requirement must hold on the EXACT ordered option/sku — evidence
from the option row or inspect_stock, not a title guess.
Apply Task-rules mappings exactly; do not invent extra gates.
If the chosen variant sells out, move to a NEW product that still clears every
firm constraint (do not take a sibling that drops a gate).
""".strip()

PACK_PREFERENCE = """
# Preference reasoning
Hard walls: category / real device, budget, stock.
Soft axes: whatever use-case the shopper named. Soft never overrides hard.
Compare ≥2 hard-compliant identities on those named axes using viewed facts.
First cart = preference-fit winner (may be slightly dearer if still in budget),
not the cheapest stub. One clarifying message if priorities conflict, then
decide. After reprice, recover a different listing if needed.
""".strip()

PACK_RANKING = """
# Ranking
Closed set of named product_id::sku tokens from the goal. View/compare them
before any cart. Rank only by the shopper's stated axes in the stated
priority. Never invent extra axes. Never cart a dominated available option.
First cart MUST be the current best available candidate.
On stock/price events: re-inspect survivors, re-rank from scratch, cart the
NEW #1, then order. place_test_order with ONLY product_id and sku unless the
shopper explicitly asked for a why.
""".strip()

PACK_RECOVERY = """
# Recovery
Never relax category / product type, budget, or in-stock.
Least-relax ladder: exact model+color → same model other color → other models.
Never skip levels. Structured model facts beat conflicting titles.
After disappearance or budget break: inspect → remove if needed → climb one
level → re-check price+stock → order. Do not leave the event unfinished.
""".strip()

PACK_JUSTIFY = """
# Justification
Find a valid in-stock in-budget product, view/inspect so every claim is
grounded, then order WITH justification.claims.
Use only fields this episode's order tool allows. Fill them from viewed facts
/ inspect_stock, never from titles. Keep the set minimal and true.
After market events, rebuild claims from fresh facts. Do not invent
capabilities. Never cart and order in the same turn.
""".strip()

STRATEGY_PACKS: dict[str, str] = {
    F_INTENT: PACK_INTENT,
    F_RECALL: PACK_RECALL,
    F_CONSTRAINT: PACK_CONSTRAINT,
    F_PREFERENCE: PACK_PREFERENCE,
    F_RANKING: PACK_RANKING,
    F_RECOVERY: PACK_RECOVERY,
    F_JUSTIFY: PACK_JUSTIFY,
}

_MODE_LEGEND = {
    MODE_ASK_SPEC: "Ask the gated firm spec via message before any search.",
    MODE_CLOSED_RANK: "Closed-set rank over named product_id::sku tokens.",
    MODE_BRAND_SOFT: "Preferred brand is soft — brand-first iff fully firm-compliant.",
    MODE_THIN_RECALL: "Expect thin recall; one near-miss refine then shortlist.",
    MODE_SOFT_COMPARE: "Compare ≥2 hard-OK candidates on named soft axes before cart.",
    MODE_LEAST_RELAX: "Climb exact → other-color → other-model; never relax walls.",
    MODE_CLAIM_BUILD: "Attach grounded claims using only fields the order tool allows.",
    MODE_INTERVENE: "Expect shopper questions; message before cart/order.",
    MODE_MARKET: "Expect market/cart shocks; recover then close with order.",
    MODE_PHRASE: "Verify distinctive phrase as title tokens on view.",
    MODE_DEFAULT: "Follow the active family recipe.",
}

# Claim hygiene lives in Python (not in the prompt).
_FIELD_REMAP: dict[str, str] = {
    "display": "resolution",
    "screen": "screen_size",
    "size": "screen_size",
    "screen size": "screen_size",
    "stock": "in_stock",
    "availability": "in_stock",
    "refresh": "refresh_rate",
    "memory": "ram",
    "cost": "price",
    "maker": "brand",
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
        "name",
        "note",
        "description",
    }
)
_CORE_CLAIM = frozenset(
    {"price", "in_stock", "brand", "resolution", "screen_size", "refresh_rate", "ram"}
)

# =============================================================================
# Clock / provider
# =============================================================================


def _arm_clock() -> None:
    global _STARTED
    _STARTED = time.monotonic()


def _left() -> float:
    if not _STARTED:
        return _BUDGET_S
    return _BUDGET_S - (time.monotonic() - _STARTED)


def _tee_to_openrouter(raw: str) -> str:
    key = (raw or "").strip()
    if not key:
        return _FAST
    mapped = _TO_OPENROUTER.get(key)
    if mapped:
        return mapped
    for tee_id, or_id in _TO_OPENROUTER.items():
        if key.endswith(tee_id.split("/", 1)[-1]):
            return or_id
    return key


def _select_model(family: str) -> str:
    forced = (getenv("SANDBOX_MODEL") or "").strip()
    if forced:
        return _tee_to_openrouter(forced)
    if family == F_RECOVERY:
        return _RECOVER
    return _FAST


def _model_chain(primary: str) -> list[str]:
    chain = [primary]
    if _left() >= 70 and _BACKUP not in chain:
        chain.append(_BACKUP)
    if primary != _FAST and _FAST not in chain and _left() >= 90:
        chain.append(_FAST)
    return chain[:2]


# =============================================================================
# Classification + prompt assembly
# =============================================================================


def _goal_head(query: str) -> str:
    text = query or ""
    marker = "Task rules:"
    cut = text.find(marker)
    return text if cut < 0 else text[:cut]


def detect_family(query: str) -> str:
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

    if _NAMED_SKU_RX.search(blob) and F_RANKING not in tally:
        tally[F_RANKING] = 1

    if not tally:
        return F_RECALL
    return max(
        tally.items(),
        key=lambda item: (item[1], _FAMILY_TIEBREAK.get(item[0], 0)),
    )[0]


def detect_mode(family: str, query: str) -> list[str]:
    head = _goal_head(query)
    low = (query or "").lower()
    head_l = head.lower()
    tags: list[str] = []

    if family == F_CONSTRAINT and (
        _ASK_GATE_RX.search(head) or _ASK_GATE_RX.search(query or "")
    ):
        tags.append(MODE_ASK_SPEC)
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
    if family == F_JUSTIFY:
        tags.append(MODE_CLAIM_BUILD)
    elif re.search(r"\bwhy\b|trustworthy reason|justification\.claims", head_l, re.I):
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
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return ordered


def _tool_name(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    fn = entry.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return str(entry.get("name") or "")


def _tool_catalog(tools: list[dict[str, Any]]) -> str:
    names = [_tool_name(entry) for entry in tools or [] if _tool_name(entry)]
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
    named = [f"{pid}::{sku}" for pid, sku in _NAMED_SKU_RX.findall(query or "")]
    named_line = ", ".join(list(dict.fromkeys(named))[:12]) if named else "(none parsed)"
    omit = family != F_JUSTIFY and MODE_CLAIM_BUILD not in modes
    mode_lines = "\n".join(
        f"  - {m}: {_MODE_LEGEND.get(m, 'follow family recipe')}" for m in modes
    )
    runtime = f"""
## Runtime
- Active family: {family}
- Mode tags:
{mode_lines}
- Available tools: {_tool_catalog(tools)}
- max_steps: {max_steps}
- max_calls_per_turn: {max_calls}
- Named candidate tokens parsed: {named_line}
- Ask-first hidden spec: {"YES — message before search" if MODE_ASK_SPEC in modes else "no"}
- Optional claims policy: {"OMIT justification/claims on order" if omit else "ATTACH grounded claims from viewed facts using only fields the order tool allows"}
- Always tool_choice=required; temperature=0; respect max_calls_per_turn.
""".strip()
    return "\n\n".join([CORE_DOCTRINE, pack, runtime])


def _wants_why(query: str) -> bool:
    head = _goal_head(query)
    if re.search(r"justification is optional", query or "", re.I):
        return bool(re.search(r"\bwhy\b|reason|justify", head, re.I))
    return bool(re.search(r"\bwhy\b|justif|reason|tell me why|walk me through", head, re.I))


# =============================================================================
# Desk ledger — last SKU + named-set rank (not a BM25 matcher)
# =============================================================================


def _variant_key(product: dict[str, Any]) -> str:
    pid = str(product.get("product_id") or product.get("id") or "")
    sku = str(product.get("sku") or "")
    if pid and sku:
        return f"{pid}::{sku}"
    return pid


def _facts(product: dict[str, Any]) -> dict[str, Any]:
    raw = (
        product.get("facts")
        or product.get("specification")
        or product.get("attributes")
        or product.get("specs")
        or {}
    )
    return raw if isinstance(raw, dict) else {}


def _number(text: Any) -> float | None:
    found = _NUM_RX.search(str(text or ""))
    if not found:
        return None
    try:
        return float(found.group(1))
    except ValueError:
        return None


def _price_of(product: dict[str, Any]) -> float | None:
    raw = product.get("price")
    if isinstance(raw, dict):
        raw = raw.get("value") or raw.get("amount") or raw.get("current")
    try:
        if raw is None or raw == "":
            return None
        return float(raw)
    except (TypeError, ValueError):
        return _number(raw)


def _in_stock(product: dict[str, Any]) -> bool:
    if "in_stock" in product:
        value = product.get("in_stock")
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        text = str(value or "").strip().lower()
        if text in {"false", "0", "no", "oos", "out of stock", "unavailable"}:
            return False
        if text in {"true", "1", "yes", "in stock", "available"}:
            return True
        return True
    availability = str(product.get("availability") or product.get("stock") or "").lower()
    if not availability:
        return True
    return availability not in {"out of stock", "oos", "unavailable", "false", "0"}


def _fold(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _axes_from_text(text: str) -> list[tuple[str, str]]:
    lowered = (text or "").lower()
    axes: list[tuple[str, str]] = []
    patterns = (
        (r"highest ([a-z ]{3,28})", "desc"),
        (r"largest ([a-z ]{3,28})", "desc"),
        (r"biggest ([a-z ]{3,28})", "desc"),
        (r"most ([a-z ]{3,28})", "desc"),
        (r"lowest ([a-z ]{3,28})", "asc"),
        (r"smallest ([a-z ]{3,28})", "asc"),
        (r"cheapest ([a-z ]{3,28})?", "asc"),
    )
    seen: set[str] = set()
    for pattern, direction in patterns:
        for match in re.finditer(pattern, lowered):
            raw = (match.group(1) or "price").strip()
            name = _canon_axis(raw)
            if not name or name in seen:
                continue
            seen.add(name)
            axes.append((name, "asc" if "cheap" in raw or direction == "asc" else direction))
    if ("cheapest" in lowered or "lowest price" in lowered) and "price" not in seen:
        axes.append(("price", "asc"))
    return axes


def _canon_axis(raw: str) -> str:
    token = _fold(raw)
    if "refresh" in token or token in {"hz"}:
        return "refresh_rate"
    if "screen" in token or "size" in token:
        return "screen_size"
    if "price" in token or "cost" in token or "cheap" in token:
        return "price"
    return token.replace(" ", "_") or "price"


def _axis_value(product: dict[str, Any], name: str) -> float | None:
    key = _fold(name).replace(" ", "_")
    if key in {"price", "cost"}:
        return _price_of(product)
    facts = _facts(product)
    if "refresh" in key or key in {"hz"}:
        blob_parts = [
            product.get("options"),
            facts.get("refresh_rate"),
            facts.get("refresh"),
            product.get("title"),
            product.get("description"),
        ]
        best: float | None = None
        for chunk in blob_parts:
            for token in re.findall(
                r"(\d+(?:\.\d+)?)\s*(?:hz|hertz)\b", str(chunk or ""), re.I
            ):
                try:
                    value = float(token)
                except ValueError:
                    continue
                if 30 <= value <= 500:
                    best = value if best is None else max(best, value)
        return best
    if "screen" in key or key in {"size"}:
        return _screen_inches_row(product)
    if key in facts:
        return _number(facts.get(key))
    for fact_key, fact_value in facts.items():
        if key in _fold(fact_key).replace(" ", "_"):
            return _number(fact_value)
    return _number(product.get(key))


def _lex_sort(
    products: list[dict[str, Any]],
    axes: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    def key(product: dict[str, Any]) -> tuple:
        parts: list[float] = []
        for name, direction in axes:
            value = _axis_value(product, name)
            if value is None:
                parts.extend([1.0, 0.0])
                continue
            parts.append(0.0)
            parts.extend([-value if direction == "desc" else value])
        return tuple(parts)

    return sorted(products, key=key)


def _inner_obs(obs: Any) -> Any:
    if isinstance(obs, dict) and isinstance(obs.get("observation"), (dict, list)):
        return obs["observation"]
    return obs


def _obs_error(obs: Any) -> str:
    texts: list[str] = []
    node: Any = obs
    for _ in range(3):
        if not isinstance(node, dict):
            break
        for key in ("error", "detail", "message"):
            value = node.get(key)
            if value:
                texts.append(str(value))
        node = node.get("observation")
    blob = " ".join(texts).lower()
    if "not_in_cart" in blob or "not in cart" in blob:
        return "not_in_cart"
    return texts[0] if texts else ""


def _merge_row(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if value in (None, "", [], {}):
            continue
        if key in {"in_stock", "price", "availability", "current_price"}:
            dst[key] = value
            if key == "current_price" and dst.get("price") in (None, ""):
                dst["price"] = value
            continue
        if key not in dst or dst[key] in (None, "", [], {}):
            dst[key] = value
        elif key in {"facts", "attributes", "specification", "specs"} and isinstance(
            value, dict
        ):
            base = dst.get(key)
            if isinstance(base, dict):
                base.update({k: v for k, v in value.items() if v not in (None, "")})
            else:
                dst[key] = dict(value)


class DeskLedger:
    """Episode memory: viewed SKUs, named set, last cart. Not a catalog ranker."""

    __slots__ = (
        "query",
        "rows",
        "named",
        "axes",
        "budget",
        "cart_key",
        "last_key",
        "event_target",
        "inspected",
        "search_count",
        "first_cart",
        "ordered",
        "asked_spec",
        "await_reply",
        "saw_market",
        "used_coaches",
        "last_error",
    )

    def __init__(self, query: str) -> None:
        self.query = query or ""
        self.rows: dict[str, dict[str, Any]] = {}
        self.named = [
            {"product_id": pid, "sku": sku} for pid, sku in _NAMED_SKU_RX.findall(query)
        ]
        self.axes = _axes_from_text(_goal_head(query))
        found_budget = _BUDGET_RX.search(_goal_head(query))
        self.budget = float(found_budget.group(1)) if found_budget else None
        self.cart_key = ""
        self.last_key = ""
        self.event_target = ""
        self.inspected: set[str] = set()
        self.search_count = 0
        self.first_cart = False
        self.ordered = False
        self.asked_spec = False
        self.await_reply = False
        self.saw_market = False
        self.used_coaches: set[str] = set()
        self.last_error = ""

    def _put(self, raw: dict[str, Any], *, hydrated: bool) -> None:
        pid = str(raw.get("product_id") or raw.get("productId") or raw.get("id") or "")
        sku = str(raw.get("sku") or raw.get("sku_id") or "")
        if not pid:
            return
        key = f"{pid}::{sku}" if sku else pid
        row = dict(self.rows.get(key) or {})
        _merge_row(row, dict(raw))
        row["product_id"] = pid
        if sku:
            row["sku"] = sku
        if hydrated:
            row["_hydrated"] = True
        self.rows[key] = row
        if sku or hydrated:
            self.last_key = key

    def _put_view(self, inner: dict[str, Any]) -> None:
        pid = str(inner.get("product_id") or inner.get("productId") or "")
        facts = inner.get("facts") if isinstance(inner.get("facts"), dict) else {}
        base = {
            "product_id": pid,
            "title": inner.get("title"),
            "brand": inner.get("brand"),
            "category_path": inner.get("category_path"),
            "facts": facts,
            "_hydrated": True,
        }
        variants = inner.get("variants") or inner.get("sku_options")
        if isinstance(variants, list) and variants:
            for var in variants:
                if not isinstance(var, dict):
                    continue
                row = dict(base)
                row.update({k: v for k, v in var.items() if v not in (None, "")})
                row["product_id"] = pid
                self._put(row, hydrated=True)
            return
        if pid:
            self._put(base, hydrated=True)

    def ingest_text(self, text: str) -> None:
        found = _BUDGET_RX.search(text or "")
        if found:
            try:
                self.budget = float(found.group(1))
            except ValueError:
                return

    def ingest(self, obs: Any) -> None:
        err = _obs_error(obs)
        if err:
            self.last_error = err
            if err == "not_in_cart":
                self.ordered = False
                self.cart_key = ""
        inner = _inner_obs(obs)
        if not isinstance(inner, dict):
            if isinstance(inner, list):
                for item in inner:
                    if isinstance(item, dict):
                        self._put(item, hydrated=False)
            return
        kind = _market_kind(obs) or _market_kind(inner)
        cart = inner.get("cart")
        added = inner.get("added")
        event = inner.get("market_event") if isinstance(inner.get("market_event"), dict) else None
        if event is None and isinstance(obs, dict) and isinstance(obs.get("market_event"), dict):
            event = obs.get("market_event")
        if kind or event:
            self.saw_market = True
            target = self.cart_key
            src = None
            if isinstance(event, dict) and isinstance(event.get("target"), dict):
                src = event.get("target")
            if src is None and isinstance(added, dict):
                src = added
            if src is None and isinstance(cart, list) and cart and isinstance(cart[0], dict):
                src = cart[0]
            if isinstance(src, dict):
                pid = str(src.get("product_id") or "")
                sku = str(src.get("sku") or "")
                if pid:
                    target = f"{pid}::{sku}" if sku else pid
            if target:
                self.event_target = target
                self.inspected.clear()
                row = self.rows.get(target)
                if row is not None:
                    if event is None or event.get("in_stock_after") is False:
                        row["in_stock"] = False
                    if isinstance(event, dict) and event.get("new_price") is not None:
                        row["price"] = event["new_price"]
        if isinstance(cart, list):
            if cart and isinstance(cart[0], dict):
                pid = str(cart[0].get("product_id") or "")
                sku = str(cart[0].get("sku") or "")
                self.cart_key = f"{pid}::{sku}" if pid and sku else pid
                self.last_error = ""
            else:
                self.cart_key = ""
        if isinstance(added, dict):
            self.first_cart = True
            self._put(added, hydrated=True)
            pid = str(added.get("product_id") or "")
            sku = str(added.get("sku") or "")
            if pid:
                self.cart_key = f"{pid}::{sku}" if sku else pid
                self.last_error = ""
        results = inner.get("results")
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    self._put(item, hydrated=False)
        compared = inner.get("compared") or inner.get("products")
        if isinstance(compared, list):
            for item in compared:
                if isinstance(item, dict):
                    self._put_view(item)
        if inner.get("variants") or (
            inner.get("title") and inner.get("product_id") and not inner.get("sku")
        ):
            self._put_view(inner)
        elif inner.get("sku") and inner.get("product_id"):
            self._put(inner, hydrated=True)
        if inner.get("done") is True:
            self.ordered = True
            self.last_error = ""

    def note_action(self, name: str, args: dict[str, Any]) -> None:
        lowered = name.lower()
        pid = str(args.get("product_id") or args.get("id") or "")
        sku = str(args.get("sku") or "")
        key = f"{pid}::{sku}" if pid and sku else pid
        if any(token in lowered for token in ("search", "find_product")):
            self.search_count += 1
        if "message" in lowered or lowered in _MESSAGE_NAMES:
            self.asked_spec = True
            self.await_reply = False
        if "inspect" in lowered and "cart" not in lowered and key:
            self.inspected.add(key)
            self.last_key = key
        if any(token in lowered for token in ("view", "compare")) and key:
            self.last_key = key
        if "add_to_cart" in lowered or lowered in _CART_NAMES:
            self.first_cart = True
        if "remove" in lowered:
            self.cart_key = ""

    def row(self, key: str) -> dict[str, Any] | None:
        return self.rows.get(key)

    def _available(self, row: dict[str, Any]) -> bool:
        if not _in_stock(row):
            return False
        if self.budget is None:
            return True
        price = _price_of(row)
        if price is None:
            return True
        return price <= self.budget + 1e-6

    def ranked_winner(self) -> dict[str, Any] | None:
        pool: list[dict[str, Any]] = []
        if self.named:
            for spec in self.named:
                key = f"{spec['product_id']}::{spec['sku']}"
                row = self.rows.get(key)
                if row is None:
                    continue
                if _variant_key(row) == self.event_target:
                    continue
                if self._available(row):
                    pool.append(row)
        else:
            for key, row in self.rows.items():
                if key == self.event_target:
                    continue
                if self._available(row):
                    pool.append(row)
        if not pool:
            return None
        if self.axes and len(pool) > 1:
            pool = _lex_sort(pool, self.axes)
        return pool[0]

    def finish_row(self) -> dict[str, Any] | None:
        if self.named:
            winner = self.ranked_winner()
            if winner is not None:
                return winner
        hydrated = [
            row
            for row in self.rows.values()
            if row.get("_hydrated")
            and row.get("sku")
            and _variant_key(row) != self.event_target
            and self._available(row)
        ]
        if hydrated:
            head = _goal_head(self.query).lower()

            def score(row: dict[str, Any]) -> tuple:
                points = 0.0
                if _spec_ok(row, self.query):
                    points += 10.0
                brand = str(row.get("brand") or "").strip().lower()
                if brand and brand in head:
                    points += 5.0
                if _variant_key(row) == self.last_key:
                    points += 1.0
                return (points,)

            hydrated.sort(key=score, reverse=True)
            return hydrated[0]
        winner = self.ranked_winner()
        if winner is not None:
            return winner
        if self.cart_key and self.cart_key != self.event_target:
            row = self.rows.get(self.cart_key)
            if row is not None and self._available(row):
                return row
        if self.last_key and self.last_key != self.event_target:
            row = self.rows.get(self.last_key)
            if row is not None:
                return row
        return None


def _rank_ready(ledger: DeskLedger) -> bool:
    if not ledger.named:
        return False
    seen = sum(
        1
        for spec in ledger.named
        if f"{spec['product_id']}::{spec['sku']}" in ledger.rows
    )
    return seen >= min(len(ledger.named), 2) or _left() < 80


def _ids(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    payload: dict[str, Any] = {}
    pid = row.get("product_id") or row.get("id")
    sku = row.get("sku")
    if pid:
        payload["product_id"] = str(pid)
    if sku:
        payload["sku"] = str(sku)
    return payload


def _tool_params(entry: Any) -> dict[str, Any]:
    schema = entry
    if isinstance(entry, dict) and "schema" in entry:
        schema = entry.get("schema")
    if not isinstance(schema, dict):
        return {}
    fn = schema.get("function") if isinstance(schema.get("function"), dict) else schema
    params = fn.get("parameters") if isinstance(fn, dict) else None
    return params if isinstance(params, dict) else {}


def _bind_args(name: str, args: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {key: value for key, value in dict(args).items() if value not in (None, "", [])}
    wanted = str(name or "").lower()
    picked = None
    for entry in tools or []:
        if _tool_name(entry).lower() == wanted:
            picked = entry
            break
    params = _tool_params(picked) if picked is not None else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    if props:
        payload = {key: value for key, value in payload.items() if key in props}
    lowered = name.lower()
    if any(token in lowered for token in ("place_test_order", "place_order", "checkout")):
        payload.pop("product_ids", None)
        payload.pop("q", None)
        payload.pop("query", None)
    if "add_to_cart" in lowered or lowered in _CART_NAMES:
        payload.pop("product_ids", None)
        payload.pop("justification", None)
    return payload


def _pick_tool(tools: list[dict[str, Any]], names: tuple[str, ...]) -> dict[str, Any] | None:
    wanted = {name.lower() for name in names}
    for entry in tools or []:
        label = _tool_name(entry).lower()
        if label in wanted:
            return {"schema": entry, "name": _tool_name(entry)}
    for entry in tools or []:
        label = _tool_name(entry).lower()
        if any(token in label for token in wanted):
            return {"schema": entry, "name": _tool_name(entry)}
    return None


def _search_query(query: str) -> str:
    head = _goal_head(query)
    quotes = re.findall(r"[\"“]([^\"”]{3,60})[\"”]", head)
    if quotes:
        return quotes[0].strip()
    tokens: list[str] = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]{1,}", head):
        low = token.lower()
        if low in _STOP or len(low) < 2:
            continue
        tokens.append(token)
        if len(tokens) >= 6:
            break
    return " ".join(tokens[:6]) or "product"


def _schema_claim_fields(tools: list[dict[str, Any]]) -> set[str] | None:
    order = _pick_tool(tools, _ORDER_NAMES)
    if order is None:
        return None
    params = _tool_params(order)
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    block = props.get("justification") if isinstance(props, dict) else None
    if not isinstance(block, dict):
        return None
    claims = (block.get("properties") or {}).get("claims") if isinstance(
        block.get("properties"), dict
    ) else None
    items = (claims or {}).get("items") if isinstance(claims, dict) else None
    item_props = (items or {}).get("properties") if isinstance(items, dict) else None
    field_node = (item_props or {}).get("field") if isinstance(item_props, dict) else None
    enum = field_node.get("enum") if isinstance(field_node, dict) else None
    if isinstance(enum, list):
        found = {str(item).strip() for item in enum if isinstance(item, str) and item.strip()}
        if found:
            return found

    blobs: list[str] = []

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(node, str) and node.strip():
            blobs.append(node)
        elif isinstance(node, dict):
            for child in node.values():
                walk(child, depth + 1)
        elif isinstance(node, list):
            for child in node[:30]:
                walk(child, depth + 1)

    walk(block)
    text = " ".join(blobs).lower()
    found: set[str] = set()
    found.update(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", text))
    if re.search(r"\bin[_\s-]?stock\b", text):
        found.add("in_stock")
    for token in ("price", "brand", "resolution", "ram"):
        if re.search(rf"\b{token}\b", text):
            found.add(token)
    reason = re.search(r"reason[s:][^.!?]{0,240}", text)
    if reason:
        for part in re.split(r"[,/]| or ", reason.group(0)):
            token = re.sub(r"[^a-z0-9]+", "_", part.strip()).strip("_")
            if token.startswith("reason"):
                continue
            if re.fullmatch(r"[a-z][a-z0-9_]{1,24}", token) and token not in {
                "reason",
                "reasons",
                "such",
                "as",
                "one",
                "supported",
                "useful",
            }:
                found.add(token)
    found -= _CLAIM_DENY
    return found or None


def _allowed_claim_fields(tools: list[dict[str, Any]], query: str) -> set[str]:
    allowed = set(_schema_claim_fields(tools) or ())
    spec = _infer_spec_field(query)
    if spec:
        allowed.add(spec)
    allowed.update({"price", "in_stock", "brand"})
    return {field for field in allowed if field.lower() not in _CLAIM_DENY}


def _price_claim(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".")


_INCH_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(?:in(?:ch(?:es)?)?|\")",
    re.I,
)


def _inches_from_size_text(text: Any) -> float | None:
    raw = str(text or "").replace("\\", "").strip()
    if not raw:
        return None
    found = _INCH_RX.search(raw)
    if found:
        try:
            value = float(found.group(1))
        except ValueError:
            value = None
        if value is not None and 4 <= value <= 100:
            return value
    if re.fullmatch(r"\d+(?:\.\d+)?", raw):
        try:
            value = float(raw)
        except ValueError:
            return None
        if 4 <= value <= 100:
            return value
    return None


def _screen_inches_row(row: dict[str, Any]) -> float | None:
    inches = _inches_from_size_text(row.get("options"))
    if inches is not None:
        return inches
    facts = _facts(row)
    for raw in (facts.get("size"), facts.get("screen_size"), facts.get("screen")):
        inches = _inches_from_size_text(raw)
        if inches is not None:
            return inches
    return None


def _screen_claim(row: dict[str, Any]) -> str:
    inches = _screen_inches_row(row)
    if inches is None:
        return ""
    if abs(inches - round(inches)) < 0.05:
        return f"{int(round(inches))}in"
    return f"{inches:g}in"


def _shopper_inches(query: str) -> float | None:
    found = re.search(
        r"(\d+(?:\.\d+)?)\s*-?\s*inch",
        _goal_head(query),
        re.I,
    )
    if not found:
        return None
    try:
        return float(found.group(1))
    except ValueError:
        return None


def _fact_resolution_blob(row: dict[str, Any]) -> str:
    facts = _facts(row)
    parts: list[str] = []
    for key, raw in facts.items():
        folded = _fold(key)
        if any(hint in folded for hint in ("display", "resolution")):
            parts.append(str(raw or ""))
        elif folded == "screen":
            parts.append(str(raw or ""))
    return " ".join(parts)


def _compact_label(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    words = [word for word in words if word not in {"and", "or", "the"}]
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    return words[0][0] + words[-1]


def _brief_pixel_floors(query: str) -> list[tuple[str, int, int]]:
    """Parse 'Name starts at A by B' floors from this episode's brief, if any."""
    found: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"([A-Za-z0-9][A-Za-z0-9+/\s]{0,20}?)\s+(?:starts\s+)?at\s+"
        r"(\d{3,5})\s*(?:by|[x×])\s*(\d{3,5})",
        query or "",
        re.I,
    ):
        raw = re.sub(r"^(?:and|or|,)\s+", "", match.group(1).strip(), flags=re.I)
        left, right = int(match.group(2)), int(match.group(3))
        long_side, short_side = max(left, right), min(left, right)
        for alias in re.split(r"[/]", raw):
            label = _compact_label(alias)
            if not label or label in seen or label in {"at", "by"}:
                continue
            seen.add(label)
            found.append((label, long_side, short_side))
    found.sort(key=lambda item: item[1] * item[2])
    return found


def _shopper_res_need(query: str) -> dict[str, Any]:
    head = _goal_head(query)
    p_mode = re.search(
        r"(\d{3,4})\s*p(?:\s+((?:[A-Za-z]{3,8}\s+)?[A-Za-z0-9]{2,4})\b)?",
        head,
        re.I,
    )
    pair = re.search(r"(\d{3,5})\s*[x×]\s*(\d{3,5})", head)
    k_mode = re.search(r"\b(\d+)\s*k\b", head, re.I)
    token = ""
    p_val: int | None = None
    if p_mode:
        p_val = int(p_mode.group(1))
        if p_mode.group(2):
            token = _compact_label(p_mode.group(2))
    if not token and k_mode and p_val is None:
        token = f"{k_mode.group(1)}k"
    pair_val: tuple[int, int] | None = None
    if pair:
        left, right = int(pair.group(1)), int(pair.group(2))
        pair_val = (max(left, right), min(left, right))
    return {"p": p_val, "pair": pair_val, "token": token}


def _pixel_pair(blob: str) -> tuple[int, int] | None:
    pair = re.search(r"(\d{3,5})\s*[x×]\s*(\d{3,5})", blob)
    if not pair:
        return None
    left, right = int(pair.group(1)), int(pair.group(2))
    return max(left, right), min(left, right)


def _res_meets(row: dict[str, Any], query: str) -> bool:
    need = _shopper_res_need(query)
    blob = _fact_resolution_blob(row)
    got = _pixel_pair(blob)
    floors = _brief_pixel_floors(query)
    token = str(need.get("token") or "")
    if floors and token:
        wanted = next((item for item in floors if item[0] == token), None)
        if wanted and got:
            return got[0] >= wanted[1] and got[1] >= wanted[2]
    if need.get("p") and got:
        return min(got) >= int(need["p"]) or max(got) >= int(need["p"])
    if need.get("pair") and got:
        want = need["pair"]
        return got[0] >= want[0] and got[1] >= want[1]
    if token:
        return token in blob.lower()
    return True


def _resolution_claim(row: dict[str, Any], query: str) -> str:
    need = _shopper_res_need(query)
    token = str(need.get("token") or "")
    blob = _fact_resolution_blob(row)
    got = _pixel_pair(blob)
    if token and _res_meets(row, query):
        return token
    if got:
        return f"{got[0]}x{got[1]}"
    p_mode = re.search(r"(\d{3,4})\s*p\b", blob, re.I)
    if p_mode:
        return f"{p_mode.group(1)}p"
    return token


def _spec_ok(row: dict[str, Any], query: str) -> bool:
    field = _infer_spec_field(query)
    if field == "resolution":
        if not _shopper_res_need(query).get("p") and not _shopper_res_need(query).get("token"):
            return True
        return _res_meets(row, query)
    if field == "screen_size":
        wanted = _shopper_inches(query)
        got = _screen_inches_row(row)
        if wanted is None or got is None:
            return True
        return abs(got - wanted) <= 0.2
    return True


def _claim_value(row: dict[str, Any], field: str, query: str = "") -> str:
    facts = _facts(row)
    if field == "price":
        price = _price_of(row)
        return _price_claim(price) if price is not None else ""
    if field == "in_stock":
        return "true" if _in_stock(row) else "false"
    if field == "brand":
        return str(row.get("brand") or facts.get("brand") or "").strip()
    if field == "screen_size":
        return _screen_claim(row)
    if field == "refresh_rate":
        value = _axis_value(row, "refresh_rate")
        if value is None:
            return ""
        if abs(value - round(value)) < 0.05:
            return str(int(round(value)))
        return str(value)
    if field == "resolution":
        return _resolution_claim(row, query)
    raw = facts.get(field) or row.get(field)
    return "" if raw is None else str(raw).strip()


def _infer_spec_field(query: str) -> str | None:
    head = _goal_head(query).lower()
    if re.search(r"\d{3,4}\s*p|\d{3,5}\s*[x×]\s*\d{3,5}|\bresolution\b|\b\d+\s*k\b", head):
        return "resolution"
    if re.search(r"\d+(?:\.\d+)?\s*-?\s*inch|\bscreen size\b", head):
        return "screen_size"
    if re.search(r"refresh|\bhz\b", head):
        return "refresh_rate"
    return None


def _field_mentioned(field: str, query: str) -> bool:
    head = _goal_head(query).lower()
    token = field.lower().replace("_", " ")
    return token in head or field.lower() in head


def _ground_claims(
    row: dict[str, Any] | None,
    query: str,
    tools: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if row is None:
        return []
    allowed = _allowed_claim_fields(tools, query)
    kept: list[dict[str, str]] = []
    used: set[str] = set()

    def add(field: str) -> None:
        if field in used or field not in allowed:
            return
        value = _claim_value(row, field, query).strip()
        if not value:
            return
        used.add(field)
        kept.append({"field": field, "value": value})

    add("brand")
    add("in_stock")
    add("price")
    spec = _infer_spec_field(query)
    if spec:
        add(spec)
    else:
        for field in ("resolution", "screen_size", "refresh_rate", "ram"):
            if field in allowed and _claim_value(row, field, query):
                add(field)
                break
    for field in sorted(allowed):
        if field in _CORE_CLAIM or field in used:
            continue
        if _field_mentioned(field, query):
            add(field)
            break
    return kept


def _canon_field(field: str, query: str) -> str:
    raw = str(field or "").strip()
    key = raw.lower()
    if key in _CLAIM_DENY and key not in _FIELD_REMAP:
        return ""
    if key in _FIELD_REMAP:
        return _FIELD_REMAP[key]
    if re.search(r"\d", raw) and key not in _CORE_CLAIM:
        return _infer_spec_field(query) or ""
    return raw


def sanitize_order_args(
    tool_name: str,
    args: dict[str, Any],
    *,
    family: str,
    query: str,
    ledger: DeskLedger,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = dict(args)
    lowered = tool_name.lower()
    is_order = any(token in lowered for token in ("place_test_order", "place_order", "checkout"))
    is_cart = "add_to_cart" in lowered or lowered in _CART_NAMES
    is_inspect = "inspect" in lowered and "cart" not in lowered
    if family == F_RANKING and (is_order or is_cart or is_inspect) and _rank_ready(ledger):
        winner = ledger.ranked_winner()
        if winner is not None:
            payload["product_id"] = str(
                winner.get("product_id") or payload.get("product_id") or ""
            )
            if winner.get("sku"):
                payload["sku"] = str(winner.get("sku"))

    if not is_order:
        return _bind_args(tool_name, payload, tools)

    wants = _wants_why(query) or family == F_JUSTIFY
    if family == F_RANKING and not wants:
        payload.pop("justification", None)
        return _bind_args(tool_name, payload, tools)
    if not wants and family != F_JUSTIFY:
        payload.pop("justification", None)
        return _bind_args(tool_name, payload, tools)

    row = ledger.row(ledger.cart_key) or ledger.finish_row()
    grounded = _ground_claims(row, query, tools)
    if family == F_JUSTIFY and grounded:
        payload["justification"] = {"claims": grounded}
        return _bind_args(tool_name, payload, tools)

    block = payload.get("justification")
    raw_claims = block.get("claims") if isinstance(block, dict) else None
    if not isinstance(raw_claims, list):
        raw_claims = []

    allowed = _allowed_claim_fields(tools, query)
    kept: list[dict[str, str]] = []
    used: set[str] = set()
    reason_used = False
    spec = _infer_spec_field(query) or ""
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        field = _canon_field(str(item.get("field") or ""), query)
        value = _claim_value(row or {}, field, query) if row else str(item.get("value") or "").strip()
        if not field or value == "":
            continue
        fl = field.lower()
        if fl in _CLAIM_DENY or (allowed and fl not in {x.lower() for x in allowed}):
            continue
        extra = fl not in _CORE_CLAIM and fl != spec
        if extra and not _field_mentioned(fl, query):
            continue
        if extra or fl == "brand":
            if reason_used:
                continue
            reason_used = True
        if fl in used:
            continue
        used.add(fl)
        kept.append({"field": field, "value": value})

    if len(kept) < 3:
        for item in grounded:
            fl = item["field"].lower()
            if fl in used:
                continue
            used.add(fl)
            kept.append(item)

    payload["justification"] = {"claims": kept}
    return _bind_args(tool_name, payload, tools)


# =============================================================================
# Coaches + planner
# =============================================================================

_COACH = {
    "ask": (
        "Desk: a firm spec is withheld until asked. Call message NOW for that "
        "requirement. Do not search until the reply arrives."
    ),
    "reply": (
        "Desk: the shopper asked a question. Call message this turn with concrete "
        "evidence. Do not advance cart/order until you answer."
    ),
    "brand": (
        "Desk: keep a preferred-brand lane and a firm-any-brand backup. First cart "
        "is the brand SKU only if it clears every firm wall."
    ),
    "refine": (
        "Desk: if recall is thin, run ONE short reformulation from literal tokens, "
        "then inspect several distinct identities before committing."
    ),
    "soft": (
        "Desk: compare ≥2 hard-compliant candidates on the named soft axes with "
        "viewed facts, then first-cart the preference-fit winner — not the cheapest stub."
    ),
    "rank": (
        "Desk: extract every product_id::sku, compare the named set, rank only by "
        "stated axes, cart current #1. Skip open catalog thrash."
    ),
    "rerank": (
        "Desk: market moved. Re-inspect named survivors, re-rank with the same "
        "priority, cart the NEW #1, then prepare to order."
    ),
    "rank_finish": (
        "Desk: stop searching. Cart current #1 and call the order tool with ONLY "
        "product_id and sku."
    ),
    "ladder": (
        "Desk: exact model+color, then same model other color, then other models. "
        "Never skip levels. Never relax budget, stock, or category. Close with order."
    ),
    "claims": (
        "Desk: on order, attach grounded justification.claims using only fields "
        "this episode's order tool allows. Fill them from viewed facts. Do not invent."
    ),
    "omit": (
        "Desk: this family does not require a why. On order send product_id + sku "
        "only — omit justification/claims."
    ),
    "market": (
        "Desk: inspect the affected sku → remove if invalid → recover → order to "
        "close the event. Never order the event target."
    ),
    "late": (
        "Desk: late window. Stop open search. Re-verify stock/price on the intended "
        "sku and place the order."
    ),
    "auth": (
        "Desk: reject accessories / off-category noise that only match via title. "
        "Prefer the real device the shopper asked for."
    ),
    "idle": (
        "Desk: no tool call. Choose a supplied tool now (search/filter, view, "
        "message, or order if the cart already holds a verified SKU)."
    ),
}


def _desk_coaches(
    *,
    family: str,
    modes: list[str],
    ledger: DeskLedger,
    turn: int,
    max_steps: int,
) -> list[str]:
    notes: list[str] = []
    late = turn >= max(2, int(max_steps * 0.7))

    def add(key: str) -> None:
        if key in ledger.used_coaches:
            return
        text = _COACH.get(key)
        if not text:
            return
        ledger.used_coaches.add(key)
        notes.append(text)

    if MODE_ASK_SPEC in modes and not ledger.asked_spec and turn <= 2:
        add("ask")
    if ledger.await_reply:
        add("reply")
    if family == F_INTENT and turn <= 4 and not ledger.first_cart:
        add("brand")
    if family == F_RECALL and ledger.search_count >= 1 and not ledger.first_cart:
        add("refine")
    if family == F_PREFERENCE and not ledger.first_cart:
        add("soft")
    if family == F_RANKING:
        if not ledger.first_cart:
            add("rank")
        if ledger.saw_market and not ledger.ordered:
            add("rerank")
        if late or (ledger.saw_market and turn >= 2):
            add("rank_finish")
        add("omit")
    if family == F_RECOVERY:
        add("ladder")
    if family == F_JUSTIFY and (ledger.first_cart or late):
        add("claims")
    if ledger.saw_market and not ledger.ordered:
        add("market")
    if family not in {F_JUSTIFY, F_RANKING} and late:
        add("omit")
    if late and not ledger.ordered:
        add("late")
    if family in {F_RECALL, F_CONSTRAINT, F_PREFERENCE} and not ledger.first_cart:
        add("auth")
    return notes[:3]


def _emit(tool: dict[str, Any] | None, args: dict[str, Any]) -> list[dict[str, Any]]:
    if tool is None:
        return []
    return [{"name": tool["name"], "args": args}]


def _desk_plan(
    *,
    family: str,
    modes: list[str],
    ledger: DeskLedger,
    tools: list[dict[str, Any]],
    query: str,
    max_calls: int,
) -> list[dict[str, Any]]:
    search_tool = _pick_tool(tools, _SEARCH_NAMES)
    view_tool = _pick_tool(tools, _VIEW_NAMES)
    compare_tool = _pick_tool(tools, _COMPARE_NAMES)
    inspect_tool = _pick_tool(tools, _INSPECT_NAMES)
    cart_tool = _pick_tool(tools, _CART_NAMES)
    remove_tool = _pick_tool(tools, _REMOVE_NAMES)
    order_tool = _pick_tool(tools, _ORDER_NAMES)
    message_tool = _pick_tool(tools, _MESSAGE_NAMES)

    if MODE_ASK_SPEC in modes and not ledger.asked_spec:
        return _emit(
            message_tool,
            {"content": "What is the required spec I should treat as firm?"},
        )

    stale = bool(
        ledger.cart_key
        and ledger.cart_key == ledger.event_target
        and ledger.saw_market
        and not ledger.ordered
    )
    if stale:
        row = ledger.row(ledger.cart_key)
        if row is not None:
            return _emit(remove_tool, _ids(row))

    recovering = bool(ledger.saw_market and not ledger.ordered)
    if ledger.await_reply and not recovering:
        return _emit(
            message_tool,
            {"content": "I will confirm against the live listing and continue."},
        )

    if family == F_RANKING and ledger.named:
        unviewed = [
            spec
            for spec in ledger.named
            if f"{spec['product_id']}::{spec['sku']}" not in ledger.rows
        ]
        if unviewed and (compare_tool or view_tool):
            ids = [spec["product_id"] for spec in unviewed]
            if compare_tool is not None and len(ids) >= 2:
                return _emit(compare_tool, {"product_ids": ids[: min(5, max_calls + 3)]})
            return _emit(view_tool, {"product_id": ids[0], "product_ids": [ids[0]]})

    survey = family in {F_PREFERENCE, F_INTENT, F_RECALL} and _left() > 70
    identities = {
        str(row.get("product_id") or "")
        for row in ledger.rows.values()
        if row.get("product_id")
    }
    if survey and len(identities) < 2 and search_tool is not None and ledger.search_count < 4:
        q = _search_query(query)
        return _emit(search_tool, {"query": q, "q": q})

    winner = ledger.finish_row()
    if winner is not None:
        ids = _ids(winner)
        key = _variant_key(winner)
        if inspect_tool is not None and key not in ledger.inspected:
            return _emit(inspect_tool, ids)
        if ledger.cart_key and ledger.cart_key != key:
            stale_row = ledger.row(ledger.cart_key)
            if stale_row is not None:
                return _emit(remove_tool, _ids(stale_row))
        if not ledger.cart_key or ledger.last_error == "not_in_cart":
            if survey and len(identities) < 2 and not ledger.saw_market:
                q = _search_query(query)
                if search_tool is not None and ledger.search_count < 4:
                    return _emit(search_tool, {"query": q, "q": q})
            return _emit(cart_tool, ids)
        order_args = dict(ids)
        if family == F_JUSTIFY or _wants_why(query):
            claims = _ground_claims(winner, query, tools)
            if claims:
                order_args["justification"] = {"claims": claims}
            elif inspect_tool is not None and key not in ledger.inspected:
                return _emit(inspect_tool, ids)
        else:
            order_args.pop("justification", None)
        return _emit(order_tool, order_args)

    if search_tool is not None and ledger.search_count < 3:
        return _emit(search_tool, {"query": _search_query(query), "q": _search_query(query)})
    if view_tool is not None and ledger.last_key:
        row = ledger.row(ledger.last_key)
        if row is not None:
            return _emit(view_tool, _ids(row))
    return []


def _mechanical(actions: list[dict[str, Any]]) -> bool:
    if not actions:
        return False
    for action in actions:
        name = str(action.get("name") or "").lower()
        if not any(hint in name for hint in _MECH_HINTS):
            return False
    return True


def _skip_llm(
    *,
    family: str,
    proposed: list[dict[str, Any]],
    ledger: DeskLedger,
    turn: int,
    max_steps: int,
) -> bool:
    if _left() < 55:
        return True
    if family == F_RANKING and proposed:
        return True
    if ledger.await_reply:
        return _left() < 90
    if ledger.first_cart and ledger.cart_key and proposed:
        names = " ".join(str(item.get("name") or "").lower() for item in proposed)
        if any(token in names for token in ("place_test_order", "place_order", "inspect")):
            return True
    if family == F_RANKING and _rank_ready(ledger) and _mechanical(proposed):
        return True
    if turn >= max(4, int(max_steps * 0.7)):
        return True
    if proposed and turn >= 3 and _left() < 110:
        return True
    return False


def _trim_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(messages) <= 16:
        return messages
    head = messages[:2]
    tail = messages[-12:]
    if tail and tail[0].get("role") == "tool":
        for index, item in enumerate(messages[2:-12], start=2):
            if item.get("role") == "assistant" and item.get("tool_calls"):
                return head + messages[index:]
    return head + tail


def _drop_same_turn_order(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = [str(item.get("name") or "").lower() for item in actions]
    carted = any("add_to_cart" in name or name in _CART_NAMES for name in names)
    if not carted:
        return actions
    kept: list[dict[str, Any]] = []
    for action, name in zip(actions, names):
        if any(token in name for token in ("place_test_order", "place_order", "checkout")):
            continue
        kept.append(action)
    return kept or actions


def _to_tool_calls(
    actions: list[dict[str, Any]],
    raw_calls: list[dict[str, Any]],
    problem_id: str,
    turn: int,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for index, action in enumerate(actions, start=1):
        raw = raw_calls[index - 1] if index - 1 < len(raw_calls) else {}
        calls.append(
            {
                "id": action.get("_id")
                or raw.get("id")
                or f"{problem_id}-desk-{turn}-{index}",
                "type": "function",
                "function": {
                    "name": action["name"],
                    "arguments": json.dumps(action.get("args") or {}, ensure_ascii=False),
                },
            }
        )
    return calls


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
    node = obs.get("market_event")
    if isinstance(node, dict):
        return str(node.get("kind") or "market_event")
    if node:
        return "market_event"
    nested = obs.get("observation")
    if isinstance(nested, dict):
        return _market_kind(nested)
    return ""


def _obs_done(obs: Any) -> bool:
    if isinstance(obs, dict):
        if obs.get("done") is True:
            return True
        nested = obs.get("observation")
        if isinstance(nested, dict) and nested.get("done") is True:
            return True
    return False


def _inference_chat(
    *,
    models: list[str],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    last_model = models[0] if models else _FAST
    for model_id in models:
        last_model = model_id
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
        except Exception:  # noqa: BLE001 — sandbox may raise variously
            continue
        if inference is None:
            continue
        choices = inference.get("choices") or []
        if not choices:
            continue
        message = choices[0].get("message") or {}
        return message, model_id
    return None, last_model


# =============================================================================
# Entry
# =============================================================================


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Dragon Strategy Desk v10. Returns dialogue transcript."""
    _arm_clock()
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
    primary_model = _select_model(family)
    ledger = DeskLedger(query)

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
                "version": 10,
                "family": family,
                "mode": modes,
                "model": primary_model,
                "provider": "openrouter",
                "lineage": "rank3_strategy_desk",
            },
        }
    ]

    for turn in range(1, max_steps + 1):
        if _left() < 8:
            break
        proposed = _desk_plan(
            family=family,
            modes=modes,
            ledger=ledger,
            tools=tools,
            query=query,
            max_calls=max_calls,
        )
        skip = _skip_llm(
            family=family,
            proposed=proposed,
            ledger=ledger,
            turn=turn,
            max_steps=max_steps,
        )
        actions: list[dict[str, Any]] = []
        assistant_content = ""
        raw_calls: list[dict[str, Any]] = []
        used_model = primary_model

        if not skip:
            coaches = _desk_coaches(
                family=family,
                modes=modes,
                ledger=ledger,
                turn=turn,
                max_steps=max_steps,
            )
            turn_messages = list(messages)
            if coaches:
                turn_messages = turn_messages + [
                    {"role": "user", "content": "Desk notes:\n- " + "\n- ".join(coaches)}
                ]
            turn_messages = _trim_messages(turn_messages)
            chain = _model_chain(primary_model)
            assistant, used_model = _inference_chat(
                models=chain,
                messages=turn_messages,
                tools=tools,
            )
            if assistant is not None:
                assistant_content = assistant.get("content") or ""
                raw_calls = list(assistant.get("tool_calls") or [])[:max_calls]
                for call in raw_calls:
                    fn = call.get("function") or {}
                    name = str(fn.get("name") or "")
                    try:
                        args = _parse_tool_args(call)
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue
                    args = sanitize_order_args(
                        name,
                        args,
                        family=family,
                        query=query,
                        ledger=ledger,
                        tools=tools,
                    )
                    actions.append({"name": name, "args": args, "_id": call.get("id")})
            if not actions and _left() >= 70 and not proposed:
                nudge = messages + [{"role": "user", "content": _COACH["idle"]}]
                assistant, used_model = _inference_chat(
                    models=chain[:1],
                    messages=_trim_messages(nudge),
                    tools=tools,
                )
                if assistant is not None:
                    assistant_content = assistant.get("content") or ""
                    raw_calls = list(assistant.get("tool_calls") or [])[:max_calls]
                    for call in raw_calls:
                        fn = call.get("function") or {}
                        name = str(fn.get("name") or "")
                        try:
                            args = _parse_tool_args(call)
                        except (ValueError, TypeError, json.JSONDecodeError):
                            continue
                        args = sanitize_order_args(
                            name,
                            args,
                            family=family,
                            query=query,
                            ledger=ledger,
                            tools=tools,
                        )
                        actions.append(
                            {"name": name, "args": args, "_id": call.get("id")}
                        )

        if not actions:
            actions = proposed[:max_calls]
        if not actions:
            break

        if ledger.await_reply:
            names = {str(item.get("name") or "").lower() for item in actions}
            recovering = ledger.saw_market and not ledger.ordered
            if (
                not recovering
                and not names.intersection(_MESSAGE_NAMES)
                and "message" not in " ".join(names)
            ):
                forced = _desk_plan(
                    family=family,
                    modes=modes,
                    ledger=ledger,
                    tools=tools,
                    query=query,
                    max_calls=max_calls,
                )
                if forced:
                    actions = forced[:max_calls]

        actions = _drop_same_turn_order(actions)[:max_calls]
        repaired: list[dict[str, Any]] = []
        for action in actions:
            name = str(action.get("name") or "")
            args = sanitize_order_args(
                name,
                dict(action.get("args") or {}),
                family=family,
                query=query,
                ledger=ledger,
                tools=tools,
            )
            lowered = name.lower()
            is_order = any(
                token in lowered for token in ("place_test_order", "place_order", "checkout")
            )
            if is_order and not ledger.cart_key:
                cart_tool = _pick_tool(tools, _CART_NAMES)
                if cart_tool is not None:
                    repaired.append(
                        {
                            "name": cart_tool["name"],
                            "args": _bind_args(cart_tool["name"], args, tools),
                        }
                    )
                    continue
            repaired.append({"name": name, "args": args})
        actions = repaired[:max_calls]
        for action in actions:
            ledger.note_action(str(action.get("name") or ""), action.get("args") or {})

        tool_calls = _to_tool_calls(actions, raw_calls, problem_id, turn)
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
                        "name": action["name"],
                        "args": action.get("args") or {},
                    },
                }
                for index, action in enumerate(actions, start=1)
            ],
        }
        try:
            result = _proxy.post("/environment/call", json_data=envelope)
        except Exception:  # noqa: BLE001
            break
        if result is None:
            break

        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
                "dragon_model": used_model,
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )

        done_any = False
        for tool_call, call_result in zip(tool_calls, result.get("calls") or []):
            obs = call_result.get("observation")
            ledger.ingest(obs)
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
                ledger.ingest_text(content)
                if _shopper_needs_reply(content):
                    ledger.await_reply = True

        if done_any or any(
            _obs_done(call.get("observation")) for call in (result.get("calls") or [])
        ):
            break

    return dialogue


__all__ = ["agent_main"]
