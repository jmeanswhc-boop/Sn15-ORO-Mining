"""ORO-Bench environment agent: a deterministic shopping protocol around an LLM.

The model formulates searches, reads product facts and picks candidates. The
controller owns the order of operations the task rules require: clarify the
requirements first, inspect a candidate before committing it, verify the cart
after every cart change, recover to a different product when the committed
item is gone or over budget, answer the shopper, and only then place the order.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
# Same models on the other allowlisted provider names; tried in order when the
# primary model keeps failing on a validator (provider outage / rate limit).
_FALLBACK_MODELS = {
    "chutes": ["deepseek-ai/DeepSeek-V3.1-TEE", "Qwen/Qwen3.5-397B-A17B-TEE", "zai-org/GLM-5.2-TEE"],
    "openrouter": ["deepseek/deepseek-chat-v3.1", "qwen/qwen3.5-397b-a17b", "z-ai/glm-5.2"],
}
# Short per-request budget: a hung inference call must never eat the 300 s problem wall.
_proxy = ProxyClient(timeout=45, max_retries=2)
# After this many consecutive failed inference turns the controller stops asking the
# model and finishes the episode deterministically.
_LLM_FAILURES_BEFORE_AUTOPILOT = 2

# Wall-clock plan for one problem (the sandbox kills a problem at 300s).
_SOFT_DEADLINE = 195.0
_HARD_DEADLINE = 250.0
# Turns kept in reserve so a forced finish (remove/add/inspect/order) always fits.
_RESERVE_TURNS = 4
# The FIRST cart add must happen with this many turns still available: the market event
# fires on it and a recovery needs remove + view/inspect + add + inspect + (reply) + order.
_FIRST_ADD_RESERVE = 11
# A recovery (after the event killed the first item) must start with this many turns left.
_RECOVERY_RESERVE = 6
_SEARCH_CAP = {"retrieval_recall": 3}
_SEARCH_CAP_DEFAULT = 8
_SEARCH_CAP_TF4 = 12
_MESSAGE_CAP = 4
_CURRENCIES = "PHP|USD|EUR|GBP|SGD|MYR|IDR|THB|VND|AUD|CAD|JPY|INR|HKD|TWD|KRW|CNY|NZD"
_AMOUNT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(" + _CURRENCIES + r")\b", re.I)
_BUDGET_WORDS = re.compile(
    r"\b(budget|at most|no more than|not more than|under|below|max(?:imum)?|cap|"
    r"up to|spend|afford|ceiling|limit)\b",
    re.I,
)
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me|confirm|reply|respond|answer)\b",
    re.I,
)
_SPEC_CLAIMS = {
    "resolution", "refresh_rate", "screen_size", "capacity", "wattage",
    "warranty_duration", "pack_quantity",
}
_SOFT_CLAIMS = {"brand", "streaming", "microphone", "privacy_cover", "autofocus", "hdr"}
_CLAIM_FIELDS = {"category", "price", "in_stock"} | _SPEC_CLAIMS | _SOFT_CLAIMS
# Feature reasons the verifier can support, with the wording it looks for.
_SOFT_FEATURE_RE = {
    "microphone": r"\b(?:mic|microphone)\b",
    "hdr": r"\b(hdr|high dynamic range)\b",
    "streaming": r"\b(streaming|conference|conferencing|zoom|teams)\b|\bvideo[- ]?call(?:ing|s)?\b",
    "autofocus": r"\bauto[- ]?focus\b",
    "privacy_cover": r"\bprivacy (?:cover|shutter)\b",
}
_SOFT_NEG_BEFORE = (
    r"\b(?:no|without|lack(?:s|ing)?|missing|exclude[ds]?|unsupported|not|cannot|can['\u2019]t|"
    r"doesn['\u2019]t|optional)\b(?:[\s-]+[\w'\u2019]+){0,5}[\s-]*$"
)
_SOFT_NEG_AFTER = (
    r"\s*(?:(?:support|feature|function|mode|capability)\s*)?[:=]?\s*(?:(?:is|are|was|were)\s+)?"
    r"(?:not|no|false|none|unsupported|absent|unavailable|missing|excluded|disabled|optional|0)\b|"
    r"^\s*[- ]free\b|^\s+sold\s+separately\b"
)

# Two markers per family, taken from each family's runtime system text (0.2.13 and 0.2.18).
_FAMILY_MARKERS = (
    ("intent_decomposition", "MIX of firm requirements"),
    ("intent_decomposition", "named preferred brand is a SOFT preference"),
    ("retrieval_recall", "Survey the realistic options"),
    ("retrieval_recall", "a good shortlist covers several distinct products"),
    ("constraint_satisfaction", "EXACT variant you order"),
    ("constraint_satisfaction", "firm, non-negotiable requirements: a category"),
    ("preference_reasoning", "loose hard request plus soft preferences"),
    ("preference_reasoning", "actual device rather than an accessory"),
    ("ranking", "stated tradeoff priority"),
    ("ranking", "highest advertised maximum refresh rate"),
    ("recovery", "preferred model and color"),
    ("recovery", "another color of the same model"),
    ("justification", "trustworthy reason"),
    ("justification", "structured justification with claims"),
)

_CLARIFY = (
    "Before I start searching, could you clarify your firm requirements? Please confirm "
    "the must-have specs, the category, any required brand, your budget, and how you "
    "would prioritize the trade-offs between your preferences."
)
_ACK = (
    "Understood. I will keep every requirement you stated, including the budget, and "
    "continue with the purchase on that basis."
)

_SYSTEM = """You are an autonomous shopping agent operating a store through tools. A supervising controller checks every action you propose; when it rejects one, the tool result explains why - follow that instruction on your next turn.

PROTOCOL (follow it in this order; do not skip steps)
1. The shopper has already been asked to confirm the requirements; their reply is in the conversation. Treat the LATEST shopper statement of budget, category, brand, spec and priorities as binding.
2. Find candidates with `search` (k=10, always pass max_price = the budget so over-budget hits do not crowd out eligible rows; the same query with a lower max_price reaches further rows) and `filter` (category / brand / max_price). Never repeat a call with identical arguments - the result is already in the conversation. Budget your turns: commit to the best candidate by roughly turn 12 of 30; a market change after the first add needs 5-6 more turns to recover and order. Search matches titles: use concrete product words - the category word plus a brand, a model name or the spec value ("iPhone 13 smartphone", "27 inch 144Hz gaming monitor"). When results are accessories (cases, protectors, cables, stands), reformulate with model names of devices that carry the required spec instead of repeating the same words. If a spec value (e.g. "6.79 inch", "144Hz") finds nothing valid within budget, search brand or model families that commonly carry that spec (budget Android lines, monitor series, etc.) together with the value; the same spec is often written differently in titles (6.79", 6.79-inch, 6.79 HD+). Every search costs budget: a few distinct queries, then work with what you have.
3. Shortlist with `compare` (up to 5 product_ids in one call) and `view`. A turn may contain several tool calls: batch independent reads (several `view` calls, or `inspect_stock` for several variants) in ONE turn instead of one per turn - turns are limited. Read `category_path` (must match the requested category; reject accessories, cases, cables, replacement parts, screen-only items, review pages), `facts` and `variants[].options`. A product must be viewed or compared BEFORE it can be added to the cart.
4. Decide yourself. Never ask the shopper to choose between options and never ask them to raise the budget or drop a requirement. Send a `message` only to answer a direct question from the shopper.
5. `add_to_cart` exactly one variant: the best item that meets EVERY stated requirement, is in stock, and is within budget. The first item you add is graded as your real choice, so it must be the genuinely best valid option, never a placeholder.
6. After every cart change the controller re-checks the item's current price and stock and tells you the result. If the item became unavailable or repriced above budget, it is removed; then pick a DIFFERENT product that still meets every requirement (follow the task's own recovery rule, e.g. same model other colour before another model, or a different listing), view it, add it, and wait for the check.
7. When the controller reports the cart item verified, call `place_test_order` with that product_id and sku. Do not wait for permission. If the shopper asked you a question, reply with `message` first, then order.

READING PRODUCTS
- Specs come from `facts` (display, refresh_rate, size, model, color, battery, water resistance, device type / OS) and from the selected variant's `options`; a variant option overrides a product-wide value. Titles are marketing text: a title listing two models or two values (e.g. two refresh rates) is ambiguous - prefer a listing whose facts state one clear value.
- Resolution classes by pixels: HD >= 1280x720, Full HD >= 1920x1080, QHD >= 2560x1440, 4K/UHD >= 3840x2160 (highest class whose both thresholds are met; lowercase qHD is quarter HD).
- In-stock and price are per variant (sku). Use the current values from view/inspect_stock, not the search row.
- Smartphone tasks: the title or device type must say smartphone, or the OS must be a phone OS; category labels alone are not enough.

ORDERING
- `place_test_order` needs product_id and sku of the verified cart item. Include a `justification` ONLY when the shopper asked for a reason; then give claims as {"field", "value"} pairs only (no summary, no evidence): the requested specification field with the value exactly as the facts show it (e.g. "144Hz", "27 inch", "3840 x 2160" or "4K"), "price" with the exact current price, "in_stock" "true", and "brand" with the exact brand field. Otherwise send no justification at all.
"""

_FAMILY_HINTS = {
    "intent_decomposition": (
        "This task mixes firm requirements (category, the one named hardware spec, budget, in stock) with a "
        "SOFT brand preference. The FIRST item you add must be of the preferred brand whenever a fully compliant "
        "in-stock item of that brand exists (judge the brand by the listing's brand field, not the title); only "
        "if none exists buy a compliant item of another brand. Search the preferred brand + spec first. If the shopper later replaces a requirement, "
        "the new value is binding and everything else stays. Check the spec on the exact variant options."
    ),
    "retrieval_recall": (
        "Coverage matters: before ordering, surface several DISTINCT matching products (different product_id "
        "and model) whose lowercase title contains every requested term as a whole word. Use at most 3 distinct "
        "search queries, k=10, in_stock true: the exact phrase first, then reformulations that reach OTHER "
        "matching products (phrase + a product-type word such as serum/toner/cream, or the words reordered). "
        "View or compare ONLY products whose title contains every requested term as a whole word (that keeps "
        "precision high) and whose category_path carries the requested category; a term glued to other characters "
        "(e.g. 'PlayStation5' for 'playstation') does not count. Then add the best in-stock matching item within budget."
    ),
    "constraint_satisfaction": (
        "The spec must hold on the EXACT variant: read variants[].options (RAM, screen size, resolution, "
        "refresh rate) and add the sku whose option states the required value. If that sku sells out, choose a "
        "DIFFERENT product that meets every requirement - never another sku of the same listing that lacks the "
        "spec. If the request does not state the spec value, the shopper's clarification reply does. The "
        "controller verifies the spec of every variant you add from its option/facts/title: a listing whose "
        "facts and title disagree or list several values cannot be ordered, so prefer listings that state the "
        "value once and clearly (e.g. size '6.1 inches', title '6.1 inch')."
    ),
    "preference_reasoning": (
        "Buy an actual device (not an accessory or review page) that best fits the shopper's use case. Inspect "
        "or compare at least two eligible products with DIFFERENT listing_id values and prefer the one whose "
        "facts explicitly state the preferred features (e.g. GPS/navigation, IP or water resistance rating, "
        "battery hours or days, rugged/drop-proof/kids protection, Wi-Fi, sport fit) - not the cheapest valid "
        "item. Your FIRST cart add is graded on preference fit and so is the final order: pick the candidate whose "
        "facts or title EXPLICITLY state evidence for as many of the shopper's stated preference axes as possible "
        "(e.g. a water/sweat rating, a sport/secure-fit design, a battery duration in hours, wireless) - all of "
        "them if one exists - never a cheap item covering one axis, and never a non-headphone product. If your first pick is repriced above budget, recover to a product with a "
        "different listing_id that also covers the preferences."
    ),
    "ranking": (
        "Rank strictly by the stated priority: highest advertised maximum refresh rate (an explicitly advertised "
        "overclock counts), then largest screen size, then lowest price. Read refresh and size from facts and "
        "the selected variant's options; never pick an item that is worse than another available one on all "
        "three axes. If the request lists candidate variants, consider ONLY those (no searching). Compare the "
        "candidates, add the true #1, and when the controller reports it gone, re-rank the remaining live options "
        "by the same rule and add the new #1."
    ),
    "recovery": (
        "Identity rules: a listing's MODEL is its `facts.model` value (compare it to the requested model code "
        "ignoring spaces/punctuation/case; a conflicting title does not override a model fact; without a model "
        "fact use an unambiguous model code in the title). Its COLOUR is the selected variant's `options` first, "
        "else a colour fact, else the title; if that source names several different colours, or none, the colour "
        "is UNRESOLVED and the listing cannot be ordered. First add the exact preferred model AND colour. When it "
        "disappears: prefer the same model (model fact) in another resolvable colour; only if none exists, a "
        "different model in the preferred colour; then any other resolvable listing. Search the model code itself "
        "so you know the sibling colours before committing. Keep budget and in-stock firm. Over-ear/on-ear headphones only when "
        "headphones are requested (earbuds and in-ear are not headphones). Before committing, search the model "
        "code/name itself and view the listings that carry it, so that after a stockout you already know the "
        "same model in other colours (check every listing's variants for colour options)."
    ),
    "justification": (
        "Order with a justification: claims only for the ONE specification field named in the request "
        "(resolution, refresh_rate or screen_size - value exactly as the facts state it), 'price' (exact "
        "current price), 'in_stock' 'true', and 'brand' (the exact brand field). No other spec fields. Pick a "
        "product whose facts AND title state a single unambiguous value for that spec (a title listing several "
        "sizes or several refresh rates cannot support a claim)."
    ),
}


# ---- specification resolver ----------------------------------------------
# Implements the public reading rules the tasks state: a selected variant's option
# value comes first; otherwise the product facts and title must agree on ONE value.
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*hz", re.I)
_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-\u2013]?\s*(?:(?:inch(?:es)?|in)\b|''|\"|\u201c|\u201d)", re.I)
_RAM_PAIR_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_RAM_TITLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x\u00d7]\s*(\d{3,5})")
_CAPACITY_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(ml|millilit(?:er|re)s?|l|lit(?:er|re)s?)(?!\w)", re.I)
_WATTAGE_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(kw|kilowatts?|w|watts?)(?!\w)", re.I)
_WARRANTY_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(years?|yrs?|months?|mos?)(?!\w)", re.I)
_QUANTITY_RE = re.compile(r"(?<![\w.])(\d+)\s*(?:-\s*)?(?:cts?|pcs?|pieces?|count|units?|pack(?!s))(?!\w)", re.I)
_NO_WARRANTY_RE = re.compile(r"\b(?:no|without)\s+warranty\b", re.I)
# Solver-visible fact group carrying each gated spec.
_SPEC_FACT_LABEL = {
    "resolution": "display",
    "refresh_rate": "refresh_rate",
    "screen_size": "size",
    "capacity": "capacity",
    "wattage": "wattage",
    "warranty_duration": "warranty_duration",
    "pack_quantity": "pack_quantity",
    "ram": "storage",
}


def _res_class(a: int, b: int) -> str | None:
    long_axis, short_axis = max(a, b), min(a, b)
    if long_axis >= 3840 and short_axis >= 2160:
        return "4k"
    if long_axis >= 2560 and short_axis >= 1440:
        return "qhd"
    if long_axis >= 1920 and short_axis >= 1080:
        return "fhd"
    if long_axis >= 1280 and short_axis >= 720:
        return "hd"
    return None


def _res_values(text: str | None) -> set[str]:
    if not text:
        return set()
    if re.search(r"\bqHD\b", text):
        return {"conflict"}
    low = text.lower()
    values = set()
    for m in _WXH_RE.finditer(low):
        c = _res_class(int(m.group(1)), int(m.group(2)))
        if c:
            values.add(c)
    if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
        values.add("4k")
    if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
        values.add("qhd")
    if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
        values.add("fhd")
    residual = re.sub(r"\b(?:full|ultra) hd\b", "", low)
    if re.search(r"\b(?:hd|720p?)\b", residual):
        values.add("hd")
    return values


def _scaled_values(text: str, pattern: "re.Pattern[str]", large: set[str], suffix: str) -> set[str]:
    out = set()
    for number, unit in pattern.findall(text):
        value = float(number) * (1000 if unit.lower() in large else 1)
        if value > 0:
            out.add(f"{value:g}{suffix}")
    return out


def _warranty_values(text: str) -> set[str]:
    return {
        f"{float(n) * (12 if u.lower().startswith('y') else 1):g}mo"
        for n, u in _WARRANTY_RE.findall(text)
    }


def _warranty_context_values(text: str) -> set[str]:
    """Durations a title or option explicitly describes as warranty terms."""
    out: set[str] = set()
    for m in _WARRANTY_RE.finditer(text):
        window = text[max(0, m.start() - 24): m.end() + 24]
        if re.search(r"\bwarrant(?:y|ies)\b", window, re.I):
            out |= _warranty_values(m.group(0))
    return out


def _spec_values(key: str, text: str | None, *, free_text: bool = False) -> set[str]:
    """Canonical values of one spec in a text. free_text marks a title or options string,
    where a duration counts as a warranty only next to the word 'warranty'."""
    if not text:
        return set()
    if key == "capacity":
        return _scaled_values(text, _CAPACITY_RE, {"l", "liter", "liters", "litre", "litres"}, "ml")
    if key == "wattage":
        return _scaled_values(text, _WATTAGE_RE, {"kw", "kilowatt", "kilowatts"}, "w")
    if key == "warranty_duration":
        return _warranty_context_values(text) if free_text else _warranty_values(text)
    if key == "pack_quantity":
        return {f"{int(n)}ct" for n in _QUANTITY_RE.findall(text)}
    if key == "refresh_rate":
        return {f"{float(m.group(1)):g}hz" for m in _HZ_RE.finditer(text)}
    if key == "screen_size":
        return {f"{float(m.group(1)):g}in" for m in _INCH_RE.finditer(text)}
    if key == "ram":
        pairs = {f"{float(a):g}gb" for a, _b in _RAM_PAIR_RE.findall(text)}
        return pairs or {f"{float(n):g}gb" for n in _RAM_TITLE_RE.findall(text)}
    if key == "resolution":
        return _res_values(text)
    return set()


def _resolve_spec(key: str, options: str | None, facts: dict | None, title: str | None) -> str | None:
    """Canonical value of one spec for a variant, or None when absent or ambiguous."""
    if key == "warranty_duration" and isinstance(facts, dict):
        warranty_text = " ".join(str(facts.get(f) or "") for f in ("warranty_duration", "warranty_type"))
        if _NO_WARRANTY_RE.search(warranty_text):
            return None
    from_options = _spec_values(key, options or "", free_text=True)
    if len(from_options) > 1:
        return None
    if from_options:
        return next(iter(from_options))
    values: set[str] = set()
    label = _SPEC_FACT_LABEL.get(key)
    if label and isinstance(facts, dict) and facts.get(label):
        values |= _spec_values(key, str(facts[label]))
    values |= _spec_values(key, title or "", free_text=True)
    if "conflict" in values or len(values) != 1:
        return None
    return next(iter(values))


def _wanted_spec(text: str, *, loose: bool = False) -> tuple[str, str] | None:
    """The firm spec a shopper statement names, in canonical form."""
    low = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?\s+screen", low)
    if m:
        return "screen_size", f"{float(m.group(1)):g}in"
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*hz\s+refresh rate", low)
    if m:
        return "refresh_rate", f"{float(m.group(1)):g}hz"
    m = re.search(r"(\d+(?:\.\d+)?)\s*gb\s+of\s+ram", low)
    if m:
        return "ram", f"{float(m.group(1)):g}gb"
    m = re.search(r"\b(4k|qhd|1080p full hd|720p hd|full hd|1440p|2160p|uhd)\s+display", low)
    if m:
        return "resolution", next(iter(_res_values(m.group(1))))
    # Runtime 0.2.18 gate attributes. The shopper says "a 500 mL capacity", "1200 W power",
    # "a 12-month warranty", "a 6-piece pack" (or asks for a pack of 6).
    for key, near in (
        ("capacity", r"capacit(?:y|ies)|volume"),
        ("wattage", r"power|wattage|motor"),
        ("warranty_duration", r"warrant(?:y|ies)"),
        ("pack_quantity", r"pack|pieces|count|bundle|set"),
    ):
        for m in re.finditer(r"[\d.]+\s*-?\s*[a-z]+", low):
            window = low[max(0, m.start() - 30): m.end() + 30]
            if not re.search(near, window):
                continue
            # the shopper writes "12-month"/"6-piece"; the catalog spells the unit apart
            vals = _spec_values(key, re.sub(r"(\d)\s*-\s*([a-z])", r"\g<1> \g<2>", m.group(0)))
            if len(vals) == 1:
                return key, next(iter(vals))
    m = re.search(r"pack of (\d+)\b", low)
    if m:
        return "pack_quantity", f"{int(m.group(1))}ct"
    if loose:
        m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(?:inch(?:es)?\b|\")", low)
        if m and not re.search(r"\bin\s+stock", m.group(0)):
            return "screen_size", f"{float(m.group(1)):g}in"
        m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*hz\b", low)
        if m:
            return "refresh_rate", f"{float(m.group(1)):g}hz"
        m = re.search(r"\b(4k|uhd|qhd|1080p|1440p|2160p|720p|full hd|fhd)\b", low)
        if m:
            vals = _res_values(m.group(1))
            if len(vals) == 1:
                return "resolution", next(iter(vals))
        for key in ("capacity", "wattage", "warranty_duration", "pack_quantity"):
            vals = _spec_values(key, low)
            if len(vals) == 1:
                return key, next(iter(vals))
    return None


_COLOR_WORDS = (
    "black", "white", "blue", "red", "green", "gray", "grey", "silver", "gold", "golden", "pink", "purple",
    "violet", "yellow", "orange", "brown", "beige", "khaki", "navy", "teal", "cyan", "rose", "graphite",
    "titanium", "sage", "mint", "lavender", "coral", "peach", "ivory", "cream", "bronze", "copper", "starlight",
    "midnight", "space", "sand", "olive", "maroon", "burgundy", "turquoise", "indigo", "charcoal", "transparent",
)
_COLOR_RE = re.compile(r"\b(" + "|".join(_COLOR_WORDS) + r")\b", re.I)


_BASE_COLORS = frozenset(
    "black white blue red green gray grey silver gold pink purple violet yellow orange brown beige khaki navy "
    "teal cyan maroon burgundy turquoise indigo olive cream ivory coral peach lavender mint".split()
)
_ALIAS_COLORS = {"grey": "gray", "golden": "gold", "violet": "purple"}


def _colors_in(text: str | None) -> set[str]:
    """Distinct base colours named in a text. A shade ("moonlight white", "sage green") counts as
    its base colour; standalone shade words (graphite, starlight, titanium...) count only when no
    base colour is present. Two different base colours = ambiguous (caller treats as unresolved)."""
    words = [w.lower() for w in _COLOR_RE.findall(text or "")]
    base = {_ALIAS_COLORS.get(w, w) for w in words if w in _BASE_COLORS}
    if base:
        return base
    return set(words)


def _norm_model(text: str | None) -> str:
    return "".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


_GOAL_STOP = frozenset(
    "i need a an the and or of for with in on at to from is are be this that my me we you it its want "
    "looking find buy one please can spend up under below at or below most stock in-stock php usd firm "
    "requirements requirement prefer ideally but matter more than category listing matches phrase "
    "screen inch inches display budget".split()
)

_ACCESSORY_RE = re.compile(
    r"\b(case|cases|cover|covers|protector|protectors|tempered|glass film|screen film|skin|sticker|"
    r"cable|charger|adapter|holder|stand|mount|strap|band|pouch|bag|lens film|replacement screen|"
    r"screen replacement|lcd|digitizer|battery replacement|flex|housing|back glass|sim tray|earbuds|"
    r"earphone|earphones|in-ear|in ear)\b",
    re.I,
)
# device_semantics._OS: an installed phone OS is positive smartphone evidence.
_PHONE_OS_RE = re.compile(
    r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian|harmony\s*os|hyper\s*os|"
    r"sailfish|ubuntu\s*touch|tizen)\b", re.I
)
# device_semantics._CONFLICTS: any of these in the title/type facts means "not a smartphone".
_PHONE_CONFLICT_RE = re.compile(
    r"\btablets?\b|\b(?:feature|basic)\s*phones?\b|\b(?:mystery|blind|lucky)\s*box(?:es)?\b|"
    r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
    r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b", re.I
)
_PHONE_NEGATION_RE = re.compile(
    r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b", re.I
)
_PHONE_BUNDLE_SPLIT_RE = re.compile(
    r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s*protector|charger|accessor(?:y|ies))\b))", re.I
)
# acceptance._CATEGORY_ALIASES: label -> canonical concept; two labels match when they share one.
_CATEGORY_ALIASES = {
    "video cards": "graphics_card", "graphics cards": "graphics_card", "video graphics cards": "graphics_card",
    "gpu": "graphics_card", "gpus": "graphics_card",
    "mobile phones": "smartphone", "cell phones": "smartphone", "cellphones": "smartphone", "cellphone": "smartphone",
    "phones": "smartphone", "smartphones": "smartphone", "unlocked phones": "smartphone", "smart phones": "smartphone",
    "wi fi": "networking", "wifi": "networking",
    "smartwatches": "smartwatch", "smart watches": "smartwatch",
    "headphones headsets": "headphones",
}


def _category_labels(labels) -> set[str]:
    out: set[str] = set()
    for label in labels:
        low = " ".join(re.findall(r"[a-z0-9]+", str(label or "").lower()))
        if low:
            out.add(low)
            out.add(_CATEGORY_ALIASES.get(low, low))
    return out


def _category_matches(expected: str | None, path: list[str], title: str | None) -> bool:
    """acceptance.category_matches: canonical-label overlap, plus its two title fallbacks."""
    if not expected:
        return True
    if _category_labels([expected]) & _category_labels(path):
        return True
    exp = " ".join(re.findall(r"[a-z0-9]+", expected.lower()))
    toks = _title_tokens(title)
    stems = toks | {t[:-1] for t in toks if len(t) > 3 and t.endswith("s")}
    if exp in ("gaming monitors", "gaming monitor"):
        high = any(float(m.group(1)) >= 120 for m in re.finditer(r"(\d{2,3}(?:\.\d+)?)\s*-?\s*hz", title or "", re.I))
        return bool(_category_labels(["Monitors", "Computer Monitors"]) & _category_labels(path)) and (
            {"gaming", "monitor"} <= stems or ("monitor" in stems and high)
        )
    if exp in ("over ear", "over ear headphones"):
        return bool(_category_labels(["Headphones"]) & _category_labels(path)) and (
            {"over", "ear"} <= stems or "overear" in stems or "circumaural" in stems
        )
    return False


# families/recovery.py admission rules for the recovery family's two category profiles.
_RECOVERY_INVALID_PREFIXES = ("customer ratings", "customer reviews")
_RECOVERY_INVALID_FRAGMENTS = ("join bild expo", "earbud", "in-ear", "in ear", "earphone", "open-ear", "true wireless")
_RECOVERY_ACCESSORY_RE = re.compile(
    r"\b(?:case|cases|cover|covers|sleeve|sleeves|skin|skins|pouch|protector|cable|cables|"
    r"strap|straps|ear ?tips?|ear ?hooks?|adapter|holder|stand|replacement|carabiner)\b"
)
_OVER_OR_ON_EAR_RE = re.compile(r"\b(?:over|on)[-\s]?ear\b", re.I)
_FEATURE_PHONE_RE = re.compile(r"\b(?:feature|basic|dumb|button|keypad) ?phones?\b")


def _recovery_admissible(category: str | None, product: dict, options: str | None) -> bool:
    """Whether the recovery verifier would even list this variant as a candidate."""
    title = str(product.get("title") or "")
    low = title.lower().strip()
    if not low or low.startswith(_RECOVERY_INVALID_PREFIXES) or _RECOVERY_ACCESSORY_RE.search(low):
        return False
    labels = _category_labels([category]) if category else set()
    if "headphones" in labels:
        if any(fragment in low for fragment in _RECOVERY_INVALID_FRAGMENTS):
            return False
        return _OVER_OR_ON_EAR_RE.search(f"{title} {options or ''}") is not None
    if "smartphone" in labels:
        if _FEATURE_PHONE_RE.search(low):
            return False
        return _smartphone_evidence(product, options) == "supported"
    return True


def _smartphone_evidence(product: dict, options: str | None = None) -> str:
    """device_semantics.smartphone_evidence over a view payload: supported / conflict / unresolved."""
    selected = str(options or "")
    if re.search(r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
                 r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b", selected, re.I):
        return "conflict"
    facts = product.get("facts") or {}
    raw_title = str(product.get("title") or "")
    title = _PHONE_BUNDLE_SPLIT_RE.split(_PHONE_NEGATION_RE.sub("", raw_title), maxsplit=1)[0]
    type_text = " ".join(_PHONE_NEGATION_RE.sub("", str(facts.get(k) or "")) for k in ("phone_type", "device_type", "description_device_type"))
    text = f"{title} {type_text}"
    if _PHONE_CONFLICT_RE.search(text):
        return "conflict"
    if any("[truncated]" in str(facts.get(k) or "") for k in ("phone_type", "device_type", "device_operating_system")):
        return "unresolved"
    if re.search(r"\bsmart\s*phones?\b", text, re.I):
        return "supported"
    os_text = " ".join(_PHONE_NEGATION_RE.sub("", str(facts.get(k) or "")) for k in ("device_operating_system", "description_operating_system")
                       if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I))
    title_os = _PHONE_OS_RE.search(title) and not re.search(r"\bcompatib", raw_title, re.I)
    if _PHONE_OS_RE.search(os_text) or title_os:
        return "supported"
    return "unresolved"


def _required_category(goal: str, sealed: str | None) -> str | None:
    if sealed:
        m = re.search(r"required catalog category is ([^.]+)\.", sealed, re.I)
        if m:
            return m.group(1).strip()
    m = re.search(r"in the ([A-Za-z&' ,-]+?) category", goal)
    return m.group(1).strip() if m else None


def _title_tokens(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _requested_phrase_text(goal: str) -> str | None:
    """The phrase a retrieval request asks to match, as written."""
    m = (
        re.search(r"matches\s+(?:the phrase\s+)?[\"\u201c']([^\"\u201d']+)[\"\u201d']", goal, re.I)
        or re.search(r"[\"\u201c]([^\"\u201d]{2,60})[\"\u201d]", goal)
        or re.search(r"matches\s+(?:the phrase\s+)?(.+?)(?:,|\s+at\s+or\s+under\b|\s+and\s+buy\b|$)", goal, re.I)
    )
    return m.group(1).strip() if m else None


def _requested_phrase(goal: str) -> set[str]:
    """Tokens of the quoted phrase a retrieval request asks to match."""
    text = _requested_phrase_text(goal)
    return _title_tokens(text) if text else set()


# The verifier's three TF4 alignment profiles, mirrored: per axis (weight, full-credit regex, half-credit
# regex or None, evidence source). Battery axes use the runtime's duration tiers: "days" profiles
# (smartwatch: >=10d 1, >=5d .75, >=2d .5, hours >=120/72/36, solar/battery life .5) and "hours"
# profiles (headphones/tablets: any days 1, hours >=15/10/6, "battery life" .25).
_TF4_PROFILES: dict[str, dict] = {
    "smartwatch": {
        "text": "facts",
        "axes": [
            ("navigation", 0.32, r"\b(gps|gnss|maps?|compass|barometer|altitude|tactical|trail|hiking|outdoor)\b", None, "facts"),
            ("rugged_water_resistance", 0.26,
             r"\b(10\s*atm|5\s*atm|50m|100m|ip68|ip69|mil-std|waterproof|water resistant|sapphire|fiber[- ]reinforced|titanium|rugged|tactical|tank)\b",
             r"\b(water|swim|steel|aluminum)\b", "facts"),
            ("battery_life", 0.26, "days", None, "facts"),
            ("outdoor_watch_credibility", 0.16, "brand:garmin|amazfit|fitbit|kospet", None, "brand"),
        ],
        "penalty": (0.2, r"\b(kid|kids|child|children|myfirst|watch phone)\b"),
        "match": {"navigation": r"gps|navig|map|compass|off.?grid", "rugged_water_resistance": r"rugged|water|drop|rough|durab|resist",
                  "battery_life": r"battery", "outdoor_watch_credibility": r"brand|credib|fitness"},
        "query": {"navigation": ["GPS", "compass", "hiking", "outdoor"], "rugged_water_resistance": ["waterproof", "5ATM", "IP68", "rugged"],
                  "battery_life": ["days battery", "long battery life"], "outdoor_watch_credibility": ["Garmin", "Amazfit"]},
        "device": ["smartwatch", "smart watch", "watch"],
    },
    "headphones": {
        "text": "facts+title",
        "axes": [
            ("sweat_water_resistance", 0.30, r"\b(ipx?[4-8]|ip5[0-9]|ip6[0-9]|sweat ?(?:proof|resistant)|water ?(?:proof|resistant))\b", None, "facts+title"),
            ("sport_secure_fit", 0.28, r"\b(sports?|running|gym|workout|fitness|ear ?hook|secure fit)\b", None, "facts+title"),
            ("battery_life", 0.22, "hours", None, "facts+title"),
            ("wireless_freedom", 0.20, r"\b(wireless|bluetooth|true wireless)\b", None, "facts+category+title"),
        ],
        "penalty": None,
        "match": {"sweat_water_resistance": r"sweat|water|resist", "sport_secure_fit": r"sport|secure|fit\b|running|workout|gym",
                  "battery_life": r"battery", "wireless_freedom": r"wireless|cable|bluetooth"},
        "query": {"sweat_water_resistance": ["IPX7", "waterproof", "sweatproof", "IPX5"], "sport_secure_fit": ["sports", "running", "gym", "ear hook"],
                  "battery_life": ["hours", "playtime", "battery"], "wireless_freedom": ["wireless", "bluetooth"]},
        "device": ["headphones", "earphones", "earbuds", "headset"],
    },
    "tablet": {
        "text": "facts+title",
        "axes": [
            ("kid_ready_durability", 0.40, r"\b(kids?|child|children|drop ?proof|rugged case|bumper)\b", None, "facts+title"),
            ("battery_life", 0.25, "hours", None, "facts+title"),
            ("kid_friendly_size", 0.20, r"\b(8|10|10\.\d|11)\s*(?:inch|in\b|\")", None, "facts+title"),
            ("wifi_ready", 0.15, r"\b(wi-?fi|wifi)\b", None, "facts+title"),
        ],
        "penalty": None,
        "match": {"kid_ready_durability": r"kid|child|durab|drop|rugged|survive|protect", "battery_life": r"battery",
                  "kid_friendly_size": r"size|inch|screen", "wifi_ready": r"wi.?fi|download|stream"},
        "query": {"kid_ready_durability": ["kids", "drop-proof", "bumper case", "rugged"], "battery_life": ["hours", "battery"],
                  "kid_friendly_size": ["10 inch", "8 inch"], "wifi_ready": ["WiFi"]},
        "device": ["tablet", "ipad", "galaxy tab", "kids tablet"],
    },
}
_TF4_PROFILE_ALIASES = {"smartwatches": "smartwatch", "smart watches": "smartwatch", "smartwatch": "smartwatch",
                        "headphones": "headphones", "headphones headsets": "headphones", "headphone": "headphones",
                        "tablets": "tablet", "tablet": "tablet"}


def _tf4_profile(category: str | None) -> dict | None:
    low = " ".join(re.findall(r"[a-z0-9]+", str(category or "").lower()))
    key = _TF4_PROFILE_ALIASES.get(low)
    return _TF4_PROFILES.get(key) if key else None


def _duration_days(text: str) -> float:
    days = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*days?", text, re.I)]
    hours = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*hours?", text, re.I)]
    if days:
        best = max(days)
        if best >= 10:
            return 1.0
        if best >= 5:
            return 0.75
        if best >= 2:
            return 0.5
    if hours:
        best = max(hours)
        if best >= 120:
            return 1.0
        if best >= 72:
            return 0.75
        if best >= 36:
            return 0.5
    if "solar" in text or "battery life" in text:
        return 0.5
    return 0.0


def _duration_hours(text: str) -> float:
    hours = [float(m.group(1)) for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*\+?\s*(?:hours?|hrs?|h)\b", text, re.I)]
    if re.search(r"\d+(?:\.\d+)?\s*days?", text, re.I):
        return 1.0
    if hours:
        best = max(hours)
        if best >= 15:
            return 1.0
        if best >= 10:
            return 0.75
        if best >= 6:
            return 0.5
    if "battery life" in text:
        return 0.25
    return 0.0


def _tf4_alignment(profile: dict, product: dict) -> tuple[float, dict[str, float]]:
    """The verifier's alignment score for one viewed product under a profile (0..1)."""
    facts = product.get("facts") or {}
    facts_text = " ".join(f"{k.replace('_', ' ')} {v}" for k, v in facts.items()).lower()
    title = str(product.get("title") or "").lower()
    category_text = " ".join(str(x) for x in (product.get("category_path") or [])).lower()
    brand = str(product.get("brand") or "").lower()
    sources = {
        "facts": facts_text,
        "facts+title": facts_text + " " + title,
        "facts+category+title": facts_text + " " + category_text + " " + title,
    }
    axes: dict[str, float] = {}
    raw = 0.0
    for name, weight, full, half, source in profile["axes"]:
        text = sources[source] if source in sources else facts_text
        if full == "days":
            v = _duration_days(text)
        elif full == "hours":
            v = _duration_hours(text)
        elif full.startswith("brand:"):
            v = 1.0 if brand in set(full[6:].split("|")) else 0.0
        elif re.search(full, text, re.I):
            v = 1.0
        elif half and re.search(half, text, re.I):
            v = 0.5
        else:
            v = 0.0
        axes[name] = v
        raw += weight * v
    if profile.get("penalty"):
        pen, pat = profile["penalty"]
        if re.search(pat, facts_text, re.I):
            raw -= pen
    return (round(max(0.0, min(1.0, raw)), 3), axes)


# Preference axes -> the words listings use for them (query vocabulary, keyed by the shopper's words).
_AXIS_QUERY_WORDS: list[tuple[str, list[str]]] = [
    (r"sweat|water|waterproof|resist", ["IPX7", "waterproof", "sweatproof", "IPX5"]),
    (r"sport|secure|fit\b|running|workout|gym", ["sports", "running", "gym", "ear hook"]),
    (r"battery", ["hours", "playtime", "battery"]),
    (r"wireless|cable|bluetooth", ["wireless", "bluetooth"]),
    (r"gps|navigation", ["GPS", "navigation"]),
    (r"rugged|durab|drop|shock", ["rugged", "shockproof", "drop-proof"]),
    (r"\bkids?\b|child", ["kids"]),
    (r"wifi|wi-fi", ["WiFi"]),
    (r"size|screen|display", ["inch display"]),
    (r"\b(noise|anc)\b", ["noise cancelling", "ANC"]),
    (r"comfort|cushion", ["comfortable", "soft"]),
]
# Category label -> the product words titles use instead of the label.
_CATEGORY_QUERY_WORDS = {
    "headphones": ["earphones", "earbuds", "headset"],
    "smartwatches": ["smart watch", "watch"],
    "tablets": ["tablet"],
    "smartphones": ["phone"],
    "laptops": ["laptop", "notebook"],
}

# Title words that never make a useful query reformulation.
_COTOKEN_STOP = {
    "the", "and", "for", "with", "of", "in", "to", "by", "on", "or", "a", "an", "new", "original", "free",
    "sale", "hot", "set", "pcs", "pc", "ml", "g", "kg", "mm", "cm", "oz", "inch", "inches", "pack", "per",
    "from", "your", "you", "our", "all", "at", "is", "it", "its", "as", "up", "off", "only", "best", "high",
    "quality", "authentic", "brand", "type", "style", "color", "colour", "size", "premium", "promo", "cod",
}



def _spec_query_words(spec: tuple[str, str] | None) -> str:
    """How a listing spells the gated value, for use inside a search query."""
    if not spec:
        return ""
    key, value = spec
    number = re.sub(r"[a-z]+$", "", value)
    return {
        "screen_size": f"{number} inch",
        "refresh_rate": f"{number}Hz",
        "ram": f"{number}GB",
        "resolution": value.upper(),
        "capacity": f"{number} ml",
        "wattage": f"{number}W",
        "warranty_duration": f"{number} months",
        "pack_quantity": f"{number} pcs",
    }.get(key, "")


def _candidate_keys_in_query(text: str) -> list[tuple[str, str]]:
    return [(a, b) for a, b in re.findall(r"(\d{6,})::([A-Za-z0-9_-]+)", text)]


def _model() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(provider, _DEFAULT_MODELS["chutes"])


def _model_candidates() -> list[str]:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    out = [_model()]
    for m in _FALLBACK_MODELS.get(provider, _FALLBACK_MODELS["chutes"]):
        if m not in out:
            out.append(m)
    return out


def _family(query: str, tools: list[dict]) -> str:
    for tool in tools or []:
        fn = tool.get("function") or {}
        if fn.get("name") == "place_test_order":
            required = (fn.get("parameters") or {}).get("required") or []
            if "justification" in required:
                return "justification"
    for name, marker in _FAMILY_MARKERS:
        if marker in query:
            return name
    return "unknown"


def _parse_amount(text: str) -> tuple[float, str] | None:
    m = _AMOUNT_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "")), m.group(2).upper()
    except ValueError:
        return None


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _Session:
    def __init__(self, problem_data: dict[str, Any]):
        env = problem_data["environment"]
        self.binding = env["binding"]
        self.policy = env["policy_view"]
        self.problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        self.max_steps = int(self.policy.get("max_steps", 30))
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn", 1)))
        self.tools = self.policy.get("tools") or []
        self.query = str(self.policy.get("query") or "")
        self.family = _family(self.query, self.tools)
        self.started = time.monotonic()
        self.turn = 0
        self.done = False
        self.dialogue: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        # knowledge
        self.rows: dict[tuple[str, str], dict] = {}        # (pid, sku) -> latest row facts
        self.products: dict[str, dict] = {}                # pid -> view/compare payload
        self.viewed: set[str] = set()
        self.listing: dict[str, str] = {}                  # pid -> listing_id (TF4)
        self.cart: list[tuple[str, str]] = []
        self.dead: set[tuple[str, str]] = set()
        self.verified: tuple[str, str] | None = None       # cart item checked after its add
        self.verified_price: float | None = None
        self.pending_add: tuple[str, str] | None = None
        self.searches: set[str] = set()
        self.cat_pool: set[str] = set()      # product ids a category filter returned
        self.messages_sent = 0
        self.pending_reply: str | None = None
        self.strikes = 0
        self.recoveries = 0
        amount = _parse_amount(self.query)
        self.budget: float | None = amount[0] if amount else None
        self.orig_budget: float | None = amount[0] if amount else None
        self.currency: str | None = amount[1] if amount else None
        self.event_seen = False
        self.first_add_done = False
        self.dead_listing: set[str] = set()
        goal = self.query.split("\n\nTask rules:", 1)[0]
        self.wanted = _wanted_spec(goal) or (
            _wanted_spec(goal, loose=True) if self.family == "justification" else None
        )
        self.explicit_candidates = _candidate_keys_in_query(goal)
        self.phrase = _requested_phrase(goal) if self.family == "retrieval_recall" else set()
        self.phrase_text = _requested_phrase_text(goal) if self.family == "retrieval_recall" else None
        self.category = _required_category(goal, None)
        self.cat_rejects: dict[Any, int] = {}
        self.spec_rejects: dict[tuple[str, str], int] = {}
        self.rank_rejects: dict[tuple[str, str], int] = {}
        self.read_history: set[str] = set()          # normalized read calls already executed
        self.dup_rejects = 0
        self.llm_failures = 0
        self.llm_dead = False
        self.model_index = 0
        self.goal_tokens = _title_tokens(goal) - _GOAL_STOP
        self.brand: str | None = None
        self.forced_commits = 0
        mp = re.search(r"\bprefer(?:ably)?\s+([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)?)", goal)
        self.pref_brand: str | None = mp.group(1).strip() if mp and self.family == "intent_decomposition" else None
        self.sealed: str | None = None
        self.audit: dict[str, dict] = {}       # pid -> {"axes": n, "evidenced": k, "detail": str}
        self.audit_seen: set[str] = set()
        self.audit_fail = 0
        self.pref_axes: list[dict] | None = None   # [{"name", "terms": set, "kind"}] in priority order
        self.pref_axes_tried = False
        self.fallback_rejected: set[str] = set()
        self.pref_color: str | None = None
        self.pref_model: str | None = None
        mr = re.search(r"prefer the ([a-z]+) ([\w][\w.+-]*)", goal, re.I)
        if mr and self.family == "recovery":
            self.pref_color, self.pref_model = mr.group(1).lower(), mr.group(2).rstrip(".,;:")
        self.recovery_rejects = 0
        self.pool_widened = False

    # ----------------------------------------------------------------- helpers
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def turns_left(self) -> int:
        return self.max_steps - self.turn

    def search_cap(self) -> int:
        if self.family == "preference_reasoning":
            return _SEARCH_CAP_TF4
        return _SEARCH_CAP.get(self.family, _SEARCH_CAP_DEFAULT)

    def note(self, text: str) -> None:
        self.messages.append({"role": "system", "content": "CONTROLLER: " + text})

    def price_ok(self, price: float | None, *, original: bool = False) -> bool:
        if price is None:
            return False
        budget = self.orig_budget if original else self.budget
        return budget is None or price <= budget + 1e-6

    # ------------------------------------------------------------- environment
    @staticmethod
    def result_rows(result: dict | None, index: int) -> list[str]:
        """Product ids the index-th call of a turn returned, in rank order."""
        calls = (result or {}).get("calls") or []
        if index >= len(calls):
            return []
        obs = calls[index].get("observation") or {}
        inner = obs.get("observation") if isinstance(obs.get("observation"), dict) else obs
        rows = (inner.get("results") if isinstance(inner, dict) else None) or []
        out = [str(r["product_id"]) for r in rows if isinstance(r, dict) and r.get("product_id")]
        return list(dict.fromkeys(out))

    def call(self, actions: list[dict[str, Any]], tool_calls: list[dict] | None = None) -> dict | None:
        """Execute one solver turn. Records tool results in the model context."""
        if self.done or not actions:
            return None
        self.turn += 1
        group = f"{self.problem_id}-turn-{self.turn}"
        if tool_calls is None:
            tool_calls = [
                {
                    "id": f"ctl-{self.turn}-{i}",
                    "type": "function",
                    "function": {"name": a["name"], "arguments": json.dumps(a["args"])},
                }
                for i, a in enumerate(actions, start=1)
            ]
            self.messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": self.turn,
            "calls": [
                {"call_id": f"{group}-{i}", "action": {"name": a["name"], "args": a["args"]}}
                for i, a in enumerate(actions, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        self.dialogue.append(
            {"role": "assistant", "content": "", "tool_calls": tool_calls, "environment_result": result}
        )
        calls = result.get("calls") or []
        for action, tc, cr in zip(actions, tool_calls, calls):
            obs = cr.get("observation") or {}
            self.messages.append(
                {"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(obs)}
            )
            # The validator wraps each result as {observation, done, error}.
            inner = obs.get("observation") if isinstance(obs.get("observation"), dict) else obs
            self.absorb(action, inner)
            if obs.get("done") or (isinstance(inner, dict) and inner.get("status") == "placed"):
                self.done = True
        um = result.get("user_message") or {}
        content = str(um.get("content") or "").strip() if isinstance(um, dict) else ""
        if content:
            self.messages.append({"role": "user", "content": content})
            self.dialogue.append({"role": "user", "content": content})
            self.absorb_user(content)
        return result

    def absorb(self, action: dict, obs: dict) -> None:
        name = action["name"]
        args = action.get("args") or {}
        if not isinstance(obs, dict):
            return
        if name in ("search", "filter"):
            in_category = name == "filter" and bool(args.get("category"))
            for row in obs.get("results") or []:
                if isinstance(row, dict) and row.get("product_id"):
                    self.rows[(str(row["product_id"]), str(row.get("sku") or ""))] = row
                    if in_category:
                        self.cat_pool.add(str(row["product_id"]))
        elif name in ("view", "compare"):
            payloads = [obs] if name == "view" else list(obs.get("compared") or [])
            for p in payloads:
                if not isinstance(p, dict) or not p.get("product_id"):
                    continue
                pid = str(p["product_id"])
                self.viewed.add(pid)
                self.products[pid] = p
                if p.get("listing_id"):
                    self.listing[pid] = str(p["listing_id"])
                for v in p.get("variants") or []:
                    if isinstance(v, dict) and v.get("sku"):
                        row = dict(v)
                        row["product_id"] = pid
                        row["title"] = p.get("title")
                        row["brand"] = p.get("brand")
                        self.rows[(pid, str(v["sku"]))] = row
        elif name == "inspect_stock":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if obs.get("product_id"):
                row = self.rows.get((pid, sku), {"product_id": pid, "sku": sku})
                row = dict(row)
                row["price"] = obs.get("price")
                row["in_stock"] = obs.get("in_stock")
                self.rows[(pid, sku)] = row
        elif name == "add_to_cart":
            added = obs.get("added") or {}
            if isinstance(added, dict) and added.get("product_id"):
                key = (str(added["product_id"]), str(added.get("sku") or ""))
                self.cart.append(key)
                self.pending_add = key
                self.verified = None
                self.first_add_done = True
                if obs.get("market_event"):
                    self.event_seen = True
        elif name == "remove_from_cart":
            key = (str(args.get("product_id") or ""), str(args.get("sku") or ""))
            self.cart = [c for c in self.cart if c != key]
            if self.verified == key:
                self.verified = None
        elif name == "place_test_order":
            if obs.get("status") == "placed":
                self.done = True

    def absorb_user(self, content: str) -> None:
        if _QUESTION_RE.search(content) and "still available" not in content.lower():
            self.pending_reply = content
        low = content.lower()
        cat = _required_category("", content)
        if cat:
            self.category = cat
        mb = re.search(r"required brand is ([^.]+)\.", content, re.I)
        if mb:
            self.brand = mb.group(1).strip()
        if self.sealed is None and re.search(r"budget is at most", content, re.I):
            self.sealed = content
        mp = re.search(r"\bprefer\s+([A-Za-z][\w&.-]*)\s+when it remains", content)
        if mp and self.family == "intent_decomposition":
            self.pref_brand = mp.group(1).strip()
        revised = _wanted_spec(content)
        if revised and (self.wanted is None or "instead of" in low or "now have" in low
                        or "restate" in low or self.wanted[0] == revised[0]):
            self.wanted = revised
        amount = None
        m = re.search(r"budget is at most\s+(\d[\d,]*(?:\.\d+)?)\s*(" + _CURRENCIES + r")", content, re.I)
        if m:
            amount = (float(m.group(1).replace(",", "")), m.group(2).upper())
        elif _BUDGET_WORDS.search(low):
            amount = _parse_amount(content)
        if amount:
            value, cur = amount
            if self.budget is None or value < self.budget:
                self.budget = value
            self.currency = self.currency or cur

    # ------------------------------------------------------------------- guard
    def guard(self, proposed: list[dict]) -> tuple[list[dict], list[dict], list[tuple[dict, str]]]:
        """Split proposed tool calls into (executable actions, their tool_calls, rejections)."""
        actions: list[dict] = []
        kept: list[dict] = []
        rejected: list[tuple[dict, str]] = []
        cart = list(self.cart)
        live_item = self.verified if self.verified in cart else None
        for tc in proposed:
            fn = tc.get("function") or {}
            name = fn.get("name")
            try:
                raw = fn.get("arguments", "{}")
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                if not isinstance(args, dict):
                    raise ValueError
            except (ValueError, TypeError):
                rejected.append((tc, "arguments must be a JSON object"))
                continue
            reason = None
            if name == "search":
                q = str(args.get("query") or "").strip()
                if not q:
                    reason = "search needs a query"
                elif self.family == "ranking" and self.explicit_candidates:
                    reason = "the request restricts the choice to the listed candidate variants; do not search"
                elif q.lower() not in self.searches and len(self.searches) >= self.search_cap():
                    reason = (
                        f"search budget exhausted ({self.search_cap()} distinct queries). Use filter, or "
                        "choose from the products already returned."
                    )
                else:
                    # The environment filters on max_price BEFORE truncating to k, so a budget cap
                    # returns eligible rows that an uncapped search would have hidden behind
                    # over-budget hits. A different cap on a repeated query string reaches new rows
                    # without spending a distinct query.
                    cap = _num(args.get("max_price"))
                    if self.budget is not None and (cap is None or cap > self.budget):
                        cap = self.budget
                    args = {"query": q, "k": 10, "in_stock": True}
                    if cap is not None:
                        args["max_price"] = cap
                    self.searches.add(q.lower())
            elif name == "filter":
                if self.family == "ranking" and self.explicit_candidates:
                    reason = "the request restricts the choice to the listed candidate variants; do not filter"
                args = {k: v for k, v in args.items() if k in ("category", "brand", "max_price", "in_stock", "k")}
                args["k"] = 10
                args["in_stock"] = True
                if self.budget is not None and _num(args.get("max_price")) is None:
                    args["max_price"] = self.budget
            elif name == "compare":
                ids = [str(x) for x in (args.get("product_ids") or []) if x][:5]
                if not ids:
                    reason = "compare needs product_ids"
                else:
                    bad = [i for i in ids if self.phrase_mismatch(i) or self.row_ineligible(i)]
                    if bad:
                        reason = (
                            f"products {bad} do not carry every requested term as a whole title word; viewing "
                            "them lowers shortlist precision. Compare only matching titles."
                        )
                args = {"product_ids": ids}
            elif name == "view":
                pid = str(args.get("product_id") or "")
                if not pid:
                    reason = "view needs product_id"
                elif self.phrase_mismatch(pid) or self.row_ineligible(pid):
                    reason = (
                        "that product is off the eligible pool (title lacks a requested term, out of stock, or over "
                        "budget); viewing it lowers shortlist precision. View only matching in-budget titles."
                    )
                args = {"product_id": pid}
            elif name == "inspect_stock":
                args = {"product_id": str(args.get("product_id") or ""), "sku": str(args.get("sku") or "")}
                if not args["product_id"] or not args["sku"]:
                    reason = "inspect_stock needs product_id and sku"
            elif name == "inspect_cart":
                args = {}
            elif name == "message":
                content = str(args.get("content") or "").strip()
                if not content:
                    reason = "message needs content"
                elif self.messages_sent >= _MESSAGE_CAP and not self.pending_reply:
                    reason = "message budget exhausted; decide yourself and continue"
                else:
                    args = {"content": content}
                    self.messages_sent += 1
                    if self.pending_reply:
                        self.pending_reply = None
            elif name == "add_to_cart":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                args = {"product_id": pid, "sku": sku}
                row = self.rows.get((pid, sku))
                if not pid or not sku:
                    reason = "add_to_cart needs product_id and sku"
                elif pid not in self.viewed:
                    reason = f"view or compare product {pid} before adding it, then add the sku you verified"
                elif (pid, sku) in self.dead:
                    reason = "that variant is no longer valid; choose a different product"
                elif row is not None and row.get("in_stock") is False:
                    reason = "that variant is out of stock; choose an in-stock one"
                elif row is not None and not self.price_ok(_num(row.get("price")), original=(self.family == "intent_decomposition" and not self.first_add_done and self.brand_matches(self.pref_brand, pid) if self.pref_brand else False)):
                    reason = f"that variant costs {row.get('price')} which exceeds the budget {self.budget}"
                elif live_item is not None:
                    reason = (
                        f"the cart already holds verified item {live_item[0]}/{live_item[1]}; place the order "
                        "for it, or remove it first if you really want another item"
                    )
                elif self.family == "preference_reasoning" and not self.first_add_done and len(set(self.listing.values())) < 2:
                    reason = "inspect or compare at least two eligible products with different listing_id before the first add"
                elif self.family == "preference_reasoning" and self.listing.get(pid) in self.dead_listing:
                    reason = "recover to a product with a DIFFERENT listing_id than the repriced one"
                elif self.pending_add is not None and self.verified is None:
                    reason = "wait for the controller check of the item you just added"
                elif any(c not in self.dead for c in cart):
                    reason = "remove the previous cart item before adding another"
                elif self.phrase_mismatch(pid):
                    reason = "that product's title does not contain every requested term; add a matching product"
                else:
                    reason = (
                        self.category_objection(pid)
                        or self.device_objection(pid)
                        or self.cheap_objection(pid, sku)
                        or self.spec_objection(pid, sku)
                        or self.required_brand_objection(pid, sku)
                        or self.preferred_brand_objection(pid, sku)
                        or self.preference_objection(pid, sku)
                        or self.recovery_objection(pid, sku)
                    )
                    if reason is None and self.family == "ranking":
                        top = self.rank_top()
                        if top is not None and top["key"] != (pid, sku) and not self.same_rank(top, (pid, sku)):
                            tp, ts = top["key"]
                            if tp in self.viewed:
                                # The stated rule decides; substitute the top-ranked variant.
                                args = {"product_id": tp, "sku": ts}
                                self.note(
                                    f"Your add of {pid}/{sku} was replaced by the top-ranked variant {tp}/{ts} "
                                    f"({top['hz']:g}Hz, {top['size']:g}in, {top['price']:g}) per the stated priority rule."
                                )
                            else:
                                reason = self.rank_objection(pid, sku)
            elif name == "remove_from_cart":
                args = {"product_id": str(args.get("product_id") or ""), "sku": str(args.get("sku") or "")}
                cart = [c for c in cart if c != (args["product_id"], args["sku"])]
                if live_item == (args["product_id"], args["sku"]):
                    live_item = None
            elif name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid or not sku:
                    if self.verified and self.verified in cart:
                        pid, sku = self.verified
                    elif cart:
                        pid, sku = cart[-1]
                if not cart:
                    reason = "the cart is empty; add a verified item first"
                elif (pid, sku) not in cart:
                    reason = "order only an item that is in the cart"
                elif self.verified != (pid, sku):
                    reason = "the item must be verified by the controller after the add before ordering"
                elif self.pending_reply:
                    reason = f"reply to the shopper first with message: they said: {self.pending_reply!r}"
                else:
                    args = {"product_id": pid, "sku": sku}
                    just = self.normalize_justification(args_just=self._raw_just(fn), pid=pid, sku=sku)
                    if just is not None:
                        args["justification"] = just
                    elif self.family == "justification":
                        reason = "justification claims are required and must be checkable; include spec, price, in_stock, brand"
            else:
                reason = f"unknown tool {name}"
            if reason is None and name in ("search", "filter", "view", "compare", "inspect_stock", "inspect_cart"):
                key = name + ":" + json.dumps(args, sort_keys=True)
                if key in self.read_history and not (name == "inspect_stock" and self.first_add_done):
                    reason = (
                        "that exact call was already made and its result is in this conversation; repeating it "
                        "returns the same rows. Choose a different query, product, or the next protocol step."
                    )
                    self.dup_rejects += 1
                else:
                    self.read_history.add(key)
            if reason:
                rejected.append((tc, reason))
                continue
            actions.append({"name": name, "args": args})
            kept.append(tc)
            if name in ("add_to_cart", "place_test_order"):
                break  # nothing may follow a cart commit or an order in the same turn
            if len(actions) >= self.max_calls:
                break
        self.autobatch_views(actions, kept)
        return actions, kept, rejected

    def autobatch_views(self, actions: list[dict], kept: list[dict]) -> None:
        """One view per turn wastes turns; add a few more unviewed candidates to the same read turn."""
        if self.family == "retrieval_recall" or not actions:
            return
        names = [a["name"] for a in actions]
        if names.count("view") != 1 or any(n in ("add_to_cart", "place_test_order", "remove_from_cart") for n in names):
            return
        chosen = {a["args"]["product_id"] for a in actions if a["name"] == "view"}
        extra: list[str] = []
        for (pid, _sku), row in reversed(list(self.rows.items())):
            if pid in chosen or pid in self.viewed or pid in extra:
                continue
            if row.get("in_stock") is False or not self.price_ok(_num(row.get("price"))):
                continue
            if self.category and _ACCESSORY_RE.search(str(row.get("title") or "")) and re.search(r"\b(smart ?phones?|mobile phones?)\b", self.category, re.I):
                continue
            extra.append(pid)
            if len(extra) >= 3:
                break
        for i, pid in enumerate(extra):
            if len(actions) >= self.max_calls:
                break
            actions.append({"name": "view", "args": {"product_id": pid}})
            kept.append({
                "id": f"ctl-batch-{self.turn + 1}-{i}",
                "type": "function",
                "function": {"name": "view", "arguments": json.dumps({"product_id": pid})},
            })

    # ------------------------------------------------------------- spec checks
    def variant_spec(self, key: str, pid: str, sku: str) -> str | None:
        row = self.rows.get((pid, sku)) or {}
        product = self.products.get(pid) or {}
        title = product.get("title") or row.get("title")
        return _resolve_spec(key, row.get("options"), product.get("facts"), title)

    def spec_objection(self, pid: str, sku: str) -> str | None:
        """Reject a variant that does not verifiably carry the firm spec."""
        if not self.wanted or self.family in ("ranking", "recovery", "retrieval_recall", "preference_reasoning"):
            return None
        key, want = self.wanted
        got = self.variant_spec(key, pid, sku)
        if got == want:
            return None
        n = self.spec_rejects.get((pid, sku), 0) + 1
        self.spec_rejects[(pid, sku)] = n
        if got is None:
            if n >= 3:
                return None
            return (
                f"spec check: the requirement is {key} = {want}, but this variant's options, facts and title do "
                f"not state one clear {key} value (missing or several different values). Such a listing cannot "
                "satisfy the requirement. Choose a variant whose option or facts state exactly that value."
            )
        return (
            f"spec check: the requirement is {key} = {want}; this variant resolves to {got} (variant option first, "
            "otherwise the single value agreed by facts and title). Choose a variant that states the required value."
        )

    def row_ineligible(self, pid: str) -> bool:
        """Retrieval family: a row already known to be out of stock or over budget is off-pool."""
        if self.family != "retrieval_recall":
            return False
        rows = [row for (p, _sku), row in self.rows.items() if p == pid]
        if not rows:
            return False
        return all(
            row.get("in_stock") is False or (_num(row.get("price")) is not None and not self.price_ok(_num(row.get("price"))))
            for row in rows
        )

    def category_objection(self, pid: str) -> str | None:
        """The verifier's category rule: the product's path must share a canonical label with
        the required category (aliases: Mobile/Cell Phones = Smartphones, Smart Watches =
        Smartwatches, Headphones & Headsets = Headphones ...), or pass one of its two title
        fallbacks (gaming monitors, over-ear headphones)."""
        if not self.category:
            return None
        product = self.products.get(pid) or {}
        path = [str(x) for x in (product.get("category_path") or [])]
        if not path:
            return None
        if _category_matches(self.category, path, product.get("title")):
            return None
        return (
            f"category check: the required category is '{self.category}' but this product's category_path is "
            f"{path} - it is not classified as that kind of product. Choose a product whose category_path carries "
            "the required category."
        )

    def brand_of(self, pid: str) -> str:
        product = self.products.get(pid) or {}
        if product.get("brand"):
            return str(product["brand"])
        for (p, _sku), row in self.rows.items():
            if p == pid and row.get("brand"):
                return str(row["brand"])
        return ""

    def brand_matches(self, expected: str | None, pid: str, *, strict: bool = False) -> bool:
        """Expected brand tokens within the catalog brand field, or (when the public brand field is
        empty) at the start of the title - the catalog's own brand for such listings comes from the
        title, as measured on the qualifying pack.

        strict=True drops the title fallback. A HARD brand requirement is checked against the
        brand FIELD alone, so a brand-less listing titled "Samsung S25 Ultra" is not in gold -
        measured: such a first add failed intended_event_commitment and cost the episode."""
        if not expected:
            return True
        exp = _title_tokens(expected)
        if not exp:
            return True
        act = _title_tokens(self.brand_of(pid))
        if exp <= act:
            return True
        if not act and not strict:
            product = self.products.get(pid) or {}
            title = str(product.get("title") or next((r.get("title") for (p, _k), r in self.rows.items() if p == pid), "") or "")
            head = _title_tokens(" ".join(title.split()[:4]))
            return exp <= head
        return False

    def required_brand_objection(self, pid: str, sku: str) -> str | None:
        """A required brand (from the shopper's reply) must hold for every add, first or recovery."""
        if not self.brand or self.family in ("retrieval_recall",):
            return None
        if self.brand_matches(self.brand, pid, strict=True):
            return None
        n = self.cat_rejects.get(("brand", pid), 0) + 1
        self.cat_rejects[("brand", pid)] = n
        if n >= 3:
            return None
        return (
            f"brand check: the shopper requires brand '{self.brand}' but this listing's brand field is "
            f"'{self.brand_of(pid) or 'empty'}' and its title does not start with it. Choose a {self.brand} listing."
        )

    def preferred_brand_objection(self, pid: str, sku: str) -> str | None:
        """TF1: the first add must honour the preferred brand when a fully valid item of it is known."""
        if self.family != "intent_decomposition" or not self.pref_brand or self.first_add_done:
            return None
        if self.brand_matches(self.pref_brand, pid):
            return None
        # The verifier judges "a preferred-brand item was available" against the ORIGINAL budget,
        # even after the shopper tightened it; the tightened budget governs the final order only.
        alts = [
            (p, k) for (p, k) in self.rows
            if p in self.viewed and p != pid and self.brand_matches(self.pref_brand, p) and self.candidate_valid(p, k, original=True)
        ]
        if not alts:
            return None
        alts.sort(key=lambda t: self.candidate_score(*t), reverse=True)
        ap, ak = alts[0]
        return (
            f"preferred-brand check: the shopper prefers {self.pref_brand} and {ap}/{ak} is a {self.pref_brand} listing "
            f"(brand field) that meets every firm requirement in stock within budget; this candidate's brand field is "
            f"'{self.brand_of(pid) or 'empty'}'. Add the preferred-brand item first."
        )

    def preference_objection(self, pid: str, sku: str) -> str | None:
        """TF4: add only a candidate whose title/facts explicitly evidence the most stated preference axes."""
        if self.family != "preference_reasoning" or not self.sealed:
            return None
        self.extract_pref_axes()
        if not self.pref_axes or pid not in self.products:
            return None
        mine, mine_note = self.pref_score(pid)
        if self.audit_fail >= 3:
            return None
        if not self.first_add_done:
            first, planned = self.tf4_first_pick()
            if first is None or first[0] == pid:
                return None
            # Another strong candidate on the same listing as the planned first add is equally fine.
            if planned is not None and self.listing.get(pid, pid) != self.listing.get(planned[0], planned[0]) and self.pref_strong(pid) and mine >= self.pref_score(first[0])[0]:
                return None
            if planned is None and mine >= self.pref_score(first[0])[0]:
                return None
            self.audit_fail += 1
            why = (
                f"{first[0]}/{first[1]} (score {self.pref_score(first[0])[0]:g}: {self.pref_score(first[0])[1]}) is the right FIRST add"
                + (f"; {planned[0]}/{planned[1]} (score {self.pref_score(planned[0])[0]:g}) is on the best listing and is kept as the replacement if the market moves after the first add" if planned else "")
            )
            return f"preference check: {pid} scores {mine:g} ({mine_note}); {why}. Add {first[0]}/{first[1]}."
        scored = [(self.pref_score(p)[0], p) for p in self.viewed if p != pid and self.candidate_valid_any(p)]
        best = max((sc for sc, _p in scored), default=-1.0)
        if mine >= best:
            return None
        self.audit_fail += 1
        better = sorted(((sc, p) for sc, p in scored if sc > mine), reverse=True)[:3]
        return (
            f"preference check: {pid} scores {mine:g} on the shopper's priority-weighted preference axes ({mine_note}). "
            "Better-evidenced candidates you already viewed: "
            + "; ".join(f"{p} (score {sc:g}: {self.pref_score(p)[1]})" for sc, p in better)
            + ". Add the best-evidenced valid candidate."
        )

    def recovery_identity(self, pid: str, sku: str) -> tuple[bool, str | None]:
        """(model matches preferred, resolved colour) using the verifier's public order:
        model from the model fact, else an unambiguous code in the title; colour from the
        selected variant's options, then a colour fact, then the title; several different
        colours in the deciding source = unresolved (None)."""
        product = self.products.get(pid) or {}
        row = self.rows.get((pid, sku)) or {}
        facts = product.get("facts") or {}
        title = str(product.get("title") or row.get("title") or "")
        want = _norm_model(self.pref_model)
        fact_model = _norm_model(str(facts.get("model") or ""))
        if fact_model:
            model_ok = bool(want) and (fact_model == want or fact_model.endswith(want) or want.endswith(fact_model) and len(fact_model) >= 4)
        else:
            model_ok = bool(want) and want in _norm_model(title) and any(ch.isdigit() for ch in want)
        color: str | None = None
        for source in (str(row.get("options") or ""), str(facts.get("color") or ""), title):
            found = _colors_in(source)
            if found:
                color = next(iter(found)) if len(found) == 1 else None
                break
        return model_ok, color

    def recovery_level(self, pid: str, sku: str) -> int | None:
        model_ok, color = self.recovery_identity(pid, sku)
        if color is None:
            return None
        color_ok = self.pref_color is not None and color == self.pref_color
        return 0 if model_ok and color_ok else 1 if model_ok else 2 if color_ok else 3

    def recovery_objection(self, pid: str, sku: str) -> str | None:
        """TF6: commit to the least-relaxed resolvable candidate (exact model+colour first)."""
        if self.family != "recovery" or not self.pref_model or self.recovery_rejects >= 3:
            return None
        mine = self.recovery_level(pid, sku)
        levels = {
            (p, k): self.recovery_level(p, k)
            for (p, k) in self.rows
            if p in self.viewed and (p, k) != (pid, sku) and self.candidate_valid(p, k)
        }
        resolved = {key: lv for key, lv in levels.items() if lv is not None}
        best = min(resolved.values(), default=None)
        if mine is None:
            if not resolved:
                return None
            self.recovery_rejects += 1
            bp, bk = min(resolved, key=lambda key: resolved[key])
            return (
                f"identity check: this variant's colour cannot be resolved (no single colour in its options, "
                f"colour fact or title), so it cannot be a valid pick. {bp}/{bk} resolves (relaxation level "
                f"{resolved[(bp, bk)]}: 0 = exact model+colour, 1 = same model other colour, 2 = other model same "
                "colour). Add the least-relaxed resolvable candidate."
            )
        if best is not None and mine > best:
            self.recovery_rejects += 1
            bp, bk = min(resolved, key=lambda key: resolved[key])
            return (
                f"relaxation check: this pick is level {mine} but {bp}/{bk} is level {best} (0 = exact model+colour, "
                "1 = same model other colour, 2 = other model same colour, 3 = other). Model identity = the model "
                "fact (else an unambiguous model code in the title); colour = the variant's options, else a colour "
                "fact, else the title. Add the least-relaxed valid candidate."
            )
        return None

    # ---- preference axes (extracted once from the shopper's own words) ----
    _AXIS_STOP = frozenset(
        "a an the and or of for with in on at to from is are be that this it its so there no any one you your "
        "my me i we they them their while when what which who how most least next close behind counts count "
        "together beat single several equally by far up under over than more less very really just also "
        "something things stays put long hard fully real relied can could would should will off out into onto "
        "all every each per some such like as into during after before hours hour day days".split()
    )

    def extract_pref_axes(self) -> None:
        """Ask the model ONCE to split the shopper's reply into distinct preference axes (name +
        the shopper's own phrase, priority order); fall back to a heuristic split. Scoring is
        then deterministic from those words, never from a per-candidate judgement."""
        if self.pref_axes_tried or not self.sealed:
            return
        self.pref_axes_tried = True
        axes: list[dict] = []
        # The shopper's reply carries a priority statement ("X counts most, with Y close behind, Z next,
        # W counts least" / "... count equally next ..."): split it deterministically first.
        head = self.sealed.split("several together")[0]
        cur = re.findall(r"\b(" + _CURRENCIES + r")\b", head, re.I)
        if cur:
            head = head[head.rfind(cur[-1]) + len(cur[-1]):]
        if re.search(r"counts? (?:most|least)", head):
            parts = re.split(r"counts? (?:most|least|equally)|close behind|\bnext\b|,|\band\b(?! (?:water|sweat))", head)
            for part in parts:
                w = [t for t in re.findall(r"[a-z]+", part.lower()) if t not in self._AXIS_STOP]
                if 1 <= len(w) <= 5:
                    axes.append({"name": " ".join(w), "phrase": ""})
            axes = axes[:6]
        if len(axes) < 2 and not self.llm_dead:
            axes = []
            prompt = (
                "A shopper described their preferences (a priority ranking followed by what each preference "
                "means). Split them into DISTINCT soft preference axes, one entry per preference, in the shopper's "
                "priority order (most important first), merging the ranking line and its explanation. Do NOT include "
                "hard requirements (category, budget, price, stock, being a genuine device rather than an accessory). "
                "Quote the shopper's own wording. Reply with JSON only: "
                "{\"axes\": [{\"name\": \"short name\", \"phrase\": \"shopper's words for it\"}]}\n\n"
                f"Shopper's words:\n{self.sealed}"
            )
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": _model_candidates()[min(self.model_index, len(_model_candidates()) - 1)],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                },
            )
            try:
                content = inference["choices"][0]["message"]["content"]
                data = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
                for a in data.get("axes") or []:
                    name = str(a.get("name") or "").strip()
                    phrase = str(a.get("phrase") or "").strip()
                    if name and not re.search(r"categor|budget|price|stock|device|accessor|genuine|product type", f"{name} {phrase}", re.I):
                        axes.append({"name": name, "phrase": phrase})
            except (TypeError, KeyError, IndexError, ValueError, AttributeError):
                axes = []
        # Term sets: the axis's own words minus words shared with other axes, stemmed to a prefix.
        raw_terms = []
        for a in axes:
            toks = [t for t in re.findall(r"[a-z]+", f"{a['name']} {a['phrase']}".lower()) if t not in self._AXIS_STOP and len(t) >= 3]
            raw_terms.append(set(toks))
        for i, a in enumerate(axes):
            others = set().union(*[raw_terms[j] for j in range(len(axes)) if j != i]) if len(axes) > 1 else set()
            own = {t for t in raw_terms[i] if t not in others}
            terms = {t[:5] if len(t) > 5 else t for t in own}
            a["words"] = sorted(own)
            text = f"{a['name']} {a['phrase']}".lower()
            kind = "battery" if "battery" in text else "wireless" if re.search(r"wireless|cable", text) else "brand" if "brand" in text else "generic"
            if kind == "wireless":
                terms |= {"wirel", "bluet"}
            a["terms"] = terms
            a["kind"] = kind
        self.pref_axes = axes

    def pref_profile(self) -> dict | None:
        return _tf4_profile(self.category)

    def pref_score(self, pid: str) -> tuple[float, str]:
        """Evidence score of a viewed product. With a known category profile this is the
        verifier's own alignment (0..1, its regexes over the same visible text); otherwise the
        priority-weighted match of the shopper's axis words (battery needs a stated duration)."""
        product = self.products.get(pid) or {}
        profile = self.pref_profile()
        if profile is not None:
            score, axes = _tf4_alignment(profile, product)
            return (score, ", ".join(f"{k}={v:g}" for k, v in axes.items() if v) or "no axis evidenced")
        if not self.pref_axes:
            return (0.0, "no axes")
        facts = product.get("facts") or {}
        text = (str(product.get("title") or "") + " " + " ".join(f"{k} {v}" for k, v in facts.items())).lower().replace("-", "")
        n = len(self.pref_axes)
        score = 0.0
        notes = []
        for i, a in enumerate(self.pref_axes):
            weight = float(n - i)
            hit = 0.0
            if a["kind"] == "battery":
                dur = [float(m.group(1)) * (24 if m.group(2).startswith("d") else 1) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*\+?\s*(hours?|hrs?|h\b|days?)", text)]
                if dur:
                    best = max(dur)
                    hit = 1.0 if best >= 15 else 0.75 if best >= 10 else 0.5 if best >= 6 else 0.25
                elif "battery" in text:
                    hit = 0.25
            elif a["kind"] == "brand":
                hit = 0.0  # brand credibility cannot be evidenced from listing text
            else:
                hit = 1.0 if any(term and term in text for term in a["terms"]) else 0.0
            if hit:
                notes.append(f"{a['name']}={hit:g}")
            score += weight * hit
        return (round(score, 2), ", ".join(notes) or "no axis evidenced")

    def pref_strong(self, pid: str) -> bool:
        """The verifier's commitment floor (alignment >= 0.65) when the category profile is
        known; otherwise the proxy: the top axis fully evidenced plus two more."""
        if self.pref_profile() is not None:
            return self.pref_score(pid)[0] >= 0.65
        if not self.pref_axes:
            return False
        product = self.products.get(pid) or {}
        facts = product.get("facts") or {}
        text = (str(product.get("title") or "") + " " + " ".join(f"{k} {v}" for k, v in facts.items())).lower().replace("-", "")
        full = []
        for a in self.pref_axes:
            if a["kind"] == "battery":
                dur = [float(m.group(1)) * (24 if m.group(2).startswith("d") else 1) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*\+?\s*(hours?|hrs?|h\b|days?)", text)]
                full.append(bool(dur) and max(dur) >= 15)
            elif a["kind"] == "brand":
                full.append(False)
            else:
                full.append(any(term and term in text for term in a["terms"]))
        return bool(full) and full[0] and sum(full) >= 3

    def tf4_first_pick(self) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
        """(first add, planned recovery). The market event hits the first add and the replacement
        must come from a different listing, so when the best-scoring listing has no equal on
        another listing, add a strong runner-up first and keep the best for the recovery."""
        cands = [(p, k) for (p, k) in self.rows if p in self.viewed and self.candidate_valid(p, k)]
        if not cands:
            return None, None
        scored = sorted(cands, key=lambda c: (self.pref_score(c[0])[0], self.candidate_score(*c)), reverse=True)
        best = scored[0]
        best_score = self.pref_score(best[0])[0]
        best_listing = self.listing.get(best[0], best[0])
        equal_elsewhere = any(
            self.listing.get(p, p) != best_listing and self.pref_score(p)[0] >= best_score for (p, k) in scored
        )
        if equal_elsewhere:
            return best, None
        runners = [c for c in scored if self.listing.get(c[0], c[0]) != best_listing and self.pref_strong(c[0])]
        if runners:
            return runners[0], best
        return best, None

    def candidate_valid_any(self, pid: str) -> bool:
        return any(self.candidate_valid(pid, k) for (p, k) in self.rows if p == pid)

    def run_preference_audit(self) -> None:
        """One structured inference call scoring viewed candidates on the shopper's stated axes."""
        pending = [p for p in self.viewed if p not in self.audit_seen and p in self.products]
        if not pending:
            return
        items = []
        for p in pending[:10]:
            pr = self.products[p]
            items.append({"product_id": p, "title": str(pr.get("title") or "")[:200], "facts": {k: str(v)[:120] for k, v in (pr.get("facts") or {}).items()}})
            self.audit_seen.add(p)
        prompt = (
            "You audit shopping candidates against a shopper's stated preferences.\n"
            f"Shopper's own words (their priorities and what each preference means):\n{self.sealed}\n\n"
            "The shopper states each preference twice (a priority ranking, then what each one means): merge "
            "those into DISTINCT axes, one entry per preference (typically 3-5 axes in total). For EACH candidate "
            "below and EACH distinct axis, decide whether the candidate's title or facts EXPLICITLY state evidence "
            "for that axis - concrete words, ratings or numbers in the listing itself (not inference from the "
            "product type, not absence of contrary evidence). "
            "List the axes in the shopper's stated priority order, most important first. "
            "Reply with JSON only: {\"axes\": [\"axis name\", ...], \"candidates\": [{\"product_id\": \"...\", "
            "\"evidenced\": [\"axis name\", ...], \"note\": \"short evidence summary\"}]}\n\n"
            f"Candidates:\n{json.dumps(items, ensure_ascii=False)}"
        )
        inference = _proxy.post(
            "/inference/chat/completions",
            json_data={
                "model": _model_candidates()[min(self.model_index, len(_model_candidates()) - 1)],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
        )
        try:
            content = inference["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", content, re.S)
            data = json.loads(m.group(0))
            axes = [str(a) for a in data.get("axes") or []]
            n_axes = max(len(axes), 1)
            weight = {a.lower(): n_axes - i for i, a in enumerate(axes)}
            for c in data.get("candidates") or []:
                pid = str(c.get("product_id") or "")
                ev = list(dict.fromkeys(str(a) for a in (c.get("evidenced") or [])))
                if pid in self.products:
                    # Priority-weighted evidence: the most important axis counts n, the least 1.
                    score = sum(weight.get(a.lower(), 1) for a in ev)
                    self.audit[pid] = {"axes": max(n_axes, len(ev)), "evidenced": score, "count": len(ev), "detail": str(c.get("note") or ", ".join(ev))[:160]}
        except (TypeError, KeyError, IndexError, ValueError, AttributeError):
            return

    def cheap_objection(self, pid: str, sku: str) -> str | None:
        """A device priced at a tiny fraction of the budget is almost always a part or placeholder."""
        if self.budget is None or self.family == "retrieval_recall":
            return None
        row = self.rows.get((pid, sku)) or {}
        price = _num(row.get("price"))
        if price is None or price >= 0.03 * self.budget:
            return None
        n = self.cat_rejects.get(("cheap", pid), 0) + 1
        self.cat_rejects[("cheap", pid)] = n
        # Between 1.5% and 3% of budget the model may insist (3 strikes); below that never.
        if n >= 3 and price >= 0.015 * self.budget:
            return None
        return (
            f"sanity check: {price:g} is under 3% of the shopper's budget {self.budget:g} - listings that cheap are "
            "usually parts, accessories or placeholders rather than the requested device. Prefer a realistically "
            "priced genuine item; add this one only if its facts clearly prove it is the complete device."
        )

    def device_objection(self, pid: str) -> str | None:
        """Smartphone requests: the listing must be a genuine phone, not an accessory or part."""
        product = self.products.get(pid) or {}
        title = str(product.get("title") or "")
        facts = product.get("facts") or {}
        if self.category and _ACCESSORY_RE.search(title) and re.search(
            r"\b(for|compatible with|fits?)\b", title, re.I
        ) and not re.search(r"\b(with|free|bundle|included)\b", title, re.I):
            return (
                "device check: the title reads like an accessory or part sold FOR a device, not the device "
                "itself. Choose a genuine device listing."
            )
        if not self.category or "smartphone" not in _category_labels([self.category]):
            return None
        status = _smartphone_evidence(product)
        if status == "conflict":
            return (
                "device check: the title or device-type facts say this is not a smartphone (a tablet, feature "
                "phone, accessory, part or mystery box). Choose a genuine smartphone listing."
            )
        if status != "supported":
            return (
                "device check: no positive smartphone evidence (title/device type saying smartphone, or a phone "
                "OS in the facts). Choose a listing with that evidence."
            )
        # An accessory sold FOR a phone ("case for iPhone 13") is not the phone; a phone bundled with a case is.
        if _ACCESSORY_RE.search(title) and re.search(r"\b(for|compatible with|fits?)\b", title, re.I) and not re.search(
            r"\b(with|free|bundle|included)\b", title, re.I
        ):
            return (
                "device check: the title reads like an accessory or part sold for a phone, not a complete "
                "smartphone. Choose a genuine smartphone listing."
            )
        return None

    @staticmethod
    def title_accessory(title: str) -> bool:
        """A title that sells a part or accessory FOR a device, readable without a view."""
        return bool(
            title
            and _ACCESSORY_RE.search(title)
            and re.search(r"\b(for|compatible with|fits?)\b", title, re.I)
            and not re.search(r"\b(with|free|bundle|included)\b", title, re.I)
        )

    def phrase_mismatch(self, pid: str) -> bool:
        if not self.phrase:
            return False
        title = None
        product = self.products.get(pid)
        if product:
            title = product.get("title")
        else:
            for (p, _sku), row in self.rows.items():
                if p == pid and row.get("title"):
                    title = row["title"]
                    break
        if title is None:
            return False
        return not self.phrase <= _title_tokens(title)

    def rank_top(self) -> dict | None:
        ranked = self.ranked_candidates()
        return ranked[0] if ranked else None

    def same_rank(self, top: dict, key: tuple[str, str]) -> bool:
        mine = next((r for r in self.ranked_candidates() if r["key"] == key), None)
        return mine is not None and mine["vec"] == top["vec"]

    def rank_objection(self, pid: str, sku: str) -> str | None:
        """Ranking family: the item added must be the top of the stated priority order."""
        if self.family != "ranking":
            return None
        ranked = self.ranked_candidates()
        if not ranked:
            return None
        top = ranked[0]
        mine = next((r for r in ranked if r["key"] == (pid, sku)), None)
        if mine is not None and mine["vec"] == top["vec"]:
            return None
        n = self.rank_rejects.get((pid, sku), 0) + 1
        self.rank_rejects[(pid, sku)] = n
        table = "; ".join(
            f"{r['key'][0]}/{r['key'][1]}: {r['hz']:g}Hz, {r['size']:g}in, {r['price']:g}" for r in ranked[:6]
        )
        why = (
            "it is not among the resolvable in-stock candidates" if mine is None
            else f"it ranks below the top ({mine['hz']:g}Hz, {mine['size']:g}in, {mine['price']:g})"
        )
        return (
            f"ranking check: by the stated rule (highest refresh rate, then largest screen, then lowest price - "
            f"variant option first, else the single value agreed by facts and title) the current top is "
            f"{top['key'][0]}/{top['key'][1]} ({top['hz']:g}Hz, {top['size']:g}in, {top['price']:g}); your pick {why}. "
            f"Ranked: {table}. View the top item if you have not, then add it (or explain via a different choice "
            "only if its facts clearly contradict this)."
        )

    def ranked_candidates(self) -> list[dict]:
        keys = list(self.explicit_candidates) or list(self.rows)
        out = []
        for pid, sku in keys:
            if (pid, sku) in self.dead:
                continue
            row = self.rows.get((pid, sku)) or {}
            if row.get("in_stock") is False:
                continue
            price = _num(row.get("price"))
            if price is None or not self.price_ok(price):
                continue
            hz = self.variant_spec("refresh_rate", pid, sku)
            size = self.variant_spec("screen_size", pid, sku)
            if hz is None or size is None:
                continue
            hzv, sizev = float(hz[:-2]), float(size[:-2])
            out.append({"key": (pid, sku), "hz": hzv, "size": sizev, "price": price, "vec": (-hzv, -sizev, price)})
        out.sort(key=lambda r: (r["vec"], r["key"]))
        return out

    @staticmethod
    def _raw_just(fn: dict) -> Any:
        try:
            raw = fn.get("arguments", "{}")
            args = json.loads(raw) if isinstance(raw, str) else raw
            return (args or {}).get("justification")
        except (ValueError, TypeError, AttributeError):
            return None

    # ------------------------------------------------------------ justification
    def normalize_justification(self, args_just: Any, pid: str, sku: str) -> dict | None:
        if self.family != "justification":
            return None
        claims_in = []
        if isinstance(args_just, dict) and isinstance(args_just.get("claims"), list):
            claims_in = [c for c in args_just["claims"] if isinstance(c, dict)]
        row = self.rows.get((pid, sku)) or {}
        product = self.products.get(pid) or {}
        out: dict[str, str] = {}
        for c in claims_in:
            field = re.sub(r"[^a-z0-9]+", "_", str(c.get("field") or "").strip().lower()).strip("_")
            value = str(c.get("value") if c.get("value") is not None else "").strip()
            if field not in _CLAIM_FIELDS or field == "category" or not value:
                continue
            if field in _SOFT_CLAIMS - {"brand"} and value.lower() not in ("true", "yes"):
                continue
            out[field] = value
        price = _num(row.get("price"))
        if price is not None:
            out["price"] = f"{price:.2f}"
        if row.get("in_stock") is True:
            out["in_stock"] = "true"
        brand = str(product.get("brand") or "").strip()
        if brand:
            out["brand"] = brand
            # A brand claim already satisfies the "one reason" rule; extra feature claims
            # only add ways to be wrong (e.g. HDR10 does not support an hdr claim).
            for f in list(out):
                if f in _SOFT_CLAIMS and f != "brand":
                    del out[f]
        elif "brand" in out:
            del out["brand"]
        if not any(f in _SOFT_CLAIMS for f in out):
            # The verifier needs one supported reason; without a brand field, cite a feature
            # the listing's own facts or title state positively.
            feature = self.soft_feature_claim(product)
            if feature:
                out[feature[0]] = feature[1]
        attr = self.gate_attribute()
        for f in list(out):
            if f in _SPEC_CLAIMS and attr is not None and f != attr:
                del out[f]
        auto = self.auto_spec_claim(product)
        if auto:
            out[auto[0]] = auto[1]
        if not any(f in _SPEC_CLAIMS for f in out) or "price" not in out or "in_stock" not in out:
            return None
        if not any(f in _SOFT_CLAIMS for f in out):
            return None
        return {"claims": [{"field": f, "value": v} for f, v in out.items()]}

    def gate_attribute(self) -> str | None:
        """The one specification field the shopper request names (goal text only)."""
        goal = self.query.split("\n\nTask rules:", 1)[0]
        low = goal.lower()
        if re.search(r"\b(4k|uhd|qhd|fhd|full hd|\d{3,4}p)\b|resolution|\d{3,4}\s*[x\u00d7]\s*\d{3,4}", low):
            return "resolution"
        if re.search(r"\d+\s*-?\s*hz\b|refresh rate", low):
            return "refresh_rate"
        if re.search(r"\d+(?:\.\d+)?\s*-?\s*(?:inch|in\b(?!\s+stock)|\")|screen size", low):
            return "screen_size"
        wanted = _wanted_spec(goal, loose=True)
        if wanted and wanted[0] in ("capacity", "wattage", "warranty_duration", "pack_quantity"):
            return wanted[0]
        return None

    def auto_spec_claim(self, product: dict) -> tuple[str, str] | None:
        facts = product.get("facts") or {}
        attr = self.gate_attribute()
        if attr == "refresh_rate" and facts.get("refresh_rate"):
            m = re.search(r"\d+(?:\.\d+)?\s*-?\s*hz", str(facts["refresh_rate"]), re.I)
            if m:
                return "refresh_rate", m.group(0)
        if attr == "screen_size" and facts.get("size"):
            m = re.search(r"\d+(?:\.\d+)?", str(facts["size"]))
            if m:
                return "screen_size", m.group(0) + " inch"
        if attr == "resolution" and facts.get("display"):
            m = re.search(r"\d{3,5}\s*[x\u00d7]\s*\d{3,5}", str(facts["display"]), re.I)
            if m:
                return "resolution", m.group(0)
            m = re.search(r"\b(4k|uhd|qhd|fhd|full hd|1080p|1440p|2160p|720p|hd)\b", str(facts["display"]), re.I)
            if m:
                return "resolution", m.group(0)
        if attr in ("capacity", "wattage", "warranty_duration", "pack_quantity"):
            label = _SPEC_FACT_LABEL[attr]
            vals = _spec_values(attr, str(facts.get(label) or ""))
            if len(vals) == 1:
                value = next(iter(vals))
                unit = {"capacity": " ml", "wattage": " w", "warranty_duration": " months", "pack_quantity": " pcs"}[attr]
                return attr, re.sub(r"[a-z]+$", "", value) + unit
        return None

    def soft_feature_claim(self, product: dict) -> tuple[str, str] | None:
        """A feature reason the verifier can support from this listing's own public text:
        the feature word must appear with no negation around it (its rule, mirrored)."""
        facts = product.get("facts") or {}
        texts = [str(v) for v in facts.values()] + [str(product.get("title") or "")]
        for field, pattern in _SOFT_FEATURE_RE.items():
            support = False
            negated = False
            for text in texts:
                for m in re.finditer(pattern, text, re.I):
                    before = re.split(r"[.;!?\n]", text[: m.start()])[-1]
                    after = re.split(r"[.;!?\n]", text[m.end():])[0]
                    if re.search(_SOFT_NEG_BEFORE, before, re.I) or re.match(_SOFT_NEG_AFTER, after, re.I):
                        negated = True
                    else:
                        support = True
            if support and not negated:
                return field, "true"
        return None

    # ------------------------------------------------------------ verification
    def verify_cart(self) -> None:
        """Deterministic post-add check; marks the item verified or dead."""
        key = self.pending_add
        self.pending_add = None
        if key is None or self.done:
            return
        pid, sku = key
        self.call([{"name": "inspect_stock", "args": {"product_id": pid, "sku": sku}}])
        row = self.rows.get(key) or {}
        price = _num(row.get("price"))
        if row.get("in_stock") is True and self.price_ok(price):
            self.verified = key
            self.verified_price = price
            self.note(
                f"Cart item {pid}/{sku} verified: in stock at {price} (budget {self.budget}). "
                "Place the order for it now with place_test_order, unless a task rule still requires "
                "another action first (for example replying to the shopper)."
            )
        else:
            self.dead.add(key)
            self.event_seen = True
            if self.listing.get(pid):
                self.dead_listing.add(self.listing[pid])
            self.recoveries += 1
            why = "out of stock" if row.get("in_stock") is not True else f"repriced to {price}, above the budget {self.budget}"
            self.call([{"name": "remove_from_cart", "args": {"product_id": pid, "sku": sku}}])
            self.twin_search(pid)
            if self.done:
                return
            hint = ""
            if self.family == "preference_reasoning":
                # Recovery is graded on the replacement's preference fit: commit to the
                # best-evidenced valid candidate on a different listing ourselves.
                self.extract_pref_axes()
                if self.pref_axes and self.turns_left() >= 7 and len(self.searches) < self.search_cap() + 2:
                    # The replacement is graded on the same axes: pull fresh candidates that name
                    # every axis (a stated duration for the battery axis) before choosing.
                    words = sorted({w for a in self.pref_axes for w in a.get("words", []) if a["kind"] not in ("brand", "battery")})
                    profile = self.pref_profile()
                    battery_words = " ".join((profile or {}).get("query", {}).get("battery_life", ["hours", "battery"])[:2])
                    q = f"{self.category or ''} {' '.join(words)} {battery_words}".strip()[:100]
                    if q.lower() not in self.searches:
                        self.searches.add(q.lower())
                        a = {"query": q, "k": 10, "in_stock": True}
                        if self.budget is not None:
                            a["max_price"] = self.budget
                        self.call([{"name": "search", "args": a}])
                        if not self.done:
                            self.view_top_candidates(limit=6)
                best_key = None
                best_n = -1.0
                for (p, k) in self.rows:
                    if p == pid or p not in self.viewed or not self.candidate_valid(p, k):
                        continue
                    n = self.pref_score(p)[0]
                    if n > best_n:
                        best_n, best_key = n, (p, k)
                if best_key is not None and self.pref_strong(best_key[0]) and self.turns_left() >= 4:
                    bp, bk = best_key
                    self.call([
                        {"name": "view", "args": {"product_id": bp}},
                        {"name": "inspect_stock", "args": {"product_id": bp, "sku": bk}},
                        {"name": "add_to_cart", "args": {"product_id": bp, "sku": bk}},
                    ])
                    if self.done:
                        return
                    self.verify_cart()
                    return
            if self.family == "preference_reasoning" and self.pref_axes:
                ranked = sorted(((self.pref_score(p)[0], p) for p in self.viewed if p != pid and self.candidate_valid_any(p)), reverse=True)[:3]
                if ranked:
                    hint = " Best-evidenced remaining candidates: " + "; ".join(f"{p} (score {n:g})" for n, p in ranked) + "."
            elif self.family == "recovery" and self.pref_model:
                lv = sorted(((self.recovery_level(p, k), p, k) for (p, k) in self.rows if p in self.viewed and self.candidate_valid(p, k) and self.recovery_level(p, k) is not None))[:3]
                if lv:
                    hint = " Least-relaxed remaining candidates: " + "; ".join(f"{p}/{k} (level {l})" for l, p, k in lv) + " (0 exact, 1 same model other colour, 2 other model same colour)."
            self.note(
                f"Cart item {pid}/{sku} is {why}; it has been removed from the cart. Recover now: choose a "
                "DIFFERENT product that meets every requirement following the task's recovery rule, view it "
                "(re-view it now even if you saw it before), add it to the cart, and wait for the check." + hint
            )

    def twin_search(self, dead_pid: str) -> None:
        """After a stockout, look for the same product on another listing and for any other
        in-category candidate. A plain title query is drowned by accessories ("Apple iPhone 13"
        returns ten cases), so the query also carries the gated spec and the category, and a
        category filter enumerates genuine in-category rows the query never reaches."""
        if self.family in ("recovery", "retrieval_recall", "ranking") or self.turns_left() < 6:
            return
        title = str((self.products.get(dead_pid) or {}).get("title") or "")
        if not title:
            title = str(next((r.get("title") for (p, _k), r in self.rows.items() if p == dead_pid), "") or "")
        title = " ".join(title.split())[:80]
        spec = _spec_query_words(self.wanted)
        actions: list[dict[str, Any]] = []
        for q in (f"{title} {spec}".strip(), title):
            if not q or q.lower() in self.searches:
                continue
            self.searches.add(q.lower())
            args: dict[str, Any] = {"query": q[:100], "k": 10, "in_stock": True}
            if self.budget is not None:
                args["max_price"] = self.budget
            actions.append({"name": "search", "args": args})
            if len(actions) >= 2:
                break
        if self.category:
            f: dict[str, Any] = {"category": self.category, "k": 10, "in_stock": True}
            if self.brand:
                f["brand"] = self.brand
            if self.budget is not None:
                f["max_price"] = self.budget
            actions.append({"name": "filter", "args": f})
        if not actions:
            return
        result = self.call(actions)
        if self.done:
            return
        surfaced: list[str] = []
        for i, _a in enumerate(actions):
            surfaced.extend(self.result_rows(result, i))
        want = _title_tokens(title)

        def row_title(pid: str) -> str:
            return str(next((r.get("title") for (p, _k), r in self.rows.items() if p == pid), "") or "")

        def rank(pid: str) -> tuple:
            overlap = len(want & _title_tokens(row_title(pid)))
            return (overlap, pid in self.cat_pool)

        pool = [
            pid for pid in dict.fromkeys(surfaced)
            if pid != dead_pid and pid not in self.viewed and not self.title_accessory(row_title(pid))
        ]
        pool.sort(key=rank, reverse=True)
        if pool:
            self.call([{"name": "view", "args": {"product_id": t}} for t in pool[:5]])

    # ------------------------------------------------------------- ranking autopilot
    def ranking_autopilot(self) -> None:
        """Explicit candidate list: read every candidate in one turn, then commit by the rule."""
        keys = [k for k in self.explicit_candidates]
        if not keys:
            return
        pids = list(dict.fromkeys(pid for pid, _ in keys))
        reads = [{"name": "view", "args": {"product_id": pid}} for pid in pids]
        reads += [{"name": "inspect_stock", "args": {"product_id": pid, "sku": sku}} for pid, sku in keys]
        self.call(reads[: self.max_calls])
        for _round in range(3):
            if self.done or self.turns_left() < 3 or self.elapsed() > _SOFT_DEADLINE:
                return
            top = self.rank_top()
            if top is None:
                return
            tp, ts = top["key"]
            if self.first_add_done:
                # Post-event: re-read the replacement's identity and current state first.
                self.call([
                    {"name": "view", "args": {"product_id": tp}},
                    {"name": "inspect_stock", "args": {"product_id": tp, "sku": ts}},
                ])
                if self.done:
                    return
                top = self.rank_top()
                if top is None:
                    return
                tp, ts = top["key"]
            self.call([{"name": "add_to_cart", "args": {"product_id": tp, "sku": ts}}])
            if self.done:
                return
            self.verify_cart()
            if self.verified == (tp, ts):
                if self.pending_reply and self.turns_left() > 1:
                    self.pending_reply = None
                    self.messages_sent += 1
                    self.call([{"name": "message", "args": {"content": _ACK}}])
                self.call([{"name": "place_test_order", "args": {"product_id": tp, "sku": ts}}])
                return

    # ------------------------------------------------------------- recovery autopilot (TF6)
    def recovery_discover(self) -> bool:
        """Find the preferred model's listings: search the model code, view hits, then search the
        matching hit's exact title to surface its sibling-colour listings. Returns True when a
        level-0 (exact model + colour) candidate is known."""
        if not self.pref_model:
            return False
        q = f"{self.pref_model}"
        actions = []
        for query in (q, f"{self.pref_model} {self.category or ''}".strip()):
            if query.lower() in self.searches:
                continue
            self.searches.add(query.lower())
            a = {"query": query, "k": 10, "in_stock": True}
            if self.budget is not None:
                a["max_price"] = self.budget
            actions.append({"name": "search", "args": a})
        if self.category:
            # A model code like "635" also occurs inside unrelated titles ("635#Cotton Shorts"),
            # so pull a category-filtered pool as well: those rows are genuine candidates.
            f: dict[str, Any] = {"category": self.category, "k": 10, "in_stock": True}
            if self.budget is not None:
                f["max_price"] = self.budget
            actions.append({"name": "filter", "args": f})
        model_rows: list[str] = []
        if actions:
            result = self.call(actions)
            for i, a in enumerate(actions):
                if a["name"] == "search":
                    model_rows.extend(self.result_rows(result, i))
            model_rows = list(dict.fromkeys(model_rows))
        if self.done:
            return False
        hits = self.model_hits()
        if not hits:
            # The code often lives in facts.model, not in the title: view the code search's own
            # top hits so recovery_identity can read their model fact.
            hits = [pid for pid in model_rows if pid not in self.viewed][:6]
        if not hits:
            hits = [pid for pid in self.cat_pool if pid not in self.viewed][:6]
        to_view = [h for h in hits if h not in self.viewed][:8]
        if to_view:
            self.call([{"name": "view", "args": {"product_id": h}} for h in to_view])
        if self.done:
            return False
        # Sibling listings share the exact title: search it to reach the other colours.
        titles = []
        for pid in hits:
            model_ok, _color = self.recovery_identity(pid, next((k for (p, k) in self.rows if p == pid), ""))
            t = str((self.products.get(pid) or {}).get("title") or "")
            if model_ok and t and t not in titles:
                titles.append(t)
        actions = []
        for t in titles[:2]:
            if t.lower() in self.searches:
                continue
            self.searches.add(t.lower())
            a = {"query": t[:120], "k": 10, "in_stock": True}
            if self.budget is not None:
                a["max_price"] = self.budget
            actions.append({"name": "search", "args": a})
        if actions:
            self.call(actions)
            if self.done:
                return False
            wanted_titles = {t.lower() for t in titles}
            hit_set = set(self.model_hits())
            more = [
                pid for (pid, _k), row in self.rows.items()
                if pid not in self.viewed
                and (pid in hit_set or str(row.get("title") or "").lower() in wanted_titles)
            ]
            more = list(dict.fromkeys(more))[:8]
            if more:
                self.call([{"name": "view", "args": {"product_id": h}} for h in more])
        return any(self.recovery_level(p, k) == 0 and self.candidate_valid(p, k) for (p, k) in self.rows if p in self.viewed)

    def model_hits(self) -> list[str]:
        """Rows whose title really names the preferred model. The model token must stand as its
        own title token (a normalized substring test matches '635#Cotton Shorts' for '635'), and
        a short or all-digit code must additionally come from the category-filtered pool."""
        want = _norm_model(self.pref_model)
        if not want:
            return []
        # Only a SHORT code is ambiguous ("635" also opens "635#Cotton Shorts"); a long code,
        # digits or not, identifies the listing on its own.
        weak = len(want) <= 4
        out: list[str] = []
        for (pid, _k), row in self.rows.items():
            title = str(row.get("title") or "")
            tokens = re.findall(r"[a-z0-9]+", title.lower())
            joined = ["".join(tokens[i:j]) for i in range(len(tokens)) for j in range(i + 1, min(i + 4, len(tokens)) + 1)]
            if want not in joined:
                continue
            if weak and self.category and pid not in self.cat_pool and pid not in self.viewed:
                continue
            if pid in self.viewed and self.category_objection(pid):
                continue
            out.append(pid)
        return list(dict.fromkeys(out))

    def recovery_best(self) -> tuple[str, str] | None:
        cands = [((self.recovery_level(p, k), self.candidate_score(p, k)), (p, k)) for (p, k) in self.rows if p in self.viewed and self.candidate_valid(p, k) and self.recovery_level(p, k) is not None]
        if not cands:
            return None
        cands.sort(key=lambda c: (c[0][0], tuple(-x if isinstance(x, (int, float)) else 0 for x in c[0][1])))
        return cands[0][1]

    def recovery_autopilot(self) -> None:
        """Commit to the exact model+colour, then recover to the least-relaxed sibling, then order."""
        for _round in range(3):
            if self.done or self.turns_left() < 4 or self.elapsed() > _SOFT_DEADLINE:
                return
            best = self.recovery_best()
            if best is None:
                return
            bp, bk = best
            if self.first_add_done:
                self.call([{"name": "view", "args": {"product_id": bp}}, {"name": "inspect_stock", "args": {"product_id": bp, "sku": bk}}])
                if self.done:
                    return
                best = self.recovery_best()
                if best is None:
                    return
                bp, bk = best
            self.call([{"name": "add_to_cart", "args": {"product_id": bp, "sku": bk}}])
            if self.done:
                return
            self.verify_cart()
            if self.verified == (bp, bk):
                if self.pending_reply and self.turns_left() > 1:
                    self.pending_reply = None
                    self.messages_sent += 1
                    self.call([{"name": "message", "args": {"content": _ACK}}])
                self.call([{"name": "place_test_order", "args": {"product_id": bp, "sku": bk}}])
                return

    # ------------------------------------------------------------- preference autopilot
    def pref_discover(self) -> None:
        """TF4: LLM-free candidate discovery. The verifier scores the final item by regex over
        its visible facts + title on the shopper's axes, so the pool must contain listings that
        NAME every axis (IPX rating, sport words, an hours figure, wireless). Queries combine the
        category's product words with each axis's listing vocabulary; every full page is paged
        once more below its cheapest row (same string, lower cap: not a new query)."""
        self.extract_pref_axes()
        if not self.pref_axes or self.budget is None:
            return
        cat = (self.category or "").strip()
        profile = self.pref_profile()
        cat_words = [cat] + self.category_product_words()
        axis_words: list[list[str]] = []
        if profile is not None:
            # The profile's axes in the verifier's weight order, with the words listings use.
            axis_words = [list(profile["query"].get(name, [])) for name, _w, full, _h, _s in profile["axes"] if not str(full).startswith("brand:")]
        else:
            for a in self.pref_axes:
                if a["kind"] == "brand":
                    continue
                text = " ".join(a.get("words", [])) + " " + a.get("name", "")
                vocab: list[str] = []
                for pat, words in _AXIS_QUERY_WORDS:
                    if re.search(pat, text, re.I):
                        vocab.extend(words)
                if a["kind"] == "battery" and "hours" not in vocab:
                    vocab.extend(["hours", "battery"])
                axis_words.append(vocab or a.get("words", [])[:2])
        if not axis_words:
            return
        queries: list[str] = []
        firsts = [v[0] for v in axis_words if v]
        seconds = [v[1] if len(v) > 1 else v[0] for v in axis_words if v]
        # the whole axis vocabulary (BM25 rewards listings naming many axes), then every-axis
        # queries in several product-word spellings, then per-axis pairs
        all_vocab = " ".join(dict.fromkeys(w for v in axis_words for w in v))
        for cw in cat_words[:2]:
            queries.append(f"{cw} {all_vocab}")
        for cw in cat_words[:3]:
            queries.append(f"{cw} {' '.join(firsts)}")
        queries.append(f"{cat_words[-1]} {' '.join(seconds)}")
        queries.append(f"{' '.join(firsts[:2])} {cat_words[1] if len(cat_words) > 1 else cat}")
        for i, v in enumerate(axis_words):
            if i >= 4:
                break
            others = [w[0] for j, w in enumerate(axis_words) if j != i and w][:2]
            queries.append(f"{cat} {' '.join(v[:2])} {' '.join(others)}")
        seen_q: set[str] = set()
        queries = [q.strip() for q in queries if q.strip() and not (q.lower() in seen_q or seen_q.add(q.lower()))]
        # turn 1: every query at the budget cap; turn 2: one deeper window per full page
        pending = [(q, self.budget) for q in queries]
        for _turn in range(2):
            if self.done or not pending or self.turns_left() <= _FIRST_ADD_RESERVE + 4:
                return
            actions = []
            for q, cap in pending[: self.max_calls]:
                self.searches.add(q.lower())
                actions.append({"name": "search", "args": {"query": q, "k": 10, "in_stock": True, "max_price": cap}})
            result = self.call(actions) or {}
            pending = []
            for a, cr in zip(actions, result.get("calls") or []):
                obs = cr.get("observation") or {}
                inner = obs.get("observation") if isinstance(obs.get("observation"), dict) else obs
                rows = [r for r in ((inner.get("results") if isinstance(inner, dict) else None) or []) if isinstance(r, dict)]
                prices = [p for p in (_num(r.get("price")) for r in rows) if p is not None]
                if len(rows) >= 10 and prices:
                    cap = round(min(prices) - 0.01, 2)
                    if cap > 0:
                        pending.append((a["args"]["query"], cap))

    def category_product_words(self) -> list[str]:
        """Words listings use for the required category: the profile's device signals, then the
        leaf words of category paths already observed (data-driven, no fixed table)."""
        out: list[str] = []
        profile = self.pref_profile()
        if profile is not None:
            out.extend(profile["device"])
        cat = (self.category or "").lower()
        out.extend(_CATEGORY_QUERY_WORDS.get(cat, []))
        counts: dict[str, int] = {}
        for product in self.products.values():
            path = [str(x) for x in (product.get("category_path") or [])]
            if path and _category_matches(self.category, path, product.get("title")):
                leaf = path[-1].lower()
                counts[leaf] = counts.get(leaf, 0) + 1
        out.extend(w for w, _n in sorted(counts.items(), key=lambda kv: -kv[1]))
        seen: set[str] = set()
        return [w for w in out if w.lower() != cat and not (w.lower() in seen or seen.add(w.lower()))][:4]

    def preference_autopilot(self) -> None:
        """TF4 end to end without the model: discover, view the best-previewed candidates,
        first add (a strong runner-up when the best listing has no equal elsewhere), verify,
        recover to the best listing, order."""
        self.pref_discover()
        if self.done or not self.pref_axes:
            return
        for _ in range(2):
            if self.done or self.turns_left() <= _FIRST_ADD_RESERVE + 2:
                break
            before = len(self.viewed)
            self.view_top_candidates(limit=self.max_calls)
            if len(self.viewed) == before:
                break
        for _round in range(3):
            if self.done or self.turns_left() < 3 or self.elapsed() > _HARD_DEADLINE:
                return
            self.force_finish()
            if self.done:
                return
            if self.best_fallback() is None:
                self.view_top_candidates(limit=6)

    # ------------------------------------------------------------- retrieval autopilot
    def phrase_rows(self) -> list[tuple[tuple[str, str], dict]]:
        """Rows whose title carries every requested term, in stock and within budget."""
        out = []
        for key, row in self.rows.items():
            if row.get("in_stock") is not True or not self.price_ok(_num(row.get("price"))):
                continue
            if not self.phrase <= _title_tokens(str(row.get("title") or "")):
                continue
            out.append((key, row))
        return out

    def recall_surface(self) -> None:
        """Surface as many distinct matching identities as the read interface allows.

        The verifier's recall counts identities that appeared in ANY observation before the
        order, precision counts only what was viewed/compared/carted, and the query penalty
        counts DISTINCT search strings. A search returns the top-10 BM25 hits priced at or
        under max_price, so re-issuing the same string with a lower cap pages through further
        hits for free; brand filters are not queries at all. Three query strings: the phrase,
        then the phrase plus the two title words most common among the matches."""
        if not self.phrase_text or not self.phrase:
            return
        q0 = self.phrase_text
        queries: list[str] = [q0]
        issued: set[tuple[str, str, float | None]] = set()
        pending: dict[tuple[str, str], list[float | None]] = {}   # key -> caps still to issue
        seen_pids: set[str] = set()
        stale = 0

        def brands() -> list[str]:
            count: dict[str, set[str]] = {}
            for (pid, _sku), row in self.phrase_rows():
                b = str(row.get("brand") or "").strip()
                if not b or b.lower() in ("none", "null", "no brand", "no brands", "n/a"):
                    continue
                count.setdefault(b, set()).add(pid)
            return [b for b, _ in sorted(count.items(), key=lambda kv: (-len(kv[1]), kv[0]))]

        def cotokens() -> list[str]:
            """Two reformulation tails: the words of the dominant sub-family of matches, then the
            words of the matches that sub-family does not cover. A multi-word tail pulls BM25
            toward that sub-family's sibling listings (other sellers, sizes, bundles) - measured
            on the sealed pools, single-word tails reach far fewer identities."""
            by_tok: dict[str, set[str]] = {}
            for (pid, _sku), row in self.phrase_rows():
                for t in _title_tokens(str(row.get("title") or "")) - self.phrase:
                    if len(t) < 3 or t.isdigit() or t in _COTOKEN_STOP:
                        continue
                    if any(p.startswith(t) or t.startswith(p) for p in self.phrase):
                        continue
                    by_tok.setdefault(t, set()).add(pid)
            if not by_tok:
                return []
            freq: dict[str, int] = {}
            for row in self.rows.values():
                for t in _title_tokens(str(row.get("title") or "")):
                    freq[t] = freq.get(t, 0) + 1
            rare = lambda t: freq.get(t, 0)
            top = max(by_tok, key=lambda x: (len(by_tok[x]), -rare(x), x))
            covered = by_tok[top]
            g1 = [t for t, s_ in sorted(by_tok.items(), key=lambda kv: (-len(kv[1] & covered), rare(kv[0]), kv[0])) if s_ & covered][:6]
            unc = set().union(*by_tok.values()) - covered
            g2 = [t for t, s_ in sorted(by_tok.items(), key=lambda kv: (-len(kv[1] & unc), -len(kv[1]), rare(kv[0]), kv[0])) if s_ & unc][:6]
            if not g2:
                g2 = [t for t in sorted(by_tok, key=lambda x: (-len(by_tok[x]), rare(x), x)) if t not in g1][:6]
            return [" ".join(g1), " ".join(g2)]

        def queue(key: tuple[str, str], caps: list[float | None]) -> None:
            lst = pending.setdefault(key, [])
            for c in caps:
                if (key[0], key[1], c) not in issued and c not in lst and (c is None or c > 0):
                    lst.append(c)

        def run() -> bool:
            actions = []
            order = [("search", q) for q in queries] + [("filter", b) for b in (brands() if self.category else [])]
            for key in order:
                caps = pending.get(key) or []
                # Deepest window first: it is the one guaranteed to reach unseen rows.
                for c in sorted(caps, key=lambda x: (x is not None, x)):
                    if len(actions) >= self.max_calls:
                        break
                    caps.remove(c)
                    issued.add((key[0], key[1], c))
                    if key[0] == "search":
                        args: dict[str, Any] = {"query": key[1], "k": 10, "in_stock": True}
                        if c is not None:
                            args["max_price"] = c
                        self.searches.add(key[1].lower())
                    else:
                        args = {"category": self.category, "brand": key[1], "k": 10, "in_stock": True, "max_price": c}
                    actions.append({"name": key[0], "args": args})
            if not actions:
                return False
            result = self.call(actions) or {}
            for a, cr in zip(actions, result.get("calls") or []):
                obs = cr.get("observation") or {}
                inner = obs.get("observation") if isinstance(obs.get("observation"), dict) else obs
                rows = inner.get("results") if isinstance(inner, dict) else None
                rows = [r for r in (rows or []) if isinstance(r, dict)]
                key = (a["name"], a["args"].get("query") or a["args"].get("brand"))
                cap = _num(a["args"].get("max_price"))
                prices = sorted(p for p in (_num(r.get("price")) for r in rows) if p is not None)
                if len(rows) < 10 or not prices:
                    continue   # every row at or under this cap has been listed
                # A full page hides further hits behind it: open a window below its cheapest row,
                # and (for the first page of a key) two more windows through the middle and top.
                cuts = [prices[0]] if cap is not None and cap < (self.budget or float("inf")) else [prices[0], prices[len(prices) // 2], prices[-1]]
                queue(key, [round(c - 0.01, 2) for c in cuts if cap is None or c - 0.01 < cap])
            return True

        queue(("search", q0), [self.budget])
        for _turn in range(9):
            if self.done or self.turns_left() <= 12 or self.elapsed() > _SOFT_DEADLINE:
                break
            if len(queries) < 3 and issued:
                extra = cotokens()
                if self.category:
                    extra.append(self.category.split()[-1].lower())
                for t in extra:
                    q = f"{q0} {t}".strip()
                    if t and q not in queries and t not in self.phrase:
                        queries.append(q)
                    if len(queries) >= 3:
                        break
            for q in queries:
                if ("search", q, self.budget) not in issued:
                    queue(("search", q), [self.budget])
            if self.category:
                for b in brands():
                    if ("filter", b, self.budget) not in issued:
                        queue(("filter", b), [self.budget])
            before = len(seen_pids)
            if not run():
                break
            seen_pids = {pid for (pid, _sku), _row in self.phrase_rows()}
            stale = stale + 1 if len(seen_pids) == before else 0
            if len(seen_pids) >= 24 or stale >= 3:
                break
        # Keep the model context small if the model ever has to continue from here.
        for m in self.messages:
            if m.get("role") == "tool" and len(str(m.get("content") or "")) > 1500 and '"results"' in str(m.get("content")):
                try:
                    obs = json.loads(m["content"])
                except Exception:
                    continue
                rows = [r for r in (obs.get("results") or []) if isinstance(r, dict) and self.phrase <= _title_tokens(str(r.get("title") or ""))]
                m["content"] = json.dumps({"action": obs.get("action"), "matching_results": [
                    {k: r.get(k) for k in ("product_id", "sku", "price", "in_stock", "brand", "title")} for r in rows
                ]})

    def recall_autopilot(self) -> None:
        """TF2 end to end: surface the pool, then view/add/verify/recover/order without the model."""
        self.recall_surface()
        for _round in range(3):
            if self.done or self.turns_left() < 3 or self.elapsed() > _HARD_DEADLINE:
                return
            self.force_finish()
            if self.done:
                return
            if self.best_fallback() is None:
                self.view_top_candidates(limit=2)

    # ------------------------------------------------------------- generic autopilot
    def gather_candidates(self) -> None:
        """LLM-free candidate discovery: goal-text searches with a budget cap + a category/brand filter."""
        goal = self.query.split("\n\nTask rules:", 1)[0]
        queries: list[str] = []
        if self.phrase:
            queries.append(" ".join(sorted(self.phrase)))
        core = []
        for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]*", goal):
            t = t.strip(".,;:+-")
            if not t or t.lower() in _GOAL_STOP or re.fullmatch(r"\d+(?:\.\d+)?", t) or t.upper() in _CURRENCIES.split("|"):
                continue
            core.append(t)
        if self.brand and self.brand.lower() not in " ".join(core).lower():
            core.insert(0, self.brand)
        if core:
            queries.append(" ".join(core[:8]))
        if self.category:
            spec = f" {self.wanted[1].replace('in', ' inch').replace('hz', 'Hz').replace('gb', 'GB')}" if self.wanted else ""
            queries.append(f"{self.brand + ' ' if self.brand else ''}{self.category}{spec}".strip())
        actions: list[dict] = []
        for q in queries:
            if len(self.searches) >= self.search_cap() or q.lower() in self.searches:
                continue
            self.searches.add(q.lower())
            a = {"query": q, "k": 10, "in_stock": True}
            if self.budget is not None:
                a["max_price"] = self.budget
            actions.append({"name": "search", "args": a})
        if self.category:
            f: dict[str, Any] = {"category": self.category, "k": 10, "in_stock": True}
            if self.brand:
                f["brand"] = self.brand
            if self.budget is not None:
                f["max_price"] = self.budget
            actions.append({"name": "filter", "args": f})
        if actions:
            self.call(actions[: self.max_calls])

    def axis_searches(self) -> None:
        """TF4: search the category together with the shopper's own preference phrases (the most
        repeated content bigrams of their reply), so the pool contains candidates for every axis."""
        text = re.sub(r"[^a-z0-9 ]+", " ", (self.sealed or "").lower())
        words = [w for w in text.split() if w not in _GOAL_STOP and len(w) > 2 and not w.isdigit()]
        stop2 = {"counts", "most", "least", "next", "close", "behind", "several", "together", "beat", "any", "single",
                 "one", "that", "than", "with", "while", "long", "hard", "fully", "there", "device", "actual",
                 "accessory", "not", "required", "catalog", "category", "budget", "stock", "under"}
        words = [w for w in words if w not in stop2]
        bigrams: dict[str, int] = {}
        for a, b in zip(words, words[1:]):
            bigrams[f"{a} {b}"] = bigrams.get(f"{a} {b}", 0) + 1
        singles: dict[str, int] = {}
        for w in words:
            singles[w] = singles.get(w, 0) + 1
        phrases = [bg for bg, n in sorted(bigrams.items(), key=lambda kv: -kv[1]) if n >= 2][:4]
        for w, n in sorted(singles.items(), key=lambda kv: -kv[1]):
            if len(phrases) >= 7:
                break
            if n >= 2 and not any(w in ph for ph in phrases):
                phrases.append(w)
        cat = self.category or ""
        self.extract_pref_axes()
        if self.pref_axes:
            combined = " ".join(sorted({w for a in self.pref_axes for w in a.get("words", []) if a["kind"] != "brand"}))[:80]
            top2 = " ".join(sorted({w for a in self.pref_axes[:2] for w in a.get("words", [])}))[:60]
            phrases = [combined, top2] + phrases
        actions = []
        for ph in phrases:
            q = f"{cat} {ph}".strip()
            if q.lower() in self.searches:
                continue
            self.searches.add(q.lower())
            a = {"query": q, "k": 10, "in_stock": True}
            if self.budget is not None:
                a["max_price"] = self.budget
            actions.append({"name": "search", "args": a})
        if actions:
            self.call(actions[: self.max_calls])

    def view_top_candidates(self, limit: int = 8) -> None:
        """View the most promising unviewed rows in one turn so validity checks can run."""
        seen: list[str] = []
        def row_key(kv):
            (pid, sku), row = kv
            if self.family == "preference_reasoning" and self.pref_axes:
                # Title-only preview of the preference score for rows not yet viewed.
                saved = self.products.get(pid)
                self.products[pid] = {"title": row.get("title"), "facts": {}}
                sc = self.pref_score(pid)[0]
                if saved is None:
                    del self.products[pid]
                else:
                    self.products[pid] = saved
                return (sc, self.candidate_score(pid, sku))
            return (0, self.candidate_score(pid, sku))

        rows = sorted(self.rows.items(), key=row_key, reverse=True)
        for (pid, _sku), row in rows:
            if pid in self.viewed or pid in seen:
                continue
            if row.get("in_stock") is False or not self.price_ok(_num(row.get("price"))):
                continue
            if self.phrase_mismatch(pid):
                continue
            seen.append(pid)
            if len(seen) >= limit:
                break
        if seen:
            self.call([{"name": "view", "args": {"product_id": pid}} for pid in seen[: self.max_calls]])

    def autopilot(self) -> None:
        """Finish the episode without the model: discover, view, commit, recover, order."""
        if self.done:
            return
        if not self.first_add_done:
            if not self.rows or self.best_fallback() is None:
                self.gather_candidates()
            if self.best_fallback() is None and not self.done:
                self.view_top_candidates()
            if self.best_fallback() is None and not self.done and self.turns_left() > 8:
                self.gather_candidates()
                self.view_top_candidates()
        for _round in range(3):
            if self.done or self.turns_left() < 3:
                return
            self.force_finish()
            if self.done:
                return
            if self.best_fallback() is None:
                self.view_top_candidates(limit=6)

    # ------------------------------------------------------------- forced end
    def candidate_valid(self, pid: str, sku: str, *, original: bool = False) -> bool:
        """Deterministic validity: in stock, within budget, category/device/spec/cheap checks pass."""
        row = self.rows.get((pid, sku)) or {}
        if (pid, sku) in self.dead or row.get("in_stock") is not True:
            return False
        price = _num(row.get("price"))
        if not self.price_ok(price, original=original):
            return False
        if self.family != "retrieval_recall" and self.budget is not None and price is not None and price < 0.03 * self.budget:
            return False
        if pid in self.products:
            saved = dict(self.cat_rejects)
            bad = self.category_objection(pid) or self.device_objection(pid)
            self.cat_rejects = saved
            if bad:
                return False
        elif self.category and self.family != "retrieval_recall" and self.title_accessory(str(row.get("title") or "")):
            return False
        if self.phrase_mismatch(pid):
            return False
        if self.wanted and self.family not in ("recovery", "retrieval_recall", "preference_reasoning"):
            if self.variant_spec(self.wanted[0], pid, sku) != self.wanted[1]:
                return False
        if self.family == "preference_reasoning" and self.listing.get(pid) in self.dead_listing:
            return False
        if self.family == "recovery" and pid in self.products and not _recovery_admissible(
            self.category, self.products[pid], str(row.get("options") or "")
        ):
            return False
        if self.brand and self.family != "retrieval_recall" and pid in self.products and not self.brand_matches(self.brand, pid, strict=True):
            return False
        return True

    def candidate_score(self, pid: str, sku: str) -> tuple:
        """Rank valid candidates: goal/brand overlap in title, viewed, then price closeness to budget."""
        row = self.rows.get((pid, sku)) or {}
        product = self.products.get(pid) or {}
        title = str(product.get("title") or row.get("title") or "")
        toks = _title_tokens(title)
        overlap = len(toks & self.goal_tokens)
        brand_ok = 1 if self.brand and _title_tokens(self.brand) <= (toks | _title_tokens(str(row.get("brand") or product.get("brand") or ""))) else 0
        price = _num(row.get("price")) or 0.0
        rel = price / self.budget if self.budget else 0.0
        return (brand_ok, overlap, pid in self.viewed, -abs(0.6 - rel))

    def best_fallback(self) -> tuple[str, str] | None:
        """Best known valid variant; never simply the cheapest row (cheap rows are parts)."""
        if self.family == "ranking":
            ranked = [r for r in self.ranked_candidates() if (self.rows.get(r["key"]) or {}).get("in_stock") is True]
            if ranked:
                return ranked[0]["key"]
        if self.family == "preference_reasoning" and not self.first_add_done and self.pref_axes:
            first, _planned = self.tf4_first_pick()
            if first is not None and first[0] not in self.fallback_rejected:
                return first
        valid = [(pid, sku) for (pid, sku) in self.rows if pid not in self.fallback_rejected and self.candidate_valid(pid, sku)]
        if not valid:
            return None
        # Unviewed rows have not passed the category/device checks: prefer viewed ones, and
        # among TF1 candidates honour the preferred brand.
        def level_key(k: tuple[str, str]) -> int:
            if self.family != "recovery" or not self.pref_model:
                return 0
            lv = self.recovery_level(*k)
            return -(lv if lv is not None else 9)

        valid.sort(key=lambda k: (k[0] in self.viewed, self.brand_matches(self.pref_brand, k[0]) if self.pref_brand else True, level_key(k), self.pref_score(k[0])[0] if self.family == "preference_reasoning" else 0, self.candidate_score(*k)), reverse=True)
        return valid[0]

    def force_finish(self) -> None:
        """Order the verified item, or the best fallback, within the remaining turns."""
        if self.done:
            return
        if self.verified and self.verified in self.cart and not self.pending_reply:
            pid, sku = self.verified
            args: dict[str, Any] = {"product_id": pid, "sku": sku}
            just = self.normalize_justification(None, pid, sku)
            if just is not None:
                args["justification"] = just
            self.call([{"name": "place_test_order", "args": args}])
            return
        if self.pending_reply and self.turns_left() > 2:
            self.pending_reply = None
            self.messages_sent += 1
            self.call([{"name": "message", "args": {"content": _ACK}}])
            if self.verified and self.verified in self.cart:
                return self.force_finish()
        if self.turns_left() < 3:
            return
        cand = None
        for _try in range(4):
            cand = self.best_fallback()
            if cand is None or cand[0] in self.viewed or self.turns_left() < 4:
                break
            # View first so the category/device checks can run on the real listing.
            self.call([{"name": "view", "args": {"product_id": cand[0]}}])
            if self.done:
                return
            if not self.candidate_valid(*cand):
                self.fallback_rejected.add(cand[0])
                cand = None
                continue
            break
        if cand is None:
            return
        pid, sku = cand
        actions = [{"name": "remove_from_cart", "args": {"product_id": c[0], "sku": c[1]}} for c in self.cart]
        # Evidence gates: the item must be viewed before the add, and (after an event) its
        # current price/stock observed after the event.
        if pid not in self.viewed or self.first_add_done:
            actions.append({"name": "view", "args": {"product_id": pid}})
        if self.first_add_done:
            actions.append({"name": "inspect_stock", "args": {"product_id": pid, "sku": sku}})
        actions.append({"name": "add_to_cart", "args": {"product_id": pid, "sku": sku}})
        self.call(actions)
        self.verify_cart()
        if self.verified == (pid, sku) and self.turns_left() >= 1:
            args = {"product_id": pid, "sku": sku}
            just = self.normalize_justification(None, pid, sku)
            if just is not None:
                args["justification"] = just
            if self.pending_reply and self.turns_left() > 1:
                self.pending_reply = None
                self.call([{"name": "message", "args": {"content": _ACK}}])
            self.call([{"name": "place_test_order", "args": args}])


def _chat(session: _Session) -> dict | None:
    """One model turn. Falls back to other allowlisted models when the primary keeps failing."""
    models = _model_candidates()
    for _attempt in range(2):
        model = models[min(session.model_index, len(models) - 1)]
        inference = _proxy.post(
            "/inference/chat/completions",
            json_data={
                "model": model,
                "messages": session.messages,
                "tools": session.tools,
                "tool_choice": "required",
                "temperature": 0,
            },
        )
        if not inference:
            # Provider trouble on this validator: try the next model on the next attempt.
            session.model_index = min(session.model_index + 1, len(models) - 1)
            if session.elapsed() > _SOFT_DEADLINE:
                break
            continue
        try:
            assistant = inference["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            continue
        if assistant.get("tool_calls"):
            session.llm_failures = 0
            return assistant
        session.messages.append({"role": "assistant", "content": assistant.get("content") or ""})
        session.messages.append(
            {"role": "user", "content": "The environment has not reported done=true. Choose one of the supplied tools to continue."}
        )
    session.llm_failures += 1
    if session.llm_failures >= _LLM_FAILURES_BEFORE_AUTOPILOT:
        session.llm_dead = True
    return None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    s = _Session(problem_data)
    hint = _FAMILY_HINTS.get(s.family, "")
    s.messages = [
        {"role": "system", "content": _SYSTEM + ("\nTASK-SPECIFIC GUIDANCE\n" + hint if hint else "")},
        {"role": "user", "content": s.query},
    ]
    s.dialogue.append({"role": "user", "content": s.query})

    # Turn 1: the public route to the sealed requirements (budget, category, brand, spec, priorities).
    s.messages_sent += 1
    s.call([{"name": "message", "args": {"content": _CLARIFY}}])
    s.pending_reply = None
    if s.family == "ranking" and s.explicit_candidates:
        s.ranking_autopilot()
    if s.family == "retrieval_recall" and s.phrase_text and not s.done:
        s.recall_autopilot()
    if s.family == "preference_reasoning" and s.sealed and not s.done:
        s.preference_autopilot()
    if s.family == "recovery" and s.pref_model and not s.done:
        # Deterministic discovery of the preferred model's listings; the autopilot commits only
        # when an exact model+colour match is known, otherwise the model continues from here.
        if s.recovery_discover():
            s.recovery_autopilot()
        elif not s.done:
            s.note("Controller discovery: the preferred model's listings found so far are in this conversation "
                   "(view results above). Compare their model facts and variant colours before committing.")

    while not s.done and s.turn < s.max_steps:
        if s.elapsed() > _HARD_DEADLINE:
            break
        if s.llm_dead:
            s.autopilot()
            break
        if s.elapsed() > _SOFT_DEADLINE or s.turns_left() <= _RESERVE_TURNS:
            s.force_finish()
            if s.done or s.elapsed() > _SOFT_DEADLINE or s.turns_left() <= 1:
                break
        # Turn-budget guards: the first add must leave room for the event + recovery, and a
        # recovery must leave room to order. Past the deadline the controller commits itself.
        if not s.first_add_done and s.turns_left() <= _FIRST_ADD_RESERVE and s.forced_commits < 2:
            s.forced_commits += 1
            if s.best_fallback() is None:
                s.view_top_candidates()
            s.force_finish()
            if s.done:
                break
            if s.verified:
                continue
            s.note("The turn budget is nearly spent: add the best valid candidate NOW, or order the verified item.")
        elif s.first_add_done and s.verified is None and s.turns_left() <= _RECOVERY_RESERVE:
            s.force_finish()
            if s.done or s.turns_left() <= 1:
                break
        # Preference tasks are graded on the first add: make sure the audit sees a real pool.
        if (
            s.family == "preference_reasoning" and not s.first_add_done and not s.pool_widened
            and s.turn >= 4 and len(s.viewed) < 8 and s.turns_left() > _FIRST_ADD_RESERVE + 2
        ):
            s.pool_widened = True
            s.axis_searches()
            if s.done:
                break
            s.view_top_candidates(limit=8)
            if s.done:
                break
            s.note("The controller viewed additional in-budget candidates so the preference audit covers a wider pool. Compare them before the first add.")
        assistant = _chat(s)
        if assistant is None:
            if s.llm_dead or s.elapsed() > _SOFT_DEADLINE:
                s.autopilot()
                break
            continue
        tool_calls = list(assistant.get("tool_calls") or [])
        content = assistant.get("content") or ""
        actions, kept, rejected = s.guard(tool_calls)
        if rejected and not actions:
            s.strikes += 1
        else:
            s.strikes = 0
        if s.dup_rejects >= 3 and not actions:
            # The model is looping on identical reads: take over the next step.
            s.dup_rejects = 0
            if s.best_fallback() is None:
                s.view_top_candidates()
            s.force_finish()
            if s.done:
                break
            s.note("Repeated identical reads were blocked; the controller advanced the protocol. Continue from the current cart state.")
            continue
        if s.pending_reply and actions and actions[0]["name"] != "message":
            # A delivered shopper question needs a reply before the order; fold one in.
            reply_tc = {
                "id": f"ctl-reply-{s.turn + 1}",
                "type": "function",
                "function": {"name": "message", "arguments": json.dumps({"content": _ACK})},
            }
            actions.insert(0, {"name": "message", "args": {"content": _ACK}})
            kept.insert(0, reply_tc)
            s.messages_sent += 1
            s.pending_reply = None
        # Record the model turn with exactly the calls that will run, plus rejection feedback.
        all_calls = kept + [tc for tc, _ in rejected]
        s.messages.append({"role": "assistant", "content": content, "tool_calls": all_calls})
        for tc, reason in rejected:
            s.messages.append(
                {"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps({"error": "rejected: " + reason})}
            )
        if rejected:
            s.dialogue.append({"role": "assistant", "content": content, "rejected_tool_calls": [
                {"name": (tc.get("function") or {}).get("name"), "arguments": (tc.get("function") or {}).get("arguments"), "reason": reason}
                for tc, reason in rejected
            ]})
        if actions:
            s.call(actions, tool_calls=kept)
            if s.pending_add is not None and not s.done:
                s.verify_cart()
        elif s.strikes >= 4:
            s.force_finish()
            if not s.done:
                s.strikes = 0
                s.note("Take a valid next step: view a candidate, add it, or order the verified item.")
    if not s.done and s.turn < s.max_steps and s.elapsed() < _HARD_DEADLINE + 30:
        s.autopilot()
    return s.dialogue


__all__ = ["agent_main"]
