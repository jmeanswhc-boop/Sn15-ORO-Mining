"""ORO Bench hybrid shopping agent.

Python parses the goal, builds catalog queries, ranks, recovers, and grounds
claims. A third-party chat model drives the tool loop with those strategies
in the system playbook, plus per-turn coaches. Planner fallback if inference
fails. Tools always come from policy_view.
"""

from __future__ import annotations

import json
import math
import re
from os import getenv
from typing import Any


_FAST_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_STRONG_MODELS = {
    "chutes": "Qwen/Qwen3.5-397B-A17B-TEE",
    "openrouter": "qwen/qwen3.5-397b-a17b",
}
_HEAVY_MODELS = {
    "chutes": "moonshotai/Kimi-K2.6-TEE",
    "openrouter": "moonshotai/kimi-k2.6",
}
_TO_OPENROUTER = {
    "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
    "deepseek-ai/DeepSeek-V3.1-TEE": "deepseek/deepseek-chat-v3.1",
    "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
    "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
    "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
    "zai-org/GLM-5.1-TEE": "z-ai/glm-5.1",
    "zai-org/GLM-5.2-TEE": "z-ai/glm-5.2",
    "moonshotai/Kimi-K2.6-TEE": "moonshotai/kimi-k2.6",
    "openai/gpt-oss-120b-TEE": "openai/gpt-oss-120b",
}
_TO_CHUTES = {slug: tee for tee, slug in _TO_OPENROUTER.items()}
_RESERVE_OPENROUTER = (
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "google/gemma-4-31b-it",
    "deepseek/deepseek-v3.2",
)
_RESERVE_CHUTES = (
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "google/gemma-4-31B-turbo-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
)
LANE_INTENT = "intent_decomposition"
LANE_RECALL = "retrieval_recall"
LANE_CONSTRAINT = "constraint_satisfaction"
LANE_PREFERENCE = "preference_reasoning"
LANE_RANK = "ranking"
LANE_RECOVER = "recovery"
LANE_JUSTIFY = "justification"

_SEARCH_NAMES = (
    "search",
    "find_product",
    "search_catalog",
    "search_products",
)
_FILTER_NAMES = ("filter", "filter_catalog", "catalog_filter")
_VIEW_NAMES = (
    "view",
    "view_product_information",
    "view_product",
    "get_product",
    "get_products",
    "lookup_product",
    "product_details",
)
_COMPARE_NAMES = ("compare", "compare_products", "product_compare")
_INSPECT_NAMES = ("inspect_stock", "check_stock", "get_stock")
_CART_INSPECT_NAMES = ("inspect_cart", "view_cart", "get_cart")
_CART_NAMES = ("add_to_cart", "add_item", "cart_add")
_REMOVE_NAMES = ("remove_from_cart", "cart_remove", "remove_item")
_ORDER_NAMES = (
    "place_test_order",
    "place_order",
    "checkout",
    "complete_order",
)
_MESSAGE_NAMES = ("message", "talk", "respond", "say", "reply")
_CLAIM_FIELDS = (
    "brand",
    "category",
    "in_stock",
    "price",
    "refresh_rate",
    "resolution",
    "screen_size",
)
_JUSTIFY_CLAIM_FIELDS = (
    "brand",
    "in_stock",
    "price",
    "refresh_rate",
    "resolution",
    "screen_size",
)
_FORBIDDEN_CLAIM_FIELDS = {
    "display",
    "size",
    "hdr",
    "category",
    "category_path",
    "device_operating_system",
    "os",
}

_ARG_ALIASES = {
    "query": ("query", "q", "search", "text", "keywords"),
    "k": ("k", "limit", "n", "top_k", "page_size"),
    "in_stock": ("in_stock", "inStock", "available"),
    "product_id": ("product_id", "productId", "id"),
    "sku": ("sku", "sku_id", "variant", "variant_id"),
    "content": ("content", "message", "text", "utterance"),
    "shop_id": ("shop_id", "shopId", "store_id"),
    "price": ("price", "price_range"),
    "product_ids": ("product_ids", "ids"),
    "page": ("page",),
    "justification": ("justification",),
    "max_price": ("max_price", "price_max", "budget"),
    "brand": ("brand",),
    "category": ("category",),
}

_PRICE_FROM_TO = re.compile(
    r"(?:from|between)\s+(\d+(?:\.\d+)?)\s+(?:to|-|and)\s+(\d+(?:\.\d+)?)",
    re.I,
)
_PRICE_UNDER = re.compile(
    r"(?:under|below|at most|max(?:imum)?|upto|up to|<=)\s*(\d+(?:\.\d+)?)",
    re.I,
)
_PRICE_ABOVE = re.compile(
    r"(?:above|over|at least|min(?:imum)?|>=)\s*(\d+(?:\.\d+)?)",
    re.I,
)
_BUDGET_NOW = re.compile(
    r"budget[^0-9]{0,24}(\d+(?:\.\d+)?)",
    re.I,
)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.I | re.S)
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
_VARIANT_KEY = re.compile(r"(\d{6,})::([A-Za-z0-9._-]+)")
_REFRESH_HZ = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hz|hertz)\b", re.I)
_INCHES = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(?:\"|''|in(?:ch(?:es)?)?)\b",
    re.I,
)
_SHOPPER_BUDGET = re.compile(
    r"(?:budget[^0-9]{0,32}|can only go up to|only go up to|new budget|"
    r"balance.{0,24}up to)\s*(\d+(?:\.\d+)?)",
    re.I,
)
_PRICE_HIKE = re.compile(
    r"(?:from|was)\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)",
    re.I,
)
_INTENT_PREFIX = re.compile(
    r"^(?:can you\s+|please\s+)?(?:i(?:'d|\s+would)?\s+)?"
    r"(?:(?:find(?:\s+and)?\s+)?(?:place|submit)(?:\s+one)?(?:\s+test)?\s+order\s+for|"
    r"need|want|find(?:\s+me)?|get(?:\s+me)?|looking for|shop for|buy|purchase)\s+",
    re.I,
)
_CLAUSE_CUT = re.compile(
    r"\b(?:at or under|at or below|under|below|priced|in stock|,|\.|!|\?|"
    r"and it has|walk me|tell me|give me)\b",
    re.I,
)
_PRODUCT_NOUNS = (
    "smartphones",
    "smartphone",
    "headphones",
    "headphone",
    "earphones",
    "earphone",
    "earbuds",
    "earbud",
    "headsets",
    "headset",
    "tablets",
    "tablet",
    "monitors",
    "monitor",
    "laptops",
    "laptop",
    "notebook",
    "cameras",
    "camera",
    "watches",
    "watch",
    "speakers",
    "speaker",
    "keyboard",
    "mouse",
    "phones",
    "phone",
    "helmet",
    "television",
    "console",
    "router",
    "printer",
    "charger",
    "cable",
    "chair",
    "desk",
    "tv",
)
_KIND_CANON = {
    "smartphones": "smartphone",
    "phones": "smartphone",
    "phone": "smartphone",
    "tablets": "tablet",
    "monitors": "monitor",
    "headphones": "headphone",
    "earphones": "headphone",
    "earphone": "headphone",
    "earbuds": "headphone",
    "earbud": "headphone",
    "headsets": "headphone",
    "headset": "headphone",
    "laptops": "laptop",
    "notebook": "laptop",
    "cameras": "camera",
    "watches": "watch",
    "speakers": "speaker",
    "televisions": "tv",
    "television": "tv",
    "tvs": "tv",
}
_KIND_CATEGORY = {
    "smartphone": "Smartphones",
    "tablet": "Tablets",
    "monitor": "Monitors",
    "headphone": "Headphones",
    "laptop": "Laptops",
    "camera": "Cameras",
    "watch": "Watches",
    "speaker": "Speakers",
    "tv": "Televisions",
}
_STOPWORDS = {
    "a",
    "an",
    "the",
    "in",
    "on",
    "for",
    "with",
    "and",
    "or",
    "me",
    "my",
    "one",
    "some",
    "any",
    "please",
    "stock",
    "in-stock",
    "instock",
    "need",
    "want",
    "find",
    "get",
    "can",
    "you",
    "looking",
    "listing",
    "listings",
    "category",
    "matches",
    "match",
    "phrase",
    "that",
    "this",
    "which",
    "buy",
    "purchase",
    "place",
    "order",
    "test",
    "an",
}
_MODIFIERS = {
    "gaming",
    "wireless",
    "curved",
    "portable",
    "original",
    "new",
    "smart",
    "full",
    "mini",
    "pro",
    "max",
    "lite",
    "plus",
    "unlocked",
    "global",
    "digital",
    "led",
    "oled",
}
_SPEC_ADJECTIVES = {
    "full",
    "ultra",
    "super",
    "quad",
    "true",
    "native",
}
_COLORS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "silver",
    "gold",
    "grey",
    "gray",
    "pink",
    "purple",
    "orange",
    "yellow",
    "navy",
    "beige",
    "brown",
    "graphite",
    "midnight",
}
_DEVICE_KINDS = {
    "smartphone",
    "tablet",
    "laptop",
    "monitor",
    "headphone",
    "camera",
    "watch",
    "speaker",
    "tv",
    "console",
    "printer",
    "router",
}
_KIND_ALIASES = {
    "smartphone": ("smartphone", "phone", "mobile"),
    "headphone": (
        "headphone",
        "headphones",
        "earphone",
        "earphones",
        "earbud",
        "earbuds",
        "headset",
    ),
    "tv": ("tv", "television"),
    "laptop": ("laptop", "notebook"),
}
_DEFAULT_RANK_AXES = (
    ("refresh_rate", "desc"),
    ("screen_size", "desc"),
    ("price", "asc"),
)
_SOFT_BRAND = re.compile(
    r"prefer(?:s)?\s+(.+?)(?:,|\.| but\b)",
    re.I,
)
_RES_SEARCH_ALIASES = {
    "720p": ("720p", "HD"),
    "1080p": ("1080p", "FHD"),
    "1440p": ("1440p", "QHD"),
    "4k": ("4k", "UHD"),
    "HD": ("HD",),
}
_MODEL_COLOR = re.compile(
    r"prefer the\s+(black|white|green|blue|red|gold|silver|pink|gray|grey|purple|yellow)\s+(\S+)",
    re.I,
)
_OS_TOKENS = (
    "android",
    "ios",
    "iphone",
    "ipados",
    "blackberry",
    "windows phone",
    "symbian",
    "harmonyos",
    "hyperos",
    "sailfish",
    "ubuntu touch",
    "tizen",
)
_ACCESSORY_HINTS = (
    "accessories",
    "replacement screen",
    "screen only",
    "tempered glass",
    "phone case",
    "back cover",
    "charger",
    "cable",
    "adapter",
    "lens",
    "stand",
    "holder",
    "dock",
    "mystery",
    "protector",
)
_SPEC_HINTS = (
    "720",
    "1080",
    "1440",
    "2160",
    "4k",
    "hd",
    "fhd",
    "qhd",
    "inch",
    "hz",
    "resolution",
)

_PROXY = None


def _get_proxy():
    global _PROXY
    if _PROXY is None:
        from src.agent.proxy_client import ProxyClient

        _PROXY = ProxyClient(timeout=120, max_retries=2)
    return _PROXY


def _provider() -> str:
    return getenv("INFERENCE_PROVIDER", "chutes") or "chutes"


def _model(kind: str = "fast") -> str:
    override = getenv("SANDBOX_MODEL")
    if override:
        return override
    table = _STRONG_MODELS if kind == "strong" else _FAST_MODELS
    return table.get(_provider(), table["chutes"])


def _normalize_model(raw: str) -> str:
    token = (raw or "").strip()
    if _provider() == "openrouter":
        return _TO_OPENROUTER.get(token, token)
    return _TO_CHUTES.get(token, token)


def _lane_model(lane: str) -> str:
    override = getenv("SANDBOX_MODEL")
    if override:
        return _normalize_model(override)
    if lane == LANE_RECALL:
        table = _FAST_MODELS
    else:
        table = _HEAVY_MODELS
    return table.get(_provider(), table["chutes"])


def _reserve_models(primary: str) -> list[str]:
    pool = _RESERVE_OPENROUTER if _provider() == "openrouter" else _RESERVE_CHUTES
    ordered = [primary]
    for item in pool:
        mapped = _normalize_model(item)
        if mapped not in ordered:
            ordered.append(mapped)
    return ordered


def _strong_query(query: str) -> bool:
    text = query.lower()
    if len(text) >= 140:
        return True
    markers = (
        " and ",
        "both",
        "prefer",
        "rather",
        "cheapest",
        "rank",
        "same shop",
        "from the same",
        "budget",
        "voucher",
        "in order",
    )
    return sum(1 for marker in markers if marker in text) >= 2


def _tool_fn(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if isinstance(function, dict):
        return function
    return tool


def _tool_name(tool: dict[str, Any]) -> str:
    return str(_tool_fn(tool).get("name") or "")


def _tool_props(tool: dict[str, Any]) -> dict[str, Any]:
    parameters = _tool_fn(tool).get("parameters") or {}
    properties = parameters.get("properties") or {}
    return properties if isinstance(properties, dict) else {}


def _pick_tool(
    tools: list[dict[str, Any]], names: tuple[str, ...]
) -> dict[str, Any] | None:
    wanted = {name.lower() for name in names}
    for tool in tools:
        if _tool_name(tool).lower() in wanted:
            return tool
    return None


def _name_in(name: str, names: tuple[str, ...]) -> bool:
    return str(name or "").lower() in {item.lower() for item in names}


def _tool_match(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    lower = str(name or "").lower()
    for tool in tools:
        if _tool_name(tool).lower() == lower:
            return tool
    return None


def _canonicalize_args(args: dict[str, Any] | None) -> dict[str, Any]:
    inverted: dict[str, str] = {}
    for logical, aliases in _ARG_ALIASES.items():
        for alias in aliases:
            inverted[alias.lower()] = logical
    out: dict[str, Any] = {}
    for key, value in (args or {}).items():
        logical = inverted.get(str(key).lower(), str(key))
        if logical not in out or out[logical] in (None, ""):
            out[logical] = value
    return out


def _fill_args(tool: dict[str, Any] | None, wanted: dict[str, Any]) -> dict[str, Any]:
    if tool is None:
        return {}
    properties = _tool_props(tool)
    if not properties:
        return {
            key: value for key, value in wanted.items() if value is not None
        }
    filled: dict[str, Any] = {}
    lower_props = {name.lower(): name for name in properties}
    for logical, value in wanted.items():
        if value is None:
            continue
        for alias in _ARG_ALIASES.get(logical, (logical,)):
            real = lower_props.get(alias.lower())
            if real is None:
                continue
            spec = properties.get(real) or {}
            filled[real] = _coerce_arg(value, spec)
            break
    return filled


def _coerce_arg(value: Any, spec: dict[str, Any]) -> Any:
    declared = spec.get("type")
    types = declared if isinstance(declared, list) else [declared]
    if "string" in types and not isinstance(value, str):
        if isinstance(value, (list, tuple)):
            return ",".join(str(item) for item in value)
        return str(value)
    if "integer" in types and isinstance(value, (int, float, str)):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value
    if "number" in types and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if "boolean" in types:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)
    if "array" in types and isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _price_bounds(text: str) -> tuple[float | None, float | None]:
    match = _PRICE_FROM_TO.search(text)
    if match:
        return float(match.group(1)), float(match.group(2))
    low = high = None
    under = _PRICE_UNDER.search(text)
    if under:
        high = float(under.group(1))
    above = _PRICE_ABOVE.search(text)
    if above:
        low = float(above.group(1))
    return low, high


def _shopper_text(query: str) -> str:
    text = query or ""
    return re.split(r"\n\s*Task rules:", text, maxsplit=1)[0].strip()


def _canon_kind(noun: str) -> str:
    token = _norm(noun)
    if not token:
        return ""
    return _KIND_CANON.get(token, token[:-1] if token.endswith("s") and token not in {"mouse"} else token)


def _kind_from_text(text: str) -> str:
    lowered = text.lower()
    hits: list[tuple[int, str]] = []
    for noun in _PRODUCT_NOUNS:
        if re.search(rf"\b{re.escape(noun)}\b", lowered):
            hits.append((len(noun), _canon_kind(noun)))
    if not hits:
        return "product"
    hits.sort(reverse=True)
    return hits[0][1]


def _kind_from_category(category: str) -> str:
    token = _canon_kind(category)
    if token in _KIND_CATEGORY or token in _DEVICE_KINDS:
        return token
    lowered = _norm(category)
    for noun in _PRODUCT_NOUNS:
        if _norm(noun) == lowered:
            return _canon_kind(noun)
    return "product"


def _kind_and_category(text: str) -> tuple[str, str | None]:
    match = re.search(r"in the (.+?) category", text, re.I)
    kind = _kind_from_text(text)
    if match:
        category = _clean_text(match.group(1))
        if kind == "product":
            kind = _kind_from_category(category)
        return kind, category
    return kind, _KIND_CATEGORY.get(kind)


def _is_spec_token(token: str) -> bool:
    if _canon_resolution(token):
        return True
    if _screen_inches_from_text(token):
        return True
    if _REFRESH_HZ.search(token):
        return True
    return False


def _hard_brand_from_text(shopper: str, kind: str) -> str:
    if not kind or kind == "product":
        return ""
    core = _INTENT_PREFIX.sub("", shopper.strip(), count=1)
    core = _CLAUSE_CUT.split(core, maxsplit=1)[0]
    tokens = core.split()
    kind_idx = next(
        (
            index
            for index, raw in enumerate(tokens)
            if _canon_kind(raw) == kind or _norm(raw) in {_norm(noun) for noun in _PRODUCT_NOUNS}
        ),
        None,
    )
    if kind_idx is None:
        return ""
    brand_tokens: list[str] = []
    for raw in reversed(tokens[:kind_idx]):
        token = re.sub(r"[^A-Za-z0-9.&-]+", "", raw)
        if not token:
            continue
        lowered = token.lower()
        if lowered in {"for", "with", "from", "of", "that", "which"}:
            break
        if lowered in _STOPWORDS:
            if brand_tokens:
                break
            continue
        if _is_spec_token(token):
            continue
        if lowered in _MODIFIERS:
            continue
        brand_tokens.append(token)
        if len(brand_tokens) >= 3:
            break
    brand_tokens.reverse()
    brand = _clean_text(" ".join(brand_tokens))
    if not brand or brand.lower() in _STOPWORDS:
        return ""
    return brand


def _has_token(blob: str, token: str) -> bool:
    needle = _norm(token)
    haystack = _norm(blob)
    if not needle or not haystack:
        return False
    return bool(re.search(rf"\b{re.escape(needle)}\b", haystack))


def _kind_match_tokens(kind: str) -> tuple[str, ...]:
    token = _norm(kind)
    if not token:
        return ()
    aliases = _KIND_ALIASES.get(token, ())
    parts = tuple(part for part in token.split() if len(part) > 2)
    seen: list[str] = []
    for item in (token, *aliases, *parts):
        if item and item not in seen:
            seen.append(item)
    return tuple(seen)


def _pref_tokens(shopper: str, *, brand: str, kind: str) -> list[str]:
    core = _INTENT_PREFIX.sub("", shopper.strip(), count=1)
    core = _CLAUSE_CUT.split(core, maxsplit=1)[0]
    skip = set(_STOPWORDS)
    skip.update(_SPEC_ADJECTIVES)
    skip.update(_norm(part) for part in brand.split() if part)
    skip.add(_norm(kind))
    skip.update(_norm(noun) for noun in _PRODUCT_NOUNS)
    for quoted in re.findall(r'"([^"]+)"', shopper):
        skip.update(_norm(part) for part in quoted.split() if part)
    prefs: list[str] = []
    seen: set[str] = set()
    for raw in core.split():
        token = re.sub(r"[^A-Za-z0-9.+-]+", "", raw)
        if not token or len(token) < 2:
            continue
        lowered = token.lower()
        if lowered in skip or _is_spec_token(token):
            continue
        keep = (
            lowered in _MODIFIERS
            or lowered in _COLORS
            or any(character.isdigit() for character in token)
        )
        if not keep or lowered in seen:
            continue
        seen.add(lowered)
        prefs.append(token)
    return prefs[:4]


def _wanted_screen(item: dict[str, Any] | None) -> float | None:
    if not item:
        return None
    for attr in item.get("attrs") or []:
        inches = _screen_inches_from_text(str(attr))
        if inches is not None:
            return inches
    return None


def _resolution_search_token(item: dict[str, Any] | None) -> str:
    if not item:
        return ""
    for attr in item.get("attrs") or []:
        blob = str(attr or "").lower()
        if re.search(r"720p|\b720\b", blob):
            return "720p"
        if re.search(r"2160|\b4k\b|uhd", blob):
            return "4k"
        if re.search(r"1440|\bqhd\b|\b2k\b", blob):
            return "1440p"
        if re.search(r"1080|full hd|\bfhd\b", blob):
            return "1080p"
        if re.search(r"\bhd\b", blob):
            return "HD"
    return ""


def _resolution_attr_from_text(text: str) -> str:
    return _resolution_search_token({"attrs": [text]})


def _has_os(product: dict[str, Any]) -> bool:
    facts = _facts(product)
    blob = _norm(
        " ".join(
            _clean_text(facts.get(key))
            for key in (
                "device_operating_system",
                "operating_system",
                "os",
            )
        )
    )
    return any(token in blob for token in _OS_TOKENS)


def _is_accessory(product: dict[str, Any]) -> bool:
    path = product.get("category_path") or []
    cat = _norm(" ".join(str(part) for part in path if part))
    title = _norm(product.get("title"))
    options = _norm(product.get("options"))
    blob = f"{cat} {title} {options}"
    if "replacement screen" in blob or "screen only" in blob:
        return True
    return any(hint in blob for hint in _ACCESSORY_HINTS)


def _genuine_kind(product: dict[str, Any], kind: str) -> bool:
    token = _norm(kind)
    if not token or token == "product":
        return True
    if token in _DEVICE_KINDS and _is_accessory(product):
        if token in {"smartphone", "tablet"}:
            phone_type = _norm(
                _facts(product).get("phone_type")
                or _facts(product).get("device_type")
                or ""
            )
            if not any(_has_token(phone_type, alias) for alias in _kind_match_tokens(token)):
                return False
        else:
            return False
    title = _norm(product.get("title"))
    facts = _facts(product)
    phone_type = _norm(
        facts.get("phone_type") or facts.get("device_type") or facts.get("product_type") or ""
    )
    path = [
        _norm(part) for part in (product.get("category_path") or []) if part
    ]
    for alias in _kind_match_tokens(token):
        if _has_token(title, alias) or _has_token(phone_type, alias):
            return True
        if any(alias in part or part.rstrip("s") == alias for part in path):
            return True
    if token == "smartphone" and _has_os(product) and path and path[-1] == "smartphones":
        return True
    return False


def _water_score(product: dict[str, Any]) -> float:
    blob = _norm(
        " ".join(
            [
                _fact_text(product),
                str(product.get("title") or ""),
                str(product.get("options") or ""),
                json.dumps(product.get("facts") or {}, sort_keys=True),
            ]
        )
    )
    score = 0.0
    if re.search(r"ipx[4-9]|ip67|ip68", blob):
        score += 4.0
    if "waterproof" in blob or "water resist" in blob:
        score += 3.0
    if "sweat" in blob:
        score += 2.0
    return score


def _model_matches(product: dict[str, Any], model: str) -> bool:
    token = re.sub(r"[^a-z0-9]+", "", model.lower())
    if not token:
        return False
    facts = _facts(product)
    blob = re.sub(
        r"[^a-z0-9]+",
        "",
        _norm(
            " ".join(
                [
                    str(facts.get("model") or ""),
                    str(product.get("title") or ""),
                    str(product.get("options") or ""),
                ]
            )
        ),
    )
    return token in blob


def _color_matches(product: dict[str, Any], color: str) -> bool:
    wanted = _norm(color)
    if not wanted:
        return False
    facts = _facts(product)
    sources = [
        str(product.get("options") or ""),
        str(facts.get("color") or facts.get("colour") or ""),
        str(product.get("title") or ""),
    ]
    for source in sources:
        text = _norm(source)
        if not text:
            continue
        if wanted in text.split() or wanted in text:
            return True
    return False


def _category_ok(product: dict[str, Any], category: str | None) -> bool:
    if not category:
        return True
    path = [_norm(part) for part in (product.get("category_path") or []) if part]
    if not path:
        return True
    wanted = _norm(category)
    return wanted in path or any(wanted in part for part in path)


def _variant_key(product: dict[str, Any]) -> str:
    product_id = str(product.get("product_id") or product.get("id") or "")
    sku = str(product.get("sku") or "")
    if product_id and sku:
        return f"{product_id}::{sku}"
    return product_id


def _parse_candidate_keys(query: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for product_id, sku in _VARIANT_KEY.findall(query or ""):
        sku = sku.rstrip(".,;:")
        if not sku:
            continue
        key = f"{product_id}::{sku}"
        if key in seen:
            continue
        seen.add(key)
        found.append({"product_id": product_id, "sku": sku})
    return found


def _parse_rank_axes(query: str) -> list[tuple[str, str]]:
    text = (query or "").lower()
    axes: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"(highest|largest|biggest|most|lowest|cheapest|smallest|least)\s+"
        r"([a-z][a-z ]{0,30}?)(?=\s*(?:\.|,|;|among|then|$))",
        text,
    ):
        intens = match.group(1)
        noun = match.group(2).strip()
        field = None
        if "refresh" in noun:
            field = "refresh_rate"
        elif "screen" in noun or re.search(r"\bsize\b", noun):
            field = "screen_size"
        elif "price" in noun or "cheap" in noun or intens == "cheapest":
            field = "price"
        if not field or field in seen:
            continue
        direction = (
            "desc"
            if intens in {"highest", "largest", "biggest", "most"}
            else "asc"
        )
        if intens == "cheapest":
            field = "price"
            direction = "asc"
        seen.add(field)
        axes.append((field, direction))
    return axes


def _wants_justification(query: str) -> bool:
    text = (query or "").lower()
    return bool(re.search(r"\bwhy\b|justif|\breason\b|walk me through", text))


def _required_claim_fields(query: str) -> list[str]:
    text = (query or "").lower()
    fields = ["in_stock", "price"]
    if re.search(
        r"720p|1080p|1440p|2160p|\b4k\b|\bfhd\b|\bqhd\b|full hd|\bhd\b|resolution",
        text,
    ):
        fields.append("resolution")
    elif re.search(r"\d[\d.]*\s*-?\s*inch|screen size|largest screen", text):
        fields.append("screen_size")
    elif re.search(r"refresh|\bhz\b", text):
        fields.append("refresh_rate")
    return fields


def _facts(product: dict[str, Any]) -> dict[str, Any]:
    raw = (
        product.get("facts")
        or product.get("specification")
        or product.get("attributes")
        or {}
    )
    return raw if isinstance(raw, dict) else {}


def _fact_text(product: dict[str, Any]) -> str:
    facts = _facts(product)
    return _norm(" ".join(_clean_text(value) for value in facts.values()))


def _canon_resolution(text: str) -> str | None:
    blob = str(text or "").lower()
    if not blob:
        return None
    if re.search(r"\b720\b|720p", blob):
        return "hd"
    if re.search(r"\b2160\b|2160p|\b4k\b|uhd", blob):
        return "4k"
    if re.search(r"\b1440\b|1440p|\bqhd\b|\b2k\b", blob):
        return "qhd"
    if re.search(r"\b1080\b|1080p|1920\s*[x×]\s*1080|\bfhd\b|full hd", blob):
        return "fhd"
    if re.search(r"\bhd\b", blob):
        return "hd"
    return None


def _canon_screen(text: str) -> str | None:
    inches = _screen_inches_from_text(text)
    if inches is None:
        return None
    if abs(inches - round(inches)) < 0.05:
        return f"{int(round(inches))}in"
    return f"{inches}in"


def _screen_inches_from_text(text: Any) -> float | None:
    found = _INCHES.findall(str(text or ""))
    if not found:
        return None
    try:
        return float(found[-1])
    except ValueError:
        return None


def _fact_size_inches(text: Any) -> float | None:
    raw = _clean_text(text)
    if not raw:
        return None
    blob = raw.lower()
    numbers = [float(token) for token in re.findall(r"\d+(?:\.\d+)?", blob)]
    if len(set(numbers)) > 1:
        return None
    inches = _screen_inches_from_text(raw)
    if inches is not None:
        return inches
    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*(?:in(?:ch(?:es)?)?|\"|'')?",
        blob,
    )
    if not match:
        return None
    value = float(match.group(1))
    if 4 <= value <= 100:
        return value
    return None


def _refresh_hz(product: dict[str, Any]) -> float | None:
    parts = [
        product.get("options"),
        _facts(product).get("refresh_rate"),
        _facts(product).get("refresh"),
        product.get("title"),
        product.get("description"),
    ]
    for part in parts:
        match = _REFRESH_HZ.search(str(part or ""))
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _screen_inches(product: dict[str, Any]) -> float | None:
    inches = _screen_inches_from_text(product.get("options"))
    if inches is not None:
        return inches
    facts = _facts(product)
    for part in (facts.get("size"), facts.get("screen_size")):
        inches = _fact_size_inches(part)
        if inches is not None:
            return inches
    return None


def _lex_rank_key(product: dict[str, Any]) -> tuple:
    return _rank_sort_key(product, _DEFAULT_RANK_AXES)


def _axis_value(product: dict[str, Any], field: str) -> float | None:
    if field == "refresh_rate":
        return _refresh_hz(product)
    if field == "screen_size":
        return _screen_inches(product)
    if field == "price":
        return _price_of(product)
    return None


def _rank_sort_key(
    product: dict[str, Any],
    axes: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
) -> tuple:
    rows = list(axes or _DEFAULT_RANK_AXES)
    key: list[float] = []
    for field, direction in rows:
        value = _axis_value(product, field)
        if direction == "desc":
            key.append(-(value if value is not None else -1.0))
        else:
            key.append(value if value is not None else 1e18)
    return tuple(key)


def _spec_attr(attr: str) -> bool:
    token = _norm(attr)
    return any(hint in token for hint in _SPEC_HINTS)


def _spec_matches(product: dict[str, Any], attr: str) -> bool:
    wanted_res = _canon_resolution(attr)
    got_res = _canon_resolution(
        str(_facts(product).get("display") or _facts(product).get("resolution") or "")
    )
    if wanted_res:
        if got_res:
            return wanted_res == got_res
        if _facts(product):
            return False
    wanted_in = _screen_inches_from_text(attr)
    got_in = _screen_inches(product)
    if wanted_in is not None and got_in is not None:
        return abs(wanted_in - got_in) <= 0.15
    wanted_hz = _REFRESH_HZ.search(str(attr or ""))
    if wanted_hz:
        got_hz = _refresh_hz(product)
        if got_hz is not None:
            return abs(got_hz - float(wanted_hz.group(1))) <= 1.0
    facts = _fact_text(product)
    token = _norm(attr)
    if facts:
        return bool(token) and token in facts
    return bool(token) and token in _blob(product)


def _price_claim(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _justification_claims(
    product: dict[str, Any],
    fields: list[str],
) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    facts = _facts(product)
    for field in fields:
        value: str | None = None
        if field == "in_stock":
            value = "true" if _in_stock(product) else "false"
        elif field == "price":
            price = _price_of(product)
            value = _price_claim(price) if price is not None else None
        elif field == "resolution":
            value = _canon_resolution(
                str(facts.get("display") or facts.get("resolution") or "")
            )
        elif field == "screen_size":
            inches = _screen_inches(product)
            value = _canon_screen(f"{inches} inch") if inches is not None else None
        elif field == "refresh_rate":
            hz = _refresh_hz(product)
            if hz is None:
                value = None
            elif abs(hz - round(hz)) < 0.05:
                value = str(int(round(hz)))
            else:
                value = str(hz)
        elif field == "brand":
            value = _clean_text(product.get("brand")) or None
        elif field == "category":
            path = product.get("category_path") or []
            if isinstance(path, list) and path:
                value = _clean_text(path[-1]) or None
        if not value:
            continue
        claims.append({"field": field, "value": str(value)})
    return claims


def _ground_justification(
    product: dict[str, Any],
    goal: dict[str, Any],
) -> list[dict[str, str]]:
    fields = list(goal.get("soft", {}).get("claim_fields") or ["in_stock", "price"])
    brand = _clean_text(product.get("brand"))
    if brand:
        fields = ["brand"] + [field for field in fields if field != "brand"]
    allowed = set(_JUSTIFY_CLAIM_FIELDS)
    fields = [field for field in fields if field in allowed]
    return [
        claim
        for claim in _justification_claims(product, fields)
        if claim.get("field") in allowed
        and claim.get("field") not in _FORBIDDEN_CLAIM_FIELDS
    ]


def _cart_is_stale(
    product: dict[str, Any] | None, goal: dict[str, Any]
) -> bool:
    if product is None:
        return False
    if not _in_stock(product):
        return True
    price = _price_of(product)
    hard = goal.get("hard") or {}
    budget = hard.get("budget") or hard.get("price_max")
    if budget is not None and price is not None and price > float(budget):
        return True
    return False


def _merge_catalog(
    catalog: list[dict[str, Any]],
    products: list[dict[str, Any]],
    *,
    viewed: bool = False,
    inspected: bool = False,
) -> None:
    by_key = {_variant_key(item): item for item in catalog if _variant_key(item)}
    for product in products:
        if not isinstance(product, dict):
            continue
        key = _variant_key(product)
        if not key:
            continue
        incoming = {field: value for field, value in product.items() if value is not None}
        current = by_key.get(key)
        if current is None:
            if viewed:
                incoming["_hydrated"] = True
            if inspected:
                incoming["_inspected"] = True
            catalog.append(incoming)
            by_key[key] = incoming
            continue
        incoming_facts = incoming.get("facts")
        if not incoming_facts:
            incoming.pop("facts", None)
        current.update(incoming)
        if viewed:
            current["_hydrated"] = True
        if inspected:
            current["_inspected"] = True


def _apply_market_patch(
    catalog: list[dict[str, Any]],
    event: dict[str, Any],
    cart_item: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    target = event.get("target") if isinstance(event.get("target"), dict) else {}
    product_id = str(target.get("product_id") or event.get("product_id") or "")
    sku = str(target.get("sku") or event.get("sku") or "")
    if not product_id and cart_item is not None:
        product_id = str(cart_item.get("product_id") or "")
        sku = sku or str(cart_item.get("sku") or "")
    kind = _norm(event.get("kind") or "")
    patch: dict[str, Any] = {}
    if product_id:
        patch["product_id"] = product_id
    if sku:
        patch["sku"] = sku
    if "stockout" in kind or event.get("in_stock_after") is False:
        patch["in_stock"] = False
    new_price = event.get("new_price")
    if new_price is not None:
        patch["price"] = new_price
    if not patch.get("product_id"):
        return cart_item
    _merge_catalog(catalog, [patch], viewed=False, inspected=False)
    key = _variant_key(patch)
    for product in catalog:
        if _variant_key(product) == key:
            product["_inspected"] = False
    if cart_item is not None and (
        not key or _variant_key(cart_item) == key or not _variant_key(cart_item)
    ):
        cart_item.update({field: value for field, value in patch.items() if value is not None})
        cart_item["_inspected"] = False
    return cart_item


def _inspect_patch(observation: dict[str, Any], args: dict[str, Any]) -> dict[str, Any] | None:
    product_id = observation.get("product_id") or args.get("product_id")
    sku = observation.get("sku") or args.get("sku")
    price = observation.get("price")
    if price is None:
        price = observation.get("current_price")
    row: dict[str, Any] = {}
    if product_id not in (None, ""):
        row["product_id"] = product_id
    if sku not in (None, ""):
        row["sku"] = sku
    if price is not None:
        row["price"] = price
    if "in_stock" in observation:
        row["in_stock"] = observation.get("in_stock")
    if not row.get("product_id"):
        return None
    return row


def _apply_llm_shopper(
    content: str,
    goal: dict[str, Any],
    cart_item: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, bool]:
    lowered = content.lower()
    item = (goal.get("items") or [{}])[0]
    budget = _SHOPPER_BUDGET.search(content) or _BUDGET_NOW.search(content)
    if budget:
        goal["hard"]["budget"] = float(budget.group(1))
        goal["hard"]["price_max"] = float(budget.group(1))
    inches = _screen_inches_from_text(content)
    if inches is not None:
        attrs = [
            attr
            for attr in (item.get("attrs") or [])
            if _screen_inches_from_text(str(attr)) is None
        ]
        item["attrs"] = [f"{inches:g} inch"] + attrs
        goal["hard"]["ask_spec"] = False
    bumped = _PRICE_HIKE.search(content)
    if bumped is not None and cart_item is not None:
        cart_item["price"] = float(bumped.group(2))
        cart_item["_inspected"] = False
    if any(
        token in lowered
        for token in (
            "out of stock",
            "sold out",
            "unavailable",
            "no longer available",
        )
    ):
        if cart_item is not None:
            cart_item["in_stock"] = False
            cart_item["_inspected"] = False
    stale = _cart_is_stale(cart_item, goal)
    await_reply = bool(_is_inquiry(content) and cart_item is None and not stale)
    return cart_item, await_reply


def _ingest_llm_observation(
    observation: Any,
    action: dict[str, Any],
    catalog: list[dict[str, Any]],
    cart_item: dict[str, Any] | None,
) -> dict[str, Any] | None:
    name = str(action.get("name") or "")
    args = action.get("args") or {}
    viewed = _name_in(name, _VIEW_NAMES + _COMPARE_NAMES)
    inspected = _name_in(name, _INSPECT_NAMES)
    products = _walk_products(observation)
    if inspected and not products and isinstance(observation, dict):
        row = _inspect_patch(observation, args)
        if row is not None:
            products = [row]
    if products:
        _merge_catalog(catalog, products, viewed=viewed, inspected=inspected)
        if cart_item is not None:
            key = _variant_key(cart_item)
            for product in catalog:
                if key and _variant_key(product) == key:
                    cart_item.update(
                        {
                            field: value
                            for field, value in product.items()
                            if value is not None
                        }
                    )
                    break
    if _name_in(name, _CART_NAMES) and isinstance(observation, dict):
        added = observation.get("added")
        if isinstance(added, dict):
            _merge_catalog(catalog, [added], viewed=False, inspected=True)
        event = observation.get("market_event")
        if isinstance(event, dict) and event:
            cart_item = _apply_market_patch(catalog, event, cart_item)
        cart_item = _lookup_product(catalog, args) or cart_item
        if isinstance(added, dict) and cart_item is None:
            cart_item = added
    if _name_in(name, _REMOVE_NAMES):
        cart_item = None
    return cart_item


def _redirect_tool(
    tool: dict[str, Any] | None, wanted: dict[str, Any]
) -> dict[str, Any] | None:
    if tool is None:
        return None
    args = _fill_args(tool, _canonicalize_args(wanted))
    if not args:
        return None
    return {"name": _tool_name(tool), "args": args}


def _filled_action_args(
    tool: dict[str, Any] | None, args: dict[str, Any]
) -> dict[str, Any]:
    wanted = _canonicalize_args(args)
    if tool is None:
        return {key: value for key, value in wanted.items() if value is not None}
    return _fill_args(tool, wanted)


def _search_variants(
    item: dict[str, Any],
    *,
    water: bool = False,
    category: str | None = None,
) -> list[str]:
    kind = _clean_text(item.get("type") or item.get("noun"))
    if _norm(kind) == "product":
        kind = ""
    brand = _clean_text(item.get("brand") or item.get("soft_brand"))
    model = _clean_text(item.get("model"))
    color = _clean_text(item.get("color"))
    screen = _wanted_screen(item)
    res_tokens = _RES_SEARCH_ALIASES.get(
        _resolution_search_token(item),
        ((_resolution_search_token(item),) if _resolution_search_token(item) else ()),
    )
    prefs = [
        _clean_text(pref)
        for pref in (item.get("prefs") or [])
        if _clean_text(pref)
    ]
    phrases = [
        _clean_text(phrase)
        for phrase in (item.get("phrases") or [])
        if _clean_text(phrase)
    ]
    attrs = [
        _clean_text(attr)
        for attr in (item.get("attrs") or [])
        if _clean_text(attr)
        and _canon_resolution(attr) is None
        and _screen_inches_from_text(attr) is None
        and _norm(attr) not in {_norm(pref) for pref in prefs}
        and _norm(attr) not in {_norm(phrase) for phrase in phrases}
    ]
    screen_text = f"{screen:g} inch" if screen is not None else ""
    category_text = _clean_text(category)
    water_queries = (f"waterproof {kind}" if kind else "waterproof",) if water else ()
    variants: list[str] = []
    for candidate in (
        *water_queries,
        " ".join(part for part in (color, model) if part),
        model,
        *phrases,
        " ".join(part for part in (brand, res_tokens[0], kind) if part) if res_tokens else "",
        " ".join(part for part in (brand, res_tokens[1] if len(res_tokens) > 1 else "", kind) if part)
        if len(res_tokens) > 1
        else "",
        " ".join(part for part in (brand, screen_text, kind) if part) if screen_text else "",
        " ".join(part for part in (brand, prefs[0] if prefs else "", kind) if part),
        " ".join(part for part in (phrases[0] if phrases else "", category_text) if part),
        " ".join(part for part in (brand, kind) if part),
        " ".join(part for part in (screen_text, kind) if part),
        " ".join(part for part in (kind, attrs[0] if attrs else "") if part),
        kind,
        f"sweat resistant {kind}" if water and kind else "",
        brand,
        category_text,
    ):
        text = _clean_text(candidate)
        if not text or _norm(text) in {"product", "item"}:
            continue
        if text.lower() not in {item.lower() for item in variants}:
            variants.append(text)
    return variants or [text for text in (phrases[:1] or [category_text, kind, brand]) if text] or [
        "in stock"
    ]


def _blob(product: dict[str, Any]) -> str:
    parts = [
        product.get("title"),
        product.get("brand"),
        product.get("description"),
        product.get("options"),
        json.dumps(product.get("attributes") or {}, sort_keys=True),
        json.dumps(product.get("sku_options") or [], sort_keys=True),
        json.dumps(product.get("facts") or {}, sort_keys=True),
        json.dumps(product.get("specification") or {}, sort_keys=True),
        json.dumps(product.get("category_path") or [], sort_keys=True),
    ]
    return _norm(" ".join(_clean_text(part) for part in parts if part))


def _price_of(product: dict[str, Any]) -> float | None:
    raw = product.get("price")
    if isinstance(raw, dict):
        raw = raw.get("value") or raw.get("amount") or raw.get("current")
    try:
        if raw is None or raw == "":
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _in_stock(product: dict[str, Any]) -> bool:
    if "in_stock" in product:
        value = product.get("in_stock")
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        text = _norm(value)
        if text in {"false", "0", "no", "oos", "out of stock", "unavailable"}:
            return False
        if text in {"true", "1", "yes", "in stock", "available"}:
            return True
        if value is None:
            return True
        return bool(value)
    availability = _norm(product.get("availability") or product.get("stock"))
    if not availability:
        return True
    return availability not in {"out of stock", "oos", "unavailable", "false", "0"}


def _feasible(
    product: dict[str, Any],
    item: dict[str, Any],
    hard: dict[str, Any],
) -> bool:
    price = _price_of(product)
    low = hard.get("price_min")
    high = hard.get("price_max")
    budget = hard.get("budget")
    if low is not None and price is not None and price < float(low):
        return False
    if high is not None and price is not None and price > float(high):
        return False
    if budget is not None and price is not None and price > float(budget):
        return False
    kind = _norm(item.get("type") or item.get("noun"))
    if hard.get("in_stock", True) and not _in_stock(product):
        return False
    shop = hard.get("shop_id")
    if shop and str(product.get("shop_id") or "") not in {"", str(shop)}:
        if str(product.get("shop_id") or "") != str(shop):
            return False
    if hard.get("genuine", True) and not _genuine_kind(product, kind):
        return False
    if not _category_ok(product, hard.get("category")):
        return False
    blob = _blob(product)
    brand = _norm(item.get("brand") or hard.get("brand") or "")
    got_brand = _norm(product.get("brand"))
    if brand:
        if got_brand:
            if brand not in got_brand and got_brand not in brand:
                return False
        elif brand not in blob:
            return False
    screen = _wanted_screen(item)
    if screen is not None:
        got = _screen_inches(product)
        if got is None or abs(got - screen) > 0.15:
            return False
    for attr in item.get("attrs") or []:
        token = _norm(attr)
        if not token or _screen_inches_from_text(attr) is not None:
            continue
        if _facts(product) and _spec_attr(attr):
            if not _spec_matches(product, attr):
                return False
            continue
        if token not in blob:
            return False
    model = _clean_text(item.get("model"))
    if model and hard.get("require_model") and not _model_matches(product, model):
        return False
    return True


def _match_score(
    product: dict[str, Any],
    item: dict[str, Any],
    hard: dict[str, Any],
    rank: str,
) -> float:
    if not _feasible(product, item, hard):
        return -1.0
    blob = _blob(product)
    score = 0.0
    brand = _norm(item.get("brand"))
    if brand and brand in blob:
        score += 3.0
    kind = _norm(item.get("type") or item.get("noun"))
    for token in kind.split():
        if len(token) > 2 and token in blob:
            score += 1.2
    for attr in item.get("attrs") or []:
        token = _norm(attr)
        if token and token in blob:
            score += 2.0
    for pref in item.get("prefs") or []:
        token = _norm(pref)
        if token and token in blob:
            score += 1.5
    price = _price_of(product) or 0.0
    if rank == "price_asc":
        score += 4.0 / (1.0 + price)
    elif rank == "price_desc":
        score += math.log1p(price)
    return score


def _without_price_outliers(
    products: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prices = sorted(
        price
        for price in (_price_of(product) for product in products)
        if price is not None and price > 0
    )
    if len(prices) < 3:
        return products
    median = prices[len(prices) // 2]
    kept = [
        product
        for product in products
        if (_price_of(product) or 0.0) >= 0.05 * median
    ]
    return kept or products


def _rrf(result_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    keep: dict[str, dict[str, Any]] = {}
    for rows in result_lists:
        for rank, product in enumerate(rows, start=1):
            key = _variant_key(product) or str(
                product.get("product_id")
                or product.get("id")
                or product.get("sku")
                or rank
            )
            scores[key] = scores.get(key, 0.0) + 1.0 / (60.0 + rank)
            keep[key] = product
    return [
        keep[key]
        for key, _score in sorted(scores.items(), key=lambda item: -item[1])
    ]


def _variant_rows(node: dict[str, Any]) -> list[dict[str, Any]]:
    product_id = node.get("product_id") or node.get("id")
    variants = node.get("variants")
    if isinstance(variants, list) and variants:
        rows: list[dict[str, Any]] = []
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            row = {key: value for key, value in node.items() if key != "variants"}
            row["product_id"] = product_id
            row["sku"] = variant.get("sku") or row.get("sku")
            if variant.get("price") is not None:
                row["price"] = variant.get("price")
            if "in_stock" in variant:
                row["in_stock"] = variant.get("in_stock")
            if variant.get("options") is not None:
                row["options"] = variant.get("options")
            row["_hydrated"] = True
            rows.append(row)
        return rows
    if product_id not in (None, "") and (
        "title" in node
        or "price" in node
        or "sku" in node
        or "brand" in node
        or "facts" in node
    ):
        return [node]
    return []


def _walk_products(payload: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        rows = _variant_rows(node)
        if rows:
            for row in rows:
                key = _variant_key(row) or str(row.get("product_id") or "")
                if key and key not in seen:
                    seen.add(key)
                    found.append(row)
            return
        for value in node.values():
            if isinstance(value, (dict, list)):
                visit(value)

    visit(payload)
    return found


def _loads_json(text: str) -> dict[str, Any] | None:
    blob = text.strip()
    fenced = _JSON_FENCE.search(blob)
    if fenced:
        blob = fenced.group(1)
    else:
        match = _JSON_OBJECT.search(blob)
        if match:
            blob = match.group(0)
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _empty_goal(query: str) -> dict[str, Any]:
    shopper = _shopper_text(query)
    low, high = _price_bounds(shopper)
    kind, category = _kind_and_category(shopper)
    lowered = shopper.lower()
    attrs: list[str] = []
    phrases: list[str] = []
    screen = _screen_inches_from_text(shopper)
    if screen is not None:
        attrs.append(f"{screen:g} inch")
    res_token = _resolution_attr_from_text(shopper)
    if res_token:
        attrs.append(res_token)
    hz_match = _REFRESH_HZ.search(shopper)
    if hz_match:
        attrs.append(f"{hz_match.group(1)} hz")
    quoted = re.search(r'matches "([^"]+)"', shopper, re.I)
    phrase = re.search(r"matches the phrase\s+([^,]+)", shopper, re.I)
    if quoted:
        phrases.append(quoted.group(1).strip())
    elif phrase:
        phrases.append(phrase.group(1).strip())
    attrs.extend(phrases)
    soft_brand = ""
    hard_brand = ""
    prefer = _SOFT_BRAND.search(shopper)
    if prefer and "prefer the" not in _norm(prefer.group(0)):
        soft_brand = _clean_text(prefer.group(1))
        if soft_brand.lower().startswith("the "):
            soft_brand = soft_brand[4:]
    if not soft_brand:
        hard_brand = _hard_brand_from_text(shopper, kind)
    prefs = _pref_tokens(shopper, brand=hard_brand or soft_brand, kind=kind)
    model = ""
    color = ""
    colored = _MODEL_COLOR.search(shopper)
    if colored:
        color = colored.group(1).lower()
        model = colored.group(2).rstrip(".,;")
    water = any(token in lowered for token in ("sweat", "water resist", "waterproof"))
    ask_spec = "if you ask" in lowered
    item = {
        "type": kind,
        "brand": hard_brand,
        "soft_brand": soft_brand,
        "attrs": attrs,
        "prefs": prefs,
        "phrases": phrases,
        "qty": 1,
        "model": model,
        "color": color,
    }
    preferences = list(prefs)
    if soft_brand:
        preferences.append(soft_brand)
    if water:
        preferences.append("sweat and water resistance")
    if color or model:
        preferences.append(" ".join(part for part in (color, model) if part))
    return {
        "items": [item],
        "hard": {
            "price_min": low,
            "price_max": high,
            "same_shop": False,
            "in_stock": True,
            "budget": high,
            "shop_id": None,
            "brand": hard_brand or None,
            "category": category,
            "ask_spec": ask_spec,
            "genuine": True,
        },
        "soft": {
            "preferences": preferences,
            "rank": "none",
            "need_reason": _wants_justification(shopper),
            "claim_fields": _required_claim_fields(shopper),
            "brand": soft_brand,
            "water": water,
        },
        "candidates": _parse_candidate_keys(query),
        "rank_axes": _parse_rank_axes(query),
    }


def _normalize_goal(parsed: dict[str, Any], query: str) -> dict[str, Any]:
    fallback = _empty_goal(query)
    items = parsed.get("items")
    if not isinstance(items, list) or not items:
        items = fallback["items"]
    clean_items = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        attrs = raw.get("attrs") or raw.get("attributes") or []
        if isinstance(attrs, str):
            attrs = [part.strip() for part in attrs.split(",") if part.strip()]
        clean_items.append(
            {
                "type": _clean_text(raw.get("type") or raw.get("noun") or raw.get("product")),
                "brand": _clean_text(raw.get("brand")),
                "attrs": [str(attr) for attr in attrs if str(attr).strip()],
                "qty": int(raw.get("qty") or 1),
            }
        )
    if not clean_items:
        clean_items = fallback["items"]
    hard = parsed.get("hard") if isinstance(parsed.get("hard"), dict) else {}
    soft = parsed.get("soft") if isinstance(parsed.get("soft"), dict) else {}
    low, high = _price_bounds(query)
    rank = str(soft.get("rank") or "none").lower()
    if rank not in {"none", "price_asc", "price_desc"}:
        lowered = query.lower()
        if "cheap" in lowered or "lowest price" in lowered:
            rank = "price_asc"
        elif "premium" in lowered or "highest price" in lowered:
            rank = "price_desc"
        else:
            rank = "none"
    same_shop = bool(hard.get("same_shop"))
    if not same_shop:
        same_shop = bool(
            re.search(r"same shop|same store|from one shop|single shop", query, re.I)
        )
    return {
        "items": clean_items,
        "hard": {
            "price_min": hard.get("price_min", low),
            "price_max": hard.get("price_max", high),
            "same_shop": same_shop,
            "in_stock": hard.get("in_stock", True) is not False,
            "budget": hard.get("budget", high),
            "shop_id": hard.get("shop_id"),
        },
        "soft": {
            "preferences": [
                str(item)
                for item in (soft.get("preferences") or [])
                if str(item).strip()
            ],
            "rank": rank,
            "need_reason": _wants_justification(_shopper_text(query)),
            "claim_fields": _required_claim_fields(_shopper_text(query)),
        },
        "candidates": _parse_candidate_keys(query),
        "rank_axes": _parse_rank_axes(_shopper_text(query)),
    }


def _assistant_text(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    return _get_proxy().post("/inference/chat/completions", json_data=payload)


_BASE_PLAYBOOK = """
You are a shopping agent. Use ONLY the supplied tools. Continue until an
observation reports done=true.

# Firm vs soft
Firm constraints must hold on the exact SKU: category, named brand when it is
stated as a requirement, numeric specs (screen size, resolution, refresh),
budget, and in-stock. Soft preferences (preferred brand, gaming, waterproof,
color) never override firm constraints.

# Matching
Do not trust marketing titles. After search, view or compare raw facts before
the first add_to_cart. Resolution comes from facts.display / facts.resolution
(hd / fhd / qhd / 4k), screen size from options or facts.size, refresh from
options or facts. Accessories, stands, cases, and replacement screens are not
the device. Category-path "accessories" is fine when the shopper asked for
that category.

# Search
Search VALUES, not JSON keys. Use the suggested catalog queries below, then
one small reformulation if recall is thin. Batch independent view/inspect
calls when max_calls_per_turn allows.

# First cart
The first add_to_cart is the scored commitment. Do not cart a placeholder.
Inspect stock on the exact SKU before carting when that tool exists.
Never add_to_cart and place_test_order in the same turn.

# Shopper messages
If the shopper asks a question BEFORE any cart, answer with the message tool
in the next turn. After a cart, if price/stock broke firm constraints, remove
and recover first — do not stall on chat. If they reveal a hidden spec or a
new budget, apply it immediately.

# Events
Prices and stock can change. Re-inspect. If over budget or OOS, remove_from_cart
and recover. Then place_test_order; do not wait for permission.

# Ordering
place_test_order only after a prior cart, and only if that SKU is still
in-stock and in-budget. Re-inspect after any price or stock event so claims
and cart state use the current numbers.
""".strip()

_LANE_PLAYBOOK = {
    LANE_INTENT: """
# Intent
Firm: category, the stated hardware spec, budget, in-stock.
Soft: a preferred brand, used only when a fully firm-compliant SKU exists.
Search the spec first (brand is optional). If the shopper asks which pick /
does it meet the spec, message with the SKU and the fact-backed spec value.
A later swap does not repair a first cart that ignored an available compliant
preferred-brand option.
""".strip(),
    LANE_RECALL: """
# Retrieval
Search the match phrase and category. Do not query the word "product".
Shortlist several distinct in-category listings, view enough to reject noise,
then buy the best in-stock item within budget. Order once verified.
""".strip(),
    LANE_CONSTRAINT: """
# Constraints
If the shopper will reveal the firm spec only when asked, message FIRST and
wait for the answer before searching. Every firm requirement must hold on the
exact SKU (option, price, stock). Titles mix good and bad variants — view then
inspect_stock. If that variant sells out, do not take a sibling that misses a
firm spec.
""".strip(),
    LANE_PREFERENCE: """
# Preferences
Hard: category, budget, in-stock, real device.
Soft: use-case (sweat/water, workout, gaming, materials). Prefer a slightly
dearer in-budget item that matches soft axes over the cheapest shell.
Search a resistance/use-case query AND the product noun. First cart must be a
real preference-fit purchase. If no soft-match appears after several searches,
order the best firm-compliant item rather than stalling.
""".strip(),
    LANE_RANK: """
# Ranking
Restrict to the candidate variants named in the query (product_id::sku).
Compare or view all of them before any add_to_cart. Rank only by the stated
axes (typical: highest refresh, then largest screen, then lowest price).
First cart must be the current best in-stock in-budget option. After events,
remove if invalid, re-rank survivors, cart the new #1, then order.
place_test_order with ONLY product_id and sku — never attach justification.
""".strip(),
    LANE_RECOVER: """
# Recovery
Never relax category, budget, or in-stock. Prefer exact model+color first
(model facts beat titles). If that item disappears or jumps over budget:
same model other color, then other models. Re-check stock and price after
every cart change before place_test_order.
""".strip(),
    LANE_JUSTIFY: """
# Justification
View or compare raw facts BEFORE the first cart. Do not justify from titles
and do not cart a decoy whose facts.display / facts.size miss the shopper spec.
Never add_to_cart and place_test_order in the same turn — first cart is the
commitment; price/stock may change afterwards. inspect_stock the exact SKU
before carting and again before ordering.

place_test_order MUST include justification.claims.
Allowlisted fields only: brand, in_stock, price, refresh_rate, resolution,
screen_size. Required: in_stock, price, and exactly ONE shopper spec field
(resolution OR screen_size OR refresh_rate). Add brand when the viewed brand
field is set. Values: resolution from facts.display (hd/fhd/qhd/4k),
screen_size like 27in, price as the latest inspected number, in_stock
true/false.
Never use display, size, hdr, category, or device_operating_system as claim
fields. If the shopper says the carted item got more expensive, recover if it
is over budget — do not answer with the message tool instead of recovering.
""".strip(),
}

_IDLE_PROMPTS = (
    "No tool call received. Choose a supplied tool now. If you have not searched "
    "yet, call search or filter using the suggested catalog queries.",
    "Still no tool call. Progress: search/filter, view a candidate, answer any "
    "pending shopper question with message, or place_test_order if the cart "
    "already holds a verified in-stock SKU.",
)


def _classify_lane(query: str, goal: dict[str, Any] | None = None) -> str:
    parsed = goal or _empty_goal(query)
    shopper = _shopper_text(query).lower()
    item = (parsed.get("items") or [{}])[0]
    if parsed.get("candidates") or parsed.get("rank_axes"):
        return LANE_RANK
    if parsed.get("soft", {}).get("need_reason"):
        return LANE_JUSTIFY
    if parsed.get("hard", {}).get("ask_spec"):
        return LANE_CONSTRAINT
    if item.get("model") and re.search(
        r"disappear|another color|same model|before changing models", shopper
    ):
        return LANE_RECOVER
    if parsed.get("soft", {}).get("water"):
        return LANE_PREFERENCE
    phrases = item.get("phrases") or []
    kind = _norm(item.get("type"))
    if phrases or (kind in {"", "product"} and parsed.get("hard", {}).get("category")):
        return LANE_RECALL
    if parsed.get("soft", {}).get("brand") and not item.get("brand"):
        return LANE_INTENT
    if item.get("brand") or item.get("attrs"):
        return LANE_CONSTRAINT
    return LANE_RECALL


def _goal_brief(goal: dict[str, Any]) -> str:
    item = (goal.get("items") or [{}])[0]
    hard = goal.get("hard") or {}
    soft = goal.get("soft") or {}
    searches = _search_variants(
        item,
        water=bool(soft.get("water")),
        category=hard.get("category"),
    )
    return (
        f"kind={item.get('type')!s}; hard_brand={item.get('brand')!s}; "
        f"soft_brand={soft.get('brand')!s}; attrs={item.get('attrs')}; "
        f"prefs={item.get('prefs')}; phrases={item.get('phrases')}; "
        f"model={item.get('model')!s}; color={item.get('color')!s}; "
        f"budget={hard.get('budget')}; category={hard.get('category')!s}; "
        f"ask_spec={hard.get('ask_spec')}; need_reason={soft.get('need_reason')}; "
        f"rank_axes={goal.get('rank_axes')}; "
        f"candidates={goal.get('candidates')}; "
        f"suggested_searches={searches[:6]}"
    )


def _compose_system(lane: str, policy: dict[str, Any], goal: dict[str, Any]) -> str:
    labels: list[str] = []
    for tool in policy.get("tools") or []:
        name = _tool_name(tool)
        if name:
            labels.append(name)
    playbook = _LANE_PLAYBOOK.get(lane, _LANE_PLAYBOOK[LANE_RECALL])
    return (
        f"{_BASE_PLAYBOOK}\n\n{playbook}\n\n"
        f"# Parsed goal\n{_goal_brief(goal)}\n\n"
        f"# Runtime\n"
        f"lane={lane}\n"
        f"tools={', '.join(labels) or '(see schemas)'}\n"
        f"max_steps={policy.get('max_steps')} "
        f"max_calls_per_turn={policy.get('max_calls_per_turn', 1)}\n"
    )


def _is_inquiry(blob: str) -> bool:
    cleaned = (blob or "").strip().lower()
    if not cleaned:
        return False
    if "?" in cleaned:
        return True
    prefixes = (
        "which ",
        "what ",
        "does ",
        "did ",
        "can you",
        "could you",
        "wait",
        "hold on",
        "before you",
        "re-check",
        "got anything",
    )
    return any(cleaned.startswith(prefix) or f" {prefix}" in f" {cleaned}" for prefix in prefixes)


def _next_coach(
    *,
    lane: str,
    step: int,
    actions: list[str],
    goal: dict[str, Any],
    await_reply: bool,
    did_cart: bool,
    cart_stale: bool = False,
) -> str | None:
    blob = " ".join(actions)
    if cart_stale:
        return (
            "Coach: the cart SKU is over budget or out of stock. "
            "remove_from_cart now, then inspect a replacement that still meets "
            "brand, spec, budget, and stock. Do not message instead of recovering, "
            "and do not place_test_order on the stale SKU."
        )
    if await_reply and not did_cart:
        return (
            "Coach: the shopper asked a question. Answer with the message tool "
            "this turn before carting or ordering."
        )
    if await_reply and did_cart:
        return (
            "Coach: shopper spoke after a cart. If the cart SKU is now over "
            "budget or out of stock, remove_from_cart and recover first. "
            "Otherwise inspect_stock, then place_test_order."
        )
    if (
        lane == LANE_CONSTRAINT
        and step == 1
        and goal.get("hard", {}).get("ask_spec")
        and "message" not in blob
    ):
        return (
            "Coach: ask the shopper for the firm hardware spec with the message "
            "tool before searching."
        )
    if lane == LANE_RANK and step == 1 and not did_cart:
        return (
            "Coach: compare or view every listed candidate_id before the first "
            "add_to_cart. Rank by the parsed rank_axes. place_test_order with "
            "only product_id and sku."
        )
    if lane == LANE_INTENT and step == 1:
        return (
            "Coach: honor the preferred brand only when a SKU also meets the "
            "firm spec, budget, and stock. Search the spec, not the brand alone."
        )
    if lane == LANE_PREFERENCE and step <= 2 and not did_cart:
        return (
            "Coach: compare soft-preference axes (water/sweat/use-case) before "
            "the first cart. Do not cart the cheapest placeholder."
        )
    if lane == LANE_JUSTIFY and not did_cart:
        return (
            "Coach: search, view raw facts, inspect_stock, then add_to_cart. "
            "Do not cart from titles and do not place_test_order in the same "
            "turn as the first cart. Claims later: brand, in_stock, price, and "
            "exactly one shopper spec field from facts/inspect_stock."
        )
    if lane == LANE_JUSTIFY and did_cart:
        return (
            "Coach: inspect_stock the cart SKU, then place_test_order with "
            "grounded claims (in_stock, price, one spec, brand). Never invent "
            "display/size/hdr/category fields."
        )
    if lane == LANE_RECALL and step == 1:
        return (
            "Coach: search the match phrase (and category) first. Do not search "
            "the word product."
        )
    if lane == LANE_RECOVER and step == 1:
        return (
            "Coach: search the preferred color+model first. If it disappears, "
            "same model other color, then other models. Keep budget and stock firm."
        )
    if step <= 2 and not any(
        name in blob for name in ("search", "filter", "view", "compare", "message")
    ):
        return (
            "Hint: start with search or filter using the suggested catalog queries, "
            "then view the best hits."
        )
    if lane == LANE_RANK and did_cart:
        return (
            "Hint: after recovery, place_test_order with only product_id and sku "
            "— omit justification."
        )
    return None


def _shrink_obs(payload: Any, cap: int = 7000) -> str:
    try:
        rendered = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        rendered = str(payload)
    if len(rendered) <= cap:
        return rendered
    return rendered[: cap - 20] + "...<truncated>"


def _parse_tool_call(call: dict[str, Any]) -> dict[str, Any] | None:
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    name = str(function.get("name") or call.get("name") or "")
    if not name:
        return None
    raw = function.get("arguments", call.get("arguments", "{}"))
    if isinstance(raw, dict):
        args = raw
    else:
        try:
            args = json.loads(raw or "{}")
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {
        "id": str(call.get("id") or ""),
        "name": name,
        "args": args,
    }


def _lookup_product(
    catalog: list[dict[str, Any]], args: dict[str, Any]
) -> dict[str, Any] | None:
    wanted = _canonicalize_args(args)
    product_id = str(wanted.get("product_id") or "")
    sku = str(wanted.get("sku") or "")
    if product_id and sku:
        key = f"{product_id}::{sku}"
        for product in reversed(catalog):
            if _variant_key(product) == key:
                return product
    for product in reversed(catalog):
        if product_id and str(product.get("product_id") or "") != product_id:
            continue
        if sku and str(product.get("sku") or "") not in {"", sku}:
            continue
        return product
    return None


def _prepare_actions(
    raw: list[dict[str, Any]],
    *,
    lane: str,
    goal: dict[str, Any],
    catalog: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    did_cart: bool,
    cart_item: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    item = (goal.get("items") or [{}])[0]
    hard = goal.get("hard") or {}
    need_reason = bool(goal.get("soft", {}).get("need_reason")) and lane != LANE_RANK
    inspect_tool = _pick_tool(tools, _INSPECT_NAMES)
    view_tool = _pick_tool(tools, _VIEW_NAMES) or _pick_tool(tools, _COMPARE_NAMES)
    cart_tool = _pick_tool(tools, _CART_NAMES)
    remove_tool = _pick_tool(tools, _REMOVE_NAMES)
    batch_has_cart = any(_name_in(str(action.get("name") or ""), _CART_NAMES) for action in raw)
    prepared: list[dict[str, Any]] = []

    def emit_probe(product: dict[str, Any] | None, args: dict[str, Any]) -> dict[str, Any] | None:
        ids = {
            "product_id": args.get("product_id") or (product or {}).get("product_id"),
            "sku": args.get("sku") or (product or {}).get("sku"),
        }
        if product is not None and _facts(product) and inspect_tool is not None:
            return _redirect_tool(inspect_tool, ids)
        return _redirect_tool(
            view_tool,
            {"product_id": ids.get("product_id"), "product_ids": [ids["product_id"]]}
            if ids.get("product_id")
            else ids,
        )

    for action in raw:
        name = str(action.get("name") or "")
        tool = _tool_match(tools, name)
        args = _filled_action_args(tool, action.get("args") or {})
        if _name_in(name, _CART_NAMES) and lane == LANE_JUSTIFY:
            product = _lookup_product(catalog, args)
            if product is None or not _facts(product):
                probe = emit_probe(product, args)
                if probe is not None:
                    prepared.append(probe)
                continue
            if not _feasible(product, item, hard):
                continue
            if inspect_tool is not None and not product.get("_inspected"):
                probe = _redirect_tool(inspect_tool, args)
                if probe is not None:
                    prepared.append(probe)
                continue
        if _name_in(name, _ORDER_NAMES):
            product = _lookup_product(catalog, args) or cart_item
            if _cart_is_stale(product, goal) and did_cart:
                redirected = _redirect_tool(remove_tool, args if args.get("product_id") else (product or {}))
                if redirected is not None:
                    prepared.append(redirected)
                continue
            if not did_cart:
                if batch_has_cart:
                    continue
                if lane == LANE_JUSTIFY:
                    probe = emit_probe(product, args)
                    if (
                        product is not None
                        and _facts(product)
                        and _feasible(product, item, hard)
                        and (product.get("_inspected") or inspect_tool is None)
                        and cart_tool is not None
                    ):
                        redirected = _redirect_tool(cart_tool, args)
                        if redirected is not None:
                            prepared.append(redirected)
                        continue
                    if probe is not None:
                        prepared.append(probe)
                    continue
                continue
            if lane == LANE_RANK or not need_reason:
                args.pop("justification", None)
                prepared.append({"name": name, "args": args})
                continue
            if product is None or not _facts(product):
                probe = emit_probe(product, args)
                if probe is not None:
                    prepared.append(probe)
                continue
            if inspect_tool is not None and not product.get("_inspected"):
                probe = _redirect_tool(inspect_tool, args if args.get("product_id") else product)
                if probe is not None:
                    prepared.append(probe)
                continue
            if not _feasible(product, item, hard):
                continue
            claims = _ground_justification(product, goal)
            if not claims:
                continue
            args["justification"] = {"claims": claims}
            prepared.append({"name": name, "args": args})
            continue
        prepared.append({"name": name, "args": args})

    if any(_name_in(row["name"], _CART_NAMES) for row in prepared) and any(
        _name_in(row["name"], _ORDER_NAMES) for row in prepared
    ):
        prepared = [row for row in prepared if not _name_in(row["name"], _ORDER_NAMES)]
    if not prepared:
        for action in raw:
            args = _canonicalize_args(action.get("args") or {})
            probe = emit_probe(_lookup_product(catalog, args), args)
            if probe is not None:
                prepared.append(probe)
                break
    return prepared


def _sanitize_action(
    action: dict[str, Any],
    *,
    lane: str,
    goal: dict[str, Any],
    catalog: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    did_cart: bool = False,
    cart_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prepared = _prepare_actions(
        [action],
        lane=lane,
        goal=goal,
        catalog=catalog,
        tools=tools or [],
        did_cart=did_cart,
        cart_item=cart_item,
    )
    return prepared[0] if prepared else {"name": str(action.get("name") or ""), "args": {}}


def _chat_with_backup(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_calls: int,
) -> tuple[str, list[dict[str, Any]]]:
    last_error: Exception | None = None
    for try_id in _reserve_models(model)[:3]:
        packet = _chat(
            model=try_id,
            messages=messages,
            tools=tools,
            tool_choice="required",
        )
        if packet is None:
            last_error = RuntimeError(f"inference failed for model={try_id}")
            continue
        try:
            message = packet["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            last_error = exc
            continue
        spoken = _assistant_text(message) if isinstance(message, dict) else ""
        raw_calls = list((message or {}).get("tool_calls") or [])[:max_calls]
        parsed = [item for item in (_parse_tool_call(call) for call in raw_calls) if item]
        if parsed or spoken:
            return spoken, parsed
    raise RuntimeError(f"inference request failed ({last_error})")


def _drive_llm(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    query = str(policy.get("query") or problem_data.get("query") or "")
    problem_id = str(problem_data.get("problem_id") or problem_data.get("id") or "problem")
    tools = list(policy.get("tools") or [])
    max_calls = max(1, int(policy.get("max_calls_per_turn") or 1))
    max_steps = max(1, int(policy.get("max_steps") or 20))
    goal = _empty_goal(query)
    lane = _classify_lane(query, goal)
    model = _lane_model(lane)

    transcript: list[dict[str, Any]] = [
        {"role": "system", "content": _compose_system(lane, policy, goal)},
        {"role": "user", "content": query},
    ]
    if goal.get("hard", {}).get("ask_spec"):
        transcript.append(
            {
                "role": "user",
                "content": (
                    "Coach: ask for the firm hardware spec with the message tool "
                    "before searching."
                ),
            }
        )
    if lane == LANE_RANK:
        transcript.append(
            {
                "role": "user",
                "content": (
                    "Coach: compare listed candidates first. place_test_order with "
                    "only product_id and sku."
                ),
            }
        )
    if lane == LANE_JUSTIFY:
        transcript.append(
            {
                "role": "user",
                "content": (
                    "Coach: view facts before the first cart; inspect_stock before "
                    "carting and before ordering. Never cart and order in the same "
                    "turn. Claims allowlist: brand, in_stock, price, and exactly "
                    "one of resolution/screen_size/refresh_rate. Never display, "
                    "size, hdr, or category."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "hybrid": {"lane": lane, "model": model, "provider": _provider()},
        }
    ]
    catalog: list[dict[str, Any]] = []
    prior_actions: list[str] = []
    await_reply = False
    did_cart = False
    cart_item: dict[str, Any] | None = None

    for step in range(1, max_steps + 1):
        coach = _next_coach(
            lane=lane,
            step=step,
            actions=prior_actions,
            goal=goal,
            await_reply=await_reply,
            did_cart=did_cart,
            cart_stale=_cart_is_stale(cart_item, goal) if did_cart else False,
        )
        if coach:
            transcript.append({"role": "user", "content": coach})

        spoken = ""
        parsed: list[dict[str, Any]] = []
        for retry in range(len(_IDLE_PROMPTS) + 1):
            spoken, parsed = _chat_with_backup(
                model=model,
                messages=transcript,
                tools=tools,
                max_calls=max_calls,
            )
            if parsed:
                break
            idle = _IDLE_PROMPTS[min(retry, len(_IDLE_PROMPTS) - 1)]
            transcript.extend(
                [
                    {"role": "assistant", "content": spoken},
                    {"role": "user", "content": idle},
                ]
            )
        else:
            raise RuntimeError("model returned no tool call")

        actions = _prepare_actions(
            parsed,
            lane=lane,
            goal=goal,
            catalog=catalog,
            tools=tools,
            did_cart=did_cart,
            cart_item=cart_item,
        )
        labels = [str(item.get("name") or "") for item in actions]
        openai_calls = []
        for index, item in enumerate(actions):
            call_id = (
                parsed[index].get("id")
                if index < len(parsed) and parsed[index].get("id")
                else f"call-{step}-{index}"
            )
            openai_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": json.dumps(item["args"]),
                    },
                }
            )
        transcript.append(
            {
                "role": "assistant",
                "content": spoken,
                "tool_calls": openai_calls,
            }
        )

        group_id = f"{problem_id}-turn-{step}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": step,
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
        result = _get_proxy().post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        dialogue.append(
            {
                "role": "assistant",
                "content": spoken,
                "tool_calls": actions,
                "environment_result": result,
            }
        )
        prior_actions = labels
        if any(_name_in(name, _CART_NAMES) for name in labels):
            did_cart = True
        if any(_name_in(name, _REMOVE_NAMES) for name in labels):
            did_cart = False
            cart_item = None
        if any(_name_in(name, _MESSAGE_NAMES) for name in labels):
            await_reply = False

        env_calls = result.get("calls") or []
        for index, (tool_call, env_call) in enumerate(zip(openai_calls, env_calls)):
            observation = env_call.get("observation")
            sent = actions[index] if index < len(actions) else {"name": "", "args": {}}
            cart_item = _ingest_llm_observation(observation, sent, catalog, cart_item)
            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _shrink_obs(observation),
                }
            )
            if isinstance(observation, dict) and observation.get("done") is True:
                return dialogue

        shopper = result.get("user_message") or {}
        if isinstance(shopper, dict):
            signal = shopper.get("env_signal") or {}
            if isinstance(signal, dict) and signal.get("kind"):
                cart_item = _apply_market_patch(catalog, signal, cart_item)
            if shopper.get("content"):
                spoken_user = str(shopper["content"])
                transcript.append({"role": "user", "content": spoken_user})
                cart_item, should_reply = _apply_llm_shopper(spoken_user, goal, cart_item)
                if cart_item is not None:
                    _merge_catalog(catalog, [cart_item], viewed=False, inspected=False)
                if _cart_is_stale(cart_item, goal) and did_cart:
                    await_reply = False
                elif should_reply:
                    await_reply = True
    return dialogue


class _Episode:
    def __init__(self, problem_data: dict[str, Any]) -> None:
        environment = problem_data["environment"]
        self.binding = environment["binding"]
        self.policy = environment["policy_view"]
        self.problem_id = str(
            problem_data.get("problem_id") or problem_data.get("id") or "problem"
        )
        self.query = str(self.policy.get("query") or problem_data.get("query") or "")
        self.tools = list(self.policy.get("tools") or [])
        self.max_steps = int(self.policy.get("max_steps") or 20)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.dialogue: list[dict[str, Any]] = []
        self.turn = 0
        self.done = False
        self.justified = False
        self.search_lists: list[list[dict[str, Any]]] = []
        self.candidates: list[dict[str, Any]] = []
        self.bound: list[dict[str, Any] | None] = []
        self.item_index = 0
        self.search_attempt = 0
        self.goal = _empty_goal(self.query)
        self.search_tool = _pick_tool(self.tools, _SEARCH_NAMES)
        self.filter_tool = _pick_tool(self.tools, _FILTER_NAMES)
        self.view_tool = _pick_tool(self.tools, _VIEW_NAMES)
        self.compare_tool = _pick_tool(self.tools, _COMPARE_NAMES)
        self.inspect_tool = _pick_tool(self.tools, _INSPECT_NAMES)
        self.cart_inspect_tool = _pick_tool(self.tools, _CART_INSPECT_NAMES)
        self.cart_tool = _pick_tool(self.tools, _CART_NAMES)
        self.remove_tool = _pick_tool(self.tools, _REMOVE_NAMES)
        self.order_tool = _pick_tool(self.tools, _ORDER_NAMES)
        self.message_tool = _pick_tool(self.tools, _MESSAGE_NAMES)
        self.event_seen = False
        self.need_recover = False
        self.last_actions: list[dict[str, Any]] = []
        self.ordered = False
        self.seen_hydrate: set[str] = set()
        self.seen_inspect: set[str] = set()
        self.asked_spec = False
        self.pending_reply: str | None = None
        self.seen_listings: set[str] = set()
        self.did_cart = False

    def run(self) -> list[dict[str, Any]]:
        self.goal = self._parse_goal()
        self.bound = [None] * len(self.goal["items"])
        for spec in self.goal.get("candidates") or []:
            key = f"{spec['product_id']}::{spec['sku']}"
            if not any(_variant_key(item) == key for item in self.candidates):
                self.candidates.append(
                    {
                        "product_id": spec["product_id"],
                        "sku": spec["sku"],
                        "_hydrated": False,
                    }
                )
        while self.turn < self.max_steps and not self.done:
            actions = self._next_actions()
            if not actions:
                break
            result = self._call_env(actions)
            if result is None:
                raise RuntimeError("environment call failed")
            self._ingest(result)
        return self.dialogue

    def _parse_goal(self) -> dict[str, Any]:
        return _empty_goal(self.query)

    def _item(self) -> dict[str, Any]:
        items = self.goal["items"]
        return items[min(self.item_index, len(items) - 1)]

    def _message_action(self, content: str) -> list[dict[str, Any]]:
        if self.message_tool is None:
            return []
        args = _fill_args(self.message_tool, {"content": content})
        return [{"name": _tool_name(self.message_tool), "args": args}]

    def _next_actions(self) -> list[dict[str, Any]]:
        if self.done or self.ordered:
            return []
        if self.need_recover:
            recovered = self._recover_actions()
            if recovered:
                return recovered
        if self.pending_reply:
            text = self.pending_reply
            self.pending_reply = None
            sent = self._message_action(text)
            if sent:
                return sent
        if self.goal["hard"].get("ask_spec") and not self.asked_spec:
            self.asked_spec = True
            sent = self._message_action(
                "What is the required hardware spec — screen size, RAM, "
                "resolution, or refresh rate?"
            )
            if sent:
                return sent
        if self.goal.get("candidates"):
            return self._ranking_actions()
        item = self._item()
        unbound = next(
            (index for index, bound in enumerate(self.bound) if bound is None),
            None,
        )
        if unbound is None:
            chosen = next((item for item in self.bound if item), None)
            if chosen is not None and self._cart_stale(chosen):
                self.need_recover = True
                recovered = self._recover_actions()
                if recovered:
                    return recovered
                chosen = next((item for item in self.bound if item), None)
            if chosen is not None:
                key = _variant_key(chosen)
                if (
                    self.inspect_tool is not None
                    and not chosen.get("_inspected")
                    and key not in self.seen_inspect
                ):
                    return self._inspect_actions([chosen])
                order = self._order_action(chosen)
                if order:
                    return [order]
            return self._model_actions()
        if unbound != self.item_index:
            self.item_index = unbound
            self.search_attempt = 0
            self.search_lists = []
        if self.search_tool is not None and self.search_attempt < 8 and not self.candidates:
            return self._search_actions(force=self.search_attempt >= 3)
        picked = self._pick_candidate()
        if picked is not None and (
            picked.get("_hydrated") or _facts(picked)
        ):
            key = _variant_key(picked)
            if (
                self.inspect_tool is not None
                and not picked.get("_inspected")
                and key not in self.seen_inspect
            ):
                return self._inspect_actions([picked])
            if (
                self.goal["hard"].get("ask_spec")
                and _wanted_screen(item) is None
            ):
                if self.search_tool is not None and self.search_attempt < 8:
                    return self._search_actions()
                return []
            return self._cart_only(picked)
        kind = str(item.get("type") or "")
        budget = self.goal["hard"].get("budget") or self.goal["hard"].get("price_max")
        brand = _norm(item.get("brand") or self.goal["hard"].get("brand") or "")
        prefs = [_norm(pref) for pref in (item.get("prefs") or []) if _norm(pref)]
        to_view = [
            product
            for product in self._pool()
            if not product.get("_hydrated")
            and str(product.get("product_id") or "") not in self.seen_hydrate
        ]

        def _hydrate_rank(product: dict[str, Any]) -> tuple:
            price = _price_of(product)
            in_budget = True
            if budget is not None and price is not None:
                in_budget = price <= float(budget)
            branded = (not brand) or brand in _norm(product.get("brand")) or brand in _blob(product)
            pref_hit = (not prefs) or any(pref in _blob(product) for pref in prefs)
            return (
                0 if _genuine_kind(product, kind) else 1,
                0 if in_budget else 1,
                0 if branded else 1,
                0 if pref_hit else 1,
            )

        to_view.sort(key=_hydrate_rank)
        to_view = to_view[:8]
        if to_view and (self.view_tool is not None or self.compare_tool is not None):
            return self._hydrate_actions(to_view)
        if picked is not None:
            key = _variant_key(picked)
            if (
                self.inspect_tool is not None
                and not picked.get("_inspected")
                and key not in self.seen_inspect
            ):
                return self._inspect_actions([picked])
            return self._cart_only(picked)
        if self.search_tool is not None and self.search_attempt < 8:
            return self._search_actions(force=self.search_attempt >= 3)
        return []

    def _ranking_actions(self) -> list[dict[str, Any]]:
        bound = self.bound[0] if self.bound else None
        if bound is not None and self._cart_stale(bound):
            self.need_recover = True
            recovered = self._recover_actions()
            if recovered:
                return recovered
            bound = self.bound[0] if self.bound else None
        if bound is not None and not self._cart_stale(bound):
            key = _variant_key(bound)
            if (
                self.inspect_tool is not None
                and not bound.get("_inspected")
                and key not in self.seen_inspect
            ):
                return self._inspect_actions([bound])
            order = self._order_action(bound)
            return [order] if order else []
        pool = self._ranked_pool()
        unviewed = [
            product
            for product in pool
            if not product.get("_hydrated")
            and str(product.get("product_id") or "") not in self.seen_hydrate
        ]
        if unviewed and (self.compare_tool is not None or self.view_tool is not None):
            return self._hydrate_actions(unviewed)
        picked = self._pick_ranked(pool)
        if picked is None:
            uninspected = [
                product
                for product in pool
                if not product.get("_inspected")
                and _variant_key(product) not in self.seen_inspect
            ]
            if uninspected and self.inspect_tool is not None:
                return self._inspect_actions(uninspected)
            return self._search_actions() if self.search_tool is not None else []
        if (
            self.inspect_tool is not None
            and not picked.get("_inspected")
            and _variant_key(picked) not in self.seen_inspect
        ):
            return self._inspect_actions([picked])
        return self._cart_only(picked)

    def _pool(self) -> list[dict[str, Any]]:
        if self.candidates:
            return list(self.candidates)
        if self.search_lists:
            self.candidates = _rrf(self.search_lists)
            return list(self.candidates)
        return []

    def _ranked_pool(self) -> list[dict[str, Any]]:
        wanted = self.goal.get("candidates") or []
        fused = self._pool()
        if not wanted:
            return fused
        by_key = {_variant_key(product): product for product in fused}
        by_pid: dict[str, list[dict[str, Any]]] = {}
        for product in fused:
            by_pid.setdefault(str(product.get("product_id") or ""), []).append(product)
        rows: list[dict[str, Any]] = []
        for spec in wanted:
            key = f"{spec['product_id']}::{spec['sku']}"
            product = by_key.get(key)
            if product is None:
                matches = by_pid.get(spec["product_id"]) or []
                product = next(
                    (
                        item
                        for item in matches
                        if str(item.get("sku") or "") == spec["sku"]
                    ),
                    None,
                )
                if product is None and matches:
                    product = dict(matches[0])
                    product["sku"] = spec["sku"]
            if product is None:
                product = {
                    "product_id": spec["product_id"],
                    "sku": spec["sku"],
                    "_hydrated": False,
                }
            rows.append(product)
        return rows

    def _hydrate_actions(self, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ids: list[str] = []
        for product in products:
            product_id = str(product.get("product_id") or product.get("id") or "")
            if product_id and product_id not in ids:
                ids.append(product_id)
        self.seen_hydrate.update(ids)
        if self.compare_tool is not None and len(ids) >= 2:
            args = _fill_args(self.compare_tool, {"product_ids": ids[:5]})
            return [{"name": _tool_name(self.compare_tool), "args": args}]
        if self.view_tool is None:
            return []
        actions: list[dict[str, Any]] = []
        for product_id in ids:
            args = _fill_args(
                self.view_tool,
                {"product_id": product_id, "product_ids": product_id},
            )
            actions.append({"name": _tool_name(self.view_tool), "args": args})
            if len(actions) >= self.max_calls:
                break
        return actions

    def _inspect_actions(self, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.inspect_tool is None:
            return []
        actions: list[dict[str, Any]] = []
        seen: set[str] = set()
        for product in products:
            key = _variant_key(product)
            if not key or key in seen:
                continue
            seen.add(key)
            self.seen_inspect.add(key)
            args = _fill_args(
                self.inspect_tool,
                {
                    "product_id": product.get("product_id") or product.get("id"),
                    "sku": product.get("sku"),
                },
            )
            if not args:
                continue
            actions.append({"name": _tool_name(self.inspect_tool), "args": args})
            if len(actions) >= self.max_calls:
                break
        return actions

    def _cart_only(self, product: dict[str, Any]) -> list[dict[str, Any]]:
        if self.cart_tool is None:
            self.bound[self.item_index] = product
            self.did_cart = True
            return self._order_action_as_list(product)
        args = _fill_args(
            self.cart_tool,
            {
                "product_id": product.get("product_id") or product.get("id"),
                "sku": product.get("sku"),
                "shop_id": product.get("shop_id"),
                "price": product.get("price"),
            },
        )
        self.bound[self.item_index] = product
        self.did_cart = True
        if self.goal["hard"].get("same_shop") and product.get("shop_id"):
            self.goal["hard"]["shop_id"] = product.get("shop_id")
        return [{"name": _tool_name(self.cart_tool), "args": args}]

    def _recover_actions(self) -> list[dict[str, Any]]:
        self.need_recover = False
        bound = (
            self.bound[self.item_index]
            if self.item_index < len(self.bound)
            else None
        )
        if bound is None or self.remove_tool is None or not self._cart_stale(bound):
            return []
        args = _fill_args(
            self.remove_tool,
            {
                "product_id": bound.get("product_id") or bound.get("id"),
                "sku": bound.get("sku"),
            },
        )
        self.bound[self.item_index] = None
        return [{"name": _tool_name(self.remove_tool), "args": args}]

    def _cart_stale(self, product: dict[str, Any] | None) -> bool:
        return _cart_is_stale(product, self.goal)

    def _search_actions(self, force: bool = False) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        item = self.goal["items"][
            min(self.item_index, len(self.goal["items"]) - 1)
        ]
        if self.search_tool is not None:
            variants = _search_variants(
                item,
                water=bool(self.goal["soft"].get("water")),
                category=self.goal["hard"].get("category"),
            )
            remaining = max(1, self.max_calls - len(actions))
            n_search = 2 if self.search_attempt == 0 and remaining >= 2 else 1
            n_search = min(n_search, max(1, len(variants)))
            for _ in range(n_search):
                query = variants[min(self.search_attempt, len(variants) - 1)]
                self.search_attempt += 1
                wanted: dict[str, Any] = {
                    "query": query,
                    "k": 10,
                    "in_stock": self.search_attempt <= 3,
                    "page": 1,
                }
                shop_id = self.goal["hard"].get("shop_id")
                if self.goal["hard"].get("same_shop") and shop_id:
                    wanted["shop_id"] = shop_id
                low = self.goal["hard"].get("price_min")
                high = self.goal["hard"].get("price_max")
                if low is not None or high is not None:
                    lo = int(low or 0)
                    hi = int(high) if high is not None else ""
                    wanted["price"] = f"{lo}-{hi}"
                args = _fill_args(self.search_tool, wanted)
                if force:
                    args = dict(args)
                    stock_key = next(
                        (
                            key
                            for key in args
                            if key.lower() in {"in_stock", "instock", "available"}
                        ),
                        None,
                    )
                    if stock_key is not None and self.search_attempt >= 3:
                        args[stock_key] = False
                actions.append({"name": _tool_name(self.search_tool), "args": args})
                if len(actions) >= self.max_calls:
                    return actions
                if self.search_attempt >= len(variants):
                    break
        if self.filter_tool is not None and len(actions) < self.max_calls:
            brand = _clean_text(item.get("brand"))
            if not brand and self.search_attempt <= 2:
                brand = _clean_text(item.get("soft_brand") or self.goal["soft"].get("brand"))
            high = self.goal["hard"].get("budget") or self.goal["hard"].get("price_max")
            wanted = {
                "brand": brand or None,
                "category": self.goal["hard"].get("category"),
                "max_price": high,
                "in_stock": True,
                "k": 10,
            }
            args = _fill_args(self.filter_tool, wanted)
            if args:
                actions.append({"name": _tool_name(self.filter_tool), "args": args})
        return actions

    def _pick_candidate(self) -> dict[str, Any] | None:
        fused = self._pool()
        item = self._item()
        if self.goal.get("candidates"):
            return self._pick_ranked(fused)
        spec_needed = bool(_wanted_screen(item)) or self.goal["soft"].get("need_reason") or any(
            _spec_attr(str(attr)) for attr in (item.get("attrs") or [])
        )
        hydrated = [
            product
            for product in fused
            if product.get("_hydrated") or _facts(product)
        ]
        if spec_needed or self.goal["soft"].get("water") or item.get("model"):
            fused = hydrated
        ranked = sorted(
            fused,
            key=lambda product: _match_score(
                product, item, self.goal["hard"], self.goal["soft"]["rank"]
            ),
            reverse=True,
        )
        viable = [
            product
            for product in ranked
            if _match_score(
                product, item, self.goal["hard"], self.goal["soft"]["rank"]
            )
            >= 0
        ]
        if not viable:
            return None
        prefs = [_norm(pref) for pref in (item.get("prefs") or []) if _norm(pref)]
        if prefs:
            matched = [
                product
                for product in viable
                if any(
                    pref in _blob(product) or pref in _norm(product.get("title"))
                    for pref in prefs
                )
            ]
            if matched:
                viable = matched
        viable = _without_price_outliers(viable)
        if not viable:
            return None
        model = _clean_text(item.get("model"))
        color = _clean_text(item.get("color"))
        if model:
            exact = [
                product
                for product in viable
                if _model_matches(product, model)
                and (not color or _color_matches(product, color))
            ]
            if exact:
                return exact[0]
            same_model = [
                product for product in viable if _model_matches(product, model)
            ]
            if same_model:
                return same_model[0]
        if self.goal["soft"].get("water"):
            scored = [
                product for product in viable if _water_score(product) > 0
            ]
            if not scored:
                if self.search_attempt < 6:
                    return None
                return viable[0]
            listings = {
                str(product.get("listing_id") or product.get("product_id"))
                for product in scored
            }
            if len(listings) < 2 and len(viable) > 1:
                extra = [product for product in viable if product not in scored][:1]
                scored = scored + extra
            scored.sort(
                key=lambda product: (
                    _water_score(product),
                    _price_of(product) or 0.0,
                ),
                reverse=True,
            )
            return scored[0]
        soft_brand = _norm(item.get("soft_brand") or self.goal["soft"].get("brand") or "")
        if soft_brand:
            branded = [
                product
                for product in viable
                if soft_brand in _blob(product)
                or soft_brand in _norm(product.get("brand"))
            ]
            if branded:
                return branded[0]
        return viable[0]

    def _pick_ranked(self, pool: list[dict[str, Any]]) -> dict[str, Any] | None:
        budget = self.goal["hard"].get("budget") or self.goal["hard"].get("price_max")
        ranked_pool: list[dict[str, Any]] = []
        for product in pool:
            if not _in_stock(product):
                continue
            price = _price_of(product)
            if budget is not None and price is not None and price > float(budget):
                continue
            ranked_pool.append(product)
        if not ranked_pool:
            return None
        axes = self.goal.get("rank_axes") or list(_DEFAULT_RANK_AXES)
        ranked_pool.sort(key=lambda product: _rank_sort_key(product, axes))
        return ranked_pool[0]

    def _preference_break(
        self, options: list[dict[str, Any]], item: dict[str, Any]
    ) -> dict[str, Any] | None:
        rows = []
        for product in options:
            rows.append(
                {
                    "product_id": product.get("product_id") or product.get("id"),
                    "title": product.get("title"),
                    "price": product.get("price"),
                    "brand": product.get("brand"),
                    "in_stock": _in_stock(product),
                }
            )
        prompt = (
            "Pick exactly one product_id from the table that best matches the "
            "soft preferences. Reply JSON {\"product_id\": \"...\"}.\n"
            f"Goal: {self.query}\nItem: {json.dumps(item, sort_keys=True)}\n"
            f"Preferences: {self.goal['soft']['preferences']}\n"
            f"Candidates: {json.dumps(rows, sort_keys=True)}"
        )
        response = _chat(
            model=_model("strong"),
            messages=[{"role": "user", "content": prompt}],
        )
        if response is None:
            return None
        message = ((response.get("choices") or [{}])[0].get("message")) or {}
        parsed = _loads_json(_assistant_text(message)) or {}
        wanted = str(parsed.get("product_id") or "")
        for product in options:
            if str(product.get("product_id") or product.get("id") or "") == wanted:
                return product
        return None

    def _order_action(
        self, product: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        if self.order_tool is None:
            return None
        chosen = product or next(
            (item for item in reversed(self.bound) if item),
            None,
        )
        wanted: dict[str, Any] = {}
        if chosen is not None:
            wanted = {
                "product_id": chosen.get("product_id") or chosen.get("id"),
                "sku": chosen.get("sku"),
                "product_ids": chosen.get("product_id") or chosen.get("id"),
            }
            if self.goal["soft"].get("need_reason"):
                claims = _ground_justification(chosen, self.goal)
                if claims:
                    wanted["justification"] = {"claims": claims}
        self.ordered = True
        return {
            "name": _tool_name(self.order_tool),
            "args": _fill_args(self.order_tool, wanted),
        }

    def _order_action_as_list(
        self, product: dict[str, Any]
    ) -> list[dict[str, Any]]:
        action = self._order_action(product)
        return [action] if action else []

    def _model_actions(self) -> list[dict[str, Any]]:
        if not self.tools:
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "Use only the supplied shopping tools. "
                    "Search with short keywords, inspect candidates, then complete "
                    "the order. Stop when an observation has done=true."
                ),
            },
            {"role": "user", "content": self.query},
        ]
        response = _chat(
            model=_model("fast"),
            messages=messages,
            tools=self.tools,
            tool_choice="required",
        )
        if response is None:
            return []
        message = ((response.get("choices") or [{}])[0].get("message")) or {}
        calls = (message.get("tool_calls") or [])[: self.max_calls]
        actions = []
        for call in calls:
            function = call.get("function") or {}
            raw = function.get("arguments") or "{}"
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            actions.append({"name": function.get("name"), "args": parsed})
        return [action for action in actions if action.get("name")]

    def _call_env(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        self.turn += 1
        self.last_actions = actions
        group_id = f"{self.problem_id}-turn-{self.turn}"
        envelope = {
            **self.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": self.turn,
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
        result = _get_proxy().post("/environment/call", json_data=envelope)
        self.dialogue.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": actions,
                "environment_result": result,
            }
        )
        return result

    def _ingest(self, result: dict[str, Any]) -> None:
        calls = result.get("calls") or []
        sent = self.last_actions or []
        for index, call in enumerate(calls):
            observation = call.get("observation") or {}
            if observation.get("done") is True:
                self.done = True
            action_name = ""
            if index < len(sent):
                action_name = str(sent[index].get("name") or "")
            lower = action_name.lower()
            viewed = lower in {name.lower() for name in _VIEW_NAMES + _COMPARE_NAMES}
            inspected = lower in {name.lower() for name in _INSPECT_NAMES}
            products = _walk_products(observation)
            if products:
                if viewed or inspected:
                    self._merge_products(
                        products, viewed=viewed, inspected=inspected
                    )
                else:
                    self.search_lists.append(products)
                    self.candidates = _rrf(
                        self.search_lists + ([self.candidates] if self.candidates else [])
                    )
                    self._merge_products(products, viewed=False, inspected=False)
            if lower in {name.lower() for name in _CART_NAMES}:
                added = observation.get("added")
                event = observation.get("market_event")
                if isinstance(added, dict):
                    self._merge_products(
                        [added], viewed=False, inspected=True
                    )
                if isinstance(event, dict) and event:
                    self.event_seen = True
                    self._apply_event(event)
                bound = (
                    self.bound[self.item_index]
                    if self.item_index < len(self.bound)
                    else None
                )
                if self._cart_stale(bound):
                    self.need_recover = True
            error = _norm(observation.get("error") or "")
            if error and any(
                token in error
                for token in ("stock", "unavailable", "price", "budget")
            ):
                self._invalidate_current()
        message = result.get("user_message") or {}
        if isinstance(message, dict):
            signal = message.get("env_signal") or {}
            if isinstance(signal, dict) and signal.get("kind"):
                self.event_seen = True
                self._apply_event(signal)
            content = str(message.get("content") or "")
            if content:
                self._apply_shopper(content)
            bound = (
                self.bound[self.item_index]
                if self.item_index < len(self.bound)
                else None
            )
            if self._cart_stale(bound):
                self.need_recover = True

    def _merge_products(
        self,
        products: list[dict[str, Any]],
        *,
        viewed: bool,
        inspected: bool,
    ) -> None:
        by_key = {_variant_key(item): item for item in self.candidates if _variant_key(item)}
        for product in products:
            key = _variant_key(product)
            if not key:
                continue
            if viewed:
                product["_hydrated"] = True
            if inspected:
                product["_inspected"] = True
            current = by_key.get(key)
            if current is None:
                self.candidates.append(product)
                by_key[key] = product
            else:
                current.update(
                    {
                        field: value
                        for field, value in product.items()
                        if value is not None
                    }
                )
                if viewed:
                    current["_hydrated"] = True
                if inspected:
                    current["_inspected"] = True
            for bound in self.bound:
                if bound is not None and _variant_key(bound) == key:
                    bound.update(
                        {
                            field: value
                            for field, value in product.items()
                            if value is not None
                        }
                    )
                    if viewed:
                        bound["_hydrated"] = True
                    if inspected:
                        bound["_inspected"] = True

    def _apply_event(self, event: dict[str, Any]) -> None:
        target = event.get("target") or {}
        product_id = str(target.get("product_id") or event.get("product_id") or "")
        sku = str(target.get("sku") or event.get("sku") or "")
        kind = _norm(event.get("kind") or "")
        patch: dict[str, Any] = {}
        if product_id:
            patch["product_id"] = product_id
        if sku:
            patch["sku"] = sku
        if "stockout" in kind or event.get("in_stock_after") is False:
            patch["in_stock"] = False
        new_price = event.get("new_price")
        if new_price is not None:
            patch["price"] = new_price
        if patch.get("product_id"):
            self._merge_products([patch], viewed=False, inspected=False)
            key = _variant_key(patch)
            if key:
                self.seen_inspect.discard(key)
            for product in self.candidates:
                if _variant_key(product) == key:
                    product["_inspected"] = False
            for bound in self.bound:
                if bound is not None and _variant_key(bound) == key:
                    bound["_inspected"] = False
                    bound.update(
                        {
                            field: value
                            for field, value in patch.items()
                            if value is not None
                        }
                    )

    def _invalidate_current(self) -> None:
        self.search_attempt = min(self.search_attempt + 1, 5)
        self.justified = False
        self.ordered = False
        bound = (
            self.bound[self.item_index]
            if self.item_index < len(self.bound)
            else None
        )
        if self._cart_stale(bound):
            self.need_recover = True

    def _apply_shopper(self, content: str) -> None:
        lowered = content.lower()
        item = self._item()
        budget = _SHOPPER_BUDGET.search(content) or _BUDGET_NOW.search(content)
        if budget:
            self.goal["hard"]["budget"] = float(budget.group(1))
            self.goal["hard"]["price_max"] = float(budget.group(1))
        inches = _screen_inches_from_text(content)
        if inches is not None:
            attrs = [
                attr
                for attr in (item.get("attrs") or [])
                if _screen_inches_from_text(str(attr)) is None
            ]
            item["attrs"] = [f"{inches:g} inch"] + attrs
            self.search_attempt = 0
            self.asked_spec = True
        bumped = _PRICE_HIKE.search(content)
        bound = (
            self.bound[self.item_index]
            if self.item_index < len(self.bound)
            else None
        )
        if bumped is not None and bound is not None:
            bound["price"] = float(bumped.group(2))
        if bound is not None:
            self.did_cart = True
        if any(
            token in lowered
            for token in (
                "out of stock",
                "sold out",
                "unavailable",
                "no longer available",
            )
        ):
            if bound is not None:
                bound["in_stock"] = False
        if bound is not None and self._cart_stale(bound):
            self.need_recover = True
        if (
            "?" in content
            and not self.need_recover
            and not self.did_cart
            and (bound is None or not self._cart_stale(bound))
        ):
            self.pending_reply = (
                "I'll keep looking for an in-stock item that meets the stated spec and budget."
            )


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Drive shopping tools with a chat model; fall back to the planner."""

    try:
        return _drive_llm(problem_data)
    except (RuntimeError, KeyError, TypeError, ValueError):
        return _Episode(problem_data).run()


__all__ = ["agent_main", "_classify_lane"]
