from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from os import getenv
from typing import Any


_SCOUT = "deepseek/deepseek-v3.2"
_DENSE = "qwen/qwen3.5-397b-a17b"
_PATCH_MODEL = "z-ai/glm-5.2"
_RESERVE = (
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "google/gemma-4-31b-it",
    "deepseek/deepseek-v3.2",
)
_TEE = {
    "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
    "deepseek-ai/DeepSeek-V3.1-TEE": "deepseek/deepseek-chat-v3.1",
    "deepseek-ai/DeepSeek-V3-0324-TEE": "deepseek/deepseek-chat-v3-0324",
    "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
    "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
    "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
    "zai-org/GLM-5.1-TEE": "z-ai/glm-5.1",
    "zai-org/GLM-5.2-TEE": "z-ai/glm-5.2",
    "moonshotai/Kimi-K2.6-TEE": "moonshotai/kimi-k2.6",
    "openai/gpt-oss-120b-TEE": "openai/gpt-oss-120b",
}

DUTY_HUNT = "hunt"
DUTY_FORK = "fork"
DUTY_ASK = "ask"
DUTY_TASTE = "taste"
DUTY_SLATE = "slate"
DUTY_MEND = "mend"
DUTY_WHY = "why"

_CLOCK_S = 281.0
_OBS_CAP = 2450
_DECOY_FRAC = 0.15

_KEY_RX = re.compile(r"(\d{6,})::([A-Za-z0-9._-]+)")
_NUM_RX = re.compile(r"(\d+(?:\.\d+)?)")
_BUDGET_RX = re.compile(
    r"(?:at or under|at or below|under|below|up to|upto|budget|<=|"
    r"at most|no more than|can only go up to|only go up to|new budget)\s*"
    r"([\d][\d,]*(?:\.\d+)?)",
    re.I,
)
_QUOTE_RX = re.compile(r"[\"\u201c]([^\"\u201d]{2,80})[\"\u201d]")
_PHRASE_RX = re.compile(
    r"matches(?:\s+the)?\s+phrase\s+([^,\"\n]+?)(?=\s*,|\s+at\s+or|\s+and\s+buy|$)",
    re.I,
)
_CAT_RX = re.compile(r"\bin the\s+(.+?)\s+category\b", re.I)
_ASK_RX = re.compile(
    r"\bif you ask\b|\bwhen you ask\b|"
    r"(?:tell|share|reveal)[^.]{0,48}\bif\b[^.]{0,24}\bask|"
    r"ask (?:me |for )?(?:the |that )?(?:exact )?(?:requirement|spec)",
    re.I,
)
_WHY_RX = re.compile(
    r"\bwhy\b|walk me through|justif|reason i can trust|give me a reason|"
    r"tell me why|structured justification|trustworthy reason|"
    r"why it(?:'s| is) a good pick",
    re.I,
)
_OPTIONAL_WHY_RX = re.compile(r"justification is optional", re.I)
_PATCH_RX = re.compile(
    r"unavailable|sold out|stockout|disappear|least-relax|least relax|"
    r"other colour|other color|try another color|same model.{0,24}other",
    re.I,
)
_PAIR_RX = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})", re.I)
_EVENT_RX = re.compile(
    r"more expensive|still available|sold out|out of stock|"
    r"just get more|price (?:from|went|changed)|repriced",
    re.I,
)
_UNIT_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*"
    r"(inch(?:es)?|in\b|hz|hertz|gb|tb|mah|watt(?:s)?|w\b|mm|cm|"
    r"oz|ml|kg|g\b|dpi|rpm|\"|'')?",
    re.I,
)
_MECH = (
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
    """
    a an the and or for to of in on at as is it if me my one buy get find
    place test order listing category matches phrase stock instock under
    below budget php remaining please can you with that this from than
    then not just here what actually rather bit more i id its have has had
    be been being was were will would should could must only also still even
    any all both each few most other some such too very so nor because until
    while about against between through during before after above up down
    out off over again further once need want ideally prefer firm
    requirement requirements matter matters sorry hold cheap cheapest
    goal settle something flimsy daily every kind really sessions lots year
    round grab save saving money device actual accessory extras nice haves
    but spend checked balance goes going pay care shopper user agent
    available within item items product products these those same another
    there when after before upon looking hello hi
    """.split()
)
_UNIT_CANON = {
    "inch": "in",
    "inches": "in",
    "in": "in",
    '"': "in",
    "''": "in",
    "hz": "hz",
    "hertz": "hz",
    "gb": "gb",
    "tb": "tb",
    "mah": "mah",
    "w": "w",
    "watt": "w",
    "watts": "w",
    "mm": "mm",
    "cm": "cm",
    "oz": "oz",
    "ml": "ml",
    "kg": "kg",
    "g": "g",
    "dpi": "dpi",
    "rpm": "rpm",
}
_JUNK = (
    "case for",
    "cover for",
    "compatible with",
    "replacement only",
    "sticker",
    "decoration",
    "pouch",
    "notebook",
    "sketch book",
    "ballpoint",
    "refill",
    "leather case",
    "flip stand",
)
_CLAIM_BLOCK = {
    "display",
    "size",
    "category",
    "category_path",
    "device_operating_system",
    "operating_system",
    "os",
    "title",
    "summary",
    "evidence",
    "reason",
    "description",
    "name",
    "note",
}
_CLAIM_SOFT = frozenset({"price", "in_stock", "brand", "category"})
_FIELD_ALIAS = {
    "stock": "in_stock",
    "availability": "in_stock",
    "cost": "price",
    "maker": "brand",
}
_IDLE = (
    "No tool call received. Choose a supplied tool now. "
    "Search or filter with the shopper's own value words, not schema keys.",
    "Still no tool call. Search/filter, view a candidate, answer a pending "
    "shopper question, or place_test_order if the cart holds a verified SKU.",
)
_COLORS = (
    "midnight", "champagne", "graphite", "titanium", "silver", "charcoal",
    "purple", "orange", "yellow", "beige", "white", "black", "green", "blue",
    "navy", "gold", "pink", "grey", "gray", "red", "brown", "ivory",
)

_PROXY = None
_STARTED = 0.0


def _proxy():
    global _PROXY
    if _PROXY is None:
        from src.agent.proxy_client import ProxyClient

        _PROXY = ProxyClient(timeout=45, max_retries=1)
    return _PROXY


def _arm_clock() -> None:
    global _STARTED
    _STARTED = time.monotonic()


def _left() -> float:
    if not _STARTED:
        return _CLOCK_S
    return _CLOCK_S - (time.monotonic() - _STARTED)


def _provider() -> str:
    return "openrouter"


def _normalize_model(raw: str) -> str:
    text = (raw or "").strip()
    return _TEE.get(text, text) or _SCOUT


def _primary_model(duties: set[str]) -> str:
    override = getenv("SANDBOX_MODEL") or ""
    if override.strip():
        return _normalize_model(override)
    if DUTY_MEND in duties and DUTY_SLATE not in duties:
        return _PATCH_MODEL
    if duties & {DUTY_WHY, DUTY_ASK}:
        return _DENSE
    return _SCOUT


def _reserve_models(primary: str) -> list[str]:
    ordered = [primary]
    if _left() < 70:
        return ordered
    for item in _RESERVE:
        if item not in ordered:
            ordered.append(item)
    return ordered[:2]


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fold(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _tokens(text: str) -> list[str]:
    return [token for token in _fold(text).split() if len(token) > 1 and token not in _STOP]


def _fn(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    return function if isinstance(function, dict) else tool


def _tool_name(tool: dict[str, Any] | None) -> str:
    if not tool:
        return ""
    return str(_fn(tool).get("name") or "")


def _props(tool: dict[str, Any] | None) -> dict[str, Any]:
    if not tool:
        return {}
    parameters = _fn(tool).get("parameters") or {}
    properties = parameters.get("properties") or {}
    return properties if isinstance(properties, dict) else {}


def _find_tool(tools: list[dict[str, Any]], *needles: str) -> dict[str, Any] | None:
    wanted = {item.lower() for item in needles}
    for tool in tools:
        name = _tool_name(tool).lower()
        if name in wanted:
            return tool
    for tool in tools:
        name = _tool_name(tool).lower()
        if any(needle in name for needle in wanted):
            return tool
    return None


def _coerce(value: Any, spec: dict[str, Any]) -> Any:
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


def _bind(tool: dict[str, Any] | None, wanted: dict[str, Any]) -> dict[str, Any]:
    if tool is None:
        return {key: value for key, value in wanted.items() if value is not None}
    properties = _props(tool)
    if not properties:
        return {key: value for key, value in wanted.items() if value is not None}
    lower = {name.lower(): name for name in properties}
    aliases = {
        "query": ("query", "q", "text", "keywords"),
        "k": ("k", "limit", "n", "top_k"),
        "in_stock": ("in_stock", "instock", "available"),
        "product_id": ("product_id", "productid", "id"),
        "sku": ("sku", "variant", "variant_id"),
        "product_ids": ("product_ids", "ids"),
        "content": ("content", "message", "text"),
        "max_price": ("max_price", "price_max", "budget"),
        "brand": ("brand",),
        "category": ("category",),
        "justification": ("justification",),
        "price": ("price",),
        "shop_id": ("shop_id", "shopid", "store_id"),
    }
    filled: dict[str, Any] = {}
    for logical, value in wanted.items():
        if value is None:
            continue
        for alias in aliases.get(logical, (logical,)):
            real = lower.get(alias.lower())
            if real is None:
                continue
            filled[real] = _coerce(value, properties.get(real) or {})
            break
    return filled


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _shopper_text(query: str) -> str:
    return re.split(r"\n\s*Task rules:", query or "", maxsplit=1)[0].strip()


def _rules_text(query: str) -> str:
    parts = re.split(r"\n\s*Task rules:", query or "", maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _named_keys(text: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for product_id, sku in _KEY_RX.findall(text or ""):
        sku = sku.rstrip(".,;:")
        key = f"{product_id}::{sku}"
        if not sku or key in seen:
            continue
        seen.add(key)
        found.append({"product_id": product_id, "sku": sku})
    return found


def _listing_key(product: dict[str, Any]) -> str:
    pid = str(product.get("product_id") or "")
    sku = str(product.get("sku") or "")
    if pid and sku:
        return f"{pid}::{sku}"
    return pid


def _facts(product: dict[str, Any]) -> dict[str, Any]:
    facts = product.get("facts") or product.get("specification") or product.get("attributes")
    return facts if isinstance(facts, dict) else {}


def _options_text(product: dict[str, Any]) -> str:
    options = product.get("options")
    if isinstance(options, dict):
        return " ".join(str(value) for value in options.values() if value not in (None, ""))
    if isinstance(options, (list, tuple)):
        return " ".join(str(item) for item in options if item not in (None, ""))
    return str(options or "").strip()


def _blob(product: dict[str, Any]) -> str:
    facts = _facts(product)
    option_text = _options_text(product)
    path = product.get("category_path") or product.get("category") or ""
    if isinstance(path, list):
        path = " ".join(str(item) for item in path)
    return " ".join(
        str(part)
        for part in (
            product.get("title"),
            product.get("brand"),
            product.get("description"),
            option_text,
            path,
            " ".join(str(value) for value in facts.values()),
        )
        if part
    )


def _number(text: Any) -> float | None:
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return float(text)
    match = _NUM_RX.search(str(text or "").replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _price_of(product: dict[str, Any]) -> float | None:
    for key in ("price", "current_price", "sale_price"):
        raw = product.get(key)
        if isinstance(raw, dict):
            raw = raw.get("value") or raw.get("amount") or raw.get("current")
        value = _number(raw)
        if value is not None:
            return value
    return _number(_facts(product).get("price"))


def _in_stock(product: dict[str, Any]) -> bool:
    if "in_stock" in product:
        flag = product.get("in_stock")
        if isinstance(flag, bool):
            return flag
        if isinstance(flag, (int, float)):
            return bool(flag)
        if isinstance(flag, str) and flag.strip():
            return flag.strip().lower() not in {"0", "false", "no", "oos", "out"}
    return True


def _price_text(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _canon_unit(raw: str) -> str:
    return _UNIT_CANON.get((raw or "").lower().strip(), (raw or "").lower().strip())


def _figure_spans(text: str, budget: float | None = None) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _UNIT_RX.finditer(text or ""):
        try:
            amount = float(match.group(1))
        except ValueError:
            continue
        if budget is not None and abs(amount - budget) < 1e-6:
            continue
        unit = _canon_unit(match.group(2) or "")
        if not unit:
            continue
        raw = _clean(match.group(0))
        key = f"{amount:g}|{unit}"
        if key in seen:
            continue
        seen.add(key)
        found.append({"amount": amount, "unit": unit, "raw": raw})
    return found


def _wants_reason(query: str) -> bool:
    shopper = _shopper_text(query)
    full = query or ""
    why = bool(_WHY_RX.search(shopper))
    if _OPTIONAL_WHY_RX.search(full) and not re.search(
        r"\bwhy\b|walk me through|reason i can trust|tell me why",
        shopper,
        re.I,
    ):
        return False
    if _OPTIONAL_WHY_RX.search(full) and not why:
        return False
    return why or bool(_WHY_RX.search(full) and not _OPTIONAL_WHY_RX.search(full))


def _held_back(query: str) -> bool:
    return bool(_ASK_RX.search(_shopper_text(query)) or _ASK_RX.search(query or ""))


def _quoted(query: str) -> list[str]:
    shopper = _shopper_text(query)
    quotes = [_clean(item) for item in _QUOTE_RX.findall(shopper)]
    if not quotes:
        hit = _PHRASE_RX.search(shopper)
        if hit:
            quotes.append(_clean(hit.group(1)))
    return [item for item in quotes if item]


def _category_hint(query: str) -> str:
    hit = _CAT_RX.search(_shopper_text(query))
    return _clean(hit.group(1)) if hit else ""


def _kind_noun(query: str) -> str:
    """Noun from this brief only — no catalog of device classes."""
    shopper = _shopper_text(query)
    folded = _fold(shopper)
    cat = _CAT_RX.search(shopper)
    if cat:
        parts = _tokens(cat.group(1))
        if parts:
            return parts[-1]
    tail = re.search(
        r"([a-z][a-z]{2,})[,.]?\s+(?:at or under|at or below|in stock)",
        folded,
    )
    if tail and tail.group(1) not in _STOP:
        return tail.group(1)
    brand = _fold(_brand_guess(query))
    head = re.search(
        r"(?:need|want|find|get|buy)\s+(?:me\s+)?(?:an?\s+)?(?:in stock\s+)?"
        r"(.+?)(?:\s+with\b|\s+at or\b|\s+in stock|\s+that\b|,|$)",
        folded,
    )
    if head:
        words = [word for word in _tokens(head.group(1)) if word != brand and word not in _STOP]
        if words:
            return words[-1]
    return ""


def _brand_guess(query: str) -> str:
    shopper = _shopper_text(query)
    hit = re.search(
        r"preferred\s+brand\s+([A-Za-z0-9][\w.&'-]{0,40})",
        shopper,
        re.I,
    )
    if hit:
        phrase = _clean(hit.group(1)).rstrip(".")
        if _fold(phrase) not in _STOP:
            return phrase
    hit = re.search(
        r"(?:[Pp]refer(?:red)?|[Ii]deally)\s+(?:the\s+)?([A-Z][A-Za-z0-9.&-]{2,})",
        shopper,
    )
    if hit:
        phrase = _clean(hit.group(1)).rstrip(".")
        if _fold(phrase) not in _STOP:
            return phrase
    hit = re.search(
        r"(?i:need|want|find|get)\s+(?:me\s+)?(?:an?\s+)?([A-Z][A-Za-z0-9.&-]{2,})",
        shopper,
    )
    if hit:
        phrase = _clean(hit.group(1)).rstrip(".")
        if _fold(phrase) not in _STOP:
            return phrase
    return ""


def _soft_brand(query: str) -> bool:
    shopper = _fold(_shopper_text(query))
    if _PATCH_RX.search(shopper):
        return False
    return ("prefer" in shopper or "ideally" in shopper) and (
        "firm" in shopper or "requirement" in shopper or "soft" in shopper
    )


def _model_guess(query: str) -> str:
    shopper = _shopper_text(query)
    hues = "|".join(re.escape(item) for item in _COLORS)
    hit = re.search(
        rf"\bprefer(?:s|red)?(?:\s+the)?\s+(?:(?:{hues})\s+)*([A-Za-z0-9][A-Za-z0-9_-]{{1,32}})",
        shopper,
        re.I,
    )
    if hit:
        token = hit.group(1).rstrip(".,;")
        if _fold(token) not in _STOP and _fold(token) not in _COLORS:
            return token
    hit = re.search(
        r"\bmodel(?:\s+code)?\s+([A-Za-z0-9._-]{2,})",
        shopper,
        re.I,
    )
    if hit:
        return hit.group(1).rstrip(".,;")
    return ""


def _color_guess(query: str) -> str:
    folded = _fold(_shopper_text(query))
    for hue in _COLORS:
        if re.search(rf"\b{re.escape(hue)}\b", folded):
            return hue
    return ""


def _axes_from(text: str) -> list[tuple[str, str]]:
    axes: list[tuple[str, str]] = []
    blob = text or ""
    patterns = (
        (r"highest\s+([a-z][a-z0-9_ /-]{2,32})", "desc"),
        (r"largest\s+([a-z][a-z0-9_ /-]{2,32})", "desc"),
        (r"most\s+([a-z][a-z0-9_ /-]{2,32})", "desc"),
        (r"lowest\s+([a-z][a-z0-9_ /-]{2,32})", "asc"),
        (r"smallest\s+([a-z][a-z0-9_ /-]{2,32})", "asc"),
        (r"cheapest|lowest price|then(?:\s+the)?\s+price", "price_asc"),
    )
    for pattern, direction in patterns:
        for match in re.finditer(pattern, blob, re.I):
            if direction == "price_asc":
                axes.append(("price", "asc"))
                continue
            name = _canon_axis(match.group(1))
            if name:
                axes.append((name, direction))
    cleaned: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, direction in axes:
        key = f"{name}:{direction}"
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((name, direction))
    return cleaned


def _canon_axis(raw: str) -> str:
    folded = _fold(raw)
    if "price" in folded or "cost" in folded:
        return "price"
    named = _norm_field(folded)
    return named or ((_tokens(raw) or [""])[0])


def _facts_blob(product: dict[str, Any]) -> str:
    option = _options_text(product)
    facts = " ".join(str(value) for value in _facts(product).values() if value not in (None, ""))
    return " ".join(part for part in (option, facts) if part)


def _axis_blob(product: dict[str, Any]) -> str:
    title = str(product.get("title") or "")
    return " ".join(part for part in (_facts_blob(product), title) if part)


def _axis_value(product: dict[str, Any], name: str) -> float | None:
    if name == "price":
        return _price_of(product)
    facts = _facts(product)
    focused = _axis_blob(product)
    blob = focused or _blob(product)
    if name == "refresh_rate":
        hits = [float(item) for item in re.findall(r"(\d+(?:\.\d+)?)\s*(?:hz|hertz)\b", blob, re.I)]
        if hits:
            return max(hits)
    if name == "screen_size":
        hit = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(?:inch|in\b|\")", focused, re.I)
        if hit:
            return float(hit.group(1))
        hit = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(?:inch|in\b|\")", blob, re.I)
        if hit:
            return float(hit.group(1))
    if name == "resolution":
        pair = _PAIR_RX.search(blob)
        if pair:
            return float(int(pair.group(1)) * int(pair.group(2)))
        hit = re.search(r"(\d{3,4})\s*p\b", blob, re.I)
        if hit:
            return float(hit.group(1))
    raw = facts.get(name) or product.get(name)
    return _number(raw)


def _lex_tuple(product: dict[str, Any], axes: list[tuple[str, str]]) -> tuple:
    used = axes or [("price", "asc")]
    parts: list[float] = []
    for name, direction in used:
        value = _axis_value(product, name)
        if value is None:
            parts.append(1e18 if direction == "desc" else -1e18)
        else:
            parts.append(-value if direction == "desc" else value)
    price = _price_of(product)
    parts.append(price if price is not None else 1e12)
    return tuple(parts)


def _lex_rank(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if not pool:
        return []
    return sorted(pool, key=lambda product: _lex_tuple(product, axes))


def _wish_terms(query: str) -> list[str]:
    shopper = _shopper_text(query)
    found: list[str] = []
    for match in re.finditer(
        r"(?:matters more|rather(?:\s+pay)?|use-case|use case|"
        r"need it to|must feel|should be|intended use)\s+([^.;]{3,48})",
        shopper,
        re.I,
    ):
        found.extend(_tokens(match.group(1)))
    folded = _fold(shopper)
    for stem in (
        "waterproof", "water resistant", "sweat", "workout", "splash",
        "quiet", "durable", "lightweight", "portable", "organic",
        "unscented", "cordless", "refillable",
    ):
        if any(token in folded for token in stem.split()):
            found.extend(stem.split())
    cleaned: list[str] = []
    seen: set[str] = set()
    for token in found:
        folded_tok = _fold(token)
        if not folded_tok or folded_tok in seen or folded_tok in _STOP:
            continue
        seen.add(folded_tok)
        cleaned.append(token)
    return cleaned[:8]


def _taste_cover(product: dict[str, Any], wishes: list[str]) -> float:
    if not wishes:
        return 0.0
    blob = _fold(_blob(product))
    hits = sum(1 for wish in wishes if _fold(wish) in blob)
    return hits / max(len(wishes), 1)


def _decoy(product: dict[str, Any], budget: Any) -> bool:
    price = _price_of(product)
    if price is None or budget in (None, ""):
        return False
    try:
        ceiling = float(budget)
    except (TypeError, ValueError):
        return False
    if ceiling <= 0:
        return False
    return price <= ceiling * _DECOY_FRAC


def _phrase_hit(product: dict[str, Any], phrase: str) -> bool:
    title = _fold(product.get("title") or "")
    needle = _fold(phrase)
    if not needle or not title:
        return False
    return all(token in title for token in needle.split())


def _rejected_in_title(product: dict[str, Any], query: str) -> bool:
    rules = _rules_text(query)
    hit = re.search(r"reject conflicting ([^.]+)", rules, re.I)
    if not hit:
        return False
    title = _fold(product.get("title") or "")
    if not title:
        return False
    for part in re.split(r",|/|\bor\b", hit.group(1)):
        token = _fold(re.sub(r"[^a-z0-9]+", " ", part)).strip()
        if token and token not in _STOP and len(token) >= 4 and token in title:
            return True
    return False


def _os_matches_brief(product: dict[str, Any], query: str) -> bool:
    blob = f"{_rules_text(query)} {_shopper_text(query)}"
    hit = re.search(r"device os must identify ([^.]+)", _fold(blob))
    if not hit:
        return False
    names = [
        _fold(part)
        for part in re.split(r",| or ", hit.group(1))
        if _fold(part) and _fold(part) not in _STOP and len(_fold(part)) > 2
    ]
    if not names:
        return False
    facts = _facts(product)
    os_val = _fold(
        " ".join(
            str(facts.get(key) or "")
            for key in (
                "device_operating_system",
                "operating_system",
                "description_operating_system",
                "phone_type",
            )
        )
    )
    return any(name in os_val for name in names)


def _kind_ok(product: dict[str, Any], kind: str, *, strict: bool = False, query: str = "") -> bool:
    if not kind:
        return True
    if query and _rejected_in_title(product, query):
        return False
    needle = _fold(kind)
    title = _fold(product.get("title") or "")
    if needle and needle in title:
        return True
    option = _options_text(product)
    facts = " ".join(str(value) for value in _facts(product).values() if value not in (None, ""))
    grounded = _fold(f"{option} {facts}")
    if needle and needle in grounded:
        return True
    if query and _os_matches_brief(product, query) and needle and needle in grounded:
        return True
    if strict:
        return False
    return needle in _fold(_blob(product)) if needle else True


def _junk(product: dict[str, Any], kind: str) -> bool:
    title = _fold(product.get("title") or "")
    if any(marker in title for marker in _JUNK):
        if kind and _fold(kind) not in title.split()[:3]:
            return True
    return False


def _brand_hit(product: dict[str, Any], brand: str) -> bool:
    if not brand:
        return False
    return _fold(brand) in _fold(_blob(product))


def _model_hit(product: dict[str, Any], model: str) -> bool:
    if not model:
        return False
    token = str(model).strip()
    if not token:
        return False
    hay = " ".join(
        str(part)
        for part in (
            product.get("title"),
            product.get("sku"),
            _options_text(product),
            " ".join(str(value) for value in _facts(product).values()),
        )
        if part
    )
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", hay, re.I) is not None


def _color_hit(product: dict[str, Any], color: str) -> bool:
    if not color:
        return False
    return re.search(rf"\b{re.escape(_fold(color))}\b", _fold(_blob(product))) is not None


def _ladder(product: dict[str, Any], model: str, color: str) -> int:
    has_model = (not model) or _model_hit(product, model)
    has_color = (not color) or _color_hit(product, color)
    if has_model and has_color:
        return 0
    if has_model:
        return 1
    return 2


def _amount_near(got: float, want: float) -> bool:
    if abs(got - want) <= 0.02:
        return True
    if abs(want - round(want)) < 1e-9 and abs(got - want) <= 0.15:
        return True
    return False


def _span_in(hay: str, span: dict[str, Any]) -> bool:
    amount = span.get("amount")
    unit = str(span.get("unit") or "")
    if amount is None:
        return False
    text = str(hay or "")
    if not text:
        return False
    unit_fold = _fold(unit)
    for match in re.finditer(r"(\d+(?:\.\d+)?)", text.replace(",", "")):
        try:
            got = float(match.group(1))
        except ValueError:
            continue
        if not _amount_near(got, float(amount)):
            continue
        if not unit:
            return True
        window = _fold(text[max(0, match.start() - 4) : match.end() + 10])
        if unit_fold and (unit_fold in window or unit_fold in _compact(window)):
            return True
        if unit_fold == "in" and re.search(r"\binch|\bin\b|\"", window):
            return True
    return False


def _span_in_facts(product: dict[str, Any], span: dict[str, Any]) -> bool:
    amount = span.get("amount")
    if amount is None:
        return False
    want = float(amount)
    facts = _facts(product)
    keys = list(facts.items())
    options = product.get("options")
    if isinstance(options, dict):
        keys.extend(options.items())
    for key, raw in keys:
        value = _number(raw)
        if value is None or not _amount_near(value, want):
            continue
        folded_key = _fold(key)
        unit = str(span.get("unit") or "")
        if not unit:
            return True
        if unit == "in" and any(token in folded_key for token in ("size", "screen", "inch", "display")):
            return True
        if unit and unit in folded_key:
            return True
        if _span_in(f"{key} {raw}", span):
            return True
    return False


def _measure_hit(product: dict[str, Any], spans: list[dict[str, Any]]) -> bool:
    if not spans:
        return True
    option = _options_text(product)
    facts = " ".join(str(value) for value in _facts(product).values())
    grounded = f"{option} {facts}"
    title = str(product.get("title") or "")
    for span in spans:
        if _span_in_facts(product, span) or _span_in(grounded, span):
            continue
        if grounded.strip() and not _span_in(title, span):
            return False
        if not grounded.strip() and not _span_in(_blob(product), span):
            return False
    return True


def _rule_floors(query: str) -> dict[str, tuple[int, int]]:
    """Pixel-class floors copied from THIS episode's rules, not a device table."""
    rules = _fold(_rules_text(query))
    floors: dict[str, tuple[int, int]] = {}
    for match in re.finditer(
        r"\b([a-z]{2,12})\s+starts at\s+(\d{3,5})\s+by\s+(\d{3,5})",
        rules,
    ):
        floors[_fold(match.group(1)).replace(" ", "")] = (int(match.group(2)), int(match.group(3)))
    for match in re.finditer(
        r"\b([a-z]{2,12}(?: [a-z]{2,12})?)\s+at\s+(\d{3,5})\s+by\s+(\d{3,5})",
        rules,
    ):
        name = _fold(match.group(1)).replace(" ", "")
        if name in {"starts", "and", "or", "from", "with"} or "start" in name:
            continue
        floors[name] = (int(match.group(2)), int(match.group(3)))
    return floors


def _shopper_class_tokens(query: str) -> list[str]:
    shopper = _shopper_text(query)
    folded = _fold(shopper)
    tokens: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        key = _fold(token).replace(" ", "")
        if key and key not in seen:
            seen.add(key)
            tokens.append(key)

    for name in _rule_floors(query):
        if name and name in _compact(folded):
            add(name)
    for match in re.finditer(r"\b((?:full\s+)?hd)\b", shopper, re.I):
        add(match.group(1))
    for match in re.finditer(r"(\d{3,4})\s*p\b", folded):
        add(f"{match.group(1)}p")
    return tokens


def _span_token(span: dict[str, Any]) -> str:
    amount = span.get("amount")
    unit = str(span.get("unit") or "")
    if amount is None:
        return ""
    number = float(amount)
    if abs(number - round(number)) < 0.05:
        body = str(int(round(number)))
    else:
        body = f"{number:g}"
    return f"{body}{unit}" if unit else body


def _listing_meets_class(product: dict[str, Any], query: str) -> bool:
    tokens = _shopper_class_tokens(query)
    if not tokens:
        return True
    grounded = _facts_blob(product)
    hay_source = grounded if (product.get("_hydrated") and grounded.strip()) else (
        grounded + " " + _blob(product)
    )
    hay = re.sub(r"\bhdmi\b", " ", _fold(hay_source))
    compact = _compact(hay)
    pair = _PAIR_RX.search(hay_source)
    got: tuple[int, int] | None = None
    if pair:
        left, right = int(pair.group(1)), int(pair.group(2))
        got = (max(left, right), min(left, right))
    floors = _rule_floors(query)
    for token in tokens:
        if token in hay.split() or token in compact:
            return True
        if token.endswith("p") and token in compact:
            return True
        floor = floors.get(token)
        if floor and got and got[0] >= floor[0] and got[1] >= floor[1]:
            return True
        p_hit = re.match(r"(\d{3,4})p$", token)
        if p_hit and got and min(got) >= int(p_hit.group(1)):
            return True
    return False


def _class_ok(product: dict[str, Any], query: str) -> bool:
    return _listing_meets_class(product, query)


def _spec_ok(product: dict[str, Any], query: str, spans: list[dict[str, Any]] | None = None) -> bool:
    if not _figure_ok(product, query, spans):
        return False
    return _class_ok(product, query)


def _figure_ok(product: dict[str, Any], query: str, spans: list[dict[str, Any]] | None = None) -> bool:
    needed = spans if spans is not None else _figure_spans(_shopper_text(query))
    if not needed:
        return True
    return _measure_hit(product, needed)


def _sku_stem(sku: Any) -> str:
    text = str(sku or "")
    return text.rsplit("-", 1)[0] if text else ""


def _norm_field(raw: str) -> str:
    text = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return re.sub(r"_+", "_", text).strip("_")


def _claim_enum(tool: dict[str, Any] | None) -> list[str]:
    spec = (_props(tool).get("justification") or {}).get("properties") or {}
    claims = spec.get("claims") or {}
    items = claims.get("items") or {}
    field = (items.get("properties") or {}).get("field") or {}
    enum = field.get("enum")
    if isinstance(enum, list) and enum:
        found: list[str] = []
        seen: set[str] = set()
        for item in enum:
            key = _norm_field(str(item))
            if key and key not in seen:
                seen.add(key)
                found.append(key)
        return found
    return []


def _fact_rows(product: dict[str, Any]) -> dict[str, str]:
    rows: dict[str, str] = {}
    price = _price_of(product)
    if price is not None:
        rows["price"] = _price_text(price)
    rows["in_stock"] = "true" if _in_stock(product) else "false"
    brand = _clean(product.get("brand") or "")
    if brand:
        rows["brand"] = brand
    for key, raw in _facts(product).items():
        fl = _norm_field(str(key))
        text = _clean(raw)
        if not fl or not text:
            continue
        rows[fl] = text
        alias = _FIELD_ALIAS.get(fl)
        if alias and alias not in rows:
            rows[alias] = text
    options = product.get("options")
    if isinstance(options, dict):
        for key, raw in options.items():
            fl = _norm_field(str(key))
            text = _clean(raw)
            if fl and text and fl not in rows:
                rows[fl] = text
                alias = _FIELD_ALIAS.get(fl)
                if alias and alias not in rows:
                    rows[alias] = text
    option_text = _options_text(product)
    if option_text and "_option" not in rows:
        rows["_option"] = option_text
    return rows


def _key_in_text(key: str, text: str) -> bool:
    fl = _norm_field(key)
    if not fl or not text:
        return False
    blob = text.lower()
    spaced = fl.replace("_", " ")
    compact = _compact(blob)
    key_compact = fl.replace("_", "")
    if fl in blob.replace("-", "_") or spaced in blob:
        return True
    if len(key_compact) >= 3 and key_compact in compact:
        return True
    return False


def _value_attested(value: str, goal: str) -> bool:
    blob = (goal or "").lower()
    if not value or not blob:
        return False
    compact = _compact(blob)
    for tok in re.findall(r"[a-z]{2,}|\d+(?:\.\d+)?", str(value).lower()):
        if tok in blob or (len(tok) >= 2 and tok in compact):
            return True
    return False


def _listing_claim_text(product: dict[str, Any], *, grounded: bool = False) -> str:
    facts = _facts(product)
    bits = []
    if not grounded:
        bits.append(str(product.get("title") or ""))
    bits.extend([str(product.get("options") or ""), str(product.get("brand") or "")])
    bits.extend(f"{key}: {value}" for key, value in facts.items() if value not in (None, ""))
    return " | ".join(bit for bit in bits if bit)


def _text_supports(value: str, listing: str) -> bool:
    tight = _compact(value)
    blob = _compact(listing)
    return bool(tight) and tight in blob


def _schema_claim_fields(tool: dict[str, Any] | None) -> list[str]:
    """Live enum only. Never scrape description prose for field names."""
    return _claim_enum(tool)


def _schema_extra_reasons(tool: dict[str, Any] | None) -> list[str]:
    spec = (_props(tool).get("justification") or {}).get("properties") or {}
    claims = spec.get("claims") or {}
    items = claims.get("items") or {}
    field = (items.get("properties") or {}).get("field") or {}
    desc = str(field.get("description") or "")
    hit = re.search(r"reason:\s*([^.]+)", desc, re.I)
    found: list[str] = []
    seen: set[str] = set()
    blob = hit.group(1) if hit else "brand"
    for part in re.split(r",|/|\bor\b", blob):
        key = _norm_field(part)
        if not key or key in _CLAIM_BLOCK or key in seen or len(key) < 3:
            continue
        seen.add(key)
        found.append(key)
    return found or ["brand"]


def _spec_keys_from_brief(query: str) -> list[str]:
    """Snake-case keys from this episode's rules parentheticals only."""
    keys: list[str] = []
    seen: set[str] = set()
    for chunk in re.findall(r"\(([^)]{8,220})\)", _rules_text(query)):
        for part in re.split(r",|/|\bor\b", chunk):
            words = [word for word in _tokens(part) if word not in _STOP]
            if not words:
                continue
            key = _norm_field(" ".join(words[-2:] if len(words) > 1 else words))
            if not key or len(key) < 3 or key in _CLAIM_SOFT or key in _CLAIM_BLOCK or key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def _align_fact_key(fact_key: str, brief: list[str], shopper: str, query: str = "") -> str:
    fl = _norm_field(fact_key)
    if not fl:
        return ""
    for key in brief:
        if fl == key or fl in key.split("_") or key.endswith(fl):
            return key
    if fl in {"size", "screen_size"} or fl.endswith("_size"):
        for key in brief:
            if "size" in key:
                return key
        return next((key for key in brief if "size" in key), "screen_size")
    if fl in {"display", "resolution"} or "pixel" in fl or "resol" in fl:
        for key in brief:
            if "resol" in key or "pixel" in key:
                return key
        return next((key for key in brief if "resol" in key), "resolution")
    return fl


def _row_matches_shopper(key: str, value: str, query: str) -> bool:
    shopper = _shopper_text(query)
    if not value:
        return False
    if any(_span_in(f"{key} {value}", span) for span in _figure_spans(shopper)):
        return True
    compact = _compact(value)
    folded = _fold(value)
    for token in _shopper_class_tokens(query):
        if token in folded or token in compact:
            return True
    if _PAIR_RX.search(str(value)) and _shopper_class_tokens(query):
        return True
    return False


def _firm_from_episode(query: str, product: dict[str, Any], rows: dict[str, str]) -> str:
    """Name the measured field from this brief + matching catalog facts."""
    shopper = _shopper_text(query)
    brief = _spec_keys_from_brief(query)
    for key in brief:
        if _key_in_text(key, shopper):
            return key
    candidates: list[tuple[int, str]] = []
    for key, value in rows.items():
        if not key or str(key).startswith("_") or key in _CLAIM_SOFT:
            continue
        if not _row_matches_shopper(key, value, query):
            continue
        aligned = _align_fact_key(key, brief, shopper, query)
        if not aligned or aligned in _CLAIM_SOFT:
            continue
        score = 3 if key not in _CLAIM_BLOCK else 2
        if aligned in brief:
            score += 2
        candidates.append((score, aligned))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    spans = _figure_spans(shopper)
    if any(str(span.get("unit") or "") == "in" for span in spans):
        return _align_fact_key("size", brief, shopper, query)
    if _shopper_class_tokens(query):
        return _align_fact_key("display", brief, shopper, query)
    return brief[0] if brief else ""


def _echo_shopper_token(product: dict[str, Any], query: str) -> str:
    grounded = _facts_blob(product)
    listing = grounded if (product.get("_hydrated") and grounded.strip()) else (
        grounded + " " + _blob(product)
    )
    compact = _compact(listing)
    for span in _figure_spans(_shopper_text(query)):
        token = _span_token(span)
        if token and (
            _compact(token) in compact
            or _span_in(listing, span)
            or _span_in_facts(product, span)
        ):
            return token
    tokens = sorted(_shopper_class_tokens(query), key=len, reverse=True)
    if not tokens or not _listing_meets_class(product, query):
        return ""
    floors = _rule_floors(query)
    named = [token for token in tokens if token in floors or not re.match(r"^\d+p$", token)]
    if not named:
        named = list(tokens)
    for token in named + [token for token in tokens if token not in named]:
        if token == "fullhd":
            return "fhd"
        return token
    return ""


def _claim_value(product: dict[str, Any], field: str, query: str, rows: dict[str, str]) -> str:
    if field == "price":
        return rows.get("price") or ""
    if field == "in_stock":
        return rows.get("in_stock") or ""
    if field == "brand":
        return rows.get("brand") or ""
    echo = _echo_shopper_token(product, query)
    if echo:
        return echo
    return rows.get(field) or ""


def _ground_claims(product: dict[str, Any], query: str, tool: dict[str, Any] | None) -> list[dict[str, str]]:
    """Copy observed listing text onto this episode's schema keys."""
    rows = _fact_rows(product)
    enum = _schema_claim_fields(tool)
    extras = _schema_extra_reasons(tool)
    firm = _firm_from_episode(query, product, rows)
    allowed = list(enum) if enum else ["in_stock", "price"]
    if firm and firm not in allowed:
        allowed.append(firm)
    for extra in extras:
        if extra not in allowed:
            allowed.append(extra)
    if "brand" not in allowed:
        allowed.append("brand")
    allowed_set = set(allowed)
    listing = _listing_claim_text(product, grounded=False) or _blob(product)
    fields = ["in_stock", "price"]
    if firm:
        fields.append(firm)
    brand_val = rows.get("brand") or ""
    if brand_val and "brand" in allowed_set:
        fields.append("brand")
    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    for field in fields:
        if field in seen or field in _CLAIM_BLOCK:
            continue
        if enum and field not in allowed_set:
            continue
        value = _claim_value(product, field, query, rows)
        if not value:
            continue
        if field not in _CLAIM_SOFT and not (
            _text_supports(value, listing)
            or _text_supports(re.sub(r"[a-z]+$", "", value), listing)
            or _echo_shopper_token(product, query) == value
            or _row_matches_shopper(field, value, query)
        ):
            continue
        seen.add(field)
        claims.append({"field": field, "value": str(value)})
    return claims[:4]


def _expand_variants(node: dict[str, Any]) -> list[dict[str, Any]]:
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
    if product_id not in (None, "") and any(
        key in node for key in ("title", "price", "sku", "brand", "facts", "in_stock")
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
        rows = _expand_variants(node)
        if rows:
            for row in rows:
                key = _listing_key(row) or str(row.get("product_id") or "")
                if key and key not in seen:
                    seen.add(key)
                    found.append(row)
            return
        for value in node.values():
            if isinstance(value, (dict, list)):
                visit(value)

    visit(payload)
    return found


def _unwrap(observation: dict[str, Any]) -> dict[str, Any]:
    nested = observation.get("observation")
    return nested if isinstance(nested, dict) else observation


def _absorb(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = dict(dst)
    for key, value in src.items():
        if value is None:
            continue
        if key in {"facts", "specification", "attributes"}:
            if not isinstance(value, dict) or not value:
                continue
            base = dict(out.get(key) or {}) if isinstance(out.get(key), dict) else {}
            base.update({item: raw for item, raw in value.items() if raw not in (None, "")})
            if base:
                out[key] = base
            continue
        if key == "price":
            incoming = _price_of({"price": value})
            current = _price_of(out)
            if incoming is None and current is not None:
                continue
            out[key] = incoming if incoming is not None else value
            continue
        if key.startswith("_") and out.get(key) and not value:
            continue
        out[key] = value
    return out


def _useful_query(query: str, kind: str = "", budget: float | None = None) -> bool:
    folded = _fold(query)
    tokens = _tokens(query)
    if not folded:
        return False
    if folded.isdigit() and len(folded) < 3:
        return False
    if budget is not None:
        try:
            if abs(float(folded.replace(" ", "")) - float(budget)) < 1e-6:
                return False
        except ValueError:
            pass
    if kind and _fold(kind) in folded:
        return True
    return bool(tokens)


def _criterion_matrix(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[list[float]]:
    used = axes or [("price", "asc")]
    matrix: list[list[float]] = []
    for product in pool:
        row: list[float] = []
        for name, direction in used:
            value = _axis_value(product, name)
            if value is None:
                row.append(0.0)
            elif direction == "asc":
                row.append(-value)
            else:
                row.append(value)
        matrix.append(row)
    return matrix


def _unit_norm(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix or not matrix[0]:
        return matrix
    cols = len(matrix[0])
    norms = []
    for col in range(cols):
        acc = math.sqrt(sum(row[col] * row[col] for row in matrix)) or 1.0
        norms.append(acc)
    return [[row[col] / norms[col] for col in range(cols)] for row in matrix]


def _entropy_weights(matrix: list[list[float]]) -> list[float]:
    if not matrix or not matrix[0]:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    shifted = []
    for col in range(cols):
        lo = min(row[col] for row in matrix)
        shifted.append([row[col] - lo + 1e-9 for row in matrix])
    weights: list[float] = []
    k = 1.0 / math.log(max(rows, 2))
    for col in range(cols):
        total = sum(shifted[col]) or 1.0
        ent = 0.0
        for value in shifted[col]:
            p = value / total
            ent -= p * math.log(p)
        weights.append(1.0 - k * ent)
    s = sum(weights) or 1.0
    return [item / s for item in weights]


def _retrieve_one(query_tf, doc_tf, df, n_docs, avgdl, coll_len, tokens) -> float:
    """Sequential Dependence Model (Metzler & Croft): unigrams + ordered bigrams."""
    unigram = 0.0
    doc_len = max(sum(doc_tf.values()), 1)
    for term, qtf in query_tf.items():
        tf = float(doc_tf.get(term, 0))
        if tf <= 0:
            continue
        p = (tf + 0.5) / (doc_len + 0.5 * max(n_docs, 1))
        unigram += float(qtf) * math.log(p + 1e-12)
    q_terms = list(query_tf.keys())
    d_terms = list(tokens)
    bigram = 0.0
    for i in range(len(q_terms) - 1):
        a, b = q_terms[i], q_terms[i + 1]
        hits = 0
        for j in range(len(d_terms) - 1):
            if d_terms[j] == a and d_terms[j + 1] == b:
                hits += 1
        if hits:
            bigram += math.log(1.0 + hits)
    return 0.7 * unigram + 0.3 * bigram


def _mabac_order(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """MABAC: distance from the border approximation area."""
    if len(pool) <= 1:
        return list(pool)
    used = axes or [("price", "asc")]
    matrix = _unit_norm(_criterion_matrix(pool, used))
    weights = _entropy_weights(matrix) or [1.0 / max(len(used), 1)] * len(used)
    n = len(pool)
    cols = len(used)
    v = [[weights[col] * (matrix[i][col] + 1.0) for col in range(cols)] for i in range(n)]
    border = []
    for col in range(cols):
        prod = 1.0
        for i in range(n):
            prod *= max(v[i][col], 1e-12)
        border.append(prod ** (1.0 / n))
    scores = [sum(v[i][col] - border[col] for col in range(cols)) for i in range(n)]
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    return [pool[i] for i in order]


def _saw_order(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if len(pool) <= 1:
        return list(pool)
    used = axes or [("price", "asc")]
    matrix = _unit_norm(_criterion_matrix(pool, used))
    weights = _entropy_weights(matrix) or [1.0 / max(len(used), 1)] * len(used)
    scores = [sum(weights[col] * row[col] for col in range(len(weights))) for row in matrix]
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
    return [pool[i] for i in order]


def _wpm_order(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if len(pool) <= 1:
        return list(pool)
    used = axes or [("price", "asc")]
    matrix = _unit_norm(_criterion_matrix(pool, used))
    weights = _entropy_weights(matrix) or [1.0 / max(len(used), 1)] * len(used)
    scores = []
    for row in matrix:
        prod = 1.0
        for col in range(len(weights)):
            prod *= max(row[col], 1e-9) ** weights[col]
        scores.append(prod)
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
    return [pool[i] for i in order]


def _choose_closed(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Lex of stated axes first; Borda of MABAC+SAW+WPM only breaks remaining ties."""
    if not pool:
        return []
    used = axes or [("price", "asc")]
    lexed = _lex_rank(pool, used)
    if len(lexed) <= 1:
        return lexed
    top = _lex_tuple(lexed[0], used)
    tied = [item for item in lexed if _lex_tuple(item, used) == top]
    if len(tied) <= 1:
        return lexed
    ballots = [_mabac_order(tied, used), _saw_order(tied, used), _wpm_order(tied, used)]
    borda: dict[str, float] = {}
    n = len(tied)
    for ballot in ballots:
        for rank, item in enumerate(ballot):
            key = _listing_key(item)
            borda[key] = borda.get(key, 0.0) + (n - rank)
    head = sorted(tied, key=lambda item: -borda.get(_listing_key(item), 0.0))
    rest = [item for item in lexed if _lex_tuple(item, used) != top]
    return head + rest


def _choose_open(pool: list[dict[str, Any]], axes: list[tuple[str, str]], fused: dict[str, float]) -> list[str]:
    ranked = sorted(pool, key=lambda item: -fused.get(_listing_key(item), 0.0))
    return [_listing_key(item) for item in ranked]


def compile_brief(query: str) -> dict[str, Any]:
    shopper = _shopper_text(query)
    rules = _rules_text(query)
    named = _named_keys(query)
    axes = _axes_from(shopper) or _axes_from(rules)
    budget_hit = _BUDGET_RX.search(shopper) or _BUDGET_RX.search(query or "")
    budget = float(budget_hit.group(1).replace(",", "")) if budget_hit else None
    quotes = _quoted(query)
    why = _wants_reason(query)
    held = _held_back(query)
    patch = bool(_PATCH_RX.search(shopper) or _PATCH_RX.search(rules))
    model = _model_guess(query)
    color = _color_guess(query)
    brand = _brand_guess(query)
    kind = _kind_noun(query)
    category = _category_hint(query)
    spans = _figure_spans(shopper, budget)
    fork = _soft_brand(query)
    taste = bool(
        re.search(
            r"rather pay|cheapest isn't|don't just grab the cheapest|matters more|"
            r"use-case|use case|intended use",
            shopper,
            re.I,
        )
    )
    scores = {
        DUTY_SLATE: 0,
        DUTY_WHY: 0,
        DUTY_MEND: 0,
        DUTY_ASK: 0,
        DUTY_FORK: 0,
        DUTY_TASTE: 0,
        DUTY_HUNT: 1,
    }
    if len(named) >= 2 or (named and axes) or re.search(
        r"stated tradeoff|strictly worse|compare only these candidate|lexico",
        rules,
        re.I,
    ):
        scores[DUTY_SLATE] += 8
    if why:
        scores[DUTY_WHY] += 6
    if patch and (model or color or "least-relax" in _fold(rules)):
        scores[DUTY_MEND] += 7
    if held:
        scores[DUTY_ASK] += 6
    if fork and not patch:
        scores[DUTY_FORK] += 5
    if taste:
        scores[DUTY_TASTE] += 5
    if quotes:
        scores[DUTY_HUNT] += 3
    primary = max(scores, key=lambda name: (scores[name], -list(scores).index(name)))
    duties = {primary}
    if why:
        duties.add(DUTY_WHY)
    if patch:
        duties.add(DUTY_MEND)
    if held:
        duties.add(DUTY_ASK)
    if fork and not patch:
        duties.add(DUTY_FORK)
    if taste:
        duties.add(DUTY_TASTE)
    wishes = _wish_terms(query) if DUTY_TASTE in duties else []
    phrases: list[str] = []
    phrases.extend(quotes)
    if model:
        phrases.append(" ".join(part for part in (color, model) if part))
        phrases.append(model)
    if brand and kind:
        for span in spans[:3]:
            phrases.append(f"{brand} {span['raw']} {kind}")
        for cue in _shopper_class_tokens(query)[:3]:
            phrases.append(f"{brand} {cue} {kind}")
        phrases.append(f"{brand} {kind}")
    if kind:
        for span in spans[:3]:
            phrases.append(f"{span['raw']} {kind}")
        for cue in _shopper_class_tokens(query)[:3]:
            phrases.append(f"{cue} {kind}")
        phrases.append(kind)
    if brand and not kind:
        phrases.append(brand)
    if category:
        phrases.append(category)
    for wish in wishes[:3]:
        if kind:
            phrases.append(f"{wish} {kind}")
        else:
            phrases.append(wish)
    cleaned: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        token = _clean(phrase)
        folded = _fold(token)
        if not token or folded in seen or not _useful_query(token, kind=kind, budget=budget):
            continue
        seen.add(folded)
        cleaned.append(token)
    if not cleaned:
        cleaned.append(" ".join(_tokens(shopper)[:6]) or "product")
    return {
        "primary": primary,
        "duties": duties,
        "budget": budget,
        "named": named,
        "axes": axes,
        "quotes": quotes,
        "ask": held,
        "why": why,
        "brand": brand,
        "soft_brand": fork,
        "kind": kind,
        "category": category,
        "model": model,
        "color": color,
        "spans": spans,
        "phrases": cleaned[:8],
        "wishes": wishes,
        "need_identities": 3 if primary == DUTY_HUNT else 2 if primary == DUTY_TASTE else 1,
    }


_CORE = """You are a shopping agent in a generated catalog. Use ONLY supplied tools.

# Chamber
Closed slates are a ballot: several independent scorers vote; Borda
aggregates. Open catalogs retrieve with sequential term dependence
(unigrams plus adjacent pairs from the shopper's own words). No device
ontology. First cart is graded.

# Beat
FILE    search with listing language (values, not schema keys).
READ    open concrete product_id+sku rows; inspect_stock when uncertain.
TALLY   cart the intended variant — the first cart line is scored.
RECOUNT after market shocks, climb the smallest concession; walls never move.
CERTIFY place_test_order after a fresh price/stock check. Close; do not loop.

# Walls vs wishes
Budget, stock, product class, and named hard specs are walls.
Brand leanings and comfort qualities are wishes. First cart takes the
wish only when that SKU also clears every wall; if the wish is
impossible, buy any fully compliant unit and move on.

# Search
Value tokens. Reformulate. Several identities. Every category.

# Figures
Numbers the shopper said vs option/facts. Episode-local class floors only.
Re-tally closed slates after shocks; do not keep a dominated cart.

# Dialogue
message on questions. Ask before search when a figure is withheld.
After about two replies, stop messaging and finish the purchase.

# Orders
If the shopper asked why: justification.claims MUST carry current price,
in_stock, and the spec they named (catalog key from viewed facts or
listing text), plus at most one extra reason (prefer brand). An order
without those claims fails even when the SKU is right. Copy observed
tokens; emit the shopper's class/size token when the listing meets it.
Otherwise product_id and sku only.

# Markets
Ban the event target. Recast. Inspect the new live nominee, then order.
"""

_DUTY_BOOK = {
    DUTY_HUNT: """# Hunt
Reformulate. Phrase in title. Several identities. Drop accessories.
""",
    DUTY_FORK: """# Fork
Brand is a wish, not a wall. First cart branded only if every wall is met.
If the maker wish is impossible, do not freeze — buy any fully compliant unit.
""",
    DUTY_ASK: """# Ask
A firm figure is withheld. Send message NOW before any search. Then match
the revealed value on option/facts (title only when facts are empty).
""",
    DUTY_TASTE: """# Taste
Use-case votes beat price. Skip cheap stubs. New listing after reprice.
""",
    DUTY_SLATE: """# Slate
Named set only. Ballot on stated axes. Re-tally after events. Usually no claims.
""",
    DUTY_MEND: """# Mend
Least-relax ladder. Same model, other color first. Walls: class, budget, stock.
Do not follow a dead SKU onto a different model. Inspect after the shock, then order.
""",
    DUTY_WHY: """# Why
place_test_order must attach justification.claims: price, in_stock, the
spec the shopper named, optional brand. Schema may have no field enum —
still include that spec. Copy observed listing text; empty claims fail.
""",
}


def _system_prompt(
    tools: list[dict[str, Any]],
    max_calls: int,
    brief: dict[str, Any],
    policy: dict[str, Any],
) -> str:
    names = [_tool_name(tool) for tool in tools if _tool_name(tool)]
    books = [_DUTY_BOOK[duty] for duty in brief.get("duties") or [] if duty in _DUTY_BOOK]
    named = brief.get("named") or []
    closed = ", ".join(f"{item['product_id']}::{item['sku']}" for item in named[:8])
    order = _find_tool(tools, "place_test_order", "place_order", "checkout")
    enum = _claim_enum(order)
    enum_line = ""
    if brief.get("why"):
        if enum:
            enum_line = f"claim_fields={enum}\n"
        else:
            enum_line = (
                "claim_fields=(schema has no field enum) "
                "use the spec the shopper named plus price and in_stock\n"
            )
    return (
        f"{_CORE}\n\n{''.join(books)}\n"
        f"# Runtime card\n"
        f"primary={brief.get('primary')} duties={sorted(brief.get('duties') or [])} "
        f"kind={brief.get('kind') or '-'} brand={brief.get('brand') or '-'} "
        f"budget={brief.get('budget')} ask={bool(brief.get('ask'))} why={bool(brief.get('why'))}\n"
        f"quotes={brief.get('quotes') or []}\n"
        f"closed_set={closed or '(open catalog)'}\n"
        f"{enum_line}"
        f"Available tools: {', '.join(names) or '(none)'}. "
        f"Emit at most {max_calls} tool call(s) per turn.\n"
        f"max_steps={policy.get('max_steps')} "
        f"max_calls_per_turn={policy.get('max_calls_per_turn', 1)}"
    )


def _is_inquiry(blob: str) -> bool:
    cleaned = (blob or "").strip().lower()
    if not cleaned:
        return False
    if cleaned.endswith("?"):
        return True
    return bool(re.search(r"\b(which|what|why|how|can you|could you|please (?:tell|confirm))\b", cleaned))


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
    if _left() < 12:
        return None
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    return _proxy().post("/inference/chat/completions", json_data=payload)


class _Store:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.waves: list[list[dict[str, Any]]] = []
        self.cart: list[dict[str, Any]] = []
        self.viewed: set[str] = set()
        self.gauged: set[str] = set()
        self.event_key: str | None = None
        self.event_open = False
        self.search_n = 0
        self.did_filter = False
        self.compare_n = 0
        self.dead: set[str] = set()
        self.dead_title = ""
        self.last_key: str | None = None
        self.need_replay = False
        self.signatures: Counter[str] = Counter()
        self.reply_n = 0

    def merge(
        self,
        products: list[dict[str, Any]],
        *,
        viewed: bool,
        gauged: bool,
        from_search: bool,
    ) -> None:
        if from_search and products:
            self.waves.append(products)
        for product in products:
            key = _listing_key(product)
            if not key:
                continue
            incoming = dict(product)
            current = self.rows.get(key)
            if current is None:
                self.rows[key] = incoming
                current = incoming
            else:
                current.update(_absorb(current, incoming))
            if viewed:
                current["_hydrated"] = True
                pid = str(current.get("product_id") or "")
                if pid:
                    self.viewed.add(pid)
                if self.event_open:
                    current["_aftershock"] = True
            if gauged:
                current["_gauged"] = True
                self.gauged.add(key)
            for item in self.cart:
                if _listing_key(item) == key:
                    item.update(current)

    def shock(self, event: dict[str, Any]) -> None:
        target = event.get("target") or {}
        product_id = str(target.get("product_id") or event.get("product_id") or "")
        sku = str(target.get("sku") or event.get("sku") or "")
        key = f"{product_id}::{sku}" if product_id and sku else product_id
        self.event_key = key or None
        self.event_open = True
        row = self.rows.get(key) if key else None
        if row is not None:
            self.dead_title = str(row.get("title") or self.dead_title)
            self.dead.add(key)
            if event.get("new_price") is not None:
                row["price"] = event.get("new_price")
            if "stockout" in _fold(event.get("kind") or "") or event.get("in_stock_after") is False:
                row["in_stock"] = False
            row["_gauged"] = False
            self.gauged.discard(key)
        self.search_n = 0
        self.did_filter = False
        self.compare_n = 0
        self.gauged.clear()
        self.need_replay = True
        for item in self.rows.values():
            item["_gauged"] = False
            item["_aftershock"] = False

    def pool(self) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for wave in self.waves:
            for product in wave:
                key = _listing_key(product)
                if key:
                    fused[key] = product
        fused.update(self.rows)
        return list(fused.values())


class _Pilot:
    def __init__(self, query: str, tools: list[dict[str, Any]], max_calls: int) -> None:
        self.query = query
        self.brief = compile_brief(query)
        self.max_calls = max(1, max_calls)
        self.store = _Store()
        self.search = _find_tool(tools, "search", "find_product", "search_catalog")
        self.filter = _find_tool(tools, "filter", "filter_catalog")
        self.view = _find_tool(tools, "view", "view_product", "get_product")
        self.compare = _find_tool(tools, "compare", "compare_products")
        self.inspect = _find_tool(tools, "inspect_stock", "check_stock")
        self.cart = _find_tool(tools, "add_to_cart", "cart_add")
        self.remove = _find_tool(tools, "remove_from_cart", "cart_remove")
        self.order = _find_tool(tools, "place_test_order", "place_order", "checkout")
        self.message = _find_tool(tools, "message", "talk", "respond")
        self.asked = False
        self.talks = 0
        self.turn = 0
        self.max_steps = 20
        self.echoed = False
        self.rewrite_sku = True
        self.stall_guard = True
        self.stall_breaks = 0
        for spec in self.brief.get("named") or []:
            key = f"{spec['product_id']}::{spec['sku']}"
            self.store.rows[key] = {
                "product_id": spec["product_id"],
                "sku": spec["sku"],
                "_hydrated": False,
            }

    def _full(self, product: dict[str, Any] | None) -> dict[str, Any]:
        if not product:
            return {}
        key = _listing_key(product)
        row = self.store.rows.get(key) or {}
        pid = str(product.get("product_id") or "")
        if pid and (not _facts(row) or not _facts(product)):
            for stored in self.store.rows.values():
                if str(stored.get("product_id") or "") == pid and _facts(stored):
                    row = _absorb(row, stored)
                    break
        return _absorb(row, product)

    def _stale(self, product: dict[str, Any] | None) -> bool:
        if product is None:
            return False
        if not _in_stock(product):
            return True
        budget = self.brief.get("budget")
        price = _price_of(product)
        if budget is not None and price is not None and price > float(budget):
            return True
        if self.store.event_key and _listing_key(product) == self.store.event_key:
            return True
        key = _listing_key(product)
        if key and key in self.store.dead:
            return True
        return False

    def _needs_recount(self, product: dict[str, Any] | None) -> bool:
        if not self.store.event_key:
            return False
        if product is None:
            return True
        return not bool(product.get("_aftershock"))

    def live(self, *, pick: bool = False) -> list[dict[str, Any]]:
        budget = self.brief.get("budget")
        kind = str(self.brief.get("kind") or "")
        source = self.store.pool()
        named = self.brief.get("named") or []
        if named:
            by_key = {_listing_key(item): item for item in source}
            source = []
            for spec in named:
                key = f"{spec['product_id']}::{spec['sku']}"
                parent = by_key.get(spec["product_id"]) or {}
                variant = by_key.get(key)
                if variant:
                    source.append(_absorb(parent, variant) if parent else variant)
                else:
                    source.append(
                        _absorb(parent, {"product_id": spec["product_id"], "sku": spec["sku"]})
                    )
        rows: list[dict[str, Any]] = []
        for product in source:
            key = _listing_key(product)
            if key in self.store.dead:
                continue
            if key and key == self.store.event_key:
                continue
            if not _in_stock(product):
                continue
            price = _price_of(product)
            if budget is not None and price is not None and price > float(budget):
                continue
            closed = bool(self.brief.get("named"))
            if not closed:
                if _junk(product, kind):
                    continue
                spec_kind = self.brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK} or (
                    DUTY_MEND in self.brief["duties"]
                )
                if kind:
                    if not product.get("title"):
                        if pick:
                            continue
                    elif not _kind_ok(product, kind, strict=spec_kind, query=self.query):
                        continue
                if _decoy(product, budget) and self.brief["primary"] != DUTY_HUNT:
                    continue
                if self.brief.get("quotes") and product.get("title"):
                    if not any(_phrase_hit(product, phrase) for phrase in self.brief["quotes"]):
                        if self.brief["primary"] == DUTY_HUNT and product.get("_hydrated"):
                            continue
            spec_duty = self.brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK}
            if (
                not closed
                and spec_duty
                and _decoy(product, budget)
            ):
                continue
            if spec_duty and (self.brief.get("spans") or DUTY_WHY in self.brief["duties"]):
                if pick and not product.get("_hydrated"):
                    if not self.store.event_key:
                        continue
                    if not _spec_ok(product, self.query, self.brief.get("spans") or []):
                        continue
                if product.get("_hydrated") and not _spec_ok(
                    product, self.query, self.brief.get("spans") or []
                ):
                    continue
            rows.append(product)
        return rows

    def retrieval_map(self, pool: list[dict[str, Any]]) -> dict[str, float]:
        query_tf = Counter(
            _tokens(" ".join(self.brief.get("phrases") or []) + " " + _shopper_text(self.query))
        )
        docs = [(_tokens(_blob(product)), product) for product in pool]
        df: Counter[str] = Counter()
        collection: Counter[str] = Counter()
        for tokens, _product in docs:
            df.update(set(tokens))
            collection.update(tokens)
        avgdl = sum(len(tokens) for tokens, _product in docs) / max(len(docs), 1)
        coll_len = sum(collection.values()) or 1
        scores: dict[str, float] = {}
        for tokens, product in docs:
            doc_tf = Counter(tokens)
            scores[_listing_key(product)] = _retrieve_one(
                query_tf, doc_tf, df, len(docs), avgdl, coll_len, tokens
            )
        return scores

    def ranked(self) -> list[dict[str, Any]]:
        pick = bool(
            (self.brief.get("spans") or DUTY_WHY in self.brief["duties"])
            and self.brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK}
        )
        pool = self.live(pick=pick)
        axes = list(self.brief.get("axes") or [])
        if DUTY_SLATE in self.brief["duties"] and (self.brief.get("named") or []):
            return _choose_closed(pool, axes)
        fused = self.retrieval_map(pool)
        by_key = {_listing_key(product): product for product in pool}
        wishes = list(self.brief.get("wishes") or [])
        budget = self.brief.get("budget")
        brand = str(self.brief.get("brand") or "")
        honor = DUTY_FORK in self.brief["duties"] and not self.store.event_key
        open_order = _choose_open(pool, axes, fused)

        def sort_key(product: dict[str, Any]) -> tuple:
            key = _listing_key(product)
            retrieval = fused.get(key, 0.0)
            taste = _taste_cover(product, wishes) if DUTY_TASTE in self.brief["duties"] else 0.0
            decoy = 1 if (
                DUTY_TASTE in self.brief["duties"]
                or self.brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK}
            ) and _decoy(product, budget) else 0
            brand_rank = 0 if honor and brand and _brand_hit(product, brand) else 1
            ready = 0 if product.get("_hydrated") else 1
            after = 0 if product.get("_aftershock") else 1
            price = _price_of(product)
            honor_after = bool(
                self.store.event_key
                and self.brief["duties"] & {DUTY_TASTE, DUTY_WHY, DUTY_SLATE}
            )
            try:
                open_pos = open_order.index(key)
            except ValueError:
                open_pos = 10_000
            return (
                decoy,
                brand_rank,
                after if honor_after else 0,
                0 if _spec_ok(product, self.query, self.brief.get("spans") or []) else 1,
                -taste,
                ready,
                open_pos,
                -retrieval,
                price if price is not None else 1e12,
            )

        return sorted(by_key.values(), key=sort_key)

    def champion(self) -> dict[str, Any] | None:
        ranked = self.ranked()
        target = self.store.event_key
        if not ranked and target:
            ranked = [
                item
                for item in self.store.pool()
                if _listing_key(item) != target
                and not self._stale(item)
                and _spec_ok(item, self.query, list(self.brief.get("spans") or []))
            ]
        if not ranked:
            return None
        live = [item for item in ranked if _listing_key(item) != target and not self._stale(item)]
        if not live:
            live = [item for item in ranked if _listing_key(item) != target]
        budget = self.brief.get("budget")
        if self.brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK}:
            nodecoy = [item for item in live if not _decoy(item, budget)]
            if nodecoy:
                live = nodecoy
            elif not target:
                live = []
        spans = list(self.brief.get("spans") or [])
        if self.brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK}:
            fitted = [item for item in live if _spec_ok(item, self.query, spans)]
            if fitted:
                live = fitted
            elif spans or DUTY_WHY in self.brief["duties"]:
                live = []
        if DUTY_TASTE in self.brief["duties"]:
            budget = self.brief.get("budget")
            wishes = list(self.brief.get("wishes") or [])
            if target:
                post = [item for item in live if item.get("_aftershock")]
                if post:
                    live = post
            fitted = [item for item in live if not _decoy(item, budget)]
            hydrated = [item for item in fitted if item.get("_hydrated")]
            strong = [item for item in hydrated if _taste_cover(item, wishes) >= 0.34]
            live = strong or hydrated or fitted or live
        if DUTY_FORK in self.brief["duties"]:
            brand = str(self.brief.get("brand") or "")
            fitted = [
                item
                for item in live
                if item.get("_hydrated") and _spec_ok(item, self.query, spans)
            ]
            off = [
                item
                for item in fitted
                if not brand or not _brand_hit(item, brand)
            ]
            if off:
                return off[0]
            if fitted and target:
                return fitted[0]
            if fitted and self.store.search_n >= 3:
                return fitted[0]
            return None
        if DUTY_MEND in self.brief["duties"] and (self.brief.get("model") or self.brief.get("color")):
            model = str(self.brief.get("model") or "")
            color = str(self.brief.get("color") or "")
            modeled = [item for item in live if model and _model_hit(item, model)]
            if color and not target:
                exact = [item for item in live if _ladder(item, model, color) == 0]
                if exact:
                    live = exact
                else:
                    return None
            elif target:
                stem = _sku_stem(target.split("::")[-1])
                siblings = [
                    item
                    for item in live
                    if stem
                    and _sku_stem(item.get("sku")) == stem
                    and (not model or _model_hit(item, model))
                ]
                modeled = [item for item in live if model and _model_hit(item, model)]
                if siblings:
                    live = siblings
                elif modeled:
                    live = modeled
                else:
                    return None
            elif modeled:
                live = modeled
            elif model:
                live = []
            live.sort(
                key=lambda item: (
                    _ladder(item, model, color),
                    0 if target and _sku_stem(item.get("sku")) == _sku_stem(target.split("::")[-1]) else 1,
                    _price_of(item) if _price_of(item) is not None else 1e12,
                )
            )
        if DUTY_WHY in self.brief["duties"] and self.brief.get("brand"):
            brand = str(self.brief.get("brand") or "")
            branded = [item for item in live if _brand_hit(item, brand)]
            if branded:
                live = branded
        return live[0] if live else None

    def _order(self, product: dict[str, Any]) -> dict[str, Any]:
        if self.order is None or not product:
            return {"name": "", "args": {}}
        product = self._full(product)
        wanted: dict[str, Any] = {
            "product_id": product.get("product_id"),
            "sku": product.get("sku"),
        }
        ranking = DUTY_SLATE in self.brief["duties"] and not self.brief.get("why")
        if ranking:
            return {"name": _tool_name(self.order), "args": _bind(self.order, wanted)}
        required = (_fn(self.order).get("parameters") or {}).get("required") or []
        needs = self.brief.get("why") or DUTY_WHY in self.brief["duties"]
        if needs or "justification" in {str(item).lower() for item in required}:
            claims = _ground_claims(product, self.query, self.order)
            if claims:
                wanted["justification"] = {"claims": claims}
        return {"name": _tool_name(self.order), "args": _bind(self.order, wanted)}

    def _search_actions(self) -> list[dict[str, Any]]:
        kind = str(self.brief.get("kind") or "")
        budget = self.brief.get("budget")
        phrases = [
            item
            for item in (self.brief.get("phrases") or ["product"])
            if _useful_query(item, kind=kind, budget=budget)
        ]
        if DUTY_MEND in self.brief["duties"] and self.brief.get("model"):
            model = str(self.brief.get("model") or "")
            color = str(self.brief.get("color") or "")
            headed = [item for item in (f"{color} {model}".strip(), model) if item]
            phrases = headed + [item for item in phrases if item not in headed]
        elif DUTY_MEND in self.brief["duties"] and self.store.dead_title and not self.echoed:
            title = " ".join(_tokens(self.store.dead_title)[:6])
            if title:
                phrases = [title] + [item for item in phrases if item != title]
            self.echoed = True
        elif self.brief.get("model") and self.store.event_key:
            model = str(self.brief.get("model") or "")
            phrases = [model] + [item for item in phrases if item != model]
        phrases = phrases or ([kind] if kind else ["product"])
        if DUTY_MEND in self.brief["duties"]:
            cap = min(3, max(len(phrases), 1))
        else:
            cap = 1 if self.store.event_key else min(2, max(len(phrases), 1))
        actions: list[dict[str, Any]] = []
        if self.search and self.store.search_n < cap:
            query = phrases[min(self.store.search_n, len(phrases) - 1)]
            self.store.search_n += 1
            actions.append(
                {
                    "name": _tool_name(self.search),
                    "args": _bind(self.search, {"query": query, "k": 10, "in_stock": True}),
                }
            )
        brand = str(self.brief.get("brand") or "")
        if DUTY_FORK in self.brief["duties"] and self.store.event_key:
            brand = ""
        if DUTY_HUNT in self.brief["duties"] and DUTY_FORK not in self.brief["duties"]:
            brand = ""
        category = str(self.brief.get("category") or "")
        if (
            self.filter
            and not self.store.did_filter
            and len(actions) < self.max_calls
            and (brand or category)
        ):
            args = _bind(
                self.filter,
                {
                    "brand": brand or None,
                    "category": category or None,
                    "max_price": self.brief.get("budget"),
                    "in_stock": True,
                    "k": 10,
                },
            )
            if args:
                self.store.did_filter = True
                actions.append({"name": _tool_name(self.filter), "args": args})
        if not actions:
            self.store.search_n += 1
        return actions[: self.max_calls]

    def _hydrate(self, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cap = 2 if self.store.event_key else 4
        if self.store.compare_n >= cap:
            return []
        ids: list[str] = []
        seen: set[str] = set()
        for product in products:
            pid = str(product.get("product_id") or "")
            if pid and pid not in seen and pid not in self.store.viewed:
                seen.add(pid)
                ids.append(pid)
        if not ids:
            return []
        self.store.compare_n += 1
        if self.compare:
            return [
                {
                    "name": _tool_name(self.compare),
                    "args": _bind(self.compare, {"product_ids": ids[:8], "product_id": ids[0]}),
                }
            ]
        if self.view:
            return [{"name": _tool_name(self.view), "args": _bind(self.view, {"product_id": ids[0]})}]
        return []

    def _view_cap(self) -> int:
        if DUTY_SLATE in self.brief["duties"]:
            return 12
        if DUTY_TASTE in self.brief["duties"]:
            return 6
        if DUTY_HUNT in self.brief["duties"]:
            return 4
        return 5

    def plan(self) -> list[dict[str, Any]]:
        if self.message and self.brief.get("ask") and not self.asked:
            return [
                {
                    "name": _tool_name(self.message),
                    "args": _bind(
                        self.message,
                        {"content": "What exact figure or attribute must the product meet?"},
                    ),
                }
            ]
        cart = self.store.cart[0] if self.store.cart else None
        if cart is not None and self._stale(cart) and self.remove:
            return [
                {
                    "name": _tool_name(self.remove),
                    "args": _bind(
                        self.remove,
                        {"product_id": cart.get("product_id"), "sku": cart.get("sku")},
                    ),
                }
            ]
        named = list(self.brief.get("named") or [])
        if named:
            missing = [
                spec["product_id"]
                for spec in named
                if spec["product_id"] not in self.store.viewed
            ]
            if missing:
                hydrate = self._hydrate([{"product_id": pid} for pid in missing])
                if hydrate:
                    return hydrate[: self.max_calls]
            if self.store.need_replay and self.compare:
                ids = [spec["product_id"] for spec in named]
                self.store.need_replay = False
                return [
                    {
                        "name": _tool_name(self.compare),
                        "args": _bind(self.compare, {"product_ids": ids[:8], "product_id": ids[0]}),
                    }
                ]
        else:
            live_now = self.live()
            recovering = bool(self.store.event_key)
            if recovering:
                model = str(self.brief.get("model") or "")
                siblings = [item for item in live_now if not model or _model_hit(item, model)]
                need_search = (not siblings) and self.store.search_n < 4
            else:
                need_search = self.store.search_n < (4 if DUTY_WHY in self.brief["duties"] else 5) and (
                    not live_now or self.store.search_n < 2
                )
            if DUTY_MEND in self.brief["duties"] and not recovering:
                need_search = self.store.search_n < 2 or need_search
                model = str(self.brief.get("model") or "")
                modeled = [item for item in live_now if not model or _model_hit(item, model)]
                ready = [item for item in modeled if item.get("_hydrated")]
                if len(ready) < 2:
                    hydrate = self._hydrate(modeled[: self._view_cap()])
                    if hydrate:
                        return hydrate[: self.max_calls]
            if recovering and DUTY_MEND in self.brief["duties"]:
                stem = _sku_stem((self.store.event_key or "").split("::")[-1])
                model = str(self.brief.get("model") or "")
                twins = [
                    item
                    for item in live_now
                    if stem
                    and _sku_stem(item.get("sku")) == stem
                    and (not model or _model_hit(item, model))
                ]
                if not twins:
                    twins = [item for item in live_now if model and _model_hit(item, model)]
                if not twins and self.store.search_n < 4:
                    need_search = True
                elif twins and not any(item.get("_hydrated") for item in twins):
                    hydrate = self._hydrate(twins[: self._view_cap()])
                    if hydrate:
                        return hydrate[: self.max_calls]
            if DUTY_WHY in self.brief["duties"] and not recovering:
                grounded = [
                    item
                    for item in live_now
                    if item.get("_hydrated")
                    and _spec_ok(item, self.query, list(self.brief.get("spans") or []))
                ]
                if not grounded and self.store.search_n < 4:
                    need_search = True
            if self.brief["primary"] in {DUTY_ASK, DUTY_FORK} and not recovering:
                spans = list(self.brief.get("spans") or [])
                fitted_n = sum(
                    1
                    for item in live_now
                    if item.get("_hydrated") and _spec_ok(item, self.query, spans)
                    and not _decoy(item, self.brief.get("budget"))
                )
                if self.store.search_n < 4 and fitted_n < 2:
                    need_search = True
            if DUTY_TASTE in self.brief["duties"] and not recovering:
                wishes = list(self.brief.get("wishes") or [])
                strong_n = sum(1 for item in live_now if _taste_cover(item, wishes) >= 0.5)
                if self.store.search_n < 6 and strong_n < 2:
                    need_search = True
            if need_search:
                search = self._search_actions()
                if search:
                    return search
            if recovering:
                model = str(self.brief.get("model") or "")
                focus = [
                    item
                    for item in live_now
                    if (not model or _model_hit(item, model)) and not item.get("_hydrated")
                ] or [item for item in live_now if not item.get("_hydrated")]
                if focus:
                    hydrate = self._hydrate(focus[: self._view_cap()])
                    if hydrate:
                        return hydrate[: self.max_calls]
            else:
                focus = (self.ranked() or live_now or self.store.pool())[: self._view_cap()]
                if len(self.store.viewed) < self._view_cap():
                    hydrate = self._hydrate(focus)
                    if hydrate:
                        return hydrate[: self.max_calls]
        champ = self.champion()
        if champ is None:
            if named:
                for spec in named:
                    key = f"{spec['product_id']}::{spec['sku']}"
                    if key == self.store.event_key or key in self.store.dead:
                        continue
                    if self.inspect and key not in self.store.gauged:
                        return [
                            {
                                "name": _tool_name(self.inspect),
                                "args": _bind(
                                    self.inspect,
                                    {"product_id": spec["product_id"], "sku": spec["sku"]},
                                ),
                            }
                        ]
                return []
            search = self._search_actions()
            if search:
                return search
            hydrate = self._hydrate(self.live() or self.store.pool())
            return hydrate[: self.max_calls]
        if self.inspect and _listing_key(champ) not in self.store.gauged:
            return [
                {
                    "name": _tool_name(self.inspect),
                    "args": _bind(
                        self.inspect,
                        {"product_id": champ.get("product_id"), "sku": champ.get("sku")},
                    ),
                }
            ]
        if self._needs_recount(champ) and self.inspect:
            return [
                {
                    "name": _tool_name(self.inspect),
                    "args": _bind(
                        self.inspect,
                        {"product_id": champ.get("product_id"), "sku": champ.get("sku")},
                    ),
                }
            ]
        if not self.store.cart and self.cart and not self._stale(champ):
            spec = self.brief["duties"] & {DUTY_WHY, DUTY_TASTE, DUTY_FORK, DUTY_ASK, DUTY_MEND}
            if spec and not champ.get("_hydrated"):
                return self._hydrate([champ])[: self.max_calls]
            return [
                {
                    "name": _tool_name(self.cart),
                    "args": _bind(
                        self.cart,
                        {"product_id": champ.get("product_id"), "sku": champ.get("sku")},
                    ),
                }
            ]
        if self.store.cart and not self._stale(self.store.cart[0]):
            cart_item = self.store.cart[0]
            cart_key = _listing_key(cart_item)
            if (
                DUTY_SLATE in self.brief["duties"]
                and champ is not None
                and _listing_key(champ) != cart_key
                and self.remove
            ):
                return [
                    {
                        "name": _tool_name(self.remove),
                        "args": _bind(
                            self.remove,
                            {"product_id": cart_item.get("product_id"), "sku": cart_item.get("sku")},
                        ),
                    }
                ]
            if self._needs_recount(champ if champ is not None else cart_item) and self.inspect:
                inspect_target = champ if champ is not None else cart_item
                return [
                    {
                        "name": _tool_name(self.inspect),
                        "args": _bind(
                            self.inspect,
                            {
                                "product_id": inspect_target.get("product_id"),
                                "sku": inspect_target.get("sku"),
                            },
                        ),
                    }
                ]
            if self.order:
                if cart_key in self.store.gauged:
                    self.store.event_open = False
                target = cart_item
                if champ is not None and champ.get("_gauged"):
                    target = champ
                return [self._order(target)]
        return []

    def finish(self) -> list[dict[str, Any]]:
        cart = self.store.cart[0] if self.store.cart else None
        if cart is not None and self._stale(cart) and self.remove:
            return [
                {
                    "name": _tool_name(self.remove),
                    "args": _bind(
                        self.remove,
                        {"product_id": cart.get("product_id"), "sku": cart.get("sku")},
                    ),
                }
            ]
        if cart is not None and not self._stale(cart) and self.order:
            if self._needs_recount(cart) and self.inspect:
                return [
                    {
                        "name": _tool_name(self.inspect),
                        "args": _bind(
                            self.inspect,
                            {"product_id": cart.get("product_id"), "sku": cart.get("sku")},
                        ),
                    }
                ]
            self.store.event_open = False
            return [self._order(cart)]
        picked = self.champion()
        if picked is None:
            live = [item for item in self.live() if _listing_key(item) != self.store.event_key]
            if self.brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK}:
                live = [
                    item
                    for item in live
                    if _spec_ok(item, self.query, list(self.brief.get("spans") or []))
                ]
            if DUTY_TASTE in self.brief["duties"]:
                wishes = list(self.brief.get("wishes") or [])
                fitted = [
                    item
                    for item in live
                    if _taste_cover(item, wishes) >= 0.2 and not _decoy(item, self.brief.get("budget"))
                ]
                live = fitted or live
            if DUTY_MEND in self.brief["duties"]:
                model = str(self.brief.get("model") or "")
                color = str(self.brief.get("color") or "")
                modeled = [
                    item
                    for item in live
                    if (not model or _model_hit(item, model)) and not self._stale(item)
                ]
                modeled.sort(key=lambda item: _ladder(item, model, color))
                live = modeled if modeled else (live if self.store.search_n >= 3 else [])
            picked = live[0] if live else None
        if picked is None:
            return []
        pid = str(picked.get("product_id") or "")
        if self.inspect and _listing_key(picked) not in self.store.gauged:
            return [
                {
                    "name": _tool_name(self.inspect),
                    "args": _bind(self.inspect, {"product_id": pid, "sku": picked.get("sku")}),
                }
            ]
        if self.cart and not self._stale(picked) and not self.store.cart:
            return [
                {
                    "name": _tool_name(self.cart),
                    "args": _bind(self.cart, {"product_id": pid, "sku": picked.get("sku")}),
                }
            ]
        if self.store.cart and self.order and not self._stale(self.store.cart[0]) and not self.store.event_open:
            return [self._order(self.store.cart[0])]
        return []

    def _next_beat(self) -> list[dict[str, Any]]:
        """Stall breaker: advance view → inspect → cart → order without IR."""
        champ = self.champion()
        if champ is None:
            live = [item for item in self.live() if not self._stale(item)]
            champ = live[0] if live else None
        if champ is None:
            return self._search_actions()
        key = _listing_key(champ)
        pid = champ.get("product_id")
        sku = champ.get("sku")
        if pid and pid not in self.store.viewed and self.view:
            return [{"name": _tool_name(self.view), "args": _bind(self.view, {"product_id": pid})}]
        if self.inspect and key not in self.store.gauged:
            return [
                {
                    "name": _tool_name(self.inspect),
                    "args": _bind(self.inspect, {"product_id": pid, "sku": sku}),
                }
            ]
        if self.cart and not self.store.cart and not self._stale(champ):
            return [
                {
                    "name": _tool_name(self.cart),
                    "args": _bind(self.cart, {"product_id": pid, "sku": sku}),
                }
            ]
        if self.store.cart and self.order and not self._stale(self.store.cart[0]):
            return [self._order(self.store.cart[0])]
        return []

    def _advance(self, product: dict[str, Any] | None) -> list[dict[str, Any]]:
        """View → inspect → cart → order the wall-clearing SKU. Never LLM SKUs."""
        if product is None or self._stale(product):
            search = self._search_actions()
            if search:
                return search
            hydrate = self._hydrate(self.live() or self.store.pool())
            if hydrate:
                return hydrate[: self.max_calls]
            return self._next_beat()
        kind = str(self.brief.get("kind") or "")
        if kind and product.get("title") and not _kind_ok(
            product, kind, strict=True, query=self.query
        ):
            return self._search_actions() or self._next_beat()
        if _decoy(product, self.brief.get("budget")) and self.brief["primary"] != DUTY_HUNT:
            return self._search_actions() or self._next_beat()
        pid = product.get("product_id")
        sku = product.get("sku")
        key = _listing_key(product)
        if self.store.event_key and not product.get("_aftershock"):
            if self.compare:
                return [
                    {
                        "name": _tool_name(self.compare),
                        "args": _bind(
                            self.compare,
                            {"product_ids": [pid], "product_id": pid},
                        ),
                    }
                ]
            if self.view:
                return [{"name": _tool_name(self.view), "args": _bind(self.view, {"product_id": pid})}]
        if not product.get("_hydrated"):
            hydrate = self._hydrate([product])
            if hydrate:
                return hydrate[: self.max_calls]
        if self.inspect and (key not in self.store.gauged or self._needs_recount(product)):
            return [
                {
                    "name": _tool_name(self.inspect),
                    "args": _bind(self.inspect, {"product_id": pid, "sku": sku}),
                }
            ]
        if self.cart and not self.store.cart and not self._stale(product):
            return [
                {
                    "name": _tool_name(self.cart),
                    "args": _bind(self.cart, {"product_id": pid, "sku": sku}),
                }
            ]
        cart_item = self.store.cart[0] if self.store.cart else None
        if cart_item is not None and not self._stale(cart_item) and self.order:
            target = product if not self._stale(product) else cart_item
            if self._needs_recount(target) and self.inspect:
                return [
                    {
                        "name": _tool_name(self.inspect),
                        "args": _bind(
                            self.inspect,
                            {"product_id": target.get("product_id"), "sku": target.get("sku")},
                        ),
                    }
                ]
            if not target.get("_hydrated"):
                hydrate = self._hydrate([target])
                if hydrate:
                    return hydrate[: self.max_calls]
            return [self._order(target)]
        return []

    def splice(
        self,
        actions: list[dict[str, Any]],
        proposed: list[dict[str, Any]],
        *,
        await_reply: bool,
    ) -> list[dict[str, Any]]:
        names = {
            _tool_name(tool).lower()
            for tool in (
                self.search, self.filter, self.view, self.compare, self.inspect,
                self.cart, self.remove, self.order, self.message,
            )
            if tool
        }
        cleaned = [action for action in actions if str(action.get("name") or "").lower() in names]
        cart_item = self.store.cart[0] if self.store.cart else None
        cart_stale = cart_item is not None and self._stale(cart_item)
        proposed_names = " ".join(str(item.get("name") or "").lower() for item in proposed)
        force = any(
            hint in proposed_names
            for hint in ("add_to_cart", "remove_from_cart", "inspect", "place_test_order", "place_order")
        )
        if cart_item is not None and self._stale(cart_item) and self.remove:
            return [
                {
                    "name": _tool_name(self.remove),
                    "args": _bind(
                        self.remove,
                        {"product_id": cart_item.get("product_id"), "sku": cart_item.get("sku")},
                    ),
                }
            ]
        if await_reply and self.message and self.store.reply_n < 2:
            if not any("message" in str(item.get("name") or "").lower() for item in cleaned):
                self.store.reply_n += 1
                return [
                    {
                        "name": _tool_name(self.message),
                        "args": _bind(
                            self.message,
                            {
                                "content": (
                                    "I checked the live listing against the firm gates "
                                    "and budget; I will proceed with a verified SKU."
                                )
                            },
                        ),
                    }
                ]
            self.store.reply_n += 1
        if self.stall_guard:
            sigs = []
            for action in cleaned:
                sig = f"{action.get('name')}|{json.dumps(action.get('args') or {}, sort_keys=True)}"
                sigs.append(sig)
                self.store.signatures[sig] += 1
            if cleaned and all(self.store.signatures[sig] >= 2 for sig in sigs):
                self.stall_breaks += 1
                beat = self._next_beat()
                if beat:
                    return beat[: self.max_calls]
        if not self.rewrite_sku:
            aligned_soft: list[dict[str, Any]] = []
            order_name = _tool_name(self.order).lower()
            for action in (cleaned or proposed)[: self.max_calls]:
                name = str(action.get("name") or "")
                args = dict(action.get("args") or {})
                if name.lower() == order_name:
                    if self._needs_recount(cart_item or self.champion()) and self.inspect:
                        champ = self.champion()
                        target = champ or cart_item
                        if target is not None:
                            aligned_soft.append(
                                {
                                    "name": _tool_name(self.inspect),
                                    "args": _bind(
                                        self.inspect,
                                        {"product_id": target.get("product_id"), "sku": target.get("sku")},
                                    ),
                                }
                            )
                            continue
                    product = cart_item or self.champion()
                    if product is None:
                        continue
                    aligned_soft.append(self._order(product))
                    continue
                aligned_soft.append({"name": name, "args": args})
            return aligned_soft or proposed[: self.max_calls]
        if DUTY_MEND in self.brief["duties"] and self.cart and not self.store.cart and self.store.event_key:
            champ = self.champion()
            model = str(self.brief.get("model") or "")
            color = str(self.brief.get("color") or "")
            ok = False
            if champ is not None and not self._stale(champ):
                if not champ.get("_hydrated"):
                    hydrate = self._hydrate([champ])
                    if hydrate:
                        return hydrate[: self.max_calls]
                if not self.store.event_key:
                    ok = _ladder(champ, model, color) == 0
                else:
                    ok = (not model) or _model_hit(champ, model)
            if ok:
                return self._advance(champ)
            return self._advance(None)
        if self.store.event_key and proposed and force:
            return self._advance(self.champion() or cart_item)
        if force and proposed and not await_reply:
            return self._advance(self.champion() or cart_item)
        if not cleaned:
            return self._advance(self.champion() or cart_item) or proposed[: self.max_calls]
        if any("message" in str(item.get("name") or "").lower() for item in cleaned):
            if self.store.event_open and self.store.cart:
                return self._advance(self.champion() or cart_item)
            if cart_item is not None and not cart_stale and self.order:
                champ = self.champion()
                target = champ or cart_item
                if self._needs_recount(target) and self.inspect and target is not None:
                    return [
                        {
                            "name": _tool_name(self.inspect),
                            "args": _bind(
                                self.inspect,
                                {"product_id": target.get("product_id"), "sku": target.get("sku")},
                            ),
                        }
                    ]
                if target is not None:
                    return [self._order(target)]
        order_name = _tool_name(self.order).lower()
        champ = self.champion()
        aligned: list[dict[str, Any]] = []
        for action in cleaned[: self.max_calls]:
            name = str(action.get("name") or "")
            args = dict(action.get("args") or {})
            lower = name.lower()
            if lower == order_name:
                if await_reply and self.message and (cart_item is None or cart_stale):
                    aligned.append(
                        {
                            "name": _tool_name(self.message),
                            "args": _bind(
                                self.message,
                                {
                                    "content": (
                                        "I checked the live listing against the firm gates "
                                        "and budget; I will proceed with a verified SKU."
                                    )
                                },
                            ),
                        }
                    )
                    continue
                if self._needs_recount(champ or cart_item) and self.inspect and (champ or cart_item) is not None:
                    inspect_target = champ if champ is not None else cart_item
                    aligned.append(
                        {
                            "name": _tool_name(self.inspect),
                            "args": _bind(
                                self.inspect,
                                {"product_id": inspect_target.get("product_id"), "sku": inspect_target.get("sku")},
                            ),
                        }
                    )
                    continue
                product = champ or cart_item
                if product is None:
                    continue
                aligned.append(self._order(product))
                continue
            if lower in {_tool_name(self.cart).lower()} and champ is not None and self.rewrite_sku:
                aligned.append(
                    {
                        "name": name,
                        "args": _bind(
                            self.cart,
                            {"product_id": champ.get("product_id"), "sku": champ.get("sku")},
                        ),
                    }
                )
                continue
            aligned.append({"name": name, "args": args})
        return aligned or self._advance(champ or cart_item)

    def ingest(self, result: dict[str, Any], sent: list[dict[str, Any]]) -> bool:
        done = False
        calls = result.get("calls") or []
        for index, call in enumerate(calls):
            observation = call.get("observation") or {}
            if isinstance(observation, dict) and observation.get("done") is True:
                done = True
            inner = _unwrap(observation) if isinstance(observation, dict) else {}
            err_blob = _fold(
                f"{observation.get('error') if isinstance(observation, dict) else ''} {inner.get('error') or ''}"
            )
            sent_args = (sent[index].get("args") or {}) if index < len(sent) else {}
            if "not_in_cart" in err_blob or "not in cart" in err_blob:
                self.store.cart = []
            if any(token in err_blob for token in ("no such", "no_candidate", "cannot add")):
                pid = str(sent_args.get("product_id") or inner.get("product_id") or "")
                sku = str(sent_args.get("sku") or inner.get("sku") or "")
                if pid and sku:
                    self.store.dead.add(f"{pid}::{sku}")
                    self.store.cart = []
            name = str(sent[index].get("name") or "").lower() if index < len(sent) else ""
            viewed = name in {_tool_name(self.view).lower(), _tool_name(self.compare).lower()}
            gauged = name == _tool_name(self.inspect).lower()
            products = _walk_products(inner)
            if gauged:
                patch = {
                    "product_id": inner.get("product_id") or sent_args.get("product_id"),
                    "sku": inner.get("sku") or sent_args.get("sku"),
                }
                if "in_stock" in inner:
                    patch["in_stock"] = inner.get("in_stock")
                if inner.get("price") is not None:
                    patch["price"] = inner.get("price")
                if inner.get("current_price") is not None:
                    patch["price"] = inner.get("current_price")
                if patch.get("product_id"):
                    products = products or [patch]
            if products:
                query = str((sent[index].get("args") or {}).get("query") or "") if index < len(sent) else ""
                if query and "search" in name:
                    products = [{**product, "_search_q": query} for product in products]
                self.store.merge(products, viewed=viewed, gauged=gauged, from_search=not viewed and not gauged)
            if isinstance(inner.get("added"), dict):
                added = dict(inner["added"])
                self.store.merge([added], viewed=False, gauged=False, from_search=False)
                key = _listing_key(added)
                merged = self.store.rows.get(key) or added
                self.store.cart = [dict(merged)]
                self.store.last_key = key or self.store.last_key
                if key and key in self.store.gauged:
                    self.store.event_open = False
            if isinstance(inner.get("cart"), list):
                self.store.cart = [item for item in inner["cart"] if isinstance(item, dict)]
            if isinstance(inner.get("removed"), dict) or inner.get("removed") is True:
                self.store.cart = []
                self.store.compare_n = 0
            event = inner.get("market_event")
            if isinstance(event, dict) and event:
                self.store.shock(event)
        message = result.get("user_message") or {}
        if isinstance(message, dict):
            signal = message.get("env_signal") or {}
            if isinstance(signal, dict) and signal.get("kind"):
                self.store.shock(signal)
            content = str(message.get("content") or "")
            if _EVENT_RX.search(content) and not self.store.event_open and not self.store.event_key:
                target: dict[str, Any] = {}
                if self.store.cart:
                    item = self.store.cart[0]
                    target = {"product_id": item.get("product_id"), "sku": item.get("sku")}
                elif self.store.last_key and "::" in self.store.last_key:
                    pid, sku = self.store.last_key.split("::", 1)
                    target = {"product_id": pid, "sku": sku}
                kind = "stockout" if re.search(r"sold out|out of stock", content, re.I) else "price_change"
                self.store.shock({"target": target, "kind": kind})
            budget = _BUDGET_RX.search(content)
            if budget:
                self.brief["budget"] = float(budget.group(1).replace(",", ""))
            if content and self.brief.get("ask"):
                self.asked = True
                extra = _figure_spans(content, self.brief.get("budget"))
                if extra:
                    self.brief["spans"] = extra
                    kind = str(self.brief.get("kind") or "")
                    headed = [
                        f"{self.brief.get('brand') or ''} {span['raw']} {kind}".strip()
                        for span in extra[:3]
                    ]
                    headed.extend(f"{span['raw']} {kind}".strip() for span in extra[:3])
                    phrases = list(self.brief.get("phrases") or [])
                    self.brief["phrases"] = [item for item in headed + phrases if item]
                    self.store.search_n = 0
                    self.store.did_filter = False
            if "message" in " ".join(str(item.get("name") or "") for item in sent).lower():
                self.asked = True
        return done


def _mechanical(actions: list[dict[str, Any]]) -> bool:
    if not actions:
        return False
    for action in actions:
        name = str(action.get("name") or "").lower()
        if not any(hint in name for hint in _MECH):
            return False
    return True


def _skip_llm(
    proposed: list[dict[str, Any]],
    await_reply: bool,
    turn: int,
    max_steps: int,
    cart_used: bool,
    pilot: _Pilot,
) -> bool:
    if _left() < 55:
        return True
    names = " ".join(str(item.get("name") or "").lower() for item in proposed)
    if proposed and "remove_from_cart" in names:
        return True
    if DUTY_ASK in pilot.brief["duties"] and not pilot.asked:
        return False
    if DUTY_WHY in pilot.brief["duties"] and not cart_used and turn <= 6:
        return False
    if DUTY_SLATE in pilot.brief["duties"] and not await_reply:
        return True
    if DUTY_MEND in pilot.brief["duties"]:
        model = str(pilot.brief.get("model") or "")
        color = str(pilot.brief.get("color") or "")
        champ = pilot.champion()
        exact = bool(champ and _ladder(champ, model, color) == 0)
        sibling = bool(champ and pilot.store.event_key and _ladder(champ, model, color) <= 1)
        if not cart_used and exact and proposed and not await_reply:
            return True
        if cart_used and sibling and any(
            hint in names for hint in ("add_to_cart", "inspect", "place_test_order", "place_order")
        ):
            return True
        if "compare" in names or "search" in names or "find_product" in names:
            return False
        if pilot.store.event_key and not sibling:
            return False
    champ = pilot.champion()
    cart = pilot.store.cart[0] if pilot.store.cart else None
    if pilot._needs_recount(champ or cart):
        if proposed and any(hint in names for hint in ("place_test_order", "place_order")):
            return True
        if not cart_used:
            return False
    if pilot.store.event_key and proposed:
        if any(
            hint in names
            for hint in ("add_to_cart", "remove_from_cart", "inspect", "place_test_order", "place_order")
        ):
            return True
    if pilot.store.event_open and proposed:
        if "compare" in names:
            return False
        return True
    if await_reply and not pilot.store.event_open:
        if cart_used:
            return True
        return _left() < 90
    if _mechanical(proposed):
        return True
    if cart_used and proposed:
        if "place_test_order" in names or "place_order" in names or "inspect" in names:
            return True
        if "add_to_cart" in names or "remove_from_cart" in names:
            return True
    taste = DUTY_TASTE in pilot.brief["duties"]
    if taste and not cart_used and not pilot.store.event_open and turn < 12 and _left() > 70:
        return False
    if turn >= max(4, int(max_steps * 0.7)):
        return True
    if proposed and turn >= 3 and _left() < 110:
        return True
    return False


def _trim_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(messages) <= 18:
        return messages
    head = messages[:2]
    tail = messages[-14:]
    if tail and tail[0].get("role") == "tool":
        for index, item in enumerate(messages[2:-14], start=2):
            if item.get("role") == "assistant" and item.get("tool_calls"):
                return head + messages[index:]
    return head + tail


def _coach(pilot: _Pilot, turn: int, await_reply: bool, cart_used: bool) -> str | None:
    if await_reply:
        if pilot.store.event_open:
            return (
                "A market event is open. Re-inspect remaining live options, "
                "cart the new winner, then order. Also answer the shopper."
            )
        return "The shopper asked a question. Answer with the message tool this turn."
    if pilot.talks >= 2:
        return (
            "You already answered. Stop calling message. View or inspect a "
            "real sellable unit, cart it, then place_test_order."
        )
    if pilot.brief.get("ask") and not pilot.asked and turn == 1:
        return "A firm figure is withheld until you ask. Send message now to collect it before any search."
    if DUTY_SLATE in pilot.brief["duties"] and turn <= 2:
        return "Compare every named candidate before the first cart. Order with ONLY product_id and sku."
    if DUTY_WHY in pilot.brief["duties"]:
        if cart_used:
            return (
                "Candidate is carted. inspect_stock if price/stock is stale, then "
                "place_test_order THIS turn with justification.claims: current price, "
                "in_stock, the spec the shopper named, optional brand. Do not send another message."
            )
        if turn <= 3:
            return (
                "View facts before the first cart. On place_test_order you MUST attach "
                "justification.claims: catalog keys from the goal and from viewed listing "
                "text — price, in_stock, the asked spec, and at most one evidenced extra "
                "(prefer brand). Schema may have no field enum; still include the asked spec. "
                "An order without claims fails even when the product is correct."
            )
    if DUTY_MEND in pilot.brief["duties"] and turn <= 2:
        return (
            "Something failed or moved. Recover with the smallest concession that still "
            "holds product class, spend ceiling and availability. Same model, other color first."
        )
    if DUTY_FORK in pilot.brief["duties"] and turn == 1:
        return (
            "First cart should take the stated leaning when a fully compliant sku offers it; "
            "otherwise take the best wall-clearing item. Answer questions with message."
        )
    if DUTY_TASTE in pilot.brief["duties"] and turn <= 2 and not cart_used:
        return "Compare the shopper's stated use-case before the first cart. Do not cart the cheapest stub."
    if DUTY_HUNT in pilot.brief["duties"] and turn == 3:
        return "One small keyword reformulation so the shortlist covers several distinct products."
    if turn >= max(3, int(pilot.max_steps * 0.7)):
        return "Only a few steps remain. Confirm the current pick's price and stock, then place_test_order."
    return None


def _model_actions(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_calls: int,
    model: str,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    last_content = ""
    raw_calls: list[dict[str, Any]] = []
    for candidate in _reserve_models(model):
        inference = _chat(
            model=candidate,
            messages=messages,
            tools=tools,
            tool_choice="required",
        )
        if inference is None:
            continue
        assistant = ((inference.get("choices") or [{}])[0].get("message")) or {}
        last_content = _assistant_text(assistant)
        raw_calls = list(assistant.get("tool_calls") or [])[:max_calls]
        break
    actions: list[dict[str, Any]] = []
    for call in raw_calls:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        try:
            args = _arguments(call)
        except ValueError:
            continue
        if name:
            actions.append({"name": name, "args": args})
    return actions, last_content, raw_calls


def _to_tool_calls(
    actions: list[dict[str, Any]],
    raw_calls: list[dict[str, Any]],
    problem_id: str,
    turn: int,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        call_id = f"{problem_id}-turn-{turn}-{index + 1}"
        if index < len(raw_calls) and raw_calls[index].get("id"):
            call_id = str(raw_calls[index]["id"])
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": action["name"],
                    "arguments": json.dumps(action.get("args") or {}),
                },
            }
        )
    return calls


def _run_pilot(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
    max_steps = int(policy.get("max_steps") or 20)
    tools = list(policy.get("tools") or [])
    query = str(policy.get("query") or problem_data.get("query") or "")
    pilot = _Pilot(query, tools, max_calls)
    pilot.max_steps = max_steps
    model = _primary_model(pilot.brief["duties"])

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(tools, max_calls, pilot.brief, policy)},
        {"role": "user", "content": query},
    ]
    if pilot.brief.get("ask"):
        messages.append(
            {
                "role": "user",
                "content": (
                    "The shopper is withholding one requirement until asked. "
                    "Send message now to get that figure before you search."
                ),
            }
        )
    if DUTY_TASTE in pilot.brief["duties"]:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Compare at least two candidates on the use-case qualities "
                    "before the first cart. Cheapness is a tiebreaker."
                ),
            }
        )
    if DUTY_SLATE in pilot.brief["duties"]:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Closed slate: compare listed candidates first. "
                    "place_test_order with ONLY product_id and sku unless a why was asked."
                ),
            }
        )
    if DUTY_WHY in pilot.brief["duties"]:
        messages.append(
            {
                "role": "user",
                "content": (
                    "On place_test_order you MUST attach justification.claims: catalog "
                    "keys from the goal and from viewed listing text — price, in_stock, "
                    "the asked spec, and at most one evidenced extra reason (prefer brand). "
                    "The schema may have no field enum; still include the spec the shopper "
                    "named. Copy observed tokens. If they named a class or size, emit that "
                    "token when the listing meets it. An order without claims fails even "
                    "when the product is correct."
                ),
            }
        )
    if DUTY_MEND in pilot.brief["duties"]:
        messages.append(
            {
                "role": "user",
                "content": (
                    "After a stock or price failure, recover with the smallest concession "
                    "that still holds product class, spend ceiling and availability. "
                    "Same model other color first. Inspect the new nominee, then order."
                ),
            }
        )
    if DUTY_FORK in pilot.brief["duties"]:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Treat brand as a wish. First cart takes it only when every wall is "
                    "already met; otherwise buy any fully compliant unit."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "chamber": {
                "primary": pilot.brief.get("primary"),
                "duties": sorted(pilot.brief.get("duties") or []),
                "model": model,
                "provider": _provider(),
                "stack": "sdm+mabac+saw+wpm",
            },
        }
    ]
    await_reply = False
    cart_used = False

    for turn in range(1, max_steps + 1):
        pilot.turn = turn
        if _left() < 8:
            proposed = pilot.finish()[:max_calls]
            skip = True
        else:
            proposed = pilot.plan()
            skip = _skip_llm(proposed, await_reply, turn, max_steps, cart_used, pilot)
        actions: list[dict[str, Any]] = []
        assistant_content = ""
        raw_calls: list[dict[str, Any]] = []
        if not skip:
            coach = _coach(pilot, turn, await_reply, cart_used)
            if coach:
                messages.append({"role": "user", "content": coach})
            messages = _trim_messages(messages)
            retries = 1 if proposed else len(_IDLE) + 1
            for retry in range(retries):
                actions, assistant_content, raw_calls = _model_actions(
                    messages, tools, max_calls, model
                )
                if actions:
                    break
                if proposed or _left() < 70:
                    break
                idle = _IDLE[min(retry, len(_IDLE) - 1)]
                dialogue.append({"role": "assistant", "content": assistant_content})
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {"role": "user", "content": idle},
                    ]
                )
        if not actions:
            actions = proposed[:max_calls]
        if actions:
            actions = pilot.splice(actions, proposed, await_reply=await_reply)
        if not actions:
            actions = pilot.finish()[:max_calls]
        if not actions:
            break

        lowered = [str(item.get("name") or "").lower() for item in actions]
        if any("add_to_cart" in name for name in lowered):
            cart_used = True
        if any("message" in name for name in lowered):
            await_reply = False
            pilot.asked = True
            pilot.talks += 1

        tool_calls = _to_tool_calls(actions, raw_calls, problem_id, turn)
        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        group_id = f"{problem_id}-turn-{turn}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {"name": item["name"], "args": item.get("args") or {}},
                }
                for index, item in enumerate(actions, start=1)
            ],
        }
        result = _proxy().post("/environment/call", json_data=envelope)
        if result is None:
            break
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
                "chamber_turn": {"primary": pilot.brief.get("primary"), "turn": turn},
            }
        )
        done = pilot.ingest(result, actions)
        for tool_call, call_row in zip(tool_calls, result.get("calls") or [], strict=False):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(call_row.get("observation"), ensure_ascii=False, default=str)[
                        :_OBS_CAP
                    ],
                }
            )
        shopper_msg = result.get("user_message")
        if shopper_msg and shopper_msg.get("content"):
            spoken = str(shopper_msg["content"])
            messages.append({"role": "user", "content": spoken})
            if _is_inquiry(spoken):
                await_reply = True
        if done:
            break
        if any(
            isinstance(call_row.get("observation"), dict) and call_row["observation"].get("done")
            for call_row in result.get("calls") or []
        ):
            break
    return dialogue


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    _arm_clock()
    dialogue: list[dict[str, Any]] = []
    try:
        dialogue = _run_pilot(problem_data)
    except Exception:
        return dialogue
    return dialogue
