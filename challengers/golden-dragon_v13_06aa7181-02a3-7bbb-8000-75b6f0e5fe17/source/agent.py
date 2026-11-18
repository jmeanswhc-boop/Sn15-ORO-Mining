"""Golden v14 — Concord.

Category-agnostic EnvPack shopper. Closed slates use ELECTRE III credibility
net-flow (outranking with veto), not TOPSIS. Open catalogs retrieve with
divergence-from-independence (DFI), not CombSUM/BM25. Duties stack from the
public brief. Python rebinds cart/order. OpenRouter DeepSeek, GLM on mend,
280s wall clock so the 300s sandbox cap is not eaten by hung inference.

Replaces Horizon v13 (Chutes + TOPSIS, no deadline skip).
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from os import getenv
from typing import Any


_SCOUT = "deepseek/deepseek-v3.2"
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

_CLOCK_S = 280.0
_OBS_CAP = 2400
_DECOY_FRAC = 0.08

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
_P_RX = re.compile(r"(\d{3,4})\s*p\b", re.I)
_EVENT_RX = re.compile(
    r"more expensive|still available|sold out|out of stock|"
    r"just get more|price (?:from|went|changed)|repriced",
    re.I,
)
_UNIT_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*"
    r"(inch(?:es)?|in\b|hz|hertz|gb|tb|mah|watt(?:s)?|w\b|mm|cm|"
    r"oz|ml|kg|g\b|dpi|rpm|p\b|\"|'')?",
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
    "p": "p",
}
_JUNK = (
    "case for",
    "cover for",
    "compatible with",
    "compatible for",
    "replacement only",
    "sticker",
    "decoration",
    "pouch",
)
# Free-text / meta drawers — never claim schema. No attribute ontology here.
_CLAIM_BLOCK = {
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
_CLAIM_SOFT = frozenset({"price", "in_stock", "brand", "condition", "currency"})
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

    def add(amount: float, unit: str, raw: str) -> None:
        if budget is not None and abs(amount - budget) < 1e-6:
            return
        unit = _canon_unit(unit)
        if not unit:
            return
        key = f"{amount:g}|{unit}"
        if key in seen:
            return
        seen.add(key)
        found.append({"amount": amount, "unit": unit, "raw": _clean(raw)})

    for match in _UNIT_RX.finditer(text or ""):
        try:
            amount = float(match.group(1))
        except ValueError:
            continue
        add(amount, match.group(2) or "", match.group(0))
    for match in _P_RX.finditer(text or ""):
        add(float(match.group(1)), "p", match.group(0))
    for width, height in _PAIR_RX.findall(text or ""):
        add(float(height), "p", f"{width}x{height}")
        add(float(width), "p", f"{width}x{height}")
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
    for blob in (_shopper_text(query), _rules_text(query), query or ""):
        hit = _CAT_RX.search(blob)
        if hit:
            return _clean(hit.group(1))
        hit = re.search(r"catalog category is\s+([^.\n]+)", blob, re.I)
        if hit:
            return _clean(hit.group(1))
    return ""


def _kind_noun(query: str) -> str:
    """Noun from this brief only — no catalog of device classes."""
    shopper = _shopper_text(query)
    folded = _fold(shopper)
    cat = _CAT_RX.search(shopper) or _CAT_RX.search(_rules_text(query))
    if not cat:
        aisle = re.search(r"catalog category is\s+([^.\n]+)", query or "", re.I)
        if aisle:
            parts = _tokens(aisle.group(1))
            if parts:
                noun = parts[-1]
                return noun[:-1] if noun.endswith("s") and len(noun) > 4 else noun
    if cat:
        parts = _tokens(cat.group(1))
        if parts:
            noun = parts[-1]
            return noun[:-1] if noun.endswith("s") and len(noun) > 4 else noun
    # Prefer the product noun after a proper-name brand: "Samsung smartphone".
    labeled = re.search(
        r"\b[a-z0-9][\w.&'-]*\s+(smartphone|tablet|monitor|headphone|phone|laptop)s?\b",
        folded,
    )
    if labeled:
        noun = labeled.group(1)
        return noun[:-1] if noun.endswith("s") and len(noun) > 4 else noun
    tail = re.search(
        r"([a-z][a-z]{2,})\s+(?:at or under|at or below|in stock)",
        folded,
    )
    if tail and tail.group(1) not in _STOP:
        return tail.group(1)
    head = re.search(
        r"(?:need|want|find|get|buy)\s+(?:me\s+)?(?:an?\s+)?(?:in[- ]stock\s+)?([a-z][a-z]{2,})",
        folded,
    )
    if head and head.group(1) not in _STOP:
        return head.group(1)
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
        r"(?:[Pp]refer(?:red)?|[Ii]deally(?:\s*,?\s*I\s+prefer)?)\s+"
        r"(?:the\s+)?([A-Z][A-Za-z0-9.&-]*(?:\s+[A-Z][A-Za-z0-9.&-]*){0,2})",
        shopper,
    )
    if hit:
        phrase = _clean(hit.group(1)).rstrip(".")
        if _fold(phrase) not in _STOP:
            return phrase
    hit = re.search(
        r"(?:Find me a[n]?\s+|(?:Can you )?get me (?:an?\s+)?(?:in[-\s]stock\s+)?|"
        r"I want a[n]?\s+|I need a[n]?\s+)"
        r"([A-Z][A-Za-z0-9.&'-]*(?:\s+[A-Z][A-Za-z0-9.&'-]*){0,2})",
        shopper,
    )
    if hit:
        phrase = _clean(hit.group(1)).rstrip(".")
        kind = _kind_noun(query)
        if (
            _fold(phrase) not in _STOP
            and _fold(phrase) not in _COLORS
            and _fold(phrase) != _fold(kind)
        ):
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
    tokens = _tokens(raw)
    return "_".join(tokens[:3]) if tokens else ""


def _axis_cues(name: str) -> tuple[str, ...]:
    """Tokens from this episode's axis phrase (no fixed attribute ontology)."""
    return tuple(token for token in _tokens(name.replace("_", " ")) if len(token) > 1)


def _axis_value(product: dict[str, Any], name: str) -> float | None:
    """Numeric axis from fact-key overlap, cue-adjacent numbers, or unit bridges.

    Unit bridges fire only when this episode's axis phrase carries the matching
    cue tokens (e.g. refresh/rate → Hz figures in the listing). No claim-field
    invention and no class maps.
    """
    folded = _fold(name.replace("_", " "))
    cues = _axis_cues(name)
    cue_set = set(cues)
    if (
        name == "price"
        or "price" in folded
        or "cost" in folded
        or cue_set & {"price", "cost", "budget"}
    ):
        return _price_of(product)
    facts = _facts(product)
    candidates: list[tuple[int, float]] = []

    for key, raw in facts.items():
        key_fold = _fold(str(key)).replace("_", " ")
        key_parts = set(key_fold.split())
        if cue_set & key_parts or any(len(cue) >= 3 and cue in key_fold for cue in cues):
            bits = re.findall(r"\d+(?:\.\d+)?", str(raw))
            for index, bit in enumerate(bits):
                try:
                    candidates.append((index, float(bit)))
                except ValueError:
                    continue

    blob = _blob(product)
    fold_blob = _fold(blob)
    cue_spans: list[int] = []
    for cue in cues:
        for match in re.finditer(rf"\b{re.escape(cue)}\b", fold_blob):
            cue_spans.append(match.start())
    if cue_spans:
        for match in re.finditer(r"(\d+(?:\.\d+)?)", fold_blob):
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            dist = min(abs(match.start() - span) for span in cue_spans)
            if dist <= 56:
                candidates.append((dist, value))

    # Cue-driven unit bridges — only when the shopper's axis phrase named them.
    if cue_set & {"refresh", "rate", "hz", "hertz", "freq", "frequency"}:
        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:hz|hertz)\b", fold_blob, re.I):
            try:
                candidates.append((0, float(match.group(1))))
            except ValueError:
                pass
    if cue_set & {"screen", "size", "display", "inch", "inches", "diagonal"}:
        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*[-–/to]+\s*(\d+(?:\.\d+)?)\s*-?\s*(?:inch(?:es)?|in\b|\")",
            fold_blob,
            re.I,
        ):
            try:
                candidates.append(
                    (2, max(float(match.group(1)), float(match.group(2))))
                )
            except ValueError:
                pass
        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*-?\s*(?:\"|''|in(?:ch(?:es)?)?)\b",
            fold_blob,
            re.I,
        ):
            try:
                candidates.append((5, float(match.group(1))))
            except ValueError:
                pass
    if cue_set & {"resol", "resolution", "pixel", "pixels"}:
        pair = _PAIR_RX.search(blob)
        if pair:
            try:
                candidates.append((0, float(int(pair.group(1)) * int(pair.group(2)))))
            except ValueError:
                pass
        hit = re.search(r"(\d{3,4})\s*p\b", fold_blob, re.I)
        if hit:
            try:
                candidates.append((1, float(hit.group(1))))
            except ValueError:
                pass

    if not candidates:
        raw = facts.get(name) or product.get(name)
        if raw is None:
            for key, value in facts.items():
                if _fold(str(key)).replace("_", " ") == folded:
                    raw = value
                    break
        return _number(raw)

    candidates.sort(key=lambda item: item[0])
    nearest = candidates[0][0]
    near_vals = [value for dist, value in candidates if dist == nearest]
    # Prefer the larger magnitude among equally near hits (overclock / dual figures).
    return max(near_vals) if near_vals else candidates[0][1]


def _lex_rank(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if not pool:
        return []
    used = axes or [("price", "asc")]
    # Prefer rows that actually expose the primary axis (avoid unmeasured stubs).
    if used:
        primary = used[0][0]
        measured = [item for item in pool if _axis_value(item, primary) is not None]
        if measured:
            pool = measured

    def key(product: dict[str, Any]) -> tuple:
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

    return sorted(pool, key=key)


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
        "waterproof", "water resistant", "sweat", "sweatproof", "workout", "splash",
        "quiet", "durable", "lightweight", "portable", "organic",
        "unscented", "cordless", "refillable", "bootcamp", "gym", "running",
        "outdoor", "weather", "earhook", "secure", "sport",
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


def _path_ok(product: dict[str, Any], kind: str) -> bool:
    if not kind:
        return True
    path = product.get("category_path") or product.get("category") or ""
    if isinstance(path, list):
        blob = _fold(" ".join(str(item) for item in path))
    else:
        blob = _fold(path)
    noun = _fold(kind)
    if blob and noun in {
        "smartphone", "phone", "tablet", "monitor", "laptop", "headphone",
    }:
        if re.search(
            r"\b(?:accessor|cases?|covers?|webcams?|charms?|cables?|chargers?|holders?)\b",
            blob,
        ):
            return False
    if not blob:
        return True
    return noun in blob or f"{noun}s" in blob


def _accessory_trap(product: dict[str, Any], kind: str) -> bool:
    """True when listing is a case/pouch/webcam/etc. for the shopper's device noun."""
    noun = _fold(kind)
    if noun not in {"smartphone", "phone", "tablet", "monitor", "laptop", "headphone"}:
        return False
    title = _fold(product.get("title") or "")
    path = product.get("category_path") or product.get("category") or ""
    path_blob = _fold(" ".join(str(item) for item in path) if isinstance(path, list) else str(path))
    if re.search(
        r"\b(?:case|cover|pouch|holster|skin|tempered|protector|webcam|camera cover|"
        r"charger|cable|mount|holder|charm|belt clip|bumper)\b",
        title,
    ):
        return True
    if re.search(r"\b(?:accessor|cases?|covers?|webcams?)\b", path_blob):
        return True
    # Accessory phrasing only — bare "for" matches too many real device titles.
    if re.search(
        r"\b(?:compatible with|compatible for|suitable for|fits)\b",
        title,
    ):
        return True
    if noun and re.search(rf"\b(?:for|fits)\s+{re.escape(noun)}\b", title):
        return True
    return False


def _kind_ok(product: dict[str, Any], kind: str) -> bool:
    if not kind:
        return True
    if _accessory_trap(product, kind):
        return False
    blob = _fold(_blob(product))
    noun = _fold(kind)
    if not noun:
        return True
    return (
        noun in blob
        or f"{noun}s" in blob
        or (noun.endswith("s") and len(noun) > 4 and noun[:-1] in blob)
    )


def _junk(product: dict[str, Any], kind: str) -> bool:
    if _accessory_trap(product, kind):
        return True
    noun = _fold(kind)
    # Shopping the accessories aisle — cases/covers are the product, not traps.
    if noun in {"accessories", "accessory", "parts", "cover", "case", "charm", "charms"}:
        return False
    title = _fold(product.get("title") or "")
    if any(marker in title for marker in _JUNK):
        if kind and noun not in title.split()[:3]:
            return True
    if kind:
        if re.search(rf"\b(?:for|fits|compatible with)\s+{re.escape(noun)}", title):
            return True
        # Off-aisle apparel / soft goods when the shopper named a device kind.
        if noun and not _kind_ok(product, kind):
            if re.search(
                r"\b(?:jacket|denim|shirt|dress|pants|trouser|shoe|sneaker|"
                r"hoodie|sweater|blouse|skirt|sock)\b",
                title,
            ):
                return True
    return False


def _brand_hit(product: dict[str, Any], brand: str) -> bool:
    if not brand:
        return False
    needle = _fold(brand)
    listed = _fold(product.get("brand") or "")
    # Hydrated rows with an explicit brand must match that field (Ringke ≠ Apple).
    if product.get("_hydrated") and product.get("brand"):
        if needle in listed or (len(needle.split()) >= 2 and all(p in listed for p in needle.split())):
            return True
        return False
    title = _fold(product.get("title") or "")
    # "Case for Apple iPhone" is not an Apple product.
    if re.search(
        rf"\b(?:for|fits|compatible with|compatible for|suitable for)\b[^\n]{{0,40}}\b{re.escape(needle)}\b",
        title,
    ):
        return False
    if re.search(
        rf"\b(?:case|cover|pouch|bumper|skin)\b[^\n]{{0,48}}\b{re.escape(needle)}\b",
        title,
    ):
        return False
    blob = _fold(_blob(product))
    if needle in blob:
        return True
    parts = needle.split()
    return len(parts) >= 2 and all(part in blob for part in parts)


def _model_hit(product: dict[str, Any], model: str) -> bool:
    if not model:
        return False
    needle = _compact(model)
    facts = _facts(product)
    fact_model = str(facts.get("model") or facts.get("model_name") or "")
    if product.get("_hydrated") and fact_model:
        return needle in _compact(fact_model) or needle in _compact(_options_text(product))
    return needle in _compact(_blob(product))


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


def _span_in(hay: str, span: dict[str, Any]) -> bool:
    amount = span.get("amount")
    unit = str(span.get("unit") or "")
    if amount is None:
        return False
    raw = str(hay or "").lower()
    blob = _fold(hay)
    compact = _compact(hay)
    token = f"{float(amount):g}"
    needles = [token]
    if abs(float(amount) - round(float(amount))) < 1e-9:
        needles.append(str(int(round(float(amount)))))
    hit = False
    for needle in needles:
        if not needle:
            continue
        # Raw keeps decimals (5.7); fold splits them into "5 7".
        if re.search(rf"(?<![\d.]){re.escape(needle)}(?![\d.])", raw):
            hit = True
            break
        spaced = _fold(needle)
        if spaced and re.search(rf"(?<!\d){re.escape(spaced)}(?!\d)", blob):
            hit = True
            break
        tight = _compact(needle)
        if not tight:
            continue
        if tight in compact and len(tight) >= 3:
            hit = True
            break
        # Short compact (5.7 → 57) only with an inch/unit cue in the same hay.
        if (
            tight in compact
            and len(tight) >= 2
            and unit == "in"
            and re.search(r"(?:inch(?:es)?|\bin\b|\"|'')", raw)
        ):
            hit = True
            break
    if not hit:
        return False
    if not unit:
        return True
    if unit == "p":
        return bool(
            re.search(
                rf"(?<![\d.]){re.escape(token)}\s*p\b|"
                rf"(?<![\d.]){re.escape(token)}\s*[x×]|"
                rf"[x×]\s*{re.escape(token)}(?![\d.])",
                raw,
                re.I,
            )
        )
    if unit == "in":
        return bool(re.search(r"(?:inch(?:es)?|\bin\b|\"|'')", raw))
    return bool(re.search(rf"\b{re.escape(unit)}\b", blob) or unit in compact)


def _cue_in(hay: str, cue: str) -> bool:
    if not cue or not hay:
        return False
    folded = _fold(hay)
    cue_f = _fold(cue)
    if not cue_f:
        return False
    # Word-boundary first so short cues never pay inside longer tokens.
    if re.search(rf"\b{re.escape(cue_f)}\b", folded):
        return True
    compact_cue = _compact(cue_f)
    # Compact fallback only for longer multi-token cues.
    if compact_cue and len(compact_cue) >= 4 and compact_cue in _compact(hay):
        return True
    return False


def _size_amount(product: dict[str, Any]) -> float | None:
    """Single non-range size figure from facts/options — driven by unit text, not key names."""
    facts = _facts(product)
    candidates: list[float] = []
    for _key, raw in facts.items():
        if raw in (None, ""):
            continue
        text = str(raw)
        if re.search(r"\d+\s*[-–/to]+\s*\d+", text, re.I):
            continue
        value = _number(raw)
        if value is None or not (1 <= value <= 100):
            continue
        if re.search(r"(?:inch(?:es)?|\bin\b|\"|'')", text, re.I) or re.search(
            r"\d+(?:\.\d+)?\s*(?:\"|''|in(?:ch(?:es)?)?)\b", text, re.I
        ):
            candidates.append(value)
    if candidates:
        return candidates[0]
    option = _options_text(product)
    if option and not re.search(r"\d+\s*[-–/to]+\s*\d+", option, re.I):
        hit = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(?:\"|''|in(?:ch(?:es)?)?)\b", option, re.I)
        if hit:
            value = float(hit.group(1))
            if 1 <= value <= 100:
                return value
    return None


def _adjacent_cues(shopper: str, span: dict[str, Any]) -> list[tuple[str, str]]:
    """Trailing labels beside a figure as (head, optional second word)."""
    raw = str(span.get("raw") or "")
    if not raw or not shopper:
        return []
    pairs: list[tuple[str, str]] = []

    def take(pattern: str) -> None:
        for match in re.finditer(pattern, shopper, re.I):
            head = _fold(match.group(1) or "")
            tail = _fold(match.group(2) or "") if match.lastindex and match.lastindex >= 2 else ""
            if head and head not in _STOP:
                pairs.append((head, tail if tail and tail not in _STOP else ""))

    take(rf"{re.escape(raw)}\s+([A-Za-z]{{2,12}})(?:\s+([A-Za-z]{{2,8}}))?")
    amount = span.get("amount")
    unit = str(span.get("unit") or "")
    if amount is not None and unit == "p":
        token = f"{float(amount):g}"
        take(rf"{re.escape(token)}\s*p\b\s+([A-Za-z]{{2,12}})(?:\s+([A-Za-z]{{2,8}}))?")
    # Single-token leading label only. Skip Title-case brand tokens ahead of the figure.
    for match in re.finditer(rf"\b([A-Za-z]{{2,12}})\s+{re.escape(raw)}\b", shopper, re.I):
        original = match.group(1)
        if original[:1].isupper() and original[1:].islower() and len(original) > 3:
            continue
        head = _fold(original)
        if head and head not in _STOP:
            pairs.append((head, ""))
    cleaned: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        cleaned.append(pair)
    return cleaned


def _cues_hit(shopper: str, span: dict[str, Any], grounded: str) -> bool:
    pairs = _adjacent_cues(shopper, span)
    if not pairs or not grounded:
        return False
    for head, tail in pairs:
        if tail:
            # Prefer the two-word phrase; still allow the leading head when attested.
            if _cue_in(grounded, f"{head} {tail}"):
                return True
            if _cue_in(grounded, head):
                return True
            continue
        if _cue_in(grounded, head):
            return True
    return False


def _measure_hit(
    product: dict[str, Any],
    spans: list[dict[str, Any]],
    query: str = "",
) -> bool:
    if not spans:
        return True
    option = _options_text(product)
    facts = _facts(product)
    fact_blob = " ".join(str(value) for value in facts.values())
    grounded = f"{option} {fact_blob}".strip()
    size_amt = _size_amount(product)
    shopper = _shopper_text(query) if query else ""
    inch_spans = [span for span in spans if span.get("unit") == "in"]
    # Hydrated size facts that disagree with the shopper inches always fail,
    # even when the marketing title repeats the asked figure.
    if product.get("_hydrated") and inch_spans and size_amt is not None:
        tol = 0.05
        if not any(abs(size_amt - float(span["amount"])) <= tol for span in inch_spans):
            return False
    if grounded or size_amt is not None:
        for span in spans:
            if _span_in(grounded, span):
                continue
            if span.get("unit") == "in" and size_amt is not None:
                tol = 0.05 if product.get("_hydrated") else 0.2
                if abs(size_amt - float(span["amount"])) <= tol:
                    continue
            if shopper and _cues_hit(shopper, span, grounded):
                continue
            if product.get("_hydrated"):
                return False
            if not _span_in(_blob(product), span):
                return False
        return True
    if product.get("_hydrated"):
        return False
    return all(_span_in(_blob(product), span) for span in spans)


def _figure_ok(product: dict[str, Any], query: str, spans: list[dict[str, Any]] | None = None) -> bool:
    needed = spans if spans is not None else _figure_spans(_shopper_text(query))
    if not needed:
        return True
    kind = _kind_noun(query)
    if kind and _junk(product, kind):
        return False
    return _measure_hit(product, needed, query=query)


def _seal_ok(
    product: dict[str, Any] | None,
    query: str,
    brief: dict[str, Any],
    *,
    require_spec_claim: bool = False,
    order_tool: dict[str, Any] | None = None,
) -> bool:
    """True when a SKU is safe to cart/order for ASK/constraint/why."""
    if not product:
        return False
    kind = str(brief.get("kind") or "")
    brand = str(brief.get("brand") or "")
    spans = list(brief.get("spans") or [])
    budget = brief.get("budget")
    if kind and _junk(product, kind):
        return False
    if kind and product.get("title") and not _kind_ok(product, kind):
        return False
    if product.get("_hydrated") and kind and not _path_ok(product, kind):
        return False
    if brief.get("firm_brand") and brand and not _brand_hit(product, brand):
        return False
    if DUTY_WHY in (brief.get("duties") or set()) and brand and not _brand_hit(product, brand):
        return False
    if _decoy(product, budget):
        return False
    if spans and not _figure_ok(product, query, spans):
        return False
    if spans and not product.get("_hydrated"):
        return False
    # Inch shoppers: when a hydrated size fact exists, it must match the ask.
    # Title-only inches remain allowed when the catalog has no size cell.
    inch_spans = [span for span in spans if span.get("unit") == "in"]
    if product.get("_hydrated") and inch_spans:
        size_amt = _size_amount(product)
        if size_amt is not None and not any(
            abs(size_amt - float(span["amount"])) <= 0.05 for span in inch_spans
        ):
            return False
    if require_spec_claim or DUTY_WHY in (brief.get("duties") or set()):
        claims = _ground_claims(product, query, order_tool)
        if spans and not _has_spec_claim(claims):
            if product.get("_hydrated") and _facts(product):
                return False
    return True


def _inch_claim(amount: float) -> str:
    if abs(amount - round(amount)) < 0.05:
        return f"{int(round(amount))}in"
    return f"{amount:g}in"


def _echo_class(shopper: str, listing: str) -> str:
    """Copy a shopper class phrase that also appears in listing text. No inventing."""
    if not listing:
        return ""
    for span in _figure_spans(shopper):
        if span.get("unit") != "p":
            continue
        for head, tail in _adjacent_cues(shopper, span):
            if tail:
                phrase = f"{head} {tail}"
                if _cue_in(listing, phrase):
                    return phrase
                if _cue_in(listing, head):
                    return head
                continue
            if _cue_in(listing, head):
                return head
    return ""


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
        if fl and text:
            rows[fl] = text
    options = product.get("options")
    if isinstance(options, dict):
        for key, raw in options.items():
            fl = _norm_field(str(key))
            text = _clean(raw)
            if fl and text and fl not in rows:
                rows[fl] = text
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
    folded = _fold(value)
    hay = _fold(listing)
    if folded.endswith("in") and _NUM_RX.search(folded):
        if _compact(folded[:-2]) in blob:
            return True
    if folded and re.search(rf"\b{re.escape(folded)}\b", hay):
        return True
    if tight and len(tight) >= 3 and re.search(rf"\b{re.escape(folded)}\b", hay):
        return True
    if tight and tight.isdigit() and tight in blob:
        return True
    return False


def _gate_field(query: str, allowed: set[str]) -> str:
    """Pick a claim field from this episode's enum via shopper↔field-name overlap."""
    shopper = _fold(_shopper_text(query))
    spans = _figure_spans(_shopper_text(query))
    core = {"in_stock", "price", "brand"}
    specs = [field for field in allowed if field not in core and field not in _CLAIM_BLOCK]
    if not specs:
        return ""
    for field in specs:
        parts = [part for part in field.replace("_", " ").split() if len(part) > 2]
        if any(part in shopper for part in parts):
            return field
    if spans:
        return specs[0]
    return ""


def _fact_spec_value(product: dict[str, Any], field: str, query: str) -> str:
    """Attest a claim value from the observed catalog cell for this enum field."""
    listing = _listing_claim_text(product, grounded=True)
    shopper = _shopper_text(query)
    facts = _facts(product)
    grounded = f"{_options_text(product)} {' '.join(str(v) for v in facts.values() if v not in (None, ''))}".strip()
    spans = _figure_spans(shopper)
    size_amt = _size_amount(product)
    raw = _clean(facts.get(field) or "")
    inch_spans = [span for span in spans if span.get("unit") == "in"]
    # Inch asks: emit Nin only when this cell (or measured size) attests the figure.
    if inch_spans and size_amt is not None:
        tol = 0.05 if product.get("_hydrated") else 0.2
        if any(abs(size_amt - float(span["amount"])) <= tol for span in inch_spans):
            if raw and (
                any(_span_in(raw, span) for span in inch_spans)
                or (_number(raw) is not None and abs(float(_number(raw)) - size_amt) <= tol)
            ):
                return _inch_claim(size_amt)
            if not raw and _number(facts.get(field)) is None:
                return ""
        if raw and any(_span_in(raw, span) for span in inch_spans):
            return _inch_claim(size_amt) if size_amt is not None else raw
        return ""
    if inch_spans:
        for span in inch_spans:
            if raw and _span_in(raw, span):
                return _inch_claim(float(span["amount"]))
            if _span_in(grounded, span) and raw and _number(raw) is not None:
                return _inch_claim(float(span["amount"]))
        return ""
    # Copy the catalog cell only when it supports the shopper figure / echoed label.
    p_spans = [span for span in spans if span.get("unit") == "p"]
    if raw and p_spans and any(_cues_hit(shopper, span, raw) for span in p_spans):
        return raw
    echoed = _echo_class(shopper, grounded or listing)
    if raw and echoed and (_cue_in(raw, echoed) or _fold(raw) == _fold(echoed)):
        return raw
    for span in spans:
        if raw and _span_in(raw, span):
            return raw
    if raw and _text_supports(raw, grounded or listing) and _value_attested(raw, shopper):
        return raw
    return ""


def _needs_figure(brief: dict[str, Any]) -> bool:
    """Shopper typed a measured figure on a figure-gated duty."""
    if not brief.get("spans"):
        return False
    return bool(
        brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK}
        or brief.get("firm_brand")
        or (brief.get("duties") or set()) & {DUTY_ASK, DUTY_WHY, DUTY_FORK}
    )


def _has_spec_claim(claims: list[dict[str, str]]) -> bool:
    return any(str(row.get("field") or "") not in _CLAIM_SOFT for row in claims)


def _catalog_spec_claims(
    product: dict[str, Any],
    query: str,
) -> list[tuple[str, str]]:
    """Attested catalog fact keys → claim pairs. Never invent schema field names."""
    shopper = _shopper_text(query)
    spans = _figure_spans(shopper)
    facts = _facts(product)
    option = _options_text(product)
    grounded = f"{option} {' '.join(str(v) for v in facts.values() if v not in (None, ''))}".strip()
    size_amt = _size_amount(product)
    echoed = _echo_class(shopper, grounded)
    inch_spans = [span for span in spans if span.get("unit") == "in"]
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, raw in facts.items():
        fl = _norm_field(str(key))
        if (
            not fl
            or fl.startswith("_")
            or fl in _CLAIM_SOFT
            or fl in _CLAIM_BLOCK
        ):
            continue
        text = _clean(raw)
        if not text:
            continue
        value = ""
        if inch_spans and size_amt is not None:
            tol = 0.05 if product.get("_hydrated") else 0.2
            carries = any(_span_in(text, span) for span in inch_spans) or (
                _number(text) is not None
                and any(abs(size_amt - float(span["amount"])) <= tol for span in inch_spans)
                and abs(float(_number(text)) - size_amt) <= tol
            )
            if carries:
                value = _inch_claim(size_amt)
        if not value and echoed and (_cue_in(text, echoed) or _fold(text) == _fold(echoed)):
            if not inch_spans:
                value = text
        if not value:
            for span in spans:
                if _span_in(text, span) or (shopper and _cues_hit(shopper, span, text)):
                    if inch_spans and span.get("unit") == "in":
                        value = _inch_claim(size_amt) if size_amt is not None else text
                    else:
                        value = text
                    break
        if not value and not spans and _value_attested(text, shopper):
            value = text
        if not value:
            continue
        if fl in seen:
            continue
        seen.add(fl)
        found.append((fl, str(value)))
    # Prefer cells whose values literally carry the shopper figure; else catalog order.
    if spans:
        found.sort(
            key=lambda pair: (
                0 if any(_span_in(str(_facts(product).get(pair[0]) or ""), span) for span in spans) else 1,
                0 if echoed and _cue_in(str(_facts(product).get(pair[0]) or ""), echoed) else 1,
                pair[0],
            )
        )
    # If inches matched via size_amt but no cell was emitted, promote any
    # numeric fact equal to that amount (still a catalog key — never invent names).
    if inch_spans and size_amt is not None:
        if not found:
            for key, raw in facts.items():
                fl = _norm_field(str(key))
                if not fl or fl in _CLAIM_SOFT or fl in _CLAIM_BLOCK or fl in seen:
                    continue
                number = _number(raw)
                if number is None:
                    continue
                if abs(float(number) - float(size_amt)) <= 0.05:
                    found.append((fl, _inch_claim(size_amt)))
                    break
    return found


def _ground_claims(product: dict[str, Any], query: str, tool: dict[str, Any] | None) -> list[dict[str, str]]:
    """Copy viewed facts onto this episode's order-tool fields."""
    allowed = _claim_enum(tool)
    shopper = _shopper_text(query)
    rows = _fact_rows(product)
    listing = _listing_claim_text(product, grounded=True) or _blob(product)
    core = {"in_stock", "price", "brand"}
    catalog_specs = _catalog_spec_claims(product, query)
    if not allowed:
        # No live enum: soft drawers + attested catalog keys (never invent schema names).
        allowed = ["in_stock", "price", "brand"]
        allowed.extend(field for field, _value in catalog_specs)
    allowed_set = set(allowed)
    fields = [item for item in ("in_stock", "price") if item in allowed_set]
    gate = _gate_field(query, allowed_set)
    specs = [field for field in allowed if field not in core and field not in _CLAIM_BLOCK]
    if not gate:
        for field in specs:
            if _fact_spec_value(product, field, query) or rows.get(field):
                gate = field
                break
    if not gate and catalog_specs:
        gate = catalog_specs[0][0]
    if gate and gate not in _CLAIM_BLOCK:
        fields.append(gate)
    if "brand" in allowed_set and (
        _key_in_text("brand", shopper)
        or re.search(r"\bprefer|\bideally\b", shopper, re.I)
        or _brand_guess(query)
    ):
        fields.append("brand")
    catalog_values = {field: value for field, value in catalog_specs}
    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    size_amt = _size_amount(product)
    for field in fields:
        if field in seen or field in _CLAIM_BLOCK or field not in allowed_set:
            continue
        if field in {"in_stock", "price", "brand"}:
            value = rows.get(field)
            if field == "brand" and not value:
                guess = _brand_guess(query)
                if guess and _brand_hit(product, guess):
                    value = guess
        else:
            value = (
                catalog_values.get(field)
                or _fact_spec_value(product, field, query)
                or rows.get(field)
            )
        if not value:
            continue
        if field not in {"in_stock", "price", "brand"}:
            listing_ok = _text_supports(value, listing) or any(
                _span_in(listing, span) for span in _figure_spans(shopper)
            )
            if size_amt is not None and _NUM_RX.search(str(value)):
                listing_ok = listing_ok or _text_supports(_inch_claim(size_amt), listing)
            if not listing_ok:
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
    """Divergence-from-independence (DFI) retrieval — Terrier family, not BM25."""
    score = 0.0
    doc_len = max(sum(doc_tf.values()), 1)
    for term, qtf in query_tf.items():
        tf = float(doc_tf.get(term, 0))
        if tf <= 0:
            continue
        expected = (float(df.get(term, 0) or 1) * doc_len) / max(coll_len, 1)
        expected = max(expected, 0.25)
        if tf > expected:
            score += float(qtf) * (tf - expected) / math.sqrt(expected)
    return score


def _electre_order(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """ELECTRE III credibility net-flow. Veto kills a pairwise outranking."""
    if len(pool) <= 1:
        return list(pool)
    used = axes or [("price", "asc")]
    matrix = _criterion_matrix(pool, used)
    weights = _entropy_weights(matrix) or [1.0 / max(len(used), 1)] * len(used)
    n = len(pool)
    cols = len(used)
    # indifference / preference / veto as fractions of column range
    ranges = []
    for col in range(cols):
        vals = [row[col] for row in matrix]
        ranges.append(max(max(vals) - min(vals), 1e-9))
    cred = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            conc = 0.0
            veto = False
            for col in range(cols):
                diff = matrix[i][col] - matrix[j][col]
                q = 0.05 * ranges[col]
                p = 0.20 * ranges[col]
                v = 0.85 * ranges[col]
                if diff <= -v:
                    veto = True
                    break
                if diff >= p:
                    cj = 1.0
                elif diff <= q:
                    cj = 0.0
                else:
                    cj = (diff - q) / max(p - q, 1e-9)
                conc += weights[col] * cj
            if veto:
                cred[i][j] = 0.0
            else:
                cred[i][j] = conc
    flow = []
    for i in range(n):
        plus = sum(cred[i][j] for j in range(n) if j != i)
        minus = sum(cred[j][i] for j in range(n) if j != i)
        flow.append(plus - minus)
    order = sorted(range(n), key=lambda i: (-flow[i], i))
    return [pool[i] for i in order]


def _choose_closed(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    ranked = _electre_order(pool, axes)
    if not ranked:
        return _lex_rank(pool, axes)
    return ranked


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
    firm_brand = bool(brand) and not fork
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
    # Firm brand + figure gate without soft-preference language: keep brand/filter
    # and treat like ASK for selection, without forcing a clarify message.
    if firm_brand and spans and not held and not why and not fork:
        scores[DUTY_ASK] += 4
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
    if firm_brand and spans and not held and not why and not fork:
        duties.add(DUTY_ASK)
    # Any measured shopper figure gates selection — even on plain HUNT.
    if spans:
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
        phrases.append(f"{brand} {kind}")
    if kind:
        for span in spans[:3]:
            phrases.append(f"{span['raw']} {kind}")
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
        "firm_brand": firm_brand,
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

# Concord
Firm gates (category, budget, in-stock, stated figures, named listing, quoted
phrase) must hold on the exact SKU. A leaning never overrides a firm miss.
First add_to_cart is graded — never park a placeholder. Outranking, not a
memorized electronics order, decides a closed slate.

# Loop
1. SURVEY — search/filter with short value queries (quoted phrase, brand,
   figure, product noun, category path). Never dump the whole instruction.
2. READ — view/compare until raw facts/options are visible. Titles lie.
3. BIND — inspect_stock, then add_to_cart the live winner.
4. MEND — after stock/price shocks, drop the stale SKU, recast, pick again.
5. SEAL — place_test_order once the live cart SKU is verified.

# Search
Send value tokens, never JSON field names. Reformulate from words that
appeared in titles. Survey several distinct identities before the first
cart on an open catalog. Groceries, tools, beauty, parts, apparel, and
electronics use the same loop: quote + category + figures the shopper said.

# Figures
Match numbers the shopper said against option rows and facts. If Task
rules define named classes for this episode, use those thresholds here
only. Do not invent conversions or category cheat sheets.

# Dialogue
If the shopper asks a question, answer with message before further cart/order.
If a firm figure is withheld until you ask, message first, then search.

# Orders
If the shopper asked why, attach justification.claims copied from viewed
facts using only fields this episode's order tool allows. Otherwise pass
ONLY product_id and sku. Never invent attributes. Never rewrite units.
Never send category/title/os/summary/evidence.

# Markets
After stockout or price_change, remove the stale cart SKU, recast, cart a
different live listing, then order. Never order the event target.
"""

_DUTY_BOOK = {
    DUTY_HUNT: """# Hunt
Reformulate at least once. A good shortlist covers several distinct products.
Quoted / "matches phrase" tokens must appear in the title. Drop accessories
via category_path and "for / compatible with" titles. Once a valid in-stock
item is verified, order — do not wait for permission.
""",
    DUTY_FORK: """# Fork
Preferred brand is SOFT. Build two piles: brand+firm vs firm-only. First
cart honors the named brand only when that SKU meets every firm gate.
After a shock, drop the brand filter if no branded SKU still clears the new
world.
""",
    DUTY_ASK: """# Ask
If a firm figure is withheld until you ask, send message NOW and do not
search until the shopper replies. Every gate must hold on the exact option
row. After sellout, pick a new product_id — a sibling that drops a gate is
illegal. Read facts, not titles.
""",
    DUTY_TASTE: """# Taste
Compare at least two hard-compliant listings on the use-case qualities the
shopper named. Cheapness is a tiebreaker. First cart must be a genuine fit,
not a 10–50 bait stub. After a reprice, recover to a different listing that
still fits the use case — search again, do not reuse leftovers.
""",
    DUTY_SLATE: """# Slate
The candidate set is closed: only the named product_id::sku pairs. Batch
view/compare them. Rank by pairwise outranking on the shopper's stated
axes, not a memorized device order. After any cart-side event, re-rank from
scratch and cart the new #1. place_test_order with ONLY product_id and sku
unless the shopper also asked why.
""",
    DUTY_MEND: """# Mend
Walls that never relax: category, budget, in-stock. Ladder: exact
model+color, then same model other color, then other models. Search the
dead listing's title to find twins. A conflicting title does not override
option/facts. Do not jump levels.
""",
    DUTY_WHY: """# Why
View facts before the first cart. Claims copy observed catalog text using
only field names this episode's place_test_order schema allows. Always
include in_stock, price, and the specification the shopper named (the
inch/p-mode/hertz they typed). Add brand when they named a maker. Format
inches as Nin when facts show that number.
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
    enum_line = f"claim_fields={enum}\n" if brief.get("why") else ""
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
                if self.event_open:
                    current["_aftershock"] = True
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
                if kind and product.get("title") and not _kind_ok(product, kind):
                    continue
                if product.get("_hydrated") and not _path_ok(product, kind):
                    continue
                if self.brief.get("quotes") and product.get("title"):
                    if not any(_phrase_hit(product, phrase) for phrase in self.brief["quotes"]):
                        if self.brief["primary"] == DUTY_HUNT and product.get("_hydrated"):
                            continue
            spec_duty = _needs_figure(self.brief) or self.brief["primary"] in {
                DUTY_ASK, DUTY_WHY, DUTY_FORK
            } or bool(self.brief.get("firm_brand"))
            if spec_duty and self.brief.get("spans"):
                if pick and not product.get("_hydrated"):
                    continue
                if product.get("_hydrated") and not _figure_ok(
                    product, self.query, self.brief.get("spans") or []
                ):
                    continue
            if pick and spec_duty and _decoy(product, budget):
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
        pick = bool(self.brief.get("spans"))
        pool = self.live(pick=pick)
        axes = list(self.brief.get("axes") or [])
        if DUTY_SLATE in self.brief["duties"] and (self.brief.get("named") or []):
            return _choose_closed(pool, axes)
        fused = self.retrieval_map(pool)
        by_key = {_listing_key(product): product for product in pool}
        wishes = list(self.brief.get("wishes") or [])
        budget = self.brief.get("budget")
        brand = str(self.brief.get("brand") or "")
        honor = bool(self.brief.get("firm_brand")) or (
            DUTY_FORK in self.brief["duties"] and not self.store.event_key
        )
        open_order = _choose_open(pool, axes, fused)
        bait_duties = {DUTY_TASTE, DUTY_FORK, DUTY_ASK, DUTY_WHY}

        def sort_key(product: dict[str, Any]) -> tuple:
            key = _listing_key(product)
            retrieval = fused.get(key, 0.0)
            taste = _taste_cover(product, wishes) if DUTY_TASTE in self.brief["duties"] else 0.0
            decoy = 1 if self.brief["duties"] & bait_duties and _decoy(product, budget) else 0
            brand_rank = 0 if honor and brand and _brand_hit(product, brand) else 1
            ready = 0 if product.get("_hydrated") else 1
            after = 0 if product.get("_aftershock") else 1
            price = _price_of(product)
            honor_after = bool(self.store.event_key and self.brief["duties"] & bait_duties)
            try:
                open_pos = open_order.index(key)
            except ValueError:
                open_pos = 10_000
            return (
                decoy,
                brand_rank,
                after if honor_after else 0,
                0 if _figure_ok(product, self.query, self.brief.get("spans") or []) else 1,
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
                and _figure_ok(item, self.query, list(self.brief.get("spans") or []))
            ]
        if not ranked:
            return None
        live = [item for item in ranked if _listing_key(item) != target and not self._stale(item)]
        if not live:
            live = [item for item in ranked if _listing_key(item) != target]
        spans = list(self.brief.get("spans") or [])
        if _needs_figure(self.brief) or self.brief["primary"] in {
            DUTY_ASK, DUTY_WHY, DUTY_FORK
        } or self.brief.get("firm_brand"):
            fitted = [item for item in live if _figure_ok(item, self.query, spans)]
            if fitted:
                live = fitted
            elif spans:
                return None
            brand = str(self.brief.get("brand") or "")
            if brand and (self.brief.get("firm_brand") or DUTY_WHY in self.brief["duties"]):
                branded = [item for item in live if _brand_hit(item, brand)]
                live = branded
                if not live:
                    return None
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
            budget = self.brief.get("budget")
            kind = str(self.brief.get("kind") or "")
            pool = [
                item
                for item in live
                if not _decoy(item, budget) and not _junk(item, kind)
            ] or [item for item in live if not _junk(item, kind)] or live
            branded = [item for item in pool if brand and _brand_hit(item, brand)]
            ready = [
                item
                for item in branded
                if item.get("_hydrated") and _figure_ok(item, self.query, spans)
            ]
            if not target and ready:
                return ready[0]
            if target:
                post = [item for item in (ready or branded or pool) if item.get("_aftershock")]
                fitted = [
                    item
                    for item in (post or ready or branded or pool)
                    if _figure_ok(item, self.query, spans) and not _junk(item, kind)
                ]
                # Never fall through to an off-spec SKU when the shopper typed figures.
                if fitted:
                    live = fitted
                elif ready:
                    live = ready
                elif spans:
                    return None
                else:
                    live = branded or []
            elif branded:
                return branded[0] if not spans else (ready[0] if ready else None)
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
                    item for item in live if stem and _sku_stem(item.get("sku")) == stem
                ]
                if siblings:
                    live = siblings
                elif modeled:
                    live = modeled
                elif self.store.search_n >= 3:
                    live = [item for item in live if not self._stale(item)]
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
        if DUTY_WHY in self.brief["duties"] or self.brief.get("firm_brand"):
            budget = self.brief.get("budget")
            fitted = [item for item in live if not _decoy(item, budget) and not _junk(item, str(self.brief.get("kind") or ""))]
            if fitted:
                live = fitted
            brand = str(self.brief.get("brand") or "")
            if brand:
                branded = [item for item in live if _brand_hit(item, brand)]
                # Firm / why brand: never fall through to an off-brand accessory.
                live = branded
            warranted = [
                item
                for item in live
                if item.get("_hydrated")
                and (
                    _catalog_spec_claims(self._full(item), self.query)
                    or _figure_ok(item, self.query, spans)
                )
            ]
            if warranted:
                live = warranted
            return live[0] if live else None
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
        if self.store.event_key and self.brief.get("spans"):
            brand = str(self.brief.get("brand") or "")
            keep_brand = bool(
                brand
                and (
                    self.brief.get("firm_brand")
                    or DUTY_WHY in self.brief["duties"]
                    or DUTY_ASK in self.brief["duties"]
                )
            )
            headed = []
            for span in (self.brief.get("spans") or [])[:2]:
                raw = str(span.get("raw") or "")
                if not raw:
                    continue
                if keep_brand:
                    headed.append(f"{brand} {raw} {kind}".strip())
                else:
                    headed.append(f"{raw} {kind}".strip())
            headed = [item for item in headed if _useful_query(item, kind=kind, budget=budget)]
            phrases = headed + [item for item in phrases if item not in headed]
        phrases = phrases or ([kind] if kind else ["product"])
        if DUTY_WHY in self.brief["duties"]:
            brand = str(self.brief.get("brand") or "")
            headed = []
            for span in (self.brief.get("spans") or [])[:2]:
                raw = str(span.get("raw") or "")
                if brand and raw:
                    headed.append(f"{brand} {raw} {kind}".strip())
                if raw and not brand:
                    headed.append(f"{raw} {kind}".strip())
            if brand:
                headed.append(f"{brand} {kind}".strip())
            headed = [item for item in headed if _useful_query(item, kind=kind, budget=budget)]
            phrases = headed + [item for item in phrases if item not in headed]
        if DUTY_MEND in self.brief["duties"]:
            cap = min(4, max(len(phrases), 1))
        elif DUTY_ASK in self.brief["duties"] and self.asked and self.brief.get("spans"):
            cap = min(5, max(len(phrases), 1))
        else:
            cap = min(3, max(len(phrases), 1)) if self.store.event_key else min(2, max(len(phrases), 1))
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
        # Soft preferred brand may drop after a shock; firm "I need a Brand X" never does.
        if (
            DUTY_FORK in self.brief["duties"]
            and self.store.event_key
            and not self.brief.get("firm_brand")
        ):
            brand = ""
        if (
            DUTY_HUNT in self.brief["duties"]
            and DUTY_FORK not in self.brief["duties"]
            and not self.brief.get("firm_brand")
            and DUTY_WHY not in self.brief["duties"]
            and DUTY_ASK not in self.brief["duties"]
        ):
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
        if DUTY_MEND in self.brief["duties"]:
            cap = 6
        elif self.store.event_key:
            cap = 3
        else:
            cap = 4
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
        if self.message and self.brief.get("ask") and not self.asked and _held_back(self.query):
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
        if (
            cart is not None
            and self.remove
            and DUTY_ASK in self.brief["duties"]
            and self.asked
            and _needs_figure(self.brief)
            and not _figure_ok(self._full(cart), self.query, list(self.brief.get("spans") or []))
        ):
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
            spec_duty = _needs_figure(self.brief) or self.brief["primary"] in {
                DUTY_ASK, DUTY_WHY, DUTY_FORK
            } or bool(self.brief.get("firm_brand"))
            if recovering:
                model = str(self.brief.get("model") or "")
                if spec_duty or _needs_figure(self.brief):
                    fitted = [
                        item
                        for item in live_now
                        if item.get("_hydrated")
                        and _figure_ok(item, self.query, list(self.brief.get("spans") or []))
                        and not _decoy(item, self.brief.get("budget"))
                    ]
                    need_search = self.store.search_n < 2 or not fitted
                else:
                    siblings = [item for item in live_now if not model or _model_hit(item, model)]
                    need_search = (not siblings) and self.store.search_n < 4
            else:
                need_search = self.store.search_n < (4 if DUTY_WHY in self.brief["duties"] else 5) and (
                    not live_now
                    or self.store.search_n < 2
                    or (
                        _needs_figure(self.brief)
                        and not any(
                            _figure_ok(item, self.query, list(self.brief.get("spans") or []))
                            for item in live_now
                            if item.get("_hydrated")
                        )
                    )
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
                twins = [item for item in live_now if stem and _sku_stem(item.get("sku")) == stem]
                if not twins and self.store.search_n < 4:
                    need_search = True
            if DUTY_WHY in self.brief["duties"] and not recovering:
                grounded = [
                    item
                    for item in live_now
                    if item.get("_hydrated")
                    and _figure_ok(item, self.query, list(self.brief.get("spans") or []))
                ]
                if not grounded and self.store.search_n < 4:
                    need_search = True
            if DUTY_TASTE in self.brief["duties"] and not recovering:
                wishes = list(self.brief.get("wishes") or [])
                if self.store.search_n < 6 and not any(
                    _taste_cover(item, wishes) >= 0.34 for item in live_now
                ):
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
            if hydrate:
                return hydrate[: self.max_calls]
            fallback = [item for item in self.live() if not self._stale(item)]
            if _needs_figure(self.brief):
                fitted = [
                    item
                    for item in fallback
                    if _figure_ok(item, self.query, list(self.brief.get("spans") or []))
                ]
                fallback = fitted
            champ = fallback[0] if fallback else None
            if champ is None:
                return self._search_actions() or []
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
        if not self.store.cart and self.cart and not self._stale(champ):
            spec = self.brief["duties"] & {DUTY_WHY, DUTY_TASTE, DUTY_FORK, DUTY_ASK}
            if spec and not champ.get("_hydrated"):
                return self._hydrate([champ])[: self.max_calls]
            if DUTY_WHY in self.brief["duties"] and not _facts(self._full(champ)):
                hydrate = self._hydrate([champ])
                if hydrate:
                    return hydrate[: self.max_calls]
            if not _seal_ok(
                self._full(champ),
                self.query,
                self.brief,
                require_spec_claim=DUTY_WHY in self.brief["duties"],
                order_tool=self.order,
            ) and (
                DUTY_ASK in self.brief["duties"]
                or DUTY_WHY in self.brief["duties"]
                or _needs_figure(self.brief)
            ):
                return self._search_actions() or []
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
            if self.store.event_open and self.inspect and cart_key not in self.store.gauged:
                return [
                    {
                        "name": _tool_name(self.inspect),
                        "args": _bind(
                            self.inspect,
                            {"product_id": cart_item.get("product_id"), "sku": cart_item.get("sku")},
                        ),
                    }
                ]
            if self.order:
                if cart_key in self.store.gauged:
                    self.store.event_open = False
                target = cart_item
                if champ is not None and champ.get("_gauged"):
                    target = champ
                if DUTY_WHY in self.brief["duties"] and not _facts(self._full(target)):
                    hydrate = self._hydrate([target])
                    if hydrate:
                        return hydrate[: self.max_calls]
                spans = list(self.brief.get("spans") or [])
                needs_figure = _needs_figure(self.brief)
                if needs_figure and not _figure_ok(self._full(target), self.query, spans):
                    if champ is not None and _figure_ok(self._full(champ), self.query, spans):
                        if self.remove and _listing_key(champ) != _listing_key(target):
                            return [
                                {
                                    "name": _tool_name(self.remove),
                                    "args": _bind(
                                        self.remove,
                                        {
                                            "product_id": target.get("product_id"),
                                            "sku": target.get("sku"),
                                        },
                                    ),
                                }
                            ]
                    return self._search_actions() or []
                if DUTY_WHY in self.brief["duties"]:
                    claims = _ground_claims(self._full(target), self.query, self.order)
                    if not _has_spec_claim(claims) and self.store.search_n < 4:
                        hydrate = self._hydrate([target])
                        if hydrate:
                            return hydrate[: self.max_calls]
                        return self._search_actions() or []
                if not _seal_ok(
                    self._full(target),
                    self.query,
                    self.brief,
                    require_spec_claim=DUTY_WHY in self.brief["duties"],
                    order_tool=self.order,
                ):
                    if (
                        DUTY_ASK in self.brief["duties"]
                        or DUTY_WHY in self.brief["duties"]
                        or _needs_figure(self.brief)
                    ):
                        champ_alt = self.champion()
                        if (
                            champ_alt is not None
                            and _listing_key(champ_alt) != _listing_key(target)
                            and self.remove
                            and _seal_ok(
                                self._full(champ_alt),
                                self.query,
                                self.brief,
                                require_spec_claim=DUTY_WHY in self.brief["duties"],
                                order_tool=self.order,
                            )
                        ):
                            return [
                                {
                                    "name": _tool_name(self.remove),
                                    "args": _bind(
                                        self.remove,
                                        {
                                            "product_id": target.get("product_id"),
                                            "sku": target.get("sku"),
                                        },
                                    ),
                                }
                            ]
                        return self._search_actions() or []
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
            if self.store.event_open and self.inspect and _listing_key(cart) not in self.store.gauged:
                return [
                    {
                        "name": _tool_name(self.inspect),
                        "args": _bind(
                            self.inspect,
                            {"product_id": cart.get("product_id"), "sku": cart.get("sku")},
                        ),
                    }
                ]
            spans = list(self.brief.get("spans") or [])
            needs_figure = _needs_figure(self.brief)
            if needs_figure and not _figure_ok(self._full(cart), self.query, spans):
                champ = self.champion()
                if champ is not None and _figure_ok(self._full(champ), self.query, spans):
                    if self.remove and _listing_key(champ) != _listing_key(cart):
                        return [
                            {
                                "name": _tool_name(self.remove),
                                "args": _bind(
                                    self.remove,
                                    {"product_id": cart.get("product_id"), "sku": cart.get("sku")},
                                ),
                            }
                        ]
                search = self._search_actions()
                if search:
                    return search
                # Still eject an ASK cart that fails the revealed figure.
                if self.remove and DUTY_ASK in self.brief["duties"] and self.asked:
                    return [
                        {
                            "name": _tool_name(self.remove),
                            "args": _bind(
                                self.remove,
                                {"product_id": cart.get("product_id"), "sku": cart.get("sku")},
                            ),
                        }
                    ]
                return []
            if DUTY_WHY in self.brief["duties"]:
                claims = _ground_claims(self._full(cart), self.query, self.order)
                if not _has_spec_claim(claims) and self.store.search_n < 4:
                    hydrate = self._hydrate([cart])
                    if hydrate:
                        return hydrate[: self.max_calls]
                    return self._search_actions() or []
            if not _seal_ok(
                self._full(cart),
                self.query,
                self.brief,
                require_spec_claim=DUTY_WHY in self.brief["duties"],
                order_tool=self.order,
            ):
                champ = self.champion()
                if champ is not None and _seal_ok(
                    self._full(champ),
                    self.query,
                    self.brief,
                    require_spec_claim=DUTY_WHY in self.brief["duties"],
                    order_tool=self.order,
                ):
                    if self.remove and _listing_key(champ) != _listing_key(cart):
                        return [
                            {
                                "name": _tool_name(self.remove),
                                "args": _bind(
                                    self.remove,
                                    {"product_id": cart.get("product_id"), "sku": cart.get("sku")},
                                ),
                            }
                        ]
                search = self._search_actions()
                if search:
                    return search
                if DUTY_ASK in self.brief["duties"] or DUTY_WHY in self.brief["duties"]:
                    return []
            self.store.event_open = False
            return [self._order(cart)]
        picked = self.champion()
        if picked is None:
            live = [item for item in self.live() if _listing_key(item) != self.store.event_key]
            if _needs_figure(self.brief):
                live = [
                    item
                    for item in live
                    if _figure_ok(item, self.query, list(self.brief.get("spans") or []))
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
            return self._search_actions() if self.store.search_n < 5 else []
        pid = str(picked.get("product_id") or "")
        if self.inspect and _listing_key(picked) not in self.store.gauged:
            return [
                {
                    "name": _tool_name(self.inspect),
                    "args": _bind(self.inspect, {"product_id": pid, "sku": picked.get("sku")}),
                }
            ]
        if self.cart and not self._stale(picked) and not self.store.cart:
            if not _seal_ok(
                self._full(picked),
                self.query,
                self.brief,
                require_spec_claim=DUTY_WHY in self.brief["duties"],
                order_tool=self.order,
            ) and (
                DUTY_ASK in self.brief["duties"]
                or DUTY_WHY in self.brief["duties"]
                or _needs_figure(self.brief)
            ):
                return self._search_actions() if self.store.search_n < 6 else []
            return [
                {
                    "name": _tool_name(self.cart),
                    "args": _bind(self.cart, {"product_id": pid, "sku": picked.get("sku")}),
                }
            ]
        if self.store.cart and self.order and not self._stale(self.store.cart[0]):
            if self.store.event_open and self.inspect and _listing_key(self.store.cart[0]) not in self.store.gauged:
                return [
                    {
                        "name": _tool_name(self.inspect),
                        "args": _bind(
                            self.inspect,
                            {
                                "product_id": self.store.cart[0].get("product_id"),
                                "sku": self.store.cart[0].get("sku"),
                            },
                        ),
                    }
                ]
            cart = self.store.cart[0]
            if not _seal_ok(
                self._full(cart),
                self.query,
                self.brief,
                require_spec_claim=DUTY_WHY in self.brief["duties"],
                order_tool=self.order,
            ):
                if self.remove and (
                    DUTY_ASK in self.brief["duties"]
                    or DUTY_WHY in self.brief["duties"]
                    or _needs_figure(self.brief)
                ):
                    return [
                        {
                            "name": _tool_name(self.remove),
                            "args": _bind(
                                self.remove,
                                {"product_id": cart.get("product_id"), "sku": cart.get("sku")},
                            ),
                        }
                    ]
                return self._search_actions() if self.store.search_n < 6 else []
            self.store.event_open = False
            return [self._order(cart)]
        return []

    def _next_beat(self) -> list[dict[str, Any]]:
        """Stall breaker: advance view → inspect → cart → order without IR."""
        champ = self.champion()
        if champ is None:
            live = [item for item in self.live() if not self._stale(item)]
            spans = list(self.brief.get("spans") or [])
            if _needs_figure(self.brief) or self.brief.get("firm_brand"):
                fitted = [
                    item
                    for item in live
                    if _figure_ok(item, self.query, spans)
                ]
                live = fitted
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
            if not _seal_ok(
                self._full(champ),
                self.query,
                self.brief,
                require_spec_claim=DUTY_WHY in self.brief["duties"],
                order_tool=self.order,
            ) and (
                DUTY_ASK in self.brief["duties"]
                or DUTY_WHY in self.brief["duties"]
                or _needs_figure(self.brief)
            ):
                return self._search_actions() or []
            return [
                {
                    "name": _tool_name(self.cart),
                    "args": _bind(self.cart, {"product_id": pid, "sku": sku}),
                }
            ]
        if self.store.cart and self.order and not self._stale(self.store.cart[0]):
            spans = list(self.brief.get("spans") or [])
            needs_figure = _needs_figure(self.brief)
            cart = self.store.cart[0]
            if needs_figure and not _figure_ok(self._full(cart), self.query, spans):
                return self._search_actions() or []
            if not _seal_ok(
                self._full(cart),
                self.query,
                self.brief,
                require_spec_claim=DUTY_WHY in self.brief["duties"],
                order_tool=self.order,
            ):
                if self.remove and (
                    DUTY_ASK in self.brief["duties"]
                    or DUTY_WHY in self.brief["duties"]
                    or needs_figure
                ):
                    return [
                        {
                            "name": _tool_name(self.remove),
                            "args": _bind(
                                self.remove,
                                {"product_id": cart.get("product_id"), "sku": cart.get("sku")},
                            ),
                        }
                    ]
                return self._search_actions() or []
            self.store.event_open = False
            return [self._order(cart)]
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
        if await_reply and self.message and not self.store.event_open:
            if not any("message" in str(item.get("name") or "").lower() for item in cleaned):
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
        if self.stall_guard:
            kinds = "|".join(
                sorted({str(action.get("name") or "").lower() for action in cleaned})
            )
            if kinds:
                self.store.signatures[kinds] += 1
            if cleaned and self.store.signatures[kinds] >= 3:
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
                    if self.store.event_open and self.inspect:
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
                    if not _seal_ok(
                        self._full(product),
                        self.query,
                        self.brief,
                        require_spec_claim=DUTY_WHY in self.brief["duties"],
                        order_tool=self.order,
                    ) and (
                        DUTY_ASK in self.brief["duties"]
                        or DUTY_WHY in self.brief["duties"]
                        or _needs_figure(self.brief)
                    ):
                        continue
                    aligned_soft.append(self._order(product))
                    continue
                aligned_soft.append({"name": name, "args": args})
            return aligned_soft or proposed[: self.max_calls]
        if DUTY_MEND in self.brief["duties"] and self.cart and not self.store.cart:
            champ = self.champion()
            model = str(self.brief.get("model") or "")
            color = str(self.brief.get("color") or "")
            ok = False
            if champ is not None and not self._stale(champ):
                if not self.store.event_key:
                    ok = _ladder(champ, model, color) == 0
                else:
                    ok = (not model) or _model_hit(champ, model)
            if ok:
                return [
                    {
                        "name": _tool_name(self.cart),
                        "args": _bind(
                            self.cart,
                            {"product_id": champ.get("product_id"), "sku": champ.get("sku")},
                        ),
                    }
                ]
            keep = [
                action
                for action in (cleaned or proposed)
                if any(
                    hint in str(action.get("name") or "").lower()
                    for hint in ("search", "view", "compare", "filter", "inspect")
                )
            ]
            if keep:
                return keep[: self.max_calls]
            hydrate = self._hydrate(self.live() or self.store.pool())
            if hydrate:
                return hydrate[: self.max_calls]
            search = self._search_actions()
            if search:
                return search
        order_name = _tool_name(self.order).lower() if self.order else ""
        cart_name = _tool_name(self.cart).lower() if self.cart else ""
        champ = self.champion()
        source = cleaned or proposed
        aligned: list[dict[str, Any]] = []
        for action in source[: self.max_calls]:
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
                if self.store.event_open and self.inspect and champ is not None:
                    aligned.append(
                        {
                            "name": _tool_name(self.inspect),
                            "args": _bind(
                                self.inspect,
                                {"product_id": champ.get("product_id"), "sku": champ.get("sku")},
                            ),
                        }
                    )
                    continue
                product = champ or cart_item
                if product is None or self._stale(product):
                    continue
                if DUTY_WHY in self.brief["duties"] and not _facts(self._full(product)):
                    hydrate = self._hydrate([product])
                    if hydrate:
                        aligned.extend(hydrate)
                        continue
                brand = str(self.brief.get("brand") or "")
                full = self._full(product)
                if DUTY_WHY in self.brief["duties"] and brand and not _brand_hit(full, brand):
                    if champ is not None and _brand_hit(self._full(champ), brand):
                        product = champ
                    elif self.store.search_n < 3:
                        search = self._search_actions()
                        if search:
                            aligned.extend(search)
                            continue
                spans = list(self.brief.get("spans") or [])
                needs_figure = _needs_figure(self.brief)
                if needs_figure and not _figure_ok(self._full(product), self.query, spans):
                    if champ is not None and _figure_ok(self._full(champ), self.query, spans):
                        product = champ
                    else:
                        search = self._search_actions()
                        if search:
                            aligned.extend(search)
                            continue
                        # Refuse to seal an off-spec SKU.
                        continue
                full = self._full(product)
                if not _seal_ok(
                    full,
                    self.query,
                    self.brief,
                    require_spec_claim=DUTY_WHY in self.brief["duties"],
                    order_tool=self.order,
                ):
                    if champ is not None and _seal_ok(
                        self._full(champ),
                        self.query,
                        self.brief,
                        require_spec_claim=DUTY_WHY in self.brief["duties"],
                        order_tool=self.order,
                    ):
                        product = champ
                    else:
                        search = self._search_actions()
                        if search:
                            aligned.extend(search)
                            continue
                        if DUTY_ASK in self.brief["duties"] or DUTY_WHY in self.brief["duties"]:
                            continue
                if DUTY_WHY in self.brief["duties"]:
                    claims = _ground_claims(self._full(product), self.query, self.order)
                    if not _has_spec_claim(claims) and self.store.search_n < 4:
                        hydrate = self._hydrate([product])
                        if hydrate:
                            aligned.extend(hydrate)
                            continue
                        search = self._search_actions()
                        if search:
                            aligned.extend(search)
                            continue
                        continue
                aligned.append(self._order(product))
                continue
            if lower == cart_name and champ is not None and self.rewrite_sku:
                if DUTY_MEND in self.brief["duties"] and not self.store.event_key:
                    model = str(self.brief.get("model") or "")
                    color = str(self.brief.get("color") or "")
                    if _ladder(champ, model, color) != 0:
                        continue
                if not _seal_ok(
                    self._full(champ),
                    self.query,
                    self.brief,
                    order_tool=self.order,
                ) and (
                    DUTY_ASK in self.brief["duties"]
                    or DUTY_WHY in self.brief["duties"]
                    or _needs_figure(self.brief)
                ):
                    search = self._search_actions()
                    if search:
                        aligned.extend(search)
                        continue
                    continue
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
        if cart_name and any(str(item.get("name") or "").lower() == cart_name for item in aligned):
            aligned = [
                item
                for item in aligned
                if str(item.get("name") or "").lower() != order_name
            ]
        if aligned:
            return aligned[: self.max_calls]
        beat = self._next_beat()
        if beat:
            return beat[: self.max_calls]
        return [
            item
            for item in proposed[: self.max_calls]
            if "order" not in str(item.get("name") or "").lower()
        ]

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
                # Shopper may restate brand/budget alongside the held figure.
                brand = _brand_guess(content)
                if brand:
                    self.brief["brand"] = brand
                    if not _soft_brand(self.query):
                        self.brief["firm_brand"] = True
                # Second budget parse — shopper often tightens after the ask.
                restated = _BUDGET_RX.search(content)
                if restated:
                    self.brief["budget"] = float(restated.group(1).replace(",", ""))
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
    if DUTY_SLATE in pilot.brief["duties"]:
        return True
    if DUTY_MEND in pilot.brief["duties"]:
        model = str(pilot.brief.get("model") or "")
        color = str(pilot.brief.get("color") or "")
        champ = pilot.champion()
        exact = bool(champ and _ladder(champ, model, color) == 0)
        sibling = bool(champ and pilot.store.event_key and _ladder(champ, model, color) <= 1)
        # After a shock, Python owns recovery — do not let the LLM thrash compare/search.
        if pilot.store.event_key and turn >= 2:
            return True
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
    if DUTY_ASK in pilot.brief["duties"] and pilot.asked and turn >= 3:
        champ = pilot.champion()
        if champ is not None and _seal_ok(
            pilot._full(champ), pilot.query, pilot.brief, order_tool=pilot.order
        ):
            return True
        # After a shock, keep Python in control for figure-gated ASK recovery.
        if pilot.store.event_key and pilot.brief.get("spans"):
            return True
    if DUTY_WHY in pilot.brief["duties"] and turn >= 3:
        champ = pilot.champion()
        if champ is not None and _seal_ok(
            pilot._full(champ),
            pilot.query,
            pilot.brief,
            require_spec_claim=True,
            order_tool=pilot.order,
        ):
            return True
        if turn >= 8 or pilot.store.search_n >= 4:
            return True
    # Figure-gated recovery: after a shock, stop LLM thrash once a fitted SKU exists.
    if pilot.store.event_key and _needs_figure(pilot.brief) and turn >= 3:
        return True
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
    if pilot.brief.get("ask") and not pilot.asked and turn == 1:
        return "One firm figure is withheld until you ask. Call message NOW before searching."
    if DUTY_SLATE in pilot.brief["duties"] and turn <= 2:
        return "Compare every named candidate before the first cart. Order with ONLY product_id and sku."
    if DUTY_WHY in pilot.brief["duties"] and turn <= 2:
        return "View facts before the first cart. Fill claims from viewed facts using only allowed fields."
    if DUTY_MEND in pilot.brief["duties"] and turn <= 2:
        return "After a stockout, keep the same model and change color first. Never order the event target."
    if DUTY_FORK in pilot.brief["duties"] and turn == 1:
        return "First cart honors the preferred brand only when that SKU also meets every firm gate."
    if DUTY_TASTE in pilot.brief["duties"] and turn <= 2 and not cart_used:
        return "Compare the shopper's stated use-case before the first cart. Do not cart the cheapest stub."
    if DUTY_HUNT in pilot.brief["duties"] and turn == 3:
        return "One small keyword reformulation so the shortlist covers several distinct products."
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
                    "View facts before the first cart. Fill claims from viewed "
                    "facts using only field names this episode's order tool allows. "
                    "Include the shopper's named spec plus in_stock and price."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "concord": {
                "primary": pilot.brief.get("primary"),
                "duties": sorted(pilot.brief.get("duties") or []),
                "model": model,
                "provider": _provider(),
                "stack": "dfi+electre3",
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
                "concord_turn": {"primary": pilot.brief.get("primary"), "turn": turn},
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
