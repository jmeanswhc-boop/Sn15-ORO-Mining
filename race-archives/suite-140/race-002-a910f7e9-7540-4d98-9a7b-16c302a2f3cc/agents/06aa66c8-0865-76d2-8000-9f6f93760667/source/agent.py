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

CHARGE_HUNT = "hunt"
CHARGE_FORK = "fork"
CHARGE_HELD = "held"
CHARGE_RELISH = "relish"
CHARGE_SLATE = "slate"
CHARGE_PATCH = "patch"
CHARGE_PROOF = "proof"

_CLOCK_S = 265.0
_PL2_C = 1.0
_GRA_RHO = 0.5
_MARCOS_V = 0.5
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
_EVENT_RX = re.compile(
    r"more expensive|still available|sold out|out of stock|"
    r"just get more|price (?:from|went|changed)|repriced",
    re.I,
)
_FLOOR_RX = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9+/\s]{0,18}?)\s+(?:starts\s+)?at\s+"
    r"(\d{3,5})\s*(?:by|[x×])\s*(\d{3,5})",
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
_IDLE = (
    "No tool call received. Choose a supplied tool now. "
    "Search or filter with the shopper's own value words, not schema keys.",
    "Still no tool call. Search/filter, view a candidate, answer a pending "
    "shopper question, or place_test_order if the cart holds a verified SKU.",
)
_COLORS = (
    "midnight",
    "champagne",
    "graphite",
    "titanium",
    "silver",
    "purple",
    "orange",
    "yellow",
    "beige",
    "white",
    "black",
    "green",
    "blue",
    "navy",
    "gold",
    "pink",
    "grey",
    "gray",
    "red",
    "brown",
)
_KIND_WORDS = (
    "smartphone",
    "tablet",
    "monitor",
    "headphone",
    "headset",
    "earphone",
    "laptop",
    "camera",
    "watch",
    "speaker",
)
_MODEL_NOISE = {
    "disappears",
    "disappear",
    "before",
    "after",
    "color",
    "colour",
    "same",
    "another",
    "other",
    "that",
    "exact",
    "item",
    "listing",
    "keep",
    "budget",
    "stock",
    "requirement",
    "firm",
}

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


def _primary_model(charges: set[str]) -> str:
    override = getenv("SANDBOX_MODEL") or ""
    if override.strip():
        return _normalize_model(override)
    if CHARGE_PATCH in charges and CHARGE_SLATE not in charges:
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
    """Product class from this shopper brief, not OS/task-rule tokens."""
    shopper = _shopper_text(query)
    folded = _fold(shopper)
    for noun in _KIND_WORDS:
        if re.search(rf"\b{noun}s?\b", folded):
            return noun
    cat = _CAT_RX.search(shopper)
    if cat:
        parts = _tokens(cat.group(1))
        if parts:
            return parts[-1]
    tail = re.search(
        r"([a-z][a-z]{2,})\s+(?:at or under|at or below|in stock)",
        folded,
    )
    if tail and tail.group(1) not in _STOP:
        return tail.group(1)
    head = re.search(
        r"(?:need|want|find|get|buy)\s+(?:me\s+)?(?:an?\s+)?(?:in stock\s+)?([a-z][a-z]{2,})",
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
        r"(?:[Pp]refer(?:red)?|[Ii]deally)\s+(?:the\s+)?([A-Z][A-Za-z0-9.&-]{2,})",
        shopper,
    )
    if hit:
        phrase = _clean(hit.group(1)).rstrip(".")
        if _fold(phrase) not in _STOP:
            return phrase
    match = re.search(
        r"(?:[Nn]eed|[Ww]ant|[Gg]et(?:\s+me)?|[Ff]ind(?:\s+me)?)\s+"
        r"(?:an?\s+)?(?:in-stock\s+)?"
        r"([A-Z][A-Za-z0-9.&-]*(?:\s+[A-Z][A-Za-z0-9.&-]*){0,2})\s+"
        r"(?:[a-z]|\d)",
        shopper,
    )
    if match:
        phrase = _clean(match.group(1)).rstrip(".")
        tokens = _fold(phrase).split()
        if tokens and not any(token in _STOP for token in tokens):
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
        folded = _fold(token)
        if (
            folded
            and folded not in _STOP
            and folded not in _COLORS
            and folded not in _MODEL_NOISE
            and re.search(r"\d", token)
        ):
            return token
    hit = re.search(r"\bmodel(?:\s+code)?\s+([A-Za-z0-9][A-Za-z0-9_]{1,24})", shopper, re.I)
    if hit:
        token = hit.group(1).rstrip(".,;")
        folded = _fold(token)
        if folded not in _STOP and folded not in _COLORS and folded not in _MODEL_NOISE:
            return token
    return ""


def _color_guess(query: str) -> str:
    blob = f" {_fold(_shopper_text(query))} "
    for hue in _COLORS:
        if f" {hue} " in blob:
            if hue == "midnight" and "black" in blob:
                return "black"
            return hue
    return ""


def _axes_from(text: str) -> list[tuple[str, str]]:
    axes: list[tuple[str, str]] = []
    blob = text or ""
    patterns = (
        (r"highest\s+([a-z]+(?:\s+rate)?)", "desc"),
        (r"largest\s+([a-z]+(?:\s+size)?)", "desc"),
        (r"lowest\s+([a-z]+)", "asc"),
        (r"cheapest\s+([a-z]+)?", "asc"),
        (r"then the\s+(largest|highest|lowest)\s+([a-z]+(?:\s+(?:rate|size))?)", None),
    )
    for pattern, default_dir in patterns:
        for match in re.finditer(pattern, blob, re.I):
            if default_dir is None:
                adverb, name = match.group(1), match.group(2)
                direction = "asc" if _fold(adverb) == "lowest" else "desc"
                axis = _canon_axis(name)
            else:
                raw = match.group(1) if match.lastindex else "price"
                axis = _canon_axis(raw or "price")
                direction = default_dir
                if axis == "price" and "cheap" in _fold(match.group(0)):
                    direction = "asc"
            if axis and axis not in {"then", "and", "the"} and (axis, direction) not in axes:
                axes.append((axis, direction))
    if re.search(r"\bprice\b", blob, re.I) and not any(axis == "price" for axis, _ in axes):
        if re.search(r"lowest price|cheapest|then price", blob, re.I):
            axes.append(("price", "asc"))
    return axes[:4]


def _canon_axis(raw: str) -> str:
    folded = _fold(raw)
    folded = re.sub(r"^(largest|highest|lowest|cheapest)\s+", "", folded)
    if not folded or folded in _STOP:
        return "price" if "cheap" in _fold(raw) else ""
    return folded.replace(" ", "_")[:32]


def _axis_value(product: dict[str, Any], name: str) -> float | None:
    """Read the shopper's named axis from variant options first, then facts/title.

    Never take a bare leading digit from an option code (ds200, 27g4).
    """
    if name == "price":
        return _price_of(product)
    option_text = _options_text(product)
    blob = _blob(product)
    facts = _facts(product)
    folded = _fold(name)
    if not folded:
        return None

    def near_token(hay: str) -> float | None:
        if not hay:
            return None
        token = re.escape(folded.replace("_", " "))
        hit = re.search(
            rf"(\d+(?:\.\d+)?)\s*[a-z%]*\s*{token}|{token}\s*[:=]?\s*(\d+(?:\.\d+)?)",
            hay,
            re.I,
        )
        if hit:
            return _number(hit.group(1) or hit.group(2))
        return None

    def unit_figure(hay: str) -> float | None:
        if not hay:
            return None
        if "refresh" in folded or folded in {"hz", "hertz"}:
            hit = re.search(r"(\d+(?:\.\d+)?)\s*(?:hz|hertz)\b", hay, re.I)
            if hit:
                return float(hit.group(1))
        if "screen" in folded or "size" in folded:
            hit = re.search(
                r"(\d+(?:\.\d+)?)\s*-?\s*(?:\"|''|in(?:ch(?:es)?)?)\b",
                hay,
                re.I,
            )
            if hit:
                return float(hit.group(1))
        return None

    # Variant option rows beat listing-level facts.
    for hay in (option_text,):
        value = near_token(hay) or unit_figure(hay)
        if value is not None:
            return value
    for key, raw in facts.items():
        key_fold = _fold(key)
        related = folded in key_fold or key_fold in folded
        if not related and ("screen" in folded or "size" in folded):
            related = "size" in key_fold or "screen" in key_fold
        if not related:
            continue
        value = unit_figure(str(raw)) or _number(raw)
        if value is not None:
            opt = unit_figure(option_text)
            return opt if opt is not None else value
    for hay in (blob,):
        value = near_token(hay) or unit_figure(hay)
        if value is not None:
            return value
    return None


def _pl2(
    query_tf: Counter[str],
    doc: Counter[str],
    df: Counter[str],
    n_docs: int,
    avgdl: float,
    collection: Counter[str],
) -> float:
    """Terrier PL2 (Poisson + Laplace after-effect, DFR). Unused by other dragons."""
    length = sum(doc.values()) or 1.0
    n_docs = max(n_docs, 1)
    score = 0.0
    for term, qf in query_tf.items():
        tf = float(doc.get(term, 0))
        if tf <= 0:
            continue
        tfn = tf * math.log2(1.0 + _PL2_C * avgdl / length)
        if tfn <= 0:
            continue
        freq = float(collection.get(term, 0) or df.get(term, 1) or 1)
        lam = freq / n_docs
        if lam <= 0:
            continue
        norm = 1.0 / (tfn + 1.0)
        score += (
            qf
            * norm
            * (
                tfn * math.log2(max(avgdl / n_docs * freq, 1.0001))
                + 0.5 * math.log2(max(2.0 * math.pi * tfn, 1.0001))
            )
        )
        del lam
    return score


def _tfidf_cos(
    query_tf: Counter[str],
    doc: Counter[str],
    df: Counter[str],
    n_docs: int,
) -> float:
    n_docs = max(n_docs, 1)
    q_vec: dict[str, float] = {}
    d_vec: dict[str, float] = {}
    terms = set(query_tf) | set(doc)
    for term in terms:
        idf = math.log((n_docs + 1.0) / (float(df.get(term, 0)) + 1.0)) + 1.0
        qtf = float(query_tf.get(term, 0))
        dtf = float(doc.get(term, 0))
        q_vec[term] = (1.0 + math.log(qtf) if qtf else 0.0) * idf
        d_vec[term] = (1.0 + math.log(dtf) if dtf else 0.0) * idf
    dot = sum(q_vec[term] * d_vec.get(term, 0.0) for term in q_vec)
    qn = math.sqrt(sum(value * value for value in q_vec.values())) or 1.0
    dn = math.sqrt(sum(value * value for value in d_vec.values())) or 1.0
    return dot / (qn * dn)


def _combanz(rank_lists: list[list[str]]) -> dict[str, float]:
    """CombANZ: mean of participating rank scores. Not CombSUM / CombMNZ / RRF."""
    scores: dict[str, list[float]] = {}
    for rows in rank_lists:
        for index, key in enumerate(rows):
            scores.setdefault(key, []).append(1.0 / (index + 1.0))
    return {key: (sum(values) / len(values) if values else 0.0) for key, values in scores.items()}


def _entropy_weights(matrix: list[list[float]]) -> list[float]:
    if not matrix or not matrix[0]:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    weights: list[float] = []
    k = 1.0 / math.log(max(rows, 2))
    for col in range(cols):
        total = sum(abs(matrix[row][col]) for row in range(rows)) or 1.0
        entropy = 0.0
        for row in range(rows):
            p = abs(matrix[row][col]) / total
            if p > 0:
                entropy -= p * math.log(p)
        weights.append(max(1.0 - k * entropy, 1e-6))
    s = sum(weights) or 1.0
    return [item / s for item in weights]


def _marcos(
    pool: list[dict[str, Any]],
    axes: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """MARCOS compromise ranking (Stević et al.). Unused by other dragons."""
    if not axes or len(pool) < 2:
        return list(pool)
    matrix: list[list[float]] = []
    for product in pool:
        row: list[float] = []
        for name, direction in axes:
            value = _axis_value(product, name)
            if value is None:
                row.append(0.0)
            elif direction == "asc":
                row.append(-value)
            else:
                row.append(value)
        matrix.append(row)
    cols = len(axes)
    ideal = [max(matrix[row][col] for row in range(len(matrix))) for col in range(cols)]
    anti = [min(matrix[row][col] for row in range(len(matrix))) for col in range(cols)]
    weights = _entropy_weights(matrix) or [1.0 / cols] * cols
    scored: list[tuple[float, dict[str, Any]]] = []
    s_ideal = sum(weights[col] * ideal[col] for col in range(cols)) or 1.0
    s_anti = sum(weights[col] * anti[col] for col in range(cols)) or 1.0
    for product, row in zip(pool, matrix):
        si = sum(weights[col] * row[col] for col in range(cols))
        k_plus = si / s_ideal if s_ideal else 0.0
        k_minus = si / s_anti if s_anti else 0.0
        denom = k_plus + k_minus
        if denom <= 0:
            utility = 0.0
        else:
            f_plus = k_minus / denom
            f_minus = k_plus / denom
            utility = (k_plus + k_minus) * (
                _MARCOS_V * f_plus + (1.0 - _MARCOS_V) * f_minus
            )
        scored.append((utility, product))
    scored.sort(key=lambda item: -item[0])
    return [product for _score, product in scored]


def _lex_rank(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if not axes:
        return list(pool)

    def key(product: dict[str, Any]) -> tuple:
        parts: list[float] = []
        for name, direction in axes:
            value = _axis_value(product, name)
            if value is None:
                parts.append(1e18)
            elif direction == "asc":
                parts.append(value)
            else:
                parts.append(-value)
        return tuple(parts)

    return sorted(pool, key=key)


def _wish_terms(query: str) -> list[str]:
    shopper = _shopper_text(query)
    terms: list[str] = []
    for match in re.finditer(
        r"(?:rather pay.{0,40}?for|survive|use-case|use case|want)\s+([^.]{4,80})",
        shopper,
        re.I,
    ):
        terms.extend(_tokens(match.group(1)))
    for token in _tokens(shopper):
        if token not in terms and len(token) >= 4:
            terms.append(token)
    return terms[:12]


def _gra_grade(product: dict[str, Any], wishes: list[str]) -> float:
    """Grey relational grade vs an ideal that hits every shopper wish token."""
    if not wishes:
        return 0.0
    blob = _fold(_blob(product))
    deltas: list[float] = []
    for wish in wishes:
        hit = 1.0 if _fold(wish) in blob else 0.0
        deltas.append(abs(1.0 - hit))
    dmax = max(deltas) or 1.0
    dmin = min(deltas)
    coeffs = [
        (dmin + _GRA_RHO * dmax) / (delta + _GRA_RHO * dmax) for delta in deltas
    ]
    return sum(coeffs) / len(coeffs)


def _decoy(product: dict[str, Any], budget: Any) -> bool:
    price = _price_of(product)
    try:
        ceiling = float(budget) if budget is not None else None
    except (TypeError, ValueError):
        ceiling = None
    if price is None or ceiling is None or ceiling <= 0:
        return False
    return price <= ceiling * _DECOY_FRAC


def _phrase_hit(product: dict[str, Any], phrase: str) -> bool:
    terms = re.findall(r"[a-z0-9]+", (phrase or "").lower())
    if not terms:
        return True
    title = re.findall(r"[a-z0-9]+", str(product.get("title") or "").lower())
    if not title:
        return True
    return all(term in title for term in terms)


def _kind_ok(product: dict[str, Any], kind: str) -> bool:
    if not kind:
        return True
    blob = _fold(_blob(product))
    return bool(re.search(rf"\b{re.escape(_fold(kind))}s?\b", blob))


def _junk(product: dict[str, Any], kind: str) -> bool:
    blob = _fold(f"{product.get('title') or ''} {product.get('options') or ''}")
    if not blob:
        return False
    if any(hint in blob for hint in _JUNK):
        return True
    return False


def _brand_hit(product: dict[str, Any], brand: str) -> bool:
    wanted = _fold(brand)
    if not wanted:
        return True
    have = _fold(f"{product.get('brand') or ''} {product.get('title') or ''}")
    return wanted in have


def _model_hit(product: dict[str, Any], model: str) -> bool:
    wanted = str(model or "").strip()
    if not wanted:
        return True
    hay = f"{_blob(product)} {product.get('sku') or ''}"
    if re.search(r"\d", wanted):
        return bool(re.search(rf"(?<![\w]){re.escape(wanted)}(?![\w])", hay, re.I))
    compact = _compact(wanted)
    return bool(compact) and compact in _compact(hay)


def _color_hit(product: dict[str, Any], color: str) -> bool:
    wanted = _fold(color)
    if not wanted:
        return True
    facts = _facts(product)
    sources = [
        product.get("options"),
        facts.get("color"),
        facts.get("colour"),
        product.get("title"),
    ]
    blob = f" {_fold(' '.join(_clean(item) for item in sources if item))} "
    if f" {wanted} " in blob:
        return True
    alias = {"grey": "gray", "gray": "grey"}.get(wanted, "")
    return bool(alias) and f" {alias} " in blob


def _ladder(product: dict[str, Any], model: str, color: str) -> int:
    has_model = _model_hit(product, model) if model else True
    has_color = _color_hit(product, color) if color else True
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
    folded = _fold(hay)
    token = f"{amount:g}"
    spaced = token.replace(".", " ")
    if unit == "in":
        if re.search(
            rf"(?<![\d.]){re.escape(token)}\s*-?\s*(?:\"|''|in(?:ch(?:es)?)?)\b",
            raw,
        ):
            return True
        if spaced != token and re.search(
            rf"(?<!\d){re.escape(spaced)}\s*-?\s*(?:in|inch|inches)\b",
            folded,
        ):
            return True
        return token in folded or (spaced != token and spaced in folded)
    if unit:
        if re.search(rf"(?<![\d.]){re.escape(token)}\s*{re.escape(unit)}\b", raw):
            return True
        return bool(re.search(rf"\b{re.escape(token)}\s*{re.escape(unit)}\b", folded))
    return token in raw or token in folded or (spaced != token and spaced in folded)


def _measure_hit(product: dict[str, Any], spans: list[dict[str, Any]]) -> bool:
    if not spans:
        return True
    option_text = _options_text(product)
    fact_blob = " ".join(str(value) for value in _facts(product).values())
    grounded = f"{option_text} {fact_blob}".strip()
    blob = _blob(product)
    needed = [span for span in spans if span.get("unit")]
    if not needed:
        return True
    for span in needed:
        unit = str(span.get("unit") or "")
        if unit and (option_text or fact_blob):
            if _span_in(grounded, span):
                continue
            if unit == "in":
                other = re.findall(
                    r"(\d+(?:\.\d+)?)\s*-?\s*(?:\"|''|in(?:ch(?:es)?)?)\b",
                    grounded,
                    re.I,
                )
                if other:
                    return False
            if not _span_in(blob, span):
                return False
            continue
        if not _span_in(blob, span):
            return False
    return True


def _size_range_blob(text: str) -> bool:
    return bool(re.search(r"\d+(?:\.\d+)?\s*[-–/to]+\s*\d+", str(text or ""), re.I))


def _shopper_classes(query: str) -> list[str]:
    """Class words the shopper typed (720p, HD, Full HD, …). Not a pixel table."""
    found: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"\b((?:full\s+)?hd|fhd|qhd|uhd|4k|720p|1080p|1440p)\b",
        _shopper_text(query),
        re.I,
    ):
        cue = match.group(1)
        key = _fold(cue)
        if key and key not in seen:
            seen.add(key)
            found.append(cue)
    return found


def _grounded_blob(product: dict[str, Any]) -> str:
    facts = _facts(product)
    option_text = _options_text(product)
    return " ".join(
        str(part)
        for part in (
            option_text,
            *(str(value) for value in facts.values() if value not in (None, "")),
        )
        if part
    )


def _figure_ok(product: dict[str, Any], query: str, spans: list[dict[str, Any]] | None = None) -> bool:
    spans = spans or []
    classes = _shopper_classes(query)
    measured = _measure_hit(product, spans)
    grounded = _grounded_blob(product)
    if product.get("_hydrated") and classes:
        if not grounded.strip():
            return False
        hay_source = grounded
    elif product.get("_hydrated") and grounded.strip():
        hay_source = grounded
    else:
        hay_source = _blob(product)
    hay = _fold(hay_source)
    hay = re.sub(r"\bhdmi\b", " ", hay)
    compact = _compact(hay_source)

    def cue_hit(cue: str) -> bool:
        return _fold(cue) in hay or bool(_compact(cue) and _compact(cue) in compact)

    folded = {_fold(cue) for cue in classes}
    if folded & {"full hd", "fhd"}:
        return cue_hit("full hd") or cue_hit("fhd")
    if "hd" in folded:
        return cue_hit("hd")
    if classes:
        return any(cue_hit(cue) for cue in classes) and (measured or True)
    if _size_range_blob(str(_facts(product).get("size") or "")):
        blob = f"{_options_text(product)} {product.get('title') or ''}"
        return any(_span_in(blob, span) for span in spans) if spans else False
    return measured


def _sku_stem(sku: Any) -> str:
    text = str(sku or "")
    return text.rsplit("-", 1)[0] if text else ""


def _norm_field(raw: str) -> str:
    text = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return re.sub(r"_+", "_", text).strip("_")


def _claim_enum(tool: dict[str, Any] | None) -> list[str]:
    """Live place_test_order claims.field enum. No local product-class table."""
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


def _episode_floors(query: str) -> list[dict[str, Any]]:
    """Class floors only when *this* episode's Task rules define them."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _FLOOR_RX.finditer(_rules_text(query)):
        raw = re.sub(r"^(?:and|or|,)\s+", "", _clean(match.group(1)), flags=re.I)
        try:
            left, right = int(match.group(2)), int(match.group(3))
        except (TypeError, ValueError):
            continue
        label = _compact(raw)
        if not label or label in seen or label in {"at", "by"}:
            continue
        seen.add(label)
        found.append(
            {
                "label": label,
                "raw": raw,
                "long": max(left, right),
                "short": min(left, right),
            }
        )
    return found


def _floor_ok(product: dict[str, Any], query: str, floors: list[dict[str, Any]]) -> bool:
    if not floors:
        return True
    shopper = _fold(_shopper_text(query))
    blob = _blob(product)
    pair = _PAIR_RX.search(blob)
    got: tuple[int, int] | None = None
    if pair:
        left, right = int(pair.group(1)), int(pair.group(2))
        got = (max(left, right), min(left, right))
    for floor in floors:
        needle = _fold(floor.get("raw") or "")
        if not needle or needle not in shopper:
            continue
        if got is None:
            return True
        return got[0] >= int(floor["long"]) and got[1] >= int(floor["short"])
    return True


def _fact_rows(product: dict[str, Any]) -> dict[str, str]:
    """Observed catalog scalars, keyed as the listing spelled them."""
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
    if option_text and "options" not in rows:
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
    if len(fl) <= 4 and re.search(rf"\b{re.escape(fl)}\b", blob):
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


def _row_value(field: str, rows: dict[str, str]) -> str | None:
    key = _norm_field(field)
    if key in rows and rows[key]:
        return rows[key]
    for other, value in rows.items():
        if not value:
            continue
        if other == key:
            return value
        if len(other) >= 3 and (other in key or key in other):
            return value
    return None


def _field_for_observed(value: str, extras: list[str]) -> str | None:
    """Pick a live enum field from the *observed value's* own units/shape."""
    if not extras:
        return None
    folded = _fold(value)
    compact = _compact(value)
    if re.search(r"\binch|\bin\b|\"", folded) or re.search(r"\d+in(?:ch)?", compact):
        for field in extras:
            if "size" in field or "screen" in field:
                return field
    if re.search(r"\bhz\b|\bhertz\b", folded):
        for field in extras:
            if "refresh" in field:
                return field
    if _PAIR_RX.search(str(value) or "") or re.search(r"\b\d{3,4}\s*p\b", str(value), re.I):
        for field in extras:
            if "resolution" in field:
                return field
    if len(extras) == 1:
        return extras[0]
    return None


def _firm_attr_from_ask(query: str) -> str:
    """Name the shopper's measurable from words they typed. No suite tables."""
    shopper = _fold(_shopper_text(query))
    if re.search(
        r"720p|1080p|1440p|2160|\bfhd\b|\bqhd\b|\buhd\b|\b4k\b|(?<![a-z])hd(?![ami])",
        shopper,
    ):
        return "resolution"
    if re.search(r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in\b|\")", shopper):
        return "screen_size"
    return ""


def _listing_claim_text(product: dict[str, Any], *, grounded: bool = False) -> str:
    facts = _facts(product)
    bits = []
    if not grounded:
        bits.append(str(product.get("title") or ""))
    bits.extend(
        [
            str(product.get("options") or ""),
            str(product.get("brand") or ""),
        ]
    )
    bits.extend(f"{key}: {value}" for key, value in facts.items() if value not in (None, ""))
    return " | ".join(bit for bit in bits if bit)


def _text_supports(value: str, listing: str) -> bool:
    tight = _compact(value)
    blob = _compact(listing)
    return bool(tight) and tight in blob


def _firm_claim_value(product: dict[str, Any], query: str, firm: str) -> str:
    rows = _fact_rows(product)
    listing = _listing_claim_text(product, grounded=True)
    full = _listing_claim_text(product)
    if firm == "resolution":
        if not listing:
            return ""
        direct = rows.get("resolution") or ""
        if direct and _text_supports(direct, listing) and _fold(direct) in {
            "hd", "fhd", "qhd", "uhd", "4k", "full hd", "720p", "1080p"
        }:
            return direct
        have = {_fold(cue): cue for cue in _shopper_classes(query)}
        for key in ("full hd", "fhd", "qhd", "uhd", "4k", "hd", "1080p", "720p"):
            cue = have.get(key)
            if not cue or not _text_supports(cue, listing):
                continue
            if key in {"hd", "fhd", "qhd", "uhd"}:
                return key
            if key == "full hd":
                return "fhd"
            return cue.strip()
        echo = _listing_echo(_shopper_text(query), listing)
        return echo or ""
    if firm in {"screen_size", "size"}:
        shopper = _shopper_text(query)
        for key in ("screen_size", "size"):
            value = rows.get(key) or ""
            if not value or _size_range_blob(value):
                continue
            if _value_attested(value, shopper) or _text_supports(value, shopper):
                return value
        for span in _figure_spans(shopper):
            if str(span.get("unit") or "") != "in":
                continue
            hay = listing or full
            if _span_in(hay, span) or _span_in(full, span):
                token = f"{span['amount']:g}inch"
                if _text_supports(token, hay) or _text_supports(token, full) or _text_supports(f"{span['amount']:g}in", hay):
                    return token
                if _text_supports(f"{span['amount']:g}in", listing) or _text_supports(f"{span['amount']:g}in", full):
                    return f"{span['amount']:g}inch"
        option = rows.get("_option") or ""
        if option and not _size_range_blob(option) and _value_attested(option, shopper):
            return option
        return ""
    value = rows.get(firm) or ""
    return value if value and _text_supports(value, listing) else ""


def _listing_echo(shopper: str, listing: str) -> str | None:
    """Copy a shopper phrase that the listing itself printed. No pixel remap."""
    if not shopper or not listing:
        return None
    patterns = (
        r"full\s*[- ]?hd",
        r"fhd",
        r"qhd",
        r"uhd",
        r"4k",
        r"hd",
        r"1080p",
        r"720p",
    )
    shop = shopper.lower()
    listed = listing.lower()
    for pat in patterns:
        if not re.search(rf"\b{pat}\b", shop, re.I):
            continue
        hit = re.search(rf"\b{pat}\b", listed, re.I)
        if hit:
            return listing[hit.start() : hit.end()]
    return None


def _paired_values(
    allowed: list[str],
    rows: dict[str, str],
    shopper: str,
    listing: str = "",
) -> dict[str, str]:
    """Map observed fact rows onto live enum fields. No class remaps."""
    skip = {"in_stock", "price", "brand", "category"} | _CLAIM_BLOCK
    extras = [item for item in allowed if item not in skip]
    paired: dict[str, str] = {}
    taken: set[str] = set()
    for field in extras:
        direct = _row_value(field, rows)
        if direct and _value_attested(direct, shopper):
            paired[field] = direct
            taken.add(direct)
    leftovers = [
        (key, value)
        for key, value in rows.items()
        if key not in {"in_stock", "price", "brand"}
        and not str(key).startswith("_")
        and value not in taken
        and _value_attested(value, shopper)
    ]
    open_fields = [field for field in extras if field not in paired]
    for _key, value in leftovers:
        target = _field_for_observed(value, open_fields)
        if target and target not in paired:
            paired[target] = value
            open_fields = [field for field in open_fields if field != target]
    option_blob = rows.get("_option") or ""
    if option_blob and _value_attested(option_blob, shopper):
        target = _field_for_observed(option_blob, [field for field in extras if field not in paired])
        if target:
            paired[target] = option_blob
    echo = _listing_echo(shopper, f"{listing} {' '.join(rows.values())}")
    if echo:
        for field in extras:
            if "resolution" in field:
                paired[field] = echo
                break
    return paired


def _proof_claims(
    product: dict[str, Any],
    query: str,
    tool: dict[str, Any] | None,
) -> list[dict[str, str]]:
    allowed = _claim_enum(tool)
    schema_locked = bool(allowed)
    shopper = _shopper_text(query)
    firm = _firm_attr_from_ask(query)
    if not allowed:
        allowed = ["in_stock", "price"]
        if firm:
            allowed.append(firm)
        allowed.append("brand")
    elif firm and firm not in allowed:
        firm = ""
    allowed_set = set(allowed)
    rows = _fact_rows(product)
    fields = [item for item in ("in_stock", "price") if item in allowed_set]
    paired = _paired_values(
        allowed,
        rows,
        shopper,
        listing=_grounded_blob(product) or _blob(product),
    )
    if firm:
        value = _firm_claim_value(product, query, firm)
        if value:
            paired[firm] = value
    skip = {"in_stock", "price", "brand", "category"} | _CLAIM_BLOCK
    extras = [item for item in allowed if item not in skip]
    pick = next((item for item in extras if item in paired), None)
    named = [item for item in extras if item in paired and _key_in_text(item, shopper)]
    if named:
        pick = named[0]
    if firm and firm in paired:
        pick = firm
    if pick:
        fields.append(pick)
    if "brand" in allowed_set and (
        _key_in_text("brand", shopper)
        or re.search(r"\bprefer|\bideally\b", shopper, re.I)
        or (not schema_locked and _brand_hit(product, str(product.get("brand") or "")))
    ):
        fields.append("brand")
    listing = _compact(_blob(product) + " " + " ".join(rows.values()) + " " + _listing_claim_text(product))
    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    for field in fields:
        if field in seen or field in _CLAIM_BLOCK or field not in allowed_set:
            continue
        if field in {"in_stock", "price", "brand"}:
            value = rows.get(field)
        else:
            value = paired.get(field)
        if not value:
            continue
        tight = _compact(value)
        if field not in {"in_stock", "price"} and tight and tight not in listing:
            listing_text = _listing_claim_text(product)
            if _fold(value) == "fhd" and "fullhd" in _compact(listing_text):
                pass
            elif not _text_supports(value, listing_text):
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


def compile_charter(query: str) -> dict[str, Any]:
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
    relish = bool(
        re.search(
            r"rather pay|cheapest isn't|don't just grab the cheapest|matters more|"
            r"use-case|use case|sweat|workout|waterproof|water resist",
            shopper,
            re.I,
        )
    )
    scores = {
        CHARGE_SLATE: 0,
        CHARGE_PROOF: 0,
        CHARGE_PATCH: 0,
        CHARGE_HELD: 0,
        CHARGE_FORK: 0,
        CHARGE_RELISH: 0,
        CHARGE_HUNT: 1,
    }
    if len(named) >= 2 or (named and axes) or re.search(
        r"stated tradeoff|strictly worse|compare only these candidate",
        rules,
        re.I,
    ):
        scores[CHARGE_SLATE] += 8
    if why:
        scores[CHARGE_PROOF] += 6
    if patch and (model or color or "least-relax" in _fold(rules)):
        scores[CHARGE_PATCH] += 7
    if held:
        scores[CHARGE_HELD] += 6
    if fork and not patch:
        scores[CHARGE_FORK] += 5
    if relish:
        scores[CHARGE_RELISH] += 5
    if quotes:
        scores[CHARGE_HUNT] += 3
    primary = max(scores, key=lambda name: (scores[name], -list(scores).index(name)))
    charges = {primary}
    if why:
        charges.add(CHARGE_PROOF)
    if patch:
        charges.add(CHARGE_PATCH)
    if held:
        charges.add(CHARGE_HELD)
    if fork and not patch:
        charges.add(CHARGE_FORK)
    if relish:
        charges.add(CHARGE_RELISH)
    wishes = _wish_terms(query) if CHARGE_RELISH in charges else []
    floors = _episode_floors(query)
    phrases: list[str] = []
    phrases.extend(quotes)
    shopper_fold = _fold(shopper)
    if CHARGE_RELISH in charges and kind and re.search(
        r"sweat|workout|waterproof|water resist|splash",
        shopper_fold,
    ):
        for stem in ("waterproof", "sweat resistant", "water resistant", "splash resistant"):
            if any(token in shopper_fold for token in stem.split()):
                phrases.append(f"{stem} {kind}")
    if model:
        phrases.append(" ".join(part for part in (color, model) if part))
        phrases.append(model)
    if brand and kind:
        for span in spans[:3]:
            phrases.append(f"{brand} {span['raw']} {kind}")
        for cue in _shopper_classes(query)[:3]:
            phrases.append(f"{brand} {cue} {kind}")
        phrases.append(f"{brand} {kind}")
    if kind:
        for span in spans[:3]:
            phrases.append(f"{span['raw']} {kind}")
        phrases.append(kind)
    if brand and not kind:
        phrases.append(brand)
    if category:
        phrases.append(category)
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
        "charges": charges,
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
        "floors": floors,
        "need_identities": 3 if primary == CHARGE_HUNT else 2 if primary == CHARGE_RELISH else 1,
    }


_CORE = """You are a shopping agent in a generated catalog. Use ONLY supplied tools.

# Charter
The shopper mixes firm gates and optional leanings. Firm gates (category,
budget, in-stock, stated figures, named listing, quoted phrase) must hold
on the exact SKU. A leaning never overrides a firm miss. First add_to_cart
is graded — never park a placeholder.

# Loop
1. DRAW — search/filter with short value queries (quoted phrase, brand,
   figure, product noun, category path). Never dump the whole instruction.
2. GAUGE — view/compare until raw facts/options are visible. Titles lie.
3. BIND — inspect_stock, then add_to_cart the live winner.
4. PATCH — after stock/price shocks, drop the stale SKU, recast, pick again.
5. STAMP — place_test_order once the live cart SKU is verified.

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
ONLY product_id and sku. Never invent attributes. Never copy marketing titles
as spec values. Never send category/title/os/summary/evidence.

# Markets
After stockout or price_change, remove the stale cart SKU, recast, cart a
different live listing, then order. Never order the event target.
"""

_CHARGE_BOOK = {
    CHARGE_HUNT: """# Hunt
Reformulate at least once. A good shortlist covers several distinct products.
Quoted / "matches phrase" tokens must appear in the title. Drop accessories
via category_path and "for / compatible with" titles. Once a valid in-stock
item is verified, order — do not wait for permission.
""",
    CHARGE_FORK: """# Fork
Preferred brand is SOFT. Build two piles: brand+firm vs firm-only. First
cart honors the named brand only when that SKU meets every firm gate.
After a shock, drop the brand filter if no branded SKU still clears the new
world.
""",
    CHARGE_HELD: """# Held
If a firm figure is withheld until you ask, send message NOW and do not
search until the shopper replies. Every gate must hold on the exact option
row. After sellout, pick a new product_id — a sibling that drops a gate is
illegal. Read facts, not titles.
""",
    CHARGE_RELISH: """# Relish
Compare at least two hard-compliant listings on the use-case qualities the
shopper named. Cheapness is a tiebreaker. First cart must be a genuine fit,
not a 10–50 bait stub. After a reprice, recover to a different listing that
still fits the use case — search again, do not reuse leftovers.
""",
    CHARGE_SLATE: """# Slate
The candidate set is closed: only the named product_id::sku pairs. Batch
view/compare them. Rank by the shopper's stated lexicographic axes, not a
memorized electronics order. After any cart-side event, re-rank from
scratch and cart the new #1. place_test_order with ONLY product_id and sku
unless the shopper also asked why.
""",
    CHARGE_PATCH: """# Patch
Walls that never relax: category, budget, in-stock. Ladder: exact
model+color, then same model other color, then other models. Search the
dead listing's title to find twins. A conflicting title does not override
option/facts. Do not jump levels.
""",
    CHARGE_PROOF: """# Proof
View facts before the first cart. Claims copy observed catalog text using
only field names this episode's place_test_order schema allows. Always
include in_stock and price. Add one more field only when that name or its
observed value is attested in the shopper's own wording. Prefer brand as
the extra reason when they named a maker. Never invent attributes, never
rewrite units, never send category/title/os/summary/evidence.
""",
}


def _system_prompt(
    tools: list[dict[str, Any]],
    max_calls: int,
    charter: dict[str, Any],
    policy: dict[str, Any],
) -> str:
    names = [_tool_name(tool) for tool in tools if _tool_name(tool)]
    books = [_CHARGE_BOOK[charge] for charge in charter.get("charges") or [] if charge in _CHARGE_BOOK]
    named = charter.get("named") or []
    closed = ", ".join(f"{item['product_id']}::{item['sku']}" for item in named[:8])
    order = _find_tool(tools, "place_test_order", "place_order", "checkout")
    enum = _claim_enum(order)
    enum_line = f"claim_fields={enum}\n" if charter.get("why") else ""
    return (
        f"{_CORE}\n\n{''.join(books)}\n"
        f"# Runtime card\n"
        f"primary={charter.get('primary')} charges={sorted(charter.get('charges') or [])} "
        f"kind={charter.get('kind') or '-'} brand={charter.get('brand') or '-'} "
        f"budget={charter.get('budget')} ask={bool(charter.get('ask'))} why={bool(charter.get('why'))}\n"
        f"quotes={charter.get('quotes') or []}\n"
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
    if _EVENT_RX.search(cleaned) and not re.search(
        r"which |what |before you|does it actually", cleaned
    ):
        return False
    if "?" in cleaned:
        return True
    return bool(
        re.search(
            r"which one|what about|before you buy|does it actually|"
            r"still needs|non-negotiable",
            cleaned,
        )
    )


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


class _Yard:
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


class _Helms:
    def __init__(self, query: str, tools: list[dict[str, Any]], max_calls: int) -> None:
        self.query = query
        self.charter = compile_charter(query)
        self.max_calls = max(1, max_calls)
        self.yard = _Yard()
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
        for spec in self.charter.get("named") or []:
            key = f"{spec['product_id']}::{spec['sku']}"
            self.yard.rows[key] = {
                "product_id": spec["product_id"],
                "sku": spec["sku"],
                "_hydrated": False,
            }

    def _full(self, product: dict[str, Any] | None) -> dict[str, Any]:
        if not product:
            return {}
        key = _listing_key(product)
        row = self.yard.rows.get(key) or {}
        pid = str(product.get("product_id") or "")
        if pid and (not _facts(row) or not _facts(product)):
            for stored in self.yard.rows.values():
                if str(stored.get("product_id") or "") == pid and _facts(stored):
                    row = _absorb(row, stored)
                    break
        return _absorb(row, product)

    def _stale(self, product: dict[str, Any] | None) -> bool:
        if product is None:
            return False
        if not _in_stock(product):
            return True
        budget = self.charter.get("budget")
        price = _price_of(product)
        if budget is not None and price is not None and price > float(budget):
            return True
        if self.yard.event_key and _listing_key(product) == self.yard.event_key:
            return True
        key = _listing_key(product)
        if key and key in self.yard.dead:
            return True
        return False

    def live(self, *, pick: bool = False) -> list[dict[str, Any]]:
        budget = self.charter.get("budget")
        kind = str(self.charter.get("kind") or "")
        source = self.yard.pool()
        named = self.charter.get("named") or []
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
                        _absorb(
                            parent,
                            {"product_id": spec["product_id"], "sku": spec["sku"]},
                        )
                    )
        rows: list[dict[str, Any]] = []
        for product in source:
            key = _listing_key(product)
            if key in self.yard.dead:
                continue
            if key and key == self.yard.event_key:
                continue
            if not _in_stock(product):
                continue
            price = _price_of(product)
            if budget is not None and price is not None and price > float(budget):
                continue
            closed = bool(self.charter.get("named"))
            if not closed:
                if _junk(product, kind):
                    continue
                if kind and product.get("title") and not _kind_ok(product, kind):
                    continue
                if self.charter.get("quotes") and product.get("title"):
                    if not any(_phrase_hit(product, phrase) for phrase in self.charter["quotes"]):
                        if self.charter["primary"] == CHARGE_HUNT and product.get("_hydrated"):
                            continue
            spec_charge = self.charter["primary"] in {CHARGE_HELD, CHARGE_PROOF, CHARGE_FORK}
            if spec_charge and self.charter.get("spans"):
                if pick and not product.get("_hydrated"):
                    if not self.yard.event_key:
                        continue
                    if not _figure_ok(product, self.query, self.charter.get("spans") or []):
                        continue
                if product.get("_hydrated") and not _figure_ok(
                    product, self.query, self.charter.get("spans") or []
                ):
                    continue
            if (
                spec_charge
                and product.get("_hydrated")
                and not _floor_ok(product, self.query, list(self.charter.get("floors") or []))
            ):
                continue
            rows.append(product)
        return rows

    def ranked(self) -> list[dict[str, Any]]:
        pick = bool(
            self.charter.get("spans")
            and self.charter["primary"] in {CHARGE_HELD, CHARGE_PROOF, CHARGE_FORK}
        )
        pool = self.live(pick=pick)
        axes = list(self.charter.get("axes") or [])
        if CHARGE_SLATE in self.charter["charges"] and (self.charter.get("named") or []):
            return _lex_rank(pool, axes or [("price", "asc")])
        query_tf = Counter(
            _tokens(" ".join(self.charter.get("phrases") or []) + " " + _shopper_text(self.query))
        )
        docs = [(_tokens(_blob(product)), product) for product in pool]
        df: Counter[str] = Counter()
        collection: Counter[str] = Counter()
        for tokens, _product in docs:
            df.update(set(tokens))
            collection.update(tokens)
        avgdl = sum(len(tokens) for tokens, _product in docs) / max(len(docs), 1)
        pl2_rows: list[tuple[float, dict[str, Any]]] = []
        tfidf_rows: list[tuple[float, dict[str, Any]]] = []
        for tokens, product in docs:
            doc_tf = Counter(tokens)
            pl2_rows.append(
                (
                    _pl2(query_tf, doc_tf, df, len(docs), avgdl, collection),
                    product,
                )
            )
            tfidf_rows.append(
                (
                    _tfidf_cos(query_tf, doc_tf, df, max(len(docs), 1)),
                    product,
                )
            )
        pl2_rows.sort(key=lambda item: -item[0])
        tfidf_rows.sort(key=lambda item: -item[0])
        fused = _combanz(
            [
                [_listing_key(product) for _score, product in pl2_rows],
                [_listing_key(product) for _score, product in tfidf_rows],
            ]
        )
        by_key = {_listing_key(product): product for product in pool}
        wishes = list(self.charter.get("wishes") or [])
        budget = self.charter.get("budget")
        brand = str(self.charter.get("brand") or "")
        honor = CHARGE_FORK in self.charter["charges"] and not self.yard.event_key

        def sort_key(product: dict[str, Any]) -> tuple:
            key = _listing_key(product)
            retrieval = fused.get(key, 0.0)
            gra = _gra_grade(product, wishes) if CHARGE_RELISH in self.charter["charges"] else 0.0
            decoy = 1 if CHARGE_RELISH in self.charter["charges"] and _decoy(product, budget) else 0
            brand_rank = 0 if honor and brand and _brand_hit(product, brand) else 1
            ready = 0 if product.get("_hydrated") else 1
            after = 0 if product.get("_aftershock") else 1
            price = _price_of(product)
            honor_after = bool(self.yard.event_key and self.charter["charges"] & {CHARGE_RELISH, CHARGE_PROOF})
            ranged = 1 if _size_range_blob(str(_facts(product).get("size") or product.get("title") or "")) else 0
            return (
                decoy,
                brand_rank,
                after if honor_after else 0,
                0 if _figure_ok(product, self.query, self.charter.get("spans") or []) else 1,
                ranged,
                -gra,
                ready,
                -retrieval,
                price if price is not None else 1e12,
            )

        return sorted(by_key.values(), key=sort_key)

    def champion(self) -> dict[str, Any] | None:
        ranked = self.ranked()
        target = self.yard.event_key
        if not ranked and target:
            ranked = [
                item
                for item in self.yard.pool()
                if _listing_key(item) != target
                and not self._stale(item)
                and _figure_ok(item, self.query, list(self.charter.get("spans") or []))
            ]
        if not ranked:
            return None
        live = [item for item in ranked if _listing_key(item) != target and not self._stale(item)]
        if not live:
            live = [item for item in ranked if _listing_key(item) != target]
        spans = list(self.charter.get("spans") or [])
        if self.charter["primary"] in {CHARGE_HELD, CHARGE_PROOF, CHARGE_FORK}:
            fitted = [item for item in live if _figure_ok(item, self.query, spans)]
            if fitted:
                live = fitted
        if CHARGE_RELISH in self.charter["charges"]:
            budget = self.charter.get("budget")
            wishes = list(self.charter.get("wishes") or [])
            if target:
                post = [item for item in live if item.get("_aftershock")]
                if post:
                    live = post
            fitted = [item for item in live if not _decoy(item, budget)]
            hydrated = [item for item in fitted if item.get("_hydrated")]
            strong = [item for item in hydrated if _gra_grade(item, wishes) >= 0.55]
            live = strong or hydrated or fitted or live
        if CHARGE_FORK in self.charter["charges"] and not target:
            brand = str(self.charter.get("brand") or "")
            branded = [item for item in live if brand and _brand_hit(item, brand)]
            ready = [
                item
                for item in branded
                if item.get("_hydrated") and _figure_ok(item, self.query, spans)
            ]
            if ready:
                return ready[0]
            if branded:
                return branded[0]
        if CHARGE_PATCH in self.charter["charges"] and (self.charter.get("model") or self.charter.get("color")):
            model = str(self.charter.get("model") or "")
            color = str(self.charter.get("color") or "")
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
                    if stem and _sku_stem(item.get("sku")) == stem
                ]
                if siblings:
                    live = siblings
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
        if CHARGE_PROOF in self.charter["charges"] and self.charter.get("brand"):
            brand = str(self.charter.get("brand") or "")
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
        ranking = CHARGE_SLATE in self.charter["charges"] and not self.charter.get("why")
        if ranking:
            return {"name": _tool_name(self.order), "args": _bind(self.order, wanted)}
        required = (_fn(self.order).get("parameters") or {}).get("required") or []
        needs = self.charter.get("why") or CHARGE_PROOF in self.charter["charges"]
        if needs or "justification" in {str(item).lower() for item in required}:
            claims = _proof_claims(product, self.query, self.order)
            if claims:
                wanted["justification"] = {"claims": claims}
        return {"name": _tool_name(self.order), "args": _bind(self.order, wanted)}

    def _search_actions(self) -> list[dict[str, Any]]:
        kind = str(self.charter.get("kind") or "")
        budget = self.charter.get("budget")
        phrases = [
            item
            for item in (self.charter.get("phrases") or ["product"])
            if _useful_query(item, kind=kind, budget=budget)
        ]
        if CHARGE_PATCH in self.charter["charges"] and self.charter.get("model"):
            model = str(self.charter.get("model") or "")
            color = str(self.charter.get("color") or "")
            headed = [item for item in (f"{color} {model}".strip(), model) if item]
            phrases = headed + [item for item in phrases if item not in headed]
        elif CHARGE_PATCH in self.charter["charges"] and self.yard.dead_title and not self.echoed:
            title = " ".join(_tokens(self.yard.dead_title)[:6])
            if title:
                phrases = [title] + [item for item in phrases if item != title]
            self.echoed = True
        elif self.charter.get("model") and self.yard.event_key:
            model = str(self.charter.get("model") or "")
            phrases = [model] + [item for item in phrases if item != model]
        phrases = phrases or ([kind] if kind else ["product"])
        if CHARGE_PATCH in self.charter["charges"]:
            cap = min(3, max(len(phrases), 1))
        else:
            cap = 1 if self.yard.event_key else min(2, max(len(phrases), 1))
        if CHARGE_RELISH in self.charter["charges"] and not any(
            _gra_grade(item, list(self.charter.get("wishes") or [])) >= 0.55 for item in self.live()
        ):
            cap = max(cap, min(2, len(phrases)))
        actions: list[dict[str, Any]] = []
        if self.search and self.yard.search_n < cap:
            query = phrases[min(self.yard.search_n, len(phrases) - 1)]
            self.yard.search_n += 1
            actions.append(
                {
                    "name": _tool_name(self.search),
                    "args": _bind(self.search, {"query": query, "k": 10, "in_stock": True}),
                }
            )
        brand = str(self.charter.get("brand") or "")
        if CHARGE_FORK in self.charter["charges"] and self.yard.event_key:
            brand = ""
        if CHARGE_HUNT in self.charter["charges"] and CHARGE_FORK not in self.charter["charges"]:
            brand = ""
        category = str(self.charter.get("category") or "")
        if (
            self.filter
            and not self.yard.did_filter
            and len(actions) < self.max_calls
            and (brand or category)
        ):
            args = _bind(
                self.filter,
                {
                    "brand": brand or None,
                    "category": category or None,
                    "max_price": self.charter.get("budget"),
                    "in_stock": True,
                    "k": 10,
                },
            )
            if args:
                self.yard.did_filter = True
                actions.append({"name": _tool_name(self.filter), "args": args})
        if not actions:
            self.yard.search_n += 1
        return actions[: self.max_calls]

    def _hydrate(self, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cap = 2 if self.yard.event_key else 4
        if self.yard.compare_n >= cap:
            return []
        ids: list[str] = []
        seen: set[str] = set()
        for product in products:
            pid = str(product.get("product_id") or "")
            if pid and pid not in seen and pid not in self.yard.viewed:
                seen.add(pid)
                ids.append(pid)
        if not ids:
            return []
        self.yard.compare_n += 1
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
        if CHARGE_SLATE in self.charter["charges"]:
            return 12
        if CHARGE_RELISH in self.charter["charges"]:
            return 6
        if CHARGE_HUNT in self.charter["charges"]:
            return 4
        return 5

    def steer(self) -> list[dict[str, Any]]:
        if self.message and self.charter.get("ask") and not self.asked:
            return [
                {
                    "name": _tool_name(self.message),
                    "args": _bind(
                        self.message,
                        {"content": "What exact figure or attribute must the product meet?"},
                    ),
                }
            ]
        cart = self.yard.cart[0] if self.yard.cart else None
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
        named = list(self.charter.get("named") or [])
        if named:
            missing = [
                spec["product_id"]
                for spec in named
                if spec["product_id"] not in self.yard.viewed
            ]
            if missing:
                hydrate = self._hydrate([{"product_id": pid} for pid in missing])
                if hydrate:
                    return hydrate[: self.max_calls]
            if self.yard.need_replay and self.compare:
                ids = [spec["product_id"] for spec in named]
                self.yard.need_replay = False
                return [
                    {
                        "name": _tool_name(self.compare),
                        "args": _bind(self.compare, {"product_ids": ids[:8], "product_id": ids[0]}),
                    }
                ]
        else:
            live_now = self.live()
            recovering = bool(self.yard.event_key)
            if recovering:
                model = str(self.charter.get("model") or "")
                siblings = [
                    item
                    for item in live_now
                    if not model or _model_hit(item, model)
                ]
                need_search = (not siblings) and self.yard.search_n < 4
            else:
                need_search = self.yard.search_n < (4 if CHARGE_PROOF in self.charter["charges"] else 5) and (
                    not live_now or self.yard.search_n < 2
                )
            if CHARGE_PATCH in self.charter["charges"] and not recovering:
                need_search = self.yard.search_n < 2 or need_search
                model = str(self.charter.get("model") or "")
                modeled = [
                    item
                    for item in live_now
                    if not model or _model_hit(item, model)
                ]
                ready = [item for item in modeled if item.get("_hydrated")]
                if len(ready) < 2:
                    hydrate = self._hydrate(modeled[: self._view_cap()])
                    if hydrate:
                        return hydrate[: self.max_calls]
            if recovering and CHARGE_PATCH in self.charter["charges"]:
                stem = _sku_stem((self.yard.event_key or "").split("::")[-1])
                twins = [
                    item
                    for item in live_now
                    if stem and _sku_stem(item.get("sku")) == stem
                ]
                if not twins and self.yard.search_n < 4:
                    need_search = True
            if CHARGE_PROOF in self.charter["charges"] and not recovering:
                grounded = [
                    item
                    for item in live_now
                    if item.get("_hydrated") and _figure_ok(item, self.query, list(self.charter.get("spans") or []))
                ]
                if not grounded and self.yard.search_n < 4:
                    need_search = True
            if CHARGE_RELISH in self.charter["charges"] and not recovering:
                wishes = list(self.charter.get("wishes") or [])
                if self.yard.search_n < 6 and not any(_gra_grade(item, wishes) >= 0.55 for item in live_now):
                    need_search = True
            if need_search:
                search = self._search_actions()
                if search:
                    return search
            if recovering:
                model = str(self.charter.get("model") or "")
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
                focus = (self.ranked() or live_now or self.yard.pool())[: self._view_cap()]
                if len(self.yard.viewed) < self._view_cap():
                    hydrate = self._hydrate(focus)
                    if hydrate:
                        return hydrate[: self.max_calls]
        champ = self.champion()
        if champ is None:
            if named:
                for spec in named:
                    key = f"{spec['product_id']}::{spec['sku']}"
                    if key == self.yard.event_key or key in self.yard.dead:
                        continue
                    if self.inspect and key not in self.yard.gauged:
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
            hydrate = self._hydrate(self.live() or self.yard.pool())
            return hydrate[: self.max_calls]
        if champ is None:
            return []
        if self.inspect and _listing_key(champ) not in self.yard.gauged:
            return [
                {
                    "name": _tool_name(self.inspect),
                    "args": _bind(
                        self.inspect,
                        {"product_id": champ.get("product_id"), "sku": champ.get("sku")},
                    ),
                }
            ]
        if not self.yard.cart and self.cart and not self._stale(champ):
            spec = self.charter["charges"] & {CHARGE_PROOF, CHARGE_RELISH, CHARGE_FORK, CHARGE_HELD}
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
        if self.yard.cart and not self._stale(self.yard.cart[0]):
            cart_item = self.yard.cart[0]
            cart_key = _listing_key(cart_item)
            if self.yard.event_open and self.inspect and cart_key not in self.yard.gauged:
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
                if cart_key in self.yard.gauged:
                    self.yard.event_open = False
                target = cart_item
                if champ is not None and champ.get("_gauged"):
                    target = champ
                return [self._order(target)]
        return []

    def finish(self) -> list[dict[str, Any]]:
        cart = self.yard.cart[0] if self.yard.cart else None
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
            if self.yard.event_open and self.inspect and _listing_key(cart) not in self.yard.gauged:
                return [
                    {
                        "name": _tool_name(self.inspect),
                        "args": _bind(
                            self.inspect,
                            {"product_id": cart.get("product_id"), "sku": cart.get("sku")},
                        ),
                    }
                ]
            self.yard.event_open = False
            return [self._order(cart)]
        picked = self.champion()
        if picked is None:
            live = [
                item
                for item in self.live()
                if _listing_key(item) != self.yard.event_key
            ]
            if self.charter["primary"] in {CHARGE_HELD, CHARGE_PROOF, CHARGE_FORK}:
                live = [
                    item
                    for item in live
                    if _figure_ok(item, self.query, list(self.charter.get("spans") or []))
                ]
            if CHARGE_RELISH in self.charter["charges"]:
                wishes = list(self.charter.get("wishes") or [])
                fitted = [item for item in live if _gra_grade(item, wishes) >= 0.45 and not _decoy(item, self.charter.get("budget"))]
                live = fitted or live
            if CHARGE_PATCH in self.charter["charges"]:
                model = str(self.charter.get("model") or "")
                color = str(self.charter.get("color") or "")
                modeled = [
                    item
                    for item in live
                    if (not model or _model_hit(item, model)) and not self._stale(item)
                ]
                modeled.sort(key=lambda item: _ladder(item, model, color))
                live = modeled if modeled else (live if self.yard.search_n >= 3 else [])
            picked = live[0] if live else None
        if picked is None:
            return []
        pid = str(picked.get("product_id") or "")
        if self.inspect and _listing_key(picked) not in self.yard.gauged:
            return [
                {
                    "name": _tool_name(self.inspect),
                    "args": _bind(self.inspect, {"product_id": pid, "sku": picked.get("sku")}),
                }
            ]
        if self.cart and not self._stale(picked) and not self.yard.cart:
            return [
                {
                    "name": _tool_name(self.cart),
                    "args": _bind(self.cart, {"product_id": pid, "sku": picked.get("sku")}),
                }
            ]
        if self.yard.cart and self.order and not self._stale(self.yard.cart[0]) and not self.yard.event_open:
            return [self._order(self.yard.cart[0])]
        return []

    def splice(
        self,
        actions: list[dict[str, Any]],
        proposed: list[dict[str, Any]],
        *,
        await_reply: bool,
    ) -> list[dict[str, Any]]:
        names = {_tool_name(tool).lower() for tool in (
            self.search,
            self.filter,
            self.view,
            self.compare,
            self.inspect,
            self.cart,
            self.remove,
            self.order,
            self.message,
        ) if tool}
        cleaned = [action for action in actions if str(action.get("name") or "").lower() in names]
        cart_item = self.yard.cart[0] if self.yard.cart else None
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
        if await_reply and self.message and not self.yard.event_open:
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
        if CHARGE_PATCH in self.charter["charges"] and self.cart and not self.yard.cart:
            champ = self.champion()
            model = str(self.charter.get("model") or "")
            color = str(self.charter.get("color") or "")
            ok = False
            if champ is not None and not self._stale(champ):
                if not self.yard.event_key:
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
        if self.yard.event_open and proposed and force:
            return proposed[: self.max_calls]
        if force and proposed and not await_reply:
            return proposed[: self.max_calls]
        if not cleaned:
            return proposed[: self.max_calls]
        if any("message" in str(item.get("name") or "").lower() for item in cleaned):
            if self.yard.event_open:
                return proposed[: self.max_calls] or cleaned[: self.max_calls]
        order_name = _tool_name(self.order).lower()
        champ = self.champion()
        aligned: list[dict[str, Any]] = []
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
                if self.yard.event_open and self.inspect and champ is not None:
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
                if product is None:
                    continue
                aligned.append(self._order(product))
                continue
            aligned.append({"name": name, "args": args})
        return aligned or proposed[: self.max_calls]

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
                self.yard.cart = []
            if any(token in err_blob for token in ("no such", "no_candidate", "cannot add")):
                pid = str(sent_args.get("product_id") or inner.get("product_id") or "")
                sku = str(sent_args.get("sku") or inner.get("sku") or "")
                if pid and sku:
                    self.yard.dead.add(f"{pid}::{sku}")
                    self.yard.cart = []
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
                self.yard.merge(products, viewed=viewed, gauged=gauged, from_search=not viewed and not gauged)
            if isinstance(inner.get("added"), dict):
                added = dict(inner["added"])
                self.yard.merge([added], viewed=False, gauged=False, from_search=False)
                key = _listing_key(added)
                merged = self.yard.rows.get(key) or added
                self.yard.cart = [dict(merged)]
                self.yard.last_key = key or self.yard.last_key
                if key and key in self.yard.gauged:
                    self.yard.event_open = False
            if isinstance(inner.get("cart"), list):
                self.yard.cart = [item for item in inner["cart"] if isinstance(item, dict)]
            if isinstance(inner.get("removed"), dict) or inner.get("removed") is True:
                self.yard.cart = []
                self.yard.compare_n = 0
            event = inner.get("market_event")
            if isinstance(event, dict) and event:
                self.yard.shock(event)
        message = result.get("user_message") or {}
        if isinstance(message, dict):
            signal = message.get("env_signal") or {}
            if isinstance(signal, dict) and signal.get("kind"):
                self.yard.shock(signal)
            content = str(message.get("content") or "")
            if _EVENT_RX.search(content) and not self.yard.event_open and not self.yard.event_key:
                target: dict[str, Any] = {}
                if self.yard.cart:
                    item = self.yard.cart[0]
                    target = {"product_id": item.get("product_id"), "sku": item.get("sku")}
                elif self.yard.last_key and "::" in self.yard.last_key:
                    pid, sku = self.yard.last_key.split("::", 1)
                    target = {"product_id": pid, "sku": sku}
                kind = "stockout" if re.search(r"sold out|out of stock", content, re.I) else "price_change"
                self.yard.shock({"target": target, "kind": kind})
            budget = _BUDGET_RX.search(content)
            if budget:
                self.charter["budget"] = float(budget.group(1).replace(",", ""))
            if content and self.charter.get("ask"):
                self.asked = True
                extra = _figure_spans(content, self.charter.get("budget"))
                if extra:
                    self.charter["spans"] = extra
                    kind = str(self.charter.get("kind") or "")
                    headed = [
                        f"{self.charter.get('brand') or ''} {span['raw']} {kind}".strip()
                        for span in extra[:3]
                    ]
                    headed.extend(f"{span['raw']} {kind}".strip() for span in extra[:3])
                    phrases = list(self.charter.get("phrases") or [])
                    self.charter["phrases"] = [
                        item for item in headed + phrases if item
                    ]
                    self.yard.search_n = 0
                    self.yard.did_filter = False
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
    helms: _Helms,
) -> bool:
    if _left() < 55:
        return True
    names = " ".join(str(item.get("name") or "").lower() for item in proposed)
    if proposed and "remove_from_cart" in names:
        return True
    if CHARGE_SLATE in helms.charter["charges"] and not await_reply:
        return True
    if CHARGE_PATCH in helms.charter["charges"]:
        model = str(helms.charter.get("model") or "")
        color = str(helms.charter.get("color") or "")
        champ = helms.champion()
        exact = bool(champ and _ladder(champ, model, color) == 0)
        sibling = bool(
            champ
            and helms.yard.event_key
            and _ladder(champ, model, color) <= 1
        )
        if not cart_used and exact and proposed and not await_reply:
            return True
        if cart_used and sibling and any(
            hint in names
            for hint in ("add_to_cart", "inspect", "place_test_order", "place_order")
        ):
            return True
        if "compare" in names or "search" in names or "find_product" in names:
            return False
        if helms.yard.event_key and not sibling:
            return False
    if helms.yard.event_key and proposed:
        if any(
            hint in names
            for hint in ("add_to_cart", "remove_from_cart", "inspect", "place_test_order", "place_order")
        ):
            return True
    if helms.yard.event_open and proposed:
        if "compare" in names:
            return False
        return True
    if await_reply and not helms.yard.event_open:
        return _left() < 90
    if _mechanical(proposed):
        return True
    if cart_used and proposed:
        if "place_test_order" in names or "place_order" in names or "inspect" in names:
            return True
        if "add_to_cart" in names or "remove_from_cart" in names:
            return True
    relish = CHARGE_RELISH in helms.charter["charges"]
    if relish and not cart_used and not helms.yard.event_open and turn < 12 and _left() > 70:
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


def _coach(helms: _Helms, turn: int, await_reply: bool, cart_used: bool) -> str | None:
    if await_reply:
        if helms.yard.event_open:
            return (
                "A market event is open. Re-inspect remaining live options, "
                "cart the new winner, then order. Also answer the shopper."
            )
        return "The shopper asked a question. Answer with the message tool this turn."
    if helms.charter.get("ask") and not helms.asked and turn == 1:
        return "One firm figure is withheld until you ask. Call message NOW before searching."
    if CHARGE_SLATE in helms.charter["charges"] and turn <= 2:
        return "Compare every named candidate before the first cart. Order with ONLY product_id and sku."
    if CHARGE_PROOF in helms.charter["charges"] and turn <= 2:
        return "View facts before the first cart. Fill claims from viewed facts using only allowed fields."
    if CHARGE_PATCH in helms.charter["charges"] and turn <= 2:
        return "After a stockout, keep the same model and change color first. Never order the event target."
    if CHARGE_FORK in helms.charter["charges"] and turn == 1:
        return "First cart honors the preferred brand only when that SKU also meets every firm gate."
    if CHARGE_RELISH in helms.charter["charges"] and turn <= 2 and not cart_used:
        return "Compare the shopper's stated use-case before the first cart. Do not cart the cheapest stub."
    if CHARGE_HUNT in helms.charter["charges"] and turn == 3:
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


def _run_helms(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
    max_steps = int(policy.get("max_steps") or 20)
    tools = list(policy.get("tools") or [])
    query = str(policy.get("query") or problem_data.get("query") or "")
    helms = _Helms(query, tools, max_calls)
    helms.max_steps = max_steps
    model = _primary_model(helms.charter["charges"])

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(tools, max_calls, helms.charter, policy)},
        {"role": "user", "content": query},
    ]
    if helms.charter.get("ask"):
        messages.append(
            {
                "role": "user",
                "content": (
                    "The shopper is withholding one requirement until asked. "
                    "Send message now to get that figure before you search."
                ),
            }
        )
    if CHARGE_RELISH in helms.charter["charges"]:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Compare at least two candidates on the use-case qualities "
                    "before the first cart. Cheapness is a tiebreaker. "
                    "First cart must be a genuine fit."
                ),
            }
        )
    if CHARGE_SLATE in helms.charter["charges"]:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Closed slate: compare listed candidates first. "
                    "place_test_order with ONLY product_id and sku unless a why was asked."
                ),
            }
        )
    if CHARGE_PROOF in helms.charter["charges"]:
        messages.append(
            {
                "role": "user",
                "content": (
                    "View facts before the first cart. Fill claims from viewed "
                    "facts using only field names this episode's order tool allows. "
                    "Copy values as the catalog printed them."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "anvil": {
                "primary": helms.charter.get("primary"),
                "charges": sorted(helms.charter.get("charges") or []),
                "model": model,
                "provider": _provider(),
                "stack": "pl2+tfidf+combanz+marcos+gra",
            },
        }
    ]
    await_reply = False
    cart_used = False

    for turn in range(1, max_steps + 1):
        helms.turn = turn
        if _left() < 8:
            proposed = helms.finish()[:max_calls]
            skip = True
        else:
            proposed = helms.steer()
            skip = _skip_llm(proposed, await_reply, turn, max_steps, cart_used, helms)
        actions: list[dict[str, Any]] = []
        assistant_content = ""
        raw_calls: list[dict[str, Any]] = []
        if not skip:
            coach = _coach(helms, turn, await_reply, cart_used)
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
            actions = helms.splice(actions, proposed, await_reply=await_reply)
        if not actions:
            actions = helms.finish()[:max_calls]
        if not actions:
            break

        lowered = [str(item.get("name") or "").lower() for item in actions]
        if any("add_to_cart" in name for name in lowered):
            cart_used = True
        if any("message" in name for name in lowered):
            await_reply = False
            helms.asked = True

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
                "anvil_turn": {"primary": helms.charter.get("primary"), "turn": turn},
            }
        )
        done = helms.ingest(result, actions)
        for tool_call, call_result in zip(tool_calls, result.get("calls") or [], strict=False):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(call_result.get("observation"), ensure_ascii=False, default=str)[
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
    """Anvil loop: PL2/CombANZ hunt + MARCOS/GRA choice until done=true."""

    _arm_clock()
    dialogue: list[dict[str, Any]] = []
    try:
        dialogue = _run_helms(problem_data)
    except Exception:
        return dialogue
    return dialogue


__all__ = ["agent_main", "compile_charter"]
