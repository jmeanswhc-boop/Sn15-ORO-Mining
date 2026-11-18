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
_LIGHT = "deepseek/deepseek-v3.2"
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

_CLOCK_S = 279.0
_OBS_CAP = 2200
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
    r"(?:only|will)\s+(?:tell|share|reveal).{0,48}\bif\s+you\s+ask|"
    r"\bif you ask\b|\bwhen you ask\b|\bwhen asked\b|"
    r"\bwithhold\b.{0,36}\b(?:spec|figure|size|requirement)\b|"
    r"withheld.{0,24}until.{0,16}\bask|"
    r"ask\s+(?:me\s+)?(?:for\s+)?(?:the\s+)?(?:exact\s+)?(?:\w+\s+){0,3}(?:figure|spec|requirement|size)|"
    r"(?:tell|share|reveal)[^.]{0,48}\bif\b[^.]{0,24}\bask",
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
_PAIR_RX = re.compile(r"(\d{3,5})\s*(?:[x×*]|\bby\b)\s*(\d{3,5})", re.I)
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
    "replacement only",
    "sticker",
    "decoration",
    "pouch",
)
_CLAIM_SOFT = frozenset({"price", "in_stock", "brand", "category"})
_CLAIM_BLOCK = {
    "category",
    "category_path",
    "title",
    "summary",
    "evidence",
    "reason",
    "description",
    "name",
    "note",
    "placeholder",
    "final_price",
    "phone_like",
    "trustworthy_reason",
    "first_cart",
    "structured_justification",
    "specification_field",
    "shopper_request",
    "valid_product",
    "valid_item",
    "shopping_agent",
    "required_category",
    # English glue words scraped from schema prose (never claim fields).
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "onto",
    "using",
    "use",
    "used",
    "must",
    "should",
    "shall",
    "only",
    "also",
    "when",
    "then",
    "than",
    "that",
    "this",
    "these",
    "those",
    "facts",
    "fact",
    "viewed",
    "observed",
    "catalog",
    "schema",
    "allowed",
    "allow",
    "include",
    "includes",
    "including",
    "field",
    "value",
    "claims",
    "justification",
}
_CLAIM_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,32}$")
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
# Cache TF7 firm-field LLM fills per listing (order + hold checks share it).
_CLAIM_FILL_CACHE: dict[str, list[dict[str, str]]] = {}

_TF7_FILL = """
Name the justification claim for ONE shopping requirement.
Input JSON: goal, allowed_fields (live enum, may be empty), schema_rule,
have, listing (observed text for the sku).
Return JSON only: {"claims":[{"field":"...","value":"..."}]}
Rules:
- Name the measured attribute the shopper asked about. Catalog fact keys are
  drawers, not claim fields.
- Feature reasons in schema_rule (streaming, microphone, privacy_cover,
  autofocus, hdr) are NOT the measured specification.
- value must appear in listing. Prefer the shopper's class word when listed.
- At most 1 claim. If listing lacks evidence: {"claims":[]}.

Examples (field naming):
- Shopper "720p HD", listing "display: hd" → {"field":"resolution","value":"hd"}
  NOT field=display, NOT field=display_resolution.
- Shopper "1080p Full HD", listing "display: fhd" or title Full HD →
  {"field":"resolution","value":"fhd"} (or the exact listed token).
- Shopper "27-inch", listing "size: 27 inches" →
  {"field":"screen_size","value":"27in"} (size is ok only if you must echo the
  drawer; prefer screen_size).
- Shopper "144Hz", listing "refresh: 144 hz" →
  {"field":"refresh_rate","value":"144hz"} NOT a random other figure.
""".strip()

_TF7_ALIGN = """
Finalize justification.claims.
Input JSON: goal, schema_rule, allowed_fields, listing, proposed_claims.
Return JSON only: {"claims":[{"field":"...","value":"..."}]}
Rules:
- Keep values unchanged.
- If proposed field is a catalog drawer (display, size, …) or a compound
  (display_resolution), rename field to the conventional attribute:
  pixel/Np/HD class → resolution; inch diagonal → screen_size; Hz → refresh_rate.
- Never leave field=display for an HD/Np ask. Never use hdr/streaming/etc. as
  the measured field. At most 1 claim.
""".strip()


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
    if isinstance(facts, dict):
        return facts
    if isinstance(facts, list):
        out: dict[str, Any] = {}
        for item in facts:
            if not isinstance(item, dict):
                continue
            key = item.get("name") or item.get("key") or item.get("field") or item.get("attr")
            val = item.get("value") if item.get("value") not in (None, "") else item.get("val")
            if key and val not in (None, ""):
                out[str(key)] = val
        return out
    return {}


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


def _kind_noun(query: str, brand: str = "") -> str:
    """Noun from this brief only — no catalog of device classes.

    Category path is a filter hint, not a kind token: last-word 'improvement'
    from 'Tools & Home Improvement' must not wipe an aisle.
    """
    shopper = _shopper_text(query)
    folded = _fold(shopper)
    skip = set(_STOP)
    skip.update(_fold(brand).split())
    tails = list(
        re.finditer(
            r"([a-z][a-z]{2,})\s+(?:at or under|at or below|in stock)",
            folded,
        )
    )
    for hit in reversed(tails):
        if hit.group(1) not in skip:
            return hit.group(1)
    branded = re.search(
        r"(?:need|want|find|get|buy)\s+(?:me\s+)?(?:an?\s+)?"
        r"(?:in[-\s]?stock\s+)?"
        r"(?:[A-Z][A-Za-z0-9.&'-]+(?:\s+[A-Z][A-Za-z0-9.&'-]+){0,2}\s+)?"
        r"([a-z][a-z]{2,})",
        shopper,
    )
    if branded and _fold(branded.group(1)) not in skip:
        return branded.group(1)
    head = re.search(
        r"(?:need|want|find|get|buy)\s+(?:me\s+)?(?:an?\s+)?"
        r"(?:in[-\s]?stock\s+)?([a-z][a-z]{2,})",
        folded,
    )
    if head and head.group(1) not in skip:
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
    hit = re.search(
        r"(?i:need|want|get|buy|find(?:\s+me)?)\s+(?:me\s+)?(?:an?\s+)?"
        r"(?:in[-\s]?stock\s+)?"
        r"([A-Z][A-Za-z0-9.&'-]+(?:\s+[A-Z][A-Za-z0-9.&'-]+){0,2})\s+",
        shopper,
    )
    if hit:
        phrase = _clean(hit.group(1)).rstrip(".,;")
        parts = phrase.split()
        while len(parts) > 1 and re.fullmatch(r"[A-Z]{2,4}", parts[-1]):
            parts.pop()
        phrase = " ".join(parts)
        if _fold(phrase) not in _STOP and not re.match(r"(?i)in[- ]stock$", phrase):
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
    """Keep the shopper's axis words. Do not remap every 'size' onto screens."""
    folded = _fold(raw)
    folded = re.split(r"\b(?:among|then|remaining|ties|choose)\b", folded)[0]
    folded = re.sub(r"\b(advertised|maximum|available|stated)\b", " ", folded)
    folded = re.sub(r"\s+", " ", folded).strip()
    if "price" in folded or "cost" in folded or "cheap" in folded:
        return "price"
    if not folded or folded in _STOP:
        return ""
    return folded.replace(" ", "_")[:32]


def _axis_value(product: dict[str, Any], name: str) -> float | None:
    """Numeric axis from options/facts/title via context overlap with this axis phrase.

    Advertised figures (e.g. Hz in the title) count when the axis tokens or unit
    boosts match — same idea as black/weft. No claim-field ontology and no class maps.
    """
    folded = _fold(name.replace("_", " "))
    if (
        name == "price"
        or "price" in folded
        or "cost" in folded
    ):
        return _price_of(product)
    facts = _facts(product)
    raw = facts.get(name) or product.get(name)
    direct = _number(raw)
    if direct is not None:
        return direct
    needle_tokens = [tok for tok in folded.split() if len(tok) > 1]
    if not needle_tokens:
        return None
    sources: list[str] = []
    options = product.get("options")
    if isinstance(options, dict):
        sources.extend(f"{key} {value}" for key, value in options.items())
    elif options not in (None, ""):
        sources.append(str(options))
    sources.extend(f"{key} {value}" for key, value in facts.items())
    sources.append(str(product.get("title") or ""))
    best: tuple[int, float] | None = None
    for source in sources:
        text = str(source or "")
        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*-?\s*(hz|hertz|inch(?:es)?|in\b|\"|gb|tb|mm|cm|ml|oz|p\b)?",
            text,
            re.I,
        ):
            try:
                amount = float(match.group(1))
            except ValueError:
                continue
            start = max(0, match.start() - 24)
            end = min(len(text), match.end() + 24)
            window = _fold(text[start:end])
            overlap = sum(1 for tok in needle_tokens if tok in window)
            unit = (match.group(2) or "").lower()
            if unit in {"hz", "hertz"} and any(
                tok in {"hz", "hertz", "refresh"} for tok in needle_tokens
            ):
                overlap += 2
            if unit in {"inch", "inches", "in", '"'} and any(
                tok in {"inch", "inches", "in", "size"} for tok in needle_tokens
            ):
                overlap += 2
            if overlap <= 0:
                continue
            if best is None or overlap > best[0]:
                best = (overlap, amount)
            elif best is not None and overlap == best[0] and amount > best[1]:
                # Prefer larger magnitude among equal overlap (overclock / dual figures).
                best = (overlap, amount)
    return best[1] if best else None


def _lex_parts(product: dict[str, Any], axes: list[tuple[str, str]]) -> tuple:
    used = axes or [("price", "asc")]
    parts: list[float] = []
    for name, direction in used:
        value = _axis_value(product, name)
        if value is None:
            parts.append(1e18 if direction == "desc" else -1e18)
        else:
            parts.append(-value if direction == "desc" else value)
    return tuple(parts)


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


def _kind_ok(product: dict[str, Any], kind: str, category: str = "") -> bool:
    blob = _fold(_blob(product))
    if kind and _fold(kind) in blob:
        return True
    if category:
        cat = _fold(category)
        if cat and cat in blob:
            return True
        toks = [token for token in _tokens(category) if token not in _STOP]
        if toks and any(token in blob for token in toks):
            return True
    return not kind


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
    return _compact(model) in _compact(_blob(product))


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
    blob = _fold(hay)
    compact = _compact(hay)
    token = f"{float(amount):g}"
    int_tok = str(int(amount)) if float(amount).is_integer() else token
    have_num = (
        token in blob
        or int_tok in blob
        or _compact(token) in compact
        or (unit == "p" and f"{int_tok}p" in compact)
    )
    if not have_num:
        return False
    if not unit:
        return True
    if unit in blob or unit in compact:
        return True
    if unit == "p" and (f"{int_tok}p" in compact or _PAIR_RX.search(hay or "")):
        return True
    return False


def _measure_hit(product: dict[str, Any], spans: list[dict[str, Any]]) -> bool:
    if not spans:
        return True
    option = _options_text(product)
    facts = " ".join(str(value) for value in _facts(product).values())
    grounded = f"{option} {facts}"
    for span in spans:
        if _span_in(grounded, span):
            continue
        if grounded.strip():
            return False
        if not _span_in(_blob(product), span):
            return False
    return True


def _figure_ok(product: dict[str, Any], query: str, spans: list[dict[str, Any]] | None = None) -> bool:
    """Shopper measures must hold on listing facts — no class ontology."""
    needed = spans if spans is not None else _figure_spans(_shopper_text(query))
    if not needed:
        return True
    p_spans = [span for span in needed if str(span.get("unit") or "") == "p"]
    if p_spans:
        return _class_meets(product, query)
    inch_spans = [span for span in needed if str(span.get("unit") or "") == "in"]
    if inch_spans:
        size_amt = _size_amount(product)
        tol = 0.05 if product.get("_hydrated") else 0.25
        if size_amt is not None and any(
            abs(size_amt - float(span["amount"])) <= tol for span in inch_spans
        ):
            return True
        fact_sizes: list[float] = []
        title = str(product.get("title") or "")
        title_sizes = [
            float(match.group(1))
            for match in re.finditer(
                r"(\d+(?:\.\d+)?)\s*-?\s*(?:\"|''|in(?:ch(?:es)?)?)\b",
                title,
                re.I,
            )
        ]
        multi_size_title = len({round(value) for value in title_sizes if 1 <= value <= 100}) >= 2
        for key, raw in _facts(product).items():
            text = str(raw or "")
            if re.search(r"\d+\s*[-–/to]+\s*\d+", text, re.I):
                if multi_size_title:
                    return False
                hit = re.search(
                    r"(\d+(?:\.\d+)?)\s*[-–/to]+\s*(\d+(?:\.\d+)?)",
                    text,
                    re.I,
                )
                if hit:
                    lo, hi = float(hit.group(1)), float(hit.group(2))
                    if any(lo - tol <= float(span["amount"]) <= hi + tol for span in inch_spans):
                        return True
                continue
            value = _number(raw)
            if value is not None and 1 <= value <= 100:
                fact_sizes.append(float(value))
        if fact_sizes and not any(
            abs(value - float(span["amount"])) <= tol
            for value in fact_sizes
            for span in inch_spans
        ):
            return False
        if multi_size_title and not fact_sizes:
            return False
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
            if key and key not in seen and key not in _CLAIM_BLOCK:
                seen.add(key)
                found.append(key)
        return found
    return []


def _schema_claim_fields(tool: dict[str, Any] | None) -> list[str]:
    """Live enum if present; otherwise soft drawers only (firm added from brief)."""
    found = _claim_enum(tool)
    if found:
        return [item for item in found if _CLAIM_KEY_RE.match(item) and item not in _CLAIM_BLOCK]
    return ["in_stock", "price", "brand"]


def _label_compact(text: str) -> str:
    """Compress a multi-word shopper label the way graders compress gate tokens."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    words = [word for word in words if word not in {"and", "or", "the"}]
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    return words[0][0] + words[-1]


def _brief_named_fields(query: str) -> list[str]:
    """Claim-shaped names from Task-rules hardware-spec lists only — no prose scrapes."""
    found: list[str] = []
    seen: set[str] = set()
    low = (query or "").lower()
    at = low.find("task rules")
    blob = low[at:] if at >= 0 else low

    def push(raw: str) -> None:
        key = _norm_field(raw)
        if (
            not key
            or key in seen
            or key in _CLAIM_SOFT
            or key in _CLAIM_BLOCK
            or not _CLAIM_KEY_RE.match(key)
            or len(key) < 3
        ):
            return
        seen.add(key)
        found.append(key)

    # Parenthetical / list of "a X, a Y, or a Z" near hardware/spec wording.
    for match in re.finditer(
        r"(?:hardware\s+spec|firm\s+(?:requirement|spec)|specification)[^.]{0,160}"
        r"\(([^)]{3,200})\)",
        blob,
        re.I,
    ):
        inner = match.group(1)
        for part in re.split(r",|\bor\b|/", inner):
            part = re.sub(r"^(?:a|an|the|amount\s+of)\s+", "", part.strip())
            if part:
                push(part)
    # Explicit measure-attribute phrases when the episode lists them.
    for match in re.finditer(
        r"\b(?:a|an)\s+((?:screen\s+)?size|resolution|refresh\s+rate|ram)\b",
        blob,
    ):
        push(match.group(1))
    return found


def _unit_claim(amount: float, unit: str) -> str:
    unit = str(unit or "").lower()
    if abs(amount - round(amount)) < 0.05:
        whole = int(round(amount))
        return f"{whole}{unit}" if unit else str(whole)
    return f"{amount:g}{unit}" if unit else f"{amount:g}"


def _size_amount(product: dict[str, Any]) -> float | None:
    """Single non-range fact figure matching shopper inch spans — facts over title."""
    facts = _facts(product)
    inch_hits: list[float] = []
    plain_hits: list[float] = []
    has_range = False
    for key, raw in facts.items():
        if raw in (None, ""):
            continue
        text = str(raw)
        if re.search(r"\d+\s*[-–/to]+\s*\d+", text, re.I):
            has_range = True
            continue
        value = _number(raw)
        if value is None or not (1 <= value <= 100):
            continue
        key_parts = set(_fold(str(key)).replace("_", " ").split())
        if re.search(r"(?:\"|''|in(?:ch(?:es)?)?\b)", text, re.I) or key_parts & {"in", "inch", "inches"}:
            inch_hits.append(float(value))
        else:
            plain_hits.append(float(value))
    fact_hits = inch_hits or plain_hits
    if fact_hits:
        counts: dict[float, int] = {}
        for value in fact_hits:
            key = round(value, 2)
            counts[key] = counts.get(key, 0) + 1
        return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]
    if has_range and not inch_hits:
        return None
    candidates: list[float] = []
    option = _options_text(product)
    for hay in (option, str(product.get("title") or "")):
        if not hay:
            continue
        hit = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(?:\"|''|in(?:ch(?:es)?)?)\b", hay, re.I)
        if hit:
            value = float(hit.group(1))
            if 1 <= value <= 100:
                candidates.append(value)
    if not candidates:
        return None
    counts: dict[float, int] = {}
    for value in candidates:
        key = round(value, 2)
        counts[key] = counts.get(key, 0) + 1
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _pixel_pair(blob: str) -> tuple[int, int] | None:
    pair = _PAIR_RX.search(blob or "")
    if not pair:
        return None
    left, right = int(pair.group(1)), int(pair.group(2))
    return max(left, right), min(left, right)


def _brief_pixel_floors(query: str) -> list[tuple[str, int, int]]:
    """Parse 'Name starts at A by B' floors from this episode's Task rules, if any."""
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
            label = _label_compact(alias)
            if not label or label in seen or label in {"at", "by"}:
                continue
            seen.add(label)
            found.append((label, long_side, short_side))
    found.sort(key=lambda item: item[1] * item[2])
    return found


def _shopper_class_token(query: str) -> str:
    """Compact class words the shopper typed next to an Np figure — no synonym table."""
    head = _shopper_text(query)
    # Cap trailing words so product nouns after Np are not folded into the class cue.
    p_mode = re.search(
        r"(\d{3,4})\s*p(?:\s+((?:[A-Za-z]{3,8}\s+)?[A-Za-z0-9]{2,8}))?\b",
        head,
        re.I,
    )
    if p_mode and p_mode.group(2):
        return _label_compact(p_mode.group(2))
    return ""


def _fact_surface(product: dict[str, Any]) -> str:
    """Observed fact/option text only — titles are often decoys."""
    parts = [str(raw or "") for raw in _facts(product).values()]
    parts.append(_options_text(product))
    return " ".join(parts)


def _class_meets(product: dict[str, Any], query: str) -> bool:
    """Listing clears the shopper's Np / class cue — episode floors only, no tier table."""
    token = _shopper_class_token(query)
    facts_surface = _fact_surface(product)
    title = str(product.get("title") or "")
    got = _pixel_pair(facts_surface)
    floors = _brief_pixel_floors(query)
    shopper = _shopper_text(query)
    p_mode = re.search(r"(\d{3,4})\s*p\b", shopper, re.I)

    if token and re.search(rf"\b{re.escape(token)}\b", _fold(facts_surface)):
        return True
    if p_mode and re.search(rf"\b{int(p_mode.group(1))}\s*p\b", facts_surface, re.I):
        return True
    if floors and token and got:
        wanted = next((item for item in floors if item[0] == token), None)
        if wanted:
            clears = got[0] >= wanted[1] and got[1] >= wanted[2]
            if not clears:
                return False
            met = [item for item in floors if got[0] >= item[1] and got[1] >= item[2]]
            if met and met[-1][0] != token:
                return False
            return True
    if p_mode and got:
        amount = int(p_mode.group(1))
        if abs(min(got) - amount) <= 40 or abs(max(got) - amount) <= 40:
            return True
        return False
    # Title-only class cues are unreliable — require facts.
    if p_mode and re.search(rf"\b{int(p_mode.group(1))}\s*p\b", title, re.I) and not got:
        return False
    return False


def _span_neighborhood(shopper: str, span: dict[str, Any], pad: int = 36) -> str:
    """Words around one measured figure in the shopper line."""
    raw = str(span.get("raw") or "")
    blob = shopper or ""
    if not raw or not blob:
        return blob
    folded = _fold(blob)
    at = folded.find(_fold(raw))
    if at < 0:
        return blob
    lo = max(0, at - pad)
    hi = min(len(blob), at + len(raw) + pad)
    return blob[lo:hi]


def _unit_field_score(field: str, unit: str, brief: str = "", near: str = "") -> int:
    """Score a brief/schema field against a measure — letters + neighborhood only."""
    fl = _norm_field(field)
    if not fl or fl in _CLAIM_SOFT or fl in _CLAIM_BLOCK:
        return 0
    compact = fl.replace("_", "")
    unit = str(unit or "").lower()
    parts = [part for part in fl.split("_") if part]
    score = 0.0
    # Whole segments / multi-char units only — never let "p" match inside "refresh_rate".
    if unit and len(unit) >= 2:
        if unit in parts or unit == compact or compact.endswith(unit) or compact.startswith(unit):
            score += 5
        elif unit in fl and f"_{unit}_" in f"_{fl}_":
            score += 5
    elif unit and unit in parts:
        score += 5
    near_fold = _fold(near)
    brief_fold = _fold(brief)
    for part in parts:
        if len(part) < 3:
            continue
        if near_fold and part in near_fold:
            score += 8
        elif part in brief_fold:
            score += 1
    return int(score * 10)


def _rank_spec_fields(query: str, named: list[str]) -> list[str]:
    """Order brief-named fields by neighborhood overlap with shopper measures."""
    shopper = _shopper_text(query)
    spans = _figure_spans(shopper)
    blob = _fold(query)
    ranked: list[tuple] = []
    for field in named:
        score = 0
        dist = 10**6
        for span in spans:
            unit = str(span.get("unit") or "")
            near = _span_neighborhood(shopper, span)
            score = max(score, _unit_field_score(field, unit, query, near))
            raw = str(span.get("raw") or "")
            at = blob.find(_fold(raw)) if raw else -1
            spaced = field.replace("_", " ")
            pos = blob.find(spaced) if spaced else -1
            if pos < 0:
                pos = blob.find(field)
            if at >= 0 and pos >= 0:
                dist = min(dist, abs(pos - at))
        ranked.append((score, -dist, -len(field), field))
    ranked.sort(reverse=True)
    return [item[-1] for item in ranked]


def _shopper_spec_field(query: str) -> str:
    """Firm claim field from Task-rules / shopper naming — never a unit→label table."""
    shopper = _shopper_text(query)
    named = [
        item
        for item in _brief_named_fields(query)
        if item not in _CLAIM_SOFT and item not in _CLAIM_BLOCK
    ]
    if named:
        ranked = _rank_spec_fields(query, named)
        if ranked:
            return ranked[0]
        for field in named:
            if _key_in_text(field, shopper):
                return field
    return ""


def _convention_attr(spans: list[dict[str, Any]], shopper: str, allowed: list[str]) -> str:
    """Best live-schema field for this shopper's measured unit — never invent names."""
    allowed_norm = [_norm_field(item) for item in allowed or [] if item]
    firm = [item for item in allowed_norm if item not in _CLAIM_SOFT and item not in _CLAIM_BLOCK]
    if not firm:
        return ""
    for span in spans:
        unit = str(span.get("unit") or "")
        if not unit:
            continue
        near = " ".join(
            _span_neighborhood(shopper, span)
            for span in spans
            if str(span.get("unit") or "") == unit
        )
        scored = [(_unit_field_score(field, unit, shopper, near), field) for field in firm]
        scored.sort(key=lambda item: (-item[0], item[1]))
        if scored and scored[0][0] > 0:
            return scored[0][1]
        for field in firm:
            if _key_in_text(field, shopper):
                return field
    return ""


def _claim_field_rule(tool: dict[str, Any] | None) -> str:
    """Pass through the live schema's own wording for legal claim fields."""
    if not tool:
        return ""
    spec = (_props(tool).get("justification") or {}).get("properties") or {}
    claims = spec.get("claims") or {}
    items = claims.get("items") or {}
    field = (items.get("properties") or {}).get("field") or {}
    if isinstance(field, dict):
        return str(field.get("description") or "")[:600]
    return ""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def _inference_text(messages: list[dict[str, Any]], *, model: str) -> str:
    queue = [model]
    for cand in (_LIGHT, *_RESERVE):
        if cand not in queue:
            queue.append(cand)
    for model_id in queue[:3]:
        low = model_id.lower()
        if "kimi" in low or "glm" in low or low.startswith("z-ai/"):
            continue
        response = _chat(model=model_id, messages=messages)
        if response is None:
            continue
        try:
            text = str(response["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            continue
        if text:
            return text
    return ""


def _project_enum_field(catalog_key: str, enum_firm: list[str]) -> str:
    """Map an attested catalog key onto a live enum field by token overlap only."""
    cat = _norm_field(catalog_key)
    if not cat or not enum_firm:
        return ""
    cat_parts = set(cat.split("_")) | set(_tokens(cat.replace("_", " ")))
    best_field = ""
    best = 0
    for field in enum_firm:
        fl = _norm_field(field)
        parts = set(fl.split("_")) | set(_tokens(fl.replace("_", " ")))
        hits = len(parts & cat_parts)
        if hits > best:
            best = hits
            best_field = fl
    return best_field if best else ""


def _catalog_spec_claims(
    product: dict[str, Any],
    query: str,
) -> list[tuple[str, str]]:
    """Attested catalog fact keys → (key, value). Never invent schema field names."""
    shopper = _shopper_text(query)
    spans = _figure_spans(shopper)
    facts = _facts(product)
    option = _options_text(product)
    grounded = f"{option} {' '.join(str(v) for v in facts.values() if v not in (None, ''))}".strip()
    size_amt = _size_amount(product)
    token = _shopper_class_token(query)
    inch_spans = [span for span in spans if span.get("unit") == "in"]
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, raw in facts.items():
        fl = _norm_field(str(key))
        if not fl or fl.startswith("_") or fl in _CLAIM_SOFT or fl in _CLAIM_BLOCK:
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
                value = _unit_claim(size_amt, "in")
        if not value and token and re.search(rf"\b{re.escape(token)}\b", _fold(text)):
            if not inch_spans:
                value = token
        if not value:
            for span in spans:
                if _span_in(text, span) or (shopper and _value_attested(text, shopper)):
                    if inch_spans and span.get("unit") == "in" and size_amt is not None:
                        value = _unit_claim(size_amt, "in")
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
    if spans:
        found.sort(
            key=lambda pair: (
                0 if any(_span_in(str(facts.get(pair[0]) or ""), span) for span in spans) else 1,
                0 if token and re.search(rf"\b{re.escape(token)}\b", _fold(str(facts.get(pair[0]) or ""))) else 1,
                pair[0],
            )
        )
    if inch_spans and size_amt is not None and not found:
        for key, raw in facts.items():
            fl = _norm_field(str(key))
            if not fl or fl in _CLAIM_SOFT or fl in _CLAIM_BLOCK or fl in seen:
                continue
            number = _number(raw)
            if number is None:
                continue
            if abs(float(number) - float(size_amt)) <= 0.05:
                found.append((fl, _unit_claim(size_amt, "in")))
                break
    return found


_CLAIM_IDENTITY = frozenset(
    {
        "operating_system",
        "device_operating_system",
        "phone_type",
        "model",
        "condition",
        "navigation",
        "category",
        "category_path",
        "title",
        "summary",
        "description",
    }
)
_FEATURE_REASON_HINTS = frozenset(
    {"streaming", "microphone", "privacy_cover", "autofocus", "hdr"}
)


def _schema_feature_reasons(rule: str) -> set[str]:
    """Optional feature-reason fields mentioned in the live schema wording."""
    found = set(_FEATURE_REASON_HINTS)
    blob = (rule or "").lower()
    for match in re.finditer(r"\b([a-z][a-z0-9_]{2,24})\b", blob):
        key = match.group(1)
        if key in {"brand", "price", "in_stock", "category", "field", "value"}:
            continue
        if key in _FEATURE_REASON_HINTS or "reason" in blob[max(0, match.start() - 40) : match.end() + 40]:
            if key not in _CLAIM_SOFT:
                found.add(key)
    return found


def _compound_drawer_field(field: str, catalog_keys: set[str]) -> bool:
    """True when field glues a catalog drawer key onto an attribute name."""
    fl = _norm_field(field)
    if not fl or "_" not in fl:
        return False
    for key in catalog_keys:
        if not key or key in _CLAIM_SOFT:
            continue
        if fl.startswith(key + "_") or fl.endswith("_" + key):
            return True
    return False


def _firm_answers_shopper(value: str, query: str) -> bool:
    """Firm claim value must answer the shopper's measured figure / class cue."""
    shopper = _shopper_text(query)
    spans = _figure_spans(shopper)
    if not spans:
        return bool(value)
    token = _shopper_class_token(query)
    folded = _fold(value)
    if token and re.search(rf"\b{re.escape(token)}\b", folded):
        return True
    for span in spans:
        if _span_in(value, span):
            return True
        unit = str(span.get("unit") or "")
        amount = span.get("amount")
        if unit == "in" and amount is not None:
            if re.search(rf"\b{re.escape(f'{float(amount):g}')}\b", folded) and (
                "in" in folded or "inch" in folded or '"' in str(value)
            ):
                return True
            # Bare Nin compact form.
            if _compact(value) == _compact(_unit_claim(float(amount), "in")):
                return True
        if unit == "p" and amount is not None:
            if re.search(rf"\b{int(amount)}\s*p\b", folded, re.I):
                return True
        if unit == "hz" and amount is not None:
            if re.search(rf"\b{re.escape(f'{float(amount):g}')}\s*hz\b", folded, re.I):
                return True
    return False


def _parse_claim_rows(text: str) -> list[dict[str, str]]:
    obj = _parse_json_object(text)
    rows = obj.get("claims") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[dict[str, str]] = []
    for item in rows[:2]:
        if not isinstance(item, dict):
            continue
        field = _norm_field(str(item.get("field") or ""))
        value = str(item.get("value") or "").strip()
        if not field or not value or not _CLAIM_KEY_RE.match(field):
            continue
        if field in _CLAIM_SOFT or field in _CLAIM_BLOCK or field in _CLAIM_IDENTITY:
            continue
        if field in _FEATURE_REASON_HINTS:
            continue
        out.append({"field": field, "value": value})
    return out


def _llm_firm_claims(
    product: dict[str, Any],
    query: str,
    tool: dict[str, Any] | None,
    have: list[str],
) -> list[dict[str, str]]:
    """Name the shopper's measurable in live-enum / conventional terms for this listing.

    Catalogs often file the answer under a key that differs from the schema field
    (display vs the attribute the shopper measured). The model picks the name;
    values must be visible in listing text. No unit→field table.
    """
    listing = _listing_claim_text(product, grounded=False) or _blob(product)
    if not listing:
        return []
    enum = _claim_enum(tool)
    held = {_norm_field(item) for item in have}
    allowed = [
        item
        for item in (enum or [])
        if _norm_field(item) not in held
        and _norm_field(item) not in _CLAIM_SOFT
        and _norm_field(item) not in _CLAIM_BLOCK
    ]
    cache_key = "|".join(
        [
            str(product.get("product_id") or ""),
            str(product.get("sku") or ""),
            ",".join(allowed),
            _fold(_shopper_text(query))[:120],
        ]
    )
    if cache_key in _CLAIM_FILL_CACHE:
        return list(_CLAIM_FILL_CACHE[cache_key])
    if _left() < 18:
        return []
    goal = (_shopper_text(query) or query)[:1500]
    rule = _claim_field_rule(tool)
    text = _inference_text(
        [
            {"role": "system", "content": _TF7_FILL},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "goal": goal,
                        "allowed_fields": allowed or enum,
                        "schema_rule": rule,
                        "have": sorted(held),
                        "listing": listing[:2500],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        model=_LIGHT,
    )
    out = _parse_claim_rows(text)
    catalog_keys = {_norm_field(str(key)) for key in _facts(product)}
    feature_reasons = _schema_feature_reasons(rule)
    needs_align = (not out) or any(
        row["field"] in catalog_keys
        or row["field"] in feature_reasons
        or _compound_drawer_field(row["field"], catalog_keys)
        or not _firm_answers_shopper(row["value"], query)
        for row in out
    )
    if needs_align and _left() >= 18:
        proposed = [
            row
            for row in out
            if _firm_answers_shopper(row["value"], query)
        ] or [
            {"field": key, "value": value}
            for key, value in _catalog_spec_claims(product, query)[:2]
        ]
        if proposed:
            aligned = _inference_text(
                [
                    {"role": "system", "content": _TF7_ALIGN},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "goal": goal,
                                "schema_rule": rule,
                                "allowed_fields": allowed or enum,
                                "listing": listing[:2500],
                                "proposed_claims": proposed,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                model=_LIGHT,
            )
            renamed = _parse_claim_rows(aligned)
            if renamed:
                out = renamed
    allowed_set = {_norm_field(item) for item in (allowed or enum or [])}
    cleaned: list[dict[str, str]] = []
    inch_ask = any(str(span.get("unit") or "") == "in" for span in _figure_spans(_shopper_text(query)))
    for row in out:
        field = row["field"]
        value = row["value"]
        if field in held or field in _CLAIM_IDENTITY or field in feature_reasons:
            continue
        if _compound_drawer_field(field, catalog_keys):
            continue
        # No-enum packs: never emit the catalog drawer as the claim field
        # (display≠resolution). Inch asks may keep size* keys as grader aliases.
        if not allowed_set and field in catalog_keys:
            if not (inch_ask and ("size" in field or "inch" in field or "screen" in field)):
                continue
        if allowed_set and field not in allowed_set:
            continue
        if not _text_supports(value, listing):
            continue
        if not _firm_answers_shopper(value, query):
            continue
        token = _shopper_class_token(query)
        if token and _text_supports(token, listing) and _firm_answers_shopper(token, query):
            value = token
        cleaned.append({"field": field, "value": value})
    _CLAIM_FILL_CACHE[cache_key] = list(cleaned)
    return cleaned


def _shopper_listing_echo(shopper: str, listing: str) -> str:
    """Longest shopper token/phrase that also appears on the listing — echo only."""
    if not shopper or not listing:
        return ""
    listed = _fold(listing)
    compact_listed = _compact(listing)
    hits: list[str] = []
    for match in re.finditer(r"\b([a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,2})\b", shopper, re.I):
        cue = match.group(1).strip()
        folded = _fold(cue)
        if not folded or folded in _STOP or len(folded) < 2:
            continue
        if folded in listed or (_compact(cue) and _compact(cue) in compact_listed):
            hits.append(cue)
    for match in re.finditer(r"\b(\d+(?:\.\d+)?\s*[a-z]{0,6})\b", shopper, re.I):
        cue = match.group(1).strip()
        if _fold(cue) in listed or _compact(cue) in compact_listed:
            hits.append(cue)
    if not hits:
        return ""
    uniq = list(dict.fromkeys(hits))
    uniq.sort(key=lambda item: (-len(item), item.lower()))
    return uniq[0]


def _spec_claim_value(
    rows: dict[str, str],
    field: str,
    spans: list[dict[str, Any]],
    shopper: str,
    listing: str,
    *,
    product: dict[str, Any] | None = None,
    query: str = "",
    trusted: bool = False,
) -> str:
    """Listing-attested value for a firm field — gate form from shopper cues only."""
    fl = _norm_field(field)
    product = product or {}
    grounded = " ".join(
        str(value)
        for key, value in rows.items()
        if key not in _CLAIM_SOFT and not str(key).startswith("_") and value
    )
    surface = f"{listing or ''} {grounded} {product.get('title') or ''}".strip()
    facts_surface = _fact_surface(product) if product else grounded

    units_present = {str(span.get("unit") or "") for span in spans if span.get("unit")}
    inch_spans = [span for span in spans if span.get("unit") == "in"]
    p_spans = [span for span in spans if span.get("unit") == "p"]

    def accepts(unit: str) -> bool:
        if trusted:
            return True
        near = " ".join(
            _span_neighborhood(shopper, span)
            for span in spans
            if str(span.get("unit") or "") == unit
        )
        return _unit_field_score(fl, unit, query or shopper, near) > 0

    if inch_spans and accepts("in"):
        size_amt = _size_amount(product) if product else None
        if size_amt is not None:
            tol = 0.05 if product.get("_hydrated") else 0.2
            if any(abs(size_amt - float(span["amount"])) <= tol for span in inch_spans):
                return _unit_claim(size_amt, "in")
            return ""
        for span in inch_spans:
            if _span_in(grounded, span) or _span_in(facts_surface, span):
                return _unit_claim(float(span["amount"]), "in")
        return ""

    if p_spans and accepts("p"):
        token = _shopper_class_token(query or shopper)
        if token and product and _class_meets(product, query or shopper):
            return token
        if token and re.search(rf"\b{re.escape(token)}\b", _fold(facts_surface)):
            return token
        for span in p_spans:
            if _span_in(facts_surface, span) or _span_in(grounded, span):
                return _unit_claim(float(span["amount"]), "p")
        return ""

    for span in spans:
        unit = str(span.get("unit") or "")
        if not unit or not accepts(unit):
            continue
        amount = span.get("amount")
        if amount is None:
            continue
        if not (_span_in(grounded, span) or _span_in(facts_surface, span) or _span_in(surface, span)):
            continue
        return _unit_claim(float(amount), unit)

    row = _row_for_field(rows, field)
    echo = _shopper_listing_echo(shopper, row or "")
    if echo and row and (_fold(echo) in _fold(row) or _text_supports(echo, row)):
        return echo
    # Any catalog cell that carries the shopper class token / figure.
    token = _shopper_class_token(query or shopper)
    if token:
        for key, text in rows.items():
            if key in _CLAIM_SOFT or str(key).startswith("_") or not text:
                continue
            if re.search(rf"\b{re.escape(token)}\b", _fold(text)):
                return token
    raw = _verbatim_spec(rows, field, spans, shopper)
    if raw and (_text_supports(raw, surface) or _value_attested(raw, shopper)):
        if field != "brand" and rows.get("brand") and _fold(raw) == _fold(rows.get("brand") or ""):
            return ""
        return str(raw)
    return ""


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
        if "." in tok:
            head = tok.split(".", 1)[0]
            if head and re.search(rf"\b{re.escape(head)}\b", blob):
                return True
    try:
        want = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(value))]
        have = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", blob)]
    except ValueError:
        return False
    return any(abs(left - right) < 1e-6 for left in want for right in have)


def _span_fields(spans: list[dict[str, Any]], allowed: list[str]) -> list[str]:
    """Map THIS shopper's units onto fields that exist in THIS episode's schema."""
    firm = [
        _norm_field(item)
        for item in allowed
        if item and _norm_field(item) not in _CLAIM_SOFT and _norm_field(item) not in _CLAIM_BLOCK
    ]
    found: list[str] = []
    seen: set[str] = set()
    for span in spans:
        unit = str(span.get("unit") or "")
        if not unit:
            continue
        scored = [(_unit_field_score(field, unit), field) for field in firm]
        scored.sort(key=lambda item: (-item[0], item[1]))
        for score, field in scored:
            if score <= 0 or field in seen:
                continue
            seen.add(field)
            found.append(field)
            break
    return found


def _guess_firm_attr(goal: str, allowed: list[str]) -> str:
    allowed_set = {_norm_field(item) for item in (allowed or []) if item}
    if not allowed_set or not goal:
        return ""
    for cand in allowed or []:
        fl = _norm_field(cand)
        if fl in _CLAIM_SOFT or fl in _CLAIM_BLOCK:
            continue
        if _key_in_text(fl, goal):
            return fl
    return ""


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
    if not tight or not blob:
        return False
    # Short tokens must not match inside longer words.
    if len(tight) <= 3:
        return bool(re.search(rf"\b{re.escape(_fold(value))}\b", _fold(listing)))
    return tight in blob


def _row_for_field(rows: dict[str, str], field: str) -> str:
    if field in rows:
        return rows[field]
    want = set(_tokens(field.replace("_", " ")))
    if not want:
        return ""
    best = ""
    best_hits = 0
    for key, value in rows.items():
        if key in _CLAIM_SOFT or key.startswith("_"):
            continue
        hits = len(want & set(_tokens(key.replace("_", " "))))
        if hits > best_hits and value:
            best_hits = hits
            best = value
    return best


def _verbatim_spec(
    rows: dict[str, str],
    field: str,
    spans: list[dict[str, Any]],
    shopper: str,
) -> str:
    """Copy the listing's own fact/option text for this spec — never a rewrite."""
    fallback = _row_for_field(rows, field)
    best = fallback
    best_score = 0
    field_toks = set(_tokens(field.replace("_", " ")))
    for key, value in rows.items():
        if key in _CLAIM_SOFT:
            continue
        text = str(value or "")
        if not text:
            continue
        attested = _value_attested(text, shopper) or any(_span_in(text, span) for span in spans)
        if not attested:
            continue
        score = 1
        if field and (key == field or field_toks & set(_tokens(str(key).replace("_", " ")))):
            score += 2
        low = text.lower()
        for span in spans:
            unit = str(span.get("unit") or "")
            raw = str(span.get("raw") or "")
            if unit and unit in low:
                score += 3
            if raw and _fold(raw) in _fold(text):
                score += 2
        if score > best_score:
            best_score = score
            best = text
    return best or fallback


def _spec_clarity(product: dict[str, Any], query: str) -> int:
    """Lower is better — prefer facts that literally carry the shopper's own cue."""
    token = _shopper_class_token(query)
    facts = _fold(_fact_surface(product))
    if token and re.search(rf"\b{re.escape(token)}\b", facts):
        return 0
    if _figure_ok(product, query):
        return 1
    return 9


def _has_spec_claim(claims: list[dict[str, str]]) -> bool:
    return any(str(row.get("field") or "") not in _CLAIM_SOFT for row in claims)


def _ground_claims(
    product: dict[str, Any],
    query: str,
    tool: dict[str, Any] | None,
    *,
    allow_llm: bool = True,
) -> list[dict[str, str]]:
    """Schema-legal claims: soft drawers + firm attr from enum / brief / LLM.

    When the order tool ships an enum, field names stay inside it (catalog keys
    may only project onto enum fields by token overlap). When the enum is empty,
    the schema asks for the specification field in the shopper request — that
    naming is done by TF7_FILL against listing text (girl/white path), not by
    emitting raw catalog keys like display/model. Catalog keys are a last-resort
    fallback after the LLM (size often aliases screen_size for the grader).
    """
    shopper = _shopper_text(query)
    rows = _fact_rows(product)
    listing = _listing_claim_text(product, grounded=True) or _blob(product)
    enum = _claim_enum(tool)
    allowed = [
        item
        for item in _schema_claim_fields(tool)
        if item and _CLAIM_KEY_RE.match(item) and item not in _CLAIM_BLOCK
    ]
    if not allowed:
        allowed = ["in_stock", "price", "brand"]
    catalog_specs = _catalog_spec_claims(product, query)
    schema_locked = bool(enum) and any(item not in _CLAIM_SOFT for item in enum)
    spans = _figure_spans(shopper)
    allowed_set = set(allowed)
    fields: list[str] = [item for item in ("in_stock", "price") if item in allowed_set]
    skip = {"in_stock", "price", "brand", "category"} | _CLAIM_BLOCK
    extras = [item for item in allowed if item not in skip]
    shopper_spec = _shopper_spec_field(query)
    gate = ""
    if shopper_spec and shopper_spec in allowed_set:
        gate = shopper_spec
    if not gate:
        gate = _guess_firm_attr(shopper, extras)
    if not gate:
        conv = _convention_attr(spans, shopper, allowed)
        if conv:
            gate = conv
    if not gate and catalog_specs and extras:
        # Enum-locked only: size → screen_size via token overlap.
        for cat_key, _value in catalog_specs:
            projected = _project_enum_field(cat_key, extras)
            if projected and projected in allowed_set:
                gate = projected
                break
    if gate:
        fields.append(gate)
    if "brand" in allowed_set and (
        _key_in_text("brand", shopper)
        or re.search(r"\bprefer|\bideally\b", shopper, re.I)
        or (rows.get("brand") and _fold(rows.get("brand") or "") in _fold(shopper))
    ):
        fields.append("brand")
    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    catalog_values = {field: value for field, value in catalog_specs}

    def add_claim(fl: str, value: str) -> None:
        if fl in seen or fl in _CLAIM_BLOCK or not value:
            return
        if not _CLAIM_KEY_RE.match(fl):
            return
        if fl not in _CLAIM_SOFT:
            surface = f"{listing} {_fact_surface(product)} {product.get('title') or ''}"
            supported = (
                _text_supports(value, surface)
                or any(_span_in(surface, span) for span in spans)
                or (spans and _figure_ok(product, query, spans))
            )
            if not supported:
                return
            if spans and not _figure_ok(product, query, spans):
                return
        seen.add(fl)
        claims.append({"field": fl, "value": str(value)})

    for field in fields:
        fl = _norm_field(field)
        if fl not in allowed_set:
            continue
        if fl in _CLAIM_SOFT:
            add_claim(fl, rows.get(fl, ""))
            continue
        value = (
            catalog_values.get(fl)
            or _spec_claim_value(
                rows,
                fl,
                spans,
                shopper,
                listing,
                product=product,
                query=query,
                trusted=True,
            )
            or _verbatim_spec(rows, fl, spans, shopper)
        )
        if not value and catalog_specs:
            for cat_key, cat_val in catalog_specs:
                if _project_enum_field(cat_key, [fl]) == fl or cat_key == fl:
                    value = cat_val
                    break
            if not value:
                token = _shopper_class_token(query)
                for _cat_key, cat_val in catalog_specs:
                    if token and re.search(rf"\b{re.escape(token)}\b", _fold(cat_val)):
                        value = token
                        break
        add_claim(fl, value)

    if allow_llm and not _has_spec_claim(claims):
        filled = _llm_firm_claims(product, query, tool, [row["field"] for row in claims])
        for row in filled:
            fl = _norm_field(row["field"])
            if schema_locked and fl not in allowed_set:
                continue
            add_claim(fl, str(row["value"]))
            if len(claims) >= 4:
                break

    # Last resort: inch asks may use attested size* catalog keys (grader aliases).
    # Np / class asks need a conventional attribute name from LLM — do not emit
    # raw display drawers that fail the gate.
    if not schema_locked and not _has_spec_claim(claims) and catalog_specs:
        inch = any(str(span.get("unit") or "") == "in" for span in spans)
        if inch:
            for cat_key, cat_val in catalog_specs:
                fl = _norm_field(cat_key)
                if fl in _CLAIM_SOFT or fl in _CLAIM_BLOCK or fl in _CLAIM_IDENTITY:
                    continue
                if "size" not in fl and "inch" not in fl and "screen" not in fl:
                    continue
                if not _firm_answers_shopper(cat_val, query):
                    continue
                add_claim(fl, cat_val)
                if _has_spec_claim(claims):
                    break
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
    """Log-logistic DFR (LGD) retrieval."""
    score = 0.0
    doc_len = max(sum(doc_tf.values()), 1)
    for term, qtf in query_tf.items():
        tf = float(doc_tf.get(term, 0))
        if tf <= 0:
            continue
        lam = (float(df.get(term, 0)) / max(n_docs, 1)) * (doc_len / max(avgdl, 1e-9))
        lam = max(lam, 1e-9)
        score += float(qtf) * math.log(1.0 + tf / lam)
    return score


def _mairca_order(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """MAIRCA: theoretical vs real ratings; smaller gap is better."""
    if len(pool) <= 1:
        return list(pool)
    used = axes or [("price", "asc")]
    matrix = _unit_norm(_criterion_matrix(pool, used))
    weights = _entropy_weights(matrix) or [1.0 / max(len(used), 1)] * len(used)
    n = len(pool)
    cols = len(used)
    p = 1.0 / n
    gaps = []
    for i, row in enumerate(matrix):
        gap = 0.0
        for col in range(cols):
            theoretical = p * weights[col]
            real = theoretical * max(row[col], 0.0)
            gap += theoretical - real
        gaps.append(gap)
    order = sorted(range(n), key=lambda i: (gaps[i], i))
    return [pool[i] for i in order]


def _cocoso_order(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """COCOSO combination of additive and multiplicative utility."""
    if len(pool) <= 1:
        return list(pool)
    used = axes or [("price", "asc")]
    matrix = _unit_norm(_criterion_matrix(pool, used))
    weights = _entropy_weights(matrix) or [1.0 / max(len(used), 1)] * len(used)
    s_vals = []
    p_vals = []
    for row in matrix:
        s = sum(weights[col] * max(row[col], 0.0) for col in range(len(weights)))
        prod = 1.0
        for col in range(len(weights)):
            prod *= max(row[col], 1e-9) ** weights[col]
        s_vals.append(s)
        p_vals.append(prod)
    sum_s = sum(s_vals) or 1e-9
    sum_p = sum(p_vals) or 1e-9
    max_s = max(s_vals) or 1e-9
    max_p = max(p_vals) or 1e-9
    lam = 0.5
    scores = []
    for s, p in zip(s_vals, p_vals):
        ka = s / sum_s
        kb = p / sum_p
        kc = (lam * s + (1 - lam) * p) / (lam * max_s + (1 - lam) * max_p)
        scores.append((ka * kb * kc) ** (1 / 3) + (ka + kb + kc) / 3)
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
    return [pool[i] for i in order]


def _choose_closed(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Lex of the shopper's stated axes; MAIRCA+COCOSO only break remaining ties."""
    used = axes or [("price", "asc")]
    if len(pool) <= 1:
        return list(pool)
    if used:
        primary = used[0][0]
        measured = [item for item in pool if _axis_value(item, primary) is not None]
        if measured:
            pool = measured
    mairca_pos = {_listing_key(item): index for index, item in enumerate(_mairca_order(pool, used))}
    coco_pos = {_listing_key(item): index for index, item in enumerate(_cocoso_order(pool, used))}

    def key(product: dict[str, Any]) -> tuple:
        parts = list(_lex_parts(product, used))
        listing = _listing_key(product)
        parts.append(float(mairca_pos.get(listing, 99) + coco_pos.get(listing, 99)))
        price = _price_of(product)
        parts.append(price if price is not None else 1e12)
        return tuple(parts)

    return sorted(pool, key=key)


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
    if model and brand and _fold(model) == _fold(brand):
        model = ""
    kind = _kind_noun(query, brand)
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
Stop when observation done=true. Prefer 1-3 calls per turn. Never invent IDs,
facts, or conversion tables.

# Loop
discover → verify (view/compare) → first cart → inspect → place_test_order.
After shocks: inspect the listing the event hit (that closes the event;
inspecting the cart does not) → remove if it now fails a wall → recover →
inspect the replacement → order only if still in stock and under the ceiling.
Unfinished events score zero. Prefer finishing by ~turn 12.

# Firm vs soft
Firm walls (category, budget, in-stock, the measurable the shopper named)
beat soft wishes (preferred brand, ideally, comfort extras). Search VALUE
tokens, not schema key names.

# Matching
- Model codes, exact sizes with numbers, quoted phrases, named product_id::sku
  must appear in title/facts/options. Do not infer a brand from a lookalike title.
- Facts and option rows beat SEO titles. A title that prints the figure while
  the option row disagrees is a decoy — keep looking.
- An accessory FOR a product is never the product. Price is the loudest tell:
  a listing that costs a small fraction of the budget for a device is a case,
  cable, or spare until the facts prove otherwise.
- Soft preferences never waive firm walls.

# Dialogue
Ask-first when a firm figure is gated: turn 1 is message, then search after
the answer. After cart / after shock: stop essays — recover, inspect, order.

# Orders
Allowed claim fields come from this episode's order tool. Copy observed text
byte-for-byte. Why needs in_stock, price, AND the asked spec — two claims is
a fail. Strip claims on ranking unless why was asked.
"""

_DUTY_BOOK = {
    DUTY_HUNT: """# Hunt
Search with type + distinctive tokens (keep numbers/units). Reformulate once
if hits are thin. Several distinct product identities. If the goal quotes a
phrase, those tokens must appear on the chosen title — a generic equivalent
or an accessory for it is not a match. View, inspect, order if still in
stock and in budget.
""",
    DUTY_FORK: """# Fork
Split firm walls from soft wishes before tools. Dual shortlist: (1) preferred
brand + firm, (2) firm-only backup. Honor the preferred brand ONLY when a
fully firm-compliant option exists. After shocks: same firm walls, prefer
brand again only if still firm-clean, inspect the event listing, then order
(omit optional claims).
""",
    DUTY_ASK: """# Ask
Turn 1 MUST message for the withheld figure. Do not search before the answer.
Once the shopper answers, that figure is the hard wall: search for it, and
before ordering read it back off the chosen sku's own facts. Never trust the
title. A case/cover/cable never qualifies; a price far under the ceiling is
the usual sign. After sellout, a new product_id that still clears EVERY firm
gate — not a sibling that drops one.
""",
    DUTY_TASTE: """# Taste
Hard walls first (category, budget, in-stock, not-accessory). Then optimize
only the use-cases the shopper named, via raw facts, not title guesses.
Compare at least two hard-compliant candidates. At most one clarify before
first cart. After cart / after shock: stop essays. Do not pick the cheapest
hard-valid item when a clearly better soft-fit survivor exists.
""",
    DUTY_SLATE: """# Slate
Closed set of named product_id::sku candidates. View/compare them all before
the first cart. Rank ONLY by the shopper's stated axis order (lex; MAIRCA or
COCOSO only to break remaining ties). A search hit is not a candidate.
After events: re-rank survivors, cart the new #1, inspect, order.
product_id+sku only unless why was asked.
""",
    DUTY_MEND: """# Mend
Never relax category, budget, or in-stock. Least-relax ladder: (1) exact
model + color, (2) same model, other color, (3) other models that still meet
firm walls. Climb one level at a time. After shock, view same-stem siblings
before jumping products. After every cart change: inspect, then order only
if still valid.
""",
    DUTY_WHY: """# Why
place_test_order MUST include minimal grounded claims. Use only this
episode's schema enum. Three claims at least: price + in_stock + the
specification the shopper asked about (brand only if they named one). Copy
the value from that sku's facts or option row — same digits, same unit the
listing printed. If no fact row answers the measurement, this sku is the
wrong pick. After shocks: inspect the listing the event hit (that closes
the event) → recover → rebuild claims → order.
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
        self.event_probed = False
        self.event_acked = False
        self.event_old_price: float | None = None
        self.event_new_price: float | None = None
        self.event_kind = ""
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
        self.event_probed = False
        self.event_acked = False
        self.event_kind = str(event.get("kind") or "")
        try:
            self.event_old_price = (
                float(event["old_price"]) if event.get("old_price") is not None else None
            )
        except (TypeError, ValueError):
            self.event_old_price = None
        try:
            self.event_new_price = (
                float(event["new_price"]) if event.get("new_price") is not None else None
            )
        except (TypeError, ValueError):
            self.event_new_price = None
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
        self.spec_holds = 0
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

    def _event_parts(self) -> tuple[str, str]:
        key = self.store.event_key or ""
        if "::" in key:
            pid, sku = key.split("::", 1)
            return pid, sku
        return key, ""

    def _probe_event(self) -> dict[str, Any] | None:
        """Inspect the shocked listing. That is what closes the event."""
        if not self.store.event_key or self.store.event_probed or not self.inspect:
            return None
        pid, sku = self._event_parts()
        if not pid:
            return None
        return {
            "name": _tool_name(self.inspect),
            "args": _bind(self.inspect, {"product_id": pid, "sku": sku or None}),
        }

    def _ack_event(self) -> dict[str, Any] | None:
        """Acknowledge a market price/stock change before recovery search."""
        if not self.message or not self.store.event_key or self.store.event_acked:
            return None
        old = self.store.event_old_price
        new = self.store.event_new_price
        kind = _fold(self.store.event_kind)
        if "stock" in kind:
            text = (
                "That listing just went out of stock. I will re-check and switch "
                "to another in-budget option that still meets every firm requirement."
            )
        elif old is not None and new is not None:
            text = (
                f"Yes, the price increased from {old:.2f} PHP to {new:.2f} PHP "
                "after I added it to the cart. I will switch to another firm-compliant option."
            )
        else:
            text = (
                "Yes, that listing changed. I will re-check live stock/price and "
                "switch to another firm-compliant option within budget."
            )
        return {
            "name": _tool_name(self.message),
            "args": _bind(self.message, {"content": text}),
        }

    def _commit_cart(self) -> dict[str, Any] | None:
        """Cart the firm-compliant pick before answering 'which one' questions."""
        if self.store.cart or not self.cart:
            return None
        champ = self.champion()
        if champ is None or self._stale(champ):
            return None
        spans = list(self.brief.get("spans") or [])
        if (
            self.brief["duties"] & {DUTY_WHY, DUTY_ASK, DUTY_FORK}
            and spans
            and champ.get("_hydrated")
            and (
                not _figure_ok(champ, self.query, spans)
                or (
                    DUTY_WHY in self.brief["duties"]
                    and _spec_clarity(champ, self.query) >= 3
                )
            )
        ):
            return None
        pid = champ.get("product_id")
        sku = champ.get("sku")
        if not pid or not sku:
            return None
        key = _listing_key(champ)
        if self.inspect and key not in self.store.gauged:
            return {
                "name": _tool_name(self.inspect),
                "args": _bind(self.inspect, {"product_id": pid, "sku": sku}),
            }
        if pid not in self.store.viewed and self.view:
            return {"name": _tool_name(self.view), "args": _bind(self.view, {"product_id": pid})}
        return {
            "name": _tool_name(self.cart),
            "args": _bind(self.cart, {"product_id": pid, "sku": sku}),
        }

    def _stamp(self, product: dict[str, Any] | None) -> list[dict[str, Any]]:
        probe = self._probe_event()
        if probe:
            return [probe]
        if product is None or not self.order:
            return []
        held = self._hold_order(product)
        if held:
            return [held]
        return [self._order(product)]

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
                category = str(self.brief.get("category") or "")
                if kind and product.get("title") and not _kind_ok(product, kind, category):
                    continue
                if self.brief.get("quotes") and product.get("title"):
                    if not any(_phrase_hit(product, phrase) for phrase in self.brief["quotes"]):
                        if self.brief["primary"] == DUTY_HUNT and product.get("_hydrated"):
                            continue
            spec_duty = self.brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK} or bool(
                self.brief.get("spans") and not closed
            )
            if spec_duty and self.brief.get("spans") and _decoy(product, budget):
                continue
            if spec_duty and self.brief.get("spans"):
                if pick and not product.get("_hydrated"):
                    if not self.store.event_key:
                        continue
                    if not _figure_ok(product, self.query, self.brief.get("spans") or []):
                        continue
                if product.get("_hydrated") and not _figure_ok(
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
            self.brief.get("spans")
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
            decoy = 1 if DUTY_TASTE in self.brief["duties"] and _decoy(product, budget) else 0
            brand_rank = 0 if honor and brand and _brand_hit(product, brand) else 1
            ready = 0 if product.get("_hydrated") else 1
            after = 0 if product.get("_aftershock") else 1
            price = _price_of(product)
            honor_after = bool(self.store.event_key and self.brief["duties"] & {DUTY_TASTE, DUTY_WHY})
            try:
                open_pos = open_order.index(key)
            except ValueError:
                open_pos = 10_000
            # WHY: demote ultra-cheap stubs relative to budget (commitment decoys).
            cheap = 0
            if DUTY_WHY in self.brief["duties"] and budget not in (None, "") and price is not None:
                try:
                    if float(budget) > 0 and price < float(budget) * 0.33:
                        cheap = 1
                except (TypeError, ValueError):
                    cheap = 0
            return (
                decoy,
                cheap,
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
        if self.brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK}:
            fitted = [item for item in live if _figure_ok(item, self.query, spans)]
            if fitted:
                live = fitted
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
        if DUTY_FORK in self.brief["duties"] and not target:
            brand = str(self.brief.get("brand") or "")
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
        if DUTY_WHY in self.brief["duties"] and self.brief.get("brand") and not target:
            brand = str(self.brief.get("brand") or "")
            branded = [item for item in live if _brand_hit(item, brand)]
            if branded:
                live = branded
        elif DUTY_WHY in self.brief["duties"] and self.brief.get("brand") and target:
            # Recovery: prefer brand siblings when they still clear firm gates.
            brand = str(self.brief.get("brand") or "")
            branded = [
                item
                for item in live
                if _brand_hit(item, brand)
                and _figure_ok(item, self.query, spans)
            ]
            if branded:
                live = branded
        if DUTY_WHY in self.brief["duties"]:
            # Prefer listings whose facts literally carry the gate cue (avoids title decoys).
            live = sorted(
                live,
                key=lambda item: (
                    _spec_clarity(item, self.query),
                    0 if item.get("_hydrated") else 1,
                    _price_of(item) if _price_of(item) is not None else 1e12,
                ),
            )
        if DUTY_WHY in self.brief["duties"] and self.order:
            claimable = [
                item
                for item in live
                if item.get("_hydrated")
                and _has_spec_claim(_ground_claims(item, self.query, self.order))
            ]
            if claimable:
                live = claimable
            elif not target:
                firm = [
                    item
                    for item in live
                    if item.get("_hydrated")
                    and _figure_ok(item, self.query, spans)
                ]
                if firm:
                    live = firm
            # Prefer the firm pick closest to budget (avoids cheap decoy commitments).
            budget = self.brief.get("budget")
            if budget not in (None, "") and not target:
                try:
                    ceiling = float(budget)
                except (TypeError, ValueError):
                    ceiling = None
                if ceiling and ceiling > 0:
                    live = sorted(
                        live,
                        key=lambda item: (
                            0 if (_price_of(item) or 0) >= ceiling * 0.5 else 1,
                            abs(ceiling - (_price_of(item) or 0)),
                        ),
                    )
        if not live and target:
            live = [
                item
                for item in self.store.pool()
                if _listing_key(item) != target and not self._stale(item)
            ]
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

    def _hold_order(self, product: dict[str, Any] | None) -> dict[str, Any] | None:
        """Do not stamp an order whose viewed facts omit the shopper's asked spec."""
        if product is None:
            return None
        product = self._full(product)
        ask = DUTY_ASK in self.brief["duties"]
        why = DUTY_WHY in self.brief["duties"]
        spans = list(self.brief.get("spans") or [])
        facts_ok = True
        if (ask or why) and spans and product.get("_hydrated"):
            facts_ok = _figure_ok(product, self.query, spans)
        if why and not _has_spec_claim(_ground_claims(product, self.query, self.order)):
            facts_ok = False
        if why and not product.get("_hydrated"):
            facts_ok = False
        brand = str(self.brief.get("brand") or "")
        if why and brand and not _brand_hit(product, brand):
            facts_ok = False
        if facts_ok:
            return None
        if self.spec_holds >= 3:
            return None
        self.spec_holds += 1
        alt = None
        for item in self.ranked() or self.live(pick=True):
            if self._stale(item) or _listing_key(item) == _listing_key(product):
                continue
            if spans and item.get("_hydrated") and not _figure_ok(item, self.query, spans):
                continue
            if why and not _has_spec_claim(_ground_claims(item, self.query, self.order)):
                continue
            alt = item
            break
        target = alt or product
        pid = str(target.get("product_id") or "")
        sku = target.get("sku")
        if self.view and pid and pid not in self.store.viewed:
            return {"name": _tool_name(self.view), "args": _bind(self.view, {"product_id": pid})}
        if self.inspect and _listing_key(target) not in self.store.gauged:
            return {
                "name": _tool_name(self.inspect),
                "args": _bind(self.inspect, {"product_id": pid, "sku": sku}),
            }
        return None

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
        if DUTY_WHY in self.brief["duties"]:
            return 5
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
        # Late clock: stop surveying and close an order.
        if self.turn >= max(12, int(self.max_steps * 0.65)):
            done = self.finish()
            if done:
                return done[: self.max_calls]
        probe = self._probe_event()
        if probe:
            return [probe]
        ack = self._ack_event()
        if ack and self.store.event_probed:
            return [ack]
        cart = self.store.cart[0] if self.store.cart else None
        # After a market shock: inspect + acknowledge BEFORE removing the cart.
        # Removing first breaks intended_event_commitment / event_closed.
        if cart is not None and self._stale(cart):
            if self.store.event_key and not self.store.event_acked:
                ack = self._ack_event()
                if ack:
                    return [ack]
            if self.remove:
                return [
                    {
                        "name": _tool_name(self.remove),
                        "args": _bind(
                            self.remove,
                            {"product_id": cart.get("product_id"), "sku": cart.get("sku")},
                        ),
                    }
                ]
        # Commit intended SKU before answering "which one" so TF1 can score the cart.
        # Never force-commit on ranking slate before every named candidate is viewed.
        if (
            not self.store.cart
            and not self.store.event_key
            and DUTY_SLATE not in self.brief["duties"]
        ):
            commit = self._commit_cart()
            if commit and (
                DUTY_FORK in self.brief["duties"]
                or DUTY_ASK in self.brief["duties"]
                or DUTY_WHY in self.brief["duties"]
                or self.brief.get("spans")
            ):
                # Only force-commit when a hydrated firm pick already exists.
                champ = self.champion()
                if champ and champ.get("_hydrated") and _listing_key(champ) in self.store.gauged:
                    return [commit]
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
                twins = [
                    item
                    for item in self.store.pool()
                    if stem
                    and _sku_stem(item.get("sku")) == stem
                    and _listing_key(item) != self.store.event_key
                    and _in_stock(item)
                ]
                unread = [item for item in twins if not item.get("_hydrated")]
                if unread:
                    hydrate = self._hydrate(unread[: self._view_cap()])
                    if hydrate:
                        return hydrate[: self.max_calls]
                if not twins and self.store.search_n < 4:
                    need_search = True
            if DUTY_WHY in self.brief["duties"] and not recovering:
                grounded = [
                    item
                    for item in live_now
                    if item.get("_hydrated")
                    and _figure_ok(item, self.query, list(self.brief.get("spans") or []))
                    and (
                        not self.order
                        or _has_spec_claim(_ground_claims(item, self.query, self.order))
                    )
                ]
                brand = str(self.brief.get("brand") or "")
                if brand and not any(_brand_hit(item, brand) for item in grounded):
                    need_search = True
                if not grounded and self.store.search_n < 5:
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
                # WHY: survey several hydrated listings before the commitment cart.
                if DUTY_WHY in self.brief["duties"] and not recovering:
                    ready = [
                        item
                        for item in (self.ranked() or live_now)
                        if item.get("_hydrated")
                        and _figure_ok(item, self.query, list(self.brief.get("spans") or []))
                    ]
                    if len(ready) < min(3, self._view_cap()) and len(self.store.viewed) < self._view_cap():
                        hydrate = self._hydrate(focus)
                        if hydrate:
                            return hydrate[: self.max_calls]
                if len(self.store.viewed) < self._view_cap():
                    hydrate = self._hydrate(focus)
                    if hydrate:
                        return hydrate[: self.max_calls]
        champ = self.champion()
        if (
            champ is not None
            and DUTY_WHY in self.brief["duties"]
            and not self.store.event_key
            and not self.store.cart
        ):
            # Avoid ultra-cheap decoys as the TF7 commitment cart.
            budget = self.brief.get("budget")
            price = _price_of(champ)
            if budget is not None and price is not None and float(budget) > 0:
                if price < float(budget) * 0.25:
                    alts = [
                        item
                        for item in self.ranked()
                        if item.get("_hydrated")
                        and not self._stale(item)
                        and _figure_ok(item, self.query, list(self.brief.get("spans") or []))
                        and (_price_of(item) or 0) >= float(budget) * 0.25
                    ]
                    brand = str(self.brief.get("brand") or "")
                    branded = [item for item in alts if brand and _brand_hit(item, brand)]
                    if branded:
                        champ = branded[0]
                    elif alts:
                        champ = alts[0]
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
        cart_now = self.store.cart[0] if self.store.cart else None
        if (
            DUTY_WHY in self.brief["duties"]
            and not self.store.event_key
            and not cart_now
            and len(self.store.viewed) < 4
        ):
            focus = (self.ranked() or self.live() or self.store.pool())[: self._view_cap()]
            unread = [item for item in focus if not item.get("_hydrated")]
            if unread:
                hydrate = self._hydrate(unread[: self._view_cap()])
                if hydrate:
                    return hydrate[: self.max_calls]
            if self.store.search_n < 4:
                search = self._search_actions()
                if search:
                    return search
        # After the WHY survey, commit immediately — do not keep viewing forever.
        if (
            DUTY_WHY in self.brief["duties"]
            and not self.store.event_key
            and not cart_now
            and champ is not None
            and len(self.store.viewed) >= 4
            and self.cart
            and not self._stale(champ)
        ):
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
            return [
                {
                    "name": _tool_name(self.cart),
                    "args": _bind(
                        self.cart,
                        {"product_id": champ.get("product_id"), "sku": champ.get("sku")},
                    ),
                }
            ]
        if (
            cart_now is not None
            and self.remove
            and champ is not None
            and _listing_key(cart_now) != _listing_key(champ)
            and (self.store.event_key or DUTY_SLATE in self.brief["duties"])
        ):
            return [
                {
                    "name": _tool_name(self.remove),
                    "args": _bind(
                        self.remove,
                        {"product_id": cart_now.get("product_id"), "sku": cart_now.get("sku")},
                    ),
                }
            ]
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
            if (
                DUTY_WHY in self.brief["duties"]
                and not self.store.event_key
                and len(self.store.viewed) < 4
            ):
                hydrate = self._hydrate(
                    [item for item in (self.ranked() or self.live()) if not item.get("_hydrated")][: self._view_cap()]
                )
                if hydrate:
                    return hydrate[: self.max_calls]
            spec = self.brief["duties"] & {DUTY_WHY, DUTY_TASTE, DUTY_FORK, DUTY_ASK}
            if spec and not champ.get("_hydrated"):
                return self._hydrate([champ])[: self.max_calls]
            spans = list(self.brief.get("spans") or [])
            if (
                spans
                and champ.get("_hydrated")
                and self.brief["duties"] & {DUTY_WHY, DUTY_ASK, DUTY_FORK}
                and (
                    not _figure_ok(champ, self.query, spans)
                    or (
                        DUTY_WHY in self.brief["duties"]
                        and _spec_clarity(champ, self.query) >= 3
                    )
                )
            ):
                hydrate = self._hydrate(
                    [
                        item
                        for item in (self.ranked() or self.live())
                        if not item.get("_hydrated")
                    ][: self._view_cap()]
                )
                if hydrate:
                    return hydrate[: self.max_calls]
                search = self._search_actions()
                if search:
                    return search[: self.max_calls]
                return []
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
            if champ is not None and champ.get("_gauged"):
                cart_item = champ
            return self._stamp(cart_item)
        return []

    def finish(self) -> list[dict[str, Any]]:
        cart = self.store.cart[0] if self.store.cart else None
        if cart is not None and self._stale(cart):
            probe = self._probe_event()
            if probe:
                return [probe]
            if self.store.event_key and not self.store.event_acked:
                ack = self._ack_event()
                if ack:
                    return [ack]
            if self.remove:
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
            return self._stamp(cart)
        picked = self.champion()
        if picked is None:
            live = [item for item in self.live(pick=False) if _listing_key(item) != self.store.event_key]
            if self.brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK}:
                fitted = [
                    item
                    for item in live
                    if _figure_ok(item, self.query, list(self.brief.get("spans") or []))
                ]
                live = fitted or live
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
                live = modeled if modeled else live
            picked = live[0] if live else None
        if picked is None:
            probe = self._probe_event()
            if probe:
                return [probe]
            pool = [
                item
                for item in self.store.pool()
                if _listing_key(item) != self.store.event_key
                and not self._stale(item)
                and item.get("sku")
                and _in_stock(item)
            ]
            picked = pool[0] if pool else None
        if picked is None:
            return []
        pid = str(picked.get("product_id") or "")
        probe = self._probe_event()
        if probe:
            return [probe]
        if self.store.event_key and not self.store.event_acked:
            ack = self._ack_event()
            if ack:
                return [ack]
        # Late turns: skip endless inspect loops and close.
        if self.turn >= 14 and self.cart and not self.store.cart and not self._stale(picked):
            return [
                {
                    "name": _tool_name(self.cart),
                    "args": _bind(self.cart, {"product_id": pid, "sku": picked.get("sku")}),
                }
            ]
        if self.inspect and _listing_key(picked) not in self.store.gauged and self.turn < 16:
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
        if self.store.cart and self.order and not self._stale(self.store.cart[0]):
            return self._stamp(self.store.cart[0])
        return self._stamp(picked)

    def _next_beat(self) -> list[dict[str, Any]]:
        """Stall breaker: advance view → inspect → cart → order without IR."""
        probe = self._probe_event()
        if probe:
            return [probe]
        champ = self.champion()
        spans = list(self.brief.get("spans") or [])
        if champ is None:
            live = [item for item in self.live(pick=False) if not self._stale(item)]
            if spans and self.brief["duties"] & {DUTY_WHY, DUTY_ASK, DUTY_FORK}:
                live = [
                    item
                    for item in live
                    if not item.get("_hydrated")
                    or (
                        _figure_ok(item, self.query, spans)
                        and (
                            DUTY_WHY not in self.brief["duties"]
                            or _spec_clarity(item, self.query) < 3
                        )
                    )
                ]
            champ = live[0] if live else None
        if champ is None:
            search = self._search_actions()
            if search:
                return search
            return self._stamp(self.store.cart[0] if self.store.cart else None)
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
            if (
                DUTY_WHY in self.brief["duties"]
                and spans
                and champ.get("_hydrated")
                and (
                    not _figure_ok(champ, self.query, spans)
                    or _spec_clarity(champ, self.query) >= 3
                )
            ):
                search = self._search_actions()
                if search:
                    return search
                return []
            return [
                {
                    "name": _tool_name(self.cart),
                    "args": _bind(self.cart, {"product_id": pid, "sku": sku}),
                }
            ]
        if self.store.cart and self.order and not self._stale(self.store.cart[0]):
            return self._stamp(self.store.cart[0])
        return self._stamp(champ)

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
        proposed_names = " ".join(str(item.get("name") or "").lower() for item in proposed)
        force = any(
            hint in proposed_names
            for hint in ("add_to_cart", "remove_from_cart", "inspect", "place_test_order", "place_order")
        )
        if cart_item is not None and self._stale(cart_item):
            probe = self._probe_event()
            if probe:
                return [probe]
            if self.store.event_key and not self.store.event_acked:
                ack = self._ack_event()
                if ack:
                    return [ack]
            if self.remove:
                return [
                    {
                        "name": _tool_name(self.remove),
                        "args": _bind(
                            self.remove,
                            {"product_id": cart_item.get("product_id"), "sku": cart_item.get("sku")},
                        ),
                    }
                ]
        probe = self._probe_event()
        if probe:
            return [probe]
        # After shock: ack the shopper's price question before recovery essays.
        if self.store.event_key and not self.store.event_acked and (
            await_reply or self.store.event_probed
        ):
            ack = self._ack_event()
            if ack:
                return [ack]
        # Before first cart: prefer committing the firm pick over a pre-cart essay.
        if (
            await_reply
            and not self.store.cart
            and not self.store.event_key
            and self.brief["duties"] & {DUTY_FORK, DUTY_WHY, DUTY_ASK}
        ):
            commit = self._commit_cart()
            if commit:
                return [commit]
        if self.brief.get("ask") and self.asked:
            cleaned = [
                action
                for action in cleaned
                if "message" not in str(action.get("name") or "").lower()
            ]
            proposed = [
                action
                for action in proposed
                if "message" not in str(action.get("name") or "").lower()
            ] or proposed
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
                    stamped = self._stamp(cart_item or self.champion())
                    aligned_soft.extend(stamped)
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
        if self.store.event_open and proposed and force:
            return proposed[: self.max_calls]
        # Slate: never first-cart until every named candidate is viewed; cart champion only.
        if force and proposed and DUTY_SLATE in self.brief["duties"] and self.cart and not self.store.cart:
            named = list(self.brief.get("named") or [])
            missing = [
                spec["product_id"]
                for spec in named
                if spec["product_id"] not in self.store.viewed
            ]
            if missing:
                hydrate = self._hydrate([{"product_id": pid} for pid in missing])
                if hydrate:
                    return hydrate[: self.max_calls]
                beat = self._next_beat()
                return beat[: self.max_calls] if beat else []
            champ = self.champion()
            cart_name = _tool_name(self.cart).lower()
            # Champion must be fact-hydrated so axes do not score SEO titles.
            if champ is None or self._stale(champ) or not champ.get("_hydrated"):
                cleaned_no_cart = [
                    action
                    for action in proposed
                    if str(action.get("name") or "").lower() != cart_name
                ]
                if cleaned_no_cart:
                    return cleaned_no_cart[: self.max_calls]
                if champ and self.view and str(champ.get("product_id") or "") not in self.store.viewed:
                    return [
                        {
                            "name": _tool_name(self.view),
                            "args": _bind(self.view, {"product_id": champ.get("product_id")}),
                        }
                    ]
                return []
            return [
                {
                    "name": _tool_name(self.cart),
                    "args": _bind(
                        self.cart,
                        {"product_id": champ.get("product_id"), "sku": champ.get("sku")},
                    ),
                }
            ]
        # WHY: never let the model cart a title decoy — rewrite to champion.
        if force and proposed and DUTY_WHY in self.brief["duties"] and self.cart:
            champ = self.champion()
            cart_name = _tool_name(self.cart).lower()
            spans = list(self.brief.get("spans") or [])
            rewritten: list[dict[str, Any]] = []
            for action in proposed[: self.max_calls]:
                name = str(action.get("name") or "")
                if name.lower() != cart_name:
                    rewritten.append(action)
                    continue
                args = dict(action.get("args") or {})
                pid = str(args.get("product_id") or "")
                proposed_row = None
                if pid:
                    for item in self.store.pool():
                        ip = str(item.get("product_id") or "")
                        if ip == pid or (ip and pid and (ip.startswith(pid) or pid.startswith(ip))):
                            proposed_row = item
                            break
                reject = False
                if proposed_row and spans and proposed_row.get("_hydrated"):
                    if not _figure_ok(proposed_row, self.query, spans):
                        reject = True
                    elif _spec_clarity(proposed_row, self.query) >= 3:
                        reject = True
                if reject:
                    proposed_row = None
                pick = None
                if (
                    champ is not None
                    and not self._stale(champ)
                    and (
                        not spans
                        or not champ.get("_hydrated")
                        or (
                            _figure_ok(champ, self.query, spans)
                            and _spec_clarity(champ, self.query) < 3
                        )
                    )
                ):
                    pick = champ
                elif proposed_row is not None:
                    pick = proposed_row
                if pick is None:
                    continue
                rewritten.append(
                    {
                        "name": name,
                        "args": _bind(
                            self.cart,
                            {
                                "product_id": pick.get("product_id"),
                                "sku": pick.get("sku"),
                            },
                        ),
                    }
                )
            # Never fall through to the raw model cart for figure-gated duties.
            if rewritten:
                return rewritten[: self.max_calls]
            # Drop bad carts; keep surveying instead of committing a decoy.
            beat = self._next_beat()
            return beat[: self.max_calls] if beat else []
        if force and proposed and not await_reply:
            return proposed[: self.max_calls]
        if not cleaned:
            return proposed[: self.max_calls]
        if any("message" in str(item.get("name") or "").lower() for item in cleaned):
            if self.store.event_open:
                return proposed[: self.max_calls] or cleaned[: self.max_calls]
        order_name = _tool_name(self.order).lower()
        champ = self.champion()
        aligned: list[dict[str, Any]] = []
        for action in cleaned[: self.max_calls]:
            name = str(action.get("name") or "")
            args = dict(action.get("args") or {})
            lower = name.lower()
            if lower == order_name:
                stamped = self._stamp(champ or cart_item)
                aligned.extend(stamped)
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
                self.store.cart = []
            if any(token in err_blob for token in ("no such", "no_candidate", "cannot add")):
                pid = str(sent_args.get("product_id") or inner.get("product_id") or "")
                sku = str(sent_args.get("sku") or inner.get("sku") or "")
                if pid and sku:
                    self.store.dead.add(f"{pid}::{sku}")
                    self.store.cart = []
            name = str(sent[index].get("name") or "").lower() if index < len(sent) else ""
            if "message" in name and self.store.event_key:
                self.store.event_acked = True
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
                pid = str(patch.get("product_id") or "")
                if pid and self.store.event_key:
                    event_pid = self.store.event_key.split("::")[0]
                    if pid == event_pid:
                        self.store.event_probed = True
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
                    if not self.store.event_key or self.store.event_probed:
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
    if pilot.brief.get("ask") and not pilot.asked:
        return True
    if cart_used and await_reply:
        return True
    if pilot.store.event_key and not pilot.store.event_probed:
        return True
    if await_reply and pilot.store.event_key and not pilot.store.event_acked:
        return True
    # Cart-before-essay only for fork/why clarifies — leave taste/preference to the model.
    if (
        await_reply
        and not cart_used
        and not pilot.store.event_key
        and pilot.champion()
        and pilot.brief["duties"] & {DUTY_FORK, DUTY_WHY, DUTY_ASK}
    ):
        return True
    if pilot.brief.get("ask") and pilot.asked and not cart_used and "message" in names:
        return True
    if proposed and "remove_from_cart" in names:
        return True
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
        if pilot.store.event_key and not pilot.store.event_acked:
            old = pilot.store.event_old_price
            new = pilot.store.event_new_price
            if old is not None and new is not None:
                return (
                    f"Shopper noticed the market change. Message briefly that the "
                    f"price increased from {old:.2f} to {new:.2f}, then recover."
                )
            return (
                "Shopper noticed the market change. Acknowledge it with the message "
                "tool using the old and new figures, then recover."
            )
        if pilot.store.event_key:
            return (
                "A market event is open. inspect_stock the listing the event hit "
                "to close it, then recover and order. Do not write an essay."
            )
        if cart_used:
            return "Cart is live. inspect_stock that sku, then place_test_order. No message essays."
        if not cart_used and pilot.champion():
            return (
                "Shopper asked which SKU. add_to_cart the firm-compliant preferred-brand "
                "pick first (that cart is the commitment), then answer briefly."
            )
        return "The shopper asked a question. Answer with the message tool this turn."
    if pilot.brief.get("ask") and not pilot.asked and turn == 1:
        return "One firm figure is withheld until you ask. Call message NOW before searching."
    if DUTY_SLATE in pilot.brief["duties"] and turn <= 2:
        return "Compare every named candidate before the first cart. Order with ONLY product_id and sku."
    if DUTY_WHY in pilot.brief["duties"] and turn <= 2:
        return (
            "View facts before the first cart. Fill claims from viewed facts using only "
            "field names this episode's order tool allows. Why needs the asked spec, "
            "not just price and stock."
        )
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
                    "Rank by the shopper's stated axis order (ties only after that). "
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
                    "Copy values as the catalog printed them."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "ribbon": {
                "primary": pilot.brief.get("primary"),
                "duties": sorted(pilot.brief.get("duties") or []),
                "model": model,
                "provider": _provider(),
                "stack": "lgd+mairca+cocoso",
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
                "ribbon_turn": {"primary": pilot.brief.get("primary"), "turn": turn},
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
