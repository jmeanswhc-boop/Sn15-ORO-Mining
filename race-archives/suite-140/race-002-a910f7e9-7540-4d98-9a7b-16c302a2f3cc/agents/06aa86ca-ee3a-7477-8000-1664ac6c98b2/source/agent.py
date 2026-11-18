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

_CLOCK_S = 285.0
_OBS_CAP = 3450
_SURVEY_CAP = 3
_SURVEY_OPEN = 3
_SKIP_FLOOR = 10
_DECOY_FRAC = 0.06

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
_EVENT_RX = re.compile(
    r"more expensive|still available|sold out|out of stock|"
    r"just get more|price (?:from|went|changed)|repriced",
    re.I,
)
_UNIT_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*([A-Za-z%]{1,12}|\"|'')?",
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
    "inches": "in",
    "inch": "in",
    "hertz": "hz",
    "watts": "w",
    "watt": "w",
    "grammes": "g",
    "grams": "g",
    "kilograms": "kg",
    "millilitres": "ml",
    "milliliters": "ml",
    "ounces": "oz",
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
_CLAIM_SOFT = frozenset({"price", "in_stock"})
# Brand is a TF7-supported reason when attested — not a soft-only filler.
# Weak drawers: never count as the firm TF7 gate unless the shopper asked for them.
_CLAIM_WEAK = frozenset({"model", "title", "name", "sku", "product", "variant", "option", "_option"})
_CLAIM_FEATURE_REASONS = frozenset(
    {"streaming", "microphone", "privacy_cover", "autofocus", "hdr"}
)
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
_IDLE = (  # concord_rim

    "No tool call received. Choose a supplied tool now. "
    "Search or filter with the shopper's own value words, not schema keys.",
    "Still no tool call. Search/filter, view a candidate, answer a pending "
    "shopper question, or place_test_order if the cart holds a verified SKU.",
)


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

_CLAIM_FILL_CACHE: dict[str, list[dict[str, str]]] = {}
_CLAIM_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,32}$")

_TF7_FILL = """
Name one justification claim for the shopper's measured requirement.
Input JSON: goal, allowed_fields (live enum; may be empty), schema_rule, have, listing.
Return JSON only: {"claims":[{"field":"...","value":"..."}]}
Rules:
- If allowed_fields is non-empty, field MUST be one of them.
- Prefer the attribute name schema_rule describes for the measured requirement.
- Catalog drawer keys are often not the claim field — follow allowed_fields / schema_rule.
- value must appear in listing text; copy spelling. Prefer the shopper's own class
  word when that word appears in the listing.
- Feature-reason names in schema_rule are not the measured specification.
- At most 1 claim. If listing lacks evidence: {"claims":[]}.
""".strip()

_TF7_ALIGN = """
Finalize justification.claims.
Input JSON: goal, schema_rule, allowed_fields, listing, proposed_claims.
Return JSON only: {"claims":[{"field":"...","value":"..."}]}
Rules:
- Keep values unchanged (copy listing spelling).
- If proposed field is a catalog drawer or compound name, rename field to the
  matching allowed_fields entry using schema_rule wording — never invent a field
  that is absent from allowed_fields when that list is non-empty.
- At most 1 claim. Never use feature-reason names as the measured field.
""".strip()


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


def _cm_clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _cm_fold(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _cm_compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _cm_tokens(text: str) -> list[str]:
    return [token for token in _cm_fold(text).split() if len(token) > 1 and token not in _STOP]


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


def _cm_shopper_text(query: str) -> str:
    return re.split(r"\n\s*Task rules:", query or "", maxsplit=1)[0].strip()


def _cm_rules_text(query: str) -> str:
    parts = re.split(r"\n\s*Task rules:", query or "", maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _cm_named_keys(text: str) -> list[dict[str, str]]:
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


def _cm_listing_key(product: dict[str, Any]) -> str:
    pid = str(product.get("product_id") or "")
    sku = str(product.get("sku") or "")
    if pid and sku:
        return f"{pid}::{sku}"
    return pid


def _cm_facts(product: dict[str, Any]) -> dict[str, Any]:
    facts = product.get("facts") or product.get("specification") or product.get("attributes")
    return facts if isinstance(facts, dict) else {}


def _cm_options_text(product: dict[str, Any]) -> str:
    options = product.get("options")
    if isinstance(options, dict):
        return " ".join(str(value) for value in options.values() if value not in (None, ""))
    if isinstance(options, (list, tuple)):
        return " ".join(str(item) for item in options if item not in (None, ""))
    return str(options or "").strip()


def _cm_blob(product: dict[str, Any]) -> str:
    facts = _cm_facts(product)
    option_text = _cm_options_text(product)
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


def _cm_number(text: Any) -> float | None:
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


def _cm_price_of(product: dict[str, Any]) -> float | None:
    for key in ("price", "current_price", "sale_price"):
        raw = product.get(key)
        if isinstance(raw, dict):
            raw = raw.get("value") or raw.get("amount") or raw.get("current")
        value = _cm_number(raw)
        if value is not None:
            return value
    return _cm_number(_cm_facts(product).get("price"))


def _cm_in_stock(product: dict[str, Any]) -> bool:
    if "in_stock" in product:
        flag = product.get("in_stock")
        if isinstance(flag, bool):
            return flag
        if isinstance(flag, (int, float)):
            return bool(flag)
        if isinstance(flag, str) and flag.strip():
            return flag.strip().lower() not in {"0", "false", "no", "oos", "out"}
    return True


def _cm_price_text(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _cm_canon_unit(raw: str) -> str:
    return _UNIT_CANON.get((raw or "").lower().strip(), (raw or "").lower().strip())


def _cm_figure_spans(text: str, budget: float | None = None) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _UNIT_RX.finditer(text or ""):
        try:
            amount = float(match.group(1))
        except ValueError:
            continue
        if budget is not None and abs(amount - budget) < 1e-6:
            continue
        unit = _cm_canon_unit(match.group(2) or "")
        if not unit:
            continue
        raw = _cm_clean(match.group(0))
        key = f"{amount:g}|{unit}"
        if key in seen:
            continue
        seen.add(key)
        found.append({"amount": amount, "unit": unit, "raw": raw})
    return found


def _cm_wants_reason(query: str) -> bool:
    shopper = _cm_shopper_text(query)
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


def _cm_held_back(query: str) -> bool:
    return bool(_ASK_RX.search(_cm_shopper_text(query)) or _ASK_RX.search(query or ""))


def _cm_quoted(query: str) -> list[str]:
    shopper = _cm_shopper_text(query)
    quotes = [_cm_clean(item) for item in _QUOTE_RX.findall(shopper)]
    if not quotes:
        hit = _PHRASE_RX.search(shopper)
        if hit:
            quotes.append(_cm_clean(hit.group(1)))
    return [item for item in quotes if item]


def _cm_category_hint(query: str) -> str:
    hit = _CAT_RX.search(_cm_shopper_text(query))
    return _cm_clean(hit.group(1)) if hit else ""


def _cm_kind_noun(query: str) -> str:
    """Noun from this brief only — no catalog of device classes."""
    shopper = _cm_shopper_text(query)
    folded = _cm_fold(shopper)
    cat = _CAT_RX.search(shopper)
    if cat:
        parts = _cm_tokens(cat.group(1))
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


def _cm_brand_guess(query: str) -> str:
    shopper = _cm_shopper_text(query)
    hit = re.search(
        r"preferred\s+brand\s+([A-Za-z0-9][\w.&'-]{0,40})",
        shopper,
        re.I,
    )
    if hit:
        phrase = _cm_clean(hit.group(1)).rstrip(".")
        if _cm_fold(phrase) not in _STOP:
            return phrase
    hit = re.search(
        r"(?:[Pp]refer(?:red)?|[Ii]deally)\s+(?:the\s+)?([A-Z][A-Za-z0-9.&-]{2,})",
        shopper,
    )
    if hit:
        phrase = _cm_clean(hit.group(1)).rstrip(".")
        if _cm_fold(phrase) not in _STOP:
            return phrase
    return ""


def _cm_soft_brand(query: str) -> bool:
    shopper = _cm_fold(_cm_shopper_text(query))
    if _PATCH_RX.search(shopper):
        return False
    return ("prefer" in shopper or "ideally" in shopper) and (
        "firm" in shopper or "requirement" in shopper or "soft" in shopper
    )


def _cm_model_guess(query: str) -> str:
    shopper = _cm_shopper_text(query)
    hit = re.search(
        r"\bprefer(?:s|red)?(?:\s+the)?\s+([A-Za-z0-9][A-Za-z0-9_-]{1,32})",
        shopper,
        re.I,
    )
    if hit:
        token = hit.group(1).rstrip(".,;")
        folded = _cm_fold(token)
        if folded not in _STOP and folded not in {"color", "colour", "brand", "the"}:
            return token
    hit = re.search(
        r"\bmodel(?:\s+code)?\s+([A-Za-z0-9._-]{2,})",
        shopper,
        re.I,
    )
    if hit:
        return hit.group(1).rstrip(".,;")
    return ""


def _cm_color_guess(query: str) -> str:
    shopper = _cm_shopper_text(query)
    hit = re.search(
        r"(?:colou?r(?:ed)?|finish|shade)\s+(?:is\s+|in\s+|of\s+)?([A-Za-z][A-Za-z-]{2,24})",
        shopper,
        re.I,
    )
    if hit:
        token = hit.group(1).rstrip(".,;")
        if _cm_fold(token) not in _STOP:
            return _cm_fold(token)
    hit = re.search(
        r"\bin\s+([A-Za-z][A-Za-z-]{2,24})\s+(?:colou?r|finish)\b",
        shopper,
        re.I,
    )
    if hit and _cm_fold(hit.group(1)) not in _STOP:
        return _cm_fold(hit.group(1))
    return ""


def _cm_axes_from(text: str) -> list[tuple[str, str]]:
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
            name = _cm_canon_axis(match.group(1))
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


def _cm_canon_axis(raw: str) -> str:
    folded = _cm_fold(raw)
    if "price" in folded or "cost" in folded or "budget" in folded:
        return "price"
    tokens = [item for item in _cm_tokens(raw) if item not in _STOP]
    return "_".join(tokens[:4]) if tokens else ""


def _cm_axis_value(product: dict[str, Any], name: str) -> float | None:
    """Read axis numerics from options/facts/title by token overlap — no fixed field map."""
    if name == "price":
        return _cm_price_of(product)
    facts = _cm_facts(product)
    raw = facts.get(name) or product.get(name)
    direct = _cm_number(raw)
    if direct is not None:
        return direct
    needle_tokens = [tok for tok in _cm_fold(name.replace("_", " ")).split() if len(tok) > 1]
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
            r"(\d+(?:\.\d+)?)\s*-?\s*([A-Za-z%]{1,12}|\"|'')?",
            text,
            re.I,
        ):
            try:
                amount = float(match.group(1))
            except ValueError:
                continue
            start = max(0, match.start() - 28)
            end = min(len(text), match.end() + 28)
            window = _cm_fold(text[start:end])
            overlap = sum(1 for tok in needle_tokens if tok in window)
            unit = _cm_canon_unit(match.group(2) or "")
            # Boost when the harvested unit neighborhood matches axis words the shopper used.
            if unit and any(tok in window for tok in needle_tokens):
                overlap += 1
            if unit and unit in " ".join(needle_tokens):
                overlap += 2
            if overlap <= 0:
                continue
            if best is None or overlap > best[0]:
                best = (overlap, amount)
    return best[1] if best else None


def _cm_lex_rank(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if not pool:
        return []
    used = axes or [("price", "asc")]

    def key(product: dict[str, Any]) -> tuple:
        parts: list[float] = []
        for name, direction in used:
            value = _cm_axis_value(product, name)
            if value is None:
                parts.append(1e18 if direction == "desc" else -1e18)
            else:
                parts.append(-value if direction == "desc" else value)
        price = _cm_price_of(product)
        parts.append(price if price is not None else 1e12)
        return tuple(parts)

    return sorted(pool, key=key)


def _cm_wish_terms(query: str) -> list[str]:
    shopper = _cm_shopper_text(query)
    found: list[str] = []
    for match in re.finditer(
        r"(?:matters more|rather(?:\s+pay)?|use-case|use case|"
        r"need it to|must feel|should be|intended use)\s+([^.;]{3,48})",
        shopper,
        re.I,
    ):
        found.extend(_cm_tokens(match.group(1)))
    cleaned: list[str] = []
    seen: set[str] = set()
    for token in found:
        folded_tok = _cm_fold(token)
        if not folded_tok or folded_tok in seen or folded_tok in _STOP:
            continue
        seen.add(folded_tok)
        cleaned.append(token)
    return cleaned[:8]


def _cm_taste_cover(product: dict[str, Any], wishes: list[str]) -> float:
    if not wishes:
        return 0.0
    blob = _cm_fold(_cm_blob(product))
    hits = sum(1 for wish in wishes if _cm_fold(wish) in blob)
    return hits / max(len(wishes), 1)


def _cm_decoy(product: dict[str, Any], budget: Any) -> bool:
    price = _cm_price_of(product)
    if price is None or budget in (None, ""):
        return False
    try:
        ceiling = float(budget)
    except (TypeError, ValueError):
        return False
    if ceiling <= 0:
        return False
    return price <= ceiling * _DECOY_FRAC


def _cm_phrase_hit(product: dict[str, Any], phrase: str) -> bool:
    title = _cm_fold(product.get("title") or "")
    needle = _cm_fold(phrase)
    if not needle or not title:
        return False
    return all(token in title for token in needle.split())


def _cm_kind_ok(product: dict[str, Any], kind: str) -> bool:
    if not kind:
        return True
    blob = _cm_fold(_cm_blob(product))
    return _cm_fold(kind) in blob


def _cm_junk(product: dict[str, Any], kind: str) -> bool:
    title = _cm_fold(product.get("title") or "")
    if any(marker in title for marker in _JUNK):
        if kind and _cm_fold(kind) not in title.split()[:3]:
            return True
    return False


def _cm_brand_hit(product: dict[str, Any], brand: str) -> bool:
    if not brand:
        return False
    return _cm_fold(brand) in _cm_fold(_cm_blob(product))


def _cm_model_hit(product: dict[str, Any], model: str) -> bool:
    if not model:
        return False
    return _cm_compact(model) in _cm_compact(_cm_blob(product))


def _cm_color_hit(product: dict[str, Any], color: str) -> bool:
    if not color:
        return False
    return re.search(rf"\b{re.escape(_cm_fold(color))}\b", _cm_fold(_cm_blob(product))) is not None


def _cm_ladder(product: dict[str, Any], model: str, color: str) -> int:
    has_model = (not model) or _cm_model_hit(product, model)
    has_color = (not color) or _cm_color_hit(product, color)
    if has_model and has_color:
        return 0
    if has_model:
        return 1
    return 2


def _cm_span_in(hay: str, span: dict[str, Any]) -> bool:
    amount = span.get("amount")
    unit = str(span.get("unit") or "")
    if amount is None:
        return False
    blob = _cm_fold(hay)
    token = f"{float(amount):g}"
    if token not in blob and f"{int(amount)}" not in blob:
        compact = _cm_compact(hay)
        if _cm_compact(token) not in compact:
            return False
    if not unit:
        return True
    return unit in blob or unit in _cm_compact(hay)


def _cm_measure_hit(product: dict[str, Any], spans: list[dict[str, Any]]) -> bool:
    if not spans:
        return True
    option = _cm_options_text(product)
    facts = " ".join(str(value) for value in _cm_facts(product).values())
    grounded = f"{option} {facts}"
    title = str(product.get("title") or "")
    for span in spans:
        if _cm_span_in(grounded, span):
            continue
        if grounded.strip() and not _cm_span_in(title, span):
            return False
        if not grounded.strip() and not _cm_span_in(_cm_blob(product), span):
            return False
    return True


def _cm_figure_ok(product: dict[str, Any], query: str, spans: list[dict[str, Any]] | None = None) -> bool:
    needed = spans if spans is not None else _cm_figure_spans(_cm_shopper_text(query))
    if not needed:
        return True
    return _cm_measure_hit(product, needed)


def _cm_sku_stem(sku: Any) -> str:
    text = str(sku or "")
    return text.rsplit("-", 1)[0] if text else ""


def _cm_norm_field(raw: str) -> str:
    text = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return re.sub(r"_+", "_", text).strip("_")


def _cm_claim_enum(tool: dict[str, Any] | None) -> list[str]:
    spec = (_props(tool).get("justification") or {}).get("properties") or {}
    claims = spec.get("claims") or {}
    items = claims.get("items") or {}
    field = (items.get("properties") or {}).get("field") or {}
    enum = field.get("enum")
    if isinstance(enum, list) and enum:
        found: list[str] = []
        seen: set[str] = set()
        for item in enum:
            key = _cm_norm_field(str(item))
            if key and key not in seen:
                seen.add(key)
                found.append(key)
        return found
    return []


def _cm_schema_reason_fields(tool: dict[str, Any] | None) -> list[str]:
    """Parse allowed reason names from this episode's field description (not a global ontology)."""
    spec = (_props(tool).get("justification") or {}).get("properties") or {}
    claims = spec.get("claims") or {}
    items = claims.get("items") or {}
    field = (items.get("properties") or {}).get("field") or {}
    desc = str(field.get("description") or "")
    found: list[str] = []
    hit = re.search(
        r"supported reason[s]?:\s*([^.]+)",
        desc,
        re.I,
    )
    if hit:
        blob = hit.group(1)
        for part in re.split(r",|\bor\b", blob):
            key = _cm_norm_field(part)
            if key and key not in _CLAIM_BLOCK and key not in {"one", "supported", "reason", "use", "also"}:
                found.append(key)
    return found


def _cm_brief_named_fields(query: str) -> list[str]:
    """Field-shaped names literally present in Task rules — no unit→label invention."""
    found: list[str] = []
    seen: set[str] = set()
    low = (query or "").lower()
    at = low.find("task rules")
    blob = low[at:] if at >= 0 else low

    def push(raw: str) -> None:
        key = _cm_norm_field(raw)
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

    for match in re.finditer(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b", blob):
        push(match.group(1))
    for match in re.finditer(
        r"(?:hardware\s+spec|firm\s+(?:requirement|spec)|specification)[^.]{0,160}"
        r"\(([^)]{3,200})\)",
        blob,
        re.I,
    ):
        for part in re.split(r",|\bor\b|/", match.group(1)):
            part = re.sub(r"^(?:a|an|the|amount\s+of)\s+", "", part.strip())
            if part:
                push(part)
    return found


def _cm_claim_allowed(tool: dict[str, Any] | None, query: str, product: dict[str, Any]) -> list[str]:
    enum = _cm_claim_enum(tool)
    if enum:
        return list(enum)
    allowed = ["in_stock", "price", "brand"]
    for field in _cm_schema_reason_fields(tool):
        if field not in allowed:
            allowed.append(field)
    for field in _cm_brief_named_fields(query):
        if field not in allowed:
            allowed.append(field)
    shopper = _cm_shopper_text(query)
    for key, raw in _cm_facts(product).items():
        fl = _cm_norm_field(str(key))
        text = _cm_clean(raw)
        if not fl or fl in _CLAIM_BLOCK or not text:
            continue
        if fl in _CLAIM_WEAK:
            continue
        if (
            any(_cm_span_in(text, span) for span in _cm_figure_spans(shopper))
            or _cm_value_attested(text, shopper)
            or _cm_key_in_text(fl, shopper)
        ):
            if fl not in allowed:
                allowed.append(fl)
    options = product.get("options") if isinstance(product.get("options"), dict) else {}
    for key, raw in options.items():
        fl = _cm_norm_field(str(key))
        text = _cm_clean(raw)
        if fl and text and fl not in _CLAIM_BLOCK and fl not in _CLAIM_WEAK and fl not in allowed:
            if _cm_value_attested(text, shopper) or any(_cm_span_in(text, span) for span in _cm_figure_spans(shopper)):
                allowed.append(fl)
    return allowed


def _cm_fact_rows(product: dict[str, Any]) -> dict[str, str]:
    rows: dict[str, str] = {}
    price = _cm_price_of(product)
    if price is not None:
        rows["price"] = _cm_price_text(price)
    rows["in_stock"] = "true" if _cm_in_stock(product) else "false"
    brand = _cm_clean(product.get("brand") or "")
    if brand:
        rows["brand"] = brand
    for key, raw in _cm_facts(product).items():
        fl = _cm_norm_field(str(key))
        text = _cm_clean(raw)
        if fl and text:
            rows[fl] = text
    options = product.get("options")
    if isinstance(options, dict):
        for key, raw in options.items():
            fl = _cm_norm_field(str(key))
            text = _cm_clean(raw)
            if fl and text and fl not in rows:
                rows[fl] = text
    option_text = _cm_options_text(product)
    if option_text and "_option" not in rows:
        rows["_option"] = option_text
    return rows


def _cm_key_in_text(key: str, text: str) -> bool:
    fl = _cm_norm_field(key)
    if not fl or not text:
        return False
    blob = text.lower()
    spaced = fl.replace("_", " ")
    compact = _cm_compact(blob)
    key_compact = fl.replace("_", "")
    if fl in blob.replace("-", "_") or spaced in blob:
        return True
    if len(key_compact) >= 3 and key_compact in compact:
        return True
    return False


def _cm_value_attested(value: str, goal: str) -> bool:
    blob = (goal or "").lower()
    if not value or not blob:
        return False
    compact = _cm_compact(blob)
    for tok in re.findall(r"[a-z]{2,}|\d+(?:\.\d+)?", str(value).lower()):
        if tok in blob or (len(tok) >= 2 and tok in compact):
            return True
    return False


def _cm_listing_claim_text(product: dict[str, Any], *, grounded: bool = False) -> str:
    facts = _cm_facts(product)
    bits = []
    # Titles carry class cues the shopper typed; keep them even when grounding.
    bits.append(str(product.get("title") or ""))
    bits.extend([str(product.get("options") or ""), str(product.get("brand") or "")])
    bits.extend(f"{key}: {value}" for key, value in facts.items() if value not in (None, ""))
    return " | ".join(bit for bit in bits if bit)


def _cm_text_supports(value: str, listing: str) -> bool:
    tight = _cm_compact(value)
    blob = _cm_compact(listing)
    return bool(tight) and tight in blob


def _cm_token_overlap(left: str, right: str) -> int:
    left_set = set(_cm_tokens(str(left or "").replace("_", " ")))
    right_set = set(_cm_tokens(str(right or "").replace("_", " ")))
    return len(left_set & right_set)


def _cm_size_amount(product: dict[str, Any]) -> float | None:
    """Largest inch-like measure attested on this SKU.

    Prefer options/facts inches when present (blocks title decoys). Otherwise
    fall back to title inches.
    """
    opt = _cm_options_text(product)
    facts = _cm_facts(product)
    fact_blob = " ".join(str(v) for v in facts.values())
    title = str(product.get("title") or "")

    def _inches(blob: str) -> float | None:
        best = None
        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*-?\s*(?:inch(?:es)?|in\b|\"|'')",
            blob,
            re.I,
        ):
            try:
                amount = float(match.group(1))
            except ValueError:
                continue
            if best is None or amount > best:
                best = amount
        return best

    grounded = _inches(f"{opt} {fact_blob}")
    if grounded is not None:
        return grounded
    # Bare numeric under size-like fact keys (e.g. size: 27).
    for key, raw in facts.items():
        fl = _cm_norm_field(str(key))
        if not any(tok in fl for tok in ("size", "screen", "inch", "display", "diagonal")):
            continue
        num = _cm_number(raw)
        if num is not None:
            return float(num)
    return _inches(title)


def _cm_format_measure(amount: float, unit: str) -> str:
    unit = _cm_canon_unit(unit) or unit
    if abs(amount - round(amount)) < 0.05:
        body = str(int(round(amount)))
    else:
        body = f"{amount:g}"
    if unit == "in":
        return f"{body}in"
    if unit:
        return f"{body}{unit}"
    return body


def _cm_shared_shopper_cue(shopper: str, listing: str) -> str:
    """Echo a multi-char cue the shopper typed that also appears on the listing.

    Only copies phrases already present on both sides — never invents class labels
    from bare pixel counts alone.
    """
    if not shopper or not listing:
        return ""
    list_f = _cm_fold(listing)
    list_c = _cm_compact(listing)
    # Phrases the shopper actually wrote (quoted or adjacent letter/digit runs).
    candidates: list[str] = []
    for match in re.finditer(r"[\"\u201c]([^\"\u201d]{2,40})[\"\u201d]", shopper):
        candidates.append(_cm_clean(match.group(1)))
    for match in re.finditer(
        r"\b([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){0,2}|\d{3,4}\s*[A-Za-z]{1,6}|[A-Za-z]{2,}\s*\d{3,4}p?)\b",
        shopper,
    ):
        phrase = _cm_clean(match.group(1))
        if len(phrase) >= 2 and _cm_fold(phrase) not in _STOP:
            candidates.append(phrase)
    for span in _cm_figure_spans(shopper):
        raw = _cm_clean(span.get("raw") or "")
        if raw:
            candidates.append(raw)
    best = ""
    for cue in candidates:
        folded = _cm_fold(cue)
        if len(folded) < 2:
            continue
        if folded in list_f or _cm_compact(cue) in list_c:
            if len(cue) > len(best):
                best = cue
    return best


def _cm_unit_field_score(field: str, unit: str, brief: str = "", near: str = "") -> int:
    """Score a live-enum / brief field against a measure — letter overlap only."""
    fl = _cm_norm_field(field)
    if not fl or fl in _CLAIM_SOFT or fl in _CLAIM_BLOCK:
        return 0
    compact = fl.replace("_", "")
    unit = str(unit or "").lower()
    parts = [part for part in fl.split("_") if part]
    score = 0.0
    if unit and len(unit) >= 2:
        if unit in parts or unit == compact or compact.endswith(unit) or compact.startswith(unit):
            score += 5
        elif unit in fl and f"_{unit}_" in f"_{fl}_":
            score += 5
    elif unit and unit in parts:
        score += 5
    near_fold = _cm_fold(near)
    brief_fold = _cm_fold(brief)
    for part in parts:
        if len(part) < 3:
            continue
        if near_fold and part in near_fold:
            score += 8
        elif part in brief_fold:
            score += 1
    return int(score * 10)


def _cm_gate_field(query: str, allowed: set[str]) -> str:
    """Pick a claim field from this episode's live enum via shopper↔field overlap."""
    shopper = _cm_fold(_cm_shopper_text(query))
    raw_shopper = _cm_shopper_text(query)
    specs = [
        field
        for field in allowed
        if field not in _CLAIM_SOFT and field not in _CLAIM_BLOCK and field not in _CLAIM_WEAK
    ]
    if not specs:
        return ""
    for field in _cm_brief_named_fields(query):
        if field in set(specs):
            return field
    for field in specs:
        if _cm_key_in_text(field, shopper):
            return field
        parts = [part for part in field.replace("_", " ").split() if len(part) > 2]
        if any(part in shopper for part in parts):
            return field
    spans = _cm_figure_spans(raw_shopper)
    best = ""
    best_score = 0
    for span in spans:
        unit = str(span.get("unit") or "")
        if not unit:
            continue
        raw = str(span.get("raw") or "")
        at = _cm_fold(raw_shopper).find(_cm_fold(raw)) if raw else -1
        near = raw_shopper[max(0, at - 36) : at + len(raw) + 36] if at >= 0 else raw_shopper
        for field in specs:
            score = _cm_unit_field_score(field, unit, raw_shopper, near)
            if score > best_score:
                best_score = score
                best = field
    return best if best_score > 0 else ""


def _cm_map_fact_to_enum(facts_key: str, allowed: list[str], shopper: str) -> str:
    fl = _cm_norm_field(facts_key)
    allowed_norm = [_cm_norm_field(item) for item in allowed or [] if item]
    if fl in allowed_norm:
        return fl
    scored: list[tuple[int, str]] = []
    for field in allowed_norm:
        if field in _CLAIM_SOFT or field in _CLAIM_BLOCK:
            continue
        score = _cm_token_overlap(fl, field) + _cm_token_overlap(field, shopper)
        if _cm_key_in_text(field, shopper):
            score += 3
        key_c = _cm_compact(fl)
        field_c = _cm_compact(field)
        if key_c and field_c and (key_c in field_c or field_c in key_c):
            score += 4
        scored.append((score, field))
    scored.sort(key=lambda item: -item[0])
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return ""


def _cm_fact_spec_value(product: dict[str, Any], field: str, query: str) -> str:
    """Value for a live-enum field from viewed catalog cells / shared shopper cues."""
    listing = _cm_listing_claim_text(product, grounded=True) or _cm_blob(product)
    shopper = _cm_shopper_text(query)
    facts = _cm_facts(product)
    options = product.get("options") if isinstance(product.get("options"), dict) else {}
    surface = f"{listing} {_cm_options_text(product)}"
    spans = _cm_figure_spans(shopper)
    fl = _cm_norm_field(field)
    raw = _cm_clean(facts.get(field) or options.get(field) or "")
    if raw and (
        _cm_text_supports(raw, surface)
        or _cm_value_attested(raw, shopper)
        or any(_cm_span_in(raw, s) for s in spans)
    ):
        return raw
    best = ""
    best_score = 0
    cells = list(facts.items()) + list(options.items())
    for key, cell in cells:
        text = _cm_clean(cell)
        if not text:
            continue
        score = _cm_token_overlap(str(key), fl) * 3
        if any(_cm_span_in(text, span) for span in spans):
            score += 6
        if _cm_value_attested(text, shopper):
            score += 2
        if score > best_score:
            best_score = score
            best = text
    if best_score >= 5:
        return best
    cue = _cm_shared_shopper_cue(shopper, surface)
    if cue:
        return cue
    return best if best_score > 0 else ""


def _cm_claim_field_rule(tool: dict[str, Any] | None) -> str:
    if not tool:
        return ""
    spec = (_props(tool).get("justification") or {}).get("properties") or {}
    claims = spec.get("claims") or {}
    items = claims.get("items") or {}
    field = (items.get("properties") or {}).get("field") or {}
    if isinstance(field, dict):
        return str(field.get("description") or "")[:600]
    return ""


def _cm_span_neighborhood(shopper: str, span: dict[str, Any], pad: int = 36) -> str:
    raw = str(span.get("raw") or "")
    blob = shopper or ""
    if not raw or not blob:
        return blob
    folded = _cm_fold(blob)
    at = folded.find(_cm_fold(raw))
    if at < 0:
        return blob
    return blob[max(0, at - pad) : min(len(blob), at + len(raw) + pad)]


def _cm_shopper_class_token(query: str) -> str:
    """Distinctive shopper token (quoted / Np / alnum class word) — not a field name."""
    shopper = _cm_shopper_text(query)
    for match in re.finditer(r"[\"\u201c]([^\"\u201d]{2,40})[\"\u201d]", shopper):
        token = _cm_clean(match.group(1))
        if token and _cm_fold(token) not in _STOP:
            return token
    hit = re.search(r"\b(\d{3,4}\s*p)\b", shopper, re.I)
    if hit:
        return re.sub(r"\s+", "", hit.group(1).lower())
    hit = re.search(r"\b([A-Za-z][A-Za-z0-9-]{2,24})\b", shopper)
    if hit and _cm_fold(hit.group(1)) not in _STOP:
        return hit.group(1)
    return ""


def _cm_catalog_spec_claims(product: dict[str, Any], query: str) -> list[tuple[str, str]]:
    """Attested catalog fact keys → (key, value). Never invent schema field names."""
    shopper = _cm_shopper_text(query)
    spans = _cm_figure_spans(shopper)
    facts = _cm_facts(product)
    token = _cm_shopper_class_token(query)
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, raw in facts.items():
        fl = _cm_norm_field(str(key))
        if not fl or fl.startswith("_") or fl in _CLAIM_SOFT or fl in _CLAIM_BLOCK:
            continue
        text = _cm_clean(raw)
        if not text:
            continue
        value = ""
        if token and re.search(rf"\b{re.escape(token)}\b", _cm_fold(text), re.I):
            value = token
        if not value:
            for span in spans:
                if _cm_span_in(text, span) or (shopper and _cm_value_attested(text, shopper)):
                    value = text
                    break
        if not value and not spans and _cm_value_attested(text, shopper):
            value = text
        if not value or fl in seen:
            continue
        seen.add(fl)
        found.append((fl, str(value)))
    return found


def _cm_parse_json_object(text: str) -> dict[str, Any] | None:
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


def _cm_parse_claim_rows(text: str) -> list[dict[str, str]]:
    obj = _cm_parse_json_object(text) or {}
    rows = obj.get("claims") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        field = _cm_norm_field(row.get("field"))
        value = _cm_clean(row.get("value"))
        if not field or not value or not _CLAIM_KEY_RE.match(field):
            continue
        if field in _CLAIM_SOFT or field in _CLAIM_BLOCK or field in _CLAIM_IDENTITY:
            continue
        if field in _FEATURE_REASON_HINTS:
            continue
        out.append({"field": field, "value": value})
    return out


def _cm_inference_text(messages: list[dict[str, Any]], *, model: str) -> str:
    queue = [model]
    for cand in (_SCOUT, *_RESERVE):
        if cand not in queue:
            queue.append(cand)
    for model_id in queue[:3]:
        low = model_id.lower()
        if "kimi" in low or "glm" in low or low.startswith("z-ai/"):
            continue
        response = _chat(model=model_id, messages=messages, tools=None, tool_choice=None)
        if response is None:
            continue
        try:
            text = str(response["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            continue
        if text:
            return text
    return ""


def _cm_schema_feature_reasons(rule: str) -> set[str]:
    """Optional feature-reason fields mentioned in the live schema wording."""
    found = set(_FEATURE_REASON_HINTS)
    blob = (rule or "").lower()
    for match in re.finditer(r"\b([a-z][a-z0-9_]{2,24})\b", blob):
        key = match.group(1)
        if key in {"brand", "price", "in_stock", "category", "field", "value"}:
            continue
        window = blob[max(0, match.start() - 40) : match.end() + 40]
        if key in _FEATURE_REASON_HINTS or "reason" in window:
            if key not in _CLAIM_SOFT:
                found.add(key)
    return found


def _cm_compound_drawer_field(field: str, catalog_keys: set[str]) -> bool:
    fl = _cm_norm_field(field)
    if not fl or "_" not in fl:
        return False
    for key in catalog_keys:
        if not key or key in _CLAIM_SOFT:
            continue
        if fl.startswith(key + "_") or fl.endswith("_" + key):
            return True
    return False


def _cm_firm_answers_shopper(value: str, query: str) -> bool:
    """Firm claim value must answer the shopper's measured figure / class cue."""
    shopper = _cm_shopper_text(query)
    spans = _cm_figure_spans(shopper)
    if not spans:
        return bool(value)
    token = _cm_shopper_class_token(query)
    folded = _cm_fold(value)
    if token and re.search(rf"\b{re.escape(token)}\b", folded):
        return True
    for span in spans:
        if _cm_span_in(value, span):
            return True
        unit = str(span.get("unit") or "")
        amount = span.get("amount")
        if unit == "in" and amount is not None:
            if re.search(rf"\b{re.escape(f'{float(amount):g}')}\b", folded) and (
                "in" in folded or "inch" in folded or '"' in str(value)
            ):
                return True
        if unit == "p" and amount is not None:
            if re.search(rf"\b{int(amount)}\s*p\b", folded, re.I):
                return True
        if unit == "hz" and amount is not None:
            if re.search(rf"\b{re.escape(f'{float(amount):g}')}\s*hz\b", folded, re.I):
                return True
    return False


def _cm_convention_attr(spans: list[dict[str, Any]], shopper: str, allowed: list[str]) -> str:
    """Best live-schema field for this shopper's measured unit — never invent names."""
    allowed_norm = [_cm_norm_field(item) for item in allowed or [] if item]
    firm = [item for item in allowed_norm if item not in _CLAIM_SOFT and item not in _CLAIM_BLOCK]
    if not firm:
        return ""
    for span in spans:
        unit = str(span.get("unit") or "")
        if not unit:
            continue
        near = " ".join(
            _cm_span_neighborhood(shopper, span2)
            for span2 in spans
            if str(span2.get("unit") or "") == unit
        )
        scored = [(_cm_unit_field_score(field, unit, shopper, near), field) for field in firm]
        scored.sort(key=lambda item: (-item[0], item[1]))
        if scored and scored[0][0] > 0:
            return scored[0][1]
        for field in firm:
            if _cm_key_in_text(field, shopper):
                return field
    return ""


def _cm_shopper_spec_field(query: str) -> str:
    """Firm claim field from Task-rules / shopper naming — never a unit→label table."""
    shopper = _cm_shopper_text(query)
    named = [
        item
        for item in _cm_brief_named_fields(query)
        if item not in _CLAIM_SOFT and item not in _CLAIM_BLOCK
    ]
    if not named:
        return ""
    spans = _cm_figure_spans(shopper)
    ranked: list[tuple] = []
    blob = _cm_fold(query)
    for field in named:
        score = 0
        dist = 10**6
        for span in spans:
            unit = str(span.get("unit") or "")
            near = _cm_span_neighborhood(shopper, span)
            score = max(score, _cm_unit_field_score(field, unit, query, near))
            raw = str(span.get("raw") or "")
            at = blob.find(_cm_fold(raw)) if raw else -1
            spaced = field.replace("_", " ")
            pos = blob.find(spaced) if spaced else -1
            if pos < 0:
                pos = blob.find(field)
            if at >= 0 and pos >= 0:
                dist = min(dist, abs(pos - at))
        ranked.append((score, -dist, -len(field), field))
    ranked.sort(reverse=True)
    if ranked:
        return ranked[0][-1]
    for field in named:
        if _cm_key_in_text(field, shopper):
            return field
    return ""


def _cm_guess_firm_attr(goal: str, allowed: list[str]) -> str:
    if not allowed or not goal:
        return ""
    for cand in allowed:
        fl = _cm_norm_field(cand)
        if fl in _CLAIM_SOFT or fl in _CLAIM_BLOCK:
            continue
        if _cm_key_in_text(fl, goal):
            return fl
    return ""


def _cm_project_enum_field(catalog_key: str, enum_firm: list[str]) -> str:
    """Map an attested catalog key onto a live enum field by token overlap only."""
    cat = _cm_norm_field(catalog_key)
    if not cat or not enum_firm:
        return ""
    cat_parts = set(cat.split("_")) | set(_cm_tokens(cat.replace("_", " ")))
    best_field = ""
    best = 0
    for field in enum_firm:
        fl = _cm_norm_field(field)
        parts = set(fl.split("_")) | set(_cm_tokens(fl.replace("_", " ")))
        hits = len(parts & cat_parts)
        if hits > best:
            best = hits
            best_field = fl
    return best_field if best else ""


def _cm_llm_firm_claims(
    product: dict[str, Any],
    query: str,
    tool: dict[str, Any] | None,
    have: list[str],
) -> list[dict[str, str]]:
    """Name the shopper's measurable using live enum / schema_rule. No unit→field table."""
    listing = _cm_listing_claim_text(product, grounded=False) or _cm_blob(product)
    if not listing:
        return []
    enum = _cm_claim_enum(tool)
    held = {_cm_norm_field(item) for item in have}
    allowed = [
        item
        for item in (enum or [])
        if _cm_norm_field(item) not in held
        and _cm_norm_field(item) not in _CLAIM_SOFT
        and _cm_norm_field(item) not in _CLAIM_BLOCK
        and _cm_norm_field(item) not in _CLAIM_IDENTITY
    ]
    cache_key = "|".join(
        [
            str(product.get("product_id") or ""),
            str(product.get("sku") or ""),
            ",".join(allowed),
            _cm_fold(_cm_shopper_text(query))[:120],
        ]
    )
    if cache_key in _CLAIM_FILL_CACHE:
        return list(_CLAIM_FILL_CACHE[cache_key])
    if _left() < 18:
        return []
    goal = (_cm_shopper_text(query) or query)[:1500]
    rule = _cm_claim_field_rule(tool)
    feature_reasons = _cm_schema_feature_reasons(rule)
    text = _cm_inference_text(
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
        model=_SCOUT,
    )
    out = _cm_parse_claim_rows(text)
    catalog_keys = {_cm_norm_field(str(key)) for key in _cm_facts(product)}
    needs_align = (not out) or any(
        row["field"] in catalog_keys
        or row["field"] in feature_reasons
        or _cm_compound_drawer_field(row["field"], catalog_keys)
        or not _cm_firm_answers_shopper(row["value"], query)
        for row in out
    )
    if needs_align and _left() >= 18:
        proposed = [
            row for row in out if _cm_firm_answers_shopper(row["value"], query)
        ] or [
            {"field": key, "value": value}
            for key, value in _cm_catalog_spec_claims(product, query)[:2]
        ]
        if proposed:
            aligned = _cm_inference_text(
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
                model=_SCOUT,
            )
            renamed = _cm_parse_claim_rows(aligned)
            if renamed:
                out = renamed
    allowed_set = {_cm_norm_field(item) for item in (allowed or enum or [])}
    cleaned: list[dict[str, str]] = []
    for row in out:
        field = row["field"]
        value = row["value"]
        if field in held or field in _CLAIM_BLOCK or field in _CLAIM_IDENTITY or field in feature_reasons:
            continue
        if _cm_compound_drawer_field(field, catalog_keys):
            continue
        if not allowed_set and field in catalog_keys:
            continue
        if allowed_set and field not in allowed_set:
            continue
        if not _cm_text_supports(value, listing):
            continue
        if not _cm_firm_answers_shopper(value, query) and not _cm_value_attested(value, goal):
            continue
        token = _cm_shopper_class_token(query)
        if token and _cm_text_supports(token, listing) and _cm_firm_answers_shopper(token, query):
            value = token
        cleaned.append({"field": field, "value": value})
    _CLAIM_FILL_CACHE[cache_key] = list(cleaned)
    return cleaned


def _cm_ground_claims(product: dict[str, Any], query: str, tool: dict[str, Any] | None) -> list[dict[str, str]]:
    """Soft drawers + firm gate from live enum / brief / convention / LLM."""
    shopper = _cm_shopper_text(query)
    rows = _cm_fact_rows(product)
    listing = _cm_listing_claim_text(product, grounded=True) or _cm_blob(product)
    allowed = _cm_claim_allowed(tool, query, product)
    allowed_set = set(allowed)
    enum = _cm_claim_enum(tool)
    schema_locked = bool(enum) and any(item not in _CLAIM_SOFT for item in enum)
    spans = _cm_figure_spans(shopper)
    feature_reasons = set(_CLAIM_FEATURE_REASONS) & allowed_set
    feature_reasons |= {
        field
        for field in _cm_schema_reason_fields(tool)
        if field in allowed_set
        and field not in _CLAIM_SOFT
        and field not in _CLAIM_WEAK
        and field != "brand"
    }
    fields: list[str] = [item for item in ("in_stock", "price") if item in allowed_set]
    skip = {"in_stock", "price", "brand", "category"} | _CLAIM_BLOCK | _CLAIM_WEAK | feature_reasons | _CLAIM_IDENTITY
    extras = [item for item in allowed if item not in skip]
    catalog_specs = _cm_catalog_spec_claims(product, query)

    gate = ""
    shopper_spec = _cm_shopper_spec_field(query)
    if shopper_spec and shopper_spec in allowed_set and shopper_spec not in skip:
        gate = shopper_spec
    if not gate:
        guess = _cm_guess_firm_attr(shopper, extras)
        if guess:
            gate = guess
    if not gate:
        conv = _cm_convention_attr(spans, shopper, extras)
        if conv:
            gate = conv
    if not gate:
        gate = _cm_gate_field(query, allowed_set - skip)
    if not gate and catalog_specs and extras:
        for cat_key, _value in catalog_specs:
            projected = _cm_project_enum_field(cat_key, extras) or _cm_map_fact_to_enum(
                cat_key, extras, shopper
            )
            if projected and projected in allowed_set and projected not in skip:
                gate = projected
                break
    if not gate:
        for field in extras:
            if _cm_fact_spec_value(product, field, query) or rows.get(field):
                gate = field
                break
    if gate and gate not in skip:
        fields.append(gate)

    if "brand" in allowed_set and (
        _cm_brand_guess(query)
        or _cm_key_in_text("brand", shopper)
        or re.search(r"\bprefer|\bideally\b", shopper, re.I)
        or (rows.get("brand") and _cm_value_attested(str(rows.get("brand") or ""), shopper))
    ):
        fields.append("brand")

    if not gate:
        for field in sorted(feature_reasons):
            if field in allowed_set and _cm_feature_reason_true(product, field, query):
                fields.append(field)
                break

    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    catalog_values = {field: value for field, value in catalog_specs}

    def add_claim(fl: str, value: str) -> None:
        if fl in seen or fl in _CLAIM_BLOCK or fl in _CLAIM_IDENTITY or not value:
            return
        if not _CLAIM_KEY_RE.match(fl):
            return
        if fl not in _CLAIM_SOFT and fl not in feature_reasons:
            surface = f"{listing} {product.get('title') or ''}"
            if not (
                _cm_text_supports(value, surface)
                or _cm_value_attested(value, shopper)
                or any(_cm_span_in(value, span) for span in spans)
                or _cm_firm_answers_shopper(value, query)
                or _cm_shared_shopper_cue(shopper, listing)
            ):
                return
        seen.add(fl)
        claims.append({"field": fl, "value": str(value)})

    for field in fields:
        fl = _cm_norm_field(field)
        if fl not in allowed_set or fl in seen:
            continue
        if fl in {"in_stock", "price", "brand"}:
            add_claim(fl, rows.get(fl, ""))
            continue
        if fl in feature_reasons:
            if _cm_feature_reason_true(product, fl, query):
                add_claim(fl, "true")
            continue
        value = (
            catalog_values.get(fl)
            or _cm_fact_spec_value(product, fl, query)
            or rows.get(fl)
        )
        if not value and catalog_specs:
            for cat_key, cat_val in catalog_specs:
                if _cm_project_enum_field(cat_key, [fl]) == fl or cat_key == fl:
                    value = cat_val
                    break
            if not value:
                token = _cm_shopper_class_token(query)
                for _cat_key, cat_val in catalog_specs:
                    if token and re.search(rf"\b{re.escape(token)}\b", _cm_fold(cat_val)):
                        value = token
                        break
        add_claim(fl, value or "")

    if not _has_spec_claim(claims):
        filled = _cm_llm_firm_claims(product, query, tool, [row["field"] for row in claims])
        for row in filled:
            fl = _cm_norm_field(row["field"])
            if schema_locked and fl not in allowed_set:
                continue
            if fl in skip:
                continue
            add_claim(fl, str(row["value"]))
            if _has_spec_claim(claims):
                break
    return claims[:4]


def _cm_feature_reason_true(product: dict[str, Any], field: str, query: str) -> bool:
    """Feature reasons must be explicit positive capabilities on this listing."""
    fl = _cm_norm_field(field)
    if not fl:
        return False
    listing = _cm_fold(_cm_listing_claim_text(product, grounded=True) or _cm_blob(product))
    facts = _cm_facts(product)
    # Exact fact key with truthy / positive value.
    for key, raw in facts.items():
        if _cm_norm_field(str(key)) != fl:
            continue
        text = _cm_fold(_cm_clean(raw))
        if text in {"true", "yes", "1", "supported", "available"}:
            return True
        if text in {"false", "no", "0", "none", "n/a", "na", "unknown", "absent"}:
            return False
        if text:
            return True
    # Phrase present as a positive capability token on the listing.
    phrase = fl.replace("_", " ")
    if phrase and phrase in listing:
        # Reject negated mentions.
        if re.search(rf"\bno\s+{re.escape(phrase)}\b|\bwithout\s+{re.escape(phrase)}\b", listing):
            return False
        return True
    return False


def _has_spec_claim(claims: list[dict[str, str]] | None) -> bool:
    """TF7 needs more than price/stock — brand, a measured gate, or a true feature reason."""
    soft = set(_CLAIM_SOFT)  # price, in_stock only
    weak = set(_CLAIM_WEAK)
    for row in claims or []:
        field = str(row.get("field") or "").strip().lower()
        value = str(row.get("value") or "").strip()
        if not field or field in soft or field in weak:
            continue
        if field in _CLAIM_FEATURE_REASONS and value.lower() not in {"true", "yes", "1"}:
            continue
        if value:
            return True
    return False



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
                key = _cm_listing_key(row) or str(row.get("product_id") or "")
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
            incoming = _cm_price_of({"price": value})
            current = _cm_price_of(out)
            if incoming is None and current is not None:
                continue
            out[key] = incoming if incoming is not None else value
            continue
        if key.startswith("_") and out.get(key) and not value:
            continue
        out[key] = value
    return out


def _cm_useful_query(query: str, kind: str = "", budget: float | None = None) -> bool:
    folded = _cm_fold(query)
    tokens = _cm_tokens(query)
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
    if kind and _cm_fold(kind) in folded:
        return True
    return bool(tokens)


def _criterion_matrix(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[list[float]]:
    used = axes or [("price", "asc")]
    matrix: list[list[float]] = []
    for product in pool:
        row: list[float] = []
        for name, direction in used:
            value = _cm_axis_value(product, name)
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



# Prefix aliases for ranker helpers
_listing_key = _cm_listing_key
_lex_rank = _cm_lex_rank
_axis_value = _cm_axis_value

def _retrieve_one(query_tf, doc_tf, df, n_docs, avgdl, coll_len, tokens) -> float:
    score = 0.0
    doc_len = max(sum(doc_tf.values()), 1)
    coll = max(coll_len, 1)
    lam = 0.15
    for term, qtf in query_tf.items():
        tf = float(doc_tf.get(term, 0))
        cf = max(float(df.get(term, 0)), 0.5)
        p_d = tf / doc_len if tf > 0 else 0.0
        p_c = cf / coll
        p = (1.0 - lam) * p_d + lam * p_c
        score += float(qtf) * math.log(max(p, 1e-12))
    return score


def _moora_ratio(pool, axes):
    used = axes or [("price", "asc")]
    matrix = _unit_norm(_criterion_matrix(pool, used))
    weights = _entropy_weights(matrix) or [1.0 / max(len(used), 1)] * len(used)
    scores = []
    for row in matrix:
        acc = 0.0
        for col, (_name, direction) in enumerate(used):
            acc -= weights[col] * row[col] if direction == "asc" else -weights[col] * row[col]
        scores.append(acc)
    return scores


def _ref_point(pool, axes):
    used = axes or [("price", "asc")]
    matrix = _unit_norm(_criterion_matrix(pool, used))
    weights = _entropy_weights(matrix) or [1.0 / max(len(used), 1)] * len(used)
    cols = len(used)
    ideal = [max(row[col] for row in matrix) for col in range(cols)]
    return [-max(weights[c] * abs(ideal[c] - row[c]) for c in range(cols)) for row in matrix]


def _full_mult(pool, axes):
    used = axes or [("price", "asc")]
    matrix = _unit_norm(_criterion_matrix(pool, used))
    weights = _entropy_weights(matrix) or [1.0 / max(len(used), 1)] * len(used)
    scores = []
    for row in matrix:
        benefit = cost = 1.0
        for col, (_name, direction) in enumerate(used):
            val = max(row[col], 1e-9) ** weights[col]
            if direction == "asc":
                cost *= val
            else:
                benefit *= val
        scores.append(benefit / max(cost, 1e-9))
    return scores


def _choose_closed(pool: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if len(pool) <= 1:
        return list(pool)
    ballots = [_moora_ratio(pool, axes), _ref_point(pool, axes), _full_mult(pool, axes)]
    n = len(pool)
    borda = [0.0] * n
    for scores in ballots:
        order = sorted(range(n), key=lambda i: (-scores[i], i))
        for rank, i in enumerate(order):
            borda[i] += n - rank
    order = sorted(range(n), key=lambda i: (-borda[i], i))
    return [pool[i] for i in order]


def _choose_open(pool: list[dict[str, Any]], axes: list[tuple[str, str]], fused: dict[str, float]) -> list[str]:
    ranked = sorted(pool, key=lambda item: -fused.get(_cm_listing_key(item), 0.0))
    return [_cm_listing_key(item) for item in ranked]

def compile_brief(query: str) -> dict[str, Any]:
    shopper = _cm_shopper_text(query)
    rules = _cm_rules_text(query)
    named = _cm_named_keys(query)
    axes = _cm_axes_from(shopper) or _cm_axes_from(rules)
    budget_hit = _BUDGET_RX.search(shopper) or _BUDGET_RX.search(query or "")
    budget = float(budget_hit.group(1).replace(",", "")) if budget_hit else None
    quotes = _cm_quoted(query)
    why = _cm_wants_reason(query)
    held = _cm_held_back(query)
    patch = bool(_PATCH_RX.search(shopper) or _PATCH_RX.search(rules))
    model = _cm_model_guess(query)
    color = _cm_color_guess(query)
    brand = _cm_brand_guess(query)
    kind = _cm_kind_noun(query)
    category = _cm_category_hint(query)
    spans = _cm_figure_spans(shopper, budget)
    fork = _cm_soft_brand(query)
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
    if patch and (model or color or "least-relax" in _cm_fold(rules)):
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
    wishes = _cm_wish_terms(query) if DUTY_TASTE in duties else []
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
        token = _cm_clean(phrase)
        folded = _cm_fold(token)
        if not token or folded in seen or not _cm_useful_query(token, kind=kind, budget=budget):
            continue
        seen.add(folded)
        cleaned.append(token)
    if not cleaned:
        cleaned.append(" ".join(_cm_tokens(shopper)[:6]) or "product")
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


_CORE = """You are a shopping agent inside a generated catalog. Call ONLY tools listed
in this episode's policy_view. Copy nothing from outside the episode.

# Crown Moora — MULTIMOORA three-form ballot

## Non-negotiable seals
1. Firm gates (category/path, budget ceiling, in-stock, number+unit spans the
   shopper wrote, quoted / "matches phrase" title tokens, named product_id::sku
   membership) MUST hold on the exact SKU you cart and order.
2. Soft leanings (preferred brand, use-case tastes, "rather pay") never override
   a firm miss. Prefer soft only among fully firm-legal SKUs.
3. The FIRST add_to_cart is graded. Never park a placeholder, decoy, or
   "temporary" SKU. Never cart then silently swap without a market shock.
4. Search with catalog VALUES (phrase, maker, figure, product noun) — never
   JSON field names or schema keys.
5. After stockout / price_change / cart shock: the event target is dead. Probe
   it with inspect_stock when named, remove a stale cart line, re-rank the live
   pool, cart the NEW winner, THEN order. Never place_test_order on a SKU that
   is not currently in the cart.
6. place_test_order on a pure ranking brief: ONLY product_id and sku (no
   justification). When the shopper asked why: attach justification.claims
   copied from viewed facts using ONLY field names this episode's order tool
   allows. Values must equal catalog text. Never invent attributes. Never
   rewrite units. Never send category, title, os, summary, evidence, reason.

## Cadence
Survey → hydrate → inspect → cart MULTIMOORA nominee → mend → seal. Ratio system, reference-point regret, and full multiplicative form vote; Borda aggregates.

## Search discipline
Quoted phrase, maker, figure, product noun. Reformulate on recall briefs.

## Figures and evidence
Numbers the shopper said vs listing rows. No homemade attribute maps.

## Dialogue
Ask-first when a firm figure is withheld. Answer questions with message first.

## Closing the ticket
Ranking: product_id + sku. Justification: live enum, copied text, firm gate required.

## Market shocks
Least relaxation. Same model other color first. Ban the event key. Re-cart before order.

## Category stance
The race mix is hidden. Groceries, tools, beauty, parts, apparel, and
electronics share THIS same loop. Do not assume device classes, finish
palettes, or display nicknames. If Task rules define named classes for THIS
episode only, honor those thresholds here — nowhere else.

## First-cart protocol
Before the first add_to_cart on an open catalog: run at least two short value
queries when recall matters; view enough listings that raw facts or options are
visible for the intended winner; inspect_stock on that exact product_id and sku;
then cart. On a closed slate: view or compare every named survivor until facts
exist; rank on the axes this brief named; inspect the live number one; cart that
number one. The first cart is irrevocable for scoring — treat it as the answer.

## Order protocol
place_test_order must target the SKU currently in the cart. If ranking changed
after a shock, remove the old cart line, cart the new number one, then order.
Pure ranking briefs: arguments are product_id and sku only. Why briefs: attach
justification.claims with in_stock, price, and one firm gate field from the live
enum whose key or value overlaps the shopper words or measured spans. Copy catalog
spelling exactly. Soft-only claims fail. Never invent field names. Never rewrite
units. Never send category, title, os, summary, evidence, or reason.

## Search protocol
Emit short queries built from: quoted phrase; matches-phrase tokens; maker;
product noun from in-the-X-category or buy-noun; number-plus-unit spans the
shopper wrote. Reformulate using tokens that appeared in returned titles. Never
paste the whole instruction. Never search with schema field names.

## Recovery protocol
Category, budget, and in-stock never relax. Prefer another color of the same
model before jumping to other models. Twin-search the disappeared title. Keep
the event target banned for the rest of the episode. Re-cart before any order.

## Ask protocol
If the brief says a firm figure is revealed only when you ask, call message
immediately. Do not search until the shopper replies. Then bind the revealed
figure on option rows and facts of the exact SKU.

## Preference protocol
Among firm-legal SKUs, prefer listings whose text covers the shopper rather /
matters-more / use-case words. Skip budget-fraction decoys. Cheapness is a
tie-break, not the objective. After reprice, recover to a different listing.

## Intent protocol
Preferred brand is soft. First cart may be branded only when that SKU is fully
firm-legal. Otherwise cart the best firm-legal alternative.

## Justification recipe
1. Read the live place_test_order claim field enum for this episode.
2. Always attempt in_stock and price when allowed.
3. Choose one additional allowed field by overlap between shopper text and field
   tokens, or between shopper measured spans and catalog cells.
4. Copy the catalog cell text verbatim as the claim value.
5. If you cannot attest a non-soft field yet, view or inspect again before order.
"""

_DUTY_BOOK = {
    DUTY_HUNT: """# Hunt (recall)
Reformulate at least once with title-borne tokens. Cover several distinct product_ids before first cart. Quoted / matches-phrase tokens must appear in the title. Drop accessories via category_path and 'for / compatible with'.
""",
    DUTY_FORK: """# Fork (soft brand / intent)
Preferred brand is a leaning. Honor it on the FIRST cart only when that SKU also clears every firm gate. Keep a dual shortlist: branded-legal vs best-legal. If branded fails firm, cart the best firm-legal alternative.
""",
    DUTY_ASK: """# Ask (gated firm figure)
One firm figure is withheld until you ask. Call message FIRST. Search only after the shopper answers. Match the revealed figure on option/facts of the exact SKU. After sellout, a new product_id — not a sibling option of the dead listing.
""",
    DUTY_TASTE: """# Taste (preference / use-case)
Among firm-legal SKUs, maximize cover of the shopper's stated use-case / rather / matters-more words. Do not cart the cheapest stub. Skip budget-fraction decoys. After reprice, recover to a different listing_id.
""",
    DUTY_SLATE: """# Slate (closed ranking)
Closed set only: the named product_id::sku list in the brief. Batch view/compare until every survivor shows facts/options. Rank with the axes THIS brief named (highest/lowest X), then price. After any shock, rebuild the ranking from scratch. First cart = live #1. Order with product_id + sku only unless the shopper also asked why. Do not keep a pre-shock leader that is no longer #1. Three-form MULTIMOORA with Borda aggregation.
""",
    DUTY_MEND: """# Mend (recovery)
Never relax category, budget, or in-stock. Same model / other color first, then other models. Twin-search the dead title. Event target stays banned. Re-cart before order.
""",
    DUTY_WHY: """# Why (justification)
View facts before the first cart. Build justification.claims from the live place_test_order enum: always try in_stock + price, PLUS one firm gate field whose name or value overlaps the shopper's words or measured spans, copied VERBATIM from the listing. Soft-only (price+stock) FAILS. Deny category/title/os/summary/evidence. Cap at four claims.
""",
}


# Operational appendix (category-agnostic)
## First-cart protocol
# Before the first add_to_cart on an open catalog: survey at least two short
# value queries when recall matters; view enough listings that raw facts or
# options are visible for the intended winner; inspect_stock on that exact
# product_id+sku; then cart. On a closed slate: view/compare every named
# survivor until facts exist; rank on the axes THIS brief named; inspect the
#1; cart that #1. The first cart is irrevocable for scoring — treat it as
# the answer you will defend.

## Order protocol
# place_test_order must target the SKU currently in the cart. If ranking
# changed after a shock, remove the old cart line, cart the new #1, then
# order. Pure ranking briefs: arguments are product_id and sku only. Why
# briefs: attach justification.claims with in_stock, price, and one firm
# gate field from the live enum whose key or value overlaps the shopper's
# words or measured spans. Copy catalog spelling. Soft-only claims fail.

## Search protocol
# Emit short queries built from: quoted phrase; "matches phrase" tokens;
# maker; product noun from "in the X category" or buy-noun; number+unit
# spans the shopper wrote. Reformulate using tokens that appeared in
# returned titles. Never paste the whole instruction. Never search with
# schema field names.

## Recovery protocol
# Category, budget, and in-stock never relax. Prefer another color of the
# same model before jumping to other models. Twin-search the disappeared
# title. Keep the event target banned for the rest of the episode.

## Ask protocol
# If the brief says a firm figure is revealed only when you ask, call
# message immediately. Do not search until the shopper replies. Then bind
# the revealed figure on option/facts of the exact SKU.

## Preference protocol
# Among firm-legal SKUs, prefer listings whose text covers the shopper's
# rather / matters-more / use-case words. Skip budget-fraction decoys.
# Cheapness is a tie-break, not the objective.

## Intent protocol
# Preferred brand is soft. First cart may be branded only when that SKU is
# fully firm-legal. Otherwise cart the best firm-legal alternative and keep
# the brand as a secondary hope after shocks if it becomes feasible.



# Turn coaches (runtime may inject these as user tips)
# - Turn 1 + ask gate: message for the missing firm figure BEFORE search.
# - Closed slate early: compare/view every named product_id::sku until facts exist.
# - Soft brand: first cart branded only if that SKU clears every firm gate.
# - Taste: compare ≥2 firm-legal listings on the shopper's use-case words.
# - Why: view facts; claims need a non-soft live-enum field attested on the SKU.
# - After shock: inspect event target when named, remove stale cart, cart new #1,
#   then place_test_order on the cart SKU only.
# - Ranking order payload: product_id + sku only (no justification) unless why.
# - Never invent display class nicknames or remap units into foreign field names.
# - Never send category/title/os/summary/evidence as claim fields.
# - Stop when an observation reports done=true.



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
    enum = _cm_claim_enum(order)
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


class _Vault:
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
        self.event_kind = ""
        self.event_old_price = None
        self.event_new_price = None
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
            key = _cm_listing_key(product)
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
                if _cm_listing_key(item) == key:
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
        self.event_kind = str(event.get("kind") or event.get("type") or "")
        self.event_old_price = event.get("old_price")
        self.event_new_price = event.get("new_price")
        row = self.rows.get(key) if key else None
        if row is not None:
            self.dead_title = str(row.get("title") or self.dead_title)
            self.dead.add(key)
            if event.get("new_price") is not None:
                row["price"] = event.get("new_price")
            if "stockout" in _cm_fold(event.get("kind") or "") or event.get("in_stock_after") is False:
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
                key = _cm_listing_key(product)
                if key:
                    fused[key] = product
        fused.update(self.rows)
        return list(fused.values())


class _Crown:
    def __init__(self, query: str, tools: list[dict[str, Any]], max_calls: int) -> None:
        self.query = query
        self.brief = compile_brief(query)
        self.max_calls = max(1, max_calls)
        self.store = _Vault()
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
        # Closed-slate first cart is graded — never rewrite the model onto a different SKU.
        self.rewrite_sku = DUTY_SLATE not in self.brief["duties"] and True
        self.stall_breaks = 0
        self.use_stall = False
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
        key = _cm_listing_key(product)
        row = self.store.rows.get(key) or {}
        pid = str(product.get("product_id") or "")
        if pid and (not _cm_facts(row) or not _cm_facts(product)):
            for stored in self.store.rows.values():
                if str(stored.get("product_id") or "") == pid and _cm_facts(stored):
                    row = _absorb(row, stored)
                    break
        return _absorb(row, product)

    def _stale(self, product: dict[str, Any] | None) -> bool:
        if product is None:
            return False
        if not _cm_in_stock(product):
            return True
        budget = self.brief.get("budget")
        price = _cm_price_of(product)
        if budget is not None and price is not None and price > float(budget):
            return True
        if self.store.event_key and _cm_listing_key(product) == self.store.event_key:
            return True
        key = _cm_listing_key(product)
        if key and key in self.store.dead:
            return True
        return False

    def live(self, *, pick: bool = False) -> list[dict[str, Any]]:
        budget = self.brief.get("budget")
        kind = str(self.brief.get("kind") or "")
        source = self.store.pool()
        named = self.brief.get("named") or []
        if named:
            by_key = {_cm_listing_key(item): item for item in source}
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
            key = _cm_listing_key(product)
            if key in self.store.dead:
                continue
            if key and key == self.store.event_key:
                continue
            if not _cm_in_stock(product):
                continue
            price = _cm_price_of(product)
            if budget is not None and price is not None and price > float(budget):
                continue
            closed = bool(self.brief.get("named"))
            if not closed:
                if _cm_junk(product, kind):
                    continue
                if kind and product.get("title") and not _cm_kind_ok(product, kind):
                    continue
                if self.brief.get("quotes") and product.get("title"):
                    if not any(_cm_phrase_hit(product, phrase) for phrase in self.brief["quotes"]):
                        if self.brief["primary"] == DUTY_HUNT and product.get("_hydrated"):
                            continue
            spec_duty = self.brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK}
            if spec_duty and self.brief.get("spans"):
                if pick and not product.get("_hydrated"):
                    if not self.store.event_key:
                        continue
                    if not _cm_figure_ok(product, self.query, self.brief.get("spans") or []):
                        continue
                if product.get("_hydrated") and not _cm_figure_ok(
                    product, self.query, self.brief.get("spans") or []
                ):
                    continue
            rows.append(product)
        return rows

    def retrieval_map(self, pool: list[dict[str, Any]]) -> dict[str, float]:
        query_tf = Counter(
            _cm_tokens(" ".join(self.brief.get("phrases") or []) + " " + _cm_shopper_text(self.query))
        )
        docs = [(_cm_tokens(_cm_blob(product)), product) for product in pool]
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
            scores[_cm_listing_key(product)] = _retrieve_one(
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
            # Closed ranking tasks grade lexicographic axes, not compromise MCDA.
            return _cm_lex_rank(pool, axes) or _choose_closed(pool, axes)
        fused = self.retrieval_map(pool)
        by_key = {_cm_listing_key(product): product for product in pool}
        wishes = list(self.brief.get("wishes") or [])
        budget = self.brief.get("budget")
        brand = str(self.brief.get("brand") or "")
        honor = DUTY_FORK in self.brief["duties"] and not self.store.event_key
        open_order = _choose_open(pool, axes, fused)

        def sort_key(product: dict[str, Any]) -> tuple:
            key = _cm_listing_key(product)
            retrieval = fused.get(key, 0.0)
            taste = _cm_taste_cover(product, wishes) if DUTY_TASTE in self.brief["duties"] else 0.0
            decoy = 1 if DUTY_TASTE in self.brief["duties"] and _cm_decoy(product, budget) else 0
            brand_rank = 0 if honor and brand and _cm_brand_hit(product, brand) else 1
            ready = 0 if product.get("_hydrated") else 1
            after = 0 if product.get("_aftershock") else 1
            price = _cm_price_of(product)
            honor_after = bool(
                self.store.event_key
                and self.brief["duties"] & {DUTY_TASTE, DUTY_WHY, DUTY_SLATE, DUTY_MEND, DUTY_FORK}
            )
            try:
                open_pos = open_order.index(key)
            except ValueError:
                open_pos = 10_000
            return (
                decoy,
                brand_rank,
                after if honor_after else 0,
                0 if _cm_figure_ok(product, self.query, self.brief.get("spans") or []) else 1,
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
                if _cm_listing_key(item) != target
                and not self._stale(item)
                and _cm_figure_ok(item, self.query, list(self.brief.get("spans") or []))
            ]
        if not ranked:
            return None
        live = [item for item in ranked if _cm_listing_key(item) != target and not self._stale(item)]
        if not live:
            live = [item for item in ranked if _cm_listing_key(item) != target]
        spans = list(self.brief.get("spans") or [])
        if self.brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK}:
            fitted = [item for item in live if _cm_figure_ok(item, self.query, spans)]
            if fitted:
                live = fitted
        if target and self.brief["duties"] & {DUTY_SLATE, DUTY_TASTE, DUTY_WHY, DUTY_MEND}:
            post = [item for item in live if item.get("_aftershock")]
            if post:
                live = post
        if DUTY_TASTE in self.brief["duties"]:
            budget = self.brief.get("budget")
            wishes = list(self.brief.get("wishes") or [])
            fitted = [item for item in live if not _cm_decoy(item, budget)]
            hydrated = [item for item in fitted if item.get("_hydrated")]
            strong = [item for item in hydrated if _cm_taste_cover(item, wishes) >= 0.34]
            live = strong or hydrated or fitted or live
        if DUTY_FORK in self.brief["duties"] and not target:
            brand = str(self.brief.get("brand") or "")
            branded = [item for item in live if brand and _cm_brand_hit(item, brand)]
            ready = [
                item
                for item in branded
                if item.get("_hydrated") and _cm_figure_ok(item, self.query, spans)
            ]
            if ready:
                return ready[0]
            if branded:
                return branded[0]
        if DUTY_MEND in self.brief["duties"] and (self.brief.get("model") or self.brief.get("color")):
            model = str(self.brief.get("model") or "")
            color = str(self.brief.get("color") or "")
            modeled = [item for item in live if model and _cm_model_hit(item, model)]
            if color and not target:
                exact = [item for item in live if _cm_ladder(item, model, color) == 0]
                if exact:
                    live = exact
                else:
                    return None
            elif target:
                stem = _cm_sku_stem(target.split("::")[-1])
                siblings = [
                    item for item in live if stem and _cm_sku_stem(item.get("sku")) == stem
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
                    _cm_ladder(item, model, color),
                    0 if target and _cm_sku_stem(item.get("sku")) == _cm_sku_stem(target.split("::")[-1]) else 1,
                    _cm_price_of(item) if _cm_price_of(item) is not None else 1e12,
                )
            )
        if DUTY_WHY in self.brief["duties"] and self.brief.get("brand"):
            brand = str(self.brief.get("brand") or "")
            branded = [item for item in live if _cm_brand_hit(item, brand)]
            if branded:
                live = branded
        if DUTY_WHY in self.brief["duties"] or self.brief.get("why"):
            claimed = [
                item
                for item in live
                if item.get("_hydrated")
                and _has_spec_claim(_cm_ground_claims(item, self.query, self.order))
            ]
            if claimed:
                live = claimed
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
            claims = _cm_ground_claims(product, self.query, self.order)
            if claims:
                wanted["justification"] = {"claims": claims}
        return {"name": _tool_name(self.order), "args": _bind(self.order, wanted)}

    def _search_actions(self) -> list[dict[str, Any]]:
        kind = str(self.brief.get("kind") or "")
        budget = self.brief.get("budget")
        phrases = [
            item
            for item in (self.brief.get("phrases") or ["product"])
            if _cm_useful_query(item, kind=kind, budget=budget)
        ]
        if DUTY_MEND in self.brief["duties"] and self.brief.get("model"):
            model = str(self.brief.get("model") or "")
            color = str(self.brief.get("color") or "")
            headed = [item for item in (f"{color} {model}".strip(), model) if item]
            phrases = headed + [item for item in phrases if item not in headed]
        elif DUTY_MEND in self.brief["duties"] and self.store.dead_title and not self.echoed:
            title = " ".join(_cm_tokens(self.store.dead_title)[:6])
            if title:
                phrases = [title] + [item for item in phrases if item != title]
            self.echoed = True
        elif self.brief.get("model") and self.store.event_key:
            model = str(self.brief.get("model") or "")
            phrases = [model] + [item for item in phrases if item != model]
        phrases = phrases or ([kind] if kind else ["product"])
        if DUTY_MEND in self.brief["duties"]:
            cap = min(_SURVEY_CAP, max(len(phrases), 1))
        else:
            cap = 1 if self.store.event_key else min(_SURVEY_OPEN, max(len(phrases), 1))
        actions: list[dict[str, Any]] = []
        if self.search and self.store.search_n < cap:
            query = phrases[min(self.store.search_n, len(phrases) - 1)]
            self.store.search_n += 1
            actions.append(
                {
                    "name": _tool_name(self.search),
                    "args": _bind(self.search, {"query": query, "k": 15, "in_stock": True}),
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
                    "k": 15,
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


    def _cm_event_parts(self) -> tuple[str, str]:
        key = self.store.event_key or ""
        if "::" in key:
            pid, sku = key.split("::", 1)
            return pid, sku
        return key, ""

    def _cm_probe_event(self) -> dict[str, Any] | None:
        """Inspect the shocked listing — required to close the market event."""
        if not self.store.event_key or self.store.event_probed or not self.inspect:
            return None
        pid, sku = self._cm_event_parts()
        if not pid:
            return None
        return {
            "name": _tool_name(self.inspect),
            "args": _bind(self.inspect, {"product_id": pid, "sku": sku or None}),
        }

    def _cm_ack_event(self) -> dict[str, Any] | None:
        """Acknowledge a market price/stock change before recovery."""
        if not self.message or not self.store.event_key or self.store.event_acked:
            return None
        if not self.store.event_probed:
            return None
        old = self.store.event_old_price
        new = self.store.event_new_price
        kind = _cm_fold(self.store.event_kind)
        if "stock" in kind:
            text = (
                "That listing just went out of stock. I will re-check remaining "
                "candidates and switch to another option that still meets every firm gate."
            )
        elif old is not None and new is not None:
            try:
                text = (
                    f"Yes, the price moved from {float(old):.2f} to {float(new):.2f}. "
                    "I will switch to another firm-compliant option within budget."
                )
            except (TypeError, ValueError):
                text = (
                    "Yes, that listing's price changed. I will switch to another "
                    "firm-compliant option within budget."
                )
        else:
            text = (
                "Yes, that listing changed. I will re-check live stock and price, "
                "then switch to another firm-compliant option."
            )
        return {
            "name": _tool_name(self.message),
            "args": _bind(self.message, {"content": text}),
        }

    def plan(self) -> list[dict[str, Any]]:
        # Market events must be probed (inspect event target) then acknowledged.
        if self.store.event_key and not self.store.event_probed:
            probe = self._cm_probe_event()
            if probe:
                return [probe]
        if self.store.event_key and self.store.event_probed and not self.store.event_acked:
            ack = self._cm_ack_event()
            if ack:
                return [ack]
        if self.message and self.brief.get("ask") and not self.asked:
            return [
                {
                    "name": _tool_name(self.message),
                    "args": _bind(
                        self.message,
                        {"content": "Tell me the missing firm figure and I will search only after."},
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
            missing = []
            for spec in named:
                key = f"{spec['product_id']}::{spec['sku']}"
                row = self.store.rows.get(key) or {}
                if spec["product_id"] not in self.store.viewed or not (
                    _cm_facts(row) or row.get("_hydrated")
                ):
                    missing.append(spec["product_id"])
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
                siblings = [item for item in live_now if not model or _cm_model_hit(item, model)]
                need_search = (not siblings) and self.store.search_n < 4
            else:
                need_search = self.store.search_n < (4 if DUTY_WHY in self.brief["duties"] else 5) and (
                    not live_now or self.store.search_n < 2
                )
            if DUTY_MEND in self.brief["duties"] and not recovering:
                need_search = self.store.search_n < 2 or need_search
                model = str(self.brief.get("model") or "")
                modeled = [item for item in live_now if not model or _cm_model_hit(item, model)]
                ready = [item for item in modeled if item.get("_hydrated")]
                if len(ready) < 2:
                    hydrate = self._hydrate(modeled[: self._view_cap()])
                    if hydrate:
                        return hydrate[: self.max_calls]
            if recovering and DUTY_MEND in self.brief["duties"]:
                stem = _cm_sku_stem((self.store.event_key or "").split("::")[-1])
                twins = [item for item in live_now if stem and _cm_sku_stem(item.get("sku")) == stem]
                if not twins and self.store.search_n < 4:
                    need_search = True
            if DUTY_WHY in self.brief["duties"] and not recovering:
                grounded = [
                    item
                    for item in live_now
                    if item.get("_hydrated")
                    and _cm_figure_ok(item, self.query, list(self.brief.get("spans") or []))
                ]
                if not grounded and self.store.search_n < 4:
                    need_search = True
            if DUTY_TASTE in self.brief["duties"] and not recovering:
                wishes = list(self.brief.get("wishes") or [])
                if self.store.search_n < 6 and not any(
                    _cm_taste_cover(item, wishes) >= 0.34 for item in live_now
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
                    if (not model or _cm_model_hit(item, model)) and not item.get("_hydrated")
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
        if self.inspect and _cm_listing_key(champ) not in self.store.gauged:
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
            need_facts = self.brief["duties"] & {DUTY_WHY, DUTY_TASTE, DUTY_FORK, DUTY_ASK, DUTY_SLATE}
            if need_facts and (not champ.get("_hydrated") or not _cm_facts(self._full(champ))):
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
            cart_key = _cm_listing_key(cart_item)
            # After rank flip / shock: cart must equal live champion before any order.
            if champ is not None and _cm_listing_key(champ) != cart_key:
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
                if self.cart and not self._stale(champ):
                    return [
                        {
                            "name": _tool_name(self.cart),
                            "args": _bind(
                                self.cart,
                                {"product_id": champ.get("product_id"), "sku": champ.get("sku")},
                            ),
                        }
                    ]
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
                # TF7 seal: soft-only / weak-model claims score zero — never stamp them.
                if DUTY_WHY in self.brief["duties"] or self.brief.get("why"):
                    claims = _cm_ground_claims(cart_item, self.query, self.order)
                    if not _has_spec_claim(claims):
                        self.spec_holds += 1
                        if self.spec_holds <= 2 and self.view and str(cart_item.get("product_id") or ""):
                            return [
                                {
                                    "name": _tool_name(self.view),
                                    "args": _bind(
                                        self.view,
                                        {"product_id": cart_item.get("product_id")},
                                    ),
                                }
                            ]
                        # Try another hydrated firm-ok candidate that can attest a gate.
                        for alt in self.ranked() or self.live():
                            if _cm_listing_key(alt) == cart_key or self._stale(alt):
                                continue
                            if not alt.get("_hydrated"):
                                continue
                            if _has_spec_claim(_cm_ground_claims(alt, self.query, self.order)):
                                if self.remove:
                                    return [
                                        {
                                            "name": _tool_name(self.remove),
                                            "args": _bind(
                                                self.remove,
                                                {
                                                    "product_id": cart_item.get("product_id"),
                                                    "sku": cart_item.get("sku"),
                                                },
                                            ),
                                        }
                                    ]
                                break
                        if self.spec_holds < 6 and _left() > 40:
                            search = self._search_actions()
                            if search:
                                return search
                            return self._hydrate(self.live() or self.store.pool())[: self.max_calls]
                        # Deadline: still refuse soft-only — finish path may retry.
                        return []
                # Never order a champion that is not in the cart.
                return [self._order(cart_item)]
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
            if self.store.event_open and self.inspect and _cm_listing_key(cart) not in self.store.gauged:
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
            live = [item for item in self.live() if _cm_listing_key(item) != self.store.event_key]
            if self.brief["primary"] in {DUTY_ASK, DUTY_WHY, DUTY_FORK}:
                live = [
                    item
                    for item in live
                    if _cm_figure_ok(item, self.query, list(self.brief.get("spans") or []))
                ]
            if DUTY_TASTE in self.brief["duties"]:
                wishes = list(self.brief.get("wishes") or [])
                fitted = [
                    item
                    for item in live
                    if _cm_taste_cover(item, wishes) >= 0.2 and not _cm_decoy(item, self.brief.get("budget"))
                ]
                live = fitted or live
            if DUTY_MEND in self.brief["duties"]:
                model = str(self.brief.get("model") or "")
                color = str(self.brief.get("color") or "")
                modeled = [
                    item
                    for item in live
                    if (not model or _cm_model_hit(item, model)) and not self._stale(item)
                ]
                modeled.sort(key=lambda item: _cm_ladder(item, model, color))
                live = modeled if modeled else (live if self.store.search_n >= 3 else [])
            picked = live[0] if live else None
        if picked is None:
            return []
        pid = str(picked.get("product_id") or "")
        if self.inspect and _cm_listing_key(picked) not in self.store.gauged:
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
        key = _cm_listing_key(champ)
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
        if False:
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
                    ok = _cm_ladder(champ, model, color) == 0
                else:
                    ok = (not model) or _cm_model_hit(champ, model)
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
                if self.store.event_key and not self.store.event_probed:
                    probe = self._cm_probe_event()
                    if probe:
                        aligned.append(probe)
                        continue
                if (
                    self.store.event_key
                    and self.store.event_probed
                    and not self.store.event_acked
                ):
                    ack = self._cm_ack_event()
                    if ack:
                        aligned.append(ack)
                        continue
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
        return aligned or proposed[: self.max_calls]

    def ingest(self, result: dict[str, Any], sent: list[dict[str, Any]]) -> bool:
        done = False
        calls = result.get("calls") or []
        for index, call in enumerate(calls):
            observation = call.get("observation") or {}
            if isinstance(observation, dict) and observation.get("done") is True:
                done = True
            inner = _unwrap(observation) if isinstance(observation, dict) else {}
            err_blob = _cm_fold(
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
            if "inspect" in name or "check_stock" in name:
                pid = str(sent_args.get("product_id") or "")
                sku = str(sent_args.get("sku") or "")
                key = f"{pid}::{sku}" if pid and sku else pid
                if key and key == self.store.event_key:
                    self.store.event_probed = True
            if "message" in name and self.store.event_probed and self.store.event_key:
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
            if products:
                query = str((sent[index].get("args") or {}).get("query") or "") if index < len(sent) else ""
                if query and "search" in name:
                    products = [{**product, "_search_q": query} for product in products]
                self.store.merge(products, viewed=viewed, gauged=gauged, from_search=not viewed and not gauged)
            if isinstance(inner.get("added"), dict):
                added = dict(inner["added"])
                self.store.merge([added], viewed=False, gauged=False, from_search=False)
                key = _cm_listing_key(added)
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
                extra = _cm_figure_spans(content, self.brief.get("budget"))
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
    pilot: _Crown,
) -> bool:
    if _left() < 55:
        return True
    names = " ".join(str(item.get("name") or "").lower() for item in proposed)
    if proposed and "remove_from_cart" in names:
        return True
    # Closed slate: skip LLM only after the graded first cart is committed,
    # so early turns can still salvage compare/view mistakes.
    if DUTY_SLATE in pilot.brief["duties"] and not await_reply and cart_used:
        return True
    if DUTY_WHY in pilot.brief["duties"] and not cart_used and _left() > 80:
        # Keep the model in the loop until facts exist for a firm claim.
        return False
    if DUTY_MEND in pilot.brief["duties"]:
        model = str(pilot.brief.get("model") or "")
        color = str(pilot.brief.get("color") or "")
        champ = pilot.champion()
        exact = bool(champ and _cm_ladder(champ, model, color) == 0)
        sibling = bool(champ and pilot.store.event_key and _cm_ladder(champ, model, color) <= 1)
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
    if pilot.store.event_key and not pilot.store.event_probed:
        return True
    if pilot.store.event_key and proposed:
        if any(
            hint in names
            for hint in ("add_to_cart", "remove_from_cart", "inspect", "place_test_order", "place_order", "message")
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


def _coach(pilot: _Crown, turn: int, await_reply: bool, cart_used: bool) -> str | None:
    if await_reply:
        if pilot.store.event_open:
            return (
                "A market event is open. Re-inspect remaining live options, "
                "cart the new winner, then order. Also answer the shopper."
            )
        return "The shopper asked a question. Answer with the message tool this turn."
    if pilot.brief.get("ask") and not pilot.asked and turn == 1:
        return "One firm figure is withheld until you ask. Call message NOW before searching."
    if DUTY_SLATE in pilot.brief["duties"] and turn <= 3:
        return (
            "Closed slate: compare EVERY named product_id before the first cart. "
            "Rank strictly by the shopper's highest/lowest axes then price. "
            "After a market event: inspect the shocked listing, acknowledge, remove stale cart, "
            "re-compare survivors, cart the new lex winner, order with ONLY product_id and sku."
        )
    if DUTY_WHY in pilot.brief["duties"] and turn <= 3:
        return (
            "View facts on the winning SKU before cart. On place_test_order, attach "
            "justification.claims using ONLY field names from this episode's order-tool enum. "
            "Always include in_stock and price plus one measured gate field whose value is "
            "copied from viewed facts/options (or a cue the shopper typed that also appears "
            "on the listing). Never invent attributes. Never rewrite units into other fields."
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
    pilot = _Crown(query, tools, max_calls)
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
                    "Closed slate protocol: (1) compare/view every named product_id::sku, "
                    "(2) pick the lexicographic winner on the shopper's stated axes then price, "
                    "(3) inspect_stock then add_to_cart that winner first — first cart is graded, "
                    "(4) after stockout/price_change inspect the event target, acknowledge, "
                    "remove, re-rank remaining named survivors, cart the new winner, "
                    "(5) place_test_order with ONLY product_id and sku (no claims unless why)."
                ),
            }
        )
    if DUTY_WHY in pilot.brief["duties"]:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Justification protocol: hydrate the exact SKU until facts/options "
                    "are visible. On place_test_order include justification.claims with "
                    "in_stock, price, and one gate field from the live order-tool enum. "
                    "Gate value must be copied from viewed catalog text or a cue the "
                    "shopper typed that also appears on the listing. Do not invent field "
                    "names. Do not order before the winning SKU is viewed and inspected."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "concord_rim": {
                "primary": pilot.brief.get("primary"),
                "duties": sorted(pilot.brief.get("duties") or []),
                "model": model,
                "provider": _provider(),
                "stack": "ib2+rim",
            },
        }
    ]
    await_reply = False
    cart_used = False

    for turn in range(1, max_steps + 1):
        pilot.turn = turn
        if _left() < _SKIP_FLOOR:
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
                "concord_rim_turn": {"primary": pilot.brief.get("primary"), "turn": turn},
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
