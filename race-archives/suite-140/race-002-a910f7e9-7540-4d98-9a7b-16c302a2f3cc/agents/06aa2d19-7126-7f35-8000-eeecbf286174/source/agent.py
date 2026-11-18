from __future__ import annotations

import json
import math
import re
from collections import Counter
from os import getenv
from typing import Any


_FAST = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_PARSE = {
    "chutes": "Qwen/Qwen3.6-27B-TEE",
    "openrouter": "qwen/qwen3.6-27b",
}
_STRONG = {
    "chutes": "Qwen/Qwen3.5-397B-A17B-TEE",
    "openrouter": "qwen/qwen3.5-397b-a17b",
}
_HEAVY = {
    "chutes": "moonshotai/Kimi-K2.6-TEE",
    "openrouter": "moonshotai/kimi-k2.6",
}
_RECOVER = {
    "chutes": "zai-org/GLM-5.2-TEE",
    "openrouter": "z-ai/glm-5.2",
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

_BM25_K1 = 0.9
_BM25_B = 0.4
_RRF_K = 60.0
_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._-]+)")
_NUM = re.compile(r"(\d+(?:\.\d+)?)")
_HZ = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hz|hertz)\b", re.I)
_INCH = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(?:\"|''|in(?:ch(?:es)?)?)\b",
    re.I,
)
_BUDGET = re.compile(
    r"(?:at or under|at or below|under|below|up to|upto|budget|<=)\s*"
    r"(\d+(?:\.\d+)?)",
    re.I,
)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.I | re.S)
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
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
)
_CLAIM_FIELDS = (
    "brand",
    "category",
    "in_stock",
    "price",
    "refresh_rate",
    "resolution",
    "screen_size",
)
_SOFT_CLAIM = (
    "brand",
    "streaming",
    "microphone",
    "privacy_cover",
    "autofocus",
    "hdr",
)

_PROXY = None


def _proxy():
    global _PROXY
    if _PROXY is None:
        from src.agent.proxy_client import ProxyClient

        _PROXY = ProxyClient(timeout=120, max_retries=2)
    return _PROXY


def _provider() -> str:
    return getenv("INFERENCE_PROVIDER", "chutes") or "chutes"


def _normalize_model(raw: str) -> str:
    token = (raw or "").strip()
    if not token:
        return token
    if _provider() == "openrouter":
        return _TO_OPENROUTER.get(token, token)
    return _TO_CHUTES.get(token, token)


def _model(role: str = "fast") -> str:
    override = getenv("SANDBOX_MODEL")
    if override:
        return _normalize_model(override)
    tables = {
        "fast": _FAST,
        "parse": _PARSE,
        "strong": _STRONG,
        "heavy": _HEAVY,
        "recover": _RECOVER,
    }
    table = tables.get(role, _FAST)
    return table.get(_provider(), table["chutes"])


def _lane_model(lane: str) -> str:
    if lane == LANE_RECALL:
        return _model("fast")
    return _model("heavy")


def _reserve_models(primary: str) -> list[str]:
    pool = (
        _RESERVE_OPENROUTER if _provider() == "openrouter" else _RESERVE_CHUTES
    )
    ordered = [primary]
    for item in pool:
        if item not in ordered:
            ordered.append(item)
    return ordered[:3]


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fold(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _tokens(text: str) -> list[str]:
    return [token for token in _fold(text).split() if len(token) > 1]


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


def _loads_json(text: str) -> dict[str, Any] | None:
    blob = (text or "").strip()
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


def _shopper_text(query: str) -> str:
    return re.split(r"\n\s*Task rules:", query or "", maxsplit=1)[0].strip()


def _named_keys(text: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for product_id, sku in _KEY_RE.findall(text or ""):
        sku = sku.rstrip(".,;:")
        key = f"{product_id}::{sku}"
        if not sku or key in seen:
            continue
        seen.add(key)
        found.append({"product_id": product_id, "sku": sku})
    return found


def _variant_key(product: dict[str, Any]) -> str:
    product_id = str(product.get("product_id") or product.get("id") or "")
    sku = str(product.get("sku") or "")
    if product_id and sku:
        return f"{product_id}::{sku}"
    return product_id


def _facts(product: dict[str, Any]) -> dict[str, Any]:
    raw = (
        product.get("facts")
        or product.get("specification")
        or product.get("attributes")
        or {}
    )
    return raw if isinstance(raw, dict) else {}


def _doc_text(product: dict[str, Any]) -> str:
    parts = [
        product.get("title"),
        product.get("brand"),
        product.get("options"),
        product.get("description"),
        json.dumps(_facts(product), sort_keys=True),
        json.dumps(product.get("category_path") or [], sort_keys=True),
    ]
    return _fold(" ".join(_clean(part) for part in parts if part))


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
        text = _fold(value)
        if text in {"false", "0", "no", "oos", "out of stock", "unavailable"}:
            return False
        if text in {"true", "1", "yes", "in stock", "available"}:
            return True
        if value is None:
            return True
        return bool(value)
    availability = _fold(product.get("availability") or product.get("stock"))
    if not availability:
        return True
    return availability not in {"out of stock", "oos", "unavailable", "false", "0"}


def _number(text: Any) -> float | None:
    found = _NUM.search(str(text or ""))
    if not found:
        return None
    try:
        return float(found.group(1))
    except ValueError:
        return None


def _hz(product: dict[str, Any]) -> float | None:
    chunks = [
        product.get("options"),
        _facts(product).get("refresh_rate"),
        _facts(product).get("refresh"),
        product.get("title"),
        product.get("description"),
    ]
    best: float | None = None
    for chunk in chunks:
        for match in _HZ.findall(str(chunk or "")):
            try:
                value = float(match)
            except ValueError:
                continue
            if best is None or value > best:
                best = value
    return best


def _inches(product: dict[str, Any]) -> float | None:
    facts = _facts(product)
    hydrated = bool(product.get("_hydrated") or facts)
    chunks = [product.get("options")]
    if not hydrated:
        chunks.append(product.get("title"))
    for chunk in chunks:
        found = _INCH.findall(str(chunk or ""))
        if found:
            try:
                return float(found[-1])
            except ValueError:
                pass
    for key in ("screen_size", "size", "display_size"):
        raw = facts.get(key)
        numbers = [float(token) for token in _NUM.findall(str(raw or ""))]
        distinct = {round(item, 1) for item in numbers if 4 <= item <= 100}
        if len(distinct) > 1:
            continue
        found = _INCH.findall(str(raw or ""))
        if found:
            try:
                return float(found[-1])
            except ValueError:
                continue
        value = _number(raw)
        if value is not None and 4 <= value <= 100:
            return value
    return None


def _resolution_class(text: Any) -> str | None:
    blob = str(text or "").lower()
    if not blob:
        return None
    if re.search(r"\b2160\b|2160p|\b4k\b|uhd", blob):
        return "4k"
    if re.search(r"\b1440\b|1440p|\bqhd\b|\b2k\b", blob):
        return "qhd"
    if re.search(r"\b1080\b|1080p|1920\s*[x×]\s*1080|\bfhd\b|full hd", blob):
        return "fhd"
    if re.search(r"\b720\b|720p|1280\s*[x×]\s*720", blob):
        return "hd"
    if re.search(r"\bhd\b", blob):
        return "hd"
    return None


def _axis_value(product: dict[str, Any], name: str) -> float | None:
    key = _fold(name).replace(" ", "_")
    if key in {"price", "cost"}:
        return _price_of(product)
    if "refresh" in key or key in {"hz"}:
        return _hz(product)
    if "screen" in key or key in {"size", "inches"}:
        return _inches(product)
    facts = _facts(product)
    if key in facts:
        return _number(facts.get(key))
    for fact_key, fact_value in facts.items():
        if key in _fold(fact_key).replace(" ", "_"):
            return _number(fact_value)
    return _number(product.get(key))


def _okapi(
    query_tf: Counter[str],
    doc_tf: Counter[str],
    df: Counter[str],
    docs: int,
    doc_len: int,
    avgdl: float,
) -> float:
    if not query_tf or not doc_tf or docs <= 0:
        return 0.0
    score = 0.0
    length_norm = 1.0 - _BM25_B + _BM25_B * (doc_len / max(avgdl, 1.0))
    for token, qtf in query_tf.items():
        tf = doc_tf.get(token, 0)
        if tf <= 0:
            continue
        idf = math.log((docs - df[token] + 0.5) / (df[token] + 0.5) + 1.0)
        tf_norm = (tf * (_BM25_K1 + 1.0)) / (tf + _BM25_K1 * length_norm)
        score += qtf * idf * tf_norm
    return score


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _rrf(result_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    keep: dict[str, dict[str, Any]] = {}
    for rows in result_lists:
        for rank, product in enumerate(rows, start=1):
            key = _variant_key(product) or str(
                product.get("product_id") or product.get("id") or rank
            )
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            keep[key] = product
    return [
        keep[key]
        for key, _score in sorted(scores.items(), key=lambda item: -item[1])
    ]


def _pareto(
    products: list[dict[str, Any]],
    axes: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    if not axes or len(products) < 2:
        return list(products)

    def vector(product: dict[str, Any]) -> list[float | None]:
        values: list[float | None] = []
        for name, direction in axes:
            value = _axis_value(product, name)
            if value is None:
                values.append(None)
            elif direction == "asc":
                values.append(-value)
            else:
                values.append(value)
        return values

    scored = [(product, vector(product)) for product in products]
    kept: list[dict[str, Any]] = []
    for index, (product, values) in enumerate(scored):
        dominated = False
        for other_index, (_other, other_values) in enumerate(scored):
            if other_index == index:
                continue
            better = False
            worse = False
            for left, right in zip(values, other_values):
                if left is None or right is None:
                    continue
                if right > left:
                    better = True
                elif right < left:
                    worse = True
            if better and not worse:
                dominated = True
                break
        if not dominated:
            kept.append(product)
    return kept or list(products)


def _lex_sort(
    products: list[dict[str, Any]],
    axes: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    def key(product: dict[str, Any]) -> tuple:
        parts: list[float] = []
        for name, direction in axes:
            value = _axis_value(product, name)
            missing = value is None
            if missing:
                parts.extend([1.0, 0.0])
                continue
            parts.append(0.0)
            parts.append(-value if direction == "desc" else value)
        return tuple(parts)

    return sorted(products, key=key)


def _axes_from_text(text: str) -> list[tuple[str, str]]:
    lowered = (text or "").lower()
    axes: list[tuple[str, str]] = []
    patterns = (
        (r"highest ([a-z ]{3,24})", "desc"),
        (r"largest ([a-z ]{3,24})", "desc"),
        (r"biggest ([a-z ]{3,24})", "desc"),
        (r"most ([a-z ]{3,24})", "desc"),
        (r"lowest ([a-z ]{3,24})", "asc"),
        (r"smallest ([a-z ]{3,24})", "asc"),
        (r"cheapest ([a-z ]{3,24})?", "asc"),
    )
    seen: set[str] = set()
    for pattern, direction in patterns:
        for match in re.finditer(pattern, lowered):
            raw = (match.group(1) or "price").strip()
            name = _canon_axis(raw)
            if name in seen:
                continue
            seen.add(name)
            axes.append((name, "asc" if "cheap" in raw or direction == "asc" else direction))
    if "cheapest" in lowered or "lowest price" in lowered:
        if "price" not in seen:
            axes.append(("price", "asc"))
    return axes


def _canon_axis(raw: str) -> str:
    token = _fold(raw)
    if "refresh" in token or token in {"hz"}:
        return "refresh_rate"
    if "screen" in token or "inch" in token or token == "size":
        return "screen_size"
    if "price" in token or "cost" in token or "cheap" in token:
        return "price"
    return token.replace(" ", "_") or "price"


def _wants_reason(query: str) -> bool:
    return bool(
        re.search(
            r"\bwhy\b|walk me through|justif|reason i can trust|give me a reason|"
            r"tell me why|a reason i can",
            _shopper_text(query),
            re.I,
        )
    )


def _color_in(text: str) -> str:
    blob = f" {_fold(text)} "
    for color in _COLORS:
        if f" {color} " in blob:
            return color
    return ""


def _model_in(text: str) -> str:
    match = re.search(
        r"prefer(?:s)? the\s+([a-z]+)\s+([A-Za-z0-9._-]{2,})",
        text or "",
        re.I,
    )
    if match:
        return match.group(2).rstrip(".,;")
    match = re.search(
        r"\bmodel(?:\s+code)?\s+([A-Za-z0-9._-]{2,})",
        text or "",
        re.I,
    )
    return match.group(1).rstrip(".,;") if match else ""


def _model_hit(product: dict[str, Any], model: str) -> bool:
    wanted = _compact(model)
    if not wanted:
        return False
    facts = _facts(product)
    sources = [
        facts.get("model"),
        product.get("options"),
        product.get("title"),
        facts.get("product_model"),
    ]
    for source in sources:
        text = _clean(source)
        if not text:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(model)}(?![a-z0-9])", text, re.I):
            return True
        compact = _compact(text)
        index = compact.find(wanted)
        if index < 0:
            continue
        after = compact[index + len(wanted) :]
        if after and after[0].isalnum():
            continue
        if index and compact[index - 1].isalnum():
            continue
        return True
    return False


def _color_hit(product: dict[str, Any], color: str) -> bool:
    wanted = _fold(color)
    if not wanted:
        return False
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


def _ladder(
    product: dict[str, Any],
    model: str,
    color: str,
) -> int:
    has_model = _model_hit(product, model) if model else True
    has_color = _color_hit(product, color) if color else True
    if has_model and has_color:
        return 0
    if has_model:
        return 1
    return 2


def _kind_token(text: str) -> str:
    blob = _fold(text)
    for noun in (
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
    ):
        if re.search(rf"\b{noun}s?\b", blob):
            return noun
    return ""


def _kind_category(kind: str) -> str:
    return {
        "smartphone": "Smartphones",
        "tablet": "Tablets",
        "monitor": "Monitors",
        "headphone": "Headphones",
        "headset": "Headphones",
        "earphone": "Headphones",
        "laptop": "Laptops",
        "camera": "Cameras",
        "watch": "Watches",
        "speaker": "Speakers",
    }.get(_fold(kind), "")


_BRAND_SKIP = {
    "a",
    "an",
    "and",
    "at",
    "below",
    "can",
    "choose",
    "compare",
    "find",
    "for",
    "full",
    "get",
    "give",
    "hd",
    "fhd",
    "i",
    "if",
    "in",
    "me",
    "need",
    "only",
    "or",
    "php",
    "please",
    "qhd",
    "remaining",
    "stock",
    "task",
    "the",
    "these",
    "ties",
    "uhd",
    "under",
    "usb",
    "walk",
    "want",
    "you",
    "among",
}


def _brand_token(text: str) -> str:
    for match in re.finditer(
        r"\b([A-Z][A-Za-z0-9.&-]*(?:\s+[A-Z][A-Za-z0-9.&-]*){0,2})\b",
        text or "",
    ):
        phrase = _clean(match.group(1)).rstrip(".")
        tokens = _fold(phrase).split()
        if not tokens or any(token in _BRAND_SKIP for token in tokens):
            continue
        if _kind_token(phrase):
            continue
        if re.search(r"\d", phrase) and _resolution_class(phrase):
            continue
        return phrase
    return ""


def _spec_query_bits(text: str) -> list[str]:
    bits: list[str] = []
    blob = text or ""
    if re.search(r"720p", blob, re.I):
        bits.append("720p")
    elif re.search(r"1080p|full hd", blob, re.I):
        bits.append("1080p")
    elif re.search(r"1440p|\bqhd\b", blob, re.I):
        bits.append("1440p")
    inch = _INCH.search(blob)
    if inch:
        bits.append(f"{inch.group(1)} inch")
    return bits


def _kind_hit(product: dict[str, Any], kind: str) -> bool:
    token = _fold(kind)
    if not token:
        return True
    facts = _facts(product)
    blob = _fold(
        " ".join(
            [
                str(product.get("title") or ""),
                str(product.get("brand") or ""),
                str(product.get("options") or ""),
                str(facts.get("phone_type") or ""),
                str(facts.get("device_type") or ""),
                str(facts.get("product_type") or ""),
                " ".join(_clean(value) for value in facts.values()),
            ]
        )
    )
    if token == "smartphone" and "feature phone" in blob:
        return False
    if token in blob or f"{token}s" in blob:
        return True
    if token == "headphone" and any(
        name in blob for name in ("earphone", "earbud", "headset")
    ):
        return True
    if token == "smartphone" and any(
        name in blob
        for name in (
            "android",
            "ios",
            "harmonyos",
            "hyperos",
            "tizen",
            "blackberry",
            "windows phone",
            "symbian",
        )
    ):
        return True
    return False


def _junk(product: dict[str, Any], kind: str) -> bool:
    blob = _fold(
        " ".join(
            [
                str(product.get("title") or ""),
                str(product.get("options") or ""),
                " ".join(str(part) for part in (product.get("category_path") or [])),
            ]
        )
    )
    if kind in {"smartphone", "tablet", "monitor"} and any(
        hint in blob
        for hint in (
            "webcam",
            "mouse pad",
            "mousepad",
            "backpack",
            "phone case",
            "replacement screen",
            "screen only",
            "charger",
            "desk mat",
        )
    ):
        return True
    return bool(kind) and not _kind_hit(product, kind)


def _brand_hit(product: dict[str, Any], brand: str) -> bool:
    wanted = _fold(brand)
    if not wanted:
        return True
    got = _fold(product.get("brand") or "")
    if got and (wanted in got or got in wanted):
        return True
    return wanted in _fold(product.get("title") or "")


def _spec_hit(product: dict[str, Any], goal: dict[str, Any]) -> bool:
    spec_res = goal.get("spec_res")
    spec_inch = goal.get("spec_inch")
    if spec_res:
        got = _resolution_from_product(product)
        if got != spec_res:
            return False
    if spec_inch is not None:
        inches = _inches(product)
        if inches is None or abs(inches - float(spec_inch)) > 0.2:
            return False
    return True


def _queried_model(product: dict[str, Any], model: str) -> bool:
    wanted = _fold(model)
    if not wanted:
        return False
    blob = _fold(product.get("_search_q") or "")
    if not blob:
        return False
    if wanted in blob.split():
        return True
    return _compact(wanted) in _compact(blob)


def _sku_stem(sku: Any) -> str:
    text = str(sku or "")
    return text.rsplit("-", 1)[0] if text else ""


def _identity(product: dict[str, Any], goal: dict[str, Any], *, strict: bool) -> bool:
    if goal.get("named_keys"):
        return True
    kind = str(goal.get("kind_token") or "")
    if _junk(product, kind):
        return False
    if kind and not _kind_hit(product, kind):
        return False
    model = str(goal.get("model") or "")
    if goal.get("kind") == "recovery" and model:
        if not (_model_hit(product, model) or _queried_model(product, model)):
            return False
    brand = str(goal.get("brand") or "")
    if brand and goal.get("kind") != "recovery":
        titled = bool(product.get("title") or product.get("brand") or strict)
        if titled and not _brand_hit(product, brand):
            return False
    if strict and goal.get("kind") == "justification" and not _spec_hit(product, goal):
        return False
    return True


def _price_text(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _screen_text(inches: float) -> str:
    if abs(inches - round(inches)) < 0.05:
        return f"{int(round(inches))}in"
    return f"{inches:g}in"


def _claim_enum(tool: dict[str, Any] | None) -> list[str]:
    spec = (_props(tool).get("justification") or {}).get("properties") or {}
    claims = spec.get("claims") or {}
    items = claims.get("items") or {}
    field = (items.get("properties") or {}).get("field") or {}
    enum = field.get("enum")
    if isinstance(enum, list) and enum:
        return [str(item) for item in enum]
    return list(_CLAIM_FIELDS) + [name for name in _SOFT_CLAIM if name not in _CLAIM_FIELDS]


def _resolution_from_product(product: dict[str, Any]) -> str | None:
    facts = _facts(product)
    for key in ("display", "resolution", "screen"):
        klass = _resolution_class(facts.get(key))
        if klass:
            return klass
    if product.get("_hydrated"):
        return None
    return _resolution_class(product.get("title"))


def _evidence_claims(
    product: dict[str, Any],
    fields: list[str],
    allowed: list[str],
) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    facts = _facts(product)
    permit = {item.lower() for item in allowed}
    for field in fields:
        if field not in permit:
            continue
        value: str | None = None
        if field == "in_stock":
            value = "true" if _in_stock(product) else "false"
        elif field == "price":
            price = _price_of(product)
            value = _price_text(price) if price is not None else None
        elif field == "resolution":
            value = _resolution_from_product(product)
        elif field == "screen_size":
            inches = _inches(product)
            value = _screen_text(inches) if inches is not None else None
        elif field == "refresh_rate":
            hz = _hz(product)
            if hz is None:
                value = None
            elif abs(hz - round(hz)) < 0.05:
                value = str(int(round(hz)))
            else:
                value = str(hz)
        elif field == "brand":
            value = _clean(product.get("brand")) or _clean(facts.get("brand")) or None
        elif field == "category":
            path = product.get("category_path") or []
            if isinstance(path, list) and path:
                value = _clean(path[-1]) or None
        elif field in _SOFT_CLAIM:
            blob = _fold(" ".join(_clean(item) for item in facts.values()))
            token = field.replace("_", " ")
            if token in blob or field in blob:
                value = "true"
        if not value:
            continue
        claims.append({"field": field, "value": str(value)})
    return claims


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
        key in node
        for key in ("title", "price", "sku", "brand", "facts", "in_stock")
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


def _unwrap(observation: dict[str, Any]) -> dict[str, Any]:
    nested = observation.get("observation")
    if isinstance(nested, dict):
        return nested
    return observation


def _need_fields(query: str) -> list[str]:
    text = (query or "").lower()
    fields = ["in_stock", "price"]
    if re.search(
        r"720p|1080p|1440p|2160p|\b4k\b|\bfhd\b|\bqhd\b|full hd|\bhd\b|resolution",
        text,
    ):
        fields.append("resolution")
    if re.search(r"\d[\d.]*\s*-?\s*inch|screen size|largest screen", text):
        fields.append("screen_size")
    if re.search(r"refresh|hz\b", text):
        fields.append("refresh_rate")
    return fields


_LANE_CUES: list[tuple[str, tuple[str, ...]]] = [
    (
        LANE_INTENT,
        (
            "mix of firm requirements and nice-to-haves",
            "soft preference, subordinate",
            "preferred-brand",
            "later correction does not repair an initial choice",
            "honor that named brand when a fully compliant",
        ),
    ),
    (
        LANE_RECALL,
        (
            "reformulate queries and inspect candidates",
            "survey the realistic options",
            "several distinct products, not one lucky hit",
            "search the catalog for what the user described",
        ),
    ),
    (
        LANE_CONSTRAINT,
        (
            "every requirement must hold for the exact variant",
            "resolution names here denote pixel classes",
            "check the specific sku's option, price, and stock",
            "i will tell you if you ask",
        ),
    ),
    (
        LANE_PREFERENCE,
        (
            "soft preferences matter",
            "blindly buying the cheapest valid item",
            "choose a strong match across several preference axes",
            "genuine intended purchase, not an exploratory",
        ),
    ),
    (
        LANE_RANK,
        (
            "stated tradeoff priority",
            "strictly worse than another available option",
            "highest advertised maximum refresh rate",
            "first cart choice must be the best-ranked",
            "compare only these candidate",
        ),
    ),
    (
        LANE_RECOVER,
        (
            "relaxable only after the exact item becomes unavailable",
            "least-relaxed in-stock option",
            "try another color of the same model before changing models",
            "conflicting title does not override",
        ),
    ),
    (
        LANE_JUSTIFY,
        (
            "when placing the order, include a structured justification",
            "ground claims in raw product facts",
            "do not justify from marketing title words alone",
            "trustworthy reason",
            "walk me through why",
        ),
    ),
]

_BASE_PLAYBOOK = """
You are a shopping agent in a generated environment.

# Contract
- Use ONLY tools supplied in this episode.
- Continue until an observation reports done=true.
- Respect max_steps and max_calls_per_turn.
- When max_calls_per_turn > 1, batch independent search/filter/compare calls.

# Standard loop
1) Discover — search / filter with value-side keywords (brand, model, kind)
2) Inspect — view / compare / inspect_stock on concrete SKUs
3) Decide — add_to_cart / remove_from_cart (first cart is scored)
4) Confirm — re-inspect after market events
5) Finish — place_test_order once the live winner is verified

# Firm vs soft
Firm: category, budget, in-stock, named model/color, stated spec.
Soft never overrides firm.

# Matching
Follow the matcher brief when present. It ranks with BM25+RRF, drops
Pareto-dominated rows, and applies the color-before-model recovery ladder.
Do not invent product IDs. Search matches VALUES, not JSON field names.

# Shopper dialogue
If a shopper user_message asks a question, answer with the message tool in
the NEXT turn before further cart/order actions.

# Market events
After stockout or price_change, remove the stale cart SKU, inspect the next
live winner, then cart and order. Never order the event target.
""".strip()

_LANE_PLAYBOOK: dict[str, str] = {
    LANE_INTENT: """
# Intent decomposition
Parse FIRM (category, spec, budget, stock) vs SOFT (preferred brand).
First add_to_cart must honor the preferred brand when a fully firm-compliant
SKU exists. A later swap does not repair a bad first cart.
On price jump: remove, re-search under the new budget, still prefer brand
only when fully compliant. Then inspect_stock and place_test_order.
""".strip(),
    LANE_RECALL: """
# Retrieval recall
Search the shopper phrase, then a short reformulation (brand+kind or
category filter). Shortlist several distinct in-stock products. Reject
accessories. View/compare, pick the best in-budget hit, order immediately.
""".strip(),
    LANE_CONSTRAINT: """
# Constraint satisfaction
If a firm spec is revealed only when asked, call message FIRST.
Every firm requirement must hold on the exact SKU (options + price + stock).
Titles mix good/bad variants — view then inspect_stock. Resolution classes:
hd/fhd/qhd/4k from facts, not marketing titles. If the variant sells out,
do not take a sibling that misses a firm spec.
""".strip(),
    LANE_PREFERENCE: """
# Preference reasoning
Hard: category, budget, stock, real device. Soft: use-case axes.
Compare candidates on soft axes before the first cart. First cart must be a
genuine preference-fit, not the cheapest placeholder.
""".strip(),
    LANE_RANK: """
# Ranking
Restrict to the named product_id::sku candidates. Compare them first.
Rank by the stated axes (typical: highest refresh from the variant options,
then largest screen, then lowest price). Drop dominated rows.
FIRST add_to_cart MUST be the current #1 in-stock in-budget option.
After stockout, remove, inspect the next #1, cart, then order.

CRITICAL — place_test_order on ranking:
- Pass ONLY product_id and sku.
- Do NOT attach justification/claims. Optional untruthful claims hard-fail
  even when the ranked pick is correct.
""".strip(),
    LANE_RECOVER: """
# Recovery
Never relax: kind/category, budget, in-stock.
Search "{color} {model}" then "{model}" (the model code may be absent from
the rendered title — still compare those search hits). Prefer exact
model+color first. After it disappears: same model, other color (prefer a
SKU that shares the event-target stem), only then other models.
Inspect the replacement after the event. Never order the event target.
""".strip(),
    LANE_JUSTIFY: """
# Justification
Search "{brand} {spec} {kind}" and filter brand+category. View/compare raw
facts BEFORE the first cart. First cart must be a spec+brand match so the
intended event can fire.
place_test_order MUST include justification.claims from viewed facts:
- required: in_stock, live price, and the shopper spec (resolution as
  hd/fhd/qhd/4k OR screen_size as {n}in)
- plus brand when the brand field is observed
Never claim from title decoys. Never invent hdr. Reject accessories.
""".strip(),
}

_IDLE = (
    "No tool call received. Choose a supplied tool now. "
    "If you have not searched yet, call search or filter.",
    "Still no tool call. Progress: search/filter, view a candidate, answer any "
    "pending shopper question with message, or place_test_order if the cart "
    "already holds a verified in-stock SKU.",
)

_GOAL_REGEX: list[tuple[str, re.Pattern[str]]] = [
    (LANE_RANK, re.compile(r"compare only these candidate", re.I)),
    (LANE_JUSTIFY, re.compile(r"walk me through why|tell me why|trustworthy reason", re.I)),
    (LANE_RECOVER, re.compile(r"if that exact .* disappears|strongly prefer", re.I)),
    (LANE_INTENT, re.compile(r"ideally,? i prefer|firm requirements matter more", re.I)),
    (LANE_PREFERENCE, re.compile(r"i'?d rather pay|for daily|workout|soft preference", re.I)),
    (LANE_CONSTRAINT, re.compile(r"i will tell you if you ask", re.I)),
]


def classify_lane(query: str) -> str:
    goal = _parse_fallback(query)
    kind = str(goal.get("kind") or "")
    if kind == "ranking" or goal.get("named_keys"):
        return LANE_RANK
    if kind == "justification":
        return LANE_JUSTIFY
    if kind == "recovery":
        return LANE_RECOVER
    lowered = (query or "").lower()
    rules = (
        lowered.split("task rules:", 1)[1] if "task rules:" in lowered else lowered
    )
    chosen = LANE_RECALL
    peak = -1
    for lane, cues in _LANE_CUES:
        score = sum(1 for cue in cues if cue in rules or cue in lowered)
        if score > peak:
            peak = score
            chosen = lane
    if peak > 0:
        return chosen
    for lane, pattern in _GOAL_REGEX:
        if pattern.search(lowered):
            return lane
    return LANE_RECALL


def _system_prompt(
    tools: list[dict[str, Any]],
    max_calls: int,
    lane: str,
    policy: dict[str, Any],
) -> str:
    names = [_tool_name(tool) for tool in tools if _tool_name(tool)]
    playbook = _LANE_PLAYBOOK.get(lane, _LANE_PLAYBOOK[LANE_RECALL])
    return (
        f"{_BASE_PLAYBOOK}\n\n"
        f"{playbook}\n\n"
        f"# Runtime\n"
        f"Pipeline: {lane}\n"
        f"Available tools: {', '.join(names) or '(none)'}. "
        f"Emit at most {max_calls} tool call(s) per turn.\n"
        f"max_steps={policy.get('max_steps')} "
        f"max_calls_per_turn={policy.get('max_calls_per_turn', 1)}"
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
    return any(cleaned.startswith(item) or f" {item}" in f" {cleaned}" for item in prefixes)


def _coach(
    lane: str,
    turn: int,
    matcher: "_Matcher",
    proposed: list[dict[str, Any]],
    await_reply: bool,
    cart_used: bool,
    action_names: list[str],
) -> str | None:
    if await_reply:
        return (
            "Coach: the shopper asked a question. Answer with the message tool "
            "in this turn before further cart or order actions."
        )
    blob = " ".join(action_names)
    shopper = _shopper_text(matcher.query).lower()
    if (
        lane == LANE_CONSTRAINT
        and turn == 1
        and "i will tell you if you ask" in shopper
        and "message" not in blob
    ):
        return (
            "Coach: the shopper has one firm hardware spec revealed only if asked. "
            "Call message NOW to ask for that exact requirement before searching."
        )
    if lane == LANE_RANK:
        lines = [
            "Coach [ranking]: compare the listed product_id::sku candidates and "
            "rank by the stated axes before the first add_to_cart. First cart = #1. "
            "On place_test_order pass ONLY product_id and sku — omit justification."
        ]
        if (
            not cart_used
            and "add_to_cart" not in blob
            and ("view" in blob or "compare" in blob)
        ):
            lines.append(
                "Hint [ranking]: you have inspected candidates — add_to_cart ONLY "
                "the best-ranked SKU now (first cart = graded #1)."
            )
        if cart_used:
            lines.append(
                "Hint [ranking]: after recovery, place_test_order with ONLY "
                "product_id and sku — do not attach justification.claims."
            )
        lines.append(matcher.brief(proposed))
        return "\n".join(lines)
    if lane == LANE_JUSTIFY:
        return (
            "Coach [justification]: view/compare facts before the first cart. "
            "Claims allowlist: in_stock, price, one spec field from the goal "
            "(resolution|screen_size|refresh_rate), plus brand when observed. "
            "Values from facts/inspect_stock, not title decoys. "
            "Forbidden: display, size, category, invented hdr.\n"
            f"{matcher.brief(proposed)}"
        )
    if lane == LANE_RECOVER:
        return (
            "Coach [recovery]: search color+model then model. After a stockout, "
            "keep the same model and change color before changing models. "
            "Never order the event target.\n"
            f"{matcher.brief(proposed)}"
        )
    if lane == LANE_INTENT and turn == 1:
        return (
            "Coach: first cart should honor the preferred brand when a fully "
            "firm-compliant SKU exists; otherwise take the best firm-compliant item.\n"
            f"{matcher.brief(proposed)}"
        )
    if lane == LANE_PREFERENCE and turn <= 2 and not cart_used:
        return (
            "Coach: compare soft-preference axes before first cart. First cart "
            "must be a real preference-fit purchase, not the cheapest placeholder.\n"
            f"{matcher.brief(proposed)}"
        )
    if lane == LANE_RECALL and turn == 3 and "search" in blob:
        return (
            "Hint [recall]: one small keyword reformulation or filter so the "
            "shortlist covers several distinct products before ordering.\n"
            f"{matcher.brief(proposed)}"
        )
    if proposed:
        return matcher.brief(proposed)
    if turn <= 2 and not any(
        name in blob for name in ("search", "filter", "view", "compare", "message")
    ):
        return (
            "Hint: start with search or filter using value-side keywords from the "
            "shopper goal, then view the best hits."
        )
    return None


def _parse_fallback(query: str) -> dict[str, Any]:
    shopper = _shopper_text(query)
    budget = _BUDGET.search(shopper)
    color = _color_in(shopper)
    model = _model_in(shopper)
    if not model:
        colored = re.search(
            rf"\b(?:{'|'.join(_COLORS)})\s+([A-Za-z0-9._-]{{2,}})",
            shopper,
            re.I,
        )
        if colored:
            model = colored.group(1).rstrip(".,;")
    category = ""
    match = re.search(r"in the (.+?) category", shopper, re.I)
    if match:
        category = _clean(match.group(1))
    kind_token = _kind_token(shopper)
    brand = _brand_token(shopper)
    if not brand:
        brand_match = re.search(
            r"\b(?:prefer(?:s)?|brand)\s+([A-Z][\w.&-]*(?:\s+[A-Z][\w.&-]*){0,2})",
            shopper,
        )
        if brand_match and "prefer the" not in _fold(brand_match.group(0)):
            brand = _clean(brand_match.group(1))
    named = _named_keys(query)
    axes = _axes_from_text(shopper)
    kind = "shop"
    if named:
        kind = "ranking"
    elif _wants_reason(shopper):
        kind = "justification"
    elif model or "disappear" in shopper.lower():
        kind = "recovery"
    spec_res = _resolution_class(shopper)
    spec_inch = None
    inch_hit = _INCH.search(shopper)
    if inch_hit:
        spec_inch = float(inch_hit.group(1))
    color_query = "black" if color == "midnight" and "black" in _fold(shopper) else color
    phrases: list[str] = []
    if model:
        phrases.append(" ".join(part for part in (color_query, model) if part))
        phrases.append(model)
    bits = _spec_query_bits(shopper)
    if brand and kind_token:
        for bit in bits:
            phrases.append(f"{brand} {bit} {kind_token}")
        phrases.append(f"{brand} {kind_token}")
    elif brand:
        phrases.append(brand)
    if kind_token and not brand and not model:
        phrases.append(kind_token)
    if category:
        phrases.append(category)
    phrases = [item for item in dict.fromkeys(phrases) if item]
    if not phrases:
        phrases = [" ".join(_tokens(shopper)[:4]) or "product"]
    return {
        "kind": kind,
        "budget": float(budget.group(1)) if budget else None,
        "in_stock": True,
        "brand": brand,
        "category": category or _kind_category(kind_token),
        "kind_token": kind_token,
        "model": model,
        "named_keys": named,
        "axes": axes,
        "need_reason": _wants_reason(shopper),
        "spec_fields": _need_fields(shopper),
        "spec_res": spec_res,
        "spec_inch": spec_inch,
        "search_phrases": phrases,
        "color": (
            "black"
            if color == "midnight" and "black" in _fold(shopper)
            else color
        ),
    }


def _normalize_objective(parsed: dict[str, Any], query: str) -> dict[str, Any]:
    fallback = _parse_fallback(query)
    kind = str(parsed.get("kind") or fallback["kind"]).lower()
    if kind not in {"ranking", "recovery", "justification", "shop"}:
        kind = fallback["kind"]
    named = parsed.get("named_keys")
    if not isinstance(named, list) or not named:
        named = fallback["named_keys"]
    clean_keys: list[dict[str, str]] = []
    for item in named:
        if isinstance(item, dict) and item.get("product_id") and item.get("sku"):
            clean_keys.append(
                {
                    "product_id": str(item["product_id"]),
                    "sku": str(item["sku"]),
                }
            )
        elif isinstance(item, str) and "::" in item:
            product_id, sku = item.split("::", 1)
            clean_keys.append({"product_id": product_id, "sku": sku})
    if not clean_keys:
        clean_keys = fallback["named_keys"]
    axes_raw = parsed.get("axes")
    axes: list[tuple[str, str]] = []
    if isinstance(axes_raw, list):
        for item in axes_raw:
            if isinstance(item, dict):
                name = _canon_axis(str(item.get("name") or item.get("attribute") or ""))
                direction = str(item.get("direction") or "desc").lower()
                if name:
                    axes.append((name, "asc" if direction == "asc" else "desc"))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                axes.append((_canon_axis(str(item[0])), str(item[1])))
    if not axes:
        axes = fallback["axes"]
    phrases = parsed.get("search_phrases") or fallback["search_phrases"]
    if isinstance(phrases, str):
        phrases = [phrases]
    if fallback["kind"] == "justification" and kind == "shop":
        kind = "justification"
    parsed_fields = [
        str(field)
        for field in (parsed.get("spec_fields") or [])
        if str(field).strip() and str(field) in _CLAIM_FIELDS
    ]
    spec_fields: list[str] = []
    for field in list(fallback["spec_fields"]) + parsed_fields:
        if field not in spec_fields:
            spec_fields.append(field)
    spec_inch = fallback.get("spec_inch")
    if spec_inch is None:
        spec_inch = parsed.get("spec_inch")
    return {
        "kind": kind if not clean_keys else "ranking" if kind == "shop" else kind,
        "budget": parsed.get("budget", fallback["budget"]),
        "in_stock": parsed.get("in_stock", True) is not False,
        "brand": _clean(parsed.get("brand") or fallback["brand"]),
        "category": _clean(parsed.get("category") or fallback["category"]),
        "model": _clean(parsed.get("model") or fallback["model"]),
        "color": _fold(parsed.get("color") or fallback["color"]),
        "named_keys": clean_keys,
        "axes": axes,
        "need_reason": bool(fallback["need_reason"] or parsed.get("need_reason")),
        "spec_fields": spec_fields or fallback["spec_fields"],
        "spec_res": fallback.get("spec_res") or parsed.get("spec_res"),
        "spec_inch": spec_inch,
        "kind_token": _fold(parsed.get("kind_token") or fallback.get("kind_token") or ""),
        "search_phrases": [
            _clean(item) for item in phrases if _clean(item)
        ]
        or fallback["search_phrases"],
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
    return _proxy().post("/inference/chat/completions", json_data=payload)


class _Index:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.lists: list[list[dict[str, Any]]] = []
        self.cart: list[dict[str, Any]] = []
        self.event_target: str | None = None
        self.event_kind: str | None = None
        self.viewed: set[str] = set()
        self.inspected: set[str] = set()
        self.search_n = 0
        self.did_filter = False

    def merge(
        self,
        products: list[dict[str, Any]],
        *,
        viewed: bool,
        inspected: bool,
        from_search: bool,
    ) -> None:
        if from_search and products:
            self.lists.append(products)
        for product in products:
            key = _variant_key(product)
            if not key:
                continue
            current = self.rows.get(key)
            if current is None:
                current = dict(product)
                self.rows[key] = current
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
                pid = str(current.get("product_id") or "")
                if pid:
                    self.viewed.add(pid)
            if inspected:
                current["_inspected"] = True
                self.inspected.add(key)
            for item in self.cart:
                if _variant_key(item) == key:
                    item.update(current)

    def apply_event(self, event: dict[str, Any]) -> None:
        target = event.get("target") or {}
        product_id = str(target.get("product_id") or event.get("product_id") or "")
        sku = str(target.get("sku") or event.get("sku") or "")
        key = f"{product_id}::{sku}" if product_id and sku else product_id
        self.event_target = key or None
        self.event_kind = _fold(event.get("kind") or "")
        patch: dict[str, Any] = {}
        if product_id:
            patch["product_id"] = product_id
        if sku:
            patch["sku"] = sku
        if "stockout" in (self.event_kind or "") or event.get("in_stock_after") is False:
            patch["in_stock"] = False
        if event.get("new_price") is not None:
            patch["price"] = event.get("new_price")
        if patch.get("product_id"):
            self.merge([patch], viewed=False, inspected=False, from_search=False)
            if key:
                self.inspected.discard(key)
                row = self.rows.get(key)
                if row is not None:
                    row["_inspected"] = False

    def fused(self) -> list[dict[str, Any]]:
        fused = _rrf(self.lists) if self.lists else []
        by_key = {
            _variant_key(item): item for item in fused if _variant_key(item)
        }
        by_key.update(self.rows)
        return list(by_key.values())


class _Matcher:
    def __init__(self, query: str, tools: list[dict[str, Any]], max_calls: int) -> None:
        self.query = query
        self.tools = tools
        self.max_calls = max(1, max_calls)
        self.goal = _parse_fallback(query)
        self.index = _Index()
        self.search = _find_tool(tools, "search", "find_product", "search_catalog")
        self.filter = _find_tool(tools, "filter", "filter_catalog")
        self.view = _find_tool(tools, "view", "view_product", "get_product")
        self.compare = _find_tool(tools, "compare", "compare_products")
        self.inspect = _find_tool(tools, "inspect_stock", "check_stock")
        self.cart_view = _find_tool(tools, "inspect_cart", "view_cart")
        self.cart = _find_tool(tools, "add_to_cart", "cart_add")
        self.remove = _find_tool(tools, "remove_from_cart", "cart_remove")
        self.order = _find_tool(tools, "place_test_order", "place_order", "checkout")
        self.message = _find_tool(tools, "message", "talk", "respond")
        for spec in self.goal.get("named_keys") or []:
            key = f"{spec['product_id']}::{spec['sku']}"
            self.index.rows[key] = {
                "product_id": spec["product_id"],
                "sku": spec["sku"],
                "_hydrated": False,
            }

    def parse_with_model(self) -> None:
        prompt = (
            "Extract the shopper objective as JSON with keys: kind "
            "(ranking|recovery|justification|shop), budget (number or null), "
            "in_stock (bool), brand, category, model, color, named_keys "
            "([{product_id, sku}]), axes ([{name, direction desc|asc}]), "
            "need_reason (bool), spec_fields (array of claim field names), "
            "search_phrases (short catalog queries). Use ranking when the "
            "shopper names product_id::sku candidates and tradeoff order. Use "
            "recovery when a preferred model/color may disappear. Use "
            "justification when the shopper asks why. Do not invent product IDs. "
            "Axes must be generic attribute names such as refresh_rate, "
            "screen_size, or price.\n\n"
            f"Shopper:\n{_shopper_text(self.query)}\n\n"
            f"Task rules:\n{self.query}"
        )
        response = _chat(
            model=_model("fast"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only JSON. Prefer attributes stated in the "
                        "shopper text. Never copy catalog answers."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        if response is None:
            return
        message = ((response.get("choices") or [{}])[0].get("message")) or {}
        parsed = _loads_json(_assistant_text(message)) or {}
        self.goal = _normalize_objective(parsed, self.query)
        for spec in self.goal.get("named_keys") or []:
            key = f"{spec['product_id']}::{spec['sku']}"
            self.index.rows.setdefault(
                key,
                {
                    "product_id": spec["product_id"],
                    "sku": spec["sku"],
                    "_hydrated": False,
                },
            )

    def eligible(self, products: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        budget = self.goal.get("budget")
        rows = []
        for product in products if products is not None else self._pool():
            if self.goal.get("in_stock", True) and not _in_stock(product):
                continue
            price = _price_of(product)
            if budget is not None and price is not None and price > float(budget):
                continue
            if not _identity(
                product,
                self.goal,
                strict=bool(product.get("_hydrated")),
            ):
                continue
            rows.append(product)
        return rows

    def _pool(self) -> list[dict[str, Any]]:
        wanted = self.goal.get("named_keys") or []
        fused = self.index.fused()
        if not wanted:
            return fused
        by_key = {_variant_key(item): item for item in fused}
        rows: list[dict[str, Any]] = []
        for spec in wanted:
            key = f"{spec['product_id']}::{spec['sku']}"
            product = by_key.get(key)
            if product is None:
                product = {
                    "product_id": spec["product_id"],
                    "sku": spec["sku"],
                    "_hydrated": False,
                }
            rows.append(product)
        return rows

    def _ladders(self, products: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
        model = str(self.goal.get("model") or "")
        color = str(self.goal.get("color") or "")
        return [(_ladder(product, model, color), product) for product in products]

    def ranked(self) -> list[dict[str, Any]]:
        pool = self.eligible()
        axes = list(self.goal.get("axes") or [])
        if axes:
            pool = _pareto(pool, axes)
            return _lex_sort(pool, axes)
        if self.goal.get("kind") == "recovery" and self.goal.get("model"):
            ranked_pairs = self._ladders(pool)
            if any(level <= 1 for level, _product in ranked_pairs):
                pool = [product for level, product in ranked_pairs if level <= 1]
            target = self.index.event_target or ""
            target_stem = _sku_stem(target.split("::")[-1] if "::" in target else "")
            ranked_pairs = self._ladders(pool)
            ranked_pairs.sort(
                key=lambda item: (
                    item[0],
                    0 if target_stem and _sku_stem(item[1].get("sku")) == target_stem else 1,
                    _price_of(item[1]) or 0.0,
                )
            )
            return [product for _level, product in ranked_pairs]
        query_tf = Counter(
            _tokens(
                " ".join(self.goal.get("search_phrases") or [])
                + " "
                + _shopper_text(self.query)
            )
        )
        docs = [(_tokens(_doc_text(product)), product) for product in pool]
        df: Counter[str] = Counter()
        for tokens, _product in docs:
            df.update(set(tokens))
        avgdl = sum(len(tokens) for tokens, _product in docs) / max(len(docs), 1)
        brand = _fold(self.goal.get("brand") or "")
        scored: list[tuple[tuple[int, float], dict[str, Any]]] = []
        for tokens, product in docs:
            bm25 = _okapi(
                query_tf,
                Counter(tokens),
                df,
                len(docs),
                len(tokens),
                avgdl,
            )
            overlap = _jaccard(set(query_tf), set(tokens))
            brand_rank = 0 if brand and _fold(product.get("brand") or "") == brand else 1
            score = bm25 + 2.0 * overlap
            scored.append(((brand_rank, -score), product))
        scored.sort(key=lambda item: item[0])
        return [product for _score, product in scored]

    def winner(self) -> dict[str, Any] | None:
        ranked = self.ranked()
        if not ranked:
            return None
        target = self.index.event_target
        live = [item for item in ranked if _variant_key(item) != target]
        return (live or ranked)[0]

    def stale(self, product: dict[str, Any] | None) -> bool:
        if product is None:
            return False
        if not _in_stock(product):
            return True
        budget = self.goal.get("budget")
        price = _price_of(product)
        if budget is not None and price is not None and price > float(budget):
            return True
        if self.index.event_target and _variant_key(product) == self.index.event_target:
            return True
        return False

    def brief(self, proposed: list[dict[str, Any]]) -> str:
        ranked = self.ranked()[:6]
        rows = []
        for product in ranked:
            rows.append(
                {
                    "key": _variant_key(product),
                    "title": product.get("title"),
                    "brand": product.get("brand"),
                    "price": product.get("price"),
                    "in_stock": _in_stock(product),
                    "hydrated": bool(product.get("_hydrated")),
                    "inspected": bool(product.get("_inspected")),
                    "refresh_hz": _hz(product),
                    "screen_in": _inches(product),
                    "ladder": _ladder(
                        product,
                        str(self.goal.get("model") or ""),
                        str(self.goal.get("color") or ""),
                    ),
                }
            )
        return (
            "Matcher state (follow unless a later observation contradicts it):\n"
            f"kind={self.goal.get('kind')} budget={self.goal.get('budget')} "
            f"model={self.goal.get('model')} color={self.goal.get('color')} "
            f"axes={self.goal.get('axes')} event={self.index.event_kind} "
            f"event_target={self.index.event_target}\n"
            f"ranked={json.dumps(rows, sort_keys=True)}\n"
            f"recommended_actions={json.dumps(proposed, sort_keys=True)}"
        )

    def propose(self) -> list[dict[str, Any]]:
        cart = self.index.cart[0] if self.index.cart else None
        if cart is not None and self.stale(cart) and self.remove:
            return [
                {
                    "name": _tool_name(self.remove),
                    "args": _bind(
                        self.remove,
                        {
                            "product_id": cart.get("product_id"),
                            "sku": cart.get("sku"),
                        },
                    ),
                }
            ]
        named = self.goal.get("named_keys") or []
        if named and self.compare:
            missing = [
                spec["product_id"]
                for spec in named
                if spec["product_id"] not in self.index.viewed
            ]
            if missing:
                return [
                    {
                        "name": _tool_name(self.compare),
                        "args": _bind(
                            self.compare,
                            {"product_ids": list(dict.fromkeys(missing))[:5]},
                        ),
                    }
                ]
        identity = self.eligible()
        if (not identity or self._needs_recovery_search(identity)) and (
            self.search or self.filter
        ):
            if self.index.search_n < 8:
                searched = self._search_actions()
                if searched:
                    return searched
        if self.compare:
            missing = []
            for product in identity or self.ranked():
                pid = str(product.get("product_id") or "")
                if pid and pid not in self.index.viewed and pid not in missing:
                    missing.append(pid)
                if len(missing) >= 5:
                    break
            if missing:
                return [
                    {
                        "name": _tool_name(self.compare),
                        "args": _bind(self.compare, {"product_ids": missing}),
                    }
                ]
        picked = self.winner()
        if picked is None:
            if self.search and self.index.search_n < 8:
                return self._search_actions()
            return []
        pid = str(picked.get("product_id") or "")
        if not picked.get("_hydrated") and pid not in self.index.viewed:
            if self.compare:
                return [
                    {
                        "name": _tool_name(self.compare),
                        "args": _bind(self.compare, {"product_ids": [pid]}),
                    }
                ]
            if self.view:
                return [
                    {
                        "name": _tool_name(self.view),
                        "args": _bind(self.view, {"product_id": pid}),
                    }
                ]
        key = _variant_key(picked)
        if self.inspect and key not in self.index.inspected:
            return [
                {
                    "name": _tool_name(self.inspect),
                    "args": _bind(
                        self.inspect,
                        {"product_id": pid, "sku": picked.get("sku")},
                    ),
                }
            ]
        if cart is None and self.cart and not self.stale(picked):
            if self.goal.get("kind") == "justification" and not _spec_hit(
                picked, self.goal
            ):
                if self.search and self.index.search_n < 8:
                    return self._search_actions()
                return []
            return [
                {
                    "name": _tool_name(self.cart),
                    "args": _bind(
                        self.cart,
                        {"product_id": pid, "sku": picked.get("sku")},
                    ),
                }
            ]
        if cart is not None and not self.stale(cart) and self.order:
            return [self._order(cart)]
        if picked is not None and self.order and not self.stale(picked):
            return [self._order(picked)]
        return []

    def _needs_recovery_search(self, identity: list[dict[str, Any]]) -> bool:
        if self.goal.get("kind") != "recovery" or not self.goal.get("model"):
            return False
        live = [
            product
            for product in identity
            if _variant_key(product) != self.index.event_target
        ]
        if not live:
            return True
        levels = [
            _ladder(
                product,
                str(self.goal.get("model") or ""),
                str(self.goal.get("color") or ""),
            )
            for product in live
        ]
        if self.index.event_target:
            return not any(level <= 1 for level in levels)
        return False

    def _search_actions(self) -> list[dict[str, Any]]:
        phrases = list(self.goal.get("search_phrases") or ["product"])
        if self.goal.get("model"):
            color = self.goal.get("color") or ""
            model = str(self.goal.get("model"))
            phrases = [
                " ".join(part for part in (color, model) if part),
                model,
            ]
            if self.index.event_target:
                phrases = [model, " ".join(part for part in (color, model) if part)]
        actions: list[dict[str, Any]] = []
        while self.index.search_n < max(len(phrases), 1) and len(actions) < self.max_calls:
            query = phrases[min(self.index.search_n, len(phrases) - 1)]
            self.index.search_n += 1
            if self.search:
                actions.append(
                    {
                        "name": _tool_name(self.search),
                        "args": _bind(
                            self.search,
                            {
                                "query": query,
                                "k": 10,
                                "in_stock": True,
                            },
                        ),
                    }
                )
            else:
                break
        brand = str(self.goal.get("brand") or "")
        if _fold(brand) in _BRAND_SKIP:
            brand = ""
        category = str(self.goal.get("category") or "")
        if (
            self.filter
            and not self.index.did_filter
            and len(actions) < self.max_calls
            and (brand or category)
        ):
            args = _bind(
                self.filter,
                {
                    "brand": brand or None,
                    "category": category or None,
                    "max_price": self.goal.get("budget"),
                    "in_stock": True,
                    "k": 10,
                },
            )
            if args:
                self.index.did_filter = True
                actions.append({"name": _tool_name(self.filter), "args": args})
        if not actions:
            self.index.search_n += 1
        return actions[: self.max_calls]

    def _order(self, product: dict[str, Any]) -> dict[str, Any]:
        if self.order is None:
            return {"name": "", "args": {}}
        wanted: dict[str, Any] = {
            "product_id": product.get("product_id"),
            "sku": product.get("sku"),
        }
        ranking = bool(self.goal.get("named_keys")) or self.goal.get("kind") == "ranking"
        needs = bool(self.goal.get("need_reason")) and not ranking
        if ranking:
            needs = False
        if needs and self.order is not None:
            allowed = _claim_enum(self.order)
            fields = list(self.goal.get("spec_fields") or [])
            for field in _need_fields(_shopper_text(self.query)):
                if field not in fields:
                    fields.append(field)
            if not fields:
                fields = ["in_stock", "price"]
            brand = _clean(product.get("brand")) or _clean(_facts(product).get("brand"))
            if brand:
                fields = ["brand"] + [field for field in fields if field != "brand"]
            seen: list[str] = []
            for field in fields:
                if field not in seen:
                    seen.append(field)
            claims = _evidence_claims(product, seen, allowed)
            if claims:
                wanted["justification"] = {"claims": claims}
        return {
            "name": _tool_name(self.order),
            "args": _bind(self.order, wanted),
        }

    def repair(
        self,
        actions: list[dict[str, Any]],
        proposed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        names = {_tool_name(tool).lower() for tool in self.tools}
        cleaned = [
            action
            for action in actions
            if str(action.get("name") or "").lower() in names
        ]
        if not cleaned:
            return proposed[: self.max_calls]
        order_name = _tool_name(self.order).lower()
        cart_name = _tool_name(self.cart).lower()
        winner = self.winner()
        repaired: list[dict[str, Any]] = []
        for action in cleaned[: self.max_calls]:
            name = str(action.get("name") or "")
            args = dict(action.get("args") or {})
            lower = name.lower()
            if lower == cart_name and winner is not None:
                picked_key = f"{args.get('product_id')}::{args.get('sku')}"
                win_key = _variant_key(winner)
                ready = bool(winner.get("_inspected")) or any(
                    str(item.get("name") or "").lower() == cart_name
                    for item in proposed
                )
                if (
                    picked_key != win_key
                    and not self.stale(winner)
                    and ready
                ):
                    args = _bind(
                        self.cart,
                        {
                            "product_id": winner.get("product_id"),
                            "sku": winner.get("sku"),
                        },
                    )
            if lower == order_name:
                product = winner
                if args.get("product_id") and args.get("sku"):
                    product = self.index.rows.get(
                        f"{args.get('product_id')}::{args.get('sku')}"
                    ) or product
                if (
                    winner is not None
                    and not self.stale(winner)
                    and winner.get("_inspected")
                ):
                    product = winner
                if product is not None:
                    action = self._order(product)
                    repaired.append(action)
                    continue
            repaired.append({"name": name, "args": args})
        return repaired or proposed[: self.max_calls]

    def ingest(self, result: dict[str, Any], sent: list[dict[str, Any]]) -> bool:
        done = False
        calls = result.get("calls") or []
        for index, call in enumerate(calls):
            observation = call.get("observation") or {}
            if observation.get("done") is True:
                done = True
            inner = _unwrap(observation)
            name = ""
            if index < len(sent):
                name = str(sent[index].get("name") or "").lower()
            viewed = name in {
                _tool_name(self.view).lower(),
                _tool_name(self.compare).lower(),
            }
            inspected = name == _tool_name(self.inspect).lower()
            products = _walk_products(inner)
            if products:
                query = ""
                if index < len(sent):
                    query = str((sent[index].get("args") or {}).get("query") or "")
                if query and name in {
                    _tool_name(self.search).lower(),
                    "search",
                    "find_product",
                    "search_catalog",
                }:
                    stamped = []
                    for product in products:
                        row = dict(product)
                        row["_search_q"] = query
                        stamped.append(row)
                    products = stamped
                self.index.merge(
                    products,
                    viewed=viewed,
                    inspected=inspected,
                    from_search=not viewed and not inspected,
                )
            if isinstance(inner.get("added"), dict):
                self.index.merge(
                    [inner["added"]],
                    viewed=False,
                    inspected=True,
                    from_search=False,
                )
                self.index.cart = [dict(inner["added"])]
            if isinstance(inner.get("cart"), list):
                self.index.cart = [
                    item for item in inner["cart"] if isinstance(item, dict)
                ]
            event = inner.get("market_event")
            if isinstance(event, dict) and event:
                self.index.apply_event(event)
        message = result.get("user_message") or {}
        if isinstance(message, dict):
            signal = message.get("env_signal") or {}
            if isinstance(signal, dict) and signal.get("kind"):
                self.index.apply_event(signal)
            content = str(message.get("content") or "")
            budget = _BUDGET.search(content)
            if budget:
                self.goal["budget"] = float(budget.group(1))
        return done


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
        try:
            args = _arguments(call)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        name = function.get("name")
        if name:
            actions.append({"name": name, "args": args, "_id": call.get("id")})
    return actions, last_content, raw_calls


def _compact_obs(observation: Any) -> str:
    blob = json.dumps(observation, default=str, ensure_ascii=False)
    if len(blob) <= 7000:
        return blob
    return blob[:6980] + "...<truncated>"


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
                or f"{problem_id}-model-{turn}-{index}",
                "type": "function",
                "function": {
                    "name": action["name"],
                    "arguments": json.dumps(action.get("args") or {}),
                },
            }
        )
    return calls


def _force_message(matcher: "_Matcher") -> list[dict[str, Any]]:
    if matcher.message is None:
        return []
    return [
        {
            "name": _tool_name(matcher.message),
            "args": _bind(
                matcher.message,
                {"content": "I will check the live listing and confirm."},
            ),
        }
    ]


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Lane-routed model loop steered by match strategies until done=true."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
    tools = list(policy.get("tools") or [])
    query = str(policy.get("query") or problem_data.get("query") or "")
    lane = classify_lane(query)
    model = _lane_model(lane)
    matcher = _Matcher(query, tools, max_calls)
    shopper = _shopper_text(query).lower()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(tools, max_calls, lane, policy)},
        {"role": "user", "content": query},
    ]
    if lane == LANE_CONSTRAINT and "i will tell you if you ask" in shopper:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Coach: the shopper has one firm hardware spec revealed only "
                    "if asked. Call message NOW to ask for that exact requirement "
                    "before searching."
                ),
            }
        )
    if lane == LANE_RANK:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Coach [ranking]: compare listed candidates first. "
                    "place_test_order with ONLY product_id and sku — omit justification."
                ),
            }
        )
    if lane == LANE_JUSTIFY:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Coach [justification]: view facts before the first cart. "
                    "Claims allowlist: in_stock, price, one spec field, plus brand "
                    "when observed. Forbidden: display, size, category, invented hdr."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = []
    await_reply = False
    cart_used = False
    prior_names: list[str] = []

    for turn in range(1, int(policy.get("max_steps") or 20) + 1):
        proposed = matcher.propose()
        coach = _coach(
            lane, turn, matcher, proposed, await_reply, cart_used, prior_names
        )
        if coach:
            messages.append({"role": "user", "content": coach})

        actions: list[dict[str, Any]] = []
        assistant_content = ""
        raw_calls: list[dict[str, Any]] = []
        for retry in range(len(_IDLE) + 1):
            actions, assistant_content, raw_calls = _model_actions(
                messages, tools, max_calls, model
            )
            if actions:
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
        if await_reply:
            names = {str(item.get("name") or "").lower() for item in actions}
            if "message" not in names:
                forced = _force_message(matcher)
                if forced:
                    actions = forced
        else:
            actions = matcher.repair(actions, proposed)
        if not actions:
            raise RuntimeError("model returned no tool call before completion")

        prior_names = [str(item.get("name") or "") for item in actions]
        lowered_names = [name.lower() for name in prior_names]
        if any("add_to_cart" in name for name in lowered_names):
            cart_used = True
        if any("message" in name for name in lowered_names):
            await_reply = False

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
        result = _proxy().post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        sent = [
            {"name": action["name"], "args": action.get("args") or {}}
            for action in actions
        ]
        for tool_call, call_result in zip(tool_calls, result.get("calls") or []):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _compact_obs(call_result.get("observation")),
                }
            )
        shopper_msg = result.get("user_message") or {}
        if isinstance(shopper_msg, dict) and shopper_msg.get("content"):
            spoken = str(shopper_msg["content"])
            messages.append({"role": "user", "content": spoken})
            if _is_inquiry(spoken):
                await_reply = True
        payloads = [
            call_result.get("observation")
            for call_result in result.get("calls") or []
        ]
        faults = [
            item.get("error")
            for item in payloads
            if isinstance(item, dict) and item.get("error")
        ]
        if faults and lane == LANE_RECOVER:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Coach [recovery]: action failed or stock/price changed "
                        f"({faults[0]}). Recover with least relaxation that still "
                        "satisfies category, budget, and in-stock."
                    ),
                }
            )
        for item in payloads:
            if not isinstance(item, dict):
                continue
            nested = (
                item.get("observation")
                if isinstance(item.get("observation"), dict)
                else item
            )
            if isinstance(nested, dict) and nested.get("market_event"):
                event = nested["market_event"]
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Coach: market_event observed ({event.get('kind')}). "
                            "Re-verify price/stock; if invalid, remove_from_cart and "
                            "recover before ordering."
                        ),
                    }
                )
                break
        if matcher.ingest(result, sent):
            break

    return dialogue


__all__ = ["agent_main", "classify_lane"]
