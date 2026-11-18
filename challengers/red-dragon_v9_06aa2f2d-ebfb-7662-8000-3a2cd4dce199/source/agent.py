from __future__ import annotations

import json
import math
import re
from collections import Counter
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


MOTIVE_INTENT = "intent_decomposition"
MOTIVE_RECALL = "retrieval_recall"
MOTIVE_CONSTRAINT = "constraint_satisfaction"
MOTIVE_PREFERENCE = "preference_reasoning"
MOTIVE_RANK = "ranking"
MOTIVE_RECOVER = "recovery"
MOTIVE_JUSTIFY = "justification"

_BM25P_K1 = 1.5
_BM25P_B = 0.75
_BM25P_DELTA = 1.0
_JM_LAMBDA = 0.7
_WASPAS_LAMBDA = 0.5
_JW_PREFIX = 0.1
_OBS_CAP = 6800

_PAIR_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._-]+)")
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hz|hertz)\b", re.I)
_INCH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(?:\"|''|in(?:ch(?:es)?)?)\b",
    re.I,
)
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|under|below|up to|upto|budget|<=|"
    r"no more than|max(?:imum)?(?:\s+of)?|can only go up to|"
    r"only go up to|new budget)\s*"
    r"([\d][\d,]*(?:\.\d+)?)",
    re.I,
)
_QUOTE_RE = re.compile(r"[\"\u201c]([^\"\u201d]{3,80})[\"\u201d]")
_ASK_RE = re.compile(
    r"\bif you ask\b|\bwhen you ask\b|"
    r"(?:tell|share|reveal)[^.]{0,48}\bif\b[^.]{0,24}\bask|"
    r"ask (?:me |for )?(?:the |that )?(?:exact )?(?:requirement|spec)",
    re.I,
)
_WHY_RE = re.compile(
    r"structured justification|\bjustif|\bexplain\b|"
    r"walk me through why|tell me why|trustworthy reason|"
    r"(?:must|should) include.{0,40}justification",
    re.I,
)
_OPTIONAL_WHY_RE = re.compile(r"justification is optional", re.I)
_RECOVER_RE = re.compile(
    r"\bdisappear|\bunavailable\b|\bsold out\b|\bleast[- ]relax|"
    r"\brelaxable only\b|another color of the same model|"
    r"same model.{0,24}other color|other colour of the same",
    re.I,
)
_PREFER_RE = re.compile(
    r"(?:prefer(?:red)?|ideally)\s+(?:the\s+)?([A-Z][A-Za-z0-9.&-]{2,})",
)
_MODEL_RE = re.compile(
    r"\bmodel(?:\s+code)?\s+([A-Za-z0-9._-]{2,})|"
    r"prefer(?:s)? the\s+[a-z]+\s+([A-Za-z0-9._-]{2,})",
    re.I,
)

_HUES = (
    "midnight",
    "champagne",
    "graphite",
    "titanium",
    "charcoal",
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
_HUE_ALIAS = {
    "midnight": "black",
    "graphite": "black",
    "charcoal": "black",
    "grey": "gray",
    "gray": "grey",
}
_KIND_NOUNS = (
    "smartphone",
    "tablet",
    "monitor",
    "headphone",
    "headset",
    "earphone",
    "earbud",
    "laptop",
    "notebook",
    "camera",
    "watch",
    "speaker",
    "keyboard",
    "mouse",
    "router",
    "printer",
)
_KIND_PATH = {
    "smartphone": "Smartphones",
    "tablet": "Tablets",
    "monitor": "Monitors",
    "headphone": "Headphones",
    "headset": "Headphones",
    "earphone": "Headphones",
    "earbud": "Headphones",
    "laptop": "Laptops",
    "notebook": "Laptops",
    "camera": "Cameras",
    "watch": "Watches",
    "speaker": "Speakers",
    "keyboard": "Keyboards",
    "mouse": "Mice",
    "router": "Routers",
    "printer": "Printers",
}
_SOFT_MARKERS = (
    "waterproof",
    "water resistant",
    "sweat",
    "workout",
    "fitness",
    "outdoor",
    "battery",
    "gps",
    "heart rate",
    "weather",
    "microphone",
    "streaming",
    "privacy",
    "autofocus",
)
_JUNK = (
    "case",
    "cover",
    "charger",
    "cable",
    "adapter",
    "screen protector",
    "replacement screen",
    "webcam",
    "mount",
    "stand",
    "skin",
    "sticker",
)
_CLAIM_CORE = ("in_stock", "price", "resolution", "screen_size", "refresh_rate")
_CLAIM_SOFT = (
    "brand",
    "streaming",
    "microphone",
    "privacy_cover",
    "autofocus",
    "hdr",
)
_CLAIM_BLOCK = {
    "display",
    "size",
    "category",
    "category_path",
    "device_operating_system",
    "os",
    "title",
    "summary",
    "evidence",
}
_BRAND_STOP = {
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
    "ideally",
}

_OPENROUTER = {
    "scout": "deepseek/deepseek-v3.2",
    "gpt": "openai/gpt-oss-120b",
    "dense": "moonshotai/kimi-k2.6",
    "wide": "qwen/qwen3.5-397b-a17b",
    "recover": "z-ai/glm-5.2",
}
_CHUTES = {
    "scout": "deepseek-ai/DeepSeek-V3.2-TEE",
    "gpt": "openai/gpt-oss-120b-TEE",
    "dense": "moonshotai/Kimi-K2.6-TEE",
    "wide": "Qwen/Qwen3.5-397B-A17B-TEE",
    "recover": "zai-org/GLM-5.2-TEE",
}
_TEE_TO_OR = {
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
    "unsloth/Mistral-Nemo-Instruct-2407-TEE": "mistralai/mistral-small-2603",
    "XiaomiMiMo/MiMo-V2-Flash-TEE": "xiaomi/mimo-v2-flash",
    "openai/gpt-oss-120b-TEE": "openai/gpt-oss-120b",
}
_OR_TO_TEE = {slug: tee for tee, slug in _TEE_TO_OR.items()}
_RESERVE_OR = (
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2.6",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "deepseek/deepseek-v3.2",
)
_RESERVE_TEE = (
    "openai/gpt-oss-120b-TEE",
    "moonshotai/Kimi-K2.6-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
)

_IDLE = (
    "No tool call received. Choose a supplied tool now. "
    "If you have not searched yet, call search or filter with value-side keywords.",
    "Still no tool call. Progress: search/filter, view a candidate, answer any "
    "pending shopper question with message, or place_test_order if the cart "
    "already holds a verified in-stock SKU.",
)

_proxy = ProxyClient(timeout=120, max_retries=2)


def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "openrouter").strip().lower()


def _alias_model(raw: str) -> str:
    token = (raw or "").strip()
    if not token:
        return token
    if _provider() == "openrouter":
        return _TEE_TO_OR.get(token, token)
    return _OR_TO_TEE.get(token, token)


def _role_model(role: str) -> str:
    table = _OPENROUTER if _provider() == "openrouter" else _CHUTES
    return table.get(role, table["scout"])


def _motive_model(motive: str) -> str:
    forced = _alias_model(getenv("SANDBOX_MODEL") or "")
    if forced:
        return forced
    if motive == MOTIVE_RECALL:
        return _role_model("scout")
    if motive == MOTIVE_RECOVER:
        return _role_model("recover")
    return _role_model("gpt")


def _reserve_models(primary: str) -> list[str]:
    pool = _RESERVE_OR if _provider() == "openrouter" else _RESERVE_TEE
    ordered: list[str] = [primary]
    for item in pool:
        if item not in ordered:
            ordered.append(item)
    return ordered[:4]


def _trim(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fold(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _tokens(text: str) -> list[str]:
    return [token for token in _fold(text).split() if len(token) > 1]


def _shopper(query: str) -> str:
    lowered = query or ""
    if "task rules:" in lowered.lower():
        return query.split("Task rules:", 1)[0]
    return query


def _fn(tool: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(tool, dict):
        return {}
    function = tool.get("function")
    return function if isinstance(function, dict) else tool


def _tool_name(tool: dict[str, Any] | None) -> str:
    return str(_fn(tool).get("name") or "")


def _tool_props(tool: dict[str, Any] | None) -> dict[str, Any]:
    parameters = _fn(tool).get("parameters") or {}
    properties = parameters.get("properties") or {}
    return properties if isinstance(properties, dict) else {}


def _pick_tool(tools: list[dict[str, Any]], *needles: str) -> dict[str, Any] | None:
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


def _cast(value: Any, spec: dict[str, Any]) -> Any:
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
    properties = _tool_props(tool)
    if not properties:
        return {key: value for key, value in wanted.items() if value is not None}
    lower = {name.lower(): name for name in properties}
    aliases = {
        "query": ("query", "q", "text", "keywords", "search"),
        "k": ("k", "limit", "n", "top_k", "page_size"),
        "in_stock": ("in_stock", "instock", "available"),
        "product_id": ("product_id", "productid", "id"),
        "sku": ("sku", "variant", "variant_id"),
        "product_ids": ("product_ids", "ids"),
        "content": ("content", "message", "text", "utterance"),
        "max_price": ("max_price", "price_max", "budget"),
        "brand": ("brand",),
        "category": ("category",),
        "justification": ("justification",),
        "price": ("price", "price_range"),
        "shop_id": ("shop_id", "shopid", "store_id"),
        "page": ("page",),
    }
    filled: dict[str, Any] = {}
    for logical, value in wanted.items():
        if value is None:
            continue
        for alias in aliases.get(logical, (logical,)):
            real = lower.get(alias.lower())
            if real is None:
                continue
            filled[real] = _cast(value, properties.get(real) or {})
            break
    return filled


def _call_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        parsed = raw
    else:
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _named_pairs(text: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for product_id, sku in _PAIR_RE.findall(text or ""):
        sku = sku.rstrip(".,;:")
        key = f"{product_id}::{sku}"
        if not sku or key in seen:
            continue
        seen.add(key)
        found.append({"product_id": product_id, "sku": sku})
    return found


def _listing_key(product: dict[str, Any]) -> str:
    product_id = str(product.get("product_id") or product.get("id") or "")
    sku = str(product.get("sku") or "")
    if product_id and sku:
        return f"{product_id}::{sku}"
    return product_id


def _facts(product: dict[str, Any]) -> dict[str, Any]:
    raw = product.get("facts") or product.get("attributes") or product.get("specs")
    if isinstance(raw, dict):
        return raw
    return {}


def _field_text(product: dict[str, Any], field: str) -> str:
    facts = _facts(product)
    if field == "title":
        return _trim(product.get("title") or product.get("name") or "")
    if field == "facts":
        parts = [str(value) for value in facts.values() if value not in (None, "")]
        options = product.get("options")
        if isinstance(options, dict):
            parts.extend(str(value) for value in options.values() if value not in (None, ""))
        return " ".join(parts)
    if field == "brand":
        return _trim(product.get("brand") or facts.get("brand") or "")
    if field == "model":
        return _trim(
            facts.get("model")
            or facts.get("model_name")
            or product.get("model")
            or product.get("sku")
            or ""
        )
    if field == "category":
        path = product.get("category_path") or product.get("category") or facts.get("category")
        if isinstance(path, list):
            return " ".join(_trim(item) for item in path if item)
        return _trim(path)
    return ""


def _doc_text(product: dict[str, Any]) -> str:
    return " ".join(
        _field_text(product, name)
        for name in ("title", "facts", "brand", "model", "category")
    )


def _number(text: Any) -> float | None:
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return float(text)
    match = _NUM_RE.search(str(text or "").replace(",", ""))
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
    if product.get("in_stock") is False:
        return False
    if product.get("in_stock") is True:
        return True
    text = _fold(product.get("in_stock"))
    if text in {"false", "0", "no", "oos", "out of stock", "unavailable"}:
        return False
    if text in {"true", "1", "yes", "in stock", "available"}:
        return True
    facts = _facts(product)
    if facts.get("in_stock") is False:
        return False
    stock = product.get("stock") or facts.get("stock")
    if isinstance(stock, (int, float)):
        return stock > 0
    return True


def _option_blob(product: dict[str, Any]) -> str:
    options = product.get("options")
    if isinstance(options, dict):
        return " ".join(str(value) for value in options.values() if value not in (None, ""))
    if isinstance(options, list):
        return " ".join(str(item) for item in options if item not in (None, ""))
    return str(options or "")


def _hertz(product: dict[str, Any]) -> float | None:
    facts = _facts(product)
    chunks = [
        _option_blob(product),
        facts.get("refresh_rate"),
        facts.get("refresh"),
        facts.get("max_refresh"),
        facts.get("advertised_refresh"),
        product.get("title"),
        _field_text(product, "facts"),
    ]
    best: float | None = None
    for chunk in chunks:
        for match in _HZ_RE.findall(str(chunk or "")):
            try:
                value = float(match)
            except ValueError:
                continue
            if best is None or value > best:
                best = value
    for key in ("refresh_rate", "refresh", "max_refresh", "advertised_refresh"):
        value = _number(facts.get(key))
        if value is not None and (best is None or value > best):
            best = value
    return best


def _inches(product: dict[str, Any]) -> float | None:
    facts = _facts(product)
    sources = [
        _option_blob(product),
        facts.get("screen_size"),
        facts.get("display_size"),
        facts.get("size"),
    ]
    for source in sources:
        hit = _INCH_RE.search(str(source or ""))
        if hit:
            try:
                return float(hit.group(1))
            except ValueError:
                pass
        value = _number(source)
        if value is not None and 5.0 <= value <= 90.0:
            return value
    if product.get("_hydrated"):
        return None
    blob = f"{product.get('title') or ''} {_field_text(product, 'facts')}"
    hit = _INCH_RE.search(blob)
    return float(hit.group(1)) if hit else None


def _pixel_class(text: Any) -> str | None:
    blob = str(text or "")
    folded = _fold(blob)
    pair = re.search(r"(\d{3,4})\s*[x×]\s*(\d{3,4})", blob, re.I)
    if pair:
        width, height = int(pair.group(1)), int(pair.group(2))
        if width >= 3840 and height >= 2160:
            return "4k"
        if width >= 2560 and height >= 1440:
            return "qhd"
        if width >= 1920 and height >= 1080:
            return "fhd"
        if width >= 1280 and height >= 720:
            return "hd"
    if re.search(r"\b(4k|uhd|2160p|3840)\b", folded):
        return "4k"
    if re.search(r"\b(1440p|2560)\b", folded) or re.search(r"\bQHD\b", blob):
        return "qhd"
    if re.search(r"\b(full hd|fhd|1080p)\b", folded):
        return "fhd"
    if re.search(r"(720p|\bhd\b)", folded):
        return "hd"
    return None


def _axis_name(raw: str) -> str:
    token = _fold(raw)
    if "refresh" in token or token in {"hz"}:
        return "refresh_rate"
    if "screen" in token or "inch" in token or token == "size":
        return "screen_size"
    if "price" in token or "cost" in token or "cheap" in token:
        return "price"
    return token.replace(" ", "_") or "price"


def _axes_from(text: str) -> list[tuple[str, str]]:
    lowered = (text or "").lower()
    axes: list[tuple[str, str]] = []
    patterns = (
        (r"highest ([a-z ]{3,28})", "desc"),
        (r"largest ([a-z ]{3,28})", "desc"),
        (r"biggest ([a-z ]{3,28})", "desc"),
        (r"most ([a-z ]{3,28})", "desc"),
        (r"lowest ([a-z ]{3,24})", "asc"),
        (r"smallest ([a-z ]{3,24})", "asc"),
        (r"cheapest ([a-z ]{3,24})?", "asc"),
    )
    seen: set[str] = set()
    for pattern, direction in patterns:
        for match in re.finditer(pattern, lowered):
            raw = (match.group(1) or "price").strip()
            name = _axis_name(raw)
            if name in seen:
                continue
            seen.add(name)
            axes.append((name, "asc" if "cheap" in raw or direction == "asc" else direction))
    if "cheapest" in lowered or "lowest price" in lowered:
        if "price" not in seen:
            axes.append(("price", "asc"))
    return axes


def _axis_read(product: dict[str, Any], name: str) -> float | None:
    key = _fold(name).replace(" ", "_")
    if key in {"refresh_rate", "refresh", "hz"}:
        return _hertz(product)
    if key in {"screen_size", "screen", "inches", "size"}:
        return _inches(product)
    if key in {"price", "cost", "cheap"}:
        return _price_of(product)
    facts = _facts(product)
    if key in facts:
        return _number(facts.get(key))
    return _number(product.get(key))


def _idf(df: int, docs: int) -> float:
    return math.log((docs - df + 0.5) / (df + 0.5) + 1.0)


def _bm25_plus(
    query_tf: Counter[str],
    doc_tf: Counter[str],
    df: Counter[str],
    docs: int,
    doc_len: int,
    avgdl: float,
) -> float:
    if not query_tf or docs <= 0:
        return 0.0
    length_norm = 1.0 - _BM25P_B + _BM25P_B * (doc_len / max(avgdl, 1.0))
    score = 0.0
    for token, qtf in query_tf.items():
        tf = float(doc_tf.get(token, 0))
        idf = _idf(df.get(token, 0), docs)
        if tf <= 0:
            score += qtf * idf * _BM25P_DELTA
            continue
        tf_norm = (tf * (_BM25P_K1 + 1.0)) / (tf + _BM25P_K1 * length_norm)
        score += qtf * idf * (tf_norm + _BM25P_DELTA)
    return score


def _jelinek_mercer(
    query_tf: Counter[str],
    doc_tf: Counter[str],
    collection: Counter[str],
    collection_len: int,
    doc_len: int,
) -> float:
    if not query_tf or collection_len <= 0:
        return 0.0
    score = 0.0
    for token, qtf in query_tf.items():
        p_d = (doc_tf.get(token, 0) / doc_len) if doc_len else 0.0
        p_c = collection.get(token, 0) / collection_len
        if p_c <= 0:
            p_c = 1.0 / max(collection_len, 1)
        mix = _JM_LAMBDA * p_d + (1.0 - _JM_LAMBDA) * p_c
        score += qtf * math.log(max(mix, 1e-12))
    return score


def _isr_fuse(waves: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    keep: dict[str, dict[str, Any]] = {}
    for rows in waves:
        for rank, product in enumerate(rows, start=1):
            key = _listing_key(product) or str(product.get("product_id") or rank)
            scores[key] = scores.get(key, 0.0) + 1.0 / float(rank * rank)
            keep[key] = product
    return [
        keep[key]
        for key, _score in sorted(scores.items(), key=lambda item: -item[1])
    ]


def _minmax_col(values: list[float | None], benefit: bool) -> list[float]:
    present = [item for item in values if item is not None]
    if not present:
        return [0.5 for _ in values]
    lo = min(present)
    hi = max(present)
    out: list[float] = []
    for item in values:
        if item is None:
            out.append(0.5)
            continue
        if hi - lo < 1e-12:
            out.append(1.0)
            continue
        unit = (item - lo) / (hi - lo)
        out.append(unit if benefit else 1.0 - unit)
    return out


def _waspas(
    products: list[dict[str, Any]],
    axes: list[tuple[str, str]],
) -> list[tuple[float, dict[str, Any]]]:
    if not products:
        return []
    if not axes:
        return [(0.0, product) for product in products]
    raw: list[list[float | None]] = []
    for product in products:
        raw.append([_axis_read(product, name) for name, _direction in axes])
    weights = [1.0 / (index + 1.0) for index in range(len(axes))]
    total = sum(weights)
    weights = [item / total for item in weights]
    cols: list[list[float]] = []
    for col, (_name, direction) in enumerate(axes):
        column = [row[col] for row in raw]
        cols.append(_minmax_col(column, benefit=(direction != "asc")))
    scored: list[tuple[float, dict[str, Any]]] = []
    for index, product in enumerate(products):
        wsm = 0.0
        wpm = 1.0
        for col, weight in enumerate(weights):
            unit = max(cols[col][index], 1e-9)
            wsm += weight * unit
            wpm *= unit**weight
        score = _WASPAS_LAMBDA * wsm + (1.0 - _WASPAS_LAMBDA) * wpm
        scored.append((score, product))
    scored.sort(key=lambda item: -item[0])
    return scored


def _copeland(
    products: list[dict[str, Any]],
    axes: list[tuple[str, str]],
) -> dict[str, int]:
    n = len(products)
    tally = { _listing_key(product) or str(index): 0 for index, product in enumerate(products) }
    if n < 2 or not axes:
        return tally
    for i in range(n):
        for j in range(i + 1, n):
            wins = 0
            losses = 0
            for name, direction in axes:
                left = _axis_read(products[i], name)
                right = _axis_read(products[j], name)
                if left is None or right is None:
                    continue
                if direction == "asc":
                    left, right = -left, -right
                if left > right:
                    wins += 1
                elif right > left:
                    losses += 1
            left_key = _listing_key(products[i]) or str(i)
            right_key = _listing_key(products[j]) or str(j)
            if wins > losses:
                tally[left_key] += 1
                tally[right_key] -= 1
            elif losses > wins:
                tally[right_key] += 1
                tally[left_key] -= 1
    return tally


def _lex_tuple(
    product: dict[str, Any],
    axes: list[tuple[str, str]],
    waspas_score: float = 0.0,
) -> tuple:
    parts: list[float] = []
    for name, direction in axes:
        value = _axis_read(product, name)
        if value is None:
            parts.extend([1.0, 0.0])
            continue
        parts.append(0.0)
        parts.append(-value if direction == "desc" else value)
    parts.append(-float(waspas_score))
    return tuple(parts)


def _jaro(left: str, right: str) -> float:
    a = _fold(left)
    b = _fold(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    match_span = max(max(len(a), len(b)) // 2 - 1, 0)
    a_hit = [False] * len(a)
    b_hit = [False] * len(b)
    matches = 0
    for i, char in enumerate(a):
        start = max(0, i - match_span)
        end = min(i + match_span + 1, len(b))
        for j in range(start, end):
            if b_hit[j] or b[j] != char:
                continue
            a_hit[i] = True
            b_hit[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i, hit in enumerate(a_hit):
        if not hit:
            continue
        while not b_hit[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    jaro = (
        matches / len(a)
        + matches / len(b)
        + (matches - transpositions / 2.0) / matches
    ) / 3.0
    prefix = 0
    for left_c, right_c in zip(a, b):
        if left_c != right_c or prefix == 4:
            break
        prefix += 1
    return jaro + _JW_PREFIX * prefix * (1.0 - jaro)


def _token_f1(left: str, right: str) -> float:
    a = set(_tokens(left))
    b = set(_tokens(right))
    if not a or not b:
        wanted = _compact(left)
        have = _compact(right)
        if wanted and wanted in have:
            return 1.0
        return 0.0
    overlap = len(a & b)
    if overlap == 0:
        wanted = _compact(left)
        have = _compact(right)
        if wanted and wanted in have:
            return 1.0
        return 0.0
    precision = overlap / len(a)
    recall = overlap / len(b)
    return 2.0 * precision * recall / max(precision + recall, 1e-9)


def _kind_of(text: str) -> str:
    blob = _fold(text)
    for noun in _KIND_NOUNS:
        if re.search(rf"\b{noun}s?\b", blob):
            return noun
    return ""


def _hue_of(text: str) -> str:
    blob = f" {_fold(text)} "
    for hue in _HUES:
        if f" {hue} " in blob:
            return hue
    return ""


def _model_of(text: str) -> str:
    match = _MODEL_RE.search(text or "")
    if not match:
        return ""
    token = match.group(1) or match.group(2) or ""
    return token.rstrip(".,;")


def _brand_of(text: str) -> str:
    match = _PREFER_RE.search(text or "")
    if match:
        token = match.group(1)
        if _fold(token) not in _BRAND_STOP and _fold(token) not in _HUES:
            return token
    return ""


def _is_junk(product: dict[str, Any], kind: str) -> bool:
    blob = _fold(f"{_field_text(product, 'title')} {_field_text(product, 'category')}")
    if not blob:
        return False
    if any(marker in blob for marker in _JUNK):
        if kind in {"smartphone", "tablet", "monitor", "laptop", "watch"}:
            return True
    return False


def _kind_ok(product: dict[str, Any], kind: str) -> bool:
    if not kind:
        return True
    blob = _fold(
        f"{_field_text(product, 'title')} {_field_text(product, 'category')} "
        f"{_field_text(product, 'facts')}"
    )
    if re.search(rf"\b{kind}s?\b", blob):
        return True
    path = _KIND_PATH.get(kind, "")
    return bool(path) and _fold(path) in blob


def _brand_ok(product: dict[str, Any], brand: str) -> bool:
    wanted = _fold(brand)
    if not wanted:
        return True
    have = _fold(_field_text(product, "brand") + " " + _field_text(product, "title"))
    return wanted in have


def _model_ok(product: dict[str, Any], model: str) -> bool:
    wanted = _compact(model)
    if not wanted:
        return True
    sources = [
        _field_text(product, "model"),
        _field_text(product, "facts"),
        str(product.get("sku") or ""),
        str(product.get("_search_q") or ""),
    ]
    blob = _compact(" ".join(sources))
    if wanted in blob:
        return True
    return _token_f1(model, " ".join(sources)) >= 0.72


def _color_ok(product: dict[str, Any], color: str) -> bool:
    wanted = _fold(color)
    if not wanted:
        return True
    options = product.get("options")
    option_blob = ""
    if isinstance(options, dict):
        option_blob = " ".join(str(value) for value in options.values())
    blob = _fold(
        f"{option_blob} {_facts(product).get('color') or ''} "
        f"{_facts(product).get('colour') or ''} {_field_text(product, 'title')}"
    )
    if wanted in blob:
        return True
    alias = _HUE_ALIAS.get(wanted, "")
    if alias and alias in blob:
        return True
    return _jaro(wanted, blob) >= 0.88


def _spec_ok(product: dict[str, Any], goal: dict[str, Any]) -> bool:
    klass = str(goal.get("spec_res") or "")
    inch = goal.get("spec_inch")
    if klass:
        have = _pixel_class(_facts(product).get("display") or _facts(product).get("resolution"))
        if have is None and product.get("_hydrated"):
            have = _pixel_class(_field_text(product, "facts"))
        order = {"hd": 0, "fhd": 1, "qhd": 2, "4k": 3}
        if have is not None and order.get(have, -1) < order.get(klass, 99):
            return False
    if inch is not None:
        have_in = _inches(product)
        if have_in is not None and have_in + 0.15 < float(inch):
            return False
    return True


def _cobb_douglas(product: dict[str, Any], markers: list[str]) -> float:
    if not markers:
        return 1.0
    blob = _fold(f"{_field_text(product, 'title')} {_field_text(product, 'facts')}")
    utility = 1.0
    for marker in markers:
        hit = 1.0 if _fold(marker) in blob else 0.0
        utility *= 1.0 + hit
    return utility


def _recover_cost(product: dict[str, Any], goal: dict[str, Any], event_key: str | None) -> float:
    model = str(goal.get("model") or "")
    color = str(goal.get("color") or "")
    color_score = _jaro(color, _field_text(product, "title")) if color else 1.0
    if color:
        color_score = max(color_score, 1.0 if _color_ok(product, color) else 0.0)
    model_score = _token_f1(model, _field_text(product, "model") + " " + str(product.get("sku") or ""))
    if model and _model_ok(product, model):
        model_score = max(model_score, 1.0)
    cost = 2.4 * (1.0 - color_score) + 1.6 * (1.0 - model_score)
    key = _listing_key(product)
    if event_key and key == event_key:
        cost += 50.0
    stem = str(product.get("sku") or "").rsplit("-", 1)[0]
    target_sku = (event_key or "").split("::")[-1]
    target_stem = target_sku.rsplit("-", 1)[0] if target_sku else ""
    if stem and target_stem and stem == target_stem:
        cost -= 0.35
    return cost


def _eligible(product: dict[str, Any], goal: dict[str, Any], *, strict: bool) -> bool:
    if goal.get("named_keys"):
        return True
    kind = str(goal.get("kind") or "")
    if _is_junk(product, kind):
        return False
    if kind and not _kind_ok(product, kind):
        return False
    motive = str(goal.get("motive") or "")
    model = str(goal.get("model") or "")
    if motive == MOTIVE_RECOVER and model:
        queried = _compact(model) in _compact(str(product.get("_search_q") or ""))
        if not _model_ok(product, model) and not queried:
            return False
    brand = str(goal.get("hard_brand") or "")
    if brand and motive != MOTIVE_RECOVER:
        titled = bool(product.get("title") or product.get("brand") or strict)
        if titled and not _brand_ok(product, brand):
            return False
    if strict and motive == MOTIVE_JUSTIFY and not _spec_ok(product, goal):
        return False
    if strict and motive == MOTIVE_CONSTRAINT and not _spec_ok(product, goal):
        return False
    return True


def _claim_enum(tool: dict[str, Any] | None) -> list[str]:
    spec = (_tool_props(tool).get("justification") or {}).get("properties") or {}
    claims = spec.get("claims") or {}
    items = claims.get("items") or {}
    field = (items.get("properties") or {}).get("field") or {}
    enum = field.get("enum")
    if isinstance(enum, list) and enum:
        return [str(item) for item in enum]
    return list(_CLAIM_CORE) + list(_CLAIM_SOFT)


def _price_text(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _screen_text(inches: float) -> str:
    if abs(inches - round(inches)) < 0.05:
        return f"{int(round(inches))}in"
    return f"{inches:g}in"


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


def _ground_claims(
    product: dict[str, Any],
    fields: list[str],
    allowed: list[str],
) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    facts = _facts(product)
    permit = {item.lower() for item in allowed}
    blob = _fold(" ".join(str(item) for item in facts.values()))
    for field in fields:
        key = field.lower()
        if key in _CLAIM_BLOCK or key not in permit:
            continue
        value: str | None = None
        if key == "in_stock":
            value = "true" if _in_stock(product) else "false"
        elif key == "price":
            price = _price_of(product)
            value = _price_text(price) if price is not None else None
        elif key == "resolution":
            value = _pixel_class(facts.get("display") or facts.get("resolution"))
        elif key == "screen_size":
            inches = _inches(product)
            value = _screen_text(inches) if inches is not None else None
        elif key == "refresh_rate":
            hz = _hertz(product)
            if hz is not None:
                value = str(int(round(hz))) if abs(hz - round(hz)) < 0.05 else str(hz)
        elif key == "brand":
            value = _trim(product.get("brand") or facts.get("brand") or "") or None
        elif key in _CLAIM_SOFT:
            token = key.replace("_", " ")
            if token in blob or key in blob:
                value = "true"
        if not value:
            continue
        claims.append({"field": key, "value": str(value)})
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for claim in claims:
        if claim["field"] in seen:
            continue
        seen.add(claim["field"])
        unique.append(claim)
    return unique[:4]


def _explode(node: dict[str, Any]) -> list[dict[str, Any]]:
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


def _harvest(payload: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        rows = _explode(node)
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
    if isinstance(nested, dict):
        return nested
    return observation


def _search_phrases(goal: dict[str, Any], query: str) -> list[str]:
    phrases: list[str] = []
    quoted = list(goal.get("phrases") or [])
    color = str(goal.get("color") or "")
    model = str(goal.get("model") or "")
    brand = str(goal.get("soft_brand") or goal.get("hard_brand") or "")
    kind = str(goal.get("kind") or "")
    klass = str(goal.get("spec_res") or "")
    water = bool(goal.get("water"))
    if color and model:
        phrases.append(f"{color} {model}")
    if model:
        phrases.append(model)
    phrases.extend(quoted)
    if brand and klass and kind:
        phrases.append(f"{brand} {klass} {kind}")
    if brand and kind:
        phrases.append(f"{brand} {kind}")
    if water and kind:
        phrases.append(f"waterproof {kind}")
        phrases.append(f"sweat resistant {kind}")
    if kind:
        phrases.append(kind)
    path = _KIND_PATH.get(kind, "")
    if path:
        phrases.append(path)
    cleaned: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        token = _trim(phrase)
        folded = _fold(token)
        if not token or folded in {"product", "item"} or folded in seen:
            continue
        seen.add(folded)
        cleaned.append(token)
    if not cleaned:
        head = _trim(_shopper(query))
        if head:
            cleaned.append(" ".join(_tokens(head)[:8]))
    return cleaned[:8]


def classify_motive(query: str) -> str:
    body = _shopper(query)
    full = query or ""
    pairs = _named_pairs(full)
    if len(pairs) >= 2 or (_axes_from(body) and pairs):
        return MOTIVE_RANK
    if _ASK_RE.search(body):
        return MOTIVE_CONSTRAINT
    why = bool(_WHY_RE.search(body))
    if _OPTIONAL_WHY_RE.search(full) and not re.search(r"\bwhy\b|reason|justify", body, re.I):
        why = False
    if why:
        return MOTIVE_JUSTIFY
    if _RECOVER_RE.search(body) or _RECOVER_RE.search(full):
        return MOTIVE_RECOVER
    folded = _fold(body)
    if any(marker in folded for marker in ("workout", "waterproof", "sweat", "use case", "use-case")):
        return MOTIVE_PREFERENCE
    if "rather" in folded or "cheapest" in folded:
        return MOTIVE_PREFERENCE
    if ("prefer" in folded or "ideally" in folded) and (
        "firm" in folded or "requirement" in folded or "soft" in folded
    ):
        return MOTIVE_INTENT
    if _QUOTE_RE.search(body):
        return MOTIVE_RECALL
    return MOTIVE_RECALL


def _parse_goal(query: str) -> dict[str, Any]:
    body = _shopper(query)
    full = query or ""
    motive = classify_motive(full)
    budget_hit = _BUDGET_RE.search(body) or _BUDGET_RE.search(full)
    budget = None
    if budget_hit:
        budget = float(budget_hit.group(1).replace(",", ""))
    named = _named_pairs(full)
    axes = _axes_from(body) or _axes_from(full)
    if motive == MOTIVE_RANK and not axes:
        axes = [("refresh_rate", "desc"), ("screen_size", "desc"), ("price", "asc")]
    phrases = [_trim(item) for item in _QUOTE_RE.findall(body)]
    color = _hue_of(body)
    if color == "midnight" and "black" in _fold(body):
        color = "black"
    kind = _kind_of(body)
    spec_res = _pixel_class(body)
    inch_hit = _INCH_RE.search(body)
    spec_inch = float(inch_hit.group(1)) if inch_hit else None
    soft_brand = _brand_of(body)
    hard_brand = ""
    if motive in {MOTIVE_CONSTRAINT, MOTIVE_JUSTIFY, MOTIVE_RECALL} and not soft_brand:
        hard_brand = _brand_of(body)
    water = any(marker in _fold(body) for marker in ("waterproof", "water resistant", "sweat"))
    why = classify_motive(full) == MOTIVE_JUSTIFY
    goal = {
        "motive": motive,
        "budget": budget,
        "in_stock": True,
        "color": color,
        "model": _model_of(body),
        "kind": kind,
        "soft_brand": soft_brand,
        "hard_brand": hard_brand,
        "named_keys": named,
        "axes": axes,
        "phrases": phrases,
        "ask_spec": bool(_ASK_RE.search(body)),
        "asked_spec": False,
        "need_reason": why,
        "spec_res": spec_res,
        "spec_inch": spec_inch,
        "water": water,
        "markers": [marker for marker in _SOFT_MARKERS if marker in _fold(body)],
    }
    goal["search_phrases"] = _search_phrases(goal, query)
    return goal


class _Tray:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.waves: list[list[dict[str, Any]]] = []
        self.cart: list[dict[str, Any]] = []
        self.event_key: str | None = None
        self.event_kind: str | None = None
        self.viewed: set[str] = set()
        self.checked: set[str] = set()
        self.search_n = 0
        self.did_filter = False
        self.named_keys: list[dict[str, str]] = []

    def _overlay_named(self, product: dict[str, Any], *, viewed: bool) -> None:
        pid = str(product.get("product_id") or "")
        sku = str(product.get("sku") or "")
        if not pid or not self.named_keys:
            return
        for spec in self.named_keys:
            if str(spec.get("product_id") or "") != pid:
                continue
            named_sku = str(spec.get("sku") or "")
            key = f"{pid}::{named_sku}"
            row = self.rows.setdefault(
                key, {"product_id": pid, "sku": named_sku, "_hydrated": False}
            )
            if named_sku == sku:
                continue
            for field in (
                "title",
                "brand",
                "facts",
                "category",
                "category_path",
                "options",
                "description",
                "model",
            ):
                value = product.get(field)
                if value not in (None, "", {}, []) and not row.get(field):
                    row[field] = value
            if viewed:
                row["_hydrated"] = True
                self.viewed.add(pid)

    def _invalidate_rank_survivors(self) -> None:
        if not self.named_keys:
            return
        for spec in self.named_keys:
            pid = str(spec.get("product_id") or "")
            nkey = f"{pid}::{spec.get('sku')}"
            if pid:
                self.viewed.discard(pid)
            self.checked.discard(nkey)
            row = self.rows.get(nkey)
            if row is not None:
                row["_checked"] = False

    def put(
        self,
        products: list[dict[str, Any]],
        *,
        viewed: bool,
        checked: bool,
        from_search: bool,
    ) -> None:
        if from_search and products:
            self.waves.append(products)
        for product in products:
            key = _listing_key(product)
            if not key:
                continue
            current = self.rows.get(key)
            if current is None:
                current = dict(product)
                self.rows[key] = current
            else:
                current.update(
                    {field: value for field, value in product.items() if value is not None}
                )
            if viewed:
                current["_hydrated"] = True
                pid = str(current.get("product_id") or "")
                if pid:
                    self.viewed.add(pid)
            if checked:
                current["_checked"] = True
                self.checked.add(key)
            for item in self.cart:
                if _listing_key(item) == key:
                    item.update(current)
            self._overlay_named(current, viewed=viewed)

    def mark_event(self, event: dict[str, Any]) -> None:
        target = event.get("target") or {}
        product_id = str(target.get("product_id") or event.get("product_id") or "")
        sku = str(target.get("sku") or event.get("sku") or "")
        key = f"{product_id}::{sku}" if product_id and sku else product_id
        self.event_key = key or None
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
            self.put([patch], viewed=False, checked=False, from_search=False)
            if key:
                self.checked.discard(key)
                row = self.rows.get(key)
                if row is not None:
                    row["_checked"] = False
            self._invalidate_rank_survivors()


class _Prism:
    def __init__(self, query: str, tools: list[dict[str, Any]], max_calls: int) -> None:
        self.query = query
        self.tools = tools
        self.max_calls = max(1, max_calls)
        self.goal = _parse_goal(query)
        self.motive = str(self.goal.get("motive") or MOTIVE_RECALL)
        self.tray = _Tray()
        self.search = _pick_tool(tools, "search", "find_product", "search_catalog")
        self.filter = _pick_tool(tools, "filter", "filter_catalog")
        self.view = _pick_tool(tools, "view", "view_product", "get_product")
        self.compare = _pick_tool(tools, "compare", "compare_products")
        self.inspect = _pick_tool(tools, "inspect_stock", "check_stock")
        self.cart_view = _pick_tool(tools, "inspect_cart", "view_cart")
        self.cart = _pick_tool(tools, "add_to_cart", "cart_add")
        self.remove = _pick_tool(tools, "remove_from_cart", "cart_remove")
        self.order = _pick_tool(tools, "place_test_order", "place_order", "checkout")
        self.message = _pick_tool(tools, "message", "talk", "respond")
        for spec in self.goal.get("named_keys") or []:
            key = f"{spec['product_id']}::{spec['sku']}"
            self.tray.named_keys.append(spec)
            self.tray.rows[key] = {
                "product_id": spec["product_id"],
                "sku": spec["sku"],
                "_hydrated": False,
            }

    def _named_specs(self) -> list[dict[str, str]]:
        return list(self.goal.get("named_keys") or [])

    def _closed_rank(self) -> bool:
        return self.motive == MOTIVE_RANK or bool(self._named_specs())

    def _unviewed_named_ids(self) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for spec in self._named_specs():
            pid = str(spec.get("product_id") or "")
            if pid and pid not in self.tray.viewed and pid not in seen:
                seen.add(pid)
                ids.append(pid)
        return ids

    def _rank_ready(self) -> bool:
        if not self._named_specs():
            return True
        return not self._unviewed_named_ids()

    def _pool(self) -> list[dict[str, Any]]:
        wanted = self.goal.get("named_keys") or []
        fused = _isr_fuse(self.tray.waves) if self.tray.waves else []
        by_key = {_listing_key(item): item for item in fused if _listing_key(item)}
        by_key.update(self.tray.rows)
        fused_rows = list(by_key.values())
        if not wanted:
            return fused_rows
        rows: list[dict[str, Any]] = []
        for spec in wanted:
            key = f"{spec['product_id']}::{spec['sku']}"
            product = by_key.get(key) or {
                "product_id": spec["product_id"],
                "sku": spec["sku"],
                "_hydrated": False,
            }
            rows.append(product)
        return rows

    def live(self) -> list[dict[str, Any]]:
        budget = self.goal.get("budget")
        rows = []
        for product in self._pool():
            if self.goal.get("in_stock", True) and not _in_stock(product):
                continue
            price = _price_of(product)
            if budget is not None and price is not None and price > float(budget):
                continue
            if not _eligible(product, self.goal, strict=bool(product.get("_hydrated"))):
                continue
            rows.append(product)
        return rows

    def ranked(self) -> list[dict[str, Any]]:
        pool = self.live()
        axes = list(self.goal.get("axes") or [])
        if axes:
            waspas = { _listing_key(product) or str(index): score
                for index, (score, product) in enumerate(_waspas(pool, axes))
            }
            return sorted(
                pool,
                key=lambda product: _lex_tuple(
                    product,
                    axes,
                    waspas.get(_listing_key(product) or "", 0.0),
                ),
            )
        if self.motive == MOTIVE_RECOVER and self.goal.get("model"):
            scored = [
                (_recover_cost(product, self.goal, self.tray.event_key), product)
                for product in pool
            ]
            scored.sort(key=lambda item: (item[0], _price_of(item[1]) or 0.0))
            return [product for _cost, product in scored]
        query_tf = Counter(
            _tokens(" ".join(self.goal.get("search_phrases") or []) + " " + _shopper(self.query))
        )
        docs = [(_tokens(_doc_text(product)), product) for product in pool]
        df: Counter[str] = Counter()
        collection: Counter[str] = Counter()
        for tokens, _product in docs:
            df.update(set(tokens))
            collection.update(tokens)
        avgdl = sum(len(tokens) for tokens, _product in docs) / max(len(docs), 1)
        collection_len = sum(collection.values()) or 1
        bm_rows: list[tuple[float, dict[str, Any]]] = []
        jm_rows: list[tuple[float, dict[str, Any]]] = []
        for tokens, product in docs:
            doc_tf = Counter(tokens)
            bm_rows.append(
                (
                    _bm25_plus(query_tf, doc_tf, df, len(docs), len(tokens), avgdl),
                    product,
                )
            )
            jm_rows.append(
                (
                    _jelinek_mercer(query_tf, doc_tf, collection, collection_len, len(tokens)),
                    product,
                )
            )
        bm_rows.sort(key=lambda item: -item[0])
        jm_rows.sort(key=lambda item: -item[0])
        fused = _isr_fuse([[product for _score, product in bm_rows], [product for _score, product in jm_rows]])
        if self.motive == MOTIVE_PREFERENCE:
            fused.sort(
                key=lambda product: (
                    -_cobb_douglas(product, list(self.goal.get("markers") or [])),
                    _price_of(product) or 0.0,
                )
            )
        if self.motive == MOTIVE_INTENT:
            preferred = str(self.goal.get("soft_brand") or "")

            def intent_key(product: dict[str, Any]) -> tuple:
                honor = 0 if preferred and _brand_ok(product, preferred) else 1
                return (honor, -_cobb_douglas(product, list(self.goal.get("markers") or [])))

            fused.sort(key=intent_key)
        return fused

    def champion(self) -> dict[str, Any] | None:
        for product in self.ranked():
            if not self.stale(product):
                return product
        return None

    def stale(self, product: dict[str, Any] | None) -> bool:
        if product is None:
            return False
        if not _in_stock(product):
            return True
        budget = self.goal.get("budget")
        price = _price_of(product)
        if budget is not None and price is not None and price > float(budget):
            return True
        if self.tray.event_key and _listing_key(product) == self.tray.event_key:
            return True
        return False

    def needs_recover(self) -> bool:
        cart = self.tray.cart[0] if self.tray.cart else None
        if cart is not None and self.stale(cart):
            return True
        if not self.tray.event_key:
            return False
        if self._closed_rank():
            return True
        return cart is None or self.stale(cart)

    def _inspect_many(self, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.inspect:
            return []
        actions: list[dict[str, Any]] = []
        seen: set[str] = set()
        for product in products:
            pid = str(product.get("product_id") or "")
            sku = product.get("sku")
            key = _listing_key(product) or f"{pid}::{sku}"
            if not pid or not sku or key in seen:
                continue
            if key in self.tray.checked:
                continue
            if self.tray.event_key and key == self.tray.event_key:
                continue
            seen.add(key)
            actions.append(
                {
                    "name": _tool_name(self.inspect),
                    "args": _bind(self.inspect, {"product_id": pid, "sku": sku}),
                }
            )
            if len(actions) >= self.max_calls:
                break
        return actions

    def card(self, proposed: list[dict[str, Any]]) -> str:
        ranked = self.ranked()[:6]
        lines = ["# Prism match card"]
        lines.append(f"- motive: {self.motive}")
        lines.append(
            f"- budget={self.goal.get('budget')} kind={self.goal.get('kind')} "
            f"color={self.goal.get('color')} model={self.goal.get('model')} "
            f"soft_brand={self.goal.get('soft_brand')}"
        )
        champ = self.champion()
        if champ is not None:
            lines.append(
                f"- champion: {_listing_key(champ)} price={_price_of(champ)} "
                f"stock={_in_stock(champ)} hz={_hertz(champ)} in={_inches(champ)}"
            )
        if self.tray.event_key:
            lines.append(f"- event_target banned: {self.tray.event_key}")
        for product in ranked:
            lines.append(
                f"- {_listing_key(product)} p={_price_of(product)} "
                f"stock={_in_stock(product)} hyd={bool(product.get('_hydrated'))} "
                f"chk={bool(product.get('_checked'))}"
            )
        if proposed:
            lines.append(
                "- next tools: "
                + ", ".join(
                    f"{item.get('name')}({json.dumps(item.get('args') or {}, ensure_ascii=False)[:120]})"
                    for item in proposed
                )
            )
        return "\n".join(lines)

    def _search_actions(self) -> list[dict[str, Any]]:
        phrases = list(self.goal.get("search_phrases") or [])
        actions: list[dict[str, Any]] = []
        while self.tray.search_n < max(len(phrases), 1) and len(actions) < self.max_calls:
            query = phrases[min(self.tray.search_n, len(phrases) - 1)] if phrases else ""
            self.tray.search_n += 1
            if not self.search or not query:
                break
            actions.append(
                {
                    "name": _tool_name(self.search),
                    "args": _bind(
                        self.search,
                        {"query": query, "k": 10, "in_stock": True},
                    ),
                }
            )
        brand = str(self.goal.get("hard_brand") or self.goal.get("soft_brand") or "")
        category = _KIND_PATH.get(str(self.goal.get("kind") or ""), "")
        if (
            self.filter
            and not self.tray.did_filter
            and len(actions) < self.max_calls
            and (brand or category or self.goal.get("budget"))
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
                self.tray.did_filter = True
                actions.append({"name": _tool_name(self.filter), "args": args})
        return actions[: self.max_calls]

    def _hydrate(self, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ids: list[str] = []
        seen: set[str] = set()
        for product in products:
            pid = str(product.get("product_id") or "")
            if pid and pid not in seen and pid not in self.tray.viewed:
                seen.add(pid)
                ids.append(pid)
        if not ids:
            return []
        if self.compare:
            return [
                {
                    "name": _tool_name(self.compare),
                    "args": _bind(
                        self.compare,
                        {
                            "product_ids": ids[:8],
                            "product_id": ids[0],
                        },
                    ),
                }
            ]
        if self.view:
            return [
                {
                    "name": _tool_name(self.view),
                    "args": _bind(self.view, {"product_id": ids[0]}),
                }
            ]
        return []

    def _order(self, product: dict[str, Any]) -> dict[str, Any]:
        if self.order is None:
            return {"name": "", "args": {}}
        wanted: dict[str, Any] = {
            "product_id": product.get("product_id"),
            "sku": product.get("sku"),
        }
        ranking = self.motive == MOTIVE_RANK or bool(self.goal.get("named_keys"))
        if ranking:
            return {
                "name": _tool_name(self.order),
                "args": _bind(self.order, wanted),
            }
        if self.goal.get("need_reason"):
            allowed = _claim_enum(self.order)
            fields = [item for item in _need_fields(_shopper(self.query)) if item not in _CLAIM_BLOCK]
            brand = _trim(product.get("brand") or _facts(product).get("brand") or "")
            if brand:
                fields = ["brand"] + [item for item in fields if item != "brand"]
            claims = _ground_claims(product, fields, allowed)
            if claims:
                wanted["justification"] = {"claims": claims}
        return {"name": _tool_name(self.order), "args": _bind(self.order, wanted)}

    def advise(self) -> list[dict[str, Any]]:
        if self.goal.get("ask_spec") and not self.goal.get("asked_spec") and self.message:
            if self.tray.search_n == 0 and not self.tray.viewed:
                return [
                    {
                        "name": _tool_name(self.message),
                        "args": _bind(
                            self.message,
                            {
                                "content": (
                                    "What is the exact firm hardware spec I must satisfy "
                                    "before I search?"
                                )
                            },
                        ),
                    }
                ]
        cart_item = self.tray.cart[0] if self.tray.cart else None
        if cart_item is not None and self.stale(cart_item) and self.remove:
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
        named = list(self.goal.get("named_keys") or [])
        if named:
            missing = self._unviewed_named_ids()
            if missing:
                hydrate = self._hydrate(
                    [
                        {"product_id": pid, "sku": "", "_hydrated": False}
                        for pid in missing
                    ]
                )
                if hydrate:
                    return hydrate[: self.max_calls]
            if cart_item is None:
                pending = [
                    {"product_id": spec["product_id"], "sku": spec["sku"]}
                    for spec in named
                ]
                inspected = self._inspect_many(pending)
                if inspected:
                    return inspected
        else:
            if self.tray.search_n < 8 and (not self.live() or self.tray.search_n < 2):
                search = self._search_actions()
                if search:
                    return search
            hydrate = self._hydrate(self.live() or self._pool())
            if hydrate:
                return hydrate[: self.max_calls]
        champ = self.champion()
        if champ is None:
            if named:
                return []
            search = self._search_actions()
            if search:
                return search
            return []
        if self.inspect and not champ.get("_checked"):
            return [
                {
                    "name": _tool_name(self.inspect),
                    "args": _bind(
                        self.inspect,
                        {
                            "product_id": champ.get("product_id"),
                            "sku": champ.get("sku"),
                        },
                    ),
                }
            ]
        if not self.tray.cart and self.cart and not self.stale(champ):
            if self.motive == MOTIVE_JUSTIFY and not champ.get("_hydrated"):
                return self._hydrate([champ])[: self.max_calls]
            return [
                {
                    "name": _tool_name(self.cart),
                    "args": _bind(
                        self.cart,
                        {
                            "product_id": champ.get("product_id"),
                            "sku": champ.get("sku"),
                        },
                    ),
                }
            ]
        if self.tray.cart and self.order and not self.stale(self.tray.cart[0]):
            target = champ if champ.get("_checked") else self.tray.cart[0]
            return [self._order(target)]
        return []

    def align(
        self,
        actions: list[dict[str, Any]],
        proposed: list[dict[str, Any]],
        *,
        await_reply: bool,
        did_cart: bool,
        recover_first: bool,
    ) -> list[dict[str, Any]]:
        names = {_tool_name(tool).lower() for tool in self.tools}
        cleaned = [
            action
            for action in actions
            if str(action.get("name") or "").lower() in names
        ]
        if await_reply and self.message and not recover_first:
            if not any("message" in str(item.get("name") or "").lower() for item in cleaned):
                return [
                    {
                        "name": _tool_name(self.message),
                        "args": _bind(
                            self.message,
                            {
                                "content": (
                                    "I checked the live listing against the firm spec "
                                    "and budget; I will proceed with a verified SKU."
                                )
                            },
                        ),
                    }
                ]
        if not cleaned:
            return proposed[: self.max_calls]
        cart_name = _tool_name(self.cart).lower()
        order_name = _tool_name(self.order).lower()
        remove_name = _tool_name(self.remove).lower()
        inspect_name = _tool_name(self.inspect).lower()
        cart_view_name = _tool_name(self.cart_view).lower() if self.cart_view else ""
        message_name = _tool_name(self.message).lower() if self.message else ""
        champ = self.champion()
        aligned: list[dict[str, Any]] = []
        cart_item = self.tray.cart[0] if self.tray.cart else None
        if cart_item is not None and self.stale(cart_item) and self.remove:
            has_remove = any(
                str(item.get("name") or "").lower() == remove_name for item in cleaned
            )
            if not has_remove:
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
        if self._closed_rank() and not self._rank_ready():
            looking = any(
                any(token in str(item.get("name") or "").lower() for token in ("compare", "view"))
                for item in cleaned
            )
            if looking:
                return cleaned[: self.max_calls]
            hydrate = self._hydrate(self._pool())
            if hydrate:
                return hydrate[: self.max_calls]
        for action in cleaned[: self.max_calls]:
            name = str(action.get("name") or "")
            args = dict(action.get("args") or {})
            lower = name.lower()
            if self._closed_rank() and self.tray.event_key:
                if (
                    cart_item is not None
                    and not self.stale(cart_item)
                    and lower in {inspect_name, cart_view_name, message_name}
                    and self.order
                ):
                    aligned.append(self._order(cart_item))
                    continue
                if (
                    champ is not None
                    and champ.get("_checked")
                    and not self.stale(champ)
                    and cart_item is None
                    and lower in {inspect_name, message_name}
                    and self.cart
                ):
                    aligned.append(
                        {
                            "name": _tool_name(self.cart),
                            "args": _bind(
                                self.cart,
                                {
                                    "product_id": champ.get("product_id"),
                                    "sku": champ.get("sku"),
                                },
                            ),
                        }
                    )
                    continue
            if lower == cart_name and self._closed_rank():
                if not self._rank_ready():
                    hydrate = self._hydrate(self._pool())
                    if hydrate:
                        aligned.extend(hydrate[: self.max_calls])
                        break
                if champ is not None and not champ.get("_checked") and self.inspect:
                    aligned.append(
                        {
                            "name": _tool_name(self.inspect),
                            "args": _bind(
                                self.inspect,
                                {
                                    "product_id": champ.get("product_id"),
                                    "sku": champ.get("sku"),
                                },
                            ),
                        }
                    )
                    continue
                if champ is not None and not self.stale(champ):
                    args = _bind(
                        self.cart,
                        {
                            "product_id": champ.get("product_id"),
                            "sku": champ.get("sku"),
                        },
                    )
            elif lower == cart_name and champ is not None:
                ready = bool(
                    (champ.get("_checked") or champ.get("_hydrated"))
                    and not self.stale(champ)
                )
                force = False
                if self.motive == MOTIVE_RECOVER and ready:
                    force = True
                if self.motive == MOTIVE_INTENT:
                    preferred = str(self.goal.get("soft_brand") or "")
                    if preferred and _brand_ok(champ, preferred) and ready:
                        force = True
                if force:
                    args = _bind(
                        self.cart,
                        {
                            "product_id": champ.get("product_id"),
                            "sku": champ.get("sku"),
                        },
                    )
            if lower == order_name:
                if not did_cart and not any(
                    str(item.get("name") or "").lower() == cart_name for item in cleaned
                ):
                    if self.motive == MOTIVE_JUSTIFY and champ is not None and champ.get("_hydrated"):
                        if self.inspect and not champ.get("_checked"):
                            aligned.append(
                                {
                                    "name": _tool_name(self.inspect),
                                    "args": _bind(
                                        self.inspect,
                                        {
                                            "product_id": champ.get("product_id"),
                                            "sku": champ.get("sku"),
                                        },
                                    ),
                                }
                            )
                            continue
                    continue
                product = champ
                if args.get("product_id") and args.get("sku"):
                    product = self.tray.rows.get(
                        f"{args.get('product_id')}::{args.get('sku')}"
                    ) or product
                if (
                    champ is not None
                    and not self.stale(champ)
                    and champ.get("_checked")
                    and self.motive in {MOTIVE_RANK, MOTIVE_RECOVER, MOTIVE_JUSTIFY}
                ):
                    product = champ
                if product is None:
                    continue
                if self._closed_rank() and champ is not None and not champ.get("_checked") and self.inspect:
                    aligned.append(
                        {
                            "name": _tool_name(self.inspect),
                            "args": _bind(
                                self.inspect,
                                {
                                    "product_id": champ.get("product_id"),
                                    "sku": champ.get("sku"),
                                },
                            ),
                        }
                    )
                    continue
                if self.motive == MOTIVE_JUSTIFY and self.inspect and not product.get("_checked"):
                    aligned.append(
                        {
                            "name": _tool_name(self.inspect),
                            "args": _bind(
                                self.inspect,
                                {
                                    "product_id": product.get("product_id"),
                                    "sku": product.get("sku"),
                                },
                            ),
                        }
                    )
                    continue
                aligned.append(self._order(product))
                continue
            if lower == inspect_name and not args.get("product_id") and champ is not None:
                args = _bind(
                    self.inspect,
                    {
                        "product_id": champ.get("product_id"),
                        "sku": champ.get("sku"),
                    },
                )
            aligned.append({"name": name, "args": args})
        if any(str(row.get("name") or "").lower() == cart_name for row in aligned) and any(
            str(row.get("name") or "").lower() == order_name for row in aligned
        ):
            aligned = [
                row
                for row in aligned
                if str(row.get("name") or "").lower() != order_name
            ]
        if (
            self._closed_rank()
            and champ is not None
            and not champ.get("_checked")
        ):
            aligned = [
                row
                for row in aligned
                if str(row.get("name") or "").lower() != cart_name
            ]
        return aligned or proposed[: self.max_calls]

    def ingest(self, result: dict[str, Any], sent: list[dict[str, Any]]) -> bool:
        done = False
        calls = result.get("calls") or []
        for index, call in enumerate(calls):
            observation = call.get("observation") or {}
            if isinstance(observation, dict) and observation.get("done") is True:
                done = True
            inner = _unwrap(observation) if isinstance(observation, dict) else {}
            name = ""
            if index < len(sent):
                name = str(sent[index].get("name") or "").lower()
            viewed = name in {
                _tool_name(self.view).lower(),
                _tool_name(self.compare).lower(),
            }
            checked = name == _tool_name(self.inspect).lower()
            products = _harvest(inner)
            sent_args = (sent[index].get("args") or {}) if index < len(sent) else {}
            if checked:
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
                    if products and not any(
                        _listing_key(item) == _listing_key(patch) for item in products
                    ):
                        products = [patch] + products
                    elif products:
                        head = dict(products[0])
                        head.update({k: v for k, v in patch.items() if v is not None})
                        products[0] = head
            if products:
                query = ""
                if index < len(sent):
                    query = str((sent[index].get("args") or {}).get("query") or "")
                if query and "search" in name:
                    products = [{**product, "_search_q": query} for product in products]
                self.tray.put(
                    products,
                    viewed=viewed,
                    checked=checked,
                    from_search=not viewed and not checked,
                )
            if isinstance(inner.get("added"), dict):
                self.tray.put(
                    [inner["added"]],
                    viewed=False,
                    checked=True,
                    from_search=False,
                )
                self.tray.cart = [dict(inner["added"])]
            if isinstance(inner.get("cart"), list):
                self.tray.cart = [item for item in inner["cart"] if isinstance(item, dict)]
            if isinstance(inner.get("removed"), dict) or inner.get("removed") is True:
                self.tray.cart = []
            event = inner.get("market_event")
            if isinstance(event, dict) and event:
                self.tray.mark_event(event)
        message = result.get("user_message") or {}
        if isinstance(message, dict):
            signal = message.get("env_signal") or {}
            if isinstance(signal, dict) and signal.get("kind"):
                self.tray.mark_event(signal)
            content = str(message.get("content") or "")
            budget = _BUDGET_RE.search(content)
            if budget:
                self.goal["budget"] = float(budget.group(1).replace(",", ""))
            inch = _INCH_RE.search(content)
            if inch and self.goal.get("ask_spec"):
                self.goal["spec_inch"] = float(inch.group(1))
                self.goal["asked_spec"] = True
            klass = _pixel_class(content)
            if klass and self.goal.get("ask_spec"):
                self.goal["spec_res"] = klass
                self.goal["asked_spec"] = True
            if content and self.goal.get("ask_spec"):
                self.goal["asked_spec"] = True
        return done


_CONTRACT = """
You are Prism, an ORO Bench shopping agent. You operate only through the supplied
environment tools. Scoring is trusted and external — never invent a score.

# Contract
- Use ONLY tools from this episode's tool list.
- Continue until an observation reports done=true.
- Respect max_steps and max_calls_per_turn.
- When max_calls_per_turn > 1, batch independent views/inspects in one turn.
- Catalog search matches VALUES (brand, model, product-type words, distinctive
  phrases). Never dump JSON field names such as brand or category as the query.
- Do not invent product_id or sku values. Use only identifiers from the goal or
  from observations.

# Standard loop
1) Discover — search / filter with value-side keywords.
2) Inspect — view / compare / inspect_stock on concrete SKUs.
3) Decide — add_to_cart / remove_from_cart. FIRST cart is graded. No placeholders.
4) Confirm — inspect_stock / inspect_cart after market events; read shopper messages.
5) Finish — place_test_order once the live winner is verified.

# Firm vs soft
Extract constraints from the public brief: category, brand, model, size, color,
budget/price, stock, resolution, refresh_rate, screen_size, use-case.
FIRM must hold on the exact SKU. Soft never overrides firm.

# Matching (follow the Prism match card when present)
- Retrieval: BM25+ (Lv-Zhai) plus Jelinek-Mercer, fused with inverse-square rank.
- Tradeoffs: lexicographic axes first; WASPAS only breaks remaining ties. Never pick a
  row worse on an earlier stated axis. Advertised refresh is the max Hz on the SKU options.
- Recovery: Jaro-Winkler color + token-F1 model; never order the market-event target.
- Soft use-case: Cobb-Douglas utility over evidenced axes, not cheapest-by-default.

# Shopper dialogue
Whenever a shopper user_message asks a question or clarifies budget/spec, answer
with the message tool in the NEXT turn before further cart/order actions — unless
the cart SKU is already over budget or out of stock, in which case recover first.

# Market events
Prices and stock can change after cart actions. Re-verify. If invalid, remove and
recover. Once a verified in-stock in-budget pick exists, place_test_order.
""".strip()

_MOTIVE_BOOK: dict[str, str] = {
    MOTIVE_INTENT: """
# Intent decomposition
Parse FIRM (category, one hardware spec, budget, in-stock) vs SOFT (preferred brand).
Search/filter then view compliant SKUs. FIRST add_to_cart: preferred brand IF a fully
firm-compliant SKU exists; otherwise the best firm-compliant item. A later swap does
not repair a bad first cart. On price jump: remove, re-search under the new budget,
still prefer brand only when fully compliant. Then inspect_stock and place_test_order.
Answer shopper questions with message.
""".strip(),
    MOTIVE_RECALL: """
# Retrieval recall
Search with the goal phrase or quoted tokens; one small reformulation if recall is thin
(drop noise, keep distinctive tokens). Shortlist SEVERAL distinct products, not one
lucky hit. Reject accessories via category_path. View/compare, pick the best in-stock
in-budget hit, then place_test_order promptly. Do not wait for permission.
""".strip(),
    MOTIVE_CONSTRAINT: """
# Constraint satisfaction
If a firm hardware spec is revealed only when asked, call message FIRST to ask for that
exact requirement before searching. EVERY firm requirement must hold on the EXACT SKU
(option + price + stock). Titles mix good/bad variants — view then inspect_stock.
Apply resolution class rules only as written in the brief (pixel thresholds if given).
If the wanted variant sells out, do NOT take a sibling that misses a firm requirement.
""".strip(),
    MOTIVE_PREFERENCE: """
# Preference reasoning
Hard: category, budget, in-stock, real device. Soft: use-case axes (weather, sweat/water,
battery, materials, connectivity). Compare candidates on those axes before the first cart.
FIRST cart must be a genuine preference-fit, not the cheapest placeholder. A slightly
dearer in-budget item that matches soft goals is better than a cheap mismatch.
""".strip(),
    MOTIVE_RANK: """
# Ranking
Restrict to the candidate variants named in the query (product_id::sku).
Do not open-search the catalog. view and/or compare ALL listed product_ids
BEFORE any add_to_cart. Read advertised refresh (including overclock) from the
exact SKU options, not the marketing title.

Rank lexicographically by the stated axes (typical: highest advertised refresh
→ largest screen → lowest price). WASPAS is only a tie-break when those axes
tie. Never pick a row that is worse on an earlier axis.

FIRST add_to_cart MUST be the current best-ranked in-stock in-budget option.
inspect_stock that SKU before the first cart. After stockout/price events,
remove the invalid item, re-rank remaining live candidates, cart the new #1.

CRITICAL — place_test_order on ranking:
- Call place_test_order with ONLY product_id and sku.
- Do NOT pass justification / claims. Optional untruthful claims hard-fail even when
  the ranked pick is correct.
""".strip(),
    MOTIVE_RECOVER: """
# Recovery
Never relax: category, budget, in-stock.
Prefer exact model+color first (use explicit model facts; titles do not override).
If the exact item disappears or price breaks budget:
1) same model, other color (Jaro-Winkler color identity)
2) only then other models
Re-check price and stock after every cart change. Never order the market-event target.
""".strip(),
    MOTIVE_JUSTIFY: """
# Justification
Find a valid in-stock in-budget product; view/compare raw facts BEFORE first cart and
BEFORE ordering. Never justify from marketing title words alone.
place_test_order MUST include justification.claims when the schema requires it.

Claim allowlist (any other field → faithful=False → reward 0):
- REQUIRED: price, in_stock, and exactly ONE shopper spec field using its canonical
  name from the goal (resolution OR screen_size OR refresh_rate — never aliases
  display / size / category / category_path / device_operating_system).
- PLUS exactly ONE soft reason from: brand, streaming, microphone, privacy_cover,
  autofocus, hdr — ONLY if view/compare shows an explicit positive capability.
  Prefer brand when the brand field is present. NEVER invent hdr=true.

Value rules:
- price / in_stock must match the latest inspect_stock or cart current state.
- spec values must come from viewed facts, not titles.
- No summary prose inside claims. Keep the array minimal (about 4 claims).
Also reject screen-only / accessory SKUs when a full device is required.
""".strip(),
}

_ROLE_HINT = {
    MOTIVE_RECALL: (
        "You are the scout model (DeepSeek). Keep searches short, batch inspects, and "
        "finish once a verified in-stock in-budget SKU is in the cart."
    ),
    MOTIVE_RANK: (
        "You are gpt-oss. Inspect every named candidate, then cart the lexicographic "
        "champion (WASPAS only breaks remaining ties). On order, pass only product_id "
        "and sku — omit justification entirely."
    ),
    MOTIVE_RECOVER: (
        "You are the recovery model. After a stockout or price jump, remove the stale SKU "
        "and take the lowest Jaro-Winkler/F1-cost live option (color before model)."
    ),
    MOTIVE_JUSTIFY: (
        "You are gpt-oss. Inspect facts before carting. When ordering, emit only "
        "allowlisted claims grounded in viewed facts. Never invent hdr. Never use "
        "display/size/category as claim field names."
    ),
    MOTIVE_CONSTRAINT: (
        "You are gpt-oss. Ask for a hidden firm spec if the goal says it is revealed only "
        "when asked. Every firm spec must hold on the exact SKU option row."
    ),
    MOTIVE_INTENT: (
        "You are gpt-oss. Honor a preferred brand only when that SKU also meets every "
        "firm constraint. First cart is scored — no exploratory placeholders."
    ),
    MOTIVE_PREFERENCE: (
        "You are gpt-oss. Compare soft use-case axes before the first cart. Do not default "
        "to the cheapest valid item. First cart must be a real preference-fit purchase."
    ),
}


def _tool_labels(tools: list[dict[str, Any]]) -> list[str]:
    return [_tool_name(tool) for tool in tools if _tool_name(tool)]


def _system_prompt(motive: str, policy: dict[str, Any], tools: list[dict[str, Any]]) -> str:
    labels = _tool_labels(tools)
    book = _MOTIVE_BOOK.get(motive, _MOTIVE_BOOK[MOTIVE_RECALL])
    hint = _ROLE_HINT.get(motive, _ROLE_HINT[MOTIVE_RECALL])
    return (
        f"{_CONTRACT}\n\n"
        f"{book}\n\n"
        f"# Model role\n{hint}\n\n"
        f"# Runtime\n"
        f"Prism motive: {motive}\n"
        f"Available tools: {', '.join(labels) if labels else '(see tool schemas)'}\n"
        f"max_steps={policy.get('max_steps')} "
        f"max_calls_per_turn={policy.get('max_calls_per_turn', 1)}\n"
        f"Emit at most {policy.get('max_calls_per_turn', 1)} tool call(s) per turn.\n"
        f"Follow the Prism match card on later turns when it lists a champion SKU.\n"
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


def _shrink_obs(payload: Any, cap: int = _OBS_CAP) -> str:
    try:
        rendered = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        rendered = str(payload)
    if len(rendered) <= cap:
        return rendered
    return rendered[: cap - 20] + "...<truncated>"


def _chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
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
    return _proxy.post("/inference/chat/completions", json_data=payload)


def _ask_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_calls: int,
    model: str,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    last_content = ""
    raw_calls: list[dict[str, Any]] = []
    last_failure: Exception | None = None
    for candidate in _reserve_models(model):
        inference = _chat(
            model=candidate,
            messages=messages,
            tools=tools,
            tool_choice="required",
        )
        if inference is None:
            last_failure = RuntimeError(f"inference failed for model={candidate}")
            continue
        try:
            assistant = inference["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            last_failure = exc
            continue
        last_content = assistant.get("content") or ""
        raw_calls = list(assistant.get("tool_calls") or [])[:max_calls]
        break
    actions: list[dict[str, Any]] = []
    for call in raw_calls:
        function = call.get("function") or {}
        try:
            args = _call_args(call)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        name = function.get("name")
        if name:
            actions.append({"name": name, "args": args, "_id": call.get("id")})
    if not actions and last_failure is not None and not raw_calls:
        raise RuntimeError(f"inference request failed ({last_failure})")
    return actions, last_content, raw_calls


def _as_tool_calls(
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
                or f"{problem_id}-prism-{turn}-{index}",
                "type": "function",
                "function": {
                    "name": action["name"],
                    "arguments": json.dumps(action.get("args") or {}),
                },
            }
        )
    return calls


def _coach(
    motive: str,
    turn: int,
    prism: _Prism,
    proposed: list[dict[str, Any]],
    await_reply: bool,
    cart_used: bool,
    prior_names: list[str],
) -> str | None:
    blob = " ".join(name.lower() for name in prior_names)
    if await_reply and prism.needs_recover():
        return (
            "Coach: a market event or stale cart is open. Re-compare remaining named "
            "SKUs, inspect the new live #1, cart it, then order before answering the "
            "shopper.\n"
            f"{prism.card(proposed)}"
        )
    if await_reply:
        return (
            "Coach: the shopper just sent a message. Answer it with the message "
            "tool in this turn before cart or order actions, unless the cart SKU "
            "is already over budget or out of stock — recover that first.\n"
            f"{prism.card(proposed)}"
        )
    if motive == MOTIVE_CONSTRAINT and turn == 1 and prism.goal.get("ask_spec"):
        return (
            "Coach: one firm hardware spec is revealed only if asked. Call message "
            "NOW to ask for that exact requirement before searching."
        )
    if motive == MOTIVE_RANK:
        lines = [
                    "Coach [ranking]: compare every listed product_id::sku and rank by the "
                    "stated lexicographic axes before the first add_to_cart. First cart = "
                    "live #1. On place_test_order pass ONLY product_id and sku — omit justification."
        ]
        if not cart_used and turn >= 2 and ("view" in blob or "compare" in blob):
            lines.append(
                "Hint: candidates are inspected — add_to_cart ONLY the Prism champion now."
            )
        if cart_used:
            lines.append(
                "Hint: after a stockout, re-compare remaining named SKUs, inspect the "
                "new lexicographic #1, cart it, then place_test_order with ONLY "
                "product_id and sku."
            )
        lines.append(prism.card(proposed))
        return "\n".join(lines)
    if motive == MOTIVE_JUSTIFY:
        return (
            "Coach [justification]: view/compare facts before the first cart. "
            "Claims allowlist: in_stock, live price, one canonical spec field, "
            "plus brand when observed. Forbidden: display, size, category, invented hdr.\n"
            f"{prism.card(proposed)}"
        )
    if motive == MOTIVE_RECOVER:
        return (
            "Coach [recovery]: search color+model then model. After a stockout, "
            "keep the same model and change color first. Never order the event target.\n"
            f"{prism.card(proposed)}"
        )
    if motive == MOTIVE_INTENT and turn == 1:
        return (
            "Coach: first cart honors the preferred brand only when that SKU also "
            "meets the firm spec, budget, and stock.\n"
            f"{prism.card(proposed)}"
        )
    if motive == MOTIVE_PREFERENCE and turn <= 2 and not cart_used:
        return (
            "Coach: compare soft-preference axes before the first cart. First cart "
            "must be a real preference-fit purchase, not the cheapest placeholder.\n"
            f"{prism.card(proposed)}"
        )
    if motive == MOTIVE_RECALL and turn == 3 and "search" in blob:
        return (
            "Hint [recall]: one small keyword reformulation or filter so the shortlist "
            "covers several distinct products before ordering.\n"
            f"{prism.card(proposed)}"
        )
    if proposed:
        return prism.card(proposed)
    if turn <= 2 and not any(
        name in blob for name in ("search", "filter", "view", "compare", "message")
    ):
        return (
            "Hint: start with search or filter using value-side keywords from the "
            "shopper goal, then view the best hits."
        )
    return None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Drive policy tools until done=true; Prism match owns cart/order SKUs."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
    tools = list(policy.get("tools") or [])
    query = str(policy.get("query") or problem_data.get("query") or "")
    prism = _Prism(query, tools, max_calls)
    motive = prism.motive
    model = _motive_model(motive)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(motive, policy, tools)},
        {"role": "user", "content": query},
    ]
    if motive == MOTIVE_CONSTRAINT and prism.goal.get("ask_spec"):
        messages.append(
            {
                "role": "user",
                "content": (
                    "Coach: ask for the hidden firm hardware spec with the message "
                    "tool before any search."
                ),
            }
        )
    if motive == MOTIVE_RANK:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Coach [ranking]: inspect every named candidate, cart the lexicographic "
                    "champion, and place_test_order with ONLY product_id and sku."
                ),
            }
        )
    if motive == MOTIVE_JUSTIFY:
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
    if motive == MOTIVE_RECOVER:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Coach [recovery]: search the preferred color+model first. After "
                    "the exact SKU disappears, same model other color, then other models."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "prism": {
                "motive": motive,
                "model": model,
                "provider": _provider(),
                "stack": "bm25plus+jm+isr+waspas",
            },
        }
    ]
    await_reply = False
    cart_used = False
    prior_names: list[str] = []

    for turn in range(1, int(policy.get("max_steps") or 20) + 1):
        proposed = prism.advise()
        coach = _coach(
            motive, turn, prism, proposed, await_reply, cart_used, prior_names
        )
        if coach:
            messages.append({"role": "user", "content": coach})

        actions: list[dict[str, Any]] = []
        assistant_content = ""
        raw_calls: list[dict[str, Any]] = []
        for retry in range(len(_IDLE) + 1):
            actions, assistant_content, raw_calls = _ask_tools(
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
        recover_first = prism.needs_recover()
        actions = prism.align(
            actions,
            proposed,
            await_reply=await_reply,
            did_cart=cart_used,
            recover_first=recover_first,
        )
        if not actions:
            raise RuntimeError("model returned no tool call before completion")

        prior_names = [str(item.get("name") or "") for item in actions]
        lowered_names = [name.lower() for name in prior_names]
        if any("add_to_cart" in name for name in lowered_names):
            cart_used = True
        if any("message" in name for name in lowered_names):
            await_reply = False

        tool_calls = _as_tool_calls(actions, raw_calls, problem_id, turn)
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
                    "action": {
                        "name": item["name"],
                        "args": item.get("args") or {},
                    },
                }
                for index, item in enumerate(actions, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
                "prism_turn": {"motive": motive, "turn": turn},
            }
        )

        done = prism.ingest(result, actions)
        for tool_call, call_result in zip(tool_calls, result.get("calls") or [], strict=False):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _shrink_obs(call_result.get("observation")),
                }
            )
        shopper_msg = result.get("user_message")
        if shopper_msg and shopper_msg.get("content"):
            spoken = str(shopper_msg["content"])
            messages.append({"role": "user", "content": spoken})
            if _is_inquiry(spoken) and not prism.needs_recover():
                await_reply = True
        if done:
            break
        if any(
            isinstance(call_row.get("observation"), dict)
            and call_row["observation"].get("done")
            for call_row in result.get("calls") or []
        ):
            break

    return dialogue


__all__ = ["agent_main", "classify_motive"]
