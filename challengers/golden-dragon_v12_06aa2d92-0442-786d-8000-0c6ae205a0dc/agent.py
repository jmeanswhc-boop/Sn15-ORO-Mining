"""ORO Bench scoreboard agent for generated environments.

Environment contract follows the public reference loop: copy the opaque
binding, emit unique call_id / idempotency_key groups, use only policy_view
tools, and stop when any observation reports done=true.

The model loop follows a lane-playbook steward (shopper dialogue, idle
retries, allowlisted third-party models). TF5 ranking, TF6 recovery, and
TF7 justification are scored by a catalog scoreboard: CombSUM of Okapi BM25
plus Jaccard, TOPSIS on stated tradeoff axes, a continuous relaxation cost
for stockouts, and schema-enum claims grounded in viewed facts.

Decisions come from the public goal, dynamic tool schemas, and live
observations. No product identifiers or suite phrases are stored.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


FAMILY_INTENT = "intent_decomposition"
FAMILY_RECALL = "retrieval_recall"
FAMILY_CONSTRAINT = "constraint_satisfaction"
FAMILY_PREFERENCE = "preference_reasoning"
FAMILY_RANK = "ranking"
FAMILY_RECOVER = "recovery"
FAMILY_JUSTIFY = "justification"

_OKAPI_K1 = 1.2
_OKAPI_B = 0.35
_RRF_K = 48.0
_COMB_BM25 = 0.72
_COMB_JAC = 0.28

_PAIR_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._-]+)")
_DIGIT = re.compile(r"(\d+(?:\.\d+)?)")
_HERTZ = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hz|hertz)\b", re.I)
_INCHES = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(?:\"|''|in(?:ch(?:es)?)?)\b",
    re.I,
)
_CEILING = re.compile(
    r"(?:at or under|at or below|under|below|up to|upto|budget|<=)\s*"
    r"(\d+(?:\.\d+)?)",
    re.I,
)
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.I | re.S)
_JSON_BLOB = re.compile(r"\{.*\}", re.S)

_HUE_WORDS = (
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
    "laptop": "Laptops",
    "camera": "Cameras",
    "watch": "Watches",
    "speaker": "Speakers",
    "keyboard": "Keyboards",
    "mouse": "Mice",
    "router": "Routers",
    "printer": "Printers",
}

_STOP_BRAND = {
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

_CLAIM_CORE = (
    "brand",
    "in_stock",
    "price",
    "refresh_rate",
    "resolution",
    "screen_size",
)
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
}

_CHUTES = {
    "scout": "deepseek-ai/DeepSeek-V3.2-TEE",
    "rank": "openai/gpt-oss-120b-TEE",
    "recover": "zai-org/GLM-5.2-TEE",
    "justify": "moonshotai/Kimi-K2.6-TEE",
    "heavy": "Qwen/Qwen3.5-397B-A17B-TEE",
}
_OPENROUTER = {
    "scout": "deepseek/deepseek-v3.2",
    "rank": "openai/gpt-oss-120b",
    "recover": "z-ai/glm-5.2",
    "justify": "moonshotai/kimi-k2.6",
    "heavy": "qwen/qwen3.5-397b-a17b",
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

_proxy = ProxyClient(timeout=120, max_retries=2)


def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()


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


def _family_model(family: str) -> str:
    if family == FAMILY_RANK:
        return _role_model("rank")
    if family == FAMILY_RECOVER:
        return _role_model("recover")
    if family == FAMILY_JUSTIFY:
        return _role_model("justify")
    if family in {FAMILY_CONSTRAINT, FAMILY_PREFERENCE, FAMILY_INTENT}:
        return _role_model("heavy")
    return _role_model("scout")


def _fallback_models(primary: str) -> list[str]:
    forced = _alias_model(getenv("SANDBOX_MODEL") or "")
    pool = _RESERVE_OR if _provider() == "openrouter" else _RESERVE_TEE
    ordered: list[str] = [primary]
    if forced and forced not in ordered:
        ordered.append(forced)
    for item in pool:
        if item not in ordered:
            ordered.append(item)
    return ordered[:4]


def _trim(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fold(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _squash(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _words(text: str) -> list[str]:
    return [token for token in _fold(text).split() if len(token) > 1]


def _goal_text(query: str) -> str:
    return re.split(r"\n\s*Task rules:", query or "", maxsplit=1)[0].strip()


def _fn(tool: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(tool, dict):
        return {}
    block = tool.get("function")
    return block if isinstance(block, dict) else tool


def _tool_label(tool: dict[str, Any] | None) -> str:
    return str(_fn(tool).get("name") or "")


def _tool_props(tool: dict[str, Any] | None) -> dict[str, Any]:
    parameters = _fn(tool).get("parameters") or {}
    properties = parameters.get("properties") or {}
    return properties if isinstance(properties, dict) else {}


def _locate(tools: list[dict[str, Any]], *needles: str) -> dict[str, Any] | None:
    wanted = {item.lower() for item in needles}
    for tool in tools:
        name = _tool_label(tool).lower()
        if name in wanted:
            return tool
    for tool in tools:
        name = _tool_label(tool).lower()
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
            filled[real] = _cast(value, properties.get(real) or {})
            break
    return filled


def _call_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _loads_json(text: str) -> dict[str, Any] | None:
    blob = (text or "").strip()
    if not blob:
        return None
    fenced = _JSON_BLOCK.search(blob)
    if fenced:
        blob = fenced.group(1)
    else:
        match = _JSON_BLOB.search(blob)
        if match:
            blob = match.group(0)
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


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
    raw = (
        product.get("facts")
        or product.get("specification")
        or product.get("attributes")
        or {}
    )
    return raw if isinstance(raw, dict) else {}


def _doc(product: dict[str, Any]) -> str:
    parts = [
        product.get("title"),
        product.get("brand"),
        product.get("options"),
        product.get("description"),
        json.dumps(_facts(product), sort_keys=True),
        json.dumps(product.get("category_path") or [], sort_keys=True),
    ]
    return _fold(" ".join(_trim(part) for part in parts if part))


def _price(product: dict[str, Any]) -> float | None:
    raw = product.get("price")
    if isinstance(raw, dict):
        raw = raw.get("value") or raw.get("amount") or raw.get("current")
    try:
        if raw is None or raw == "":
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _stocked(product: dict[str, Any]) -> bool:
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
    found = _DIGIT.search(str(text or ""))
    if not found:
        return None
    try:
        return float(found.group(1))
    except ValueError:
        return None


def _hertz(product: dict[str, Any]) -> float | None:
    chunks = [
        product.get("options"),
        _facts(product).get("refresh_rate"),
        _facts(product).get("refresh"),
        product.get("title"),
        product.get("description"),
    ]
    best: float | None = None
    for chunk in chunks:
        for match in _HERTZ.findall(str(chunk or "")):
            try:
                value = float(match)
            except ValueError:
                continue
            if best is None or value > best:
                best = value
    return best


def _screen(product: dict[str, Any]) -> float | None:
    facts = _facts(product)
    hydrated = bool(product.get("_seen") or facts)
    chunks = [product.get("options")]
    if not hydrated:
        chunks.append(product.get("title"))
    for chunk in chunks:
        found = _INCHES.findall(str(chunk or ""))
        if found:
            try:
                return float(found[-1])
            except ValueError:
                pass
    for key in ("screen_size", "size", "display_size"):
        raw = facts.get(key)
        numbers = [float(token) for token in _DIGIT.findall(str(raw or ""))]
        distinct = {round(item, 1) for item in numbers if 4 <= item <= 100}
        if len(distinct) > 1:
            continue
        found = _INCHES.findall(str(raw or ""))
        if found:
            try:
                return float(found[-1])
            except ValueError:
                continue
        value = _number(raw)
        if value is not None and 4 <= value <= 100:
            return value
    return None


def _pixel_class(text: Any) -> str | None:
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
    if re.search(r"\bhd\b", blob) and "qhd" not in blob:
        return "hd"
    return None


def _axis_read(product: dict[str, Any], name: str) -> float | None:
    key = _fold(name).replace(" ", "_")
    if key in {"price", "cost"}:
        return _price(product)
    if "refresh" in key or key in {"hz"}:
        return _hertz(product)
    if "screen" in key or key in {"size", "inches"}:
        return _screen(product)
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
    length_norm = 1.0 - _OKAPI_B + _OKAPI_B * (doc_len / max(avgdl, 1.0))
    for token, qtf in query_tf.items():
        tf = doc_tf.get(token, 0)
        if tf <= 0:
            continue
        idf = math.log((docs - df[token] + 0.5) / (df[token] + 0.5) + 1.0)
        tf_norm = (tf * (_OKAPI_K1 + 1.0)) / (tf + _OKAPI_K1 * length_norm)
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
            key = _listing_key(product) or str(
                product.get("product_id") or product.get("id") or rank
            )
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            keep[key] = product
    return [
        keep[key]
        for key, _score in sorted(scores.items(), key=lambda item: -item[1])
    ]


def _topsis(
    products: list[dict[str, Any]],
    axes: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    if not products:
        return []
    if not axes:
        return list(products)
    matrix: list[list[float | None]] = []
    for product in products:
        row: list[float | None] = []
        for name, direction in axes:
            value = _axis_read(product, name)
            if value is None:
                row.append(None)
            elif direction == "asc":
                row.append(-value)
            else:
                row.append(value)
        matrix.append(row)
    cols = len(axes)
    norms = [0.0] * cols
    for row in matrix:
        for index, value in enumerate(row):
            if value is not None:
                norms[index] += value * value
    norms = [math.sqrt(item) if item > 0 else 1.0 for item in norms]
    scaled: list[list[float]] = []
    for row in matrix:
        scaled.append(
            [
                (0.0 if value is None else value / norms[index])
                for index, value in enumerate(row)
            ]
        )
    ideal = [max(col) for col in zip(*scaled)] if scaled else []
    anti = [min(col) for col in zip(*scaled)] if scaled else []
    scored: list[tuple[float, dict[str, Any]]] = []
    for product, row in zip(products, scaled):
        plus = math.sqrt(sum((left - right) ** 2 for left, right in zip(row, ideal)))
        minus = math.sqrt(sum((left - right) ** 2 for left, right in zip(row, anti)))
        closeness = minus / (plus + minus) if (plus + minus) else 0.0
        scored.append((closeness, product))
    scored.sort(key=lambda item: -item[0])
    return [product for _score, product in scored]


def _lex_break(
    products: list[dict[str, Any]],
    axes: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    def key(product: dict[str, Any]) -> tuple:
        parts: list[float] = []
        for name, direction in axes:
            value = _axis_read(product, name)
            if value is None:
                parts.extend([1.0, 0.0])
                continue
            parts.append(0.0)
            parts.append(-value if direction == "desc" else value)
        return tuple(parts)

    return sorted(products, key=key)


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
            name = _axis_name(raw)
            if name in seen:
                continue
            seen.add(name)
            axes.append(
                (name, "asc" if "cheap" in raw or direction == "asc" else direction)
            )
    if "cheapest" in lowered or "lowest price" in lowered:
        if "price" not in seen:
            axes.append(("price", "asc"))
    return axes


def _needs_why(query: str) -> bool:
    return bool(
        re.search(
            r"\bwhy\b|walk me through|justif|reason i can trust|give me a reason|"
            r"tell me why|a reason i can|structured justification",
            _goal_text(query),
            re.I,
        )
    )


def _hue_of(text: str) -> str:
    blob = f" {_fold(text)} "
    for color in _HUE_WORDS:
        if f" {color} " in blob:
            return color
    return ""


def _model_of(text: str) -> str:
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


def _model_ok(product: dict[str, Any], model: str) -> bool:
    wanted = _squash(model)
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
        text = _trim(source)
        if not text:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(model)}(?![a-z0-9])", text, re.I):
            return True
        compact = _squash(text)
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


def _hue_ok(product: dict[str, Any], color: str) -> bool:
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
    blob = f" {_fold(' '.join(_trim(item) for item in sources if item))} "
    if f" {wanted} " in blob:
        return True
    alias = {"grey": "gray", "gray": "grey"}.get(wanted, "")
    return bool(alias) and f" {alias} " in blob


def _relax_cost(
    product: dict[str, Any],
    model: str,
    color: str,
    event_stem: str,
) -> float:
    has_model = _model_ok(product, model) if model else True
    has_color = _hue_ok(product, color) if color else True
    if has_model and has_color:
        cost = 0.0
    elif has_model:
        cost = 1.0
    else:
        cost = 3.0
    stem = _sku_stem(product.get("sku"))
    if event_stem and stem == event_stem:
        cost -= 0.15
    price = _price(product)
    if price is not None:
        cost += min(price, 1.0e7) / 1.0e9
    return cost


def _kind_of(text: str) -> str:
    blob = _fold(text)
    for noun in _KIND_WORDS:
        if re.search(rf"\b{noun}s?\b", blob):
            return noun
    return ""


def _brand_of(text: str) -> str:
    for match in re.finditer(
        r"\b([A-Z][A-Za-z0-9.&-]*(?:\s+[A-Z][A-Za-z0-9.&-]*){0,2})\b",
        text or "",
    ):
        phrase = _trim(match.group(1)).rstrip(".")
        tokens = _fold(phrase).split()
        if not tokens or any(token in _STOP_BRAND for token in tokens):
            continue
        if _kind_of(phrase):
            continue
        if re.search(r"\d", phrase) and _pixel_class(phrase):
            continue
        return phrase
    return ""


def _spec_bits(text: str) -> list[str]:
    bits: list[str] = []
    blob = text or ""
    if re.search(r"720p", blob, re.I):
        bits.append("720p")
    elif re.search(r"1080p|full hd", blob, re.I):
        bits.append("1080p")
    elif re.search(r"1440p|\bqhd\b", blob, re.I):
        bits.append("1440p")
    elif re.search(r"2160p|\b4k\b", blob, re.I):
        bits.append("4k")
    inch = _INCHES.search(blob)
    if inch:
        bits.append(f"{inch.group(1)} inch")
    return bits


def _kind_ok(product: dict[str, Any], kind: str) -> bool:
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
                " ".join(_trim(value) for value in facts.values()),
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


def _accessory(product: dict[str, Any], kind: str) -> bool:
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
    return bool(kind) and not _kind_ok(product, kind)


def _brand_ok(product: dict[str, Any], brand: str) -> bool:
    wanted = _fold(brand)
    if not wanted:
        return True
    got = _fold(product.get("brand") or "")
    if got and (wanted in got or got in wanted):
        return True
    return wanted in _fold(product.get("title") or "")


def _product_pixels(product: dict[str, Any]) -> str | None:
    facts = _facts(product)
    for key in ("display", "resolution", "screen"):
        klass = _pixel_class(facts.get(key))
        if klass:
            return klass
    if product.get("_seen"):
        return None
    return _pixel_class(product.get("title"))


def _spec_ok(product: dict[str, Any], goal: dict[str, Any]) -> bool:
    spec_res = goal.get("spec_res")
    spec_inch = goal.get("spec_inch")
    if spec_res:
        got = _product_pixels(product)
        if got != spec_res:
            return False
    if spec_inch is not None:
        inches = _screen(product)
        if inches is None or abs(inches - float(spec_inch)) > 0.2:
            return False
    return True


def _query_has_model(product: dict[str, Any], model: str) -> bool:
    wanted = _fold(model)
    if not wanted:
        return False
    blob = _fold(product.get("_query") or "")
    if not blob:
        return False
    if wanted in blob.split():
        return True
    return _squash(wanted) in _squash(blob)


def _sku_stem(sku: Any) -> str:
    text = str(sku or "")
    return text.rsplit("-", 1)[0] if text else ""


def _keeps(
    product: dict[str, Any],
    goal: dict[str, Any],
    *,
    strict: bool,
) -> bool:
    if goal.get("named_keys"):
        return True
    kind = str(goal.get("kind_token") or "")
    if _accessory(product, kind):
        return False
    if kind and not _kind_ok(product, kind):
        return False
    model = str(goal.get("model") or "")
    if goal.get("family") == FAMILY_RECOVER and model:
        if not (_model_ok(product, model) or _query_has_model(product, model)):
            return False
    brand = str(goal.get("brand") or "")
    if brand and goal.get("family") != FAMILY_RECOVER:
        titled = bool(product.get("title") or product.get("brand") or strict)
        if titled and not _brand_ok(product, brand):
            return False
    if strict and goal.get("family") == FAMILY_JUSTIFY and not _spec_ok(product, goal):
        return False
    return True


def _money(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _inch_text(inches: float) -> str:
    if abs(inches - round(inches)) < 0.05:
        return f"{int(round(inches))}in"
    return f"{inches:g}in"


def _claim_fields(tool: dict[str, Any] | None) -> list[str]:
    spec = (_tool_props(tool).get("justification") or {}).get("properties") or {}
    claims = spec.get("claims") or {}
    items = claims.get("items") or {}
    field = (items.get("properties") or {}).get("field") or {}
    enum = field.get("enum")
    if isinstance(enum, list) and enum:
        return [str(item) for item in enum]
    return list(_CLAIM_CORE) + [name for name in _CLAIM_SOFT if name not in _CLAIM_CORE]


def _needed_fields(query: str) -> list[str]:
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


def _compile_claims(
    product: dict[str, Any],
    fields: list[str],
    allowed: list[str],
) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    facts = _facts(product)
    permit = {item.lower() for item in allowed} - _CLAIM_BLOCK
    blob = _fold(" ".join(_trim(item) for item in facts.values()))
    for field in fields:
        if field not in permit:
            continue
        value: str | None = None
        if field == "in_stock":
            value = "true" if _stocked(product) else "false"
        elif field == "price":
            price = _price(product)
            value = _money(price) if price is not None else None
        elif field == "resolution":
            value = _product_pixels(product)
        elif field == "screen_size":
            inches = _screen(product)
            value = _inch_text(inches) if inches is not None else None
        elif field == "refresh_rate":
            hz = _hertz(product)
            if hz is None:
                value = None
            elif abs(hz - round(hz)) < 0.05:
                value = str(int(round(hz)))
            else:
                value = str(hz)
        elif field == "brand":
            value = _trim(product.get("brand")) or _trim(facts.get("brand")) or None
        elif field in _CLAIM_SOFT:
            token = field.replace("_", " ")
            if token in blob or field in blob:
                value = "true"
        if not value:
            continue
        claims.append({"field": field, "value": str(value)})
    return claims


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
            row["_seen"] = True
            rows.append(row)
        return rows
    if product_id not in (None, "") and any(
        key in node
        for key in ("title", "price", "sku", "brand", "facts", "in_stock")
    ):
        return [node]
    return []


def _collect(payload: Any) -> list[dict[str, Any]]:
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


def _inner(observation: dict[str, Any]) -> dict[str, Any]:
    nested = observation.get("observation")
    if isinstance(nested, dict):
        return nested
    return observation


def classify_family(query: str) -> str:
    shopper = _goal_text(query)
    lowered = (query or "").lower()
    pairs = _named_pairs(query)
    if pairs:
        return FAMILY_RANK
    if re.search(
        r"\b(highest|lowest|largest|smallest|cheapest)\b.+\b(among|ties|tradeoff)\b",
        shopper,
        re.I,
    ):
        return FAMILY_RANK
    if _needs_why(query):
        return FAMILY_JUSTIFY
    if re.search(
        r"disappear|unavailable|sold out|another color|same model|least[- ]relax",
        shopper,
        re.I,
    ) and (_model_of(shopper) or _hue_of(shopper)):
        return FAMILY_RECOVER
    if re.search(r"only if (you )?ask|ask (me )?for (the )?exact", shopper, re.I):
        return FAMILY_CONSTRAINT
    if re.search(r"soft preference|i'?d rather|use[- ]case|workout|sweat", shopper, re.I):
        return FAMILY_PREFERENCE
    if re.search(r"prefer(?:red)? brand|firm requirements matter|nice-to-have", lowered):
        return FAMILY_INTENT
    return FAMILY_RECALL


def _read_goal(query: str, family: str) -> dict[str, Any]:
    shopper = _goal_text(query)
    budget_hit = _CEILING.search(shopper)
    color = _hue_of(shopper)
    model = _model_of(shopper)
    if not model:
        colored = re.search(
            rf"\b(?:{'|'.join(_HUE_WORDS)})\s+([A-Za-z0-9._-]{{2,}})",
            shopper,
            re.I,
        )
        if colored:
            model = colored.group(1).rstrip(".,;")
    category = ""
    match = re.search(r"in the (.+?) category", shopper, re.I)
    if match:
        category = _trim(match.group(1))
    kind_token = _kind_of(shopper)
    brand = _brand_of(shopper)
    if not brand:
        brand_match = re.search(
            r"\b(?:prefer(?:s)?|brand)\s+([A-Z][\w.&-]*(?:\s+[A-Z][\w.&-]*){0,2})",
            shopper,
        )
        if brand_match and "prefer the" not in _fold(brand_match.group(0)):
            brand = _trim(brand_match.group(1))
    named = _named_pairs(query)
    axes = _axes_from(shopper)
    spec_res = _pixel_class(shopper)
    spec_inch = None
    inch_hit = _INCHES.search(shopper)
    if inch_hit:
        spec_inch = float(inch_hit.group(1))
    color_query = "black" if color == "midnight" and "black" in _fold(shopper) else color
    phrases: list[str] = []
    if model:
        phrases.append(" ".join(part for part in (color_query, model) if part))
        phrases.append(model)
    bits = _spec_bits(shopper)
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
    if not phrases:
        phrases = [" ".join(_words(shopper)[:4]) or kind_token or "in stock"]
    phrases = [item for item in dict.fromkeys(phrases) if item]
    return {
        "family": family,
        "budget": float(budget_hit.group(1)) if budget_hit else None,
        "in_stock": True,
        "brand": brand,
        "category": category or _KIND_PATH.get(kind_token, ""),
        "kind_token": kind_token,
        "model": model,
        "named_keys": named,
        "axes": axes,
        "need_reason": _needs_why(query),
        "spec_fields": _needed_fields(shopper),
        "spec_res": spec_res,
        "spec_inch": spec_inch,
        "search_phrases": phrases,
        "color": color_query,
        "ask_spec": bool(
            re.search(r"only if (you )?ask|ask (me )?for (the )?exact", shopper, re.I)
        ),
    }


def _merge_goal(parsed: dict[str, Any], query: str, family: str) -> dict[str, Any]:
    base = _read_goal(query, family)
    kind = str(parsed.get("kind") or parsed.get("family") or base["family"]).lower()
    if kind in {
        FAMILY_RANK,
        FAMILY_RECOVER,
        FAMILY_JUSTIFY,
        FAMILY_INTENT,
        FAMILY_RECALL,
        FAMILY_CONSTRAINT,
        FAMILY_PREFERENCE,
    }:
        family = kind
    named = parsed.get("named_keys") or parsed.get("candidates")
    clean_keys: list[dict[str, str]] = []
    if isinstance(named, list):
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
        clean_keys = base["named_keys"]
    axes_raw = parsed.get("axes") or parsed.get("rank_axes")
    axes: list[tuple[str, str]] = []
    if isinstance(axes_raw, list):
        for item in axes_raw:
            if isinstance(item, dict):
                name = _axis_name(str(item.get("name") or item.get("attribute") or ""))
                direction = str(item.get("direction") or "desc").lower()
                if name:
                    axes.append((name, "asc" if direction == "asc" else "desc"))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                axes.append((_axis_name(str(item[0])), str(item[1])))
    if not axes:
        axes = base["axes"]
    phrases = parsed.get("search_phrases") or base["search_phrases"]
    if isinstance(phrases, str):
        phrases = [phrases]
    spec_fields = [
        str(field)
        for field in (parsed.get("spec_fields") or [])
        if str(field).strip()
    ]
    merged_fields: list[str] = []
    for field in list(base["spec_fields"]) + spec_fields:
        if field not in merged_fields:
            merged_fields.append(field)
    spec_inch = base.get("spec_inch")
    if spec_inch is None:
        spec_inch = parsed.get("spec_inch")
    return {
        "family": family if not clean_keys else FAMILY_RANK if family == FAMILY_RECALL else family,
        "budget": parsed.get("budget", base["budget"]),
        "in_stock": parsed.get("in_stock", True) is not False,
        "brand": _trim(parsed.get("brand") or base["brand"]),
        "category": _trim(parsed.get("category") or base["category"]),
        "model": _trim(parsed.get("model") or base["model"]),
        "color": _fold(parsed.get("color") or base["color"]),
        "named_keys": clean_keys,
        "axes": axes,
        "need_reason": bool(base["need_reason"] or parsed.get("need_reason")),
        "spec_fields": merged_fields or base["spec_fields"],
        "spec_res": base.get("spec_res") or parsed.get("spec_res"),
        "spec_inch": spec_inch,
        "kind_token": _fold(parsed.get("kind_token") or base.get("kind_token") or ""),
        "search_phrases": [_trim(item) for item in phrases if _trim(item)]
        or base["search_phrases"],
        "ask_spec": bool(base.get("ask_spec") or parsed.get("ask_spec")),
    }


_CONTRACT = """
You are Horizon, an ORO Bench shopping steward in a generated environment.

# Contract
- Use ONLY tools supplied in this episode's tool list.
- Continue until an observation reports done=true.
- Respect max_steps and max_calls_per_turn.
- When max_calls_per_turn > 1, batch independent inspect/compare calls.
- Copy no catalog answers from memory. Every SKU must come from observations.

# Standard loop
1) Discover — search / filter with VALUE-side keywords (brand, model, kind, spec tokens).
   Catalog search matches VALUES, not JSON field names. Never query the word "product".
2) Inspect — view / compare / inspect_stock on concrete SKUs. Titles mix good and bad variants.
3) Decide — add_to_cart / remove_from_cart. The first cart is scored. No placeholders.
4) Confirm — re-inspect after market events. Never order a stockout or over-budget SKU.
5) Finish — place_test_order once the live winner is verified.

# Firm vs soft
Firm constraints always win: category/kind, budget, in-stock, named model/color, stated hardware spec.
Soft preferences never override firm constraints.

# Matching
Follow the scoreboard brief when present. It fuses Okapi BM25 with Jaccard via CombSUM,
ranks named candidates with TOPSIS on the shopper's axes, and applies a continuous
relaxation cost after stockouts. Do not invent product IDs.

# Shopper dialogue
If a shopper user_message asks a question, answer with the message tool in the NEXT
turn before further cart/order actions. Keep replies short and fact-backed.

# Market events
After stockout or price_change, remove the stale cart SKU, inspect the next live
winner, then cart and order. Never order the event target.
""".strip()

_FAMILY_BOOK = {
    FAMILY_INTENT: """
# Intent decomposition
Parse FIRM (category, hardware spec, budget, stock) vs SOFT (preferred brand).
Search the spec first. First add_to_cart honors the preferred brand only when a
fully firm-compliant SKU exists. A later swap does not repair a bad first cart.
On a price jump: remove, re-search under the new budget, still prefer brand only
when fully compliant. Then inspect_stock and place_test_order.
""".strip(),
    FAMILY_RECALL: """
# Retrieval recall
Search the shopper phrase, then one short reformulation (brand+kind or category
filter). Shortlist several distinct in-stock products. Reject accessories using
category_path from view/compare. Pick the best in-budget hit and order promptly.
""".strip(),
    FAMILY_CONSTRAINT: """
# Constraint satisfaction
If a firm hardware spec is revealed only when asked, call message FIRST and wait.
Every firm requirement must hold on the exact SKU (options + price + stock).
Resolution names denote pixel classes from facts, not marketing titles:
HD, Full HD, QHD, 4K/UHD. If the wanted variant sells out, do not take a sibling
that misses a firm spec — find another product that satisfies all.
""".strip(),
    FAMILY_PREFERENCE: """
# Preference reasoning
Hard: category, budget, stock, real device. Soft: use-case axes.
Compare candidates on soft axes before the first cart. First cart must be a
genuine preference-fit, not the cheapest placeholder. Prefer a slightly dearer
in-budget item that matches soft goals.
""".strip(),
    FAMILY_RANK: """
# Ranking (TF5) — follow this sequence exactly
The shopper named a closed candidate set as product_id::sku pairs and a
strict tradeoff order (typical: highest advertised refresh including overclock,
then largest screen, then lowest price).

1. Compare or view EVERY listed product_id before any add_to_cart.
2. Rank only the named SKUs. Drop any option that is strictly worse on every
   stated axis. TOPSIS closeness is the primary score; the stated axis order
   breaks remaining ties.
3. FIRST add_to_cart MUST be the current best in-stock in-budget option.
4. After a stockout or price jump: remove_from_cart, re-rank survivors that are
   still in stock and in budget, inspect the new #1, cart it, then order.
5. Never pick a dominated option. Never invent a candidate outside the list.

CRITICAL — place_test_order on ranking:
- Pass ONLY product_id and sku.
- Do NOT attach justification or claims. Optional untruthful claims hard-fail
  even when the ranked pick is correct.
""".strip(),
    FAMILY_RECOVER: """
# Recovery (TF6) — least relaxation
Never relax: kind/category, budget, in-stock.
Prefer the exact model+color first. Model facts beat conflicting titles.

Search the color+model phrase, then the model code alone (the code may be absent
from the rendered title — still inspect those hits).

If the exact item disappears or the price breaks budget after a cart action:
1) same model, other color (prefer a SKU that shares the event-target stem)
2) only then other models
Re-check price and stock after every cart change. Inspect the replacement after
the event. Never order the event target.
""".strip(),
    FAMILY_JUSTIFY: """
# Justification (TF7) — evidence-backed claims
Search brand + spec tokens + kind, and/or filter brand+category.
View or compare raw facts BEFORE the first cart. Never justify from title words.
First cart must be a spec+brand match so later events stay on the intended item.
Never add_to_cart and place_test_order in the same turn.

place_test_order MUST include justification.claims from viewed / inspected facts:

Claim allowlist (any other field → faithful=False → reward 0):
- REQUIRED: price, in_stock, and exactly ONE shopper spec field using its
  canonical name from the goal (resolution OR screen_size OR refresh_rate).
- PLUS brand when the brand field is present on the viewed product.
- Optional one soft reason from streaming, microphone, privacy_cover,
  autofocus, hdr — ONLY if view/compare shows an explicit positive capability.
  NEVER invent hdr=true (or any feature) when unknown or absent.

Value rules:
- price / in_stock must match the latest inspect_stock or cart current state.
- resolution from facts.display / facts.resolution as hd|fhd|qhd|4k, not titles.
- screen_size like 27in from facts or variant options, not decoy titles.
- No summary prose, no evidence free text inside claims.
- Keep the claims array minimal.

Forbidden claim fields: display, size, category, category_path,
device_operating_system.
Reject screen-only / accessory SKUs when a full device is required.
""".strip(),
}

_IDLE = (
    "No tool call received. Choose a supplied tool now. "
    "If you have not searched yet, call search or filter using value-side keywords.",
    "Still no tool call. Progress: search/filter, view a candidate, answer any "
    "pending shopper question with message, or place_test_order if the cart "
    "already holds a verified in-stock SKU.",
)


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    return [_tool_label(tool) for tool in tools if _tool_label(tool)]


def _system_prompt(
    family: str,
    policy: dict[str, Any],
    tools: list[dict[str, Any]],
) -> str:
    labels = _tool_names(tools)
    book = _FAMILY_BOOK.get(family, _FAMILY_BOOK[FAMILY_RECALL])
    return (
        f"{_CONTRACT}\n\n"
        f"{book}\n\n"
        f"# Runtime\n"
        f"Steward family: {family}\n"
        f"Available tools: {', '.join(labels) if labels else '(see tool schemas)'}\n"
        f"max_steps={policy.get('max_steps')} "
        f"max_calls_per_turn={policy.get('max_calls_per_turn', 1)}\n"
        f"Emit at most {policy.get('max_calls_per_turn', 1)} tool call(s) per turn.\n"
    )


def _is_question(blob: str) -> bool:
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


def _spoken(message: dict[str, Any]) -> str:
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


def _shrink(payload: Any, cap: int = 7000) -> str:
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
    for candidate in _fallback_models(model):
        inference = _chat(
            model=candidate,
            messages=messages,
            tools=tools,
            tool_choice="required",
        )
        if inference is None:
            continue
        assistant = ((inference.get("choices") or [{}])[0].get("message")) or {}
        last_content = _spoken(assistant)
        raw_calls = list(assistant.get("tool_calls") or [])[:max_calls]
        if raw_calls:
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
                or f"{problem_id}-model-{turn}-{index}",
                "type": "function",
                "function": {
                    "name": action["name"],
                    "arguments": json.dumps(action.get("args") or {}),
                },
            }
        )
    return calls


class _Shelf:
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
                    {
                        field: value
                        for field, value in product.items()
                        if value is not None
                    }
                )
            if viewed:
                current["_seen"] = True
                pid = str(current.get("product_id") or "")
                if pid:
                    self.viewed.add(pid)
            if checked:
                current["_checked"] = True
                self.checked.add(key)
            for item in self.cart:
                if _listing_key(item) == key:
                    item.update(current)

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

    def fused(self) -> list[dict[str, Any]]:
        fused = _rrf(self.waves) if self.waves else []
        by_key = {
            _listing_key(item): item for item in fused if _listing_key(item)
        }
        by_key.update(self.rows)
        return list(by_key.values())


class _Board:
    def __init__(self, query: str, tools: list[dict[str, Any]], max_calls: int) -> None:
        self.query = query
        self.tools = tools
        self.max_calls = max(1, max_calls)
        self.family = classify_family(query)
        self.goal = _read_goal(query, self.family)
        self.shelf = _Shelf()
        self.search = _locate(tools, "search", "find_product", "search_catalog")
        self.filter = _locate(tools, "filter", "filter_catalog")
        self.view = _locate(tools, "view", "view_product", "get_product")
        self.compare = _locate(tools, "compare", "compare_products")
        self.inspect = _locate(tools, "inspect_stock", "check_stock")
        self.cart_view = _locate(tools, "inspect_cart", "view_cart")
        self.cart = _locate(tools, "add_to_cart", "cart_add")
        self.remove = _locate(tools, "remove_from_cart", "cart_remove")
        self.order = _locate(tools, "place_test_order", "place_order", "checkout")
        self.message = _locate(tools, "message", "talk", "respond")
        for spec in self.goal.get("named_keys") or []:
            key = f"{spec['product_id']}::{spec['sku']}"
            self.shelf.rows[key] = {
                "product_id": spec["product_id"],
                "sku": spec["sku"],
                "_seen": False,
            }

    def refine(self) -> None:
        prompt = (
            "Extract the shopper objective as JSON with keys: family "
            "(ranking|recovery|justification|intent_decomposition|retrieval_recall|"
            "constraint_satisfaction|preference_reasoning), budget (number or null), "
            "in_stock (bool), brand, category, model, color, named_keys "
            "([{product_id, sku}]), axes ([{name, direction desc|asc}]), "
            "need_reason (bool), spec_fields (array of claim field names), "
            "search_phrases (short catalog queries using VALUES not field names). "
            "Use ranking when the shopper names product_id::sku candidates and a "
            "tradeoff order. Use recovery when a preferred model/color may become "
            "unavailable. Use justification when the shopper asks for a reason. "
            "Do not invent product IDs. Axes must be generic attribute names such as "
            "refresh_rate, screen_size, or price.\n\n"
            f"Shopper:\n{_goal_text(self.query)}\n\n"
            f"Full public goal:\n{self.query}"
        )
        response = _chat(
            model=_role_model("scout"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only JSON. Prefer attributes stated in the shopper "
                        "text. Never copy catalog answers or product identifiers "
                        "that were not written in the goal."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        if response is None:
            return
        message = ((response.get("choices") or [{}])[0].get("message")) or {}
        parsed = _loads_json(_spoken(message)) or {}
        self.goal = _merge_goal(parsed, self.query, self.family)
        self.family = str(self.goal.get("family") or self.family)
        for spec in self.goal.get("named_keys") or []:
            key = f"{spec['product_id']}::{spec['sku']}"
            self.shelf.rows.setdefault(
                key,
                {
                    "product_id": spec["product_id"],
                    "sku": spec["sku"],
                    "_seen": False,
                },
            )

    def live(self, products: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        budget = self.goal.get("budget")
        rows = []
        for product in products if products is not None else self._pool():
            if self.goal.get("in_stock", True) and not _stocked(product):
                continue
            price = _price(product)
            if budget is not None and price is not None and price > float(budget):
                continue
            if not _keeps(product, self.goal, strict=bool(product.get("_seen"))):
                continue
            rows.append(product)
        return rows

    def _pool(self) -> list[dict[str, Any]]:
        wanted = self.goal.get("named_keys") or []
        fused = self.shelf.fused()
        if not wanted:
            return fused
        by_key = {_listing_key(item): item for item in fused}
        rows: list[dict[str, Any]] = []
        for spec in wanted:
            key = f"{spec['product_id']}::{spec['sku']}"
            product = by_key.get(key)
            if product is None:
                product = {
                    "product_id": spec["product_id"],
                    "sku": spec["sku"],
                    "_seen": False,
                }
            rows.append(product)
        return rows

    def order_rows(self) -> list[dict[str, Any]]:
        pool = self.live()
        axes = list(self.goal.get("axes") or [])
        if axes:
            ranked = _topsis(pool, axes)
            return _lex_break(ranked, axes) if ranked else ranked
        if self.goal.get("family") == FAMILY_RECOVER and self.goal.get("model"):
            target = self.shelf.event_key or ""
            stem = _sku_stem(target.split("::")[-1] if "::" in target else "")
            scored = [
                (
                    _relax_cost(
                        product,
                        str(self.goal.get("model") or ""),
                        str(self.goal.get("color") or ""),
                        stem,
                    ),
                    product,
                )
                for product in pool
            ]
            scored.sort(key=lambda item: item[0])
            keep = [product for cost, product in scored if cost < 2.0]
            return keep or [product for _cost, product in scored]
        query_tf = Counter(
            _words(
                " ".join(self.goal.get("search_phrases") or [])
                + " "
                + _goal_text(self.query)
            )
        )
        docs = [(_words(_doc(product)), product) for product in pool]
        df: Counter[str] = Counter()
        for tokens, _product in docs:
            df.update(set(tokens))
        avgdl = sum(len(tokens) for tokens, _product in docs) / max(len(docs), 1)
        brand = _fold(self.goal.get("brand") or "")
        scored: list[tuple[tuple[int, float], dict[str, Any]]] = []
        raw_scores: list[float] = []
        rows: list[dict[str, Any]] = []
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
            raw_scores.append(_COMB_BM25 * bm25 + _COMB_JAC * overlap)
            rows.append(product)
        peak = max(raw_scores) if raw_scores else 1.0
        for product, raw in zip(rows, raw_scores):
            brand_rank = 0 if brand and _fold(product.get("brand") or "") == brand else 1
            score = raw / peak if peak else 0.0
            scored.append(((brand_rank, -score), product))
        scored.sort(key=lambda item: item[0])
        return [product for _score, product in scored]

    def winner(self) -> dict[str, Any] | None:
        ranked = self.order_rows()
        if not ranked:
            return None
        target = self.shelf.event_key
        live = [item for item in ranked if _listing_key(item) != target]
        return (live or ranked)[0]

    def stale(self, product: dict[str, Any] | None) -> bool:
        if product is None:
            return False
        if not _stocked(product):
            return True
        budget = self.goal.get("budget")
        price = _price(product)
        if budget is not None and price is not None and price > float(budget):
            return True
        if self.shelf.event_key and _listing_key(product) == self.shelf.event_key:
            return True
        return False

    def brief(self, proposed: list[dict[str, Any]]) -> str:
        ranked = self.order_rows()[:6]
        rows = []
        for product in ranked:
            rows.append(
                {
                    "key": _listing_key(product),
                    "title": product.get("title"),
                    "brand": product.get("brand"),
                    "price": product.get("price"),
                    "in_stock": _stocked(product),
                    "seen": bool(product.get("_seen")),
                    "checked": bool(product.get("_checked")),
                    "refresh_hz": _hertz(product),
                    "screen_in": _screen(product),
                    "relax_cost": _relax_cost(
                        product,
                        str(self.goal.get("model") or ""),
                        str(self.goal.get("color") or ""),
                        _sku_stem(
                            (self.shelf.event_key or "").split("::")[-1]
                        ),
                    ),
                }
            )
        return (
            "Scoreboard (follow unless a later observation contradicts it):\n"
            f"family={self.goal.get('family')} budget={self.goal.get('budget')} "
            f"model={self.goal.get('model')} color={self.goal.get('color')} "
            f"axes={self.goal.get('axes')} event={self.shelf.event_kind} "
            f"event_target={self.shelf.event_key}\n"
            f"ranked={json.dumps(rows, sort_keys=True)}\n"
            f"next_actions={json.dumps(proposed, sort_keys=True)}"
        )

    def _search_actions(self) -> list[dict[str, Any]]:
        phrases = list(self.goal.get("search_phrases") or ["in stock"])
        if self.goal.get("model"):
            color = self.goal.get("color") or ""
            model = str(self.goal.get("model"))
            phrases = [
                " ".join(part for part in (color, model) if part),
                model,
            ]
            if self.shelf.event_key:
                phrases = [model, " ".join(part for part in (color, model) if part)]
        actions: list[dict[str, Any]] = []
        while self.shelf.search_n < max(len(phrases), 1) and len(actions) < self.max_calls:
            query = phrases[min(self.shelf.search_n, len(phrases) - 1)]
            self.shelf.search_n += 1
            if self.search:
                actions.append(
                    {
                        "name": _tool_label(self.search),
                        "args": _bind(
                            self.search,
                            {"query": query, "k": 10, "in_stock": True},
                        ),
                    }
                )
            else:
                break
        brand = str(self.goal.get("brand") or "")
        if _fold(brand) in _STOP_BRAND:
            brand = ""
        category = str(self.goal.get("category") or "")
        if (
            self.filter
            and not self.shelf.did_filter
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
                self.shelf.did_filter = True
                actions.append({"name": _tool_label(self.filter), "args": args})
        if not actions:
            self.shelf.search_n += 1
        return actions[: self.max_calls]

    def _needs_recovery_search(self, identity: list[dict[str, Any]]) -> bool:
        if self.goal.get("family") != FAMILY_RECOVER or not self.goal.get("model"):
            return False
        live = [
            product
            for product in identity
            if _listing_key(product) != self.shelf.event_key
        ]
        if not live:
            return True
        costs = [
            _relax_cost(
                product,
                str(self.goal.get("model") or ""),
                str(self.goal.get("color") or ""),
                _sku_stem((self.shelf.event_key or "").split("::")[-1]),
            )
            for product in live
        ]
        if self.shelf.event_key:
            return not any(cost < 2.0 for cost in costs)
        return False

    def _checkout(self, product: dict[str, Any]) -> dict[str, Any]:
        if self.order is None:
            return {"name": "", "args": {}}
        wanted: dict[str, Any] = {
            "product_id": product.get("product_id"),
            "sku": product.get("sku"),
        }
        ranking = bool(self.goal.get("named_keys")) or self.goal.get("family") == FAMILY_RANK
        needs = bool(self.goal.get("need_reason")) and not ranking
        if ranking:
            needs = False
        if needs and self.order is not None:
            allowed = _claim_fields(self.order)
            fields = list(self.goal.get("spec_fields") or [])
            for field in _needed_fields(_goal_text(self.query)):
                if field not in fields:
                    fields.append(field)
            if not fields:
                fields = ["in_stock", "price"]
            brand = _trim(product.get("brand")) or _trim(_facts(product).get("brand"))
            if brand:
                fields = ["brand"] + [field for field in fields if field != "brand"]
            seen: list[str] = []
            for field in fields:
                if field not in seen and field not in _CLAIM_BLOCK:
                    seen.append(field)
            claims = _compile_claims(product, seen, allowed)
            if claims:
                wanted["justification"] = {"claims": claims}
        return {
            "name": _tool_label(self.order),
            "args": _bind(self.order, wanted),
        }

    def next_actions(self) -> list[dict[str, Any]]:
        cart = self.shelf.cart[0] if self.shelf.cart else None
        if cart is not None and self.stale(cart) and self.remove:
            return [
                {
                    "name": _tool_label(self.remove),
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
                if spec["product_id"] not in self.shelf.viewed
            ]
            if missing:
                return [
                    {
                        "name": _tool_label(self.compare),
                        "args": _bind(
                            self.compare,
                            {"product_ids": list(dict.fromkeys(missing))[:5]},
                        ),
                    }
                ]
        identity = self.live()
        if (not identity or self._needs_recovery_search(identity)) and (
            self.search or self.filter
        ):
            if self.shelf.search_n < 8:
                searched = self._search_actions()
                if searched:
                    return searched
        if self.compare:
            missing = []
            for product in identity or self.order_rows():
                pid = str(product.get("product_id") or "")
                if pid and pid not in self.shelf.viewed and pid not in missing:
                    missing.append(pid)
                if len(missing) >= 5:
                    break
            if missing:
                return [
                    {
                        "name": _tool_label(self.compare),
                        "args": _bind(self.compare, {"product_ids": missing}),
                    }
                ]
        picked = self.winner()
        if picked is None:
            if self.search and self.shelf.search_n < 8:
                return self._search_actions()
            return []
        pid = str(picked.get("product_id") or "")
        if not picked.get("_seen") and pid not in self.shelf.viewed:
            if self.compare:
                return [
                    {
                        "name": _tool_label(self.compare),
                        "args": _bind(self.compare, {"product_ids": [pid]}),
                    }
                ]
            if self.view:
                return [
                    {
                        "name": _tool_label(self.view),
                        "args": _bind(self.view, {"product_id": pid}),
                    }
                ]
        key = _listing_key(picked)
        if self.inspect and key not in self.shelf.checked:
            return [
                {
                    "name": _tool_label(self.inspect),
                    "args": _bind(
                        self.inspect,
                        {"product_id": pid, "sku": picked.get("sku")},
                    ),
                }
            ]
        if cart is None and self.cart and not self.stale(picked):
            if self.goal.get("family") == FAMILY_JUSTIFY and not _spec_ok(
                picked, self.goal
            ):
                if self.search and self.shelf.search_n < 8:
                    return self._search_actions()
                return []
            return [
                {
                    "name": _tool_label(self.cart),
                    "args": _bind(
                        self.cart,
                        {"product_id": pid, "sku": picked.get("sku")},
                    ),
                }
            ]
        if cart is not None and not self.stale(cart) and self.order:
            return [self._checkout(cart)]
        if picked is not None and self.order and not self.stale(picked):
            return [self._checkout(picked)]
        return []

    def rebind(
        self,
        actions: list[dict[str, Any]],
        proposed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        names = {_tool_label(tool).lower() for tool in self.tools}
        cleaned = [
            action
            for action in actions
            if str(action.get("name") or "").lower() in names
        ]
        if not cleaned:
            return proposed[: self.max_calls]
        focus = self.goal.get("family") in {
            FAMILY_RANK,
            FAMILY_RECOVER,
            FAMILY_JUSTIFY,
        }
        if not focus:
            return cleaned[: self.max_calls]
        order_name = _tool_label(self.order).lower()
        cart_name = _tool_label(self.cart).lower()
        winner = self.winner()
        repaired: list[dict[str, Any]] = []
        for action in cleaned[: self.max_calls]:
            name = str(action.get("name") or "")
            args = dict(action.get("args") or {})
            lower = name.lower()
            if lower == cart_name and winner is not None:
                picked_key = f"{args.get('product_id')}::{args.get('sku')}"
                win_key = _listing_key(winner)
                ready = bool(winner.get("_checked")) or any(
                    str(item.get("name") or "").lower() == cart_name
                    for item in proposed
                )
                if picked_key != win_key and not self.stale(winner) and ready:
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
                    product = self.shelf.rows.get(
                        f"{args.get('product_id')}::{args.get('sku')}"
                    ) or product
                if (
                    winner is not None
                    and not self.stale(winner)
                    and winner.get("_checked")
                ):
                    product = winner
                if product is not None:
                    repaired.append(self._checkout(product))
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
            inner = _inner(observation if isinstance(observation, dict) else {})
            name = ""
            if index < len(sent):
                name = str(sent[index].get("name") or "").lower()
            viewed = name in {
                _tool_label(self.view).lower(),
                _tool_label(self.compare).lower(),
            }
            checked = name == _tool_label(self.inspect).lower()
            products = _collect(inner)
            if products:
                query = ""
                if index < len(sent):
                    query = str((sent[index].get("args") or {}).get("query") or "")
                if query and name in {
                    _tool_label(self.search).lower(),
                    "search",
                    "find_product",
                    "search_catalog",
                }:
                    stamped = []
                    for product in products:
                        row = dict(product)
                        row["_query"] = query
                        stamped.append(row)
                    products = stamped
                self.shelf.put(
                    products,
                    viewed=viewed,
                    checked=checked,
                    from_search=not viewed and not checked,
                )
            if isinstance(inner.get("added"), dict):
                self.shelf.put(
                    [inner["added"]],
                    viewed=False,
                    checked=True,
                    from_search=False,
                )
                self.shelf.cart = [dict(inner["added"])]
            if isinstance(inner.get("cart"), list):
                self.shelf.cart = [
                    item for item in inner["cart"] if isinstance(item, dict)
                ]
            event = inner.get("market_event")
            if isinstance(event, dict) and event:
                self.shelf.mark_event(event)
        message = result.get("user_message") or {}
        if isinstance(message, dict):
            signal = message.get("env_signal") or {}
            if isinstance(signal, dict) and signal.get("kind"):
                self.shelf.mark_event(signal)
            content = str(message.get("content") or "")
            budget = _CEILING.search(content)
            if budget:
                self.goal["budget"] = float(budget.group(1))
        return done


def _coach(
    family: str,
    turn: int,
    board: _Board,
    proposed: list[dict[str, Any]],
    await_reply: bool,
    cart_used: bool,
    prior_names: list[str],
) -> str | None:
    blob = " ".join(name.lower() for name in prior_names)
    if await_reply:
        return (
            "Coach: the shopper just sent a message. Answer it with the message "
            "tool in this turn before cart or order actions."
        )
    if family == FAMILY_CONSTRAINT and turn == 1 and board.goal.get("ask_spec"):
        return (
            "Coach: one firm hardware spec is revealed only if asked. Call message "
            "NOW to ask for that exact requirement before searching."
        )
    if family == FAMILY_RANK:
        if turn == 1 and not cart_used:
            return (
                "Coach [TF5]: compare every listed product_id::sku candidate and "
                "rank by the stated axes BEFORE the first add_to_cart. First cart "
                "is the current TOPSIS #1. On place_test_order pass ONLY "
                "product_id and sku — omit justification entirely."
            )
        if not cart_used and turn >= 2 and ("view" in blob or "compare" in blob):
            return (
                "Coach [TF5]: candidates are inspected. add_to_cart ONLY the "
                "scoreboard #1 now (first cart is graded)."
            )
        if cart_used:
            return (
                "Coach [TF5]: after recovery, place_test_order with ONLY product_id "
                "and sku — do not attach justification.claims."
            )
    if family == FAMILY_JUSTIFY:
        if not cart_used:
            return (
                "Coach [TF7]: view/compare raw facts before the first cart. "
                "Claims later: in_stock, live price, one canonical spec field, "
                "plus brand when observed. Forbidden: display, size, category, "
                "invented hdr."
            )
        return (
            "Coach [TF7]: inspect_stock the cart SKU, then place_test_order with "
            "minimal grounded claims. Never invent hdr or use title decoys."
        )
    if family == FAMILY_RECOVER and turn == 1:
        return (
            "Coach [TF6]: search color+model then model. After a stockout, keep "
            "the same model and change color first. Never order the event target."
        )
    if family == FAMILY_INTENT and turn == 1:
        return (
            "Coach: first cart honors the preferred brand only when that SKU also "
            "meets the firm spec, budget, and stock."
        )
    if family == FAMILY_PREFERENCE and turn <= 2 and not cart_used:
        return (
            "Coach: compare soft-preference axes before the first cart. First cart "
            "must be a real preference-fit purchase."
        )
    if proposed:
        return board.brief(proposed)
    if turn <= 2 and not any(
        name in blob for name in ("search", "filter", "view", "compare", "message")
    ):
        return (
            "Hint: start with search or filter using value-side keywords from the "
            "shopper goal, then view the best hits."
        )
    return None


def _talk(board: _Board) -> list[dict[str, Any]]:
    if board.message is None:
        return []
    return [
        {
            "name": _tool_label(board.message),
            "args": _bind(
                board.message,
                {"content": "I will check the live listing and confirm the fit."},
            ),
        }
    ]


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Steward the environment tools until done=true; scoreboard owns TF5-TF7."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
    tools = list(policy.get("tools") or [])
    query = str(policy.get("query") or problem_data.get("query") or "")
    board = _Board(query, tools, max_calls)
    board.refine()
    family = str(board.goal.get("family") or board.family)
    board.family = family
    model = _family_model(family)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(family, policy, tools)},
        {"role": "user", "content": query},
    ]
    if family == FAMILY_CONSTRAINT and board.goal.get("ask_spec"):
        messages.append(
            {
                "role": "user",
                "content": (
                    "Coach: ask for the hidden firm hardware spec with the message "
                    "tool before any search."
                ),
            }
        )
    if family == FAMILY_RANK:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Coach [TF5]: inspect every named candidate, cart the TOPSIS "
                    "winner, and place_test_order with ONLY product_id and sku."
                ),
            }
        )
    if family == FAMILY_JUSTIFY:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Coach [TF7]: view facts before the first cart. Claims allowlist: "
                    "in_stock, price, one spec field, plus brand when observed. "
                    "Forbidden: display, size, category, invented hdr."
                ),
            }
        )
    if family == FAMILY_RECOVER:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Coach [TF6]: search the preferred color+model first. After the "
                    "exact SKU disappears, same model other color, then other models."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "horizon": {
                "family": family,
                "model": model,
                "provider": _provider(),
            },
        }
    ]
    await_reply = False
    cart_used = False
    prior_names: list[str] = []

    for turn in range(1, int(policy.get("max_steps") or 20) + 1):
        proposed = board.next_actions()
        coach = _coach(
            family, turn, board, proposed, await_reply, cart_used, prior_names
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
        if await_reply:
            names = {str(item.get("name") or "").lower() for item in actions}
            if "message" not in names:
                forced = _talk(board)
                if forced:
                    actions = forced
        else:
            actions = board.rebind(actions, proposed)
        if not actions:
            raise RuntimeError("model returned no tool call before completion")

        prior_names = [str(item.get("name") or "") for item in actions]
        lowered_names = [name.lower() for name in prior_names]
        if any("add_to_cart" in name for name in lowered_names):
            cart_used = True
        if any("message" in name for name in lowered_names):
            await_reply = False

        tool_calls = _as_tool_calls(actions, raw_calls, problem_id, turn)
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
        result = _proxy.post("/environment/call", json_data=envelope)
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
                    "content": _shrink(call_result.get("observation")),
                }
            )
        shopper_msg = result.get("user_message") or {}
        if isinstance(shopper_msg, dict) and shopper_msg.get("content"):
            spoken = str(shopper_msg["content"])
            messages.append({"role": "user", "content": spoken})
            if _is_question(spoken):
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
        if faults and family == FAMILY_RECOVER:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Coach [TF6]: action failed or stock/price changed "
                        f"({faults[0]}). Recover with least relaxation that still "
                        "satisfies kind, budget, and in-stock. Never order the "
                        "event target."
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
                            "recover before ordering. Never order the event target."
                        ),
                    }
                )
                break
        if board.ingest(result, sent):
            break

    return dialogue


__all__ = ["agent_main", "classify_family"]
