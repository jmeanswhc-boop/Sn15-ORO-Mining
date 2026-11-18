
from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_OR_MODELS = [
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen3.6-27b",
]
_CH_MODELS = [
    "moonshotai/Kimi-K2.6-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
    "Qwen/Qwen3.6-27B-TEE",
]

_CLAIM_OK = frozenset(
    {
        "brand",
        "category",
        "in_stock",
        "price",
        "refresh_rate",
        "resolution",
        "screen_size",
    }
)
_CLAIM_BLOCK = frozenset({"category"})
_ALIAS = {
    "display": "resolution",
    "screen": "screen_size",
    "size": "screen_size",
    "screen size": "screen_size",
    "refresh": "refresh_rate",
    "hz": "refresh_rate",
    "category_path": "category",
}

_VARIANT_RE = re.compile(r"(\d{6,})::([A-Za-z0-9][A-Za-z0-9._-]*)")
_ASK_RE = re.compile(
    r"tell you if you ask|if you ask|ask me.*firm|firm requirement.*ask",
    re.I,
)
_COLOR_RE = (
    r"black|white|green|red|blue|gold|silver|pink|purple|gray|grey|"
    r"yellow|orange|moonlight\s+white"
)

_proxy = ProxyClient(timeout=120, max_retries=3)

_VOICE = """You are an EnvPack shopping agent. Use only supplied tools. Stop when
done=true. Obey THIS episode's shopper text + Task rules. Never memorize catalog IDs.

## Intent
Silently split into ≤4 aspects (type, firm budget/specs, soft prefs).
Shopper question / budget change → one `message`, then resume.
If a firm spec is withheld until you ask → first tool is `message`.

## Retrieve
search/filter with short reformulations; view several distinct product_ids.
After ~3–4 weak searches, stop thrashing: view best hits and finish.
Prefer in_stock when availability matters.

## Constraints / prefs
Firm specs are non-negotiable (category_path, size/resolution/refresh, budget, stock).
Match numbers and brand/model tokens literally after view.
Soft brand only after every firm check passes; among valid picks, prefer better soft fit.

## Closed ranking
If THIS text lists product_id::sku keys (≥2 or compare-only framing): that list is
the whole catalog. view each id; rank by stated priority; cart/order a listed key only.

## Recovery
On stock/price shock or market_event: COMMIT same preferred model / other color
(never ask permission) → remove_from_cart → view replacement → inspect_stock →
cart → place_test_order. Never order out-of-stock. Never quit.

## Justification
Only if shopper asked why/reason. Claims from latest view + inspect only.
Always ground price/in_stock; include brand when view has it; never claim category.
Omit justification when not asked.

## Finish
search → view → inspect_stock → add_to_cart → place_test_order.
When steps are tight, stop exploring and order a live in-budget sku.
"""


def _backend() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw in ("chutes", "openrouter"):
        return raw
    if getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "chutes"


def _models() -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    return list(_OR_MODELS if _backend() == "openrouter" else _CH_MODELS)


def _head(query: str) -> str:
    return query.split("Task rules:", 1)[0].strip()


def _closed_variants(query: str) -> list[tuple[str, str]]:
    head = _head(query)
    low = head.lower()
    pairs = _VARIANT_RE.findall(head)
    framed = any(
        s in low
        for s in (
            "compare only",
            "candidate variant",
            "candidate variants",
            "these candidate",
            "only these",
        )
    )
    if not framed and len(pairs) < 2:
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pid, sku in pairs:
        sku = sku.rstrip(".,;:)")
        key = f"{pid}::{sku}"
        if key in seen:
            continue
        seen.add(key)
        out.append((pid, sku))
    return out


def _wants_why(query: str) -> bool:
    low = _head(query).lower()
    return any(
        n in low
        for n in ("why", "reason", "walk me through", "tell me why", "justif", "good pick")
    )


def _force_ask(query: str) -> bool:
    return bool(_ASK_RE.search(_head(query)))


def _soft_brand(query: str) -> str | None:
    """Soft preferred brand from ideally/prefer phrasing (intent TF)."""
    head = _head(query)
    m = re.search(
        r"(?i)\bideally\b[^.]{0,60}\bprefer\s+([A-Za-z][A-Za-z0-9]*(?:\s+[A-Za-z][A-Za-z0-9]*){0,3})",
        head,
    )
    if not m:
        return None
    brand = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".,;:")
    if brand.lower() in {"a", "an", "the", "something", "one"}:
        return None
    return brand


def _soft_wish_tokens(query: str) -> list[str]:
    """Use-case soft tokens for preference tasks (sweat/water/gym/…)."""
    head = _head(query).lower()
    cues = (
        "sweat",
        "waterproof",
        "water-resistant",
        "water resistant",
        "water",
        "ipx",
        "gym",
        "workout",
        "bootcamp",
        "outdoor",
        "running",
        "battery",
        "noise",
        "wireless",
        "bluetooth",
    )
    found: list[str] = []
    for c in cues:
        if " " in c or "-" in c:
            if c in head:
                found.append(c)
        elif re.search(rf"\b{re.escape(c)}\b", head):
            found.append(c)
    # Avoid double-counting water when waterproof already matched.
    if any(x.startswith("water") and x != "water" for x in found):
        found = [x for x in found if x != "water"]
    return found


def _sku_family(sku: str) -> str:
    s = str(sku or "").strip()
    if not s:
        return ""
    if "-" in s:
        return s.rsplit("-", 1)[0]
    return s


def _num(text: str | None) -> float | None:
    if text is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(text).replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _rank_key_from_facts(facts: dict[str, str]) -> tuple[float, float, float]:
    """Higher refresh, larger screen, lower price — race-general TF5 order."""
    hz = _num(facts.get("refresh_rate"))
    inches = _num(facts.get("screen_size"))
    price = _num(facts.get("price"))
    return (
        hz if hz is not None else -1.0,
        inches if inches is not None else -1.0,
        -(price) if price is not None else -1e18,
    )


def _best_closed_pick(
    candidates: list[tuple[str, str]],
    *,
    facts_by_pid: dict[str, dict[str, str]],
    titles: dict[str, str],
) -> tuple[str, str] | None:
    if not candidates:
        return None
    scored: list[tuple[tuple[float, float, float], str, str]] = []
    for pid, sku in candidates:
        facts = dict(facts_by_pid.get(pid) or {})
        if not facts.get("screen_size") and titles.get(pid):
            m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", titles[pid], re.I)
            if m:
                facts["screen_size"] = m.group(1)
        if not facts.get("refresh_rate") and titles.get(pid):
            m = re.search(r"(\d+)\s*hz", titles[pid], re.I)
            if m:
                facts["refresh_rate"] = m.group(1)
        scored.append((_rank_key_from_facts(facts), pid, sku))
    scored.sort(reverse=True)
    return scored[0][1], scored[0][2]


def _retrieval_phrase(query: str) -> str | None:
    """Title/phrase match from 'find … that matches …' retrieval asks."""
    head = _head(query)
    m = re.search(
        r"matches\s+(?:the\s+phrase\s+)?[\"']([^\"']+)[\"']",
        head,
        re.I,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip() or None
    m = re.search(
        r"matches\s+(?:the\s+phrase\s+)?([A-Za-z0-9][\w\s./-]{0,80}?)"
        r"(?=\s*,|\s+at\s+or|\s+and\s+buy)",
        head,
        re.I,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip() or None
    return None


def _retrieval_category(query: str) -> str | None:
    m = re.search(r"\bin the\s+(.+?)\s+category\b", _head(query), re.I)
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(1)).strip() or None


def _named_specs(query: str) -> list[str]:
    low = _head(query).lower()
    fields: list[str] = []
    if any(
        x in low
        for x in ("720p", "1080p", "1440p", "4k", "uhd", "qhd", "full hd", " resolution")
    ) or re.search(r"\bhd\b", low):
        fields.append("resolution")
    if any(x in low for x in ("inch", "screen", "display size")):
        fields.append("screen_size")
    if "refresh" in low or re.search(r"\b\d+\s*hz\b", low):
        fields.append("refresh_rate")
    return fields


def _schema_claims(tools: list[dict[str, Any]]) -> frozenset[str]:
    for tool in tools:
        fn = tool.get("function") or tool
        if (fn.get("name") or "") != "place_test_order":
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        field = (
            ((props.get("justification") or {}).get("properties") or {})
            .get("claims", {})
            .get("items", {})
            .get("properties", {})
            .get("field", {})
        )
        enum = field.get("enum")
        if isinstance(enum, list) and enum:
            return frozenset(str(x) for x in enum)
    return _CLAIM_OK


def _map_field(field: str, allowed: frozenset[str]) -> str | None:
    raw = str(field or "").strip()
    if not raw:
        return None
    key = raw.lower()
    if key in _CLAIM_BLOCK or raw in _CLAIM_BLOCK:
        return None
    if raw in allowed:
        return raw
    if key in allowed:
        return key
    mapped = _ALIAS.get(key)
    if mapped and mapped in allowed and mapped not in _CLAIM_BLOCK:
        return mapped
    return None


def _tidy_value(field: str, value: Any) -> str:
    text = str(value).strip()
    if field == "in_stock":
        low = text.lower()
        if low in ("true", "1", "yes", "in stock", "instock"):
            return "true"
        if low in ("false", "0", "no", "out of stock"):
            return "false"
    if field == "price":
        cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
        if cleaned:
            try:
                return str(float(cleaned))
            except ValueError:
                return text
    if field == "refresh_rate":
        m = re.search(r"(\d+(?:\.\d+)?)\s*hz", text, flags=re.I)
        if m:
            return f"{m.group(1)}Hz"
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            return f"{text}Hz"
    if field == "screen_size":
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            return f"{m.group(1)} inches"
    if field == "resolution":
        low = text.lower().replace("×", "x")
        if low in ("hd", "fhd", "qhd", "uhd", "4k"):
            return "uhd" if low == "4k" else low
        return text
    return text


def _harvest_view(observation: Any) -> tuple[str | None, dict[str, str], str]:
    """Return (product_id, facts, title) from a view observation."""
    facts: dict[str, str] = {}
    title = ""
    pid: str | None = None
    if not isinstance(observation, dict):
        return None, facts, title
    node = observation.get("observation")
    if not isinstance(node, dict):
        node = observation
    if node.get("product_id") not in (None, ""):
        pid = str(node["product_id"])
    title = str(node.get("title") or "")

    def walk(obj: Any, depth: int = 0) -> None:
        if depth > 6 or obj is None:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in ("brand", "resolution", "screen_size", "refresh_rate", "display", "size") and v not in (
                    None,
                    "",
                    "None",
                ):
                    field = lk
                    if lk == "display":
                        field = "resolution"
                    if lk == "size":
                        field = "screen_size"
                    facts.setdefault(field, str(v).strip())
                elif isinstance(v, (dict, list)):
                    walk(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj[:20]:
                walk(item, depth + 1)

    walk(node)
    nested = node.get("facts")
    if isinstance(nested, dict):
        if nested.get("display") not in (None, ""):
            facts.setdefault("resolution", str(nested["display"]).strip())
        if nested.get("resolution") not in (None, ""):
            facts.setdefault("resolution", str(nested["resolution"]).strip())
        if nested.get("size") not in (None, ""):
            facts.setdefault("screen_size", str(nested["size"]).strip())
        if nested.get("refresh") not in (None, "") or nested.get("refresh_rate") not in (
            None,
            "",
        ):
            facts.setdefault(
                "refresh_rate",
                str(nested.get("refresh") or nested.get("refresh_rate")).strip(),
            )
    attrs = node.get("attributes") or node.get("specs")
    if isinstance(attrs, list):
        for a in attrs:
            if not isinstance(a, dict):
                continue
            name = str(a.get("name") or a.get("key") or "").lower()
            val = a.get("value") or a.get("val")
            if not val:
                continue
            if "brand" in name:
                facts.setdefault("brand", str(val).strip())
            if "resolution" in name or name in ("display", "screen resolution"):
                facts.setdefault("resolution", str(val).strip())
            if "screen" in name and "size" in name:
                facts.setdefault("screen_size", str(val).strip())
            if "refresh" in name:
                facts.setdefault("refresh_rate", str(val).strip())
    if node.get("brand") not in (None, "", "None"):
        facts["brand"] = str(node["brand"]).strip()
    return pid, facts, title


def _firm_brand(query: str) -> str | None:
    """Required brand only when THIS text names Brand before a product/spec token.

    Must NOT match 'I need a smartphone at or below …' (category ≠ brand).
    """
    head = _head(query)
    if re.search(r"(?i)\bideally\b", head):
        return None
    m = re.search(
        r"(?i)(?:i\s+want|i\s+need|find\s+me|get\s+me|can you get me)\s+"
        r"(?:an?\s+)?(?:in-stock\s+)?"
        r"(?-i:([A-Za-z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,2}))\s+"
        r"(?:720p|1080p|1440p|2160p|4k|full\s*hd|\bhd\b|\d+(?:\.\d+)?-?\s*inch(?:es)?"
        r"|smartphone|tablet|monitor|headphones?|earbuds?|laptop|gaming)\b",
        head,
    )
    if not m:
        return None
    brand = re.sub(r"\s+", " ", m.group(1)).strip()
    blocked = {
        "the",
        "new",
        "good",
        "best",
        "cheap",
        "in",
        "an",
        "a",
        "stock",
        "in-stock",
        "smartphone",
        "tablet",
        "monitor",
        "headphone",
        "headphones",
        "earbuds",
        "laptop",
        "gaming",
        "at",
        "or",
        "below",
        "under",
    }
    if brand.lower() in blocked or any(w in blocked for w in brand.lower().split()):
        return None
    return brand


def _pref_color(query: str) -> str | None:
    head = _head(query)
    m = re.search(
        rf"(?:strongly\s+)?prefer(?:\s+the)?\s+({_COLOR_RE})\s+",
        head,
        flags=re.I,
    )
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(1)).strip().lower()


def _listing_mentions(blob: str, token: str) -> bool:
    if not token:
        return True
    return token.lower() in (blob or "").lower()


def _absorb_facts(obj: Any, bag: dict[str, str], depth: int = 0) -> None:
    if depth > 6 or obj is None:
        return
    if isinstance(obj, dict):
        key_map = {
            "brand": "brand",
            "price": "price",
            "current_price": "price",
            "in_stock": "in_stock",
            "stock": "in_stock",
            "refresh_rate": "refresh_rate",
            "refresh": "refresh_rate",
            "resolution": "resolution",
            "screen_size": "screen_size",
            "screen size": "screen_size",
        }
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in key_map and not isinstance(v, (dict, list)):
                field = key_map[lk]
                if field not in _CLAIM_BLOCK and v not in (None, "", "None"):
                    bag.setdefault(field, str(v).strip())
            else:
                _absorb_facts(v, bag, depth + 1)
    elif isinstance(obj, list):
        for item in obj[:20]:
            _absorb_facts(item, bag, depth + 1)


def _model_token(query: str) -> str | None:
    head = _head(query)
    patterns = (
        rf"prefer(?:\s+the)?\s+(?:{_COLOR_RE})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"eye on the\s+(?:{_COLOR_RE})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"you have your eye on the\s+(?:{_COLOR_RE})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"strongly\s+prefer(?:\s+the)?\s+(?:{_COLOR_RE})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
    )
    token = None
    for pat in patterns:
        m = re.search(pat, head, flags=re.I)
        if m:
            token = m.group(1).strip()
            break
    if not token:
        return None
    token = token.split(".")[0]
    token = re.split(
        r"\s+\bIf\b|\s+\bspecifically\b|\s+\bgenuine\b|\s+\bnot\b|,",
        token,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    token = token.strip(".,;:()[]\"'")
    if token.lower() in {
        "black",
        "white",
        "green",
        "red",
        "blue",
        "gold",
        "silver",
        "pink",
        "purple",
        "gray",
        "grey",
        "yellow",
        "orange",
        "the",
        "a",
        "an",
    }:
        return None
    return token or None


def _commit_text(model: str | None) -> str:
    if model:
        return (
            f"The preferred option went out of stock when I added it to cart. "
            f"Following your rules, I will now take the same model ({model}) in "
            f"another color/variant that is still in stock, keep budget and category "
            f"firm, and only change models if no same-model variant remains."
        )
    return (
        "That item went out of stock when I added it to cart. "
        "I'm now selecting another in-stock option that still matches your "
        "firm budget and category requirements."
    )


def _scrub_order(
    args: dict[str, Any],
    *,
    query: str,
    allowed: frozenset[str],
    stock: dict[str, Any] | None,
    facts_by_pid: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Build claims only from the ordered product's view + latest stock."""
    out = dict(args)
    if not _wants_why(query):
        out.pop("justification", None)
        return out
    pid = str(out.get("product_id") or "")
    view = dict(facts_by_pid.get(pid) or {})
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(field: str, raw: Any) -> None:
        if field not in allowed or field in _CLAIM_BLOCK or field in seen:
            return
        value = _tidy_value(field, raw)
        if not value:
            return
        if field == "resolution":
            # Always prefer the exact viewed display string for this pid.
            if view.get("resolution"):
                value = str(view["resolution"]).strip()
            elif value.lower() in ("hd", "fhd", "qhd", "uhd"):
                return
        cleaned.append({"field": field, "value": value})
        seen.add(field)

    if stock:
        if stock.get("price") is not None:
            add("price", stock.get("price"))
        if stock.get("in_stock") is not None:
            add("in_stock", stock.get("in_stock"))
    if view.get("brand"):
        add("brand", view["brand"])
    for spec in _named_specs(query):
        if view.get(spec):
            add(spec, view[spec])

    if cleaned:
        out["justification"] = {"claims": cleaned}
    else:
        out.pop("justification", None)
    return out


def _parse_args(
    tool_call: dict[str, Any],
    *,
    query: str,
    allowed: frozenset[str],
    stock: dict[str, Any] | None,
    facts_by_pid: dict[str, dict[str, str]],
) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    if (tool_call["function"].get("name") or "") == "place_test_order":
        return _scrub_order(
            parsed,
            query=query,
            allowed=allowed,
            stock=stock,
            facts_by_pid=facts_by_pid,
        )
    return parsed


def _rewrite_commit(
    args: dict[str, Any],
    *,
    query: str,
    force: bool = False,
) -> dict[str, Any]:
    content = str(args.get("content") or args.get("message") or "")
    low = content.lower()
    stockish = any(
        x in low
        for x in (
            "out of stock",
            "went out",
            "still available",
            "no longer",
            "not available",
            "unavailable",
            "oos",
            "price increased",
            "more expensive",
            "exceeds your budget",
            "above your",
            "over your",
        )
    )
    asks = any(
        x in low
        for x in (
            "should i",
            "do you want",
            "would you like",
            "any preference",
            "preferences?",
            "your preference",
            "or look for a different",
            "or a different",
            "let me know",
            "what would you like",
        )
    )
    model = _model_token(query)
    model_l = (model or "").lower()
    commits = bool(model_l) and model_l in low and (
        "same" in low
        or "another color" in low
        or "other color" in low
        or "same model" in low
    )
    if force:
        out = dict(args)
        out["content"] = _commit_text(model)
        out.pop("message", None)
        return out
    if not (asks or stockish):
        return args
    if commits and not asks and model and f"({model})" in content and ". If" not in content:
        return args
    out = dict(args)
    out["content"] = _commit_text(model)
    out.pop("message", None)
    return out


def _buyer_nudge(text: str) -> bool:
    low = (text or "").lower().strip()
    if len(low) < 8:
        return False
    return any(
        c in low
        for c in (
            "?",
            "which one",
            "before you buy",
            "hold on",
            "wait,",
            "still available",
            "i can only go",
            "non-negotiable",
            "re-check",
            "tell you if you ask",
            "firm requirement",
            "please keep to",
            "compare the current",
            "screen",
            "ram",
            "budget",
            "inch",
        )
    )


def _market_nudge(text: str) -> bool:
    low = (text or "").lower()
    return any(
        c in low
        for c in (
            "still available",
            "more expensive",
            "got more expensive",
            "i can only go",
            "re-check",
            "price jump",
            "price changed",
        )
    )


def _stock_blob(observation: Any) -> dict[str, Any] | None:
    if not isinstance(observation, dict):
        return None
    node = observation.get("observation")
    if isinstance(node, dict) and (
        "in_stock" in node or "price" in node or "added" in node
    ):
        if "added" in node and isinstance(node["added"], dict):
            return node["added"]
        cart = node.get("cart")
        if isinstance(cart, list) and cart and isinstance(cart[0], dict):
            return cart[0]
        return node
    if "in_stock" in observation or "price" in observation:
        return observation
    return None


def _is_oos(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    val = payload.get("in_stock")
    if val is False:
        return True
    if isinstance(val, str) and val.strip().lower() in ("false", "0", "no"):
        return True
    return False


def _infer(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    last_err: Exception | None = None
    for model in _models():
        for _try in range(2):
            try:
                inference = _proxy.post(
                    "/inference/chat/completions",
                    json_data={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "required",
                        "temperature": 0,
                    },
                )
                if inference is None:
                    last_err = RuntimeError(f"inference failed for {model}")
                    time.sleep(0.3)
                    continue
                return inference
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.3)
    raise RuntimeError(f"inference unavailable: {last_err}")


def _world(envelope: dict[str, Any]) -> dict[str, Any]:
    base_id = str(envelope.get("call_id") or "call")
    for attempt in range(3):
        payload = dict(envelope)
        if attempt:
            suffix = f"r{attempt}-{int(time.time() * 1000) % 1_000_000}"
            payload["call_id"] = f"{base_id}-{suffix}"
            payload["idempotency_key"] = payload["call_id"]
            calls = []
            for index, call in enumerate(payload.get("calls") or [], start=1):
                c = dict(call)
                c["call_id"] = f"{payload['call_id']}-{index}"
                calls.append(c)
            payload["calls"] = calls
        result = _proxy.post("/environment/call", json_data=payload)
        if result is not None:
            return result
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"environment call failed after retries ({base_id})")


def _names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [str(c.get("function", {}).get("name") or "") for c in tool_calls]


def _reject_closed(
    tool_calls: list[dict[str, Any]],
    *,
    candidates: list[tuple[str, str]],
    viewed: set[str],
) -> str | None:
    if not candidates:
        return None
    cand_pids = {pid for pid, _ in candidates}
    cand_keys = {f"{pid}::{sku}" for pid, sku in candidates}
    names = _names(tool_calls)
    if any(n in ("search", "filter") for n in names) and len(viewed & cand_pids) < len(
        cand_pids
    ):
        return (
            "Closed candidate set: do not search/filter until every listed product is viewed. "
            f"Missing: {sorted(cand_pids - viewed)}"
        )
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        raw = call["function"].get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        if name == "view":
            pid = str(args.get("product_id") or "")
            if pid and pid not in cand_pids:
                return (
                    f"view product_id {pid} is outside the listed candidates. "
                    f"View only: {sorted(cand_pids)}"
                )
        if name in ("add_to_cart", "inspect_stock", "place_test_order"):
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and pid not in cand_pids:
                return (
                    f"{name} product_id {pid} is outside listed candidates "
                    f"{sorted(cand_pids)}"
                )
            if pid and sku and f"{pid}::{sku}" not in cand_keys:
                if name in ("add_to_cart", "place_test_order"):
                    return (
                        f"{name} {pid}::{sku} is not a listed variant. "
                        f"Use one of: {sorted(cand_keys)}"
                    )
    return None


def _reject_claim_ready(
    tool_calls: list[dict[str, Any]],
    *,
    query: str,
    facts_by_pid: dict[str, dict[str, str]],
    firm_brand: str | None,
) -> str | None:
    """Light gate: order only after view; brand must match when firm."""
    if not _wants_why(query):
        return None
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        if name != "place_test_order":
            continue
        raw = call["function"].get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        pid = str(args.get("product_id") or "")
        if not pid or pid not in facts_by_pid:
            return "view this product_id before ordering so justification claims are grounded"
        brand = str((facts_by_pid.get(pid) or {}).get("brand") or "").strip()
        if firm_brand and brand and brand.lower() != firm_brand.lower():
            return (
                f"ordered brand `{brand}` != firm `{firm_brand}`; "
                "view/cart a matching brand listing"
            )
    return None


def _reject_oos_order(
    tool_calls: list[dict[str, Any]],
    *,
    stock: dict[str, Any] | None,
) -> str | None:
    if not _is_oos(stock):
        return None
    if "place_test_order" in _names(tool_calls):
        return (
            "Rejected: latest stock/cart shows in_stock=false (market shock). "
            "remove_from_cart if needed, pick another candidate, inspect_stock, "
            "then order only an in-stock sku."
        )
    return None


def _reject_needs_cart(
    tool_calls: list[dict[str, Any]],
    *,
    carted: set[tuple[str, str]],
) -> str | None:
    """EnvPack requires add_to_cart before place_test_order."""
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        if name != "place_test_order":
            continue
        raw = call["function"].get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        if not pid:
            return "place_test_order needs product_id; add_to_cart that sku first"
        if (pid, sku) not in carted and (pid, "") not in carted:
            return (
                "place_test_order requires the sku already in cart — "
                "call add_to_cart for this product_id/sku first (do not order bare)"
            )
    return None


def _reject_recovery(
    tool_calls: list[dict[str, Any]],
    *,
    query: str,
    model: str | None,
    color: str | None,
    titles: dict[str, str],
    facts_by_pid: dict[str, dict[str, str]],
    viewed: set[str],
    color_relaxed: bool,
    shocked_pid: str | None,
    shocked_sku: str | None,
    firm_brand: str | None,
    phrase: str | None,
) -> str | None:
    """After preferred color fails: prefer same model; keep firm brand. No infinite locks."""
    phrase_toks = [t for t in re.findall(r"[a-z0-9]+", (phrase or "").lower()) if len(t) > 1]
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        if name not in ("add_to_cart", "place_test_order", "search"):
            continue
        raw = call["function"].get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        if name == "search" and phrase_toks and shocked_pid:
            q = str(args.get("query") or "").lower()
            # Keep at least one phrase token after a retrieval stockout.
            if not any(t in q for t in phrase_toks):
                return (
                    f"recovery search must keep phrase tokens from `{phrase}` "
                    "(do not drift to unrelated product names)"
                )
        if name == "search" and color_relaxed and model:
            q = str(args.get("query") or "")
            if model.lower() not in q.lower():
                return (
                    f"recovery search must include preferred model token `{model}` "
                    "(same model, other color — do not open a new model line)"
                )
        if name not in ("add_to_cart", "place_test_order"):
            continue
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        if not pid:
            continue
        # Viewed products may have empty structured facts — still allow cart.
        if name == "add_to_cart" and pid not in viewed and pid not in facts_by_pid:
            return "view this product_id before carting"
        if shocked_pid and shocked_sku and pid == shocked_pid and sku == shocked_sku:
            return "do not reuse the shocked sku; recover on another live option"
        blob = " ".join(
            [
                titles.get(pid, ""),
                str((facts_by_pid.get(pid) or {}).get("brand") or ""),
                str((facts_by_pid.get(pid) or {}).get("model") or ""),
                sku,
            ]
        )
        brand = str((facts_by_pid.get(pid) or {}).get("brand") or "").strip()
        if firm_brand and brand and brand.lower() != firm_brand.lower():
            return (
                f"firm brand is `{firm_brand}`; view/cart a listing whose brand matches"
            )
        # Same-model lock after shock: title/facts mention token OR same sku family.
        if color_relaxed and model and name in ("add_to_cart", "place_test_order"):
            same_family = bool(
                shocked_sku
                and sku
                and _sku_family(shocked_sku) == _sku_family(sku)
            )
            same_pid = bool(shocked_pid and pid == shocked_pid)
            if not (
                _listing_mentions(blob, model) or same_family or same_pid
            ):
                return (
                    f"recover on model `{model}` (or another sku of the shocked "
                    "listing family) — do not switch to an unrelated product yet"
                )
        if (
            color_relaxed
            and color
            and model
            and _listing_mentions(blob, model)
            and re.search(rf"\b{re.escape(color)}\b", blob, flags=re.I)
        ):
            return (
                f"preferred color `{color}` already failed; "
                "cart another color of the same model"
            )
    return None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    max_steps = int(policy["max_steps"])
    tools = policy["tools"]
    query = str(policy.get("query") or "")
    allowed = _schema_claims(tools)
    candidates = _closed_variants(query)
    cand_pids = {pid for pid, _ in candidates}
    ask_first = _force_ask(query)
    model_hint = _model_token(query)
    pref_color = _pref_color(query)
    firm_brand = _firm_brand(query)
    soft_brand = _soft_brand(query)
    soft_wishes = _soft_wish_tokens(query)
    phrase = _retrieval_phrase(query)
    ret_cat = _retrieval_category(query)

    boot = (
        f"{query}\n\n"
        "Procedure: ask firm specs if withheld-until-asked; else retrieve → view → "
        "inspect_stock → cart → place_test_order. "
        "On stockout: message must COMMIT to same preferred model/other color "
        "(never ask permission); then remove → search that model token → view → "
        "inspect_stock → cart → order. Never switch models while a same-model "
        "other-color option may exist. "
        "Justification only if asked: claims ONLY from the ordered product's view "
        "+ latest inspect (brand/specs/price/stock); never claim category; "
        "never invent resolution class tokens."
    )
    if soft_brand and not firm_brand:
        boot += (
            f"\n\nSOFT BRAND PREFERENCE: `{soft_brand}`. Honor it on the FIRST cart "
            "when a fully firm-compliant listing of that brand exists; firm specs win "
            "if the soft brand has no compliant option."
        )
    if soft_wishes:
        boot += (
            "\n\nSOFT USE-CASE TOKENS: "
            + ", ".join(soft_wishes)
            + ". Among firm-valid options, prefer the listing whose view facts/title "
            "best match these tokens (do not default to cheapest alone)."
        )
    if phrase:
        boot += (
            f"\n\nRETRIEVAL PHRASE: keep title tokens from `{phrase}` in every search. "
            "Survey several distinct product_ids that match the phrase + category + budget, "
            "then inspect_stock → add_to_cart → place_test_order. "
            "Never place_test_order before add_to_cart. "
            "After stockout: COMMIT → remove → search the SAME phrase → view a "
            "different match → inspect → cart → order (do not drift to unrelated names)."
        )
        if ret_cat:
            boot += f" Category must stay under `{ret_cat}`."
    if ask_first:
        boot += (
            "\n\nASK-FIRST: First tool call must be `message` for the withheld firm "
            "hardware spec. After the reply, search with that spec and finish."
        )
    if model_hint:
        boot += (
            f"\n\nPREFERRED MODEL TOKEN: `{model_hint}`. First find/view/cart that "
            "token (preferred color if named). On stockout, recover THE SAME token "
            "in another color before changing models."
        )
    if firm_brand:
        boot += (
            f"\n\nFirm brand `{firm_brand}`: keep it on every cart/order after shocks."
        )
    if candidates:
        listed = ", ".join(f"{pid}::{sku}" for pid, sku in candidates)
        boot += (
            "\n\nCLOSED CANDIDATE SET (mandatory): "
            f"{listed}. "
            "view each listed product_id; rank by stated priority; order a listed key only."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _VOICE},
        {"role": "user", "content": boot},
    ]
    dialogue: list[dict[str, Any]] = []
    must_message = bool(ask_first)
    asked_spec = False
    viewed: set[str] = set()
    searchish = 0
    stock: dict[str, Any] | None = None
    facts_by_pid: dict[str, dict[str, str]] = {}
    titles: dict[str, str] = {}
    shock = False
    reinspect = False
    force_commit = False
    color_relaxed = False
    shocked_pid: str | None = None
    shocked_sku: str | None = None
    reject_streak = 0
    carted: set[tuple[str, str]] = set()

    for turn in range(1, max_steps + 1):
        left = max_steps - turn + 1
        # Near budget: stop soft recovery/claim thrash so we can finish.
        soft_gates = left > 5 and reject_streak < 3
        if must_message:
            hint = (
                "REQUIRED this turn: call `message` once. "
                "If stockout/'still available?': STATE you will take another "
                "color/variant of the SAME preferred model (do not ask permission)."
            )
            if model_hint:
                hint += f" Preferred model token: {model_hint}."
            if phrase and not model_hint:
                hint = (
                    "REQUIRED this turn: call `message` once. "
                    f"COMMIT you will buy another in-stock listing matching `{phrase}` "
                    "within budget/category (do not ask permission)."
                )
            messages.append({"role": "user", "content": hint})
        elif candidates and len(viewed & cand_pids) < len(cand_pids):
            missing = sorted(cand_pids - viewed)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Still missing views for closed-set product_ids: "
                        f"{missing}. Call view on the next missing id (or compare)."
                    ),
                }
            )
        elif shock or reinspect:
            if phrase and not model_hint:
                recover = (
                    "Market shock / re-inspect required: remove_from_cart if cart is stale. "
                    f"Search again with phrase `{phrase}`, VIEW a different product_id "
                    "that still matches, inspect_stock, add_to_cart, place_test_order. "
                    "Do not quit; do not ask permission; do not order without cart."
                )
            else:
                recover = (
                    "Market shock / re-inspect required: remove_from_cart if cart is stale. "
                    "Recover SAME preferred model other color/variant first"
                )
                if model_hint:
                    recover += (
                        f" — search ONLY `{model_hint}` (optionally with another color), "
                        f"then VIEW a different variant of `{model_hint}`"
                    )
                recover += (
                    ", inspect_stock an in-stock sku, then add_to_cart → place_test_order. "
                    "Do not quit; do not ask permission; do not switch models; "
                    "do not order without fresh view+inspect_stock+cart."
                )
            messages.append({"role": "user", "content": recover})
            shock = False
        elif not candidates and searchish >= 3 and len(viewed) < 2:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Anti-thrash: stop repeating the same search. "
                        "Reformulate once OR view best hits, then inspect_stock and finish."
                    ),
                }
            )
        elif not candidates and searchish >= 5 and len(viewed) >= 2:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Search budget spent. Among viewed products meeting firm constraints, "
                        "inspect_stock → add_to_cart → place_test_order NOW."
                    ),
                }
            )
        elif left <= 8 and not must_message and len(viewed) >= 1:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Only ~{left} steps left. Finish path NOW: "
                        "inspect_stock (if needed) → add_to_cart → place_test_order "
                        "on a live in-budget match. Never order without carting first."
                    ),
                }
            )
        elif left <= 6 and not must_message:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Only ~{left} steps left. If you have an in-stock in-budget sku, "
                        "add_to_cart then place_test_order NOW."
                    ),
                }
            )

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for _attempt in range(4):
            inference = _infer(messages=messages, tools=tools)
            assistant = inference["choices"][0]["message"]
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if not tool_calls:
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {
                            "role": "user",
                            "content": "No tool call. Call a supplied tool now.",
                        },
                    ]
                )
                continue
            if must_message and "message" not in _names(tool_calls):
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": "Rejected. Call only `message` this turn.",
                        },
                    ]
                )
                continue
            if not must_message:
                why = _reject_closed(
                    tool_calls, candidates=candidates, viewed=viewed
                )
                if why:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": f"Rejected: {why}"},
                        ]
                    )
                    reject_streak += 1
                    continue
                oos = _reject_oos_order(tool_calls, stock=stock)
                if oos:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": oos},
                        ]
                    )
                    shock = True
                    reinspect = True
                    reject_streak += 1
                    continue
                need_cart = _reject_needs_cart(tool_calls, carted=carted)
                if need_cart:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": f"Rejected: {need_cart}"},
                        ]
                    )
                    reject_streak += 1
                    continue
                # Same-model / phrase recovery locks always (not only soft_gates).
                rec = _reject_recovery(
                    tool_calls,
                    query=query,
                    model=model_hint,
                    color=pref_color,
                    titles=titles,
                    facts_by_pid=facts_by_pid,
                    viewed=viewed,
                    color_relaxed=color_relaxed,
                    shocked_pid=shocked_pid,
                    shocked_sku=shocked_sku,
                    firm_brand=firm_brand,
                    phrase=phrase,
                )
                if rec:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": f"Rejected: {rec}"},
                        ]
                    )
                    reject_streak += 1
                    continue
                if soft_gates:
                    claim_why = _reject_claim_ready(
                        tool_calls,
                        query=query,
                        facts_by_pid=facts_by_pid,
                        firm_brand=firm_brand,
                    )
                    if claim_why:
                        messages.extend(
                            [
                                {
                                    "role": "assistant",
                                    "content": assistant_content,
                                    "tool_calls": tool_calls,
                                },
                                {"role": "user", "content": f"Rejected: {claim_why}"},
                            ]
                        )
                        reject_streak += 1
                        continue
                names_try = _names(tool_calls)
                if (
                    reinspect
                    and "place_test_order" in names_try
                    and "inspect_stock" not in names_try
                ):
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {
                                "role": "user",
                                "content": (
                                    "Rejected: after a market event you must inspect_stock "
                                    "(and fix cart) before place_test_order."
                                ),
                            },
                        ]
                    )
                    continue
            reject_streak = 0
            break
        else:
            if candidates and len(viewed & cand_pids) < len(cand_pids):
                miss = sorted(cand_pids - viewed)[0]
                tool_calls = [
                    {
                        "id": f"fn-view-{turn}",
                        "type": "function",
                        "function": {
                            "name": "view",
                            "arguments": json.dumps({"product_id": miss}),
                        },
                    }
                ]
                assistant_content = f"Viewing closed-set candidate {miss}."
            elif must_message and ask_first and not asked_spec:
                tool_calls = [
                    {
                        "id": f"fn-ask-{turn}",
                        "type": "function",
                        "function": {
                            "name": "message",
                            "arguments": json.dumps(
                                {
                                    "content": (
                                        "What is the firm hardware requirement "
                                        "(RAM, screen size, resolution, or refresh rate)?"
                                    )
                                }
                            ),
                        },
                    }
                ]
                assistant_content = "Asking for the withheld firm spec."
            elif must_message:
                tool_calls = [
                    {
                        "id": f"fn-msg-{turn}",
                        "type": "function",
                        "function": {
                            "name": "message",
                            "arguments": json.dumps(
                                {
                                    "content": (
                                        _rewrite_commit(
                                            {"content": ""},
                                            query=query,
                                            force=True,
                                        ).get("content")
                                        or (
                                            "I'll take another in-stock color/variant "
                                            "of the same preferred model and continue."
                                        )
                                    )
                                }
                            ),
                        },
                    }
                ]
                assistant_content = "Committing after model stall."
            elif stock and not _is_oos(stock):
                pid = str(stock.get("product_id") or "")
                sku = str(stock.get("sku") or "")
                tool_calls = [
                    {
                        "id": f"fn-order-{turn}",
                        "type": "function",
                        "function": {
                            "name": "place_test_order",
                            "arguments": json.dumps(
                                {"product_id": pid, "sku": sku}
                                if pid and sku
                                else {"product_id": pid}
                            ),
                        },
                    }
                ]
                assistant_content = "Finishing on verified in-stock sku after model stall."
            else:
                q = query.split("\n", 1)[0][:120]
                tool_calls = [
                    {
                        "id": f"fn-search-{turn}",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps(
                                {"query": q, "in_stock": True}
                            ),
                        },
                    }
                ]
                assistant_content = "Searching after model stall."

        if must_message:
            only = [
                c
                for c in tool_calls
                if c.get("function", {}).get("name") == "message"
            ]
            if only:
                tool_calls = only[:1]

        names = _names(tool_calls)
        if any(n in ("search", "filter") for n in names):
            searchish += 1
        if "message" in names and ask_first:
            asked_spec = True

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        group = f"{problem_id}-turn-{turn}"
        calls_out: list[dict[str, Any]] = []
        for index, call in enumerate(tool_calls, start=1):
            args = _parse_args(
                call,
                query=query,
                allowed=allowed,
                stock=stock,
                facts_by_pid=facts_by_pid,
            )
            name = call.get("function", {}).get("name") or ""
            if name == "message" and isinstance(args, dict):
                rewritten = _rewrite_commit(args, query=query, force=force_commit)
                if rewritten is not args or force_commit:
                    args = rewritten
                    args.pop("message", None)
            calls_out.append(
                {
                    "call_id": f"{group}-{index}",
                    "action": {"name": name, "args": args},
                }
            )
        envelope = {
            **binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": calls_out,
        }
        result = _world(envelope)
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for call in tool_calls:
            if (call.get("function", {}).get("name") or "") == "view":
                raw = call["function"].get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = {}
                if isinstance(args, dict) and args.get("product_id"):
                    viewed.add(str(args["product_id"]))

        if "message" in names:
            must_message = False
            force_commit = False

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            obs = call_result.get("observation")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(obs),
                }
            )
            payload = _stock_blob(obs if isinstance(obs, dict) else {})
            if payload is None and isinstance(obs, dict):
                payload = _stock_blob({"observation": obs})
            name = tool_call.get("function", {}).get("name") or ""
            if name == "view" and isinstance(obs, dict):
                pid, got, title = _harvest_view(obs)
                if not pid:
                    raw = tool_call["function"].get("arguments", "{}")
                    try:
                        a = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        a = {}
                    if isinstance(a, dict) and a.get("product_id"):
                        pid = str(a["product_id"])
                if pid:
                    viewed.add(pid)
                    if title:
                        titles[pid] = title
                    # Always register the pid — retrieval SKUs often lack brand/specs.
                    facts_by_pid.setdefault(pid, {})
                    if got:
                        facts_by_pid[pid] = {**facts_by_pid[pid], **got}
            if name == "inspect_stock":
                reinspect = False
            if name == "add_to_cart":
                raw = tool_call["function"].get("arguments", "{}")
                try:
                    a = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    a = {}
                err = None
                if isinstance(obs, dict):
                    err = obs.get("error")
                    if err is None and isinstance(obs.get("observation"), dict):
                        err = obs["observation"].get("error")
                if isinstance(a, dict):
                    cpid = str(a.get("product_id") or "")
                    csku = str(a.get("sku") or "")
                    if (
                        cpid
                        and not err
                        and payload
                        and not _is_oos(payload)
                    ):
                        carted.add((cpid, csku))
                        carted.add((cpid, ""))
                    elif cpid:
                        carted.discard((cpid, csku))
                        carted.discard((cpid, ""))
            if name == "remove_from_cart":
                raw = tool_call["function"].get("arguments", "{}")
                try:
                    a = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    a = {}
                if isinstance(a, dict):
                    cpid = str(a.get("product_id") or "")
                    csku = str(a.get("sku") or "")
                    carted.discard((cpid, csku))
                    carted.discard((cpid, ""))
            if name in ("inspect_stock", "add_to_cart", "inspect_cart") and payload:
                stock = payload
                if _is_oos(payload):
                    shock = True
                    reinspect = True
                    force_commit = True
                    # Same-model other-color when query named a model token.
                    if model_hint:
                        color_relaxed = True
                    raw = tool_call["function"].get("arguments", "{}")
                    try:
                        a = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        a = {}
                    if isinstance(a, dict):
                        shocked_pid = str(
                            a.get("product_id") or payload.get("product_id") or ""
                        ) or None
                        shocked_sku = str(a.get("sku") or payload.get("sku") or "") or None
                        if shocked_pid:
                            carted.discard((shocked_pid, shocked_sku or ""))
                            carted.discard((shocked_pid, ""))
            if isinstance(obs, dict):
                inner = (
                    obs.get("observation")
                    if isinstance(obs.get("observation"), dict)
                    else obs
                )
                if isinstance(inner, dict) and inner.get("market_event"):
                    shock = True
                    reinspect = True
                    force_commit = True
                    if model_hint:
                        color_relaxed = True
                    added = inner.get("added")
                    if isinstance(added, dict):
                        stock = added
                        shocked_pid = str(added.get("product_id") or shocked_pid or "") or None
                        shocked_sku = str(added.get("sku") or shocked_sku or "") or None
                        if shocked_pid:
                            carted.discard((shocked_pid, shocked_sku or ""))
                            carted.discard((shocked_pid, ""))
                    me = inner.get("market_event")
                    if isinstance(me, dict):
                        tgt = me.get("target") if isinstance(me.get("target"), dict) else {}
                        tpid = str(tgt.get("product_id") or shocked_pid or "") or None
                        tsku = str(tgt.get("sku") or shocked_sku or "") or None
                        if tpid:
                            shocked_pid = tpid
                            shocked_sku = tsku
                            carted.discard((tpid, tsku or ""))
                            carted.discard((tpid, ""))

        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            messages.append({"role": "user", "content": content})
            if _buyer_nudge(str(content)):
                low = str(content).lower()
                if ask_first and asked_spec and "?" not in low:
                    must_message = False
                else:
                    must_message = True
                if _market_nudge(str(content)):
                    reinspect = True
                    force_commit = True
                elif "?" in low or any(
                    x in low
                    for x in ("which", "still available", "hold on", "before you")
                ):
                    must_message = True
                    if any(
                        x in low
                        for x in (
                            "still available",
                            "out of stock",
                            "hold on",
                            "before you",
                        )
                    ):
                        force_commit = True
        if any(
            (call.get("observation") or {}).get("done") for call in result["calls"]
        ):
            break

        if (
            not must_message
            and candidates
            and len(viewed & cand_pids) >= len(cand_pids)
        ):
            best = _best_closed_pick(
                candidates, facts_by_pid=facts_by_pid, titles=titles
            )
            tip = (
                "All listed candidates viewed. Rank by stated priority "
                "(usually highest refresh → largest screen → lowest price), "
                "inspect_stock the winner, add_to_cart listed sku, place_test_order."
            )
            if best:
                tip += (
                    f" Controller hint from viewed facts: prefer `{best[0]}::{best[1]}` "
                    "unless a listed option clearly beats it on the stated priority."
                )
            messages.append({"role": "user", "content": tip})
        elif not must_message and not candidates and not shock:
            finish = (
                "Episode not done. Prefer inspect_stock → add_to_cart → "
                "place_test_order on an in-stock sku that meets firm constraints. "
                "Omit justification unless the shopper asked why."
            )
            if soft_wishes:
                finish += (
                    " Soft-fit check: prefer a viewed listing that mentions "
                    + "/".join(soft_wishes[:4])
                    + " when still firm-valid."
                )
            messages.append({"role": "user", "content": finish})

    return dialogue


__all__ = ["agent_main"]
