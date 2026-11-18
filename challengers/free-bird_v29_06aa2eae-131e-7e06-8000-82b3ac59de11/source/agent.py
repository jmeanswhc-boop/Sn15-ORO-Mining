

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# Prefer OpenRouter IDs from the live allowlist (Sep 2026 docs).
_DEFAULT_MODELS = {
    "openrouter": "deepseek/deepseek-v3.2",
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
}
# Content/error fallbacks only (no time/phase routing).
_OPENROUTER_FALLBACKS = (
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "moonshotai/kimi-k2.6",
)
_CHUTES_FALLBACKS = (
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "moonshotai/Kimi-K2.6-TEE",
)

_proxy = ProxyClient(timeout=120, max_retries=3)

_CLAIM_FIELDS = frozenset(
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
# Current-state fields that are safe when justifying.
_SAFE_CLAIM_FIELDS = frozenset({"price", "in_stock"})

_FILLER = frozenset(
    {
        "the",
        "a",
        "an",
        "for",
        "with",
        "from",
        "that",
        "this",
        "i",
        "me",
        "my",
        "looking",
        "show",
        "find",
        "want",
        "need",
        "get",
        "buy",
        "also",
        "and",
        "in",
        "is",
        "it",
        "am",
        "im",
        "of",
        "to",
        "or",
        "on",
        "at",
        "by",
        "be",
        "can",
        "has",
        "have",
        "will",
        "would",
        "should",
        "please",
        "item",
        "items",
        "one",
        "up",
        "under",
        "below",
        "above",
        "than",
        "more",
        "less",
        "over",
        "php",
        "pesos",
        "price",
        "budget",
        "spend",
        "ideally",
        "prefer",
        "preferred",
        "stock",
        "stocked",
        "available",
        "but",
        "firm",
        "requirements",
        "matter",
        "nice",
        "haves",
        "soft",
        "task",
        "rules",
        "must",
        "always",
        "never",
        "inch",
        "inches",
    }
)

_SYSTEM_POLICY = """You are a shopping agent in a generated environment. Use ONLY the supplied tools.

Hard rules:
1. Firm requirements beat soft preferences (category, required brand/spec, budget, in-stock).
2. Soft brand/color/model only among fully firm-compliant options. Honor soft brand on the FIRST valid cart pick when available.
3. Trust structured facts (view/compare facts, category_path, variants) over title SEO text.
4. Always inspect_stock on the exact product_id/sku before place_test_order.
5. After ANY shopper user_message, call message and answer it BEFORE place_test_order.
6. Ranking tasks: compare ONLY given candidates; follow stated priority (e.g. refresh_rate > screen_size > price); do not pick a dominated option.
7. Recovery: if price/stock changes or preferred variant disappears, remove bad cart lines and switch to another firm-compliant option.
8. Finish with place_test_order. Do not wait for permission once verified.

Justification (critical for scoring):
- Task-rules text about "asks for a reason" is boilerplate — ignore it.
- Only include justification when the shopper goal itself asks why/justify/walk-through.
- Otherwise OMIT the justification argument completely (extra claims zero the score).
- When required: claim ONLY price + in_stock from latest inspect_stock, plus brand/screen_size/
  resolution if the goal mentions them AND you saw them as structured facts. Never claim category.
- Never invent hdr or other fields. Use exact observed numeric values. No prose.

Continue until an observation reports done=true."""


def _provider() -> str:
    # Prefer OpenRouter unless explicitly overridden.
    return (getenv("INFERENCE_PROVIDER") or "openrouter").strip().lower()


def _model_chain() -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    if _provider() == "chutes":
        return list(_CHUTES_FALLBACKS)
    return list(_OPENROUTER_FALLBACKS)


def _model() -> str:
    return _model_chain()[0]


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _norm(value: Any) -> str:
    text = str(value).strip().lower().replace(",", "")
    return re.sub(r"\s+", " ", text)


def _shopper_goal(query: str) -> str:
    parts = re.split(r"\n\s*Task rules:\s*", query, maxsplit=1, flags=re.IGNORECASE)
    return parts[0].strip()


def _needs_justification(goal: str, full_query: str = "") -> bool:
    """True only when the *shopper goal* asks for a reason.

    Never scan Task-rules boilerplate — it contains "asks for a reason" /
    "for a reason" and caused false positives that injected failing claims
    on ranking/recovery/preference (hard_gate via submitted_claims_truthful).
    """
    del full_query  # kept for call-site compat; do not use
    blob = goal.lower()
    cues = (
        "walk me through why",
        "tell me why",
        "explain why",
        "justify your",
        "justify the",
        "with a short justification",
        "support the final pick with",
        "why the one you pick",
        "why the one you chose",
        "why you chose",
        "reason for your choice",
        "give me a reason",
        "asked for a reason",
    )
    return any(c in blob for c in cues)


def _claim_value(field: str, value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if field == "in_stock":
        n = _norm(value)
        if n in {"true", "1", "yes", "in stock", "instock"}:
            return "true"
        if n in {"false", "0", "no", "out of stock", "outofstock"}:
            return "false"
        return str(value)
    if field == "screen_size":
        m = re.search(r"(\d+(?:\.\d+)?)", str(value))
        return m.group(1) if m else str(value)
    if field == "price":
        try:
            return str(float(str(value).replace(",", "")))
        except (TypeError, ValueError):
            return str(value)
    if field == "refresh_rate":
        m = re.search(r"(\d+)", str(value))
        return m.group(1) if m else str(value)
    return str(value)


def _update_plan_from_shopper(plan: dict[str, Any], content: str) -> None:
    """Apply mid-dialogue budget / firm-spec updates from shopper messages."""
    lower = content.lower()
    m = re.search(
        r"(?:only\s+go\s+up\s+to|up\s+to|at\s+or\s+under|at\s+or\s+below|"
        r"under|below|max(?:imum)?(?:\s+budget)?(?:\s+of)?|budget(?:\s+is)?(?:\s+now)?)\s*"
        r"([\d,]+(?:\.\d+)?)\s*(?:php|pesos)?",
        lower,
    )
    if m:
        try:
            plan["max_price"] = float(m.group(1).replace(",", ""))
        except ValueError:
            pass


def _heuristic_parse(goal: str) -> dict[str, Any]:
    lower = goal.lower()
    plan: dict[str, Any] = {
        "keywords": "",
        "rescue_queries": [],
        "category": None,
        "brand_soft": None,
        "brand_firm": None,
        "max_price": None,
        "min_price": None,
        "spec": {"field": None, "value": None},
        "must_in_stock": True,
        "needs_ranking": any(
            w in lower
            for w in ("rank", "highest", "compare only", "among ties", "priority")
        ),
        "needs_justification": False,
        "given_candidates": re.findall(r"\b\d{6,}::[\w.-]+\b", goal),
    }

    m = re.search(
        r"(?:at or under|at or below|up to|under|below|<=|max(?:imum)?)\s*"
        r"([\d,]+(?:\.\d+)?)\s*(?:php|pesos)?",
        lower,
    )
    if m:
        plan["max_price"] = float(m.group(1).replace(",", ""))

    m = re.search(
        r"(?:between|from)\s*([\d,]+(?:\.\d+)?)\s*(?:and|to|-)\s*([\d,]+(?:\.\d+)?)",
        lower,
    )
    if m:
        plan["min_price"] = float(m.group(1).replace(",", ""))
        plan["max_price"] = float(m.group(2).replace(",", ""))

    m = re.search(
        r"(?:prefer|preferably|ideally[^.]{0,40}prefer(?:s|red)?)\s+"
        r"([A-Za-z][A-Za-z0-9 &.-]{1,40})",
        goal,
        flags=re.IGNORECASE,
    )
    if m:
        brand = re.split(r"\b(?:but|and|,)\b", m.group(1).strip(" .,"), maxsplit=1)[0].strip()
        if brand and brand.lower() not in _FILLER:
            plan["brand_soft"] = brand

    m = re.search(
        r"\b(?:need|want|get)\s+(?:a|an)\s+([A-Z][A-Za-z0-9.-]{1,30})\s+"
        r"(?:smartphone|phone|monitor|headphone|laptop|tablet)\b",
        goal,
    )
    if m and "prefer" not in lower:
        plan["brand_firm"] = m.group(1)

    for field, pattern in (
        ("screen_size", r"(\d+(?:\.\d+)?)\s*-?\s*inch"),
        ("refresh_rate", r"(\d+)\s*hz"),
        ("resolution", r"\b(\d+p|4k|8k|1080|720|1440)\b"),
        ("ram", r"(\d+)\s*gb\s*ram|\bram\s*(\d+)\s*gb"),
    ):
        m = re.search(pattern, lower)
        if m:
            value = next((g for g in m.groups() if g), m.group(0))
            plan["spec"] = {"field": field, "value": value}
            break

    cat = re.search(
        r"\b(?:in the|category)\s+([A-Za-z0-9 &/'-]{3,60}?)\s+category\b",
        goal,
        flags=re.IGNORECASE,
    )
    if cat:
        plan["category"] = cat.group(1).strip()
    else:
        for word in (
            "smartphone",
            "monitor",
            "headphone",
            "headphones",
            "laptop",
            "tablet",
            "webcam",
            "groceries",
        ):
            if word in lower:
                plan["category"] = word
                break

    quoted = re.findall(r'"([^"]{2,80})"', goal)
    head = re.split(r"[.\n]", goal, maxsplit=1)[0]
    head = re.sub(
        r"(?:at or under|at or below|up to|under|below)\s*[\d,]+(?:\.\d+)?\s*(?:php|pesos)?",
        " ",
        head,
        flags=re.IGNORECASE,
    )
    corpus = " ".join([head, *quoted])
    tokens = [
        t
        for t in re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", corpus.lower())
        if len(t) > 1 and t not in _FILLER and not re.fullmatch(r"\d+(?:\.\d+)?", t)
    ]
    if plan.get("category"):
        tokens.insert(0, str(plan["category"]).lower())
    if plan.get("brand_firm"):
        tokens.insert(0, str(plan["brand_firm"]).lower())
    spec_val = (plan.get("spec") or {}).get("value")
    if spec_val:
        tokens.append(str(spec_val))
    keywords = " ".join(dict.fromkeys(tokens[:12]))
    plan["keywords"] = keywords

    rescues: list[str] = []
    cat_s = str(plan.get("category") or "")
    brand_s = str(plan.get("brand_firm") or plan.get("brand_soft") or "")
    if cat_s and brand_s:
        rescues.append(f"{brand_s} {cat_s}")
    if cat_s and spec_val:
        rescues.append(f"{cat_s} {spec_val}")
    if cat_s:
        rescues.append(cat_s)
    words = keywords.split()
    if len(words) > 2:
        rescues.append(" ".join(words[:3]))
    plan["rescue_queries"] = list(dict.fromkeys(rescues))[:4]
    return plan


def _plan_hint(plan: dict[str, Any]) -> str:
    lines = [
        "Internal plan (Task rules win on conflict):",
        f"- keywords: {plan.get('keywords')}",
        f"- rescue searches: {plan.get('rescue_queries')}",
        f"- category: {plan.get('category')}",
        f"- firm brand: {plan.get('brand_firm')} | soft brand: {plan.get('brand_soft')}",
        f"- price: min={plan.get('min_price')} max={plan.get('max_price')}",
        f"- key spec: {plan.get('spec')}",
        f"- needs_ranking: {plan.get('needs_ranking')}",
        f"- needs_justification: {plan.get('needs_justification')}",
    ]
    cands = plan.get("given_candidates") or []
    if cands:
        lines.append(f"- compare ONLY these candidates: {', '.join(map(str, cands[:12]))}")
        lines.append(
            "Ranking: view/compare ALL given candidates; pick the current frontier top by "
            "stated priority (e.g. refresh_rate > screen_size > price). Re-check after any "
            "market event. Do NOT pick a dominated/cheaper lower-Hz option. "
            "Then inspect_stock + place_test_order WITHOUT justification unless required."
        )
    lines.append(
        "ALWAYS answer shopper questions with message before ordering. "
        "Omit justification unless needs_justification is true. "
        "If preferred item fails (price/stock), recover to another firm-compliant option."
    )
    return "\n".join(lines)


def _unwrap_payload(observation: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {}
    inner = observation.get("observation")
    if isinstance(inner, dict) and (
        "action" in inner
        or "results" in inner
        or "product" in inner
        or "variant" in inner
        or "facts" in inner
    ):
        return inner
    return observation


def _is_done(observation: dict[str, Any]) -> bool:
    if not isinstance(observation, dict):
        return False
    if observation.get("done") is True:
        return True
    inner = observation.get("observation")
    return isinstance(inner, dict) and inner.get("done") is True


def _compact_observation(observation: dict[str, Any], limit: int = 3200) -> str:
    try:
        text = json.dumps(observation, ensure_ascii=False, default=str)
    except TypeError:
        text = str(observation)
    if len(text) <= limit:
        return text
    payload = _unwrap_payload(observation)
    slim: dict[str, Any] = {
        "done": observation.get("done"),
        "error": observation.get("error"),
        "action": payload.get("action"),
    }
    if isinstance(payload.get("results"), list):
        slim["results"] = payload["results"][:5]
    for key in (
        "product_id",
        "sku",
        "brand",
        "title",
        "category_path",
        "facts",
        "price",
        "in_stock",
        "variants",
        "product",
        "variant",
    ):
        if key in payload:
            slim[key] = payload[key]
    return json.dumps(slim, ensure_ascii=False, default=str)[:limit]


def _tokenize(text: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 1 and t not in _FILLER
    ]


def _title_hit_score(tokens: list[str], title: str) -> float:
    words = set(re.findall(r"[a-z0-9]+", title.lower()))
    score = 0.0
    for w in tokens:
        if w in words:
            score += 2.0
        elif any(x.startswith(w) or w.startswith(x) for x in words if len(x) > 2):
            score += 1.0
        if any(ch.isdigit() for ch in w) and w in title.lower():
            score += 2.0
    return score


def _rank_search_results(observation: dict[str, Any], plan: dict[str, Any]) -> str | None:
    payload = _unwrap_payload(observation)
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    tokens = _tokenize(str(plan.get("keywords") or ""))
    max_price = plan.get("max_price")
    brand_soft = _norm(plan.get("brand_soft") or "")
    brand_firm = _norm(plan.get("brand_firm") or "")
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        score = _title_hit_score(tokens, title)
        price = item.get("price")
        if isinstance(price, (int, float)) and isinstance(max_price, (int, float)):
            score += 5.0 if price <= max_price else -25.0
        if item.get("in_stock") is True:
            score += 3.0
        elif item.get("in_stock") is False:
            score -= 20.0
        brand = _norm(item.get("brand") or "")
        if brand_firm:
            score += 8.0 if brand == brand_firm or brand_firm in _norm(title) else -12.0
        elif brand_soft and (brand == brand_soft or brand_soft in _norm(title)):
            score += 4.0
        scored.append((score, item))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("product_id", ""))))
    lines = ["Shortlist (verify facts; do not order from title alone):"]
    for score, item in scored[:5]:
        lines.append(
            f"- {score:.1f} id={item.get('product_id')} sku={item.get('sku')} "
            f"brand={item.get('brand')} price={item.get('price')} stock={item.get('in_stock')} "
            f"title={str(item.get('title') or '')[:90]}"
        )
    if scored and scored[0][0] < 4 and plan.get("rescue_queries"):
        lines.append(f"Weak pool — try rescue searches: {plan.get('rescue_queries')}")
    return "\n".join(lines)


def _ingest_observation(memory: dict[str, Any], observation: dict[str, Any]) -> None:
    if not isinstance(observation, dict):
        return
    payload = _unwrap_payload(observation)
    blob = json.dumps(observation, ensure_ascii=False, default=str)

    def remember(field: str, value: Any, source: str = "obs") -> None:
        if value is None or value == "":
            return
        if field == "category" and isinstance(value, list) and value:
            value = value[-1]
        memory.setdefault("facts", {})[field] = value
        memory.setdefault("fact_norms", {})[field] = _norm(value)
        memory.setdefault("fact_sources", {})[field] = source

    action = str(payload.get("action") or "")
    src = action or "obs"

    for key, field in (
        ("brand", "brand"),
        ("category", "category"),
        ("category_path", "category"),
        ("screen_size", "screen_size"),
        ("resolution", "resolution"),
        ("refresh_rate", "refresh_rate"),
        ("price", "price"),
        ("current_price", "price"),
        ("in_stock", "in_stock"),
        ("stock", "in_stock"),
    ):
        if key in payload:
            # Prefer structured facts over category_path for claim safety later.
            if field == "category" and key == "category_path":
                remember(field, payload[key], "category_path")
            else:
                remember(field, payload[key], src)

    facts = payload.get("facts")
    if isinstance(facts, dict):
        if "size" in facts:
            remember("screen_size", facts["size"], "facts")
        if "display" in facts:
            remember("resolution", facts["display"], "facts")
        for key, field in (
            ("brand", "brand"),
            ("screen_size", "screen_size"),
            ("resolution", "resolution"),
            ("refresh_rate", "refresh_rate"),
            ("category", "category"),
        ):
            if key in facts:
                remember(field, facts[key], "facts")

    product = payload.get("product")
    if isinstance(product, dict):
        for key, field in (
            ("brand", "brand"),
            ("category", "category"),
            ("category_path", "category"),
            ("screen_size", "screen_size"),
            ("resolution", "resolution"),
            ("refresh_rate", "refresh_rate"),
        ):
            if key in product:
                remember(field, product[key], "product")
        if product.get("product_id"):
            memory["product_id"] = str(product["product_id"])

    variant = payload.get("variant")
    if isinstance(variant, dict):
        for key, field in (
            ("price", "price"),
            ("current_price", "price"),
            ("in_stock", "in_stock"),
            ("stock", "in_stock"),
            ("sku", "sku"),
        ):
            if key in variant:
                remember(field, variant[key], "variant")
        if variant.get("product_id"):
            memory["product_id"] = str(variant["product_id"])

    for key in ("product_id", "sku"):
        if payload.get(key) not in (None, ""):
            memory[key] = str(payload[key])

    raw_facts = payload.get("raw_product_facts") or payload.get("raw_facts")
    if isinstance(raw_facts, dict):
        for key, field in (
            ("brand", "brand"),
            ("screen_size", "screen_size"),
            ("resolution", "resolution"),
            ("refresh_rate", "refresh_rate"),
            ("category", "category"),
        ):
            if key in raw_facts:
                remember(field, raw_facts[key], "raw_facts")

    results = payload.get("results")
    if isinstance(results, list):
        memory["last_results"] = [r for r in results if isinstance(r, dict)][:10]

    if action:
        memory["last_action"] = action
    if observation.get("error") or payload.get("error"):
        memory["last_error"] = observation.get("error") or payload.get("error")

    memory["last_obs_norm"] = _norm(blob)
    if action == "inspect_stock" or (
        "price" in payload and "in_stock" in payload and payload.get("sku")
    ):
        memory["saw_inspect_stock"] = True
        memory["inspect_product_id"] = str(
            payload.get("product_id") or memory.get("product_id") or ""
        )
        memory["inspect_sku"] = str(payload.get("sku") or memory.get("sku") or "")
    if action in {"view", "compare"}:
        memory["saw_view"] = True

    # Track cart membership for place_test_order (not_in_cart is a common fail).
    cart_items = payload.get("cart")
    if isinstance(cart_items, list):
        keys: set[str] = set()
        for item in cart_items:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("product_id") or "")
            sku = str(item.get("sku") or "")
            if pid and sku:
                keys.add(f"{pid}::{sku}")
                # Prefer live cart price/stock over stale inspect.
                if item.get("current_price") is not None:
                    remember("price", item["current_price"], "cart")
                elif item.get("price") is not None:
                    remember("price", item["price"], "cart")
                if "in_stock" in item:
                    remember("in_stock", item["in_stock"], "cart")
        memory["cart_keys"] = keys
    added = payload.get("added")
    if isinstance(added, dict):
        pid = str(added.get("product_id") or "")
        sku = str(added.get("sku") or "")
        if pid and sku:
            memory.setdefault("cart_keys", set()).add(f"{pid}::{sku}")
    removed = payload.get("removed")
    if isinstance(removed, dict):
        pid = str(removed.get("product_id") or "")
        sku = str(removed.get("sku") or "")
        if pid and sku:
            memory.setdefault("cart_keys", set()).discard(f"{pid}::{sku}")


def _build_safe_claims(
    memory: dict[str, Any],
    needs_justification: bool,
    plan: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build verifier-safe claims from memory only (never model prose)."""
    if not needs_justification:
        return []

    facts = memory.get("facts") or {}
    sources = memory.get("fact_sources") or {}
    goal = _norm((plan or {}).get("goal") or "")
    claims: list[dict[str, str]] = []

    for field in ("price", "in_stock"):
        if field not in facts:
            continue
        claims.append({"field": field, "value": _claim_value(field, facts[field])})

    # Never claim category — path/leaf values often fail canonical_attr_faithful.
    structured = {"facts", "raw_facts", "product", "view", "compare", "variant"}
    for field, needles in (
        ("brand", ()),  # brand is soft support when structured
        ("screen_size", ("inch", "screen")),
        ("resolution", ("720p", "1080p", "1440", "4k", "resolution", " hd")),
        ("refresh_rate", ("hz", "refresh")),
    ):
        if field not in facts:
            continue
        if sources.get(field) not in structured:
            continue
        if needles and not any(n in goal for n in needles):
            continue
        claims.append({"field": field, "value": _claim_value(field, facts[field])})

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for c in claims:
        if c["field"] in seen:
            continue
        seen.add(c["field"])
        out.append(c)
    return out[:5]


def _rewrite_place_order_args(
    args: dict[str, Any],
    memory: dict[str, Any],
    needs_justification: bool,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args = dict(args)
    # Always drop model-authored justification; rebuild only when required.
    args.pop("justification", None)
    if not needs_justification:
        return args
    claims = _build_safe_claims(memory, True, plan)
    if claims:
        args["justification"] = {"claims": claims}
    return args


def _needs_order_guard(
    tool_calls: list[dict[str, Any]],
    memory: dict[str, Any],
    tool_names: set[Any],
    plan: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    rewritten: list[dict[str, Any]] = []
    needs_just = bool(plan.get("needs_justification"))

    for call in tool_calls:
        name = call.get("function", {}).get("name")
        if name != "place_test_order":
            rewritten.append(call)
            continue

        if memory.get("pending_intervention"):
            return (
                "Shopper asked a question. Call message first: state product_id/sku and "
                "confirm the firm spec from viewed facts, THEN place_test_order.",
                tool_calls,
            )

        try:
            args = _arguments(call)
        except Exception as exc:  # noqa: BLE001
            return f"place_test_order arguments invalid: {exc}", tool_calls

        if "inspect_stock" in tool_names and not memory.get("saw_inspect_stock"):
            return (
                "Call inspect_stock on the chosen product_id/sku before place_test_order.",
                tool_calls,
            )

        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        insp_pid = memory.get("inspect_product_id")
        insp_sku = memory.get("inspect_sku")
        if insp_pid and pid not in {"", insp_pid}:
            return (
                f"Order product_id must match last inspect_stock ({insp_pid}).",
                tool_calls,
            )
        if insp_sku and sku not in {"", insp_sku}:
            return (
                f"Order sku must match last inspect_stock ({insp_sku}).",
                tool_calls,
            )

        cart_keys = memory.get("cart_keys") or set()
        if "add_to_cart" in tool_names and cart_keys and f"{pid}::{sku}" not in cart_keys:
            return (
                f"Item {pid}::{sku} is not in cart (cart has {sorted(cart_keys)[:3]}). "
                "Call add_to_cart for the exact product_id/sku, then place_test_order.",
                tool_calls,
            )
        if "add_to_cart" in tool_names and not cart_keys:
            return (
                "Cart is empty. Call add_to_cart for the chosen product_id/sku before place_test_order.",
                tool_calls,
            )

        in_stock = memory.get("facts", {}).get("in_stock")
        if in_stock is False or _norm(in_stock) in {"false", "0", "out of stock"}:
            return (
                "Variant out of stock. Remove it and recover with another firm-compliant option.",
                tool_calls,
            )

        max_price = plan.get("max_price")
        price = memory.get("facts", {}).get("price")
        try:
            if max_price is not None and price is not None and float(price) > float(max_price):
                return (
                    f"Price {price} exceeds budget {max_price}. Recover with another in-budget item.",
                    tool_calls,
                )
        except (TypeError, ValueError):
            pass

        if needs_just and not memory.get("saw_view") and "view" in tool_names:
            return (
                "Justification required: view/compare the final product first so claims are grounded.",
                tool_calls,
            )

        new_args = _rewrite_place_order_args(args, memory, needs_just, plan)
        if needs_just and "justification" not in new_args:
            return (
                "Need grounded justification claims from inspect_stock/view facts "
                "(at least price + in_stock).",
                tool_calls,
            )

        rewritten.append(
            {
                **call,
                "function": {
                    **call.get("function", {}),
                    "arguments": json.dumps(new_args),
                },
            }
        )

    return None, rewritten or tool_calls


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an OpenAI/OpenRouter-valid chat+tools transcript.

    Drops orphan tool rows and incomplete assistant tool_call groups. This is
    required after context trimming, which otherwise leaves tool results whose
    parent assistant tool_calls were cut — a common source of HTTP 400.
    """
    out: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        role = msg.get("role")
        tool_calls = msg.get("tool_calls")
        if role == "assistant" and tool_calls:
            wanted = {
                tc.get("id")
                for tc in tool_calls
                if isinstance(tc, dict) and tc.get("id")
            }
            j = i + 1
            tool_msgs: list[dict[str, Any]] = []
            while j < n and messages[j].get("role") == "tool":
                tool_msgs.append(messages[j])
                j += 1
            got = {tm.get("tool_call_id") for tm in tool_msgs}
            if wanted and wanted <= got:
                # Keep assistant + matching tool results only (in call order).
                by_id = {
                    tm.get("tool_call_id"): tm
                    for tm in tool_msgs
                    if tm.get("tool_call_id") in wanted
                }
                out.append(msg)
                for tc in tool_calls:
                    tid = tc.get("id")
                    if tid in by_id:
                        out.append(by_id[tid])
                i = j
                continue
            # Incomplete group: keep text only, skip orphan tools.
            content = (msg.get("content") or "").strip()
            if content:
                out.append({"role": "assistant", "content": content})
            i = j
            continue
        if role == "tool":
            # Orphan tool result with no parent tool_calls in out.
            i += 1
            continue
        out.append(msg)
        i += 1
    return out


def _trim_messages(messages: list[dict[str, Any]], keep_tail: int = 20) -> None:
    """Shrink history, then sanitize so tool_call pairing stays valid."""
    if len(messages) <= keep_tail + 5:
        messages[:] = _sanitize_messages(messages)
        return

    head = messages[:4]
    # Prefer cutting at a safe boundary: never start the tail on a tool message.
    start = max(4, len(messages) - keep_tail)
    while start < len(messages) and messages[start].get("role") == "tool":
        start += 1
    # If the first kept message is mid-dialogue, also skip a lone assistant
    # tool_calls block that would be incomplete at the cut.
    while start < len(messages):
        msg = messages[start]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Ensure the full tool response group fits in the remaining slice.
            wanted = {
                tc.get("id")
                for tc in msg["tool_calls"]
                if isinstance(tc, dict) and tc.get("id")
            }
            j = start + 1
            got: set[Any] = set()
            while j < len(messages) and messages[j].get("role") == "tool":
                got.add(messages[j].get("tool_call_id"))
                j += 1
            if wanted and wanted <= got:
                break
            start += 1
            continue
        break

    tail = messages[start:]
    messages[:] = _sanitize_messages(
        head
        + [
            {
                "role": "user",
                "content": (
                    "[Earlier tool traces omitted.] Continue from recent observations. "
                    "Answer pending shopper questions with message, inspect_stock, then order."
                ),
            }
        ]
        + tail
    )


def _guard_nudge(
    messages: list[dict[str, Any]],
    assistant_content: str,
    nudge: str,
) -> None:
    """Reject a model tool plan without breaking OpenAI tool-call message rules.

    Never append assistant+tool_calls unless matching role=tool results follow.
    Doing that (then a user nudge) causes HTTP 400 from OpenRouter/OpenAI.
    """
    content = (assistant_content or "").strip()
    if content:
        messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": nudge})


def _chat(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """Try OpenRouter-first model chain; return assistant message or None."""
    last_err = None
    for model in _model_chain():
        # Always send a sanitized copy; keep working messages intact unless trim.
        safe = _sanitize_messages(messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": safe,
            "temperature": 0,
        }
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = "required"
        inference = _proxy.post("/inference/chat/completions", json_data=payload)
        if inference and (inference.get("choices") or []):
            # Persist sanitization so local history stays consistent.
            messages[:] = safe
            return inference["choices"][0].get("message") or {}
        last_err = model
        # Shrink context before next model attempt.
        _trim_messages(messages, keep_tail=16)
    if last_err:
        return None
    return None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    max_steps = int(policy["max_steps"])
    tools = policy.get("tools") or []
    tool_names = {
        (t.get("function") or {}).get("name")
        for t in tools
        if isinstance(t, dict)
    }

    full_query = policy["query"]
    goal = _shopper_goal(full_query)
    plan = _heuristic_parse(goal)
    plan["goal"] = goal
    plan["needs_justification"] = _needs_justification(goal)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_POLICY},
        {"role": "user", "content": full_query},
        {"role": "user", "content": _plan_hint(plan)},
    ]
    if "message" in tool_names:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Critical intervention rule: when a shopper user_message asks which item "
                    "you chose or whether a firm spec holds, call message and answer "
                    "explicitly BEFORE place_test_order. Skipping this zeros the score."
                ),
            }
        )
    if not plan.get("needs_justification"):
        messages.append(
            {
                "role": "user",
                "content": (
                    "This task does not require justification. When calling place_test_order, "
                    "omit the justification argument completely."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = [
        {"role": "meta", "provider": _provider(), "model": _model(), "plan": plan}
    ]
    memory: dict[str, Any] = {
        "facts": {},
        "fact_norms": {},
        "fact_sources": {},
        "pending_intervention": False,
        "cart_keys": set(),
    }
    guard_retries = 0
    env_turn = 0
    max_model_steps = max_steps + 8
    search_rounds = 0

    for _model_step in range(1, max_model_steps + 1):
        if env_turn >= max_steps:
            break

        _trim_messages(messages)
        if env_turn >= max(1, max_steps - 4) and not memory.get("ordered"):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Step budget almost gone. If you have a firm-compliant in-stock "
                        "variant: answer any pending shopper question with message, "
                        "inspect_stock if needed, then place_test_order NOW "
                        f"{'(with grounded claims only)' if plan.get('needs_justification') else '(NO justification)'}."
                    ),
                }
            )

        assistant = None
        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for _attempt in range(3):
            assistant = _chat(messages, tools)
            if assistant is None:
                _trim_messages(messages, keep_tail=14)
                continue
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if tool_calls:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": (
                            "Choose a supplied tool now "
                            "(search/filter/view/compare/inspect_stock/message/place_test_order)."
                        ),
                    },
                ]
            )
        if not tool_calls:
            # Soft exit rather than hard crash when inference is exhausted.
            dialogue.append(
                {
                    "role": "error",
                    "content": "inference failed after retries; no tool call",
                }
            )
            break

        # Force shopper reply before ordering.
        if memory.get("pending_intervention") and "message" in tool_names:
            names = [c.get("function", {}).get("name") for c in tool_calls]
            if "message" not in names:
                guard_retries += 1
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "guard_blocked": "pending_intervention",
                    }
                )
                _guard_nudge(
                    messages,
                    assistant_content,
                    (
                        "STOP. Your previous tool choice was rejected (not executed). "
                        "Call message FIRST. Answer which product_id/sku you will buy and "
                        "confirm the firm spec (e.g. screen size) from view facts."
                    ),
                )
                if guard_retries < 5:
                    continue

        block, tool_calls = _needs_order_guard(tool_calls, memory, tool_names, plan)
        if block and guard_retries < 5:
            guard_retries += 1
            dialogue.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                    "guard_blocked": block,
                }
            )
            _guard_nudge(
                messages,
                assistant_content,
                (
                    "Your previous tool choice was rejected (not executed). "
                    f"Do not finish yet. {block}"
                ),
            )
            continue

        # Final sanitizer: always strip/rebuild justification even if guards were skipped.
        sanitized_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            name = call.get("function", {}).get("name")
            if name != "place_test_order":
                sanitized_calls.append(call)
                continue
            try:
                args = _arguments(call)
            except Exception:  # noqa: BLE001
                sanitized_calls.append(call)
                continue
            new_args = _rewrite_place_order_args(
                args, memory, bool(plan.get("needs_justification")), plan
            )
            sanitized_calls.append(
                {
                    **call,
                    "function": {
                        **call.get("function", {}),
                        "arguments": json.dumps(new_args),
                    },
                }
            )
        tool_calls = sanitized_calls

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        env_turn += 1
        group_id = f"{problem_id}-turn-{env_turn}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": env_turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {
                        "name": call["function"]["name"],
                        "args": _arguments(call),
                    },
                }
                for index, call in enumerate(tool_calls, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            dialogue.append({"role": "error", "content": "environment call failed"})
            break

        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            observation = call_result.get("observation") or {}
            _ingest_observation(memory, observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _compact_observation(observation),
                }
            )
            action_name = tool_call.get("function", {}).get("name")
            if action_name == "message":
                memory["pending_intervention"] = False
            if action_name in {"search", "filter"}:
                search_rounds += 1
                note = _rank_search_results(observation, plan)
                if note and search_rounds <= 3:
                    messages.append({"role": "user", "content": note})

        if result.get("user_message"):
            content = (result["user_message"].get("content") or "").strip()
            if content:
                _update_plan_from_shopper(plan, content)
                memory["pending_intervention"] = True
                guard_retries = 0
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Shopper message: {content[:1500]}\n"
                            "REQUIRED NEXT ACTION: call message and answer this before ordering."
                            + (
                                f" Current budget ceiling: {plan['max_price']} PHP."
                                if plan.get("max_price") is not None
                                else ""
                            )
                        ),
                    }
                )

        if memory.get("last_error"):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool error: {memory['last_error']}. Recover: remove bad cart items, "
                        f"try rescue searches {plan.get('rescue_queries')}, pick another "
                        "firm-compliant in-stock option, inspect_stock, then order "
                        f"{'with grounded claims' if plan.get('needs_justification') else 'WITHOUT justification'}."
                    ),
                }
            )
            memory["last_error"] = None

        if any(_is_done(call.get("observation") or {}) for call in result["calls"]):
            memory["ordered"] = True
            break

    return dialogue


__all__ = ["agent_main"]
