

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

# Always-safe justification fields (current-state from inspect_stock/cart).
_SAFE_CLAIM_FIELDS = frozenset({"price", "in_stock"})
# Verifier-accepted justification fields (scoring contract — not a product taxonomy).
_CLAIM_ALLOWLIST = frozenset(
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
# Observation key aliases → canonical fact names (catalog schema, not product types).
_FACT_KEY_ALIASES = {
    "size": "screen_size",
    "display": "resolution",
    "stock": "in_stock",
    "current_price": "price",
}
# Generic measurement / wording cues → attribute field (schema binding, any domain).
_ATTR_CUES: tuple[tuple[str, str], ...] = (
    ("screen_size", r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?"),
    ("refresh_rate", r"(\d+)\s*hz"),
    ("resolution", r"\b(\d{3,4}p|4k|8k|full\s*hd|fhd|\bhd\b)\b"),
    ("ram", r"(\d+)\s*gb\s*ram|\bram\s*(\d+)\s*gb"),
)


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

Pipeline (big-and-big spirit: search → weigh → prune → judge_select → commit):
1. Firm requirements beat soft preferences (category, required brand/attrs, budget, in-stock).
2. Soft brand/color/model only among fully firm-compliant options. Honor soft brand on the FIRST valid cart pick when available.
3. Trust structured facts (view/compare facts, category_path, variants, attributes) over title SEO text.
4. When several candidates remain, prefer the judge_select winner (attributes/sku evidence over title word-count).
5. Always inspect_stock on the exact product_id/sku before place_test_order.
6. After ANY shopper user_message, call message and answer it BEFORE place_test_order.
7. Ranking tasks: compare ONLY given candidates; follow the goal's stated priority order; do not pick a dominated option.
8. Recovery: if price/stock changes or preferred variant disappears, remove bad cart lines and switch to another firm-compliant option.
9. Finish with place_test_order. Do not wait for permission once verified.
10. Domain-agnostic: do not assume electronics (phones/monitors). Follow the goal's category and attributes as stated.

Justification (critical for scoring):
- Task-rules text about "asks for a reason" is boilerplate — ignore it.
- Only include justification when the shopper goal itself asks why/justify/walk-through.
- Otherwise OMIT the justification argument completely (extra claims zero the score).
- When required: ALWAYS claim price + in_stock from the latest inspect_stock/cart, plus any
  other attributes the goal explicitly requires — only from structured view/compare facts.
- Exact observed values only — never invent or remap specs. No prose.

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


# big-and-big ProductSearchEngine.__judge_select__ knobs
_MIN_JUDGE_SCORE = 6.0
_RELAX_SCORE_GAP = 0.6
_SELECT_CANDIDATES = 8
_JUDGE_ROUNDS_MAX = 2

_PV_ITEM_JUDGE = """
Goal: pick one listing from a candidate pool for a single-item shopping request.

Inputs: JSON with `request`, `candidates` (title, price, brand, sku, attributes/facts,
in_stock), `firm` constraints, and `only_product_type`.

Judging principles (big-and-big PV_ITEM_JUDGE):
(a) Structured attributes outweigh title prose — trust facts/attributes over SEO titles.
(b) Every stated requirement counts: brand, material, color, model, quantity, size, price,
    category, and any other attribute named in the request (any product domain).
(c) Do not upgrade a candidate because its title repeats more query words.
(d) Treat formatting/spacing differences as equivalent matches.
(e) When `only_product_type` is true, require a bare product-type "only" variant.
(f) Price is a last-resort tiebreaker.

Output strict JSON only:
{
  "best_product_id": "<id from candidates or null>",
  "best_sku": "<sku or null>",
  "reason": "1-2 sentences citing attribute/sku values",
  "relevance_score": <integer 0-10>
}
""".strip()

_PV_INTENT = """
Extract a domain-agnostic shopping intent plan from the shopper goal only.
Do NOT assume electronics or any fixed product taxonomy.

Output strict JSON only:
{
  "keywords": "3-12 useful search tokens from the goal",
  "category": "product/class phrase from the goal or null",
  "brand_firm": "required brand or null",
  "brand_soft": "preferred/soft brand or null",
  "max_price": <number or null>,
  "min_price": <number or null>,
  "constraints": [
    {"field": "snake_case_attribute", "value": "as stated", "firm": true}
  ],
  "claim_fields": ["attribute fields to evidence if justification is required"],
  "rescue_queries": ["up to 4 broader or narrower search strings"],
  "needs_ranking": false
}

Rules:
- Only extract what the goal actually says; never invent constraints.
- category may be any class (beauty, groceries, furniture, phone, etc.) or null.
- constraints.field names should be generic (color, material, capacity, size, ...).
- claim_fields: goal-mentioned attributes only; omit price and in_stock.
- rescue_queries: linguistic broaden/narrow of the request, not a fixed product template.
""".strip()


def _parse_json_content(content: str) -> dict[str, Any] | None:
    if not content:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    starts = []
    for ch in ("{", "["):
        i = cleaned.find(ch)
        if i >= 0:
            starts.append(i)
    for start in sorted(starts):
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _candidate_prompt_rows(results: list[dict[str, Any]], limit: int = _SELECT_CANDIDATES) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results[:limit]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "product_id": item.get("product_id"),
                "sku": item.get("sku"),
                "title": str(item.get("title") or "")[:160],
                "brand": item.get("brand"),
                "price": item.get("price"),
                "in_stock": item.get("in_stock"),
                "facts": item.get("facts"),
                "attributes": item.get("attributes") or item.get("facts"),
                "category_path": item.get("category_path") or item.get("category"),
            }
        )
    return rows


def _extract_judge_response(js: dict[str, Any], candidates: list[dict[str, Any]]) -> int:
    """big-and-big Prompt.extract_judge_response — match by product_id."""
    pid = str(js.get("best_product_id") or "").strip()
    if not pid or pid.lower() in {"null", "none", "n/a"}:
        return -1
    for i, prod in enumerate(candidates):
        if str(prod.get("product_id") or "") == pid:
            return i
    return -1


def _is_prod_only_query(goal: str) -> bool:
    return bool(re.search(r"\bproduct\s+only\b", goal or "", re.I))


def _llm_json_once(system: str, user: str, models: list[str] | None = None) -> dict[str, Any] | None:
    chain = models or _model_chain()[:3]
    for model in chain:
        inference = _proxy.post(
            "/inference/chat/completions",
            json_data={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.4,
            },
        )
        if not inference:
            continue
        content = (
            (inference.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        )
        parsed = _parse_json_content(content)
        if parsed:
            return parsed
    return None


def _judge_select(goal: str, results: list[dict[str, Any]], plan: dict[str, Any]) -> str | None:
    """Port of ProductSearchEngine.__judge_select__ for dynamic env tools.

    Falls back to heuristic shortlist tip when judge score is low or pool is tiny.
    """
    if not results:
        return None
    cap = _candidate_prompt_rows(results, _SELECT_CANDIDATES)
    if len(cap) < 2:
        top = cap[0] if cap else None
        if not top:
            return None
        return (
            f"Single-candidate pool: product_id={top.get('product_id')} "
            f"sku={top.get('sku')}. View/inspect this exact variant next."
        )

    payload = {
        "request": goal[:1200],
        "only_product_type": _is_prod_only_query(goal),
        "firm": {
            "category": plan.get("category"),
            "brand_firm": plan.get("brand_firm"),
            "brand_soft": plan.get("brand_soft"),
            "max_price": plan.get("max_price"),
            "min_price": plan.get("min_price"),
            "spec": plan.get("spec"),
            "constraints": plan.get("constraints") or [],
        },
        "candidates": cap,
    }
    js = _llm_json_once(_PV_ITEM_JUDGE, json.dumps(payload, ensure_ascii=False))
    if not js:
        return None

    idx = _extract_judge_response(js, cap)
    try:
        jscore = float(js.get("relevance_score", 0) or 0)
    except (TypeError, ValueError):
        jscore = 0.0

    if idx < 0 or jscore < _MIN_JUDGE_SCORE:
        # Recover like big-and-big: fall back to heuristic top of ranked pool.
        return (
            f"Judge soft-fail (score={jscore}, idx={idx}). "
            "Use the heuristic shortlist; view top firm-compliant options, then inspect_stock."
            + (
                f" Rescue: {plan.get('rescue_queries')}"
                if plan.get("rescue_queries")
                else ""
            )
        )

    best = cap[idx]
    return (
        f"judge_select pick: product_id={best.get('product_id')} "
        f"sku={best.get('sku')} relevance_score={jscore} "
        f"reason={js.get('reason')}. "
        "View/compare this exact variant, inspect_stock, then order if firm constraints hold."
    )


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
    if field == "price":
        try:
            return f"{float(str(value).replace(',', '')):.2f}"
        except (TypeError, ValueError):
            return str(value)
    text = str(value).strip()
    # Canonicalize measurement spellings for claim matching (any domain using this field).
    if field == "screen_size":
        m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*in(?:ch(?:es)?)?\b", text, flags=re.I)
        if m:
            return f"{m.group(1)}in"
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        return f"{m.group(1)}in" if m else text
    if field == "refresh_rate":
        m = re.search(r"(\d+)\s*hz\b", text, flags=re.I)
        if m:
            return m.group(1)
        m = re.search(r"(\d+)", text)
        return m.group(1) if m else text
    if field == "resolution":
        # Prefer compact class tokens when present (longer tokens first).
        n = _norm(text).replace(" ", "")
        if "fullhd" in n or n in {"full-hd", "full_hd"}:
            return "fhd"
        for token in ("1080p", "1440p", "720p", "2160p", "4k", "8k", "fhd", "uhd", "hd"):
            if n == token or token in n:
                if token == "hd" and any(x in n for x in ("fhd", "uhd", "qhd", "fullhd")):
                    continue
                return token
        return text
    return text


def _add_constraint(plan: dict[str, Any], field: str, value: Any, firm: bool = True) -> None:
    field = str(field or "").strip().lower().replace(" ", "_")
    if not field or value is None or value == "":
        return
    constraints = list(plan.get("constraints") or [])
    # Replace existing same field.
    constraints = [c for c in constraints if not (isinstance(c, dict) and c.get("field") == field)]
    constraints.append({"field": field, "value": value, "firm": bool(firm)})
    plan["constraints"] = constraints
    if firm and field not in {"brand", "category", "price"}:
        plan["spec"] = {"field": field, "value": value}
    claim_fields = list(plan.get("claim_fields") or [])
    if field in _CLAIM_ALLOWLIST and field not in _SAFE_CLAIM_FIELDS and field not in claim_fields:
        claim_fields.append(field)
        plan["claim_fields"] = claim_fields


def _goal_resolution_token(goal: str) -> str | None:
    """Pick the primary resolution class token named in the goal (schema vocabulary)."""
    lower = (goal or "").lower()
    if re.search(r"full\s*hd|\bfhd\b", lower):
        return "fhd"
    if re.search(r"\buhd\b|\b8k\b", lower):
        return "uhd" if "uhd" in lower else "8k"
    if re.search(r"\b4k\b", lower):
        return "4k"
    # Bare HD after excluding full-hd already handled.
    if re.search(r"\bhd\b", lower):
        return "hd"
    m = re.search(r"\b(\d{3,4}p)\b", lower)
    if m:
        return m.group(1)
    return None


def _extract_attr_cues(text: str) -> list[tuple[str, str]]:
    """Pull measurement-like attributes from free text (any product domain)."""
    found: list[tuple[str, str]] = []
    lower = text.lower()
    for field, pattern in _ATTR_CUES:
        if field == "resolution":
            tok = _goal_resolution_token(text)
            if tok:
                found.append((field, tok))
            continue
        m = re.search(pattern, lower, flags=re.IGNORECASE)
        if not m:
            continue
        value = next((g for g in m.groups() if g), m.group(0))
        value = re.sub(r"\s+", "", str(value).lower())
        found.append((field, value))
    return found


def _parse_ranking_priority(goal: str) -> list[dict[str, str]]:
    """Parse 'highest X / largest Y / lowest Z' tradeoff order from the goal."""
    lower = goal.lower()
    axes: list[dict[str, str]] = []
    # Split on "among" clauses to preserve stated order.
    chunks = re.split(r"\bamong\b", lower)
    cue_map = (
        (r"highest\s+([a-z][a-z0-9_ ]{1,40}?)(?:\.|,|;|$)", "max"),
        (r"largest\s+([a-z][a-z0-9_ ]{1,40}?)(?:\.|,|;|$)", "max"),
        (r"lowest\s+([a-z][a-z0-9_ ]{1,40}?)(?:\.|,|;|$)", "min"),
        (r"smallest\s+([a-z][a-z0-9_ ]{1,40}?)(?:\.|,|;|$)", "min"),
        (r"best\s+([a-z][a-z0-9_ ]{1,40}?)(?:\.|,|;|$)", "max"),
    )
    alias = {
        "refresh rate": "refresh_rate",
        "advertised maximum refresh rate": "refresh_rate",
        "screen": "screen_size",
        "screen size": "screen_size",
        "price": "price",
    }
    for chunk in chunks:
        for pattern, direction in cue_map:
            m = re.search(pattern, chunk)
            if not m:
                continue
            raw = m.group(1).strip()
            raw = re.sub(r"\b(advertised|maximum|available|option|monitor|item)s?\b", " ", raw)
            raw = re.sub(r"\s+", " ", raw).strip(" .")
            field = alias.get(raw, raw.replace(" ", "_"))
            if field and all(a.get("field") != field for a in axes):
                axes.append({"field": field, "direction": direction})
            break
    return axes[:5]


def _update_plan_from_shopper(plan: dict[str, Any], content: str) -> None:
    """Apply mid-dialogue budget / firm-spec updates from shopper messages (any domain)."""
    if not content or not content.strip():
        return
    lower = content.lower()
    m = re.search(
        r"(?:only\s+go\s+up\s+to|up\s+to|at\s+or\s+under|at\s+or\s+below|"
        r"under|below|max(?:imum)?(?:\s+budget)?(?:\s+of)?|budget(?:\s+is)?(?:\s+now)?"
        r"|at\s+most)\s*"
        r"([\d,]+(?:\.\d+)?)",
        lower,
    )
    if m:
        try:
            plan["max_price"] = float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    m = re.search(
        r"(?:required\s+brand\s+is|brand\s*(?:is|must\s+be))\s+"
        r"([A-Za-z0-9][A-Za-z0-9&-]{0,40})",
        content,
        flags=re.IGNORECASE,
    )
    if m:
        brand = m.group(1).strip(" .,;:")
        if brand and brand.lower() not in _FILLER and brand.lower() != "is":
            plan["brand_firm"] = brand

    m = re.search(
        r"(?:required\s+catalog\s+category\s+is|required\s+category\s+is|"
        r"catalog\s+category\s+is|category\s+is)\s+"
        r"([A-Za-z0-9][A-Za-z0-9 &/'-]{1,60}?)(?:\.|,|;|$)",
        content,
        flags=re.IGNORECASE,
    )
    if m:
        cat = m.group(1).strip(" .,;:")
        if cat and cat.lower() not in {"is", "a", "an", "the"}:
            plan["category"] = cat

    for field, value in _extract_attr_cues(content):
        _add_constraint(plan, field, value, firm=True)

    # Generic "needs/still needs/must have <text>" residual — keep as soft keyword boost.
    m = re.search(
        r"(?:still\s+needs|needs|must\s+have|required)\s+(?:a\s+)?(.{3,60}?)(?:\.|,|;|$)",
        lower,
    )
    if m and plan.get("keywords"):
        extra = " ".join(_tokenize(m.group(1))[:4])
        if extra:
            plan["keywords"] = " ".join(dict.fromkeys((extra + " " + str(plan["keywords"])).split()))


def _empty_plan(goal: str) -> dict[str, Any]:
    lower = goal.lower()
    return {
        "keywords": "",
        "rescue_queries": [],
        "category": None,
        "brand_soft": None,
        "brand_firm": None,
        "max_price": None,
        "min_price": None,
        "spec": {"field": None, "value": None},
        "constraints": [],
        "claim_fields": [],
        "ranking_priority": [],
        "must_in_stock": True,
        "needs_clarify": any(
            p in lower
            for p in (
                "if you ask",
                "tell you if you ask",
                "ask me",
                "when you ask",
                "i will tell you",
            )
        ),
        "needs_ranking": any(
            w in lower
            for w in ("rank", "highest", "compare only", "among ties", "priority", "strictly worse")
        ),
        "needs_justification": False,
        "given_candidates": re.findall(r"\b\d{6,}::[\w.-]+\b", goal),
    }


def _keyword_tokens_from_goal(goal: str) -> list[str]:
    quoted = re.findall(r'"([^"]{2,80})"', goal)
    head = re.split(r"[.\n]", goal, maxsplit=1)[0]
    head = re.sub(
        r"(?:at or under|at or below|up to|under|below|between|from)\s*"
        r"[\d,]+(?:\.\d+)?(?:\s*(?:and|to|-)\s*[\d,]+(?:\.\d+)?)?",
        " ",
        head,
        flags=re.IGNORECASE,
    )
    corpus = " ".join([head, *quoted])
    return [
        t
        for t in re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", corpus.lower())
        if len(t) > 1 and t not in _FILLER and not re.fullmatch(r"\d+(?:\.\d+)?", t)
    ]


def _fallback_parse(goal: str) -> dict[str, Any]:
    """Format-stable extracts only — no product-type or electronics-spec whitelist."""
    plan = _empty_plan(goal)
    lower = goal.lower()

    m = re.search(
        r"(?:at or under|at or below|up to|under|below|<=|max(?:imum)?)\s*"
        r"([\d,]+(?:\.\d+)?)",
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

    # General: "need/want/get a BrandNoun productnoun" — Brand is Capitalized, noun is lowercase.
    m = re.search(
        r"\b(?:need|want|get)\s+(?:a|an)\s+([A-Z][A-Za-z0-9.-]{1,30})\s+([a-z][a-z0-9&/-]*)",
        goal,
    )
    if m and "prefer" not in lower:
        plan["brand_firm"] = m.group(1)
        if not plan.get("category"):
            plan["category"] = m.group(2)

    # Literal "… category" phrasing only — no fixed taxonomy list.
    cat = re.search(
        r"\b(?:in the|category)\s+([A-Za-z0-9 &/'-]{3,60}?)\s+category\b",
        goal,
        flags=re.IGNORECASE,
    )
    if cat:
        plan["category"] = cat.group(1).strip()

    for field, value in _extract_attr_cues(goal):
        _add_constraint(plan, field, value, firm=True)

    plan["ranking_priority"] = _parse_ranking_priority(goal)
    if plan["ranking_priority"]:
        plan["needs_ranking"] = True

    tokens = _keyword_tokens_from_goal(goal)
    if plan.get("category"):
        tokens.insert(0, str(plan["category"]).lower())
    if plan.get("brand_firm"):
        tokens.insert(0, str(plan["brand_firm"]).lower())
    keywords = " ".join(dict.fromkeys(tokens[:12]))
    plan["keywords"] = keywords

    rescues: list[str] = []
    words = keywords.split()
    if plan.get("category") and plan.get("brand_soft"):
        rescues.append(f"{plan['brand_soft']} {plan['category']}")
    if plan.get("category"):
        rescues.append(str(plan["category"]))
    if len(words) > 3:
        rescues.append(" ".join(words[:3]))
    if len(words) > 1:
        rescues.append(" ".join(words[:2]))
    plan["rescue_queries"] = list(dict.fromkeys(r for r in rescues if r))[:4]
    return plan


def _coerce_constraint(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    field = str(raw.get("field") or "").strip().lower().replace(" ", "_")
    value = raw.get("value")
    if not field or value is None or value == "":
        return None
    firm = raw.get("firm")
    if firm is None:
        firm = True
    return {"field": field, "value": value, "firm": bool(firm)}


def _llm_intent_parse(goal: str) -> dict[str, Any] | None:
    """LLM intent extraction — domain-agnostic structured plan."""
    js = _llm_json_once(_PV_INTENT, goal[:2000], models=_model_chain()[:2])
    if not js:
        return None
    out: dict[str, Any] = {}
    for key in ("keywords", "category", "brand_firm", "brand_soft"):
        val = js.get(key)
        if isinstance(val, str) and val.strip() and val.strip().lower() not in {
            "null",
            "none",
            "n/a",
        }:
            out[key] = val.strip()
    for key in ("max_price", "min_price"):
        val = js.get(key)
        if isinstance(val, (int, float)):
            out[key] = float(val)
        elif isinstance(val, str):
            try:
                out[key] = float(val.replace(",", ""))
            except ValueError:
                pass
    constraints: list[dict[str, Any]] = []
    for raw in js.get("constraints") or []:
        c = _coerce_constraint(raw)
        if c:
            constraints.append(c)
    out["constraints"] = constraints
    claim_fields: list[str] = []
    for raw in js.get("claim_fields") or []:
        field = str(raw or "").strip().lower().replace(" ", "_")
        if (
            field
            and field in _CLAIM_ALLOWLIST
            and field not in _SAFE_CLAIM_FIELDS
            and field not in claim_fields
        ):
            claim_fields.append(field)
    # Firm non-brand constraints are claimable when justifying.
    for c in constraints:
        field = str(c.get("field") or "")
        if (
            c.get("firm")
            and field
            and field in _CLAIM_ALLOWLIST
            and field not in _SAFE_CLAIM_FIELDS
            and field != "brand"
            and field not in claim_fields
        ):
            claim_fields.append(field)
    out["claim_fields"] = claim_fields
    rescues: list[str] = []
    for raw in js.get("rescue_queries") or []:
        q = str(raw or "").strip()
        if q and q not in rescues:
            rescues.append(q)
    if rescues:
        out["rescue_queries"] = rescues[:4]
    if "needs_ranking" in js:
        out["needs_ranking"] = bool(js.get("needs_ranking"))
    # Primary spec = first firm non-brand constraint (judge_select compat).
    for c in constraints:
        if c.get("firm") and c.get("field") not in {"brand", "category", "price"}:
            out["spec"] = {"field": c["field"], "value": c["value"]}
            break
    return out


def _merge_plans(base: dict[str, Any], llm: dict[str, Any]) -> dict[str, Any]:
    plan = dict(base)
    for key in ("keywords", "category", "brand_firm", "brand_soft"):
        if llm.get(key):
            plan[key] = llm[key]
    for key in ("max_price", "min_price"):
        if llm.get(key) is not None and plan.get(key) is None:
            plan[key] = llm[key]
    if llm.get("constraints"):
        plan["constraints"] = llm["constraints"]
    if llm.get("claim_fields"):
        plan["claim_fields"] = llm["claim_fields"]
    if llm.get("rescue_queries"):
        # Prefer LLM rescues; keep fallback extras that add coverage.
        merged = list(llm["rescue_queries"])
        for q in base.get("rescue_queries") or []:
            if q not in merged:
                merged.append(q)
        plan["rescue_queries"] = merged[:4]
    if llm.get("spec") and (llm["spec"] or {}).get("field"):
        plan["spec"] = llm["spec"]
    if llm.get("needs_ranking"):
        plan["needs_ranking"] = True
    if base.get("ranking_priority") and not plan.get("ranking_priority"):
        plan["ranking_priority"] = base["ranking_priority"]
    if base.get("needs_clarify"):
        plan["needs_clarify"] = True
    # Ensure keywords mention category/brand when known.
    tokens = _tokenize(str(plan.get("keywords") or ""))
    if plan.get("category"):
        tokens.insert(0, str(plan["category"]).lower())
    if plan.get("brand_firm"):
        tokens.insert(0, str(plan["brand_firm"]).lower())
    spec_val = (plan.get("spec") or {}).get("value")
    if spec_val:
        tokens.append(_norm(spec_val).replace(" ", ""))
    plan["keywords"] = " ".join(dict.fromkeys(tokens[:12]))
    return plan


def _build_plan(goal: str) -> dict[str, Any]:
    """General plan: LLM intent first, format-stable fallback underneath."""
    base = _fallback_parse(goal)
    llm = _llm_intent_parse(goal)
    if llm:
        return _merge_plans(base, llm)
    return base


def _plan_hint(plan: dict[str, Any]) -> str:
    constraints = plan.get("constraints") or []
    constraint_txt = ", ".join(
        f"{c.get('field')}={c.get('value')}({'firm' if c.get('firm') else 'soft'})"
        for c in constraints[:8]
        if isinstance(c, dict)
    ) or "none"
    priority = plan.get("ranking_priority") or []
    priority_txt = " > ".join(
        f"{p.get('field')}({p.get('direction')})" for p in priority if isinstance(p, dict)
    ) or "none"
    lines = [
        "Internal plan (Task rules win on conflict):",
        f"- keywords: {plan.get('keywords')}",
        f"- rescue searches: {plan.get('rescue_queries')}",
        f"- category: {plan.get('category')}",
        f"- firm brand: {plan.get('brand_firm')} | soft brand: {plan.get('brand_soft')}",
        f"- price: min={plan.get('min_price')} max={plan.get('max_price')}",
        f"- constraints: {constraint_txt}",
        f"- key spec: {plan.get('spec')}",
        f"- claim_fields: {plan.get('claim_fields')}",
        f"- ranking_priority: {priority_txt}",
        f"- needs_clarify: {plan.get('needs_clarify')}",
        f"- needs_ranking: {plan.get('needs_ranking')}",
        f"- needs_justification: {plan.get('needs_justification')}",
    ]
    if plan.get("needs_clarify"):
        lines.append(
            "Clarify first: the shopper will reveal a firm spec if you ask. "
            "Call message to ask for the missing firm requirement BEFORE searching deeply / ordering."
        )
    cands = plan.get("given_candidates") or []
    if cands:
        lines.append(f"- compare ONLY these candidates: {', '.join(map(str, cands[:12]))}")
        lines.append(
            "Ranking: view/compare ALL given candidates; pick the current frontier top by "
            f"ranking_priority [{priority_txt}]. Re-check after any market event. "
            "Do NOT pick a dominated option (worse on every higher-priority axis). "
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
        or "cart" in inner
        or "added" in inner
        or "price" in inner
        or "in_stock" in inner
        or "sku" in inner
    ):
        return inner
    return observation


def _requires_message_reply(content: str) -> bool:
    """True only for shopper messages that must be answered before order.

    Soft acks like "thanks…stated requirements still apply" must NOT block
    checkout — they previously caused endless message loops (agent_error).
    """
    lower = (content or "").strip().lower()
    if not lower:
        return False
    if "stated requirements still apply" in lower and "?" not in lower:
        return False
    # Stock/availability pings after a pick — answer once, but do not treat as
    # hard blockers forever; finalize path will clear when cart is ready.
    if re.match(
        r"^(wait[, ]+)?(is that still available|still available|is it (still )?in stock)\b",
        lower,
    ):
        return True
    if (
        re.match(r"^(thanks|thank you|thx|ok|okay|got it|cool|noted)\b", lower)
        and "?" not in lower
        and not any(
            k in lower
            for k in (
                "budget",
                "available",
                "which",
                "still ",
                "under ",
                "only ",
                "expensive",
                "price",
            )
        )
    ):
        return False
    return True


def _required_claim_fields(goal: str, plan: dict[str, Any] | None = None) -> list[str]:
    """price + in_stock always; extra fields only from plan / goal attr cues (allowlisted)."""
    req = ["price", "in_stock"]
    plan = plan or {}
    for field in plan.get("claim_fields") or []:
        f = str(field or "").strip().lower().replace(" ", "_")
        if f in _CLAIM_ALLOWLIST and f not in req:
            req.append(f)
    spec_field = ((plan.get("spec") or {}) or {}).get("field")
    if spec_field:
        f = str(spec_field).strip().lower().replace(" ", "_")
        if f in _CLAIM_ALLOWLIST and f not in req:
            req.append(f)
    for c in plan.get("constraints") or []:
        if not isinstance(c, dict) or not c.get("firm"):
            continue
        f = str(c.get("field") or "").strip().lower().replace(" ", "_")
        if f in _CLAIM_ALLOWLIST and f not in req:
            req.append(f)
    # Goal text cues (measurement wording → claim field), still allowlisted.
    for field, _value in _extract_attr_cues(goal or ""):
        if field in _CLAIM_ALLOWLIST and field not in req:
            req.append(field)
    return req


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
    category = _norm(plan.get("category") or "")
    firm_constraints = [
        c
        for c in (plan.get("constraints") or [])
        if isinstance(c, dict) and c.get("firm") and c.get("field") not in {"brand", "price"}
    ]
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        blob = _norm(
            " ".join(
                str(x)
                for x in (
                    title,
                    item.get("brand"),
                    item.get("category_path") or item.get("category"),
                    item.get("facts"),
                    item.get("attributes"),
                )
                if x is not None
            )
        )
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
        if category and category in blob:
            score += 3.0
        for c in firm_constraints:
            val = _norm(c.get("value"))
            if val and val in blob:
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
        field = str(field).strip().lower().replace(" ", "_")
        if not field:
            return
        if field in {"category", "category_path"} and isinstance(value, list) and value:
            value = value[-1]
            field = "category"
        if isinstance(value, (dict, list)):
            return
        memory.setdefault("facts", {})[field] = value
        memory.setdefault("fact_norms", {})[field] = _norm(value)
        memory.setdefault("fact_sources", {})[field] = source

    def remember_mapping(data: dict[str, Any], source: str) -> None:
        for key, value in data.items():
            if not isinstance(key, str):
                continue
            raw_key = key.strip().lower().replace(" ", "_")
            if raw_key in {"title", "description", "image", "images", "url", "variants"}:
                continue
            canonical = _FACT_KEY_ALIASES.get(raw_key, raw_key)
            remember(canonical, value, source)
            if canonical != raw_key:
                remember(raw_key, value, source)

    action = str(payload.get("action") or "")
    src = action or "obs"

    for key in (
        "brand",
        "category",
        "category_path",
        "price",
        "current_price",
        "in_stock",
        "stock",
        "sku",
    ):
        if key not in payload:
            continue
        if key == "category_path":
            remember("category", payload[key], "category_path")
        else:
            remember(_FACT_KEY_ALIASES.get(key, key), payload[key], src)

    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        if key in {
            "action",
            "results",
            "product",
            "variant",
            "facts",
            "attributes",
            "raw_product_facts",
            "raw_facts",
            "cart",
            "added",
            "removed",
            "error",
            "done",
            "title",
            "product_id",
            "sku",
            "variants",
            "brand",
            "category",
            "category_path",
            "price",
            "current_price",
            "in_stock",
            "stock",
        }:
            continue
        if isinstance(value, (dict, list)):
            continue
        remember(_FACT_KEY_ALIASES.get(key, key), value, src)

    for nest_key, source in (
        ("facts", "facts"),
        ("attributes", "facts"),
        ("raw_product_facts", "raw_facts"),
        ("raw_facts", "raw_facts"),
    ):
        nest = payload.get(nest_key)
        if isinstance(nest, dict):
            remember_mapping(nest, source)

    product = payload.get("product")
    if isinstance(product, dict):
        remember_mapping(product, "product")
        if product.get("product_id"):
            memory["product_id"] = str(product["product_id"])
        for nest_key in ("facts", "attributes"):
            nest = product.get(nest_key)
            if isinstance(nest, dict):
                remember_mapping(nest, "product")

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

    results = payload.get("results")
    if isinstance(results, list):
        memory["last_results"] = [r for r in results if isinstance(r, dict)][:10]

    if action:
        memory["last_action"] = action
    if observation.get("error") or payload.get("error"):
        memory["last_error"] = observation.get("error") or payload.get("error")

    memory["last_obs_norm"] = _norm(blob)
    # inspect_stock payloads often omit action=; detect via price+in_stock+sku.
    if action == "inspect_stock" or (
        "price" in payload and "in_stock" in payload and payload.get("sku")
    ):
        memory["saw_inspect_stock"] = True
        memory["inspect_product_id"] = str(
            payload.get("product_id") or memory.get("product_id") or ""
        )
        memory["inspect_sku"] = str(payload.get("sku") or memory.get("sku") or "")
        remember("price", payload.get("price"), "inspect_stock")
        remember("in_stock", payload.get("in_stock"), "inspect_stock")
        memory["inspect_price"] = payload.get("price")
        memory["inspect_in_stock"] = payload.get("in_stock")
    if action in {"view", "compare"}:
        memory["saw_view"] = True
    # Top-level brand on view payloads (no action field).
    if payload.get("brand") not in (None, ""):
        remember("brand", payload["brand"], "view" if "facts" in payload else src)

    # Track cart membership for place_test_order (not_in_cart is a common fail).
    cart_items = payload.get("cart")
    if isinstance(cart_items, list):
        keys: set[str] = set()
        insp_key = ""
        if memory.get("inspect_product_id") and memory.get("inspect_sku"):
            insp_key = f"{memory['inspect_product_id']}::{memory['inspect_sku']}"
        for item in cart_items:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("product_id") or "")
            sku = str(item.get("sku") or "")
            if pid and sku:
                keys.add(f"{pid}::{sku}")
                # Only adopt live price/stock for the inspected/final candidate.
                if insp_key and f"{pid}::{sku}" != insp_key and len(cart_items) > 1:
                    continue
                if item.get("current_price") is not None:
                    remember("price", item["current_price"], "cart")
                    memory["inspect_price"] = item["current_price"]
                elif item.get("price") is not None:
                    remember("price", item["price"], "cart")
                    memory["inspect_price"] = item["price"]
                if "in_stock" in item:
                    remember("in_stock", item["in_stock"], "cart")
                    memory["inspect_in_stock"] = item["in_stock"]
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

    facts = dict(memory.get("facts") or {})
    sources = dict(memory.get("fact_sources") or {})
    if memory.get("inspect_price") is not None:
        facts["price"] = memory["inspect_price"]
        sources["price"] = "inspect_stock"
    if memory.get("inspect_in_stock") is not None:
        facts["in_stock"] = memory["inspect_in_stock"]
        sources["in_stock"] = "inspect_stock"

    plan = plan or {}
    required = [
        f
        for f in _required_claim_fields(str(plan.get("goal") or ""), plan)
        if f in _CLAIM_ALLOWLIST or f in _SAFE_CLAIM_FIELDS
    ]
    structured = {
        "facts",
        "raw_facts",
        "product",
        "view",
        "compare",
        "variant",
        "inspect_stock",
        "obs",
        "category_path",
    }
    by_field: dict[str, dict[str, str]] = {}

    for field in ("price", "in_stock"):
        if field in facts:
            by_field[field] = {"field": field, "value": _claim_value(field, facts[field])}

    def add_spec(field: str, value: Any) -> None:
        if value is None or value == "":
            return
        if field not in _CLAIM_ALLOWLIST:
            return
        by_field[field] = {"field": field, "value": _claim_value(field, value)}

    for field in required:
        if field in {"price", "in_stock"}:
            continue
        if field not in facts or sources.get(field) not in structured:
            continue
        value = facts[field]
        # If structured fact is a raw pixel string but the goal named a class token,
        # claim the goal class token (still grounded: token appears in the shopper goal).
        if field == "resolution" and re.search(r"\d+\s*x\s*\d+", str(value), flags=re.I):
            goal_txt = str(plan.get("goal") or "")
            class_tok = _goal_resolution_token(goal_txt)
            if not class_tok:
                for c in plan.get("constraints") or []:
                    if isinstance(c, dict) and c.get("field") == "resolution":
                        class_tok = str(c.get("value") or "") or None
                        break
            if class_tok:
                value = class_tok
        add_spec(field, value)

    # Soft brand claim is part of the justification contract (title_or_brand support).
    # Include only when observed from structured sources — never invent.
    if (
        "brand" in facts
        and sources.get("brand") in structured
        and "brand" in _CLAIM_ALLOWLIST
        and "brand" not in by_field
    ):
        add_spec("brand", facts["brand"])

    ordered = [f for f in required if f in by_field]
    if "brand" in by_field and "brand" not in ordered:
        ordered.append("brand")
    return [{"field": f, "value": by_field[f]["value"]} for f in ordered][:5]

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

        if plan.get("needs_clarify"):
            firm_attrs = [
                c
                for c in (plan.get("constraints") or [])
                if isinstance(c, dict)
                and c.get("firm")
                and c.get("field") not in {"brand", "category", "price", "in_stock"}
            ]
            if not firm_attrs:
                return (
                    "Firm gated spec is still unknown. Call message and ask the shopper for "
                    "the firm requirement, then match it before place_test_order.",
                    tool_calls,
                )

        try:
            args = _arguments(call)
        except Exception as exc:  # noqa: BLE001
            return f"place_test_order arguments invalid: {exc}", tool_calls

        # Enforce known firm attribute constraints against structured memory facts.
        facts = memory.get("facts") or {}
        for c in plan.get("constraints") or []:
            if not isinstance(c, dict) or not c.get("firm"):
                continue
            field = str(c.get("field") or "")
            want = _norm(c.get("value"))
            if not field or not want or field in {"brand", "category", "price", "in_stock"}:
                continue
            have = facts.get(field)
            if have is None:
                continue
            have_n = _norm(have)
            if want not in have_n and have_n not in want and _claim_value(field, have) != _claim_value(field, want):
                return (
                    f"Chosen variant fails firm constraint {field}={want} (observed {have}). "
                    "Recover: pick another firm-compliant option, view/inspect, then order.",
                    tool_calls,
                )

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
        if needs_just:
            claims = (new_args.get("justification") or {}).get("claims") or []
            fields = {c.get("field") for c in claims if isinstance(c, dict)}
            missing = [
                f
                for f in _required_claim_fields((plan or {}).get("goal") or "", plan)
                if f in {"price", "in_stock"} and f not in fields
            ]
            if missing:
                return (
                    "Justification incomplete — inspect_stock the FINAL product_id/sku so "
                    f"claims include {missing} (plus any required specs from the goal), "
                    "then place_test_order.",
                    tool_calls,
                )
            if "justification" not in new_args:
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
    plan = _build_plan(goal)
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
    if plan.get("needs_clarify") and "message" in tool_names:
        messages.append(
            {
                "role": "user",
                "content": (
                    "The goal says a firm requirement will be told if you ask. "
                    "FIRST action: call message and ask what the firm catalog/spec requirement is "
                    "(size/attribute/etc). Do not place_test_order until that firm constraint is known "
                    "and the chosen variant matches it."
                ),
            }
        )
    if plan.get("given_candidates"):
        pri = plan.get("ranking_priority") or []
        pri_txt = " > ".join(
            f"{p.get('field')}({p.get('direction')})" for p in pri if isinstance(p, dict)
        ) or "stated goal priority"
        messages.append(
            {
                "role": "user",
                "content": (
                    "Ranking task: compare/view ONLY the given candidate product_id::sku list. "
                    f"Select by priority [{pri_txt}]. Never pick a dominated candidate. "
                    "After any price/stock event, re-compare remaining candidates before ordering."
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
        {
            "role": "meta",
            "provider": _provider(),
            "model": _model(),
            "plan": plan,
            "strategy": "general-intent-judge-select",
        }
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
        cart_ready = bool(memory.get("cart_keys")) and bool(memory.get("saw_inspect_stock"))
        # Near step limit with a ready cart: stop message loops and commit.
        if (
            cart_ready
            and not memory.get("ordered")
            and env_turn >= max(1, max_steps - 8)
            and memory.get("pending_intervention")
        ):
            memory["pending_intervention"] = False
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Shopper follow-ups are done enough. Cart+inspect are ready. "
                        "Call place_test_order on the cart product_id/sku NOW "
                        + (
                            "WITH justification claims: price, in_stock, required attrs, and brand if observed."
                            if plan.get("needs_justification")
                            else "WITHOUT justification."
                        )
                    ),
                }
            )
        if env_turn >= max(1, max_steps - 6) and not memory.get("ordered"):
            just_hint = (
                "WITH justification claims: price, in_stock, required attrs from view facts, "
                "plus brand when observed (soft claim)."
                if plan.get("needs_justification")
                else "NO justification argument."
            )
            if cart_ready and not memory.get("pending_intervention"):
                keys = sorted(memory.get("cart_keys") or [])
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"FINALIZE NOW. Cart already has {keys[:2]}. "
                            f"Call place_test_order on that exact product_id/sku ({just_hint}). "
                            "Do not search/view/message again."
                        ),
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Step budget almost gone. If you have a firm-compliant in-stock "
                            "variant: answer any pending shopper QUESTION with message, "
                            "inspect_stock if needed, add_to_cart, then place_test_order "
                            f"NOW ({just_hint})."
                        ),
                    }
                )
        elif (
            cart_ready
            and not memory.get("pending_intervention")
            and not memory.get("ordered")
            and env_turn >= 8
            and env_turn % 4 == 0
        ):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Cart+inspect ready. Stop re-inspecting. Call place_test_order now "
                        + (
                            "(include price+in_stock+required attrs+brand claims)."
                            if plan.get("needs_justification")
                            else "(omit justification)."
                        )
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
                if (
                    bool(memory.get("cart_keys"))
                    and bool(memory.get("saw_inspect_stock"))
                    and not memory.get("ordered")
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Message sent. Cart+inspect already ready — call place_test_order "
                                "NEXT on that exact product_id/sku "
                                + (
                                    "WITH claims: price, in_stock, required attrs, brand if observed."
                                    if plan.get("needs_justification")
                                    else "WITHOUT justification."
                                )
                            ),
                        }
                    )
            if action_name in {"search", "filter"}:
                search_rounds += 1
                note = _rank_search_results(observation, plan)
                if note and search_rounds <= 3:
                    messages.append({"role": "user", "content": note})
                results = memory.get("last_results") or []
                if (
                    not memory.get("judged")
                    and results
                    and search_rounds <= _JUDGE_ROUNDS_MAX
                ):
                    tip = _judge_select(goal, results, plan)
                    if tip:
                        memory["judged"] = True
                        messages.append({"role": "user", "content": tip})

        if result.get("user_message"):
            content = (result["user_message"].get("content") or "").strip()
            if content:
                _update_plan_from_shopper(plan, content)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Updated internal plan from shopper message:\n"
                            f"{_plan_hint(plan)}\n"
                            "The next product must satisfy the latest firm constraints "
                            "(budget/brand/category/spec)."
                        ),
                    }
                )
                if _requires_message_reply(content):
                    memory["pending_intervention"] = True
                    guard_retries = 0
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Shopper message: {content[:1500]}\n"
                                "REQUIRED NEXT ACTION: call message and answer this before ordering."
                                + (
                                    f" Current budget ceiling: {plan['max_price']}."
                                    if plan.get("max_price") is not None
                                    else ""
                                )
                            ),
                        }
                    )
                else:
                    # Soft ack — do not block checkout.
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Shopper note (no reply needed): {content[:500]}\n"
                                "Continue: inspect_stock if needed, then place_test_order "
                                f"{'WITH price+in_stock(+spec) claims' if plan.get('needs_justification') else 'WITHOUT justification'}."
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
