

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# Live allowlist (Sep 2026 docs). Prefer OpenRouter when provider is openrouter.
_DEFAULT_MODELS = {
    "openrouter": "deepseek/deepseek-v3.2",
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
}
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

# 0824-X_P L_Specs knobs (strategy, not suite answers).
_RRF_K = 60
_RRF_WEIGHT = 0.15
_SELF_CONSISTENCY_GAP = 1.5
_ENABLE_DUAL_PATH = True
_LOW_JUDGE_SCORE = 6.0
_FAST_ACCEPT_SCORE = 8.0
_FAST_ACCEPT_COV = 0.5
_FUZZY_PRICE_RATIO = 0.08
_FUZZY_PRICE_ABS = 1.0
_TEMP_STEP = 0.3
_TEMP_MAX = 1.2
_PRODUCT_SHORTLIST = 10
_SELECT_CANDIDATES = 30
_LLM_RETRY_MAX = 3
_VOUCHER_WINDOW_EPS = 0.5
_COVERAGE_MATCH_FLOOR = 0.8
_RESCUE_SCORE_FLOOR = 0.55
_RELAX_SCORE_GAP = 0.6
_MAX_GATE_SWAPS = 2
_SCORE_N_THRES = 0.4

# PatchConfig spirit from 0824-X_P.
_RELAX_ON_ZERO_RECALL = True
_ALWAYS_SUBMIT = True
_HARD_CONSTRAINT_GATE = True
_VERIFY_PICK = True
_LITERAL_GATE = True

# Role model chains (0824 L_Gateway.get_role_llm_priorities spirit).
_OPENROUTER_SELECT = (
    "z-ai/glm-5.2",
    "moonshotai/kimi-k2.6",
    "openai/gpt-oss-120b",
    "z-ai/glm-5.1",
    "google/gemma-4-31b-it",
    "deepseek/deepseek-v3.2",
)
_OPENROUTER_ALT_SELECT = (
    "z-ai/glm-5.2",
    "moonshotai/kimi-k2.6",
    "openai/gpt-oss-120b",
    "z-ai/glm-5.1",
)
_OPENROUTER_DUAL = (
    "z-ai/glm-5.2",
    "moonshotai/kimi-k2.6",
    "openai/gpt-oss-120b",
)
_CHUTES_SELECT = (
    "moonshotai/Kimi-K2.6-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "Qwen/Qwen3.6-27B-TEE",
    "zai-org/GLM-5.2-TEE",
    "google/gemma-4-31B-turbo-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
)
_CHUTES_ALT_SELECT = (
    "moonshotai/Kimi-K2.6-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "Qwen/Qwen3.6-27B-TEE",
)
_CHUTES_DUAL = (
    "moonshotai/Kimi-K2.6-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
)

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

_INVALID_IDS = frozenset({"", "none", "null", "0", "undefined", "n/a"})

# Generic English function words only — not suite-specific product vocab.
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "with",
        "from",
        "that",
        "this",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "is",
        "it",
        "i",
        "me",
        "my",
        "am",
        "im",
        "be",
        "can",
        "has",
        "have",
        "will",
        "would",
        "should",
        "please",
        "want",
        "need",
        "get",
        "buy",
        "find",
        "show",
        "looking",
        "also",
        "but",
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
        "prefer",
        "preferred",
        "ideally",
        "stock",
        "available",
        "firm",
        "soft",
        "task",
        "rules",
        "must",
        "always",
        "never",
        "php",
        "pesos",
        "same",
        "shop",
        "store",
        "seller",
        "voucher",
        "coupon",
        "discount",
        "about",
        "after",
        "before",
        "between",
        "within",
        "per",
        "as",
    }
)

# Generic number/word synonyms only (0824 L_Specs.SYNONYMS / P_EXT_SEARCH_UNITS spirit).
_SYNONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("1", "one"),
    ("2", "two"),
    ("3", "three"),
    ("4", "four"),
    ("5", "five"),
    ("6", "six"),
    ("7", "seven"),
    ("8", "eight"),
    ("9", "nine"),
    ("10", "ten"),
    ("1", "single"),
    ("2", "double"),
    ("3", "triple"),
    ("pcs", "piece"),
    ("pcs", "pack"),
    ("waterproof", "water-resistant"),
    ("shockproof", "shock-resistant"),
)

_RX_INJECTION = re.compile(
    r"(?is)(?:^|\n)\s*(?:system\s*:|ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions|"
    r"you\s+are\s+now\s+|developer\s+mode\s*|jailbreak\s*:|do\s+not\s+follow\s+the\s+rules)"
)
_RX_ROLE_PLAY = re.compile(
    r"(?is)\b(?:act\s+as\s+(?:DAN|an?\s+unrestricted)|pretend\s+you\s+have\s+no\s+restrictions)\b"
)
_RX_PRICE_OVER = re.compile(
    r"(?:greater|more|over|above|at\s+least|minimum|min|>=)\s*(?:than\s*)?(\d+(?:\.\d+)?)",
    re.I,
)
_RX_PRICE_UNDER = re.compile(
    r"(?:less|under|below|at\s+most|maximum|max|<=)\s*(?:than\s*)?(\d+(?:\.\d+)?)",
    re.I,
)
_RX_PRICE_BETWEEN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:to|and|-)\s*(\d+(?:\.\d+)?)\s*(?:pesos|php|usd|dollars?)?",
    re.I,
)
# 0824 L_Specs voucher/shop classifiers (public goal text only).
_RX_VOUCHER = re.compile(
    r"\bmy\s+budget\s+is\b|"
    r"\bbudget\s+of\s+\d+|"
    r"\bwithin\s+(?:my\s+)?budget\b|"
    r"\b(?:percent(?:age)?|fixed)\s+discount\b|"
    r"\bvoucher\b.*\b(?:fixed|percentage)\s+discount\b|"
    r"\b(?:fixed|percentage)\s+discount\b.*\bvoucher\b|"
    r"\bvoucher\b.*\bthreshold\b|"
    r"\b(?:voucher|coupon|promo(?:\s+code)?)\b|"
    r"\bspend\s+at\s+least\b",
    re.I | re.S,
)
_RX_SHOP = re.compile(
    r"\b(?:same|one|single|identical)\s+shops?\b|"
    r"\bfrom\s+(?:the\s+)?same\s+(?:store|seller|shop)\b|"
    r"\ball\s+(?:from|in)\s+(?:the\s+)?same\s+(?:store|seller|shop)s?\b|"
    r"\b(?:look(?:ing)?(?:\s+for)?|find|show)\b(?:\s+\w+){0,10}\s+shops?\b",
    re.I,
)

_SYSTEM_POLICY = """You are a shopping agent in a generated environment. Use ONLY the supplied tools.

0824-X_P pipeline (classify → search → RRF/fit → dual-path select → verify → commit):
1. Route the goal as product, same-shop multi-item, or voucher/budget basket.
2. Search/filter broadly; view/compare several distinct candidates.
3. Score with atom coverage + RRF search-rank fusion; prefer structured facts over SEO titles.
4. Dual-path: primary judge + independent alt judge; dual_compare when they disagree.
5. inspect_stock on the exact product_id/sku → add_to_cart → answer shopper
   messages → place_test_order.

Literal matching (MATCH_RULES spirit — non-negotiable):
- Numbers, identifiers, codes, variant names, pack counts must appear literally
  in product facts/variants. A range that merely contains a number is NOT a match.
- Soft descriptive words may paraphrase; firm identifiers may not.
- Prefer structured facts / category_path / variants over noisy SEO titles.
- Exact variant: order the sku that satisfies the firm attribute, not a sibling.

Evidence preference when several items all-match:
- Cleaner attribute evidence (fewer extra tokens) beats messy compound values.
- Dedicated attribute keys beat catch-all fields; structured beats title-only.
- Narrow exact values beat ranges; original search rank is only a final tie-break.

Firm vs soft:
- Firm requirements (category, required brand/spec, budget, in-stock, pack/code)
  always beat soft preferences.
- Soft brand/color only among fully firm-compliant options.

Hard gates:
- Always inspect_stock on the exact product_id/sku before place_test_order.
- After ANY real shopper question, call message and answer BEFORE ordering.
- Ranking tasks: compare ONLY given candidates; follow stated priority order.
- Recovery: if price/stock/preferred variant fails, remove bad cart lines and switch.
- Finish with place_test_order. Do not wait for permission once verified.

Justification:
- Ignore Task-rules boilerplate about "asks for a reason".
- Include justification only if the shopper goal itself asks why/justify/walk-through.
- Otherwise OMIT justification entirely (extra claims zero the score).
- When required: claim only price + in_stock from latest inspect_stock, plus
  brand/screen_size/resolution/refresh_rate if the goal mentions them AND you
  observed them as structured facts. Never invent fields. Exact observed values only.

Continue until an observation reports done=true."""

_JUDGE_PROMPT = """Select the single best shopping candidate for the query.
Trust attributes/facts/variants over title SEO text.
Numbers/ids/variant codes must appear LITERALLY (MATCH_RULES).
Every firm constraint (budget, in-stock, category, required brand/spec, pack/code)
must hold. Soft brand is secondary. Prefer complete constraint coverage.
Return JSON only:
{"best_product_id":"<id or null>","best_sku":"<sku or null>","reason":"1 sentence","relevance_score":0-10,"all_match":true|false}
"""

_ALT_JUDGE_PROMPT = """You are a STRICT independent second opinion (0824 Path-B / PV_SELECT_ALT).
Path-A already proposed a candidate; give your own pick from the candidates.
Priority:
1. Reject any candidate missing a hard constraint (color/pack/model/size/brand/negation).
2. Prefer structured attributes/facts/variants over SEO titles.
3. Prefer the most direct attribute match (fewest extra tokens).
4. If none fully match, return null (do not force a weak guess).
Return JSON only:
{"best_product_id":"<id or null>","best_sku":"<sku or null>","reason":"1 sentence","relevance_score":0-10,"all_match":true|false}
"""

_DUAL_COMPARE_PROMPT = """Choose which of two candidate answers better satisfies the shopping request.
Candidate A is primary pipeline; Candidate B is alternate pipeline.
Rules:
1. Prefer the candidate that satisfies MORE required constraints with clear evidence.
2. If both satisfy all constraints, prefer the more exact / direct match.
3. If still tied, prefer Candidate A.
4. Never invent product details; only use the provided JSON.
Return JSON only: {"best":"A"|"B","reason":"one sentence"}
"""

_VERIFY_PROMPT = """Verify whether this candidate truly satisfies EVERY firm constraint.
Literal tokens and numeric/pack/model codes must appear in product facts.
Return JSON only:
{"ok":true|false,"missing":["constraint",...],"reason":"1 sentence"}
"""


def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "openrouter").strip().lower()


def _model_chain() -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    if _provider() == "chutes":
        return list(_CHUTES_FALLBACKS)
    return list(_OPENROUTER_FALLBACKS)


def _role_models(role: str) -> list[str]:
    """0824 L_Gateway.get_role_llm_priorities — role-preferred then general chain."""
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    chutes = _provider() == "chutes"
    if role == "select":
        preferred = _CHUTES_SELECT if chutes else _OPENROUTER_SELECT
    elif role == "alt_select":
        preferred = _CHUTES_ALT_SELECT if chutes else _OPENROUTER_ALT_SELECT
    elif role == "dual_compare":
        preferred = _CHUTES_DUAL if chutes else _OPENROUTER_DUAL
    else:
        preferred = ()
    out: list[str] = []
    for m in list(preferred) + _model_chain():
        if m not in out:
            out.append(m)
    return out


def _model() -> str:
    return _model_chain()[0]


def _temp_for(role: str, attempt: int = 0) -> float:
    base = {
        "parse": 0.0,
        "score": 0.4,
        "judge": 0.5,
        "select": 0.4,
        "alt_select": 0.55,
        "dual_compare": 0.3,
        "verify": 0.2,
        "vote": 0.3,
        "act": 0.0,
    }.get(role, 0.0)
    return min(_TEMP_MAX, base + max(0, attempt) * _TEMP_STEP)


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _norm(value: Any) -> str:
    text = str(value).strip().lower().replace(",", "")
    return re.sub(r"\s+", " ", text)


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "null", "none", "n/a", "undefined", "nan"}
    if isinstance(value, (list, dict, tuple, set)) and len(value) == 0:
        return True
    return False


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if _is_nullish(value):
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        m = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
        return float(m.group(0)) if m else default
    except (TypeError, ValueError):
        return default


def _sanitize_product_id(value: Any) -> str | None:
    pid = str(value or "").strip()
    if pid.lower() in _INVALID_IDS:
        return None
    return pid or None


def _sanitize_goal(query: str) -> str:
    q = str(query or "")
    q = _RX_INJECTION.sub(" ", q)
    q = _RX_ROLE_PLAY.sub(" ", q)
    q = re.sub(r"[ \t]{2,}", " ", q)
    q = re.sub(r"\n{3,}", "\n\n", q)
    return q.strip() or str(query or "").strip()


def _shopper_goal(query: str) -> str:
    parts = re.split(r"\n\s*Task rules:\s*", query, maxsplit=1, flags=re.IGNORECASE)
    return parts[0].strip()


def _needs_justification(goal: str) -> bool:
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


def _parse_json_content(content: str) -> dict[str, Any] | None:
    cleaned = re.sub(
        r"<think(?:ing)?>.*?</think(?:ing)?>",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"```json?\s*|```\s*", "", cleaned).strip()
    try:
        out = json.loads(cleaned)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            out = json.loads(cleaned[start : end + 1])
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            return None
    return None


def _combo_route(goal: str) -> str:
    """0824-X_P SoftRouter / L_Support.guess_challenge from public goal text only."""
    q = goal.lower()
    if _RX_VOUCHER.search(q):
        return "voucher"
    if _RX_SHOP.search(q):
        return "shop"
    price_hits = len(
        re.findall(
            r"(?:under|below|above|over|between|from)\s*[\d,]+|"
            r"[\d,]+\s*-\s*[\d,]+\s*(?:php|pesos)?",
            q,
        )
    )
    if price_hits >= 2 or re.search(
        r"\b(?:both|and also|first[,:]|second[,:]|third[,:])\b", q
    ):
        return "multi_item"
    return "product"


def _syn_expand(token: str) -> list[str]:
    out = [token]
    for a, b in _SYNONYM_PAIRS:
        if a == token and b not in out:
            out.append(b)
        if b == token and a not in out:
            out.append(a)
    return out


def _price_range_str(min_price: float | None, max_price: float | None) -> str | None:
    if min_price is None and max_price is None:
        return None
    lo = f"{min_price:g}" if min_price is not None else ""
    hi = f"{max_price:g}" if max_price is not None else ""
    return f"{lo}-{hi}"


def _sanitize_price_range(price_range: str | None) -> str | None:
    if _is_nullish(price_range):
        return None
    s = re.sub(r"[^\d\-.]", "", str(price_range).strip())
    if not s or s == "-":
        return None
    left_raw, sep, right_raw = s.partition("-")
    if not sep:
        return None
    left, right = left_raw.strip(), right_raw.strip()
    if left and right:
        lo = _coerce_float(left)
        hi = _coerce_float(right)
        if hi < lo:
            lo, hi = hi, lo
        return f"{lo:g}-{hi:g}"
    if left:
        return f"{_coerce_float(left):g}-"
    if right:
        return f"-{_coerce_float(right):g}"
    return None


def _parse_price_bounds(price_range: str | None) -> tuple[float | None, float | None]:
    cleaned = _sanitize_price_range(price_range)
    if not cleaned:
        return None, None
    left_raw, _, right_raw = cleaned.partition("-")
    lo = _coerce_float(left_raw) if left_raw.strip() else None
    hi = _coerce_float(right_raw) if right_raw.strip() else None
    return lo, hi


def _widen_price(
    min_price: float | None, max_price: float | None
) -> tuple[float | None, float | None]:
    lo, hi = min_price, max_price
    if lo is not None:
        lo = max(0.0, lo * (1.0 - _FUZZY_PRICE_RATIO))
    if hi is not None:
        hi = hi * (1.0 + _FUZZY_PRICE_RATIO)
    return lo, hi


def _price_fits(
    price: Any,
    min_price: float | None,
    max_price: float | None,
    *,
    fuzzy: bool = True,
) -> bool:
    try:
        pv = float(str(price).replace(",", ""))
    except (TypeError, ValueError):
        return False
    lo, hi = min_price, max_price
    if lo is None and hi is None:
        return True
    if fuzzy and lo is not None and hi is not None:
        span = max(hi - lo, 0.0)
        pad = max(_FUZZY_PRICE_ABS, span * _FUZZY_PRICE_RATIO)
        flo, fhi = lo - pad, hi + pad
    elif fuzzy:
        flo = (lo - _FUZZY_PRICE_ABS) if lo is not None else None
        fhi = (hi + _FUZZY_PRICE_ABS) if hi is not None else None
    else:
        flo, fhi = lo, hi
    if flo is not None and pv < flo:
        return False
    if fhi is not None and pv > fhi:
        return False
    return True


def _heuristic_parse(goal: str) -> dict[str, Any]:
    lower = goal.lower()
    lane = _combo_route(goal)
    plan: dict[str, Any] = {
        "lane": lane,
        "keywords": "",
        "atoms": [],
        "rescue_queries": [],
        "literal_tokens": [],
        "category": None,
        "brand_soft": None,
        "brand_firm": None,
        "max_price": None,
        "min_price": None,
        "price_range": None,
        "spec": {"field": None, "value": None},
        "must_in_stock": True,
        "needs_ranking": any(
            w in lower
            for w in ("rank", "highest", "compare only", "among ties", "priority")
        ),
        "needs_justification": False,
        "given_candidates": re.findall(r"\b\d{6,}::[\w.-]+\b", goal),
        "goal": goal,
        "same_shop": lane == "shop",
        "voucher_like": lane == "voucher",
        "gate_swaps": 0,
    }

    m = re.search(
        r"(?:at or under|at or below|up to|under|below|<=|max(?:imum)?)\s*"
        r"([\d,]+(?:\.\d+)?)\s*(?:php|pesos)?",
        lower,
    )
    if m:
        plan["max_price"] = float(m.group(1).replace(",", ""))

    m = re.search(
        r"(?:at or above|at least|over|above|>=|min(?:imum)?)\s*"
        r"([\d,]+(?:\.\d+)?)\s*(?:php|pesos)?",
        lower,
    )
    if m and plan["min_price"] is None:
        plan["min_price"] = float(m.group(1).replace(",", ""))

    m = re.search(
        r"(?:between|from)\s*([\d,]+(?:\.\d+)?)\s*(?:and|to|-)\s*([\d,]+(?:\.\d+)?)",
        lower,
    )
    if m:
        plan["min_price"] = float(m.group(1).replace(",", ""))
        plan["max_price"] = float(m.group(2).replace(",", ""))

    # Fallback from 0824 fuzzy boundary regexes.
    if plan["min_price"] is None and plan["max_price"] is None:
        mb = _RX_PRICE_BETWEEN.search(goal)
        if mb:
            plan["min_price"] = float(mb.group(1))
            plan["max_price"] = float(mb.group(2))
        else:
            mu = _RX_PRICE_UNDER.search(goal)
            if mu:
                plan["max_price"] = float(mu.group(1))
            mo = _RX_PRICE_OVER.search(goal)
            if mo:
                plan["min_price"] = float(mo.group(1))

    plan["price_range"] = _price_range_str(plan["min_price"], plan["max_price"])

    m = re.search(
        r"(?:prefer|preferably|ideally[^.]{0,40}prefer(?:s|red)?)\s+"
        r"([A-Za-z][A-Za-z0-9 &.-]{1,40})",
        goal,
        flags=re.IGNORECASE,
    )
    if m:
        brand = re.split(r"\b(?:but|and|,)\b", m.group(1).strip(" .,"), maxsplit=1)[0].strip()
        if brand and brand.lower() not in _STOP:
            plan["brand_soft"] = brand

    m = re.search(
        r"\b(?:need|want|get)\s+(?:a|an)\s+([A-Z][A-Za-z0-9.-]{1,30})\s+"
        r"([A-Za-z][A-Za-z0-9-]{2,30})\b",
        goal,
    )
    if m and "prefer" not in lower:
        plan["brand_firm"] = m.group(1)
        if not plan.get("category"):
            plan["category"] = m.group(2)

    for field, pattern in (
        ("screen_size", r"(\d+(?:\.\d+)?)\s*-?\s*inch"),
        ("refresh_rate", r"(\d+)\s*hz"),
        ("resolution", r"\b(\d+p|4k|8k|1080|720|1440)\b"),
        ("ram", r"(\d+)\s*gb\s*ram|\bram\s*(\d+)\s*gb"),
        ("pack", r"(\d+)\s*(?:pcs|pc|pack|set)\b"),
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

    literals = re.findall(r"\b(?=[a-z0-9-]*\d)[a-z0-9][a-z0-9.-]{1,24}\b", lower)
    plan["literal_tokens"] = list(dict.fromkeys(literals))[:12]

    quoted = re.findall(r'"([^"]{2,80})"', goal)
    head = re.split(r"[.\n]", goal, maxsplit=1)[0]
    head = re.sub(
        r"(?:at or under|at or below|up to|under|below|at or above|at least|over|above)\s*"
        r"[\d,]+(?:\.\d+)?\s*(?:php|pesos)?",
        " ",
        head,
        flags=re.IGNORECASE,
    )
    corpus = " ".join([head, *quoted])
    tokens = [
        t
        for t in re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", corpus.lower())
        if len(t) > 1 and t not in _STOP and not re.fullmatch(r"\d+(?:\.\d+)?", t)
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
    plan["atoms"] = list(dict.fromkeys(tokens[:16]))

    rescues: list[str] = []
    cat_s = str(plan.get("category") or "")
    brand_s = str(plan.get("brand_firm") or plan.get("brand_soft") or "")
    if cat_s and brand_s:
        rescues.append(f"{brand_s} {cat_s}")
    if cat_s and spec_val:
        rescues.append(f"{cat_s} {spec_val}")
    if brand_s and spec_val:
        rescues.append(f"{brand_s} {spec_val}")
    if cat_s:
        rescues.append(cat_s)
    words = keywords.split()
    if len(words) > 2:
        rescues.append(" ".join(words[:3]))
        rescues.append(" ".join(words[:2]))
    if len(words) > 3:
        rescues.append(" ".join(words[1:4]))
    wlo, whi = _widen_price(plan.get("min_price"), plan.get("max_price"))
    if wlo is not None or whi is not None:
        plan["widened_price_range"] = _price_range_str(wlo, whi)
    plan["rescue_queries"] = list(dict.fromkeys(q for q in rescues if q.strip()))[:5]
    return plan


def _plan_hint(plan: dict[str, Any]) -> str:
    lines = [
        "Internal 0824-X_P plan (Task rules win on conflict):",
        f"- lane: {plan.get('lane')}",
        f"- keywords/atoms: {plan.get('keywords')} | {plan.get('atoms')}",
        f"- literal tokens (must appear in facts): {plan.get('literal_tokens')}",
        f"- rescue searches: {plan.get('rescue_queries')}",
        f"- category: {plan.get('category')}",
        f"- firm brand: {plan.get('brand_firm')} | soft brand: {plan.get('brand_soft')}",
        f"- price: {plan.get('price_range')} (fuzzy pad {_FUZZY_PRICE_ABS}/{_FUZZY_PRICE_RATIO})",
        f"- widened salvage band: {plan.get('widened_price_range')}",
        f"- key spec: {plan.get('spec')}",
        f"- same_shop: {plan.get('same_shop')} | voucher_like: {plan.get('voucher_like')}",
        f"- needs_ranking: {plan.get('needs_ranking')}",
        f"- needs_justification: {plan.get('needs_justification')}",
        "Workflow: classify → search/filter → RRF+fit shortlist → dual-path "
        "(primary + alt + dual_compare) → verify-pick → inspect_stock → "
        "add_to_cart → message if asked → place_test_order.",
    ]
    if plan.get("same_shop") or plan.get("lane") == "shop":
        lines.append(
            "Same-shop multi-item: every selected line must share one shop_id/seller."
        )
    if plan.get("voucher_like") or plan.get("lane") == "voucher":
        lines.append(
            f"Voucher/budget basket: keep cart total inside budget (±{_VOUCHER_WINDOW_EPS})."
        )
    cands = plan.get("given_candidates") or []
    if cands:
        lines.append(f"- compare ONLY these candidates: {', '.join(map(str, cands[:12]))}")
        lines.append(
            "Ranking: view/compare all given candidates; follow stated priority; "
            "do not pick a dominated option. Order WITHOUT justification unless required."
        )
    lines.append(
        "ALWAYS answer shopper questions with message before ordering. "
        "Omit justification unless needs_justification is true."
    )
    return "\n".join(lines)


def _update_plan_from_shopper(plan: dict[str, Any], content: str) -> None:
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
            plan["price_range"] = _price_range_str(plan.get("min_price"), plan["max_price"])
            wlo, whi = _widen_price(plan.get("min_price"), plan["max_price"])
            plan["widened_price_range"] = _price_range_str(wlo, whi)
        except ValueError:
            pass


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
    lower = (content or "").strip().lower()
    if not lower:
        return False
    if "stated requirements still apply" in lower and "?" not in lower:
        return False
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
        "shop_id",
        "seller",
    ):
        if key in payload:
            slim[key] = payload[key]
    return json.dumps(slim, ensure_ascii=False, default=str)[:limit]


def _item_blob(item: dict[str, Any]) -> str:
    return _norm(
        json.dumps(
            {
                "title": item.get("title"),
                "brand": item.get("brand"),
                "sku": item.get("sku"),
                "facts": item.get("facts"),
                "attributes": item.get("attributes"),
                "category_path": item.get("category_path"),
                "variants": item.get("variants"),
            },
            ensure_ascii=False,
            default=str,
        )
    )


def _atom_hit(atom: str, blob: str) -> float:
    """FitScore-style atom match with synonym expand. Returns 0..1+bonus."""
    cands = _syn_expand(atom)
    best = 0.0
    for c in cands:
        if not c:
            continue
        if re.search(rf"\b{re.escape(c)}\b", blob):
            best = max(best, 1.0)
        elif c in blob:
            best = max(best, 0.7)
        elif any(ch.isdigit() for ch in c) and c in blob:
            best = max(best, 1.05)
    return best


def _literal_coverage(item: dict[str, Any], plan: dict[str, Any]) -> float:
    """0824 L_Support.literal_coverage — hard tokens must appear in product blob."""
    lits = plan.get("literal_tokens") or []
    if not lits:
        return 1.0
    blob = _item_blob(item)
    hit = sum(1 for lit in lits if lit and re.search(rf"\b{re.escape(lit)}\b", blob))
    return hit / max(1, len(lits))


def _fit_detail(item: dict[str, Any], plan: dict[str, Any]) -> dict[str, float]:
    """Per-constraint detail scores (0824 Weight.detail / FitScore spirit)."""
    blob = _item_blob(item)
    title = _norm(item.get("title") or "")
    detail: dict[str, float] = {}
    for i, atom in enumerate(plan.get("atoms") or []):
        key = f"atom_{i}_{atom}"
        detail[key] = max(_atom_hit(atom, blob), 0.85 * _atom_hit(atom, title))

    for lit in plan.get("literal_tokens") or []:
        detail[f"lit_{lit}"] = 1.0 if re.search(rf"\b{re.escape(lit)}\b", blob) else 0.0

    brand_firm = _norm(plan.get("brand_firm") or "")
    if brand_firm:
        brand = _norm(item.get("brand") or "")
        detail["brand_firm"] = (
            1.0 if brand == brand_firm or brand_firm in title or brand_firm in blob else 0.0
        )

    brand_soft = _norm(plan.get("brand_soft") or "")
    if brand_soft and not brand_firm:
        brand = _norm(item.get("brand") or "")
        detail["brand_soft"] = (
            1.0 if brand == brand_soft or brand_soft in title else 0.5
        )

    cat = _norm(plan.get("category") or "")
    if cat:
        path = _norm(item.get("category_path") or item.get("category") or "")
        detail["category"] = 1.0 if cat in path or cat in blob else 0.0

    spec = plan.get("spec") or {}
    if spec.get("value"):
        sv = _norm(spec["value"])
        detail["spec"] = 1.0 if sv in blob or sv in title else 0.0

    price = item.get("price")
    if plan.get("min_price") is not None or plan.get("max_price") is not None:
        if _price_fits(price, plan.get("min_price"), plan.get("max_price"), fuzzy=False):
            detail["price"] = 1.0
        elif _price_fits(price, plan.get("min_price"), plan.get("max_price"), fuzzy=True):
            detail["price"] = 0.65
        else:
            detail["price"] = 0.0

    if item.get("in_stock") is True:
        detail["in_stock"] = 1.0
    elif item.get("in_stock") is False:
        detail["in_stock"] = 0.0
    else:
        detail["in_stock"] = 0.5

    return detail


def _constraint_coverage(detail: dict[str, float], min_score: float = _COVERAGE_MATCH_FLOOR) -> float:
    if not detail:
        return 0.0
    matched = sum(1 for v in detail.values() if v >= min_score)
    return matched / len(detail)


def _fit_score(item: dict[str, Any], plan: dict[str, Any], rank: int) -> tuple[float, dict[str, float]]:
    """0824 score_n + RRF: coverage fused with reciprocal search rank."""
    detail = _fit_detail(item, plan)
    coverage = _constraint_coverage(detail)
    lit_cov = _literal_coverage(item, plan)
    score = sum(detail.values())
    score += 4.0 * coverage
    score += 3.0 * lit_cov
    if coverage < _RESCUE_SCORE_FLOOR or lit_cov < _SCORE_N_THRES:
        score -= 5.0
    # RRF: 1/(k+rank) fused with coverage (L_Engine key spirit).
    rrf = 1.0 / (_RRF_K + rank)
    score += _RRF_WEIGHT * rrf * 100.0
    # Mild search-rank prior as final soft nudge.
    score += max(0.0, 1.5 - 0.08 * rank)
    return score, detail


def _should_run_dual_path(scores: list[float], coverages: list[float] | None = None) -> bool:
    if not _ENABLE_DUAL_PATH or len(scores) < 2:
        return False
    ordered = sorted(scores, reverse=True)
    if ordered[0] - ordered[1] < _SELF_CONSISTENCY_GAP:
        return True
    if coverages and len(coverages) >= 2:
        return abs(coverages[0] - coverages[1]) < 0.12
    return False


def _dual_path_pick(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
    *,
    heuristic_gap: float,
) -> dict[str, Any] | None:
    if first and not second:
        return first
    if second and not first:
        return second
    if not first or not second:
        return None
    a = _sanitize_product_id(first.get("best_product_id"))
    b = _sanitize_product_id(second.get("best_product_id"))
    if a and a == b:
        return first
    if heuristic_gap >= _SELF_CONSISTENCY_GAP:
        return first
    return None  # force dual_compare


def _rank_search_results(observation: dict[str, Any], plan: dict[str, Any]) -> str | None:
    payload = _unwrap_payload(observation)
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        if _RELAX_ON_ZERO_RECALL and plan.get("rescue_queries"):
            return (
                "Empty recall pool. Relax: run rescue searches "
                f"{plan.get('rescue_queries')}"
                + (
                    f" and/or widen price filter to {plan.get('widened_price_range')}"
                    if plan.get("widened_price_range")
                    else ""
                )
                + ", then re-score with RRF + literal coverage."
            )
        return None

    scored: list[tuple[float, dict[str, Any], dict[str, float]]] = []
    for i, item in enumerate(results[:_SELECT_CANDIDATES]):
        if not isinstance(item, dict):
            continue
        sc, detail = _fit_score(item, plan, i)
        scored.append((sc, item, detail))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("product_id", ""))))
    shortlist = scored[:_PRODUCT_SHORTLIST]

    lines = [
        "0824 RRF+fit shortlist (literal coverage; trust structured fields over titles):",
    ]
    for score, item, detail in shortlist[:6]:
        cov = _constraint_coverage(detail)
        lit = _literal_coverage(item, plan)
        lines.append(
            f"- {score:.2f} cov={cov:.2f} lit={lit:.2f} id={item.get('product_id')} "
            f"sku={item.get('sku')} brand={item.get('brand')} "
            f"price={item.get('price')} stock={item.get('in_stock')} "
            f"title={str(item.get('title') or '')[:90]}"
        )

    if shortlist:
        top_score = shortlist[0][0]
        gap = (
            shortlist[0][0] - shortlist[1][0]
            if len(shortlist) > 1
            else 99.0
        )
        top_cov = _constraint_coverage(shortlist[0][2])
        if top_score / max(1.0, len(plan.get("atoms") or [1]) + 3) * 10 < _LOW_JUDGE_SCORE:
            lines.append(
                "Low fit confidence — salvage with rescue searches / widened price, "
                "then re-judge dual-path."
            )
            if plan.get("rescue_queries"):
                lines.append(f"Rescue queries: {plan.get('rescue_queries')}")
            if plan.get("widened_price_range"):
                lines.append(f"Widened price band: {plan.get('widened_price_range')}")
        elif (
            top_score >= _FAST_ACCEPT_SCORE
            and top_cov >= _FAST_ACCEPT_COV
            and gap >= _SELF_CONSISTENCY_GAP
        ):
            top = shortlist[0][1]
            lines.append(
                f"Fast-accept candidate: product_id={top.get('product_id')} "
                f"sku={top.get('sku')} (high fit + clear gap)."
            )
        elif _should_run_dual_path(
            [s for s, _, _ in shortlist[:3]],
            [_constraint_coverage(d) for _, _, d in shortlist[:3]],
        ):
            ids = [str(it.get("product_id")) for _, it, _ in shortlist[:3]]
            lines.append(f"Tight race — run dual-path select among: {', '.join(ids)}")
    return "\n".join(lines)


def _llm_json(
    system: str,
    user: str,
    *,
    role: str = "judge",
    models: list[str] | None = None,
) -> dict[str, Any] | None:
    chain = models or _role_models(role)[:3] or _model_chain()[:2]
    for attempt, model in enumerate(chain):
        inference = _proxy.post(
            "/inference/chat/completions",
            json_data={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": _temp_for(role, attempt),
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


def _candidate_rows(
    results: list[dict[str, Any]], plan: dict[str, Any], limit: int = 8
) -> list[dict[str, Any]]:
    rows = []
    for i, item in enumerate(results[:limit]):
        sc, detail = _fit_score(item, plan, i)
        rows.append(
            {
                "product_id": item.get("product_id"),
                "sku": item.get("sku"),
                "title": str(item.get("title") or "")[:140],
                "brand": item.get("brand"),
                "price": item.get("price"),
                "in_stock": item.get("in_stock"),
                "facts": item.get("facts"),
                "fit_score": round(sc, 3),
                "coverage": round(_constraint_coverage(detail), 3),
                "literal_coverage": round(_literal_coverage(item, plan), 3),
            }
        )
    return rows


def _llm_select(goal: str, results: list[dict[str, Any]], plan: dict[str, Any]) -> str | None:
    """0824 dual-path: primary select → alt select → dual_compare → verify_pick."""
    if not results:
        return None
    rows = _candidate_rows(results, plan)
    payload = {
        "request": goal[:1200],
        "lane": plan.get("lane"),
        "literal_tokens": plan.get("literal_tokens"),
        "firm": {
            "category": plan.get("category"),
            "brand_firm": plan.get("brand_firm"),
            "price_range": plan.get("price_range"),
            "spec": plan.get("spec"),
            "same_shop": plan.get("same_shop"),
        },
        "candidates": rows,
    }
    user = json.dumps(payload, ensure_ascii=False)
    first = _llm_json(_JUDGE_PROMPT, user, role="select")
    scores = [float(r.get("fit_score") or 0) for r in rows]
    coverages = [float(r.get("coverage") or 0) for r in rows]
    gap = (scores[0] - scores[1]) if len(scores) > 1 else 99.0

    second = None
    if _should_run_dual_path(scores, coverages) or (
        first and first.get("all_match") is False
    ):
        second = _llm_json(_ALT_JUDGE_PROMPT, user, role="alt_select")

    pick = _dual_path_pick(first, second, heuristic_gap=gap)
    if pick is None and first and second:
        vote = _llm_json(
            _DUAL_COMPARE_PROMPT,
            json.dumps(
                {
                    "query": goal[:800],
                    "A": first,
                    "B": second,
                    "candidates": rows[:6],
                },
                ensure_ascii=False,
            ),
            role="dual_compare",
        )
        winner = str((vote or {}).get("best") or "A").strip().upper()[:1]
        pick = first if winner != "B" else second
    if pick is None:
        pick = first

    if not pick or not pick.get("best_product_id"):
        return None

    tip = (
        f"Dual-path pick: product_id={pick.get('best_product_id')} "
        f"sku={pick.get('best_sku')} score={pick.get('relevance_score')} "
        f"all_match={pick.get('all_match')} reason={pick.get('reason')}. "
        "View/inspect this exact variant next if firm constraints hold."
    )

    if _VERIFY_PICK and pick.get("best_product_id"):
        target = None
        for item in results:
            if str(item.get("product_id")) == str(pick.get("best_product_id")):
                target = item
                break
        if target is not None:
            # Heuristic verify_pick (price + literals) before LLM verify.
            if not _price_fits(
                target.get("price"),
                plan.get("min_price"),
                plan.get("max_price"),
                fuzzy=True,
            ):
                tip += (
                    " HEURISTIC VERIFY FAILED: price out of band. "
                    "Do NOT commit — search another firm-compliant option."
                )
                plan["gate_swaps"] = int(plan.get("gate_swaps") or 0) + 1
                return tip
            if _LITERAL_GATE and _literal_coverage(target, plan) < _SCORE_N_THRES:
                tip += (
                    " HEURISTIC VERIFY FAILED: weak literal coverage. "
                    "Do NOT commit — prefer a candidate with literal token hits."
                )
                plan["gate_swaps"] = int(plan.get("gate_swaps") or 0) + 1
                return tip

            verified = _llm_json(
                _VERIFY_PROMPT,
                json.dumps(
                    {
                        "query": goal[:1000],
                        "candidate": {
                            "product_id": target.get("product_id"),
                            "sku": target.get("sku"),
                            "title": str(target.get("title") or "")[:160],
                            "brand": target.get("brand"),
                            "price": target.get("price"),
                            "in_stock": target.get("in_stock"),
                            "facts": target.get("facts"),
                        },
                        "firm": {
                            "price_range": plan.get("price_range"),
                            "brand_firm": plan.get("brand_firm"),
                            "spec": plan.get("spec"),
                            "literal_tokens": plan.get("literal_tokens"),
                            "category": plan.get("category"),
                        },
                    },
                    ensure_ascii=False,
                ),
                role="verify",
            )
            if verified and verified.get("ok") is False:
                missing = verified.get("missing") or []
                tip += (
                    f" VERIFY FAILED missing={missing}. "
                    "Do NOT commit yet — search/filter another firm-compliant option."
                )
                plan["gate_swaps"] = int(plan.get("gate_swaps") or 0) + 1
            elif verified and verified.get("ok") is True:
                tip += " VERIFY OK — proceed to inspect_stock on this exact variant."
    return tip


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
        ("shop_id", "shop_id"),
        ("seller", "shop_id"),
    ):
        if key in payload:
            remember(
                field,
                payload[key],
                "category_path" if key == "category_path" else src,
            )

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
            ("shop_id", "shop_id"),
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

    cart_items = payload.get("cart")
    if isinstance(cart_items, list):
        keys: set[str] = set()
        shops: set[str] = set()
        for item in cart_items:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("product_id") or "")
            sku = str(item.get("sku") or "")
            shop = str(item.get("shop_id") or item.get("seller") or "")
            if shop:
                shops.add(shop)
            if pid and sku:
                keys.add(f"{pid}::{sku}")
                if item.get("current_price") is not None:
                    remember("price", item["current_price"], "cart")
                elif item.get("price") is not None:
                    remember("price", item["price"], "cart")
                if "in_stock" in item:
                    remember("in_stock", item["in_stock"], "cart")
        memory["cart_keys"] = keys
        memory["cart_shops"] = shops
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
    if not needs_justification:
        return []
    facts = memory.get("facts") or {}
    sources = memory.get("fact_sources") or {}
    goal = _norm((plan or {}).get("goal") or "")
    claims: list[dict[str, str]] = []

    for field in ("price", "in_stock"):
        if field in facts:
            claims.append({"field": field, "value": _claim_value(field, facts[field])})

    structured = {"facts", "raw_facts", "product", "view", "compare", "variant"}
    for field, needles in (
        ("brand", ()),
        ("screen_size", ("inch", "screen")),
        ("resolution", ("720p", "1080p", "1440", "4k", "resolution")),
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
        if c["field"] in seen or c["field"] not in _CLAIM_FIELDS:
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
    args.pop("justification", None)
    if not needs_justification:
        return args
    claims = _build_safe_claims(memory, True, plan)
    if claims:
        args["justification"] = {"claims": claims}
    return args


def _hard_gate_blocks(
    args: dict[str, Any],
    memory: dict[str, Any],
    plan: dict[str, Any],
) -> str | None:
    """0824 HARD_CONSTRAINT_GATE / verify_pick before commit."""
    if not _HARD_CONSTRAINT_GATE:
        return None
    facts = memory.get("facts") or {}
    in_stock = facts.get("in_stock")
    if in_stock is False or _norm(in_stock) in {"false", "0", "out of stock"}:
        return "Hard gate: out of stock. Swap to another firm-compliant option."

    max_price = plan.get("max_price")
    price = facts.get("price")
    try:
        if max_price is not None and price is not None:
            pv = float(price)
            if pv > float(max_price) + _FUZZY_PRICE_ABS:
                return f"Hard gate: price {price} exceeds budget {max_price}."
    except (TypeError, ValueError):
        pass

    brand_firm = _norm(plan.get("brand_firm") or "")
    if brand_firm:
        brand = _norm(facts.get("brand") or "")
        obs = memory.get("last_obs_norm") or ""
        if brand and brand != brand_firm and brand_firm not in obs:
            if int(plan.get("gate_swaps") or 0) < _MAX_GATE_SWAPS:
                return (
                    f"Hard gate: firm brand {plan.get('brand_firm')} not confirmed. "
                    "Swap candidate or re-view facts."
                )

    del args  # args inspected only via memory/plan for gate
    return None


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
        if not _sanitize_product_id(pid):
            return "place_test_order needs a valid product_id.", tool_calls

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
                f"Item {pid}::{sku} is not in cart. "
                "Call add_to_cart for the exact product_id/sku, then place_test_order.",
                tool_calls,
            )
        if "add_to_cart" in tool_names and not cart_keys:
            return (
                "Cart is empty. Call add_to_cart before place_test_order.",
                tool_calls,
            )

        if plan.get("same_shop") or plan.get("lane") == "shop":
            shops = memory.get("cart_shops") or set()
            if len(shops) > 1:
                return (
                    "Same-shop required: cart has mixed shop_ids. "
                    "Remove mismatched lines and keep one seller.",
                    tool_calls,
                )

        gate = _hard_gate_blocks(args, memory, plan)
        if gate:
            plan["gate_swaps"] = int(plan.get("gate_swaps") or 0) + 1
            return gate, tool_calls

        if needs_just and not memory.get("saw_view") and "view" in tool_names:
            return (
                "Justification required: view/compare the final product first.",
                tool_calls,
            )

        new_args = _rewrite_place_order_args(args, memory, needs_just, plan)
        if needs_just and "justification" not in new_args:
            return (
                "Need grounded justification claims (at least price + in_stock).",
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
            content = (msg.get("content") or "").strip()
            if content:
                out.append({"role": "assistant", "content": content})
            i = j
            continue
        if role == "tool":
            i += 1
            continue
        out.append(msg)
        i += 1
    return out


def _trim_messages(messages: list[dict[str, Any]], keep_tail: int = 20) -> None:
    if len(messages) <= keep_tail + 5:
        messages[:] = _sanitize_messages(messages)
        return
    head = messages[:4]
    start = max(4, len(messages) - keep_tail)
    while start < len(messages) and messages[start].get("role") == "tool":
        start += 1
    while start < len(messages):
        msg = messages[start]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
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
    messages[:] = _sanitize_messages(
        head
        + [
            {
                "role": "user",
                "content": (
                    "[Earlier traces omitted.] Continue 0824 dual-path commit: answer "
                    "pending shopper messages, inspect_stock on the exact variant, "
                    "then place_test_order."
                ),
            }
        ]
        + messages[start:]
    )


def _guard_nudge(
    messages: list[dict[str, Any]],
    assistant_content: str,
    nudge: str,
) -> None:
    content = (assistant_content or "").strip()
    if content:
        messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": nudge})


def _chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    attempt: int = 0,
) -> dict[str, Any] | None:
    for model in _model_chain():
        safe = _sanitize_messages(messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": safe,
            "temperature": _temp_for("act", attempt),
        }
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = "required"
        inference = _proxy.post("/inference/chat/completions", json_data=payload)
        if inference and (inference.get("choices") or []):
            messages[:] = safe
            return inference["choices"][0].get("message") or {}
        _trim_messages(messages, keep_tail=16)
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

    full_query = _sanitize_goal(str(policy.get("query") or ""))
    goal = _shopper_goal(full_query)
    plan = _heuristic_parse(goal)
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
                    "Critical: when a shopper user_message asks which item you chose or "
                    "whether a firm spec holds, call message and answer BEFORE place_test_order."
                ),
            }
        )
    if not plan.get("needs_justification"):
        messages.append(
            {
                "role": "user",
                "content": (
                    "This task does not require justification. Omit the justification "
                    "argument on place_test_order completely."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "provider": _provider(),
            "model": _model(),
            "plan": plan,
            "strategy": "0824-xp-rrf-dualpath-commit",
        }
    ]
    memory: dict[str, Any] = {
        "facts": {},
        "fact_norms": {},
        "fact_sources": {},
        "pending_intervention": False,
        "cart_keys": set(),
        "cart_shops": set(),
    }
    guard_retries = 0
    env_turn = 0
    max_model_steps = max_steps + 8
    search_rounds = 0
    selected = False

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
                        f"{'(grounded claims only)' if plan.get('needs_justification') else '(NO justification)'}."
                        + (" ALWAYS_SUBMIT: do not stall." if _ALWAYS_SUBMIT else "")
                    ),
                }
            )

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for attempt in range(_LLM_RETRY_MAX):
            assistant = _chat(messages, tools, attempt=attempt)
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
                            "(search/filter/view/compare/inspect_stock/message/"
                            "add_to_cart/place_test_order)."
                        ),
                    },
                ]
            )
        if not tool_calls:
            dialogue.append(
                {"role": "error", "content": "inference failed after retries; no tool call"}
            )
            break

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
                        "STOP. Previous tool choice was not executed. "
                        "Call message FIRST with product_id/sku and firm-spec confirmation."
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
                f"Previous tool choice was not executed. Do not finish yet. {block}",
            )
            continue

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
                results = memory.get("last_results") or []
                if not selected and results and search_rounds <= 2:
                    tip = _llm_select(goal, results, plan)
                    if tip:
                        selected = True
                        messages.append({"role": "user", "content": tip})
                # If top two fit scores are within relax gap, nudge another view.
                if len(results) >= 2 and search_rounds <= 3:
                    s0, _ = _fit_score(results[0], plan, 0)
                    s1, _ = _fit_score(results[1], plan, 1)
                    if abs(s0 - s1) < _RELAX_SCORE_GAP:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Top two fit scores are within "
                                    f"{_RELAX_SCORE_GAP}. View/compare both before commit."
                                ),
                            }
                        )

        if result.get("user_message"):
            content = (result["user_message"].get("content") or "").strip()
            if content:
                _update_plan_from_shopper(plan, content)
                if _requires_message_reply(content):
                    memory["pending_intervention"] = True
                    guard_retries = 0
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Shopper message: {content[:1500]}\n"
                                "REQUIRED NEXT ACTION: call message and answer before ordering."
                                + (
                                    f" Current budget ceiling: {plan['max_price']}."
                                    if plan.get("max_price") is not None
                                    else ""
                                )
                            ),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Shopper note (no reply required): {content[:800]}"
                            ),
                        }
                    )

        if memory.get("last_error"):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool error: {memory['last_error']}. Salvage: remove bad cart lines, "
                        f"try rescue searches {plan.get('rescue_queries')}"
                        + (
                            f" or widened price {plan.get('widened_price_range')}"
                            if plan.get("widened_price_range")
                            else ""
                        )
                        + ", pick another firm-compliant in-stock option with literal token "
                        "match, inspect_stock, then order "
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
