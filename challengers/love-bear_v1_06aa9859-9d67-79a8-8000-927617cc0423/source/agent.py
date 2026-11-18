

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# Live allowlist (Sep 2026 docs). Prefer OpenRouter unless INFERENCE_PROVIDER=chutes.
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
    "google/gemma-4-31b-it",
    "openai/gpt-oss-120b",
)
_CHUTES_FALLBACKS = (
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "moonshotai/Kimi-K2.6-TEE",
    "google/gemma-4-31B-turbo-TEE",
    "openai/gpt-oss-120b-TEE",
)
# leopard Specs role priorities (allowlisted subset; content-driven role, not phase).
_OPENROUTER_SELECT = ("google/gemma-4-31b-it", "deepseek/deepseek-v3.2")
_OPENROUTER_ALT_SELECT = (
    "deepseek/deepseek-v3.2",
    "openai/gpt-oss-120b",
    "google/gemma-4-31b-it",
)
_OPENROUTER_DUAL = (
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-chat-v3.1",
)
_OPENROUTER_CLASSIFY = ("google/gemma-4-31b-it", "deepseek/deepseek-v3.2")
_CHUTES_SELECT = (
    "google/gemma-4-31B-turbo-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
)
_CHUTES_ALT_SELECT = (
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.1-TEE",
)
_CHUTES_DUAL = (
    "zai-org/GLM-5.1-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
    "moonshotai/Kimi-K2.6-TEE",
)
_CHUTES_CLASSIFY = (
    "google/gemma-4-31B-turbo-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
)

_proxy = ProxyClient(timeout=120, max_retries=3)

# Always-safe justification fields (current-state from inspect_stock/cart).
_SAFE_CLAIM_FIELDS = frozenset({"price", "in_stock"})
# Fallback when place_test_order schema has no field enum (tool schema is preferred).
_DEFAULT_CLAIM_FIELDS = frozenset(
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
    "screen": "screen_size",
    "display": "resolution",
    "stock": "in_stock",
    "current_price": "price",
    "memory": "ram",
    "ram_gb": "ram",
    "refresh": "refresh_rate",
}
_FIELD_ALIASES = {
    "size": "screen_size",
    "screen": "screen_size",
    "display": "resolution",
    "refresh": "refresh_rate",
    "hz": "refresh_rate",
    "memory": "ram",
    "stock": "in_stock",
}
# Generic measurement / wording cues → attribute field (schema binding, any domain).
_ATTR_CUES: tuple[tuple[str, str], ...] = (
    ("screen_size", r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?"),
    ("refresh_rate", r"(\d+)\s*hz"),
    ("resolution", r"\b(\d{3,4}p|4k|8k|full\s*hd|fhd|\bhd\b)\b"),
    ("ram", r"(\d+)\s*gb\s*ram|\bram\s*(\d+)\s*gb"),
)
# Soft color words used only for preference parsing (not a product taxonomy).
_COLOR_WORDS = frozenset(
    {
        "black",
        "white",
        "red",
        "blue",
        "green",
        "gold",
        "silver",
        "pink",
        "gray",
        "grey",
        "brown",
        "yellow",
        "purple",
        "orange",
        "beige",
        "navy",
        "bronze",
    }
)

# leopard Specs.ID_KEYS + packing attrs — hard-gate before place_test_order.
_GATE_HARD_KEYS = frozenset(
    {
        "brand",
        "color",
        "color_family",
        "model",
        "id",
        "compatibility",
        "material",
        "packaging",
        "storage",
        "pack",
        "set",
        "size",
        "capacity",
        "volume",
        "screen_size",
        "refresh_rate",
        "resolution",
        "ram",
        "no",
    }
)

# leopard Specs / DualSolveConfig portable knobs (content-triggered, not wall-clock).
_SELF_CONSISTENCY_GAP = 1.5
_TIGHT_RACE_GAP = 0.12
_MIN_JUDGE_SCORE = 6.0
_FAST_ACCEPT_SCORE = 8.0
_FAST_ACCEPT_COV = 0.5
_HOST_CONF_HIGH = 0.85
_HOST_CONF_LOW = 0.55
_SCORE_N_THRES = 0.4
_FUZZY_PRICE_RATIO = 0.08
_FUZZY_PRICE_ABS_TOL = 1.0
_TEMP_RETRY_STEP = 0.3
_TEMP_MAX = 1.2
_ALT_SELECT_TEMPERATURE = 0.55
_COMPARE_TEMPERATURE = 0.25
_RRF_K = 60
_RRF_WEIGHT = 0.15
_RANK_BONUS_C = 0.05
_RANK_LIMIT = 300
_ENABLE_DUAL_PATH = True
_INVALID_PRODUCT_IDS = frozenset({"", "none", "null", "0", "undefined", "n/a"})


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

Pipeline (leopard spirit: classify → decompose → search → weigh/RRF → prune → dual-path select → hard-gate → commit):
1. Decompose the goal into MECE key→value constraints (product_type, brand, color, material, size, …).
2. Firm / hard-gate attrs beat soft preferences (budget, in-stock, identity keys). Soft brand/color only among firm-compliant options.
3. Trust structured facts (view/compare facts, category_path, variants, attributes, sku_options) over title SEO text.
4. Dual-path select: Path-A (PV_SELECT) + Path-B (PV_SELECT_ALT) + DUAL_COMPARE when the pool is tight.
   Prefer evidence quality R1–R6 (directness, key specificity, structured>title, narrow value, coverage, then rank).
5. Always inspect_stock on the exact product_id/sku before place_test_order.
6. After ANY shopper user_message that asks a question, call message and answer BEFORE place_test_order.
7. Ranking tasks: compare ONLY given candidates; follow the goal's stated priority order; never pick a dominated option.
8. Recovery: if price/stock changes or preferred variant disappears, remove bad cart lines.
   Prefer least relaxation: same model+color → same model/other color → other model.
   View/compare facts so model/color evidence is grounded before re-ordering.
9. Finish with place_test_order once verified. Do not wait for permission.
10. Domain-agnostic: do not assume a fixed product taxonomy. Follow the goal's stated category and attributes.
11. Honor soft brand/model/color on the FIRST cart pick only among fully firm-compliant options.
12. Numbers/ids/variant codes must match LITERALLY (MATCH_RULES) — no range inference, no brand guessing.

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


def _role_models(role: str) -> list[str]:
    """leopard Gateway.get_role_llm_priorities — role-preferred then general chain."""
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    chutes = _provider() == "chutes"
    if role in {"select", "judge"}:
        preferred = _CHUTES_SELECT if chutes else _OPENROUTER_SELECT
    elif role == "alt_select":
        preferred = _CHUTES_ALT_SELECT if chutes else _OPENROUTER_ALT_SELECT
    elif role == "dual_compare":
        preferred = _CHUTES_DUAL if chutes else _OPENROUTER_DUAL
    elif role in {"classify", "parse", "intent"}:
        preferred = _CHUTES_CLASSIFY if chutes else _OPENROUTER_CLASSIFY
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
        "intent": 0.0,
        "score": 0.4,
        "judge": 0.4,
        "select": 0.4,
        "alt_select": _ALT_SELECT_TEMPERATURE,
        "dual_compare": _COMPARE_TEMPERATURE,
        "verify": 0.2,
        "act": 0.0,
    }.get(role, 0.0)
    return min(_TEMP_MAX, base + max(0, attempt) * _TEMP_RETRY_STEP)


# leopard ProductEngine / DualSolveConfig knobs
_RELAX_SCORE_GAP = 0.6
_SELECT_CANDIDATES = 10
_JUDGE_ROUNDS_MAX = 2
_ALT_PRODUCT_SHORTLIST = 20

_MATCH_RULES_BRIEF = """
MATCH_RULES (condensed):
1. Numbers, ids, codes, variant names must match LITERALLY — no range inference, no brand guessing.
2. Descriptive constraints may use common-sense paraphrases; collective words (set/pack/kit) are not ids.
3. Titles are SEO-noisy — do not reject solely for extra title words; structured attributes beat titles.
4. One matching attribute/sku line is enough; conflicting alternate lines are OK.
5. When sku_options exist, variant constraints must hit at least one sku line.
6. Negation ("no X"/"without X") requires absence of that feature as a positive claim.
7. Recommendation prefers direct structured evidence over noisy titles; original rank is last resort.
""".strip()

_PV_ITEM_JUDGE = """
Goal: pick ONE listing from a candidate pool (leopard PV_SELECT / Path-A).

Inputs: JSON with `request`, `candidates` (title, price, brand, sku, attributes/facts,
in_stock), `firm` constraints, and `only_product_type`.

{rules}

Matching / evidence rules:
(a) Structured attributes and sku_options outweigh title SEO prose.
(b) Treat `firm` as the definitive pass/fail constraint list — do NOT invent new must-haves.
(c) A candidate is all-matching only if every firm constraint is satisfied (fuzzy formatting OK).
(d) Among all-matching, apply R1→R6:
    R1 — fewest extraneous tokens in the matching attribute/sku value
    R2 — dedicated attribute keys beat catch-all fields
    R3 — structured field evidence beats title-only
    R4 — exact/narrow values beat compound/range values
    R5 — broader descriptive coverage without R1 penalties
    R6 — original ranking only as final tie-breaker
(e) Exact variant match when the user named color/pack/size/model/capacity.
(f) When `only_product_type` is true, require a bare product-type "only" variant.
(g) Price is a last-resort tiebreaker (not a reason to drop a better evidence match).
(h) If no all-matching candidate exists, return null.

Output strict JSON only:
{{
  "best_product_id": "<id from candidates or null>",
  "best_sku": "<sku or null>",
  "reason": "1-2 sentences citing attribute/sku values",
  "relevance_score": <integer 0-10>,
  "all_match": true|false
}}
""".strip().format(rules=_MATCH_RULES_BRIEF)

_PV_SELECT_ALT = """
You are a STRICT independent second opinion (leopard PV_SELECT_ALT / Path-B).
Path-A already proposed a candidate; give your own pick from the candidates.

{rules}

Priority:
1. Reject any candidate missing a hard constraint (color/pack/model/size/brand/negation).
2. Prefer structured attributes/facts/variants over SEO titles.
3. Prefer the most direct attribute match (fewest extra tokens).
4. If none fully match, return null (do not force a weak guess).

Output strict JSON only:
{{
  "best_product_id": "<id from candidates or null>",
  "best_sku": "<sku or null>",
  "reason": "1-2 sentences citing attribute/sku values",
  "relevance_score": <integer 0-10>,
  "all_match": true|false
}}
""".strip().format(rules=_MATCH_RULES_BRIEF)

_DUAL_COMPARE = """
Choose which of two candidate answers better satisfies the shopping request
(leopard DUAL_COMPARE). Candidate A is Path-A / primary; Candidate B is Path-B / alternate.

{rules}

Decision Rules:
1. Prefer the candidate that satisfies MORE required constraints with clear evidence.
2. If both satisfy all constraints, prefer the more exact / direct match (structured, fewer extras).
3. If still tied, prefer Candidate A.
4. Never invent product details; only use the provided JSON.

Return JSON only: {{"best":"A"|"B","reason":"one sentence"}}
""".strip().format(rules=_MATCH_RULES_BRIEF)

_PV_INTENT = """
Decompose a shopper goal into MECE key→value constraints (leopard A_DECOMPOSE spirit).
Do NOT assume electronics or any fixed product taxonomy. Extract ONLY what the goal states.

Output strict JSON only:
{{
  "keywords": "2-12 search tokens joined (brand + product_type + distinctive attrs)",
  "keyword_pairs": ["product_type→…", "brand→…", "color→…"],
  "category": "product/class phrase from the goal or null",
  "brand_firm": "required brand or null",
  "brand_soft": "preferred/soft brand or null",
  "max_price": <number or null>,
  "min_price": <number or null>,
  "constraints": [
    {{"field": "snake_case_attribute", "value": "as stated", "firm": true}}
  ],
  "claim_fields": ["attribute fields to evidence if justification is required"],
  "rescue_queries": ["up to 4 broader or narrower search strings"],
  "needs_ranking": false
}}

Rules:
- keyword_pairs use key→value; mutually exclusive and collectively exhaustive for the goal.
- Always include a product_type pair when a product class is named.
- Soft preferences (ideally / prefer / nice to have) → firm=false; must/need/required → firm=true.
- claim_fields: goal-mentioned attributes only; omit price and in_stock.
- rescue_queries: linguistic broaden/narrow (drop soft attrs, keep product_type+brand), not fixed templates.
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
    """leopard Support.extract_select_response — match by product_id."""
    pid = str(js.get("best_product_id") or "").strip()
    if not pid or pid.lower() in {"null", "none", "n/a"}:
        return -1
    for i, prod in enumerate(candidates):
        if str(prod.get("product_id") or "") == pid:
            return i
    return -1


def _is_prod_only_query(goal: str) -> bool:
    return bool(re.search(r"\bproduct\s+only\b", goal or "", re.I))


def _sanitize_product_id(value: Any) -> str | None:
    pid = str(value or "").strip()
    if pid.lower() in _INVALID_PRODUCT_IDS:
        return None
    return pid or None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    try:
        out = float(re.sub(r"[^\d.\-]", "", str(value).split()[0]))
        return out if math.isfinite(out) else default
    except (TypeError, ValueError, IndexError):
        return default


def _price_fits(price: Any, max_price: Any, min_price: Any = None, *, fuzzy: bool = True) -> bool:
    """Soft budget edges for firm price filters (portable, not suite-specific)."""
    if price is None:
        return True
    pv = _coerce_float(price, default=float("nan"))
    if not math.isfinite(pv):
        return True
    tol = _FUZZY_PRICE_ABS_TOL if fuzzy else 0.0
    if max_price is not None:
        hi = _coerce_float(max_price)
        if pv > hi + tol and pv > hi * (1.0 + (_FUZZY_PRICE_RATIO if fuzzy else 0.0)):
            return False
    if min_price is not None:
        lo = _coerce_float(min_price)
        if pv < lo - tol and pv < lo * (1.0 - (_FUZZY_PRICE_RATIO if fuzzy else 0.0)):
            return False
    return True


def _llm_json_once(
    system: str,
    user: str,
    models: list[str] | None = None,
    *,
    temperature: float | None = None,
    role: str = "judge",
) -> dict[str, Any] | None:
    chain = models or _role_models(role)[:3]
    for attempt, model in enumerate(chain):
        temp = (
            float(temperature)
            if temperature is not None
            else _temp_for(role, attempt)
        )
        inference = _proxy.post(
            "/inference/chat/completions",
            json_data={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temp,
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


def _heuristic_pool_scores(
    results: list[dict[str, Any]], plan: dict[str, Any]
) -> list[float]:
    tokens = _tokenize(str(plan.get("keywords") or ""))
    scores: list[float] = []
    for rank, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        score = _title_hit_score(tokens, title)
        # leopard Specs RRF fusion on search rank.
        score += _RRF_WEIGHT * (1.0 / (_RRF_K + rank + 1)) * 100.0
        score += max(0.0, 1.0 - rank / _RANK_LIMIT) * _RANK_BONUS_C
        if item.get("in_stock") is True:
            score += 3.0
        if not _price_fits(item.get("price"), plan.get("max_price"), plan.get("min_price")):
            score -= 20.0
        scores.append(score)
    return scores


def _should_self_consistency(scores: list[float]) -> bool:
    """Tight top-2 gap → run second opinion (leopard DualSolve content gate)."""
    if len(scores) < 2:
        return False
    ordered = sorted(scores, reverse=True)
    return (ordered[0] - ordered[1]) < _SELF_CONSISTENCY_GAP


def _should_run_dual_path(scores: list[float]) -> bool:
    """Content-triggered dual path — never wall-clock / phase routing."""
    if not _ENABLE_DUAL_PATH or len(scores) < 2:
        return False
    ordered = sorted(scores, reverse=True)
    if ordered[0] - ordered[1] < _SELF_CONSISTENCY_GAP:
        return True
    if ordered[0] - ordered[1] < _TIGHT_RACE_GAP * 10:
        return True
    return False


def _vote_product_id(votes: list[dict[str, Any]]) -> str | None:
    tallies: dict[str, list[float]] = defaultdict(list)
    for row in votes:
        if not isinstance(row, dict):
            continue
        pid = _sanitize_product_id(row.get("best_product_id") or row.get("product_id"))
        if not pid:
            continue
        try:
            sc = float(row.get("relevance_score", row.get("score", 0)) or 0)
        except (TypeError, ValueError):
            sc = 0.0
        tallies[pid].append(sc)
    if not tallies:
        return None
    ranked = sorted(
        ((pid, sum(scs), len(scs)) for pid, scs in tallies.items()),
        key=lambda t: (t[1], t[2]),
        reverse=True,
    )
    return ranked[0][0]


def _dual_path_pick(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
    *,
    heuristic_gap: float,
) -> dict[str, Any] | None:
    """Return a pick dict, or None when dual_compare is required."""
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
    return None


def _judge_select(goal: str, results: list[dict[str, Any]], plan: dict[str, Any]) -> str | None:
    """leopard DualCandidateOrchestrator spirit on EnvPack search results.

    Path-A PV_SELECT → (tight race) Path-B PV_SELECT_ALT → DUAL_COMPARE → tip.
    Dual is content-triggered from score gaps, never wall-clock phase routing.
    """
    if not results:
        return None
    cap = _candidate_prompt_rows(results, min(_SELECT_CANDIDATES, _ALT_PRODUCT_SHORTLIST))
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
            "keyword_pairs": plan.get("keyword_pairs") or [],
        },
        "candidates": cap,
    }
    user_blob = json.dumps(payload, ensure_ascii=False)
    heur = _heuristic_pool_scores(cap, plan)
    gap = (heur[0] - heur[1]) if len(heur) > 1 else 99.0
    low_conf = _should_self_consistency(heur)
    run_dual = _should_run_dual_path(heur)

    first = _llm_json_once(
        _PV_ITEM_JUDGE,
        user_blob,
        role="select",
        temperature=_temp_for("select", 1 if low_conf else 0),
    )
    # Host weak / not all-match → always open Path-B.
    host_score = _coerce_float((first or {}).get("relevance_score"), 0.0)
    if first and first.get("all_match") is False:
        run_dual = True
    if host_score < _MIN_JUDGE_SCORE or host_score < _HOST_CONF_LOW * 10:
        run_dual = True
    if host_score < _HOST_CONF_HIGH * 10 and gap < _SELF_CONSISTENCY_GAP:
        run_dual = True

    second = None
    if run_dual:
        second = _llm_json_once(
            _PV_SELECT_ALT,
            user_blob,
            role="alt_select",
            temperature=_temp_for("alt_select", 0),
        )

    pick = _dual_path_pick(first, second, heuristic_gap=gap)
    compare_note = ""
    if pick is None and first and second:
        vote = _llm_json_once(
            _DUAL_COMPARE,
            json.dumps(
                {
                    "query": goal[:800],
                    "A": first,
                    "B": second,
                    "candidates": cap[:6],
                },
                ensure_ascii=False,
            ),
            role="dual_compare",
            temperature=_COMPARE_TEMPERATURE,
        )
        winner = str((vote or {}).get("best") or "A").strip().upper()[:1]
        pick = first if winner != "B" else second
        compare_note = f" dual_compare={winner} ({(vote or {}).get('reason')})"
    if pick is None:
        pick = first or second

    if not pick:
        return None

    # Optional self-consistency vote when Path-A alone and scores are tight.
    if not second and low_conf and first:
        alt_vote = _llm_json_once(
            _PV_ITEM_JUDGE,
            user_blob,
            role="select",
            temperature=_temp_for("select", 1),
        )
        if alt_vote:
            voted = _vote_product_id([first, alt_vote])
            if voted:
                matching = [
                    v
                    for v in (first, alt_vote)
                    if _sanitize_product_id(v.get("best_product_id")) == voted
                ]
                if matching:
                    pick = max(
                        matching,
                        key=lambda v: _coerce_float(v.get("relevance_score"), 0.0),
                    )

    idx = _extract_judge_response(pick, cap)
    jscore = _coerce_float(pick.get("relevance_score"), 0.0)

    if idx < 0 or jscore < _MIN_JUDGE_SCORE:
        rescue = plan.get("rescue_queries") or []
        return (
            f"Dual-path soft-fail (score={jscore}, idx={idx}). "
            "Use the heuristic shortlist; view top firm-compliant options, then inspect_stock."
            + (f" Rescue: {rescue}" if rescue else "")
            + compare_note
        )

    best = cap[idx]
    tip = (
        f"leopard dual-path pick: product_id={best.get('product_id')} "
        f"sku={best.get('sku')} relevance_score={jscore} "
        f"reason={pick.get('reason')}."
        f"{compare_note} "
        "Hard-gate: verify firm/identity attrs on view facts, inspect_stock, then order."
    )
    if jscore >= _FAST_ACCEPT_SCORE and gap >= _SELF_CONSISTENCY_GAP:
        tip += " Fast-accept — prioritize this variant unless stock/price fails."
    return tip


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


def _claim_field_allowlist(tools: list[dict[str, Any]] | None = None) -> frozenset[str]:
    """Read allowed claim fields from place_test_order tool schema."""
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or tool
        if (fn.get("name") or "") != "place_test_order":
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        claims = (
            ((props.get("justification") or {}).get("properties") or {})
            .get("claims", {})
            .get("items", {})
            .get("properties", {})
            .get("field", {})
        )
        enum = claims.get("enum")
        if isinstance(enum, list) and enum:
            return frozenset(str(x) for x in enum)
    return _DEFAULT_CLAIM_FIELDS


def _allowed_claim_fields(plan: dict[str, Any] | None = None) -> frozenset[str]:
    plan = plan or {}
    raw = plan.get("claim_allowlist")
    if isinstance(raw, (list, tuple, set, frozenset)) and raw:
        return frozenset(str(x) for x in raw)
    return _DEFAULT_CLAIM_FIELDS


def _normalize_claim_field(field: str, allowed: frozenset[str]) -> str | None:
    raw = str(field or "").strip()
    if not raw:
        return None
    key = raw.lower().replace(" ", "_")
    if raw in allowed:
        return raw
    if key in allowed:
        return key
    mapped = _FIELD_ALIASES.get(key) or _FIELD_ALIASES.get(raw.lower())
    if mapped and mapped in allowed:
        return mapped
    return None


def _claim_value(field: str, value: Any) -> str:
    """Normalize only price / in_stock. Spec claims keep observed text (no invent/remap)."""
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
        cleaned = re.sub(r"[^\d.]", "", str(value).replace(",", ""))
        if cleaned:
            try:
                return f"{float(cleaned):.2f}"
            except ValueError:
                pass
        return str(value).strip()
    # Light formatting for screen_size: verifier expects values like "12.5in".
    if field == "screen_size":
        text = str(value).strip()
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            return f"{m.group(1)}in"
        return text
    if field == "refresh_rate":
        text = str(value).strip()
        m = re.search(r"(\d+)\s*hz\b", text, flags=re.I)
        if m:
            return f"{m.group(1)}hz" if "hz" in text.lower() else m.group(1)
        m = re.search(r"(\d+)", text)
        return m.group(1) if m else text
    return str(value).strip()


def _facts_agree(field: str, observed: Any, want: Any) -> bool:
    """Whether an observation agrees with a requirement without inventing claim tokens."""
    if observed is None or want is None or want == "":
        return False
    c = str(want).strip().lower()
    o = str(observed).strip().lower()
    if not c or not o:
        return False
    if field == "price":
        try:
            return abs(
                float(re.sub(r"[^\d.]", "", c) or "nan")
                - float(re.sub(r"[^\d.]", "", o) or "nan")
            ) < 0.05
        except ValueError:
            return c in o or o in c
    if field == "in_stock":
        def as_bool(x: str) -> bool | None:
            if x in {"true", "1", "yes", "in stock", "instock"}:
                return True
            if x in {"false", "0", "no", "out of stock", "outofstock"}:
                return False
            return None
        cb, ob = as_bool(c), as_bool(o)
        return cb is not None and cb == ob
    if field in {"refresh_rate", "screen_size", "ram"}:
        cm = re.search(r"(\d+(?:\.\d+)?)", c)
        om = re.search(r"(\d+(?:\.\d+)?)", o)
        if not (cm and om):
            return False
        if field == "refresh_rate":
            return cm.group(1) == om.group(1)
        return abs(float(cm.group(1)) - float(om.group(1))) < 0.05
    if field == "resolution":
        cn = re.sub(r"\s+", "", c)
        on = re.sub(r"\s+", "", o)
        if c in o or o in c or cn in on or on in cn:
            return True
        # Common class aliases that appear together in catalog text — still require
        # shared tokens in the observation, never invent from pixels alone.
        aliases = {
            "fhd": ("fhd", "fullhd", "full-hd", "1080p"),
            "fullhd": ("fhd", "fullhd", "full-hd", "1080p"),
            "full hd": ("fhd", "fullhd", "full-hd", "1080p"),
            "hd": ("hd", "720p"),
            "uhd": ("uhd", "4k", "2160p"),
            "4k": ("uhd", "4k", "2160p"),
        }
        for alias in aliases.get(c, ()):
            if alias in on:
                # Bare "hd" must not match only via "fhd"/"uhd".
                if c == "hd" and any(x in on for x in ("fhd", "fullhd", "uhd", "qhd")):
                    if "720p" in on or re.search(r"(?<![a-z])hd(?![a-z])", on):
                        return True
                    continue
                return True
        return False
    return c == o or c in o or o in c


def _wanted_attr(plan: dict[str, Any], field: str) -> Any:
    for c in plan.get("constraints") or []:
        if isinstance(c, dict) and str(c.get("field") or "") == field and c.get("value") not in (None, ""):
            return c.get("value")
    spec = plan.get("spec") or {}
    if str(spec.get("field") or "") == field and spec.get("value") not in (None, ""):
        return spec.get("value")
    for f, v in _extract_attr_cues(str(plan.get("goal") or "")):
        if f == field:
            return v
    return None


def _add_constraint(plan: dict[str, Any], field: str, value: Any, firm: bool = True) -> None:
    field = str(field or "").strip().lower().replace(" ", "_")
    if not field or value is None or value == "":
        return
    constraints = list(plan.get("constraints") or [])
    constraints = [c for c in constraints if not (isinstance(c, dict) and c.get("field") == field)]
    constraints.append({"field": field, "value": value, "firm": bool(firm)})
    plan["constraints"] = constraints
    if firm and field not in {"brand", "category", "price"}:
        plan["spec"] = {"field": field, "value": value}
    claim_fields = list(plan.get("claim_fields") or [])
    allowed = _allowed_claim_fields(plan)
    if field in allowed and field not in _SAFE_CLAIM_FIELDS and field not in claim_fields:
        claim_fields.append(field)
        plan["claim_fields"] = claim_fields


def _goal_resolution_token(goal: str) -> str | None:
    """Return a resolution token as written in the goal (no invented class labels)."""
    lower = (goal or "").lower()
    # Prefer the most specific token present.
    if re.search(r"\bfull\s*hd\b|\bfhd\b", lower):
        return "fhd"
    if re.search(r"\buhd\b", lower):
        return "uhd"
    if re.search(r"\b8k\b", lower):
        return "8k"
    if re.search(r"\b4k\b", lower):
        return "4k"
    m = re.search(r"\b(\d{3,4}p)\b", lower)
    if m:
        # When the goal pairs a pixel class with bare "hd" (e.g. "720p HD"), the
        # firm gate is usually the hd class — keep the pixel token in search text.
        if re.search(r"(?<![a-z])hd(?![a-z])", lower) and not re.search(
            r"full\s*hd|\bfhd\b", lower
        ):
            return "hd"
        return m.group(1)
    if re.search(r"\bhd\b", lower):
        return "hd"
    return None


def _parse_soft_preferences(goal: str) -> dict[str, Any]:
    """Extract soft model/color preferences and exclusion phrases (domain-agnostic)."""
    out: dict[str, Any] = {
        "preferred_model": None,
        "preferred_color": None,
        "exclude_terms": [],
        "needs_recovery_ladder": False,
    }
    text = goal or ""
    lower = text.lower()

    # Explicit recovery / preference language → enable ladder guidance.
    if any(
        p in lower
        for p in (
            "if that exact",
            "another color",
            "before changing models",
            "before switching models",
            "disappears",
            "becomes unavailable",
            "strongly prefer",
            "ideally in",
        )
    ):
        out["needs_recovery_ladder"] = True

    # "prefer the gold Y81" / "strongly prefer the grey AE11"
    m = re.search(
        r"(?:strongly\s+)?prefer(?:s|red)?\s+(?:the\s+)?([A-Za-z][A-Za-z0-9_-]*)\s+([A-Za-z0-9][A-Za-z0-9_-]{1,20})",
        text,
        flags=re.I,
    )
    if m:
        a, b = m.group(1), m.group(2)
        if a.lower() in _COLOR_WORDS:
            out["preferred_color"] = a.lower()
            out["preferred_model"] = b
        elif b.lower() in _COLOR_WORDS:
            out["preferred_model"] = a
            out["preferred_color"] = b.lower()
        else:
            out["preferred_model"] = b if any(ch.isdigit() for ch in b) else a

    # "AKZ25, ideally in white" / "want the AKZ25, ideally in white"
    m = re.search(
        r"\b([A-Za-z]{1,10}\d[\w-]{0,12})\b[^.]{0,40}?\bideally\s+in\s+([A-Za-z]+)",
        text,
        flags=re.I,
    )
    if m:
        out["preferred_model"] = out["preferred_model"] or m.group(1)
        col = m.group(2).lower()
        if col in _COLOR_WORDS:
            out["preferred_color"] = out["preferred_color"] or col

    # "the gold Y81" / "in white" near a model token
    if not out["preferred_model"]:
        m = re.search(
            r"\b(?:the|want|get|buy)\s+(?:the\s+)?(?:([a-z]+)\s+)?([A-Za-z]{1,12}\d[\w-]{0,12})\b",
            text,
            flags=re.I,
        )
        if m:
            if m.group(1) and m.group(1).lower() in _COLOR_WORDS:
                out["preferred_color"] = m.group(1).lower()
            out["preferred_model"] = m.group(2)

    if not out["preferred_color"]:
        m = re.search(r"\bideally\s+in\s+([A-Za-z]+)\b", lower)
        if m and m.group(1) in _COLOR_WORDS:
            out["preferred_color"] = m.group(1)

    # Exclusion phrases: "not earbuds", "not a feature phone", "genuine X, not Y"
    excludes: list[str] = []
    for m in re.finditer(
        r"\bnot\s+(?:a|an|the\s+)?([a-z][a-z0-9][a-z0-9 &/-]{0,40})",
        lower,
    ):
        term = m.group(1).strip(" .,;")
        # Cut at connectors.
        term = re.split(r"\s+(?:or|and|,|;)\s+", term, maxsplit=1)[0].strip()
        if term and term not in _FILLER and len(term) > 2:
            excludes.append(term)
    out["exclude_terms"] = list(dict.fromkeys(excludes))[:6]
    if out["preferred_model"] or out["preferred_color"]:
        out["needs_recovery_ladder"] = True
    return out


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
        "keyword_pairs": [],
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
        "preferred_model": None,
        "preferred_color": None,
        "exclude_terms": [],
        "needs_recovery_ladder": False,
        "claim_allowlist": [],
        "given_candidates": re.findall(r"\b\d{6,}::[\w.-]+\b", goal),
    }


def _parse_keyword_pairs(raw_pairs: Any) -> list[list[str]]:
    """Normalize key→value pairs into [[key, value], ...]."""
    out: list[list[str]] = []
    if not isinstance(raw_pairs, list):
        return out
    for raw in raw_pairs:
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            key = str(raw[0]).strip().lower().replace(" ", "_")
            val = str(raw[-1]).strip()
        else:
            text = str(raw or "").strip()
            if "→" in text:
                key, _, val = text.partition("→")
            elif "->" in text:
                key, _, val = text.partition("->")
            elif ":" in text:
                key, _, val = text.partition(":")
            else:
                continue
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
        if key and val and val.lower() not in {"null", "none", "n/a"}:
            out.append([key, val])
    return out


def _pairs_to_search_tokens(pairs: list[list[str]]) -> list[str]:
    """Emit search tokens from key→value pairs (leopard SearchOption.update_q spirit)."""
    segs: list[str] = []
    schemaish = {
        "product_type",
        "color",
        "color_family",
        "material",
        "brand",
        "style",
        "size",
        "model",
        "compatibility",
        "pattern",
        "category",
        "capacity",
        "volume",
        "feature",
    }
    for pair in pairs:
        if len(pair) < 2:
            continue
        key, val = pair[0], pair[-1]
        if key and key not in schemaish:
            for m in key.replace("_", " ").split():
                m = m.lower().strip()
                if m and m not in segs:
                    segs.append(m)
        for m in str(val).split():
            m = m.lower().strip()
            if m and m not in segs:
                segs.append(m)
    return segs


def _progressive_rescues(keywords: str, plan: dict[str, Any]) -> list[str]:
    """leopard DualCandidateOrchestrator._round_search_q / short-keyword rescues."""
    rescues: list[str] = []
    words = [w for w in str(keywords or "").split() if w]
    prefer_keys = {
        "product_type",
        "brand",
        "model",
        "compatibility",
        "color",
        "color_family",
    }
    short_segs: list[str] = []
    for pair in plan.get("keyword_pairs") or []:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            # Also accept "key→value" strings from LLM intent.
            if isinstance(pair, str) and "→" in pair:
                key, _, val = pair.partition("→")
                key, val = key.strip().lower(), val.strip()
            else:
                continue
        else:
            key = str(pair[0]).strip().lower()
            val = str(pair[-1]).strip()
        if key in prefer_keys or "brand" in key or "type" in key:
            for tok in val.lower().split():
                tok = tok.strip()
                if tok and tok not in short_segs:
                    short_segs.append(tok)
    if short_segs:
        rescues.append(" ".join(short_segs[:6]))
        if len(short_segs) > 2:
            rescues.append(" ".join(short_segs[:2]))
    if plan.get("category") and plan.get("brand_firm"):
        rescues.append(f"{plan['brand_firm']} {plan['category']}")
    if plan.get("category") and plan.get("brand_soft"):
        rescues.append(f"{plan['brand_soft']} {plan['category']}")
    if plan.get("category"):
        rescues.append(str(plan["category"]))
    if len(words) > 3:
        rescues.append(" ".join(words[:3]))
    if len(words) > 1:
        rescues.append(" ".join(words[:2]))
    if words:
        rescues.append(words[0])
    return list(dict.fromkeys(r for r in rescues if r))[:4]


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

    m = re.search(
        r"\b([A-Za-z][A-Za-z0-9&-]{1,30})\s+would\s+be\s+my\s+pick\b",
        goal,
        flags=re.IGNORECASE,
    )
    if m and not plan.get("brand_soft"):
        brand = m.group(1).strip()
        if brand.lower() not in _FILLER:
            plan["brand_soft"] = brand

    m = re.search(
        r"\blean\s+toward\s+([A-Za-z][A-Za-z0-9&-]{1,30})\b",
        goal,
        flags=re.IGNORECASE,
    )
    if m and not plan.get("brand_soft"):
        brand = m.group(1).strip()
        if brand.lower() not in _FILLER:
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
    pairs: list[list[str]] = []
    if plan.get("category"):
        pairs.append(["product_type", str(plan["category"])])
    if plan.get("brand_firm"):
        pairs.append(["brand", str(plan["brand_firm"])])
    elif plan.get("brand_soft"):
        pairs.append(["brand", str(plan["brand_soft"])])
    for c in plan.get("constraints") or []:
        if isinstance(c, dict) and c.get("field") and c.get("value"):
            pairs.append([str(c["field"]), str(c["value"])])
    plan["keyword_pairs"] = pairs
    prefs = _parse_soft_preferences(goal)
    for k, v in prefs.items():
        if v:
            plan[k] = v
    # Soft model/color also become soft constraints for ranking (not hard-gate).
    if plan.get("preferred_model"):
        _add_constraint(plan, "model", plan["preferred_model"], firm=False)
    if plan.get("preferred_color"):
        _add_constraint(plan, "color", plan["preferred_color"], firm=False)
    if plan.get("preferred_model"):
        tokens = list(
            dict.fromkeys(
                (
                    [str(plan.get("preferred_color") or "").lower(), str(plan["preferred_model"]).lower()]
                    + keywords.split()
                )
            )
        )
        tokens = [t for t in tokens if t][:12]
        plan["keywords"] = " ".join(tokens)
        keywords = plan["keywords"]
        # Prefer model-focused rescues for recovery tasks.
        model_rescues = [
            " ".join(x for x in (plan.get("preferred_color"), plan.get("preferred_model")) if x),
            str(plan["preferred_model"]),
        ]
        plan["rescue_queries"] = list(
            dict.fromkeys([*model_rescues, *_progressive_rescues(keywords, plan)])
        )[:4]
    else:
        plan["rescue_queries"] = _progressive_rescues(keywords, plan)
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
    """LLM intent extraction — leopard A_DECOMPOSE MECE key→value on dynamic env."""
    js = _llm_json_once(
        _PV_INTENT,
        goal[:2000],
        role="intent",
        temperature=0.0,
    )
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
    pairs = _parse_keyword_pairs(js.get("keyword_pairs"))
    if pairs:
        out["keyword_pairs"] = pairs
        if not js.get("constraints"):
            lifted: list[dict[str, Any]] = []
            for key, val in pairs:
                if key == "product_type":
                    if not out.get("category"):
                        out["category"] = val
                    continue
                if key == "brand":
                    if not out.get("brand_firm") and not out.get("brand_soft"):
                        out["brand_firm"] = val
                    continue
                lifted.append({"field": key, "value": val, "firm": True})
            if lifted:
                js = dict(js)
                js["constraints"] = lifted
        pair_tokens = _pairs_to_search_tokens(pairs)
        if pair_tokens and not out.get("keywords"):
            out["keywords"] = " ".join(pair_tokens[:12])
    constraints: list[dict[str, Any]] = []
    for raw in js.get("constraints") or []:
        c = _coerce_constraint(raw)
        if c:
            constraints.append(c)
    out["constraints"] = constraints
    claim_fields: list[str] = []
    allowed = _allowed_claim_fields(out if isinstance(out, dict) else None)
    for raw in js.get("claim_fields") or []:
        field = str(raw or "").strip().lower().replace(" ", "_")
        field = _normalize_claim_field(field, allowed) or field
        if (
            field
            and field in allowed
            and field not in _SAFE_CLAIM_FIELDS
            and field not in claim_fields
        ):
            claim_fields.append(field)
    for c in constraints:
        field = str(c.get("field") or "")
        field = _normalize_claim_field(field, allowed) or field
        if (
            c.get("firm")
            and field
            and field in allowed
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
    if llm.get("keyword_pairs"):
        plan["keyword_pairs"] = llm["keyword_pairs"]
    if llm.get("constraints"):
        plan["constraints"] = llm["constraints"]
    if llm.get("claim_fields"):
        plan["claim_fields"] = llm["claim_fields"]
    if llm.get("rescue_queries"):
        merged = list(llm["rescue_queries"])
        for q in base.get("rescue_queries") or []:
            if q not in merged:
                merged.append(q)
        for q in _progressive_rescues(str(plan.get("keywords") or ""), plan):
            if q not in merged:
                merged.append(q)
        plan["rescue_queries"] = merged[:4]
    else:
        plan["rescue_queries"] = _progressive_rescues(str(plan.get("keywords") or ""), plan)
    if llm.get("spec") and (llm["spec"] or {}).get("field"):
        plan["spec"] = llm["spec"]
    if llm.get("needs_ranking"):
        plan["needs_ranking"] = True
    if base.get("ranking_priority") and not plan.get("ranking_priority"):
        plan["ranking_priority"] = base["ranking_priority"]
    if base.get("needs_clarify"):
        plan["needs_clarify"] = True
    tokens = _tokenize(str(plan.get("keywords") or ""))
    if plan.get("category"):
        tokens.insert(0, str(plan["category"]).lower())
    if plan.get("brand_firm"):
        tokens.insert(0, str(plan["brand_firm"]).lower())
    if plan.get("keyword_pairs"):
        for t in _pairs_to_search_tokens(plan["keyword_pairs"]):
            if t not in tokens:
                tokens.append(t)
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
        f"- keyword_pairs: {plan.get('keyword_pairs')}",
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
        f"- preferred_model: {plan.get('preferred_model')} | preferred_color: {plan.get('preferred_color')}",
        f"- exclude_terms: {plan.get('exclude_terms')}",
        f"- needs_recovery_ladder: {plan.get('needs_recovery_ladder')}",
    ]
    if plan.get("needs_recovery_ladder"):
        lines.append(
            "Recovery ladder (least relaxation first): keep budget+stock firm; "
            "try preferred model+color, then same model/other color, then other model. "
            "Always view/compare so model/color evidence is grounded before ordering."
        )
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
    allowed = _allowed_claim_fields(plan)
    for field in plan.get("claim_fields") or []:
        f = _normalize_claim_field(str(field or ""), allowed)
        if f and f not in req:
            req.append(f)
    spec_field = ((plan.get("spec") or {}) or {}).get("field")
    if spec_field:
        f = _normalize_claim_field(str(spec_field), allowed)
        if f and f not in req:
            req.append(f)
    for c in plan.get("constraints") or []:
        if not isinstance(c, dict) or not c.get("firm"):
            continue
        f = _normalize_claim_field(str(c.get("field") or ""), allowed)
        if f and f not in req:
            req.append(f)
    for field, _value in _extract_attr_cues(goal or ""):
        f = _normalize_claim_field(field, allowed)
        if f and f not in req:
            req.append(f)
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
    for pair in plan.get("keyword_pairs") or []:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            tokens.extend(_tokenize(str(pair[-1])))
    tokens = list(dict.fromkeys(tokens))
    max_price = plan.get("max_price")
    min_price = plan.get("min_price")
    brand_soft = _norm(plan.get("brand_soft") or "")
    brand_firm = _norm(plan.get("brand_firm") or "")
    category = _norm(plan.get("category") or "")
    firm_constraints = [
        c
        for c in (plan.get("constraints") or [])
        if isinstance(c, dict) and c.get("firm") and c.get("field") not in {"brand", "price"}
    ]
    scored: list[tuple[float, dict[str, Any]]] = []
    for rank, item in enumerate(results):
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
        # leopard Specs RRF + rank bonus on search position.
        score += _RRF_WEIGHT * (1.0 / (_RRF_K + rank + 1)) * 100.0
        score += max(0.0, 1.0 - rank / _RANK_LIMIT) * _RANK_BONUS_C
        price = item.get("price")
        if isinstance(price, (int, float)) and (
            max_price is not None or min_price is not None
        ):
            # Fuzzy budget edges: exact fit >> soft edge >> miss
            if _price_fits(price, max_price, min_price, fuzzy=False):
                score += 5.0
            elif _price_fits(price, max_price, min_price, fuzzy=True):
                score += 1.0
            else:
                score -= 25.0
        if item.get("in_stock") is True:
            score += 3.0
        elif item.get("in_stock") is False:
            score -= 20.0
        brand = _norm(item.get("brand") or "")
        if brand_firm:
            score += 8.0 if brand == brand_firm or brand_firm in _norm(title) else -12.0
        elif brand_soft and (brand == brand_soft or brand_soft in _norm(title)):
            score += 4.0
        pref_model = _norm(plan.get("preferred_model") or "")
        pref_color = _norm(plan.get("preferred_color") or "")
        if pref_model and pref_model in blob:
            score += 7.0
        if pref_color and pref_color in blob:
            score += 4.0
        for ex in plan.get("exclude_terms") or []:
            exn = _norm(ex)
            if exn and exn in blob:
                score -= 15.0
        if category and category in blob:
            score += 3.0
        for c in firm_constraints:
            field = str(c.get("field") or "")
            val = _norm(c.get("value"))
            if not val:
                continue
            hit = val in blob
            # Hard-gate identity keys weigh more (leopard ID_KEYS).
            boost = 6.0 if field in _GATE_HARD_KEYS else 4.0
            score += boost if hit else (-8.0 if field in _GATE_HARD_KEYS else -2.0)
        scored.append((score, item))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("product_id", ""))))
    # Extend shortlist on near-ties (leopard prune soft floor spirit).
    top_n = 5
    if scored and len(scored) > top_n:
        cutoff = scored[top_n - 1][0]
        tied = [(s, it) for s, it in scored if s >= cutoff - _RELAX_SCORE_GAP]
        if len(tied) > top_n:
            scored = tied
    lines = [
        "leopard RRF+fit shortlist (verify structured facts; do not order from title alone):"
    ]
    for score, item in scored[:6]:
        lines.append(
            f"- {score:.1f} id={item.get('product_id')} sku={item.get('sku')} "
            f"brand={item.get('brand')} price={item.get('price')} stock={item.get('in_stock')} "
            f"title={str(item.get('title') or '')[:90]}"
        )
    if scored and len(scored) > 1:
        gap = scored[0][0] - scored[1][0]
        if gap < _SELF_CONSISTENCY_GAP:
            ids = [str(it.get("product_id")) for _, it in scored[:3]]
            lines.append(
                f"Tight race (gap={gap:.2f}) — dual-path select among: {', '.join(ids)}"
            )
        elif scored[0][0] >= _FAST_ACCEPT_SCORE and gap >= _SELF_CONSISTENCY_GAP:
            top = scored[0][1]
            lines.append(
                f"Fast-accept candidate: product_id={top.get('product_id')} "
                f"sku={top.get('sku')} (high fit + clear gap)."
            )
    if scored and scored[0][0] < 4 and plan.get("rescue_queries"):
        lines.append(
            f"Weak pool — try progressive rescue searches: {plan.get('rescue_queries')}"
        )
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
    if payload.get("title") not in (None, ""):
        memory["title"] = str(payload.get("title"))
    product0 = payload.get("product")
    if isinstance(product0, dict) and product0.get("title"):
        memory["title"] = str(product0.get("title"))

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
        # Ban the last inspected/carted key after an environment failure (recovery events).
        bad = None
        if memory.get("inspect_product_id") and memory.get("inspect_sku"):
            bad = f"{memory['inspect_product_id']}::{memory['inspect_sku']}"
        elif memory.get("product_id") and memory.get("sku"):
            bad = f"{memory['product_id']}::{memory['sku']}"
        if bad:
            memory.setdefault("banned_keys", set()).add(bad)
            memory["event_target_key"] = bad

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
        key = f"{memory['inspect_product_id']}::{memory['inspect_sku']}"
        if payload.get("in_stock") is False or _norm(payload.get("in_stock")) in {
            "false",
            "0",
            "out of stock",
        }:
            memory.setdefault("banned_keys", set()).add(key)
            memory["event_target_key"] = key
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
    """Build verifier-safe claims from memory only (never invent/remap class tokens)."""
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
    allowed = _allowed_claim_fields(plan)
    required = [
        f
        for f in _required_claim_fields(str(plan.get("goal") or ""), plan)
        if f in allowed or f in _SAFE_CLAIM_FIELDS
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
        canon = _normalize_claim_field(field, allowed)
        if canon is None:
            return
        by_field[canon] = {"field": canon, "value": _claim_value(canon, value)}

    for field in required:
        if field in {"price", "in_stock", "brand"}:
            continue
        if field not in facts or sources.get(field) not in structured:
            continue
        observed = facts[field]
        want = _wanted_attr(plan, field)
        if want is not None and not _facts_agree(field, observed, want):
            continue
        # Prefer verifier/gate spelling for measure fields when observation agrees.
        if field in {"screen_size", "refresh_rate"} and want is not None:
            add_spec(field, want)
        elif field == "resolution" and want is not None:
            wn = _norm(want).replace(" ", "")
            on = _norm(observed).replace(" ", "")
            add_spec(field, want if wn and wn in on else observed)
        else:
            add_spec(field, observed)

    if (
        "brand" in facts
        and sources.get("brand") in structured
        and "brand" in allowed
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

        # Enforce firm attrs with _facts_agree (no invented tokens). Soft prefs
        # are checked separately for recovery evidence grounding.
        facts = memory.get("facts") or {}
        evidence_blob = _norm(
            " ".join(
                str(x)
                for x in (
                    memory.get("title"),
                    facts.get("model"),
                    facts.get("color"),
                    facts.get("color_family"),
                    facts.get("brand"),
                    *(str(v) for v in facts.values() if not isinstance(v, (dict, list))),
                )
                if x is not None
            )
        )
        for c in plan.get("constraints") or []:
            if not isinstance(c, dict) or not c.get("firm"):
                continue
            field = str(c.get("field") or "")
            want = c.get("value")
            if not field or want in (None, "") or field in {"brand", "category", "price", "in_stock"}:
                continue
            have = facts.get(field)
            if have is None:
                # Catalogs use varying keys — do not stall the episode. Ranking +
                # disagree-checks handle mismatches when the field is observed.
                continue
            if not _facts_agree(field, have, want):
                return (
                    f"Hard-gate fail: {field} want={want} observed={have}. "
                    "Recover: pick another firm-compliant option, view/inspect, then order.",
                    tool_calls,
                )

        # Recovery: require a view so model/color can be grounded; nudge once, then
        # allow least-relaxed alternatives without infinite blocks.
        pref_model = _norm(plan.get("preferred_model") or "")
        pref_color = _norm(plan.get("preferred_color") or "")
        if plan.get("needs_recovery_ladder") and (pref_model or pref_color):
            if not memory.get("saw_view") and "view" in tool_names:
                return (
                    "Recovery/preference task: view/compare the candidate so model/color "
                    "facts are grounded before place_test_order.",
                    tool_calls,
                )
            if (
                pref_model
                and pref_model not in evidence_blob
                and not memory.get("recovery_model_nudge")
            ):
                memory["recovery_model_nudge"] = True
                return (
                    f"Preferred model '{plan.get('preferred_model')}' not yet grounded in "
                    "viewed title/facts. Search/view a listing that evidences that model "
                    "(or same model/other color after an event), then inspect and order.",
                    tool_calls,
                )
            if (
                pref_color
                and pref_model
                and pref_model in evidence_blob
                and pref_color not in evidence_blob
                and not memory.get("recovery_color_nudge")
            ):
                memory["recovery_color_nudge"] = True
                return (
                    f"Preferred color '{plan.get('preferred_color')}' not grounded. "
                    "If unavailable, view another color of the same model (color-before-model), "
                    "then inspect_stock and order.",
                    tool_calls,
                )

        if "inspect_stock" in tool_names and not memory.get("saw_inspect_stock"):
            return (
                "Call inspect_stock on the chosen product_id/sku before place_test_order.",
                tool_calls,
            )

        pid = _sanitize_product_id(args.get("product_id")) or ""
        sku = str(args.get("sku") or "")
        if not pid:
            return (
                "place_test_order needs a valid product_id from search/view/inspect.",
                tool_calls,
            )
        order_key = f"{pid}::{sku}" if sku else pid
        banned = memory.get("banned_keys") or set()
        if order_key in banned or (
            memory.get("event_target_key") and order_key == memory.get("event_target_key")
        ):
            return (
                f"Do not reorder failed/event target {order_key}. Remove it if still in cart, "
                "then recover with least relaxation: same model/other color → other model. "
                "View+inspect a different in-stock product_id/sku before place_test_order.",
                tool_calls,
            )
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
            if order_key:
                memory.setdefault("banned_keys", set()).add(order_key)
                memory["event_target_key"] = order_key
            return (
                "Variant out of stock. Remove it from cart. Recovery ladder: "
                "same model+color if still listed → same model/other color → other model. "
                "View facts to ground the replacement, then inspect_stock and order.",
                tool_calls,
            )

        max_price = plan.get("max_price")
        min_price = plan.get("min_price")
        price = memory.get("facts", {}).get("price")
        if price is not None and (max_price is not None or min_price is not None):
            if not _price_fits(price, max_price, min_price, fuzzy=True):
                return (
                    f"Price {price} outside budget "
                    f"[min={min_price}, max={max_price}] (fuzzy gate). "
                    "Recover with another in-budget item.",
                    tool_calls,
                )

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
    plan["claim_allowlist"] = sorted(_claim_field_allowlist(tools))
    prefs = _parse_soft_preferences(goal)
    for key, val in prefs.items():
        if val and not plan.get(key):
            plan[key] = val

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
    else:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Justification required. Before the first add_to_cart: view/compare the "
                    "candidate and confirm firm specs from structured facts (not title SEO). "
                    "After cart/inspect, place_test_order WITH claims for price, in_stock, and "
                    "each firm attribute — use exact observed fact values only (never invent)."
                ),
            }
        )
    if plan.get("needs_recovery_ladder"):
        pref_bits = " ".join(
            x
            for x in (plan.get("preferred_color"), plan.get("preferred_model"))
            if x
        ).strip()
        messages.append(
            {
                "role": "user",
                "content": (
                    "Preference/recovery task. FIRST cart commitment: search/view/inspect/"
                    f"add the preferred option ({pref_bits or 'named model+color'}) when a "
                    "fully firm-compliant in-stock listing exists — this initial commitment matters. "
                    "If price/stock later fails: never reorder that failed product_id/sku; remove it; "
                    "least-relax to same model/other color before other models; view facts so "
                    "model/color evidence is grounded; inspect_stock; then order the replacement."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "provider": _provider(),
            "model": _model(),
            "plan": plan,
            "strategy": "leopard-dualpath-rrf-commit",
        }
    ]
    memory: dict[str, Any] = {
        "facts": {},
        "fact_norms": {},
        "fact_sources": {},
        "pending_intervention": False,
        "cart_keys": set(),
        "banned_keys": set(),
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
            memory["recovery_color_nudge"] = True
            memory["recovery_model_nudge"] = True
            banned = sorted(memory.get("banned_keys") or [])[:2]
            pref = " ".join(
                x
                for x in (plan.get("preferred_model"),)
                if x
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool error: {memory['last_error']}. "
                        f"Ban failed keys {banned or '[last inspect]'}. Remove bad cart items. "
                        f"Recover least-relax on model {pref or '(preferred model)'}: other color "
                        "first, then other models. View to ground evidence, inspect_stock a NEW "
                        "in-stock variant, then order "
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
