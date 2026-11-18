from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_proxy = ProxyClient(timeout=120, max_retries=2)

_EXTRACT_TRIES = 3
_BLOCKED_WHILE_PENDING = frozenset({"add_to_cart", "place_test_order"})

_SYSTEM = """You are OROScout, a general shopping agent for ORO Bench generated environments.
Use the STRATEGY SHEET: decomposed product_type, compact query, keywords, and
filter args are authoritative for search/filter. Prefer `filter` after a
search when category/budget/stock are known. Prefer `compare` when 2+ viewed
candidates remain. Follow SCORE_TOP / LLM_SHORTLIST / LLM_PICK when present.
Local constraint coverage + retrieval fusion rank candidates before LLM judge.


Use ONLY the tools supplied in this episode. Continue until an observation
reports done=true.

Universal discipline:
1. Honor HARD constraints from the STRATEGY SHEET and the shopper goal
   (category/type, budget, in-stock, brand/specs when stated as firm).
2. Soft preferences (preferred brand, use-case fit, exact model/color, theme
   phrase) guide search and ranking but never override hard constraints.
3. Verify with view/compare/inspect_stock using structured facts — do not rely
   on title guesses alone. Prefer category_path when a category is required.
   Firm gated specs can be ANY attribute named in the goal (size, RAM, color,
   capacity, material, resolution, etc.) — confirm that exact key in facts.
4. When pending_question is true, your NEXT action MUST be `message`. Never
   cart or order until you have answered.
5. On market price/stock events: `message` with NO question marks and without
   asking the shopper to choose → remove_from_cart → inspect a compliant
   backup → add_to_cart → place_test_order. Recover autonomously.
6. Reserve the last ~8 steps for verify / event recovery / order.
7. place_test_order: pass product_id + sku as one pair from the listing you
   inspected (never mix product_id from pick A with sku from listing B). Include
   justification.claims ONLY when strategy is fact_justify AND the sheet lists
   REQUIRED_CLAIMS. Claim values MUST be strings copied from viewed/inspected
   facts for THAT sku (any attribute name the sheet requires). For every other
   strategy, omit justification entirely — invented fields like device_type,
   brand=null, or use_case fail hard truthfulness gates even when the product
   is correct. After a successful add_to_cart with no market event, order that
   cart item immediately — do not remove/re-add thrash.

Strategy-specific guidance (follow the active strategy on the sheet):
- firm_vs_soft: firm specs/budget/stock beat preferred brand. When you name a
  pick in `message`, cart that same product unless a later firm constraint
  forces a messaged switch.
- theme_recall: search the exact theme/phrase first; reformulations must keep
  those tokens and must not invent a new product type (no serum/case/cover/
  housing unless the theme already contains that word). Run several distinct
  theme-preserving searches + category/budget filter before ordering; paginate
  when the tool allows.
- ask_then_constrain: if a firm spec will be revealed when asked, message to
  ask first; then satisfy every firm constraint precisely.
- preference_fit: among hard-valid options, optimize use-case fit (e.g. durability,
  battery, connectivity) — not cheapest-only. Clarify priorities before the
  first cart when the sheet says so. Compare multiple distinct listings.
  After a price spike, recover to a different underlying listing.
- closed_rank: only consider candidates listed in the goal/sheet; apply the
  stated lexicographic priority; after stockout re-rank remaining options.
- exact_then_relax: first cart the exact preferred model+color; after stockout
  relax minimally (usually other color of same model before changing models).
- fact_justify: first cart only when gated facts are explicit; order with
  string justification.claims covering required fields from owned facts.
- general: satisfy the goal with verify-then-order discipline.
"""

_STRATEGY_EXTRACT_PROMPT = """Analyze this shopping goal for a general shopping agent.
Return JSON only (no markdown). Infer a reusable strategy — do not assume a
fixed task-family label or problem id.

JSON schema:
{{
  "strategy": "firm_vs_soft" | "theme_recall" | "ask_then_constrain" | "preference_fit" | "closed_rank" | "exact_then_relax" | "fact_justify" | "general",
  "category": <string|null>,
  "budget_max": <number|null>,
  "in_stock": <true|false>,
  "firm_brand": <string|null>,
  "preferred_brand": <string|null>,
  "theme_phrase": <string|null>,
  "primary_query": <string|null>,
  "reformulations": [<strings that keep theme tokens>],
  "gated_attribute": {{"key": <string|null>, "value": <string|null>}},
  "underspecified": <true|false>,
  "needs_priority_clarification": <true|false>,
  "use_case": <string|null>,
  "soft_priorities": [<strings>],
  "candidates": [{{"product_id": <string>, "sku": <string>}}],
  "rank_priority": [<attribute names from the goal, highest priority first>],
  "preferred_model": <string|null>,
  "preferred_color": <string|null>,
  "relaxation_hint": <string|null>,
  "required_claims": [<attribute names that must be evidenced on order>],
  "attributes": {{<any firm attr name from the goal: value>}},
  "notes": <string>
}}

Guidance:
- gated_attribute.key/value: the single firm product spec if one is stated
  (examples of keys — not a closed list: screen_size, ram, capacity, color,
  material, resolution, battery, water_resistance, …). Use null when none.
- attributes: map of ALL firm named specs from the goal (same open vocabulary).
- required_claims: only when the shopper asks for an evidenced reason; include
  in_stock, price, and whatever gated/firm attrs must be justified. Do NOT add
  category/device_type/use_case.
- rank_priority: copy the goal's lexicographic axes in order; do not invent
  monitor/phone axes when the goal does not name them.
- theme_recall when an exact quoted/theme phrase must match listings.
- closed_rank when the goal lists specific product_id::sku candidates and a
  lexicographic tradeoff.
- ask_then_constrain when a firm requirement will be told if you ask.
- exact_then_relax when an exact model+color is preferred with budget/stock firm.
- preference_fit when soft multi-axis use-case preferences matter more than
  buying the cheapest hard-valid item.
- fact_justify when the shopper asks for a trustworthy reason / evidence-backed
  justification on the order.
- firm_vs_soft when there is a soft brand preference under firm specs/budget.
- Otherwise general.

Goal:
{query}
"""

_UPDATE_EXTRACT_PROMPT = """Update the shopping strategy sheet from a shopper message.
Return JSON only. null means unchanged (except preferred_brand "" clears it).

Current sheet:
{sheet}

Shopper message:
{message}

JSON schema:
{{
  "budget_max": <number|null>,
  "gated_attribute": {{"key": <string|null>, "value": <string|null>}} | null,
  "preferred_brand": <string|null>,
  "preferred_model": <string|null>,
  "preferred_color": <string|null>,
  "soft_priorities": [<strings>] | null,
  "is_question": <true|false>,
  "notes": <string|null>
}}
"""

_DECOMPOSE_PROMPT = """Decompose this shopping goal into search/filter building blocks.
Ignore any "Task rules:" boilerplate. Return JSON only.

Goal:
{query}

JSON schema:
{{
  "product_type": <short type noun(s) from the goal|null>,
  "compact_query": <search-friendly rewrite: keep type + theme + firm specs; drop budget, stock, task rules, filler>,
  "keywords": [<3 to 8 indexed VALUE tokens: type, brand, theme, specs — not JSON-key words like size/color/brand/category>],
  "alt_keywords": [<same list with exactly one add OR one remove, or empty if no gain>],
  "attributes": {{<open-vocab firm attr: compact value>}},
  "filter": {{
    "category": <catalog category if stated|null>,
    "max_price": <number|null>,
    "in_stock": <true|false>,
    "brand": <firm brand only|null>
  }},
  "paraphrases": [<0-3 token swaps that already appear, e.g. gray/grey, t-shirt/tshirt>]
}}

Rules:
- compact_query must keep quoted/theme tokens verbatim when present.
- Do not invent a product type absent from the goal (no serum/case/cover unless said).
- keywords are value-side only; do not add size/color/material as bare keys.
- attributes must be mutually exclusive and collectively cover firm specs only.
"""

_KEYWORD_ALT_PROMPT = """The catalog indexes JSON values (not keys). Given the current keywords,
recommend AT MOST one alternative set by adding OR removing exactly one token.
If no change is clearly better, return alt=null.

Query: {query}
Keywords: {keywords}

JSON only:
{{"rationale": <str>, "alt": <full keyword string or null>}}
"""

_FILTER_PROMPT = """Precision-filter catalog candidates against REQUIRED constraints.
A candidate PASSes only if every required constraint is PASS (UNKNOWN = FAIL).

Required constraints (from the shopper, not invented):
{constraints}

Matching:
- Literal for brands, models, codes, numbers, colors, quoted theme tokens.
- Semantic for other words (kid=child, waterproof=water-resistant).
- Title is an unordered token bag: token presence counts; do not over-parse SEO titles.
- Structured facts/attributes are stronger evidence than title.

Candidates:
{products}

JSON only:
{{
  "verdicts": [{{"id": <product_id or product_id::sku>, "pass": <bool>, "note": <short PASS/FAIL list>}}],
  "shortlist": [<ids that pass>]
}}
"""

_SELECT_PROMPT = """Select the single best candidate that satisfies ALL required constraints.
Prefer: structured fact evidence over title; exact wording; explicit gated specs;
then the stated ranking/use-case priorities. Do not invent missing specs.

Required constraints:
{constraints}

Priorities / use-case (soft):
{priorities}

Candidates:
{products}

JSON only:
{{
  "best_id": <product_id>,
  "best_sku": <sku or null>,
  "rationale": <one sentence>
}}
"""


@dataclass
class StrategySheet:
    strategy: str = "general"
    category: str | None = None
    budget: float | None = None
    in_stock_required: bool = True
    firm_brand: str | None = None
    preferred_brand: str | None = None
    theme_phrase: str | None = None
    primary_query: str | None = None
    reformulations: list[str] = field(default_factory=list)
    gated_key: str | None = None
    gated_value: str | None = None
    underspecified: bool = False
    needs_priority_clarification: bool = False
    asked_clarify: bool = False
    use_case: str | None = None
    soft_priorities: list[str] = field(default_factory=list)
    candidates: list[dict[str, str]] = field(default_factory=list)
    rank_priority: list[str] = field(default_factory=list)
    preferred_model: str | None = None
    preferred_color: str | None = None
    relaxation_hint: str | None = None
    required_claims: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    pending_question: bool = False
    last_shopper_text: str | None = None
    last_product_id: str | None = None
    last_sku: str | None = None
    committed_product_id: str | None = None
    committed_sku: str | None = None
    event_recovery_mode: bool = False
    saw_market_event: bool = False
    candidate_facts: dict[str, dict[str, Any]] = field(default_factory=dict)
    first_cart_done: bool = False
    computed_top_id: str | None = None
    computed_top_sku: str | None = None
    search_count: int = 0
    product_type: str | None = None
    compact_query: str | None = None
    keywords: list[str] = field(default_factory=list)
    alt_keywords: list[str] = field(default_factory=list)
    filter_args: dict[str, Any] = field(default_factory=dict)
    shortlist: list[str] = field(default_factory=list)
    llm_pick_id: str | None = None
    llm_pick_sku: str | None = None
    scored_keys: set[str] = field(default_factory=set)
    llm_select_done: bool = False
    # extreme-inspired local scoring
    candidate_scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    search_ranks: dict[str, int] = field(default_factory=dict)
    rrf_scores: dict[str, float] = field(default_factory=dict)
    search_pass: int = 0
    constraint_axes: list[tuple[str, str]] = field(default_factory=list)
    # Cart pair integrity (TF4/TF7 failure: product_id overwritten, sku left stale)
    last_inspected_id: str | None = None
    last_inspected_sku: str | None = None
    cart_has_item: bool = False
    cart_product_id: str | None = None
    cart_sku: str | None = None
    cart_fail_streak: int = 0
    def status_block(self) -> str:
        lines = [
            f"STRATEGY: {self.strategy}",
            "STRATEGY SHEET (authoritative for this episode):",
            "HARD: "
            + "; ".join(
                [
                    f"category={self.category or 'unknown'}",
                    f"budget<={self.budget if self.budget is not None else 'unknown'}",
                    f"in_stock={self.in_stock_required}",
                    f"firm_brand={self.firm_brand or 'any'}",
                    f"gated={self.gated_key or 'attr'}:{self.gated_value or 'unknown'}",
                ]
            ),
        ]
        if self.preferred_brand:
            lines.append(f"SOFT preferred_brand={self.preferred_brand}")
        if self.theme_phrase:
            lines.append(f"THEME: {self.theme_phrase}")
        if self.product_type:
            lines.append(f"PRODUCT_TYPE: {self.product_type}")
        if self.compact_query:
            lines.append(f"COMPACT_QUERY: {self.compact_query}")
        if self.keywords:
            lines.append("KEYWORDS: " + " ".join(self.keywords))
        if self.alt_keywords:
            lines.append("ALT_KEYWORDS: " + " ".join(self.alt_keywords))
        if self.filter_args:
            lines.append(f"FILTER_ARGS: {json.dumps(self.filter_args)}")
        qplan = _search_plan(self)
        if qplan:
            lines.append("SEARCH PLAN (run in order, then filter):")
            for i, q in enumerate(qplan[:6], start=1):
                lines.append(f"  {i}. {q}")
        if self.shortlist:
            lines.append("LLM_SHORTLIST: " + ", ".join(self.shortlist[:8]))
        top = _top_scored_keys(self, limit=5)
        if top:
            lines.append("SCORE_TOP (coverage+rrf; prefer view/inspect these):")
            for key in top:
                sc = self.candidate_scores.get(key) or {}
                lines.append(
                    f"  - {key} cov={sc.get('coverage', 0):.2f} "
                    f"fit={sc.get('score_n', 0):.2f} miss={sc.get('missing', [])}"
                )
        if self.llm_pick_id:
            lines.append(
                f"LLM_PICK: {self.llm_pick_id}::{self.llm_pick_sku or 'unknown'} "
                "(view/inspect this unless event recovery needs a different backup)"
            )
        if self.last_inspected_id:
            lines.append(
                f"LAST_INSPECTED: {self.last_inspected_id}::"
                f"{self.last_inspected_sku or '?'} "
                "(add_to_cart MUST use this exact pair)"
            )
        if self.cart_has_item and self.cart_product_id:
            lines.append(
                f"CART_HAS: {self.cart_product_id}::{self.cart_sku or '?'} "
                "→ place_test_order this pair next"
            )
        if self.use_case:
            lines.append(f"USE_CASE: {self.use_case}")
        if self.soft_priorities:
            lines.append("SOFT_PRIORITIES: " + ", ".join(self.soft_priorities[:8]))
        if self.candidates:
            lines.append("CANDIDATES (closed set when strategy=closed_rank):")
            for c in self.candidates[:12]:
                lines.append(f"  - {c.get('product_id')}::{c.get('sku')}")
        if self.rank_priority:
            lines.append("RANK_PRIORITY: " + " > ".join(self.rank_priority))
        if self.preferred_model or self.preferred_color:
            lines.append(
                "PREFERRED: "
                f"model={self.preferred_model or 'unknown'}; "
                f"color={self.preferred_color or 'unknown'}; "
                f"relax={self.relaxation_hint or 'minimal / color before model'}"
            )
        if self.attributes:
            lines.append(f"ATTRIBUTES: {json.dumps(self.attributes)}")
        if self.required_claims:
            lines.append(
                "REQUIRED_CLAIMS (string values from owned facts only): "
                + ", ".join(self.required_claims)
            )
        else:
            lines.append("REQUIRED_CLAIMS: none — omit justification on order.")
        if self.underspecified and not self.gated_value and not self.asked_clarify:
            lines.append(
                "UNDERSPECIFIED: ask via `message` for the missing firm spec "
                "before carting."
            )
        if (
            self.needs_priority_clarification
            and not self.asked_clarify
            and not self.event_recovery_mode
        ):
            lines.append(
                "CLARIFY: message about soft priority tradeoffs before first cart."
            )
        if self.committed_product_id:
            lines.append(
                "NAMED_PICK: "
                f"product_id={self.committed_product_id} "
                f"sku={self.committed_sku or 'unknown'} "
                "(keep if still hard-compliant; message before switching)"
            )
        lines.append(
            f"pending_question: {'yes' if self.pending_question else 'no'}"
        )
        if self.event_recovery_mode:
            lines.append(
                "EVENT RECOVERY: message with NO '?', then remove → inspect "
                "backup → cart → order. Do not ask the shopper to choose."
            )
        if self.last_shopper_text:
            lines.append(f"last_shopper_message: {self.last_shopper_text[:240]}")
        if self.pending_question:
            lines.append(
                "ACTION REQUIRED NOW: call `message` and answer the shopper."
            )
        if self.notes:
            lines.append("NOTES: " + " | ".join(self.notes[-3:]))
        return "\n".join(lines)


def _model() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(
        provider, _DEFAULT_MODELS["chutes"]
    )


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _user_text(user_message: Any) -> str | None:
    if user_message is None:
        return None
    if isinstance(user_message, str):
        text = user_message.strip()
        return text or None
    if isinstance(user_message, dict):
        content = user_message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def _observation_blob(observation: Any) -> str:
    try:
        return json.dumps(observation, ensure_ascii=False)
    except Exception:
        return str(observation)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("no JSON object in model response")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON payload is not an object")
    return data


def _as_float(value: Any) -> float | None:
    if value is None or value is False:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", value.replace(",", ""))
            if m:
                return float(m.group(1))
    return None


def _as_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "yes", "1"}:
            return True
        if lower in {"false", "no", "0"}:
            return False
    return default


def _theme_from_query(query: str) -> str | None:
    patterns = (
        r'matches\s+"([^"]+)"',
        r"matches\s+'([^']+)'",
        r"matches\s+the\s+phrase\s+([^,\n]+?)(?:,|\s+at\s+or|\s+and\s+buy)",
        r'match(?:es)?\s+[“"]([^”"]+)[”"]',
    )
    for pat in patterns:
        match = re.search(pat, query, flags=re.I)
        if match:
            theme = match.group(1).strip().strip("'\"")
            if theme:
                return theme
    return None


def _theme_tokens(theme: str) -> list[str]:
    return [
        t
        for t in re.split(r"\s+", theme.lower())
        if t and t not in {"a", "an", "the", "and", "or", "for"}
    ]


_INVENTED_TYPE_WORDS = frozenset(
    {
        "cutter",
        "cutting",
        "housing",
        "panel",
        "replacement",
        "tool",
        "tools",
        "pipe",
        "cement",
        "tape",
        "connector",
        "serum",
        "toner",
        "cream",
        "lotion",
        "case",
        "cover",
        "charger",
        "cable",
        "gamepad",
        "pad",
        "skincare",
    }
)


def _reformulation_keeps_theme(text: str, theme: str) -> bool:
    tokens = _theme_tokens(theme)
    if not tokens:
        return True
    lower = text.lower()
    need = len(tokens) if len(tokens) <= 2 else max(2, (len(tokens) + 1) // 2)
    hit = sum(1 for t in tokens if t in lower)
    return hit >= need


def _introduces_invented_type(text: str, theme: str) -> bool:
    theme_l = theme.lower()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in _INVENTED_TYPE_WORDS and word not in theme_l:
            return True
    return False


def _sanitize_theme_queries(sheet: StrategySheet, query: str) -> None:
    heuristic = _theme_from_query(query)
    if heuristic:
        sheet.theme_phrase = heuristic
    if not sheet.theme_phrase:
        return
    theme = sheet.theme_phrase.strip()
    sheet.primary_query = theme
    cleaned: list[str] = []
    for item in sheet.reformulations:
        text = item.strip()
        if not text or text.lower() == theme.lower():
            continue
        if not _reformulation_keeps_theme(text, theme):
            continue
        if _introduces_invented_type(text, theme):
            continue
        if text not in cleaned:
            cleaned.append(text)
    extras = [f'"{theme}"', theme.upper() if theme != theme.upper() else theme]
    words = theme.split()
    if len(words) >= 2:
        extras.append(" ".join(reversed(words)))
        extras.append(" ".join(words[:2]))
    if sheet.category and sheet.category.lower() not in theme.lower():
        extras.append(f"{theme} {sheet.category}")
    for extra in extras:
        extra = extra.strip()
        if (
            extra
            and extra.lower() != theme.lower()
            and _reformulation_keeps_theme(extra, theme)
            and not _introduces_invented_type(extra, theme)
            and extra not in cleaned
        ):
            cleaned.append(extra)
    sheet.reformulations = cleaned[:6]


# ---------------------------------------------------------------------------
# Extreme-inspired local scoring (coverage, synonyms, RRF, prune-before-LLM)
# Adapted for ORO Bench facts blobs — no Lazada shop/voucher stack.
# ---------------------------------------------------------------------------
_SCOUT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "with",
        "from",
        "to",
        "of",
        "in",
        "on",
        "by",
        "at",
        "as",
        "is",
        "be",
        "suitable",
        "compatible",
        "about",
        "above",
        "below",
        "under",
        "over",
        "than",
        "less",
        "more",
        "within",
        "between",
    }
)
_SCOUT_SYNONYMS: list[tuple[str, str]] = [
    (a, b)
    for a, b in [
        ("grey", "gray"),
        ("colour", "color"),
        ("colours", "colors"),
        ("waterproof", "water-resistant"),
        ("waterproof", "water resistant"),
        ("water-proof", "waterproof"),
        ("shockproof", "shock-resistant"),
        ("dustproof", "dust-resistant"),
        ("wifi", "wi-fi"),
        ("tshirt", "t-shirt"),
        ("freesize", "one size"),
        ("freesize", "onesize"),
        ("kid", "child"),
        ("kids", "children"),
        ("pcs", "pieces"),
        ("pc", "piece"),
        ("hd", "720p"),
        ("fhd", "1080p"),
        ("uhd", "4k"),
        ("ipx4", "waterproof"),
        ("ipx5", "waterproof"),
        ("ipx7", "waterproof"),
        ("bluetooth", "wireless"),
        ("earhook", "ear hook"),
        ("earbuds", "earphones"),
        ("tws", "wireless"),
    ]
]
# Preference-fit semantic axes derived from use-case language (open vocab).
_PREF_SIGNAL_GROUPS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (
        ("sweat", "water", "waterproof", "rain", "weather", "ipx"),
        ("sweat water resistance", "waterproof", "water resistant", "ipx4"),
    ),
    (
        ("sport", "secure", "fit", "earhook", "workout", "bootcamp", "run"),
        ("sport secure fit", "ear hook", "secure fit"),
    ),
    (
        ("battery", "charge", "mah", "long lasting"),
        ("battery life", "long battery"),
    ),
    (
        ("wireless", "bluetooth", "tws", "cordless"),
        ("wireless", "bluetooth"),
    ),
]
_COVERAGE_PRUNE_FLOOR = 0.35
_RRF_K = 60.0
_RRF_WEIGHT = 0.15
_RANK_BONUS_C = 0.05
_RANK_LIMIT = 40


def _normalize_scout(text: str) -> str:
    raw = str(text or "").lower()
    raw = raw.replace("→", " ").replace("_", " ")
    raw = re.sub(r"[^\w\s\.\-\+/]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _syn_expand(token: str) -> list[str]:
    out = [token]
    t = token.lower()
    for a, b in _SCOUT_SYNONYMS:
        if a == t and b not in out:
            out.append(b)
        if b == t and a not in out:
            out.append(a)
    return out


def _is_kw_in(token: str, haystack: str, *, as_is: bool = False) -> bool:
    if not token or not haystack:
        return False
    t = _normalize_scout(token)
    h = _normalize_scout(haystack)
    if not t:
        return False
    if as_is or t.isnumeric():
        return re.search(rf"(?<!\d){re.escape(t)}(?!\d)", h) is not None
    for cand in _syn_expand(t):
        c = _normalize_scout(cand)
        if not c:
            continue
        if " " in c:
            if c in h:
                return True
        elif re.search(rf"\b{re.escape(c)}\b", h):
            return True
    return False


def _facts_blob(facts: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, val in facts.items():
        if key == "evidence_text":
            continue
        parts.append(f"{key} {val}")
    if isinstance(facts.get("evidence_text"), str):
        parts.append(facts["evidence_text"])
    return _normalize_scout(" ".join(parts))


def _constraint_axes(sheet: StrategySheet) -> list[tuple[str, str]]:
    axes: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(key: str, val: Any) -> None:
        if val is None:
            return
        text = str(val).strip()
        if not text:
            return
        sig = f"{key}:{text.lower()}"
        if sig in seen:
            return
        seen.add(sig)
        axes.append((key, text))

    if sheet.theme_phrase:
        add("theme", sheet.theme_phrase)
    if sheet.product_type:
        add("product_type", sheet.product_type)
    if sheet.firm_brand:
        add("brand", sheet.firm_brand)
    elif sheet.preferred_brand and sheet.strategy != "firm_vs_soft":
        add("brand_soft", sheet.preferred_brand)
    if sheet.gated_key and sheet.gated_value:
        add(sheet.gated_key, sheet.gated_value)
    for key, val in (sheet.attributes or {}).items():
        if key.lower() in {"brand", "category"}:
            continue
        add(key, val)
    if sheet.preferred_model:
        add("model", sheet.preferred_model)
    if sheet.preferred_color:
        add("color", sheet.preferred_color)
    for kw in sheet.keywords:
        tok = str(kw).strip()
        if tok and tok.lower() not in _SCOUT_STOPWORDS:
            add("keyword", tok)
    # Soft preference axes (extreme-style semantic tokens).
    for p in sheet.soft_priorities:
        add("soft", p)
    if sheet.use_case:
        add("use_case", sheet.use_case)
    # Derive preference axes from use-case / soft text (TF4 multi-axis fit).
    if sheet.strategy == "preference_fit":
        blob = _normalize_scout(
            " ".join(
                [
                    sheet.use_case or "",
                    " ".join(sheet.soft_priorities or []),
                    sheet.last_shopper_text or "",
                ]
            )
        )
        for triggers, signals in _PREF_SIGNAL_GROUPS:
            if any(t in blob for t in triggers):
                for sig in signals:
                    add("soft", sig)
    sheet.constraint_axes = axes
    return axes


def _axis_hit(key: str, val: str, facts: dict[str, Any], blob: str) -> bool:
    # Prefer structured fact for the named key.
    if key not in {"theme", "keyword", "soft", "use_case", "brand_soft", "product_type"}:
        got = _lookup_fact(facts, key)
        if got is not None and _values_compatible(got, val):
            return True
    tokens = [
        t
        for t in _normalize_scout(val).split()
        if t and t not in _SCOUT_STOPWORDS
    ]
    if not tokens:
        return _is_kw_in(val, blob)
    # Literal-ish: numbers/colors must all hit; otherwise require most tokens.
    hard = sum(1 for t in tokens if t.isnumeric() or len(t) <= 2)
    hits = sum(1 for t in tokens if _is_kw_in(t, blob, as_is=t.isnumeric()))
    if hard:
        return hits == len(tokens)
    need = len(tokens) if len(tokens) <= 2 else max(2, (len(tokens) + 1) // 2)
    return hits >= need


def _score_facts(
    sheet: StrategySheet, key: str, facts: dict[str, Any], *, rank: int | None = None
) -> dict[str, Any]:
    axes = sheet.constraint_axes or _constraint_axes(sheet)
    blob = _facts_blob(facts)
    detail: dict[str, float] = {}
    missing: list[str] = []
    matched = 0
    for ax_key, ax_val in axes:
        ok = _axis_hit(ax_key, ax_val, facts, blob)
        # Soft axes weigh less when missing.
        weight = 0.55 if ax_key in {"soft", "use_case", "brand_soft", "keyword"} else 1.0
        if ax_key == "theme":
            weight = 1.25
        if ok:
            detail[f"{ax_key}:{ax_val}"] = weight
            matched += 1
        else:
            detail[f"{ax_key}:{ax_val}"] = 0.0
            if ax_key not in {"soft", "use_case", "brand_soft"}:
                missing.append(f"{ax_key}={ax_val}")

    total_w = sum(1.25 if k.startswith("theme:") else (0.55 if any(k.startswith(p) for p in ("soft:", "use_case:", "brand_soft:", "keyword:")) else 1.0) for k in detail) or 1.0
    score = sum(detail.values())
    coverage = score / total_w if total_w else 1.0

    # Hard commerce gates as multipliers (extreme basic_check spirit).
    hard_ok = True
    if sheet.in_stock_required and facts.get("in_stock") is False:
        hard_ok = False
        coverage *= 0.15
    price = _as_float(facts.get("price"))
    if sheet.budget is not None and price is not None and price > sheet.budget:
        hard_ok = False
        coverage *= 0.2
    if sheet.firm_brand:
        brand = _lookup_fact(facts, "brand")
        if brand is not None and not _values_compatible(brand, sheet.firm_brand):
            hard_ok = False
            coverage *= 0.25

    use_rank = rank if rank is not None else sheet.search_ranks.get(key, _RANK_LIMIT)
    rank_bonus = max(1.0 - use_rank / _RANK_LIMIT, 0.0) * _RANK_BONUS_C
    rrf = sheet.rrf_scores.get(key, 0.0)
    top_rrf = max(sheet.rrf_scores.values()) if sheet.rrf_scores else 1.0
    rrf_n = (rrf / top_rrf) if top_rrf else 0.0
    score_n = coverage + rank_bonus + _RRF_WEIGHT * rrf_n
    if sheet.strategy == "preference_fit":
        # Stronger soft boosts: prefer listings with multiple use-case axes,
        # not title-only single-signal cheapest picks.
        soft_hits = 0
        soft_axes = [
            ax_val
            for ax_key, ax_val in axes
            if ax_key in {"soft", "use_case"}
        ]
        if not soft_axes:
            soft_axes = list(sheet.soft_priorities or [])
            if sheet.use_case:
                soft_axes.append(sheet.use_case)
        for p in soft_axes:
            if _axis_hit("soft", p, facts, blob):
                soft_hits += 1
        score_n += 0.12 * soft_hits
        # Prefer structured attr hits over title-only: bonus when soft tokens
        # appear in non-title fact keys.
        structured = " ".join(
            f"{k} {v}"
            for k, v in facts.items()
            if k not in {"title", "evidence_text"} and v is not None
        )
        struct_hits = sum(
            1 for p in soft_axes if _is_kw_in(str(p).split()[0], structured)
        )
        score_n += 0.06 * struct_hits

    return {
        "coverage": coverage,
        "score": score,
        "score_n": score_n,
        "matched": matched,
        "total": len(axes),
        "missing": missing[:6],
        "hard_ok": hard_ok,
        "detail": detail,
        "rank": use_rank,
    }


def _rescore_all(sheet: StrategySheet) -> None:
    _constraint_axes(sheet)
    for key, facts in sheet.candidate_facts.items():
        sheet.candidate_scores[key] = _score_facts(sheet, key, facts)


def _top_scored_keys(
    sheet: StrategySheet, *, limit: int = 8, min_coverage: float | None = None
) -> list[str]:
    if not sheet.candidate_scores and sheet.candidate_facts:
        _rescore_all(sheet)
    items = list(sheet.candidate_scores.items())
    if min_coverage is not None:
        kept = [(k, s) for k, s in items if float(s.get("coverage") or 0) >= min_coverage]
        if kept:
            items = kept
    items.sort(
        key=lambda kv: (
            0 if kv[1].get("hard_ok", True) else 1,
            -float(kv[1].get("score_n") or 0),
            -float(kv[1].get("coverage") or 0),
        )
    )
    return [k for k, _ in items[:limit]]


def _resolve_fact_key(sheet: StrategySheet, cid: str) -> str | None:
    text = (cid or "").strip()
    if not text:
        return None
    if text in sheet.candidate_facts:
        return text
    matches = [k for k in sheet.candidate_facts if k == text or k.startswith(f"{text}::")]
    if len(matches) == 1:
        return matches[0]
    if matches:
        for k in matches:
            if "::" in k:
                return k
        return matches[0]
    if "::" in text:
        pid = text.split("::", 1)[0]
        if pid in sheet.candidate_facts:
            return pid
    return None


def _note_search_hit(sheet: StrategySheet, key: str, rank: int) -> None:
    prev = sheet.search_ranks.get(key, 10**9)
    if rank < prev:
        sheet.search_ranks[key] = rank
    sheet.rrf_scores[key] = sheet.rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)


def _goal_only(query: str) -> str:
    return (query.split("Task rules:")[0] or query).strip()


def _token_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [t for t in re.split(r"\s+", value.strip()) if t]
    return []


def _search_plan(sheet: StrategySheet) -> list[str]:
    plan: list[str] = []
    seen: set[str] = set()

    def add(q: str | None) -> None:
        if not q:
            return
        text = " ".join(q.split())
        key = text.lower()
        if not text or key in seen:
            return
        if sheet.theme_phrase:
            if not _reformulation_keeps_theme(text, sheet.theme_phrase):
                return
            if _introduces_invented_type(text, sheet.theme_phrase):
                return
        seen.add(key)
        plan.append(text)

    add(sheet.theme_phrase)
    add(sheet.primary_query)
    add(sheet.compact_query)
    if sheet.keywords:
        add(" ".join(sheet.keywords))
    if sheet.alt_keywords:
        add(" ".join(sheet.alt_keywords))
    for q in sheet.reformulations:
        add(q)
    if sheet.product_type and sheet.theme_phrase:
        add(f"{sheet.theme_phrase} {sheet.product_type}")
    return plan


def _constraint_lines(sheet: StrategySheet) -> str:
    parts = [
        f"category={sheet.category or 'any'}",
        f"budget<={sheet.budget if sheet.budget is not None else 'any'}",
        f"in_stock={sheet.in_stock_required}",
        f"firm_brand={sheet.firm_brand or 'any'}",
        f"theme={sheet.theme_phrase or 'none'}",
        f"product_type={sheet.product_type or 'unknown'}",
        f"gated={sheet.gated_key or 'attr'}:{sheet.gated_value or 'unknown'}",
    ]
    if sheet.attributes:
        parts.append("attrs=" + json.dumps(sheet.attributes))
    return "; ".join(parts)


def _render_candidates_for_llm(
    sheet: StrategySheet, keys: list[str] | None = None, limit: int = 8
) -> str:
    items: list[str] = []
    if keys is None:
        keys = _top_scored_keys(sheet, limit=limit)
    for key in keys[:limit]:
        facts = sheet.candidate_facts.get(key)
        if not isinstance(facts, dict):
            continue
        slim = {
            k: v
            for k, v in facts.items()
            if k != "evidence_text"
            and (
                k
                in {
                    "title",
                    "brand",
                    "price",
                    "in_stock",
                    "category",
                    "category_path",
                }
                or k in sheet.attributes
                or k == sheet.gated_key
            )
        }
        extra = 0
        for k, v in facts.items():
            if k in slim or k in {"title", "evidence_text"}:
                continue
            slim[k] = v
            extra += 1
            if extra >= 8:
                break
        sc = sheet.candidate_scores.get(key) or {}
        slim["_scout"] = {
            "coverage": round(float(sc.get("coverage") or 0), 3),
            "score_n": round(float(sc.get("score_n") or 0), 3),
            "missing": sc.get("missing") or [],
            "hard_ok": sc.get("hard_ok", True),
        }
        items.append(f"{key}: {json.dumps(slim, ensure_ascii=False, default=str)}")
    return "\n".join(items) if items else "(none)"


def _apply_decompose(sheet: StrategySheet, query: str) -> None:
    try:
        data = _chat_json(_DECOMPOSE_PROMPT.format(query=_goal_only(query)))
    except Exception:
        sheet.notes.append("decompose_failed")
        return

    ptype = data.get("product_type")
    if isinstance(ptype, str) and ptype.strip():
        sheet.product_type = ptype.strip()
        if not sheet.category:
            sheet.category = ptype.strip()

    compact = data.get("compact_query")
    if isinstance(compact, str) and compact.strip():
        sheet.compact_query = compact.strip()
        if not sheet.primary_query:
            sheet.primary_query = sheet.compact_query

    kws = _token_list(data.get("keywords"))
    if kws:
        sheet.keywords = kws[:8]
    alts = _token_list(data.get("alt_keywords"))
    if alts and alts != sheet.keywords:
        sheet.alt_keywords = alts[:8]

    attrs = data.get("attributes")
    if isinstance(attrs, dict):
        for key, val in attrs.items():
            name = str(key).strip()
            if not name or val is None or not str(val).strip():
                continue
            if name.lower() in {"category", "device_type", "use_case"}:
                continue
            sheet.attributes.setdefault(name, str(val).strip())
        if not sheet.gated_key:
            for key, val in sheet.attributes.items():
                if key.lower() in {"brand", "category"}:
                    continue
                sheet.gated_key = key
                sheet.gated_value = val
                break

    filt = data.get("filter")
    if isinstance(filt, dict):
        cleaned: dict[str, Any] = {}
        cat = filt.get("category")
        if isinstance(cat, str) and cat.strip():
            cleaned["category"] = cat.strip()
            if not sheet.category:
                sheet.category = cat.strip()
        price = _as_float(filt.get("max_price"))
        if price is not None:
            cleaned["max_price"] = price
            if sheet.budget is None:
                sheet.budget = price
        instock = _as_bool(filt.get("in_stock"), default=None)
        if instock is not None:
            cleaned["in_stock"] = instock
        brand = filt.get("brand")
        if isinstance(brand, str) and brand.strip():
            cleaned["brand"] = brand.strip()
            if not sheet.firm_brand:
                sheet.firm_brand = brand.strip()
        sheet.filter_args = cleaned

    if sheet.strategy == "theme_recall" or sheet.theme_phrase:
        _sanitize_theme_queries(sheet, query)
    if sheet.compact_query and sheet.theme_phrase:
        if not _reformulation_keeps_theme(sheet.compact_query, sheet.theme_phrase):
            sheet.compact_query = sheet.theme_phrase


def _maybe_keyword_alt(sheet: StrategySheet, query: str) -> None:
    if not sheet.keywords or sheet.alt_keywords:
        return
    try:
        data = _chat_json(
            _KEYWORD_ALT_PROMPT.format(
                query=_goal_only(query)[:800],
                keywords=" ".join(sheet.keywords),
            )
        )
    except Exception:
        return
    alt = data.get("alt")
    tokens = _token_list(alt)
    if tokens and tokens != sheet.keywords:
        if sheet.theme_phrase:
            joined = " ".join(tokens)
            if not _reformulation_keeps_theme(joined, sheet.theme_phrase):
                return
            if _introduces_invented_type(joined, sheet.theme_phrase):
                return
        sheet.alt_keywords = tokens[:8]


def _llm_filter_candidates(sheet: StrategySheet) -> str | None:
    if sheet.strategy == "closed_rank" or not sheet.candidate_facts:
        return None
    _rescore_all(sheet)
    # Extreme-style coverage prune before LLM: only judge fresh, decent-coverage keys.
    ranked = _top_scored_keys(sheet, limit=16, min_coverage=_COVERAGE_PRUNE_FLOOR)
    fresh = [k for k in ranked if k not in sheet.scored_keys]
    if len(fresh) < 2:
        fresh = [k for k in _top_scored_keys(sheet, limit=12) if k not in sheet.scored_keys]
    if len(fresh) < 2:
        return None
    pool = fresh[:8]
    try:
        data = _chat_json(
            _FILTER_PROMPT.format(
                constraints=_constraint_lines(sheet),
                products=_render_candidates_for_llm(sheet, keys=pool, limit=8),
            )
        )
    except Exception:
        sheet.notes.append("llm_filter_failed")
        return None

    sheet.scored_keys.update(pool)
    short: list[str] = []
    for source in (data.get("verdicts"), data.get("shortlist")):
        if not isinstance(source, list):
            continue
        for row in source:
            if isinstance(row, dict):
                cid = str(row.get("id") or "").strip()
                if not _as_bool(row.get("pass"), default=True):
                    continue
            else:
                cid = str(row).strip()
            resolved = _resolve_fact_key(sheet, cid)
            if resolved and resolved not in short:
                short.append(resolved)
    # Local coverage fallback / merge: keep high-coverage hard_ok keys even if LLM sparse.
    for key in _top_scored_keys(sheet, limit=6, min_coverage=0.55):
        sc = sheet.candidate_scores.get(key) or {}
        if sc.get("hard_ok") and key not in short:
            short.append(key)
        if len(short) >= 8:
            break
    if not short:
        return None
    # Sort shortlist by local score_n (extreme fuse).
    short.sort(
        key=lambda k: -float((sheet.candidate_scores.get(k) or {}).get("score_n") or 0)
    )
    sheet.shortlist = short[:8]
    best = sheet.shortlist[0]
    if "::" in best and not sheet.llm_pick_id:
        sheet.llm_pick_id = best.split("::", 1)[0]
        sheet.llm_pick_sku = best.split("::", 1)[1]
    return (
        "SCOUT FILTER shortlist (coverage-pruned + LLM): "
        + ", ".join(sheet.shortlist)
        + ". View/inspect top ids, then compare if needed. "
        "When carting, use the SAME product_id::sku pair you just inspected."
    )


def _llm_select_best(sheet: StrategySheet) -> str | None:
    if sheet.llm_select_done or sheet.strategy == "closed_rank":
        return None
    if sheet.event_recovery_mode:
        return None
    _rescore_all(sheet)
    viewed = [
        k
        for k, facts in sheet.candidate_facts.items()
        if facts.get("price") is not None or facts.get("in_stock") is not None
    ]
    if len(viewed) < 2 and not sheet.shortlist:
        return None
    # Extreme: sort by coverage then score_n before LLM select.
    pool = sheet.shortlist or viewed
    pool = [k for k in pool if k in sheet.candidate_facts]
    pool = sorted(
        pool,
        key=lambda k: (
            0 if (sheet.candidate_scores.get(k) or {}).get("hard_ok", True) else 1,
            -float((sheet.candidate_scores.get(k) or {}).get("coverage") or 0),
            -float((sheet.candidate_scores.get(k) or {}).get("score_n") or 0),
        ),
    )[:8]
    if len(pool) < 1:
        return None
    # Fast-accept: extreme should_fast_accept spirit.
    top = pool[0]
    top_sc = sheet.candidate_scores.get(top) or {}
    if (
        float(top_sc.get("coverage") or 0) >= 0.85
        and top_sc.get("hard_ok", True)
        and (len(pool) == 1 or float(top_sc.get("coverage") or 0) >= float((sheet.candidate_scores.get(pool[1]) or {}).get("coverage") or 0) + 0.12)
    ):
        sheet.llm_select_done = True
        if "::" in top:
            _set_llm_pick_pair(sheet, top.split("::", 1)[0], top.split("::", 1)[1])
        return (
            f"SCOUT FAST-ACCEPT pick: {sheet.llm_pick_id or top}::"
            f"{sheet.llm_pick_sku or '?'} (coverage={float(top_sc.get('coverage') or 0):.2f}). "
            "inspect_stock then cart/order that exact pair."
        )
    if len(pool) < 2:
        sheet.llm_select_done = True
        if "::" in top:
            _set_llm_pick_pair(sheet, top.split("::", 1)[0], top.split("::", 1)[1])
        return (
            f"SCOUT pick: {sheet.llm_pick_id or top}::{sheet.llm_pick_sku or '?'}. "
            "inspect_stock then cart/order that exact pair."
        )
    try:
        data = _chat_json(
            _SELECT_PROMPT.format(
                constraints=_constraint_lines(sheet),
                priorities=", ".join(sheet.soft_priorities or sheet.rank_priority)
                or (sheet.use_case or "hard constraints first"),
                products=_render_candidates_for_llm(sheet, keys=pool, limit=8),
            )
        )
    except Exception:
        # Fall back to local top.
        sheet.llm_select_done = True
        if "::" in top:
            _set_llm_pick_pair(sheet, top.split("::", 1)[0], top.split("::", 1)[1])
        return (
            f"SCOUT LOCAL pick (LLM select failed): {sheet.llm_pick_id or top}::"
            f"{sheet.llm_pick_sku or '?'}. inspect_stock then cart/order that exact pair."
        )
    sheet.llm_select_done = True
    best = data.get("best_id")
    sku = data.get("best_sku")
    resolved = None
    if isinstance(best, str) and best.strip():
        resolved = _resolve_fact_key(sheet, best.strip())
    if not resolved:
        resolved = top
    if isinstance(sku, str) and sku.strip() and "::" not in resolved:
        _set_llm_pick_pair(sheet, resolved.split("::", 1)[0], sku.strip())
    elif "::" in resolved:
        _set_llm_pick_pair(sheet, resolved.split("::", 1)[0], resolved.split("::", 1)[1])
    # else leave pick unset until inspect provides a full pair
    return (
        f"SCOUT COMPARE pick: product_id={sheet.llm_pick_id or resolved} "
        f"sku={sheet.llm_pick_sku or 'inspect to obtain'}. "
        "inspect_stock this SKU if hard-compliant, then cart/order that exact pair."
    )


# Claim fields that commonly fail faithfulness when invented — never auto-add.
_UNSAFE_CLAIM_FIELDS = frozenset(
    {
        "category",
        "device_type",
        "device_operating_system",
        "use_case",
        "title",
        "evidence_text",
    }
)
# Commerce fields always safe to claim when observed.
_CORE_CLAIM_FIELDS = ("in_stock", "price")


def _facts_for_product(
    sheet: StrategySheet, product_id: str | None, sku: str | None
) -> dict[str, Any]:
    if not product_id:
        return {}
    keys: list[str] = []
    if sku:
        keys.append(f"{product_id}::{sku}")
    keys.append(product_id)
    merged: dict[str, Any] = {}
    for key in keys:
        prev = sheet.candidate_facts.get(key)
        if prev:
            merged.update(prev)
    return merged


def _norm_fact_key(key: str) -> str:
    """Normalize attribute names for open-vocab matching (no fixed attr list)."""
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


# Evaluator / goal claim fields often differ from catalog fact keys (TF7).
# These are vocabulary bridges, not product-family attribute maps.
_FACT_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "resolution": (
        "display",
        "display_resolution",
        "screen_resolution",
        "video_resolution",
        "displayresolution",
    ),
    "screen_size": ("size", "screen", "screensize", "display_size", "panel_size"),
    "refresh_rate": ("refresh", "refreshrate", "hz", "max_refresh_rate"),
    "brand": ("brand_name", "manufacturer"),
}


def _resolution_tier(text: Any) -> str | None:
    """Map free-text / pixel dims to HD class used by ORO Bench task rules."""
    if text is None:
        return None
    raw = str(text).strip().lower()
    if not raw:
        return None
    nums = [int(x) for x in re.findall(r"\d{3,4}", raw)]
    if len(nums) >= 2:
        long_s, short_s = max(nums[0], nums[1]), min(nums[0], nums[1])
        if long_s >= 3840 and short_s >= 2160:
            return "4k"
        if long_s >= 2560 and short_s >= 1440:
            return "qhd"
        if long_s >= 1920 and short_s >= 1080:
            return "fhd"
        if long_s >= 1280 and short_s >= 720:
            return "hd"
    if re.search(r"\b(4k|uhd|2160)\b", raw):
        return "4k"
    if re.search(r"\b(qhd|1440p|1440)\b", raw):
        return "qhd"
    if re.search(r"\b(full\s*hd|fhd|1080p|1080)\b", raw):
        return "fhd"
    if re.search(r"\b(720p|720)\b", raw) or re.search(r"\bhd\b", raw):
        return "hd"
    return None


def _lookup_fact(facts: dict[str, Any], field: str) -> Any:
    """Find a fact by exact, case-insensitive, normalized key, or known alias."""
    if not field:
        return None
    if field in facts and facts[field] is not None:
        return facts[field]
    lower = field.lower()
    for key, val in facts.items():
        if str(key).lower() == lower and val is not None:
            return val
    want = _norm_fact_key(field)
    if want:
        for key, val in facts.items():
            if val is not None and _norm_fact_key(str(key)) == want:
                return val
    for alt in _FACT_KEY_ALIASES.get(lower, ()):
        if alt in facts and facts[alt] is not None:
            return facts[alt]
        alt_n = _norm_fact_key(alt)
        for key, val in facts.items():
            if val is not None and _norm_fact_key(str(key)) == alt_n:
                return val
    return None


def _find_fact_supporting_need(
    facts: dict[str, Any], field: str, need: Any
) -> Any | None:
    """Return an observed value that supports `need` for claim field `field`."""
    got = _lookup_fact(facts, field)
    if got is not None and _values_compatible(got, need):
        return got
    if need is None or need == "":
        return got
    # Any structured fact (not title/evidence) whose value matches the need.
    for key, val in facts.items():
        if val is None or str(key) in {"title", "evidence_text", "category_path"}:
            continue
        if _values_compatible(val, need):
            return val
    evidence = facts.get("evidence_text") or facts.get("title")
    if evidence is not None and _values_compatible(evidence, need):
        return evidence
    return got


def _claim_value_from_fact(field: str, raw: Any) -> str | None:
    """Serialize an observed fact as a claim string — no product-type rewriting."""
    if raw is None:
        return None
    if field == "in_stock":
        return "true" if raw is not False else "false"
    if field == "price":
        price = _as_float(raw)
        if price is None:
            return None
        return str(int(price)) if float(price).is_integer() else str(price)
    if isinstance(raw, bool):
        return "true" if raw else "false"
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and raw.is_integer():
            return str(int(raw))
        return str(raw)
    text = str(raw).strip()
    if not text or text.lower() in {"null", "none", "n/a", "unknown"}:
        return None
    return text


def _values_compatible(observed: Any, need: Any) -> bool:
    """Loose match for arbitrary gated specs (size, RAM, color, material, …)."""
    if need is None or need == "":
        return True
    if observed is None:
        return False
    o = str(observed).strip().lower()
    n = str(need).strip().lower()
    if not n:
        return True
    if o == n:
        return True
    if n in o or o in n:
        return True
    # Resolution class: "1920 x 1080" ↔ fhd / 1080p / Full HD (TF7).
    ot, nt = _resolution_tier(o), _resolution_tier(n)
    if ot and nt and ot == nt:
        return True
    # Synonym-aware hit (extreme is_kw_in spirit) for waterproof/grey/etc.
    if _is_kw_in(n, o) or _is_kw_in(o, n):
        return True
    # Compare leading numbers when both sides are quantitative (6.64in vs 6.64,
    # 16GB vs 16, 120Hz vs 120, 27 vs 27-inch, etc.).
    on = _as_float(o)
    nn = _as_float(n)
    if on is not None and nn is not None and abs(on - nn) < 1e-9:
        return True
    # Token overlap for multi-word specs ("waterproof ip67" vs "ip67").
    o_toks = set(re.findall(r"[a-z0-9]+", o))
    n_toks = set(re.findall(r"[a-z0-9]+", n))
    if n_toks and n_toks.issubset(o_toks):
        return True
    # Expand need tokens via synonyms then require all to hit observed blob.
    need_atoms = [t for t in re.findall(r"[a-z0-9]+", n) if t not in _SCOUT_STOPWORDS]
    if need_atoms and all(
        any(_is_kw_in(cand, o) for cand in _syn_expand(t)) for t in need_atoms
    ):
        return True
    return False


def _ingest_product_facts(sheet: StrategySheet, observation: Any) -> None:
    """Harvest whatever structured facts the env returns — any product type."""
    blob = observation
    if isinstance(observation, dict):
        for key in ("observation", "data", "result"):
            if isinstance(observation.get(key), (dict, list)):
                blob = observation[key]
                break

    products: list[dict[str, Any]] = []
    if isinstance(blob, dict):
        if "compared" in blob and isinstance(blob["compared"], list):
            products.extend([p for p in blob["compared"] if isinstance(p, dict)])
        elif "results" in blob and isinstance(blob["results"], list):
            products.extend([p for p in blob["results"] if isinstance(p, dict)])
        elif "product_id" in blob:
            products.append(blob)
        elif "price" in blob or "in_stock" in blob:
            products.append(blob)
    elif isinstance(blob, list):
        products.extend([p for p in blob if isinstance(p, dict)])

    for rank, prod in enumerate(products):
        pid = prod.get("product_id")
        sku = prod.get("sku") or prod.get("sku_id")
        facts_src = prod.get("facts") if isinstance(prod.get("facts"), dict) else {}
        attrs_src = (
            prod.get("attributes")
            if isinstance(prod.get("attributes"), dict)
            else prod.get("attrs")
            if isinstance(prod.get("attrs"), dict)
            else prod.get("specs")
            if isinstance(prod.get("specs"), dict)
            else {}
        )
        merged: dict[str, Any] = {}
        for key in ("price", "in_stock", "brand", "currency", "title", "color", "model"):
            if key in prod and prod[key] is not None:
                merged[key] = prod[key]
        for src in (facts_src, attrs_src):
            for key, val in src.items():
                if val is None or not str(key).strip():
                    continue
                merged[str(key)] = val
        path = prod.get("category_path")
        if isinstance(path, list) and path:
            merged["category"] = (
                path[-1] if isinstance(path[-1], str) else str(path[-1])
            )
            merged["category_path"] = " > ".join(str(x) for x in path)
        elif isinstance(path, str) and path.strip():
            merged["category_path"] = path.strip()
        text_parts: list[str] = []
        if isinstance(merged.get("title"), str):
            text_parts.append(merged["title"])
        for fk, fv in list(merged.items()):
            if fk in {"title", "evidence_text", "category_path"}:
                continue
            text_parts.append(f"{fk} {fv}")
        if text_parts:
            merged["evidence_text"] = " ".join(text_parts)
        if isinstance(pid, str) and pid:
            key = f"{pid}::{sku}" if isinstance(sku, str) and sku else pid
            prev = sheet.candidate_facts.get(key) or {}
            old_text = prev.get("evidence_text")
            prev.update({k: v for k, v in merged.items() if v is not None})
            if isinstance(old_text, str) and old_text:
                new_text = prev.get("evidence_text")
                if not isinstance(new_text, str) or len(new_text) < len(old_text):
                    prev["evidence_text"] = old_text
            sheet.candidate_facts[key] = prev
            _note_search_hit(sheet, key, rank)
            # Also index bare product_id for resolve convenience.
            if "::" in key and pid not in sheet.candidate_facts:
                sheet.candidate_facts[pid] = {
                    k: v for k, v in prev.items() if k != "sku"
                }
                _note_search_hit(sheet, pid, rank)
    if products:
        _rescore_all(sheet)


def _required_claim_fields(sheet: StrategySheet) -> list[str]:
    """Any sheet-declared claim field is allowed if we can observe it later."""
    fields: list[str] = []
    for c in sheet.required_claims:
        name = str(c).strip()
        if not name or name in _UNSAFE_CLAIM_FIELDS:
            continue
        if name not in fields:
            fields.append(name)
    for must in _CORE_CLAIM_FIELDS:
        if must not in fields:
            fields.insert(0, must)
    if (
        sheet.gated_key
        and sheet.gated_key not in fields
        and sheet.gated_key not in _UNSAFE_CLAIM_FIELDS
    ):
        fields.append(sheet.gated_key)
    return fields


def _needed_attr_value(sheet: StrategySheet, field: str) -> str | None:
    if sheet.attributes and field in sheet.attributes:
        val = sheet.attributes.get(field)
        if val is not None and str(val).strip():
            return str(val).strip()
    if sheet.gated_key and field == sheet.gated_key and sheet.gated_value:
        return str(sheet.gated_value).strip()
    return None


def _build_fact_claims(
    sheet: StrategySheet, product_id: str | None, sku: str | None
) -> list[dict[str, str]]:
    """Build claims ONLY from observed facts for arbitrary attribute names.

    Catalog keys may differ from claim field names (display↔resolution,
    size↔screen_size). Emit the sheet's claim field with a grounded value.
    """
    facts = _facts_for_product(sheet, product_id, sku)
    claims: list[dict[str, str]] = []
    for field_name in _required_claim_fields(sheet):
        need = _needed_attr_value(sheet, field_name)
        raw = _find_fact_supporting_need(facts, field_name, need)
        if raw is None:
            raw = _lookup_fact(facts, field_name)
        claim_src: Any = raw
        if need and raw is not None and _values_compatible(raw, need):
            claim_src = need
        value = _claim_value_from_fact(field_name, claim_src)
        if value is None:
            continue
        claims.append({"field": field_name, "value": value})
    return claims


def _prepare_order_args(sheet: StrategySheet, args: dict[str, Any]) -> dict[str, Any]:
    """Hard rule from eval failures: non-fact_justify orders must omit claims."""
    out = dict(args)
    pid = out.get("product_id")
    sku = out.get("sku")
    if not isinstance(pid, str):
        pid = sheet.last_product_id
    if not isinstance(sku, str):
        sku = sheet.last_sku
    if isinstance(pid, str):
        out["product_id"] = pid
    if isinstance(sku, str):
        out["sku"] = sku

    if sheet.strategy != "fact_justify":
        out.pop("justification", None)
        return out

    claims = _build_fact_claims(
        sheet,
        pid if isinstance(pid, str) else None,
        sku if isinstance(sku, str) else None,
    )
    out["justification"] = {"claims": claims}
    return out


def _is_fact_backed(
    sheet: StrategySheet, product_id: str | None, sku: str | None
) -> tuple[bool, str]:
    """Require observed structured facts for firm brand/budget/stock + gated attrs."""
    if not product_id:
        return False, "missing product_id"
    facts = _facts_for_product(sheet, product_id, sku)
    if not facts:
        return False, "no viewed/inspected facts for this product yet"

    brand_need = sheet.firm_brand or (
        sheet.attributes.get("brand") if sheet.attributes else None
    )
    if brand_need:
        brand = _lookup_fact(facts, "brand")
        if not _values_compatible(brand, brand_need):
            return False, f"brand fact mismatch (need {brand_need}, got {brand!r})"

    if sheet.in_stock_required and facts.get("in_stock") is False:
        return False, "not in stock"

    price = _as_float(facts.get("price"))
    if sheet.budget is not None and price is not None and price > sheet.budget:
        return False, f"price {price} exceeds budget {sheet.budget}"

    attrs = dict(sheet.attributes or {})
    if sheet.gated_key and sheet.gated_value and sheet.gated_key not in attrs:
        attrs[sheet.gated_key] = sheet.gated_value

    for attr, need in attrs.items():
        if not need or attr in {"brand", "category"}:
            continue
        got = _find_fact_supporting_need(facts, attr, need)
        if got is None:
            return False, f"missing structured fact for {attr}"
        if not _values_compatible(got, need):
            return False, f"{attr} fact {got!r} incompatible with required {need!r}"
    return True, "ok"


def _recompute_closed_rank_top(sheet: StrategySheet) -> None:
    if sheet.strategy != "closed_rank" or not sheet.candidates:
        return
    priority = list(sheet.rank_priority) if sheet.rank_priority else ["price"]
    if "price" not in priority:
        priority = priority + ["price"]
    _rescore_all(sheet)

    def sort_key(cand: dict[str, str]) -> tuple:
        key = f"{cand['product_id']}::{cand['sku']}"
        facts = sheet.candidate_facts.get(key) or sheet.candidate_facts.get(
            cand["product_id"]
        ) or {}
        sc = sheet.candidate_scores.get(key) or sheet.candidate_scores.get(
            cand["product_id"]
        ) or {}
        in_stock = facts.get("in_stock")
        price = _as_float(facts.get("price"))
        eligible = True
        if sheet.in_stock_required and in_stock is False:
            eligible = False
        if sheet.budget is not None and price is not None and price > sheet.budget:
            eligible = False
        parts: list[Any] = [0 if eligible else 1]
        for axis in priority:
            raw = _lookup_fact(facts, axis)
            num = _as_float(raw)
            axis_l = axis.lower()
            direction = (
                "min"
                if axis_l == "price" or "cost" in axis_l or "cheap" in axis_l
                else "max"
            )
            if num is None:
                parts.append((1, 0.0))
            elif direction == "min":
                parts.append((0, num))
            else:
                parts.append((0, -num))
        # Extreme-style coverage as soft tiebreak after lex axes.
        parts.append((0, -float(sc.get("coverage") or 0)))
        parts.append((0, -float(sc.get("score_n") or 0)))
        return tuple(parts)

    ranked = sorted(sheet.candidates, key=sort_key)
    top = ranked[0]
    key = f"{top['product_id']}::{top['sku']}"
    if sheet.candidate_facts.get(key) or sheet.candidate_facts.get(top["product_id"]):
        sheet.computed_top_id = top["product_id"]
        sheet.computed_top_sku = top["sku"]


def _needs_ask_spec(sheet: StrategySheet) -> bool:
    return (
        sheet.strategy == "ask_then_constrain"
        and sheet.underspecified
        and not sheet.gated_value
        and not sheet.asked_clarify
    )


def _violates_ask_spec_gate(
    sheet: StrategySheet, tool_calls: list[dict[str, Any]]
) -> bool:
    if not _needs_ask_spec(sheet):
        return False
    names = _call_names(tool_calls)
    if not names:
        return True
    blocked = {
        "search",
        "filter",
        "view",
        "compare",
        "inspect_stock",
        "add_to_cart",
        "place_test_order",
    }
    if any(name in blocked for name in names):
        return True
    return "message" not in names


def _violates_fact_justify_cart_gate(
    sheet: StrategySheet, tool_calls: list[dict[str, Any]]
) -> str | None:
    if sheet.strategy != "fact_justify":
        return None
    names = _call_names(tool_calls)
    if "add_to_cart" not in names and "place_test_order" not in names:
        return None
    for call in tool_calls:
        name = (call.get("function") or {}).get("name")
        if name not in {"add_to_cart", "place_test_order"}:
            continue
        try:
            args = _arguments(call)
        except Exception:
            return "could not parse cart/order args"
        args = _resolve_cart_order_pair(
            sheet, args, for_order=(name == "place_test_order")
        )
        pid = args.get("product_id")
        sku = args.get("sku")
        if not isinstance(pid, str):
            pid = sheet.last_product_id
        if not isinstance(sku, str):
            sku = sheet.last_sku
        ok, reason = _is_fact_backed(
            sheet,
            pid if isinstance(pid, str) else None,
            sku if isinstance(sku, str) else None,
        )
        if not ok:
            return reason
        if name == "place_test_order":
            claims = _build_fact_claims(
                sheet,
                pid if isinstance(pid, str) else None,
                sku if isinstance(sku, str) else None,
            )
            have = {c["field"] for c in claims}
            need = set(_required_claim_fields(sheet))
            missing = need - have
            if missing:
                return f"missing fact-backed claims: {sorted(missing)}"
    return None


def _violates_theme_search_gate(
    sheet: StrategySheet, tool_calls: list[dict[str, Any]]
) -> str | None:
    if sheet.strategy != "theme_recall" or not sheet.theme_phrase:
        return None
    theme = sheet.theme_phrase
    for call in tool_calls:
        if (call.get("function") or {}).get("name") != "search":
            continue
        try:
            args = _arguments(call)
        except Exception:
            continue
        q = args.get("query")
        if not isinstance(q, str) or not q.strip():
            return "search query missing"
        if not _reformulation_keeps_theme(q, theme):
            return f"search must keep theme tokens from `{theme}`"
        if _introduces_invented_type(q, theme):
            return (
                f"search invents a product type not in theme `{theme}` — "
                "use theme-preserving reformulations only"
            )
    return None


def _strip_id(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _fact_key(product_id: str, sku: str | None) -> str:
    if sku:
        return f"{product_id}::{sku}"
    return product_id


def _sku_owner(sheet: StrategySheet, sku: str) -> str | None:
    """Return product_id that owns this sku in candidate_facts, if unique."""
    owners = [
        k.split("::", 1)[0]
        for k in sheet.candidate_facts
        if "::" in k and k.split("::", 1)[1] == sku
    ]
    owners = list(dict.fromkeys(owners))
    if len(owners) == 1:
        return owners[0]
    return None


def _pair_known(sheet: StrategySheet, product_id: str, sku: str | None) -> bool:
    if not product_id:
        return False
    if sku and _fact_key(product_id, sku) in sheet.candidate_facts:
        return True
    if product_id in sheet.candidate_facts and not sku:
        return True
    if sku and _sku_owner(sheet, sku) == product_id:
        return True
    return False


def _set_llm_pick_pair(
    sheet: StrategySheet, product_id: str | None, sku: str | None
) -> None:
    if not product_id:
        return
    sheet.llm_pick_id = product_id
    sheet.llm_pick_sku = sku


def _resolve_cart_order_pair(
    sheet: StrategySheet, args: dict[str, Any], *, for_order: bool = False
) -> dict[str, Any]:
    """Keep product_id and sku as one atomic pair — never mix pick id with foreign sku.

    Eval failure pattern: inspect A::skuX then add_to_cart(llm_pick_B, skuX) → no_candidate.
    """
    out = dict(args)
    pid = _strip_id(out.get("product_id"))
    sku = _strip_id(out.get("sku"))

    def apply(p: str, s: str | None) -> dict[str, Any]:
        out["product_id"] = p
        if s:
            out["sku"] = s
        elif "sku" in out and not s:
            # Do not leave a stale foreign sku attached to a different product.
            out.pop("sku", None)
        return out

    # Closed-rank: always the computed lex top as a full pair.
    if sheet.strategy == "closed_rank" and sheet.computed_top_id:
        return apply(sheet.computed_top_id, sheet.computed_top_sku)

    # Ordering with an item already in cart: finish that pair.
    if for_order and sheet.cart_has_item and sheet.cart_product_id:
        return apply(sheet.cart_product_id, sheet.cart_sku)

    # If model supplied both and they form a known / consistent pair, keep them.
    if pid and sku:
        owner = _sku_owner(sheet, sku)
        if _pair_known(sheet, pid, sku):
            _set_llm_pick_pair(sheet, pid, sku)
            return apply(pid, sku)
        # Cross-wire: sku belongs to a different product than pid.
        if owner and owner != pid:
            # Prefer the just-inspected owner of this sku.
            if (
                sheet.last_inspected_sku == sku
                and sheet.last_inspected_id
            ):
                _set_llm_pick_pair(
                    sheet, sheet.last_inspected_id, sheet.last_inspected_sku
                )
                return apply(sheet.last_inspected_id, sheet.last_inspected_sku)
            _set_llm_pick_pair(sheet, owner, sku)
            return apply(owner, sku)
        # Inspected pair wins over stale llm_pick product_id.
        if (
            sheet.last_inspected_id
            and sheet.last_inspected_sku
            and (pid != sheet.last_inspected_id or sku != sheet.last_inspected_sku)
            and (
                sku == sheet.last_inspected_sku
                or pid == sheet.last_inspected_id
            )
        ):
            _set_llm_pick_pair(
                sheet, sheet.last_inspected_id, sheet.last_inspected_sku
            )
            return apply(sheet.last_inspected_id, sheet.last_inspected_sku)
        # Unknown but both provided: keep model's pair (do NOT overwrite pid alone).
        return apply(pid, sku)

    # After a successful inspect, that pair is authoritative for the next cart.
    if (
        not sheet.event_recovery_mode
        and sheet.last_inspected_id
        and sheet.last_inspected_sku
    ):
        if not pid or pid == sheet.last_inspected_id or not sku:
            _set_llm_pick_pair(
                sheet, sheet.last_inspected_id, sheet.last_inspected_sku
            )
            return apply(sheet.last_inspected_id, sheet.last_inspected_sku)

    # Full llm pick pair only (never product_id without its sku).
    if (
        not sheet.event_recovery_mode
        and sheet.llm_pick_id
        and sheet.llm_pick_sku
    ):
        return apply(sheet.llm_pick_id, sheet.llm_pick_sku)

    if pid and not sku:
        # Fill sku only from the same product's known keys / last inspect.
        if (
            sheet.last_inspected_id == pid
            and sheet.last_inspected_sku
        ):
            return apply(pid, sheet.last_inspected_sku)
        if sheet.llm_pick_id == pid and sheet.llm_pick_sku:
            return apply(pid, sheet.llm_pick_sku)
        matches = [
            k for k in sheet.candidate_facts if k.startswith(f"{pid}::")
        ]
        if len(matches) == 1:
            return apply(pid, matches[0].split("::", 1)[1])
        return apply(pid, None)

    # Scout fallback: full key with sku when available.
    if not sheet.event_recovery_mode:
        top = _top_scored_keys(sheet, limit=1, min_coverage=0.55)
        if top:
            sc = sheet.candidate_scores.get(top[0]) or {}
            if sc.get("hard_ok", True) and "::" in top[0]:
                p, s = top[0].split("::", 1)
                _set_llm_pick_pair(sheet, p, s)
                return apply(p, s)

    return out


def _prefer_pick(
    sheet: StrategySheet, args: dict[str, Any]
) -> dict[str, Any]:
    return _resolve_cart_order_pair(sheet, args, for_order=False)


def _prepare_action_args(
    sheet: StrategySheet, name: str, args: dict[str, Any]
) -> dict[str, Any]:
    if name == "place_test_order":
        out = _resolve_cart_order_pair(sheet, args, for_order=True)
        out = _prepare_order_args(sheet, out)
        return out
    if name == "add_to_cart":
        return _resolve_cart_order_pair(sheet, args, for_order=False)
    if name == "inspect_stock":
        # Do not rewrite inspect targets via llm_pick — model must inspect the
        # listing it intends to cart; we only record the pair after the call.
        return dict(args)
    if name == "filter":
        out = dict(args)
        for key, val in sheet.filter_args.items():
            if key not in out or out.get(key) in (None, ""):
                out[key] = val
        if "category" not in out and sheet.category:
            out["category"] = sheet.category
        if "max_price" not in out and sheet.budget is not None:
            out["max_price"] = sheet.budget
        if "in_stock" not in out and sheet.in_stock_required:
            out["in_stock"] = True
        return out
    if name == "search":
        out = dict(args)
        q = out.get("query")
        plan = _search_plan(sheet)
        if not isinstance(q, str) or not q.strip():
            out["query"] = (plan[0] if plan else None) or sheet.primary_query or (
                " ".join(sheet.keywords) if sheet.keywords else None
            )
        elif sheet.strategy == "theme_recall" and sheet.theme_phrase:
            if not _reformulation_keeps_theme(str(q), sheet.theme_phrase):
                out["query"] = plan[0] if plan else sheet.theme_phrase
        return out
    return args


def _chat_json(prompt: str, *, temperature: float = 0) -> dict[str, Any]:
    last_error: Exception | None = None
    temp = temperature
    for _ in range(_EXTRACT_TRIES):
        inference = _proxy.post(
            "/inference/chat/completions",
            json_data={
                "model": _model(),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
            },
        )
        if inference is None:
            last_error = RuntimeError("inference failed")
            temp = min(temp + 0.2, 0.6)
            continue
        try:
            content = inference["choices"][0]["message"].get("content") or ""
            return _extract_json_object(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            temp = min(temp + 0.2, 0.6)
    raise RuntimeError(f"json extract failed: {last_error}")


def _apply_gated(sheet: StrategySheet, gated: Any) -> None:
    if not isinstance(gated, dict):
        return
    key = gated.get("key")
    value = gated.get("value")
    if isinstance(key, str) and key.strip():
        sheet.gated_key = key.strip()
    if value is not None and str(value).strip():
        sheet.gated_value = str(value).strip()


def _parse_candidates(query: str, data: dict[str, Any]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    cands = data.get("candidates")
    if isinstance(cands, list):
        for item in cands:
            if not isinstance(item, dict):
                continue
            pid, sku = item.get("product_id"), item.get("sku")
            if isinstance(pid, str) and isinstance(sku, str) and pid and sku:
                key = f"{pid}::{sku}"
                if key not in seen:
                    seen.add(key)
                    found.append({"product_id": pid.strip(), "sku": sku.strip()})
    for match in re.finditer(r"(\d{6,})::([A-Za-z0-9_-]+)", query):
        pid, sku = match.group(1), match.group(2)
        key = f"{pid}::{sku}"
        if key not in seen:
            seen.add(key)
            found.append({"product_id": pid, "sku": sku})
    return found


def _init_sheet(query: str) -> StrategySheet:
    sheet = StrategySheet()
    try:
        data = _chat_json(_STRATEGY_EXTRACT_PROMPT.format(query=query))
    except Exception:
        sheet.notes.append("strategy_extract_failed")
        _apply_decompose(sheet, query)
        _maybe_keyword_alt(sheet, query)
        return sheet

    strategy = str(data.get("strategy") or "general").strip().lower()
    allowed = {
        "firm_vs_soft",
        "theme_recall",
        "ask_then_constrain",
        "preference_fit",
        "closed_rank",
        "exact_then_relax",
        "fact_justify",
        "general",
    }
    sheet.strategy = strategy if strategy in allowed else "general"

    category = data.get("category")
    if isinstance(category, str) and category.strip():
        sheet.category = category.strip()
    budget = _as_float(data.get("budget_max"))
    if budget is not None:
        sheet.budget = budget
    in_stock = _as_bool(data.get("in_stock"), default=True)
    if in_stock is not None:
        sheet.in_stock_required = in_stock

    for attr, dest in (
        ("firm_brand", "firm_brand"),
        ("preferred_brand", "preferred_brand"),
        ("theme_phrase", "theme_phrase"),
        ("primary_query", "primary_query"),
        ("use_case", "use_case"),
        ("preferred_model", "preferred_model"),
        ("preferred_color", "preferred_color"),
        ("relaxation_hint", "relaxation_hint"),
    ):
        val = data.get(attr)
        if isinstance(val, str) and val.strip():
            setattr(sheet, dest, val.strip())

    if not sheet.primary_query and sheet.theme_phrase:
        sheet.primary_query = sheet.theme_phrase

    reforms = data.get("reformulations")
    if isinstance(reforms, list):
        cleaned: list[str] = []
        for item in reforms:
            if isinstance(item, str) and item.strip():
                text = item.strip()
                if text != sheet.primary_query and text not in cleaned:
                    cleaned.append(text)
        sheet.reformulations = cleaned[:4]

    _apply_gated(sheet, data.get("gated_attribute"))
    sheet.underspecified = bool(_as_bool(data.get("underspecified"), default=False))
    if sheet.gated_value:
        sheet.underspecified = False
    sheet.needs_priority_clarification = bool(
        _as_bool(data.get("needs_priority_clarification"), default=False)
    )

    priorities = data.get("soft_priorities")
    if isinstance(priorities, list):
        sheet.soft_priorities = [
            str(p).strip() for p in priorities if str(p).strip()
        ][:8]

    sheet.candidates = _parse_candidates(query, data)
    priority = data.get("rank_priority")
    if isinstance(priority, list):
        sheet.rank_priority = [str(x).strip() for x in priority if str(x).strip()]

    attrs = data.get("attributes")
    if isinstance(attrs, dict):
        sheet.attributes = {
            str(k).strip(): str(v).strip()
            for k, v in attrs.items()
            if str(k).strip() and v is not None and str(v).strip()
        }
        # Prefer LLM gated_attribute; else take the first firm attribute as gate.
        if not sheet.gated_key and sheet.attributes:
            for key, val in sheet.attributes.items():
                if key.lower() in {"brand", "category"}:
                    continue
                sheet.gated_key = key
                sheet.gated_value = val
                break

    claims = data.get("required_claims")
    if isinstance(claims, list):
        sheet.required_claims = [
            str(c).strip()
            for c in claims
            if isinstance(c, str) and str(c).strip() and str(c).strip() != "category"
        ]

    if sheet.strategy == "theme_recall" or sheet.theme_phrase:
        _sanitize_theme_queries(sheet, query)
        if sheet.theme_phrase and sheet.strategy == "general":
            sheet.strategy = "theme_recall"

    # Heuristic: quoted phrase match goals are theme_recall even if extract missed.
    if not sheet.theme_phrase:
        guessed = _theme_from_query(query)
        if guessed:
            sheet.theme_phrase = guessed
            sheet.strategy = "theme_recall"
            _sanitize_theme_queries(sheet, query)

    # Heuristic: "tell you if you ask" → ask_then_constrain.
    lower_q = query.lower()
    if "if you ask" in lower_q or "tell you if" in lower_q:
        sheet.strategy = "ask_then_constrain"
        sheet.underspecified = True

    # Heuristic: justification / walk-me-through goals → fact_justify.
    if any(
        p in lower_q
        for p in (
            "walk me through why",
            "tell me why",
            "why it's a good pick",
            "why the one you pick",
            "trustworthy reason",
        )
    ):
        sheet.strategy = "fact_justify"

    # Claims are only safe for fact_justify; other strategies fail hard checks
    # when the model invents justification fields (ShopScout eval pattern).
    if sheet.strategy != "fact_justify":
        sheet.required_claims = []
    else:
        sheet.required_claims = _required_claim_fields(sheet)

    notes = data.get("notes")
    if isinstance(notes, str) and notes.strip():
        sheet.notes.append(notes.strip())

    _apply_decompose(sheet, query)
    _maybe_keyword_alt(sheet, query)
    _constraint_axes(sheet)
    return sheet


def _apply_shopper_update(sheet: StrategySheet, text: str) -> None:
    sheet.last_shopper_text = text
    try:
        data = _chat_json(
            _UPDATE_EXTRACT_PROMPT.format(
                sheet=sheet.status_block()[:1200], message=text
            )
        )
    except Exception:
        sheet.pending_question = "?" in text
        return

    budget = _as_float(data.get("budget_max"))
    if budget is not None:
        sheet.budget = budget
    if "gated_attribute" in data:
        _apply_gated(sheet, data.get("gated_attribute"))
        if sheet.gated_value:
            sheet.underspecified = False
    for key in ("preferred_brand", "preferred_model", "preferred_color"):
        if key in data:
            val = data.get(key)
            if val == "" or val is None:
                setattr(sheet, key, None)
            elif isinstance(val, str) and val.strip():
                setattr(sheet, key, val.strip())
    priorities = data.get("soft_priorities")
    if isinstance(priorities, list):
        sheet.soft_priorities = [
            str(p).strip() for p in priorities if str(p).strip()
        ][:8]
    is_q = _as_bool(data.get("is_question"), default=None)
    sheet.pending_question = ("?" in text) if is_q is None else bool(is_q)
    notes = data.get("notes")
    if isinstance(notes, str) and notes.strip():
        sheet.notes.append(notes.strip())
    sheet.llm_select_done = False


def _call_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [
        str(name)
        for call in tool_calls
        for name in [(call.get("function") or {}).get("name")]
        if name
    ]


def _message_has_question(tool_calls: list[dict[str, Any]]) -> bool:
    for call in tool_calls:
        if (call.get("function") or {}).get("name") != "message":
            continue
        try:
            args = _arguments(call)
        except Exception:
            continue
        content = args.get("content") or args.get("text") or ""
        if isinstance(content, str) and "?" in content:
            return True
    return False


def _violates_pending_gate(tool_calls: list[dict[str, Any]]) -> bool:
    names = _call_names(tool_calls)
    if not names:
        return True
    if any(name in _BLOCKED_WHILE_PENDING for name in names):
        return True
    return "message" not in names


def _track_product_args(sheet: StrategySheet, name: str, args: dict[str, Any]) -> None:
    if name not in {
        "view",
        "inspect_stock",
        "add_to_cart",
        "remove_from_cart",
        "place_test_order",
    }:
        return
    pid = _strip_id(args.get("product_id"))
    sku = _strip_id(args.get("sku"))
    if pid:
        sheet.last_product_id = pid
    if sku:
        sheet.last_sku = sku
    # inspect_stock establishes the only legal next cart pair.
    if name == "inspect_stock" and pid and sku:
        sheet.last_inspected_id = pid
        sheet.last_inspected_sku = sku
        _set_llm_pick_pair(sheet, pid, sku)


def _violates_cart_thrash_gate(
    sheet: StrategySheet, tool_calls: list[dict[str, Any]]
) -> str | None:
    """Block remove/search thrash when a hard-valid item is already in cart.

    Applies even during event_recovery_mode once a *stable* (non-spike) cart
    exists — otherwise TF7 burns the episode on remove/re-add loops.
    """
    if not sheet.cart_has_item:
        return None
    names = _call_names(tool_calls)
    if "remove_from_cart" in names:
        return (
            f"cart already holds {sheet.cart_product_id}::{sheet.cart_sku} — "
            "call place_test_order on that pair now; do not remove unless a "
            "market event broke it"
        )
    if "place_test_order" in names:
        return None
    # Allow message / inspect_stock; block wandering tools that burn steps.
    blocked = {"search", "filter", "view", "compare", "add_to_cart"}
    if any(n in blocked for n in names) and "place_test_order" not in names:
        return (
            "item already in cart — next action must be place_test_order "
            f"({sheet.cart_product_id}::{sheet.cart_sku})"
            + (
                " with justification.claims from owned facts"
                if sheet.strategy == "fact_justify"
                else ""
            )
        )
    return None


def _handle_cart_observation(
    sheet: StrategySheet, name: str, args: dict[str, Any], observation: Any
) -> str | None:
    """Update cart state; return a corrective nudge on no_candidate / success."""
    blob = observation if isinstance(observation, dict) else {}
    obs = blob.get("observation") if isinstance(blob.get("observation"), dict) else blob
    err = blob.get("error") or (obs.get("error") if isinstance(obs, dict) else None)

    if name == "add_to_cart":
        if err or (isinstance(obs, dict) and obs.get("error")):
            sheet.cart_fail_streak += 1
            sheet.cart_has_item = False
            sheet.cart_product_id = None
            sheet.cart_sku = None
            sheet.llm_pick_id = None
            sheet.llm_pick_sku = None
            fix_pid = sheet.last_inspected_id
            fix_sku = sheet.last_inspected_sku
            if fix_pid and fix_sku:
                _set_llm_pick_pair(sheet, fix_pid, fix_sku)
                return (
                    f"CART FAILED ({err or obs.get('error')}). "
                    f"Retry add_to_cart with the inspected pair ONLY: "
                    f"product_id={fix_pid} sku={fix_sku}. "
                    "Never mix one product_id with another listing's sku."
                )
            return (
                f"CART FAILED ({err or 'error'}). inspect_stock a hard-compliant "
                "listing then add_to_cart with that exact product_id and sku."
            )
        added = obs.get("added") if isinstance(obs, dict) else None
        if isinstance(added, dict) and added.get("product_id"):
            sheet.cart_fail_streak = 0
            _set_llm_pick_pair(
                sheet,
                str(added.get("product_id")),
                str(added["sku"])
                if added.get("sku") is not None
                else _strip_id(args.get("sku")),
            )
            has_event = "market_event" in _observation_blob(observation).lower()
            if has_event:
                # Spike/stockout: cart is not orderable — recovery continues.
                sheet.cart_has_item = False
                sheet.cart_product_id = None
                sheet.cart_sku = None
                return None
            # Stable cart (incl. post-recovery backup): end recovery, order next.
            sheet.cart_has_item = True
            sheet.cart_product_id = str(added.get("product_id"))
            sheet.cart_sku = (
                str(added["sku"])
                if added.get("sku") is not None
                else _strip_id(args.get("sku"))
            )
            sheet.event_recovery_mode = False
            claim_bit = ""
            if sheet.strategy == "fact_justify":
                claim_bit = (
                    " Include justification.claims for REQUIRED_CLAIMS "
                    "(use owned facts; display/size count for resolution/"
                    "screen_size). "
                )
            return (
                f"CART OK: {sheet.cart_product_id}::{sheet.cart_sku}. "
                f"Next tool MUST be place_test_order on that exact pair."
                f"{claim_bit}Do not remove_from_cart or search."
            )
        return None

    if name == "remove_from_cart":
        sheet.cart_has_item = False
        sheet.cart_product_id = None
        sheet.cart_sku = None
        return None

    if name == "inspect_stock" and not err:
        pid = _strip_id(args.get("product_id"))
        sku = _strip_id(args.get("sku"))
        if pid and sku:
            sheet.last_inspected_id = pid
            sheet.last_inspected_sku = sku
            _set_llm_pick_pair(sheet, pid, sku)
        return None

    return None


def _market_event_nudge(observation: Any, sheet: StrategySheet) -> str | None:
    blob = _observation_blob(observation).lower()
    if (
        "market_event" not in blob
        and "price_change" not in blob
        and "stockout" not in blob
    ):
        return None
    sheet.saw_market_event = True
    sheet.event_recovery_mode = True
    sheet.llm_select_done = False
    sheet.llm_pick_id = None
    sheet.llm_pick_sku = None
    sheet.cart_has_item = False
    sheet.cart_product_id = None
    sheet.cart_sku = None
    sheet.last_inspected_id = None
    sheet.last_inspected_sku = None
    budget = f"{sheet.budget}" if sheet.budget is not None else "the budget"
    extra = ""
    if sheet.strategy == "preference_fit":
        extra = (
            " Recover to a different listing (not a twin variant of the same "
            "listing) that still fits the use-case priorities."
        )
    elif sheet.strategy == "exact_then_relax":
        extra = (
            f" Prefer minimal relaxation for model={sheet.preferred_model} "
            f"color={sheet.preferred_color} "
            f"({sheet.relaxation_hint or 'other color of same model first'})."
        )
    elif sheet.strategy == "closed_rank":
        extra = " Re-rank only the remaining listed candidates."
    elif sheet.strategy == "theme_recall":
        extra = f" Keep theme `{sheet.theme_phrase or ''}` on the backup."
    return (
        "MARKET EVENT active. Recover autonomously:\n"
        "1) `message` acknowledging the change — NO question marks, do not ask "
        "which option the shopper prefers.\n"
        "2) remove_from_cart on the broken target.\n"
        f"3) inspect_stock on another hard-compliant backup (budget<={budget})."
        f"{extra}\n"
        "4) add_to_cart + place_test_order that backup.\n"
        "Asking the shopper to choose will fail the episode gate."
    )


def _opening_nudge(sheet: StrategySheet) -> str:
    base = sheet.status_block() + "\n\n"
    tips = {
        "theme_recall": (
            "Search with the EXACT theme first (see SEARCH PLAN). Then KEYWORDS / "
            "ALT_KEYWORDS. After 1-2 searches, call `filter` with FILTER_ARGS. "
            "Reformulations must keep theme tokens and must NOT invent types."
        ),
        "firm_vs_soft": (
            "Satisfy firm constraints first. Use preferred brand only when fully "
            "firm-compliant. If you name a pick in message, cart that same id. "
            "Order with product_id+sku only — no justification.claims."
        ),
        "ask_then_constrain": (
            "FIRST action must be `message` asking for the hidden firm spec. "
            "Do not search/cart until it is known. Then verify and order with "
            "NO justification.claims."
        ),
        "preference_fit": (
            "Clarify soft priorities if needed, compare multiple DISTINCT listings, "
            "and cart a hard-valid option that fits the use case — not cheapest-only. "
            "Always add_to_cart with the exact product_id::sku you just inspected "
            "(never mix ids). After a successful cart, place_test_order immediately. "
            "Order with product_id+sku only — no justification.claims."
        ),
        "closed_rank": (
            "Only use listed candidates. View/compare all of them, apply "
            "lexicographic priority from the sheet, cart the current top. After "
            "stockout, re-rank remaining. Order with NO justification.claims."
        ),
        "exact_then_relax": (
            "View and cart the exact preferred model+color first. After stockout, "
            "relax minimally (usually other color of same model), inspect, then "
            "order with product_id+sku only — no justification.claims."
        ),
        "fact_justify": (
            "View/inspect until gated facts are EXPLICIT on the SKU you will cart. "
            "Never first-cart a title-only decoy. Cart the inspected product_id::sku "
            "pair atomically, then place_test_order with string justification.claims "
            "covering REQUIRED_CLAIMS from owned facts only. Do not thrash remove/re-add."
        ),
    }
    extra = (
        " Workflow: search(SEARCH PLAN) → filter(FILTER_ARGS) → view SCORE_TOP / "
        "shortlisted ids → compare if 2+ remain → inspect_stock → add_to_cart "
        "SAME pair → place_test_order. Never mix product_id from one listing with "
        "sku from another. Local coverage+RRF ranks candidates; follow SCORE_TOP / "
        "LLM_SHORTLIST / LLM_PICK."
    )
    return base + tips.get(
        sheet.strategy,
        "Satisfy hard constraints, verify with tools, then place_test_order.",
    ) + extra


def _steps_remaining_nudge(turn: int, max_steps: int, sheet: StrategySheet) -> str | None:
    remaining = max_steps - turn + 1
    if sheet.cart_has_item and sheet.cart_product_id:
        return (
            f"CART READY ({remaining} steps left): place_test_order NOW on "
            f"{sheet.cart_product_id}::{sheet.cart_sku}."
            + (
                " Include fact-backed justification.claims."
                if sheet.strategy == "fact_justify"
                else " No justification.claims."
            )
        )
    if remaining > 10:
        return None
    if sheet.event_recovery_mode:
        return (
            f"Only ~{remaining} steps left and EVENT RECOVERY is active. "
            "Finish remove → inspect backup → cart → place_test_order now."
        )
    if sheet.last_inspected_id and sheet.last_inspected_sku:
        return (
            f"Only ~{remaining} steps left. add_to_cart then place_test_order "
            f"using inspected pair {sheet.last_inspected_id}::"
            f"{sheet.last_inspected_sku}."
        )
    return (
        f"Only ~{remaining} steps left. Verify a hard-compliant pick and "
        "place_test_order soon."
    )


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    max_steps = int(policy["max_steps"])
    tools = policy["tools"]
    query = policy["query"]

    sheet = _init_sheet(query)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": query},
        {"role": "user", "content": _opening_nudge(sheet)},
    ]
    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "event": "strategy_extract",
            "strategy": sheet.strategy,
            "sheet": {
                "category": sheet.category,
                "budget": sheet.budget,
                "theme_phrase": sheet.theme_phrase,
                "firm_brand": sheet.firm_brand,
                "preferred_brand": sheet.preferred_brand,
                "preferred_model": sheet.preferred_model,
                "preferred_color": sheet.preferred_color,
                "candidates": list(sheet.candidates),
                "rank_priority": list(sheet.rank_priority),
                "required_claims": list(sheet.required_claims),
                "attributes": dict(sheet.attributes),
                "product_type": sheet.product_type,
                "compact_query": sheet.compact_query,
                "keywords": list(sheet.keywords),
                "filter_args": dict(sheet.filter_args),
                "underspecified": sheet.underspecified,
                "needs_priority_clarification": sheet.needs_priority_clarification,
            },
        }
    ]

    for turn in range(1, max_steps + 1):
        messages.append({"role": "user", "content": sheet.status_block()})
        step_nudge = _steps_remaining_nudge(turn, max_steps, sheet)
        if step_nudge:
            messages.append({"role": "user", "content": step_nudge})

        tool_calls: list[dict[str, Any]] = []
        assistant_content = ""
        for _attempt in range(4):
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": _model(),
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if inference is None:
                raise RuntimeError("inference request failed")
            assistant = inference["choices"][0]["message"]
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if not tool_calls:
                dialogue.append({"role": "assistant", "content": assistant_content})
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {
                            "role": "user",
                            "content": (
                                "Choose one of the supplied tools. "
                                + (
                                    "pending_question=true → use `message` now."
                                    if sheet.pending_question
                                    else "Continue toward a verified compliant order."
                                )
                            ),
                        },
                    ]
                )
                continue

            if sheet.pending_question and _violates_pending_gate(tool_calls):
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "pending_question_gate",
                    }
                )
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
                                "Rejected: pending_question is true. Call ONLY "
                                "`message` and answer the shopper now."
                            ),
                        },
                    ]
                )
                tool_calls = []
                continue

            if _violates_ask_spec_gate(sheet, tool_calls):
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "ask_spec_gate",
                    }
                )
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
                                "Rejected: firm spec is still unknown. Call "
                                "`message` now and ask what the firm requirement "
                                "is before searching or carting."
                            ),
                        },
                    ]
                )
                tool_calls = []
                continue

            theme_rej = _violates_theme_search_gate(sheet, tool_calls)
            if theme_rej:
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "theme_search_gate",
                        "reason": theme_rej,
                    }
                )
                plan = [sheet.primary_query or sheet.theme_phrase or ""]
                plan.extend(sheet.reformulations[:4])
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
                                f"Rejected theme search: {theme_rej}. Use one of: "
                                + " | ".join(q for q in plan if q)
                            ),
                        },
                    ]
                )
                tool_calls = []
                continue

            fact_rej = _violates_fact_justify_cart_gate(sheet, tool_calls)
            if fact_rej:
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "fact_justify_gate",
                        "reason": fact_rej,
                    }
                )
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
                                "Rejected fact_justify cart/order: "
                                f"{fact_rej}. View/inspect a SKU until gated "
                                "facts are explicit, then retry."
                            ),
                        },
                    ]
                )
                tool_calls = []
                continue

            thrash_rej = _violates_cart_thrash_gate(sheet, tool_calls)
            if thrash_rej:
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "cart_thrash_gate",
                        "reason": thrash_rej,
                    }
                )
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": f"Rejected: {thrash_rej}",
                        },
                    ]
                )
                tool_calls = []
                continue

            if sheet.event_recovery_mode and _message_has_question(tool_calls):
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                        "rejected": "event_recovery_no_questions",
                    }
                )
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
                                "Rejected: during EVENT RECOVERY your `message` "
                                "must not contain '?'. Acknowledge and proceed "
                                "autonomously — do not ask the shopper to choose."
                            ),
                        },
                    ]
                )
                tool_calls = []
                continue
            break
        else:
            raise RuntimeError("model returned no valid tool call before completion")

        answering_now = sheet.pending_question and "message" in _call_names(tool_calls)
        prepared_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            name = str((call.get("function") or {}).get("name") or "")
            try:
                args = _arguments(call)
            except Exception:
                args = {}
            args = _prepare_action_args(sheet, name, args)
            _track_product_args(sheet, name, args)
            if name == "search":
                sheet.search_count += 1
            if name == "add_to_cart":
                sheet.first_cart_done = True
            if (
                name == "message"
                and not sheet.event_recovery_mode
                and sheet.strategy == "firm_vs_soft"
                and sheet.last_product_id
                and (
                    answering_now
                    or (
                        sheet.committed_product_id
                        and sheet.last_product_id != sheet.committed_product_id
                    )
                )
            ):
                sheet.committed_product_id = sheet.last_product_id
                sheet.committed_sku = sheet.last_sku
            prepared_calls.append(
                {
                    "id": call.get("id"),
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }
            )
        tool_calls = prepared_calls

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
                        "name": call["function"]["name"],
                        "args": _arguments(call),
                    },
                }
                for index, call in enumerate(tool_calls, start=1)
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
                "strategy": sheet.strategy,
                "pending_question": sheet.pending_question,
                "event_recovery_mode": sheet.event_recovery_mode,
            }
        )

        if "message" in _call_names(tool_calls):
            if (
                (sheet.underspecified and not sheet.gated_value)
                or sheet.needs_priority_clarification
            ):
                sheet.asked_clarify = True
            if sheet.pending_question and sheet.strategy == "firm_vs_soft":
                if sheet.last_product_id and not sheet.committed_product_id:
                    sheet.committed_product_id = sheet.last_product_id
                    sheet.committed_sku = sheet.last_sku
            sheet.pending_question = False

        nudges: list[str] = []
        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            observation = call_result["observation"]
            _ingest_product_facts(sheet, observation)
            _recompute_closed_rank_top(sheet)
            used = str((tool_call.get("function") or {}).get("name") or "")
            try:
                used_args = _arguments(tool_call)
            except Exception:
                used_args = {}
            cart_nudge = _handle_cart_observation(
                sheet, used, used_args, observation
            )
            if cart_nudge and cart_nudge not in nudges:
                nudges.append(cart_nudge)
            if used == "search":
                sheet.search_pass += 1
            if (
                used in {"search", "filter", "compare", "view", "inspect_stock"}
                and not sheet.pending_question
                and not _needs_ask_spec(sheet)
                and not sheet.cart_has_item
            ):
                # After search, local scores already updated; LLM filter once
                # enough structured/high-coverage candidates exist.
                if used in {"view", "compare", "inspect_stock", "filter"} or (
                    used == "search" and len(sheet.candidate_facts) >= 3
                ):
                    filter_nudge = _llm_filter_candidates(sheet)
                    if filter_nudge and filter_nudge not in nudges:
                        nudges.append(filter_nudge)
            if (
                used in {"view", "compare", "inspect_stock"}
                and not sheet.pending_question
                and not _needs_ask_spec(sheet)
                and not sheet.cart_has_item
            ):
                select_nudge = _llm_select_best(sheet)
                if select_nudge and select_nudge not in nudges:
                    nudges.append(select_nudge)
            # Near-miss rescue: high local score but missing gated attr → nudge view others.
            if (
                used in {"view", "inspect_stock"}
                and sheet.candidate_scores
                and not sheet.cart_has_item
            ):
                near = [
                    (k, s)
                    for k, s in sheet.candidate_scores.items()
                    if float(s.get("score_n") or 0) >= 0.55
                    and s.get("missing")
                    and s.get("hard_ok", True)
                ]
                near.sort(key=lambda kv: -float(kv[1].get("score_n") or 0))
                if near and not sheet.llm_pick_id:
                    miss = near[0][1].get("missing") or []
                    nudges.append(
                        "SCOUT NEAR-MISS: top scored candidates still miss "
                        f"{miss[:3]}. View another SCORE_TOP listing or reformulate "
                        "search with the missing literal tokens."
                    )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(observation),
                }
            )
            event_nudge = _market_event_nudge(observation, sheet)
            if event_nudge and event_nudge not in nudges:
                nudges.append(event_nudge)
        if "place_test_order" in _call_names(tool_calls):
            sheet.event_recovery_mode = False
            sheet.cart_has_item = False

        user_text = _user_text(result.get("user_message"))
        if user_text:
            messages.append({"role": "user", "content": user_text})
            _apply_shopper_update(sheet, user_text)
            if sheet.pending_question:
                nudges.append(
                    "Shopper intervention detected. Next tool MUST be `message`. "
                    "Answer briefly before carting or ordering."
                )
            else:
                nudges.append(
                    "Shopper updated constraints. Refresh the STRATEGY SHEET, "
                    "re-verify hard constraints, then continue."
                )

        for nudge in nudges:
            messages.append({"role": "user", "content": nudge})

        if any(call["observation"]["done"] for call in result["calls"]):
            break

    return dialogue


__all__ = ["agent_main"]
