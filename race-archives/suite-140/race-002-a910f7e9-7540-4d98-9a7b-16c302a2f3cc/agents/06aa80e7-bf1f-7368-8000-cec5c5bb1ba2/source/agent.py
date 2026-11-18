from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------
# Models / proxy
# ---------------------------------------------------------------------------

_LIGHT = "deepseek/deepseek-v3.2"
_DENSE = "qwen/qwen3.5-397b-a17b"
_FALLBACKS = (
    "google/gemma-4-31b-it",
    "openai/gpt-oss-120b",
    "deepseek/deepseek-chat-v3.1",
)
_PROVIDER_MAP = {
    "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
    "deepseek-ai/DeepSeek-V3.1-TEE": "deepseek/deepseek-chat-v3.1",
    "deepseek-ai/DeepSeek-V3-0324-TEE": "deepseek/deepseek-chat-v3-0324",
    "deepseek-ai/DeepSeek-R1-0528-TEE": "deepseek/deepseek-r1-0528",
    "Qwen/Qwen3-32B-TEE": "qwen/qwen3-32b",
    "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
    "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
    "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
    "openai/gpt-oss-120b-TEE": "openai/gpt-oss-120b",
    "XiaomiMiMo/MiMo-V2-Flash-TEE": "xiaomi/mimo-v2-flash",
    "unsloth/Mistral-Nemo-Instruct-2407-TEE": "mistralai/mistral-small-2603",
    "zai-org/GLM-5.1-TEE": _DENSE,
    "zai-org/GLM-5.2-TEE": _DENSE,
    "moonshotai/Kimi-K2.6-TEE": _DENSE,
    "moonshotai/Kimi-K2.5-TEE": _DENSE,
    "moonshotai/kimi-k2.6": _DENSE,
    "z-ai/glm-5.1": _DENSE,
    "z-ai/glm-5.2": _DENSE,
}

_http = ProxyClient(timeout=120, max_retries=2)
_OBS_LIMIT = 6000
_CALL_CAP = 16

# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

TF1 = "intent_decomposition"
TF2 = "retrieval_recall"
TF3 = "constraint_satisfaction"
TF4 = "preference_reasoning"
TF5 = "ranking"
TF6 = "recovery"
TF7 = "justification"
_ALL = (TF1, TF2, TF3, TF4, TF5, TF6, TF7)
_TIE_ORDER = (TF5, TF6, TF7, TF3, TF1, TF4, TF2)

_META_CLAIM_FIELDS = frozenset(
    {
        "category",
        "category_path",
        "title",
        "summary",
        "evidence",
        "reason",
        "description",
        "name",
        "note",
        "display",
        "device_operating_system",
        "operating_system",
        "os",
    }
)

# ---------------------------------------------------------------------------
# Lightweight structural extractors (query contract only)
# ---------------------------------------------------------------------------

_RX_VARIANT = re.compile(r"(?<!\d)(\d{6,})::([A-Za-z0-9._-]+)")
_RX_QUOTE = re.compile(r"[\"\u201c]([^\"\u201d]{3,55})[\"\u201d]")
_RX_CEILING = re.compile(
    r"(?:budget|cap|ceiling|at most|no more than|under|below|up to|"
    r"max(?:imum)?(?:\s+(?:of|price))?)\s*[:=]?\s*([\d][\d,]*(?:\.\d+)?)",
    re.I,
)
_RX_BRAND = re.compile(
    r"preferred\s+brand\s+([A-Za-z0-9][\w.&'-]{0,40})"
    r"|prefer(?:red)?\s+(?:the\s+)?brand\s+([A-Za-z0-9][\w.&'-]{0,40})",
    re.I,
)
_RX_ASK = re.compile(
    r"(?:only|will)\s+(?:tell|share|reveal).{0,48}\bif\s+you\s+ask|"
    r"\bif you ask\b|\bwhen asked\b|\bwithhold\b.{0,40}\bspec\b|"
    r"ask\s+(?:me\s+)?(?:for\s+)?(?:the\s+)?(?:exact\s+)?(?:figure|spec|requirement)",
    re.I,
)
_RX_WHY = re.compile(
    r"\bwhy\s+(?:this|that|your|the|it)\b|"
    r"\bwhy\b.{0,24}(?:the one|it'?s|your pick|you (?:choose|pick|order))|"
    r"\btell me why\b|\bgive me a reason\b|\bjustif(?:y|ication)\b|"
    r"\bexplain\s+(?:the\s+)?(?:choice|pick|order)\b|"
    r"include\s+justification|claims?\s+must\s+be\s+grounded",
    re.I,
)
_RX_OPTIONAL_WHY = re.compile(
    r"claims?\s+(?:are|is)\s+optional|justification\s+(?:is\s+)?optional",
    re.I,
)


def _goal_head(query: str) -> str:
    text = query or ""
    cut = text.find("Task rules:")
    return text if cut < 0 else text[:cut]


def _uniq(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _tool_labels(tools: list[Any]) -> list[str]:
    names: list[str] = []
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
        elif entry.get("name"):
            names.append(str(entry["name"]))
    return names


def _claim_enum(tools: list[Any]) -> list[str]:
    """Pull place_test_order justification.claims.field enum from live schema."""
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        if not isinstance(fn, dict) or str(fn.get("name") or "") != "place_test_order":
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        just = props.get("justification") if isinstance(props.get("justification"), dict) else {}
        jprops = just.get("properties") if isinstance(just.get("properties"), dict) else {}
        claims = jprops.get("claims") if isinstance(jprops.get("claims"), dict) else {}
        items = claims.get("items") if isinstance(claims.get("items"), dict) else {}
        iprops = items.get("properties") if isinstance(items.get("properties"), dict) else {}
        field = iprops.get("field") if isinstance(iprops.get("field"), dict) else {}
        enum = field.get("enum")
        if isinstance(enum, list):
            return [str(x) for x in enum if isinstance(x, str) and x.strip()]
    return []


def _resolve_model(raw: str) -> str:
    mapped = _PROVIDER_MAP.get(raw, raw)
    low = mapped.lower()
    if "kimi" in low or "glm" in low or low.startswith(("z-ai/", "moonshotai/")):
        return _DENSE
    return mapped


def _pick_model(family: str) -> str:
    override = (getenv("SANDBOX_MODEL") or "").strip()
    if override:
        return _resolve_model(override)
    return _LIGHT if family == TF2 else _DENSE


# ---------------------------------------------------------------------------
# Brief + family routing (structural cues from the public query)
# ---------------------------------------------------------------------------


class Brief:
    __slots__ = (
        "query",
        "goal",
        "variants",
        "ceiling",
        "phrase",
        "brand_wish",
        "needs_ask",
        "needs_why",
        "omit_claims",
        "tools",
        "claim_fields",
        "max_steps",
        "max_calls",
    )

    def __init__(
        self,
        *,
        query: str,
        tools: list[Any],
        max_steps: int,
        max_calls: int,
    ) -> None:
        goal = _goal_head(query).strip()
        quote = _RX_QUOTE.search(goal)
        ceil = _RX_CEILING.search(goal)
        brand = _RX_BRAND.search(goal)
        why_hit = bool(_RX_WHY.search(goal))
        optional = bool(_RX_OPTIONAL_WHY.search(query))
        needs_why = why_hit and not (optional and not re.search(r"\bwhy\b|reason|justify", goal, re.I))
        self.query = query or ""
        self.goal = goal
        self.variants = _uniq([f"{a}::{b}" for a, b in _RX_VARIANT.findall(query or "")])
        self.ceiling = ceil.group(1) if ceil else ""
        self.phrase = quote.group(1).strip() if quote else ""
        self.brand_wish = (
            (brand.group(1) or brand.group(2) or "").strip().rstrip(".,;:") if brand else ""
        )
        self.needs_ask = bool(_RX_ASK.search(goal))
        self.needs_why = needs_why
        self.omit_claims = not needs_why
        self.tools = _tool_labels(tools)
        self.claim_fields = _claim_enum(tools)
        self.max_steps = max(1, int(max_steps))
        self.max_calls = max(1, min(_CALL_CAP, int(max_calls)))

    def sheet(self) -> str:
        lines = ["EPISODE FACTS"]
        if self.variants:
            lines.append("Named variants: " + ", ".join(self.variants))
        if self.ceiling:
            lines.append(f"Spend ceiling token: {self.ceiling}")
        if self.phrase:
            lines.append(f'Quoted phrase: "{self.phrase}"')
        if self.brand_wish:
            lines.append(f"Preferred brand (soft): {self.brand_wish}")
        if self.needs_ask:
            lines.append("Hidden firm figure: ask via message before search.")
        if self.needs_why:
            lines.append("Justification claims required on place_test_order.")
        elif self.omit_claims:
            lines.append("Omit optional justification/claims.")
        return "\n".join(lines) if len(lines) > 1 else ""


def _cue_hits(text: str, patterns: tuple[str, ...]) -> int:
    low = text.lower()
    return sum(1 for p in patterns if p in low)


def route_family(brief: Brief) -> str:
    """Infer TF from structural query features + light lexical cues."""
    blob = brief.query
    scores = {tf: 0 for tf in _ALL}

    # Structural signals (generalize across EnvPacks)
    if len(brief.variants) >= 2:
        scores[TF5] += 4
    if brief.needs_ask:
        scores[TF3] += 3
    if brief.needs_why:
        scores[TF7] += 3
    if brief.phrase:
        scores[TF2] += 2
    if brief.brand_wish:
        scores[TF1] += 2

    # Light lexical cues — task-contract language, not product suite content
    scores[TF5] += _cue_hits(
        blob,
        ("trade-off", "tradeoff", "re-rank", "rerank", "strictly worse", "closed set", "named candidate"),
    )
    scores[TF6] += _cue_hits(
        blob,
        ("least-relax", "least relax", "fallback", "same model", "other colour", "other color", "sold out", "unavailable"),
    )
    scores[TF7] += _cue_hits(
        blob,
        ("justification", "grounded claim", "tell me why", "give me a reason"),
    )
    scores[TF1] += _cue_hits(
        blob,
        ("nice-to-have", "nice to have", "preferred brand", "firm", "soft"),
    )
    if "firm" in blob.lower() and "soft" in blob.lower():
        scores[TF1] += 2
    scores[TF3] += _cue_hits(
        blob,
        ("exact variant", "exact sku", "specific sku", "gated spec", "every firm requirement", "every requirement must"),
    )
    scores[TF4] += _cue_hits(
        blob,
        ("not the cheapest", "not just the cheapest", "soft axes", "use-case", "use case", "intended use", "rather pay", "matters more"),
    )
    scores[TF2] += _cue_hits(
        blob,
        ("reformulate", "shortlist", "distinct product", "off-category", "off category"),
    )

    if not any(scores.values()):
        return TF2

    ranked = sorted(
        ((tf, sc) for tf, sc in scores.items() if sc > 0),
        key=lambda kv: (-kv[1], _TIE_ORDER.index(kv[0])),
    )
    primary, pscore = ranked[0]
    # Closed candidate sets dominate near-ties
    if len(brief.variants) >= 2 and primary != TF5 and scores[TF5] >= pscore - 1:
        return TF5
    return primary


# ---------------------------------------------------------------------------
# Playbooks (original wording; strategies specialized per TF)
# ---------------------------------------------------------------------------

_CORE = """
You are KG, an ORO Bench shopping agent.

Contract
- Use ONLY the supplied environment tools.
- Continue until an observation reports done=true.
- Respect max_steps and max_calls_per_turn.
- Parallelize tool calls when the limit allows and it saves steps.

Healthy loop
1) Discover — search / filter (or message first when a firm figure is withheld)
2) Inspect — view / compare / inspect_stock
3) Decide — add_to_cart / remove_from_cart
4) Confirm — inspect_stock / inspect_cart on the final SKU
5) Finish — place_test_order (with justification.claims only when required)

Precision
A candidate is valid ONLY if every FIRM constraint holds on the EXACT sku ordered.
Soft preferences never override firm walls.
Search matches VALUES, not JSON field names — prefer type/brand/model/phrase tokens.
One small reformulation is enough when the first search is weak.

Messaging
If the shopper asks a question, call message THIS turn with a concrete answer
before cart/order. Ignoring a question fails even when the final SKU is valid.

Market events
After stock_changed, price_changed, failed cart, or remove_from_cart: re-check
price/stock, recover per family rules, then place_test_order so the event closes.

First cart is graded — never cart a placeholder. Leave room after a shock to
inspect and order.
""".strip()

_PLAY: dict[str, str] = {
    TF1: """
# TF1 · Intent decomposition
Separate BINDING walls from WISH preferences before any tool call.

Binding: category, measurable firm specs, budget/ceiling, in-stock.
Wish: preferred brand and nice-to-haves.

Keep two shortlists in parallel:
1) brand + every binding wall
2) binding-only backup (any brand)

First cart rule: take the preferred brand ONLY when a fully binding-compliant
option exists; otherwise cart the best binding-compliant item of any brand.
If the brand wish is impossible under the walls, do not freeze — buy any fully
compliant unit. After shocks, rebuild under the same walls, prefer brand again
only if still clean, inspect_stock, then order. Omit optional claims.
""".strip(),
    TF2: """
# TF2 · Retrieval recall
Start from the shopper's distinctive tokens / quoted phrase.

Plan:
1) search with product type + distinctive tokens (keep numbers/units)
2) if empty or off-class: exactly ONE near-miss refine
3) collect several distinct product identities before committing
4) a quoted phrase must appear in viewed title/facts — brand alone is not enough
5) drop accessories and off-path categories
6) once an in-budget in-stock match is verified, order promptly

Do not wait for permission after a verified winner.
""".strip(),
    TF3: """
# TF3 · Constraint satisfaction
Every firm gate must hold on the EXACT ordered sku.

Ask-gate: if a firm figure is withheld until asked, call message FIRST and do
not search/filter until the shopper replies. Restate their figure verbatim.

Never trust titles alone — view the option row, then inspect_stock that sku.
Sellout recovery: switch to a NEW product_id that still clears EVERY firm gate;
a sibling option that drops a firm figure is invalid.
Before ordering, mentally fill a verification matrix: each firm gate → observed
source on this exact sku (view facts / option row / inspect_stock). Titles and
search snippets are not sources. After a market event the matrix is stale —
re-verify. Finish with place_test_order (omit optional claims).
""".strip(),
    TF4: """
# TF4 · Preference reasoning
Hard walls first: category, budget, in-stock, real device (not accessory).
Soft axes: only use-case preferences the shopper actually named — invent none.

Before the first cart:
- optionally one short clarify message
- compare/view ≥2 hard-compliant candidates on named soft axes using raw facts
- a slightly dearer item that fits soft goals better is preferred if still in budget

After cart / after shock: stop essays — recover → inspect_stock → place_test_order.
Do not default to cheapest among hard-valid survivors when a better soft fit exists.
""".strip(),
    TF5: """
# TF5 · Ranking
Closed set of named product_id::sku candidates only — do not open-search.

Plan:
1) batch view/compare every named variant before any cart
2) rank ONLY by the stated tradeoff priority — invent no axes
3) never pick a strictly dominated option
4) first cart = current #1 (first cart is graded)
5) after stock/price/cart events: re-rank survivors from scratch, cart the NEW #1,
   inspect_stock, then place_test_order

Read axis values from title, option labels, AND facts. Missing facts do not
authorize inventing a different axis. place_test_order = product_id + sku only
unless a why/justification is explicitly required.
""".strip(),
    TF6: """
# TF6 · Recovery
Never relax: category, budget, or in-stock.

Least-relaxation ladder (climb one level at a time; do not skip):
0) exact model + exact color
1) same model, other color
2) other models that still meet firm walls

Prefer structured model/color facts over conflicting marketing titles.
After every cart change: inspect_stock, then order only if still valid.
Omit optional justification/claims unless the goal asks for a reason.
""".strip(),
    TF7: """
# TF7 · Justification
place_test_order MUST include minimal grounded justification.claims.

Field names: use the live tool-schema enum when present; otherwise snake_case
attribute names for the specs the shopper asked about. Never meta labels
(title, category, summary, evidence, display, OS).

Minimal set: price + in_stock + the asked specification (+ brand if named).
Values must equal viewed / inspect_stock facts for the exact sku — copy text,
do not rewrite units or invent features.
After shocks: inspect_stock the affected listing → remove if invalid → recover →
rebuild claims from fresh observations → place_test_order.
""".strip(),
}


def _system_prompt(family: str, brief: Brief, policy: dict[str, Any]) -> str:
    tools = ", ".join(brief.tools) if brief.tools else "(see tool schemas)"
    parts = [_CORE, _PLAY.get(family, _PLAY[TF2])]
    sheet = brief.sheet()
    if sheet:
        parts.append(sheet)
    if family == TF5 and brief.variants:
        parts.append("Named candidate variants (view these; re-rank after events):\n" + ", ".join(brief.variants))
    if family == TF7 and brief.claim_fields:
        parts.append("Allowed claim fields from tool schema:\n" + ", ".join(brief.claim_fields[:24]))
    parts.append(
        "# Runtime\n"
        f"Detected family: {family}\n"
        f"Available tools: {tools}\n"
        f"max_steps={policy.get('max_steps')} max_calls_per_turn={policy.get('max_calls_per_turn', 1)}"
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _parse_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("function", {}).get("arguments", "{}")
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


def _fn_name(call: dict[str, Any]) -> str:
    fn = call.get("function") if isinstance(call, dict) else None
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return "?"


def _call_names(calls: list[dict[str, Any]]) -> list[str]:
    return [_fn_name(c) for c in calls]


def _shrink(obs: Any, limit: int = _OBS_LIMIT) -> str:
    try:
        text = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obs)
    if len(text) <= limit:
        return text
    return text[: limit - 16] + "...<truncated>"


def _is_question(text: str) -> bool:
    body = (text or "").strip().lower()
    if not body:
        return False
    if "?" in body:
        return True
    openers = (
        "which",
        "what",
        "does",
        "did",
        "can you",
        "could you",
        "would you",
        "wait",
        "hold on",
        "before you",
        "re-check",
        "recheck",
    )
    return any(body.startswith(w) for w in openers)


def _market_kind(obs: Any) -> str:
    if not isinstance(obs, dict):
        return ""
    if obs.get("stock_changed"):
        return "stock_changed"
    if obs.get("price_changed"):
        return "price_changed"
    if obs.get("event"):
        return str(obs.get("event"))
    node = obs.get("market_event")
    if isinstance(node, dict):
        return str(node.get("kind") or "market_event")
    if node:
        return "market_event"
    nested = obs.get("observation")
    if isinstance(nested, dict):
        return _market_kind(nested)
    return ""


def _as_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _done(result_calls: list[Any]) -> bool:
    for call in result_calls:
        if not isinstance(call, dict):
            continue
        obs = call.get("observation")
        if isinstance(obs, dict) and obs.get("done"):
            return True
    return False


# ---------------------------------------------------------------------------
# Catalog memory + claim polish (TF7)
# ---------------------------------------------------------------------------


class CatalogMemory:
    """Remember product_id/sku rows seen in observations for grounding claims."""

    __slots__ = ("rows", "order", "focus")

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.order: list[tuple[str, str]] = []
        self.focus: list[str] = []

    def note_focus(self, args: dict[str, Any]) -> None:
        pid = str(args.get("product_id") or "")
        if not pid:
            return
        if pid in self.focus:
            self.focus.remove(pid)
        self.focus.insert(0, pid)

    def learn(self, obs: Any, depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(obs, list):
            for item in obs[:16]:
                self.learn(item, depth + 1)
            return
        if not isinstance(obs, dict):
            return
        pid = str(obs.get("product_id") or "").strip()
        sku = str(obs.get("sku") or "").strip()
        if pid and sku:
            key = (pid, sku)
            if key not in self.rows:
                self.order.append(key)
            row = self.rows.setdefault(key, {})
            for field in ("price", "in_stock", "title", "brand"):
                if obs.get(field) is not None:
                    row[field] = obs.get(field)
            facts = obs.get("facts")
            if isinstance(facts, dict):
                for k, v in facts.items():
                    if v is not None and str(k).lower() not in _META_CLAIM_FIELDS:
                        row[str(k)] = v
        for value in obs.values():
            if isinstance(value, (dict, list)):
                self.learn(value, depth + 1)

    def invalid(self, key: tuple[str, str], brief: Brief) -> bool:
        row = self.rows.get(key)
        if not row:
            return False
        stock = row.get("in_stock")
        if stock is False or str(stock).lower() in {"false", "0", "no", "out_of_stock"}:
            return True
        ceiling = _as_float(brief.ceiling)
        price = _as_float(row.get("price"))
        return ceiling is not None and price is not None and price > ceiling

    def pick(self, brief: Brief) -> tuple[str, str] | None:
        named = {v.split("::", 1)[0] for v in brief.variants if v}
        pool = [k for k in self.order if not named or k[0] in named]
        ranked = [k for pid in self.focus for k in pool if k[0] == pid]
        for key in [*ranked, *pool]:
            if all(key) and not self.invalid(key, brief) and self.rows.get(key):
                return key
        return None

    def ground_claims(self, pid: str, sku: str, allowed: list[str]) -> list[dict[str, str]]:
        row = self.rows.get((pid, sku)) or self.rows.get((pid, "")) or {}
        if not row:
            return []
        claims: list[dict[str, str]] = []
        seen: set[str] = set()

        def add(field: str, value: Any) -> None:
            fl = field.lower().strip()
            if not fl or fl in seen or fl in _META_CLAIM_FIELDS:
                return
            if allowed and field not in allowed and fl not in {a.lower() for a in allowed}:
                # try case-insensitive match into allowed enum
                match = next((a for a in allowed if a.lower() == fl), None)
                if match is None:
                    return
                field = match
            text = str(value).strip() if value is not None else ""
            if fl == "in_stock":
                text = "true" if text.lower() in {"true", "1", "yes", "in_stock"} else text
            if text == "":
                return
            seen.add(fl)
            claims.append({"field": field, "value": text})

        prefer = ["price", "in_stock", "brand"]
        for key in prefer:
            if key in row:
                add(key, row[key])
        for key, value in row.items():
            if key in prefer or key in {"title"}:
                continue
            if isinstance(value, (str, int, float, bool)):
                add(str(key), value)
            if len(claims) >= 4:
                break
        return claims[:4]


def _catalog_keys(obs_list: list[Any], *, limit: int = 12) -> list[str]:
    skip = _META_CLAIM_FIELDS | {
        "error",
        "done",
        "calls",
        "observation",
        "event",
        "stock_changed",
        "price_changed",
        "products",
        "results",
        "items",
        "product_id",
        "sku",
        "options",
        "sku_options",
        "market_event",
        "title",
    }
    keys: list[str] = []
    seen: set[str] = set()

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 6 or len(keys) >= limit:
            return
        if isinstance(node, dict):
            for raw_key, value in node.items():
                key = str(raw_key).strip()
                low = key.lower()
                if not key or low in skip:
                    walk(value, depth + 1)
                    continue
                if low not in seen and isinstance(value, (str, int, float, bool)):
                    seen.add(low)
                    keys.append(key)
                else:
                    walk(value, depth + 1)
        elif isinstance(node, list):
            for item in node[:12]:
                walk(item, depth + 1)

    for obs in obs_list:
        walk(obs)
    return keys[:limit]


def _wants_claims(family: str, brief: Brief) -> bool:
    return family == TF7 or brief.needs_why


def polish_order_args(
    args: dict[str, Any],
    *,
    family: str,
    brief: Brief,
    catalog: CatalogMemory,
) -> dict[str, Any]:
    out = dict(args)
    if not _wants_claims(family, brief):
        out.pop("justification", None)
        return out

    just = out.get("justification")
    claims_in: list[Any] = []
    if isinstance(just, dict) and isinstance(just.get("claims"), list):
        claims_in = just["claims"]

    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    for claim in claims_in:
        if not isinstance(claim, dict):
            continue
        field = str(claim.get("field") or "").strip()
        value = claim.get("value")
        fl = field.lower()
        if not field or fl in _META_CLAIM_FIELDS:
            continue
        text = str(value).strip() if value is not None else ""
        if fl == "in_stock":
            text = "true" if text.lower() in {"true", "1", "yes", "in_stock"} else text
        if text == "" or fl in seen:
            continue
        if brief.claim_fields:
            allowed_l = {a.lower(): a for a in brief.claim_fields}
            if fl not in allowed_l:
                continue
            field = allowed_l[fl]
        seen.add(fl)
        cleaned.append({"field": field, "value": text})

    if len(cleaned) < 2:
        pid = str(out.get("product_id") or "")
        sku = str(out.get("sku") or "")
        grounded = catalog.ground_claims(pid, sku, brief.claim_fields)
        for claim in grounded:
            fl = claim["field"].lower()
            if fl not in seen:
                seen.add(fl)
                cleaned.append(claim)

    if cleaned:
        out["justification"] = {"claims": cleaned[:4]}
    return out


# ---------------------------------------------------------------------------
# Pace keeper — break identical loops toward seal
# ---------------------------------------------------------------------------


class PaceKeeper:
    __slots__ = ("counts", "breaks")

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.breaks = 0

    @staticmethod
    def signature(name: str, args: dict[str, Any]) -> str:
        try:
            body = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            body = str(sorted(args.items()))
        return f"{name}|{body}"

    def bump(self, name: str, args: dict[str, Any]) -> int:
        key = self.signature(name, args)
        n = self.counts.get(key, 0) + 1
        self.counts[key] = n
        return n

    def next_step(
        self,
        brief: Brief,
        catalog: CatalogMemory,
        state: dict[str, Any],
        *,
        family: str,
    ) -> tuple[str, dict[str, Any]] | None:
        if brief.needs_ask and not state.get("asked_spec"):
            return None
        pick = catalog.pick(brief)
        if not pick:
            return None
        pid, sku = pick
        args = {"product_id": pid, "sku": sku}
        if pid not in catalog.focus:
            return ("view", {"product_id": pid})
        if state.get("cart_key") != pick:
            return ("add_to_cart", args)
        if state.get("verified") != pick:
            return ("inspect_stock", args)
        order_args = dict(args)
        if _wants_claims(family, brief):
            claims = catalog.ground_claims(pid, sku, brief.claim_fields)
            if claims:
                order_args["justification"] = {"claims": claims}
        return ("place_test_order", order_args)


def _rewrite_call(call: dict[str, Any], name: str, args: dict[str, Any]) -> None:
    fn = call.get("function")
    if isinstance(fn, dict):
        fn["name"] = name
        fn["arguments"] = json.dumps(args)


# ---------------------------------------------------------------------------
# Coaching
# ---------------------------------------------------------------------------


def _opening_coaches(family: str, brief: Brief) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    if family == TF3 and brief.needs_ask:
        notes.append(
            {
                "role": "user",
                "content": (
                    "Coach: a firm requirement appears withheld until you ask. "
                    "Call message first for that exact figure before any search/filter."
                ),
            }
        )
    if family == TF5:
        notes.append(
            {
                "role": "user",
                "content": (
                    "Coach [TF5]: view/compare the named candidate variants and rank by "
                    "the stated tradeoff axes before add_to_cart. After any stock/cart "
                    "event, re-rank and order the new #1. place_test_order = product_id+sku only."
                ),
            }
        )
    if family == TF7:
        notes.append(
            {
                "role": "user",
                "content": (
                    "Coach [TF7]: on place_test_order, justification.claims must copy "
                    "catalog JSON keys from view/inspect_stock — price, in_stock, the "
                    "asked spec key, and at most one evidenced extra (prefer brand)."
                ),
            }
        )
    if family == TF1:
        notes.append(
            {
                "role": "user",
                "content": (
                    "Coach [TF1]: keep a brand+firm shortlist and a firm-only backup. "
                    "Preferred brand wins the first cart only when fully firm-compliant; "
                    "otherwise cart the best firm-compliant any-brand item."
                ),
            }
        )
    return notes


def _turn_coach(
    *,
    family: str,
    turn: int,
    brief: Brief,
    last_tools: list[str],
    last_obs: list[Any],
    await_reply: bool,
    saw_market: bool,
    ordered: bool,
    first_cart: bool,
    asked_spec: bool,
) -> str | None:
    used = " ".join(last_tools)
    if await_reply:
        return (
            "Coach: the shopper just spoke. Call message THIS turn with a concrete "
            "answer (identity + key firm evidence + price/stock) before cart/order."
        )
    if family == TF3 and brief.needs_ask and not asked_spec and "message" not in used:
        return (
            "Coach: call message now to request the withheld firm figure. "
            "Do not search until they reply."
        )
    if family == TF5 and not ordered:
        if saw_market and turn >= 2 and "place_test_order" not in used:
            return (
                "Coach [TF5]: market/cart state changed. Re-inspect named survivors, "
                "re-rank by stated priority, cart the NEW #1, inspect_stock, then "
                "place_test_order with only product_id+sku."
            )
        if turn >= 8:
            return (
                "Coach [TF5]: stop searching. Re-rank named in-stock candidates, "
                "cart the current #1 if needed, and place_test_order now."
            )
        if turn == 1:
            return (
                "Hint [TF5]: batch-view the named variants and rank on stated axes "
                "before the first cart."
            )
        if turn >= 2 and not first_cart and ("view" in used or "compare" in used):
            return "Coach [TF5]: candidates inspected — add_to_cart only the current #1."
    if family == TF1 and turn == 1:
        return (
            "Coach [TF1]: dual shortlist — brand+firm and firm-only. First cart honors "
            "brand only when fully firm-compliant."
        )
    if family == TF7 and not ordered and turn in (2, 5):
        keys = _catalog_keys(last_obs)
        extra = (" Observed keys: " + ", ".join(keys) + ".") if keys else ""
        return (
            "Coach [TF7]: claims must copy catalog keys from view/inspect_stock."
            + extra
        )
    if family == TF6 and saw_market and not ordered:
        return (
            "Coach [TF6]: climb the least-relaxation ladder without skipping — "
            "exact model+color → same model other color → other models. Never relax "
            "category, budget, or in-stock. Then place_test_order."
        )
    if family == TF3 and saw_market and not ordered:
        return (
            "Coach [TF3]: current SKU gone. Switch to a NEW product_id that still meets "
            "every firm constraint — siblings that drop a firm spec are invalid."
        )
    if family == TF2 and turn == 3 and "search" in used:
        return (
            "Hint: one small keyword reformulation or filter so the shortlist covers "
            "several distinct products before ordering."
        )
    if family == TF4 and turn == 2 and "compare" not in used:
        return (
            "Hint: compare candidates on soft preference axes (not only price) "
            "before the first cart add."
        )
    if not ordered and turn >= max(8, brief.max_steps - 2):
        return (
            "Coach: stop open search. Verify the carted SKU (or cart the current valid "
            "pick), then place_test_order so the episode closes."
        )
    if saw_market and not ordered and turn >= 2:
        return (
            "Coach: catalog/cart state moved. Re-check price/stock on the affected SKU. "
            "If invalid, remove and recover, then place_test_order."
        )
    if turn <= 2 and not any(n in used for n in ("search", "filter", "view", "compare", "message")):
        return (
            "Hint: start with search or filter using value-side keywords from the "
            "shopper goal, then view the best hits."
        )
    return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _infer(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_calls: int,
) -> tuple[str, list[dict[str, Any]]]:
    ladder = [model]
    for fb in _FALLBACKS:
        if fb not in ladder:
            ladder.append(fb)
    last_err: Exception | None = None
    for mid in ladder[:3]:
        inference = _http.post(
            "/inference/chat/completions",
            json_data={
                "model": mid,
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "temperature": 0,
            },
        )
        if inference is None:
            last_err = RuntimeError(f"inference failed for model={mid}")
            continue
        try:
            assistant = inference["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            last_err = exc
            continue
        content = assistant.get("content") or ""
        tool_calls = list(assistant.get("tool_calls") or [])[:max_calls]
        return content, tool_calls
    raise RuntimeError(f"inference request failed ({last_err})")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    query = str(policy.get("query") or problem_data.get("query") or "")
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    tools = policy.get("tools") or []
    brief = Brief(
        query=query,
        tools=tools,
        max_steps=int(policy.get("max_steps", 20)),
        max_calls=int(policy.get("max_calls_per_turn", 1)),
    )
    family = route_family(brief)
    model = _pick_model(family)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(family, brief, policy)},
        {"role": "user", "content": query},
        *_opening_coaches(family, brief),
    ]
    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "kg": {
                "version": 1,
                "family": family,
                "model": model,
                "variants": brief.variants,
                "provider": "openrouter",
            },
        }
    ]

    catalog = CatalogMemory()
    pace = PaceKeeper()
    state: dict[str, Any] = {
        "cart_key": None,
        "verified": None,
        "asked_spec": False,
        "first_cart": False,
        "ordered": False,
        "await_reply": False,
        "saw_market": False,
    }
    last_tools: list[str] = []
    last_obs: list[Any] = []
    stuck = (
        "No tool call received. Choose a supplied tool now.",
        "Still no tool call. Progress: search/filter, view a candidate, or place_test_order if cart holds a verified SKU.",
    )

    for turn in range(1, brief.max_steps + 1):
        hint = _turn_coach(
            family=family,
            turn=turn,
            brief=brief,
            last_tools=last_tools,
            last_obs=last_obs,
            await_reply=bool(state["await_reply"]),
            saw_market=bool(state["saw_market"]),
            ordered=bool(state["ordered"]),
            first_cart=bool(state["first_cart"]),
            asked_spec=bool(state["asked_spec"]),
        )
        if hint:
            messages.append({"role": "user", "content": hint})

        content = ""
        tool_calls: list[dict[str, Any]] = []
        for attempt in range(2):
            content, tool_calls = _infer(
                model=model,
                messages=messages,
                tools=tools,
                max_calls=brief.max_calls,
            )
            if tool_calls:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": stuck[min(attempt, len(stuck) - 1)]},
                ]
            )
        if not tool_calls:
            raise RuntimeError("model returned no tool call before completion")

        # Sanitize + pace-break identical loops
        prepared: list[dict[str, Any]] = []
        for call in tool_calls:
            name = _fn_name(call)
            args = _parse_args(call)
            if name == "place_test_order":
                args = polish_order_args(args, family=family, brief=brief, catalog=catalog)
                # Fill missing ids from last cart focus when model omits them
                if not args.get("product_id") or not args.get("sku"):
                    pick = catalog.pick(brief)
                    if pick:
                        args.setdefault("product_id", pick[0])
                        args.setdefault("sku", pick[1])
                _rewrite_call(call, name, args)
            catalog.note_focus(args)
            repeats = pace.bump(name, args)
            if repeats >= 2 and name not in {"message", "place_test_order"}:
                nxt = pace.next_step(brief, catalog, state, family=family)
                if nxt:
                    name, args = nxt
                    _rewrite_call(call, name, args)
                    pace.breaks += 1
            prepared.append(call)

        # Late-episode seal nudge: if still looping inspect without order
        if (
            not state["ordered"]
            and turn >= max(6, brief.max_steps - 3)
            and all(_fn_name(c) in {"inspect_stock", "inspect_cart", "view", "compare"} for c in prepared)
        ):
            nxt = pace.next_step(brief, catalog, state, family=family)
            if nxt and prepared:
                _rewrite_call(prepared[0], nxt[0], nxt[1])

        messages.append({"role": "assistant", "content": content, "tool_calls": prepared})
        group_id = f"{problem_id}-turn-{turn}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {"name": _fn_name(call), "args": _parse_args(call)},
                }
                for index, call in enumerate(prepared, start=1)
            ],
        }
        result = _http.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")

        dialogue.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": prepared,
                "environment_result": result,
                "kg_turn": {
                    "family": family,
                    "turn": turn,
                    "pace_breaks": pace.breaks,
                },
            }
        )

        last_tools = _call_names(prepared)
        last_obs = []
        state["await_reply"] = False
        for tool_call, call_result in zip(prepared, result.get("calls") or [], strict=False):
            obs = call_result.get("observation") if isinstance(call_result, dict) else None
            last_obs.append(obs)
            catalog.learn(obs)
            kind = _market_kind(obs)
            if kind:
                state["saw_market"] = True
            name = _fn_name(tool_call)
            args = _parse_args(tool_call)
            if name == "message":
                state["asked_spec"] = True
            if name == "add_to_cart":
                state["first_cart"] = True
                state["cart_key"] = (str(args.get("product_id") or ""), str(args.get("sku") or ""))
            if name == "inspect_stock":
                state["verified"] = (str(args.get("product_id") or ""), str(args.get("sku") or ""))
            if name == "place_test_order":
                state["ordered"] = True
            if name == "remove_from_cart":
                state["cart_key"] = None
                state["verified"] = None
                state["saw_market"] = True
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id") or f"{group_id}-tool",
                    "content": _shrink(obs),
                }
            )

        user_message = result.get("user_message") if isinstance(result, dict) else None
        if isinstance(user_message, dict) and user_message.get("content"):
            text = str(user_message["content"])
            messages.append({"role": "user", "content": text})
            if _is_question(text):
                state["await_reply"] = True

        if _done(result.get("calls") or []):
            break

        # Immediate post-event coaches
        if state["saw_market"] and not state["ordered"]:
            if family == TF5:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Coach [TF5]: re-rank remaining named candidates, cart the "
                            "new #1, inspect_stock, place_test_order (product_id+sku only)."
                        ),
                    }
                )
            elif family == TF6:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Coach [TF6]: climb least-relax ladder; never relax category/"
                            "budget/stock; then place_test_order."
                        ),
                    }
                )

    return dialogue


__all__ = ["agent_main"]
