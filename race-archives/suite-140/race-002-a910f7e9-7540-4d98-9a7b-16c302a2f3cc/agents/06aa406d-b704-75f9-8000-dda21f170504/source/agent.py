from __future__ import annotations

import json
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

_SYSTEM = """You are a general shopping agent for ORO Bench generated environments.

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
7. place_test_order: pass product_id + sku. Include justification.claims ONLY
   when strategy is fact_justify AND the sheet lists REQUIRED_CLAIMS. Claim
   values MUST be strings copied from viewed/inspected facts for THAT sku
   (any attribute name the sheet requires). For every other strategy, omit
   justification entirely — invented fields like device_type, brand=null, or
   use_case fail hard truthfulness gates even when the product is correct.

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
        qplan = [self.primary_query] if self.primary_query else []
        qplan.extend(self.reformulations)
        qplan = [q for q in qplan if q]
        if qplan:
            lines.append("SEARCH PLAN:")
            for i, q in enumerate(qplan[:5], start=1):
                lines.append(f"  {i}. {q}")
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


def _lookup_fact(facts: dict[str, Any], field: str) -> Any:
    """Find a fact by exact key, case-insensitive key, or simple alias."""
    if not field:
        return None
    if field in facts and facts[field] is not None:
        return facts[field]
    lower = field.lower()
    for key, val in facts.items():
        if str(key).lower() == lower and val is not None:
            return val
    # Generic short aliases (not product-family specific).
    aliases = {
        "size": ("screen_size", "display_size", "size"),
        "screen_size": ("screen_size", "size", "display_size"),
        "memory": ("memory", "ram"),
        "ram": ("ram", "memory"),
        "display": ("display", "resolution"),
        "resolution": ("resolution", "display"),
        "refresh": ("refresh_rate", "refresh"),
        "refresh_rate": ("refresh_rate", "refresh"),
        "color": ("color", "colour"),
        "colour": ("colour", "color"),
    }
    for alt in aliases.get(lower, ()):
        if alt in facts and facts[alt] is not None:
            return facts[alt]
        for key, val in facts.items():
            if str(key).lower() == alt and val is not None:
                return val
    return None


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
    # Compare leading numbers when both sides are quantitative (6.64in vs 6.64,
    # 16GB vs 16, 120Hz vs 120, etc.).
    on = _as_float(o)
    nn = _as_float(n)
    if on is not None and nn is not None and abs(on - nn) < 1e-9:
        return True
    # Token overlap for multi-word specs ("waterproof ip67" vs "ip67").
    o_toks = set(re.findall(r"[a-z0-9]+", o))
    n_toks = set(re.findall(r"[a-z0-9]+", n))
    if n_toks and n_toks.issubset(o_toks):
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

    for prod in products:
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
        # Copy ALL structured fact/attribute keys — do not whitelist by product type.
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
        # Keep a searchable blob (theme / soft-preference coverage), like tf2.
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


def _build_fact_claims(
    sheet: StrategySheet, product_id: str | None, sku: str | None
) -> list[dict[str, str]]:
    """Build claims ONLY from observed facts for arbitrary attribute names."""
    facts = _facts_for_product(sheet, product_id, sku)
    claims: list[dict[str, str]] = []
    for field_name in _required_claim_fields(sheet):
        raw = _lookup_fact(facts, field_name)
        value = _claim_value_from_fact(field_name, raw)
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
        got = _lookup_fact(facts, attr)
        if got is None:
            # Fall back to evidence blob token coverage (tf2-style), still general.
            evidence = str(facts.get("evidence_text") or facts.get("title") or "")
            if evidence and _values_compatible(evidence, need):
                continue
            return False, f"missing structured fact for {attr}"
        if not _values_compatible(got, need):
            return False, f"{attr} fact {got!r} incompatible with required {need!r}"
    return True, "ok"


def _recompute_closed_rank_top(sheet: StrategySheet) -> None:
    if sheet.strategy != "closed_rank" or not sheet.candidates:
        return
    # Use sheet-declared priority only; default to price (universal), never
    # phone/monitor-specific axes.
    priority = list(sheet.rank_priority) if sheet.rank_priority else ["price"]
    if "price" not in priority:
        priority = priority + ["price"]

    def sort_key(cand: dict[str, str]) -> tuple:
        key = f"{cand['product_id']}::{cand['sku']}"
        facts = sheet.candidate_facts.get(key) or {}
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
            # Prefer higher numeric axes except price (and explicit *cost* names).
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
        return tuple(parts)

    ranked = sorted(sheet.candidates, key=sort_key)
    top = ranked[0]
    key = f"{top['product_id']}::{top['sku']}"
    if sheet.candidate_facts.get(key):
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


def _prepare_action_args(
    sheet: StrategySheet, name: str, args: dict[str, Any]
) -> dict[str, Any]:
    if name == "place_test_order":
        out = _prepare_order_args(sheet, args)
        if (
            sheet.strategy == "closed_rank"
            and sheet.computed_top_id
            and sheet.computed_top_sku
            and not sheet.event_recovery_mode
        ):
            # Prefer computed frontier top when still pre-event / eligible.
            facts = _facts_for_product(
                sheet, sheet.computed_top_id, sheet.computed_top_sku
            )
            if facts:
                out["product_id"] = sheet.computed_top_id
                out["sku"] = sheet.computed_top_sku
        return out
    if name == "add_to_cart" and sheet.strategy == "closed_rank":
        if (
            not sheet.event_recovery_mode
            and sheet.computed_top_id
            and sheet.computed_top_sku
        ):
            out = dict(args)
            out["product_id"] = sheet.computed_top_id
            out["sku"] = sheet.computed_top_sku
            return out
    if name == "search" and sheet.strategy == "theme_recall" and sheet.theme_phrase:
        out = dict(args)
        q = out.get("query")
        if not isinstance(q, str) or not q.strip():
            out["query"] = sheet.primary_query or sheet.theme_phrase
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
    pid = args.get("product_id")
    sku = args.get("sku")
    if isinstance(pid, str) and pid.strip():
        sheet.last_product_id = pid.strip()
    if isinstance(sku, str) and sku.strip():
        sheet.last_sku = sku.strip()


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
            "Search with the EXACT theme first (see SEARCH PLAN). Reformulations "
            "must keep theme tokens and must NOT invent product types. Run "
            "primary + several reformulations, filter(category/budget/in_stock), "
            "paginate when possible, then verify and order with NO justification."
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
            "Never first-cart a title-only decoy. Order with string "
            "justification.claims covering REQUIRED_CLAIMS from owned facts only."
        ),
    }
    return base + tips.get(
        sheet.strategy,
        "Satisfy hard constraints, verify with tools, then place_test_order.",
    )


def _steps_remaining_nudge(turn: int, max_steps: int, sheet: StrategySheet) -> str | None:
    remaining = max_steps - turn + 1
    if remaining > 10:
        return None
    if sheet.event_recovery_mode:
        return (
            f"Only ~{remaining} steps left and EVENT RECOVERY is active. "
            "Finish remove → inspect backup → cart → place_test_order now."
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
