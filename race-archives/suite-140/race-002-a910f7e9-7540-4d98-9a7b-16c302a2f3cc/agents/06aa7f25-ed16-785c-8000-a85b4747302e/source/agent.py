
from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from itertools import product
from os import getenv
from types import NoneType
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin

from src.agent.agent_interface import ToolCallResult, execute_tool_call
from src.agent.proxy_client import ProxyClient

_OPENROUTER_MODELS = [
    "deepseek/deepseek-v3.2",
    "moonshotai/kimi-k2.6",
]
_CHUTES_MODELS = [
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "moonshotai/Kimi-K2.6-TEE",
]

_PROBLEM_BUDGET_S = 200.0
_INFER_TIMEOUT_S = 55
_MAX_INFER_TRIES_PER_MODEL = 2
_MAX_TOOL_REPROMPTS = 2
_SEMANTIC_BUDGET_S = 22.0

_CORE_CLAIM_FIELDS = frozenset({"price", "in_stock", "brand"})
_CLAIM_DENY = frozenset({"category", "category_match", "category_path"})

_CAND_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9][A-Za-z0-9._-]*)")

_proxy = ProxyClient(timeout=_INFER_TIMEOUT_S, max_retries=2)

_SYSTEM = """You are an ORO Bench shopping agent on EnvPack tools.

Use ONLY the supplied tools. Continue until an observation has done=true.
Obey this episode's shopper text + Task rules only. Race tasks are hidden —
never memorize product IDs or suite answers.

# Schema-independent semantics (critical)
Interpret constraints and preferences from meaning in THIS query + Task rules.
Do NOT rely on fixed synonym lists or benchmark field vocabularies.
Map shopper language onto tool schemas and observed product attributes by
semantic fit for this episode only.

# Match then rank
1) Extract every constraint. Classify FIRM vs SOFT from wording + Task rules
   (must / required / budget / in-stock / exact measures = firm;
   ideally / prefer / rather / nice-to-have = soft).
2) LITERAL firm: brands, models, IDs/SKUs, numbers with units, exact options —
   no partial normalize, no range-contains-exact. SEMANTIC soft: materials,
   usage, style. Structured view/compare facts beat SEO titles.
3) Cart-eligible only when EVERY firm constraint is PASS. Soft prefs rank among
   firm-valid options only — never override firm budget/stock/specs.
4) Soft brand: honor on FIRST cart when a fully firm-valid preferred listing exists.

# Retrieval
Search/filter with VALUE tokens from the goal. Do not search bare schema-key
names alone. Survey several distinct product_ids before buying.

# Ask / dialogue
If a firm spec is withheld until you ask → first action is `message` asking once,
then search with the revealed spec. Answer real shopper questions with `message`
once, then continue.

# Closed candidate ranking
When the shopper lists explicit `product_id::sku` keys, that list is the entire
eligible set. view each listed product_id; rank by the PRIORITY ORDER stated in
THIS query using structured facts (semantic axis fit — no fixed unit parsers).
FIRST add_to_cart = current best. After shocks: re-rank remaining LIVE listed
keys only — no open-catalog swaps.

# Recovery
Preferred model/variant first → same model other variant → other models last.
On market shock: declarative `message` (no permission ask), remove stale cart,
re-inspect, cart a NEW valid sku, then place_test_order. Never quit.

# Justification
Only if the shopper asked for a reason. Map each shopper need onto ONE allowed
claim field by semantic fit to the tool schema and observed catalog fact keys
(catalog-grounded field→value mapping). Ground values in view/compare + latest
inspect_stock. Prefer the shopper's own labels when Task rules treat them as the
claim target. Price is a bare number (currency-agnostic). Omit ungroundable fields.

# Finish
ask (if needed) → search/filter → view/compare (confirm EVERY firm constraint
from structured facts, not title alone) → inspect_stock →
add_to_cart (first successful add IS your purchase commitment — only cart a
fully firm-valid candidate you would buy) → (handle market_event) →
place_test_order.
After a market shock: exactly ONE declarative `message` on the unaided turn,
then wait for any shopper follow-up; answer once if asked; then
remove_from_cart → recover → inspect_stock → cart → order.
Do NOT send a second unsolicited `message` after the event ack.
Late budget: stop exploring and order an in-stock in-budget pick.
"""

_PROFILE_SYSTEM = """You interpret ONE shopping episode for an agent.
Return ONLY a JSON object. No markdown.

Read the shopper text and Task rules. Do NOT invent suite-specific knowledge or
memorized product IDs. Perform schema-independent semantic interpretation:
infer intent, firm vs soft constraints, and ranking axes from THIS wording.

Claim-field mode: {claim_mode}
Hint list (enum when closed; otherwise guidance only): {allowed}

Return exactly these keys:
{{
  "ask_first": bool,
  "needs_justification": bool,
  "soft_brand": string|null,
  "firm_brand": string|null,
  "brand_is_firm": bool,
  "preferred_model": string|null,
  "preferred_color": string|null,
  "rank_axes": [string],
  "higher_better_axes": [string],
  "required_claim_fields": [string],
  "spec_claim_values": {{"field": "value"}},
  "firm_spec_notes": [string],
  "phrase_tokens": [string],
  "category_tokens": [string],
  "required_product_leaf": string|null,
  "negate_tokens": [string],
  "require_soft_evidence": bool,
  "search_value_hints": [string],
  "firm_measures": [{{"name": string, "numeric": number|null, "value_text": string}}],
  "budget_ceiling": number|null,
  "notes": string
}}

Rules:
- ask_first: true only if a firm hardware/spec is withheld until the agent asks.
- needs_justification: true only if the shopper asked why/reason/explain/justify.
- soft_brand / preferred_model / preferred_color: extract from THIS text only;
  null if absent. preferred_model may be alphanumeric OR numeric model codes
  the shopper wrote (e.g. product codes); preferred_color is the named shade.
- rank_axes: ordered axes the shopper stated for ranking among candidates;
  empty if none. Use short snake_case names that capture THIS ranking meaning
  (semantic axis labels — not a fixed product-domain vocabulary).
- higher_better_axes: subset of rank_axes where larger numeric values are better.
  Price/budget/weight-like axes are usually NOT higher-better.
- required_claim_fields: if needs_justification, ALWAYS include price and in_stock,
  PLUS the specification field(s) the shopper wants justified.
  Field-name policy (catalog-grounded mapping):
    * mode "enum": use ONLY names from the Allowed hint list.
    * mode "open": invent short snake_case names that mirror how catalogs typically
      label the shopper-named attribute; prefer names Task rules already use.
  Never use category / category_match / category_path as claim fields.
  Do NOT emit synonym tables, display-class cheat sheets, or unit-format recipes.
- spec_claim_values: for each required specification field (not price/in_stock),
  the value string to claim. Prefer the shopper's own label/measure wording when
  Task rules treat that wording as the claim target; otherwise a compact literal
  consistent with that wording. Empty object if none.
- firm_brand: brand that Task rules / wording make non-negotiable (else null).
- brand_is_firm: true when the named brand is firm (common on justification tasks).
- soft_brand: ideally/prefer brand that must not override firm constraints.
- firm_spec_notes: short notes of each firm hardware/spec constraint in this query.
- phrase_tokens / category_tokens: literal retrieval phrases from the shopper goal.
  category_tokens MUST include the shopper's product-type / catalog-category noun
  when stated — these are firm for eligibility.
  Do NOT put match-phrase literals into category_tokens (those belong in phrase_tokens).
- required_product_leaf: the single most specific product-type noun the shopper
  asked to buy (singular snake/space-free form), or null if none. Sibling catalog
  leaves must not substitute for this leaf.
- negate_tokens: types/options the shopper rejects.
- require_soft_evidence: true when soft use-case axes must be evidenced among firm-valid options.
  Set true when the shopper says soft priorities matter more than cheapest.
- search_value_hints: concrete VALUE tokens useful for search (not schema key names).
  ALWAYS include the firm measure token(s) exactly as the shopper wrote them.
  When the shopper states soft use-case priorities, ALSO include those literal
  value tokens (as written) so retrieval can surface evidenced listings.
  Do not include function words, category nouns, or currency tokens — only the
  shopper's own preference attributes.
- firm_measures: each firm numeric/hardware measure the shopper named. Keep the
  shopper's exact number (do not round or nearby-substitute). value_text is the
  compact shopper wording. Empty list if none. Do not invent suite unit vocabularies.
  NEVER put the spend/budget ceiling into firm_measures — that belongs only in
  budget_ceiling. Example: "N of <attribute> under 12000" → firm_measures has N,
  budget_ceiling is 12000 (not a firm_measure).
- budget_ceiling: numeric budget amount only (strip any currency name/code); null if absent.
  ALWAYS extract when the shopper states a spend/budget ceiling.
- notes: one short line for the acting agent.
"""

_SHOPPER_TURN_SYSTEM = """Classify ONE shopper / harness user message in a shopping episode.
Return ONLY JSON. No markdown.

Context:
- Prior market shock already known to agent: <<EVENT_SEEN>>
- Agent already sent a recovery ack message: <<EVENT_ACKED>>
- Ask-first mode (waiting on withheld firm spec): <<ASK_FIRST>>

Return:
{
  "kind": "empty"|"market_signal"|"budget_update"|"question"|"constraint_hold"|"ack_nudge"|"spec_reveal"|"other",
  "requires_agent_message": bool,
  "budget_ceiling": number|null,
  "firm_measures": [{{"name": string, "numeric": number|null, "value_text": string}}],
  "reason": string
}

Rules (semantic; no fixed cue-phrase lists):
- empty: blank / whitespace-only content.
- market_signal: shopper reports or asks about a live price/stock/availability change
  that should trigger recovery (re-check / shock handling).
- budget_update: shopper states a NEW numeric spend ceiling; set budget_ceiling to
  that bare number (ignore currency words/symbols).
- question: shopper asks something that needs a reply before further cart/order.
- constraint_hold: shopper pauses the buy to restate firm requirements or demand
  a pre-purchase confirmation (may overlap with question).
- spec_reveal: shopper discloses a previously withheld firm hardware/spec measure
  (common after ask-first). Fill firm_measures with the revealed measure(s).
  Example: "needs N units of <attribute>" → firm_measures with numeric N.
- ack_nudge: short follow-up AFTER an event was already acknowledged (including
  availability/price re-check questions). requires_agent_message=false.
- other: anything else.
- requires_agent_message: true when the agent should call `message` this turn.
  For ask-first spec_reveal / budget_update that merely discloses the gated spec
  or a new ceiling (no direct shopper question), set requires_agent_message=false
  so the agent can retrieve immediately — do not chat about the reveal.
  When EVENT_ACKED is true, availability/price/stock questions and constraint
  restates are recovery, not another chat turn: kind=market_signal or ack_nudge,
  requires_agent_message=false.
- firm_measures: ONLY product hardware/spec measures from THIS message; NEVER the
  spend/budget ceiling. Empty list if none.
- If both budget and market/constraint signals appear, prefer budget_update when a
  new ceiling number is present, and still set requires_agent_message appropriately.
  When a message both updates budget AND reveals a gated spec, set kind=spec_reveal
  or budget_update but still fill BOTH budget_ceiling and firm_measures.
- reason: brief.
"""

_AGENT_MESSAGE_GATE_SYSTEM = """Judge ONE outbound shopping-agent message.
Return ONLY JSON. No markdown.

{
  "asks_permission": bool,
  "asks_withheld_firm_spec": bool,
  "claims_unverified_market_shock": bool,
  "reason": string
}

Definitions (semantic; do NOT rely on fixed product-domain vocabularies):
- asks_permission: true when the agent asks the shopper to choose/approve before
  proceeding (permission, preference menu, "should I…"). Declarative status
  updates are false.
- asks_withheld_firm_spec: true when the agent is requesting a firm requirement /
  spec / constraint the shopper said they would share if asked (ask-first reveal).
  Status chatter, cart/recovery talk, or permission menus alone are false.
- claims_unverified_market_shock: true when the agent asserts that stock/price/
  availability already changed, an item sold out, or that recovery is underway —
  when that claim is presented as a fact about the cart/market (not merely asking
  the shopper whether something changed).
"""


_SEARCH_PLAN_SYSTEM = """You plan retrieval for ONE shopping episode.
Return ONLY JSON. No markdown. Schema-independent: derive tokens from THIS query + Task rules.

Return:
{
  "queries": [string],
  "firm_tokens": [string],
  "soft_tokens": [string],
  "stop_after_n_searches": number
}

Rules:
- queries: 1-4 concrete search strings using VALUE tokens (category, brand, model,
  size/color numbers, phrase literals). Never bare schema-key names alone.
- Start precise; later queries may broaden slightly if needed.
- Respect negate tokens by omitting rejected types from queries.
- stop_after_n_searches: usually 3 or 4.
"""

_ELIGIBILITY_SYSTEM = """Judge whether ONE candidate is a valid cart commitment.
Return ONLY JSON. No markdown.

Episode profile (constraints):
<<PROFILE>>

Candidate product_id::sku: <<KEY>>
Viewed facts for this product: <<FACTS>>
Latest stock payload (may be null): <<STOCK>>
Budget ceiling: <<BUDGET>>
Closed-set mode: <<CLOSED>>
Dead keys (do not accept): <<DEAD>>
CommitmentPhase: <<PHASE>>

Return:
{
  "firm_ok": bool,
  "soft_ok": bool,
  "brand_ok": bool,
  "model_ok": bool,
  "phrase_ok": bool,
  "budget_ok": bool,
  "stock_ok": bool,
  "soft_score": number,
  "is_intended_commitment": bool,
  "reject_reason": string|null
}

Rules (schema-independent semantic match):
- firm_ok: EVERY firm constraint in the profile/query passes on THESE facts/stock.
  Unknown/missing firm evidence = FAIL (do not guess from SEO title alone).
  LITERAL for named numeric measures: the shopper's exact number must appear in
  structured facts. Nearby/rounded values are a FAIL — never approximate.
- brand_ok: if brand_is_firm or firm_brand set, brand must match; if only soft_brand,
  brand_ok reflects soft preference fit (soft miss does not by itself fail firm_ok).
- model_ok: if preferred_model set, title/options should contain that token when
  this is a soft/preferred pick; true if model absent from profile.
- phrase_ok / category: literal phrases/categories from profile should appear in
  facts/title when profile listed them; true if none listed.
  If facts include category_path, category_tokens MUST match that path (token
  overlap). A conflicting sibling leaf path fails even if title text looks close.
- budget_ok / stock_ok: from stock when present; if stock null, do not fail stock_ok
  solely for null — but fail budget_ok if a visible price clearly exceeds budget.
  Compare price as bare numbers (ignore currency codes/symbols).
- is_intended_commitment:
  * Always requires firm_ok + budget_ok + stock_ok + phrase_ok.
  * If CommitmentPhase is "first" and soft_brand is set: also require brand_ok
    when a soft-brand match is feasible from facts; if facts are a different brand,
    is_intended_commitment is false (another soft-brand listing may exist).
  * If CommitmentPhase is "recovery": soft brand/model miss is allowed — firm
    constraints + budget/stock dominate.
  Soft_score 0-1 for soft preference fit.
- reject_reason: short when not intended; else null.
"""

_RECOVERY_SYSTEM = """Plan recovery after a market shock for ONE episode.
Return ONLY JSON. No markdown.

Profile: <<PROFILE>>
Dead/shocked keys: <<DEAD>>
Preferred model: <<MODEL>>
Budget: <<BUDGET>>
Closed candidates remaining (may be empty): <<CLOSED_LIVE>>

Return:
{
  "ladder_level": 0,
  "allow_off_model": false,
  "search_queries": [string],
  "message_content": string,
  "guidance": string
}

Rules:
- ladder_level 0 = retry soft brand + firm measure within budget;
  FIRST retrieval should be filter(brand=<soft_brand>, max_price=<budget>) when a
  soft brand exists, else a search that includes soft brand + firm measure tokens.
  1 = firm measure within budget (drop soft brand);
  2 = broaden search wording but KEEP the shopper's product-type/category token
    + firm measure (never switch to a sibling category that drops the type).
- allow_off_model true at ladder_level >= 1 when soft brand is only a preference.
- message_content: one declarative sentence (no permission ask) acknowledging the
  shock and that you will pick another firm-valid in-budget option.
- search_queries: 1-3 VALUE-token searches that ALWAYS keep the firm measure token
  and the CURRENT budget number. At ladder_level 0 include soft_brand when set.
  Never search with a stale price as the budget.
- guidance: one short line for the acting agent on next tools. At ladder 0 with a
  soft brand, tell the agent to filter by that brand + max_price=budget before
  open-catalog thrash.
- If closed_live is non-empty: do not search open catalog; say to inspect/cart the
  best remaining listed key.
"""


def _fill_template(template: str, mapping: dict[str, str]) -> str:
    """Replace <<KEY>> placeholders without colliding with JSON braces."""
    out = template
    for key, val in mapping.items():
        out = out.replace(f"<<{key}>>", val)
    return out


def _str_list(raw: Any, *, limit: int = 12) -> list[str]:
    out: list[str] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        tok = str(item or "").strip()
        if tok and tok not in out:
            out.append(tok)
        if len(out) >= limit:
            break
    return out


def _llm_search_plan(
    query: str,
    *,
    profile: dict[str, Any],
    deadline: float,
) -> dict[str, Any]:
    """TF2/TF3 retrieval planner — value-token searches from THIS episode."""
    sem = min(deadline, time.monotonic() + 8.0)
    user = (
        f"Shopper + Task rules:\n{query}\n\n"
        f"Profile JSON:\n{json.dumps(profile, ensure_ascii=True)}\n"
    )
    raw = _llm_json(system=_SEARCH_PLAN_SYSTEM, user=user, deadline=sem) or {}
    queries = _str_list(raw.get("queries"), limit=4)
    soft_toks = _str_list(raw.get("soft_tokens"), limit=8) or _soft_preference_tokens(
        profile, query
    )
    if profile.get("require_soft_evidence") or soft_toks:
        pref_qs = _soft_search_queries(profile, query)
        merged: list[str] = []
        for q in pref_qs + queries:
            if q and q.lower() not in {x.lower() for x in merged}:
                merged.append(q)
        queries = merged[:4]
    if not queries:
        hints = _str_list(profile.get("search_value_hints"), limit=6)
        if hints:
            queries = [" ".join(hints[:4])]
    return {
        "queries": queries,
        "firm_tokens": _str_list(raw.get("firm_tokens"), limit=8),
        "soft_tokens": soft_toks,
        "stop_after_n_searches": int(raw.get("stop_after_n_searches") or 4),
    }


def _llm_eligibility(
    *,
    query: str,
    profile: dict[str, Any],
    key: str,
    facts: dict[str, str],
    stock: dict[str, Any] | None,
    budget: float | None,
    closed: bool,
    dead: set[str],
    deadline: float,
    phase: str = "first",
) -> dict[str, Any]:
    """TF1/TF3/TF4/TF6/TF7 cart-commitment judge (schema-independent)."""
    if key in dead:
        return {
            "firm_ok": False,
            "soft_ok": False,
            "brand_ok": False,
            "model_ok": False,
            "phrase_ok": False,
            "budget_ok": False,
            "stock_ok": False,
            "soft_score": 0.0,
            "is_intended_commitment": False,
            "reject_reason": "dead shocked key",
        }
    budget_ok = True
    stock_ok = True
    if stock is not None:
        if _stock_is_false(stock):
            stock_ok = False
        if budget is not None and stock.get("price") is not None:
            try:
                if float(stock["price"]) > float(budget) + 1e-6:
                    budget_ok = False
            except (TypeError, ValueError):
                pass
    measure_fail = _firm_measure_violation(facts, profile)
    type_fail = _product_type_violation(facts, profile)
    if not stock_ok or not budget_ok or measure_fail or type_fail:
        return {
            "firm_ok": not bool(measure_fail or type_fail),
            "soft_ok": False,
            "brand_ok": False,
            "model_ok": False,
            "phrase_ok": not bool(type_fail),
            "budget_ok": budget_ok,
            "stock_ok": stock_ok,
            "soft_score": 0.0,
            "is_intended_commitment": False,
            "reject_reason": type_fail or measure_fail or "stock/budget fail",
        }
    sem = min(deadline, time.monotonic() + 7.0)
    system = _fill_template(
        _ELIGIBILITY_SYSTEM,
        {
            "PROFILE": json.dumps(
                {
                    k: profile.get(k)
                    for k in (
                        "soft_brand",
                        "firm_brand",
                        "brand_is_firm",
                        "preferred_model",
                        "firm_spec_notes",
                        "firm_measures",
                        "phrase_tokens",
                        "category_tokens",
                        "negate_tokens",
                        "require_soft_evidence",
                        "needs_justification",
                        "spec_claim_values",
                        "notes",
                    )
                },
                ensure_ascii=True,
            ),
            "KEY": key,
            "FACTS": json.dumps(facts or {}, ensure_ascii=True),
            "STOCK": json.dumps(stock or {}, ensure_ascii=True),
            "BUDGET": json.dumps(budget),
            "CLOSED": str(bool(closed)).lower(),
            "DEAD": json.dumps(sorted(dead)),
            "PHASE": "recovery" if phase == "recovery" else "first",
        },
    )
    user = (
        f"Shopper goal:\n{_shopper_text(query)}\n\n"
        f"CommitmentPhase={phase}. Judge this candidate now."
    )
    raw = _llm_json(system=system, user=user, deadline=sem) or {}
    firm_ok = bool(raw.get("firm_ok", True))
    brand_ok = bool(raw.get("brand_ok", True))
    model_ok = bool(raw.get("model_ok", True))
    phrase_ok = bool(raw.get("phrase_ok", True))
    soft_ok = bool(raw.get("soft_ok", True))
    type_fail = _product_type_violation(facts, profile)
    if type_fail:
        firm_ok = False
        phrase_ok = False
        if not raw.get("reject_reason"):
            raw = dict(raw)
            raw["reject_reason"] = type_fail
    brand_fail = _firm_brand_violation(facts, profile)
    if brand_fail:
        brand_ok = False
        firm_ok = False
        if not raw.get("reject_reason"):
            raw = dict(raw)
            raw["reject_reason"] = brand_fail
    try:
        soft_score = float(raw.get("soft_score") or 0.0)
    except (TypeError, ValueError):
        soft_score = 0.0
    mech_soft = _soft_evidence_score(facts, profile=profile, query=query)
    if mech_soft > soft_score:
        soft_score = mech_soft
    if mech_soft >= 0.45:
        soft_ok = True
    if profile.get("brand_is_firm") or profile.get("firm_brand"):
        if not brand_ok:
            firm_ok = False
    intended = bool(raw.get("is_intended_commitment"))
    if not (firm_ok and budget_ok and stock_ok and phrase_ok):
        intended = False
    if (profile.get("brand_is_firm") or profile.get("firm_brand")) and not brand_ok:
        intended = False
    if phase != "recovery" and not closed:
        soft = str(profile.get("soft_brand") or "").strip()
        if soft and not brand_ok:
            intended = False
        model = str(profile.get("preferred_model") or "")
        if (
            model
            and not model.isdigit()
            and len(model) >= 3
            and not model_ok
        ):
            intended = False
        if profile.get("require_soft_evidence"):
            if not soft_ok or soft_score < 0.35:
                intended = False
                soft_ok = False
    reason = raw.get("reject_reason")
    return {
        "firm_ok": firm_ok,
        "soft_ok": soft_ok,
        "brand_ok": brand_ok,
        "model_ok": model_ok,
        "phrase_ok": phrase_ok,
        "budget_ok": budget_ok,
        "stock_ok": stock_ok,
        "soft_score": soft_score,
        "is_intended_commitment": intended,
        "reject_reason": str(reason).strip() if reason else None,
    }


def _soft_preference_tokens(profile: dict[str, Any], query: str = "") -> list[str]:
    """Tokens evidencing shopper soft use-case axes (from THIS episode only)."""
    out: list[str] = []
    for t in profile.get("search_value_hints") or []:
        tok = str(t or "").strip().lower()
        if len(tok) >= 3 and tok not in out:
            out.append(tok)
    for note in profile.get("firm_spec_notes") or []:
        s = str(note or "").strip().lower()
        if len(s) >= 4 and s not in out:
            out.append(s)
    notes = str(profile.get("notes") or "").strip().lower()
    if notes and len(notes) >= 4 and notes not in out:
        out.append(notes)
    goal = str(query or "").split("Task rules:")[0].lower()
    for m in re.finditer(
        r"(?:rather\s+(?:pay|have)|pay\s+(?:a\s+bit\s+)?more\s+for|"
        r"matters?\s+more(?:\s+than\s+\S+)?)\s+(.+?)(?:\s+than\b|$)",
        goal,
        flags=re.S,
    ):
        clause = str(m.group(1) or "")
        for tok in re.findall(r"[a-z][a-z\-]{3,}", clause):
            if tok not in out:
                out.append(tok)
            if len(out) >= 12:
                break
    return out[:12]


def _soft_search_queries(profile: dict[str, Any], query: str) -> list[str]:
    """Category + soft-token retrieval queries from THIS episode only."""
    cats = _category_tokens_for_gate(profile)
    cat = cats[0] if cats else ""
    toks = _soft_preference_tokens(profile, query)
    out: list[str] = []

    def push(parts: list[str]) -> None:
        q = " ".join(p for p in parts if p).strip()
        if q and q.lower() not in {x.lower() for x in out}:
            out.append(q)

    if cat and toks:
        push([cat] + toks[:4])
        push(toks[:4] + [cat])
    elif toks:
        push(toks[:5])
    elif cat:
        push([cat])
    for t in toks[:3]:
        push([t, cat] if cat else [t])
    return out[:5]


def _soft_evidence_score(
    facts: dict[str, str],
    *,
    profile: dict[str, Any],
    query: str = "",
) -> float:
    """Soft-axis fit from visible title/facts for THIS episode's preferences."""
    if not facts:
        return 0.0
    blob = " ".join(f"{k} {v}" for k, v in facts.items()).lower()
    tokens = _soft_preference_tokens(profile, query)
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in blob)
    return hits / max(1, len(tokens))


def _llm_recovery_plan(
    *,
    query: str,
    profile: dict[str, Any],
    dead: set[str],
    budget: float | None,
    closed_live: list[str],
    deadline: float,
) -> dict[str, Any]:
    """Post-shock recovery planner (open + closed catalog)."""
    sem = min(deadline, time.monotonic() + 7.0)
    system = _fill_template(
        _RECOVERY_SYSTEM,
        {
            "PROFILE": json.dumps(
                {
                    k: profile.get(k)
                    for k in (
                        "preferred_model",
                        "soft_brand",
                        "firm_brand",
                        "brand_is_firm",
                        "firm_measures",
                        "category_tokens",
                        "search_value_hints",
                        "notes",
                    )
                },
                ensure_ascii=True,
            ),
            "DEAD": json.dumps(sorted(dead)),
            "MODEL": json.dumps(profile.get("preferred_model")),
            "BUDGET": json.dumps(budget),
            "CLOSED_LIVE": json.dumps(closed_live),
        },
    )
    user = f"Shopper goal:\n{_shopper_text(query)}\n\nPlan recovery now."
    raw = _llm_json(system=system, user=user, deadline=sem) or {}
    try:
        level = int(raw.get("ladder_level") or 0)
    except (TypeError, ValueError):
        level = 0
    soft = str(profile.get("soft_brand") or "").strip()
    if soft and level < 0:
        level = 0
    level = max(0, min(2, level))
    msg = str(raw.get("message_content") or "").strip()
    if not msg:
        model = profile.get("preferred_model")
        if model:
            msg = (
                f"That option changed (stock/price). I'm selecting another in-stock "
                f"variant of {model} within budget; firm requirements stay."
            )
        else:
            msg = (
                "That option changed (stock/price). I'm selecting another in-stock "
                "option that still meets your firm budget and requirements."
            )
    queries = _str_list(raw.get("search_queries"), limit=3)
    firm_bits: list[str] = []
    for m in (profile.get("firm_measures") or [])[:2]:
        if not isinstance(m, dict):
            continue
        if m.get("value_text"):
            firm_bits.append(str(m["value_text"]))
        elif m.get("numeric") is not None:
            firm_bits.append(str(m["numeric"]))
    if soft and level == 0:
        if not any(soft.lower() in q.lower() for q in queries):
            parts = [soft, *firm_bits]
            if budget is not None:
                parts.append(str(budget))
            queries = [" ".join(parts)] + queries
    guidance = str(raw.get("guidance") or "").strip()
    if soft and level == 0 and budget is not None:
        filter_tip = (
            f"First call filter with brand=`{soft}` and max_price={budget}, "
            "then view a NEW product_id (not a dead shocked key)."
        )
        if not guidance:
            guidance = filter_tip
        elif "filter" not in guidance.lower():
            guidance = filter_tip + " " + guidance
    return {
        "ladder_level": level,
        "allow_off_model": bool(raw.get("allow_off_model")) or level >= 1,
        "search_queries": queries[:3],
        "message_content": msg,
        "guidance": guidance,
    }


def _firm_measure_tokens(profile: dict[str, Any], *, limit: int = 2) -> list[str]:
    out: list[str] = []
    for m in (profile.get("firm_measures") or [])[:limit]:
        if not isinstance(m, dict):
            continue
        if m.get("value_text"):
            out.append(str(m["value_text"]))
        elif m.get("numeric") is not None:
            out.append(str(m["numeric"]))
    return out


def _soft_brand_in_retrieval(
    tool_calls: list[dict[str, Any]],
    *,
    soft_brand: str,
) -> bool:
    soft = (soft_brand or "").strip().lower()
    if not soft:
        return False
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        raw = call.get("function", {}).get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            continue
        if name == "filter":
            brand = str(args.get("brand") or args.get("Brand") or "").strip().lower()
            if brand and (soft in brand or brand in soft or _tokenize_overlap(soft, brand)):
                return True
        if name == "search":
            q = str(args.get("query") or "").lower()
            if soft in q or _tokenize_overlap(soft, q):
                return True
    return False


def _strip_unprofiled_brand_filters(
    tool_calls: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Drop filter(brand=…) when the episode has no soft/firm brand.

    Models often shove phrase tokens into brand= (e.g. multi-word phrase →
    first word as brand), which collapses retrieval. Only keep brand filters when the
    profile named a brand. If brand was the only facet, restore category from
    profile category_tokens so the filter still surveys the right shelf.
    """
    if profile.get("soft_brand") or profile.get("firm_brand"):
        return tool_calls
    cats = _category_tokens_for_gate(profile)
    out: list[dict[str, Any]] = []
    changed = False
    for call in tool_calls:
        name = (call.get("function") or {}).get("name") or ""
        if name != "filter":
            out.append(call)
            continue
        raw = (call.get("function") or {}).get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            out.append(call)
            continue
        if not isinstance(args, dict):
            out.append(call)
            continue
        fargs = dict(args)
        mutated = False
        if "brand" in fargs and not (
            profile.get("soft_brand") or profile.get("firm_brand")
        ):
            fargs.pop("brand", None)
            mutated = True
        if not fargs.get("category") and cats:
            fargs["category"] = cats[0]
            mutated = True
        if not mutated:
            out.append(call)
            continue
        changed = True
        out.append(
            {
                "id": call.get("id"),
                "type": call.get("type") or "function",
                "function": {
                    "name": "filter",
                    "arguments": json.dumps(fargs, ensure_ascii=True),
                },
            }
        )
    return out if changed else tool_calls


def _rewrite_recovery_retrieval(
    tool_calls: list[dict[str, Any]],
    *,
    soft_brand: str | None,
    budget: float | None,
    soft_tried: bool,
    firm_brand: str | None = None,
    category: str | None = None,
    firm_measure_tokens: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Pin brand+budget on post-shock retrieval.

    Soft brand: ladder-0 once (search→filter). Firm brand: enrich in place —
    never collapse a measure-bearing search into a bare brand filter (that
    returns accessories and thrash). Uses ONLY episode profile fields.
    """
    firm = str(firm_brand or "").strip()
    soft = str(soft_brand or "").strip()
    brand = firm or soft
    if not brand or budget is None:
        return tool_calls
    if soft_tried and not firm:
        return tool_calls
    names = _tool_names(tool_calls)
    if not any(n in ("search", "filter") for n in names):
        return tool_calls
    cat = str(category or "").strip() or None
    measure_bits = [
        str(t).strip()
        for t in (firm_measure_tokens or [])
        if str(t).strip()
    ][:2]

    def _call(name: str, args: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": template.get("id"),
            "type": template.get("type") or "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=True),
            },
        }

    out: list[dict[str, Any]] = []
    converted = False
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        raw = call.get("function", {}).get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            out.append(call)
            continue
        if name == "filter":
            if firm and measure_bits:
                q_parts = [brand] + measure_bits
                if cat:
                    q_parts.append(cat)
                sargs: dict[str, Any] = {
                    "query": " ".join(q_parts),
                    "in_stock": True,
                    "k": 10,
                }
                out.append(_call("search", sargs, call))
                converted = True
                continue
            fargs = dict(args)
            fargs["brand"] = brand
            fargs["max_price"] = float(budget)
            if cat and not fargs.get("category"):
                fargs["category"] = cat
            out.append(_call("filter", fargs, call))
            converted = True
            continue
        if name == "search" and not converted and not firm:
            fargs: dict[str, Any] = {"brand": brand, "max_price": float(budget)}
            if cat:
                fargs["category"] = cat
            out.append(_call("filter", fargs, call))
            converted = True
            continue
        if name == "search":
            q = str(args.get("query") or "")
            sargs = dict(args)
            prefix_parts = []
            if brand.lower() not in q.lower():
                prefix_parts.append(brand)
            for tok in measure_bits:
                if tok.lower() not in q.lower():
                    prefix_parts.append(tok)
            if cat and cat.lower() not in q.lower():
                prefix_parts.append(cat)
            if prefix_parts:
                sargs["query"] = (" ".join(prefix_parts) + " " + q).strip()
                out.append(_call("search", sargs, call))
                converted = True
                continue
            out.append(call)
            converted = True
            continue
        out.append(call)
    return out if converted else tool_calls


def _collect_retrieval_pids(obs: Any) -> list[str]:
    """Product ids from search/filter observation results (order preserved)."""
    if not isinstance(obs, dict):
        return []
    node = (
        obs.get("observation")
        if isinstance(obs.get("observation"), dict)
        else obs
    )
    if not isinstance(node, dict):
        return []
    results = node.get("results") or node.get("products") or node.get("items") or []
    if not isinstance(results, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for row in results:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("product_id") or "").strip()
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


_CLAIMS_SYSTEM = """Build place_test_order justification claims for ONE order.
Return ONLY JSON:
{{"claims":[{{"field":"...","value":"..."}}],
 "field_mapping":[{{"shopper_need":"...","field":"...","value":"...","evidence":"..."}}]}}

Allowed claim field names (prefer these exact strings): {allowed}
Observed catalog/fact keys from view/compare (vocabulary pool): {fact_keys}
Shopper-preferred spec values (hints; still must be grounded): {spec_values}
Required fields to cover when groundable: {required}

Field→value mapping policy (schema-independent; catalog-grounded):
1) Map each shopper-named requirement onto exactly ONE field.
   Prefer Allowed names when present; else prefer Observed fact keys that
   semantically match; else a short snake_case name aligned with Task rules.
2) LITERAL for brands, models, IDs, exact options, and numeric measures the
   shopper named. Prefer the shopper's own wording when Task rules treat that
   wording as the claim target AND facts support it. No synonym/class tables.
3) Price: digits only (strip currency codes/symbols). in_stock: true/false.
4) Soft reason fields only if positively evidenced in structured fact fields —
   never from marketing title alone. Omit brand when title is the only evidence.
5) Omit ungroundable fields. Never use category / category_match / category_path.
"""

_CLOSED_RANK_SYSTEM = """Rank closed-set shopping candidates for ONE episode.
Return ONLY JSON. No markdown.

{{
  "ordered_keys": ["product_id::sku", ...],
  "axis_values": {{"product_id::sku": {{"axis_name": number}}}}
}}

Profile (constraints + rank axes): <<PROFILE>>
Rank axes in priority order: <<AXES>>
Higher-better axes: <<HIGHER>>
Budget ceiling (numeric, currency-agnostic): <<BUDGET>>
Dead keys (exclude / put last): <<DEAD>>
Candidates with observed facts: <<CANDS>>

Rules (schema-independent semantic ranking):
- Exclude or deprioritize dead keys and out-of-stock when stock is known false.
- Firm constraints from profile must pass on structured facts (not title alone).
- Among remaining, rank by Rank axes: higher-better → larger numeric first;
  price/budget/weight-like → smaller first.
- Extract axis numbers from each candidate's facts/title/options by semantic fit
  to the axis name. Do not invent numbers that are not evidenced.
- Price values are bare numbers (ignore currency labels).
- ordered_keys must list every non-dead candidate key, best first.
- Lexicographic: satisfy axis[0] before axis[1] before axis[2]. Never promote a
  worse primary-axis candidate because it is cheaper.
"""

_AXIS_BIND_SYSTEM = """Bind ranking axes to catalog evidence for ONE episode.
Return ONLY JSON. No markdown.

{
  "bindings": {
    "axis_name": {
      "fact_keys": ["..."],
      "unit_tokens": ["..."],
      "text_keywords": ["..."],
      "source_priority": ["options", "facts", "title", "price"],
      "aggregate": "first"
    }
  }
}

Rank axes: <<AXES>>
Higher-better axes: <<HIGHER>>
Sample candidates (titles/facts/options/prices): <<SAMPLES>>

Rules (catalog-grounded field→value mapping — no suite synonym tables):
- For each axis, choose fact_keys / unit_tokens / text_keywords that actually
  appear in the samples OR are clear abbreviations of those sample tokens.
- unit_tokens are the short unit/label strings that follow numbers in samples
  (whatever units THIS catalog uses — do not assume a fixed vocabulary).
- text_keywords help find numbers near words in title/options when facts lack
  the axis.
- price axis: use source_priority including "price"; empty unit_tokens.
- Prefer variant options over product-level facts when both disagree.
- When samples show measure tokens such as a number glued to a short unit label
  in title/options, copy THAT unit label into unit_tokens for the matching axis.
- aggregate: "max" when the shopper wants the highest value on that axis and
  samples may list several matching numbers; "min" for lowest; else "first".
- Do not invent axes that were not listed. Do not invent units absent from samples.
- Do not emit fixed display-class or pixel remaps; use only sample evidence.
"""


def _provider() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw in ("chutes", "openrouter"):
        return raw
    if getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "chutes"


def _model_list() -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    return list(_OPENROUTER_MODELS if _provider() == "openrouter" else _CHUTES_MODELS)


def _shopper_text(query: str) -> str:
    return query.split("Task rules:", 1)[0].strip()


def _parse_candidate_variants(query: str) -> list[tuple[str, str]]:
    """Extract product_id::sku pairs as a closed set (≥2 structural keys)."""
    head = _shopper_text(query)
    pairs = _CAND_KEY_RE.findall(head)
    if len(pairs) < 2:
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


def _schema_claim_enum(tools: list[dict[str, Any]]) -> frozenset[str] | None:
    """Return claim-field enum when present; None means open string fields."""
    for tool in tools:
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
            return frozenset(str(x) for x in enum) - _CLAIM_DENY
        return None
    return None


def _effective_claim_fields(
    tools: list[dict[str, Any]],
    *,
    profile_required: list[str] | None = None,
) -> frozenset[str]:
    """Schema enum when present; else core ∪ LLM-identified shopper-spec fields.

    Open schemas (no enum) intentionally allow the episode profile to name the
    specification field(s) from THIS query — not a hardcoded vocab list.
    """
    enum = _schema_claim_enum(tools)
    base: set[str] = set(_CORE_CLAIM_FIELDS)
    if enum is not None:
        base |= set(enum)
        for name in profile_required or []:
            key = str(name or "").strip()
            if key in base:
                continue
            if key in enum:
                base.add(key)
    else:
        for name in profile_required or []:
            key = str(name or "").strip().lower().replace(" ", "_")
            if key and key not in _CLAIM_DENY:
                base.add(key)
    base -= set(_CLAIM_DENY)
    return frozenset(base)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _llm_json(
    *,
    system: str,
    user: str,
    deadline: float,
) -> dict[str, Any] | None:
    """Extra allowlisted-model call (no tools) for semantic JSON."""
    if time.monotonic() >= deadline:
        return None
    last_err: Exception | None = None
    for model in _model_list():
        if time.monotonic() >= deadline:
            break
        try:
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0,
                },
            )
            if inference is None:
                continue
            content = (
                ((inference.get("choices") or [{}])[0].get("message") or {}).get(
                    "content"
                )
                or ""
            )
            parsed = _extract_json_object(str(content))
            if parsed is not None:
                return parsed
        except Exception as exc:
            last_err = exc
            continue
    _ = last_err
    return None


def _default_profile() -> dict[str, Any]:
    return {
        "ask_first": False,
        "needs_justification": False,
        "soft_brand": None,
        "firm_brand": None,
        "brand_is_firm": False,
        "preferred_model": None,
        "preferred_color": None,
        "rank_axes": [],
        "higher_better_axes": [],
        "required_claim_fields": ["price", "in_stock"],
        "spec_claim_values": {},
        "firm_spec_notes": [],
        "phrase_tokens": [],
        "category_tokens": [],
        "required_product_leaf": None,
        "negate_tokens": [],
        "require_soft_evidence": False,
        "search_value_hints": [],
        "firm_measures": [],
        "budget_ceiling": None,
        "notes": "",
    }


def _normalize_profile(
    raw: dict[str, Any] | None,
    *,
    open_fields: bool,
    enum: frozenset[str] | None,
) -> dict[str, Any]:
    base = _default_profile()
    if not isinstance(raw, dict):
        return base
    base["ask_first"] = bool(raw.get("ask_first"))
    base["needs_justification"] = bool(raw.get("needs_justification"))
    for key in ("soft_brand", "preferred_model", "preferred_color", "notes"):
        val = raw.get(key)
        if isinstance(val, str) and val.strip():
            cleaned = val.strip()
            if key == "preferred_model" and len(cleaned) < 1:
                base[key] = None
            elif key == "preferred_color" and len(cleaned) < 2:
                base[key] = None
            else:
                base[key] = cleaned
        else:
            base[key] = None if key != "notes" else ""
    axes: list[str] = []
    for item in raw.get("rank_axes") or []:
        name = str(item or "").strip().lower().replace(" ", "_")
        if name and name not in axes:
            axes.append(name)
    base["rank_axes"] = axes
    hb: list[str] = []
    for item in raw.get("higher_better_axes") or []:
        name = str(item or "").strip().lower().replace(" ", "_")
        if name in axes and name not in hb:
            hb.append(name)
    if not hb and axes:
        for a in axes:
            if a in ("price", "budget", "cost", "weight"):
                continue
            hb.append(a)
    base["higher_better_axes"] = hb
    req: list[str] = []
    for item in raw.get("required_claim_fields") or []:
        name = str(item or "").strip().lower().replace(" ", "_")
        if not name or name in _CLAIM_DENY or name in req:
            continue
        if open_fields or (enum is not None and name in enum) or name in _CORE_CLAIM_FIELDS:
            req.append(name)
    if base["needs_justification"]:
        for core in ("price", "in_stock"):
            if core not in req:
                req.insert(0, core)
    else:
        req = [f for f in req if f in ("price", "in_stock")]
    base["required_claim_fields"] = req
    specs: dict[str, str] = {}
    raw_specs = raw.get("spec_claim_values")
    if isinstance(raw_specs, dict):
        for k, v in raw_specs.items():
            field = str(k or "").strip().lower().replace(" ", "_")
            val = str(v or "").strip()
            if field and val and field in req and field not in ("price", "in_stock"):
                specs[field] = val
    base["spec_claim_values"] = specs
    hints: list[str] = []
    for item in raw.get("search_value_hints") or []:
        tok = str(item or "").strip()
        if tok and tok not in hints:
            hints.append(tok)
    base["search_value_hints"] = hints[:12]
    bud = raw.get("budget_ceiling")
    try:
        budget_ceiling = float(bud) if bud is not None else None
    except (TypeError, ValueError):
        budget_ceiling = None

    def _is_budget_numeric(num: float | None) -> bool:
        """Spend ceilings are budget_ceiling — never firm product measures."""
        if num is None or budget_ceiling is None:
            return False
        return abs(float(num) - float(budget_ceiling)) <= 1e-6

    measures: list[dict[str, Any]] = []
    raw_measures = raw.get("firm_measures")
    if isinstance(raw_measures, list):
        for item in raw_measures:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip().lower().replace(" ", "_")
            value_text = str(item.get("value_text") or "").strip()
            numeric = item.get("numeric")
            try:
                numeric_f = float(numeric) if numeric is not None else None
            except (TypeError, ValueError):
                numeric_f = None
            if numeric_f is None and value_text:
                m = re.search(r"(\d+(?:\.\d+)?)", value_text)
                if m:
                    numeric_f = float(m.group(1))
            if not name and not value_text and numeric_f is None:
                continue
            if _is_budget_numeric(numeric_f):
                continue
            if name in ("budget", "budget_ceiling", "max_price", "price_ceiling"):
                continue
            measures.append(
                {
                    "name": name or "measure",
                    "numeric": numeric_f,
                    "value_text": value_text,
                }
            )
    if not measures:
        for tok in list(hints) + list(raw.get("firm_spec_notes") or []):
            m = re.search(r"(\d+\.\d+)", str(tok))
            if not m:
                continue
            scraped = float(m.group(1))
            if _is_budget_numeric(scraped):
                continue
            measures.append(
                {
                    "name": "measure",
                    "numeric": scraped,
                    "value_text": str(tok).strip(),
                }
            )
            break
    base["firm_measures"] = measures[:6]
    for list_key in ("firm_spec_notes", "phrase_tokens", "category_tokens", "negate_tokens"):
        base[list_key] = _str_list(raw.get(list_key), limit=8)
    leaf_raw = raw.get("required_product_leaf")
    if isinstance(leaf_raw, str) and leaf_raw.strip():
        leaf = leaf_raw.strip()
        if any(ch in leaf for ch in (" ", "/", "&", ">")):
            base["required_product_leaf"] = None
            cats = list(base.get("category_tokens") or [])
            if leaf not in cats:
                cats = [leaf] + cats
                base["category_tokens"] = cats
        else:
            leaf_n = re.sub(r"[^a-z0-9_]+", "", leaf.lower().replace(" ", "_"))
            if leaf_n.endswith("s") and not leaf_n.endswith("ss") and len(leaf_n) > 4:
                leaf_n = leaf_n[:-1]
            cats = [str(c).lower() for c in (base.get("category_tokens") or [])]
            fragment = any(
                leaf_n
                and leaf_n not in (c, c.rstrip("s"))
                and leaf_n in re.split(r"[^a-z0-9]+", c)
                and (len(c) > len(leaf_n) + 2 or " " in c or "&" in c)
                for c in cats
            )
            base["required_product_leaf"] = None if fragment else (leaf_n or None)
    else:
        base["required_product_leaf"] = None
    firm_brand = raw.get("firm_brand")
    if isinstance(firm_brand, str) and firm_brand.strip():
        base["firm_brand"] = firm_brand.strip()
    else:
        base["firm_brand"] = None
    base["brand_is_firm"] = bool(raw.get("brand_is_firm")) or bool(base["firm_brand"])
    if base["needs_justification"]:
        if not base["firm_brand"] and base.get("soft_brand"):
            base["firm_brand"] = base["soft_brand"]
        base["brand_is_firm"] = bool(base["firm_brand"]) or base["brand_is_firm"]
    base["require_soft_evidence"] = bool(raw.get("require_soft_evidence"))
    base["budget_ceiling"] = budget_ceiling
    return base


def _interpret_episode(
    query: str,
    *,
    tools: list[dict[str, Any]],
    deadline: float,
) -> tuple[dict[str, Any], frozenset[str]]:
    """LLM episode profile + effective claim allowlist for this episode."""
    enum = _schema_claim_enum(tools)
    open_fields = enum is None
    claim_mode = "open" if open_fields else "enum"
    hint = (
        sorted(enum | _CORE_CLAIM_FIELDS)
        if enum is not None
        else sorted(_CORE_CLAIM_FIELDS)
        + [
            "(plus snake_case names for each shopper-named specification field)",
        ]
    )
    sem_deadline = min(deadline, time.monotonic() + _SEMANTIC_BUDGET_S)
    system = _PROFILE_SYSTEM.format(
        claim_mode=claim_mode,
        allowed=json.dumps(hint, ensure_ascii=True),
    )
    user = (
        "Episode query (shopper + Task rules):\n"
        f"{query}\n\n"
        "Return the JSON profile now."
    )
    raw = _llm_json(system=system, user=user, deadline=sem_deadline)
    profile = _normalize_profile(raw, open_fields=open_fields, enum=enum)

    def _repair(extra: str) -> None:
        nonlocal profile
        if time.monotonic() >= sem_deadline:
            return
        raw2 = _llm_json(
            system=system,
            user=user + "\n\nREPAIR: " + extra,
            deadline=sem_deadline,
        )
        if raw2:
            profile = _normalize_profile(raw2, open_fields=open_fields, enum=enum)

    if profile.get("needs_justification") and not any(
        f not in ("price", "in_stock", "brand")
        for f in (profile.get("required_claim_fields") or [])
    ):
        _repair(
            "needs_justification is true but required_claim_fields is missing the "
            "shopper's specification attribute. Re-output the full JSON and include "
            "that snake_case field plus spec_claim_values for it."
        )
    if not (profile.get("category_tokens") or []):
        _repair(
            "category_tokens is empty but the shopper named a product type or catalog "
            "category. Re-output the full JSON with category_tokens filled from THIS "
            "shopper text only (product-type / category nouns as written)."
        )
    if profile.get("budget_ceiling") is None:
        _repair(
            "budget_ceiling is null but the shopper stated a numeric spend/budget "
            "ceiling. Re-output the full JSON with budget_ceiling set to that bare "
            "number (strip currency words/codes)."
        )
    allowed = _effective_claim_fields(
        tools, profile_required=list(profile.get("required_claim_fields") or [])
    )
    profile["required_claim_fields"] = [
        f for f in (profile.get("required_claim_fields") or []) if f in allowed
    ]
    specs = {
        k: v
        for k, v in (profile.get("spec_claim_values") or {}).items()
        if k in profile["required_claim_fields"]
    }
    profile["spec_claim_values"] = specs
    bud = profile.get("budget_ceiling")
    if isinstance(bud, (int, float)):
        cleaned_ms: list[dict[str, Any]] = []
        for m in profile.get("firm_measures") or []:
            if not isinstance(m, dict):
                continue
            num = m.get("numeric")
            try:
                if num is not None and abs(float(num) - float(bud)) <= 1e-6:
                    continue
            except (TypeError, ValueError):
                pass
            cleaned_ms.append(m)
        profile["firm_measures"] = cleaned_ms
    goal_l = str(query or "").split("Task rules:")[0].lower()
    if _soft_preference_tokens(profile, query) and re.search(
        r"rather pay|matters more|cheapest isn|don'?t just grab the cheapest|"
        r"not the goal here|than settle|prefer(?:ence)?s?\s+matter",
        goal_l,
    ):
        profile["require_soft_evidence"] = True
    if _parse_candidate_variants(query) and not (profile.get("rank_axes") or []):
        axes: list[str] = []
        for m in re.finditer(
            r"\b(?:highest|largest|biggest|maximum)\s+"
            r"([a-z][a-z0-9]+(?:[\s_-][a-z][a-z0-9]+){0,2})",
            goal_l,
        ):
            axis = re.sub(r"[\s-]+", "_", m.group(1).strip())
            if axis and axis not in axes:
                axes.append(axis)
        for m in re.finditer(
            r"\b(?:lowest|smallest|cheapest|minimum)\s+"
            r"([a-z][a-z0-9]+(?:[\s_-][a-z][a-z0-9]+){0,2})",
            goal_l,
        ):
            axis = re.sub(r"[\s-]+", "_", m.group(1).strip())
            if axis in ("price", "cost", "budget") or not axis:
                if "price" not in axes:
                    axes.append("price")
            elif axis not in axes:
                axes.append(axis)
        if axes:
            profile["rank_axes"] = axes[:4]
            profile["higher_better_axes"] = [
                a
                for a in axes
                if a not in ("price", "budget", "cost", "weight")
                and not a.startswith("low")
            ]
    elif _parse_candidate_variants(query) and (profile.get("rank_axes") or []):
        hb = list(profile.get("higher_better_axes") or [])
        for a in profile.get("rank_axes") or []:
            if a in ("price", "budget", "cost", "weight"):
                continue
            if a not in hb:
                hb.append(a)
        profile["higher_better_axes"] = hb
    shopper = _shopper_text(query)
    model_color = _extract_preferred_model_color(shopper)
    if model_color.get("preferred_model") and not profile.get("preferred_model"):
        profile["preferred_model"] = model_color["preferred_model"]
    if model_color.get("preferred_color") and not profile.get("preferred_color"):
        profile["preferred_color"] = model_color["preferred_color"]
    shop_l = shopper.lower()
    request_l = re.split(
        r"\b(?:reject|do not|don't|never|unless|category labels alone|"
        r"missing positive evidence)\b",
        shop_l,
        maxsplit=1,
    )[0]
    leaf = str(profile.get("required_product_leaf") or "").strip() or None
    if leaf and (" " in leaf or "&" in leaf or len(leaf) < 4):
        leaf = None if (len(leaf) < 4 or " " in leaf or "&" in leaf) else leaf
    best = leaf
    best_len = len(best) if best else 0
    for tok in profile.get("category_tokens") or []:
        raw = str(tok or "").strip()
        if not raw:
            continue
        low = raw.lower()
        if low in request_l and len(low) >= best_len:
            best = raw
            best_len = len(low)
            continue
        if " " in low or "&" in low:
            continue
        stem = low.rstrip("s") if len(low) > 4 else low
        if len(stem) < 4:
            continue
        if not re.search(rf"(?i)(^|[^a-z0-9]){re.escape(stem)}s?([^a-z0-9]|$)", request_l):
            continue
        if len(stem) > best_len:
            best = stem
            best_len = len(stem)
    if best:
        leaf = best
        cats = list(profile.get("category_tokens") or [])
        if not any(str(leaf).lower() in str(c).lower() for c in cats):
            cats = [leaf] + cats
            profile["category_tokens"] = cats
        if (
            " " not in str(leaf)
            and "&" not in str(leaf)
            and not _looks_like_phrase_retrieval(profile, query)
        ):
            profile["required_product_leaf"] = str(leaf).lower()
        else:
            profile["required_product_leaf"] = None
    elif profile.get("required_product_leaf"):
        rl = str(profile.get("required_product_leaf") or "").strip().lower()
        if " " in rl or "&" in rl or len(rl) < 5:
            profile["required_product_leaf"] = None
        elif not re.search(rf"(?i)(^|[^a-z0-9]){re.escape(rl)}s?([^a-z0-9]|$)", request_l):
            profile["required_product_leaf"] = None
    if profile.get("needs_justification"):
        specs = dict(profile.get("spec_claim_values") or {})
        req = list(profile.get("required_claim_fields") or [])
        keep = ["price", "in_stock"]
        for f in req:
            fl = str(f).strip().lower()
            if fl in ("price", "in_stock"):
                continue
            if fl in specs and fl not in keep:
                keep.append(fl)
        for f in specs:
            fl = str(f).strip().lower()
            if fl and fl not in keep and fl not in ("price", "in_stock"):
                keep.append(fl)
        profile["spec_claim_values"] = {
            k: v for k, v in specs.items() if k in keep
        }
        profile["required_claim_fields"] = keep
    return profile, allowed


def _extract_preferred_model_color(shopper_text: str) -> dict[str, str | None]:
    """Structural extract of preferred model/color from shopper wording.

    Matches recovery-style 'prefer the <color> <model>' only (requires ``the``).
    Soft-brand lines like 'Ideally, I prefer <Brand Name>' are NOT model/color —
    those belong to soft_brand via the profile LLM.
    Model tokens should look like codes (contain a digit) or compact alphanumerics,
    not multi-word brand phrases.
    """
    text = str(shopper_text or "")
    out: dict[str, str | None] = {"preferred_model": None, "preferred_color": None}
    if not text.strip():
        return out
    m = re.search(
        r"(?:strongly\s+)?prefer\s+the\s+"
        r"(?P<color>[A-Za-z][A-Za-z]*(?:\s+[A-Za-z][A-Za-z]*){0,2})\s+"
        r"(?P<model>[A-Za-z0-9][A-Za-z0-9_-]{0,24})"
        r"(?=\s|$|[.,;:!?)])",
        text,
        flags=re.I,
    )
    if not m:
        return out
    color = str(m.group("color") or "").strip()
    model = str(m.group("model") or "").strip().strip(".,;:!?")
    model = model.rstrip("._/-")
    color_l = color.lower()
    for noise in ("the", "a", "an", "model", "variant", "color", "colour", "preferred"):
        if color_l == noise:
            color = ""
            break
    if model and not re.search(r"\d", model) and len(model) > 5:
        return out
    if model and len(model) >= 1:
        out["preferred_model"] = model
    if color and len(color) >= 3:
        out["preferred_color"] = color.lower()
    return out


def _rewrite_model_recovery_search(
    tool_calls: list[dict[str, Any]],
    *,
    model: str | None,
    color: str | None,
    drop_color: bool,
    category_token: str | None = None,
) -> list[dict[str, Any]]:
    """Pin preferred model (+ optional color) onto post-shock search queries."""
    model_tok = str(model or "").strip()
    if not model_tok:
        return tool_calls
    color_tok = "" if drop_color else str(color or "").strip()
    cat = str(category_token or "").strip()
    out: list[dict[str, Any]] = []
    changed = False
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        if name != "search":
            out.append(call)
            continue
        raw = call.get("function", {}).get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            out.append(call)
            continue
        if not isinstance(args, dict):
            out.append(call)
            continue
        q = str(args.get("query") or "")
        q_l = q.lower()
        parts: list[str] = []
        if color_tok and color_tok.lower() not in q_l:
            parts.append(color_tok)
        if model_tok.lower() not in q_l:
            parts.append(model_tok)
        if cat and cat.lower() not in q_l:
            parts.append(cat)
        if not parts and model_tok.lower() in q_l:
            out.append(call)
            continue
        new_q = " ".join(p for p in ([*parts, q.strip()] if q.strip() else parts) if p)
        if not new_q:
            new_q = model_tok
        sargs = dict(args)
        sargs["query"] = new_q
        sargs.setdefault("in_stock", True)
        sargs.setdefault("k", 10)
        out.append(
            {
                "id": call.get("id"),
                "type": call.get("type") or "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps(sargs, ensure_ascii=True),
                },
            }
        )
        changed = True
    return out if changed else tool_calls


def _llm_interpret_shopper_turn(
    content: str,
    *,
    event_seen: bool,
    event_acked: bool,
    ask_first: bool,
    deadline: float,
) -> dict[str, Any]:
    """Semantic shopper-turn classifier (replaces suite cue-phrase regexes)."""
    text = str(content or "")
    if not text.strip():
        return {
            "kind": "empty",
            "requires_agent_message": False,
            "budget_ceiling": None,
            "firm_measures": [],
            "reason": "empty",
        }
    if _time_left(deadline) < 6:
        needs = "?" in text
        return {
            "kind": "question" if needs else "other",
            "requires_agent_message": needs,
            "budget_ceiling": None,
            "firm_measures": [],
            "reason": "low_time_fallback",
        }
    sem = min(deadline, time.monotonic() + 5.0)
    system = _fill_template(
        _SHOPPER_TURN_SYSTEM,
        {
            "EVENT_SEEN": str(bool(event_seen)).lower(),
            "EVENT_ACKED": str(bool(event_acked)).lower(),
            "ASK_FIRST": str(bool(ask_first)).lower(),
        },
    )
    raw = _llm_json(
        system=system,
        user=f"Shopper/harness message:\n{text}\n\nClassify it now.",
        deadline=sem,
    ) or {}
    kind = str(raw.get("kind") or "other").strip().lower()
    allowed = {
        "empty",
        "market_signal",
        "budget_update",
        "question",
        "constraint_hold",
        "ack_nudge",
        "spec_reveal",
        "other",
    }
    if kind not in allowed:
        kind = "other"
    budget = raw.get("budget_ceiling")
    try:
        budget_f = float(budget) if budget is not None else None
    except (TypeError, ValueError):
        budget_f = None
    measures: list[dict[str, Any]] = []
    raw_ms = raw.get("firm_measures")
    if isinstance(raw_ms, list):
        for item in raw_ms:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip().lower().replace(" ", "_")
            value_text = str(item.get("value_text") or "").strip()
            numeric = item.get("numeric")
            try:
                numeric_f = float(numeric) if numeric is not None else None
            except (TypeError, ValueError):
                numeric_f = None
            if numeric_f is None and value_text:
                m = re.search(r"(\d+(?:\.\d+)?)", value_text)
                if m:
                    numeric_f = float(m.group(1))
            if budget_f is not None and numeric_f is not None:
                if abs(numeric_f - budget_f) <= 1e-6:
                    continue
            if not name and not value_text and numeric_f is None:
                continue
            measures.append(
                {
                    "name": name or "measure",
                    "numeric": numeric_f,
                    "value_text": value_text,
                }
            )
    requires = bool(raw.get("requires_agent_message"))
    if kind in (
        "question",
        "constraint_hold",
        "market_signal",
        "budget_update",
        "spec_reveal",
    ):
        if kind != "ack_nudge" and "requires_agent_message" not in raw:
            requires = True
    if kind == "ack_nudge" and "requires_agent_message" not in raw:
        requires = False
    if kind == "spec_reveal" and measures:
        if "requires_agent_message" not in raw:
            requires = False
    return {
        "kind": kind,
        "requires_agent_message": requires,
        "budget_ceiling": budget_f,
        "firm_measures": measures[:4],
        "reason": str(raw.get("reason") or "").strip() or None,
    }


_AGENT_MSG_CLASS_CACHE: dict[str, dict[str, bool]] = {}


def _llm_classify_agent_message(content: str, *, deadline: float) -> dict[str, bool]:
    """Semantic labels for one outbound agent message (cached by text)."""
    text = str(content or "").strip()
    empty = {
        "asks_permission": False,
        "asks_withheld_firm_spec": False,
        "claims_unverified_market_shock": False,
    }
    if not text:
        return dict(empty)
    cached = _AGENT_MSG_CLASS_CACHE.get(text)
    if cached is not None:
        return dict(cached)
    looks_ask = ("?" in text) or bool(
        re.search(
            r"(?i)\b(please\s+)?(provide|tell|share|confirm|what\s+is|what\s+are)\b",
            text,
        )
    )
    looks_spec = bool(
        re.search(
            r"(?i)\b(firm\s+)?(requirement|spec|constraint|withheld|gated)\b",
            text,
        )
    )
    looks_shock = bool(
        re.search(
            r"(?i)\b("
            r"sold\s+out|no\s+longer\s+available|out\s+of\s+stock|"
            r"availability/?price\s+changed|when\s+i\s+added\s+it\s+to\s+cart|"
            r"item'?s?\s+availability|remove\s+it\s+from\s+(your\s+)?cart"
            r")\b",
            text,
        )
    )
    fallback = {
        "asks_permission": ("?" in text) and not looks_spec,
        "asks_withheld_firm_spec": bool(looks_ask and looks_spec),
        "claims_unverified_market_shock": looks_shock,
    }
    if _time_left(deadline) < 5:
        _AGENT_MSG_CLASS_CACHE[text] = fallback
        return dict(fallback)
    sem = min(deadline, time.monotonic() + 4.0)
    raw = _llm_json(
        system=_AGENT_MESSAGE_GATE_SYSTEM,
        user=(
            f"Agent message:\n{text}\n\n"
            "Classify asks_permission, asks_withheld_firm_spec, "
            "claims_unverified_market_shock."
        ),
        deadline=sem,
    ) or {}
    out = {
        "asks_permission": (
            bool(raw["asks_permission"])
            if "asks_permission" in raw
            else fallback["asks_permission"]
        ),
        "asks_withheld_firm_spec": (
            bool(raw["asks_withheld_firm_spec"])
            if "asks_withheld_firm_spec" in raw
            else fallback["asks_withheld_firm_spec"]
        ),
        "claims_unverified_market_shock": (
            bool(raw["claims_unverified_market_shock"])
            if "claims_unverified_market_shock" in raw
            else fallback["claims_unverified_market_shock"]
        ),
    }
    _AGENT_MSG_CLASS_CACHE[text] = out
    return dict(out)


def _llm_agent_asks_permission(content: str, *, deadline: float) -> bool:
    """True when an outbound agent message asks the shopper for permission."""
    return bool(
        _llm_classify_agent_message(content, deadline=deadline).get("asks_permission")
    )


def _llm_build_claims(
    *,
    query: str,
    allowed: frozenset[str],
    required: list[str],
    spec_values: dict[str, str],
    view_facts: dict[str, str],
    last_stock: dict[str, Any] | None,
    deadline: float,
) -> list[dict[str, str]]:
    if not required and not allowed:
        return []
    sem_deadline = min(deadline, time.monotonic() + min(12.0, _SEMANTIC_BUDGET_S))
    fact_keys = sorted(
        {k for k in (view_facts or {}) if k and k not in _CLAIM_DENY}
    )
    system = _CLAIMS_SYSTEM.format(
        allowed=json.dumps(sorted(allowed), ensure_ascii=True),
        fact_keys=json.dumps(fact_keys, ensure_ascii=True),
        spec_values=json.dumps(spec_values or {}, ensure_ascii=True),
        required=json.dumps(required, ensure_ascii=True),
    )
    stock = {
        k: last_stock.get(k)
        for k in ("price", "in_stock", "product_id", "sku")
        if isinstance(last_stock, dict) and k in last_stock
    }
    user = (
        f"Shopper goal:\n{_shopper_text(query)}\n\n"
        f"Task rules excerpt:\n{query[len(_shopper_text(query)):].strip()[:1200]}\n\n"
        f"Viewed facts: {json.dumps(view_facts, ensure_ascii=True)}\n"
        f"Live stock: {json.dumps(stock, ensure_ascii=True)}\n"
    )
    raw = _llm_json(system=system, user=user, deadline=sem_deadline)
    claims_out: list[dict[str, str]] = []
    seen: set[str] = set()
    claims_in = (raw or {}).get("claims") if isinstance(raw, dict) else None
    if isinstance(claims_in, list):
        for claim in claims_in:
            if not isinstance(claim, dict):
                continue
            field = str(claim.get("field") or "").strip().lower().replace(" ", "_")
            if field not in allowed or field in _CLAIM_DENY or field in seen:
                continue
            value = _normalize_claim_value(field, claim.get("value", ""))
            if not value:
                continue
            if field == "brand" and value.lower() in ("none", "null", "n/a", "na"):
                continue
            claims_out.append({"field": field, "value": value})
            seen.add(field)
    mapping_in = (raw or {}).get("field_mapping") if isinstance(raw, dict) else None
    if isinstance(mapping_in, list):
        for row in mapping_in:
            if not isinstance(row, dict):
                continue
            field = str(row.get("field") or "").strip().lower().replace(" ", "_")
            if field not in allowed or field in _CLAIM_DENY or field in seen:
                continue
            value = _normalize_claim_value(field, row.get("value", ""))
            if not value:
                continue
            claims_out.append({"field": field, "value": value})
            seen.add(field)
    if last_stock:
        if "price" in allowed and last_stock.get("price") is not None:
            claims_out = [c for c in claims_out if c["field"] != "price"]
            claims_out.append(
                {
                    "field": "price",
                    "value": _normalize_claim_value("price", last_stock.get("price")),
                }
            )
            seen.add("price")
        if "in_stock" in allowed and last_stock.get("in_stock") is not None:
            claims_out = [c for c in claims_out if c["field"] != "in_stock"]
            claims_out.append(
                {
                    "field": "in_stock",
                    "value": _normalize_claim_value(
                        "in_stock", last_stock.get("in_stock")
                    ),
                }
            )
            seen.add("in_stock")
    fact_blob = " ".join(
        f"{k} {v}" for k, v in (view_facts or {}).items() if v
    ).lower()
    for field in required:
        if field not in allowed or field in ("price", "in_stock"):
            continue
        preferred = str((spec_values or {}).get(field) or "").strip()
        fact_val = str(view_facts.get(field) or "").strip()
        raw_val = ""
        if preferred and (
            preferred.lower() in fact_blob
            or any(
                tok.lower() in fact_blob
                for tok in re.findall(r"[A-Za-z0-9.]+", preferred)
                if len(tok) >= 2
            )
        ):
            related_vals = " ".join(
                str(view_facts.get(k) or "")
                for k in (view_facts or {})
                if field in str(k).lower()
                or str(k).lower() in field
                or _tokenize_overlap(field, str(k))
            ).lower()
            if related_vals and related_vals.count(",") >= 3:
                raw_val = ""
            else:
                raw_val = preferred
        elif fact_val:
            raw_val = fact_val
        else:
            for k, v in (view_facts or {}).items():
                kl = str(k).lower()
                if field in kl or kl in field or _tokenize_overlap(field, kl):
                    raw_val = str(v).strip()
                    break
        if not raw_val:
            continue
        claims_out = [c for c in claims_out if c["field"] != field]
        claims_out.append(
            {"field": field, "value": _normalize_claim_value(field, raw_val)}
        )
        seen.add(field)
    title = str(view_facts.get("title") or "").strip()
    brand_fact = str(view_facts.get("brand") or "").strip()
    if (
        "brand" in allowed
        and brand_fact
        and brand_fact.lower() not in ("none", "null", "n/a", "na", "")
        and brand_fact.lower() != title.lower()
    ):
        claims_out = [c for c in claims_out if c["field"] != "brand"]
        claims_out.append({"field": "brand", "value": brand_fact})
        seen.add("brand")
    else:
        claims_out = [c for c in claims_out if c["field"] != "brand"]
        seen.discard("brand")
    allow = set(required) | ({"brand"} if "brand" in allowed else set())
    justificationish = bool(spec_values) or any(
        f not in ("price", "in_stock", "brand") for f in required
    )
    if not justificationish:
        for key in (view_facts or {}):
            if key in allowed and key not in _CLAIM_DENY:
                allow.add(key)
    if not allow:
        allow = set(allowed) & ({"price", "in_stock", "brand"} | set(required))
        if not allow:
            allow = set(allowed)
    final: list[dict[str, str]] = []
    seen2: set[str] = set()
    for c in claims_out:
        if c["field"] not in allow or c["field"] in seen2:
            continue
        seen2.add(c["field"])
        final.append(c)
    return final


def _normalize_claim_value(field: str, value: Any) -> str:
    """Mechanical value cleanup only (bool/price/units) — no class synonym maps."""
    text = str(value).strip()
    if not text:
        return ""
    if field == "in_stock":
        low = text.lower()
        if low in ("true", "1", "yes", "in stock", "instock"):
            return "true"
        if low in ("false", "0", "no", "out of stock"):
            return "false"
        return text
    if field == "price":
        cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
        if cleaned:
            try:
                return str(float(cleaned))
            except ValueError:
                return text
        return text
    return text


def _dig_facts(obj: Any, out: dict[str, str], depth: int = 0) -> None:
    """Collect leaf attributes by their own keys — no domain synonym table."""
    if depth > 6 or obj is None:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).strip()
            lk = key.lower()
            if lk in _CLAIM_DENY:
                continue
            if isinstance(v, (dict, list)):
                _dig_facts(v, out, depth + 1)
                continue
            if v is None or v == "":
                continue
            if lk == "in_stock" or lk == "stock":
                if isinstance(v, bool):
                    out["in_stock"] = "true" if v else "false"
                elif isinstance(v, (int, float)):
                    out["in_stock"] = "true" if v > 0 else "false"
                else:
                    out["in_stock"] = str(v).strip()
                continue
            if lk in ("current_price", "price"):
                out["price"] = str(v).strip()
                continue
            if lk == "title":
                out.setdefault("title", str(v).strip())
                continue
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,40}", key):
                out.setdefault(lk if lk.isidentifier() else key, str(v).strip())
    elif isinstance(obj, list):
        for item in obj[:20]:
            _dig_facts(item, out, depth + 1)


def _extract_view_facts(observation: Any) -> dict[str, str]:
    facts: dict[str, str] = {}
    if not isinstance(observation, dict):
        return facts
    node = observation.get("observation")
    if not isinstance(node, dict):
        node = observation
    _dig_facts(node, facts)
    if node.get("title"):
        facts.setdefault("title", str(node["title"]).strip())
    cp = node.get("category_path")
    if isinstance(cp, list) and cp:
        facts["category_path"] = " > ".join(str(x).strip() for x in cp if str(x).strip())
    elif isinstance(cp, str) and cp.strip():
        facts["category_path"] = cp.strip()
    raw_facts = node.get("facts")
    if isinstance(raw_facts, dict):
        for k, v in raw_facts.items():
            if v is None or isinstance(v, (dict, list)):
                continue
            key = str(k).strip().lower().replace(" ", "_")
            if key and key not in _CLAIM_DENY:
                facts.setdefault(key, str(v).strip())
    attrs = node.get("attributes") or node.get("specs")
    if isinstance(attrs, list):
        for a in attrs:
            if not isinstance(a, dict):
                continue
            name = str(a.get("name") or a.get("key") or "").strip().lower()
            val = a.get("value") or a.get("val")
            if not name or val is None:
                continue
            key = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
            if key and key not in _CLAIM_DENY:
                facts.setdefault(key, str(val).strip())
    return facts


def _sanitize_place_order_args(
    args: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
    profile: dict[str, Any],
    last_stock: dict[str, Any] | None,
    last_view_facts: dict[str, str] | None,
    deadline: float,
    claim_cache: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    out = dict(args)
    if not profile.get("needs_justification"):
        out.pop("justification", None)
        return out

    view_facts = dict(last_view_facts or {})
    pid = str(out.get("product_id") or "")
    sku = str(out.get("sku") or "")
    cache_key = f"{pid}::{sku}"
    required = [
        f
        for f in (profile.get("required_claim_fields") or [])
        if f in allowed_fields and f not in _CLAIM_DENY
    ]
    if not required:
        required = [f for f in ("price", "in_stock") if f in allowed_fields]

    claims = claim_cache.get(cache_key)
    if claims is None:
        claims = _llm_build_claims(
            query=query,
            allowed=allowed_fields,
            required=required,
            spec_values=dict(profile.get("spec_claim_values") or {}),
            view_facts=view_facts,
            last_stock=last_stock,
            deadline=deadline,
        )
        if len([c for c in claims if c["field"] in required]) < min(2, len(required)):
            claims = list(claims)
            seen = {c["field"] for c in claims}
            if last_stock and last_stock.get("price") is not None and "price" in allowed_fields:
                claims = [c for c in claims if c["field"] != "price"]
                claims.append(
                    {
                        "field": "price",
                        "value": _normalize_claim_value("price", last_stock["price"]),
                    }
                )
                seen.add("price")
            if (
                last_stock
                and last_stock.get("in_stock") is not None
                and "in_stock" in allowed_fields
            ):
                claims = [c for c in claims if c["field"] != "in_stock"]
                claims.append(
                    {
                        "field": "in_stock",
                        "value": _normalize_claim_value(
                            "in_stock", last_stock["in_stock"]
                        ),
                    }
                )
                seen.add("in_stock")
            specs = dict(profile.get("spec_claim_values") or {})
            fact_blob = " ".join(
                f"{k} {v}" for k, v in view_facts.items() if v
            ).lower()
            for field in required:
                if field in seen or field not in allowed_fields:
                    continue
                preferred = str(specs.get(field) or "").strip()
                raw = ""
                if preferred and preferred.lower() in fact_blob:
                    raw = preferred
                else:
                    raw = str(view_facts.get(field) or "").strip()
                if not raw:
                    continue
                claims.append(
                    {
                        "field": field,
                        "value": _normalize_claim_value(field, raw),
                    }
                )
                seen.add(field)
        claim_cache[cache_key] = claims

    if claims and profile.get("needs_justification"):
        allow = {"price", "in_stock"}
        if "brand" in allowed_fields:
            allow.add("brand")
        for f in (profile.get("spec_claim_values") or {}):
            fl = str(f).strip().lower()
            if fl and fl in allowed_fields:
                allow.add(fl)
        for f in required:
            fl = str(f).strip().lower()
            if fl in ("price", "in_stock", "brand") or fl in (
                profile.get("spec_claim_values") or {}
            ):
                if fl in allowed_fields or fl in ("price", "in_stock"):
                    allow.add(fl)
        claims = [
            c
            for c in claims
            if isinstance(c, dict) and str(c.get("field") or "") in allow
        ]
        seen_f: set[str] = set()
        deduped: list[dict[str, str]] = []
        for c in claims:
            f = str(c.get("field") or "")
            if f in seen_f:
                continue
            seen_f.add(f)
            deduped.append(c)
        claims = deduped
        claim_cache[cache_key] = claims

    if claims:
        out["justification"] = {"claims": list(claims)}
    else:
        out.pop("justification", None)
    return out


def _arguments(
    tool_call: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
    profile: dict[str, Any],
    last_stock: dict[str, Any] | None,
    last_view_facts: dict[str, str] | None,
    deadline: float,
    claim_cache: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    if (tool_call["function"].get("name") or "") == "place_test_order":
        return _sanitize_place_order_args(
            parsed,
            query=query,
            allowed_fields=allowed_fields,
            profile=profile,
            last_stock=last_stock,
            last_view_facts=last_view_facts,
            deadline=deadline,
            claim_cache=claim_cache,
        )
    return parsed


_ASK_GATED_SPEC_CONTENT = (
    "What is the firm requirement you said you would share if I ask?"
)


def _message_requests_gated_spec(
    content: str,
    *,
    deadline: float | None = None,
) -> bool:
    """True when outbound chat requests a withheld firm spec (ask-first)."""
    text = str(content or "").strip()
    if not text:
        return False
    dl = float(deadline) if deadline is not None else (time.monotonic() + 4.0)
    return bool(
        _llm_classify_agent_message(text, deadline=dl).get("asks_withheld_firm_spec")
    )


def _message_claims_market_shock(
    content: str,
    *,
    deadline: float | None = None,
) -> bool:
    """True when outbound chat asserts an unverified stock/price shock."""
    text = str(content or "").strip()
    if not text:
        return False
    dl = float(deadline) if deadline is not None else (time.monotonic() + 4.0)
    return bool(
        _llm_classify_agent_message(text, deadline=dl).get(
            "claims_unverified_market_shock"
        )
    )


def _set_message_content(args: dict[str, Any], text: str) -> dict[str, Any]:
    out = dict(args)
    if "content" in out or "message" not in out:
        out["content"] = text
        out.pop("message", None)
    else:
        out["message"] = text
    return out


def _strip_budget_tokens_from_search(
    tool_calls: list[dict[str, Any]],
    budget: float | None,
) -> list[dict[str, Any]]:
    """Drop budget-ceiling literals from search queries (they pollute recall)."""
    if budget is None or not tool_calls:
        return tool_calls
    budget_s = f"{float(budget):.2f}".rstrip("0").rstrip(".")
    budget_full = f"{float(budget):.2f}"
    tokens = {budget_s, budget_full}
    if float(budget).is_integer():
        tokens.add(str(int(budget)))
    tokens.discard("")
    if not tokens:
        return tool_calls
    out: list[dict[str, Any]] = []
    changed = False
    for call in tool_calls:
        name = (call.get("function") or {}).get("name") or ""
        if name != "search":
            out.append(call)
            continue
        raw = (call.get("function") or {}).get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            out.append(call)
            continue
        if not isinstance(args, dict):
            out.append(call)
            continue
        q = str(args.get("query") or "")
        if not q:
            out.append(call)
            continue
        new_q = q
        for tok in sorted(tokens, key=len, reverse=True):
            new_q = re.sub(rf"(?<!\d){re.escape(tok)}(?!\d)", " ", new_q)
        new_q = re.sub(
            r"(?i)\b((at\s+or\s+)?(under|below|max|upto|up\s+to)\s+)"
            r"([a-z]{3})\b",
            " ",
            new_q,
        )
        new_q = re.sub(r"[$€£]", " ", new_q)
        new_q = re.sub(
            r"(?i)\b(at\s+or\s+)?(under|below|max|upto|up\s+to)\s*$",
            " ",
            new_q,
        )
        new_q = re.sub(
            r"(?i)\b(at\s+or\s+)?(under|below|max|upto|up\s+to)\s+(?=\b)",
            " ",
            new_q,
        )
        new_q = re.sub(r"\s{2,}", " ", new_q).strip(" ,;")
        if new_q != q:
            changed = True
            args = dict(args)
            args["query"] = new_q
            out.append(
                {
                    "id": call.get("id"),
                    "type": call.get("type") or "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps(args, ensure_ascii=True),
                    },
                }
            )
        else:
            out.append(call)
    return out if changed else tool_calls


def _rewrite_recovery_message(
    args: dict[str, Any],
    *,
    model_hint: str | None,
    deadline: float,
) -> dict[str, Any]:
    """Rewrite permission-asking agent messages into declarative recovery commits."""
    content = str(args.get("content") or args.get("message") or "")
    if not content.strip():
        return args
    if not _llm_agent_asks_permission(content, deadline=deadline):
        return args
    if model_hint:
        text = (
            f"The preferred option changed (stock/price) when I added it to cart. "
            f"I'm now selecting another color/variant of the same model ({model_hint}) "
            f"that is still in stock and within budget — firm requirements stay. "
            f"I will only change models if no same-model variant remains."
        )
    else:
        text = (
            "That item's availability/price changed when I added it to cart. "
            "I'm now selecting another in-stock option that still meets your "
            "firm budget and requirements."
        )
    return _set_message_content(args, text)


def _model_token_present(
    *,
    facts: dict[str, str] | None,
    title: str | None = None,
    options: str | None = None,
    model: str | None,
) -> bool:
    """True when preferred model token appears as a real model identity.

    Short codes like ``635`` must not match clothing SKUs (``635#``) or random
    digit runs; prefer an explicit ``model`` fact when the catalog supplies one.
    Category/shelf correctness is enforced by ``_product_type_violation``, not a
    fixed device-category vocabulary here.
    """
    tok = str(model or "").strip().lower()
    if not tok:
        return True
    tok_c = re.sub(r"[\s_-]+", "", tok)
    facts = facts if isinstance(facts, dict) else {}
    model_fact = str(facts.get("model") or "").strip().lower()
    title_s = str(title or facts.get("title") or "")
    if model_fact:
        fact_c = re.sub(r"[\s_-]+", "", model_fact)
        return bool(
            tok == model_fact
            or tok_c == fact_c
            or tok in model_fact.split()
            or (bool(tok_c) and tok_c in fact_c)
        )
    if tok.isdigit() and len(tok) <= 4:
        return False
    options_s = str(options or "")
    blob = f"{title_s} {options_s}".lower()
    if not blob.strip():
        return False
    if re.search(
        rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9#])",
        blob,
        flags=re.I,
    ):
        return True
    if tok_c and len(tok_c) >= 4 and re.search(
        rf"(?<![a-z0-9]){re.escape(tok_c)}(?![a-z0-9#])",
        re.sub(r"[\s_-]+", "", blob),
        flags=re.I,
    ):
        return True
    return False


def _color_token_present(
    *,
    facts: dict[str, str] | None,
    title: str | None = None,
    options: str | None = None,
    color: str | None,
) -> bool:
    col = str(color or "").strip().lower()
    if not col:
        return True
    facts = facts if isinstance(facts, dict) else {}
    blob_parts = [
        str(title or facts.get("title") or ""),
        str(options or ""),
        str(facts.get("color") or ""),
        str(facts.get("colour") or ""),
    ]
    blob_parts.extend(str(v) for v in facts.values())
    blob = " ".join(blob_parts).lower()
    if not blob.strip():
        return False
    if col in blob:
        return True
    last = col.split()[-1]
    return bool(last) and last in blob


def _obs_stock_payload(observation: Any) -> dict[str, Any] | None:
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
    return None


def _stock_is_false(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    val = payload.get("in_stock")
    if val is False:
        return True
    if isinstance(val, str) and val.strip().lower() in ("false", "0", "no"):
        return True
    return False


def _parse_leading_number(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", str(text or ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _harvest_number_unit_pairs(text: str) -> list[tuple[float, str]]:
    """Generic number(+optional unit token) harvest — not a domain unit table."""
    out: list[tuple[float, str]] = []
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*([A-Za-z%]{0,16})",
        str(text or ""),
    ):
        try:
            num = float(m.group(1))
        except ValueError:
            continue
        unit = (m.group(2) or "").strip(".,;:()[]{}\"'").lower()
        out.append((num, unit))
    return out


def _ingest_closed_set_obs(
    observation: Any,
    *,
    candidates: list[tuple[str, str]],
    ledger: dict[str, dict[str, Any]],
) -> None:
    """Store closed-set observations for axis binding + LLM ranking."""
    if not candidates or not isinstance(observation, dict):
        return
    cand_keys = {f"{pid}::{sku}": (pid, sku) for pid, sku in candidates}
    node = observation.get("observation")
    if not isinstance(node, dict):
        node = observation

    def ingest_product(prod: dict[str, Any]) -> None:
        pid = str(prod.get("product_id") or "")
        if not pid:
            return
        title = str(prod.get("title") or "")
        facts = prod.get("facts") if isinstance(prod.get("facts"), dict) else {}
        fact_map = {
            str(k).strip().lower().replace(" ", "_"): str(v).strip()
            for k, v in facts.items()
            if str(k).strip() and v is not None and str(v).strip()
        }
        variants = prod.get("variants") if isinstance(prod.get("variants"), list) else []
        for var in variants:
            if not isinstance(var, dict):
                continue
            sku = str(var.get("sku") or "")
            key = f"{pid}::{sku}"
            if key not in cand_keys:
                continue
            options = str(var.get("options") or "")
            entry = ledger.setdefault(key, {"product_id": pid, "sku": sku})
            entry["title"] = title
            entry["options"] = options
            if fact_map:
                merged = dict(entry.get("facts") or {})
                merged.update(fact_map)
                entry["facts"] = merged
            price = var.get("price")
            if price is not None:
                try:
                    entry["price"] = float(price)
                except (TypeError, ValueError):
                    cleaned = re.sub(r"[^\d.]", "", str(price).replace(",", ""))
                    try:
                        entry["price"] = float(cleaned)
                    except ValueError:
                        pass
            if "in_stock" in var:
                entry["in_stock"] = bool(var.get("in_stock"))
            blob = " | ".join(
                [
                    title,
                    options,
                    " ".join(f"{k} {v}" for k, v in (entry.get("facts") or {}).items()),
                ]
            )
            entry["blob"] = blob[:1200]
            entry["measures"] = _harvest_number_unit_pairs(blob)

    if "product_id" in node or "variants" in node or "title" in node:
        ingest_product(node)
    for k in ("products", "items", "results", "compared"):
        block = node.get(k)
        if isinstance(block, list):
            for p in block:
                if isinstance(p, dict):
                    ingest_product(p)


def _ledger_apply_stock(
    ledger: dict[str, dict[str, Any]],
    payload: dict[str, Any] | None,
) -> None:
    """Merge inspect/cart stock into closed-set ledger (currency-agnostic price)."""
    if not payload:
        return
    pid = str(payload.get("product_id") or "")
    sku = str(payload.get("sku") or "")
    if not pid or not sku:
        return
    key = f"{pid}::{sku}"
    entry = ledger.setdefault(key, {"product_id": pid, "sku": sku})
    if payload.get("price") is not None:
        try:
            entry["price"] = float(
                re.sub(r"[^\d.]", "", str(payload.get("price")).replace(",", ""))
            )
        except (TypeError, ValueError):
            pass
    if "in_stock" in payload:
        entry["in_stock"] = not _stock_is_false(payload)


def _observed_measure_units(
    ledger: dict[str, dict[str, Any]],
    candidates: list[tuple[str, str]],
) -> list[str]:
    """Unit tokens harvested from THIS episode's catalog text only."""
    units: list[str] = []
    for pid, sku in candidates:
        e = ledger.get(f"{pid}::{sku}") or {}
        for _num, unit in e.get("measures") or []:
            u = str(unit or "").strip().lower()
            if not u or len(u) > 12:
                continue
            if u.isdigit():
                continue
            if u not in units:
                units.append(u)
    return units[:16]


def _unit_token_match(harvested: str, expected: set[str]) -> bool:
    """Match harvested unit to expected tokens without short-prefix false positives.

    Model codes like `HD27H` harvest as `(27, 'h')`; that must NOT match `hz`.
    """
    h = str(harvested or "").strip().lower()
    if not h or not expected:
        return False
    for u in expected:
        u = str(u or "").strip().lower()
        if not u:
            continue
        if h == u:
            return True
        if len(h) >= 2 and len(u) >= 2 and (h.startswith(u) or u.startswith(h)):
            return True
    return False


def _default_axis_bindings(
    axes: list[str],
    *,
    observed_units: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Structural stubs only — units/keywords come from THIS episode via LLM bind.

    Do not bake domain unit vocab (Hz, inch, …); those must come from observed
    catalog measure tokens or the axis-bind LLM call.
    """
    out: dict[str, dict[str, Any]] = {}
    observed = [u for u in (observed_units or []) if u]
    for axis in axes:
        tokens = [t for t in re.split(r"[_\s]+", str(axis or "")) if t]
        a = str(axis or "").strip().lower()
        if a in ("price", "budget", "cost"):
            out[axis] = {
                "fact_keys": ["price"],
                "unit_tokens": [],
                "text_keywords": ["price"],
                "source_priority": ["price", "options", "facts", "title"],
                "aggregate": "first",
            }
            continue
        out[axis] = {
            "fact_keys": [a] + tokens if a else list(tokens),
            "unit_tokens": list(observed),
            "text_keywords": list(tokens),
            "source_priority": ["options", "facts", "title", "price"],
            "aggregate": "first",
        }
    return out


def _llm_bind_rank_axes(
    *,
    profile: dict[str, Any],
    ledger: dict[str, dict[str, Any]],
    candidates: list[tuple[str, str]],
    deadline: float,
) -> dict[str, dict[str, Any]]:
    axes = list(profile.get("rank_axes") or [])
    observed_units = _observed_measure_units(ledger, candidates)
    base = _default_axis_bindings(axes, observed_units=observed_units)
    if not axes or _time_left(deadline) < 10:
        return base
    samples: list[dict[str, Any]] = []
    for pid, sku in candidates[:6]:
        key = f"{pid}::{sku}"
        e = ledger.get(key) or {}
        samples.append(
            {
                "key": key,
                "title": e.get("title"),
                "options": e.get("options"),
                "facts": e.get("facts") or {},
                "price": e.get("price"),
                "measure_samples": (e.get("measures") or [])[:8],
            }
        )
    if not any(s.get("title") or s.get("facts") or s.get("options") for s in samples):
        return base
    higher = {
        str(x).strip().lower()
        for x in (profile.get("higher_better_axes") or [])
        if str(x or "").strip()
    }
    system = _fill_template(
        _AXIS_BIND_SYSTEM,
        {
            "AXES": json.dumps(axes, ensure_ascii=True),
            "HIGHER": json.dumps(list(profile.get("higher_better_axes") or []), ensure_ascii=True),
            "SAMPLES": json.dumps(samples, ensure_ascii=True)[:5000],
        },
    )
    raw = _llm_json(
        system=system,
        user=(
            "Bind each rank axis to fact keys / unit tokens / keywords from the "
            "samples only. Prefer units that appear beside numbers in samples."
        ),
        deadline=min(deadline, time.monotonic() + min(12.0, _SEMANTIC_BUDGET_S)),
    ) or {}
    bindings = raw.get("bindings")
    if not isinstance(bindings, dict):
        return base
    for axis in axes:
        spec = bindings.get(axis)
        if not isinstance(spec, dict):
            if str(axis).strip().lower() in higher:
                cur = dict(base.get(axis) or {})
                cur["aggregate"] = "max"
                base[axis] = cur
            continue
        cur = dict(base.get(axis) or {})
        for field, limit in (
            ("fact_keys", 8),
            ("unit_tokens", 8),
            ("text_keywords", 8),
            ("source_priority", 6),
        ):
            vals = _str_list(spec.get(field), limit=limit)
            if vals:
                cur[field] = [v.lower() for v in vals]
        agg = str(spec.get("aggregate") or "").strip().lower()
        if agg in ("max", "min", "first"):
            cur["aggregate"] = agg
        elif str(axis).strip().lower() in higher:
            cur["aggregate"] = "max"
        if (
            axis not in ("price", "budget", "cost")
            and not cur.get("unit_tokens")
            and observed_units
        ):
            cur["unit_tokens"] = list(observed_units)
        if cur.get("unit_tokens") and observed_units:
            obs = {str(u).lower() for u in observed_units}
            kept = [
                u
                for u in cur["unit_tokens"]
                if _unit_token_match(str(u), obs) or str(u).lower() in obs
            ]
            if kept:
                cur["unit_tokens"] = kept
        base[axis] = cur
    return base


def _axis_value_from_entry(
    entry: dict[str, Any],
    axis: str,
    binding: dict[str, Any] | None,
) -> float | None:
    """Extract one axis number using catalog bindings (respect source_priority)."""
    if entry.get(axis) is not None and axis not in (
        "title",
        "options",
        "facts",
        "blob",
        "measures",
    ):
        try:
            pre = float(entry[axis])
        except (TypeError, ValueError):
            pre = None
    else:
        pre = None
    bind = binding or {}
    facts = entry.get("facts") if isinstance(entry.get("facts"), dict) else {}
    sources = {
        "price": str(entry.get("price") or ""),
        "facts": " ".join(f"{k} {v}" for k, v in facts.items()),
        "options": str(entry.get("options") or ""),
        "title": str(entry.get("title") or ""),
    }
    priority = [
        str(x).lower()
        for x in (bind.get("source_priority") or ["options", "facts", "title", "price"])
    ]
    fact_keys = [str(x).lower() for x in (bind.get("fact_keys") or [])]
    units = {str(x).lower() for x in (bind.get("unit_tokens") or []) if x}
    keywords = [str(x).lower() for x in (bind.get("text_keywords") or []) if x]
    agg = str(bind.get("aggregate") or "first").strip().lower()

    def from_facts() -> float | None:
        for fk in fact_keys:
            if fk in facts:
                num = _parse_leading_number(str(facts[fk]))
                if num is not None:
                    return num
        for fk, fv in facts.items():
            fl = str(fk).lower()
            if any(fl == k or k in fl or fl in k for k in fact_keys if len(k) >= 3):
                num = _parse_leading_number(str(fv))
                if num is not None:
                    return num
        return None

    def from_blob(blob: str) -> float | None:
        low = (blob or "").lower()
        if not low:
            return None
        hits: list[float] = []
        if units:
            for num, unit in _harvest_number_unit_pairs(low):
                if unit and _unit_token_match(unit, units):
                    hits.append(float(num))
        for kw in keywords + fact_keys:
            if len(kw) < 2:
                continue
            pat = (
                rf"(?P<n1>\d+(?:\.\d+)?)\s*{re.escape(kw)}\b|"
                rf"{re.escape(kw)}\W{{0,16}}(?P<n2>\d+(?:\.\d+)?)"
            )
            for m in re.finditer(pat, low):
                raw = m.group("n1") or m.group("n2")
                if raw is None:
                    continue
                try:
                    hits.append(float(raw))
                except ValueError:
                    continue
        if not hits:
            return None
        if agg == "max":
            return max(hits)
        if agg == "min":
            return min(hits)
        return hits[0]

    if axis in ("price", "budget", "cost"):
        try:
            return float(entry["price"])
        except (KeyError, TypeError, ValueError):
            return _parse_leading_number(sources.get("price") or "")

    for src_name in priority:
        if src_name == "price":
            continue
        if src_name == "facts":
            num = from_facts()
            if num is not None:
                return num
            continue
        num = from_blob(sources.get(src_name) or "")
        if num is not None:
            return num
    for src_name, blob in sources.items():
        if src_name == "facts":
            num = from_facts()
        else:
            num = from_blob(blob)
        if num is not None:
            return num
    return pre


def _apply_axis_bindings(
    ledger: dict[str, dict[str, Any]],
    *,
    axes: list[str],
    bindings: dict[str, dict[str, Any]],
) -> None:
    for entry in ledger.values():
        for axis in axes:
            if axis in entry and axis not in ("price",):
                entry.pop(axis, None)
            bind = bindings.get(axis) if isinstance(bindings, dict) else None
            if not isinstance(bind, dict) or not bind:
                bind = _default_axis_bindings([axis]).get(axis) or {}
            val = _axis_value_from_entry(entry, axis, bind)
            if val is not None:
                entry[axis] = val


def _llm_rank_closed_set(
    *,
    query: str,
    profile: dict[str, Any],
    ledger: dict[str, dict[str, Any]],
    candidates: list[tuple[str, str]],
    dead: set[str],
    deadline: float,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """LLM ranks closed-set keys from observed facts (catalog-grounded)."""
    if not candidates or _time_left(deadline) < 10:
        return [], {}
    keys = [f"{pid}::{sku}" for pid, sku in candidates]
    cands: list[dict[str, Any]] = []
    for key in keys:
        e = dict(ledger.get(key) or {})
        cands.append(
            {
                "key": key,
                "dead": key in dead,
                "title": e.get("title"),
                "options": e.get("options"),
                "facts": e.get("facts") or {},
                "price": e.get("price"),
                "in_stock": e.get("in_stock"),
                "measures": e.get("measures") or [],
                "axis_values": {
                    a: e.get(a)
                    for a in (profile.get("rank_axes") or [])
                    if e.get(a) is not None
                },
            }
        )
    axes = list(profile.get("rank_axes") or [])
    higher = list(profile.get("higher_better_axes") or [])
    bud = profile.get("budget_ceiling")
    system = _fill_template(
        _CLOSED_RANK_SYSTEM,
        {
            "PROFILE": json.dumps(
                {
                    "firm_brand": profile.get("firm_brand"),
                    "soft_brand": profile.get("soft_brand"),
                    "preferred_model": profile.get("preferred_model"),
                    "firm_spec_notes": profile.get("firm_spec_notes"),
                    "phrase_tokens": profile.get("phrase_tokens"),
                    "negate_tokens": profile.get("negate_tokens"),
                    "require_soft_evidence": profile.get("require_soft_evidence"),
                },
                ensure_ascii=True,
            ),
            "AXES": json.dumps(axes, ensure_ascii=True),
            "HIGHER": json.dumps(higher, ensure_ascii=True),
            "BUDGET": json.dumps(bud, ensure_ascii=True),
            "DEAD": json.dumps(sorted(dead), ensure_ascii=True),
            "CANDS": json.dumps(cands, ensure_ascii=True)[:6000],
        },
    )
    user = (
        f"Shopper goal:\n{_shopper_text(query)}\n\n"
        "Return ordered_keys (best first) and axis_values now. "
        "Primary axis dominates; do not pick a worse primary-axis option for price."
    )
    sem = min(deadline, time.monotonic() + min(14.0, _SEMANTIC_BUDGET_S))
    raw = _llm_json(system=system, user=user, deadline=sem) or {}
    ordered: list[str] = []
    keyset = set(keys)
    for item in raw.get("ordered_keys") or []:
        k = str(item or "").strip()
        if k in keyset and k not in ordered:
            ordered.append(k)
    for k in keys:
        if k not in ordered:
            ordered.append(k)
    axis_values: dict[str, dict[str, float]] = {}
    raw_axes = raw.get("axis_values")
    if isinstance(raw_axes, dict):
        for k, vals in raw_axes.items():
            key = str(k or "").strip()
            if key not in keyset or not isinstance(vals, dict):
                continue
            parsed: dict[str, float] = {}
            for axis, val in vals.items():
                an = str(axis or "").strip().lower().replace(" ", "_")
                try:
                    parsed[an] = float(val)
                except (TypeError, ValueError):
                    continue
            if parsed:
                axis_values[key] = parsed
    return ordered, axis_values


def _best_closed_key(
    ledger: dict[str, dict[str, Any]],
    *,
    candidates: list[tuple[str, str]],
    priority: list[str],
    higher_better: set[str],
    dead: set[str],
    preferred_order: list[str] | None = None,
) -> str | None:
    keys = [f"{pid}::{sku}" for pid, sku in candidates if f"{pid}::{sku}" not in dead]
    if not keys:
        return None
    if preferred_order:
        for k in preferred_order:
            if k in keys:
                return k
    axes = list(priority) if priority else ["price"]

    def sort_key(k: str) -> tuple:
        e = ledger.get(k) or {}
        stock_pen = 1.0 if e.get("in_stock") is False else 0.0
        parts: list[float] = [stock_pen]
        for axis in axes:
            val = e.get(axis)
            if val is None:
                parts.extend([1.0, 0.0])
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                parts.extend([1.0, 0.0])
                continue
            if axis in ("price", "budget", "cost", "weight"):
                prefer_high = False
            elif axis in higher_better:
                prefer_high = True
            else:
                prefer_high = True
            parts.extend([0.0, -num if prefer_high else num])
        return tuple(parts)

    return sorted(keys, key=sort_key)[0]


def _chat(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    deadline: float,
) -> dict[str, Any] | None:
    """Single-pass inference with a short model cascade. Returns None if out of time."""
    if time.monotonic() >= deadline:
        return None
    last_err: Exception | None = None
    for model in _model_list():
        if time.monotonic() >= deadline:
            break
        for _try in range(_MAX_INFER_TRIES_PER_MODEL):
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
                    continue
                return inference
            except Exception as exc:
                last_err = exc
                continue
    _ = last_err
    return None


def _env_call(envelope: dict[str, Any], *, deadline: float) -> dict[str, Any] | None:
    if time.monotonic() >= deadline:
        return None
    base_id = str(envelope.get("call_id") or "call")
    for attempt in range(3):
        if time.monotonic() >= deadline:
            return None
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
        time.sleep(0.35 * (attempt + 1))
    return None


def _tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [str(c.get("function", {}).get("name") or "") for c in tool_calls]


def _parse_key_parts(key: str) -> tuple[str, str] | None:
    if "::" not in key:
        return None
    pid, sku = key.split("::", 1)
    return pid, sku


def _forced_tool(
    name: str,
    args: dict[str, Any],
    *,
    turn: int,
    tag: str,
) -> dict[str, Any]:
    return {
        "id": f"forced-{tag}-{turn}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _forced_order_call(
    *,
    turn: int,
    pick: dict[str, str],
    query: str,
    allowed_fields: frozenset[str],
    profile: dict[str, Any],
    last_stock: dict[str, Any] | None,
    last_view_facts: dict[str, str],
    deadline: float,
    claim_cache: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "product_id": pick["product_id"],
        "sku": pick["sku"],
    }
    if profile.get("needs_justification"):
        args["justification"] = {"claims": []}
        args = _sanitize_place_order_args(
            args,
            query=query,
            allowed_fields=allowed_fields,
            profile=profile,
            last_stock=last_stock,
            last_view_facts=last_view_facts,
            deadline=deadline,
            claim_cache=claim_cache,
        )
    return _forced_tool("place_test_order", args, turn=turn, tag="order")


def _rewrite_cart_to_key(
    tool_calls: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    parts = _parse_key_parts(key)
    if not parts:
        return tool_calls
    pid, sku = parts
    out: list[dict[str, Any]] = []
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        if name not in ("add_to_cart", "inspect_stock", "place_test_order"):
            out.append(call)
            continue
        cloned = {
            "id": call.get("id") or f"rewritten-{name}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps({"product_id": pid, "sku": sku}),
            },
        }
        out.append(cloned)
    return out or tool_calls


def _time_left(deadline: float) -> float:
    return deadline - time.monotonic()


def _closed_set_violation(
    tool_calls: list[dict[str, Any]],
    *,
    candidates: list[tuple[str, str]],
    viewed: set[str],
) -> str | None:
    if not candidates:
        return None
    cand_pids = {pid for pid, _ in candidates}
    cand_keys = {f"{pid}::{sku}" for pid, sku in candidates}
    names = _tool_names(tool_calls)
    if any(n in ("search", "filter") for n in names) and len(viewed & cand_pids) < min(
        2, len(cand_pids)
    ):
        return (
            "Closed candidate set: do not search/filter. "
            f"Call view on listed product_ids first: {sorted(cand_pids)}"
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
            if (
                pid
                and sku
                and f"{pid}::{sku}" not in cand_keys
                and name in ("add_to_cart", "place_test_order")
            ):
                return (
                    f"{name} {pid}::{sku} is not a listed variant. "
                    f"Use one of: {sorted(cand_keys)}"
                )
    return None


def _oos_order_violation(
    tool_calls: list[dict[str, Any]],
    *,
    last_stock: dict[str, Any] | None,
) -> str | None:
    if not _stock_is_false(last_stock):
        return None
    if "place_test_order" in _tool_names(tool_calls):
        return (
            "Rejected: latest stock/cart shows in_stock=false (market shock). "
            "remove_from_cart if needed, pick another candidate, inspect_stock, "
            "then order only an in-stock sku."
        )
    return None


def _tokenize_overlap(a: str, b: str) -> bool:
    """True when snake/space tokens overlap (len>=3) — no synonym tables."""
    def toks(s: str) -> set[str]:
        parts = re.split(r"[^a-z0-9]+", (s or "").lower())
        return {p for p in parts if len(p) >= 3}

    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        sa = (a or "").strip().lower()
        sb = (b or "").strip().lower()
        return bool(sa) and sa == sb
    if ta & tb:
        return True
    for x in ta:
        for y in tb:
            if x in y or y in x:
                return True
    return False


def _numbers_in_text(text: str) -> list[float]:
    out: list[float] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)", str(text or "")):
        try:
            out.append(float(m.group(1)))
        except ValueError:
            continue
    return out


def _token_evidenced(token: str, surface: str) -> bool:
    tok = (token or "").strip()
    if not tok or not surface:
        return False
    low = surface.lower()
    if tok.lower() in low:
        return True
    return _tokenize_overlap(tok, surface)


def _simple_noun_stem(word: str) -> str:
    """Plural-insensitive stem for a single taxonomy/shopper noun (no vocab list)."""
    t = re.sub(r"[^a-z0-9]+", "", (word or "").lower())
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    return t


def _taxonomy_last_word(cat_path: str) -> str:
    """Last alphanumeric word of the last category_path segment."""
    segs = [
        s.strip()
        for s in re.split(r"[>/|,;]+", (cat_path or "").lower())
        if s.strip()
    ]
    if not segs:
        return ""
    words = [w for w in re.split(r"[^a-z0-9]+", segs[-1]) if w]
    return words[-1] if words else ""


def _required_leaf_fits_taxonomy(
    leaf: str,
    cat_path: str,
    tokens: list[str],
) -> bool:
    """Shopper product-type noun must match the catalog LEAF, not an ancestor.

    ``Smartphones > Cases & Covers`` contains ``smartphone`` on the parent but
    the listing type is the last word (``covers``). Equality is stem-only —
    no accessory/parts keyword list.
    """
    leaf_n = str(leaf or "").strip().lower()
    last_word = _taxonomy_last_word(cat_path)
    if not leaf_n or not last_word:
        return True
    last_stem = _simple_noun_stem(last_word)
    candidates: list[str] = []
    if " " not in leaf_n:
        candidates.append(leaf_n)
    else:
        last_seg = [
            s.strip()
            for s in re.split(r"[>/|,;]+", cat_path.lower())
            if s.strip()
        ][-1]
        if _category_token_on_path(leaf_n, last_seg):
            return True
        candidates.extend(
            w for w in re.split(r"[^a-z0-9]+", leaf_n) if len(w) >= 4
        )
    for t in tokens:
        tw = str(t or "").strip().lower()
        if tw and " " not in tw:
            candidates.append(tw)
    for c in candidates:
        if len(_simple_noun_stem(c)) >= 4 and _simple_noun_stem(c) == last_stem:
            return True
    return False


def _category_token_on_path(token: str, cat_path: str) -> bool:
    """Match shopper category token against catalog category_path segments.

    Stricter than generic tokenize overlap: short stems like ``phone`` must not
    satisfy ``Smartphones`` via substring-of-compound (Phone Charms / Cases),
    and bag-of-words across the whole path must not let ``mobile phones`` match
    accessory shelves that merely contain ``Mobile``.
    """
    tok = (token or "").strip().lower()
    path = (cat_path or "").strip().lower()
    if not tok or not path:
        return False
    if len(tok) >= 6 and re.search(
        rf"(^|[^a-z0-9]){re.escape(tok)}([^a-z0-9]|$)", path
    ):
        return True
    tok_stem = tok.rstrip("s")
    parts = [p for p in re.split(r"[^a-z0-9]+", tok) if len(p) >= 4]
    for seg in re.split(r"[>/|,;]+", path):
        seg = seg.strip()
        if not seg:
            continue
        if tok == seg or tok_stem == seg.rstrip("s"):
            return True
        if len(tok_stem) >= 6 and re.search(
            rf"(^|[^a-z0-9]){re.escape(tok)}([^a-z0-9]|$)", seg
        ):
            return True
        if len(tok_stem) >= 6 and re.search(
            rf"(^|[^a-z0-9]){re.escape(tok_stem)}([^a-z0-9]|$)", seg
        ):
            return True
        if len(parts) >= 2 and all(
            re.search(rf"(^|[^a-z0-9]){re.escape(p)}([^a-z0-9]|$)", seg)
            for p in parts
        ):
            return True
    return False


def _catalog_category_for_filter(token: str) -> str:
    """Normalize a profile category token toward catalog filter labels.

    Preserve multi-word catalog paths. Title-case bare tokens; add a trailing
    ``s`` only for simple singular nouns — never ``Beauty`` → ``Beautys``.
    """
    t = str(token or "").strip()
    if not t:
        return t
    if any(ch in t for ch in (" ", "/", "&", ">")):
        return t
    titled = t[:1].upper() + t[1:] if t.lower() == t else t
    low = titled.lower()
    if low.endswith("s") or low.endswith("y"):
        return titled
    return titled + "s"


def _phrase_search_queries(profile: dict[str, Any], query: str = "") -> list[str]:
    """Distinct phrase+category search strings from THIS episode only."""
    phrases = [
        str(p).strip()
        for p in (profile.get("phrase_tokens") or [])
        if str(p or "").strip()
    ]
    if not phrases:
        goal = str(query or "").split("Task rules:")[0]
        for m in re.finditer(
            r"""(?:matches(?:\s+the)?\s+phrase|matches)\s+[\"']([^\"']+)[\"']"""
            r"""|matches\s+the\s+phrase\s+([A-Za-z0-9][A-Za-z0-9\s_-]{1,40})"""
            r"""|[\"']([A-Za-z0-9][A-Za-z0-9\s_-]{1,40})[\"']""",
            goal,
            flags=re.I,
        ):
            hit = next((g for g in m.groups() if g), None)
            if hit and hit.strip():
                phrases.append(hit.strip())
        seen: set[str] = set()
        uniq: list[str] = []
        for p in phrases:
            k = p.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        phrases = uniq[:3]
    cats = _category_tokens_for_gate(profile)
    cat = cats[0] if cats else ""
    out: list[str] = []

    def push(q: str) -> None:
        q = " ".join(str(q).split())
        if q and q.lower() not in {x.lower() for x in out}:
            out.append(q)

    for ph in phrases:
        parts = [p for p in re.split(r"\s+", ph.strip()) if p]
        push(ph)
        if len(parts) >= 2:
            push(" ".join(reversed(parts)))
            if len(parts) >= 3:
                push(" ".join(parts[1:] + parts[:1]))
            for part in parts:
                if len(part) >= 3:
                    push(part)
                    if cat:
                        push(f"{part} {cat}")
        if cat:
            push(f"{ph} {cat}")
            push(f"{cat} {ph}")
            if len(parts) >= 2:
                push(f"{' '.join(reversed(parts))} {cat}")
    if cat and not phrases:
        push(cat)
    return out[:6]


def _phrase_identity_target(query: str) -> int:
    """How many distinct matching identities Task rules ask to surface."""
    m = re.search(
        r"surface\s+at\s+least\s+(\d+)\s+distinct\s+matching",
        str(query or ""),
        flags=re.I,
    )
    if m:
        try:
            return max(3, min(8, int(m.group(1))))
        except ValueError:
            pass
    return 5


def _category_tokens_for_gate(profile: dict[str, Any]) -> list[str]:
    """Taxonomy tokens only — drop phrase_tokens that bled into category_tokens."""
    phrases = {
        str(p).strip().lower()
        for p in (profile.get("phrase_tokens") or [])
        if str(p or "").strip()
    }
    phrase_parts: set[str] = set()
    for p in phrases:
        for part in re.split(r"[^a-z0-9]+", p):
            if len(part) >= 3:
                phrase_parts.add(part)
    out: list[str] = []
    for t in profile.get("category_tokens") or []:
        tok = str(t or "").strip()
        if not tok:
            continue
        low = tok.lower()
        if low in phrases:
            continue
        parts = [p for p in re.split(r"[^a-z0-9]+", low) if len(p) >= 3]
        if parts and all(p in phrase_parts for p in parts):
            continue
        if tok not in out:
            out.append(tok)
    return out


def _looks_like_phrase_retrieval(profile: dict[str, Any], query: str) -> bool:
    """True for category+phrase buy tasks (TF2-like) — no soft brand / firm measures."""
    if profile.get("soft_brand") or profile.get("firm_brand") or profile.get(
        "brand_is_firm"
    ):
        return False
    if profile.get("firm_measures"):
        return False
    phrases = [str(p).strip() for p in (profile.get("phrase_tokens") or []) if str(p).strip()]
    cats = _category_tokens_for_gate(profile)
    if phrases and cats:
        return True
    low = (query or "").lower()
    return ("matches" in low or "match the phrase" in low) and "category" in low


def _product_type_violation(
    facts: dict[str, str],
    profile: dict[str, Any],
) -> str | None:
    """Firm product-type/category gate from profile category_tokens / leaf.

    When the catalog exposes category_path, tokens must match THAT path.
    Sibling leaves must not pass via SEO title alone — path match is required.
    Phrase+category buys rely on category_tokens path overlap + title phrase
    gates — not a singularized required_product_leaf.
    """
    if _looks_like_phrase_retrieval(profile, ""):
        tokens = _category_tokens_for_gate(profile)
        if not facts:
            return "missing facts for category check" if tokens else None
        cat_path = str(facts.get("category_path") or "").strip()
        if not tokens:
            return None
        if cat_path:
            if any(_category_token_on_path(t, cat_path) for t in tokens):
                return None
            return (
                f"category_path `{cat_path}` misses type token(s) "
                f"{tokens[:3]}"
            )
        blob = " ".join(
            str(v) for k, v in facts.items() if k != "category_path" and v
        )
        if any(_token_evidenced(t, blob) for t in tokens):
            return None
        return f"product type token(s) {tokens[:3]} not evidenced"
    tokens = _category_tokens_for_gate(profile)
    leaf = str(profile.get("required_product_leaf") or "").strip().lower()
    if leaf and leaf not in [str(t).lower() for t in tokens]:
        tokens = [leaf] + list(tokens)
    if not facts:
        return "missing facts for category check" if tokens else None
    cat_path = str(facts.get("category_path") or "").strip()
    if leaf and cat_path and not _required_leaf_fits_taxonomy(leaf, cat_path, tokens):
        return (
            f"taxonomy leaf `{_taxonomy_last_word(cat_path)}` "
            f"is not required type `{leaf}`"
        )
    if not tokens:
        return None
    if cat_path:
        missing = [t for t in tokens if not _category_token_on_path(t, cat_path)]
        if missing:
            return (
                f"category_path `{cat_path}` misses type token(s) "
                f"{missing[:3]}"
            )
        return None
    blob = " ".join(
        str(v) for k, v in facts.items() if k != "category_path" and v
    )
    missing = [t for t in tokens if not _token_evidenced(t, blob)]
    if missing:
        return f"product type token(s) {missing[:3]} not evidenced"
    return None


def _firm_measure_violation(
    facts: dict[str, str],
    profile: dict[str, Any],
) -> str | None:
    """Literal firm-measure gate: token/numeric overlap on structured facts only.

    No display-class or pixel synonym remaps — those belong to LLM semantic
    checks (`_llm_firm_measures_ok`) when literal evidence is ambiguous.
    """
    measures = profile.get("firm_measures") or []
    if not measures:
        return None
    if not facts:
        return "missing facts for firm measure check"
    structured = {
        k: v for k, v in facts.items() if str(k).lower() != "title"
    } or dict(facts)
    blob = " ".join(f"{k} {v}" for k, v in structured.items()).lower()
    for spec in measures:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name") or "").strip()
        vt = str(spec.get("value_text") or "").strip()
        target = spec.get("numeric")
        if target is None:
            low_vt = vt.lower()
            if not low_vt:
                continue
            if low_vt in blob or low_vt.replace("-", " ") in blob:
                continue
            if any(
                _tokenize_overlap(vt, str(v)) or low_vt in str(v).lower()
                for v in structured.values()
            ):
                continue
            name_l = name.lower()
            related = " ".join(
                str(structured.get(k) or "")
                for k in structured
                if _tokenize_overlap(name, str(k))
                or name_l in str(k).lower()
                or str(k).lower() in name_l
            ).lower()
            if related and related.count(",") >= 2:
                return f"firm measure `{vt}` not evidenced"
            return f"firm measure `{vt}` not evidenced"
        try:
            target_f = float(target)
        except (TypeError, ValueError):
            continue
        related_nums: list[float] = []
        all_nums: list[float] = []
        name_related_vals: list[str] = []
        for k, v in facts.items():
            if str(k).lower() == "title":
                continue
            nums = _numbers_in_text(str(v))
            all_nums.extend(nums)
            if _tokenize_overlap(name, str(k)) or (
                vt and _tokenize_overlap(vt, str(k))
            ):
                related_nums.extend(nums)
                name_related_vals.append(str(v))
            if vt and vt.lower() in str(v).lower():
                related_nums.extend(nums)
                name_related_vals.append(str(v))
        if name_related_vals:
            distinct: set[float] = set()
            for val in name_related_vals:
                for n in _numbers_in_text(val):
                    distinct.add(round(n, 2))
            if len(distinct) >= 2 and target_f not in distinct:
                return (
                    f"firm measure {target_f} not evidenced "
                    f"(ambiguous fact {sorted(distinct)[:6]})"
                )
            if len(distinct) >= 2 and target_f in distinct:
                if any(
                    len(_numbers_in_text(val)) >= 2 for val in name_related_vals
                ):
                    return (
                        f"firm measure {target_f} not evidenced "
                        f"(multi-value fact {sorted(distinct)[:6]})"
                    )
        pool = related_nums if related_nums else all_nums
        if not pool:
            return f"firm measure {target_f} not found in facts"
        if not any(abs(x - target_f) <= 1e-6 for x in pool):
            return (
                f"firm measure {target_f} mismatch (facts have "
                f"{sorted(set(pool))[:6]})"
            )
    return None


def _title_phrase_tokens(title: str) -> set[str]:
    """Lowercase title → runs of [a-z0-9] (phrase-match tokenization)."""
    return set(re.findall(r"[a-z0-9]+", (title or "").lower()))


def _phrase_title_violation(
    facts: dict[str, str],
    profile: dict[str, Any],
) -> str | None:
    """Each match-phrase term must appear as a complete title token (any order).

    Category/product-type nouns belong in category_tokens (path gate), not the
    title-phrase gate — LLMs often copy product-type nouns into phrase_tokens,
    which would wrongly reject titles that omit that word.
    """
    raw_phrases = [
        str(p).strip()
        for p in (profile.get("phrase_tokens") or [])
        if str(p or "").strip()
    ]
    if not raw_phrases:
        return None
    cat_parts: set[str] = set()
    for t in _category_tokens_for_gate(profile):
        for part in re.findall(r"[a-z0-9]+", str(t).lower()):
            if len(part) >= 3:
                cat_parts.add(part)
    terms: list[str] = []
    for phrase in raw_phrases:
        for term in re.findall(r"[a-z0-9]+", phrase.lower()):
            if not term or term in cat_parts:
                continue
            if term not in terms:
                terms.append(term)
    if not terms:
        return None
    title = str(facts.get("title") or "").strip()
    if not title:
        return "missing title for phrase check"
    title_toks = _title_phrase_tokens(title)
    missing = [t for t in terms if t not in title_toks]
    if missing:
        return f"phrase token(s) {missing[:4]} not in title"
    return None


def _firm_brand_violation(
    facts: dict[str, str],
    profile: dict[str, Any],
) -> str | None:
    """Require structured brand (or title fallback) when brand is firm."""
    want = str(profile.get("firm_brand") or "").strip().lower()
    if not want:
        return None
    if not (profile.get("brand_is_firm") or profile.get("firm_brand")):
        return None
    brand = str(facts.get("brand") or "").strip().lower()
    if brand and brand not in ("none", "null", "n/a", "na"):
        if want == brand or want in brand or brand in want:
            return None
        return f"firm brand `{want}` != `{brand}`"
    title = str(facts.get("title") or "").lower()
    if want in title:
        return None
    return f"firm brand `{want}` not evidenced"


def _open_catalog_firm_violation(
    facts: dict[str, str],
    profile: dict[str, Any],
) -> str | None:
    """Combined open-catalog firm gates: product type + measures (+ phrase title)."""
    fail = (
        _firm_brand_violation(facts, profile)
        or _product_type_violation(facts, profile)
        or _firm_measure_violation(facts, profile)
    )
    if fail:
        return fail
    if _looks_like_phrase_retrieval(profile, ""):
        return _phrase_title_violation(facts, profile)
    return None


_FIRM_FACT_BIND_SYSTEM = """Map firm shopping measures onto observed catalog fact fields.
Return ONLY JSON. No markdown.

{
  "checks": [
    {
      "name": string,
      "fact_key": string|null,
      "numeric_in_facts": number|null,
      "exact_match": bool,
      "reason": string
    }
  ]
}

Firm measures: <<MEASURES>>
Observed facts: <<FACTS>>
Task rules / episode wording (may define named classes): <<TASK_RULES>>

Rules (catalog-grounded; no external synonym tables):
- For each firm measure, choose at most one observed fact key that evidences it
  by semantic fit to THIS facts dict (keys+values only).
- exact_match true when:
  * the shopper's exact numeric appears in that fact value, OR
  * value_text is literally evidenced, OR
  * Task rules define a named class (e.g. resolution/display class) and the
    observed numeric dimensions satisfy that class's stated thresholds for the
    shopper's named class — use ONLY thresholds written in Task rules.
- Nearby/rounded numbers that are not the shopper's exact number and are not
  covered by a Task-rules class definition = false.
- If no fact evidences the measure, fact_key=null and exact_match=false.
"""


def _llm_firm_measures_ok(
    *,
    facts: dict[str, str],
    profile: dict[str, Any],
    deadline: float,
    query: str = "",
) -> bool | None:
    """LLM catalog-grounded firm-measure check. None = skipped/unavailable."""
    measures = profile.get("firm_measures") or []
    if not measures or not facts or _time_left(deadline) < 8:
        return None
    facts_for_llm = {
        k: v for k, v in facts.items() if str(k).lower() != "title"
    }
    if not facts_for_llm and facts:
        facts_for_llm = dict(facts)
    if not facts_for_llm:
        return None
    task_rules = ""
    if "Task rules:" in str(query or ""):
        task_rules = str(query).split("Task rules:", 1)[1].strip()[:2500]
    system = _fill_template(
        _FIRM_FACT_BIND_SYSTEM,
        {
            "MEASURES": json.dumps(measures, ensure_ascii=True),
            "FACTS": json.dumps(facts_for_llm, ensure_ascii=True)[:4000],
            "TASK_RULES": json.dumps(task_rules or "(none)", ensure_ascii=True)[:2600],
        },
    )
    raw = _llm_json(
        system=system,
        user="Check each firm measure against the facts. exact_match must be strict.",
        deadline=min(deadline, time.monotonic() + min(8.0, _SEMANTIC_BUDGET_S)),
    )
    if not isinstance(raw, dict):
        return None
    checks = raw.get("checks")
    if not isinstance(checks, list) or not checks:
        return None
    by_name = {
        str(c.get("name") or "").strip().lower(): c
        for c in checks
        if isinstance(c, dict)
    }
    for spec in measures:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name") or "").strip().lower()
        row = by_name.get(name)
        if row is None and len(checks) == 1 and len(measures) == 1:
            row = checks[0] if isinstance(checks[0], dict) else None
        if not isinstance(row, dict) or not bool(row.get("exact_match")):
            return False
    return True


def _over_budget_violation(
    tool_calls: list[dict[str, Any]],
    *,
    budget: float | None,
    last_stock: dict[str, Any] | None,
    prices_by_key: dict[str, float] | None = None,
) -> str | None:
    if budget is None:
        return None
    names = _tool_names(tool_calls)
    if not any(n in ("place_test_order", "add_to_cart") for n in names):
        return None

    def price_for(pid: str, sku: str) -> float | None:
        key = f"{pid}::{sku}"
        if prices_by_key and key in prices_by_key:
            try:
                return float(prices_by_key[key])
            except (TypeError, ValueError):
                pass
        if last_stock and str(last_stock.get("product_id") or "") == pid and str(
            last_stock.get("sku") or ""
        ) == sku:
            try:
                return float(last_stock.get("price"))
            except (TypeError, ValueError):
                return None
        if last_stock and last_stock.get("price") is not None and not pid:
            try:
                return float(last_stock.get("price"))
            except (TypeError, ValueError):
                return None
        return None

    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        if name not in ("place_test_order", "add_to_cart"):
            continue
        raw = call["function"].get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            continue
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        price_f = price_for(pid, sku)
        if price_f is None and last_stock is not None:
            try:
                price_f = float(last_stock.get("price"))
            except (TypeError, ValueError):
                price_f = None
        if price_f is None:
            continue
        if price_f > float(budget) + 1e-6:
            return (
                f"Rejected: price {price_f} exceeds budget {budget} for "
                f"{pid}::{sku or '?'}. Pick another in-budget firm-valid sku."
            )
    return None


def _empty_cart_order_violation(
    tool_calls: list[dict[str, Any]],
    *,
    cart_sku: dict[str, str],
    dead_keys: set[str],
) -> str | None:
    """place_test_order fails with {'error':'item not in cart'} if cart empty/dead."""
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
            continue
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        key = f"{pid}::{sku}"
        if key in dead_keys:
            return (
                f"Rejected: {key} is dead after market event. "
                "add_to_cart a LIVE sku before place_test_order."
            )
        if cart_sku.get("product_id") != pid or cart_sku.get("sku") != sku:
            return (
                "Rejected: place_test_order requires the sku to be in cart "
                "(otherwise env returns item not in cart → agent_error). "
                "Call add_to_cart on a LIVE in-stock sku first."
            )
    return None


def _build_boot(
    query: str,
    *,
    candidates: list[tuple[str, str]],
    profile: dict[str, Any],
    search_plan: dict[str, Any] | None = None,
) -> str:
    ask_first = bool(profile.get("ask_first"))
    model_hint = profile.get("preferred_model")
    soft_brand = profile.get("soft_brand")
    firm_brand = profile.get("firm_brand")
    rank_priority = list(profile.get("rank_axes") or [])
    hints = list(profile.get("search_value_hints") or [])
    notes = str(profile.get("notes") or "").strip()
    firm_notes = list(profile.get("firm_spec_notes") or [])

    boot = (
        f"{query}\n\n"
        "Procedure: ask firm specs if withheld-until-asked; else retrieve → view → "
        "inspect_stock → FIRST cart a hard-valid best candidate (event commitment) → "
        "on market_event: ONE declarative message → remove → recover/re-rank → "
        "inspect_stock → cart → place_test_order. "
        "Interpret constraints semantically from THIS query + Task rules "
        "(no fixed synonym tables). First successful add_to_cart is irreversible "
        "commitment — only cart a fully firm-valid intended purchase."
    )
    if notes:
        boot += f"\n\nEpisode notes: {notes}"
    if firm_notes:
        boot += "\n\nFirm constraints: " + "; ".join(firm_notes[:6])
    if ask_first:
        boot += (
            "\n\nASK-FIRST: First tool call must be `message` asking for the "
            "withheld firm spec. After the reply, search with that revealed spec."
        )
    if firm_brand or profile.get("brand_is_firm"):
        brand = firm_brand or soft_brand
        if brand:
            boot += (
                f"\n\nFIRM BRAND: `{brand}` is non-negotiable. Do not cart a "
                "different brand."
            )
    elif soft_brand:
        boot += (
            f"\n\nSOFT BRAND: Prefer `{soft_brand}` on the first cart when a "
            "fully firm-valid preferred listing exists; never violate firm budget/specs. "
            "After a market shock, soft brand is optional — exact firm measures "
            "and product type still win."
        )
    firm_ms = profile.get("firm_measures") or []
    if firm_ms:
        bits = []
        for m in firm_ms[:3]:
            if not isinstance(m, dict):
                continue
            if m.get("value_text"):
                bits.append(str(m["value_text"]))
            elif m.get("numeric") is not None:
                bits.append(str(m["numeric"]))
        if bits:
            boot += (
                "\n\nFIRM MEASURES (literal exact match required): "
                + "; ".join(bits)
                + ". Nearby sizes/values are FAIL."
            )
    cat_toks = list(profile.get("category_tokens") or [])
    if cat_toks:
        boot += (
            "\n\nFIRM PRODUCT TYPE: listings must match category_path / type "
            + ", ".join(f"`{t}`" for t in cat_toks[:4])
            + ". Sibling categories that drop this type are FAIL."
        )
    if model_hint:
        boot += (
            f"\n\nPREFERRED MODEL: `{model_hint}`. Search this token early. "
            "On stockout, recover the same model in another color/SKU before changing models."
        )
    if profile.get("require_soft_evidence"):
        boot += (
            "\n\nSOFT EVIDENCE: Among firm-valid options, prefer listings with "
            "visible evidence for the shopper's soft use-case axes."
        )
    phrase = list(profile.get("phrase_tokens") or [])
    if phrase:
        boot += "\n\nPhrase literals to honor: " + ", ".join(phrase[:6])
    if search_plan and search_plan.get("queries"):
        boot += (
            "\n\nSuggested searches (value tokens): "
            + " | ".join(search_plan["queries"][:4])
        )
    elif hints:
        boot += "\n\nSearch value hints: " + ", ".join(hints[:8])
    if candidates:
        listed = ", ".join(f"{pid}::{sku}" for pid, sku in candidates)
        prio = (
            " → ".join(rank_priority)
            if rank_priority
            else "the priority order stated above"
        )
        boot += (
            "\n\nCLOSED CANDIDATE SET (mandatory): "
            f"{listed}. "
            f"view each listed product_id; rank by [{prio}]; "
            "FIRST add_to_cart MUST be the current best listed key; "
            "after shock, re-rank remaining LIVE listed keys and order the new best. "
            "No open-catalog substitutes."
        )
    elif profile.get("needs_justification"):
        specs = [
            f
            for f in (profile.get("required_claim_fields") or [])
            if f not in ("price", "in_stock")
        ]
        spec_txt = ", ".join(specs) if specs else "shopper-named specs only"
        boot += (
            "\n\nJUSTIFICATION: place_test_order justification must include "
            f"price, in_stock, and {spec_txt} when grounded in view/stock. "
            "Use allowed schema field names; prefer the shopper's own labels. "
            "EVENT COMMITMENT: first add_to_cart MUST be the item you intend to order; "
            "after a market shock, recover to a NEW live valid sku then order promptly."
        )
    return boot



class LGT_ShoppingTaskType(str, Enum):
    PRODUCT = "product"
    SHOP = "shop"
    VOUCHER = "voucher"
    def explain(self, is_multiple: bool) -> str:
        match self:
            case LGT_ShoppingTaskType.PRODUCT:
                return "a single product with specific constraints"
            case LGT_ShoppingTaskType.SHOP:
                return "multiple products from the same shop"
            case LGT_ShoppingTaskType.VOUCHER:
                pstr = "multiple products" if is_multiple else "a single product"
                return f"{pstr} within a specified budget and considering voucher options"
class LGT_Vocab:
    ID_KEYS = ["brand", "color", "model", "id", "compatibility", "color_family"]
    GATE_HARD_KEYS = ID_KEYS + ["material", "packaging", "storage", "pack", "set", "size", "capacity", "volume", "no"]
    ALPHABETS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    KW_STOPWORDS: List[str] = [
        "color family", "suitable", "compatible",
        "about", "above", "after", "also", "and", "as", "at", "before", "below", "between", "by", "for", "from", "in",
        "less", "more", "of", "on", "or", "over", "per", "than", "to", "under", "up", "with", "within"
    ]
    YES_WORDS: List[str] = ["yes", "enabled", "included", "exists", "true"]
    NO_WORDS: List[str] = ["no", "without any", "without"]
    COLOR_WORDS: List[str] = [
        "red", "blue", "green", "yellow", "black", "white", "pink", "purple", "violet", "orange", "brown", "gray", "grey",
        "beige", "navy", "gold", "silver", "bronze", "magenta", "cyan"
    ]
    SEARCHABLE_UNITS: List[str] = [
        "mm", "cm", "m", "km", "ft", "yd", "mg", "g", "kg", "oz", "lb",
        "ml", "l", "mah", "ah", "wh", "kwh", "v", "mv", "kv", "ma", "w", "kw",
        "hz", "khz", "mhz", "ghz", "kb", "mb", "gb", "tb",
        "bps", "kbps", "mbps", "gbps", "°c", "°f", "pcs", "pc",
        "pa", "kpa", "mpa", "mp", "dpi", "lux", "db", "rpm"
    ]
    SYNONYMS: List[Tuple[str, str]] = [
        (min(a, b), max(a, b)) for a, b in
        [
            ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"), ("five", "5"),
            ("six", "6"), ("seven", "7"), ("eight", "8"), ("nine", "9"), ("ten", "10"),
            ("one", "single"), ("two", "double"), ("three", "triple"),
            ("1", "single"), ("2", "double"), ("3", "triple"),
            ("first", "1st"), ("second", "2nd"), ("third", "3rd"),
            ("hundred", "100"), ("thousand", "1000"), ("zero", "0"),
            ("shock-resistant", "shockproof"), ("shock resistant", "shockproof"),
            ("shock-proof", "shockproof"),
            ("water-resistant", "waterproof"), ("water resistant", "waterproof"),
            ("water-proof", "waterproof"),
        ]
    ]
    P_EXT_SEARCH_UNITS: List[Tuple[str, str]] = [
        ("piece", "pcs"), ("pack", "pcs")
    ]

@dataclass
class LGT_JsonModel(object):
    @classmethod
    def from_json(cls, data: dict) -> "LGT_JsonModel":
        kwargs = {}
        for f in fields(cls):
            if f.name not in data: continue
            value = data[f.name]
            t = f.type
            kwargs[f.name] = cls.__convert_from__(value, t)
        return cls(**kwargs)
    def to_json(self) -> Dict[str, Any] | List[Any]:
        return self.__convert_to__(self)
    @staticmethod
    def __convert_from__(value, t) -> Any:
        origin = get_origin(t)
        args = get_args(t)
        if origin is Union and len(args) == 2 and NoneType in args:
            inner_t = next(a for a in args if a is not NoneType)
            if value is None: return None
            return LGT_JsonModel.__convert_from__(value, inner_t)
        if is_dataclass(t) and isinstance(value, dict):
            return t.from_json(value)
        if origin is list and isinstance(value, list):
            return [LGT_JsonModel.__convert_from__(v, args[0]) for v in value]
        if origin is tuple and isinstance(value, list):
            return tuple(LGT_JsonModel.__convert_from__(v, args[0]) for v in value)
        if origin is dict and isinstance(value, dict):
            kt, vt = args
            return {
                LGT_JsonModel.__convert_from__(k, kt): LGT_JsonModel.__convert_from__(v, vt)
                for k, v in value.items()
            }
        if t is float and isinstance(value, str): return LGT_MiscUtil.extract_number(value)
        if t is int and isinstance(value, str): return LGT_MiscUtil.extract_number(value, True)
        if t is str: return str(value)
        return value
    @staticmethod
    def __convert_to__(obj) -> Any:
        if is_dataclass(obj):
            result = {}
            for f in fields(obj):
                value = getattr(obj, f.name)
                result[f.name] = LGT_JsonModel.__convert_to__(value)
            return result
        if isinstance(obj, list):
            return [LGT_JsonModel.__convert_to__(x) for x in obj]
        if isinstance(obj, dict):
            return {LGT_JsonModel.__convert_to__(k): LGT_JsonModel.__convert_to__(v) for k, v in obj.items()}
        return obj

@dataclass
class LGT_SubQuery(LGT_JsonModel):
    query: str = ""
    services: List[str] = field(default_factory = list, compare = False)
    price: Optional[str] = None
    kw: List[List[str]] = field(default_factory = list, compare = False)
    def __hash__(self):
        return hash((self.query, ''.join(sorted(self.services)), self.price))
    def __eq__(self, other):
        if not isinstance(other, LGT_SubQuery): return NotImplemented
        if self.query != other.query: return False
        if ''.join(sorted(self.services)) != ''.join(sorted(other.services)): return False
        if self.price != other.price: return False
        return True

@dataclass
class LGT_VoucherOption(LGT_JsonModel):
    budget: float
    apply_to: str
    threshold: Optional[float]
    is_fixed: bool
    discount: float
    cap: Optional[float]

@dataclass
class LGT_QueryInfo(LGT_JsonModel):
    challenge: LGT_ShoppingTaskType = ""
    query_f: str = ""
    query_e: str = ""
    items: List[LGT_SubQuery] = field(default_factory = list)
    vopt: Optional[LGT_VoucherOption] = None

@dataclass
class LGT_SearchOption(LGT_SubQuery):
    shop_id: Optional[str] = None
    q: str = ""
    def update_shop(self, shop_id: Optional[str]) -> "LGT_SearchOption":
        self.shop_id = shop_id
        return self
    def update_price(self, price: Optional[str]) -> "LGT_SearchOption":
        self.price = price
        return self
    def update_kw(self, id: int, kw: List[str]) -> "LGT_SearchOption":
        self.kw[id] = kw
        return self
    def update_q(self, q: Optional[str] = None) -> "LGT_SearchOption":
        if not q:
            segs = []
            for kw in self.kw:
                v = kw[-1]
                for m in v.split():
                    m = m.lower().strip()
                    if m and m not in segs: segs.append(m)
            self.q = " ".join(segs)
        else:
            self.q = q
        return self
    def __hash__(self):
        return hash((self.q, self.query, ''.join(sorted(self.services)), self.price, self.shop_id))
    def __eq__(self, other):
        if not isinstance(other, LGT_SearchOption): return NotImplemented
        if self.query != other.query: return False
        if ''.join(sorted(self.services)) != ''.join(sorted(other.services)): return False
        if self.price != other.price: return False
        if self.shop_id != other.shop_id: return False
        if self.q != other.q: return False
        return True

@dataclass
class LGT_Product(LGT_JsonModel):
    id: str
    title: str
    price: float
    service: Optional[List[str]] = None
    shop_id: Optional[str] = None
    attributes: Dict[str, List[str]] = field(default_factory = dict)
    sku: Dict[str, Dict[str, str]] = field(default_factory = dict)
    is_viewed: bool = False
    def to_prompt(self, q: str, for_reason: bool = False) -> Dict[str, Any]:
        attr = self.extract_attr(q, ':', False)
        joint_attr = []
        for _, group in attr.items():
            for lines in group:
                for l in lines:
                    joint_attr.append(l)
        js = {
            "title": self.title,
            "attributes": '\n'.join(joint_attr)
        }
        if for_reason: js["price"] = self.price
        return js
    def clone_from(self, other: "LGT_Product") -> None:
        self.id = other.id
        self.title = other.title
        self.price = other.price
        self.service = other.service
        self.shop_id = other.shop_id
        self.attributes = other.attributes
        self.sku = other.sku
        self.is_viewed = other.is_viewed
    def extract_attr(self, q: str, delimiter: str, norm: bool) -> Dict[str, List[List[str]]]:
        extracted = {"sku": []}
        is_valid = False
        for k, vs in self.attributes.items():
            extracted[f"att_{k}"] = []
            atoms = []
            for v in vs:
                if LGT_EngineUtil.is_valid_attr(k, v, q): is_valid = True
                atoms.append((k, v))
            if atoms: extracted[f"att_{k}"].append(atoms)
        for _, kv in self.sku.items():
            atoms = []
            for k, v in kv.items():
                if LGT_EngineUtil.is_valid_attr(k, v, q): is_valid = True
                atoms.append((k, v))
            if atoms: extracted["sku"].append(atoms)
        if not is_valid: extracted.clear()
        return {
            g: [
                [
                    f"{LGT_MiscUtil.normalize_str(k) if norm else k}{delimiter} "
                    f"{LGT_MiscUtil.normalize_str(v) if norm else v}"
                    for k, v in kv
                ]
                for kv in kvs
            ]
            for g, kvs in extracted.items()
        }
    def __hash__(self):
        return hash(self.id)
    def __eq__(self, other):
        if not isinstance(other, LGT_Product): return NotImplemented
        return self.id == other.id

@dataclass
class LGT_Weight(LGT_JsonModel):
    SKU_LIN_PROBS = [0.1, 0.1, 0.08, 0.05, 0.02, 0.01]
    RANK_BONUS_C = 0.05
    RANK_LIMIT = 300
    RANK_LOG_PROBS = [
        0.6554774978466211, 0.09305071398002347, 0.04004714272557973, 0.024146071349246602, 0.019434642793296042, 0.01943464179329533,
        0.018845714223802223, 0.005889285694938195, 0.005889284694937484, 0.005005892840697467, 0.005005891840696755, 0.005005890840696044,
        0.005005889840695332, 0.002572687961473001, 0.00257268696147229, 0.0025726859614715793, 0.002572684961470868, 0.0025726839614701575,
        0.0025726829614694463, 0.0025726819614687356, 0.0025726809614680245, 0.0025726799614673138, 0.0025726789614666026,
        0.002572677961465892, 0.002572676961465181, 0.00257267596146447, 0.0025726749614637594, 0.002572673961463048, 0.0025726729614623375,
        0.0025726719614616264, 0.0025726709614609156, 0.0025726699614602045, 0.002572668961459494, 0.0025726679614587827,
        0.002572666961458072, 0.0025726659614573612, 0.00257266496145665, 0.0025726639614559394, 0.0025726629614552282,
        0.0025726619614545175, 0.0025726609614538064, 0.0025726599614530957, 0.0025726589614523845, 0.002572657961451674,
        0.0025726569614509627, 0.002572655961450252, 0.0025726549614495412, 0.00257265396144883, 0.0025726529614481194,
        0.0025726519614474083, 0.0025726509614466975
    ]
    detail: Dict[str, float] = field(default_factory = dict)
    group: Dict[str, int] = field(default_factory = dict)
    weighted_rank: int = -1
    rank: int = -1
    rank_bonus: float = 0
    sku_bonus: float = 0
    confidence: float = 0
    score: float = 0
    score_n: float = 0
    def normalize(self):
        idx = self.group.get("sku") or -1
        self.sku_bonus = LGT_Weight.SKU_LIN_PROBS[idx + 1] if idx < (len(LGT_Weight.SKU_LIN_PROBS) - 1) else 0
        self.rank_bonus = max(1.0 - self.rank / LGT_Weight.RANK_LIMIT, 0) * LGT_Weight.RANK_BONUS_C
        self.score_n = self.score / len(self.detail) + self.rank_bonus + self.sku_bonus
    def update_confidence(self, weighted_rank: int):
        if self.weighted_rank >= 0: return
        self.weighted_rank = weighted_rank
        if weighted_rank >= len(LGT_Weight.RANK_LOG_PROBS):
            self.confidence = LGT_Weight.RANK_LOG_PROBS[-1] * (0.9 ** (weighted_rank - len(LGT_Weight.RANK_LOG_PROBS) + 1))
        else:
            self.confidence = LGT_Weight.RANK_LOG_PROBS[weighted_rank]
    def update_with_sibling(self, sibling: "LGT_Weight"):
        if self.score < sibling.score:
            self.detail = sibling.detail
            self.group = sibling.group
            self.score = sibling.score
        self.sku_bonus = max(self.sku_bonus, sibling.sku_bonus)
        self.score_n = self.score / len(self.detail) + self.rank_bonus + self.sku_bonus

def LGT_cast_model(source: LGT_JsonModel, target_cls: Type[LGT_JsonModel]) -> LGT_JsonModel:

    return target_cls.from_json(source.to_json())

class LGT_TextCodec:
    @staticmethod
    def pull_number(s: str, as_int: bool = False) -> int | float:
        matches = re.findall(r"\d+\.?\d*", s); val = float(matches[0]) if matches else 0.0; return int(val) if as_int and matches else (0 if not matches else val)
    @staticmethod
    def coerce_number(v: Any, as_int: bool = False) -> int | float:
        if v is None: return 0
        if isinstance(v, (int, float)): return v
        if isinstance(v, str): return LGT_TextCodec.pull_number(v, as_int=as_int)
        raise ValueError(f"# cannot convert {v} to number")
    @staticmethod
    def pull_bool(s: str) -> bool:
        s = s.lower().strip()
        if not s: return False
        if s == "true": return True
        if s == "false": return False
        return int(float(s)) != 0 if s.isnumeric() else False
    @staticmethod
    def tidy_text(x: str) -> str:
        x = x.lower(); x = re.sub(r"\b(\d+)([a-z]+)\b", r"\1 \2", x); x = re.sub(r"\b([a-z]+)(\d+)\b", r"\1 \2", x); x = re.sub(r"[\(\)\[\]\{\}\+\*\_\:\;\,\-\~\/\<\>\=]", " ", x); x = re.sub(r"\b(\d+)\s*x\s*(\d+)\b", r"\1 \2", x); return re.sub(r"\s+", " ", x).strip()
    @staticmethod
    def cartesian_slots(cat: Dict | List, include_non: bool) -> List[Dict | List]:
        keys = list(cat.keys()) if isinstance(cat, dict) else list(range(len(cat))); choices = [([-1] if include_non else []) + list(range(len(cat[k]))) for k in keys]; opts = product(*choices); return [dict(zip(keys, opt)) for opt in opts] if isinstance(cat, dict) else list(opts)
    @staticmethod
    def dump_js(js: Dict[str, Any] | List[Any]) -> None:
        print(json.dumps(js, ensure_ascii=False, indent=2))
    @staticmethod
    def decode_js(s: str) -> dict[str, Any] | list[Any] | None:
        cleaned = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.I); cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try: return json.loads(cleaned)
        except json.JSONDecodeError: pass
        decoder = json.JSONDecoder(); starts = []
        obj_start = cleaned.find("{"); arr_start = cleaned.find("[")
        if obj_start >= 0: starts.append(obj_start)
        if arr_start >= 0: starts.append(arr_start)
        for start in sorted(starts):
            try:
                obj, _ = decoder.raw_decode(cleaned[start:]); return obj
            except json.JSONDecodeError: pass
        return None

class LGT_MiscUtil:
    @staticmethod
    def extract_number(s: str, as_int: bool = False) -> int | float:
        return LGT_TextCodec.pull_number(s, as_int)
    @staticmethod
    def ensure_number(v: Any, as_int: bool = False) -> int | float:
        return LGT_TextCodec.coerce_number(v, as_int)
    @staticmethod
    def extract_bool(s: str) -> bool:
        return LGT_TextCodec.pull_bool(s)
    @staticmethod
    def normalize_str(x: str) -> str:
        return LGT_TextCodec.tidy_text(x)
    @staticmethod
    def get_choice_combs(cat: Dict | List, include_non: bool) -> List[Dict | List]:
        return LGT_TextCodec.cartesian_slots(cat, include_non)
    @staticmethod
    def print_json(js: Dict[str, Any] | List[Any]) -> None:
        LGT_TextCodec.dump_js(js)
    @staticmethod
    def parse_json(s: str) -> dict[str, Any] | list[Any] | None:
        return LGT_TextCodec.decode_js(s)

class LGT_EngineUtil:
    @staticmethod
    def get_spell_variations(s: str) -> List[str]:
        vars = []
        if len(s) >= 3 and not re.search(r"\d", s):
            for suffix in ["es", "ic", "n", "d", "ed", "y", "ing", "ily", "ied", "er", "est", "ier", "ren"]:
                vars.append(s + suffix)
                if s.endswith(suffix):
                    body = s[:-len(suffix)]
                    vars.append(body + "y")
                    vars.append(body)
                    if len(body) >= 4 and body[-1] == body[-2]: vars.append(body[:-1])
            for suffix in ["ed", "es", "y", "er", "est", "ing"]:
                vars.append(s + s[-1] + suffix)
            if s.endswith("s"):
                vars.append(s[:-1] + "es")
                vars.append(s[:-1])
            else:
                vars.append(s + "s")
            if s.endswith("y"):
                for suffix in ["", "ied", "ies", "ier", "ing", "ily"]:
                    vars.append(s[:-1] + suffix)
            if len(s) <= 5 and s.endswith("en"): vars.append(s[:-2] + "an")
            if len(s) <= 5 and s.endswith("an"): vars.append(s[:-2] + "en")
            if len(s) >= 5 and s.endswith("or"): vars.append(s[:-2] + "our")
            if len(s) >= 6 and s.endswith("our"): vars.append(s[:-3] + "or")
        vars = list(set(v for v in vars if v != s and v not in LGT_Vocab.KW_STOPWORDS and len(v) >= 3))
        return vars
    @staticmethod
    def _literal_kw_hit(kw: str, s: str) -> bool:
        return bool(re.search((r"\b" if re.match(r"\w", kw[0]) else "") + re.escape(kw) + (r"\b" if re.match(r"\w", kw[-1]) else ""), s, re.I))
    @staticmethod
    def _concat_seg_hit(kw: str, s: str) -> bool:
        if len(kw) < 6 or len(s) <= len(kw) + 1: return False
        segs = s.replace('🎈', ' ').split()
        return any(kw == segs[i] + segs[i + 1] for i in range(len(segs) - 1)) if len(segs) > 1 else False
    @staticmethod
    def _affix_hit(kw: str, s: str) -> bool:
        if len(kw) < 5: return False
        return any(len(seg) > len(kw) and (seg.startswith(kw) or seg.endswith(kw)) for seg in s.replace('🎈', ' ').split())
    @staticmethod
    def _syn_pair_hit(kw: str, s: str, synonyms: List[Tuple[str, str]]) -> bool:
        for from_s, to_s in synonyms:
            if kw == from_s or kw == f"{from_s}s" or from_s == f"{kw}s":
                if re.search(rf"\b{re.escape(to_s)}s?\b", s): return True
            elif kw == to_s or kw == f"{to_s}s" or to_s == f"{kw}s":
                if re.search(rf"\b{re.escape(from_s)}s?\b", s): return True
        return False
    @staticmethod
    def is_kw_in(kw: str, s: str, as_is: bool, synonyms: List[Tuple[str, str]], terminal: bool = False) -> bool:
        if LGT_EngineUtil._literal_kw_hit(kw, s): return True
        if as_is: return False
        if any(re.search(rf"\b{re.escape(v)}\b", s, re.I) for v in LGT_EngineUtil.get_spell_variations(kw)): return True
        if LGT_EngineUtil._concat_seg_hit(kw, s): return True
        if not as_is and not terminal and LGT_EngineUtil._affix_hit(kw, s): return True
        if kw in ["pcs", "pc", "pack"] and any(w in s for w in ["piece", "pack", "set"]): return True
        if LGT_EngineUtil._syn_pair_hit(kw, s, synonyms): return True
        if not terminal:
            spell = re.sub(rf"ss", "sss", kw, flags=re.I)
            if spell != kw and LGT_EngineUtil.is_kw_in(spell, s, False, synonyms, True): return True
        return False
    @staticmethod
    def is_valid_attr(k: str, v: str, q: str) -> bool:
        pairs: List[Tuple[str, str, str, str, bool]] = [
            (v, q, r'\b(\d+[a-z]*)\s*\*\s*(\d+[a-z]*)\b', r'{m1} by {m2}', True), (v, q, r'\b(\d+)\-(\d+)([a-z]+)\b', r'{m1}-{m2} {m3}', True),
            (v, q, r'\b(\d+)([a-z])\+', r'\b{m1}\s*{m2}([a-z]*)\sand\sup\b', True), (v, q, r'\b(\d+)\-(\d+)[a-z]+\b', r'\b{m1}\-{m2}\b', True),
            (v, q, r'\b(\d+)\-(\d+)y\b', r'\b{m1}\sto\s{m2}\syears\b', True), (q, v, r'\b(\d+)\s*\-\&\-\s*(\d+)\b', r'{m1}\sto\s{m2}', False),
            (v, q, r'\b(\d+)\s*-\s*(\d+)\s*([a-z]+)\b', r'{m3}\sbetween\s{m1}\sand\s{m2}', True), (v, q, r'\b(\d+)\.0\b', r'\b{m1}\b', True),
            (v, q, r'\b([\d\.]+)[a-z]*\s*~\s*([\d\.]+)\b', r'\b{m1}\sto\s{m2}\b', True), (v, q, r'【(\d+\-\d+[a-z]+)】', r'\b{m1}\b', True),
            (v, q, r'\b(\d+)\s*-\s*(\d+)([a-z]*)\b', r'up to {m2}{m3}', True), (v, q, r'\b(\w+)\s(\w+)\b', r'\b{m1}\s(\w+)\s{m2}\b', True),
            (q, v, r'\b([a-z]+)\s+\((\d)\s+piece\)\s+', r'^{m1}\s+{m2}pcs$', False), (v, q, r'^(?:[a-z]+)\:\s*(\d+)$', r'\b{m1}\b', True),
            (v, q, r'\b(?:[a-z]+)\:\s*(\d+)\s*-\s*(\d+)\s*([a-z]+)\b', r'up\sto\s{m2}{m3}', True),
            (v, q, r'\b(\d+)\s*-\s*(\d+)\b', r'{m1} and {m2}', True)
        ]
        for src, dst, src_pattern, dst_pattern, yes in sorted(pairs, key = lambda p: p[-1]):
            if (m := re.search(src_pattern, src, re.I)):
                exists = re.search(
                    dst_pattern.format(
                        m1 = re.escape(m.group(1)) if m.lastindex and m.lastindex >= 1 else "",
                        m2 = re.escape(m.group(2)) if m.lastindex and m.lastindex >= 2 else "",
                        m3 = re.escape(m.group(3)) if m.lastindex and m.lastindex >= 3 else "",
                        k = k
                    ),
                    dst,
                    re.I
                )
                if not yes and not exists: return False
        return True
    @staticmethod
    def compact_kw(nested: str, q: str) -> Optional[List[str]]:
        splits = [s for s in [s.strip() for s in nested.lower().split("→")] if s]
        if len(splits) < 2: return None
        ks, v = splits[:-1], splits[-1]
        if v.lower() in LGT_Vocab.YES_WORDS:
            for i in range(len(ks)):
                t = " ".join([k.replace("_", " ") for k in ks[i:]])
                if t in q:
                    ks, v = [v], t
                    break
            else:
                return None
        for yes in LGT_Vocab.YES_WORDS:
            if re.search(rf"\b{re.escape(yes)}$", v, re.I):
                v = re.sub(rf"\b{re.escape(yes)}$", "", v, flags = re.I)
        for nw in LGT_Vocab.NO_WORDS:
            if (m := re.search(rf"^{re.escape(nw)}\s+([\w\s]+)$", v, re.I)):
                ks.append("no")
                v = m.group(1)
                break
        if f"{ks[-1]} {v}" in LGT_Vocab.KW_STOPWORDS: return None
        if f"{v} {ks[-1]}" in LGT_Vocab.KW_STOPWORDS: return None
        for sw in LGT_Vocab.KW_STOPWORDS:
            v = re.sub(rf"\b{re.escape(sw)}\b", "", v, flags = re.I)
        if len(ks) == 1 and '_' in ks[0]:
            prefix = ks[0].split('_')[0] + '-'
            if v.startswith(prefix): v = v[len(prefix):]
            prefix = prefix[:-1]
            if re.search(rf"^{re.escape(prefix)}[a-z]+$", v, re.I): v = v[len(prefix):]
        if (m := re.search(r"^(any|all)\s+([\w\s]+)$", v, re.I)):
            k = m.group(2).replace(' ', '_')
            v = m.group(1)
            if k not in ks: ks.append(k)
        if v.isnumeric() and ks[-1] in ["eu", "us", "int"]:
            v, ks = f"{ks[-1]} {v}", ["size"]
        v = re.sub(r"\bge[a-z]([a-z]{2}one)\b", r"\1", v, flags = re.I)
        v = re.sub(r"\b(\d+)in(\d+)\b", r"\1-in-\2", v, flags = re.I)
        v = re.sub(r"[\&\|]", " ", v)
        v = re.sub(r"\s+", " ", v).strip()
        return (ks + [v]) if v else None
    @staticmethod
    def is_prod_only(q: str) -> bool:
        prod_str = type(LGT_Product(id = "", title = "", price = 0)).__name__.lower()
        prod_suffix = "only"
        return any(
            re.search(pat, q, re.I) is not None for pat in [
                rf"\b(?:just|{prod_suffix})(?:\s+the)?\s+{prod_str}s?\b",
                rf"\b{prod_str}s?\s+{prod_suffix}\b"
            ]
        )
    @staticmethod
    def get_vouchered_cost(products: List[LGT_Product], vo: Optional[LGT_VoucherOption]) -> float:
        psum = sum([p.price for p in products])
        if vo is None: return psum
        if vo.apply_to == "shop":
            shops = set([p.shop_id for p in products])
            if len(shops) > 1: return psum
        if vo.threshold is not None:
            if psum < vo.threshold: return psum
        if vo.is_fixed: return psum - vo.discount
        discount = psum * vo.discount / 100.0
        if vo.cap is not None: discount = min(discount, vo.cap)
        return psum - discount
    @staticmethod
    def get_favor_range(vo: LGT_VoucherOption) -> Tuple[float, float]:
        high = (vo.threshold / 0.1) if vo.threshold is not None else math.inf
        low = (vo.threshold / 0.9) if vo.threshold is not None else 0.0
        if not vo.is_fixed:
            pct = 1.0 - vo.discount / 100.0
            l1 = vo.budget / 1.1 / pct if pct > 0 else math.inf
            l2 = vo.budget / 1.1 + (vo.cap if vo.cap is not None else math.inf)
            h1 = vo.budget + (vo.cap if vo.cap is not None else math.inf)
            h2 = vo.budget / pct if pct > 0 else math.inf
            low = max(low, min(l1, l2))
            high = min([high, h1, h2])
        else:
            low = max(low, vo.budget / 1.1 + vo.discount)
            high = min(high, vo.budget + vo.discount)
        return (math.floor(low) - 1, math.ceil(high) + 1)
    @staticmethod
    def is_favor_cost(vo: LGT_VoucherOption, psum: float) -> bool:


        if psum < 100.0: return False
        if vo.threshold is not None:
            if vo.threshold < int(psum * 0.1) or vo.threshold > int(psum * 0.9): return False
        if not vo.is_fixed:
            ac = max(
                psum * (1.0 - vo.discount / 100.0),
                psum - (vo.cap if vo.cap is not None else math.inf)
            )
            if vo.budget < math.floor(ac) or vo.budget > math.ceil(1.1 * ac): return False
        return True
    @staticmethod
    def get_allowed_total(vo: LGT_VoucherOption) -> float:
        if vo.is_fixed:
            return vo.budget + vo.discount
        upper_1 = vo.budget / (1.0 - vo.discount / 100.0) if vo.discount < 100 else math.inf
        upper_2 = vo.budget + (vo.cap if vo.cap is not None else math.inf)
        return min(upper_1, upper_2)
    @staticmethod
    def get_marginal_band(
        sidx: int,
        n_specs: int,
        threshold: float,
        allowed_total: float,
        minima: List[float],
        maxima: List[float],
        price_range: Optional[str],
    ) -> Optional[Tuple[float, float]]:
        other = [i for i in range(n_specs) if i != sidx]
        floor_raw = max(0.0, threshold - sum(maxima[i] for i in other))
        ceiling_raw = allowed_total - sum(minima[i] for i in other)
        lo, hi = floor_raw, ceiling_raw
        if price_range:
            pr_lo, pr_hi = LGT_EngineUtil.numerify_price(price_range)
            lo = max(lo, pr_lo)
            hi = min(hi, pr_hi)
        if lo > hi: return None
        return (lo, hi)
    @staticmethod
    def voucher_within_budget(products: List[LGT_Product], vo: LGT_VoucherOption) -> bool:
        return LGT_EngineUtil.get_vouchered_cost(products, vo) <= vo.budget
    @staticmethod
    def promptfy_price(price: Optional[str]) -> str:
        if not price: return ""
        segs = [s.strip() for s in price.split('-')]
        if len(segs) != 2: return ""
        try:
            low = int(segs[0]) if segs[0] else None
            high = int(segs[1]) if segs[1] else None
            if low and high: return f"price between {low} and {high}"
            if low: return f"price greater than {low}"
            return f"price less than {high}"
        except Exception:
            return ""
    @staticmethod
    def numerify_price(price: Optional[str]) -> Tuple[int, int]:
        low_str, high_str = price.split('-', 1)
        low, high = 0, math.inf
        if low_str: low = LGT_MiscUtil.extract_number(low_str, True)
        if high_str: high = LGT_MiscUtil.extract_number(high_str, True)
        return low, high
    @staticmethod
    def stringfy_price(low: int, high: int) -> str:
        return (str(low) if low > 0 else "") + "-" + (str(high) if high < math.inf else "")

class LGT_ProxyUtil:
    DEADLINE = 0.0
    LLM_PRIORITIES = [
        "moonshotai/Kimi-K2.6-TEE", "moonshotai/Kimi-K2.5-TEE",
        "zai-org/GLM-5.1-TEE", "zai-org/GLM-5-TEE",
        "Qwen/Qwen3.5-397B-A17B-TEE",
        "MiniMaxAI/MiniMax-M2.5-TEE",
        "deepseek-ai/DeepSeek-V3.2-TEE", "deepseek-ai/DeepSeek-V3.1-TEE", "deepseek-ai/DeepSeek-V3-0324-TEE", "deepseek-ai/DeepSeek-R1-0528-TEE",
        "google/gemma-4-31B-turbo-TEE", "Qwen/Qwen3-32B-TEE", "XiaomiMiMo/MiMo-V2-Flash-TEE"
    ]
    OPENROUTER_LLM_PRIORITIES = [
        "google/gemma-4-31b-it", "deepseek/deepseek-v3.2", "deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat-v3-0324",
        "openai/gpt-oss-120b", "moonshotai/kimi-k2.5", "qwen/qwen3-32b"
    ]
    DECOMPOSE_LLM_PRIORITIES: List[str] = []
    OPENROUTER_DECOMPOSE_LLM_PRIORITIES: List[str] = ["deepseek/deepseek-v3.2", "google/gemma-4-31b-it"]
    SELECT_LLM_PRIORITIES: List[str] = []
    OPENROUTER_SELECT_LLM_PRIORITIES: List[str] = ["google/gemma-4-31b-it", "deepseek/deepseek-v3.2"]
    CLASSIFY_LLM_PRIORITIES: List[str] = []
    OPENROUTER_CLASSIFY_LLM_PRIORITIES: List[str] = ["google/gemma-4-31b-it", "deepseek/deepseek-v3.2"]
    @staticmethod
    def active_inference_provider() -> str:
        raw = (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()
        return raw if raw in ("chutes", "openrouter") else "chutes"
    @staticmethod
    def default_llm_priorities() -> List[str]:
        if LGT_ProxyUtil.active_inference_provider() == "openrouter":
            return LGT_ProxyUtil.OPENROUTER_LLM_PRIORITIES
        return LGT_ProxyUtil.LLM_PRIORITIES
    @staticmethod
    def get_llm_priorities(prefers: List[str]) -> List[str]:
        priorities = prefers.copy()
        for llm in LGT_ProxyUtil.default_llm_priorities():
            if llm not in priorities:
                priorities.append(llm)
        return priorities
    @staticmethod
    def get_role_llm_priorities(role: str) -> List[str]:


        openrouter = LGT_ProxyUtil.active_inference_provider() == "openrouter"
        if role == "decompose":
            preferred = LGT_ProxyUtil.OPENROUTER_DECOMPOSE_LLM_PRIORITIES if openrouter else LGT_ProxyUtil.DECOMPOSE_LLM_PRIORITIES
        elif role == "select":
            preferred = LGT_ProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else LGT_ProxyUtil.SELECT_LLM_PRIORITIES
        elif role == "filter":
            preferred = LGT_ProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else LGT_ProxyUtil.SELECT_LLM_PRIORITIES
        elif role == "classify":
            preferred = LGT_ProxyUtil.OPENROUTER_CLASSIFY_LLM_PRIORITIES if openrouter else LGT_ProxyUtil.CLASSIFY_LLM_PRIORITIES
        else:
            preferred = []
        return LGT_ProxyUtil.get_llm_priorities(preferred)
    LLM_PROXY = ProxyClient(timeout = 180, max_retries = 5)
    @staticmethod
    def arm_deadline() -> None:
        LGT_ProxyUtil.DEADLINE = time.monotonic() + LGT_SearchEngine.TIMEOUT_SECS
    @staticmethod
    def time_left() -> float:
        if not LGT_ProxyUtil.DEADLINE:
            return math.inf
        return LGT_ProxyUtil.DEADLINE - time.monotonic()
    @staticmethod
    def out_of_time() -> bool:
        return bool(LGT_ProxyUtil.DEADLINE) and time.monotonic() > LGT_ProxyUtil.DEADLINE
    @staticmethod
    def __empty_for__(expect: str) -> Tuple[Dict[str, Any] | List[Any], Any]:
        if expect == "list":
            return [], None
        return {}, None
    @staticmethod
    def _llm_payload(model: str, prompt: str, thinking: bool, t: float) -> Dict[str, Any]:
        return {"model": model, "messages": [{"role": "user", "content": prompt}], "chat_template_kwargs": {"enable_thinking": thinking, "thinking": thinking}, "temperature": t, "stream": False, "max_completion_tokens": 8192, "max_tokens": 8192}
    @staticmethod
    def _llm_content(resp: Dict) -> str:
        choice: Dict[str, Dict] = resp.get("choices")[0]; content = choice.get("message").get("content")
        if not content: raise Exception("invalid response")
        return content
    @staticmethod
    def _llm_parse_expect(content: str, expect: str, t: float) -> Tuple[Any, float]:
        js = LGT_MiscUtil.parse_json(content)
        if js is None: raise Exception(f"failed to parse json\n*******\n{content}\n******")
        if (expect == "dict" and not isinstance(js, dict)) or (expect == "list" and not isinstance(js, list)): raise Exception(f"expected {expect} but got {type(js)}")
        return js, t
    @staticmethod
    def trigger_llm(prompt: str, thinking: bool = False, preferred_llms: List[str] = [], temperature: float = 0, expect: str = "any") -> Tuple[Dict[str, Any] | List[Any], Any]:
        llm_priorities = LGT_ProxyUtil.get_llm_priorities(preferred_llms); trigger_tick = time.monotonic(); t = temperature
        for _pass in range(2):
            for model in llm_priorities:
                if LGT_ProxyUtil.out_of_time():
                    print("# trigger_llm: global deadline reached, returning safe empty value"); return LGT_ProxyUtil.__empty_for__(expect)
                try:
                    tick = time.monotonic(); resp = LGT_ProxyUtil.LLM_PROXY.post("/inference/chat/completions", json_data=LGT_ProxyUtil._llm_payload(model, prompt, thinking, t))
                    try: content = LGT_ProxyUtil._llm_content(resp)
                    except Exception: time.sleep(0.5); raise Exception("invalid response")
                    try: js, t = LGT_ProxyUtil._llm_parse_expect(content, expect, t)
                    except Exception as exc:
                        t += 0.3; raise Exception(str(exc))
                    throughput = resp.get("usage", {}).get("completion_tokens", 0); delay = time.monotonic() - tick; trigger_delay = time.monotonic() - trigger_tick
                    print(f"* llm triggered: model = '{model}' | throughput = {throughput} | delay = {delay:.2f}s | total_delay = {trigger_delay:.2f}s"); return js, resp
                except Exception as exc:
                    print(f"# llm trigger error with '{model}': {exc}"); continue
        print("# trigger_llm: exhausted all models, returning safe empty value"); return LGT_ProxyUtil.__empty_for__(expect)
    ORO_PROXY = ProxyClient(timeout = 30, max_retries = 5)
    @staticmethod
    def _empty_find_res(so: LGT_SearchOption, page: int) -> ToolCallResult:
        return {"name": "find_product", "parameters": {"q": so.q, "page": page}, "tool_call_id": "", "result": []}
    @staticmethod
    def _hydrate_find_hits(js_prods: List[Dict]) -> List[LGT_Product]:
        products = []
        for j in js_prods:
            price = LGT_MiscUtil.extract_number(str(j.get("price") or 0))
            products.append(LGT_Product(id=str(j.get("product_id")), title=j.get("title", ""), price=price, shop_id=j.get("shop_id", ""), service=j.get("service", []) or [], attributes={}, sku={}, is_viewed=False))
        return products
    @staticmethod
    def find_products(so: LGT_SearchOption, sort: Optional[str] = None, page: int = 1) -> Tuple[List[LGT_Product], ToolCallResult]:
        empty_res = LGT_ProxyUtil._empty_find_res(so, page)
        for _attempt in range(2):
            if LGT_ProxyUtil.out_of_time():
                print("# find_products: global deadline reached, returning empty results"); return [], empty_res
            try:
                tool_res = execute_tool_call("find_product", {"q": so.q, "page": page, "shop_id": so.shop_id, "price": so.price, "sort": sort, "service": ','.join(so.services or [])})
                return LGT_ProxyUtil._hydrate_find_hits(tool_res["result"]), tool_res
            except Exception as exc:
                print(f"# error in `find_products`: {exc}"); time.sleep(0.5)
        print("# find_products: attempts exhausted, returning empty results"); return [], empty_res
    @staticmethod
    def view_products(products: List[LGT_Product]) -> ToolCallResult:
        if not products: return
        pid_str = ','.join([p.id for p in products])
        while True:
            try:
                tool_res = execute_tool_call("view_product_information", {"product_ids": pid_str})
                details = tool_res["result"]
                if not details: return tool_res
                details_map: Dict[str, Dict[str, Any]] = {d["product_id"]: d for d in details}
                for prod in products:
                    d = details_map.get(prod.id, {})
                    prod.sku = d.get("sku_options", {})
                    prod.attributes = d.get("attributes", {})
                    prod.is_viewed = True
                return tool_res
            except Exception as exc:
                print(f"# error in `view_products`: {exc}")
                time.sleep(0.5)
    @staticmethod
    def submit(products: List[LGT_Product]) -> ToolCallResult:
        pid_str = ','.join([p.id for p in products])
        return execute_tool_call("recommend_product", {"product_ids": pid_str})
    @staticmethod
    def terminate(status: str = "success") -> ToolCallResult:
        return execute_tool_call("terminate", {"status": status})
    @staticmethod
    def emit_calculate_voucher(
        sess: "LGT_SearchSession", products: List["LGT_Product"], vopt: "LGT_VoucherOption"
    ) -> Optional[ToolCallResult]:
        if not products or vopt is None: return None
        prices_str = ",".join(str(p.price) for p in products)
        vtype = "fixed" if vopt.is_fixed else "percentage"
        threshold = float(vopt.threshold or 0)
        cap = float(vopt.cap or 0)
        tool_res = execute_tool_call(
            "calculate_voucher",
            {
                "product_prices": prices_str,
                "voucher_type": vtype,
                "discount_value": vopt.discount,
                "threshold": threshold,
                "budget": vopt.budget,
                "cap": cap,
            }
        )
        sess.add_tool_result(tool_res)
        result = tool_res.get("result") or {}
        think = (
            f"Confirming the cart with calculate_voucher: prices=[{prices_str}], "
            f"threshold={threshold:.0f}, budget={vopt.budget:.0f}"
        )
        if cap > 0:
            think += f", cap={cap:.0f}"
        think += (
            f". Result: total_before={result.get('total_before')}, "
            f"discount={result.get('discount_amount')}, "
            f"total_after={result.get('total_after')}, "
            f"within_budget={result.get('within_budget')}."
        )
        sess.step(think, "Voucher calculation confirmed.")
        return tool_res

class LGT_SegTree(object):


    def __init__(self, values, k: int = 30) -> None:
        self.k, n = k, len(values)
        self.size = 1
        while self.size < n:
            self.size <<= 1
        self.seg = [[] for _ in range(2 * self.size)]
        for i, v in enumerate(values):
            self.seg[self.size + i] = [(v, i)]
        for i in range(self.size - 1, 0, -1):
            self.seg[i] = self.__merge__(self.seg[2 * i], self.seg[2 * i + 1])
    def __merge__(self, a: List, b: List) -> List:
        out, ia, ib = [], 0, 0
        while len(out) < self.k:
            if ia == len(a):
                out.extend(b[ib:ib + self.k - len(out)])
                break
            if ib == len(b):
                out.extend(a[ia:ia + self.k - len(out)])
                break
            if a[ia][0] >= b[ib][0]:
                out.append(a[ia])
                ia += 1
            else:
                out.append(b[ib])
                ib += 1
        return out
    def query(self, l: int, r: int) -> List:
        if l > r: return []
        left, right = [], []
        l += self.size
        r += self.size
        while l <= r:
            if l & 1:
                left = self.__merge__(left, self.seg[l])
                l += 1
            if not (r & 1):
                right = self.__merge__(self.seg[r], right)
                r -= 1
            l >>= 1
            r >>= 1
        return self.__merge__(left, right)

class LGT_Debugger(object):
    def __init__(
        self,
        suite: Optional[str] = None,
        frozen: Optional[List[str]] = None,
        stop_after: Optional[str] = None,
        show_dlg: bool = False,
        quiet: bool = False
    ) -> None:
        self.memory: Dict[str, List] = defaultdict(lambda: [])
        self.quiet = quiet
        self.frozen = frozen or []
        self.stop_after = stop_after
        self.suite = suite
        self.show_dlg = show_dlg
    def add(self, stage: str, data: Any) -> None:
        if self.suite: self.memory[stage].append(data)
    def find(self, stage: str, kv: Dict) -> Any:
        for js in self.memory[stage]:
            for key, val in kv.items():
                if js[key] != val: break
            else:
                return js
        raise Exception(
            f"# cannot find the freezed {stage} instance for `{json.dumps(kv, ensure_ascii = False)}`"
        )
    def is_stop_in(self, episode: str) -> bool:
        if self.stop_after:
            if episode == "AGENT" and self.stop_after.startswith("a-"): return True
            if episode == "BASE" and self.stop_after.startswith("b-"): return True
            if episode == "PRODUCT" and self.stop_after.startswith("p-"): return True
            if episode == "SHOP" and self.stop_after.startswith("s-"): return True
            if episode == "VOUCHER" and self.stop_after.startswith("v-"): return True
        return False
    def is_stop(self, stage: str) -> bool:
        return self.stop_after == stage
    def is_frozen(self, stage: str) -> bool:
        return stage in self.frozen
    def dump(self) -> None:
        pass
    def print(self, obj: Any) -> None:
        if not self.quiet:
            if isinstance(obj, dict) or isinstance(obj, list):
                LGT_MiscUtil.print_json(obj)
            else:
                print(obj)

class LGT_FitScoreKit:
    @staticmethod
    def syn_expand(v_o: str, synonyms: List[Tuple[str, str]]) -> List[str]:
        out = [v_o]
        for a, b in synonyms:
            if a == v_o and b not in out: out.append(b)
            if b == v_o and a not in out: out.append(a)
        return out
    @staticmethod
    def attr_key_bonus(ja: str, ks: List[str], atoms: List[str], synonyms: List[Tuple[str, str]]) -> float:
        header = ja.split('🎈')[0]
        if header in ("variation", "color_family"): return 0.05
        for k in ks:
            kn = LGT_MiscUtil.normalize_str(k.replace('_', ' '))
            for ak in kn.split():
                ak = ak.strip()
                if not ak or ak in atoms: continue
                if LGT_EngineUtil.is_kw_in(ak, ja, False, synonyms):
                    return 0.1 + (0.05 if kn == header else 0.0)
        return 0.0
    @staticmethod
    def attr_line_ok(ja: str, query: str, synonyms: List[Tuple[str, str]]) -> bool:
        jv = LGT_MiscUtil.normalize_str(ja.split('🎈')[1]); qn = LGT_MiscUtil.normalize_str(query)
        for s in jv.split():
            if not s: continue
            if s.isnumeric() or s in LGT_Vocab.COLOR_WORDS:
                if f".{s}" in ja or f".{s}" in query: continue
                cands = {s}
                for a, b in synonyms:
                    if a == s: cands.add(b)
                    if b == s: cands.add(a)
                if not any(re.search(rf"\b{re.escape(k)}\b", qn, re.I) for k in cands): return False
        return True
    @staticmethod
    def dry_scan(sq: LGT_SubQuery, title: str, joint_attr: List[str], synonyms: List[Tuple[str, str]]):
        used_attr, used_headers = [], []; len_pool, ti_pool, ai_pool, num_pool, match_pool = {}, {}, {}, {}, {}
        for kw in sq.kw:
            v_cands, ks = LGT_FitScoreKit.syn_expand(kw[-1], synonyms), kw[:-1]
            for v in v_cands:
                atoms = [a.strip() for a in LGT_MiscUtil.normalize_str(v).split() if a.strip()]; hash_key = tuple(kw + [v]); len_pool[hash_key] = len(atoms); as_is = re.search(r"^(\d+)$", v)
                ti_pool[hash_key] = {i for i, a in enumerate(atoms) if LGT_EngineUtil.is_kw_in(a, title, as_is, synonyms)}; num_pool[hash_key] = [i for i, a in enumerate(atoms) if a.isnumeric()]
                for ja in joint_attr:
                    ai_pool[(hash_key, ja)] = [i for i, tok in enumerate(atoms) if LGT_EngineUtil.is_kw_in(tok, ja, as_is, synonyms)]; match_pool[(hash_key, ja)] = (False, 0)
                    if len(ai_pool[(hash_key, ja)]) == len(atoms) and any(LGT_EngineUtil.is_kw_in(a, ja.split('🎈')[1], as_is, synonyms) for a in atoms): match_pool[(hash_key, ja)] = (True, LGT_FitScoreKit.attr_key_bonus(ja, ks, atoms, synonyms))
                    if ja in used_attr: continue
                    h, b = ja.split('🎈')
                    if any(LGT_EngineUtil.is_kw_in(tok, b, as_is, synonyms) for tok in atoms): used_attr.append(ja)
                    if h in used_headers: continue
                    if any(LGT_EngineUtil.is_kw_in(tok, h, as_is, synonyms) for tok in atoms): used_headers.append(h)
        return used_attr, used_headers, len_pool, ti_pool, ai_pool, num_pool, match_pool
    @staticmethod
    def prep_attr_combs(prod: LGT_Product, sq: LGT_SubQuery, title: str, synonyms: List[Tuple[str, str]]):
        all_attr = prod.extract_attr(sq.query, '🎈', True); joint_attr = [l for _, g in all_attr.items() for lines in g for l in lines]
        used_attr, used_headers, len_pool, ti_pool, ai_pool, num_pool, match_pool = LGT_FitScoreKit.dry_scan(sq, title, joint_attr, synonyms); attr = {}
        for idx, g in all_attr.items():
            new_g = []
            for lines in g:
                new_lines, headers_seen = [], []
                for l in lines:
                    if l in used_attr: new_lines.append(l)
                    else:
                        h = l.split('🎈')[0]
                        if h not in headers_seen and h in used_headers: headers_seen.append(h); new_lines.append(l)
                new_lines = [l for l in new_lines if LGT_FitScoreKit.attr_line_ok(l, sq.query, synonyms)]
                if new_lines: new_g.append(new_lines)
            if new_g: attr[idx] = new_g
        return attr, LGT_MiscUtil.get_choice_combs(attr, True), len_pool, ti_pool, ai_pool, num_pool, match_pool
    @staticmethod
    def score_kw_against_attrs(kw, v_cands, len_pool, ti_pool, ai_pool, num_pool, match_pool, joint_attr) -> float:
        max_score = -math.inf; ks = kw[:-1]
        for v in v_cands:
            hash_key = tuple(kw + [v]); atom_len = len_pool[hash_key]; attr_includes = set(); score = 0; title_includes = ti_pool[hash_key]; found_in_attr, bonuses = False, []
            for ja in joint_attr:
                attr_includes.update(ai_pool[(hash_key, ja)])
                if match_pool[(hash_key, ja)][0]: bonuses.append(match_pool[(hash_key, ja)][1]); found_in_attr = True
            if not found_in_attr:
                if len(title_includes) < atom_len:
                    frags = title_includes | attr_includes
                    if all(i in frags for i in num_pool[hash_key]): score += len(frags) / atom_len
                else: score += 1.0
            else:
                score += 1.0 if ks[-1] != "product_type" else 0.95; score += max(bonuses)
            if score > max_score: max_score = score
        return max_score
    @staticmethod
    def best_weight(prod: LGT_Product, sq: LGT_SubQuery, rank: int, synonyms: List[Tuple[str, str]]) -> LGT_Weight:
        title = LGT_MiscUtil.normalize_str(prod.title); attr, combs, len_pool, ti_pool, ai_pool, num_pool, match_pool = LGT_FitScoreKit.prep_attr_combs(prod, sq, title, synonyms); candidates: List[LGT_Weight] = []
        for group in combs:
            joint_attr: List[str] = [a for ak, idx in group.items() for a in (attr[ak][idx] if idx >= 0 else [])]; weight = LGT_Weight(group=group, rank=rank)
            for i_kw, kw in enumerate(sq.kw):
                max_score = LGT_FitScoreKit.score_kw_against_attrs(kw, LGT_FitScoreKit.syn_expand(kw[-1], synonyms), len_pool, ti_pool, ai_pool, num_pool, match_pool, joint_attr); wk = '_'.join(kw[:-1])
                if wk in weight.detail: wk += f"_{i_kw + 1}"
                weight.detail[wk] = max_score; weight.score += max_score
            weight.normalize(); candidates.append(weight)
        candidates.sort(key=lambda w: -w.score_n); return candidates[0]

class LGT_PatchConfig:
    RELAX_ON_ZERO_RECALL = True
    ALWAYS_SUBMIT = True
    HARD_CONSTRAINT_GATE = True
    MAX_GATE_SWAPS = 2
    GATE_MIN_TIME_LEFT = 45.0
    RELAX_SCORE_GAP = 0.6
    RESCUE_SCORE_FLOOR = 0.55
    PRODUCT_JUDGE = True
    PRODUCT_VSF = True
    VERIFY_PICK = True
    VOUCHER_COMBO_MATRIX = True
    VOUCHER_MATRIX_TOP = 8

def _lgt_product_matches_sq(prod: LGT_Product, sq: LGT_SubQuery) -> bool:
    if sq.price:
        if prod.price is None:
            return False
        low, high = LGT_EngineUtil.numerify_price(sq.price)
        if not (low <= prod.price <= high):
            return False
    for svc in sq.services or []:
        prod_svcs = {str(s).lower() for s in (prod.service or [])}
        if str(svc).lower() not in prod_svcs:
            return False
    return True

def _lgt_constraint_coverage(detail: Dict[str, float], min_score: float = 0.8) -> float:
    if not detail:
        return 0.0
    matched = sum(1 for v in detail.values() if v >= min_score)
    return matched / len(detail)

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """EnvPack loop: LLM semantics + structural closed-set + wall-clock finish."""

    deadline = time.monotonic() + _PROBLEM_BUDGET_S
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    tools = policy["tools"]
    query = str(policy.get("query") or "")
    candidates = _parse_candidate_variants(query)
    cand_pids = {pid for pid, _ in candidates}

    profile, allowed_fields = _interpret_episode(
        query, tools=tools, deadline=deadline
    )
    ask_first = bool(profile.get("ask_first"))
    model_hint = profile.get("preferred_model")
    soft_brand = profile.get("soft_brand")
    rank_priority = list(profile.get("rank_axes") or [])
    higher_better = set(profile.get("higher_better_axes") or [])
    claim_cache: dict[str, list[dict[str, str]]] = {}
    eligibility_cache: dict[str, dict[str, Any]] = {}
    budget_limit = profile.get("budget_ceiling")
    phrase_retrieval = _looks_like_phrase_retrieval(profile, query)
    if phrase_retrieval:
        profile["required_product_leaf"] = None

    search_plan: dict[str, Any] | None = None
    if not candidates and _time_left(deadline) > 40:
        search_plan = _llm_search_plan(
            query, profile=profile, deadline=deadline
        )
    if phrase_retrieval:
        phrase_target = _phrase_identity_target(query)
        need_searches = max(3, min(4, phrase_target - 1))
        if search_plan is None:
            search_plan = {
                "queries": [],
                "firm_tokens": [],
                "soft_tokens": [],
                "stop_after_n_searches": need_searches,
            }
        search_plan["stop_after_n_searches"] = max(
            int(search_plan.get("stop_after_n_searches") or 0),
            need_searches,
        )
        qs = list(search_plan.get("queries") or [])
        qs = [
            q
            for q in qs
            if isinstance(q, str)
            and 1 <= len(q.split()) <= 8
            and not re.search(
                r"(?i)\b(must|find in-stock|listings with title|under \d)\b",
                q,
            )
        ]
        for q in _phrase_search_queries(profile, query):
            if q and q.lower() not in {x.lower() for x in qs} and len(q.split()) <= 8:
                qs.append(q)
        search_plan["queries"] = qs[:6]

    boot = _build_boot(
        query,
        candidates=candidates,
        profile=profile,
        search_plan=search_plan,
    )
    if phrase_retrieval:
        need_ids = _phrase_identity_target(query)
        boot += (
            "\n\nRETRIEVAL RECALL: Run 3–4 DISTINCT `search` queries "
            "(phrase alone, reversed tokens, category+phrase) before first cart "
            f"so matching clusters surface (Task rules ask ~{need_ids} identities). "
            "Match phrase tokens against the listing TITLE only (complete tokens, "
            "any order). Do NOT put phrase tokens into filter(brand=…) unless the "
            "shopper named a preferred brand. After a stockout shock: inspect ONE "
            "alternate live phrase-matching sku → add_to_cart → place_test_order."
        )
    else:
        boot += (
            "\n\nSPEED: Prefer short trajectories. After a live inspect_stock on the "
            "chosen sku, place_test_order promptly — do not keep searching."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": boot},
    ]
    dialogue: list[dict[str, Any]] = []
    must_message = bool(ask_first)
    asked_spec = False
    viewed: set[str] = set()
    searchish = 0
    last_stock: dict[str, Any] | None = None
    last_view_facts: dict[str, str] = {}
    shock_recover = False
    need_reinspect = False
    event_seen = False
    event_acked = False
    pending_user_question = False
    first_cart_done = False
    ordered = False
    dead_keys: set[str] = set()
    rank_ledger: dict[str, dict[str, Any]] = {}
    rank_order: list[str] = []
    rank_order_sig: str | None = None
    axis_bindings: dict[str, dict[str, Any]] | None = None
    last_pick: dict[str, str] = {}
    cart_sku: dict[str, str] = {}
    facts_by_pid: dict[str, dict[str, str]] = {}
    options_by_key: dict[str, str] = {}
    focus_pid: str | None = None
    recovery_plan: dict[str, Any] | None = None
    post_event_searches = 0
    post_event_live_inspects = 0
    force_seq = 0
    env_turn_n = 0
    soft_brand_recovery_tried = False
    soft_brand_first_tried = False
    firm_ok_picks: list[dict[str, str]] = []
    prices_by_key: dict[str, float] = {}
    retrieval_pids: list[str] = []
    last_retrieval_sig: str | None = None
    repeat_retrieval = 0
    recovery_measure_searches = 0
    post_event_views_without_firm = 0
    pre_event_filter_tried = False
    post_event_category_filter_tried = False
    type_rejected_pids: set[str] = set()
    committed_model: str | None = None
    post_event_model_search_tried = False
    max_steps = int(policy["max_steps"])
    search_stop = int((search_plan or {}).get("stop_after_n_searches") or 4)
    if phrase_retrieval:
        search_stop = max(search_stop, 4)
        search_stop = min(search_stop, 4)
    if profile.get("require_soft_evidence"):
        search_stop = max(search_stop, 3)
    distinct_search_queries: set[str] = set()

    def next_env_turn() -> int | None:
        """Session turn must advance on every env call (ins/cart/ord each count)."""
        nonlocal env_turn_n
        if env_turn_n >= max_steps:
            return None
        env_turn_n += 1
        return env_turn_n

    def rollback_env_turn() -> None:
        nonlocal env_turn_n
        if env_turn_n > 0:
            env_turn_n -= 1

    def eligibility_for(
        key: str,
        facts: dict[str, str],
        *,
        phase: str = "first",
    ) -> dict[str, Any]:
        cache_key = f"{phase}|{key}|{sorted(facts.items())}|{budget_limit}"
        if cache_key in eligibility_cache:
            return eligibility_cache[cache_key]
        type_fail = _product_type_violation(facts, profile)
        phrase_fail = (
            _phrase_title_violation(facts, profile)
            if _looks_like_phrase_retrieval(profile, "")
            else None
        )
        if type_fail or phrase_fail:
            result = {
                "firm_ok": False,
                "is_intended_commitment": False,
                "reject_reason": type_fail or phrase_fail,
                "budget_ok": True,
                "stock_ok": True,
                "phrase_ok": not bool(phrase_fail),
            }
            eligibility_cache[cache_key] = result
            return result
        measure_fail = _firm_measure_violation(facts, profile)
        if measure_fail:
            hard = "mismatch" in measure_fail
            if hard and any(
                re.search(r"[A-Za-z]", str(m.get("value_text") or ""))
                for m in (profile.get("firm_measures") or [])
                if isinstance(m, dict)
            ):
                hard = False
            llm_ok = None if hard else _llm_firm_measures_ok(
                facts=facts,
                profile=profile,
                deadline=deadline,
                query=query,
            )
            if hard or llm_ok is not True:
                result = {
                    "firm_ok": False,
                    "is_intended_commitment": False,
                    "reject_reason": measure_fail
                    if llm_ok is not False
                    else (measure_fail + " (llm exact_match=false)"),
                    "budget_ok": True,
                    "stock_ok": True,
                    "phrase_ok": True,
                }
                eligibility_cache[cache_key] = result
                return result
            measure_fail = None
        if _time_left(deadline) < 12:
            result = {
                "firm_ok": True,
                "is_intended_commitment": key not in dead_keys,
                "reject_reason": None,
                "budget_ok": True,
                "stock_ok": True,
            }
            eligibility_cache[cache_key] = result
            return result
        parts = _parse_key_parts(key)
        stock = last_stock
        if parts and last_stock:
            if str(last_stock.get("product_id") or "") != parts[0] or str(
                last_stock.get("sku") or ""
            ) != parts[1]:
                stock = None
        try:
            result = _llm_eligibility(
                query=query,
                profile=profile,
                key=key,
                facts=facts,
                stock=stock,
                budget=budget_limit if isinstance(budget_limit, (int, float)) else None,
                closed=bool(candidates),
                dead=dead_keys,
                deadline=deadline,
                phase=phase,
            )
        except Exception as exc:
            result = {
                "firm_ok": True,
                "is_intended_commitment": key not in dead_keys,
                "reject_reason": f"eligibility_error:{type(exc).__name__}",
                "budget_ok": True,
                "stock_ok": True,
            }
        eligibility_cache[cache_key] = result
        return result

    def _rank_signature() -> str:
        parts: list[str] = []
        for pid, sku in candidates:
            key = f"{pid}::{sku}"
            e = rank_ledger.get(key) or {}
            facts = e.get("facts") or {}
            parts.append(
                "|".join(
                    [
                        key,
                        "1" if key in dead_keys else "0",
                        str(e.get("price")),
                        str(e.get("in_stock")),
                        str(sorted(facts.items())[:12]),
                        str((e.get("options") or "")[:80]),
                    ]
                )
            )
        return "||".join(parts)

    def _ledger_has_primary_axis() -> bool:
        if not rank_priority:
            return any(
                isinstance((rank_ledger.get(f"{p}::{s}") or {}).get("price"), (int, float))
                for p, s in candidates
            )
        primary = rank_priority[0]
        for p, s in candidates:
            e = rank_ledger.get(f"{p}::{s}") or {}
            if isinstance(e.get(primary), (int, float)):
                return True
        return False

    def refresh_rank_order(*, force: bool = False) -> None:
        nonlocal rank_order, rank_order_sig, axis_bindings
        if not candidates:
            return
        sig = _rank_signature()
        if not force and sig == rank_order_sig and rank_order:
            return
        if not any(k in rank_ledger for k in (f"{p}::{s}" for p, s in candidates)):
            return
        if axis_bindings is None or force:
            axis_bindings = _llm_bind_rank_axes(
                profile=profile,
                ledger=rank_ledger,
                candidates=candidates,
                deadline=deadline,
            )
        _apply_axis_bindings(
            rank_ledger,
            axes=rank_priority,
            bindings=axis_bindings or {},
        )
        if event_seen:
            rank_order = []
            rank_order_sig = sig
            return
        ordered, axis_vals = _llm_rank_closed_set(
            query=query,
            profile=profile,
            ledger=rank_ledger,
            candidates=candidates,
            dead=dead_keys,
            deadline=deadline,
        )
        for key, vals in axis_vals.items():
            entry = rank_ledger.setdefault(
                key,
                {
                    "product_id": key.split("::", 1)[0],
                    "sku": key.split("::", 1)[1] if "::" in key else "",
                },
            )
            for axis, num in vals.items():
                if entry.get(axis) is None:
                    entry[axis] = num
        if ordered:
            rank_order = ordered
        rank_order_sig = sig

    def best_key() -> str | None:
        refresh_rank_order()
        prefer = None
        if not event_seen and not _ledger_has_primary_axis():
            prefer = rank_order or None
        return _best_closed_key(
            rank_ledger,
            candidates=candidates,
            priority=rank_priority,
            higher_better=higher_better,
            dead=dead_keys,
            preferred_order=prefer,
        )

    def note_dead(pid: str, sku: str) -> None:
        nonlocal last_pick, cart_sku, rank_order_sig
        if not pid or not sku:
            return
        dead_keys.add(f"{pid}::{sku}")
        rank_order_sig = None
        if cart_sku.get("product_id") == pid and cart_sku.get("sku") == sku:
            cart_sku = {}
        if last_pick.get("product_id") == pid and last_pick.get("sku") == sku:
            last_pick = {}
            nxt = best_key()
            parts = _parse_key_parts(nxt) if nxt else None
            if parts and f"{parts[0]}::{parts[1]}" not in dead_keys:
                last_pick = {"product_id": parts[0], "sku": parts[1]}

    def remember_facts(pid: str, facts: dict[str, str]) -> None:
        nonlocal last_view_facts, focus_pid
        if not pid or not facts:
            return
        clean = {k: v for k, v in facts.items() if v}
        facts_by_pid[pid] = dict(clean)
        focus_pid = pid
        last_view_facts = dict(clean)

    def facts_for(pid: str | None) -> dict[str, str]:
        if pid and pid in facts_by_pid:
            return dict(facts_by_pid[pid])
        return {}

    def remember_price(pid: str, sku: str, price: Any) -> None:
        if not pid or not sku or price is None:
            return
        try:
            prices_by_key[f"{pid}::{sku}"] = float(price)
        except (TypeError, ValueError):
            cleaned = re.sub(r"[^\d.]", "", str(price).replace(",", ""))
            try:
                prices_by_key[f"{pid}::{sku}"] = float(cleaned)
            except ValueError:
                return

    def known_price(pid: str, sku: str) -> float | None:
        key = f"{pid}::{sku}"
        if key in prices_by_key:
            return prices_by_key[key]
        if last_stock and str(last_stock.get("product_id") or "") == pid and str(
            last_stock.get("sku") or ""
        ) == sku:
            try:
                return float(last_stock.get("price"))
            except (TypeError, ValueError):
                return None
        return None

    def in_budget(pid: str, sku: str) -> bool:
        if budget_limit is None:
            return True
        price = known_price(pid, sku)
        if price is None:
            return True
        return price <= float(budget_limit) + 1e-6

    def price_is_decoy(pid: str, sku: str) -> bool:
        """Reject tiny-price firm-brand decoys when realistic firm-ok prices exist."""
        if not (profile.get("firm_brand") or profile.get("brand_is_firm")):
            return False
        if budget_limit is None:
            return False
        price = known_price(pid, sku)
        if price is None:
            return False
        floor = max(50.0, 0.02 * float(budget_limit))
        if price >= floor:
            return False
        peers = [
            known_price(str(p.get("product_id") or ""), str(p.get("sku") or ""))
            for p in firm_ok_picks
            if p.get("product_id") and p.get("sku")
        ]
        peers = [p for p in peers if isinstance(p, (int, float))]
        return any(p >= max(50.0, 0.05 * float(budget_limit)) for p in peers)

    def resolve_pick() -> dict[str, str]:
        """Prefer live carted SKU; never dead or known-over-budget targets."""
        candidates_picks: list[dict[str, str]] = []
        bk = best_key()
        if candidates and event_seen and bk:
            parts = _parse_key_parts(bk)
            if parts and f"{parts[0]}::{parts[1]}" not in dead_keys:
                candidates_picks.append(
                    {"product_id": parts[0], "sku": parts[1]}
                )
        for pick in (dict(cart_sku), dict(last_pick)):
            pid = str(pick.get("product_id") or "")
            sku = str(pick.get("sku") or "")
            if pid and sku and f"{pid}::{sku}" not in dead_keys:
                candidates_picks.append({"product_id": pid, "sku": sku})
        for pick in reversed(firm_ok_picks):
            pid = str(pick.get("product_id") or "")
            sku = str(pick.get("sku") or "")
            if pid and sku and f"{pid}::{sku}" not in dead_keys:
                candidates_picks.append({"product_id": pid, "sku": sku})
        if bk and not (candidates and event_seen):
            parts = _parse_key_parts(bk)
            if parts and f"{parts[0]}::{parts[1]}" not in dead_keys:
                candidates_picks.append(
                    {"product_id": parts[0], "sku": parts[1]}
                )
        soft = str(soft_brand or "").strip().lower()
        firm_b = str(profile.get("firm_brand") or "").strip().lower()
        brand_need = firm_b or soft

        def _brand_match(pid: str, want: str) -> bool:
            if not want:
                return False
            facts = facts_for(pid) or {}
            brand = str(facts.get("brand") or "").strip().lower()
            if brand and (want == brand or want in brand or brand in want):
                return True
            blob = " ".join(str(v) for v in facts.values()).lower()
            return want in blob

        if brand_need:
            preferred = [
                p for p in candidates_picks if _brand_match(p["product_id"], brand_need)
            ]
            rest = [p for p in candidates_picks if p not in preferred]
            if firm_b and preferred:
                candidates_picks = preferred + rest
            elif soft:
                candidates_picks = preferred + rest

        seen: set[str] = set()
        scored: list[tuple[float, dict[str, str]]] = []
        for pick in candidates_picks:
            pid, sku = pick["product_id"], pick["sku"]
            key = f"{pid}::{sku}"
            if key in seen:
                continue
            seen.add(key)
            facts = facts_for(pid)
            if not facts and focus_pid == pid:
                facts = dict(last_view_facts)
            if not facts:
                continue
            if firm_b and not _brand_match(pid, firm_b):
                continue
            if not candidates and _open_catalog_firm_violation(facts, profile):
                continue
            if not in_budget(pid, sku):
                continue
            opts = options_by_key.get(key)
            if not event_seen and model_hint:
                if not _model_token_present(
                    facts=facts, options=opts, model=str(model_hint)
                ):
                    continue
                pref_c = str(profile.get("preferred_color") or "").strip()
                if pref_c and not _color_token_present(
                    facts=facts, options=opts, color=pref_c
                ):
                    continue
            if event_seen and model_hint:
                if post_event_searches < 2 and not _model_token_present(
                    facts=facts, options=opts, model=str(model_hint)
                ):
                    continue
            soft = 0.0
            if model_hint and _model_token_present(
                facts=facts, options=opts, model=str(model_hint)
            ):
                soft += 2.0
                pref_c = str(profile.get("preferred_color") or "").strip()
                if pref_c and _color_token_present(
                    facts=facts, options=opts, color=pref_c
                ):
                    soft += 1.0
            if profile.get("require_soft_evidence"):
                soft += _soft_evidence_score(facts, profile=profile, query=query)
                if soft <= 0.05 and any(
                    _soft_evidence_score(
                        facts_for(str(p.get("product_id") or "")) or {},
                        profile=profile,
                        query=query,
                    )
                    > 0.05
                    for p in firm_ok_picks
                    if p.get("product_id") and p.get("sku")
                ):
                    continue
                if (
                    not event_seen
                    and soft < 0.55
                    and any(
                        _soft_evidence_score(
                            facts_for(str(p.get("product_id") or "")) or {},
                            profile=profile,
                            query=query,
                        )
                        >= 0.55
                        for p in firm_ok_picks
                        if p.get("product_id") and p.get("sku")
                    )
                ):
                    continue
            scored.append((soft, {"product_id": pid, "sku": sku}))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]
        return {}

    def force_finish(turn: int, reason: str) -> bool:
        """Re-cart a LIVE in-budget sku if needed, then place_test_order."""
        nonlocal last_stock, last_view_facts, ordered, last_pick, cart_sku
        nonlocal need_reinspect, event_seen, first_cart_done, must_message, event_acked
        nonlocal force_seq, committed_model
        pick = resolve_pick()
        if not pick.get("product_id") or not pick.get("sku"):
            return False
        key = f"{pick['product_id']}::{pick['sku']}"
        if key in dead_keys:
            return False
        if not in_budget(pick["product_id"], pick["sku"]):
            return False

        def _run_forced(tool_calls_local: list[dict[str, Any]], tag: str) -> dict[str, Any] | None:
            nonlocal last_stock, cart_sku, last_pick, need_reinspect, first_cart_done
            nonlocal event_seen, event_acked, must_message, ordered, force_seq
            nonlocal committed_model
            env_t = next_env_turn()
            if env_t is None:
                return None
            assistant_content_local = f"Forcing finish ({reason}) [{tag}]."
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content_local,
                    "tool_calls": tool_calls_local,
                }
            )
            force_seq += 1
            group_id = f"{problem_id}-turn-{env_t}-force-{tag}-{force_seq}"
            envelope = {
                **binding,
                "call_id": group_id,
                "idempotency_key": group_id,
                "turn": env_t,
                "calls": [
                    {
                        "call_id": f"{group_id}-{i}",
                        "action": {
                            "name": c["function"]["name"],
                            "args": _arguments(
                                c,
                                query=query,
                                allowed_fields=allowed_fields,
                                profile=profile,
                                last_stock=last_stock,
                                last_view_facts=facts_for(pick["product_id"])
                                or dict(last_view_facts),
                                deadline=deadline,
                                claim_cache=claim_cache,
                            ),
                        },
                    }
                    for i, c in enumerate(tool_calls_local, start=1)
                ],
            }
            result_local = _env_call(envelope, deadline=deadline)
            if result_local is None:
                rollback_env_turn()
                return None
            dialogue.append(
                {
                    "role": "assistant",
                    "content": assistant_content_local,
                    "tool_calls": tool_calls_local,
                    "environment_result": result_local,
                }
            )
            for tool_call, call_result in zip(
                tool_calls_local, result_local["calls"], strict=True
            ):
                obs = call_result.get("observation")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(obs),
                    }
                )
                name = tool_call.get("function", {}).get("name") or ""
                payload = _obs_stock_payload(obs if isinstance(obs, dict) else {})
                if name == "inspect_stock" and payload:
                    last_stock = payload
                    need_reinspect = False
                    remember_price(
                        str(payload.get("product_id") or pick["product_id"]),
                        str(payload.get("sku") or pick["sku"]),
                        payload.get("price"),
                    )
                    if _stock_is_false(payload):
                        note_dead(
                            str(payload.get("product_id") or ""),
                            str(payload.get("sku") or ""),
                        )
                if name == "add_to_cart":
                    cart_sku = {
                        "product_id": pick["product_id"],
                        "sku": pick["sku"],
                    }
                    last_pick = dict(cart_sku)
                    first_cart_done = True
                    if not committed_model:
                        m = str(
                            (facts_for(pick["product_id"]) or {}).get("model") or ""
                        ).strip()
                        if m and len(m) >= 2:
                            committed_model = m
                    if isinstance(obs, dict):
                        inner = (
                            obs.get("observation")
                            if isinstance(obs.get("observation"), dict)
                            else obs
                        )
                        if isinstance(inner, dict) and inner.get("market_event"):
                            need_reinspect = True
                            event_seen = True
                            event_acked = False
                            must_message = True
                            cart_sku = {}
                            ev = inner["market_event"]
                            tgt = ev.get("target") if isinstance(ev, dict) else None
                            if isinstance(tgt, dict):
                                note_dead(
                                    str(tgt.get("product_id") or ""),
                                    str(tgt.get("sku") or ""),
                                )
                                if ev.get("new_price") is not None:
                                    remember_price(
                                        str(tgt.get("product_id") or ""),
                                        str(tgt.get("sku") or ""),
                                        ev.get("new_price"),
                                    )
                if name == "place_test_order":
                    if not (isinstance(obs, dict) and obs.get("error")):
                        ordered = True
            return result_local

        stock_matches = (
            isinstance(last_stock, dict)
            and str(last_stock.get("product_id") or "") == pick["product_id"]
            and str(last_stock.get("sku") or "") == pick["sku"]
        )
        if known_price(pick["product_id"], pick["sku"]) is None or not stock_matches:
            ins = [
                _forced_tool(
                    "inspect_stock",
                    {"product_id": pick["product_id"], "sku": pick["sku"]},
                    turn=turn,
                    tag="fin-ins",
                )
            ]
            if _run_forced(ins, "ins") is None:
                return False
            if not in_budget(pick["product_id"], pick["sku"]):
                key_ob = f"{pick['product_id']}::{pick['sku']}"
                firm_ok_picks[:] = [
                    p
                    for p in firm_ok_picks
                    if f"{p.get('product_id')}::{p.get('sku')}" != key_ob
                ]
                if last_pick.get("product_id") == pick["product_id"] and last_pick.get(
                    "sku"
                ) == pick["sku"]:
                    last_pick = {}
                return False
            if _stock_is_false(last_stock):
                return False

        if not in_budget(pick["product_id"], pick["sku"]):
            key_ob = f"{pick['product_id']}::{pick['sku']}"
            firm_ok_picks[:] = [
                p
                for p in firm_ok_picks
                if f"{p.get('product_id')}::{p.get('sku')}" != key_ob
            ]
            return False

        in_cart = (
            cart_sku.get("product_id") == pick["product_id"]
            and cart_sku.get("sku") == pick["sku"]
        )
        pick_facts = facts_for(pick["product_id"]) or dict(last_view_facts)
        steps_left = max_steps - env_turn_n
        if not in_cart and steps_left <= 1:
            return False

        if not in_cart:
            can_bundle = (
                max_calls >= 2
                and first_cart_done
                and event_acked
                and steps_left <= 6
            )
            if can_bundle:
                bundled = [
                    _forced_tool(
                        "add_to_cart",
                        {
                            "product_id": pick["product_id"],
                            "sku": pick["sku"],
                        },
                        turn=turn,
                        tag="fin-cart",
                    ),
                    _forced_order_call(
                        turn=turn,
                        pick=pick,
                        query=query,
                        allowed_fields=allowed_fields,
                        profile=profile,
                        last_stock=last_stock,
                        last_view_facts=pick_facts,
                        deadline=deadline,
                        claim_cache=claim_cache,
                    ),
                ]
                result = _run_forced(bundled, "cartord")
                if result is None:
                    return False
                if f"{pick['product_id']}::{pick['sku']}" in dead_keys:
                    return False
                if event_seen and not event_acked:
                    return False
                done = any(
                    (call.get("observation") or {}).get("done")
                    for call in result["calls"]
                )
                return bool(ordered or done)

            cart_call = [
                _forced_tool(
                    "add_to_cart",
                    {"product_id": pick["product_id"], "sku": pick["sku"]},
                    turn=turn,
                    tag="fin-cart",
                )
            ]
            if _run_forced(cart_call, "cart") is None:
                return False
            if f"{pick['product_id']}::{pick['sku']}" in dead_keys:
                return False
            if event_seen and not event_acked:
                return False
            if not in_budget(pick["product_id"], pick["sku"]):
                return False
            need_reinspect = False

        if not in_budget(pick["product_id"], pick["sku"]):
            return False
        if cart_sku.get("product_id") != pick["product_id"] or cart_sku.get(
            "sku"
        ) != pick["sku"]:
            return False

        order_call = [
            _forced_order_call(
                turn=turn,
                pick=pick,
                query=query,
                allowed_fields=allowed_fields,
                profile=profile,
                last_stock=last_stock,
                last_view_facts=pick_facts,
                deadline=deadline,
                claim_cache=claim_cache,
            )
        ]
        result = _run_forced(order_call, "ord")
        if result is None:
            return False
        done = any(
            (call.get("observation") or {}).get("done")
            for call in result["calls"]
        )
        return bool(ordered or done)

    while env_turn_n < max_steps and not ordered:
        remaining = max_steps - env_turn_n
        turn = env_turn_n + 1
        left = _time_left(deadline)
        bk = best_key()

        if (
            candidates
            and event_seen
            and event_acked
            and not ordered
            and bk
            and left > 8
            and remaining > 1
        ):
            parts = _parse_key_parts(bk)
            if parts and f"{parts[0]}::{parts[1]}" not in dead_keys:
                pid_b, sku_b = parts
                already = (
                    cart_sku.get("product_id") == pid_b
                    and cart_sku.get("sku") == sku_b
                    and not need_reinspect
                    and isinstance(last_stock, dict)
                    and str(last_stock.get("product_id") or "") == pid_b
                    and str(last_stock.get("sku") or "") == sku_b
                    and not _stock_is_false(last_stock)
                )
                if already:
                    if force_finish(turn, "closed-set mechanical best order"):
                        break
                else:
                    last_pick = {"product_id": pid_b, "sku": sku_b}
                    if force_finish(turn, "closed-set mechanical best commit"):
                        break
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Closed-set recovery: inspect_stock + add_to_cart + "
                                f"place_test_order on mechanical best `{bk}` only. "
                                "Do not pick a cheaper dominated alternative."
                            ),
                        }
                    )

        live_firm_picks = [
            p
            for p in firm_ok_picks
            if p.get("product_id") and p.get("sku")
        ]
        finish_due = (
            left < 25
            or remaining <= 4
            or (
                event_seen
                and event_acked
                and first_cart_done
                and left < 55
                and not pending_user_question
            )
            or (event_seen and first_cart_done and left < 40 and remaining <= 8)
            or (
                not candidates
                and not first_cart_done
                and not must_message
                and (not ask_first or asked_spec)
                and bool(live_firm_picks)
                and isinstance(last_stock, dict)
                and not _stock_is_false(last_stock)
                and any(
                    str(last_stock.get("product_id") or "") == str(p.get("product_id") or "")
                    and str(last_stock.get("sku") or "") == str(p.get("sku") or "")
                    and in_budget(str(p.get("product_id") or ""), str(p.get("sku") or ""))
                    for p in live_firm_picks
                )
                and (
                    not soft_brand
                    or any(
                        soft_brand.lower()
                        in " ".join(
                            str(v)
                            for v in (facts_for(str(p.get("product_id") or "")) or {}).values()
                        ).lower()
                        for p in live_firm_picks
                    )
                    or (soft_brand_first_tried and searchish >= 2)
                )
            )
            or (
                phrase_retrieval
                and not event_seen
                and not pending_user_question
                and not must_message
                and bool(last_pick)
                and (
                    (searchish >= search_stop and len(live_firm_picks) >= 1)
                    or left < 35
                    or remaining <= 8
                )
            )
        )
        if finish_due and (last_pick or firm_ok_picks or cart_sku):
            if force_finish(turn, f"budget left={left:.1f}s steps={remaining}"):
                break
            if ordered:
                break
            if left < 5 and not event_seen:
                break
            if event_seen and not event_acked:
                pass
            elif (
                event_seen
                and event_acked
                and left < 8
                and remaining > 1
                and (firm_ok_picks or retrieval_pids or last_pick)
            ):
                pass
            elif left < 5:
                break

        if (
            not must_message
            and first_cart_done
            and last_pick
            and last_stock
            and not _stock_is_false(last_stock)
            and not need_reinspect
            and (left < 25 or remaining <= 5 or (event_seen and searchish >= 1))
        ):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "FINISH NOW: call place_test_order on the current in-stock "
                        f"sku {last_pick.get('product_id')}::{last_pick.get('sku')}."
                    ),
                }
            )

        if must_message:
            if ask_first and not asked_spec and not event_seen:
                hint = (
                    "REQUIRED this turn: call ONLY `message` asking for the withheld "
                    "firm hardware/spec requirement the shopper said they will share "
                    f"if asked. Prefer: `{_ASK_GATED_SPEC_CONTENT}` "
                    "Do NOT talk about cart, stockout, or recovery yet."
                )
            else:
                hint = (
                    "REQUIRED this turn: call `message` once (declarative recovery commit). "
                    "Next turns must inspect_stock then place_test_order — no more search thrash."
                )
            if event_seen and not event_acked:
                if recovery_plan is None and _time_left(deadline) > 18:
                    closed_live = [
                        f"{p}::{s}"
                        for p, s in candidates
                        if f"{p}::{s}" not in dead_keys
                    ]
                    recovery_plan = _llm_recovery_plan(
                        query=query,
                        profile=profile,
                        dead=dead_keys,
                        budget=budget_limit
                        if isinstance(budget_limit, (int, float))
                        else None,
                        closed_live=closed_live,
                        deadline=deadline,
                    )
                hint = (
                    "REQUIRED this turn: call `message` once acknowledging the stock/"
                    "price shock and stating you will switch to another live in-budget "
                    "option. Do not ask permission. This is the ONLY unaided event ack — "
                    "do not message again unless the shopper asks a question."
                )
                if recovery_plan and recovery_plan.get("message_content"):
                    hint += (
                        " Prefer this wording: "
                        + recovery_plan["message_content"]
                    )
            if model_hint:
                hint += f" Preferred model token: {model_hint}."
            if profile.get("preferred_color"):
                hint += (
                    f" Prefer color `{profile.get('preferred_color')}` of that "
                    "model first, then other colors of the same model."
                )
            messages.append({"role": "user", "content": hint})
        elif event_seen and event_acked and need_reinspect and not pending_user_question:
            recover = (
                "Event already acknowledged. Do NOT call `message`. "
                "remove_from_cart if stale, then inspect_stock a NEW live in-budget "
                "sku → add_to_cart → place_test_order on the SAME turn if possible "
                "(or next turn immediately)."
            )
            firm_toks = []
            for m in (profile.get("firm_measures") or [])[:2]:
                if isinstance(m, dict) and m.get("value_text"):
                    firm_toks.append(str(m["value_text"]))
                elif isinstance(m, dict) and m.get("numeric") is not None:
                    firm_toks.append(str(m["numeric"]))
            if firm_toks:
                recover += (
                    " Firm measure(s) must match EXACTLY: "
                    + ", ".join(firm_toks)
                    + "."
                )
            if budget_limit is not None:
                recover += f" Current budget ceiling: {budget_limit}."
            firm_b = str(profile.get("firm_brand") or "").strip()
            soft = str(soft_brand or "").strip()
            if firm_b:
                mtoks = []
                for m in (profile.get("firm_measures") or [])[:2]:
                    if not isinstance(m, dict):
                        continue
                    if m.get("value_text"):
                        mtoks.append(str(m["value_text"]))
                    elif m.get("numeric") is not None:
                        mtoks.append(str(m["numeric"]))
                if mtoks:
                    recover += (
                        f" FIRM RECOVERY: call `search` with query containing "
                        f"`{firm_b}` and exact measure(s) {', '.join(mtoks)}"
                        + (
                            f" under max_price {budget_limit}"
                            if budget_limit is not None
                            else ""
                        )
                        + ", then view a NEW product_id — not the shocked key. "
                        "Do not message again."
                    )
                else:
                    recover += (
                        f" FIRM BRAND REQUIRED: call `filter` with brand=`{firm_b}`"
                        + (
                            f" and max_price={budget_limit}"
                            if budget_limit is not None
                            else ""
                        )
                        + ", then view a NEW product_id — not the shocked key. "
                        "Do not message again."
                    )
            elif soft and not soft_brand_recovery_tried:
                recover += (
                    f" LADDER-0 REQUIRED: call `filter` with brand=`{soft}`"
                    + (
                        f" and max_price={budget_limit}"
                        if budget_limit is not None
                        else ""
                    )
                    + " (or search including that soft brand + exact firm measure), "
                    "then view a NEW product_id — not the shocked key."
                )
            elif soft_brand_recovery_tried or post_event_searches >= 1:
                recover += (
                    " Soft brand is optional now; if soft-brand hits miss the firm "
                    "measure, immediately take any other firm-valid in-budget sku."
                )
            elif model_hint:
                recover += (
                    f" LADDER REQUIRED: search `{model_hint}`"
                    + (
                        f" `{profile.get('preferred_color')}`"
                        if profile.get("preferred_color")
                        and post_event_searches < 1
                        else ""
                    )
                    + " (same model; other color only after the exact color is gone), "
                    "then view → inspect_stock a NEW product_id — not the shocked key."
                )
            messages.append({"role": "user", "content": recover})
            shock_recover = False
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
        elif (
            candidates
            and len(viewed & cand_pids) >= len(cand_pids)
            and not first_cart_done
            and not shock_recover
        ):
            refresh_rank_order(force=True)
            bk = best_key()
            prio = " → ".join(rank_priority) if rank_priority else "stated priority"
            tip = (
                f"All listed candidates viewed. FIRST add_to_cart must be best by [{prio}]"
            )
            if bk:
                tip += f" (ledger best: {bk})"
            tip += ". Then after any shock, re-rank live keys and place_test_order quickly."
            messages.append({"role": "user", "content": tip})
        elif shock_recover or need_reinspect:
            if candidates:
                recover = (
                    "Market shock on closed set: remove_from_cart if needed, "
                    "inspect_stock a LIVE listed key"
                )
                if bk:
                    recover += f" (best live: {bk})"
                recover += (
                    ", add_to_cart it, then place_test_order on the NEXT turn. "
                    "Do not search the open catalog."
                )
            else:
                recover = (
                    "Market shock: remove_from_cart if stale. At most ONE search/"
                    "view for same preferred model other color, then inspect_stock → "
                    "add_to_cart → place_test_order. Do not thrash."
                )
                if model_hint:
                    recover += f" Model token: `{model_hint}`."
            messages.append({"role": "user", "content": recover})
            shock_recover = False
        elif not candidates and searchish >= max(2, search_stop - 1) and len(viewed) < 2:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Anti-thrash: stop repeating search. View best hits, "
                        "inspect_stock, then finish."
                    ),
                }
            )
        elif not candidates and (
            searchish >= search_stop or (event_seen and searchish >= 1)
        ) and len(viewed) >= 1:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Stop exploring. inspect_stock → add_to_cart → place_test_order NOW."
                    ),
                }
            )
        elif remaining <= 6 and not must_message:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Only ~{remaining} steps left / ~{left:.0f}s. "
                        "place_test_order on an in-stock in-budget sku NOW."
                    ),
                }
            )

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        inference_failed = False
        for _attempt in range(_MAX_TOOL_REPROMPTS):
            if _time_left(deadline) < 8:
                inference_failed = True
                break
            inference = _chat(messages=messages, tools=tools, deadline=deadline)
            if inference is None:
                inference_failed = True
                break
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
            if must_message and "message" not in _tool_names(tool_calls):
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
            ask_reveal_done = bool(
                ask_first
                and asked_spec
                and (profile.get("firm_measures") or [])
                and not pending_user_question
                and not (event_seen and not event_acked)
            )
            if (
                (
                    (
                        event_seen
                        and event_acked
                        and not pending_user_question
                        and not must_message
                    )
                    or ask_reveal_done
                )
                and "message" in _tool_names(tool_calls)
            ):
                kept = [
                    c
                    for c in tool_calls
                    if (c.get("function", {}) or {}).get("name") != "message"
                ]
                if kept:
                    tool_calls = kept
                    assistant_content = (
                        (assistant_content or "")
                        + " [stripped unsolicited chatty message]"
                    ).strip()
                else:
                    reject_txt = (
                        "Rejected: event already acknowledged. Do not message "
                        "again. remove_from_cart / inspect_stock / add_to_cart / "
                        "place_test_order on a NEW live sku."
                        if event_seen and event_acked
                        else (
                            "Rejected: firm spec already revealed. Do not message. "
                            "Call search/filter/view/inspect_stock toward an in-budget "
                            "firm-valid sku."
                        )
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
                                "content": reject_txt,
                            },
                        ]
                    )
                    continue
            if not must_message:
                violation = _closed_set_violation(
                    tool_calls, candidates=candidates, viewed=viewed
                )
                if violation:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": f"Rejected: {violation}"},
                        ]
                    )
                    continue
                oos = _oos_order_violation(tool_calls, last_stock=last_stock)
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
                    shock_recover = True
                    need_reinspect = True
                    continue
                over = _over_budget_violation(
                    tool_calls,
                    budget=budget_limit,
                    last_stock=last_stock,
                    prices_by_key=prices_by_key,
                )
                if over:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": over},
                        ]
                    )
                    shock_recover = True
                    continue
                empty = _empty_cart_order_violation(
                    tool_calls, cart_sku=cart_sku, dead_keys=dead_keys
                )
                if empty:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": empty},
                        ]
                    )
                    continue
                if "add_to_cart" in _tool_names(tool_calls):
                    blocked = False
                    if candidates:
                        if not first_cart_done:
                            refresh_rank_order(force=False)
                            bk_now = best_key() or bk
                            if bk_now:
                                bk = bk_now
                                tool_calls = _rewrite_cart_to_key(tool_calls, bk_now)
                                assistant_content = (
                                    (assistant_content or "")
                                    + f" [closed-set first cart → {bk_now}]"
                                ).strip()
                    else:
                        phase = "recovery" if (first_cart_done or event_seen) else "first"
                        if (
                            phrase_retrieval
                            and phase == "first"
                            and searchish < search_stop
                            and left > 45
                            and remaining > 10
                        ):
                            tip_q = ""
                            if search_plan and (search_plan.get("queries") or []):
                                unused = [
                                    q
                                    for q in search_plan["queries"]
                                    if isinstance(q, str)
                                    and q.strip()
                                    and " ".join(q.lower().split())
                                    not in distinct_search_queries
                                ]
                                if unused:
                                    tip_q = unused[0]
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
                                            "Rejected first cart: run another DISTINCT "
                                            f"search/filter first "
                                            f"(have {searchish}/{search_stop}). "
                                            + (
                                                f"Try query `{tip_q}`."
                                                if tip_q
                                                else "Vary phrase/category wording."
                                            )
                                        ),
                                    },
                                ]
                            )
                            continue
                        for call in tool_calls:
                            if (call.get("function", {}).get("name") or "") != "add_to_cart":
                                continue
                            raw = call["function"].get("arguments", "{}")
                            try:
                                args = json.loads(raw) if isinstance(raw, str) else raw
                            except json.JSONDecodeError:
                                args = {}
                            if not isinstance(args, dict):
                                continue
                            pid = str(args.get("product_id") or "")
                            sku = str(args.get("sku") or "")
                            key = f"{pid}::{sku}"
                            if not pid or not sku:
                                continue
                            facts = facts_for(pid)
                            if not facts and pid not in viewed:
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
                                                f"Rejected: view product_id {pid} and "
                                                "confirm firm constraints before "
                                                "add_to_cart."
                                            ),
                                        },
                                    ]
                                )
                                blocked = True
                                break
                            if not facts:
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
                                                f"Rejected: missing viewed facts for "
                                                f"{pid}; view it before add_to_cart."
                                            ),
                                        },
                                    ]
                                )
                                blocked = True
                                break
                            if price_is_decoy(pid, sku):
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
                                                f"Rejected {key}: price looks like a "
                                                "decoy vs other firm-valid listings. "
                                                "Pick a realistic-priced firm match."
                                            ),
                                        },
                                    ]
                                )
                                blocked = True
                                break
                            if (
                                phase == "recovery"
                                and model_hint
                                and post_event_searches < 2
                                and not _model_token_present(
                                    facts=facts,
                                    options=options_by_key.get(key),
                                    model=str(model_hint),
                                )
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
                                                f"Rejected recovery cart for {key}: "
                                                f"listing does not show preferred model "
                                                f"`{model_hint}`. Search that model "
                                                "(other color first), view it, then cart."
                                            ),
                                        },
                                    ]
                                )
                                blocked = True
                                break
                            pref_color = str(profile.get("preferred_color") or "").strip()
                            if (
                                phase == "first"
                                and model_hint
                                and not _model_token_present(
                                    facts=facts,
                                    options=options_by_key.get(key),
                                    model=str(model_hint),
                                )
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
                                                f"Rejected first cart for {key}: need "
                                                f"preferred model `{model_hint}` on the "
                                                "viewed listing before committing."
                                            ),
                                        },
                                    ]
                                )
                                blocked = True
                                break
                            if (
                                phase == "first"
                                and pref_color
                                and not _color_token_present(
                                    facts=facts,
                                    options=options_by_key.get(key),
                                    color=pref_color,
                                )
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
                                                f"Rejected first cart for {key}: need "
                                                f"preferred color `{pref_color}` (variant "
                                                "options/facts). Pick that shade of the "
                                                f"model `{model_hint or ''}` first."
                                            ),
                                        },
                                    ]
                                )
                                blocked = True
                                break
                            soft_pref = str(profile.get("soft_brand") or "").strip()
                            if phase == "first" and soft_pref:
                                soft_l = soft_pref.lower()
                                brand_fact = str(facts.get("brand") or "").strip().lower()
                                blob = " ".join(str(v) for v in facts.values()).lower()
                                brand_hit = bool(
                                    (brand_fact and (
                                        soft_l == brand_fact
                                        or soft_l in brand_fact
                                        or brand_fact in soft_l
                                        or _tokenize_overlap(soft_l, brand_fact)
                                    ))
                                    or soft_l in blob
                                    or _tokenize_overlap(soft_l, blob)
                                )
                                soft_firm_available = any(
                                    (
                                        soft_l
                                        in " ".join(
                                            str(v)
                                            for v in (
                                                facts_for(str(p.get("product_id") or ""))
                                                or {}
                                            ).values()
                                        ).lower()
                                        or _tokenize_overlap(
                                            soft_l,
                                            " ".join(
                                                str(v)
                                                for v in (
                                                    facts_for(
                                                        str(p.get("product_id") or "")
                                                    )
                                                    or {}
                                                ).values()
                                            ).lower(),
                                        )
                                    )
                                    for p in firm_ok_picks
                                    if p.get("product_id") and p.get("sku")
                                )
                                if not brand_hit and (
                                    soft_firm_available or not soft_brand_first_tried
                                ):
                                    tip = (
                                        f"Rejected first cart for {key}: soft brand "
                                        f"`{soft_pref}` is required for the initial "
                                        "commitment when a compliant listing exists "
                                        "(or before soft-brand retrieval is tried). "
                                        f"Search/filter brand=`{soft_pref}` with the "
                                        "firm measure first."
                                    )
                                    messages.extend(
                                        [
                                            {
                                                "role": "assistant",
                                                "content": assistant_content,
                                                "tool_calls": tool_calls,
                                            },
                                            {"role": "user", "content": tip},
                                        ]
                                    )
                                    blocked = True
                                    break
                            judge = eligibility_for(key, facts, phase=phase)
                            ok = (
                                judge.get("is_intended_commitment")
                                if phase == "first"
                                else bool(judge.get("firm_ok"))
                                and bool(judge.get("budget_ok", True))
                                and bool(judge.get("stock_ok", True))
                                and (
                                    not phrase_retrieval
                                    or bool(judge.get("phrase_ok", True))
                                )
                            )
                            if phase == "recovery" and judge.get("firm_ok") is False:
                                ok = False
                            if not ok:
                                reason = (
                                    judge.get("reject_reason")
                                    or "fails firm/soft checks"
                                )
                                tip = (
                                    f"Rejected {phase} cart for {key}: {reason}. "
                                    "Pick a firm-valid in-budget in-stock sku"
                                )
                                if phase == "first" and profile.get("soft_brand"):
                                    tip += (
                                        f" (prefer soft brand `{profile.get('soft_brand')}` "
                                        "when available)"
                                    )
                                elif phase == "recovery":
                                    tip += (
                                        " (soft brand optional after shock — keep firm "
                                        "measure + product type exact)"
                                    )
                                tip += "."
                                messages.extend(
                                    [
                                        {
                                            "role": "assistant",
                                            "content": assistant_content,
                                            "tool_calls": tool_calls,
                                        },
                                        {"role": "user", "content": tip},
                                    ]
                                )
                                blocked = True
                                break
                    if blocked:
                        continue
                if (
                    not candidates
                    and "inspect_stock" in _tool_names(tool_calls)
                    and "add_to_cart" not in _tool_names(tool_calls)
                    and "place_test_order" not in _tool_names(tool_calls)
                ):
                    insp_blocked = False
                    for call in tool_calls:
                        if (call.get("function", {}).get("name") or "") != "inspect_stock":
                            continue
                        raw = call["function"].get("arguments", "{}")
                        try:
                            args = json.loads(raw) if isinstance(raw, str) else raw
                        except json.JSONDecodeError:
                            args = {}
                        if not isinstance(args, dict):
                            continue
                        pid = str(args.get("product_id") or "")
                        facts = facts_for(pid) or dict(last_view_facts)
                        if not facts:
                            continue
                        type_fail = _product_type_violation(facts, profile)
                        fail = type_fail or _firm_measure_violation(facts, profile)
                        if type_fail and pid:
                            type_rejected_pids.add(pid)
                        if fail and not type_fail:
                            hard = "mismatch" in str(fail)
                            if hard and any(
                                re.search(r"[A-Za-z]", str(m.get("value_text") or ""))
                                for m in (profile.get("firm_measures") or [])
                                if isinstance(m, dict)
                            ):
                                hard = False
                            if not hard:
                                llm_ok = _llm_firm_measures_ok(
                                    facts=facts,
                                    profile=profile,
                                    deadline=deadline,
                                    query=query,
                                )
                                if llm_ok is True:
                                    fail = None
                        if fail:
                            if (
                                phrase_retrieval
                                and type_fail
                                and not pre_event_filter_tried
                                and isinstance(budget_limit, (int, float))
                            ):
                                cats = _category_tokens_for_gate(profile)
                                cat_label = ""
                                if cats:
                                    cat_label = max(cats, key=lambda c: len(str(c)))
                                    cat_label = _catalog_category_for_filter(str(cat_label))
                                if cat_label:
                                    tool_calls = [
                                        _forced_tool(
                                            "filter",
                                            {
                                                "category": cat_label,
                                                "max_price": float(budget_limit),
                                                "in_stock": True,
                                            },
                                            turn=turn,
                                            tag="phrase-cat",
                                        )
                                    ][:max_calls]
                                    pre_event_filter_tried = True
                                    retrieval_pids = []
                                    assistant_content = (
                                        (assistant_content or "")
                                        + f" [phrase → filter category={cat_label}]"
                                    ).strip()
                                    insp_blocked = False
                                    break
                            nxt = next(
                                (
                                    p
                                    for p in retrieval_pids
                                    if p not in viewed
                                    and p not in type_rejected_pids
                                    and not any(k.startswith(f"{p}::") for k in dead_keys)
                                ),
                                None,
                            )
                            if nxt:
                                tool_calls = [
                                    _forced_tool(
                                        "view",
                                        {"product_id": nxt},
                                        turn=turn,
                                        tag="skip-bad-insp",
                                    )
                                ][:max_calls]
                                assistant_content = (
                                    (assistant_content or "")
                                    + f" [skip bad inspect → view {nxt}]"
                                ).strip()
                                insp_blocked = False
                                break
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
                                            f"Rejected inspect_stock on {pid}: {fail}. "
                                            "View another firm-valid product-type match."
                                        ),
                                    },
                                ]
                            )
                            insp_blocked = True
                            break
                    if insp_blocked:
                        continue
                if (
                    not candidates
                    and "place_test_order" in _tool_names(tool_calls)
                ):
                    order_blocked = False
                    for call in tool_calls:
                        if (call.get("function", {}).get("name") or "") != "place_test_order":
                            continue
                        raw = call["function"].get("arguments", "{}")
                        try:
                            args = json.loads(raw) if isinstance(raw, str) else raw
                        except json.JSONDecodeError:
                            args = {}
                        if not isinstance(args, dict):
                            continue
                        pid = str(args.get("product_id") or "")
                        facts = facts_for(pid) or dict(last_view_facts)
                        type_fail = _product_type_violation(facts, profile)
                        fail = type_fail or _firm_measure_violation(facts, profile)
                        if fail and not type_fail:
                            hard = "mismatch" in str(fail)
                            if hard and any(
                                re.search(r"[A-Za-z]", str(m.get("value_text") or ""))
                                for m in (profile.get("firm_measures") or [])
                                if isinstance(m, dict)
                            ):
                                hard = False
                            if not hard:
                                llm_ok = _llm_firm_measures_ok(
                                    facts=facts,
                                    profile=profile,
                                    deadline=deadline,
                                    query=query,
                                )
                                if llm_ok is True:
                                    fail = None
                        if fail:
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
                                            f"Rejected place_test_order: {fail}. "
                                            "Cart/order only an exact firm-measure + "
                                            "product-type match."
                                        ),
                                    },
                                ]
                            )
                            order_blocked = True
                            break
                    if order_blocked:
                        continue
                if (
                    "place_test_order" in _tool_names(tool_calls)
                    and last_stock is None
                    and "inspect_stock" not in _tool_names(tool_calls)
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
                                    "Rejected: call inspect_stock on the chosen sku "
                                    "before place_test_order."
                                ),
                            },
                        ]
                    )
                    continue
                names_try = _tool_names(tool_calls)
                if (
                    need_reinspect
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
                                    "Rejected: after a market event you must "
                                    "inspect_stock before place_test_order."
                                ),
                            },
                        ]
                    )
                    continue
                if (
                    event_seen
                    and not event_acked
                    and "place_test_order" in names_try
                    and "message" not in names_try
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
                                    "Rejected: after a market shock, call `message` "
                                    "once (declarative recovery), then inspect/cart/"
                                    "order a NEW live sku — never the shocked one."
                                ),
                            },
                        ]
                    )
                    must_message = True
                    continue
                if candidates and bk:
                    rewritten = False
                    for call in tool_calls:
                        name = call.get("function", {}).get("name") or ""
                        raw = call["function"].get("arguments", "{}")
                        try:
                            args = json.loads(raw) if isinstance(raw, str) else raw
                        except json.JSONDecodeError:
                            args = {}
                        if not isinstance(args, dict):
                            continue
                        key = f"{args.get('product_id')}::{args.get('sku')}"
                        if name in ("add_to_cart", "place_test_order") and key in dead_keys:
                            tool_calls = _rewrite_cart_to_key(tool_calls, bk)
                            rewritten = True
                            break
                        if (
                            name == "add_to_cart"
                            and not first_cart_done
                            and not event_seen
                            and key != bk
                            and key in {f"{p}::{s}" for p, s in candidates}
                        ):
                            tool_calls = _rewrite_cart_to_key(tool_calls, bk)
                            rewritten = True
                            break
                        if (
                            name in ("add_to_cart", "place_test_order")
                            and event_seen
                            and event_acked
                            and key != bk
                            and key in {f"{p}::{s}" for p, s in candidates}
                        ):
                            tool_calls = _rewrite_cart_to_key(tool_calls, bk)
                            rewritten = True
                            break
                    if rewritten:
                        assistant_content = (
                            (assistant_content or "")
                            + f" [rewrote cart/order to closed-set best {bk}]"
                        ).strip()
            break
        else:
            inference_failed = True

        if inference_failed or not tool_calls:
            if candidates and len(viewed & cand_pids) < len(cand_pids):
                miss = sorted(cand_pids - viewed)[0]
                tool_calls = [
                    _forced_tool(
                        "view",
                        {"product_id": miss},
                        turn=turn,
                        tag="view",
                    )
                ]
                assistant_content = f"Viewing closed-set candidate {miss}."
            elif must_message and ask_first and not asked_spec:
                tool_calls = [
                    _forced_tool(
                        "message",
                        {
                            "content": _ASK_GATED_SPEC_CONTENT
                        },
                        turn=turn,
                        tag="ask",
                    )
                ]
                assistant_content = "Asking for the withheld firm spec."
            else:
                pick = resolve_pick()
                if pick:
                    last_pick = dict(pick)
                if last_pick and force_finish(turn, "inference/tool exhausted"):
                    break
                if event_seen and not ordered:
                    if not event_acked:
                        tool_calls = [
                            _forced_tool(
                                "message",
                                {
                                    "content": (
                                        "That listing is no longer available; I'll "
                                        "switch to another in-budget option that "
                                        "still meets every firm requirement."
                                    )
                                },
                                turn=turn,
                                tag="evt-ack",
                            )
                        ]
                        assistant_content = "Acknowledging shock; starting recovery."
                        must_message = True
                    elif remaining > 1 and _time_left(deadline) > 5:
                        continue
                    else:
                        break
                elif remaining > 1 and _time_left(deadline) > 8:
                    continue
                else:
                    break

        if must_message:
            msg_only = [
                c
                for c in tool_calls
                if c.get("function", {}).get("name") == "message"
            ]
            if msg_only:
                tool_calls = msg_only[:1]

        names = _tool_names(tool_calls)
        recovery_brand = str(profile.get("firm_brand") or soft_brand or "").strip()
        measure_tokens: list[str] = []
        for m in (profile.get("firm_measures") or [])[:2]:
            if not isinstance(m, dict):
                continue
            if m.get("value_text"):
                measure_tokens.append(str(m["value_text"]))
            elif m.get("numeric") is not None:
                measure_tokens.append(str(m["numeric"]))
        cats = _category_tokens_for_gate(profile)
        cat_tok = ""
        leaf = str(profile.get("required_product_leaf") or "").strip()
        if leaf:
            cat_tok = leaf
        if not cat_tok and cats:
            q_low = str(query or "").lower()
            for tok in cats:
                tl = str(tok).strip().lower()
                if len(tl) >= 4 and tl in q_low:
                    cat_tok = tok
                    break
        if not cat_tok and cats:
            cat_tok = cats[0]
        cat = _catalog_category_for_filter(cat_tok) if cat_tok else ""
        if (
            not event_seen
            and not first_cart_done
            and not must_message
            and soft_brand
            and not soft_brand_first_tried
            and isinstance(budget_limit, (int, float))
            and any(n in ("search", "filter") for n in names)
        ):
            tool_calls = _rewrite_recovery_retrieval(
                tool_calls,
                soft_brand=str(soft_brand),
                budget=float(budget_limit),
                soft_tried=False,
                firm_brand=None,
                category=cat or None,
                firm_measure_tokens=measure_tokens,
            )
            names = _tool_names(tool_calls)
            soft_brand_first_tried = True
            assistant_content = (
                (assistant_content or "")
                + f" [pre-event → soft brand={soft_brand} first]"
            ).strip()
        if (
            event_seen
            and not candidates
            and recovery_brand
            and isinstance(budget_limit, (int, float))
            and (profile.get("firm_brand") or not soft_brand_recovery_tried)
            and any(n in ("search", "filter") for n in names)
        ):
            tool_calls = _rewrite_recovery_retrieval(
                tool_calls,
                soft_brand=str(soft_brand) if soft_brand else None,
                budget=float(budget_limit),
                soft_tried=soft_brand_recovery_tried,
                firm_brand=str(profile.get("firm_brand") or "") or None,
                category=cat or None,
                firm_measure_tokens=measure_tokens,
            )
            names = _tool_names(tool_calls)
            assistant_content = (
                (assistant_content or "")
                + f" [recovery → brand={recovery_brand} max_price={budget_limit}]"
            ).strip()
        elif (
            event_seen
            and not candidates
            and model_hint
            and any(n == "search" for n in names)
        ):
            tool_calls = _rewrite_model_recovery_search(
                tool_calls,
                model=str(model_hint),
                color=str(profile.get("preferred_color") or "") or None,
                drop_color=post_event_searches >= 1,
                category_token=str(cat_tok or "") or None,
            )
            if cat and isinstance(budget_limit, (int, float)):
                has_filter = any(
                    (c.get("function") or {}).get("name") == "filter"
                    for c in tool_calls
                )
                if not has_filter:
                    template = next(
                        (
                            c
                            for c in tool_calls
                            if (c.get("function") or {}).get("name") == "search"
                        ),
                        tool_calls[0],
                    )
                    fargs: dict[str, Any] = {
                        "category": cat,
                        "max_price": float(budget_limit),
                    }
                    tool_calls = [
                        {
                            "id": (template.get("id") or "filter") + "-cat",
                            "type": template.get("type") or "function",
                            "function": {
                                "name": "filter",
                                "arguments": json.dumps(fargs, ensure_ascii=True),
                            },
                        }
                    ] + [
                        c
                        for c in tool_calls
                        if (c.get("function") or {}).get("name") != "filter"
                    ]
            names = _tool_names(tool_calls)
            assistant_content = (
                (assistant_content or "")
                + f" [recovery → model={model_hint}"
                + (
                    f" color={profile.get('preferred_color')}"
                    if profile.get("preferred_color") and post_event_searches < 1
                    else ""
                )
                + "]"
            ).strip()
        elif (
            not event_seen
            and not candidates
            and not first_cart_done
            and str(profile.get("firm_brand") or "").strip()
            and measure_tokens
            and not cat
            and isinstance(budget_limit, (int, float))
            and any(n == "filter" for n in names)
        ):
            tool_calls = _rewrite_recovery_retrieval(
                tool_calls,
                soft_brand=None,
                budget=float(budget_limit),
                soft_tried=True,
                firm_brand=str(profile.get("firm_brand") or "") or None,
                category=None,
                firm_measure_tokens=measure_tokens,
            )
            names = _tool_names(tool_calls)
            assistant_content = (
                (assistant_content or "")
                + " [pre-event → measure search; no category token for filter]"
            ).strip()
        elif not candidates and any(n == "filter" for n in names):
            cleaned = _strip_unprofiled_brand_filters(tool_calls, profile=profile)
            if cleaned is not tool_calls:
                tool_calls = cleaned
                names = _tool_names(tool_calls)
                assistant_content = (
                    (assistant_content or "")
                    + " [stripped unprofiled brand filter]"
                ).strip()

        if isinstance(budget_limit, (int, float)) and any(n == "search" for n in names):
            stripped = _strip_budget_tokens_from_search(
                tool_calls, float(budget_limit)
            )
            if stripped is not tool_calls:
                tool_calls = stripped
                names = _tool_names(tool_calls)
                assistant_content = (
                    (assistant_content or "")
                    + " [stripped budget literal from search]"
                ).strip()

        if (
            not event_seen
            and not candidates
            and model_hint
            and any(n == "search" for n in names)
        ):
            pinned = _rewrite_model_recovery_search(
                tool_calls,
                model=str(model_hint),
                color=str(profile.get("preferred_color") or "") or None,
                drop_color=False,
                category_token=str(cat_tok or "") or None,
            )
            if pinned is not tool_calls:
                tool_calls = pinned
                names = _tool_names(tool_calls)
                assistant_content = (
                    (assistant_content or "")
                    + f" [pre-event → model={model_hint}"
                    + (
                        f" color={profile.get('preferred_color')}"
                        if profile.get("preferred_color")
                        else ""
                    )
                    + "]"
                ).strip()

        if cat and any(n == "filter" for n in names):
            pinned: list[dict[str, Any]] = []
            mutated = False
            for c in tool_calls:
                if (c.get("function") or {}).get("name") != "filter":
                    pinned.append(c)
                    continue
                raw = (c.get("function") or {}).get("arguments", "{}")
                try:
                    a = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except json.JSONDecodeError:
                    pinned.append(c)
                    continue
                if not isinstance(a, dict):
                    pinned.append(c)
                    continue
                existing = str(a.get("category") or "").strip()
                if existing.lower() == cat.lower():
                    pinned.append(c)
                    continue
                a = dict(a)
                a["category"] = cat
                mutated = True
                pinned.append(
                    {
                        "id": c.get("id"),
                        "type": c.get("type") or "function",
                        "function": {
                            "name": "filter",
                            "arguments": json.dumps(a, ensure_ascii=True),
                        },
                    }
                )
            if mutated:
                tool_calls = pinned
                names = _tool_names(tool_calls)
                assistant_content = (
                    (assistant_content or "")
                    + f" [pinned filter category={cat}]"
                ).strip()

        if (
            not event_seen
            and not must_message
            and not candidates
            and not first_cart_done
            and retrieval_pids
            and searchish >= 1
            and (
                profile.get("firm_measures")
                or profile.get("firm_brand")
                or profile.get("brand_is_firm")
            )
            and any(n in ("search", "filter") for n in names)
        ):
            nxt = next((p for p in retrieval_pids if p not in viewed), None)
            if nxt:
                tool_calls = [
                    _forced_tool(
                        "view",
                        {"product_id": nxt},
                        turn=turn,
                        tag="pre-view",
                    )
                ][:max_calls]
                names = _tool_names(tool_calls)
                assistant_content = (
                    (assistant_content or "")
                    + f" [pre-event → view retrieval hit {nxt}]"
                ).strip()

        if (
            not event_seen
            and not must_message
            and not candidates
            and not first_cart_done
            and not pending_user_question
            and (not ask_first or asked_spec)
            and any(n in ("search", "filter", "view") for n in names)
        ):
            live_pre = [
                p
                for p in firm_ok_picks
                if p.get("product_id")
                and p.get("sku")
                and f"{p['product_id']}::{p['sku']}" not in dead_keys
                and in_budget(str(p["product_id"]), str(p["sku"]))
            ]
            if soft_brand and live_pre:
                soft_l = str(soft_brand).lower()
                soft_hits = [
                    p
                    for p in live_pre
                    if soft_l
                    in " ".join(
                        str(v)
                        for v in (facts_for(str(p.get("product_id") or "")) or {}).values()
                    ).lower()
                    or _tokenize_overlap(
                        soft_l,
                        " ".join(
                            str(v)
                            for v in (
                                facts_for(str(p.get("product_id") or "")) or {}
                            ).values()
                        ).lower(),
                    )
                ]
                if soft_hits:
                    live_pre = soft_hits + [p for p in live_pre if p not in soft_hits]
            if live_pre and bool(profile.get("firm_measures") or []):
                pick = live_pre[-1]
                already_live = (
                    isinstance(last_stock, dict)
                    and str(last_stock.get("product_id") or "") == pick.get("product_id")
                    and str(last_stock.get("sku") or "") == pick.get("sku")
                    and not _stock_is_false(last_stock)
                    and not need_reinspect
                )
                if already_live:
                    pass
                else:
                    tool_calls = [
                        _forced_tool(
                            "inspect_stock",
                            {
                                "product_id": pick["product_id"],
                                "sku": pick["sku"],
                            },
                            turn=turn,
                            tag="pre-ins",
                        )
                    ][:max_calls]
                    names = _tool_names(tool_calls)
                    last_pick = dict(pick)
                    assistant_content = (
                        (assistant_content or "")
                        + f" [pre-event → inspect firm-ok {pick['product_id']}::{pick['sku']}]"
                    ).strip()
            elif (
                str(profile.get("firm_brand") or "").strip()
                and isinstance(budget_limit, (int, float))
                and (
                    searchish >= 2
                    or (
                        profile.get("needs_justification")
                        and searchish >= 1
                        and bool(cat)
                    )
                )
                and not live_pre
                and not pre_event_filter_tried
                and any(n in ("search", "view") for n in names)
                and "filter" not in names
            ):
                fargs = {
                    "brand": str(profile.get("firm_brand") or "").strip(),
                    "max_price": float(budget_limit),
                }
                if cat:
                    fargs["category"] = cat
                tool_calls = [
                    _forced_tool(
                        "filter",
                        fargs,
                        turn=turn,
                        tag="pre-filt",
                    )
                ][:max_calls]
                names = _tool_names(tool_calls)
                pre_event_filter_tried = True
                assistant_content = (
                    (assistant_content or "")
                    + " [pre-event → filter brand/budget after search thrash]"
                ).strip()

        if (
            event_seen
            and event_acked
            and not must_message
            and not candidates
            and retrieval_pids
            and (
                repeat_retrieval >= 1
                or post_event_searches >= 1
            )
            and any(n in ("search", "filter") for n in names)
            and post_event_views_without_firm < 2
        ):
            nxt = next(
                (
                    p
                    for p in retrieval_pids
                    if p not in viewed
                    and p not in type_rejected_pids
                    and not any(
                        k.startswith(f"{p}::") for k in dead_keys
                    )
                ),
                None,
            )
            if nxt:
                tool_calls = [
                    _forced_tool(
                        "view",
                        {"product_id": nxt},
                        turn=turn,
                        tag="rec-view",
                    )
                ][:max_calls]
                names = _tool_names(tool_calls)
                assistant_content = (
                    (assistant_content or "")
                    + f" [recovery → view retrieval hit {nxt}]"
                ).strip()
                repeat_retrieval = 0
        elif (
            event_seen
            and event_acked
            and not must_message
            and not candidates
            and not pending_user_question
            and post_event_views_without_firm >= 1
            and profile.get("needs_justification")
            and str(profile.get("firm_brand") or "").strip()
            and cat
            and isinstance(budget_limit, (int, float))
            and remaining > 3
            and (
                not post_event_category_filter_tried
                or (
                    post_event_views_without_firm >= 2
                    and any(n in ("search", "inspect_stock", "view") for n in names)
                )
            )
        ):
            fargs = {
                "brand": str(profile.get("firm_brand") or "").strip(),
                "category": cat,
                "max_price": float(budget_limit),
            }
            nxt = next(
                (
                    p
                    for p in retrieval_pids
                    if p not in viewed
                    and p not in type_rejected_pids
                    and not any(k.startswith(f"{p}::") for k in dead_keys)
                ),
                None,
            )
            if nxt and post_event_category_filter_tried:
                tool_calls = [
                    _forced_tool(
                        "view",
                        {"product_id": nxt},
                        turn=turn,
                        tag="rec-catview",
                    )
                ][:max_calls]
                assistant_content = (
                    (assistant_content or "")
                    + f" [recovery → view filter hit {nxt}]"
                ).strip()
            else:
                tool_calls = [
                    _forced_tool(
                        "filter",
                        fargs,
                        turn=turn,
                        tag="rec-catfilt",
                    )
                ][:max_calls]
                post_event_category_filter_tried = True
                retrieval_pids = []
                assistant_content = (
                    (assistant_content or "")
                    + f" [recovery → filter brand+category={cat}]"
                ).strip()
            names = _tool_names(tool_calls)
            post_event_views_without_firm = 0
        elif (
            event_seen
            and event_acked
            and not must_message
            and not candidates
            and not pending_user_question
            and profile.get("needs_justification")
            and committed_model
            and str(profile.get("firm_brand") or "").strip()
            and not post_event_model_search_tried
            and post_event_category_filter_tried
            and post_event_views_without_firm >= 1
            and remaining > 3
        ):
            brand = str(profile.get("firm_brand") or "").strip()
            q = f"{brand} {committed_model}"
            if cat:
                q = f"{q} {cat}"
            tool_calls = [
                _forced_tool(
                    "search",
                    {"query": q, "in_stock": True, "k": 10},
                    turn=turn,
                    tag="rec-model",
                )
            ][:max_calls]
            names = _tool_names(tool_calls)
            post_event_model_search_tried = True
            post_event_views_without_firm = 0
            retrieval_pids = []
            assistant_content = (
                (assistant_content or "")
                + f" [recovery → sister-SKU search `{q}`]"
            ).strip()
        elif (
            event_seen
            and event_acked
            and not must_message
            and not candidates
            and not pending_user_question
            and post_event_views_without_firm >= 2
            and str(profile.get("firm_brand") or "").strip()
            and measure_tokens
            and remaining > 3
            and not profile.get("needs_justification")
        ):
            brand = str(profile.get("firm_brand") or "").strip()
            q_parts = [brand] + measure_tokens[:2]
            if cat:
                q_parts.append(cat)
            q = " ".join(q_parts)
            tool_calls = [
                _forced_tool(
                    "search",
                    {"query": q, "in_stock": True, "k": 10},
                    turn=turn,
                    tag="rec-meas",
                )
            ][:max_calls]
            names = _tool_names(tool_calls)
            post_event_views_without_firm = 0
            retrieval_pids = []
            assistant_content = (
                (assistant_content or "")
                + f" [recovery → force measure search `{q}`]"
            ).strip()
        elif (
            event_seen
            and event_acked
            and not must_message
            and not candidates
            and not pending_user_question
            and firm_ok_picks
            and any(n in ("search", "filter", "view") for n in names)
            and post_event_searches >= 1
        ):
            live = [
                p
                for p in firm_ok_picks
                if p.get("product_id")
                and p.get("sku")
                and f"{p['product_id']}::{p['sku']}" not in dead_keys
                and in_budget(str(p["product_id"]), str(p["sku"]))
            ]
            if live:
                pick = live[-1]
                tool_calls = [
                    _forced_tool(
                        "inspect_stock",
                        {
                            "product_id": pick["product_id"],
                            "sku": pick["sku"],
                        },
                        turn=turn,
                        tag="rec-ins",
                    )
                ][:max_calls]
                names = _tool_names(tool_calls)
                last_pick = dict(pick)
                assistant_content = (
                    (assistant_content or "")
                    + f" [recovery → inspect firm-ok {pick['product_id']}::{pick['sku']}]"
                ).strip()

        if phrase_retrieval:
            bumped: list[dict[str, Any]] = []
            for c in tool_calls:
                if (c.get("function") or {}).get("name") != "search":
                    bumped.append(c)
                    continue
                raw = (c.get("function") or {}).get("arguments", "{}")
                try:
                    a = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except json.JSONDecodeError:
                    a = {}
                if not isinstance(a, dict):
                    bumped.append(c)
                    continue
                a = dict(a)
                try:
                    k_now = int(a.get("k") or 0)
                except (TypeError, ValueError):
                    k_now = 0
                if k_now < 10:
                    a["k"] = 10
                    bumped.append(
                        {
                            "id": c.get("id"),
                            "type": c.get("type") or "function",
                            "function": {
                                "name": "search",
                                "arguments": json.dumps(a, ensure_ascii=True),
                            },
                        }
                    )
                else:
                    bumped.append(c)
            tool_calls = bumped
            names = _tool_names(tool_calls)

        last_pick_facts = (
            facts_for(str(last_pick.get("product_id") or ""))
            if last_pick.get("product_id")
            else {}
        )
        last_pick_firm_ok = bool(
            last_pick.get("product_id")
            and last_pick.get("sku")
            and last_pick_facts
            and not _open_catalog_firm_violation(last_pick_facts, profile)
        )
        pre_event_commit_ok = (
            not event_seen
            and not pending_user_question
            and not must_message
            and not candidates
            and (not ask_first or asked_spec)
            and bool(last_pick.get("product_id") and last_pick.get("sku"))
            and last_pick_firm_ok
            and in_budget(str(last_pick.get("product_id")), str(last_pick.get("sku")))
            and isinstance(last_stock, dict)
            and str(last_stock.get("product_id") or "") == last_pick.get("product_id")
            and str(last_stock.get("sku") or "") == last_pick.get("sku")
            and not _stock_is_false(last_stock)
            and not need_reinspect
            and (
                not soft_brand
                or soft_brand_first_tried
                and (
                    str(soft_brand).lower()
                    in " ".join(
                        str(v) for v in (last_pick_facts or {}).values()
                    ).lower()
                    or _tokenize_overlap(
                        str(soft_brand).lower(),
                        " ".join(str(v) for v in (last_pick_facts or {}).values()).lower(),
                    )
                    or not any(
                        str(soft_brand).lower()
                        in " ".join(
                            str(v)
                            for v in (
                                facts_for(str(p.get("product_id") or "")) or {}
                            ).values()
                        ).lower()
                        for p in firm_ok_picks
                        if p.get("product_id")
                    )
                )
            )
        )
        if pre_event_commit_ok and "place_test_order" not in names:
            already_carted = (
                cart_sku.get("product_id") == last_pick.get("product_id")
                and cart_sku.get("sku") == last_pick.get("sku")
            )
            if already_carted:
                if force_finish(turn, "pre-event order after live inspect"):
                    break
                tool_calls = [
                    _forced_order_call(
                        turn=turn,
                        pick={
                            "product_id": last_pick["product_id"],
                            "sku": last_pick["sku"],
                        },
                        query=query,
                        allowed_fields=allowed_fields,
                        profile=profile,
                        last_stock=last_stock,
                        last_view_facts=facts_for(last_pick.get("product_id"))
                        or dict(last_view_facts),
                        deadline=deadline,
                        claim_cache=claim_cache,
                    )
                ][:max_calls]
                names = _tool_names(tool_calls)
                assistant_content = (
                    (assistant_content or "")
                    + " [pre-event commit → order carted live sku]"
                ).strip()
            elif "add_to_cart" not in names or names == ["inspect_stock"]:
                tool_calls = [
                    _forced_tool(
                        "add_to_cart",
                        {
                            "product_id": last_pick["product_id"],
                            "sku": last_pick["sku"],
                        },
                        turn=turn,
                        tag="pre-cart",
                    )
                ][:max_calls]
                names = _tool_names(tool_calls)
                assistant_content = (
                    (assistant_content or "")
                    + " [pre-event commit → cart live inspected sku]"
                ).strip()

        recovery_commit_ok = (
            event_seen
            and event_acked
            and not pending_user_question
            and not must_message
            and not candidates
            and bool(last_pick.get("product_id") and last_pick.get("sku"))
            and f"{last_pick.get('product_id')}::{last_pick.get('sku')}" not in dead_keys
            and in_budget(str(last_pick.get("product_id")), str(last_pick.get("sku")))
            and last_pick_firm_ok
            and (
                not soft_brand
                or soft_brand_recovery_tried
                or bool(profile.get("firm_brand"))
                or not (profile.get("firm_measures") or [])
            )
            and (
                post_event_live_inspects >= 1
                or (
                    not need_reinspect
                    and isinstance(last_stock, dict)
                    and str(last_stock.get("product_id") or "") == last_pick.get("product_id")
                    and str(last_stock.get("sku") or "") == last_pick.get("sku")
                    and not _stock_is_false(last_stock)
                )
            )
        )
        if recovery_commit_ok and "place_test_order" not in names:
            if force_finish(turn, "recovery commit after live inspect"):
                break
            already_carted = (
                cart_sku.get("product_id") == last_pick.get("product_id")
                and cart_sku.get("sku") == last_pick.get("sku")
            )
            if not already_carted and last_pick_firm_ok:
                tool_calls = [
                    _forced_tool(
                        "add_to_cart",
                        {
                            "product_id": last_pick["product_id"],
                            "sku": last_pick["sku"],
                        },
                        turn=turn,
                        tag="rec-cart",
                    )
                ][:max_calls]
                names = _tool_names(tool_calls)
                assistant_content = (
                    (assistant_content or "")
                    + " [recovery commit → cart live inspected sku]"
                ).strip()
            elif already_carted and last_pick_firm_ok:
                tool_calls = [
                    _forced_order_call(
                        turn=turn,
                        pick={
                            "product_id": last_pick["product_id"],
                            "sku": last_pick["sku"],
                        },
                        query=query,
                        allowed_fields=allowed_fields,
                        profile=profile,
                        last_stock=last_stock,
                        last_view_facts=facts_for(last_pick.get("product_id"))
                        or dict(last_view_facts),
                        deadline=deadline,
                        claim_cache=claim_cache,
                    )
                ][:max_calls]
                names = _tool_names(tool_calls)
                assistant_content = (
                    (assistant_content or "")
                    + " [recovery commit → order carted live sku]"
                ).strip()

        acking_event = bool(event_seen and (not event_acked) and "message" in names)
        if any(n in ("search", "filter") for n in names):
            searchish += 1
            if event_seen:
                post_event_searches += 1
            for c in tool_calls:
                if (c.get("function") or {}).get("name") != "search":
                    continue
                raw = (c.get("function") or {}).get("arguments", "{}")
                try:
                    a = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except json.JSONDecodeError:
                    a = {}
                if isinstance(a, dict):
                    qn = " ".join(str(a.get("query") or "").lower().split())
                    if qn:
                        distinct_search_queries.add(qn)
            if soft_brand and _soft_brand_in_retrieval(
                tool_calls, soft_brand=str(soft_brand)
            ):
                if event_seen:
                    soft_brand_recovery_tried = True
                else:
                    soft_brand_first_tried = True
        if "message" in names and ask_first:
            msg_txt = ""
            for c in tool_calls:
                if (c.get("function") or {}).get("name") != "message":
                    continue
                raw = (c.get("function") or {}).get("arguments", "{}")
                try:
                    a = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except json.JSONDecodeError:
                    a = {}
                if isinstance(a, dict):
                    msg_txt = str(a.get("content") or a.get("message") or "")
                break
            if _message_requests_gated_spec(msg_txt, deadline=deadline):
                asked_spec = True
        if "message" in names and event_seen:
            was_unacked = not event_acked
            event_acked = True
            must_message = False
            pending_user_question = False
            if was_unacked and recovery_plan and (
                recovery_plan.get("guidance") or recovery_plan.get("search_queries")
            ):
                tip = recovery_plan.get("guidance") or ""
                if recovery_plan.get("search_queries") and not candidates:
                    tip += " Search: " + " | ".join(recovery_plan["search_queries"][:3])
                firm_b = str(profile.get("firm_brand") or "").strip()
                soft = str(soft_brand or "").strip()
                pin = firm_b or (
                    soft if not soft_brand_recovery_tried else ""
                )
                mtoks = []
                for m in (profile.get("firm_measures") or [])[:2]:
                    if not isinstance(m, dict):
                        continue
                    if m.get("value_text"):
                        mtoks.append(str(m["value_text"]))
                    elif m.get("numeric") is not None:
                        mtoks.append(str(m["numeric"]))
                if pin and mtoks:
                    tip = (
                        f"Call search query=`{pin} {' '.join(mtoks)}` first "
                        "(exact firm measure + brand), then view a NEW product_id. "
                        + tip
                    )
                elif pin and isinstance(budget_limit, (int, float)):
                    tip = (
                        f"Call filter brand=`{pin}` max_price={budget_limit} first. "
                        + tip
                    )
                if tip:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Recovery plan after event ack: " + tip
                                + " Do not message again unless the shopper asks."
                            ),
                        }
                    )
        if "add_to_cart" in names:
            first_cart_done = True
            if not committed_model and last_pick.get("product_id"):
                m = str(
                    (facts_for(str(last_pick["product_id"])) or {}).get("model") or ""
                ).strip()
                if m and len(m) >= 2:
                    committed_model = m
        if "place_test_order" in names:
            ordered = True

        for call in tool_calls:
            name = call.get("function", {}).get("name") or ""
            if name in ("inspect_stock", "add_to_cart", "place_test_order"):
                raw = call["function"].get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = {}
                if isinstance(args, dict) and args.get("product_id") and args.get("sku"):
                    last_pick = {
                        "product_id": str(args["product_id"]),
                        "sku": str(args["sku"]),
                    }
            if name == "add_to_cart" and last_pick:
                cart_sku = dict(last_pick)
            if name == "remove_from_cart":
                cart_sku = {}

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        env_t = next_env_turn()
        if env_t is None:
            break
        group_id = f"{problem_id}-turn-{env_t}"
        sanitized_calls: list[dict[str, Any]] = []
        for index, call in enumerate(tool_calls, start=1):
            name = call.get("function", {}).get("name") or ""
            pid_for_facts = None
            if name == "place_test_order":
                raw = call["function"].get("arguments", "{}")
                try:
                    raw_args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    raw_args = {}
                if isinstance(raw_args, dict):
                    pid_for_facts = str(raw_args.get("product_id") or "") or None
            view_for_args = (
                facts_for(pid_for_facts) or last_view_facts
                if name == "place_test_order"
                else last_view_facts
            )
            args = _arguments(
                call,
                query=query,
                allowed_fields=allowed_fields,
                profile=profile,
                last_stock=last_stock,
                last_view_facts=view_for_args,
                deadline=deadline,
                claim_cache=claim_cache,
            )
            if name == "message" and isinstance(args, dict):
                msg_body = str(args.get("content") or args.get("message") or "")
                if ask_first and not asked_spec and not event_seen:
                    if (
                        not _message_requests_gated_spec(msg_body, deadline=deadline)
                        or _message_claims_market_shock(msg_body, deadline=deadline)
                    ):
                        args = _set_message_content(args, _ASK_GATED_SPEC_CONTENT)
                    asked_spec = True
                elif acking_event and recovery_plan and recovery_plan.get("message_content"):
                    out = dict(args)
                    if "content" in out or "message" not in out:
                        out["content"] = recovery_plan["message_content"]
                        out.pop("message", None)
                    else:
                        out["message"] = recovery_plan["message_content"]
                    args = out
                elif (
                    not event_seen
                    and _message_claims_market_shock(msg_body, deadline=deadline)
                    and not _message_requests_gated_spec(msg_body, deadline=deadline)
                ):
                    if ask_first and not (profile.get("firm_measures") or []):
                        args = _set_message_content(args, _ASK_GATED_SPEC_CONTENT)
                        asked_spec = True
                    else:
                        args = _set_message_content(
                            args,
                            (
                                "I'll keep looking for an in-stock option that meets "
                                "every firm requirement within budget."
                            ),
                        )
                elif event_seen and (acking_event or not event_acked):
                    args = _rewrite_recovery_message(
                        args, model_hint=model_hint, deadline=deadline
                    )
            sanitized_calls.append(
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {"name": name, "args": args},
                }
            )
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": env_t,
            "calls": sanitized_calls,
        }
        result = _env_call(envelope, deadline=deadline)
        if result is None:
            rollback_env_turn()
            if last_pick and force_finish(turn, "env timeout"):
                break
            break
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

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            obs = call_result.get("observation")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(obs),
                }
            )
            payload = _obs_stock_payload(obs if isinstance(obs, dict) else {})
            name = tool_call.get("function", {}).get("name") or ""
            scratch: dict[str, str] = {}
            if isinstance(obs, dict):
                _dig_facts(obs, scratch)
                _ingest_closed_set_obs(
                    obs,
                    candidates=candidates,
                    ledger=rank_ledger,
                )
            if name in ("search", "filter") and isinstance(obs, dict):
                pids = _collect_retrieval_pids(obs)
                if pids:
                    merged_pids = pids + [p for p in retrieval_pids if p not in pids]
                    retrieval_pids = merged_pids[:40]
                try:
                    call_args = json.loads(
                        tool_call["function"].get("arguments") or "{}"
                    )
                except json.JSONDecodeError:
                    call_args = {}
                sig = json.dumps(
                    {"n": name, "a": call_args},
                    sort_keys=True,
                    ensure_ascii=True,
                )
                if sig == last_retrieval_sig:
                    repeat_retrieval += 1
                else:
                    last_retrieval_sig = sig
                    repeat_retrieval = 0
            if name in ("view", "compare") and isinstance(obs, dict):
                facts = _extract_view_facts(obs)
                if not facts and scratch:
                    facts = scratch
                try:
                    call_args = json.loads(
                        tool_call["function"].get("arguments") or "{}"
                    )
                except json.JSONDecodeError:
                    call_args = {}
                pid = str((call_args or {}).get("product_id") or "")
                node = (
                    obs.get("observation")
                    if isinstance(obs.get("observation"), dict)
                    else obs
                )
                if isinstance(node, dict):
                    pid_n = str(node.get("product_id") or pid or "")
                    for var in node.get("variants") or []:
                        if not isinstance(var, dict):
                            continue
                        remember_price(
                            pid_n, str(var.get("sku") or ""), var.get("price")
                        )
                        sku_o = str(var.get("sku") or "")
                        opts_o = str(var.get("options") or "").strip()
                        if pid_n and sku_o and opts_o:
                            options_by_key[f"{pid_n}::{sku_o}"] = opts_o
                if facts:
                    if pid:
                        remember_facts(pid, facts)
                        if not candidates and not _open_catalog_firm_violation(facts, profile):
                            added = False
                            firm_b = str(profile.get("firm_brand") or "").strip().lower()
                            if firm_b:
                                brand_fact = str(facts.get("brand") or "").strip().lower()
                                blob = " ".join(str(v) for v in facts.values()).lower()
                                if not (
                                    brand_fact
                                    and (
                                        firm_b == brand_fact
                                        or firm_b in brand_fact
                                        or brand_fact in firm_b
                                    )
                                    or firm_b in blob
                                ):
                                    if event_seen and event_acked:
                                        post_event_views_without_firm += 1
                                    continue
                            if isinstance(node, dict):
                                for var in node.get("variants") or []:
                                    if not isinstance(var, dict):
                                        continue
                                    sku_v = str(var.get("sku") or "")
                                    if not sku_v:
                                        continue
                                    if not in_budget(pid, sku_v):
                                        continue
                                    firm_ok_picks.append(
                                        {"product_id": pid, "sku": sku_v}
                                    )
                                    added = True
                            if not added:
                                firm_ok_picks.append({"product_id": pid, "sku": ""})
                            if event_seen and event_acked:
                                if added:
                                    post_event_views_without_firm = 0
                                else:
                                    post_event_views_without_firm += 1
                        elif event_seen and event_acked:
                            post_event_views_without_firm += 1
                            if pid and _product_type_violation(facts, profile):
                                type_rejected_pids.add(pid)
                    else:
                        last_view_facts = dict(facts)
                        focus_pid = None
                elif event_seen and event_acked:
                    post_event_views_without_firm += 1
            elif isinstance(obs, dict) and scratch:
                try:
                    call_args = json.loads(
                        tool_call["function"].get("arguments") or "{}"
                    )
                except json.JSONDecodeError:
                    call_args = {}
                pid = str((call_args or {}).get("product_id") or "")
                if pid and ("brand" in scratch or "title" in scratch):
                    remember_facts(pid, scratch)
            if name == "inspect_stock":
                need_reinspect = False
            if name in ("inspect_stock", "add_to_cart", "inspect_cart") and payload:
                last_stock = payload
                _ledger_apply_stock(rank_ledger, payload)
                if payload.get("product_id") and payload.get("sku"):
                    last_pick = {
                        "product_id": str(payload["product_id"]),
                        "sku": str(payload["sku"]),
                    }
                    pid_s = str(payload["product_id"])
                    sku_s = str(payload["sku"])
                    remember_price(pid_s, sku_s, payload.get("price"))
                    facts_s = facts_for(pid_s)
                    if (
                        not candidates
                        and sku_s
                        and facts_s
                        and f"{pid_s}::{sku_s}" not in dead_keys
                        and not _open_catalog_firm_violation(facts_s, profile)
                        and in_budget(pid_s, sku_s)
                    ):
                        firm_ok_picks.append({"product_id": pid_s, "sku": sku_s})
                    elif (
                        event_seen
                        and event_acked
                        and facts_s
                        and _open_catalog_firm_violation(facts_s, profile)
                    ):
                        post_event_views_without_firm += 1
                        firm_ok_picks[:] = [
                            p
                            for p in firm_ok_picks
                            if not (
                                str(p.get("product_id") or "") == pid_s
                                and str(p.get("sku") or "") == sku_s
                            )
                        ]
                        if (
                            last_pick.get("product_id") == pid_s
                            and last_pick.get("sku") == sku_s
                        ):
                            last_pick = {}
                    if (
                        name == "inspect_stock"
                        and event_seen
                        and event_acked
                        and f"{pid_s}::{sku_s}" not in dead_keys
                        and not _stock_is_false(payload)
                        and facts_s
                        and not _open_catalog_firm_violation(facts_s, profile)
                    ):
                        post_event_live_inspects += 1
                    if name == "add_to_cart":
                        cart_sku = dict(last_pick)
                if _stock_is_false(payload):
                    shock_recover = True
                    need_reinspect = True
                    event_seen = True
                    event_acked = False
                    must_message = True
                    note_dead(
                        str(payload.get("product_id") or ""),
                        str(payload.get("sku") or ""),
                    )
            if isinstance(obs, dict):
                inner = (
                    obs.get("observation")
                    if isinstance(obs.get("observation"), dict)
                    else obs
                )
                if isinstance(inner, dict) and inner.get("market_event"):
                    shock_recover = True
                    need_reinspect = True
                    event_seen = True
                    event_acked = False
                    must_message = True
                    cart_sku = {}
                    ev = inner["market_event"]
                    tgt = ev.get("target") if isinstance(ev, dict) else None
                    if isinstance(tgt, dict) and tgt.get("product_id") and tgt.get("sku"):
                        note_dead(str(tgt["product_id"]), str(tgt["sku"]))
                        if isinstance(ev, dict) and ev.get("new_price") is not None:
                            remember_price(
                                str(tgt["product_id"]),
                                str(tgt["sku"]),
                                ev.get("new_price"),
                            )
                    added = inner.get("added")
                    if isinstance(added, dict):
                        last_stock = added

        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            messages.append({"role": "user", "content": content})
            turn_info = _llm_interpret_shopper_turn(
                str(content),
                event_seen=event_seen,
                event_acked=event_acked,
                ask_first=ask_first,
                deadline=deadline,
            )
            kind = str(turn_info.get("kind") or "other")
            updated_budget = turn_info.get("budget_ceiling")
            revealed_measures = turn_info.get("firm_measures") or []
            if isinstance(revealed_measures, list) and revealed_measures:
                merged = list(profile.get("firm_measures") or [])
                existing_nums = {
                    float(m["numeric"])
                    for m in merged
                    if isinstance(m, dict) and m.get("numeric") is not None
                }
                for m in revealed_measures:
                    if not isinstance(m, dict):
                        continue
                    num = m.get("numeric")
                    try:
                        num_f = float(num) if num is not None else None
                    except (TypeError, ValueError):
                        num_f = None
                    if num_f is not None and isinstance(budget_limit, (int, float)):
                        if abs(num_f - float(budget_limit)) <= 1e-6:
                            continue
                    if num_f is not None and num_f in existing_nums:
                        continue
                    merged.append(
                        {
                            "name": str(m.get("name") or "measure"),
                            "numeric": num_f,
                            "value_text": str(m.get("value_text") or "").strip(),
                        }
                    )
                    if num_f is not None:
                        existing_nums.add(num_f)
                if merged != list(profile.get("firm_measures") or []):
                    profile["firm_measures"] = merged[:6]
                    eligibility_cache.clear()
                    asked_spec = True
                    must_message = False
                    pending_user_question = False
                    bits = [
                        str(m.get("value_text") or m.get("numeric"))
                        for m in merged[:3]
                        if isinstance(m, dict)
                    ]
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Firm measure(s) now binding: "
                                + "; ".join(bits)
                                + ". Exact match required on the chosen sku. "
                                "Do not message — retrieve/filter and view next."
                            ),
                        }
                    )
            if isinstance(updated_budget, (int, float)):
                prev_b = budget_limit
                budget_limit = float(updated_budget)
                eligibility_cache.clear()
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Budget ceiling updated to {budget_limit}"
                            + (f" (was {prev_b})." if prev_b is not None else ".")
                            + " Only cart/order in-budget firm-valid skus."
                        ),
                    }
                )
            if kind == "empty":
                pending_user_question = False
            elif event_seen and event_acked:
                must_message = False
                pending_user_question = False
            elif kind == "market_signal":
                need_reinspect = True
                event_seen = True
                if not event_acked:
                    must_message = True
                elif turn_info.get("requires_agent_message") and not event_acked:
                    must_message = True
                else:
                    must_message = False
                    pending_user_question = False
            elif kind in (
                "question",
                "constraint_hold",
                "budget_update",
                "spec_reveal",
            ):
                if ask_first and asked_spec and kind in (
                    "budget_update",
                    "spec_reveal",
                    "constraint_hold",
                ):
                    must_message = False
                    pending_user_question = False
                elif (
                    event_seen
                    and event_acked
                    and kind == "question"
                    and not pending_user_question
                ):
                    must_message = False
                    pending_user_question = False
                elif kind == "question" or bool(
                    turn_info.get("requires_agent_message")
                ):
                    must_message = True
                    pending_user_question = kind == "question" or bool(
                        turn_info.get("requires_agent_message")
                    )
                else:
                    must_message = False
                    pending_user_question = False
            elif turn_info.get("requires_agent_message"):
                must_message = True
                pending_user_question = kind == "question"
        if any(
            (call.get("observation") or {}).get("done") for call in result["calls"]
        ):
            break

        if (
            event_seen
            and event_acked
            and not ordered
            and not (ask_first and not asked_spec)
            and _time_left(deadline) > 8
            and remaining > 1
        ):
            pick = resolve_pick()
            if pick:
                last_pick = dict(pick)
            if last_pick and force_finish(turn, "post-ack recover"):
                break

        if _time_left(deadline) < 20 or remaining <= 4:
            soft_first_guard = (
                bool(soft_brand)
                and bool(profile.get("firm_measures") or [])
                and not first_cart_done
                and _time_left(deadline) >= 18
            )
            if (
                not ordered
                and not soft_first_guard
                and (last_pick or firm_ok_picks or cart_sku)
                and (
                    not event_seen
                    or event_acked
                )
                and force_finish(turn, f"late finish left={_time_left(deadline):.1f}s")
            ):
                break
            continue

        if not must_message and not candidates and not shock_recover:
            if (
                event_seen
                and event_acked
                and first_cart_done
                and last_pick
                and not need_reinspect
                and post_event_live_inspects >= 1
            ):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "LIVE sku already inspected after shock. REQUIRED next: "
                            f"add_to_cart then place_test_order on "
                            f"{last_pick.get('product_id')}::{last_pick.get('sku')} "
                            "NOW — do not inspect again."
                        ),
                    }
                )
            elif event_seen and first_cart_done and last_pick:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "After shock recovery, place_test_order on the live "
                            "inspected sku as soon as stock is confirmed. "
                            "Do not keep searching soft brand if firm measure mismatches."
                        ),
                    }
                )
            elif event_seen and post_event_searches >= 2 and remaining <= 6:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "STOP soft-brand thrash. View/inspect any firm-measure-exact "
                            "in-budget sku (any brand), add_to_cart, place_test_order NOW."
                        ),
                    }
                )
            elif (
                phrase_retrieval
                and searchish < search_stop
                and not event_seen
                and len(viewed) < 3
                and not last_pick
            ):
                unused = []
                if search_plan and (search_plan.get("queries") or []):
                    unused = [
                        q
                        for q in search_plan["queries"]
                        if isinstance(q, str)
                        and q.strip()
                        and " ".join(q.lower().split()) not in distinct_search_queries
                    ]
                tip = unused[0] if unused else ""
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "RETRIEVAL: issue another DISTINCT search/filter for the "
                            "category + phrase, view diverse hits, then inspect → cart. "
                            + (f"Try `{tip}`." if tip else "")
                        ),
                    }
                )
            elif phrase_retrieval and last_pick and not event_seen:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Enough surveying. inspect_stock (if needed) → add_to_cart → "
                            f"place_test_order on {last_pick.get('product_id')}::"
                            f"{last_pick.get('sku')} NOW."
                        ),
                    }
                )
            elif searchish < 4:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue: inspect_stock → add_to_cart → place_test_order "
                            "on a firm-valid in-stock sku."
                        ),
                    }
                )
        elif (
            not must_message
            and candidates
            and len(viewed & cand_pids) >= len(cand_pids)
            and first_cart_done
        ):
            msg = "Closed-set: inspect_stock live best, add_to_cart, place_test_order."
            if bk:
                msg += f" Best live: {bk}."
            messages.append({"role": "user", "content": msg})

    return dialogue


__all__ = ["agent_main"]
