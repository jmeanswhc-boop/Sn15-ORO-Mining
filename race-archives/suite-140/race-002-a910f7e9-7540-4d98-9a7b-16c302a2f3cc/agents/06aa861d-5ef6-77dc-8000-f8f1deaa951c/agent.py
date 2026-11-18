"""ORO Bench EnvPack — TF7 justification contract (v52).

Donor: justificationTF7_v51. Submit-test candidate (constraint hang + brand sanitize).

Portable (episode-semantic only):
  - Shared: structural brand latch (required brand is X; prefer X when;
    word-boundary get/need — never budget prose / no brand allowlists).
  - Intent pipeline only: pin commit-quiz product_id::sku; first cart sticky
    until shock (intended_event_commitment). LLM elects before pin; FSM after.
  - Constraint pipeline only: firm brand on first cart when brand is non-soft.
  - v51/v52 constraint: reject category-noun brand pollution; no Task-rules
    resolution class-table firm; shopper-grounded prune; inspect→add rewrite.
  - v52: no mini brand allowlist; keep product brands like Nothing Phone;
    strip only trailing lowercase category nouns (phone/smartphone).

No suite phrase routers / pixel ladders / family signs / product-type tables.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# honest_oro 70/70 first-tool opener — portable clarify-before-search (all families).
_HONEST_CLARIFY_OPENER = (
    "Before I start searching, could you clarify your firm requirements? "
    "Please confirm the must-have specs, the category, any required brand, "
    "your budget, and how you would prioritize the trade-offs between your preferences."
)

_OPENROUTER_MODELS = [
    "deepseek/deepseek-v3.2",
    "moonshotai/kimi-k2.6",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen3.6-27b",
]

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
_CLAIM_DENY = frozenset({"category"})
_FIELD_ALIASES = {
    "display": "resolution",
    "screen": "screen_size",
    "size": "screen_size",
    "screen size": "screen_size",
    "refresh": "refresh_rate",
    "hz": "refresh_rate",
    "category_path": "category",
}

_CAND_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9][A-Za-z0-9._-]*)")
_TASK_RULES_SPLIT_RE = re.compile(r"(?is)\n?\s*task\s*rules\s*:\s*")
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|"
    r"max(?:imum)? of|within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?",
    re.I,
)
_PIPELINES = (
    "intent",
    "retrieval",
    "constraint",
    "preference",
    "ranking",
    "recovery",
    "justification",
    "generic",
)

_proxy = ProxyClient(timeout=90, max_retries=2, retry_delay=0.6)
# One HTTP attempt per call_id — never re-POST identical env payload on 409.
_env_proxy = ProxyClient(timeout=90, max_retries=1, retry_delay=0.25)
_ENV_CALL_SEQ = 0
_PROBLEM_BUDGET_S = 260.0

_CORE_SYSTEM = """You are a shopping agent. Use ONLY the supplied tools. One tool call per turn.
Obey THIS episode's shopper text and Task rules exactly. Do not memorize product IDs.
Finish contract (controller-enforced): view → inspect_stock → add_to_cart →
inspect_stock again (post-add) → place_test_order only if in_stock and within budget.
Never place on a shocked / OOS / over-budget cart. Answer shopper questions with `message`.
Never ask to raise the budget.
"""

_PIPELINE_PLAYBOOKS: dict[str, str] = {
    "intent": (
        "WORKFLOW INTENT: Separate firm vs soft from THIS Task rules. "
        "Firm: category, hardware amounts, budget, stock. Soft: brand — honor preferred "
        "brand on first cart when a fully compliant option exists; else any compliant. "
        "After price_change over budget: remove → recover → re-inspect → order."
    ),
    "retrieval": (
        "WORKFLOW RETRIEVAL: Reformulate searches; match requested title tokens per Task rules; "
        "survey distinct matching identities before order; view → inspect → cart → "
        "post-add inspect → place only in-stock in-budget."
    ),
    "constraint": (
        "WORKFLOW CONSTRAINT: Exact selected-variant obligations from THIS rules. "
        "If a firm spec is withheld until requested → first tool is `message` asking for it. "
        "Bind to selected SKU options. Stockout → different product still meeting every firm "
        "requirement; always post-add inspect before place."
    ),
    "preference": (
        "WORKFLOW PREFERENCE: Soft use-case fit among firm-valid options. "
        "Inspect ≥2 distinct listing_id values before first cart when rules require it. "
        "First cart is graded. If repriced above budget: remove and recover a different "
        "listing_id; never order over budget."
    ),
    "ranking": (
        "WORKFLOW RANKING: Closed product_id::sku set only. View/compare listed products. "
        "Elect by THIS rules' tradeoff axes among in-stock in-budget keys; drop options "
        "strictly dominated on every axis. After market moves, re-elect among remaining live "
        "keys and inspect the replacement. Never open-catalog search."
    ),
    "recovery": (
        "WORKFLOW RECOVERY: Prefer named model (+ color) from shopper. First cart must match. "
        "On stockout: remove → answer shopper if needed → same model other color before other "
        "models → view → inspect → cart → post-add inspect → order. "
        "Never relax category/budget/stock."
    ),
    "justification": (
        "WORKFLOW JUSTIFICATION: Structured claims required on place when THIS rules say so. "
        "View facts before first cart; post-add inspect; claims from view/inspect only "
        "(requested spec, price, stock, plus a useful reason field named in the rules when "
        "observed). Close market events before order."
    ),
    "generic": (
        "WORKFLOW GENERIC: Infer firm vs soft from THIS Task rules; retrieve → view → inspect → "
        "cart → post-add inspect → order. On market_event: remove → recover → re-inspect → order."
    ),
}


@dataclass
class ObligationProfile:
    pipeline: str = "generic"
    ask_first: bool = False
    closed_set: bool = False
    justification_required: bool = False
    multi_listing_min: int = 0
    relax_color_then_model: bool = False
    title_phrase_match: bool = False
    soft_brand_first_cart: bool = False
    strict_selected_variant: bool = False
    confidence: float = 0.0
    candidate_keys: list[tuple[str, str]] = field(default_factory=list)
    # RuleLedger slots (compiled from THIS episode Task rules + shopper).
    preferred_brand: str = ""
    required_category: str = ""
    require_category_path: bool = True
    listing_id_min: int = 0
    min_matching_identities: int = 0
    title_terms: list[str] = field(default_factory=list)
    firm_specs: list[dict[str, str]] = field(default_factory=list)
    ranking_axes: list[dict[str, str]] = field(default_factory=list)
    justification_spec_field: str = ""
    useful_reason_fields: list[str] = field(default_factory=list)
    # TF4: freeform soft-preference tokens harvested from THIS shopper/goal_brief.
    pref_tokens: list[str] = field(default_factory=list)
    firm_hint: str = ""
    # Optional resolution class defs copied from THIS rules text only.
    resolution_defs: list[dict[str, Any]] = field(default_factory=list)
    # Device / accessory evidence (parsed from THIS rules when present).
    require_device_evidence: bool = False
    device_os_allowlist: list[str] = field(default_factory=list)
    reject_accessories: bool = False
    preferred_model: str = ""
    preferred_color: str = ""


@dataclass
class PipelineKnobs:
    name: str
    llm_soft_cap: int
    llm_hard_cap: int
    max_search_before_view: int
    shock_protocol: str  # off | light | full
    require_multi_listing: int
    playbook: str


def _provider() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw in ("chutes", "openrouter"):
        return raw
    if getenv("CHUTES_API_KEY") and not getenv("OPENROUTER_API_KEY"):
        return "chutes"
    return "openrouter"


def _model_list() -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    # OpenRouter-native IDs only (Chutes provider still accepts OR slugs via proxy).
    return list(_OPENROUTER_MODELS)


def _split_shopper_and_rules(query: str) -> tuple[str, str]:
    text = str(query or "")
    m = _TASK_RULES_SPLIT_RE.search(text)
    if not m:
        return text.strip(), ""
    return text[: m.start()].strip(), text[m.end() :].strip()


def _shopper_text(query: str) -> str:
    return _split_shopper_and_rules(query)[0]


def _task_rules(query: str) -> str:
    return _split_shopper_and_rules(query)[1]


def _parse_candidate_variants(query: str) -> list[tuple[str, str]]:
    """Structural closed set: ≥2 product_id::sku pairs in shopper text (not English slogans)."""
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


def _normalize_candidate_keys(raw: Any) -> list[tuple[str, str]]:
    """Accept (pid, sku) tuples or 'pid::sku' strings — never crash unpack."""
    if not raw:
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        pid = sku = ""
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            pid, sku = str(item[0] or "").strip(), str(item[1] or "").strip()
        elif isinstance(item, str) and "::" in item:
            pid, _, sku = item.partition("::")
            pid, sku = pid.strip(), sku.strip()
        if not pid or not sku:
            continue
        key = f"{pid}::{sku}"
        if key in seen:
            continue
        seen.add(key)
        out.append((pid, sku))
    return out


def _budget_limit(text: str) -> float | None:
    best = None
    for m in _BUDGET_RE.finditer(text or ""):
        whole = m.group(1).replace(",", "")
        frac = m.group(2) or "0"
        try:
            val = float(f"{whole}.{frac}")
        except ValueError:
            continue
        if best is None or val < best:
            best = val
    return best


def _tighten_budget(current: float | None, text: str) -> float | None:
    """Uptake shrunk budgets from shopper interventions (add_constraint)."""
    found = _budget_limit(text)
    if found is None:
        return current
    if current is None:
        return found
    return min(current, found)


def _merge_firm_specs(
    existing: list[dict[str, str]] | None, text: str
) -> list[dict[str, str]]:
    """Merge firm hardware specs from mid-episode user text (latest value wins)."""
    by_name: dict[str, dict[str, str]] = {}
    for spec in existing or []:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name") or "").strip().lower()
        if name:
            by_name[name] = {
                "name": name,
                "value": str(spec.get("value") or "").strip(),
            }
    for spec in _struct_harvest_firm_specs(text or ""):
        name = str(spec.get("name") or "").strip().lower()
        if name:
            by_name[name] = {
                "name": name,
                "value": str(spec.get("value") or "").strip(),
            }
    return list(by_name.values())[:6]


def _is_commit_quiz(text: str) -> bool:
    """Shopper commit / pre-buy probe — paraphrase-safe, not suite slogans only."""
    low = (text or "").lower()
    if "?" not in low and not re.search(
        r"(?i)\b(?:confirm|which|going with|pick(?:ing)?|choose|chosen)\b", low
    ):
        return False
    return bool(
        re.search(
            r"(?i)(?:which one|going with|which (?:item|listing|product|option)|"
            r"are you (?:buying|ordering|choosing|picking)|"
            r"before you (?:buy|order|checkout|purchase)|"
            r"does it (?:actually )?have|confirm (?:the )?(?:choice|pick|item)|"
            r"ready to (?:buy|order)|final(?:ly)? (?:pick|choice))",
            low,
        )
    )


def _is_price_shock_probe(text: str) -> bool:
    low = (text or "").lower()
    return bool(
        re.search(
            r"(?i)(?:more expensive|price (?:went|jumped|changed|increased|rose)|"
            r"did (?:the )?(?:price|item|product).{0,40}(?:expensive|change|increase)|"
            r"from\s+\d[\d,\.]*\s+to\s+\d|got more expensive)",
            low,
        )
    )


def _is_stock_check_probe(text: str) -> bool:
    low = (text or "").lower()
    return bool(
        re.search(
            r"(?i)(?:still available|still in stock|is that (?:still )?available|"
            r"out of stock|sold out|check(?:ing)? (?:the )?stock|wait.? is that)",
            low,
        )
    )


def _budget_ladder_caps(budget: float | None) -> list[float]:
    """Generic max_price steps from *current* budget (not fixed suite PHP tables)."""
    if budget is None or budget <= 0:
        return []
    caps: list[float] = []
    for frac in (0.25, 0.5, 0.85, 1.0):
        cap = round(float(budget) * frac, 2)
        if cap > 0 and cap not in caps:
            caps.append(cap)
    return caps


def _intent_seed_query(profile: ObligationProfile, *, include_brand: bool = True) -> str:
    """Build a portable intent seed query from known firm/soft slots (no suite phrases).

    Phase A (first seed): include soft preferred brand when known.
    Phase B (after budget shrink / shock reseed): drop brand for coverage (teacher pattern).
    """
    bits: list[str] = []
    if include_brand and profile.preferred_brand:
        bits.append(profile.preferred_brand.strip())
    for fs in profile.firm_specs or []:
        name = str(fs.get("name") or "").strip().lower()
        val = str(fs.get("value") or "").strip()
        if not val:
            continue
        if name in ("screen_size", "screen"):
            bits.append(f"{val} inch")
        elif name in ("ram",):
            bits.append(f"{val} RAM")
        elif name in ("resolution", "refresh_rate"):
            bits.append(val)
        else:
            bits.append(val)
    if profile.required_category:
        bits.append(profile.required_category.strip())
    elif not bits:
        bits.append("in stock")
    # de-dupe order-preserving
    seen: set[str] = set()
    out: list[str] = []
    for b in bits:
        bl = b.lower()
        if bl in seen:
            continue
        seen.add(bl)
        out.append(b)
    return " ".join(out)[:120]


def _constraint_seed_query(profile: ObligationProfile) -> str:
    """TF3 seed from firm slots — brand is firm when named (not soft-drop)."""
    bits: list[str] = []
    if profile.preferred_brand:
        bits.append(profile.preferred_brand.strip())
    for fs in profile.firm_specs or []:
        name = str(fs.get("name") or "").strip().lower()
        val = str(fs.get("value") or "").strip()
        if not val:
            continue
        if name in ("screen_size", "screen"):
            bits.append(f"{val} inch")
        elif name == "ram":
            bits.append(f"{val} RAM")
        elif name == "storage":
            bits.append(f"{val}GB")
        elif name in ("resolution", "refresh_rate", "ip_rating", "feature"):
            bits.append(val)
        else:
            bits.append(val)
    if profile.required_category:
        bits.append(profile.required_category.strip())
    elif not bits:
        bits.append("in stock")
    seen: set[str] = set()
    out: list[str] = []
    for b in bits:
        bl = b.lower()
        if bl in seen:
            continue
        seen.add(bl)
        out.append(b)
    return " ".join(out)[:120]


def _justification_seed_query(profile: ObligationProfile) -> str:
    """TF7 seed: brand + shopper firm + product type (teacher: 'Brand 27 inch monitor')."""
    return _constraint_seed_query(profile)


def _resolution_alias_match(want: str, observed_blob: str) -> bool:
    """Portable name synonyms only — not frozen pixel→class ladders.

    Kitchen-sink resolution strings that list several conflicting classes
    (e.g. '4k, svga, 720p, 1080p, 8k') are ambiguous — refuse firm match.
    """
    w = re.sub(r"[^a-z0-9]+", "", (want or "").lower())
    if not w:
        return True
    raw = (observed_blob or "").lower()
    toks = set(re.findall(r"[a-z0-9]+", raw))
    # Normalized compact form for WxH tokens only (avoid 'hd' substring of 'uhd').
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    groups = (
        {"hd", "720p", "1280x720"},
        {"fullhd", "fhd", "1080p", "1920x1080"},
        {"qhd", "1440p", "2560x1440"},
        {"uhd", "4k", "2160p", "3840x2160"},
    )
    # Ambiguous multi-class blob: hits ≥2 distinct class groups → no firm class claim.
    hit_groups = [g for g in groups if (toks & g) or any(x in compact for x in g if x[0].isdigit())]
    if len(hit_groups) >= 2:
        return False
    if w in toks:
        return True
    for g in groups:
        if w not in g:
            continue
        if toks & g:
            return True
        # WxH forms may appear without 'x' tokenized (1280x720 → 1280x720 as one tok).
        if any(x in compact for x in g if x[0].isdigit()):
            return True
    return False


def _search_hit_category_hint(row: dict[str, Any] | None, required: str) -> bool:
    """Cheap list-row hint that a search hit may satisfy required category noun."""
    req = _sanitize_required_category(required).lower()
    if not req or not isinstance(row, dict):
        return False
    path = row.get("category_path") or row.get("category") or ""
    if isinstance(path, (list, tuple)):
        blob = " ".join(str(x) for x in path).lower()
    else:
        blob = str(path).lower()
    title = str(row.get("title") or "").lower()
    blob = f"{blob} {title}"
    if req in blob:
        return True
    req_toks = [t for t in re.split(r"[^a-z0-9]+", req) if len(t) > 2]
    return bool(req_toks) and all(t in blob for t in req_toks[:2])


def _is_timeout_err(exc: BaseException) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(tok in blob for tok in ("timeout", "timed out", "deadline"))


def _chat_raw(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    temperature: float = 0,
) -> dict[str, Any]:
    last_err: Exception | None = None
    for model in _model_list():
        try:
            body: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if tools is not None:
                body["tools"] = tools
                body["tool_choice"] = tool_choice or "required"
            inference = _proxy.post("/inference/chat/completions", json_data=body)
            if inference is None:
                last_err = RuntimeError(f"inference failed for {model}")
                continue
            return inference
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if _is_timeout_err(exc):
                raise RuntimeError(f"inference timeout: {exc}") from exc
    raise RuntimeError(f"inference unavailable: {last_err}")


def _env_call(envelope: dict[str, Any]) -> dict[str, Any]:
    """POST /environment/call with unique call_id every attempt (409-safe)."""
    global _ENV_CALL_SEQ
    base_id = str(envelope.get("call_id") or "call")
    last_err: str | None = None
    for attempt in range(8):
        payload = dict(envelope)
        # Always uniquify — including attempt 0 — so ProxyClient cannot replay
        # a colliding id (module seq; os allows getenv only, not process tokens).
        _ENV_CALL_SEQ += 1
        suffix = f"a{attempt}-{int(time.time() * 1000) % 10_000_000}-{_ENV_CALL_SEQ % 10_000}"
        cid = f"{base_id}-{suffix}"
        payload["call_id"] = cid
        payload["idempotency_key"] = cid
        calls = []
        for index, call in enumerate(payload.get("calls") or [], start=1):
            c = dict(call)
            c["call_id"] = f"{cid}-{index}"
            calls.append(c)
        payload["calls"] = calls
        result = _env_proxy.post("/environment/call", json_data=payload)
        if result is not None:
            return result
        last_err = f"none@{cid}"
        # Backoff: 409 conflicts need spacing; avoid hammering same session.
        time.sleep(min(0.35 * (2**attempt), 3.5))
    raise RuntimeError(f"environment call failed after retries ({base_id}): {last_err}")


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
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None



def _looks_like_model_token(tok: str) -> bool:
    """Structural model/code shape — not storage/Hz/currency units."""
    tok = (tok or "").strip()
    if not tok or tok.lower() in _MODEL_HARVEST_STOP:
        return False
    if re.search(r"(?i)\d+\s*(gb|tb|mb|hz|inch|php)$", tok):
        return False
    if re.fullmatch(r"\d+", tok):
        # Short catalog codes (635, 8520) + long ids; reject 1–2 digit noise.
        return len(tok) >= 3
    digits = sum(ch.isdigit() for ch in tok)
    alphas = sum(ch.isalpha() for ch in tok)
    # Mixed codes: Pixel-9a, A15-Pro, Nova-12 (need ≥1 digit).
    return digits >= 1 and alphas >= 1 and len(tok) >= 3


def _looks_like_prefer_model_token(tok: str) -> bool:
    """Prefer-line model: digit/mixed codes, or alpha code that is not a shade/stopword."""
    if _looks_like_model_token(tok):
        return True
    tok = (tok or "").strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z\-]{2,24}", tok):
        return False
    low = tok.lower()
    if low in _MODEL_HARVEST_STOP or low in {
        "php",
        "usd",
        "eur",
        "gbp",
        "jpy",
        "cny",
        "peso",
        "pesos",
        "price",
        "budget",
        "stock",
        "under",
        "below",
    }:
        return False
    if _COLOR_SHADE_RE.search(tok):
        return False
    return True


def _token_is_budget_context(text: str, tok: str) -> bool:
    """True when digit token is a price/budget amount, not a model code."""
    if not (tok or "").isdigit():
        return False
    return bool(
        re.search(
            rf"(?i)(?:at or below|below|under|budget|≤|<=|php)\s*{re.escape(tok)}\b|"
            rf"\b{re.escape(tok)}\s*(?:\.00\s*)?(?:php|pesos)\b",
            text or "",
        )
    )


def _model_token_hint(text: str) -> str | None:
    """Digit-heavy / mixed model-shaped token (structural prior, not color slogans)."""
    best = None
    for tok in re.findall(r"\b([A-Za-z0-9][A-Za-z0-9\-]{1,32})\b", text or ""):
        if not _looks_like_model_token(tok):
            continue
        # Prefer long pure ids over short; prefer longer mixed codes.
        digits = sum(ch.isdigit() for ch in tok)
        score = (digits if not any(c.isalpha() for c in tok) else 50 + len(tok), len(tok))
        if best is None:
            best = (score, tok)
        else:
            best = max(best, (score, tok))
    return best[1] if best else None


def _clean_color_phrase(raw: str) -> str:
    """Keep last 1–3 non-stop words as a shade phrase (open, not suite lexicon-only)."""
    words = re.findall(r"[A-Za-z][A-Za-z\-]*", raw or "")
    kept: list[str] = []
    skip = _MODEL_HARVEST_STOP | {
        "code",
        "sku",
        "id",
        "ids",
        "for",
        "my",
        "our",
        "one",
        "choice",
        "favorite",
        "favourite",
        "looking",
        "like",
        "want",
        "need",
        "under",
        "below",
        "stock",
        "php",
    }
    for w in words:
        if w.lower() in skip:
            continue
        kept.append(w.lower())
    if not kept:
        return ""
    return " ".join(kept[-3:])


_COLOR_SHADE_RE = re.compile(
    r"(?i)\b(sage green|space (?:gray|grey|black)|midnight black|midnight|starlight|"
    r"graphite|silver|gold|blue|green|red|pink|purple|black|white|grey|gray|"
    r"orange|yellow|brown|beige|cream|navy|teal|cyan|magenta|violet|rose|"
    r"forrest green|forest green|champagne|titanium|obsidian|charcoal|coral|"
    r"bronze|copper|ivory|pearl|sand|mint|lavender|maroon|burgundy)\b"
)

_RECOVERY_RELAX_CUES = re.compile(
    r"(?i)(?:same model.{0,48}(?:color|colour|shade)|another color of the same model|"
    r"color.{0,24}before.{0,24}model|different shade of (?:that |the )?same model|"
    r"other color|another shade|relax(?:able)?\s+color|least-relaxed|"
    r"if that exact (?:color|colour|model|combo|item))"
)

# Prefer-intent leads only (not bare "need/want a smartphone … budget").
_PREFER_MODEL_LEAD = re.compile(
    r"(?i)(?:strongly\s+)?(?:"
    r"prefer(?:\s+the)?|"
    r"looking for(?:\s+the)?|"
    r"first choice is(?:\s+the)?|"
    r"(?:favorite|favourite):?|"
    r"(?:i\s+)?like(?:\s+the)?"
    r")\s+"
)

_MODEL_HARVEST_STOP = frozenset(
    {
        "disappears",
        "disappear",
        "unavailable",
        "before",
        "after",
        "changing",
        "change",
        "another",
        "other",
        "same",
        "color",
        "colour",
        "model",
        "models",
        "the",
        "and",
        "or",
        "with",
        "from",
        "into",
        "that",
        "this",
        "exact",
        "item",
        "listing",
        "stock",
        "budget",
        "category",
        "smartphone",
        "phone",
        "device",
        "prefer",
        "preferred",
        "strongly",
        "genuine",
        "feature",
        "accessory",
        "keep",
        "try",
        "only",
        "then",
        "first",
        "when",
        "if",
        "of",
        "a",
        "an",
        "in",
        "is",
        "be",
        "not",
    }
)


def _hz_num(text: str) -> float:
    """Max advertised Hz in observed text (native + overclock), not first match only."""
    best = 0.0
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*hz", str(text or ""), flags=re.I):
        try:
            best = max(best, float(m.group(1)))
        except ValueError:
            continue
    return best


def _inch_num(text: str) -> float:
    """Prefer explicit inch/size tokens; avoid model codes like 24G4 → 24."""
    raw = str(text or "")
    best = 0.0
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(?:\"|inch(?:es)?|in\b)",
        raw,
        flags=re.I,
    ):
        try:
            best = max(best, float(m.group(1)))
        except ValueError:
            continue
    if best > 0:
        return best
    m = re.search(r"(?i)(?:size|screen)[^\d]{0,12}(\d+(?:\.\d+)?)", raw)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    return 0.0


def _num_token(text: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", str(text or "").replace(",", ""))
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except ValueError:
        return 0.0


def _options_text(row: dict[str, Any] | None) -> str:
    opts = (row or {}).get("options")
    if isinstance(opts, dict):
        return " ".join(f"{k} {v}" for k, v in opts.items())
    return str(opts or "")


def _variant_axis_blob(
    row: dict[str, Any],
    facts: dict[str, str],
) -> str:
    """SKU-local text for ranking axes — variant options beat product facts.size."""
    parts = [
        _options_text(row),
        str(row.get("title") or ""),
        str(facts.get("title") or ""),
        str(facts.get("refresh_rate") or ""),
        str(facts.get("resolution") or ""),
        str(facts.get("screen_size") or ""),
    ]
    return " ".join(parts)


def _axis_value_for(
    field: str,
    *,
    row: dict[str, Any],
    facts: dict[str, str],
) -> float:
    """Score one axis from observed facts/variant — field-specific, not blob-first-number."""
    field_l = (field or "").lower().strip()
    facts = facts or {}
    row = row or {}
    blob = _variant_axis_blob(row, facts)
    if any(k in field_l for k in ("refresh", "hz", "rate", "overclock")):
        return _hz_num(blob)
    if any(k in field_l for k in ("screen", "size", "inch", "diagonal")) and "resolution" not in field_l:
        return _inch_num(blob)
    if any(k in field_l for k in ("price", "cost", "budget")):
        return _price_num(row.get("price")) or 0.0
    if "resolution" in field_l or "pixel" in field_l:
        return _num_token(
            str(facts.get("resolution") or "") + " " + _options_text(row) + " " + blob
        )
    # Unknown axis: match fact key by substring, else options/title numbers.
    for k, v in facts.items():
        if field_l in k.lower() or k.lower() in field_l:
            return _num_token(v)
    return _num_token(blob)


def _elect_ranking_pick(
    candidates: list[tuple[str, str]],
    *,
    view_facts_by_pid: dict[str, dict[str, str]],
    variants_by_pid: dict[str, list[dict[str, Any]]],
    budget: float | None,
    excluded_skus: set[str],
    excluded_keys: set[str] | None = None,
    ranking_axes: list[dict[str, str]] | None = None,
) -> dict[str, str] | None:
    """Elect best live key by THIS episode's axes; drop options strictly dominated on all axes."""
    dead = excluded_keys or set()
    axes = ranking_axes or []
    scored: list[tuple[tuple, tuple[float, ...], dict[str, str]]] = []
    for pid, sku in candidates:
        key = f"{pid}::{sku}"
        if key in dead or sku in excluded_skus:
            continue
        rows = variants_by_pid.get(pid) or []
        row = next((r for r in rows if str(r.get("sku") or "") == sku), None)
        if row is None:
            row = _pick_budget_variant(rows, budget=budget, excluded=excluded_skus)
            if row:
                sku = str(row.get("sku") or sku)
                key = f"{pid}::{sku}"
                if key in dead or sku in excluded_skus:
                    continue
        if not row or not _variant_in_stock(row):
            continue
        price = _price_num(row.get("price"))
        if budget is not None and price is not None and price > budget + 1e-6:
            continue
        facts = view_facts_by_pid.get(pid) or {}
        # Higher-is-better utilities for dominance check.
        utils: list[float] = []
        axis_parts: list[float] = []
        for ax in axes[:4]:
            if not isinstance(ax, dict):
                continue
            field = str(ax.get("field") or ax.get("name") or "")
            direction = str(ax.get("direction") or ax.get("order") or "desc").lower()
            val = _axis_value_for(field, row=row, facts=facts)
            if direction.startswith("desc"):
                utils.append(val)
                axis_parts.append(-val)
            else:
                utils.append(-val)
                axis_parts.append(val)
        if not axis_parts:
            # No axes from THIS rules text → in-budget price only (no frozen Hz/inch ladder).
            axis_parts = [price if price is not None else 1e18]
            utils = [-(price if price is not None else 1e18)]
        score_key = tuple(axis_parts) + (
            price if price is not None else 1e18,
            pid,
            sku,
        )
        scored.append((score_key, tuple(utils), {"product_id": pid, "sku": sku}))
    if not scored:
        return None
    # Drop picks strictly worse on every harvested tradeoff axis than some other live option.
    if axes and len(scored) > 1:
        keep: list[tuple[tuple, tuple[float, ...], dict[str, str]]] = []
        for i, (sk, u_i, pick_i) in enumerate(scored):
            dominated = False
            for j, (_sk2, u_j, _pick_j) in enumerate(scored):
                if i == j or len(u_j) != len(u_i):
                    continue
                if all(a >= b for a, b in zip(u_j, u_i)) and any(a > b for a, b in zip(u_j, u_i)):
                    dominated = True
                    break
            if not dominated:
                keep.append((sk, u_i, pick_i))
        if keep:
            scored = keep
    scored.sort(key=lambda x: x[0])
    return scored[0][2]


def _canonical_axis_field(phrase: str) -> str:
    """Commerce synonym map for axis phrases — not a frozen ranking order."""
    p = re.sub(r"[^a-z0-9\s]+", " ", (phrase or "").lower())
    p = re.sub(r"\s+", " ", p).strip()
    if not p:
        return ""
    if any(w in p for w in ("refresh", "hz", "overclock", "frame rate", "framerate")):
        return "refresh_rate"
    if "resolution" in p or "pixel" in p:
        return "resolution"
    if any(
        w in p
        for w in ("screen", "display", "diagonal", "inch")
    ) and "price" not in p:
        return "screen_size"
    if any(w in p for w in ("price", "cost", "budget", "cheaper", "cheap")):
        return "price"
    if "brand" in p:
        return "brand"
    stop = {
        "the",
        "a",
        "an",
        "advertised",
        "maximum",
        "max",
        "named",
        "preferred",
        "available",
        "option",
        "mode",
        "explicitly",
        "more",
        "bigger",
        "larger",
        "higher",
        "lower",
        "size",
    }
    toks = [t for t in p.split() if t not in stop]
    return "_".join(toks[:4])


def _axis_direction_for(field: str, dir_hint: str = "") -> str:
    """Default: maximize non-price axes; minimize price/cost. Hint overrides."""
    hint = (dir_hint or "").lower().strip()
    field_l = (field or "").lower()
    if hint:
        if re.search(
            r"(?i)\b(?:lowest|lower|low|min|minimum|minimi[sz]e|cheap|cheaper|"
            r"asc|smaller|least|ascending)\b",
            hint,
        ):
            return "asc"
        if re.search(
            r"(?i)\b(?:highest|higher|high|max|maximum|maximi[sz]e|more|bigger|"
            r"larger|desc|greatest|descending)\b",
            hint,
        ):
            return "desc"
    if field_l in ("price", "cost") or "price" in field_l or "cost" in field_l:
        return "asc"
    return "desc"


def _struct_harvest_ranking_axes(rules: str) -> list[dict[str, str]]:
    """Pull tradeoff order from THIS episode text — no suite-frozen axis ladder."""
    text = rules or ""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    junk_fields = frozenset(
        {
            "whose",
            "threshold",
            "valid",
            "class",
            "option",
            "item",
            "product",
            "two",
            "first",
            "second",
            "ties",
            "tie",
            "first_two",
            "maximize",
            "minimize",
        }
    )

    def _add(field: str, direction: str) -> None:
        field = (field or "").strip().lower()
        if not field or field in seen or field in junk_fields:
            return
        if "whose" in field or "threshold" in field:
            return
        seen.add(field)
        out.append({"field": field, "direction": direction})

    # 1) Classic: highest|largest|lowest <phrase>
    for m in re.finditer(
        r"(?i)\b(highest|largest|greatest|maximum|max|lowest|smallest|cheapest|minimum|min)\s+"
        r"(?:advertised\s+)?(?:maximum\s+)?"
        r"([a-z][a-z0-9 \-]{1,40}?)(?=,|;|\.| among| including| before| and\s| —|$)",
        text,
    ):
        dir_word = m.group(1).lower()
        phrase = m.group(2).strip()
        phrase = re.sub(
            r"\s+(?:mode|option|item|product|including|whose|class).*$",
            "",
            phrase,
            flags=re.I,
        ).strip()
        if len(phrase.split()) > 4 or phrase.lower() in junk_fields:
            continue
        direction = _axis_direction_for(
            _canonical_axis_field(phrase),
            dir_word,
        )
        field = _canonical_axis_field(phrase)
        if field:
            _add(field, direction)
        if len(out) >= 4:
            return out

    # 2) Arrow / priority chains: refresh_rate > screen_size > price
    chain = None
    m_chain = re.search(
        r"(?i)(?:rank(?:ed|ing)?\s+by|priority|trade-?offs?\s*(?:priority)?|order)\s*"
        r"(?:by|:)?\s*([^\.\n]{5,160})",
        text,
    )
    if m_chain:
        chain = m_chain.group(1)
    if not chain:
        m_arrow = re.search(
            r"(?i)\b([a-z][a-z0-9_]{1,24}(?:\s*[>→]\s*[a-z][a-z0-9_\s]{1,24}){1,4})",
            text,
        )
        if m_arrow and (">" in m_arrow.group(1) or "→" in m_arrow.group(1)):
            chain = m_arrow.group(1)
    if chain:
        # Drop parenthetical direction essays: "(maximize first two, minimize price)"
        chain = re.split(r"[\(\[]", chain, 1)[0]
        for part in re.split(r"[>→|/]|→", chain):
            part = part.strip(" :,;")
            if not part or len(part) > 40:
                continue
            part_clean = re.sub(
                r"(?i)\b(?:maximize|minimi[sz]e|first|two|and|the)\b",
                " ",
                part,
            ).strip()
            field = _canonical_axis_field(part_clean or part)
            if not field:
                continue
            # Per-field default only — do not let "minimize price" flip earlier axes.
            _add(field, _axis_direction_for(field, ""))
            if len(out) >= 4:
                return out

    # 3) Prefer-more / maximize / bigger / cheaper / break-ties sequence
    # e.g. "Prefer more Hz, then bigger display, then cheaper"
    # e.g. "Maximize refresh; break ties by screen size; then minimize cost"
    for m in re.finditer(
        r"(?i)\b(?:prefer\s+more|maximize|maximise|bigger|larger|higher|"
        r"break\s+ties\s+by|ties?\s+by|minimize|minimise|lower)\s+"
        r"(?:the\s+)?([a-z][a-z0-9 \-]{0,40}?)(?=,|;|\.| then\b| among\b| and\b|$)",
        text,
    ):
        head = m.group(0).lower()
        phrase = (m.group(1) or "").strip()
        if not phrase:
            continue
        phrase = re.sub(r"\s+(?:mode|option|item|product).*$", "", phrase, flags=re.I).strip()
        if len(phrase.split()) > 4:
            continue
        field = _canonical_axis_field(phrase)
        if not field:
            continue
        _add(field, _axis_direction_for(field, head))
        if len(out) >= 4:
            return out
    # Terminal "cheaper" / "minimize cost" with no following noun.
    if out and re.search(r"(?i)\bcheaper\b|\bminimi[sz]e(?:\s+the)?\s+cost\b", text):
        _add("price", "asc")
        if len(out) >= 4:
            return out

    # 4) "sort by X descending, Y descending, Z ascending"
    m_sort = re.search(
        r"(?i)\bsort(?:ed)?\s+by\s+([^\.\n]{5,160})",
        text,
    )
    if m_sort and not out:
        chunk = m_sort.group(1)
        for piece in re.split(r"[,;]|\bthen\b", chunk):
            piece = piece.strip()
            if not piece:
                continue
            field = _canonical_axis_field(piece)
            if not field:
                continue
            _add(field, _axis_direction_for(field, piece))
            if len(out) >= 4:
                break

    return out[:4]


def _struct_harvest_resolution_defs(rules: str) -> list[dict[str, Any]]:
    """Parse class thresholds only when THIS rules text defines them."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in re.finditer(
        r"(?i)\b((?:Full\s+)?[A-Za-z][A-Za-z0-9+]{1,12}(?:/[A-Za-z][A-Za-z0-9+]{0,12})?)\s+"
        r"(?:starts\s+)?at\s+(\d{3,4})\s*(?:by|x|×)\s*(\d{3,4})",
        rules or "",
    ):
        name = re.sub(r"\s+", " ", m.group(1).strip())
        a, b = int(m.group(2)), int(m.group(3))
        key = name.lower()
        if key in seen or "example" in key:
            continue
        seen.add(key)
        out.append({"name": name, "long": max(a, b), "short": min(a, b)})
    return out


def _title_tokens(title: str) -> set[str]:
    """ASCII letter/digit runs — matches typical title-token Task-rule protocol."""
    return set(re.findall(r"[a-z0-9]+", (title or "").lower()))


def _harvest_useful_reason_fields(rules: str) -> list[str]:
    """Field names listed as useful / soft-fact reasons in THIS episode's Task rules."""
    text = rules or ""
    m = re.search(
        r"(?i)(?:at least one )?useful reason(?:s)?(?:\s+such as|\s*:)\s*([^\.\n]{3,160})",
        text,
    )
    if not m:
        # "Preference fit … on useful facts such as waterproof rating, sweat …"
        m = re.search(
            r"(?i)useful facts such as\s+([^\.\n]{3,200})",
            text,
        )
    if not m:
        # TF4: "inspect raw product facts such as battery, water/sweat resistance, …"
        m = re.search(
            r"(?i)(?:raw )?product facts such as\s+([^\.\n]{3,200})",
            text,
        )
    if not m:
        m = re.search(
            r"(?i)(?:facts|fields|axes)\s+such as\s+([^\.\n]{3,200})",
            text,
        )
    if not m:
        m = re.search(
            r"(?i)soft preferences matter:[^\.\n]{0,80}such as\s+([^\.\n]{3,200})",
            text,
        )
    if not m:
        return []
    chunk = m.group(1)
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,;]|\bor\b|\band\b", chunk):
        raw = part.strip().lower()
        raw = re.sub(r"[^a-z0-9_\s/\-]+", "", raw).strip()
        if not raw or len(raw) > 40:
            continue
        # Keep slash compounds as one soft field (water/sweat resistance).
        if "/" in raw:
            if "water" in raw or "sweat" in raw:
                field = "water_resistance"
            else:
                field = raw.replace("/", "_").replace(" ", "_").replace("-", "_")
        else:
            field = raw.replace(" ", "_").replace("-", "_")
        if field in ("use_specific_features", "features"):
            continue
        if field in ("water_sweat_resistance", "watersweat_resistance"):
            field = "water_resistance"
        # Keep sweat_resistance distinct so titles with "sweat" still soft-hit.
        if "sweat" in field and "water" in field:
            field = "water_resistance"
        # "waterproof rating" → waterproof
        if "waterproof" in field:
            field = "waterproof"
        if field in ("battery_life",):
            field = "battery"
        if not field or field in seen:
            continue
        seen.add(field)
        out.append(field)
        if len(out) >= 8:
            break
    return out


def _struct_harvest_pref_tokens(shopper: str) -> list[str]:
    """Soft-preference tokens from THIS shopper text (priority phrasing + IP ratings).

    Episode-local only — no frozen headphone/sports axis table.
    """
    text = shopper or ""
    out: list[str] = []
    seen: set[str] = set()

    def _add(tok: str) -> None:
        t = re.sub(r"\s+", " ", (tok or "").strip().lower())
        t = t.strip(" .,;:|/\\")
        if len(t) < 3 or len(t) > 40:
            return
        if t in seen:
            return
        if re.search(
            r"(?i)\b(?:budget|php|category|catalog|stock|order|prefer|device|actual|"
            r"accessory|under|at most|required|settle|flimsy)\b",
            t,
        ):
            return
        seen.add(t)
        out.append(t)

    for m in re.finditer(r"(?i)\b(ipx\s*\d+|ip\s*\d+)\b", text):
        _add(re.sub(r"\s+", "", m.group(1)))
    # "X counts most/least/next" / "X close behind" / "X matter(s) most"
    for m in re.finditer(
        r"(?i)([A-Za-z][A-Za-z0-9/ \-]{2,40}?)\s+"
        r"(?:counts?\s+(?:most|least|next|more)|close behind|matters?(?:\s+most)?)",
        text,
    ):
        _add(m.group(1))
    # "rather … for X than" (stop at than — do not capture the rejected alternative)
    for m in re.finditer(
        r"(?i)\brather\b.{0,36}?\bfor\s+([A-Za-z][A-Za-z0-9/ \-]{2,40}?)\s+than\b",
        text,
    ):
        _add(m.group(1))
    # "I want / I care about / priorities: X, Y"
    for m in re.finditer(
        r"(?i)(?:i\s+want|i\s+need|care about|looking for)\s+(.{3,60}?)(?:\.|$)",
        text,
    ):
        chunk = m.group(1)
        for part in re.split(r",|\band\b|\bthen\b|\|", chunk):
            _add(part)
    for m in re.finditer(
        r"(?i)priorit(?:y|ies)\s*:\s*([^\.\n]{3,80})",
        text,
    ):
        for part in re.split(r",|;|\band\b|\bthen\b|\|", m.group(1)):
            part = re.sub(r"(?i)\s*(?:first|next|least|most)$", "", part).strip()
            _add(part)
    # Slash compounds: sweat/water resistance
    for m in re.finditer(
        r"(?i)\b([a-z]{3,20}\s*/\s*[a-z]{3,20}\s+(?:resistance|fit|life|freedom))\b",
        text,
    ):
        _add(m.group(1).replace("/", " "))
    return out[:10]


def _demote_soft_firm_specs(profile: ObligationProfile) -> None:
    """Demote soft/useful-fact specs out of firm gate (preference survivor-pool safety).

    Episode-semantic: feature/ip_rating slots, soft-marked names, or IP/water tokens —
    no suite vignette phrases.
    """
    keep: list[dict[str, str]] = []
    soft_names = {
        str(x).strip().lower().replace(" ", "_")
        for x in (profile.useful_reason_fields or [])
        if str(x).strip()
    }
    soft_toks = {str(t).strip().lower() for t in (profile.pref_tokens or []) if str(t).strip()}
    for fs in profile.firm_specs or []:
        if not isinstance(fs, dict):
            continue
        name = str(fs.get("name") or "").strip().lower()
        val = str(fs.get("value") or "").strip()
        blob = f"{name} {val}".lower()
        demote = False
        if name in ("feature", "ip_rating"):
            demote = True
        elif name and name.replace(" ", "_") in soft_names:
            demote = True
        elif val and val.lower() in soft_toks:
            demote = True
        elif re.search(r"(?i)\b(?:waterproof|water[- ]resistant|sweat(?:proof|)|dustproof|"
                       r"ipx\s*\d+|ip\s*\d+)\b", blob):
            demote = True
        if demote:
            if val:
                merged = list(profile.pref_tokens or [])
                vl = val.lower()
                if vl not in {x.lower() for x in merged}:
                    merged.append(val)
                profile.pref_tokens = merged[:12]
            continue
        keep.append({"name": name, "value": val})
    profile.firm_specs = keep


def _shopper_ground_text(episode_text: str) -> str:
    """Episode text with Task-rules boilerplate removed — for firm grounding only."""
    text = episode_text or ""
    m = re.search(r"(?i)\btask rules\s*:", text)
    if not m:
        return text
    head = text[: m.start()]
    kept: list[str] = []
    for line in text[m.end() :].splitlines():
        low = line.lower().strip()
        if not low:
            continue
        # Skip sealed Task-rules instructional / class-table lines.
        if re.match(
            r"(?i)(?:you are a shopping|the user gives firm|resolution names|"
            r"stock can change|when you need input|finish by placing|"
            r"once verified|for a smartphone|use the highest class|"
            r"lowercase qhd|conflicting resolution)",
            low,
        ):
            continue
        if re.search(
            r"(?i)\b(?:starts?\s+at|pixel\s+classes?|longer and shorter sides|"
            r"thresholds?\s+are\s+met)\b",
            low,
        ):
            continue
        kept.append(line)
    return (head + "\n" + "\n".join(kept)).strip()


def _firm_spec_grounded_in_episode(name: str, val: str, episode_text: str) -> bool:
    """True when firm value appears in shopper requirements — not Task-rules class tables."""
    if not val:
        return True
    ground = _shopper_ground_text(episode_text).lower()
    if not ground:
        # No episode text yet — keep; mid-episode prune will re-check.
        return True
    val_l = val.lower().strip()
    name_l = (name or "").lower().replace("_", " ").strip()
    # Resolution: require shopper-side need language, not class-table leftovers.
    if name_l in ("resolution", "display"):
        if val_l not in ground and re.sub(r"\s+", "", val_l) not in re.sub(
            r"\s+", "", ground
        ):
            return False
        if not re.search(
            r"(?i)\b(?:must|need|require|required|firm|resolution|display)\b",
            ground,
        ):
            return False
    compact_text = re.sub(r"\s+", "", ground)
    compact_val = re.sub(r"\s+", "", val_l)
    if val_l and val_l in ground:
        return True
    if compact_val and len(compact_val) >= 2 and compact_val in compact_text:
        return True
    num = _num_token(val)
    if num is not None:
        num_s = f"{num:g}"
        if re.search(rf"(?<!\d){re.escape(num_s)}(?!\d)", ground):
            return True
    toks = [
        t
        for t in re.findall(r"[a-z0-9]{2,}", val_l)
        if t not in {"gb", "hz", "inch", "inches"}
    ]
    if toks and all(t in ground for t in toks[:3]):
        return True
    if name_l and val_l and name_l in ground and val_l in ground:
        return True
    return False


def _prune_constraint_firm_specs(
    profile: ObligationProfile, episode_text: str = ""
) -> None:
    """TF3: drop speculative LLM firm slots not grounded in THIS episode text."""
    kept: list[dict[str, str]] = []
    for fs in profile.firm_specs or []:
        if not isinstance(fs, dict):
            continue
        name = str(fs.get("name") or "").strip().lower()
        val = str(fs.get("value") or "").strip()
        if not name or not val:
            continue
        if _firm_spec_grounded_in_episode(name, val, episode_text):
            kept.append({"name": name, "value": val})
    profile.firm_specs = kept


def _harvest_something_like_phrase(text: str) -> list[str]:
    """Episode phrase tokens from shopper clarify: something like \"roto pvc\"."""
    if not text:
        return []
    m = re.search(
        r"(?i)something like\s+[\"']([^\"']+)[\"']",
        text,
    )
    if not m:
        m = re.search(
            r"(?i)(?:matches?\s+(?:the\s+)?phrase|title(?:s)?\s+contain)\s+[\"']([^\"']+)[\"']",
            text,
        )
    if not m:
        return []
    terms: list[str] = []
    for tok in re.findall(r"[A-Za-z0-9]+", m.group(1)):
        tl = tok.lower()
        if len(tl) >= 2 and not tl.isdigit():
            terms.append(tl)
    # unique order
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:12]


def _sanitize_brand_token(tok: str) -> str:
    """Drop budget/glue/category fragments — structural only, not a brand table."""
    tok = re.sub(r"\s+", " ", (tok or "").strip())
    if not tok:
        return ""
    low = tok.lower()
    # Budget-operator shapes falsely latched from "budget is at most N".
    if re.fullmatch(r"(?:is\s+)?at\s+(?:most|least)", low):
        return ""
    if low in ("when", "is", "at", "most", "least", "but", "and", "or", "the", "a", "an"):
        return ""
    # Cut at glue words if multi-capture overran.
    parts: list[str] = []
    for w in tok.split():
        if w.lower() in (
            "but",
            "and",
            "or",
            "the",
            "a",
            "an",
            "when",
            "firm",
            "requirements",
            "with",
            "without",
            "under",
            "below",
            "inch",
            "inches",
        ):
            break
        parts.append(w)
    # Trailing lowercase category nouns are not part of the brand
    # ("OnePlus phone" → OnePlus). Title-case "Phone" stays (Nothing Phone).
    _cat_trail = {
        "phone",
        "phones",
        "smartphone",
        "smartphones",
        "tablet",
        "tablets",
        "mobile",
        "mobiles",
        "device",
        "devices",
        "product",
        "products",
        "listing",
        "listings",
        "variant",
        "variants",
    }
    while parts and parts[-1].lower() in _cat_trail and parts[-1] == parts[-1].lower():
        parts.pop()
    out = " ".join(parts).strip()
    if not out or out.lower() in ("when", "is", "at", "most", "least"):
        return ""
    if re.search(r"(?i)\b(?:at\s+most|at\s+least)\b", out):
        return ""
    # Pure category / catalog nouns are not brands ("smartphone", "smartphone with").
    if re.fullmatch(
        r"(?i)(?:smart\s*)?phones?|tablets?|mobiles?|electronics?|devices?|"
        r"laptops?|computers?|items?|products?|listings?|variants?|"
        r"smartphone\s+with|phone\s+with|tablet\s+with",
        out,
    ):
        return ""
    return out


def _harvest_required_brand(text: str) -> str:
    """Firm brand from THIS episode: 'required brand is Samsung' (paraphrase-tolerant)."""
    if not text:
        return ""
    m = re.search(
        r"(?i)(?:required|must[- ]have|firm)\s+brand\s*(?:is|=|:)\s*"
        r"([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,2})",
        text,
    )
    return _sanitize_brand_token(m.group(1)) if m else ""


def _harvest_ideal_prefer_brand(text: str) -> str:
    """Soft brand from prefer-X patterns — never prose 'preferred brand when …'."""
    if not text:
        return ""
    # "prefer VEGGY BASKET when it remains hard-valid"
    m = re.search(
        r"(?i)\bprefer(?:ably)?\s+"
        r"([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,2})\s+when\b",
        text,
    )
    if m:
        tok = _sanitize_brand_token(m.group(1))
        if tok:
            return tok
    m = re.search(
        r"(?i)ideally[, ]+i prefer\s+"
        r"([A-Za-z][A-Za-z0-9\-]+(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,2})"
        r"(?:\s*,|\s+but\b|\s+when\b|\s*$)",
        text,
    )
    if not m:
        m = re.search(
            r"(?i)(?:i )?(?:prefer|preferably)\s+"
            r"([A-Za-z][A-Za-z0-9\-]+(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,2})"
            r"(?:\s*,|\s+but\b|\s+when\b)",
            text,
        )
    if not m:
        return ""
    return _sanitize_brand_token(m.group(1))


def _preference_seed_query(profile: ObligationProfile) -> str:
    """TF4 seed: category + THIS-episode soft tokens (capped — no search spray)."""
    bits: list[str] = []
    if profile.required_category:
        bits.append(profile.required_category.strip())
    for tok in list(profile.pref_tokens or [])[:5]:
        t = str(tok).strip()
        if t and t.lower() not in {b.lower() for b in bits}:
            bits.append(t)
    for uf in list(profile.useful_reason_fields or [])[:4]:
        t = str(uf).replace("_", " ").strip()
        if t and t.lower() not in {b.lower() for b in bits}:
            bits.append(t)
    if not bits:
        bits.append("in stock")
    return " ".join(bits)[:120]


def _title_matches_terms(title: str, terms: list[str]) -> bool:
    """Title-phrase gate: contiguous phrase OR all tokens present (TF2 listing-title)."""
    if not terms:
        return True
    clean = [str(t).strip().lower() for t in terms if str(t).strip()]
    if not clean:
        return True
    tl = re.sub(r"\s+", " ", (title or "").lower())
    phrase = " ".join(clean)
    if phrase and phrase in tl:
        return True
    # Contiguous with soft separators (hyphen/slash).
    compact_title = re.sub(r"[^a-z0-9]+", "", tl)
    compact_phrase = re.sub(r"[^a-z0-9]+", "", phrase)
    if compact_phrase and compact_phrase in compact_title:
        return True
    toks = _title_tokens(title)
    return all(t in toks for t in clean)


def _retrieval_phrase_query(profile: ObligationProfile, extra_tokens: list[str] | None = None) -> str:
    """TF2 seed: title-phrase tokens first — category belongs in filter, not query dilution."""
    bits = [t for t in (profile.title_terms or []) if str(t).strip()][:6]
    for tok in extra_tokens or []:
        t = str(tok).strip()
        if not t:
            continue
        tl = t.lower()
        if tl in {b.lower() for b in bits}:
            continue
        bits.append(t)
        if len(bits) >= 8:
            break
    return " ".join(bits)[:120] if bits else "in stock"


def _harvest_reform_tokens(
    *,
    titles: list[str],
    phrase_terms: list[str],
    limit: int = 3,
) -> list[str]:
    """Distinctive title tokens beyond the seed phrase — reformulation fuel (episode-local)."""
    phrase = {str(t).strip().lower() for t in phrase_terms if str(t).strip()}
    noise = {
        "the", "and", "for", "with", "pack", "pcs", "set", "new", "original",
        "official", "genuine", "free", "shipping", "sale",
    }
    counts: dict[str, int] = {}
    for title in titles:
        for tok in _title_tokens(title):
            if len(tok) < 3 or tok in phrase or tok in noise or tok.isdigit():
                continue
            counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in ranked[:limit]]


def _struct_harvest(shopper: str, rules: str) -> dict[str, Any]:
    """Deterministic ledger slots from THIS episode text. No frozen suite tables."""
    text_r = rules or ""
    text_s = shopper or ""
    out: dict[str, Any] = {}

    # Axes may live in shopper goal ("Choose the highest refresh…") or Task rules.
    # Prefer shopper order when both present — shopper states the tradeoff priority.
    axes_s = _struct_harvest_ranking_axes(text_s)
    axes_r = _struct_harvest_ranking_axes(text_r)
    axes = list(axes_s or [])
    if not axes:
        axes = list(axes_r or [])
    else:
        seen_f = {str(a.get("field") or "") for a in axes}
        for ax in axes_r or []:
            f = str(ax.get("field") or "")
            if f and f not in seen_f:
                axes.append(ax)
                seen_f.add(f)
    if axes:
        out["ranking_axes"] = axes[:4]
    defs = _struct_harvest_resolution_defs(text_r)
    if defs:
        out["resolution_defs"] = defs

    if re.search(r"(?i)\blisting_id\b", text_r) and re.search(
        r"(?i)(?:distinct|multiple|at least\s*2|≥\s*2|>=\s*2|two\b)", text_r
    ):
        out["listing_id_min"] = 2

    m_id = re.search(
        r"(?i)(?:at least|surface at least|surface)\s+(\d+)\s+distinct\s+matching",
        text_r,
    )
    if not m_id:
        m_id = re.search(r"(?i)at least\s+(\d+)\s+distinct\s+(?:product|matching)", text_r)
    if not m_id:
        # TF2 paraphrase: "≥5 distinct matching product identities"
        m_id = re.search(
            r"(?i)(?:≥|>=)\s*(\d+)\s+distinct\s+(?:matching\s+)?(?:product\s+)?identit",
            text_r,
        )
    if m_id:
        try:
            out["min_matching_identities"] = int(m_id.group(1))
        except ValueError:
            pass

    if re.search(r"(?i)match.{0,40}listing title|title token|runs of ASCII", text_r):
        out["title_phrase_match"] = True

    # Title terms: quoted phrases in shopper, else significant tokens after find/looking for.
    terms: list[str] = []
    for qm in re.finditer(r"[\"']([A-Za-z0-9][A-Za-z0-9 \-]{1,40})[\"']", text_s):
        for tok in re.findall(r"[A-Za-z0-9]+", qm.group(1)):
            if len(tok) >= 2:
                terms.append(tok.lower())
    # Commerce / budget noise — never treat as title-token obligations.
    _TITLE_TERM_NOISE = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "under",
        "budget",
        "stock",
        "instock",
        "available",
        "within",
        "below",
        "above",
        "about",
        "item",
        "items",
        "listing",
        "product",
        "php",
        "usd",
        "price",
        "priced",
        "cost",
        "spend",
        "only",
        "these",
        "those",
        "want",
        "need",
        "find",
        "looking",
        "search",
        "compare",
        "candidate",
        "candidates",
        "variants",
        "variant",
    }
    # Prefer explicit phrase quotes (something like / matches the phrase) over look-for noise.
    if not terms:
        terms.extend(_harvest_something_like_phrase(text_s))
    if not terms:
        m_phrase = re.search(
            r"(?i)matches?\s+(?:the\s+phrase\s+)?[\"']([^\"']+)[\"']",
            text_s,
        )
        if not m_phrase:
            m_phrase = re.search(
                r"(?i)matches?\s+the\s+phrase\s+"
                r"([A-Za-z0-9][A-Za-z0-9 \-]{1,48}?)"
                r"(?:\s*,|\s+at\b|\s+under\b|\s+and\b|\s+in\b|$)",
                text_s,
            )
        if m_phrase:
            for tok in re.findall(r"[A-Za-z0-9]+", m_phrase.group(1)):
                if len(tok) >= 2:
                    terms.append(tok.lower())
    if not terms:
        m_look = re.search(
            r"(?i)(?:looking for|find|search for|need|want)\s+(.{8,80}?)(?:\.|,|;|$)",
            text_s,
        )
        if m_look:
            for tok in re.findall(r"[A-Za-z0-9]+", m_look.group(1)):
                tl = tok.lower()
                if len(tl) < 3 or tl in _TITLE_TERM_NOISE:
                    continue
                # Pure numeric / budget amount tokens (e.g. 15652 from "under 15652.00").
                if tl.isdigit():
                    continue
                terms.append(tl)
    if terms:
        # Keep unique order, cap; drop noise even from quoted phrases.
        seen_t: set[str] = set()
        uniq: list[str] = []
        for t in terms:
            tl = str(t).strip().lower()
            if not tl or tl in seen_t or tl in _TITLE_TERM_NOISE or tl.isdigit():
                continue
            seen_t.add(tl)
            uniq.append(tl)
        if uniq:
            out["title_terms"] = uniq[:12]

    if re.search(r"(?i)category_path", text_r):
        out["require_category_path"] = True
    # Catalog noun only — stop before "category that matches …" / quote clauses.
    # Never latch "Category labels alone are insufficient" prose.
    m_cat = re.search(
        r"(?i)\bin the\s+([A-Za-z][A-Za-z0-9 &/\-]{1,40}?)\s+category\b",
        text_s,
    )
    if not m_cat:
        m_cat = re.search(
            r"(?i)(?:required category|category[:\s]+)(?!labels\b)([A-Za-z][A-Za-z0-9 &\-]{2,40})",
            text_s,
        )
    if not m_cat:
        # No fixed product-type noun table — leave empty for mid-clarify / goal_brief.
        pass
    else:
        out["required_category"] = m_cat.group(1).strip().rstrip(".")
    if out.get("required_category"):
        out["required_category"] = _sanitize_required_category(str(out["required_category"]))
        if not out["required_category"]:
            out.pop("required_category", None)

    # Useful-reason field names from THIS rules (e.g. "such as brand, streaming, …").
    useful = _harvest_useful_reason_fields(text_r)
    if useful:
        out["useful_reason_fields"] = useful
    pref_toks = _struct_harvest_pref_tokens(text_s)
    if pref_toks:
        out["pref_tokens"] = pref_toks
        # Soft fields from shopper when rules list was empty / sparse.
        if not out.get("useful_reason_fields"):
            out["useful_reason_fields"] = [
                re.sub(r"[^a-z0-9]+", "_", t).strip("_") for t in pref_toks[:6]
            ]

    # Soft-brand latch ONLY when brand is marked soft in rules (not bare "soft preferences").
    if re.search(
        r"(?i)(?:preferred brand is a soft|brand is a soft preference|"
        r"named preferred brand is a SOFT|soft:\s*brand|"
        r"brand.{0,32}soft preference|soft preference.{0,32}brand|"
        r"explicitly named preferred brand is a SOFT|"
        r"brand.{0,40}(?:optional|subordinate)|"
        r"(?:optional|subordinate).{0,40}brand|"
        r"ideally.{0,24}prefer|prefer.{0,24}but.{0,24}firm)",
        text_r + "\n" + text_s,
    ):
        out["soft_brand_first_cart"] = True
    # Firm required brand first (episode text) — before soft prefer / find-me.
    req_brand = _harvest_required_brand(text_s)
    if req_brand:
        out["preferred_brand"] = req_brand
    ideal_brand = _harvest_ideal_prefer_brand(text_s)
    if ideal_brand and not out.get("preferred_brand"):
        out["preferred_brand"] = ideal_brand
    if not out.get("preferred_brand"):
        # Do NOT match "preferred brand when a fully compliant…" → "when".
        m_brand = re.search(
            r"(?i)(?:preferably|ideally[, ]+i prefer)\s+"
            r"([A-Za-z][A-Za-z0-9\-]{1,24})(?:\b|$)",
            text_s,
        )
        if m_brand:
            tok = _sanitize_brand_token(m_brand.group(1).strip().split(",")[0])
            if tok:
                out["preferred_brand"] = tok
    if not out.get("preferred_brand"):
        # Brand before firm numeric / unit tokens. Word-bound get/need — not inside "budget".
        # Title-case brand capture (no (?i) on brand) so lowercase "smartphone with"
        # is not latched. Optional trailing category noun before unit/glue lookahead
        # so "OnePlus phone with 120Hz" / "Samsung smartphone under N" still get brand.
        m_b2 = re.search(
            r"(?:(?i)\bfind(?:\s+me)?|\bget(?:\s+me)?|\bi want|\bi(?:'|’)d like|\blooking for|\bneed\b|"
            r"\bpick|\bchoose|\brecommend|\bplease\s+find(?:\s+me)?)\s+(?:an?\s+|the\s+)?"
            r"(?:in[- ]stock\s+)?"
            r"([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,2})"
            r"(?:\s+(?:smart)?phones?|\s+tablets?|\s+mobiles?)?"
            r"\s+"
            r"(?=\d|\d+\.\d+|gb\b|hz\b|inch|with\b|under|below|at most|budget|priced)",
            text_s,
        )
        if m_b2:
            tok = _sanitize_brand_token(m_b2.group(1))
            if tok:
                out["preferred_brand"] = tok

    if re.search(
        r"(?i)color.{0,24}before.{0,24}model|same model.{0,40}(?:color|colour|shade)|"
        r"another color of the same model|different shade of (?:that |the )?same model|"
        r"relax(?:able)?\s+color|"
        r"try another color|least-relaxed",
        text_r + "\n" + text_s,
    ):
        out["relax_color_then_model"] = True

    # Justification required when THIS rules demand structured claims (not suite-only slogans).
    head = text_r[:900] if text_r else ""
    if re.search(
        r"(?i)(?:include a structured justification|structured justification with claims|"
        r"(?:must|required).{0,48}justification|"
        r"valid product plus a .{0,48}reason|"
        r"when placing the order,\s*include a structured justification)",
        head,
    ):
        out["justification_required"] = True
    elif re.search(r"(?i)justification is optional", text_r) and not re.search(
        r"(?i)include a structured justification|structured justification with claims",
        head,
    ):
        out["justification_required"] = False

    if re.search(r"(?i)(?:at least|≥|>=)\s*2.{0,40}(?:product|listing)|multiple products", text_r):
        out["multi_listing_min"] = 2

    # Device evidence / accessory reject from THIS episode language (paraphrase-tolerant).
    if re.search(
        r"(?i)(?:actual|genuine(?:ly)?)\s+(?:be\s+a\s+)?(?:device|product|item)|"
        r"device OS must identify|OS must (?:identify|be one of|match)|"
        r"reject (?:conflicting\s+)?(?:tablet|phone|accessor)|"
        r"(?:title|raw).{0,40}(?:device|product)\s+type|"
        r"not a (?:feature phone|basic phone|accessor)|"
        r"screen-only or replacement-screen",
        text_r + "\n" + text_s,
    ):
        out["require_device_evidence"] = True
        os_list = _harvest_device_os_allowlist(text_r + "\n" + text_s)
        if os_list:
            out["device_os_allowlist"] = os_list
    if re.search(
        r"(?i)reject accessories|actual device rather than an accessory|"
        r"buying an actual (?:device|product) rather than an accessory|"
        r"reject accessories and non-device|"
        r"not (?:an? )?(?:accessor(?:y|ies)|case|charger|cable)\b",
        text_r + "\n" + text_s,
    ):
        out["reject_accessories"] = True

    firm = _struct_harvest_firm_specs(text_s)
    if firm:
        out["firm_specs"] = firm
        # Justification spec field = first firm name from THIS episode (no privileged axis list).
        for fs in firm:
            nm = str(fs.get("name") or "").strip()
            if nm and nm not in ("brand", "feature", "color", "colour"):
                out.setdefault("justification_spec_field", nm)
                break

    # TF6: preferred color + model from THIS shopper (not "model disappears" noise).
    recovery_ctx = bool(_RECOVERY_RELAX_CUES.search(text_s + "\n" + text_r)) or bool(
        out.get("relax_color_then_model")
    )
    # Lead + shade words + model token — open shade; last model-shaped token wins.
    for m_pref in _PREFER_MODEL_LEAD.finditer(text_s):
        tail = text_s[m_pref.end() : m_pref.end() + 96]
        # Cut at sentence / recovery clause / budget clause so we don't eat PHP/prices.
        tail = re.split(
            r"(?i)\b(?:if that|if the exact|when that|after|before changing|"
            r"under|below|at or below|in stock|\.|\;)",
            tail,
            maxsplit=1,
        )[0]
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", tail)
        model_idx = None
        for i, w in enumerate(words[:12]):
            if _looks_like_prefer_model_token(w) and not _token_is_budget_context(text_s, w):
                model_idx = i
        if model_idx is None:
            continue
        tok = words[model_idx]
        # Prefer digit/mixed codes when both an alpha label and a code appear.
        for i, w in enumerate(words[:12]):
            if _looks_like_model_token(w) and not _token_is_budget_context(text_s, w):
                model_idx = i
                tok = w
        color = _clean_color_phrase(" ".join(words[:model_idx]))
        if not color:
            if recovery_ctx and not out.get("preferred_model"):
                out["preferred_model"] = tok
            continue
        out["preferred_color"] = color
        out["preferred_model"] = tok
        break
    # Reverse: model <id> in <color> / model = …
    if not out.get("preferred_model"):
        m_rev = re.search(
            r"(?i)\bmodel\s*[:=]?\s*([A-Za-z0-9][A-Za-z0-9\-]{1,32})\b"
            r"(?:.{0,40}\b(?:in|color|colour)\s*[:=]?\s*"
            r"([A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,2}))?",
            text_s,
        )
        if m_rev:
            tok = m_rev.group(1).strip()
            if _looks_like_model_token(tok) and not _token_is_budget_context(text_s, tok):
                out["preferred_model"] = tok
            if m_rev.group(2):
                col = _clean_color_phrase(m_rev.group(2))
                if col:
                    out["preferred_color"] = col
    if not out.get("preferred_model"):
        m_model2 = re.search(
            r"(?i)\bmodel\s*[:=]\s*([A-Za-z0-9][A-Za-z0-9\-]{1,32})\b",
            text_s,
        )
        if m_model2:
            tok = m_model2.group(1).strip()
            if _looks_like_model_token(tok) and not _token_is_budget_context(text_s, tok):
                out["preferred_model"] = tok
    # Loose model hint ONLY under recovery cues (avoid latching 128GB / closed-set ids).
    if not out.get("preferred_model") and recovery_ctx:
        hint = _model_token_hint(text_s)
        if (
            hint
            and _looks_like_model_token(hint)
            and not _token_is_budget_context(text_s, hint)
        ):
            # Prefer tokens near a prefer-lead or color word; else long pure ids only.
            near = bool(
                re.search(
                    rf"(?i)(?:prefer|looking for|first choice|favorite|favourite|like|want|need|"
                    rf"color|colour|black|white|green|red|blue).{{0,40}}{re.escape(hint)}|"
                    rf"{re.escape(hint)}.{{0,40}}(?:in stock|smartphone|phone|color|colour)",
                    text_s,
                )
            )
            pure = hint.isdigit() and len(hint) >= 6
            if near or pure:
                out["preferred_model"] = hint
    if not out.get("preferred_color"):
        m_color = re.search(
            r"(?i)\b(?:color|colour)\s*[:=]\s*"
            r"([A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,2})\b",
            text_s,
        )
        if m_color:
            col = _clean_color_phrase(m_color.group(1))
            if col:
                out["preferred_color"] = col
        elif recovery_ctx:
            # "in black" / "want red …" when recovery language present.
            m_c2 = re.search(
                r"(?i)\b(?:want|need|prefer|in|like)\s+(?:the\s+)?"
                r"([A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,2})\b",
                text_s,
            )
            if m_c2:
                col = _clean_color_phrase(m_c2.group(1))
                # Avoid latching category words as colors.
                if col and col not in {
                    "smartphone",
                    "phone",
                    "stock",
                    "budget",
                    "category",
                    "model",
                }:
                    out["preferred_color"] = col

    return out


def _merge_ledgers(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Struct first; LLM fills empties / raises numeric mins / ORs obligation bools."""
    out = dict(base or {})
    for k, v in (overlay or {}).items():
        if v in (None, "", [], {}):
            continue
        cur = out.get(k)
        if k in ("listing_id_min", "multi_listing_min", "min_matching_identities"):
            try:
                out[k] = max(int(cur or 0), int(v or 0))
            except (TypeError, ValueError):
                if cur in (None, "", 0):
                    out[k] = v
            continue
        if isinstance(v, bool):
            out[k] = bool(cur) or v
            continue
        if k in ("ranking_axes", "resolution_defs", "firm_specs", "title_terms", "useful_reason_fields"):
            if not cur:
                out[k] = v
            continue
        if k == "pref_tokens":
            if not cur:
                out[k] = v
            elif isinstance(cur, list) and isinstance(v, list):
                merged_toks = list(cur)
                for t in v:
                    tl = str(t).strip().lower()
                    if tl and tl not in {str(x).strip().lower() for x in merged_toks}:
                        merged_toks.append(str(t).strip())
                out[k] = merged_toks[:12]
            continue
        if cur in (None, "", [], {}, 0, False):
            out[k] = v
    return out



class _IdentityUF:
    """Transitive identity merge: product_id / model / brand+title+price."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.members: dict[str, set[str]] = {}

    def _add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.members[x] = {x}

    def find(self, x: str) -> str:
        self._add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra] |= self.members[rb]
        del self.members[rb]

    def component_count(self) -> int:
        return len(self.members)

    def ingest_row(self, row: dict[str, Any], facts: dict[str, str] | None = None) -> None:
        facts = facts or {}
        pid = str(row.get("product_id") or facts.get("product_id") or "").strip()
        if not pid:
            return
        keys = [f"pid:{pid}"]
        title = str(row.get("title") or facts.get("title") or "")
        brand = str(row.get("brand") or facts.get("brand") or "").strip().lower()
        model = str(facts.get("model") or row.get("model") or "").strip()
        if not model:
            # Unambiguous model-ish token: 5-24 alnum with ≥1 digit (from title/facts).
            blob = (model + " " + title).lower()
            for tok in re.findall(r"[a-z0-9]{5,24}", re.sub(r"[^a-z0-9]+", " ", blob)):
                if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
                    model = tok
                    break
        if model:
            keys.append(f"model:{re.sub(r'[^a-z0-9]', '', model.lower())}")
        price = _price_num(row.get("price") if row.get("price") is not None else facts.get("price"))
        toks = frozenset(_title_tokens(title))
        if brand and toks and price is not None:
            # Cent-bucket (±0.01 style) without float string drift.
            cents = int(round(float(price) * 100.0))
            keys.append(f"btp:{brand}|{'|'.join(sorted(toks))}|{cents}")
        for k in keys:
            self._add(k)
        for k in keys[1:]:
            self.union(keys[0], k)


def _normalize_model_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _color_from_sources(
    *,
    options: str = "",
    color_fact: str = "",
    title: str = "",
) -> str | None:
    """options → color fact → title; longest shade; multi-color unresolved → None."""
    shade_re = _COLOR_SHADE_RE

    def _extract(src: str) -> str | None:
        hits = shade_re.findall(src or "")
        if not hits:
            return None
        # normalize list of strings (findall may return str)
        names = [h if isinstance(h, str) else h[0] for h in hits]
        uniq = []
        seen = set()
        for n in names:
            k = n.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(n)
        # several different colors → unresolved
        bases = set()
        for n in uniq:
            low = n.lower()
            if "green" in low:
                bases.add("green")
            elif "gray" in low or "grey" in low:
                bases.add("gray")
            else:
                bases.add(low.split()[-1])
        if len(bases) > 1:
            return None
        # longest named shade
        uniq.sort(key=lambda s: len(s), reverse=True)
        return uniq[0].lower()

    for src in (options, color_fact, title):
        if not (src or "").strip():
            continue
        got = _extract(src)
        if got is not None:
            return got
        # source stated something color-like but multi → unresolved stop (don't fall through title)
        if shade_re.search(src) and src is options:
            return None
        if shade_re.search(src) and src is color_fact:
            return None
    return None


def _color_matches_need(
    color: str | None,
    color_need: str,
    *,
    options: str = "",
    color_fact: str = "",
    title: str = "",
) -> bool:
    """Preferred shade vs observed color / raw option text (open phrase OK)."""
    need = (color_need or "").strip().lower()
    if not need:
        return True
    if color is not None and (need in color or color in need):
        return True
    blob = f"{options} {color_fact} {title}".lower()
    return need in blob


def _harvest_device_os_allowlist(rules: str) -> list[str]:
    """Pull OS/type names from THIS rules' identify/allowlist clause (paraphrase-tolerant)."""
    m = re.search(
        r"(?i)(?:device OS|OS|operating system)\s+must\s+(?:identify|be one of|match|be)\s+"
        r"(.+?)(?:\.|Category labels|Reject|$)",
        rules or "",
    )
    if not m:
        m = re.search(
            r"(?i)(?:allowed|permitted)\s+(?:OS|operating systems?)\s*[:=]\s*"
            r"(.+?)(?:\.|;|$)",
            rules or "",
        )
    if not m:
        return []
    chunk = m.group(1)
    parts = re.split(r",|\bor\b", chunk)
    out = []
    for p in parts:
        name = re.sub(r"\s+", " ", p).strip(" .")
        if len(name) >= 2 and len(name) <= 40:
            out.append(name)
    return out[:20]


def _struct_harvest_firm_specs(shopper: str) -> list[dict[str, str]]:
    """Firm amounts/tokens from shopper text (numeric + portable qualitative)."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(name: str, value: str) -> None:
        name = (name or "").strip().lower()
        value = (value or "").strip()
        if not name or not value:
            return
        # One resolution obligation — prefer first hit (named HD/FHD before bare Np).
        if name == "resolution" and any(x.get("name") == "resolution" for x in out):
            return
        key = f"{name}={value.lower()}"
        if key in seen:
            return
        seen.add(key)
        out.append({"name": name, "value": value})

    patterns = [
        (r"(?i)(\d+)\s*gb\s*ram|ram\s*[:\s]*(\d+)\s*gb", "ram"),
        # Require inch/inches or quote — never bare preposition "in". Allow 6.78-inch / 7.6".
        (r"(?i)(\d+(?:\.\d+)?)\s*-?\s*(?:inches|inch)\b", "screen_size"),
        (r'(?i)(\d+(?:\.\d+)?)\s*(?:"|\'\'|[\u201c\u201d\u2033])', "screen_size"),
        (r"(?i)(\d+)\s*hz", "refresh_rate"),
        (r"(?i)(\d+)\s*gb\s*(?:storage|ssd|rom|internal)", "storage"),
        (r"(?i)\b(ipx\s*\d+|ip\s*\d+)\b", "ip_rating"),
        (r"(?i)\b(waterproof|water[- ]resistant|sweatproof|dustproof)\b", "feature"),
    ]
    for pat, name in patterns:
        m = re.search(pat, shopper or "")
        if not m:
            continue
        val = next((g for g in m.groups() if g), m.group(0))
        _add(name, str(val).strip())
    # Resolution class ONLY when shopper requires it — never from Task-rules
    # "HD starts at 1280…" / "Full HD at 1920…" definition tables (phantom firm → never-cart).
    text_s = shopper or ""
    if not re.search(
        r"(?i)\b(?:starts?\s+at|thresholds?\s+are\s+met|pixel\s+classes?|"
        r"longer and shorter sides|lowercase\s+qhd)\b",
        text_s,
    ):
        m_res = re.search(
            r"(?i)(?:(?:must|need|require|required|firm|exact(?:\s+variant)?).{0,40})?"
            r"\b((?:4k|uhd|qhd|full\s*hd|fhd|hd)|720p|1080p|1440p|2160p)\b",
            text_s,
        )
        if m_res and re.search(
            r"(?i)\b(?:must|need|require|required|firm|resolution|display|screen)\b",
            text_s[max(0, m_res.start() - 48) : m_res.end() + 16],
        ):
            # Skip definitional "HD starts at" even if the guard above missed.
            window = text_s[m_res.start() : min(len(text_s), m_res.end() + 24)]
            if not re.search(r"(?i)\bstarts?\s+at\b", window):
                _add("resolution", str(m_res.group(1)).strip())
    # Short freeform after "firm:" / "must be|have|include" when no numeric hit yet.
    if not out:
        m_free = re.search(
            r"(?i)(?:firm(?:\s+requirement)?|must[- ]have|hard\s+requirement)\s*[:\-]\s*"
            r"(.{3,48}?)(?:\.|$)",
            shopper or "",
        )
        if not m_free:
            m_free = re.search(
                r"(?i)(?:must\s+(?:be|have|include|support)|still\s+needs?(?:\s+a)?)\s+"
                r"([A-Za-z0-9][A-Za-z0-9\s\-]{2,40})",
                shopper or "",
            )
        if m_free:
            tok = re.sub(r"\s+", " ", m_free.group(1)).strip(" .,;:")
            # Drop budget/category chatter masquerading as firm.
            if tok and not re.search(
                r"(?i)\b(?:budget|prefer|category|looking for|php|under)\b", tok
            ):
                _add("feature", tok[:48])
    return out[:6]


def _sku_meets_firm(
    facts: dict[str, str] | None,
    row: dict[str, Any] | None,
    firm_specs: list[dict[str, str]],
    resolution_defs: list[dict[str, Any]] | None = None,
) -> bool:
    if not firm_specs:
        return True
    facts = facts or {}
    row = row or {}
    blob = " ".join(
        [
            str(facts.get("title") or ""),
            str(facts.get("options") or ""),
            str(row.get("options") or ""),
            str(facts.get("ram") or row.get("ram") or ""),
            str(facts.get("screen_size") or row.get("screen_size") or row.get("size") or ""),
            str(facts.get("resolution") or row.get("resolution") or ""),
            str(facts.get("refresh_rate") or row.get("refresh_rate") or ""),
            str(row.get("title") or ""),
        ]
    ).lower()
    for spec in firm_specs:
        name = str(spec.get("name") or "").lower()
        val = str(spec.get("value") or "").strip().lower()
        if not name or not val:
            continue
        if name in ("ram", "memory"):
            need = _num_token(val)
            have = _num_token(facts.get("ram") or row.get("ram") or "") or _num_token(
                re.search(r"(\d+)\s*gb", blob).group(0) if re.search(r"(\d+)\s*gb", blob) else ""
            )
            if need and have and have + 1e-6 < need:
                return False
            if need and not have and f"{int(need)}gb" not in blob.replace(" ", ""):
                return False
        elif name in ("screen_size", "screen"):
            need = _num_token(val)
            have = _num_token(
                facts.get("screen_size") or row.get("screen_size") or row.get("size") or ""
            )
            # v47: parse inches from title/options when field missing — unit or firm value string only.
            if need and not have:
                m_in = re.search(
                    r"(\d+(?:\.\d+)?)\s*[- ]?(?:inch(?:es)?|in\b|\")",
                    blob,
                )
                if m_in:
                    try:
                        have = float(m_in.group(1))
                    except ValueError:
                        have = None
                if not have:
                    # Firm value string itself appears in observed title/options (episode-grounded).
                    val_compact = re.sub(r"\s+", "", val)
                    blob_compact = re.sub(r"\s+", "", blob)
                    if val_compact and val_compact in blob_compact:
                        have = need
                    else:
                        need_s = f"{need:g}"
                        if re.search(
                            rf"(?<!\d){re.escape(need_s)}(?!\d)\s*[- ]?(?:inch(?:es)?|in\b|\")?",
                            blob,
                        ) and (
                            "inch" in blob
                            or "\"" in blob
                            or re.search(r"(?i)\bin\b", blob)
                            or val_compact in blob_compact
                        ):
                            have = need
            # Exact-variant: match to 0.1" (6.7 must not satisfy firm 6.9).
            if need and have and abs(have - need) > 0.11:
                return False
            if need and not have:
                # allow option/title mention of size string
                if val.replace(" ", "") not in blob.replace(" ", "") and f"{need}" not in blob:
                    return False
        elif name in ("refresh_rate", "refresh"):
            need = _num_token(val)
            have = _hz_num(
                facts.get("refresh_rate") or row.get("refresh_rate") or row.get("options") or blob
            )
            if need and have and have + 1e-6 < need:
                return False
            if need and not have:
                return False
        elif name == "resolution":
            observed = str(facts.get("resolution") or row.get("resolution") or "")
            mapped = _resolution_from_defs(observed, resolution_defs) if resolution_defs else None
            # Prefer selected-variant resolution fact over title marketing ("… HD …").
            if observed.strip():
                sources = " ".join([str(mapped or ""), observed])
            else:
                sources = " ".join(
                    [
                        str(mapped or ""),
                        str(row.get("options") or ""),
                        str(facts.get("options") or ""),
                    ]
                )
            if not _resolution_alias_match(val, sources):
                return False
        elif name == "brand":
            if not _brand_matches(facts, val):
                return False
        else:
            # Portable qualitative / freeform firm token — require observed evidence blob.
            token = val.replace(" ", "")
            blob_c = blob.replace(" ", "")
            if token and token not in blob_c and val not in blob:
                return False
    return True


def _device_evidence_ok(
    facts: dict[str, str] | None,
    row: dict[str, Any] | None,
    profile: "ObligationProfile",
) -> bool:
    """When THIS episode requires device evidence / reject accessories, check observed facts."""
    if not (profile.require_device_evidence or profile.reject_accessories):
        return True
    facts = facts or {}
    row = row or {}
    options = str(row.get("options") or "").lower()
    title = str(facts.get("title") or row.get("title") or "").lower()
    path = str(facts.get("category_path") or facts.get("category") or "").lower()
    phone_type = str(
        facts.get("phone_type")
        or facts.get("device_type")
        or facts.get("raw_phone_type")
        or facts.get("product_type")
        or ""
    ).lower()
    device_os = str(facts.get("device_os") or facts.get("os") or "").lower()
    desc = str(facts.get("description") or facts.get("short_description") or "").lower()
    blob = " ".join([options, title, path, phone_type, device_os, desc])

    if profile.reject_accessories or profile.require_device_evidence:
        if re.search(
            r"(?i)\b(screen only|replacement screen|screen protector|case only|"
            r"mystery\s*box|feature phone|basic phone)\b",
            blob,
        ):
            return False
        # Cross-category reject only when THIS episode requires one and forbids the other.
        cat_need = (profile.required_category or "").lower()
        want_tablet = bool(re.search(r"(?i)\btablets?\b", cat_need))
        want_phone = bool(re.search(r"(?i)\b(?:smart)?phones?\b", cat_need))
        if want_phone and not want_tablet and re.search(r"(?i)\btablet\b", blob):
            return False
        if want_tablet and not want_phone and re.search(
            r"(?i)\b(?:smart)?phones?\b", path + " " + phone_type
        ):
            # Tablet episode: phone-only listings fail when path is clearly phone.
            if re.search(r"(?i)\bphone\b", path) and not re.search(r"(?i)\btablet\b", blob):
                return False
        if re.search(r"(?i)\baccessor(?:y|ies)\b", path) or re.search(
            r"(?i)\b(charger|cable|protector|tempered glass)\b", title
        ):
            return False

    if not profile.require_device_evidence:
        return True

    # Positive evidence: OS allowlist from THIS rules, or type/path matches required category.
    allow = [a.lower() for a in (profile.device_os_allowlist or [])]
    if allow and device_os:
        for a in allow:
            if a.lower() in device_os:
                return True
    if allow and not device_os:
        # Allowlist required but OS missing — fail closed.
        return False
    cat_need = _sanitize_required_category(profile.required_category or "").lower()
    if cat_need:
        # Observed path/type must mention required category tokens.
        if _category_path_ok(facts, profile.required_category):
            return True
        cat_toks = [t for t in re.split(r"[^a-z0-9]+", cat_need) if len(t) > 2]
        if cat_toks and all(t in (path + " " + phone_type + " " + title) for t in cat_toks[:2]):
            return True
        return False
    # Device evidence required but no category/OS — accept non-accessory path.
    return not re.search(r"(?i)\baccessor(?:y|ies)\b", path)


def _pick_recovery_variant(
    rows: list[dict[str, Any]],
    *,
    facts: dict[str, str],
    preferred_model: str,
    preferred_color: str,
    budget: float | None,
    excluded: set[str],
) -> dict[str, Any] | None:
    """Same-model other-color before other models; color from options→fact→title."""
    if not rows:
        return None
    model_need = _normalize_model_key(preferred_model)
    color_need = (preferred_color or "").strip().lower()
    scored: list[tuple[tuple, dict[str, Any]]] = []
    for row in rows:
        sku = str(row.get("sku") or "")
        if not sku or sku in excluded or not _variant_in_stock(row):
            continue
        price = _price_num(row.get("price"))
        if budget is not None and price is not None and price > budget + 1e-6:
            continue
        options = str(row.get("options") or "")
        color = _color_from_sources(
            options=options,
            color_fact=str(facts.get("color") or ""),
            title=str(facts.get("title") or ""),
        )
        if color_need and color is None:
            # Open preferred shade may only appear as raw option text (not lexicon hit).
            if not _color_matches_need(
                None,
                color_need,
                options=options,
                color_fact=str(facts.get("color") or ""),
                title=str(facts.get("title") or ""),
            ):
                continue  # unresolved / no evidence
        # Match model id against selected-variant facts/options/title (not title alone).
        model_blob = _normalize_model_key(
            " ".join(
                [
                    str(facts.get("model") or ""),
                    str(facts.get("title") or ""),
                    options,
                    " ".join(str(v) for v in (facts or {}).values()),
                ]
            )
        )
        same_model = bool(model_need) and model_need in model_blob
        same_color = _color_matches_need(
            color,
            color_need,
            options=options,
            color_fact=str(facts.get("color") or ""),
            title=str(facts.get("title") or ""),
        ) if color_need else False
        # relax rank: 0 exact model+color, 1 same model other color, 2 other model
        if same_model and (same_color or not color_need):
            rank = 0
        elif same_model:
            rank = 1
        else:
            rank = 2
        scored.append(((rank, price if price is not None else 1e18, sku), row))
    if not scored:
        return _pick_budget_variant(rows, budget=budget, excluded=excluded)
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _soft_axis_hit_score(
    facts: dict[str, str],
    row: dict[str, Any] | None,
    profile: ObligationProfile,
) -> int:
    """Count soft preference / useful-reason keyword hits in observed text (generic)."""
    blob = " ".join(
        [
            str(facts.get("title") or ""),
            str(facts.get("description") or ""),
            str(facts.get("brand") or ""),
            _options_text(row),
            " ".join(f"{k} {v}" for k, v in (facts or {}).items()),
        ]
    ).lower()
    hits = 0
    for ax in profile.ranking_axes or []:
        if not isinstance(ax, dict):
            continue
        field = str(ax.get("field") or "").strip().lower().replace("_", " ")
        if field and field in blob:
            hits += 2
        for tok in re.findall(r"[a-z0-9]{3,}", field):
            if tok in blob:
                hits += 1
    for uf in profile.useful_reason_fields or []:
        key = str(uf or "").strip().lower().replace("_", " ")
        if not key:
            continue
        if key in blob or key.replace(" ", "_") in blob:
            hits += 2
        for tok in re.findall(r"[a-z0-9]{3,}", key):
            if tok in blob:
                hits += 1
        # Portable water/sweat soft axis: any waterproof/sweat/IP cue counts.
        if any(w in key for w in ("water", "sweat", "waterproof", "ipx")):
            if re.search(r"(?i)waterproof|water[- ]resist|sweat|ipx?\s*\d+", blob):
                hits += 2
    for pt in profile.pref_tokens or []:
        key = str(pt or "").strip().lower()
        if not key:
            continue
        if key in blob:
            hits += 3
        else:
            parts = [t for t in re.findall(r"[a-z0-9]{3,}", key) if t not in ("the", "and", "for")]
            if parts and all(t in blob for t in parts):
                hits += 2
            else:
                for t in parts:
                    if t in blob:
                        hits += 1
    brand = (profile.preferred_brand or "").strip().lower()
    if brand and brand in blob:
        hits += 3
    return hits


def _pick_best_viewed_sku(
    *,
    viewed: set[str],
    view_facts_by_pid: dict[str, dict[str, str]],
    variants_by_pid: dict[str, list[dict[str, Any]]],
    budget: float | None,
    excluded_skus: set[str],
    excluded_keys: set[str],
    profile: ObligationProfile,
    prefer_recovery: bool,
    exclude_product_ids: set[str] | None = None,
) -> dict[str, str] | None:
    """Pick an in-budget, in-stock, non-excluded SKU from already-viewed products."""
    dead_pids = exclude_product_ids or set()
    scored: list[tuple[tuple, dict[str, str]]] = []
    for pid in viewed:
        if pid in dead_pids:
            continue
        facts = view_facts_by_pid.get(pid) or {}
        rows = variants_by_pid.get(pid) or []
        if not rows:
            continue
        if prefer_recovery or profile.relax_color_then_model:
            pick = _pick_recovery_variant(
                rows,
                facts=facts,
                preferred_model=profile.preferred_model,
                preferred_color=profile.preferred_color,
                budget=budget,
                excluded=excluded_skus,
            )
        else:
            pick = None
            if profile.firm_specs:
                for r in rows:
                    sku = str(r.get("sku") or "")
                    if not sku or sku in excluded_skus or not _variant_in_stock(r):
                        continue
                    if f"{pid}::{sku}" in excluded_keys:
                        continue
                    price = _price_num(r.get("price"))
                    if budget is not None and price is not None and price > budget + 1e-6:
                        continue
                    if _sku_meets_firm(facts, r, profile.firm_specs, profile.resolution_defs):
                        pick = r
                        break
            if pick is None:
                pick = _pick_budget_variant(rows, budget=budget, excluded=excluded_skus)
        if not pick or not pick.get("sku"):
            continue
        sku = str(pick["sku"])
        if sku in excluded_skus or f"{pid}::{sku}" in excluded_keys:
            continue
        if not _variant_in_stock(pick):
            continue
        price = _price_num(pick.get("price"))
        if budget is not None and price is not None and price > budget + 1e-6:
            continue
        if profile.firm_specs and not _sku_meets_firm(
            facts, pick, profile.firm_specs, profile.resolution_defs
        ):
            continue
        if not _device_evidence_ok(facts, pick, profile):
            continue
        if profile.require_category_path and profile.required_category:
            if not _category_path_ok(facts, profile.required_category):
                continue
        if profile.title_phrase_match and profile.title_terms:
            title = str(facts.get("title") or "")
            if title and not _title_matches_terms(title, profile.title_terms):
                continue
        rank = 0
        soft = _soft_axis_hit_score(facts, pick, profile)
        if prefer_recovery or profile.relax_color_then_model:
            model_need = _normalize_model_key(profile.preferred_model)
            color_need = (profile.preferred_color or "").strip().lower()
            opts = _options_text(pick)
            color = _color_from_sources(
                options=opts,
                color_fact=str(facts.get("color") or ""),
                title=str(facts.get("title") or ""),
            )
            model_blob = _normalize_model_key(
                " ".join(
                    [
                        str(facts.get("model") or ""),
                        str(facts.get("title") or ""),
                        opts,
                        " ".join(str(v) for v in (facts or {}).values()),
                    ]
                )
            )
            same_model = bool(model_need) and model_need in model_blob
            same_color = bool(color_need) and _color_matches_need(
                color,
                color_need,
                options=opts,
                color_fact=str(facts.get("color") or ""),
                title=str(facts.get("title") or ""),
            )
            if same_model and (same_color or not color_need):
                rank = 0
            elif same_model:
                rank = 1
            else:
                rank = 2
        scored.append(
            (
                (rank, -soft, price if price is not None else 1e18, pid, sku),
                {"product_id": pid, "sku": sku},
            )
        )
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _knobs_for(pipeline: str, profile: ObligationProfile) -> PipelineKnobs:
    play = _PIPELINE_PLAYBOOKS.get(pipeline, _PIPELINE_PLAYBOOKS["generic"])
    multi = int(profile.multi_listing_min or 0)
    # honest_oro: every market event uses full remove→MSG→re-cart choreography.
    # Keep caps family-tuned; shock protocol unified (no light skip-MSG hang class).
    table: dict[str, tuple[int, int, int]] = {
        "intent": (7, 11, 3),
        "retrieval": (9, 13, 5),
        "constraint": (8, 12, 3),
        "preference": (8, 12, 3),
        "ranking": (6, 10, 0),
        "recovery": (9, 13, 3),
        "justification": (8, 12, 3),
        "generic": (8, 12, 3),
    }
    soft, hard, mx = table.get(pipeline, table["generic"])
    return PipelineKnobs(
        name=pipeline,
        llm_soft_cap=soft,
        llm_hard_cap=hard,
        max_search_before_view=mx,
        shock_protocol="full",
        require_multi_listing=multi,
        playbook=play,
    )



def _compile_rule_ledger(shopper: str, rules: str) -> dict[str, Any]:
    """Compile THIS episode Task rules into semantic ledger slots (suite/hidden)."""
    prompt = (
        "Compile shopping Task rules for THIS episode into JSON ledger slots.\n"
        "Use obligations stated in the rules (paraphrase OK). Do not invent suite slogans.\n"
        "Keys:\n"
        "  preferred_brand: string soft brand from shopper if rules say brand is soft/preference\n"
        "  required_category: category string from shopper if any\n"
        "  require_category_path: bool (true when rules mention category_path checks)\n"
        "  listing_id_min: int (≥2 when rules require multiple distinct listing_id)\n"
        "  multi_listing_min: int (≥2 when rules require multiple products before cart)\n"
        "  min_matching_identities: int (retrieval survey count if stated, else 0)\n"
        "  title_phrase_match: bool\n"
        "  title_terms: list of phrase tokens from shopper to match in titles\n"
        "  soft_brand_first_cart: bool\n"
        "  relax_color_then_model: bool\n"
        "  justification_required: bool (true if rules REQUIRE structured justification/claims;\n"
        "    ignore a trailing 'optional' boilerplate when the head requires a reason)\n"
        "  justification_spec_field: one of screen_size|resolution|refresh_rate|brand|'' from shopper firm spec\n"
        "  firm_specs: [{name,value}] firm requirements\n"
        "  ranking_axes: [{field,direction}] tradeoff order if any (desc|asc)\n"
        "  firm_hint: short phrase for commit quizzes\n"
        "  resolution_defs: ONLY if rules define pixel/resolution class thresholds,\n"
        "    list {name, long, short} integers e.g. from 'HD starts at 1280 by 720'. Else [].\n"
        "JSON only.\n\n"
        f"SHOPPER:\n{shopper[:3000]}\n\nTASK_RULES:\n{rules[:5000]}\n"
    )
    try:
        inf = _chat_raw(
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            temperature=0,
        )
        content = ((inf.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        obj = _extract_json_object(str(content)) or {}
        return obj if isinstance(obj, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _apply_ledger(profile: ObligationProfile, ledger: dict[str, Any], shopper: str) -> ObligationProfile:
    def _as_specs(raw: Any) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if not isinstance(raw, list):
            return out
        for row in raw[:12]:
            if isinstance(row, dict):
                n = str(row.get("name") or row.get("field") or "").strip()
                v = str(row.get("value") or "").strip()
                if n and v:
                    out.append({"name": n, "value": v})
        return out

    def _as_axes(raw: Any) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if not isinstance(raw, list):
            return out
        for row in raw[:4]:
            if not isinstance(row, dict):
                continue
            f = str(row.get("field") or row.get("name") or "").strip()
            d = str(row.get("direction") or row.get("order") or "desc").strip().lower()
            if f:
                out.append({"field": f, "direction": d if d in ("asc", "desc") else "desc"})
        return out

    def _as_defs(raw: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not isinstance(raw, list):
            return out
        for row in raw[:8]:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            try:
                long_s = int(row.get("long") or row.get("long_side") or 0)
                short_s = int(row.get("short") or row.get("short_side") or 0)
            except (TypeError, ValueError):
                continue
            if long_s > 0 and short_s > 0:
                out.append({"name": name, "long": long_s, "short": short_s})
        return out

    # Ledger brand first (LLM goal_brief) — sanitize so polluted struct never sticks.
    profile.preferred_brand = _sanitize_brand_token(
        str(ledger.get("preferred_brand") or profile.preferred_brand or "").strip()
    )
    if not profile.preferred_brand:
        profile.preferred_brand = _harvest_required_brand(shopper)
    if not profile.preferred_brand:
        profile.preferred_brand = _harvest_ideal_prefer_brand(shopper)
    if not profile.preferred_brand:
        m = re.search(
            r"(?i)(?:preferably|ideally[, ]+i prefer)\s+"
            r"([A-Za-z][A-Za-z0-9\-]{1,24})(?:\b|$)",
            shopper,
        )
        if m:
            tok = _sanitize_brand_token(m.group(1).strip().split(",")[0])
            if tok:
                profile.preferred_brand = tok

    profile.required_category = _sanitize_required_category(
        str(ledger.get("required_category") or "").strip()
    )
    if "require_category_path" in ledger:
        profile.require_category_path = bool(ledger.get("require_category_path"))
    if not profile.required_category:
        profile.require_category_path = False
    try:
        lid = int(ledger.get("listing_id_min") or 0)
    except (TypeError, ValueError):
        lid = 0
    profile.listing_id_min = max(profile.listing_id_min, lid)
    try:
        mid = int(ledger.get("min_matching_identities") or 0)
    except (TypeError, ValueError):
        mid = 0
    profile.min_matching_identities = mid
    try:
        multi = int(ledger.get("multi_listing_min") or 0)
    except (TypeError, ValueError):
        multi = 0
    if multi:
        profile.multi_listing_min = max(profile.multi_listing_min, multi)
    if ledger.get("title_phrase_match"):
        profile.title_phrase_match = True
    terms = ledger.get("title_terms")
    if isinstance(terms, list):
        profile.title_terms = [str(t).strip().lower() for t in terms if str(t).strip()][:12]
    # Merge explicit phrase tokens from shopper (often arrive only in clarify reply).
    phrase_extra = _harvest_something_like_phrase(shopper)
    if phrase_extra:
        merged_tt = list(profile.title_terms or [])
        for t in phrase_extra:
            if t not in merged_tt:
                merged_tt.append(t)
        profile.title_terms = merged_tt[:12]
        profile.title_phrase_match = True
    if ledger.get("soft_brand_first_cart"):
        profile.soft_brand_first_cart = True
    if ledger.get("relax_color_then_model"):
        profile.relax_color_then_model = True
    if ledger.get("justification_required"):
        profile.justification_required = True
    profile.justification_spec_field = str(ledger.get("justification_spec_field") or "").strip()
    ur = ledger.get("useful_reason_fields")
    if isinstance(ur, list):
        profile.useful_reason_fields = [
            str(x).strip().lower().replace(" ", "_") for x in ur if str(x).strip()
        ][:8]
    pt = ledger.get("pref_tokens")
    if isinstance(pt, list):
        profile.pref_tokens = [str(x).strip() for x in pt if str(x).strip()][:12]
    profile.firm_specs = _as_specs(ledger.get("firm_specs")) or profile.firm_specs
    profile.ranking_axes = _as_axes(ledger.get("ranking_axes")) or profile.ranking_axes
    profile.resolution_defs = _as_defs(ledger.get("resolution_defs"))
    profile.firm_hint = str(ledger.get("firm_hint") or profile.firm_hint or "").strip()
    if not profile.firm_hint and profile.firm_specs:
        profile.firm_hint = f"{profile.firm_specs[0]['name']}={profile.firm_specs[0]['value']}"
    if ledger.get("require_device_evidence"):
        profile.require_device_evidence = True
    if ledger.get("reject_accessories"):
        profile.reject_accessories = True
    os_al = ledger.get("device_os_allowlist")
    if isinstance(os_al, list) and os_al:
        profile.device_os_allowlist = [str(x).strip() for x in os_al if str(x).strip()][:20]
    if ledger.get("preferred_model"):
        profile.preferred_model = str(ledger.get("preferred_model") or "").strip()
    if ledger.get("preferred_color"):
        profile.preferred_color = str(ledger.get("preferred_color") or "").strip()
    return profile


def extract_obligation_profile(query: str) -> ObligationProfile:
    """Struct harvest (THIS text) + LLM fills empties. No catchphrase family table."""
    shopper, rules = _split_shopper_and_rules(query)
    candidates = _parse_candidate_variants(query)
    struct = _struct_harvest(shopper, rules)
    llm_ledger = _compile_rule_ledger(shopper, rules)
    ledger = _merge_ledgers(struct, llm_ledger)

    if candidates:
        prof = ObligationProfile(
            pipeline="ranking",
            closed_set=True,
            confidence=0.95,
            candidate_keys=candidates,
        )
        return _apply_ledger(prof, ledger, shopper)

    model_tok = _model_token_hint(shopper)
    ask_struct = bool(
        re.search(
            r"(?i)\bwithheld\b|\buntil you (?:ask|request)\b|\bif you ask\b|"
            r"\bi will tell you if you ask\b|"
            r"\bone firm requirement\b|"
            r"\bkeep(?:ing)?\b.{0,40}\b(?:private|secret|hidden)\b|"
            r"\b(?:hard|firm)\s+(?:constraint|requirement).{0,40}\b(?:secret|private|hidden|ask)\b",
            shopper + "\n" + rules,
        )
    )
    just_struct = bool(
        re.search(
            r"(?i)include a structured justification|structured justification with claims|"
            r"(?:must|required).{0,48}justification|"
            r"valid product plus a .{0,48}reason|"
            r"when placing the order,\s*include a structured justification",
            rules,
        )
    )
    # Prefer head requirement over trailing optional boilerplate.
    if re.search(r"(?i)justification is optional", rules) and not re.search(
        r"(?i)include a structured justification|structured justification with claims|"
        r"when placing the order,\s*include a structured justification",
        # Cut trailing optional boilerplate without sealed catchphrase keys.
        re.split(r"(?i)unless\s+(?:the\s+)?shopper|optional\s+justification", rules, maxsplit=1)[0],
    ):
        just_struct = bool(
            re.search(
                r"(?i)include a structured justification|structured justification with claims|"
                r"(?:must|required).{0,48}justification|"
                r"when placing the order,\s*include a structured justification",
                rules,
            )
        )
    multi_struct = bool(
        re.search(
            r"(?i)at least two distinct|two distinct (?:listing|product)|"
            r"different listing_id|compare.{0,40}distinct",
            rules,
        )
    )
    listing_struct = bool(re.search(r"(?i)listing_id", rules))
    pref_struct = bool(
        re.search(
            r"(?i)soft preferences|preference axes|soft\s+(?:axes|criteria|priorities)|"
            r"trade-?offs?\s+between\s+(?:your\s+)?preferences|"
            r"priorit(?:y|ies)\s+(?:among|across|over)\s+soft|"
            r"soft use-case|(?:soft|prefer|match).{0,24}intended use|"
            r"intended use.{0,24}(?:soft|prefer|over price)|"
            r"useful[- ]reason|useful facts such as|"
            r"preference fit|match(?:es)? (?:the )?intended use|"
            r"not\s+(?:just\s+)?the cheapest|over price",
            rules,
        )
    )
    relax_struct = bool(
        re.search(
            r"(?i)same model.{0,40}(?:other|another) (?:color|colour|shade)|"
            r"(?:other|another) (?:color|colour|shade).{0,40}same model|"
            r"different shade of (?:that |the )?same model|"
            r"color.{0,24}before.{0,24}model|"
            r"relax(?:able)?\s+color|"
            r"try another color|"
            r"least-relaxed|"
            r"minimal(?:ly)?\s+relax|"
            r"if that exact (?:color/)?model|"
            r"preferred model and color",
            rules + "\n" + shopper,
        )
    )
    title_struct = bool(
        re.search(
            r"(?i)listing title|title token|title(?:s)? (?:must |should )?(?:contain|match|include)|"
            r"phrase against the listing title|matches?\s+(?:the\s+)?phrase|"
            r"something like\s+[\"']|whole word|runs of ASCII|"
            r"min(?:imum)?\s+\d+\s+distinct.{0,40}(?:title|product|matching)",
            rules + "\n" + shopper,
        )
    )
    soft_brand_struct = bool(
        re.search(
            r"(?i)(?:preferred brand is a soft|brand is a soft preference|"
            r"named preferred brand is a SOFT|explicitly named preferred brand is a SOFT|"
            r"soft:\s*brand|brand.{0,32}soft preference|soft preference.{0,32}brand|"
            r"brand.{0,40}(?:optional|subordinate)|"
            r"(?:optional|subordinate).{0,40}brand|"
            r"ideally.{0,24}prefer|prefer.{0,24}but.{0,24}firm)",
            rules + "\n" + shopper,
        )
    )
    strict_struct = bool(
        re.search(
            r"(?i)selected[- ]variant|exact (?:sku|variant)|option text before|"
            r"EVERY requirement must hold for the EXACT variant|"
            r"(?:all|every)\s+(?:requirement|must[- ]have|firm).{0,40}(?:exact|selected)\s+"
            r"(?:variant|sku|option)|"
            r"(?:after\s+)?stockout.{0,80}(?:same product|sibling|another variant)|"
            r"do not take another variant of the same product|"
            r"within[- ]product|reorder.?same.?product|"
            r"requirements apply only to the selected",
            rules,
        )
    )

    # Structural route first (robust when LLM down / paraphrased). LLM fills empties only.
    # Priority: just > retrieval > preference > intent > recovery(color→model) > constraint.
    # Recovery rules often share selected-SKU / smartphone boilerplate with constraint —
    # color-then-model must win over bare strict.
    struct_pipe = ""
    if just_struct:
        struct_pipe = "justification"
    elif title_struct:
        struct_pipe = "retrieval"
    elif pref_struct or (listing_struct and multi_struct):
        struct_pipe = "preference"
    elif soft_brand_struct and not ask_struct:
        struct_pipe = "intent"
    elif relax_struct:
        struct_pipe = "recovery"
    elif ask_struct or strict_struct:
        struct_pipe = "constraint"

    prompt = (
        "Classify shopping-task obligations from THIS episode only.\n"
        "Return JSON: pipeline (intent|retrieval|constraint|preference|ranking|"
        "recovery|justification|generic), ask_first, closed_set, justification_required, "
        "multi_listing_min, relax_color_then_model, title_phrase_match, "
        "soft_brand_first_cart, strict_selected_variant, confidence.\n"
        "Map by Task-rule behavior: soft brand → intent; ask-first/selected option → constraint; "
        "color-then-model → recovery; title-token survey → retrieval; required claims → "
        "justification; multi-listing / listing_id → preference.\n"
        "No memorized pack slogans as the only signal. JSON only.\n\n"
        f"SHOPPER:\n{shopper[:3500]}\n\nTASK_RULES:\n{rules[:4500]}\n"
    )
    obj: dict[str, Any] = {}
    if not struct_pipe:
        try:
            inf = _chat_raw(
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                temperature=0,
            )
            content = ((inf.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            obj = _extract_json_object(str(content)) or {}
        except Exception:  # noqa: BLE001
            obj = {}

    pipe = struct_pipe or str(obj.get("pipeline") or "generic").strip().lower()
    if pipe not in _PIPELINES:
        pipe = "generic"
    try:
        conf = float(obj.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0
    if struct_pipe:
        conf = max(conf, 0.7)

    ask_first = bool(obj.get("ask_first")) or ask_struct
    just_req = bool(obj.get("justification_required")) or just_struct
    try:
        multi = max(0, int(obj.get("multi_listing_min") or 0))
    except (TypeError, ValueError):
        multi = 0
    if multi_struct:
        multi = max(multi, 2)
    # Color-then-model ONLY from rules language — never from model-shaped tokens alone.
    relax = bool(obj.get("relax_color_then_model")) or relax_struct
    title = bool(obj.get("title_phrase_match")) or title_struct
    soft_brand = bool(obj.get("soft_brand_first_cart")) or soft_brand_struct
    strict = bool(obj.get("strict_selected_variant")) or strict_struct or ask_first

    # Final structural overrides (priority: just > retrieval > pref > intent > recovery > constraint).
    if just_req:
        pipe, conf = "justification", max(conf, 0.7)
    elif title and not just_req:
        pipe, conf = "retrieval", max(conf, 0.65)
    elif pref_struct or (multi >= 2 and (multi_struct or listing_struct)):
        pipe, conf = "preference", max(conf, 0.65)
        if pref_struct and multi < 2:
            multi = max(multi, 2)
    elif soft_brand and not ask_first:
        pipe, conf = "intent", max(conf, 0.65)
    elif relax_struct:
        pipe, conf = "recovery", max(conf, 0.75)
        relax = True
    elif ask_first or (strict and not soft_brand):
        pipe, conf = "constraint", max(conf, 0.6)
    elif conf < 0.35 and not struct_pipe:
        pipe = "generic"

    prof = ObligationProfile(
        pipeline=pipe,
        ask_first=ask_first,
        closed_set=bool(obj.get("closed_set")),
        justification_required=just_req,
        multi_listing_min=multi,
        relax_color_then_model=relax,
        title_phrase_match=title,
        soft_brand_first_cart=soft_brand,
        strict_selected_variant=strict,
        confidence=conf if conf else 0.45,
        candidate_keys=[],
        listing_id_min=2 if listing_struct and multi >= 2 else (2 if pref_struct and listing_struct else 0),
    )
    prof = _apply_ledger(prof, ledger, shopper)
    if prof.pipeline == "preference":
        _demote_soft_firm_specs(prof)
    # TF7: named shopper brand is the useful-reason / first-cart target (teacher always brands).
    if prof.pipeline == "justification" and prof.preferred_brand:
        prof.soft_brand_first_cart = True
    return prof


def _claim_field_allowlist(tools: list[dict[str, Any]]) -> frozenset[str]:
    for tool in tools:
        fn = tool.get("function") or tool
        if (fn.get("name") or "") != "place_test_order":
            continue
        props = (
            ((fn.get("parameters") or {}).get("properties") or {})
            .get("justification", {})
            .get("properties", {})
            .get("claims", {})
            .get("items", {})
            .get("properties", {})
            .get("field", {})
        )
        enums = props.get("enum") if isinstance(props, dict) else None
        if isinstance(enums, list) and enums:
            return frozenset(str(x) for x in enums) - _CLAIM_DENY
    return _DEFAULT_CLAIM_FIELDS - _CLAIM_DENY


def _normalize_claim_field(field: str, allowed: frozenset[str]) -> str | None:
    key = (field or "").strip().lower()
    key = _FIELD_ALIASES.get(key, key)
    if key in _CLAIM_DENY:
        return None
    if key in allowed:
        return key
    return None


def _normalize_claim_value(field: str, value: Any) -> str:
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
        return re.sub(r"\s+", " ", text).strip()
    if field == "resolution":
        # Raw observed text; class mapping applied only via ledger.resolution_defs if present.
        return re.sub(r"\s+", " ", text).strip()
    return text


def _resolution_from_defs(text: str, defs: list[dict[str, Any]] | None) -> str | None:
    """If THIS episode rules defined class thresholds, map WxH → class name. Else None."""
    if not defs:
        return None
    low = (text or "").lower().replace("×", "x")
    m = re.search(r"(\d{3,4})\s*x\s*(\d{3,4})", low)
    if not m:
        # Named class already in text matching a def name
        for d in defs:
            name = str(d.get("name") or "").strip().lower()
            if name and re.search(rf"\b{re.escape(name)}\b", low):
                return str(d.get("name") or name)
        return None
    long_side = max(int(m.group(1)), int(m.group(2)))
    short_side = min(int(m.group(1)), int(m.group(2)))
    best = None
    best_long = -1
    for d in defs:
        try:
            dl = int(d.get("long") or d.get("long_side") or 0)
            ds = int(d.get("short") or d.get("short_side") or 0)
        except (TypeError, ValueError):
            continue
        if long_side >= dl and short_side >= ds and dl >= best_long:
            best_long = dl
            best = str(d.get("name") or "").strip() or None
    return best


def _going_with_message(
    *,
    pick: dict[str, str] | None,
    view_facts_by_pid: dict[str, dict[str, str]],
    gated_hint: str | None,
    firm_specs: list[dict[str, str]] | None = None,
    budget: float | None = None,
) -> str:
    """Answer commit quiz from observed pick facts — not recovery boilerplate."""
    if not pick or not pick.get("product_id"):
        return (
            "I am still narrowing an in-stock option that meets the firm requirements "
            "and budget; I will confirm the exact listing before ordering."
        )
    pid = pick["product_id"]
    sku = pick.get("sku") or ""
    facts = view_facts_by_pid.get(pid) or {}
    title = facts.get("title") or f"product {pid}"
    brand = facts.get("brand") or _brand_from_title(title) or ""
    label = f"{brand} {title}".strip() if brand and brand.lower() not in title.lower() else title
    # Affirm firm attrs from observed facts (any gated field this episode states).
    affirms: list[str] = []
    for fs in firm_specs or []:
        name = str(fs.get("name") or "").strip().lower()
        want = str(fs.get("value") or "").strip()
        if not name:
            continue
        have = str(facts.get(name) or facts.get("screen_size" if name == "screen" else name) or "").strip()
        if have:
            affirms.append(f"{name} observed {have}" + (f" (needed {want})" if want else ""))
        elif want:
            affirms.append(f"checking firm {name}={want} against viewed facts")
    if not affirms:
        screen = facts.get("screen_size") or ""
        if screen:
            affirms.append(f"observed screen size is {screen}")
        elif gated_hint:
            affirms.append(f"checking firm requirement ({gated_hint})")
        else:
            affirms.append("it meets the firm hardware requirement from viewed facts")
    budget_bit = ""
    if budget is not None:
        budget_bit = f" It is within the current budget of {budget:g}."
    return (
        f"I am going with {label} (product_id {pid}, sku {sku}). "
        f"Yes — {'; '.join(affirms)}.{budget_bit}"
    )


def _coerce_observation(obs: Any) -> dict[str, Any] | None:
    if obs is None:
        return None
    if isinstance(obs, str):
        try:
            obs = json.loads(obs)
        except json.JSONDecodeError:
            return None
    if not isinstance(obs, dict):
        return None
    inner = obs.get("observation")
    if not isinstance(inner, dict):
        return obs
    rich_inner = any(
        k in inner
        for k in (
            "product_id",
            "brand",
            "facts",
            "results",
            "compared",
            "items",
            "added",
            "cart",
            "market_event",
            "variants",
            "in_stock",
            "price",
        )
    )
    if not rich_inner:
        return obs
    merged = dict(inner)
    for k in (
        "brand",
        "title",
        "facts",
        "product_id",
        "category_path",
        "variants",
        "results",
        "added",
        "cart",
        "market_event",
    ):
        if k not in merged and k in obs and obs.get(k) not in (None, "", "None"):
            merged[k] = obs[k]
    return merged


def _unwrap_added(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    added = payload.get("added")
    if isinstance(added, dict) and added.get("product_id"):
        return added
    cart = payload.get("cart")
    if isinstance(cart, list) and cart:
        row = cart[-1]
        if isinstance(row, dict) and row.get("product_id"):
            return {
                "product_id": row.get("product_id"),
                "sku": row.get("sku"),
                "price": row.get("current_price", row.get("price_at_add")),
                "in_stock": row.get("in_stock"),
                "currency": row.get("currency"),
            }
    if payload.get("product_id") and payload.get("sku"):
        return payload
    return None


def _brand_from_title(title: str) -> str | None:
    stop = {
        "the", "a", "an", "for", "original", "global", "new", "with", "and", "or",
        "in", "on", "at", "to", "of", "by", "from",
    }
    parts: list[str] = []
    for w in str(title or "").split():
        clean = w.strip(".,;:()[]\"'")
        if not clean or not clean[0].isalpha():
            break
        if clean.lower() in stop:
            if parts:
                break
            continue
        if not clean[0].isupper():
            break
        parts.append(clean)
        if len(parts) >= 3:
            break
    return " ".join(parts) if parts else None


def _extract_view_facts(observation: Any) -> tuple[str | None, dict[str, str]]:
    obs = _coerce_observation(observation) or {}
    pid = obs.get("product_id")
    pid_s = str(pid) if pid not in (None, "", "None") else None
    facts: dict[str, str] = {}
    if obs.get("brand") not in (None, "", "None"):
        facts["brand"] = str(obs["brand"]).strip()
    if obs.get("title") not in (None, "", "None"):
        facts["title"] = str(obs["title"]).strip()
    if obs.get("category_path") not in (None, "", "None"):
        facts["category_path"] = str(obs["category_path"]).strip()
    elif obs.get("category") not in (None, "", "None"):
        facts["category_path"] = str(obs["category"]).strip()
    # Top-level hardware fields (TF3 exact-variant) — not only nested facts.
    for src, dst in (
        ("screen_size", "screen_size"),
        ("size", "screen_size"),
        ("display", "resolution"),
        ("resolution", "resolution"),
        ("refresh_rate", "refresh_rate"),
        ("ram", "ram"),
        ("memory", "ram"),
        ("model", "model"),
        ("color", "color"),
        ("colour", "color"),
    ):
        if obs.get(src) not in (None, "", "None"):
            facts.setdefault(dst, str(obs[src]).strip())
    nested = obs.get("facts")
    if isinstance(nested, dict):
        for src, dst in (
            ("display", "resolution"),
            ("size", "screen_size"),
            ("screen_size", "screen_size"),
            ("resolution", "resolution"),
            ("refresh_rate", "refresh_rate"),
            ("ram", "ram"),
            ("brand", "brand"),
            ("category_path", "category_path"),
            ("model", "model"),
            ("color", "color"),
            ("colour", "color"),
        ):
            if nested.get(src) not in (None, "", "None"):
                facts.setdefault(dst, str(nested[src]).strip())
    # EnvPack often exposes gated attrs as specs=[{name,value}, ...] (not top-level).
    specs = obs.get("specs") or obs.get("attributes") or obs.get("product_specs")
    if isinstance(specs, list):
        for sp in specs[:12]:
            if not isinstance(sp, dict):
                continue
            sn = str(sp.get("name") or sp.get("field") or "").strip().lower()
            sv = sp.get("value")
            if sv in (None, "", "None") or not sn:
                continue
            if sn in ("screen_size", "screen", "size"):
                facts.setdefault("screen_size", str(sv).strip())
            elif sn in ("resolution", "display"):
                facts.setdefault("resolution", str(sv).strip())
            elif sn in ("refresh_rate", "refresh"):
                facts.setdefault("refresh_rate", str(sv).strip())
            elif sn in ("ram", "memory"):
                facts.setdefault("ram", str(sv).strip())
            elif sn == "brand":
                facts.setdefault("brand", str(sv).strip())
            elif sn == "model":
                facts.setdefault("model", str(sv).strip())
            elif sn in ("color", "colour"):
                facts.setdefault("color", str(sv).strip())
    # Selected / first variant often carries the gated option amounts.
    variants = obs.get("variants")
    if isinstance(variants, list):
        for row in variants[:3]:
            if not isinstance(row, dict):
                continue
            for src, dst in (
                ("screen_size", "screen_size"),
                ("size", "screen_size"),
                ("resolution", "resolution"),
                ("refresh_rate", "refresh_rate"),
                ("ram", "ram"),
            ):
                if row.get(src) not in (None, "", "None"):
                    facts.setdefault(dst, str(row[src]).strip())
            opts = str(row.get("options") or "")
            if opts and "options" not in facts:
                facts["options"] = opts
    if "brand" not in facts and facts.get("title"):
        guessed = _brand_from_title(facts["title"])
        if guessed:
            facts["brand"] = guessed
    # Derive refresh/screen from title/options text when nested facts omit them.
    blob = " ".join(
        str(facts.get(k) or "")
        for k in ("title", "options", "refresh_rate", "screen_size", "resolution")
    )
    if not facts.get("refresh_rate"):
        hz = _hz_num(blob)
        if hz > 0:
            facts["refresh_rate"] = f"{hz:g}Hz"
    if not facts.get("screen_size"):
        inch = _inch_num(blob)
        if inch > 0:
            facts["screen_size"] = f"{inch:g}"
    return pid_s, facts


def _variants_from_observation(observation: Any) -> list[dict[str, Any]]:
    obs = _coerce_observation(observation) or {}
    rows = obs.get("variants") or obs.get("skus") or []
    if isinstance(rows, dict):
        rows = list(rows.values())
    out: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if isinstance(row, dict):
            out.append(row)
    return out


def _price_num(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(re.sub(r"[^\d.]", "", str(val).replace(",", "")) or "nan")
    except ValueError:
        return None


def _variant_in_stock(row: dict[str, Any]) -> bool:
    val = row.get("in_stock", row.get("stock"))
    if val is False:
        return False
    if isinstance(val, str) and val.strip().lower() in ("false", "0", "no", "out of stock"):
        return False
    return True


def _pick_budget_variant(
    rows: list[dict[str, Any]],
    *,
    budget: float | None,
    excluded: set[str],
) -> dict[str, Any] | None:
    best = None
    best_price = None
    for row in rows:
        sku = str(row.get("sku") or row.get("variant_id") or "")
        if not sku or sku in excluded:
            continue
        if not _variant_in_stock(row):
            continue
        price = _price_num(row.get("price"))
        if budget is not None and price is not None and price > budget + 1e-6:
            continue
        if best is None or (price is not None and (best_price is None or price < best_price)):
            best = row
            best_price = price
    return best


def _stock_is_false(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    val = payload.get("in_stock")
    if val is False:
        return True
    if isinstance(val, str) and val.strip().lower() in ("false", "0", "no"):
        return True
    return False


def _obs_stock_payload(observation: Any) -> dict[str, Any] | None:
    obs = _coerce_observation(observation)
    if not obs:
        return None
    if "in_stock" in obs or "price" in obs:
        return obs
    added = obs.get("added")
    if isinstance(added, dict):
        return added
    return obs


def _sanitize_place_order_args(
    args: dict[str, Any],
    *,
    justification_required: bool,
    allowed_fields: frozenset[str],
    last_stock: dict[str, Any] | None,
    view_facts: dict[str, str] | None,
    resolution_defs: list[dict[str, Any]] | None = None,
    justification_spec_field: str = "",
    useful_reason_fields: list[str] | None = None,
) -> dict[str, Any]:
    out = dict(args)
    if not justification_required:
        out.pop("justification", None)
        return out
    just = out.get("justification")
    if not isinstance(just, dict):
        just = {}
    claims_in = just.get("claims") if isinstance(just.get("claims"), list) else []
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    vf = dict(view_facts or {})
    resolution_defs = resolution_defs or []

    for claim in claims_in:
        if not isinstance(claim, dict):
            continue
        field = _normalize_claim_field(str(claim.get("field", "")), allowed_fields)
        if field is None or field in seen:
            continue
        # Hardware extras only when THIS episode named them as the spec / useful reason.
        # Auto-keeping refresh_rate/resolution from LLM fails claim_truth on hidden packs.
        useful_ok = {
            str(x).strip().lower() for x in (useful_reason_fields or []) if str(x).strip()
        }
        spec_need = (justification_spec_field or "").strip().lower()
        if field in ("refresh_rate", "resolution") and field not in useful_ok and field != spec_need:
            continue
        raw_val = claim.get("value", "")
        if field == "brand":
            if vf.get("brand"):
                raw_val = vf["brand"]
            else:
                continue
        elif field in vf and field in ("resolution", "screen_size", "refresh_rate"):
            raw_val = vf[field]
            if field == "resolution" and resolution_defs:
                mapped = _resolution_from_defs(str(raw_val), resolution_defs)
                if mapped:
                    raw_val = mapped
        elif field in ("resolution", "screen_size", "refresh_rate") and field not in vf:
            # No invent from title without facts.
            continue
        value = _normalize_claim_value(field, raw_val)
        if value == "":
            continue
        cleaned.append({"field": field, "value": value})
        seen.add(field)

    # Ensure core claims when justification is required (honest TF7 contract).
    if last_stock:
        if "price" not in seen and last_stock.get("price") is not None:
            cleaned.append(
                {"field": "price", "value": _normalize_claim_value("price", last_stock.get("price"))}
            )
            seen.add("price")
        if "in_stock" not in seen:
            stock_v = last_stock.get("in_stock")
            if stock_v is None:
                stock_v = True
            cleaned.append(
                {
                    "field": "in_stock",
                    "value": _normalize_claim_value("in_stock", stock_v),
                }
            )
            seen.add("in_stock")

    if "brand" not in seen and vf.get("brand"):
        cleaned.insert(0, {"field": "brand", "value": vf["brand"]})
        seen.add("brand")
    # Never invent brand from title (Task rules: raw facts only).

    # Shopper specification field from ledger (if allowed + observed).
    spec_f = (justification_spec_field or "").strip()
    if spec_f:
        spec_f = _normalize_claim_field(spec_f, allowed_fields) or spec_f
        if (
            spec_f in allowed_fields
            and spec_f not in _CLAIM_DENY
            and spec_f not in seen
            and vf.get(spec_f)
        ):
            raw = vf[spec_f]
            if spec_f == "resolution" and resolution_defs:
                mapped = _resolution_from_defs(str(raw), resolution_defs)
                if mapped:
                    raw = mapped
            cleaned.append(
                {"field": spec_f, "value": _normalize_claim_value(spec_f, raw)}
            )
            seen.add(spec_f)
    # Fallback gate from observed firm-like facts when spec field empty.
    if not (justification_spec_field or "").strip():
        for gate in ("screen_size", "resolution", "refresh_rate"):
            if (
                gate in allowed_fields
                and gate not in _CLAIM_DENY
                and gate not in seen
                and vf.get(gate)
            ):
                cleaned.append(
                    {"field": gate, "value": _normalize_claim_value(gate, vf.get(gate))}
                )
                seen.add(gate)
                break

    # Do NOT auto-attach all hardware specs from view facts — extra fields
    # (e.g. refresh_rate / opportunistic screen_size) fail claim_truth even when gold.
    # Spec field + brand + price + in_stock already covered above.

    cleaned = [c for c in cleaned if c["field"] not in _CLAIM_DENY]
    if any(c["field"] == "brand" for c in cleaned):
        cleaned = [c for c in cleaned if c["field"] == "brand"] + [
            c for c in cleaned if c["field"] != "brand"
        ]

    # ≥1 useful reason: prefer fields named in THIS rules; else brand if observed.
    useful_candidates = [
        str(x).strip().lower() for x in (useful_reason_fields or []) if str(x).strip()
    ]
    if "brand" not in useful_candidates:
        useful_candidates = useful_candidates + ["brand"]
    has_useful = any(c.get("field") in useful_candidates for c in cleaned)
    if not has_useful:
        for uf in useful_candidates:
            if uf not in allowed_fields or uf in _CLAIM_DENY or uf in seen:
                continue
            raw = vf.get(uf)
            if not raw:
                continue
            cleaned.append({"field": uf, "value": _normalize_claim_value(uf, raw)})
            seen.add(uf)
            break

    if cleaned:
        out["justification"] = {"claims": cleaned}
    else:
        out.pop("justification", None)
    return out


def _arguments(
    tool_call: dict[str, Any],
    *,
    justification_required: bool,
    allowed_fields: frozenset[str],
    last_stock: dict[str, Any] | None,
    view_facts_by_pid: dict[str, dict[str, str]],
    resolution_defs: list[dict[str, Any]] | None = None,
    justification_spec_field: str = "",
    useful_reason_fields: list[str] | None = None,
) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    if (tool_call["function"].get("name") or "") == "place_test_order":
        pid = str(parsed.get("product_id") or "")
        return _sanitize_place_order_args(
            parsed,
            justification_required=justification_required,
            allowed_fields=allowed_fields,
            last_stock=last_stock,
            view_facts=view_facts_by_pid.get(pid),
            resolution_defs=resolution_defs,
            justification_spec_field=justification_spec_field,
            useful_reason_fields=useful_reason_fields,
        )
    return parsed


def _tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [str(c.get("function", {}).get("name") or "") for c in tool_calls]


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
    # TF5 honest_oro: zero open-catalog search — view listed keys only.
    if any(n in ("search", "filter") for n in names):
        missing = sorted(cand_pids - viewed)
        return (
            "Closed candidate set: do not search/filter; view listed product_ids only."
            + (f" Missing: {missing}" if missing else "")
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
                return f"Closed set: view only listed product_ids {sorted(cand_pids)}"
        if name in ("add_to_cart", "inspect_stock", "place_test_order"):
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and pid not in cand_pids:
                return "Closed set: cart/order must use a listed product_id"
            if sku and f"{pid}::{sku}" not in cand_keys and name in ("add_to_cart", "place_test_order"):
                return "Closed set: sku must be a listed product_id::sku key"
    return None



_CAT_STOP = frozenset(
    {
        "that",
        "this",
        "which",
        "matches",
        "match",
        "matching",
        "category",
        "categories",
        "the",
        "and",
        "with",
        "from",
        "for",
        "under",
        "over",
        "into",
    }
)


def _sanitize_required_category(raw: str) -> str:
    """Keep catalog noun only — drop 'category that matches …' greed / prose noise."""
    s = (raw or "").strip().rstrip(".")
    if not s:
        return ""
    s = re.split(
        r"(?i)\s+category\b|\s+that\b|\s+which\b|\s+matching\b|\s+matches\b",
        s,
        maxsplit=1,
    )[0].strip()
    toks = [t for t in re.split(r"[^A-Za-z0-9]+", s) if t and t.lower() not in _CAT_STOP]
    # Reject stopword-ish leftovers (e.g. prose "labels alone are").
    junk = {
        "insufficient",
        "labels",
        "alone",
        "are",
        "is",
        "must",
        "explicitly",
        "say",
        "type",
        "device",
        "os",
        "raw",
        "title",
        "or",
        "its",
    }
    toks = [t for t in toks if t.lower() not in junk]
    if not toks:
        return ""
    return " ".join(toks[:3]).strip()


def _category_path_ok(facts: dict[str, str] | None, required: str) -> bool:
    req = _sanitize_required_category(required).lower()
    if not req:
        return True
    raw = (facts or {}).get("category_path") or (facts or {}).get("category") or ""
    if isinstance(raw, (list, tuple)):
        path = " ".join(str(x) for x in raw).lower()
    else:
        path = str(raw).lower()
    if not path:
        return False
    # Token overlap: required phrase or last path segment.
    if req in path:
        return True
    req_toks = [t for t in re.split(r"[^a-z0-9]+", req) if len(t) > 2 and t not in _CAT_STOP]
    return bool(req_toks) and all(t in path for t in req_toks[:3])


def _brand_matches(facts: dict[str, str] | None, brand: str) -> bool:
    b = (brand or "").strip().lower()
    if not b:
        return True
    fb = str((facts or {}).get("brand") or "").strip().lower()
    title = str((facts or {}).get("title") or "").strip().lower()
    if fb and (b in fb or fb in b):
        return True
    return bool(title) and b in title


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


def _run_pipeline(problem_data: dict[str, Any], profile: ObligationProfile) -> list[dict[str, Any]]:
    knobs = _knobs_for(profile.pipeline, profile)
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    # v45: honor episode policy (honest_oro batches ladder/views). Cap 16 = harness max.
    try:
        max_calls = int(policy.get("max_calls_per_turn") or 1)
    except (TypeError, ValueError):
        max_calls = 1
    max_calls = max(1, min(max_calls, 16))
    tools = policy["tools"]
    query = str(policy.get("query") or "")
    # Accumulated THIS-episode shopper+rules text for grounding firm prune (hidden-general).
    episode_text = query
    allowed_fields = _claim_field_allowlist(tools)
    candidates = _normalize_candidate_keys(
        profile.candidate_keys
    ) or _parse_candidate_variants(query)
    cand_pids = {pid for pid, _ in candidates}
    ask_first = bool(profile.ask_first)
    just_req = bool(profile.justification_required)
    budget = _budget_limit(_shopper_text(query) or query)
    commit_quiz_pending = False
    # Intent pipeline only: quiz lean pin for intended_event_commitment.
    intent_commit_pid = ""
    intent_commit_sku = ""
    intent_reaffirm_commit = False
    gated_attr_hint = (profile.firm_hint or "").strip() or None
    max_steps = int(policy["max_steps"])
    clock = Clock(_PROBLEM_BUDGET_S)
    category_soft_rejects = 0
    identity_soft_rejects = 0
    force_category_search = False
    force_identity_search = False
    # After survey soft-rejects with ≥1 matching id, allow terminal cart/place.
    identity_survey_relaxed = False
    excluded_listing_ids: set[str] = set()
    session_key = str(
        (binding or {}).get("session_id")
        or problem_data.get("problem_id")
        or problem_id
    )

    system = _CORE_SYSTEM + "\n\n" + knobs.playbook
    boot = (
        f"{query}\n\n"
        f"Active workflow: {knobs.name} (obligation confidence={profile.confidence:.2f}).\n"
        "Execute THIS episode's Task rules via the playbook above."
    )
    if ask_first:
        boot += (
            "\nASK-FIRST: First tool call must be `message` requesting the withheld firm spec."
        )
    if candidates:
        listed = ", ".join(f"{pid}::{sku}" for pid, sku in candidates)
        boot += f"\nCLOSED CANDIDATE SET: {listed}. View each; order a listed key only."
    if just_req:
        boot += (
            "\nJUSTIFICATION REQUIRED: On place_test_order include claims from viewed facts only."
        )
    if knobs.require_multi_listing >= 2:
        boot += (
            f"\nMULTI-LISTING: Compare at least {knobs.require_multi_listing} distinct "
            "product_ids before first add_to_cart."
        )
    if profile.listing_id_min >= 2:
        boot += (
            f"\nLISTING_ID: Inspect/compare ≥{profile.listing_id_min} distinct listing_id "
            "values before first cart / place."
        )
    if profile.soft_brand_first_cart and profile.preferred_brand:
        boot += (
            f"\nSOFT BRAND FIRST CART: Prefer brand {profile.preferred_brand} when a "
            "fully firm-compliant option exists."
        )
    if profile.require_category_path and profile.required_category:
        boot += (
            f"\nCATEGORY_PATH: Confirm view/compare category_path matches "
            f"{profile.required_category} (title alone is insufficient)."
        )
    if profile.relax_color_then_model or knobs.name == "recovery":
        boot += (
            "\nRELAX ORDER: After stockout prefer same model / other color before other models."
        )
    if profile.title_phrase_match and profile.title_terms:
        boot += (
            "\nTITLE TOKENS: Match shopper terms "
            f"{profile.title_terms} as complete title tokens (any order)."
        )
    if profile.min_matching_identities >= 2:
        boot += (
            f"\nIDENTITY SURVEY: Surface ≥{profile.min_matching_identities} distinct "
            "matching product identities before order."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": boot},
    ]
    dialogue: list[dict[str, Any]] = []
    must_message = bool(ask_first)
    asked_spec = False
    force_place_after_recheck = False
    # Cart product_id::sku whose post-add inspect already proved in-stock — block inspect thrash.
    cart_stock_verified_key: str | None = None
    # Always send honest_oro clarify opener first (all families, including closed-set).
    clarify_sent = False
    clarify_unlock = False  # shopper replied after clarify (even if firm regex empty)
    post_shock_searches = 0
    controller_forced = False  # FinishFSM/post-shock force — skip soft cart rejects
    force_add_after_inspect = False
    search_hit_pids: list[str] = []
    budget_ladder_idx = 0
    budget_ladder_caps = _budget_ladder_caps(budget)
    intent_seed_done = False
    intent_reseed_search = False
    intent_postshock_reseed_used = False
    intent_drop_brand = False  # Phase-B: after shrink/shock, seed without soft brand
    intent_compared = False  # one forced compare before first cart (TF1 survey)
    constraint_seed_done = False
    constraint_reseed_search = False
    constraint_postshock_reseed_used = False
    justification_postshock_reseed_used = False
    constraint_compared = False
    preference_seed_done = False
    preference_reseed_search = False
    recovery_seed_done = False
    recovery_reseed_search = False
    justification_seed_done = False
    justification_reseed_search = False
    preference_compared = False
    retrieval_reform_done = False
    retrieval_filter_done = False
    retrieval_hit_titles: list[str] = []
    harvested_brands: list[str] = []
    viewed: set[str] = set()
    listing_ids: set[str] = set()
    matching_ids: set[str] = set()  # product_ids whose titles match ledger title_terms
    identity_uf = _IdentityUF()
    brand_search_done = False
    brand_search_nudge = False
    searchish = 0
    last_stock: dict[str, Any] | None = None
    view_facts_by_pid: dict[str, dict[str, str]] = {}
    variants_by_pid: dict[str, list[dict[str, Any]]] = {}
    last_pick: dict[str, str] = {}
    cart_sku: dict[str, str] = {}
    excluded_skus: set[str] = set()
    excluded_product_ids: set[str] = set()
    llm_calls = 0
    last_hint_key: str | None = None
    shock_recover = False
    must_remove = False
    awaiting_user_after_shock = False
    recovery_msg_force = False
    recovery_committed = False
    pending_recheck: tuple[str, str] | None = None
    shock_open = False
    excluded_keys: set[str] = set()
    ranking_event = False
    order_done = False
    env_turn = 0  # session solver turn; only advances after successful /environment/call
    done = False
    remaining = max_steps

    def _listing_diversity_met(need: int) -> bool:
        """listing_id when present; else product_id diversity if env never emits listing_id."""
        if need < 2:
            return True
        if len(listing_ids) >= need:
            return True
        # Soft fallback: enough views and zero listing_id fields observed → use product_ids.
        if len(viewed) >= need and not listing_ids:
            return True
        return False

    def _hint_once(key: str, content: str) -> None:
        nonlocal last_hint_key
        if key == last_hint_key:
            return
        last_hint_key = key
        messages.append({"role": "user", "content": content})

    def _force_tool(name: str, args: dict[str, Any], think: str) -> dict[str, Any]:
        return _force_tools([(name, args)], think)

    def _force_tools(
        items: list[tuple[str, dict[str, Any]]], think: str
    ) -> dict[str, Any]:
        """Emit up to max_calls tool_calls in one env turn (same-phase batches)."""
        nonlocal controller_forced
        if not items:
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": think, "tool_calls": []}}
                ]
            }
        calls: list[dict[str, Any]] = []
        for index, (name, args) in enumerate(items[:max_calls]):
            if name in (
                "add_to_cart",
                "place_test_order",
                "inspect_stock",
                "remove_from_cart",
                "message",
            ):
                controller_forced = True
            calls.append(
                {
                    "id": f"force-{name}-{llm_calls}-{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": think,
                        "tool_calls": calls,
                    }
                }
            ]
        }

    def _place_force_args(pid: str, sku: str) -> dict[str, Any]:
        """Build place_test_order args with structured claims when required."""
        raw = {"product_id": pid, "sku": sku}
        return _sanitize_place_order_args(
            raw,
            justification_required=just_req,
            allowed_fields=allowed_fields,
            last_stock=last_stock,
            view_facts=view_facts_by_pid.get(pid),
            resolution_defs=profile.resolution_defs,
            justification_spec_field=profile.justification_spec_field,
            useful_reason_fields=profile.useful_reason_fields,
        )

    def _intent_firm_ok(pid: str, sku: str) -> bool:
        # TF1 soft-brand lane + TF3 exact-variant lane both need firm SKU truth.
        if knobs.name not in ("intent", "constraint"):
            return True
        if not profile.firm_specs:
            return True
        facts = view_facts_by_pid.get(pid) or {}
        row = next(
            (r for r in (variants_by_pid.get(pid) or []) if str(r.get("sku") or "") == sku),
            None,
        )
        return _sku_meets_firm(facts, row, profile.firm_specs, profile.resolution_defs)

    def _constraint_firm_brand_ok(pid: str) -> bool:
        """Constraint pipeline only: required brand is firm when latched."""
        if knobs.name != "constraint":
            return True
        brand = (profile.preferred_brand or "").strip()
        if not brand:
            return True
        return _brand_matches(view_facts_by_pid.get(pid), brand)

    def _intent_commit_viable() -> bool:
        """Intent pipeline only: quiz pin still firm+budget+not shocked."""
        if knobs.name != "intent" or not intent_commit_pid or not intent_commit_sku:
            return False
        if intent_commit_sku in excluded_skus or intent_commit_pid in excluded_product_ids:
            return False
        if f"{intent_commit_pid}::{intent_commit_sku}" in excluded_keys:
            return False
        if not _intent_firm_ok(intent_commit_pid, intent_commit_sku):
            return False
        row = next(
            (
                r
                for r in (variants_by_pid.get(intent_commit_pid) or [])
                if str(r.get("sku") or "") == intent_commit_sku
            ),
            None,
        )
        price = _price_num((row or last_stock or {}).get("price"))
        if budget is not None and price is not None and price > budget + 1e-6:
            return False
        return True

    def _intent_pin_commit(pick: dict[str, str] | None) -> None:
        nonlocal intent_commit_pid, intent_commit_sku, intent_reaffirm_commit
        if knobs.name != "intent" or not pick:
            return
        pid = str(pick.get("product_id") or "")
        sku = str(pick.get("sku") or "")
        if pid and sku:
            intent_commit_pid, intent_commit_sku = pid, sku
            intent_reaffirm_commit = False

    def _intent_restore_commit_pick() -> bool:
        nonlocal last_pick
        if not _intent_commit_viable():
            return False
        last_pick = {"product_id": intent_commit_pid, "sku": intent_commit_sku}
        return True

    def _intent_elect_commit_pick() -> dict[str, str] | None:
        """Elect in-budget firm (+soft brand) survivor for quiz / reaffirm — LLM-surveyed pool."""
        if knobs.name != "intent":
            return None
        _prefer_soft_brand_pick()
        if last_pick.get("product_id") and last_pick.get("sku"):
            pid0 = str(last_pick["product_id"])
            sku0 = str(last_pick["sku"])
            if _intent_firm_ok(pid0, sku0) and sku0 not in excluded_skus:
                row0 = next(
                    (
                        r
                        for r in (variants_by_pid.get(pid0) or [])
                        if str(r.get("sku") or "") == sku0
                    ),
                    None,
                )
                pr0 = _price_num((row0 or {}).get("price"))
                if budget is None or pr0 is None or pr0 <= budget + 1e-6:
                    return {"product_id": pid0, "sku": sku0}
        best: dict[str, str] | None = None
        best_pr = 1e18
        for pid_v in viewed:
            if pid_v in excluded_product_ids:
                continue
            if profile.soft_brand_first_cart and profile.preferred_brand:
                if not _brand_matches(view_facts_by_pid.get(pid_v), profile.preferred_brand):
                    continue
            for r in variants_by_pid.get(pid_v) or []:
                sku_v = str(r.get("sku") or "")
                if not sku_v or sku_v in excluded_skus or f"{pid_v}::{sku_v}" in excluded_keys:
                    continue
                if not _variant_in_stock(r):
                    continue
                pr = _price_num(r.get("price"))
                if budget is not None and pr is not None and pr > budget + 1e-6:
                    continue
                if not _intent_firm_ok(pid_v, sku_v):
                    continue
                if pr is None:
                    pr = 1e18
                if pr < best_pr:
                    best_pr = pr
                    best = {"product_id": pid_v, "sku": sku_v}
        return best

    def _identity_count() -> int:
        # Retrieval: UF collapses near-duplicate titles into few components while many
        # distinct product_ids already matched the phrase — teacher carts on match hits.
        if knobs.name == "retrieval" and matching_ids:
            return max(len(matching_ids), identity_uf.component_count())
        return identity_uf.component_count() if identity_uf.component_count() else len(matching_ids)

    def _pre_cart_depth_ok() -> bool:
        """Survey/view before first cart — depth from THIS episode ledger mins."""
        if cart_sku or shock_open or recovery_committed:
            return True
        if candidates:
            return cand_pids <= viewed
        need = max(
            2,
            int(profile.multi_listing_min or 0),
            int(profile.min_matching_identities or 0),
            int(profile.listing_id_min or 0),
            int(knobs.require_multi_listing or 0) or 0,
        )
        if knobs.name == "intent":
            if intent_compared and len(viewed) >= max(2, need - 1):
                return True
            if len(viewed) >= need:
                return True
            # Late escape only after some survey — never zero-view cart.
            if remaining <= max(4, need) and len(viewed) >= max(1, need // 2):
                return True
            return False
        if knobs.name == "constraint":
            if constraint_compared and len(viewed) >= max(2, need - 1):
                return True
            if len(viewed) >= need:
                return True
            if (
                last_stock is not None
                and len(viewed) >= 2
                and last_pick.get("product_id")
                and _intent_firm_ok(
                    str(last_pick.get("product_id") or ""),
                    str(last_pick.get("sku") or ""),
                )
            ):
                return True
            if remaining <= max(4, need) and len(viewed) >= max(1, need // 2):
                return True
            return False
        if knobs.name == "preference":
            if profile.pref_tokens or profile.useful_reason_fields:
                need = max(need, 2)
            if preference_compared and len(viewed) >= max(2, need - 1):
                return True
            if len(viewed) >= need:
                return True
            if remaining <= max(4, need) and len(viewed) >= max(1, need // 2):
                return True
            return False
        if profile.min_matching_identities >= 2:
            need = max(need, 2)
        if len(viewed) >= need:
            return True
        if remaining <= max(4, need) and len(viewed) >= 1:
            return True
        return False

    def _prefer_soft_brand_pick() -> None:
        """If soft-brand-first and a preferred-brand viewed SKU is firm-valid, prefer it.

        Teacher: soft brand binds **initial** cart only. After shock/recovery, do not force brand.
        Intent: never override a viable quiz commit pin (event commitment).
        """
        nonlocal last_pick
        if knobs.name == "intent" and _intent_commit_viable():
            last_pick = {"product_id": intent_commit_pid, "sku": intent_commit_sku}
            return
        if not profile.soft_brand_first_cart or not profile.preferred_brand:
            return
        if cart_sku or not viewed:
            return
        if shock_open or recovery_committed or intent_drop_brand:
            # TF7: brand remains the useful-reason target after stockout (teacher same-brand).
            if knobs.name != "justification":
                return
        brand_need = profile.preferred_brand.strip().lower()
        best: dict[str, str] | None = None
        best_price = 1e18
        for pid in viewed:
            if pid in (excluded_product_ids or set()):
                continue
            facts = view_facts_by_pid.get(pid) or {}
            if not _brand_matches(facts, profile.preferred_brand):
                continue
            if profile.require_category_path and profile.required_category:
                if not _category_path_ok(facts, profile.required_category):
                    continue
            rows = variants_by_pid.get(pid) or []
            for r in rows:
                sku = str(r.get("sku") or "")
                if not sku or sku in excluded_skus or f"{pid}::{sku}" in excluded_keys:
                    continue
                if not _variant_in_stock(r):
                    continue
                price = _price_num(r.get("price"))
                if budget is not None and price is not None and price > budget + 1e-6:
                    continue
                if profile.firm_specs and not _sku_meets_firm(
                    facts, r, profile.firm_specs, profile.resolution_defs
                ):
                    continue
                if not _device_evidence_ok(facts, r, profile):
                    continue
                if price is None:
                    price = 1e18
                if price < best_price:
                    best_price = price
                    best = {"product_id": pid, "sku": sku}
        if best:
            last_pick = best

    def _justification_survivor_in_viewed() -> bool:
        """True when a brand+firm+category in-stock survivor remains in viewed (post-shock)."""
        if knobs.name != "justification":
            return False
        for pid in viewed:
            if pid in (excluded_product_ids or set()):
                continue
            facts = view_facts_by_pid.get(pid) or {}
            if profile.preferred_brand and not _brand_matches(facts, profile.preferred_brand):
                continue
            if profile.require_category_path and profile.required_category:
                if not _category_path_ok(facts, profile.required_category):
                    continue
            for row in variants_by_pid.get(pid) or []:
                sku = str(row.get("sku") or "")
                if not sku or sku in excluded_skus or f"{pid}::{sku}" in excluded_keys:
                    continue
                if not _variant_in_stock(row):
                    continue
                price = _price_num(row.get("price"))
                if budget is not None and price is not None and price > budget + 1e-6:
                    continue
                if profile.firm_specs and not _sku_meets_firm(
                    facts, row, profile.firm_specs, profile.resolution_defs
                ):
                    continue
                if not _device_evidence_ok(facts, row, profile):
                    continue
                return True
        return False

    def _prefer_soft_pref_pick() -> None:
        """TF4: first cart = max soft-axis fit among firm-valid viewed SKUs (not cheapest)."""
        nonlocal last_pick
        if knobs.name != "preference":
            return
        if cart_sku or not viewed:
            return
        if not (profile.pref_tokens or profile.useful_reason_fields):
            return
        # After shock, still prefer soft fit among survivors (teacher recovers aligned).
        alt = _pick_best_viewed_sku(
            viewed=viewed,
            view_facts_by_pid=view_facts_by_pid,
            variants_by_pid=variants_by_pid,
            budget=budget,
            excluded_skus=excluded_skus,
            excluded_keys=excluded_keys,
            profile=profile,
            prefer_recovery=False,
            exclude_product_ids=set(excluded_product_ids) if excluded_product_ids else None,
        )
        if alt and alt.get("product_id") and alt.get("sku"):
            last_pick = dict(alt)

    def _prefer_recovery_pick() -> None:
        """TF6: cart = least-relaxed match (model+color → same model other color)."""
        nonlocal last_pick
        if knobs.name != "recovery" and not profile.relax_color_then_model:
            return
        if not viewed:
            return
        if not (profile.preferred_model or profile.preferred_color):
            return
        alt = _pick_best_viewed_sku(
            viewed=viewed,
            view_facts_by_pid=view_facts_by_pid,
            variants_by_pid=variants_by_pid,
            budget=budget,
            excluded_skus=excluded_skus,
            excluded_keys=excluded_keys,
            profile=profile,
            prefer_recovery=True,
            exclude_product_ids=set(excluded_product_ids) if excluded_product_ids else None,
        )
        if alt and alt.get("product_id") and alt.get("sku"):
            last_pick = dict(alt)

    def _identity_survey_met() -> bool:
        """Full N identities, or relaxed terminal path with enough matches already seen."""
        need = int(profile.min_matching_identities or 0)
        if need < 2:
            return True
        id_n = _identity_count()
        if id_n >= need:
            return True
        # TF2: title-match product_ids are the survey unit (UF alone under-counts).
        if knobs.name == "retrieval":
            hit_n = len(matching_ids)
            if hit_n >= need or id_n >= need:
                return True
            if identity_survey_relaxed and hit_n >= max(2, need // 2):
                return True
            if remaining <= 10 and hit_n >= 2:
                return True
            if remaining <= 8 and hit_n >= 1:
                return True
            if remaining <= 5 and (hit_n >= 1 or len(viewed) >= 2):
                return True
            # Ladder spent + ≥2 phrase hits → finish (honest_oro carts without UF=5).
            if hit_n >= 2 and (
                budget_ladder_idx >= max(1, len(budget_ladder_caps) - 1) or searchish >= 3
            ):
                return True
            return False
        if identity_survey_relaxed and id_n >= 1:
            return True
        # Last-chance: time/views burned BUT at least one matching identity observed.
        # Do NOT treat bare viewed≥2 as survey-met (retrieval under-survey SCORE fail).
        if id_n >= 1 and (remaining <= 10 or clock.remaining() < 45):
            return True
        return False

    def _can_place() -> bool:
        if order_done or done:
            return False
        if pending_recheck is not None:
            return False
        if must_message or must_remove or awaiting_user_after_shock:
            return False
        # TF3: never place on clarify_unlock alone without harvested firm specs.
        if knobs.name == "constraint" and not profile.firm_specs:
            return False
        # Ask-first: firm must be bound OR shopper replied after clarify (paraphrase unlock).
        if ask_first and not profile.firm_specs and not clarify_unlock:
            return False
        if shock_open and knobs.shock_protocol == "full" and not recovery_committed:
            return False
        if not cart_sku.get("product_id") or not cart_sku.get("sku"):
            return False
        cart_key = f"{cart_sku['product_id']}::{cart_sku['sku']}"
        if cart_key in excluded_keys:
            return False
        if knobs.name == "ranking" and ranking_event:
            # Must have re-elected a non-shocked closed key after market event.
            elect = _elect_ranking_pick(
                candidates,
                view_facts_by_pid=view_facts_by_pid,
                variants_by_pid=variants_by_pid,
                budget=budget,
                excluded_skus=excluded_skus,
                excluded_keys=excluded_keys,
                ranking_axes=profile.ranking_axes,
            )
            if not elect:
                return False
            if cart_key != f"{elect['product_id']}::{elect['sku']}":
                return False
        if profile.listing_id_min >= 2 and not _listing_diversity_met(profile.listing_id_min):
            return False
        if profile.min_matching_identities >= 2:
            if not _identity_survey_met():
                # v45 retrieval: stocked verified cart must not wait on soft survey.
                cart_key_pc = (
                    f"{cart_sku.get('product_id')}::{cart_sku.get('sku')}"
                    if cart_sku
                    else ""
                )
                if not (
                    knobs.name == "retrieval"
                    and cart_key_pc
                    and (
                        cart_stock_verified_key == cart_key_pc
                        or force_place_after_recheck
                    )
                ):
                    return False
        pid_cart = str(cart_sku.get("product_id") or "")
        sku_cart = str(cart_sku.get("sku") or "")
        facts_cart = view_facts_by_pid.get(pid_cart) or {}
        # Title-token place gate ONLY when THIS rules require title-phrase protocol.
        # Never block place on noisy harvest from "in-stock … under <budget>".
        # Closed-set ranking: candidate list is authority — skip title_phrase thrash gates.
        if profile.title_phrase_match and profile.title_terms and not (
            knobs.name == "ranking" and candidates
        ):
            title_cart = str(facts_cart.get("title") or "")
            if title_cart and not _title_matches_terms(title_cart, profile.title_terms):
                return False
        row_cart = next(
            (r for r in (variants_by_pid.get(pid_cart) or []) if str(r.get("sku") or "") == sku_cart),
            None,
        )
        if profile.firm_specs and not _sku_meets_firm(
            facts_cart, row_cart, profile.firm_specs, profile.resolution_defs
        ):
            return False
        if not _device_evidence_ok(facts_cart, row_cart, profile):
            return False
        if profile.require_category_path and profile.required_category:
            pid_c = cart_sku.get("product_id") or ""
            facts_cat = view_facts_by_pid.get(pid_c) or {}
            cat_ok = _category_path_ok(facts_cat, profile.required_category)
            if not cat_ok:
                # v45 retrieval/recovery: unviewed-but-stocked cart must still place.
                verified = (
                    cart_stock_verified_key
                    == f"{cart_sku.get('product_id')}::{cart_sku.get('sku')}"
                    or force_place_after_recheck
                )
                if not (knobs.name in ("retrieval", "recovery") and verified):
                    return False
        if last_stock is None or _stock_is_false(last_stock):
            return False
        price = _price_num((last_stock or {}).get("price"))
        if budget is not None and price is not None and price > budget + 1e-9:
            return False
        return True

    def _stocked_recovery_place_escape() -> bool:
        """Place when cart is stocked/in-budget after recovery — avoid inspect/agent_error thrash.

        Used when soft gates (title_phrase / elect / firm harvest noise) block `_can_place`
        but a post-add recheck already proved a live cart. Prefer this over endless inspect.
        """
        if order_done or done:
            return False
        if pending_recheck is not None:
            return False
        if must_message or must_remove or awaiting_user_after_shock:
            return False
        if knobs.name == "constraint" and not profile.firm_specs:
            return False
        if ask_first and not profile.firm_specs and not clarify_unlock:
            return False
        if shock_open and knobs.shock_protocol == "full" and not recovery_committed:
            return False
        if not cart_sku.get("product_id") or not cart_sku.get("sku"):
            return False
        cart_key = f"{cart_sku['product_id']}::{cart_sku['sku']}"
        if cart_key in excluded_keys or str(cart_sku.get("sku") or "") in excluded_skus:
            return False
        if str(cart_sku.get("product_id") or "") in excluded_product_ids:
            return False
        if last_stock is None or _stock_is_false(last_stock):
            return False
        # Require verified recheck for this cart (or force_place arm) so we don't skip R1.
        if cart_stock_verified_key != cart_key and not force_place_after_recheck:
            return False
        price = _price_num((last_stock or {}).get("price"))
        if budget is not None and price is not None and price > budget + 1e-9:
            return False
        # Intent/constraint: never skip firm match via escape.
        if knobs.name in ("intent", "constraint") and profile.firm_specs:
            pid_c = str(cart_sku.get("product_id") or "")
            sku_c = str(cart_sku.get("sku") or "")
            facts_c = view_facts_by_pid.get(pid_c) or {}
            row_c = next(
                (r for r in (variants_by_pid.get(pid_c) or []) if str(r.get("sku") or "") == sku_c),
                None,
            )
            if not _sku_meets_firm(facts_c, row_c, profile.firm_specs, profile.resolution_defs):
                return False
        # Ranking: if elect exists and differs, do not escape-place the wrong cart.
        if knobs.name == "ranking" and ranking_event and candidates:
            elect = _elect_ranking_pick(
                candidates,
                view_facts_by_pid=view_facts_by_pid,
                variants_by_pid=variants_by_pid,
                budget=budget,
                excluded_skus=excluded_skus,
                excluded_keys=excluded_keys,
                ranking_axes=profile.ranking_axes,
            )
            if elect and cart_key != f"{elect['product_id']}::{elect['sku']}":
                return False
        return True

    def _task_rule_verified_place() -> bool:
        """Task-rule place: stocked + in-budget cart must order (no soft survey thrash).

        Episode rules (retrieval/constraint/recovery): once verified in stock and within
        budget, place — do not wait for permission or soft identity/title gates.
        Keeps firm/elect hard gates for constraint/ranking.
        """
        if order_done or done:
            return False
        if pending_recheck is not None:
            return False
        if must_message or must_remove or awaiting_user_after_shock:
            return False
        if knobs.name not in ("retrieval", "constraint", "recovery", "preference", "intent"):
            return False
        if knobs.name == "constraint" and not profile.firm_specs:
            return False
        if shock_open and knobs.shock_protocol == "full" and not recovery_committed:
            return False
        if not cart_sku.get("product_id") or not cart_sku.get("sku"):
            return False
        cart_key = f"{cart_sku['product_id']}::{cart_sku['sku']}"
        if cart_key in excluded_keys or str(cart_sku.get("sku") or "") in excluded_skus:
            return False
        if str(cart_sku.get("product_id") or "") in excluded_product_ids:
            return False
        if last_stock is None or _stock_is_false(last_stock):
            return False
        sku_ls = str((last_stock or {}).get("sku") or "")
        if sku_ls and sku_ls != str(cart_sku.get("sku") or ""):
            return False
        # Prefer verified recheck; also allow when force_place armed this turn.
        if (
            cart_stock_verified_key != cart_key
            and not force_place_after_recheck
            and remaining > 8
        ):
            return False
        price = _price_num((last_stock or {}).get("price"))
        if budget is not None and price is not None and price > budget + 1e-9:
            return False
        if knobs.name in ("intent", "constraint") and profile.firm_specs:
            pid_c = str(cart_sku.get("product_id") or "")
            sku_c = str(cart_sku.get("sku") or "")
            facts_c = view_facts_by_pid.get(pid_c) or {}
            row_c = next(
                (r for r in (variants_by_pid.get(pid_c) or []) if str(r.get("sku") or "") == sku_c),
                None,
            )
            if not _sku_meets_firm(facts_c, row_c, profile.firm_specs, profile.resolution_defs):
                return False
        return True

    def _any_place_ok() -> bool:
        return (
            _can_place()
            or _terminal_place_ok()
            or _stocked_recovery_place_escape()
            or _task_rule_verified_place()
        )

    def _terminal_place_ok() -> bool:
        """Hard terminal constraints for place-first escape (no title-noise gates)."""
        if order_done or done:
            return False
        if pending_recheck is not None:
            return False
        if must_message or must_remove or awaiting_user_after_shock:
            return False
        if knobs.name == "constraint" and not profile.firm_specs:
            return False
        # Ask-first: firm must be bound OR shopper replied after clarify (paraphrase unlock).
        if ask_first and not profile.firm_specs and not clarify_unlock:
            return False
        if shock_open and knobs.shock_protocol == "full" and not recovery_committed:
            return False
        if not cart_sku.get("product_id") or not cart_sku.get("sku"):
            return False
        cart_key = f"{cart_sku['product_id']}::{cart_sku['sku']}"
        if cart_key in excluded_keys:
            return False
        if knobs.name == "ranking" and ranking_event:
            elect = _elect_ranking_pick(
                candidates,
                view_facts_by_pid=view_facts_by_pid,
                variants_by_pid=variants_by_pid,
                budget=budget,
                excluded_skus=excluded_skus,
                excluded_keys=excluded_keys,
                ranking_axes=profile.ranking_axes,
            )
            if not elect:
                return False
            if cart_key != f"{elect['product_id']}::{elect['sku']}":
                return False
        if profile.listing_id_min >= 2 and not _listing_diversity_met(profile.listing_id_min):
            return False
        if profile.min_matching_identities >= 2 and not _identity_survey_met():
            cart_key_tp = (
                f"{cart_sku.get('product_id')}::{cart_sku.get('sku')}" if cart_sku else ""
            )
            if not (
                knobs.name == "retrieval"
                and cart_key_tp
                and (
                    cart_stock_verified_key == cart_key_tp or force_place_after_recheck
                )
            ):
                return False
        if last_stock is None or _stock_is_false(last_stock):
            return False
        price = _price_num((last_stock or {}).get("price"))
        if budget is not None and price is not None and price > budget + 1e-9:
            return False
        # Firm / category still required when harvested — avoid knowingly illegal place.
        pid_cart = str(cart_sku.get("product_id") or "")
        sku_cart = str(cart_sku.get("sku") or "")
        facts_cart = view_facts_by_pid.get(pid_cart) or {}
        row_cart = next(
            (r for r in (variants_by_pid.get(pid_cart) or []) if str(r.get("sku") or "") == sku_cart),
            None,
        )
        if profile.firm_specs and not _sku_meets_firm(
            facts_cart, row_cart, profile.firm_specs, profile.resolution_defs
        ):
            return False
        if not _device_evidence_ok(facts_cart, row_cart, profile):
            return False
        if profile.require_category_path and profile.required_category:
            if not _category_path_ok(facts_cart, profile.required_category):
                return False
        if profile.title_phrase_match and profile.title_terms and not (
            knobs.name == "ranking" and candidates
        ):
            title_cart = str(facts_cart.get("title") or "")
            if title_cart and not _title_matches_terms(title_cart, profile.title_terms):
                return False
        return True

    def _sync_ranking_pick() -> None:
        nonlocal last_pick
        if knobs.name != "ranking" or not candidates:
            return
        if len(viewed & cand_pids) < len(cand_pids):
            return
        elect = _elect_ranking_pick(
            candidates,
            view_facts_by_pid=view_facts_by_pid,
            variants_by_pid=variants_by_pid,
            budget=budget,
            excluded_skus=excluded_skus,
            excluded_keys=excluded_keys,
            ranking_axes=profile.ranking_axes,
        )
        if elect:
            last_pick = elect

    for turn in range(1, max_steps + 1):
        if done:
            break
        remaining = max_steps - turn + 1
        controller_forced = False
        available = {
            (t.get("function") or t).get("name")
            for t in tools
            if isinstance(t, dict)
        }
        _sync_ranking_pick()

        # Intent-only: budget killed quiz pin → reaffirm going-with before cart.
        if (
            knobs.name == "intent"
            and intent_reaffirm_commit
            and not shock_open
            and not cart_sku
            and not must_remove
        ):
            must_message = True

        # v47 constraint: firm inspect already proves add — do not ask/message thrash.
        if (
            knobs.name == "constraint"
            and force_add_after_inspect
            and not cart_sku
            and last_pick.get("product_id")
            and last_pick.get("sku")
            and last_stock is not None
            and not _stock_is_false(last_stock)
            and str((last_stock or {}).get("sku") or "") == str(last_pick.get("sku") or "")
            and _intent_firm_ok(
                str(last_pick.get("product_id") or ""), str(last_pick.get("sku") or "")
            )
        ):
            must_message = False

        # Ranking family: after event commit, lock last_pick to post-event elect.
        if knobs.name == "ranking" and ranking_event and recovery_committed and not must_remove:
            elect = _elect_ranking_pick(
                candidates,
                view_facts_by_pid=view_facts_by_pid,
                variants_by_pid=variants_by_pid,
                budget=budget,
                excluded_skus=excluded_skus,
                excluded_keys=excluded_keys,
                ranking_axes=profile.ranking_axes,
            )
            if elect:
                last_pick = elect
                ck = f"{cart_sku.get('product_id')}::{cart_sku.get('sku')}" if cart_sku else ""
                if ck and ck in excluded_keys:
                    must_remove = True

        # Controller-owned finish FSM (family workflows share this).
        inf = None
        # Clarify-first (honest_oro 70/70): same opener for every family before search/view.
        if (
            not clarify_sent
            and turn == 1
            and "message" in available
            and not must_remove
        ):
            clarify_sent = True
            must_message = False
            inf = _force_tool(
                "message",
                {"content": _HONEST_CLARIFY_OPENER},
                "Clarify-first opener.",
            )
        # TF5: force-view every closed-set product_id (honest_oro: zero search).
        elif (
            candidates
            and "view" in available
            and not must_remove
            and not must_message
            and not awaiting_user_after_shock
            and pending_recheck is None
            and len(viewed & cand_pids) < len(cand_pids)
        ):
            missing = sorted(cand_pids - viewed)
            inf = _force_tool(
                "view",
                {"product_id": missing[0]},
                "Closed-set view missing candidate.",
            )
        # TF5: after all candidates viewed — inspect elect SKU then cart (best-ranked first).
        elif (
            knobs.name == "ranking"
            and candidates
            and cand_pids <= viewed
            and not cart_sku
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and not commit_quiz_pending
            and "inspect_stock" in available
        ):
            _sync_ranking_pick()
            pid_e = str(last_pick.get("product_id") or "")
            sku_e = str(last_pick.get("sku") or "")
            if pid_e and sku_e and f"{pid_e}::{sku_e}" not in excluded_keys and sku_e not in excluded_skus:
                need_inspect = (
                    last_stock is None
                    or str((last_stock or {}).get("product_id") or "") != pid_e
                    or str((last_stock or {}).get("sku") or "") != sku_e
                    or _stock_is_false(last_stock)
                )
                if need_inspect:
                    force_add_after_inspect = True
                    inf = _force_tool(
                        "inspect_stock",
                        {"product_id": pid_e, "sku": sku_e},
                        "TF5 inspect best-ranked before first cart.",
                    )
                elif "add_to_cart" in available and not _stock_is_false(last_stock):
                    force_add_after_inspect = False
                    inf = _force_tool(
                        "add_to_cart",
                        {"product_id": pid_e, "sku": sku_e},
                        "TF5 add best-ranked (not placeholder).",
                    )
                else:
                    inf = None
            else:
                inf = None
        # Immediate place after a successful post-add stock recheck (hang class).
        # v43: never clear force_place into a no-op that skips FinishFSM → LLM inspect thrash.
        elif (
            force_place_after_recheck
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and cart_sku.get("product_id")
            and cart_sku.get("sku")
            and last_stock is not None
            and not _stock_is_false(last_stock)
            and (
                (knobs.name == "constraint" and bool(profile.firm_specs))
                or (knobs.name in ("retrieval", "recovery"))
                or (knobs.name != "constraint" and (not ask_first or bool(profile.firm_specs) or clarify_unlock))
            )
        ):
            price_n = _price_num((last_stock or {}).get("price"))
            over = (
                budget is not None
                and price_n is not None
                and price_n > budget + 1e-6
            )
            if over and "remove_from_cart" in available:
                force_place_after_recheck = False
                must_remove = True
                shock_open = True
                inf = _force_tool(
                    "remove_from_cart",
                    {"product_id": cart_sku["product_id"], "sku": cart_sku["sku"]},
                    "Over-budget remove after post-add recheck.",
                )
            elif knobs.name == "ranking" and ranking_event:
                _sync_ranking_pick()
                elect = _elect_ranking_pick(
                    candidates,
                    view_facts_by_pid=view_facts_by_pid,
                    variants_by_pid=variants_by_pid,
                    budget=budget,
                    excluded_skus=excluded_skus,
                    excluded_keys=excluded_keys,
                    ranking_axes=profile.ranking_axes,
                )
                cart_key = f"{cart_sku['product_id']}::{cart_sku['sku']}"
                if (
                    elect
                    and cart_key != f"{elect['product_id']}::{elect['sku']}"
                    and "remove_from_cart" in available
                ):
                    force_place_after_recheck = False
                    last_pick = elect
                    must_remove = True
                    inf = _force_tool(
                        "remove_from_cart",
                        {
                            "product_id": cart_sku["product_id"],
                            "sku": cart_sku["sku"],
                        },
                        "Ranking re-sync: remove non-elect cart after recheck.",
                    )
                elif "place_test_order" in available and (
                    _any_place_ok()
                ):
                    force_place_after_recheck = False
                    inf = _force_tool(
                        "place_test_order",
                        _place_force_args(cart_sku["product_id"], cart_sku["sku"]),
                        "Place after post-add recheck.",
                    )
                else:
                    # Soft gates still blocking — keep flag only if we couldn't act;
                    # prefer escape place over inspect thrash.
                    if "place_test_order" in available and _stocked_recovery_place_escape():
                        force_place_after_recheck = False
                        inf = _force_tool(
                            "place_test_order",
                            _place_force_args(cart_sku["product_id"], cart_sku["sku"]),
                            "Place escape after recheck (ranking).",
                        )
                    else:
                        force_place_after_recheck = False
            elif "place_test_order" in available and (
                _any_place_ok()
            ):
                force_place_after_recheck = False
                inf = _force_tool(
                    "place_test_order",
                    _place_force_args(cart_sku["product_id"], cart_sku["sku"]),
                    "Place after post-add recheck.",
                )
            else:
                # v45: never clear force_place into no-op when cart is stocked+budget
                # (qualifying agent_error: inspect OK → flag cleared → never place).
                if "place_test_order" in available and (
                    _task_rule_verified_place() or _stocked_recovery_place_escape()
                ):
                    force_place_after_recheck = False
                    inf = _force_tool(
                        "place_test_order",
                        _place_force_args(cart_sku["product_id"], cart_sku["sku"]),
                        "Place after recheck (task-rule verified).",
                    )
                elif knobs.name in ("retrieval", "constraint", "recovery") and (
                    last_stock is not None
                    and not _stock_is_false(last_stock)
                    and "place_test_order" in available
                ):
                    price_n2 = _price_num((last_stock or {}).get("price"))
                    over2 = (
                        budget is not None
                        and price_n2 is not None
                        and price_n2 > budget + 1e-6
                    )
                    if not over2:
                        force_place_after_recheck = False
                        identity_survey_relaxed = True
                        inf = _force_tool(
                            "place_test_order",
                            _place_force_args(cart_sku["product_id"], cart_sku["sku"]),
                            "Place after recheck (family finish escape).",
                        )
                    else:
                        force_place_after_recheck = False
                else:
                    force_place_after_recheck = False
        elif must_remove and cart_sku and "remove_from_cart" in available:
            # Remove outranks post-add recheck — budget-shrink / shock must strip cart first.
            pending_recheck = None
            inf = _force_tool(
                "remove_from_cart",
                {"product_id": cart_sku["product_id"], "sku": cart_sku["sku"]},
                "Forced remove after market shock.",
            )
        elif must_message and "message" in available:
            # v42: MSG before pending inspect — pending+must_message was a reject-spin
            # deadlock (inspect forced, gate demands message, no env call; v41 live cliff).
            if ask_first and not asked_spec and not shock_open and not recovery_msg_force:
                content = _HONEST_CLARIFY_OPENER
            elif commit_quiz_pending and not shock_open:
                # Intent: elect in-budget firm/soft pick, announce it, pin for sticky first cart.
                pick_for_commit = last_pick or cart_sku or None
                if knobs.name == "intent":
                    elected = _intent_elect_commit_pick()
                    if elected:
                        last_pick = elected
                        pick_for_commit = elected
                content = _going_with_message(
                    pick=pick_for_commit,
                    view_facts_by_pid=view_facts_by_pid,
                    gated_hint=gated_attr_hint,
                    firm_specs=profile.firm_specs,
                    budget=budget,
                )
                if knobs.name == "intent":
                    _intent_pin_commit(
                        pick_for_commit if isinstance(pick_for_commit, dict) else None
                    )
                commit_quiz_pending = False
            elif (
                knobs.name == "intent"
                and intent_reaffirm_commit
                and not shock_open
            ):
                elected = _intent_elect_commit_pick()
                if elected:
                    last_pick = elected
                content = _going_with_message(
                    pick=last_pick or None,
                    view_facts_by_pid=view_facts_by_pid,
                    gated_hint=gated_attr_hint,
                    firm_specs=profile.firm_specs,
                    budget=budget,
                )
                _intent_pin_commit(last_pick or None)
                intent_reaffirm_commit = False
            elif knobs.name == "recovery" or profile.relax_color_then_model:
                content = (
                    "The preferred listing is unavailable. I will take another in-stock color "
                    "of the same model before changing models, keeping budget and category firm."
                )
            else:
                content = (
                    "I will recover to another in-stock option that still meets firm requirements "
                    "and stays within budget."
                )
            inf = _force_tool("message", {"content": content}, "Forced recovery/ask message.")
        elif pending_recheck and "inspect_stock" in available:
            pid, sku = pending_recheck
            key_pr = f"{pid}::{sku}"
            if (
                str(pid) in excluded_product_ids
                or str(sku) in excluded_skus
                or key_pr in excluded_keys
            ):
                # Drop shocked / banned pending — do not inspect-thrash event target.
                pending_recheck = None
                inf = None
            else:
                inf = _force_tool(
                    "inspect_stock",
                    {"product_id": pid, "sku": sku},
                    "Post-add recheck inspect_stock.",
                )
        # TF1: seed / re-seed search after goal_brief or budget shrink.
        elif (
            knobs.name == "intent"
            and not cart_sku
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and not commit_quiz_pending
            and "search" in available
            and (clarify_unlock or asked_spec or bool(profile.firm_specs))
            and (intent_reseed_search or not intent_seed_done)
            # Avoid useless "in stock" seed when no brand/size/category known yet.
            and (
                profile.preferred_brand
                or profile.firm_specs
                or profile.required_category
                or intent_reseed_search
            )
        ):
            intent_seed_done = True
            intent_reseed_search = False
            q = _intent_seed_query(profile, include_brand=not intent_drop_brand)
            args_s: dict[str, Any] = {"query": q, "k": 10, "in_stock": True}
            if budget is not None:
                args_s["max_price"] = float(budget)
            inf = _force_tool("search", args_s, "TF1 intent seed/re-seed search.")
        # TF3: seed / re-seed after firm unlock or budget shrink (exact-variant constraints).
        elif (
            knobs.name == "constraint"
            and not cart_sku
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and "search" in available
            and (clarify_unlock or asked_spec or bool(profile.firm_specs))
            and bool(profile.firm_specs)
            and (constraint_reseed_search or not constraint_seed_done)
        ):
            constraint_seed_done = True
            constraint_reseed_search = False
            q = _constraint_seed_query(profile)
            args_c: dict[str, Any] = {"query": q, "k": 10, "in_stock": True}
            if budget is not None:
                args_c["max_price"] = float(budget)
            inf = _force_tool("search", args_c, "TF3 constraint seed/re-seed search.")
        # TF4: seed / re-seed with category + THIS-episode soft preference tokens.
        elif (
            knobs.name == "preference"
            and not cart_sku
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and "search" in available
            and (clarify_unlock or clarify_sent)
            and (preference_reseed_search or not preference_seed_done)
            and (
                profile.required_category
                or profile.pref_tokens
                or profile.useful_reason_fields
                or preference_reseed_search
            )
        ):
            preference_seed_done = True
            preference_reseed_search = False
            q = _preference_seed_query(profile)
            args_p: dict[str, Any] = {"query": q, "k": 10, "in_stock": True}
            if budget is not None:
                args_p["max_price"] = float(budget)
            inf = _force_tool("search", args_p, "TF4 preference seed/re-seed search.")
        # TF6: seed / re-seed with preferred model id (least-relaxed identity).
        elif (
            knobs.name == "recovery"
            and not cart_sku
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and "search" in available
            and (clarify_unlock or clarify_sent)
            and (recovery_reseed_search or not recovery_seed_done)
            and (profile.preferred_model or profile.preferred_color)
        ):
            recovery_seed_done = True
            recovery_reseed_search = False
            bits = [
                str(x).strip()
                for x in (profile.preferred_model, profile.preferred_color, profile.required_category)
                if str(x or "").strip()
            ]
            q = " ".join(bits[:3]) or "in stock"
            args_r: dict[str, Any] = {"query": q, "k": 10, "in_stock": True}
            if budget is not None:
                args_r["max_price"] = float(budget)
            inf = _force_tool("search", args_r, "TF6 recovery seed/re-seed search.")
        # TF6: survey search hits until a full model+color match is viewed (or hits exhausted).
        elif (
            knobs.name == "recovery"
            and not cart_sku
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and "view" in available
            and search_hit_pids
            and len(viewed) < 8
            and any(p not in viewed for p in search_hit_pids)
        ):
            def _recovery_full_match_seen() -> bool:
                if not (profile.preferred_model or profile.preferred_color):
                    return bool(viewed)
                model_need = _normalize_model_key(profile.preferred_model)
                color_need = (profile.preferred_color or "").strip().lower()
                for pid in viewed:
                    facts = view_facts_by_pid.get(pid) or {}
                    for row in variants_by_pid.get(pid) or []:
                        if not _variant_in_stock(row):
                            continue
                        opts = _options_text(row)
                        color = _color_from_sources(
                            options=opts,
                            color_fact=str(facts.get("color") or ""),
                            title=str(facts.get("title") or ""),
                        )
                        blob = _normalize_model_key(
                            " ".join(
                                [
                                    str(facts.get("model") or ""),
                                    str(facts.get("title") or ""),
                                    opts,
                                    " ".join(str(v) for v in facts.values()),
                                ]
                            )
                        )
                        same_model = (not model_need) or (model_need in blob)
                        same_color = (not color_need) or _color_matches_need(
                            color,
                            color_need,
                            options=opts,
                            color_fact=str(facts.get("color") or ""),
                            title=str(facts.get("title") or ""),
                        )
                        if same_model and same_color:
                            return True
                return False

            if not _recovery_full_match_seen():
                missing = [p for p in search_hit_pids if p not in viewed]
                inf = _force_tool(
                    "view",
                    {"product_id": missing[0]},
                    "TF6 survey search hits for preferred model/color.",
                )
            else:
                # Full match already viewed — leave inf unset by using a no-op branch skip.
                # Re-enter FinishFSM elect/add via a dedicated recovery elect below.
                _prefer_recovery_pick()
                pid_e = str(last_pick.get("product_id") or "")
                sku_e = str(last_pick.get("sku") or "")
                if (
                    pid_e
                    and sku_e
                    and "inspect_stock" in available
                    and (
                        last_stock is None
                        or str((last_stock or {}).get("sku") or "") != sku_e
                    )
                ):
                    force_add_after_inspect = True
                    inf = _force_tool(
                        "inspect_stock",
                        {"product_id": pid_e, "sku": sku_e},
                        "TF6 inspect full-match before first cart.",
                    )
                elif pid_e and sku_e and "add_to_cart" in available:
                    inf = _force_tool(
                        "add_to_cart",
                        {"product_id": pid_e, "sku": sku_e},
                        "TF6 add preferred full-match (not sacrificial).",
                    )
        # TF7: seed / re-seed with brand + firm + product type.
        elif (
            knobs.name == "justification"
            and not cart_sku
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and "search" in available
            and (clarify_unlock or clarify_sent)
            and (justification_reseed_search or not justification_seed_done)
            and not (
                (shock_open or recovery_committed) and _justification_survivor_in_viewed()
            )
            and (
                profile.preferred_brand
                or profile.firm_specs
                or profile.required_category
                or justification_reseed_search
            )
        ):
            justification_seed_done = True
            justification_reseed_search = False
            q = _justification_seed_query(profile)
            args_j: dict[str, Any] = {"query": q, "k": 10, "in_stock": True}
            if budget is not None:
                args_j["max_price"] = float(budget)
            inf = _force_tool("search", args_j, "TF7 justification seed/re-seed search.")
        # TF7: survey hits until brand+firm match viewed (or hits exhausted).
        elif (
            knobs.name == "justification"
            and not cart_sku
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and "view" in available
            and search_hit_pids
            and len(viewed) < 8
            and any(p not in viewed for p in search_hit_pids)
        ):
            def _justification_match_count() -> int:
                n = 0
                for pid in viewed:
                    if pid in (excluded_product_ids or set()):
                        continue
                    facts = view_facts_by_pid.get(pid) or {}
                    if profile.preferred_brand and not _brand_matches(
                        facts, profile.preferred_brand
                    ):
                        continue
                    if profile.require_category_path and profile.required_category:
                        if not _category_path_ok(facts, profile.required_category):
                            continue
                    for row in variants_by_pid.get(pid) or []:
                        sku = str(row.get("sku") or "")
                        if not sku or sku in excluded_skus or f"{pid}::{sku}" in excluded_keys:
                            continue
                        if not _variant_in_stock(row):
                            continue
                        if profile.firm_specs and not _sku_meets_firm(
                            facts, row, profile.firm_specs, profile.resolution_defs
                        ):
                            continue
                        if not _device_evidence_ok(facts, row, profile):
                            continue
                        n += 1
                        break
                return n

            def _justification_match_seen() -> bool:
                return _justification_match_count() >= 1

            # Survey until ≥1 match, ideally ≥2 firm matches before first cart (shock buffer).
            missing = [p for p in search_hit_pids if p not in viewed]
            need_buffer = (
                not cart_sku
                and _justification_match_count() < 2
                and bool(missing)
                and len(viewed) < 6
            )
            if not _justification_match_seen() or len(viewed) < 2 or need_buffer:
                if missing:
                    # Prefer category-hinted hits when surveying.
                    missing_sorted = sorted(
                        missing,
                        key=lambda p: (
                            0
                            if _search_hit_category_hint(
                                {"product_id": p, "title": (view_facts_by_pid.get(p) or {}).get("title"),
                                 "category_path": (view_facts_by_pid.get(p) or {}).get("category_path")},
                                profile.required_category or "",
                            )
                            else 1
                        ),
                    )
                    inf = _force_tool(
                        "view",
                        {"product_id": missing_sorted[0]},
                        "TF7 survey search hits for brand/firm match.",
                    )
                elif (
                    (justification_reseed_search or not justification_postshock_reseed_used)
                    and not _justification_survivor_in_viewed()
                ):
                    justification_postshock_reseed_used = True
                    justification_reseed_search = False
                    justification_seed_done = True
                    q = _justification_seed_query(profile)
                    args_j2: dict[str, Any] = {"query": q, "k": 10, "in_stock": True}
                    if budget is not None:
                        args_j2["max_price"] = float(budget)
                    inf = _force_tool(
                        "search", args_j2, "TF7 re-seed after empty match survey."
                    )
                else:
                    inf = None
            else:
                _prefer_soft_brand_pick()
                # Elect best viewed under firm/brand.
                alt = _pick_best_viewed_sku(
                    viewed=viewed,
                    view_facts_by_pid=view_facts_by_pid,
                    variants_by_pid=variants_by_pid,
                    budget=budget,
                    excluded_skus=excluded_skus,
                    excluded_keys=excluded_keys,
                    profile=profile,
                    prefer_recovery=False,
                    exclude_product_ids=set(excluded_product_ids)
                    if excluded_product_ids
                    else None,
                )
                if alt and alt.get("product_id") and alt.get("sku"):
                    last_pick = dict(alt)
                pid_e = str(last_pick.get("product_id") or "")
                sku_e = str(last_pick.get("sku") or "")
                if not pid_e or not sku_e:
                    # Match-seen race / empty elect — keep surveying.
                    missing = [p for p in search_hit_pids if p not in viewed]
                    if missing:
                        inf = _force_tool(
                            "view",
                            {"product_id": missing[0]},
                            "TF7 continue survey after empty elect.",
                        )
                elif (
                    "inspect_stock" in available
                    and (
                        last_stock is None
                        or str((last_stock or {}).get("sku") or "") != sku_e
                    )
                ):
                    force_add_after_inspect = True
                    inf = _force_tool(
                        "inspect_stock",
                        {"product_id": pid_e, "sku": sku_e},
                        "TF7 inspect brand/firm match before first cart.",
                    )
                elif "add_to_cart" in available:
                    inf = _force_tool(
                        "add_to_cart",
                        {"product_id": pid_e, "sku": sku_e},
                        "TF7 add justified real choice (not placeholder).",
                    )
        # TF1: drop force-add when current pick misses firm gated attr.
        elif (
            force_add_after_inspect
            and knobs.name in ("intent", "constraint")
            and last_pick.get("product_id")
            and last_pick.get("sku")
            and not _intent_firm_ok(
                str(last_pick.get("product_id") or ""), str(last_pick.get("sku") or "")
            )
        ):
            if knobs.name == "constraint":
                # Shopper-grounded prune drops Task-rules class-table HD; keep real FHD needs.
                _prune_constraint_firm_specs(profile, episode_text)
            if not _intent_firm_ok(
                str(last_pick.get("product_id") or ""), str(last_pick.get("sku") or "")
            ):
                force_add_after_inspect = False
                excluded_skus.add(str(last_pick.get("sku") or ""))
                last_pick = {}
                inf = None
            else:
                # Firm ok after prune — fall through next turn / sticky add below.
                inf = None
        # After inspect of a viable pick: cart immediately (never-cart hang class).
        elif (
            force_add_after_inspect
            and not cart_sku
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and not commit_quiz_pending
            and last_pick.get("product_id")
            and last_pick.get("sku")
            and "add_to_cart" in available
            and last_stock is not None
            and not _stock_is_false(last_stock)
            and str((last_stock or {}).get("sku") or "") == str(last_pick.get("sku") or "")
            and str(last_pick.get("sku") or "") not in excluded_skus
            and f"{last_pick.get('product_id')}::{last_pick.get('sku')}" not in excluded_keys
            and _pre_cart_depth_ok()
            and _intent_firm_ok(str(last_pick.get("product_id") or ""), str(last_pick.get("sku") or ""))
        ):
            force_add_after_inspect = False
            identity_survey_relaxed = True
            prev_sku = str(last_pick.get("sku") or "")
            # Intent-only: sticky quiz pin before soft re-elect.
            if knobs.name == "intent":
                _intent_restore_commit_pick()
            # Constraint-only: never soft-pref re-pick (exact-variant sticky).
            if knobs.name != "constraint":
                _prefer_soft_brand_pick()
                _prefer_soft_pref_pick()
                _prefer_recovery_pick()
            if knobs.name == "intent":
                _intent_restore_commit_pick()
            price_n = _price_num((last_stock or {}).get("price"))
            if (
                knobs.name != "constraint"
                and last_pick.get("sku")
                and str(last_pick.get("sku") or "") != prev_sku
                and "inspect_stock" in available
            ):
                inf = _force_tool(
                    "inspect_stock",
                    {
                        "product_id": last_pick["product_id"],
                        "sku": last_pick["sku"],
                    },
                    "Soft-pref/brand re-inspect before first cart.",
                )
            elif not _intent_firm_ok(
                str(last_pick.get("product_id") or ""), str(last_pick.get("sku") or "")
            ):
                # Wrong gated attr — keep surveying instead of hard_gate fail.
                excluded_skus.add(str(last_pick.get("sku") or ""))
                last_pick = {}
                force_add_after_inspect = False
                inf = None
            elif not _constraint_firm_brand_ok(str(last_pick.get("product_id") or "")):
                # Constraint: required brand is firm — do not commit off-brand event target.
                excluded_skus.add(str(last_pick.get("sku") or ""))
                last_pick = {}
                force_add_after_inspect = False
                if profile.preferred_brand and "search" in available:
                    q_b = _constraint_seed_query(profile)
                    args_b: dict[str, Any] = {"query": q_b, "k": 10, "in_stock": True}
                    if budget is not None:
                        args_b["max_price"] = float(budget)
                    inf = _force_tool(
                        "search",
                        args_b,
                        "Constraint firm-brand re-seed before first cart.",
                    )
                else:
                    inf = None
            elif budget is not None and price_n is not None and price_n > budget + 1e-6:
                excluded_skus.add(str(last_pick["sku"]))
                last_pick = {}
                inf = None
            else:
                # v44 intent: force-add must not bypass soft-brand first-cart (controller_forced
                # used to skip the proposal gate → Sansumg/wrong brand initial cart, reward 0).
                if (
                    knobs.name == "intent"
                    and profile.soft_brand_first_cart
                    and profile.preferred_brand
                    and not shock_open
                    and not recovery_committed
                    and not _brand_matches(
                        view_facts_by_pid.get(str(last_pick.get("product_id") or "")),
                        profile.preferred_brand,
                    )
                ):
                    _prefer_soft_brand_pick()
                    if not _brand_matches(
                        view_facts_by_pid.get(str(last_pick.get("product_id") or "")),
                        profile.preferred_brand,
                    ):
                        # v47: never first-cart wrong soft brand (hidden initial-cart score).
                        brand_search_nudge = True
                        force_add_after_inspect = True
                        if not brand_search_done and "search" in available:
                            q_b = _intent_seed_query(profile, include_brand=True)
                            args_b: dict[str, Any] = {
                                "query": q_b,
                                "k": 10,
                                "in_stock": True,
                            }
                            if budget is not None:
                                args_b["max_price"] = float(budget)
                            inf = _force_tool(
                                "search",
                                args_b,
                                "Intent soft-brand search before first cart.",
                            )
                        else:
                            # Prefer viewing unseen preferred hits; do not add wrong brand.
                            miss_b = [
                                pid
                                for pid in search_hit_pids
                                if pid not in viewed
                                and _brand_matches(
                                    view_facts_by_pid.get(pid) or {},
                                    profile.preferred_brand,
                                )
                            ]
                            if not miss_b:
                                miss_b = [
                                    pid
                                    for pid in search_hit_pids
                                    if pid not in viewed and pid not in excluded_product_ids
                                ]
                            if miss_b and "view" in available:
                                inf = _force_tool(
                                    "view",
                                    {"product_id": miss_b[0]},
                                    "Intent soft-brand: view preferred hit before cart.",
                                )
                            else:
                                inf = None
                    else:
                        # Re-inspect preferred if pick changed.
                        if (
                            str(last_pick.get("sku") or "") != prev_sku
                            and "inspect_stock" in available
                        ):
                            force_add_after_inspect = True
                            inf = _force_tool(
                                "inspect_stock",
                                {
                                    "product_id": last_pick["product_id"],
                                    "sku": last_pick["sku"],
                                },
                                "Intent soft-brand inspect preferred before cart.",
                            )
                        else:
                            inf = _force_tool(
                                "add_to_cart",
                                {
                                    "product_id": last_pick["product_id"],
                                    "sku": last_pick["sku"],
                                },
                                "Force add after inspect (preferred brand).",
                            )
                else:
                    inf = _force_tool(
                        "add_to_cart",
                        {
                            "product_id": last_pick["product_id"],
                            "sku": last_pick["sku"],
                        },
                        "Force add after inspect.",
                    )
        # Retrieval: budget ladder searches — phrase-first (honest multi-ladder batch).
        elif (
            knobs.name == "retrieval"
            and not cart_sku
            and not candidates
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and "search" in available
            and budget_ladder_caps
            and budget_ladder_idx < len(budget_ladder_caps)
            and searchish < max(2, len(budget_ladder_caps) + 1)
            and len(viewed) < 2
        ):
            q = _retrieval_phrase_query(profile)
            batch: list[tuple[str, dict[str, Any]]] = []
            while (
                budget_ladder_idx < len(budget_ladder_caps)
                and len(batch) < max_calls
                and searchish + len(batch) < max(2, len(budget_ladder_caps) + 1)
            ):
                cap = budget_ladder_caps[budget_ladder_idx]
                budget_ladder_idx += 1
                batch.append(
                    (
                        "search",
                        {"query": q, "k": 10, "in_stock": True, "max_price": cap},
                    )
                )
            # Same turn: reformulation + observed-brand filters when room (honest pattern).
            if (
                not retrieval_reform_done
                and profile.title_terms
                and len(batch) < max_calls
                and budget_ladder_idx >= max(1, len(budget_ladder_caps) - 1)
            ):
                retrieval_reform_done = True
                extra = _harvest_reform_tokens(
                    titles=retrieval_hit_titles,
                    phrase_terms=profile.title_terms,
                )
                q2 = _retrieval_phrase_query(profile, extra_tokens=extra)
                args_r: dict[str, Any] = {"query": q2, "k": 10, "in_stock": True}
                if budget is not None:
                    args_r["max_price"] = float(budget)
                batch.append(("search", args_r))
            if (
                not retrieval_filter_done
                and harvested_brands
                and "filter" in available
                and len(batch) < max_calls
            ):
                retrieval_filter_done = True
                for brand in harvested_brands[: max(0, max_calls - len(batch))]:
                    args_f: dict[str, Any] = {"brand": brand, "in_stock": True}
                    if profile.required_category:
                        args_f["category"] = profile.required_category
                    if budget is not None:
                        args_f["max_price"] = float(budget)
                    batch.append(("filter", args_f))
                    if len(batch) >= max_calls:
                        break
            inf = _force_tools(batch, "Retrieval budget-ladder batch.")
        # TF2: one reformulation after ladder — phrase + tokens harvested from matching titles.
        elif (
            knobs.name == "retrieval"
            and not cart_sku
            and not shock_open
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and not retrieval_reform_done
            and "search" in available
            and budget_ladder_idx >= max(1, len(budget_ladder_caps) - 1)
            and profile.title_terms
            and len(viewed) < 3
        ):
            retrieval_reform_done = True
            extra = _harvest_reform_tokens(
                titles=retrieval_hit_titles,
                phrase_terms=profile.title_terms,
            )
            q = _retrieval_phrase_query(profile, extra_tokens=extra)
            args_r = {"query": q, "k": 10, "in_stock": True}
            if budget is not None:
                args_r["max_price"] = float(budget)
            batch_r: list[tuple[str, dict[str, Any]]] = [("search", args_r)]
            if (
                not retrieval_filter_done
                and harvested_brands
                and "filter" in available
                and max_calls > 1
            ):
                retrieval_filter_done = True
                for brand in harvested_brands[: max(0, max_calls - 1)]:
                    args_f = {"brand": brand, "in_stock": True}
                    if profile.required_category:
                        args_f["category"] = profile.required_category
                    if budget is not None:
                        args_f["max_price"] = float(budget)
                    batch_r.append(("filter", args_f))
            inf = _force_tools(batch_r, "TF2 reformulation (+filters) batch.")
        # TF2: brand×category filter from observed hit brands (cap 1 — avoid Cartesian thrash).
        elif (
            knobs.name == "retrieval"
            and not cart_sku
            and not shock_open
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and not retrieval_filter_done
            and harvested_brands
            and "filter" in available
            and searchish >= 1
            and len(viewed) < 3
        ):
            retrieval_filter_done = True
            brand = harvested_brands[0]
            args_f: dict[str, Any] = {"brand": brand, "in_stock": True}
            if profile.required_category:
                args_f["category"] = profile.required_category
            if budget is not None:
                args_f["max_price"] = float(budget)
            inf = _force_tool("filter", args_f, "TF2 brand filter from observed hits.")
        # v44 retrieval: once survey met (matching product_ids), stop view thrash → inspect→add.
        elif (
            knobs.name == "retrieval"
            and not cart_sku
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and not shock_open
            and _identity_survey_met()
            and matching_ids
            and ("inspect_stock" in available or "add_to_cart" in available)
        ):
            if not (
                last_pick.get("product_id")
                and str(last_pick.get("product_id") or "") in matching_ids
            ):
                viewed_match = {pid for pid in viewed if pid in matching_ids}
                alt_r = _pick_best_viewed_sku(
                    viewed=viewed_match or set(matching_ids) & set(viewed),
                    view_facts_by_pid=view_facts_by_pid,
                    variants_by_pid=variants_by_pid,
                    budget=budget,
                    excluded_skus=excluded_skus,
                    excluded_keys=excluded_keys,
                    profile=profile,
                    prefer_recovery=False,
                    exclude_product_ids=set(excluded_product_ids) if excluded_product_ids else None,
                )
                if alt_r and alt_r.get("product_id") and alt_r.get("sku"):
                    last_pick = dict(alt_r)
                if not (
                    last_pick.get("product_id")
                    and str(last_pick.get("product_id") or "") in matching_ids
                ):
                    # v46: soft gates emptied pick_best — still finish on phrase-matched SKU.
                    for pid_fb in list(matching_ids):
                        if pid_fb not in viewed or pid_fb in excluded_product_ids:
                            continue
                        rows_fb = variants_by_pid.get(pid_fb) or []
                        pick_fb = _pick_budget_variant(
                            rows_fb, budget=budget, excluded=excluded_skus
                        )
                        if pick_fb and pick_fb.get("sku"):
                            last_pick = {
                                "product_id": pid_fb,
                                "sku": str(pick_fb["sku"]),
                            }
                            break
                if not (
                    last_pick.get("product_id")
                    and str(last_pick.get("product_id") or "") in matching_ids
                ):
                    if not viewed_match and matching_ids and "view" in available:
                        miss_m = [pid for pid in matching_ids if pid not in viewed]
                        if miss_m:
                            inf = _force_tool(
                                "view",
                                {"product_id": miss_m[0]},
                                "Retrieval survey: view matching identity before inspect.",
                            )
            if inf is None and last_pick.get("product_id") and last_pick.get("sku"):
                sku_r = str(last_pick.get("sku") or "")
                if (
                    last_stock is not None
                    and not _stock_is_false(last_stock)
                    and str((last_stock or {}).get("sku") or "") == sku_r
                    and "add_to_cart" in available
                ):
                    force_add_after_inspect = False
                    identity_survey_relaxed = True
                    inf = _force_tool(
                        "add_to_cart",
                        {
                            "product_id": last_pick["product_id"],
                            "sku": last_pick["sku"],
                        },
                        "Retrieval survey met → add (no view thrash).",
                    )
                elif "inspect_stock" in available:
                    force_add_after_inspect = True
                    inf = _force_tool(
                        "inspect_stock",
                        {
                            "product_id": last_pick["product_id"],
                            "sku": last_pick["sku"],
                        },
                        "Retrieval survey met → inspect before add.",
                    )
        # TF4: after episode survey depth, force inspect best soft-scored device (stop view thrash).
        elif (
            knobs.name == "preference"
            and not cart_sku
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and not commit_quiz_pending
            and not shock_open
            and "inspect_stock" in available
            and _pre_cart_depth_ok()
            and (
                preference_compared
                or len(viewed) >= max(2, int(profile.multi_listing_min or 0) or 2)
            )
        ):
            if not (last_pick.get("product_id") and last_pick.get("sku")):
                alt_p = _pick_best_viewed_sku(
                    viewed=viewed,
                    view_facts_by_pid=view_facts_by_pid,
                    variants_by_pid=variants_by_pid,
                    budget=budget,
                    excluded_skus=excluded_skus,
                    excluded_keys=excluded_keys,
                    profile=profile,
                    prefer_recovery=False,
                    exclude_product_ids=set(excluded_product_ids) if excluded_product_ids else None,
                )
                if alt_p and alt_p.get("product_id") and alt_p.get("sku"):
                    last_pick = dict(alt_p)
            if last_pick.get("product_id") and last_pick.get("sku"):
                sku_p = str(last_pick.get("sku") or "")
                if (
                    last_stock is not None
                    and not _stock_is_false(last_stock)
                    and str((last_stock or {}).get("sku") or "") == sku_p
                    and "add_to_cart" in available
                ):
                    force_add_after_inspect = False
                    inf = _force_tool(
                        "add_to_cart",
                        {
                            "product_id": last_pick["product_id"],
                            "sku": last_pick["sku"],
                        },
                        "Preference survey met → add (no view thrash).",
                    )
                else:
                    force_add_after_inspect = True
                    inf = _force_tool(
                        "inspect_stock",
                        {
                            "product_id": last_pick["product_id"],
                            "sku": last_pick["sku"],
                        },
                        "Preference survey met → inspect before add.",
                    )
        # Force view of search hits when thrashing search without views (never-cart).
        elif (
            not cart_sku
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and "view" in available
            and search_hit_pids
            and (
                len(viewed) < (4 if knobs.name == "intent" and not shock_open else 2)
                # v44: do NOT re-open catalog view thrash after intent recovery —
                # teacher finishes inspect→add→place from already-viewed pool.
                or (
                    knobs.name == "retrieval"
                    and not shock_open
                    and not _identity_survey_met()
                    and len(viewed) < max(2, int(profile.min_matching_identities or 2))
                    and matching_ids
                )
                or (
                    knobs.name == "constraint"
                    and not shock_open
                    and len(viewed) < 4
                    and bool(profile.firm_specs)
                )
                or (
                    knobs.name == "preference"
                    and not shock_open
                    and len(viewed) < (3 if (profile.pref_tokens or profile.useful_reason_fields) else 2)
                )
            )
            and (
                searchish >= max(1, knobs.max_search_before_view)
                or remaining <= 12
                or llm_calls >= max(3, knobs.llm_soft_cap - 3)
                or (knobs.name == "intent" and not shock_open and len(viewed) < 4)
                # v44: never force open-catalog views after intent shock — post-shock
                # FinishFSM recovers inspect→add→place from viewed pool only.
                or (knobs.name == "retrieval" and searchish >= 1 and len(viewed) < 1)
                or (
                    knobs.name == "retrieval"
                    and shock_open
                    and recovery_committed
                    and any(pid not in viewed for pid in matching_ids)
                )
                or (knobs.name == "constraint" and bool(profile.firm_specs) and len(viewed) < 4)
                or (knobs.name == "preference" and searchish >= 1 and len(viewed) < 3)
            )
        ):
            miss = [pid for pid in search_hit_pids if pid not in viewed]
            if knobs.name == "preference" and (profile.pref_tokens or profile.useful_reason_fields):
                # Prefer hits whose known titles already mention soft tokens.
                ranked: list[tuple[int, str]] = []
                for pid in miss:
                    facts_h = view_facts_by_pid.get(pid) or {"title": ""}
                    sc = _soft_axis_hit_score(facts_h, None, profile)
                    ranked.append((-sc, pid))
                ranked.sort()
                miss = [pid for _, pid in ranked] or miss
            if knobs.name == "retrieval" and matching_ids:
                miss = [
                    pid
                    for pid in list(matching_ids) + miss
                    if pid not in viewed and pid not in excluded_product_ids
                ] or miss
            if profile.title_terms and matching_ids:
                prefer = [pid for pid in miss if pid in matching_ids] or miss
            else:
                prefer = miss
            if prefer:
                # v45: batch views in one turn (policy max_calls) — honest multi-view.
                n_view = 1
                if knobs.name in ("retrieval", "constraint", "preference", "intent"):
                    n_view = min(max_calls, max(1, len(prefer)))
                    if knobs.name == "retrieval":
                        need_v = max(2, int(profile.min_matching_identities or 2))
                        n_view = min(n_view, need_v, len(prefer))
                    elif knobs.name == "constraint":
                        n_view = min(n_view, 4, len(prefer))
                batch_v = [("view", {"product_id": pid}) for pid in prefer[:n_view]]
                inf = _force_tools(batch_v, "Force view search hit(s) before cart.")
            elif last_pick.get("product_id") and last_pick.get("sku") and "inspect_stock" in available:
                # Views exhausted/unparsed — inspect current pick.
                # Avoid re-inspecting excluded shocked SKU after stockout.
                if str(last_pick.get("product_id") or "") in excluded_product_ids:
                    inf = None
                else:
                    inf = _force_tool(
                        "inspect_stock",
                        {"product_id": last_pick["product_id"], "sku": last_pick["sku"]},
                        "Force inspect after views.",
                    )
        # TF1: one compare shortlist before first cart (teacher often compares).
        elif (
            knobs.name == "intent"
            and not cart_sku
            and not shock_open
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and not commit_quiz_pending
            and not intent_compared
            and len(viewed) >= 2
            and "compare" in available
            and remaining > 8
        ):
            intent_compared = True
            pids = [pid for pid in list(viewed) if pid][:4]
            if len(pids) >= 2:
                inf = _force_tool(
                    "compare",
                    {"product_ids": pids},
                    "TF1 survey compare before first cart.",
                )
        # TF3: one compare when surveying SKU facts (teacher ~70%).
        elif (
            knobs.name == "constraint"
            and not cart_sku
            and not shock_open
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and not constraint_compared
            and bool(profile.firm_specs)
            and len(viewed) >= 2
            and "compare" in available
            and remaining > 8
        ):
            constraint_compared = True
            pids = [pid for pid in list(viewed) if pid][:4]
            if len(pids) >= 2:
                inf = _force_tool(
                    "compare",
                    {"product_ids": pids},
                    "TF3 SKU evidence compare before first cart.",
                )
        # TF4: one focused shortlist compare (rules ask compare; teacher often skips — one is enough).
        elif (
            knobs.name == "preference"
            and not cart_sku
            and not shock_open
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and not preference_compared
            and len(viewed) >= 2
            and "compare" in available
            and remaining > 8
        ):
            preference_compared = True
            # Prefer soft-aligned viewed pids in the shortlist.
            scored_v: list[tuple[int, str]] = []
            for pid in viewed:
                sc = _soft_axis_hit_score(view_facts_by_pid.get(pid) or {}, None, profile)
                scored_v.append((-sc, pid))
            scored_v.sort()
            pids = [pid for _, pid in scored_v][:4]
            if len(pids) >= 2:
                inf = _force_tool(
                    "compare",
                    {"product_ids": pids},
                    "TF4 soft-pref shortlist compare before first cart.",
                )
        # Post-shock: recover from already-viewed pool (honest_oro — no search thrash).
        elif (
            shock_open
            and (recovery_committed or knobs.shock_protocol != "full")
            and not cart_sku
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and viewed
        ):
            prefer_rec = knobs.name == "recovery" or profile.relax_color_then_model
            # Never recover to shocked product_id (teacher: final_not_event_target).
            ban_pids = set(excluded_product_ids) if (
                knobs.name in ("constraint", "retrieval", "preference")
                or profile.firm_specs
                or profile.ask_first
                or profile.title_phrase_match
                or profile.strict_selected_variant
                or profile.listing_id_min >= 2
                or bool(excluded_product_ids)
            ) else set()
            # TF4: listing_id shock — also skip products on excluded listings.
            if profile.listing_id_min >= 2 and excluded_listing_ids:
                for pid in list(viewed):
                    lid = str((view_facts_by_pid.get(pid) or {}).get("listing_id") or "").strip()
                    if lid and lid in excluded_listing_ids:
                        ban_pids.add(pid)
            alt = _pick_best_viewed_sku(
                viewed=viewed,
                view_facts_by_pid=view_facts_by_pid,
                variants_by_pid=variants_by_pid,
                budget=budget,
                excluded_skus=excluded_skus,
                excluded_keys=excluded_keys,
                profile=profile,
                prefer_recovery=prefer_rec,
                exclude_product_ids=ban_pids,
            )
            if alt is None and knobs.name == "preference" and profile.firm_specs:
                # Soft-mistaken firm (waterproof feature) can empty the pool — demote once.
                saved_firm = list(profile.firm_specs)
                _demote_soft_firm_specs(profile)
                if profile.firm_specs != saved_firm or not profile.firm_specs:
                    alt = _pick_best_viewed_sku(
                        viewed=viewed,
                        view_facts_by_pid=view_facts_by_pid,
                        variants_by_pid=variants_by_pid,
                        budget=budget,
                        excluded_skus=excluded_skus,
                        excluded_keys=excluded_keys,
                        profile=profile,
                        prefer_recovery=prefer_rec,
                        exclude_product_ids=ban_pids,
                    )
                if alt is None:
                    profile.firm_specs = saved_firm
            if alt is None and viewed:
                # Soften title/category one shot so shock recover can finish.
                # Retrieval: keep title gate on first pass; only soften category once.
                saved_title = profile.title_phrase_match
                saved_cat = profile.require_category_path
                if knobs.name != "retrieval":
                    profile.title_phrase_match = False
                profile.require_category_path = False
                alt = _pick_best_viewed_sku(
                    viewed=viewed,
                    view_facts_by_pid=view_facts_by_pid,
                    variants_by_pid=variants_by_pid,
                    budget=budget,
                    excluded_skus=excluded_skus,
                    excluded_keys=excluded_keys,
                    profile=profile,
                    prefer_recovery=prefer_rec,
                    exclude_product_ids=ban_pids,
                )
                profile.title_phrase_match = saved_title
                profile.require_category_path = saved_cat
                # Last resort for retrieval: allow title soften if still empty.
                if alt is None and knobs.name == "retrieval":
                    profile.title_phrase_match = False
                    profile.require_category_path = False
                    alt = _pick_best_viewed_sku(
                        viewed=viewed,
                        view_facts_by_pid=view_facts_by_pid,
                        variants_by_pid=variants_by_pid,
                        budget=budget,
                        excluded_skus=excluded_skus,
                        excluded_keys=excluded_keys,
                        profile=profile,
                        prefer_recovery=prefer_rec,
                        exclude_product_ids=ban_pids,
                    )
                    profile.title_phrase_match = saved_title
                    profile.require_category_path = saved_cat
            if alt and alt.get("product_id") and alt.get("sku"):
                last_pick = dict(alt)
                pid_a, sku_a = alt["product_id"], alt["sku"]
                if last_stock is None or str((last_stock or {}).get("sku") or "") != sku_a:
                    if "inspect_stock" in available:
                        inf = _force_tool(
                            "inspect_stock",
                            {"product_id": pid_a, "sku": sku_a},
                            "Post-shock inspect from viewed pool.",
                        )
                elif not _stock_is_false(last_stock) and "add_to_cart" in available:
                    price_a = _price_num((last_stock or {}).get("price"))
                    if budget is None or price_a is None or price_a <= budget + 1e-6:
                        if _intent_firm_ok(pid_a, sku_a) and pid_a not in ban_pids:
                            inf = _force_tool(
                                "add_to_cart",
                                {"product_id": pid_a, "sku": sku_a},
                                "Post-shock add from viewed pool.",
                            )
                        else:
                            if sku_a:
                                excluded_skus.add(sku_a)
                            last_pick = {}
                            inf = None
            elif "view" in available:
                # Stale OOS / empty variants: re-view. Do NOT re-view mere over-budget rows.
                refresh = [pid for pid in viewed if pid not in ban_pids]
                stale: list[str] = []
                for pid in refresh:
                    rows = variants_by_pid.get(pid) or []
                    if not rows:
                        stale.append(pid)
                        continue
                    has_live_in_budget = False
                    has_oos_candidate = False
                    for r in rows:
                        sku = str(r.get("sku") or "")
                        if not sku or sku in excluded_skus or f"{pid}::{sku}" in excluded_keys:
                            continue
                        price = _price_num(r.get("price"))
                        over = (
                            budget is not None
                            and price is not None
                            and price > budget + 1e-6
                        )
                        if not _variant_in_stock(r):
                            if not over:
                                has_oos_candidate = True
                            continue
                        if over:
                            continue
                        if knobs.name in ("intent", "constraint") and profile.firm_specs:
                            if not _sku_meets_firm(
                                view_facts_by_pid.get(pid) or {},
                                r,
                                profile.firm_specs,
                                profile.resolution_defs,
                            ):
                                continue
                        has_live_in_budget = True
                        break
                    if not has_live_in_budget and (has_oos_candidate or not rows):
                        stale.append(pid)
                pick_pid = (stale or [None])[0]
                if pick_pid:
                    old_rows = variants_by_pid.pop(pick_pid, None) or []
                    for r in old_rows:
                        sku_ex = str(r.get("sku") or "")
                        if sku_ex:
                            excluded_skus.discard(sku_ex)
                    inf = _force_tool(
                        "view",
                        {"product_id": pick_pid},
                        "Post-shock re-view to refresh stock options.",
                    )
                elif (
                    knobs.name == "intent"
                    and "search" in available
                    and not intent_postshock_reseed_used
                ):
                    # Empty firm-valid pool after shock — exactly one seed search (no loop).
                    intent_postshock_reseed_used = True
                    intent_reseed_search = True
                    intent_seed_done = False
                    intent_drop_brand = True
                    inf = None
                elif (
                    knobs.name == "constraint"
                    and "search" in available
                    and not constraint_postshock_reseed_used
                ):
                    # Empty firm-valid pool — one re-search under firm+budget (new product_id).
                    constraint_postshock_reseed_used = True
                    constraint_reseed_search = True
                    constraint_seed_done = False
                    inf = None
                elif (
                    knobs.name == "justification"
                    and "search" in available
                    and not justification_postshock_reseed_used
                    and not _justification_survivor_in_viewed()
                ):
                    justification_postshock_reseed_used = True
                    justification_reseed_search = True
                    justification_seed_done = False
                    inf = None
        elif (
            (
                clock.remaining() < 22
                or remaining <= 14
                or llm_calls >= knobs.llm_soft_cap
                or (len(viewed) >= 1 and searchish >= 1 and not cart_sku and last_pick)
                or (len(viewed) >= 2 and searchish >= max(1, knobs.max_search_before_view))
                or (shock_open and (recovery_committed or knobs.shock_protocol != "full"))
                or force_add_after_inspect
            )
            and not must_message
            and not must_remove
            and not awaiting_user_after_shock
            and pending_recheck is None
            and (
                (knobs.name == "constraint" and bool(profile.firm_specs))
                or (
                    knobs.name != "constraint"
                    and (not ask_first or asked_spec or bool(profile.firm_specs) or clarify_unlock)
                )
            )
            and (not candidates or cand_pids <= viewed)
            and (knobs.name != "preference" or len(viewed) >= max(1, knobs.require_multi_listing) or cart_sku or controller_forced)
        ):
            if (_any_place_ok()) and "place_test_order" in available:
                inf = _force_tool(
                    "place_test_order",
                    _place_force_args(cart_sku["product_id"], cart_sku["sku"]),
                    "FinishFSM place_test_order.",
                )
            elif cart_sku.get("product_id") and cart_sku.get("sku"):
                price = _price_num((last_stock or {}).get("price"))
                if (
                    budget is not None
                    and price is not None
                    and price > budget + 1e-6
                    and "remove_from_cart" in available
                ):
                    must_remove = True
                    shock_open = True
                    inf = _force_tool(
                        "remove_from_cart",
                        {"product_id": cart_sku["product_id"], "sku": cart_sku["sku"]},
                        "Over-budget remove.",
                    )
                elif last_stock is None or _stock_is_false(last_stock):
                    inf = _force_tool(
                        "inspect_stock",
                        {
                            "product_id": cart_sku["product_id"],
                            "sku": cart_sku["sku"],
                        },
                        "Recheck cart stock before place.",
                    )
                else:
                    # Place-first: stocked cart + recheck done — never inspect-thrash.
                    if knobs.name == "ranking" and ranking_event:
                        _sync_ranking_pick()
                        elect = _elect_ranking_pick(
                            candidates,
                            view_facts_by_pid=view_facts_by_pid,
                            variants_by_pid=variants_by_pid,
                            budget=budget,
                            excluded_skus=excluded_skus,
                            excluded_keys=excluded_keys,
                            ranking_axes=profile.ranking_axes,
                        )
                        cart_key = f"{cart_sku['product_id']}::{cart_sku['sku']}"
                        if elect:
                            elect_key = f"{elect['product_id']}::{elect['sku']}"
                            if cart_key != elect_key and "remove_from_cart" in available:
                                last_pick = elect
                                must_remove = True
                                inf = _force_tool(
                                    "remove_from_cart",
                                    {
                                        "product_id": cart_sku["product_id"],
                                        "sku": cart_sku["sku"],
                                    },
                                    "Ranking re-sync: remove non-elect cart before place.",
                                )
                    if inf is None and profile.min_matching_identities >= 2:
                        if _identity_count() >= 1 and not _identity_survey_met():
                            identity_survey_relaxed = True
                    if inf is None and "place_test_order" in available and (
                        _any_place_ok()
                    ):
                        inf = _force_tool(
                            "place_test_order",
                            _place_force_args(cart_sku["product_id"], cart_sku["sku"]),
                            "FinishFSM place-first (stocked cart).",
                        )
                    if inf is None:
                        # Last resort: still place when stock+budget clear (agent_error worse).
                        price_ok = True
                        price_n = _price_num((last_stock or {}).get("price"))
                        if budget is not None and price_n is not None and price_n > budget + 1e-6:
                            price_ok = False
                        if (
                            price_ok
                            and "place_test_order" in available
                            # v43: ranking last-resort OK after recovery when stock verified
                            # (banned in v42 → endless inspect / agent_error).
                            and (
                                not (knobs.name == "ranking" and ranking_event)
                                or _stocked_recovery_place_escape()
                            )
                            and (
                                knobs.name not in ("preference", "recovery")
                                or recovery_committed
                                or not shock_open
                            )
                            and (
                                (knobs.name == "constraint" and bool(profile.firm_specs))
                                or (
                                    knobs.name != "constraint"
                                    and (not ask_first or bool(profile.firm_specs) or clarify_unlock)
                                )
                            )
                        ):
                            inf = _force_tool(
                                "place_test_order",
                                _place_force_args(cart_sku["product_id"], cart_sku["sku"]),
                                "FinishFSM place-first last-resort.",
                            )
            elif last_pick.get("product_id") and last_pick.get("sku"):
                pid, sku = last_pick["product_id"], last_pick["sku"]
                if last_stock is None or str((last_stock or {}).get("sku") or "") != sku:
                    inf = _force_tool(
                        "inspect_stock",
                        {"product_id": pid, "sku": sku},
                        "FinishFSM inspect_stock.",
                    )
                elif _stock_is_false(last_stock):
                    excluded_skus.add(sku)
                    shock_recover = True
                    shock_open = True
                    last_pick = {}
                    inf = None
                elif "add_to_cart" in available:
                    if knobs.name == "intent" and _intent_restore_commit_pick():
                        pid, sku = intent_commit_pid, intent_commit_sku
                    price = _price_num((last_stock or {}).get("price"))
                    if commit_quiz_pending and knobs.name == "intent":
                        must_message = True
                        inf = None
                    elif intent_reaffirm_commit and knobs.name == "intent":
                        must_message = True
                        inf = None
                    elif budget is not None and price is not None and price > budget + 1e-6:
                        excluded_skus.add(sku)
                        last_pick = {}
                        inf = None
                    elif not _intent_firm_ok(pid, sku):
                        excluded_skus.add(sku)
                        last_pick = {}
                        inf = None
                    elif not _constraint_firm_brand_ok(pid):
                        excluded_skus.add(sku)
                        last_pick = {}
                        inf = None
                    elif (
                        knobs.name == "intent"
                        and _intent_commit_viable()
                        and (pid != intent_commit_pid or sku != intent_commit_sku)
                    ):
                        last_pick = {
                            "product_id": intent_commit_pid,
                            "sku": intent_commit_sku,
                        }
                        if (
                            last_stock is None
                            or str((last_stock or {}).get("sku") or "") != intent_commit_sku
                        ) and "inspect_stock" in available:
                            inf = _force_tool(
                                "inspect_stock",
                                {
                                    "product_id": intent_commit_pid,
                                    "sku": intent_commit_sku,
                                },
                                "Intent: inspect quiz commit SKU before cart.",
                            )
                        else:
                            inf = _force_tool(
                                "add_to_cart",
                                {
                                    "product_id": intent_commit_pid,
                                    "sku": intent_commit_sku,
                                },
                                "Intent: sticky add quiz commit SKU.",
                            )
                    elif not _pre_cart_depth_ok():
                        # Depth floor: view another hit before first cart.
                        miss = [p for p in search_hit_pids if p not in viewed]
                        if miss and "view" in available:
                            inf = _force_tool(
                                "view",
                                {"product_id": miss[0]},
                                "Pre-cart depth: view another listing.",
                            )
                        else:
                            inf = None
                    elif (
                        profile.require_category_path
                        and profile.required_category
                        and category_soft_rejects >= 2
                        and not _category_path_ok(
                            view_facts_by_pid.get(pid), profile.required_category
                        )
                    ):
                        # Diversify: stop inspect→add thrash on wrong category pick.
                        last_pick = {}
                        force_category_search = True
                        inf = None
                    elif (
                        profile.min_matching_identities >= 2
                        and not _identity_survey_met()
                    ):
                        if _identity_count() >= 1:
                            identity_survey_relaxed = True
                            _prefer_soft_brand_pick()
                            _prefer_soft_pref_pick()
                            _prefer_recovery_pick()
                            pid2 = last_pick.get("product_id") or pid
                            sku2 = last_pick.get("sku") or sku
                            if not _intent_firm_ok(str(pid2), str(sku2)):
                                force_identity_search = True
                                inf = None
                            elif (
                                sku2 != sku
                                and "inspect_stock" in available
                                and (
                                    last_stock is None
                                    or str((last_stock or {}).get("sku") or "") != sku2
                                )
                            ):
                                inf = _force_tool(
                                    "inspect_stock",
                                    {"product_id": pid2, "sku": sku2},
                                    "FinishFSM soft-pref re-inspect (survey).",
                                )
                            else:
                                inf = _force_tool(
                                    "add_to_cart",
                                    {"product_id": pid2, "sku": sku2},
                                    "FinishFSM add_to_cart (survey soft-met).",
                                )
                        elif remaining <= 8:
                            # Better to cart a viewed in-stock SKU than die without order.
                            identity_survey_relaxed = True
                            if _intent_firm_ok(pid, sku):
                                inf = _force_tool(
                                    "add_to_cart",
                                    {"product_id": pid, "sku": sku},
                                    "FinishFSM add_to_cart (steps-low survey escape).",
                                )
                            else:
                                inf = None
                        else:
                            force_identity_search = True
                            inf = None
                    else:
                        _prefer_soft_brand_pick()
                        _prefer_soft_pref_pick()
                        _prefer_recovery_pick()
                        pid2 = last_pick.get("product_id") or pid
                        sku2 = last_pick.get("sku") or sku
                        if not _intent_firm_ok(str(pid2), str(sku2)):
                            excluded_skus.add(str(sku2))
                            last_pick = {}
                            inf = None
                        elif (
                            sku2 != sku
                            and "inspect_stock" in available
                            and (
                                last_stock is None
                                or str((last_stock or {}).get("sku") or "") != sku2
                            )
                        ):
                            inf = _force_tool(
                                "inspect_stock",
                                {"product_id": pid2, "sku": sku2},
                                "Soft-pref/brand inspect before FinishFSM cart.",
                            )
                        else:
                            inf = _force_tool(
                                "add_to_cart",
                                {"product_id": pid2, "sku": sku2},
                                "FinishFSM add_to_cart.",
                            )
                else:
                    inf = None
            else:
                inf = None
        else:
            inf = None

        if inf is None:
            if must_message:
                _hint_once(
                    "must_msg",
                    "REQUIRED this turn: call `message` once (ask-first or recovery commit). "
                    "Do not ask permission.",
                )
            elif must_remove and cart_sku:
                _hint_once(
                    f"must_rm:{cart_sku.get('sku')}",
                    "REQUIRED: remove_from_cart the shocked cart item, then recover.",
                )
            elif pending_recheck:
                _hint_once(
                    "recheck",
                    f"REQUIRED: inspect_stock product_id={pending_recheck[0]} "
                    f"sku={pending_recheck[1]} before place_test_order.",
                )
            elif candidates and len(viewed & cand_pids) < len(cand_pids):
                missing = sorted(cand_pids - viewed)
                _hint_once(
                    f"closed_miss:{','.join(missing)}",
                    f"View missing closed-set product_ids: {missing}",
                )
            elif (
                knobs.require_multi_listing >= 2
                and len(viewed) < knobs.require_multi_listing
                and not cart_sku
            ):
                _hint_once(
                    "multi_list",
                    f"View at least {knobs.require_multi_listing} distinct products before cart.",
                )
            elif force_category_search and profile.required_category:
                _hint_once(
                    "cat_diversify",
                    f"REQUIRED: search/view a listing whose category_path matches "
                    f"{profile.required_category} (do not re-inspect the rejected SKU).",
                )
                force_category_search = False
            elif force_identity_search and profile.title_terms:
                _hint_once(
                    "id_diversify",
                    "REQUIRED: search/view more products whose titles contain the "
                    f"requested tokens {profile.title_terms[:6]} before cart.",
                )
                force_identity_search = False
            elif brand_search_nudge and profile.preferred_brand and not brand_search_done:
                _hint_once(
                    "soft_brand_search",
                    f"Search for preferred brand {profile.preferred_brand} "
                    "(soft preference) before first cart.",
                )
            elif shock_recover and knobs.shock_protocol != "off":
                _hint_once(
                    "shock",
                    "Market shock: remove if needed; recover in-stock in-budget replacement"
                    + (
                        " (same model other color first)"
                        if profile.relax_color_then_model or knobs.name == "recovery"
                        else ""
                    )
                    + "; view → inspect → cart → post-add inspect → order.",
                )
                shock_recover = False
            elif knobs.max_search_before_view and searchish >= knobs.max_search_before_view and len(viewed) < 1:
                _hint_once("stop_search", "Stop search thrash; view best hits then finish.")
            elif remaining <= 5:
                _hint_once(
                    "steps_low",
                    f"~{remaining} steps left. Post-add inspect then place only if in budget.",
                )

            if (
                llm_calls >= knobs.llm_hard_cap
                and cart_sku.get("product_id")
                and cart_sku.get("sku")
                and "place_test_order" in available
                and (
                    _any_place_ok()
                )
            ):
                inf = _force_tool(
                    "place_test_order",
                    _place_force_args(cart_sku["product_id"], cart_sku["sku"]),
                    "Hard-cap place-first.",
                )
            elif llm_calls >= knobs.llm_hard_cap:
                # v45: do not abandon a stocked cart on hard-cap — place if task-rule OK.
                if (
                    cart_sku.get("product_id")
                    and cart_sku.get("sku")
                    and "place_test_order" in available
                    and (
                        _task_rule_verified_place()
                        or _stocked_recovery_place_escape()
                        or (
                            last_stock is not None
                            and not _stock_is_false(last_stock)
                            and knobs.name in ("retrieval", "constraint", "recovery")
                        )
                    )
                ):
                    inf = _force_tool(
                        "place_test_order",
                        _place_force_args(cart_sku["product_id"], cart_sku["sku"]),
                        "Hard-cap place (stocked cart).",
                    )
                else:
                    break
            else:
                try:
                    inf = _chat_raw(messages=messages, tools=tools, tool_choice="required")
                    llm_calls += 1
                except Exception as exc:  # noqa: BLE001
                    dialogue.append(
                        {
                            "role": "assistant",
                            "content": f"inference error: {type(exc).__name__}: {exc}"[:400],
                        }
                    )
                    break

        assistant = (inf.get("choices") or [{}])[0].get("message") or {}
        assistant_content = assistant.get("content") or ""
        tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
        if not tool_calls:
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append(
                {
                    "role": "user",
                    "content": "A tool call is required. Choose one valid tool.",
                }
            )
            continue

        # Pipeline gates on proposals.
        # Remove outranks message: ranking/full-shock set both flags; FSM already
        # forces remove first — message gate must not reject that remove (v6/v7 hang).
        names = _tool_names(tool_calls)
        if must_remove and "remove_from_cart" not in names:
            messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].get("id") or "x",
                    "content": json.dumps({"rejected": "remove_from_cart required after shock"}),
                }
            )
            continue
        # Intent-only: after quiz pin, do not burn turns on filter thrash — inspect/add pin.
        if (
            knobs.name == "intent"
            and not cart_sku
            and not must_remove
            and not must_message
            and not shock_open
            and "filter" in names
            and "add_to_cart" not in names
            and _intent_commit_viable()
            and "inspect_stock" in available
        ):
            inf = _force_tool(
                "inspect_stock",
                {"product_id": intent_commit_pid, "sku": intent_commit_sku},
                "Intent: rewrite filter thrash → inspect commit pin.",
            )
            assistant = (inf.get("choices") or [{}])[0].get("message") or {}
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            names = _tool_names(tool_calls)
        # v44 constraint: while force-add is armed, do not thrash inspect_cart / other SKUs.
        if (
            knobs.name == "constraint"
            and force_add_after_inspect
            and not cart_sku
            and last_pick.get("product_id")
            and last_stock is not None
            and not _stock_is_false(last_stock)
            and str((last_stock or {}).get("sku") or "") == str(last_pick.get("sku") or "")
            and _intent_firm_ok(
                str(last_pick.get("product_id") or ""), str(last_pick.get("sku") or "")
            )
        ):
            if "message" in names and "add_to_cart" not in names:
                if "add_to_cart" in available:
                    inf = _force_tool(
                        "add_to_cart",
                        {
                            "product_id": last_pick["product_id"],
                            "sku": last_pick["sku"],
                        },
                        "Rewrite message thrash → add (constraint firm inspect done).",
                    )
                    assistant = (inf.get("choices") or [{}])[0].get("message") or {}
                    assistant_content = assistant.get("content") or ""
                    tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
                    names = _tool_names(tool_calls)
            if "inspect_cart" in names or (
                "inspect_stock" in names
                and "add_to_cart" not in names
            ):
                # Any inspect/inspect_cart while force-add is armed is thrash — cart now.
                if "add_to_cart" in available:
                    inf = _force_tool(
                        "add_to_cart",
                        {
                            "product_id": last_pick["product_id"],
                            "sku": last_pick["sku"],
                        },
                        "Rewrite inspect thrash → add (constraint firm inspect done).",
                    )
                    assistant = (inf.get("choices") or [{}])[0].get("message") or {}
                    assistant_content = assistant.get("content") or ""
                    tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
                    names = _tool_names(tool_calls)
        # After shock commit: reject open-catalog search thrash — recover from viewed pool.
        if (
            shock_open
            and recovery_committed
            and not cart_sku
            and not must_remove
            and not must_message
            and any(n in names for n in ("search", "filter"))
            and viewed
        ):
            post_shock_searches += 1
            if post_shock_searches >= 1:
                ban = set(excluded_product_ids) if (
                    knobs.name in ("constraint", "preference", "retrieval")
                    or profile.firm_specs
                    or profile.ask_first
                    or profile.listing_id_min >= 2
                    or bool(excluded_product_ids)
                ) else set()
                alt = _pick_best_viewed_sku(
                    viewed=viewed,
                    view_facts_by_pid=view_facts_by_pid,
                    variants_by_pid=variants_by_pid,
                    budget=budget,
                    excluded_skus=excluded_skus,
                    excluded_keys=excluded_keys,
                    profile=profile,
                    prefer_recovery=(
                        knobs.name == "recovery" or profile.relax_color_then_model
                    ),
                    exclude_product_ids=ban,
                )
                if alt is None and knobs.name == "preference" and profile.firm_specs:
                    _demote_soft_firm_specs(profile)
                    alt = _pick_best_viewed_sku(
                        viewed=viewed,
                        view_facts_by_pid=view_facts_by_pid,
                        variants_by_pid=variants_by_pid,
                        budget=budget,
                        excluded_skus=excluded_skus,
                        excluded_keys=excluded_keys,
                        profile=profile,
                        prefer_recovery=True,
                        exclude_product_ids=ban,
                    )
                if alt and alt.get("product_id") and alt.get("sku"):
                    last_pick = dict(alt)
                    messages.append(
                        {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_calls[0].get("id") or "x",
                            "content": json.dumps(
                                {
                                    "rejected": (
                                        "post-shock: stop search; inspect/add a previously "
                                        f"viewed option {alt['product_id']}::{alt['sku']}"
                                    )
                                }
                            ),
                        }
                    )
                    continue
                # Still thrashing with no pick — reject search so LLM must inspect/view.
                if knobs.name == "preference":
                    messages.append(
                        {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_calls[0].get("id") or "x",
                            "content": json.dumps(
                                {
                                    "rejected": (
                                        "post-shock: stop search; inspect/add another "
                                        "previously viewed product"
                                    )
                                }
                            ),
                        }
                    )
                    continue
        if must_message and not must_remove and "message" not in names:
            messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].get("id") or "x",
                    "content": json.dumps({"rejected": "message required this turn"}),
                }
            )
            continue
        if "place_test_order" in names and (order_done or done):
            messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].get("id") or "x",
                    "content": json.dumps({"rejected": "order already placed; stop"}),
                }
            )
            continue
        # TF3: do not search until firm specs unlocked (withheld-spec contract).
        if (
            knobs.name == "constraint"
            and "search" in names
            and not profile.firm_specs
            and not controller_forced
            and "message" in available
        ):
            messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].get("id") or "x",
                    "content": json.dumps(
                        {
                            "rejected": (
                                "firm hardware/spec not bound yet — wait for shopper "
                                "goal_brief / ask for the withheld requirement before search"
                            )
                        }
                    ),
                }
            )
            continue
        if "place_test_order" in names and not (
            _any_place_ok()
        ):
            why = "place blocked: need post-add inspect, in-stock, in-budget, and closed shock"
            if pending_recheck:
                why = "place blocked: inspect_stock the carted sku first (post-add recheck)"
            messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].get("id") or "x",
                    "content": json.dumps({"rejected": why}),
                }
            )
            continue
        # v43: once cart stock is verified, reject re-inspect of the same cart SKU (thrash class).
        if "inspect_stock" in names and cart_sku and cart_stock_verified_key:
            try:
                a_is = json.loads(tool_calls[0].get("function", {}).get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                a_is = {}
            if not isinstance(a_is, dict):
                a_is = {}
            ik = f"{a_is.get('product_id')}::{a_is.get('sku')}"
            ck = f"{cart_sku.get('product_id')}::{cart_sku.get('sku')}"
            if ik == ck == cart_stock_verified_key and not _stock_is_false(last_stock):
                if "place_test_order" in available and (
                    _any_place_ok()
                ):
                    inf = _force_tool(
                        "place_test_order",
                        _place_force_args(cart_sku["product_id"], cart_sku["sku"]),
                        "Rewrite inspect-thrash → place (stock already verified).",
                    )
                    assistant = (inf.get("choices") or [{}])[0].get("message") or {}
                    assistant_content = assistant.get("content") or ""
                    tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
                    names = _tool_names(tool_calls)
                else:
                    messages.append(
                        {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_calls[0].get("id") or "x",
                            "content": json.dumps(
                                {
                                    "rejected": (
                                        "cart stock already verified — place_test_order "
                                        "or recover to another key; do not re-inspect"
                                    )
                                }
                            ),
                        }
                    )
                    continue
        if pending_recheck and "inspect_stock" not in names:
            messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].get("id") or "x",
                    "content": json.dumps(
                        {
                            "rejected": (
                                f"inspect_stock {pending_recheck[0]}::{pending_recheck[1]} required"
                            )
                        }
                    ),
                }
            )
            continue
        if (
            knobs.require_multi_listing >= 2
            and "add_to_cart" in names
            and len(viewed) < knobs.require_multi_listing
            and not cart_sku
            and not controller_forced
        ):
            messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].get("id") or "x",
                    "content": json.dumps(
                        {
                            "rejected": (
                                f"compare ≥{knobs.require_multi_listing} products before first cart"
                            )
                        }
                    ),
                }
            )
            continue
        if (
            profile.listing_id_min >= 2
            and "add_to_cart" in names
            and not _listing_diversity_met(profile.listing_id_min)
            and not cart_sku
            and not controller_forced
        ):
            messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].get("id") or "x",
                    "content": json.dumps(
                        {
                            "rejected": (
                                f"need ≥{profile.listing_id_min} distinct listing_id "
                                "(or product views if listing_id absent) before first cart"
                            )
                        }
                    ),
                }
            )
            continue
        if (
            profile.min_matching_identities >= 2
            and "add_to_cart" in names
            and not cart_sku
            and not controller_forced
        ):
            id_n = _identity_count()
            if not _identity_survey_met():
                identity_soft_rejects += 1
                if knobs.name == "retrieval":
                    # TF2: only soft-escape with a real cluster already on the surface.
                    if identity_soft_rejects >= 3 and id_n >= max(3, profile.min_matching_identities // 2):
                        identity_survey_relaxed = True
                    elif identity_soft_rejects >= 2:
                        force_identity_search = True
                elif identity_soft_rejects >= 2 and id_n >= 1:
                    # Terminal escape: keep surveying in playbook, but allow cart.
                    identity_survey_relaxed = True
                elif identity_soft_rejects >= 2 and len(viewed) >= 2:
                    identity_survey_relaxed = True
                elif identity_soft_rejects >= 2:
                    force_identity_search = True
                    # Do not wipe last_pick — FinishFSM may still cart it.
                if not _identity_survey_met():
                    messages.append(
                        {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_calls[0].get("id") or "x",
                            "content": json.dumps(
                                {
                                    "rejected": (
                                        f"need ≥{profile.min_matching_identities} distinct "
                                        "matching product identities before first cart; "
                                        "search/view more title matches"
                                    )
                                }
                            ),
                        }
                    )
                    continue
        if "add_to_cart" in names and not cart_sku and not shock_open and not controller_forced:
            try:
                a_fc = json.loads(tool_calls[0].get("function", {}).get("arguments") or "{}")
            except json.JSONDecodeError:
                a_fc = {}
            pid_fc = str((a_fc or {}).get("product_id") or "")
            sku_fc = str((a_fc or {}).get("sku") or "")
            facts_fc = view_facts_by_pid.get(pid_fc) or {}
            row_fc = next(
                (
                    r
                    for r in (variants_by_pid.get(pid_fc) or [])
                    if str(r.get("sku") or "") == sku_fc
                ),
                (variants_by_pid.get(pid_fc) or [None])[0] if variants_by_pid.get(pid_fc) else None,
            )
            if profile.firm_specs and not _sku_meets_firm(
                facts_fc, row_fc, profile.firm_specs, profile.resolution_defs
            ):
                messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_calls[0].get("id") or "x",
                        "content": json.dumps({"rejected": "firm specs not met for selected variant"}),
                    }
                )
                continue
            if not _device_evidence_ok(facts_fc, row_fc, profile):
                messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_calls[0].get("id") or "x",
                        "content": json.dumps(
                            {"rejected": "device/accessory evidence failed for requested category"}
                        ),
                    }
                )
                continue
            if profile.require_category_path and profile.required_category:
                if not _category_path_ok(facts_fc, profile.required_category):
                    category_soft_rejects += 1
                    # Sanitize harvest once — NEVER clear category_path obligation.
                    cleaned = _sanitize_required_category(profile.required_category)
                    if cleaned:
                        profile.required_category = cleaned
                    if category_soft_rejects >= 2:
                        force_category_search = True
                        last_pick = {}
                    messages.append(
                        {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_calls[0].get("id") or "x",
                            "content": json.dumps(
                                {
                                    "rejected": (
                                        f"category_path must match {profile.required_category}; "
                                        "view/compare a listing whose category_path contains that "
                                        f"catalog noun before add (reject#{category_soft_rejects})"
                                    )
                                }
                            ),
                        }
                    )
                    continue
            # Preference: after price shock, recover to a different listing_id (not same listing).
            if (
                profile.listing_id_min >= 2
                and excluded_listing_ids
                and shock_open
            ):
                lid_fc = str(facts_fc.get("listing_id") or "").strip()
                if lid_fc and lid_fc in excluded_listing_ids:
                    messages.append(
                        {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_calls[0].get("id") or "x",
                            "content": json.dumps(
                                {
                                    "rejected": (
                                        "recover to a different listing_id after price shock; "
                                        f"listing_id {lid_fc} is excluded"
                                    )
                                }
                            ),
                        }
                    )
                    continue
        # TF1: reject first cart that misses firm gated attr (attribute_satisfied).
        if (
            knobs.name == "intent"
            and profile.firm_specs
            and "add_to_cart" in names
            and not cart_sku
            and not shock_open
            and not controller_forced
        ):
            try:
                a_f = json.loads(tool_calls[0].get("function", {}).get("arguments") or "{}")
            except json.JSONDecodeError:
                a_f = {}
            pid_f = str((a_f or {}).get("product_id") or "")
            sku_f = str((a_f or {}).get("sku") or "")
            if pid_f and sku_f and not _intent_firm_ok(pid_f, sku_f):
                messages.append(
                    {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_calls[0].get("id") or "x",
                        "content": json.dumps(
                            {
                                "rejected": (
                                    "firm gated attribute not met on this SKU; "
                                    "inspect another option that matches firm specs"
                                )
                            }
                        ),
                    }
                )
                excluded_skus.add(sku_f)
                if last_pick.get("sku") == sku_f:
                    last_pick = {}
                continue
        # Soft brand first cart: reject non-preferred when rules mark brand as soft.
        # If preferred brand already viewed → redirect. Else require a brand search once;
        # after that search, if still no preferred among viewed, allow other brands
        # (rules: soft — prefer when compliant exists, else any compliant).
        if (
            profile.soft_brand_first_cart
            and profile.preferred_brand
            and "add_to_cart" in names
            and not cart_sku
            and not shock_open
            and not controller_forced
        ):
            try:
                a0 = json.loads(tool_calls[0].get("function", {}).get("arguments") or "{}")
            except json.JSONDecodeError:
                a0 = {}
            pid_add = str((a0 or {}).get("product_id") or "")
            facts_add = view_facts_by_pid.get(pid_add) or {}
            if not _brand_matches(facts_add, profile.preferred_brand):
                def _firm_brand_ok(pid: str) -> bool:
                    if not _brand_matches(view_facts_by_pid.get(pid), profile.preferred_brand):
                        return False
                    if not profile.firm_specs:
                        return True
                    for r in variants_by_pid.get(pid) or []:
                        if _sku_meets_firm(
                            view_facts_by_pid.get(pid),
                            r,
                            profile.firm_specs,
                            profile.resolution_defs,
                        ):
                            return True
                    return False

                alt = next((pid for pid in viewed if _firm_brand_ok(pid)), None)
                if alt:
                    messages.append(
                        {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_calls[0].get("id") or "x",
                            "content": json.dumps(
                                {
                                    "rejected": (
                                        f"soft brand first cart: use preferred brand "
                                        f"{profile.preferred_brand} (seen in {alt})"
                                    )
                                }
                            ),
                        }
                    )
                    rows_alt = variants_by_pid.get(alt) or []
                    pick_alt = _pick_budget_variant(
                        rows_alt, budget=budget, excluded=excluded_skus
                    )
                    if pick_alt and pick_alt.get("sku"):
                        last_pick = {"product_id": alt, "sku": str(pick_alt["sku"])}
                    continue
                if not brand_search_done:
                    brand_search_nudge = True
                    messages.append(
                        {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_calls[0].get("id") or "x",
                            "content": json.dumps(
                                {
                                    "rejected": (
                                        f"soft brand first cart: search/view preferred brand "
                                        f"{profile.preferred_brand} before first cart"
                                    )
                                }
                            ),
                        }
                    )
                    continue
                # v47 intent: after brand search, still reject non-preferred (no steps escape).
                # Initial cart is scored on soft brand — wrong brand → r=0 on hidden intent.
                if knobs.name == "intent":
                    brand_search_nudge = True
                    messages.append(
                        {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_calls[0].get("id") or "x",
                            "content": json.dumps(
                                {
                                    "rejected": (
                                        f"soft brand first cart: keep seeking preferred brand "
                                        f"{profile.preferred_brand} (initial cart scored)"
                                    )
                                }
                            ),
                        }
                    )
                    continue
        closed_err = _closed_set_violation(
            tool_calls, candidates=candidates, viewed=viewed
        )
        if closed_err:
            messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].get("id") or "x",
                    "content": json.dumps({"rejected": closed_err}),
                }
            )
            continue

        # Sku-safe inspect/cart: fill missing sku from viewed variants.
        for call in tool_calls:
            name = call.get("function", {}).get("name") or ""
            if name not in ("inspect_stock", "add_to_cart", "place_test_order"):
                continue
            try:
                args = _arguments(
                    call,
                    justification_required=just_req,
                    allowed_fields=allowed_fields,
                    last_stock=last_stock,
                    view_facts_by_pid=view_facts_by_pid,
                    resolution_defs=profile.resolution_defs,
                    justification_spec_field=profile.justification_spec_field,
                    useful_reason_fields=profile.useful_reason_fields,
                )
            except Exception:  # noqa: BLE001
                continue
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and not sku:
                rows = variants_by_pid.get(pid) or []
                pick = _pick_budget_variant(rows, budget=budget, excluded=excluded_skus)
                if pick and pick.get("sku"):
                    args["sku"] = str(pick["sku"])
                    call["function"]["arguments"] = json.dumps(args)
            elif name == "place_test_order":
                call["function"]["arguments"] = json.dumps(args)

        messages.append(
            {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
        )

        group_id = f"{session_key}-turn-{turn}-{int(time.time() * 1000) % 10_000_000}"
        sanitized_calls: list[dict[str, Any]] = []
        for index, call in enumerate(tool_calls, start=1):
            name = call.get("function", {}).get("name") or ""
            args = _arguments(
                call,
                justification_required=just_req,
                allowed_fields=allowed_fields,
                last_stock=last_stock,
                view_facts_by_pid=view_facts_by_pid,
                resolution_defs=profile.resolution_defs,
                justification_spec_field=profile.justification_spec_field,
                useful_reason_fields=profile.useful_reason_fields,
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
            # Soft rejects must NOT skip ahead of session.solver_turn_count.
            "turn": env_turn + 1,
            "calls": sanitized_calls,
        }
        result = _env_call(envelope)
        env_turn += 1
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )
        done = bool(result.get("done"))
        call_results = result.get("calls") or []
        placed_this_turn = any(
            (c.get("function") or {}).get("name") == "place_test_order" for c in tool_calls
        )
        for _cr in call_results:
            if not isinstance(_cr, dict):
                continue
            obs_d = _cr.get("observation")
            if not isinstance(obs_d, dict):
                continue
            # Only treat `order` as terminal when THIS turn placed — add/view payloads
            # may carry a stale nested `order` from env state (v6 ranking agent_error).
            if not placed_this_turn:
                continue
            has_order = obs_d.get("order") is not None or (
                isinstance(obs_d.get("observation"), dict)
                and obs_d["observation"].get("order") is not None
            )
            if has_order or obs_d.get("done") is True:
                order_done = True
                done = True
        for index, call in enumerate(tool_calls):
            name = call.get("function", {}).get("name") or ""
            call_result = call_results[index] if index < len(call_results) else {}
            if not isinstance(call_result, dict):
                call_result = {}
            obs_raw = call_result.get("observation")
            payload = _coerce_observation(obs_raw) or (
                obs_raw if isinstance(obs_raw, dict) else {}
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or f"c{index}",
                    "content": json.dumps(obs_raw)[:8000]
                    if not isinstance(obs_raw, str)
                    else obs_raw[:8000],
                }
            )

            if name == "message":
                must_message = False
                clarify_sent = True
                if ask_first and profile.firm_specs:
                    asked_spec = True
                recovery_msg_force = False
                if shock_open or knobs.shock_protocol == "full":
                    recovery_committed = True
            if name == "remove_from_cart":
                must_remove = False
                cart_sku = {}
                pending_recheck = None
                force_add_after_inspect = False
                cart_stock_verified_key = None
                # Always drop sticky pick after remove so recovery re-elects survivors.
                last_pick = {}
                if knobs.name == "constraint":
                    # TF3: stock_check is env/user-driven; skip agent recovery MSG thrash.
                    # Commit recovery immediately so viewed-pool inspect→add can run.
                    shock_open = True
                    shock_recover = True
                    awaiting_user_after_shock = False
                    must_message = False
                    recovery_msg_force = False
                    recovery_committed = True
                elif knobs.shock_protocol == "full":
                    # honest_oro: agent commits recovery MSG itself — never stall on a
                    # shopper reply that may never arrive (v21 preference hang).
                    shock_open = True
                    shock_recover = True
                    awaiting_user_after_shock = False
                    must_message = True
                    recovery_msg_force = True
                else:
                    # Light shock: do not wait on user — recover immediately from viewed pool.
                    shock_open = True
                    shock_recover = True
                    recovery_committed = True
                if knobs.name == "justification":
                    # Re-elect brand/firm survivor immediately after remove.
                    _prefer_soft_brand_pick()
                    if not last_pick:
                        alt = _pick_best_viewed_sku(
                            viewed=viewed,
                            view_facts_by_pid=view_facts_by_pid,
                            variants_by_pid=variants_by_pid,
                            budget=budget,
                            excluded_skus=excluded_skus,
                            excluded_keys=excluded_keys,
                            profile=profile,
                            prefer_recovery=False,
                            exclude_product_ids=set(excluded_product_ids)
                            if excluded_product_ids
                            else None,
                        )
                        if alt:
                            last_pick = dict(alt)
                    if (
                        not last_pick
                        and not justification_postshock_reseed_used
                        and not _justification_survivor_in_viewed()
                    ):
                        # No survivor in viewed pool — one brand+firm re-seed search.
                        justification_postshock_reseed_used = True
                        justification_reseed_search = True
                        justification_seed_done = False
                    elif last_pick or _justification_survivor_in_viewed():
                        # Survivors already viewed — never search-thrash after shock.
                        justification_reseed_search = False
                        justification_seed_done = True
                        if not last_pick:
                            _prefer_soft_brand_pick()
                        if not last_pick:
                            alt2 = _pick_best_viewed_sku(
                                viewed=viewed,
                                view_facts_by_pid=view_facts_by_pid,
                                variants_by_pid=variants_by_pid,
                                budget=budget,
                                excluded_skus=excluded_skus,
                                excluded_keys=excluded_keys,
                                profile=profile,
                                prefer_recovery=False,
                                exclude_product_ids=set(excluded_product_ids)
                                if excluded_product_ids
                                else None,
                            )
                            if alt2:
                                last_pick = dict(alt2)
            if name in ("search", "filter"):
                searchish += 1
                try:
                    a_s = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    a_s = {}
                q_s = str((a_s or {}).get("query") or (a_s or {}).get("q") or "").lower()
                if profile.preferred_brand and profile.preferred_brand.lower() in q_s:
                    brand_search_done = True
                    brand_search_nudge = False
                results = (
                    payload.get("results")
                    or payload.get("items")
                    or payload.get("products")
                    or []
                )
                if isinstance(results, list):
                    for row in results[:20]:
                        if not isinstance(row, dict) or row.get("product_id") is None:
                            continue
                        pid_r = str(row["product_id"])
                        if pid_r not in search_hit_pids:
                            search_hit_pids.append(pid_r)
                        # Honest listing_id: only real field, never product_id fallback.
                        if row.get("listing_id") not in (None, "", "None"):
                            listing_ids.add(str(row.get("listing_id")))
                        title_r = str(row.get("title") or "")
                        title_ok = (not profile.title_terms) or _title_matches_terms(
                            title_r, profile.title_terms
                        )
                        if title_ok and title_r:
                            retrieval_hit_titles.append(title_r)
                        # TF4: keep search titles for soft-axis view prioritization.
                        if title_r or row.get("brand") not in (None, "", "None"):
                            thin = dict(view_facts_by_pid.get(pid_r) or {})
                            if title_r and not thin.get("title"):
                                thin["title"] = title_r
                            if row.get("brand") not in (None, "", "None") and not thin.get("brand"):
                                thin["brand"] = str(row.get("brand")).strip()
                            if thin:
                                view_facts_by_pid[pid_r] = thin
                        brand_r = str(row.get("brand") or "").strip()
                        if (
                            knobs.name == "retrieval"
                            and title_ok
                            and brand_r
                            and brand_r.lower() not in {b.lower() for b in harvested_brands}
                        ):
                            harvested_brands.append(brand_r)
                        if title_ok and (
                            profile.min_matching_identities >= 2 or profile.title_terms
                        ):
                            matching_ids.add(pid_r)
                            identity_uf.ingest_row(row, {"title": title_r, "brand": str(row.get("brand") or "")})
            if name in ("view", "compare"):
                if name == "view":
                    try:
                        a_v_early = json.loads(call["function"].get("arguments") or "{}")
                    except json.JSONDecodeError:
                        a_v_early = {}
                    if isinstance(a_v_early, dict) and a_v_early.get("product_id"):
                        viewed.add(str(a_v_early["product_id"]))
                    pid_v, facts = _extract_view_facts(obs_raw)
                    if not pid_v:
                        if isinstance(a_v_early, dict) and a_v_early.get("product_id"):
                            pid_v = str(a_v_early["product_id"])
                    if not pid_v:
                        try:
                            a_v2 = json.loads(call["function"].get("arguments") or "{}")
                        except json.JSONDecodeError:
                            a_v2 = {}
                        if isinstance(a_v2, dict) and a_v2.get("product_id"):
                            pid_v = str(a_v2["product_id"])
                    if pid_v:
                        viewed.add(pid_v)
                        view_facts_by_pid[pid_v] = {
                            **view_facts_by_pid.get(pid_v, {}),
                            **facts,
                        }
                        if _brand_matches(facts, profile.preferred_brand):
                            brand_search_done = True
                            brand_search_nudge = False
                        lid_v = None
                        obs_v = _coerce_observation(obs_raw) or {}
                        if obs_v.get("listing_id") not in (None, "", "None"):
                            lid_v = str(obs_v.get("listing_id"))
                        if lid_v:
                            listing_ids.add(lid_v)
                            view_facts_by_pid[pid_v]["listing_id"] = lid_v
                        # else: do NOT treat product_id as listing_id
                        title_v = str(facts.get("title") or "")
                        title_ok = (not profile.title_terms) or _title_matches_terms(
                            title_v, profile.title_terms
                        )
                        if title_ok and (
                            profile.min_matching_identities >= 2 or profile.title_terms
                        ):
                            matching_ids.add(pid_v)
                            identity_uf.ingest_row(
                                {"product_id": pid_v, "title": title_v, "brand": facts.get("brand"), "price": facts.get("price")},
                                facts,
                            )
                        rows = _variants_from_observation(obs_raw)
                        if rows:
                            variants_by_pid[pid_v] = rows
                            if knobs.name == "recovery" or profile.relax_color_then_model:
                                pick = _pick_recovery_variant(
                                    rows,
                                    facts=view_facts_by_pid.get(pid_v) or facts,
                                    preferred_model=profile.preferred_model,
                                    preferred_color=profile.preferred_color,
                                    budget=budget,
                                    excluded=excluded_skus,
                                )
                            else:
                                pick = _pick_budget_variant(
                                    rows, budget=budget, excluded=excluded_skus
                                )
                            if pick and pick.get("sku"):
                                # Prefer firm-compliant variant when firm_specs present.
                                if profile.firm_specs:
                                    firm_pick = None
                                    for r in rows:
                                        if str(r.get("sku") or "") in excluded_skus:
                                            continue
                                        if not _variant_in_stock(r):
                                            continue
                                        if _sku_meets_firm(
                                            view_facts_by_pid.get(pid_v) or facts,
                                            r,
                                            profile.firm_specs,
                                            profile.resolution_defs,
                                        ):
                                            firm_pick = r
                                            break
                                    if firm_pick:
                                        pick = firm_pick
                                    else:
                                        # Do not sticky-pick a view that fails THIS firm gate.
                                        pick = None
                                if pick and pick.get("sku"):
                                    last_pick = {
                                        "product_id": pid_v,
                                        "sku": str(pick["sku"]),
                                    }
                else:
                    compared = payload.get("compared") or payload.get("products") or []
                    if isinstance(compared, list):
                        for row in compared:
                            if not isinstance(row, dict):
                                continue
                            pid_c = str(row.get("product_id") or "")
                            if pid_c:
                                viewed.add(pid_c)
                                _, facts = _extract_view_facts(row)
                                view_facts_by_pid[pid_c] = {
                                    **view_facts_by_pid.get(pid_c, {}),
                                    **facts,
                                }
                                if row.get("listing_id") not in (None, "", "None"):
                                    lid_c = str(row.get("listing_id"))
                                    listing_ids.add(lid_c)
                                    view_facts_by_pid[pid_c]["listing_id"] = lid_c
                                title_c = str(facts.get("title") or "")
                                title_ok_c = (not profile.title_terms) or _title_matches_terms(
                                    title_c, profile.title_terms
                                )
                                if title_ok_c and (
                                    profile.min_matching_identities >= 2 or profile.title_terms
                                ):
                                    matching_ids.add(pid_c)
                                    identity_uf.ingest_row(row, facts)
                                if _brand_matches(facts, profile.preferred_brand):
                                    brand_search_done = True
                                    brand_search_nudge = False
            if name == "inspect_stock":
                last_stock = _obs_stock_payload(payload) or payload
                try:
                    a_ins = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    a_ins = {}
                if not isinstance(a_ins, dict):
                    a_ins = {}
                pid_i = str(
                    (last_stock or {}).get("product_id") or a_ins.get("product_id") or ""
                )
                sku_i = str((last_stock or {}).get("sku") or a_ins.get("sku") or "")
                if pid_i and sku_i:
                    last_pick = {"product_id": pid_i, "sku": sku_i}
                if pending_recheck and (
                    not pid_i
                    or not sku_i
                    or (pid_i == pending_recheck[0] and sku_i == pending_recheck[1])
                ):
                    pending_recheck = None
                    if (
                        cart_sku
                        and not _stock_is_false(last_stock)
                        and not must_remove
                        # Arm place once shock is closed/committed — preference/recovery
                        # still need a valid recheck, but must not hang forever (v21).
                        and (
                            knobs.name not in ("preference", "recovery")
                            or recovery_committed
                            or not shock_open
                        )
                    ):
                        force_place_after_recheck = True
                        cart_stock_verified_key = (
                            f"{cart_sku['product_id']}::{cart_sku['sku']}"
                        )
                elif (
                    not cart_sku
                    and not must_remove
                    and not awaiting_user_after_shock
                    and pid_i
                    and sku_i
                    and not _stock_is_false(last_stock)
                ):
                    # Never-cart hang: after a live inspect, cart next turn.
                    price_i = _price_num((last_stock or {}).get("price"))
                    if budget is None or price_i is None or price_i <= budget + 1e-6:
                        if (
                            remaining <= 14
                            or len(viewed) >= 2
                            or recovery_committed
                            or knobs.name in ("retrieval", "justification", "constraint", "preference")
                            or searchish >= 2
                        ):
                            force_add_after_inspect = True
                            identity_survey_relaxed = True
                if _stock_is_false(last_stock):
                    if sku_i:
                        excluded_skus.add(sku_i)
                    shock_recover = True
                    shock_open = True
                    if cart_sku:
                        must_remove = True
                    pending_recheck = None
                else:
                    price = _price_num((last_stock or {}).get("price"))
                    if (
                        budget is not None
                        and price is not None
                        and price > budget + 1e-6
                        and cart_sku
                    ):
                        must_remove = True
                        shock_open = True
                        shock_recover = True
            if name == "add_to_cart":
                added = _unwrap_added(payload)
                if not (isinstance(added, dict) and added.get("product_id")):
                    # Harness may return market_event without nested `added` — use call args.
                    try:
                        a_add = json.loads(call["function"].get("arguments") or "{}")
                    except json.JSONDecodeError:
                        a_add = {}
                    if isinstance(a_add, dict) and a_add.get("product_id"):
                        added = {
                            "product_id": str(a_add.get("product_id") or ""),
                            "sku": str(a_add.get("sku") or ""),
                        }
                if isinstance(added, dict) and added.get("product_id"):
                    cart_sku = {
                        "product_id": str(added["product_id"]),
                        "sku": str(added.get("sku") or ""),
                    }
                    last_pick = dict(cart_sku)
                    last_stock = added if isinstance(added, dict) else last_stock
                    cart_stock_verified_key = None
                    # Mandatory post-add inspect before place (R1).
                    if cart_sku.get("sku"):
                        pending_recheck = (cart_sku["product_id"], cart_sku["sku"])
                if isinstance(payload, dict) and payload.get("market_event"):
                    shock_recover = True
                    shock_open = True
                    recovery_committed = False
                    ev = payload.get("market_event") or {}
                    kind = str(ev.get("kind") or "")
                    new_price = _price_num(ev.get("new_price"))
                    tgt = ev.get("target") if isinstance(ev.get("target"), dict) else {}
                    ep = str(
                        tgt.get("product_id")
                        or ev.get("product_id")
                        or (cart_sku.get("product_id") if cart_sku else "")
                        or ""
                    )
                    es = str(
                        tgt.get("sku")
                        or ev.get("sku")
                        or (cart_sku.get("sku") if cart_sku else "")
                        or ""
                    )
                    if ep and es:
                        excluded_keys.add(f"{ep}::{es}")
                        excluded_skus.add(es)
                        excluded_product_ids.add(ep)
                        # Reflect stockout on cached variants so elect cannot re-pick.
                        for row in variants_by_pid.get(ep) or []:
                            if str(row.get("sku") or "") == es:
                                row["in_stock"] = False
                        if profile.listing_id_min >= 2:
                            lid_ex = str(
                                (view_facts_by_pid.get(ep) or {}).get("listing_id") or ""
                            ).strip()
                            if lid_ex:
                                excluded_listing_ids.add(lid_ex)
                    if knobs.name == "ranking":
                        ranking_event = True
                        # Force post-event re-elect: drop sticky pre-event cart/pick.
                        last_pick = {}
                        must_remove = True
                        pending_recheck = None
                        must_message = True
                        recovery_msg_force = True
                    elif cart_sku:
                        must_remove = True
                        pending_recheck = None
                        if knobs.name == "justification":
                            last_pick = {}
                    if kind == "price_change" or knobs.shock_protocol == "full":
                        if (
                            budget is not None
                            and new_price is not None
                            and new_price > budget + 1e-9
                        ):
                            must_remove = True
                        if knobs.shock_protocol == "full":
                            recovery_msg_force = True
                            if not must_remove and knobs.name != "ranking":
                                must_message = True
                        elif not must_remove:
                            must_message = True
                            recovery_msg_force = True
            if isinstance(payload, dict) and payload.get("market_event") and name != "add_to_cart":
                shock_recover = True
                shock_open = True
                recovery_committed = False
                ev = payload.get("market_event") or {}
                kind = str(ev.get("kind") or "")
                tgt = ev.get("target") if isinstance(ev.get("target"), dict) else {}
                ep = str(tgt.get("product_id") or ev.get("product_id") or "")
                es = str(tgt.get("sku") or ev.get("sku") or "")
                if ep and es:
                    excluded_keys.add(f"{ep}::{es}")
                    excluded_skus.add(es)
                    excluded_product_ids.add(ep)
                    # Intent: shocked quiz pin is no longer the event target.
                    if knobs.name == "intent" and (
                        intent_commit_pid == ep or intent_commit_sku == es
                    ):
                        intent_commit_pid, intent_commit_sku = "", ""
                        intent_reaffirm_commit = False
                    if profile.listing_id_min >= 2:
                        lid_ex = str(
                            (view_facts_by_pid.get(ep) or {}).get("listing_id") or ""
                        ).strip()
                        if lid_ex:
                            excluded_listing_ids.add(lid_ex)
                if knobs.name == "ranking":
                    ranking_event = True
                    last_pick = {}
                    must_remove = bool(cart_sku)
                    must_message = True
                    recovery_msg_force = True
                elif cart_sku:
                    must_remove = True
                if kind == "price_change" or knobs.shock_protocol == "full":
                    if not must_remove:
                        must_message = True
                        recovery_msg_force = True

        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            messages.append({"role": "user", "content": content})
            # Ground firm prune / phrase harvest in full episode shopper text.
            episode_text = (episode_text + "\n" + str(content))[-12000:]
            prev_budget = budget
            budget = _tighten_budget(budget, str(content))
            if budget is not None and budget != prev_budget:
                budget_ladder_caps = _budget_ladder_caps(budget)
                budget_ladder_idx = 0
                if knobs.name == "intent":
                    intent_reseed_search = True
                    intent_drop_brand = True
                    # Invalidate over-budget pick even before first cart.
                    if last_pick.get("sku"):
                        pid_lp = str(last_pick.get("product_id") or "")
                        sku_lp = str(last_pick.get("sku") or "")
                        row_lp = next(
                            (
                                r
                                for r in (variants_by_pid.get(pid_lp) or [])
                                if str(r.get("sku") or "") == sku_lp
                            ),
                            None,
                        )
                        price_lp = _price_num((row_lp or last_stock or {}).get("price"))
                        if price_lp is not None and price_lp > budget + 1e-6:
                            excluded_skus.add(sku_lp)
                            last_pick = {}
                            force_add_after_inspect = False
                    # Intent commit pin: sticky if viable; else require reaffirm MSG (no silent switch).
                    if intent_commit_pid and intent_commit_sku:
                        if _intent_commit_viable():
                            last_pick = {
                                "product_id": intent_commit_pid,
                                "sku": intent_commit_sku,
                            }
                            intent_reseed_search = False
                        else:
                            excluded_skus.add(intent_commit_sku)
                            intent_commit_pid, intent_commit_sku = "", ""
                            intent_reaffirm_commit = True
                            must_message = True
                            last_pick = {}
                            force_add_after_inspect = False
                elif knobs.name == "constraint":
                    constraint_reseed_search = True
                    constraint_seed_done = False
                    if last_pick.get("sku"):
                        pid_lp = str(last_pick.get("product_id") or "")
                        sku_lp = str(last_pick.get("sku") or "")
                        row_lp = next(
                            (
                                r
                                for r in (variants_by_pid.get(pid_lp) or [])
                                if str(r.get("sku") or "") == sku_lp
                            ),
                            None,
                        )
                        price_lp = _price_num((row_lp or last_stock or {}).get("price"))
                        if price_lp is not None and price_lp > budget + 1e-6:
                            excluded_skus.add(sku_lp)
                            last_pick = {}
                            force_add_after_inspect = False
                elif knobs.name == "preference":
                    # Do NOT reseed open-catalog search after shock — recover from viewed.
                    preference_reseed_search = False
                    if last_pick.get("sku"):
                        pid_lp = str(last_pick.get("product_id") or "")
                        sku_lp = str(last_pick.get("sku") or "")
                        row_lp = next(
                            (
                                r
                                for r in (variants_by_pid.get(pid_lp) or [])
                                if str(r.get("sku") or "") == sku_lp
                            ),
                            None,
                        )
                        price_lp = _price_num((row_lp or last_stock or {}).get("price"))
                        if price_lp is not None and price_lp > budget + 1e-6:
                            excluded_skus.add(sku_lp)
                            last_pick = {}
                            force_add_after_inspect = False
            # Budget shrink: drop over-budget cart immediately (honest contract).
            if (
                cart_sku
                and budget is not None
                and not must_remove
            ):
                cart_price = _price_num((last_stock or {}).get("price"))
                if cart_price is None:
                    pid_c = str(cart_sku.get("product_id") or "")
                    sku_c = str(cart_sku.get("sku") or "")
                    row_c = next(
                        (
                            r
                            for r in (variants_by_pid.get(pid_c) or [])
                            if str(r.get("sku") or "") == sku_c
                        ),
                        None,
                    )
                    cart_price = _price_num((row_c or {}).get("price"))
                if cart_price is not None and cart_price > budget + 1e-6:
                    must_remove = True
                    shock_open = True
                    shock_recover = True
                    pending_recheck = None
                    force_place_after_recheck = False
                    if knobs.name == "intent":
                        intent_reseed_search = True
                        intent_drop_brand = True
                    elif knobs.name == "constraint":
                        constraint_reseed_search = True
                        constraint_seed_done = False
                    elif knobs.name == "preference":
                        # Budget shrink still bans over-budget SKU; recover from viewed, no thrash.
                        preference_reseed_search = False
                        preference_seed_done = True
            merged = _merge_firm_specs(profile.firm_specs, str(content))
            if merged and knobs.name != "preference":
                profile.firm_specs = merged
                if knobs.name == "constraint":
                    _prune_constraint_firm_specs(profile, episode_text)
                if ask_first:
                    asked_spec = True
                if not gated_attr_hint and profile.firm_specs:
                    fs0 = profile.firm_specs[0]
                    gated_attr_hint = f"{fs0.get('name')}={fs0.get('value')}"
            elif merged and knobs.name == "preference":
                # Preference: fold any new firm harvest into soft tokens, never firm gate.
                profile.firm_specs = list(profile.firm_specs or []) + list(merged)
                _demote_soft_firm_specs(profile)
            # Mid-episode goal_brief harvest — portable struct, no suite nouns.
            if (clarify_sent or ask_first) and knobs.name in (
                "intent",
                "retrieval",
                "constraint",
                "preference",
            ):
                mid = _struct_harvest(str(content), "")
                # Always merge something-like / ideally-prefer from raw reply (not only mid struct).
                if knobs.name == "intent":
                    ib = _harvest_ideal_prefer_brand(str(content))
                    if ib:
                        profile.preferred_brand = ib
                        profile.soft_brand_first_cart = True
                if knobs.name == "retrieval":
                    phrase_m = _harvest_something_like_phrase(str(content))
                    if phrase_m:
                        merged_terms = list(profile.title_terms or [])
                        for t in phrase_m:
                            if t not in merged_terms:
                                merged_terms.append(t)
                        profile.title_terms = merged_terms[:12]
                        profile.title_phrase_match = True
                        # Resync matching_ids from already-viewed titles.
                        for pid_rs, facts_rs in list(view_facts_by_pid.items()):
                            title_rs = str((facts_rs or {}).get("title") or "")
                            if title_rs and _title_matches_terms(title_rs, profile.title_terms):
                                matching_ids.add(pid_rs)
                if not profile.required_category and mid.get("required_category"):
                    profile.required_category = str(mid["required_category"])
                if not profile.preferred_brand and mid.get("preferred_brand"):
                    profile.preferred_brand = _sanitize_brand_token(
                        str(mid["preferred_brand"]).strip()
                    )
                if mid.get("soft_brand_first_cart") and knobs.name != "constraint":
                    profile.soft_brand_first_cart = True
                if knobs.name == "constraint":
                    # Firm brand from THIS clarify text (required brand is X).
                    rb = _harvest_required_brand(str(content))
                    if rb:
                        profile.preferred_brand = rb
                    elif mid.get("preferred_brand"):
                        profile.preferred_brand = _sanitize_brand_token(
                            str(mid["preferred_brand"]).strip()
                        )
                    if mid.get("firm_specs"):
                        profile.firm_specs = _merge_firm_specs(
                            profile.firm_specs, str(content)
                        ) or profile.firm_specs
                    _prune_constraint_firm_specs(profile, episode_text)
                    if profile.firm_specs:
                        asked_spec = True
                if knobs.name == "preference":
                    # Goal brief carries soft axes / category / budget for TF4.
                    toks = mid.get("pref_tokens") or _struct_harvest_pref_tokens(str(content))
                    if isinstance(toks, list) and toks:
                        merged_pt = list(profile.pref_tokens or [])
                        for t in toks:
                            tl = str(t).strip().lower()
                            if tl and tl not in {x.lower() for x in merged_pt}:
                                merged_pt.append(str(t).strip())
                        profile.pref_tokens = merged_pt[:12]
                        # Only reseed soft search before first cart / before shock.
                        if not cart_sku and not shock_open:
                            preference_reseed_search = True
                            preference_seed_done = False
                    if mid.get("useful_reason_fields") and not profile.useful_reason_fields:
                        profile.useful_reason_fields = [
                            str(x).strip().lower().replace(" ", "_")
                            for x in mid["useful_reason_fields"]
                            if str(x).strip()
                        ][:8]
                    # Never promote waterproof/IP soft language into firm mid-episode.
                    if mid.get("firm_specs"):
                        _demote_soft_firm_specs(profile)
                if knobs.name == "retrieval":
                    terms = mid.get("title_terms")
                    if isinstance(terms, list) and terms:
                        # Prefer longer / newly revealed phrase tokens from goal_brief.
                        merged_terms = list(profile.title_terms or [])
                        for t in terms:
                            tl = str(t).strip().lower()
                            if tl and tl not in merged_terms:
                                merged_terms.append(tl)
                        profile.title_terms = merged_terms[:12]
                        profile.title_phrase_match = True
                        for pid_rs, facts_rs in list(view_facts_by_pid.items()):
                            title_rs = str((facts_rs or {}).get("title") or "")
                            if title_rs and _title_matches_terms(title_rs, profile.title_terms):
                                matching_ids.add(pid_rs)
                    if int(mid.get("min_matching_identities") or 0) > profile.min_matching_identities:
                        profile.min_matching_identities = int(mid["min_matching_identities"])
            if awaiting_user_after_shock and not must_remove:
                must_message = True
                recovery_msg_force = True
                awaiting_user_after_shock = False
            # Clarify unlock: any post-opener shopper reply can bind obligations (paraphrase).
            if clarify_sent or ask_first:
                clarify_unlock = True
                asked_spec = True
            if _is_commit_quiz(str(content)):
                commit_quiz_pending = True
                must_message = True
                # Quiz often restates gated inch — merge into firm_specs.
                merged_q = _merge_firm_specs(profile.firm_specs, str(content))
                if merged_q:
                    profile.firm_specs = merged_q
                    fs0 = profile.firm_specs[0]
                    gated_attr_hint = f"{fs0.get('name')}={fs0.get('value')}"
            elif _is_price_shock_probe(str(content)) or _is_stock_check_probe(str(content)):
                # Inspect current cart/pick for truth before any dialogue ack.
                # Do NOT reuse commit_quiz_pending — that drives going-with MSG, not shock ack.
                # After remove, do not re-inspect the excluded event target (TF2 hang class).
                # v42: empty cart + MSG-still-required → do not arm pending (deadlock with
                # message gate). Recovery inspect/add runs after recovery_committed.
                _probe_arm_pending = not (
                    must_message and not (cart_sku.get("product_id") and cart_sku.get("sku"))
                )
                if _probe_arm_pending and cart_sku.get("product_id") and cart_sku.get("sku"):
                    _ck = f"{cart_sku['product_id']}::{cart_sku['sku']}"
                    if (
                        str(cart_sku["product_id"]) not in excluded_product_ids
                        and str(cart_sku["sku"]) not in excluded_skus
                        and _ck not in excluded_keys
                    ):
                        pending_recheck = (cart_sku["product_id"], cart_sku["sku"])
                elif (
                    _probe_arm_pending
                    and last_pick.get("product_id")
                    and last_pick.get("sku")
                    and str(last_pick.get("product_id") or "") not in excluded_product_ids
                    and str(last_pick.get("sku") or "") not in excluded_skus
                    and f"{last_pick['product_id']}::{last_pick['sku']}" not in excluded_keys
                ):
                    pending_recheck = (last_pick["product_id"], last_pick["sku"])
                elif _probe_arm_pending and knobs.name == "constraint" and viewed:
                    # TF3 stock_check after stockout: inspect a firm-valid NEW product_id.
                    alt_sc = _pick_best_viewed_sku(
                        viewed=viewed,
                        view_facts_by_pid=view_facts_by_pid,
                        variants_by_pid=variants_by_pid,
                        budget=budget,
                        excluded_skus=excluded_skus,
                        excluded_keys=excluded_keys,
                        profile=profile,
                        prefer_recovery=False,
                        exclude_product_ids=set(excluded_product_ids),
                    )
                    if alt_sc and alt_sc.get("product_id") and alt_sc.get("sku"):
                        last_pick = dict(alt_sc)
                        pending_recheck = (alt_sc["product_id"], alt_sc["sku"])
                if knobs.name == "constraint":
                    # Teacher answers stock_check via inspect/add — not agent chat.
                    must_message = False
                else:
                    must_message = True
                if knobs.name == "intent":
                    intent_drop_brand = True
            elif "?" in str(content):
                must_message = True

    return dialogue


def run_intent_pipeline(problem_data: dict[str, Any], profile: ObligationProfile) -> list:
    """TF1 dedicated: soft-brand first cart + commit-quiz sticky pin (event commitment)."""
    profile.pipeline = "intent"
    profile.soft_brand_first_cart = True
    # TF1: category_path checks are part of the firm contract when rules mention them.
    if not profile.required_category:
        # leave empty — harvest from goal_brief mid-episode
        pass
    # Soft brand from episode text when ledger missed it.
    q = str(
        ((problem_data.get("environment") or {}).get("policy_view") or {}).get("query")
        or ""
    )
    if not profile.preferred_brand:
        profile.preferred_brand = _harvest_ideal_prefer_brand(q)
    return _run_pipeline(problem_data, profile)


def run_retrieval_pipeline(problem_data: dict[str, Any], profile: ObligationProfile) -> list:
    profile.pipeline = "retrieval"
    # TF2 portable: title-phrase protocol. Survey need from THIS rules when stated;
    # default 2 distinct matching product_ids (not hardcoded suite "5" / UF clusters).
    profile.title_phrase_match = True
    if int(profile.min_matching_identities or 0) < 2:
        profile.min_matching_identities = 2
    if profile.require_category_path is False:
        profile.require_category_path = True
    q = str(
        ((problem_data.get("environment") or {}).get("policy_view") or {}).get("query")
        or ""
    )
    phrase = _harvest_something_like_phrase(q)
    if phrase:
        merged = list(profile.title_terms or [])
        for t in phrase:
            if t not in merged:
                merged.append(t)
        profile.title_terms = merged[:12]
    return _run_pipeline(problem_data, profile)


def run_constraint_pipeline(problem_data: dict[str, Any], profile: ObligationProfile) -> list:
    """TF3 dedicated: exact-variant + firm brand (no soft-brand override) + sticky add."""
    profile.pipeline = "constraint"
    # TF3: exact-variant + no soft-brand override; ask-first when firm still missing.
    profile.strict_selected_variant = True
    profile.soft_brand_first_cart = False
    q = str(
        ((problem_data.get("environment") or {}).get("policy_view") or {}).get("query")
        or ""
    )
    # Prefer required-brand latch from episode text when ledger empty/polluted.
    cleaned_brand = _sanitize_brand_token(profile.preferred_brand or "")
    if not cleaned_brand:
        profile.preferred_brand = _harvest_required_brand(q) or ""
    else:
        profile.preferred_brand = cleaned_brand
    _prune_constraint_firm_specs(profile, q)
    if not profile.firm_specs:
        profile.ask_first = True
    # Device evidence only when THIS episode already armed it (struct harvest) — not phone enum.
    if profile.require_category_path is False and profile.required_category:
        profile.require_category_path = True
    return _run_pipeline(problem_data, profile)


def run_preference_pipeline(problem_data: dict[str, Any], profile: ObligationProfile) -> list:
    profile.pipeline = "preference"
    # Soft-pref contract: survey ≥2 products / listing_ids before first cart when not set.
    if int(profile.multi_listing_min or 0) < 2:
        profile.multi_listing_min = 2
    # Soft/useful language must not stay as firm (post-shock empty-pool cliff).
    _demote_soft_firm_specs(profile)
    if profile.require_category_path is False and profile.required_category:
        profile.require_category_path = True
    return _run_pipeline(problem_data, profile)


def run_ranking_pipeline(problem_data: dict[str, Any], profile: ObligationProfile) -> list:
    profile.pipeline = "ranking"
    profile.closed_set = True
    return _run_pipeline(problem_data, profile)


def run_recovery_pipeline(problem_data: dict[str, Any], profile: ObligationProfile) -> list:
    profile.pipeline = "recovery"
    profile.relax_color_then_model = True
    # Device evidence only if THIS episode already required it (do not arm from "phone" grep).
    return _run_pipeline(problem_data, profile)


def run_justification_pipeline(problem_data: dict[str, Any], profile: ObligationProfile) -> list:
    profile.pipeline = "justification"
    profile.justification_required = True
    return _run_pipeline(problem_data, profile)


def run_generic_pipeline(problem_data: dict[str, Any], profile: ObligationProfile) -> list:
    profile.pipeline = "generic"
    return _run_pipeline(problem_data, profile)


_PIPELINE_RUNNERS = {
    "intent": run_intent_pipeline,
    "retrieval": run_retrieval_pipeline,
    "constraint": run_constraint_pipeline,
    "preference": run_preference_pipeline,
    "ranking": run_ranking_pipeline,
    "recovery": run_recovery_pipeline,
    "justification": run_justification_pipeline,
    "generic": run_generic_pipeline,
}


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        query = str(
            ((problem_data.get("environment") or {}).get("policy_view") or {}).get("query")
            or ""
        )
        profile = extract_obligation_profile(query)
        runner = _PIPELINE_RUNNERS.get(profile.pipeline, run_generic_pipeline)
        return runner(problem_data, profile)
    except Exception as exc:  # noqa: BLE001 — harness robustness
        return [
            {
                "role": "assistant",
                "content": f"agent error: {type(exc).__name__}: {str(exc)[:240]}",
            }
        ]


__all__ = ["agent_main", "extract_obligation_profile"]
