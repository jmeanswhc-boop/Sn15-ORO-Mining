from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# =============================================================================
# OpenRouter-only inference
# =============================================================================

_STONE_LIGHT = "deepseek/deepseek-v3.2"
_STONE_DENSE = "moonshotai/kimi-k2.6"
_STONE_BACKUP = (
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "google/gemma-4-31b-it",
)

_TEE_MAP = {
    "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
    "deepseek-ai/DeepSeek-V3.1-TEE": "deepseek/deepseek-chat-v3.1",
    "deepseek-ai/DeepSeek-V3-0324-TEE": "deepseek/deepseek-chat-v3-0324",
    "deepseek-ai/DeepSeek-R1-0528-TEE": "deepseek/deepseek-r1-0528",
    "Qwen/Qwen3-32B-TEE": "qwen/qwen3-32b",
    "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
    "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
    "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
    "zai-org/GLM-5.1-TEE": "z-ai/glm-5.1",
    "zai-org/GLM-5.2-TEE": "z-ai/glm-5.2",
    "moonshotai/Kimi-K2.6-TEE": "moonshotai/kimi-k2.6",
    "openai/gpt-oss-120b-TEE": "openai/gpt-oss-120b",
    "XiaomiMiMo/MiMo-V2-Flash-TEE": "xiaomi/mimo-v2-flash",
    "unsloth/Mistral-Nemo-Instruct-2407-TEE": "mistralai/mistral-small-2603",
}

_proxy = ProxyClient(timeout=120, max_retries=2)
_OBS_LIMIT = 6200

# Vein identifiers (capability themes — not EnvPack Task-rules copies)
VEIN_SPLIT = "split"  # TF1 intent
VEIN_SWEEP = "sweep"  # TF2 recall
VEIN_LOCK = "lock"  # TF3 constraints
VEIN_TASTE = "taste"  # TF4 preference
VEIN_PODIUM = "podium"  # TF5 ranking
VEIN_MEND = "mend"  # TF6 recovery
VEIN_VOUCH = "vouch"  # TF7 justification

_ALL_VEINS = (
    VEIN_PODIUM,
    VEIN_MEND,
    VEIN_VOUCH,
    VEIN_LOCK,
    VEIN_SPLIT,
    VEIN_TASTE,
    VEIN_SWEEP,
)

# Tie-break priority when vein scores collide (specialised first).
_VEIN_PRIORITY = {
    VEIN_PODIUM: 7,
    VEIN_MEND: 6,
    VEIN_VOUCH: 5,
    VEIN_LOCK: 4,
    VEIN_SPLIT: 3,
    VEIN_TASTE: 2,
    VEIN_SWEEP: 1,
}

_DENSE_VEINS = frozenset(
    {VEIN_SPLIT, VEIN_LOCK, VEIN_TASTE, VEIN_PODIUM, VEIN_MEND, VEIN_VOUCH}
)

# Quarry stages
STAGE_SURVEY = "survey"
STAGE_BIND = "bind"
STAGE_SET = "set"
STAGE_SEAL = "seal"

# Secondary grit blends when #2 is close in absolute gap OR at least
# this fraction of the primary score (multi-label tasks).
_BLEND_GAP = 2
_BLEND_RATIO = 0.45

# =============================================================================
# Facet probes (generic; no suite query fragments)
# =============================================================================

_RX_VARIANT = re.compile(r"\b\d{6,}::[\w.-]+")
_RX_CEILING = re.compile(
    r"(?:at or (?:under|below)|under|below|up to|no more than|max(?:imum)?(?:\s+of)?)"
    r"\s*([\d][\d,]*(?:\.\d+)?)",
    re.I,
)
_RX_PHRASE = re.compile(r"[\"\u201c]([^\"\u201d]{3,60})[\"\u201d]")
_RX_ASK = re.compile(
    r"\bif you ask\b|\bwhen you ask\b|"
    r"(?:tell|share|reveal)[^.]{0,40}\bif\b[^.]{0,20}\bask|"
    r"ask (?:me |for )?(?:the |that )?(?:exact )?(?:requirement|spec)",
    re.I,
)
_RX_WHY = re.compile(
    r"(?:must|should) include.{0,40}justification|"
    r"structured justification|"
    r"\bwhy\b|\breason\b|\bjustif|\bexplain\b|"
    r"ground claims|trustworthy reason",
    re.I,
)
_RX_OPTIONAL_WHY = re.compile(r"justification is optional", re.I)

# Vein cue weights: (token, weight) scored in shopper goal vs operating brief.
_GOAL_CUES: dict[str, tuple[tuple[str, int], ...]] = {
    VEIN_PODIUM: (("tradeoff", 4), ("rank", 3), ("dominat", 3), ("highest", 1)),
    VEIN_MEND: (("disappear", 4), ("unavailable", 3), ("sold out", 3), ("relax", 2)),
    VEIN_VOUCH: (("why", 4), ("reason", 3), ("justif", 3), ("explain", 2)),
    VEIN_LOCK: (("if you ask", 4), ("exact", 2), ("varian", 2), ("sku", 1)),
    VEIN_SPLIT: (("prefer", 3), ("ideally", 3), ("firm", 2), ("soft", 2)),
    VEIN_TASTE: (("rather", 3), ("cheapest", 3), ("use-case", 2), ("matters", 2)),
    VEIN_SWEEP: (("matches", 3), ("phrase", 2), ("listing", 1), ("shortlist", 2)),
}

_BRIEF_CUES: dict[str, tuple[tuple[str, int], ...]] = {
    VEIN_PODIUM: (("tradeoff", 3), ("re-rank", 3), ("strictly worse", 2), ("lexico", 2)),
    VEIN_MEND: (("least-relax", 4), ("same model", 2), ("other color", 2)),
    VEIN_VOUCH: (("claim", 3), ("ground", 2), ("trustworthy", 2)),
    VEIN_LOCK: (("pixel", 3), ("exact variant", 2), ("threshold", 2)),
    VEIN_SPLIT: (("nice-to-have", 3), ("subordinate", 3), ("preferred brand", 2)),
    VEIN_TASTE: (("preference axes", 3), ("intended use", 2), ("placeholder", 1)),
    VEIN_SWEEP: (("reformulate", 3), ("shortlist", 2), ("lucky", 2)),
}

_CLAIM_ALIAS = {
    "display": "resolution",
    "display resolution": "resolution",
    "pixel class": "resolution",
    "resolution class": "resolution",
    "screen": "screen_size",
    "screen size": "screen_size",
    "size": "screen_size",
    "diagonal": "screen_size",
    "refresh": "refresh_rate",
    "refresh rate": "refresh_rate",
    "memory": "ram",
    "stock": "in_stock",
    "availability": "in_stock",
    "available": "in_stock",
    "cost": "price",
    "maker": "brand",
    "manufacturer": "brand",
}

_CLAIM_OK = frozenset(
    {
        "resolution",
        "screen_size",
        "refresh_rate",
        "ram",
        "price",
        "in_stock",
        "brand",
        "streaming",
        "microphone",
        "privacy_cover",
        "autofocus",
        "hdr",
    }
)
_CLAIM_SOFT = frozenset(
    {"streaming", "microphone", "privacy_cover", "autofocus", "hdr"}
)
_CLAIM_DROP = frozenset(
    {
        "category",
        "category_path",
        "title",
        "name",
        "summary",
        "evidence",
        "reason",
        "note",
        "description",
        "os",
        "operating system",
        "device_operating_system",
    }
)
_RX_SPEC_AS_FIELD = re.compile(
    r"(?i)^(?:\d{3,4}\s*p(?:\s*hd)?|full\s*hd|qhd|uhd|4k|hd|"
    r"\d+(?:\.\d+)?\s*-?\s*inch(?:es)?)$"
)

_EMPTY_NUDGES = (
    "Stone quarry: no tool call. Pick a supplied tool now "
    "(search/filter, view, message, or place_test_order).",
    "Still idle. Advance the quarry: discover, inspect an exact sku, "
    "or seal with place_test_order if the cart is verified.",
)

# =============================================================================
# Facet bag + Vein Lattice
# =============================================================================


def _split_brief(text: str) -> tuple[str, str]:
    head, sep, tail = (text or "").partition("\n\n")
    if not sep:
        return (text or "").strip(), ""
    body = tail.strip()
    first, nl, rest = body.partition("\n")
    if first.rstrip().endswith(":") and len(first) < 48:
        body = rest.strip()
    return head.strip(), body


def _uniq(seq: list[str]) -> list[str]:
    return list(dict.fromkeys(seq))


class FacetBag:
    """Public signals harvested once from problem_data query + tools."""

    __slots__ = (
        "raw",
        "goal",
        "brief",
        "variants",
        "phrase",
        "ceiling",
        "ask_gate",
        "wants_why",
        "tool_names",
        "scores",
        "primary",
        "secondary",
        "blend",
    )

    def __init__(self, query: str, tools: list[dict[str, Any]]) -> None:
        self.raw = query or ""
        self.goal, self.brief = _split_brief(self.raw)
        self.variants = _uniq(_RX_VARIANT.findall(self.raw))
        hit = _RX_PHRASE.search(self.goal)
        self.phrase = hit.group(1) if hit else ""
        ceil = _RX_CEILING.search(self.goal) or _RX_CEILING.search(self.raw)
        self.ceiling = ceil.group(1) if ceil else ""
        self.ask_gate = bool(_RX_ASK.search(self.goal))
        why = bool(_RX_WHY.search(self.goal))
        if _RX_OPTIONAL_WHY.search(self.raw) and not re.search(
            r"\bwhy\b|reason|justify", self.goal, re.I
        ):
            why = False
        self.wants_why = why
        self.tool_names = _tool_names_from_schema(tools)
        self.scores = self._score_veins()
        self.primary, self.secondary, self.blend = self._lattice()

    def _score_veins(self) -> dict[str, int]:
        tally = {v: 0 for v in _ALL_VEINS}
        g = self.goal.lower()
        b = self.brief.lower()
        for vein, cues in _GOAL_CUES.items():
            tally[vein] += sum(w for tok, w in cues if tok in g)
        for vein, cues in _BRIEF_CUES.items():
            tally[vein] += sum(w for tok, w in cues if tok in b)

        # Structural evidence outranks vocabulary (FacetBag mixer).
        if len(self.variants) >= 2:
            tally[VEIN_PODIUM] += 9
        if self.ask_gate:
            tally[VEIN_LOCK] += 7
        if self.wants_why:
            tally[VEIN_VOUCH] += 6
        if self.phrase:
            tally[VEIN_SWEEP] += 4
        if "prefer" in g and ("firm" in g or "requirement" in g):
            tally[VEIN_SPLIT] += 4
        if "nice-to-have" in b or "subordinate" in b:
            tally[VEIN_SPLIT] += 3
        return tally

    def _lattice(self) -> tuple[str, str | None, bool]:
        best = max(self.scores.values())
        if best <= 0:
            return VEIN_SWEEP, None, False
        ranked = sorted(
            self.scores.items(),
            key=lambda kv: (kv[1], _VEIN_PRIORITY.get(kv[0], 0)),
            reverse=True,
        )
        primary = ranked[0][0]
        secondary = None
        blend = False
        if len(ranked) > 1 and ranked[1][1] > 0:
            gap = ranked[0][1] - ranked[1][1]
            ratio = ranked[1][1] / float(ranked[0][1])
            if gap <= _BLEND_GAP or ratio >= _BLEND_RATIO:
                secondary = ranked[1][0]
                blend = True
        return primary, secondary, blend

    def facet_card(self) -> str:
        lines = ["# Facet card (from public brief)"]
        lines.append(f"- primary vein: {self.primary}")
        if self.blend and self.secondary:
            lines.append(
                f"- secondary grit (blended): {self.secondary} "
                f"(scores {self.scores})"
            )
        else:
            lines.append(f"- vein scores: { {k: v for k, v in self.scores.items() if v} }")
        if self.ceiling:
            lines.append(f"- spend ceiling hint: {self.ceiling}")
        if self.phrase:
            lines.append(f"- quoted listing phrase: {self.phrase}")
        if self.variants:
            lines.append("- named variants: " + ", ".join(self.variants))
        if self.ask_gate:
            lines.append("- withheld firm figure: ask with message before search")
        if self.wants_why:
            lines.append("- order must carry grounded justification.claims")
        return "\n".join(lines)


# =============================================================================
# Strata playbooks — distilled from rank-5 donors (mechanisms only)
# =============================================================================

_BEDROCK = """
You are Stone, an ORO Bench shopping agent running a Vein Lattice quarry.

# Contract
- Use ONLY the supplied environment tools.
- Continue until an observation reports done=true.
- Respect max_steps and max_calls_per_turn.
- Parallel tool calls are fine when the turn limit allows.

# Quarry stages
1) SURVEY — search / filter with value-side keywords (never dump schema key names).
2) BIND — view / compare / inspect_stock on concrete product_id + sku.
3) SET — add_to_cart / remove_from_cart. First cart is graded: no placeholders.
4) SEAL — place_test_order once verified. Attach justification.claims only when required.

# Firm vs soft
Label every constraint FIRM or SOFT from the public brief.
Soft preferences never override firm gates (budget, stock, category, exact specs).

# Matching
- Literal: brands, model codes, sizes named as codes, uncommon tokens.
- Spec classes: only thresholds written in the brief — never invent floors.
- Soft paraphrase: everyday wording for soft axes only.

# Search grit (AurumGauge / Doublemark near-miss idea)
Catalog search matches VALUES. Prefer product-type nouns, brand, model, distinctive
phrase fragments. If the first page is thin: ONE small reformulation (drop noise OR
token cleanup OR category+filter) — do not rewrite five times.

# Dialogue
If the shopper asks a question, call message this/next turn before cart/order.
Keep messages short and concrete.
""".strip()

_STRATA: dict[str, str] = {
    VEIN_SPLIT: """
# Primary strata — SPLIT (TF1 Intent) · Doublemark / mother
Plan a short decision sequence:
1. List firm requirements; list soft leanings (preferred brand is soft).
2. Search/filter → view compliant exact skus.
3. First cart: preferred brand IF a fully firm-compliant sku exists; else best firm item.
4. Closest-when-impossible: if the soft brand cannot clear every firm gate, take any
   firm-compliant item — do not stall.
5. inspect_stock → place_test_order (omit optional claims).
Mother tally idea: among firm-equal options prefer clearer structured evidence over
title poetry.
""".strip(),
    VEIN_SWEEP: """
# Primary strata — SWEEP (TF2 Recall) · AurumGauge / my_teacher
- Pass-1 distinctive tokens; Pass-2 only if thin (one near-miss refine).
- Shortlist several DISTINCT product identities before buying.
- Quoted phrases must appear as title tokens on view — brand-only is not enough.
- Reject accessories / off-category noise.
- Dual-check idea (my_teacher): if two strong candidates disagree on a hard gate,
  keep the one that clears more firm literals.
- Once verified in-stock and in-budget, order promptly.
""".strip(),
    VEIN_LOCK: """
# Primary strata — LOCK (TF3 Constraints) · Tools
- EVERY firm requirement must hold on the EXACT ordered sku/option.
- Titles mix options — always view the option row, then inspect_stock on that sku.
- If a firm figure is withheld until asked: Turn 1 = message; do not search first.
- Pre-submit hard gate (Tools/Perfectstar): re-check literals/numbers/codes only.
- Sellout: move to a NEW product_id that still clears ALL firm gates — no sibling trap.
- Apply class/threshold rules only as written in the brief.
""".strip(),
    VEIN_TASTE: """
# Primary strata — TASTE (TF4 Preference) · Agent8 / ROM
Hard: category, budget, in-stock, real device.
Soft: use-case axes (durability, materials, environment, connectivity, feel).
ROM soft-adjective rule: vague feel words must NOT veto when fabric/type/size
evidence already matches.
Before first cart: compare ≥2 candidates on soft axes; prefer a slightly dearer
fit that still clears the budget. Never cart a cheap placeholder.
""".strip(),
    VEIN_PODIUM: """
# Primary strata — PODIUM (TF5 Ranking) · sim
- Restrict to named candidate variants (product_id::sku). Prefer viewing those IDs;
  do not burn the step budget on open search.
- Rank ONLY by the stated lexicographic / tradeoff priority. Never invent axes.
- sim early-batch idea: among equals that clear the axes, prefer the earlier
  listed survivor — still re-rank from scratch after market events.
- After stock/price/cart events: re-inspect survivors, RE-RANK, cart the NEW #1,
  inspect_stock, place_test_order. Do not keep a dominated pre-event pick.
- place_test_order = product_id + sku ONLY (no optional claims).
""".strip(),
    VEIN_MEND: """
# Primary strata — MEND (TF6 Recovery) · Perfectstar
Never relax: category, budget, in-stock.
Constraint ledger: keep the decomposed firm list; do not invent new gates mid-run.
Least-relaxation ladder after disappear / stock change:
  exact model+color → same model other color → other model still firm-compliant.
Zero-recall widen (Perfectstar): drop soft filters before touching firm walls.
Hard-gate swap ≤2 replacements, then seal with place_test_order.
""".strip(),
    VEIN_VOUCH: """
# Primary strata — VOUCH (TF7 Justification)
- View/compare raw facts before first cart and before order.
- place_test_order MUST include justification.claims as field/value pairs.
- Claim fields are catalog attribute keys: resolution, screen_size, brand, price,
  in_stock (plus schema feature reasons only when evidenced and relevant).
- Never use marketing aliases (display/size) as the field name.
- Keep claims minimal; ground every value in view/inspect observations.
""".strip(),
}

_GRIT_OVERLAY: dict[str, str] = {
    VEIN_SPLIT: (
        "# Secondary grit — SPLIT\n"
        "Keep firm/soft separation alive even under another primary vein; "
        "first cart still respects preferred brand only when fully compliant."
    ),
    VEIN_SWEEP: (
        "# Secondary grit — SWEEP\n"
        "Force a multi-identity shortlist before commit; one thin reformulation "
        "if recall is weak."
    ),
    VEIN_LOCK: (
        "# Secondary grit — LOCK\n"
        "Exact-sku hard gate on the final option; ask withheld specs before search "
        "when the brief says so."
    ),
    VEIN_TASTE: (
        "# Secondary grit — TASTE\n"
        "Soft axes beat cheapest-valid among firm-equal options; no placeholders."
    ),
    VEIN_PODIUM: (
        "# Secondary grit — PODIUM\n"
        "If named variants exist, treat them as a closed set and re-rank after events."
    ),
    VEIN_MEND: (
        "# Secondary grit — MEND\n"
        "On stock/price failures climb the least-relax ladder; never relax budget/stock/category."
    ),
    VEIN_VOUCH: (
        "# Secondary grit — VOUCH\n"
        "When the shopper asked why, attach allowlisted claims on the final order."
    ),
}

_STAGE_GUIDE = """
# Stage guide
- Early turns: SURVEY + BIND (and ask gated specs if needed).
- Mid turns: SET first cart once evidence is enough — do not hoard steps.
- Late turns (≥ ~70% of max_steps): stop open search; SEAL the order path.
- Always leave 1–2 turns for inspect_stock + place_test_order after events.
""".strip()


# =============================================================================
# Helpers
# =============================================================================


def _tool_names_from_schema(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
        elif isinstance(tool, dict) and tool.get("name"):
            names.append(str(tool["name"]))
    return names


def _openrouter_model(facets: FacetBag) -> str:
    override = (getenv("SANDBOX_MODEL") or "").strip()
    if override:
        return _TEE_MAP.get(override, override)
    if facets.primary in _DENSE_VEINS or (
        facets.secondary and facets.secondary in _DENSE_VEINS
    ):
        return _STONE_DENSE
    return _STONE_LIGHT


def _compose_system(facets: FacetBag, policy: dict[str, Any]) -> str:
    parts = [
        _BEDROCK,
        _STRATA.get(facets.primary, _STRATA[VEIN_SWEEP]),
    ]
    if facets.blend and facets.secondary and facets.secondary != facets.primary:
        parts.append(_GRIT_OVERLAY.get(facets.secondary, ""))
    parts.append(facets.facet_card())
    parts.append(_STAGE_GUIDE)
    tools = ", ".join(facets.tool_names) if facets.tool_names else "(see schemas)"
    parts.append(
        "# Runtime\n"
        f"Available tools: {tools}\n"
        f"max_steps={policy.get('max_steps')} "
        f"max_calls_per_turn={policy.get('max_calls_per_turn', 1)}"
    )
    return "\n\n".join(p for p in parts if p)


def _parse_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
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


def _compact(obs: Any, limit: int = _OBS_LIMIT) -> str:
    try:
        text = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obs)
    if len(text) <= limit:
        return text
    return text[: limit - 18] + "...<truncated>"


def _call_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for call in tool_calls:
        try:
            out.append(str(call["function"]["name"]))
        except (KeyError, TypeError):
            out.append("?")
    return out


def _guess_spec_field(goal: str) -> str | None:
    g = (goal or "").lower()
    if re.search(r"\b\d+(?:\.\d+)?\s*-?\s*inch|\bscreen size\b", g):
        return "screen_size"
    if re.search(r"\b(?:\d{3,4}\s*p|720p|1080p|full\s*hd|qhd|uhd|4k|resolution)\b", g):
        return "resolution"
    if re.search(r"\brefresh\b|\bhz\b", g):
        return "refresh_rate"
    if re.search(r"\bram\b|\bgb\b", g):
        return "ram"
    return None


def _canon_field(field: str, goal: str) -> str:
    raw = str(field or "").strip()
    key = raw.lower()
    if key in _CLAIM_ALIAS:
        return _CLAIM_ALIAS[key]
    if _RX_SPEC_AS_FIELD.match(raw):
        return _guess_spec_field(goal) or "resolution"
    return raw


def _canon_value(field: str, value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if field == "screen_size":
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            return f"{m.group(1)} inches"
    if field == "resolution":
        low = text.lower().replace(" ", "")
        if low in {"720p", "720phd", "hd"} or ("1280" in low and "720" in low):
            return "hd"
        if low in {"1080p", "fullhd", "fhd"} or ("1920" in low and "1080" in low):
            if "1920" in text and "1080" in text:
                return re.sub(r"\s+", " ", text)
            return "1920 x 1080"
        if low in {"1440p", "qhd"} or ("2560" in low and "1440" in low):
            if "2560" in text and "1440" in text:
                return re.sub(r"\s+", " ", text)
            return "2560 x 1440"
    if field == "in_stock":
        return "true" if text.lower() in {"true", "1", "yes", "in_stock"} else text
    return text


def _sanitize_order_args(
    name: str, args: dict[str, Any], facets: FacetBag
) -> dict[str, Any]:
    out = dict(args)
    if name != "place_test_order":
        return out

    primary = facets.primary
    needs_why = facets.wants_why or primary == VEIN_VOUCH

    if primary == VEIN_PODIUM and not needs_why:
        out.pop("justification", None)
        return out

    if not needs_why and primary != VEIN_VOUCH:
        # Optional claims are pure downside on non-vouch veins.
        out.pop("justification", None)
        return out

    just = out.get("justification")
    if not isinstance(just, dict):
        return out
    claims = just.get("claims")
    if not isinstance(claims, list):
        return out

    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    goal_l = facets.goal.lower()
    soft_kept = 0
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        field = _canon_field(str(claim.get("field") or ""), facets.goal)
        value = _canon_value(field, claim.get("value"))
        fl = field.lower()
        if not fl or value == "" or fl in _CLAIM_DROP:
            continue
        if fl not in _CLAIM_OK and fl not in _CLAIM_ALIAS.values():
            # Allow unknown catalog keys sparingly if not obviously prose.
            if " " in fl or len(fl) > 32:
                continue
        if fl in _CLAIM_SOFT:
            if fl not in goal_l or soft_kept >= 1:
                continue
            soft_kept += 1
        if fl in seen:
            continue
        seen.add(fl)
        cleaned.append({"field": field, "value": value})

    if cleaned:
        out["justification"] = {"claims": cleaned}
    else:
        out.pop("justification", None)
    return out


# =============================================================================
# Episode state + coaches
# =============================================================================


class QuarryState:
    __slots__ = (
        "stage",
        "turn",
        "await_reply",
        "asked_gate",
        "market_shift",
        "ordered",
        "first_cart",
        "viewed",
        "searched",
        "last_tools",
        "last_obs",
        "used_coaches",
    )

    def __init__(self) -> None:
        self.stage = STAGE_SURVEY
        self.turn = 0
        self.await_reply = False
        self.asked_gate = False
        self.market_shift = False
        self.ordered = False
        self.first_cart = False
        self.viewed = False
        self.searched = False
        self.last_tools: list[str] = []
        self.last_obs: list[Any] = []
        self.used_coaches: set[str] = set()

    def note_tools(self, names: list[str]) -> None:
        self.last_tools = names
        if any(n in names for n in ("search", "filter")):
            self.searched = True
            if self.stage == STAGE_SURVEY:
                self.stage = STAGE_BIND
        if any(n in names for n in ("view", "compare", "inspect_stock", "inspect_cart")):
            self.viewed = True
            if self.stage in (STAGE_SURVEY, STAGE_BIND):
                self.stage = STAGE_BIND
        if "message" in names:
            self.await_reply = False
            self.asked_gate = True
        if "add_to_cart" in names:
            self.first_cart = True
            self.stage = STAGE_SET
        if "remove_from_cart" in names:
            self.market_shift = True
        if "place_test_order" in names:
            self.ordered = True
            self.stage = STAGE_SEAL

    def note_obs(self, obs_list: list[Any]) -> None:
        self.last_obs = obs_list
        for obs in obs_list:
            if not isinstance(obs, dict):
                continue
            if obs.get("error") or obs.get("stock_changed") or obs.get("price_changed"):
                self.market_shift = True
            if obs.get("event"):
                self.market_shift = True

    def note_shopper(self, text: str) -> None:
        if not text:
            return
        if "?" in text or re.search(
            r"\b(?:which|what|why|how|prefer|tell me)\b", text, re.I
        ):
            self.await_reply = True


def _coach(facets: FacetBag, state: QuarryState, max_steps: int) -> str | None:
    """Priority-queue coaches: interventions > gates > events > stage nudges."""

    def once(key: str, text: str) -> str | None:
        if key in state.used_coaches:
            return None
        state.used_coaches.add(key)
        return text

    if state.await_reply:
        return (
            "Stone coach · reply owed: the shopper asked something. "
            "Call message with a concrete answer before cart/order."
        )

    if (
        facets.ask_gate
        and facets.primary == VEIN_LOCK
        and state.turn <= 1
        and not state.asked_gate
        and "message" not in state.last_tools
    ):
        return once(
            "ask_gate",
            "Stone coach · LOCK gate: a firm figure is withheld until asked. "
            "Call message first; do not search yet.",
        )

    # Blend lock grit even when lock is secondary.
    if (
        facets.blend
        and facets.secondary == VEIN_LOCK
        and facets.ask_gate
        and state.turn <= 1
        and not state.asked_gate
    ):
        return once(
            "ask_gate_grit",
            "Stone coach · secondary LOCK grit: ask the withheld firm figure "
            "with message before surveying.",
        )

    if facets.primary == VEIN_PODIUM and not state.ordered:
        if state.market_shift and state.turn >= 2:
            return (
                "Stone coach · PODIUM re-cut: catalog/cart moved. Re-inspect surviving "
                "named variants, rebuild the ranking from scratch on the stated axes, "
                "cart the NEW #1, inspect_stock, place_test_order (id+sku only)."
            )
        if state.turn == 1 and facets.variants:
            return once(
                "podium_open",
                "Stone coach · PODIUM open: view/compare the named variants "
                + ", ".join(facets.variants[:8])
                + " and rank by the stated priority before carting.",
            )
        if state.turn >= max(8, int(max_steps * 0.7)):
            return (
                "Stone coach · PODIUM seal: stop open search. Re-rank in-stock named "
                "candidates, cart current #1 if needed, place_test_order now "
                "(product_id+sku only)."
            )

    if state.market_shift and not state.ordered:
        if facets.primary == VEIN_MEND or facets.secondary == VEIN_MEND:
            return once(
                f"mend_{state.turn}",
                "Stone coach · MEND ladder: after the shift, climb least-relax "
                "(same model/other color before other models). Never relax budget, "
                "stock, or category. Then inspect_stock and seal.",
            )
        if facets.primary == VEIN_SPLIT:
            return once(
                f"split_event_{state.turn}",
                "Stone coach · SPLIT recover: remove invalid cart lines, re-find under "
                "updated firm walls; keep preferred brand only if still fully compliant.",
            )

    if facets.primary == VEIN_VOUCH and state.turn >= 2 and not state.ordered:
        return once(
            "vouch_claims",
            "Stone coach · VOUCH: on place_test_order use catalog claim keys "
            "(resolution, screen_size, brand, price, in_stock) — never display/size aliases.",
        )

    if facets.primary == VEIN_SWEEP and state.turn == 3 and state.searched:
        return once(
            "sweep_widen",
            "Stone coach · SWEEP: one small keyword reformulation or filter, then "
            "view several distinct product identities before ordering.",
        )

    if facets.primary == VEIN_TASTE and state.turn == 2 and not state.first_cart:
        return once(
            "taste_compare",
            "Stone coach · TASTE: compare candidates on soft use-case axes "
            "(not only price) before the first cart add.",
        )

    if (
        state.turn <= 2
        and not state.searched
        and not state.viewed
        and "message" not in state.last_tools
        and not (facets.ask_gate and facets.primary == VEIN_LOCK)
    ):
        return once(
            "survey_start",
            "Stone coach · SURVEY: start with search/filter using value-side keywords "
            "from the shopper goal, then view the best hits.",
        )

    # Late-stage seal pressure for any vein.
    if (
        not state.ordered
        and state.first_cart
        and state.turn >= max(7, int(max_steps * 0.75))
    ):
        return once(
            "late_seal",
            "Stone coach · late SEAL: inspect_stock on the cart sku and "
            "place_test_order before steps run out.",
        )

    return None


def _stone_chat(
    *,
    engine: str,
    transcript: list[dict[str, Any]],
    toolkit: list[dict[str, Any]],
    call_budget: int,
) -> tuple[str, list[dict[str, Any]]]:
    """OpenRouter chat with a short backup ladder (no phase routing)."""
    ladder = [engine]
    for mid in _STONE_BACKUP:
        if mid not in ladder:
            ladder.append(mid)

    last_err: Exception | None = None
    for mid in ladder[:3]:
        payload = _proxy.post(
            "/inference/chat/completions",
            json_data={
                "model": mid,
                "messages": transcript,
                "tools": toolkit,
                "tool_choice": "required",
                "temperature": 0,
            },
        )
        if payload is None:
            last_err = RuntimeError(f"stone chat failed engine={mid}")
            continue
        try:
            msg = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            last_err = exc
            continue
        prose = msg.get("content") or ""
        calls = list(msg.get("tool_calls") or [])[:call_budget]
        return prose, calls
    raise RuntimeError(f"stone chat exhausted ladder ({last_err})")


# =============================================================================
# Entry
# =============================================================================


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Drive the EnvPack episode with Vein Lattice mixing until done=true."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    query = str(policy.get("query") or problem_data.get("query") or "")
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
    max_steps = max(1, int(policy.get("max_steps", 20)))
    tools = policy.get("tools") or []

    facets = FacetBag(query, tools)
    model = _openrouter_model(facets)
    state = QuarryState()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _compose_system(facets, policy)},
        {"role": "user", "content": query},
    ]

    # Opening gate for LOCK primary (or blended lock grit).
    if facets.ask_gate and (
        facets.primary == VEIN_LOCK
        or (facets.blend and facets.secondary == VEIN_LOCK)
    ):
        opening = _coach(facets, state, max_steps)
        if opening:
            messages.append({"role": "user", "content": opening})

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "stone": {
                "version": 1,
                "primary": facets.primary,
                "secondary": facets.secondary,
                "blend": facets.blend,
                "scores": facets.scores,
                "model": model,
                "provider": "openrouter",
                "mixer": "vein_lattice",
            },
        }
    ]

    for turn in range(1, max_steps + 1):
        state.turn = turn
        hint = _coach(facets, state, max_steps)
        if hint:
            messages.append({"role": "user", "content": hint})

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for attempt in range(len(_EMPTY_NUDGES) + 1):
            assistant_content, tool_calls = _stone_chat(
                engine=model,
                transcript=messages,
                toolkit=tools,
                call_budget=max_calls,
            )
            if tool_calls:
                break
            dialogue.append({"role": "assistant", "content": assistant_content})
            nudge = _EMPTY_NUDGES[min(attempt, len(_EMPTY_NUDGES) - 1)]
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": nudge},
                ]
            )
        else:
            raise RuntimeError("model returned no tool call before completion")

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )

        group_id = f"{problem_id}-turn-{turn}"
        payloads = []
        names = _call_names(tool_calls)
        for index, call in enumerate(tool_calls, start=1):
            name = call["function"]["name"]
            args = _sanitize_order_args(name, _parse_args(call), facets)
            if isinstance(call.get("function"), dict):
                call["function"]["arguments"] = json.dumps(args)
            payloads.append(
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {"name": name, "args": args},
                }
            )

        state.note_tools(names)

        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": payloads,
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
                "stone_turn": {
                    "turn": turn,
                    "stage": state.stage,
                    "primary": facets.primary,
                    "secondary": facets.secondary,
                },
            }
        )

        obs_list: list[Any] = []
        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            obs = call_result.get("observation")
            obs_list.append(obs)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _compact(obs),
                }
            )
        state.note_obs(obs_list)

        user_message = result.get("user_message")
        if user_message and user_message.get("content"):
            text = str(user_message["content"])
            messages.append({"role": "user", "content": text})
            state.note_shopper(text)

        if any(
            isinstance(call.get("observation"), dict)
            and call["observation"].get("done")
            for call in result.get("calls") or []
        ):
            break

    return dialogue


__all__ = ["agent_main"]
