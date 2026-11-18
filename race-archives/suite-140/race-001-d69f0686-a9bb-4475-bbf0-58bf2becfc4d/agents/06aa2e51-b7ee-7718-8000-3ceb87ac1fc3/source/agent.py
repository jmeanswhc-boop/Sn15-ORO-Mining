"""koko — ORO Bench EnvPack agent (OpenRouter only).

Rank-5 ShoppingBench mechanisms (races 119–147) distilled into EnvPack
playbooks. Mixing uses **Ink Cascade** — a rule-ordered base ink plus
optional tint/glaze washes — not single-family hard-pick and not score-gap
blending.
"""

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------
# OpenRouter allowlist
# ---------------------------------------------------------------------------

_INK_LIGHT = "deepseek/deepseek-v3.2"
_INK_DENSE = "moonshotai/kimi-k2.6"
_INK_BACKUP = (
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "google/gemma-4-31b-it",
)

_CHUTES_BRIDGE = {
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
_OBS_CAP = 5800

# Ink identifiers (capability themes — generic fingerprints only)
INK_FORK = "fork"  # TF1 intent
INK_HUNT = "hunt"  # TF2 recall
INK_SEAL = "seal"  # TF3 constraints
INK_PALATE = "palate"  # TF4 preference
INK_LEDGER = "ledger"  # TF5 ranking
INK_PATCH = "patch"  # TF6 recovery
INK_WARRANT = "warrant"  # TF7 justification

_DENSE_INKS = frozenset(
    {INK_FORK, INK_SEAL, INK_PALATE, INK_LEDGER, INK_PATCH, INK_WARRANT}
)

# Relay phases
PHASE_DRAFT = "draft"
PHASE_PROOF = "proof"
PHASE_STAMP = "stamp"
PHASE_CLOSE = "close"

_IDLE_NUDGES = (
    "Koko relay: no tool chosen. Pick a supplied tool "
    "(search/filter, view, message, or place_test_order).",
    "Still idle. Advance the cascade: discover, prove an exact sku, "
    "or close with place_test_order once the cart is verified.",
)

# ---------------------------------------------------------------------------
# Signal probes (generic; no suite query copies)
# ---------------------------------------------------------------------------

_RX_VARIANT = re.compile(r"\b\d{6,}::[\w.-]+")
_RX_QUOTE = re.compile(r"[\"\u201c]([^\"\u201d]{3,60})[\"\u201d]")
_RX_BUDGET = re.compile(
    r"(?:at or (?:under|below)|under|below|up to|no more than|max(?:imum)?(?:\s+of)?)"
    r"\s*([\d][\d,]*(?:\.\d+)?)",
    re.I,
)
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

_RX_RANK = re.compile(
    r"tradeoff|best-?ranked|strictly worse|\bre-?rank\b|\bdominat|"
    r"candidate variants|lexico",
    re.I,
)
_RX_RECOVER = re.compile(
    r"least-?relax|relaxable|becomes? unavailable|sold[- ]?out|"
    r"same model.{0,40}(?:other|another) color|"
    r"exact (?:item|model|color).{0,60}disappear",
    re.I,
)
_RX_FIRM_SOFT = re.compile(
    r"nice[- ]to[- ]have|\bsubordinate\b|preferred brand|"
    r"\bfirm\b.{0,80}\bsoft\b|\bsoft\b.{0,80}\bfirm\b|"
    r"initial (?:cart )?choice",
    re.I,
)
_RX_PREFER = re.compile(
    r"preference axes|blindly buying the cheapest|intended use|"
    r"rather pay|use[- ]case|loose hard request",
    re.I,
)
_RX_RECALL = re.compile(
    r"\breformulate\b|\blucky hit\b|\bshortlist\b|"
    r"survey the realistic|distinct products|"
    r"matches? (?:the )?(?:phrase )?",
    re.I,
)
_RX_EXACT = re.compile(
    r"exact(?:ly)? varian|specific sku|every requirement must|"
    r"pixel class|resolution class|firm requirement on the spec",
    re.I,
)

_FIELD_ALIAS = {
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
_FIELD_OK = frozenset(
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
_FIELD_SOFT = frozenset(
    {"streaming", "microphone", "privacy_cover", "autofocus", "hdr"}
)
_FIELD_JUNK = frozenset(
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
_RX_VALUE_AS_FIELD = re.compile(
    r"(?i)^(?:\d{3,4}\s*p(?:\s*hd)?|full\s*hd|qhd|uhd|4k|hd|"
    r"\d+(?:\.\d+)?\s*-?\s*inch(?:es)?)$"
)


# =============================================================================
# Ink Cascade — boolean plate + ordered base + companion washes
# =============================================================================


def _partition_query(text: str) -> tuple[str, str]:
    head, sep, tail = (text or "").partition("\n\n")
    if not sep:
        return (text or "").strip(), ""
    body = tail.strip()
    first, nl, rest = body.partition("\n")
    if first.rstrip().endswith(":") and len(first) < 48:
        body = rest.strip()
    return head.strip(), body


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _schema_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
        elif isinstance(tool, dict) and tool.get("name"):
            names.append(str(tool["name"]))
    return names


class InkPlate:
    """Public signals as booleans + counts; cascade emits base/tint/glaze."""

    __slots__ = (
        "raw",
        "shopper",
        "ops",
        "variants",
        "quote",
        "ceiling",
        "ask_gate",
        "needs_claims",
        "flag_rank",
        "flag_recover",
        "flag_fork",
        "flag_palate",
        "flag_recall",
        "flag_exact",
        "tool_names",
        "base",
        "tint",
        "glaze",
        "washes",
        "cascade_trace",
    )

    def __init__(self, query: str, tools: list[dict[str, Any]]) -> None:
        self.raw = query or ""
        self.shopper, self.ops = _partition_query(self.raw)
        self.variants = _dedupe(_RX_VARIANT.findall(self.raw))
        qhit = _RX_QUOTE.search(self.shopper)
        self.quote = qhit.group(1) if qhit else ""
        bhit = _RX_BUDGET.search(self.shopper) or _RX_BUDGET.search(self.raw)
        self.ceiling = bhit.group(1) if bhit else ""
        self.ask_gate = bool(_RX_ASK.search(self.shopper))
        why = bool(_RX_WHY.search(self.shopper))
        if _RX_OPTIONAL_WHY.search(self.raw) and not re.search(
            r"\bwhy\b|reason|justify", self.shopper, re.I
        ):
            why = False
        self.needs_claims = why
        joined = f"{self.shopper}\n{self.ops}"
        self.flag_rank = bool(_RX_RANK.search(joined)) or len(self.variants) >= 2
        self.flag_recover = bool(_RX_RECOVER.search(joined))
        self.flag_fork = bool(_RX_FIRM_SOFT.search(joined))
        self.flag_palate = bool(_RX_PREFER.search(joined))
        self.flag_recall = bool(_RX_RECALL.search(joined)) or bool(self.quote)
        self.flag_exact = bool(_RX_EXACT.search(joined)) or self.ask_gate
        self.tool_names = _schema_names(tools)
        self.base, self.tint, self.glaze, self.cascade_trace = self._cascade()
        self.washes = [w for w in (self.tint, self.glaze) if w]

    def _cascade(self) -> tuple[str, str | None, str | None, list[str]]:
        """Ink Cascade: first matching rule paints BASE; companions attach by rule.

        Unlike score-max + gap/ratio blending, this is a discrete decision table:
        specialised closed-set / recovery / claims rules fire before softer cues.
        Companions are forced co-activations (ask-gate glaze, claims tint, …),
        not the runner-up score.
        """
        trace: list[str] = []

        # --- BASE (ordered; first hit wins) ---
        if len(self.variants) >= 2 or (
            self.flag_rank and not self.flag_recover and not self.needs_claims
        ):
            base = INK_LEDGER
            trace.append("base<=named_or_rank")
        elif self.flag_recover:
            base = INK_PATCH
            trace.append("base<=recover_cues")
        elif self.needs_claims:
            base = INK_WARRANT
            trace.append("base<=claims_required")
        elif self.ask_gate or (self.flag_exact and not self.flag_fork):
            base = INK_SEAL
            trace.append("base<=ask_or_exact")
        elif self.flag_fork:
            base = INK_FORK
            trace.append("base<=firm_soft")
        elif self.flag_palate:
            base = INK_PALATE
            trace.append("base<=preference")
        elif self.flag_recall:
            base = INK_HUNT
            trace.append("base<=recall")
        else:
            base = INK_HUNT
            trace.append("base<=default_hunt")

        tint: str | None = None
        glaze: str | None = None

        # --- Companion washes (co-activation; skip if same as base) ---
        def _attach(role: str, ink: str, reason: str) -> None:
            nonlocal tint, glaze
            if ink == base:
                return
            if role == "tint" and tint is None:
                tint = ink
                trace.append(f"tint<={reason}")
            elif role == "glaze" and glaze is None and ink != tint:
                glaze = ink
                trace.append(f"glaze<={reason}")

        # Ask-gate always glazes SEAL discipline onto non-seal bases.
        if self.ask_gate and base != INK_SEAL:
            _attach("glaze", INK_SEAL, "ask_gate")

        # Claims requirement tints WARRANT when another ink is base.
        if self.needs_claims and base != INK_WARRANT:
            _attach("tint", INK_WARRANT, "claims")

        # Ranking + recovery co-signal → patch wash (post-event ladder).
        if base == INK_LEDGER and self.flag_recover:
            _attach("tint", INK_PATCH, "ledger+recover")

        # Intent + preference → palate wash for soft-axis compare.
        if base == INK_FORK and self.flag_palate:
            _attach("tint", INK_PALATE, "fork+palate")

        # Exact sku pressure under fork/palate → seal tint.
        if base in (INK_FORK, INK_PALATE) and self.flag_exact and tint is None:
            _attach("tint", INK_SEAL, "exact_under_soft")

        # Recall under firm/soft language → fork tint.
        if base == INK_HUNT and self.flag_fork:
            _attach("tint", INK_FORK, "hunt+fork")

        # Closed-set variants under a non-ledger base → ledger glaze.
        if base != INK_LEDGER and len(self.variants) >= 2:
            _attach("glaze", INK_LEDGER, "closed_set")

        return base, tint, glaze, trace

    def active_set(self) -> list[str]:
        out = [self.base]
        out.extend(self.washes)
        return out

    def has(self, ink: str) -> bool:
        return ink == self.base or ink in self.washes

    def plate_card(self) -> str:
        lines = ["# Ink plate (from public brief)"]
        lines.append(f"- base ink: {self.base}")
        if self.tint:
            lines.append(f"- tint wash: {self.tint}")
        if self.glaze:
            lines.append(f"- glaze wash: {self.glaze}")
        lines.append(f"- cascade: {', '.join(self.cascade_trace)}")
        if self.ceiling:
            lines.append(f"- spend ceiling hint: {self.ceiling}")
        if self.quote:
            lines.append(f"- quoted listing phrase: {self.quote}")
        if self.variants:
            lines.append("- named variants: " + ", ".join(self.variants))
        if self.ask_gate:
            lines.append("- withheld firm figure: ask with message before search")
        if self.needs_claims:
            lines.append("- order must carry grounded justification.claims")
        return "\n".join(lines)


# =============================================================================
# Playbooks — rank-5 mechanisms only (Doublemark, AurumGauge, Tools, …)
# =============================================================================

_PRIMER = """
You are koko, an ORO Bench shopping agent running an Ink Cascade relay.

# Contract
- Use ONLY the supplied environment tools.
- Continue until an observation reports done=true.
- Respect max_steps and max_calls_per_turn.
- Parallel tool calls are fine when the turn limit allows.

# Relay phases
1) DRAFT — search / filter with value-side keywords (never dump schema key names).
2) PROOF — view / compare / inspect_stock on concrete product_id + sku.
3) STAMP — add_to_cart / remove_from_cart. First cart is graded: no placeholders.
4) CLOSE — place_test_order once verified. Attach justification.claims only when required.

# Firm vs soft
Label every constraint FIRM or SOFT from the public brief.
Soft preferences never override firm gates (budget, stock, category, exact specs).

# Matching
- Literal: brands, model codes, sizes named as codes, uncommon tokens.
- Spec classes: only thresholds written in the brief — never invent floors.
- Soft paraphrase: everyday wording for soft axes only.

# Search (AurumGauge near-miss + Doublemark cleanup)
Catalog search matches VALUES. Prefer product-type nouns, brand, model, distinctive
phrase fragments. If the first page is thin: ONE small reformulation (drop noise OR
token cleanup OR category+filter) — do not rewrite five times.

# Dialogue
If the shopper asks a question, call message this/next turn before cart/order.
Keep messages short and concrete.
""".strip()

_BASE_INK: dict[str, str] = {
    INK_FORK: """
# Base ink — FORK (TF1 Intent) · Doublemark / mother
Plan a short decision sequence:
1. List firm requirements; list soft leanings (preferred brand is soft).
2. Search/filter → view compliant exact skus.
3. First cart: preferred brand IF a fully firm-compliant sku exists; else best firm item.
4. Closest-when-impossible: if the soft brand cannot clear every firm gate, take any
   firm-compliant item — do not stall.
5. inspect_stock → place_test_order (omit optional claims).
Among firm-equal options prefer clearer structured evidence over title poetry.
""".strip(),
    INK_HUNT: """
# Base ink — HUNT (TF2 Recall) · AurumGauge / my_teacher
- Pass-1 distinctive tokens; Pass-2 only if thin (one near-miss refine).
- Shortlist several DISTINCT product identities before buying.
- Quoted phrases must appear as title tokens on view — brand-only is not enough.
- Reject accessories / off-category noise.
- Dual-check: if two strong candidates disagree on a hard gate, keep the one that
  clears more firm literals.
- Once verified in-stock and in-budget, order promptly.
""".strip(),
    INK_SEAL: """
# Base ink — SEAL (TF3 Constraints) · Tools
- EVERY firm requirement must hold on the EXACT ordered sku/option.
- Titles mix options — always view the option row, then inspect_stock on that sku.
- If a firm figure is withheld until asked: Turn 1 = message; do not search first.
- Pre-submit hard gate: re-check literals/numbers/codes only.
- Sellout: move to a NEW product_id that still clears ALL firm gates — no sibling trap.
- Apply class/threshold rules only as written in the brief.
""".strip(),
    INK_PALATE: """
# Base ink — PALATE (TF4 Preference) · Agent8 / ROM
Hard: category, budget, in-stock, real device.
Soft: use-case axes (durability, materials, environment, connectivity, feel).
Vague feel words must NOT veto when fabric/type/size evidence already matches.
Before first cart: compare ≥2 candidates on soft axes; prefer a slightly dearer
fit that still clears the budget. Never cart a cheap placeholder.
""".strip(),
    INK_LEDGER: """
# Base ink — LEDGER (TF5 Ranking) · sim
- Restrict to named candidate variants (product_id::sku). Prefer viewing those IDs;
  do not burn the step budget on open search.
- Rank ONLY by the stated lexicographic / tradeoff priority. Never invent axes.
- Among equals that clear the axes, prefer the earlier listed survivor — still
  re-rank from scratch after market events.
- After stock/price/cart events: re-inspect survivors, RE-RANK, cart the NEW #1,
  inspect_stock, place_test_order. Do not keep a dominated pre-event pick.
- place_test_order = product_id + sku ONLY (no optional claims).
""".strip(),
    INK_PATCH: """
# Base ink — PATCH (TF6 Recovery) · Perfectstar
Never relax: category, budget, in-stock.
Constraint ledger: keep the decomposed firm list; do not invent new gates mid-run.
Least-relaxation ladder after disappear / stock change:
  exact model+color → same model other color → other model still firm-compliant.
Zero-recall widen: drop soft filters before touching firm walls.
Hard-gate swap ≤2 replacements, then close with place_test_order.
""".strip(),
    INK_WARRANT: """
# Base ink — WARRANT (TF7 Justification)
- View/compare raw facts before first cart and before order.
- place_test_order MUST include justification.claims as field/value pairs.
- Claim fields are catalog attribute keys: resolution, screen_size, brand, price,
  in_stock (plus schema feature reasons only when evidenced and relevant).
- Never use marketing aliases (display/size) as the field name.
- Keep claims minimal; ground every value in view/inspect observations.
""".strip(),
}

_WASH: dict[str, str] = {
    INK_FORK: (
        "# Tint/glaze wash — FORK\n"
        "Keep firm/soft separation alive; first cart still respects preferred brand "
        "only when fully compliant."
    ),
    INK_HUNT: (
        "# Tint/glaze wash — HUNT\n"
        "Force a multi-identity shortlist before commit; one thin reformulation "
        "if recall is weak."
    ),
    INK_SEAL: (
        "# Tint/glaze wash — SEAL\n"
        "Exact-sku hard gate on the final option; ask withheld specs before search "
        "when the brief says so."
    ),
    INK_PALATE: (
        "# Tint/glaze wash — PALATE\n"
        "Soft axes beat cheapest-valid among firm-equal options; no placeholders."
    ),
    INK_LEDGER: (
        "# Tint/glaze wash — LEDGER\n"
        "If named variants exist, treat them as a closed set and re-rank after events."
    ),
    INK_PATCH: (
        "# Tint/glaze wash — PATCH\n"
        "On stock/price failures climb the least-relax ladder; never relax "
        "budget/stock/category."
    ),
    INK_WARRANT: (
        "# Tint/glaze wash — WARRANT\n"
        "When the shopper asked why, attach allowlisted claims on the final order."
    ),
}

_RELAY_GUIDE = """
# Relay guide
- Early turns: DRAFT + PROOF (and ask gated specs if needed).
- Mid turns: STAMP first cart once evidence is enough — do not hoard steps.
- Late turns (≥ ~70% of max_steps): stop open search; CLOSE the order path.
- Always leave 1–2 turns for inspect_stock + place_test_order after events.
""".strip()


# =============================================================================
# Inference + sanitizers
# =============================================================================


def _pick_model(plate: InkPlate) -> str:
    override = (getenv("SANDBOX_MODEL") or "").strip()
    if override:
        return _CHUTES_BRIDGE.get(override, override)
    active = plate.active_set()
    if any(ink in _DENSE_INKS for ink in active):
        return _INK_DENSE
    return _INK_LIGHT


def _compose_system(plate: InkPlate, policy: dict[str, Any]) -> str:
    parts = [_PRIMER, _BASE_INK.get(plate.base, _BASE_INK[INK_HUNT])]
    for wash in plate.washes:
        parts.append(_WASH.get(wash, ""))
    parts.append(plate.plate_card())
    parts.append(_RELAY_GUIDE)
    tools = ", ".join(plate.tool_names) if plate.tool_names else "(see schemas)"
    parts.append(
        "# Runtime\n"
        f"Available tools: {tools}\n"
        f"max_steps={policy.get('max_steps')} "
        f"max_calls_per_turn={policy.get('max_calls_per_turn', 1)}"
    )
    return "\n\n".join(p for p in parts if p)


def _load_args(tool_call: dict[str, Any]) -> dict[str, Any]:
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


def _shrink(obs: Any, limit: int = _OBS_CAP) -> str:
    try:
        text = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obs)
    if len(text) <= limit:
        return text
    return text[: limit - 16] + "...<truncated>"


def _names_of(tool_calls: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for call in tool_calls:
        try:
            out.append(str(call["function"]["name"]))
        except (KeyError, TypeError):
            out.append("?")
    return out


def _guess_field(goal: str) -> str | None:
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


def _fix_field(field: str, goal: str) -> str:
    raw = str(field or "").strip()
    key = raw.lower()
    if key in _FIELD_ALIAS:
        return _FIELD_ALIAS[key]
    if _RX_VALUE_AS_FIELD.match(raw):
        return _guess_field(goal) or "resolution"
    return raw


def _fix_value(field: str, value: Any) -> str:
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


def _scrub_order(name: str, args: dict[str, Any], plate: InkPlate) -> dict[str, Any]:
    out = dict(args)
    if name != "place_test_order":
        return out

    needs_why = plate.needs_claims or plate.has(INK_WARRANT)

    if plate.has(INK_LEDGER) and not needs_why:
        out.pop("justification", None)
        return out

    if not needs_why:
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
    goal_l = plate.shopper.lower()
    soft_kept = 0
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        field = _fix_field(str(claim.get("field") or ""), plate.shopper)
        value = _fix_value(field, claim.get("value"))
        fl = field.lower()
        if not fl or value == "" or fl in _FIELD_JUNK:
            continue
        if fl not in _FIELD_OK and fl not in _FIELD_ALIAS.values():
            if " " in fl or len(fl) > 32:
                continue
        if fl in _FIELD_SOFT:
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
# Relay board + coaches
# =============================================================================


class RelayBoard:
    __slots__ = (
        "phase",
        "turn",
        "reply_owed",
        "asked",
        "market_moved",
        "ordered",
        "carted",
        "viewed",
        "searched",
        "last_tools",
        "last_obs",
        "spent_hints",
        "live_washes",
    )

    def __init__(self, plate: InkPlate) -> None:
        self.phase = PHASE_DRAFT
        self.turn = 0
        self.reply_owed = False
        self.asked = False
        self.market_moved = False
        self.ordered = False
        self.carted = False
        self.viewed = False
        self.searched = False
        self.last_tools: list[str] = []
        self.last_obs: list[Any] = []
        self.spent_hints: set[str] = set()
        self.live_washes: list[str] = list(plate.washes)

    def note_tools(self, names: list[str]) -> None:
        self.last_tools = names
        if any(n in names for n in ("search", "filter")):
            self.searched = True
            if self.phase == PHASE_DRAFT:
                self.phase = PHASE_PROOF
        if any(n in names for n in ("view", "compare", "inspect_stock", "inspect_cart")):
            self.viewed = True
            if self.phase in (PHASE_DRAFT, PHASE_PROOF):
                self.phase = PHASE_PROOF
        if "message" in names:
            self.reply_owed = False
            self.asked = True
        if "add_to_cart" in names:
            self.carted = True
            self.phase = PHASE_STAMP
        if "remove_from_cart" in names:
            self.market_moved = True
        if "place_test_order" in names:
            self.ordered = True
            self.phase = PHASE_CLOSE

    def note_obs(self, obs_list: list[Any]) -> None:
        self.last_obs = obs_list
        for obs in obs_list:
            if not isinstance(obs, dict):
                continue
            if obs.get("error") or obs.get("stock_changed") or obs.get("price_changed"):
                self.market_moved = True
            if obs.get("event"):
                self.market_moved = True

    def note_shopper(self, text: str) -> None:
        if not text:
            return
        if "?" in text or re.search(
            r"\b(?:which|what|why|how|prefer|tell me)\b", text, re.I
        ):
            self.reply_owed = True

    def inject_patch_wash(self, plate: InkPlate) -> str | None:
        """Mid-episode cascade re-paint: market shift adds PATCH wash if absent."""
        if not self.market_moved or self.ordered:
            return None
        if plate.has(INK_PATCH) or INK_PATCH in self.live_washes:
            return None
        self.live_washes.append(INK_PATCH)
        return _WASH[INK_PATCH]


def _hint(plate: InkPlate, board: RelayBoard, max_steps: int) -> str | None:
    """Priority coaches: reply > ask-gate > ledger events > washes > phase."""

    def once(key: str, text: str) -> str | None:
        if key in board.spent_hints:
            return None
        board.spent_hints.add(key)
        return text

    if board.reply_owed:
        return (
            "Koko coach · reply owed: the shopper asked something. "
            "Call message with a concrete answer before cart/order."
        )

    seal_active = plate.base == INK_SEAL or plate.has(INK_SEAL)
    if (
        plate.ask_gate
        and seal_active
        and board.turn <= 1
        and not board.asked
        and "message" not in board.last_tools
    ):
        return once(
            "ask_gate",
            "Koko coach · SEAL gate: a firm figure is withheld until asked. "
            "Call message first; do not search yet.",
        )

    if plate.has(INK_LEDGER) and not board.ordered:
        if board.market_moved and board.turn >= 2:
            return (
                "Koko coach · LEDGER re-cut: catalog/cart moved. Re-inspect surviving "
                "named variants, rebuild the ranking from scratch on the stated axes, "
                "cart the NEW #1, inspect_stock, place_test_order (id+sku only)."
            )
        if board.turn == 1 and plate.variants:
            return once(
                "ledger_open",
                "Koko coach · LEDGER open: view/compare the named variants "
                + ", ".join(plate.variants[:8])
                + " and rank by the stated priority before carting.",
            )
        if board.turn >= max(8, int(max_steps * 0.7)):
            return (
                "Koko coach · LEDGER close: stop open search. Re-rank in-stock named "
                "candidates, cart current #1 if needed, place_test_order now "
                "(product_id+sku only)."
            )

    if board.market_moved and not board.ordered:
        if plate.has(INK_PATCH) or INK_PATCH in board.live_washes:
            return once(
                f"patch_{board.turn}",
                "Koko coach · PATCH ladder: after the shift, climb least-relax "
                "(same model/other color before other models). Never relax budget, "
                "stock, or category. Then inspect_stock and close.",
            )
        if plate.base == INK_FORK:
            return once(
                f"fork_event_{board.turn}",
                "Koko coach · FORK recover: remove invalid cart lines, re-find under "
                "updated firm walls; keep preferred brand only if still fully compliant.",
            )

    if plate.has(INK_WARRANT) and board.turn >= 2 and not board.ordered:
        return once(
            "warrant_claims",
            "Koko coach · WARRANT: on place_test_order use catalog claim keys "
            "(resolution, screen_size, brand, price, in_stock) — never display/size aliases.",
        )

    if plate.base == INK_HUNT and board.turn == 3 and board.searched:
        return once(
            "hunt_widen",
            "Koko coach · HUNT: one small keyword reformulation or filter, then "
            "view several distinct product identities before ordering.",
        )

    if plate.base == INK_PALATE and board.turn == 2 and not board.carted:
        return once(
            "palate_compare",
            "Koko coach · PALATE: compare candidates on soft use-case axes "
            "(not only price) before the first cart add.",
        )

    if (
        board.turn <= 2
        and not board.searched
        and not board.viewed
        and "message" not in board.last_tools
        and not (plate.ask_gate and seal_active)
    ):
        return once(
            "draft_start",
            "Koko coach · DRAFT: start with search/filter using value-side keywords "
            "from the shopper goal, then view the best hits.",
        )

    if (
        not board.ordered
        and board.carted
        and board.turn >= max(7, int(max_steps * 0.75))
    ):
        return once(
            "late_close",
            "Koko coach · late CLOSE: inspect_stock on the cart sku and "
            "place_test_order before steps run out.",
        )

    return None


def _chat(
    *,
    engine: str,
    transcript: list[dict[str, Any]],
    toolkit: list[dict[str, Any]],
    call_budget: int,
) -> tuple[str, list[dict[str, Any]]]:
    ladder = [engine]
    for mid in _INK_BACKUP:
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
            last_err = RuntimeError(f"koko chat failed engine={mid}")
            continue
        try:
            msg = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            last_err = exc
            continue
        prose = msg.get("content") or ""
        calls = list(msg.get("tool_calls") or [])[:call_budget]
        return prose, calls
    raise RuntimeError(f"koko chat exhausted ladder ({last_err})")


# =============================================================================
# Entry
# =============================================================================


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Drive the EnvPack episode with Ink Cascade mixing until done=true."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    query = str(policy.get("query") or problem_data.get("query") or "")
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
    max_steps = max(1, int(policy.get("max_steps", 20)))
    tools = policy.get("tools") or []

    plate = InkPlate(query, tools)
    model = _pick_model(plate)
    board = RelayBoard(plate)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _compose_system(plate, policy)},
        {"role": "user", "content": query},
    ]

    if plate.ask_gate and (plate.base == INK_SEAL or plate.has(INK_SEAL)):
        opening = _hint(plate, board, max_steps)
        if opening:
            messages.append({"role": "user", "content": opening})

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "koko": {
                "version": 1,
                "base": plate.base,
                "tint": plate.tint,
                "glaze": plate.glaze,
                "cascade": plate.cascade_trace,
                "model": model,
                "provider": "openrouter",
                "mixer": "ink_cascade",
            },
        }
    ]

    for turn in range(1, max_steps + 1):
        board.turn = turn

        # Observation-driven cascade re-paint (still one strategy).
        wash_note = board.inject_patch_wash(plate)
        if wash_note:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Koko cascade re-paint: market shifted — adding PATCH wash.\n"
                        + wash_note
                    ),
                }
            )

        hint = _hint(plate, board, max_steps)
        if hint:
            messages.append({"role": "user", "content": hint})

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for attempt in range(len(_IDLE_NUDGES) + 1):
            assistant_content, tool_calls = _chat(
                engine=model,
                transcript=messages,
                toolkit=tools,
                call_budget=max_calls,
            )
            if tool_calls:
                break
            dialogue.append({"role": "assistant", "content": assistant_content})
            nudge = _IDLE_NUDGES[min(attempt, len(_IDLE_NUDGES) - 1)]
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
        names = _names_of(tool_calls)
        for index, call in enumerate(tool_calls, start=1):
            name = call["function"]["name"]
            args = _scrub_order(name, _load_args(call), plate)
            if isinstance(call.get("function"), dict):
                call["function"]["arguments"] = json.dumps(args)
            payloads.append(
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {"name": name, "args": args},
                }
            )

        board.note_tools(names)

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
                "koko_turn": {
                    "turn": turn,
                    "phase": board.phase,
                    "base": plate.base,
                    "live_washes": list(board.live_washes),
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
                    "content": _shrink(obs),
                }
            )
        board.note_obs(obs_list)

        user_message = result.get("user_message")
        if user_message and user_message.get("content"):
            text = str(user_message["content"])
            messages.append({"role": "user", "content": text})
            board.note_shopper(text)

        if any(
            isinstance(call.get("observation"), dict)
            and call["observation"].get("done")
            for call in result.get("calls") or []
        ):
            break

    return dialogue


__all__ = ["agent_main"]
