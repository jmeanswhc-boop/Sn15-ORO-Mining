"""Policy-first shopping agent for ORO Bench generated environments.

Unlike the minimal reference loop that blindly trusts every model tool call,
this agent keeps a live episode ledger and a deterministic controller that:

* extracts a shopper mandate from the public goal + later notes
* forces clarification, discovery, inspection, recovery, and checkout when
  the next step is unambiguous from observations
* asks the LLM only when the controller must deliberate
* grounds checkout claims from viewed / inspected facts and the live schema

All branching is derived from ``problem_data`` fields and runtime observations.
"""

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_proxy = ProxyClient(timeout=180, max_retries=4)

# Provider-native allowlisted IDs. Routing uses only provider config + problem cues.
_MODELS: dict[str, dict[str, Any]] = {
    "chutes": {
        "strong": "Qwen/Qwen3.5-397B-A17B-TEE",
        "fast": "deepseek-ai/DeepSeek-V3.2-TEE",
        "fallback": (
            "moonshotai/Kimi-K2.6-TEE",
            "zai-org/GLM-5.2-TEE",
            "Qwen/Qwen3.6-27B-TEE",
        ),
    },
    "openrouter": {
        "strong": "qwen/qwen3.5-397b-a17b",
        "fast": "deepseek/deepseek-v3.2",
        "fallback": (
            "moonshotai/kimi-k2.6",
            "z-ai/glm-5.2",
            "qwen/qwen3.6-27b",
        ),
    },
}

_TEE_TO_OPENROUTER = {
    "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
    "deepseek-ai/DeepSeek-V3.1-TEE": "deepseek/deepseek-chat-v3.1",
    "deepseek-ai/DeepSeek-V3-0324-TEE": "deepseek/deepseek-chat-v3-0324",
    "deepseek-ai/DeepSeek-R1-0528-TEE": "deepseek/deepseek-r1-0528",
    "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
    "Qwen/Qwen3-32B-TEE": "qwen/qwen3-32b",
    "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
    "moonshotai/Kimi-K2.6-TEE": "moonshotai/kimi-k2.6",
    "zai-org/GLM-5.1-TEE": "z-ai/glm-5.1",
    "zai-org/GLM-5.2-TEE": "z-ai/glm-5.2",
    "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
    "XiaomiMiMo/MiMo-V2-Flash-TEE": "xiaomi/mimo-v2-flash",
    "openai/gpt-oss-120b-TEE": "openai/gpt-oss-120b",
    "unsloth/Mistral-Nemo-Instruct-2407-TEE": "mistralai/mistral-small-2603",
}

_WORD_RE = re.compile(r"[a-z0-9]+", re.I)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_VARIANT_RE = re.compile(r"\b(\d{6,})::([A-Za-z0-9_-]+)\b")
_BUDGET_RE = re.compile(
    r"(?:"
    r"(?:under|below|upto|up\s+to|at\s+or\s+under|no\s+more\s+than|max(?:imum)?|"
    r"budget(?:\s+of)?|spend(?:\s+up\s+to)?|can\s+spend(?:\s+up\s+to)?|less\s+than)\s*"
    r"(?:of\s+)?"
    r"(?P<a>\d[\d,]*(?:\.\d+)?)"
    r"|"
    r"(?P<b>\d[\d,]*(?:\.\d+)?)\s*(?P<cur>[a-z]{3})\b"
    r")",
    re.I,
)
_DIR_UP = frozenset(
    "higher more greater larger better maximize maximise prefer preferring".split()
)
_DIR_DOWN = frozenset(
    "lower less smaller cheaper minimize minimise".split()
)
_RANK_AXIS_RE = re.compile(
    r"\b(?P<dir>higher|more|greater|larger|better|prefer(?:ring)?|maximize|maximise|"
    r"lower|less|smaller|cheaper|minimize|minimise)\s+"
    r"(?P<axis>[a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*){0,3})"
    r"(?=\s*(?:,|;|\.|then|and|before|after|than|$))",
    re.I,
)
_FIRM_CUE_RE = re.compile(
    r"\b(?:must|need(?:s|ed)?|require(?:s|d)?|only|exactly|strict(?:ly)?|"
    r"have\s+to|has\s+to|cannot|can't|won't|will\s+not)\b",
    re.I,
)
_SOFT_CUE_RE = re.compile(
    r"\b(?:prefer(?:red|ably)?|ideally|nice\s+to\s+have|if\s+possible|"
    r"would\s+like|bonus|optional|soft)\b",
    re.I,
)
_DEFER_ASK_RE = (
    re.compile(
        r"\b(?:ask|asking)\b.{0,48}\b(?:tell|share|reveal|provide|give)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:tell|share|reveal|provide|give)\b.{0,48}\b(?:ask|asking)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:haven'?t|have not|not yet)\s+(?:told|said|shared|given)\b",
        re.I,
    ),
    re.compile(
        r"\bif\b.{0,24}\bask\b.{0,40}\b(?:tell|share|reveal|provide|give)\b",
        re.I,
    ),
    re.compile(
        r"\bwill\s+tell\b.{0,64}\bif\s+you\s+ask\b",
        re.I,
    ),
    re.compile(
        r"\bone\s+firm\s+requirement\b.{0,80}\bask\b",
        re.I,
    ),
)

# Shopping-protocol procedure ids derived from public goal/schema structure.
# Names describe control strategy — not EnvPack family labels or product niches.
PROC_ASK = "ask_gate"  # deferred firm spec until agent asks
PROC_RECALL = "phrase_recall"  # quote/phrase discovery + survey shortlist
PROC_STRICT = "strict_match"  # every stated atom must evidence on the SKU
PROC_SOFT = "soft_prefer"  # soft wish subordinate to firm constraints
PROC_USE = "use_fit"  # intended-use soft axes among compliant options
PROC_RANK = "axis_rank"  # named variants + comparative axes
PROC_ADAPT = "event_adapt"  # recover after stock/price/cart events
PROC_CLAIM = "claim_why"  # evidenced justification at checkout

_WHY_GOAL_RE = re.compile(
    r"\b(?:why\b.{0,28}(?:pick|choose|order|one|it)|tell me why|walk me through why|"
    r"give me a reason|reason I can|justify|justification)\b",
    re.I,
)
_PHRASE_GOAL_RE = re.compile(
    r"(?:matches?\s+(?:the\s+)?(?:phrase\s+)?|"
    r"that\s+matches\s+)"
    r"[\"\u201c]?([^\"\u201d\n,]{2,80})",
    re.I,
)
_SOFT_OBJECT_RE = re.compile(
    r"(?:prefer(?:red|ably)?|ideally(?:\s*,)?\s*I\s+prefer)\s+"
    r"(?:the\s+)?(?:brand\s+)?([A-Za-z][A-Za-z0-9][\w.&'-]{0,40})",
    re.I,
)
_USE_FIT_RE = re.compile(
    r"\b(?:it'?s\s+for|for)\s+(?!the\b|a\b|an\b|this\b|that\b|my\b|your\b|me\b|"
    r"us\b|him\b|her\b|them\b|it\b|in\b|stock\b|under\b|below\b|up\b)"
    r"[a-z][a-z0-9-]{2,}",
    re.I,
)
_ADAPT_GOAL_RE = re.compile(
    r"\b(?:if\s+(?:it|that|the\s+item)\s+(?:becomes?\s+)?(?:unavailable|disappears?|"
    r"sold\s*out)|recover|fallback|another\s+(?:colou?r|color|option|variant))\b",
    re.I,
)
_PREF_MODEL_RE = re.compile(
    r"(?:strongly\s+prefer|prefer(?:red)?|want|keep)\s+(?:the\s+)?"
    r"(?:(?P<color>[A-Za-z]{3,12})\s+)?"
    r"(?P<code>[A-Za-z]*\d{2,}[A-Za-z0-9-]*)",
    re.I,
)
_CATEGORY_GOAL_RE = re.compile(
    r"(?:(?:in|from)\s+the\s+|"
    r"(?:catalog\s+)?category\s+is\s+)"
    r"([A-Za-z][A-Za-z0-9 &/+'-]{2,48}?)"
    r"(?:\s+category\b|(?=\s*[.,;]|\s*$))",
    re.I,
)
_FIRM_MATTERS_RE = re.compile(
    r"\bfirm\s+requirements?\b.{0,40}\bmatter|\bbut\s+the\s+firm\b",
    re.I,
)

# Generic English / shopping discourse tokens only — not suite wording.
_STOP = frozenset(
    """
    a an the and or but for with from that this these those into onto over under
    at to of in on by is are was were be been being it its as if then than so
    not no yes you your we our they their i me my need want find get buy purchase
    please can could would should must also just only more most less least very
    item items product products listing listings order orders cart shop shopping
    available stock stocked budget price prices within below above prefer preferred
    ideally soft firm requirement requirements category categories brand brands
    looking look help thanks thank hi hello there here about around
    something anything everything nothing somehow somewhere
    place test checkout choose pick confirm currently
    """.split()
)

_OBS_CAP = 3400
_STR_CAP = 320
_LIST_CAP = 8
_LEDGER_CAND_CAP = 36
_HISTORY_CAP = 16
_BOARD_SHOW = 6


def _sku_stem(sku: str) -> str:
    """Variant family stem: everything before the final '-suffix' when present."""
    text = str(sku or "").strip()
    if not text:
        return ""
    if "-" in text:
        return text.rsplit("-", 1)[0].lower()
    return text.lower()

# Number + nearby token windows — category-agnostic constraint atoms.
_NUM_NEAR_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z][A-Za-z0-9_+/-]{0,12})?"
    r"|"
    r"(?P<unit2>[A-Za-z][A-Za-z0-9_+/-]{0,12})\s*(?P<num2>\d+(?:\.\d+)?)",
)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1", "in_stock", "available"}:
            return True
        if text in {"false", "no", "n", "0", "out_of_stock", "unavailable", "sold_out"}:
            return False
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _product_blob(row: dict[str, Any], viewed: dict[str, Any] | None = None) -> str:
    parts: list[Any] = [row, viewed or {}]
    facts = (viewed or {}).get("facts") if isinstance((viewed or {}).get("facts"), dict) else {}
    if facts:
        parts.append(facts)
    return json.dumps(parts, ensure_ascii=False, default=str).lower()


def _constraint_atoms(query: str, notes: list[str] | None = None) -> list[str]:
    """Extract compact, category-agnostic requirement atoms from public text.

    Atoms are quoted phrases, decimal/integer figures, and figure+token pairs.
    No product-category vocabulary is used.
    """
    text = _preamble(query)
    if notes:
        text = f"{text} {' '.join(notes[-3:])}"
    # Drop price-change chatter so old/new listing prices never become spec atoms.
    text = re.sub(
        r"(?:more\s+expensive|price(?:\s+change)?|went\s+from|from|now|was|were|became|"
        r"increased\s+to|dropped\s+to|raised\s+to|updated\s+to|costs?)\s+"
        r"\d[\d,]*(?:\.\d+)?(?:\s*(?:to|-|→)\s*\d[\d,]*(?:\.\d+)?)?",
        " ",
        text,
        flags=re.I,
    )
    # Bare money amounts (two-decimal) are never product-spec atoms.
    text = re.sub(r"\b\d{2,}[.,]\d{2}\b", " ", text)
    atoms: list[str] = []
    seen: set[str] = set()

    def add(atom: str) -> None:
        key = atom.strip().lower()
        if len(key) < 2 or key in seen or key in _STOP:
            return
        seen.add(key)
        atoms.append(key)

    for match in re.finditer(r'"([^"\n]{2,80})"|“([^”\n]{2,80})”', text):
        add(match.group(1) or match.group(2) or "")

    budget = _parse_budget(text)
    budget_s = None
    if budget is not None:
        budget_s = str(int(budget)) if float(budget).is_integer() else str(budget)

    for match in _NUM_NEAR_RE.finditer(text):
        num = match.group("num") or match.group("num2")
        unit = match.group("unit") or match.group("unit2") or ""
        if not num:
            continue
        try:
            num_f = float(num.replace(",", ""))
        except ValueError:
            continue
        # Never treat the budget ceiling itself as a product-spec atom.
        if budget is not None and abs(num_f - budget) <= 1e-6:
            continue
        if budget_s and num.replace(",", "") == budget_s.replace(",", ""):
            continue
        unit_l = unit.lower().strip()
        if unit_l in {
            "php",
            "usd",
            "eur",
            "gbp",
            "cad",
            "aud",
            "jpy",
            "cny",
            "inr",
            "price",
            "budget",
        }:
            continue
        # Two-decimal magnitudes in money range are prices, not product specs.
        if num_f >= 20 and re.search(r"\.\d{2}$", num.replace(",", "")):
            continue
        # Large bare integers near money range without a unit are usually prices.
        if not unit_l and num_f >= 200 and "." not in num.replace(",", ""):
            continue
        # Stop-word "units" (a/to/the) are discourse, not measures — keep the figure.
        if unit_l in _STOP or len(unit_l) <= 1:
            add(num)
            # Compact figure+suffix forms that appear in listings (e.g. 1080p).
            if unit_l and unit_l not in _STOP and re.fullmatch(r"[a-z]", unit_l):
                add(f"{num}{unit_l}")
            continue
        if unit_l:
            add(f"{num} {unit_l}")
            add(num)
        else:
            add(num)

    # Schema-like snake_case tokens mentioned in the request (generic).
    for token in re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", text.lower()):
        if token not in {"product_id", "in_stock", "max_price"}:
            add(token)

    return atoms[:24]


def _atoms_hit(atoms: list[str], blob: str) -> tuple[int, int]:
    if not atoms:
        return 0, 0
    hits = 0
    blob_nums = set(_NUM_RE.findall(blob))
    blob_floats: list[float] = []
    for cand in blob_nums:
        try:
            blob_floats.append(float(cand))
        except ValueError:
            continue
    for atom in atoms:
        compact_atom = re.sub(r"\s+", "", atom)
        compact_blob = re.sub(r"\s+", "", blob)
        if atom in blob or (len(compact_atom) >= 2 and compact_atom in compact_blob):
            hits += 1
            continue
        # Figure atoms: exact token or near-equal float (7.6 vs 7.60).
        num_m = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\s*[a-z%][a-z0-9_+/-]*)?", atom)
        if num_m:
            try:
                target = float(num_m.group(1))
            except ValueError:
                continue
            if num_m.group(1) in blob_nums or any(abs(x - target) <= 1e-9 for x in blob_floats):
                hits += 1
    return hits, len(atoms)


def _moneyish_token(token: str) -> bool:
    """True for listing-price-like tokens that must never drive search queries."""
    raw = (token or "").strip().replace(",", "")
    if not raw:
        return False
    if re.fullmatch(r"\d+\.\d{2}", raw):
        try:
            return float(raw) >= 10.0
        except ValueError:
            return False
    if re.fullmatch(r"\d+(?:\.\d+)?", raw):
        try:
            return float(raw) >= 200.0
        except ValueError:
            return False
    return False


def _model_token_hit(token: str, blob: str) -> bool:
    """Match prefer/model codes without over-matching short digit fragments."""
    tok = (token or "").strip().lower()
    if not tok or not blob:
        return False
    if len(tok) <= 3 and any(ch.isdigit() for ch in tok):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", blob, re.I))
    return tok in blob


def _anchor_title_query(ledger: "Ledger") -> str | None:
    """Build a rediscovery query from the failed listing's own title tokens."""
    for pid in list(ledger.anchor_pids):
        viewed = ledger.viewed.get(pid) or {}
        title = str(viewed.get("title") or (ledger.candidates.get(pid) or {}).get("title") or "")
        if not title:
            # Also check candidate rows keyed by product::sku.
            for key, row in ledger.candidates.items():
                if key.startswith(f"{pid}::") or str(row.get("product_id") or "") == pid:
                    title = str(row.get("title") or "")
                    if title:
                        break
        if not title:
            continue
        toks = [t for t in _WORD_RE.findall(title) if t.lower() not in _STOP and len(t) >= 3]
        digs = [t for t in toks if any(ch.isdigit() for ch in t)]
        alphas = [t for t in toks if t.isalpha() and len(t) >= 4]
        parts: list[str] = []
        if alphas:
            parts.append(alphas[0])
        if digs:
            parts.append(digs[0])
        elif alphas[1:]:
            parts.append(alphas[1])
        q = " ".join(parts).strip()
        if q and q.lower() not in ledger.tried_queries:
            return q
    return None


def _value_attested(value: Any, blob: str) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        token = "true" if value else "false"
        return token in blob
    text = str(value).strip().lower()
    if len(text) < 1:
        return False
    if text in blob:
        return True
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 2 and compact in re.sub(r"\s+", "", blob):
        return True
    num = _as_float(value)
    if num is not None:
        for cand in _NUM_RE.findall(blob):
            try:
                if abs(float(cand) - num) <= 1e-9:
                    return True
            except ValueError:
                continue
    return False


# ---------------------------------------------------------------------------
# Provider / inference
# ---------------------------------------------------------------------------


def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()


def _model_ladder(hard: bool) -> list[str]:
    provider = _provider()
    catalog = _MODELS.get(provider, _MODELS["chutes"])
    override = (getenv("SANDBOX_MODEL") or "").strip()
    ordered: list[str] = []
    # Always lead with the strong allowlisted model for selection quality.
    ordered.append(catalog["strong"])
    if override:
        if provider == "openrouter" and override.endswith("-TEE"):
            ordered.append(_TEE_TO_OPENROUTER.get(override, catalog["fast"]))
        else:
            ordered.append(override)
    ordered.append(catalog["fast"])
    ordered.extend(catalog["fallback"])
    uniq: list[str] = []
    seen: set[str] = set()
    for model in ordered:
        if model and model not in seen:
            seen.add(model)
            uniq.append(model)
    return uniq


def _chat(
    *,
    models: list[str],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    last: Exception | None = None
    for model in models:
        try:
            result = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if result is None:
                raise RuntimeError("inference request failed")
            return result
        except Exception as exc:  # noqa: BLE001 — try next allowlisted model
            last = exc
            continue
    raise RuntimeError(f"inference request failed: {last}")


# ---------------------------------------------------------------------------
# Tool schema helpers
# ---------------------------------------------------------------------------


def _fn(spec: dict[str, Any]) -> dict[str, Any] | None:
    raw = spec.get("function") if isinstance(spec, dict) else None
    return raw if isinstance(raw, dict) else None


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {
        fn["name"]
        for spec in tools
        for fn in [_fn(spec)]
        if fn and isinstance(fn.get("name"), str)
    }


def _required_args(tools: list[dict[str, Any]], name: str) -> list[str]:
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get("name") != name:
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        return [x for x in (params.get("required") or []) if isinstance(x, str)]
    return []


def _tool_props(tools: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get("name") != name:
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        return props
    return {}


def _claim_fields(tools: list[dict[str, Any]]) -> set[str]:
    """Discover allowed justification field names from the live order schema only."""
    allowed: set[str] = set()
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get("name") != "place_test_order":
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
        has_enum = False
        if isinstance(enum, list):
            vals = [str(x) for x in enum if isinstance(x, str) and x.strip()]
            if vals:
                has_enum = True
                allowed.update(vals)
        # When the schema has no enum, accept snake_case identifiers mentioned in-schema.
        if not has_enum:
            blob = json.dumps(params, ensure_ascii=False).lower()
            for token in re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", blob):
                if token not in {
                    "product_id",
                    "tool_call",
                    "call_id",
                    "max_price",
                    "in_stock",
                }:
                    allowed.add(token)
        if "justification" in json.dumps(params, ensure_ascii=False).lower() or allowed:
            allowed.update({"price", "in_stock"})
    return allowed


def _open_spec_claims(tools: list[dict[str, Any]]) -> bool:
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get("name") != "place_test_order":
            continue
        blob = json.dumps(fn.get("parameters") or {}, ensure_ascii=False).lower()
        if "specification" in blob and "shopper" in blob:
            return True
        if "specification field" in blob:
            return True
    return False


def _soft_schema_fields(tools: list[dict[str, Any]], allowed: set[str]) -> set[str]:
    soft: set[str] = set()
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get("name") != "place_test_order":
            continue
        blob = json.dumps(fn.get("parameters") or {}, ensure_ascii=False).lower()
        for field in allowed:
            if field in {"price", "in_stock"}:
                continue
            surface = field.replace("_", " ")
            if field in blob and any(
                cue in blob for cue in ("reason", "capability", "positive", "why", "benefit")
            ):
                soft.add(field)
            if surface in blob and "positive" in blob:
                soft.add(field)
    return soft


# ---------------------------------------------------------------------------
# Text / mandate extraction (from public query only)
# ---------------------------------------------------------------------------


def _preamble(query: str) -> str:
    return (query or "").split("Task rules:", 1)[0].strip()


def _soft_tokens(text: str) -> list[str]:
    """Extract soft preference objects (brand/name tokens) from public goal text."""
    out: list[str] = []
    seen: set[str] = set()
    for match in _SOFT_OBJECT_RE.finditer(text or ""):
        raw = (match.group(1) or "").strip().rstrip(".,;:")
        if not raw or len(raw) < 2:
            continue
        key = raw.lower()
        if key in _STOP or key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out[:4]


def _prefer_model_tokens(text: str) -> list[str]:
    """Extract preferred model/code/color tokens from public goal text."""
    out: list[str] = []
    seen: set[str] = set()

    def add(part: str | None) -> None:
        if not part:
            return
        key = part.lower().strip(".,;:")
        if len(key) < 2 or key in seen or key in _STOP:
            return
        seen.add(key)
        out.append(part.strip(".,;:"))

    for match in _PREF_MODEL_RE.finditer(text or ""):
        add(match.group("color"))
        add(match.group("code"))
    # Also capture "model <code>" / "code <token>" discourse without category words.
    for match in re.finditer(
        r"\b(?:model|code|variant)\s+([A-Za-z]*\d{2,}[A-Za-z0-9-]*)\b",
        text or "",
        re.I,
    ):
        add(match.group(1))
    return out[:8]


def _goal_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r'"([^"\n]{2,80})"|“([^”\n]{2,80})”', text or ""):
        body = (match.group(1) or match.group(2) or "").strip()
        key = body.lower()
        if len(body) >= 2 and key not in seen:
            seen.add(key)
            phrases.append(body)
    for match in _PHRASE_GOAL_RE.finditer(text or ""):
        body = (match.group(1) or "").strip().strip("\"'“”")
        key = body.lower()
        if len(body) >= 2 and key not in seen:
            seen.add(key)
            phrases.append(body)
    return phrases[:4]


def _category_mentions(text: str) -> list[str]:
    out: list[str] = []
    for match in _CATEGORY_GOAL_RE.finditer(text or ""):
        label = (match.group(1) or "").strip()
        if label and label.lower() not in _STOP:
            out.append(label)
    return out[:3]


def _classify_procedure(
    query: str,
    *,
    tools: list[dict[str, Any]],
) -> str:
    """Pick one shopping procedure from public goal + tool schema cues only.

    Does not use product-category vocabularies or suite answer tables. Branching
    is driven by structural signals: named variants, deferred specs, soft vs
    firm discourse, quoted phrases, use-context language, and order schema.
    """
    goal = _preamble(query)
    scores: dict[str, int] = {
        PROC_ASK: 0,
        PROC_RECALL: 0,
        PROC_STRICT: 0,
        PROC_SOFT: 0,
        PROC_USE: 0,
        PROC_RANK: 0,
        PROC_ADAPT: 0,
        PROC_CLAIM: 0,
    }

    named = _named_variants(query)
    if len(named) >= 2:
        scores[PROC_RANK] += 6
    if _rank_axes(goal):
        scores[PROC_RANK] += 2

    if _needs_clarification(query):
        scores[PROC_ASK] += 5
        scores[PROC_STRICT] += 2

    needs_why = "justification" in _required_args(tools, "place_test_order")
    if needs_why or _WHY_GOAL_RE.search(goal):
        scores[PROC_CLAIM] += 5

    softs = _soft_tokens(goal)
    if softs and (_FIRM_MATTERS_RE.search(goal) or _FIRM_CUE_RE.search(goal)):
        scores[PROC_SOFT] += 5
    elif softs:
        scores[PROC_SOFT] += 3
    if _SOFT_CUE_RE.search(goal) and _FIRM_MATTERS_RE.search(goal):
        scores[PROC_SOFT] += 2

    phrases = _goal_phrases(goal)
    if phrases or _category_mentions(goal):
        scores[PROC_RECALL] += 4
    if re.search(r"\bmatches?\b", goal, re.I) and phrases:
        scores[PROC_RECALL] += 2

    if _USE_FIT_RE.search(goal) and not softs:
        scores[PROC_USE] += 4
    elif _USE_FIT_RE.search(goal):
        scores[PROC_USE] += 2

    if _ADAPT_GOAL_RE.search(goal):
        scores[PROC_ADAPT] += 4
    # Digit/code preference objects imply least-relax recovery, not brand-only soft.
    prefs = _prefer_model_tokens(goal)
    if any(any(ch.isdigit() for ch in p) for p in prefs):
        scores[PROC_ADAPT] += 6

    atoms = _constraint_atoms(query)
    if len(atoms) >= 2 and scores[PROC_SOFT] == 0 and scores[PROC_RECALL] == 0:
        scores[PROC_STRICT] += 2
    if not any(scores.values()):
        scores[PROC_STRICT] = 1

    # Stable priority when tied: rank → ask → claim → soft → adapt → use → recall → strict.
    order = (
        PROC_RANK,
        PROC_ASK,
        PROC_CLAIM,
        PROC_SOFT,
        PROC_ADAPT,
        PROC_USE,
        PROC_RECALL,
        PROC_STRICT,
    )
    best = max(scores.values())
    for proc in order:
        if scores[proc] == best:
            return proc
    return PROC_STRICT


def _procedure_playbook(proc: str) -> list[str]:
    """Short procedure-specific instructions for the deliberative model."""
    books: dict[str, list[str]] = {
        PROC_ASK: [
            "Procedure ask_gate: a firm specification is withheld until you ask.",
            "Send one focused message first; do not cart until the shopper answers.",
            "After clarification, treat the new atom as non-negotiable on the exact SKU.",
        ],
        PROC_RECALL: [
            "Procedure phrase_recall: prioritize the quoted/phrase tokens in search.",
            "Reformulate with keyword splits; survey several distinct listings before carting.",
            "Verify category_path against the requested category using observed path text.",
        ],
        PROC_STRICT: [
            "Procedure strict_match: every stated numeric/phrase atom must evidence on the SKU.",
            "Titles alone are insufficient — use view/compare facts for the selected variant.",
            "Reject listings that miss any firm atom even if cheaper.",
        ],
        PROC_SOFT: [
            "Procedure soft_prefer: firm constraints always beat soft wishes.",
            "Among fully compliant options, honor the named soft preference on the FIRST cart.",
            "Never buy a soft-matching item that fails a firm atom.",
        ],
        PROC_USE: [
            "Procedure use_fit: hard gates are category, budget, and stock.",
            "Rank compliant items by how well raw facts match the stated use context.",
            "Do not default to the cheapest compliant listing when a better use-fit exists.",
        ],
        PROC_RANK: [
            "Procedure axis_rank: compare only the named product_id::sku variants.",
            "Score by the stated axis order (primary, then secondary, then price).",
            "Cart the Pareto/lexicographic winner that is live in-stock within budget.",
        ],
        PROC_ADAPT: [
            "Procedure event_adapt: after cart or stock failure, remove and switch quickly.",
            "Prefer least change: same product other sku/option before unrelated listings.",
            "Always re-inspect_stock after cart events before place_test_order.",
        ],
        PROC_CLAIM: [
            "Procedure claim_why: view/compare the final candidate before the first cart.",
            "Checkout claims must use observed facts for the ordered SKU plus live price/stock.",
            "Never invent field values from the title alone.",
        ],
    }
    return books.get(proc, books[PROC_STRICT])


def _parse_budget(text: str) -> float | None:
    best: float | None = None
    for match in _BUDGET_RE.finditer(text or ""):
        raw = match.group("a") or match.group("b")
        if not raw:
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if value < 5:
            continue
        if best is None or value < best:
            best = value
    return best


def _parse_budget_update(text: str) -> float | None:
    """Budget updates from shopper notes — only when they explicitly revise the ceiling.

    Ignores incidental price figures in event notices (e.g. 'more expensive from
    4790 to 10618 PHP') so those cannot raise the shopping ceiling.
    """
    blob = text or ""
    # Drop pure price-change discourse before parsing.
    scrubbed = re.sub(
        r"(?:more\s+expensive|price(?:\s+change)?|went\s+from|from)\s+"
        r"\d[\d,]*(?:\.\d+)?\s*(?:to|-)?\s*\d[\d,]*(?:\.\d+)?\s*(?:php|usd|eur|gbp)?",
        " ",
        blob,
        flags=re.I,
    )
    if not re.search(
        r"\b(?:budget|spend|under|below|at\s+most|no\s+more\s+than|cap|ceiling|"
        r"can\s+only\s+(?:spend|go)|only\s+go\s+up\s+to|up\s+to|max(?:imum)?(?:\s+price)?)\b",
        scrubbed,
        re.I,
    ):
        return None
    return _parse_budget(scrubbed)


def _notes_have_firm_spec(notes: list[str], *, prior_atoms: list[str] | None = None) -> bool:
    """True when shopper notes already supply a concrete firm figure/phrase."""
    if not notes:
        return False
    blob = " ".join(notes[-3:])
    prior = {a.lower() for a in (prior_atoms or [])}
    fresh = _constraint_atoms(blob)
    for atom in fresh:
        if atom.lower() not in prior and any(ch.isdigit() for ch in atom):
            return True
    # Structural: imperative + figure in the same note (category-agnostic).
    if _NUM_RE.search(blob) and re.search(
        r"\b(?:required|must|need(?:s)?|exactly|still\s+needs?|spec)\b",
        blob,
        re.I,
    ):
        return True
    return bool(re.search(r'"[^"\n]{2,80}"', blob))


def _needs_clarification(query: str) -> bool:
    text = _preamble(query)
    if not text:
        return False
    if any(pat.search(text) for pat in _DEFER_ASK_RE):
        return True
    return False


def _named_variants(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _VARIANT_RE.finditer(text or ""):
        key = f"{match.group(1)}::{match.group(2)}"
        if key in seen:
            continue
        seen.add(key)
        found.append((match.group(1), match.group(2)))
    return found


def _rank_axes(text: str) -> list[tuple[str, int]]:
    """Parse comparative ranking axes: (normalized_axis, direction) where +1=higher, -1=lower."""
    axes: list[tuple[str, int]] = []
    seen: set[str] = set()
    dir_noise = _DIR_UP | _DIR_DOWN
    for match in _RANK_AXIS_RE.finditer(text or ""):
        direction_word = match.group("dir").lower()
        raw_axis = match.group("axis").strip().lower()
        # Drop leading direction words accidentally captured after "prefer higher X".
        parts = [p for p in raw_axis.split() if p not in dir_noise and p not in _STOP]
        if not parts:
            continue
        axis = re.sub(r"[^a-z0-9_]+", "", "_".join(parts))
        if len(axis) < 3 or axis in seen or axis in _STOP:
            continue
        direction = -1 if direction_word in _DIR_DOWN else 1
        if "price" in axis:
            direction = -1
        seen.add(axis)
        axes.append((axis, direction))
    return axes[:4]


def _goal_tokens(query: str, notes: list[str] | None = None) -> set[str]:
    text = _preamble(query)
    if notes:
        text = f"{text} {' '.join(notes[-3:])}"
    tokens = {t.lower() for t in _WORD_RE.findall(text) if len(t) > 1 and t.lower() not in _STOP}
    tokens.update(_NUM_RE.findall(text.lower()))
    return tokens


def _search_seeds(
    query: str,
    notes: list[str] | None = None,
    *,
    procedure: str = PROC_STRICT,
) -> list[str]:
    """Ladder of discovery queries from distinctive public tokens (specific → broader)."""
    chunks = [_preamble(query)]
    if notes:
        chunks.extend(notes[-2:])
    text = " ".join(chunks)
    seeds: list[str] = []
    phrases = _goal_phrases(text)
    for phrase in phrases:
        seeds.append(phrase)
        # Keyword splits for recall-style discovery (no suite synonym tables).
        parts = [p for p in _WORD_RE.findall(phrase) if p.lower() not in _STOP]
        if len(parts) >= 2:
            seeds.append(" ".join(parts))
        if len(parts) >= 3:
            seeds.append(" ".join(parts[:2]))
            seeds.append(parts[-1])
        for part in parts:
            if len(part) >= 4:
                seeds.append(part)

    cats = _category_mentions(text)
    for cat in cats:
        seeds.append(cat)

    noise = _STOP | _DIR_UP | _DIR_DOWN | {
        "usd",
        "eur",
        "gbp",
        "cad",
        "aud",
        "jpy",
        "cny",
        "inr",
        "php",
        "then",
        "than",
        "after",
        "before",
        "versus",
        "vs",
        "one",
        "done",
        "true",
        "false",
        "call",
        "tool",
    }
    modelish = [
        m.group(1)
        for m in re.finditer(r"\b([A-Za-z]*\d{2,}[A-Za-z0-9-]*)\b", text)
        if m.group(1).lower() not in noise and not m.group(1).isdigit()
    ]
    # Skip bare budget numbers as search tokens.
    budget = _parse_budget(text)
    budget_str = str(int(budget)) if budget is not None and float(budget).is_integer() else (
        str(budget) if budget is not None else None
    )
    words: list[str] = []
    seen: set[str] = set()
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9.+-]{2,}", text):
        low = tok.lower().strip(".,;:")
        if low in noise or low in seen:
            continue
        if budget_str and low == budget_str:
            continue
        seen.add(low)
        words.append(tok.strip(".,;:"))
        if len(words) >= 12:
            break
    alpha = [w for w in words if not re.search(r"\d", w)]
    mixed = [w for w in words if w not in alpha]
    softs = _soft_tokens(text)

    if procedure == PROC_SOFT and softs:
        primary = (softs[:1] + alpha[:3] + mixed[:2] + modelish[:2])[:6]
        seeds.append(" ".join(primary))
        if softs and alpha[:2]:
            seeds.append(f"{softs[0]} {' '.join(alpha[:2])}")
    elif procedure == PROC_RECALL:
        primary = (alpha[:3] + mixed[:2] + modelish[:2])[:6]
        if primary:
            seeds.append(" ".join(primary))
        if alpha[:2]:
            seeds.append(" ".join(alpha[:2]))
        if alpha:
            seeds.append(alpha[0])
    elif procedure == PROC_USE:
        # Prefer concrete use/context tokens + category mentions; skip stopwords.
        use_bits = [w for w in alpha if w.lower() not in _STOP][:5]
        primary = (use_bits[:3] + mixed[:2] + modelish[:1])[:6]
        if primary:
            seeds.append(" ".join(primary))
        if cats:
            seeds.append(" ".join((cats[:1] + use_bits[:2])))
        if use_bits[:2]:
            seeds.append(" ".join(use_bits[:2]))
    elif procedure == PROC_ADAPT:
        prefs = _prefer_model_tokens(text)
        colors = [p for p in prefs if p.isalpha()]
        codes = [p for p in prefs if any(ch.isdigit() for ch in p)]
        # Long digit codes first — they are the strongest least-relax anchors.
        for p in sorted(codes, key=lambda x: (-len(x), x)):
            seeds.append(p)
        if codes and colors:
            seeds.append(f"{codes[0]} {colors[0]}")
        if codes:
            seeds.append(" ".join(codes[:2]))
        if prefs:
            seeds.append(" ".join(prefs[:3]))
        primary = (alpha[:3] + mixed[:2] + modelish[:2])[:6]
        if primary:
            seeds.append(" ".join(primary))
    elif procedure == PROC_ASK:
        # After notes arrive, atom figures dominate rediscovery.
        atom_bits = [
            a
            for a in _constraint_atoms(query, notes)
            if any(ch.isdigit() for ch in a)
        ]
        softs = _soft_tokens(text)
        if softs and atom_bits:
            seeds.append(f"{softs[0]} {atom_bits[0]}")
        for atom in atom_bits[:3]:
            seeds.append(atom)
        primary = (softs[:1] + alpha[:3] + mixed[:2] + atom_bits[:2])[:6]
        if primary:
            seeds.append(" ".join(primary))
        if alpha[:2]:
            seeds.append(" ".join(alpha[:2]))
    else:
        primary = (alpha[:4] + mixed[:2] + modelish[:2])[:6]
        if primary:
            seeds.append(" ".join(primary))
        if len(primary) > 3:
            seeds.append(" ".join(primary[:3]))
        if alpha[:2]:
            seeds.append(" ".join(alpha[:2]))

    # Atom-informed seeds: figure+unit pairs from the goal help strict matching.
    for atom in _constraint_atoms(query, notes):
        if " " in atom and any(ch.isdigit() for ch in atom):
            seeds.append(atom)
            if softs:
                seeds.append(f"{softs[0]} {atom}")

    out: list[str] = []
    seen_s: set[str] = set()
    for s in seeds:
        key = s.lower().strip()
        if not key or key in seen_s:
            continue
        seen_s.add(key)
        out.append(s.strip())
    return out or ["in stock"]


def _fields_in_text(query: str, allowed: set[str], extra: str = "") -> set[str]:
    if not allowed:
        return set()
    q = f"{_preamble(query)} {extra}".lower()
    q_norm = re.sub(r"[^a-z0-9]+", " ", q)
    compact = q_norm.replace(" ", "")
    hits: set[str] = set()
    for field in allowed:
        if field in {"price", "in_stock"}:
            continue
        surface = field.replace("_", " ")
        if field in q_norm or surface in q or surface.replace(" ", "") in compact:
            hits.add(field)
    return hits


def _overlap(value: Any, goal: set[str]) -> int:
    if value in (None, ""):
        return 0
    text = str(value).lower()
    tokens = set(_WORD_RE.findall(text))
    score = len(tokens & goal)
    for num in _NUM_RE.findall(text):
        if num in goal or any(num in g or g in num for g in goal if any(ch.isdigit() for ch in g)):
            score += 2
    return score


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _keys_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4:
        return a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a)
    return False


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    """Merge nested observation fact blocks without inventing attributes."""
    flat = dict(row)
    for nest in ("facts", "specification", "attributes", "specs"):
        block = row.get(nest)
        if isinstance(block, dict):
            for key, value in block.items():
                flat.setdefault(key, value)
    path = row.get("category_path")
    if isinstance(path, list) and path:
        leaf = path[-1] if isinstance(path[-1], str) else path[0]
        if isinstance(leaf, str) and leaf.strip():
            flat.setdefault("category_path_leaf", leaf)
    elif isinstance(path, str) and path.strip():
        flat.setdefault("category_path_leaf", path.split(",")[-1].strip().strip("[]\"' "))
    return flat


def _lookup(blobs: list[dict[str, Any]], field: str) -> Any:
    needle = _norm_key(field)
    if not needle:
        return None
    best: tuple[int, Any] | None = None
    for blob in blobs:
        stack: list[Any] = [blob]
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            for key, value in node.items():
                if value in (None, ""):
                    continue
                kn = _norm_key(key)
                if _keys_match(kn, needle):
                    rank = abs(len(kn) - len(needle))
                    if best is None or rank < best[0]:
                        best = (rank, value)
                if isinstance(value, dict):
                    stack.append(value)
                elif isinstance(value, list):
                    stack.extend(item for item in value if isinstance(item, dict))
    return None if best is None else best[1]


def _goal_nums_near(query: str, field: str, notes: list[str] | None = None) -> set[str]:
    text = _preamble(query).lower()
    if notes:
        text = f"{text} {' '.join(notes[-3:]).lower()}"
    surfaces = {
        field.lower(),
        field.replace("_", " ").lower(),
        _norm_key(field),
        *[p for p in field.lower().split("_") if len(p) >= 4],
    }
    hits: set[str] = set()
    for surface in surfaces:
        if len(surface) < 3:
            continue
        for match in re.finditer(re.escape(surface), text):
            window = text[max(0, match.start() - 32) : match.end() + 32]
            hits.update(_NUM_RE.findall(window))
    return hits


# ---------------------------------------------------------------------------
# JSON / observation helpers
# ---------------------------------------------------------------------------


def _repair_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    fixed = text.replace("'", '"')
    fixed = re.sub(r",\s*}", "}", fixed)
    fixed = re.sub(r",\s*]", "]", fixed)
    try:
        parsed = json.loads(fixed)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("function", {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        return _repair_json(raw)
    return {}


def _compact(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= _STR_CAP else value[: _STR_CAP - 3] + "..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_compact(item, depth + 1) for item in value[:_LIST_CAP]]
    if isinstance(value, dict):
        return {str(k): _compact(v, depth + 1) for k, v in value.items()}
    return str(value)[:200]


def _compact_obs(observation: dict[str, Any]) -> str:
    text = json.dumps(_compact(observation), ensure_ascii=False, default=str)
    return text if len(text) <= _OBS_CAP else text[: _OBS_CAP - 3] + "..."


def _unwrap_observation(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split envelope ``{done, error, observation: {...}}`` from the inner payload.

    Runtime tools v5 nest the tool body under ``observation``. The ledger must
    read the inner payload; ``done`` / ``error`` stay on the envelope.
    """
    if not isinstance(raw, dict):
        return {}, {"raw": raw}
    envelope = dict(raw)
    inner = envelope.get("observation")
    if isinstance(inner, dict) and (
        "results" in inner
        or "compared" in inner
        or "product_id" in inner
        or "variants" in inner
        or "facts" in inner
        or "action" in inner
        or "items" in inner
        or "cart" in inner
        or "price" in inner
        or "in_stock" in inner
    ):
        payload = dict(inner)
    elif isinstance(inner, dict) and not any(
        k in envelope for k in ("results", "product_id", "compared", "price", "in_stock")
    ):
        # Nested body without obvious keys — still prefer inner when envelope is meta-only.
        meta_keys = {"done", "error", "observation", "ok", "status"}
        if set(envelope.keys()) <= meta_keys | {"observation"}:
            payload = dict(inner)
        else:
            payload = {k: v for k, v in envelope.items() if k not in {"done", "error"}}
    else:
        payload = {k: v for k, v in envelope.items() if k not in {"done", "error"}}
        if "observation" in payload and not isinstance(payload.get("observation"), dict):
            pass
        elif "observation" in payload and isinstance(payload["observation"], dict):
            # Flat-ish: merge inner fields without clobbering siblings.
            nested = payload.pop("observation")
            for key, value in nested.items():
                payload.setdefault(key, value)
    return envelope, payload


# ---------------------------------------------------------------------------
# Episode ledger
# ---------------------------------------------------------------------------


class Ledger:
    """Mutable world state derived only from the public goal and observations."""

    def __init__(
        self,
        query: str,
        allowed_claims: set[str],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.query = query
        self.allowed_claims = set(allowed_claims)
        self.budget = _parse_budget(_preamble(query))
        self.needs_ask = _needs_clarification(query)
        self.named = _named_variants(query)
        self.rank_axes = _rank_axes(_preamble(query))
        self.firm_signal = bool(_FIRM_CUE_RE.search(_preamble(query)))
        self.soft_signal = bool(_SOFT_CUE_RE.search(_preamble(query)))
        self.soft_tokens = _soft_tokens(_preamble(query))
        self.phrases = _goal_phrases(_preamble(query))
        self.categories = _category_mentions(_preamble(query))
        self.prefer_models = _prefer_model_tokens(_preamble(query))
        self.procedure = _classify_procedure(query, tools=tools or [])
        # Ask-gate procedure always requires clarification before carting.
        if self.procedure == PROC_ASK:
            self.needs_ask = True
        self.mentioned = _fields_in_text(query, allowed_claims)
        self.atoms = _constraint_atoms(query)
        self.seeds = _search_seeds(query, procedure=self.procedure)
        self.seed_idx = 0
        self.notes: list[str] = []
        self.candidates: dict[str, dict[str, Any]] = {}
        self.viewed: dict[str, dict[str, Any]] = {}
        self.stock: dict[str, dict[str, Any]] = {}
        self.cart: list[dict[str, str]] = []
        self.actions: list[str] = []
        self.last_error: str | None = None
        self.pending_q = False
        self.asked = False
        self.clarified = False
        self.needs_recheck = False
        self.checked: set[str] = set()
        self.must_recover = False
        self.search_n = 0
        self.empty_searches = 0
        self.view_n = 0
        self.cart_n = 0
        self.failed: set[str] = set()
        self.tried_queries: set[str] = set()
        self.same_search_streak = 0
        self.last_search_query = ""
        self.identical_action_streak = 0
        self.last_action_stamp = ""
        self.post_cart_pending: set[str] = set()
        self.priority_asked = False
        self.goal_atoms = list(self.atoms)
        self.post_clarify_search = False
        self.initial_cart_failed = False
        self.anchor_stems: set[str] = set()
        self.anchor_pids: set[str] = set()
        self.anchor_prices: list[float] = []
        self.event_target_key: str | None = None
        self.post_event_rerank_done = False

    def _refresh(self) -> None:
        blob = " ".join(self.notes[-3:])
        updated = _parse_budget_update(blob)
        if updated is not None:
            # Notes may tighten a ceiling; never raise it from event chatter.
            if self.budget is None or updated <= self.budget + 1e-9:
                self.budget = updated
                # A tightened ceiling invalidates an over-budget cart immediately.
                if self.cart:
                    top = self.cart[-1]
                    if self.over_budget(top["product_id"], top["sku"]):
                        self.must_recover = True
                        self.initial_cart_failed = True
        self.mentioned = _fields_in_text(self.query, self.allowed_claims, extra=blob)
        self.atoms = _constraint_atoms(self.query, self.notes)
        self.soft_tokens = _soft_tokens(f"{_preamble(self.query)} {blob}")
        self.phrases = _goal_phrases(f"{_preamble(self.query)} {blob}")
        self.prefer_models = _prefer_model_tokens(f"{_preamble(self.query)} {blob}")
        self.seeds = _search_seeds(self.query, self.notes, procedure=self.procedure)
        self.rank_axes = _rank_axes(f"{_preamble(self.query)} {blob}")
        for pid, sku in _named_variants(blob):
            key = f"{pid}::{sku}"
            if key not in {f"{a}::{b}" for a, b in self.named}:
                self.named.append((pid, sku))
        # Once notes carry a firm figure, stop asking and rediscover.
        if self.needs_ask and _notes_have_firm_spec(self.notes, prior_atoms=self.goal_atoms):
            self.clarified = True
            self.needs_ask = False
            if not self.post_clarify_search:
                self.seed_idx = 0
                self.tried_queries.clear()
                self.same_search_streak = 0
                self.post_clarify_search = True
                self.empty_searches = 0

    def note_shopper(self, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        self.notes.append(text[:500])
        self.notes = self.notes[-5:]
        self._refresh()
        # Availability pings are action requests, not new preference questions.
        avail_ping = bool(
            re.search(
                r"\bstill available\b|\bdo you (?:still )?have\b|\bcheck(?:\s+again)?\b|"
                r"\bis that (?:still )?(?:available|in stock)\b",
                text,
                re.I,
            )
        )
        if "?" in text and not avail_ping:
            self.pending_q = True
        elif avail_ping and self.cart:
            self.needs_recheck = True
            self.post_cart_pending.update(item["key"] for item in self.cart)
            self.pending_q = False
        if self.asked and (
            self.clarified or _notes_have_firm_spec(self.notes, prior_atoms=self.goal_atoms)
        ):
            self.clarified = True
            self.needs_ask = False
        price_event = bool(
            re.search(
                r"more\s+expensive|price\s+change|went\s+(?:up|from)|re-?check before ordering",
                text,
                re.I,
            )
        )
        if self.cart:
            # Remember over-budget / failed live state BEFORE invalidating stock cache.
            known_over = self.must_recover or any(
                self.over_budget(item["product_id"], item["sku"]) for item in self.cart
            )
            self.needs_recheck = True
            self.checked.clear()
            for item in self.cart:
                self.stock.pop(item["key"], None)
                self.stock.pop(item["product_id"], None)
                self.post_cart_pending.add(item["key"])
            if known_over or price_event:
                self.must_recover = True
                self.initial_cart_failed = True
            # Never clear must_recover on chat — shopper notes must not unwind a failed check.

    def remember(self, name: str, args: dict[str, Any]) -> None:
        self.actions.append(name)
        self.actions = self.actions[-30:]
        stamp = name + ":" + json.dumps(args, sort_keys=True, default=str)[:180]
        if stamp == self.last_action_stamp:
            self.identical_action_streak += 1
        else:
            self.identical_action_streak = 1
            self.last_action_stamp = stamp
        if name in {"search", "filter"}:
            self.search_n += 1
            if self.post_clarify_search and self.clarified:
                self.post_clarify_search = False
            q = str(args.get("query") or "").strip().lower()
            if q:
                if q == self.last_search_query:
                    self.same_search_streak += 1
                else:
                    self.same_search_streak = 1
                    self.last_search_query = q
                self.tried_queries.add(q)
        if name in {"view", "compare"}:
            self.view_n += 1
        if name == "message":
            self.asked = True
            self.pending_q = False
            self.priority_asked = True
        if name == "add_to_cart":
            self.cart_n += 1
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and sku:
                key = f"{pid}::{sku}"
                self.cart = [c for c in self.cart if c.get("key") != key]
                self.cart.append({"product_id": pid, "sku": sku, "key": key})
                stem = _sku_stem(sku)
                if stem:
                    self.anchor_stems.add(stem)
                self.anchor_pids.add(pid)
                known_price = _as_float(
                    (self.candidates.get(key) or {}).get("price")
                    or (self.viewed.get(pid) or {}).get("price")
                )
                if known_price is not None and known_price > 0:
                    self.anchor_prices.append(known_price)
                    self.anchor_prices = self.anchor_prices[-8:]
                if self.event_target_key is None:
                    self.event_target_key = key
                # Market events often fire on cart — invalidate prior stock and
                # require a fresh inspect_stock before any place_test_order.
                self.stock.pop(key, None)
                self.stock.pop(pid, None)
                self.checked.discard(key)
                self.needs_recheck = True
                self.must_recover = False
                self.post_cart_pending.add(key)
        if name == "remove_from_cart":
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            key = f"{pid}::{sku}"
            if key:
                self.failed.add(key)
                stem = _sku_stem(sku)
                if stem:
                    self.anchor_stems.add(stem)
                if pid:
                    self.anchor_pids.add(pid)
                self.event_target_key = key
            self.cart = [c for c in self.cart if c.get("key") != key]
            self.must_recover = False
            self.needs_recheck = False
            self.checked.discard(key)
            self.post_cart_pending.discard(key)
        if name == "place_test_order":
            self.identical_action_streak = 0

    def _store(self, row: dict[str, Any], *, viewed: bool = False) -> None:
        pid = str(row.get("product_id") or "")
        if not pid:
            return
        sku = str(row.get("sku") or "")
        key = f"{pid}::{sku}" if sku else pid
        prev = self.candidates.get(key, {})
        merged = {**prev, **{k: v for k, v in row.items() if v is not None}}
        merged["product_id"] = pid
        if sku:
            merged["sku"] = sku
        self.candidates[key] = merged
        if viewed:
            slim = {
                k: merged.get(k)
                for k in (
                    "product_id",
                    "sku",
                    "title",
                    "brand",
                    "price",
                    "in_stock",
                    "category_path",
                    "facts",
                    "listing_id",
                )
                if merged.get(k) is not None
            }
            self.viewed[pid] = {**self.viewed.get(pid, {}), **slim}
        if len(self.candidates) > _LEDGER_CAND_CAP:
            ranked = sorted(
                self.candidates.items(),
                key=lambda kv: (
                    0 if _sku_stem(str(kv[1].get("sku") or "")) in self.anchor_stems else 1,
                    0 if str(kv[1].get("product_id") or "") in self.anchor_pids else 1,
                    0 if kv[1].get("product_id") in self.viewed else 1,
                    0 if (kv[0] in self.stock or kv[1].get("product_id") in self.stock) else 1,
                    float(kv[1]["price"]) if isinstance(kv[1].get("price"), (int, float)) else 1e18,
                ),
            )
            keep = {k for k, _ in ranked[:_LEDGER_CAND_CAP]}
            self.candidates = {k: v for k, v in self.candidates.items() if k in keep}

    def ingest(self, name: str, args: dict[str, Any], observation: dict[str, Any]) -> None:
        if not isinstance(observation, dict):
            return
        envelope, payload = _unwrap_observation(observation)
        if envelope.get("error") or payload.get("error"):
            self.last_error = str(envelope.get("error") or payload.get("error"))[:240]
            return
        if name in {"search", "filter"}:
            rows = [r for r in (payload.get("results") or []) if isinstance(r, dict)]
            if not rows:
                self.empty_searches += 1
                self.seed_idx = min(self.seed_idx + 1, max(0, len(self.seeds) - 1))
            else:
                self.empty_searches = 0
            for row in rows:
                self._store(row)
        elif name == "compare":
            for row in payload.get("compared") or payload.get("results") or []:
                if isinstance(row, dict):
                    self._store(row, viewed=True)
        elif name == "view":
            self._store(payload, viewed=True)
            pid = str(payload.get("product_id") or args.get("product_id") or "")
            if pid:
                facts = payload.get("facts") if isinstance(payload.get("facts"), dict) else {}
                self.viewed[pid] = {
                    **self.viewed.get(pid, {}),
                    "product_id": pid,
                    "title": payload.get("title"),
                    "brand": payload.get("brand"),
                    "category_path": payload.get("category_path"),
                    "facts": facts,
                    "listing_id": payload.get("listing_id"),
                    "price": payload.get("price"),
                    "in_stock": payload.get("in_stock"),
                }
            for row in (payload.get("variants") or [])[:16]:
                if isinstance(row, dict):
                    self._store({**row, "product_id": row.get("product_id") or pid})
            if pid:
                variants = [v for v in (payload.get("variants") or []) if isinstance(v, dict)]
                if variants:
                    self.viewed.setdefault(pid, {})
                    self.viewed[pid]["variants"] = variants[:16]
            # Link variant stems only for products we already committed to (cart/event).
            # Do NOT promote every listing that merely mentions a prefer-code — that
            # pollutes anchors with accessories and unrelated catalog noise.
            parent_stem = _sku_stem(str(payload.get("sku") or ""))
            parent_anchor = bool(
                pid in self.anchor_pids
                or (parent_stem and parent_stem in self.anchor_stems)
            )
            if parent_anchor:
                for row in (payload.get("variants") or [])[:16]:
                    if isinstance(row, dict):
                        vst = _sku_stem(str(row.get("sku") or ""))
                        if vst:
                            self.anchor_stems.add(vst)
                        vpid = str(row.get("product_id") or pid or "")
                        if vpid:
                            self.anchor_pids.add(vpid)
                        vsku = str(row.get("sku") or "")
                        if vpid and vsku:
                            self._store({**row, "product_id": vpid, "sku": vsku})
        elif name == "inspect_stock":
            pid = str(payload.get("product_id") or args.get("product_id") or "")
            sku = str(payload.get("sku") or args.get("sku") or "")
            if pid:
                stock_body = {
                    k: v
                    for k, v in payload.items()
                    if k not in {"done", "error", "observation", "action"}
                }
                key = f"{pid}::{sku}"
                self.stock[key] = stock_body
                self.stock[pid] = stock_body
                self._store({"product_id": pid, "sku": sku, **stock_body})
                self.checked.add(key)
                live_price = _as_float(stock_body.get("price"))
                if live_price and any(item.get("key") == key for item in self.cart):
                    self.anchor_prices.append(live_price)
                    self.anchor_prices = self.anchor_prices[-8:]
                # Fresh observation satisfies the post-cart inspect requirement.
                self.post_cart_pending.discard(key)
                carted = any(
                    item.get("key") == key or item.get("product_id") == pid
                    for item in self.cart
                )
                if carted or self.needs_recheck:
                    if self.live_ok(pid, sku):
                        self.needs_recheck = False
                        self.must_recover = False
                    else:
                        self.needs_recheck = False
                        self.must_recover = True
                        self.failed.add(key)
                        stem = _sku_stem(sku)
                        if stem:
                            self.anchor_stems.add(stem)
                        if pid:
                            self.anchor_pids.add(pid)
                        self.event_target_key = key
                        if self.cart_n <= 1:
                            self.initial_cart_failed = True
        elif name == "inspect_cart":
            rebuilt: list[dict[str, str]] = []
            stack: list[Any] = [payload]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    pid = str(node.get("product_id") or "")
                    sku = str(node.get("sku") or "")
                    if pid and sku:
                        rebuilt.append({"product_id": pid, "sku": sku, "key": f"{pid}::{sku}"})
                    stack.extend(node.values())
                elif isinstance(node, list):
                    stack.extend(node)
            if rebuilt:
                seen: set[str] = set()
                uniq: list[dict[str, str]] = []
                for item in rebuilt:
                    if item["key"] in seen:
                        continue
                    seen.add(item["key"])
                    uniq.append(item)
                self.cart = uniq

    def live_ok(self, product_id: str, sku: str) -> bool:
        key = f"{product_id}::{sku}"
        if key in self.post_cart_pending:
            return False
        stock = self.stock.get(key) or self.stock.get(product_id)
        if not isinstance(stock, dict):
            return False
        flagged = _as_bool(stock.get("in_stock"))
        if flagged is not True:
            return False
        price = _as_float(stock.get("price"))
        if self.budget is not None:
            if price is None:
                return False
            if price > self.budget + 1e-9:
                return False
        return True

    def over_budget(self, product_id: str, sku: str) -> bool:
        if self.budget is None:
            return False
        key = f"{product_id}::{sku}"
        stock = self.stock.get(key) or self.stock.get(product_id)
        if not isinstance(stock, dict):
            return False
        price = _as_float(stock.get("price"))
        return price is not None and price > self.budget + 1e-9

    def ready_order(self) -> dict[str, str] | None:
        if self.must_recover or self.needs_recheck or not self.cart:
            return None
        top = self.cart[-1]
        if top["key"] in self.post_cart_pending:
            return None
        if not self.live_ok(top["product_id"], top["sku"]):
            return None
        if self.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM}:
            if not self.constraints_ok(top["product_id"], top["sku"]):
                return None
        return {"product_id": top["product_id"], "sku": top["sku"]}

    def evidence_ratio(self, row: dict[str, Any], viewed: dict[str, Any] | None = None) -> float:
        atoms = self.atoms
        if not atoms:
            return 1.0
        blob = _product_blob(row, viewed)
        hits, total = _atoms_hit(atoms, blob)
        return hits / max(total, 1)

    def _soft_hit(self, row: dict[str, Any], viewed: dict[str, Any] | None = None) -> bool:
        if not self.soft_tokens:
            return False
        blob = _product_blob(row, viewed)
        return any(tok.lower() in blob for tok in self.soft_tokens)

    def _phrase_hit(self, row: dict[str, Any], viewed: dict[str, Any] | None = None) -> bool:
        if not self.phrases:
            return False
        blob = _product_blob(row, viewed)
        for phrase in self.phrases:
            pl = phrase.lower()
            if pl in blob:
                return True
            parts = [p.lower() for p in _WORD_RE.findall(phrase) if p.lower() not in _STOP]
            if parts and all(p in blob for p in parts):
                return True
        return False

    def _category_hit(self, row: dict[str, Any], viewed: dict[str, Any] | None = None) -> bool:
        if not self.categories:
            return True
        path = (viewed or {}).get("category_path")
        if path is None:
            path = row.get("category_path")
        path_text = json.dumps(path, ensure_ascii=False, default=str).lower() if path else ""
        if not path_text or path_text in {"null", "[]", '""'}:
            return False
        for cat in self.categories:
            tokens = [t.lower() for t in _WORD_RE.findall(cat) if t.lower() not in _STOP and len(t) > 2]
            if tokens and any(t in path_text for t in tokens):
                return True
        return False

    def commitment_price(self) -> float | None:
        """Last carted listing price — the class we should stay near after events."""
        refs = [p for p in self.anchor_prices if p and p > 0]
        if refs:
            return max(refs)
        return None

    def price_class_ok(
        self,
        price_f: float | None,
        product_id: str | None = None,
        sku: str | None = None,
    ) -> bool:
        """Reject substitutes that are far cheaper than the item we already committed to.

        Generic: a replacement under a tightened ceiling should still be the same
        order of magnitude as the last cart, not an unrelated low-price listing.
        Same-product / same-SKU-stem variants may be cheaper and stay eligible.
        """
        if price_f is None:
            return True
        if product_id and product_id in self.anchor_pids:
            return True
        stem = _sku_stem(sku or "")
        if stem and stem in self.anchor_stems:
            return True
        if not (self.failed or self.initial_cart_failed or self.cart_n > 0):
            return True
        ref = self.commitment_price()
        if ref is None:
            return True
        floor = 0.35 * ref
        if self.budget is not None:
            floor = min(floor, 0.35 * self.budget)
        return price_f + 1e-9 >= floor

    def evidence_need(self) -> float:
        if self.procedure == PROC_ASK:
            return 0.75 if len(self.atoms) <= 3 else 0.6
        if self.procedure in {PROC_STRICT, PROC_CLAIM}:
            return 0.67 if len(self.atoms) <= 3 else 0.5
        if self.procedure == PROC_SOFT:
            return 0.5
        if self.procedure == PROC_RECALL:
            return 0.34
        if self.procedure in {PROC_USE, PROC_ADAPT}:
            return 0.34
        return 0.5

    def constraints_ok(self, product_id: str, sku: str) -> bool:
        """True when public constraint atoms are evidenced on the viewed listing."""
        if not self.atoms and self.procedure not in {PROC_RECALL}:
            return True
        key = f"{product_id}::{sku}"
        row = self.candidates.get(key) or self.candidates.get(product_id) or {}
        viewed = self.viewed.get(product_id) or {}
        if not viewed and product_id not in self.viewed:
            return False
        if self.procedure == PROC_RECALL:
            if self.phrases and not self._phrase_hit(row, viewed):
                return False
            if self.categories and not self._category_hit(row, viewed):
                return False
            return True
        ratio = self.evidence_ratio(row, viewed)
        return ratio + 1e-9 >= self.evidence_need()

    def _axis_score(self, row: dict[str, Any], viewed: dict[str, Any]) -> float:
        if not self.rank_axes:
            return 0.0
        product = _flatten({**row, **viewed})
        total = 0.0
        for idx, (axis, direction) in enumerate(self.rank_axes):
            weight = 4.0 - idx
            observed = _lookup([product, viewed, row], axis)
            if observed is None:
                # Try numeric tokens from facts that share axis stem.
                facts = viewed.get("facts") if isinstance(viewed.get("facts"), dict) else {}
                for key, value in {**product, **facts}.items():
                    if not isinstance(key, str):
                        continue
                    if _keys_match(_norm_key(key), _norm_key(axis)):
                        observed = value
                        break
            if observed is None:
                continue
            try:
                num = float(str(observed).replace(",", ""))
            except ValueError:
                # Non-numeric: reward textual presence for "higher/prefer" axes.
                total += weight * (0.5 if direction > 0 else 0.0)
                continue
            total += weight * direction * num
        return total

    def ranked(self) -> list[tuple[float, str, dict[str, Any]]]:
        goal = _goal_tokens(self.query, self.notes)
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        recovering = bool(self.failed) or self.initial_cart_failed or self.cart_n > 0
        use_blob = f"{_preamble(self.query)} {' '.join(self.notes[-2:])}".lower()
        use_tokens = {
            t.lower()
            for t in _WORD_RE.findall(use_blob)
            if t.lower() not in _STOP and len(t) > 3
        }
        # Prefer model/code tokens for adapt; keep distinct from generic goal tokens.
        model_toks = [t.lower() for t in self.prefer_models]
        code_toks = [t for t in model_toks if any(ch.isdigit() for ch in t)]
        color_toks = [t for t in model_toks if t.isalpha() and len(t) >= 3]
        for key, row in self.candidates.items():
            if key in self.failed:
                continue
            score = 0.0
            pid = str(row.get("product_id") or "")
            sku = str(row.get("sku") or "")
            stem = _sku_stem(sku)
            viewed = self.viewed.get(pid) or {}
            if pid in self.viewed:
                score += 4
            if key in self.stock or pid in self.stock:
                score += 3
                if self.live_ok(pid, sku):
                    score += 4
                else:
                    score -= 5
            if row.get("in_stock") is True:
                score += 1
            price = row.get("price")
            if any(pid == n[0] and (not sku or sku == n[1]) for n in self.named):
                score += 8 if self.procedure == PROC_RANK else 5
            elif any(pid == n[0] for n in self.named):
                score += 3 if self.procedure == PROC_RANK else 2
            title = str(row.get("title") or "")
            score += min(4, _overlap(title, goal))
            # Numeric / atom evidence from the public goal against listing text.
            ev = self.evidence_ratio(row, viewed)
            if self.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM}:
                score += 10.0 * ev
                if ev + 1e-9 < self.evidence_need():
                    score -= 6
            else:
                score += 6.0 * ev
            goal_nums = {t for t in goal if any(ch.isdigit() for ch in t)}
            title_nums = set(_NUM_RE.findall(title.lower()))
            if goal_nums and title_nums & goal_nums:
                score += 2
            facts = viewed.get("facts") if isinstance(viewed.get("facts"), dict) else {}
            for field in self.mentioned:
                observed = _lookup([viewed, row, facts], field)
                if observed is None:
                    continue
                score += 2 + min(3, _overlap(observed, goal))
            score += self._axis_score(row, viewed)

            if self.procedure == PROC_SOFT and self._soft_hit(row, viewed):
                score += 9
            elif self.procedure == PROC_SOFT and self.soft_tokens and pid in self.viewed:
                score -= 2
            # Soft brand still helps after ask-gate clarification among firm matches.
            if self.procedure == PROC_ASK and self.soft_tokens and self._soft_hit(row, viewed):
                score += 6

            if self.procedure == PROC_RECALL:
                if self._phrase_hit(row, viewed):
                    score += 10
                else:
                    score -= 4
                if self.categories:
                    score += 5 if self._category_hit(row, viewed) else -6

            if (
                self.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM}
                and self.categories
                and pid in self.viewed
            ):
                score += 6 if self._category_hit(row, viewed) else -12

            if self.procedure == PROC_USE and pid in self.viewed:
                blob = _product_blob(row, viewed)
                fit = sum(1 for t in use_tokens if t in blob)
                score += min(10, fit * 1.5)
                # Observation-driven: boost measurable facts whose keys appear in the use text.
                fact_items = facts.items() if isinstance(facts, dict) else ()
                present_keys = {str(fk).lower() for fk, _fv in fact_items}
                for fk, fv in fact_items:
                    fkl = str(fk).lower()
                    if not any(t in fkl for t in use_tokens if len(t) > 3) and fkl not in use_blob:
                        continue
                    score += 4
                    fv_l = str(fv).lower()
                    if any(t in fv_l for t in use_tokens if len(t) > 3):
                        score += 3
                    num = _as_float(fv)
                    if num is not None and num > 0:
                        score += min(6.0, (num ** 0.5) / 12.0)
                # Comparative: keys that other viewed listings use for this goal.
                relevant_keys: set[str] = set()
                for other in self.viewed.values():
                    ofacts = other.get("facts") if isinstance(other.get("facts"), dict) else {}
                    for ok in ofacts:
                        okl = str(ok).lower()
                        if okl in use_blob or any(t in okl for t in use_tokens if len(t) > 3):
                            relevant_keys.add(okl)
                for rk in relevant_keys:
                    if rk in present_keys:
                        score += 4
                    else:
                        score -= 3
                title_hits = sum(1 for t in use_tokens if t in title.lower())
                score += min(6, title_hits)

            if self.procedure == PROC_ADAPT or recovering:
                # Least-relax: same variant stem, then same model code, color last.
                if stem and stem in self.anchor_stems:
                    score += 14
                if pid in self.anchor_pids:
                    score += 5
                for failed_key in self.failed:
                    if failed_key.startswith(f"{pid}::"):
                        score += 6
                        break
                    failed_sku = failed_key.split("::", 1)[-1]
                    if stem and stem == _sku_stem(failed_sku):
                        score += 12
                        break
                blob = _product_blob(row, viewed)
                code_hits = sum(1 for tok in code_toks if _model_token_hit(tok, blob))
                color_hits = sum(1 for tok in color_toks if tok in blob)
                if code_hits and color_hits:
                    score += 12
                elif code_hits:
                    score += 8
                elif color_hits:
                    score += 3
                elif code_toks and recovering:
                    score -= 4
                score += 3 * sum(1 for tok in model_toks if _model_token_hit(tok, blob))

            if self.procedure == PROC_CLAIM and pid in self.viewed and facts:
                score += 3
                # Prefer listings whose facts cover more goal numbers/tokens.
                blob = _product_blob(row, viewed)
                score += min(4, sum(1 for t in goal_nums if t in blob))

            # Structural accessory demotion: titles that look like "X for Y" attachments.
            # Only when recovering with a model code — never demote ordinary "for <use>" titles.
            title_l = title.lower()
            if (self.procedure == PROC_ADAPT or recovering) and code_toks and " for " in title_l:
                for tok in code_toks:
                    if tok and re.search(rf"\bfor\b.{{0,40}}\b{re.escape(tok)}\b", title_l):
                        score -= 12
                        break
                lead = title_l.split(" for ", 1)[0]
                if any(_model_token_hit(tok, title_l) for tok in code_toks) and not any(
                    _model_token_hit(tok, lead) for tok in code_toks
                ):
                    score -= 6
                if any(tok in title_l for tok in code_toks):
                    # Accessory listings often mention the preferred code after "for".
                    score -= 8
                if recovering:
                    score -= 6
            if self.budget is not None:
                price_f = _as_float(price)
                # Prefer live stock price when known.
                stock = self.stock.get(key) or self.stock.get(pid) or {}
                live_price = _as_float(stock.get("price"))
                if live_price is not None:
                    price_f = live_price
                if price_f is not None:
                    if price_f > self.budget + 1e-9:
                        score -= 8
                    elif self.procedure == PROC_USE:
                        score += 1.5 + min(2.5, 2.5 * (price_f / max(self.budget, 1.0)))
                    elif recovering or self.procedure == PROC_ADAPT:
                        # Stay in the committed price class; cheapest unrelated listings lose.
                        if not self.price_class_ok(price_f, pid, sku):
                            score -= 16
                        else:
                            ref = self.commitment_price() or self.budget
                            closeness = 1.0 - min(1.0, abs(price_f - ref) / max(ref, 1.0))
                            util = price_f / max(self.budget, 1.0)
                            score += 2 + 3.0 * closeness + 2.0 * util
                    else:
                        score += 2 + min(2.0, 2.0 * (price_f / max(self.budget, 1.0)))
            ranked.append((score, key, row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked

    def best_sibling(self) -> dict[str, str] | None:
        """Least-relax alternate: same SKU stem (or prefer-code), live and under budget."""
        need = self.evidence_need() if self.atoms else 0.0
        deferred: list[tuple[float, str, dict[str, Any]]] = []
        # When we already know a failed SKU stem, only accept stem/extra siblings here.
        # Loose prefer-code hits must not preempt a stem rediscovery search.
        stem_only = bool(self.anchor_stems)
        # Harvest variant SKUs from viewed anchors so we do not depend on search luck.
        extra: list[tuple[str, str]] = []
        for pid in list(self.anchor_pids):
            viewed = self.viewed.get(pid) or {}
            for var in viewed.get("variants") or []:
                if not isinstance(var, dict):
                    continue
                vpid = str(var.get("product_id") or pid or "")
                vsku = str(var.get("sku") or "")
                if vpid and vsku:
                    extra.append((vpid, vsku))
                    self._store({**var, "product_id": vpid, "sku": vsku})
        for _score, key, row in self.ranked():
            pid = str(row.get("product_id") or "")
            sku = str(row.get("sku") or "")
            if not pid or not sku or key in self.failed:
                continue
            stem = _sku_stem(sku)
            blob = _product_blob(row, self.viewed.get(pid))
            code_toks = [
                t.lower()
                for t in self.prefer_models
                if any(ch.isdigit() for ch in t)
            ]
            # Least-relax siblings often share the SKU stem across different product_ids.
            stem_ok = bool(stem and stem in self.anchor_stems)
            code_ok = bool(code_toks) and any(_model_token_hit(tok, blob) for tok in code_toks)
            extra_ok = any(pid == ep and sku == es for ep, es in extra)
            same_pid = pid in self.anchor_pids
            if not (stem_ok or extra_ok or same_pid):
                if code_ok and not stem_only:
                    deferred.append((_score, key, row))
                continue
            hit = self._sibling_action(pid, sku, key, row, need)
            if hit:
                return hit
        if not stem_only:
            for _score, key, row in deferred:
                pid = str(row.get("product_id") or "")
                sku = str(row.get("sku") or "")
                hit = self._sibling_action(pid, sku, key, row, need)
                if hit:
                    return hit
        # Last resort: chase harvested variant rows that may not be ranked yet.
        for vpid, vsku in extra:
            key = f"{vpid}::{vsku}"
            if key in self.failed or not vpid or not vsku:
                continue
            row = self.candidates.get(key) or {"product_id": vpid, "sku": vsku}
            hit = self._sibling_action(vpid, vsku, key, row, need)
            if hit:
                return hit
        return None

    def _sibling_action(
        self,
        pid: str,
        sku: str,
        key: str,
        row: dict[str, Any],
        need: float,
    ) -> dict[str, str] | None:
        if pid not in self.viewed:
            return {"action": "view", "product_id": pid, "sku": sku}
        viewed = self.viewed.get(pid) or {}
        if self.atoms and self.evidence_ratio(row, viewed) + 1e-9 < need:
            return None
        if key not in self.stock and pid not in self.stock:
            return {"action": "inspect_stock", "product_id": pid, "sku": sku}
        if self.live_ok(pid, sku):
            p = _as_float((self.stock.get(f"{pid}::{sku}") or self.stock.get(pid) or {}).get("price"))
            if p is None:
                p = _as_float(row.get("price"))
            if not self.price_class_ok(p, pid, sku):
                return None
            return {"action": "add_to_cart", "product_id": pid, "sku": sku}
        return None

    def next_seed(self) -> str:
        if not self.seeds:
            return "in stock"
        idx = min(self.seed_idx, len(self.seeds) - 1)
        candidate = self.seeds[idx]
        # Skip already-tried queries and money-only noise when possible.
        for offset in range(len(self.seeds)):
            s = self.seeds[(idx + offset) % len(self.seeds)]
            if s.lower() in self.tried_queries:
                continue
            parts = s.split()
            if parts and all(_moneyish_token(p) for p in parts):
                continue
            return s
        return candidate

    def recovery_step(self) -> dict[str, str] | None:
        sib = self.best_sibling()
        if sib:
            return sib
        need = self.evidence_need()
        for score, key, row in self.ranked():
            pid = str(row.get("product_id") or "")
            sku = str(row.get("sku") or "")
            if not pid or not sku or key in self.failed:
                continue
            if score < 1 and self.view_n > 0:
                continue
            if pid not in self.viewed:
                return {"action": "view", "product_id": pid, "sku": sku}
            viewed = self.viewed.get(pid) or {}
            if self.atoms and self.evidence_ratio(row, viewed) + 1e-9 < need:
                continue
            if self.procedure == PROC_RECALL and self.phrases and not self._phrase_hit(row, viewed):
                continue
            price_f = _as_float(row.get("price"))
            stock = self.stock.get(key) or self.stock.get(pid) or {}
            live_p = _as_float(stock.get("price"))
            if live_p is not None:
                price_f = live_p
            if not self.price_class_ok(price_f, pid, sku):
                continue
            if key in self.stock or pid in self.stock:
                if self.live_ok(pid, sku):
                    return {"action": "add_to_cart", "product_id": pid, "sku": sku}
                continue
            return {"action": "inspect_stock", "product_id": pid, "sku": sku}
        for _score, key, row in self.ranked():
            pid = str(row.get("product_id") or "")
            sku = str(row.get("sku") or "")
            if pid and sku and key not in self.failed and pid not in self.viewed:
                return {"action": "view", "product_id": pid, "sku": sku}
        return None

    def pick_cart_target(self) -> dict[str, str] | None:
        """Best evidenced, live-valid cart candidate under the active procedure."""
        recovering = bool(self.failed) or self.initial_cart_failed
        if self.procedure != PROC_RANK and (
            self.procedure == PROC_ADAPT or recovering
        ):
            sib = self.best_sibling()
            if sib and sib["action"] in {"inspect_stock", "add_to_cart", "view"}:
                return sib
        need = self.evidence_need()
        soft_ready: dict[str, str] | None = None
        prefer_filter = bool(
            self.procedure == PROC_ADAPT and self.prefer_models and self.failed
        )
        stem_filter = bool(
            self.anchor_stems and recovering and self.procedure == PROC_ADAPT
        )
        code_toks = [
            t.lower() for t in self.prefer_models if any(ch.isdigit() for ch in t)
        ]
        for score, key, row in self.ranked():
            pid = str(row.get("product_id") or "")
            sku = str(row.get("sku") or "")
            if not pid or not sku or key in self.failed:
                continue
            if stem_filter:
                stem = _sku_stem(sku)
                if not (stem and stem in self.anchor_stems) and pid not in self.anchor_pids:
                    continue
            elif prefer_filter:
                stem = _sku_stem(sku)
                blob = _product_blob(row, self.viewed.get(pid))
                stem_ok = bool(stem and stem in self.anchor_stems)
                code_ok = bool(code_toks) and any(_model_token_hit(tok, blob) for tok in code_toks)
                if not (stem_ok or code_ok or pid in self.anchor_pids):
                    continue
            viewed = self.viewed.get(pid) or {}
            if pid not in self.viewed:
                return {"action": "view", "product_id": pid, "sku": sku}
            if self.atoms and self.evidence_ratio(row, viewed) + 1e-9 < need:
                continue
            if self.procedure == PROC_RECALL:
                if self.phrases and not self._phrase_hit(row, viewed):
                    continue
                if self.categories and not self._category_hit(row, viewed):
                    continue
            if self.procedure == PROC_CLAIM and not viewed.get("facts"):
                return {"action": "view", "product_id": pid, "sku": sku}
            price_f = _as_float(row.get("price"))
            live = self.stock.get(key) or self.stock.get(pid) or {}
            live_p = _as_float(live.get("price"))
            if live_p is not None:
                price_f = live_p
            if (
                recovering
                and self.procedure != PROC_RANK
                and not self.price_class_ok(price_f, pid, sku)
            ):
                continue
            # Ranking replacements must stay inside the named variant set.
            if self.procedure == PROC_RANK and self.named:
                named_keys = {f"{p}::{s}" for p, s in self.named}
                if key not in named_keys and pid not in {p for p, _ in self.named}:
                    continue
            if self.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM, PROC_RECALL}:
                if self.categories and not self._category_hit(row, viewed):
                    continue
            if key not in self.stock and pid not in self.stock:
                return {"action": "inspect_stock", "product_id": pid, "sku": sku}
            if not self.live_ok(pid, sku):
                continue
            cand = {"action": "add_to_cart", "product_id": pid, "sku": sku}
            if self.procedure == PROC_SOFT and self.soft_tokens:
                if self._soft_hit(row, viewed):
                    return cand
                if soft_ready is None:
                    soft_ready = cand
                continue
            return cand
        if soft_ready and not recovering:
            return soft_ready
        # Prefer-filter found nothing — fall back once so episodes can still finish.
        if prefer_filter:
            prefer_filter = False
            for score, key, row in self.ranked():
                pid = str(row.get("product_id") or "")
                sku = str(row.get("sku") or "")
                if not pid or not sku or key in self.failed:
                    continue
                if pid not in self.viewed:
                    return {"action": "view", "product_id": pid, "sku": sku}
                if key not in self.stock and pid not in self.stock:
                    return {"action": "inspect_stock", "product_id": pid, "sku": sku}
                if self.live_ok(pid, sku):
                    return {"action": "add_to_cart", "product_id": pid, "sku": sku}
        return None

    def hint(self, turn: int, max_steps: int) -> str:
        left = max_steps - turn + 1
        if self.needs_ask and not self.clarified:
            return "Ask one focused question about the missing firm specification before carting."
        if self.procedure == PROC_USE and not self.priority_asked and not self.cart:
            return "Optionally clarify use priorities with message, then compare facts before first cart."
        if self.pending_q:
            return "Reply to the shopper with message before ordering."
        if self.must_recover and self.cart:
            return f"Live check failed for {self.cart[-1]['key']}: remove, replace, re-inspect, order."
        if self.post_cart_pending and self.cart:
            top = self.cart[-1]
            return (
                f"Cart changed — inspect_stock {top['key']} again before place_test_order "
                "(price/stock may have moved)."
            )
        if self.needs_recheck and self.cart:
            top = self.cart[-1]
            if top["key"] not in self.checked:
                return f"Re-inspect_stock {top['key']} once after the shopper update."
        if self.last_error:
            return f"Recover from error ({self.last_error}) then continue."
        recent = self.actions[-4:]
        if recent and len(set(recent)) == 1 and recent[0] in {"search", "filter", "message", "inspect_stock"}:
            return f"Stop looping on {recent[0]}; advance the purchase pipeline."
        if self.named and self.view_n < min(2, len(self.named)):
            return "View/compare named product::sku keys before ranking or carting."
        if not self.candidates:
            return f"Discover with search/filter (try: {self.next_seed()}), in_stock=true."
        if self.procedure == PROC_RECALL and self.view_n < 2:
            return "Survey multiple phrase-matching listings before carting."
        if self.view_n == 0:
            return "Inspect a shortlist with view/compare; verify exact sku facts."
        if not self.cart:
            if self.procedure == PROC_SOFT and self.soft_tokens:
                return (
                    f"Cart a firm-compliant item honoring soft preference "
                    f"({', '.join(self.soft_tokens)}) when available."
                )
            if left <= 8:
                return "Cart the best fully compliant inspected variant now."
            return "Cart only a fully compliant intended purchase; honor soft prefs when feasible."
        top = self.cart[-1]
        if top["key"] not in self.stock and top["product_id"] not in self.stock:
            return "inspect_stock the carted sku once, then place_test_order."
        if not self.live_ok(top["product_id"], top["sku"]):
            return "Carted sku failed live checks — remove_from_cart and recover."
        if left <= 10:
            return "Verified — place_test_order NOW with grounded claims if required."
        return "Live checks passed — place_test_order for the carted variant."

    def board(self, turn: int, max_steps: int) -> str:
        lines = [
            f"ledger turn={turn}/{max_steps} left={max_steps - turn + 1}",
            f"procedure={self.procedure}",
            (
                f"budget={self.budget if self.budget is not None else 'none'} "
                f"ask={self.needs_ask and not self.clarified} "
                f"recheck={self.needs_recheck} recover={self.must_recover}"
            ),
            (
                f"searches={self.search_n} empty={self.empty_searches} "
                f"views={self.view_n} cands={len(self.candidates)} "
                f"cart={len(self.cart)} adds={self.cart_n}"
            ),
            f"seed={self.next_seed()}",
            f"next={self.hint(turn, max_steps)}",
        ]
        if self.soft_tokens:
            lines.append("soft=" + ", ".join(self.soft_tokens))
        if self.phrases:
            lines.append("phrases=" + " | ".join(self.phrases[:3]))
        if self.categories:
            lines.append("category_mentions=" + " | ".join(self.categories))
        if self.named:
            lines.append("named=" + ", ".join(f"{p}::{s}" for p, s in self.named[:8]))
        if self.rank_axes:
            lines.append(
                "rank_axes="
                + ", ".join(f"{a}:{'up' if d > 0 else 'down'}" for a, d in self.rank_axes)
            )
        if self.mentioned:
            lines.append("claim_fields=" + ", ".join(sorted(self.mentioned)))
        if self.atoms:
            lines.append("constraint_atoms=" + " | ".join(self.atoms[:10]))
        if self.notes:
            lines.append("notes:")
            for note in self.notes[-2:]:
                lines.append(f"  - {note}")
        if self.cart:
            lines.append("cart:")
            for item in self.cart[-3:]:
                stock = self.stock.get(item["key"]) or self.stock.get(item["product_id"]) or {}
                lines.append(
                    f"  - {item['key']} price={stock.get('price', '?')} "
                    f"in_stock={stock.get('in_stock', '?')}"
                )
        for score, key, row in self.ranked()[:_BOARD_SHOW]:
            pid = str(row.get("product_id") or "")
            ev = self.evidence_ratio(row, self.viewed.get(pid))
            lines.append(
                f"cand score={score:.1f} ev={ev:.2f} {key} "
                f"title={str(row.get('title') or '')[:60]} "
                f"price={row.get('price', '?')} stock={row.get('in_stock', '?')}"
            )
        if self.cart:
            top = self.cart[-1]
            viewed = self.viewed.get(top["product_id"]) or {}
            facts = viewed.get("facts") if isinstance(viewed.get("facts"), dict) else {}
            if facts:
                preview = ", ".join(f"{k}={str(v)[:36]}" for k, v in list(facts.items())[:8])
                lines.append(f"final_facts={preview}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claim grounding
# ---------------------------------------------------------------------------


def _ground_order(
    args: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    ledger: Ledger,
) -> dict[str, Any]:
    cleaned = dict(args)
    required = _required_args(tools, "place_test_order")
    needs = "justification" in required
    if not needs:
        cleaned.pop("justification", None)
        return cleaned

    pid = str(cleaned.get("product_id") or "")
    sku = str(cleaned.get("sku") or "")
    if (not pid or not sku) and ledger.cart:
        top = ledger.cart[-1]
        pid = pid or top["product_id"]
        sku = sku or top["sku"]
        cleaned["product_id"] = pid
        cleaned["sku"] = sku
    key = f"{pid}::{sku}"
    stock = dict(ledger.stock.get(key) or ledger.stock.get(pid) or {})
    viewed = dict(ledger.viewed.get(pid) or {})
    candidate = dict(ledger.candidates.get(key) or ledger.candidates.get(pid) or {})
    product = _flatten({**candidate, **viewed, **stock})
    allowed = set(_claim_fields(tools))
    open_spec = _open_spec_claims(tools)
    soft_fields = _soft_schema_fields(tools, allowed)
    goal = _goal_tokens(ledger.query, ledger.notes)
    goal_text = f"{_preamble(ledger.query)} {' '.join(ledger.notes[-3:])}".lower()
    prefer = set(ledger.mentioned)

    raw_claims: list[dict[str, Any]] = []
    just = cleaned.get("justification")
    if isinstance(just, dict) and isinstance(just.get("claims"), list):
        raw_claims = [c for c in just["claims"] if isinstance(c, dict)]

    if open_spec:
        allowed.update({"price", "in_stock"})
        allowed.update(soft_fields)
        allowed.update(prefer)
        for claim in raw_claims:
            field = str(claim.get("field") or "").strip()
            if field and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,40}", field):
                allowed.add(field)
                prefer.add(field)
    if not allowed:
        allowed = {"price", "in_stock"}

    grounded: list[dict[str, str]] = []
    seen: set[str] = set()
    attest_blob = _product_blob(product, viewed)

    def add(field: str, value: Any) -> None:
        if field not in allowed or field in seen:
            return
        if value is None or value == "":
            return
        if isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value).strip()
        if not text or text.lower() in {"unknown", "none", "n/a", "null"}:
            return
        if field != "in_stock" and text.lower() == "false":
            return
        # Non-core claims must be attested by viewed/inspected product text.
        if field not in {"price", "in_stock"} and not _value_attested(value, attest_blob):
            return
        grounded.append({"field": field, "value": text[:80]})
        seen.add(field)

    def best_value(field: str) -> Any:
        direct = _lookup([product, stock, viewed, candidate], field)
        if direct not in (None, ""):
            # Still require goal corroboration for non-core fields when possible.
            if field in {"price", "in_stock"}:
                return direct
            if _overlap(direct, goal) > 0 or _value_attested(direct, goal_text):
                return direct
            # Key match alone is enough when the field surface is in the goal.
            surface = field.replace("_", " ").lower()
            if surface in goal_text or field.lower() in goal_text:
                return direct
            text = str(direct).lower().strip()
            if any(tok == text or tok in text.split() for tok in goal if tok.isalpha() and len(tok) >= 3):
                return direct
            if any(atom in text for atom in ledger.atoms):
                return direct
            # Fall through to scored search for a better corroborating value.
        want_nums = _goal_nums_near(ledger.query, field, ledger.notes)
        if not want_nums:
            want_nums = {t for t in goal if any(ch.isdigit() for ch in t)}
        best_val: Any = None
        best_score = 0
        for key_name, value in product.items():
            if value in (None, "", True, False) or isinstance(value, (dict, list)):
                continue
            if not isinstance(key_name, str):
                continue
            if key_name in {
                "product_id",
                "sku",
                "title",
                "price",
                "in_stock",
                "currency",
                "listing_id",
            } and field not in {"price", "in_stock"}:
                continue
            text = str(value).lower().strip()
            nums = set(_NUM_RE.findall(text))
            shared = nums & want_nums if want_nums else set()
            overlap = 0
            if _keys_match(_norm_key(key_name), _norm_key(field)):
                overlap += 8
            if shared:
                overlap += 3 * len(shared)
            if len(text) >= 2 and (
                text in goal_text or any(tok == text for tok in goal if len(tok) >= 2)
            ):
                overlap += 6
            # Compact coded values (e.g. letter quality codes) that appear in the goal.
            if 1 < len(text) <= 12 and text.replace(" ", "").isalnum():
                compact = text.replace(" ", "")
                if compact in goal_text.replace(" ", "") or any(
                    compact in tok or tok in compact for tok in goal if len(tok) >= 2
                ):
                    overlap += 5
            for tok in goal:
                if tok.isalpha() and len(tok) >= 2 and (tok in text or text in tok):
                    overlap += 2
            for atom in ledger.atoms:
                if atom in text:
                    overlap += 3
            overlap += min(2, _overlap(value, goal))
            if overlap > best_score:
                best_score = overlap
                best_val = value
        return best_val if best_score >= 4 else None

    add("price", stock.get("price", product.get("price")))
    stock_flag = stock.get("in_stock", product.get("in_stock"))
    flagged = _as_bool(stock_flag)
    if flagged is not None:
        add("in_stock", flagged)
    else:
        add("in_stock", stock_flag)

    for claim in raw_claims:
        field = str(claim.get("field") or "").strip()
        if not field or field in {"price", "in_stock"}:
            continue
        if field not in allowed and not open_spec:
            continue
        if open_spec:
            allowed.add(field)
        observed = best_value(field)
        if observed is not None:
            add(field, observed)
            continue
        raw_val = claim.get("value")
        if raw_val in (None, ""):
            continue
        if _value_attested(raw_val, attest_blob):
            add(field, raw_val)

    for field in list(prefer):
        if field in seen or field in {"price", "in_stock"}:
            continue
        observed = best_value(field)
        if observed is not None:
            add(field, observed)

    # One schema-allowed field with strong goal corroboration (avoid claim spam).
    if not any(f not in {"price", "in_stock"} | soft_fields for f in seen):
        ranked_fields: list[tuple[int, str, Any]] = []
        for field in sorted(allowed):
            if field in {"price", "in_stock"} | soft_fields:
                continue
            surface = field.replace("_", " ").lower()
            parts = [p for p in field.lower().split("_") if len(p) >= 3]
            in_goal = (
                surface in goal_text
                or field.lower() in goal_text
                or any(p in goal_text for p in parts)
                or bool(_goal_nums_near(ledger.query, field, ledger.notes))
            )
            observed = best_value(field)
            if observed is None or not _value_attested(observed, attest_blob):
                continue
            score = 0
            if in_goal:
                score += 4
            score += min(3, _overlap(observed, goal))
            if score > 0:
                ranked_fields.append((score, field, observed))
        ranked_fields.sort(key=lambda row: (-row[0], row[1]))
        if ranked_fields:
            _s, field, value = ranked_fields[0]
            add(field, value)

    # At most one extra observation-backed claim when none yet — avoid unfaithful extras.
    if not any(f not in {"price", "in_stock"} | soft_fields for f in seen):
        scored: list[tuple[int, str, Any]] = []
        hints = [
            str(c).strip()
            for c in list(prefer) + [str(c.get("field") or "") for c in raw_claims]
            if str(c).strip() and str(c).strip() not in {"price", "in_stock"}
        ]
        hints.extend(sorted(f for f in allowed if f not in {"price", "in_stock"}))
        goal_nums = {t for t in goal if any(ch.isdigit() for ch in t)}
        for key_name, value in product.items():
            if not isinstance(key_name, str) or value in (None, ""):
                continue
            if key_name in {
                "product_id",
                "sku",
                "title",
                "price",
                "in_stock",
                "currency",
                "listing_id",
                "category_path",
                "variants",
                "options",
                "condition",
            }:
                continue
            if isinstance(value, (dict, list, bool)):
                continue
            text = str(value).lower().strip()
            overlap = 0
            if len(text) >= 2 and (
                text in goal_text or any(tok == text for tok in goal if len(tok) >= 2)
            ):
                overlap += 6
            overlap += min(3, _overlap(value, goal))
            nums = set(_NUM_RE.findall(text))
            overlap += 2 * len(nums & goal_nums)
            for atom in ledger.atoms:
                if atom in text or re.sub(r"\s+", "", atom) in re.sub(r"\s+", "", text):
                    overlap += 4
            if overlap <= 0:
                continue
            if key_name in {"model", "condition", "listing_id"} and not (nums & goal_nums):
                continue
            claim_name = None
            for cand in hints:
                if _keys_match(_norm_key(cand), _norm_key(key_name)):
                    claim_name = cand
                    overlap += 3
                    break
            if claim_name is None and open_spec:
                if re.fullmatch(r"[a-z][a-z0-9_]{1,40}", key_name):
                    claim_name = key_name
            if claim_name is None:
                continue
            if not _value_attested(value, attest_blob):
                continue
            if open_spec or claim_name in allowed:
                scored.append((overlap, claim_name, value))
        scored.sort(key=lambda row: (-row[0], row[1]))
        if scored:
            _s, field, value = scored[0]
            if open_spec:
                allowed.add(field)
            add(field, value)

    # Schema-supported reason field: prefer an attested brand when the request names one
    # or when open-spec justification asks for a positive reason.
    if open_spec and "brand" not in seen:
        brand_val = product.get("brand")
        if brand_val not in (None, "") and _value_attested(brand_val, attest_blob):
            brand_l = str(brand_val).lower()
            if any(tok.lower() == brand_l for tok in ledger.soft_tokens) or brand_l in goal_text:
                allowed.add("brand")
                add("brand", brand_val)
            elif not any(f not in {"price", "in_stock"} for f in seen):
                allowed.add("brand")
                add("brand", brand_val)
            elif ledger.procedure == PROC_CLAIM:
                allowed.add("brand")
                add("brand", brand_val)

    if soft_fields and not (seen & soft_fields):
        for field in sorted(soft_fields):
            observed = _lookup([product], field)
            if observed is True or str(observed).strip().lower() in {"true", "yes", "1"}:
                add(field, "true")
                break
            if observed not in (None, "") and _value_attested(observed, attest_blob):
                add(field, observed)
                break

    # Guarantee at least one non-price specification claim for open-spec checkouts.
    if open_spec and not any(f not in {"price", "in_stock"} for f in seen):
        for key_name, value in product.items():
            if not isinstance(key_name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", key_name):
                continue
            if key_name in {"product_id", "sku", "title", "price", "in_stock", "currency", "brand"}:
                continue
            if value in (None, "") or isinstance(value, (dict, list, bool)):
                continue
            if _value_attested(value, attest_blob) and _overlap(value, goal) > 0:
                allowed.add(key_name)
                add(key_name, value)
                break

    if needs and not grounded:
        add("price", stock.get("price", product.get("price")))
        stock_flag = stock.get("in_stock", product.get("in_stock", True))
        flagged = _as_bool(stock_flag)
        add("in_stock", flagged if flagged is not None else stock_flag)

    cleaned["justification"] = {"claims": grounded}
    return cleaned


# ---------------------------------------------------------------------------
# Normalize + controller
# ---------------------------------------------------------------------------


def _normalize(
    name: str,
    args: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    ledger: Ledger,
) -> dict[str, Any]:
    cleaned = dict(args)
    if name == "search":
        query = str(cleaned.get("query") or "").strip()
        # Digit-only SKU stems / model codes are valid rediscovery queries after events.
        stem_or_code = bool(
            query
            and (
                query.lower() in {s.lower() for s in ledger.anchor_stems}
                or any(query.lower() == t.lower() for t in ledger.prefer_models)
                or (query.isdigit() and len(query) >= 4)
            )
        )
        if not query or (
            not stem_or_code
            and (query.isdigit() or len(_WORD_RE.findall(query)) < 2)
        ):
            query = ledger.next_seed()
            if ledger.anchor_stems and (ledger.failed or ledger.initial_cart_failed):
                query = sorted(ledger.anchor_stems)[0]
        # Never resubmit an identical discovery query — climb the seed ladder.
        if query.lower() in ledger.tried_queries and ledger.same_search_streak >= 1:
            ledger.seed_idx = min(ledger.seed_idx + 1, max(0, len(ledger.seeds) - 1))
            alt = ledger.next_seed()
            if alt.lower() != query.lower():
                query = alt
            else:
                tokens = [t for t in _WORD_RE.findall(query) if t.lower() not in _STOP]
                query = " ".join(tokens[: max(2, len(tokens) - 1)]) or query
        # Drop accidental duplicated whitespace-separated tokens from seed stitching.
        # Do NOT tokenize with _WORD_RE here — it splits decimals (7.6 → 7 6).
        parts = [p for p in query.split() if p]
        dedup: list[str] = []
        for tok in parts:
            if not dedup or dedup[-1].lower() != tok.lower():
                dedup.append(tok)
        if len(dedup) > 10 and '"' not in query:
            cleaned["query"] = " ".join(dedup[:8])
        else:
            cleaned["query"] = " ".join(dedup) if dedup else query
        cleaned.setdefault("in_stock", True)
        cleaned.setdefault("k", 10)
        props = _tool_props(tools, "search")
        if ledger.budget is not None and "max_price" in props and "max_price" not in cleaned:
            cleaned["max_price"] = ledger.budget
    if name == "filter":
        cleaned.setdefault("in_stock", True)
        cleaned.setdefault("k", 10)
        if ledger.budget is not None and ledger.budget >= 5 and "max_price" not in cleaned:
            cleaned["max_price"] = ledger.budget
        props = _tool_props(tools, "filter")
        if ledger.categories and "category" in props and "category" not in cleaned:
            cleaned["category"] = ledger.categories[0]
        q = str(cleaned.get("query") or "").strip()
        if not q and ledger.seeds:
            cleaned["query"] = ledger.next_seed()
        elif q.lower() in ledger.tried_queries and ledger.same_search_streak >= 1:
            cleaned["query"] = ledger.next_seed()
    if name == "compare":
        ids = cleaned.get("product_ids")
        if isinstance(ids, list):
            cleaned["product_ids"] = [str(x) for x in ids[:5] if str(x).strip()]
        elif isinstance(ids, str) and ids.strip():
            cleaned["product_ids"] = [p.strip() for p in ids.split(",") if p.strip()][:5]
        if not cleaned.get("product_ids") and ledger.named:
            cleaned["product_ids"] = [pid for pid, _sku in ledger.named[:5]]
    if name == "message":
        cleaned["content"] = str(cleaned.get("content") or "").strip()[:400]
    if name == "place_test_order":
        if (not cleaned.get("product_id") or not cleaned.get("sku")) and ledger.cart:
            top = ledger.cart[-1]
            cleaned.setdefault("product_id", top["product_id"])
            cleaned.setdefault("sku", top["sku"])
        cleaned = _ground_order(cleaned, tools=tools, ledger=ledger)
    return cleaned


def _shopper_needs_verbal(ledger: "Ledger") -> bool:
    """True when notes ask for a spoken selection / firm-spec confirmation."""
    blob = " ".join(ledger.notes[-3:])
    if not blob:
        return False
    if re.search(
        r"\bwhich one\b|\bare you going\b|\bdoes it actually\b|"
        r"\bnon-negotiable\b|\bbefore you buy\b|\bconfirm\b|"
        r"\bgoing with\b|\btell me (?:which|what)\b",
        blob,
        re.I,
    ):
        return True
    # Any unanswered shopper question except pure availability pings.
    if ledger.pending_q and "?" in blob:
        if re.search(
            r"\bstill available\b|\bis that (?:still )?(?:available|in stock)\b",
            blob,
            re.I,
        ) and not re.search(r"\bwhich one\b|\bbefore you buy\b", blob, re.I):
            return False
        return True
    return False


def _forced_plan(
    *,
    ledger: Ledger,
    tools: list[dict[str, Any]],
    max_calls: int,
    remaining: int,
) -> list[tuple[str, dict[str, Any]]] | None:
    """Return a deterministic action plan when the next step is unambiguous."""
    names = _tool_names(tools)

    if ledger.needs_ask and not ledger.clarified and "message" in names and not ledger.asked:
        return [
            (
                "message",
                {
                    "content": (
                        "Before I pick an item, which firm product specification must it "
                        "satisfy beyond budget and in-stock availability?"
                    )
                },
            )
        ]

    # Answer selection / firm-spec interventions early (family intervention gate).
    if (
        ledger.pending_q
        and "message" in names
        and _shopper_needs_verbal(ledger)
        and "message" not in ledger.actions[-8:]
    ):
        pick = ledger.cart[-1] if ledger.cart else None
        if pick is None:
            for _s, _k, row in ledger.ranked()[:1]:
                pid = str(row.get("product_id") or "")
                sku = str(row.get("sku") or "")
                if pid and sku:
                    pick = {"product_id": pid, "sku": sku}
                    break
        atom = next((a for a in ledger.atoms if any(ch.isdigit() for ch in a)), None)
        if pick:
            content = (
                f"I am going with {pick['product_id']}::{pick['sku']}"
                + (f", confirming the firm {atom} requirement." if atom else ".")
            )
        else:
            content = (
                "I will verify listings against your stated constraints and confirm "
                "the chosen product_id::sku before checkout."
            )
        return [("message", {"content": content[:400]})]

    # After clarification, always rediscover with the new atom (even if junk
    # candidates were collected before the shopper answered).
    if (
        ledger.post_clarify_search
        and ledger.clarified
        and not ledger.cart
        and "search" in names
    ):
        atoms = [a for a in ledger.atoms if any(ch.isdigit() for ch in a)]
        soft = ledger.soft_tokens[:1]
        if soft and atoms:
            seed = f"{soft[0]} {atoms[0]}"
        elif atoms:
            seed = atoms[0]
        else:
            seed = ledger.next_seed()
        args = {"query": seed, "in_stock": True, "k": 12}
        if ledger.budget is not None and "max_price" in _tool_props(tools, "search"):
            args["max_price"] = ledger.budget
        return [("search", args)]

    # Over-budget / OOS cart must be cleared before any further checkout attempts.
    # Use must_recover OR last-known over-budget even if stock cache was invalidated.
    if ledger.cart and "remove_from_cart" in names:
        top = ledger.cart[-1]
        stock = ledger.stock.get(top["key"]) or ledger.stock.get(top["product_id"])
        priced_over = False
        if isinstance(stock, dict) and ledger.budget is not None:
            price = _as_float(stock.get("price"))
            priced_over = price is not None and price > ledger.budget + 1e-9
        if ledger.must_recover or priced_over or ledger.over_budget(top["product_id"], top["sku"]):
            plan: list[tuple[str, dict[str, Any]]] = [
                ("remove_from_cart", {"product_id": top["product_id"], "sku": top["sku"]})
            ]
            ledger.must_recover = True
            ledger.failed.add(top["key"])
            stem = _sku_stem(top.get("sku") or "")
            if not stem and "::" in top.get("key", ""):
                stem = _sku_stem(top["key"].split("::", 1)[-1])
            if stem:
                ledger.anchor_stems.add(stem)
            ledger.anchor_pids.add(top["product_id"])
            ledger.event_target_key = top["key"]
            # Least-relax: only chase an in-hand same-stem sibling. Otherwise SEARCH the
            # stem immediately — never fall through to prefer-code / ranked noise views.
            nxt = None
            if stem:
                for _score, key, row in ledger.ranked():
                    pid = str(row.get("product_id") or "")
                    sku = str(row.get("sku") or "")
                    if not pid or not sku or key in ledger.failed:
                        continue
                    if _sku_stem(sku) != stem:
                        continue
                    nxt = ledger._sibling_action(
                        pid,
                        sku,
                        key,
                        row,
                        ledger.evidence_need() if ledger.atoms else 0.0,
                    )
                    if nxt:
                        break
            if nxt and nxt["action"] in names and len(plan) < max_calls:
                args = {"product_id": nxt["product_id"]}
                if nxt["action"] != "view":
                    args["sku"] = nxt["sku"]
                plan.append((nxt["action"], args))
            elif "search" in names and len(plan) < max_calls and (
                stem or ledger.prefer_models or ledger.anchor_stems or ledger.anchor_pids
            ):
                # Title-derived rediscovery first (observation text), then prefer-code.
                prefs = [
                    t
                    for t in ledger.prefer_models
                    if any(ch.isdigit() for ch in t) and not _moneyish_token(t)
                ]
                colors = [
                    t
                    for t in ledger.prefer_models
                    if t.isalpha() and not _moneyish_token(t)
                ]
                title_q = _anchor_title_query(ledger)
                if title_q:
                    q = title_q
                elif prefs and colors:
                    q = f"{prefs[0]} {colors[0]}"
                elif prefs:
                    q = prefs[0]
                elif stem:
                    q = stem
                else:
                    q = sorted(ledger.anchor_stems)[0] if ledger.anchor_stems else ledger.next_seed()
                sargs: dict[str, Any] = {"query": q, "in_stock": True, "k": 12}
                if ledger.budget is not None and "max_price" in _tool_props(tools, "search"):
                    sargs["max_price"] = ledger.budget
                plan.append(("search", sargs))
            elif ledger.procedure == PROC_RANK and ledger.named and len(plan) < max_calls:
                # One post-event re-compare, then cart the new winner — never loop.
                keep = [
                    pid
                    for pid, sku in ledger.named
                    if f"{pid}::{sku}" not in ledger.failed
                ][:5]
                if (
                    not ledger.post_event_rerank_done
                    and "compare" in names
                    and len(keep) >= 2
                ):
                    ledger.post_event_rerank_done = True
                    plan.append(("compare", {"product_ids": keep}))
                else:
                    ledger.post_event_rerank_done = True
                    # Refresh live prices on remaining named variants, then cart.
                    if "inspect_stock" in names:
                        for pid, sku in ledger.named:
                            key = f"{pid}::{sku}"
                            if key in ledger.failed:
                                continue
                            if key not in ledger.checked:
                                plan.append(("inspect_stock", {"product_id": pid, "sku": sku}))
                                break
                    if len(plan) < 2:
                        target = ledger.pick_cart_target()
                        if target and target["action"] in names:
                            args = {"product_id": target["product_id"]}
                            if target["action"] != "view":
                                args["sku"] = target["sku"]
                            plan.append((target["action"], args))
            elif "search" in names and len(plan) < max_calls:
                # Stem / prefer-code rediscovery BEFORE unrelated recovery views.
                # Never seed searches with money-like atoms from price events.
                prefs = [
                    t
                    for t in ledger.prefer_models
                    if any(ch.isdigit() for ch in t) and not _moneyish_token(t)
                ]
                colors = [
                    t
                    for t in ledger.prefer_models
                    if t.isalpha() and not _moneyish_token(t)
                ]
                atoms = [
                    a
                    for a in ledger.atoms
                    if any(ch.isdigit() for ch in a) and not _moneyish_token(a)
                ][:2]
                soft = [t for t in ledger.soft_tokens[:1] if not _moneyish_token(t)]
                if stem:
                    q = stem
                elif prefs and colors:
                    q = f"{prefs[0]} {colors[0]}"
                elif prefs:
                    q = prefs[0]
                elif soft and atoms:
                    q = f"{soft[0]} {atoms[0]}"
                elif atoms:
                    q = " ".join(atoms)
                else:
                    q = ledger.next_seed()
                if not _moneyish_token(q) and not all(_moneyish_token(p) for p in q.split()):
                    sargs: dict[str, Any] = {"query": q, "in_stock": True, "k": 12}
                    if ledger.budget is not None and "max_price" in _tool_props(tools, "search"):
                        sargs["max_price"] = ledger.budget
                    plan.append(("search", sargs))
                else:
                    nxt = ledger.recovery_step()
                    if nxt and nxt["action"] in names and len(plan) < max_calls:
                        args = {"product_id": nxt["product_id"]}
                        if nxt["action"] != "view":
                            args["sku"] = nxt["sku"]
                        plan.append((nxt["action"], args))
            else:
                nxt = ledger.recovery_step()
                if nxt and nxt["action"] in names and len(plan) < max_calls:
                    args = {"product_id": nxt["product_id"]}
                    if nxt["action"] != "view":
                        args["sku"] = nxt["sku"]
                    plan.append((nxt["action"], args))
            return plan[:max_calls]

    # Recovering with an empty cart: chase sibling stems before broad rediscovery.
    # Ranking named-variant tasks must re-compare axes, not divert into prefer search.
    if not ledger.cart and (ledger.failed or ledger.initial_cart_failed):
        if ledger.procedure == PROC_RANK and ledger.named:
            keep = [
                pid
                for pid, sku in ledger.named
                if f"{pid}::{sku}" not in ledger.failed
            ][:5]
            if (
                not ledger.post_event_rerank_done
                and "compare" in names
                and len(keep) >= 2
            ):
                ledger.post_event_rerank_done = True
                return [("compare", {"product_ids": keep})]
            ledger.post_event_rerank_done = True
            # Re-inspect one remaining named variant for live post-event price.
            if "inspect_stock" in names:
                for pid, sku in ledger.named:
                    key = f"{pid}::{sku}"
                    if key in ledger.failed:
                        continue
                    if key not in ledger.checked:
                        return [("inspect_stock", {"product_id": pid, "sku": sku})]
            target = ledger.pick_cart_target()
            if target and target["action"] in names:
                args = {"product_id": target["product_id"]}
                if target["action"] != "view":
                    args["sku"] = target["sku"]
                return [(target["action"], args)]
            if keep and "view" in names:
                unseen = [pid for pid in keep if pid not in ledger.viewed]
                if unseen:
                    return [("view", {"product_id": unseen[0]})]
                return [("view", {"product_id": keep[0]})]
        sib = ledger.best_sibling()
        if sib and sib["action"] in names:
            args = {"product_id": sib["product_id"]}
            if sib["action"] != "view":
                args["sku"] = sib["sku"]
            return [(sib["action"], args)]
        if "search" in names:
            stems = sorted(ledger.anchor_stems)
            prefs = [
                t
                for t in ledger.prefer_models
                if any(ch.isdigit() for ch in t) and not _moneyish_token(t)
            ]
            colors = [
                t for t in ledger.prefer_models if t.isalpha() and not _moneyish_token(t)
            ]
            q = None
            title_q = _anchor_title_query(ledger)
            if title_q:
                q = title_q
            elif prefs:
                cand = f"{prefs[0]} {colors[0]}" if colors else prefs[0]
                if cand.lower() not in ledger.tried_queries:
                    q = cand
            if q is None:
                for stem in stems:
                    if stem.lower() not in ledger.tried_queries:
                        q = stem
                        break
            if q is None:
                atoms = [
                    a
                    for a in ledger.atoms
                    if any(ch.isdigit() for ch in a) and not _moneyish_token(a)
                ][:2]
                soft = [t for t in ledger.soft_tokens[:1] if not _moneyish_token(t)]
                cand = " ".join(soft + atoms) if (atoms or soft) else ledger.next_seed()
                if cand.lower() not in ledger.tried_queries:
                    q = cand
            if q and not all(_moneyish_token(p) for p in q.split()):
                sargs = {"query": q, "in_stock": True, "k": 12}
                if ledger.budget is not None and "max_price" in _tool_props(tools, "search"):
                    sargs["max_price"] = ledger.budget
                return [("search", sargs)]
        # Stem-filter recovery (adapt only): never cart non-stem noise while anchors exist.
        if ledger.anchor_stems and ledger.procedure == PROC_ADAPT:
            sib = ledger.best_sibling()
            if sib and sib["action"] in names:
                args = {"product_id": sib["product_id"]}
                if sib["action"] != "view":
                    args["sku"] = sib["sku"]
                return [(sib["action"], args)]
            # After several rediscovery attempts, allow code-matched non-stem only if
            # we still have no stem sibling — otherwise keep searching title seeds.
            if ledger.search_n >= 4 and "search" not in ledger.actions[-1:]:
                pass
            elif ledger.search_n < 6 and "search" in names:
                title_q = _anchor_title_query(ledger)
                prefs = [
                    t
                    for t in ledger.prefer_models
                    if any(ch.isdigit() for ch in t) and not _moneyish_token(t)
                ]
                q = title_q or (prefs[0] if prefs else None)
                if q and q.lower() not in ledger.tried_queries:
                    sargs = {"query": q, "in_stock": True, "k": 12}
                    if ledger.budget is not None and "max_price" in _tool_props(tools, "search"):
                        sargs["max_price"] = ledger.budget
                    return [("search", sargs)]
        nxt = ledger.recovery_step()
        if nxt and nxt["action"] in names:
            # Adapt+stem: refuse non-stem recovery_step carts/views until searches exhausted.
            if (
                ledger.procedure == PROC_ADAPT
                and ledger.anchor_stems
                and ledger.search_n < 5
            ):
                nsku = str(nxt.get("sku") or "")
                if _sku_stem(nsku) not in ledger.anchor_stems and nxt.get("product_id") not in ledger.anchor_pids:
                    title_q = _anchor_title_query(ledger)
                    if title_q and "search" in names:
                        return [("search", {"query": title_q, "in_stock": True, "k": 12})]
                    if "search" in names and ledger.prefer_models:
                        prefs = [
                            t
                            for t in ledger.prefer_models
                            if any(ch.isdigit() for ch in t) and not _moneyish_token(t)
                        ]
                        if prefs:
                            return [("search", {"query": prefs[0], "in_stock": True, "k": 12})]
            args = {"product_id": nxt["product_id"]}
            if nxt["action"] != "view":
                args["sku"] = nxt["sku"]
            return [(nxt["action"], args)]

    # Always re-inspect after cart adds — market events commonly apply then.
    if ledger.post_cart_pending and ledger.cart and "inspect_stock" in names:
        top = ledger.cart[-1]
        if top["key"] in ledger.post_cart_pending or top["key"] not in ledger.stock:
            return [("inspect_stock", {"product_id": top["product_id"], "sku": top["sku"]})]

    if ledger.needs_recheck and ledger.cart and "inspect_stock" in names:
        top = ledger.cart[-1]
        if top["key"] not in ledger.checked:
            return [("inspect_stock", {"product_id": top["product_id"], "sku": top["sku"]})]

    # Answer selection / firm-spec questions before checkout (intervention gate).
    if (
        ledger.pending_q
        and "message" in names
        and ledger.identical_action_streak < 2
        and _shopper_needs_verbal(ledger)
    ):
        pick = ledger.cart[-1] if ledger.cart else None
        if pick is None:
            for _s, _k, row in ledger.ranked()[:1]:
                pid = str(row.get("product_id") or "")
                sku = str(row.get("sku") or "")
                if pid and sku:
                    pick = {"product_id": pid, "sku": sku}
                    break
        atom = next((a for a in ledger.atoms if any(ch.isdigit() for ch in a)), None)
        if pick:
            content = (
                f"I am going with {pick['product_id']}::{pick['sku']}"
                + (f", confirming the firm {atom} requirement." if atom else ".")
            )
        else:
            content = (
                "I will verify listings against your stated constraints, budget, "
                "and stock, then confirm the chosen product_id::sku before checkout."
            )
        return [("message", {"content": content[:400]})]

    ready = ledger.ready_order()
    if ready and "place_test_order" in names:
        return [("place_test_order", _ground_order(dict(ready), tools=tools, ledger=ledger))]

    # Break add↔inspect thrashing once post-cart inspect passed.
    if ledger.cart and "place_test_order" in names:
        top = ledger.cart[-1]
        if (
            top["key"] not in ledger.post_cart_pending
            and ledger.live_ok(top["product_id"], top["sku"])
            and not ledger.must_recover
            and not (ledger.pending_q and _shopper_needs_verbal(ledger))
        ):
            return [
                (
                    "place_test_order",
                    _ground_order(
                        {"product_id": top["product_id"], "sku": top["sku"]},
                        tools=tools,
                        ledger=ledger,
                    ),
                )
            ]

    if ledger.pending_q and "message" in names and ledger.identical_action_streak < 2:
        # Remaining pending questions (non-checkout path).
        if ledger.clarified and ledger.candidates and not ledger.cart:
            pass
        else:
            pick = ledger.cart[-1] if ledger.cart else None
            if pick is None:
                for _s, _k, row in ledger.ranked()[:1]:
                    pid = str(row.get("product_id") or "")
                    sku = str(row.get("sku") or "")
                    if pid and sku:
                        pick = {"product_id": pid, "sku": sku}
                        break
            if pick:
                atom = next((a for a in ledger.atoms if any(ch.isdigit() for ch in a)), None)
                content = (
                    f"Planning {pick['product_id']}::{pick['sku']}"
                    + (f" for the firm {atom} requirement" if atom else "")
                    + "; I will re-check live price/stock before ordering."
                )
            else:
                content = (
                    "I will verify listings against your stated constraints, budget, "
                    "and stock, then confirm the chosen product_id::sku before checkout."
                )
            return [("message", {"content": content[:400]})]

    if not ledger.cart and ledger.failed:
        nxt = ledger.recovery_step()
        if nxt and nxt["action"] in names:
            args = {"product_id": nxt["product_id"]}
            if nxt["action"] != "view":
                args["sku"] = nxt["sku"]
            return [(nxt["action"], args)]

    if ledger.named and ledger.view_n == 0:
        if "compare" in names and len(ledger.named) >= 2:
            return [("compare", {"product_ids": [pid for pid, _ in ledger.named[:5]]})]
        if "view" in names:
            pid, _sku = ledger.named[0]
            return [("view", {"product_id": pid})]

    if ledger.procedure == PROC_RANK and ledger.named and "view" in names:
        named_pids = [pid for pid, _ in ledger.named]
        unseen = [pid for pid in named_pids if pid not in ledger.viewed]
        if unseen:
            return [("view", {"product_id": unseen[0]})]

    # Use-fit: the goal already states intended use — never spend a turn asking.
    if ledger.procedure == PROC_USE:
        ledger.priority_asked = True

    if ledger.same_search_streak >= 2 and ledger.candidates and "view" in names:
        for _score, key, row in ledger.ranked():
            pid = str(row.get("product_id") or "")
            if pid and pid not in ledger.viewed and key not in ledger.failed:
                return [("view", {"product_id": pid})]
        nxt = ledger.recovery_step()
        if nxt and nxt["action"] in names:
            args = {"product_id": nxt["product_id"]}
            if nxt["action"] != "view":
                args["sku"] = nxt["sku"]
            return [(nxt["action"], args)]

    if ledger.same_search_streak >= 2 and not ledger.candidates and "search" in names:
        ledger.seed_idx = min(ledger.seed_idx + 1, max(0, len(ledger.seeds) - 1))
        seed = ledger.next_seed()
        args: dict[str, Any] = {"query": seed, "in_stock": True, "k": 10}
        if ledger.procedure == PROC_RECALL:
            args["k"] = 12
        if ledger.budget is not None and "max_price" in _tool_props(tools, "search"):
            args["max_price"] = ledger.budget
        return [("search", args)]

    # Inspect discoveries, but do not auto-cart from ranking alone — let the model choose.
    if ledger.candidates and "view" in names:
        recent = ledger.actions[-3:]
        stalled_search = bool(recent) and all(a in {"search", "filter"} for a in recent)
        min_views = 2 if ledger.procedure in {PROC_RECALL, PROC_USE, PROC_CLAIM} else 1
        if ledger.view_n < min_views or stalled_search or ledger.same_search_streak >= 2:
            for _score, key, row in ledger.ranked():
                pid = str(row.get("product_id") or "")
                if pid and pid not in ledger.viewed and key not in ledger.failed:
                    return [("view", {"product_id": pid})]

    # Procedure-aware cart nudge: cart once a shortlist is evidenced.
    cart_horizon = {
        PROC_RECALL: 14,
        PROC_USE: 14,
        PROC_CLAIM: 16,
        PROC_SOFT: 18,
        PROC_ASK: 16,
        PROC_STRICT: 16,
        PROC_ADAPT: 18,
        PROC_RANK: 22,
    }.get(ledger.procedure, 16)
    if ledger.failed or ledger.initial_cart_failed:
        # After a cart event, stop surveying and replace while steps remain.
        cart_horizon = max(cart_horizon, 26)
    elif ledger.procedure == PROC_USE and ledger.view_n >= 2:
        # After a short fact survey, cart the best use-fit before events force a scramble.
        cart_horizon = max(cart_horizon, 24)
    elif ledger.procedure in {PROC_SOFT, PROC_ASK, PROC_STRICT} and ledger.view_n >= 2:
        cart_horizon = max(cart_horizon, 22)
    min_views_before_cart = {
        PROC_RECALL: 2,
        PROC_USE: 2,
        PROC_CLAIM: 1,
        PROC_SOFT: 1,
        PROC_ASK: 1,
        PROC_STRICT: 1,
        PROC_ADAPT: 1,
        PROC_RANK: 1,
    }.get(ledger.procedure, 1)
    if (
        remaining <= cart_horizon
        and ledger.candidates
        and not ledger.cart
        and ledger.view_n >= min_views_before_cart
    ):
        if ledger.needs_ask and not ledger.clarified:
            pass
        else:
            target = ledger.pick_cart_target()
            if target and target["action"] in names:
                args = {"product_id": target["product_id"]}
                if target["action"] != "view":
                    args["sku"] = target["sku"]
                plan = [(target["action"], args)]
                # Leave room to order: batch post-cart inspect on the cart turn.
                if (
                    target["action"] == "add_to_cart"
                    and remaining <= 4
                    and max_calls >= 2
                    and "inspect_stock" in names
                ):
                    plan.append(
                        (
                            "inspect_stock",
                            {
                                "product_id": target["product_id"],
                                "sku": target["sku"],
                            },
                        )
                    )
                return plan[:max_calls]

    # If still uncarted very late, take the best live inspected option.
    if remaining <= 10 and ledger.candidates and not ledger.cart and ledger.view_n > 0:
        if not (ledger.needs_ask and not ledger.clarified):
            for _score, key, row in ledger.ranked():
                pid = str(row.get("product_id") or "")
                sku = str(row.get("sku") or "")
                if not pid or not sku or key in ledger.failed:
                    continue
                if ledger.atoms and pid in ledger.viewed:
                    if ledger.evidence_ratio(row, ledger.viewed.get(pid) or {}) + 1e-9 < ledger.evidence_need():
                        continue
                if pid not in ledger.viewed and "view" in names:
                    return [("view", {"product_id": pid})]
                if key not in ledger.stock and pid not in ledger.stock and "inspect_stock" in names:
                    return [("inspect_stock", {"product_id": pid, "sku": sku})]
                live_p = _as_float((ledger.stock.get(key) or ledger.stock.get(pid) or {}).get("price"))
                if live_p is None:
                    live_p = _as_float(row.get("price"))
                if not ledger.price_class_ok(live_p, pid, sku):
                    continue
                if (
                    ledger.categories
                    and ledger.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM, PROC_RECALL}
                    and not ledger._category_hit(row, ledger.viewed.get(pid) or {})
                ):
                    continue
                if ledger.live_ok(pid, sku) and "add_to_cart" in names:
                    plan = [("add_to_cart", {"product_id": pid, "sku": sku})]
                    if remaining <= 4 and max_calls >= 2 and "inspect_stock" in names:
                        plan.append(("inspect_stock", {"product_id": pid, "sku": sku}))
                    return plan[:max_calls]
                break

    if ledger.cart and "inspect_stock" in names:
        top = ledger.cart[-1]
        if top["key"] not in ledger.stock and top["product_id"] not in ledger.stock:
            return [("inspect_stock", {"product_id": top["product_id"], "sku": top["sku"]})]

    if (
        not ledger.candidates
        and 1 <= ledger.empty_searches <= max(2, len(ledger.seeds))
        and "search" in names
    ):
        seed = ledger.next_seed()
        if seed.lower() not in ledger.tried_queries:
            args = {"query": seed, "in_stock": True, "k": 10}
            if ledger.budget is not None and "max_price" in _tool_props(tools, "search"):
                args["max_price"] = ledger.budget
            return [("search", args)]

    return None



def _guard(
    pairs: list[tuple[str, dict[str, Any]]],
    *,
    ledger: Ledger,
    tools: list[dict[str, Any]],
    max_calls: int,
) -> list[tuple[str, dict[str, Any]]]:
    names = _tool_names(tools)
    out: list[tuple[str, dict[str, Any]]] = []

    if ledger.needs_ask and not ledger.clarified and "message" in names and not ledger.asked:
        risky = {"add_to_cart", "place_test_order", "remove_from_cart"}
        if any(n in risky for n, _ in pairs) or not any(n == "message" for n, _ in pairs):
            return [
                (
                    "message",
                    {
                        "content": (
                            "What firm specification is required for this purchase "
                            "besides budget and stock?"
                        )
                    },
                )
            ]

    for name, args in pairs:
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")

        if (
            ledger.needs_ask
            and not ledger.clarified
            and not ledger.asked
            and name in {"add_to_cart", "place_test_order"}
            and "message" in names
        ):
            out.append(
                (
                    "message",
                    {
                        "content": (
                            "What firm specification is required for this purchase "
                            "besides budget and stock?"
                        )
                    },
                )
            )
            continue

        if name == "add_to_cart" and pid and pid not in ledger.viewed and "view" in names:
            out.append(("view", {"product_id": pid}))
            continue

        if name == "view" and pid:
            # Never treat a prefer-model code as a product_id unless search returned it.
            prefer_codes = {t.lower() for t in ledger.prefer_models if any(ch.isdigit() for ch in t)}
            known_pids = {
                str(r.get("product_id") or "")
                for r in ledger.candidates.values()
            } | set(ledger.viewed.keys())
            if pid.lower() in prefer_codes and pid not in known_pids and "search" in names:
                q = pid
                colors = [t for t in ledger.prefer_models if t.isalpha()]
                if colors:
                    q = f"{pid} {colors[0]}"
                sargs: dict[str, Any] = {"query": q, "in_stock": True, "k": 12}
                if ledger.budget is not None and "max_price" in _tool_props(tools, "search"):
                    sargs["max_price"] = ledger.budget
                out.append(("search", sargs))
                continue

        # After clarification, block redundant "what firm spec" re-asks.
        if name == "message" and ledger.clarified:
            content = str(args.get("content") or "").lower()
            if re.search(r"\bfirm\b.*\b(?:spec|requirement)\b|\btell me if you ask\b", content):
                target = ledger.pick_cart_target()
                if target and target["action"] in names:
                    args2 = {"product_id": target["product_id"]}
                    if target["action"] != "view":
                        args2["sku"] = target["sku"]
                    out.append((target["action"], args2))
                    continue
                if "search" in names:
                    atoms = [a for a in ledger.atoms if any(ch.isdigit() for ch in a)]
                    soft = ledger.soft_tokens[:1]
                    q = f"{soft[0]} {atoms[0]}" if soft and atoms else (atoms[0] if atoms else ledger.next_seed())
                    sargs = {"query": q, "in_stock": True, "k": 12}
                    if ledger.budget is not None and "max_price" in _tool_props(tools, "search"):
                        sargs["max_price"] = ledger.budget
                    out.append(("search", sargs))
                    continue

        # Soft-prefer: bias first cart toward a soft-matching compliant item when ready.
        if (
            name == "add_to_cart"
            and ledger.procedure == PROC_SOFT
            and ledger.soft_tokens
            and pid
            and sku
            and ledger.cart_n == 0
        ):
            row = ledger.candidates.get(f"{pid}::{sku}") or ledger.candidates.get(pid) or {}
            viewed = ledger.viewed.get(pid) or {}
            if not ledger._soft_hit(row, viewed):
                alt = ledger.pick_cart_target()
                if (
                    alt
                    and alt["action"] == "add_to_cart"
                    and (alt["product_id"] != pid or alt["sku"] != sku)
                ):
                    alt_row = (
                        ledger.candidates.get(f"{alt['product_id']}::{alt['sku']}")
                        or ledger.candidates.get(alt["product_id"])
                        or {}
                    )
                    alt_view = ledger.viewed.get(alt["product_id"]) or {}
                    if ledger._soft_hit(alt_row, alt_view):
                        out.append(
                            (
                                "add_to_cart",
                                {"product_id": alt["product_id"], "sku": alt["sku"]},
                            )
                        )
                        continue

        # Prefer evidenced carts, but do not hard-block — late episodes must still order.
        if (
            name == "add_to_cart"
            and pid
            and sku
            and ledger.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM}
            and ledger.atoms
            and ledger.view_n < 12
        ):
            row = ledger.candidates.get(f"{pid}::{sku}") or ledger.candidates.get(pid) or {}
            viewed = ledger.viewed.get(pid) or {}
            if ledger.evidence_ratio(row, viewed) + 1e-9 < ledger.evidence_need():
                nxt = ledger.pick_cart_target()
                if nxt and nxt["action"] in names:
                    if nxt["action"] != "add_to_cart" or (
                        nxt["product_id"] != pid or nxt.get("sku") != sku
                    ):
                        args2 = {"product_id": nxt["product_id"]}
                        if nxt["action"] != "view":
                            args2["sku"] = nxt["sku"]
                        out.append((nxt["action"], args2))
                        continue

        # If the sku is already carted AND post-cart inspect passed, checkout instead.
        if name == "add_to_cart" and pid and sku:
            key = f"{pid}::{sku}"
            if (
                ledger.cart
                and any(c.get("key") == key for c in ledger.cart)
                and key not in ledger.post_cart_pending
                and ledger.live_ok(pid, sku)
                and "place_test_order" in names
            ):
                out.append(
                    (
                        "place_test_order",
                        _ground_order(
                            {"product_id": pid, "sku": sku},
                            tools=tools,
                            ledger=ledger,
                        ),
                    )
                )
                continue
            # Fresh cart add this turn — never combine with place_test_order.
            out.append((name, args))
            continue

        if name in {"search", "filter"}:
            q = str(args.get("query") or "").strip().lower()
            if q and q == ledger.last_search_query and ledger.same_search_streak >= 2:
                if ledger.candidates and "view" in names:
                    for _score, key, row in ledger.ranked():
                        pid2 = str(row.get("product_id") or "")
                        if pid2 and pid2 not in ledger.viewed and key not in ledger.failed:
                            out.append(("view", {"product_id": pid2}))
                            break
                    else:
                        out.append((name, args))
                    continue
                args = dict(args)
                args["query"] = ledger.next_seed()

        if name == "place_test_order" and pid and sku:
            key = f"{pid}::{sku}"
            # Intervention gate: reply to unanswered shopper questions before checkout.
            if (
                ledger.pending_q
                and _shopper_needs_verbal(ledger)
                and "message" in names
                and "message" not in ledger.actions[-10:]
            ):
                atom = next((a for a in ledger.atoms if any(ch.isdigit() for ch in a)), None)
                content = (
                    f"I am going with {pid}::{sku}"
                    + (f", confirming the firm {atom} requirement." if atom else ".")
                )
                out.append(("message", {"content": content[:400]}))
                continue
            if key in ledger.post_cart_pending and "inspect_stock" in names:
                out.append(("inspect_stock", {"product_id": pid, "sku": sku}))
                continue
            if ledger.must_recover and "remove_from_cart" in names:
                out.append(("remove_from_cart", {"product_id": pid, "sku": sku}))
                continue
            if ledger.over_budget(pid, sku):
                ledger.must_recover = True
                ledger.failed.add(key)
                if "remove_from_cart" in names:
                    out.append(("remove_from_cart", {"product_id": pid, "sku": sku}))
                elif "inspect_stock" in names:
                    out.append(("inspect_stock", {"product_id": pid, "sku": sku}))
                continue
            if ledger.needs_recheck and key not in ledger.checked and "inspect_stock" in names:
                out.append(("inspect_stock", {"product_id": pid, "sku": sku}))
                continue
            has_stock = key in ledger.stock or pid in ledger.stock
            if not has_stock and "inspect_stock" in names:
                out.append(("inspect_stock", {"product_id": pid, "sku": sku}))
                continue
            if has_stock and not ledger.live_ok(pid, sku):
                ledger.must_recover = True
                ledger.failed.add(key)
                if "remove_from_cart" in names:
                    out.append(("remove_from_cart", {"product_id": pid, "sku": sku}))
                continue
            if (
                "justification" in _required_args(tools, "place_test_order")
                and pid not in ledger.viewed
                and "view" in names
            ):
                out.append(("view", {"product_id": pid}))
                continue
            # Final hard gate: never emit place_test_order when live checks fail.
            if not ledger.live_ok(pid, sku):
                continue

        out.append((name, args))

    unique: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for name, args in out:
        stamp = name + ":" + json.dumps(args, sort_keys=True, default=str)
        if stamp in seen:
            continue
        seen.add(stamp)
        unique.append((name, args))
    # Never submit add_to_cart and place_test_order in the same turn group.
    names_present = {n for n, _ in unique}
    if "add_to_cart" in names_present and "place_test_order" in names_present:
        unique = [(n, a) for n, a in unique if n != "place_test_order"]
    return unique[:max_calls]


def _fallback_plan(
    *,
    ledger: Ledger,
    tools: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """Last-resort deterministic action when the model returns nothing usable."""
    names = _tool_names(tools)
    if ledger.cart and ledger.post_cart_pending and "inspect_stock" in names:
        top = ledger.cart[-1]
        return [("inspect_stock", {"product_id": top["product_id"], "sku": top["sku"]})]
    if ledger.cart and "place_test_order" in names:
        top = ledger.cart[-1]
        if (
            top["key"] not in ledger.post_cart_pending
            and not ledger.must_recover
            and not ledger.over_budget(top["product_id"], top["sku"])
            and ledger.live_ok(top["product_id"], top["sku"])
        ):
            return [
                (
                    "place_test_order",
                    _ground_order(
                        {"product_id": top["product_id"], "sku": top["sku"]},
                        tools=tools,
                        ledger=ledger,
                    ),
                )
            ]
    ready = ledger.ready_order()
    if ready and "place_test_order" in names:
        return [("place_test_order", _ground_order(dict(ready), tools=tools, ledger=ledger))]
    if ledger.cart and "inspect_stock" in names:
        top = ledger.cart[-1]
        if top["key"] not in ledger.stock and top["product_id"] not in ledger.stock:
            return [("inspect_stock", {"product_id": top["product_id"], "sku": top["sku"]})]
        if (
            not ledger.live_ok(top["product_id"], top["sku"])
            or ledger.over_budget(top["product_id"], top["sku"])
        ) and "remove_from_cart" in names:
            return [
                (
                    "remove_from_cart",
                    {"product_id": top["product_id"], "sku": top["sku"]},
                )
            ]
    nxt = ledger.recovery_step()
    if nxt and nxt["action"] in names:
        args = {"product_id": nxt["product_id"]}
        if nxt["action"] != "view":
            args["sku"] = nxt["sku"]
        return [(nxt["action"], args)]
    if ledger.candidates and "view" in names:
        for _score, key, row in ledger.ranked():
            pid = str(row.get("product_id") or "")
            if pid and pid not in ledger.viewed and key not in ledger.failed:
                return [("view", {"product_id": pid})]
    if "search" in names:
        return [("search", {"query": ledger.next_seed(), "in_stock": True, "k": 10})]
    return []


def _system_prompt(
    tools: list[dict[str, Any]],
    *,
    max_calls: int,
    max_steps: int,
    ledger: Ledger,
    needs_justification: bool,
    claim_fields: list[str],
) -> str:
    tool_lines: list[str] = []
    for spec in tools:
        fn = _fn(spec)
        if not fn or not isinstance(fn.get("name"), str):
            continue
        desc = str(fn.get("description") or "").strip()
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        arg_names = ", ".join(sorted(str(k) for k in props.keys()))
        tool_lines.append(
            f"- {fn['name']}: {desc}" + (f" Args: {arg_names}." if arg_names else "")
        )

    parts = [
        "You are a careful shopping agent. Obey the shopper request and every Task rules block.",
        "Copy product_id and sku only from observations — never invent identifiers.",
        "Work for ANY product category using only the shopper's stated constraints and tool observations.",
        "",
        f"Active procedure: {ledger.procedure}",
        *_procedure_playbook(ledger.procedure),
        "",
        "Shared process:",
        "1. If a firm requirement is deferred until you ask, message first.",
        "2. Discover with short constraint-focused search/filter queries; reformulate on weak hits.",
        "3. view/compare a shortlist and verify EVERY firm requirement on the exact sku using raw facts.",
        "4. First add_to_cart must be a genuine fully compliant choice, never a placeholder.",
        "5. Soft preferences guide ranking among compliant options; firm constraints always win.",
        "6. After add_to_cart OR any post-cart shopper update, inspect_stock the carted sku "
        "again before place_test_order — price and stock can change on cart events. "
        "If the fresh check fails budget or stock, remove_from_cart and recover.",
        "7. Never place_test_order in the same turn as add_to_cart.",
        "8. When verified by a post-cart inspect_stock, place_test_order immediately.",
        "9. Do not repeat the same search/filter query; broaden, drop optional tokens, or inspect hits.",
        f"10. Up to {max_calls} tool calls per turn; budget {max_steps} turns. Batch independent reads.",
        "",
        "Tools:",
        *tool_lines,
    ]
    if ledger.needs_ask and not ledger.clarified:
        parts.append(
            "This request is underspecified. Your first action should ask for the missing "
            "firm product specification."
        )
    if ledger.named:
        shown = ", ".join(f"{pid}::{sku}" for pid, sku in ledger.named[:12])
        parts.extend(
            [
                "Named variants to compare/view before carting:",
                shown,
                "Respect the stated tradeoff order (primary metric, then secondary, then price).",
            ]
        )
    if ledger.rank_axes:
        axes = ", ".join(
            f"{a}({'higher' if d > 0 else 'lower'})" for a, d in ledger.rank_axes
        )
        parts.append(f"Ranking axes from the request: {axes}.")
    if needs_justification:
        parts.extend(
            [
                "Checkout requires structured justification claims grounded in view/compare/"
                "inspect_stock of the FINAL product.",
                f"Allowed fields: {', '.join(claim_fields)}.",
                "Always include price and in_stock from the latest inspect_stock.",
                "Include exactly one claim for the shopper's constrained attribute "
                "(snake_case) with the observed value, plus one positive capability/reason "
                "field when the schema allows it.",
            ]
        )
    else:
        parts.append("Omit justification unless the schema requires it.")
    parts.append("Continue until an observation reports done=true.")
    return "\n".join(parts)


def _trim(messages: list[dict[str, Any]]) -> None:
    if len(messages) <= _HISTORY_CAP:
        return
    head = messages[:2]
    tail = messages[-(_HISTORY_CAP - 3) :]
    messages[:] = head + [
        {
            "role": "user",
            "content": "Earlier turns are summarized on the ledger. Prefer ledger + recent tools.",
        }
    ] + tail


def _problem_is_hard(
    *,
    ledger: Ledger,
    needs_justification: bool,
    max_steps: int,
) -> bool:
    """Cost-aware model choice from public problem cues only."""
    if needs_justification or ledger.needs_ask or ledger.named or ledger.rank_axes:
        return True
    if ledger.procedure in {PROC_ASK, PROC_SOFT, PROC_CLAIM, PROC_RANK, PROC_ADAPT, PROC_USE}:
        return True
    if ledger.soft_signal and ledger.firm_signal:
        return True
    if max_steps >= 24:
        return True
    return len(_preamble(ledger.query)) > 420


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    tools = list(policy.get("tools") or [])
    query = str(policy.get("query") or "")
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, min(10, int(policy.get("max_calls_per_turn", 1))))
    max_steps = int(policy["max_steps"])

    allowed = _claim_fields(tools)
    needs_justification = "justification" in _required_args(tools, "place_test_order")
    ledger = Ledger(query, allowed, tools=tools)
    models = _model_ladder(
        _problem_is_hard(
            ledger=ledger,
            needs_justification=needs_justification,
            max_steps=max_steps,
        )
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _system_prompt(
                tools,
                max_calls=max_calls,
                max_steps=max_steps,
                ledger=ledger,
                needs_justification=needs_justification,
                claim_fields=sorted(allowed) or ["price", "in_stock"],
            ),
        },
        {"role": "user", "content": query},
    ]
    dialogue: list[dict[str, Any]] = []

    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        action_pairs: list[tuple[str, dict[str, Any]]] = []
        assistant_content = ""

        forced = _forced_plan(
            ledger=ledger,
            tools=tools,
            max_calls=max_calls,
            remaining=remaining,
        )
        if forced is not None:
            action_pairs = forced
        else:
            _trim(messages)
            ask = list(messages)
            ask.append({"role": "user", "content": ledger.board(turn, max_steps)})
            tool_calls: list[dict[str, Any]] = []
            for attempt in range(2):
                inference = _chat(models=models, messages=ask, tools=tools)
                assistant = inference["choices"][0]["message"]
                assistant_content = assistant.get("content") or ""
                tool_calls = list(assistant.get("tool_calls") or [])[:max_calls]
                if tool_calls:
                    break
                ask.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {
                            "role": "user",
                            "content": (
                                "Tool call required. Clarify, inspect, recover, or order."
                                if attempt == 0
                                else "Call one supplied tool with valid JSON arguments."
                            ),
                        },
                    ]
                )
            for call in tool_calls:
                try:
                    name = call["function"]["name"]
                    if name not in _tool_names(tools):
                        continue
                    args = _normalize(
                        name,
                        _arguments(call),
                        tools=tools,
                        ledger=ledger,
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                action_pairs.append((name, args))
            action_pairs = _guard(
                action_pairs, ledger=ledger, tools=tools, max_calls=max_calls
            )

        # Checkout immediately once the carted sku passes live checks.
        ready = ledger.ready_order()
        if (
            ready
            and "place_test_order" in _tool_names(tools)
            and not ledger.must_recover
            and not ledger.post_cart_pending
            and not any(
                ledger.over_budget(ready["product_id"], ready["sku"]) for _ in (0,)
            )
            and not any(n == "place_test_order" for n, _ in action_pairs)
            and not any(n == "remove_from_cart" for n, _ in action_pairs)
        ):
            action_pairs = [
                ("place_test_order", _ground_order(dict(ready), tools=tools, ledger=ledger))
            ]

        rebuilt: list[tuple[str, dict[str, Any]]] = []
        for name, args in action_pairs:
            if name == "place_test_order":
                args = _ground_order(dict(args), tools=tools, ledger=ledger)
            else:
                args = _normalize(name, args, tools=tools, ledger=ledger)
            rebuilt.append((name, args))
        action_pairs = rebuilt
        # Hard split: never add_to_cart and place_test_order together.
        if any(n == "add_to_cart" for n, _ in action_pairs):
            action_pairs = [(n, a) for n, a in action_pairs if n != "place_test_order"]
        # Never order while a post-cart inspect is still required.
        if ledger.post_cart_pending or ledger.needs_recheck:
            sanitized: list[tuple[str, dict[str, Any]]] = []
            for name, args in action_pairs:
                if name != "place_test_order":
                    sanitized.append((name, args))
                    continue
                pid = str(args.get("product_id") or "")
                sku = str(args.get("sku") or "")
                if pid and sku and "inspect_stock" in _tool_names(tools):
                    sanitized.append(("inspect_stock", {"product_id": pid, "sku": sku}))
            action_pairs = sanitized or action_pairs

        if not action_pairs:
            action_pairs = _fallback_plan(ledger=ledger, tools=tools)
        if not action_pairs:
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": "Previous arguments were invalid. Call a schema-valid tool.",
                    },
                ]
            )
            continue

        prepared: list[dict[str, Any]] = []
        for index, (name, args) in enumerate(action_pairs, start=1):
            prepared.append(
                {
                    "id": f"{problem_id}-t{turn}-c{index}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )
            ledger.remember(name, args)

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": prepared,
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
                    "action": {"name": name, "args": args},
                }
                for index, (name, args) in enumerate(action_pairs, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            retry_id = f"{group_id}-retry"
            envelope = {**envelope, "call_id": retry_id, "idempotency_key": retry_id}
            result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")

        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": prepared,
                "environment_result": result,
            }
        )

        for (name, args), tool_call, call_result in zip(
            action_pairs, prepared, result["calls"], strict=True
        ):
            observation = call_result.get("observation") or {}
            if not isinstance(observation, dict):
                observation = {"raw": observation}
            envelope, payload = _unwrap_observation(observation)
            # Ledger reads the unwrapped tool body; keep envelope error/done semantics.
            merged = {**payload}
            if envelope.get("error") is not None:
                merged.setdefault("error", envelope.get("error"))
            ledger.ingest(name, args, observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _compact_obs(merged if merged else observation),
                }
            )

        user_message = result.get("user_message")
        if isinstance(user_message, dict) and user_message.get("content"):
            content = str(user_message["content"])
            # Environment may join multiple simulator utterances with newlines.
            for chunk in re.split(r"\n+", content):
                chunk = chunk.strip()
                if chunk:
                    ledger.note_shopper(chunk)
            messages.append({"role": "user", "content": content})
        elif isinstance(user_message, str) and user_message.strip():
            for chunk in re.split(r"\n+", user_message):
                chunk = chunk.strip()
                if chunk:
                    ledger.note_shopper(chunk)
            messages.append({"role": "user", "content": user_message})

        if any(
            isinstance(call.get("observation"), dict) and call["observation"].get("done")
            for call in result.get("calls") or []
        ):
            break

    return dialogue


__all__ = ["agent_main"]
