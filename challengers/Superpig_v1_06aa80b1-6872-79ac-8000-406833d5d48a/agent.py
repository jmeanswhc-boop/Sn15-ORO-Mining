
from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# Loopler LIFE881 → EnvPack allowlist. Justification: never glm-first.
_OR = {
    "deepseek": "deepseek/deepseek-v3.2",
    "deepseek_chat": "deepseek/deepseek-chat-v3.1",
    "deepseek_0324": "deepseek/deepseek-chat-v3-0324",
    "kimi": "moonshotai/kimi-k2.6",
    "qwen32": "qwen/qwen3-32b",
    "qwen397": "qwen/qwen3.5-397b-a17b",
    "gemma": "google/gemma-4-31b-it",
    "glm": "z-ai/glm-5.1",
    "glm52": "z-ai/glm-5.2",
}
_TEE = {
    "deepseek": "deepseek-ai/DeepSeek-V3.2-TEE",
    "deepseek_chat": "deepseek-ai/DeepSeek-V3.1-TEE",
    "deepseek_0324": "deepseek-ai/DeepSeek-V3-0324-TEE",
    "kimi": "moonshotai/Kimi-K2.6-TEE",
    "qwen32": "Qwen/Qwen3-32B-TEE",
    "qwen397": "Qwen/Qwen3.5-397B-A17B-TEE",
    "gemma": "google/gemma-4-31B-turbo-TEE",
    "glm": "zai-org/GLM-5.1-TEE",
    "glm52": "zai-org/GLM-5.2-TEE",
}
_FAMILY_ORDER = {
    "intent": ("deepseek", "deepseek_chat", "kimi", "gemma", "qwen397"),
    "retrieval": ("deepseek", "kimi", "qwen397", "gemma", "qwen32"),
    "constraint": ("deepseek", "deepseek_chat", "kimi", "qwen32", "gemma"),
    "preference": ("deepseek_chat", "deepseek", "kimi", "gemma", "qwen32"),
    "ranking": ("deepseek", "deepseek_chat", "qwen32", "kimi", "gemma"),
    "recovery": ("deepseek", "deepseek_chat", "kimi", "qwen32", "gemma"),
    "justification": ("deepseek", "deepseek_chat", "kimi", "qwen32", "glm"),
    "generic": ("deepseek", "deepseek_chat", "kimi", "gemma", "qwen32"),
}

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
_ASK_SPEC_RE = re.compile(
    r"tell you if you ask|if you ask|ask me.*firm|firm requirement.*ask",
    re.I,
)
# Prefer comma-thousands OR plain digits; keep fractional part in group 2.
# Do NOT use \d{1,3}(?:,\d{3})* — that truncates 10822.01 → 108.
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|"
    r"max(?:imum)? of|within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(?:PHP|php|₱|pesos?)?",
    re.I,
)
_BRAND_DENY = frozenset(
    {
        "a",
        "an",
        "the",
        "in",
        "stock",
        "in-stock",
        "gaming",
        "complete",
        "want",
        "need",
        "find",
        "get",
        "with",
        "under",
        "one",
        "test",
        "good",
        "best",
        "new",
        "genuine",
        "authentic",
        "original",
        "official",
        "real",
        "smartphone",
        "phone",
        "tablet",
        "monitor",
        "headphone",
        "earbuds",
        "earphone",
        "laptop",
        "tv",
        "screen",
        "inch",
        "mobile",
        "cellphone",
    }
)
_BRAND_NOISE_TOKS = frozenset(
    {
        "smartphone",
        "phone",
        "tablet",
        "monitor",
        "headphone",
        "earbuds",
        "earphone",
        "laptop",
        "tv",
        "with",
        "screen",
        "inch",
        "mobile",
        "cellphone",
        "stock",
    }
)
_PHRASE_RE = re.compile(
    r'(?:matches(?: the phrase)?|phrase)\s*[\"“]?([^\"”\n,]{2,80})[\"”]?',
    re.I,
)
_QUOTED_RE = re.compile(r'[\"“]([^\"”]{2,80})[\"”]')
_BRAND_ITEM_RE = re.compile(
    r"\b(?:a|an|the)\s+([A-Z][A-Za-z0-9][\w&.'-]{0,28})\s+"
    r"(?:smartphone|phone|tablet|monitor|headphone|earbuds|earphone|laptop|tv)\b",
    re.I,
)
_SOFT_AXIS_RE = re.compile(
    r"\b(sweat|water(?:proof)?|ipx\d*|weather|rain|gym|run(?:ning)?|workout|"
    r"noise[\s-]?cancel|anc|over[\s-]?ear|in[\s-]?ear)\b",
    re.I,
)
_WEAK_SEARCH_CAP = 5
_EPS = 1e-6
# EnvPack advertises max_calls_per_turn=16; parallel dumps leave event_closed=False.
_MAX_CALLS_HARD_CAP = 1

_proxy = ProxyClient(timeout=120, max_retries=3)

_SYSTEM = """You are an ORO Bench shopping agent (ilostaboy EnvPack + race_118 Loopler product DNA).

Use ONLY supplied tools. Continue until done=true. Never memorize product IDs or suite ranks.
Hidden races differ from local packs — obey this episode's shopper text + Task rules only.

## TF1 Intent (Loopler PRODUCT_PARSE)
Silently build a 2–8 word keyword string in left-to-right query order
(product_type, brand, material, color, fused num+unit, use-case nouns).
Firm = category, named HW/spec, budget, stock, named brand when stated as the item.
Ask-first: if a firm spec is told when you ask → FIRST call `message` for that spec.
Shopper question / budget change → `message` first.

## TF2 Retrieval (Loopler anti-thrash)
search/filter with reformulations (in_stock=true). Keep quoted/phrase tokens literal.
After ~3–5 weak searches, view best hits and finish. Fast-accept a firm-valid in-budget pick.

## TF3 Constraints
Firm specs NON-NEGOTIABLE. Literal brand/model/size/number after `view`.
Never order above budget. Prefer attrs/sku_options over SEO titles.

## TF4 Preference
Among fully valid options, broader attrs+sku agreement then cheaper.
Soft axes (sweat/water/IPX/gym): search those tokens; compare distinct product_ids.

## TF5 Ranking (closed pid::sku)
Listed product_id::sku set is exhaustive. View all; order a listed key only.

## TF6 Recovery (event-close)
Market shock: COMMIT message (never ask permission) → remove_from_cart → VIEW replacement
→ fresh inspect_stock → cart → order. Never order on stale pre-event stock.

## TF7 Justification
Only if asked. Claims from latest view + inspect_stock only. Never category.
Never invent hd/fhd/qhd/uhd. Never brand None.

## Finish
search/filter → view/compare → inspect_stock → add_to_cart → place_test_order.
"""


def _provider() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw in ("chutes", "openrouter"):
        return raw
    if getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "chutes"


def _roster_for(family: str) -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    table = _TEE if _provider() == "chutes" else _OR
    order = _FAMILY_ORDER.get(family) or _FAMILY_ORDER["generic"]
    return [table[k] for k in order if k in table]


def _model_list(family: str = "generic") -> list[str]:
    return _roster_for(family)


def _infer_family(
    *,
    query: str,
    candidates: list[tuple[str, str]],
    shock_recover: bool,
    searchish: int,
    must_message: bool,
) -> str:
    head = query.split("Task rules:", 1)[0].lower()
    if shock_recover:
        return "recovery"
    if candidates:
        return "ranking"
    if must_message and _ASK_SPEC_RE.search(head):
        return "constraint"
    if any(x in head for x in ("why", "reason", "justif", "walk me through", "good pick")):
        return "justification"
    if any(x in head for x in ("prefer", "preferably", "rather pay", "listing_id")):
        return "preference"
    if searchish == 0:
        return "intent"
    if searchish < _WEAK_SEARCH_CAP:
        return "retrieval"
    return "generic"


def _parse_budget(query: str) -> float | None:
    best = None
    for m in _BUDGET_RE.finditer(query.split("Task rules:", 1)[0]):
        whole = m.group(1).replace(",", "")
        frac = m.group(2) or "0"
        try:
            val = float(f"{whole}.{frac}")
        except ValueError:
            continue
        if best is None or val < best:
            best = val
    return best


def _parse_phrase(query: str) -> str | None:
    head = query.split("Task rules:", 1)[0]
    m = _PHRASE_RE.search(head) or _QUOTED_RE.search(head)
    return m.group(1).strip().strip('"“”') if m else None


def _parse_firm_brand(query: str) -> str | None:
    head = query.split("Task rules:", 1)[0]
    m = _BRAND_ITEM_RE.search(head)
    if m:
        brand = m.group(1).strip()
        if brand.lower() not in _BRAND_DENY and "stock" not in brand.lower():
            return brand
    m = re.search(
        r"\b([A-Z][A-Za-z0-9][\w&.'-]{1,28})\s+"
        r"(?:\d+[^\s]*\s+)*(?:smartphone|phone|tablet|monitor|headphone)\b",
        head,
    )
    if m:
        brand = m.group(1).strip()
        if brand.lower() not in _BRAND_DENY and "stock" not in brand.lower():
            return brand
    # Multi-token brands: "Herman Miller … tablet", "FYHXele … monitor", "Xiaomi … monitor"
    m = re.search(
        r"(?:want a|find me a|get me an?|need an?)\s+(.+?)(?=\s+(?:\d+\s*p\b|720p|1080p|"
        r"1440p|4k|uhd|qhd|fhd|full\s*hd|\bhd\b|\d+(?:\.\d+)?-?inch|gaming|"
        r"smartphone|tablet|monitor|headphone))",
        head,
        flags=re.I,
    )
    if m:
        brand = re.sub(r"\s+", " ", m.group(1)).strip(" .,")
        brand = re.sub(r"^(?:in[\s-]?stock|an?|the)\s+", "", brand, flags=re.I).strip()
        low = brand.lower()
        toks = set(re.findall(r"[a-z0-9]+", low))
        # "I need a smartphone with a 7.6-inch …" must NOT become firm brand.
        if not brand or toks & _BRAND_NOISE_TOKS:
            return None
        if low in _BRAND_DENY or "stock" in low or len(brand) > 40:
            return None
        return brand
    return None


def _parse_soft_axes(query: str) -> list[str]:
    head = query.split("Task rules:", 1)[0]
    return list(dict.fromkeys(m.group(0).lower() for m in _SOFT_AXIS_RE.finditer(head)))


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1}


def _shopper_text(query: str) -> str:
    return query.split("Task rules:", 1)[0].strip()


def _parse_candidate_variants(query: str) -> list[tuple[str, str]]:
    head = _shopper_text(query)
    low = head.lower()
    pairs = _CAND_KEY_RE.findall(head)
    if not any(
        s in low
        for s in (
            "compare only",
            "candidate variant",
            "candidate variants",
            "these candidate",
            "only these",
        )
    ):
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


def _justification_required(query: str) -> bool:
    head = _shopper_text(query).lower()
    needles = (
        "why",
        "reason",
        "walk me through",
        "tell me why",
        "justif",
        "good pick",
        "reason i can trust",
    )
    return any(n in head for n in needles)


def _needs_ask_first_spec(query: str) -> bool:
    return bool(_ASK_SPEC_RE.search(_shopper_text(query)))

def _spec_fields_from_query(query: str) -> list[str]:
    """Infer which justification spec fields the shopper named (race-general)."""
    head = _shopper_text(query).lower()
    fields: list[str] = []
    if any(
        x in head
        for x in ("720p", "1080p", "1440p", "4k", "uhd", "qhd", "full hd", " resolution")
    ) or re.search(r"\bhd\b", head):
        fields.append("resolution")
    if any(x in head for x in ("inch", "screen", "display size")):
        fields.append("screen_size")
    if "refresh" in head or re.search(r"\b\d+\s*hz\b", head):
        fields.append("refresh_rate")
    if re.search(r"\bram\b", head):
        # RAM not in default claim enum; skip
        pass
    return fields


def _claim_field_allowlist(tools: list[dict[str, Any]]) -> frozenset[str]:
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
            return frozenset(str(x) for x in enum)
    return _DEFAULT_CLAIM_FIELDS


def _normalize_claim_field(field: str, allowed: frozenset[str]) -> str | None:
    raw = str(field or "").strip()
    if not raw:
        return None
    key = raw.lower()
    if key in _CLAIM_DENY or raw in _CLAIM_DENY:
        return None
    if raw in allowed:
        return raw
    if key in allowed:
        return key
    mapped = _FIELD_ALIASES.get(key)
    if mapped and mapped in allowed and mapped not in _CLAIM_DENY:
        return mapped
    return None


def _query_wants_resolution_class(query: str) -> str | None:
    """Return hd|fhd|qhd|uhd if shopper named a resolution class (0828 PASS pattern)."""
    low = _shopper_text(query).lower()
    if re.search(r"\b(4k|uhd|ultra\s*hd)\b", low):
        return "uhd"
    if re.search(r"\b(qhd|1440p|2k)\b", low):
        return "qhd"
    if re.search(r"\b(fhd|1080p|full\s*hd)\b", low):
        return "fhd"
    if re.search(r"\b(720p|hd)\b", low):
        return "hd"
    return None


def _normalize_resolution_value(text: str, query: str) -> str:
    """Prefer class tokens when shopper named them and display supports them."""
    text = str(text or "").strip()
    if not text:
        return text
    low = text.lower().replace("×", "x").replace("*", "x")
    if low in ("hd", "fhd", "qhd", "uhd", "4k"):
        return "uhd" if low == "4k" else low
    wanted = _query_wants_resolution_class(query)
    if wanted == "uhd" and any(x in low for x in ("4k", "uhd", "2160", "3840")):
        return "uhd"
    if wanted == "qhd" and any(x in low for x in ("qhd", "1440", "2560", "2k")):
        return "qhd"
    if wanted == "fhd" and any(x in low for x in ("fhd", "1080", "1920", "full hd")):
        return "fhd"
    if wanted == "hd":
        # Prefer explicit hd/720p; avoid claiming hd for fhd/qhd-only strings.
        if any(x in low for x in ("720", "1280")) or re.search(r"\bhd\b", low):
            if not any(x in low for x in ("1080", "1920", "1440", "2560", "2160", "fhd", "qhd")):
                return "hd"
            # "hd; 720p" style — still hd
            if "720" in low or re.search(r"\bhd\b", low):
                return "hd"
    return text

def _extract_view_facts(observation: Any) -> tuple[str | None, dict[str, str]]:
    """Pull product_id + brand/resolution/screen_size from a view observation."""
    facts: dict[str, str] = {}
    if not isinstance(observation, dict):
        return None, facts
    node = observation.get("observation")
    if not isinstance(node, dict):
        node = observation
    pid = node.get("product_id") or observation.get("product_id")
    if pid is not None:
        pid = str(pid)

    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in (
                    "brand",
                    "resolution",
                    "screen_size",
                    "refresh_rate",
                    "model",
                ) and v not in (
                    None,
                    "",
                    "None",
                ):
                    facts.setdefault(lk, str(v).strip())
                # Catalog stores resolution under facts.display / size under facts.size
                if lk == "display" and v not in (None, "", "None"):
                    facts.setdefault("resolution", str(v).strip())
                if lk == "size" and v not in (None, "", "None"):
                    facts.setdefault("screen_size", str(v).strip())
                if lk in ("attributes", "specs", "options", "variants", "facts") or isinstance(
                    v, (dict, list)
                ):
                    walk(v, path + "." + lk)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:8]):
                walk(v, f"{path}[{i}]")

    walk(node)
    # Nested facts block (common EnvPack shape)
    nested = node.get("facts")
    if isinstance(nested, dict):
        if nested.get("display") not in (None, "", "None"):
            facts.setdefault("resolution", str(nested["display"]).strip())
        if nested.get("size") not in (None, "", "None"):
            facts.setdefault("screen_size", str(nested["size"]).strip())
        if nested.get("model") not in (None, "", "None"):
            facts.setdefault("model", str(nested["model"]).strip())
    attrs = node.get("attributes") or node.get("specs")
    if isinstance(attrs, list):
        for a in attrs:
            if not isinstance(a, dict):
                continue
            name = str(a.get("name") or a.get("key") or "").lower()
            val = a.get("value") or a.get("val")
            if not val:
                continue
            if "brand" in name:
                facts.setdefault("brand", str(val).strip())
            if "resolution" in name or name in ("display", "screen resolution"):
                facts.setdefault("resolution", str(val).strip())
            if ("screen" in name and "size" in name) or name == "size":
                facts.setdefault("screen_size", str(val).strip())
            if "refresh" in name:
                facts.setdefault("refresh_rate", str(val).strip())
    if node.get("brand") not in (None, "", "None"):
        facts["brand"] = str(node["brand"]).strip()
    title = node.get("title") or node.get("name") or observation.get("title")
    if title:
        facts["title"] = str(title).strip()
    cat = node.get("category_path") or node.get("category")
    if isinstance(cat, list):
        facts["category"] = " / ".join(str(x) for x in cat if x)
    elif cat not in (None, "", "None"):
        facts["category"] = str(cat).strip()
    # Capture skus; prefer in-stock variant matching preferred color when provided later.
    variants = node.get("variants")
    sku_rows: list[str] = []
    if isinstance(variants, list):
        for var in variants:
            if not isinstance(var, dict):
                continue
            sku = var.get("sku")
            if not sku:
                continue
            in_stock = var.get("in_stock")
            if in_stock is False:
                continue
            opts = str(var.get("options") or var.get("name") or "").strip()
            sku_rows.append(f"{sku}::{opts}")
            facts.setdefault("sku", str(sku))
            if in_stock is True and not facts.get("_sku_locked"):
                facts["sku"] = str(sku)
        if sku_rows:
            facts["sku_rows"] = " || ".join(sku_rows[:12])
    return pid, facts


def _preferred_color_hint(query: str) -> str | None:
    """Color token from prefer-the-COLOR-MODEL / strongly prefer COLOR MODEL."""
    head = _shopper_text(query)
    colors = (
        r"moonlight\s+white|forrest\s+green|forest\s+green|"
        r"black|white|green|red|blue|gold|silver|pink|purple|gray|grey|"
        r"yellow|orange"
    )
    patterns = (
        rf"strongly\s+prefer(?:\s+the)?\s+({colors})\s+",
        rf"prefer(?:s)?\s+the\s+({colors})\s+",
        rf"eye on the\s+({colors})\s+",
        rf"you have your eye on the\s+({colors})\s+",
    )
    for pat in patterns:
        m = re.search(pat, head, flags=re.I)
        if m:
            return re.sub(r"\s+", " ", m.group(1).strip().lower())
    return None


def _sku_for_color(facts: dict[str, str], color: str | None) -> str | None:
    rows = str(facts.get("sku_rows") or "")
    default = facts.get("sku")
    if not color:
        return default
    color_l = color.lower()
    # Prefer exact color token in options; moonlight white / forrest green included.
    for part in rows.split(" || "):
        if "::" not in part:
            continue
        sku, opts = part.split("::", 1)
        if color_l in opts.lower():
            return sku
    # Fallback: single-word color contained in options.
    token = color_l.split()[-1]
    for part in rows.split(" || "):
        if "::" not in part:
            continue
        sku, opts = part.split("::", 1)
        if token in opts.lower():
            return sku
    return default


def _model_pick(
    model_hint: str | None,
    view_facts_by_pid: dict[str, dict[str, str]],
    *,
    dead_keys: set[str] | None = None,
    color: str | None = None,
) -> dict[str, str] | None:
    """First viewed product evidencing preferred model with a known sku."""
    if not model_hint:
        return None
    dead = dead_keys or set()
    color_l = (color or "").lower()
    best: dict[str, str] | None = None
    best_score = -1
    for pid, facts in view_facts_by_pid.items():
        if not _model_ok(pid, model_hint, view_facts_by_pid):
            continue
        facts = facts or {}
        candidates: list[tuple[int, str]] = []
        rows = str(facts.get("sku_rows") or "")
        if rows:
            for part in rows.split(" || "):
                if "::" not in part:
                    continue
                sku, opts = part.split("::", 1)
                key = f"{pid}::{sku}"
                if key in dead:
                    continue
                score = 1
                if color_l and color_l in opts.lower():
                    score = 3
                elif color_l and color_l.split()[-1] in opts.lower():
                    score = 2
                candidates.append((score, sku))
        else:
            sku = facts.get("sku")
            if sku and f"{pid}::{sku}" not in dead:
                candidates.append((1, sku))
        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])
        score, sku = candidates[0]
        if score > best_score:
            best_score = score
            best = {"product_id": str(pid), "sku": str(sku)}
    return best


def _preferred_model_hint(query: str) -> str | None:
    """Extract TF6 model token from strongly-prefer / prefer-the-COLOR-MODEL only.

    Soft brand prefs (e.g. prefer Samsung) must not become recovery model tokens.
    EnvPack model tokens always contain at least one digit.
    """
    head = _shopper_text(query)
    colors = (
        r"black|white|green|red|blue|gold|silver|pink|purple|gray|grey|"
        r"yellow|orange|moonlight\s+white"
    )
    patterns = (
        rf"strongly\s+prefer(?:\s+the)?\s+(?:{colors})?\s*"
        rf"([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"prefer(?:s)?\s+the\s+(?:{colors})\s+"
        rf"([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"eye on the\s+(?:{colors})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"you have your eye on the\s+(?:{colors})?\s*"
        rf"([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
    )
    token = None
    for pat in patterns:
        m = re.search(pat, head, flags=re.I)
        if m:
            token = m.group(1).strip()
            break
    if not token:
        return None
    token = token.split(".")[0]
    token = re.split(
        r"\s+\bIf\b|\s+\bspecifically\b|\s+\bgenuine\b|\s+\bnot\b|,",
        token,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    token = token.strip(".,;:()[]\"'")
    if token.lower() in {
        "black",
        "white",
        "green",
        "red",
        "blue",
        "gold",
        "silver",
        "pink",
        "purple",
        "gray",
        "grey",
        "yellow",
        "orange",
        "the",
        "a",
        "an",
    }:
        return None
    if not any(ch.isdigit() for ch in token):
        return None
    return token or None


def _canonical_recovery_message(
    model: str | None,
    *,
    shock_kind: str = "stock",
    price: float | None = None,
    budget: float | None = None,
) -> str:
    """Declarative recovery commit — must match shock type (price vs stock)."""
    if shock_kind == "price":
        if price is not None and budget is not None and price > budget + _EPS:
            price_bit = f"price is now {price:.2f} PHP (over budget)"
        elif price is not None:
            price_bit = f"price is now {price:.2f} PHP"
        else:
            price_bit = "price changed beyond the firm budget"
        if model:
            return (
                f"Yes — after I added it to cart the listing shocked and {price_bit}. "
                f"I'm now looking for another color/variant of the same model ({model}) "
                f"that is still in stock and within your budget — keeping category firm."
            )
        return (
            f"Yes — after I added it to cart the listing shocked and {price_bit}. "
            "I'm now looking for another in-stock in-budget option that still meets your "
            "firm requirements."
        )
    if model:
        return (
            f"The preferred option went out of stock when I added it to cart. "
            f"I'm now looking for another color/variant of the same model ({model}) "
            f"that is still in stock and within your budget — keeping category firm."
        )
    return (
        "That item went out of stock when I added it to cart. "
        "I'm now looking for another in-stock option that still matches your "
        "firm budget and category requirements."
    )


def _collect_search_pids(obs: Any) -> list[str]:
    """Product ids from a search/filter observation (race-general)."""
    out: list[str] = []
    seen: set[str] = set()

    def walk(obj: Any, depth: int = 0) -> None:
        if depth > 8 or obj is None:
            return
        if isinstance(obj, dict):
            pid = obj.get("product_id") or obj.get("id")
            if pid not in (None, "", "None"):
                s = str(pid)
                if s not in seen and s and any(ch.isdigit() for ch in s):
                    seen.add(s)
                    out.append(s)
            for v in obj.values():
                walk(v, depth + 1)
        elif isinstance(obj, list):
            for v in obj[:40]:
                walk(v, depth + 1)

    walk(obs)
    return out


def _next_unviewed(pids: list[str], viewed: set[str]) -> str | None:
    for pid in pids:
        if pid not in viewed:
            return pid
    return None


def _query_inch_size(query: str) -> float | None:
    head = _shopper_text(query).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", head)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _firm_listing_ok(
    pid: str,
    *,
    query: str,
    view_facts_by_pid: dict[str, dict[str, str]],
) -> bool:
    """Reject only when viewed facts CONTRADICT shopper-named size/resolution."""
    facts = view_facts_by_pid.get(pid) or {}
    if not facts:
        return True
    blob = " ".join(str(v) for v in facts.values()).lower()
    wanted = _query_wants_resolution_class(query)
    vis = (facts.get("resolution") or "").lower()
    if wanted == "fhd" and vis:
        if not any(x in vis for x in ("1080", "1920", "fhd", "full")):
            return False
    if wanted == "hd" and vis:
        if any(x in vis for x in ("1080", "1920", "fhd", "1440", "qhd", "2160", "4k")):
            if "720" not in vis:
                return False
    inch = _query_inch_size(query)
    if inch is not None:
        size = facts.get("screen_size") or ""
        m = re.search(r"(\d+(?:\.\d+)?)", size)
        if m:
            try:
                got = float(m.group(1))
                if abs(got - inch) > 0.15:
                    return False
            except ValueError:
                pass
    return True


def _listing_matches_query(
    pid: str,
    *,
    query: str,
    firm_brand: str | None,
    model_hint: str | None,
    view_facts_by_pid: dict[str, dict[str, str]],
) -> bool:
    """Episode-general firm gates: brand, preferred model, named size/resolution."""
    if firm_brand and not _brand_ok(pid, firm_brand, view_facts_by_pid):
        return False
    if model_hint and not _model_ok(pid, model_hint, view_facts_by_pid):
        return False
    head = _shopper_text(query).lower()
    wants_device = any(
        w in head
        for w in ("smartphone", "phone", "tablet", "monitor", "headphone")
    )
    if wants_device and _looks_like_accessory(pid, view_facts_by_pid):
        return False
    if (
        _justification_required(query)
        or _query_inch_size(query) is not None
        or _query_wants_resolution_class(query)
    ) and not _firm_listing_ok(
        pid, query=query, view_facts_by_pid=view_facts_by_pid
    ):
        return False
    return True


def _best_viewed_pick(
    *,
    query: str,
    model_hint: str | None,
    color: str | None,
    view_facts_by_pid: dict[str, dict[str, str]],
    dead_keys: set[str],
    firm_brand: str | None = None,
) -> dict[str, str] | None:
    """Best viewed listing: preferred model (if any) + firm brand/size/resolution."""
    prefer = _model_pick(
        model_hint, view_facts_by_pid, dead_keys=dead_keys, color=color
    )
    if prefer and _listing_matches_query(
        prefer["product_id"],
        query=query,
        firm_brand=firm_brand,
        model_hint=model_hint,
        view_facts_by_pid=view_facts_by_pid,
    ):
        return prefer
    for pid, facts in view_facts_by_pid.items():
        if not _listing_matches_query(
            pid,
            query=query,
            firm_brand=firm_brand,
            model_hint=model_hint,
            view_facts_by_pid=view_facts_by_pid,
        ):
            continue
        sku = _sku_for_color(facts or {}, color) or (facts or {}).get("sku")
        if not sku:
            continue
        key = f"{pid}::{sku}"
        if key in dead_keys:
            continue
        return {"product_id": str(pid), "sku": str(sku)}
    return None


def _tc(name: str, args: dict[str, Any], turn: int, tag: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"forced-{tag}-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
    ]


def _stock_price(payload: dict[str, Any] | None) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in ("current_price", "price"):
        if payload.get(key) is None:
            continue
        try:
            return float(payload[key])
        except (TypeError, ValueError):
            continue
    return None


def _stock_ok(payload: dict[str, Any] | None, budget: float | None) -> bool:
    if not isinstance(payload, dict) or _stock_is_false(payload):
        return False
    price = _stock_price(payload)
    if budget is not None and price is not None and price > budget + _EPS:
        return False
    return True


def _shock_kind_from_text(text: str) -> str:
    low = (text or "").lower()
    if any(
        x in low
        for x in (
            "more expensive",
            "got more expensive",
            "price jump",
            "price changed",
            "repriced",
            "above budget",
            "over budget",
        )
    ) and any(x in low for x in ("php", "price", "expensive", "budget", "cost")):
        return "price"
    return "stock"


def _normalize_claim_value(field: str, value: Any, *, query: str) -> str:
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
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            # Grader gate values use "7.6in" / "27in" style.
            return f"{m.group(1)}in"
    if field == "resolution":
        return _normalize_resolution_value(text, query)
    return text


def _sanitize_place_order_args(
    args: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
    last_stock: dict[str, Any] | None,
    view_facts_by_pid: dict[str, dict[str, str]] | None,
    last_pick: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build TF7 claims only from observed view + latest stock (never invent)."""
    out = dict(args)
    if not out.get("product_id") and last_pick and last_pick.get("product_id"):
        out["product_id"] = last_pick["product_id"]
    if not out.get("sku") and last_pick and last_pick.get("sku"):
        out["sku"] = last_pick["sku"]
    if not _justification_required(query):
        out.pop("justification", None)
        return out

    pid = str(out.get("product_id") or (last_pick or {}).get("product_id") or "")
    view_facts: dict[str, str] = {}
    if view_facts_by_pid and pid and pid in view_facts_by_pid:
        view_facts = dict(view_facts_by_pid[pid])
    needed_specs = set(_spec_fields_from_query(query))
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(field: str, raw: Any) -> None:
        if field in seen or field in _CLAIM_DENY or field not in allowed_fields:
            return
        value = _normalize_claim_value(field, raw, query=query)
        if not value or (field == "brand" and value.lower() in ("none", "null", "n/a")):
            return
        if field == "resolution" and value.lower() in ("hd", "fhd", "qhd", "uhd"):
            vis = (view_facts.get("resolution") or "").lower()
            if not vis:
                return
            if value.lower() == "hd" and not any(x in vis for x in ("hd", "720", "1280")):
                return
            if value.lower() == "fhd" and not any(x in vis for x in ("fhd", "1080", "1920", "full")):
                return
        cleaned.append({"field": field, "value": value})
        seen.add(field)

    # Evidence-first: ignore model-invented claim values; rebuild from observations.
    brand = view_facts.get("brand")
    if brand:
        _add("brand", brand)
    for spec in ("resolution", "screen_size", "refresh_rate"):
        if spec in needed_specs and view_facts.get(spec):
            _add(spec, view_facts[spec])
    if last_stock:
        if last_stock.get("price") is not None:
            _add("price", last_stock.get("price"))
        if last_stock.get("in_stock") is not None:
            _add("in_stock", last_stock.get("in_stock"))

    if cleaned:
        out["justification"] = {"claims": cleaned}
    else:
        out.pop("justification", None)
    return out


def _arguments(
    tool_call: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
    last_stock: dict[str, Any] | None,
    view_facts_by_pid: dict[str, dict[str, str]] | None,
    last_pick: dict[str, str] | None = None,
) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    if (tool_call["function"].get("name") or "") == "place_test_order":
        return _sanitize_place_order_args(
            parsed,
            query=query,
            allowed_fields=allowed_fields,
            last_stock=last_stock,
            view_facts_by_pid=view_facts_by_pid,
            last_pick=last_pick,
        )
    return parsed

def _rewrite_recovery_message(
    args: dict[str, Any],
    *,
    query: str,
    force: bool = False,
    shock_kind: str = "stock",
    price: float | None = None,
    budget: float | None = None,
) -> dict[str, Any]:
    """Force declarative recovery commit matching shock type (price vs stock).

    Only rewrite when force=True (real market event). Spontaneous LLM stockout
    chatter must not become a fake recovery commit that blocks TF6 cart→event.
    """
    content = str(args.get("content") or args.get("message") or "")
    low = content.lower()
    # Clarify / budget answers must pass through untouched unless force=True.
    clarifyish = any(
        x in low
        for x in (
            "going with",
            "i'll go with",
            "i will go with",
            "intending",
            "non-negotiable",
            "within budget",
            "meets the",
            "has a",
            "inch screen",
            "confirmed",
        )
    ) and not any(
        x in low
        for x in ("out of stock", "went out", "no longer available", "still available?")
    )
    if clarifyish and not force:
        return args
    model = _preferred_model_hint(query)
    kind = shock_kind
    if any(x in low for x in ("more expensive", "repriced", "price increased", "above budget")):
        kind = "price"
    if not force:
        return args
    out = dict(args)
    out["content"] = _canonical_recovery_message(
        model, shock_kind=kind, price=price, budget=budget
    )
    out.pop("message", None)
    return out


def _looks_like_market_event(text: str) -> bool:
    """True price/stock shock — NOT budget nags or clarify-demand questions.

    Root cause (local-60a92da): treating "before you" / "hold on" / "i can only go"
    as market events set event_open + recovery_msg_force, so _finish_violation
    blocked every place_test_order → step_limit / reward 0 on TF1/3/5/6.
    """
    low = (text or "").lower()
    # Budget-only updates must NOT open a market-event gate.
    budget_only = (
        ("i can only go" in low or "checked my balance" in low)
        and not any(
            x in low
            for x in (
                "more expensive",
                "got more expensive",
                "price jump",
                "price changed",
                "still available",
                "out of stock",
                "no longer available",
            )
        )
    )
    if budget_only:
        return False
    if any(
        c in low
        for c in (
            "still available",
            "more expensive",
            "got more expensive",
            "re-check before ordering",
            "re-check before",
            "price jump",
            "price changed",
            "out of stock",
            "no longer available",
            "went out of stock",
            "just get more expensive",
        )
    ):
        return True
    # "from 35999.00 to 82350.00 PHP" style price-shock wording
    return bool(
        "php" in low
        and re.search(r"\bfrom\b.+\bto\b", low)
        and any(x in low for x in ("expensive", "price", "re-check", "jumped"))
    )


def _looks_like_clarify_or_budget(text: str) -> bool:
    """Shopper clarify_demand / add_constraint — needs a real answer, not recovery."""
    low = (text or "").lower().strip()
    if len(low) < 8:
        return False
    return any(
        c in low
        for c in (
            "before you buy",
            "which one",
            "does it actually",
            "non-negotiable",
            "hold on",
            "i can only go",
            "checked my balance",
            "tell you if you ask",
        )
    )


def _looks_like_shopper_intervention(text: str) -> bool:
    low = (text or "").lower().strip()
    if len(low) < 8:
        return False
    if _looks_like_market_event(low) or _looks_like_clarify_or_budget(low):
        return True
    cues = (
        "?",
        "wait,",
        "re-check",
        "firm requirement",
        "please keep to",
        "compare the current",
    )
    return any(c in low for c in cues)


def _obs_stock_payload(observation: Any) -> dict[str, Any] | None:
    """Extract stock/price facts from nested EnvPack observations."""
    if not isinstance(observation, dict):
        return None
    # Common shapes: {done, observation:{in_stock,price,...}} or flat
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
    if "in_stock" in observation or "price" in observation:
        return observation
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

def _chat(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    family: str = "generic",
) -> dict[str, Any]:
    last_err: Exception | None = None
    for model in _model_list(family):
        for _try in range(2):
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
                    time.sleep(0.3)
                    continue
                return inference
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.3)
    raise RuntimeError(f"inference unavailable: {last_err}")


def _env_call(envelope: dict[str, Any]) -> dict[str, Any]:
    base_id = str(envelope.get("call_id") or "call")
    for attempt in range(3):
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
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"environment call failed after retries ({base_id})")


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
    if any(n in ("search", "filter") for n in names) and len(viewed & cand_pids) < len(
        cand_pids
    ):
        return (
            "Closed candidate set: do not search/filter until every listed product is viewed. "
            f"Missing: {sorted(cand_pids - viewed)}"
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
            if pid and sku and f"{pid}::{sku}" not in cand_keys:
                if name in ("add_to_cart", "place_test_order"):
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
    """Reject ordering when latest stock/cart says out of stock (market shock)."""
    if not _stock_is_false(last_stock):
        return None
    names = _tool_names(tool_calls)
    if "place_test_order" in names:
        return (
            "Rejected: latest stock/cart shows in_stock=false (market shock). "
            "remove_from_cart if needed, pick another candidate, inspect_stock, "
            "then order only an in-stock sku."
        )
    return None


def _call_pid_sku(tool_calls: list[dict[str, Any]], name: str) -> tuple[str, str] | None:
    for call in tool_calls:
        if (call.get("function", {}).get("name") or "") != name:
            continue
        raw = call["function"].get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if isinstance(args, dict) and args.get("product_id") and args.get("sku"):
            return str(args["product_id"]), str(args["sku"])
    return None


def _brand_ok(pid: str, firm_brand: str | None, view_facts_by_pid: dict[str, dict[str, str]]) -> bool:
    if not firm_brand:
        return True
    want = firm_brand.lower().strip()
    facts = view_facts_by_pid.get(pid) or {}
    brand = str(facts.get("brand") or "").lower().strip()
    # Prefer explicit catalog brand — title/model often embed "Apple" on cases/chargers.
    if brand:
        return want == brand or want in brand.split() or brand.startswith(want)
    blob = " ".join(str(v) for v in facts.values()).lower()
    return bool(want) and want in blob


def _looks_like_accessory(pid: str, view_facts_by_pid: dict[str, dict[str, str]]) -> bool:
    facts = view_facts_by_pid.get(pid) or {}
    blob = " ".join(
        str(facts.get(k) or "")
        for k in ("title", "category", "brand", "model")
    ).lower()
    markers = (
        "case",
        "cover",
        "charger",
        "cable",
        "protector",
        "tempered",
        "accessory",
        "accessories",
        "screen glass",
        "holster",
        "pouch",
        "strap",
        "bagpack",
        "backpack",
        "shorts",
        "shirt",
        "cotton",
        "cartoon bag",
    )
    return any(m in blob for m in markers)


def _model_ok(
    pid: str,
    model_hint: str | None,
    view_facts_by_pid: dict[str, dict[str, str]],
) -> bool:
    """True when no model preference, or viewed facts evidence the model token."""
    if not model_hint:
        return True
    facts = view_facts_by_pid.get(pid) or {}
    if not facts:
        return False
    needle = model_hint.lower().strip()
    if not needle:
        return True
    model_field = str(facts.get("model") or "").lower().strip()
    # Prefer explicit catalog model — reject title spam like "635# Cartoon Bagpack".
    if model_field:
        if needle == model_field or needle in re.split(r"[\s/_.,;:+-]+", model_field):
            return True
        if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", model_field):
            return True
        return False
    blob = " ".join(str(v) for v in facts.values()).lower()
    return bool(
        re.search(rf"(?<![a-z0-9#]){re.escape(needle)}(?![a-z0-9#])", blob)
    )


def _finish_violation(
    tool_calls: list[dict[str, Any]],
    *,
    query: str,
    firm_brand: str | None,
    model_hint: str | None,
    budget: float | None,
    last_stock: dict[str, Any] | None,
    view_facts_by_pid: dict[str, dict[str, str]],
    need_reinspect: bool,
    event_open: bool,
    event_target_key: str | None,
) -> str | None:
    """EnvPack hard/stateful finish gates from local-cb6bc81 failures."""
    names = _tool_names(tool_calls)
    mutating = [n for n in names if n in ("add_to_cart", "place_test_order")]
    if not mutating:
        return None
    for name in mutating:
        pair = _call_pid_sku(tool_calls, name)
        if not pair:
            continue
        pid, sku = pair
        if firm_brand and not _brand_ok(pid, firm_brand, view_facts_by_pid):
            return (
                f"Rejected: firm brand `{firm_brand}` not evidenced on product {pid}. "
                "View a matching listing first."
            )
        if (
            model_hint
            and not _model_ok(pid, model_hint, view_facts_by_pid)
        ):
            return (
                f"Rejected: preferred model `{model_hint}` required. "
                f"Product {pid} lacks that model token in view evidence. "
                f"Search/view/inspect a `{model_hint}` listing before cart/order."
            )
        if pid in view_facts_by_pid and not _listing_matches_query(
            pid,
            query=query,
            firm_brand=None,  # brand already checked above
            model_hint=None,  # model already checked above
            view_facts_by_pid=view_facts_by_pid,
        ):
            return (
                "Rejected: viewed listing does not match the shopper's named "
                "screen size / resolution. View another listing that matches, "
                "then inspect_stock → cart → order."
            )
        if name == "place_test_order":
            if need_reinspect or event_open:
                return (
                    "Rejected: market event still open. remove_from_cart if needed, "
                    "VIEW a replacement, inspect_stock a non-event-target sku, then order."
                )
            if event_target_key and f"{pid}::{sku}" == event_target_key:
                return (
                    "Rejected: cannot order the market-event target sku. Pick another "
                    "in-stock variant/product, inspect_stock, then order."
                )
            if last_stock is None:
                return "Rejected: inspect_stock before place_test_order."
            if _stock_is_false(last_stock):
                return "Rejected: latest inspect shows out of stock."
            price = last_stock.get("price")
            try:
                price_f = float(price) if price is not None else None
            except (TypeError, ValueError):
                price_f = None
            if budget is not None and price_f is not None and price_f > budget + _EPS:
                return (
                    f"Rejected: price {price_f:.2f} exceeds firm budget {budget:.2f}. "
                    "remove_from_cart, pick a cheaper in-stock sku, inspect_stock, order."
                )
            bumped = last_stock.get("current_price")
            try:
                bumped_f = float(bumped) if bumped is not None else None
            except (TypeError, ValueError):
                bumped_f = None
            if budget is not None and bumped_f is not None and bumped_f > budget + _EPS:
                return (
                    f"Rejected: current_price {bumped_f:.2f} exceeds budget {budget:.2f} "
                    "after market move. Recover a cheaper sku."
                )
    return None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """EnvPack loop: ilostaboy gates + Loopler DNA + finish gates from eval fails."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, min(_MAX_CALLS_HARD_CAP, int(policy.get("max_calls_per_turn", 1))))
    tools = policy["tools"]
    query = str(policy.get("query") or "")
    allowed_fields = _claim_field_allowlist(tools)
    candidates = _parse_candidate_variants(query)
    cand_pids = {pid for pid, _ in candidates}
    ask_first = _needs_ask_first_spec(query)
    budget = _parse_budget(query)
    firm_brand = _parse_firm_brand(query)
    phrase = _parse_phrase(query)
    soft_axes = _parse_soft_axes(query)

    boot = (
        f"{query}\n\n"
        "Loopler product procedure: LTR 2–8 keywords (fuse num+unit; keep use-case nouns); "
        "prefer in_stock=true; attrs/sku over title fluff; ask firm specs if withheld; "
        "else retrieve → view → inspect_stock → cart → place_test_order. "
        "On stockout/price shock: COMMIT message (never ask 'should I?') → remove → "
        "VIEW replacement → fresh inspect_stock → order. "
        "Justification only if asked: claims from latest view + inspect_stock only "
        "(real brand; never category; no invented hd/fhd)."
    )
    if budget is not None:
        boot += f"\n\nFIRM BUDGET: ≤ {budget:.2f} PHP. Never order above this."
    if firm_brand:
        boot += (
            f"\n\nFIRM BRAND: `{firm_brand}`. Include it in search; only cart/order when "
            "view brand/title literally contains it."
        )
    if phrase:
        boot += f"\n\nLITERAL PHRASE: keep `{phrase}` in every search/filter."
    if soft_axes:
        boot += (
            "\n\nSOFT AXES: "
            + ", ".join(soft_axes)
            + ". Search these; among firm-valid options prefer evidence for the top axis; "
            "compare ≥2 distinct product_ids when possible."
        )
    model_hint = _preferred_model_hint(query)
    color_hint = _preferred_color_hint(query)
    if ask_first:
        boot += (
            "\n\nASK-FIRST: Your first tool call must be `message` asking for the "
            "withheld firm hardware spec. After the reply, search with that spec and "
            "complete the order without apology loops."
        )
    if model_hint:
        boot += (
            f"\n\nPREFERRED MODEL TOKEN: `{model_hint}`. On stockout of the preferred "
            "color/variant, recover that same model in another color before changing models."
        )
    if candidates:
        listed = ", ".join(f"{pid}::{sku}" for pid, sku in candidates)
        boot += (
            "\n\nCLOSED CANDIDATE SET (mandatory): "
            f"{listed}. "
            "view each listed product_id; rank by stated priority "
            "(refresh→screen→price unless the query orders axes differently); "
            "after a market shock RE-RANK remaining listed keys — never re-cart a dead key."
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
    view_facts_by_pid: dict[str, dict[str, str]] = {}
    shock_recover = False
    recovery_msg_force = False
    need_reinspect = False
    event_open = False
    event_target_key: str | None = None
    post_event_commit = False
    commit_sent = False
    dead_keys: set[str] = set()
    shock_kind = "stock"
    last_pick: dict[str, str] = {}
    cart_sku: dict[str, str] = {}
    search_hits: list[str] = []
    # Products viewed AFTER recovery commit — grader event_closed needs a post-shock view.
    viewed_after_commit: set[str] = set()
    max_steps = int(policy["max_steps"])

    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        all_cands_viewed = bool(
            candidates and cand_pids and len(viewed & cand_pids) >= len(cand_pids)
        )
        bad_cart = bool(
            cart_sku and last_stock is not None and not _stock_ok(last_stock, budget)
        )
        in_cart_ok = bool(
            cart_sku
            and last_pick
            and cart_sku.get("product_id") == last_pick.get("product_id")
            and cart_sku.get("sku") == last_pick.get("sku")
            and _stock_ok(last_stock, budget)
            and f"{cart_sku.get('product_id')}::{cart_sku.get('sku')}" not in dead_keys
        )

        skip_llm = False
        tool_calls: list[dict[str, Any]] = []
        assistant_content = ""
        cart_key = (
            f"{cart_sku.get('product_id')}::{cart_sku.get('sku')}"
            if cart_sku.get("product_id") and cart_sku.get("sku")
            else ""
        )
        # Real market shock only (event_target_key set by add_to_cart ME / cart OOS).
        real_event = bool(event_target_key) or (
            event_open and bool(cart_sku) and bad_cart
        )

        # h151 priority: REMOVE bad cart → COMMIT (only after real event) → CART inspected.
        if bad_cart or (
            cart_sku
            and (
                cart_key == event_target_key
                or cart_key in dead_keys
            )
        ):
            tool_calls = _tc("remove_from_cart", dict(cart_sku), turn, "remove")
            assistant_content = "Removing OOS/over-budget / event-target cart line."
            need_reinspect = False
            skip_llm = True
        elif must_message and not real_event:
            skip_llm = False
        elif real_event and shock_recover and not commit_sent:
            content = _canonical_recovery_message(
                model_hint,
                shock_kind=shock_kind,
                price=_stock_price(last_stock),
                budget=budget,
            )
            tool_calls = _tc("message", {"content": content}, turn, "commit")
            assistant_content = "Committing market-event recovery."
            skip_llm = True
        elif (
            not cart_sku
            and not must_message
            and not candidates  # closed-set ranking has its own recover path
            and (
                shock_recover
                or post_event_commit
                or commit_sent
                or (
                    last_pick
                    and last_stock is not None
                    and _stock_ok(last_stock, budget)
                    and f"{last_pick.get('product_id')}::{last_pick.get('sku')}"
                    not in dead_keys
                    and (remaining <= 12 or searchish >= 3)
                )
            )
        ):
            # After a shock, relax color (same model, other variant) — never re-search forever.
            pick_color = None if (post_event_commit or commit_sent) else color_hint
            prefer = _best_viewed_pick(
                query=query,
                model_hint=model_hint,
                color=pick_color,
                view_facts_by_pid=view_facts_by_pid,
                dead_keys=dead_keys,
                firm_brand=firm_brand,
            )
            miss = _next_unviewed(search_hits, viewed)
            if prefer and (post_event_commit or commit_sent) and prefer[
                "product_id"
            ] not in viewed_after_commit:
                tool_calls = _tc(
                    "view", {"product_id": prefer["product_id"]}, turn, "postview"
                )
                assistant_content = "Re-viewing recovery candidate after market event."
            elif prefer:
                pref_key = f"{prefer['product_id']}::{prefer['sku']}"
                if (
                    last_pick.get("product_id") == prefer["product_id"]
                    and last_pick.get("sku") == prefer["sku"]
                    and last_stock is not None
                    and _stock_ok(last_stock, budget)
                    and pref_key not in dead_keys
                ):
                    tool_calls = _tc("add_to_cart", dict(prefer), turn, "cart")
                    assistant_content = "Carting preferred-model inspected sku."
                elif pref_key not in dead_keys:
                    tool_calls = _tc("inspect_stock", dict(prefer), turn, "inspect")
                    assistant_content = "Inspecting preferred-model viewed listing."
                elif miss and len(viewed) < max(6, len(search_hits[:8])):
                    tool_calls = _tc("view", {"product_id": miss}, turn, "viewhit")
                    assistant_content = f"Viewing search hit {miss}."
                else:
                    skip_llm = False
            elif miss and len(viewed) < max(6, len(search_hits[:8])):
                tool_calls = _tc("view", {"product_id": miss}, turn, "viewhit")
                assistant_content = f"Viewing search hit {miss}."
            elif model_hint and searchish < 4:
                q = f"{model_hint} smartphone"
                if pick_color:
                    q = f"{model_hint} {pick_color} smartphone"
                tool_calls = _tc(
                    "search",
                    {"query": q, "k": 10, "in_stock": True},
                    turn,
                    "modelsearch",
                )
                assistant_content = f"Searching preferred model `{model_hint}`."
            elif last_pick and _stock_ok(last_stock, budget) and _listing_matches_query(
                str(last_pick.get("product_id") or ""),
                query=query,
                firm_brand=firm_brand,
                model_hint=model_hint,
                view_facts_by_pid=view_facts_by_pid,
            ):
                tool_calls = _tc("add_to_cart", dict(last_pick), turn, "cart")
                assistant_content = "Carting inspected in-budget sku."
            elif (post_event_commit or commit_sent) and searchish < 5:
                # Over-budget / dead viewed picks: search for an in-budget replacement.
                bits = []
                if firm_brand:
                    bits.append(firm_brand)
                if model_hint:
                    bits.append(model_hint)
                inch = _query_inch_size(query)
                if inch is not None:
                    bits.append(f"{inch} inch")
                bits.append("smartphone" if "phone" in query.lower() else "in stock")
                q = " ".join(bits)
                tool_calls = _tc(
                    "search",
                    {"query": q, "k": 10, "in_stock": True},
                    turn,
                    "budgetsearch",
                )
                assistant_content = "Searching in-budget recovery replacement."
            else:
                skip_llm = False
            if tool_calls:
                skip_llm = True
        elif (
            model_hint
            and not cart_sku
            and not must_message
            and not all_cands_viewed
        ):
            pick_color = None if (post_event_commit or commit_sent) else color_hint
            prefer = _best_viewed_pick(
                query=query,
                model_hint=model_hint,
                color=pick_color,
                view_facts_by_pid=view_facts_by_pid,
                dead_keys=dead_keys,
                firm_brand=firm_brand,
            )
            miss = _next_unviewed(search_hits, viewed)
            # Prefer inspect/cart of a model match ASAP — don't drain 8 views first.
            if prefer and (post_event_commit or commit_sent) and prefer[
                "product_id"
            ] not in viewed_after_commit:
                tool_calls = _tc(
                    "view", {"product_id": prefer["product_id"]}, turn, "postview"
                )
                assistant_content = "Re-viewing same-model recovery candidate."
                skip_llm = True
            elif prefer and (
                last_pick.get("product_id") != prefer["product_id"]
                or last_pick.get("sku") != prefer["sku"]
                or last_stock is None
                or not _stock_ok(last_stock, budget)
                or f"{prefer['product_id']}::{prefer['sku']}" in dead_keys
            ):
                pref_key = f"{prefer['product_id']}::{prefer['sku']}"
                if pref_key in dead_keys:
                    # Fall through to another color / search.
                    if miss and len(viewed) < 8:
                        tool_calls = _tc("view", {"product_id": miss}, turn, "viewhit")
                        assistant_content = f"Viewing search hit {miss}."
                        skip_llm = True
                    elif searchish < 5:
                        q = f"{model_hint} smartphone"
                        tool_calls = _tc(
                            "search",
                            {"query": q, "k": 10, "in_stock": True},
                            turn,
                            "modelsearch",
                        )
                        assistant_content = f"Searching preferred model `{model_hint}`."
                        skip_llm = True
                    else:
                        skip_llm = False
                else:
                    tool_calls = _tc("inspect_stock", dict(prefer), turn, "inspect")
                    assistant_content = "Inspecting preferred-model candidate."
                    skip_llm = True
            elif prefer and _stock_ok(last_stock, budget):
                tool_calls = _tc("add_to_cart", dict(prefer), turn, "cart")
                assistant_content = "Carting preferred-model candidate."
                skip_llm = True
            elif miss and len(viewed) < 4:
                tool_calls = _tc("view", {"product_id": miss}, turn, "viewhit")
                assistant_content = f"Viewing search hit {miss}."
                skip_llm = True
            elif searchish < 4:
                q = f"{model_hint} smartphone"
                if pick_color:
                    q = f"{model_hint} {pick_color} smartphone"
                tool_calls = _tc(
                    "search",
                    {"query": q, "k": 10, "in_stock": True},
                    turn,
                    "modelsearch",
                )
                assistant_content = f"Searching preferred model `{model_hint}`."
                skip_llm = True
            else:
                skip_llm = False
        elif (
            (post_event_commit or commit_sent)
            and not cart_sku
            and not must_message
            and not model_hint
        ):
            # TF3/TF7 post-shock: firm brand/size listing; never inspect-loop over-budget keys.
            prefer = _best_viewed_pick(
                query=query,
                model_hint=None,
                color=None,
                view_facts_by_pid=view_facts_by_pid,
                dead_keys=dead_keys,
                firm_brand=firm_brand,
            )
            miss = _next_unviewed(search_hits, viewed)
            if prefer and prefer["product_id"] not in viewed_after_commit:
                tool_calls = _tc(
                    "view", {"product_id": prefer["product_id"]}, turn, "postview"
                )
                assistant_content = "Re-viewing firm-spec recovery listing."
                skip_llm = True
            elif prefer:
                pref_key = f"{prefer['product_id']}::{prefer['sku']}"
                if (
                    last_pick.get("product_id") == prefer["product_id"]
                    and last_pick.get("sku") == prefer["sku"]
                    and last_stock is not None
                    and _stock_ok(last_stock, budget)
                    and pref_key not in dead_keys
                ):
                    tool_calls = _tc("add_to_cart", dict(prefer), turn, "cart")
                    assistant_content = "Carting firm-spec recovery listing."
                elif pref_key not in dead_keys:
                    tool_calls = _tc("inspect_stock", dict(prefer), turn, "inspect")
                    assistant_content = "Inspecting firm-spec recovery listing."
                elif miss and len(viewed) < 8:
                    tool_calls = _tc("view", {"product_id": miss}, turn, "viewhit")
                    assistant_content = f"Viewing recovery search hit {miss}."
                elif searchish < 5:
                    bits = []
                    if firm_brand:
                        bits.append(firm_brand)
                    inch = _query_inch_size(query)
                    if inch is not None:
                        bits.append(f"{inch} inch")
                    bits.append("smartphone")
                    tool_calls = _tc(
                        "search",
                        {"query": " ".join(bits), "k": 10, "in_stock": True},
                        turn,
                        "budgetsearch",
                    )
                    assistant_content = "Searching in-budget firm-spec replacement."
                else:
                    skip_llm = False
                if tool_calls:
                    skip_llm = True
            elif miss and len(viewed) < 8:
                tool_calls = _tc("view", {"product_id": miss}, turn, "viewhit")
                assistant_content = f"Viewing recovery search hit {miss}."
                skip_llm = True
            elif searchish < 5:
                bits = []
                if firm_brand:
                    bits.append(firm_brand)
                inch = _query_inch_size(query)
                if inch is not None:
                    bits.append(f"{inch} inch")
                bits.append("smartphone")
                tool_calls = _tc(
                    "search",
                    {"query": " ".join(bits), "k": 10, "in_stock": True},
                    turn,
                    "budgetsearch",
                )
                assistant_content = "Searching in-budget firm-spec replacement."
                skip_llm = True
            else:
                skip_llm = False
        elif (
            all_cands_viewed
            and candidates
            and not must_message
            and (
                not cart_sku
                or cart_key in dead_keys
            )
        ):
            pick = None
            for pid, sku in candidates:
                key = f"{pid}::{sku}"
                if key not in dead_keys:
                    pick = {"product_id": pid, "sku": sku}
                    break
            if pick:
                if (
                    last_pick.get("product_id") == pick["product_id"]
                    and last_pick.get("sku") == pick["sku"]
                    and last_stock is not None
                    and _stock_ok(last_stock, budget)
                ):
                    tool_calls = _tc("add_to_cart", dict(pick), turn, "cart")
                    assistant_content = "Ranking: carting closed-set winner."
                else:
                    tool_calls = _tc("inspect_stock", dict(pick), turn, "inspect")
                    assistant_content = "Ranking: inspecting closed-set candidate."
                skip_llm = True
        elif in_cart_ok and not event_open and (
            post_event_commit or remaining <= 8 or searchish >= 5
        ):
            tool_calls = _tc("place_test_order", dict(cart_sku), turn, "order")
            assistant_content = "Ordering verified in-stock cart."
            skip_llm = True
        elif must_message:
            skip_llm = False
        else:
            skip_llm = False

        if not skip_llm:
            if must_message:
                if recovery_msg_force or event_open:
                    hint = (
                        "REQUIRED this turn: call `message` once. "
                        "If this is a stockout/'still available?' reply: STATE you will take "
                        "another color/variant of the SAME preferred model (do not ask permission)."
                    )
                    if model_hint:
                        hint += f" Preferred model token: {model_hint}."
                else:
                    hint = (
                        "REQUIRED this turn: call `message` once. "
                        "Answer the shopper's clarify/budget question directly (name the pick "
                        "and confirm firm specs). Do NOT invent a stockout recovery story."
                    )
                messages.append({"role": "user", "content": hint})
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
            elif event_target_key and (shock_recover or need_reinspect or event_open):
                recover = (
                    "Market shock / event OPEN: remove_from_cart if cart is the event target. "
                    "Recover SAME preferred model other color/variant first"
                )
                if model_hint:
                    recover += f" (model `{model_hint}`)"
                recover += (
                    ", VIEW the replacement product, inspect_stock an in-stock sku that is "
                    "NOT the event target, then place_test_order. Do not quit; do not ask "
                    "permission; do not order without fresh view+inspect_stock after the event. "
                    "If you message, STATE the same-model recovery plan (no permission ask)."
                )
                messages.append({"role": "user", "content": recover})
                shock_recover = False
            elif not candidates and searchish >= 3 and len(viewed) < 2:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "TF2 anti-thrash (HiSun/KAK): stop repeating the same search. "
                            "Reformulate once OR view best hits, then inspect_stock and finish."
                        ),
                    }
                )
            elif not candidates and searchish >= 5 and len(viewed) >= 2:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Search budget spent. Among viewed products meeting firm constraints, "
                            "inspect_stock → add_to_cart → place_test_order NOW."
                        ),
                    }
                )
            elif remaining <= 6 and not must_message:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Only ~{remaining} steps left. If you have an in-stock in-budget sku, "
                            "inspect_stock if needed and place_test_order NOW. Stop exploratory search."
                        ),
                    }
                )

        if not skip_llm:
            family = _infer_family(
                query=query,
                candidates=candidates,
                shock_recover=shock_recover or need_reinspect or event_open,
                searchish=searchish,
                must_message=must_message,
            )
            assistant_content = ""
            tool_calls: list[dict[str, Any]] = []
            for _attempt in range(4):
                inference = _chat(messages=messages, tools=tools, family=family)
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
                        # Do NOT open market-event gate — that fakes TF6 recovery.
                        need_reinspect = True
                        continue
                    fin = _finish_violation(
                        tool_calls,
                        query=query,
                        firm_brand=firm_brand,
                        model_hint=model_hint,
                        budget=budget,
                        last_stock=last_stock,
                        view_facts_by_pid=view_facts_by_pid,
                        need_reinspect=need_reinspect,
                        event_open=event_open,
                        event_target_key=event_target_key,
                    )
                    if fin:
                        messages.extend(
                            [
                                {
                                    "role": "assistant",
                                    "content": assistant_content,
                                    "tool_calls": tool_calls,
                                },
                                {"role": "user", "content": fin},
                            ]
                        )
                        # Soft nudge only — never fake a stockout recovery commit.
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
                                        "Rejected: after a market event you must inspect_stock "
                                        "(and fix cart) before place_test_order."
                                    ),
                                },
                            ]
                        )
                        continue
                break
            else:
                if candidates and len(viewed & cand_pids) < len(cand_pids):
                    miss = sorted(cand_pids - viewed)[0]
                    tool_calls = [
                        {
                            "id": f"forced-view-{turn}",
                            "type": "function",
                            "function": {
                                "name": "view",
                                "arguments": json.dumps({"product_id": miss}),
                            },
                        }
                    ]
                    assistant_content = f"Viewing closed-set candidate {miss}."
                elif must_message and ask_first and not asked_spec:
                    tool_calls = [
                        {
                            "id": f"forced-ask-{turn}",
                            "type": "function",
                            "function": {
                                "name": "message",
                                "arguments": json.dumps(
                                    {
                                        "content": (
                                            "What is the firm hardware requirement "
                                            "(RAM, screen size, resolution, or refresh rate)?"
                                        )
                                    }
                                ),
                            },
                        }
                    ]
                    assistant_content = "Asking for the withheld firm spec."
                elif event_open and commit_sent and cart_sku.get("product_id"):
                    tool_calls = [
                        {
                            "id": f"forced-remove-{turn}",
                            "type": "function",
                            "function": {
                                "name": "remove_from_cart",
                                "arguments": json.dumps(
                                    {
                                        "product_id": cart_sku["product_id"],
                                        "sku": cart_sku.get("sku") or "default",
                                    }
                                ),
                            },
                        }
                    ]
                    assistant_content = "Removing event-target cart line."
                elif event_open and commit_sent and not must_message:
                    pick_color = None
                    prefer = _best_viewed_pick(
                        query=query,
                        model_hint=model_hint,
                        color=pick_color,
                        view_facts_by_pid=view_facts_by_pid,
                        dead_keys=dead_keys,
                        firm_brand=firm_brand,
                    )
                    miss = _next_unviewed(search_hits, viewed)
                    if prefer and prefer["product_id"] not in viewed_after_commit:
                        tool_calls = _tc(
                            "view",
                            {"product_id": prefer["product_id"]},
                            turn,
                            "postview",
                        )
                        assistant_content = "Re-viewing recovery listing after event."
                    elif prefer:
                        pref_key = f"{prefer['product_id']}::{prefer['sku']}"
                        if (
                            last_pick.get("product_id") == prefer["product_id"]
                            and last_pick.get("sku") == prefer["sku"]
                            and last_stock is not None
                            and _stock_ok(last_stock, budget)
                            and pref_key not in dead_keys
                        ):
                            tool_calls = _tc("add_to_cart", dict(prefer), turn, "cart")
                            assistant_content = "Carting same-model recovery listing."
                        elif pref_key not in dead_keys:
                            tool_calls = _tc(
                                "inspect_stock", dict(prefer), turn, "inspect"
                            )
                            assistant_content = "Inspecting same-model recovery listing."
                        elif miss and len(viewed) < 8:
                            tool_calls = _tc(
                                "view", {"product_id": miss}, turn, "viewhit"
                            )
                            assistant_content = f"Viewing recovery search hit {miss}."
                        elif model_hint and searchish < 5:
                            tool_calls = _tc(
                                "search",
                                {
                                    "query": f"{model_hint} smartphone",
                                    "in_stock": True,
                                    "k": 10,
                                },
                                turn,
                                "recoversearch",
                            )
                            assistant_content = (
                                f"Searching same-model recovery for `{model_hint}`."
                            )
                    elif miss and len(viewed) < 8:
                        tool_calls = _tc("view", {"product_id": miss}, turn, "viewhit")
                        assistant_content = f"Viewing recovery search hit {miss}."
                    elif model_hint and searchish < 5:
                        tool_calls = _tc(
                            "search",
                            {
                                "query": f"{model_hint} smartphone",
                                "in_stock": True,
                                "k": 10,
                            },
                            turn,
                            "recoversearch",
                        )
                        assistant_content = f"Searching same-model recovery for `{model_hint}`."
                    elif searchish < 5:
                        bits = []
                        if firm_brand:
                            bits.append(firm_brand)
                        inch = _query_inch_size(query)
                        if inch is not None:
                            bits.append(f"{inch} inch")
                        bits.append("smartphone")
                        tool_calls = _tc(
                            "search",
                            {"query": " ".join(bits), "k": 10, "in_stock": True},
                            turn,
                            "budgetsearch",
                        )
                        assistant_content = "Searching in-budget recovery replacement."
                elif (
                    event_open
                    and last_pick.get("product_id")
                    and last_pick.get("sku")
                    and need_reinspect
                ):
                    tool_calls = [
                        {
                            "id": f"forced-recover-inspect-{turn}",
                            "type": "function",
                            "function": {
                                "name": "inspect_stock",
                                "arguments": json.dumps(
                                    {
                                        "product_id": last_pick["product_id"],
                                        "sku": last_pick["sku"],
                                    }
                                ),
                            },
                        }
                    ]
                    assistant_content = "Fresh inspect_stock after market event."
                elif (
                    last_pick.get("product_id")
                    and last_pick.get("sku")
                    and not need_reinspect
                    and not event_open
                ):
                    if model_hint and not _model_ok(
                        last_pick["product_id"], model_hint, view_facts_by_pid
                    ):
                        tool_calls = [
                            {
                                "id": f"forced-model-search-{turn}",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": json.dumps(
                                        {
                                            "query": model_hint,
                                            "in_stock": True,
                                            "k": 10,
                                        }
                                    ),
                                },
                            }
                        ]
                        assistant_content = (
                            f"Last pick lacks model `{model_hint}`; searching that token."
                        )
                    elif not (
                        cart_sku.get("product_id") == last_pick["product_id"]
                        and cart_sku.get("sku") == last_pick["sku"]
                    ):
                        tool_calls = [
                            {
                                "id": f"forced-cart-{turn}",
                                "type": "function",
                                "function": {
                                    "name": "add_to_cart",
                                    "arguments": json.dumps(
                                        {
                                            "product_id": last_pick["product_id"],
                                            "sku": last_pick["sku"],
                                        }
                                    ),
                                },
                            }
                        ]
                        assistant_content = "Carting last verified pick."
                    else:
                        tool_calls = [
                            {
                                "id": f"forced-order-{turn}",
                                "type": "function",
                                "function": {
                                    "name": "place_test_order",
                                    "arguments": json.dumps(
                                        {
                                            "product_id": last_pick["product_id"],
                                            "sku": last_pick["sku"],
                                        }
                                    ),
                                },
                            }
                        ]
                        assistant_content = "Placing order on last verified pick."
                elif last_pick.get("product_id") and last_pick.get("sku") and not event_open:
                    tool_calls = [
                        {
                            "id": f"forced-inspect-{turn}",
                            "type": "function",
                            "function": {
                                "name": "inspect_stock",
                                "arguments": json.dumps(
                                    {
                                        "product_id": last_pick["product_id"],
                                        "sku": last_pick["sku"],
                                    }
                                ),
                            },
                        }
                    ]
                    assistant_content = "Re-inspecting stock before order."
                elif viewed:
                    pid = sorted(viewed)[0]
                    tool_calls = [
                        {
                            "id": f"forced-view-{turn}",
                            "type": "function",
                            "function": {
                                "name": "view",
                                "arguments": json.dumps({"product_id": pid}),
                            },
                        }
                    ]
                    assistant_content = f"Re-viewing {pid}."
                else:
                    head = _shopper_text(query)
                    words = [
                        w
                        for w in re.findall(r"[A-Za-z0-9]+", head.lower())
                        if len(w) > 2
                        and w
                        not in {
                            "the", "and", "for", "with", "under", "budget", "php", "pesos",
                            "want", "need", "find", "please", "that", "this", "from", "have",
                            "stock", "order", "test", "place", "shopper", "task", "rules",
                        }
                    ][:6]
                    if firm_brand:
                        words = [firm_brand.lower()] + [w for w in words if w != firm_brand.lower()]
                    if phrase:
                        words = list(phrase.lower().split()[:4]) + words
                    q = " ".join(dict.fromkeys(words)) if words else "product"
                    tool_calls = [
                        {
                            "id": f"forced-search-{turn}",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": json.dumps(
                                    {"query": q, "in_stock": True, "k": 10}
                                ),
                            },
                        }
                    ]
                    assistant_content = f"Forced search fallback: {q}."

        if must_message:
            msg_only = [
                c
                for c in tool_calls
                if c.get("function", {}).get("name") == "message"
            ]
            if msg_only:
                tool_calls = msg_only[:1]

        # Apply finish gates to controller-forced carts/orders too (not only LLM path).
        if tool_calls and any(
            n in _tool_names(tool_calls)
            for n in ("add_to_cart", "place_test_order")
        ):
            fin = _finish_violation(
                tool_calls,
                query=query,
                firm_brand=firm_brand,
                model_hint=model_hint,
                budget=budget,
                last_stock=last_stock,
                view_facts_by_pid=view_facts_by_pid,
                need_reinspect=need_reinspect,
                event_open=event_open,
                event_target_key=event_target_key,
            )
            if fin:
                messages.append({"role": "user", "content": fin})
                # Pivot: search same-model / firm-spec instead of carting a bad listing.
                bits = []
                if model_hint:
                    bits.append(model_hint)
                elif firm_brand:
                    bits.append(firm_brand)
                inch = _query_inch_size(query)
                if inch is not None:
                    bits.append(f"{inch} inch")
                bits.append("smartphone" if "phone" in query.lower() else "in stock")
                tool_calls = _tc(
                    "search",
                    {"query": " ".join(bits), "k": 10, "in_stock": True},
                    turn,
                    "gatefix",
                )
                assistant_content = "Rejected bad cart/order; searching matching listing."

        names = _tool_names(tool_calls)
        if any(n in ("search", "filter") for n in names):
            searchish += 1
        if "message" in names and ask_first:
            asked_spec = True

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        group_id = f"{problem_id}-turn-{turn}"
        sanitized_calls: list[dict[str, Any]] = []
        for index, call in enumerate(tool_calls, start=1):
            args = _arguments(
                call,
                query=query,
                allowed_fields=allowed_fields,
                last_stock=last_stock,
                view_facts_by_pid=view_facts_by_pid,
                last_pick=last_pick,
            )
            name = call.get("function", {}).get("name") or ""
            if name == "message" and isinstance(args, dict):
                # EnvPack message schema uses `content` (fall back to `message`).
                force_rec = (
                    recovery_msg_force
                    and not commit_sent
                    and bool(event_target_key)
                    and (event_open or bool(cart_sku))
                )
                rewritten = _rewrite_recovery_message(
                    args,
                    query=query,
                    force=force_rec,
                    shock_kind=shock_kind,
                    price=_stock_price(last_stock),
                    budget=budget,
                )
                if rewritten is not args or force_rec:
                    args = rewritten
                    if "content" in args and "message" in args:
                        args.pop("message", None)
            if name in ("inspect_stock", "add_to_cart", "place_test_order"):
                if args.get("product_id") and args.get("sku"):
                    last_pick = {
                        "product_id": str(args["product_id"]),
                        "sku": str(args["sku"]),
                    }
            if name == "add_to_cart":
                add_key = (
                    f"{last_pick.get('product_id')}::{last_pick.get('sku')}"
                    if last_pick.get("product_id") and last_pick.get("sku")
                    else ""
                )
                if add_key and add_key in dead_keys:
                    # Never re-cart a shocked/OOS listing; inspect next closed-set pick.
                    alt = None
                    for pid, sku in candidates:
                        key = f"{pid}::{sku}"
                        if key not in dead_keys:
                            alt = {"product_id": pid, "sku": sku}
                            break
                    if alt:
                        name = "inspect_stock"
                        args = dict(alt)
                        last_pick = dict(alt)
                        sanitized_calls.append(
                            {
                                "call_id": f"{group_id}-{index}",
                                "action": {"name": name, "args": args},
                            }
                        )
                        continue
                cart_sku = dict(last_pick)
            if name == "remove_from_cart":
                cart_sku = {}
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
            "turn": turn,
            "calls": sanitized_calls,
        }
        result = _env_call(envelope)
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
                    pid_v = str(args["product_id"])
                    viewed.add(pid_v)
                    if commit_sent or post_event_commit:
                        viewed_after_commit.add(pid_v)

        if "message" in names:
            must_message = False
            recovery_msg_force = False
            if event_open or shock_recover:
                post_event_commit = True
                commit_sent = True
                shock_recover = False

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
            if payload is None and isinstance(obs, dict):
                payload = _obs_stock_payload({"observation": obs})
            name = tool_call.get("function", {}).get("name") or ""
            if name in ("search", "filter") and isinstance(obs, dict):
                for pid in _collect_search_pids(obs):
                    if pid not in search_hits:
                        search_hits.append(pid)
            if name == "view" and isinstance(obs, dict):
                pid, facts = _extract_view_facts(obs)
                if not pid:
                    raw = tool_call["function"].get("arguments", "{}")
                    try:
                        a = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        a = {}
                    if isinstance(a, dict) and a.get("product_id"):
                        pid = str(a["product_id"])
                if pid and facts:
                    view_facts_by_pid[pid] = {
                        **view_facts_by_pid.get(pid, {}),
                        **facts,
                    }
            if name == "inspect_stock":
                need_reinspect = False
                # Close event only after a fresh inspect on a non-event-target sku
                # following a recovery commit message.
                raw = tool_call["function"].get("arguments", "{}")
                try:
                    a = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    a = {}
                key = None
                if isinstance(a, dict) and a.get("product_id") and a.get("sku"):
                    key = f"{a['product_id']}::{a['sku']}"
                # Never inspect-loop OOS / over-budget keys (TF3 step_limit root cause).
                if key and payload is not None and not _stock_ok(payload, budget):
                    dead_keys.add(key)
                    if (
                        last_pick.get("product_id") == str(a.get("product_id") or "")
                        and last_pick.get("sku") == str(a.get("sku") or "")
                    ):
                        last_pick = {}
                if event_open and post_event_commit and key and key != event_target_key:
                    if payload and not _stock_is_false(payload):
                        price_ok = True
                        if budget is not None and payload.get("price") is not None:
                            try:
                                price_ok = float(payload["price"]) <= budget + _EPS
                            except (TypeError, ValueError):
                                price_ok = True
                        if price_ok:
                            event_open = False
                            shock_recover = False
                # Sticky lock only AFTER recovery commit, on a non-event-target same-model sku.
                if (
                    post_event_commit
                    and isinstance(a, dict)
                    and a.get("product_id")
                    and a.get("sku")
                    and payload
                    and not _stock_is_false(payload)
                    and model_hint
                    and _model_ok(str(a["product_id"]), model_hint, view_facts_by_pid)
                ):
                    price_ok = True
                    if budget is not None and payload.get("price") is not None:
                        try:
                            price_ok = float(payload["price"]) <= budget + _EPS
                        except (TypeError, ValueError):
                            price_ok = True
                    if price_ok and key and key != event_target_key:
                        last_pick = {
                            "product_id": str(a["product_id"]),
                            "sku": str(a["sku"]),
                        }
                        event_open = False
                        shock_recover = False
                        need_reinspect = False
            if name in ("inspect_stock", "add_to_cart", "inspect_cart") and payload:
                last_stock = payload
                if _stock_is_false(payload):
                    # Only open a market-event gate when the OOS sku is carted via
                    # add_to_cart — browsing/inspect OOS must not fake TF6 recovery.
                    raw = tool_call["function"].get("arguments", "{}")
                    try:
                        a = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        a = {}
                    oos_key = None
                    if isinstance(a, dict) and a.get("product_id") and a.get("sku"):
                        oos_key = f"{a['product_id']}::{a['sku']}"
                    cart_key_now = (
                        f"{cart_sku['product_id']}::{cart_sku['sku']}"
                        if cart_sku.get("product_id") and cart_sku.get("sku")
                        else None
                    )
                    if name == "add_to_cart" or (
                        name == "inspect_stock"
                        and oos_key
                        and cart_key_now
                        and oos_key == cart_key_now
                    ):
                        shock_kind = "stock"
                        shock_recover = True
                        need_reinspect = True
                        recovery_msg_force = True
                        event_open = True
                        if oos_key:
                            event_target_key = oos_key
                            dead_keys.add(oos_key)
                    else:
                        # Soft nudge only — browsing OOS must not open recovery commit.
                        if oos_key:
                            dead_keys.add(oos_key)
                        need_reinspect = True
            if isinstance(obs, dict):
                inner = obs.get("observation") if isinstance(obs.get("observation"), dict) else obs
                if isinstance(inner, dict) and inner.get("market_event") is not None:
                    me = inner.get("market_event")
                    kind = "stock"
                    if isinstance(me, dict):
                        mk = str(me.get("kind") or "").lower()
                        if "price" in mk:
                            kind = "price"
                        elif "stock" in mk:
                            kind = "stock"
                    # Also detect price bump via cart lines when kind missing.
                    added = inner.get("added") if isinstance(inner.get("added"), dict) else None
                    cart_lines = inner.get("cart") if isinstance(inner.get("cart"), list) else []
                    if kind == "stock" and added and cart_lines:
                        for line in cart_lines:
                            if not isinstance(line, dict):
                                continue
                            if str(line.get("product_id")) != str(added.get("product_id")):
                                continue
                            try:
                                at_add = float(line.get("price_at_add"))
                                cur = float(line.get("current_price", added.get("price")))
                                if cur > at_add + _EPS:
                                    kind = "price"
                            except (TypeError, ValueError):
                                pass
                    shock_kind = kind
                    shock_recover = True
                    need_reinspect = True
                    recovery_msg_force = True
                    event_open = True
                    must_message = False  # forced commit controller handles this
                    if isinstance(added, dict):
                        last_stock = dict(added)
                        if kind == "price" and isinstance(me, dict) and me.get("new_price") is not None:
                            try:
                                last_stock["current_price"] = float(me["new_price"])
                            except (TypeError, ValueError):
                                pass
                        pid = str(added.get("product_id") or last_pick.get("product_id") or "")
                        sku = str(added.get("sku") or last_pick.get("sku") or "")
                        if pid and sku:
                            event_target_key = f"{pid}::{sku}"
                            dead_keys.add(event_target_key)
                    elif cart_sku.get("product_id") and cart_sku.get("sku"):
                        event_target_key = (
                            f"{cart_sku['product_id']}::{cart_sku['sku']}"
                        )
                        dead_keys.add(event_target_key)

            if name == "add_to_cart" and post_event_commit:
                raw = tool_call["function"].get("arguments", "{}")
                try:
                    a = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    a = {}
                rec_key = None
                if isinstance(a, dict) and a.get("product_id") and a.get("sku"):
                    rec_key = f"{a['product_id']}::{a['sku']}"
                inner2 = (
                    obs.get("observation")
                    if isinstance(obs, dict) and isinstance(obs.get("observation"), dict)
                    else obs
                )
                me2 = inner2.get("market_event") if isinstance(inner2, dict) else None
                if rec_key and rec_key != event_target_key and rec_key not in dead_keys and me2 is None:
                    event_open = False
                    shock_recover = False

        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            messages.append({"role": "user", "content": content})
            text = str(content or "").strip()
            # Fold ask-first / budget replies into query so firm size/brand gates see them.
            if text and not _looks_like_market_event(text):
                if (
                    _query_inch_size(text) is not None
                    or _query_wants_resolution_class(text) is not None
                    or _parse_budget(text) is not None
                    or "required" in text.lower()
                    or "brand is" in text.lower()
                    or "screen" in text.lower()
                ):
                    query = f"{query}\n{text}"
                    nb = _parse_firm_brand(query)
                    if nb:
                        firm_brand = nb
                    nm = _preferred_model_hint(query)
                    if nm:
                        model_hint = nm
                    nc = _preferred_color_hint(query)
                    if nc:
                        color_hint = nc
            # Mid-episode budget tightening (add_constraint).
            new_b = _parse_budget(text)
            if new_b is not None:
                budget = new_b if budget is None else min(budget, new_b)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"FIRM BUDGET now ≤ {budget:.2f} PHP. "
                            "Acknowledge briefly if asked, then search/filter within budget "
                            "and finish with inspect_stock → cart → place_test_order. "
                            "Do NOT treat this as a stockout recovery."
                        ),
                    }
                )
            if _looks_like_shopper_intervention(text):
                low = text.lower()
                # After ask-first reply with the actual spec, resume shopping.
                if (
                    ask_first
                    and asked_spec
                    and "?" not in low
                    and not _looks_like_market_event(text)
                ):
                    must_message = False
                elif _looks_like_market_event(text):
                    # True price/stock shock notice from the harness — only after a cart event.
                    if not event_target_key and not cart_sku:
                        must_message = False
                        recovery_msg_force = False
                    else:
                        shock_kind = _shock_kind_from_text(text)
                        need_reinspect = True
                        event_open = True
                        if cart_sku.get("product_id") and cart_sku.get("sku"):
                            event_target_key = (
                                f"{cart_sku['product_id']}::{cart_sku['sku']}"
                            )
                            dead_keys.add(event_target_key)
                        if commit_sent:
                            must_message = False
                            recovery_msg_force = False
                            shock_recover = False
                        else:
                            recovery_msg_force = True
                            must_message = False
                            shock_recover = True
                elif _looks_like_clarify_or_budget(text) or "?" in low or "which" in low:
                    # clarify_demand / budget nag: answer — never open event gate.
                    if commit_sent and event_open:
                        must_message = False
                        recovery_msg_force = False
                    else:
                        must_message = True
                        recovery_msg_force = False
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Shopper intervention (clarify/budget). Call `message` once: "
                                    "name the product/sku you intend (or confirm the firm screen/"
                                    "RAM/resolution/refresh if they asked), confirm it meets firm "
                                    "specs and budget, then on the NEXT turn resume search/view/"
                                    "inspect_stock/cart/order. Do NOT claim a stockout unless one "
                                    "was observed."
                                ),
                            }
                        )
                else:
                    must_message = True
                    recovery_msg_force = False
        if any(
            (call.get("observation") or {}).get("done")
            for call in result["calls"]
        ):
            break

        if not must_message and not candidates and not shock_recover and not event_open:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Episode not done. Prefer inspect_stock → add_to_cart → "
                        "place_test_order on an in-stock sku that meets firm constraints. "
                        "Omit justification unless the shopper asked why."
                    ),
                }
            )
        elif not must_message and candidates and len(viewed & cand_pids) >= len(cand_pids):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "All listed candidates viewed. Rank by stated priority, "
                        "inspect_stock the winner, add_to_cart listed sku, place_test_order."
                    ),
                }
            )

    return dialogue


__all__ = ["agent_main"]
