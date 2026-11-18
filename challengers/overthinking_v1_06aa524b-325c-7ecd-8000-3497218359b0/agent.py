"""ORO Bench EnvPack — essentialsHybrid hiddenRace v11.

Fix for next-race hidden problems after v10 FAIL (0.74 vs v7 0.873).
Keep v7 recovery muscle (shock rewrite + market_event backup + price commit).
Replace pack catchphrase routers with Task-rules / structure essentials.
Neutral procedure prompt: no miner-agent names, no TF headers.
No problem-id / gold-SKU hardcodes.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_OPENROUTER_MODELS = [
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen3.6-27b",
]
_CHUTES_MODELS = [
    "moonshotai/Kimi-K2.6-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
    "Qwen/Qwen3.6-27B-TEE",
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
# Prefer omitting category claims (schema may list field; faithfulness often fails).
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
    r"(?:ask|asked|asking).{0,48}(?:firm|requirement|spec|hardware)|"
    r"(?:firm|requirement|spec).{0,48}(?:ask|asked|asking)|"
    r"withheld|tell you if you ask|if you ask",
    re.I,
)
# Task-rules: include structured justification on order (required wins over optional boilerplate).
_JUSTIF_REQUIRED_RE = re.compile(
    r"include a structured justification|structured justification with claims|"
    r"(?:must|required).{0,40}justification|justification.{0,40}(?:must|required)",
    re.I,
)

_proxy = ProxyClient(timeout=120, max_retries=3)

_SYSTEM = """You are a shopping agent for a tool-using environment.

Use ONLY the supplied tools. Continue until done=true.
Obey this episode's shopper text and Task rules exactly.
Never memorize product IDs. Do not invent catalog facts.

Intent:
Decompose into firm constraints vs soft preferences (≤4 sub-goals). Firm always wins.
If the shopper asks a question, changes budget, or asks which item: NEXT action is `message`, then resume.
Ask-first: if a firm spec is withheld until you ask → FIRST call `message` for that spec.

Retrieval:
search/filter with reformulations. Survey a few distinct product_ids.
After ~3–4 weak searches, view best hits and proceed (do not thrash).
Prefer in_stock=true when availability matters.

Constraints:
Firm specs are NON-NEGOTIABLE (category_path, RAM/screen/resolution/refresh, budget, stock).
Match brands/models/sizes/numbers LITERALLY after `view`. Soft brand only after all firm pass.

Preference:
Among fully valid options, prefer cheaper / better fit to soft prefs. Soft ≠ override firm.

Closed candidate ranking:
When the shopper lists multiple product_id::sku keys, that set is exhaustive.
view every listed product_id; rank by the stated priority; cart+order MUST be a listed key.
Never open-catalog search for substitutes on a closed set.

Recovery:
Stock/price shock or market_event: message with a definite plan → remove_from_cart → other candidate
(same model/color ladder when relevant) → inspect_stock → add_to_cart → place_test_order.
NEVER ask permission. NEVER order on out-of-stock / pre-event assumptions. Never quit.

Justification:
When Task rules require structured justification on the order (or the shopper asks for a reason),
claims MUST match latest view/inspect_stock facts plus current price/stock.
Omit unsupported fields; do not invent. Prefer brand + price + in_stock plus named specs when supported.

Finish:
search/filter → view/compare → inspect_stock → add_to_cart → place_test_order.
Late in the budget: stop exploring and order an in-stock in-budget pick.
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


def _task_rules(query: str) -> str:
    parts = query.split("Task rules:", 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _parse_candidate_variants(query: str) -> list[tuple[str, str]]:
    """Closed set from structure: ≥2 product_id::sku pairs (no pack catchphrases)."""
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


def _justification_required(query: str) -> bool:
    """Task-rules obligation, else shopper asks for a reason (contract modality)."""
    rules = _task_rules(query)
    if rules and _JUSTIF_REQUIRED_RE.search(rules):
        return True
    head = _shopper_text(query).lower()
    return any(w in head for w in ("why", "reason", "justif"))


def _needs_ask_first_spec(query: str) -> bool:
    return bool(_ASK_SPEC_RE.search(_shopper_text(query))) or bool(
        _ASK_SPEC_RE.search(_task_rules(query))
    )


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
    """Map shopper resolution language to claim-enum class tokens when present."""
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
    """Prefer exact view strings; allow class tokens when shopper named them."""
    text = str(text or "").strip()
    if not text:
        return text
    low = text.lower().replace("×", "x")
    if low in ("hd", "fhd", "qhd", "uhd", "4k"):
        return "uhd" if low == "4k" else low
    wanted = _query_wants_resolution_class(query)
    if wanted == "hd" and any(x in low for x in ("720", "1280", " hd", "hd ")):
        if "1080" not in low and "1920" not in low and "1440" not in low:
            return "hd"
    if wanted == "fhd" and any(x in low for x in ("1080", "1920", "full hd", "fhd")):
        return "fhd"
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
                if lk in ("brand", "resolution", "screen_size", "refresh_rate", "title", "model", "display") and v not in (
                    None,
                    "",
                    "None",
                ):
                    facts.setdefault(lk, str(v).strip())
                if lk in ("attributes", "specs", "options", "variants") or isinstance(
                    v, (dict, list)
                ):
                    walk(v, path + "." + lk)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:8]):
                walk(v, f"{path}[{i}]")

    walk(node)
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
            if "screen" in name and "size" in name:
                facts.setdefault("screen_size", str(val).strip())
            if "refresh" in name:
                facts.setdefault("refresh_rate", str(val).strip())
    if node.get("brand") not in (None, "", "None"):
        facts["brand"] = str(node["brand"]).strip()
    title = node.get("title") or node.get("name")
    if title not in (None, "", "None"):
        facts["title"] = str(title).strip()
    return pid, facts


def _preferred_model_hint(query: str) -> str | None:
    """Extract preferred model token; stop at `.` / `If` (v1 TF6 garble fix)."""
    head = _shopper_text(query)
    colors = (
        r"black|white|green|red|blue|gold|silver|pink|purple|gray|grey|"
        r"yellow|orange|moonlight\s+white"
    )
    patterns = (
        rf"prefer(?:\s+the)?\s+(?:{colors})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"eye on the\s+(?:{colors})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"you have your eye on the\s+(?:{colors})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
    )
    token = None
    for pat in patterns:
        m = re.search(pat, head, flags=re.I)
        if m:
            token = m.group(1).strip()
            break
    if not token:
        cleaned = re.sub(r"\d+\.\d+", " ", head)
        closed_pids = {pid for pid, _sku in _CAND_KEY_RE.findall(head)}
        cleaned = _CAND_KEY_RE.sub(" ", cleaned)
        for tok in re.findall(r"\b\d{6,}\b", cleaned):
            if tok not in closed_pids:
                return tok
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
    return token or None


def _is_model_code(token: str | None) -> bool:
    return bool(token) and any(ch.isdigit() for ch in token)


def _short_title(title: str) -> str:
    text = " ".join(str(title or "").split())
    if len(text) <= 72:
        return text or "that item"
    return text[:69].rstrip() + "..."


def _pixel_class(text: str) -> str | None:
    low = (text or "").lower().replace("×", "x").replace("*", "x")
    m = re.search(r"(\d{3,5})\s*x\s*(\d{3,5})", low)
    if m:
        long_side, short_side = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        if long_side >= 3840 and short_side >= 2160:
            return "uhd"
        if long_side >= 2560 and short_side >= 1440:
            return "qhd"
        if long_side >= 1920 and short_side >= 1080:
            return "fhd"
        if long_side >= 1280 and short_side >= 720:
            return "hd"
        return None
    if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
        return "uhd"
    if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
        return "qhd"
    if re.search(r"\b(?:fhd|1080p?|full hd)\b", low):
        return "fhd"
    if re.search(r"\b(?:720p|hd)\b", low):
        return "hd"
    return None


def _view_blob(facts: dict[str, str] | None) -> str:
    facts = facts or {}
    return " ".join(
        str(facts.get(k) or "")
        for k in ("title", "brand", "resolution", "display", "screen_size", "model")
    )


def _resolution_supported(text: str, wanted: str | None) -> bool:
    if not wanted:
        return True
    return _pixel_class(text) == wanted


def _price_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def _canonical_recovery_message(
    model: str | None,
    shock: dict[str, Any] | None = None,
) -> str:
    """Price commits bind observed change. Stockout keeps v5 title commit."""
    shock = shock or {}
    kind = str(shock.get("kind") or "stockout")
    title = _short_title(str(shock.get("title") or ""))
    if kind == "price_change":
        old = _price_text(shock.get("old_price"))
        new = _price_text(shock.get("new_price"))
        if old and new:
            lead = f"The price rose from {old} to {new}."
        else:
            lead = "The price rose above the budget."
        return (
            f"{lead} I'm now selecting another in-stock option that still meets "
            "the stated specification and stays within budget."
        )
    return (
        f"{title} went out of stock. I will take another color of that same product "
        "if one is still in stock, keep budget and category firm, and only then "
        "change product."
    )


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
            return f"{m.group(1)} inches"
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
) -> dict[str, Any]:
    out = dict(args)
    if not _justification_required(query):
        out.pop("justification", None)
        return out
    just = out.get("justification")
    if not isinstance(just, dict):
        just = {}
    claims_in = just.get("claims") if isinstance(just.get("claims"), list) else []
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    needed_specs = set(_spec_fields_from_query(query))
    pid = str(out.get("product_id") or "")
    view_facts: dict[str, str] = {}
    if view_facts_by_pid and pid and pid in view_facts_by_pid:
        view_facts = dict(view_facts_by_pid[pid])

    for claim in claims_in:
        if not isinstance(claim, dict):
            continue
        field = _normalize_claim_field(str(claim.get("field", "")), allowed_fields)
        if field is None or field in seen:
            continue
        # Drop hardware specs the shopper did not ask for.
        if field in ("resolution", "screen_size", "refresh_rate") and field not in needed_specs:
            continue
        raw_val = claim.get("value", "")
        if field == "brand":
            if view_facts.get("brand"):
                raw_val = view_facts["brand"]
            else:
                continue
        elif field in view_facts and field in ("resolution", "screen_size", "refresh_rate"):
            raw_val = view_facts[field]
        value = _normalize_claim_value(field, raw_val, query=query)
        if value == "":
            continue
        if field == "resolution":
            wanted = _query_wants_resolution_class(query)
            blob = _view_blob(view_facts)
            if wanted:
                if not _resolution_supported(blob, wanted):
                    continue
                value = wanted
            elif not view_facts.get("resolution") and not view_facts.get("display"):
                continue
        cleaned.append({"field": field, "value": value})
        seen.add(field)

    if last_stock:
        if "price" not in seen and last_stock.get("price") is not None:
            cleaned.append(
                {
                    "field": "price",
                    "value": _normalize_claim_value(
                        "price", last_stock.get("price"), query=query
                    ),
                }
            )
            seen.add("price")
        if "in_stock" not in seen and last_stock.get("in_stock") is not None:
            cleaned.append(
                {
                    "field": "in_stock",
                    "value": _normalize_claim_value(
                        "in_stock", last_stock.get("in_stock"), query=query
                    ),
                }
            )
            seen.add("in_stock")

    if "brand" not in seen and view_facts.get("brand"):
        brand = view_facts["brand"]
        if brand and brand.lower() not in ("none", "null", "n/a"):
            cleaned.append({"field": "brand", "value": brand})
            seen.add("brand")

    for spec in needed_specs:
        if (
            spec in allowed_fields
            and spec not in seen
            and spec not in _CLAIM_DENY
        ):
            if view_facts.get(spec):
                cleaned.append(
                    {
                        "field": spec,
                        "value": _normalize_claim_value(
                            spec, view_facts[spec], query=query
                        ),
                    }
                )
                seen.add(spec)
            elif spec == "resolution":
                wanted = _query_wants_resolution_class(query)
                if wanted and _resolution_supported(_view_blob(view_facts), wanted):
                    cleaned.append({"field": "resolution", "value": wanted})
                    seen.add("resolution")

    cleaned = [c for c in cleaned if c["field"] not in _CLAIM_DENY]
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
            last_stock=last_stock,
            view_facts_by_pid=view_facts_by_pid,
        )
    return parsed


def _rewrite_recovery_message(
    args: dict[str, Any],
    *,
    query: str,
    force: bool = False,
    shock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rewrite only a shock reply or a permission ask. Leave other messages alone."""
    content = str(args.get("content") or args.get("message") or "")
    low = content.lower()
    stockish = any(
        x in low
        for x in (
            "out of stock",
            "went out",
            "still available",
            "no longer",
            "not available",
            "unavailable",
            "oos",
            "price increased",
            "more expensive",
            "exceeds your budget",
            "above your",
            "over your",
        )
    )
    asks = any(
        x in low
        for x in (
            "should i",
            "do you want",
            "would you like",
            "any preference",
            "preferences?",
            "your preference",
            "or look for a different",
            "or a different",
            "let me know",
            "what would you like",
        )
    )
    model = _preferred_model_hint(query)
    model_l = (model or "").lower()
    commits = bool(model_l) and model_l in low and (
        "same" in low
        or "another color" in low
        or "other color" in low
        or "same model" in low
    )
    asking_spec = "?" in content and any(
        w in low for w in ("requirement", "spec", "which", "what ")
    )
    if force and not asking_spec:
        out = dict(args)
        out["content"] = _canonical_recovery_message(model, shock)
        out.pop("message", None)
        return out
    need = asks or stockish
    if not need:
        return args
    if commits and not asks:
        # Still rewrite if model token is garbled / polluted
        if model and f"({model})" in content and ". If" not in content:
            return args
    out = dict(args)
    out["content"] = _canonical_recovery_message(model, shock)
    out.pop("message", None)
    return out


def _looks_like_market_event(text: str) -> bool:
    low = (text or "").lower()
    return any(
        c in low
        for c in (
            "still available",
            "more expensive",
            "got more expensive",
            "i can only go",
            "re-check",
            "price jump",
            "price changed",
        )
    )


def _looks_like_shopper_intervention(text: str) -> bool:
    low = (text or "").lower().strip()
    if len(low) < 8:
        return False
    cues = (
        "?",
        "which one",
        "before you buy",
        "hold on",
        "wait,",
        "still available",
        "i can only go",
        "non-negotiable",
        "re-check",
        "tell you if you ask",
        "firm requirement",
        "please keep to",
        "compare the current",
        "screen",
        "ram",
        "budget",
        "inch",
    )
    return any(c in low for c in cues)


def _first_cart_model_violation(
    tool_calls: list[dict[str, Any]],
    *,
    query: str,
    view_facts_by_pid: dict[str, dict[str, str]],
    candidates: list[tuple[str, str]],
) -> str | None:
    """Named model code: first cart must be a product whose view contains it."""
    if candidates:
        return None
    token = _preferred_model_hint(query)
    if not _is_model_code(token):
        return None
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
        if token.lower() in _view_blob(view_facts_by_pid.get(pid)).lower():
            return None
        return (
            f"Rejected: view a product whose title or facts contain {token} "
            "before the first add_to_cart. Do not cart a different product first."
        )
    return None


def _shock_from_observation(obs: Any, facts: dict[str, str] | None) -> dict[str, Any] | None:
    if not isinstance(obs, dict):
        return None
    inner = obs.get("observation") if isinstance(obs.get("observation"), dict) else obs
    if not isinstance(inner, dict):
        return None
    event = inner.get("market_event")
    if not isinstance(event, dict):
        return None
    kind = str(event.get("kind") or "")
    if kind not in ("stockout", "price_change"):
        if event.get("in_stock_after") is False:
            kind = "stockout"
        elif event.get("new_price") not in (None, ""):
            kind = "price_change"
        else:
            kind = "stockout"
    facts = facts or {}
    return {
        "kind": kind,
        "old_price": event.get("old_price"),
        "new_price": event.get("new_price"),
        "title": facts.get("title") or "",
        "brand": facts.get("brand") or "",
    }


def _fill_shock_identity(
    shock: dict[str, Any] | None,
    view_facts_by_pid: dict[str, dict[str, str]],
    pid: str,
) -> dict[str, Any] | None:
    if not shock:
        return None
    facts = view_facts_by_pid.get(pid) or {}
    if not shock.get("title"):
        shock["title"] = facts.get("title") or ""
    if not shock.get("brand"):
        shock["brand"] = facts.get("brand") or ""
    return shock


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
) -> dict[str, Any]:
    last_err: Exception | None = None
    for model in _model_list():
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


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """EnvPack loop: hybrid essentials + v7 recovery muscle."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    tools = policy["tools"]
    query = str(policy.get("query") or "")
    allowed_fields = _claim_field_allowlist(tools)
    candidates = _parse_candidate_variants(query)
    cand_pids = {pid for pid, _ in candidates}
    ask_first = _needs_ask_first_spec(query)

    boot = (
        f"{query}\n\n"
        "Procedure: ask firm specs if withheld-until-asked; else retrieve → view → "
        "inspect_stock → cart → place_test_order. "
        "On stockout: message must COMMIT to same preferred model/other color "
        "(never ask 'should I?'); then remove → recover → order. "
        "Justification only if asked: include brand + exact view specs + current "
        "price/in_stock; never claim category; do not invent hd/fhd."
    )
    model_hint = _preferred_model_hint(query)
    if model_hint and not _is_model_code(model_hint):
        model_hint = None
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
            "view each listed product_id; rank by stated priority; order a listed key only."
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
    last_shock: dict[str, Any] | None = None
    need_reinspect = False
    last_pick: dict[str, str] = {}
    cart_sku: dict[str, str] = {}
    max_steps = int(policy["max_steps"])

    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        if must_message:
            if (last_shock or {}).get("kind") == "price_change":
                hint = (
                    "REQUIRED this turn: call `message` once. "
                    "STATE that the price rose above the budget and you will replace it "
                    "with another item that still meets the stated specification and budget. "
                    "Do not ask permission. Do not call it a stockout."
                )
            else:
                hint = (
                    "REQUIRED this turn: call `message` once. "
                    "If this is a stockout/'still available?' reply: STATE you will take "
                    "another color of that same product if one is still in stock "
                    "(do not ask permission, do not name a model-code template)."
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
        elif shock_recover or need_reinspect:
            if (last_shock or {}).get("kind") == "price_change":
                recover = (
                    "Price rose above the budget: remove_from_cart if the carted item "
                    "no longer fits. VIEW a different in-budget item that still meets "
                    "the stated specification, inspect_stock, then place_test_order. "
                    "Do not quit. Do not ask permission. Do not treat this as a stockout. "
                    "Claim only facts that the ordered product's view supports."
                )
            else:
                recover = (
                    "Stock shock / re-inspect required: remove_from_cart if cart is stale. "
                    "Recover another color of that same product first if one is in stock"
                )
                if model_hint and _is_model_code(model_hint):
                    recover += f" (the named model token is `{model_hint}`)"
                recover += (
                    ", VIEW the replacement, inspect_stock an in-stock sku, then "
                    "place_test_order. Do not quit; do not ask the shopper what to do; "
                    "do not order without fresh view+inspect_stock."
                )
            messages.append({"role": "user", "content": recover})
            shock_recover = False
        elif not candidates and searchish >= 3 and len(viewed) < 2:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Stop repeating the same search. Reformulate once OR view the best hits, "
                        "then inspect_stock and finish."
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

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for _attempt in range(4):
            inference = _chat(messages=messages, tools=tools)
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
                cart_bad = _first_cart_model_violation(
                    tool_calls,
                    query=query,
                    view_facts_by_pid=view_facts_by_pid,
                    candidates=candidates,
                )
                if cart_bad:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": cart_bad},
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
            else:
                raise RuntimeError("model returned no valid tool call before completion")

        if must_message:
            msg_only = [
                c
                for c in tool_calls
                if c.get("function", {}).get("name") == "message"
            ]
            if msg_only:
                tool_calls = msg_only[:1]

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
            )
            name = call.get("function", {}).get("name") or ""
            if name == "message" and isinstance(args, dict):
                # EnvPack message schema uses `content` (fall back to `message`).
                rewritten = _rewrite_recovery_message(
                    args, query=query, force=recovery_msg_force, shock=last_shock
                )
                if rewritten is not args or recovery_msg_force:
                    args = rewritten
                    if "content" in args and "message" in args:
                        args.pop("message", None)
            if name == "place_test_order" and last_pick:
                if not args.get("product_id"):
                    args["product_id"] = last_pick.get("product_id")
                if not args.get("sku"):
                    args["sku"] = last_pick.get("sku")
            if name in ("inspect_stock", "add_to_cart", "place_test_order"):
                if args.get("product_id") and args.get("sku"):
                    last_pick = {
                        "product_id": str(args["product_id"]),
                        "sku": str(args["sku"]),
                    }
            if name == "add_to_cart":
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
                    viewed.add(str(args["product_id"]))

        if "message" in names:
            must_message = False
            recovery_msg_force = False

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
            if name == "compare" and isinstance(obs, dict):
                node = obs.get("observation") if isinstance(obs.get("observation"), dict) else obs
                compared = node.get("compared") if isinstance(node, dict) else None
                if isinstance(compared, list):
                    for prod in compared:
                        if not isinstance(prod, dict):
                            continue
                        cpid, cfacts = _extract_view_facts(prod)
                        if cpid and cfacts:
                            view_facts_by_pid[cpid] = {
                                **view_facts_by_pid.get(cpid, {}),
                                **cfacts,
                            }
            if name == "inspect_stock":
                need_reinspect = False
            pid_now = last_pick.get("product_id", "")
            raw_args = tool_call["function"].get("arguments", "{}")
            try:
                a_now = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                a_now = {}
            if isinstance(a_now, dict) and a_now.get("product_id"):
                pid_now = str(a_now["product_id"])
            if name in ("add_to_cart", "inspect_stock", "view", "compare", "inspect_cart"):
                shock = _shock_from_observation(
                    obs if isinstance(obs, dict) else {},
                    view_facts_by_pid.get(pid_now),
                )
                if shock:
                    last_shock = _fill_shock_identity(shock, view_facts_by_pid, pid_now)
                    shock_recover = True
                    need_reinspect = True
                    recovery_msg_force = True
            if name in ("inspect_stock", "add_to_cart", "inspect_cart") and payload:
                last_stock = payload
                if _stock_is_false(payload) and (last_shock or {}).get("kind") != "price_change":
                    last_shock = _fill_shock_identity(
                        {"kind": "stockout", "title": "", "brand": ""},
                        view_facts_by_pid,
                        pid_now,
                    )
                    shock_recover = True
                    need_reinspect = True
                    recovery_msg_force = True
            if isinstance(obs, dict):
                inner = obs.get("observation") if isinstance(obs.get("observation"), dict) else obs
                if isinstance(inner, dict) and inner.get("market_event"):
                    shock_recover = True
                    need_reinspect = True
                    recovery_msg_force = True
                    added = inner.get("added")
                    if isinstance(added, dict):
                        last_stock = added

        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            messages.append({"role": "user", "content": content})
            if _looks_like_shopper_intervention(str(content)):
                # After ask-first reply with the actual spec, do NOT force another
                # message — resume shopping. Force message only for questions.
                low = str(content).lower()
                if ask_first and asked_spec and "?" not in low:
                    must_message = False
                else:
                    must_message = True
                if _looks_like_market_event(str(content)):
                    need_reinspect = True
                    recovery_msg_force = True
                    priceish = any(
                        x in low
                        for x in (
                            "more expensive",
                            "price jump",
                            "price changed",
                            "got more expensive",
                            "i can only go",
                        )
                    )
                    pid_um = last_pick.get("product_id", "")
                    if priceish:
                        last_shock = _fill_shock_identity(
                            {
                                "kind": "price_change",
                                "old_price": (last_shock or {}).get("old_price"),
                                "new_price": (last_shock or {}).get("new_price"),
                                "title": (last_shock or {}).get("title") or "",
                                "brand": (last_shock or {}).get("brand") or "",
                            },
                            view_facts_by_pid,
                            pid_um,
                        )
                    elif (last_shock or {}).get("kind") != "price_change":
                        last_shock = _fill_shock_identity(
                            {
                                "kind": "stockout",
                                "title": (last_shock or {}).get("title") or "",
                                "brand": (last_shock or {}).get("brand") or "",
                            },
                            view_facts_by_pid,
                            pid_um,
                        )
                elif "?" in low or any(
                    x in low for x in ("which", "still available", "hold on", "before you")
                ):
                    must_message = True
                    if any(
                        x in low
                        for x in ("still available", "out of stock", "hold on", "before you")
                    ):
                        recovery_msg_force = True
        if any(
            (call.get("observation") or {}).get("done")
            for call in result["calls"]
        ):
            break

        if not must_message and not candidates and not shock_recover:
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
