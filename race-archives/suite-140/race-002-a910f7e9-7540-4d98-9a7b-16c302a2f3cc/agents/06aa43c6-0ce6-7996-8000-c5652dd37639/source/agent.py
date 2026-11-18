"""ORO Bench EnvPack agent — failureteacher DNA + Lovelystar TF gates.

Runtime only:
  ProxyClient POST /inference/chat/completions
  ProxyClient POST /environment/call
  agent_main(problem_data) -> dialogue

Chassis: lean EnvPack ProxyClient loop (ilostaboy/failureteacher live path; AST padding stripped).
Behavior DNA: failureteacher multi-donor TF merge + Lovelystar firm/soft policy.
ShoppingBench engines omitted.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# OpenRouter by default; matching Chutes TEE when INFERENCE_PROVIDER=chutes.
_OR = {
    "kimi": "moonshotai/kimi-k2.6",
    "deepseek": "deepseek/deepseek-v3.2",
    "deepseek_chat": "deepseek/deepseek-chat-v3.1",
    "qwen": "qwen/qwen3.5-397b-a17b",
    "glm": "z-ai/glm-5.2",
}
_TEE = {
    "kimi": "moonshotai/Kimi-K2.6-TEE",
    "deepseek": "deepseek-ai/DeepSeek-V3.2-TEE",
    "deepseek_chat": "deepseek-ai/DeepSeek-V3.1-TEE",
    "qwen": "Qwen/Qwen3.5-397B-A17B-TEE",
    "glm": "zai-org/GLM-5.2-TEE",
}
# Per-family ladders (failureteacher / ilostaboy DNA). Justification: never glm-first.
_FAMILY_ORDER = {
    "intent": ("kimi", "deepseek", "glm", "qwen", "deepseek_chat"),
    "retrieval": ("kimi", "qwen", "deepseek", "glm", "deepseek_chat"),
    "constraint": ("deepseek", "glm", "kimi", "qwen", "deepseek_chat"),
    "preference": ("kimi", "deepseek", "qwen", "glm", "deepseek_chat"),
    "ranking": ("deepseek", "qwen", "kimi", "glm", "deepseek_chat"),
    "recovery": ("deepseek", "kimi", "glm", "qwen", "deepseek_chat"),
    "justification": ("deepseek", "kimi", "deepseek_chat", "qwen", "glm"),
    "generic": ("deepseek", "kimi", "qwen", "glm", "deepseek_chat"),
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

_proxy = ProxyClient(timeout=120, max_retries=3)


def _provider() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw in ("chutes", "openrouter"):
        return raw
    if getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "chutes"


_SYSTEM = """You are an ORO Bench shopping agent (failureteacher DNA + Lovelystar TF gates).
Use ONLY supplied tools. Continue until done=true. Never memorize product IDs or suite ranks.
Hidden races differ from local packs — obey this episode's shopper text + Task rules only.

## TF1 Intent (Lovelystar)
Silently decompose into ≤4 mutually exclusive sub-goals (keywords, firm budget/specs, soft prefs).
If shopper asks a question / changes budget / asks which item: NEXT action is `message`, then resume.
Ask-first: if they say a firm spec is told when you ask → FIRST call `message` for that spec.

## TF2 Retrieval (HiSun lean + madlol; KAK anti-thrash)
search/filter with reformulations. Survey a few distinct product_ids — enough to be confident.
Hard cap mindset: after ~3–5 weak searches, view best hits and proceed (do not thrash).
Prefer in_stock=true when availability matters.

## TF3 Constraints (madlol / ELON)
Firm specs are NON-NEGOTIABLE (category_path, RAM/screen/resolution/refresh, budget, stock).
Match brands/models/sizes/numbers LITERALLY after `view`. Soft brand only after all firm pass.

## TF4 Preference (Lovelystar)
Among fully valid options, prefer better soft fit, then cheaper. Soft ≠ override firm.

## TF5 Ranking (0828 closedRank)
When shopper lists product_id::sku ("compare only these"): that set is CLOSED.
view every listed product_id; rank by stated priority; cart+order MUST be a listed key.
Never open-catalog search for substitutes.

## TF6 Recovery (Lovelystar + 0828 finishSafe) — CRITICAL
On stockout or price shock: `message` that COMMITS to the same preferred model in another color
(never "should I?" / "any preference?"). Then remove_from_cart → search that model →
inspect_stock → cart → order. NEVER order on out-of-stock. Never quit.

## TF7 Justification (Lovelystar + 0828)
Only if shopper asks why/reason/walk me through. Claims MUST match viewed facts.display /
facts.size plus latest inspect_stock. Claim ONLY fields the shopper named (plus price /
in_stock / brand when present). resolution = hd|fhd|qhd|uhd when facts.display supports
that class — never bare 720p, never invent class, never claim brand "None".
Never first-cart a null-brand or wrong-resolution listing when brand/class is firm.
Never claim category. Omit justification entirely when not asked.

## Finish discipline
search/filter → view/compare → inspect_stock → add_to_cart → place_test_order.
Late in the budget, stop exploring and order an in-stock in-budget pick.
"""


def _shopper_text(query: str) -> str:
    return query.split("Task rules:", 1)[0].strip()


def _parse_candidate_variants(query: str) -> list[tuple[str, str]]:
    head = _shopper_text(query)
    low = head.lower()
    pairs = _CAND_KEY_RE.findall(head)
    if not any(
        (
            s in low
            for s in (
                "compare only",
                "candidate variant",
                "candidate variants",
                "these candidate",
                "only these",
            )
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
    return any((n in head for n in needles))


def _needs_ask_first_spec(query: str) -> bool:
    return bool(_ASK_SPEC_RE.search(_shopper_text(query)))


def _spec_fields_from_query(query: str) -> list[str]:
    """Infer which justification spec fields the shopper named (race-general)."""
    head = _shopper_text(query).lower()
    fields: list[str] = []
    if any(
        (
            x in head
            for x in (
                "720p",
                "1080p",
                "1440p",
                "4k",
                "uhd",
                "qhd",
                "full hd",
                " resolution",
            )
        )
    ) or re.search("\\bhd\\b", head):
        fields.append("resolution")
    if any((x in head for x in ("inch", "screen", "display size"))):
        fields.append("screen_size")
    if "refresh" in head or re.search("\\b\\d+\\s*hz\\b", head):
        fields.append("refresh_rate")
    if re.search("\\bram\\b", head):
        pass
    return fields


def _claim_field_allowlist(tools: list[dict[str, Any]]) -> frozenset[str]:
    for tool in tools:
        fn = tool.get("function") or tool
        if (fn.get("name") or "") != "place_test_order":
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        claims = (
            ((props.get("justification") or {}).get("properties") or {})
            .get("claims", {})
            .get("items", {})
            .get("properties", {})
            .get("field", {})
        )
        enum = claims.get("enum")
        if isinstance(enum, list) and enum:
            return frozenset((str(x) for x in enum))
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
    if mapped and mapped in allowed and (mapped not in _CLAIM_DENY):
        return mapped
    return None


def _query_wants_resolution_class(query: str) -> str | None:
    """Return hd|fhd|qhd|uhd if shopper named a resolution class."""
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


def _display_supports_class(display: str, cls: str) -> bool:
    """Whether facts.display can faithfully back a class claim.

    Bare `720p` does NOT support claiming `hd` (live TF7). Need explicit `hd`
    or 1280-class dims. Pixel strings support the matching class only.
    """
    raw = str(display or "").strip()
    if not raw or not cls:
        return False
    low = raw.casefold().replace("×", "x").replace("*", "x")
    parts = [p.strip() for p in re.split(r"[,;/|]", low) if p.strip()] or [low]
    for part in parts:
        if cls == "uhd" and any(x in part for x in ("3840", "2160", "4k", "uhd")):
            return True
        if cls == "qhd" and any(x in part for x in ("2560", "1440", "qhd")):
            return True
        if cls == "fhd" and any(x in part for x in ("1920", "1080", "fhd", "full hd")):
            return True
        if cls == "hd" and (
            re.search(r"(?<![a-z])hd(?![a-z])", part) or "1280" in part
        ):
            return True
    return False


def _resolution_claim_value(display: str, query: str) -> str | None:
    """Build a TF7 resolution claim from facts.display."""
    raw = str(display or "").strip()
    if not raw:
        return None
    wanted = _query_wants_resolution_class(query)
    if wanted and _display_supports_class(raw, wanted):
        return wanted
    # When shopper named a class, never fall back to unsupported pixels/720p.
    if wanted:
        return None
    # Exact pixel / nickname only when no class was named (rare).
    low = raw.lower().replace("×", "x")
    if low in ("hd", "fhd", "qhd", "uhd", "4k"):
        return "uhd" if low == "4k" else low
    if re.search(r"\d{3,4}\s*x\s*\d{3,4}", low):
        return raw
    return None


def _normalize_resolution_value(text: str, query: str) -> str:
    """Ground resolution claims in the view display string."""
    claim = _resolution_claim_value(text, query)
    return claim if claim is not None else ""


def _query_named_brand(query: str) -> str | None:
    head = _shopper_text(query)
    m = re.search(
        r"(?:want a|find me a|get me an?|need an?)\s+(.+?)(?=\s+(?:\d+\s*p\b|720p|1080p|1440p|4k|uhd|qhd|fhd|full\s*hd|\bhd\b|\d+(?:\.\d+)?-?inch|gaming|smartphone|tablet|monitor|headphone))",
        head,
        flags=re.I,
    )
    if not m:
        return None
    brand = re.sub(r"\s+", " ", m.group(1)).strip(" .,")
    if not brand or brand.lower() in {"in-stock", "in stock", "an", "a"}:
        return None
    if len(brand) > 40:
        return None
    return brand


def _extract_view_facts(observation: Any) -> tuple[str | None, dict[str, str]]:
    """Pull product_id + brand/resolution/screen_size from a view observation.

    Catalog specs live under facts.display / facts.size — map those.
    """
    facts: dict[str, str] = {}
    if not isinstance(observation, dict):
        return (None, facts)
    node = observation.get("observation")
    if not isinstance(node, dict):
        node = observation
    pid = node.get("product_id") or observation.get("product_id")
    if pid is not None:
        pid = str(pid)

    raw_facts = node.get("facts") if isinstance(node.get("facts"), dict) else {}
    if node.get("brand") not in (None, "", "None"):
        facts["brand"] = str(node["brand"]).strip()
    display = str(raw_facts.get("display") or raw_facts.get("resolution") or "").strip()
    if display and display.lower() not in ("none", "null"):
        facts["display"] = display
        facts["resolution"] = display
    size = str(
        raw_facts.get("size") or raw_facts.get("screen_size") or raw_facts.get("screen") or ""
    ).strip()
    if size and size.lower() not in ("none", "null"):
        facts["size"] = size
        facts["screen_size"] = size
    for key in ("model", "phone_type", "device_operating_system", "operating_system"):
        val = raw_facts.get(key)
        if val not in (None, "", "None"):
            facts[key] = str(val).strip()
    if node.get("title"):
        facts["title"] = str(node["title"]).strip()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk == "display" and v not in (None, "", "None"):
                    facts.setdefault("display", str(v).strip())
                    facts.setdefault("resolution", str(v).strip())
                elif lk in ("size", "screen_size") and v not in (None, "", "None"):
                    facts.setdefault("size", str(v).strip())
                    facts.setdefault("screen_size", str(v).strip())
                elif lk in ("brand", "resolution", "refresh_rate", "model") and v not in (
                    None,
                    "",
                    "None",
                ):
                    facts.setdefault(lk, str(v).strip())
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(obj, list):
            for item in obj[:12]:
                walk(item)

    walk(node)
    return (pid, facts)


def _product_gate_violation(
    *,
    query: str,
    pid: str,
    view_facts_by_pid: dict[str, dict[str, str]],
) -> str | None:
    """Reject TF7 cart/order picks that cannot pass brand/resolution gates."""
    if not _justification_required(query):
        return None
    facts = view_facts_by_pid.get(str(pid) or "") or {}
    brand_want = _query_named_brand(query)
    brand_have = str(facts.get("brand") or "").strip()
    if brand_want:
        if not brand_have or brand_have.lower() in ("none", "null"):
            return (
                f"Rejected: brand is null/missing; shopper requires `{brand_want}`. "
                "View another branded listing first."
            )
        if (
            brand_want.casefold() not in brand_have.casefold()
            and brand_have.casefold() not in brand_want.casefold()
        ):
            return (
                f"Rejected: brand `{brand_have}` does not match required `{brand_want}`."
            )
    wanted = _query_wants_resolution_class(query)
    display = facts.get("display") or facts.get("resolution") or ""
    if wanted and not display:
        return "Rejected: no facts.display observed; view a fact-backed listing first."
    if wanted and display and not _display_supports_class(display, wanted):
        return (
            f"Rejected: facts.display `{display}` does not support required "
            f"resolution class `{wanted}`."
        )
    if wanted and display and _resolution_claim_value(display, query) is None:
        return (
            f"Rejected: cannot form a faithful `{wanted}` resolution claim from "
            f"facts.display `{display}`."
        )
    return None


def _preferred_model_hint(query: str) -> str | None:
    """Extract TF6 model token from strongly-prefer / prefer-the-COLOR-MODEL only.

    Soft brands (`prefer Meizu`) must not become recovery model tokens.
    EnvPack tokens always contain a digit (`431915030732`, `635`, `8520`).
    """
    head = _shopper_text(query)
    colors = (
        "black|white|green|red|blue|gold|silver|pink|purple|gray|grey|"
        "yellow|orange|moonlight\\s+white"
    )
    patterns = (
        rf"strongly\s+prefer(?:\s+the)?\s+(?:{colors})?\s*"
        rf"([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"prefer(?:s)?\s+the\s+(?:{colors})\s+"
        rf"([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"eye on the\s+(?:{colors})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
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
    if not re.search(r"\d", token):
        return None
    return token or None


def _normalize_claim_value(field: str, value: Any, *, query: str) -> str:
    text = str(value).strip()
    if field == "in_stock":
        low = text.lower()
        if low in ("true", "1", "yes", "in stock", "instock"):
            return "true"
        if low in ("false", "0", "no", "out of stock"):
            return "false"
    if field == "price":
        cleaned = re.sub("[^\\d.]", "", text.replace(",", ""))
        if cleaned:
            try:
                return str(float(cleaned))
            except ValueError:
                return text
    if field == "refresh_rate":
        m = re.search("(\\d+(?:\\.\\d+)?)\\s*hz", text, flags=re.I)
        if m:
            return f"{m.group(1)}Hz"
        if re.fullmatch("\\d+(?:\\.\\d+)?", text):
            return f"{text}Hz"
    if field == "screen_size":
        m = re.search("(\\d+(?:\\.\\d+)?)", text)
        if m:
            return f"{m.group(1)} inches"
    if field == "resolution":
        return _normalize_resolution_value(text, query)
    return text


def _dig_facts(obj: Any, out: dict[str, str], depth: int = 0) -> None:
    """Accumulate observed product facts for TF7 claim grounding (0828 finishSafe)."""
    if depth > 6 or obj is None:
        return
    if isinstance(obj, dict):
        key_map = {
            "brand": "brand",
            "category": "category",
            "category_path": "category",
            "price": "price",
            "current_price": "price",
            "in_stock": "in_stock",
            "stock": "in_stock",
            "refresh_rate": "refresh_rate",
            "refresh": "refresh_rate",
            "resolution": "resolution",
            "display": "resolution",
            "screen_size": "screen_size",
            "screen size": "screen_size",
            "size": "screen_size",
        }
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in key_map and (not isinstance(v, (dict, list))):
                field = key_map[lk]
                if field == "brand" and str(v).strip().lower() in ("none", "null", "n/a", ""):
                    continue
                if field in _CLAIM_DENY:
                    continue
                if field == "category" and isinstance(v, str) and ("/" in v):
                    v = v.split("/")[-1].strip() or v
                if field == "in_stock":
                    if isinstance(v, bool):
                        v = "true" if v else "false"
                    elif isinstance(v, (int, float)):
                        v = "true" if v > 0 else "false"
                out[field] = str(v).strip()
            else:
                _dig_facts(v, out, depth + 1)
    elif isinstance(obj, list):
        for item in obj[:20]:
            _dig_facts(item, out, depth + 1)


def _claim_values_compatible(field: str, claimed: str, observed: str) -> bool:
    c = claimed.strip().lower()
    o = observed.strip().lower()
    if not c or not o:
        return False
    if field == "price":
        try:
            return (
                abs(
                    float(re.sub("[^\\d.]", "", c) or "nan")
                    - float(re.sub("[^\\d.]", "", o) or "nan")
                )
                < 0.05
            )
        except ValueError:
            return c in o or o in c
    if field == "in_stock":

        def as_bool(x: str) -> bool | None:
            if x in ("true", "1", "yes", "in stock", "instock"):
                return True
            if x in ("false", "0", "no", "out of stock"):
                return False
            return None

        cb, ob = (as_bool(c), as_bool(o))
        return cb is not None and cb == ob
    if field in ("refresh_rate", "screen_size"):
        cm = re.search("(\\d+(?:\\.\\d+)?)", c)
        om = re.search("(\\d+(?:\\.\\d+)?)", o)
        if not (cm and om):
            return False
        if field == "refresh_rate":
            return cm.group(1) == om.group(1)
        return abs(float(cm.group(1)) - float(om.group(1))) < 0.05
    if field == "resolution":
        return c in o or o in c or c.replace(" ", "") in o.replace(" ", "")
    return c == o or c in o or o in c


def _sanitize_place_order_args(
    args: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
    last_stock: dict[str, Any] | None,
    last_view_facts: dict[str, str] | None,
    view_facts_by_pid: dict[str, dict[str, str]] | None = None,
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
    elif last_view_facts:
        view_facts = dict(last_view_facts)
    for claim in claims_in:
        if not isinstance(claim, dict):
            continue
        field = _normalize_claim_field(str(claim.get("field", "")), allowed_fields)
        if field is None or field in seen:
            continue
        # Extra unnamed specs (e.g. resolution on a size-only query) zero TF7.
        if field in ("resolution", "screen_size", "refresh_rate") and field not in needed_specs:
            continue
        raw_val = claim.get("value", "")
        if field == "brand":
            if view_facts.get("brand"):
                raw_val = view_facts["brand"]
            else:
                continue
        elif field in view_facts and field in (
            "resolution",
            "screen_size",
            "refresh_rate",
        ):
            raw_val = view_facts[field]
        value = _normalize_claim_value(field, raw_val, query=query)
        if value == "" or value.lower() in ("none", "null", "n/a"):
            continue
        if field == "brand" and value.lower() in ("none", "null", "n/a"):
            continue
        if field == "resolution":
            vis = view_facts.get("display") or view_facts.get("resolution") or ""
            claim_val = _resolution_claim_value(vis or value, query)
            if not claim_val:
                continue
            value = claim_val
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
    for spec in _spec_fields_from_query(query):
        if (
            spec in allowed_fields
            and spec not in seen
            and (spec not in _CLAIM_DENY)
            and view_facts.get(spec)
        ):
            cleaned.append(
                {
                    "field": spec,
                    "value": _normalize_claim_value(
                        spec, view_facts[spec], query=query
                    ),
                }
            )
            seen.add(spec)
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
    last_view_facts: dict[str, str] | None,
    view_facts_by_pid: dict[str, dict[str, str]] | None = None,
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
            last_view_facts=last_view_facts,
            view_facts_by_pid=view_facts_by_pid,
        )
    return parsed


def _rewrite_recovery_message(args: dict[str, Any], *, query: str) -> dict[str, Any]:
    """Force declarative same-model recovery commit (0828_v2 / LS PASS pattern)."""
    content = str(args.get("content") or "")
    low = content.lower()
    if not any(
        (x in low for x in ("out of stock", "went out", "still available", "no longer"))
    ):
        return args
    asks = any(
        (
            x in low
            for x in (
                "should i",
                "do you want",
                "would you like",
                "any preference",
                "preferences?",
                "or look for a different",
                "or a different",
            )
        )
    )
    model = _preferred_model_hint(query)
    if not (asks or model):
        if model and "same" not in low and (model.lower().split()[0] not in low):
            pass
        else:
            return args
    if not model:
        if asks:
            args = dict(args)
            args["content"] = (
                "That item went out of stock when I added it to cart. I'm now selecting another in-stock option that still matches your firm budget and category requirements."
            )
        return args
    args = dict(args)
    args["content"] = (
        f"The preferred option went out of stock when I added it to cart. I'm now looking for another color/variant of the same model ({model}) that is still in stock and within your budget — keeping category firm. I will only change models if no same-model variant remains."
    )
    return args


def _looks_like_market_event(text: str) -> bool:
    low = (text or "").lower()
    return any(
        (
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
    return any((c in low for c in cues))


def _obs_stock_payload(observation: Any) -> dict[str, Any] | None:
    """Extract stock/price facts from nested EnvPack observations."""
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


def _family(query: str) -> str:
    head = _shopper_text(query).lower()
    if _parse_candidate_variants(query):
        return "ranking"
    if _justification_required(query):
        return "justification"
    if re.search(r"prefer(?:s)? the [a-z]+ [a-z0-9][a-z0-9.\-]*\d", head):
        return "recovery"
    if _needs_ask_first_spec(query):
        return "constraint"
    if (
        ("matches" in head and ("phrase" in head or '"' in head or "'" in head))
        or ("find an in-stock" in head and "category" in head)
        or "surface at least" in query.lower()
    ):
        return "retrieval"
    if any(
        x in head
        for x in ("rather", "cheapest", "preference", "prefer", "tradeoff", "trade-off")
    ) and any(x in head for x in ("among", "options", "between", "rather", "cheapest")):
        return "preference"
    if "firm requirement" in head or "ideally" in head or "prefer" in head:
        return "intent"
    return "generic"


def _model_list(family: str = "generic") -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    table = _TEE if _provider() == "chutes" else _OR
    order = _FAMILY_ORDER.get(family) or _FAMILY_ORDER["generic"]
    return [table[key] for key in order]


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
            suffix = f"r{attempt}-{int(time.time() * 1000) % 1000000}"
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
    if any((n in ("search", "filter") for n in names)) and len(
        viewed & cand_pids
    ) < len(cand_pids):
        return f"Closed candidate set: do not search/filter until every listed product is viewed. Missing: {sorted(cand_pids - viewed)}"
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
                return f"view product_id {pid} is outside the listed candidates. View only: {sorted(cand_pids)}"
        if name == "compare":
            ids = args.get("product_ids") or []
            if isinstance(ids, list):
                bad = [str(pid) for pid in ids if str(pid) not in cand_pids]
                if bad:
                    return f"compare includes product_ids outside the closed set: {bad}. Compare only: {sorted(cand_pids)}"
        if name in ("add_to_cart", "inspect_stock", "place_test_order"):
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and pid not in cand_pids:
                return f"{name} product_id {pid} is outside listed candidates {sorted(cand_pids)}"
            if pid and sku and (f"{pid}::{sku}" not in cand_keys):
                if name in ("add_to_cart", "place_test_order"):
                    return f"{name} {pid}::{sku} is not a listed variant. Use one of: {sorted(cand_keys)}"
    return None


def _oos_order_violation(
    tool_calls: list[dict[str, Any]], *, last_stock: dict[str, Any] | None
) -> str | None:
    """Reject ordering when latest stock/cart says out of stock (market shock)."""
    if not _stock_is_false(last_stock):
        return None
    names = _tool_names(tool_calls)
    if "place_test_order" in names:
        return "Rejected: latest stock/cart shows in_stock=false (market shock). remove_from_cart if needed, pick another candidate, inspect_stock, then order only an in-stock sku."
    return None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """EnvPack loop: LS phases + safe-band gates for hidden races."""
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
    family = _family(query)
    boot = (
        f"{query}\n\nProcedure: ask firm specs if withheld-until-asked; else retrieve → view → "
        "inspect_stock → cart → place_test_order. On stockout: message must COMMIT to same "
        "preferred model/other color (never ask 'should I?'); then remove → recover → order. "
        "Justification only if asked: claim ONLY named specs + price/in_stock + real brand "
        "(never brand None; never extra resolution; resolution class only when facts.display supports it)."
    )
    model_hint = _preferred_model_hint(query)
    if ask_first:
        boot += "\n\nASK-FIRST: Your first tool call must be `message` asking for the withheld firm hardware spec. After the reply, search with that spec and complete the order without apology loops."
    if model_hint:
        boot += f"\n\nPREFERRED MODEL TOKEN: `{model_hint}`. On stockout of the preferred color/variant, recover that same model in another color before changing models."
    if candidates:
        listed = ", ".join((f"{pid}::{sku}" for pid, sku in candidates))
        boot += f"\n\nCLOSED CANDIDATE SET (mandatory): {listed}. view each listed product_id; rank by stated priority; order a listed key only."
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
    view_facts_by_pid: dict[str, dict[str, str]] = {}
    shock_recover = False
    need_reinspect = False
    last_pick: dict[str, str] = {}
    cart_sku: dict[str, str] = {}
    max_steps = int(policy["max_steps"])
    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        if must_message:
            hint = "REQUIRED this turn: call `message` once. If this is a stockout/'still available?' reply: STATE you will take another color/variant of the SAME preferred model (do not ask permission)."
            if model_hint:
                hint += f" Preferred model token: {model_hint}."
            messages.append({"role": "user", "content": hint})
        elif candidates and len(viewed & cand_pids) < len(cand_pids):
            missing = sorted(cand_pids - viewed)
            messages.append(
                {
                    "role": "user",
                    "content": f"Still missing views for closed-set product_ids: {missing}. Call view on the next missing id (or compare).",
                }
            )
        elif shock_recover or need_reinspect:
            recover = "Market shock / re-inspect required: remove_from_cart if cart is stale. Recover SAME preferred model other color/variant first"
            if model_hint:
                recover += f" (model `{model_hint}`)"
            recover += ", inspect_stock an in-stock sku, then place_test_order. Do not quit; do not ask the shopper what to do; do not order without fresh inspect_stock."
            messages.append({"role": "user", "content": recover})
            shock_recover = False
        elif not candidates and searchish >= 3 and (len(viewed) < 2):
            messages.append(
                {
                    "role": "user",
                    "content": "TF2 anti-thrash (HiSun/KAK): stop repeating the same search. Reformulate once OR view best hits, then inspect_stock and finish.",
                }
            )
        elif not candidates and searchish >= 5 and (len(viewed) >= 2):
            messages.append(
                {
                    "role": "user",
                    "content": "Search budget spent. Among viewed products meeting firm constraints, inspect_stock → add_to_cart → place_test_order NOW.",
                }
            )
        elif remaining <= 6 and (not must_message):
            messages.append(
                {
                    "role": "user",
                    "content": f"Only ~{remaining} steps left. If you have an in-stock in-budget sku, inspect_stock if needed and place_test_order NOW. Stop exploratory search.",
                }
            )
        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for _attempt in range(4):
            try:
                inference = _chat(messages=messages, tools=tools, family=family)
            except RuntimeError:
                tool_calls = []
                assistant_content = ""
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
                names_try = _tool_names(tool_calls)
                if (
                    need_reinspect
                    and "place_test_order" in names_try
                    and ("inspect_stock" not in names_try)
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
                                "content": "Rejected: after a market event you must inspect_stock (and fix cart) before place_test_order.",
                            },
                        ]
                    )
                    continue
                gate_hit = None
                for call in tool_calls:
                    n = call.get("function", {}).get("name") or ""
                    if n not in ("add_to_cart", "place_test_order"):
                        continue
                    raw = call["function"].get("arguments", "{}")
                    try:
                        a = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        a = {}
                    pid = str((a or {}).get("product_id") or "")
                    if n == "add_to_cart" and pid not in view_facts_by_pid:
                        continue
                    gate_hit = _product_gate_violation(
                        query=query,
                        pid=pid,
                        view_facts_by_pid=view_facts_by_pid,
                    )
                    if gate_hit:
                        break
                if gate_hit:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": gate_hit},
                        ]
                    )
                    continue
            break
        if not tool_calls:
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
            elif must_message and ask_first and (not asked_spec):
                tool_calls = [
                    {
                        "id": f"forced-ask-{turn}",
                        "type": "function",
                        "function": {
                            "name": "message",
                            "arguments": json.dumps(
                                {
                                    "content": "What is the firm hardware requirement (RAM, screen size, resolution, or refresh rate)?"
                                }
                            ),
                        },
                    }
                ]
                assistant_content = "Asking for the withheld firm spec."
            else:
                seed = re.sub(r"\s+", " ", _shopper_text(query))[:160]
                tool_calls = [
                    {
                        "id": f"forced-search-{turn}",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps(
                                {"query": seed, "k": 10, "in_stock": True}
                            ),
                        },
                    }
                ]
                assistant_content = "Starting catalog search."
        if must_message:
            msg_only = [
                c for c in tool_calls if c.get("function", {}).get("name") == "message"
            ]
            if msg_only:
                tool_calls = msg_only[:1]
        names = _tool_names(tool_calls)
        if any((n in ("search", "filter") for n in names)):
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
                last_view_facts=last_view_facts,
                view_facts_by_pid=view_facts_by_pid,
            )
            name = call.get("function", {}).get("name") or ""
            if name == "message" and isinstance(args, dict):
                rewritten = _rewrite_recovery_message(args, query=query)
                if rewritten is not args:
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
            if isinstance(obs, dict):
                _dig_facts(obs, last_view_facts)
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
                if facts:
                    last_view_facts = {**last_view_facts, **facts}
                if pid and facts:
                    view_facts_by_pid[pid] = {
                        **view_facts_by_pid.get(pid, {}),
                        **facts,
                    }
            if name == "inspect_stock":
                need_reinspect = False
            if name in ("inspect_stock", "add_to_cart", "inspect_cart") and payload:
                last_stock = payload
                if _stock_is_false(payload):
                    shock_recover = True
                    need_reinspect = True
            if isinstance(obs, dict):
                inner = (
                    obs.get("observation")
                    if isinstance(obs.get("observation"), dict)
                    else obs
                )
                if isinstance(inner, dict) and inner.get("market_event"):
                    shock_recover = True
                    need_reinspect = True
                    added = inner.get("added")
                    if isinstance(added, dict):
                        last_stock = added
        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            messages.append({"role": "user", "content": content})
            if _looks_like_shopper_intervention(str(content)):
                low = str(content).lower()
                if ask_first and asked_spec and ("?" not in low):
                    must_message = False
                else:
                    must_message = True
                if _looks_like_market_event(str(content)):
                    need_reinspect = True
                elif "?" in low or any(
                    (
                        x in low
                        for x in ("which", "still available", "hold on", "before you")
                    )
                ):
                    must_message = True
        if any(
            ((call.get("observation") or {}).get("done") for call in result["calls"])
        ):
            break
        if not must_message and (not candidates) and (not shock_recover):
            messages.append(
                {
                    "role": "user",
                    "content": "Episode not done. Prefer inspect_stock → add_to_cart → place_test_order on an in-stock sku that meets firm constraints. Omit justification unless the shopper asked why.",
                }
            )
        elif (
            not must_message
            and candidates
            and (len(viewed & cand_pids) >= len(cand_pids))
        ):
            messages.append(
                {
                    "role": "user",
                    "content": "All listed candidates viewed. Rank by stated priority, inspect_stock the winner, add_to_cart listed sku, place_test_order.",
                }
            )
    return dialogue


__all__ = ["agent_main"]
