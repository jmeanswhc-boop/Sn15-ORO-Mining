"""first_king_v2 — EnvPack TF1–TF7 prompt enrichment (race-general).

Chassis: first_king EnvPack /environment/call loop (original first_king.py untouched).
Enrichment drivers (policies only — no suite IDs / query fragments):
  • R100–R147 ShoppingBench cue stats → firm budget, literal attrs, soft prefer,
    multi-intent nearest-scope, search value-tokens
  • ELON_v6 → literal vs semantic match; all firm constraints PASS before cart
  • ORO Bench TF families + EnvPack goal shapes → per-family playbooks
  • failure_is_teacher gates → ask-first, OOS/reinspect, fact-grounded claims
  • Local-run diagnosis (policies, not pack answers): event-first commitment,
    post-event re-rank, minimal TF7 claims, same-model recovery search

Integrity: one strategy; branch only on public shopper text + tools + observations.
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
]
_CHUTES_MODELS = [
    "moonshotai/Kimi-K2.6-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
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
# Category claims often fail faithfulness; omit unless tool schema forces them.
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
_PREF_MODEL_RE = re.compile(
    r"prefer(?:\s+the)?\s+"
    r"(?:black|white|green|red|blue|gold|silver|pink|purple|gray|grey|yellow|orange|"
    r"moonlight\s+white)?\s*"
    r"([A-Za-z0-9][A-Za-z0-9.\-]{2,40})",
    re.I,
)
_PRIORITY_AXIS_RE = re.compile(
    r"(?:highest|largest|lowest|best|prefer(?:ably)?|priority(?:\s+is|\s*:)?)\s+"
    r"(refresh(?:\s*rate)?|screen(?:\s*size)?|price|budget|resolution|ram|"
    r"battery|weight|storage)",
    re.I,
)
_AXIS_ALIAS = {
    "refresh": "refresh_rate",
    "refresh rate": "refresh_rate",
    "screen": "screen_size",
    "screen size": "screen_size",
    "budget": "price",
}

_proxy = ProxyClient(timeout=120, max_retries=3)

_SYSTEM = """You are an ORO Bench shopping agent on EnvPack tools.

Use ONLY the supplied tools. Continue until an observation has done=true.
Obey this episode's shopper text + Task rules only. Race tasks are hidden and
differ from any local suite — never memorize product IDs or suite ranks.

# Matching discipline (ELON-style, race-general)
Extract every constraint, then classify each as LITERAL or SEMANTIC.
- LITERAL (must appear explicitly; no partial normalize): brands, model names,
  identifiers/SKUs, numbers with units, serial names, colors, exact sizes.
  Example: "100ml" rejects "50-150ml"; "Green 2" rejects bare "green".
- SEMANTIC (paraphrase OK): materials, usage, style, common synonyms
  (child↔kid, waterproof↔water-resistant). Prefer false positive over missing
  a clearly matching listing for soft semantic attributes only.
- Title text is an unordered keyword bag — token presence matters, not prose.
- Structured attributes/options beat noisy SEO titles when they conflict.
- A candidate is cart-eligible only when EVERY firm constraint is PASS.
  UNKNOWN firm evidence = FAIL until you view more or ask.

# Constraint scope (multi-intent)
- A modifier applies to the nearest product unless the shopper said "all"/"both".
- Do not invent shared constraints across products. One price span ≠ multi-item budget
  unless separately stated. Kits/multipacks are ONE product, not separate SKUs.

# Event commitment (all families — critical)
Market events usually fire on the FIRST add_to_cart. Therefore:
1) First cart must be a hard-valid, currently best candidate (not a decoy).
2) When a market_event hits that cart (stockout or price jump): remove_from_cart,
   reply once with a declarative `message`, then recover/re-rank, inspect_stock,
   cart a NEW live valid sku, and place_test_order.
3) Never place_test_order on the dead/pre-event sku. Never skip re-inspect after a shock.
4) Do not thrash search after a shock — recover then finish.

# TF1 — Intent decomposition
Silently split into ≤4 mutually exclusive sub-goals (product type, firm budget/specs,
soft prefs). Firm beats soft. Soft brand ("ideally/prefer X"): honor on FIRST cart
when a fully firm-valid preferred item exists; otherwise keep firm and proceed.
Named sizes/RAM/resolution in the goal are LITERAL — do not cart a near-miss inch.
If the shopper asks a question or changes budget/spec: `message` once, then resume.

# TF2 — Retrieval recall
search/filter with value-side tokens from the query (product type + size/color/capacity
numbers). Avoid searching bare schema-key words alone. Survey several distinct
product_ids before buying. After ~3–4 weak searches, view best hits and finish —
do not thrash the same query.

# TF3 — Constraint satisfaction
Firm: required brand, category/device identity, named hardware specs, budget, in-stock.
Match firm specs LITERALLY after `view` (and inspect_stock for live price/stock).
If a firm spec is withheld until you ask → FIRST action is `message` asking for it,
then search with the revealed spec. Mid-episode budget/spec interventions override
prior assumptions — re-check before cart/order. After a stockout shock, recover with
at most 2 focused searches, then view → inspect → cart → order (do not burn the step budget).

# TF4 — Preference reasoning
Among options that already satisfy every firm constraint, prefer listings with visible
evidence for EACH soft use-case axis the shopper named (battery, sweat/water resistance,
secure fit, wireless, etc.). Do not cart a candidate that is silent on a named axis when
another viewed candidate covers more axes. Soft prefs never override firm budget/stock.

# TF5 — Ranking (closed candidate set)
When the shopper lists `product_id::sku` ("compare only these" / candidate variants):
that list is the ENTIRE eligible set. view each listed product_id; read the listed sku's
refresh/size/price/stock; rank by the PRIORITY ORDER STATED IN THIS QUERY
(do not assume a default axis order).
FIRST add_to_cart MUST be the current best listed key (this is the event commitment).
After a stockout/price shock on that key: remove, message, re-inspect remaining LIVE
listed keys, cart the NEW best by the same priority, then order. Never open-catalog search.

# TF6 — Recovery
Search the preferred model token early. Ladder: exact color+model → same model other
color/SKU sibling → other models last. On stockout: declarative message (never ask
permission), remove_from_cart, search/view the SAME model token, inspect_stock, order.
NEVER order a random unrelated model while same-model variants remain untried.
Never quit with "cannot complete".

# TF7 — Justification
Only if the shopper asked why/reason/walk-through. Emit ONLY claims the shopper needs:
always current price + in_stock from latest inspect_stock; plus named specs they stated
(resolution/screen_size/refresh); plus brand when visible. Do NOT add extra spec claims
(extra unsupported fields fail the whole justification). Match claim VALUES to visible
facts. If the shopper used class language (720p/HD/FHD/QHD/4K), prefer that class token
when the listing evidence supports it — do not invent classes. Never claim category.
Omit justification entirely when not asked.

# Finish path
ask (if needed) → search/filter → view/compare → inspect_stock →
(message if asked) → add_to_cart → (handle market_event) → place_test_order.
Late in the step budget: stop exploring and order an in-stock in-budget pick.
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
    """Extract product_id::sku pairs when the shopper frames a closed ranking set."""
    head = _shopper_text(query)
    low = head.lower()
    pairs = _CAND_KEY_RE.findall(head)
    framed = any(
        s in low
        for s in (
            "compare only",
            "candidate variant",
            "candidate variants",
            "these candidate",
            "only these",
        )
    )
    if not framed and len(pairs) < 2:
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


def _parse_rank_priority(query: str) -> list[str]:
    """Ordered ranking axes from this episode's wording (no suite default)."""
    head = _shopper_text(query)
    found: list[str] = []
    for m in _PRIORITY_AXIS_RE.finditer(head):
        raw = m.group(1).lower().strip()
        axis = _AXIS_ALIAS.get(raw, raw.replace(" ", "_"))
        if axis == "refresh_rate" or "refresh" in raw:
            axis = "refresh_rate"
        elif "screen" in raw:
            axis = "screen_size"
        elif raw in ("price", "budget"):
            axis = "price"
        if axis not in found:
            found.append(axis)
    # Tie-break language often lists axes in sequence without "highest" each time.
    if not found:
        low = head.lower()
        for axis, needles in (
            ("refresh_rate", ("refresh rate", "refresh")),
            ("screen_size", ("screen size", "largest screen")),
            ("price", ("lowest price", "cheapest")),
        ):
            if any(n in low for n in needles) and axis not in found:
                found.append(axis)
    return found


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


def _soft_prefer_brand(query: str) -> str | None:
    head = _shopper_text(query)
    # Prefer explicit "ideally … Prefer Brand" / "prefer Brand" patterns.
    m = re.search(
        r"(?:ideally[^.]{0,80}?\bprefer(?:ably)?\s+|prefer(?:ably)?\s+)"
        r"(?:the\s+)?([A-Za-z][\w&.'\-]{1,32})",
        head,
        flags=re.I,
    )
    if not m:
        m = re.search(
            r"\bideally[, ]+([A-Za-z][\w&.'\-]{1,32})\b",
            head,
            flags=re.I,
        )
    if not m:
        return None
    brand = m.group(1).strip().rstrip(".,;:")
    stop = {
        "the",
        "a",
        "an",
        "but",
        "when",
        "if",
        "that",
        "this",
        "one",
        "it",
        "to",
        "for",
        "and",
        "or",
        "in",
        "on",
        "with",
        "firm",
        "requirements",
        "requirement",
        "matter",
        "matters",
        "more",
        "i",
    }
    if brand.lower() in stop or len(brand) < 2:
        return None
    return brand


def _preferred_model_hint(query: str) -> str | None:
    head = _shopper_text(query)
    m = _PREF_MODEL_RE.search(head)
    if not m:
        return None
    token = m.group(1).strip().rstrip(".,;:")
    token = re.split(r"\b(if|specifically|genuine|not|before)\b", token, maxsplit=1)[
        0
    ].strip()
    if len(token) < 3:
        return None
    return token


def _spec_fields_from_query(query: str) -> list[str]:
    """Spec claim fields the shopper explicitly named (race-general)."""
    head = _shopper_text(query).lower()
    fields: list[str] = []
    if any(
        x in head
        for x in (
            "720p",
            "1080p",
            "1440p",
            "4k",
            "uhd",
            "qhd",
            "full hd",
            "fhd",
            " resolution",
            "720p hd",
        )
    ) or re.search(r"\bhd\b", head):
        fields.append("resolution")
    if any(
        x in head
        for x in ("inch", "screen size", "display size", "-inch", " inches")
    ) or re.search(r"\b\d+(?:\.\d+)?-?inch\b", head):
        fields.append("screen_size")
    if "refresh" in head or re.search(r"\b\d+\s*hz\b", head):
        fields.append("refresh_rate")
    return fields


def _shopper_resolution_class(query: str) -> str | None:
    head = _shopper_text(query).lower()
    if any(x in head for x in ("4k", "uhd", "ultra hd", "2160")):
        return "uhd"
    if any(x in head for x in ("qhd", "1440p", "2k")):
        return "qhd"
    if any(x in head for x in ("fhd", "1080p", "full hd")):
        return "fhd"
    if any(x in head for x in ("720p", "hd ready")) or re.search(
        r"(?<![a-z])hd(?![a-z])", head
    ):
        return "hd"
    return None


def _evidence_supports_res_class(text: str, cls: str) -> bool:
    low = (text or "").lower().replace("×", "x")
    if cls == "uhd":
        return any(x in low for x in ("4k", "uhd", "2160", "3840"))
    if cls == "qhd":
        return any(x in low for x in ("qhd", "1440", "2560"))
    if cls == "fhd":
        return any(x in low for x in ("fhd", "1080", "1920", "full hd"))
    if cls == "hd":
        if any(x in low for x in ("fhd", "full hd", "1080", "qhd", "1440", "4k", "uhd")):
            # higher class present without hd-only cue — still ok if 720 present
            return bool(re.search(r"\b720\b|\b1280\s*x\s*720\b|\bhd\b", low))
        return bool(re.search(r"\b720\b|\b1280\b|\bhd\b", low))
    return False


def _required_claim_fields(query: str, allowed: frozenset[str]) -> list[str]:
    """Minimal faithful claim set: price/in_stock + shopper-named specs (+ brand soft)."""
    req: list[str] = []
    for f in ("price", "in_stock"):
        if f in allowed:
            req.append(f)
    for f in _spec_fields_from_query(query):
        if f in allowed and f not in req and f not in _CLAIM_DENY:
            req.append(f)
    if "brand" in allowed:
        req.append("brand")
    return req


def _resolution_claim_value(query: str, view_facts: dict[str, str]) -> str | None:
    """Prefer shopper class token when evidence supports it; else visible string."""
    vis = str(view_facts.get("resolution") or "").strip()
    blob = " ".join(
        str(view_facts.get(k) or "")
        for k in ("resolution", "screen_size", "title", "display")
    )
    cls = _shopper_resolution_class(query)
    if cls and _evidence_supports_res_class(blob + " " + vis, cls):
        return cls
    if vis:
        # Avoid noisy "pixels" suffix that often fails faithfulness vs class gates.
        cleaned = re.sub(r"\s*pixels?\s*$", "", vis, flags=re.I).strip()
        if cls and _evidence_supports_res_class(cleaned, cls):
            return cls
        return cleaned
    return cls


def _parse_hz(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*hz", text or "", flags=re.I)
    return float(m.group(1)) if m else None


def _parse_inches(text: str) -> float | None:
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:inches|inch|in\b|\")",
        text or "",
        flags=re.I,
    )
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", text or "", flags=re.I)
    return float(m.group(1)) if m else None


def _ingest_closed_set_obs(
    observation: Any,
    *,
    candidates: list[tuple[str, str]],
    ledger: dict[str, dict[str, Any]],
) -> None:
    """Update closed-set ranking ledger from public view/compare observations."""
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
        fact_blob = " ".join(f"{k} {v}" for k, v in facts.items())
        variants = prod.get("variants") if isinstance(prod.get("variants"), list) else []
        for var in variants:
            if not isinstance(var, dict):
                continue
            sku = str(var.get("sku") or "")
            key = f"{pid}::{sku}"
            if key not in cand_keys:
                continue
            options = str(var.get("options") or "")
            blob = " ".join([title, fact_blob, options, str(facts.get("display") or "")])
            entry = ledger.setdefault(key, {"product_id": pid, "sku": sku})
            price = var.get("price")
            if price is not None:
                try:
                    entry["price"] = float(price)
                except (TypeError, ValueError):
                    pass
            if "in_stock" in var:
                entry["in_stock"] = bool(var.get("in_stock"))
            hz = _parse_hz(blob)
            if hz is not None:
                entry["refresh_rate"] = hz
            inches = _parse_inches(blob) or _parse_inches(str(facts.get("size") or ""))
            if inches is not None:
                entry["screen_size"] = inches

    if "product_id" in node or "variants" in node or "title" in node:
        ingest_product(node)
    # compare may nest products
    for k in ("products", "items", "results", "compared"):
        block = node.get(k)
        if isinstance(block, list):
            for p in block:
                if isinstance(p, dict):
                    ingest_product(p)


def _best_closed_key(
    ledger: dict[str, dict[str, Any]],
    *,
    candidates: list[tuple[str, str]],
    priority: list[str],
    dead: set[str],
) -> str | None:
    """Pick best live listed key by stated priority (higher refresh/size, lower price)."""
    keys = [f"{pid}::{sku}" for pid, sku in candidates if f"{pid}::{sku}" not in dead]
    if not keys:
        return None
    axes = priority or ["refresh_rate", "screen_size", "price"]

    def sort_key(k: str) -> tuple:
        e = ledger.get(k) or {}
        if e.get("in_stock") is False:
            # push OOS last
            stock_pen = 1
        else:
            stock_pen = 0
        parts: list[float] = [float(stock_pen)]
        for axis in axes:
            if axis == "price":
                val = e.get("price")
                parts.append(float(val) if isinstance(val, (int, float)) else 1e18)
            elif axis == "refresh_rate":
                val = e.get("refresh_rate")
                parts.append(-float(val) if isinstance(val, (int, float)) else 0.0)
            elif axis == "screen_size":
                val = e.get("screen_size")
                parts.append(-float(val) if isinstance(val, (int, float)) else 0.0)
            else:
                parts.append(0.0)
        return tuple(parts)

    keys.sort(key=sort_key)
    return keys[0] if keys else None


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


def _normalize_resolution_value(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return text
    low = text.lower().replace("×", "x")
    if low in ("hd", "fhd", "qhd", "uhd", "4k"):
        return "uhd" if low == "4k" else low
    return text


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
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            return f"{m.group(1)} inches"
    if field == "resolution":
        return _normalize_resolution_value(text)
    return text


def _dig_facts(obj: Any, out: dict[str, str], depth: int = 0) -> None:
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
            "screen_size": "screen_size",
            "screen size": "screen_size",
        }
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in key_map and not isinstance(v, (dict, list)):
                field = key_map[lk]
                if field in _CLAIM_DENY:
                    continue
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
    raw_facts = node.get("facts")
    if isinstance(raw_facts, dict):
        if raw_facts.get("display") and "resolution" not in facts:
            facts["resolution"] = str(raw_facts["display"]).strip()
            facts.setdefault("display", str(raw_facts["display"]).strip())
        if raw_facts.get("size") and "screen_size" not in facts:
            facts["screen_size"] = str(raw_facts["size"]).strip()
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
    return facts


def _sanitize_place_order_args(
    args: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
    last_stock: dict[str, Any] | None,
    last_view_facts: dict[str, str] | None,
) -> dict[str, Any]:
    out = dict(args)
    if not _justification_required(query):
        out.pop("justification", None)
        return out

    view_facts = dict(last_view_facts or {})
    required = _required_claim_fields(query, allowed_fields)
    required_set = set(required)
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()

    just = out.get("justification")
    claims_in = (
        just.get("claims")
        if isinstance(just, dict) and isinstance(just.get("claims"), list)
        else []
    )
    for claim in claims_in:
        if not isinstance(claim, dict):
            continue
        field = _normalize_claim_field(str(claim.get("field", "")), allowed_fields)
        if field is None or field in seen or field not in required_set:
            continue
        raw_val = claim.get("value", "")
        if field == "resolution":
            mapped = _resolution_claim_value(query, view_facts)
            if mapped:
                raw_val = mapped
        elif field in view_facts and field in ("screen_size", "refresh_rate", "brand"):
            raw_val = view_facts[field]
        value = _normalize_claim_value(field, raw_val)
        if not value:
            continue
        cleaned.append({"field": field, "value": value})
        seen.add(field)

    if last_stock:
        if "price" in required_set and last_stock.get("price") is not None:
            val = _normalize_claim_value("price", last_stock.get("price"))
            cleaned = [c for c in cleaned if c["field"] != "price"]
            cleaned.append({"field": "price", "value": val})
            seen.add("price")
        if "in_stock" in required_set and last_stock.get("in_stock") is not None:
            val = _normalize_claim_value("in_stock", last_stock.get("in_stock"))
            cleaned = [c for c in cleaned if c["field"] != "in_stock"]
            cleaned.append({"field": "in_stock", "value": val})
            seen.add("in_stock")

    if "brand" in required_set and "brand" not in seen and view_facts.get("brand"):
        brand = view_facts["brand"]
        if brand.lower() not in ("none", "null", "n/a"):
            cleaned.append({"field": "brand", "value": brand})
            seen.add("brand")

    if "resolution" in required_set and "resolution" not in seen:
        mapped = _resolution_claim_value(query, view_facts)
        if mapped:
            cleaned.append({"field": "resolution", "value": mapped})
            seen.add("resolution")

    for spec in ("screen_size", "refresh_rate"):
        if spec in required_set and spec not in seen and view_facts.get(spec):
            cleaned.append(
                {"field": spec, "value": _normalize_claim_value(spec, view_facts[spec])}
            )
            seen.add(spec)

    cleaned = [
        c
        for c in cleaned
        if c["field"] in required_set and c["field"] not in _CLAIM_DENY and c.get("value")
    ]
    final: list[dict[str, str]] = []
    seen2: set[str] = set()
    for c in cleaned:
        if c["field"] in seen2:
            continue
        seen2.add(c["field"])
        final.append(c)

    if final:
        out["justification"] = {"claims": final}
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
        )
    return parsed


def _rewrite_recovery_message(
    args: dict[str, Any],
    *,
    query: str,
) -> dict[str, Any]:
    content = str(args.get("content") or args.get("message") or "")
    low = content.lower()
    shock = any(
        x in low
        for x in (
            "out of stock",
            "went out",
            "still available",
            "no longer",
            "more expensive",
            "price",
            "budget",
            "went up",
        )
    )
    if not shock:
        return args
    asks = any(
        x in low
        for x in (
            "should i",
            "do you want",
            "would you like",
            "any preference",
            "or look for a different",
            "or a different",
        )
    )
    model = _preferred_model_hint(query)
    if not asks and not model:
        if not any(x in low for x in ("i'm now", "i will", "switching", "selecting")):
            asks = True
    if not asks and not model:
        return args
    out = dict(args)
    if model:
        text = (
            f"The preferred option changed (stock/price) when I added it to cart. "
            f"I'm now selecting another color/variant of the same model ({model}) "
            f"that is still in stock and within budget — firm requirements stay. "
            f"I will only change models if no same-model variant remains."
        )
    else:
        text = (
            "That item's availability/price changed when I added it to cart. "
            "I'm now selecting another in-stock option that still meets your "
            "firm budget and requirements."
        )
    if "content" in out or "message" not in out:
        out["content"] = text
        out.pop("message", None)
    else:
        out["message"] = text
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
    )
    return any(c in low for c in cues)


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


def _build_boot(
    query: str,
    *,
    candidates: list[tuple[str, str]],
    ask_first: bool,
    model_hint: str | None,
    soft_brand: str | None,
    rank_priority: list[str],
) -> str:
    boot = (
        f"{query}\n\n"
        "Procedure: ask firm specs if withheld-until-asked; else retrieve → view → "
        "inspect_stock → FIRST cart a hard-valid best candidate (event commitment) → "
        "on market_event: remove → declarative message → recover/re-rank → "
        "inspect_stock → cart → place_test_order. "
        "Justification only if asked: ONLY price+in_stock+named specs+brand; "
        "no extra fields."
    )
    if ask_first:
        boot += (
            "\n\nASK-FIRST (TF3): First tool call must be `message` asking for the "
            "withheld firm hardware spec. After the reply, search with that spec."
        )
    if soft_brand:
        boot += (
            f"\n\nSOFT BRAND (TF1/TF4): Prefer `{soft_brand}` on the first cart when a "
            "fully firm-valid preferred listing exists; never violate firm budget/specs."
        )
    if model_hint:
        boot += (
            f"\n\nPREFERRED MODEL (TF6): `{model_hint}`. Search this token early. "
            "On stockout, recover the same model in another color/SKU before changing models."
        )
    if candidates:
        listed = ", ".join(f"{pid}::{sku}" for pid, sku in candidates)
        prio = (
            " → ".join(rank_priority)
            if rank_priority
            else "the priority order stated above"
        )
        boot += (
            "\n\nCLOSED CANDIDATE SET / TF5 (mandatory): "
            f"{listed}. "
            f"view each listed product_id; rank by [{prio}]; "
            "FIRST add_to_cart MUST be the current best listed key; "
            "after shock, re-rank remaining LIVE listed keys and order the new best. "
            "No open-catalog substitutes."
        )
    elif _justification_required(query):
        specs = _spec_fields_from_query(query)
        spec_txt = ", ".join(specs) if specs else "(no named hardware specs)"
        boot += (
            "\n\nTF7: place_test_order justification must include price, in_stock"
            + (f", {spec_txt}" if specs else "")
            + ", and brand when visible. Do NOT include any other claim fields. "
            "If the shopper named HD/FHD/QHD/4K, claim that class when evidence supports it."
        )
    return boot


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """EnvPack loop: enriched TF1–TF7 prompts + closed-set / recovery / claim gates."""

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
    model_hint = _preferred_model_hint(query)
    soft_brand = _soft_prefer_brand(query)
    rank_priority = _parse_rank_priority(query)

    boot = _build_boot(
        query,
        candidates=candidates,
        ask_first=ask_first,
        model_hint=model_hint,
        soft_brand=soft_brand,
        rank_priority=rank_priority,
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
    first_cart_done = False
    dead_keys: set[str] = set()
    rank_ledger: dict[str, dict[str, Any]] = {}
    max_steps = int(policy["max_steps"])

    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        best_key = _best_closed_key(
            rank_ledger,
            candidates=candidates,
            priority=rank_priority,
            dead=dead_keys,
        )
        if must_message:
            hint = (
                "REQUIRED this turn: call `message` once. "
                "If stockout/price jump/'still available?': STATE you will take another "
                "valid option (same preferred model/color ladder when relevant). "
                "Do not ask permission."
            )
            if model_hint:
                hint += f" Preferred model token: {model_hint}."
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
        elif (
            candidates
            and len(viewed & cand_pids) >= len(cand_pids)
            and not first_cart_done
            and not shock_recover
        ):
            prio = " → ".join(rank_priority) if rank_priority else "stated priority"
            tip = (
                f"All listed candidates viewed. FIRST add_to_cart must be the current "
                f"best by [{prio}]"
            )
            if best_key:
                tip += f" (ledger best right now: {best_key})"
            tip += (
                ". inspect_stock that listed sku if needed, then add_to_cart it "
                "(event commitment). Do not cart a dominated listed option first."
            )
            messages.append({"role": "user", "content": tip})
        elif shock_recover or need_reinspect:
            if candidates:
                recover = (
                    "Market shock on closed set: remove_from_cart if needed. "
                    "Re-inspect remaining LIVE listed candidates"
                )
                if best_key:
                    recover += f" (current best live: {best_key})"
                recover += (
                    ", then add_to_cart the NEW best listed key and place_test_order. "
                    "Do not reopen the dead sku; no open-catalog search."
                )
            else:
                recover = (
                    "Market shock / re-inspect required: remove_from_cart if cart is stale. "
                    "Recover SAME preferred model other color/variant first"
                )
                if model_hint:
                    recover += (
                        f" — search/view model `{model_hint}` before any other model"
                    )
                recover += (
                    ", inspect_stock an in-stock sku, then place_test_order. "
                    "Do not quit; do not ask the shopper what to do; max 2 recovery searches."
                )
            messages.append({"role": "user", "content": recover})
            shock_recover = False
        elif not candidates and searchish >= 3 and len(viewed) < 2:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "TF2 anti-thrash: stop repeating the same search. "
                        "Reformulate once OR view best hits, then inspect_stock and finish."
                    ),
                }
            )
        elif (
            not candidates
            and (searchish >= 5 or (event_seen and searchish >= 2))
            and len(viewed) >= 1
        ):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Stop exploratory search. Among viewed products meeting firm "
                        "constraints (and preference evidence when relevant), "
                        "inspect_stock → add_to_cart → place_test_order NOW."
                    ),
                }
            )
        elif remaining <= 6 and not must_message:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Only ~{remaining} steps left. If you have an in-stock "
                        "in-budget sku, inspect_stock if needed and place_test_order NOW."
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
                                    "Rejected: after a market event you must "
                                    "inspect_stock before place_test_order."
                                ),
                            },
                        ]
                    )
                    continue
                reject_reason = None
                if candidates and best_key:
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
                            reject_reason = (
                                f"Rejected: {key} is dead after market event. "
                                f"Use a live listed key such as {best_key}."
                            )
                            break
                        if (
                            name == "add_to_cart"
                            and not first_cart_done
                            and not event_seen
                            and key != best_key
                            and key in {f"{p}::{s}" for p, s in candidates}
                            and (rank_ledger.get(best_key) or {}).get("refresh_rate")
                            is not None
                        ):
                            reject_reason = (
                                "Rejected: first cart must be the current best "
                                f"listed key by stated priority: {best_key} "
                                f"(not {key})."
                            )
                            break
                if reject_reason:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": reject_reason},
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
        if "add_to_cart" in names:
            first_cart_done = True

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
            )
            name = call.get("function", {}).get("name") or ""
            if name == "message" and isinstance(args, dict):
                args = _rewrite_recovery_message(args, query=query)
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
            name = tool_call.get("function", {}).get("name") or ""
            if isinstance(obs, dict):
                _dig_facts(obs, last_view_facts)
                _ingest_closed_set_obs(obs, candidates=candidates, ledger=rank_ledger)
            if name in ("view", "compare") and isinstance(obs, dict):
                facts = _extract_view_facts(obs)
                if facts:
                    last_view_facts = {**last_view_facts, **facts}
            if name == "inspect_stock":
                need_reinspect = False
            if name in ("inspect_stock", "add_to_cart", "inspect_cart") and payload:
                last_stock = payload
                if _stock_is_false(payload):
                    shock_recover = True
                    need_reinspect = True
                    event_seen = True
                    pid = str(payload.get("product_id") or "")
                    sku = str(payload.get("sku") or "")
                    if pid and sku:
                        dead_keys.add(f"{pid}::{sku}")
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
                    ev = inner["market_event"]
                    tgt = ev.get("target") if isinstance(ev, dict) else None
                    if isinstance(tgt, dict) and tgt.get("product_id") and tgt.get("sku"):
                        dead_keys.add(f"{tgt['product_id']}::{tgt['sku']}")
                    added = inner.get("added")
                    if isinstance(added, dict):
                        last_stock = added

        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            messages.append({"role": "user", "content": content})
            if _looks_like_shopper_intervention(str(content)):
                low = str(content).lower()
                if ask_first and asked_spec and "?" not in low:
                    must_message = False
                else:
                    must_message = True
                if _looks_like_market_event(str(content)):
                    need_reinspect = True
                    event_seen = True
                elif "?" in low or any(
                    x in low
                    for x in ("which", "still available", "hold on", "before you")
                ):
                    must_message = True
        if any(
            (call.get("observation") or {}).get("done") for call in result["calls"]
        ):
            break

        if not must_message and not candidates and not shock_recover:
            nudge = (
                "Episode not done. Prefer inspect_stock → add_to_cart → "
                "place_test_order on an in-stock sku that meets firm constraints. "
                "Omit justification unless the shopper asked why."
            )
            if soft_brand and len(viewed) >= 1:
                nudge += (
                    f" If a firm-valid `{soft_brand}` option was viewed, prefer it "
                    "for the first cart."
                )
            if model_hint and event_seen:
                nudge += (
                    f" After shock, search `{model_hint}` same-model variants before "
                    "switching models."
                )
            messages.append({"role": "user", "content": nudge})
        elif (
            not must_message
            and candidates
            and len(viewed & cand_pids) >= len(cand_pids)
            and first_cart_done
        ):
            prio = (
                " → ".join(rank_priority)
                if rank_priority
                else "the shopper's stated priority"
            )
            msg = (
                f"Closed-set continue: rank live listed keys by [{prio}], "
                "inspect_stock the winner, add_to_cart listed sku, place_test_order."
            )
            if best_key:
                msg += f" Current best live key: {best_key}."
            messages.append({"role": "user", "content": msg})

    return dialogue


__all__ = ["agent_main"]
