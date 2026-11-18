"""ORO Bench EnvPack agent — ilostaboy live loop + edge DNA + Lovelystar TF policy.

Runtime only:
  ProxyClient POST /inference/chat/completions
  ProxyClient POST /environment/call
  agent_main(problem_data) -> dialogue

Chassis: lean EnvPack ProxyClient loop (ilostaboy/edge live path; AST padding stripped).
Behavior DNA: edge recovery/closed-set/justification discipline + Lovelystar TF1–TF7.
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
# Per-family ladders (edge/ilostaboy DNA). Justification: never glm-first.
_FAMILY_ORDER = {
    "intent": ("kimi", "deepseek", "qwen", "deepseek_chat", "glm"),
    "retrieval": ("kimi", "qwen", "deepseek", "deepseek_chat", "glm"),
    "constraint": ("deepseek", "kimi", "qwen", "deepseek_chat", "glm"),
    "preference": ("kimi", "deepseek", "qwen", "deepseek_chat", "glm"),
    "ranking": ("deepseek", "qwen", "kimi", "deepseek_chat", "glm"),
    "recovery": ("deepseek", "kimi", "qwen", "deepseek_chat", "glm"),
    "justification": ("deepseek", "kimi", "deepseek_chat", "qwen", "glm"),
    "generic": ("deepseek", "kimi", "qwen", "deepseek_chat", "glm"),
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


_SYSTEM = """You are an ORO Bench shopping agent (edge DNA + Lovelystar TF gates on EnvPack tools).
Use ONLY supplied tools. Continue until done=true. Never memorize product IDs.
Obey this episode's shopper text and Task rules only.

## TF1 Intent (Lovelystar)
Silently decompose into firm budget/specs/category/stock vs soft brand/color/prefs (≤4 sub-goals mindset).
If the shopper asks a question or changes budget, the next tool is `message`, then resume.
If a firm spec is told when you ask, the FIRST call must be `message`.

## TF2 Retrieval
search/filter with in_stock=true, a few differently worded queries, then view.
After 3–5 weak searches, stop thrashing and finish: view → inspect_stock → cart → order.

## TF3 Constraint
After view, match brand/model/size/number literally.
Soft brand never overrides a failed firm spec (category, budget, stock, named hardware).

## TF4 Preference
Among fully valid options, prefer better soft fit, then cheaper.
When the shopper names sweat / water resistance / sport fit / battery / wireless,
search and prefer titles/facts that attest those axes (waterproof, IPX, sweatproof).
Soft preference never overrides budget, stock, or category.
On a price shock: message that you will recover to another listing that still
fits those priorities, remove_from_cart the shocked sku, then search → view →
inspect_stock → cart → order. Never leave the shocked sku in the cart.

## TF5 Ranking (closed set)
If the query lists product_id::sku, that set is CLOSED.
View every listed product_id. Rank refresh_rate → screen_size → lowest price.
First cart MUST be the current best listed key. After a market shock: remove it,
then cart the new best remaining listed key (never re-order the shocked sku).

## TF6 Recovery — CRITICAL
On stockout or price shock / "still available?":
  1) `message` that COMMITS to the same preferred model in another color (never "should I?").
  2) remove_from_cart → search that exact model token → inspect_stock → cart → order.
Never place_test_order on an out-of-stock line. Never quit. Never switch models
while same-model variants remain.

## TF7 Justification
Only if the shopper asks why / reason / walk me through.
Claims must match viewed facts.display / facts.size plus latest inspect_stock.
resolution claims are hd|fhd|qhd|uhd only when facts.display supports that class.
Never first-cart a null-brand or wrong-resolution listing when brand/class is firm.
Never claim category. If not asked, omit justification entirely.

## Finish
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
    """Return hd|fhd|qhd|uhd if shopper named a resolution class (0828 PASS pattern)."""
    low = _shopper_text(query).lower()
    if re.search("\\b(4k|uhd|ultra\\s*hd)\\b", low):
        return "uhd"
    if re.search("\\b(qhd|1440p|2k)\\b", low):
        return "qhd"
    if re.search("\\b(fhd|1080p|full\\s*hd)\\b", low):
        return "fhd"
    if re.search("\\b(720p|hd)\\b", low):
        return "hd"
    return None


def _display_class(text: str) -> str:
    """Map a facts.display string onto hd|fhd|qhd|uhd when possible."""
    token = str(text or "").casefold().replace("×", "x").replace("*", "x")
    if any(part in token for part in ("3840", "2160", "4k", "uhd")):
        return "uhd"
    if any(part in token for part in ("2560", "1440", "qhd")):
        return "qhd"
    if any(part in token for part in ("1920", "1080", "fhd", "full hd")):
        return "fhd"
    if any(part in token for part in ("1280", "720p", "720")) or re.search(
        r"(?<![a-z])hd(?![a-z])", token
    ):
        return "hd"
    return ""


def _display_supports_class(display: str, cls: str) -> bool:
    """Whether facts.display can faithfully back a class claim (hd|fhd|qhd|uhd).

    Live judge: bare `720p` does NOT support claiming `hd`; need an explicit `hd`
    token or 1280-class dims. Pixel strings support the matching class.
    """
    raw = str(display or "").strip()
    if not raw or not cls:
        return False
    low = raw.casefold().replace("×", "x").replace("*", "x")
    parts = [p.strip() for p in re.split(r"[,;/|]", low) if p.strip()]
    if not parts:
        parts = [low]
    for part in parts:
        if cls == "uhd" and any(x in part for x in ("3840", "2160", "4k", "uhd")):
            return True
        if cls == "qhd" and any(x in part for x in ("2560", "1440", "qhd")):
            return True
        if cls == "fhd" and any(x in part for x in ("1920", "1080", "fhd", "full hd")):
            return True
        if cls == "hd":
            if re.search(r"(?<![a-z])hd(?![a-z])", part) or "1280" in part:
                return True
    return False


def _resolution_claim_value(display: str, query: str) -> str | None:
    """Build a TF7 resolution claim from facts.display (class tokens only)."""
    raw = str(display or "").strip()
    if not raw:
        return None
    wanted = _query_wants_resolution_class(query)
    if wanted and _display_supports_class(raw, wanted):
        return wanted
    # Fall back to a single supported class when the shopper named none.
    for cls in ("uhd", "qhd", "fhd", "hd"):
        if _display_supports_class(raw, cls):
            if wanted and cls != wanted:
                return None
            return cls
    return None


def _normalize_resolution_value(text: str, query: str) -> str:
    """Ground resolution claims in the view display string."""
    claim = _resolution_claim_value(text, query)
    return claim if claim is not None else ""


def _hz_from_text(text: str) -> float:
    vals = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*hz", str(text or ""), flags=re.I)]
    return max(vals) if vals else 0.0


def _inches_from_text(text: str) -> float:
    raw = str(text or "")
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:inches|inch|in\b|")', raw, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    # Parenthetical sizes like "(27 inch)" already covered; bare sensible sizes only.
    vals = []
    for num in re.findall(r'(\d+(?:\.\d+)?)', raw):
        try:
            v = float(num)
        except ValueError:
            continue
        if 5.0 <= v <= 49.0:
            vals.append(v)
    return max(vals) if vals else 0.0


def _extract_view_facts(observation: Any) -> tuple[str | None, dict[str, str]]:
    """Pull product_id + brand/resolution/screen_size from a view observation.

    Catalog hardware specs live under facts.display / facts.size — map those.
    Also stash variant sku rows (options/price/stock) for closed-set ranking.
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
    path = node.get("category_path")
    if isinstance(path, list) and path:
        facts["category"] = str(path[-1]).strip()
    hz = _hz_from_text(facts.get("title", "")) or _hz_from_text(
        str(raw_facts.get("refresh_rate") or "")
    )
    if hz:
        facts["refresh_rate"] = f"{hz:g}Hz"
        facts["_hz"] = str(hz)

    variants = node.get("variants")
    if isinstance(variants, list):
        slim: list[dict[str, Any]] = []
        for row in variants[:24]:
            if not isinstance(row, dict):
                continue
            slim.append(
                {
                    "sku": str(row.get("sku") or ""),
                    "price": row.get("price"),
                    "in_stock": row.get("in_stock"),
                    "options": row.get("options"),
                }
            )
        if slim:
            facts["_variants_json"] = json.dumps(slim)

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


def _preferred_model_hint(query: str) -> str | None:
    """Extract TF6 model token from 'strongly prefer the COLOR MODEL' only.

    Soft brand prefs (`Ideally, I prefer Meizu`) must NOT become recovery models.
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
    # Soft brand words without digits are not recovery model tokens.
    if not re.search(r"\d", token):
        return None
    return token or None


def _query_named_brand(query: str) -> str | None:
    """Brand token when the shopper leads with a named brand before the product class."""
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


def _variant_rows(facts: dict[str, str]) -> list[dict[str, Any]]:
    raw = facts.get("_variants_json")
    if not raw:
        return []
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return rows if isinstance(rows, list) else []


def _variant_for_sku(facts: dict[str, str], sku: str) -> dict[str, Any] | None:
    want = str(sku or "")
    for row in _variant_rows(facts):
        if str(row.get("sku") or "") == want:
            return row
    return None


def _rank_tuple_for_key(
    pid: str,
    sku: str,
    view_facts_by_pid: dict[str, dict[str, str]],
) -> tuple[float, float, float]:
    """Higher is better for (refresh_hz, screen_in, -price)."""
    facts = view_facts_by_pid.get(pid) or {}
    variant = _variant_for_sku(facts, sku) or {}
    hz = _hz_from_text(facts.get("title", "")) or _hz_from_text(
        facts.get("refresh_rate", "")
    )
    opt = str(variant.get("options") or "")
    size = _inches_from_text(opt) or _inches_from_text(
        facts.get("screen_size") or facts.get("size") or ""
    ) or _inches_from_text(facts.get("title", ""))
    try:
        price = float(variant["price"]) if variant.get("price") is not None else 1e18
    except (TypeError, ValueError):
        price = 1e18
    return (hz, size, -price)


def _best_closed_keys(
    candidates: list[tuple[str, str]],
    view_facts_by_pid: dict[str, dict[str, str]],
    *,
    exclude: set[str] | None = None,
) -> list[tuple[str, str]]:
    exclude = exclude or set()
    scored: list[tuple[tuple[float, float, float], str, str]] = []
    for pid, sku in candidates:
        key = f"{pid}::{sku}"
        if key in exclude:
            continue
        scored.append((_rank_tuple_for_key(pid, sku, view_facts_by_pid), pid, sku))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [(pid, sku) for _, pid, sku in scored]


def _product_gate_violation(
    *,
    query: str,
    family: str,
    pid: str,
    sku: str,
    view_facts_by_pid: dict[str, dict[str, str]],
    model_hint: str | None,
) -> str | None:
    """Reject cart/order picks that cannot pass family hard gates."""
    facts = view_facts_by_pid.get(str(pid) or "") or {}
    if family == "justification" or _justification_required(query):
        brand_want = _query_named_brand(query)
        brand_have = str(facts.get("brand") or "").strip()
        if brand_want:
            if not brand_have or brand_have.lower() in ("none", "null"):
                return (
                    f"Rejected: brand is null/missing; shopper requires `{brand_want}`. "
                    "View another branded listing first."
                )
            if brand_want.casefold() not in brand_have.casefold() and brand_have.casefold() not in brand_want.casefold():
                return (
                    f"Rejected: brand `{brand_have}` does not match required `{brand_want}`."
                )
        wanted = _query_wants_resolution_class(query)
        display = facts.get("display") or facts.get("resolution") or ""
        if wanted and display and not _display_supports_class(display, wanted):
            return (
                f"Rejected: facts.display `{display}` does not support required "
                f"resolution class `{wanted}`."
            )
        if wanted and not display:
            return "Rejected: no facts.display observed; view a fact-backed listing first."
        if wanted and display and _resolution_claim_value(display, query) is None:
            return (
                f"Rejected: cannot form a faithful `{wanted}` resolution claim from "
                f"facts.display `{display}`."
            )
    if family == "recovery" and model_hint:
        if not facts:
            return None
        blob = " ".join(
            str(facts.get(k) or "")
            for k in ("model", "title", "display", "phone_type")
        ).casefold()
        if model_hint.casefold() not in blob:
            return (
                f"Rejected: product does not attest preferred model `{model_hint}`. "
                "Search/view that model token before ordering."
            )
    if family == "ranking" and sku:
        # Ranking gate is handled by closed-set best-key checks.
        return None
    return None


def _canonical_recovery_message(
    model: str | None, *, preference: bool = False
) -> str:
    """Declarative recovery commit (no permission ask)."""
    if preference:
        return (
            "That listing changed price or availability when I added it to cart. "
            "I will remove it and recover to another in-stock listing that still "
            "fits your stated priorities within budget — without asking you to choose."
        )
    if model:
        return (
            f"The preferred option went out of stock when I added it to cart. "
            f"Following your rules, I will now take the same model ({model}) in "
            f"another color/variant that is still in stock, keep budget and category "
            f"firm, and only change models if no same-model variant remains."
        )
    return (
        "That item went out of stock when I added it to cart. I'm now selecting "
        "another in-stock option that still matches your firm budget and category "
        "requirements."
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
    if view_facts_by_pid and pid and (pid in view_facts_by_pid):
        view_facts = dict(view_facts_by_pid[pid])
    for claim in claims_in:
        if not isinstance(claim, dict):
            continue
        field = _normalize_claim_field(str(claim.get("field", "")), allowed_fields)
        if field is None or field in seen:
            continue
        if (
            field in ("resolution", "screen_size", "refresh_rate")
            and field not in needed_specs
        ):
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
        if value == "":
            continue
        if field == "resolution":
            vis = view_facts.get("display") or view_facts.get("resolution") or ""
            claim = _resolution_claim_value(vis or value, query)
            if not claim:
                continue
            value = claim
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
        if spec in allowed_fields and spec not in seen and (spec not in _CLAIM_DENY):
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
            # never invent hd/fhd without a view-grounded string
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
    preference: bool = False,
) -> dict[str, Any]:
    """Force declarative recovery commit.

    EnvPack message tool args use key `content`. On stockout / price shock /
    'still available?' turns (`force=True`), always emit the canonical commit.
    """
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
            "price jump",
            "price jumped",
            "repriced",
            "more expensive",
            "above your budget",
            "exceeds your budget",
            "over budget",
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
    model = None if preference else _preferred_model_hint(query)
    model_l = (model or "").lower()
    commits = (
        bool(model_l)
        and model_l in low
        and (
            "same" in low
            or "another color" in low
            or "other color" in low
            or "same model" in low
        )
    ) or (
        preference
        and any(
            x in low
            for x in (
                "recover",
                "priorities",
                "another listing",
                "another option",
                "fits your",
            )
        )
        and not asks
    )
    if force:
        out = dict(args)
        out["content"] = _canonical_recovery_message(model, preference=preference)
        out.pop("message", None)
        return out
    need = asks or stockish
    if not need:
        return args
    if commits and (not asks):
        return args
    out = dict(args)
    out["content"] = _canonical_recovery_message(model, preference=preference)
    out.pop("message", None)
    return out


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
    if re.search(
        r"strongly\s+prefer(?:s)?\s+the\s+[a-z]+\s+[a-z0-9][a-z0-9.\-]*\d",
        head,
    ) or re.search(r"prefer(?:s)?\s+the\s+[a-z]+\s+[a-z0-9][a-z0-9.\-]*\d", head):
        return "recovery"
    if _needs_ask_first_spec(query):
        return "constraint"
    if ("matches" in head and ("phrase" in head or '"' in head or "'" in head)) or (
        "find an in-stock" in head and "category" in head
    ):
        return "retrieval"
    if any(
        x in head
        for x in (
            "rather",
            "cheapest",
            "preference",
            "tradeoff",
            "trade-off",
            "sweat",
            "water resistance",
            "waterproof",
        )
    ):
        return "preference"
    if "firm requirement" in head or "ideally" in head or "prefer" in head:
        return "intent"
    return "generic"


def _parse_budget(query: str) -> float | None:
    head = _shopper_text(query)
    m = re.search(
        r"(?:under|below|at or under|at or below|up to|budget(?:\s+of)?)\s*(\d+(?:\.\d+)?)",
        head,
        flags=re.I,
    )
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)\s*PHP", head, flags=re.I)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _preference_search_hint(query: str) -> str:
    head = _shopper_text(query).lower()
    bits: list[str] = []
    if any(x in head for x in ("sweat", "water resistance", "waterproof", "weather")):
        bits.append("sweatproof waterproof IPX sport")
    if any(
        x in head
        for x in ("wireless", "headphone", "earbuds", "earbud", "headset", "bluetooth")
    ):
        bits.append("wireless bluetooth")
    if "battery" in head:
        bits.append("long battery")
    if "sport" in head or "gym" in head or "run" in head or "workout" in head:
        bits.append("sport fit")
    # TF4 axes are usually all in play for headphone preference episodes.
    if "headphone" in head or "earbud" in head or "headset" in head:
        if "wireless bluetooth" not in bits:
            bits.append("wireless bluetooth")
        if "sweatproof waterproof IPX sport" not in " ".join(bits):
            bits.append("sweatproof waterproof IPX")
    return " ".join(bits).strip()


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
    ) < min(2, len(cand_pids)):
        return f"Closed candidate set: do not search/filter. Call view on listed product_ids first: {sorted(cand_pids)}"
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
                    return (
                        f"compare includes product_ids outside the closed set: {bad}. "
                        f"Compare only: {sorted(cand_pids)}"
                    )
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
    budget = _parse_budget(query)
    pref_hint = _preference_search_hint(query) if family == "preference" else ""
    boot = (
        f"{query}\n\n"
        "Procedure: ask firm specs if withheld-until-asked; else retrieve → view → "
        "inspect_stock → cart → place_test_order. "
        "On stockout/price shock: message must COMMIT to recovery (never ask 'should I?'); "
        "then remove_from_cart → recover → order. "
        "Justification only if asked: include brand + exact view specs + current "
        "price/in_stock; never claim category; do not invent hd/fhd."
    )
    model_hint = _preferred_model_hint(query)
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
    if family == "preference":
        boot += (
            "\n\nPREFERENCE: After clarifying priority if needed, search with the shopper's "
            "soft axes attested in titles/facts"
        )
        if pref_hint:
            boot += f" (include terms like: {pref_hint})"
        boot += (
            ". On price shock: commit to recovering another priority-fitting listing, "
            "remove_from_cart the shocked sku, then continue. Do not order the shocked sku."
        )
    if candidates:
        listed = ", ".join((f"{pid}::{sku}" for pid, sku in candidates))
        boot += (
            f"\n\nCLOSED CANDIDATE SET (mandatory): {listed}. "
            "view each listed product_id; rank refresh → screen → lowest price using "
            "title Hz + variant options/price; FIRST add_to_cart the current best; "
            "after shock remove it and cart the next-best remaining listed key only."
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
    recovery_committed = False
    shocked_keys: set[str] = set()
    cart_stale: dict[str, str] = {}
    for turn in range(1, int(policy["max_steps"]) + 1):
        ranked = (
            _best_closed_keys(candidates, view_facts_by_pid, exclude=shocked_keys)
            if candidates and len(viewed & cand_pids) >= len(cand_pids)
            else []
        )
        if must_message and recovery_committed and not ask_first:
            # Already committed once — do not burn another message turn.
            must_message = False
            recovery_msg_force = False
        if must_message:
            if family == "preference" and recovery_msg_force:
                hint = (
                    "REQUIRED this turn: call `message` once. COMMIT that you will "
                    "remove the shocked listing and recover another in-stock option "
                    "that still fits the shopper's priorities within budget "
                    "(do not ask permission)."
                )
            else:
                hint = (
                    "REQUIRED this turn: call `message` once. If this is a "
                    "stockout/'still available?' reply: STATE you will take another "
                    "color/variant of the SAME preferred model (do not ask permission)."
                )
            if model_hint and family != "preference":
                hint += f" Preferred model token: {model_hint}."
            messages.append({"role": "user", "content": hint})
        elif candidates and len(viewed & cand_pids) < len(cand_pids):
            missing = sorted(cand_pids - viewed)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Still missing views for closed-set product_ids: {missing}. "
                        "Call view on the next missing id (or compare)."
                    ),
                }
            )
        elif shock_recover or cart_stale:
            if family == "ranking" and ranked:
                best_pid, best_sku = ranked[0]
                recover = (
                    "Market shock on closed set: remove_from_cart the shocked sku, then "
                    f"inspect_stock/add_to_cart the NEW best listed key "
                    f"{best_pid}::{best_sku} (refresh→size→price among remaining). "
                    "Do not re-cart the shocked sku."
                )
            elif family == "preference":
                recover = (
                    "Market shock / stale cart: remove_from_cart the shocked sku, then "
                    "search a priority-fitting in-stock replacement"
                )
                if pref_hint:
                    recover += f" (terms: {pref_hint})"
                recover += (
                    ", inspect_stock, add_to_cart, place_test_order. "
                    "Do not order the shocked sku; do not ask the shopper what to do."
                )
            else:
                recover = (
                    "Market shock: prior sku not safely in stock. remove_from_cart if "
                    "needed. Recover SAME preferred model other color/variant first"
                )
                if model_hint:
                    recover += (
                        f" (search query MUST include model token `{model_hint}`)"
                    )
                recover += (
                    ", inspect_stock an in-stock sku, then place_test_order. "
                    "Do not quit; do not ask the shopper what to do."
                )
            messages.append({"role": "user", "content": recover})
            shock_recover = False
        elif ranked:
            best_pid, best_sku = ranked[0]
            order_line = ", ".join(f"{p}::{s}" for p, s in ranked[:4])
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Closed-set rank (refresh→size→price): {order_line}. "
                        f"inspect_stock then add_to_cart {best_pid}::{best_sku} next "
                        "(or place_test_order that key if already carted safely)."
                    ),
                }
            )
        elif not candidates and searchish >= 3 and (len(viewed) < 2):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Stop repeating the same search. Reformulate once OR view "
                        "best hits, then inspect_stock and finish."
                    ),
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
                    continue
                if cart_stale and "place_test_order" in _tool_names(tool_calls):
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
                                    "Rejected: cart still holds the shocked sku "
                                    f"{cart_stale.get('product_id')}::{cart_stale.get('sku')}. "
                                    "Call remove_from_cart on that sku before ordering a replacement."
                                ),
                            },
                        ]
                    )
                    continue
                if (
                    cart_stale
                    and "add_to_cart" in _tool_names(tool_calls)
                    and "remove_from_cart" not in _tool_names(tool_calls)
                ):
                    # Allow add only after remove; nudge if model skips remove.
                    for call in tool_calls:
                        if (call.get("function", {}).get("name") or "") != "add_to_cart":
                            continue
                        raw = call["function"].get("arguments", "{}")
                        try:
                            a = json.loads(raw) if isinstance(raw, str) else raw
                        except json.JSONDecodeError:
                            a = {}
                        pid = str((a or {}).get("product_id") or "")
                        sku = str((a or {}).get("sku") or "")
                        if pid == cart_stale.get("product_id") and sku == cart_stale.get(
                            "sku"
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
                                            "Rejected: do not re-cart the shocked sku. "
                                            "remove_from_cart it, then pick a different in-budget listing."
                                        ),
                                    },
                                ]
                            )
                            break
                    else:
                        pass
                    if messages and "Rejected: do not re-cart" in str(
                        messages[-1].get("content")
                    ):
                        continue
                # Closed-set: first cart / order must be current best remaining key.
                if ranked and any(
                    n in _tool_names(tool_calls)
                    for n in ("add_to_cart", "place_test_order", "inspect_stock")
                ):
                    best_pid, best_sku = ranked[0]
                    best_key = f"{best_pid}::{best_sku}"
                    bad = False
                    for call in tool_calls:
                        n = call.get("function", {}).get("name") or ""
                        if n not in ("add_to_cart", "place_test_order", "inspect_stock"):
                            continue
                        raw = call["function"].get("arguments", "{}")
                        try:
                            a = json.loads(raw) if isinstance(raw, str) else raw
                        except json.JSONDecodeError:
                            a = {}
                        pid = str((a or {}).get("product_id") or "")
                        sku = str((a or {}).get("sku") or "")
                        key = f"{pid}::{sku}"
                        if key in shocked_keys or (
                            n in ("add_to_cart", "place_test_order")
                            and key != best_key
                        ):
                            bad = True
                            break
                    if bad:
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
                                        "Rejected: closed-set rank requires "
                                        f"{best_key} next (refresh→size→price among "
                                        "non-shocked listed keys)."
                                    ),
                                },
                            ]
                        )
                        continue
                # Brand / resolution / model hard gates before cart or order.
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
                    sku = str((a or {}).get("sku") or "")
                    # Require a prior view for justification/recovery gates on order;
                    # allow first cart only when view facts already exist for pid.
                    if n == "add_to_cart" and pid not in view_facts_by_pid:
                        continue
                    gate_hit = _product_gate_violation(
                        query=query,
                        family=family,
                        pid=pid,
                        sku=sku,
                        view_facts_by_pid=view_facts_by_pid,
                        model_hint=model_hint,
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
            elif ranked:
                bp, bs = ranked[0]
                tool_calls = [
                    {
                        "id": f"forced-inspect-{turn}",
                        "type": "function",
                        "function": {
                            "name": "inspect_stock",
                            "arguments": json.dumps(
                                {"product_id": bp, "sku": bs}
                            ),
                        },
                    }
                ]
                assistant_content = f"Inspecting ranked closed-set key {bp}::{bs}."
            else:
                seed = re.sub(r"\s+", " ", _shopper_text(query))[:160]
                if model_hint and (recovery_committed or family == "recovery"):
                    seed = f"{model_hint} smartphone"
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
                view_facts_by_pid=view_facts_by_pid,
            )
            name = call.get("function", {}).get("name") or ""
            if name == "message" and isinstance(args, dict):
                args = _rewrite_recovery_message(
                    args,
                    query=query,
                    force=recovery_msg_force,
                    preference=(family == "preference"),
                )
                if recovery_msg_force or family in ("recovery", "preference", "ranking", "retrieval", "intent"):
                    # Mark commit when we emit a recovery-shaped message.
                    low = str(args.get("content") or "").lower()
                    if any(
                        x in low
                        for x in (
                            "went out of stock",
                            "changed price",
                            "recover",
                            "another color",
                            "same model",
                        )
                    ):
                        recovery_committed = True
            if name == "search" and isinstance(args, dict) and model_hint and (
                recovery_committed or shock_recover or cart_stale or family == "recovery"
            ):
                q = str(args.get("query") or "")
                if model_hint not in q:
                    args["query"] = f"{model_hint} {q}".strip()
                args["in_stock"] = True
            if name == "remove_from_cart" and isinstance(args, dict):
                if (
                    str(args.get("product_id") or "") == cart_stale.get("product_id")
                    and str(args.get("sku") or "") == cart_stale.get("sku")
                ) or (
                    cart_stale
                    and str(args.get("product_id") or "") == cart_stale.get("product_id")
                ):
                    cart_stale = {}
            if name in ("add_to_cart", "inspect_stock", "place_test_order") and isinstance(
                args, dict
            ):
                if args.get("product_id") and args.get("sku"):
                    # Track latest intended pick for stale-cart compares.
                    pass
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
            if not ask_first or recovery_committed:
                recovery_committed = True
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
                    view_facts_by_pid[pid] = {**view_facts_by_pid.get(pid, {}), **facts}
            if name in ("inspect_stock", "add_to_cart", "inspect_cart") and payload:
                last_stock = payload
                if _stock_is_false(payload):
                    shock_recover = True
                    recovery_msg_force = not recovery_committed
                    must_message = not recovery_committed
                    if payload.get("product_id") and payload.get("sku"):
                        cart_stale = {
                            "product_id": str(payload["product_id"]),
                            "sku": str(payload["sku"]),
                        }
                        shocked_keys.add(
                            f"{cart_stale['product_id']}::{cart_stale['sku']}"
                        )
                elif budget is not None and payload.get("price") is not None:
                    try:
                        price = float(payload["price"])
                    except (TypeError, ValueError):
                        price = None
                    if price is not None and price > budget + 1e-6:
                        shock_recover = True
                        recovery_msg_force = not recovery_committed
                        must_message = not recovery_committed
                        if payload.get("product_id") and payload.get("sku"):
                            cart_stale = {
                                "product_id": str(payload["product_id"]),
                                "sku": str(payload["sku"]),
                            }
                            shocked_keys.add(
                                f"{cart_stale['product_id']}::{cart_stale['sku']}"
                            )
            if isinstance(obs, dict):
                inner = (
                    obs.get("observation")
                    if isinstance(obs.get("observation"), dict)
                    else obs
                )
                if isinstance(inner, dict) and inner.get("market_event"):
                    shock_recover = True
                    recovery_msg_force = not recovery_committed
                    must_message = not recovery_committed
                    added = inner.get("added")
                    if isinstance(added, dict):
                        last_stock = added
                        if added.get("product_id") and added.get("sku"):
                            cart_stale = {
                                "product_id": str(added["product_id"]),
                                "sku": str(added["sku"]),
                            }
                            shocked_keys.add(
                                f"{cart_stale['product_id']}::{cart_stale['sku']}"
                            )
                    elif last_stock and last_stock.get("product_id"):
                        cart_stale = {
                            "product_id": str(last_stock.get("product_id")),
                            "sku": str(last_stock.get("sku") or ""),
                        }
                        if cart_stale.get("sku"):
                            shocked_keys.add(
                                f"{cart_stale['product_id']}::{cart_stale['sku']}"
                            )
        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            messages.append({"role": "user", "content": content})
            if _looks_like_shopper_intervention(str(content)):
                low = str(content).lower()
                if ask_first and asked_spec and ("?" not in low):
                    must_message = False
                elif "?" in low or any(
                    (
                        x in low
                        for x in ("which", "still available", "hold on", "before you")
                    )
                ):
                    if not recovery_committed:
                        must_message = True
                    if (not recovery_committed) and any(
                        (
                            x in low
                            for x in (
                                "still available",
                                "out of stock",
                                "hold on",
                                "before you",
                            )
                        )
                    ):
                        recovery_msg_force = True
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
            ranked_now = _best_closed_keys(
                candidates, view_facts_by_pid, exclude=shocked_keys
            )
            if ranked_now:
                bp, bs = ranked_now[0]
                tip = (
                    f"All listed candidates viewed. Next best listed key is {bp}::{bs}. "
                    "inspect_stock → add_to_cart → place_test_order that key."
                )
            else:
                tip = (
                    "All listed candidates viewed. Rank by stated priority, "
                    "inspect_stock the winner, add_to_cart listed sku, place_test_order."
                )
            messages.append({"role": "user", "content": tip})
    return dialogue


__all__ = ["agent_main"]
