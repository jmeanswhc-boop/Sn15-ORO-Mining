
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from os import getenv
from typing import Any, Iterable

from src.agent.proxy_client import ProxyClient


_OPENROUTER_MODELS = [
    "deepseek/deepseek-v3.2",
    "moonshotai/kimi-k2.6",
]
_CHUTES_MODELS = [
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "moonshotai/Kimi-K2.6-TEE",
]

_PROBLEM_BUDGET_S = 120.0
_INFER_TIMEOUT_S = 45
_MAX_INFER_TRIES_PER_MODEL = 2
_MAX_TOOL_REPROMPTS = 2
_MAX_ENV_RETRIES = 2
_SURVEY_MIN_LEFT_S = 28.0

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
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|budget(?:\s+is|\s+of)?|"
    r"can spend up to|at most)\s*(\d[\d,]*(?:\.\d+)?)",
    re.I,
)
_PREF_MODEL_RE = re.compile(
    r"prefer(?:\s+the)?\s+"
    r"(?:black|white|green|red|blue|gold|silver|pink|purple|gray|grey|yellow|orange|"
    r"moonlight\s+white)?\s*"
    r"([A-Za-z0-9][A-Za-z0-9.\-]{2,40})",
    re.I,
)
_SOFT_BRAND_RE = re.compile(
    r"(?:ideally[^.]{0,80}?\bprefer(?:ably)?\s+|prefer(?:ably)?\s+)"
    r"(?:the\s+)?([A-Za-z][\w&.'\-]{1,32})",
    re.I,
)
_PRIORITY_AXIS_RE = re.compile(
    r"(?:highest|largest|lowest|best|prefer(?:ably)?|priority(?:\s+is|\s*:)?)\s+"
    r"(refresh(?:\s*rate)?|screen(?:\s*size)?|price|budget|resolution|ram|"
    r"battery|weight|storage)",
    re.I,
)
_SOFT_AXIS_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("battery_life", ("battery", "long lasting", "all-day charge")),
    ("sweat_water_resistance", ("sweat", "water resist", "waterproof", "ipx", "rain")),
    ("sport_secure_fit", ("secure fit", "sport", "workout", "run", "ear hook")),
    ("wireless_freedom", ("wireless", "bluetooth", "no cable", "cordless")),
    ("durability", ("durable", "rugged", "tough", "heavy duty")),
)

_proxy = ProxyClient(timeout=_INFER_TIMEOUT_S, max_retries=1)


class Phase(str, Enum):

    ASK = "ask_first"
    EXPLORE = "explore_retrieve"
    CLOSE_VIEW = "closed_set_view"
    COMMIT = "event_commit"
    DIALOGUE = "shopper_dialogue"
    SHOCK = "market_shock"
    RECOVER = "recover_ladder"
    FINISH = "finish_order"


_TF_PLAYBOOKS: dict[str, str] = {
    "TF1": (
        "Intent/Lovelystar: silently split ≤4 sub-goals. Firm budget/specs beat soft "
        "prefer/ideally brand. Soft brand only on FIRST cart when fully firm-valid. "
        "Literal sizes/RAM/resolution — no near-miss inches."
    ),
    "TF2": (
        "Retrieval/HiSun+KAK: value-side search tokens; when the shopper gives a match "
        "phrase, FIRST cart title must cover those tokens literally. Survey distinct "
        "product_ids; after ~3–4 weak searches view and finish — never 2× thrash. "
        "After stockout: ONE declarative message then recover/view/cart — never message loop."
    ),
    "TF3": (
        "Constraints/madlol+ELON: firm specs NON-NEGOTIABLE after view. Ask-first when "
        "withheld-until-asked. After stockout: NEVER re-cart/order the shocked SKU. "
        "≤2 focused recovery searches → inspect a NEW product → cart → place_test_order."
    ),
    "TF4": (
        "Preference/Lovelystar: SURVEY 2–3 firm-valid candidates before FIRST cart. "
        "Among them, commit to the listing with the MOST visible soft-axis evidence "
        "(battery, sweat/water, fit, wireless…). Soft axes RANK — they do not excuse a "
        "wrong firm listing. Do not first-cart wired 3.5mm when wireless is desired. "
        "After shock: re-rank LIVE survivors the same way."
    ),
    "TF5": (
        "Ranking/0828: listed product_id::sku is exhaustive. FIRST cart = current best by "
        "stated priority (event commitment). After shock: re-rank LIVE listed keys only."
    ),
    "TF6": (
        "Recovery/Lovelystar+0828: search preferred model EARLY. SURVEY ≥2 listings that "
        "literally contain that model token (different colors/SKUs) BEFORE FIRST cart. "
        "FIRST cart MUST be a preferred-model listing (event commitment). After shock: "
        "ONE declarative message → ladder same model other color/SKU → other models. "
        "Never ask permission; never message thrash."
    ),
    "TF7": (
        "Justification/0828: only if asked why/reason. Claims = price + in_stock + named "
        "specs; brand ONLY when a real brand is visible — never claim brand=None/null. "
        "Prefer HD/FHD/QHD/UHD class tokens when the shopper used that language and "
        "title/options literally support it. Never claim category.\n"
        "EVENT COMMITMENT (critical): the FIRST add_to_cart is scored. Survey ≥2 "
        "views, then cart a claimable firm-valid listing with gate-attr evidence. "
        "Do NOT cart a wrong-brand/unclaimable decoy hoping recovery fixes it. "
        "After market_event: remove → ONE declarative message → search same brand+specs → "
        "inspect → cart → place_test_order."
    ),
}


def _compose_system_prompt() -> str:
    body = "\n\n".join(f"## {k}\n{v}" for k, v in _TF_PLAYBOOKS.items())
    return (
        "You are an ORO Bench shopping agent (failure_is_teacher_v2 phase machine).\n"
        "Use ONLY supplied tools. Continue until done=true. Never memorize product IDs "
        "or suite ranks. Obey this episode's shopper text + Task rules only.\n\n"
        "# Matching (ELON-style)\n"
        "- LITERAL: brands, models, IDs, numbers+units, colors — must appear explicitly.\n"
        "- SEMANTIC: materials/usage synonyms OK for soft attrs only.\n"
        "- Title = unordered keyword bag; structured attributes beat SEO prose.\n"
        "- Cart only when every FIRM constraint is PASS (UNKNOWN firm = FAIL).\n\n"
        "# Event commitment (all families)\n"
        "Market events usually fire on FIRST add_to_cart. SURVEY then cart the intended "
        "shockable best (preferred model / best soft-axis / closed-set best). Later gold "
        "does NOT repair a bad first cart. On market_event: remove_from_cart → ONE "
        "declarative message → recover/re-rank → inspect_stock → cart a LIVE non-dead "
        "sku → place_test_order. Never re-order the shocked/dead sku.\n\n"
        f"{body}\n\n"
        "# Finish path\n"
        "ask? → search/filter → view/compare → inspect_stock → add_to_cart → "
        "(handle market_event) → place_test_order. Late budget / wall clock: "
        "stop exploring and place_test_order promptly — open episodes are agent_errors."
    )


_SYSTEM = _compose_system_prompt()


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


def shopper_head(query: str) -> str:
    return query.split("Task rules:", 1)[0].strip()


def parse_closed_candidates(query: str) -> list[tuple[str, str]]:
    head = shopper_head(query)
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


def parse_rank_priority(query: str) -> list[str]:
    head = shopper_head(query)
    found: list[str] = []
    for m in _PRIORITY_AXIS_RE.finditer(head):
        raw = m.group(1).lower().strip()
        if "refresh" in raw:
            axis = "refresh_rate"
        elif "screen" in raw:
            axis = "screen_size"
        elif raw in ("price", "budget"):
            axis = "price"
        else:
            axis = raw.replace(" ", "_")
        if axis not in found:
            found.append(axis)
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


def justification_needed(query: str) -> bool:
    head = shopper_head(query).lower()
    return any(
        n in head
        for n in (
            "why",
            "reason",
            "walk me through",
            "tell me why",
            "justif",
            "good pick",
            "reason i can trust",
        )
    )


def needs_ask_first(query: str) -> bool:
    return bool(_ASK_SPEC_RE.search(shopper_head(query)))


def soft_prefer_brand(query: str) -> str | None:
    m = _SOFT_BRAND_RE.search(shopper_head(query))
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


def preferred_model_token(query: str) -> str | None:
    head = shopper_head(query)
    m = _PREF_MODEL_RE.search(head)
    if not m:
        return None
    token = m.group(1).strip().rstrip(".,;:")
    token = re.split(r"\b(if|specifically|genuine|not|before)\b", token, maxsplit=1)[
        0
    ].strip()
    if len(token) < 3:
        return None
    soft = soft_prefer_brand(query)
    if soft and token.lower() == soft.lower():
        return None
    if not re.search(r"\d", token):
        if token.isalpha():
            return None
    low = head.lower()
    if re.search(rf"\bideally\b.{{0,40}}\bprefer(?:ably)?\s+{re.escape(token)}\b", low):
        if not re.search(r"\d", token):
            return None
    return token


def is_strict_model_hint(hint: str | None) -> bool:
    if not hint:
        return False
    return bool(re.search(r"\d", hint))


_BRAND_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "me",
        "my",
        "in",
        "stock",
        "instock",
        "get",
        "want",
        "find",
        "need",
        "buy",
        "can",
        "you",
        "please",
        "full",
        "hd",
        "fhd",
        "qhd",
        "uhd",
        "4k",
        "720p",
        "1080p",
        "1440p",
        "gaming",
        "inch",
        "inches",
        "monitor",
        "smartphone",
        "phone",
        "tablet",
        "laptop",
        "headphone",
        "earbuds",
        "under",
        "at",
        "or",
        "and",
        "with",
        "for",
        "why",
        "good",
        "pick",
        "trust",
        "walk",
        "through",
        "right",
    }
)


def firm_request_brand(query: str) -> str | None:
    head = shopper_head(query)
    patterns = (
        r"(?:want|need|find|get|buy)\s+(?:me\s+)?(?:an?\s+|in-stock\s+|instock\s+)*"
        r"([A-Z][A-Za-z0-9][A-Za-z0-9.&-]{0,28}(?:\s+[A-Z][A-Za-z0-9.&-]{1,20})?)",
        r"(?:in-stock|instock)\s+([A-Z][A-Za-z0-9][A-Za-z0-9.&-]{0,28})",
    )
    for pat in patterns:
        m = re.search(pat, head, flags=re.I)
        if not m:
            continue
        brand = m.group(1).strip().rstrip(".,;:")

        parts = [p for p in brand.split() if p.lower() not in _BRAND_STOP]
        if not parts:
            continue
        while parts and (
            parts[-1].lower() in _BRAND_STOP
            or re.fullmatch(r"\d+(?:\.\d+)?(?:p|k|hz|in|inch)?", parts[-1], re.I)
        ):
            parts.pop()
        if not parts:
            continue
        out = " ".join(parts)
        if len(out) < 2 or out.lower() in _BRAND_STOP:
            continue
        return out
    return None


def brand_is_nullish(value: Any) -> bool:
    low = str(value or "").strip().lower()
    return low in ("", "none", "null", "n/a", "na", "unknown", "undefined", "-")


def brand_evidence_ok(brand: str | None, facts: dict[str, str] | None) -> bool:
    if not brand:
        return True
    facts = facts or {}
    blob = " ".join(
        str(facts.get(k) or "") for k in ("brand", "title", "source_brand", "name")
    ).lower()
    if not blob.strip():
        return False
    want = brand.lower().strip()
    if want in blob:
        return True

    toks = [t for t in re.split(r"[\s.&-]+", want) if len(t) >= 3]
    return bool(toks) and all(t in blob for t in toks)


def resolution_evidence_strength(query: str, facts: dict[str, str] | None) -> int:
    facts = facts or {}
    blob = " ".join(
        str(facts.get(k) or "")
        for k in ("resolution", "title", "display", "screen_size", "options")
    ).lower()
    cls = shopper_resolution_class(query)
    if not cls:
        return 0
    if not evidence_supports_res_class(blob, cls):
        return 0
    score = 1
    if cls == "fhd" and any(x in blob for x in ("full hd", "fhd")):
        score += 2
    if cls == "fhd" and "1080" in blob:
        score += 1
    if cls == "hd" and re.search(r"\bhd\b", blob) and "full hd" not in blob:
        score += 1
    if cls == "hd" and "720" in blob:
        score += 1
    if cls == "qhd" and any(x in blob for x in ("qhd", "1440")):
        score += 2
    if cls == "uhd" and any(x in blob for x in ("uhd", "4k", "2160")):
        score += 2
    return score


def shopper_screen_inches(query: str) -> float | None:
    head = shopper_head(query)
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-?\s*)?(?:inch|inches|in\b|\")",
        head,
        flags=re.I,
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def inch_values_in_text(text: str) -> list[float]:
    vals: list[float] = []
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(?:\"|''|in(?:ch(?:es)?)?\b)",
        text or "",
        flags=re.I,
    ):
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            continue

    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*-?\s*inch", text or "", flags=re.I):
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            continue

    out: list[float] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def listing_size_ambiguous(text: str, want: float | None) -> bool:
    inches = inch_values_in_text(text)
    if len(inches) < 2:
        return False

    distinct = []
    for v in inches:
        if not any(abs(v - d) < 0.3 for d in distinct):
            distinct.append(v)
    if len(distinct) < 2:
        return False
    if want is None:
        return True

    has_want = any(abs(v - want) < 0.35 for v in distinct)
    has_other = any(abs(v - want) >= 0.35 for v in distinct)
    return has_want and has_other


def retrieval_phrase_tokens(query: str) -> list[str]:
    head = shopper_head(query)
    low = head.lower()
    cue = any(
        s in low
        for s in (
            "matches",
            "match the phrase",
            "matching",
            "that matches",
            "the phrase",
        )
    )
    tokens: list[str] = []
    for qm in re.finditer(r'"([^"]{2,80})"|“([^”]{2,80})”', head):
        phrase = (qm.group(1) or qm.group(2) or "").strip().lower()
        tokens.extend(re.findall(r"[a-z0-9]{2,}", phrase))
    if not tokens and cue:
        m = re.search(
            r"matches?\s+(?:the\s+phrase\s+)?[\"“]?([^\"”\n,.]{2,80})",
            head,
            flags=re.I,
        )
        if m:
            raw = m.group(1).strip().lower()

            raw = re.split(
                r"\b(?:at or under|under|below|in stock|, and buy|category)\b",
                raw,
                maxsplit=1,
            )[0]
            tokens.extend(re.findall(r"[a-z0-9]{2,}", raw))
    if not cue and not tokens:
        return []
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "category",
        "listing",
        "in",
        "stock",
        "buy",
        "one",
        "phrase",
        "find",
        "an",
    }
    out: list[str] = []
    for t in tokens:
        if t in stop or t in out:
            continue
        out.append(t)
    return out[:6]


def title_covers_tokens(text: str, tokens: list[str]) -> bool:
    if not tokens:
        return True
    low = (text or "").lower()
    if not low.strip():
        return False
    return all(t in low for t in tokens)


def soft_axis_hit_count(axes: list[str], facts: dict[str, str] | None) -> int:
    if not axes:
        return 0
    blob = " ".join(str(v) for v in (facts or {}).values()).lower()
    needles = {
        "battery_life": ("battery", "mah", "hour", "all-day"),
        "sweat_water_resistance": (
            "sweat",
            "water",
            "ipx",
            "waterproof",
            "splash",
        ),
        "sport_secure_fit": ("sport", "earhook", "ear hook", "secure", "workout", "run"),
        "wireless_freedom": ("wireless", "bluetooth", "bt ", "cordless"),
        "durability": ("durable", "rugged", "tough"),
    }
    hits = 0
    for axis in axes:
        if any(n in blob for n in needles.get(axis, ())):
            hits += 1
    return hits


def text_has_model_hint(text: str, hint: str | None) -> bool:
    if not hint:
        return True
    return hint.lower() in (text or "").lower()


def named_spec_fields(query: str) -> list[str]:
    head = shopper_head(query).lower()
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
        )
    ) or re.search(r"\bhd\b", head):
        fields.append("resolution")
    if any(
        x in head for x in ("inch", "screen size", "display size", "-inch", " inches")
    ) or re.search(r"\b\d+(?:\.\d+)?-?inch\b", head):
        fields.append("screen_size")
    if "refresh" in head or re.search(r"\b\d+\s*hz\b", head):
        fields.append("refresh_rate")
    return fields


def extract_soft_axes(query: str) -> list[str]:
    head = shopper_head(query).lower()
    hit: list[str] = []
    for axis, needles in _SOFT_AXIS_PATTERNS:
        if any(n in head for n in needles):
            hit.append(axis)
    return hit


def parse_budget_ceiling(query: str) -> float | None:
    m = _BUDGET_RE.search(shopper_head(query))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def shopper_resolution_class(query: str) -> str | None:
    head = shopper_head(query).lower()
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


def evidence_supports_res_class(text: str, cls: str) -> bool:
    low = (text or "").lower().replace("×", "x")
    if cls == "uhd":
        return any(x in low for x in ("4k", "uhd", "2160", "3840"))
    if cls == "qhd":
        return any(x in low for x in ("qhd", "1440", "2560"))
    if cls == "fhd":
        return any(x in low for x in ("fhd", "1080", "1920", "full hd"))
    if cls == "hd":
        return bool(re.search(r"\b720\b|\b1280\b|\bhd\b", low))
    return False


def claim_field_allowlist(tools: list[dict[str, Any]]) -> frozenset[str]:
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


def normalize_claim_field(field: str, allowed: frozenset[str]) -> str | None:
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


def normalize_claim_value(field: str, value: Any) -> str:
    text = str(value).strip()
    if field == "brand":
        return "" if brand_is_nullish(text) else text
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
        low = text.lower().replace("×", "x")
        if low in ("hd", "fhd", "qhd", "uhd", "4k"):
            return "uhd" if low == "4k" else low
        return re.sub(r"\s*pixels?\s*$", "", text, flags=re.I).strip()
    return text


def dig_facts(obj: Any, out: dict[str, str], depth: int = 0) -> None:
    if depth > 6 or obj is None:
        return
    if isinstance(obj, dict):
        key_map = {
            "brand": "brand",
            "price": "price",
            "current_price": "price",
            "in_stock": "in_stock",
            "stock": "in_stock",
            "refresh_rate": "refresh_rate",
            "refresh": "refresh_rate",
            "resolution": "resolution",
            "screen_size": "screen_size",
            "screen size": "screen_size",
            "title": "title",
            "sku": "sku",
            "product_id": "product_id",
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
                dig_facts(v, out, depth + 1)
    elif isinstance(obj, list):
        for item in obj[:20]:
            dig_facts(item, out, depth + 1)


def extract_view_facts(observation: Any) -> dict[str, str]:
    facts: dict[str, str] = {}
    if not isinstance(observation, dict):
        return facts
    node = observation.get("observation")
    if not isinstance(node, dict):
        node = observation
    dig_facts(node, facts)
    if node.get("title"):
        facts.setdefault("title", str(node["title"]).strip())
    raw_facts = node.get("facts")
    if isinstance(raw_facts, dict):
        if raw_facts.get("display") and "resolution" not in facts:
            facts["resolution"] = str(raw_facts["display"]).strip()
            facts.setdefault("display", str(raw_facts["display"]).strip())
        if raw_facts.get("size") and "screen_size" not in facts:
            facts["screen_size"] = str(raw_facts["size"]).strip()
    return facts


def obs_stock_payload(observation: Any) -> dict[str, Any] | None:
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


def stock_is_false(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    val = payload.get("in_stock")
    if val is False:
        return True
    if isinstance(val, str) and val.strip().lower() in ("false", "0", "no"):
        return True
    return False


def tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [str(c.get("function", {}).get("name") or "") for c in tool_calls]


def looks_like_market_event(text: str) -> bool:
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


def looks_like_shopper_intervention(text: str) -> bool:
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


def time_left(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def chat_completion(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    deadline: float,
) -> dict[str, Any] | None:
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


def env_call(envelope: dict[str, Any], *, deadline: float) -> dict[str, Any] | None:
    if time.monotonic() >= deadline:
        return None
    base_id = str(envelope.get("call_id") or "call")
    for attempt in range(_MAX_ENV_RETRIES):
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
        time.sleep(0.2 * (attempt + 1))
    return None


class ClaimComposer:

    def __init__(self, query: str, allowed: frozenset[str]) -> None:
        self.query = query
        self.allowed = allowed
        self.required = self._required()

    def _required(self) -> list[str]:
        req: list[str] = []
        for f in ("price", "in_stock"):
            if f in self.allowed:
                req.append(f)
        for f in named_spec_fields(self.query):
            if f in self.allowed and f not in req and f not in _CLAIM_DENY:
                req.append(f)


        return req

    def _resolution_value(self, view_facts: dict[str, str]) -> str | None:
        vis = str(view_facts.get("resolution") or "").strip()
        blob = " ".join(
            str(view_facts.get(k) or "")
            for k in ("resolution", "screen_size", "title", "display", "options")
        )
        surface = (blob + " " + vis).strip()
        if not surface:
            return None
        cls = shopper_resolution_class(self.query)

        if cls and evidence_supports_res_class(surface, cls):
            return cls
        if vis:
            cleaned = re.sub(r"\s*pixels?\s*$", "", vis, flags=re.I).strip()
            if cls and evidence_supports_res_class(cleaned, cls):
                return cls


            if cls:
                return None
            return cleaned
        return None

    def sanitize(
        self,
        args: dict[str, Any],
        *,
        last_stock: dict[str, Any] | None,
        last_view_facts: dict[str, str] | None,
    ) -> dict[str, Any]:
        out = dict(args)
        if not justification_needed(self.query):
            out.pop("justification", None)
            return out
        view_facts = dict(last_view_facts or {})
        required_set = set(self.required)

        optional_brand = "brand" in self.allowed
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
            field = normalize_claim_field(str(claim.get("field", "")), self.allowed)
            if field is None or field in seen:
                continue
            if field not in required_set and not (field == "brand" and optional_brand):
                continue
            raw_val = claim.get("value", "")
            if field == "resolution":
                mapped = self._resolution_value(view_facts)
                if mapped:
                    raw_val = mapped
                elif shopper_resolution_class(self.query):


                    continue
            elif field in view_facts and field in ("screen_size", "refresh_rate", "brand"):
                raw_val = view_facts[field]
            if field == "brand" and brand_is_nullish(raw_val):
                continue
            value = normalize_claim_value(field, raw_val)
            if not value:
                continue
            cleaned.append({"field": field, "value": value})
            seen.add(field)

        if last_stock:
            if "price" in required_set and last_stock.get("price") is not None:
                cleaned = [c for c in cleaned if c["field"] != "price"]
                cleaned.append(
                    {
                        "field": "price",
                        "value": normalize_claim_value("price", last_stock.get("price")),
                    }
                )
                seen.add("price")
            if "in_stock" in required_set and last_stock.get("in_stock") is not None:
                cleaned = [c for c in cleaned if c["field"] != "in_stock"]
                cleaned.append(
                    {
                        "field": "in_stock",
                        "value": normalize_claim_value(
                            "in_stock", last_stock.get("in_stock")
                        ),
                    }
                )
                seen.add("in_stock")

        if optional_brand and "brand" not in seen and view_facts.get("brand"):
            brand = view_facts["brand"]
            if not brand_is_nullish(brand):
                cleaned.append({"field": "brand", "value": brand})
                seen.add("brand")

        if "resolution" in required_set and "resolution" not in seen:
            mapped = self._resolution_value(view_facts)
            if mapped:
                cleaned.append({"field": "resolution", "value": mapped})
                seen.add("resolution")

        for spec in ("screen_size", "refresh_rate"):
            if spec in required_set and spec not in seen and view_facts.get(spec):
                cleaned.append(
                    {
                        "field": spec,
                        "value": normalize_claim_value(spec, view_facts[spec]),
                    }
                )
                seen.add(spec)

        allow_fields = required_set | ({"brand"} if optional_brand else set())
        final: list[dict[str, str]] = []
        seen2: set[str] = set()
        for c in cleaned:
            if (
                c["field"] in allow_fields
                and c["field"] not in _CLAIM_DENY
                and c.get("value")
                and not (c["field"] == "brand" and brand_is_nullish(c.get("value")))
                and c["field"] not in seen2
            ):
                seen2.add(c["field"])
                final.append(c)
        if final:
            out["justification"] = {"claims": final}
        else:
            out.pop("justification", None)
        return out


class ClosedSetRanker:

    def __init__(
        self,
        candidates: list[tuple[str, str]],
        priority: list[str],
    ) -> None:
        self.candidates = candidates
        self.priority = priority or ["refresh_rate", "screen_size", "price"]
        self.ledger: dict[str, dict[str, Any]] = {}
        self.dead: set[str] = set()

    @property
    def keys(self) -> list[str]:
        return [f"{pid}::{sku}" for pid, sku in self.candidates]

    @property
    def pids(self) -> set[str]:
        return {pid for pid, _ in self.candidates}

    def mark_dead(self, pid: str, sku: str) -> None:
        if pid and sku:
            self.dead.add(f"{pid}::{sku}")

    @staticmethod
    def _hz(text: str) -> float | None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*hz", text or "", flags=re.I)
        return float(m.group(1)) if m else None

    @staticmethod
    def _inches(text: str) -> float | None:
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:inches|inch|in\b|\")",
            text or "",
            flags=re.I,
        )
        if m:
            return float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", text or "", flags=re.I)
        return float(m.group(1)) if m else None

    def ingest(self, observation: Any) -> None:
        if not self.candidates or not isinstance(observation, dict):
            return
        cand_keys = {f"{pid}::{sku}": (pid, sku) for pid, sku in self.candidates}
        node = observation.get("observation")
        if not isinstance(node, dict):
            node = observation

        def one(prod: dict[str, Any]) -> None:
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
                blob = " ".join(
                    [title, fact_blob, options, str(facts.get("display") or "")]
                )
                entry = self.ledger.setdefault(key, {"product_id": pid, "sku": sku})
                price = var.get("price")
                if price is not None:
                    try:
                        entry["price"] = float(price)
                    except (TypeError, ValueError):
                        pass
                if "in_stock" in var:
                    entry["in_stock"] = bool(var.get("in_stock"))
                hz = self._hz(blob)
                if hz is not None:
                    entry["refresh_rate"] = hz
                inches = self._inches(blob) or self._inches(str(facts.get("size") or ""))
                if inches is not None:
                    entry["screen_size"] = inches

        if "product_id" in node or "variants" in node or "title" in node:
            one(node)
        for k in ("products", "items", "results", "compared"):
            block = node.get(k)
            if isinstance(block, list):
                for p in block:
                    if isinstance(p, dict):
                        one(p)

    def best_live(self) -> str | None:
        keys = [k for k in self.keys if k not in self.dead]
        if not keys:
            return None

        def sort_key(k: str) -> tuple:
            e = self.ledger.get(k) or {}
            stock_pen = 1.0 if e.get("in_stock") is False else 0.0
            parts: list[float] = [stock_pen]
            for axis in self.priority:
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
        return keys[0]


class ShockHandler:

    def __init__(self, model_hint: str | None) -> None:
        self.model_hint = model_hint

    def rewrite(self, args: dict[str, Any]) -> dict[str, Any]:
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
        if not asks and not self.model_hint:
            if not any(x in low for x in ("i'm now", "i will", "switching", "selecting")):
                asks = True
        if not asks and not self.model_hint:
            return args
        out = dict(args)
        if self.model_hint:
            text = (
                f"The preferred option changed (stock/price) when I added it to cart. "
                f"I'm now selecting another color/variant of the same model "
                f"({self.model_hint}) that is still in stock and within budget — "
                f"firm requirements stay. I will only change models if no same-model "
                f"variant remains."
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


@dataclass
class EpisodeBook:

    query: str
    candidates: list[tuple[str, str]]
    rank_priority: list[str]
    ask_first: bool
    model_hint: str | None
    soft_brand: str | None
    firm_brand: str | None
    soft_axes: list[str]
    budget: float | None
    phrase_tokens: list[str]
    screen_inches: float | None
    claims: ClaimComposer
    ranker: ClosedSetRanker
    shock: ShockHandler
    phase: Phase = Phase.EXPLORE
    must_message: bool = False
    asked_spec: bool = False
    viewed: set[str] = field(default_factory=set)
    searchish: int = 0
    post_shock_search: int = 0
    last_stock: dict[str, Any] | None = None
    last_view_facts: dict[str, str] = field(default_factory=dict)
    facts_by_pid: dict[str, dict[str, str]] = field(default_factory=dict)
    focus_pid: str | None = None
    model_hint_seen: bool = False
    event_seen: bool = False
    first_cart_done: bool = False
    need_reinspect: bool = False
    last_pick: dict[str, str] = field(default_factory=dict)
    cart_sku: dict[str, str] = field(default_factory=dict)
    phase_notes: list[str] = field(default_factory=list)
    # fusion: commitment + shock hygiene
    model_view_pids: set[str] = field(default_factory=set)
    shock_message_sent: bool = False
    event_target_key: str | None = None
    post_shock_ack: bool = False

    @classmethod
    def from_query(cls, query: str, tools: list[dict[str, Any]]) -> "EpisodeBook":
        allowed = claim_field_allowlist(tools)
        cands = parse_closed_candidates(query)
        prio = parse_rank_priority(query)
        ask = needs_ask_first(query)
        model = preferred_model_token(query)
        firm = firm_request_brand(query) if justification_needed(query) else None
        book = cls(
            query=query,
            candidates=cands,
            rank_priority=prio,
            ask_first=ask,
            model_hint=model,
            soft_brand=soft_prefer_brand(query),
            firm_brand=firm,
            soft_axes=extract_soft_axes(query),
            budget=parse_budget_ceiling(query),
            phrase_tokens=retrieval_phrase_tokens(query),
            screen_inches=shopper_screen_inches(query),
            claims=ClaimComposer(query, allowed),
            ranker=ClosedSetRanker(cands, prio),
            shock=ShockHandler(model),
            phase=Phase.ASK if ask else (Phase.CLOSE_VIEW if cands else Phase.EXPLORE),
            must_message=bool(ask),
        )
        return book

    def note(self, msg: str) -> None:
        self.phase_notes.append(msg)

    def transition(self, new_phase: Phase, why: str = "") -> None:
        if new_phase != self.phase:
            self.note(f"{self.phase.value}→{new_phase.value}" + (f" ({why})" if why else ""))
            self.phase = new_phase

    def remember_product_facts(self, pid: str, facts: dict[str, str]) -> None:
        if not pid:
            return
        clean = {k: v for k, v in facts.items() if v and not brand_is_nullish(v)}
        self.facts_by_pid[pid] = dict(clean)
        self.focus_pid = pid
        self.last_view_facts = dict(clean)
        blob = " ".join(clean.values())
        if self.model_hint and text_has_model_hint(blob, self.model_hint):
            self.model_hint_seen = True
            self.model_view_pids.add(pid)

    def facts_for(self, pid: str | None) -> dict[str, str]:
        if pid and pid in self.facts_by_pid:
            return dict(self.facts_by_pid[pid])
        if pid and self.focus_pid == pid:
            return dict(self.last_view_facts)
        return {}

    def mark_event_target(self, pid: str, sku: str) -> None:
        pid = str(pid or "")
        sku = str(sku or "")
        if pid and sku:
            key = f"{pid}::{sku}"
            self.event_target_key = key
            self.ranker.mark_dead(pid, sku)

    def is_dead_key(self, pid: str, sku: str) -> bool:
        key = f"{str(pid or '')}::{str(sku or '')}"
        if not pid or not sku:
            return False
        if key in self.ranker.dead:
            return True
        if self.event_target_key and key == self.event_target_key:
            return True
        return False

    def best_soft_pid(self) -> str | None:
        """Rank viewed listings by soft-axis evidence (TF4)."""
        if not self.soft_axes or not self.facts_by_pid:
            return None
        scored: list[tuple[int, str]] = []
        for pid, facts in self.facts_by_pid.items():
            hits = soft_axis_hit_count(self.soft_axes, facts)
            blob = " ".join(str(v) for v in facts.values()).lower()
            if "wireless_freedom" in self.soft_axes and any(
                x in blob for x in ("3.5mm", "3.5 mm", "wired earphone", "wired headphone")
            ) and not any(x in blob for x in ("wireless", "bluetooth", "bt5", "bt 5")):
                hits = -1
            scored.append((hits, pid))
        scored.sort(key=lambda t: t[0], reverse=True)
        if not scored or scored[0][0] < 0:
            return None
        return scored[0][1]

    def best_model_pid(self) -> str | None:
        if not self.model_hint:
            return None
        if self.model_view_pids:
            # Prefer most recently focused model view if still in set
            if self.focus_pid in self.model_view_pids:
                return self.focus_pid
            return sorted(self.model_view_pids)[0]
        for pid, facts in self.facts_by_pid.items():
            blob = " ".join(str(v) for v in facts.values())
            if text_has_model_hint(blob, self.model_hint):
                return pid
        return None

    def note_dead(self, pid: str, sku: str) -> None:
        self.ranker.mark_dead(pid, sku)
        key = f"{pid}::{sku}"
        if (
            self.cart_sku.get("product_id") == pid
            and self.cart_sku.get("sku") == sku
        ):
            self.cart_sku = {}
        if (
            self.last_pick.get("product_id") == pid
            and self.last_pick.get("sku") == sku
        ):
            self.last_pick = {}
        if key == self.event_target_key or not self.event_target_key:
            if pid and sku:
                self.event_target_key = key
        _ = key
    def boot_user_message(self) -> str:
        q = self.query
        boot = (
            f"{q}\n\n"
            "Phase plan: ask (if needed) → retrieve/view → inspect_stock → "
            "FIRST cart hard-valid best (event commitment) → on market_event: "
            "remove → declarative message → recover/re-rank → inspect → cart → "
            "place_test_order. "
            "TF7 only if asked: price+in_stock+named specs; brand only if visible "
            "(never brand=None)."
        )
        if self.ask_first:
            boot += (
                "\n\nPHASE ASK (TF3): First tool call = `message` for the withheld "
                "firm hardware spec."
            )
        if self.firm_brand:
            boot += (
                f"\n\nFIRM PRODUCT BRAND (TF7 commitment): `{self.firm_brand}` is part "
                "of the request. FIRST add_to_cart must be a listing whose title/brand "
                "shows that brand AND visible evidence for named specs. Survey ≥2 "
                "views before first cart; never cart a different brand as a decoy."
            )
        if self.phrase_tokens:
            boot += (
                "\n\nRETRIEVAL PHRASE: FIRST cart title should cover tokens "
                f"{self.phrase_tokens} (literal match beats near-miss synonyms)."
            )
        if self.soft_brand:
            boot += (
                f"\n\nSOFT BRAND: Prefer `{self.soft_brand}` on first cart when "
                "fully firm-valid."
            )
        if self.soft_axes:
            boot += (
                "\n\nSOFT AXES (TF4): SURVEY 2–3 candidates then FIRST-cart the one "
                "with the most visible evidence for: "
                + ", ".join(self.soft_axes)
                + ". Soft axes rank among firm-valid; they do not override firm specs."
            )
        if self.budget is not None:
            boot += f"\n\nBUDGET CEILING: stay at or under {self.budget}."
        if self.model_hint and is_strict_model_hint(self.model_hint):
            boot += (
                f"\n\nPREFERRED MODEL (TF6): `{self.model_hint}`. Search early; "
                "SURVEY >=2 listings containing that model token (colors/SKUs) BEFORE "
                "FIRST cart; FIRST cart must keep that model. After shock recover same "
                "model other color/SKU before other models. ONE declarative message only."
            )
        elif self.model_hint:
            # Non-strict leftover — treat as soft cue only (should be rare after filter).
            boot += (
                f"\n\nSOFT MODEL CUE: `{self.model_hint}` is optional preference, not a "
                "hard first-cart model lock."
            )
        if self.candidates:
            listed = ", ".join(f"{p}::{s}" for p, s in self.candidates)
            prio = " → ".join(self.rank_priority) if self.rank_priority else "stated priority"
            boot += (
                f"\n\nCLOSED SET (TF5): {listed}. Rank by [{prio}]. "
                "FIRST add_to_cart = current best listed key; after shock re-rank LIVE listed."
            )
        elif justification_needed(self.query):
            specs = named_spec_fields(self.query)
            boot += (
                "\n\nTF7: justification fields = price, in_stock"
                + (f", {', '.join(specs)}" if specs else "")
                + "; brand only when visible (never None/null). Use HD/FHD/QHD/4K class "
                "when the shopper used that language and evidence supports it. "
                "FIRST cart must be claimable (gate attr visible) — survey ≥2 products "
                "with view/compare before the sacrificial first cart."
            )
        boot += (
            "\n\nSPEED: Prefer short trajectories. After a live inspect_stock on the "
            "chosen sku, place_test_order promptly — do not keep searching."
        )
        return boot

    def phase_hint(self, *, remaining: int) -> str | None:
        best = self.ranker.best_live()
        if self.must_message or self.phase == Phase.DIALOGUE:
            if self.event_seen and self.shock_message_sent:
                # Already acked shock — do not re-demand message loops.
                self.must_message = False
                if self.phase == Phase.DIALOGUE:
                    self.transition(
                        Phase.RECOVER if self.event_seen else Phase.EXPLORE,
                        "post-shock ack",
                    )
            else:
                hint = (
                    "REQUIRED this turn: `message` once. On stockout/price jump: STATE you "
                    "will take another valid option (same preferred model ladder when relevant). "
                    "Do not ask permission. Do not message again after this."
                )
                if self.model_hint:
                    hint += f" Model token: {self.model_hint}."
                return hint
        if self.candidates and len(self.viewed & self.ranker.pids) < len(self.ranker.pids):
            missing = sorted(self.ranker.pids - self.viewed)
            self.transition(Phase.CLOSE_VIEW, "missing views")
            return (
                f"Closed-set views missing: {missing}. view next id (or compare)."
            )
        if self.need_reinspect or self.phase == Phase.SHOCK:
            self.transition(Phase.RECOVER, "shock")
            if self.candidates:
                msg = (
                    "Market shock on closed set: remove_from_cart if needed. "
                    "Re-inspect remaining LIVE listed candidates"
                )
                if best:
                    msg += f" (best live: {best})"
                msg += (
                    ", add_to_cart NEW best listed key, place_test_order. "
                    "No dead sku; no open-catalog search."
                )
                return msg
            msg = (
                "Market shock: remove_from_cart if stale. ONE declarative message if not "
                "yet sent. Recover SAME preferred model other color/variant first"
            )
            if self.firm_brand:
                msg += f" — search `{self.firm_brand}`"
                if shopper_resolution_class(self.query):
                    msg += f" + {shopper_resolution_class(self.query)}"
                msg += " again, then view a different firm-valid SKU"
            elif self.model_hint:
                msg += f" — search/view `{self.model_hint}` before other models"
            soft = self.best_soft_pid()
            if soft and self.soft_axes:
                msg += f" — prefer viewed soft-axis leader `{soft}`"
            msg += (
                ", inspect_stock, place_test_order. Max 2 recovery searches; never "
                "re-cart the shocked/dead sku; no permission asks; no message loops."
            )
            return msg
        # TF6 survey-before-commit (strict model codes with digits only)
        if (
            self.model_hint
            and is_strict_model_hint(self.model_hint)
            and not self.first_cart_done
            and not self.candidates
            and len(self.model_view_pids) < 2
            and remaining > 6
        ):
            tip = (
                f"TF6 survey: search/view more listings containing `{self.model_hint}` "
                f"(have {len(self.model_view_pids)} model views; need >=2) before FIRST cart."
            )
            return tip
        if (
            self.model_hint
            and is_strict_model_hint(self.model_hint)
            and not self.first_cart_done
            and not self.candidates
            and len(self.model_view_pids) >= 1
        ):
            mp = self.best_model_pid()
            tip = (
                f"TF6 commitment: FIRST add_to_cart must keep model `{self.model_hint}`"
            )
            if mp:
                tip += f" (viewed model lead: {mp})"
            tip += ". Do not cart a different-model decoy."
            return tip
        if (
            self.soft_axes
            and not self.first_cart_done
            and not self.candidates
            and len(self.viewed) < 2
            and remaining > 6
        ):
            return (
                "TF4 survey: view/compare at least 2 firm-valid candidates, then FIRST "
                f"cart a strong soft-axis match among them {self.soft_axes}."
            )
        if (
            self.soft_axes
            and not self.first_cart_done
            and not self.candidates
            and len(self.viewed) >= 2
        ):
            sp = self.best_soft_pid()
            tip = (
                "TF4 commitment: prefer a firm-valid listing with strong soft-axis "
                f"evidence {self.soft_axes}"
            )
            if sp:
                tip += f" (hint among viewed: `{sp}` — still cart the shockable best)"
            tip += "."
            return tip
        if (
            justification_needed(self.query)
            and not self.first_cart_done
            and not self.candidates
            and len(self.viewed) < 2
        ):
            tip = (
                "TF7 commitment: view/compare at least 2 firm-valid candidates before "
                "FIRST add_to_cart. Prefer listings with literal gate evidence"
            )
            if self.firm_brand:
                tip += f" and brand `{self.firm_brand}` on title"
            tip += ". Do not cart a wrong-brand decoy."
            return tip
        if (
            justification_needed(self.query)
            and not self.first_cart_done
            and not self.candidates
            and len(self.viewed) >= 2
            and resolution_evidence_strength(self.query, self.last_view_facts) < 1
            and named_spec_fields(self.query)
        ):
            return (
                "TF7: current view lacks clear gate-attr evidence for the shopper's "
                "named specs. View another candidate with literal HD/FHD/size evidence "
                "before first cart."
            )
        if (
            self.candidates
            and len(self.viewed & self.ranker.pids) >= len(self.ranker.pids)
            and not self.first_cart_done
        ):
            self.transition(Phase.COMMIT, "ready to commit")
            prio = " → ".join(self.rank_priority) if self.rank_priority else "stated priority"
            tip = (
                f"All listed viewed. FIRST add_to_cart = best by [{prio}]"
            )
            if best:
                tip += f" (ledger best: {best})"
            tip += ". inspect_stock then cart (event commitment)."
            return tip
        if not self.candidates and self.searchish >= 3 and len(self.viewed) < 2:
            return (
                "TF2/KAK anti-thrash: stop repeating search. Reformulate once OR view "
                "best hits, then inspect_stock and finish."
            )
        if not self.candidates and (
            self.searchish >= 5 or (self.event_seen and self.searchish >= 2)
        ) and len(self.viewed) >= 1:
            self.transition(Phase.FINISH, "search budget")
            tip = (
                "Stop exploring. Among viewed firm-valid options"
            )
            if self.soft_axes:
                tip += f" covering soft axes {self.soft_axes}"
            tip += ", inspect_stock → add_to_cart → place_test_order NOW."
            return tip
        if remaining <= 6:
            self.transition(Phase.FINISH, "late budget")
            return (
                f"Only ~{remaining} steps left. inspect_stock if needed and "
                "place_test_order on an in-stock in-budget sku NOW."
            )
        return None


def closed_set_violation(
    tool_calls: list[dict[str, Any]],
    book: EpisodeBook,
) -> str | None:
    if not book.candidates:
        return None
    cand_pids = book.ranker.pids
    cand_keys = set(book.ranker.keys)
    viewed = book.viewed
    names = tool_names(tool_calls)
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
                    f"view product_id {pid} is outside listed candidates. "
                    f"View only: {sorted(cand_pids)}"
                )
        if name in ("add_to_cart", "inspect_stock", "place_test_order"):
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and pid not in cand_pids:
                return (
                    f"{name} product_id {pid} outside listed {sorted(cand_pids)}"
                )
            if (
                pid
                and sku
                and f"{pid}::{sku}" not in cand_keys
                and name in ("add_to_cart", "place_test_order")
            ):
                return (
                    f"{name} {pid}::{sku} not a listed variant. "
                    f"Use one of: {sorted(cand_keys)}"
                )
    return None


def oos_order_violation(
    tool_calls: list[dict[str, Any]],
    last_stock: dict[str, Any] | None,
) -> str | None:
    if not stock_is_false(last_stock):
        return None
    if "place_test_order" in tool_names(tool_calls):
        return (
            "Rejected: latest stock/cart shows in_stock=false. "
            "remove_from_cart if needed, pick another candidate, inspect_stock, then order."
        )
    return None


def parse_tool_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def commit_gate_reject(tool_calls: list[dict[str, Any]], book: EpisodeBook) -> str | None:
    best = book.ranker.best_live()
    if not book.candidates or not best:
        return None
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        try:
            args = parse_tool_args(call)
        except (ValueError, json.JSONDecodeError, KeyError):
            continue
        key = f"{args.get('product_id')}::{args.get('sku')}"
        if name in ("add_to_cart", "place_test_order") and key in book.ranker.dead:
            return (
                f"Rejected: {key} is dead after market event. "
                f"Use a live listed key such as {best}."
            )
        if (
            name == "add_to_cart"
            and not book.first_cart_done
            and not book.event_seen
            and key != best
            and key in set(book.ranker.keys)
            and (book.ranker.ledger.get(best) or {}).get("refresh_rate") is not None
        ):
            return (
                "Rejected: first cart must be the current best listed key by "
                f"stated priority: {best} (not {key})."
            )
    return None


def dead_key_reject(
    tool_calls: list[dict[str, Any]],
    book: EpisodeBook,
) -> str | None:
    """Always ban cart/order of shocked or dead keys (TF3/TF6)."""
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        if name not in ("add_to_cart", "place_test_order", "inspect_stock"):
            continue
        if name == "inspect_stock" and not book.event_seen:
            continue
        try:
            args = parse_tool_args(call)
        except (ValueError, json.JSONDecodeError, KeyError):
            continue
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        if not pid or not sku:
            continue
        if book.is_dead_key(pid, sku) and name in ("add_to_cart", "place_test_order"):
            alt = ""
            mp = book.best_model_pid()
            sp = book.best_soft_pid()
            if mp and mp != pid:
                alt = f" Prefer viewed model lead `{mp}`."
            elif sp and sp != pid:
                alt = f" Prefer soft-axis lead `{sp}`."
            return (
                f"Rejected: {pid}::{sku} is dead/shocked. Never re-cart or order it."
                f"{alt} Search/view a NEW live firm-valid sku."
            )
    return None


def event_commitment_reject(
    tool_calls: list[dict[str, Any]],
    book: EpisodeBook,
) -> str | None:
    """Survey-then-commit gates. Soft axes rank; preferred model must win first cart."""
    if book.first_cart_done or book.event_seen or book.candidates:
        return None
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        if name != "add_to_cart":
            continue
        try:
            args = parse_tool_args(call)
        except (ValueError, json.JSONDecodeError, KeyError):
            args = {}
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        if book.is_dead_key(pid, sku):
            return (
                f"Rejected: {pid}::{sku} is dead. Cart a live listing instead."
            )
        facts = book.facts_for(pid) or (
            dict(book.last_view_facts) if book.focus_pid == pid else {}
        )

        if not facts and book.focus_pid and book.focus_pid != pid:
            return (
                "Rejected: view/inspect the product you are about to cart first so "
                "commitment evidence is tied to that listing (no cross-product fact bleed)."
            )
        if not facts:
            facts = dict(book.last_view_facts)
        blob = " ".join(str(v) for v in facts.values())

        # TF6: survey preferred-model variants before committing (strict model codes only).
        if book.model_hint and is_strict_model_hint(book.model_hint):
            has_model = text_has_model_hint(blob, book.model_hint)
            if len(book.model_view_pids) < 2 and book.searchish < 5:
                if not has_model:
                    return (
                        "Rejected: SURVEY preferred model first. Search/view listings "
                        f"containing `{book.model_hint}` (>=2 model views) before FIRST "
                        "cart. Do not cart a non-model decoy."
                    )
                if len(book.model_view_pids) < 2:
                    return (
                        "Rejected: TF6 survey incomplete. View another color/SKU of "
                        f"`{book.model_hint}` before FIRST cart (event commitment)."
                    )
            if book.model_hint_seen or book.model_view_pids:
                if not has_model:
                    return (
                        "Rejected: a preferred-model listing was already viewed. FIRST "
                        f"cart must keep model token `{book.model_hint}` "
                        "(intended event commitment)."
                    )

        if book.firm_brand and not brand_evidence_ok(book.firm_brand, facts):
            return (
                "Rejected: FIRST add_to_cart must match firm product brand "
                f"`{book.firm_brand}` on that listing's title/brand. "
                "Wrong-brand decoys break event commitment permanently."
            )
        if book.phrase_tokens and not title_covers_tokens(blob, book.phrase_tokens):
            return (
                "Rejected: FIRST cart title must cover the shopper's match phrase "
                f"tokens {book.phrase_tokens}. Prefer a literal phrase hit before carting."
            )

        # TF4: soft axes are hints only — hard-block wired-vs-wireless.
        # Never force soft-axis leader as first cart (can miss intended event target).
        if book.soft_axes:
            low_blob = blob.lower()
            if "wireless_freedom" in book.soft_axes and any(
                x in low_blob for x in ("3.5mm", "3.5 mm", "wired earphone", "wired headphone")
            ) and not any(x in low_blob for x in ("wireless", "bluetooth", "bt5", "bt 5")):
                return (
                    "Rejected: FIRST cart looks wired (3.5mm) while the shopper wants "
                    "wireless freedom. Prefer a Bluetooth/wireless listing."
                )

        if "resolution" in named_spec_fields(book.query):
            if resolution_evidence_strength(book.query, facts) < 1:
                return (
                    "Rejected: FIRST cart needs visible resolution evidence for the "
                    "shopper's HD/FHD/QHD/UHD request on THIS listing."
                )
        if book.screen_inches is not None:
            if listing_size_ambiguous(blob, book.screen_inches):
                return (
                    "Rejected: FIRST cart title looks like a multi-size mashup. Prefer a "
                    f"clean single ~{book.screen_inches:g}-inch listing for commitment."
                )
            inches = inch_values_in_text(blob)
            if inches and not any(abs(v - book.screen_inches) < 0.35 for v in inches):
                return (
                    "Rejected: FIRST cart should show the requested screen size "
                    f"~{book.screen_inches:g} inches on the listing."
                )
        if justification_needed(book.query) and len(book.viewed) < 2:
            return (
                "Rejected: justification episodes need >=2 product views before FIRST "
                "add_to_cart (event commitment survey)."
            )
    return None


def order_gate_reject(
    tool_calls: list[dict[str, Any]],
    book: EpisodeBook,
) -> str | None:
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        if name != "place_test_order":
            continue
        try:
            args = parse_tool_args(call)
        except (ValueError, json.JSONDecodeError, KeyError):
            args = {}
        pid = str(args.get("product_id") or "")
        sku = str(args.get("sku") or "")
        key = f"{pid}::{sku}"
        if book.is_dead_key(pid, sku) or key in book.ranker.dead:
            return (
                f"Rejected: {key} is dead after market event. "
                "add_to_cart a LIVE alternative before place_test_order."
            )
        in_cart = (
            book.cart_sku.get("product_id") == pid and book.cart_sku.get("sku") == sku
        )
        if not in_cart:
            return (
                "Rejected: place_test_order requires the sku to be in cart "
                "(error otherwise: item not in cart). Call add_to_cart on a LIVE "
                "in-stock sku first, then place_test_order."
            )
        if justification_needed(book.query):
            facts = book.facts_for(pid)
            if not facts:
                facts = dict(book.last_view_facts) if book.focus_pid == pid else {}
            if "resolution" in named_spec_fields(book.query):
                if resolution_evidence_strength(book.query, facts) < 1:
                    return (
                        "Rejected: cannot place_test_order — this listing lacks visible "
                        "resolution evidence for the shopper's request. View/order a "
                        "listing that literally supports the HD/FHD class."
                    )
            if book.firm_brand and not brand_evidence_ok(book.firm_brand, facts):
                return (
                    f"Rejected: order brand must match `{book.firm_brand}` on the listing."
                )
    return None


def build_action_args(tool_call: dict[str, Any], book: EpisodeBook) -> dict[str, Any]:
    args = parse_tool_args(tool_call)
    name = tool_call.get("function", {}).get("name") or ""
    if name == "place_test_order":
        pid = str(args.get("product_id") or "")
        facts = book.facts_for(pid) or dict(book.last_view_facts)
        return book.claims.sanitize(
            args,
            last_stock=book.last_stock,
            last_view_facts=facts,
        )
    if name == "message":
        return book.shock.rewrite(args)
    return args


def parse_key_parts(key: str) -> tuple[str, str] | None:
    if "::" not in key:
        return None
    pid, sku = key.split("::", 1)
    return pid, sku


def forced_tool(
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


def forced_order_call(
    *,
    turn: int,
    pick: dict[str, str],
    book: EpisodeBook,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "product_id": pick["product_id"],
        "sku": pick["sku"],
    }
    if justification_needed(book.query):
        args["justification"] = {"claims": []}
        facts = book.facts_for(pick.get("product_id")) or dict(book.last_view_facts)
        args = book.claims.sanitize(
            args,
            last_stock=book.last_stock,
            last_view_facts=facts,
        )
    return forced_tool("place_test_order", args, turn=turn, tag="order")


def rewrite_cart_to_key(
    tool_calls: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    parts = parse_key_parts(key)
    if not parts:
        return tool_calls
    pid, sku = parts
    out: list[dict[str, Any]] = []
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        if name not in ("add_to_cart", "inspect_stock", "place_test_order"):
            out.append(call)
            continue
        payload: dict[str, Any] = {"product_id": pid, "sku": sku}
        if name == "place_test_order":
            try:
                raw = call["function"].get("arguments", "{}")
                old = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError, KeyError):
                old = {}
            if isinstance(old, dict) and "justification" in old:
                payload["justification"] = old["justification"]
        out.append(
            {
                "id": call.get("id") or f"rewritten-{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(payload)},
            }
        )
    return out


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:

    deadline = time.monotonic() + _PROBLEM_BUDGET_S
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    tools = policy["tools"]
    query = str(policy.get("query") or "")
    book = EpisodeBook.from_query(query, tools)
    max_steps = int(policy["max_steps"])
    ordered = False

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": book.boot_user_message()},
    ]
    dialogue: list[dict[str, Any]] = []

    def resolve_pick() -> dict[str, str]:
        def sku_for(pid: str) -> str:
            facts = book.facts_for(pid)
            sku = str(facts.get("sku") or "")
            if sku:
                return sku
            if book.last_pick.get("product_id") == pid:
                sku = str(book.last_pick.get("sku") or "")
                if sku:
                    return sku
            if book.cart_sku.get("product_id") == pid:
                sku = str(book.cart_sku.get("sku") or "")
                if sku:
                    return sku
            for key, entry in book.ranker.ledger.items():
                if str(entry.get("product_id") or "") == pid:
                    sk = str(entry.get("sku") or "")
                    if sk and not book.is_dead_key(pid, sk):
                        return sk
            return ""

        candidates: list[dict[str, str]] = []
        # Prefer model / soft-axis survivors after shock (TF4/TF6 ladder).
        if book.event_seen:
            for pid in filter(None, [book.best_model_pid(), book.best_soft_pid()]):
                sku = sku_for(pid)
                if sku and not book.is_dead_key(pid, sku):
                    candidates.append({"product_id": pid, "sku": sku})
        candidates.extend(
            [
                dict(book.cart_sku or {}),
                dict(book.last_pick or {}),
            ]
        )
        best = book.ranker.best_live()
        if best:
            parts = parse_key_parts(best)
            if parts:
                candidates.append({"product_id": parts[0], "sku": parts[1]})
        for pick in candidates:
            pid = str(pick.get("product_id") or "")
            sku = str(pick.get("sku") or "")
            if not pid or not sku:
                continue
            if book.is_dead_key(pid, sku):
                continue
            return {"product_id": pid, "sku": sku}
        return {}

    def force_finish(turn: int, reason: str) -> bool:
        nonlocal ordered
        pick = resolve_pick()
        if not pick.get("product_id") or not pick.get("sku"):
            return False
        if book.is_dead_key(pick["product_id"], pick["sku"]):
            return False
        book.transition(Phase.FINISH, reason)

        in_cart = (
            book.cart_sku.get("product_id") == pick["product_id"]
            and book.cart_sku.get("sku") == pick["sku"]
        )
        calls: list[dict[str, Any]] = []

        if (
            book.need_reinspect
            or stock_is_false(book.last_stock)
            or book.last_stock is None
            or not in_cart
        ):
            calls.append(
                forced_tool(
                    "inspect_stock",
                    {"product_id": pick["product_id"], "sku": pick["sku"]},
                    turn=turn,
                    tag="fin-ins",
                )
            )


        if not in_cart:
            calls.append(
                forced_tool(
                    "add_to_cart",
                    {"product_id": pick["product_id"], "sku": pick["sku"]},
                    turn=turn,
                    tag="fin-cart",
                )
            )
        calls.append(forced_order_call(turn=turn, pick=pick, book=book))


        if max_calls <= 1:
            if not in_cart:
                tool_calls = [c for c in calls if c["function"]["name"] == "add_to_cart"][:1]
                if not tool_calls:
                    tool_calls = calls[:1]
            elif book.need_reinspect or stock_is_false(book.last_stock) or book.last_stock is None:
                tool_calls = [c for c in calls if c["function"]["name"] == "inspect_stock"][:1] or calls[:1]
            else:
                tool_calls = [c for c in calls if c["function"]["name"] == "place_test_order"][:1]
        else:
            tool_calls = calls[:max_calls]

        assistant_content = f"Forcing finish ({reason})."
        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        group_id = f"{problem_id}-turn-{turn}-force"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{i}",
                    "action": {
                        "name": c["function"]["name"],
                        "args": build_action_args(c, book),
                    },
                }
                for i, c in enumerate(tool_calls, start=1)
            ],
        }
        result = env_call(envelope, deadline=deadline)
        if result is None:
            return False
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
                "fit_phase": book.phase.value,
            }
        )
        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            obs = call_result.get("observation")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(obs),
                }
            )
            name = tool_call.get("function", {}).get("name") or ""
            payload = obs_stock_payload(obs if isinstance(obs, dict) else {})
            if isinstance(obs, dict):
                try:
                    call_args = parse_tool_args(tool_call)
                except (ValueError, json.JSONDecodeError, KeyError):
                    call_args = {}
                pid = str(call_args.get("product_id") or "")
                facts = extract_view_facts(obs)
                if pid and facts:
                    book.remember_product_facts(pid, {**(book.facts_for(pid)), **facts})
            if name == "inspect_stock":
                book.need_reinspect = False
                if payload:
                    book.last_stock = payload
            if name == "add_to_cart":
                book.cart_sku = {
                    "product_id": pick["product_id"],
                    "sku": pick["sku"],
                }
                book.last_pick = dict(book.cart_sku)
                book.first_cart_done = True

            if name == "place_test_order":

                if not (
                    isinstance(obs, dict)
                    and obs.get("error")
                ):
                    ordered = True
            if name in ("inspect_stock", "add_to_cart", "place_test_order"):
                try:
                    args = parse_tool_args(tool_call)
                except (ValueError, json.JSONDecodeError, KeyError):
                    args = {}
                if args.get("product_id") and args.get("sku"):
                    book.last_pick = {
                        "product_id": str(args["product_id"]),
                        "sku": str(args["sku"]),
                    }
            if isinstance(obs, dict):
                inner = (
                    obs.get("observation")
                    if isinstance(obs.get("observation"), dict)
                    else obs
                )
                if isinstance(inner, dict) and inner.get("market_event"):
                    book.need_reinspect = True
                    book.event_seen = True
                    book.cart_sku = {}
                    ev = inner["market_event"]
                    tgt = ev.get("target") if isinstance(ev, dict) else None
                    if isinstance(tgt, dict):
                        book.note_dead(
                            str(tgt.get("product_id") or ""),
                            str(tgt.get("sku") or ""),
                        )
                if payload and stock_is_false(payload):
                    book.need_reinspect = True
                    book.event_seen = True
                    book.note_dead(
                        str(payload.get("product_id") or ""),
                        str(payload.get("sku") or ""),
                    )
        done = any(
            (call.get("observation") or {}).get("done") for call in result["calls"]
        )


        if (
            not done
            and book.cart_sku
            and f"{book.cart_sku.get('product_id')}::{book.cart_sku.get('sku')}"
            not in book.ranker.dead
            and time_left(deadline) > 8
        ):

            order_only = [
                forced_order_call(turn=turn, pick=dict(book.cart_sku), book=book)
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Forcing order after re-cart ({reason}).",
                    "tool_calls": order_only,
                }
            )
            gid2 = f"{problem_id}-turn-{turn}-force-order"
            env2 = {
                **binding,
                "call_id": gid2,
                "idempotency_key": gid2,
                "turn": turn,
                "calls": [
                    {
                        "call_id": f"{gid2}-1",
                        "action": {
                            "name": "place_test_order",
                            "args": build_action_args(order_only[0], book),
                        },
                    }
                ],
            }
            res2 = env_call(env2, deadline=deadline)
            if res2 is not None:
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": f"Forcing order after re-cart ({reason}).",
                        "tool_calls": order_only,
                        "environment_result": res2,
                        "fit_phase": book.phase.value,
                    }
                )
                for tool_call, call_result in zip(
                    order_only, res2["calls"], strict=True
                ):
                    obs = call_result.get("observation")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(obs),
                        }
                    )
                    if not (isinstance(obs, dict) and obs.get("error")):
                        ordered = True
                done = any(
                    (call.get("observation") or {}).get("done")
                    for call in res2["calls"]
                )
        return done

    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        left = time_left(deadline)
        best = book.ranker.best_live()


        if left < 12 or (remaining <= 2 and resolve_pick()):
            # Phase-aware: don't abort survey into a decoy cart while time remains.
            surveying = (
                not book.first_cart_done
                and not book.candidates
                and left >= _SURVEY_MIN_LEFT_S
                and (
                    (
                        book.model_hint
                        and is_strict_model_hint(book.model_hint)
                        and len(book.model_view_pids) < 2
                    )
                    or (book.soft_axes and len(book.viewed) < 2)
                    or (justification_needed(book.query) and len(book.viewed) < 2)
                )
            )
            if not surveying:
                if force_finish(turn, f"budget left={left:.1f}s steps={remaining}"):
                    break
                if ordered or left < 5:
                    break


        if (
            book.event_seen
            and not book.must_message
            and book.shock_message_sent
            and resolve_pick()
            and (
                book.post_shock_search >= 3
                or remaining <= 6
                or left < 20
            )
        ):
            if force_finish(turn, f"post-shock finish searches={book.post_shock_search}"):
                break


        if (
            not book.must_message
            and book.first_cart_done
            and book.last_pick
            and book.last_stock
            and not stock_is_false(book.last_stock)
            and not book.need_reinspect
            and not book.is_dead_key(
                str(book.last_pick.get("product_id") or ""),
                str(book.last_pick.get("sku") or ""),
            )
            and (left < 25 or remaining <= 5 or (book.event_seen and book.searchish >= 1))
        ):
            book.transition(Phase.FINISH, "post-cart finish nudge")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "FINISH NOW: call place_test_order on the current in-stock "
                        f"sku {book.last_pick.get('product_id')}::{book.last_pick.get('sku')}."
                    ),
                }
            )

        hint = book.phase_hint(remaining=remaining)
        if hint and left >= 15:
            messages.append({"role": "user", "content": hint})

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        inference_failed = False
        for _attempt in range(_MAX_TOOL_REPROMPTS):
            if time_left(deadline) < 8:
                inference_failed = True
                break
            inference = chat_completion(messages=messages, tools=tools, deadline=deadline)
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
            if book.must_message and "message" not in tool_names(tool_calls):
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
            if not book.must_message:
                violation = closed_set_violation(tool_calls, book)
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
                oos = oos_order_violation(tool_calls, book.last_stock)
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
                    book.need_reinspect = True
                    book.transition(Phase.SHOCK, "oos order")
                    continue
                names_try = tool_names(tool_calls)
                if (
                    book.need_reinspect
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

                if book.candidates and best:
                    rewritten = False
                    for call in tool_calls:
                        name = call.get("function", {}).get("name") or ""
                        try:
                            args = parse_tool_args(call)
                        except (ValueError, json.JSONDecodeError, KeyError):
                            continue
                        key = f"{args.get('product_id')}::{args.get('sku')}"
                        if name in ("add_to_cart", "place_test_order") and key in book.ranker.dead:
                            tool_calls = rewrite_cart_to_key(tool_calls, best)
                            rewritten = True
                            break
                        if (
                            name == "add_to_cart"
                            and not book.first_cart_done
                            and not book.event_seen
                            and key != best
                            and key in set(book.ranker.keys)
                            and (book.ranker.ledger.get(best) or {}).get("refresh_rate")
                            is not None
                        ):
                            tool_calls = rewrite_cart_to_key(tool_calls, best)
                            rewritten = True
                            break
                    if rewritten:
                        assistant_content = (
                            (assistant_content or "")
                            + f" [rewrote cart/order to closed-set best {best}]"
                        ).strip()
                        break
                commit_rej = commit_gate_reject(tool_calls, book)
                if commit_rej and book.candidates and best:
                    tool_calls = rewrite_cart_to_key(tool_calls, best)
                    assistant_content = (
                        (assistant_content or "")
                        + f" [rewrote after commit gate to {best}]"
                    ).strip()
                    break
                if commit_rej:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": commit_rej},
                        ]
                    )
                    continue
                dead_rej = dead_key_reject(tool_calls, book)
                if dead_rej:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": dead_rej},
                        ]
                    )
                    continue
                cart_rej = event_commitment_reject(tool_calls, book)
                if cart_rej:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": cart_rej},
                        ]
                    )
                    continue
                order_rej = order_gate_reject(tool_calls, book)
                if order_rej:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": order_rej},
                        ]
                    )
                    continue
            break
        else:
            inference_failed = True

        if inference_failed or not tool_calls:
            if book.candidates and len(book.viewed & book.ranker.pids) < len(
                book.ranker.pids
            ):
                miss = sorted(book.ranker.pids - book.viewed)[0]
                tool_calls = [
                    forced_tool(
                        "view",
                        {"product_id": miss},
                        turn=turn,
                        tag="view",
                    )
                ]
                assistant_content = f"Viewing closed-set candidate {miss}."
            elif book.must_message and book.ask_first and not book.asked_spec:
                tool_calls = [
                    forced_tool(
                        "message",
                        {
                            "content": (
                                "What is the firm hardware requirement "
                                "(RAM, screen size, resolution, or refresh rate)?"
                            )
                        },
                        turn=turn,
                        tag="ask",
                    )
                ]
                assistant_content = "Asking for the withheld firm spec."
            elif resolve_pick():
                if force_finish(turn, "inference/tool exhausted"):
                    break
                break
            else:

                break

        if book.must_message:
            msg_only = [
                c
                for c in tool_calls
                if c.get("function", {}).get("name") == "message"
            ]
            if msg_only:
                tool_calls = msg_only[:1]

        names = tool_names(tool_calls)
        if any(n in ("search", "filter") for n in names):
            book.searchish += 1
            if book.event_seen:
                book.post_shock_search += 1
        if "message" in names and book.ask_first:
            book.asked_spec = True
        if "add_to_cart" in names:
            book.first_cart_done = True
            book.transition(Phase.COMMIT, "carted")
        if "place_test_order" in names:
            ordered = True
            book.transition(Phase.FINISH, "ordered")

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
            args = build_action_args(call, book)
            name = call.get("function", {}).get("name") or ""
            if name in ("inspect_stock", "add_to_cart", "place_test_order"):
                if args.get("product_id") and args.get("sku"):
                    book.last_pick = {
                        "product_id": str(args["product_id"]),
                        "sku": str(args["sku"]),
                    }
            if name == "add_to_cart":
                book.cart_sku = dict(book.last_pick)
            if name == "remove_from_cart":
                book.cart_sku = {}
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
        result = env_call(envelope, deadline=deadline)
        if result is None:
            if resolve_pick() and force_finish(turn, "env timeout"):
                break
            break
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
                "fit_phase": book.phase.value,
            }
        )

        for call in tool_calls:
            if (call.get("function", {}).get("name") or "") == "view":
                try:
                    args = parse_tool_args(call)
                except (ValueError, json.JSONDecodeError, KeyError):
                    args = {}
                if args.get("product_id"):
                    book.viewed.add(str(args["product_id"]))

        if "message" in names:
            book.must_message = False
            if book.event_seen:
                book.shock_message_sent = True
                book.post_shock_ack = True
                book.transition(Phase.RECOVER, "shock message once")
            elif book.phase == Phase.DIALOGUE:
                book.transition(
                    Phase.EXPLORE if not book.candidates else Phase.CLOSE_VIEW
                )

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            obs = call_result.get("observation")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(obs),
                }
            )
            payload = obs_stock_payload(obs if isinstance(obs, dict) else {})
            if payload is None and isinstance(obs, dict):
                payload = obs_stock_payload({"observation": obs})
            name = tool_call.get("function", {}).get("name") or ""
            if isinstance(obs, dict):
                book.ranker.ingest(obs)
            if name in ("view", "compare") and isinstance(obs, dict):

                try:
                    call_args = parse_tool_args(tool_call)
                except (ValueError, json.JSONDecodeError, KeyError):
                    call_args = {}
                pid = str(call_args.get("product_id") or "")
                facts = extract_view_facts(obs)
                if not pid and facts:

                    dig: dict[str, str] = {}
                    dig_facts(obs, dig)
                    pid = str(dig.get("product_id") or "")
                if facts:
                    if pid:
                        book.remember_product_facts(pid, facts)
                    else:

                        book.last_view_facts = dict(facts)
                        book.focus_pid = None
            if name == "inspect_stock":
                book.need_reinspect = False
                try:
                    call_args = parse_tool_args(tool_call)
                except (ValueError, json.JSONDecodeError, KeyError):
                    call_args = {}
                pid = str(call_args.get("product_id") or "")
                if pid and isinstance(obs, dict):
                    extra = extract_view_facts(obs)
                    if extra:
                        merged = {**(book.facts_for(pid)), **extra}
                        book.remember_product_facts(pid, merged)
            if name in ("inspect_stock", "add_to_cart", "inspect_cart") and payload:
                book.last_stock = payload
                if payload.get("product_id") and payload.get("sku"):
                    book.last_pick = {
                        "product_id": str(payload["product_id"]),
                        "sku": str(payload["sku"]),
                    }

                    pid = str(payload["product_id"])
                    if pid in book.facts_by_pid:
                        book.focus_pid = pid
                        book.last_view_facts = dict(book.facts_by_pid[pid])
                if stock_is_false(payload):
                    book.need_reinspect = True
                    book.event_seen = True
                    book.transition(Phase.SHOCK, "oos payload")
                    book.mark_event_target(
                        str(payload.get("product_id") or ""),
                        str(payload.get("sku") or ""),
                    )
                    book.note_dead(
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
                    book.need_reinspect = True
                    book.event_seen = True
                    book.transition(Phase.SHOCK, "market_event")
                    ev = inner["market_event"]
                    tgt = ev.get("target") if isinstance(ev, dict) else None
                    if isinstance(tgt, dict):
                        book.mark_event_target(
                            str(tgt.get("product_id") or ""),
                            str(tgt.get("sku") or ""),
                        )
                        book.note_dead(
                            str(tgt.get("product_id") or ""),
                            str(tgt.get("sku") or ""),
                        )
                    added = inner.get("added")
                    if isinstance(added, dict):
                        book.last_stock = added

        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            messages.append({"role": "user", "content": content})
            if looks_like_shopper_intervention(str(content)):
                low = str(content).lower()
                marketish = looks_like_market_event(str(content)) or any(
                    x in low
                    for x in ("still available", "more expensive", "got more expensive")
                )
                if marketish:
                    book.need_reinspect = True
                    book.event_seen = True
                    book.transition(Phase.SHOCK, "user market cue")
                    # Exactly one declarative ack after shock — never message-loop.
                    if not book.shock_message_sent:
                        book.must_message = True
                    else:
                        book.must_message = False
                elif book.ask_first and book.asked_spec and "?" not in low:
                    book.must_message = False
                elif book.shock_message_sent and book.event_seen:
                    book.must_message = False
                else:
                    book.must_message = True
                    book.transition(Phase.DIALOGUE, "shopper")
                    if "?" in low or any(
                        x in low for x in ("which", "hold on", "before you")
                    ):
                        book.must_message = True

        if any((call.get("observation") or {}).get("done") for call in result["calls"]):
            break


        if time_left(deadline) < 20 or remaining <= 4:
            continue

        if not book.must_message and not book.candidates and not book.need_reinspect:
            nudge = (
                "Episode not done. Prefer inspect_stock → add_to_cart → "
                "place_test_order on an in-stock firm-valid sku. "
                "Omit justification unless the shopper asked why."
            )
            if book.soft_brand and book.viewed:
                nudge += (
                    f" Prefer firm-valid `{book.soft_brand}` for first cart when available."
                )
            if book.model_hint and book.event_seen:
                nudge += (
                    f" After shock, search `{book.model_hint}` same-model variants "
                    "before switching models. Never re-cart the shocked sku."
                )
            if book.soft_axes:
                lead = book.best_soft_pid()
                nudge += (
                    f" Prefer candidates with evidence for soft axes {book.soft_axes}."
                )
                if lead and not book.first_cart_done:
                    nudge += f" Soft-axis leader among viewed: `{lead}`."
            messages.append({"role": "user", "content": nudge})
        elif (
            not book.must_message
            and book.candidates
            and len(book.viewed & book.ranker.pids) >= len(book.ranker.pids)
            and book.first_cart_done
        ):
            prio = (
                " → ".join(book.rank_priority)
                if book.rank_priority
                else "stated priority"
            )
            msg = (
                f"Closed-set continue: rank live listed keys by [{prio}], "
                "inspect_stock winner, add_to_cart listed sku, place_test_order."
            )
            if best:
                msg += f" Current best live: {best}."
            messages.append({"role": "user", "content": msg})

    return dialogue


__all__ = ["agent_main"]
