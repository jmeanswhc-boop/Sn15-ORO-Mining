
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from os import getenv
from typing import Any, Iterable

from src.agent.proxy_client import ProxyClient

# --------------------------------------------------------------------------- models
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

_proxy = ProxyClient(timeout=120, max_retries=3)


# --------------------------------------------------------------------------- phases
class Phase(str, Enum):
    """Explicit shopping phases (FIT v2 control plane)."""

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
        "Retrieval/HiSun+KAK: value-side search tokens; survey distinct product_ids; "
        "after ~3–4 weak searches view and finish — never 2× thrash."
    ),
    "TF3": (
        "Constraints/madlol+ELON: firm specs NON-NEGOTIABLE after view. Ask-first when "
        "withheld-until-asked. After stockout: ≤2 focused recovery searches then finish."
    ),
    "TF4": (
        "Preference/Lovelystar: among firm-valid, prefer listings with visible evidence "
        "for EACH named soft axis (battery, sweat/water, fit, wireless…). Soft ≠ override firm."
    ),
    "TF5": (
        "Ranking/0828: listed product_id::sku is exhaustive. FIRST cart = current best by "
        "stated priority (event commitment). After shock: re-rank LIVE listed keys only."
    ),
    "TF6": (
        "Recovery/Lovelystar+0828: search preferred model early. Ladder exact → same model "
        "other color/SKU → other models. Declarative message on shock; never ask permission."
    ),
    "TF7": (
        "Justification/0828: only if asked why/reason. Claims = price + in_stock + named "
        "specs + brand only. No extra fields. Prefer HD/FHD/QHD/UHD class when shopper "
        "used that language and evidence supports it. Never claim category."
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
        "Market events usually fire on FIRST add_to_cart. Cart a hard-valid best "
        "candidate first. On market_event: remove_from_cart → declarative message → "
        "recover/re-rank → inspect_stock → cart → place_test_order. Never order the "
        "dead/pre-event sku without a fresh inspect.\n\n"
        f"{body}\n\n"
        "# Finish path\n"
        "ask? → search/filter → view/compare → inspect_stock → add_to_cart → "
        "(handle market_event) → place_test_order. Late budget: stop exploring and order."
    )


_SYSTEM = _compose_system_prompt()


# --------------------------------------------------------------------------- query cues
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
    m = _PREF_MODEL_RE.search(shopper_head(query))
    if not m:
        return None
    token = m.group(1).strip().rstrip(".,;:")
    token = re.split(r"\b(if|specifically|genuine|not|before)\b", token, maxsplit=1)[
        0
    ].strip()
    return token if len(token) >= 3 else None


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
    """FIT-only: soft preference axis labels mentioned in shopper text."""
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


# --------------------------------------------------------------------------- infra
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


def chat_completion(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
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


def env_call(envelope: dict[str, Any]) -> dict[str, Any]:
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


# --------------------------------------------------------------------------- composers
class ClaimComposer:
    """Build minimal faithful place_test_order justification (TF7)."""

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
        if "brand" in self.allowed:
            req.append("brand")
        return req

    def _resolution_value(self, view_facts: dict[str, str]) -> str | None:
        vis = str(view_facts.get("resolution") or "").strip()
        blob = " ".join(
            str(view_facts.get(k) or "")
            for k in ("resolution", "screen_size", "title", "display")
        )
        cls = shopper_resolution_class(self.query)
        if cls and evidence_supports_res_class(blob + " " + vis, cls):
            return cls
        if vis:
            cleaned = re.sub(r"\s*pixels?\s*$", "", vis, flags=re.I).strip()
            if cls and evidence_supports_res_class(cleaned, cls):
                return cls
            return cleaned
        return cls

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
            if field is None or field in seen or field not in required_set:
                continue
            raw_val = claim.get("value", "")
            if field == "resolution":
                mapped = self._resolution_value(view_facts)
                if mapped:
                    raw_val = mapped
            elif field in view_facts and field in ("screen_size", "refresh_rate", "brand"):
                raw_val = view_facts[field]
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

        if "brand" in required_set and "brand" not in seen and view_facts.get("brand"):
            brand = view_facts["brand"]
            if brand.lower() not in ("none", "null", "n/a"):
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

        final: list[dict[str, str]] = []
        seen2: set[str] = set()
        for c in cleaned:
            if (
                c["field"] in required_set
                and c["field"] not in _CLAIM_DENY
                and c.get("value")
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
    """Track listed variant metrics and pick best live key by priority."""

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
    """Rewrite stock/price shock messages into declarative recovery commits."""

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


# --------------------------------------------------------------------------- episode book
@dataclass
class EpisodeBook:
    """Mutable episode control state for the phase machine."""

    query: str
    candidates: list[tuple[str, str]]
    rank_priority: list[str]
    ask_first: bool
    model_hint: str | None
    soft_brand: str | None
    soft_axes: list[str]
    budget: float | None
    claims: ClaimComposer
    ranker: ClosedSetRanker
    shock: ShockHandler
    phase: Phase = Phase.EXPLORE
    must_message: bool = False
    asked_spec: bool = False
    viewed: set[str] = field(default_factory=set)
    searchish: int = 0
    last_stock: dict[str, Any] | None = None
    last_view_facts: dict[str, str] = field(default_factory=dict)
    event_seen: bool = False
    first_cart_done: bool = False
    need_reinspect: bool = False
    last_pick: dict[str, str] = field(default_factory=dict)
    cart_sku: dict[str, str] = field(default_factory=dict)
    phase_notes: list[str] = field(default_factory=list)

    @classmethod
    def from_query(cls, query: str, tools: list[dict[str, Any]]) -> "EpisodeBook":
        allowed = claim_field_allowlist(tools)
        cands = parse_closed_candidates(query)
        prio = parse_rank_priority(query)
        ask = needs_ask_first(query)
        model = preferred_model_token(query)
        book = cls(
            query=query,
            candidates=cands,
            rank_priority=prio,
            ask_first=ask,
            model_hint=model,
            soft_brand=soft_prefer_brand(query),
            soft_axes=extract_soft_axes(query),
            budget=parse_budget_ceiling(query),
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

    def boot_user_message(self) -> str:
        q = self.query
        boot = (
            f"{q}\n\n"
            "Phase plan: ask (if needed) → retrieve/view → inspect_stock → "
            "FIRST cart hard-valid best (event commitment) → on market_event: "
            "remove → declarative message → recover/re-rank → inspect → cart → "
            "place_test_order. "
            "TF7 only if asked: price+in_stock+named specs+brand only."
        )
        if self.ask_first:
            boot += (
                "\n\nPHASE ASK (TF3): First tool call = `message` for the withheld "
                "firm hardware spec."
            )
        if self.soft_brand:
            boot += (
                f"\n\nSOFT BRAND: Prefer `{self.soft_brand}` on first cart when "
                "fully firm-valid."
            )
        if self.soft_axes:
            boot += (
                "\n\nSOFT AXES (TF4): Seek visible evidence for: "
                + ", ".join(self.soft_axes)
                + "."
            )
        if self.budget is not None:
            boot += f"\n\nBUDGET CEILING: stay at or under {self.budget}."
        if self.model_hint:
            boot += (
                f"\n\nPREFERRED MODEL (TF6): `{self.model_hint}`. Search early; "
                "after shock recover same model other color/SKU before other models."
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
                + ", brand when visible. No other claim fields. Use HD/FHD/QHD/4K class "
                "when the shopper used that language and evidence supports it."
            )
        return boot

    def phase_hint(self, *, remaining: int) -> str | None:
        best = self.ranker.best_live()
        if self.must_message or self.phase == Phase.DIALOGUE:
            hint = (
                "REQUIRED this turn: `message` once. On stockout/price jump: STATE you "
                "will take another valid option (same preferred model ladder when relevant). "
                "Do not ask permission."
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
                "Market shock: remove_from_cart if stale. Recover SAME preferred model "
                "other color/variant first"
            )
            if self.model_hint:
                msg += f" — search/view `{self.model_hint}` before other models"
            msg += (
                ", inspect_stock, place_test_order. Max 2 recovery searches; no permission asks."
            )
            return msg
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
    """Reject dead-key carts and non-best first carts on closed sets."""
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


def build_action_args(tool_call: dict[str, Any], book: EpisodeBook) -> dict[str, Any]:
    args = parse_tool_args(tool_call)
    name = tool_call.get("function", {}).get("name") or ""
    if name == "place_test_order":
        return book.claims.sanitize(
            args,
            last_stock=book.last_stock,
            last_view_facts=book.last_view_facts,
        )
    if name == "message":
        return book.shock.rewrite(args)
    return args


# --------------------------------------------------------------------------- agent_main
def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """FIT v2: phase-machine EnvPack loop with race-general commitment gates."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    tools = policy["tools"]
    query = str(policy.get("query") or "")
    book = EpisodeBook.from_query(query, tools)
    max_steps = int(policy["max_steps"])

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": book.boot_user_message()},
    ]
    dialogue: list[dict[str, Any]] = []

    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        hint = book.phase_hint(remaining=remaining)
        if hint:
            messages.append({"role": "user", "content": hint})

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for _attempt in range(4):
            inference = chat_completion(messages=messages, tools=tools)
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
                commit_rej = commit_gate_reject(tool_calls, book)
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
            break
        else:
            if book.candidates and len(book.viewed & book.ranker.pids) < len(
                book.ranker.pids
            ):
                miss = sorted(book.ranker.pids - book.viewed)[0]
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
            elif book.must_message and book.ask_first and not book.asked_spec:
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
        if "message" in names and book.ask_first:
            book.asked_spec = True
        if "add_to_cart" in names:
            book.first_cart_done = True
            book.transition(Phase.COMMIT, "carted")

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
        result = env_call(envelope)
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
            if book.phase == Phase.DIALOGUE:
                book.transition(Phase.EXPLORE if not book.candidates else Phase.CLOSE_VIEW)

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
                dig_facts(obs, book.last_view_facts)
                book.ranker.ingest(obs)
            if name in ("view", "compare") and isinstance(obs, dict):
                facts = extract_view_facts(obs)
                if facts:
                    book.last_view_facts = {**book.last_view_facts, **facts}
            if name == "inspect_stock":
                book.need_reinspect = False
            if name in ("inspect_stock", "add_to_cart", "inspect_cart") and payload:
                book.last_stock = payload
                if stock_is_false(payload):
                    book.need_reinspect = True
                    book.event_seen = True
                    book.transition(Phase.SHOCK, "oos payload")
                    book.ranker.mark_dead(
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
                        book.ranker.mark_dead(
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
                if book.ask_first and book.asked_spec and "?" not in low:
                    book.must_message = False
                else:
                    book.must_message = True
                    book.transition(Phase.DIALOGUE, "shopper")
                if looks_like_market_event(str(content)):
                    book.need_reinspect = True
                    book.event_seen = True
                    book.transition(Phase.SHOCK, "user market cue")
                elif "?" in low or any(
                    x in low
                    for x in ("which", "still available", "hold on", "before you")
                ):
                    book.must_message = True

        if any((call.get("observation") or {}).get("done") for call in result["calls"]):
            break

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
                    "before switching models."
                )
            if book.soft_axes:
                nudge += (
                    f" Prefer candidates with evidence for soft axes {book.soft_axes}."
                )
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
            best = book.ranker.best_live()
            if best:
                msg += f" Current best live: {best}."
            messages.append({"role": "user", "content": msg})

    return dialogue


__all__ = ["agent_main"]
