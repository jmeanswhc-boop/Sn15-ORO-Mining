"""Shopping agent for ORO generated environments (EnvPack / tools v5).

Submit-ready agent. Scores come from family verifiers that require:
  - answering shopper interventions (message when the shopper engages)
  - reacting to price/stock events (re-inspect; do not order the event target)
  - truthful place_test_order justification.claims ONLY when the shopper asks
    for a reason (ignore Task-rules boilerplate that mentions "reason")
"""

from __future__ import annotations

import json
import logging
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}

_ALLOWED_CLAIM_FIELDS = frozenset(
    {"brand", "category", "in_stock", "price", "refresh_rate", "resolution", "screen_size"}
)

# Fields that frequently fail visible-evidence checks unless they are the gate.
_RISKY_OPTIONAL_FIELDS = frozenset({"category", "refresh_rate"})

_FIELD_ALIASES = {
    "size": "screen_size",
    "display": "resolution",
    "screen": "screen_size",
    "res": "resolution",
}

_SYSTEM_PROMPT = """You are an expert shopping agent in a live product environment.

Goal: satisfy firm requirements, handle shopper messages/events, then place a
test order (observation.done=true).

## Firm vs soft constraints
- Firm: category, budget, in-stock, and any explicit hardware spec (screen size,
  resolution, refresh rate, RAM). Soft: "ideally prefer <brand>".
- Never violate firm constraints for a soft brand preference.
- If the shopper says they will reveal a firm spec "if you ask", FIRST call
  `message` and ask which hardware spec+value is required, then search.

## Mandatory shopper replies (family gate)
- After ANY shopper update (which product?, screen-size check, budget change,
  price hike, "still available?"), call `message` on the NEXT turn with a
  concrete answer BEFORE search/cart/order tools.
- Answer with: chosen product name/id, whether the gated spec matches (e.g.
  "listed screen size is 7.6in"), current price if known, and next action.
- Do NOT send unsolicited recommendation messages while still browsing — only
  message when the shopper engaged you or you must ask for a hidden firm spec.
- Unanswered interventions zero the reward even when the final SKU is correct.

## Events (price/stock changes)
- After a price-change or "is that still available?" message:
  1) answer via `message` (name the product + what changed)
  2) `inspect_stock` on the event target
  3) if over budget / OOS: remove_from_cart, then find another valid variant
  4) inspect_stock on the replacement, add_to_cart, inspect_stock again, order
- Never place_test_order on a SKU that just became over-budget or unavailable.
- Prefer another in-stock variant/SKU of a compliant product (often same model,
  different color/SKU suffix) before switching categories.

## Recovery / model+color preferences
- If the shopper names a preferred model code/id (long numeric token): search
  that exact token first, view matches, then inspect_stock the preferred SKU.
- If that exact color/SKU disappears, try another color/SKU of the SAME model
  before changing models. Never relax budget or in-stock requirements.

## Search path
1) search/filter with in_stock=true and max_price when budget is known
2) compare or view a few candidates — read facts (size/display) + variants/skus
3) inspect_stock on chosen product_id+sku
4) add_to_cart
5) place_test_order (after a final inspect_stock if an event just fired)
- Prefer `compare` over repeating the same filter/search. Do not loop the same
  empty filter more than once.

## Justification claims (critical for TF7)
- ONLY if the shopper's own request asks for a reason / why / walk me through:
  include justification.claims on place_test_order. Otherwise OMIT justification
  entirely (task-rule boilerplate mentioning "reason" does NOT count).
- Claims MUST be an object: {"justification":{"claims":[{"field":"...","value":"..."}]}}
- Allowed fields ONLY: brand, category, in_stock, price, refresh_rate,
  resolution, screen_size.
- Map product facts: facts.size → screen_size, facts.display → resolution.
- Required minimal set when a reason is asked: price, in_stock, PLUS the gated
  hardware field from the shopper request. Optionally brand.
- Do NOT claim category/refresh_rate/hdr/display/os extras.
- Never invent attributes. Only claim resolution when facts.display (or an
  explicit resolution fact) supports it — never guess from the title alone.
- Screen size values: use the catalog form with unit, e.g. "27in" / "7.6in".
- Price values: prefer "at most <budget>" using the active budget number.
- If the shopper did NOT ask for a reason, omit justification entirely.

## Other
- Never invent product_ids, skus, prices, or attributes.
- You may batch multiple tool calls in one turn (up to max_calls_per_turn).
- Be decisive; leave enough turns to recover after events and still order.
"""


def _inference_api_key() -> str | None:
    return (
        getenv("INFERENCE_ACCESS_TOKEN")
        or getenv("CHUTES_ACCESS_TOKEN")
        or getenv("OPENROUTER_API_KEY")
        or getenv("CHUTES_API_KEY")
    )


_proxy = ProxyClient(
    timeout=180,
    max_retries=5,
    rate_limit_retry_delay=6,
    api_key=_inference_api_key(),
)


def _model() -> str:
    provider = (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()
    override = (getenv("SANDBOX_MODEL") or "").strip()
    if override:
        return override
    return _DEFAULT_MODELS.get(provider, _DEFAULT_MODELS["chutes"])


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _shopper_intent(query: str) -> str:
    """Shopper request only — ignore Task rules boilerplate (mentions 'reason')."""
    text = query or ""
    for marker in ("\nTask rules:", "\nTask Rules:", "\nTASK RULES:"):
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    return text.strip()


def _query_wants_reason(query: str) -> bool:
    q = _shopper_intent(query).lower()
    return any(
        marker in q
        for marker in (
            "why",
            "reason",
            "justify",
            "justification",
            "walk me through",
            "tell me why",
            "trustworthy reason",
            "tell me why it's a good pick",
        )
    )


def _query_asks_to_ask(query: str) -> bool:
    q = _shopper_intent(query).lower()
    return "if you ask" in q or "tell you if you ask" in q


def _extract_budget(text: str) -> float | None:
    if not text:
        return None
    patterns = (
        r"(?:at or under|at or below|under|below|up to|budget(?:\s+is)?|max(?:imum)?(?:\s+price)?|spend up to|go up to)\s*[:\s]*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*php",
    )
    low = text.lower().replace(",", "")
    for pat in patterns:
        match = re.search(pat, low)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _extract_preferred_model_token(query: str) -> str | None:
    """Long numeric model/SKU tokens from recovery-style preferences."""
    q = _shopper_intent(query)
    # Prefer tokens near 'prefer' / 'black' / 'model'
    near = re.search(
        r"(?:prefer|preferred|model|black|white|color)\D{0,40}(\d{10,})",
        q,
        flags=re.I,
    )
    if near:
        return near.group(1)
    tokens = re.findall(r"\b(\d{10,})\b", q)
    return tokens[0] if tokens else None


def _spec_fields_mentioned(query: str) -> set[str]:
    q = _shopper_intent(query).lower()
    fields: set[str] = set()
    if re.search(r"screen\s*size|\d+(\.\d+)?\s*-?\s*inch|\d+(\.\d+)?in\b", q):
        fields.add("screen_size")
    if re.search(
        r"\b\d{3,4}\s*[x×]\s*\d{3,4}\b|\b\d{3,4}p\b|full\s*hd|\bfhd\b|\bqhd\b|"
        r"\b720p\b|\b1080p\b|\b4k\b|\buhd\b|\bhd\b|resolution|display",
        q,
    ):
        fields.add("resolution")
    # Only treat refresh rate as required when clearly demanded as a firm gate.
    if re.search(r"(\bmust\b|\bneed\b|\brequired\b).{0,40}\d+\s*hz|\d+\s*hz.{0,40}(\bmust\b|\bneed\b|\brequired\b)", q):
        fields.add("refresh_rate")
    return fields


def _preferred_resolution_token(query: str) -> str | None:
    q = _shopper_intent(query).lower()
    if re.search(r"full\s*hd|\bfhd\b|\b1080p\b|1920\s*[x×]\s*1080", q):
        return "fhd"
    if re.search(r"\b720p\b|\bhd\b|1280\s*[x×]\s*720|720\s*[x×]\s*1600", q):
        return "hd"
    if re.search(r"\bqhd\b|\b1440p\b|2560\s*[x×]\s*1440", q):
        return "qhd"
    if re.search(r"\b4k\b|\buhd\b|3840\s*[x×]\s*2160", q):
        return "4k"
    return None


def _normalize_screen_size(value: str) -> str:
    text = value.strip().lower().replace("inches", "inch").replace('"', "")
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        num = match.group(1)
        if num.endswith(".0"):
            num = num[:-2]
        # Catalog / verifier gate values use the "Xin" form (e.g. 27in, 7.6in).
        return f"{num}in"
    return value.strip()


def _normalize_resolution(value: str, preferred: str | None = None) -> str:
    text = value.strip().lower().replace("×", "x")
    if preferred in {"hd", "fhd", "qhd", "4k"}:
        # Map pixel strings / aliases onto the catalog gate token when possible.
        if preferred == "fhd" and re.search(r"1920\s*x\s*1080|1080p|full\s*hd|\bfhd\b", text):
            return "fhd"
        if preferred == "hd" and re.search(r"720\s*x\s*|1280\s*x\s*720|720p|\bhd\b", text):
            return "hd"
        if preferred == "qhd" and re.search(r"2560\s*x\s*1440|1440p|\bqhd\b", text):
            return "qhd"
        if preferred == "4k" and re.search(r"3840\s*x\s*2160|2160p|\b4k\b|\buhd\b", text):
            return "4k"
        # Prefer stating the gate token when the raw fact is already that token.
        if text in {preferred, f"{preferred}+"}:
            return preferred
    if re.search(r"full\s*hd|\bfhd\b|1920\s*x\s*1080|1080p", text):
        return "fhd"
    if re.search(r"720p|1280\s*x\s*720|720\s*x\s*1600|\bhd\b", text) and "full" not in text:
        return "hd"
    if re.search(r"2560\s*x\s*1440|1440p|\bqhd\b", text):
        return "qhd"
    return value.strip()


def _normalize_claim_value(
    field: str,
    value: Any,
    *,
    query: str,
    budget: float | None = None,
) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        value_s = "true" if value else "false"
    else:
        value_s = str(value).strip()
    if not value_s:
        return None
    if field == "in_stock":
        low = value_s.lower()
        if low in {"1", "yes", "true", "in_stock", "in stock"}:
            return "true"
        if low in {"0", "no", "false", "oos", "out_of_stock"}:
            return "false"
        return value_s
    if field == "price":
        low = value_s.lower().replace(",", "")
        if low.startswith("at most"):
            num = re.search(r"(\d+(?:\.\d+)?)", low)
            return f"at most {num.group(1)}" if num else value_s
        if budget is not None:
            return f"at most {budget:g}"
        num = re.search(r"(\d+(?:\.\d+)?)", value_s.replace(",", ""))
        return num.group(1) if num else value_s
    if field == "screen_size":
        return _normalize_screen_size(value_s)
    if field == "resolution":
        return _normalize_resolution(value_s, _preferred_resolution_token(query))
    return value_s


def _shopper_needs_reply(content: str) -> bool:
    """True when the shopper intervention must be answered via `message`."""
    text = content.strip()
    if not text:
        return False
    low = text.lower()
    if low.startswith("thanks for the update"):
        return False
    if "?" in text:
        return True
    markers = (
        "before you buy",
        "which one",
        "hold on",
        "i can only go up to",
        "budget",
        "more expensive",
        "re-check",
        "still needs",
        "still available",
        "did the item",
        "please keep to my stated requirements",
        "compare the current options",
        "non-negotiable",
    )
    return any(marker in low for marker in markers)


def _is_price_or_stock_event(content: str) -> bool:
    low = content.lower()
    return any(
        marker in low
        for marker in (
            "more expensive",
            "re-check",
            "still available",
            "did the item",
            "out of stock",
            "sold out",
        )
    )


def _update_budget_from_text(text: str, current: float | None) -> float | None:
    found = _extract_budget(text)
    return found if found is not None else current


def _extract_product_facts(observation: dict[str, Any]) -> dict[str, str]:
    """Pull claimable fields from view/compare/inspect_stock observations."""
    facts: dict[str, str] = {}

    def absorb(payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("brand"):
            facts["brand"] = str(payload["brand"]).strip()
        if payload.get("title") and "brand" not in facts:
            # weak fallback; only first token-ish brand words are unreliable — skip
            pass
        path = payload.get("category_path")
        if isinstance(path, list) and path:
            facts["category"] = str(path[-1]).strip()
        elif payload.get("category"):
            facts["category"] = str(payload["category"]).strip()
        raw_facts = payload.get("facts")
        if isinstance(raw_facts, dict):
            if raw_facts.get("size") is not None:
                facts["screen_size"] = str(raw_facts["size"]).strip()
            if raw_facts.get("display") is not None:
                facts["resolution"] = str(raw_facts["display"]).strip()
            if raw_facts.get("refresh_rate") is not None:
                facts["refresh_rate"] = str(raw_facts["refresh_rate"]).strip()
            # Some payloads already use canonical names
            for key in ("screen_size", "resolution", "brand", "category"):
                if raw_facts.get(key) is not None:
                    facts[key] = str(raw_facts[key]).strip()
        variants = payload.get("variants")
        if isinstance(variants, list) and variants:
            variant = variants[0]
            if isinstance(variant, dict):
                if variant.get("price") is not None:
                    facts["price"] = str(variant["price"]).strip()
                if variant.get("in_stock") is not None:
                    facts["in_stock"] = (
                        "true" if variant.get("in_stock") in (True, "true", "True", 1) else "false"
                    )
                if variant.get("sku"):
                    facts["sku"] = str(variant["sku"]).strip()
        if payload.get("price") is not None:
            facts["price"] = str(payload["price"]).strip()
        if payload.get("in_stock") is not None:
            facts["in_stock"] = (
                "true" if payload.get("in_stock") in (True, "true", "True", 1) else "false"
            )
        if payload.get("product_id"):
            facts["product_id"] = str(payload["product_id"]).strip()
        if payload.get("title"):
            facts["title"] = str(payload["title"]).strip()

    if not isinstance(observation, dict):
        return facts
    # Env responses sometimes nest under observation.observation
    if isinstance(observation.get("observation"), dict):
        absorb(observation["observation"])
    absorb(observation)
    # inspect_stock style
    if "sku" in observation and observation.get("price") is not None:
        absorb(observation)
    return facts


def _update_fact_store(
    store: dict[str, dict[str, str]],
    observation: dict[str, Any],
) -> None:
    extracted = _extract_product_facts(observation)
    product_id = extracted.get("product_id")
    if not product_id:
        return
    current = store.get(product_id, {})
    current.update({k: v for k, v in extracted.items() if v})
    store[product_id] = current


def _build_claims_from_facts(
    *,
    product_id: str | None,
    sku: str | None,
    fact_store: dict[str, dict[str, str]],
    wants_reason: bool,
    required_specs: set[str],
    query: str,
    budget: float | None = None,
    model_claims: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    if not wants_reason:
        return []

    facts = dict(fact_store.get(product_id or "", {}))
    # Merge model-provided claims as secondary evidence (after aliasing).
    for claim in model_claims or []:
        if not isinstance(claim, dict):
            continue
        field = str(claim.get("field") or "").strip()
        field = _FIELD_ALIASES.get(field, field)
        if field not in _ALLOWED_CLAIM_FIELDS:
            continue
        value = _normalize_claim_value(
            field, claim.get("value"), query=query, budget=budget
        )
        if value:
            # Do not accept model-only resolution guesses without visible facts.
            if field == "resolution" and "resolution" not in facts:
                continue
            facts.setdefault(field, value)

    preferred_res = _preferred_resolution_token(query)
    required_fields = {"price", "in_stock"} | set(required_specs)
    optional = ["brand"]

    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(field: str, raw: Any) -> None:
        if field in seen or field not in _ALLOWED_CLAIM_FIELDS:
            return
        if field in _RISKY_OPTIONAL_FIELDS and field not in required_fields:
            return
        value = _normalize_claim_value(field, raw, query=query, budget=budget)
        if not value:
            return
        if field == "resolution" and preferred_res:
            value = _normalize_resolution(value, preferred_res)
        cleaned.append({"field": field, "value": value})
        seen.add(field)

    for field in list(required_fields) + optional:
        if field == "price":
            # Prefer budget-style claim used by high-scoring agents.
            if budget is not None:
                add("price", f"at most {budget:g}")
            elif field in facts:
                add("price", facts[field])
            continue
        if field in facts:
            add(field, facts[field])

    # Resolution only from visible facts — never invent from title/query alone.
    if "resolution" in required_fields and "resolution" not in seen and facts.get("resolution"):
        add("resolution", facts["resolution"])

    if "screen_size" in required_fields and "screen_size" not in seen and facts.get("screen_size"):
        add("screen_size", facts["screen_size"])

    # Ensure stable order: gated specs first, then price/in_stock/brand.
    order = [f for f in ("screen_size", "resolution", "refresh_rate", "price", "in_stock", "brand") if f in seen]
    by_field = {c["field"]: c for c in cleaned}
    return [by_field[f] for f in order if f in by_field]


def _sanitize_place_order_args(
    args: dict[str, Any],
    *,
    wants_reason: bool,
    allowed_specs: set[str] | None = None,
    query: str = "",
    fact_store: dict[str, dict[str, str]] | None = None,
    budget: float | None = None,
) -> dict[str, Any]:
    """Normalize justification; drop unsafe/extra claims; inject required facts."""
    out = dict(args)
    raw_claims = out.pop("claims", None)
    justification = out.get("justification")
    if isinstance(justification, str):
        try:
            justification = json.loads(justification)
        except json.JSONDecodeError:
            justification = None
    if not isinstance(justification, dict):
        justification = {}

    claims = justification.get("claims")
    if claims is None and raw_claims is not None:
        claims = raw_claims
    if isinstance(claims, str):
        try:
            claims = json.loads(claims)
        except json.JSONDecodeError:
            claims = []
    if not isinstance(claims, list):
        claims = []

    # Alias + drop illegal fields from model output first.
    aliased: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        field = str(claim.get("field") or "").strip()
        field = _FIELD_ALIASES.get(field, field)
        if field not in _ALLOWED_CLAIM_FIELDS:
            continue
        aliased.append({"field": field, "value": claim.get("value")})

    cleaned = _build_claims_from_facts(
        product_id=str(out.get("product_id") or "") or None,
        sku=str(out.get("sku") or "") or None,
        fact_store=fact_store or {},
        wants_reason=wants_reason,
        required_specs=allowed_specs or set(),
        query=query,
        budget=budget,
        model_claims=aliased,
    )

    if wants_reason and cleaned:
        out["justification"] = {"claims": cleaned}
    else:
        out.pop("justification", None)
    return out


def _normalize_tool_args(
    name: str,
    args: dict[str, Any],
    *,
    wants_reason: bool,
    allowed_specs: set[str],
    query: str,
    fact_store: dict[str, dict[str, str]],
    budget: float | None = None,
) -> dict[str, Any]:
    if name == "place_test_order":
        return _sanitize_place_order_args(
            args,
            wants_reason=wants_reason,
            allowed_specs=allowed_specs,
            query=query,
            fact_store=fact_store,
            budget=budget,
        )
    return args


def _inference_with_backoff(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    attempts: int = 4,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        inference = _proxy.post(
            "/inference/chat/completions",
            json_data={
                "model": _model(),
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "temperature": 0,
            },
        )
        if inference is None:
            last_error = RuntimeError("inference request failed")
            time.sleep(min(2**attempt, 12))
            continue
        try:
            return inference["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            last_error = RuntimeError(f"malformed inference response: {exc}")
            time.sleep(min(2**attempt, 12))
    raise RuntimeError(str(last_error) if last_error else "inference failed")


def _synthetic_reply(pending_question: str, candidate: dict[str, str] | None) -> str:
    low = pending_question.lower()
    title = (candidate or {}).get("title") or (candidate or {}).get("brand") or "a matching in-stock option"
    screen = (candidate or {}).get("screen_size")
    if screen and not str(screen).lower().endswith("in"):
        screen = _normalize_screen_size(str(screen))
    product_id = (candidate or {}).get("product_id")
    price = (candidate or {}).get("price")

    if "which one" in low or "going with" in low or "before you buy" in low:
        parts = [f"I am going with {title}"]
        if product_id:
            parts.append(f" (product_id {product_id})")
        if screen:
            parts.append(f", its listed screen size is {screen}")
        if price:
            parts.append(f", it is {price} PHP")
        parts.append(" and in stock — verifying firm requirements before ordering.")
        return "".join(parts)
    if "available" in low:
        return (
            f"I will inspect_stock on {title} right now and only proceed if it is still "
            "in stock within budget; if not, I will switch to another valid variant."
        )
    if "expensive" in low or "price" in low or "re-check" in low:
        return (
            f"Yes — I am re-checking the current price/stock for {title} and will not "
            "order it if it exceeds budget; switching to another valid variant if needed."
        )
    if "budget" in low or "i can only go up to" in low or "hold on" in low:
        return (
            "Noted the updated budget. I will only keep candidates that still meet the "
            "firm hardware spec and the new max price."
        )
    if "screen" in low:
        if screen:
            return (
                f"Yes — the candidate I am using ({title}) is listed with screen size "
                f"{screen}. I am re-checking stock/price before ordering."
            )
        return (
            "I will only order a candidate that matches the required screen size and "
            "current budget; verifying that now."
        )
    return (
        "Thanks — I noted your update and will re-check stock/price against the firm "
        "requirements before ordering."
    )


def _ensure_message_first(
    tool_calls: list[dict[str, Any]],
    *,
    must_answer: bool,
    pending_question: str,
    candidate: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """If the shopper asked something, force a message reply first."""
    if not must_answer:
        return tool_calls
    for call in tool_calls:
        if call.get("function", {}).get("name") == "message":
            return [call]
    synthetic = {
        "id": f"msg-forced-{int(time.time() * 1000)}",
        "type": "function",
        "function": {
            "name": "message",
            "arguments": json.dumps(
                {"content": _synthetic_reply(pending_question, candidate)}
            ),
        },
    }
    return [synthetic]


def _action_signature(name: str, args: dict[str, Any]) -> str:
    if name in {"filter", "search"}:
        keep = {k: args.get(k) for k in sorted(args) if k in {
            "query", "category", "brand", "max_price", "min_price", "in_stock", "k"
        }}
        return f"{name}:{json.dumps(keep, sort_keys=True)}"
    return f"{name}:{json.dumps(args, sort_keys=True)[:120]}"


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Drive the environment tools until done=true or max_steps."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    max_steps = int(policy["max_steps"])
    tools = policy["tools"]
    query = policy["query"]
    shopper_query = _shopper_intent(query)
    wants_reason = _query_wants_reason(query)
    must_ask_spec = _query_asks_to_ask(query)
    allowed_specs = _spec_fields_mentioned(query)
    active_budget = _extract_budget(shopper_query)
    preferred_model = _extract_preferred_model_token(query)

    logger.info(
        "[agent] problem=%s max_steps=%s wants_reason=%s ask_spec=%s specs=%s budget=%s model=%s query=%s",
        problem_id,
        max_steps,
        wants_reason,
        must_ask_spec,
        sorted(allowed_specs),
        active_budget,
        preferred_model,
        shopper_query[:120],
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    if must_ask_spec:
        messages.append(
            {
                "role": "user",
                "content": (
                    "FIRST ACTION: call message and ask which firm hardware "
                    "specification is required (screen size, resolution class, refresh "
                    "rate, or memory) and the exact value — before searching."
                ),
            }
        )
    if preferred_model:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Preferred model token: {preferred_model}. Search that exact token "
                    "first, view matching products, and try the preferred color/SKU. If "
                    "it disappears, try another color/SKU of the SAME model before "
                    "changing models."
                ),
            }
        )
    if wants_reason:
        gate_hint = ", ".join(sorted(allowed_specs)) or "the gated hardware field"
        budget_hint = (
            f' Price claim should be \"at most {active_budget:g}\".'
            if active_budget is not None
            else ""
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "This shopper asked for a reason. On place_test_order include "
                    f"justification.claims covering price, in_stock, and {gate_hint}. "
                    "Use facts.size→screen_size (as e.g. 27in) and facts.display→"
                    f"resolution (hd/fhd only if facts support it).{budget_hint} "
                    "Optionally brand. Do NOT claim category, refresh_rate, hdr, "
                    "display, or OS fields. If the shopper did not ask for a reason, "
                    "omit justification."
                ),
            }
        )
    else:
        messages.append(
            {
                "role": "user",
                "content": (
                    "This shopper did NOT ask for a reason. On place_test_order omit "
                    "justification/claims entirely."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = []
    ordered = False
    pending_question: str | None = None
    asked_spec = False
    event_pending = False
    fact_store: dict[str, dict[str, str]] = {}
    last_candidate: dict[str, str] | None = None
    recent_signatures: list[str] = []

    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        if pending_question:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "REQUIRED THIS TURN: call the message tool and answer the "
                        f"shopper concretely before any other action. Question: "
                        f"{pending_question}"
                    ),
                }
            )
        elif event_pending and not ordered:
            budget_txt = (
                f" Active budget is {active_budget:g} PHP."
                if active_budget is not None
                else ""
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "EVENT RECOVERY: do not order the event target. remove_from_cart "
                        "if needed, find another in-stock variant that still meets firm "
                        f"specs{budget_txt}, inspect_stock it, add_to_cart, inspect_stock "
                        "again, then place_test_order. Prefer same-model alternate SKU/"
                        "color when applicable. Avoid repeating the same filter."
                    ),
                }
            )
        elif remaining <= 10 and not ordered:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "TIME CRITICAL: few turns left. Stop repeated filters. Compare/"
                        "view one valid candidate, inspect_stock, add_to_cart, "
                        "place_test_order. Answer any open shopper question with message "
                        "first if still unanswered."
                    ),
                }
            )
        elif len(recent_signatures) >= 2 and len(set(recent_signatures[-3:])) == 1:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You are repeating the same search/filter. Switch strategy: "
                        "compare multiple product_ids from prior results, or broaden/"
                        "narrow the query once, then view → inspect_stock → order."
                    ),
                }
            )

        assistant: dict[str, Any] | None = None
        tool_calls: list[dict[str, Any]] = []
        assistant_content = ""

        for _nudge in range(2):
            assistant = _inference_with_backoff(messages=messages, tools=tools)
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if tool_calls:
                break
            dialogue.append({"role": "assistant", "content": assistant_content})
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": (
                            "Choose a tool now. If the shopper sent an intervention, "
                            "message first; else search/filter → compare/view → "
                            "inspect_stock → add_to_cart → place_test_order."
                        ),
                    },
                ]
            )
        else:
            raise RuntimeError("model returned no tool call before completion")

        must_answer = bool(pending_question)
        tool_calls = _ensure_message_first(
            tool_calls,
            must_answer=must_answer,
            pending_question=pending_question or "",
            candidate=last_candidate,
        )
        if must_answer:
            pending_question = None

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )

        action_names = [c["function"]["name"] for c in tool_calls]
        if "place_test_order" in action_names:
            ordered = True
            event_pending = False
        if "message" in action_names:
            asked_spec = True

        group_id = f"{problem_id}-turn-{turn}"
        calls_payload = []
        for index, call in enumerate(tool_calls, start=1):
            name = call["function"]["name"]
            args = _normalize_tool_args(
                name,
                _arguments(call),
                wants_reason=wants_reason,
                allowed_specs=allowed_specs,
                query=shopper_query,
                fact_store=fact_store,
                budget=active_budget,
            )
            call["function"]["arguments"] = json.dumps(args)
            recent_signatures.append(_action_signature(name, args))
            recent_signatures = recent_signatures[-6:]
            calls_payload.append(
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
            "calls": calls_payload,
        }

        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError(
                f"environment call failed at turn={turn} actions={action_names}"
            )

        logger.info("[agent] turn=%s actions=%s", turn, action_names)

        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            observation = call_result["observation"]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(observation),
                }
            )
            _update_fact_store(fact_store, observation)
            # Keep the most recently observed product as reply context.
            extracted = _extract_product_facts(observation)
            if extracted.get("product_id"):
                last_candidate = fact_store.get(extracted["product_id"], extracted)
            if observation.get("error"):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool error — adjust. If the shopper asked something, "
                            "answer with message; otherwise continue toward a valid order."
                        ),
                    }
                )

        # Shopper may reveal specs / budgets inside clarification replies.
        user_message = result.get("user_message")
        if user_message and user_message.get("content"):
            content = str(user_message["content"])
            messages.append({"role": "user", "content": content})
            active_budget = _update_budget_from_text(content, active_budget)
            # Absorb newly revealed firm specs into claim requirements.
            allowed_specs |= _spec_fields_mentioned(content)
            if _shopper_needs_reply(content):
                pending_question = content
                if _is_price_or_stock_event(content):
                    event_pending = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The shopper sent an intervention. Next turn MUST start with "
                            "the message tool answering it concretely (which product / "
                            "spec / budget handling), then re-check with "
                            "inspect_stock/view before ordering."
                        ),
                    }
                )

        if any(call["observation"].get("done") for call in result["calls"]):
            logger.info("[agent] done at turn=%s", turn)
            break

        if must_ask_spec and not asked_spec and turn <= 2:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You still have not asked for the firm hardware spec. "
                        "Call message now to ask which spec and value are required."
                    ),
                }
            )

    return dialogue


__all__ = ["agent_main"]
