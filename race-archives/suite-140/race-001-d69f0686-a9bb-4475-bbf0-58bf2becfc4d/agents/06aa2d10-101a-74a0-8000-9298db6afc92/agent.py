

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_proxy = ProxyClient(timeout=120, max_retries=3)

_CLAIM_FIELDS = frozenset(
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

_FILLER = frozenset(
    {
        "the",
        "a",
        "an",
        "for",
        "with",
        "from",
        "that",
        "this",
        "i",
        "me",
        "my",
        "looking",
        "show",
        "find",
        "want",
        "need",
        "get",
        "buy",
        "also",
        "and",
        "in",
        "is",
        "it",
        "am",
        "im",
        "of",
        "to",
        "or",
        "on",
        "at",
        "by",
        "be",
        "can",
        "has",
        "have",
        "will",
        "would",
        "should",
        "please",
        "item",
        "items",
        "one",
        "up",
        "under",
        "below",
        "above",
        "than",
        "more",
        "less",
        "over",
        "php",
        "pesos",
        "price",
        "budget",
        "spend",
        "ideally",
        "prefer",
        "preferred",
        "stock",
        "stocked",
        "available",
        "but",
        "firm",
        "requirements",
        "matter",
        "nice",
        "haves",
        "soft",
        "there",
        "are",
        "was",
        "were",
        "been",
        "being",
        "does",
        "did",
        "doing",
        "into",
        "about",
        "then",
        "them",
        "they",
        "their",
        "you",
        "your",
        "we",
        "our",
        "if",
        "when",
        "while",
        "because",
        "so",
        "just",
        "only",
        "very",
        "really",
        "too",
        "also",
        "still",
        "already",
        "yet",
        "not",
        "no",
        "yes",
        "ok",
        "okay",
        "task",
        "rules",
        "must",
        "always",
        "never",
        "every",
        "each",
        "any",
        "all",
        "some",
        "such",
        "same",
        "other",
        "another",
        "both",
        "either",
        "neither",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "until",
        "without",
        "within",
        "along",
        "across",
        "behind",
        "beyond",
        "plus",
        "per",
        "via",
        "vs",
        "etc",
        "eg",
        "ie",
        "inch",
        "inches",
    }
)

_SYSTEM_POLICY = """You are a shopping agent in a generated environment. Use ONLY the supplied tools.

Decision rules:
1. Separate FIRM requirements from SOFT preferences in the shopper query and Task rules.
2. Firm requirements always win (category, hardware/spec musts, budget, in-stock, explicit musts).
3. Soft preferences (preferred brand/color/model) apply only among fully firm-compliant items.
4. Typical flow: search and/or filter → view/compare promising candidates → inspect_stock →
   add_to_cart → read any shopper user_message → place_test_order.
5. Always verify CURRENT price and stock with inspect_stock before ordering. Prices/stock can change.
6. Check category_path / raw facts from view/compare. A title alone is not enough; reject
   accessories and wrong device classes when a real device/category is required.
7. After cart actions, read shopper messages. Apply updates; answer clarifications with message.
8. If asked to rank/compare given candidates, compare them and follow the stated ranking criteria.
9. If an action fails or a pick goes out of budget/stock, recover: remove bad cart lines and switch
   to another firm-compliant option (brand may change when firm rules require it).
10. Finish by placing a valid in-stock order within budget. Do not wait for permission once verified.
11. Survey several distinct candidates before committing when the task asks for recall/coverage.
12. Reformulate search queries when results are weak (drop fillers, try synonyms, use filter).

Justification / claims:
- When place_test_order needs a reason, include structured justification.claims only.
- Claims MUST be truthful and grounded in view/compare/inspect_stock for the FINAL product.
- Allowed claim fields only: category, brand, screen_size, resolution, refresh_rate, price, in_stock.
- field/value only. No summary prose. Do not invent unobserved specs.
- price/in_stock must match the latest inspect_stock (or equivalent current-state) observation.

Continue until an observation reports done=true."""

_DECOMPOSE_PROMPT = """Extract shopping constraints from the shopper goal (ignore Task rules boilerplate).
Return ONLY JSON with this shape:
{
  "keywords": "<short search query without budget/currency>",
  "category": "<category if stated, else null>",
  "brand_soft": "<preferred brand if soft preference, else null>",
  "brand_firm": "<required brand if firm, else null>",
  "max_price": <number or null>,
  "min_price": <number or null>,
  "spec": {"field": "<screen_size|resolution|refresh_rate|ram|other|null>", "value": "<raw value or null>"},
  "must_in_stock": true,
  "needs_ranking": false,
  "needs_justification": false,
  "given_candidates": ["product_id::sku", ...]
}
Use null/[] when unknown. Do not invent products."""


def _model() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(
        provider, _DEFAULT_MODELS["chutes"]
    )


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _norm(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text


def _shopper_goal(query: str) -> str:
    """Public goal text before Task rules (keeps prompts focused)."""
    parts = re.split(r"\n\s*Task rules:\s*", query, maxsplit=1, flags=re.IGNORECASE)
    return parts[0].strip()


def _parse_json_content(content: str) -> dict[str, Any] | None:
    cleaned = re.sub(
        r"<think(?:ing)?>.*?</think(?:ing)?>",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"```json?\s*|```\s*", "", cleaned).strip()
    try:
        out = json.loads(cleaned)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            out = json.loads(cleaned[start : end + 1])
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            return None
    return None


def _heuristic_parse(goal: str) -> dict[str, Any]:
    """Cheap constraint extraction inspired by prior shopping agents (no suite lists)."""
    lower = goal.lower()
    plan: dict[str, Any] = {
        "keywords": "",
        "category": None,
        "brand_soft": None,
        "brand_firm": None,
        "max_price": None,
        "min_price": None,
        "spec": {"field": None, "value": None},
        "must_in_stock": "in stock" in lower or "in-stock" in lower,
        "needs_ranking": any(
            w in lower for w in ("rank", "highest", "compare only", "among ties")
        ),
        "needs_justification": any(
            w in lower for w in ("walk me through", "why", "reason", "justify")
        ),
        "given_candidates": re.findall(r"\b\d{6,}::[\w.-]+\b", goal),
    }

    m = re.search(
        r"(?:at or under|at or below|up to|under|below|<=|max(?:imum)?)\s*"
        r"([\d,]+(?:\.\d+)?)\s*(?:php|pesos)?",
        lower,
    )
    if m:
        plan["max_price"] = float(m.group(1).replace(",", ""))

    m = re.search(r"(?:between|from)\s*([\d,]+(?:\.\d+)?)\s*(?:and|to|-)\s*([\d,]+(?:\.\d+)?)", lower)
    if m:
        plan["min_price"] = float(m.group(1).replace(",", ""))
        plan["max_price"] = float(m.group(2).replace(",", ""))

    m = re.search(
        r"(?:prefer|preferably|ideally[^.]{0,40}prefer(?:s|red)?)\s+([A-Za-z][A-Za-z0-9 &.-]{1,40})",
        goal,
        flags=re.IGNORECASE,
    )
    if m:
        brand = m.group(1).strip(" .,")
        brand = re.split(r"\b(?:but|and|,)\b", brand, maxsplit=1)[0].strip()
        if brand and brand.lower() not in _FILLER:
            plan["brand_soft"] = brand

    m = re.search(
        r"\b([A-Za-z][A-Za-z0-9.-]{1,30})\s+(?:smartphone|phone|monitor|headphone|laptop)\b",
        goal,
        flags=re.IGNORECASE,
    )
    if m and "prefer" not in lower:
        token = m.group(1)
        if token.lower() not in _FILLER and token[0].isupper():
            plan["brand_firm"] = token

    for field, pattern in (
        ("screen_size", r"(\d+(?:\.\d+)?)\s*-?\s*inch"),
        ("refresh_rate", r"(\d+)\s*hz"),
        ("resolution", r"\b(\d+p|4k|8k|1080|720|1440)\b"),
        ("ram", r"(\d+)\s*gb\s*ram|\bram\s*(\d+)\s*gb"),
    ):
        m = re.search(pattern, lower)
        if m:
            value = next((g for g in m.groups() if g), m.group(0))
            plan["spec"] = {"field": field, "value": value}
            break

    cat = re.search(
        r"\b(?:in the|category)\s+([A-Za-z0-9 &/'-]{3,60}?)\s+category\b",
        goal,
        flags=re.IGNORECASE,
    )
    if cat:
        plan["category"] = cat.group(1).strip()
    else:
        for word in (
            "smartphone",
            "monitor",
            "headphone",
            "headphones",
            "laptop",
            "webcam",
            "groceries",
        ):
            if word in lower:
                plan["category"] = word
                break

    # Prefer the first sentence / clause for search keywords.
    head = re.split(r"[.\n]", goal, maxsplit=1)[0]
    head = re.sub(
        r"(?:at or under|at or below|up to|under|below)\s*[\d,]+(?:\.\d+)?\s*(?:php|pesos)?",
        " ",
        head,
        flags=re.IGNORECASE,
    )
    head = re.sub(
        r"(?:ideally|i prefer|prefer|preferably)[\s\w&./-]{0,40}",
        " ",
        head,
        flags=re.IGNORECASE,
    )
    # Also harvest a few tokens from the rest of the goal (soft prefs / match phrases).
    rest = goal[len(head) :] if len(goal) > len(head) else ""
    quoted = re.findall(r'"([^"]{2,80})"', goal)
    corpus = " ".join([head, rest[:180], *quoted])
    tokens = [
        t
        for t in re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", corpus.lower())
        if len(t) > 1 and t not in _FILLER and not re.fullmatch(r"\d+(?:\.\d+)?", t)
    ]
    # Keep useful numeric specs (e.g. 7.6 from screen size) beside category words.
    spec_val = (plan.get("spec") or {}).get("value")
    if spec_val:
        tokens.append(str(spec_val))
    if plan.get("category"):
        tokens.insert(0, str(plan["category"]).lower())
    if plan.get("brand_firm"):
        tokens.insert(0, str(plan["brand_firm"]).lower())
    plan["keywords"] = " ".join(dict.fromkeys(tokens[:12]))
    return plan


def _llm_refine_plan(goal: str, seed: dict[str, Any]) -> dict[str, Any]:
    """Optional LLM refine of constraints; falls back to heuristic seed."""
    inference = _proxy.post(
        "/inference/chat/completions",
        json_data={
            "model": _model(),
            "messages": [
                {"role": "system", "content": _DECOMPOSE_PROMPT},
                {"role": "user", "content": goal},
            ],
            "temperature": 0,
        },
    )
    if not inference:
        return seed
    content = (inference.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    parsed = _parse_json_content(content)
    if not parsed:
        return seed
    out = dict(seed)
    for key in (
        "keywords",
        "category",
        "brand_soft",
        "brand_firm",
        "max_price",
        "min_price",
        "must_in_stock",
        "needs_ranking",
        "needs_justification",
        "given_candidates",
        "spec",
    ):
        if key in parsed and parsed[key] not in (None, "", []):
            out[key] = parsed[key]
    return out


def _plan_hint(plan: dict[str, Any]) -> str:
    lines = [
        "Internal plan (follow Task rules if they conflict):",
        f"- search keywords: {plan.get('keywords') or '(derive from goal)'}",
        f"- category: {plan.get('category')}",
        f"- firm brand: {plan.get('brand_firm')}",
        f"- soft brand: {plan.get('brand_soft')}",
        f"- price window: min={plan.get('min_price')} max={plan.get('max_price')}",
        f"- key spec: {plan.get('spec')}",
        f"- must_in_stock: {plan.get('must_in_stock')}",
        f"- needs_ranking: {plan.get('needs_ranking')}",
        f"- needs_justification: {plan.get('needs_justification')}",
    ]
    cands = plan.get("given_candidates") or []
    if cands:
        lines.append(f"- given candidates to compare: {', '.join(map(str, cands[:12]))}")
    lines.append(
        "Start with search/filter using keywords + budget/stock filters when available. "
        "Prefer soft brand only after firm filters pass. End with inspect_stock then place_test_order."
    )
    return "\n".join(lines)


def _tokenize(text: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 1 and t not in _FILLER
    ]


def _title_hit_score(tokens: list[str], title: str) -> float:
    words = set(re.findall(r"[a-z0-9]+", title.lower()))
    score = 0.0
    for w in tokens:
        if w in words:
            score += 2.0
        elif any(x.startswith(w) or w.startswith(x) for x in words if len(x) > 2):
            score += 1.0
        if any(ch.isdigit() for ch in w) and w in title.lower():
            score += 2.0
    return score


def _unwrap_payload(observation: dict[str, Any]) -> dict[str, Any]:
    """Runtime observations may nest payload under observation.observation."""
    if not isinstance(observation, dict):
        return {}
    inner = observation.get("observation")
    if isinstance(inner, dict) and (
        "action" in inner or "results" in inner or "product" in inner or "variant" in inner
    ):
        return inner
    return observation


def _is_done(observation: dict[str, Any]) -> bool:
    if not isinstance(observation, dict):
        return False
    if observation.get("done") is True:
        return True
    inner = observation.get("observation")
    return isinstance(inner, dict) and inner.get("done") is True


def _rank_search_results(observation: dict[str, Any], plan: dict[str, Any]) -> str | None:
    payload = _unwrap_payload(observation)
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    tokens = _tokenize(str(plan.get("keywords") or ""))
    max_price = plan.get("max_price")
    brand_soft = _norm(plan.get("brand_soft") or "")
    brand_firm = _norm(plan.get("brand_firm") or "")
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        score = _title_hit_score(tokens, title)
        price = item.get("price")
        if isinstance(price, (int, float)) and isinstance(max_price, (int, float)):
            score += 5.0 if price <= max_price else -25.0
        if item.get("in_stock") is True:
            score += 3.0
        elif item.get("in_stock") is False:
            score -= 20.0
        brand = _norm(item.get("brand") or "")
        if brand_firm:
            score += 8.0 if brand == brand_firm or brand_firm in _norm(title) else -12.0
        elif brand_soft and (brand == brand_soft or brand_soft in _norm(title)):
            score += 4.0
        scored.append((score, item))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("product_id", ""))))
    lines = ["Ranked search hits (heuristic; verify with view/inspect_stock):"]
    for score, item in scored[:8]:
        lines.append(
            f"- score={score:.1f} id={item.get('product_id')} sku={item.get('sku')} "
            f"brand={item.get('brand')} price={item.get('price')} stock={item.get('in_stock')} "
            f"title={str(item.get('title') or '')[:100]}"
        )
    return "\n".join(lines)


def _ingest_observation(memory: dict[str, Any], observation: dict[str, Any]) -> None:
    if not isinstance(observation, dict):
        return
    payload = _unwrap_payload(observation)
    blob = json.dumps(observation, ensure_ascii=False)

    def remember(field: str, value: Any) -> None:
        if value is None or value == "":
            return
        memory.setdefault("facts", {})[field] = value
        memory.setdefault("fact_norms", {})[field] = _norm(value)

    for key, field in (
        ("brand", "brand"),
        ("category", "category"),
        ("category_path", "category"),
        ("screen_size", "screen_size"),
        ("resolution", "resolution"),
        ("refresh_rate", "refresh_rate"),
        ("price", "price"),
        ("current_price", "price"),
        ("in_stock", "in_stock"),
        ("stock", "in_stock"),
    ):
        if key in payload:
            remember(field, payload[key])

    product = payload.get("product")
    if isinstance(product, dict):
        for key, field in (
            ("brand", "brand"),
            ("category", "category"),
            ("category_path", "category"),
            ("screen_size", "screen_size"),
            ("resolution", "resolution"),
            ("refresh_rate", "refresh_rate"),
        ):
            if key in product:
                remember(field, product[key])
        if product.get("product_id"):
            memory["product_id"] = str(product["product_id"])

    variant = payload.get("variant")
    if isinstance(variant, dict):
        for key, field in (
            ("price", "price"),
            ("current_price", "price"),
            ("in_stock", "in_stock"),
            ("stock", "in_stock"),
            ("sku", "sku"),
        ):
            if key in variant:
                remember(field, variant[key])
        if variant.get("product_id"):
            memory["product_id"] = str(variant["product_id"])

    for key in ("product_id", "sku"):
        if payload.get(key) not in (None, ""):
            memory[key] = str(payload[key])

    raw_facts = payload.get("raw_product_facts") or payload.get("raw_facts")
    if isinstance(raw_facts, dict):
        for key, field in (
            ("brand", "brand"),
            ("screen_size", "screen_size"),
            ("resolution", "resolution"),
            ("refresh_rate", "refresh_rate"),
            ("category", "category"),
        ):
            if key in raw_facts:
                remember(field, raw_facts[key])

    # Cache search results for recovery hints.
    results = payload.get("results")
    if isinstance(results, list):
        memory["last_results"] = [r for r in results if isinstance(r, dict)][:10]

    action = payload.get("action")
    if action:
        memory["last_action"] = action
    if observation.get("error") or payload.get("error"):
        memory["last_error"] = observation.get("error") or payload.get("error")

    memory["last_obs_norm"] = _norm(blob)
    memory["saw_inspect_stock"] = memory.get("saw_inspect_stock", False) or (
        action == "inspect_stock" or "inspect_stock" in blob
    )
    memory["saw_view"] = memory.get("saw_view", False) or action in {"view", "compare"}


def _claims_are_grounded(
    args: dict[str, Any], memory: dict[str, Any]
) -> tuple[bool, str]:
    justification = args.get("justification")
    if not isinstance(justification, dict):
        return True, ""
    claims = justification.get("claims")
    if not claims:
        return True, ""
    if not isinstance(claims, list):
        return False, "justification.claims must be a list of {field,value}."

    fact_norms: dict[str, str] = memory.get("fact_norms") or {}
    last_obs = memory.get("last_obs_norm") or ""

    for claim in claims:
        if not isinstance(claim, dict):
            return False, "each claim must be an object with field and value."
        field = claim.get("field")
        value = claim.get("value")
        if field not in _CLAIM_FIELDS:
            return False, f"claim field '{field}' is not allowed."
        if value is None or str(value).strip() == "":
            return False, f"claim '{field}' has empty value."

        claimed = _norm(value)
        known = fact_norms.get(field)
        if known is not None:
            if claimed == known or claimed in known or known in claimed:
                continue
            if field in {"price", "in_stock"} and claimed in last_obs:
                continue
            return (
                False,
                f"claim {field}={value!r} does not match observed "
                f"{field}={memory.get('facts', {}).get(field)!r}.",
            )
        if claimed not in last_obs and not any(
            claimed in _norm(json.dumps(v)) for v in (memory.get("facts") or {}).values()
        ):
            return (
                False,
                f"claim {field}={value!r} was not observed via view/compare/inspect_stock.",
            )
    return True, ""


def _needs_order_guard(
    tool_calls: list[dict[str, Any]], memory: dict[str, Any], tool_names: set[Any]
) -> str | None:
    for call in tool_calls:
        name = call.get("function", {}).get("name")
        if name != "place_test_order":
            continue
        try:
            args = _arguments(call)
        except Exception as exc:  # noqa: BLE001
            return f"place_test_order arguments invalid: {exc}"
        if "inspect_stock" in tool_names and not memory.get("saw_inspect_stock"):
            return (
                "Call inspect_stock on the chosen product_id/sku before place_test_order "
                "so price and stock are current."
            )
        in_stock = memory.get("facts", {}).get("in_stock")
        if in_stock is False or _norm(in_stock) in {"false", "0", "out of stock"}:
            return "Chosen variant is not in stock; recover with another firm-compliant option."
        ok, reason = _claims_are_grounded(args, memory)
        if not ok:
            return reason
    return None


def _chat(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": _model(),
        "messages": messages,
        "temperature": 0,
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "required"
    inference = _proxy.post("/inference/chat/completions", json_data=payload)
    if inference is None:
        raise RuntimeError("inference request failed")
    return inference["choices"][0]["message"]


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Shopping loop: parse → dynamic tools → verify → order, with recovery guards."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    max_steps = int(policy["max_steps"])
    tools = policy.get("tools") or []
    tool_names = {
        (t.get("function") or {}).get("name")
        for t in tools
        if isinstance(t, dict)
    }

    full_query = policy["query"]
    goal = _shopper_goal(full_query)
    plan = _heuristic_parse(goal)
    # One optional refine; ignore failures and keep heuristic plan.
    try:
        plan = _llm_refine_plan(goal, plan)
    except Exception:  # noqa: BLE001
        pass

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_POLICY},
        {"role": "user", "content": full_query},
        {"role": "user", "content": _plan_hint(plan)},
    ]
    if "inspect_stock" in tool_names:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Reminder: before place_test_order, call inspect_stock on the chosen "
                    "variant and ground any justification claims in viewed facts."
                ),
            }
        )

    dialogue: list[dict[str, Any]] = [{"role": "plan", "content": plan}]
    memory: dict[str, Any] = {"facts": {}, "fact_norms": {}}
    guard_retries = 0
    env_turn = 0
    max_model_steps = max_steps + 6
    stagnant_searches = 0

    for _model_step in range(1, max_model_steps + 1):
        if env_turn >= max_steps:
            break

        for _attempt in range(2):
            assistant = _chat(messages, tools)
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
                            "The environment has not reported done=true. "
                            "Choose one of the supplied tools to continue."
                        ),
                    },
                ]
            )
        else:
            raise RuntimeError("model returned no tool call before completion")

        guard_reason = _needs_order_guard(tool_calls, memory, tool_names)
        if guard_reason and guard_retries < 3:
            guard_retries += 1
            dialogue.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                    "guard_blocked": guard_reason,
                }
            )
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
                            "Do not place the order yet. "
                            f"{guard_reason} "
                            "View/compare/inspect_stock as needed, then place_test_order with "
                            "truthful field/value claims only from those observations."
                        ),
                    },
                ]
            )
            continue

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        env_turn += 1
        group_id = f"{problem_id}-turn-{env_turn}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": env_turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {
                        "name": call["function"]["name"],
                        "args": _arguments(call),
                    },
                }
                for index, call in enumerate(tool_calls, start=1)
            ],
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
            }
        )

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            observation = call_result.get("observation") or {}
            _ingest_observation(memory, observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(observation),
                }
            )
            rank_note = _rank_search_results(observation, plan)
            if rank_note:
                stagnant_searches += 1
                messages.append({"role": "user", "content": rank_note})
                if stagnant_searches >= 2 and plan.get("keywords"):
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "If these hits are weak, reformulate: try filter with category/"
                                f"brand/max_price, or search keywords '{plan.get('keywords')}' "
                                "with fewer adjectives. Then view top distinct candidates."
                            ),
                        }
                    )

        if result.get("user_message"):
            messages.append(
                {"role": "user", "content": result["user_message"]["content"]}
            )

        if memory.get("last_error"):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Last tool error: {memory['last_error']}. "
                        "Recover: adjust args, remove bad cart items if needed, try another "
                        "firm-compliant candidate, then inspect_stock before ordering."
                    ),
                }
            )
            memory["last_error"] = None

        if any(_is_done(call.get("observation") or {}) for call in result["calls"]):
            break

    return dialogue


__all__ = ["agent_main"]
