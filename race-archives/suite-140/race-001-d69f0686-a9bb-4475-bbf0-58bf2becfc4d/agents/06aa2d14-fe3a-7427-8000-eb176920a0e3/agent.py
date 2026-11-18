from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_OPENROUTER_DEFAULT = "deepseek/deepseek-v3.2"
_OPENROUTER_HARD = "moonshotai/kimi-k2.6"
_OPENROUTER_FALLBACKS = (
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "google/gemma-4-31b-it",
    "deepseek/deepseek-v3.2",
)

_proxy = ProxyClient(timeout=120, max_retries=2)

_FAMILY_FINGERPRINTS: list[tuple[str, list[re.Pattern[str]]]] = [
    (
        "ranking",
        [
            re.compile(r"tradeoff priority|best-ranked|strictly worse", re.I),
            re.compile(r"candidate variants|\bre-?rank\b|\bdominat", re.I),
            re.compile(r"\d{8,}::\S+"),
        ],
    ),
    (
        "recovery",
        [
            re.compile(r"least-?relax|relaxable only|becomes? unavailable", re.I),
            re.compile(r"same model.{0,40}(?:other|another) color", re.I),
            re.compile(r"exact (?:item|model|color).{0,60}disappear", re.I),
        ],
    ),
    (
        "justification",
        [

            re.compile(
                r"(?:must|should) include.{0,40}justification|"
                r"structured justification with claims|"
                r"trustworthy reason|"
                r"ground claims",
                re.I,
            ),
            re.compile(
                r"walk me through why|tell me why|give me a reason|"
                r"\bwhy (?:the one|it'?s|your pick)\b",
                re.I,
            ),
        ],
    ),
    (
        "intent_decomposition",
        [
            re.compile(r"nice[- ]to[- ]have", re.I),
            re.compile(r"\bsubordinate\b.{0,40}\b(?:firm|requirement)|preferred brand", re.I),
            re.compile(r"initial (?:cart )?choice.{0,80}(?:soft|prefer|brand)", re.I),
            re.compile(r"\bfirm\b.{0,80}\bsoft\b|\bsoft\b.{0,80}\bfirm\b", re.I),
        ],
    ),
    (
        "constraint_satisfaction",
        [
            re.compile(r"exact(?:ly)? varian|specific sku", re.I),
            re.compile(r"every requirement must|pixel class|resolution class", re.I),
            re.compile(
                r"firm requirement on the spec|reveal(?:ed)? only when|"
                r"tell you if you ask|if you ask",
                re.I,
            ),
        ],
    ),
    (
        "preference_reasoning",
        [
            re.compile(r"preference axes|blindly buying the cheapest", re.I),
            re.compile(r"intended use|loose hard request", re.I),
            re.compile(r"rather pay|sweat|water resistance|use[- ]case", re.I),
        ],
    ),
    (
        "retrieval_recall",
        [
            re.compile(r"\breformulate\b|\blucky hit\b|\bshortlist\b", re.I),
            re.compile(r"survey the realistic|distinct products", re.I),
            re.compile(r"matches? (?:the )?(?:phrase )?\"[^\"]+\"", re.I),
        ],
    ),
]

_ASK_SPEC_RX = re.compile(
    r"(?:tell|reveal|share).{0,40}\bif you ask\b|"
    r"\bif you ask\b|"
    r"ask (?:me |for )?(?:the |that )?(?:exact )?(?:requirement|spec)",
    re.I,
)

_PROMPT_CORE = """
You are DarkGirl v6, an ORO Bench shopping agent.

# Contract
- Use ONLY the supplied tools from the environment.
- Continue until an observation reports done=true.
- Respect max_steps and max_calls_per_turn.
- Prefer parallel tool calls when the limit allows and it saves steps.

# Healthy loop
1) Discover — search / filter
2) Inspect — view / compare / inspect_stock
3) Decide — add_to_cart / remove_from_cart
4) Confirm — inspect_stock / inspect_cart on the final SKU
5) Finish — place_test_order (with justification.claims when required)

# Precision discipline (from champion select prompts)
A candidate is valid ONLY if every FIRM constraint from the shopper goal is
satisfied on the EXACT variant you order.

## Constraint extraction
Extract constraints into categories such as:
brand, model, series, identifier/SKU, number, size, capacity, color, material,
feature, compatibility, category, budget/price, stock, resolution, refresh_rate,
screen_size, use-case preference.

Mark each as FIRM (must hold) or SOFT (preference / nice-to-have) using the
Task rules. Soft preferences never override firm ones.

## Matching modes
- Literal: brands, model codes, identifiers, uncommon tokens, exact sizes,
  colors named as codes. Must appear explicitly (casing may differ).
- Size aliases only: s/small, m/medium, l/large, xl/extra large, etc.
- Semantic: everyday paraphrases when Task rules allow them.
  Prefer false positives over false negatives for soft semantic checks.
- Spec classes (resolution, size bands, etc.): follow definitions in the
  Task rules when they provide them — do not invent thresholds.

## Search / keyword refine (from champion review prompts)
Catalog search matches VALUES, not JSON keys. Prefer value tokens:
product type words, brand, model codes, distinctive phrase fragments.
Avoid dumping field names like "brand" or "category" as keywords.
Hyphens ≈ spaces. Morphological variants usually match.
If the first search is weak, make ONE small reformulation (drop noise,
normalize size words, keep distinctive tokens) — do not rewrite everything.

# Messaging
Use message when Task rules require clarifying priorities or revealing a
hidden firm spec. Keep messages short.
""".strip()

_PROMPT_FAMILY: dict[str, str] = {
    "intent_decomposition": """
# Family: TF1 Intent Decomposition
Refs: Ostrich / IamSUPER (decompose firm vs soft, then act).

Plan a short decision sequence before committing:
1. List firm requirements (category, one hardware spec, budget, in-stock).
2. List soft preferences (named preferred brand).
3. search/filter → view compliant SKUs.
4. First cart choice: preferred brand IF a fully compliant SKU exists;
   otherwise the best firm-compliant item. No placeholders.
5. inspect_stock → place_test_order.

The first cart add is scored separately from later fixes: if a fully
compliant preferred-brand option exists, choose it first.
""".strip(),
    "retrieval_recall": """
# Family: TF2 Retrieval Recall
Refs: Paris / Bolt-2026 (precision select + keyword refine).

- Reformulate queries; try phrase, tokens, and category+filter.
- Survey several DISTINCT products before buying — avoid committing after
  a single weak hit.
- View/compare promising hits; reject accessories and off-category noise.
- Once a solid shortlist exists, buy the best in-stock item within budget.
- Do not wait for permission after a verified in-stock, in-budget pick.
""".strip(),
    "constraint_satisfaction": """
# Family: TF3 Constraint Satisfaction
Refs: Storm / Slackbro (exact-variant discipline).

- EVERY firm requirement must hold on the EXACT SKU ordered.
- Titles often mix compliant and non-compliant variants — always view, then
  inspect_stock on the chosen SKU (price + stock + option).
- If the shopper withholds a firm spec until asked, call message FIRST to
  request that requirement before searching.
- Apply any class/threshold rules written in the Task rules.
""".strip(),
    "preference_reasoning": """
# Family: TF4 Preference Reasoning
Refs: sangaria / IamSUPER (tradeoffs, not cheapest-by-default).

Hard: category, budget, in-stock, real device (not accessory/review page).
Soft: use-case preferences stated by the shopper (durability, battery,
materials, connectivity, environment of use, etc.).

Before the first cart add:
- Optionally message to clarify priorities.
- compare/view candidates across soft axes.
- Prefer a slightly dearer item that matches soft goals if still in budget.
""".strip(),
    "ranking": """
# Family: TF5 Ranking
Refs: PriceMeridian / boston (ordered tradeoffs).

- Restrict to the candidate variants named in the query (product_id::sku).
  Prefer viewing those IDs directly; do not burn the step budget on open search.
- Rank ONLY by the tradeoff priority stated in the Task rules / goal
  (e.g. refresh rate → screen size → price). Never invent extra axes.
- View/compare facts before the first cart add; the first cart choice must be
  the best-ranked AVAILABLE option — never a placeholder.
- After stockout / price / cart events (often after remove_from_cart or a
  failed add): re-inspect remaining candidates, RE-RANK from scratch with the
  same priority, cart the new #1, inspect_stock, then place_test_order.
  Do NOT keep or re-order the pre-event #1 if it is no longer best/available.
- Never pick an option that is strictly dominated on every stated axis.
- Always finish with place_test_order before max_steps.

CRITICAL — place_test_order:
- Pass ONLY product_id and sku.
- Do NOT include justification/claims unless the Task rules require them.
""".strip(),
    "recovery": """
# Family: TF6 Recovery
Refs: Abudullah (least-relaxation recovery).

Never relax: category, budget, in-stock.
Prefer the exact named model/color first.
If it disappears after a cart action / stock change, recover with the
least relaxation allowed by the Task rules, then re-check price/stock
before place_test_order.
Use explicit model facts when supplied; conflicting titles do not override them.
""".strip(),
    "justification": """
# Family: TF7 Justification

- View/compare raw facts before the first cart add and before ordering.
- Do not justify from marketing title words alone.
- place_test_order MUST include justification.claims as field/value pairs.
- Claim `field` must be a catalog attribute key (or a schema-listed reason),
  NOT marketing phrases. Prefer keys that match the shopper's firm spec:
  resolution, screen_size, brand, price, in_stock.
  Schema may also allow feature reasons (e.g. hdr) ONLY when view/compare
  shows that capability as explicitly true — never invent them.
- Do not use aliases like display/size/\"720p HD\" as the field name; map them
  to resolution or screen_size. Put the measured value in `value`.
- Ground every claim in viewed or inspect_stock/cart state.
- Keep claims minimal (spec + price + in_stock + brand when relevant).
  No summary/evidence prose inside claims.
""".strip(),
}

_STUCK_NUDGES = (
    "No tool call received. Choose a supplied tool now. "
    "If you have not searched yet, call search or filter.",
    "Still no tool call. Progress the episode: search/filter, view a candidate, "
    "or place_test_order if the cart already holds a verified in-stock SKU.",
)

_COACH_ASK_SPEC = (
    "Coach: a firm requirement appears to be withheld until you ask. "
    "Call message first to request that exact requirement before searching."
)

_COACH_RECOVERY = (
    "Coach: an action failed or stock/price changed ({error}). Recover with the "
    "least relaxation that still satisfies category, budget, and in-stock."
)

_COACH_RANK_RERANK = (
    "Coach [TF5]: market/cart state changed. Re-inspect the named candidate "
    "variants that are still available, re-rank by the stated tradeoff "
    "priority, cart the NEW #1 (not the pre-event pick), inspect_stock, then "
    "place_test_order with only product_id+sku."
)

_COACH_RANK_FINISH = (
    "Coach [TF5]: stop searching. Re-rank the named candidates still in stock, "
    "add the current #1 to cart if needed, and place_test_order now "
    "(product_id+sku only)."
)

_COACH_TF7_CLAIMS = (
    "Coach [TF7]: on place_test_order, justification.claims must use attribute "
    "keys such as resolution, screen_size, brand, price, in_stock — never "
    "aliases like display/size. Values must match view/inspect_stock facts."
)

_CLAIM_FIELD_ALIASES = {
    "display": "resolution",
    "resolution class": "resolution",
    "pixel class": "resolution",
    "screen": "screen_size",
    "size": "screen_size",
    "screen size": "screen_size",
    "stock": "in_stock",
    "availability": "in_stock",
}

_FEATURE_REASON_FIELDS = frozenset(
    {"streaming", "microphone", "privacy_cover", "autofocus", "hdr"}
)

_VARIANT_ID_RX = re.compile(r"\b(\d{10,}::[A-Za-z0-9._-]+)\b")
_SPEC_PHRASE_AS_FIELD_RX = re.compile(
    r"(?i)^(?:\d{3,4}\s*[pP](?:\s*HD)?|full\s*hd|qhd|uhd|4k|"
    r"\d+(?:\.\d+)?\s*-?\s*inch(?:es)?)$"
)

def _model_for(family: str) -> str:

    override = (getenv("SANDBOX_MODEL") or "").strip()
    if override:

        mapped = _map_chutes_to_openrouter(override)
        return mapped
    hard = {
        "constraint_satisfaction",
        "preference_reasoning",
        "ranking",
        "recovery",
        "justification",
        "intent_decomposition",
    }
    return _OPENROUTER_HARD if family in hard else _OPENROUTER_DEFAULT

def _map_chutes_to_openrouter(model: str) -> str:
    table = {
        "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
        "deepseek-ai/DeepSeek-V3.1-TEE": "deepseek/deepseek-chat-v3.1",
        "deepseek-ai/DeepSeek-V3-0324-TEE": "deepseek/deepseek-chat-v3-0324",
        "deepseek-ai/DeepSeek-R1-0528-TEE": "deepseek/deepseek-r1-0528",
        "Qwen/Qwen3-32B-TEE": "qwen/qwen3-32b",
        "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
        "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
        "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
        "zai-org/GLM-5.1-TEE": "z-ai/glm-5.1",
        "zai-org/GLM-5.2-TEE": "z-ai/glm-5.2",
        "moonshotai/Kimi-K2.6-TEE": "moonshotai/kimi-k2.6",
        "openai/gpt-oss-120b-TEE": "openai/gpt-oss-120b",
        "XiaomiMiMo/MiMo-V2-Flash-TEE": "xiaomi/mimo-v2-flash",
        "unsloth/Mistral-Nemo-Instruct-2407-TEE": "mistralai/mistral-small-2603",
    }
    return table.get(model, model)

def _detect_family(query: str) -> str:

    text = query or ""
    scores: dict[str, int] = {}
    for fam, patterns in _FAMILY_FINGERPRINTS:
        hits = sum(1 for rx in patterns if rx.search(text))
        if hits:
            scores[fam] = hits

    if "justification is optional" in text.lower() and scores.get("justification"):
        goal = text.split("Task rules:")[0] if "Task rules:" in text else text
        if not re.search(r"\bwhy\b|reason|justify", goal, re.I):
            scores.pop("justification", None)

    if not scores:
        return "retrieval_recall"

    priority = {
        "ranking": 7,
        "recovery": 6,
        "justification": 5,
        "constraint_satisfaction": 4,
        "intent_decomposition": 3,
        "preference_reasoning": 2,
        "retrieval_recall": 1,
    }
    return max(scores.items(), key=lambda kv: (kv[1], priority.get(kv[0], 0)))[0]

def _needs_ask_spec(query: str) -> bool:
    goal = query.split("Task rules:")[0] if "Task rules:" in query else query
    return bool(_ASK_SPEC_RX.search(goal))

def _named_variants(query: str) -> list[str]:
    return list(dict.fromkeys(_VARIANT_ID_RX.findall(query or "")))

def _infer_spec_claim_field(query: str) -> str | None:

    goal = (query or "").split("Task rules:")[0].lower()
    if re.search(r"\b\d+(?:\.\d+)?\s*-?\s*inch|\bscreen size\b", goal):
        return "screen_size"
    if re.search(
        r"\b(?:\d{3,4}\s*p|720p|1080p|full\s*hd|qhd|uhd|4k|resolution)\b",
        goal,
    ):
        return "resolution"
    if re.search(r"\brefresh\b|\bhz\b", goal):
        return "refresh_rate"
    if re.search(r"\bram\b|\bgb\b", goal):
        return "ram"
    return None

def _normalize_claim_field(field: str, query: str) -> str:
    raw = str(field or "").strip()
    key = raw.lower()
    if key in _CLAIM_FIELD_ALIASES:
        return _CLAIM_FIELD_ALIASES[key]
    if _SPEC_PHRASE_AS_FIELD_RX.match(raw):
        return _infer_spec_claim_field(query) or "resolution"
    return raw

def _normalize_claim_value(field: str, value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if field == "screen_size":
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            return f"{m.group(1)} inches"
    if field == "resolution":
        low = text.lower().replace(" ", "")
        if low in {"720p", "720phd", "hd"} or "1280" in low and "720" in low:
            return "hd"
        if low in {"1080p", "1080pfullhd", "fullhd", "fhd"} or (
            "1920" in low and "1080" in low
        ):

            if "1920" in text and "1080" in text:
                return re.sub(r"\s+", " ", text)
            return "1920 x 1080"
        if low in {"1440p", "qhd"} or ("2560" in low and "1440" in low):
            if "2560" in text and "1440" in text:
                return re.sub(r"\s+", " ", text)
            return "2560 x 1440"
    if field == "in_stock":
        return "true" if text.lower() in {"true", "1", "yes", "in_stock"} else text
    return text

def _sanitize_tool_args(
    name: str, args: dict[str, Any], *, family: str, query: str
) -> dict[str, Any]:

    out = dict(args)
    if name != "place_test_order":
        return out

    if family == "ranking":

        goal = query.split("Task rules:")[0] if "Task rules:" in query else query
        needs_why = bool(re.search(r"\bwhy\b|justif|reason|tell me why", goal, re.I))
        if not needs_why:
            out.pop("justification", None)
        return out

    if family != "justification":
        return out

    just = out.get("justification")
    if not isinstance(just, dict):
        return out
    claims = just.get("claims")
    if not isinstance(claims, list):
        return out

    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    goal_l = (query.split("Task rules:")[0] if "Task rules:" in query else query).lower()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        field = _normalize_claim_field(str(claim.get("field") or ""), query)
        value = _normalize_claim_value(field, claim.get("value"))
        if not field or value == "":
            continue
        fl = field.lower()
        if fl in _FEATURE_REASON_FIELDS and fl not in goal_l:
            continue

        if fl in seen:
            continue
        seen.add(fl)
        cleaned.append({"field": field, "value": value})

    if cleaned:
        out["justification"] = {"claims": cleaned}
    return out

def _tool_name_list(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
        elif isinstance(tool, dict) and tool.get("name"):
            names.append(str(tool["name"]))
    return names

def _system_prompt(family: str, policy: dict[str, Any], query: str) -> str:
    names = _tool_name_list(policy.get("tools") or [])
    family_prompt = _PROMPT_FAMILY.get(family, _PROMPT_FAMILY["retrieval_recall"])
    variants = _named_variants(query)
    variant_line = ""
    if family == "ranking" and variants:
        variant_line = (
            "Named candidate variants (view these; re-rank after events):\n"
            + ", ".join(variants)
            + "\n"
        )
    return (
        f"{_PROMPT_CORE}\n\n"
        f"{family_prompt}\n\n"
        f"{variant_line}"
        f"# Runtime\n"
        f"Detected task family: {family}\n"
        f"Available tools: {', '.join(names) if names else '(see tool schemas)'}\n"
        f"max_steps={policy.get('max_steps')} "
        f"max_calls_per_turn={policy.get('max_calls_per_turn', 1)}\n"
    )

def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    if isinstance(raw, dict):
        parsed = raw
    else:
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed

def _compact_observation(obs: Any, limit: int = 6000) -> str:
    try:
        text = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obs)
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "...<truncated>"

def _tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    names = []
    for call in tool_calls:
        try:
            names.append(str(call["function"]["name"]))
        except (KeyError, TypeError):
            names.append("?")
    return names

def _progress_hint(
    family: str,
    turn: int,
    last_tool_names: list[str],
    last_obs: list[Any],
    query: str,
    *,
    saw_cart_event: bool = False,
    ordered: bool = False,
) -> str | None:
    joined = " ".join(last_tool_names)

    if (
        family == "constraint_satisfaction"
        and turn == 1
        and _needs_ask_spec(query)
        and "message" not in joined
    ):
        return _COACH_ASK_SPEC

    if family == "ranking" and not ordered:
        if saw_cart_event and turn >= 2:
            if "place_test_order" not in joined:
                return _COACH_RANK_RERANK
        if turn >= 8:
            return _COACH_RANK_FINISH
        if turn == 1:
            return (
                "Hint [TF5]: view/compare the named candidate variants and rank "
                "by the stated tradeoff axes before add_to_cart. After any "
                "stock/cart event, re-rank and order the new #1. "
                "place_test_order = product_id+sku only."
            )

    if turn <= 2 and not any(
        n in joined for n in ("search", "filter", "view", "compare", "message")
    ):
        return (
            "Hint: start with search or filter using value-side keywords from the "
            "shopper goal, then view the best hits."
        )
    if family == "justification" and turn >= 2:
        return _COACH_TF7_CLAIMS
    if family == "recovery" and any(
        isinstance(o, dict) and o.get("error") for o in last_obs
    ):
        return _COACH_RECOVERY.format(error=str(
            next(o.get("error") for o in last_obs if isinstance(o, dict) and o.get("error"))
        ))
    if family == "retrieval_recall" and turn == 3 and "search" in joined:
        return (
            "Hint: make one small keyword reformulation or add a filter so the "
            "shortlist covers several distinct products before ordering."
        )
    if family == "preference_reasoning" and turn == 2 and "compare" not in joined:
        return (
            "Hint: compare candidates on soft preference axes (not only price) "
            "before the first cart add."
        )
    return None

def _infer(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_calls: int,
) -> tuple[str, list[dict[str, Any]]]:

    models_to_try = [model]
    for fb in _OPENROUTER_FALLBACKS:
        if fb not in models_to_try:
            models_to_try.append(fb)

    last_err: Exception | None = None
    for mid in models_to_try[:3]:
        inference = _proxy.post(
            "/inference/chat/completions",
            json_data={
                "model": mid,
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "temperature": 0,
            },
        )
        if inference is None:
            last_err = RuntimeError(f"inference failed for model={mid}")
            continue
        try:
            assistant = inference["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            last_err = exc
            continue
        content = assistant.get("content") or ""
        tool_calls = list(assistant.get("tool_calls") or [])[:max_calls]
        return content, tool_calls
    raise RuntimeError(f"inference request failed ({last_err})")

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    query = str(policy.get("query") or problem_data.get("query") or "")
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
    max_steps = max(1, int(policy.get("max_steps", 20)))
    tools = policy.get("tools") or []

    family = _detect_family(query)
    model = _model_for(family)
    ask_spec = family == "constraint_satisfaction" and _needs_ask_spec(query)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(family, policy, query)},
        {"role": "user", "content": query},
    ]
    if ask_spec:
        messages.append({"role": "user", "content": _COACH_ASK_SPEC})

    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "darkgirl": {
                "version": 6,
                "family": family,
                "model": model,
                "provider": "openrouter",
            },
        }
    ]

    last_tool_names: list[str] = []
    last_obs: list[Any] = []
    saw_cart_event = False
    ordered = False

    for turn in range(1, max_steps + 1):
        hint = _progress_hint(
            family,
            turn,
            last_tool_names,
            last_obs,
            query,
            saw_cart_event=saw_cart_event,
            ordered=ordered,
        )
        if hint:
            messages.append({"role": "user", "content": hint})

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for attempt in range(len(_STUCK_NUDGES) + 1):
            assistant_content, tool_calls = _infer(
                model=model,
                messages=messages,
                tools=tools,
                max_calls=max_calls,
            )
            if tool_calls:
                break
            dialogue.append({"role": "assistant", "content": assistant_content})
            nudge = _STUCK_NUDGES[min(attempt, len(_STUCK_NUDGES) - 1)]
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": nudge},
                ]
            )
        else:
            raise RuntimeError("model returned no tool call before completion")

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )

        group_id = f"{problem_id}-turn-{turn}"
        call_payloads = []
        for index, call in enumerate(tool_calls, start=1):
            name = call["function"]["name"]
            args = _sanitize_tool_args(
                name, _arguments(call), family=family, query=query
            )

            if isinstance(call.get("function"), dict):
                call["function"]["arguments"] = json.dumps(args)
            call_payloads.append(
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {"name": name, "args": args},
                }
            )
            if name == "remove_from_cart":
                saw_cart_event = True
            if name == "place_test_order":
                ordered = True

        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": call_payloads,
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
                "darkgirl_turn": {"family": family, "turn": turn},
            }
        )

        last_tool_names = _tool_names(tool_calls)
        last_obs = []
        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            obs = call_result.get("observation")
            last_obs.append(obs)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _compact_observation(obs),
                }
            )
            if isinstance(obs, dict) and (
                obs.get("error")
                or obs.get("stock_changed")
                or obs.get("price_changed")
                or obs.get("event")
            ):
                saw_cart_event = True

        user_message = result.get("user_message")
        if user_message and user_message.get("content"):
            messages.append(
                {"role": "user", "content": str(user_message["content"])}
            )

        errors = [
            o.get("error")
            for o in last_obs
            if isinstance(o, dict) and o.get("error")
        ]
        if errors and family == "recovery":
            messages.append(
                {
                    "role": "user",
                    "content": _COACH_RECOVERY.format(error=errors[0]),
                }
            )
        if family == "ranking" and saw_cart_event and not ordered and errors:
            messages.append({"role": "user", "content": _COACH_RANK_RERANK})

        if any(
            isinstance(call.get("observation"), dict)
            and call["observation"].get("done")
            for call in result.get("calls") or []
        ):
            break

    return dialogue

__all__ = ["agent_main"]
