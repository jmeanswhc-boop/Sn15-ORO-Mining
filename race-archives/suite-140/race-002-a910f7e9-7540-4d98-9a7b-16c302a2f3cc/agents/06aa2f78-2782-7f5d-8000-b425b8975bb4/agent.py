from __future__ import annotations

import json
import logging
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

LOGGER = logging.getLogger("agent.nimbus_a_v6")

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}

_MODEL_PRIORITIES = [
    "moonshotai/Kimi-K2.6-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "Qwen/Qwen3.6-27B-TEE",
    "deepseek-ai/DeepSeek-V3-0324-TEE",
    "Qwen/Qwen3-32B-TEE",
    "XiaomiMiMo/MiMo-V2-Flash-TEE",
    "google/gemma-4-31B-turbo-TEE",
    "zai-org/GLM-5.2-TEE",
    "zai-org/GLM-5.1-TEE",
    "openai/gpt-oss-120b-TEE",
    "deepseek-ai/DeepSeek-R1-0528-TEE",
]

_VARIANT_RE = re.compile(r"\b(\d{10,})::([\w.\-]+)\b")
_BUDGET_RE = re.compile(
    r"(?:under|below|at\s+or\s+under|at\s+or\s+below|up\s+to|budget(?:\s+of)?|"
    r"spend\s+up\s+to|max(?:imum)?(?:\s+price)?|<=)\s*[:.]?\s*"
    r"(\d+(?:\.\d+)?)\s*(?:php|pesos?)?",
    re.I,
)
_SOFT_BRAND_RE = re.compile(
    r"(?:ideally[, ]+(?:i\s+)?prefer(?:\s+brand)?|prefer(?:red)?(?:\s+brand)?|"
    r"nice[- ]to[- ]have(?:\s+brand)?|soft\s+preference(?:\s+brand)?)\s+"
    r"([A-Za-z][A-Za-z0-9.\-]{1,40})(?:[.,\s]|$)",
    re.I,
)
# Named brand in justification-style goals: "Samsung 720p...", "Herman Miller tablet..."
_NAMED_BRAND_RE = re.compile(
    r"(?:(?:want|find(?:\s+me)?|get(?:\s+me)?)\s+(?:an?\s+)?|(?:can you get me\s+(?:an?\s+)?)|"
    r"(?:^|\b))"
    r"(?:in-stock\s+)?"
    r"([A-Za-z][A-Za-z0-9.\-]{1,24}(?:\s+[A-Za-z][A-Za-z0-9.\-]{1,24})?)\s+"
    r"(?:(?:\d+p|full\s*hd|hd|4k|qhd|fhd)\b|\d+(?:\.\d+)?\s*-?\s*inch|"
    r"smartphone|tablet|monitor|gaming)",
    re.I,
)
_SOFT_FEATURE_PATTERNS: dict[str, re.Pattern[str]] = {
    "streaming": re.compile(
        r"\b(streaming|conference|conferencing|zoom|teams)\b|\bvideo[- ]?call(?:ing|s)?\b",
        re.I,
    ),
    "microphone": re.compile(r"\b(?:mic|microphone)\b", re.I),
    "privacy_cover": re.compile(r"\bprivacy (?:cover|shutter)\b", re.I),
    "autofocus": re.compile(r"\bauto[- ]?focus\b", re.I),
    "hdr": re.compile(r"\b(hdr|high dynamic range)\b", re.I),
}
_SOFT_CLAIM_PRIORITY = (
    "brand",
    "streaming",
    "microphone",
    "privacy_cover",
    "autofocus",
    "hdr",
)
_SCREEN_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(?:inch|inches|in)\b|\b(\d+(?:\.\d+)?)in\b",
    re.I,
)
_REFRESH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz\b", re.I)
_RESOLUTION_TOKEN_RE = re.compile(
    r"\b(4k|uhd|fhd|qhd|hd|720p|1080p|1440p|2160p)\b", re.I
)

# Per-task limits are step-based (max_steps / max_calls_per_turn), not wall-clock.
# Keep HTTP timeouts generous so a slow inference does not look like an episode kill.
_proxy = ProxyClient(timeout=90, max_retries=2, retry_delay=0.5)


# ========================== Nimbus strategy prompts ==========================

_SYSTEM_CORE = """You are Nimbus-A, an elite shopping agent competing on ORO Bench.

Use ONLY the tools supplied in this episode. Names and schemas are dynamic —
never invent tools or assume legacy ShoppingBench APIs (find_product,
recommend_product, terminate).

## Decision discipline (from Nimbus-A)
1. Understand the shopper goal BEFORE acting. Separate FIRM requirements from
   SOFT preferences. Firm requirements always win.
2. Search/filter efficiently with short focused queries. Do not dump the entire
   shopper message into a search box when a compact keyword query works better.
3. Inspect candidates carefully (view / compare). Do not finalize on title alone.
4. Verify CURRENT price and stock (inspect_stock) before add_to_cart / order.
5. Recover when observations contradict earlier assumptions (empty results,
   out-of-stock, price above budget, shopper correction messages, catalog events).
6. Prefer the best fully compliant option. Never sacrifice a firm constraint for
   a cheaper or preferred soft attribute.
7. Minimize unnecessary calls, but never skip verification that the task rules
   require. Respect max_steps and max_calls_per_turn.
8. Finish by placing a valid order (place_test_order or the environment's
   completion tool) once a verified compliant variant is in cart — do not wait
   for permission unless the tools/rules require a shopper message first.
9. When you receive a "Nimbus observation scorer" shortlist, treat it as a
   deterministic firm>soft>lex ranking over products you already observed.
   Follow it unless a newer observation/intervention contradicts it.

## Firm vs soft (critical)
- Typical FIRM: category/product type, explicit hardware/spec constraints,
  budget/max price, in-stock / availability, explicitly listed candidate set.
- Typical SOFT: preferred brand, nice-to-have attributes, aesthetic extras.
- If a preferred brand has a fully firm-compliant in-stock option, choose it for
  the INITIAL cart commitment when task rules say so.
- If no preferred-brand item meets every firm requirement, take any firm-
  compliant item. Do not invent compliance.

## Ranking / tradeoff tasks
When the shopper lists candidate variants (product_id::sku) and a priority order
(e.g. highest refresh rate, then largest screen, then lowest price):
- Consider ONLY those candidates (view/compare them; do not wander the catalog).
- Apply the stated lexicographic priority exactly.
- Never choose a candidate that is strictly dominated on the stated metrics.
- If the environment changes mid-episode (price/stock/event), re-evaluate the
  frontier and switch cart contents before ordering.

## Justification
When the shopper asks why / for a reason, supply place_test_order.justification
claims that are TRUE and grounded in facts you observed via view/compare/
inspect_stock. Never fabricate.

## Messaging / interventions (HARD GATE on many families)
Whenever the shopper sends a user_message / question / correction / challenge
("which one are you going with?", budget change, "re-check price", etc.):
you MUST call the `message` tool with a short direct answer BEFORE continuing
search/cart/order actions. Ignoring shopper interventions fails the episode even
if the final product is correct.

## Cart commitment discipline
- Prefer ONE deliberate first add_to_cart of a firm-compliant candidate.
- Avoid add→remove thrashing before answering interventions.
- On ranking/tradeoff tasks with an explicit candidate list: after viewing facts,
  first add the current lexicographic winner; if a later environment/price/stock
  event changes the frontier, then switch cart to the new winner before ordering.

## place_test_order.justification
- Include justification ONLY when the shopper asked for a reason/why.
- On pure ranking/tradeoff candidate-list tasks, OMIT justification entirely.
- When justification IS required, claims MUST cover ALL of:
  1) price (current inspected price),
  2) in_stock ("true"),
  3) the firm gated attribute from the goal (screen_size / resolution /
     refresh_rate) using CANONICAL forms from view/compare facts
     (screen_size "27in"; resolution "hd"/"fhd"/"4k"),
  4) AT LEAST ONE truthful SOFT claim, preferred order:
     brand, then streaming / microphone / privacy_cover / autofocus / hdr.
     Soft feature claims use value "true" only when that feature appears in
     structured product facts (not title prose alone).
     Brand claims require the structured observation brand field (non-null).
- NEVER infer brand from title spelling (e.g. title "Samsungs..." with
  brand=null is NOT a Samsung claim). If brand is null, use another evidenced
  soft field or keep searching — do not assert lookalikes.
- Title/description may guide search only; they are not justification evidence.

Temperature mindset: be decisive and consistent. One careful action sequence.
Continue until an observation reports done=true."""


_PLAN_PROMPT = """You are the planning module of Nimbus-A for an ORO Bench shopping episode.

Given the shopper goal and available tool names, produce a compact JSON plan.
Do NOT solve with memorized products. Reason only from the goal text.

Return JSON:
{{
  "firm_requirements": ["..."],
  "soft_preferences": ["..."],
  "budget": <number|null>,
  "category_hint": <str|null>,
  "preferred_brand": <str|null>,
  "candidate_variants": ["product_id::sku", ...],
  "ranking_priorities": ["metric_in_priority_order", ...],
  "needs_justification": <bool>,
  "search_queries": ["short keyword query", ...],
  "filter_hints": {{"brand": <str|null>, "category": <str|null>, "max_price": <number|null>, "in_stock": true}},
  "workflow": ["ordered high-level steps using only available tools"],
  "risks": ["likely failure modes to watch for"]
}}

Available tools: {tools}
Shopper goal:
{query}
"""


# ============================== utilities ===================================

def _provider() -> str:
    explicit = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if explicit in ("chutes", "openrouter"):
        return explicit
    if getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "chutes"


def _model_chain(prefer_strong: bool = False) -> list[str]:
    """Short chain — long cascades burned the 300s remote episode budget."""
    chain: list[str] = []
    override = getenv("SANDBOX_MODEL")
    if override:
        chain.append(override)
    if prefer_strong:
        for m in (
            "deepseek-ai/DeepSeek-V3.2-TEE",
            "moonshotai/Kimi-K2.6-TEE",
            "Qwen/Qwen3.5-397B-A17B-TEE",
        ):
            if m not in chain:
                chain.append(m)
    for m in _MODEL_PRIORITIES:
        if m not in chain:
            chain.append(m)
    fallback = getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(
        _provider(), _DEFAULT_MODELS["chutes"]
    )
    if fallback not in chain:
        chain.append(fallback)
    # Cap: at most 4 models so a total outage fails fast instead of timing out.
    return chain[:4]


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            names.append(str(name))
    return names


def _shopper_goal_text(query: str) -> str:
    """Shopper ask only — Task rules boilerplate must not drive cues.

    Rules text mentions "evidence"/"reason"/"screen size" generically and was
    falsely flipping needs_justification / gate_attr / ranking_like.
    """
    if not query:
        return ""
    parts = re.split(r"\n\s*Task rules:\s*", query, maxsplit=1, flags=re.I)
    return parts[0].strip()


_BRAND_STOPWORDS = frozenset(
    {
        "in-stock",
        "instock",
        "gaming",
        "full",
        "new",
        "an",
        "a",
        "the",
        "any",
        "my",
        "our",
        "this",
        "that",
        "one",
        "some",
        "good",
        "best",
        "another",
        "other",
        "same",
        "such",
        "only",
        "also",
        "just",
        "with",
        "from",
        "into",
        "over",
        "under",
        "black",
        "white",
        "green",
        "blue",
        "red",
        "gold",
        "silver",
        "pink",
        "gray",
        "grey",
    }
)


def _clean_brand_token(raw: str | None) -> str | None:
    if not raw:
        return None
    text = raw.strip().strip("\"'")
    if not text or text.lower() in _BRAND_STOPWORDS:
        return None
    parts = text.split()
    while parts and parts[0].lower() in {"the", "a", "an"}:
        parts = parts[1:]
    text = " ".join(parts).strip()
    if not text or text.lower() in _BRAND_STOPWORDS:
        return None
    return text


def _analyze_query(query: str) -> dict[str, Any]:
    """Deterministic cues derived only from the current problem query."""
    shopper = _shopper_goal_text(query)
    shop_low = shopper.lower()
    variants = [
        {"product_id": m.group(1), "sku": m.group(2)} for m in _VARIANT_RE.finditer(shopper)
    ]
    budgets = [float(x) for x in _BUDGET_RE.findall(shopper)]
    brand = None
    bm = _SOFT_BRAND_RE.search(shopper)
    if bm:
        brand = _clean_brand_token(bm.group(1))
    brand_is_firm = False
    # Justification goals often name the brand as a hard requirement.
    if not brand:
        nm = _NAMED_BRAND_RE.search(shopper)
        if nm:
            cand = _clean_brand_token(nm.group(1))
            if cand:
                brand = cand
                brand_is_firm = True
    elif any(
        k in shop_low
        for k in ("why", "reason", "walk me through", "tell me why", "justif")
    ):
        # Soft-prefer language still present, but justification tasks with an
        # explicit named brand in the ask should treat it as selection-critical.
        nm = _NAMED_BRAND_RE.search(shopper)
        if nm:
            cand = _clean_brand_token(nm.group(1))
            if cand and cand.lower() == brand.lower():
                brand_is_firm = True
    ranking_like = any(
        k in shop_low
        for k in (
            "compare only these candidate",
            "highest refresh",
            "among ties",
            "strictly worse",
            "tradeoff priority",
        )
    ) or (
        bool(variants)
        and any(k in shop_low for k in ("compare only", "tradeoff", "priority", "among ties"))
    )
    # Only the shopper ask can require justification — not Task rules text.
    needs_justification = any(
        k in shop_low
        for k in (
            "why",
            "reason",
            "tell me why",
            "walk me through",
            "justif",
            "evidence",
        )
    )
    recovery_like = any(
        k in shop_low
        for k in (
            "prices and stock can change",
            "repriced",
            "switch to another",
            "recovery",
            "if your pick",
        )
    )
    preference_like = any(
        k in shop_low
        for k in (
            "soft preference",
            "nice-to-have",
            "nice to have",
            "prefer",
            "weighted",
            "trade-off among soft",
        )
    ) and not needs_justification and not ranking_like
    gate_attr = None
    if not ranking_like and not recovery_like:
        # Ranking uses these as lex objectives, not firm equality gates.
        # Recovery must not inherit equality gates from incidental size/Hz prose.
        if re.search(
            r"\b\d+(?:\.\d+)?\s*-?\s*inch|\bscreen\s*size\b|\b\d+(?:\.\d+)?in\b",
            shop_low,
        ):
            gate_attr = "screen_size"
        elif re.search(
            r"\b\d+p\b|\bfhd\b|\b4k\b|\buhd\b|\bresolution\b|\b720p\b|\b1080p\b",
            shop_low,
        ):
            gate_attr = "resolution"
        elif re.search(r"\brefresh\s*rate\b|\b\d+\s*hz\b", shop_low):
            gate_attr = "refresh_rate"
    # Pure ranking candidate-list tasks omit justification even if rules mention it.
    if ranking_like:
        needs_justification = False
        brand_is_firm = False
    elif recovery_like:
        brand_is_firm = False
        needs_justification = False
    elif needs_justification:
        ranking_like = False
        if brand and not brand_is_firm:
            # Named-brand justification asks: brand is required for gold.
            nm = _NAMED_BRAND_RE.search(shopper)
            if nm:
                cand = _clean_brand_token(nm.group(1))
                if cand:
                    brand_is_firm = True
                    brand = cand
    return {
        "candidate_variants": variants,
        "budget": min(budgets) if budgets else None,
        "preferred_brand": brand,
        "brand_is_firm": brand_is_firm,
        "ranking_like": ranking_like,
        "needs_justification": needs_justification,
        "recovery_like": recovery_like,
        "preference_like": preference_like,
        "gate_attr": gate_attr,
        "hard_looking": ranking_like
        or needs_justification
        or recovery_like
        or len(variants) >= 2
        or ("firm" in shop_low and "soft" in shop_low),
    }


def _strategy_brief(analysis: dict[str, Any], tool_names: list[str]) -> str:
    lines = [
        "Episode strategy brief (derived from this problem only):",
        f"- tools: {', '.join(tool_names) or '(see schemas)'}",
    ]
    if analysis["budget"] is not None:
        lines.append(f"- detected budget ceiling cue: {analysis['budget']}")
    if analysis["preferred_brand"]:
        lines.append(
            f"- soft preferred brand cue: {analysis['preferred_brand']} "
            "(honor only when firm-compliant)"
        )
    if analysis["candidate_variants"]:
        keys = [f"{v['product_id']}::{v['sku']}" for v in analysis["candidate_variants"]]
        lines.append("- explicit candidate variants in the goal (restrict to these):")
        for k in keys:
            lines.append(f"  * {k}")
    if analysis.get("gate_attr"):
        lines.append(
            f"- firm gated attribute cue: {analysis['gate_attr']} "
            "(must be satisfied and, if justifying, claimed canonically)"
        )
    if analysis["needs_justification"]:
        lines.append(
            "- justification required: place_test_order.justification MUST include "
            "price + in_stock + gated attribute + at least one truthful soft claim "
            "(prefer observed brand; else streaming/microphone/privacy_cover/"
            "autofocus/hdr only when evidenced). Never invent soft claims."
        )
        if analysis.get("preferred_brand"):
            lines.append(
                f"- named brand in goal: {analysis['preferred_brand']} "
                "(treat as firm for selection when justification is required)"
            )
    if analysis["ranking_like"]:
        lines.append(
            "- ranking/tradeoff mode: view/compare candidates, apply lexicographic "
            "priorities, commit static winner to cart first, re-rank after any "
            "environment event, then order the frontier winner. Omit justification."
        )
    if analysis["recovery_like"]:
        lines.append(
            "- recovery-sensitive: re-check inspect_stock after cart changes / shopper messages."
        )
    # Suggested default workflow using whatever tools exist.
    flow = []
    if analysis["candidate_variants"] and "view" in tool_names:
        flow.append("view each listed candidate (batch if max_calls allows)")
    elif "filter" in tool_names or "search" in tool_names:
        flow.append("filter/search with firm constraints")
    if "compare" in tool_names:
        flow.append("compare shortlist")
    if "inspect_stock" in tool_names:
        flow.append("inspect_stock on chosen variant")
    if "add_to_cart" in tool_names:
        flow.append("add_to_cart")
    if "place_test_order" in tool_names:
        flow.append("place_test_order (with justification if required)")
    if flow:
        lines.append("- suggested workflow: " + " → ".join(flow))
    return "\n".join(lines)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("function", {}).get("arguments", "{}")
    if isinstance(raw, dict):
        parsed = raw
    else:
        try:
            parsed = json.loads(raw or "{}")
        except Exception:
            # Recover malformed JSON arguments — common LLM failure mode.
            repaired = re.sub(r",\s*}", "}", str(raw))
            repaired = re.sub(r",\s*]", "]", repaired)
            try:
                parsed = json.loads(repaired)
            except Exception as exc:
                raise ValueError(f"invalid tool arguments: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _inference(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float,
    prefer_strong: bool,
    tool_choice: str | None,
    deadline: float | None = None,
) -> dict[str, Any]:
    last_err: Exception | None = None
    payload_base: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if tools is not None:
        payload_base["tools"] = tools
        if tool_choice:
            payload_base["tool_choice"] = tool_choice

    for model in _model_chain(prefer_strong=prefer_strong):
        if deadline is not None and time.monotonic() >= deadline:
            break
        for attempt in range(2):  # fail over models quickly; do not abort the episode
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                resp = _proxy.post(
                    "/inference/chat/completions",
                    json_data={**payload_base, "model": model},
                )
                if resp is None:
                    raise RuntimeError("inference returned None")
                if not resp.get("choices"):
                    raise RuntimeError(f"empty choices from {model}")
                return resp
            except Exception as exc:
                last_err = exc
                msg = str(exc).lower()
                delay = 0.25 * (attempt + 1)
                if "429" in msg or "rate" in msg:
                    delay = 0.8 * (attempt + 1)
                LOGGER.warning("inference fail model=%s attempt=%s: %s", model, attempt, exc)
                time.sleep(delay)
    raise RuntimeError(f"all inference models failed: {last_err}")


def _plan_episode(
    query: str,
    tool_names: list[str],
    prefer_strong: bool,
) -> dict[str, Any] | None:
    """Nimbus-style decompose/plan step via JSON LLM (no environment tools)."""
    messages = [
        {
            "role": "user",
            "content": _PLAN_PROMPT.format(
                tools=", ".join(tool_names) or "(none listed)",
                query=query,
            ),
        }
    ]
    try:
        resp = _inference(
            messages=messages,
            tools=None,
            temperature=0.0,
            prefer_strong=prefer_strong,
            tool_choice=None,
        )
        content = resp["choices"][0]["message"].get("content") or ""
        return _parse_json_object(content)
    except Exception as exc:
        LOGGER.warning("planning step skipped: %s", exc)
        return None


def _observation_done(result: dict[str, Any]) -> bool:
    for call in result.get("calls") or []:
        obs = call.get("observation") or {}
        if obs.get("done") is True:
            return True
    return False


def _nudge_for_observation(
    turn: int,
    max_steps: int,
    analysis: dict[str, Any],
    last_actions: list[str],
    last_obs_blob: str,
    pending_intervention: str | None,
) -> str | None:
    """Low-variance recovery nudges based on observations + remaining budget."""
    if pending_intervention:
        return (
            "SHOPPER INTERVENTION PENDING. Your NEXT action MUST include the `message` "
            "tool answering them briefly and concretely (confirm the pick + firm specs / "
            "acknowledge budget or price changes). Then continue with tools.\n"
            f"Shopper said: {pending_intervention[:600]}"
        )
    remaining = max_steps - turn
    low = (last_obs_blob or "").lower()
    empty = any(
        s in low
        for s in (
            '"products": []',
            '"results": []',
            '"candidates": []',
            "no products",
            "0 products",
            "not found",
            "out of stock",
            "out_of_stock",
        )
    )
    if empty and remaining > 2:
        return (
            "Previous retrieval looked empty or invalid. Reformulate: shorten keywords, "
            "relax soft preferences only, keep firm constraints, and try filter/search again."
        )
    if analysis["ranking_like"] and turn <= 3 and not any(
        a in ("view", "compare") for a in last_actions
    ):
        return (
            "Ranking task: first view/compare the listed candidate variants before cart/order."
        )
    if analysis["ranking_like"] and "add_to_cart" not in last_actions and turn >= 3:
        return (
            "Ranking task: FIRST add_to_cart MUST be the current lexicographic winner "
            "(static rank-1). Do not cart a dominated/secondary option first. "
            "After any later market event, re-inspect_stock ALL candidates, then switch "
            "cart to the NEW lex winner and order that — never revert to the stale pick."
        )
    if "repriced" in low or "price changed" in low or "no longer" in low or "out of stock" in low:
        return (
            "Environment change detected. Re-verify stock/price with inspect_stock on "
            "every candidate still under consideration, then switch cart to the NEW "
            "lex/firm winner if the previous pick is invalid. Do not order a dominated "
            "or stale pre-event pick after the frontier moved."
        )
    if remaining <= 4 and "place_test_order" not in last_actions:
        return (
            "Step budget is low. If you have a verified firm-compliant in-stock variant, "
            "inspect_stock if needed, ensure it is in cart, then place_test_order now."
        )
    return None


def _user_message_text(result: dict[str, Any]) -> str | None:
    um = result.get("user_message")
    if not um:
        return None
    if isinstance(um, dict):
        content = um.get("content")
        return str(content) if content is not None else None
    return str(um)


def _draft_intervention_reply(shopper_text: str, query: str) -> str:
    """Short deterministic reply; keeps variance low vs another LLM call."""
    low = shopper_text.lower()
    if "balance" in low or "budget" in low or "up to" in low:
        return (
            "Understood — I will apply your updated budget immediately and only keep "
            "options that still meet every firm requirement (including the gated spec) "
            "and current in-stock price."
        )
    if "more expensive" in low or "re-check" in low or "repriced" in low:
        return (
            "Thanks for the heads-up. I will re-check the current price and stock with "
            "inspect_stock and switch to another firm-compliant in-budget option if needed."
        )
    if "which one" in low or "going with" in low or "non-negotiable" in low:
        return (
            "I am verifying candidates against your firm requirements now (gated spec, "
            "budget, in-stock). I will message the specific pick with the confirming facts "
            "before ordering."
        )
    if "compare" in low or "requirements" in low:
        return (
            "Acknowledged. I will stick to your stated firm requirements and compare the "
            "current compliant options before placing any order."
        )
    # Generic fallback grounded in the original goal's firmness language.
    return (
        "Acknowledged. I will follow your latest instruction while preserving every firm "
        "requirement from your goal, re-verify live price/stock, then proceed."
    )


def _ensure_message_action(
    tool_calls: list[dict[str, Any]],
    *,
    pending_intervention: str,
    query: str,
    tool_names: list[str],
    max_calls: int,
    turn: int,
) -> list[dict[str, Any]]:
    if "message" not in tool_names:
        return tool_calls
    if any(
        (c.get("function") or {}).get("name") == "message" for c in tool_calls
    ):
        return tool_calls[:max_calls]
    reply = _draft_intervention_reply(pending_intervention, query)
    synthetic = {
        "id": f"nimbus-intervening-msg-{turn}",
        "type": "function",
        "function": {
            "name": "message",
            "arguments": json.dumps({"content": reply}, ensure_ascii=False),
        },
    }
    return ([synthetic] + tool_calls)[:max_calls]


def _split_variant_key(key: str | None) -> tuple[str, str] | None:
    if not key or "::" not in key:
        return None
    pid, sku = key.split("::", 1)
    if not pid or not sku:
        return None
    return pid, sku


def _rewrite_call_args(call: dict[str, Any], name: str, args: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function") or {}
    return {
        **call,
        "function": {
            **fn,
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def _format_price_claim(price: float) -> str:
    # Match cents exactly; prefer compact form without trailing .0 noise.
    cents = round(float(price) * 100)
    if cents % 100 == 0:
        return str(cents // 100)
    return f"{cents / 100:.2f}".rstrip("0").rstrip(".")


def _normalize_claim_value(field: str, value: Any) -> str | None:
    if value is None:
        return None
    val = str(value).strip()
    if not val:
        return None
    if field == "in_stock":
        low = val.lower()
        if low in ("true", "1", "yes"):
            return "true"
        if low in ("false", "0", "no"):
            return "false"
        return None
    if field == "screen_size":
        m = re.search(r"(\d+(?:\.\d+)?)\s*(in|inch|inches)?", val, re.I)
        if m:
            num = float(m.group(1))
            return f"{num:g}in"
        return None
    if field == "refresh_rate":
        m = re.search(r"(\d+(?:\.\d+)?)", val)
        return m.group(1) if m else None
    if field == "resolution":
        return _norm_res(val)
    if field == "price":
        m = re.search(r"(\d+(?:\.\d+)?)", val.replace(",", ""))
        return m.group(1) if m else None
    if field in _SOFT_FEATURE_PATTERNS:
        low = val.lower()
        if low in ("true", "1", "yes"):
            return "true"
        return None
    if field == "brand":
        return val
    return val


def _mine_soft_features(*texts: Any) -> set[str]:
    """Positive soft-feature hits from observed text (no hallucination)."""
    blob = " ".join(str(t) for t in texts if t)
    if not blob.strip():
        return set()
    found: set[str] = set()
    for field, pat in _SOFT_FEATURE_PATTERNS.items():
        for match in pat.finditer(blob):
            before = re.split(r"[.;!?\n]", blob[: match.start()])[-1]
            after = re.split(r"[.;!?\n]", blob[match.end() :])[0]
            negative_before = re.search(
                r"\b(?:no|without|lack(?:s|ing)?|missing|exclude[ds]?|unsupported|not|"
                r"cannot|can['’]t|doesn['’]t|optional)\b(?:[\s-]+[\w'’]+){0,5}[\s-]*$",
                before,
                re.I,
            )
            negative_after = re.match(
                r"\s*(?:(?:support|feature|function|mode|capability)\s*)?[:=]?\s*"
                r"(?:(?:is|are|was|were)\s+)?"
                r"(?:not|no|false|none|unsupported|absent|unavailable|missing|excluded|"
                r"disabled|optional|0)\b",
                after,
                re.I,
            )
            if negative_before or negative_after:
                continue
            found.add(field)
            break
    return found


def _verified_brand(rec: dict[str, Any]) -> str | None:
    """Structured catalog brand only. Title lookalikes are NEVER brand evidence."""
    brand = rec.get("brand")
    if brand is None:
        return None
    text = str(brand).strip()
    return text or None


def _verified_soft_claim(rec: dict[str, Any]) -> dict[str, str] | None:
    """One soft claim from verified brand or evidenced soft features (not title-brand)."""
    brand = _verified_brand(rec)
    if brand:
        return {"field": "brand", "value": brand}
    # Soft features may be mined from facts (preferred) or title (env scorer allows
    # title support for feature fields — never for brand).
    features = set(rec.get("soft_features_facts") or set()) | set(
        rec.get("soft_features") or set()
    )
    for field in _SOFT_CLAIM_PRIORITY:
        if field == "brand":
            continue
        if field in features:
            return {"field": field, "value": "true"}
    return None


def _claim_verified_against_rec(
    field: str,
    value: str,
    rec: dict[str, Any],
    *,
    gate_attr: str | None,
    gate_value: Any,
) -> bool:
    """Factual firewall: every claim must match structured / allowed evidence."""
    if field == "price":
        price = rec.get("price")
        if price is None:
            return False
        try:
            return abs(float(value) - float(price)) < 0.011
        except Exception:
            return False
    if field == "in_stock":
        return value == "true" and rec.get("in_stock") is True
    if field == "brand":
        actual = _verified_brand(rec)
        return bool(actual) and actual.casefold() == value.casefold()
    if field in _SOFT_FEATURE_PATTERNS:
        features = set(rec.get("soft_features_facts") or set()) | set(
            rec.get("soft_features") or set()
        )
        return value == "true" and field in features
    if field == "screen_size":
        size = rec.get("screen_size_fact")
        if size is None:
            return False
        want = _normalize_claim_value("screen_size", f"{float(size):g}in")
        return want == value and (
            gate_value is None or abs(float(size) - float(gate_value)) <= 0.05
        )
    if field == "resolution":
        res = rec.get("resolution_fact")
        if not res:
            return False
        want = _norm_res(gate_value) if gate_value is not None else None
        return res == value and (want is None or res == want)
    if field == "refresh_rate":
        hz = rec.get("refresh_rate")
        if hz is None:
            return False
        try:
            ok = abs(float(value) - float(hz)) < 0.51
        except Exception:
            return False
        if gate_value is not None:
            ok = ok and abs(float(hz) - float(gate_value)) < 0.51
        return ok
    return False


def _build_justification_claims(
    scorer: "ObservationScorer",
    *,
    product_id: Any,
    sku: Any,
    gate_attr: str | None,
) -> list[dict[str, str]] | None:
    """Build hard trio + one VERIFIED soft claim (never title-inferred brand)."""
    if product_id is None or sku is None:
        return None
    key = f"{product_id}::{sku}"
    rec = scorer.candidates.get(key)
    if not rec:
        return None
    if not scorer.is_committable(rec):
        return None
    claims: list[dict[str, str]] = []
    price = rec.get("price")
    if price is None:
        return None
    claims.append({"field": "price", "value": _format_price_claim(float(price))})
    if rec.get("in_stock") is not True:
        return None
    claims.append({"field": "in_stock", "value": "true"})

    if gate_attr == "screen_size":
        size = rec.get("screen_size_fact")
        if size is None:
            return None
        if scorer.gate_value is not None and abs(float(size) - float(scorer.gate_value)) > 0.05:
            return None
        claims.append({"field": "screen_size", "value": f"{float(size):g}in"})
    elif gate_attr == "resolution":
        res = rec.get("resolution_fact")
        if not res:
            return None
        want = _norm_res(scorer.gate_value) if scorer.gate_value else None
        if want and res != want:
            return None
        claims.append({"field": "resolution", "value": str(res)})
    elif gate_attr == "refresh_rate" and rec.get("refresh_rate") is not None:
        if scorer.gate_value is not None and abs(
            float(rec["refresh_rate"]) - float(scorer.gate_value)
        ) > 0.5:
            return None
        claims.append(
            {"field": "refresh_rate", "value": f"{float(rec['refresh_rate']):g}"}
        )
    else:
        return None

    soft = _verified_soft_claim(rec)
    if soft is None:
        return None
    claims.append(soft)
    return claims


def _align_ranking_tool_calls(
    tool_calls: list[dict[str, Any]],
    scorer: "ObservationScorer",
) -> list[dict[str, Any]]:
    """Force ranking cart/order onto the current lex winner; strip justifications.

    Pre-event: first commitment should be the static lex winner.
    Post-event: never order the stale pre-event pick once a new winner exists.
    """
    if not scorer.analysis.get("ranking_like"):
        return tool_calls
    winner_key = scorer.current_lex_winner_key()
    winner = _split_variant_key(winner_key)
    if not winner:
        # Still strip justifications on ranking orders.
        out: list[dict[str, Any]] = []
        for call in tool_calls:
            fn = call.get("function") or {}
            if fn.get("name") != "place_test_order":
                out.append(call)
                continue
            try:
                args = _arguments(call)
            except Exception:
                out.append(call)
                continue
            args.pop("justification", None)
            out.append(_rewrite_call_args(call, "place_test_order", args))
        return out

    w_pid, w_sku = winner
    event_seen = bool(scorer.event_seen)
    static_key = scorer.static_committed_key
    ready = scorer.ranking_ready() or event_seen
    post_ok = scorer.post_event_ready_to_commit()
    pending = scorer.pending_post_event_inspect()
    out = []
    for call in tool_calls:
        fn = call.get("function") or {}
        name = fn.get("name")
        if name not in ("add_to_cart", "place_test_order", "remove_from_cart"):
            out.append(call)
            continue
        try:
            args = dict(_arguments(call))
        except Exception:
            out.append(call)
            continue
        if name == "place_test_order":
            args.pop("justification", None)
            if event_seen and not post_ok and pending:
                # Premature order after event → force another inspect_stock first.
                pk = pending[0]
                sp = _split_variant_key(pk)
                if sp:
                    out.append(
                        _rewrite_call_args(
                            call, "inspect_stock", {"product_id": sp[0], "sku": sp[1]}
                        )
                    )
                    continue
            if ready and post_ok:
                args["product_id"] = w_pid
                args["sku"] = w_sku
            out.append(_rewrite_call_args(call, name, args))
            continue
        if name == "add_to_cart":
            if event_seen and not post_ok and pending:
                pk = pending[0]
                sp = _split_variant_key(pk)
                if sp:
                    out.append(
                        _rewrite_call_args(
                            call, "inspect_stock", {"product_id": sp[0], "sku": sp[1]}
                        )
                    )
                    continue
            if ready and post_ok:
                args["product_id"] = w_pid
                args["sku"] = w_sku
            elif ready and not event_seen:
                args["product_id"] = w_pid
                args["sku"] = w_sku
            out.append(_rewrite_call_args(call, name, args))
            continue
        # remove_from_cart: if model removes the live winner post-event, redirect
        # removal toward a non-winner (static commitment) when known.
        cur = f"{args.get('product_id')}::{args.get('sku')}"
        if event_seen and post_ok and cur == winner_key and static_key and static_key != winner_key:
            sp = _split_variant_key(static_key)
            if sp:
                args["product_id"], args["sku"] = sp
        out.append(_rewrite_call_args(call, name, args))
    return out


def _next_search_args(scorer: "ObservationScorer") -> dict[str, Any]:
    """Reformulate search when commitment is blocked (rotate to avoid loops)."""
    n = getattr(scorer, "explore_attempts", 0)
    parts: list[str] = []
    brand = _clean_brand_token(scorer.preferred_brand)
    if brand and n % 4 != 2:
        parts.append(brand)
    if scorer.category_hint and n % 5 != 3:
        parts.append(scorer.category_hint.rstrip("s"))
    if scorer.gate_attr == "resolution" and scorer.gate_value:
        # Alternate HD synonyms so search escapes filter echo chambers.
        res = str(scorer.gate_value).lower()
        alts = {
            "hd": ["720p", "HD", "720p HD"],
            "fhd": ["1080p", "Full HD", "FHD"],
            "4k": ["4K", "UHD", "2160p"],
        }
        choices = alts.get(res, [str(scorer.gate_value)])
        parts.append(choices[n % len(choices)])
    elif scorer.gate_attr == "screen_size" and scorer.gate_value is not None:
        parts.append(f"{float(scorer.gate_value):g} inch")
    elif scorer.gate_attr == "refresh_rate" and scorer.gate_value is not None:
        parts.append(f"{float(scorer.gate_value):g}Hz")
    if scorer.budget is not None and n % 3 != 1:
        parts.append(f"under {float(scorer.budget):g}")
    # Diversify via synonym rotation / optional brand omission above — never
    # inject brand-family tokens (e.g. "Galaxy") that only fit one vendor.
    query = " ".join(parts).strip() or (scorer.query.split("\n")[0][:120])
    return {"query": query, "in_stock": True}


def _next_explore_call(scorer: "ObservationScorer") -> tuple[str, dict[str, Any]]:
    """Rotate filter/search/view so recovery does not spam one filter forever."""
    scorer.explore_attempts = getattr(scorer, "explore_attempts", 0) + 1
    n = scorer.explore_attempts
    # Prefer viewing promising unviewed hits before another filter echo.
    for rec in scorer.ranked():
        if rec.get("viewed"):
            continue
        if rec["key"] in scorer.rejected_keys:
            continue
        if scorer._verified_brand_hit(rec) or (
            not scorer.brand_is_firm and scorer._title_brand_hint(rec)
        ) or (scorer.brand_is_firm and scorer._title_brand_hint(rec) and n >= 2):
            return "view", {"product_id": rec["product_id"]}
    brand = _clean_brand_token(scorer.preferred_brand)
    # Prefer filter early; after a few misses, force diversified search.
    if brand and scorer.brand_is_firm and n <= 2:
        args: dict[str, Any] = {
            "brand": brand,
            "in_stock": True,
        }
        if scorer.budget is not None:
            args["max_price"] = float(scorer.budget)
        if scorer.category_hint and n == 1:
            args["category"] = scorer.category_hint
        return "filter", args
    return "search", _next_search_args(scorer)


def _align_justification_tool_calls(
    tool_calls: list[dict[str, Any]],
    scorer: "ObservationScorer",
    *,
    needs_justification: bool,
    ranking_like: bool = False,
) -> list[dict[str, Any]]:
    """VERIFY → COMMIT gate for cart/order (TF7 + gated TF1/TF3; not recovery/rank)."""
    if ranking_like or scorer.analysis.get("recovery_like"):
        return tool_calls
    # Preference / open recall: do not rewrite first-cart commitments.
    if scorer.analysis.get("preference_like") and not needs_justification:
        return tool_calls
    # Strict path: justification always; other families only when a firm gate exists.
    if not needs_justification and not scorer.gate_attr:
        return tool_calls
    # VERIFY → COMMIT: only candidates that pass required attrs (+ soft when needed).
    ready = [
        r
        for r in scorer.ranked()
        if scorer.is_committable(r) and r["key"] not in scorer.rejected_keys
    ]
    best = ready[0] if ready else None

    def _blocked_explore(call: dict[str, Any]) -> dict[str, Any]:
        name, args = _next_explore_call(scorer)
        return _rewrite_call_args(call, name, args)

    out: list[dict[str, Any]] = []
    for call in tool_calls:
        fn = call.get("function") or {}
        name = fn.get("name")
        if name not in ("add_to_cart", "place_test_order", "remove_from_cart"):
            out.append(call)
            continue
        try:
            args = dict(_arguments(call))
        except Exception:
            out.append(call)
            continue

        if name == "add_to_cart":
            cur_key = f"{args.get('product_id')}::{args.get('sku')}"
            # Never commit a non-verified / previously rejected candidate.
            if cur_key in scorer.rejected_keys or (
                cur_key in scorer.candidates
                and not scorer.is_committable(scorer.candidates[cur_key])
            ):
                if cur_key in scorer.candidates and not scorer.is_committable(
                    scorer.candidates[cur_key]
                ):
                    scorer.reject_candidate(
                        cur_key,
                        scorer._firm_failures(scorer.candidates[cur_key])
                        or ["unverified"],
                    )
                if best:
                    args["product_id"] = best["product_id"]
                    args["sku"] = best["sku"]
                else:
                    out.append(_blocked_explore(call))
                    continue
            elif best and (
                scorer.static_committed_key is None
                or (
                    scorer.event_seen
                    and scorer.static_committed_key
                    and scorer.static_committed_key != best["key"]
                )
            ):
                args["product_id"] = best["product_id"]
                args["sku"] = best["sku"]
            # Revalidate after price-change before allowing a new cart add.
            if scorer.event_seen and best:
                if not scorer.is_committable(best):
                    out.append(_blocked_explore(call))
                    continue
                if args.get("product_id") == best["product_id"]:
                    scorer.post_event_carted = True
            out.append(_rewrite_call_args(call, name, args))
            continue

        if name == "place_test_order":
            pending = scorer.pending_post_event_inspect()
            if scorer.event_seen and pending:
                sp = _split_variant_key(pending[0])
                if sp:
                    out.append(
                        _rewrite_call_args(
                            call, "inspect_stock", {"product_id": sp[0], "sku": sp[1]}
                        )
                    )
                    continue
            # One-shot removal of invalidated first cart after a price/stock event.
            if (
                scorer.event_seen
                and not scorer.stale_cart_cleared
                and scorer.static_committed_key
                and best
                and scorer.static_committed_key != best["key"]
            ):
                sp = _split_variant_key(scorer.static_committed_key)
                if sp:
                    scorer.stale_cart_cleared = True
                    scorer.reject_candidate(
                        scorer.static_committed_key, ["price_or_stock_event"]
                    )
                    out.append(
                        _rewrite_call_args(
                            call,
                            "remove_from_cart",
                            {"product_id": sp[0], "sku": sp[1]},
                        )
                    )
                    continue
            # After clearing stale cart, commit a revalidated replacement before order.
            if (
                scorer.event_seen
                and best
                and scorer.stale_cart_cleared
                and not scorer.post_event_carted
            ):
                if not scorer.is_committable(best):
                    out.append(_blocked_explore(call))
                    continue
                scorer.post_event_carted = True
                out.append(
                    _rewrite_call_args(
                        call,
                        "add_to_cart",
                        {"product_id": best["product_id"], "sku": best["sku"]},
                    )
                )
                continue
            if best and scorer.is_committable(best):
                args["product_id"] = best["product_id"]
                args["sku"] = best["sku"]
                out.append(_rewrite_call_args(call, name, args))
                continue
            # No verified commitment — do not order title-decoy / unverified items.
            out.append(_blocked_explore(call))
            continue

        if name == "remove_from_cart":
            scorer.stale_cart_cleared = True
            cur = f"{args.get('product_id')}::{args.get('sku')}"
            if cur:
                scorer.reject_candidate(cur, ["removed"])
        out.append(_rewrite_call_args(call, name, args))
    return out


def _sanitize_tool_calls(
    tool_calls: list[dict[str, Any]],
    *,
    needs_justification: bool,
    ranking_like: bool,
    gate_attr: str | None,
    scorer: "ObservationScorer | None" = None,
) -> list[dict[str, Any]]:
    """Deterministic factual firewall before /environment/call."""
    cleaned: list[dict[str, Any]] = []
    soft_allowed = set(_SOFT_CLAIM_PRIORITY) | {"brand"}
    for call in tool_calls:
        fn = call.get("function") or {}
        name = fn.get("name")
        if name != "place_test_order":
            cleaned.append(call)
            continue
        try:
            args = _arguments(call)
        except Exception:
            cleaned.append(call)
            continue
        # Ranking candidate-list tasks: never send justification.
        if ranking_like or not needs_justification:
            args.pop("justification", None)
            cleaned.append(_rewrite_call_args(call, "place_test_order", args))
            continue

        # Justification path: ONLY verified evidence may appear. Never preserve
        # model-generated soft claims (e.g. brand from title) when rebuild fails.
        rebuilt: list[dict[str, str]] | None = None
        if scorer is not None:
            rebuilt = _build_justification_claims(
                scorer,
                product_id=args.get("product_id"),
                sku=args.get("sku"),
                gate_attr=gate_attr,
            )
        if rebuilt:
            args["justification"] = {"claims": rebuilt}
            cleaned.append(_rewrite_call_args(call, "place_test_order", args))
            continue

        # Fallback: filter model claims against structured evidence, then ensure
        # soft claim from verified sources only.
        just = args.get("justification")
        safe: list[dict[str, str]] = []
        rec: dict[str, Any] = {}
        if scorer is not None:
            key = f"{args.get('product_id')}::{args.get('sku')}"
            rec = scorer.candidates.get(key) or {}
        gate_value = scorer.gate_value if scorer is not None else None
        if isinstance(just, dict) and isinstance(just.get("claims"), list) and rec:
            seen: set[str] = set()
            for cl in just["claims"]:
                if not isinstance(cl, dict):
                    continue
                field = str(cl.get("field") or "").strip()
                if field == "category" or field in seen:
                    continue
                val = _normalize_claim_value(field, cl.get("value"))
                if val is None:
                    continue
                if not _claim_verified_against_rec(
                    field, val, rec, gate_attr=gate_attr, gate_value=gate_value
                ):
                    continue
                safe.append({"field": field, "value": val})
                seen.add(field)
            has_soft = any(c["field"] in soft_allowed for c in safe)
            if not has_soft:
                soft = _verified_soft_claim(rec)
                if soft and soft["field"] not in seen:
                    safe.append(soft)
            # Must still cover price + in_stock + gate + soft.
            needed = {"price", "in_stock"}
            if gate_attr:
                needed.add(gate_attr)
            fields = {c["field"] for c in safe}
            if needed <= fields and any(f in soft_allowed for f in fields):
                args["justification"] = {"claims": safe}
                cleaned.append(_rewrite_call_args(call, "place_test_order", args))
                continue

        # Cannot assert verified claims — refuse to order with invented soft claims.
        if scorer is not None:
            key = f"{args.get('product_id')}::{args.get('sku')}"
            scorer.reject_candidate(key, ["justification_unverified"])
            name, explore_args = _next_explore_call(scorer)
            cleaned.append(_rewrite_call_args(call, name, explore_args))
        else:
            args.pop("justification", None)
            cleaned.append(_rewrite_call_args(call, "place_test_order", args))
    return cleaned


# ===================== observation-based candidate scorer =====================

def _unwrap_payload(obs: Any) -> dict[str, Any]:
    """Environment observations are often {done,error,observation:{...}}."""
    if not isinstance(obs, dict):
        return {}
    inner = obs.get("observation")
    if isinstance(inner, dict) and any(
        k in inner
        for k in ("results", "compared", "product_id", "facts", "variants", "sku", "price")
    ):
        return inner
    return obs


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(m.group(0)) if m else None


def _norm_res(value: Any) -> str | None:
    if value is None:
        return None
    low = re.sub(r"\s+", " ", str(value).lower()).strip()
    if not low:
        return None
    compact = low.replace(" ", "")
    if "2160" in compact or "4k" in compact or "uhd" in compact:
        return "4k"
    if "1440" in compact or "qhd" in compact:
        return "qhd"
    if "1080" in compact or "fhd" in compact or "fullhd" in compact:
        return "fhd"
    if "720" in compact or re.search(r"\bhd\b", low):
        return "hd"
    tok = _RESOLUTION_TOKEN_RE.search(low)
    return tok.group(1).lower() if tok else None


def _norm_screen_in(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _SCREEN_RE.search(str(value))
    if m:
        return float(m.group(1) or m.group(2))
    # bare number in facts like "27"
    m2 = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(m2.group(1)) if m2 else None


def _norm_refresh_hz(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Bare numbers are ambiguous (could be inches); only trust explicit Hz text
        # unless the value is clearly a refresh-like magnitude.
        num = float(value)
        return num if num >= 30 else None
    text = str(value).lower()
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*hz\b", text)]
    if nums:
        return max(nums)
    # Truncated catalog titles sometimes end with "200H" / "300H".
    nums_h = [float(x) for x in re.findall(r"(\d{2,3})\s*h(?:z)?\b", text)]
    nums_h = [n for n in nums_h if 30 <= n <= 600]
    if nums_h:
        return max(nums_h)
    if "refresh" in text:
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            num = float(m.group(1))
            return num if num >= 30 else None
    return None


def _truthy_stock(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    low = str(value).strip().lower()
    if low in ("true", "1", "yes", "in_stock", "in stock"):
        return True
    if low in ("false", "0", "no", "out_of_stock", "out of stock"):
        return False
    return None


def _parse_lex_priorities(query: str) -> list[tuple[str, str]]:
    """Return [(metric, direction)] in priority order. direction: max|min."""
    q = query.lower()
    found: list[tuple[str, str]] = []

    def add(metric: str, direction: str) -> None:
        if metric not in {m for m, _ in found}:
            found.append((metric, direction))

    # Order by first mention in the goal text.
    patterns = [
        (r"highest(?:\s+advertised)?(?:\s+maximum)?\s+refresh", "refresh_rate", "max"),
        (r"highest\s+refresh", "refresh_rate", "max"),
        (r"largest\s+screen", "screen_size", "max"),
        (r"lowest\s+price", "price", "min"),
        (r"cheapest", "price", "min"),
    ]
    hits: list[tuple[int, str, str]] = []
    for pat, metric, direction in patterns:
        m = re.search(pat, q)
        if m:
            hits.append((m.start(), metric, direction))
    hits.sort()
    for _, metric, direction in hits:
        add(metric, direction)
    if not found and "refresh" in q:
        add("refresh_rate", "max")
        add("screen_size", "max")
        add("price", "min")
    return found


def _extract_gate_value(query: str, gate_attr: str | None) -> Any:
    if not gate_attr:
        return None
    if gate_attr == "screen_size":
        m = _SCREEN_RE.search(query)
        if m:
            return float(m.group(1) or m.group(2))
    if gate_attr == "resolution":
        m = _RESOLUTION_TOKEN_RE.search(query)
        if m:
            return _norm_res(m.group(1))
    if gate_attr == "refresh_rate":
        m = _REFRESH_RE.search(query)
        if m:
            return float(m.group(1))
    return None


class ObservationScorer:
    """Nimbus-inspired deterministic scorer over live environment observations.

    Preserves the Weight-class idea (firm match first, soft preference second,
    rank/lex tie-breaks) but scores viewed/search facts from ORO Bench tools
    instead of ShoppingBench find_product pools.
    """

    # Small rank bonuses inspired by legacy Weight.RANK_BONUS_C / SKU_BONUS_C,
    # kept tiny so firm/soft gates dominate.
    _SEARCH_RANK_BONUS = [1e-2 - i * 2e-5 for i in range(80)]

    def __init__(self, analysis: dict[str, Any], query: str) -> None:
        self.query = query
        self.analysis = analysis
        self.budget = analysis.get("budget")
        self.preferred_brand = (analysis.get("preferred_brand") or "").strip() or None
        self.brand_is_firm = bool(analysis.get("brand_is_firm"))
        self.gate_attr = analysis.get("gate_attr")
        shopper = _shopper_goal_text(query)
        self.gate_value = _extract_gate_value(shopper, self.gate_attr)
        self.lex_priorities = _parse_lex_priorities(shopper)
        self.restrict_keys = {
            f"{v['product_id']}::{v['sku']}"
            for v in (analysis.get("candidate_variants") or [])
        }
        self.category_hint = self._infer_category_hint(shopper)
        # key -> candidate dict
        self.candidates: dict[str, dict[str, Any]] = {}
        # Ranking event state machine
        self.event_seen = False
        self.static_committed_key: str | None = None
        self.inspected_after_event: set[str] = set()
        self._pre_event_snapshot_winner: str | None = None
        self.stale_cart_cleared = False
        self.post_event_carted = False
        # key -> rejection reasons (avoid re-committing invalid candidates)
        self.rejected_keys: dict[str, list[str]] = {}
        self.rejection_reasons: dict[str, list[str]] = self.rejected_keys
        self.explore_attempts = 0
        self.viewed_after_reject: set[str] = set()

    def reject_candidate(self, key: str | None, reasons: list[str]) -> None:
        if not key:
            return
        prev = self.rejected_keys.setdefault(key, [])
        for reason in reasons:
            if reason not in prev:
                prev.append(reason)

    def is_committable(self, rec: dict[str, Any]) -> bool:
        """True only when required attrs + event conditions are evidence-backed.

        SEARCH HINT ≠ VERIFIED FACT. Title brand lookalikes never satisfy firm brand.
        """
        if not rec or rec.get("key") in self.rejected_keys:
            return False
        if not rec.get("viewed"):
            return False
        fails = self._firm_failures(rec)
        if fails:
            return False
        if rec.get("in_stock") is not True:
            return False
        if rec.get("price") is None:
            return False
        if not self.analysis.get("ranking_like"):
            # Gate attribute must be fact-backed before commitment (all non-ranking).
            if self.gate_attr == "screen_size" and rec.get("screen_size_fact") is None:
                return False
            if self.gate_attr == "resolution" and not rec.get("resolution_fact"):
                return False
            if self.analysis.get("needs_justification"):
                if _verified_soft_claim(rec) is None:
                    return False
        return True

    @staticmethod
    def _infer_category_hint(query: str) -> str | None:
        q = query.lower()
        for label, pats in (
            ("smartphones", (r"\bsmartphone\b", r"\bphone\b")),
            ("monitors", (r"\bmonitor\b",)),
            ("tablets", (r"\btablet\b",)),
            ("headphones", (r"\bheadphone\b", r"\bearbud\b", r"\bearphone\b")),
        ):
            if any(re.search(p, q) for p in pats):
                return label
        return None

    def update_budget(self, budget: float | None) -> None:
        if budget is not None:
            self.budget = float(budget)

    def note_market_event(self, text: str | None) -> bool:
        """Mark a price/stock market event from shopper/env text."""
        if not text:
            return False
        low = text.lower()
        triggers = (
            "re-check",
            "recheck",
            "repriced",
            "price changed",
            "more expensive",
            "still available",
            "out of stock",
            "no longer",
            "went out",
            "updated price",
            "price update",
            "env_signal",
            "forced_event",
            "stockout",
        )
        hit = any(t in low for t in triggers)
        # Apply explicit price-change magnitudes onto the committed candidate.
        m = re.search(
            r"(?:more expensive|repriced|changed).*?(\d+(?:\.\d+)?)\s*(?:php)?\s*to\s*(\d+(?:\.\d+)?)",
            low,
            re.I,
        )
        if not m:
            m = re.search(
                r"from\s+(\d+(?:\.\d+)?)\s*(?:php)?\s*to\s+(\d+(?:\.\d+)?)",
                low,
                re.I,
            )
        if m and self.static_committed_key:
            rec = self.candidates.get(self.static_committed_key)
            if rec is not None:
                rec["price"] = float(m.group(2))
                self.inspected_after_event.add(self.static_committed_key)
                hit = True
        if "out of stock" in low or "still available" in low:
            if self.static_committed_key:
                rec = self.candidates.get(self.static_committed_key)
                if rec is not None and ("out of stock" in low or "no longer" in low):
                    rec["in_stock"] = False
                    self.inspected_after_event.add(self.static_committed_key)
                    hit = True
        if hit:
            if not self.event_seen:
                self._pre_event_snapshot_winner = self.current_lex_winner_key()
            self.event_seen = True
            return True
        return False

    def note_cart_commit(self, product_id: Any, sku: Any) -> None:
        if product_id is None or sku is None:
            return
        key = self._key(product_id, sku)
        if self.static_committed_key is None:
            self.static_committed_key = key

    def current_lex_winner_key(self) -> str | None:
        ranked = self.ranked()
        if not ranked:
            return None
        # Prefer stock/budget-ok; ranked() already sorts hard failures last.
        return ranked[0].get("key")

    def ranking_ready(self) -> bool:
        """True once enough candidate facts exist to lock a static rank-1 cart."""
        if not self.restrict_keys:
            return any(c.get("viewed") or c.get("price") is not None for c in self.candidates.values())
        ready = 0
        for key in self.restrict_keys:
            rec = self.candidates.get(key) or {}
            if rec.get("viewed") or rec.get("price") is not None or rec.get("refresh_rate") is not None:
                ready += 1
        return ready >= max(2, len(self.restrict_keys) - 1)

    def post_event_ready_to_commit(self) -> bool:
        """After a market event, require re-inspect before forcing cart/order."""
        if not self.event_seen:
            return True
        if self.static_committed_key and self.static_committed_key not in self.inspected_after_event:
            return False
        winner = self.current_lex_winner_key()
        if self.restrict_keys:
            # Ranking-style closed candidate set: require winner re-checked.
            if winner and winner not in self.inspected_after_event:
                return False
            return True
        # Open-catalog (TF7): re-inspect of the invalidated commitment is enough;
        # the replacement is validated via firm_ok + its own inspect/view evidence.
        return True

    def pending_post_event_inspect(self) -> list[str]:
        if not self.event_seen:
            return []
        if self.restrict_keys:
            return sorted(self.restrict_keys - self.inspected_after_event)
        if self.static_committed_key and self.static_committed_key not in self.inspected_after_event:
            return [self.static_committed_key]
        return []

    def ingest_observation(self, raw_obs: Any, *, search_rank_base: int = 0) -> int:
        """Ingest one tool observation. Returns number of candidate rows touched."""
        payload = _unwrap_payload(raw_obs)
        if not payload:
            return 0
        n = 0
        if isinstance(payload.get("results"), list):
            for i, row in enumerate(payload["results"]):
                if isinstance(row, dict):
                    n += int(self._upsert_search_row(row, rank=search_rank_base + i))
        if isinstance(payload.get("compared"), list):
            for row in payload["compared"]:
                if isinstance(row, dict):
                    n += self._upsert_view_like(row)
        if "product_id" in payload and (
            "facts" in payload or "variants" in payload or "sku" in payload
        ):
            # view or inspect_stock style
            if "facts" in payload or "variants" in payload or "category_path" in payload:
                n += self._upsert_view_like(payload)
            else:
                n += int(self._upsert_stock_row(payload))
        return n

    def _key(self, product_id: Any, sku: Any) -> str:
        return f"{product_id}::{sku}"

    def _allowed(self, product_id: Any, sku: Any) -> bool:
        if not self.restrict_keys:
            return True
        return self._key(product_id, sku) in self.restrict_keys

    def _ensure(self, product_id: str, sku: str) -> dict[str, Any]:
        key = self._key(product_id, sku)
        rec = self.candidates.get(key)
        if rec is None:
            rec = {
                "key": key,
                "product_id": str(product_id),
                "sku": str(sku),
                "title": "",
                "brand": None,
                "category_path": [],
                "facts": {},
                "price": None,
                "in_stock": None,
                "screen_size": None,
                "resolution": None,
                "refresh_rate": None,
                "search_rank": None,
                "viewed": False,
                "soft_features": set(),
                "soft_features_facts": set(),
                "resolution_fact": None,
                "screen_size_fact": None,
                "brand_verified": False,
            }
            self.candidates[key] = rec
        return rec

    def _set_structured_brand(self, rec: dict[str, Any], brand: Any) -> None:
        """Record catalog brand. Explicit null clears any prior value (UNKNOWN)."""
        if brand is None:
            rec["brand"] = None
            rec["brand_verified"] = False
            return
        text = str(brand).strip()
        if not text:
            rec["brand"] = None
            rec["brand_verified"] = False
            return
        rec["brand"] = text
        rec["brand_verified"] = True

    def _apply_facts(self, rec: dict[str, Any], facts: dict[str, Any]) -> None:
        rec["facts"] = {**(rec.get("facts") or {}), **facts}
        # screen size from product facts (may be wrong for multi-SKU listings)
        for k in ("size", "screen_size", "display_size", "screen"):
            if k in facts:
                got = _norm_screen_in(facts[k])
                if got is not None:
                    rec["screen_size_fact"] = got
                    if rec["screen_size"] is None:
                        rec["screen_size"] = got
        # resolution / refresh often buried in display strings
        blob = " ".join(str(v) for v in facts.values())
        if rec["resolution"] is None or "display" in facts:
            for k in ("resolution", "display", "screen_resolution"):
                if k in facts:
                    got = _norm_res(facts[k])
                    if got:
                        if k in ("display", "resolution", "screen_resolution"):
                            rec["resolution_fact"] = got
                        rec["resolution"] = got
                        break
            if rec["resolution"] is None:
                rec["resolution"] = _norm_res(blob)
        if rec["refresh_rate"] is None:
            for k in ("refresh_rate", "refresh"):
                if k in facts:
                    rec["refresh_rate"] = _norm_refresh_hz(facts[k])
                    if rec["refresh_rate"] is not None:
                        break
            if rec["refresh_rate"] is None:
                rec["refresh_rate"] = _norm_refresh_hz(blob)
        # Soft features from structured facts only → justification-grade evidence.
        fact_blob = " ".join(str(v) for v in facts.values())
        mined = _mine_soft_features(fact_blob)
        if mined:
            rec["soft_features_facts"] = set(rec.get("soft_features_facts") or set()) | mined
            rec["soft_features"] = set(rec.get("soft_features") or set()) | mined

    def _apply_variant_text(
        self,
        rec: dict[str, Any],
        *,
        title: str | None = None,
        options: Any = None,
    ) -> None:
        """Mine title/options for SEARCH/RANK hints. Never promotes brand."""
        if options is not None and str(options).strip():
            opt = str(options)
            # Only trust explicit inch markers in options (avoid "r34w-30" → 34).
            m = _SCREEN_RE.search(opt)
            if m:
                rec["screen_size"] = float(m.group(1) or m.group(2))
            hz = _norm_refresh_hz(opt)
            if hz is not None:
                rec["refresh_rate"] = hz
        text = title or rec.get("title") or ""
        if text:
            if rec["refresh_rate"] is None:
                rec["refresh_rate"] = _norm_refresh_hz(text)
            if rec["screen_size"] is None:
                rec["screen_size"] = _norm_screen_in(text)
            if rec["resolution"] is None:
                rec["resolution"] = _norm_res(text)
        # Title/options soft features: search hints + env-allowed soft feature claims.
        # NEVER used as brand evidence.
        fact_blob = " ".join(str(v) for v in (rec.get("facts") or {}).values())
        mined = _mine_soft_features(text, options)
        if mined:
            rec["soft_features"] = set(rec.get("soft_features") or set()) | mined
        # Keep fact-mined features authoritative when present.
        fact_mined = _mine_soft_features(fact_blob)
        if fact_mined:
            rec["soft_features_facts"] = (
                set(rec.get("soft_features_facts") or set()) | fact_mined
            )
            rec["soft_features"] = set(rec.get("soft_features") or set()) | fact_mined

    def _record_price_stock(
        self, rec: dict[str, Any], price: float | None, stock: bool | None
    ) -> None:
        """Apply live price/stock; detect market moves that flip ranking."""
        key = rec["key"]
        if price is not None:
            old = rec.get("price")
            if old is not None and abs(float(old) - float(price)) > 1e-6:
                if not self.event_seen:
                    self._pre_event_snapshot_winner = self.current_lex_winner_key()
                self.event_seen = True
            rec["price"] = price
        if stock is not None:
            old_s = rec.get("in_stock")
            if old_s is not None and bool(old_s) != bool(stock):
                if not self.event_seen:
                    self._pre_event_snapshot_winner = self.current_lex_winner_key()
                self.event_seen = True
            rec["in_stock"] = stock
        if self.event_seen:
            self.inspected_after_event.add(key)

    def _upsert_search_row(self, row: dict[str, Any], *, rank: int) -> bool:
        pid, sku = row.get("product_id"), row.get("sku")
        if pid is None or sku is None or not self._allowed(pid, sku):
            return False
        rec = self._ensure(str(pid), str(sku))
        rec["title"] = row.get("title") or rec["title"]
        if "brand" in row:
            self._set_structured_brand(rec, row.get("brand"))
        price = _as_float(row.get("price"))
        stock = _truthy_stock(row.get("in_stock"))
        # Search rows seed facts; only treat as event once we already had values.
        if rec.get("price") is not None or rec.get("in_stock") is not None:
            self._record_price_stock(rec, price, stock)
        else:
            if price is not None:
                rec["price"] = price
            if stock is not None:
                rec["in_stock"] = stock
        if rec["search_rank"] is None:
            rec["search_rank"] = rank
        if rec["screen_size"] is None:
            rec["screen_size"] = _norm_screen_in(rec["title"])
        if rec["resolution"] is None:
            rec["resolution"] = _norm_res(rec["title"])
        if rec["refresh_rate"] is None:
            rec["refresh_rate"] = _norm_refresh_hz(rec["title"])
        return True

    def _upsert_stock_row(self, row: dict[str, Any]) -> bool:
        pid, sku = row.get("product_id"), row.get("sku")
        if pid is None or sku is None or not self._allowed(pid, sku):
            return False
        rec = self._ensure(str(pid), str(sku))
        price = _as_float(row.get("price"))
        stock = _truthy_stock(row.get("in_stock"))
        self._record_price_stock(rec, price, stock)
        return True

    def _upsert_view_like(self, row: dict[str, Any]) -> int:
        pid = row.get("product_id")
        if pid is None:
            return 0
        title = row.get("title") or ""
        brand = row.get("brand")
        path = row.get("category_path") or []
        facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
        variants = row.get("variants")
        n = 0
        if isinstance(variants, list) and variants:
            for var in variants:
                if not isinstance(var, dict):
                    continue
                sku = var.get("sku")
                if sku is None or not self._allowed(pid, sku):
                    continue
                rec = self._ensure(str(pid), str(sku))
                rec["viewed"] = True
                rec["title"] = title or rec["title"]
                if "brand" in row:
                    self._set_structured_brand(rec, brand)
                if path:
                    rec["category_path"] = path
                self._apply_facts(rec, facts)
                self._apply_variant_text(
                    rec, title=title or rec.get("title"), options=var.get("options")
                )
                price = _as_float(var.get("price"))
                stock = _truthy_stock(var.get("in_stock"))
                if rec.get("price") is not None or rec.get("in_stock") is not None:
                    self._record_price_stock(rec, price, stock)
                else:
                    if price is not None:
                        rec["price"] = price
                    if stock is not None:
                        rec["in_stock"] = stock
                n += 1
        else:
            # single sku on the row itself
            sku = row.get("sku") or "default"
            if not self._allowed(pid, sku):
                return 0
            rec = self._ensure(str(pid), str(sku))
            rec["viewed"] = True
            rec["title"] = title or rec["title"]
            if "brand" in row:
                self._set_structured_brand(rec, brand)
            if path:
                rec["category_path"] = path
            self._apply_facts(rec, facts)
            self._apply_variant_text(
                rec, title=title or rec.get("title"), options=row.get("options")
            )
            price = _as_float(row.get("price"))
            stock = _truthy_stock(row.get("in_stock"))
            if rec.get("price") is not None or rec.get("in_stock") is not None:
                self._record_price_stock(rec, price, stock)
            else:
                if price is not None:
                    rec["price"] = price
                if stock is not None:
                    rec["in_stock"] = stock
            n = 1
        return n

    def _device_category_ok(self, rec: dict[str, Any]) -> bool:
        """Reject clear category conflicts (e.g. Feature Phones when smartphone required)."""
        if not self.category_hint:
            return True
        path = " ".join(str(x) for x in (rec.get("category_path") or [])).lower()
        title = (rec.get("title") or "").lower()
        facts = rec.get("facts") or {}
        fact_blob = " ".join(str(v) for v in facts.values()).lower()
        hint = self.category_hint
        if hint == "smartphones":
            if "feature phone" in path or "basic phone" in path:
                return False
            phone_type = str(facts.get("phone_type") or facts.get("device_type") or "").lower()
            if any(x in phone_type for x in ("feature", "basic", "keypad")) and "smart" not in phone_type:
                return False
            if "smartphone" in path or "smartphones" in path:
                return True
            os_hit = re.search(
                r"\b(android|ios|iphone|ipados|blackberry|windows phone|symbian|"
                r"harmonyos|hyperos|sailfish|ubuntu touch|tizen)\b",
                fact_blob,
                re.I,
            )
            if os_hit:
                return True
            if "smartphone" in phone_type or "smart phone" in phone_type:
                return True
            # Positive smartphone evidence missing after view → not committable.
            if rec.get("viewed") and path:
                return False
            return True
        stem = hint.rstrip("s")
        if path and stem not in path and hint not in path and stem not in title:
            return False
        return True

    def _firm_failures(self, rec: dict[str, Any]) -> list[str]:
        fails: list[str] = []
        if rec.get("in_stock") is False:
            fails.append("in_stock")
        if self.budget is not None and rec.get("price") is not None:
            if float(rec["price"]) > float(self.budget) + 1e-6:
                fails.append("budget")
        if self.gate_attr == "screen_size" and self.gate_value is not None:
            # Ranking may use title/options; commitment families need facts after view.
            if self.analysis.get("ranking_like"):
                got = rec.get("screen_size")
                if got is None or abs(float(got) - float(self.gate_value)) > 0.05:
                    fails.append("screen_size")
            elif rec.get("viewed"):
                got = rec.get("screen_size_fact")
                if got is None or abs(float(got) - float(self.gate_value)) > 0.05:
                    fails.append("screen_size")
            else:
                got = rec.get("screen_size")
                if got is None or abs(float(got) - float(self.gate_value)) > 0.05:
                    fails.append("screen_size")
        if self.gate_attr == "resolution" and self.gate_value is not None:
            want = _norm_res(self.gate_value)
            if self.analysis.get("needs_justification") or (
                rec.get("viewed") and not self.analysis.get("ranking_like")
            ):
                got = rec.get("resolution_fact")
            else:
                got = rec.get("resolution")
            if not got or not want or got != want:
                fails.append("resolution")
        if self.gate_attr == "refresh_rate" and self.gate_value is not None:
            got = rec.get("refresh_rate")
            if got is None or abs(float(got) - float(self.gate_value)) > 0.5:
                fails.append("refresh_rate")
        if self.brand_is_firm and self.preferred_brand:
            # Structured brand only — title "Samsungs..." with brand=null FAILS.
            if not self._verified_brand_hit(rec):
                fails.append("brand")
        if not self._device_category_ok(rec):
            fails.append("category")
        if self.restrict_keys and rec["key"] not in self.restrict_keys:
            fails.append("not_in_candidate_set")
        return fails

    def _verified_brand_hit(self, rec: dict[str, Any]) -> bool:
        """Firm brand match against structured catalog brand only."""
        if not self.preferred_brand:
            return False
        brand = _verified_brand(rec)
        if not brand:
            return False
        pref = self.preferred_brand.lower()
        b = brand.lower()
        return pref in b or b in pref

    def _title_brand_hint(self, rec: dict[str, Any]) -> bool:
        """Textual hint for SEARCH/RANK only — never justification/commitment evidence."""
        if not self.preferred_brand:
            return False
        pref = self.preferred_brand.lower()
        title = (rec.get("title") or "").lower()
        return pref in title

    def _soft_brand_hit(self, rec: dict[str, Any]) -> bool:
        """Backward-compatible name: verified structured brand only."""
        return self._verified_brand_hit(rec)

    def score_one(self, rec: dict[str, Any]) -> dict[str, Any]:
        fails = self._firm_failures(rec)
        firm_ok = not fails
        soft_ok = self._verified_brand_hit(rec)
        title_hint = self._title_brand_hint(rec)
        # verified required attributes >> textual similarity
        score = 0.0
        if firm_ok:
            score += 10.0
        else:
            score -= 10.0 * len(fails)
        if soft_ok:
            score += 3.0
        elif title_hint and not soft_ok:
            # Tiny search hint only; never enough to beat a verified candidate.
            score += 0.15
            if self.brand_is_firm:
                score -= 2.0  # penalize decoys when brand is required
        if rec.get("brand_verified"):
            score += 0.5
        if rec.get("resolution_fact") or rec.get("screen_size_fact"):
            score += 0.4
        if rec.get("viewed"):
            score += 0.05
        if _verified_soft_claim(rec) is not None:
            score += 0.35
        if rec["key"] in self.rejected_keys:
            score -= 5.0
        rank = rec.get("search_rank")
        if isinstance(rank, int) and 0 <= rank < len(self._SEARCH_RANK_BONUS):
            score += self._SEARCH_RANK_BONUS[rank]
        # Slight preference for lower price among equals (non-lex mode).
        if rec.get("price") is not None and not self.lex_priorities:
            score += max(0.0, 0.2 - (float(rec["price"]) / 1e7))
        return {
            **rec,
            "firm_ok": firm_ok,
            "firm_failures": fails,
            "soft_brand_ok": soft_ok,
            "title_brand_hint": title_hint,
            "score": score,
        }

    def _lex_tuple(self, rec: dict[str, Any]) -> tuple:
        """Lexicographic key; higher is better after polarity flip."""
        parts: list[float] = []
        for metric, direction in self.lex_priorities:
            if metric == "refresh_rate":
                val = rec.get("refresh_rate")
            elif metric == "screen_size":
                val = rec.get("screen_size")
            elif metric == "price":
                val = rec.get("price")
            else:
                val = None
            if val is None:
                # Missing metric sorts last.
                parts.append(float("-inf"))
                continue
            num = float(val)
            parts.append(num if direction == "max" else -num)
        # Stable extras: prefer in-stock, then lower price, then key.
        parts.append(1.0 if rec.get("in_stock") else 0.0)
        parts.append(-(float(rec["price"]) if rec.get("price") is not None else 1e18))
        return tuple(parts)

    def ranked(self) -> list[dict[str, Any]]:
        scored = [self.score_one(rec) for rec in self.candidates.values()]
        if self.analysis.get("ranking_like") and self.lex_priorities:
            # Among stock/budget-ok candidates, sort by lex priorities (higher better).
            def sort_key(item: dict[str, Any]) -> tuple:
                hard = [
                    f
                    for f in item["firm_failures"]
                    if f in ("in_stock", "budget", "not_in_candidate_set")
                ]
                hard_rank = 0 if not hard else 1
                # Negate lex tuple so Python ascending sort prefers better metrics.
                return (hard_rank, tuple(-x for x in self._lex_tuple(item)), item["key"])

            scored.sort(key=sort_key)
        else:
            scored.sort(
                key=lambda item: (
                    0 if item["firm_ok"] else 1,
                    0 if item["soft_brand_ok"] else 1,
                    -item["score"],
                    float(item["price"]) if item.get("price") is not None else 1e18,
                    item["key"],
                )
            )
        return scored

    def advice(self, *, top_k: int = 5) -> str | None:
        if not self.candidates:
            return None
        ranked = self.ranked()
        lines = [
            "Nimbus observation scorer (deterministic; firm > soft > lex/rank):",
            f"- budget={self.budget} gate={self.gate_attr}:{self.gate_value} "
            f"soft_brand={self.preferred_brand} lex={self.lex_priorities}",
        ]
        firm = [r for r in ranked if r["firm_ok"]]
        soft_firm = [r for r in firm if r["soft_brand_ok"]]
        if self.preferred_brand:
            if soft_firm:
                best = soft_firm[0]
                lines.append(
                    f"- INITIAL cart recommendation: {best['key']} "
                    f"(firm-ok + preferred brand), price={best.get('price')}, "
                    f"stock={best.get('in_stock')}"
                )
            elif firm:
                best = firm[0]
                lines.append(
                    f"- No firm-ok preferred-brand hit; use firm-ok fallback {best['key']} "
                    f"price={best.get('price')} stock={best.get('in_stock')}"
                )
            else:
                lines.append("- No firm-ok candidates yet; keep searching/viewing.")
        elif firm:
            best = firm[0]
            lines.append(
                f"- Top firm-ok pick: {best['key']} price={best.get('price')} "
                f"stock={best.get('in_stock')} score={best['score']:.4f}"
            )
        else:
            lines.append("- No firm-ok candidates yet; keep searching/viewing.")

        lines.append("- Shortlist:")
        for rec in ranked[:top_k]:
            lines.append(
                f"  * {rec['key']} firm_ok={rec['firm_ok']} soft={rec['soft_brand_ok']} "
                f"fail={rec['firm_failures']} price={rec.get('price')} "
                f"stock={rec.get('in_stock')} screen={rec.get('screen_size')} "
                f"res={rec.get('resolution')} hz={rec.get('refresh_rate')} "
                f"viewed={rec.get('viewed')} score={rec['score']:.4f}"
            )
        if self.analysis.get("ranking_like") and ranked:
            winner = ranked[0]
            if self.event_seen:
                pending = self.pending_post_event_inspect()
                lines.append(
                    f"- MARKET EVENT ACTIVE. Post-event lex winner NOW: {winner['key']} "
                    f"(price={winner.get('price')} stock={winner.get('in_stock')})."
                )
                if self.static_committed_key and self.static_committed_key != winner["key"]:
                    lines.append(
                        f"- FORBIDDEN to order stale pre-event pick "
                        f"{self.static_committed_key}. Remove it from cart; "
                        f"add_to_cart {winner['key']}; place_test_order that key only."
                    )
                else:
                    lines.append(
                        f"- Switch cart to {winner['key']} if needed, then place_test_order "
                        "with NO justification."
                    )
                if pending:
                    lines.append(
                        "- Still must inspect_stock these candidates after the event: "
                        + ", ".join(pending[:8])
                    )
            else:
                lines.append(
                    f"- Lex/ranking winner right now: {winner['key']} "
                    "(FIRST add_to_cart MUST be this static rank-1 commitment; "
                    "after any later price/stock event, re-inspect ALL candidates and "
                    "switch to the NEW winner — never revert)."
                )
            lines.append("- Ranking orders: omit place_test_order.justification entirely.")
        return "\n".join(lines)



def _emergency_finish_calls(
    scorer: "ObservationScorer",
    *,
    needs_justification: bool,
    ranking_like: bool,
) -> list[dict[str, Any]]:
    """Deterministic last-resort tool sequence when the LLM path stalls."""
    ready = [
        r
        for r in scorer.ranked()
        if scorer.is_committable(r) and r["key"] not in scorer.rejected_keys
    ]
    if not ready:
        # Best effort: any viewed in-stock under budget.
        ready = [
            r
            for r in scorer.ranked()
            if r.get("viewed")
            and r.get("in_stock") is True
            and r.get("price") is not None
            and (
                scorer.budget is None
                or float(r["price"]) <= float(scorer.budget) + 1e-6
            )
            and r["key"] not in scorer.rejected_keys
        ]
    if not ready:
        return []
    best = ready[0]
    calls: list[dict[str, Any]] = []
    # Ensure stock/price are fresh, then cart + order.
    calls.append(
        {
            "id": "emerg-inspect",
            "type": "function",
            "function": {
                "name": "inspect_stock",
                "arguments": json.dumps(
                    {"product_id": best["product_id"], "sku": best["sku"]}
                ),
            },
        }
    )
    calls.append(
        {
            "id": "emerg-add",
            "type": "function",
            "function": {
                "name": "add_to_cart",
                "arguments": json.dumps(
                    {"product_id": best["product_id"], "sku": best["sku"]}
                ),
            },
        }
    )
    order_args: dict[str, Any] = {
        "product_id": best["product_id"],
        "sku": best["sku"],
    }
    calls.append(
        {
            "id": "emerg-order",
            "type": "function",
            "function": {
                "name": "place_test_order",
                "arguments": json.dumps(order_args),
            },
        }
    )
    # Run through sanitizer so TF7 justification stays factual when required.
    return _sanitize_tool_calls(
        calls,
        needs_justification=needs_justification,
        ranking_like=ranking_like,
        gate_attr=scorer.gate_attr,
        scorer=scorer,
    )


def _budget_from_text(text: str) -> float | None:
    budgets = [float(x) for x in _BUDGET_RE.findall(text or "")]
    return min(budgets) if budgets else None

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))

    query = str(policy.get("query") or problem_data.get("query") or "")
    tools = policy.get("tools") or []
    max_steps = int(policy.get("max_steps") or 20)
    max_calls = int(policy.get("max_calls_per_turn") or 1)
    names = _tool_names(tools)

    analysis = _analyze_query(query)
    # Strong models only for ranking/justification (step-efficient).
    prefer_strong = bool(
        analysis.get("ranking_like") or analysis.get("needs_justification")
    )
    brief = _strategy_brief(analysis, names)
    scorer = ObservationScorer(analysis, query)

    # Planning is optional; default skip saves one inference round.
    skip_plan = getenv("NIMBUS_SKIP_PLAN", "1") not in ("0", "false", "False")
    plan = None if skip_plan else _plan_episode(
        query, names, prefer_strong=prefer_strong
    )
    if isinstance(plan, dict):
        # Merge planner budget/brand cues when present (still from this problem only).
        pb = plan.get("budget")
        if isinstance(pb, (int, float)):
            scorer.update_budget(float(pb))
        pref = plan.get("preferred_brand")
        if isinstance(pref, str) and pref.strip() and not scorer.preferred_brand:
            scorer.preferred_brand = pref.strip()
    plan_txt = (
        json.dumps(plan, ensure_ascii=False, indent=2)
        if isinstance(plan, dict)
        else "(planner unavailable — follow system rules and strategy brief)"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_CORE},
        {
            "role": "user",
            "content": (
                f"{brief}\n\n"
                f"Internal plan JSON:\n{plan_txt}\n\n"
                f"Shopper goal:\n{query}\n\n"
                f"Limits: max_steps={max_steps}, max_calls_per_turn={max_calls}.\n"
                "Choose the next tool action(s) now."
            ),
        },
    ]
    dialogue: list[dict[str, Any]] = [
        {
            "role": "meta",
            "analysis": {
                "ranking_like": analysis["ranking_like"],
                "needs_justification": analysis["needs_justification"],
                "n_candidates": len(analysis["candidate_variants"]),
                "budget": analysis["budget"],
                "gate_attr": analysis.get("gate_attr"),
                "lex_priorities": scorer.lex_priorities,
            },
            "plan": plan,
        }
    ]

    last_actions: list[str] = []
    last_obs_blob = ""
    pending_intervention: str | None = None
    consecutive_infer_fails = 0

    for turn in range(1, max_steps + 1):
        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        temperature = 0.0
        emergency_only = False

        # Step-budget endgame (NOT wall-clock): use remaining max_steps.
        steps_left = max_steps - turn + 1
        if steps_left <= 2 and scorer.candidates:
            tool_calls = _emergency_finish_calls(
                scorer,
                needs_justification=bool(analysis["needs_justification"]),
                ranking_like=bool(analysis["ranking_like"]),
            )
            if tool_calls:
                assistant_content = "emergency_finish_low_steps"
                emergency_only = True

        if not tool_calls:
            for _attempt in range(3):
                choice = "required" if _attempt == 0 else "auto"
                try:
                    inference = _inference(
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        prefer_strong=prefer_strong,
                        tool_choice=choice,
                    )
                    consecutive_infer_fails = 0
                except Exception as exc:
                    LOGGER.warning("inference fail turn=%s attempt=%s: %s", turn, _attempt, exc)
                    if _attempt < 2:
                        temperature = min(0.5, temperature + 0.2)
                        continue
                    consecutive_infer_fails += 1
                    # Do NOT return dialogue early — that caused YC agent_error at step ~8.
                    tool_calls = _emergency_finish_calls(
                        scorer,
                        needs_justification=bool(analysis["needs_justification"]),
                        ranking_like=bool(analysis["ranking_like"]),
                    )
                    if tool_calls:
                        assistant_content = f"emergency_finish_after_inference_error: {exc}"
                        emergency_only = True
                        break
                    # No candidate yet: deterministic explore instead of aborting the episode.
                    if "filter" in names and scorer.budget is not None:
                        tool_calls = [
                            {
                                "id": f"explore-{turn}",
                                "type": "function",
                                "function": {
                                    "name": "filter",
                                    "arguments": json.dumps(
                                        {
                                            "in_stock": True,
                                            "k": 10,
                                            "max_price": float(scorer.budget),
                                        }
                                    ),
                                },
                            }
                        ]
                        assistant_content = "deterministic_explore_after_inference_error"
                        break
                    if "search" in names:
                        tool_calls = [
                            {
                                "id": f"explore-{turn}",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": json.dumps(
                                        {
                                            "query": query.split("\n")[0][:120],
                                            "in_stock": True,
                                            "k": 10,
                                        }
                                    ),
                                },
                            }
                        ]
                        assistant_content = "deterministic_explore_after_inference_error"
                        break
                    dialogue.append({"role": "assistant", "error": f"inference: {exc}"})
                    # Continue to next turn rather than killing the episode.
                    tool_calls = []
                    break

                if tool_calls:
                    break

                assistant = inference["choices"][0]["message"]
                assistant_content = assistant.get("content") or ""
                raw_calls = assistant.get("tool_calls") or []
                valid_calls: list[dict[str, Any]] = []
                for call in raw_calls[:max_calls]:
                    try:
                        _arguments(call)
                        valid_calls.append(call)
                    except Exception as exc:
                        LOGGER.warning("dropping malformed tool call: %s", exc)
                tool_calls = valid_calls
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
                                "You must call one of the supplied tools. "
                                "Emit valid tool arguments as a JSON object."
                            ),
                        },
                    ]
                )
                temperature = min(0.6, temperature + 0.25)
            else:
                if not tool_calls:
                    dialogue.append(
                        {
                            "role": "assistant",
                            "content": assistant_content or "no tool call",
                            "error": "no_tool_call",
                        }
                    )
                    # Prefer emergency finish over ending the episode early.
                    tool_calls = _emergency_finish_calls(
                        scorer,
                        needs_justification=bool(analysis["needs_justification"]),
                        ranking_like=bool(analysis["ranking_like"]),
                    )
                    if not tool_calls:
                        continue
                    assistant_content = "emergency_finish_after_no_tool_call"
                    emergency_only = True

        if not tool_calls:
            continue

        if pending_intervention:
            tool_calls = _ensure_message_action(
                tool_calls,
                pending_intervention=pending_intervention,
                query=query,
                tool_names=names,
                max_calls=max_calls,
                turn=turn,
            )
            pending_intervention = None

        tool_calls = _sanitize_tool_calls(
            tool_calls,
            needs_justification=bool(analysis["needs_justification"]),
            ranking_like=bool(analysis["ranking_like"]),
            gate_attr=analysis.get("gate_attr"),
            scorer=scorer,
        )
        tool_calls = _align_ranking_tool_calls(tool_calls, scorer)
        tool_calls = _align_justification_tool_calls(
            tool_calls,
            scorer,
            needs_justification=bool(analysis["needs_justification"]),
            ranking_like=bool(analysis["ranking_like"]),
        )
        # Rebuild justification AFTER alignment may rewrite product_id/sku.
        if analysis["needs_justification"] and not analysis["ranking_like"]:
            tool_calls = _sanitize_tool_calls(
                tool_calls,
                needs_justification=True,
                ranking_like=False,
                gate_attr=analysis.get("gate_attr"),
                scorer=scorer,
            )

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )

        group_id = f"{problem_id}-turn-{turn}"
        try:
            envelope = {
                **binding,
                "call_id": group_id,
                "idempotency_key": group_id,
                "turn": turn,
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
        except Exception as exc:
            dialogue.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                    "error": f"bad_args: {exc}",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool arguments were invalid ({exc}). Call a tool with valid JSON args.",
                }
            )
            continue

        result = None
        for env_try in range(3):
            result = _proxy.post("/environment/call", json_data=envelope)
            if result is not None:
                break
            time.sleep(0.8 * (env_try + 1))
        if result is None:
            dialogue.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                    "error": "environment_call_failed",
                }
            )
            break

        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        last_actions = [c["function"]["name"] for c in tool_calls]
        for call in tool_calls:
            name = call["function"]["name"]
            if name != "add_to_cart":
                continue
            try:
                args = _arguments(call)
            except Exception:
                continue
            scorer.note_cart_commit(args.get("product_id"), args.get("sku"))
        obs_parts: list[str] = []
        ingested = 0
        for tool_call, call_result in zip(tool_calls, result.get("calls") or []):
            obs = call_result.get("observation", {})
            obs_txt = json.dumps(obs, ensure_ascii=False)
            obs_parts.append(obs_txt)
            ingested += scorer.ingest_observation(obs)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content": obs_txt,
                }
            )
        last_obs_blob = "\n".join(obs_parts)

        um_text = _user_message_text(result)
        if um_text:
            pending_intervention = um_text
            messages.append({"role": "user", "content": um_text})
            # Shopper budget corrections must update the firm scorer immediately.
            new_budget = _budget_from_text(um_text)
            if new_budget is not None:
                scorer.update_budget(new_budget)
            if scorer.note_market_event(um_text) and analysis.get("ranking_like"):
                pending = scorer.pending_post_event_inspect()
                winner = scorer.current_lex_winner_key()
                tip = (
                    "RANKING MARKET EVENT: inspect_stock every listed candidate "
                    "(especially any not yet re-checked), then remove the stale cart "
                    "item if it is no longer the lex winner, add_to_cart the NEW winner, "
                    "and place_test_order that winner with NO justification. "
                    "Do not revert to the pre-event pick."
                )
                if pending:
                    tip += " Still need inspect_stock on: " + ", ".join(pending[:8]) + "."
                if winner:
                    tip += f" Current lex winner key: {winner}."
                messages.append({"role": "user", "content": tip})

        if _observation_done(result):
            break
        # Keep using remaining max_steps after a failed emergency finish attempt.

        nudge = _nudge_for_observation(
            turn,
            max_steps,
            analysis,
            last_actions,
            last_obs_blob,
            pending_intervention,
        )
        if nudge:
            messages.append({"role": "user", "content": nudge})

        # Inject deterministic shortlist after we learned something new.
        if ingested > 0 and not pending_intervention:
            advice = scorer.advice(top_k=5)
            if advice:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            advice
                            + "\nAct on this shortlist while still obeying tool schemas and "
                            "shopper interventions. Prefer inspect_stock → add_to_cart → "
                            "place_test_order on the recommended firm-ok key when ready."
                        ),
                    }
                )

    return dialogue


__all__ = ["agent_main", "ObservationScorer"]
