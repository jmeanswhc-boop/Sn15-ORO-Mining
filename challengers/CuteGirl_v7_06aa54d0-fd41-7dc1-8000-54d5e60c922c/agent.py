from __future__ import annotations

import json
import re
from os import getenv
from typing import Any, Protocol

from src.agent.proxy_client import ProxyClient


_MODEL_LIGHT = "deepseek/deepseek-v3.2"
_MODEL_DENSE = "qwen/qwen3.5-397b-a17b"
_FALLBACKS = (
    "google/gemma-4-31b-it",
    "openai/gpt-oss-120b",
    "deepseek/deepseek-chat-v3.1",
)
_CHUTES_TO_OR: tuple[tuple[str, str], ...] = (
    ("deepseek-ai/DeepSeek-V3.2-TEE", "deepseek/deepseek-v3.2"),
    ("deepseek-ai/DeepSeek-V3.1-TEE", "deepseek/deepseek-chat-v3.1"),
    ("deepseek-ai/DeepSeek-V3-0324-TEE", "deepseek/deepseek-chat-v3-0324"),
    ("deepseek-ai/DeepSeek-R1-0528-TEE", "deepseek/deepseek-r1-0528"),
    ("Qwen/Qwen3-32B-TEE", "qwen/qwen3-32b"),
    ("Qwen/Qwen3.5-397B-A17B-TEE", "qwen/qwen3.5-397b-a17b"),
    ("Qwen/Qwen3.6-27B-TEE", "qwen/qwen3.6-27b"),
    ("google/gemma-4-31B-turbo-TEE", "google/gemma-4-31b-it"),
    ("openai/gpt-oss-120b-TEE", "openai/gpt-oss-120b"),
    ("XiaomiMiMo/MiMo-V2-Flash-TEE", "xiaomi/mimo-v2-flash"),
    ("unsloth/Mistral-Nemo-Instruct-2407-TEE", "mistralai/mistral-small-2603"),
    ("zai-org/GLM-5.1-TEE", "qwen/qwen3.5-397b-a17b"),
    ("zai-org/GLM-5.2-TEE", "qwen/qwen3.5-397b-a17b"),
    ("moonshotai/Kimi-K2.6-TEE", "qwen/qwen3.5-397b-a17b"),
    ("moonshotai/Kimi-K2.5-TEE", "qwen/qwen3.5-397b-a17b"),
)


_http = ProxyClient(timeout=120, max_retries=2)
_OBS_CAP = 6000

_MAX_CALLS_CAP = 16


TF1 = "intent_decomposition"
TF2 = "retrieval_recall"
TF3 = "constraint_satisfaction"
TF4 = "preference_reasoning"
TF5 = "ranking"
TF6 = "recovery"
TF7 = "justification"

_ALL_TFS: tuple[str, ...] = (TF1, TF2, TF3, TF4, TF5, TF6, TF7)


_RING: tuple[str, ...] = (TF2, TF1, TF4, TF3, TF7, TF5, TF6)
_MUTEX = frozenset(
    {
        frozenset({TF5, TF2}),

        frozenset({TF1, TF4}),
    }
)
_SAT_RATIO = 0.55
_SAT_GAP = 1


_TF_PRIORITY = (TF5, TF6, TF7, TF3, TF1, TF4, TF2)

_CLAIM_DENY = frozenset(
    {
        "category",
        "category_path",
        "title",
        "summary",
        "evidence",
        "reason",
        "description",
        "name",
        "note",


        "display",
        "device_operating_system",
        "operating_system",
        "description_operating_system",
    }
)
_TF7_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,32}$")


_RX_VARIANT = re.compile(r"(?<!\d)(\d{6,})::([A-Za-z0-9._-]+)")
_RX_QUOTE = re.compile(r"[\"\u201c](?P<body>[^\"\u201d]{3,55})[\"\u201d]")
_RX_CEILING = re.compile(
    r"(?:budget|cap|ceiling|at most|no more than|under|below|up to|"
    r"max(?:imum)?(?:\s+(?:of|price))?)\s*[:=]?\s*([\d][\d,]*(?:\.\d+)?)",
    re.I,
)
_RX_ASK = re.compile(
    r"(?:only|will)\s+(?:tell|share|reveal).{0,48}\bif\s+you\s+ask|"
    r"\bif you ask\b|\bwhen asked\b|\bwithhold\b.{0,36}\bspec\b|"
    r"ask\s+(?:me\s+)?(?:for\s+)?(?:the\s+)?(?:exact\s+)?(?:figure|spec|requirement)",
    re.I,
)
_RX_WHY = re.compile(
    r"\bwhy\s+(?:this|that|your|the|it)\b|"
    r"\bwhy\b.{0,24}(?:the one|it'?s|your pick|you (?:choose|pick|order))|"
    r"\bwalk me through why\b|\btell me why\b|\bgive me a reason\b|"
    r"\ba reason\b.{0,24}\btrust\b|\breason\s+for\b|\breason I can\b|"
    r"\bjustif(?:y|ication)\b|\bexplain\s+(?:the\s+)?(?:choice|pick|order)\b|"
    r"claims?\s+must\s+be\s+grounded|include\s+justification|"
    r"\bstructured justification\b",
    re.I,
)
_RX_OPTIONAL_WHY = re.compile(
    r"claims?\s+(?:are|is)\s+optional|justification\s+(?:is\s+)?optional",
    re.I,
)
_RX_BRAND = re.compile(
    r"preferred\s+brand\s+([A-Za-z0-9][\w.&'-]{0,40})"
    r"|prefer(?:red)?\s+(?:the\s+)?brand\s+([A-Za-z0-9][\w.&'-]{0,40})",
    re.I,
)
_RX_PRIORITY = re.compile(
    r"(?:trade-?off|priority|rank(?:ing)?|lexico(?:graphic)?).{0,120}",
    re.I,
)


_FINGERPRINTS: dict[str, tuple[re.Pattern[str], ...]] = {
    TF5: (
        re.compile(
            r"\btrade-?off\b.{0,40}\bpriorit|\brank(?:ing|ed)?\b|\bre-?rank\b",
            re.I,
        ),
        re.compile(
            r"\bdominat|\bstrictly worse\b|\bclosed set\b|\bnamed candidates?\b",
            re.I,
        ),
        re.compile(r"\d{8,}::\S+"),
    ),
    TF6: (
        re.compile(
            r"\bleast[- ]relax|\brecover\b.{0,24}\bunavailable|\bfallback\b",
            re.I,
        ),
        re.compile(
            r"\bsame model\b.{0,48}(?:\bother\b|\banother\b).{0,12}\bcolou?r\b",
            re.I,
        ),
        re.compile(r"\bdisappear|\bsold\s*out\b|\bunavailable\b", re.I),
    ),
    TF7: (
        re.compile(
            r"\bjustification\b.{0,24}\bclaims?|\bclaims?\b.{0,24}\bjustif|"
            r"\bground(?:ed)? claims?\b",
            re.I,
        ),
        re.compile(
            r"\btell me why\b|\bgive me a reason\b|\bwalk me through why\b|"
            r"\bwhy\b.{0,16}(?:the one|it'?s|your pick)|"
            r"\bexplain\s+(?:the\s+)?(?:choice|pick|order)\b",
            re.I,
        ),
    ),
    TF1: (
        re.compile(r"nice[- ]to[- ]have", re.I),
        re.compile(
            r"\bsubordinate\b.{0,40}\b(?:firm|requirement)|preferred brand",
            re.I,
        ),
        re.compile(r"\bfirm\b.{0,80}\bsoft\b|\bsoft\b.{0,80}\bfirm\b", re.I),
    ),
    TF3: (
        re.compile(r"exact(?:ly)?\s+(?:varian|sku|option)|specific sku", re.I),
        re.compile(
            r"every (?:firm )?requirement must|\bgated (?:spec|requirement)\b",
            re.I,
        ),
        re.compile(
            r"\bif you ask\b|\bwhen asked\b|\bwithhold\b.{0,36}\bspec\b",
            re.I,
        ),
    ),
    TF4: (
        re.compile(
            r"\bnot (?:just )?the cheapest\b|\bsoft (?:axes|preferences?)\b",
            re.I,
        ),
        re.compile(r"intended use|use[- ]case|\brather pay\b|\bmatters more\b", re.I),
        re.compile(r"preference\s+axes|don'?t just grab the cheapest", re.I),
    ),
    TF2: (
        re.compile(r"\breformulate\b|\bshortlist\b|\blucky hit\b", re.I),
        re.compile(
            r"\bdistinct (?:products|identities)\b|\boff[- ]category\b",
            re.I,
        ),
        re.compile(r"matches? (?:the )?(?:phrase )?\"[^\"]+\"", re.I),
    ),
}


def _shopper_head(query: str) -> str:
    text = query or ""
    cut = text.find("Task rules:")
    return text if cut < 0 else text[:cut]


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for entry in tools:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            out.append(str(fn["name"]))
        elif entry.get("name"):
            out.append(str(entry["name"]))
    return out


def _claim_field_enum(tools: list[dict[str, Any]]) -> list[str]:
    """Allowed justification.claim field names from the live tool schema."""
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        if not isinstance(fn, dict) or fn.get("name") != "place_test_order":
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        if not isinstance(props, dict):
            continue
        just = props.get("justification") or {}
        if not isinstance(just, dict):
            continue
        claims = (just.get("properties") or {}).get("claims") or {}
        if not isinstance(claims, dict):
            continue
        items = claims.get("items") or {}
        if not isinstance(items, dict):
            continue
        field = (items.get("properties") or {}).get("field") or {}
        if not isinstance(field, dict):
            continue
        enum = field.get("enum")
        if not isinstance(enum, list) or not enum:
            continue
        out: list[str] = []
        seen: set[str] = set()
        for raw in enum:
            fl = str(raw or "").strip().lower()
            if fl and fl not in seen:
                seen.add(fl)
                out.append(fl)
        return out
    return []


def _claim_field_rule(tools: list[dict[str, Any]]) -> str:
    """The live schema's own wording for which claim fields are legal.

    Packs state the policy in the `field` description when they ship no enum, so
    the environment's sentence is passed through verbatim rather than mirrored in
    a local table.
    """
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        if not isinstance(fn, dict) or fn.get("name") != "place_test_order":
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        just = props.get("justification") if isinstance(props, dict) else None
        claims = (just.get("properties") or {}).get("claims") if isinstance(just, dict) else None
        items = claims.get("items") if isinstance(claims, dict) else None
        field = (items.get("properties") or {}).get("field") if isinstance(items, dict) else None
        if isinstance(field, dict):
            return str(field.get("description") or "")[:600]
    return ""


def _named_variants(text: str) -> list[str]:
    found = [f"{a}::{b.rstrip('.,;:')}" for a, b in _RX_VARIANT.findall(text or "")]
    out: list[str] = []
    seen: set[str] = set()
    for tok in found:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _ring_neighbours(tf: str) -> set[str]:
    if tf not in _RING:
        return set()
    i = _RING.index(tf)
    n = len(_RING)
    return {_RING[(i - 1) % n], _RING[(i + 1) % n]}


def _resolve_model(raw: str) -> str:
    key = (raw or "").strip()
    if not key:
        return _MODEL_DENSE
    for chutes_id, or_id in _CHUTES_TO_OR:
        if key == chutes_id or key.endswith(chutes_id.split("/", 1)[-1]):
            key = or_id
            break
    low = key.lower()
    if (
        "kimi" in low
        or "glm" in low
        or low.startswith("z-ai/")
        or low.startswith("moonshotai/")
    ):
        return _MODEL_DENSE
    return key


def _light_model() -> str:
    forced = (getenv("SANDBOX_MODEL") or "").strip()
    return _resolve_model(forced) if forced else _MODEL_LIGHT


def _inference_text(messages: list[dict[str, Any]], *, model: str) -> str:
    queue = [model]
    for cand in (_MODEL_LIGHT, *_FALLBACKS):
        if cand not in queue:
            queue.append(cand)
    for model_id in queue[:3]:
        low = model_id.lower()
        if "kimi" in low or "glm" in low or low.startswith("z-ai/"):
            continue
        response = _http.post(
            "/inference/chat/completions",
            json_data={
                "model": model_id,
                "messages": messages,
                "temperature": 0,
            },
        )
        if response is None:
            continue
        try:
            text = str(response["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            continue
        if text:
            return text
    return ""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def _chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_calls: int,
) -> tuple[str, list[dict[str, Any]]]:
    queue = [model]


    for cand in (_MODEL_LIGHT, *_FALLBACKS):
        if cand not in queue:
            queue.append(cand)
    failure: Exception | None = None
    for model_id in queue[:3]:
        low = model_id.lower()
        if "kimi" in low or "glm" in low or low.startswith("z-ai/"):
            continue
        response = _http.post(
            "/inference/chat/completions",
            json_data={
                "model": model_id,
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "temperature": 0,
            },
        )
        if response is None:
            failure = RuntimeError(f"empty inference model={model_id}")
            continue
        try:
            msg = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            failure = exc
            continue
        return msg.get("content") or "", list(msg.get("tool_calls") or [])[:max_calls]
    raise RuntimeError(f"CuteGirl inference exhausted ({failure})")


class EpisodeFacts:
    __slots__ = (
        "query",
        "goal",
        "variants",
        "ceiling",
        "phrase",
        "brand_wish",
        "priority_snip",
        "needs_ask",
        "needs_why",
        "omit_claims",
        "claim_keys",
        "claim_enum",
        "claim_rule",
        "claim_facts",
        "claim_text",
        "claim_last",
        "spec_wants",
        "spec_text",
        "tool_names",
        "max_steps",
        "max_calls",
        "brief",
    )

    def __init__(
        self,
        query: str,
        goal: str,
        *,
        variants: list[str] | None = None,
        ceiling: str = "",
        phrase: str = "",
        brand_wish: str = "",
        priority_snip: str = "",
        needs_ask: bool = False,
        needs_why: bool = False,
        omit_claims: bool = True,
        claim_keys: list[str] | None = None,
        claim_enum: list[str] | None = None,
        claim_rule: str = "",
        tool_names: list[str] | None = None,
        max_steps: int = 20,
        max_calls: int = 1,
        brief: str = "",
    ) -> None:
        self.query = query
        self.goal = goal
        self.variants = list(variants or [])
        self.ceiling = ceiling
        self.phrase = phrase
        self.brand_wish = brand_wish
        self.priority_snip = priority_snip
        self.needs_ask = needs_ask
        self.needs_why = needs_why
        self.omit_claims = omit_claims
        self.claim_keys = list(claim_keys or [])
        self.claim_enum = list(claim_enum or [])

        self.claim_rule = claim_rule

        self.claim_facts: dict[str, dict[str, str]] = {}

        self.claim_text: dict[str, str] = {}

        self.claim_last = ""


        self.spec_wants: list[str] = _stated_measures(goal)
        self.spec_text: dict[str, str] = {}
        self.tool_names = list(tool_names or [])
        self.max_steps = max_steps
        self.max_calls = max_calls
        self.brief = brief


class Route:
    __slots__ = ("primary", "satellite", "scores", "trace", "facts")

    def __init__(
        self,
        primary: str,
        satellite: str | None,
        scores: dict[str, int],
        trace: list[str],
        facts: EpisodeFacts,
    ) -> None:
        self.primary = primary
        self.satellite = satellite
        self.scores = scores
        self.trace = trace
        self.facts = facts

    def active(self) -> list[str]:
        out = [self.primary]
        if self.satellite:
            out.append(self.satellite)
        return out


def _extract_facts(
    query: str,
    *,
    tools: list[dict[str, Any]],
    max_steps: int,
    max_calls: int,
) -> EpisodeFacts:
    goal = _shopper_head(query).strip()
    quote = _RX_QUOTE.search(goal)
    ceil = _RX_CEILING.search(goal)
    brand = _RX_BRAND.search(goal)
    pri = _RX_PRIORITY.search(goal)
    why_hit = bool(_RX_WHY.search(goal))
    optional = bool(_RX_OPTIONAL_WHY.search(query))
    needs_why = why_hit and not (
        optional and not re.search(r"\bwhy\b|reason|justify", goal, re.I)
    )
    return EpisodeFacts(
        query=query or "",
        goal=goal,
        variants=_named_variants(query),
        ceiling=(ceil.group(1) if ceil else ""),
        phrase=(quote.group("body").strip() if quote else ""),
        brand_wish=(
            (brand.group(1) or brand.group(2) or "").strip().rstrip(".,;:")
            if brand
            else ""
        ),
        priority_snip=(pri.group(0).strip()[:160].rstrip(".,;") if pri else ""),
        needs_ask=bool(_RX_ASK.search(goal)),
        needs_why=needs_why,
        omit_claims=not needs_why,
        claim_enum=_claim_field_enum(tools),
        claim_rule=_claim_field_rule(tools),
        tool_names=_tool_names(tools),
        max_steps=max(1, int(max_steps)),
        max_calls=max(1, int(max_calls)),
    )


def _score_families(facts: EpisodeFacts) -> dict[str, int]:
    blob = facts.query
    scores = {tf: 0 for tf in _ALL_TFS}
    for tf, patterns in _FINGERPRINTS.items():
        for pat in patterns:
            if pat.search(blob):
                scores[tf] += 1

    if len(facts.variants) >= 2:
        scores[TF5] += 2
    if facts.needs_ask:
        scores[TF3] += 2
    if facts.needs_why:
        scores[TF7] += 2
    if facts.phrase:
        scores[TF2] += 1
    if facts.brand_wish:
        scores[TF1] += 1
    if not any(scores.values()):
        scores[TF2] = 1
    return scores


def route_episode(
    query: str,
    *,
    tools: list[dict[str, Any]],
    max_steps: int,
    max_calls: int,
) -> Route:
    """Route one problem to a primary TF (+ optional ring-adjacent satellite)."""
    facts = _extract_facts(
        query, tools=tools, max_steps=max_steps, max_calls=max_calls
    )
    scores = _score_families(facts)
    trace: list[str] = [f"score:{tf}={scores[tf]}" for tf in _ALL_TFS if scores[tf]]

    ranked = sorted(
        ((tf, scores[tf]) for tf in _ALL_TFS if scores[tf] > 0),
        key=lambda kv: (
            -kv[1],
            _TF_PRIORITY.index(kv[0]) if kv[0] in _TF_PRIORITY else 99,
        ),
    )
    primary, pscore = ranked[0]

    if len(facts.variants) >= 2 and primary != TF5:
        if scores[TF5] >= pscore - 1:
            primary, pscore = TF5, max(scores[TF5], pscore)
            trace.append("force_tf5@named")

    satellite: str | None = None
    neighbours = _ring_neighbours(primary)
    best: tuple[str, int] | None = None
    for tf, score in ranked[1:]:
        if tf not in neighbours:
            continue
        if frozenset({primary, tf}) in _MUTEX:
            continue
        strong = score >= max(1, int(pscore * _SAT_RATIO)) or (pscore - score) <= _SAT_GAP
        if not strong:
            continue
        if best is None or score > best[1]:
            best = (tf, score)
    if best:
        satellite = best[0]
        trace.append(f"satellite<={satellite}@{best[1]}")
    trace.append(f"primary<={primary}@{pscore}")
    return Route(
        primary=primary,
        satellite=satellite,
        scores=scores,
        trace=trace,
        facts=facts,
    )


class TurnCtx:
    __slots__ = (
        "route",
        "turn",
        "last_tools",
        "last_obs",
        "await_reply",
        "saw_market",
        "ordered",
        "first_cart",
        "cart_live",
        "asked_spec",
        "search_count",
        "viewed_named",
        "message_count",
        "inspect_streak",
    )

    def __init__(
        self,
        *,
        route: Route,
        turn: int,
        last_tools: list[str],
        last_obs: list[Any],
        await_reply: bool,
        saw_market: bool,
        ordered: bool,
        first_cart: bool,
        cart_live: bool,
        asked_spec: bool,
        search_count: int,
        viewed_named: bool,
        message_count: int,
        inspect_streak: int,
    ) -> None:
        self.route = route
        self.turn = turn
        self.last_tools = last_tools
        self.last_obs = last_obs
        self.await_reply = await_reply
        self.saw_market = saw_market
        self.ordered = ordered
        self.first_cart = first_cart
        self.cart_live = cart_live
        self.asked_spec = asked_spec
        self.search_count = search_count
        self.viewed_named = viewed_named
        self.message_count = message_count
        self.inspect_streak = inspect_streak

    @property
    def facts(self) -> EpisodeFacts:
        return self.route.facts

    @property
    def used(self) -> str:
        return " ".join(self.last_tools)


class CuteVocab:
    """Lightweight shopping vocabulary helpers (generic ecommerce, not suite tables)."""

    ID_KEYS: tuple[str, ...] = (
        "brand",
        "model",
        "color",
        "colour",
        "size",
        "compatibility",
        "sku",
        "product_id",
    )
    KW_STOPWORDS: frozenset[str] = frozenset(
        {
            "a",
            "an",
            "the",
            "and",
            "or",
            "of",
            "for",
            "from",
            "with",
            "within",
            "without",
            "to",
            "in",
            "on",
            "at",
            "by",
            "as",
            "is",
            "are",
            "be",
            "than",
            "then",
            "that",
            "this",
            "these",
            "those",
            "about",
            "above",
            "below",
            "under",
            "over",
            "more",
            "less",
            "also",
            "only",
            "just",
            "need",
            "want",
            "looking",
            "find",
            "get",
            "place",
            "order",
            "please",
            "stock",
            "in-stock",
            "available",
        }
    )
    YES_WORDS: frozenset[str] = frozenset(
        {"yes", "true", "1", "in_stock", "instock", "available", "enabled", "included"}
    )
    NO_WORDS: frozenset[str] = frozenset(
        {"no", "false", "0", "out_of_stock", "oos", "unavailable", "without", "none"}
    )

    SEARCHABLE_UNITS: frozenset[str] = frozenset(
        {
            "mm",
            "cm",
            "m",
            "inch",
            "inches",
            "mg",
            "g",
            "kg",
            "oz",
            "lb",
            "ml",
            "l",
            "mah",
            "wh",
            "v",
            "w",
            "kw",
            "hz",
            "mhz",
            "ghz",
            "kb",
            "mb",
            "gb",
            "tb",
            "dpi",
            "pcs",
            "pc",
        }
    )

    @classmethod
    def is_yes(cls, value: str) -> bool:
        return str(value or "").strip().lower() in cls.YES_WORDS

    @classmethod
    def is_no(cls, value: str) -> bool:
        blob = str(value or "").strip().lower()
        return blob in cls.NO_WORDS or blob.startswith("without ")

    @classmethod
    def search_tokens(cls, text: str, *, limit: int = 8) -> list[str]:
        """Strip filler words; keep distinctive value tokens for search reformulation."""
        raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9._%+/\"-]{0,40}", text or "")
        out: list[str] = []
        seen: set[str] = set()
        for tok in raw:
            low = tok.lower().strip(".,;:\"'")
            if not low or low in cls.KW_STOPWORDS or low in seen:
                continue
            if low.isdigit() and len(low) > 10:
                continue
            seen.add(low)
            out.append(tok.strip(".,;:\"'"))
            if len(out) >= limit:
                break
        return out


_RX_MEASURE = re.compile(r'(\d+(?:\.\d+)?)\s*-?\s*([a-z"]+)', re.I)


def _stated_measures(text: str, *, limit: int = 4) -> list[str]:
    """Figures the shopper attached to a unit of measure, as written.

    A bare number is skipped, so a budget or a currency amount never lands here.
    """
    out: list[str] = []
    for num, unit in _RX_MEASURE.findall(text or ""):
        low = unit.lower().strip(".,;:")
        if low != '"':


            if len(low) < 2 or low not in CuteVocab.SEARCHABLE_UNITS:
                continue
        if num not in out:
            out.append(num)
        if len(out) >= limit:
            break
    return out


def _lowered_ceiling(text: str, current: str) -> str:
    """A tighter spending wall the shopper just named, else the current one.

    Only downward moves are taken, and a figure carrying a unit of measure is
    ignored so "under 6 inches" can never be mistaken for a budget.
    """
    now = _as_float(current)
    best = now


    best_text = current
    for hit in _RX_CEILING.finditer(text or ""):
        tail = text[hit.end() : hit.end() + 12].lstrip()
        unit = re.match(r'-?\s*([a-z"]+)', tail, re.I)
        if unit:
            low = unit.group(1).lower().strip(".,;:")
            if low == '"' or (len(low) > 1 and low in CuteVocab.SEARCHABLE_UNITS):
                continue
        found = _as_float(hit.group(1))
        if found is None or found <= 0:
            continue
        if best is None or found < best:
            best = found
            best_text = hit.group(1).replace(",", "")
    if best is None or best == now:
        return current
    return best_text


def _measure_met(evidence: str, figure: str) -> bool:
    """True when the evidence states this figure, comparing numerically.

    Numeric comparison treats a trailing zero as the same figure while keeping two
    neighbouring decimals apart — the shopper's wall is exact, not approximate.
    """
    try:
        want = float(figure)
    except ValueError:
        return True
    for found in re.findall(r"\d+(?:\.\d+)?", evidence or ""):
        try:
            if float(found) == want:
                return True
        except ValueError:
            continue
    return False


class CutePrompt:
    """Detailed operator playbooks with descriptions + examples (P_Prompt style)."""

    SKELETON = """
You are an AI Assistant operating as an EnvPack shopping agent.

# Contract
- Use only supplied tools. Stop when observation done=true.
- Obey max_steps; prefer 1-3 tool calls per turn.
- Never invent unobserved catalog facts, IDs, or conversion tables.

# Beats
discover → verify (view/compare) → first cart → inspect_stock → place_test_order.

# Firm vs soft
Firm walls beat soft wishes. Search VALUE tokens, not schema key names.

# Seal (critical — finish fast, but never on a stale snapshot)
- inspect_stock the final sku immediately before place_test_order.
- Never re-issue a call you already made with identical arguments; advance instead.
- After shocks: remove invalid → recover → inspect_stock → order only if still
  in stock and under the latest ceiling.
- Prefer finishing by ~turn 12. Unfinished events score zero.
- If justification is required, include ONLY the listed claim fields with values
  copied from view/inspect. If optional, omit justification.
""".strip()

    MATCH_RULES = """
# Matching rules (when comparing candidates)

1. Literal identifiers are non-negotiable
   - Model codes, exact sizes with numbers, quoted phrases, named product_id::sku
     must appear explicitly in title/facts/options.
   - Example OK: a stated numeric size matches the viewed option facts.
   - Example NOT OK: inferring a brand from a lookalike title alone.

2. Prefer structured facts over SEO titles
   - view/compare `facts` and option rows beat marketing title piles.
   - Example OK: facts.battery supports a battery soft axis.
   - Example NOT OK: trusting only "long battery" in a noisy title.

3. Category must match via category_path when available
   - A phone-like title is not enough if category_path is accessories.

4. Soft preferences never waive firm walls
   - Example OK: skip preferred brand that fails firm size/budget/stock.
   - Example NOT OK: buy cheaper off-spec item because it is nicer.

5. After shocks, re-verify current price and stock on the exact sku
   - Example OK: inspect_stock → still valid → place_test_order.
   - Example NOT OK: place_test_order on a pre-shock price snapshot.

6. An accessory FOR a product is never the product
   - A listing describing a case, cover, cable, mount, or spare part fails a
     request for the device, however well its title matches.
   - Price is the loudest tell: when a listing costs a small fraction of the
     stated budget for a device, assume accessory until the facts prove device.
   - Example OK: the facts describe the device's own hardware.
   - Example NOT OK: a silicone case whose model row names the phone.

7. The firm spec must be stated in the chosen sku's own facts
   - A number in the title or model string is marketing text and may be a decoy
     planted on a listing whose real facts disagree.
   - When the facts row for that attribute contradicts the request, the sku is
     wrong — keep looking instead of claiming the title.
   - Example OK: facts state the requested size for the selected option.
   - Example NOT OK: title says the requested size, facts say another.
""".strip()

    SEAL_RULES = """
# Seal checklist before place_test_order
1. Exact product_id + sku chosen and (usually) in cart
2. inspect_stock shows in_stock=true and price within the latest ceiling
3. Firm gates still hold on that sku
4. If justification is required: schema-legal claims only, values copied from
   view/inspect (not invented)
5. If justification is optional: omit justification entirely
""".strip()

    BRIEF = """
Author 6-10 short shopping-operator bullets for THIS episode only.
JSON only: {"bullets":["..."]}
Use only the shopper goal + Task rules. No products/IDs/claim-field laundry lists.
After price/stock shocks: recover → inspect_stock → place_test_order only if still valid.
""".strip()

    PLAY_INTENT = """
# TF1 · intent_decomposition
Split firm walls from soft wishes BEFORE tools.

## What counts as firm
- Required category, budget/ceiling, in-stock, and the stated measurable spec
  (size / capacity / similar figure the shopper named).
## What counts as soft
- Preferred brand, nice-to-haves, "ideally", comfort extras.

## Dual shortlists
Keep:
1) brand+firm candidates
2) firm-only backup (any brand)

## First-cart rule
Honor the preferred brand ONLY when a fully firm-compliant option exists.
Example OK: preferred brand meets every firm gate → first cart that brand.
Example NOT OK: cart a cheaper other-brand first while a compliant preferred brand was available.

## After shocks
Remove invalid lines, rebuild with the SAME firm walls, prefer brand again only if still firm-clean,
inspect_stock, then order (omit optional claims).
""".strip()

    PLAY_RECALL = """
# TF2 · retrieval_recall
Find the listing that matches the shopper's distinctive tokens / quoted phrase.

## Pass plan
1) search with type + distinctive tokens (keep numbers/units; drop filler words)
2) If hits are thin or off-category: ONE small reformulation
3) Shortlist several distinct product identities; drop accessories via category_path
4) view the best hit; inspect_stock; order only if still in stock & in budget

## Phrase goals
If the goal quotes a phrase, verify those tokens on title/facts before buying.
EVERY distinctive token must be present — a generic equivalent of the same kind
of product is not a match, and neither is an accessory for it.
Example OK: the quoted phrase appears on the chosen title.
Example NOT OK: ordering a near-category item that lacks the phrase.
Example NOT OK: the phrase names a branded item and you buy an unbranded one.

## Reformulation examples
- Drop stopwords; keep brand/type/number/unit tokens
- Switch search → filter(category, max_price, in_stock) when text search is noisy
""".strip()

    PLAY_CONSTRAINT = """
# TF3 · constraint_satisfaction
Every firm gate must hold on the EXACT ordered sku/option.

## Ask-gate
If a firm figure is withheld until asked:
- Turn 1 MUST call message for that exact figure
- Do NOT search/filter before the answer
Example OK: "What is the exact firm figure I must meet?"
Example NOT OK: searching first and guessing the withheld number.

Once the shopper answers, restate that figure verbatim and treat it as the hard
wall for the rest of the episode. It is the single thing this task is graded on:
search for it, and before ordering read it back off the chosen sku's own facts.
A candidate whose facts do not state that figure is wrong no matter how well it
fits budget, stock, brand, or category.

## Exactness
Never trust title alone — view the option row; inspect_stock that sku.
The request names a device, so a case/cover/cable/part listing never qualifies;
a price far under the ceiling is the usual sign you are looking at one.
Sellout → new product_id that still clears EVERY firm gate (no sibling trap).
A sibling option of the same product that drops a firm gate is not a recovery:
switch products rather than settling for the closest remaining option.

## Verification matrix — run this before every order
Take the firm gates one at a time and, for each, write down:
- the gate, as the shopper stated it
- WHERE you saw it hold: a view fact row, or the option row, or an inspect_stock
  field for the exact sku you are about to buy. A title is not a source; neither
  is the search result line, the price, or your own inference.
- that the sku you are ordering — not a sibling, not the product in general — is
  the one that source described
Any gate you cannot fill in this way is a gate you have not met, and the order is
wrong. Go and read the facts of the candidate that does meet it instead.
After a market event, every line of the matrix is stale: re-verify each gate on
the sku you now intend to buy before ordering it.

## Seal
inspect_stock → in stock + under ceiling + firm facts hold → place_test_order.
""".strip()

    PLAY_PREFERENCE = """
# TF4 · preference_reasoning
Hard walls first; then optimize soft axes the shopper actually named.

## Hard walls
category, budget/ceiling, in-stock, authenticity / not-accessory when a device is required.

## Soft axes
Only use-cases/preferences present in the shopper goal — invent none.
Compare ≥2 hard-compliant candidates via compare/view.
Prefer candidates whose RAW product facts support those soft axes
(e.g. battery / connectivity / materials rows), not title guesses alone.

## Dialogue budget
At most ONE short clarify message before first cart.
After cart / after shock: stop essays — recover → inspect_stock → place_test_order.

## Examples
OK: two in-budget candidates compared on named soft axes via raw facts; order the better soft fit.
NOT OK: looping inspect_cart while cart is empty after a price shock.
NOT OK: picking cheapest hard-valid item when a clearly better soft-fit survivor exists.
""".strip()

    PLAY_RANKING = """
# TF5 · ranking
Closed set of named product_id::sku candidates.

## Plan
1) view/compare the named variants directly (avoid open search) — batch them all
   in one turn before you cart anything, so the ranking is built on every row
2) Rank ONLY by the shopper's stated tradeoff priority
3) Never pick a strictly dominated option
4) First cart = current #1 (first cart is graded)
5) After market/cart events: re-rank survivors, cart NEW #1, inspect, order

The named set is closed. The order must be one of those product_id::sku pairs;
a listing you found by searching is not a candidate however good it looks, and
ordering one fails the episode even when every other check passes.

## Where the axis values live
The stated axes are frequently absent from the `facts` block. Read every axis
off ALL observed text for that candidate: the title string, the selected
variant's option label, then `facts`. A number printed only in the title still
counts as that candidate's advertised value.

If an axis is missing from `facts`, do NOT rank on a different attribute that
happens to be in `facts`. An attribute the shopper never named is not a
tiebreaker at any priority level, no matter how much better it looks.

## Dominance check before you order
List each surviving in-stock candidate as a row of the stated axes in priority
order. Your pick must not be beaten by any other row: if another candidate is
equal-or-better on every stated axis and better on at least one, it wins and
your pick was dominated. Re-read the titles before concluding a tie.

## place_test_order
product_id + sku only unless a why/justification is explicitly required.
Example OK: omit justification when Task rules say claims are optional.
Example NOT OK: inventing extra claim fields on a ranking episode.
""".strip()

    PLAY_RECOVERY = """
# TF6 · recovery
Never relax category, budget, or in-stock.

## Least-relax ladder
1) exact model + color
2) same model, other color
3) other models that still meet firm walls

Climb one level at a time and never skip one: a level is only exhausted once you
have seen it fail — sold out, over the ceiling, or absent from the catalog. If the
exact item is available, commit to it rather than shopping around.

Prefer structured model facts over conflicting marketing titles. Read colour from
the variant/option label first, then a product colour fact, and only then the
title; when it stays unresolved, do not invent one.

After every cart change: inspect_stock, then order only if still valid. Re-check
each firm wall on the sku you switched to — the walls the old pick satisfied say
nothing about the new one.

## Examples
OK: requested color sold out → same model other color still in budget.
NOT OK: jumping to a different category because it is cheaper.
NOT OK: ordering a sku that inspect_stock shows out of stock.
""".strip()

    PLAY_JUSTIFY = """
# TF7 · justification
place_test_order MUST include minimal grounded justification.claims.

## Field names
When the tool schema ships an enum, use only those fields. Otherwise obey the
schema's own written rule and name the specification field the way a product
catalog conventionally names that kind of measurement — the attribute, not the
catalog's local key spelling and not the shopper's surface wording. A quantity
the shopper quoted is still that attribute even if they wrote it as a shorthand.
Never use meta labels (title, category, description, summary, reason) and never
pad with feature flags you did not see as an explicit positive value.

Name the attribute on its own. Do not build the field out of the catalog's key
plus the attribute: the key is only the drawer the value was filed in, and a
field spelled that way is read as a field that does not exist.

## Minimal set — three claims at least
price + in_stock + the specification the shopper asked about (+ brand if they
named one). Two claims is a FAIL: if the shopper stated a measurable
requirement, the claim list MUST carry it under the field that names it.

When the catalog files the number under a broad container key, do not claim the
container: name the specific property the shopper asked about and give it that
observed value.

## Finding the spec claim
Scan that sku's `facts` rows and find the ONE row whose value answers the
measurement the shopper quoted. Claim it under the attribute name for that
measurement, with the value copied byte-for-byte from that row — same digits,
same spacing, same wording. Renaming the field is expected; rewriting the value
is not. If no `facts` row answers the shopper's measurement, this sku is the
wrong pick: choose one whose facts do show it rather than quoting the title.
If a row states that measurement but disagrees with what the shopper asked for,
the title was a decoy and the sku is wrong — switch instead of claiming it.

A measurement is only self-describing with its unit attached. When the listing
prints the value with its unit, quote it with the unit; when several rows carry
the same measurement, quote the row that states the unit rather than a bare
number. Never attach a unit the listing did not print.

When one row bundles several specs, quote only the part that answers the
shopper's measurement — the grader reads the claim as a statement about that one
attribute, so a compound row reads as the wrong value for it.

## Value copying
Copy the value verbatim from what the environment showed for that exact sku —
the raw `facts` rows first, and when `facts` does not carry it, the variant
`options` label or the listing title (that text is still observed evidence).
Example OK: shopper named a measurable; the sku's option label shows it; claim
that enum field with the shown value.
Example NOT OK: sending only price + in_stock when a measurable was requested;
claiming a firm attr you never saw on that sku; claiming over-budget price as fine.

## After shocks
inspect_stock the listing the event hit (this is what closes the event; reading
the cart instead does not) → remove it if it now breaks a firm wall → recover to
an in-budget in-stock survivor → inspect once → rebuild claims →
place_test_order. An unclosed event scores zero even with a valid final pick.
""".strip()

    TF7_KEY_PLAN = """
You extract MINIMAL justification.claim field names for ONE shopping episode.
Read the shopper goal lines and Task rules. Reply with JSON only:
{"firm_attr":"...","claim_keys":["price","in_stock","..."]}
Rules:
- Always include price and in_stock when claims are required.
- firm_attr: the SINGLE schema attribute for the firm measurable in the shopper request.
- Optional brand ONLY if the shopper named a brand.
- claim_keys MUST be chosen ONLY from allowed_fields in the user message.
- Do NOT invent fields outside allowed_fields.
- Maximum 4 keys. snake_case only. No unit conversions or synonym tables.
""".strip()

    TF7_FILL = """
Name the justification claim for ONE shopping requirement.
Input JSON: goal, allowed_fields (live enum, may be empty), schema_rule (the
tool's own wording for legal fields), have (fields already claimed), listing
(observed text for the exact sku being ordered).
Return JSON only: {"claims":[{"field":"...","value":"..."}, ...]}
Rules:
- Cover the specification the shopper stated that `have` is missing.
- Obey schema_rule. When allowed_fields is non-empty, field MUST come from it.
  When it is empty, name the specification field from the shopper's own request
  in snake_case (the attribute they asked about, not the catalog's key spelling).
- value MUST be a short token visibly present in `listing` text (title, facts
  rows, or option label). Quote it as it appears there.
- Do not restate price or in_stock — those are already handled.
- No guessing: if `listing` does not evidence the specification, return
  {"claims":[]}.
- At most 2 claims.
""".strip()

    TF7_ALIGN = """
Finalize justification.claims for a justification episode.
Input JSON: goal, claim_keys (allowed), proposed_claims.
Return JSON only: {"claims":[{"field":"...","value":"..."}, ...]}
Rules:
- Keep ONLY fields in claim_keys. Drop everything else.
- If a proposed field is not allowed but its value belongs to an allowed key,
  move the value onto that allowed key.
- Values pass through unchanged — no invented values or unit conversions.
- Dedupe by field. Minimal set only.
""".strip()

    @classmethod
    def family_playbook(cls, tf_id: str, facts: EpisodeFacts) -> str:
        body_map = {
            TF1: cls.PLAY_INTENT,
            TF2: cls.PLAY_RECALL,
            TF3: cls.PLAY_CONSTRAINT,
            TF4: cls.PLAY_PREFERENCE,
            TF5: cls.PLAY_RANKING,
            TF6: cls.PLAY_RECOVERY,
            TF7: cls.PLAY_JUSTIFY,
        }
        body = body_map.get(tf_id, "")
        extras: list[str] = []
        if facts.ceiling:
            extras.append(f"- Episode ceiling: {facts.ceiling}")
        if facts.phrase:
            extras.append(f'- Episode phrase: "{facts.phrase}"')
            toks = CuteVocab.search_tokens(facts.phrase, limit=6)
            if toks:
                extras.append("- Phrase tokens for search: " + " ".join(toks))
        if facts.brand_wish:
            extras.append(f"- Episode brand wish: {facts.brand_wish}")
        if facts.variants:
            extras.append("- Named candidates: " + ", ".join(facts.variants))
        if facts.priority_snip:
            extras.append(f"- Priority cue: {facts.priority_snip}")
        if facts.needs_ask:
            extras.append("- Ask-gate ACTIVE: message first for the withheld firm figure.")
        if facts.needs_why:
            extras.append(
                "- Justification.claims REQUIRED on place_test_order: price, in_stock, "
                "and the specification the shopper asked about."
            )
            if facts.claim_enum:
                extras.append("- Schema claim fields: " + ", ".join(facts.claim_enum))
            if facts.claim_rule:
                extras.append("- Schema claim rule: " + facts.claim_rule)
        elif facts.omit_claims:
            extras.append("- Omit optional justification claims.")
        if tf_id == TF2 and facts.goal:
            toks = CuteVocab.search_tokens(facts.goal, limit=8)
            if toks:
                extras.append("- Suggested search tokens: " + " ".join(toks))
        if not extras:
            return body
        return body + "\n\n## Episode read-off\n" + "\n".join(extras)


class TFPack(Protocol):
    tf_id: str
    dense: bool

    def tactics(self, facts: EpisodeFacts) -> str: ...
    def prepare(self, facts: EpisodeFacts) -> None: ...
    def opening(self, facts: EpisodeFacts) -> list[str]: ...
    def turn_hint(self, ctx: TurnCtx) -> str | None: ...
    def on_market(self, ctx: TurnCtx, kind: str) -> str | None: ...
    def hard_gate(self, ctx: TurnCtx) -> str | None: ...
    def scrub_order(
        self, args: dict[str, Any], facts: EpisodeFacts
    ) -> dict[str, Any]: ...


def _base_scrub(args: dict[str, Any], facts: EpisodeFacts) -> dict[str, Any]:
    """Default: strip optional claims unless TF7 needs them."""
    payload = dict(args)
    if facts.omit_claims and not facts.needs_why:
        payload.pop("justification", None)
    return payload


class TF1Intent:
    tf_id = TF1
    dense = False

    def tactics(self, facts: EpisodeFacts) -> str:
        return CutePrompt.family_playbook(TF1, facts)

    def prepare(self, facts: EpisodeFacts) -> None:
        return None

    def opening(self, facts: EpisodeFacts) -> list[str]:
        brand = f' "{facts.brand_wish}"' if facts.brand_wish else " preferred brand"
        return [
            f"AI Assistant · TF1: brand+firm shortlist and firm-only backup.{brand} "
            "wins first cart only when fully firm-compliant."
        ]

    def turn_hint(self, ctx: TurnCtx) -> str | None:
        if ctx.turn == 1:
            return self.opening(ctx.facts)[0]
        if ctx.saw_market and not ctx.ordered and ctx.turn >= 2:
            return (
                "AI Assistant · TF1: after shock, remove if invalid, pick the best "
                "firm-compliant option (brand if available), inspect_stock, order "
                "only if still valid."
            )
        return None

    def on_market(self, ctx: TurnCtx, kind: str) -> str | None:
        return (
            f"AI Assistant · TF1 market ({kind}): re-validate firm walls, inspect_stock, "
            "then place_test_order only if in stock & under ceiling (omit optional claims)."
        )

    def hard_gate(self, ctx: TurnCtx) -> str | None:
        return None

    def scrub_order(self, args: dict[str, Any], facts: EpisodeFacts) -> dict[str, Any]:
        return _base_scrub(args, facts)


class TF2Recall:
    tf_id = TF2
    dense = False

    def tactics(self, facts: EpisodeFacts) -> str:
        return CutePrompt.family_playbook(TF2, facts)

    def prepare(self, facts: EpisodeFacts) -> None:
        return None

    def opening(self, facts: EpisodeFacts) -> list[str]:
        return []

    def turn_hint(self, ctx: TurnCtx) -> str | None:
        if ctx.turn == 3 and ctx.search_count >= 1:
            tip = (
                f' Confirm phrase "{ctx.facts.phrase}" on title.'
                if ctx.facts.phrase
                else ""
            )
            return (
                "AI Assistant · TF2: one small reformulation so several distinct "
                f"identities appear; verify title/category; then order.{tip}"
            )
        return None

    def on_market(self, ctx: TurnCtx, kind: str) -> str | None:
        return (
            f"AI Assistant · TF2 market ({kind}): recover an in-budget in-stock listing, "
            "inspect_stock that exact sku, then place_test_order only if still valid."
        )

    def hard_gate(self, ctx: TurnCtx) -> str | None:
        return None

    def scrub_order(self, args: dict[str, Any], facts: EpisodeFacts) -> dict[str, Any]:
        return _base_scrub(args, facts)


class TF3Constraint:
    tf_id = TF3
    dense = False

    def tactics(self, facts: EpisodeFacts) -> str:
        return CutePrompt.family_playbook(TF3, facts)

    def prepare(self, facts: EpisodeFacts) -> None:
        return None

    def opening(self, facts: EpisodeFacts) -> list[str]:
        if not facts.needs_ask:
            return []
        extra = f" (ceiling {facts.ceiling})" if facts.ceiling else ""
        return [
            "AI Assistant · TF3 ask-gate: a firm requirement is withheld until you ask. "
            f"Call message THIS turn for that exact figure{extra}. Do not search yet."
        ]

    def turn_hint(self, ctx: TurnCtx) -> str | None:
        if (
            ctx.facts.needs_ask
            and ctx.turn == 1
            and not ctx.asked_spec
            and "message" not in ctx.used
        ):
            return self.opening(ctx.facts)[0]
        if ctx.saw_market and not ctx.ordered:
            return (
                "AI Assistant · TF3 sellout: switch to a NEW product_id that still meets "
                "every firm gate — inspect_stock that sku, then place_test_order "
                "only if in stock and in budget."
            )
        return None

    def on_market(self, ctx: TurnCtx, kind: str) -> str | None:
        return self.turn_hint(ctx)

    def hard_gate(self, ctx: TurnCtx) -> str | None:
        return None

    def scrub_order(self, args: dict[str, Any], facts: EpisodeFacts) -> dict[str, Any]:
        return _base_scrub(args, facts)


class TF4Preference:
    tf_id = TF4
    dense = False

    def tactics(self, facts: EpisodeFacts) -> str:
        return CutePrompt.family_playbook(TF4, facts)

    def prepare(self, facts: EpisodeFacts) -> None:
        return None

    def opening(self, facts: EpisodeFacts) -> list[str]:
        ceil = f" inside ceiling {facts.ceiling}" if facts.ceiling else ""
        return [
            "AI Assistant · TF4: compare ≥2 hard-compliant candidates on named soft axes "
            f"via compare/view before add_to_cart. First cart = preference winner{ceil}. "
            "After cart: inspect_stock then place_test_order — no message essays."
        ]

    def turn_hint(self, ctx: TurnCtx) -> str | None:
        f = ctx.facts
        if ctx.saw_market and not ctx.ordered and ctx.turn >= 2:
            return self._seal(f, cart_live=ctx.cart_live)
        if ctx.inspect_streak >= 2 and not ctx.ordered:
            return self._seal(f, cart_live=ctx.cart_live)
        if (ctx.first_cart or ctx.cart_live) and not ctx.ordered:
            if (
                "inspect_stock" in ctx.used
                or "inspect_cart" in ctx.used
                or ctx.turn >= 3
                or ctx.message_count >= 1
            ):
                return self._seal(f, cart_live=True)
        if ctx.turn >= max(8, int(f.max_steps * 0.4)) and not ctx.ordered:
            return self._seal(f, cart_live=ctx.cart_live)
        if ctx.turn <= 2 and "compare" not in ctx.used and not ctx.first_cart:
            return self.opening(f)[0]
        if ctx.message_count >= 1 and not ctx.first_cart and not ctx.ordered:
            return (
                "AI Assistant · TF4: clarification done. Compare/view, add_to_cart the "
                "winner, then place_test_order — no more messages."
            )
        return None

    def _seal(self, facts: EpisodeFacts, *, cart_live: bool) -> str:
        ceil = f" Stay under ceiling {facts.ceiling}." if facts.ceiling else ""
        if cart_live:
            action = (
                "Cart holds a pick — place_test_order NOW "
                "(one inspect is enough; never loop inspect)."
            )
        else:
            action = (
                "Cart empty after shock — add_to_cart a hard-valid soft-fit SKU, "
                "then place_test_order. Do not loop inspect on an empty cart."
            )
        return f"AI Assistant · TF4 seal: STOP messaging/inspect loops. {action}{ceil}"

    def on_market(self, ctx: TurnCtx, kind: str) -> str | None:
        del kind
        return self._seal(ctx.facts, cart_live=ctx.cart_live)

    def hard_gate(self, ctx: TurnCtx) -> str | None:
        if not ctx.saw_market or ctx.ordered:
            return None
        if ctx.inspect_streak < 1 and ctx.turn < 4:
            return None
        ceil = f" Stay under ceiling {ctx.facts.ceiling}." if ctx.facts.ceiling else ""
        if ctx.cart_live:
            return (
                "AI Assistant hard gate · TF4: after shock, inspect_stock the carted SKU; "
                "if price/stock invalid, remove and recover. Then place_test_order."
                + ceil
            )
        return (
            "AI Assistant hard gate · TF4: cart empty after shock. Recover a hard-valid "
            "soft-fit SKU via view/compare/add_to_cart, inspect_stock, then "
            "place_test_order." + ceil
        )

    def scrub_order(self, args: dict[str, Any], facts: EpisodeFacts) -> dict[str, Any]:
        return _base_scrub(args, facts)


class TF5Ranking:
    tf_id = TF5
    dense = False

    def tactics(self, facts: EpisodeFacts) -> str:
        return CutePrompt.family_playbook(TF5, facts)

    def prepare(self, facts: EpisodeFacts) -> None:
        return None

    def opening(self, facts: EpisodeFacts) -> list[str]:
        named = ", ".join(facts.variants) if facts.variants else "every named product_id::sku"
        pri = f" Priority cue: {facts.priority_snip}." if facts.priority_snip else ""
        return [
            f"AI Assistant · TF5: batch view/compare [{named}], rank on stated axes, "
            f"first-cart current #1.{pri} place_test_order = product_id+sku only."
        ]

    def turn_hint(self, ctx: TurnCtx) -> str | None:
        f = ctx.facts
        if ctx.saw_market and not ctx.ordered:
            named = ", ".join(f.variants) if f.variants else "surviving named candidates"
            return (
                f"AI Assistant · TF5 re-rank: state moved, so the old #1 is void. "
                f"Re-inspect [{named}], rebuild the ranking on the stated axes, then "
                "remove_from_cart the stale pick and add_to_cart the NEW #1 — the cart "
                "must hold the current winner before place_test_order. Read each axis "
                "off the title and the selected option label too, not just facts, and "
                "reject any candidate another survivor beats on every stated axis."
            )
        if ctx.turn >= 8 and not ctx.ordered:
            return (
                "AI Assistant · TF5 finish: ensure current #1 is carted and "
                "place_test_order immediately (product_id+sku only)."
            )
        if ctx.turn == 1:
            return self.opening(f)[0]
        if (
            ctx.turn >= 2
            and not ctx.first_cart
            and (ctx.viewed_named or "view" in ctx.used or "compare" in ctx.used)
        ):
            return (
                "AI Assistant · TF5: candidates inspected — add_to_cart only the "
                "current #1 (first cart graded)."
            )
        if ctx.first_cart and ctx.turn >= 3 and not ctx.ordered:
            return (
                "AI Assistant · TF5: seal with place_test_order(product_id, sku) only — "
                "no justification."
            )
        return None

    def on_market(self, ctx: TurnCtx, kind: str) -> str | None:
        del kind
        return self.turn_hint(ctx)

    def hard_gate(self, ctx: TurnCtx) -> str | None:
        if not ctx.saw_market or ctx.ordered:
            return None
        return (
            "AI Assistant hard gate · TF5: after the shock, re-rank the named set and "
            "make the cart hold the current #1 (remove the stale one first). Ordering "
            "the pre-shock leader scores zero even when it is in stock."
        )

    def scrub_order(self, args: dict[str, Any], facts: EpisodeFacts) -> dict[str, Any]:
        payload = dict(args)
        if not facts.needs_why:
            payload.pop("justification", None)
        return payload


class TF6Recovery:
    tf_id = TF6
    dense = False

    def tactics(self, facts: EpisodeFacts) -> str:
        return CutePrompt.family_playbook(TF6, facts)

    def prepare(self, facts: EpisodeFacts) -> None:
        return None

    def opening(self, facts: EpisodeFacts) -> list[str]:
        return []

    def turn_hint(self, ctx: TurnCtx) -> str | None:
        for obs in ctx.last_obs:
            if isinstance(obs, dict) and obs.get("error"):
                return (
                    f"AI Assistant · TF6: failure/market ({obs.get('error')}). Climb "
                    "exact model+color → other color → other models. Never relax "
                    "category/budget/stock."
                )
        if ctx.saw_market and not ctx.ordered:
            return (
                "AI Assistant · TF6 ladder: same model / other color before other models. "
                "Finish with place_test_order after inspect."
            )
        return None

    def on_market(self, ctx: TurnCtx, kind: str) -> str | None:
        return (
            f"AI Assistant · TF6 market ({kind}): climb the least-relax ladder, "
            "inspect_stock the survivor, place_test_order only if still in stock "
            "and under budget."
        )

    def hard_gate(self, ctx: TurnCtx) -> str | None:
        return None

    def scrub_order(self, args: dict[str, Any], facts: EpisodeFacts) -> dict[str, Any]:
        return _base_scrub(args, facts)


def _tf7_norm_key(raw: str) -> str:
    fl = str(raw or "").strip().lower()
    fl = fl.replace("-", "_").replace(" ", "_").replace("/", "_")
    fl = re.sub(r"[^a-z0-9_]+", "", fl)
    fl = re.sub(r"_+", "_", fl).strip("_")
    return fl


def _bare_attribute(field: str, observed: dict[str, str]) -> str:
    """Drop a catalog container key used as a prefix on a claim field name.

    The catalog's key for a row is the drawer the value was filed in, not the
    property the shopper asked about, so `<observed key>_<attribute>` reduces to
    `<attribute>`. The prefix has to be a key this listing actually showed, so
    nothing here encodes a vocabulary of its own.
    """
    parts = field.split("_")
    for cut in range(1, len(parts)):
        if "_".join(parts[:cut]) in observed:
            tail = "_".join(parts[cut:])
            if tail:
                return tail
    return field


def _tf7_key_in_text(key: str, text: str) -> bool:
    """True when a candidate claim key is attested in the shopper-facing text."""
    fl = _tf7_norm_key(key)
    if not fl or not text:
        return False
    blob = text.lower()
    spaced = fl.replace("_", " ")
    compact = re.sub(r"[^a-z0-9]+", "", blob)
    key_compact = fl.replace("_", "")
    if fl in blob.replace("-", "_") or spaced in blob:
        return True
    if len(key_compact) >= 3 and key_compact in compact:
        return True

    if len(fl) <= 4 and re.search(rf"\b{re.escape(fl)}\b", blob):
        return True
    if len(spaced) <= 4 and re.search(rf"\b{re.escape(spaced)}\b", blob):
        return True
    return False


def _tf7_sanitize_keys(
    raw_keys: list[str],
    *,
    firm_attr: str = "",
    goal: str = "",
    allowed: list[str] | None = None,
) -> list[str]:
    allowed_set = {_tf7_norm_key(x) for x in (allowed or []) if x}
    out: list[str] = []
    seen: set[str] = set()

    def add(key: str) -> bool:
        fl = _tf7_norm_key(key)
        if (
            not fl
            or fl in seen
            or fl in _CLAIM_DENY
            or not _TF7_KEY_RE.match(fl)
            or fl.count("_") > 2
        ):
            return False

        if allowed_set and fl not in allowed_set:
            return False
        seen.add(fl)
        out.append(fl)
        return True

    add("price")
    add("in_stock")
    firm = _tf7_norm_key(firm_attr)
    if firm and firm not in {"price", "in_stock", "brand", "category"}:

        if not allowed_set or firm in allowed_set:
            add(firm)


    brand_ok = bool(re.search(r"\bbrand\b", goal, re.I)) or bool(
        _RX_BRAND.search(goal)
    )
    for key in raw_keys:
        if _tf7_norm_key(key) == "brand" and brand_ok:
            add("brand")
            break

    has_firm = any(k not in {"price", "in_stock", "brand", "category"} for k in out)
    if (not has_firm) and allowed_set and goal:
        for cand in allowed_set:
            if cand in {"price", "in_stock", "brand", "category"} or cand in seen:
                continue
            if _tf7_key_in_text(cand, goal):
                add(cand)
                break
    return out[:4]


def _harvest_claim_evidence(facts: EpisodeFacts, obs: Any) -> None:
    """Collect live view/inspect values for claim keys (no suite tables/remaps)."""
    if not isinstance(obs, dict):
        return

    def put(key: str, value: Any, pid: str) -> None:
        fl = _tf7_norm_key(key)
        if not fl or value is None:
            return
        text = str(value).strip()
        if not text:
            return
        if fl == "in_stock":
            text = "true" if CuteVocab.is_yes(text) or value is True else (
                "false" if CuteVocab.is_no(text) or value is False else text
            )
        if pid:
            facts.claim_facts.setdefault(pid, {})[fl] = text

    def keep_text(pid: str, node: dict[str, Any]) -> None:
        """Remember the listing's own descriptive text for later grounding."""
        if not pid:
            return
        bits = [str(node.get("title") or ""), str(node.get("options") or "")]
        raw = node.get("facts")
        if isinstance(raw, dict):
            bits.extend(f"{k}: {v}" for k, v in raw.items())
        fresh = " | ".join(b for b in bits if b)
        if not fresh:
            return
        prior = facts.claim_text.get(pid, "")
        if fresh not in prior:
            facts.claim_text[pid] = (prior + " | " + fresh if prior else fresh)[:1800]

        spec = " | ".join(b for b in bits[1:] if b)
        if not spec:
            return
        held = facts.spec_text.get(pid, "")
        if spec not in held:
            facts.spec_text[pid] = (held + " | " + spec if held else spec)[:1800]

    def walk(node: Any, pid: str = "") -> None:
        if not isinstance(node, dict):
            return
        here = str(node.get("product_id") or "") or pid
        keep_text(here, node)
        if "price" in node:
            put("price", node.get("price"), here)
        if "in_stock" in node:
            put("in_stock", node.get("in_stock"), here)
        if node.get("brand"):
            put("brand", node.get("brand"), here)
        raw = node.get("facts")
        if isinstance(raw, dict):

            for k, v in raw.items():
                put(str(k), v, here)
            if here:
                facts.claim_last = here
        for key in ("observation", "compared", "product", "variants", "results"):
            child = node.get(key)
            if isinstance(child, dict):
                walk(child, here)
            elif isinstance(child, list):
                for item in child[:8]:
                    walk(item, here)

    walk(obs)
    inner = obs.get("observation")
    if isinstance(inner, dict):
        walk(inner)


def _claims_from_facts(
    facts: EpisodeFacts, product_id: str, *, limit: int = 4
) -> list[dict[str, str]]:
    """Build schema-legal claims from what was actually observed on this listing.

    Field names come from the live tool enum; values come from viewed facts.
    """
    enum = [_tf7_norm_key(k) for k in (facts.claim_enum or []) if k]
    if not enum:
        enum = [_tf7_norm_key(k) for k in facts.claim_keys if k]

    rows = dict(facts.claim_facts.get(str(product_id), {}))
    head = [k for k in ("price", "in_stock") if k in enum]
    rest = [k for k in enum if k not in head and k not in _CLAIM_DENY]


    planned = {_tf7_norm_key(k) for k in facts.claim_keys}
    rest.sort(
        key=lambda k: (
            0 if k in planned else 1,
            0 if _tf7_key_in_text(k, facts.goal) else 1,
            0 if _value_attested(rows.get(k, ""), facts.goal) else 1,
        )
    )
    out: list[dict[str, str]] = []
    for fl in [*head, *rest]:
        value = rows.get(fl, "")
        if value:
            out.append({"field": fl, "value": value})
        if len(out) >= limit:
            break
    return out


def _text_supports(value: str, listing: str) -> bool:
    """True when the value appears in the listing's own observed text."""
    tight = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    blob = re.sub(r"[^a-z0-9]+", "", str(listing).lower())
    return bool(tight) and tight in blob


def _attested_value(
    observed: dict[str, str], field: str, value: str, listing: str = ""
) -> str:
    """The observed text this claim quotes, or "" when the facts never showed it.

    Matches on the value as well as the key: a catalog files a measurement under its
    own column name while the shopper and the grader call the attribute something
    else, so a claim that renames the field still quotes observed evidence.
    """
    tight = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    if len(tight) < 2:
        return ""
    shown = observed.get(field)
    if shown and str(shown).strip():
        shown = str(shown).strip()
        tight_shown = re.sub(r"[^a-z0-9]+", "", shown.lower())
        if field in ("price", "in_stock") or tight_shown == tight:
            return shown


        if tight in tight_shown:
            return str(value).strip()


        if tight_shown in tight and _text_supports(value, listing):
            return str(value).strip()
        return shown
    for key, raw in observed.items():


        if key in ("price", "in_stock"):
            continue
        blob = re.sub(r"[^a-z0-9]+", "", str(raw).lower())
        if not blob or blob in ("true", "false"):
            continue
        if tight in blob:
            return str(value).strip()


        if blob in tight and _text_supports(value, listing):
            return str(value).strip()
    return ""


def _observed_fact_rows(last_obs: list[Any], *, limit: int = 10) -> str:
    """Echo the fact rows the environment just showed, so claims quote real text."""
    rows: list[str] = []
    seen: set[str] = set()

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 5 or len(rows) >= limit:
            return
        if isinstance(node, dict):
            block = node.get("facts")
            if isinstance(block, dict):
                for key, val in block.items():
                    text = f"{key}={val}"
                    if text not in seen and len(rows) < limit:
                        seen.add(text)
                        rows.append(text)
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for value in node[:8]:
                walk(value, depth + 1)

    for obs in last_obs or []:
        walk(obs)
    return ", ".join(rows)


def _tf7_llm_claims(
    facts: EpisodeFacts, product_id: str, have: list[str]
) -> list[dict[str, str]]:
    """Name the shopper's measurable in live-enum terms for this listing.

    Catalogs often expose a spec only in prose (a title or option label) under a
    key that differs from the schema field, so the naming is done by the model
    against the live enum. No mapping table lives here: every returned value must
    be visible in this listing's observed text or it is dropped.
    """
    listing = facts.claim_text.get(str(product_id), "")
    held = {_tf7_norm_key(k) for k in have}
    enum = [k for k in (facts.claim_enum or []) if _tf7_norm_key(k) not in held]
    if not listing:
        return []
    text = _inference_text(
        [
            {"role": "system", "content": CutePrompt.TF7_FILL},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "goal": facts.goal[:1500],
                        "allowed_fields": enum,
                        "schema_rule": facts.claim_rule,
                        "have": sorted(held),
                        "listing": listing,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        model=_light_model(),
    )
    obj = _parse_json_object(text)
    rows = obj.get("claims") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        return []
    allowed = {_tf7_norm_key(k) for k in enum}
    out: list[dict[str, str]] = []
    for item in rows[:2]:
        if not isinstance(item, dict):
            continue
        field = _tf7_norm_key(str(item.get("field") or ""))
        value = str(item.get("value") or "").strip()
        if not field or field in held or field in _CLAIM_DENY or not value:
            continue


        if (allowed and field not in allowed) or not _TF7_KEY_RE.match(field):
            continue
        if not _text_supports(value, listing):
            continue
        out.append({"field": field, "value": value})
    return out


def _value_attested(value: str, goal: str) -> bool:
    """True when an observed fact value shows up in the shopper's own wording."""
    blob = (goal or "").lower()
    if not value or not blob:
        return False
    compact = re.sub(r"[^a-z0-9]+", "", blob)
    for tok in re.findall(r"[a-z]{2,}|\d+(?:\.\d+)?", str(value).lower()):
        if tok in blob or (len(tok) >= 2 and tok in compact):
            return True
    return False


def _guess_firm_attr(goal: str, allowed: list[str]) -> str:
    """Pick a schema-legal firm attr only when its key appears in the shopper goal."""
    allowed_set = {_tf7_norm_key(x) for x in (allowed or [])}
    if not allowed_set or not goal:
        return ""
    for cand in allowed_set:
        if cand in {"price", "in_stock", "brand", "category"}:
            continue
        if _tf7_key_in_text(cand, goal):
            return cand
    return ""


class TF7Justification:
    tf_id = TF7
    dense = False

    def tactics(self, facts: EpisodeFacts) -> str:
        return CutePrompt.family_playbook(TF7, facts)

    def prepare(self, facts: EpisodeFacts) -> None:

        if not facts.needs_why:
            facts.claim_keys = []
            return
        firm = _guess_firm_attr(facts.goal, facts.claim_enum)
        raw = (
            ["brand"]
            if (
                re.search(r"\bbrand\b", facts.goal, re.I)
                or _RX_BRAND.search(facts.goal)
            )
            else []
        )
        keys = _tf7_sanitize_keys(
            raw,
            firm_attr=firm,
            goal=facts.goal,
            allowed=facts.claim_enum,
        )
        if "price" not in keys:
            keys.insert(0, "price")
        if "in_stock" not in keys:
            keys.insert(1 if keys[:1] == ["price"] else 0, "in_stock")
        facts.claim_keys = keys[:4]

    def opening(self, facts: EpisodeFacts) -> list[str]:
        return [self._coach(facts)]

    def _coach(self, facts: EpisodeFacts, rows: str = "") -> str:
        enum = (
            " Schema enum for this episode: " + ", ".join(facts.claim_enum) + "."
            if facts.claim_enum
            else ""
        )
        seen = (
            f" Fact rows just observed: {rows}. Quote from these — the part that "
            "answers the shopper, not the whole row."
            if rows
            else ""
        )
        return (
            "AI Assistant · TF7: place_test_order MUST include justification.claims — "
            "price, in_stock, AND the specification the shopper asked about."
            f"{enum} Price + in_stock alone is a FAIL. Find the facts row on that sku "
            "whose value answers the measurement they quoted, name the field by the "
            "conventional attribute name for that measurement (not the catalog's key "
            "spelling, not their wording), and copy that row's value byte-for-byte — "
            f"never from another listing.{seen}"
        )

    def _seal(self, facts: EpisodeFacts, rows: str = "") -> str:
        del facts
        seen = f" Fact rows just observed: {rows}." if rows else ""
        return (
            "AI Assistant · TF7 seal: inspect the final SKU once if needed, then "
            "place_test_order with claims for price, in_stock, and the specification "
            "the shopper asked about, each value copied from that sku's fresh facts."
            f"{seen} If no row states the measurement they asked for, or a row "
            "contradicts it, this is the wrong sku — switch before ordering."
        )

    @staticmethod
    def _rows_for_coach(ctx: TurnCtx) -> str:
        """Fact rows to quote from — this turn's if any, else the last listing read.

        The seal usually follows an inspect_stock or cart call, which carry no facts,
        so falling back to what was harvested keeps real text in front of the model.
        """
        fresh = _observed_fact_rows(ctx.last_obs)
        if fresh:
            return fresh
        stored = ctx.facts.claim_facts
        pid = ctx.facts.claim_last or (list(stored)[-1] if stored else "")
        latest = stored.get(pid) or {}
        rows = [f"{k}={v}" for k, v in latest.items() if k not in ("price", "in_stock")]
        return ", ".join(rows[:10])

    def turn_hint(self, ctx: TurnCtx) -> str | None:
        if ctx.ordered:
            return None
        rows = self._rows_for_coach(ctx)
        if ctx.saw_market and (ctx.first_cart or ctx.cart_live or ctx.turn >= 3):
            return self._seal(ctx.facts, rows)
        if ctx.turn in (2, 5) or (
            ctx.turn >= 2 and "view" not in ctx.used and "compare" not in ctx.used
        ):
            return self._coach(ctx.facts, rows)
        if ctx.first_cart and ctx.turn >= 4:
            if ctx.cart_live:
                return self._seal(ctx.facts, rows)
            return self._coach(ctx.facts, rows)
        return None

    def on_market(self, ctx: TurnCtx, kind: str) -> str | None:
        del kind
        return self._seal(ctx.facts, self._rows_for_coach(ctx))

    def hard_gate(self, ctx: TurnCtx) -> str | None:
        if not ctx.saw_market or ctx.ordered:
            return None
        keys = f" ({', '.join(ctx.facts.claim_keys)})" if ctx.facts.claim_keys else ""
        ceil = f" Obey ceiling {ctx.facts.ceiling}." if ctx.facts.ceiling else ""
        if ctx.inspect_streak < 1 and ctx.turn < 4:
            return (
                "AI Assistant hard gate · TF7: after shock, inspect_stock / view the "
                f"survivor, rebuild schema claims{keys}, then place_test_order."
                + ceil
            )
        if ctx.cart_live:
            return (
                "AI Assistant hard gate · TF7: cart live. Confirm price+stock still valid"
                f"{ceil} then place_test_order with claims{keys}. "
                "If over budget or OOS, remove and recover first."
            )
        return (
            "AI Assistant hard gate · TF7: cart empty after shock. "
            "add_to_cart a budget-valid in-stock SKU, inspect once, "
            f"place_test_order with claims{keys}.{ceil}"
        )

    def scrub_order(self, args: dict[str, Any], facts: EpisodeFacts) -> dict[str, Any]:
        payload = dict(args)
        block = payload.get("justification")
        block = block if isinstance(block, dict) else {}
        raw_claims = block.get("claims")


        raw_claims = raw_claims if isinstance(raw_claims, list) else []


        allowed = {_tf7_norm_key(k) for k in (facts.claim_enum or []) if k}
        rows_here = facts.claim_facts.get(str(args.get("product_id") or "")) or {}
        kept: list[dict[str, str]] = []
        used: set[str] = set()
        for item in raw_claims:
            if not isinstance(item, dict):
                continue
            fl = _tf7_norm_key(str(item.get("field") or ""))
            if not allowed:


                fl = _bare_attribute(fl, rows_here)
            if not fl or fl in _CLAIM_DENY or fl in used:
                continue
            if allowed and fl not in allowed:
                continue
            value = "" if item.get("value") is None else str(item.get("value")).strip()
            if fl == "in_stock" and CuteVocab.is_yes(value):
                value = "true"
            if value == "":
                continue
            used.add(fl)
            kept.append({"field": fl, "value": value})


        pid = str(payload.get("product_id") or "")
        final: list[dict[str, str]] = []
        seen_fields: set[str] = set()
        for row in kept + _claims_from_facts(facts, pid):
            if row["field"] in seen_fields:
                continue
            seen_fields.add(row["field"])
            final.append(row)
            if len(final) >= 4:
                break


        soft = {"price", "in_stock", "brand", "category"}
        if not any(row["field"] not in soft for row in final):

            final = final[:3]
            seen_fields = {row["field"] for row in final}
            for row in _tf7_llm_claims(facts, pid, list(seen_fields)):
                if row["field"] in seen_fields:
                    continue
                seen_fields.add(row["field"])
                final.append(row)
                if len(final) >= 4:
                    break


        observed = facts.claim_facts.get(pid) or {}
        listing = facts.claim_text.get(pid, "")
        attested: list[dict[str, str]] = []
        for row in final:
            shown = _attested_value(observed, row["field"], row["value"], listing)
            if shown:
                row["value"] = shown
                attested.append(row)
        payload["justification"] = {"claims": attested or final[:2]}
        return payload

    def _align(
        self,
        query: str,
        claims: list[dict[str, str]],
        claim_keys: list[str],
    ) -> list[dict[str, str]]:
        if not claims or not claim_keys:
            return claims
        text = _inference_text(
            [
                {"role": "system", "content": CutePrompt.TF7_ALIGN},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "goal": query[:6000],
                            "claim_keys": claim_keys,
                            "proposed_claims": claims,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            model=_light_model(),
        )
        obj = _parse_json_object(text)
        allowed = {k.lower() for k in claim_keys}
        source = obj.get("claims") if isinstance(obj, dict) else None
        rows = source if isinstance(source, list) else claims
        kept: list[dict[str, str]] = []
        used: set[str] = set()
        for item in rows:
            if not isinstance(item, dict):
                continue
            fl = _tf7_norm_key(str(item.get("field") or ""))
            value = "" if item.get("value") is None else str(item.get("value")).strip()
            if not fl or fl not in allowed or fl in used or value == "":
                continue
            used.add(fl)
            if fl == "in_stock" and CuteVocab.is_yes(value):
                value = "true"
            kept.append({"field": fl, "value": value})
        if kept:
            return kept
        return [
            c
            for c in claims
            if _tf7_norm_key(str(c.get("field") or "")) in allowed
        ]


PACKS: dict[str, Any] = {
    TF1: TF1Intent(),
    TF2: TF2Recall(),
    TF3: TF3Constraint(),
    TF4: TF4Preference(),
    TF5: TF5Ranking(),
    TF6: TF6Recovery(),
    TF7: TF7Justification(),
}


def _pack(tf: str) -> Any:
    return PACKS[tf]


_IDLE = (
    "AI Assistant: no tool call. Choose a supplied tool now.",
    "AI Assistant: still idle. Discover → verify → cart → order.",
)


def _llm_brief(query: str, route: Route) -> str:
    text = _inference_text(
        [
            {"role": "system", "content": CutePrompt.BRIEF},
            {
                "role": "user",
                "content": (
                    f"primary={route.primary}; satellite={route.satellite or '-'}.\n\n"
                    f"{query[:8000]}"
                ),
            },
        ],
        model=_light_model(),
    )
    obj = _parse_json_object(text)
    bullets: list[str] = []
    if isinstance(obj, dict) and isinstance(obj.get("bullets"), list):
        bullets = [str(b).strip() for b in obj["bullets"] if str(b).strip()][:12]
    if not bullets and text:
        bullets = [ln.strip() for ln in text.splitlines() if ln.strip()][:12]
    return "\n".join(f"- {b.lstrip('- ').strip()}" for b in bullets)


def _compose_system(route: Route) -> str:
    facts = route.facts
    primary = _pack(route.primary)
    parts = [
        CutePrompt.SKELETON,
        primary.tactics(facts),
    ]
    if route.satellite:
        parts.append(_pack(route.satellite).tactics(facts))
        parts.append(
            f"# Satellite note\n- Assist only; primary={route.primary} decides the seal."
        )
    if facts.brief:
        parts.append("# Episode brief\n" + facts.brief)
    fact_rows: list[str] = []
    if facts.ceiling:
        fact_rows.append(f"- spend ceiling: {facts.ceiling}")
    if facts.phrase:
        fact_rows.append(f'- phrase: "{facts.phrase}"')
    if facts.brand_wish:
        fact_rows.append(f"- brand wish: {facts.brand_wish}")
    if facts.variants:
        fact_rows.append("- named candidates: " + ", ".join(facts.variants))
    if facts.needs_ask:
        fact_rows.append("- ask-gate active")
    if facts.needs_why:
        fact_rows.append("- justification.claims required")
        if facts.claim_keys:
            fact_rows.append("- claim keys: " + ", ".join(facts.claim_keys))
    elif facts.omit_claims:
        fact_rows.append("- omit optional claims")
    if fact_rows:
        parts.append("# Read-off\n" + "\n".join(fact_rows))
    tools = ", ".join(facts.tool_names) if facts.tool_names else "(see schemas)"
    scored = ", ".join(f"{k}={v}" for k, v in route.scores.items() if v)
    parts.append(
        "# Runtime\n"
        f"- route: primary={route.primary} satellite={route.satellite or '-'}\n"
        f"- scores: {scored}\n"
        f"- trace: {','.join(route.trace)}\n"
        f"- tools: {tools}\n"
        f"- max_steps={facts.max_steps} max_calls_per_turn={facts.max_calls}"
    )
    return "\n\n".join(parts)


def _coach_reply() -> str:
    return (
        "AI Assistant: shopper spoke. Call message THIS turn with identity + firm "
        "evidence + price/stock before cart/order."
    )


def _coach_deadline(ctx: TurnCtx) -> str | None:
    if ctx.ordered:
        return None
    left = ctx.facts.max_steps - ctx.turn + 1


    if left > 8:
        return None
    return (
        f"AI Assistant deadline: turn {ctx.turn}, ~{left} step(s) left. "
        "inspect_stock the chosen sku, then place_test_order NOW "
        "only if in stock and under ceiling — unfinished events score zero."
    )


def _coach_glint(ctx: TurnCtx) -> str | None:
    bootstrap = ("search", "filter", "view", "compare", "message")
    if ctx.turn > 2 or any(n in ctx.used for n in bootstrap):
        return None
    if ctx.facts.phrase:
        tip = f' Start from phrase "{ctx.facts.phrase}" plus type/brand tokens.'
    else:
        tip = " Open with search/filter on value-side keywords from the goal."
    return "AI Assistant GLINT:" + tip + " Then view strongest hits."


def _turn_coach(ctx: TurnCtx) -> str | None:
    primary = _pack(ctx.route.primary)
    seal_over_chat = (
        ctx.route.primary == TF4
        and not ctx.ordered
        and (ctx.saw_market or ctx.first_cart or ctx.message_count >= 1 or ctx.turn >= 8)
    )
    if ctx.await_reply and not seal_over_chat:
        return _coach_reply()

    hint = primary.turn_hint(ctx)
    if hint:
        return hint


    if ctx.route.satellite:
        sat_hint = _pack(ctx.route.satellite).turn_hint(ctx)
        if sat_hint and ctx.turn <= 3:
            return sat_hint

    glint = _coach_glint(ctx)
    if glint:
        return glint

    if (
        not ctx.ordered
        and (ctx.first_cart or ctx.cart_live)
        and (not ctx.await_reply or seal_over_chat)
        and ("inspect_stock" in ctx.used or "inspect_cart" in ctx.used or ctx.turn >= 5)
        and ctx.turn >= 3
    ):
        extra = (
            " with grounded justification.claims covering every listed claim key."
            if ctx.route.primary == TF7
            else " (omit optional claims)."
        )
        return (
            "AI Assistant seal: cart is set. inspect_stock that exact sku, then "
            "place_test_order" + extra
        )

    if not ctx.ordered and ctx.turn >= 10:
        return _coach_deadline(ctx)

    if not ctx.ordered:
        return _coach_deadline(ctx)
    return None


def _pick_model(route: Route) -> str:
    """Dense model everywhere except plain recall, which the light model handles.

    Every family but retrieval_recall needs multi-constraint reasoning; the light
    model stalls on those (it re-emits identical calls at temperature 0).
    """
    forced = (getenv("SANDBOX_MODEL") or "").strip()
    if forced:
        return _resolve_model(forced)
    return _MODEL_LIGHT if route.primary == TF2 else _MODEL_DENSE


def _scrub_order(
    name: str,
    args: dict[str, Any],
    *,
    route: Route,
) -> dict[str, Any]:
    if name != "place_test_order":
        return dict(args)
    return _pack(route.primary).scrub_order(dict(args), route.facts)


def _needs_reply(text: str) -> bool:
    blob = (text or "").strip().lower()
    if not blob:
        return False
    if "?" in blob:
        return True
    cues = (
        r"^\s*which\b",
        r"^\s*what\b",
        r"^\s*does\b",
        r"^\s*did\b",
        r"\bcan you\b",
        r"\bcould you\b",
        r"^\s*wait\b",
        r"\bhold on\b",
        r"\bbefore you\b",
        r"\bre-?check\b",
    )
    return any(re.search(pat, blob) for pat in cues)


def _market_kind(obs: Any) -> str:
    if not isinstance(obs, dict):
        return ""
    blob = json.dumps(obs, ensure_ascii=False).lower()
    if "price" in blob and ("change" in blob or "increased" in blob or "decreased" in blob):
        return "price_change"
    if "sold" in blob and "out" in blob:
        return "sold_out"
    if "stock" in blob and ("change" in blob or "unavailable" in blob):
        return "stock_change"
    if obs.get("error"):
        return "error"
    return ""


def _market_target(obs: Any, depth: int = 0) -> tuple[str, str]:
    """The listing a market event named, so its post-event state can be observed."""
    if depth > 5 or not isinstance(obs, dict):
        return ("", "")
    event = obs.get("market_event")
    if isinstance(event, dict):
        target = event.get("target")
        if isinstance(target, dict):
            pid = str(target.get("product_id") or "")
            sku = str(target.get("sku") or "")
            if pid:
                return (pid, sku)
    for value in obs.values():
        if isinstance(value, dict):
            found = _market_target(value, depth + 1)
            if found[0]:
                return found
        elif isinstance(value, list):
            for item in value[:8]:
                found = _market_target(item, depth + 1)
                if found[0]:
                    return found
    return ("", "")


def _parse_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    fn = tool_call.get("function") if isinstance(tool_call, dict) else None
    if not isinstance(fn, dict):
        return {}
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
            return dict(obj) if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _obs_text(obs: Any, *, cap: int = _OBS_CAP) -> str:
    try:
        text = json.dumps(obs, ensure_ascii=False) if not isinstance(obs, str) else obs
    except (TypeError, ValueError):
        text = str(obs)
    return text if len(text) <= cap else text[:cap] + "…"


def _call_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for call in tool_calls:
        fn = call.get("function") if isinstance(call, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return names


def _fn_name(call: dict[str, Any]) -> str:
    fn = call.get("function") if isinstance(call, dict) else None
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return ""


def _fn_id(call: dict[str, Any], fallback: str) -> str:
    try:
        cid = call.get("id")
        return str(cid) if cid else fallback
    except (AttributeError, TypeError):
        return fallback


def _note_named(args: dict[str, Any], named: set[str]) -> bool:
    if not named:
        return False
    pid = str(args.get("product_id") or "")
    sku = str(args.get("sku") or "")
    if pid and sku and f"{pid}::{sku}" in named:
        return True
    if pid and any(tok.startswith(pid) for tok in named):
        return True
    return bool(pid)


class LoopGuard:
    """Break identical-action loops and keep the episode moving toward a seal.

    A greedy decoder handed the same context re-emits the same call forever, so
    repeats are counted and, once a batch adds nothing new, the call is rewritten
    to the next beat using listings that were actually observed.
    """

    __slots__ = ("counts", "rows", "seq", "focus", "breaks", "blocks")

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

        self.seq: list[tuple[str, str]] = []


        self.focus: list[str] = []
        self.breaks = 0
        self.blocks = 0

    def note_focus(self, args: dict[str, Any]) -> None:
        pid = str(args.get("product_id") or "")
        if not pid:
            return
        if pid in self.focus:
            self.focus.remove(pid)
        self.focus.insert(0, pid)

    @staticmethod
    def sign(name: str, args: dict[str, Any]) -> str:
        try:
            body = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            body = str(sorted(args.items()))
        return f"{name}|{body}"

    def bump(self, name: str, args: dict[str, Any]) -> int:
        key = self.sign(name, args)
        seen = self.counts.get(key, 0) + 1
        self.counts[key] = seen
        return seen

    def learn(self, obs: Any, depth: int = 0) -> None:
        """Remember every (product_id, sku) listing the environment has shown."""
        if depth > 5:
            return
        if isinstance(obs, list):
            for item in obs[:12]:
                self.learn(item, depth + 1)
            return
        if not isinstance(obs, dict):
            return
        pid = str(obs.get("product_id") or "")
        sku = str(obs.get("sku") or "")
        if pid and sku:
            key = (pid, sku)
            if key not in self.rows:
                self.seq.append(key)
            row = self.rows.setdefault(key, {})
            for field in ("price", "in_stock", "title", "brand"):
                if obs.get(field) is not None:
                    row[field] = obs.get(field)
        for value in obs.values():
            if isinstance(value, (dict, list)):
                self.learn(value, depth + 1)

    def invalid(self, key: tuple[str, str], facts: EpisodeFacts) -> bool:
        """True when the last observation of this listing failed stock or budget."""
        row = self.rows.get(key)
        if not row:
            return False
        if row.get("in_stock") is False:
            return True
        ceiling = _as_float(facts.ceiling)
        price = _as_float(row.get("price"))
        return ceiling is not None and price is not None and price > ceiling

    def pick(self, facts: EpisodeFacts) -> tuple[str, str] | None:
        """A valid listing the model already engaged with, else the top result.

        Never picks on price: the cheapest row is usually an accessory, not the
        product the shopper asked for.
        """
        named = {v.split("::", 1)[0] for v in facts.variants if v}
        pool = [k for k in self.seq if not named or k[0] in named]
        ranked = [k for pid in self.focus for k in pool if k[0] == pid]
        for key in [*ranked, *pool]:
            if not all(key):
                continue
            if not self.invalid(key, facts) and self.rows.get(key):
                return key
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _sku_valid(
    obs: Any, probe: tuple[str, str], facts: EpisodeFacts, depth: int = 0
) -> bool:
    """True when this observation shows the probed sku in stock and in budget."""
    if depth > 5:
        return False
    if isinstance(obs, list):
        return any(_sku_valid(item, probe, facts, depth + 1) for item in obs[:12])
    if not isinstance(obs, dict):
        return False
    pid, sku = probe
    here = str(obs.get("product_id") or "")
    row_sku = str(obs.get("sku") or "")
    if here == pid and (not row_sku or row_sku == sku) and "in_stock" in obs:
        if obs.get("in_stock") is False or CuteVocab.is_no(str(obs.get("in_stock"))):
            return False
        ceiling = _as_float(facts.ceiling)
        price = _as_float(obs.get("price"))
        if ceiling is not None and price is not None and price > ceiling:
            return False
        return True
    return any(
        _sku_valid(value, probe, facts, depth + 1)
        for value in obs.values()
        if isinstance(value, (dict, list))
    )


def _next_beat(
    guard: LoopGuard, state: dict[str, Any], facts: EpisodeFacts
) -> tuple[str, dict[str, Any]] | None:
    """Deterministic forward move: verify → cart → verify → order."""
    if facts.needs_ask and not state["asked_spec"]:
        return None
    pick = guard.pick(facts)
    if not pick:
        return None
    pid, sku = pick
    args = {"product_id": pid, "sku": sku}
    if pid not in guard.focus:

        return ("view", {"product_id": pid})
    if state["cart_key"] != pick:
        return ("add_to_cart", args)
    if state["verified"] != pick:
        return ("inspect_stock", args)
    return ("place_test_order", args)


def _rewrite_call(call: dict[str, Any], name: str, args: dict[str, Any]) -> None:
    fn = call.get("function")
    if isinstance(fn, dict):
        fn["name"] = name
        fn["arguments"] = json.dumps(args)


def _done(result_calls: list[Any]) -> bool:
    for call in result_calls:
        if not isinstance(call, dict):
            continue
        obs = call.get("observation")
        if isinstance(obs, dict) and obs.get("done"):
            return True
    return False


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """ORO Bench entry — route → TF pack(s) → EnvPack loop until done=true."""

    env = problem_data["environment"]
    binding = env["binding"]
    policy = env["policy_view"]
    query = str(policy.get("query") or problem_data.get("query") or "")
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, min(_MAX_CALLS_CAP, int(policy.get("max_calls_per_turn", 1))))
    max_steps = max(1, int(policy.get("max_steps", 20)))
    tools = policy.get("tools") or []

    route = route_episode(
        query, tools=tools, max_steps=max_steps, max_calls=max_calls
    )
    primary = _pack(route.primary)
    primary.prepare(route.facts)
    if route.satellite:
        _pack(route.satellite).prepare(route.facts)


    route.facts.brief = ""
    model = _pick_model(route)

    open_msgs: list[dict[str, Any]] = []
    for tip in primary.opening(route.facts):
        open_msgs.append({"role": "user", "content": tip})

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _compose_system(route)},
        {"role": "user", "content": query},
        *open_msgs,
    ]

    transcript: list[dict[str, Any]] = [
        {
            "role": "meta",
            "cutegirl": {
                "version": 7,
                "mixer": "tf_router",
                "primary": route.primary,
                "satellite": route.satellite,
                "scores": dict(route.scores),
                "trace": list(route.trace),
                "model": model,
                "provider": "openrouter",
                "claim_keys": list(route.facts.claim_keys),
                "brief": bool(route.facts.brief),
            },
        }
    ]

    state: dict[str, Any] = {
        "await_reply": False,
        "saw_market": False,
        "event_target": ("", ""),
        "event_probed": False,
        "event_holds": 0,
        "spec_holds": 0,
        "ceiling_said": "",
        "ordered": False,
        "first_cart": False,
        "cart_live": False,
        "asked_spec": False,
        "search_count": 0,
        "viewed_named": False,
        "message_count": 0,
        "inspect_streak": 0,
        "last_tools": [],
        "last_obs": [],

        "verified": None,
        "cart_key": None,
        "post_event_cart": False,
    }
    guard = LoopGuard()
    named = set(route.facts.variants)
    patch_boosted = False

    for turn in range(1, max_steps + 1):
        ctx = TurnCtx(
            route=route,
            turn=turn,
            last_tools=state["last_tools"],
            last_obs=state["last_obs"],
            await_reply=state["await_reply"],
            saw_market=state["saw_market"],
            ordered=state["ordered"],
            first_cart=state["first_cart"],
            cart_live=state["cart_live"],
            asked_spec=state["asked_spec"],
            search_count=state["search_count"],
            viewed_named=state["viewed_named"],
            message_count=state["message_count"],
            inspect_streak=state["inspect_streak"],
        )
        coach = _turn_coach(ctx)
        if coach:
            messages.append({"role": "user", "content": coach})
        gate = primary.hard_gate(ctx)
        if gate:
            messages.append({"role": "user", "content": gate})

        content = ""
        tool_calls: list[dict[str, Any]] = []
        for attempt in range(len(_IDLE) + 1):
            content, tool_calls = _chat(
                model=model,
                messages=messages,
                tools=tools,
                max_calls=max_calls,
            )
            if tool_calls:
                break
            transcript.append({"role": "assistant", "content": content})
            nudge = _IDLE[min(attempt, len(_IDLE) - 1)]
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": nudge})
        else:
            raise RuntimeError("CuteGirl: model never produced a tool call")

        messages.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls}
        )

        batch_id = f"{problem_id}-cg-{turn}"
        planned: list[list[Any]] = []
        for call in tool_calls:
            name = _fn_name(call)
            if not name:
                raise RuntimeError("CuteGirl: tool call missing function name")
            planned.append([call, name, _scrub_order(name, _parse_args(call), route=route)])


        if not state["ordered"] and all(
            guard.counts.get(guard.sign(n, a), 0) >= 2 for _, n, a in planned
        ):
            guard.breaks += 1
            beat = _next_beat(guard, state, route.facts)
            if beat:
                bname, bargs = beat
                bargs = _scrub_order(bname, bargs, route=route)
                call = planned[0][0]
                _rewrite_call(call, bname, bargs)
                planned = [[call, bname, bargs]]
                del tool_calls[1:]
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "AI Assistant STALL: that exact call was already made and the "
                        "observation has not changed. Repeating it wastes the step "
                        "budget — advance the beat now (verify a fresh candidate, cart "
                        "it, inspect_stock, then place_test_order). Do not re-issue an "
                        "identical call."
                    ),
                }
            )


        for row in planned:
            call, name, args = row
            if name != "place_test_order":
                continue
            key = (str(args.get("product_id") or ""), str(args.get("sku") or ""))


            if (
                state["event_target"][0]
                and not state["event_probed"]
                and state["event_holds"] < 3
            ):
                state["event_holds"] += 1
                probe = {
                    "product_id": state["event_target"][0],
                    "sku": state["event_target"][1],
                }
                _rewrite_call(call, "inspect_stock", probe)
                row[1], row[2] = "inspect_stock", probe
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "AI Assistant SEAL GATE: order held — the listing the market "
                            "event hit has not had its stock and price re-read since it "
                            "moved, so the event is still open and the episode scores "
                            "zero. Read this inspect_stock result, then order your "
                            "replacement."
                        ),
                    }
                )
                continue


            missing = [
                fig
                for fig in (route.facts.spec_wants if route.primary == TF3 else [])
                if not _measure_met(route.facts.spec_text.get(key[0], ""), fig)
            ]


            if missing and not state["spec_holds"] and turn <= max_steps - 6:
                state["spec_holds"] += 1


                fit = [
                    seen
                    for seen in guard.seq
                    if all(
                        _measure_met(route.facts.spec_text.get(seen[0], ""), fig)
                        for fig in route.facts.spec_wants
                    )
                    and route.facts.spec_text.get(seen[0])
                    and not guard.invalid(seen, route.facts)
                ]
                lead = (
                    "The listings you already read that DO state it: "
                    + ", ".join(f"{p}::{s}" for p, s in fit[:3])
                    + ". Pick from these."
                    if fit
                    else (
                        "No listing you have read states it yet — search that figure "
                        "and read the facts of the results before ordering again."
                    )
                )
                probe = (
                    {"product_id": fit[0][0], "sku": fit[0][1]}
                    if fit
                    else {"product_id": key[0]}
                )
                nxt = "inspect_stock" if fit else "view"
                _rewrite_call(call, nxt, probe)
                row[1], row[2] = nxt, probe
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "AI Assistant SEAL GATE: order held — nothing observed on "
                            f"this listing states {', '.join(missing)}, which the shopper "
                            "made non-negotiable, so it is the wrong item however well it "
                            "fits budget, stock and brand. A figure in the title does not "
                            "count: titles are copy and can disagree with the variant. "
                            f"{lead} Keep the shopper's category and brand, stay under the "
                            "ceiling, and never fall back to an accessory."
                        ),
                    }
                )
                continue


            if guard.blocks >= 2:
                continue


            if guard.invalid(key, route.facts):
                guard.blocks += 1
                probe = {"product_id": key[0], "sku": key[1]}
                _rewrite_call(call, "inspect_stock", probe)
                row[1], row[2] = "inspect_stock", probe
                ceiling = route.facts.ceiling or "the stated ceiling"
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "AI Assistant SEAL GATE: order held — the last observation of "
                            f"this sku was out of stock or above {ceiling}. Ordering it "
                            "fails the budget/stock check outright. Read this "
                            "inspect_stock result: if it still fails, recover to another "
                            "listing that meets every firm wall, cart that one, and order "
                            "it instead."
                        ),
                    }
                )
                continue


            if route.facts.needs_why and key[0] not in route.facts.claim_facts:
                guard.blocks += 1
                probe = {"product_id": key[0]}
                _rewrite_call(call, "view", probe)
                row[1], row[2] = "view", probe
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "AI Assistant SEAL GATE: order held — no facts observed for "
                            "this listing, so justification.claims cannot be grounded. "
                            "Read this view result, confirm the firm attribute holds on "
                            "the exact sku, then place_test_order with claims copied "
                            "from it."
                        ),
                    }
                )
                continue
            stale = state["verified"] != key
            recommit = state["saw_market"] and not state["post_event_cart"]
            if not (stale or recommit):
                continue
            guard.blocks += 1
            probe = {"product_id": key[0], "sku": key[1]}
            _rewrite_call(call, "inspect_stock", probe)
            row[1], row[2] = "inspect_stock", probe
            why = (
                "the cart has not been re-committed since the market moved"
                if recommit
                else "this sku has no fresh stock/price check"
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"AI Assistant SEAL GATE: order held because {why}. "
                        "Read this inspect_stock result: if it is out of stock or over "
                        "the ceiling, recover to another valid listing and re-cart it; "
                        "otherwise re-cart the winner if needed and place_test_order "
                        "next turn."
                    ),
                }
            )

        calls_out: list[dict[str, Any]] = []
        for idx, (call, name, args) in enumerate(planned, start=1):
            guard.bump(name, args)
            if name in ("view", "compare", "inspect_stock", "add_to_cart"):
                guard.note_focus(args)
            calls_out.append(
                {
                    "call_id": f"{batch_id}-{idx}",
                    "action": {"name": name, "args": args},
                }
            )
            if name in ("search", "filter"):
                state["search_count"] += 1
                state["inspect_streak"] = 0
            elif name == "message":
                state["await_reply"] = False
                state["asked_spec"] = True
                state["message_count"] += 1
                state["inspect_streak"] = 0
            elif name == "add_to_cart":
                state["first_cart"] = True
                state["cart_live"] = True
                state["inspect_streak"] = 0
                state["cart_key"] = (
                    str(args.get("product_id") or ""),
                    str(args.get("sku") or ""),
                )
                if state["saw_market"]:
                    state["post_event_cart"] = True
            elif name == "remove_from_cart":
                state["saw_market"] = True
                state["cart_live"] = False
                state["inspect_streak"] = 0
                state["verified"] = None
                state["cart_key"] = None
                if not patch_boosted and route.primary != TF6:

                    patch_boosted = True
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "AI Assistant: cart shock — climb a least-relax recovery "
                                "ladder without relaxing category/budget/stock."
                            ),
                        }
                    )
                mkt = primary.on_market(
                    TurnCtx(
                        route=route,
                        turn=turn,
                        last_tools=state["last_tools"],
                        last_obs=state["last_obs"],
                        await_reply=state["await_reply"],
                        saw_market=True,
                        ordered=state["ordered"],
                        first_cart=state["first_cart"],
                        cart_live=False,
                        asked_spec=state["asked_spec"],
                        search_count=state["search_count"],
                        viewed_named=state["viewed_named"],
                        message_count=state["message_count"],
                        inspect_streak=0,
                    ),
                    "remove_from_cart",
                )
                if mkt:
                    messages.append({"role": "user", "content": mkt})
            elif name == "place_test_order":
                state["ordered"] = True
                state["inspect_streak"] = 0
            elif name in ("inspect_stock", "inspect_cart"):
                state["inspect_streak"] += 1
            elif name in ("view", "compare"):
                state["inspect_streak"] = 0
                if _note_named(args, named):
                    state["viewed_named"] = True
            else:
                state["inspect_streak"] = 0

        env_body = {
            **binding,
            "call_id": batch_id,
            "idempotency_key": batch_id,
            "turn": turn,
            "calls": calls_out,
        }
        result = _http.post("/environment/call", json_data=env_body)
        if result is None:
            raise RuntimeError("CuteGirl: /environment/call returned empty")

        transcript.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
                "environment_result": result,
                "cutegirl_turn": {
                    "primary": route.primary,
                    "satellite": route.satellite,
                    "turn": turn,
                    "state": {
                        k: state[k]
                        for k in (
                            "await_reply",
                            "saw_market",
                            "ordered",
                            "first_cart",
                            "cart_live",
                            "asked_spec",
                            "search_count",
                            "viewed_named",
                            "message_count",
                            "inspect_streak",
                        )
                    },
                },
            }
        )

        state["last_tools"] = _call_names(tool_calls)
        state["last_obs"] = []
        result_calls = result.get("calls") or []
        for idx, (tool_call, call_result) in enumerate(
            zip(tool_calls, result_calls), start=1
        ):
            obs = (
                call_result.get("observation")
                if isinstance(call_result, dict)
                else None
            )
            state["last_obs"].append(obs)
            _harvest_claim_evidence(route.facts, obs)
            guard.learn(obs)
            called = planned[idx - 1] if idx - 1 < len(planned) else None

            if called and called[1] == "inspect_stock":
                probe = (
                    str(called[2].get("product_id") or ""),
                    str(called[2].get("sku") or ""),
                )
                if all(probe):
                    if _sku_valid(obs, probe, route.facts):
                        state["verified"] = probe

                    if probe[0] == state["event_target"][0]:
                        state["event_probed"] = True
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _fn_id(tool_call, f"{batch_id}-{idx}"),
                    "content": _obs_text(obs),
                }
            )
            kind = _market_kind(obs)
            if kind:
                if not state["saw_market"]:

                    state["verified"] = None
                    state["post_event_cart"] = False
                state["saw_market"] = True

                target = _market_target(obs)
                if target[0] and target != state["event_target"]:
                    state["event_target"] = target
                    state["event_probed"] = False
                if not state["ordered"]:
                    tip = primary.on_market(
                        TurnCtx(
                            route=route,
                            turn=turn,
                            last_tools=state["last_tools"],
                            last_obs=state["last_obs"],
                            await_reply=state["await_reply"],
                            saw_market=True,
                            ordered=False,
                            first_cart=state["first_cart"],
                            cart_live=state["cart_live"],
                            asked_spec=state["asked_spec"],
                            search_count=state["search_count"],
                            viewed_named=state["viewed_named"],
                            message_count=state["message_count"],
                            inspect_streak=state["inspect_streak"],
                        ),
                        kind,
                    )
                    if tip:
                        messages.append({"role": "user", "content": tip})

        spoken_msg = result.get("user_message")
        if spoken_msg and spoken_msg.get("content"):
            spoken = str(spoken_msg["content"])
            messages.append({"role": "user", "content": spoken})
            for figure in _stated_measures(spoken):

                if figure not in route.facts.spec_wants:
                    route.facts.spec_wants.append(figure)
            said = state["ceiling_said"] or route.facts.ceiling
            tighter = _lowered_ceiling(spoken, said)
            if tighter != said:


                state["ceiling_said"] = tighter
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"AI Assistant: the ceiling is now {tighter}. The order is "
                            "graded against this figure, not the opening one — anything "
                            "priced above it fails outright."
                        ),
                    }
                )
            if _needs_reply(spoken):
                state["await_reply"] = True

        if (
            route.primary in {TF4, TF7}
            and not state["ordered"]
            and state["inspect_streak"] >= 2
        ):
            tip = primary.hard_gate(
                TurnCtx(
                    route=route,
                    turn=turn,
                    last_tools=state["last_tools"],
                    last_obs=state["last_obs"],
                    await_reply=state["await_reply"],
                    saw_market=state["saw_market"],
                    ordered=False,
                    first_cart=state["first_cart"],
                    cart_live=state["cart_live"],
                    asked_spec=state["asked_spec"],
                    search_count=state["search_count"],
                    viewed_named=state["viewed_named"],
                    message_count=state["message_count"],
                    inspect_streak=state["inspect_streak"],
                )
            )
            if tip:
                messages.append({"role": "user", "content": tip})

        if _done(result_calls):
            break

    return transcript


__all__ = ["agent_main", "route_episode", "PACKS"]
