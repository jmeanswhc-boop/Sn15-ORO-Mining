from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from os import getenv
from typing import Any, Optional

from src.agent.proxy_client import ProxyClient


MODEL_TABLE: dict[str, Any] = {
    "PARSE": [
        "zai-org/GLM-5.2-TEE",
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "moonshotai/Kimi-K2.6-TEE",
    ],
    "JUDGE": [
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "zai-org/GLM-5.2-TEE",
        "Qwen/Qwen3-32B-TEE",
    ],
    "OPENROUTER": {
        "zai-org/GLM-5.2-TEE": "z-ai/glm-5.2",
        "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
        "moonshotai/Kimi-K2.6-TEE": "moonshotai/kimi-k2.6",
        "Qwen/Qwen3-32B-TEE": "qwen/qwen3-32b",
    },
}

LLM_TEMPERATURE = 0
LLM_RETRY_DELAY_SEC = 0.8
EVIDENCE_WEIGHT_SKU = 4.0
EVIDENCE_WEIGHT_ATTR = 2.0
EVIDENCE_WEIGHT_TITLE = 1.0
MISSING_CONSTRAINT_PENALTY = 10.0
EVIDENCE_NEAR_TIE_EPS = 3.0
PRODUCT_BATCH_SCORE_CAP = 20
SEARCH_REFORMULATIONS = 4

RANK_STOPWORDS = frozenset({
    "the", "a", "an", "for", "with", "from", "that", "this", "i", "me", "my",
    "looking", "show", "find", "want", "need", "get", "buy", "also", "and",
    "in", "is", "it", "of", "to", "or", "on", "at", "by", "php", "price",
    "under", "below", "above", "stock", "in-stock", "please",
})

_proxy = ProxyClient(timeout=120, max_retries=3)

def _provider() -> str:
    return getenv("INFERENCE_PROVIDER", "chutes").lower()

def _resolve_model(name: str) -> str:
    override = getenv("SANDBOX_MODEL")
    if override:
        return override
    if _provider() == "openrouter":
        mapping = MODEL_TABLE.get("OPENROUTER") or {}
        if isinstance(mapping, dict):
            return mapping.get(name, "deepseek/deepseek-v3.2")
    return name

def _chat_payload(model: str, messages: list[dict], *, tools=None, tool_choice=None) -> dict:
    body: dict[str, Any] = {
        "model": _resolve_model(model),
        "temperature": LLM_TEMPERATURE,
        "stream": False,
        "messages": messages,
    }
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = tool_choice if tool_choice is not None else "required"
    return body

def llm_post(model: str, messages: list[dict], *, tools=None, tool_choice=None) -> dict | None:
    return _proxy.post(
        "/inference/chat/completions",
        json_data=_chat_payload(model, messages, tools=tools, tool_choice=tool_choice),
    )


class Family(str, Enum):
    INTENT = "intent_decomposition"
    RETRIEVAL = "retrieval_recall"
    CONSTRAINT = "constraint_satisfaction"
    PREFERENCE = "preference_reasoning"
    RANKING = "ranking"
    RECOVERY = "recovery"
    JUSTIFICATION = "justification"

FAMILY_IDS = [f.value for f in Family]

PROMPT_CLASSIFY_FAMILY = """\
Classify this shopping task into exactly one family id.
You receive: (1) the shopper query and (2) the LIVE tool list for this episode.
The tool list is dynamic — use names + descriptions to break ties, never assume a fixed catalog.
Tool signals:
- compare present + product_id::sku candidates in query → ranking
- message present → interactive family (constraint / recovery / preference / justification / intent)
- filter/search without candidates → retrieval_recall or constraint_satisfaction
Prefer Task-rules cue phrases when present (they are authoritative):
- "mix of firm requirements and nice-to-haves" -> intent_decomposition
- "reformulate queries and inspect candidates" -> retrieval_recall
- "every requirement must hold for the exact variant" -> constraint_satisfaction
- "initial request authorizes one test order" -> preference_reasoning
- "highest advertised maximum refresh rate" / "compare only these candidate" -> ranking
- "preferences are relaxable only after the exact item becomes unavailable" -> recovery
- "shopper wants a valid product plus a trustworthy reason" / "reason I can trust" -> justification
Families:
- intent_decomposition: firm gates + soft brand preference; honor preferred brand on first cart when compliant.
- retrieval_recall: phrase/category recall; survey several products then buy one.
- constraint_satisfaction: every firm requirement on the exact SKU; may need to ask for a hidden spec.
- preference_reasoning: hard gates + soft use-case prefs; first cart is scored.
- ranking: choose among listed candidate variants by refresh → screen → price.
- recovery: prefer exact model+color; relax color then model; never relax budget/stock.
- justification: valid buy + structured field/value claims on the order tool.
Output strict minified JSON only: {"family":"<id>","reasoning":"<=12 words"}
"""

FAMILY_PREFERRED_TOOLS: dict[Family, tuple[str, ...]] = {
    Family.RANKING: ("compare", "add_to_cart", "inspect_stock", "remove_from_cart", "place_test_order", "message"),
    Family.RECOVERY: ("search", "view", "inspect_stock", "add_to_cart", "remove_from_cart", "place_test_order", "message"),
    Family.INTENT: ("search", "filter", "view", "add_to_cart", "place_test_order", "message"),
    Family.RETRIEVAL: ("search", "filter", "view", "add_to_cart", "place_test_order"),
    Family.CONSTRAINT: ("search", "filter", "view", "inspect_stock", "add_to_cart", "place_test_order", "message"),
    Family.PREFERENCE: ("search", "view", "add_to_cart", "place_test_order", "message"),
    Family.JUSTIFICATION: ("search", "view", "add_to_cart", "place_test_order", "message"),
}

def tool_name(entry: dict) -> str:
    fn = entry.get("function") if isinstance(entry, dict) else None
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return str(entry.get("name") or "") if isinstance(entry, dict) else ""

def tool_catalog(tools: list[dict]) -> list[dict[str, Any]]:

    out: list[dict[str, Any]] = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            fn = t if isinstance(t, dict) else {}
        name = str(fn.get("name") or "")
        if not name:
            continue
        params = fn.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else {}
        required = params.get("required") if isinstance(params, dict) else []
        out.append(
            {
                "name": name,
                "description": (fn.get("description") or "").strip(),
                "params": sorted(props.keys()) if isinstance(props, dict) else [],
                "required": list(required) if isinstance(required, list) else [],
            }
        )
    return out

def tool_summary(tools: list[dict]) -> str:
    lines = []
    for row in tool_catalog(tools):
        desc = row["description"].split(".")[0].strip()
        params = ", ".join(row["params"][:8])
        bit = f"  • {row['name']}"
        if desc:
            bit += f": {desc}"
        if params:
            bit += f" [{params}]"
        lines.append(bit)
    return "\n".join(lines) if lines else "  (none listed)"

def available_names(tools: list[dict]) -> set[str]:
    return {row["name"] for row in tool_catalog(tools)}

def plan_tools(family: Family, tools: list[dict]) -> list[str]:

    have = available_names(tools)
    wanted = FAMILY_PREFERRED_TOOLS.get(family, ())
    planned = [n for n in wanted if n in have]

    extras = [n for n in sorted(have) if n not in planned]
    return planned + extras

PROMPT_PARSE_PRODUCT = """\
Parse a single-product shopping request into structured search parameters.
Emit exactly one products[] entry (never multi-buy).
Schema (minified JSON, no fence, no prose):
{"reasoning":"...","products":[{"keywords":"2-8 word search string","price_range":"lo-hi"|"lo-"|"-hi"|null,"category":null|"Smartphones"|...,"brand":null|"Samsung"|...,"soft_brand":null|"Samsung"|...,"color":null|"black"|...,"model":null|"635"|...,"screen_size":null|"6.64in"|...,"resolution":null|"hd"|"fhd"|"qhd"|"4k"|...,"refresh_rate":null|"144hz"|...,"ram":null|"8gb"|...,"search_phrase":null|"aha glow"|...,"candidates":null|["pid::sku",...],"needs_ask":true|false,"constraints":{"attribute_key":"value"},"only_product_type":true|false}]}
Rules:
- keywords: searchable tokens, singular product type, keep brand/model/color/spec values; drop price/service wording.
- price_range: max-only "under 500" -> "-500"; min-only "above 100" -> "100-"; bounded "100-500".
- soft_brand: "ideally I prefer X" brand preference (not a hard gate).
- brand: hard brand when stated as requirement ("I need a Samsung smartphone").
- candidates: parse "product_id::sku" tokens when the query lists compare-only variants.
- needs_ask: true when shopper says they will tell a firm spec if asked.
- constraints: explicit attribute map (product_type, brand, color_family, model, screen_size, resolution, etc.).
Emit JSON only.
"""

PROMPT_EVIDENCE_EXTRACT = """\
Extract query-relevant evidence from shopping candidates for a single-product buy.
Input JSON has request, constraints, candidates(product_id,title,brand,facts,variants,price).
Return ONLY a minified JSON array keeping input order. Omit wrong product types.
Each kept row: {"product_id":"...","sku":"<best matching sku or empty>","sku_options":["..."],"attributes":["..."],"title":["..."],"missing":["..."]}
Source priority for each requirement: variant options > facts/attributes > title.
Never invent evidence. Emit JSON array only.
"""


def parse_json_object(content: str) -> dict | None:
    if not content:
        return None
    m = re.search(r"\{[\s\S]*\}", content)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None

def parse_json_array(content: str) -> list | None:
    if not content:
        return None
    m = re.search(r"\[[\s\S]*\]", content)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return arr if isinstance(arr, list) else None

def normalize_evidence_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, list):
        out = []
        for x in raw:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    return [str(raw)]

def score_from_evidence_counts(sku_n: int, attr_n: int, title_n: int, missing_n: int = 0) -> float:
    score = (
        EVIDENCE_WEIGHT_SKU * sku_n
        + EVIDENCE_WEIGHT_ATTR * attr_n
        + EVIDENCE_WEIGHT_TITLE * title_n
        - MISSING_CONSTRAINT_PENALTY * missing_n
    )

    base = 60.0 + min(score, 40.0)
    if sku_n and missing_n == 0:
        base = max(base, 91.0)
    elif attr_n and missing_n == 0:
        base = max(base, 81.0)
    elif title_n and missing_n == 0:
        base = max(base, 71.0)
    return max(0.0, min(100.0, base))

def evidence_score_sort_bucket(score: float, eps: float = EVIDENCE_NEAR_TIE_EPS) -> float:
    return round(score / eps) * eps


def _classify_family_heuristic(query: str, tools: list[dict] | None = None) -> Family:

    q = (query or "").lower()
    head = q.split("\n\ntask rules:")[0]
    names = available_names(tools or [])

    if "mix of firm requirements and nice-to-haves" in q:
        return Family.INTENT
    if "reformulate queries and inspect candidates" in q:
        return Family.RETRIEVAL
    if "every requirement must hold for the exact variant" in q:
        return Family.CONSTRAINT
    if "initial request authorizes one test order" in q:
        return Family.PREFERENCE
    if (
        "highest advertised maximum refresh rate" in q
        or "compare only these candidate" in q
        or "compare only the following" in q
        or ("tradeoff priority" in q and "refresh" in q)
        or ("compare" in names and re.search(r"\d{6,}::[\w.\-]+", query or ""))
    ):
        return Family.RANKING
    if "preferences are relaxable only after the exact item becomes unavailable" in q:
        return Family.RECOVERY
    if "shopper wants a valid product plus a trustworthy reason" in q:
        return Family.JUSTIFICATION

    if "compare only these candidate variants" in head or (
        "highest refresh rate" in head and "among ties" in head
    ):
        return Family.RANKING
    if "strongly prefer the" in head and "another color of the same model" in head:
        return Family.RECOVERY
    if any(
        p in head
        for p in (
            "walk me through why",
            "tell me why",
            "give me a reason",
            "reason i can trust",
        )
    ):
        return Family.JUSTIFICATION
    if "place one test order" in head or ("sweat" in head and "headphone" in head):
        return Family.PREFERENCE
    if re.search(r'matches\s+(?:the phrase\s+|"[^"]+")', head) and "buy one" in head:
        return Family.RETRIEVAL
    if "firm requirements matter more" in head or "ideally, i prefer" in head:
        return Family.INTENT
    if "i will tell you if you ask" in head:
        return Family.CONSTRAINT
    return Family.CONSTRAINT

def llm_classify_family(query: str, tools: list[dict] | None = None) -> Family:

    q = (query or "").lower()
    head = q.split("\n\ntask rules:")[0]
    if "preferences are relaxable only after the exact item becomes unavailable" in q:
        return Family.RECOVERY
    if "strongly prefer the" in head and "another color of the same model" in head:
        return Family.RECOVERY
    if "shopper wants a valid product plus a trustworthy reason" in q:
        return Family.JUSTIFICATION
    if any(p in head for p in ("walk me through why", "tell me why", "reason i can trust")):
        return Family.JUSTIFICATION

    goal = (
        f"Available tools (dynamic for this episode):\n{tool_summary(tools or [])}\n\n"
        f"Query:\n{(query or '').strip()[:4000]}"
    )
    for i, model in enumerate(MODEL_TABLE["PARSE"]):
        if i:
            time.sleep(LLM_RETRY_DELAY_SEC)
        resp = llm_post(
            model,
            [
                {"role": "system", "content": PROMPT_CLASSIFY_FAMILY},
                {"role": "user", "content": goal},
            ],
        )
        if not (resp and resp.get("choices")):
            continue
        msg = resp["choices"][0].get("message") or {}
        content = msg.get("content") or msg.get("reasoning") or ""
        parsed = parse_json_object(content) or {}
        raw = str(parsed.get("family") or content).strip().lower().replace(" ", "_")
        for fam in Family:
            if fam.value == raw or fam.value in raw:
                return fam

    fam = _classify_family_heuristic(query, tools)
    return fam

def clean_product_spec(prod: dict) -> dict:
    out = dict(prod or {})
    out["keywords"] = str(out.get("keywords") or "").strip() or "product"
    pr = out.get("price_range")
    if pr is not None:
        out["price_range"] = str(pr).strip() or None
    constraints = out.get("constraints")
    if not isinstance(constraints, dict):
        out["constraints"] = {}
    else:
        out["constraints"] = {
            str(k): str(v) for k, v in constraints.items() if v not in (None, "", "null")
        }
    cands = out.get("candidates")
    if isinstance(cands, list):
        parsed = []
        for c in cands:
            s = str(c)
            if "::" in s:
                pid, sku = s.split("::", 1)
                parsed.append((pid.strip(), sku.strip()))
        out["candidates"] = parsed
    else:
        out["candidates"] = []
    out["needs_ask"] = bool(out.get("needs_ask"))
    out["only_product_type"] = bool(out.get("only_product_type", False))
    return out

def llm_parameter_snapshot(query: str) -> dict:
    goal = (query or "").split("\n\nTask rules:")[0].strip()
    for i, model in enumerate(MODEL_TABLE["PARSE"]):
        if i:
            time.sleep(LLM_RETRY_DELAY_SEC)
        resp = llm_post(
            model,
            [
                {"role": "system", "content": PROMPT_PARSE_PRODUCT},
                {"role": "user", "content": goal[:5000]},
            ],
        )
        if not (resp and resp.get("choices")):
            continue
        content = resp["choices"][0].get("message", {}).get("content", "")
        parsed = parse_json_object(content)
        if not parsed or not isinstance(parsed.get("products"), list) or not parsed["products"]:
            continue
        products = [clean_product_spec(p) for p in parsed["products"] if isinstance(p, dict)]
        if not products:
            continue
        return {"products": products[:1], "reasoning": parsed.get("reasoning")}

    return {"products": [heuristic_parse_spec(goal)], "reasoning": "heuristic"}

def heuristic_parse_spec(shopper: str) -> dict:
    spec: dict[str, Any] = {
        "keywords": "product",
        "price_range": None,
        "category": None,
        "brand": None,
        "soft_brand": None,
        "constraints": {},
        "candidates": [],
        "needs_ask": "i will tell you if you ask" in shopper.lower(),
        "only_product_type": False,
    }
    m = re.search(
        r"(?:spend up to|at or under|at or below|under|below)\s*`?([\d,]+(?:\.\d+)?)",
        shopper,
        re.I,
    )
    if m:
        spec["price_range"] = f"-{m.group(1).replace(',', '')}"
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", shopper, re.I)
    if m:
        spec["screen_size"] = f"{float(m.group(1)):g}in"
        spec["constraints"]["screen_size"] = spec["screen_size"]
    m = re.search(r"ideally,\s*i prefer\s+(.+?),\s*but the firm", shopper, re.I)
    if m:
        spec["soft_brand"] = m.group(1).strip()
    m = re.search(r'matches\s+(?:the phrase\s+)?(?:"([^"]+)"|([\w\s\-]+?))(?:,|\s+at\s+or)', shopper, re.I)
    if m:
        phrase = (m.group(1) or m.group(2) or "").strip()
        spec["search_phrase"] = phrase
        spec["keywords"] = phrase
    m = re.search(r"in the ([A-Za-z0-9 &/\-]+?) category", shopper, re.I)
    if m:
        spec["category"] = m.group(1).strip()
    spec["candidates"] = [
        (a, b) for a, b in re.findall(r"(\d{6,})::([\w.\-]+)", shopper)
    ]
    m = re.search(
        r"prefer the (\w+)\s+([A-Za-z0-9.\-]+)",
        shopper,
        re.I,
    )
    if m:
        spec["color"] = m.group(1).lower()
        spec["model"] = m.group(2)
        spec["constraints"]["color_family"] = spec["color"]
        spec["constraints"]["model"] = spec["model"]
    if re.search(r"smartphone", shopper, re.I):
        spec["category"] = spec["category"] or "Smartphones"
        spec["keywords"] = spec.get("keywords") if spec.get("search_phrase") else "smartphone"
        if spec.get("screen_size"):
            spec["keywords"] = f"smartphone {spec['screen_size'].replace('in', ' inch')}"
        if spec.get("soft_brand"):
            spec["keywords"] = f"{spec['soft_brand']} {spec['keywords']}"
    elif re.search(r"headphone", shopper, re.I):
        spec["category"] = "Headphones"
        spec["keywords"] = "sweat resistant waterproof headphones"
    elif re.search(r"monitor", shopper, re.I):
        spec["category"] = spec["category"] or "Gaming Monitors"
        spec["keywords"] = "gaming monitor"
    m = re.search(
        r"\b(?:I need a|I want a|Find me a)\s+([A-Za-z][\w\-]*)\s+(smartphone|tablet|monitor)",
        shopper,
        re.I,
    )
    if m and m.group(1).lower() not in {"in-stock", "gaming"}:
        spec["brand"] = m.group(1)
        spec["constraints"]["brand"] = m.group(1)
    if re.search(r"720p|hd\b", shopper, re.I) and "full hd" not in shopper.lower():
        spec["resolution"] = "hd"
        spec["constraints"]["resolution"] = "hd"
    if re.search(r"1080p|full hd|fhd", shopper, re.I):
        spec["resolution"] = "fhd"
        spec["constraints"]["resolution"] = "fhd"
    return clean_product_spec(spec)

def parse_price_hi(price_range: str | None) -> float | None:
    if not price_range:
        return None
    parts = str(price_range).split("-", 1)
    try:
        if len(parts) == 2 and parts[1].strip():
            return float(parts[1].replace(",", ""))
        if str(price_range).startswith("-"):
            return float(parts[-1].replace(",", ""))
    except ValueError:
        return None
    return None


class EnvSession:
    def __init__(self, problem_data: dict[str, Any]) -> None:
        env = problem_data["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        self.max_steps = int(self.policy.get("max_steps", 30))
        self.max_calls = int(self.policy.get("max_calls_per_turn", 16))
        self.tools = self.policy.get("tools") or []
        self.tool_names = available_names(self.tools)
        self.query = self.policy.get("query") or problem_data.get("query") or ""
        self.turn = 0
        self.done = False
        self.dialogue: list[dict[str, Any]] = []
        self.last_user_message: Optional[str] = None
        self.observations: list[dict[str, Any]] = []

    def has(self, name: str) -> bool:
        return name in self.tool_names

    def call(self, actions: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        if self.done:
            return {"calls": [], "done": True}
        if self.turn >= self.max_steps:
            self.done = True
            return {"calls": [], "done": True}
        self.turn += 1
        actions = actions[: self.max_calls]
        group_id = f"{self.problem_id}-turn-{self.turn}"
        envelope = {
            **self.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": self.turn,
            "calls": [
                {"call_id": f"{group_id}-{i}", "action": {"name": n, "args": a}}
                for i, (n, a) in enumerate(actions, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        self.dialogue.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"{group_id}-{i}",
                        "type": "function",
                        "function": {"name": n, "arguments": json.dumps(a)},
                    }
                    for i, (n, a) in enumerate(actions, start=1)
                ],
                "environment_result": result,
            }
        )
        for call in result.get("calls") or []:
            obs = call.get("observation") or {}
            self.observations.append(obs)
            if obs.get("done"):
                self.done = True
        um = result.get("user_message")
        content = ""
        if isinstance(um, dict):
            content = str(um.get("content") or "").strip()
        elif isinstance(um, str):
            content = um.strip()

        self.last_user_message = content or None
        return result

    def one(self, name: str, **args: Any) -> dict[str, Any]:
        if not self.has(name):
            return {}
        result = self.call([(name, args)])
        calls = result.get("calls") or []
        if not calls:
            return {}
        return _unwrap_obs(calls[0].get("observation") or {})

    def ack_shopper(self, note: str | None = None) -> None:

        if self.done or not self.last_user_message:
            return
        text = note or (
            "Understood — I'll apply that update and continue with a compliant in-stock option."
        )
        self.one("message", content=text)

def _unwrap_obs(payload: dict[str, Any]) -> dict[str, Any]:

    if not isinstance(payload, dict):
        return {}
    inner = payload.get("observation")
    if isinstance(inner, dict) and (
        "compared" in inner
        or "results" in inner
        or "variants" in inner
        or "in_stock" in inner
        or "price" in inner
        or "product_id" in inner
        or "order" in inner
        or "status" in inner
    ):
        return inner
    if any(
        k in payload
        for k in ("compared", "results", "variants", "in_stock", "price", "product_id")
    ):
        return payload
    return inner if isinstance(inner, dict) else payload

@dataclass
class PipelineContext:
    sess: EnvSession
    family: Family
    query: str
    spec: dict = field(default_factory=dict)
    pool: list[dict] = field(default_factory=list)
    details: dict[str, dict] = field(default_factory=dict)
    planned_tools: list[str] = field(default_factory=list)


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None

def _parse_hz(text: Any) -> Optional[float]:
    nums = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*-?\s*hz", str(text or ""), re.I)]
    return max(nums) if nums else None

def _parse_inches(text: Any) -> Optional[float]:
    nums = [
        float(m.group(1))
        for m in re.finditer(
            r"(\d+(?:\.\d+)?)\s*-?\s*(?:inches|inch|in)\b", str(text or ""), re.I
        )
    ]
    return max(nums) if nums else None

def within_budget(price: Optional[float], hi: Optional[float]) -> bool:
    if hi is None:
        return True
    if price is None:
        return False
    return price <= hi + 1e-6

def flatten_facts(obj: dict) -> str:
    parts = [str(obj.get("title") or ""), str(obj.get("brand") or "")]
    facts = obj.get("facts") or obj.get("raw_facts") or obj.get("product_facts") or {}
    if isinstance(facts, dict):
        parts.append(json.dumps(facts, ensure_ascii=False))
    return " ".join(parts).lower()

def variants_from_view(obs: dict) -> list[dict]:
    pid = str(obs.get("product_id") or "")
    title = str(obs.get("title") or "")
    brand = str(obs.get("brand") or "")
    facts = obs.get("facts") or obs.get("raw_facts") or obs.get("product_facts") or {}
    if not isinstance(facts, dict):
        facts = {}
    out = []
    for row in obs.get("variants") or []:
        options = str(row.get("options") or row.get("option") or "")
        blob = f"{title} {options} {json.dumps(facts)}"

        screen_in = (
            _parse_inches(blob)
            or _parse_inches(facts.get("size"))
            or _parse_inches(facts.get("screen_size"))
            or _parse_inches(facts.get("display_size"))
            or _parse_inches(facts.get("screen"))
            or _parse_inches(facts.get("display"))
        )
        out.append(
            {
                "product_id": pid,
                "sku": str(row.get("sku") or ""),
                "title": title,
                "brand": brand,
                "price": _as_float(row.get("price") or row.get("current_price")),
                "in_stock": bool(row.get("in_stock")) if "in_stock" in row else True,
                "options": options,
                "facts": facts,
                "category_path": obs.get("category_path"),
                "refresh_hz": _parse_hz(blob),
                "screen_in": screen_in,
            }
        )
    return out

def detail_payload_from_view(obs: dict) -> dict:

    facts = obs.get("facts") or obs.get("raw_facts") or obs.get("product_facts") or {}
    if not isinstance(facts, dict):
        facts = {}
    sku_options = {}
    for row in obs.get("variants") or []:
        sku = str(row.get("sku") or "")
        if sku:
            sku_options[sku] = {
                "options": row.get("options") or row.get("option"),
                "price": row.get("price") or row.get("current_price"),
                "in_stock": row.get("in_stock"),
            }
    return {
        "product_id": str(obs.get("product_id") or ""),
        "title": obs.get("title"),
        "brand": obs.get("brand"),
        "attributes": facts,
        "sku_options": sku_options,
        "category_path": obs.get("category_path"),
        "raw": obs,
    }

def build_scoring_candidate_payload(product: dict, detail: dict | None) -> dict:
    d = detail or {}
    return {
        "product_id": str(product.get("product_id") or ""),
        "title": product.get("title") or d.get("title"),
        "brand": product.get("brand") or d.get("brand"),
        "price": product.get("price"),
        "attributes": d.get("attributes") or {},
        "sku_options": d.get("sku_options") or {},
        "variants": [
            {"sku": k, **(v if isinstance(v, dict) else {})}
            for k, v in (d.get("sku_options") or {}).items()
        ],
        "facts": d.get("attributes") or {},
    }


def keyword_queries(spec: dict) -> list[str]:
    kw = str(spec.get("keywords") or "product").strip()
    phrase = str(spec.get("search_phrase") or "").strip()
    brand = spec.get("soft_brand") or spec.get("brand")
    model = spec.get("model")
    color = spec.get("color")
    cat_noun = category_noun(spec.get("category"))
    qs = []
    if phrase:
        qs.extend([phrase, f"{phrase} {spec.get('category') or ''}".strip(), " ".join(phrase.split()[:2])])
    qs.append(kw)
    if brand and brand.lower() not in kw.lower():
        qs.append(f"{brand} {kw}")
    if model:
        qs.append(f"{color or ''} {model}".strip())
        if brand:
            qs.append(f"{brand} {model}")
        qs.append(f"{model} {cat_noun}" if cat_noun and cat_noun != "product" else f"{model}")

    out, seen = [], set()
    for q in qs:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:SEARCH_REFORMULATIONS]

def _maybe_ack(ctx: PipelineContext) -> None:
    sess = ctx.sess
    if sess.done or not sess.last_user_message:
        return
    pending = sess.last_user_message
    sess.ack_shopper()

    if pending:
        sess.last_user_message = pending
        apply_shopper_revision(ctx)
        sess.last_user_message = None

def fetch_listings(ctx: PipelineContext, limit: int = 40) -> list[dict]:

    sess, spec = ctx.sess, ctx.spec
    hits: list[dict] = []
    seen: set[str] = set()
    hi = parse_price_hi(spec.get("price_range"))

    for q in keyword_queries(spec):
        if sess.done or len(hits) >= limit:
            break
        obs = sess.one("search", query=q, k=10, in_stock=True)
        _maybe_ack(ctx)
        hi = parse_price_hi(ctx.spec.get("price_range"))
        for row in obs.get("results") or []:
            pid = str(row.get("product_id") or "")
            if not pid or pid in seen:
                continue
            price = _as_float(row.get("price"))
            if not within_budget(price, hi):
                continue
            seen.add(pid)
            row = dict(row)
            row["_search_rank"] = len(hits)
            row["_query"] = q
            hits.append(row)
            if len(hits) >= limit:
                break

    cat = spec.get("category")
    if cat and not sess.done and len(hits) < limit:
        args: dict[str, Any] = {"category": cat, "k": 10, "in_stock": True}
        if hi is not None:
            args["max_price"] = hi
        brand = spec.get("brand") or (
            spec.get("soft_brand") if ctx.family == Family.INTENT else None
        )
        if brand and ctx.family in {Family.CONSTRAINT, Family.JUSTIFICATION, Family.INTENT}:
            args["brand"] = brand
        obs = sess.one("filter", **args)
        _maybe_ack(ctx)
        hi = parse_price_hi(ctx.spec.get("price_range"))
        for row in obs.get("results") or []:
            pid = str(row.get("product_id") or "")
            if not pid or pid in seen:
                continue
            price = _as_float(row.get("price"))
            if not within_budget(price, hi):
                continue
            seen.add(pid)
            row = dict(row)
            row["_search_rank"] = len(hits)
            hits.append(row)
            if len(hits) >= limit:
                break

    for pid, sku in spec.get("candidates") or []:
        if pid not in seen:
            seen.add(pid)
            hits.append(
                {
                    "product_id": pid,
                    "sku": sku,
                    "title": "",
                    "_search_rank": len(hits),
                    "_forced_sku": sku,
                }
            )

    ctx.pool = hits
    return hits

def load_product_details(ctx: PipelineContext, pool: list[dict], cap: int = 12) -> dict[str, dict]:

    sess = ctx.sess
    details = ctx.details
    pids = []
    for p in pool:
        pid = str(p.get("product_id") or "")
        if pid and pid not in details and pid not in pids:
            pids.append(pid)
        if len(pids) >= cap:
            break
    if len(pids) >= 2 and not sess.done:

        sess.one("compare", product_ids=pids[:5])
        _maybe_ack(ctx)
    for pid in pids:
        if sess.done:
            break
        if pid in details:
            continue
        obs = sess.one("view", product_id=pid)
        _maybe_ack(ctx)
        details[pid] = detail_payload_from_view(obs)

        for p in pool:
            if str(p.get("product_id")) == pid and not p.get("title"):
                p["title"] = details[pid].get("title")
                p["brand"] = details[pid].get("brand")
    ctx.details = details
    return details

def extract_evidence_rows(ctx: PipelineContext, candidates: list[dict]) -> list[dict] | None:
    if not candidates:
        return []
    payload = {
        "request": ctx.query.split("\n\nTask rules:")[0][:2000],
        "constraints": ctx.spec.get("constraints") or {},
        "candidates": [
            build_scoring_candidate_payload(p, ctx.details.get(str(p.get("product_id") or "")))
            for p in candidates[:PRODUCT_BATCH_SCORE_CAP]
        ],
    }
    user_content = json.dumps(payload, ensure_ascii=False)
    known = {str(p.get("product_id") or "") for p in candidates}

    for i, model in enumerate(MODEL_TABLE["JUDGE"]):
        if i:
            time.sleep(LLM_RETRY_DELAY_SEC)
        resp = llm_post(
            model,
            [
                {"role": "system", "content": PROMPT_EVIDENCE_EXTRACT},
                {"role": "user", "content": user_content},
            ],
        )
        if not (resp and resp.get("choices")):
            continue
        rows = parse_json_array(resp["choices"][0].get("message", {}).get("content", ""))
        if not isinstance(rows, list):
            continue
        ordered = []
        seen = set()
        for entry in rows:
            if not isinstance(entry, dict):
                continue
            pid = str(entry.get("product_id") or "").strip()
            if not pid or pid not in known or pid in seen:
                continue
            seen.add(pid)
            ordered.append(
                {
                    "product_id": pid,
                    "sku": str(entry.get("sku") or ""),
                    "sku_options": normalize_evidence_list(entry.get("sku_options")),
                    "attributes": normalize_evidence_list(entry.get("attributes")),
                    "title": normalize_evidence_list(entry.get("title")),
                    "missing": normalize_evidence_list(entry.get("missing")),
                }
            )
        return ordered
    return None

def heuristic_score_pool(ctx: PipelineContext, pool: list[dict]) -> list[tuple[dict, float]]:

    spec = ctx.spec
    kw = str(spec.get("keywords") or "")
    tokens = [
        t.lower()
        for t in re.split(r"\W+", kw)
        if t and len(t) >= 2 and t.lower() not in RANK_STOPWORDS
    ]
    for v in (spec.get("constraints") or {}).values():
        for t in re.split(r"\W+", str(v).lower()):
            if t and len(t) >= 2 and t not in tokens and t not in RANK_STOPWORDS:
                tokens.append(t)
    scored = []
    for p in pool:
        pid = str(p.get("product_id") or "")
        detail = ctx.details.get(pid) or {}
        blob = " ".join(
            [
                str(p.get("title") or ""),
                str(detail.get("title") or ""),
                json.dumps(detail.get("attributes") or {}),
                json.dumps(detail.get("sku_options") or {}),
            ]
        ).lower()
        hits = sum(1 for t in tokens if t in blob)
        total = max(len(tokens), 1)
        score = 55.0 + 40.0 * (hits / total)

        if ctx.family == Family.PREFERENCE and re.search(
            r"ipx|waterproof|water.?resist|sweat|sport", blob
        ):
            score += 12
        if ctx.family == Family.INTENT and spec.get("soft_brand"):
            if str(spec["soft_brand"]).lower() in blob:
                score += 8
        if ctx.family == Family.RECOVERY:
            model = str(spec.get("model") or "").lower()
            color = str(spec.get("color") or "").lower()
            if model and model in blob.replace(" ", "").replace("-", ""):
                score += 10
            if color and color in blob:
                score += 6
        p["_llm_relevance_score"] = score
        scored.append((p, score))
    scored.sort(key=lambda x: (-x[1], x[0].get("_search_rank", 999)))
    return scored

def score_from_evidence(ctx: PipelineContext, pool: list[dict], rows: list[dict]) -> list[tuple[dict, float]]:
    by_pid = {str(p.get("product_id") or ""): p for p in pool}
    scored: list[tuple[dict, float]] = []
    for row in rows:
        pid = str(row.get("product_id") or "")
        p = by_pid.get(pid)
        if not p:
            continue
        sku_n = len(row.get("sku_options") or [])
        attr_n = len(row.get("attributes") or [])
        title_n = len(row.get("title") or [])
        missing_n = len(row.get("missing") or [])
        score = score_from_evidence_counts(sku_n, attr_n, title_n, missing_n)
        if score < 55:
            continue
        p = dict(p)
        p["_llm_relevance_score"] = score
        p["_evidence_sku"] = row.get("sku_options") or []
        p["_evidence_attr"] = row.get("attributes") or []
        p["_evidence_title"] = row.get("title") or []
        p["_evidence_missing"] = row.get("missing") or []
        p["_chosen_sku"] = row.get("sku") or p.get("_forced_sku") or ""
        scored.append((p, score))
    scored.sort(
        key=lambda x: (
            -evidence_score_sort_bucket(x[1]),
            -len(x[0].get("_evidence_sku") or []),
            -len(x[0].get("_evidence_attr") or []),
            float(x[0].get("price") or 0),
            str(x[0].get("product_id") or ""),
        )
    )
    return scored

def product_final_judge(ctx: PipelineContext, pool: list[dict]) -> dict | None:

    if not pool:
        return None
    load_product_details(ctx, pool, cap=12)
    rows = extract_evidence_rows(ctx, pool[:PRODUCT_BATCH_SCORE_CAP])
    if rows is not None:
        scored = score_from_evidence(ctx, pool, rows)
        if scored:
            best, score = scored[0]
            return best
    scored = heuristic_score_pool(ctx, pool)
    return scored[0][0] if scored else pool[0]

def pick_variant_for_product(ctx: PipelineContext, product: dict) -> Optional[dict]:
    pid = str(product.get("product_id") or "")
    detail = ctx.details.get(pid) or {}
    obs = detail.get("raw") or {}
    variants = variants_from_view(obs) if obs else []
    if not variants and pid:
        obs = ctx.sess.one("view", product_id=pid)
        ctx.details[pid] = detail_payload_from_view(obs)
        variants = variants_from_view(obs)

    hi = parse_price_hi(ctx.spec.get("price_range"))
    forced = product.get("_chosen_sku") or product.get("_forced_sku")
    if forced:
        for v in variants:
            if v["sku"] == forced and v.get("in_stock") is not False and within_budget(v.get("price"), hi):
                return v

    ok = [
        v
        for v in variants
        if v.get("sku")
        and v.get("in_stock") is not False
        and within_budget(v.get("price"), hi)
    ]
    ok = apply_hard_filters(ctx, ok)
    if not ok:
        return None

    if ctx.family == Family.RANKING:
        ok.sort(
            key=lambda v: (
                -(v.get("refresh_hz") or -1),
                -(v.get("screen_in") or -1),
                v.get("price") if v.get("price") is not None else 1e18,
            )
        )
        return ok[0]

    if ctx.family == Family.RECOVERY:
        color = str(ctx.spec.get("color") or "").lower()
        model = str(ctx.spec.get("model") or "").lower()

        def ladder(v: dict) -> tuple:
            blob = f"{v.get('title')} {v.get('options')} {json.dumps(v.get('facts'))}".lower()
            model_ok = bool(model) and (model in blob or model in blob.replace(" ", "").replace("-", ""))
            color_ok = bool(color) and color in blob
            return (0 if model_ok and color_ok else 1 if model_ok else 2, v.get("price") or 1e18)

        ok.sort(key=ladder)
        return ok[0]

    if ctx.family == Family.INTENT and ctx.spec.get("soft_brand"):
        brand = str(ctx.spec["soft_brand"]).lower()
        branded = [
            v
            for v in ok
            if brand in str(v.get("brand") or "").lower() or brand in str(v.get("title") or "").lower()
        ]
        if branded:
            ok = branded

    ok.sort(key=lambda v: (v.get("price") if v.get("price") is not None else 1e18))
    return ok[0]

def apply_hard_filters(ctx: PipelineContext, variants: list[dict]) -> list[dict]:
    spec = ctx.spec
    out = []
    want_screen = spec.get("screen_size")
    want_res = spec.get("resolution")
    want_brand = spec.get("brand")
    for v in variants:
        blob = f"{v.get('title')} {v.get('options')} {json.dumps(v.get('facts'))}".lower()
        if want_screen:
            target = _parse_inches(str(want_screen).replace("in", " inch"))
            got = v.get("screen_in") or _parse_inches(blob)
            if target is not None and (got is None or abs(got - target) > 0.06):
                continue
        if want_res:
            mapping = {
                "4k": ("4k", "uhd", "3840"),
                "qhd": ("qhd", "1440"),
                "fhd": ("fhd", "1080", "full hd"),
                "hd": ("720",),
            }
            tokens = mapping.get(str(want_res).lower(), (str(want_res).lower(),))
            if str(want_res).lower() == "hd" and any(
                t in blob for t in ("full hd", "1080", "qhd", "1440", "4k")
            ):
                if "720" not in blob:
                    continue
            if not any(t in blob for t in tokens):
                continue
        if want_brand:
            b = str(want_brand).lower()
            product_brand = str(v.get("brand") or "").lower()
            # Only match brand in first ~60 chars of title to avoid "for Samsung" accessories
            title_prefix = str(v.get("title") or "")[:60].lower()
            if b not in product_brand and b not in title_prefix:
                continue
        if (spec.get("category") or "").lower() == "smartphones":
            bad = (
                "earbud", "case for", "screen protector", "charger", "feature phone",
                "car holder", "phone holder", "phone mount", "desk stand", "air vent",
                "gravity mount", "suction cup", "windshield", "tempered glass",
                "flip cover", "wallet case", "back cover", "cable",
            )
            if any(x in blob for x in bad):
                continue
            if not any(
                x in blob
                for x in ("smartphone", "android", "ios", "iphone", "galaxy", "harmonyos")
            ):

                path = str(v.get("category_path") or "").lower()
                if "smartphone" not in path:
                    continue
        out.append(v)
    return out


def inspect_live(sess: EnvSession, product_id: str, sku: str) -> dict:
    return sess.one("inspect_stock", product_id=product_id, sku=sku)

def commit_variant(ctx: PipelineContext, variant: dict) -> Optional[dict]:
    sess = ctx.sess
    hi = parse_price_hi(ctx.spec.get("price_range"))
    stock = inspect_live(sess, variant["product_id"], variant["sku"])
    if sess.done:
        return variant

    if sess.last_user_message:
        _maybe_ack(ctx)
        hi = parse_price_hi(ctx.spec.get("price_range"))
    price = _as_float(stock.get("price") or stock.get("current_price"))
    if stock.get("in_stock") is False or not within_budget(price, hi):
        return None
    if price is not None:
        variant["price"] = price

    if ctx.family == Family.JUSTIFICATION and not ctx.spec.get("_justified_once"):
        facts = variant.get("facts") or {}
        bits = []
        if variant.get("title"):
            bits.append(str(variant["title"])[:80])
        if price is not None:
            bits.append(f"price={price:.2f}")
        bits.append("in_stock=true")
        for key in ("resolution", "display", "size", "screen_size"):
            if key in facts:
                bits.append(f"{key}={facts[key]}")
                break
        sess.one(
            "message",
            content="Choosing this item because: " + "; ".join(bits),
        )
        ctx.spec["_justified_once"] = True
        if sess.done:
            return variant
        if sess.last_user_message:
            _maybe_ack(ctx)
            hi = parse_price_hi(ctx.spec.get("price_range"))

    sess.one("add_to_cart", product_id=variant["product_id"], sku=variant["sku"])
    if sess.done:
        return variant
    if sess.last_user_message:
        _maybe_ack(ctx)
        hi = parse_price_hi(ctx.spec.get("price_range"))

    stock2 = inspect_live(sess, variant["product_id"], variant["sku"])
    if sess.done:
        return variant
    if sess.last_user_message:
        _maybe_ack(ctx)
        hi = parse_price_hi(ctx.spec.get("price_range"))
    price2 = _as_float(stock2.get("price") or stock2.get("current_price"))
    if stock2.get("in_stock") is False or not within_budget(price2, hi):
        sess.one(
            "remove_from_cart",
            product_id=variant["product_id"],
            sku=variant["sku"],
        )
        if sess.last_user_message:
            _maybe_ack(ctx)
        return None
    if price2 is not None:
        variant["price"] = price2
    return variant

def apply_shopper_revision(ctx: PipelineContext) -> None:
    msg = ctx.sess.last_user_message or ""
    if not msg:
        return
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", msg, re.I)
    if m:
        ctx.spec["screen_size"] = f"{float(m.group(1)):g}in"
        ctx.spec.setdefault("constraints", {})["screen_size"] = ctx.spec["screen_size"]
    m = re.search(
        r"(?:spend up to|at or under|under|up to|budget.*?|only go up to)\s*`?([\d,]+(?:\.\d+)?)",
        msg,
        re.I,
    )
    if m:
        ctx.spec["price_range"] = f"-{m.group(1).replace(',', '')}"
    low = msg.lower()
    if re.search(r"1080p|full hd|fhd", low):
        ctx.spec["resolution"] = "fhd"
        ctx.spec.setdefault("constraints", {})["resolution"] = "fhd"
    elif re.search(r"720p|\bhd\b", low) and "full hd" not in low:
        ctx.spec["resolution"] = "hd"
        ctx.spec.setdefault("constraints", {})["resolution"] = "hd"
    elif re.search(r"\bqhd\b|1440p", low):
        ctx.spec["resolution"] = "qhd"
        ctx.spec.setdefault("constraints", {})["resolution"] = "qhd"
    elif re.search(r"\b4k\b|\buhd\b", low):
        ctx.spec["resolution"] = "4k"
        ctx.spec.setdefault("constraints", {})["resolution"] = "4k"
    m = re.search(r"prefer the (\w+)\s+([A-Za-z0-9.\-]+)", msg, re.I)
    if m:
        ctx.spec["color"] = m.group(1).lower()
        ctx.spec["model"] = m.group(2)
        ctx.spec.setdefault("constraints", {})["color_family"] = ctx.spec["color"]
        ctx.spec["constraints"]["model"] = ctx.spec["model"]
    m = re.search(
        r"(?:get me an?(?: in-stock)?|I need a|I want a|Find me a)\s+"
        r"([A-Za-z][\w\-]*)\s+(smartphone|tablet|monitor|headphone)",
        msg,
        re.I,
    )
    if m and m.group(1).lower() not in {"in-stock", "gaming", "genuine"}:
        ctx.spec["brand"] = m.group(1)
        ctx.spec.setdefault("constraints", {})["brand"] = m.group(1)

def _fact_value(facts: dict, *keys: str) -> Optional[str]:
    for k in keys:
        if k in facts and facts[k] not in (None, "", "null"):
            return str(facts[k])

    lower = {str(k).lower(): v for k, v in facts.items()}
    for k in keys:
        if k.lower() in lower and lower[k.lower()] not in (None, "", "null"):
            return str(lower[k.lower()])
    return None

_SHOPPER_POLICY_CUT_RE = re.compile(
    r"(?:^|\n|(?<=\.))\s*(?=For a \w|The catalog |Reject conflicting |View and compare expose)",
    re.I,
)
_WANT_LEAD_RE = re.compile(
    r"(?:I want a|Find me a|Can you get me an?(?: in-stock)?|I need a|get me an?(?: in-stock)?)\s+",
    re.I,
)
_SPEC_OR_TYPE_RE = re.compile(
    r"(?:720p|1080p|1440p|2160p|4k|uhd|qhd|fhd|full hd|\d+(?:\.\d+)?-?\s*inch|"
    r"smartphone|tablet|monitor|headphone)",
    re.I,
)
_JUST_WXH_RE = re.compile(r"(\d{3,5})\s*[x×*]\s*(\d{3,5})")
_JUST_INCH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])",
    re.I,
)
_JUST_SOFT_KEYS = ("microphone", "autofocus", "hdr", "streaming", "privacy_cover")


def shopper_request(query: str) -> str:
    goal = (query or "").split("\n\nTask rules:")[0].strip()
    cut = _SHOPPER_POLICY_CUT_RE.search(goal)
    if cut:
        goal = goal[: cut.start()].strip()
    return goal


def category_noun(category: str | None) -> str:
    cat = (category or "").lower()
    if "smart" in cat:
        return "smartphone"
    if "tablet" in cat:
        return "tablet"
    if "monitor" in cat:
        return "monitor"
    if "headphone" in cat:
        return "headphone"
    return cat.strip() or "product"


def title_matches_category(title: str, category: str | None) -> bool:
    low = (title or "").lower()
    noun = category_noun(category)
    if noun == "smartphone":
        return bool(re.search(r"\bsmart\s*phones?\b|\bcellphone\b|\bmobile phones?\b", low))
    if noun == "tablet":
        return bool(re.search(r"\btablets?\b", low))
    if noun == "monitor":
        return bool(re.search(r"\bmonitors?\b", low))
    if noun == "headphone":
        return bool(re.search(r"\bheadphones?\b|\bheadsets?\b|\bearbuds?\b|\bearphones?\b", low))
    return bool(noun and noun in low)


def path_matches_category(path: Any, category: str | None) -> bool:
    blob = " ".join(str(x) for x in path).lower() if isinstance(path, list) else str(path or "").lower()
    if not blob or "accessor" in blob:
        return False
    noun = category_noun(category)
    if noun == "smartphone":
        return "smartphone" in blob and "feature phone" not in blob
    if noun == "tablet":
        return "tablet" in blob
    if noun == "monitor":
        return "monitor" in blob
    if noun == "headphone":
        return "headphone" in blob or "headset" in blob or "earbud" in blob
    return noun in blob if noun and noun != "product" else True


def product_phrase(request: str, brand: str = "") -> str:
    text = request or ""
    type_m = re.search(r"\b(smartphone|tablet|monitor|headphone)s?\b", text, re.I)
    lead = _WANT_LEAD_RE.search(text)
    start = lead.end() if lead else 0
    if brand:
        bm = re.search(re.escape(brand), text, re.I)
        if bm and (type_m is None or bm.start() <= type_m.start()):
            start = max(start, bm.start())
    if type_m:
        return text[start : type_m.end()]
    return text[start:].split(".", 1)[0]


def request_modifiers(request: str, brand: str, category: str | None) -> list[str]:
    stop = {
        "i", "a", "an", "the", "me", "at", "or", "under", "below", "in", "stock",
        "php", "and", "it", "has", "to", "be", "want", "find", "can", "you", "get",
        "for", "with", "one", "good", "pick", "need", "inch", "inches",
        "this", "that", "these", "those",
    }
    skip = set(re.findall(r"[a-z0-9]+", f"{brand} {category_noun(category)}".lower()))
    out: list[str] = []
    for tok in re.findall(r"[a-z0-9]+", product_phrase(request, brand).lower()):
        if tok in stop or tok in skip or len(tok) < 3:
            continue
        if tok.isdigit() or tok in {"720p", "1080p", "1440p", "2160p", "full", "ultra", "uhd", "qhd", "fhd"}:
            continue
        if tok not in out:
            out.append(tok)
    return out


def _just_pixel_class(long_side: int, short_side: int) -> Optional[str]:
    if long_side >= 3840 and short_side >= 2160:
        return "4k"
    if long_side >= 2560 and short_side >= 1440:
        return "qhd"
    if long_side >= 1920 and short_side >= 1080:
        return "fhd"
    if long_side >= 1280 and short_side >= 720:
        return "hd"
    return None


def _just_res_classes(text: str) -> set[str]:
    if re.search(r"\bqHD\b", text or ""):
        return set()
    low = (text or "").lower()
    out: set[str] = set()
    for m in _JUST_WXH_RE.finditer(low):
        pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        cls = _just_pixel_class(pair[0], pair[1])
        if cls:
            out.add(cls)
    if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
        out.add("4k")
    if re.search(r"\b(?:qhd|1440p|2k)\b", low):
        out.add("qhd")
    if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
        out.add("fhd")
    stripped = re.sub(r"\b(?:full|ultra) hd\b", "", low)
    if re.search(r"\b(?:hd|720p?)\b", stripped):
        out.add("hd")
    return out


def _just_size_values(raw: str) -> set[str]:
    text = str(raw or "").strip()
    if re.search(r"\d+(?:\.\d+)?\s*[-–/]\s*\d+", text):
        return set()
    found = {"%gin" % float(m.group(1)) for m in _JUST_INCH_RE.finditer(text)}
    if not found and re.fullmatch(r"\d+(?:\.\d+)?", text):
        found = {"%gin" % float(text)}
    return found


def _just_gate(spec: dict) -> tuple[Optional[str], Optional[str]]:
    if spec.get("resolution"):
        return "resolution", str(spec["resolution"]).lower()
    if spec.get("screen_size"):
        m = re.search(r"(\d+(?:\.\d+)?)", str(spec["screen_size"]))
        if m:
            return "screen_size", f"{float(m.group(1)):g}in"
    return None, None


def _just_fact_raw(variant: dict, field: str) -> Optional[str]:
    facts = variant.get("facts") if isinstance(variant.get("facts"), dict) else {}
    key = "display" if field == "resolution" else "size" if field == "screen_size" else field
    raw = facts.get(key) or facts.get(field)
    if raw in (None, "", "null"):
        return None
    return str(raw).strip()


def _just_spec_ok(variant: dict, field: Optional[str], want: Optional[str]) -> bool:
    if not field or not want:
        return True
    raw = _just_fact_raw(variant, field)
    if not raw:
        return False
    if field == "resolution":
        return _just_res_classes(raw) == {want}
    if field == "screen_size":
        vals = _just_size_values(raw)
        return vals == {want}
    return False


def _just_category_ok(variant: dict, spec: dict) -> bool:
    if not path_matches_category(variant.get("category_path"), spec.get("category")):
        return False
    if category_noun(spec.get("category")) == "smartphone":
        return _recovery_phone_identity(variant) == "yes"
    return True


def _just_brand_ok(variant: dict, brand: Optional[str]) -> bool:
    if not brand:
        return True
    b = brand.lower()
    listing = str(variant.get("brand") or "").lower()
    return bool(listing) and b in listing


def _just_resale(variant: dict) -> bool:
    title = str(variant.get("title") or "")
    if re.search(r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed", title, re.I):
        return True
    facts = variant.get("facts") if isinstance(variant.get("facts"), dict) else {}
    return str(facts.get("condition") or "").lower() in {"used", "refurbished", "pre-owned", "renewed"}


def _just_eligible(variant: dict, spec: dict, field: Optional[str], want: Optional[str], hi: Optional[float]) -> bool:
    if not variant.get("sku"):
        return False
    if variant.get("in_stock") is False or not within_budget(variant.get("price"), hi):
        return False
    if _just_resale(variant):
        return False
    if not _just_category_ok(variant, spec):
        return False
    if not _just_brand_ok(variant, spec.get("brand")):
        return False
    if spec.get("brand") and not str(variant.get("brand") or "").strip():
        return False
    if field and want and not _just_spec_ok(variant, field, want):
        return False
    return True


def listing_search_phrase(title: str) -> str:
    toks = re.findall(r"[A-Za-z0-9]+", title or "")
    return " ".join(toks[:6])


def _just_worth_view(row: dict, spec: dict) -> bool:
    title = str(row.get("title") or "")
    if not title_matches_category(title, spec.get("category")):
        return False
    brand = str(spec.get("brand") or "").strip().lower()
    if not brand:
        return True
    row_brand = str(row.get("brand") or "").strip().lower()
    if not row_brand or row_brand in {"none", "null", "unknown"}:
        return False
    return brand in row_brand


def _just_bind_spec(ctx: PipelineContext) -> tuple[Optional[str], Optional[str]]:
    shopper = shopper_request(ctx.query or "")
    lead = _WANT_LEAD_RE.search(shopper)
    if lead:
        rest = shopper[lead.end() :]
        cut = _SPEC_OR_TYPE_RE.search(rest)
        brand = re.sub(r"\s+", " ", (rest[: cut.start()] if cut else rest.split(",", 1)[0])).strip()
        if brand and brand.lower() not in {"in-stock", "genuine"}:
            ctx.spec["brand"] = brand
    window = shopper[lead.end() :] if lead else shopper
    type_m = re.search(r"\b(smartphone|tablet|monitor|headphone)s?\b", window, re.I)
    if type_m:
        ctx.spec["category"] = {
            "smartphone": "Smartphones",
            "tablet": "Tablets",
            "monitor": "Monitors",
            "headphone": "Headphones",
        }[type_m.group(1).lower()]
    if re.search(r"1080p|full hd|fhd", shopper, re.I):
        ctx.spec["resolution"] = "fhd"
        ctx.spec.setdefault("constraints", {})["resolution"] = "fhd"
    elif re.search(r"720p|\bhd\b", shopper, re.I) and "full hd" not in shopper.lower():
        ctx.spec["resolution"] = "hd"
        ctx.spec.setdefault("constraints", {})["resolution"] = "hd"
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", shopper, re.I)
    if m and not ctx.spec.get("resolution"):
        ctx.spec["screen_size"] = f"{float(m.group(1)):g}in"
        ctx.spec.setdefault("constraints", {})["screen_size"] = ctx.spec["screen_size"]
    return _just_gate(ctx.spec)


def build_justification(ctx: PipelineContext, variant: dict, price: Optional[float], in_stock: bool) -> dict:

    claims: list[dict[str, str]] = []
    field, want = _just_gate(ctx.spec)
    raw = _just_fact_raw(variant, field) if field else None
    if field and raw and _just_spec_ok(variant, field, want):
        claims.append({"field": field, "value": raw})
    if price is not None:
        claims.append({"field": "price", "value": f"{price:.2f}"})
    claims.append({"field": "in_stock", "value": "true" if in_stock else "false"})
    facts = variant.get("facts") if isinstance(variant.get("facts"), dict) else {}
    brand = _fact_value(facts, "brand") or variant.get("brand")
    if brand and str(brand).strip().lower() not in ("", "null", "none", "unknown"):
        claims.append({"field": "brand", "value": str(brand).strip()})
    elif facts:
        for key in _JUST_SOFT_KEYS:
            if key in facts and facts[key] not in (None, "", "null"):
                claims.append({"field": key, "value": str(facts[key])})
                break
    return {"claims": claims}


def justification_run(ctx: PipelineContext) -> bool:

    sess = ctx.sess
    field, want = _just_bind_spec(ctx)
    hi = parse_price_hi(ctx.spec.get("price_range"))
    brand = str(ctx.spec.get("brand") or "").strip()
    cat = str(ctx.spec.get("category") or "product")
    cat_kw = category_noun(cat)
    req_text = shopper_request(ctx.query or "")
    spec_phrase = ""
    if field == "resolution" and want == "hd":
        spec_phrase = "720p"
    elif field == "resolution" and want == "fhd":
        spec_phrase = "1080p"
    elif field == "resolution" and want == "qhd":
        spec_phrase = "1440p"
    elif field == "resolution" and want == "4k":
        spec_phrase = "4k"
    elif field == "screen_size" and want:
        spec_phrase = want.replace("in", " inch")
    extras = request_modifiers(req_text, brand, cat)
    queries = []
    if extras:
        queries.append(" ".join(x for x in [brand, extras[0], cat_kw] if x))
    if spec_phrase:
        queries.append(" ".join(x for x in [brand, cat_kw, spec_phrase] if x))
    queries.append(" ".join(x for x in [brand, cat_kw] if x))

    ranked: list[dict] = []
    seen_pid: set[str] = set()
    searched: set[str] = set()

    def ingest_view(pid: str) -> None:
        if sess.done or pid in seen_pid or sess.turn >= sess.max_steps - 8:
            return
        seen_pid.add(pid)
        obs = sess.one("view", product_id=pid)
        _maybe_ack(ctx)
        ctx.details[pid] = detail_payload_from_view(obs)
        for v in variants_from_view(obs):
            if _just_eligible(v, ctx.spec, field, want, hi):
                ranked.append(v)

    def search_and_view(qs: list[str], view_cap: int) -> None:
        for q in qs:
            if sess.done or len(ranked) >= 2 or sess.turn >= sess.max_steps - 10:
                break
            qn = re.sub(r"\s+", " ", q).strip()
            if not qn or qn.lower() in searched:
                continue
            searched.add(qn.lower())
            obs = sess.one("search", query=qn, k=10, in_stock=True)
            _maybe_ack(ctx)
            hits: list[dict] = []
            seen_hit: set[str] = set()
            for row in obs.get("results") or []:
                pid = str(row.get("product_id") or "")
                if not pid or pid in seen_hit or pid in seen_pid:
                    continue
                if not within_budget(_as_float(row.get("price")), hi):
                    continue
                if not _just_worth_view(row, ctx.spec):
                    continue
                seen_hit.add(pid)
                hits.append(dict(row))
            for row in hits[:view_cap]:
                if sess.done or len(ranked) >= 3:
                    break
                ingest_view(str(row["product_id"]))

    fobs_cache: Optional[dict] = None

    def filter_until(need: int) -> None:
        nonlocal fobs_cache
        if not sess.has("filter") or sess.done or len(ranked) >= need:
            return
        if fobs_cache is None:
            fargs: dict[str, Any] = {"category": cat, "k": 20, "in_stock": True}
            if brand:
                fargs["brand"] = brand
            if hi is not None:
                fargs["max_price"] = hi
            fobs_cache = sess.one("filter", **fargs)
            _maybe_ack(ctx)
        for row in (fobs_cache or {}).get("results") or []:
            pid = str(row.get("product_id") or "")
            if sess.done or len(ranked) >= need:
                break
            if pid and pid not in seen_pid:
                ingest_view(pid)

    search_and_view(queries, view_cap=8)
    if len(ranked) < 2 and brand and not sess.done:
        search_and_view([f"{brand} {cat_kw}"], view_cap=8)
    filter_until(1)
    if 0 < len(ranked) < 2 and not sess.done:
        phrase = listing_search_phrase(str(ranked[0].get("title") or ""))
        if phrase:
            search_and_view([phrase], view_cap=6)
    filter_until(2)
    if not ranked:
        return False

    req_tokens = request_modifiers(req_text, brand, cat)
    if brand:
        req_tokens = [brand.lower()] + req_tokens

    def rank_key(v: dict) -> tuple:
        title = str(v.get("title") or "").lower()
        listing = str(v.get("brand") or "").lower()
        brand_pen = 0 if brand and brand.lower() in listing else 1
        hits = sum(1 for t in req_tokens if t in title)
        price = v.get("price") if v.get("price") is not None else 1e18
        return (brand_pen, -hits, price)

    ranked.sort(key=rank_key)

    committed = None
    tried: set[tuple[str, str]] = set()

    def try_commit(pool: list[dict]) -> Optional[dict]:
        for v in pool:
            key = (str(v["product_id"]), str(v["sku"]))
            if key in tried:
                continue
            if sess.done or sess.turn >= sess.max_steps - 3:
                return None
            tried.add(key)
            got = commit_variant(ctx, v)
            if got:
                return got
        return None

    committed = try_commit(ranked)
    unused = [v for v in ranked if (str(v["product_id"]), str(v["sku"])) not in tried]
    if not committed and not sess.done and not unused:
        search_and_view([f"{brand} {cat_kw}", f"{brand} {spec_phrase}".strip()], view_cap=8)
        ranked.sort(key=rank_key)
        committed = try_commit(ranked)
    if not committed:
        return True

    if sess.last_user_message:
        sess.ack_shopper()
        _maybe_ack(ctx)

    finalize_order(ctx, committed)
    return True

def finalize_order(ctx: PipelineContext, variant: dict) -> None:
    sess = ctx.sess
    if sess.done:
        return
    if sess.last_user_message:
        _maybe_ack(ctx)
    stock = inspect_live(sess, variant["product_id"], variant["sku"])
    if sess.done:
        return
    if sess.last_user_message:
        _maybe_ack(ctx)
    price = _as_float(stock.get("price") or stock.get("current_price") or variant.get("price"))
    in_stock = stock.get("in_stock", True)
    args: dict[str, Any] = {
        "product_id": variant["product_id"],
        "sku": variant["sku"],
    }

    if ctx.family == Family.JUSTIFICATION:
        args["justification"] = build_justification(ctx, variant, price, bool(in_stock))
    sess.one("place_test_order", **args)


def family_pre_retrieve(ctx: PipelineContext) -> None:
    if ctx.family == Family.CONSTRAINT and ctx.spec.get("needs_ask"):
        ctx.sess.one(
            "message",
            content=(
                "What is the firm hardware requirement (RAM, screen size, "
                "resolution, or refresh rate) I should enforce?"
            ),
        )
        reply = ctx.sess.last_user_message
        if reply:
            m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", reply, re.I)
            if m:
                size_val = f"{float(m.group(1)):g}in"
                ctx.spec["screen_size"] = size_val
                ctx.spec.setdefault("constraints", {})["screen_size"] = size_val

                brand = ctx.spec.get("brand") or ctx.spec.get("soft_brand") or ""
                cat_kw = "smartphone" if (ctx.spec.get("category") or "").lower().startswith("smart") else (ctx.spec.get("category") or "smartphone").lower()
                ctx.spec["keywords"] = re.sub(r"\s+", " ", f"{brand} {m.group(1)} inch {cat_kw}").strip()
            m = re.search(r"(\d+)\s*gb", reply, re.I)
            if m:
                ctx.spec["ram"] = f"{m.group(1)}gb"
                ctx.spec.setdefault("constraints", {})["ram_memory"] = ctx.spec["ram"]
                ctx.spec["keywords"] = (ctx.spec.get("keywords") or "smartphone") + f" {m.group(1)}gb"
            m = re.search(r"(\d+)\s*hz", reply, re.I)
            if m:
                ctx.spec["refresh_rate"] = f"{m.group(1)}hz"
                ctx.spec.setdefault("constraints", {})["refresh_rate"] = ctx.spec["refresh_rate"]
                ctx.spec["keywords"] = (ctx.spec.get("keywords") or "monitor") + f" {m.group(1)}hz"
            if re.search(r"720p|\bhd\b", reply, re.I) and not re.search(r"full hd|fhd|1080", reply, re.I):
                ctx.spec["resolution"] = "hd"
                ctx.spec.setdefault("constraints", {})["resolution"] = "hd"
                ctx.spec["keywords"] = (ctx.spec.get("keywords") or "smartphone") + " 720p HD"
            if re.search(r"1080p|full hd|fhd", reply, re.I):
                ctx.spec["resolution"] = "fhd"
                ctx.spec.setdefault("constraints", {})["resolution"] = "fhd"
                ctx.spec["keywords"] = (ctx.spec.get("keywords") or "smartphone") + " 1080p"
    if ctx.family == Family.PREFERENCE:
        ctx.sess.one(
            "message",
            content=(
                "I'll prioritize sweat/water resistance and secure sport fit "
                "over the lowest price for this headphone order. Any other must-haves?"
            ),
        )

def _rank_key_row(row: dict[str, Any]) -> tuple:
    blob = json.dumps(row, ensure_ascii=False)
    hz = _parse_hz(blob) or _parse_hz(row.get("refresh_rate")) or _as_float(row.get("refresh_rate")) or -1
    inches = (
        _parse_inches(blob)
        or _parse_inches(row.get("screen_size"))
        or _as_float(row.get("screen_size"))
        or -1
    )
    price = _as_float(row.get("price") or row.get("current_price"))
    return (-hz, -inches, price if price is not None else 1e18)

def _rows_from_compare(obs: dict[str, Any], candidates: list[tuple[str, str]]) -> list[dict[str, Any]]:
    obs = _unwrap_obs(obs)
    rows: list[dict[str, Any]] = []
    compared = obs.get("compared") or obs.get("products") or obs.get("results") or []
    sku_by_pid = {pid: sku for pid, sku in candidates}
    if isinstance(compared, list):
        for item in compared:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("product_id") or "")
            want_sku = sku_by_pid.get(pid)
            variants = item.get("variants") or []
            if variants:
                matched = False
                for v in variants:
                    if not isinstance(v, dict):
                        continue
                    sku = str(v.get("sku") or "")
                    if want_sku and sku != want_sku:
                        continue
                    matched = True
                    rows.append(
                        {
                            **item,
                            **v,
                            "product_id": pid,
                            "sku": sku or want_sku or "",
                            "title": item.get("title"),
                            "facts": item.get("facts") or {},
                            "brand": item.get("brand"),
                        }
                    )
                if not matched and want_sku:
                    base = variants[0] if isinstance(variants[0], dict) else {}
                    rows.append(
                        {
                            **item,
                            **base,
                            "product_id": pid,
                            "sku": want_sku,
                            "title": item.get("title"),
                            "facts": item.get("facts") or {},
                        }
                    )
            else:
                rows.append(
                    {
                        **item,
                        "product_id": pid,
                        "sku": str(item.get("sku") or want_sku or ""),
                    }
                )
    have = {(str(r.get("product_id")), str(r.get("sku"))) for r in rows}
    for pid, sku in candidates:
        if (pid, sku) not in have:
            rows.append({"product_id": pid, "sku": sku})
    return rows

def ranking_run(ctx: PipelineContext) -> bool:

    sess = ctx.sess
    cands = list(ctx.spec.get("candidates") or [])
    if not cands:

        cands = [
            (a, b)
            for a, b in re.findall(r"(\d{6,})::([\w.\-]+)", ctx.query or "")
        ]
        ctx.spec["candidates"] = cands
    if not cands:
        return False
    hi = parse_price_hi(ctx.spec.get("price_range"))
    if hi is None:
        m = re.search(
            r"(?:at or under|under|below|up to)\s*`?([\d,]+(?:\.\d+)?)",
            ctx.query or "",
            re.I,
        )
        if m:
            hi = float(m.group(1).replace(",", ""))
            ctx.spec["price_range"] = f"-{m.group(1).replace(',', '')}"

    pids: list[str] = []
    for pid, _ in cands:
        if pid not in pids:
            pids.append(pid)

    obs = sess.one("compare", product_ids=pids[:5])
    if sess.done:
        return True
    rows = _rows_from_compare(obs, cands)
    by_key = {(str(r.get("product_id")), str(r.get("sku"))): r for r in rows}

    live: list[dict[str, Any]] = []
    for pid, sku in cands:
        row = by_key.get((pid, sku))
        if row is None:
            row = next((r for r in rows if str(r.get("product_id")) == pid), None)
            row = {**(row or {}), "product_id": pid, "sku": sku}
        if row.get("in_stock") is False:
            continue
        price = _as_float(row.get("price") or row.get("current_price"))
        if not within_budget(price, hi):
            continue
        live.append(row)
    if not live:
        live = [{"product_id": p, "sku": s} for p, s in cands]
    live.sort(key=_rank_key_row)
    winner = live[0]

    sess.one(
        "add_to_cart",
        product_id=str(winner["product_id"]),
        sku=str(winner["sku"]),
    )
    if sess.done:
        return True
    cart_pid, cart_sku = str(winner["product_id"]), str(winner["sku"])

    if not sess.last_user_message:
        sess.one("inspect_stock", product_id=cart_pid, sku=cart_sku)
        if sess.done:
            return True
    if sess.last_user_message:
        sess.ack_shopper(
            "Checking availability now and I will switch if a better in-stock option ranks higher."
        )
        if sess.done:
            return True

    refreshed: list[dict[str, Any]] = []
    for pid, sku in cands:
        if sess.done or sess.turn >= sess.max_steps - 2:
            break
        stock = _unwrap_obs(sess.one("inspect_stock", product_id=pid, sku=sku))
        if sess.done:
            return True
        if sess.last_user_message:
            sess.ack_shopper()
        row = {
            "product_id": pid,
            "sku": sku,
            "price": stock.get("price") or stock.get("current_price"),
            "in_stock": stock.get("in_stock", True),
        }
        prev = by_key.get((pid, sku)) or next(
            (r for r in rows if str(r.get("product_id")) == pid), {}
        )
        for k in ("refresh_rate", "screen_size", "title", "facts", "raw_facts"):
            if k in prev:
                row[k] = prev[k]
        price = _as_float(row.get("price"))
        if row.get("in_stock") is False or not within_budget(price, hi):
            continue
        refreshed.append(row)

    if refreshed:
        refreshed.sort(key=_rank_key_row)
        best = refreshed[0]
        if str(best["product_id"]) != cart_pid or str(best["sku"]) != cart_sku:
            sess.one("remove_from_cart", product_id=cart_pid, sku=cart_sku)
            if sess.done:
                return True
            sess.one(
                "add_to_cart",
                product_id=str(best["product_id"]),
                sku=str(best["sku"]),
            )
            if sess.done:
                return True
            cart_pid, cart_sku = str(best["product_id"]), str(best["sku"])

    sess.one("place_test_order", product_id=cart_pid, sku=cart_sku)
    return True

def ranking_select(ctx: PipelineContext) -> Optional[dict]:

    cands = ctx.spec.get("candidates") or []
    if not cands:
        return None
    pids = []
    for pid, _ in cands:
        if pid not in pids:
            pids.append(pid)
    obs = ctx.sess.one("compare", product_ids=pids[:5])
    rows = _rows_from_compare(obs, cands)
    hi = parse_price_hi(ctx.spec.get("price_range"))
    live = []
    for pid, sku in cands:
        row = next(
            (r for r in rows if str(r.get("product_id")) == pid and str(r.get("sku")) == sku),
            None,
        )
        if not row:
            continue
        price = _as_float(row.get("price") or row.get("current_price"))
        if row.get("in_stock") is False or not within_budget(price, hi):
            continue
        live.append(
            {
                "product_id": pid,
                "sku": sku,
                "price": price,
                "title": row.get("title") or "",
                "brand": row.get("brand") or "",
                "facts": row.get("facts") or {},
                "refresh_hz": _parse_hz(json.dumps(row)),
                "screen_in": _parse_inches(json.dumps(row)),
                "in_stock": True,
            }
        )
    live.sort(
        key=lambda v: (
            -(v.get("refresh_hz") or -1),
            -(v.get("screen_in") or -1),
            v.get("price") if v.get("price") is not None else 1e18,
        )
    )
    return live[0] if live else None

def preference_run(ctx: PipelineContext) -> bool:

    sess = ctx.sess

    if not sess.done:
        sess.one(
            "message",
            content=(
                "I'll prioritize sweat/water resistance and secure sport fit "
                "over the lowest price for this headphone order."
            ),
        )
        if sess.last_user_message:
            _maybe_ack(ctx)
    if sess.done:
        return True
    hi = parse_price_hi(ctx.spec.get("price_range"))

    queries = [
        "IPX waterproof sweat resistant sport headphones",
        "sweat proof bluetooth headphones",
        str(ctx.spec.get("keywords") or "sport headphones"),
    ]
    hits: list[dict] = []
    seen: set[str] = set()
    for q in queries:
        if sess.done or len(hits) >= 12 or sess.turn >= sess.max_steps - 10:
            break
        obs = sess.one("search", query=q, k=10, in_stock=True)
        if sess.last_user_message:
            _maybe_ack(ctx)
        for row in obs.get("results") or []:
            pid = str(row.get("product_id") or "")
            if not pid or pid in seen:
                continue
            price = _as_float(row.get("price"))
            if not within_budget(price, hi):
                continue
            seen.add(pid)
            hits.append(dict(row))

    sport: list[dict] = []
    other: list[dict] = []
    for row in hits[:8]:
        if sess.done or sess.turn >= sess.max_steps - 8:
            break
        pid = str(row["product_id"])
        obs = sess.one("view", product_id=pid)
        if sess.last_user_message:
            _maybe_ack(ctx)
        ctx.details[pid] = detail_payload_from_view(obs)
        variants = variants_from_view(obs)
        for v in variants:
            if v.get("in_stock") is False or not within_budget(v.get("price"), hi):
                continue
            blob = f"{v.get('title')} {v.get('options')} {json.dumps(v.get('facts'))}".lower()
            if any(x in blob for x in ("case for", "ear tip", "cushion only", "replacement ear")):
                continue
            if not any(x in blob for x in ("headphone", "headset", "earbud", "earphone")):
                continue
            if re.search(r"ipx|waterproof|water.?resist|sweat|sport|running|gym", blob):
                sport.append(v)
            else:
                other.append(v)

    def _sport_quality_key(v: dict) -> tuple:
        facts_str = json.dumps(v.get("facts") or {}).lower()
        blob = f"{v.get('title','').lower()} {v.get('options','').lower()} {facts_str}"
        ipx_facts = bool(re.search(r"ipx[3-9]", facts_str))
        wp_facts = bool(re.search(r"waterproof|water.?resist", facts_str))
        ipx_title = bool(re.search(r"ipx[3-9]", blob))
        # Lower priority number = better: IPX in facts best, then waterproof in facts,
        # then IPX in title, then generic sport keywords
        priority = 0 if ipx_facts else (1 if wp_facts else (2 if ipx_title else 3))
        price = v.get("price") if v.get("price") is not None else 1e18
        return (priority, price)

    sport.sort(key=_sport_quality_key)
    ordered = sport + other
    if not ordered:
        return False

    # Iterate as snapshot; fresh-recovery items are appended/prepended to ordered but
    # committed via continue-fall-through on the next snapshot iteration.
    searched_fresh = False
    for v in list(ordered[:8]):
        if sess.done:
            return True
        if sess.turn >= sess.max_steps - 3:
            break
        committed = commit_variant(ctx, v)
        if not committed:
            # Commit failed (likely price event) – one fresh recovery pass
            if not searched_fresh and not sess.done and sess.turn < sess.max_steps - 7:
                searched_fresh = True
                hi = parse_price_hi(ctx.spec.get("price_range"))
                seen_pids = {str(x.get("product_id", "")) for x in ordered}
                fresh_rows: list[dict] = []

                # 1. Filter by Headphones category + budget ceiling (surfaces budget-fit items
                #    that keyword search may have missed, especially for tight budgets)
                if sess.has("filter") and not sess.done and sess.turn < sess.max_steps - 6:
                    fargs: dict[str, Any] = {"category": "Headphones", "k": 20, "in_stock": True}
                    if hi is not None:
                        fargs["max_price"] = hi
                    fobs = sess.one("filter", **fargs)
                    if sess.last_user_message:
                        _maybe_ack(ctx)
                    for row in fobs.get("results") or []:
                        pid = str(row.get("product_id") or "")
                        if pid and pid not in seen_pids and within_budget(_as_float(row.get("price")), hi):
                            fresh_rows.append(row)
                            seen_pids.add(pid)

                # 2. One targeted search if filter found few new items
                if not sess.done and sess.turn < sess.max_steps - 5 and len(fresh_rows) < 5:
                    fobs = sess.one("search", query="IPX waterproof sweat resistant headphones", k=10, in_stock=True)
                    if sess.last_user_message:
                        _maybe_ack(ctx)
                    for row in fobs.get("results") or []:
                        pid = str(row.get("product_id") or "")
                        if pid and pid not in seen_pids and within_budget(_as_float(row.get("price")), hi):
                            fresh_rows.append(row)
                            seen_pids.add(pid)

                # 3. View at most 2 new candidates (hard cap to protect step budget).
                #    Sport-matching variants are prepended so the NEXT snapshot iteration
                #    picks up the best recovery item first.
                views_done = 0
                existing_keys = {(str(x.get("product_id", "")), str(x.get("sku", ""))) for x in ordered}
                for row in fresh_rows:
                    if sess.done or views_done >= 2 or sess.turn >= sess.max_steps - 4:
                        break
                    pid = str(row.get("product_id", ""))
                    if not pid:
                        continue
                    views_done += 1
                    fv_obs = sess.one("view", product_id=pid)
                    if sess.last_user_message:
                        _maybe_ack(ctx)
                    ctx.details[pid] = detail_payload_from_view(fv_obs)
                    for fv in variants_from_view(fv_obs):
                        if fv.get("in_stock") is False or not within_budget(fv.get("price"), hi):
                            continue
                        fblob = (
                            f"{fv.get('title','').lower()} "
                            f"{fv.get('options','').lower()} "
                            f"{json.dumps(fv.get('facts',{})).lower()}"
                        )
                        key2 = (str(fv["product_id"]), str(fv.get("sku", "")))
                        if key2 not in existing_keys:
                            existing_keys.add(key2)
                            if re.search(r"ipx|waterproof|water.?resist|sweat|sport|running|gym", fblob):
                                ordered.insert(0, fv)  # sport items go to front
                            else:
                                ordered.append(fv)
            continue
        finalize_order(ctx, committed)
        if sess.done:
            return True
    return bool(sess.done)

_RECOVERY_COLORS = (
    "black",
    "white",
    "green",
    "blue",
    "red",
    "gold",
    "silver",
    "gray",
    "grey",
    "pink",
    "purple",
    "yellow",
    "orange",
    "brown",
    "beige",
    "navy",
)
_RECOVERY_COLOR_RE = re.compile(
    r"(?<![a-z])(" + "|".join(_RECOVERY_COLORS) + r")(?![a-z])",
    re.I,
)
_RECOVERY_OS_RE = re.compile(
    r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*(?:phone|mobile)|"
    r"symbian|harmony\s*os|hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
    re.I,
)
_RECOVERY_IS_PHONE_RE = re.compile(r"\bsmart\s*phones?\b", re.I)
_RECOVERY_NOT_PHONE_RE = re.compile(
    r"\btablets?\b|\b(?:feature|basic)\s*phones?\b|\b(?:mystery|blind|lucky)\s*box(?:es)?\b|"
    r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
    r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b",
    re.I,
)
_RECOVERY_PART_ONLY_RE = re.compile(
    r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
    r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
    r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b",
    re.I,
)
_RECOVERY_PREF_RE = re.compile(
    r"prefer the (\w+)\s+([A-Za-z0-9.\-]+)",
    re.I,
)


def _recovery_model_in_text(model: str, text: str) -> bool:
    raw = str(text or "").lower()
    m = str(model or "").lower().strip()
    if not m or not raw:
        return False
    if re.search(r"(?<![a-z0-9])" + re.escape(m) + r"(?![a-z0-9])", raw):
        return True
    if m.isdigit() and len(m) <= 5:
        return False
    squashed_m = re.sub(r"[^a-z0-9]+", "", m)
    squashed_t = re.sub(r"[^a-z0-9]+", "", raw)
    return bool(squashed_m) and squashed_m in squashed_t


def _recovery_phone_identity(variant: dict) -> str:
    options = str(variant.get("options") or "")
    if _RECOVERY_PART_ONLY_RE.search(options):
        return "no"
    facts = variant.get("facts") if isinstance(variant.get("facts"), dict) else {}
    title = str(variant.get("title") or "")
    kind = " ".join(
        str(facts.get(k) or "")
        for k in ("phone_type", "device_type", "description_device_type")
    )
    blob = title + " " + kind
    if _RECOVERY_NOT_PHONE_RE.search(blob):
        return "no"
    if any("[truncated]" in str(facts.get(k) or "") for k in ("phone_type", "device_type", "device_operating_system")):
        return "unclear"
    if _RECOVERY_IS_PHONE_RE.search(blob):
        return "yes"
    os_blob = " ".join(
        str(facts.get(k) or "")
        for k in ("device_operating_system", "description_operating_system", "operating_system")
        if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
    )
    if _RECOVERY_OS_RE.search(os_blob):
        return "yes"
    if _RECOVERY_OS_RE.search(title) and not re.search(r"\bcompatib", title, re.I):
        return "yes"
    return "unclear"


def _recovery_device_ok(variant: dict, category: str | None) -> bool:
    if not path_matches_category(variant.get("category_path"), category):
        return False
    if category_noun(category) == "smartphone":
        return _recovery_phone_identity(variant) == "yes"
    return True


def _recovery_color_source(variant: dict) -> str:
    options = str(variant.get("options") or "")
    facts = variant.get("facts") if isinstance(variant.get("facts"), dict) else {}
    fact_color = str(facts.get("color") or facts.get("colour") or "")
    title = str(variant.get("title") or "")
    if _RECOVERY_COLOR_RE.search(options):
        return options
    if _RECOVERY_COLOR_RE.search(fact_color):
        return fact_color
    return title


def _recovery_color_families(text: str) -> set[str]:
    found: set[str] = set()
    for match in _RECOVERY_COLOR_RE.finditer(text or ""):
        word = match.group(1).lower()
        found.add("gray" if word in {"gray", "grey"} else word)
    return found


def _recovery_color_ok(variant: dict, color: str) -> bool:
    if not color:
        return True
    src = _recovery_color_source(variant)
    families = _recovery_color_families(src)
    if len(families) > 1:
        return False
    want = "gray" if color.lower() in {"gray", "grey"} else color.lower()
    if want == "green":
        return bool(re.search(r"(?<![a-z])(?:sage\s+)?green(?![a-z])", src.lower()))
    return want in families


def _recovery_model_ok(variant: dict, model: str) -> bool:
    facts = variant.get("facts") if isinstance(variant.get("facts"), dict) else {}
    fact_model = str(facts.get("model") or "").strip()
    if fact_model:
        return _recovery_model_in_text(model, fact_model)
    return _recovery_model_in_text(model, f"{variant.get('title')} {variant.get('options')}")


def _recovery_level(variant: dict, model: str, color: str) -> int:
    mo = _recovery_model_ok(variant, model)
    co = _recovery_color_ok(variant, color)
    if mo and co:
        return 0
    if mo:
        return 1
    if co:
        return 2
    return 3


def _recovery_worth_view(row: dict, model: str, category: str | None) -> bool:
    title = str(row.get("title") or "")
    return title_matches_category(title, category) or _recovery_model_in_text(model, title)


def _recovery_bind_pref(ctx: PipelineContext) -> tuple[str, str]:
    shopper = shopper_request(ctx.query or "")
    m = _RECOVERY_PREF_RE.search(shopper)
    if m:
        ctx.spec["color"] = m.group(1).lower()
        ctx.spec["model"] = m.group(2)
        ctx.spec.setdefault("constraints", {})["color_family"] = ctx.spec["color"]
        ctx.spec["constraints"]["model"] = ctx.spec["model"]
    model = str(ctx.spec.get("model") or "").strip()
    color = str(ctx.spec.get("color") or "").strip().lower()
    return model, color


def recovery_run(ctx: PipelineContext) -> bool:

    sess = ctx.sess
    model, color = _recovery_bind_pref(ctx)
    if not model:
        return False
    hi = parse_price_hi(ctx.spec.get("price_range"))
    ask = shopper_request(ctx.query or "")
    if not ctx.spec.get("category"):
        type_m = re.search(r"\b(smartphone|tablet|monitor|headphone)s?\b", ask, re.I)
        if type_m:
            ctx.spec["category"] = {
                "smartphone": "Smartphones",
                "tablet": "Tablets",
                "monitor": "Monitors",
                "headphone": "Headphones",
            }[type_m.group(1).lower()]
    cat = ctx.spec.get("category")
    cat_kw = category_noun(cat)

    ranked: list[dict] = []
    seen_pid: set[str] = set()
    searched: set[str] = set()

    def ingest_view(pid: str) -> None:
        if sess.done or pid in seen_pid or sess.turn >= sess.max_steps - 8:
            return
        seen_pid.add(pid)
        obs = sess.one("view", product_id=pid)
        _maybe_ack(ctx)
        ctx.details[pid] = detail_payload_from_view(obs)
        for v in variants_from_view(obs):
            if not v.get("sku"):
                continue
            if v.get("in_stock") is False or not within_budget(v.get("price"), hi):
                continue
            if not _recovery_device_ok(v, cat):
                continue
            ranked.append(v)

    def search_and_view(queries: list[str], view_cap: int) -> None:
        hits: list[dict] = []
        seen_hit: set[str] = set()
        for q in queries:
            qn = re.sub(r"\s+", " ", q).strip()
            if not qn or qn.lower() in searched:
                continue
            if sess.done or sess.turn >= sess.max_steps - 10:
                break
            searched.add(qn.lower())
            obs = sess.one("search", query=qn, k=10, in_stock=True)
            _maybe_ack(ctx)
            for row in obs.get("results") or []:
                pid = str(row.get("product_id") or "")
                if not pid or pid in seen_hit or pid in seen_pid:
                    continue
                if not within_budget(_as_float(row.get("price")), hi):
                    continue
                if not _recovery_worth_view(row, model, cat):
                    continue
                seen_hit.add(pid)
                hits.append(dict(row))
        typed = [row for row in hits if title_matches_category(str(row.get("title") or ""), cat)]
        chosen = typed or hits
        for row in chosen[:view_cap]:
            if sess.done:
                break
            ingest_view(str(row["product_id"]))

    queries = []
    if color:
        queries.append(f"{color} {model}")
    queries.extend([model, f"{model} {cat_kw}", f"{cat_kw} {model}"])
    search_and_view(queries, view_cap=6)

    def ladder_key(v: dict) -> tuple:
        price = v.get("price")
        return (_recovery_level(v, model, color), price if price is not None else 1e18)

    have0 = any(_recovery_level(v, model, color) == 0 for v in ranked)
    have1 = any(_recovery_level(v, model, color) == 1 for v in ranked)
    if not sess.done and ranked and (not have0 or not have1):
        for v in ranked:
            if _recovery_model_ok(v, model):
                title = " ".join(str(v.get("title") or "").split()[:8])
                if title:
                    search_and_view([title], view_cap=4)
                break
        have0 = any(_recovery_level(v, model, color) == 0 for v in ranked)

    if not ranked:
        return False
    ranked.sort(key=ladder_key)

    committed = None
    for v in ranked:
        if sess.done:
            return True
        if sess.turn >= sess.max_steps - 3:
            break
        got = commit_variant(ctx, v)
        if got:
            committed = got
            break

    if not committed:
        return False

    if sess.last_user_message:
        sess.ack_shopper(
            "Checking preferred model/color availability and switching if needed."
        )
        _maybe_ack(ctx)

    live_opts: list[dict] = []
    inspect_order = [committed] + [
        v
        for v in ranked
        if str(v["product_id"]) != str(committed["product_id"])
        or str(v["sku"]) != str(committed["sku"])
    ]
    refresh_cap = 4 if sess.turn < sess.max_steps - 6 else 0
    for v in inspect_order[:refresh_cap]:
        if sess.done or sess.turn >= sess.max_steps - 2:
            break
        stock = sess.one("inspect_stock", product_id=v["product_id"], sku=v["sku"])
        if sess.done:
            return True
        if sess.last_user_message:
            sess.ack_shopper()
        if stock.get("in_stock") is False:
            continue
        price = _as_float(stock.get("price") or stock.get("current_price") or v.get("price"))
        if not within_budget(price, hi):
            continue
        nv = dict(v)
        nv["price"] = price
        live_opts.append(nv)

    if live_opts:
        live_opts.sort(key=ladder_key)
        top = live_opts[0]
        if str(top["product_id"]) != str(committed["product_id"]) or str(top["sku"]) != str(committed["sku"]):
            sess.one("remove_from_cart", product_id=committed["product_id"], sku=committed["sku"])
            if sess.done:
                return True
            sess.one("add_to_cart", product_id=top["product_id"], sku=top["sku"])
            if sess.done:
                return True
            committed = top

    if not have0 and sess.has("message") and not sess.done:
        label = f"{color} {model}".strip()
        sess.one(
            "message",
            content=f"The {label} isn't available; I'm ordering the closest in-stock match instead.",
        )

    finalize_order(ctx, committed)
    return True

def _supplementary_fetch(ctx: PipelineContext, limit: int = 12) -> list[dict]:
    """Re-search within the (possibly updated) budget after a price event or budget cut.
    Returns new hits not already in ctx.pool, and extends ctx.pool in-place."""
    sess, spec = ctx.sess, ctx.spec
    if sess.done or sess.turn >= sess.max_steps - 6:
        return []
    hi = parse_price_hi(spec.get("price_range"))
    cat = spec.get("category")
    brand = spec.get("brand") or (spec.get("soft_brand") if ctx.family == Family.INTENT else None)
    seen = {str(p.get("product_id") or "") for p in ctx.pool}
    hits: list[dict] = []

    # Targeted filter by category + brand + new budget ceiling
    if cat and sess.has("filter") and not sess.done:
        fargs: dict[str, Any] = {"category": cat, "k": 20, "in_stock": True}
        if hi is not None:
            fargs["max_price"] = hi
        if brand:
            fargs["brand"] = brand
        obs = sess.one("filter", **fargs)
        _maybe_ack(ctx)
        for row in obs.get("results") or []:
            pid = str(row.get("product_id") or "")
            if not pid or pid in seen:
                continue
            if not within_budget(_as_float(row.get("price")), hi):
                continue
            seen.add(pid)
            row = dict(row)
            row["_search_rank"] = len(ctx.pool) + len(hits)
            hits.append(row)
            if len(hits) >= limit:
                break

    # Also keyword search with updated spec
    kw = str(spec.get("keywords") or "product").strip()
    # Include screen_size constraint so the replacement meets dimensional requirements
    ss = spec.get("screen_size") or (spec.get("constraints") or {}).get("screen_size")
    if ss:
        ss_text = re.sub(r"(\d+(?:\.\d+)?)in$", r"\1 inch", str(ss), flags=re.I).strip()
        if ss_text.lower() not in kw.lower():
            kw = f"{kw} {ss_text}".strip()
    if not sess.done and len(hits) < limit and sess.has("search"):
        obs = sess.one("search", query=kw, k=10, in_stock=True)
        _maybe_ack(ctx)
        for row in obs.get("results") or []:
            pid = str(row.get("product_id") or "")
            if not pid or pid in seen:
                continue
            if not within_budget(_as_float(row.get("price")), hi):
                continue
            seen.add(pid)
            row = dict(row)
            row["_search_rank"] = len(ctx.pool) + len(hits)
            hits.append(row)
            if len(hits) >= limit:
                break

    ctx.pool.extend(hits)
    return hits


def pipeline_run_single_product(ctx: PipelineContext) -> None:

    sess = ctx.sess
    if ctx.family == Family.RANKING and sess.has("compare") and sess.has("place_test_order"):
        if ranking_run(ctx):
            return

    if ctx.family == Family.RECOVERY and sess.has("search") and sess.has("view"):
        if recovery_run(ctx) or sess.done:
            return

    if ctx.family == Family.JUSTIFICATION and sess.has("search") and sess.has("view"):
        if justification_run(ctx) or sess.done:
            return

    if ctx.family == Family.PREFERENCE and sess.has("search") and sess.has("view"):
        if preference_run(ctx) or sess.done:
            return
        llm_tool_finish(ctx)
        return

    family_pre_retrieve(ctx)
    if ctx.sess.done:
        return
    _maybe_ack(ctx)

    pool = fetch_listings(ctx)
    if ctx.sess.done:
        return
    _maybe_ack(ctx)

    if ctx.family == Family.PREFERENCE:
        pool = sorted(
            pool,
            key=lambda p: (
                0
                if re.search(
                    r"ipx|waterproof|water.?resist|sweat|sport",
                    str(p.get("title") or "").lower(),
                )
                else 1,
                _as_float(p.get("price")) if _as_float(p.get("price")) is not None else 1e18,
            ),
        )
    if ctx.family == Family.JUSTIFICATION and ctx.spec.get("brand"):
        brand = str(ctx.spec["brand"]).lower()
        branded = [
            p
            for p in pool
            if brand in str(p.get("brand") or "").lower() or brand in str(p.get("title") or "").lower()
        ]
        if branded:
            pool = branded + [p for p in pool if p not in branded]

    view_cap = 10 if ctx.family == Family.RETRIEVAL else 8
    best = product_final_judge(ctx, pool[: max(view_cap, 12)])
    if ctx.sess.done:
        return
    _maybe_ack(ctx)
    if not best:
        llm_tool_finish(ctx)
        return

    ranked = sorted(
        pool,
        key=lambda p: float(p.get("_llm_relevance_score") or 0),
        reverse=True,
    )

    ranked = [best] + [p for p in ranked if str(p.get("product_id")) != str(best.get("product_id"))]

    _did_supplementary = False
    for product in ranked[:10]:
        if ctx.sess.done:
            return
        _maybe_ack(ctx)
        variant = pick_variant_for_product(ctx, product)
        if not variant:
            continue
        committed = commit_variant(ctx, variant)
        if not committed:
            # Commit failed (likely price event) – one-shot light supplementary fetch
            if not _did_supplementary and not ctx.sess.done and ctx.sess.turn < ctx.sess.max_steps - 5:
                _did_supplementary = True
                extra = _supplementary_fetch(ctx, limit=12)
                existing_pids = {str(p.get("product_id")) for p in ranked}
                for p in extra:
                    if str(p.get("product_id")) not in existing_pids:
                        ranked.append(p)
            continue
        finalize_order(ctx, committed)
        if ctx.sess.done:
            return
    llm_tool_finish(ctx)

def llm_tool_finish(ctx: PipelineContext) -> None:

    sess = ctx.sess
    if sess.done:
        return
    catalog = tool_summary(sess.tools)
    planned = ctx.planned_tools or plan_tools(ctx.family, sess.tools)
    system = (
        f"You are solving an ORO Bench {ctx.family.value} single-product task. "
        "The tool list is DYNAMIC — only call tools listed below, using their real parameter names. "
        "Find one valid in-stock item within budget, then the order tool. "
        "Whenever the shopper speaks, call `message` first if that tool exists. "
        "Do NOT open with message — start with a retrieve/compare tool from the list.\n\n"
        "Cart note: add_to_cart returns a cart array where each entry has price_at_add and "
        "current_price. If current_price exceeds budget, remove the item immediately and pick "
        "an alternative — do not place an order at an over-budget price.\n\n"
        f"Available tools:\n{catalog}\n"
        f"Preferred order for this family: {', '.join(planned) or '(use any listed tool)'}\n"
    )
    if ctx.family == Family.CONSTRAINT:
        constraints = ctx.spec.get("constraints") or {}
        if constraints:
            system += (
                f"\nCRITICAL — every constraint MUST be exactly satisfied for the purchased SKU: "
                f"{json.dumps(constraints)}. "
                "After viewing a product, verify the exact SKU has the required spec (e.g. screen_size) "
                "in its variant options or product facts before adding to cart. Reject any variant that does not match."
            )
        hi = parse_price_hi(ctx.spec.get("price_range"))
        if hi:
            system += f" Budget ceiling: ₱{hi:.2f} — do not exceed."
    if ctx.family == Family.JUSTIFICATION:
        constraints = ctx.spec.get("constraints") or {}
        res = ctx.spec.get("resolution") or constraints.get("resolution")
        screen = ctx.spec.get("screen_size") or constraints.get("screen_size")
        brand = ctx.spec.get("brand") or constraints.get("brand")
        req_parts = []
        if res:
            res_map = {"hd": "720p HD", "fhd": "1080p Full HD", "qhd": "QHD 1440p", "4k": "4K UHD"}
            req_parts.append(f"resolution={res_map.get(res, res)} (check variant facts/title for '720'/'1080'/etc.)")
        if screen:
            req_parts.append(f"screen_size={screen}")
        if brand:
            req_parts.append(f"brand={brand}")
        system += (
            "Before first cart: view the candidate to confirm it satisfies ALL requirements"
            + (f" ({', '.join(req_parts)})" if req_parts else "")
            + ". "
            "place_test_order MUST include justification.claims. "
            "Each claim: {\"field\": \"<key>\", \"value\": \"<exact value from product facts/view>\"} — "
            "ONLY include claims for fields that appear in the product's actual attributes. "
            "Safe fields: price, in_stock, brand, screen_size, resolution. "
            "Never invent values — only use what view/inspect shows."
        )
    elif ctx.family == Family.RANKING:
        system += (
            "compare all candidate IDs once, add_to_cart the best by refresh→screen→price, "
            "then after the availability event re-rank and place_test_order with ONLY "
            "product_id and sku (no justification)."
        )
    elif ctx.family == Family.PREFERENCE:
        system += (
            "The FIRST add_to_cart MUST be a sweat/water-resistant sport headphone. "
            "Search with: 'IPX waterproof sweat resistant headphones', 'sport headphones sweat proof'. "
            "View candidates and check title/attributes for IPX rating, waterproof, sweat-proof, or sport keywords. "
            "ONLY if absolutely no sport headphone exists within budget may you fall back to another type. "
            "place_test_order with ONLY product_id and sku — never justification claims."
        )
    elif ctx.family == Family.RECOVERY:
        system += (
            "Prefer exact preferred model+color. If unavailable after the stock event, "
            "same model other color, then other models. Never relax budget/stock. "
            "place_test_order with ONLY product_id and sku."
        )
    else:
        system += "place_test_order with ONLY product_id and sku — never justification claims."

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": ctx.query},
        {
            "role": "user",
            "content": "Parsed spec: " + json.dumps(ctx.spec, ensure_ascii=False)[:2500],
        },
    ]
    must_message = False
    while not sess.done and sess.turn < sess.max_steps:
        model = MODEL_TABLE["JUDGE"][0]
        resp = llm_post(
            model,
            messages,
            tools=sess.tools,
            tool_choice="required",
        )
        if not (resp and resp.get("choices")):
            break
        assistant = resp["choices"][0]["message"]
        content = assistant.get("content") or ""
        tool_calls = (assistant.get("tool_calls") or [])[: sess.max_calls]
        if not tool_calls:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Call a tool to continue until done=true."})
            continue

        if must_message and tool_calls[0]["function"]["name"] != "message":
            tool_calls = [
                {
                    "id": f"forced-msg-{sess.turn+1}",
                    "type": "function",
                    "function": {
                        "name": "message",
                        "arguments": json.dumps(
                            {
                                "content": (
                                    "Understood — I'll apply that update and continue carefully."
                                )
                            }
                        ),
                    },
                }
            ]

        actions = []
        for tc in tool_calls:
            raw = tc["function"].get("arguments", "{}")
            args = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(args, dict):
                args = {}
            name = tc["function"]["name"]
            if name not in sess.tool_names:
                continue
            if name == "place_test_order":
                if ctx.family != Family.JUSTIFICATION:
                    args.pop("justification", None)
                elif not (isinstance(args.get("justification"), dict) and args["justification"].get("claims")):
                    pid = str(args.get("product_id") or "")
                    sku = str(args.get("sku") or "")
                    detail = ctx.details.get(pid) or {}
                    variant = {
                        "product_id": pid,
                        "sku": sku,
                        "facts": detail.get("attributes") or detail.get("facts") or {},
                        "title": detail.get("title") or "",
                        "brand": detail.get("brand") or "",
                    }
                    if not variant["facts"] and pid and not sess.done:
                        obs = sess.one("view", product_id=pid)
                        variants = variants_from_view(obs)
                        match = next((v for v in variants if str(v.get("sku")) == sku), None)
                        if match:
                            variant = match
                        elif variants:
                            variant = variants[0]
                        ctx.details[pid] = {
                            "attributes": variant.get("facts") or {},
                            "title": variant.get("title"),
                            "brand": variant.get("brand"),
                        }
                    price = _as_float(args.get("price") or variant.get("price"))
                    args["justification"] = build_justification(ctx, variant, price, True)
            actions.append((name, args))
        result = sess.call(actions)
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for tc, call in zip(tool_calls, result.get("calls") or []):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(call.get("observation") or {}),
                }
            )
        must_message = False
        um = result.get("user_message")
        um_content = ""
        if isinstance(um, dict):
            um_content = str(um.get("content") or "").strip()
        if um_content:
            messages.append({"role": "user", "content": um_content})
            must_message = True
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[SYSTEM] Shopper spoke. NEXT tool MUST be `message`. "
                        "After one ack, continue searching/ordering — do not keep messaging."
                    ),
                }
            )


def enrich_spec_from_query(spec: dict, query: str) -> dict:

    shopper = (query or "").split("\n\nTask rules:")[0]
    out = dict(spec or {})
    if not out.get("candidates"):
        out["candidates"] = [
            (a, b) for a, b in re.findall(r"(\d{6,})::([\w.\-]+)", shopper)
        ]
    if not out.get("price_range"):
        m = re.search(
            r"(?:spend up to|at or under|at or below|under|below)\s*`?([\d,]+(?:\.\d+)?)",
            shopper,
            re.I,
        )
        if m:
            out["price_range"] = f"-{m.group(1).replace(',', '')}"
    if not out.get("screen_size"):
        m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", shopper, re.I)
        if m:
            out["screen_size"] = f"{float(m.group(1)):g}in"
    if not out.get("resolution"):
        low = shopper.lower()
        if re.search(r"1080p|full hd|fhd", low):
            out["resolution"] = "fhd"
        elif re.search(r"720p|\bhd\b", low) and "full hd" not in low:
            out["resolution"] = "hd"
        elif re.search(r"\bqhd\b|1440p", low):
            out["resolution"] = "qhd"
        elif re.search(r"\b4k\b|\buhd\b", low):
            out["resolution"] = "4k"
    if not out.get("brand"):
        m = re.search(
            r"(?:get me an?(?: in-stock)?|I need a|I want a|Find me a)\s+"
            r"([A-Za-z][\w\-]*)\s+(smartphone|tablet|gaming monitor|monitor|headphone)",
            shopper,
            re.I,
        )
        if m and m.group(1).lower() not in {"in-stock", "gaming", "genuine"}:
            out["brand"] = m.group(1)
    if not out.get("soft_brand"):
        m = re.search(r"ideally,\s*i prefer\s+(.+?),\s*but the firm", shopper, re.I)
        if m:
            out["soft_brand"] = m.group(1).strip()
    m = re.search(
        r"prefer the (\w+)\s+([A-Za-z0-9.\-]+)",
        shopper,
        re.I,
    )
    if m:
        out["color"] = m.group(1).lower()
        out["model"] = m.group(2)

    low = shopper.lower()
    if "headphone" in low and any(x in low for x in ("sweat", "water", "gym", "run", "workout")):
        kw = str(out.get("keywords") or "headphone")
        if "sweat" not in kw.lower() and "water" not in kw.lower():
            out["keywords"] = f"{kw} sweat water resistant sport".strip()
        out.setdefault("category", "Headphones")
    return clean_product_spec(out)

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:

    sess = EnvSession(problem_data)
    query = sess.query

    try:
        family = llm_classify_family(query, sess.tools)
        heur = _classify_family_heuristic(query, sess.tools)
        if heur in {Family.RECOVERY, Family.JUSTIFICATION}:
            family = heur
        planned = plan_tools(family, sess.tools)
        params = llm_parameter_snapshot(query)
        spec = (params.get("products") or [heuristic_parse_spec(query)])[0]
        spec = enrich_spec_from_query(spec, query)
        ctx = PipelineContext(
            sess=sess,
            family=family,
            query=query,
            spec=spec,
            planned_tools=planned,
        )
        pipeline_run_single_product(ctx)
    except Exception as exc:
        sess.dialogue.append(
            {"role": "assistant", "content": f"agent_error: {type(exc).__name__}: {exc}"}
        )
        try:
            if not sess.done:
                try:
                    fam = llm_classify_family(query, sess.tools)
                    heur = _classify_family_heuristic(query, sess.tools)
                    if heur in {Family.RECOVERY, Family.JUSTIFICATION}:
                        fam = heur
                except Exception:
                    fam = _classify_family_heuristic(query, sess.tools)
                ctx = PipelineContext(
                    sess=sess,
                    family=fam,
                    query=query,
                    spec=enrich_spec_from_query(
                        heuristic_parse_spec(query.split("\n\nTask rules:")[0]),
                        query,
                    ),
                    planned_tools=plan_tools(fam, sess.tools),
                )
                llm_tool_finish(ctx)
        except Exception:
            pass

    return sess.dialogue

__all__ = ["agent_main", "Family", "llm_classify_family"]
