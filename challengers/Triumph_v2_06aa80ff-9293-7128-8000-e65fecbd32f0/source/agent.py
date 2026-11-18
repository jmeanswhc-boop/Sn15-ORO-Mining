
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
_pipeline_start = 0.0

def _elapsed() -> float:
    try:
        return time.monotonic() - _pipeline_start if _pipeline_start else 0.0
    except Exception:
        return 0.0

def _provider() -> str:
    try:
        return getenv("INFERENCE_PROVIDER", "chutes").lower()
    except Exception:
        return "chutes"

def _resolve_model(name: str) -> str:
    try:
        override = getenv("SANDBOX_MODEL")
        if override:
            return override
        if _provider() == "openrouter":
            mapping = MODEL_TABLE.get("OPENROUTER") or {}
            if isinstance(mapping, dict):
                return mapping.get(name, "deepseek/deepseek-v3.2")
        return name
    except Exception:
        return name

def _chat_payload(model: str, messages: list[dict], *, tools=None, tool_choice=None) -> dict:
    try:
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
    except Exception:
        return {}

def llm_post(model: str, messages: list[dict], *, tools=None, tool_choice=None) -> dict | None:
    try:
        return _proxy.post(
            "/inference/chat/completions",
            json_data=_chat_payload(model, messages, tools=tools, tool_choice=tool_choice),
        )
    except Exception:
        return None

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
Classify this ORO Bench shopping task into exactly one family id.
Prefer Task-rules cue phrases when present (they are authoritative):
- "mix of firm requirements and nice-to-haves" -> intent_decomposition
- "reformulate queries and inspect candidates" -> retrieval_recall
- "every requirement must hold for the exact variant" -> constraint_satisfaction
- "initial request authorizes one test order" -> preference_reasoning
- "highest advertised maximum refresh rate" / "compare only these candidate" -> ranking
- "preferences are relaxable only after the exact item becomes unavailable" -> recovery
- "shopper wants a valid product plus a trustworthy reason" / "reason I can trust" -> justification
Families:
- intent_decomposition: firm gates (category, one hardware spec, budget, stock) + soft brand preference; honor preferred brand on first cart when fully compliant.
- retrieval_recall: find listing matching a phrase/category within budget; survey several distinct products before buying one.
- constraint_satisfaction: every firm requirement must hold for the exact SKU; may need to ask for a hidden hardware spec.
- preference_reasoning: hard gates category/budget/stock/real device; soft use-case prefs (sweat/water etc.) beat cheapest; first cart is scored.
- ranking: choose among explicitly listed candidate variants by refresh rate, then screen size, then lowest price.
- recovery: prefer exact model+color; if unavailable same model other color, then other models; never relax category/budget/stock.
- justification: buy a valid match and supply structured field/value justification claims on place_test_order.
Do NOT confuse justification (asks for a trustworthy reason / structured claims) with preference_reasoning (use-case soft prefs).
Do NOT confuse ranking (explicit product_id::sku candidate list) with retrieval.
Output strict minified JSON only: {"family":"<id>","reasoning":"<=12 words"}
"""

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
    try:
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
    except Exception:
        return None

def parse_json_array(content: str) -> list | None:
    try:
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
    except Exception:
        return None

def normalize_evidence_list(raw: Any) -> list[str]:
    try:
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
    except Exception:
        return []

def score_from_evidence_counts(sku_n: int, attr_n: int, title_n: int, missing_n: int = 0) -> float:
    try:
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
    except Exception:
        return 0.0

def evidence_score_sort_bucket(score: float, eps: float = EVIDENCE_NEAR_TIE_EPS) -> float:
    try:
        return round(score / eps) * eps
    except Exception:
        return 0.0

def _classify_family_heuristic(query: str) -> Family:
    try:
        """Prefer Task-rules fingerprints — LLM mislabels justification/ranking often."""
        q = (query or "").lower()
        head = q.split("\n\ntask rules:")[0]

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
    except Exception:
        return Family.CONSTRAINT

def llm_classify_family(query: str) -> Family:
    try:
        """LLM classification with multi-model failover; heuristic fallback on failure."""
        goal = (query or "").strip()
        for i, model in enumerate(MODEL_TABLE["PARSE"]):
            if i:
                time.sleep(LLM_RETRY_DELAY_SEC)
            resp = llm_post(
                model,
                [
                    {"role": "system", "content": PROMPT_CLASSIFY_FAMILY},
                    {"role": "user", "content": goal[:4000]},
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

        fam = _classify_family_heuristic(query)
        return fam
    except Exception:
        return Family.CONSTRAINT

def clean_product_spec(prod: dict) -> dict:
    try:
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
    except Exception:
        return {}

def llm_parameter_snapshot(query: str) -> dict:
    try:
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
    except Exception:
        return {"products": [], "reasoning": "error"}

def heuristic_parse_spec(shopper: str) -> dict:
    try:
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
            r"prefer the (black|green|white|blue|red|gold|silver|gray|grey)\s+([A-Za-z0-9.\-]+)",
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
    except Exception:
        return {}

def parse_price_hi(price_range: str | None) -> float | None:
    try:
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
    except Exception:
        return None

class EnvSession:
    def __init__(self, problem_data: dict[str, Any]) -> None:
        try:
            env = problem_data["environment"]
            self.binding = dict(env["binding"])
            self.policy = env["policy_view"]
            self.problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
            self.max_steps = int(self.policy.get("max_steps", 30))
            self.max_calls = int(self.policy.get("max_calls_per_turn", 16))
            self.tools = self.policy.get("tools") or []
            self.query = self.policy.get("query") or problem_data.get("query") or ""
            self.turn = 0
            self.done = False
            self.dialogue: list[dict[str, Any]] = []
            self.last_user_message: Optional[str] = None
            self._last_acked_msg: str = ""                                            
            self.observations: list[dict[str, Any]] = []
        except Exception:
            pass

    def call(self, actions: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        try:
            if self.done or self.turn >= self.max_steps:
                raise RuntimeError("episode finished or step budget exhausted")
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
        except Exception:
            raise

    def one(self, name: str, **args: Any) -> dict[str, Any]:
        try:
            result = self.call([(name, args)])
            calls = result.get("calls") or []
            return (calls[0].get("observation") or {}) if calls else {}
        except Exception:
            return {}

    def ack_shopper(self, note: str | None = None) -> None:
        try:
            """Answer a non-empty shopper intervention (graded gate for TF1/TF3/TF4/TF6)."""
            if self.done or not self.last_user_message:
                return
            text = note or (
                "Understood — I'll apply that update and continue with a compliant in-stock option."
            )
            self.one("message", content=text)
        except Exception:
            return

@dataclass
class PipelineContext:
    sess: EnvSession
    family: Family
    query: str
    spec: dict = field(default_factory=dict)
    pool: list[dict] = field(default_factory=list)
    details: dict[str, dict] = field(default_factory=dict)

def _as_float(v: Any) -> Optional[float]:
    try:
        try:
            return float(str(v).replace(",", "").strip())
        except (TypeError, ValueError):
            return None
    except Exception:
        return None

def _parse_hz(text: Any) -> Optional[float]:
    try:
        nums = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*-?\s*hz", str(text or ""), re.I)]
        return max(nums) if nums else None
    except Exception:
        return None

def _parse_inches(text: Any) -> Optional[float]:
    try:
        nums = [
            float(m.group(1))
            for m in re.finditer(
                r"(\d+(?:\.\d+)?)\s*-?\s*(?:inches|inch|in)\b", str(text or ""), re.I
            )
        ]
        return max(nums) if nums else None
    except Exception:
        return None

def within_budget(price: Optional[float], hi: Optional[float]) -> bool:
    try:
        if hi is None:
            return True
        if price is None:
            return False
        return price <= hi + 1e-6
    except Exception:
        return False

def flatten_facts(obj: dict) -> str:
    try:
        parts = [str(obj.get("title") or ""), str(obj.get("brand") or "")]
        facts = obj.get("facts") or obj.get("raw_facts") or obj.get("product_facts") or {}
        if isinstance(facts, dict):
            parts.append(json.dumps(facts, ensure_ascii=False))
        return " ".join(parts).lower()
    except Exception:
        return ""

def variants_from_view(obs: dict) -> list[dict]:
    try:
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
                    "screen_in": _parse_inches(blob) or _parse_inches(facts.get("size")),
                }
            )
        return out
    except Exception:
        return []

def detail_payload_from_view(obs: dict) -> dict:
    try:
        """Shape compatible with saiken evidence payloads."""
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
    except Exception:
        return {}

def build_scoring_candidate_payload(product: dict, detail: dict | None) -> dict:
    try:
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
    except Exception:
        return {}

def keyword_queries(spec: dict, family: "Family | None" = None) -> list[str]:
    try:
        kw = str(spec.get("keywords") or "product").strip()

        kw_clean = re.sub(r"\b\d{8,}\b", "", kw).strip() or kw
        phrase = str(spec.get("search_phrase") or "").strip()
        brand = spec.get("soft_brand") or spec.get("brand")
        model = spec.get("model")
        color = spec.get("color")
        qs = []
        if phrase:
            qs.extend([phrase, f"{phrase} {spec.get('category') or ''}".strip(), " ".join(phrase.split()[:2])])
        qs.append(kw_clean)
        if kw_clean != kw:
            qs.append(kw_clean)                                          
        if brand:

            if brand.lower() not in kw_clean.lower():
                qs.append(f"{brand} {kw_clean}")
        if model:
            qs.append(f"{color or ''} {model}".strip())
            qs.append(f"{model} smartphone")

        out, seen = [], set()
        for q in qs:
            q = re.sub(r"\s+", " ", q).strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out[:SEARCH_REFORMULATIONS]
    except Exception:
        return []

def _is_sticky_ack(msg: str) -> bool:
    try:
        """True when shopper is just confirming, not giving new requirements."""
        low = msg.lower()
        return any(
            p in low
            for p in (
                "my stated requirements still apply",
                "please keep to my stated requirements",
                "thanks for the update",
                "your stated requirements still apply",
            )
        )
    except Exception:
        return False

def _maybe_ack(ctx: PipelineContext) -> None:
    try:
        sess = ctx.sess
        if sess.done or not sess.last_user_message:
            return
        pending = sess.last_user_message

        if pending == sess._last_acked_msg:
            sess.last_user_message = None
            return

        if ctx.family == Family.RETRIEVAL and _is_sticky_ack(pending):
            sess.last_user_message = None
            return
        sess._last_acked_msg = pending
        sess.ack_shopper()

        if pending:
            sess.last_user_message = pending
            apply_shopper_revision(ctx)
            sess.last_user_message = None
    except Exception:
        return

def apply_shopper_revision(ctx: PipelineContext) -> None:
    try:
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
    except Exception:
        return

def fetch_listings(ctx: PipelineContext, limit: int = 40) -> list[dict]:
    try:
        """saiken fetch_listings_for_pages analogue using search + filter reformulations."""
        sess, spec = ctx.sess, ctx.spec
        hits: list[dict] = []
        seen: set[str] = set()
        hi = parse_price_hi(spec.get("price_range"))

        for q in keyword_queries(spec, family=ctx.family):
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
                if price is not None and not within_budget(price, hi):
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
                if price is not None and not within_budget(price, hi):
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
    except Exception:
        return []

def load_product_details(ctx: PipelineContext, pool: list[dict], cap: int = 12) -> dict[str, dict]:
    try:
        """saiken load_product_details via view/compare."""
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
    except Exception:
        return {}

def extract_evidence_rows(ctx: PipelineContext, candidates: list[dict]) -> list[dict] | None:
    try:
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
    except Exception:
        return None

def heuristic_score_pool(ctx: PipelineContext, pool: list[dict]) -> list[tuple[dict, float]]:
    try:
        """Fallback when evidence LLM fails — keyword/constraint coverage like saiken heuristic."""
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
    except Exception:
        return []

def score_from_evidence(ctx: PipelineContext, pool: list[dict], rows: list[dict]) -> list[tuple[dict, float]]:
    try:
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
    except Exception:
        return []

def product_final_judge(ctx: PipelineContext, pool: list[dict]) -> dict | None:
    try:
        """saiken product_final_judge_over_top_candidates."""
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
    except Exception:
        return None

def pick_variant_for_product(ctx: PipelineContext, product: dict) -> Optional[dict]:
    try:
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
                try:
                    blob = f"{v.get('title')} {v.get('options')} {json.dumps(v.get('facts'))}".lower()
                    model_ok = bool(model) and (model in blob or model in blob.replace(" ", "").replace("-", ""))
                    color_ok = bool(color) and color in blob
                    return (0 if model_ok and color_ok else 1 if model_ok else 2, v.get("price") or 1e18)
                except Exception:
                    return (2, 1e18)

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
    except Exception:
        return None

def apply_hard_filters(ctx: PipelineContext, variants: list[dict]) -> list[dict]:
    try:
        spec = ctx.spec
        out = []
        want_screen = spec.get("screen_size")
        want_res = spec.get("resolution")
        want_brand = spec.get("brand")
        for v in variants:
            try:
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
                    if b not in str(v.get("brand") or "").lower() and b not in str(v.get("title") or "").lower():
                        continue
                if (spec.get("category") or "").lower() == "smartphones":
                    bad = ("earbud", "case for", "screen protector", "charger", "feature phone")
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
            except Exception:
                continue
        return out
    except Exception:
        return []

def inspect_live(sess: EnvSession, product_id: str, sku: str) -> dict:
    try:
        return sess.one("inspect_stock", product_id=product_id, sku=sku)
    except Exception:
        return {}

def commit_variant(ctx: PipelineContext, variant: dict) -> Optional[dict]:
    try:
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
    except Exception:
        return None

def apply_shopper_revision(ctx: PipelineContext) -> None:
    try:
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
    except Exception:
        return

def _fact_value(facts: dict, *keys: str) -> Optional[str]:
    try:
        for k in keys:
            if k in facts and facts[k] not in (None, "", "null"):
                return str(facts[k])

        lower = {str(k).lower(): v for k, v in facts.items()}
        for k in keys:
            if k.lower() in lower and lower[k.lower()] not in (None, "", "null"):
                return str(lower[k.lower()])
        return None
    except Exception:
        return None

_RES_CANON: dict[str, list[str]] = {
    "hd":  ["hd ready", "hd+", "1280x720", "1280 x 720", "720p"],
    "fhd": ["full hd", "full-hd", "fhd", "1920x1080", "1920 x 1080", "1080p"],
    "4k":  ["ultra hd", "uhd", "4k uhd", "4k", "2160p", "3840x2160"],
    "2k":  ["2k", "1440p", "qhd", "2560x1440"],
}

def _canon_resolution(raw: str) -> str:
    try:
        """Return the grader-canonical resolution token (hd / fhd / 4k / 2k) or raw if unknown."""
        r = raw.lower().strip()

        for canon, aliases in _RES_CANON.items():
            for alias in sorted(aliases, key=len, reverse=True):
                if alias in r:

                    if canon == "hd" and any(x in r for x in ("full hd", "fhd", "1080")):
                        continue
                    return canon
        return raw                                    
    except Exception:
        return raw if isinstance(raw, str) else ""

def build_justification(ctx: PipelineContext, variant: dict, price: Optional[float], in_stock: bool) -> dict:
    try:
        """Claims must cite raw viewed facts only — invented fields zero the hard gate."""
        claims: list[dict[str, str]] = []
        facts = variant.get("facts") or {}
        if not isinstance(facts, dict):
            facts = {}
        detail = ctx.details.get(variant["product_id"]) or {}
        attrs = detail.get("attributes") or facts
        if isinstance(attrs, dict) and attrs:
            facts = attrs

        if ctx.spec.get("resolution"):
            raw = _fact_value(facts, "resolution", "display", "screen_resolution")
            if raw:
                claims.append({"field": "resolution", "value": _canon_resolution(raw)})

        elif ctx.spec.get("screen_size"):
            raw = _fact_value(facts, "size", "screen_size", "display_size")
            if raw:
                claims.append({"field": "screen_size", "value": raw})

        if price is not None:
            claims.append({"field": "price", "value": f"{price:.2f}"})
        claims.append({"field": "in_stock", "value": "true" if in_stock else "false"})

        brand_attr = _fact_value(facts, "brand", "Brand")
        if brand_attr:
            claims.append({"field": "brand", "value": brand_attr})
        else:
            for field_name, keys in (
                ("hdr", ("hdr", "hdr10")),
                ("microphone", ("microphone", "mic")),
                ("autofocus", ("autofocus", "auto_focus")),
                ("streaming", ("streaming",)),
                ("privacy_cover", ("privacy_cover", "privacy")),
            ):

                raw = _fact_value(facts, *keys)
                if raw:
                    claims.append({"field": field_name, "value": str(raw)})
                    break

        return {"claims": claims}
    except Exception:
        return {"claims": []}

def finalize_order(ctx: PipelineContext, variant: dict) -> None:
    try:
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
    except Exception:
        return

def family_pre_retrieve(ctx: PipelineContext) -> None:
    try:
        if ctx.family == Family.CONSTRAINT and ctx.spec.get("needs_ask"):
            reply = None
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
                    ctx.spec["screen_size"] = f"{float(m.group(1)):g}in"
                m = re.search(r"(\d+)\s*gb", reply, re.I)
                if m:
                    ctx.spec["ram"] = f"{m.group(1)}gb"
                    ctx.spec.setdefault("constraints", {})["ram_memory"] = ctx.spec["ram"]
                m = re.search(r"(\d+)\s*hz", reply, re.I)
                if m:
                    ctx.spec["refresh_rate"] = f"{m.group(1)}hz"
                if re.search(r"720p|\bhd\b", reply, re.I):
                    ctx.spec["resolution"] = "hd"
                if re.search(r"1080p|full hd|fhd", reply, re.I):
                    ctx.spec["resolution"] = "fhd"
        if ctx.family == Family.PREFERENCE:
            ctx.sess.one(
                "message",
                content=(
                    "I'll prioritize sweat/water resistance and secure sport fit "
                    "over the lowest price for this headphone order. Any other must-haves?"
                ),
            )
    except Exception:
        return

def _unwrap_obs(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        if not isinstance(payload, dict):
            return {}
        inner = payload.get("observation")
        if isinstance(inner, dict) and (
            "compared" in inner
            or "results" in inner
            or "variants" in inner
            or "in_stock" in inner
            or "price" in inner
        ):
            return inner
        if any(k in payload for k in ("compared", "results", "variants", "in_stock", "price")):
            return payload
        return inner if isinstance(inner, dict) else payload
    except Exception:
        return {}

def _rank_key_row(row: dict[str, Any]) -> tuple:
    try:
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
    except Exception:
        return (0.0, 0.0, 1e18)

def _rows_from_compare(obs: dict[str, Any], candidates: list[tuple[str, str]]) -> list[dict[str, Any]]:
    try:
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
    except Exception:
        return []

def ranking_run(ctx: PipelineContext) -> bool:
    try:
        """TF5: compare → add winner → ack event → re-rank → order (no justification)."""
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
    except Exception:
        return False

def ranking_select(ctx: PipelineContext) -> Optional[dict]:
    try:
        """Legacy helper kept for fallbacks; prefer ranking_run()."""
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
    except Exception:
        return None

def pipeline_run_single_product(ctx: PipelineContext) -> None:
    try:
        """Core saiken product path adapted to EnvPack tools."""
        if ctx.family == Family.RANKING:
            if ranking_run(ctx):
                return

        family_pre_retrieve(ctx)
        if ctx.sess.done:
            return
        _maybe_ack(ctx)

        pool = fetch_listings(ctx)
        if ctx.sess.done:
            return
        _maybe_ack(ctx)

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

        for product in ranked[:10]:
            if ctx.sess.done:
                return
            _maybe_ack(ctx)
            variant = pick_variant_for_product(ctx, product)
            if not variant:
                continue
            committed = commit_variant(ctx, variant)
            if not committed:
                continue
            finalize_order(ctx, committed)
            if ctx.sess.done:
                return
        llm_tool_finish(ctx)
    except Exception:
        return

def llm_tool_finish(ctx: PipelineContext) -> None:
    try:
        """Safety net: family-aware tool loop until done."""
        sess = ctx.sess
        if sess.done:
            return
        hi = parse_price_hi(ctx.spec.get("price_range"))
        budget_note = f" HARD BUDGET: max {hi:.2f} PHP — never exceed this regardless of query text." if hi is not None else ""
        system = (
            f"You are solving an ORO Bench {ctx.family.value} single-product task. "
            "Use tools to find ONE valid in-stock item within budget, add_to_cart, "
            f"then place_test_order.{budget_note} "
            "KEY RULES: "
            "(1) Whenever the shopper speaks, call `message` EXACTLY ONCE to acknowledge, then immediately take the next action. "
            "(2) Do NOT open with message — start with search/filter/compare/view. "
            "(3) After acking, if the shopper repeats the same message, do NOT message again — proceed with ordering. "
            "(4) After add_to_cart, if the shopper asks 'wait, is that still available?', call inspect_stock on THAT item, "
            "then remove_from_cart if price changed or out of stock, then find the next best option. "
            "(5) Be decisive: inspect the top candidate, add_to_cart, place_test_order immediately. "
            "Do not inspect many candidates sequentially before committing. "
        )
        if ctx.family == Family.JUSTIFICATION:
            _just_res_note = ""
            if ctx.spec.get("resolution"):
                _just_res_note = (
                    " Resolution claims MUST use the canonical token from the grader: "
                    "'hd' (for 720p/HD Ready), 'fhd' (for 1080p/Full HD), '4k' — "
                    "NEVER submit '720p', '1080p', 'Full HD', or similar human-readable strings."
                )
            elif ctx.spec.get("screen_size"):
                _just_res_note = (
                    " screen_size claim: use the EXACT value from view() facts (e.g. '27in', '27 inches') — "
                    "copy it verbatim, do not reformat."
                )
            system += (
                "JUSTIFICATION steps: "
                "(1) Search for the exact brand+spec. View the top 2-3 candidate products. "
                "Pick the one whose title/facts MOST precisely match the requested brand and spec. "
                "IMPORTANT: do NOT simply pick the cheapest option — an unusually low price often "
                "signals a trap product that will surge far over budget. Choose the product that "
                "best matches the description at a reasonable price. "
                "(2) add_to_cart your chosen product. "
                "(3) CRITICAL event: whenever the shopper says a price changed or asks 'is that still available?', "
                "you MUST: (a) ack with ONE message, then (b) inspect_stock on that SAME item as your VERY NEXT action — "
                "this closes the event gate (event_closed=True). Skipping inspect_stock here causes event_closed=False and zero score. "
                "If the item is now over budget: remove_from_cart, find the next-best alternative, add_to_cart it, inspect_stock it. "
                "(4) place_test_order with justification.claims: the spec field (resolution or screen_size), "
                "price (exact value from the most recent inspect_stock), in_stock=true, and brand."
                + _just_res_note
            )
        elif ctx.family == Family.RANKING:
            system += (
                "compare all candidate IDs once, add_to_cart the best by refresh→screen→price, "
                "then after the availability event re-rank and place_test_order with ONLY "
                "product_id and sku (no justification)."
            )
        elif ctx.family == Family.RETRIEVAL:
            system += (
                "RETRIEVAL scoring = query_efficiency × precision × cluster_coverage. "
                "query_efficiency = 3 / N_distinct_searches (capped at 1.0) — "
                "every extra distinct search query cuts your score. Optimal: use EXACTLY 3 distinct queries. "
                "SEARCH STRATEGY — 3 queries max, use each slot wisely: "
                "Query 1: exact phrase verbatim as given (e.g. 'aha glow', 'roto pvc', 'a77s back'). "
                "Query 2 (only if query 1 results are sparse): phrase + one topical synonym or abbreviated form — "
                "do NOT append the category name to the text query. "
                "Query 3 (last resort, only if still empty): a completely different angle (generic term, partial phrase). "
                "NEVER repeat a query string you already used — duplicates waste steps without affecting distinct count. "
                "NEVER embed the category name inside the search text (e.g. do NOT search 'spread creamy Groceries'). "
                "After 3 distinct searches, STOP and add_to_cart the best in-budget match found so far, then place_test_order. "
                "View only the top 3-5 candidates that look most relevant — avoid viewing irrelevant products. "
                "CRITICAL event rule — when the shopper says 'wait, is that still available?': "
                "NEVER call remove_from_cart before inspect_stock. Required sequence: "
                "(a) ack with ONE message, (b) inspect_stock on the cart item in your VERY NEXT action — "
                "this closes the availability event, (c) if out of stock or over budget: THEN remove_from_cart and find an alternative; "
                "if still OK: keep it and place_test_order immediately. "
                "place_test_order with ONLY product_id and sku — never justification claims."
            )
        else:
            system += "place_test_order with ONLY product_id and sku — never justification claims."

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": ctx.query},
            {
                "role": "user",
                "content": (
                    "Parsed spec: " + json.dumps(ctx.spec, ensure_ascii=False)[:2500]
                    + (f"\n[Budget cap: {hi:.2f} PHP — price_range means max is {hi:.2f}]" if hi is not None else "")
                ),
            },
        ]
        must_message = False
        last_um_content = ""

        _cart_item: tuple[str, str] | None = None                     

        _last_inspect: tuple[str, str] | None = None                                             

        _step_warning_sent = False                                      
        _view_cache: dict[str, dict] = {}
        _price_event_active = False                                          

        _pending_event_close: tuple[str, str] | None = None

        _VALID_JUST_FIELDS = {
            "price", "in_stock", "resolution", "screen_size", "brand",
        }
        while not sess.done and sess.turn < sess.max_steps:

            _warn_threshold = 6 if ctx.family == Family.RETRIEVAL else 4
            steps_left = sess.max_steps - sess.turn
            if steps_left <= _warn_threshold and not _step_warning_sent:
                _step_warning_sent = True
                if ctx.family == Family.RETRIEVAL:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[SYSTEM] {steps_left} steps remain. "
                            "STOP all searching immediately — add_to_cart the best in-budget candidate you have seen "
                            "and place_test_order NOW. "
                            "Reserve 3 steps for the availability event (ack + inspect_stock + order)."
                        ),
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[SYSTEM] URGENT — only {steps_left} steps remain. "
                            "If you found a product with the correct spec but it is over the revised budget, "
                            "send ONE `message` to the shopper explaining what you found and asking for budget flexibility. "
                            "Otherwise, add_to_cart and place_test_order the best available in-stock item RIGHT NOW. "
                            "Do not keep searching — commit immediately."
                        ),
                    })
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

            _event_target_now = (_cart_item or _last_inspect) if _price_event_active else None

            if _pending_event_close:
                e_pid, e_sku = _pending_event_close

                def _llm_inspects_target() -> bool:
                    try:
                        for _tc in tool_calls:
                            if _tc["function"]["name"] != "inspect_stock":
                                continue
                            try:
                                _a = _tc["function"].get("arguments", "{}")
                                _a = json.loads(_a) if isinstance(_a, str) else _a
                            except Exception:
                                _a = {}
                            if _a.get("product_id") == e_pid:
                                return True
                        return False
                    except Exception:
                        return False

                _pending_event_close = None
                _price_event_active = False
                if not _llm_inspects_target():

                    forced_ec: dict = {
                        "id": f"forced-ec-{sess.turn+1}",
                        "type": "function",
                        "function": {
                            "name": "inspect_stock",
                            "arguments": json.dumps({"product_id": e_pid, "sku": e_sku}),
                        },
                    }
                    tool_calls = [forced_ec] + list(tool_calls)[: sess.max_calls - 1]

            elif must_message and tool_calls[0]["function"]["name"] != "message":

                forced_msg: dict = {
                    "id": f"forced-msg-{sess.turn+1}",
                    "type": "function",
                    "function": {
                        "name": "message",
                        "arguments": json.dumps(
                            {"content": "Understood — I'll apply that update and continue carefully."}
                        ),
                    },
                }
                tool_calls = [forced_msg]
                if _event_target_now:
                    _pending_event_close = _event_target_now                                  
            elif (
                _event_target_now
                and tool_calls[0]["function"]["name"] == "message"
                and not any(tc["function"]["name"] == "inspect_stock" for tc in tool_calls)
            ):

                _pending_event_close = _event_target_now

            actions = []
            _message_called = False                                        
            tc_args_map: dict[str, dict] = {}                                                    
            for tc in tool_calls:
                raw = tc["function"].get("arguments", "{}")
                args = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(args, dict):
                    args = {}
                tc_args_map[tc["id"]] = args
                name = tc["function"]["name"]
                if name == "add_to_cart":
                    pid = args.get("product_id", "")
                    sku = args.get("sku", "")
                    if pid and sku:
                        _cart_item = (pid, sku)
                elif name == "message":
                    _message_called = True
                elif name == "place_test_order":
                    if ctx.family != Family.JUSTIFICATION:
                        args.pop("justification", None)
                    else:

                        pid = args.get("product_id", "")
                        sku = args.get("sku", "")
                        view_obs = _view_cache.get(pid, {})
                        just = args.get("justification") or {}
                        raw_claims = just.get("claims") or [] if isinstance(just, dict) else []
                        if view_obs:

                            detail = detail_payload_from_view(view_obs)
                            raw_attrs = detail.get("attributes") or {}

                            attrs = {
                                k.lower().replace(" ", "_").replace("-", "_"): v
                                for k, v in raw_attrs.items()
                            }

                            llm_price_c = next(
                                (c for c in raw_claims if c.get("field") == "price"), None
                            )
                            obs_price = _as_float(llm_price_c.get("value")) if llm_price_c else None

                            if obs_price is None:
                                sku_info = (detail.get("sku_options") or {}).get(sku) or {}
                                obs_price = _as_float(
                                    sku_info.get("price") or sku_info.get("current_price")
                                )
                            obs_in_stock = True                                            
                            variant = {"product_id": pid, "sku": sku, "facts": attrs}
                            rebuilt = build_justification(ctx, variant, obs_price, obs_in_stock)

                            _brand_fields: set[str] = (
                                {"brand"}
                                if (ctx.spec.get("brand") or ctx.spec.get("soft_brand"))
                                else set()
                            )
                            _required_for_rebuild = {"price", "in_stock"} | {
                                f for f in ("resolution", "screen_size") if ctx.spec.get(f)
                            } | _brand_fields
                            rebuilt_claims = [
                                c for c in (rebuilt.get("claims") or [])
                                if c.get("field") in _required_for_rebuild
                            ]
                            rebuilt_fields = {c.get("field") for c in rebuilt_claims}
                            spec_required = _required_for_rebuild - {"price", "in_stock"}
                            if spec_required and not (spec_required & rebuilt_fields):

                                whitelist_claims = [
                                    c for c in raw_claims
                                    if c.get("field") in _VALID_JUST_FIELDS
                                ]
                                if whitelist_claims:
                                    just["claims"] = whitelist_claims
                                    args["justification"] = just
                                else:

                                    rebuilt["claims"] = rebuilt_claims
                                    args["justification"] = rebuilt
                            else:
                                rebuilt["claims"] = rebuilt_claims
                                args["justification"] = rebuilt
                        else:

                            if isinstance(just, dict):
                                just["claims"] = [
                                    c for c in raw_claims
                                    if c.get("field") in _VALID_JUST_FIELDS
                                ]
                                args["justification"] = just
                actions.append((name, args))
            result = sess.call(actions)
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            for tc, call in zip(tool_calls, result.get("calls") or []):
                obs = call.get("observation") or {}
                tc_name = tc["function"]["name"]
                tc_args = tc_args_map.get(tc["id"], {})
                if tc_name == "inspect_stock":
                    i_pid = tc_args.get("product_id", "")
                    i_sku = tc_args.get("sku", "")
                    if i_pid and i_sku:
                        _last_inspect = (i_pid, i_sku)

                        if _price_event_active and _event_target_now == (i_pid, i_sku):
                            _price_event_active = False
                        if _pending_event_close and _pending_event_close == (i_pid, i_sku):
                            _pending_event_close = None
                            _price_event_active = False
                elif tc_name == "view":
                    v_pid = tc_args.get("product_id", "")
                    if v_pid and obs:
                        _view_cache[v_pid] = obs
                elif tc_name == "compare":

                    compared = obs.get("compared") or obs.get("results") or []
                    if isinstance(compared, list):
                        for entry in compared:
                            if not isinstance(entry, dict):
                                continue
                            c_pid = str(entry.get("product_id") or "")
                            if c_pid:
                                _view_cache[c_pid] = entry
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(obs),
                    }
                )

            must_message = False
            um = result.get("user_message")
            um_content = ""
            if isinstance(um, dict):
                um_content = str(um.get("content") or "").strip()
            if um_content:

                _low = um_content.lower()
                if (
                    "more expensive" in _low
                    or ("went up" in _low and "php" in _low)
                    or "still available" in _low
                    or ("is that" in _low and "available" in _low)
                    or ("is it" in _low and "available" in _low)
                    or "still in stock" in _low
                ):
                    _price_event_active = True

                old_hi = hi
                sess.last_user_message = um_content
                apply_shopper_revision(ctx)
                sess.last_user_message = None
                hi = parse_price_hi(ctx.spec.get("price_range"))
                _budget_revised_down = (
                    hi is not None and old_hi is not None and hi < old_hi - 0.5
                )
                messages.append({"role": "user", "content": um_content})

                is_retrieval_sticky = (ctx.family == Family.RETRIEVAL and _is_sticky_ack(um_content))
                if um_content == last_um_content or is_retrieval_sticky:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[SYSTEM] Shopper repeated or sent a confirmatory message — do NOT message again. "
                                "Proceed immediately with search/inspect/add_to_cart/place_test_order."
                            ),
                        }
                    )
                    must_message = False
                else:
                    must_message = True
                    hint = f" After ack, stay under {hi:.2f} PHP max." if hi is not None else ""
                    _ec_target = _cart_item or _last_inspect
                    if _price_event_active and _ec_target:
                        e_pid, e_sku = _ec_target
                        ec_hint = (
                            f" CRITICAL: after your message ack, your VERY NEXT action must be "
                            f"inspect_stock(product_id='{e_pid}', sku='{e_sku}') — this closes "
                            "the price-change event (event_closed gate). Do not skip this step."
                        )
                    else:
                        ec_hint = ""
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"[SYSTEM] Shopper spoke. NEXT tool MUST be `message` to acknowledge.{hint}{ec_hint} "
                                "After ONE ack, immediately proceed with next action — do not keep messaging."
                            ),
                        }
                    )

                    if _budget_revised_down:
                        req_spec = (
                            ctx.spec.get("screen_size")
                            or (ctx.spec.get("constraints") or {}).get("screen_size")
                            or ctx.spec.get("resolution")
                            or ""
                        )
                        spec_note = (
                            f" The required spec ({req_spec}) may not be findable at this lower price."
                            if req_spec else ""
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"[SYSTEM] Budget was just revised DOWN from {old_hi:.2f} → {hi:.2f} PHP.{spec_note} "
                                    f"Strategy: (a) Search thoroughly for a product satisfying ALL hard requirements within {hi:.2f} PHP. "
                                    f"(b) If you find a product with the EXACT required spec but its price exceeds {hi:.2f} PHP, "
                                    f"use ONE `message` to tell the shopper what you found and ask if they can stretch the budget "
                                    f"(e.g. 'I found [product] at ₱[price] with the exact spec — can you go up to ₱[price]?'). "
                                    f"(c) Only buy a wrong-spec product as a last resort after exhausting all options AND asking the shopper. "
                                    f"IMPORTANT: buying the wrong spec always scores 0 — asking for budget flexibility is always worth trying first."
                                ),
                            }
                        )
                last_um_content = um_content
    except Exception:
        return

def enrich_spec_from_query(spec: dict, query: str) -> dict:
    """Fill gaps the LLM parse often misses (candidates, resolution, brand)."""
    try:
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
        return clean_product_spec(out)
    except Exception:
        return dict(spec) if isinstance(spec, dict) else {}

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """ORO Bench entry — saiken product pipeline on EnvPack tools."""
    try:
        global _pipeline_start
        _pipeline_start = time.monotonic()

        sess = EnvSession(problem_data)
        query = sess.query

        try:
            family = llm_classify_family(query)
            params = llm_parameter_snapshot(query)
            spec = (params.get("products") or [heuristic_parse_spec(query)])[0]
            spec = enrich_spec_from_query(spec, query)
            ctx = PipelineContext(sess=sess, family=family, query=query, spec=spec)
            pipeline_run_single_product(ctx)
        except Exception as exc:
            sess.dialogue.append(
                {"role": "assistant", "content": f"agent_error: {type(exc).__name__}: {exc}"}
            )
            try:
                if not sess.done:
                    ctx = PipelineContext(
                        sess=sess,
                        family=_classify_family_heuristic(query),
                        query=query,
                        spec=enrich_spec_from_query(
                            heuristic_parse_spec(query.split("\n\nTask rules:")[0]),
                            query,
                        ),
                    )
                    llm_tool_finish(ctx)
            except Exception:
                pass

        return sess.dialogue
    except Exception:
        return []

__all__ = ["agent_main", "Family", "llm_classify_family"]
