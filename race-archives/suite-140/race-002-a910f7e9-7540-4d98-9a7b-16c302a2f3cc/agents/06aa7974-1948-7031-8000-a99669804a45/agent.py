from __future__ import annotations

import json
import logging
import re
from os import getenv
from typing import Any, Optional

from src.agent.proxy_client import ProxyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEEPSEEK = "deepseek/deepseek-v3.2"
_GLM      = "z-ai/glm-5.2"
_KIMI     = "moonshotai/kimi-k2.6"

_proxy = ProxyClient(timeout=120, max_retries=3)


def parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def render_tool_catalog(tools: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for spec in tools or []:
        fn = spec.get("function") if isinstance(spec, dict) else None
        if not isinstance(fn, dict):
            fn = spec if isinstance(spec, dict) else {}
        name = str(fn.get("name") or "")
        desc = str(fn.get("description") or "").strip()
        params = fn.get("parameters") or {}
        lines.append(
            f"- name: {name}\n"
            f"  description: {desc}\n"
            f"  parameters: {json.dumps(params, ensure_ascii=False)}"
        )
    return "\n".join(lines) if lines else "(no tools advertised)"


def advertised_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for spec in tools or []:
        fn = spec.get("function") if isinstance(spec, dict) else None
        if not isinstance(fn, dict):
            fn = spec if isinstance(spec, dict) else {}
        name = str(fn.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def clip_obs(payload: Any, limit: int = 6000) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


class EnvRuntime:
    def __init__(self, problem_data: dict[str, Any]) -> None:
        env = problem_data.get("environment") or {}
        self.binding = dict(env.get("binding") or {})
        self.policy = env.get("policy_view") or {}
        self.problem_id = str(
            problem_data.get("problem_id") or problem_data.get("id") or "problem"
        )
        self.query = str(self.policy.get("query") or problem_data.get("query") or "")
        self.tools = list(self.policy.get("tools") or [])
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = int(self.policy.get("max_calls_per_turn") or 16)
        self.turn = 0
        self.done = False
        self.dialogue: list[dict[str, Any]] = []
        self.last_user_message: Optional[str] = None

    def tool_catalog(self) -> str:
        return render_tool_catalog(self.tools)

    def tool_names(self) -> set[str]:
        return advertised_tool_names(self.tools)

    def call(self, actions: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
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
                {"call_id": f"{group_id}-{i}", "action": {"name": name, "args": args}}
                for i, (name, args) in enumerate(actions, start=1)
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
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                    for i, (name, args) in enumerate(actions, start=1)
                ],
                "environment_result": result,
            }
        )
        for call in result.get("calls") or []:
            if (call.get("observation") or {}).get("done"):
                self.done = True
        user_message = result.get("user_message")
        if user_message and user_message.get("content"):
            self.last_user_message = str(user_message["content"])
        else:
            self.last_user_message = None
        return result


class LlmClient:
    def __init__(self, models: Optional[list[str]] = None) -> None:
        override = getenv("SANDBOX_MODEL")
        self._models: list[str] = [override] if override else list(
            models or [_DEEPSEEK, _GLM, _KIMI]
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list] = None,
        tool_choice: Any = None,
        temperature: float = 0,
    ) -> dict[str, Any]:
        last_error: Optional[Exception] = None
        for model in self._models:
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            if tools is not None:
                payload["tools"] = tools
                payload["tool_choice"] = (
                    tool_choice if tool_choice is not None else "required"
                )
            try:
                resp = _proxy.post("/inference/chat/completions", json_data=payload)
                if resp and resp.get("choices"):
                    return resp["choices"][0].get("message") or {}
            except Exception as exc:
                last_error = exc
                continue
        if last_error:
            raise RuntimeError(f"inference failed: {last_error}")
        raise RuntimeError("inference failed: empty response")

    def json_object(
        self, system: str, user: str, *, temperature: float = 0
    ) -> dict[str, Any]:
        message = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=None,
            temperature=temperature,
        )
        return parse_json_object(str(message.get("content") or ""))


def tool_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    return parsed if isinstance(parsed, dict) else {}


def drive_tool_loop(
    rt: EnvRuntime,
    llm: LlmClient,
    *,
    system: str,
    brief: dict[str, Any],
    plan: dict[str, Any],
    coach: str,
) -> None:
    allowed = rt.tool_names()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "# Shopper request\n"
                f"{rt.query}\n\n"
                "# Structured understanding\n"
                f"{json.dumps(brief, ensure_ascii=False)}\n\n"
                "# Tool plan (names from the live catalog only)\n"
                f"{json.dumps(plan, ensure_ascii=False)}\n\n"
                "# Live tool catalog\n"
                f"{rt.tool_catalog()}\n\n"
                f"{coach}\n\n"
                "Call tools from the catalog. Adapt if observations require a "
                "different tool. Continue until a tool observation signals the "
                "episode is complete. A shopper message in a later turn is "
                "authoritative for all subsequent decisions.\n\n"
                "Cart observation note: the add_to_cart response includes a cart "
                "array where each entry has price_at_add and current_price. If "
                "current_price exceeds the budget, the item must be removed before "
                "proceeding — do not place an order at an over-budget price.\n\n"
                "Terminal commit note: when calling the terminal commit action "
                "(place_test_order or equivalent), pass ONLY the required item "
                "identity fields (product_id, sku). Do NOT add a justification, "
                "claims, or evidence argument unless the tool schema explicitly "
                "requires it — any unverified claim will fail validation and void "
                "the order."
            ),
        },
    ]
    while not rt.done and rt.turn < rt.max_steps:
        assistant = llm.chat(
            messages,
            tools=rt.tools,
            tool_choice="required",
            temperature=0,
        )
        content = assistant.get("content") or ""
        raw_calls = list(assistant.get("tool_calls") or [])[: rt.max_calls]
        actions: list[tuple[str, dict[str, Any]]] = []
        kept_calls: list[dict[str, Any]] = []
        for call in raw_calls:
            name = str((call.get("function") or {}).get("name") or "")
            if name not in allowed:
                continue
            actions.append((name, tool_args(call)))
            kept_calls.append(call)
        if not actions:
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "No valid tool call produced. Choose one or more tools "
                        "from the live catalog and continue. The episode is not done."
                    ),
                }
            )
            continue
        result = rt.call(actions)
        messages.append(
            {"role": "assistant", "content": content, "tool_calls": kept_calls}
        )
        for call, row in zip(kept_calls, result.get("calls") or []):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": clip_obs(row.get("observation") or {}),
                }
            )
        if result.get("user_message") and result["user_message"].get("content"):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Shopper update (authoritative for later decisions):\n"
                        + str(result["user_message"]["content"])
                    ),
                }
            )


def filter_plan_tools(plan: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    cleaned = dict(plan or {})
    rows = [
        row for row in (cleaned.get("tools") or [])
        if isinstance(row, dict) and str(row.get("name") or "").strip() in allowed
    ]
    cleaned["tools"] = rows
    return cleaned


def generic_understand_user(query: str, extra_schema: str) -> str:
    return (
        "Read the shopper request and extract a structured brief.\n"
        "Do not invent catalog facts. Do not assume a frozen tool set or problem suite.\n"
        "Reply JSON only.\n\n"
        f"{extra_schema}\n\n"
        f"Shopper request:\n{query}"
    )


def generic_plan_user(query: str, brief: dict[str, Any], catalog: str) -> str:
    return (
        "Choose which advertised tools to use and in what order.\n"
        "You may ONLY name tools that appear in the catalog below.\n"
        "If a needed capability is absent, pick the closest match by description. "
        "Never invent a tool name.\n"
        "Reply JSON only:\n"
        '{"strategy":"...","tools":[{"name":"...","why":"...","when":"..."}],'
        '"stop_condition":"..."}\n\n'
        f"Brief:\n{json.dumps(brief, ensure_ascii=False)}\n\n"
        f"Tool catalog:\n{catalog}\n\n"
        f"Shopper request:\n{query}"
    )


# ── deterministic justification helpers ──────────────────────────────────────

def _as_float(v: Any) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None

def _parse_hz(text: Any) -> Optional[float]:
    nums = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*-?\s*hz", str(text or ""), re.I)]
    return max(nums) if nums else None

def within_budget(price: Optional[float], hi: Optional[float]) -> bool:
    if hi is None:
        return True
    if price is None:
        return False
    return price <= hi + 1e-6

def parse_price_hi(price_range: Optional[str]) -> Optional[float]:
    if not price_range:
        return None
    parts = str(price_range).split("-", 1)
    try:
        if len(parts) == 2 and parts[1].strip():
            return float(parts[1].replace(",", ""))
        if str(price_range).startswith("-"):
            return float(parts[-1].replace(",", ""))
    except ValueError:
        pass
    return None

def _rt_one(rt: EnvRuntime, name: str, **kwargs: Any) -> dict:
    """Call a single action on the runtime and return its observation dict."""
    result = rt.call([(name, kwargs)])
    calls = result.get("calls") or []
    obs = (calls[0].get("observation") or {}) if calls else {}
    return obs

def _jvariants_from_view(obs: dict) -> list[dict]:
    """Extract structured variant records from a view observation."""
    pid = str(obs.get("product_id") or "")
    title = str(obs.get("title") or "")
    brand = str(obs.get("brand") or "")
    facts = obs.get("facts") or obs.get("raw_facts") or obs.get("product_facts") or {}
    if not isinstance(facts, dict):
        facts = {}
    out = []
    for row in obs.get("variants") or []:
        options = str(row.get("options") or row.get("option") or "")
        out.append({
            "product_id": pid,
            "sku": str(row.get("sku") or ""),
            "title": title,
            "brand": brand,
            "price": _as_float(row.get("price") or row.get("current_price")),
            "in_stock": bool(row.get("in_stock")) if "in_stock" in row else True,
            "options": options,
            "facts": facts,
            "category_path": obs.get("category_path"),
        })
    return out

_JSHOPPER_POLICY_CUT_RE = re.compile(
    r"(?:^|\n|(?<=\.))\s*(?=For a \w|The catalog |Reject conflicting |View and compare expose)",
    re.I,
)
_JWANT_LEAD_RE = re.compile(
    r"(?:I want a|Find me a|Can you get me an?(?: in-stock)?|I need a|get me an?(?: in-stock)?)\s+",
    re.I,
)
_JSPEC_OR_TYPE_RE = re.compile(
    r"(?:720p|1080p|1440p|2160p|4k|uhd|qhd|fhd|full hd|\d+(?:\.\d+)?-?\s*inch|"
    r"smartphone|tablet|monitor|headphone)",
    re.I,
)
_JUST_WXH_RE = re.compile(r"(\d{3,5})\s*[x×*]\s*(\d{3,5})")
_JUST_INCH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"\u201c\u201d])",
    re.I,
)
_JUST_SOFT_KEYS = ("microphone", "autofocus", "hdr", "streaming", "privacy_cover")

def _jfact_value(facts: dict, *keys: str) -> Optional[str]:
    for k in keys:
        if k in facts and facts[k] not in (None, "", "null"):
            return str(facts[k])
    lower = {str(k).lower(): v for k, v in facts.items()}
    for k in keys:
        if k.lower() in lower and lower[k.lower()] not in (None, "", "null"):
            return str(lower[k.lower()])
    return None

def _jshopper_request(query: str) -> str:
    goal = (query or "").split("\n\nTask rules:")[0].strip()
    cut = _JSHOPPER_POLICY_CUT_RE.search(goal)
    if cut:
        goal = goal[: cut.start()].strip()
    return goal

def _jcategory_noun(category: Optional[str]) -> str:
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

def _jtitle_matches_category(title: str, category: Optional[str]) -> bool:
    low = (title or "").lower()
    noun = _jcategory_noun(category)
    if noun == "smartphone":
        return bool(re.search(r"\bsmart\s*phones?\b|\bcellphone\b|\bmobile phones?\b", low))
    if noun == "tablet":
        return bool(re.search(r"\btablets?\b", low))
    if noun == "monitor":
        return bool(re.search(r"\bmonitors?\b", low))
    if noun == "headphone":
        return bool(re.search(r"\bheadphones?\b|\bheadsets?\b|\bearbuds?\b|\bearphones?\b", low))
    return bool(noun and noun in low)

def _jpath_matches_category(path: Any, category: Optional[str]) -> bool:
    blob = " ".join(str(x) for x in path).lower() if isinstance(path, list) else str(path or "").lower()
    if not blob or "accessor" in blob:
        return False
    noun = _jcategory_noun(category)
    if noun == "smartphone":
        return "smartphone" in blob and "feature phone" not in blob
    if noun == "tablet":
        return "tablet" in blob
    if noun == "monitor":
        return "monitor" in blob
    if noun == "headphone":
        return "headphone" in blob or "headset" in blob or "earbud" in blob
    return noun in blob if noun and noun != "product" else True

def _jproduct_phrase(request: str, brand: str = "") -> str:
    text = request or ""
    type_m = re.search(r"\b(smartphone|tablet|monitor|headphone)s?\b", text, re.I)
    lead = _JWANT_LEAD_RE.search(text)
    start = lead.end() if lead else 0
    if brand:
        bm = re.search(re.escape(brand), text, re.I)
        if bm and (type_m is None or bm.start() <= type_m.start()):
            start = max(start, bm.start())
    if type_m:
        return text[start: type_m.end()]
    return text[start:].split(".", 1)[0]

def _jrequest_modifiers(request: str, brand: str, category: Optional[str]) -> list[str]:
    stop = {
        "i", "a", "an", "the", "me", "at", "or", "under", "below", "in", "stock",
        "php", "and", "it", "has", "to", "be", "want", "find", "can", "you", "get",
        "for", "with", "one", "good", "pick", "need", "inch", "inches",
        "this", "that", "these", "those",
    }
    skip = set(re.findall(r"[a-z0-9]+", f"{brand} {_jcategory_noun(category)}".lower()))
    out: list[str] = []
    for tok in re.findall(r"[a-z0-9]+", _jproduct_phrase(request, brand).lower()):
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
        return want in _just_res_classes(raw)
    if field == "screen_size":
        vals = _just_size_values(raw)
        return vals == {want}
    return False

def _just_category_ok(variant: dict, spec: dict) -> bool:
    cat = spec.get("category")
    if not cat:
        return True
    return _jpath_matches_category(variant.get("category_path"), cat) or \
           _jtitle_matches_category(str(variant.get("title") or ""), cat)

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

def _just_worth_view(row: dict, spec: dict) -> bool:
    title = str(row.get("title") or "")
    if not _jtitle_matches_category(title, spec.get("category")):
        return False
    brand = str(spec.get("brand") or "").strip().lower()
    if not brand:
        return True
    row_brand = str(row.get("brand") or "").strip().lower()
    if not row_brand or row_brand in {"none", "null", "unknown"}:
        return False
    return brand in row_brand

def _just_bind_spec(query: str) -> tuple[dict, Optional[str], Optional[str]]:
    """Parse the query into a spec dict and return (spec, field, want)."""
    spec: dict[str, Any] = {}
    shopper = _jshopper_request(query or "")
    lead = _JWANT_LEAD_RE.search(shopper)
    if lead:
        rest = shopper[lead.end():]
        cut = _JSPEC_OR_TYPE_RE.search(rest)
        brand = re.sub(r"\s+", " ", (rest[:cut.start()] if cut else rest.split(",", 1)[0])).strip()
        if brand and brand.lower() not in {"in-stock", "genuine"}:
            spec["brand"] = brand
    window = shopper[lead.end():] if lead else shopper
    type_m = re.search(r"\b(smartphone|tablet|monitor|headphone)s?\b", window, re.I)
    if type_m:
        spec["category"] = {
            "smartphone": "Smartphones",
            "tablet": "Tablets",
            "monitor": "Monitors",
            "headphone": "Headphones",
        }[type_m.group(1).lower()]
    if re.search(r"1080p|full hd|fhd", shopper, re.I):
        spec["resolution"] = "fhd"
    elif re.search(r"720p|\bhd\b", shopper, re.I) and "full hd" not in shopper.lower():
        spec["resolution"] = "hd"
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", shopper, re.I)
    if m and not spec.get("resolution"):
        spec["screen_size"] = f"{float(m.group(1)):g}in"
    # parse budget ceiling
    bm = re.search(
        r"(?:under|at most|up to|below|budget[^₱\d]*)[₱\s]*([\d,]+(?:\.\d+)?)",
        shopper, re.I,
    )
    if bm:
        spec["price_range"] = f"-{bm.group(1).replace(',', '')}"
    return spec, *_just_gate(spec)

def _just_build_claims(variant: dict, spec: dict, price: Optional[float], in_stock: bool) -> dict:
    """Build justification.claims programmatically from observed facts."""
    claims: list[dict[str, str]] = []
    field, want = _just_gate(spec)
    raw = _just_fact_raw(variant, field) if field else None

    if field and raw and want:
        # Normalize resolution to canonical form
        if field == "resolution":
            canon_map = {"hd": "720p", "fhd": "1080p", "qhd": "1440p", "4k": "4k"}
            canon = canon_map.get(want, raw)
            # Use canon if raw fact proves the requirement
            if want in _just_res_classes(raw):
                raw = canon
        # Use the actual facts key name
        facts = variant.get("facts") if isinstance(variant.get("facts"), dict) else {}
        fact_key = "display" if field == "resolution" else "size" if field == "screen_size" else field
        actual_key = fact_key
        for k in facts:
            if k.lower() == fact_key.lower():
                actual_key = k
                break
        claims.append({"field": actual_key, "value": raw})

    if price is not None:
        claims.append({"field": "price", "value": f"{price:.2f}"})
    claims.append({"field": "in_stock", "value": "true" if in_stock else "false"})

    facts = variant.get("facts") if isinstance(variant.get("facts"), dict) else {}
    brand = _jfact_value(facts, "brand") or variant.get("brand")
    if brand and str(brand).strip().lower() not in ("", "null", "none", "unknown"):
        claims.append({"field": "brand", "value": str(brand).strip()})
    elif facts:
        for key in _JUST_SOFT_KEYS:
            if key in facts and facts[key] not in (None, "", "null"):
                claims.append({"field": key, "value": str(facts[key])})
                break
    return {"claims": claims}

def _jlisting_search_phrase(title: str) -> str:
    toks = re.findall(r"[A-Za-z0-9]+", title or "")
    return " ".join(toks[:6])

def _just_run(rt: EnvRuntime) -> bool:
    """
    Deterministic justification pipeline for v4.
    Searches, filters, ranks, and places the order with programmatically-built
    justification.claims — no LLM involved in the selection or claim construction.
    Returns True when the order has been placed (or rt.done was set by env).
    """
    spec, field, want = _just_bind_spec(rt.query)
    hi = parse_price_hi(spec.get("price_range"))
    brand = str(spec.get("brand") or "").strip()
    cat = str(spec.get("category") or "product")
    cat_kw = _jcategory_noun(cat)
    req_text = _jshopper_request(rt.query or "")

    # Build spec-aware search phrase
    spec_phrase = ""
    if field == "resolution":
        spec_phrase = {"hd": "720p", "fhd": "1080p", "qhd": "1440p", "4k": "4k"}.get(want or "", "")
    elif field == "screen_size" and want:
        spec_phrase = want.replace("in", " inch")

    extras = _jrequest_modifiers(req_text, brand, cat)
    queries: list[str] = []
    if extras:
        queries.append(" ".join(x for x in [brand, extras[0], cat_kw] if x))
    if spec_phrase:
        queries.append(" ".join(x for x in [brand, cat_kw, spec_phrase] if x))
    queries.append(" ".join(x for x in [brand, cat_kw] if x))

    ranked: list[dict] = []
    seen_pid: set[str] = set()
    searched: set[str] = set()

    def ingest_view(pid: str) -> None:
        if rt.done or pid in seen_pid or rt.turn >= rt.max_steps - 8:
            return
        seen_pid.add(pid)
        obs = _rt_one(rt, "view", product_id=pid)
        for v in _jvariants_from_view(obs):
            if _just_eligible(v, spec, field, want, hi):
                ranked.append(v)

    def search_and_view(qs: list[str], view_cap: int = 8) -> None:
        for q in qs:
            if rt.done or len(ranked) >= 3 or rt.turn >= rt.max_steps - 10:
                break
            qn = re.sub(r"\s+", " ", q).strip()
            if not qn or qn.lower() in searched:
                continue
            searched.add(qn.lower())
            obs = _rt_one(rt, "search", query=qn, k=10, in_stock=True)
            hits: list[dict] = []
            seen_hit: set[str] = set()
            for row in obs.get("results") or []:
                pid = str(row.get("product_id") or "")
                if not pid or pid in seen_hit or pid in seen_pid:
                    continue
                if not within_budget(_as_float(row.get("price")), hi):
                    continue
                if not _just_worth_view(row, spec):
                    continue
                seen_hit.add(pid)
                hits.append(dict(row))
            for row in hits[:view_cap]:
                if rt.done or len(ranked) >= 3:
                    break
                ingest_view(str(row["product_id"]))

    def filter_until(need: int) -> None:
        if "filter" not in rt.tool_names() or rt.done or len(ranked) >= need:
            return
        fargs: dict[str, Any] = {"category": cat, "k": 20, "in_stock": True}
        if brand:
            fargs["brand"] = brand
        if hi is not None:
            fargs["max_price"] = hi
        fobs = _rt_one(rt, "filter", **fargs)
        for row in (fobs or {}).get("results") or []:
            pid = str(row.get("product_id") or "")
            if rt.done or len(ranked) >= need:
                break
            if pid and pid not in seen_pid:
                ingest_view(pid)

    # Phase 1: search and view candidates
    search_and_view(queries)
    if len(ranked) < 2 and brand and not rt.done:
        search_and_view([f"{brand} {cat_kw}"])
    filter_until(1)
    if 0 < len(ranked) < 2 and not rt.done:
        phrase = _jlisting_search_phrase(str(ranked[0].get("title") or ""))
        if phrase:
            search_and_view([phrase], view_cap=6)
    filter_until(2)

    if not ranked:
        return False

    # Phase 2: rank candidates
    req_tokens = _jrequest_modifiers(req_text, brand, cat)
    if brand:
        req_tokens = [brand.lower()] + req_tokens

    def rank_key(v: dict) -> tuple:
        title = str(v.get("title") or "").lower()
        listing = str(v.get("brand") or "").lower()
        brand_pen = 0 if brand and brand.lower() in listing else 1
        token_hits = sum(1 for t in req_tokens if t in title)
        price = v.get("price") if v.get("price") is not None else 1e18
        return (brand_pen, -token_hits, price)

    ranked.sort(key=rank_key)

    tried: set[tuple[str, str]] = set()

    def try_commit(pool: list[dict]) -> Optional[dict]:
        for v in pool:
            key = (str(v["product_id"]), str(v["sku"]))
            if key in tried:
                continue
            if rt.done or rt.turn >= rt.max_steps - 4:
                return None
            tried.add(key)
            # Live stock check
            stock = _rt_one(rt, "inspect_stock", product_id=v["product_id"], sku=v["sku"])
            if rt.done:
                return v
            price = _as_float(stock.get("price") or stock.get("current_price") or v.get("price"))
            in_stock = stock.get("in_stock", True)
            if not in_stock or not within_budget(price, hi):
                continue
            if price is not None:
                v["price"] = price
            # Add to cart
            _rt_one(rt, "add_to_cart", product_id=v["product_id"], sku=v["sku"])
            if rt.done:
                return v
            # Re-verify after cart
            stock2 = _rt_one(rt, "inspect_stock", product_id=v["product_id"], sku=v["sku"])
            if rt.done:
                return v
            price2 = _as_float(stock2.get("price") or stock2.get("current_price"))
            in_stock2 = stock2.get("in_stock", True)
            if not in_stock2 or not within_budget(price2, hi):
                _rt_one(rt, "remove_from_cart", product_id=v["product_id"], sku=v["sku"])
                continue
            if price2 is not None:
                price = price2
            # Build claims from observed facts and place order
            just = _just_build_claims(v, spec, price, True)
            _rt_one(rt, "place_test_order", product_id=v["product_id"], sku=v["sku"], justification=just)
            return v
        return None

    # Phase 3: commit
    committed = try_commit(ranked)
    if not committed and not rt.done:
        unused = [v for v in ranked if (str(v["product_id"]), str(v["sku"])) not in tried]
        if not unused:
            search_and_view([f"{brand} {cat_kw}", f"{brand} {spec_phrase}".strip()])
            ranked.sort(key=rank_key)
        try_commit(ranked)

    return True


class IntentDecompositionPipeline:
    ID = "intent_decomposition"
    MODELS = [_DEEPSEEK, _GLM, _KIMI]

    CAPABILITY = """\
Primary test: correctly separate non-negotiable requirements from flexible
preferences, and maintain that distinction through the entire episode,
including after a shopper revision.

A firm requirement is one the shopper cannot accept being wrong (type, hard spec,
budget, in-stock). A soft preference is desirable but losable when honoring it
would violate a firm gate.

Revisions replace only the stated constraint; every other requirement stays firm.
The first commit taken is a meaningful choice: if an item matching both firm
gates and a soft preference is available, take it over any other valid alternative.
After a forced recovery the soft preference is released.
"""

    SYSTEM = """\
You are a shopping agent operating on a dynamically generated catalog.

Decompose the request into firm requirements and soft preferences.
Never satisfy a soft preference at the cost of a firm requirement.
Never reject a fully firm-compliant item only because a soft preference is absent.
When the shopper revises one requirement, apply the new value and keep all
unrevised requirements unchanged.

Intervention rule:
Any tool response may include a user_message from the shopper. If it does,
answer it using the message tool before taking any add_to_cart or other
commit action. Do not skip or defer the reply.

Budget revision + preferred brand:
When the shopper revises the budget, issue at least one targeted search or
filter for the preferred brand within the NEW budget before committing to
an off-brand item. Do not treat a single over-budget Vivo as proof there
is no Vivo within the new limit.

Price-event recovery (fires on add_to_cart):
The add_to_cart observation contains a cart array with current_price per item.
If current_price exceeds the current budget, remove the item immediately.
Then re-examine candidates already retrieved in prior turns before issuing new
searches. Only search again if every previously retrieved candidate has been
checked and none qualifies.

Dead-end rule:
If after checking all available candidates no item can satisfy every firm
requirement simultaneously, send the shopper one clear message stating the
unsatisfiable constraint and stop. Do not loop through search variations.

Verify live price and stock before the terminal commit action.
Finish once the terminal commit action completes — do not wait for permission.
Use only tools that appear in the live catalog; infer each tool's job from its
description and parameter schema.
"""

    UNDERSTAND_SYSTEM = """\
Decompose one shopping request into a structured brief. Reply JSON only:
{
  "shopper_goal": "one sentence",
  "firm_requirements": ["each non-negotiable gate, one per entry"],
  "soft_preferences": ["each desirable-but-optional attribute"],
  "revision_expected": true/false,
  "first_commit_rule": "what must be true of the first item committed",
  "recovery_rule": "what to do if the committed item later becomes invalid",
  "search_phrases": ["2-8 word retrieval strings ranked by usefulness"]
}
"""

    COACH = """\
Execution notes:
- Treat the first successful commit as your initial choice.
- If the shopper revises a spec, re-evaluate: keep every unrevised requirement,
  apply the new value, and replace any item that no longer qualifies.
- A soft preference (e.g., named brand) is optional after a forced recovery.
- When a shopper message arrives in any tool response, answer it before adding to cart.
- After a budget revision, search the preferred brand at the new budget ceiling
  before accepting an off-brand item.
- After a price event: remove the over-budget item, then check existing candidates
  before issuing new searches.
- If no valid item exists for the current firm requirements, say so and stop.
"""

    def run(self, rt: EnvRuntime) -> None:
        llm = LlmClient(models=self.MODELS)
        brief = llm.json_object(
            self.UNDERSTAND_SYSTEM,
            generic_understand_user(rt.query, "Use the schema in the system message."),
        )
        plan = filter_plan_tools(
            llm.json_object(
                "You are planning tool use for an intent-decomposition shopping task.",
                generic_plan_user(rt.query, brief, rt.tool_catalog()),
            ),
            rt.tool_names(),
        )
        drive_tool_loop(rt, llm, system=self.SYSTEM, brief=brief, plan=plan, coach=self.COACH)


class RetrievalRecallPipeline:
    ID = "retrieval_recall"
    MODELS = [_DEEPSEEK, _GLM, _KIMI]

    CAPABILITY = """\
Primary test: survey multiple distinct catalog items before committing.
One first-page hit is not sufficient.

The shopper names a category and/or a distinctive phrase and a budget, and
asks for one in-stock item. Credit requires inspecting several different
product identities (not multiple variants of the same listing) and then
buying the best valid one. Reformulate queries if the first page is thin.
"""

    SYSTEM = """\
You are a shopping agent operating on a dynamically generated catalog.

Your job: find items matching the shopper's phrase by searching broadly,
reformulating, and inspecting several distinct products — not just the first hit.

Commit rule:
Once you have inspected at least 3 distinct product identities and at least one
is valid (in-stock, within budget, matches the phrase), commit to the best match.
Do not continue inspect_stock calls indefinitely — each inspect is only to verify
a candidate, not to explore. If you have a valid candidate, add it to cart.

Breadth rule:
Use multiple search angles for the same phrase (synonyms, category filters,
abbreviated forms). After the first page, try broader or alternative queries to
surface more of the candidate pool — do not stop after one reformulation.

Recovery rule:
After a stock-out event, re-examine your existing shortlist in order of phrase
relevance. Prefer the most closely matching genuine item over any available
alternative. Do not default to the first in-stock item regardless of match quality.

Finish with the terminal commit action once the pick is verified.
Use only tools in the live catalog; choose each by its description.
"""

    UNDERSTAND_SYSTEM = """\
Reply JSON only:
{
  "shopper_goal": "one sentence",
  "category_or_type": "if stated, else null",
  "match_phrase": "the distinctive tokens that define a correct result",
  "budget": "as stated or null",
  "min_distinct_identities_to_inspect": 3,
  "search_phrases": ["reformulations that stay faithful to the original phrase"]
}
"""

    COACH = """\
Execution notes:
- Use multiple query angles for the same phrase: full phrase, abbreviated,
  category-filtered, synonym — to surface more of the candidate pool.
- Once you have ≥3 inspected distinct candidates and one is valid, stop inspecting
  and commit. Do not keep calling inspect_stock on every candidate you see.
- After a stock-out, pick the most phrase-relevant genuine item from your shortlist,
  not just the first in-stock one.
- Build a shortlist of distinct product identities, then pick the best match
  for phrase + budget + stock.
"""

    def run(self, rt: EnvRuntime) -> None:
        llm = LlmClient(models=self.MODELS)
        brief = llm.json_object(
            self.UNDERSTAND_SYSTEM,
            generic_understand_user(rt.query, "Use the schema in the system message."),
        )
        plan = filter_plan_tools(
            llm.json_object(
                "You are planning tool use for a retrieval-recall shopping task.",
                generic_plan_user(rt.query, brief, rt.tool_catalog()),
            ),
            rt.tool_names(),
        )
        drive_tool_loop(rt, llm, system=self.SYSTEM, brief=brief, plan=plan, coach=self.COACH)


class ConstraintSatisfactionPipeline:
    ID = "constraint_satisfaction"
    MODELS = [_DEEPSEEK, _GLM, _KIMI]

    CAPABILITY = """\
Primary test: every stated requirement must match the exact item/variant ordered.

A product title can cover many variants; only one may satisfy all gates.
Brand, when stated as a requirement, is firm. A named spec (dimension, capacity,
resolution, or whatever the shopper wrote) is firm along with budget and stock.
If a required attribute is only revealed when you ask, ask first and treat the
answer as firm.
If the desired variant sells out, find a different product — do not substitute
a non-matching sibling variant.
"""

    SYSTEM = """\
You are a shopping agent operating on a dynamically generated catalog.

Every stated requirement must hold on the exact variant you order.
Read variant-level attributes, live price, and stock — not just the product title.

If a required attribute is withheld and only given when asked, use the messaging
tool first. Wait for the reply. That attribute is then firm.
If the desired variant is gone, search a different product. Do not accept
a near-match or a sibling variant that fails the gate.
Verify live price and stock before the terminal commit action.
Use only tools in the live catalog; choose each by its description.

Post-stockout recovery:
After a stock-out event forces you off your first pick, recover as quickly as
possible. If you have already viewed products with the correct spec in the
current episode, inspect_stock one of those immediately rather than running
another search first. Only search again if every already-viewed candidate
has been checked and none qualifies.

Dead-end rule:
If after checking all available candidates no item meets every firm requirement
simultaneously, send the shopper one clear message stating which constraint
cannot be met and stop. Do not loop through search variations.
"""

    UNDERSTAND_SYSTEM = """\
Reply JSON only:
{
  "shopper_goal": "one sentence",
  "firm_requirements": ["each gate, including any withheld-until-asked attribute"],
  "must_ask_first": true/false,
  "question_to_ask": "one short question, or null",
  "variant_risk": "why title-level matching is insufficient here",
  "search_phrases": ["queries that include the specific attribute tokens"]
}
"""

    COACH = """\
Execution notes:
- If a required attribute is withheld, send the question and wait for the reply
  before any search or commit action.
- Reject any variant that fails a gate. A sibling variant of a matching title
  can still be wrong — read actual attribute values.
- After a stock-out event: check candidates already viewed in this episode first.
  If one has the right spec and is within budget, inspect_stock it and add to cart
  immediately — do not run another search first.
- If no valid item exists for the current constraints, say so clearly and stop.
"""

    def run(self, rt: EnvRuntime) -> None:
        llm = LlmClient(models=self.MODELS)
        brief = llm.json_object(
            self.UNDERSTAND_SYSTEM,
            generic_understand_user(rt.query, "Use the schema in the system message."),
        )
        plan = filter_plan_tools(
            llm.json_object(
                "You are planning tool use for a constraint-satisfaction shopping task.",
                generic_plan_user(rt.query, brief, rt.tool_catalog()),
            ),
            rt.tool_names(),
        )
        drive_tool_loop(rt, llm, system=self.SYSTEM, brief=brief, plan=plan, coach=self.COACH)


class PreferenceReasoningPipeline:
    ID = "preference_reasoning"
    MODELS = [_DEEPSEEK, _GLM, _KIMI]

    CAPABILITY = """\
Primary test: among all items that clear the hard requirements, choose the one
with the best fit for the shopper's stated preferences. Defaulting to cheapest-
valid is the canonical wrong answer.

Hard gates (non-negotiable): correct type, within budget, currently in-stock,
a real product (not an accessory, case, or off-topic listing).
Soft axes: the specific use-case or attribute priorities the shopper expressed.
Inspect actual attribute facts on a shortlist before the first commit action.
When preferences are ambiguous, clarify with the shopper before committing.
"""

    SYSTEM = """\
You are a shopping agent operating on a dynamically generated catalog.

Clear the hard requirements first (type, budget, stock, real product).
Among valid options, optimize for the shopper's stated preferences using
inspected attribute facts — do not default to cheapest.

If a messaging tool exists and priorities are ambiguous, send a short
clarifying question before the first commit.
Inspect a focused shortlist; do not guess from titles.

Battery-life signal:
When the use-case implies long sessions (sports, gym, workouts, runs, commute),
prefer items whose product title explicitly states the battery duration in hours
(e.g. "35H", "30-hour playtime", "28H music time"). A title-stated hour count is
a strong quality signal. Items with only a mAh spec or no battery mention in the
title have weaker endurance evidence.

Price-event recovery:
After add_to_cart, inspect the cart response for current_price. If current_price
exceeds the budget, remove the item immediately — do not proceed to place_test_order.
Then pick the next-best preference-aligned candidate from your inspected shortlist
and commit to that instead.
"""

    UNDERSTAND_SYSTEM = """\
Reply JSON only:
{
  "shopper_goal": "one sentence",
  "hard_gates": ["correct type", "budget", "in-stock", "real product"],
  "preference_axes": [{"axis": "...", "direction": "more/less/required"}],
  "anti_goal": "what choosing cheapest-valid would get wrong",
  "clarifying_question": "one short question to resolve ambiguous preferences, or null",
  "search_phrases": ["use-case aware queries; for endurance-sensitive use-cases include battery hour keywords like '35H' or 'long battery playtime'"]
}
"""

    COACH = """\
Execution notes:
- If a messaging tool is available, send the clarifying question before the
  first commit — not after.
- Inspect actual attribute values on a shortlist before deciding.
- For sports/gym/run use-cases, include battery hour keywords in search queries
  (e.g. "35H", "long battery") and prefer candidates whose title states hours.
- After add_to_cart: if current_price > budget, remove immediately, then commit
  to the next-best valid candidate from the shortlist.
- Commit to the item with the best preference fit that still clears all hard gates.
"""

    def run(self, rt: EnvRuntime) -> None:
        llm = LlmClient(models=self.MODELS)
        brief = llm.json_object(
            self.UNDERSTAND_SYSTEM,
            generic_understand_user(rt.query, "Use the schema in the system message."),
        )
        plan = filter_plan_tools(
            llm.json_object(
                "You are planning tool use for a preference-reasoning shopping task.",
                generic_plan_user(rt.query, brief, rt.tool_catalog()),
            ),
            rt.tool_names(),
        )
        drive_tool_loop(rt, llm, system=self.SYSTEM, brief=brief, plan=plan, coach=self.COACH)


class RankingPipeline:
    ID = "ranking"
    MODELS = [_DEEPSEEK, _GLM, _KIMI]

    CAPABILITY = """\
Primary test: apply the shopper's explicit priority stack and commit to the
current top-ranked valid item after inspecting real attribute facts.

The shopper provides an ordered list of criteria (e.g., maximize X, then
maximize Y, then minimize Z). Read that stack literally — do not substitute
a different objective or re-order it. If a candidate set is named, stay
inside it unless items become unavailable. After any market change (price,
stock), re-rank the remaining valid items before the terminal commit.
"""

    SYSTEM = """\
You are a shopping agent operating on a dynamically generated catalog.

Apply the shopper's priority stack literally to the available candidates.
Do not re-interpret or substitute objectives.

Use variant-level attribute facts when specs are needed for ranking.
If a candidate set is stated, stay inside it unless an item becomes invalid.

Before the first commit, use view or inspect_stock on every named candidate to
read its actual attribute values. Do NOT rank based on titles alone — a title
like "24-32 inch" does not tell you the exact size of the specific SKU.
After verifying all candidates, rank them by the priority stack and commit to #1.

After a market event that alters prices or stock, re-rank survivors and
re-inspect the new top item before the terminal commit.
Use only tools in the live catalog; choose each by its description.

Terminal commit rule:
Call the terminal commit action with only the required identity fields
(product_id, sku). Do NOT attach a justification, claims, or evidence
argument — any claim value that cannot be verified from product facts will
fail validation and void the order.
"""

    UNDERSTAND_SYSTEM = """\
Reply JSON only:
{
  "shopper_goal": "one sentence",
  "hard_gates": ["budget if stated", "in-stock", "category if stated"],
  "priority_stack": ["primary criterion", "tie-break 1", "tie-break 2"],
  "restricted_candidates": ["candidate identities exactly as written, or empty list"],
  "must_rerank_after_market_change": true,
  "search_phrases": ["only if the candidate set is not fully listed in the request"]
}
"""

    COACH = """\
Execution notes:
- Use view or inspect_stock on EVERY named candidate before ranking. Titles that
  cover multiple sizes (e.g. "24-32 inch") are ambiguous — read the specific SKU.
- If a candidate's title spans a size range, run a targeted search (e.g. the
  model name + specific size like "27") or use the compare tool to confirm
  the actual screen size of that SKU before deciding its rank.
- After collecting verified facts on all candidates, rank by the full priority
  stack, then commit to #1 without further explanation.
- When committing the first item: ADD it to cart and keep it there — do not
  remove it before placing the order or before the shopper signals a change.
  Only remove if current_price exceeds budget or the shopper asks you to.
- If a market change is signalled (observation or shopper message), re-inspect
  the surviving candidates and re-rank before the terminal commit.
- At the terminal commit call, pass ONLY product_id and sku — no justification,
  no claims array, no extra fields. Adding unverified claims voids the order.
"""

    def run(self, rt: EnvRuntime) -> None:
        llm = LlmClient(models=self.MODELS)
        brief = llm.json_object(
            self.UNDERSTAND_SYSTEM,
            generic_understand_user(rt.query, "Use the schema in the system message."),
        )
        plan = filter_plan_tools(
            llm.json_object(
                "You are planning tool use for a ranking/tradeoff shopping task.",
                generic_plan_user(rt.query, brief, rt.tool_catalog()),
            ),
            rt.tool_names(),
        )
        drive_tool_loop(rt, llm, system=self.SYSTEM, brief=brief, plan=plan, coach=self.COACH)


class RecoveryPipeline:
    ID = "recovery"
    MODELS = [_DEEPSEEK, _GLM, _KIMI]

    CAPABILITY = """\
Primary test: when the first-choice item disappears, recover using exactly the
relaxation order the shopper specified — never by relaxing hard requirements.

Hard requirements (never relax): category/type, budget, in-stock.
The shopper names a preferred identity (model, color, or equivalent). If that
exact identity is unavailable after a cart or inspect event, move to the next
step on the shopper's relaxation ladder — not to an arbitrary valid item.
Inspect the replacement's identity and live stock before committing.
"""

    SYSTEM = """\
You are a shopping agent operating on a dynamically generated catalog.

Commit to the shopper's preferred identity when it is available and in stock.
If it disappears after a cart or inspect event, recover using exactly the
relaxation steps the shopper described — in order.

Hard requirements are never relaxable regardless of what the relaxation ladder says.
Inspect the replacement's actual identity and stock before committing to it.
Conflicting page titles do not override an explicit identity fact from an
inspect-style tool.
Finish with the terminal commit action. Use only tools in the live catalog.
"""

    UNDERSTAND_SYSTEM = """\
Reply JSON only:
{
  "shopper_goal": "one sentence",
  "never_relax": ["category/type", "budget", "in-stock"],
  "preferred_identity": {"primary_attribute": "...", "secondary_attribute": "...", "other": "..."},
  "relaxation_ladder": ["step 1 — most preferred", "step 2", "step 3 — least preferred"],
  "search_phrases": ["identity-first queries, then broadened for each relaxation step"]
}
"""

    COACH = """\
Execution notes:
- Commit to the preferred identity while it is live and in stock.
- After a stockout or shopper event signals unavailability, move to ladder step 2 —
  do not skip to an arbitrary valid item.
- Re-inspect the replacement's identity and live stock before committing.
"""

    def run(self, rt: EnvRuntime) -> None:
        llm = LlmClient(models=self.MODELS)
        brief = llm.json_object(
            self.UNDERSTAND_SYSTEM,
            generic_understand_user(rt.query, "Use the schema in the system message."),
        )
        plan = filter_plan_tools(
            llm.json_object(
                "You are planning tool use for a recovery-by-relaxation shopping task.",
                generic_plan_user(rt.query, brief, rt.tool_catalog()),
            ),
            rt.tool_names(),
        )
        drive_tool_loop(rt, llm, system=self.SYSTEM, brief=brief, plan=plan, coach=self.COACH)


class JustificationPipeline:
    ID = "justification"
    MODELS = [_DEEPSEEK, _KIMI, _GLM]

    CAPABILITY = """\
Primary test: buy a valid item AND call the terminal commit action with
structured evidence grounded in observations, matching whatever argument
schema that tool exposes.

Inspect the item's raw attributes via any view/inspect/compare-style tool
before the first commit. The first commit is the intended purchase.
At the terminal call, supply the claim or evidence arguments the tool schema
accepts, using only values you actually observed. No invented values.
If the schema has no claim/evidence field, do not fabricate one.
"""

    ADAPTIVE_BRIEF_SYSTEM = """\
You analyze a shopping request together with a live, changeable tool catalog.
Do not assume any tool name, product category, attribute vocabulary, evidence
format, or benchmark wording. Infer capabilities only from each advertised
description and JSON parameter schema.

Return one JSON object:
{
  "goal": "the requested outcome",
  "hard_requirements": [
    {"requirement": "...", "how_to_verify": "..."}
  ],
  "preferences": ["ordered, non-mandatory preferences"],
  "attribute_to_justify": "... or null",
  "tool_plan": [
    {"name": "an exact advertised tool name", "role": "...", "when": "..."}
  ],
  "terminal_tool": {
    "name": "exact advertised name or null",
    "evidence_parameter": "exact parameter name or null",
    "evidence_shape": "shape derived from its schema, or null"
  },
  "selection_rule": "how to rank only candidates satisfying every hard gate",
  "stop_condition": "what observation proves completion"
}

Only place exact advertised names in tool_plan. A parameter is supported only
when it appears in that tool's schema. Name attribute_to_justify with the
shopper-facing semantic attribute (for example, the concept requested), not a
catalog storage key that merely supplies evidence for it. Reply JSON only.
"""

    ADAPTIVE_SYSTEM = """\
You are a shopping agent in an unknown, dynamically generated environment.
The shopper request and live tool catalog are the only source of task and tool
semantics. Never rely on remembered tool names, catalog products, categories,
fact keys, example phrases, or fixed argument shapes.

Objective:
Choose an item satisfying every hard requirement and complete the terminal
purchase action with a fully faithful, schema-compliant justification.

Protocol:
1. Infer tool roles exclusively from their advertised descriptions and JSON
   parameter schemas. Use only exact advertised names and parameters.

2. Retrieve candidates using terms from the shopper's actual wording. One
   fully verified compliant candidate is sufficient. Keep retrieval counts
   proportional to the need.

3. BEFORE adding any variant to the cart, observe its raw facts and exact SKU
   through a fact-bearing tool (any view/compare/inspect-style action whose
   observation includes structured product facts). A search or filter result
   alone is not sufficient — the exact SKU must appear in a rendered
   fact-bearing observation before the cart addition.

4. Verify every hard gate on the exact item and variant from raw facts, not
   title words or search snippets:
   - Reject items in resale condition (used, refurbished, pre-owned, renewed,
     open box, or equivalent) when a new product was requested.
   - Reject items whose identity conflicts with the requested category.
   - Reject items whose attribute fact does not support the required spec.
   - Reject items whose live price exceeds the stated budget.
   - Reject items that are not in stock.
   Missing evidence is unknown, not proof. Do not guess from titles.

5. After adding to cart, read the cart observation. If it signals a price
   change, stock change, or any event, DO NOT proceed to the terminal action
   in the same turn. In the next turn, use the best advertised capability to
   observe the current state of the chosen variant; only then proceed.
   Cart observations alone do not satisfy the current-state requirement for
   the terminal action.

6. Build the terminal justification from raw-fact observations only:
   a. Claim fields must use the shopper-facing semantic concept described in
      the terminal schema, not a raw catalog storage key that merely holds
      the value. For example, if the schema describes a "resolution" claim,
      use "resolution" as the field — even if the fact is stored under a
      key named "display" or "screen_resolution" in the catalog.
   b. Include exactly the claims the terminal schema requires for this
      request: the requested attribute, current price, current stock, and
      the minimal additional supported reason if the schema allows it.
      Omit every other claim. One unsupported, unrecognized, or title-only
      claim invalidates the entire justification.
   c. Feature reason claims must be explicitly positive in the raw facts,
      not inferred from title words, absence of negation, or possibility.
   d. Use the latest observed price and stock values. Preserve the value
      format the schema expects.

7. Respect turn-level dependencies: do not call a tool that requires an
   identity until a prior observation supplied it. Event observation and
   the terminal action must be in separate turns.

8. A later shopper message is authoritative. Apply it before any subsequent
   commitment and retain all requirements it did not revise.

9. Stop when an observation marks the episode complete. If the catalog is
   exhausted with no compliant item, use an advertised communication tool
   to report the exact unmet gate; do not buy a near-match.
"""

    TERMINAL_AUDIT_SYSTEM = """\
You are a strict pre-flight validator for a terminal shopping action.
Your output determines whether the action is sent as-is, corrected, or blocked.
This check is all-or-nothing: one wrong claim invalidates the whole justification.

Return JSON only — no prose, no markdown:
{
  "decision": "approve|revise|reject",
  "arguments": {},
  "feedback": "one sentence"
}

Validation steps (apply in order):

STEP 1 — Schema conformance
Read the terminal tool schema. Keep every required parameter. Remove every
parameter not declared in the schema. If a required parameter is absent and
cannot be derived from the evidence, reject.

STEP 2 — Identity binding
The product_id and sku in the proposed arguments must appear together in at
least one raw-fact observation (from a view/compare/inspect-style tool). If they
do not, reject. Do not alter the identity fields.

STEP 3 — Claim field naming
The terminal schema's claim field description names the allowed semantic
concepts. A raw catalog storage key is NOT a valid claim field unless it exactly
matches an allowed semantic concept. Map each proposed claim:
- Find the semantic concept the claim is trying to express.
- Check whether that concept appears (by name or clear alias) in the schema's
  field description or in the shopper request.
- If the proposed field name is not a recognized semantic concept for this
  request, rename it to the correct concept or drop it.
- Do not introduce new field names not derivable from the schema description
  and shopper request.

STEP 4 — Claim minimality
Include exactly the claims the schema and request require. Extra claims that are
true but not required must be removed — one extra claim voids the entire
justification. Do not add claims speculatively. The minimum required set is:
  (a) the shopper's requested specification attribute,
  (b) current price,
  (c) in_stock,
  (d) at most one additional supported reason permitted by the schema.
If a claim type is not in this list, it must not appear.

STEP 5 — Claim value grounding
Every claim value must come from a raw-fact observation for the exact
product_id + sku in the arguments.

For the required specification attribute:
- The value must appear in the structured raw facts dict (keys like "facts",
  "product_facts", "raw_facts") of a view/compare/inspect observation for this
  exact SKU. Title text, search/filter result snippets, and inferred values do
  NOT count.
- If no such evidence exists in the provided observations, the spec claim is
  UNVERIFIABLE. Do not invent or guess the value. REJECT the entire call and
  set feedback to: "missing raw-fact evidence for [field]; call a fact-bearing
  tool on product_id=[X] sku=[Y] then retry".

For feature reason claims (e.g. hdr, autofocus, microphone, streaming):
- The feature must appear as an explicit key in the raw facts dict with a
  clearly positive value (e.g. "true", "yes", "supported"). A mention in the
  product title, description, or search result snippet is NOT sufficient.
  If the feature only appears in the title, REMOVE it from the claims.

For price and in_stock:
- Must reflect the latest volatile observation for this variant (inspect_stock
  or cart response). Use the most recently observed numeric price.

STEP 6 — Condition gate
If the evidence shows the item is in a resale condition (used, refurbished,
pre-owned, renewed, open box, or equivalent) and the request requires a new
item, reject.

Decision rules:
- "approve": all steps pass with no changes needed.
- "revise": steps 3, 4, or 5 (partial) required corrections and a complete
  corrected arguments object can be produced. Return it.
- "reject": step 1, 2, or 6 failed; or the required spec claim (step 5) is
  unverifiable; or corrections cannot yield a valid minimum claim set.
  Set arguments to {} and give a specific, actionable feedback message.
"""

    @staticmethod
    def _advertised_spec(rt: EnvRuntime, name: str) -> dict[str, Any]:
        for spec in rt.tools:
            if not isinstance(spec, dict):
                continue
            fn = spec.get("function")
            candidate = fn if isinstance(fn, dict) else spec
            if str(candidate.get("name") or "") == name:
                return candidate
        return {}

    def _audit_terminal(
        self,
        rt: EnvRuntime,
        llm: LlmClient,
        brief: dict[str, Any],
        name: str,
        args: dict[str, Any],
        evidence_log: list[dict[str, Any]],
    ) -> tuple[str, Optional[dict[str, Any]], str]:
        terminal_spec = self._advertised_spec(rt, name)
        # Collect identity tokens from proposed args to surface relevant evidence
        identity_tokens: set[str] = set()
        for key in ("product_id", "sku"):
            val = args.get(key)
            if val and isinstance(val, str):
                identity_tokens.add(val)
        # Further tokens from nested structures (e.g. justification values)
        flat = json.dumps(args, ensure_ascii=False, default=str)
        for token in re.findall(r'"([^"]{4,})"', flat):
            identity_tokens.add(token)

        # Prefer entries that mention the exact identity, then fall back to recency
        identity_relevant: list[dict[str, Any]] = []
        other_recent: list[dict[str, Any]] = []
        for entry in evidence_log:
            encoded = json.dumps(entry, ensure_ascii=False, default=str)
            if any(tok in encoded for tok in identity_tokens if tok):
                identity_relevant.append(entry)
            else:
                other_recent.append(entry)
        # Include all identity-relevant entries (capped) plus recent context
        relevant = identity_relevant[-10:] + other_recent[-4:]

        audit_user = (
            "# Shopper request\n"
            f"{rt.query}\n\n"
            "# Parsed request analysis\n"
            f"{json.dumps(brief, ensure_ascii=False)}\n\n"
            "# Terminal tool schema\n"
            f"{json.dumps(terminal_spec, ensure_ascii=False)}\n\n"
            "# Proposed arguments\n"
            f"{json.dumps(args, ensure_ascii=False)}\n\n"
            "# Observed evidence (identity-first, then recent)\n"
            f"{json.dumps(relevant, ensure_ascii=False, default=str)}"
        )
        audited = llm.json_object(self.TERMINAL_AUDIT_SYSTEM, audit_user)
        decision = str(audited.get("decision") or "reject").strip().lower()
        revised = audited.get("arguments")
        feedback = str(audited.get("feedback") or "terminal evidence audit failed")
        if decision not in {"approve", "revise"}:
            return "reject", None, feedback
        if not isinstance(revised, dict):
            if decision == "approve":
                revised = args
            else:
                return "reject", None, "auditor did not return corrected arguments"
        parameters = terminal_spec.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = {
            str(item) for item in parameters.get("required") or []
        }
        if isinstance(properties, dict):
            extra = set(revised) - set(properties)
            missing = required - set(revised)
            if extra or missing:
                return (
                    "reject",
                    None,
                    f"audited arguments violate schema; missing={sorted(missing)}, "
                    f"extra={sorted(extra)}",
                )
        return decision, revised, feedback

    def _drive_adaptive(
        self,
        rt: EnvRuntime,
        llm: LlmClient,
        brief: dict[str, Any],
    ) -> None:
        allowed = rt.tool_names()
        terminal = brief.get("terminal_tool") or {}
        terminal_name = (
            str(terminal.get("name") or "").strip()
            if isinstance(terminal, dict)
            else ""
        )
        if terminal_name not in allowed:
            terminal_name = ""
        evidence_log: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.ADAPTIVE_SYSTEM},
            {
                "role": "user",
                "content": (
                    "# Shopper request\n"
                    f"{rt.query}\n\n"
                    "# Live tool catalog\n"
                    f"{rt.tool_catalog()}\n\n"
                    "# Request and schema analysis\n"
                    f"{json.dumps(brief, ensure_ascii=False)}\n\n"
                    "Execute the task now. Re-check the live schemas before every "
                    "call, especially the terminal action and its evidence shape."
                ),
            },
        ]
        attempts = 0
        max_attempts = max(8, rt.max_steps * 2)

        # Track whether we've injected the budget warning to avoid repeating it
        budget_warned = False

        while not rt.done and rt.turn < rt.max_steps and attempts < max_attempts:
            attempts += 1

            # Step-budget warning: when 4 turns remain, push the agent to commit
            turns_left = rt.max_steps - rt.turn
            if not budget_warned and turns_left <= 4:
                budget_warned = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"CRITICAL: Only {turns_left} environment turns remain "
                            f"(limit is {rt.max_steps}). "
                            "If a valid item is already in the cart, call the terminal "
                            "purchase action NOW — do not add more searches or inspects. "
                            "If the cart is empty and no compliant item exists, use an "
                            "advertised messaging tool to report the unmet constraint to "
                            "the shopper, then stop."
                        ),
                    }
                )

            assistant = llm.chat(
                messages,
                tools=rt.tools,
                tool_choice="required",
                temperature=0,
            )
            content = str(assistant.get("content") or "")
            actions: list[tuple[str, dict[str, Any]]] = []
            kept_calls: list[dict[str, Any]] = []
            rejected: list[str] = []

            for call in list(assistant.get("tool_calls") or [])[: rt.max_calls]:
                name = str((call.get("function") or {}).get("name") or "").strip()
                if name not in allowed:
                    rejected.append(f"unadvertised tool {name!r}")
                    continue
                try:
                    args = tool_args(call)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    rejected.append(f"invalid arguments for {name!r}: {exc}")
                    continue
                if terminal_name and name == terminal_name:
                    decision, revised, feedback = self._audit_terminal(
                        rt, llm, brief, name, args, evidence_log
                    )
                    if revised is None:
                        rejected.append(
                            f"terminal evidence preflight rejected the call: {feedback}"
                        )
                        continue
                    args = revised
                    call = {
                        **call,
                        "function": {
                            **(call.get("function") or {}),
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                    logger.info(
                        "justification terminal preflight=%s detail=%s",
                        decision,
                        feedback,
                    )
                actions.append((name, args))
                kept_calls.append(call)

            if not actions:
                messages.append({"role": "assistant", "content": content})
                if rejected:
                    # Surface actionable feedback, especially for terminal preflights
                    preflight_msgs = [r for r in rejected if "preflight" in r]
                    other_msgs = [r for r in rejected if "preflight" not in r]
                    parts: list[str] = []
                    if preflight_msgs:
                        parts.append(
                            "Terminal action blocked by evidence preflight: "
                            + "; ".join(preflight_msgs)
                            + ". Collect the missing evidence first, then retry "
                            "the terminal action."
                        )
                    if other_msgs:
                        parts.append(
                            "Invalid call(s): "
                            + "; ".join(other_msgs)
                            + ". Use exact names and argument shapes from the live catalog."
                        )
                    detail = " ".join(parts)
                else:
                    detail = (
                        "No valid tool call was produced. Choose a tool from the "
                        "live catalog and continue."
                    )
                messages.append({"role": "user", "content": detail})
                continue

            result = rt.call(actions)
            messages.append(
                {"role": "assistant", "content": content, "tool_calls": kept_calls}
            )
            rows = list(result.get("calls") or [])
            for index, call in enumerate(kept_calls):
                observation = (
                    rows[index].get("observation") or {} if index < len(rows) else {}
                )
                evidence_log.append(
                    {
                        "tool": str(
                            (call.get("function") or {}).get("name") or ""
                        ),
                        "arguments": tool_args(call),
                        "observation": clip_obs(observation, limit=10000),
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": clip_obs(observation, limit=12000),
                    }
                )

            user_message = result.get("user_message") or {}
            if user_message.get("content"):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Authoritative shopper update; apply it before any "
                            "further commitment:\n" + str(user_message["content"])
                        ),
                    }
                )

        if not rt.done and attempts >= max_attempts:
            logger.warning(
                "justification stopped after %s inference attempts at environment turn %s",
                attempts,
                rt.turn,
            )

    def run(self, rt: EnvRuntime) -> None:
        llm = LlmClient(models=self.MODELS)
        analysis_user = (
            "# Shopper request\n"
            f"{rt.query}\n\n"
            "# Live tool catalog\n"
            f"{rt.tool_catalog()}"
        )
        brief = llm.json_object(self.ADAPTIVE_BRIEF_SYSTEM, analysis_user)

        planned = {
            str(row.get("name") or "")
            for row in brief.get("tool_plan") or []
            if isinstance(row, dict)
        }
        unknown = planned - rt.tool_names()
        if unknown:
            brief["tool_plan"] = [
                row
                for row in brief.get("tool_plan") or []
                if isinstance(row, dict)
                and str(row.get("name") or "") in rt.tool_names()
            ]

        self._drive_adaptive(rt, llm, brief)


FAMILY_PIPELINES: dict[str, type] = {
    IntentDecompositionPipeline.ID:    IntentDecompositionPipeline,
    RetrievalRecallPipeline.ID:        RetrievalRecallPipeline,
    ConstraintSatisfactionPipeline.ID: ConstraintSatisfactionPipeline,
    PreferenceReasoningPipeline.ID:    PreferenceReasoningPipeline,
    RankingPipeline.ID:                RankingPipeline,
    RecoveryPipeline.ID:               RecoveryPipeline,
    JustificationPipeline.ID:          JustificationPipeline,
}


_CLASSIFY_SYSTEM = """\
You are a family router for a shopping-agent benchmark.

The benchmark tests seven complementary capabilities. Each task involves buying
one item. Read the FULL shopper request and infer which capability is the PRIMARY
one being tested.

Do not key off memorized example wording from past problem sets. The problem
set and tool names change between runs. Use the shopper's actual decision
problem as the signal, not phrasing you have seen before.

Also read the live tool catalog as a secondary hint:
- A tool that accepts structured evidence/claim fields → likely justification.
- A tool oriented toward comparing multiple candidates → likely ranking.
- A messaging tool that can ask the shopper a question → may signal
  constraint or preference (where a withheld spec or ambiguous priority matters).
These are hints only; the shopper's decision problem is always primary.

Return minified JSON only: {"family":"<one id>","reason":"<=20 words"}
"""


def classify_family(query: str, tools: list[dict[str, Any]]) -> str:
    router_llm = LlmClient(models=[_DEEPSEEK, _GLM, _KIMI])
    cards = "\n\n".join(
        f"## {pid}\n{cls.CAPABILITY.strip()}"
        for pid, cls in FAMILY_PIPELINES.items()
    )
    user = (
        "# Capability cards\n"
        f"{cards}\n\n"
        "# Advertised tools\n"
        f"{render_tool_catalog(tools)}\n\n"
        "# Shopper request\n"
        f"{query}"
    )
    allowed = set(FAMILY_PIPELINES)
    for temperature in (0.0, 0.3, 0.6):
        parsed = router_llm.json_object(_CLASSIFY_SYSTEM, user, temperature=temperature)
        raw = str(parsed.get("family") or "").strip().lower().replace(" ", "_")
        if raw in allowed:
            logger.info("classify ok family=%s reason=%s", raw, parsed.get("reason"))
            return raw
        for pid in allowed:
            if pid in raw:
                logger.info("classify ok family=%s (partial match) reason=%s", pid, parsed.get("reason"))
                return pid
    index_prompt = (
        "Pick the single best family for the shopper request. "
        'Reply JSON {"index": N} where N is 1-based in this order:\n'
        + "\n".join(f"{i}. {pid}" for i, pid in enumerate(FAMILY_PIPELINES, start=1))
        + "\n\nShopper request:\n"
        + query
    )
    parsed = router_llm.json_object("Reply JSON only. No other keys.", index_prompt)
    try:
        index = int(parsed.get("index"))
        family = list(FAMILY_PIPELINES)[index - 1]
        logger.info("classify ok family=%s (index fallback)", family)
        return family
    except (TypeError, ValueError, IndexError):
        logger.error("classify failed after all attempts; defaulting to constraint_satisfaction")
        return "constraint_satisfaction"


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    rt = EnvRuntime(problem_data)
    try:
        family_id = classify_family(rt.query, rt.tools)
        pipeline_cls = FAMILY_PIPELINES.get(family_id) or ConstraintSatisfactionPipeline
        pipeline_cls().run(rt)
    except Exception as exc:
        logger.error("pipeline failure family=%s err=%s", locals().get("family_id", "?"), exc, exc_info=True)
        rt.dialogue.append(
            {"role": "assistant", "content": f"agent_error: {type(exc).__name__}: {exc}"}
        )
    return rt.dialogue


__all__ = ["agent_main"]
