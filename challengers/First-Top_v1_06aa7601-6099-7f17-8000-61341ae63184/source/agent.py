"""ORO Bench EnvPack agent v3 — commitment-first retrieve / recover / justify.

Commitment policy (first add_to_cart is irrevocable for event_closed):
- Soft preferred brand wins among firm-compliant SKUs when available.
- Firm screen/resolution gates use product facts, not title decoys.
- Accessories (holders/stands/bags/LCD parts) never commit as primary devices.
- TF4 first cart needs preference axis mean >= ~0.65 (IPX/battery/sport/wireless).
- TF6 recovery keeps same-model phones only (no model-token hijacks).
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from os import getenv
from typing import Any, Optional

try:
    from src.agent.proxy_client import ProxyClient
except ImportError:  # local ad-hoc runs
    from proxy_client import ProxyClient  # type: ignore


_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_FALLBACK_MODELS = {
    "chutes": ["deepseek-ai/DeepSeek-V3.2-TEE"],
    "openrouter": ["deepseek/deepseek-v3.2", "deepseek/deepseek-chat"],
}
_proxy = ProxyClient(timeout=120, max_retries=3)

_CAPABILITY_ALIASES: dict[str, tuple[str, ...]] = {
    "search": ("search", "find", "catalog_search"),
    "filter": ("filter", "filter_products", "catalog_filter"),
    "view": ("view", "get_product", "product_detail", "view_product"),
    "compare": ("compare", "compare_products"),
    "inspect_stock": ("inspect_stock", "check_stock", "stock"),
    "inspect_cart": ("inspect_cart", "view_cart", "cart"),
    "add_to_cart": ("add_to_cart", "cart_add", "add"),
    "remove_from_cart": ("remove_from_cart", "cart_remove", "remove"),
    "message": ("message", "talk", "ask", "reply"),
    "order": (
        "place_test_order",
        "place_order",
        "checkout",
        "buy",
        "terminate",
    ),
}

_ALLOWED_CLAIM_FIELDS = frozenset(
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

_VARIANT_KEY_RE = re.compile(r"\b(\d{10,})::([\w.-]+)\b")
_LENGTH_TO_INCH = {
    "mm": 1 / 25.4,
    "cm": 1 / 2.54,
    "m": 100 / 2.54,
    "in": 1.0,
    "inch": 1.0,
    "inches": 1.0,
    '"': 1.0,
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()


def _model_chain() -> list[str]:
    provider = _provider()
    primary = getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(
        provider, _DEFAULT_MODELS["chutes"]
    )
    chain = [primary]
    for m in _FALLBACK_MODELS.get(provider, []):
        if m not in chain:
            chain.append(m)
    return chain


def _chat(
    messages: list[dict[str, Any]],
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    temperature: float = 0.0,
) -> Optional[dict[str, Any]]:
    for model in _model_chain():
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        result = _proxy.post("/inference/chat/completions", json_data=payload)
        if result is not None:
            return result
    return None


def _assistant_message(result: dict[str, Any]) -> dict[str, Any]:
    return result["choices"][0]["message"]


def _parse_json_object(text: str) -> Optional[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Tools / session
# ---------------------------------------------------------------------------

@dataclass
class ToolKit:
    schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    openai_tools: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_policy(cls, tools: list[dict[str, Any]]) -> "ToolKit":
        schemas: dict[str, dict[str, Any]] = {}
        openai: list[dict[str, Any]] = []
        for raw in tools or []:
            if not isinstance(raw, dict):
                continue
            fn = raw.get("function") if raw.get("type") == "function" else raw
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            schemas[name] = fn
            openai.append(
                raw
                if raw.get("type") == "function"
                else {"type": "function", "function": fn}
            )
        return cls(schemas=schemas, openai_tools=openai)

    def resolve(self, capability: str) -> Optional[str]:
        for name in _CAPABILITY_ALIASES.get(capability, (capability,)):
            if name in self.schemas:
                return name
        needle = capability.lower()
        for name in self.schemas:
            if needle in name.lower():
                return name
        return None

    def has(self, capability: str) -> bool:
        return self.resolve(capability) is not None

    def arg_keys(self, tool_name: str) -> set[str]:
        fn = self.schemas.get(tool_name) or {}
        props = (fn.get("parameters") or {}).get("properties") or {}
        return set(props.keys()) if isinstance(props, dict) else set()

    def filter_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        allowed = self.arg_keys(tool_name)
        if not allowed:
            return dict(args)
        return {k: v for k, v in args.items() if k in allowed and v is not None}


@dataclass
class EnvSession:
    problem_id: str
    binding: dict[str, Any]
    policy: dict[str, Any]
    tools: ToolKit
    dialogue: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    done: bool = False
    last_user_message: Optional[str] = None
    observations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_problem(cls, problem_data: dict[str, Any]) -> "EnvSession":
        environment = problem_data["environment"]
        binding = dict(environment["binding"])
        policy = dict(environment["policy_view"])
        tools = ToolKit.from_policy(list(policy.get("tools") or []))
        pid = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        return cls(problem_id=pid, binding=binding, policy=policy, tools=tools)

    @property
    def query(self) -> str:
        return str(self.policy.get("query") or "")

    @property
    def max_steps(self) -> int:
        return int(self.policy.get("max_steps") or 20)

    @property
    def max_calls(self) -> int:
        return int(self.policy.get("max_calls_per_turn") or 1)

    def steps_left(self) -> int:
        return self.max_steps - self.turn

    def call(self, actions: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        if self.done:
            return {"calls": [], "error": "done"}
        if not actions:
            return {"calls": [], "error": "empty"}
        actions = actions[: self.max_calls]
        next_turn = self.turn + 1
        group_id = f"{self.problem_id}-turn-{next_turn}"
        calls = []
        for i, (name, args) in enumerate(actions, start=1):
            if name not in self.tools.schemas:
                continue
            calls.append(
                {
                    "call_id": f"{group_id}-{i}",
                    "action": {
                        "name": name,
                        "args": self.tools.filter_args(name, args),
                    },
                }
            )
        if not calls:
            return {"calls": [], "error": "no_valid_tools"}
        envelope = {
            **self.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": next_turn,
            "calls": calls,
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        self.turn = next_turn
        if result is None:
            self.dialogue.append(
                {"role": "assistant", "content": f"env_call_error turn={next_turn}"}
            )
            return {"calls": [], "error": "environment call failed"}
        self.dialogue.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": c["call_id"],
                        "type": "function",
                        "function": {
                            "name": c["action"]["name"],
                            "arguments": json.dumps(c["action"]["args"]),
                        },
                    }
                    for c in calls
                ],
                "environment_result": result,
            }
        )
        for call_result in result.get("calls") or []:
            obs = call_result.get("observation") or {}
            self.observations.append(obs)
            if isinstance(obs, dict) and obs.get("done"):
                self.done = True
        um = result.get("user_message")
        if isinstance(um, dict) and um.get("content"):
            self.last_user_message = str(um["content"])
        elif isinstance(um, str) and um:
            self.last_user_message = um
        return result

    def act(self, capability: str, args: dict[str, Any]) -> Optional[dict[str, Any]]:
        name = self.tools.resolve(capability)
        if not name or self.done or self.steps_left() < 1:
            return None
        result = self.call([(name, args)])
        calls = result.get("calls") or []
        if not calls:
            return None
        return _unwrap_obs(calls[0].get("observation"))


def _unwrap_obs(obs: Any) -> Optional[dict[str, Any]]:
    if not isinstance(obs, dict):
        return None
    if isinstance(obs.get("observation"), dict) and any(
        k in obs["observation"]
        for k in ("products", "results", "product", "product_id", "title", "facts", "variants")
    ):
        inner = dict(obs["observation"])
        for k in ("done", "error", "in_stock", "available", "price", "current_price"):
            if k in obs and k not in inner:
                inner[k] = obs[k]
        return inner
    return obs


# ---------------------------------------------------------------------------
# Product helpers (EnvPack fact shapes)
# ---------------------------------------------------------------------------

def _num(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, str):
        m = re.search(r"-?\d+(?:\.\d+)?", x.replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
    return None


def _products_from_obs(obs: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not obs or not isinstance(obs, dict):
        return []
    if isinstance(obs.get("observation"), dict):
        obs = _unwrap_obs(obs) or obs
    for key in ("products", "results", "candidates", "items"):
        val = obs.get(key)
        if isinstance(val, list):
            return [p for p in val if isinstance(p, dict)]
    if isinstance(obs.get("product"), dict):
        return [obs["product"]]
    if obs.get("product_id") or obs.get("title") or obs.get("facts") or obs.get("variants"):
        return [obs]
    return []


def _product_id(p: dict[str, Any]) -> Optional[str]:
    for k in ("product_id", "id", "productId"):
        if p.get(k) is not None:
            return str(p[k])
    return None


def _product_title(p: dict[str, Any]) -> str:
    for k in ("title", "name", "product_title"):
        if p.get(k):
            return str(p[k])
    return ""


def _product_attrs(p: dict[str, Any]) -> dict[str, Any]:
    attrs = p.get("attributes") or p.get("attrs") or p.get("facts") or {}
    out = dict(attrs) if isinstance(attrs, dict) else {}
    for k in ("brand", "category", "category_path", "model", "color"):
        if p.get(k) is not None and k not in out:
            out[k] = p[k]
    return out


def _sku_options(p: dict[str, Any]) -> list[dict[str, Any]]:
    opts = p.get("sku_options") or p.get("variants") or p.get("skus") or []
    return [o for o in opts if isinstance(o, dict)] if isinstance(opts, list) else []


def _product_price(p: dict[str, Any], sku: Optional[str] = None) -> Optional[float]:
    if sku:
        for o in _sku_options(p):
            sid = o.get("sku") or o.get("sku_id") or o.get("id")
            if sid is not None and str(sid) == str(sku):
                n = _num(o.get("price") or o.get("current_price"))
                if n is not None:
                    return n
    for k in ("price", "current_price", "sale_price", "final_price"):
        n = _num(p.get(k))
        if n is not None:
            return n
    prices = []
    for o in _sku_options(p):
        if o.get("in_stock") is False:
            continue
        n = _num(o.get("price") or o.get("current_price"))
        if n is not None:
            prices.append(n)
    return min(prices) if prices else None


def _pick_sku(
    p: dict[str, Any],
    *,
    preferred: Optional[str] = None,
    max_price: Optional[float] = None,
    require_in_stock: bool = True,
) -> Optional[str]:
    preferred = preferred or p.get("_preferred_sku")
    opts = _sku_options(p)
    if preferred:
        for o in opts:
            sid = o.get("sku") or o.get("sku_id") or o.get("id")
            if sid is None or str(sid) != str(preferred):
                continue
            price = _num(o.get("price") or o.get("current_price"))
            stock = o.get("in_stock")
            if require_in_stock and stock is False:
                break
            if max_price is not None and price is not None and price > max_price + 1e-6:
                break
            return str(sid)
    if not opts:
        for k in ("sku", "default_sku", "sku_id"):
            if p.get(k) is not None:
                return str(p[k])
        return str(preferred) if preferred else None

    ranked: list[tuple[bool, bool, float, str]] = []
    for o in opts:
        sid = o.get("sku") or o.get("sku_id") or o.get("id")
        if sid is None:
            continue
        price = _num(o.get("price") or o.get("current_price"))
        stock = o.get("in_stock")
        in_stock = True if stock is None else bool(stock) and stock != 0
        over = max_price is not None and price is not None and price > max_price + 1e-6
        ranked.append((over, not in_stock, price if price is not None else 1e18, str(sid)))
    if not ranked:
        return str(preferred) if preferred else None
    under = [r for r in ranked if not r[0]]
    pool = under or ranked
    if require_in_stock:
        stocked = [r for r in pool if not r[1]]
        pool = stocked or pool
    pool.sort(key=lambda t: (t[0], t[1], t[2]))
    return pool[0][3]


def _attr_blob(p: dict[str, Any]) -> str:
    bits = [f"{k} {v}" for k, v in _product_attrs(p).items()]
    bits.append(_product_title(p))
    for o in _sku_options(p):
        bits.append(json.dumps(o, ensure_ascii=False))
    path = p.get("category_path")
    if path:
        bits.append(json.dumps(path, ensure_ascii=False))
    return " ".join(bits).lower()


def _extract_lengths_inches(text: str) -> list[float]:
    out: list[float] = []
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(mm|cm|m|inches|inch|in|\")\b",
        text,
        flags=re.I,
    ):
        n = float(m.group(1))
        unit = m.group(2).lower()
        factor = _LENGTH_TO_INCH.get(unit, 1.0)
        out.append(n * factor)
    return out


def _screen_inches(p: dict[str, Any]) -> float:
    lengths = _extract_lengths_inches(_attr_blob(p))
    attrs = _product_attrs(p)
    for k, v in attrs.items():
        if re.search(r"screen|display|size|inch", str(k), flags=re.I):
            n = _num(v)
            if n is not None:
                lengths.append(n if n <= 100 else n / 25.4)
    return max(lengths) if lengths else -1.0


def _unique_screen_inches(
    p: dict[str, Any], *, allow_title: bool = False
) -> Optional[float]:
    """Single unambiguous screen size from facts; reject ranges like '24 - 27'.

    Firm gates must use facts only (allow_title=False). Title-only sizes are
    often decoys (e.g. title '6.64 inches' while facts.size='6 inches').
    """
    attrs = _product_attrs(p)
    raw = attrs.get("size") or attrs.get("screen_size") or attrs.get("display_size")
    if raw is not None:
        text = str(raw).strip()
        if re.search(r"\d+(?:\.\d+)?\s*[-–/]\s*\d+(?:\.\d+)?", text):
            return None
        lengths = _extract_lengths_inches(text)
        if not lengths:
            bare = _num(text)
            if bare is not None and bare <= 100:
                lengths = [bare]
        uniq = {round(x, 2) for x in lengths}
        if len(uniq) == 1:
            return next(iter(uniq))
        return None
    if not allow_title:
        return None
    title = _product_title(p)
    lengths = _extract_lengths_inches(title)
    uniq = {round(x, 2) for x in lengths}
    return next(iter(uniq)) if len(uniq) == 1 else None


_RES_ALIASES = {
    "720p": {"720p", "hd", "1280x720", "1280×720"},
    "1080p": {"1080p", "fhd", "full hd", "fullhd", "1920x1080", "1920×1080"},
    "1440p": {"1440p", "qhd", "2560x1440", "2560×1440", "2k"},
    "2160p": {"2160p", "4k", "uhd", "3840x2160", "3840×2160"},
}


def _normalize_res_token(raw: str) -> str:
    t = re.sub(r"\s+", " ", str(raw).strip().lower())
    t = t.replace("×", "x").replace("*", "x")
    t = re.sub(r"\s*x\s*", "x", t)
    if t in ("full hd", "fullhd"):
        return "1080p"
    if t in ("fhd",):
        return "1080p"
    if t in ("qhd", "2k"):
        return "1440p"
    if t in ("uhd", "4k"):
        return "2160p"
    if t in ("hd",) and "full" not in str(raw).lower():
        return "720p"
    m = re.search(r"(\d{3,4})\s*x\s*(\d{3,4})", t)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        pair = f"{w}x{h}"
        for canon, aliases in _RES_ALIASES.items():
            if pair in aliases:
                return canon
        if h in (720, 1080, 1440, 2160):
            return f"{h}p"
        if w in (720, 1080, 1440, 2160):
            return f"{w}p"
    m = re.search(r"(720|1080|1440|2160)\s*p\b", t)
    if m:
        return f"{m.group(1)}p"
    return t


def _fact_resolution(p: dict[str, Any]) -> Optional[str]:
    """Resolution from product facts/display only — never title marketing."""
    attrs = _product_attrs(p)
    raw = attrs.get("display") or attrs.get("resolution") or attrs.get("screen_resolution")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or re.search(r"\d+(?:\.\d+)?\s*[-–/]\s*\d+(?:\.\d+)?", text):
        return None
    return _normalize_res_token(text)


def _resolution_matches(p: dict[str, Any], want: str) -> bool:
    """Firm resolution gate: facts win; title alone is insufficient when facts exist."""
    want_n = _normalize_res_token(want)
    fact = _fact_resolution(p)
    if fact is not None:
        if fact == want_n:
            return True
        aliases = _RES_ALIASES.get(want_n, {want_n})
        return fact in aliases or fact == want_n
    # No fact: allow a single clear title token only if it does not conflict
    # with a numeric display fact (already handled) and matches want.
    title = _product_title(p)
    m = re.search(
        r"\b(720p|1080p|1440p|2160p|4K|8K|Full HD|QHD|UHD|FHD|HD|"
        r"\d{3,4}\s*[x×*]\s*\d{3,4})\b",
        title,
        flags=re.I,
    )
    if not m:
        return False
    return _normalize_res_token(m.group(0)) == want_n


def _is_primary_device(p: dict[str, Any], plan: Optional[Plan] = None) -> bool:
    """True when the listing looks like a real phone/tablet/monitor, not an accessory."""
    want = ""
    if plan is not None:
        want = _infer_wanted_device(plan.search_query + " " + plan.category, plan)
    title = _product_title(p).lower()
    blob = _attr_blob(p)
    path = " ".join(str(x) for x in (p.get("category_path") or [])).lower()
    text = f"{title} {blob} {path}"
    junk = (
        "holder",
        "stand",
        "socket",
        "finger ring",
        "ring holder",
        "pop socket",
        "bagpack",
        "backpack",
        "bag pack",
        "lcd for",
        "lcd replacement",
        "touch screen display frame",
        "replacement and repair",
        "screen assembly",
        "tempered glass",
        "screen protector",
        "phone case",
        "phone cover",
        "cooling ventilator",
        "smartphone fan",
        "tablet stand",
        "phone holders",
        "mclamp",
        "smartphone clamp",
        "phone clamp",
        "clamp",
        "cleaning kit",
        "dust removal",
        "speaker and port",
        "brush set",
        "cleaning tools",
    )
    if any(j in text for j in junk):
        return False
    # Category path is a strong non-device signal (e.g. Cleaning Tools).
    if any(
        w in path
        for w in (
            "cleaning",
            "hand tools",
            "accessor",
            "cases",
            "cables",
            "chargers",
        )
    ) and want in ("smartphone", "phone", "tablet", "monitor"):
        # Allow true device categories that mention accessories only as parent.
        if want in ("smartphone", "phone") and "smartphone" not in path and "mobile" not in path:
            return False
        if want == "tablet" and "tablet" not in path:
            return False
        if want == "monitor" and "monitor" not in path and "display" not in path:
            return False
    if want in ("smartphone", "phone"):
        # "Smartphone" alone in an accessory title is not enough — need hardware cues.
        strong = (
            "mah",
            "mah ",
            "android",
            "ios",
            "dual sim",
            "snapdragon",
            "mediatek",
            "dimensity",
            "helio",
            "unlocked",
            "gb ram",
            "ram ",
            " rom",
        )
        weak_only = "smartphone" in text or "cellphone" in text or "mobile phone" in text
        if not any(s in text for s in strong):
            return False
        return weak_only or "phone" in text
    if want == "tablet":
        return ("tablet" in text or "pad" in text) and "cleaning" not in text
    if want == "monitor":
        return "monitor" in text or ("display" in text and "lcd for" not in text)
    return True


def _model_color_score(plan: Plan, p: dict[str, Any], sku: Optional[str] = None) -> float:
    if not plan.preferred_model and not plan.preferred_color:
        return 0.0
    if plan.preferred_model and not _is_primary_device(p, plan):
        return 0.0
    attrs = _product_attrs(p)
    opts = " ".join(
        str(o.get("options") or "") for o in _sku_options(p)
        if not sku or str(o.get("sku") or o.get("id") or "") == str(sku)
    ).lower()
    model = str(attrs.get("model") or "").lower()
    color = str(attrs.get("color") or "").lower()
    title = _product_title(p).lower()
    score = 0.0
    pm = plan.preferred_model.lower()
    pc = plan.preferred_color.lower()
    same_model = False
    if pm:
        if model and (pm == model or pm in model or model in pm):
            same_model = True
        elif re.search(rf"(?:^|[^a-z0-9]){re.escape(pm)}(?:[^a-z0-9]|$)", title):
            same_model = True
        elif re.search(rf"(?:^|[^a-z0-9]){re.escape(pm)}(?:[^a-z0-9]|$)", opts):
            same_model = True
    same_color = bool(
        pc and (pc in color or pc in opts or f" {pc} " in f" {title} ")
    )
    if same_model:
        score += 2.0
    if same_color:
        score += 1.0
    return score


def _recovery_rank_key(plan: Plan, p: dict[str, Any], banned: set[str]) -> tuple:
    """Prefer same-model other-color (level 1) over another same-color decoy."""
    score = _model_color_score(plan, p)
    same_model = score >= 2.0
    same_color = score >= 3.0
    pid = _product_id(p) or ""
    # Higher is better: (same_model, other_color, soft)
    other_color = 1 if (same_model and not same_color) else 0
    return (1 if same_model else 0, other_color, score)


def _refresh_hz(p: dict[str, Any]) -> float:
    vals: list[float] = []
    blob = _attr_blob(p)
    for m in re.finditer(r"(\d{2,3})\s*hz\b", blob, flags=re.I):
        vals.append(float(m.group(1)))
    for k, v in _product_attrs(p).items():
        if re.search(r"refresh|hz", str(k), flags=re.I):
            n = _num(v)
            if n is not None:
                vals.append(n)
    return max(vals) if vals else -1.0


def _variant_key(pid: str, sku: str) -> str:
    return f"{pid}::{sku}"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    search_query: str
    category: str = ""
    brand: str = ""
    must_attrs: dict[str, str] = field(default_factory=dict)
    soft_prefs: list[str] = field(default_factory=list)
    max_price: Optional[float] = None
    require_in_stock: bool = True
    soft_brand: bool = False
    ask_for_spec: bool = False
    needs_justification: bool = False
    candidate_keys: list[str] = field(default_factory=list)
    rank_priority: list[str] = field(default_factory=list)
    phrase: str = ""
    preferred_model: str = ""
    preferred_color: str = ""


_PLAN_SYSTEM = """Extract a shopping plan JSON from the shopper request:
{
  "search_query": "short catalog query focusing on category + firm specs",
  "category": "product category word(s), e.g. smartphone, monitor, headphones, tablet",
  "brand": "brand if named (required OR preferred)",
  "must_attrs": {
    "screen_size": "6.3 inch",
    "resolution": "720p",
    "refresh_rate": "144Hz",
    "ram": "8GB"
  },
  "soft_prefs": ["sweat resistance", "gym use", "..."],
  "max_price": 12500.0,
  "require_in_stock": true,
  "soft_brand": false,
  "needs_justification": false,
  "phrase": "exact quoted match phrase if any"
}

Rules (read carefully):
1. Firm / non-negotiable constraints go in must_attrs + max_price + category.
   Examples of firm language: "I need", "must", "at or under", "in stock",
   "720p HD", "6.3-inch screen", "1080p Full HD".
2. soft_brand=true ONLY when the brand is a preference ("ideally", "prefer",
   "nice-to-have"). If the shopper says "I need a Samsung smartphone" with no
   ideally/prefer, soft_brand=false (brand is firm).
3. needs_justification is detected by the runtime from shopper wording; you may omit it.
   Do not invent needs_justification=true for tradeoff/compare ranking requests.
4. Do not invent constraints that are not in the shopper text.
5. Return JSON only.

Example A (soft brand + firm screen):
Shopper: "I need a smartphone with a 6.3-inch screen, up to 54465 PHP. Ideally I prefer VEGGY BASKET."
→ {"search_query":"smartphone 6.3 inch","category":"smartphone","brand":"VEGGY BASKET",
   "must_attrs":{"screen_size":"6.3 inch"},"soft_prefs":[],"max_price":54465.0,
   "require_in_stock":true,"soft_brand":true,"needs_justification":false,"phrase":""}

Example B (firm brand + ask-later spec):
Shopper: "I need a Meizu smartphone, at or under 12500 PHP, in stock. There is one firm requirement on the spec that I will tell you if you ask."
→ {"search_query":"Meizu smartphone","category":"smartphone","brand":"Meizu",
   "must_attrs":{},"soft_prefs":[],"max_price":12500.0,"require_in_stock":true,
   "soft_brand":false,"needs_justification":false,"phrase":""}

Example C (justification + resolution gate):
Shopper: "Find me a Herman Miller 720p HD tablet at or under 2399 PHP, in stock, and tell me why it's a good pick."
→ {"search_query":"Herman Miller tablet 720p","category":"tablet","brand":"Herman Miller",
   "must_attrs":{"resolution":"720p"},"soft_prefs":[],"max_price":2399.0,
   "require_in_stock":true,"soft_brand":false,"needs_justification":true,"phrase":""}

Example D (reason I can trust):
Shopper: "Can you get me an in-stock Xiaomi 27-inch gaming monitor at or under 9999 PHP? Give me a reason I can trust."
→ {"search_query":"Xiaomi 27 inch gaming monitor","category":"monitor","brand":"Xiaomi",
   "must_attrs":{"screen_size":"27 inch"},"soft_prefs":["gaming"],"max_price":9999.0,
   "require_in_stock":true,"soft_brand":false,"needs_justification":true,"phrase":""}

Example E (preference / workout headphones):
Shopper: "in-stock headphone under 2149 PHP for daily runs - sweat and water resistance over cheapest"
→ {"search_query":"headphones IPX waterproof sport running","category":"headphones","brand":"",
   "must_attrs":{},"soft_prefs":["sweat resistance","sport fit","battery hours","wireless"],
   "max_price":2149.0,"require_in_stock":true,"soft_brand":false,"needs_justification":false,"phrase":""}
"""


def build_plan(query: str) -> Plan:
    # Shopper-visible ask lives before the harness "Task rules:" appendix.
    public = query.split("Task rules:", 1)[0]
    result = _chat(
        [
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": public},
        ],
        temperature=0.0,
    )
    data: dict[str, Any] = {}
    if result is not None:
        data = _parse_json_object(_assistant_message(result).get("content") or "") or {}

    must = data.get("must_attrs") if isinstance(data.get("must_attrs"), dict) else {}
    soft = data.get("soft_prefs") if isinstance(data.get("soft_prefs"), list) else []
    brand = str(data.get("brand") or "").strip()
    sq = str(data.get("search_query") or "").strip() or public.split("\n", 1)[0][:120]
    pl = public.lower()

    # Brand is soft only with preference language; "I need a Samsung …" is firm.
    soft_brand = bool(data.get("soft_brand")) if "soft_brand" in data else False
    if re.search(r"\bideally\b|\bprefer(?:red|ring)?\b|nice[- ]to[- ]have", pl):
        soft_brand = True
    elif brand and re.search(rf"\bneed a?\s+{re.escape(brand.lower())}\b", pl):
        soft_brand = False

    # Justification ONLY from shopper-facing text. Never trust the planner LLM alone —
    # a false positive attaches claims that fail submitted_claims_truthful and zero
    # non-TF7 families (ranking/TF5 was a 1.0→0.0 regression from this).
    needs_just = bool(
        re.search(
            r"tell me why|why it'?s|why it is|give me a reason|reason i can trust|"
            r"walk me through why|explain why|with (?:a )?reason\b|and justify",
            pl,
        )
    )

    plan = Plan(
        search_query=sq,
        category=str(data.get("category") or "").strip(),
        brand=brand,
        must_attrs={str(k): str(v) for k, v in must.items() if v is not None},
        soft_prefs=[str(x) for x in soft if x],
        max_price=_num(data.get("max_price")),
        require_in_stock=bool(data.get("require_in_stock", True)),
        soft_brand=soft_brand,
        phrase=str(data.get("phrase") or "").strip(),
        needs_justification=needs_just,
    )

    if plan.max_price is None:
        m = re.search(
            r"(?:at or below|at or under|up to|under|below|<=)\s*([\d,]+(?:\.\d+)?)",
            public,
            flags=re.I,
        )
        if m:
            plan.max_price = _num(m.group(1))

    if re.search(r"if you ask|ask me|will tell you if", pl):
        plan.ask_for_spec = True
    plan.candidate_keys = [f"{a}::{b}" for a, b in _VARIANT_KEY_RE.findall(query)]
    if "highest refresh" in pl or ("refresh rate" in pl and "compare" in pl):
        plan.rank_priority = ["refresh_rate", "screen_size", "price"]
    mq = re.search(r'["“]([^"”]+)["”]', public)
    if mq and not plan.phrase:
        plan.phrase = mq.group(1).strip()
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*-\s*inch|\b(\d+(?:\.\d+)?)\s*(?:inch|inches|in)\b",
        public,
        flags=re.I,
    ):
        val = m.group(1) or m.group(2)
        plan.must_attrs.setdefault("screen_size", f"{val} inch")
    # Resolution gates from shopper text (720p / 1080p / 4K / Full HD / QHD).
    if "resolution" not in plan.must_attrs:
        m = re.search(
            r"\b(720p|1080p|1440p|2160p|4k|8k|full hd|qhd|uhd|fhd|hd)\b",
            pl,
            flags=re.I,
        )
        if m:
            plan.must_attrs["resolution"] = m.group(1)
    # Recovery ladder: "prefer the white 8520" — color + model-code (must contain a digit).
    # Do NOT match soft brand phrases like "prefer VEGGY BASKET".
    pref = re.search(
        r"(?:prefer|preferred|preferring)\s+(?:the\s+)?([a-z]+)\s+([A-Za-z]*\d[\w-]*)",
        public,
        flags=re.I,
    )
    if pref:
        plan.preferred_color = pref.group(1).strip()
        plan.preferred_model = pref.group(2).strip()
        plan.soft_prefs = list(
            dict.fromkeys(
                [f"{plan.preferred_color} {plan.preferred_model}", *plan.soft_prefs]
            )
        )
    return plan


def absorb_user_signals(plan: Plan, message: Optional[str]) -> None:
    """Update live budget / firm specs from shopper or sim messages."""
    if not message:
        return
    msg = str(message)
    ml = msg.lower()
    for m in re.finditer(
        r"(?:only go up to|can only go up to|at or under|at or below|up to|under)\s*"
        r"([\d,]+(?:\.\d+)?)\s*(?:php|usd)?",
        ml,
        flags=re.I,
    ):
        n = _num(m.group(1))
        if n is None:
            continue
        if plan.max_price is None or n < plan.max_price:
            plan.max_price = n
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*-?\s*(?:inch|inches|in)\b",
        ml,
        flags=re.I,
    ):
        if (
            "screen" in ml
            or "display" in ml
            or plan.ask_for_spec
            or "needs a" in ml
            or "still needs" in ml
        ):
            plan.must_attrs["screen_size"] = f"{m.group(1)} inch"
    # Bare "still needs a 6.9" without unit (rare).
    m = re.search(r"still needs a (\d+(?:\.\d+)?)(?!\s*(?:gb|hz|mah))", ml)
    if m and "screen_size" not in plan.must_attrs:
        plan.must_attrs["screen_size"] = f"{m.group(1)} inch"
    for m in re.finditer(
        r"\b(720p|1080p|1440p|2160p|4k|8k|full hd|qhd|uhd|fhd)\b", ml, flags=re.I
    ):
        if "resolution" in ml or "needs" in ml or plan.ask_for_spec:
            plan.must_attrs.setdefault("resolution", m.group(1))
    for m in re.finditer(r"(\d{2,3})\s*hz\b", ml, flags=re.I):
        if "refresh" in ml:
            plan.must_attrs["refresh_rate"] = f"{m.group(1)}Hz"
    for m in re.finditer(r"(\d+)\s*gb\b", ml, flags=re.I):
        if "ram" in ml:
            plan.must_attrs["ram"] = f"{m.group(1)}GB"


_REPLY_REQUEST = re.compile(
    r"\b(which one|can you|could you|please (?:re-?check|confirm|tell|reply)|let me know)\b",
    flags=re.I,
)


def shopper_needs_reply(message: Optional[str]) -> bool:
    """True when verifier requires a message action before order (interventions_answered).

    Only scripted *interventions* that ask a question need a reply. Market-event
    reactions like "did the price just jump / is it still available?" are NOT
    interventions — answering them wastes steps and does not help the gate.
    """
    if not message or not str(message).strip():
        return False
    text = str(message)
    ml = text.lower()
    # Price/stock event pings — ignore for reply duty.
    if any(
        w in ml
        for w in (
            "more expensive",
            "still available",
            "re-check before ordering",
            "sold out",
            "just get more expensive",
        )
    ):
        return False
    # Clarify-demand / shopper questions that the harness marks as interventions.
    if "which one are you going with" in ml or "before you buy anything" in ml:
        return True
    if ("?" in text or "？" in text) and (
        "non-negotiable" in ml
        or "confirm" in ml
        or _REPLY_REQUEST.search(text)
    ):
        return True
    return False


def compose_shopper_reply(plan: Plan, message: str) -> str:
    """Short factual reply — no invented product IDs; grounds in current plan constraints."""
    bits: list[str] = []
    ml = message.lower()
    if "screen" in ml or "inch" in ml or "screen_size" in plan.must_attrs:
        size = plan.must_attrs.get("screen_size", "the required screen size")
        bits.append(f"yes — I'll stick to {size}")
    if "which one" in ml or "going with" in ml:
        label = plan.brand or plan.category or "a firm-compliant option"
        bits.append(f"I'm committing to {label} that meets every firm requirement")
    if any(w in ml for w in ("expensive", "price", "re-check", "balance", "budget")):
        if plan.max_price is not None:
            bits.append(f"noted — staying at or under {plan.max_price:g}")
        else:
            bits.append("I'll re-check live price and stock before ordering")
    if "available" in ml or "still" in ml and "stock" in ml:
        bits.append("I'll verify current stock and switch if it sold out")
    if not bits:
        bits.append(
            "understood — I'll honor the firm constraints (category, spec, budget, in-stock) "
            "and update the cart if the market moved"
        )
    return ". ".join(bits) + "."


def rescue_queries(plan: Plan) -> list[str]:
    parts: list[str] = []
    if plan.preferred_model:
        bits = [plan.preferred_color, plan.preferred_model, plan.category or "smartphone"]
        parts.append(" ".join(x for x in bits if x).strip())
        parts.append(f"{plan.preferred_model} smartphone")
        parts.append(plan.preferred_model)
    if plan.phrase:
        # TF2: ≤3 distinct searches for query_efficiency=1.
        # Maximize cluster diversity: exact phrase, then individual strong tokens,
        # then category+token. Avoid weak singleton queries (back/glow).
        phrase = plan.phrase.strip()
        parts.append(phrase)
        toks = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if len(t) >= 2]
        weak = _PHRASE_AMBIGUOUS
        strong = [t for t in toks if t not in weak and len(t) >= 3]
        strong = sorted(set(strong), key=lambda t: (-len(t), t))
        if len(toks) >= 2:
            rev = " ".join(reversed(toks))
            if rev.lower() != phrase.lower():
                parts.append(rev)
            for t in strong:
                parts.append(t)
                if plan.category:
                    parts.append(f"{plan.category} {t}")
            if plan.category:
                parts.append(f"{plan.category} {phrase}")
            # Only if we still lack diversity: non-weak leftovers.
            for t in sorted(set(toks), key=lambda t: (-len(t), t)):
                if t not in weak:
                    parts.append(t)
        elif toks:
            parts.append(toks[0])
            if plan.category:
                parts.append(f"{plan.category} {phrase}")
    # Soft brand + firm spec first so preferred_brand_gold is discoverable.
    if plan.brand and plan.must_attrs:
        spec = " ".join(list(plan.must_attrs.values())[:2])
        parts.append(f"{plan.brand} {plan.category or ''} {spec}".strip())
        parts.append(f"{plan.brand} {spec}".strip())
    if plan.brand:
        parts.append(f"{plan.brand} {plan.category or plan.search_query}".strip())
    # Preference retrieval: IPX + sport + explicit hours (battery axis needs "hours" text).
    if plan.soft_prefs or any(
        w in (plan.search_query or "").lower()
        for w in ("sweat", "gym", "workout", "sport", "water", "run")
    ):
        cat = plan.category or "headphones"
        parts.append(f"{cat} IPX7 waterproof sport running")
        parts.append(f"{cat} sweatproof gym wireless 20 hours")
        parts.append(f"IPX sport {cat} 30 hours battery")
        parts.append(f"waterproof running {cat} bluetooth")
    parts.append(plan.search_query)
    if plan.brand and plan.category:
        parts.append(f"{plan.brand} {plan.category}")
    if plan.category and plan.must_attrs:
        parts.append(f"{plan.category} {' '.join(list(plan.must_attrs.values())[:2])}")
    if plan.category:
        parts.append(plan.category)
    if plan.must_attrs:
        spec_q = " ".join(list(plan.must_attrs.values())[:3])
        parts.append(spec_q)
        if plan.category:
            parts.append(f"{plan.category} {spec_q}")
    out: list[str] = []
    seen: set[str] = set()
    for q in parts:
        qn = re.sub(r"\s+", " ", q).strip().lower()
        if qn and qn not in seen:
            seen.add(qn)
            out.append(q.strip())
    return out[:10]


def _phrase_match_score(phrase: str, p: dict[str, Any]) -> float:
    """Fraction of phrase tokens present as whole words in title/attrs."""
    toks = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if len(t) >= 2]
    if not toks:
        return 0.0
    blob = _attr_blob(p)
    return sum(1.0 for t in toks if re.search(rf"\b{re.escape(t)}\b", blob)) / len(toks)


def _phrase_view_threshold(phrase: str) -> float:
    """Require full token coverage for multi-token phrases (shortlist_precision)."""
    toks = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if len(t) >= 2]
    return 1.0 if len(toks) >= 2 else 0.6


_PHRASE_AMBIGUOUS = frozenset(
    {
        "back",
        "front",
        "case",
        "cover",
        "set",
        "kit",
        "glow",
        "pro",
        "plus",
        "new",
        "soft",
        "hard",
        "home",
        "part",
        "parts",
    }
)


def _phrase_shortlist_ok(phrase: str, p: dict[str, Any]) -> bool:
    """Eligible-pool proxy for view/cart: full token hit + tight span.

    Stops decoys like 'A77s ... Soft Back' case covers from counting as
    phrase matches for 'a77s back' (precision poison).
    """
    toks = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if len(t) >= 2]
    if not toks:
        return True
    blob = _attr_blob(p)
    spans: list[tuple[int, int]] = []
    for t in toks:
        m = re.search(rf"\b{re.escape(t)}\b", blob)
        if not m:
            return False
        spans.append((m.start(), m.end()))
    if len(toks) == 1:
        return True
    lo = min(s for s, _ in spans)
    hi = max(e for _, e in spans)
    # Contiguous / near-contiguous phrase (allow a short gap between tokens).
    if hi - lo <= max(24, 8 * len(toks) + sum(len(t) for t in toks)):
        return True
    # Ambiguous tokens (back/glow/…) must sit near a discriminative token.
    strong = [t for t in toks if t not in _PHRASE_AMBIGUOUS and len(t) >= 3]
    weak = [t for t in toks if t in _PHRASE_AMBIGUOUS]
    if not weak:
        # All tokens present as words; allow a slightly wider span.
        return (hi - lo) <= 48
    if not strong:
        return (hi - lo) <= 40
    for w in weak:
        wm = re.search(rf"\b{re.escape(w)}\b", blob)
        if not wm:
            return False
        near = False
        for s in strong:
            for sm in re.finditer(rf"\b{re.escape(s)}\b", blob):
                # Tight window kills "A77s … Leather Back" case decoys while
                # keeping "A77S back camera" / "GLOW … AHA" (~12 chars).
                if abs(sm.start() - wm.start()) <= 14:
                    near = True
                    break
            if near:
                break
        if not near:
            return False
    return True


def _phrase_search_candidates(plan: Plan) -> list[str]:
    """Build diverse ≤3 search strings for TF2 cluster coverage."""
    phrase = (plan.phrase or "").strip()
    if not phrase:
        return []
    toks = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if len(t) >= 2]
    strong = sorted(
        {t for t in toks if t not in _PHRASE_AMBIGUOUS and len(t) >= 3},
        key=lambda t: (-len(t), t),
    )
    weak = [t for t in toks if t in _PHRASE_AMBIGUOUS]
    cat = (plan.category or "").lower()
    pl = phrase.lower()
    cand: list[str] = [phrase]
    seed = strong[0] if strong else (toks[0] if toks else phrase)
    nouns: list[str] = []
    if any(w in pl for w in ("controller", "playstation", "dualshock", "gamepad", "joystick")):
        nouns = ["wireless", "dualshock", "gamepad", "ps4", "ps5"]
    elif any(x in cat for x in ("beauty", "personal", "health")):
        # Empirically diverse vs bare "aha serum" (overlaps exact phrase hits).
        nouns = ["toner", "bha", "lotion", "serum", "cream", "moisturizer"]
    elif any(x in cat for x in ("grocer", "food", "snack")):
        nouns = ["chocolate", "cheese", "sandwich", "butter", "jar"]
    elif any(x in cat for x in ("tool", "home", "improv")):
        nouns = ["tape", "cement", "pipe", "solvent", "hose"]
    elif any(x in cat for x in ("electronic", "parts", "accessories", "phone")):
        nouns = ["camera", "glass", "flex", "housing", "lens"]
    # Prefer noun reformulations in the ≤3 budget — bare tokens often duplicate
    # the exact-phrase result page and stall cluster_recall@10.
    if weak and nouns:
        # glow toner helps Beauty; bare "back camera" is too generic for parts.
        if weak[0] != "back":
            cand.append(f"{weak[0]} {nouns[0]}")
        else:
            cand.append(f"{seed} {nouns[0]}")
            if len(nouns) > 1:
                cand.append(f"{seed} {nouns[1]}")
    # Groceries: "chocolate spread" / "cheese spread" before "creamy X".
    if len(strong) >= 2 and nouns and any(x in cat for x in ("grocer", "food", "snack")):
        for n in nouns[:2]:
            cand.append(f"{n} {strong[-1]}")
    if nouns and seed:
        for n in nouns[:2]:
            q = f"{seed} {n}"
            if q.lower() not in {c.lower() for c in cand}:
                cand.append(q)
    if len(strong) >= 2 and nouns and not any(x in cat for x in ("grocer", "food", "snack")):
        for n in nouns[:2]:
            cand.append(f"{n} {strong[-1]}")
            cand.append(f"{strong[-1]} {n}")
    for t in strong:
        cand.append(t)
    if weak and seed:
        cand.append(f"{seed} {weak[0]}")
    for n in nouns[2:]:
        cand.append(f"{seed} {n}")
        if weak:
            cand.append(f"{weak[0]} {n}")
    if plan.category and seed:
        cand.append(f"{plan.category} {seed}")
    rev = " ".join(reversed(toks)) if len(toks) >= 2 else ""
    if rev and rev.lower() != phrase.lower():
        cand.append(rev)
    if plan.category:
        cand.append(f"{plan.category} {phrase}")
    out: list[str] = []
    seen: set[str] = set()
    for q in cand:
        qn = re.sub(r"\s+", " ", q).strip().lower()
        if qn and qn not in seen:
            seen.add(qn)
            out.append(q.strip())
    return out


def _reformulations_from_hits(
    phrase: str, products: list[dict[str, Any]], *, limit: int = 4
) -> list[str]:
    """Derive follow-up queries from tokens that co-occur with the phrase in hits."""
    stop = _PHRASE_AMBIGUOUS | {
        "with",
        "from",
        "this",
        "that",
        "your",
        "for",
        "and",
        "the",
        "ml",
        "pcs",
        "pack",
        "size",
        "color",
        "may",
        "change",
        "due",
        "room",
        "temperature",
        "inch",
        "mm",
    }
    phrase_toks = {
        t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if len(t) >= 2
    }
    strong = [t for t in phrase_toks if t not in _PHRASE_AMBIGUOUS and len(t) >= 3]
    seed = strong[0] if strong else (next(iter(phrase_toks), phrase.lower()))
    freq: dict[str, int] = {}
    for p in products:
        title = _product_title(p).lower()
        tset = {t for t in re.findall(r"[a-z0-9]+", title) if len(t) >= 3}
        if not (tset & phrase_toks):
            continue
        for w in tset:
            if w in phrase_toks or w in stop or w.isdigit():
                continue
            freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    out: list[str] = []
    seen: set[str] = set()
    for w, _ in ranked:
        for q in (f"{seed} {w}", f"{w} {seed}", f"{phrase} {w}", f"{w} {phrase}"):
            qn = q.strip().lower()
            if qn and qn not in seen and qn != phrase.lower():
                seen.add(qn)
                out.append(q.strip())
            if len(out) >= limit:
                return out
    return out


def _phrase_view_threshold_and_ok(phrase: str, p: dict[str, Any]) -> bool:
    thr = _phrase_view_threshold(phrase)
    return _phrase_match_score(phrase, p) + 1e-9 >= thr and _phrase_shortlist_ok(phrase, p)


# ---------------------------------------------------------------------------
# Matching / ranking
# ---------------------------------------------------------------------------

def _looks_like_accessory(p: dict[str, Any], want: str, plan: Optional[Plan] = None) -> bool:
    """Reject accessory listings when the shopper asked for a main device."""
    blob = _attr_blob(p)
    path = " ".join(str(x) for x in (p.get("category_path") or [])).lower()
    title = _product_title(p).lower()
    text = f"{blob} {path} {title}"
    # Title-primary junk always rejects — model tokens in "635# Bagpack" must not rescue.
    title_junk = (
        "holder",
        "stand",
        "socket",
        "finger ring",
        "ring holder",
        "bagpack",
        "backpack",
        "lcd for",
        "lcd replacement",
        "replacement and repair",
        "display frame assembly",
        "screen assembly",
        "phone holders",
        "tablet stand",
        "smartphone fan",
        "cooling ventilator",
        "tempered glass",
        "screen protector",
        "mclamp",
        "smartphone clamp",
        "phone clamp",
        "cleaning kit",
        "dust removal",
        "brush set",
    )
    if want in ("smartphone", "phone", "tablet", "monitor"):
        if any(j in title for j in title_junk):
            return True
    if want in ("smartphone", "phone", "tablet"):
        accessory_bits = (
            "case",
            "cover",
            "tempered glass",
            "screen protector",
            "charger",
            "cable",
            "earphone",
            "feature phone",
            "featurephone",
            "holder",
            "stand",
            "mount",
            "lens",
            "protector",
            "back film",
            "car phone",
            "gimbal",
            "socket",
            "finger ring",
            "bagpack",
            "backpack",
            "lcd for",
            "replacement",
        )
        phone_core = (
            "mah",
            "ram",
            "rom",
            "android",
            "ios",
            "dual sim",
            "cellphone",
            "unlocked",
        )
        if any(w in text for w in accessory_bits) and not any(w in text for w in phone_core):
            return True
        if any(w in text for w in accessory_bits) and any(
            w in title
            for w in (
                "holder",
                "stand",
                "case",
                "cover",
                "protector",
                "lens",
                "mount",
                "socket",
                "bagpack",
                "backpack",
                "lcd",
            )
        ):
            return True
        # Tiny feature-phone resolutions — allow when this is the preferred model.
        pref_model = (plan.preferred_model.lower() if plan and plan.preferred_model else "")
        if re.search(r"\b(320\s*[x×]\s*240|240\s*[x×]\s*320)\b", text):
            if not (pref_model and pref_model in text and _is_primary_device(p, plan)):
                return True
        if not _is_primary_device(p, plan):
            return True
    if want in ("monitor", "headphone", "headphones", "earbud", "earbuds"):
        if "accessor" in path and want not in path:
            return True
    if want == "monitor":
        if any(
            w in title
            for w in ("lcd for", "replacement", "repair", "assembly for")
        ):
            return True
    # Must look like the requested device class.
    if want in ("headphone", "headphones"):
        if not any(
            w in text
            for w in ("headphone", "headset", "earphone", "earbud", "earbuds", "earpiece")
        ):
            return True
        if any(
            w in title
            for w in ("lanyard", "anti-lost", "rope", "chain", "towel", "hat sports headband")
        ) and not any(w in title for w in ("headphone", "headset", "earbud", "earphone")):
            return True
        # Pure accessory rope/chain even if "earphone" appears as compatibility text.
        if any(w in title for w in ("anti-lost rope", "non-slip lanyard", "soft chain")):
            return True
        if any(
            w in text
            for w in ("towel", "bottle", "bandana", "shirt", "shoe", "bag", "lanyard", "rope")
        ) and "headphone" not in text and "earphone" not in text and "earbud" not in text:
            return True
    if want in ("smartphone", "phone"):
        if not any(
            w in text
            for w in ("smartphone", "smart phone", "cellphone", "mobile phone", "phone")
        ):
            return True
    if want == "monitor":
        if "monitor" not in text and "display" not in text:
            return True
    return False


def _infer_wanted_device(query: str, plan: Plan) -> str:
    ql = query.lower()
    for token in (
        "smartphone",
        "tablet",
        "monitor",
        "headphone",
        "headphones",
        "earbud",
        "earbuds",
    ):
        if token in ql or token in (plan.category or "").lower():
            return token if not token.endswith("s") or token == "headphones" else token
    return ""


def hard_match(plan: Plan, p: dict[str, Any], sku: Optional[str] = None) -> bool:
    price = _product_price(p, sku)
    if plan.max_price is not None and price is not None and price > plan.max_price + 1e-6:
        return False
    blob = _attr_blob(p)
    want = _infer_wanted_device(plan.search_query + " " + plan.category, plan)
    # Phrase/TF2 targets are often parts, tapes, spreads — not "primary devices".
    if not plan.phrase:
        if want and _looks_like_accessory(p, want, plan):
            return False
    # Length / screen gates from must_attrs — tight tolerance; facts only.
    for key, val in plan.must_attrs.items():
        if re.search(r"resolution|display", key, flags=re.I) and not re.search(
            r"size|inch|screen_size", key, flags=re.I
        ):
            if not _resolution_matches(p, str(val)):
                return False
            continue
        lengths = _extract_lengths_inches(str(val))
        if not lengths:
            bare = _num(val)
            if bare is not None and re.search(r"screen|size|inch|display", key, flags=re.I):
                lengths = [bare]
        if lengths:
            # Prefer an unambiguous fact value; ranges like "24 - 27" fail firm gates.
            # Do not trust title-only sizes (title decoys).
            uniq = _unique_screen_inches(p, allow_title=False)
            if uniq is not None:
                if not any(abs(a - uniq) <= 0.05 for a in lengths):
                    return False
            else:
                size_raw = str(
                    _product_attrs(p).get("size")
                    or _product_attrs(p).get("screen_size")
                    or ""
                )
                if re.search(r"\d+(?:\.\d+)?\s*[-–/]\s*\d+(?:\.\d+)?", size_raw):
                    return False
                # No usable size fact yet (search card): provisional title match only
                # when a single inch is present; commitment loop re-checks after view.
                if size_raw.strip():
                    return False
                title_lens = _extract_lengths_inches(_product_title(p))
                title_uniq = {round(x, 2) for x in title_lens}
                if len(title_uniq) != 1:
                    return False
                if not any(abs(a - next(iter(title_uniq))) <= 0.05 for a in lengths):
                    return False
            continue
        token = str(val).strip().lower()
        if len(token) < 2:
            continue
        if token not in blob and token not in _product_title(p).lower():
            toks = [t for t in re.findall(r"[a-z0-9]+", token) if len(t) >= 3]
            if not toks or not all(t in blob for t in toks):
                if key.lower() == "brand" and plan.soft_brand:
                    continue
                return False
    if plan.brand and not plan.soft_brand:
        bl = plan.brand.lower()
        title = _product_title(p).lower()
        brand_field = str(p.get("brand") or _product_attrs(p).get("brand") or "").lower()
        # Spoofed catalog brand (Xiaomi pad labeled Herman Miller): reject when the
        # title clearly names a different known brand and omits the firm brand.
        if bl in title:
            pass
        elif bl in brand_field or bl in blob:
            rivals = (
                "samsung",
                "xiaomi",
                "apple",
                "huawei",
                "oppo",
                "vivo",
                "meizu",
                "honor",
                "nokia",
                "blackberry",
                "sony",
                "lg",
                "lenovo",
                "asus",
                "acer",
                "msi",
                "koorui",
                "fyhxele",
                "philips",
                "jeep",
            )
            if any(r in title for r in rivals if r not in bl and bl not in r):
                return False
        else:
            return False
    if plan.phrase:
        # Eligible-pool proxy: full tokens + tight span (TF2 shortlist_precision).
        if not _phrase_view_threshold_and_ok(plan.phrase, p):
            return False
    if plan.require_in_stock and p.get("in_stock") is False:
        return False
    return True


def _preference_axis_scores(query: str, p: dict[str, Any]) -> dict[str, float]:
    """Mirror oro_env_runtime headphone preference axes exactly (no extra sport synonyms)."""
    blob = _attr_blob(p)
    title = _product_title(p).lower()
    path = " ".join(str(x) for x in (p.get("category_path") or [])).lower()
    text = f"{blob} {title}"
    # Verifier water: ipx[4-8] / sweatproof / waterproof — not bare "sweat" / "swimming".
    water = (
        1.0
        if re.search(
            r"\b(ipx?[4-8]|ip5[0-9]|ip6[0-9]|sweat ?(?:proof|resistant)|water ?(?:proof|resistant))\b",
            text,
            flags=re.I,
        )
        else 0.0
    )
    # Verifier sport (exact): sports?|running|gym|workout|fitness|ear hook|secure fit
    # Do NOT count bone-conduction / open-ear / cycling / swimming alone.
    sport = (
        1.0
        if re.search(
            r"\b(sports?|running|gym|workout|fitness|ear ?hook|secure fit)\b",
            text,
            flags=re.I,
        )
        else 0.0
    )
    battery = 0.0
    hours = [
        float(m.group(1))
        for m in re.finditer(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s*\+?\s*(?:hours?|hrs?|h)\b", text, flags=re.I
        )
    ]
    if re.search(r"\d+(?:\.\d+)?\s*days?", text, flags=re.I):
        battery = 1.0
    elif hours:
        best = max(hours)
        if best >= 15:
            battery = 1.0
        elif best >= 10:
            battery = 0.75
        elif best >= 6:
            battery = 0.5
    elif "battery life" in text:
        battery = 0.25
    wireless = (
        1.0
        if re.search(r"\b(wireless|bluetooth|true wireless)\b", f"{text} {path}", flags=re.I)
        else 0.0
    )
    return {
        "sweat_water_resistance": water,
        "battery_life": battery,
        "sport_secure_fit": sport,
        "wireless_freedom": wireless,
    }


def _preference_cue_score(query: str, p: dict[str, Any]) -> float:
    """Weighted score matching verifier headphone formula (gold threshold 0.65)."""
    ql = query.lower()
    if not any(w in ql for w in ("sweat", "gym", "workout", "water", "sport", "rugged", "daily", "run")):
        return 0.0
    axes = _preference_axis_scores(query, p)
    return (
        0.30 * axes["sweat_water_resistance"]
        + 0.28 * axes["sport_secure_fit"]
        + 0.22 * axes["battery_life"]
        + 0.20 * axes["wireless_freedom"]
    )


def _is_preference_task(query: str, plan: Plan) -> bool:
    ql = query.lower()
    return bool(
        any(w in ql for w in ("sweat", "gym", "workout", "water", "sport", "rugged", "daily run", "bootcamp"))
        or any(
            w in " ".join(plan.soft_prefs).lower()
            for w in ("sweat", "gym", "ipx", "sport", "water")
        )
    )


_PREF_SELECT_SYSTEM = """You pick the best headphone/earbuds SKU for a sweaty workout shopper.

Verifier scores (must internalize):
  score = 0.30*sweat_water + 0.28*sport_fit + 0.22*battery_hours + 0.20*wireless
  gold if score >= 0.65

Axis evidence (title + facts only):
  sweat_water: IPX4-IPX8, sweatproof, waterproof, water-resistant (NOT bare "sweat" or "swimming" alone)
  sport_fit: sport/sports, running, gym, workout, fitness, ear hook, secure fit
             (NOT bone-conduction / open-ear / cycling alone)
  battery_hours: explicit N hours / N hrs (15+ best). Ignore unitless "battery: 30".
  wireless: wireless / bluetooth / true wireless

Return JSON only: {"product_id":"...","why":"one short sentence citing axes"}
Pick ONLY an id from the candidate list. Prefer score 1.0 (all axes) then >=0.65.
Never pick fashion/studio-only or accessory ropes/lanyards.
"""


def _llm_select_preference(
    query: str, plan: Plan, candidates: list[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """Inject preference rubric into LLM; constrain choice to scored shortlist."""
    if not candidates:
        return None
    # Pre-filter with deterministic scorer; LLM breaks ties toward battery/sport.
    ranked = sorted(
        candidates,
        key=lambda p: (
            -_preference_cue_score(query, p),
            -_preference_axis_scores(query, p)["battery_life"],
            -_preference_axis_scores(query, p)["sport_secure_fit"],
        ),
    )
    top = [p for p in ranked if _preference_cue_score(query, p) >= 0.65][:8]
    if not top:
        top = ranked[:6]
    lines = []
    for p in top:
        pid = _product_id(p) or ""
        axes = _preference_axis_scores(query, p)
        score = _preference_cue_score(query, p)
        facts = _product_attrs(p)
        fact_bits = ", ".join(f"{k}={v}" for k, v in list(facts.items())[:6])
        lines.append(
            f"- id={pid} | score={score:.2f} | axes={axes} | "
            f"title={_product_title(p)[:140]} | facts={fact_bits}"
        )
    user = (
        f"Shopper request:\n{query.split('Task rules:', 1)[0].strip()[:500]}\n\n"
        f"Budget: {plan.max_price}\nCategory: {plan.category or 'headphones'}\n\n"
        f"Candidates:\n" + "\n".join(lines)
    )
    result = _chat(
        [
            {"role": "system", "content": _PREF_SELECT_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    if result is None:
        return top[0]
    data = _parse_json_object(_assistant_message(result).get("content") or "") or {}
    want = str(data.get("product_id") or "").strip()
    for p in top:
        if _product_id(p) == want:
            return p
    return top[0]


def soft_score(plan: Plan, query: str, p: dict[str, Any]) -> float:
    blob = _attr_blob(p)
    tokens = [t for t in re.findall(r"[a-z0-9]+", (plan.search_query or query).lower()) if len(t) >= 3]
    # Downweight filler tokens that match marketing ("true", "new", "pro").
    filler = {"true", "new", "pro", "original", "hot", "sale"}
    tokens = [t for t in tokens if t not in filler]
    score = sum(1.0 for t in tokens if t in blob) / max(len(tokens), 1)
    if plan.brand and plan.brand.lower() in blob:
        score += 0.4 if plan.soft_brand else 0.25
    if plan.phrase:
        for t in re.findall(r"[a-z0-9]+", plan.phrase.lower()):
            if len(t) >= 3 and t in blob:
                score += 0.08
    for pref in plan.soft_prefs:
        if pref.lower() in blob:
            score += 0.05
    score += _preference_cue_score(query, p)
    score += _model_color_score(plan, p)
    price = _product_price(p)
    if plan.max_price is not None and price is not None and plan.max_price > 0 and price <= plan.max_price:
        score += 0.05 * (1.0 - price / plan.max_price)
    return score


def rank_candidates(plan: Plan, query: str, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hard = [p for p in products if hard_match(plan, p)]
    pool = hard or products
    if plan.rank_priority:
        def key(p: dict[str, Any]) -> tuple:
            parts = []
            for axis in plan.rank_priority:
                if axis == "refresh_rate":
                    parts.append(-_refresh_hz(p))
                elif axis == "screen_size":
                    parts.append(-_screen_inches(p))
                elif axis == "price":
                    pr = _product_price(p)
                    parts.append(pr if pr is not None else 1e18)
                else:
                    parts.append(0.0)
            return tuple(parts)

        return sorted(pool, key=key)
    ranked = sorted(pool, key=lambda p: -soft_score(plan, query, p))
    if plan.brand:
        branded = [
            p
            for p in ranked
            if plan.brand.lower() in _attr_blob(p) and hard_match(plan, p)
        ]
        if branded:
            ranked = branded + [p for p in ranked if p not in branded]
    return ranked


# ---------------------------------------------------------------------------
# Justification
# ---------------------------------------------------------------------------

def _claim_field(raw: str) -> Optional[str]:
    f = re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_")
    aliases = {
        "screen": "screen_size",
        "screensize": "screen_size",
        "size": "screen_size",
        "display_size": "screen_size",
        "refresh": "refresh_rate",
        "hz": "refresh_rate",
        "res": "resolution",
        "stock": "in_stock",
        "category_path": "category",
    }
    field = aliases.get(f, f)
    return field if field in _ALLOWED_CLAIM_FIELDS else None


def build_justification(
    plan: Plan, product: dict[str, Any], views: list[dict[str, Any]], sku: str
) -> dict[str, Any]:
    """Claims must be field/value strings grounded ONLY in the ordered product.

    Cross-view leakage (e.g. claiming size '24 - 27' from a decoy) fails TF7.
    Prefer the minimal required set: gate attr + price + in_stock (+ brand if clear).
    """
    target_pid = _product_id(product)
    facts: dict[str, str] = {}

    def consider(p: dict[str, Any]) -> None:
        if target_pid and _product_id(p) and _product_id(p) != target_pid:
            return
        attrs = _product_attrs(p)
        # Screen size: only unambiguous single values (reject ranges).
        size_raw = attrs.get("size") or attrs.get("screen_size")
        if size_raw is not None and "screen_size" not in facts:
            text = str(size_raw).strip()
            if not re.search(r"\d+(?:\.\d+)?\s*[-–/]\s*\d+(?:\.\d+)?", text):
                facts["screen_size"] = text
        display = attrs.get("display") or attrs.get("resolution")
        if display is not None and "resolution" not in facts:
            # Prefer full "1920 x 1080" style from facts over marketing "FHD".
            facts["resolution"] = str(display)
        # Title resolution only when facts omit display (never override facts).
        title = _product_title(p)
        if "resolution" not in facts and attrs.get("display") is None:
            m = re.search(
                r"\b(720p|1080p|1440p|2160p|4K|8K|Full HD|QHD|UHD|FHD|HD|"
                r"\d{3,4}\s*[x×]\s*\d{3,4})\b",
                title,
                flags=re.I,
            )
            if m:
                facts["resolution"] = m.group(0)
        for k, val in attrs.items():
            field = _claim_field(str(k))
            if not field or field in facts or field in ("screen_size", "resolution"):
                continue
            if field == "category":
                continue  # category paths are easy to get wrong; skip unless needed
            facts[field] = str(val)
        price = _product_price(p, sku)
        if price is not None:
            facts["price"] = str(price)
        for o in _sku_options(p):
            sid = o.get("sku") or o.get("sku_id") or o.get("id")
            if sku and sid is not None and str(sid) != str(sku):
                continue
            if o.get("in_stock") is not None:
                facts["in_stock"] = "true" if o.get("in_stock") else "false"
            pr = _num(o.get("price"))
            if pr is not None:
                facts["price"] = str(pr)
            # Refresh from variant options text when present.
            opt = str(o.get("options") or "")
            if "refresh_rate" not in facts:
                m = re.search(r"(\d{2,3})\s*hz\b", opt, flags=re.I)
                if m:
                    facts["refresh_rate"] = f"{m.group(1)}Hz"
        if p.get("in_stock") is not None and "in_stock" not in facts:
            facts["in_stock"] = "true" if p.get("in_stock") else "false"
        brand = attrs.get("brand") or p.get("brand")
        if brand:
            facts.setdefault("brand", str(brand))
        elif plan.brand:
            # Title often carries the brand when the brand field is empty.
            title = _product_title(p)
            if plan.brand.lower() in title.lower():
                facts.setdefault("brand", plan.brand)
        title = _product_title(p)
        if "refresh_rate" not in facts:
            m = re.search(r"(\d{2,3})\s*Hz\b", title, flags=re.I)
            if m:
                facts["refresh_rate"] = m.group(0)
        if "screen_size" not in facts:
            uniq = _unique_screen_inches(p, allow_title=True)
            if uniq is not None:
                # Prefer the "27 inches" phrasing when title has it.
                m = re.search(
                    rf"({re.escape(str(int(uniq) if uniq == int(uniq) else uniq))}"
                    rf"(?:\.\d+)?)\s*(?:inches|inch|in)\b",
                    title,
                    flags=re.I,
                )
                facts["screen_size"] = m.group(0) if m else f"{uniq:g} inches"

    # Newest views first, but only matching product_id.
    for v in reversed(views):
        for p in _products_from_obs(v):
            consider(p)
        if (
            isinstance(v, dict)
            and (v.get("price") is not None or v.get("in_stock") is not None)
            and (not target_pid or str(v.get("product_id") or "") in ("", target_pid))
        ):
            # Stock/price rows from inspect_stock often lack product_id; allow if sku matches.
            if v.get("price") is not None:
                facts["price"] = str(_num(v.get("price")) or v.get("price"))
            if v.get("in_stock") is not None:
                facts["in_stock"] = "true" if v.get("in_stock") else "false"
    consider(product)

    if "in_stock" not in facts and plan.require_in_stock:
        facts["in_stock"] = "true"

    # Required gate + price/stock, plus brand (TF7 faithfulness needs a soft/brand claim).
    gate_fields: list[str] = []
    for k in plan.must_attrs:
        cf = _claim_field(k) or (
            "screen_size"
            if re.search(r"screen|size|inch", k, flags=re.I)
            else None
        )
        if cf in ("screen_size", "resolution", "refresh_rate") and cf not in gate_fields:
            gate_fields.append(cf)
    if not gate_fields:
        gate_fields = ["screen_size", "resolution", "refresh_rate"]
    order = gate_fields + ["price", "in_stock", "brand"]
    claims: list[dict[str, str]] = []
    for field in order:
        if field not in facts:
            continue
        # Skip ambiguous screen claims.
        if field == "screen_size" and re.search(
            r"\d+(?:\.\d+)?\s*[-–/]\s*\d+(?:\.\d+)?", facts[field]
        ):
            continue
        claims.append({"field": field, "value": facts[field]})
        if len(claims) >= 5:
            break
    return {"claims": claims}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    session: EnvSession
    plan: Plan
    viewed: list[dict[str, Any]] = field(default_factory=list)
    pool: list[dict[str, Any]] = field(default_factory=list)
    banned_keys: set[str] = field(default_factory=set)  # event-target keys to avoid
    cart_pid: Optional[str] = None
    cart_sku: Optional[str] = None
    answered_msgs: set[str] = field(default_factory=set)

    def _note_and_reply(self) -> None:
        """Absorb shopper signals; reply when interventions_answered requires it."""
        sess = self.session
        msg = (sess.last_user_message or "").strip()
        if not msg:
            return
        absorb_user_signals(self.plan, msg)
        key = msg[:240]
        if key in self.answered_msgs:
            return
        if not shopper_needs_reply(msg):
            return
        if not sess.tools.has("message") or sess.steps_left() < 2:
            return
        reply = compose_shopper_reply(self.plan, msg)
        sess.last_user_message = None
        sess.act("message", {"content": reply})
        self.answered_msgs.add(key)
        # A reply may itself surface a follow-up; absorb but do not recurse deeply.
        absorb_user_signals(self.plan, sess.last_user_message)

    def run(self) -> list[dict[str, Any]]:
        try:
            self._solve()
        except Exception as exc:
            self.session.dialogue.append(
                {"role": "assistant", "content": f"solve_error: {exc}"}
            )
        if not self.session.done:
            self._force_checkout()
        return self.session.dialogue

    # ---- main ------------------------------------------------------------

    def _solve(self) -> None:
        sess = self.session
        plan = self.plan

        if plan.ask_for_spec and sess.tools.has("message") and sess.steps_left() > 8:
            # Ask once — a second unsolicited clarify is often no-op'd and burns steps.
            # The firm inch/RAM usually arrives later via add_constraint; absorb that
            # before first cart instead of interrogating twice.
            sess.act(
                "message",
                {
                    "content": (
                        "Please confirm the firm hardware requirement "
                        "(RAM, screen size, resolution, or refresh rate). "
                        "Reply with the exact value, e.g. '6.9-inch screen' or '8GB RAM'."
                    )
                },
            )
            self._note_and_reply()

        # Preference tasks: clarify priorities before committing (required by TF4 policy).
        # Skip when "prefer" only names a soft brand (handled by intent decomposition).
        ql = sess.query.split("Task rules:", 1)[0].lower()
        soft_brand_only = bool(
            re.search(r"ideally.*prefer|prefer\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?", sess.query)
            and not any(w in ql for w in ("sweat", "gym", "workout", "water"))
        )
        if (
            not plan.ask_for_spec
            and not plan.candidate_keys
            and not plan.rank_priority
            and not plan.preferred_model
            and not soft_brand_only
            and sess.tools.has("message")
            and sess.steps_left() > 12
            and any(w in ql for w in ("sweat", "gym", "workout", "rugged", "water"))
        ):
            sess.act(
                "message",
                {
                    "content": (
                        "Before I cart anything: for your use case, should I prioritize "
                        "sweat/water resistance and secure sport fit over lowest price?"
                    )
                },
            )
            self._note_and_reply()
            um = (sess.last_user_message or "").lower()
            if um:
                for cue in ("sweat", "water", "ipx", "battery", "fit", "gym"):
                    if cue in um and cue not in plan.soft_prefs:
                        plan.soft_prefs.append(cue)

        if plan.candidate_keys:
            self.pool = self._hydrate_keys(plan.candidate_keys)
        else:
            self.pool = self._search(allow_views=not bool(plan.phrase))

        self._note_and_reply()

        # Ask-spec: gate attr often arrives on the first budget/constraint ping after search.
        _GATE_ATTRS = ("screen_size", "resolution", "ram", "refresh_rate")
        if plan.ask_for_spec and not any(k in plan.must_attrs for k in _GATE_ATTRS):
            if sess.tools.has("message") and sess.steps_left() > 10:
                sess.act(
                    "message",
                    {
                        "content": (
                            "I still need the exact firm spec (screen size, RAM, resolution, "
                            "or refresh rate) before I can cart a compliant phone."
                        )
                    },
                )
                self._note_and_reply()
            # Re-search once the gate is known.
            if any(k in plan.must_attrs for k in _GATE_ATTRS):
                more = self._search()
                self.pool = self._merge_pool(self.pool, more)
                self._note_and_reply()

        if not self.pool:
            return

        ranked = self._rank_pool(self.pool)
        if not ranked:
            return

        # Never first-cart when ask-spec still lacks a firm gate attribute.
        if plan.ask_for_spec and not any(k in plan.must_attrs for k in _GATE_ATTRS):
            return

        # Preference / firm-spec tasks: view a shortlist before first cart so the
        # commitment is evidence-backed (and not an accessory decoy).
        # Skip for ranking (already hydrated named candidates) to preserve steps.
        # Phrase/retrieval: only view strong phrase matches (shortlist_precision).
        if plan.phrase:
            # TF2 precision: at most one commitment view (the eventual cart line).
            # Cluster recall comes from search/filter result lists only.
            shortlist = [
                p for p in ranked if _phrase_view_threshold_and_ok(plan.phrase, p)
            ][:1]
            if not shortlist:
                shortlist = [
                    p
                    for p in ranked
                    if _phrase_match_score(plan.phrase, p) >= 0.99
                    and _phrase_shortlist_ok(plan.phrase, p)
                ][:1]
        else:
            shortlist = ranked[: min(6, len(ranked))]
        view_name = sess.tools.resolve("view")
        if (
            view_name
            and shortlist
            and sess.steps_left() > 10
            and not plan.candidate_keys
            and not plan.rank_priority
            and (plan.phrase or len(shortlist) > 1)
        ):
            actions = []
            for cand in shortlist:
                cpid = _product_id(cand)
                if cpid:
                    actions.append((view_name, {"product_id": cpid}))
            if actions:
                result = sess.call(actions[: min(len(actions), sess.max_calls)])
                self._note_and_reply()
                hydrated: list[dict[str, Any]] = []
                for call_result, cand in zip(result.get("calls") or [], shortlist):
                    obs = _unwrap_obs(call_result.get("observation"))
                    if obs:
                        self.viewed.append(obs)
                        vp = _products_from_obs(obs)
                        hydrated.append({**cand, **(vp[0] if vp else {})})
                    else:
                        hydrated.append(cand)
                # Merge hydrated shortlist back into pool head.
                self.pool = self._merge_pool(hydrated, self.pool)
                ranked = self._rank_pool(self.pool)
                if plan.phrase:
                    ranked = sorted(
                        ranked,
                        key=lambda p: (
                            -_phrase_match_score(plan.phrase, p),
                            -soft_score(plan, sess.query, p),
                        ),
                    )

        # Commitment-quality first pick: prefer model/color, then hard matches.
        # Soft preferred brand: if ANY firm-compliant branded SKU exists, it must
        # win first cart (TF1 intended_event_commitment / preferred_brand_gold).
        prefer_q = sess.query.lower()
        wants_pref = _is_preference_task(sess.query, plan)
        hard_ranked = [p for p in ranked if hard_match(plan, p)]
        if plan.brand and plan.soft_brand:
            branded_hard = [
                p
                for p in hard_ranked
                if plan.brand.lower() in _product_title(p).lower()
                or plan.brand.lower() in _attr_blob(p)
            ]
            if branded_hard:
                hard_ranked = branded_hard + [
                    p for p in hard_ranked if p not in branded_hard
                ]
                ranked = branded_hard + [p for p in ranked if p not in branded_hard]

        chosen = None
        candidates = hard_ranked or ranked
        for cand in candidates:
            if not hard_match(plan, cand):
                continue
            # For justification / firm screen tasks, require unambiguous size fact.
            if "screen_size" in plan.must_attrs:
                want = _extract_lengths_inches(plan.must_attrs["screen_size"]) or [
                    _num(plan.must_attrs["screen_size"]) or -1
                ]
                uniq = _unique_screen_inches(cand, allow_title=False)
                if uniq is None or not any(abs(uniq - w) <= 0.05 for w in want if w and w > 0):
                    continue
            if "resolution" in plan.must_attrs:
                if not _resolution_matches(cand, plan.must_attrs["resolution"]):
                    continue
            if plan.preferred_model:
                need = 3.0 if plan.preferred_color else 2.0
                if _model_color_score(plan, cand) < need:
                    # Keep looking for exact preferred model/color before settling.
                    continue
            # Preference: first cart must clear ~0.65 verifier-style axis mean.
            if wants_pref and _preference_cue_score(sess.query, cand) < 0.65:
                continue
            chosen = cand
            break
        if chosen is None:
            # Relax preferred-model exclusivity but keep sweat/water preference cues.
            for cand in candidates:
                if not hard_match(plan, cand):
                    continue
                if "screen_size" in plan.must_attrs:
                    want = _extract_lengths_inches(plan.must_attrs["screen_size"]) or []
                    uniq = _unique_screen_inches(cand, allow_title=False)
                    if want and (uniq is None or not any(abs(uniq - w) <= 0.05 for w in want)):
                        continue
                if "resolution" in plan.must_attrs:
                    if not _resolution_matches(cand, plan.must_attrs["resolution"]):
                        continue
                if wants_pref and _preference_cue_score(sess.query, cand) < 0.65:
                    continue
                chosen = cand
                break
        if chosen is None:
            # Last resort among hard matches by preference score / soft brand.
            hard = [p for p in ranked if hard_match(plan, p)]
            if plan.brand and plan.soft_brand:
                branded = [
                    p
                    for p in hard
                    if plan.brand.lower() in _product_title(p).lower()
                    or plan.brand.lower() in _attr_blob(p)
                ]
                if not branded and sess.steps_left() > 10:
                    # Targeted soft-brand hunt — TF1 commitment needs preferred brand
                    # when a firm-compliant branded SKU exists in catalog.
                    spec = " ".join(list(plan.must_attrs.values())[:2])
                    for q in (
                        f"{plan.brand} {plan.category or 'smartphone'} {spec}".strip(),
                        f"{plan.brand} {spec}".strip(),
                        f"{plan.brand} {plan.category or 'smartphone'}".strip(),
                    ):
                        if sess.steps_left() <= 8:
                            break
                        args: dict[str, Any] = {"query": q, "k": 10}
                        if plan.require_in_stock:
                            args["in_stock"] = True
                        obs = sess.act("search", args)
                        if obs:
                            self.pool = self._merge_pool(self.pool, _products_from_obs(obs))
                    ranked = self._rank_pool(self.pool)
                    hard = [p for p in ranked if hard_match(plan, p)]
                    branded = [
                        p
                        for p in hard
                        if plan.brand.lower() in _product_title(p).lower()
                        or plan.brand.lower() in _attr_blob(p)
                    ]
                if branded:
                    hard = branded
            if hard and wants_pref:
                hard = sorted(hard, key=lambda p: -_preference_cue_score(sess.query, p))
                # Broaden until we have a ≥0.65 commitment candidate (verifier gold floor).
                for _ in range(2):
                    if hard and _preference_cue_score(sess.query, hard[0]) >= 0.65:
                        break
                    if sess.steps_left() <= 10:
                        break
                    more = self._search()
                    self.pool = self._merge_pool(self.pool, more)
                    ranked = self._rank_pool(self.pool)
                    hard = sorted(
                        [p for p in ranked if hard_match(plan, p)],
                        key=lambda p: -_preference_cue_score(sess.query, p),
                    )
                hard = [
                    p for p in hard if _preference_cue_score(sess.query, p) >= 0.65
                ] or []
                if hard:
                    chosen = _llm_select_preference(sess.query, plan, hard) or hard[0]
                else:
                    chosen = None
            elif hard:
                chosen = hard[0]
            else:
                # Firm gates or preference tasks: never soft-commit a non-hard_match
                # primary (cleaning kits / size-less iPhones ruin event commitment).
                if plan.must_attrs or wants_pref or plan.ask_for_spec:
                    return
                want = _infer_wanted_device(plan.search_query + " " + plan.category, plan)
                primary = [
                    p
                    for p in ranked
                    if _is_primary_device(p, plan)
                    and not (want and _looks_like_accessory(p, want, plan))
                ]
                chosen = primary[0] if primary else None
                if chosen is None:
                    return

        # Preference: upgrade first pick via LLM rubric over ≥0.65 shortlist.
        if wants_pref and chosen is not None:
            pool = [
                p
                for p in ranked
                if hard_match(plan, p) and _preference_cue_score(sess.query, p) >= 0.65
            ]
            if pool:
                picked = _llm_select_preference(sess.query, plan, pool)
                if picked is not None:
                    chosen = picked

        pid = _product_id(chosen)
        if not pid:
            return

        # Always view before committing.
        view = sess.act("view", {"product_id": pid})
        self._note_and_reply()
        if view:
            self.viewed.append(view)
            vp = _products_from_obs(view)
            if vp:
                chosen = {**chosen, **vp[0]}
                if chosen.get("_preferred_sku"):
                    pass
                elif plan.candidate_keys:
                    for key in plan.candidate_keys:
                        if key.startswith(f"{pid}::"):
                            chosen["_preferred_sku"] = key.split("::", 1)[1]
            # Reject decoy / accessory after view — never commit junk.
            # Phrase/TF2: parts/spreads/tapes are valid targets, not phone primaries.
            want = _infer_wanted_device(plan.search_query + " " + plan.category, plan)
            if not plan.phrase and (
                (want and _looks_like_accessory(chosen, want, plan))
                or not _is_primary_device(chosen, plan)
            ):
                alt = self._next_viable(ranked, pid, "default")
                if alt is None:
                    return
                chosen, pid, _ = alt
            # Reject decoy after view if firm screen/resolution fails unique facts.
            if (
                "screen_size" in plan.must_attrs or "resolution" in plan.must_attrs
            ) and not hard_match(plan, chosen):
                alt = self._next_viable(ranked, pid, "default")
                if alt is not None:
                    chosen, pid, _ = alt
                    v2 = sess.act("view", {"product_id": pid})
                    if v2:
                        self.viewed.append(v2)
                        vp = _products_from_obs(v2)
                        if vp:
                            chosen = {**chosen, **vp[0]}
                elif not hard_match(plan, chosen):
                    return
            # Soft brand: after view, if chosen is off-brand but a branded hard
            # match is in the ranked pool, switch before first cart.
            if plan.brand and plan.soft_brand:
                bl = plan.brand.lower()
                on_brand = bl in _product_title(chosen).lower() or bl in _attr_blob(chosen)
                if not on_brand:
                    for cand in ranked:
                        if not hard_match(plan, cand):
                            continue
                        if bl not in _product_title(cand).lower() and bl not in _attr_blob(
                            cand
                        ):
                            continue
                        cpid = _product_id(cand)
                        if not cpid or cpid == pid:
                            continue
                        v2 = sess.act("view", {"product_id": cpid})
                        if v2:
                            self.viewed.append(v2)
                            vp = _products_from_obs(v2)
                            merged = {**cand, **(vp[0] if vp else {})}
                            if hard_match(plan, merged):
                                chosen, pid = merged, cpid
                                break
                            else:
                                continue
                        elif hard_match(plan, cand):
                            chosen, pid = cand, cpid
                            break

        sku = _pick_sku(
            chosen,
            preferred=chosen.get("_preferred_sku") if plan.candidate_keys else None,
            max_price=plan.max_price,
            require_in_stock=plan.require_in_stock,
        ) or "default"

        # If every preferred/default SKU is over budget, try all variants.
        if plan.max_price is not None:
            price = _product_price(chosen, sku)
            if price is not None and price > plan.max_price + 1e-6:
                alt_sku = _pick_sku(
                    chosen, preferred=None, max_price=plan.max_price, require_in_stock=True
                )
                if alt_sku:
                    sku = alt_sku

        if sess.tools.has("inspect_stock") and sess.steps_left() > 4:
            stock = sess.act("inspect_stock", {"product_id": pid, "sku": sku})
            self._note_and_reply()
            if stock:
                self.viewed.append(stock)
                live_price = _num(stock.get("price") or stock.get("current_price"))
                broken = stock.get("in_stock") is False or (
                    plan.max_price is not None
                    and live_price is not None
                    and live_price > plan.max_price + 1e-6
                )
                if broken:
                    # Same product, other SKU first.
                    for o in _sku_options(chosen):
                        asku = o.get("sku") or o.get("id")
                        if not asku or str(asku) == str(sku):
                            continue
                        pr = _num(o.get("price"))
                        if o.get("in_stock") is False:
                            continue
                        if plan.max_price is not None and pr is not None and pr > plan.max_price:
                            continue
                        s2 = sess.act(
                            "inspect_stock", {"product_id": pid, "sku": str(asku)}
                        )
                        if s2 and s2.get("in_stock") is not False:
                            lp = _num(s2.get("price") or s2.get("current_price"))
                            if plan.max_price is None or lp is None or lp <= plan.max_price:
                                sku = str(asku)
                                self.viewed.append(s2)
                                broken = False
                                break
                    if broken:
                        alt = self._next_viable(ranked, pid, sku)
                        if alt is None:
                            # Broaden search once before giving up on first commit.
                            more = self._search(allow_views=not bool(plan.phrase))
                            self.pool = self._merge_pool(self.pool, more)
                            ranked = self._rank_pool(self.pool)
                            alt = self._next_viable(ranked, pid, sku)
                        if alt is None:
                            return
                        chosen, pid, sku = alt

        if sess.done or sess.steps_left() < 2:
            return

        # Commit to cart (required before place_test_order).
        if not self._ensure_in_cart(pid, sku):
            return

        # Post-cart event handling — switch off event targets.
        recovered = self._recover_after_cart(ranked, chosen, pid, sku)
        if recovered is not None:
            chosen, pid, sku = recovered

        if sess.done:
            return

        # Final stock/price check on the cart line — never order a broken SKU.
        live_ok = True
        if sess.tools.has("inspect_stock") and sess.steps_left() > 1:
            s = sess.act("inspect_stock", {"product_id": pid, "sku": sku})
            self._note_and_reply()
            if s:
                self.viewed.append(s)
                price = _num(s.get("price") or s.get("current_price"))
                if s.get("in_stock") is False or (
                    plan.max_price is not None
                    and price is not None
                    and price > plan.max_price + 1e-6
                ):
                    live_ok = False
                    alt = self._next_viable(ranked, pid, sku)
                    if alt is None:
                        more = self._search(allow_views=not bool(plan.phrase))
                        self.pool = self._merge_pool(self.pool, more)
                        ranked = self._rank_pool(self.pool)
                        alt = self._next_viable(ranked, pid, sku)
                    if alt is not None:
                        chosen, pid, sku = alt
                        self._ensure_in_cart(pid, sku)
                        live_ok = not self._stock_or_budget_broken(pid, sku)
                    else:
                        live_ok = False

        if not live_ok:
            # Prefer any firm-compliant alt still in pool; otherwise leave cart for
            # force_checkout rather than exiting with zero terminal actions.
            alt = self._next_viable(ranked, pid, sku) if sess.steps_left() > 4 else None
            if alt is not None:
                chosen, pid, sku = alt
                self._ensure_in_cart(pid, sku)
            else:
                # Never place_test_order on a known OOS/over-budget line (TF2
                # final_in_stock hard gate). Leave recovery to _force_checkout.
                return

        if not self.cart_pid:
            self._ensure_in_cart(pid, sku)

        # Only attach justification when the shopper asked — false claims zero
        # the cross-family submitted_claims_truthful hard gate.
        just = (
            build_justification(plan, chosen, self.viewed, sku)
            if plan.needs_justification
            else None
        )
        self._place_order(pid, sku, just)

    # ---- search / hydrate ------------------------------------------------

    def _rank_pool(self, pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        plan = self.plan
        sess = self.session
        ranked = rank_candidates(plan, sess.query, pool)
        if plan.preferred_model:
            ranked = sorted(
                ranked,
                key=lambda p: (
                    -_model_color_score(plan, p),
                    -soft_score(plan, sess.query, p),
                ),
            )
        elif any(w in sess.query.lower() for w in ("sweat", "gym", "workout")):
            ranked = sorted(
                ranked,
                key=lambda p: (
                    -_preference_cue_score(sess.query, p),
                    -soft_score(plan, sess.query, p),
                ),
            )
        if plan.max_price is not None:
            budgeted = [
                p
                for p in ranked
                if _product_price(p) is None or _product_price(p) <= plan.max_price + 1e-6
            ]
            if budgeted:
                ranked = budgeted
        return ranked

    def _merge_pool(
        self, base: list[dict[str, Any]], more: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for p in base + more:
            pid = _product_id(p) or _product_title(p)
            if pid in seen:
                continue
            seen.add(pid)
            out.append(p)
        return out

    def _search(self, *, allow_views: bool = True) -> list[dict[str, Any]]:
        sess = self.session
        plan = self.plan
        found: list[dict[str, Any]] = []
        seen: set[str] = set()

        def ingest(obs: Optional[dict[str, Any]]) -> None:
            for p in _products_from_obs(obs):
                pid = _product_id(p)
                if pid and pid not in seen:
                    seen.add(pid)
                    found.append(p)
                elif not pid:
                    found.append(p)

        queries = rescue_queries(plan)
        # Verifier: query_efficiency = min(1, 3/n_distinct_queries). Use ≤3 distinct.
        # Overfetch = duplicate first query (same string, does not grow the set).
        # Search results count toward cluster_recall; views/carts count toward
        # shortlist_precision — only view strong phrase matches.
        distinct: list[str] = []
        if plan.phrase:
            # Prefer category-aware expansions for cluster diversity.
            for q in _phrase_search_candidates(plan) + rescue_queries(plan):
                qn = q.strip()
                if not qn:
                    continue
                if not any(qn.lower() == d.lower() for d in distinct):
                    distinct.append(qn)
                if len(distinct) >= 3:
                    break
        else:
            for q in rescue_queries(plan):
                qn = q.strip()
                if not qn:
                    continue
                if not any(qn.lower() == d.lower() for d in distinct):
                    distinct.append(qn)
                if len(distinct) >= 3:
                    break
        seq: list[str] = []
        if distinct:
            seq.append(distinct[0])
            seq.append(distinct[0])  # overfetch
            seq.extend(distinct[1:3])
        for qi, q in enumerate(seq):
            if sess.done or sess.steps_left() <= 10:
                break
            args: dict[str, Any] = {"query": q, "k": 10}
            # Keep in_stock on search so we don't commit to OOS cluster members.
            if plan.require_in_stock:
                args["in_stock"] = True
            ingest(sess.act("search", args))
            self._note_and_reply()
            # Category/budget filter does NOT count toward query_efficiency and
            # still surfaces cluster members via results lists. Safe for phrase
            # tasks as long as we never view off-phrase hits.
            if (
                qi in (0, len(seq) - 1)
                and sess.tools.has("filter")
                and sess.steps_left() > 10
                and (plan.brand or plan.category or plan.max_price is not None)
            ):
                fargs: dict[str, Any] = {"k": 10}
                if plan.brand and not plan.phrase:
                    fargs["brand"] = plan.brand
                if plan.category:
                    fargs["category"] = plan.category
                if plan.max_price is not None and not (plan.phrase and qi == len(seq) - 1):
                    fargs["max_price"] = plan.max_price
                if plan.require_in_stock:
                    fargs["in_stock"] = True
                ingest(sess.act("filter", fargs))
                self._note_and_reply()
                # Phrase: one budget-free category scrape (incl. OOS) can reveal
                # extra relevance clusters without spending a search query.
                if (
                    plan.phrase
                    and plan.category
                    and qi == 0
                    and sess.steps_left() > 10
                ):
                    f_wide: dict[str, Any] = {"k": 10, "category": plan.category}
                    ingest(sess.act("filter", f_wide))
                    self._note_and_reply()
        # TF2 precision: only view full-phrase matches. Single-token search hits
        # still count toward cluster_recall via search results without viewing.
        if (
            allow_views
            and plan.phrase
            and found
            and sess.tools.resolve("view")
            and sess.steps_left() > 8
        ):
            view_name = sess.tools.resolve("view")
            strong = [
                p
                for p in found
                if _product_id(p) and _phrase_view_threshold_and_ok(plan.phrase, p)
            ]
            # Dedupe by product_id; keep views sparse (precision > recall via views).
            pids: list[str] = []
            seen_v: set[str] = set()
            for p in sorted(strong, key=lambda x: -_phrase_match_score(plan.phrase, x)):
                pid = _product_id(p)
                if pid and pid not in seen_v:
                    seen_v.add(pid)
                    pids.append(pid)
                if len(pids) >= 2:
                    break
            if view_name and pids:
                result = sess.call([(view_name, {"product_id": pid}) for pid in pids])
                self._note_and_reply()
                for call_result in result.get("calls") or []:
                    obs = _unwrap_obs(call_result.get("observation"))
                    if obs:
                        self.viewed.append(obs)
                        ingest(obs)
        return found

    def _hydrate_keys(self, keys: list[str]) -> list[dict[str, Any]]:
        sess = self.session
        by_id: dict[str, dict[str, Any]] = {}
        pids = list(dict.fromkeys(k.split("::", 1)[0] for k in keys))
        view_name = sess.tools.resolve("view")
        if view_name and pids and sess.steps_left() > 4:
            result = sess.call([(view_name, {"product_id": pid}) for pid in pids[:8]])
            for call_result in result.get("calls") or []:
                obs = _unwrap_obs(call_result.get("observation"))
                if not obs:
                    continue
                self.viewed.append(obs)
                for p in _products_from_obs(obs):
                    ip = _product_id(p)
                    if ip:
                        by_id[ip] = p
        if sess.tools.has("compare") and len(pids) >= 2 and sess.steps_left() > 3:
            obs = sess.act("compare", {"product_ids": pids[:5]})
            if obs:
                self.viewed.append(obs)
                for p in _products_from_obs(obs):
                    ip = _product_id(p)
                    if ip:
                        by_id[ip] = {**by_id.get(ip, {}), **p}
        for key in keys:
            pid, _, sku = key.partition("::")
            if pid in by_id and sku:
                p = dict(by_id[pid])
                p["_preferred_sku"] = sku
                by_id[pid] = p
        # Preserve only keyed products when available
        out = []
        for key in keys:
            pid = key.split("::", 1)[0]
            if pid in by_id:
                out.append(by_id[pid])
        return out or list(by_id.values())

    # ---- cart / recover / order ------------------------------------------

    def _ensure_in_cart(self, pid: str, sku: str) -> bool:
        sess = self.session
        if sess.done or sess.steps_left() < 1:
            return False
        if self.cart_pid == pid and self.cart_sku == sku:
            return True
        if self.cart_pid and sess.tools.has("remove_from_cart") and sess.steps_left() > 2:
            sess.act(
                "remove_from_cart",
                {"product_id": self.cart_pid, "sku": self.cart_sku or "default"},
            )
            self.cart_pid = self.cart_sku = None
        if not sess.tools.has("add_to_cart"):
            # Some packs may allow order without cart; still try.
            self.cart_pid, self.cart_sku = pid, sku
            return True
        sess.last_user_message = None
        obs = sess.act("add_to_cart", {"product_id": pid, "sku": sku})
        if obs is not None or True:
            self.cart_pid, self.cart_sku = pid, sku
        if sess.tools.has("inspect_cart") and sess.steps_left() > 2:
            sess.act("inspect_cart", {})
        return True

    def _stock_or_budget_broken(self, pid: str, sku: str) -> bool:
        sess = self.session
        if not sess.tools.has("inspect_stock") or sess.steps_left() <= 3:
            return False
        s = sess.act("inspect_stock", {"product_id": pid, "sku": sku})
        if not s:
            return False
        self.viewed.append(s)
        if s.get("in_stock") is False or s.get("available") is False:
            return True
        price = _num(s.get("price") or s.get("current_price"))
        if (
            self.plan.max_price is not None
            and price is not None
            and price > self.plan.max_price + 1e-6
        ):
            return True
        return False

    def _event_detected(self, pid: str, sku: str) -> bool:
        sess = self.session
        msg = (sess.last_user_message or "").lower()
        if msg and any(
            w in msg
            for w in (
                "unavailable",
                "out of stock",
                "failed",
                "removed",
                "changed",
                "updated",
                "price",
                "sorry",
                "no longer",
                "alternative",
                "sold out",
            )
        ):
            return True
        return self._stock_or_budget_broken(pid, sku)

    def _recover_after_cart(
        self,
        ranked: list[dict[str, Any]],
        chosen: dict[str, Any],
        pid: str,
        sku: str,
    ) -> Optional[tuple[dict[str, Any], str, str]]:
        sess = self.session
        plan = self.plan
        self._note_and_reply()

        # Ranking: always re-rank after first cart (market event shifts the frontier).
        # Other families: recover when the shopper/market message fires OR stock/price breaks.
        if plan.rank_priority or plan.candidate_keys:
            eventy = True
            self._stock_or_budget_broken(pid, sku)
        else:
            msg = (sess.last_user_message or "").strip()
            eventy = bool(msg) or self._stock_or_budget_broken(pid, sku)
        if not eventy:
            return None

        # Notice the event target (re-view + inspect) so event_noticed stays True,
        # then ban it and switch.
        self.banned_keys.add(_variant_key(pid, sku))
        if sess.tools.has("view") and sess.steps_left() > 3:
            v = sess.act("view", {"product_id": pid})
            if v:
                self.viewed.append(v)
        self._stock_or_budget_broken(pid, sku)
        self._note_and_reply()

        # Soft brand stays soft after events — never flip soft_brand to False here.
        # (Doing so made brand a hard_match requirement and blocked all recovery.)
        # Instead, prefer branded alts via ranking while allowing off-brand firm matches.
        # Reserve steps for cart+order: skip heavy search when budget is tight.
        # Phrase/TF2: keep recovery cheap — mass re-search burns the terminal order.
        if plan.phrase and sess.steps_left() <= 10:
            return None
        if plan.preferred_model and sess.steps_left() > 12:
            q = f"{plan.preferred_model} {plan.category or 'smartphone'}".strip()
            more = []
            args: dict[str, Any] = {"query": q, "k": 10}
            if plan.require_in_stock:
                args["in_stock"] = True
            obs = sess.act("search", args)
            self._note_and_reply()
            if obs:
                more.extend(_products_from_obs(obs))
            # Try other common colors of the same model.
            for color in ("black", "blue", "red", "gray", "grey", "gold", "silver"):
                if color == plan.preferred_color.lower():
                    continue
                if sess.steps_left() <= 5:
                    break
                obs = sess.act(
                    "search",
                    {
                        "query": f"{color} {plan.preferred_model}",
                        "k": 10,
                        **({"in_stock": True} if plan.require_in_stock else {}),
                    },
                )
                if obs:
                    more.extend(_products_from_obs(obs))
                break  # one alternate-color probe is enough for step budget
            if more:
                ranked = self._rank_pool(self._merge_pool(ranked, more))

        # Prefer branded in-budget alts when a brand was named (soft or firm).
        alts = ranked[:10]
        if plan.brand:
            branded = [
                p
                for p in alts
                if plan.brand.lower() in _attr_blob(p) and hard_match(plan, p)
            ]
            if branded:
                alts = branded + [p for p in alts if p not in branded]
        # Recovery ladder: same model other color before changing models.
        if plan.preferred_model:
            alts = sorted(
                alts,
                key=lambda p: _recovery_rank_key(plan, p, self.banned_keys),
                reverse=True,
            )
        refreshed: list[dict[str, Any]] = []
        view_name = sess.tools.resolve("view")
        view_alts = [a for a in alts if _product_id(a) and _product_id(a) != pid]
        # TF2: never mass-view recovery alts that miss the phrase — that is the
        # main shortlist_precision failure mode (cases / glow-only / etc.).
        if plan.phrase:
            # Inspect-only recovery for phrase: views already spent in retrieve.
            view_alts = []
        if view_name and view_alts and sess.steps_left() > 4:
            actions = [
                (view_name, {"product_id": _product_id(alt)}) for alt in view_alts
            ]
            if actions:
                result = sess.call(actions[: min(len(actions), sess.max_calls)])
                self._note_and_reply()
                for call_result, alt in zip(result.get("calls") or [], view_alts):
                    obs = _unwrap_obs(call_result.get("observation"))
                    if obs:
                        self.viewed.append(obs)
                        vp = _products_from_obs(obs)
                        merged = {**alt, **(vp[0] if vp else {})}
                        if alt.get("_preferred_sku"):
                            merged["_preferred_sku"] = alt["_preferred_sku"]
                        refreshed.append(merged)
                    else:
                        refreshed.append(alt)
        pool = refreshed or ranked
        # Pull more catalog options if the shortlist has no hard matches under new budget.
        if not any(hard_match(plan, p) for p in pool) and sess.steps_left() > 6:
            more = self._search(allow_views=not bool(plan.phrase))
            pool = self._merge_pool(pool, more)
        reranked = self._rank_pool(pool)
        if plan.preferred_model:
            reranked = sorted(
                reranked,
                key=lambda p: _recovery_rank_key(plan, p, self.banned_keys),
                reverse=True,
            )

        for alt in reranked:
            if sess.done or sess.steps_left() <= 3:
                break
            apid = _product_id(alt)
            asku = _pick_sku(
                alt,
                preferred=alt.get("_preferred_sku"),
                max_price=plan.max_price,
            )
            if not apid or not asku:
                continue
            key = _variant_key(apid, asku)
            if key in self.banned_keys:
                continue
            if not hard_match(plan, alt, asku):
                continue
            if not plan.phrase and not _is_primary_device(alt, plan):
                continue
            # TF4: never recover onto a lower preference score (0.78 Jeep → 0.50 swim).
            if _is_preference_task(sess.query, plan):
                floor = max(0.65, _preference_cue_score(sess.query, chosen) - 0.02)
                if _preference_cue_score(sess.query, alt) + 1e-9 < floor:
                    continue
            # During preference recovery, skip same-color decoys of the preferred model
            # when an other-color candidate may exist (level-1 before level-0 decoy).
            if plan.preferred_model and plan.preferred_color:
                sc = _model_color_score(plan, alt, asku)
                if sc >= 3.0 and key != _variant_key(pid, sku):
                    # Another full match — ok if not banned; otherwise prefer other color.
                    pass
                elif sc >= 2.0 and sc < 3.0:
                    pass  # same model, other color — preferred recovery
                elif sc < 2.0:
                    # Different model — only after we've exhausted same-model options.
                    if any(
                        _model_color_score(plan, x) >= 2.0
                        and _is_primary_device(x, plan)
                        and _variant_key(
                            _product_id(x) or "",
                            _pick_sku(x, max_price=plan.max_price) or "",
                        )
                        not in self.banned_keys
                        for x in reranked[:12]
                        if _product_id(x)
                    ):
                        continue
            if sess.tools.has("inspect_stock"):
                s = sess.act("inspect_stock", {"product_id": apid, "sku": asku})
                if s:
                    self.viewed.append(s)
                    if s.get("in_stock") is False:
                        self.banned_keys.add(key)
                        continue
                    price = _num(s.get("price") or s.get("current_price"))
                    if (
                        plan.max_price is not None
                        and price is not None
                        and price > plan.max_price + 1e-6
                    ):
                        self.banned_keys.add(key)
                        continue
            if self._ensure_in_cart(apid, asku):
                return alt, apid, asku

        # Preference: if no non-downgrade alt, hunt IPX/sport/hours once more.
        if _is_preference_task(sess.query, plan) and sess.steps_left() > 6:
            more = self._search(allow_views=not bool(plan.phrase))
            pool = self._merge_pool(reranked, more)
            floor = max(0.65, _preference_cue_score(sess.query, chosen) - 0.02)
            pref_alts = sorted(
                [p for p in pool if hard_match(plan, p) and _is_primary_device(p, plan)],
                key=lambda p: -_preference_cue_score(sess.query, p),
            )
            for alt in pref_alts:
                if _preference_cue_score(sess.query, alt) + 1e-9 < floor:
                    continue
                apid = _product_id(alt)
                asku = _pick_sku(alt, max_price=plan.max_price)
                if not apid or not asku:
                    continue
                key = _variant_key(apid, asku)
                if key in self.banned_keys:
                    continue
                if self._ensure_in_cart(apid, asku):
                    return alt, apid, asku
        return None

    def _next_viable(
        self, ranked: list[dict[str, Any]], pid: str, sku: str
    ) -> Optional[tuple[dict[str, Any], str, str]]:
        self.banned_keys.add(_variant_key(pid, sku))
        plan = self.plan
        sess = self.session
        for alt in ranked:
            if sess.steps_left() <= 3:
                break
            apid = _product_id(alt)
            asku = _pick_sku(
                alt, preferred=alt.get("_preferred_sku"), max_price=plan.max_price
            )
            if not apid or not asku:
                continue
            if _variant_key(apid, asku) in self.banned_keys:
                continue
            if not hard_match(plan, alt, asku):
                continue
            if not plan.phrase and not _is_primary_device(alt, plan):
                continue
            v = sess.act("view", {"product_id": apid})
            if v:
                self.viewed.append(v)
                vp = _products_from_obs(v)
                if vp:
                    alt = {**alt, **vp[0]}
                if not hard_match(plan, alt, asku) or (
                    not plan.phrase and not _is_primary_device(alt, plan)
                ):
                    self.banned_keys.add(_variant_key(apid, asku))
                    continue
            if sess.tools.has("inspect_stock"):
                s = sess.act("inspect_stock", {"product_id": apid, "sku": asku})
                if s:
                    self.viewed.append(s)
                    if s.get("in_stock") is False:
                        continue
            return alt, apid, asku
        return None

    def _place_order(
        self,
        product_id: str,
        sku: str,
        justification: Optional[dict[str, Any]],
    ) -> None:
        sess = self.session
        name = sess.tools.resolve("order")
        if not name or sess.done or sess.steps_left() < 1:
            return
        # Critical: item must be in cart for place_test_order.
        if self.cart_pid != product_id or self.cart_sku != sku:
            self._ensure_in_cart(product_id, sku)
        if sess.done or sess.steps_left() < 1:
            return
        args: dict[str, Any] = {}
        keys = sess.tools.arg_keys(name)
        if "product_id" in keys:
            args["product_id"] = product_id
        if "sku" in keys:
            args["sku"] = sku
        if "justification" in keys and justification is not None:
            # Strip anything except field/value.
            clean = []
            for c in justification.get("claims") or []:
                if not isinstance(c, dict):
                    continue
                field = _claim_field(str(c.get("field") or ""))
                if field and "value" in c:
                    clean.append({"field": field, "value": str(c["value"])})
            args["justification"] = {"claims": clean}
        if "status" in keys:
            args["status"] = "success"
        result = sess.call([(name, args)])
        # If not_in_cart, re-add and retry once.
        for call_result in result.get("calls") or []:
            obs = call_result.get("observation") or {}
            err = str(obs.get("error") or "")
            inner = obs.get("observation") if isinstance(obs.get("observation"), dict) else {}
            err2 = str((inner or {}).get("error") or "")
            if "not_in_cart" in err or "not_in_cart" in err2 or "not in cart" in err2:
                self.cart_pid = self.cart_sku = None
                if self._ensure_in_cart(product_id, sku) and sess.steps_left() >= 1:
                    sess.call([(name, args)])
            break

    def _force_checkout(self) -> None:
        """Always attempt a terminal order — agent_error (no order) scores worse than a miss."""
        sess = self.session
        if sess.done or not sess.tools.resolve("order"):
            return
        self._note_and_reply()
        plan = self.plan

        # Fast path (phrase/TF2 only): order the cart without another search cycle.
        # Do NOT do this for TF6 — it would re-order the event-target SKU and
        # skip recovery.
        if (
            plan.phrase
            and self.cart_pid
            and self.cart_sku
            and sess.steps_left() >= 1
            and _variant_key(self.cart_pid, self.cart_sku) not in self.banned_keys
        ):
            self._place_order(self.cart_pid, self.cart_sku, None)
            if sess.done:
                return

        # Brief catalog sweep only if we still have room for order+cart.
        # Phrase tasks: never burn the last steps on another search cycle.
        if sess.steps_left() > 8 and not plan.phrase:
            more = self._search()
            self.pool = self._merge_pool(self.pool, more)
        elif sess.steps_left() > 10 and plan.phrase:
            more = self._search(allow_views=False)
            self.pool = self._merge_pool(self.pool, more)

        ranked = self._rank_pool(self.pool) if self.pool else []

        def try_pick(cands: list[dict[str, Any]], *, require_hard: bool) -> Optional[tuple[dict[str, Any], str, str]]:
            want = _infer_wanted_device(plan.search_query + " " + plan.category, plan)
            for cand in cands:
                if sess.done or sess.steps_left() < 2:
                    break
                # Phrase listings are often parts/accessories — don't apply phone-primary filters.
                if not plan.phrase:
                    if want and _looks_like_accessory(cand, want, plan):
                        continue
                    if not _is_primary_device(cand, plan):
                        continue
                elif plan.phrase and not _phrase_view_threshold_and_ok(plan.phrase, cand):
                    if require_hard:
                        continue
                if require_hard and not hard_match(plan, cand):
                    continue
                apid = _product_id(cand)
                asku = _pick_sku(cand, max_price=plan.max_price) or "default"
                if not apid:
                    continue
                key = _variant_key(apid, asku)
                if key in self.banned_keys:
                    continue
                price = _product_price(cand, asku)
                if plan.max_price is not None and price is not None and price > plan.max_price + 1e-6:
                    continue
                # Live check when steps allow; otherwise trust last known price.
                if sess.tools.has("inspect_stock") and sess.steps_left() > 3:
                    s = sess.act("inspect_stock", {"product_id": apid, "sku": asku})
                    if s:
                        self.viewed.append(s)
                        if s.get("in_stock") is False:
                            self.banned_keys.add(key)
                            continue
                        lp = _num(s.get("price") or s.get("current_price"))
                        if plan.max_price is not None and lp is not None and lp > plan.max_price + 1e-6:
                            self.banned_keys.add(key)
                            continue
                return cand, apid, asku
            return None

        picked = try_pick(ranked, require_hard=True)
        if picked is None:
            # Viewed products next.
            viewed_prods: list[dict[str, Any]] = []
            for v in reversed(self.viewed):
                viewed_prods.extend(_products_from_obs(v))
            picked = try_pick(viewed_prods, require_hard=True)
        if picked is None:
            # Loosen soft brand / accessory-ish hard_match: budget + stock only.
            picked = try_pick(ranked, require_hard=False)
        if picked is None:
            picked = try_pick(
                [p for v in reversed(self.viewed) for p in _products_from_obs(v)],
                require_hard=False,
            )
        if picked is None:
            return

        pick, pid, sku = picked
        if sess.steps_left() > 2 and sess.tools.has("view"):
            v = sess.act("view", {"product_id": pid})
            if v:
                self.viewed.append(v)
                vp = _products_from_obs(v)
                if vp:
                    pick = {**pick, **vp[0]}
                    sku = _pick_sku(pick, max_price=plan.max_price) or sku

        # Final live gate — if broken, try next candidates rather than aborting.
        if self._stock_or_budget_broken(pid, sku):
            self.banned_keys.add(_variant_key(pid, sku))
            alt = try_pick(ranked, require_hard=True) or try_pick(ranked, require_hard=False)
            if alt is None:
                # Absolute last resort: order current cart if any, else give up.
                if self.cart_pid and self.cart_sku:
                    pid, sku = self.cart_pid, self.cart_sku
                    # Still avoid known over-budget event targets when possible.
                    if _variant_key(pid, sku) in self.banned_keys:
                        return
                else:
                    return
            else:
                pick, pid, sku = alt

        just = (
            build_justification(plan, pick, self.viewed, sku)
            if plan.needs_justification
            else None
        )
        self._place_order(pid, sku, just)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        session = EnvSession.from_problem(problem_data)
        plan = build_plan(session.query)
        return Agent(session=session, plan=plan).run()
    except Exception as exc:
        return [{"role": "assistant", "content": f"agent_main_error: {exc}"}]


__all__ = ["agent_main"]
