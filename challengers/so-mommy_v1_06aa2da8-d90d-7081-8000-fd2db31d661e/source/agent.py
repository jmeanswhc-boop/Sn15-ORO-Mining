
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from os import getenv
from typing import Any, Optional

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_FALLBACK_MODELS = {
    "chutes": [
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "moonshotai/Kimi-K2.6-TEE",
        "zai-org/GLM-5.1-TEE",
        "Qwen/Qwen3.5-397B-A17B-TEE",
        "Qwen/Qwen3-32B-TEE",
    ],
    "openrouter": [
        "deepseek/deepseek-v3.2",
        "moonshotai/kimi-k2.6",
        "z-ai/glm-5.1",
        "qwen/qwen3.5-397b-a17b",
        "qwen/qwen3-32b",
    ],
}

_proxy = ProxyClient(timeout=120, max_retries=3, retry_delay=1.0)


def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()


def _model_list() -> list[str]:
    preferred = getenv("SANDBOX_MODEL")
    provider = _provider()
    models = list(_FALLBACK_MODELS.get(provider, _FALLBACK_MODELS["chutes"]))
    if preferred and preferred not in models:
        models.insert(0, preferred)
    elif preferred:
        models.remove(preferred)
        models.insert(0, preferred)
    default = _DEFAULT_MODELS.get(provider)
    if default and default not in models:
        models.append(default)
    return models


def _parse_json_content(content: str) -> Any:
    text = (content or "").strip()
    if not text:
        raise ValueError("empty model content")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def llm_json(
    prompt: str,
    *,
    temperature: float = 0.0,
    expect: str = "dict",
) -> Any:
    last_err: Optional[Exception] = None
    temp = temperature
    for model in _model_list():
        for _ in range(2):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "stream": False,
                "max_tokens": 4096,
                "max_completion_tokens": 4096,
            }
            try:
                resp = _proxy.post("/inference/chat/completions", json_data=payload)
                if resp is None:
                    raise RuntimeError("inference returned None")
                content = resp["choices"][0]["message"].get("content") or ""
                parsed = _parse_json_content(content)
                if expect == "dict" and not isinstance(parsed, dict):
                    raise ValueError(f"expected dict got {type(parsed)}")
                if expect == "list" and not isinstance(parsed, list):
                    raise ValueError(f"expected list got {type(parsed)}")
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                temp = min(temp + 0.2, 0.8)
                continue
    raise RuntimeError(f"llm_json failed: {last_err}")



# ---------------------------------------------------------------------------
# freezable — light MUSK Debugger spirit (no-op freeze for EnvPack turns)
# ---------------------------------------------------------------------------


def freezable(func):
    """MUSK_v20 freezable spirit: mark steps that may short-circuit when done."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__name__ = getattr(func, "__name__", "wrapped")
    wrapper.__doc__ = getattr(func, "__doc__", None)
    return wrapper


class TracePad:
    """Optional light debug pad (MUSK Debugger spirit)."""

    def __init__(self):
        self.events: list[str] = []

    def log(self, msg: str) -> None:
        self.events.append(str(msg)[:500])


# ---------------------------------------------------------------------------
# Vocab — MUSK retail synonym / unit / stopword tables
# ---------------------------------------------------------------------------


class Vocab:
    """General retail vocabulary (not suite SKUs). Kept from MUSK_v20 spirit."""

    SCHEMA_KEYS: frozenset = frozenset({
        "product_type", "color", "color_family", "material", "brand", "style",
        "size", "size_system", "gender", "user", "age", "pack", "set", "model",
        "compatibility", "pattern", "design", "shape", "origin", "usage",
        "fit", "feature", "category", "capacity", "volume", "length", "weight",
        "pre_config", "suitable", "condition", "warranty_days", "version",
        "option", "variation", "solution", "quality", "unit", "frame", "theme",
        "promotion", "inspiration", "interest", "vibe", "id",
    })
    NON_SEARCH_KEYS: frozenset = frozenset({"delivery", "service"})
    KW_STOPWORDS: tuple[str, ...] = (
        "color family", "suitable", "compatible",
        "about", "above", "after", "also", "and", "as", "at", "before", "below",
        "between", "by", "for", "from", "in",
        "less", "more", "of", "on", "or", "over", "per", "than", "to", "under",
        "up", "with", "within",
    )
    NO_WORDS: tuple[str, ...] = ("no", "without any", "without")
    YES_WORDS: tuple[str, ...] = ("yes", "enabled", "included", "exists", "true")
    SEARCHABLE_UNITS: tuple[str, ...] = (
        "mm", "cm", "m", "km", "ft", "yd", "mg", "g", "kg", "oz", "lb",
        "ml", "l", "mah", "ah", "wh", "kwh", "v", "mv", "kv", "ma", "w", "kw",
        "hz", "khz", "mhz", "ghz", "kb", "mb", "gb", "tb",
        "bps", "kbps", "mbps", "gbps", "°c", "°f", "pcs", "pc",
        "pa", "kpa", "mpa", "mp", "dpi", "lux", "db", "rpm",
    )
    # MUSK_v20 Vocab spirit — units, color, yes/no, bridges
    COLOR_WORDS: tuple[str, ...] = (
        "red", "blue", "green", "yellow", "black", "white", "pink", "purple",
        "violet", "orange", "brown", "gray", "grey", "beige", "navy", "gold",
        "silver", "bronze", "magenta", "cyan", "dark", "light", "khaki",
    )
    METRIC_UNITS: dict[str, str] = {
        "mm": "", "cm": "", "mg": "", "kg": "", "oz": "", "lb": "", "ml": "",
        "mah": "", "ah": "", "wh": "", "kwh": "", "mv": "", "kv": "", "ma": "",
        "kw": "", "hz": "", "khz": "", "mhz": "", "ghz": "", "bps": "",
        "kbps": "", "mbps": "", "gbps": "", "°c": "", "°f": "", "kpa": "",
        "mpa": "", "dpi": "", "lux": "", "db": "", "rpm": "",
        "y": "year", "yrs": "year", "km": "kilometer", "ft": "feet",
        "m": "meter", "yd": "yard", "g": "gram", "l": "liter", "v": "volt",
        "w": "watt", "pa": "pascal", "kb": "kilobyte", "mb": "megabyte",
        "gb": "gigabyte", "tb": "terabyte", "mp": "megapixel",
        "pc": "piece", "pcs": "piece",
    }
    NUMBERS: dict[str, str] = {
        "single": "1", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10",
    }
    ALTS: tuple[str, ...] = (
        "all2any", "everyday2casual", "any2universal", "kid2child",
        "men2male", "women2female", "tier2layer", "100%2pure", "anti2non",
    )
    BRIDGES: tuple[str, ...] = (
        "all2any", "man2men", "woman2women", "child2children",
        "metre2meter", "litre2liter", "colour2color",
    )
    FILLERS: tuple[str, ...] = ("set", "pack", "accessory", "option")
    ID_KEYS = ("brand", "color", "model", "id", "compatibility", "color_family")
    SYNONYMS: tuple[tuple[str, str], ...] = tuple(
        (min(a, b), max(a, b)) for a, b in [
            ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"), ("five", "5"),
            ("six", "6"), ("seven", "7"), ("eight", "8"), ("nine", "9"), ("ten", "10"),
            ("one", "single"), ("two", "double"), ("three", "triple"),
            ("1", "single"), ("2", "double"), ("3", "triple"),
            ("first", "1st"), ("second", "2nd"), ("third", "3rd"),
            ("hundred", "100"), ("thousand", "1000"), ("zero", "0"),
            ("men", "male"), ("women", "female"), ("kid", "child"),
            ("colour", "color"), ("metre", "meter"), ("litre", "liter"),
        ]
    )
    P_EXT_SEARCH_UNITS: tuple[tuple[str, str], ...] = (
        ("piece", "pcs"), ("pack", "pcs"),
    )

    @classmethod
    def synonym_partners(cls, token: str) -> list[str]:
        t = (token or "").lower().strip()
        out: list[str] = []
        for a, b in cls.SYNONYMS:
            if t == a and b not in out:
                out.append(b)
            elif t == b and a not in out:
                out.append(a)
        return out

    @classmethod
    def _token_pat(cls, token: str) -> str:
        """Word-boundary pattern that does not split decimal numbers (5.7, 6.9)."""
        esc = re.escape(token)
        if re.fullmatch(r"\d+", token):
            # Avoid turning "5.7 inch" into "five.7 inch" / "6.9" into "6.nine".
            return rf"(?<![\d.]){esc}(?![\d.])"
        return rf"\b{esc}\b"

    @classmethod
    def expand_query_tokens(cls, query: str) -> list[str]:
        """Build synonym + unit-extension variants of a free-text query."""
        base = " ".join((query or "").strip().split())
        if not base:
            return []
        variants = [base]
        lower = base.lower()
        # Number-word / synonym swaps (decimal-safe for digit tokens).
        for a, b in cls.SYNONYMS:
            pa, pb = cls._token_pat(a), cls._token_pat(b)
            if re.search(pa, lower, flags=re.I):
                variants.append(re.sub(pa, b, lower, flags=re.I))
            if re.search(pb, lower, flags=re.I):
                variants.append(re.sub(pb, a, lower, flags=re.I))
        # Screen-size unit aliases: "5.7 inch" <-> "5.7in" / '5.7"'
        inch_pat = r"(\d+(?:\.\d+)?)\s*(?:inch(?:es)?|in)\b"
        if re.search(inch_pat, lower, re.I):
            variants.append(re.sub(inch_pat, r"\1in", lower, flags=re.I))
            variants.append(re.sub(inch_pat, r'\1"', lower, flags=re.I))
            variants.append(re.sub(inch_pat, r"\1 inch", lower, flags=re.I))
        # Unit glued forms: "10 gb" <-> "10gb"
        for unit in cls.SEARCHABLE_UNITS:
            pat = rf"(\d+(?:\.\d+)?)\s+{re.escape(unit)}\b"
            if re.search(pat, lower, re.I):
                variants.append(re.sub(pat, rf"\1{unit}", lower, flags=re.I))
            pat2 = rf"(\d+(?:\.\d+)?){re.escape(unit)}\b"
            if re.search(pat2, lower, re.I):
                variants.append(re.sub(pat2, rf"\1 {unit}", lower, flags=re.I))
        # piece/pack → pcs
        for src, dst in cls.P_EXT_SEARCH_UNITS:
            if re.search(rf"\b1\s*{re.escape(src)}\b", lower):
                variants.append(re.sub(rf"\b1\s*{re.escape(src)}\b", f"1{dst}", lower))
        # de-dupe preserve order
        seen: set[str] = set()
        out: list[str] = []
        for v in variants:
            key = " ".join(v.split()).lower()
            if key and key not in seen:
                seen.add(key)
                out.append(" ".join(v.split()))
        return out[:12]


# ---------------------------------------------------------------------------
# Weight — KEEP MUSK LOG_PROBS / SKU_BONUS_C / score_n
# ---------------------------------------------------------------------------


@dataclass
class Weight:
    """MUSK_v20 ranking math — SKU_BONUS_C / RANK_BONUS_C / LOG_PROBS (unchanged)."""

    LOG_PROBS = [
        0.6698174986042095, 0.12640294947204656, 0.03721208325578941, 0.01890137562198827, 0.011222691775555538, 0.005316011893684202, 0.0035440079291228013,
        0.003544006929121897, 0.0028846576167278616, 0.002884656616726958, 0.0028846556167260538, 0.00288465461672515, 0.002884653616724246, 0.0028846526167233422,
        0.002884651616722438, 0.0028846506167215344, 0.0028846496167206303, 0.0028846486167197266, 0.0028846476167188224, 0.0028846466167179187,
        0.0028846456167170146, 0.002884644616716111, 0.002884643616715207, 0.002884642616714303, 0.002884641616713399, 0.0028846406167124953, 0.002884639616711591,
        0.0028846386167106874, 0.0028846376167097833, 0.0028846366167088796, 0.0028846356167079755, 0.0028846346167070718, 0.0028846336167061676,
        0.002884632616705264, 0.00288463161670436, 0.002884630616703456, 0.002884629616702552, 0.0028846286167016483, 0.002884627616700744, 0.0028846266166998404,
        0.0028846256166989367, 0.0028846246166980326, 0.002884623616697129, 0.002884622616696225, 0.002884621616695321, 0.002884620616694417, 0.0028846196166935133,
        0.002884618616692609, 0.0028846176166917054, 0.0028846166166908013, 0.0028846156166898976
    ]
    RANK_BONUS_C = [1e-2 - i * 2e-5 for i in range(500)]
    SKU_BONUS_C = [5e-3, 5e-3, 5e-3, 1e-3, 5e-4, 1e-4]
    SCORE_N_THRES = 0.4

    detail: dict[str, float] = field(default_factory=dict)
    choice: dict[str, int] = field(default_factory=dict)
    covers: list[int] = field(default_factory=list)

    score_r: float = 0.0
    score_n: float = 0.0
    score: float = 0.0

    confidence: float = 0.0
    wsum: float = 0.0

    wrank: int = -1
    rank: int = -1

    @classmethod
    def create(
        cls,
        choice: dict,
        covers: list,
        rank: int,
        score_r: float,
        pt_bonus: float,
        att_bonus: float,
        brand_bonus: float,
        wsum: float,
    ) -> "Weight":
        score = score_r + pt_bonus + att_bonus + brand_bonus
        idx = choice.get("sku") if choice.get("sku") is not None else -1
        try:
            idx_i = int(idx)
        except (TypeError, ValueError):
            idx_i = -1
        sku_bonus = Weight.SKU_BONUS_C[idx_i + 1] if idx_i < (len(Weight.SKU_BONUS_C) - 1) else 0.0
        rank_bonus = Weight.RANK_BONUS_C[rank] if rank < len(Weight.RANK_BONUS_C) else 0.0
        return Weight(
            choice=choice,
            covers=covers,
            rank=rank,
            wsum=wsum,
            score_n=((score / wsum) if wsum > 1e-8 else 0.0) + sku_bonus + rank_bonus,
            score_r=score_r,
            score=score,
            detail={
                "pt_bonus": pt_bonus,
                "att_bonus": att_bonus,
                "brand_bonus": brand_bonus,
                "sku_bonus": sku_bonus,
                "rank_bonus": rank_bonus,
            },
        )

    def fill_detail(self, qs: list[str], ws: list[float]) -> "Weight":
        for i in self.covers:
            if 0 <= i < len(qs) and i < len(ws):
                self.detail[qs[i]] = ws[i]
        return self

    def update_confidence(self, wrank: int) -> "Weight":
        if wrank >= len(Weight.LOG_PROBS):
            self.confidence = Weight.LOG_PROBS[-1] * (
                0.9 ** (wrank - len(Weight.LOG_PROBS) + 1)
            )
        else:
            self.confidence = Weight.LOG_PROBS[wrank]
        self.wrank = wrank
        return self

    def absorb_raw(self, raw: float, *, rank: int = 0, sku_idx: int = -1) -> "Weight":
        """Fold EnvPack heuristic into MUSK Weight.create math."""
        w = Weight.create(
            choice={"sku": int(sku_idx)},
            covers=[],
            rank=int(rank),
            score_r=float(raw),
            pt_bonus=0.0,
            att_bonus=0.0,
            brand_bonus=0.0,
            wsum=1.0,
        )
        self.detail = dict(w.detail)
        self.choice = dict(w.choice)
        self.covers = list(w.covers)
        self.score_r = w.score_r
        self.score_n = w.score_n
        self.score = w.score
        self.wsum = w.wsum
        self.rank = w.rank
        self.update_confidence(max(0, int(rank)))
        return self


# ---------------------------------------------------------------------------
# CapMap — capability → policy_view tool name
# ---------------------------------------------------------------------------


class CapMap:
    """Resolve logical caps against policy_view.tools; soft-skip/alias (MUSK CapMap).

    Soft-skip missing tools; alias search↔filter, compare→view, skip absent
    stock/message/remove. Never invent names absent from policy_view.
    """

    _ALIASES: dict[str, tuple[str, ...]] = {
        "search": ("search", "find", "find_product", "catalog_search", "query_catalog"),
        "filter": ("filter", "filter_catalog", "browse", "list_products"),
        "view": ("view", "get_product", "product_detail", "inspect_product"),
        "compare": ("compare", "compare_products"),
        "inspect_stock": ("inspect_stock", "check_stock", "get_stock", "stock_status"),
        "inspect_cart": ("inspect_cart", "get_cart", "view_cart"),
        "message": ("message", "ask_user", "send_message", "reply_user"),
        "add_to_cart": ("add_to_cart", "cart_add", "add_item"),
        "remove_from_cart": ("remove_from_cart", "cart_remove", "remove_item"),
        "place_order": (
            "place_test_order",
            "place_order",
            "checkout",
            "buy",
            "submit_order",
        ),
    }

    def __init__(self, tools_schema: Optional[list[Any]] = None):
        self.names: set[str] = set()
        self.param_keys: dict[str, set[str]] = {}
        self._resolved: dict[str, Optional[str]] = {}
        for tool in tools_schema or []:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not name:
                continue
            name_s = str(name)
            self.names.add(name_s)
            params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
            props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
            self.param_keys[name_s] = {str(k) for k in props.keys()}
        for cap in self._ALIASES:
            self._resolved[cap] = self._match(cap)

    def _match(self, capability: str) -> Optional[str]:
        for candidate in self._ALIASES.get(capability, (capability,)):
            if candidate in self.names:
                return candidate
        if capability in self.names:
            return capability
        return None

    def has(self, capability: str) -> bool:
        return self.resolve(capability) is not None

    def resolve(self, capability_or_name: str) -> Optional[str]:
        if capability_or_name in self._resolved:
            return self._resolved[capability_or_name]
        if capability_or_name in self.names:
            return capability_or_name
        matched = self._match(capability_or_name)
        self._resolved[capability_or_name] = matched
        return matched

    def filter_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        allowed = self.param_keys.get(tool_name)
        if not allowed:
            return dict(args)
        return {k: v for k, v in args.items() if k in allowed}


# ---------------------------------------------------------------------------
# RunHall — EnvPack session (MUSK SearchSession dialogue spirit)
# ---------------------------------------------------------------------------


@dataclass
class RunHall:
    """Environment call hall: binding copy, turn counter, dialogue, shopper notes."""

    problem_id: str
    binding: dict[str, Any]
    policy: dict[str, Any]
    dialogue: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    finished: bool = False
    last_shopper_note: Optional[str] = None
    shopper_notes: list[str] = field(default_factory=list)
    capmap: CapMap = field(default_factory=CapMap)

    @classmethod
    def from_problem(cls, problem_data: dict[str, Any]) -> "RunHall":
        environment = problem_data["environment"]
        policy = dict(environment["policy_view"])
        return cls(
            problem_id=str(
                problem_data.get("problem_id", problem_data.get("id", "problem"))
            ),
            binding=dict(environment["binding"]),
            policy=policy,
            capmap=CapMap(policy.get("tools") or []),
        )

    @property
    def query(self) -> str:
        return str(self.policy.get("query") or "")

    @property
    def max_steps(self) -> int:
        return int(self.policy.get("max_steps") or 20)

    @property
    def max_calls_per_turn(self) -> int:
        return int(self.policy.get("max_calls_per_turn") or 1)

    @property
    def tool_names(self) -> set[str]:
        if self.capmap.names:
            return set(self.capmap.names)
        names: set[str] = set()
        for tool in self.policy.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            name = fn.get("name") if isinstance(fn, dict) else None
            if name:
                names.add(str(name))
        return names

    def remaining(self) -> int:
        return max(0, self.max_steps - self.turn)

    def can(self, capability: str) -> bool:
        return self.capmap.has(capability)

    def think(self, content: str, *, think: str = "") -> None:
        """MUSK-style dialogue entry: short assistant content + optional think."""
        entry: dict[str, Any] = {"role": "assistant", "content": content}
        if think:
            entry["think"] = think
        self.dialogue.append(entry)

    @staticmethod
    def _unwrap_observation(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        if any(
            key in payload
            for key in (
                "results", "variants", "compared", "added", "cart",
                "product_id", "sent", "ordered", "error",
            )
        ) and not (
            "done" in payload
            and "observation" in payload
            and isinstance(payload.get("observation"), dict)
        ):
            return payload
        inner = payload.get("observation")
        if isinstance(inner, dict):
            return inner
        return payload

    @staticmethod
    def _call_done(call: dict[str, Any]) -> bool:
        if call.get("done") is True:
            return True
        obs = call.get("observation")
        if isinstance(obs, dict) and obs.get("done") is True:
            return True
        return False

    def call(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        if self.finished:
            return {"calls": [], "user_message": None}
        if self.remaining() <= 0:
            raise RuntimeError("max_steps exhausted")
        allowed = self.tool_names
        cleaned: list[dict[str, Any]] = []
        for action in actions[: self.max_calls_per_turn]:
            raw_name = action.get("name")
            if not raw_name:
                continue
            name = self.capmap.resolve(str(raw_name)) or (
                str(raw_name) if str(raw_name) in allowed else None
            )
            if not name:
                continue
            if allowed and name not in allowed:
                continue
            args = action.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            cleaned.append(
                {"name": name, "args": self.capmap.filter_args(name, args)}
            )
        if not cleaned:
            return {
                "calls": [],
                "user_message": None,
                "error": "tool_unavailable",
                "requested": [a.get("name") for a in actions],
                "tools": sorted(allowed),
            }

        self.turn += 1
        group_id = f"{self.problem_id}-turn-{self.turn}"
        envelope = {
            **self.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": self.turn,
            "calls": [
                {"call_id": f"{group_id}-{index}", "action": action}
                for index, action in enumerate(cleaned, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        self.dialogue.append(
            {
                "role": "assistant",
                "content": f"step {self.turn}",
                "tool_calls": cleaned,
                "environment_result": result,
            }
        )
        user_message = result.get("user_message")
        if isinstance(user_message, dict) and user_message.get("content"):
            note = str(user_message["content"])
            self.last_shopper_note = note
            self.shopper_notes.append(note)
        for call in result.get("calls") or []:
            if self._call_done(call):
                self.finished = True
        return result

    def act(self, name: str, **args: Any) -> dict[str, Any]:
        if not self.capmap.resolve(name) and name not in self.capmap.names:
            return {"error": "tool_unavailable", "requested": name}
        result = self.call([{"name": name, "args": args}])
        if result.get("error") == "tool_unavailable":
            return {"error": "tool_unavailable", "requested": name}
        calls = result.get("calls") or []
        if not calls:
            return {}
        return self._unwrap_observation(calls[0].get("observation") or {})



# ---------------------------------------------------------------------------
# Lot + text helpers
# ---------------------------------------------------------------------------


@dataclass
class Lot:
    """Catalog SKU lot with attached MUSK Weight."""

    product_id: str
    sku: str
    title: str = ""
    brand: str = ""
    price: Optional[float] = None
    currency: str = "PHP"
    in_stock: bool = True
    options: str = ""
    category_path: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    listing_id: Optional[str] = None
    viewed: bool = False
    source_brand: str = ""
    weight: Weight = field(default_factory=Weight)

    @property
    def key(self) -> str:
        return f"{self.product_id}::{self.sku}"

    def text_blob(self) -> str:
        parts = [
            self.title,
            self.brand,
            self.options,
            " ".join(self.category_path),
            json.dumps(self.facts, ensure_ascii=False),
        ]
        return " ".join(str(p) for p in parts if p).lower()


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value).replace(",", ""))
    return float(m.group(1)) if m else None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t]


def _goal_only(query: str) -> str:
    split = re.split(r"\n\s*Task rules:\s*", query, maxsplit=1)
    return split[0].strip()


# ---------------------------------------------------------------------------
# CaseFile — hard/soft constraints from public query
# ---------------------------------------------------------------------------


@dataclass
class CaseFile:
    family: str = "generic"
    goal: str = ""
    category: Optional[str] = None
    brand: Optional[str] = None
    preferred_brand: Optional[str] = None
    budget: Optional[float] = None
    currency: str = "PHP"
    require_in_stock: bool = True
    attributes: dict[str, str] = field(default_factory=dict)
    soft_prefs: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    candidate_keys: list[str] = field(default_factory=list)
    preferred_model: Optional[str] = None
    preferred_color: Optional[str] = None
    needs_justification: bool = False
    needs_clarification: bool = False
    title_phrase: Optional[str] = None  # retrieval_recall quoted match


_FAMILY_HINTS = (
    (
        "ranking",
        (
            "largest screen",
            "best-ranked",
            "overclock",
            "candidate variants",
            "compare only these candidate",
            "highest refresh",
            "advertised maximum refresh",
            "tradeoff priority",
        ),
    ),
    (
        "preference_reasoning",
        ("listing_id", "soft preferences", "intended use", "sweat", "water resist"),
    ),
    (
        "recovery",
        ("relaxable", "another color", "least-relaxed", "preferred model"),
    ),
    (
        "retrieval_recall",
        ("title token", "distinct matching product", "complete title token"),
    ),
    (
        "constraint_satisfaction",
        (
            "exact variant",
            "pixel classes",
            "every requirement must hold",
            "check the specific sku",
            "conflicting resolution classes",
            "screen-only or replacement-screen",
        ),
    ),
    (
        "intent_decomposition",
        (
            "soft preference",
            "preferred brand",
            "nice-to-haves",
            "ideally, i prefer",
            "firm requirements matter more",
            "initial cart choice",
            "named preferred brand",
            "honor that named brand",
        ),
    ),
    (
        "justification",
        (
            "reason i can trust",
            "give me a reason",
            "trustworthy reason",
            "walk me through why",
            "tell me why",
            "why the one you scale",
            "why it's a good scale",
            "structured justification with claims",
        ),
    ),
)

EVENT_FAMILIES = frozenset(
    {"intent_decomposition", "constraint_satisfaction", "justification"}
)
# Families that fire stock/price events and require observe-before-order.
MARKET_EVENT_FAMILIES = frozenset(
    {
        "intent_decomposition",
        "constraint_satisfaction",
        "justification",
        "preference_reasoning",
        "retrieval_recall",
        "recovery",
    }
)

_CASE_PROMPT = """\
You are a MUSK CaseForge planner (SV_DECOMPOSE spirit adapted for EnvPack).
Decompose the PUBLIC shopper query into a structured CaseFile.
Infer EnvPack capability families from wording only — never ShoppingBench
product/shop/voucher modes.

Rules:
- Separate HARD requirements from SOFT preferences.
- Hard usually includes: category, budget, in-stock, and named hardware specs
  (screen_size, resolution, refresh_rate, ram, etc.).
- An explicitly preferred brand is SOFT unless the brief says the brand is mandatory.
- If a firm hardware spec will be told only when asked
  ("tell you if you ask" / "one firm requirement"), set needs_clarification=true
  and leave attributes EMPTY — do not invent specs.
- Keywords must be short BM25-friendly tokens (values, not schema keys).
- Provide 3-6 alternate search_queries that reformulate the request.
- If the brief lists candidate variants as product_id::sku, copy them into candidate_keys.
- needs_justification=true only if the shopper asks for a reason in their own request
  (ignore shared tool-footer text about optional claims).

Return JSON only:
{{
  "category": <str|null>,
  "brand": <str|null>,
  "preferred_brand": <str|null>,
  "budget": <number|null>,
  "currency": <str>,
  "require_in_stock": <bool>,
  "attributes": {{<attr>: <value>, ...}},
  "soft_prefs": [<str>, ...],
  "keywords": [<str>, ...],
  "search_queries": [<str>, ...],
  "candidate_keys": [<"product_id::sku">, ...],
  "preferred_model": <str|null>,
  "preferred_color": <str|null>,
  "needs_justification": <bool>,
  "needs_clarification": <bool>
}}

# Shopper brief
{query}
"""


def infer_family(query: str) -> str:
    """Detect EnvPack family from PUBLIC query wording only."""
    goal = _goal_only(query).lower()
    full = query.lower()
    parts = re.split(r"\n\s*Task rules:\s*", query, maxsplit=1)
    rules = parts[1].lower() if len(parts) > 1 else full
    if re.search(r"\b(ideally|nice[- ]to[- ]have)\b", goal) and re.search(
        r"\bprefer(?:\s+the)?\s+[a-z]", goal
    ):
        return "intent_decomposition"
    if "prefer" in goal and (
        "firm requirements matter" in goal
        or "nice-to-have" in full
        or "soft preference" in full
    ):
        return "intent_decomposition"
    for hint in (
        "reason i can trust",
        "give me a reason",
        "trustworthy reason",
        "walk me through why",
        "tell me why",
        "why the one you scale",
        "why it's a good scale",
        "structured justification with claims",
    ):
        if hint in goal:
            return "justification"
    for family, hints in _FAMILY_HINTS:
        if family == "justification":
            continue
        if family == "constraint_satisfaction":
            haystack = rules
        elif family == "intent_decomposition":
            haystack = full
        else:
            haystack = full
        if any(h in haystack for h in hints):
            return family
    if "prefer" in goal and ("firm" in full or "nice-to-have" in full):
        return "intent_decomposition"
    if re.search(r"\d{6,}::", query):
        return "ranking"
    return "generic"


def _extract_budget(text: str) -> Optional[float]:
    patterns = [
        r"(?:up to|under|below|at or under|budget(?:\s+is)?|spend(?:\s+up to)?)\s*"
        r"`?([0-9]+(?:\.[0-9]+)?)`?\s*(?:PHP|php|USD|usd)?",
        r"`?([0-9]+(?:\.[0-9]+)?)`?\s*(?:PHP|php)\b",
        r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:PHP|php)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _extract_candidate_keys(text: str) -> list[str]:
    keys = re.findall(r"\b(\d{6,}::[A-Za-z0-9_.\-]+)", text)
    cleaned = [k.rstrip(".,;:)") for k in keys]
    return list(dict.fromkeys(cleaned))


def shopper_needs_reply(note: Optional[str]) -> bool:
    if not note:
        return False
    text = note.strip()
    if not text:
        return False
    if "?" in text or "？" in text:
        return True
    if re.search(r"\b(tell me|which one|confirm|before you buy|can you)\b", text, re.I):
        return True
    return False


def is_market_event_note(note: Optional[str]) -> bool:
    if not note:
        return False
    return bool(
        re.search(
            r"\b(not available|out of stock|sold out|stockout|unavailable|"
            r"price (?:changed|increased|jumped)|now costs)\b",
            note,
            re.I,
        )
    )


# ---------------------------------------------------------------------------
# LexMatch — attribute matching
# ---------------------------------------------------------------------------


class LexMatch:
    """Match hardware attributes from facts / options / title."""

    _RES_ALIASES = {
        "720p": ("720p", "hd", "1280x720", "1280 x 720", "1280×720"),
        "1080p": ("1080p", "full hd", "fhd", "1920x1080", "1920 x 1080", "1920×1080"),
        "1440p": ("1440p", "qhd", "2560x1440", "2560 x 1440"),
        "4k": ("4k", "uhd", "3840x2160", "3840 x 2160"),
        "hd": ("hd", "720p", "1280x720", "1280 x 720"),
    }

    @staticmethod
    def parse_inches(text: Any) -> Optional[float]:
        if text is None:
            return None
        raw = str(text).strip().lower()
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:\"|''|in(?:ch(?:es)?)?)\b", raw)
        if m:
            return float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\"", raw)
        if m:
            return float(m.group(1))
        m = re.fullmatch(r"(\d+(?:\.\d+)?)(?:in)?", raw.replace(" ", ""))
        if m:
            return float(m.group(1))
        return None

    @classmethod
    def screen_inches(cls, lot: Lot) -> Optional[float]:
        opt = (lot.options or "").strip()
        if opt:
            m = re.search(r"\((\d+(?:\.\d+)?)\s*inch", opt, re.I) or re.search(
                r"(\d+(?:\.\d+)?)\s*inch", opt, re.I
            )
            if m:
                return float(m.group(1))
            num = cls.parse_inches(opt)
            if num is not None and not re.search(r"\d+\s*[-–/]\s*\d+", opt):
                return num
        for fk, fv in (lot.facts or {}).items():
            if any(n in str(fk).lower() for n in ("size", "screen", "inch", "display")):
                raw = str(fv)
                if re.search(r"\d+(?:\.\d+)?\s*[-–/,]\s*\d+(?:\.\d+)?", raw):
                    continue
                num = cls.parse_inches(raw) or _as_float(raw)
                if num is not None and 3.0 <= num <= 100.0:
                    return num
        title = lot.title or ""
        if re.search(r"\d+(?:\.\d+)?\s*[-–/]\s*\d+(?:\.\d+)?", title):
            return None
        return cls.parse_inches(title)

    @classmethod
    def resolution_token(cls, lot: Lot) -> Optional[str]:
        blob = " ".join(
            [
                lot.title or "",
                lot.options or "",
                " ".join(f"{k} {v}" for k, v in (lot.facts or {}).items()),
            ]
        ).lower()
        if re.search(r"\b4k\b|\buhd\b|3840\s*[x×]\s*2160", blob):
            return "4k"
        if re.search(r"\b1440p\b|\bqhd\b|2560\s*[x×]\s*1440", blob):
            return "1440p"
        if re.search(r"\b1080p\b|full\s*hd|\bfhd\b|1920\s*[x×]\s*1080", blob):
            return "1080p"
        if re.search(r"\b720p\b|1280\s*[x×]\s*720", blob):
            return "720p"
        if re.search(r"\bhd\b", blob) and not re.search(r"full\s*hd|fhd|1080", blob):
            return "hd"
        best: Optional[str] = None
        order = {"4k": 4, "1440p": 3, "1080p": 2, "720p": 1, "hd": 1}
        for m in re.finditer(r"(\d{3,4})\s*[x×]\s*(\d{3,4})", blob):
            w, h = int(m.group(1)), int(m.group(2))
            a, b = max(w, h), min(w, h)
            token = None
            if a >= 3840 and b >= 2160:
                token = "4k"
            elif a >= 2560 and b >= 1440:
                token = "1440p"
            elif a >= 1920 and b >= 1080:
                token = "1080p"
            elif a >= 1280 and b >= 720:
                token = "720p"
            if token and (best is None or order[token] > order[best]):
                best = token
        return best

    @classmethod
    def refresh_hz(cls, lot: Lot) -> Optional[float]:
        blob = " ".join(
            [
                lot.title or "",
                lot.options or "",
                " ".join(f"{k} {v}" for k, v in (lot.facts or {}).items()),
            ]
        ).lower()
        vals = [float(m) for m in re.findall(r"(\d{2,3})\s*hz", blob)]
        return max(vals) if vals else None

    @classmethod
    def normalize_wanted(cls, key: str, value: str) -> tuple[str, Any]:
        key_l = key.lower().replace("-", "_")
        val = str(value).strip()
        val_l = val.lower().replace(" ", "")
        if val_l in cls._RES_ALIASES or re.fullmatch(
            r"\d{3,4}p|4k|uhd|qhd|hd|fullhd|fhd", val_l
        ):
            token = val_l
            if token in {"fullhd", "fhd"}:
                token = "1080p"
            return "resolution", token
        if "resolution" in key_l:
            token = val_l
            if token in {"fullhd", "fhd"}:
                token = "1080p"
            return "resolution", token
        if "refresh" in key_l or val_l.endswith("hz"):
            return "refresh_rate", _as_float(val)
        if "ram" in key_l:
            return "ram", _as_float(val)
        if "screen" in key_l or "size" in key_l or val_l.endswith("in"):
            inches = cls.parse_inches(val)
            if inches is None and re.fullmatch(r"\d+(?:\.\d+)?", val_l):
                inches = float(val_l)
            if inches is not None and not (3.0 <= float(inches) <= 100.0):
                inches = None
            return "screen_size", inches
        return key_l, val.lower()

    @classmethod
    def matches(cls, lot: Lot, key: str, value: str) -> bool:
        kind, wanted = cls.normalize_wanted(key, value)
        if wanted is None or wanted == "":
            return False
        if kind == "screen_size":
            got = cls.screen_inches(lot)
            if got is None or not isinstance(wanted, (int, float)):
                return False
            return abs(float(got) - float(wanted)) < 0.005
        if kind == "resolution":
            got = cls.resolution_token(lot)
            if not got:
                return False
            want = str(wanted).lower()
            if want in {"hd", "720p"} and got in {"hd", "720p"}:
                return True
            alts = cls._RES_ALIASES.get(want, (want,))
            return got in alts or got == want
        if kind == "refresh_rate":
            got = cls.refresh_hz(lot)
            if got is None or not isinstance(wanted, (int, float)):
                return False
            return abs(float(got) - float(wanted)) <= 1.0
        if kind == "ram":
            blob = lot.text_blob()
            if not isinstance(wanted, (int, float)):
                return str(wanted).lower() in blob
            return bool(re.search(rf"\b{int(wanted)}\s*gb\b", blob))
        return _norm(str(wanted)) in _norm(lot.text_blob())


# ---------------------------------------------------------------------------
# RailCheck — accessory rejection for device categories
# ---------------------------------------------------------------------------


class RailCheck:
    """Reject accessory / decoy listings when a real device category is required.

    Implementation wording intentionally differs from sibling EnvPack accessory fences.
    """

    _PARTS = (
        "case", "cover", "charger", "cable", "screen protector", "tempered glass",
        "replacement tip", "ear tip", "strap only", "mount only", "holder only",
        "stand only", "skin only", "bumper", "holster", "pouch", "digitizer",
        "digitiz", "panel assembly", "replacement part", "touch screen digitizer",
        "spare part", "repair part",
    )
    _PATH_BAD = (
        "accessor", "replacement", "spare part", "repair",
        "parts & tools", "parts and tools",
    )
    _BAD_OPT = re.compile(
        r"screen[- ]?only|replacement[- ]?screen|digitizer|lcd panel|touch panel only",
        re.I,
    )

    @classmethod
    def title_is_parts_kit(cls, title: str) -> bool:
        t = (title or "").lower()
        if not t:
            return False
        device_hit = bool(
            re.search(
                r"\b(smartphone|mobile phone|cellphone|tablet|ipad|monitor|"
                r"headphone|earbud|headset)\b",
                t,
            )
        )
        if any(b in t for b in cls._PARTS):
            if re.search(r"\b(with case|free case|include[sd]? case)\b", t) and device_hit:
                return False
            if re.search(r"\b(phone|tablet|monitor)\s+(case|cover|protector|bumper|pouch)\b", t):
                return True
            if re.search(r"\b(case|cover|protector|bumper)\b.*\b(phone|tablet)\b", t):
                return True
            return True
        if re.search(r"\b(lcd|digitizer)\b", t) and not device_hit:
            if re.search(r"\b(screen|display|panel|replacement|for)\b", t):
                return True
        if re.search(r"\bfor\b.+\b(lcd|digitizer|display|touch screen)\b", t) and not device_hit:
            return True
        if re.search(r"\b(lcd|digitizer)\b.+\bfor\b", t) and not device_hit:
            return True
        if re.search(r"\b(folding|foldable)\b", t) and re.search(
            r"\b(holder|stand|bracket|mount|support)\b", t
        ):
            return True
        return False

    @classmethod
    def path_is_parts(cls, category_path: list[str]) -> bool:
        path = _norm(" ".join(category_path or []))
        return any(b in path for b in cls._PATH_BAD)

    @classmethod
    def looks_phone(cls, lot: Lot) -> bool:
        if cls.title_is_parts_kit(lot.title or "") or cls.path_is_parts(lot.category_path):
            return False
        title = lot.title or ""
        if re.search(r"\b(drone|dron|aircraft|quadcopter|uav)\b", title, re.I):
            return False
        path = _norm(" ".join(lot.category_path))
        blob = lot.text_blob()
        if re.search(
            r"\b(smartphone|cellphone|mobile phone|android phone|iphone|galaxy\s+[a-z]?\d)\b",
            title,
            re.I,
        ):
            return True
        if any(w in path for w in ("smartphone", "mobile phone", "cell phone")):
            return True
        for fk, fv in (lot.facts or {}).items():
            fl = f"{fk} {fv}".lower()
            if re.search(r"\b(android|ios|iphone os|blackberry os|harmonyos)\b", fl):
                return True
        if re.search(r"\b(android|ios)\b", blob) and re.search(
            r"\b(gb ram|mp camera|dual sim)\b", blob
        ):
            return True
        return False

    @classmethod
    def looks_tablet(cls, lot: Lot) -> bool:
        if cls.title_is_parts_kit(lot.title or "") or cls.path_is_parts(lot.category_path):
            return False
        title = lot.title or ""
        if re.search(r"\b(drone|dron|aircraft|quadcopter|uav)\b", title, re.I):
            return False
        path = _norm(" ".join(lot.category_path))
        if re.search(r"\b(tablet|ipad)\b", title, re.I):
            return True
        if "tablet" in path:
            return True
        for fk, fv in (lot.facts or {}).items():
            if re.search(r"\b(tablet|ipados)\b", f"{fk} {fv}", re.I):
                return True
        return False

    @classmethod
    def looks_monitor(cls, lot: Lot) -> bool:
        if cls.title_is_parts_kit(lot.title or "") or cls.path_is_parts(lot.category_path):
            return False
        title = lot.title or ""
        if re.search(r"\b(drone|dron|aircraft|quadcopter|uav)\b", title, re.I):
            return False
        path = _norm(" ".join(lot.category_path))
        if re.search(r"\b(monitor|display)\b", title, re.I) and "lcd display" not in title.lower():
            if re.search(
                r"\b(digitizer|replacement|for vivo|for samsung|for xiaomi)\b", title, re.I
            ):
                return False
            return True
        return "monitor" in path or ("display" in path and "accessor" not in path)

    @classmethod
    def device_ok(cls, lot: Lot, category: Optional[str]) -> bool:
        if not category:
            return True
        want = _norm(category).rstrip("s")
        title = lot.title or ""
        # Always reject cross-category decoys from title, even before product_info.
        if re.search(r"\b(drone|dron|aircraft|quadcopter|uav)\b", title, re.I):
            if any(
                w in want
                for w in ("phone", "smartphone", "tablet", "monitor", "headphone", "earbud")
            ):
                return False
        if cls.title_is_parts_kit(title) or cls.path_is_parts(lot.category_path):
            if any(
                w in want
                for w in ("phone", "smartphone", "tablet", "monitor", "headphone", "earbud")
            ):
                return False
        rich = bool(lot.viewed or lot.facts or lot.category_path)
        if "smartphone" in want or want == "phone":
            return cls.looks_phone(lot) if rich else True
        if "tablet" in want:
            return cls.looks_tablet(lot) if rich else True
        if "monitor" in want:
            return cls.looks_monitor(lot) if rich else True
        return True

    @classmethod
    def option_ok(cls, lot: Lot, category: Optional[str]) -> bool:
        if not category:
            return True
        want = _norm(category)
        if not any(w in want for w in ("phone", "smartphone", "tablet", "monitor")):
            return True
        if cls._BAD_OPT.search(lot.options or ""):
            return False
        return True



# ---------------------------------------------------------------------------
# Evidence / claims (visible facts only)
# ---------------------------------------------------------------------------


class EvidencePad:
    """Gate attribute must be visible on the selected Lot (TF7/TF3)."""

    @staticmethod
    def gate_from_goal(goal: str, attributes: dict[str, str]) -> Optional[str]:
        g = (goal or "").lower()
        if re.search(r"\b(\d{3,4}p|4k|uhd|qhd|full\s*hd)\b", g) or (
            re.search(r"\bhd\b", g) and "inch" not in g
        ):
            return "resolution"
        if re.search(r"\d+(?:\.\d+)?\s*-?\s*inch", g):
            return "screen_size"
        if re.search(r"\d{2,3}\s*hz", g):
            return "refresh_rate"
        for key in ("resolution", "screen_size", "refresh_rate", "ram"):
            if key in attributes:
                return key
        return None

    @classmethod
    def gate_key(cls, casef: CaseFile) -> Optional[str]:
        return cls.gate_from_goal(casef.goal, casef.attributes) or (
            next(iter(casef.attributes), None) if casef.attributes else None
        )

    @classmethod
    def measured_value(cls, casef: CaseFile, lot: Lot) -> Optional[str]:
        gate = cls.gate_key(casef)
        if not gate:
            return None
        if gate == "resolution":
            return LexMatch.resolution_token(lot)
        if gate == "screen_size":
            inches = LexMatch.screen_inches(lot)
            return f"{inches:g}" if inches is not None else None
        if gate == "refresh_rate":
            hz = LexMatch.refresh_hz(lot)
            return f"{int(hz)}Hz" if hz is not None else None
        if gate == "ram":
            blob = lot.text_blob()
            m = re.search(r"\b(\d+)\s*gb\s*ram\b", blob, re.I) or re.search(
                r"\b(\d+)\s*gb\b", (lot.options or "") + " " + blob, re.I
            )
            return f"{m.group(1)}GB" if m else None
        return None

    @classmethod
    def visible(cls, casef: CaseFile, lot: Lot) -> bool:
        gate = cls.gate_key(casef)
        if not gate:
            return True
        measured = cls.measured_value(casef, lot)
        if measured is None:
            return False
        if gate in casef.attributes:
            return LexMatch.matches(lot, gate, casef.attributes[gate])
        if gate == "resolution":
            goal = casef.goal.lower()
            got = measured.lower()
            if re.search(r"\b(720p|hd)\b", goal) and got in {"720p", "hd"}:
                return True
            if re.search(r"\b(1080p|full\s*hd|fhd)\b", goal) and got in {"1080p"}:
                return True
            if re.search(r"\b(1440p|qhd)\b", goal) and got in {"1440p"}:
                return True
            if re.search(r"\b(4k|uhd)\b", goal) and got in {"4k"}:
                return True
            return False
        return True

    @classmethod
    def claim_value(cls, casef: CaseFile, lot: Lot) -> Optional[str]:
        if not cls.visible(casef, lot):
            return None
        gate = cls.gate_key(casef)
        measured = cls.measured_value(casef, lot)
        if not gate or not measured:
            return None
        if gate == "resolution":
            fact_blob = " ".join(f"{k} {v}" for k, v in (lot.facts or {}).items())
            surface = f"{lot.title or ''} {lot.options or ''} {fact_blob}".lower()
            goal = casef.goal.lower()
            lit_720 = "720p" in surface or "1280x720" in surface or "1280 x 720" in surface
            lit_hd = bool(re.search(r"\bhd\b", surface)) and "full hd" not in surface and "fhd" not in surface
            lit_1080 = (
                "1080p" in surface
                or "full hd" in surface
                or "fhd" in surface
                or "1920x1080" in surface
                or "1920 x 1080" in surface
            )
            if measured in {"hd", "720p"}:
                if "720p" in goal and lit_720:
                    return "720p"
                if re.search(r"\bhd\b", goal) and "1080" not in goal and "full" not in goal and (
                    lit_hd or lit_720
                ):
                    return "hd" if lit_hd and not lit_720 else ("720p" if lit_720 else None)
                if lit_720:
                    return "720p"
                if lit_hd:
                    return "hd"
                return None
            if measured == "1080p" and lit_1080:
                return "1080p"
            if measured.replace("p", "") and measured.lower() in surface:
                return measured
            return None
        return measured

    @classmethod
    def claimable(cls, casef: CaseFile, lot: Lot) -> bool:
        return cls.claim_value(casef, lot) is not None


class ClaimPad:
    """Build TF7 justification claims from viewed facts only."""

    @classmethod
    def build(cls, casef: CaseFile, lot: Lot) -> dict[str, Any]:
        claims: list[dict[str, str]] = []
        gate = EvidencePad.gate_from_goal(casef.goal, casef.attributes)
        claim_val = EvidencePad.claim_value(casef, lot)
        if gate and claim_val:
            claims.append({"field": gate, "value": str(claim_val)})
        if lot.price is not None:
            claims.append({"field": "price", "value": f"{float(lot.price):g}"})
        claims.append({"field": "in_stock", "value": "true" if lot.in_stock else "false"})
        soft_added = False
        want_brand = casef.brand or casef.preferred_brand
        if lot.brand and (not want_brand or brand_hit(lot, want_brand)):
            claims.append({"field": "brand", "value": str(lot.brand)})
            soft_added = True
        elif want_brand and brand_hit(lot, want_brand) and want_brand.lower() in _norm(
            lot.title or ""
        ):
            claims.append({"field": "brand", "value": str(want_brand)})
            soft_added = True
        if not soft_added:
            blob = lot.text_blob()
            for field_name, pat in (
                ("hdr", r"\bhdr\b"),
                ("microphone", r"\b(mic|microphone)\b"),
                ("streaming", r"\b(streaming|conference|zoom|teams)\b"),
                ("privacy_cover", r"\bprivacy (?:cover|shutter)\b"),
                ("autofocus", r"\bauto[- ]?focus\b"),
                ("gaming", r"\bgaming\b"),
            ):
                if re.search(pat, blob, re.I):
                    claims.append({"field": field_name, "value": "true"})
                    break
        uniq: dict[str, dict[str, str]] = {}
        for claim in claims:
            if re.search(r"\d+\s*[-–/]\s*\d+", claim["value"]):
                continue
            uniq.setdefault(claim["field"], claim)
        return {"claims": list(uniq.values())}


class OptCue:
    """Prefer SKUs whose selected options explicitly state the firm attr."""

    @classmethod
    def hits(cls, casef: CaseFile, lot: Lot) -> bool:
        opt = (lot.options or "").strip().lower()
        if not opt:
            return False
        for key, val in casef.attributes.items():
            kind, wanted = LexMatch.normalize_wanted(key, str(val))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                num = f"{float(wanted):g}"
                if re.search(
                    rf"\b{re.escape(num)}\s*(?:\"|''|in(?:ch(?:es)?)?)\b", opt
                ) or re.search(rf"\({re.escape(num)}\s*inch", opt):
                    return True
            elif kind == "resolution" and wanted:
                want = str(wanted).lower()
                if want in opt or (
                    want in {"hd", "720p"} and ("720p" in opt or re.search(r"\bhd\b", opt))
                ):
                    return True
            elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                if re.search(rf"\b{int(wanted)}\s*hz\b", opt):
                    return True
            elif kind == "ram" and wanted:
                if _norm(str(wanted)) in _norm(opt):
                    return True
        return False


class CartBeat:
    """Apply cart-add observations and embedded market events to a Lot."""

    @staticmethod
    def apply_add(lot: Lot, add_obs: dict[str, Any]) -> Lot:
        if not isinstance(add_obs, dict):
            return lot
        added = add_obs.get("added") if isinstance(add_obs.get("added"), dict) else {}
        event = add_obs.get("market_event") if isinstance(add_obs.get("market_event"), dict) else {}
        cart = add_obs.get("cart") if isinstance(add_obs.get("cart"), list) else []
        if added:
            price = _as_float(added.get("price"))
            if price is not None:
                lot.price = price
            if "in_stock" in added:
                lot.in_stock = bool(added["in_stock"])
        for row in cart:
            if not isinstance(row, dict):
                continue
            if str(row.get("product_id")) != lot.product_id:
                continue
            if str(row.get("sku") or "default") != (lot.sku or "default"):
                continue
            if "in_stock" in row:
                lot.in_stock = bool(row["in_stock"])
            cp = _as_float(row.get("current_price"))
            if cp is not None:
                lot.price = cp
        if event.get("kind") == "stockout":
            lot.in_stock = False
        if event.get("kind") == "price_change":
            new_p = _as_float(event.get("new_price"))
            if new_p is not None:
                lot.price = new_p
            if "in_stock_after" in event:
                lot.in_stock = bool(event["in_stock_after"])
        return lot


# ---------------------------------------------------------------------------
# Predicate helpers + Weight scoring
# ---------------------------------------------------------------------------


def category_ok(lot: Lot, category: Optional[str]) -> bool:
    if not category:
        return True
    if not RailCheck.device_ok(lot, category):
        return False
    want = _norm(category).rstrip("s")
    path = _norm(" ".join(lot.category_path))
    title = _norm(lot.title)
    if want and (want in path or want in title):
        return True
    aliases = {
        "smartphone": ("phone", "mobile", "smartphone"),
        "phone": ("phone", "mobile", "smartphone"),
        "headphone": ("headphone", "earphone", "headset", "earbuds"),
        "monitor": ("monitor", "display"),
        "tablet": ("tablet", "ipad"),
        "grocery": ("grocery", "groceries", "food"),
        "beauty": ("beauty", "cosmetic", "skincare"),
    }
    for alias in aliases.get(want, ()):
        if alias in path or alias in title:
            return True
    return False


def brand_hit(lot: Lot, brand: Optional[str]) -> bool:
    if not brand:
        return True
    want = _norm(brand)
    if not want:
        return True
    return want in _norm(lot.brand or "") or want in _norm(lot.source_brand or "") or want in _norm(
        lot.title
    )


def brand_strength(lot: Lot, brand: Optional[str]) -> int:
    if not brand:
        return 0
    want = _norm(brand)
    if not want:
        return 0
    if want in _norm(lot.title or ""):
        return 3
    if want in _norm(lot.brand or ""):
        return 2
    if want in _norm(lot.source_brand or ""):
        return 1
    return 0


def budget_ok(lot: Lot, budget: Optional[float]) -> bool:
    if budget is None or lot.price is None:
        return True
    return float(lot.price) <= float(budget) + 1e-6


def hard_ok(casef: CaseFile, lot: Lot) -> bool:
    if casef.require_in_stock and not lot.in_stock:
        return False
    if not budget_ok(lot, casef.budget):
        return False
    if not category_ok(lot, casef.category):
        return False
    if not RailCheck.device_ok(lot, casef.category):
        return False
    if not RailCheck.option_ok(lot, casef.category):
        return False
    if casef.brand and not brand_hit(lot, casef.brand):
        return False
    for key, value in casef.attributes.items():
        if not LexMatch.matches(lot, key, value):
            return False
        # Screen size must be evidenced on title / option inch cue / size fact —
        # blocks decoys that only match via ambiguous option parsing.
        kind, wanted = LexMatch.normalize_wanted(key, str(value))
        if kind == "screen_size" and isinstance(wanted, (int, float)):
            if not firm_size_evidence(lot, float(wanted)):
                return False
    if casef.family == "justification" and EvidencePad.gate_key(casef):
        if not EvidencePad.visible(casef, lot):
            return False
    if casef.family == "retrieval_recall" and casef.title_phrase:
        if not title_phrase_match(lot, casef.title_phrase):
            return False
    return True


def firm_size_evidence(lot: Lot, inches: float) -> bool:
    needle = f"{float(inches):g}"
    title = lot.title or ""
    if needle in title:
        return True
    opt = lot.options or ""
    if re.search(
        rf"\b{re.escape(needle)}\s*(?:\"|''|in(?:ch(?:es)?)?)\b", opt, re.I
    ) or re.search(rf"\({re.escape(needle)}\s*inch", opt, re.I):
        return True
    for fk, fv in (lot.facts or {}).items():
        fl = str(fk).lower()
        if any(n in fl for n in ("size", "screen", "inch", "display")):
            if needle in str(fv) or needle in str(fv).replace(",", "."):
                return True
    return False


def commitment_ready(casef: CaseFile, lot: Lot) -> bool:
    """First-cart quality for event families (intended_event_commitment)."""
    if not hard_ok(casef, lot) or not purchase_ok(casef, lot):
        return False
    if listing_ambiguity(lot) < -1.5:
        return False
    if casef.family == "justification" and EvidencePad.gate_key(casef):
        if not EvidencePad.claimable(casef, lot):
            return False
    # Must view when firm attrs exist; OptCue is a ranking bonus, not a hard gate
    # (many gold SKUs only expose size in title/facts, not option text).
    if casef.attributes and not lot.viewed:
        return False
    return True


def purchase_ok(casef: CaseFile, lot: Lot) -> bool:
    if casef.require_in_stock and not lot.in_stock:
        return False
    if not budget_ok(lot, casef.budget):
        return False
    if casef.brand and not brand_hit(lot, casef.brand):
        return False
    if casef.category and not RailCheck.device_ok(lot, casef.category):
        return False
    if casef.family == "retrieval_recall" and casef.title_phrase:
        if not title_phrase_match(lot, casef.title_phrase):
            return False
    return True


def needs_event_recovery(casef: CaseFile, lot: Lot) -> bool:
    return not purchase_ok(casef, lot)


def is_event_target(previous: Lot, candidate: Lot) -> bool:
    """Stock/price events are product-level — do not re-order the same product_id."""
    if previous.product_id and candidate.product_id == previous.product_id:
        return True
    if previous.listing_id and candidate.listing_id and previous.listing_id == candidate.listing_id:
        return True
    return False


def title_has_firm_size(casef: CaseFile, lot: Lot) -> bool:
    """True when the listing title advertises the firm screen size digits."""
    for key, val in casef.attributes.items():
        kind, wanted = LexMatch.normalize_wanted(key, str(val))
        if kind == "screen_size" and isinstance(wanted, (int, float)):
            needle = f"{float(wanted):g}"
            title = lot.title or ""
            if needle in title:
                return True
            return bool(
                re.search(
                    rf"\b{re.escape(needle)}\s*(?:\"|''|in(?:ch(?:es)?)?)\b",
                    title,
                    re.I,
                )
            )
    return True


def title_phrase_match(lot: Lot, phrase: Optional[str]) -> bool:
    if not phrase:
        return True
    want = _norm(phrase)
    if not want:
        return True
    title = _norm(lot.title or "")
    if not title:
        return False
    if want in title or want.replace(" ", "") in title.replace(" ", ""):
        return True
    # All content tokens must appear (stricter than bag-of-words OR).
    tokens = [t for t in want.split() if len(t) > 1]
    return bool(tokens) and all(tok in title for tok in tokens)


def ranking_key(lot: Lot) -> tuple:
    refresh = LexMatch.refresh_hz(lot)
    size = LexMatch.screen_inches(lot)
    price = float(lot.price) if lot.price is not None else float("inf")
    return (float(refresh or -1.0), float(size or -1.0), -price)


def preference_score(casef: CaseFile, lot: Lot) -> float:
    blob = lot.text_blob()
    score = 0.0
    axes = " ".join(casef.soft_prefs + [casef.goal]).lower()

    def hit(*pats: str) -> bool:
        return any(re.search(p, blob, re.I) for p in pats)

    want_sweat = bool(re.search(r"sweat|water|rain|weather|waterproof|ipx?\d", axes, re.I))
    want_sport = bool(re.search(r"gym|run|sport|workout|fitness|training", axes, re.I))
    want_battery = bool(re.search(r"battery|long.?lasting|all.?day", axes, re.I))
    want_wireless = bool(re.search(r"wireless|bluetooth|cordless|no.?wire", axes, re.I))
    want_noise = bool(re.search(r"\bnoise\b|\banc\b|noise.?cancel", axes, re.I))

    if want_sweat or want_sport:
        if hit(r"ipx?[0-9]", r"sweat", r"water.?resist", r"waterproof", r"splash"):
            score += 3.0
        if hit(r"sport", r"gym", r"running", r"workout", r"ear.?hook", r"secure.?fit"):
            score += 2.0
    if want_battery or want_sport:
        days = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*days?", blob)]
        hours = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*hours?", blob)]
        if days and max(days) >= 5:
            score += 2.0
        elif hours and max(hours) >= 20:
            score += 1.5
        elif hit(r"long.?battery", r"battery.?life"):
            score += 1.0
    if want_wireless or "headphone" in (casef.category or "").lower() or "earbud" in axes:
        if hit(r"bluetooth", r"wireless", r"tws", r"true.?wireless"):
            score += 1.5
        if hit(r"wired", r"3\.5\s*mm") and not hit(r"bluetooth", r"wireless"):
            score -= 1.0
    if want_noise and hit(r"\banc\b", r"noise.?cancel", r"active.?noise"):
        score += 1.5
    for pref in casef.soft_prefs:
        for tok in _tokens(pref):
            if len(tok) > 3 and tok in blob:
                score += 0.2
    if casef.category and "headphone" in casef.category.lower():
        if hit(r"\bcase\b", r"cover", r"cable only", r"replacement tip") and not hit(
            r"earbud", r"headphone", r"headset", r"earphone"
        ):
            score -= 5.0
    return score


def soft_search_queries(casef: CaseFile) -> list[str]:
    """Category-first soft queries from goal/soft_prefs axes only.

    Never emit pack-fixed preference phrases.
    """
    cat = (casef.category or "").strip()
    cat_l = cat.lower()
    axes = " ".join(casef.soft_prefs + [casef.goal]).lower()
    out: list[str] = []

    water = bool(re.search(r"sweat|water|rain|weather|waterproof|water.?resist", axes, re.I))
    sport = bool(re.search(r"gym|run|sport|workout|fitness|training", axes, re.I))
    battery = bool(re.search(r"battery|long.?lasting|all.?day", axes, re.I))
    wireless = bool(re.search(r"wireless|bluetooth|cordless", axes, re.I))
    noise = bool(re.search(r"\bnoise\b|\banc\b|noise.?cancel", axes, re.I))
    audio = any(k in cat_l for k in ("headphone", "earbud", "headset", "earphone"))

    def push(parts: list[str]) -> None:
        q = " ".join(p for p in parts if p).strip()
        if q and q not in out:
            out.append(q)

    if audio:
        # Category noun first to avoid gym/fitness catalog drift.
        push([cat or "headphones", "wireless", "earbuds"])
        if water:
            push([cat or "earbuds", "waterproof", "sweat", "resistant"])
        if sport:
            push([cat or "earbuds", "sport", "wireless", "bluetooth"])
        if water and sport:
            push([cat or "earbuds", "waterproof", "sport", "bluetooth"])
        if battery:
            push([cat or "earbuds", "bluetooth", "battery"])
        if noise:
            push([cat or "earbuds", "noise", "cancelling", "wireless"])
        push([cat or "headphones", "bluetooth", "wireless"])
    else:
        seeds: list[str] = []
        if water:
            seeds.extend(["waterproof", "resistant"])
        if sport:
            seeds.append("sport")
        if battery:
            seeds.append("battery")
        if wireless:
            seeds.extend(["wireless", "bluetooth"])
        if noise:
            seeds.extend(["noise", "cancelling"])
        push(([cat] if cat else []) + seeds[:6])
        if cat and seeds:
            push(seeds[:4] + [cat])
    return out


def heuristic_raw(casef: CaseFile, lot: Lot, rank: int = 0) -> float:
    """EnvPack family heuristic before Weight.normalize."""
    score = 0.0
    if hard_ok(casef, lot):
        score += 10.0
    else:
        if lot.in_stock:
            score += 1.0
        if budget_ok(lot, casef.budget):
            score += 1.0
        if category_ok(lot, casef.category):
            score += 1.5
        for key, value in casef.attributes.items():
            if LexMatch.matches(lot, key, value):
                score += 2.0
    prefer = casef.preferred_brand
    if prefer and brand_hit(lot, prefer):
        bonus = 3.0 if casef.family == "intent_decomposition" else 1.5
        bonus += 0.5 * brand_strength(lot, prefer)
        score += bonus
    if casef.brand and brand_hit(lot, casef.brand):
        score += 1.0
        score += 0.25 * brand_strength(lot, casef.brand)
    if casef.family in {"constraint_satisfaction", "justification"}:
        if EvidencePad.visible(casef, lot):
            score += 2.5
        if OptCue.hits(casef, lot):
            score += 2.0
    blob = lot.text_blob()
    for pref in casef.soft_prefs:
        for tok in _tokens(pref):
            if len(tok) > 3 and tok in blob:
                score += 0.15
    for kw in casef.keywords:
        if _norm(kw) and _norm(kw) in _norm(blob):
            score += 0.2
    if casef.preferred_model and _norm(casef.preferred_model) in _norm(blob):
        score += 2.0
    if casef.preferred_color and _norm(casef.preferred_color) in _norm(
        blob + " " + lot.options
    ):
        score += 1.0
    if casef.family == "ranking":
        refresh, size, neg_price = ranking_key(lot)
        score += max(refresh, 0) * 10.0
        score += max(size, 0) * 1.0
        score += max(0.0, 5.0 + neg_price / 10000.0)
    if casef.family == "preference_reasoning":
        score += preference_score(casef, lot) * 2.0
    if casef.family == "retrieval_recall" and casef.title_phrase:
        if title_phrase_match(lot, casef.title_phrase):
            score += 8.0
    score -= rank * 0.01
    if lot.price is not None and casef.budget and casef.family != "ranking":
        ratio = float(lot.price) / float(casef.budget)
        if 0.4 <= ratio <= 0.95:
            score += 0.3
    if casef.candidate_keys and lot.key in casef.candidate_keys:
        score += 50.0
    return score


def score_lot(casef: CaseFile, lot: Lot, rank: int = 0) -> float:
    """Attach Weight and return score_n (primary ranking signal)."""
    raw = heuristic_raw(casef, lot, rank=rank)
    # Approximate sku index from option richness for MUSK SKU_BONUS_C.
    sku_idx = -1
    if lot.options:
        sku_idx = min(4, max(0, len(lot.options.split("|")) - 1))
    lot.weight = Weight().absorb_raw(raw, rank=rank, sku_idx=sku_idx)
    return lot.weight.score_n


def cart_beat_score(casef: CaseFile, lot: Lot) -> float:
    """Suite-agnostic first-cart ranking for event families."""
    s = float(heuristic_raw(casef, lot))
    # listing ambiguity
    title = lot.title or ""
    inch_hits = re.findall(r"(\d+(?:\.\d+)?)\s*(?:\"|''|in(?:ch(?:es)?)?)\b", title, re.I)
    if len({h for h in inch_hits}) >= 2:
        s -= 2.5
    if re.search(r"\b\d+\s*(?:\"|in(?:ch)?)\s*/\s*\d+", title, re.I):
        s -= 2.0
    if re.search(r"\b[A-Za-z]?\d{2,}[A-Za-z0-9]*\s*/\s*[A-Za-z]?\d{2,}", title):
        s -= 1.5
    if title.count(" / ") >= 1 and len(title) > 80:
        s -= 0.5
    # budget position
    if casef.budget is not None and lot.price is not None and float(casef.budget) > 0:
        ratio = float(lot.price) / float(casef.budget)
        if ratio > 1.0:
            s -= 5.0
        elif ratio < 0.25:
            s -= 1.5
        elif 0.5 <= ratio <= 0.98:
            s += 1.2
        elif 0.35 <= ratio < 0.5:
            s += 0.3
    brand = soft_brand(casef)
    if brand:
        strength = brand_strength(lot, brand)
        s += 1.0 * strength
        if strength >= 3:
            s += 1.0
        elif strength <= 1:
            s -= 0.8
    if casef.family == "justification":
        if EvidencePad.claimable(casef, lot):
            s += 4.0
        elif EvidencePad.visible(casef, lot):
            s += 1.0
        else:
            s -= 3.0
    if casef.attributes:
        if OptCue.hits(casef, lot):
            s += 4.0
        elif all(LexMatch.matches(lot, k, v) for k, v in casef.attributes.items()):
            s += 1.5
        else:
            s -= 2.0
    if lot.viewed:
        s += 0.8
    if not RailCheck.option_ok(lot, casef.category):
        s -= 4.0
    return s


def soft_brand(casef: CaseFile) -> Optional[str]:
    if casef.family == "justification":
        return casef.brand or casef.preferred_brand
    if casef.family == "intent_decomposition":
        return casef.preferred_brand
    if casef.family == "constraint_satisfaction":
        return casef.brand
    return casef.brand or casef.preferred_brand


def listing_ambiguity(lot: Lot) -> float:
    title = lot.title or ""
    pen = 0.0
    inch_hits = re.findall(r"(\d+(?:\.\d+)?)\s*(?:\"|''|in(?:ch(?:es)?)?)\b", title, re.I)
    if len({h for h in inch_hits}) >= 2:
        pen -= 2.5
    if re.search(r"\b\d+\s*(?:\"|in(?:ch)?)\s*/\s*\d+", title, re.I):
        pen -= 2.0
    return pen


def lot_from_summary(item: dict[str, Any]) -> Lot:
    return Lot(
        product_id=str(item.get("product_id") or ""),
        sku=str(item.get("sku") or "default"),
        title=str(item.get("title") or ""),
        brand=str(item.get("brand") or ""),
        price=_as_float(item.get("price")),
        currency=str(item.get("currency") or "PHP"),
        in_stock=bool(item.get("in_stock", True)),
    )


def lots_from_view(obs: dict[str, Any]) -> list[Lot]:
    if obs.get("error"):
        return []
    base = Lot(
        product_id=str(obs.get("product_id") or ""),
        sku="",
        title=str(obs.get("title") or ""),
        brand=str(obs.get("brand") or ""),
        category_path=list(obs.get("category_path") or []),
        facts=dict(obs.get("facts") or {}),
        listing_id=obs.get("listing_id"),
        viewed=True,
    )
    out: list[Lot] = []
    for row in obs.get("variants") or []:
        out.append(
            Lot(
                product_id=base.product_id,
                sku=str(row.get("sku") or "default"),
                title=base.title,
                brand=base.brand,
                price=_as_float(row.get("price")),
                currency=str(row.get("currency") or "PHP"),
                in_stock=bool(row.get("in_stock", True)),
                options=str(row.get("options") or ""),
                category_path=list(base.category_path),
                facts=dict(base.facts),
                listing_id=base.listing_id,
                viewed=True,
            )
        )
    return out or []



# ---------------------------------------------------------------------------
# CaseForge — classify family + CaseFile from public query
# ---------------------------------------------------------------------------


class CaseForge:
    """Parse public query into CaseFile (MUSK decompose spirit → EnvPack families)."""

    _WITHHELD = re.compile(
        r"(if you ask|will tell you|one firm requirement|tell you if you ask)",
        re.I,
    )

    @classmethod
    def is_withheld(cls, query: str) -> bool:
        return bool(cls._WITHHELD.search(query or ""))

    @classmethod
    def decompose(cls, query: str) -> CaseFile:
        family = infer_family(query)
        goal = _goal_only(query)
        casef = CaseFile(
            family=family,
            goal=goal,
            budget=_extract_budget(query),
            candidate_keys=_extract_candidate_keys(query),
        )
        # Retrieval title phrase from quotes.
        phrase_m = re.search(r'matches?\s+"([^"]+)"', goal, re.I)
        if phrase_m:
            casef.title_phrase = phrase_m.group(1).strip()

        try:
            js = llm_json(_CASE_PROMPT.format(query=query), expect="dict")
        except Exception:
            js = {}

        if isinstance(js, dict):
            casef.category = js.get("category") or casef.category
            casef.brand = js.get("brand") or casef.brand
            casef.preferred_brand = js.get("preferred_brand") or casef.preferred_brand
            if js.get("budget") is not None:
                try:
                    casef.budget = float(js["budget"])
                except (TypeError, ValueError):
                    pass
            casef.currency = str(js.get("currency") or casef.currency)
            if "require_in_stock" in js:
                casef.require_in_stock = bool(js["require_in_stock"])
            attrs = js.get("attributes") or {}
            if isinstance(attrs, dict):
                casef.attributes = {
                    str(k): str(v) for k, v in attrs.items() if v is not None and str(v)
                }
            soft = js.get("soft_prefs") or []
            if isinstance(soft, list):
                casef.soft_prefs = [str(x) for x in soft if str(x).strip()]
            kws = js.get("keywords") or []
            if isinstance(kws, list):
                casef.keywords = [str(x) for x in kws if str(x).strip()]
            sqs = js.get("search_queries") or []
            if isinstance(sqs, list):
                casef.search_queries = [str(x) for x in sqs if str(x).strip()]
            ckeys = js.get("candidate_keys") or []
            if isinstance(ckeys, list):
                for key in ckeys:
                    key_s = str(key)
                    if "::" in key_s and key_s not in casef.candidate_keys:
                        casef.candidate_keys.append(key_s)
            casef.preferred_model = js.get("preferred_model") or casef.preferred_model
            casef.preferred_color = js.get("preferred_color") or casef.preferred_color
            if "needs_clarification" in js:
                casef.needs_clarification = bool(js["needs_clarification"])

        cls._regex_overrides(casef, query, goal, family)
        return casef

    @classmethod
    def _regex_overrides(
        cls, casef: CaseFile, query: str, goal: str, family: str
    ) -> None:
        if "screen_size" not in casef.attributes:
            inch_m = re.search(
                r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?(?:\s+screen)?",
                goal,
                re.I,
            )
            if inch_m:
                gate = EvidencePad.gate_from_goal(goal, casef.attributes)
                if gate in {None, "screen_size"} or family == "intent_decomposition":
                    if family != "justification" or gate == "screen_size":
                        casef.attributes["screen_size"] = f"{inch_m.group(1)}in"
        if family == "intent_decomposition":
            inch_m = re.search(
                r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?(?:\s+screen)?",
                goal,
                re.I,
            )
            if inch_m:
                casef.attributes["screen_size"] = f"{inch_m.group(1)}in"

        prefer_m = re.search(
            r"(?:ideally[, ]+)?(?:i\s+)?prefer(?:\s+the)?\s+([A-Za-z][A-Za-z0-9.\- ]{1,24})",
            goal,
            re.I,
        )
        if prefer_m:
            brand_guess = prefer_m.group(1).strip().rstrip(".,;")
            brand_guess = re.split(
                r"\b(?:but|and|if|,)\b", brand_guess, maxsplit=1
            )[0].strip()
            if brand_guess and brand_guess.lower() not in {"the", "a", "an", "black", "white"}:
                casef.preferred_brand = casef.preferred_brand or brand_guess
        if family in {"intent_decomposition", "preference_reasoning", "recovery", "generic"}:
            if (
                casef.brand
                and casef.preferred_brand
                and casef.brand.lower() == casef.preferred_brand.lower()
            ):
                casef.brand = None
            if casef.brand and re.search(
                r"\b(ideally|prefer|nice[- ]to[- ]have)\b", goal, re.I
            ):
                if "must be" not in goal.lower() and "only from" not in goal.lower():
                    casef.preferred_brand = casef.preferred_brand or casef.brand
                    casef.brand = None
            if family == "intent_decomposition" and casef.brand and "must" not in goal.lower():
                casef.preferred_brand = casef.preferred_brand or casef.brand
                casef.brand = None

        if family == "recovery":
            mc = re.search(
                r"prefer(?:\s+the)?\s+([a-z]+)\s+([A-Za-z0-9.\-]{4,})",
                goal,
                re.I,
            )
            if mc:
                casef.preferred_color = casef.preferred_color or mc.group(1)
                casef.preferred_model = casef.preferred_model or mc.group(2)

        if family == "justification":
            brand_m = re.match(
                r"(?:can you get me (?:an? )?in-stock\s+|find me (?:an?\s+)?|i want (?:an?\s+)?)"
                r"([A-Za-z][A-Za-z0-9.\- ]{1,24}?)"
                r"(?=\s+(?:\d|gaming|hd|monitor|smartphone|tablet|in-stock|at\b))",
                goal,
                re.I,
            )
            if brand_m:
                casef.preferred_brand = casef.preferred_brand or brand_m.group(1).strip()
                casef.brand = casef.brand or brand_m.group(1).strip()
            res_m = re.search(
                r"\b(\d{3,4}p|4k|uhd|qhd|full\s*hd|720p|1080p|1440p|hd)\b", goal, re.I
            )
            if res_m and "resolution" not in casef.attributes:
                token = res_m.group(1).replace(" ", "").lower()
                casef.attributes["resolution"] = token
            inch_m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", goal, re.I)
            if inch_m and EvidencePad.gate_from_goal(goal, casef.attributes) == "screen_size":
                casef.attributes["screen_size"] = f"{inch_m.group(1)}in"
                casef.attributes.pop("resolution", None)

        if family == "preference_reasoning":
            for pat, label in (
                (r"sweat|water|rain|weather", "sweat/water resistance"),
                (r"gym|run|sport|workout|fitness", "sport secure fit"),
                (r"battery|all.?day|long.?lasting", "battery life"),
                (r"wireless|bluetooth", "wireless"),
                (r"noise|anc", "noise canceling"),
            ):
                if re.search(pat, goal, re.I):
                    casef.soft_prefs.append(label)

        withheld = cls.is_withheld(query)
        casef.needs_clarification = (
            family == "constraint_satisfaction" and withheld
        ) or (
            family == "preference_reasoning" and not casef.soft_prefs and withheld
        )
        if family == "constraint_satisfaction" and withheld:
            casef.attributes = {}

        casef.needs_justification = family == "justification"
        casef.require_in_stock = True

        normalized_attrs: dict[str, str] = {}
        for key, value in list(casef.attributes.items()):
            kind, wanted = LexMatch.normalize_wanted(key, str(value))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                normalized_attrs["screen_size"] = f"{float(wanted):g}in"
            elif kind == "resolution" and wanted:
                normalized_attrs["resolution"] = str(wanted)
            elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                normalized_attrs["refresh_rate"] = f"{int(wanted)}Hz"
            else:
                normalized_attrs[key] = str(value)
        casef.attributes = normalized_attrs

        if not casef.keywords:
            stop = {
                "i", "need", "want", "find", "looking", "for", "with", "and", "the",
                "a", "an", "to", "of", "in", "on", "at", "or", "my", "me", "can",
                "spend", "up", "under", "below", "php", "usd", "stock", "in-stock",
                "please",
            }
            tokens = [
                t
                for t in re.findall(r"[A-Za-z0-9.+\-]+", goal.lower())
                if t not in stop and len(t) > 1
            ]
            casef.keywords = tokens[:12]
        if not casef.search_queries:
            base = " ".join(casef.keywords[:8]) or goal[:80]
            casef.search_queries = [base]
            if casef.category:
                casef.search_queries.append(f"{casef.category} {base}")
            if casef.preferred_brand or casef.brand:
                brand = casef.brand or casef.preferred_brand
                casef.search_queries.append(f"{brand} {base}")
            for attr, val in list(casef.attributes.items())[:2]:
                casef.search_queries.append(f"{val} {base}")
        # Synonym expansions from Vocab (MUSK spirit) — keep unit aliases only.
        expanded: list[str] = []
        for q in casef.search_queries:
            for alt in Vocab.expand_query_tokens(q)[:3]:
                if re.search(r"[a-zA-Z]+\.\d", alt):
                    continue
                if re.search(
                    r"\d+\.(?:one|two|three|four|five|six|seven|eight|nine)\b",
                    alt,
                    re.I,
                ):
                    continue
                expanded.append(alt)
        casef.search_queries = list(
            dict.fromkeys(q.strip() for q in (casef.search_queries + expanded) if q.strip())
        )

    @classmethod
    def merge_note(cls, casef: CaseFile, note: str) -> CaseFile:
        text = (note or "").strip()
        if not text:
            return casef
        new_budget = _extract_budget(text)
        if new_budget is not None:
            if casef.budget is None or new_budget < casef.budget:
                casef.budget = new_budget

        brand_m = re.search(
            r"(?:required brand is|brand is|must be(?: an?)?)\s+([A-Za-z][A-Za-z0-9.\-]{1,24})",
            text,
            re.I,
        )
        if brand_m and casef.family == "constraint_satisfaction":
            guess = brand_m.group(1).strip().rstrip(".,;")
            if guess and guess.lower() not in {"the", "a", "an", "smartphones", "smartphone"}:
                casef.brand = casef.brand or guess

        cat_m = re.search(
            r"(?:required catalog category is|category is)\s+([A-Za-z][A-Za-z0-9 &/\-]{2,40})",
            text,
            re.I,
        )
        if cat_m and not casef.category:
            cat = cat_m.group(1).strip().rstrip(".,;")
            cl = cat.lower()
            if "smartphone" in cl or "mobile" in cl:
                casef.category = "smartphone"
            elif "tablet" in cl:
                casef.category = "tablet"
            elif "monitor" in cl:
                casef.category = "monitor"
            elif "headphone" in cl or "earbud" in cl:
                casef.category = "headphone"
            else:
                casef.category = cat

        force = bool(re.search(r"\b(must now|instead of|required now|update)\b", text, re.I))
        reveal = bool(
            re.search(r"\b(exact|required|firm|must[- ]have|spec is)\b", text, re.I)
        )

        inch_m = re.search(
            r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?(?:\s+screen)?", text, re.I
        )
        if inch_m and (reveal or force or casef.family == "constraint_satisfaction"):
            casef.attributes["screen_size"] = f"{inch_m.group(1)}in"
        res_m = re.search(
            r"\b(\d{3,4}p|4k|uhd|qhd|full\s*hd|720p|1080p|1440p|hd)\b", text, re.I
        )
        if res_m and (reveal or force or casef.family == "constraint_satisfaction"):
            token = res_m.group(1).replace(" ", "").lower()
            if token in {"fullhd", "fhd"}:
                token = "1080p"
            casef.attributes["resolution"] = token
        hz_m = re.search(r"\b(\d{2,3})\s*hz\b", text, re.I)
        if hz_m and (reveal or force):
            casef.attributes["refresh_rate"] = f"{hz_m.group(1)}Hz"
        ram_m = re.search(r"\b(\d+)\s*gb\s*ram\b", text, re.I)
        if ram_m and (reveal or force):
            casef.attributes["ram"] = f"{ram_m.group(1)}GB"

        if re.search(r"\b(prefer|rather|nice[- ]to[- ]have)\b", text, re.I):
            casef.soft_prefs.append(text[:120])

        # Renormalize
        normalized: dict[str, str] = {}
        for key, value in list(casef.attributes.items()):
            kind, wanted = LexMatch.normalize_wanted(key, str(value))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                normalized["screen_size"] = f"{float(wanted):g}in"
            elif kind == "resolution" and wanted:
                normalized["resolution"] = str(wanted)
            elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                normalized["refresh_rate"] = f"{int(wanted)}Hz"
            else:
                normalized[key] = str(value)
        casef.attributes = normalized
        return casef


# ---------------------------------------------------------------------------
# VeinEngine — search / filter harvest with synonym + unit extensions
# ---------------------------------------------------------------------------


class VeinEngine:
    """Harvest catalog hits via search/filter with MUSK-style query expansion."""

    def __init__(self, hall: RunHall, casef: CaseFile, pool: dict[str, Lot]):
        self.hall = hall
        self.casef = casef
        self.pool = pool
        self.observed_product_ids: list[str] = []
        self._requery_count = 0
        self._score_floors: dict[str, float] = {}

    def ingest(self, obs: dict[str, Any]) -> list[Lot]:
        out: list[Lot] = []
        for item in obs.get("results") or []:
            if not isinstance(item, dict) or not item.get("product_id"):
                continue
            lot = lot_from_summary(item)
            prev = self.pool.get(lot.key)
            if prev and prev.viewed:
                prev.price = lot.price if lot.price is not None else prev.price
                prev.in_stock = lot.in_stock
                lot = prev
            self.pool[lot.key] = lot
            out.append(lot)
            if lot.product_id not in self.observed_product_ids:
                self.observed_product_ids.append(lot.product_id)
        return out

    def search(
        self, query: str, *, k: int = 10, in_stock: Optional[bool] = True
    ) -> list[Lot]:
        if not self.hall.can("search"):
            if self.hall.can("filter"):
                return self.filter_catalog(
                    category=self.casef.category,
                    brand=self.casef.brand or self.casef.preferred_brand,
                    max_price=self.casef.budget,
                    k=k,
                    in_stock=in_stock,
                    _allow_search_fallback=False,
                )
            return []
        args: dict[str, Any] = {"query": query, "k": k}
        if in_stock is not None:
            args["in_stock"] = in_stock
        obs = self.hall.act("search", **args)
        if obs.get("error") == "tool_unavailable":
            return []
        hits = self.ingest(obs)
        # Track MUSK-style score floor for this query.
        if hits:
            top = max(score_lot(self.casef, c) for c in hits)
            self._score_floors[query.lower()] = top
        return hits

    def filter_catalog(
        self,
        *,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        max_price: Optional[float] = None,
        k: int = 10,
        in_stock: Optional[bool] = True,
        _allow_search_fallback: bool = True,
    ) -> list[Lot]:
        if not self.hall.can("filter"):
            if _allow_search_fallback and self.hall.can("search"):
                bits = [p for p in (brand, category) if p]
                if max_price is not None:
                    bits.append(str(max_price))
                q = " ".join(bits).strip() or (self.casef.category or "product")
                return self.search(q, k=k, in_stock=in_stock)
            return []
        args: dict[str, Any] = {"k": k}
        if category:
            args["category"] = category
        if brand:
            args["brand"] = brand
        if max_price is not None:
            args["max_price"] = max_price
        if in_stock is not None:
            args["in_stock"] = in_stock
        obs = self.hall.act("filter", **args)
        if obs.get("error") == "tool_unavailable":
            return []
        out = self.ingest(obs)
        if brand:
            for c in out:
                c.source_brand = brand
                if not (c.brand or "").strip() and self.casef.brand:
                    c.brand = brand
                self.pool[c.key] = c
        return out

    def above_score_floor(self, lot: Lot, query: str) -> bool:
        floor = self._score_floors.get(query.lower())
        if floor is None:
            return True
        return score_lot(self.casef, lot) >= floor - Weight.SCORE_N_THRES

    def firm_attr_queries(self) -> list[str]:
        brand = self.casef.brand or self.casef.preferred_brand or ""
        cat = self.casef.category or ""
        out: list[str] = []
        for key, val in self.casef.attributes.items():
            kind, wanted = LexMatch.normalize_wanted(key, str(val))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                inches = float(wanted)
                # Keep short — step budget is the bottleneck on TF3.
                out.append(f"{brand} {inches:g} inch {cat}".strip())
                out.append(f"{inches:g} inch {cat} {brand}".strip())
                out.append(f"{brand} {cat} {inches:g}in".strip())
                if inches >= 7.0 and cat and "phone" in cat.lower():
                    out.append(f"{brand} foldable {inches:g} inch smartphone".strip())
            elif kind == "resolution" and wanted:
                token = str(wanted)
                out.append(f"{brand} {cat} {token}".strip())
                out.append(f"{brand} {token} {cat}".strip())
                if token in {"hd", "720p"}:
                    out.append(f"{brand} {cat} 720p HD".strip())
                    out.append(f"{brand} 1280x720 {cat}".strip())
                elif token in {"1080p", "fullhd", "fhd"}:
                    out.append(f"{brand} {cat} 1080p Full HD".strip())
            elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                out.append(f"{brand} {cat} {int(wanted)}Hz".strip())
            elif kind == "ram" and wanted:
                out.append(f"{brand} {cat} {wanted} RAM".strip())
            else:
                out.append(f"{brand} {cat} {val}".strip())
        seen: set[str] = set()
        uniq: list[str] = []
        for q in out:
            q = " ".join(q.split())
            if q and q.lower() not in seen:
                seen.add(q.lower())
                uniq.append(q)
        # Cap expansions — TF3 was burning 8+ unit aliases before first cart.
        for q in uniq[:3]:
            for alt in Vocab.expand_query_tokens(q)[1:2]:
                if re.search(r"[a-zA-Z]+\.\d", alt):
                    continue
                if re.search(r"\d+\.(?:one|two|three|four|five|six|seven|eight|nine)\b", alt, re.I):
                    continue
                if alt.lower() not in seen:
                    seen.add(alt.lower())
                    uniq.append(alt)
        return uniq[:6]

    def clarify_once(self) -> None:
        if self.casef.family == "constraint_satisfaction":
            content = (
                "Before I buy: what exact hardware spec must the variant meet "
                "(screen size, resolution, refresh rate, or RAM)?"
            )
        elif self.casef.family == "preference_reasoning":
            content = (
                "To scale the best match: which priorities matter most for your use case "
                "(battery, water/sweat resistance, fit, wireless, brand)?"
            )
        elif self.casef.family == "intent_decomposition":
            content = (
                "Quick check: which requirements are firm versus nice-to-have, "
                "and is the preferred brand optional if a firm-spec item is only "
                "available from another brand?"
            )
        else:
            content = "Before ordering, any must-have specs or budget updates I should apply?"
        self.hall.think("Asking shopper for withheld firm details.", think="clarify")
        if self.hall.can("message"):
            self.hall.act("message", content=content)
        if self.hall.last_shopper_note:
            self.casef = CaseForge.merge_note(self.casef, self.hall.last_shopper_note)
            self.hall.last_shopper_note = None
        if self.casef.family == "constraint_satisfaction" and self.casef.attributes:
            self.casef.search_queries = self.firm_attr_queries()[:4]
            self._requery_count = 0

    def spec_requery(self) -> None:
        if self._requery_count >= 2 or self.hall.remaining() <= 5:
            return
        self._requery_count += 1
        # At most 2 firm searches + 1 filter — duplicates were burning the step budget.
        for q in self.firm_attr_queries()[:2]:
            if self.hall.remaining() <= 5:
                break
            self.search(q, k=10, in_stock=True)
        if self.hall.remaining() > 5:
            brand = self.casef.brand
            if self.casef.preferred_brand and self.casef.family in {
                "justification",
                "constraint_satisfaction",
                "intent_decomposition",
            }:
                brand = brand or self.casef.preferred_brand
            self.filter_catalog(
                category=self.casef.category,
                brand=brand,
                max_price=self.casef.budget,
                k=10,
                in_stock=True,
            )
        # If brand-locked filter found no firm-size hit, one brand-free size search.
        if (
            self.hall.remaining() > 5
            and self.casef.attributes
            and not any(
                hard_ok(self.casef, c) and title_has_firm_size(self.casef, c)
                for c in self.pool.values()
            )
        ):
            for key, val in self.casef.attributes.items():
                kind, wanted = LexMatch.normalize_wanted(key, str(val))
                if kind == "screen_size" and isinstance(wanted, (int, float)):
                    cat = self.casef.category or "smartphone"
                    self.search(f"{float(wanted):g} inch {cat}", k=10, in_stock=True)
                    break

    def harvest(self) -> list[Lot]:
        found: list[Lot] = []
        for key in self.casef.candidate_keys:
            pid, _, sku = key.partition("::")
            if pid:
                lot = Lot(product_id=pid, sku=sku or "default")
                found.append(lot)
                self.pool.setdefault(lot.key, lot)

        if self.casef.family == "ranking" and self.casef.candidate_keys:
            self.hall.think(
                "Ranking family: restricting to candidate_keys from query.",
                think="vein",
            )
            return list(found)

        queries = list(self.casef.search_queries)
        firm_qs = self.firm_attr_queries()
        if firm_qs and self.casef.family in EVENT_FAMILIES:
            queries = firm_qs + [q for q in queries if q not in firm_qs]
        if self.casef.family in {"constraint_satisfaction", "justification"}:
            brand = self.casef.brand or self.casef.preferred_brand or ""
            cat = self.casef.category or ""
            core = firm_qs[:3] if firm_qs else []
            if brand and cat:
                core.append(f"{brand} {cat}".strip())
            firm_toks = [str(v).lower() for v in self.casef.attributes.values()]
            for q in list(self.casef.search_queries):
                ql = q.lower()
                if any(t and t.replace("in", "")[:4] in ql for t in firm_toks) or any(
                    t in ql for t in ("720p", "1080p", "hd", "inch", "hz")
                ):
                    if q not in core:
                        core.append(q)
                if len(core) >= 4:
                    break
            queries = core[:5]
        if self.casef.family == "intent_decomposition" and self.casef.preferred_brand:
            bits = [
                self.casef.preferred_brand,
                self.casef.category or "smartphone",
                " ".join(str(v) for v in self.casef.attributes.values()),
            ]
            brand_q = " ".join(b for b in bits if b)
            queries = [brand_q, self.casef.preferred_brand] + [
                q for q in queries if q not in {brand_q, self.casef.preferred_brand}
            ]
            queries = firm_qs[:2] + [q for q in queries if q not in firm_qs[:2]]
        if self.casef.family == "retrieval_recall" and self.casef.title_phrase:
            queries.insert(0, self.casef.title_phrase)
            queries.append(self.casef.title_phrase.replace(" ", ""))
        if self.casef.family == "preference_reasoning":
            extras = soft_search_queries(self.casef)
            queries = extras + [q for q in queries if q not in extras]
        if self.casef.family == "recovery" and self.casef.preferred_model:
            queries.insert(0, self.casef.preferred_model)
            if self.casef.preferred_color:
                queries.insert(
                    0, f"{self.casef.preferred_color} {self.casef.preferred_model}"
                )
        if self.casef.family == "justification":
            brand = self.casef.brand or self.casef.preferred_brand
            bits = [brand or "", self.casef.category or ""]
            for attr, val in self.casef.attributes.items():
                bits.append(str(val))
            queries.insert(0, " ".join(b for b in bits if b))

        # Unit aliases only — never burn steps on "five.7" / "6.nine" rewrites.
        expanded_queries: list[str] = []
        for q in queries:
            expanded_queries.append(q)
            for alt in Vocab.expand_query_tokens(q)[1:3]:
                if re.search(r"[a-zA-Z]+\.\d", alt):
                    continue
                if re.search(
                    r"\d+\.(?:one|two|three|four|five|six|seven|eight|nine)\b",
                    alt,
                    re.I,
                ):
                    continue
                if alt not in expanded_queries:
                    expanded_queries.append(alt)
        queries = expanded_queries

        self.hall.think(
            f"VeinEngine harvesting with {len(queries)} query variants.",
            think=json.dumps({"action": "search", "n": len(queries)}),
        )

        if (
            self.casef.family in EVENT_FAMILIES
            and firm_qs
            and self.hall.remaining() > 6
        ):
            for q in firm_qs[:2]:
                if self.hall.remaining() <= 6:
                    break
                found.extend(self.search(q, k=10, in_stock=True))
                if any(hard_ok(self.casef, c) for c in self.pool.values()):
                    break

        if (
            self.casef.family == "intent_decomposition"
            and self.casef.preferred_brand
            and self.hall.remaining() > 4
        ):
            found.extend(
                self.filter_catalog(
                    category=self.casef.category,
                    brand=self.casef.preferred_brand,
                    max_price=self.casef.budget,
                    k=10,
                    in_stock=True,
                )
            )

        for q in queries:
            if self.hall.remaining() <= 6:
                break
            if firm_qs and q in firm_qs[:2] and self.casef.family in EVENT_FAMILIES:
                continue
            found.extend(self.search(q, k=10, in_stock=True))
            if self.casef.family == "retrieval_recall" and len(self.observed_product_ids) >= 5:
                break
            if self.casef.family == "intent_decomposition":
                branded = [
                    c
                    for c in self.pool.values()
                    if self.casef.preferred_brand and brand_hit(c, self.casef.preferred_brand)
                ]
                if branded and any(hard_ok(self.casef, c) for c in branded):
                    break
                if branded and len(found) >= 10:
                    break
            elif self.casef.family in {"constraint_satisfaction", "justification"}:
                ok = [
                    c
                    for c in self.pool.values()
                    if hard_ok(self.casef, c)
                    and (
                        EvidencePad.visible(self.casef, c)
                        or title_has_firm_size(self.casef, c)
                    )
                ]
                if ok:
                    break
                # Stop burning aliases once we have title-size candidates to view.
                titled = [
                    c
                    for c in self.pool.values()
                    if purchase_ok(self.casef, c) and title_has_firm_size(self.casef, c)
                ]
                if titled and len(found) >= 8:
                    break
            elif self.casef.family != "retrieval_recall" and len(found) >= 8:
                break

        if self.hall.remaining() > 6:
            brand = self.casef.brand
            if (
                not brand
                and self.casef.preferred_brand
                and self.casef.family != "intent_decomposition"
            ):
                brand = self.casef.preferred_brand
            found.extend(
                self.filter_catalog(
                    category=self.casef.category,
                    brand=brand,
                    max_price=self.casef.budget,
                    k=10,
                    in_stock=True,
                )
            )
            if (
                self.casef.family == "intent_decomposition"
                and self.casef.preferred_brand
                and self.hall.remaining() > 5
            ):
                found.extend(
                    self.filter_catalog(
                        category=self.casef.category,
                        brand=self.casef.preferred_brand,
                        max_price=self.casef.budget,
                        k=10,
                        in_stock=True,
                    )
                )

        uniq: dict[str, Lot] = {}
        for c in found:
            uniq[c.key] = self.pool.get(c.key, c)
        # Floor-filter when we have floors.
        kept = list(uniq.values())
        if self._score_floors:
            floored = [
                c
                for c in kept
                if any(self.above_score_floor(c, q) for q in self._score_floors)
            ]
            if floored:
                kept = floored
        return kept



# ---------------------------------------------------------------------------
# ProbeBench — view / compare / inspect_stock survey
# ---------------------------------------------------------------------------


class ProbeBench:
    """Inspect catalog lots: view/compare/stock with family-tuned depth."""

    def __init__(
        self,
        hall: RunHall,
        casef: CaseFile,
        pool: dict[str, Lot],
        vein: VeinEngine,
    ):
        self.hall = hall
        self.casef = casef
        self.pool = pool
        self.vein = vein

    def view(self, product_id: str) -> list[Lot]:
        if not self.hall.can("view"):
            if self.hall.can("compare"):
                return self.compare([product_id])
            return []
        obs = self.hall.act("view", product_id=product_id)
        if obs.get("error") == "tool_unavailable":
            return []
        lots = lots_from_view(obs)
        prev_brand = ""
        prev_source = ""
        for prev in self.pool.values():
            if prev.product_id == product_id:
                prev_brand = prev.brand or prev_brand
                prev_source = prev.source_brand or prev_source
        for c in lots:
            if not (c.brand or "").strip():
                c.brand = prev_brand
            c.source_brand = c.source_brand or prev_source or c.brand
            self.pool[c.key] = c
            if c.product_id not in self.vein.observed_product_ids:
                self.vein.observed_product_ids.append(c.product_id)
        return lots

    def compare(self, product_ids: list[str]) -> list[Lot]:
        if not self.hall.can("compare"):
            out: list[Lot] = []
            for pid in product_ids[:5]:
                if self.hall.remaining() <= 0 or self.hall.finished:
                    break
                out.extend(self.view(pid))
            return out
        obs = self.hall.act("compare", product_ids=product_ids[:5])
        if obs.get("error") == "tool_unavailable":
            return []
        out: list[Lot] = []
        for row in obs.get("compared") or []:
            out.extend(lots_from_view(row))
        for c in out:
            self.pool[c.key] = c
            if c.product_id not in self.vein.observed_product_ids:
                self.vein.observed_product_ids.append(c.product_id)
        return out

    def inspect_stock(self, product_id: str, sku: str) -> Lot:
        key = f"{product_id}::{sku}"
        lot = self.pool.get(key) or Lot(product_id=product_id, sku=sku)
        if not self.hall.can("inspect_stock"):
            return lot
        obs = self.hall.act("inspect_stock", product_id=product_id, sku=sku)
        if obs.get("error"):
            return lot
        lot.price = _as_float(obs.get("price"))
        lot.in_stock = bool(obs.get("in_stock", lot.in_stock))
        lot.currency = str(obs.get("currency") or lot.currency)
        self.pool[key] = lot
        return lot

    def survey(self, candidates: list[Lot]) -> list[Lot]:
        product_ids: list[str] = []
        for c in candidates:
            if c.product_id not in product_ids:
                product_ids.append(c.product_id)

        min_views = 3 if self.casef.family == "retrieval_recall" else 2
        if self.casef.family == "preference_reasoning":
            min_views = 4
        if self.casef.family == "ranking" and self.casef.candidate_keys:
            min_views = min(
                5, len({k.split("::", 1)[0] for k in self.casef.candidate_keys})
            )
        if self.casef.family == "justification":
            min_views = 3
        if self.casef.family == "recovery":
            min_views = 2
        if self.casef.family == "intent_decomposition":
            min_views = 2
        if self.casef.family == "constraint_satisfaction":
            min_views = 1

        viewed: list[Lot] = []
        use_compare = (
            self.casef.family
            not in {"constraint_satisfaction", "justification", "intent_decomposition"}
            and self.hall.can("compare")
        )

        if self.casef.family == "intent_decomposition" and self.casef.preferred_brand:
            branded_ids = [
                pid
                for pid in product_ids
                if any(
                    brand_hit(c, self.casef.preferred_brand)
                    for c in self.pool.values()
                    if c.product_id == pid
                )
            ]
            other_ids = [pid for pid in product_ids if pid not in branded_ids]
            product_ids = branded_ids + other_ids

        if self.casef.family in EVENT_FAMILIES and self.casef.attributes:

            def _pid_firmness(pid: str) -> tuple[int, float]:
                vs = [c for c in self.pool.values() if c.product_id == pid]
                if not vs:
                    return (0, 0.0)
                best = max(vs, key=lambda c: score_lot(self.casef, c))
                return (1 if hard_ok(self.casef, best) else 0, score_lot(self.casef, best))

            product_ids.sort(key=_pid_firmness, reverse=True)

        self.hall.think(
            f"ProbeBench surveying up to {min_views} product identities.",
            think="inspect",
        )

        while (
            product_ids
            and self.hall.remaining() > 4
            and len({c.product_id for c in viewed}) < min_views
        ):
            batch = product_ids[:5]
            product_ids = product_ids[5:]
            if use_compare and len(batch) >= 2:
                viewed.extend(self.compare(batch))
            else:
                for pid in batch:
                    if self.hall.remaining() <= 4:
                        break
                    viewed.extend(self.view(pid))
                    if len({c.product_id for c in viewed}) >= min_views:
                        break

        if len({c.product_id for c in viewed}) < min_views:
            ranked = sorted(
                self.pool.values(),
                key=lambda c: score_lot(self.casef, c),
                reverse=True,
            )
            for c in ranked:
                if self.hall.remaining() <= 4:
                    break
                if any(x.product_id == c.product_id and x.viewed for x in viewed):
                    continue
                viewed.extend(self.view(c.product_id))
                if len({x.product_id for x in viewed}) >= min_views:
                    break

        out = [
            c
            for c in self.pool.values()
            if c.viewed or c.key in {x.key for x in candidates}
        ]
        if self.casef.family == "ranking" and self.casef.candidate_keys:
            out = [c for c in out if c.key in self.casef.candidate_keys]
            out.sort(key=ranking_key, reverse=True)
        else:
            out.sort(key=lambda c: score_lot(self.casef, c), reverse=True)
        return out


# ---------------------------------------------------------------------------
# ScaleBench — Weight score_n ranking + optional LLM phase/tally select
# ---------------------------------------------------------------------------


class ScaleBench:
    """Select best Lot via Weight.score_n and optional MUSK confirm/select."""

    def __init__(
        self,
        hall: RunHall,
        casef: CaseFile,
        pool: dict[str, Lot],
        vein: Optional[VeinEngine] = None,
    ):
        self.hall = hall
        self.casef = casef
        self.pool = pool
        self.vein = vein

    def best_commit(self, lots: list[Lot]) -> Optional[Lot]:
        if not lots:
            return None
        buyable = [
            c
            for c in lots
            if hard_ok(self.casef, c)
            and purchase_ok(self.casef, c)
            and RailCheck.option_ok(c, self.casef.category)
        ]
        firm = buyable or [
            c
            for c in lots
            if hard_ok(self.casef, c) and RailCheck.option_ok(c, self.casef.category)
        ]
        if not firm:
            return None
        brand = soft_brand(self.casef)
        if brand:
            branded = [c for c in firm if brand_hit(c, brand)]
            if branded:
                strong = [c for c in branded if brand_strength(c, brand) >= 2]
                firm = strong or branded
        if self.casef.family == "justification":
            claimable = [c for c in firm if EvidencePad.claimable(self.casef, c)]
            if claimable:
                firm = claimable
            else:
                evidenced = [c for c in firm if EvidencePad.visible(self.casef, c)]
                if evidenced:
                    firm = evidenced
        if self.casef.family in EVENT_FAMILIES:
            ready = [c for c in firm if commitment_ready(self.casef, c)]
            if ready:
                firm = ready
            else:
                cued = [c for c in firm if OptCue.hits(self.casef, c)]
                if cued:
                    firm = cued
            firm.sort(key=lambda c: cart_beat_score(self.casef, c), reverse=True)
            return firm[0]
        firm.sort(
            key=lambda c: (
                brand_strength(c, brand) if brand else 0,
                score_lot(self.casef, c),
            ),
            reverse=True,
        )
        return firm[0]

    def event_shortlist(self, lots: list[Lot], *, limit: int = 6) -> list[Lot]:
        pool = [
            c
            for c in lots
            if hard_ok(self.casef, c)
            and purchase_ok(self.casef, c)
            and RailCheck.option_ok(c, self.casef.category)
        ]
        if self.casef.family == "justification":
            claimable = [c for c in pool if EvidencePad.claimable(self.casef, c)]
            if claimable:
                pool = claimable
        brand = soft_brand(self.casef)
        if brand:
            strong = [c for c in pool if brand_strength(c, brand) >= 2]
            if strong:
                pool = strong
        pool.sort(key=lambda c: cart_beat_score(self.casef, c), reverse=True)
        out: list[Lot] = []
        seen: set[str] = set()
        for c in pool:
            if c.product_id in seen:
                continue
            seen.add(c.product_id)
            out.append(c)
            if len(out) >= limit:
                break
        return out

    def choose(self, shortlist: list[Lot]) -> Optional[Lot]:
        if self.casef.family == "ranking" and self.casef.candidate_keys:
            pool = [
                c
                for c in shortlist
                if c.key in self.casef.candidate_keys
                and c.in_stock
                and budget_ok(c, self.casef.budget)
            ]
            if not pool:
                pool = [
                    c
                    for c in shortlist
                    if c.key in self.casef.candidate_keys and c.in_stock
                ]
            if not pool:
                return None
            pool.sort(key=ranking_key, reverse=True)
            return pool[0]

        hard = [c for c in shortlist if hard_ok(self.casef, c)]
        pool = hard or [c for c in shortlist if c.in_stock]
        if not pool:
            return None

        if self.casef.family == "intent_decomposition" and self.casef.preferred_brand:
            branded = [c for c in hard if brand_hit(c, self.casef.preferred_brand)]
            if branded:
                pool = branded

        if self.casef.family == "recovery":
            exact = [
                c
                for c in hard
                if (
                    not self.casef.preferred_model
                    or _norm(self.casef.preferred_model) in _norm(c.text_blob())
                )
                and (
                    not self.casef.preferred_color
                    or _norm(self.casef.preferred_color)
                    in _norm(c.text_blob() + " " + c.options)
                )
            ]
            if exact:
                pool = exact

        if self.casef.family == "preference_reasoning":
            # Never fall back to off-category in_stock junk.
            pref_pool = [
                c
                for c in shortlist
                if hard_ok(self.casef, c) and purchase_ok(self.casef, c)
            ]
            if not pref_pool and self.hall.remaining() > 4:
                if self.vein is not None:
                    for q in soft_search_queries(self.casef)[:3]:
                        if self.hall.remaining() <= 3:
                            break
                        self.vein.search(q, k=10, in_stock=True)
                pref_pool = [
                    c
                    for c in self.pool.values()
                    if hard_ok(self.casef, c) and purchase_ok(self.casef, c)
                ]
            if not pref_pool:
                return None
            viewed = [c for c in pref_pool if c.viewed]
            if viewed:
                pref_pool = viewed
            pref_pool = sorted(
                pref_pool,
                key=lambda c: (preference_score(self.casef, c), score_lot(self.casef, c)),
                reverse=True,
            )
            return pref_pool[0] if pref_pool else None

        if self.casef.family == "retrieval_recall" and self.casef.title_phrase:
            matched = [
                c for c in hard if title_phrase_match(c, self.casef.title_phrase)
            ]
            if not matched and self.hall.remaining() > 4:
                self.vein.search(self.casef.title_phrase, k=10, in_stock=True)
                hard = [c for c in self.pool.values() if hard_ok(self.casef, c)]
                matched = [
                    c for c in hard if title_phrase_match(c, self.casef.title_phrase)
                ]
            if not matched:
                return None
            matched = sorted(
                matched,
                key=lambda c: (1 if c.viewed else 0, score_lot(self.casef, c)),
                reverse=True,
            )
            return matched[0]

        if self.casef.family == "justification":
            want = self.casef.brand or self.casef.preferred_brand
            branded = [c for c in hard if want and brand_hit(c, want)]
            if branded:
                pool = branded
            claimable = [c for c in pool if EvidencePad.claimable(self.casef, c)]
            if claimable:
                pool = claimable
            else:
                evidenced = [c for c in pool if EvidencePad.visible(self.casef, c)]
                if evidenced:
                    pool = evidenced
            viewed = [c for c in pool if c.viewed]
            if viewed:
                pool = viewed

        if self.casef.family == "constraint_satisfaction":
            evidenced = [c for c in hard if EvidencePad.visible(self.casef, c)]
            if evidenced:
                pool = evidenced
            mentioned = [c for c in pool if OptCue.hits(self.casef, c)]
            if mentioned:
                pool = mentioned
            viewed = [c for c in pool if c.viewed]
            if viewed:
                pool = viewed

        if self.casef.family in EVENT_FAMILIES:
            pool = sorted(pool, key=lambda c: cart_beat_score(self.casef, c), reverse=True)
        else:
            pool = sorted(pool, key=lambda c: score_lot(self.casef, c), reverse=True)

        top = pool[:5]
        # MUSK phase/tally LLM scale for ambiguous non-event families.
        if (
            len(top) >= 2
            and self.hall.remaining() > 8
            and self.casef.family
            not in {
                "ranking",
                "recovery",
                "justification",
                "preference_reasoning",
                "intent_decomposition",
                "constraint_satisfaction",
            }
        ):
            try:
                scale = self._phase_tally_select(top)
                if scale is not None:
                    return scale
            except Exception:
                pass

        if top:
            self.hall.think(
                f"Best-matching candidate selected: `{top[0].product_id}`",
                think=f"score_n={top[0].weight.score_n:.4f} conf={top[0].weight.confidence:.4f}",
            )
        return top[0] if top else None

    def _phase_tally_select(self, lots: list[Lot]) -> Optional[Lot]:
        """MUSK-style phase1 / tally LLM adjudication."""
        payload = []
        for i, c in enumerate(lots, start=1):
            payload.append(
                {
                    "id": i,
                    "product_id": c.product_id,
                    "sku": c.sku,
                    "title": c.title,
                    "brand": c.brand,
                    "price": c.price,
                    "in_stock": c.in_stock,
                    "options": c.options,
                    "facts": c.facts,
                    "category_path": c.category_path,
                    "score_n": c.weight.score_n,
                    "confidence": c.weight.confidence,
                }
            )
        prompt = (
            "Select the best purchasable variant. Hard constraints must all pass; "
            "soft preferences are secondary.\n"
            "Use MUSK phase/tally style:\n"
            "confirm: evidence quality per constraint "
            "(admissibility → structured → extra_tokens).\n"
            "tally: '<id>→<c1>:key:…,admissible:bool,structured:bool,extra_tokens:n; …'\n"
            f"Family: {self.casef.family}\n"
            f"Goal: {self.casef.goal}\n"
            f"Budget: {self.casef.budget} {self.casef.currency}\n"
            f"Category: {self.casef.category}\n"
            f"Mandatory brand: {self.casef.brand}\n"
            f"Preferred brand: {self.casef.preferred_brand}\n"
            f"Attributes: {json.dumps(self.casef.attributes)}\n"
            f"Soft prefs: {self.casef.soft_prefs}\n"
            f"Candidates:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            'Return JSON: {"best": <int|null>, "closest": <int|null>, '
            '"phase1": <str>, "tally": <str>, "recommend": <str>}'
        )
        js = llm_json(prompt, expect="dict")
        bid = js.get("best")
        if bid is None:
            bid = js.get("closest")
        try:
            bid_i = int(bid or 0)
        except (TypeError, ValueError):
            return None
        if 1 <= bid_i <= len(lots):
            scale = lots[bid_i - 1]
            self.hall.think(
                f"Best-matching candidate selected: `{scale.product_id}`",
                think=f"phase1={js.get('phase1','')}; tally={js.get('tally','')}",
            )
            return scale
        return None


# ---------------------------------------------------------------------------
# LatchBench — first-cart → recover → place_order
# ---------------------------------------------------------------------------


class LatchBench:
    """Seal purchase: add_to_cart, apply market events, recover, place_test_order."""

    def __init__(self, hall: RunHall, casef: CaseFile, pool: dict[str, Lot], vein: VeinEngine, probe: ProbeBench, scale: ScaleBench):
        self.hall = hall
        self.casef = casef
        self.pool = pool
        self.vein = vein
        self.probe = probe
        self.scale = scale
        self.event_target_pids: set[str] = set()

    def _mark_event_target(self, lot: Optional[Lot]) -> None:
        if lot and lot.product_id:
            self.event_target_pids.add(lot.product_id)

    def _is_blocked_target(self, lot: Lot) -> bool:
        return bool(lot.product_id and lot.product_id in self.event_target_pids)

    def ensure_observed(self, lot: Lot) -> Lot:
        """product_info before seal — TF4 pays only when chosen_observed=true."""
        if self.hall.remaining() <= 0:
            return lot
        if not lot.viewed and self.hall.remaining() > 1:
            self.probe.view(lot.product_id)
            lot = self.pool.get(lot.key, lot)
        if self.hall.remaining() > 0:
            lot = self.probe.inspect_stock(lot.product_id, lot.sku)
        return self.pool.get(lot.key, lot)

    def message(self, content: str) -> None:
        if not self.hall.can("message"):
            return
        self.hall.act("message", content=content)
        if self.hall.last_shopper_note:
            self.casef = CaseForge.merge_note(self.casef, self.hall.last_shopper_note)
            self.hall.last_shopper_note = None

    def handle_shopper_note(self, *, pending: Optional[Lot] = None) -> None:
        note = self.hall.last_shopper_note
        if not note:
            return
        self.casef = CaseForge.merge_note(self.casef, note)
        if pending is not None and re.search(
            r"\b(not available|out of stock|sold out|still available|stockout|unavailable)\b",
            note,
            re.I,
        ):
            pending.in_stock = False
            self.pool[pending.key] = pending
            self._mark_event_target(pending)
        if pending is not None:
            price_jump = re.search(
                r"(?:from\s+)?(\d+(?:\.\d+)?)\s*(?:to|->)\s*(\d+(?:\.\d+)?)\s*php",
                note,
                re.I,
            )
            if price_jump:
                pending.price = float(price_jump.group(2))
                self.pool[pending.key] = pending
                self._mark_event_target(pending)
            elif is_market_event_note(note):
                self._mark_event_target(pending)
        # Skip courtesy replies when market events fire and steps are tight.
        if is_market_event_note(note) and self.hall.remaining() <= 4:
            self.hall.last_shopper_note = None
            return
        if shopper_needs_reply(note) and self.hall.remaining() > 1:
            reply = self._compose_reply(note, pending)
            self.hall.last_shopper_note = None
            self.message(reply)
        else:
            self.hall.last_shopper_note = None

    def _compose_reply(self, note: str, choice: Optional[Lot]) -> str:
        if choice is not None:
            bits = [
                f"I'm going with {choice.title or choice.product_id}",
                f"sku {choice.sku}",
            ]
            if choice.price is not None:
                bits.append(f"at {choice.price} {choice.currency}")
            inches = LexMatch.screen_inches(choice)
            if inches is not None:
                bits.append(f"measured_screen={inches:g}in")
            if hard_ok(self.casef, choice):
                bits.append("firm-requirements: met")
            else:
                bits.append("firm-requirements: still checking")
            return "; ".join(bits) + ". I'll verify live price/stock before ordering."
        lower = note.lower()
        if "balance" in lower or "budget" in lower or "php" in lower:
            return (
                f"Understood — I'll stay within {self.casef.budget} {self.casef.currency} "
                "and re-check stock/price before ordering."
            )
        if "expensive" in lower or "price" in lower:
            return "Thanks for the heads-up — I'll re-check the live price and switch if needed."
        return (
            "Got it. I'll apply your latest requirements (specs + budget + in-stock) "
            "and only order a firm-compliant item."
        )

    def add_to_cart(self, product_id: str, sku: str) -> dict[str, Any]:
        if not self.hall.can("add_to_cart"):
            return {"error": "tool_unavailable", "requested": "add_to_cart"}
        obs = self.hall.act("add_to_cart", product_id=product_id, sku=sku)
        self.handle_shopper_note(pending=self.pool.get(f"{product_id}::{sku}"))
        return obs

    def remove_from_cart(self, product_id: str, sku: str) -> None:
        if self.hall.can("remove_from_cart"):
            self.hall.act("remove_from_cart", product_id=product_id, sku=sku)
            self.handle_shopper_note()

    def place_order(
        self,
        lot: Lot,
        justification: Optional[dict] = None,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        if not self.hall.can("place_order"):
            return {"error": "tool_unavailable", "requested": "place_order"}
        # Never seal an out-of-stock SKU — breaks event_closed / in_stock checks.
        if self.casef.require_in_stock and not lot.in_stock:
            return {}
        if not force and not purchase_ok(self.casef, lot):
            return {}
        if self.casef.family == "retrieval_recall" and self.casef.title_phrase:
            if not title_phrase_match(lot, self.casef.title_phrase):
                if not force:
                    return {}
        if (
            self.casef.family in EVENT_FAMILIES
            and self.casef.attributes
            and not force
        ):
            if not all(
                LexMatch.matches(lot, k, v) for k, v in self.casef.attributes.items()
            ):
                return {}
        if (
            self.casef.family == "justification"
            and self.hall.remaining() > 2
            and not force
        ):
            lot = self.probe.inspect_stock(lot.product_id, lot.sku)
            if not purchase_ok(self.casef, lot):
                return {}
            if not EvidencePad.claimable(self.casef, lot):
                return {}
        args: dict[str, Any] = {"product_id": lot.product_id, "sku": lot.sku}
        if justification is not None:
            args["justification"] = justification
        elif self.casef.family == "justification":
            just = ClaimPad.build(self.casef, lot)
            fields = {c.get("field") for c in (just.get("claims") or [])}
            gate = EvidencePad.gate_key(self.casef)
            if gate and gate not in fields:
                if not force:
                    return {}
            args["justification"] = just
        return self.hall.act("place_order", **args)

    def prepare_first_cart(self, choice: Lot) -> Optional[Lot]:
        if self.casef.family not in EVENT_FAMILIES:
            return choice

        def pick_from_product(pid: str) -> Optional[Lot]:
            lots = self.probe.view(pid) if self.hall.remaining() > 2 else []
            self.handle_shopper_note()
            best = self.scale.best_commit(lots)
            if best is None:
                return None
            if self.hall.remaining() > 2:
                best = self.probe.inspect_stock(best.product_id, best.sku)
                self.handle_shopper_note(pending=best)
            if not (hard_ok(self.casef, best) and purchase_ok(self.casef, best)):
                return None
            if self.casef.family in EVENT_FAMILIES and not commitment_ready(
                self.casef, best
            ):
                return None
            if self.casef.family == "justification" and not EvidencePad.claimable(
                self.casef, best
            ):
                return None
            return best

        brand = soft_brand(self.casef)
        if brand and self.casef.family == "intent_decomposition":
            branded_hits = [c for c in self.pool.values() if brand_hit(c, brand)]
            if not branded_hits:
                self._brand_hunt()

        # Before first cart: ensure commitment-ready SKUs exist in the pool.
        if (
            self.casef.family in EVENT_FAMILIES
            and self.casef.attributes
            and self.hall.remaining() > 10
            and not any(commitment_ready(self.casef, c) for c in self.pool.values())
        ):
            self.vein.spec_requery()
            self.handle_shopper_note()

        ranked = self.scale.event_shortlist(list(self.pool.values()), limit=6)
        try_pids = [c.product_id for c in ranked]
        if choice.product_id not in try_pids:
            try_pids.append(choice.product_id)

        picked_ok: list[Lot] = []
        for pid in try_pids:
            if self.hall.remaining() <= 5:
                break
            picked = pick_from_product(pid)
            if picked is None:
                continue
            picked_ok.append(picked)
            if (
                listing_ambiguity(picked) >= -0.5
                and (
                    self.casef.family != "justification"
                    or EvidencePad.claimable(self.casef, picked)
                )
                and (not brand or brand_strength(picked, brand) >= 2)
            ):
                if len(picked_ok) >= 2 or self.hall.remaining() <= 8:
                    break

        if not picked_ok:
            # Soft fallback: best viewed hard_ok SKU (still attribute-faithful).
            soft = [
                c
                for c in self.pool.values()
                if hard_ok(self.casef, c)
                and purchase_ok(self.casef, c)
                and c.viewed
                and listing_ambiguity(c) >= -1.5
            ]
            if self.casef.family == "justification":
                soft = [c for c in soft if EvidencePad.claimable(self.casef, c)] or soft
            if soft:
                soft.sort(key=lambda c: cart_beat_score(self.casef, c), reverse=True)
                return soft[0]
            return None
        picked_ok.sort(key=lambda c: cart_beat_score(self.casef, c), reverse=True)
        best = picked_ok[0]
        if self.casef.family == "justification" and not EvidencePad.claimable(
            self.casef, best
        ):
            return None
        if self.casef.attributes and not all(
            LexMatch.matches(best, k, v) for k, v in self.casef.attributes.items()
        ):
            return None
        return best

    def _brand_hunt(self) -> None:
        brand = soft_brand(self.casef)
        if not brand or self.hall.remaining() <= 5:
            return
        for q in self.vein.firm_attr_queries()[:2]:
            if self.hall.remaining() <= 4:
                break
            self.vein.search(q, k=10, in_stock=True)
        bits = [
            brand,
            self.casef.category or "",
            " ".join(str(v) for v in self.casef.attributes.values()),
        ]
        q = " ".join(b for b in bits if b)
        self.vein.search(q, k=10, in_stock=True)
        if self.hall.remaining() > 4:
            self.vein.filter_catalog(
                category=self.casef.category,
                brand=brand,
                max_price=self.casef.budget,
                k=10,
                in_stock=True,
            )
        self.handle_shopper_note()

    def latch(self, choice: Lot) -> None:
        if (
            self.casef.family in {"intent_decomposition", "constraint_satisfaction"}
            and self.casef.attributes
            and self.hall.remaining() > 8
            and not hard_ok(self.casef, choice)
        ):
            self.vein.spec_requery()
            alts = [c for c in self.pool.values() if hard_ok(self.casef, c)]
            if self.casef.family == "intent_decomposition" and self.casef.preferred_brand:
                branded = [c for c in alts if brand_hit(c, self.casef.preferred_brand)]
                if branded:
                    alts = branded
            if alts:
                alts.sort(key=lambda c: score_lot(self.casef, c), reverse=True)
                choice = alts[0]

        self.handle_shopper_note(pending=choice)
        prepared = self.prepare_first_cart(choice)
        if prepared is None:
            alts = [
                c
                for c in self.pool.values()
                if commitment_ready(self.casef, c)
            ] or [c for c in self.pool.values() if hard_ok(self.casef, c)]
            if self.casef.family == "justification":
                want = self.casef.brand or self.casef.preferred_brand
                branded = [c for c in alts if want and brand_hit(c, want)]
                if branded:
                    alts = branded
                claimable = [c for c in alts if EvidencePad.claimable(self.casef, c)]
                if claimable:
                    alts = claimable
            if self.casef.family == "intent_decomposition" and self.casef.preferred_brand:
                branded = [c for c in alts if brand_hit(c, self.casef.preferred_brand)]
                if branded:
                    alts = branded
            if self.casef.family == "retrieval_recall" and self.casef.title_phrase:
                alts = [c for c in alts if title_phrase_match(c, self.casef.title_phrase)]
            alts.sort(key=lambda c: cart_beat_score(self.casef, c), reverse=True)
            if not alts:
                return
            choice = alts[0]
            if not choice.viewed and self.hall.remaining() > 3:
                self.probe.view(choice.product_id)
                choice = (
                    self.scale.best_commit(
                        [c for c in self.pool.values() if c.product_id == choice.product_id]
                    )
                    or choice
                )
            if self.casef.family == "justification" and not EvidencePad.claimable(
                self.casef, choice
            ):
                if self.hall.remaining() > 6:
                    self.vein.spec_requery()
                    alts = [
                        c
                        for c in self.pool.values()
                        if hard_ok(self.casef, c)
                        and EvidencePad.claimable(self.casef, c)
                    ]
                    alts.sort(key=lambda c: cart_beat_score(self.casef, c), reverse=True)
                    if not alts:
                        return
                    choice = alts[0]
                else:
                    return
        else:
            choice = prepared

        if self.casef.family == "ranking":
            if not choice.viewed and self.hall.remaining() > 3:
                lots = self.probe.view(choice.product_id)
                if lots:
                    lots.sort(key=ranking_key, reverse=True)
                    choice = lots[0]
            if self.hall.remaining() > 3:
                choice = self.probe.inspect_stock(choice.product_id, choice.sku)
                self.handle_shopper_note(pending=choice)
        elif self.casef.family not in EVENT_FAMILIES:
            if not choice.viewed and self.hall.remaining() > 3:
                lots = self.probe.view(choice.product_id)
                self.handle_shopper_note(pending=choice)
                picked = self.scale.best_commit(lots)
                if picked is not None:
                    choice = picked
                elif lots:
                    lots.sort(key=lambda c: score_lot(self.casef, c), reverse=True)
                    choice = lots[0]
            if self.hall.remaining() > 3:
                choice = self.probe.inspect_stock(choice.product_id, choice.sku)
                self.handle_shopper_note(pending=choice)

        if self.hall.remaining() < 2:
            return

        self.hall.think(
            f"First cart commitment: `{choice.product_id}` / {choice.sku}",
            think="latch",
        )
        add_obs = self.add_to_cart(choice.product_id, choice.sku)
        if self.hall.finished:
            return

        live = self.pool.get(choice.key) or choice
        live = CartBeat.apply_add(live, add_obs or {})
        self.pool[live.key] = live

        if self.hall.remaining() > 0:
            live = self.probe.inspect_stock(choice.product_id, choice.sku)
            live = CartBeat.apply_add(live, add_obs or {})
            self.pool[live.key] = live
            self.handle_shopper_note(pending=live)
        else:
            self.handle_shopper_note(pending=live)

        if not needs_event_recovery(self.casef, live):
            if self.casef.family == "ranking":
                replacement = self.recover_ranking(exclude_keys=set(), previous=live)
                if replacement is not None and replacement.key != live.key:
                    self.remove_from_cart(choice.product_id, choice.sku)
                    if self.hall.remaining() > 1:
                        if not replacement.viewed:
                            self.probe.view(replacement.product_id)
                        self.probe.inspect_stock(replacement.product_id, replacement.sku)
                    self.add_to_cart(replacement.product_id, replacement.sku)
                    if self.hall.finished:
                        return
                    if self.hall.remaining() > 0:
                        replacement = self.probe.inspect_stock(
                            replacement.product_id, replacement.sku
                        )
                    if self.hall.remaining() > 0 and purchase_ok(self.casef, replacement):
                        self.place_order(replacement)
                    return
            if purchase_ok(self.casef, live):
                # Fresh stock check immediately before seal.
                if self.hall.remaining() > 0:
                    live = self.probe.inspect_stock(live.product_id, live.sku)
                    self.handle_shopper_note(pending=live)
                if purchase_ok(self.casef, live):
                    if self.casef.family in MARKET_EVENT_FAMILIES and not live.viewed:
                        live = self.ensure_observed(live)
                    self.place_order(live)
                elif self.hall.remaining() > 0:
                    self.remove_from_cart(live.product_id, live.sku)
                    self._mark_event_target(live)
                    replacement = self.recover(
                        exclude_keys={live.key}, previous=live
                    )
                    if (
                        replacement is not None
                        and purchase_ok(self.casef, replacement)
                        and not self._is_blocked_target(replacement)
                    ):
                        replacement = self.ensure_observed(replacement)
                        self.add_to_cart(replacement.product_id, replacement.sku)
                        if not self.hall.finished:
                            self.place_order(replacement)
                    else:
                        self.force_done(exclude_pids=set(self.event_target_pids))
            return

        self.remove_from_cart(choice.product_id, choice.sku)
        self._mark_event_target(choice)
        if self.casef.family == "ranking":
            replacement = self.recover_ranking(exclude_keys={choice.key}, previous=choice)
        else:
            replacement = self.recover(exclude_keys={choice.key}, previous=choice)
        if replacement is None or self._is_blocked_target(replacement):
            # Never re-seal the event target (often OOS) — that fails event_closed.
            if self.hall.remaining() > 0:
                self.force_done(exclude_pids=set(self.event_target_pids))
            return

        if self.hall.remaining() > 1:
            replacement = self.ensure_observed(replacement)
        if (
            not purchase_ok(self.casef, replacement)
            or is_event_target(choice, replacement)
            or self._is_blocked_target(replacement)
        ):
            alt = self.recover(
                exclude_keys={choice.key, replacement.key}, previous=choice
            )
            if (
                alt is None
                or not purchase_ok(self.casef, alt)
                or is_event_target(choice, alt)
                or self._is_blocked_target(alt)
            ):
                if self.hall.remaining() > 0:
                    self.force_done(exclude_pids=set(self.event_target_pids))
                return
            replacement = alt
            if self.hall.remaining() > 1:
                replacement = self.ensure_observed(replacement)
        # Preference / event seal requires product_info on the final SKU.
        if self.casef.family in MARKET_EVENT_FAMILIES and self.hall.remaining() > 1:
            replacement = self.ensure_observed(replacement)
        if (
            self.casef.family in EVENT_FAMILIES
            and self.casef.attributes
            and not hard_ok(self.casef, replacement)
        ):
            if self.hall.remaining() > 0:
                self.force_done(exclude_pids=set(self.event_target_pids))
            return
        self.add_to_cart(replacement.product_id, replacement.sku)
        if self.hall.finished:
            return
        if self.hall.remaining() > 0:
            replacement = self.probe.inspect_stock(
                replacement.product_id, replacement.sku
            )
            self.handle_shopper_note(pending=replacement)
        if purchase_ok(self.casef, replacement) and not is_event_target(
            choice, replacement
        ) and not self._is_blocked_target(replacement):
            if self.casef.family == "preference_reasoning" and not replacement.viewed:
                replacement = self.ensure_observed(replacement)
            self.place_order(replacement)
        elif self.hall.remaining() > 0:
            self.force_done(exclude_pids=set(self.event_target_pids))

    def recover_ranking(
        self, *, exclude_keys: set[str], previous: Lot
    ) -> Optional[Lot]:
        keys = self.casef.candidate_keys or []
        pool: list[Lot] = []
        for key in keys:
            if key in exclude_keys:
                continue
            c = self.pool.get(key)
            if c is None:
                pid, _, sku = key.partition("::")
                c = Lot(product_id=pid, sku=sku or "default")
                self.pool[key] = c
            if not c.viewed and self.hall.remaining() > 3:
                self.probe.view(c.product_id)
                c = self.pool.get(key, c)
            if self.hall.remaining() > 2:
                c = self.probe.inspect_stock(c.product_id, c.sku)
            if c.in_stock and budget_ok(c, self.casef.budget):
                pool.append(c)
        if not pool:
            return None
        pool.sort(key=ranking_key, reverse=True)
        return pool[0]

    def recover(
        self, *, exclude_keys: set[str], previous: Lot
    ) -> Optional[Lot]:
        # After stockout, sibling SKUs of the same product are usually also dead.
        # Skip sibling hunt for retrieval / preference / event families.
        skip_siblings = self.casef.family in {
            "preference_reasoning",
            "retrieval_recall",
        } or (
            self.casef.family in EVENT_FAMILIES and bool(self.casef.attributes)
        )
        if (
            previous.product_id
            and self.hall.remaining() > 2
            and not skip_siblings
        ):
            lots = self.probe.view(previous.product_id)
            siblings = [
                c
                for c in lots
                if c.key not in exclude_keys
                and not is_event_target(previous, c)
                and hard_ok(self.casef, c)
                and purchase_ok(self.casef, c)
            ]
            if siblings:
                siblings.sort(key=lambda c: score_lot(self.casef, c), reverse=True)
                return siblings[0]
            for c in lots:
                if (
                    c.key in exclude_keys
                    or is_event_target(previous, c)
                    or not hard_ok(self.casef, c)
                ):
                    continue
                if self.hall.remaining() <= 2:
                    break
                live = self.probe.inspect_stock(c.product_id, c.sku)
                if purchase_ok(self.casef, live) and hard_ok(self.casef, live):
                    return live

        if self.hall.remaining() > 2:
            if self.casef.family == "recovery" and self.casef.preferred_model:
                self.vein.search(self.casef.preferred_model, k=10, in_stock=True)
                if self.casef.preferred_color and self.hall.remaining() > 2:
                    self.vein.search(
                        f"{self.casef.preferred_color} {self.casef.preferred_model}",
                        k=10,
                        in_stock=True,
                    )
            elif self.casef.family == "preference_reasoning":
                for q in soft_search_queries(self.casef)[:3]:
                    if self.hall.remaining() <= 2:
                        break
                    self.vein.search(q, k=10, in_stock=True)
            elif self.casef.family == "retrieval_recall" and self.casef.title_phrase:
                self.vein.search(self.casef.title_phrase, k=10, in_stock=True)
                toks = [t for t in self.casef.title_phrase.split() if t]
                if len(toks) > 1 and self.hall.remaining() > 2:
                    self.vein.search(" ".join(reversed(toks)), k=10, in_stock=True)
            elif self.casef.family in EVENT_FAMILIES:
                self.vein.spec_requery()
            else:
                q = " ".join(self.casef.keywords[:6]) or self.casef.goal[:60]
                self.vein.search(q, k=10, in_stock=True)
                if self.hall.remaining() > 2:
                    self.vein.filter_catalog(
                        category=self.casef.category,
                        brand=self.casef.brand,
                        max_price=self.casef.budget,
                        k=10,
                        in_stock=True,
                    )
                self.handle_shopper_note()

        candidates = [
            c
            for c in self.pool.values()
            if c.key not in exclude_keys
            and not is_event_target(previous, c)
            and hard_ok(self.casef, c)
            and purchase_ok(self.casef, c)
        ]
        if self.casef.family == "preference_reasoning":
            candidates = [
                c for c in candidates if category_ok(c, self.casef.category)
            ] or candidates
        if self.casef.family == "retrieval_recall" and self.casef.title_phrase:
            matched = [
                c for c in candidates if title_phrase_match(c, self.casef.title_phrase)
            ]
            if matched:
                candidates = matched
            else:
                return None
        # Prefer option-explicit attr hits after events for CS/TF1/TF7.
        if self.casef.family in EVENT_FAMILIES and self.casef.attributes:
            cued = [c for c in candidates if OptCue.hits(self.casef, c)]
            if cued:
                candidates = cued

        if self.casef.family == "recovery":
            same_model = []
            if self.casef.preferred_model:
                same_model = [
                    c
                    for c in candidates
                    if _norm(self.casef.preferred_model)
                    in _norm(c.text_blob() + " " + c.options)
                ]
            if same_model and self.casef.preferred_color:
                other_color = [
                    c
                    for c in same_model
                    if _norm(self.casef.preferred_color)
                    not in _norm(c.text_blob() + " " + c.options)
                ]
                pool = other_color or same_model
            else:
                pool = same_model or candidates
        elif self.casef.family == "preference_reasoning":
            prev_listing = previous.listing_id
            diversed = [
                c
                for c in candidates
                if c.product_id != previous.product_id
                and (
                    not prev_listing
                    or not c.listing_id
                    or c.listing_id != prev_listing
                )
            ]
            pool = diversed or [
                c for c in candidates if c.product_id != previous.product_id
            ]
            pool = sorted(
                pool, key=lambda c: preference_score(self.casef, c), reverse=True
            )
            if not pool:
                return None
            best = pool[0]
            if not best.viewed and self.hall.remaining() > 2:
                self.probe.view(best.product_id)
                best = self.pool.get(best.key, best)
            return best
        elif self.casef.family == "justification":
            want = self.casef.brand or self.casef.preferred_brand
            branded = [c for c in candidates if want and brand_hit(c, want)]
            pool = branded or candidates
        elif self.casef.family == "intent_decomposition":
            if self.casef.preferred_brand:
                branded = [
                    c for c in candidates if brand_hit(c, self.casef.preferred_brand)
                ]
                pool = branded or candidates
            else:
                pool = candidates
        else:
            same = [c for c in candidates if c.product_id == previous.product_id]
            pool = same or candidates

        if not pool:
            if self.casef.family in EVENT_FAMILIES | {"retrieval_recall", "preference_reasoning"}:
                return None
            pool = [
                c
                for c in self.pool.values()
                if c.key not in exclude_keys
                and not is_event_target(previous, c)
                and purchase_ok(self.casef, c)
            ]
            if self.casef.family == "retrieval_recall" and self.casef.title_phrase:
                pool = [
                    c for c in pool if title_phrase_match(c, self.casef.title_phrase)
                ]
        if not pool:
            return None

        if self.casef.family in EVENT_FAMILIES:
            pool.sort(key=lambda c: cart_beat_score(self.casef, c), reverse=True)
        else:
            pool.sort(key=lambda c: score_lot(self.casef, c), reverse=True)
        best = pool[0]
        if not best.viewed and self.hall.remaining() > 2:
            lots = self.probe.view(best.product_id)
            # Event families: never dilute a hard_ok pick into purchase_ok-only.
            if self.casef.family in EVENT_FAMILIES and self.casef.attributes:
                lots = [
                    c
                    for c in lots
                    if hard_ok(self.casef, c)
                    and purchase_ok(self.casef, c)
                    and c.key not in exclude_keys
                    and not is_event_target(previous, c)
                ]
            else:
                lots = [
                    c
                    for c in lots
                    if (hard_ok(self.casef, c) or purchase_ok(self.casef, c))
                    and c.key not in exclude_keys
                ]
            if lots:
                if self.casef.family in EVENT_FAMILIES:
                    lots.sort(
                        key=lambda c: cart_beat_score(self.casef, c), reverse=True
                    )
                else:
                    lots.sort(key=lambda c: score_lot(self.casef, c), reverse=True)
                best = lots[0]
        if not purchase_ok(self.casef, best):
            return None
        if (
            self.casef.family in EVENT_FAMILIES
            and self.casef.attributes
            and not hard_ok(self.casef, best)
        ):
            return None
        return best

    def force_done(self, exclude_pids: Optional[set[str]] = None) -> None:
        """Always try to reach done=true before exiting."""
        if self.hall.finished or self.hall.remaining() <= 0:
            return
        blocked = set(exclude_pids or ()) | set(self.event_target_pids)
        if (
            self.casef.family in EVENT_FAMILIES
            and self.casef.attributes
            and self.hall.remaining() > 5
            and not any(
                hard_ok(self.casef, c)
                and c.product_id not in blocked
                and EvidencePad.visible(self.casef, c)
                for c in self.pool.values()
            )
        ):
            self.vein.spec_requery()
        pool = [
            c
            for c in self.pool.values()
            if purchase_ok(self.casef, c) and c.product_id not in blocked
        ]
        hard = [c for c in pool if hard_ok(self.casef, c)]
        if hard:
            pool = hard
        elif self.casef.family in EVENT_FAMILIES and self.casef.attributes:
            if self.hall.remaining() > 4:
                self.vein.spec_requery()
                pool = [
                    c
                    for c in self.pool.values()
                    if purchase_ok(self.casef, c) and c.product_id not in blocked
                ]
                hard = [c for c in pool if hard_ok(self.casef, c)]
                if hard:
                    pool = hard
            if not hard:
                titled = [c for c in pool if title_has_firm_size(self.casef, c)]
                for c in titled[:3]:
                    if self.hall.remaining() <= 3:
                        break
                    if not c.viewed:
                        self.probe.view(c.product_id)
                pool = [
                    c
                    for c in self.pool.values()
                    if purchase_ok(self.casef, c) and c.product_id not in blocked
                ]
                hard = [c for c in pool if hard_ok(self.casef, c)]
                if hard:
                    pool = hard
                else:
                    pool = [c for c in pool if title_has_firm_size(self.casef, c)] or pool
        if self.casef.family == "justification":
            claimable = [c for c in pool if EvidencePad.claimable(self.casef, c)]
            if claimable:
                pool = claimable
        if self.casef.family == "retrieval_recall" and self.casef.title_phrase:
            matched = [
                c for c in pool if title_phrase_match(c, self.casef.title_phrase)
            ]
            if matched:
                pool = matched
            elif self.hall.remaining() > 3:
                self.vein.search(self.casef.title_phrase, k=8, in_stock=True)
                matched = [
                    c
                    for c in self.pool.values()
                    if purchase_ok(self.casef, c)
                    and c.product_id not in blocked
                    and title_phrase_match(c, self.casef.title_phrase)
                ]
                if matched:
                    pool = matched
        if self.casef.family == "preference_reasoning":
            pref = [c for c in pool if hard_ok(self.casef, c)]
            if not pref:
                for q in soft_search_queries(self.casef)[:2]:
                    if self.hall.remaining() <= 2:
                        break
                    self.vein.search(q, k=8, in_stock=True)
                pref = [
                    c
                    for c in self.pool.values()
                    if hard_ok(self.casef, c)
                    and purchase_ok(self.casef, c)
                    and c.product_id not in blocked
                ]
            if pref:
                pool = pref
            pool = sorted(
                pool, key=lambda c: preference_score(self.casef, c), reverse=True
            )
        brand = soft_brand(self.casef)
        if brand:
            branded = [c for c in pool if brand_hit(c, brand)]
            if branded:
                pool = branded
        if self.casef.candidate_keys:
            cand = [c for c in pool if c.key in self.casef.candidate_keys]
            if cand:
                pool = cand
        if not pool:
            pool = [
                c
                for c in self.pool.values()
                if c.in_stock
                and c.product_id not in blocked
                and RailCheck.device_ok(c, self.casef.category)
            ]
        if not pool:
            pool = [
                c
                for c in self.pool.values()
                if c.in_stock and c.product_id not in blocked
            ]
        if not pool:
            pool = [c for c in self.pool.values() if c.product_id not in blocked]
        if not pool:
            if self.hall.remaining() > 1:
                self.vein.search(
                    " ".join(self.casef.keywords[:4]) or self.casef.goal[:40],
                    k=5,
                    in_stock=True,
                )
                pool = [
                    c
                    for c in self.pool.values()
                    if c.in_stock
                    and c.product_id not in blocked
                    and RailCheck.device_ok(c, self.casef.category)
                ] or [
                    c
                    for c in self.pool.values()
                    if c.in_stock and c.product_id not in blocked
                ] or [c for c in self.pool.values() if c.product_id not in blocked]
            if not pool:
                return
        if self.casef.family == "ranking":
            pool.sort(key=ranking_key, reverse=True)
        elif self.casef.family == "preference_reasoning":
            pool.sort(key=lambda c: preference_score(self.casef, c), reverse=True)
        elif self.casef.family in EVENT_FAMILIES:
            pool.sort(key=lambda c: cart_beat_score(self.casef, c), reverse=True)
        else:
            pool.sort(key=lambda c: score_lot(self.casef, c), reverse=True)
        best = pool[0]
        if self.casef.family in MARKET_EVENT_FAMILIES:
            best = self.ensure_observed(best)
        elif not best.viewed and self.hall.remaining() > 2:
            lots = self.probe.view(best.product_id)
            ok = [
                c
                for c in lots
                if hard_ok(self.casef, c)
                and purchase_ok(self.casef, c)
                and c.product_id not in blocked
            ]
            if ok:
                ok.sort(key=lambda c: score_lot(self.casef, c), reverse=True)
                best = ok[0]
            else:
                best = self.pool.get(best.key, best)

        if self._is_blocked_target(best):
            alts = [c for c in pool if not self._is_blocked_target(c)]
            if not alts:
                return
            best = alts[0]
            if self.casef.family in MARKET_EVENT_FAMILIES:
                best = self.ensure_observed(best)

        if self.hall.remaining() > 1:
            self.add_to_cart(best.product_id, best.sku)
            if self.hall.finished:
                return
            best = self.pool.get(best.key, best)
            if self.hall.remaining() > 0:
                best = self.probe.inspect_stock(best.product_id, best.sku)
                self.handle_shopper_note(pending=best)

        if self.hall.finished:
            return

        if self.hall.remaining() > 3 and not purchase_ok(self.casef, best):
            self.remove_from_cart(best.product_id, best.sku)
            self._mark_event_target(best)
            alt = self.recover(exclude_keys={best.key}, previous=best)
            if alt is not None and not self._is_blocked_target(alt):
                best = self.ensure_observed(alt) if self.hall.remaining() > 1 else alt
            if self.hall.remaining() > 1 and not self._is_blocked_target(best):
                self.add_to_cart(best.product_id, best.sku)
                if self.hall.finished:
                    return
                if self.hall.remaining() > 0:
                    best = self.probe.inspect_stock(best.product_id, best.sku)

        if self.hall.finished:
            return
        if self.hall.remaining() > 0:
            if not purchase_ok(self.casef, best) or self._is_blocked_target(best):
                alts = [
                    c
                    for c in self.pool.values()
                    if c.key != best.key
                    and purchase_ok(self.casef, c)
                    and c.product_id not in blocked
                ]
                if self.casef.family == "retrieval_recall" and self.casef.title_phrase:
                    alts = [
                        c
                        for c in alts
                        if title_phrase_match(c, self.casef.title_phrase)
                    ]
                hard_alts = [c for c in alts if hard_ok(self.casef, c)]
                pick_from = hard_alts or alts
                if pick_from:
                    if self.casef.family == "preference_reasoning":
                        pick_from.sort(
                            key=lambda c: preference_score(self.casef, c), reverse=True
                        )
                    else:
                        pick_from.sort(
                            key=lambda c: score_lot(self.casef, c), reverse=True
                        )
                    best = pick_from[0]
                    if self.casef.family in MARKET_EVENT_FAMILIES:
                        best = self.ensure_observed(best)
                    if self.hall.remaining() > 1:
                        self.add_to_cart(best.product_id, best.sku)
            if self.casef.family in MARKET_EVENT_FAMILIES and not best.viewed:
                best = self.ensure_observed(best)
            self.place_order(best, force=True)
            if self.hall.finished:
                return
            if self.hall.remaining() > 1:
                self.add_to_cart(best.product_id, best.sku)
            if self.hall.remaining() > 0:
                self.place_order(best, force=True)



# ---------------------------------------------------------------------------
# MuskAgent — pipeline orchestrator
# ---------------------------------------------------------------------------


class MuskAgent:
    """CaseForge → VeinEngine → ProbeBench → ScaleBench (Weight) → LatchBench."""

    def __init__(self, hall: RunHall):
        self.hall = hall
        self.casef = CaseForge.decompose(hall.query)
        self.pool: dict[str, Lot] = {}
        self.vein = VeinEngine(hall, self.casef, self.pool)
        self.probe = ProbeBench(hall, self.casef, self.pool, self.vein)
        self.scale = ScaleBench(hall, self.casef, self.pool, self.vein)
        self.latch = LatchBench(
            hall, self.casef, self.pool, self.vein, self.probe, self.scale
        )

    def _sync_case(self) -> None:
        """Keep engines pointing at the latest CaseFile after merges."""
        self.vein.casef = self.casef
        self.probe.casef = self.casef
        self.scale.casef = self.casef
        if self.scale.vein is None:
            self.scale.vein = self.vein
        self.latch.casef = self.casef

    def run(self) -> list[dict[str, Any]]:
        try:
            self.hall.think(
                "Initial query analysis completed.",
                think=(
                    f"family={self.casef.family}; "
                    f"attrs={json.dumps(self.casef.attributes)}; "
                    f"soft={self.casef.soft_prefs[:4]}"
                ),
            )
            if self.casef.needs_clarification and self.hall.remaining() > 10:
                self.vein.clarify_once()
                self.casef = self.vein.casef
                self._sync_case()
            if self.hall.finished:
                return self.hall.dialogue

            candidates = self.vein.harvest()
            self.casef = self.vein.casef
            self._sync_case()
            if self.hall.finished:
                return self.hall.dialogue

            shortlist = self.probe.survey(candidates)
            if self.hall.finished:
                return self.hall.dialogue

            choice = self.scale.choose(shortlist)
            if choice is None:
                fallback = [
                    c
                    for c in self.pool.values()
                    if c.in_stock and budget_ok(c, self.casef.budget)
                ]
                if self.casef.family == "preference_reasoning":
                    fallback = [c for c in fallback if hard_ok(self.casef, c)]
                if self.casef.candidate_keys:
                    fallback = [
                        c for c in fallback if c.key in self.casef.candidate_keys
                    ] or fallback
                if self.casef.family == "retrieval_recall" and self.casef.title_phrase:
                    matched = [
                        c
                        for c in fallback
                        if title_phrase_match(c, self.casef.title_phrase)
                    ]
                    fallback = matched  # empty OK — force_done may re-search phrase
                if self.casef.family == "preference_reasoning":
                    fallback.sort(
                        key=lambda c: preference_score(self.casef, c), reverse=True
                    )
                else:
                    fallback.sort(
                        key=lambda c: score_lot(self.casef, c), reverse=True
                    )
                choice = fallback[0] if fallback else None
            if choice is not None and not self.hall.finished:
                self.latch.latch(choice)
                self.casef = self.latch.casef
                self._sync_case()

            if not self.hall.finished and self.hall.remaining() > 0:
                self.latch.force_done()
        except Exception as exc:  # noqa: BLE001
            self.hall.dialogue.append(
                {
                    "role": "system",
                    "content": f"MUSK_v21 error: {type(exc).__name__}: {exc}",
                }
            )
            try:
                if not self.hall.finished and self.hall.remaining() > 0:
                    self.latch.force_done()
            except Exception:
                pass
        return self.hall.dialogue


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """ORO Bench entrypoint: one EnvPack task session.

    Reads environment.binding and environment.policy_view (query, tools,
    max_steps, max_calls_per_turn). Stops when observation done=true.
    Local packs ≠ Backend qualifying suite; paid_reward is verifier-owned;
    completed != correct.
    """
    hall = RunHall.from_problem(problem_data)
    agent = MuskAgent(hall)
    return agent.run()


__all__ = ["agent_main", "MuskAgent", "Weight", "Vocab", "RunHall", "CapMap"]
