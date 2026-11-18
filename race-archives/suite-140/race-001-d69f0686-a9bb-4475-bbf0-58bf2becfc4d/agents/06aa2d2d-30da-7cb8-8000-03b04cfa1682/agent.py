
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from os import getenv
from typing import Any, Optional

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------
# Inference (ProxyClient only)
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


def _infer_provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()


def _ordered_models() -> list[str]:
    preferred = getenv("SANDBOX_MODEL")
    provider = _infer_provider()
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


def _extract_json_blob(content: str) -> Any:
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


def chat_json(
    prompt: str,
    *,
    temperature: float = 0.0,
    expect: str = "dict",
) -> Any:
    last_err: Optional[Exception] = None
    temp = temperature
    for model in _ordered_models():
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
                parsed = _extract_json_blob(content)
                if expect == "dict" and not isinstance(parsed, dict):
                    raise ValueError(f"expected dict got {type(parsed)}")
                if expect == "list" and not isinstance(parsed, list):
                    raise ValueError(f"expected list got {type(parsed)}")
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                temp = min(temp + 0.2, 0.8)
                continue
    raise RuntimeError(f"chat_json failed: {last_err}")


# ---------------------------------------------------------------------------
# CapabilityAtlas — resolve logical shopping caps against policy_view.tools
# ---------------------------------------------------------------------------


class CapabilityAtlas:
    """Map logical shopping capabilities onto this task's ``policy_view.tools``.

    Per ORO Bench: tool schemas are part of the environment, not a fixed import
    list. Exact schema names win; then synonym aliases. Never invent a tool
    name that is absent from the advertised schema. Callers must soft-skip
    when ``supports`` / ``bind`` returns missing.
    """

    _SYNONYMS: dict[str, tuple[str, ...]] = {
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
        self._bound: dict[str, Optional[str]] = {}
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
        for cap in self._SYNONYMS:
            self._bound[cap] = self._pick(cap)

    def _pick(self, capability: str) -> Optional[str]:
        for candidate in self._SYNONYMS.get(capability, (capability,)):
            if candidate in self.names:
                return candidate
        if capability in self.names:
            return capability
        return None

    def supports(self, capability: str) -> bool:
        return self.bind(capability) is not None

    def bind(self, capability_or_name: str) -> Optional[str]:
        if capability_or_name in self._bound:
            return self._bound[capability_or_name]
        if capability_or_name in self.names:
            return capability_or_name
        matched = self._pick(capability_or_name)
        self._bound[capability_or_name] = matched
        return matched

    def slim_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        allowed = self.param_keys.get(tool_name)
        if not allowed:
            return dict(args)
        return {k: v for k, v in args.items() if k in allowed}


# ---------------------------------------------------------------------------
# Arena — EnvPack session (binding copy each turn)
# ---------------------------------------------------------------------------


@dataclass
class Arena:
    problem_id: str
    binding: dict[str, Any]
    policy: dict[str, Any]
    dialogue: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    finished: bool = False
    last_shopper_note: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    atlas: CapabilityAtlas = field(default_factory=CapabilityAtlas)

    @classmethod
    def open(cls, problem_data: dict[str, Any]) -> "Arena":
        environment = problem_data["environment"]
        policy = dict(environment["policy_view"])
        return cls(
            problem_id=str(
                problem_data.get("problem_id", problem_data.get("id", "problem"))
            ),
            binding=dict(environment["binding"]),
            policy=policy,
            atlas=CapabilityAtlas(policy.get("tools") or []),
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
        if self.atlas.names:
            return set(self.atlas.names)
        names: set[str] = set()
        for tool in self.policy.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            name = fn.get("name") if isinstance(fn, dict) else None
            if name:
                names.add(str(name))
        return names

    def steps_left(self) -> int:
        return max(0, self.max_steps - self.turn)

    def can(self, capability: str) -> bool:
        return self.atlas.supports(capability)

    def think(self, message: str) -> None:
        """Optional internal note (not an environment call)."""
        if message:
            self.notes.append(message)

    @staticmethod
    def _unwrap(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        flat_keys = (
            "results",
            "variants",
            "compared",
            "added",
            "cart",
            "product_id",
            "sent",
            "ordered",
            "error",
        )
        if any(k in payload for k in flat_keys) and not (
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
    def _obs_done(call: dict[str, Any]) -> bool:
        if call.get("done") is True:
            return True
        obs = call.get("observation")
        if isinstance(obs, dict) and obs.get("done") is True:
            return True
        return False

    def dispatch(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        if self.finished:
            return {"calls": [], "user_message": None}
        if self.steps_left() <= 0:
            raise RuntimeError("max_steps exhausted")
        allowed = self.tool_names
        cleaned: list[dict[str, Any]] = []
        for action in actions[: self.max_calls_per_turn]:
            raw_name = action.get("name")
            if not raw_name:
                continue
            name = self.atlas.bind(str(raw_name)) or (
                str(raw_name) if str(raw_name) in allowed else None
            )
            if not name:
                continue
            if allowed and name not in allowed:
                continue
            args = action.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            cleaned.append({"name": name, "args": self.atlas.slim_args(name, args)})
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
                "tool_calls": cleaned,
                "environment_result": result,
            }
        )
        user_message = result.get("user_message")
        if isinstance(user_message, dict) and user_message.get("content"):
            note = str(user_message["content"])
            self.last_shopper_note = note
            self.notes.append(note)
        for call in result.get("calls") or []:
            if self._obs_done(call):
                self.finished = True
        return result

    def invoke(self, name: str, **args: Any) -> dict[str, Any]:
        if not self.atlas.bind(name) and name not in self.atlas.names:
            return {"error": "tool_unavailable", "requested": name}
        result = self.dispatch([{"name": name, "args": args}])
        if result.get("error") == "tool_unavailable":
            return {"error": "tool_unavailable", "requested": name}
        calls = result.get("calls") or []
        if not calls:
            return {}
        return self._unwrap(calls[0].get("observation") or {})


# ---------------------------------------------------------------------------
# Morphology / token coverage (ELON MiscUtil + Vocab spirit, simplified)
# ---------------------------------------------------------------------------


class MorphLexicon:
    STOP = frozenset(
        {
            "color",
            "family",
            "suitable",
            "compatible",
            "support",
            "made",
            "aged",
            "needs",
            "that",
            "helps",
            "contains",
            "includes",
            "option",
            "about",
            "above",
            "after",
            "also",
            "before",
            "below",
            "between",
            "for",
            "from",
            "comes",
            "less",
            "more",
            "over",
            "than",
            "under",
            "with",
            "within",
            "without",
            "provides",
            "sold",
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "at",
            "my",
            "me",
            "i",
            "need",
            "want",
            "find",
            "looking",
            "please",
            "php",
            "usd",
            "stock",
        }
    )
    COLOR = frozenset(
        {
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "pink",
            "purple",
            "violet",
            "orange",
            "brown",
            "gray",
            "grey",
            "beige",
            "navy",
            "gold",
            "silver",
            "bronze",
            "magenta",
            "cyan",
            "dark",
            "light",
        }
    )
    MORPH_SUFFIXES: list[tuple[str, list[str]]] = [
        ("inesses", ["y"]),
        ("iness", ["y"]),
        ("ies", ["y", ""]),
        ("ness", ["y", ""]),
        ("ing", ["e", ""]),
        ("ed", ["e", ""]),
        ("est", ["e", ""]),
        ("er", ["e", ""]),
        ("ish", ["e", ""]),
        ("ily", ["y"]),
        ("es", ["e", ""]),
        ("s", [""]),
        ("ful", [""]),
        ("less", [""]),
        ("ly", [""]),
        ("en", [""]),
    ]
    NUM_WORDS = {
        "single": "1",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    SIZE_MAP = {
        "s": "small",
        "m": "medium",
        "l": "large",
        "xl": "xlarge",
        "xxl": "2xlarge",
        "xxxl": "3xlarge",
    }
    BRIDGES = {
        "colour": "color",
        "grey": "gray",
        "metre": "meter",
        "litre": "liter",
        "women": "woman",
        "men": "man",
        "children": "child",
        "kid": "child",
        "tshirt": "t-shirt",
        "aircon": "airconditioner",
    }
    FILLERS = frozenset({"set", "pack", "accessory", "option", "pair", "unit"})


class MorphMatch:
    """Morphological equality + query-token coverage for WeightScore."""

    @staticmethod
    def scrub(text: str, *, split_alnum: bool = True) -> str:
        s = (text or "").lower().replace("'s", "")
        s = re.sub(r"(?<=\d)\s*[*x]\s*(?=\d)", " ", s)
        s = re.sub(r"[^a-z0-9.\u4e00-\u9fff]", " ", s)
        s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
        if split_alnum:
            s = re.sub(r"\b(\d+)([a-z]+)\b", r"\1 \2", s)
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def stems(cls, word: str) -> set[str]:
        word = (word or "").lower().strip()
        roots = {word}
        if word in MorphLexicon.BRIDGES:
            roots.add(MorphLexicon.BRIDGES[word])
        queue = [word]
        while queue:
            w = queue.pop()
            for suffix, replacements in MorphLexicon.MORPH_SUFFIXES:
                if not w.endswith(suffix) or len(w) <= len(suffix) + 2:
                    continue
                base = w[: -len(suffix)]
                for r in replacements:
                    t = base + r
                    if len(t) >= 2 and t not in roots:
                        roots.add(t)
                        queue.append(t)
                # de-double final consonant after stripping -ing/-ed
                if suffix in {"ing", "ed"} and len(base) >= 3 and base[-1] == base[-2]:
                    t = base[:-1]
                    if t not in roots:
                        roots.add(t)
                        queue.append(t)
        return roots

    @classmethod
    def equal(cls, a: str, b: str) -> bool:
        a, b = (a or "").lower().strip(), (b or "").lower().strip()
        if a == b:
            return True
        if len(a) < 2 or len(b) < 2:
            return False
        if re.search(r"\d", a + b):
            return False
        if len(a) > 6 and len(b) > 6 and a[:6] == b[:6]:
            return True
        return bool(cls.stems(a) & cls.stems(b))

    @classmethod
    def tokenize_query(cls, text: str) -> list[str]:
        toks = [
            t
            for t in cls.scrub(text).split()
            if t and t not in MorphLexicon.STOP and len(t) > 1
        ]
        return toks

    @classmethod
    def coverage(
        cls,
        surface_tokens: list[str],
        query_tokens: list[str],
        *,
        key_len: Optional[int] = None,
    ) -> tuple[set[int], bool]:
        """Return indices of query tokens covered by surface, plus value-used flag."""
        key_len = len(surface_tokens) if key_len is None else key_len
        cover: set[int] = set()
        contrib: set[int] = set()
        for j, token in enumerate(surface_tokens):
            for i, qt in enumerate(query_tokens):
                if cls.equal(token, qt):
                    cover.add(i)
                    contrib.add(j)
                if i < len(query_tokens) - 1:
                    a, b = query_tokens[i], query_tokens[i + 1]
                    if token.startswith(a) and cls.equal(token[len(a) :], b):
                        cover.update({i, i + 1})
                        contrib.add(j)
                for lit, num in MorphLexicon.NUM_WORDS.items():
                    if {lit, num} == {qt, token}:
                        cover.add(i)
                        contrib.add(j)
                        break
            # size abbreviations
            if token in MorphLexicon.SIZE_MAP:
                long = MorphLexicon.SIZE_MAP[token]
                for i, qt in enumerate(query_tokens):
                    if qt == long or cls.equal(qt, long):
                        cover.add(i)
                        contrib.add(j)
        for j in range(len(surface_tokens) - 1):
            h, y = surface_tokens[j], surface_tokens[j + 1]
            for i, v in enumerate(query_tokens):
                if v.startswith(h) and cls.equal(v[len(h) :], y):
                    cover.add(i)
                    contrib.update({j, j + 1})
        val_used = any(j >= key_len for j in contrib)
        return cover, val_used


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value).replace(",", ""))
    return float(m.group(1)) if m else None


def _canon(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _toks(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t]


def _recall_phrase(goal: str) -> Optional[str]:
    """Extract TF2 title-token / phrase target from the public goal."""
    m = re.search(r'matches?\s+"([^"]+)"', goal or "", re.I)
    if m:
        return m.group(1).strip()
    m = re.search(
        r"matches?(?:\s+the)?\s+phrase\s+([A-Za-z0-9][A-Za-z0-9 \-_/]*?)"
        r"(?=\s*,|\s+at\s+or|\s+and\s+buy|\s*$)",
        goal or "",
        re.I,
    )
    if m:
        return m.group(1).strip(" .,;:")
    return None



# ---------------------------------------------------------------------------
# MissionBrief — shopper intent (hard vs soft)
# ---------------------------------------------------------------------------

_FAMILY_CUES = (
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
            "why the one you pick",
            "why it's a good pick",
            "structured justification with claims",
        ),
    ),
)


@dataclass
class MissionBrief:
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
    compact: str = ""
    attr_map: dict[str, str] = field(default_factory=dict)


_BRIEF_PROMPT = """\
You are a shopping mission analyst for an interactive product catalog.
Convert the shopper brief into a structured purchase mission.

Discipline:
1. HARD constraints vs SOFT preferences must be separated cleanly.
2. HARD usually covers: category, budget ceiling, in-stock, named hardware
   (screen_size / resolution / refresh_rate / ram).
3. A named preferred brand is SOFT unless the brief marks it mandatory.
4. When the shopper withholds a firm hardware value until asked
   ("tell you if you ask" / "one firm requirement"), set needs_clarification=true
   and leave attributes empty — invent nothing.
5. keywords = short BM25 value tokens (not schema key names).
6. Emit 3-6 search_queries that rephrase the ask. For resolution gates include
   both class tokens and pixel forms (720p / HD / 1280x720). For inch gates
   spell "N inch".
7. Copy product_id::sku candidate keys when present.
8. needs_justification only when the shopper's own goal asks for a reason —
   ignore tool-footer boilerplate about optional claims.
9. When justification is required, the gate attribute must mirror the shopper's
   named hardware cue.

JSON only:
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
  "needs_clarification": <bool>,
  "compact": <str>,
  "attr_map": {{"product_type": <str>, ...}}
}}

# Shopper brief
{query}
"""


_SELECT_PROMPT = """\
You pick the single best listing for a shopper from an ordered candidate batch.
Precision first: every hard constraint must PASS. Soft prefs are tie-breakers.

Evaluate candidates in the given order. Prefer literal brand/model/spec evidence
in title/options/facts over SEO fluff. Product type is pre-verified — do not
reject on type alone. Title is an unordered token bag.

Shopper goal: {goal}
Family: {family}
Budget: {budget} {currency}
Hard attributes: {attributes}
Mandatory brand: {brand}
Preferred brand: {preferred_brand}
Soft prefs: {soft_prefs}

Candidates (1-indexed):
{products}

Return JSON: {{"best": <1-based index or null>, "rationale": <short str>}}
"""


def _goal_section(query: str) -> str:
    parts = re.split(r"\n\s*Task rules:\s*", query, maxsplit=1)
    return parts[0].strip()


def classify_family(query: str) -> str:
    """Infer TF1–TF7 from public query (goal vs Task rules split)."""
    goal = _goal_section(query).lower()
    full = query.lower()
    parts = re.split(r"\n\s*Task rules:\s*", query, maxsplit=1)
    rules = parts[1].lower() if len(parts) > 1 else full

    # Soft-brand TF1 before CS rule phrases.
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

    # TF7 only from goal hints (footer mentions justification globally).
    for hint in (
        "reason i can trust",
        "give me a reason",
        "trustworthy reason",
        "walk me through why",
        "tell me why",
        "why the one you pick",
        "why it's a good pick",
        "structured justification with claims",
    ):
        if hint in goal:
            return "justification"

    for family, hints in _FAMILY_CUES:
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


def _pull_budget(text: str) -> Optional[float]:
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


def _pull_candidate_keys(text: str) -> list[str]:
    keys = re.findall(r"\b(\d{6,}::[A-Za-z0-9_.\-]+)", text)
    cleaned = [k.rstrip(".,;:)") for k in keys]
    return list(dict.fromkeys(cleaned))


def compose_brief(query: str) -> MissionBrief:
    family = classify_family(query)
    goal = _goal_section(query)
    brief = MissionBrief(
        family=family,
        goal=goal,
        budget=_pull_budget(query),
        candidate_keys=_pull_candidate_keys(query),
    )
    try:
        js = chat_json(_BRIEF_PROMPT.format(query=query), expect="dict")
    except Exception:
        js = {}

    if isinstance(js, dict):
        brief.category = js.get("category") or brief.category
        brief.brand = js.get("brand") or brief.brand
        brief.preferred_brand = js.get("preferred_brand") or brief.preferred_brand
        if js.get("budget") is not None:
            try:
                brief.budget = float(js["budget"])
            except (TypeError, ValueError):
                pass
        brief.currency = str(js.get("currency") or brief.currency)
        if "require_in_stock" in js:
            brief.require_in_stock = bool(js["require_in_stock"])
        attrs = js.get("attributes") or {}
        if isinstance(attrs, dict):
            brief.attributes = {
                str(k): str(v) for k, v in attrs.items() if v is not None and str(v)
            }
        soft = js.get("soft_prefs") or []
        if isinstance(soft, list):
            brief.soft_prefs = [str(x) for x in soft if str(x).strip()]
        kws = js.get("keywords") or []
        if isinstance(kws, list):
            brief.keywords = [str(x) for x in kws if str(x).strip()]
        sqs = js.get("search_queries") or []
        if isinstance(sqs, list):
            brief.search_queries = [str(x) for x in sqs if str(x).strip()]
        ckeys = js.get("candidate_keys") or []
        if isinstance(ckeys, list):
            for key in ckeys:
                key_s = str(key)
                if "::" in key_s and key_s not in brief.candidate_keys:
                    brief.candidate_keys.append(key_s)
        brief.preferred_model = js.get("preferred_model") or brief.preferred_model
        brief.preferred_color = js.get("preferred_color") or brief.preferred_color
        if "needs_clarification" in js:
            brief.needs_clarification = bool(js["needs_clarification"])
        brief.compact = str(js.get("compact") or "")
        amap = js.get("attr_map") or {}
        if isinstance(amap, dict):
            brief.attr_map = {str(k): str(v) for k, v in amap.items() if v}

    # Goal-stated inches
    if "screen_size" not in brief.attributes:
        inch_m = re.search(
            r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?(?:\s+screen)?", goal, re.I
        )
        if inch_m:
            gate = ClaimForge.gate_from_goal(goal, brief.attributes)
            if gate in {None, "screen_size"} or family == "intent_decomposition":
                if family != "justification" or gate == "screen_size":
                    brief.attributes["screen_size"] = f"{inch_m.group(1)}in"
    if family == "intent_decomposition":
        inch_m = re.search(
            r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?(?:\s+screen)?", goal, re.I
        )
        if inch_m:
            brief.attributes["screen_size"] = f"{inch_m.group(1)}in"

    prefer_m = re.search(
        r"(?:ideally[, ]+)?(?:i\s+)?prefer(?:\s+the)?\s+([A-Za-z][A-Za-z0-9.\- ]{1,24})",
        goal,
        re.I,
    )
    if prefer_m:
        brand_guess = prefer_m.group(1).strip().rstrip(".,;")
        brand_guess = re.split(r"\b(?:but|and|if|,)\b", brand_guess, maxsplit=1)[0].strip()
        if brand_guess and brand_guess.lower() not in {"the", "a", "an", "black", "white"}:
            brief.preferred_brand = brief.preferred_brand or brand_guess

    if family in {"intent_decomposition", "preference_reasoning", "recovery", "generic"}:
        if (
            brief.brand
            and brief.preferred_brand
            and brief.brand.lower() == brief.preferred_brand.lower()
        ):
            brief.brand = None
        if brief.brand and re.search(r"\b(ideally|prefer|nice[- ]to[- ]have)\b", goal, re.I):
            if "must be" not in goal.lower() and "only from" not in goal.lower():
                brief.preferred_brand = brief.preferred_brand or brief.brand
                brief.brand = None
        if family == "intent_decomposition" and brief.brand and "must" not in goal.lower():
            brief.preferred_brand = brief.preferred_brand or brief.brand
            brief.brand = None

    if family == "recovery":
        mc = re.search(
            r"prefer(?:\s+the)?\s+([a-z]+)\s+([A-Za-z0-9.\-]{4,})", goal, re.I
        )
        if mc:
            brief.preferred_color = brief.preferred_color or mc.group(1)
            brief.preferred_model = brief.preferred_model or mc.group(2)

    if family == "justification":
        brand_m = re.match(
            r"(?:can you get me (?:an? )?in-stock\s+|find me (?:an?\s+)?|i want (?:an?\s+)?)"
            r"([A-Za-z][A-Za-z0-9.\- ]{1,24}?)"
            r"(?=\s+(?:\d|gaming|hd|monitor|smartphone|tablet|in-stock|at\b))",
            goal,
            re.I,
        )
        if brand_m:
            brief.preferred_brand = brief.preferred_brand or brand_m.group(1).strip()
            brief.brand = brief.brand or brand_m.group(1).strip()
        res_m = re.search(
            r"\b(\d{3,4}p|4k|uhd|qhd|full\s*hd|720p|1080p|1440p|hd)\b", goal, re.I
        )
        if res_m and "resolution" not in brief.attributes:
            token = res_m.group(1).replace(" ", "").lower()
            brief.attributes["resolution"] = token
        inch_m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", goal, re.I)
        if inch_m and ClaimForge.gate_from_goal(goal, brief.attributes) == "screen_size":
            brief.attributes["screen_size"] = f"{inch_m.group(1)}in"
            brief.attributes.pop("resolution", None)

    if family == "preference_reasoning":
        for pat, label in (
            (r"sweat|water|rain|weather", "sweat/water resistance"),
            (r"gym|run|sport|workout|fitness", "sport secure fit"),
            (r"battery|all.?day|long.?lasting", "battery life"),
            (r"wireless|bluetooth", "wireless"),
            (r"noise|anc", "noise canceling"),
        ):
            if re.search(pat, goal, re.I):
                brief.soft_prefs.append(label)

    withheld = RevealProbe.is_withheld(query)
    brief.needs_clarification = (
        family == "constraint_satisfaction" and withheld
    ) or (family == "preference_reasoning" and not brief.soft_prefs and withheld)
    if family == "constraint_satisfaction" and withheld:
        brief.attributes = {}

    brief.needs_justification = family == "justification"
    brief.require_in_stock = True

    normalized: dict[str, str] = {}
    for key, value in list(brief.attributes.items()):
        kind, wanted = FactLens.normalize_want(key, str(value))
        if kind == "screen_size" and isinstance(wanted, (int, float)):
            normalized["screen_size"] = f"{float(wanted):g}in"
        elif kind == "resolution" and wanted:
            normalized["resolution"] = str(wanted)
        elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
            normalized["refresh_rate"] = f"{int(wanted)}Hz"
        else:
            normalized[key] = str(value)
    brief.attributes = normalized

    if not brief.keywords:
        brief.keywords = MorphMatch.tokenize_query(goal)[:12]
    if not brief.compact:
        bits = list(brief.keywords[:10])
        if brief.category:
            bits = [brief.category] + bits
        brief.compact = " ".join(dict.fromkeys(bits))
    if not brief.attr_map and brief.category:
        brief.attr_map = {"product_type": brief.category}

    if not brief.search_queries:
        base = " ".join(brief.keywords[:8]) or goal[:80]
        brief.search_queries = [base]
        if brief.category:
            brief.search_queries.append(f"{brief.category} {base}")
        brand = brief.brand or brief.preferred_brand
        if brand:
            brief.search_queries.append(f"{brand} {base}")
        for _, val in list(brief.attributes.items())[:2]:
            brief.search_queries.append(f"{val} {base}")
    brief.search_queries = list(
        dict.fromkeys(q.strip() for q in brief.search_queries if q.strip())
    )
    return brief


# ---------------------------------------------------------------------------
# Listing — SKU candidate
# ---------------------------------------------------------------------------


@dataclass
class Listing:
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
    weight: Optional["WeightScore"] = None

    @property
    def key(self) -> str:
        return f"{self.product_id}::{self.sku}"

    def blob(self) -> str:
        parts = [
            self.title,
            self.brand,
            self.options,
            " ".join(self.category_path),
            json.dumps(self.facts, ensure_ascii=False),
        ]
        return " ".join(str(p) for p in parts if p).lower()


def listing_from_hit(item: dict[str, Any]) -> Listing:
    return Listing(
        product_id=str(item.get("product_id") or ""),
        sku=str(item.get("sku") or "default"),
        title=str(item.get("title") or ""),
        brand=str(item.get("brand") or ""),
        price=_num(item.get("price")),
        currency=str(item.get("currency") or "PHP"),
        in_stock=bool(item.get("in_stock", True)),
    )


def enrich_listing(listing: Listing, obs: dict[str, Any]) -> Listing:
    if obs.get("error"):
        return listing
    listing.title = str(obs.get("title") or listing.title)
    listing.brand = str(obs.get("brand") or listing.brand)
    listing.category_path = list(obs.get("category_path") or [])
    listing.facts = dict(obs.get("facts") or {})
    listing.listing_id = obs.get("listing_id") or listing.listing_id
    listing.viewed = True
    return listing


def listings_from_view(obs: dict[str, Any]) -> list[Listing]:
    if obs.get("error"):
        return []
    base = Listing(
        product_id=str(obs.get("product_id") or ""),
        sku="",
        title=str(obs.get("title") or ""),
        brand=str(obs.get("brand") or ""),
        category_path=list(obs.get("category_path") or []),
        facts=dict(obs.get("facts") or {}),
        listing_id=obs.get("listing_id"),
        viewed=True,
    )
    out: list[Listing] = []
    for row in obs.get("variants") or []:
        v = Listing(
            product_id=base.product_id,
            sku=str(row.get("sku") or "default"),
            title=base.title,
            brand=base.brand,
            price=_num(row.get("price")),
            currency=str(row.get("currency") or "PHP"),
            in_stock=bool(row.get("in_stock", True)),
            options=str(row.get("options") or ""),
            category_path=list(base.category_path),
            facts=dict(base.facts),
            listing_id=base.listing_id,
            viewed=True,
        )
        out.append(v)
    if not out and base.product_id:
        base.sku = "default"
        out.append(base)
    return out



# ---------------------------------------------------------------------------
# FactLens — hardware facts (ELON morph + coverage flavored)
# ---------------------------------------------------------------------------


class FactLens:
    """Normalize / match hardware attributes from facts, options, titles."""

    _RES = {
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
        # Quote-aware: 7.6" / 7.6in / 7.6 inch (do not require \b after ")
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
    def screen_inches(cls, listing: Listing) -> Optional[float]:
        # Only treat as display size for device-like categories/titles.
        path = _canon(" ".join(listing.category_path))
        title_l = (listing.title or "").lower()
        deviceish = bool(
            re.search(
                r"\b(monitor|display|laptop|notebook|tablet|ipad|phone|smartphone|"
                r"television|tv|screen)\b",
                f"{title_l} {path}",
                re.I,
            )
        )
        if not deviceish and not any(
            n in path for n in ("monitor", "display", "laptop", "tablet", "phone", "tv")
        ):
            # Pipe/beauty/etc. titles often contain bare "1in"/"3in" that are not screens.
            if not re.search(r"\b(inch|inches|\"|''|screen|display)\b", title_l, re.I):
                return None
        opt = (listing.options or "").strip()
        if opt:
            m = re.search(r"\((\d+(?:\.\d+)?)\s*inch", opt, re.I) or re.search(
                r"(\d+(?:\.\d+)?)\s*inch", opt, re.I
            )
            if m:
                return float(m.group(1))
            num = cls.parse_inches(opt)
            if num is not None and not re.search(r"\d+\s*[-–/]\s*\d+", opt):
                return num
        for fk, fv in (listing.facts or {}).items():
            if any(n in str(fk).lower() for n in ("size", "screen", "inch", "display")):
                raw = str(fv)
                if re.search(r"\d+(?:\.\d+)?\s*[-–/,]\s*\d+(?:\.\d+)?", raw):
                    continue
                num = cls.parse_inches(raw) or _num(raw)
                if num is not None and 3.0 <= num <= 100.0:
                    return num
        title = listing.title or ""
        if re.search(r"\d+(?:\.\d+)?\s*[-–/]\s*\d+(?:\.\d+)?", title):
            return None
        if not deviceish:
            return None
        return cls.parse_inches(title)

    @classmethod
    def resolution_token(cls, listing: Listing) -> Optional[str]:
        blob = " ".join(
            [
                listing.title or "",
                listing.options or "",
                " ".join(f"{k} {v}" for k, v in (listing.facts or {}).items()),
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
    def refresh_hz(cls, listing: Listing) -> Optional[float]:
        blob = " ".join(
            [
                listing.title or "",
                listing.options or "",
                " ".join(f"{k} {v}" for k, v in (listing.facts or {}).items()),
            ]
        ).lower()
        vals = [float(m) for m in re.findall(r"(\d{2,3})\s*hz", blob)]
        return max(vals) if vals else None

    @classmethod
    def normalize_want(cls, key: str, value: str) -> tuple[str, Any]:
        key_l = key.lower().replace("-", "_")
        val = str(value).strip()
        val_l = val.lower().replace(" ", "")
        if val_l in cls._RES or re.fullmatch(r"\d{3,4}p|4k|uhd|qhd|hd|fullhd|fhd", val_l):
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
            return "refresh_rate", _num(val)
        if "ram" in key_l:
            return "ram", _num(val)
        if "screen" in key_l or "size" in key_l or val_l.endswith("in"):
            inches = cls.parse_inches(val)
            if inches is None and re.fullmatch(r"\d+(?:\.\d+)?", val_l):
                inches = float(val_l)
            if inches is not None and not (3.0 <= float(inches) <= 100.0):
                inches = None
            return "screen_size", inches
        return key_l, val.lower()

    @classmethod
    def matches(cls, listing: Listing, key: str, value: str) -> bool:
        kind, wanted = cls.normalize_want(key, value)
        if wanted is None or wanted == "":
            return False
        if kind == "screen_size":
            got = cls.screen_inches(listing)
            if got is None or not isinstance(wanted, (int, float)):
                return False
            return abs(float(got) - float(wanted)) < 0.005
        if kind == "resolution":
            got = cls.resolution_token(listing)
            if not got:
                return False
            want = str(wanted).lower()
            if want in {"hd", "720p"} and got in {"hd", "720p"}:
                return True
            alts = cls._RES.get(want, (want,))
            return got in alts or got == want
        if kind == "refresh_rate":
            got = cls.refresh_hz(listing)
            if got is None or not isinstance(wanted, (int, float)):
                return False
            return abs(float(got) - float(wanted)) <= 1.0
        if kind == "ram":
            blob = listing.blob()
            if not isinstance(wanted, (int, float)):
                return str(wanted).lower() in blob
            return bool(re.search(rf"\b{int(wanted)}\s*gb\b", blob))
        return _canon(str(wanted)) in _canon(listing.blob())


# ---------------------------------------------------------------------------
# AccessoryFence — device/accessory rejection (no LCD Screen false-positive)
# ---------------------------------------------------------------------------


class AccessoryFence:
    _ACCESSORY = (
        "case",
        "cover",
        "charger",
        "cable",
        "screen protector",
        "tempered glass",
        "replacement tip",
        "ear tip",
        "strap only",
        "mount only",
        "holder only",
        "stand only",
        "skin only",
        "bumper",
        "holster",
        "pouch",
        "digitizer",
        "digitiz",
        "panel assembly",
        "replacement part",
        "touch screen digitizer",
        "touch screen digitiz",
        "spare part",
        "repair part",
    )
    _BAD_PATH = (
        "accessor",
        "replacement",
        "spare part",
        "repair",
        "parts & tools",
        "parts and tools",
    )
    _BAD_OPT = re.compile(
        r"screen[- ]?only|replacement[- ]?screen|digitizer|lcd panel|touch panel only",
        re.I,
    )

    @classmethod
    def title_is_accessory(cls, title: str) -> bool:
        t = (title or "").lower()
        if not t:
            return False
        device_hit = bool(
            re.search(
                r"\b(smartphone|mobile phone|cellphone|tablet|ipad|monitor|headphone|earbud|headset)\b",
                t,
            )
        )
        if any(b in t for b in cls._ACCESSORY):
            if re.search(r"\b(with case|free case|include[sd]? case)\b", t) and device_hit:
                return False
            if re.search(r"\b(phone|tablet|monitor)\s+(case|cover|protector|bumper|pouch)\b", t):
                return True
            if re.search(r"\b(case|cover|protector|bumper)\b.*\b(phone|tablet)\b", t):
                return True
            return True
        # LCD/digitizer panels — but do NOT false-positive "LCD Screen" on phones
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
    def path_is_accessory(cls, category_path: list[str]) -> bool:
        path = _canon(" ".join(category_path or []))
        return any(b in path for b in cls._BAD_PATH)

    @classmethod
    def looks_phone(cls, listing: Listing) -> bool:
        if cls.title_is_accessory(listing.title or "") or cls.path_is_accessory(
            listing.category_path
        ):
            return False
        title = listing.title or ""
        path = _canon(" ".join(listing.category_path))
        blob = listing.blob()
        if re.search(
            r"\b(smartphone|cellphone|mobile phone|android phone|iphone|galaxy\s+[a-z]?\d)\b",
            title,
            re.I,
        ):
            return True
        if any(w in path for w in ("smartphone", "mobile phone", "cell phone")):
            return True
        for fk, fv in (listing.facts or {}).items():
            fl = f"{fk} {fv}".lower()
            if re.search(r"\b(android|ios|iphone os|blackberry os|harmonyos)\b", fl):
                return True
        if re.search(r"\b(android|ios)\b", blob) and re.search(
            r"\b(gb ram|mp camera|dual sim)\b", blob
        ):
            return True
        return False

    @classmethod
    def looks_tablet(cls, listing: Listing) -> bool:
        if cls.title_is_accessory(listing.title or "") or cls.path_is_accessory(
            listing.category_path
        ):
            return False
        title = listing.title or ""
        if re.search(r"\b(drone|dron|aircraft|quadcopter|uav)\b", title, re.I):
            return False
        path = _canon(" ".join(listing.category_path))
        if re.search(r"\b(tablet|ipad)\b", title, re.I):
            return True
        if "tablet" in path:
            return True
        for fk, fv in (listing.facts or {}).items():
            if re.search(r"\b(tablet|ipados)\b", f"{fk} {fv}", re.I):
                return True
        return False

    @classmethod
    def looks_monitor(cls, listing: Listing) -> bool:
        if cls.title_is_accessory(listing.title or "") or cls.path_is_accessory(
            listing.category_path
        ):
            return False
        title = listing.title or ""
        path = _canon(" ".join(listing.category_path))
        if re.search(r"\b(monitor|display)\b", title, re.I) and "lcd display" not in title.lower():
            if re.search(
                r"\b(digitizer|replacement|for vivo|for samsung|for xiaomi)\b", title, re.I
            ):
                return False
            return True
        return "monitor" in path or ("display" in path and "accessor" not in path)

    @classmethod
    def option_ok(cls, listing: Listing, category: Optional[str]) -> bool:
        if not category:
            return True
        want = _canon(category)
        if not any(w in want for w in ("phone", "smartphone", "tablet", "monitor")):
            return True
        if cls._BAD_OPT.search(listing.options or ""):
            return False
        return True

    @classmethod
    def looks_audio(cls, listing: Listing) -> bool:
        """True when listing is an audio wearable/device, not gym/fitness gear."""
        title = listing.title or ""
        path = _canon(" ".join(listing.category_path))
        blob = f"{title} {path}".lower()
        if re.search(
            r"\b(resistance band|yoga band|exercise band|fitness loop|pro bands|"
            r"dumbbell|kettlebell|pull.?up|stretch(er)? band)\b",
            blob,
            re.I,
        ):
            if not re.search(
                r"\b(earbud|earbuds|earphone|earphones|headphone|headphones|"
                r"headset|tws|airpod)\b",
                blob,
                re.I,
            ):
                return False
        if re.search(
            r"\b(earbud|earbuds|earphone|earphones|headphone|headphones|headset|tws)\b",
            title,
            re.I,
        ):
            return True
        if any(k in path for k in ("headphone", "earbud", "headset", "earphone", "audio")):
            return True
        return False

    @classmethod
    def accepts(cls, listing: Listing, category: Optional[str]) -> bool:
        if not category:
            return True
        want = _canon(category).rstrip("s")
        title = listing.title or ""
        if cls.title_is_accessory(title) or cls.path_is_accessory(listing.category_path):
            if any(
                w in want
                for w in ("phone", "smartphone", "tablet", "monitor", "headphone", "earbud")
            ):
                return False
        rich = bool(listing.viewed or listing.facts or listing.category_path)
        if "smartphone" in want or want == "phone":
            return cls.looks_phone(listing) if rich else True
        if "tablet" in want:
            return cls.looks_tablet(listing) if rich else True
        if "monitor" in want:
            return cls.looks_monitor(listing) if rich else True
        if any(w in want for w in ("headphone", "earbud", "headset", "earphone")):
            # Even before view: reject obvious non-audio fitness listings.
            if not cls.looks_audio(listing):
                # Allow unknown search hits with empty path only if title isn't fitness gear.
                if re.search(
                    r"\b(band|dumbbell|yoga|resistance|stretcher)\b", title, re.I
                ) and not re.search(
                    r"\b(earbud|headphone|earphone|headset|tws)\b", title, re.I
                ):
                    return False
            if rich:
                return cls.looks_audio(listing)
        return True


# ---------------------------------------------------------------------------
# RevealProbe — TF3 ask once / parse reveal / never invent
# ---------------------------------------------------------------------------


class RevealProbe:
    _WITHHELD = re.compile(
        r"(if you ask|will tell you|one firm requirement|tell you if you ask)",
        re.I,
    )

    @classmethod
    def is_withheld(cls, query: str) -> bool:
        return bool(cls._WITHHELD.search(query or ""))

    @classmethod
    def absorb_note(cls, brief: MissionBrief, note: str) -> MissionBrief:
        text = (note or "").strip()
        if not text:
            return brief
        new_budget = _pull_budget(text)
        if new_budget is not None:
            if brief.budget is None or new_budget < brief.budget:
                brief.budget = new_budget

        brand_m = re.search(
            r"(?:required brand is|brand is|must be(?: an?)?)\s+([A-Za-z][A-Za-z0-9.\-]{1,24})",
            text,
            re.I,
        )
        if brand_m and brief.family == "constraint_satisfaction":
            guess = brand_m.group(1).strip().rstrip(".,;")
            if guess and guess.lower() not in {
                "the",
                "a",
                "an",
                "smartphones",
                "smartphone",
            }:
                brief.brand = brief.brand or guess

        cat_m = re.search(
            r"(?:required catalog category is|category is)\s+([A-Za-z][A-Za-z0-9 &/\-]{2,40})",
            text,
            re.I,
        )
        if cat_m and not brief.category:
            cat = cat_m.group(1).strip().rstrip(".,;")
            cl = cat.lower()
            if "smartphone" in cl or "mobile" in cl:
                brief.category = "smartphone"
            elif "tablet" in cl:
                brief.category = "tablet"
            elif "monitor" in cl:
                brief.category = "monitor"
            elif "headphone" in cl or "earbud" in cl:
                brief.category = "headphone"
            else:
                brief.category = cat

        replace = re.search(
            r"(?:must now have|change one requirement.*?have)\s+(.+?)\s+instead of",
            text,
            re.I | re.S,
        )
        force = bool(replace) or "must now" in text.lower() or "instead of" in text.lower()
        reveal = bool(
            re.search(
                r"(exact variant|required|still needs|firm requirement|must have)",
                text,
                re.I,
            )
        )

        m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?(?:\s+screen)?", text, re.I)
        if m:
            val = f"{m.group(1)}in"
            if force or reveal or "screen_size" not in brief.attributes:
                brief.attributes["screen_size"] = val
        m = re.search(r"\b(\d{3,4}p|4k|uhd|qhd|full\s*hd|hd)\b", text, re.I)
        if m:
            val = m.group(1).replace(" ", "")
            if force or reveal or "resolution" not in brief.attributes:
                brief.attributes["resolution"] = val
        m = re.search(r"\b(\d{2,3})\s*hz\b", text, re.I)
        if m:
            val = f"{m.group(1)}Hz"
            if force or reveal or "refresh_rate" not in brief.attributes:
                brief.attributes["refresh_rate"] = val
        m = re.search(r"\b(\d+)\s*gb\s*ram\b", text, re.I)
        if m:
            val = f"{m.group(1)}GB"
            if force or reveal or "ram" not in brief.attributes:
                brief.attributes["ram"] = val

        if "prefer" in text.lower() or "rather" in text.lower():
            brief.soft_prefs.append(text)

        normalized: dict[str, str] = {}
        for key, value in list(brief.attributes.items()):
            kind, wanted = FactLens.normalize_want(key, str(value))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                normalized["screen_size"] = f"{float(wanted):g}in"
            elif kind == "resolution" and wanted:
                normalized["resolution"] = str(wanted)
            elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                normalized["refresh_rate"] = f"{int(wanted)}Hz"
            else:
                normalized[key] = str(value)
        brief.attributes = normalized
        return brief


def shopper_wants_reply(note: Optional[str]) -> bool:
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


# ---------------------------------------------------------------------------
# ProofLedger / ClaimForge — TF7 evidence from visible facts only
# ---------------------------------------------------------------------------


class ClaimForge:
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
    def forge(cls, brief: MissionBrief, listing: Listing) -> dict[str, Any]:
        claims: list[dict[str, str]] = []
        gate = cls.gate_from_goal(brief.goal, brief.attributes)
        claim_val = ProofLedger.claim_value(brief, listing)
        if gate and claim_val:
            claims.append({"field": gate, "value": str(claim_val)})
        if listing.price is not None:
            claims.append({"field": "price", "value": f"{float(listing.price):g}"})
        claims.append(
            {"field": "in_stock", "value": "true" if listing.in_stock else "false"}
        )
        soft_added = False
        want_brand = brief.brand or brief.preferred_brand
        if listing.brand and (not want_brand or brand_hits(listing, want_brand)):
            claims.append({"field": "brand", "value": str(listing.brand)})
            soft_added = True
        elif (
            want_brand
            and brand_hits(listing, want_brand)
            and want_brand.lower() in _canon(listing.title or "")
        ):
            claims.append({"field": "brand", "value": str(want_brand)})
            soft_added = True
        if not soft_added:
            blob = listing.blob()
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


class ProofLedger:
    """Gate attribute must be visible; claims never echo plan wants."""

    @classmethod
    def gate_key(cls, brief: MissionBrief) -> Optional[str]:
        return ClaimForge.gate_from_goal(brief.goal, brief.attributes) or (
            next(iter(brief.attributes), None) if brief.attributes else None
        )

    @classmethod
    def measured(cls, brief: MissionBrief, listing: Listing) -> Optional[str]:
        gate = cls.gate_key(brief)
        if not gate:
            return None
        if gate == "resolution":
            return FactLens.resolution_token(listing)
        if gate == "screen_size":
            inches = FactLens.screen_inches(listing)
            return f"{inches:g}" if inches is not None else None
        if gate == "refresh_rate":
            hz = FactLens.refresh_hz(listing)
            return f"{int(hz)}Hz" if hz is not None else None
        if gate == "ram":
            blob = listing.blob()
            m = re.search(r"\b(\d+)\s*gb\s*ram\b", blob, re.I) or re.search(
                r"\b(\d+)\s*gb\b", (listing.options or "") + " " + blob, re.I
            )
            return f"{m.group(1)}GB" if m else None
        return None

    @classmethod
    def visible(cls, brief: MissionBrief, listing: Listing) -> bool:
        gate = cls.gate_key(brief)
        if not gate:
            return True
        measured = cls.measured(brief, listing)
        if measured is None:
            return False
        if gate in brief.attributes:
            return FactLens.matches(listing, gate, brief.attributes[gate])
        if gate == "resolution":
            goal = brief.goal.lower()
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
    def claimable(cls, brief: MissionBrief, listing: Listing) -> bool:
        return cls.claim_value(brief, listing) is not None

    @classmethod
    def claim_value(cls, brief: MissionBrief, listing: Listing) -> Optional[str]:
        if not cls.visible(brief, listing):
            return None
        gate = cls.gate_key(brief)
        measured = cls.measured(brief, listing)
        if not gate or not measured:
            return None
        if gate == "resolution":
            fact_blob = " ".join(f"{k} {v}" for k, v in (listing.facts or {}).items())
            surface = f"{listing.title or ''} {listing.options or ''} {fact_blob}".lower()
            goal = brief.goal.lower()
            lit_720 = "720p" in surface or "1280x720" in surface or "1280 x 720" in surface
            lit_hd = (
                bool(re.search(r"\bhd\b", surface))
                and "full hd" not in surface
                and "fhd" not in surface
            )
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
                if (
                    re.search(r"\bhd\b", goal)
                    and "1080" not in goal
                    and "full" not in goal
                    and (lit_hd or lit_720)
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


# ---------------------------------------------------------------------------
# SkuExact — firm-attr option preference
# ---------------------------------------------------------------------------


class SkuExact:
    @classmethod
    def option_mentions(cls, brief: MissionBrief, listing: Listing) -> bool:
        opt = (listing.options or "").strip().lower()
        if not opt:
            return False
        for key, val in brief.attributes.items():
            kind, wanted = FactLens.normalize_want(key, str(val))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                num = f"{float(wanted):g}"
                if re.search(
                    rf"\b{re.escape(num)}\s*(?:\"|''|in(?:ch(?:es)?)?)\b", opt
                ) or re.search(rf"\({re.escape(num)}\s*inch", opt):
                    return True
            elif kind == "resolution" and wanted:
                want = str(wanted).lower()
                if want in opt or (
                    want in {"hd", "720p"}
                    and ("720p" in opt or re.search(r"\bhd\b", opt))
                ):
                    return True
            elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                if re.search(rf"\b{int(wanted)}\s*hz\b", opt):
                    return True
            elif kind == "ram" and wanted:
                if _canon(str(wanted)) in _canon(opt):
                    return True
        return False


# ---------------------------------------------------------------------------
# MarketLedger — cart-add market events
# ---------------------------------------------------------------------------


class MarketLedger:
    @staticmethod
    def apply_add(listing: Listing, add_obs: dict[str, Any]) -> Listing:
        if not isinstance(add_obs, dict):
            return listing
        added = add_obs.get("added") if isinstance(add_obs.get("added"), dict) else {}
        event = (
            add_obs.get("market_event")
            if isinstance(add_obs.get("market_event"), dict)
            else {}
        )
        cart = add_obs.get("cart") if isinstance(add_obs.get("cart"), list) else []
        if added:
            price = _num(added.get("price"))
            if price is not None:
                listing.price = price
            if "in_stock" in added:
                listing.in_stock = bool(added["in_stock"])
        for row in cart:
            if not isinstance(row, dict):
                continue
            if str(row.get("product_id")) != listing.product_id:
                continue
            if str(row.get("sku") or "default") != (listing.sku or "default"):
                continue
            if "in_stock" in row:
                listing.in_stock = bool(row["in_stock"])
            cp = _num(row.get("current_price"))
            if cp is not None:
                listing.price = cp
        if event.get("kind") == "stockout":
            listing.in_stock = False
        if event.get("kind") == "price_change":
            new_p = _num(event.get("new_price"))
            if new_p is not None:
                listing.price = new_p
            if "in_stock_after" in event:
                listing.in_stock = bool(event["in_stock_after"])
        return listing



# ---------------------------------------------------------------------------
# WeightScore — ELON-style token coverage ranking
# ---------------------------------------------------------------------------


@dataclass
class WeightScore:
    covers: list[int] = field(default_factory=list)
    score_r: float = 0.0
    score_n: float = 0.0
    score: float = 0.0
    confidence: float = 0.0
    wrank: int = -1
    rank: int = -1
    detail: dict[str, float] = field(default_factory=dict)
    pt_bonus: float = 0.0
    att_bonus: float = 0.0
    sku_bonus: float = 0.0
    rank_bonus: float = 0.0

    # Decaying confidence ladder (ELON LOG_PROBS spirit, rebuilt)
    _CONF = [
        0.67,
        0.13,
        0.04,
        0.02,
        0.011,
        0.005,
        0.0035,
        0.003,
        0.0029,
        0.0028,
    ]
    _RANK_BONUS = [0.01 - i * 2e-5 for i in range(400)]
    _SKU_BONUS = [0.005, 0.005, 0.005, 0.001, 0.0005, 0.0001]

    @classmethod
    def weigh(
        cls,
        listing: Listing,
        brief: MissionBrief,
        *,
        rank: int = 0,
        query_tokens: Optional[list[str]] = None,
        token_weights: Optional[list[float]] = None,
    ) -> "WeightScore":
        qs = query_tokens or MorphMatch.tokenize_query(
            brief.compact or " ".join(brief.keywords) or brief.goal
        )
        if not qs:
            qs = MorphMatch.tokenize_query(brief.goal)[:16]
        ws = token_weights or [
            (
                1.0
                if (
                    re.search(r"\d", t)
                    or t in MorphLexicon.COLOR
                    or t in MorphLexicon.SIZE_MAP.values()
                )
                else (0.5 if t in MorphLexicon.FILLERS else 0.75)
            )
            for t in qs
        ]
        wsum = sum(ws) or 1.0

        title_toks = MorphMatch.scrub(listing.title or "", split_alnum=True).split()
        title_cover, _ = MorphMatch.coverage(title_toks, qs)

        # product_type coverage bonus
        pt = MorphMatch.tokenize_query(
            brief.attr_map.get("product_type") or brief.category or ""
        )
        pt_cover, _ = MorphMatch.coverage(title_toks, pt) if pt else (set(), False)
        pt_bonus = 0.02 * len(pt_cover) / len(pt) if pt else 0.0

        # attribute / options / facts as additional coverage surface
        att_surface: list[str] = []
        for fk, fv in (listing.facts or {}).items():
            att_surface.extend(MorphMatch.scrub(f"{fk} {fv}").split())
        if listing.options:
            att_surface.extend(MorphMatch.scrub(listing.options).split())
        if listing.brand:
            att_surface.extend(MorphMatch.scrub(listing.brand).split())
        att_cover, _ = MorphMatch.coverage(att_surface, qs) if att_surface else (set(), False)

        # non-product-type query indices get attribute bonus
        att_indice = set(range(len(qs)))
        for i, ch in enumerate(qs):
            for s in pt:
                if MorphMatch.equal(ch, s):
                    att_indice.discard(i)
                    break
        focused = att_cover & att_indice
        att_bonus = 0.4 * len(focused) / len(att_indice) if att_indice else 0.0

        total = title_cover | att_cover
        score_r = sum(ws[i] for i in total if i < len(ws))
        score = score_r + pt_bonus + att_bonus

        sku_idx = 0
        if listing.sku and listing.sku != "default":
            sku_idx = min(2, len(listing.sku) // 4)
        sku_bonus = cls._SKU_BONUS[sku_idx] if sku_idx < len(cls._SKU_BONUS) else 0.0
        rank_bonus = cls._RANK_BONUS[rank] if rank < len(cls._RANK_BONUS) else 0.0
        score_n = (score / wsum if wsum > 1e-8 else 0.0) + sku_bonus + rank_bonus

        detail = {qs[i]: ws[i] for i in total if i < len(qs)}
        return cls(
            covers=sorted(total),
            score_r=score_r,
            score_n=score_n,
            score=score,
            rank=rank,
            detail=detail,
            pt_bonus=pt_bonus,
            att_bonus=att_bonus,
            sku_bonus=sku_bonus,
            rank_bonus=rank_bonus,
        )

    def set_confidence(self, wrank: int) -> "WeightScore":
        if wrank >= len(self._CONF):
            self.confidence = self._CONF[-1] * (0.9 ** (wrank - len(self._CONF) + 1))
        else:
            self.confidence = self._CONF[wrank]
        self.wrank = wrank
        return self


def fib_windows(n0: int = 3, n1: int = 2, limit: int = 40) -> list[int]:
    """Fibonacci batch sizes: 3, 2, 5, 7, 12, ..."""
    a, b = n0, n1
    out = [a, b]
    while sum(out) < limit:
        a, b = b, a + b
        out.append(b)
    return out


# ---------------------------------------------------------------------------
# Hard / soft gates (policy helpers)
# ---------------------------------------------------------------------------


def category_ok(listing: Listing, category: Optional[str]) -> bool:
    if not category:
        return True
    if not AccessoryFence.accepts(listing, category):
        return False
    want = _canon(category).rstrip("s")
    path = _canon(" ".join(listing.category_path))
    title = _canon(listing.title)
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


def brand_hits(listing: Listing, brand: Optional[str], *, soft: bool = False) -> bool:
    if not brand:
        return True
    want = _canon(brand)
    if not want:
        return True
    vb = _canon(listing.brand or "")
    vs = _canon(listing.source_brand or "")
    hit = want in vb or want in vs or want in _canon(listing.title)
    return hit


def brand_force(listing: Listing, brand: Optional[str]) -> int:
    if not brand:
        return 0
    want = _canon(brand)
    if not want:
        return 0
    if want in _canon(listing.title or ""):
        return 3
    if want in _canon(listing.brand or ""):
        return 2
    if want in _canon(listing.source_brand or ""):
        return 1
    return 0


def budget_ok(listing: Listing, budget: Optional[float]) -> bool:
    if budget is None or listing.price is None:
        return True
    return float(listing.price) <= float(budget) + 1e-6


def firm_ok(brief: MissionBrief, listing: Listing) -> bool:
    if brief.require_in_stock and not listing.in_stock:
        return False
    if not budget_ok(listing, brief.budget):
        return False
    if not category_ok(listing, brief.category):
        return False
    if not AccessoryFence.accepts(listing, brief.category):
        return False
    if not AccessoryFence.option_ok(listing, brief.category):
        return False
    if brief.brand and not brand_hits(listing, brief.brand):
        return False
    for key, value in brief.attributes.items():
        if not FactLens.matches(listing, key, value):
            return False
    if brief.family == "justification" and ProofLedger.gate_key(brief):
        if not ProofLedger.visible(brief, listing):
            return False
    return True


def buyable(brief: MissionBrief, listing: Listing) -> bool:
    if brief.require_in_stock and not listing.in_stock:
        return False
    if not budget_ok(listing, brief.budget):
        return False
    if brief.brand and not brand_hits(listing, brief.brand):
        return False
    if brief.category and not AccessoryFence.accepts(listing, brief.category):
        return False
    return True


def needs_recover(brief: MissionBrief, listing: Listing) -> bool:
    return not buyable(brief, listing)


def ranking_tuple(listing: Listing) -> tuple:
    refresh = FactLens.refresh_hz(listing)
    size = FactLens.screen_inches(listing)
    price = float(listing.price) if listing.price is not None else float("inf")
    return (float(refresh or -1.0), float(size or -1.0), -price)


def preference_fit(brief: MissionBrief, listing: Listing) -> float:
    """Soft-axis fit from the shopper's stated preferences (suite-agnostic).

    Prefer *rated / measurable* evidence of an asked-for soft property over
    vague marketing synonyms. Axis detection is driven only by the public
    goal + soft_prefs text — never by suite task IDs or fixed product lines.
    """
    blob = listing.blob()
    score = 0.0
    axes = " ".join(brief.soft_prefs + [brief.goal]).lower()

    def hit(*pats: str) -> bool:
        return any(re.search(p, blob, re.I) for p in pats)

    want_water = bool(
        re.search(r"sweat|water|rain|weather|waterproof|water.?resist|ipx?\d", axes, re.I)
    )
    want_sport = bool(re.search(r"gym|run|sport|workout|fitness|training", axes, re.I))
    want_battery = bool(re.search(r"battery|long.?lasting|all.?day", axes, re.I))
    want_wireless = bool(re.search(r"wireless|bluetooth|cordless|no.?wire", axes, re.I))
    want_noise = bool(re.search(r"\bnoise\b|\banc\b|noise.?cancel", axes, re.I))
    pay_more = bool(
        re.search(
            r"pay (?:a bit )?more|cheapest isn.?t|don.?t just grab the cheapest|rather pay",
            axes,
            re.I,
        )
    )

    if want_water or want_sport:
        # Rated ingress marks beat unquantified marketing adjectives.
        has_rated_water = hit(r"ipx?\s*[0-9]", r"ip\s*x\s*[0-9]", r"atm\s*\d", r"\d+\s*m\s*water")
        if has_rated_water and hit(r"ipx?\s*[5-9]", r"ip\s*x\s*[5-9]"):
            score += 5.0
        elif has_rated_water:
            score += 4.0
        elif hit(r"sweat", r"water.?resist", r"waterproof", r"splash"):
            score += 2.0
        if hit(r"sport", r"gym", r"running", r"workout", r"ear.?hook", r"secure.?fit"):
            score += 2.0
        # Use-case contradiction: sleep/lounge cues vs outdoor/sport ask.
        if (
            want_sport
            and hit(r"\bsleep", r"sleeping", r"lounge")
            and not has_rated_water
        ):
            score -= 1.5

    if want_battery or want_sport:
        days = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*days?", blob)]
        hours = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*hours?", blob)]
        mah = [float(m) for m in re.findall(r"(\d{3,5})\s*mah", blob)]
        if days and max(days) >= 5:
            score += 2.0
        elif hours and max(hours) >= 20:
            score += 1.5
        elif mah and max(mah) >= 300:
            score += 1.0
        elif hit(r"long.?battery", r"battery.?life"):
            score += 0.8

    cat = (brief.category or "").lower()
    if want_wireless or any(k in cat for k in ("headphone", "earbud", "headset", "earphone")):
        if hit(r"bluetooth", r"wireless", r"tws", r"true.?wireless"):
            score += 1.5
        if hit(r"wired", r"3\.5\s*mm") and not hit(r"bluetooth", r"wireless"):
            score -= 1.0
    if want_noise and hit(r"\banc\b", r"noise.?cancel", r"active.?noise"):
        score += 1.5

    # Generic soft-token overlap (values from the brief only).
    for pref in brief.soft_prefs:
        for tok in _toks(pref):
            if len(tok) > 3 and tok in blob:
                score += 0.25
    for tok in _toks(brief.goal)[:12]:
        if len(tok) > 4 and tok in blob and tok not in {
            "stock", "order", "under", "below", "place", "find", "test"
        }:
            score += 0.05

    # Accessory / category decoys when a wearable/audio device is requested.
    if any(k in cat for k in ("headphone", "earbud", "headset", "earphone")):
        if not AccessoryFence.looks_audio(listing):
            score -= 8.0
        if hit(r"\bcase\b", r"cover", r"cable only", r"replacement tip") and not hit(
            r"earbud", r"headphone", r"headset", r"earphone"
        ):
            score -= 5.0
        # Soft-token overlap with "sport/gym" must not promote fitness equipment.
        if hit(
            r"resistance band",
            r"yoga band",
            r"exercise band",
            r"dumbbell",
            r"kettlebell",
        ):
            score -= 10.0

    if pay_more and brief.budget and listing.price is not None and brief.budget > 0:
        ratio = float(listing.price) / float(brief.budget)
        if 0.15 <= ratio <= 0.85:
            score += 0.8
        elif ratio < 0.08:
            score -= 0.6
    return score


def rated_pref_evidence(brief: MissionBrief, listing: Listing) -> int:
    """0–2 tier: measurable evidence of soft prefs beats marketing adjectives."""
    axes = " ".join(brief.soft_prefs + [brief.goal]).lower()
    blob = listing.blob()
    tier = 0
    if re.search(r"sweat|water|rain|weather|waterproof|water.?resist|ipx?\d", axes, re.I):
        if re.search(r"ipx?\s*[5-9]|ip\s*x\s*[5-9]", blob, re.I):
            tier = max(tier, 2)
        elif re.search(r"ipx?\s*[0-9]|atm\s*\d|\d+\s*m\s*water", blob, re.I):
            tier = max(tier, 1)
    if re.search(r"battery|long.?lasting|all.?day", axes, re.I):
        if re.search(r"\d+\s*(mah|hours?|days?)", blob, re.I):
            tier = max(tier, max(tier, 1))
    if re.search(r"refresh|hz\b", axes, re.I):
        if re.search(r"\d{2,3}\s*hz", blob, re.I):
            tier = max(tier, 1)
    return tier


def preference_rank_key(brief: MissionBrief, listing: Listing) -> tuple:
    """Preference sort key — rated evidence first; never WeightScore coverage."""
    pay_more = bool(
        re.search(
            r"pay (?:a bit )?more|cheapest isn.?t",
            " ".join(brief.soft_prefs + [brief.goal]),
            re.I,
        )
    )
    return (
        rated_pref_evidence(brief, listing),
        preference_fit(brief, listing),
        1 if listing.viewed else 0,
        float(listing.price or 0.0) if pay_more else 0.0,
    )


def soft_search_queries(brief: MissionBrief) -> list[str]:
    """Compact catalog queries from category + soft axes (no LLM prose dumps).

    Long soft_pref sentences pollute BM25 toward unrelated fitness gear.
    Keep short modality + axis tokens only.
    """
    cat = (brief.category or "").strip()
    cat_l = cat.lower()
    axes = " ".join(brief.soft_prefs + [brief.goal]).lower()
    out: list[str] = []

    water = bool(re.search(r"sweat|water|rain|weather|waterproof|resist", axes, re.I))
    sport = bool(re.search(r"gym|run|sport|workout|fitness|training|bootcamp", axes, re.I))
    battery = bool(re.search(r"battery|long.?lasting|all.?day", axes, re.I))
    wireless = bool(re.search(r"wireless|bluetooth|cordless", axes, re.I))
    noise = bool(re.search(r"\bnoise\b|\banc\b|noise.?cancel", axes, re.I))

    audio = any(k in cat_l for k in ("headphone", "earbud", "headset", "earphone"))
    phone = any(k in cat_l for k in ("phone", "smartphone"))
    monitor = "monitor" in cat_l

    def push(parts: list[str]) -> None:
        q = " ".join(p for p in parts if p).strip()
        if q and q not in out:
            out.append(q)

    if audio:
        # Category noun first so BM25 stays in audio, not gym bands.
        push(["wireless", "earbuds", cat or "headphones"])
        if water:
            push(["waterproof", "sweat", "resistant", "wireless", "earbuds"])
        if sport:
            push(["sport", "wireless", "earbuds", "bluetooth"])
        if water and sport:
            push(["waterproof", "sport", "earbuds", "bluetooth"])
        if battery:
            push([cat or "earbuds", "bluetooth", "battery"])
        if noise:
            push(["noise", "cancelling", "wireless", "earbuds"])
        if wireless or True:
            push([cat or "headphones", "bluetooth", "wireless"])
    elif phone:
        push([cat or "smartphone"])
        if water:
            push([cat or "smartphone", "waterproof"])
    elif monitor:
        push([cat or "monitor"])
        if re.search(r"gaming|hz|refresh", axes, re.I):
            push(["gaming", "monitor", "hz"])
    else:
        # Generic: category + up to 4 short axis seeds (never full soft_pref prose).
        seeds: list[str] = []
        if water:
            seeds.extend(["waterproof", "resistant"])
        if sport:
            seeds.extend(["sport"])
        if battery:
            seeds.append("battery")
        if wireless:
            seeds.extend(["wireless", "bluetooth"])
        # Only short tokens from soft_prefs (drop filler prose).
        for pref in brief.soft_prefs:
            for tok in _toks(pref):
                if 3 <= len(tok) <= 12 and tok not in {
                    "that", "with", "from", "this", "have", "more", "than",
                    "under", "durable", "heavy", "daily", "kind", "every",
                    "rather", "actually", "holds", "goal", "here", "use",
                    "all", "for", "the", "and", "outdoor", "sessions",
                }:
                    seeds.append(tok)
                if len(seeds) >= 6:
                    break
        seeds = list(dict.fromkeys(seeds))[:6]
        push(([cat] if cat else []) + seeds)
        if cat and seeds:
            push(seeds + [cat])

    if brief.keywords:
        kw = [str(k) for k in brief.keywords[:5] if k and len(str(k)) <= 16]
        if kw:
            push(([cat] if cat else []) + kw)
    return out


def title_phrase_match(listing: Listing, phrase: Optional[str]) -> bool:
    if not phrase:
        return False
    title = _canon(listing.title or "")
    if not title:
        return False
    if _canon(phrase) in title:
        return True
    tokens = [t for t in _canon(phrase).split() if len(t) > 1]
    return bool(tokens) and all(tok in title for tok in tokens)


def composite_utility(brief: MissionBrief, listing: Listing, rank: int = 0) -> float:
    """Utility overlay on top of WeightScore for family-specific axes."""
    base = listing.weight.score_n if listing.weight else 0.0
    # Scale coverage into a familiar band, then add policy bonuses.
    score = base * 12.0
    if firm_ok(brief, listing):
        score += 10.0
    else:
        if listing.in_stock:
            score += 1.0
        if budget_ok(listing, brief.budget):
            score += 1.0
        if category_ok(listing, brief.category):
            score += 1.5
        for key, value in brief.attributes.items():
            if FactLens.matches(listing, key, value):
                score += 2.0
    prefer = brief.preferred_brand
    if prefer and brand_hits(listing, prefer):
        bonus = 3.0 if brief.family == "intent_decomposition" else 1.5
        bonus += 0.5 * brand_force(listing, prefer)
        score += bonus
    if brief.brand and brand_hits(listing, brief.brand):
        score += 1.0 + 0.25 * brand_force(listing, brief.brand)
    if brief.family in {"constraint_satisfaction", "justification"}:
        if ProofLedger.visible(brief, listing):
            score += 2.5
        if SkuExact.option_mentions(brief, listing):
            score += 2.0
    blob = listing.blob()
    for pref in brief.soft_prefs:
        for tok in _toks(pref):
            if len(tok) > 3 and tok in blob:
                score += 0.15
    for kw in brief.keywords:
        if _canon(kw) and _canon(kw) in _canon(blob):
            score += 0.2
    if brief.preferred_model and _canon(brief.preferred_model) in _canon(blob):
        score += 2.0
    if brief.preferred_color and _canon(brief.preferred_color) in _canon(
        blob + " " + listing.options
    ):
        score += 1.0
    if brief.family == "ranking":
        refresh, size, neg_price = ranking_tuple(listing)
        score += max(refresh, 0) * 10.0
        score += max(size, 0) * 1.0
        score += max(0.0, 5.0 + neg_price / 10000.0)
    if brief.family == "preference_reasoning":
        score += preference_fit(brief, listing) * 2.0
    score -= rank * 0.01
    if listing.price is not None and brief.budget and brief.family != "ranking":
        ratio = float(listing.price) / float(brief.budget)
        if 0.4 <= ratio <= 0.95:
            score += 0.3
    if brief.candidate_keys and listing.key in brief.candidate_keys:
        score += 50.0
    return score


# ---------------------------------------------------------------------------
# FirstCartLadder — event-family first-cart commitment
# ---------------------------------------------------------------------------


class FirstCartLadder:
    EVENT_FAMILIES = frozenset(
        {"intent_decomposition", "constraint_satisfaction", "justification"}
    )

    @classmethod
    def soft_brand(cls, brief: MissionBrief) -> Optional[str]:
        if brief.family == "justification":
            return brief.brand or brief.preferred_brand
        if brief.family == "intent_decomposition":
            return brief.preferred_brand
        if brief.family == "constraint_satisfaction":
            return brief.brand
        return brief.brand or brief.preferred_brand

    @classmethod
    def ambiguity_penalty(cls, listing: Listing) -> float:
        title = listing.title or ""
        pen = 0.0
        inch_hits = re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:\"|''|in(?:ch(?:es)?)?)\b", title, re.I
        )
        if len({h for h in inch_hits}) >= 2:
            pen -= 2.5
        if re.search(r"\b\d+\s*(?:\"|in(?:ch)?)\s*/\s*\d+", title, re.I):
            pen -= 2.0
        if re.search(r"\b[A-Za-z]?\d{2,}[A-Za-z0-9]*\s*/\s*[A-Za-z]?\d{2,}", title):
            pen -= 1.5
        if title.count(" / ") >= 1 and len(title) > 80:
            pen -= 0.5
        return pen

    @classmethod
    def budget_position(cls, brief: MissionBrief, listing: Listing) -> float:
        if brief.budget is None or listing.price is None:
            return 0.0
        budget = float(brief.budget)
        if budget <= 0:
            return 0.0
        ratio = float(listing.price) / budget
        if ratio > 1.0:
            return -5.0
        if ratio < 0.25:
            return -1.5
        if 0.5 <= ratio <= 0.98:
            return 1.2
        if 0.35 <= ratio < 0.5:
            return 0.3
        return 0.0

    @classmethod
    def ladder_score(cls, brief: MissionBrief, listing: Listing) -> float:
        s = float(composite_utility(brief, listing))
        s += cls.ambiguity_penalty(listing)
        s += cls.budget_position(brief, listing)
        brand = cls.soft_brand(brief)
        if brand:
            strength = brand_force(listing, brand)
            s += 1.0 * strength
            if strength >= 3:
                s += 1.0
            elif strength <= 1:
                s -= 0.8
        if brief.family == "justification":
            if ProofLedger.claimable(brief, listing):
                s += 4.0
            elif ProofLedger.visible(brief, listing):
                s += 1.0
            else:
                s -= 3.0
        if brief.attributes and SkuExact.option_mentions(brief, listing):
            s += 2.5
        if listing.viewed:
            s += 0.8
        if not AccessoryFence.option_ok(listing, brief.category):
            s -= 4.0
        return s

    @classmethod
    def climb(cls, brief: MissionBrief, listings: list[Listing]) -> Optional[Listing]:
        if not listings:
            return None
        buyable_pool = [
            v
            for v in listings
            if firm_ok(brief, v)
            and buyable(brief, v)
            and AccessoryFence.option_ok(v, brief.category)
        ]
        firm = buyable_pool or [
            v
            for v in listings
            if firm_ok(brief, v) and AccessoryFence.option_ok(v, brief.category)
        ]
        if not firm:
            return None
        brand = cls.soft_brand(brief)
        if brand:
            branded = [v for v in firm if brand_hits(v, brand)]
            if branded:
                strong = [v for v in branded if brand_force(v, brand) >= 2]
                firm = strong or branded
        if brief.family == "justification":
            claimable = [v for v in firm if ProofLedger.claimable(brief, v)]
            if claimable:
                firm = claimable
            else:
                evidenced = [v for v in firm if ProofLedger.visible(brief, v)]
                if evidenced:
                    firm = evidenced
        if brief.family in cls.EVENT_FAMILIES:
            firm.sort(key=lambda v: cls.ladder_score(brief, v), reverse=True)
            return firm[0]
        firm.sort(
            key=lambda v: (
                brand_force(v, brand) if brand else 0,
                composite_utility(brief, v),
            ),
            reverse=True,
        )
        return firm[0]

    @classmethod
    def shortlist(
        cls, brief: MissionBrief, listings: list[Listing], *, limit: int = 6
    ) -> list[Listing]:
        pool = [
            v
            for v in listings
            if firm_ok(brief, v)
            and buyable(brief, v)
            and AccessoryFence.option_ok(v, brief.category)
        ]
        if brief.family == "justification":
            claimable = [v for v in pool if ProofLedger.claimable(brief, v)]
            if claimable:
                pool = claimable
        brand = cls.soft_brand(brief)
        if brand:
            strong = [v for v in pool if brand_force(v, brand) >= 2]
            if strong:
                pool = strong
        pool.sort(key=lambda v: cls.ladder_score(brief, v), reverse=True)
        out: list[Listing] = []
        seen: set[str] = set()
        for v in pool:
            if v.product_id in seen:
                continue
            seen.add(v.product_id)
            out.append(v)
            if len(out) >= limit:
                break
        return out



# ---------------------------------------------------------------------------
# ElonAgent + engines (scout → inspect → select → close)
# ---------------------------------------------------------------------------


class ElonAgent:
    def __init__(self, arena: Arena):
        self.arena = arena
        self.brief = compose_brief(arena.query)
        self.pool: dict[str, Listing] = {}
        self.seen_pids: list[str] = []
        self._requery_n = 0
        self._qs = MorphMatch.tokenize_query(
            self.brief.compact or " ".join(self.brief.keywords) or self.brief.goal
        )
        self._qs_w = [
            (
                1.0
                if (
                    re.search(r"\d", t)
                    or t in MorphLexicon.COLOR
                    or t in MorphLexicon.SIZE_MAP.values()
                )
                else (0.5 if t in MorphLexicon.FILLERS else 0.75)
            )
            for t in self._qs
        ]

    # -- tool wrappers ------------------------------------------------------

    def search(self, query: str, *, k: int = 10, in_stock: Optional[bool] = True) -> list[Listing]:
        if not self.arena.can("search"):
            if self.arena.can("filter"):
                return self.filter_catalog(
                    category=self.brief.category,
                    brand=self.brief.brand or self.brief.preferred_brand,
                    max_price=self.brief.budget,
                    k=k,
                    in_stock=in_stock,
                    allow_search_fallback=False,
                )
            return []
        args: dict[str, Any] = {"query": query, "k": k}
        if in_stock is not None:
            args["in_stock"] = in_stock
        obs = self.arena.invoke("search", **args)
        if obs.get("error") == "tool_unavailable":
            return []
        return self._ingest(obs)

    def filter_catalog(
        self,
        *,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        max_price: Optional[float] = None,
        k: int = 10,
        in_stock: Optional[bool] = True,
        allow_search_fallback: bool = True,
    ) -> list[Listing]:
        if not self.arena.can("filter"):
            if allow_search_fallback and self.arena.can("search"):
                bits = [p for p in (brand, category) if p]
                if max_price is not None:
                    bits.append(str(max_price))
                q = " ".join(bits).strip() or (self.brief.category or "product")
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
        obs = self.arena.invoke("filter", **args)
        if obs.get("error") == "tool_unavailable":
            return []
        out = self._ingest(obs)
        if brand:
            for v in out:
                v.source_brand = brand
                if not (v.brand or "").strip() and self.brief.brand:
                    v.brand = brand
                self.pool[v.key] = v
        return out

    def view(self, product_id: str) -> list[Listing]:
        if not self.arena.can("view"):
            if self.arena.can("compare"):
                return self.compare([product_id])
            return []
        obs = self.arena.invoke("view", product_id=product_id)
        if obs.get("error") == "tool_unavailable":
            return []
        variants = listings_from_view(obs)
        prev_brand = ""
        prev_source = ""
        for prev in self.pool.values():
            if prev.product_id == product_id:
                prev_brand = prev.brand or prev_brand
                prev_source = prev.source_brand or prev_source
        for v in variants:
            if not (v.brand or "").strip():
                v.brand = prev_brand
            v.source_brand = v.source_brand or prev_source or v.brand
            self._stamp_weight(v)
            self.pool[v.key] = v
            if v.product_id not in self.seen_pids:
                self.seen_pids.append(v.product_id)
        return variants

    def compare(self, product_ids: list[str]) -> list[Listing]:
        if not self.arena.can("compare"):
            out: list[Listing] = []
            for pid in product_ids[:5]:
                if self.arena.steps_left() <= 0 or self.arena.finished:
                    break
                out.extend(self.view(pid))
            return out
        obs = self.arena.invoke("compare", product_ids=product_ids[:5])
        if obs.get("error") == "tool_unavailable":
            return []
        out: list[Listing] = []
        for row in obs.get("compared") or []:
            out.extend(listings_from_view(row))
        for v in out:
            self._stamp_weight(v)
            self.pool[v.key] = v
            if v.product_id not in self.seen_pids:
                self.seen_pids.append(v.product_id)
        return out

    def inspect_stock(self, product_id: str, sku: str) -> Listing:
        key = f"{product_id}::{sku}"
        v = self.pool.get(key) or Listing(product_id=product_id, sku=sku)
        if not self.arena.can("inspect_stock"):
            return v
        obs = self.arena.invoke("inspect_stock", product_id=product_id, sku=sku)
        if obs.get("error"):
            return v
        v.price = _num(obs.get("price"))
        v.in_stock = bool(obs.get("in_stock", v.in_stock))
        v.currency = str(obs.get("currency") or v.currency)
        self.pool[key] = v
        return v

    def inspect_cart(self) -> dict[str, Any]:
        if not self.arena.can("inspect_cart"):
            return {}
        obs = self.arena.invoke("inspect_cart")
        return {} if obs.get("error") else obs

    def message(self, content: str) -> None:
        if not self.arena.can("message"):
            return
        self.arena.invoke("message", content=content)
        if self.arena.last_shopper_note:
            self.brief = RevealProbe.absorb_note(self.brief, self.arena.last_shopper_note)
            self.arena.last_shopper_note = None

    def add_to_cart(self, product_id: str, sku: str) -> dict[str, Any]:
        if not self.arena.can("add_to_cart"):
            return {"error": "tool_unavailable", "requested": "add_to_cart"}
        obs = self.arena.invoke("add_to_cart", product_id=product_id, sku=sku)
        self._absorb_shopper(pending=self.pool.get(f"{product_id}::{sku}"))
        return obs

    def remove_from_cart(self, product_id: str, sku: str) -> None:
        if self.arena.can("remove_from_cart"):
            self.arena.invoke("remove_from_cart", product_id=product_id, sku=sku)
            self._absorb_shopper()

    def place_order(
        self,
        listing: Listing,
        justification: Optional[dict] = None,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        if not self.arena.can("place_order"):
            return {"error": "tool_unavailable", "requested": "place_order"}
        if not force and not buyable(self.brief, listing):
            return {}
        if (
            self.brief.family == "justification"
            and self.arena.steps_left() > 0
            and not force
        ):
            listing = self.inspect_stock(listing.product_id, listing.sku)
            if not buyable(self.brief, listing):
                return {}
            if not ProofLedger.claimable(self.brief, listing):
                return {}
        args: dict[str, Any] = {"product_id": listing.product_id, "sku": listing.sku}
        if justification is not None:
            args["justification"] = justification
        elif self.brief.family == "justification":
            just = ClaimForge.forge(self.brief, listing)
            fields = {c.get("field") for c in (just.get("claims") or [])}
            gate = ProofLedger.gate_key(self.brief)
            if gate and gate not in fields:
                if not force:
                    return {}
            args["justification"] = just
        return self.arena.invoke("place_order", **args)

    def _stamp_weight(self, listing: Listing, rank: int = 0) -> None:
        listing.weight = WeightScore.weigh(
            listing,
            self.brief,
            rank=rank,
            query_tokens=self._qs,
            token_weights=self._qs_w,
        )

    def _ingest(self, obs: dict[str, Any]) -> list[Listing]:
        out: list[Listing] = []
        for i, item in enumerate(obs.get("results") or []):
            if not isinstance(item, dict) or not item.get("product_id"):
                continue
            v = listing_from_hit(item)
            prev = self.pool.get(v.key)
            if prev and prev.viewed:
                prev.price = v.price if v.price is not None else prev.price
                prev.in_stock = v.in_stock
                v = prev
            self._stamp_weight(v, rank=i)
            self.pool[v.key] = v
            out.append(v)
            if v.product_id not in self.seen_pids:
                self.seen_pids.append(v.product_id)
        return out

    def _absorb_shopper(self, pending: Optional[Listing] = None) -> None:
        note = self.arena.last_shopper_note
        if not note:
            return
        self.brief = RevealProbe.absorb_note(self.brief, note)
        if pending is not None and re.search(
            r"\b(not available|out of stock|sold out|still available|stockout|unavailable)\b",
            note,
            re.I,
        ):
            pending.in_stock = False
            self.pool[pending.key] = pending
        if pending is not None:
            price_jump = re.search(
                r"(?:from\s+)?(\d+(?:\.\d+)?)\s*(?:to|->)\s*(\d+(?:\.\d+)?)\s*php",
                note,
                re.I,
            )
            if price_jump:
                pending.price = float(price_jump.group(2))
                self.pool[pending.key] = pending
        # Prefer recover/search budget over courtesy replies when steps are tight
        # or a market event already forces a cart change.
        market_hit = bool(
            pending is not None
            and re.search(
                r"\b(not available|out of stock|sold out|price|increased|unavailable)\b",
                note,
                re.I,
            )
        )
        reply_budget = self.arena.steps_left() > (
            4 if self.brief.family in {"preference_reasoning", "retrieval_recall"} else 2
        )
        if (
            shopper_wants_reply(note)
            and reply_budget
            and not market_hit
        ):
            reply = self._reply(note, pending)
            self.arena.last_shopper_note = None
            self.message(reply)
        else:
            self.arena.last_shopper_note = None

    def _reply(self, note: str, choice: Optional[Listing]) -> str:
        if choice is not None:
            bits = [
                f"I'm going with {choice.title or choice.product_id}",
                f"sku {choice.sku}",
            ]
            if choice.price is not None:
                bits.append(f"at {choice.price} {choice.currency}")
            inches = FactLens.screen_inches(choice)
            if inches is not None:
                bits.append(f"measured_screen={inches:g}in")
            bits.append(
                "firm-requirements: met"
                if firm_ok(self.brief, choice)
                else "firm-requirements: still checking"
            )
            return "; ".join(bits) + ". I'll verify live price/stock before ordering."
        lower = note.lower()
        if "balance" in lower or "budget" in lower or "php" in lower:
            return (
                f"Understood — I'll stay within {self.brief.budget} {self.brief.currency} "
                "and re-check stock/price before ordering."
            )
        if "expensive" in lower or "price" in lower:
            return "Thanks for the heads-up — I'll re-check the live price and switch if needed."
        return (
            "Got it. I'll apply your latest requirements (specs + budget + in-stock) "
            "and only order a firm-compliant item."
        )

    # -- pipeline -----------------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        try:
            self.arena.think(f"family={self.brief.family} budget={self.brief.budget}")
            if self.brief.needs_clarification and self.arena.steps_left() > 10:
                self._clarify_once()
            if self.arena.finished:
                return self.arena.dialogue

            scouted = ScoutEngine(self).retrieve()
            if self.arena.finished:
                return self.arena.dialogue

            inspected = InspectEngine(self).survey(scouted)
            if self.arena.finished:
                return self.arena.dialogue

            choice = SelectEngine(self).pick(inspected)
            if choice is None:
                fallback = [
                    v
                    for v in self.pool.values()
                    if v.in_stock and budget_ok(v, self.brief.budget)
                ]
                if self.brief.family == "preference_reasoning":
                    fallback = [v for v in fallback if firm_ok(self.brief, v)]
                elif self.brief.family == "retrieval_recall":
                    phrase = _recall_phrase(self.brief.goal)
                    phrase_hits = [
                        v for v in fallback if title_phrase_match(v, phrase)
                    ]
                    fallback = phrase_hits  # empty is OK — force_done may re-search
                if self.brief.candidate_keys:
                    fallback = [
                        v for v in fallback if v.key in self.brief.candidate_keys
                    ] or fallback
                if self.brief.family == "preference_reasoning":
                    fallback.sort(
                        key=lambda v: preference_rank_key(self.brief, v), reverse=True
                    )
                elif self.brief.family == "retrieval_recall":
                    phrase = _recall_phrase(self.brief.goal)
                    fallback.sort(
                        key=lambda v: (
                            1 if title_phrase_match(v, phrase) else 0,
                            composite_utility(self.brief, v),
                        ),
                        reverse=True,
                    )
                else:
                    fallback.sort(
                        key=lambda v: composite_utility(self.brief, v), reverse=True
                    )
                choice = fallback[0] if fallback else None
            if choice is not None and not self.arena.finished:
                CloseEngine(self).seal(choice)

            if not self.arena.finished and self.arena.steps_left() > 0:
                self._force_done()
        except Exception as exc:  # noqa: BLE001
            self.arena.dialogue.append(
                {
                    "role": "system",
                    "content": f"ELON_v19 error: {type(exc).__name__}: {exc}",
                }
            )
            try:
                if not self.arena.finished and self.arena.steps_left() > 0:
                    self._force_done()
            except Exception:
                pass
        return self.arena.dialogue

    def _clarify_once(self) -> None:
        if self.brief.family == "constraint_satisfaction":
            self.message(
                "Before I buy: what exact hardware spec must the variant meet "
                "(screen size, resolution, refresh rate, or RAM)?"
            )
        elif self.brief.family == "preference_reasoning":
            self.message(
                "To pick the best match: which priorities matter most for your use case "
                "(battery, water/sweat resistance, fit, wireless, brand)?"
            )
        elif self.brief.family == "intent_decomposition":
            self.message(
                "Quick check: which requirements are firm versus nice-to-have, "
                "and is the preferred brand optional if a firm-spec item is only available from another brand?"
            )
        else:
            self.message(
                "Before ordering, any must-have specs or budget updates I should apply?"
            )
        self._absorb_shopper()
        if self.brief.family == "constraint_satisfaction" and self.brief.attributes:
            self.brief.search_queries = self._firm_queries()[:4]
            self._requery_n = 0
            self._qs = MorphMatch.tokenize_query(
                self.brief.compact or " ".join(self.brief.keywords) or self.brief.goal
            )

    def _firm_queries(self) -> list[str]:
        brand = self.brief.brand or self.brief.preferred_brand or ""
        cat = self.brief.category or ""
        out: list[str] = []
        for key, val in self.brief.attributes.items():
            kind, wanted = FactLens.normalize_want(key, str(val))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                inches = float(wanted)
                out.append(f"{brand} {inches:g} inch {cat}".strip())
                out.append(f"{inches:g} inch {cat} {brand}".strip())
                if inches >= 7.0 and cat and "phone" in cat.lower():
                    out.append(f"{brand} foldable {inches:g} inch smartphone".strip())
                    out.append(f"{brand} folding phone {inches:g}".strip())
                    out.append(f"{brand} z fold {inches:g} inch".strip())
            elif kind == "resolution" and wanted:
                token = str(wanted)
                out.append(f"{brand} {cat} {token}".strip())
                out.append(f"{brand} {token} {cat}".strip())
                if token in {"hd", "720p"}:
                    out.append(f"{brand} {cat} 720p HD".strip())
                    out.append(f"{brand} 1280x720 {cat}".strip())
                elif token in {"1080p", "fullhd", "fhd"}:
                    out.append(f"{brand} {cat} 1080p Full HD".strip())
                    out.append(f"{brand} 1920x1080 {cat}".strip())
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
        return uniq

    def _spec_requery(self) -> None:
        if self._requery_n >= 2 or self.arena.steps_left() <= 5:
            return
        self._requery_n += 1
        for q in self._firm_queries()[:3]:
            if self.arena.steps_left() <= 5:
                break
            self.search(q, k=10, in_stock=True)
        bits = [
            self.brief.brand or self.brief.preferred_brand or "",
            self.brief.category or "",
            " ".join(str(v) for v in self.brief.attributes.values()),
        ]
        q = " ".join(b for b in bits if b).strip()
        if q and self.arena.steps_left() > 5:
            self.search(q, k=10, in_stock=True)
        if self.arena.steps_left() > 5:
            self.filter_catalog(
                category=self.brief.category,
                brand=self.brief.brand
                or (
                    self.brief.preferred_brand
                    if self.brief.family
                    in {
                        "justification",
                        "constraint_satisfaction",
                        "intent_decomposition",
                    }
                    else None
                ),
                max_price=self.brief.budget,
                k=10,
                in_stock=True,
            )
        self._absorb_shopper()

    def _force_done(self) -> None:
        if self.arena.finished or self.arena.steps_left() <= 0:
            return
        if (
            self.brief.family
            in {"intent_decomposition", "constraint_satisfaction", "justification"}
            and self.brief.attributes
            and self.arena.steps_left() > 5
            and not any(
                firm_ok(self.brief, v) and ProofLedger.visible(self.brief, v)
                for v in self.pool.values()
            )
        ):
            self._spec_requery()
        pool = [v for v in self.pool.values() if buyable(self.brief, v)]
        hard = [v for v in pool if firm_ok(self.brief, v)]
        if hard:
            pool = hard
        elif self.brief.family in FirstCartLadder.EVENT_FAMILIES:
            if self.arena.steps_left() > 4 and self.brief.attributes:
                self._spec_requery()
                pool = [v for v in self.pool.values() if buyable(self.brief, v)]
                hard = [v for v in pool if firm_ok(self.brief, v)]
                if hard:
                    pool = hard
        if self.brief.family == "justification":
            claimable = [v for v in pool if ProofLedger.claimable(self.brief, v)]
            if claimable:
                pool = claimable
        brand = FirstCartLadder.soft_brand(self.brief)
        if brand:
            branded = [v for v in pool if brand_hits(v, brand)]
            if branded:
                pool = branded
        if self.brief.candidate_keys:
            cand = [v for v in pool if v.key in self.brief.candidate_keys]
            if cand:
                pool = cand
        if not pool:
            pool = [
                v
                for v in self.pool.values()
                if v.in_stock and AccessoryFence.accepts(v, self.brief.category)
            ]
        if not pool:
            pool = [v for v in self.pool.values() if v.in_stock]
        if not pool:
            pool = list(self.pool.values())
        if not pool:
            if self.arena.steps_left() > 1:
                self.search(
                    " ".join(self.brief.keywords[:4]) or self.brief.goal[:40],
                    k=5,
                    in_stock=True,
                )
                pool = [
                    v
                    for v in self.pool.values()
                    if v.in_stock and AccessoryFence.accepts(v, self.brief.category)
                ] or [v for v in self.pool.values() if v.in_stock] or list(
                    self.pool.values()
                )
            if not pool:
                return
        if self.brief.family == "ranking":
            pool.sort(key=ranking_tuple, reverse=True)
        elif self.brief.family == "preference_reasoning":
            pool = [v for v in pool if firm_ok(self.brief, v)] or pool
            pool.sort(
                key=lambda v: preference_rank_key(self.brief, v), reverse=True
            )
        elif self.brief.family == "retrieval_recall":
            phrase = _recall_phrase(self.brief.goal)
            matched = [v for v in pool if title_phrase_match(v, phrase)]
            if matched:
                pool = matched
            elif phrase and self.arena.steps_left() > 3:
                self.search(phrase, k=8, in_stock=True)
                pool = [
                    v
                    for v in self.pool.values()
                    if buyable(self.brief, v) and title_phrase_match(v, phrase)
                ]
                if not pool:
                    return
            elif phrase:
                return
            pool.sort(
                key=lambda v: (
                    1 if title_phrase_match(v, phrase) else 0,
                    composite_utility(self.brief, v),
                ),
                reverse=True,
            )
        elif self.brief.family in FirstCartLadder.EVENT_FAMILIES:
            pool.sort(
                key=lambda v: FirstCartLadder.ladder_score(self.brief, v), reverse=True
            )
        else:
            pool.sort(key=lambda v: composite_utility(self.brief, v), reverse=True)
        best = pool[0]
        if not best.viewed and self.arena.steps_left() > 2:
            variants = self.view(best.product_id)
            ok = [
                v for v in variants if firm_ok(self.brief, v) and buyable(self.brief, v)
            ]
            if ok:
                ok.sort(key=lambda v: composite_utility(self.brief, v), reverse=True)
                best = ok[0]
            else:
                best = self.pool.get(best.key, best)

        if self.arena.steps_left() > 1:
            self.add_to_cart(best.product_id, best.sku)
            if self.arena.finished:
                return
            best = self.pool.get(best.key, best)
            if self.arena.steps_left() > 0:
                best = self.inspect_stock(best.product_id, best.sku)
                self._absorb_shopper(pending=best)

        if self.arena.finished:
            return

        if self.arena.steps_left() > 3 and not buyable(self.brief, best):
            self.remove_from_cart(best.product_id, best.sku)
            alt = CloseEngine(self).recover(exclude_keys={best.key}, previous=best)
            if alt is not None:
                best = alt
            if self.arena.steps_left() > 1:
                self.add_to_cart(best.product_id, best.sku)
                if self.arena.finished:
                    return
                if self.arena.steps_left() > 0:
                    best = self.inspect_stock(best.product_id, best.sku)

        if self.arena.finished:
            return
        if self.arena.steps_left() > 0:
            if not buyable(self.brief, best):
                alts = [
                    v
                    for v in self.pool.values()
                    if v.key != best.key and buyable(self.brief, v)
                ]
                hard_alts = [v for v in alts if firm_ok(self.brief, v)]
                pick_from = hard_alts or alts
                if pick_from:
                    pick_from.sort(
                        key=lambda v: composite_utility(self.brief, v), reverse=True
                    )
                    best = pick_from[0]
                    if self.arena.steps_left() > 1:
                        self.add_to_cart(best.product_id, best.sku)
            self.place_order(best, force=True)
            if self.arena.finished:
                return
            if self.arena.steps_left() > 1:
                self.add_to_cart(best.product_id, best.sku)
            if self.arena.steps_left() > 0:
                self.place_order(best, force=True)



class ScoutEngine:
    """Retrieve / search / filter / brand hunt / firm-attr requeries."""

    def __init__(self, agent: ElonAgent):
        self.agent = agent

    def retrieve(self) -> list[Listing]:
        brief = self.agent.brief
        arena = self.agent.arena
        found: list[Listing] = []

        for key in brief.candidate_keys:
            pid, _, sku = key.partition("::")
            if pid:
                v = Listing(product_id=pid, sku=sku or "default")
                found.append(v)
                self.agent.pool.setdefault(v.key, v)

        if brief.family == "ranking" and brief.candidate_keys:
            return list(found)

        queries = list(brief.search_queries)
        firm_qs = self.agent._firm_queries()
        if firm_qs and brief.family in {
            "intent_decomposition",
            "constraint_satisfaction",
            "justification",
        }:
            queries = firm_qs + [q for q in queries if q not in firm_qs]

        if brief.family in {"constraint_satisfaction", "justification"}:
            brand = brief.brand or brief.preferred_brand or ""
            cat = brief.category or ""
            core = firm_qs[:3] if firm_qs else []
            if brand and cat:
                core.append(f"{brand} {cat}".strip())
            firm_toks = [str(v).lower() for v in brief.attributes.values()]
            for q in list(brief.search_queries):
                ql = q.lower()
                if any(t and t.replace("in", "")[:4] in ql for t in firm_toks) or any(
                    t in ql for t in ("720p", "1080p", "hd", "inch", "hz")
                ):
                    if q not in core:
                        core.append(q)
                if len(core) >= 4:
                    break
            queries = core[:5]

        if brief.family == "intent_decomposition" and brief.preferred_brand:
            bits = [
                brief.preferred_brand,
                brief.category or "smartphone",
                " ".join(str(v) for v in brief.attributes.values()),
            ]
            brand_q = " ".join(b for b in bits if b)
            queries = [brand_q, brief.preferred_brand] + [
                q for q in queries if q not in {brand_q, brief.preferred_brand}
            ]
            queries = firm_qs[:2] + [q for q in queries if q not in firm_qs[:2]]

        if brief.family == "retrieval_recall":
            phrase = _recall_phrase(brief.goal)
            if phrase:
                toks = [t for t in phrase.split() if t]
                alts = [
                    phrase,
                    phrase.replace(" ", ""),
                    " ".join(reversed(toks)) if len(toks) > 1 else phrase,
                    f"{phrase} {brief.category or ''}".strip(),
                ]
                if len(toks) > 1:
                    alts.extend(toks)
                queries = alts + [q for q in queries if q not in alts]

        if brief.family == "preference_reasoning":
            extras = soft_search_queries(brief)
            queries = extras + [q for q in queries if q not in extras]

        if brief.family == "recovery" and brief.preferred_model:
            queries.insert(0, brief.preferred_model)
            if brief.preferred_color:
                queries.insert(
                    0, f"{brief.preferred_color} {brief.preferred_model}"
                )

        if brief.family == "justification":
            brand = brief.brand or brief.preferred_brand
            bits = [brand or "", brief.category or ""]
            for _, val in brief.attributes.items():
                bits.append(str(val))
            queries.insert(0, " ".join(b for b in bits if b))

        # Compact ELON-style keyword variants — skip on TF2 so phrase hits stay first.
        if brief.family != "retrieval_recall":
            for kv in self._keyword_variants():
                if kv not in queries:
                    queries.append(kv)

        if (
            brief.family
            in {"intent_decomposition", "constraint_satisfaction", "justification"}
            and firm_qs
            and arena.steps_left() > 6
        ):
            for q in firm_qs[:2]:
                if arena.steps_left() <= 6:
                    break
                found.extend(self.agent.search(q, k=10, in_stock=True))
                self.agent._absorb_shopper()
                if any(firm_ok(brief, v) for v in self.agent.pool.values()):
                    break

        if (
            brief.family == "intent_decomposition"
            and brief.preferred_brand
            and arena.steps_left() > 4
        ):
            found.extend(
                self.agent.filter_catalog(
                    category=brief.category,
                    brand=brief.preferred_brand,
                    max_price=brief.budget,
                    k=10,
                    in_stock=True,
                )
            )
            self.agent._absorb_shopper()

        for q in queries:
            if arena.steps_left() <= 6:
                break
            if firm_qs and q in firm_qs[:2] and brief.family in {
                "intent_decomposition",
                "constraint_satisfaction",
                "justification",
            }:
                continue
            found.extend(self.agent.search(q, k=10, in_stock=True))
            self.agent._absorb_shopper()
            if brief.family == "retrieval_recall":
                phrase = _recall_phrase(brief.goal)
                matched_n = len(
                    {
                        v.product_id
                        for v in self.agent.pool.values()
                        if title_phrase_match(v, phrase)
                    }
                )
                # Capability floor — not a suite roster size. Keep surveying
                # while matches are scarce; stop once a solid distinct set exists.
                if matched_n >= 5:
                    break
                if matched_n >= 3 and arena.turn >= max(6, arena.max_steps // 3):
                    break
                if len(self.agent.seen_pids) >= 15:
                    break
            if brief.family == "intent_decomposition":
                branded = [
                    v
                    for v in self.agent.pool.values()
                    if brief.preferred_brand and brand_hits(v, brief.preferred_brand)
                ]
                if branded and any(firm_ok(brief, v) for v in branded):
                    break
                if branded and len(found) >= 10:
                    break
            elif brief.family in {"constraint_satisfaction", "justification"}:
                ok = [
                    v
                    for v in self.agent.pool.values()
                    if firm_ok(brief, v) and ProofLedger.visible(brief, v)
                ]
                if ok:
                    break
            elif brief.family != "retrieval_recall" and len(found) >= 8:
                break

        if arena.steps_left() > 6 and brief.family not in {
            "preference_reasoning",
        }:
            brand = brief.brand
            if (
                not brand
                and brief.preferred_brand
                and brief.family != "intent_decomposition"
            ):
                brand = brief.preferred_brand
            # Retrieval: category-scoped filter expands recall without brand stamps.
            if brief.family == "retrieval_recall":
                brand = None
            found.extend(
                self.agent.filter_catalog(
                    category=brief.category,
                    brand=brand,
                    max_price=brief.budget,
                    k=10,
                    in_stock=True,
                )
            )
            self.agent._absorb_shopper()
            if (
                brief.family == "intent_decomposition"
                and brief.preferred_brand
                and arena.steps_left() > 5
            ):
                found.extend(
                    self.agent.filter_catalog(
                        category=brief.category,
                        brand=brief.preferred_brand,
                        max_price=brief.budget,
                        k=10,
                        in_stock=True,
                    )
                )
                self.agent._absorb_shopper()

        uniq: dict[str, Listing] = {}
        for v in found:
            uniq[v.key] = self.agent.pool.get(v.key, v)
        return list(uniq.values())

    def _keyword_variants(self) -> list[str]:
        """ELON-style compact keyword reformulations (add/drop one token)."""
        brief = self.agent.brief
        base = MorphMatch.tokenize_query(
            brief.compact or " ".join(brief.keywords) or brief.goal
        )[:8]
        if not base:
            return []
        out: list[str] = [" ".join(base)]
        # Drop fillers
        trimmed = [t for t in base if t not in MorphLexicon.FILLERS]
        if trimmed and trimmed != base:
            out.append(" ".join(trimmed))
        # Bridge synonyms
        bridged = []
        for t in base:
            bridged.append(MorphLexicon.BRIDGES.get(t, t))
        if bridged != base:
            out.append(" ".join(bridged))
        brand = brief.brand or brief.preferred_brand
        if brand:
            out.append(f"{brand} {' '.join(base[:5])}")
        return list(dict.fromkeys(out))[:4]


class InspectEngine:
    """view / compare with family-specific min views."""

    def __init__(self, agent: ElonAgent):
        self.agent = agent

    def survey(self, candidates: list[Listing]) -> list[Listing]:
        brief = self.agent.brief
        arena = self.agent.arena
        product_ids: list[str] = []
        for v in candidates:
            if v.product_id not in product_ids:
                product_ids.append(v.product_id)

        min_views = 3 if brief.family == "retrieval_recall" else 2
        if brief.family == "preference_reasoning":
            min_views = 4
        if brief.family == "ranking" and brief.candidate_keys:
            min_views = min(
                5, len({k.split("::", 1)[0] for k in brief.candidate_keys})
            )
        if brief.family == "justification":
            min_views = 3
        if brief.family == "recovery":
            min_views = 2
        if brief.family == "intent_decomposition":
            min_views = 2
        if brief.family == "constraint_satisfaction":
            min_views = 1

        viewed: list[Listing] = []
        use_compare = (
            brief.family
            not in {
                "constraint_satisfaction",
                "justification",
                "intent_decomposition",
            }
            and arena.can("compare")
        )

        if brief.family == "intent_decomposition" and brief.preferred_brand:
            branded_ids = [
                pid
                for pid in product_ids
                if any(
                    brand_hits(v, brief.preferred_brand)
                    for v in self.agent.pool.values()
                    if v.product_id == pid
                )
            ]
            other_ids = [pid for pid in product_ids if pid not in branded_ids]
            product_ids = branded_ids + other_ids

        if brief.family in {
            "intent_decomposition",
            "constraint_satisfaction",
            "justification",
        } and brief.attributes:

            def _pid_firmness(pid: str) -> tuple[int, float]:
                vs = [v for v in self.agent.pool.values() if v.product_id == pid]
                if not vs:
                    return (0, 0.0)
                best = max(vs, key=lambda v: composite_utility(brief, v))
                return (
                    1 if firm_ok(brief, best) else 0,
                    composite_utility(brief, best),
                )

            product_ids.sort(key=_pid_firmness, reverse=True)

        # Preference family: lead inspect with stronger soft-axis evidence first.
        if brief.family == "preference_reasoning":

            def _pid_pref(pid: str) -> tuple:
                vs = [v for v in self.agent.pool.values() if v.product_id == pid]
                if not vs:
                    return (0.0, 0, 0.0)
                best = max(vs, key=lambda v: preference_fit(brief, v))
                return preference_rank_key(brief, best)

            product_ids.sort(key=_pid_pref, reverse=True)

        # Retrieval family: lead with titles that carry every recalled phrase token.
        if brief.family == "retrieval_recall":
            phrase = _recall_phrase(brief.goal)

            def _pid_recall(pid: str) -> tuple:
                vs = [v for v in self.agent.pool.values() if v.product_id == pid]
                if not vs:
                    return (0, 0.0)
                best = max(vs, key=lambda v: 1 if title_phrase_match(v, phrase) else 0)
                return (
                    1 if title_phrase_match(best, phrase) else 0,
                    composite_utility(brief, best),
                )

            product_ids.sort(key=_pid_recall, reverse=True)

        while (
            product_ids
            and arena.steps_left() > 4
            and len({v.product_id for v in viewed}) < min_views
        ):
            batch = product_ids[:5]
            product_ids = product_ids[5:]
            if use_compare and len(batch) >= 2:
                viewed.extend(self.agent.compare(batch))
            else:
                for pid in batch:
                    if arena.steps_left() <= 4:
                        break
                    viewed.extend(self.agent.view(pid))
                    if len({v.product_id for v in viewed}) >= min_views:
                        break
            self.agent._absorb_shopper(pending=viewed[0] if viewed else None)

        if len({v.product_id for v in viewed}) < min_views:
            ranked = sorted(
                self.agent.pool.values(),
                key=lambda v: (
                    v.weight.score_n if v.weight else 0.0,
                    composite_utility(brief, v),
                ),
                reverse=True,
            )
            for v in ranked:
                if arena.steps_left() <= 4:
                    break
                if any(x.product_id == v.product_id and x.viewed for x in viewed):
                    continue
                viewed.extend(self.agent.view(v.product_id))
                self.agent._absorb_shopper(pending=v)
                if len({x.product_id for x in viewed}) >= min_views:
                    break

        out = [
            v
            for v in self.agent.pool.values()
            if v.viewed or v.key in {c.key for c in candidates}
        ]
        # Re-weigh after views
        for i, v in enumerate(out):
            self.agent._stamp_weight(v, rank=i)
        if brief.family == "ranking" and brief.candidate_keys:
            out = [v for v in out if v.key in brief.candidate_keys]
            out.sort(key=ranking_tuple, reverse=True)
        else:
            out.sort(
                key=lambda v: (
                    v.weight.score_n if v.weight else 0.0,
                    composite_utility(brief, v),
                ),
                reverse=True,
            )
        return out


class SelectEngine:
    """WeightScore rank + Fibonacci windows; optional LLM pick."""

    SELECT_BATCH = 17
    MAX_SELECT_TRIES = 1

    def __init__(self, agent: ElonAgent):
        self.agent = agent

    def pick(self, shortlist: list[Listing]) -> Optional[Listing]:
        brief = self.agent.brief

        if brief.family == "ranking" and brief.candidate_keys:
            pool = [
                v
                for v in shortlist
                if v.key in brief.candidate_keys
                and v.in_stock
                and budget_ok(v, brief.budget)
            ]
            if not pool:
                pool = [
                    v
                    for v in shortlist
                    if v.key in brief.candidate_keys and v.in_stock
                ]
            if not pool:
                return None
            pool.sort(key=ranking_tuple, reverse=True)
            return pool[0]

        # Assign coverage weights + confidence ranks
        ranked = list(shortlist)
        for i, v in enumerate(ranked):
            self.agent._stamp_weight(v, rank=i)
        ranked.sort(key=lambda v: -(v.weight.score if v.weight else 0.0))
        if ranked:
            last = ranked[0].weight.score if ranked[0].weight else 0.0
            wrank = 0
            for v in ranked:
                sc = v.weight.score if v.weight else 0.0
                if sc < last:
                    last = sc
                    wrank += 1
                if v.weight:
                    v.weight.set_confidence(wrank)
            ranked.sort(key=lambda v: -(v.weight.score_n if v.weight else 0.0))

        hard = [v for v in ranked if firm_ok(brief, v)]
        pool = hard or [v for v in ranked if v.in_stock]
        if not pool:
            return None

        if brief.family == "intent_decomposition" and brief.preferred_brand:
            branded = [v for v in hard if brand_hits(v, brief.preferred_brand)]
            if branded:
                pool = branded

        if brief.family == "recovery":
            exact = [
                v
                for v in hard
                if (
                    not brief.preferred_model
                    or _canon(brief.preferred_model) in _canon(v.blob())
                )
                and (
                    not brief.preferred_color
                    or _canon(brief.preferred_color)
                    in _canon(v.blob() + " " + v.options)
                )
            ]
            if exact:
                pool = exact

        if brief.family == "preference_reasoning":
            # Never fall back to off-category in_stock junk (fitness bands etc.).
            pref_pool = [
                v
                for v in shortlist
                if firm_ok(brief, v) and buyable(brief, v)
            ]
            if not pref_pool and self.agent.arena.steps_left() > 4:
                for q in soft_search_queries(brief)[:3]:
                    if self.agent.arena.steps_left() <= 3:
                        break
                    self.agent.search(q, k=10, in_stock=True)
                self.agent._absorb_shopper()
                pref_pool = [
                    v
                    for v in self.agent.pool.values()
                    if firm_ok(brief, v) and buyable(brief, v)
                ]
            if not pref_pool:
                return None
            viewed = [v for v in pref_pool if v.viewed]
            if viewed:
                pref_pool = viewed
            pref_pool = sorted(
                pref_pool,
                key=lambda v: preference_rank_key(brief, v),
                reverse=True,
            )
            return pref_pool[0]

        if brief.family == "retrieval_recall":
            phrase = _recall_phrase(brief.goal)
            matched = [v for v in pool if title_phrase_match(v, phrase)]
            if not matched and phrase and self.agent.arena.steps_left() > 4:
                self.agent.search(phrase, k=10, in_stock=True)
                self.agent._absorb_shopper()
                pool = [v for v in self.agent.pool.values() if firm_ok(brief, v)] or [
                    v for v in self.agent.pool.values() if v.in_stock
                ]
                matched = [v for v in pool if title_phrase_match(v, phrase)]
            if matched:
                pool = matched
            elif phrase:
                # Do not place a non-phrase SKU for recall tasks.
                return None
            pool = sorted(
                pool,
                key=lambda v: (
                    1 if title_phrase_match(v, phrase) else 0,
                    1 if v.viewed else 0,
                    composite_utility(brief, v),
                ),
                reverse=True,
            )
            return pool[0] if pool else None

        if brief.family == "justification":
            want = brief.brand or brief.preferred_brand
            branded = [v for v in hard if want and brand_hits(v, want)]
            if branded:
                pool = branded
            claimable = [v for v in pool if ProofLedger.claimable(brief, v)]
            if claimable:
                pool = claimable
            else:
                evidenced = [v for v in pool if ProofLedger.visible(brief, v)]
                if evidenced:
                    pool = evidenced
            viewed = [v for v in pool if v.viewed]
            if viewed:
                pool = viewed

        if brief.family == "constraint_satisfaction":
            evidenced = [v for v in hard if ProofLedger.visible(brief, v)]
            if evidenced:
                pool = evidenced
            mentioned = [v for v in pool if SkuExact.option_mentions(brief, v)]
            if mentioned:
                pool = mentioned
            viewed = [v for v in pool if v.viewed]
            if viewed:
                pool = viewed

        if brief.family in FirstCartLadder.EVENT_FAMILIES:
            pool = sorted(
                pool, key=lambda v: FirstCartLadder.ladder_score(brief, v), reverse=True
            )
        else:
            pool = sorted(
                pool,
                key=lambda v: (
                    v.weight.score_n if v.weight else 0.0,
                    composite_utility(brief, v),
                ),
                reverse=True,
            )

        # Fibonacci-batch LLM select (ELON PV_SELECT spirit) for non-event families
        if (
            len(pool) >= 2
            and self.agent.arena.steps_left() > 8
            and brief.family
            not in {
                "ranking",
                "recovery",
                "justification",
                "preference_reasoning",
                "intent_decomposition",
                "constraint_satisfaction",
                "retrieval_recall",
            }
        ):
            picked = self._fibonacci_select(pool)
            if picked is not None:
                return picked
        return pool[0] if pool else None

    def _fibonacci_select(self, candidates: list[Listing]) -> Optional[Listing]:
        windows = fib_windows(3, 2, limit=min(len(candidates), self.SELECT_BATCH * 2))
        cursor = 0
        tries = 0
        for win in windows:
            if cursor >= len(candidates):
                break
            batch = candidates[cursor : cursor + win]
            cursor += win
            tries += 1
            try:
                pick = self._llm_batch(batch)
                if pick is not None:
                    return pick
            except Exception:
                pass
            if tries >= self.MAX_SELECT_TRIES:
                break
        return None

    def _llm_batch(self, batch: list[Listing]) -> Optional[Listing]:
        brief = self.agent.brief
        payload = []
        for i, v in enumerate(batch, start=1):
            payload.append(
                {
                    "id": i,
                    "product_id": v.product_id,
                    "sku": v.sku,
                    "title": v.title,
                    "brand": v.brand,
                    "price": v.price,
                    "in_stock": v.in_stock,
                    "options": v.options,
                    "facts": v.facts,
                    "category_path": v.category_path,
                    "coverage": v.weight.score_n if v.weight else 0.0,
                }
            )
        prompt = _SELECT_PROMPT.format(
            goal=brief.goal,
            family=brief.family,
            budget=brief.budget,
            currency=brief.currency,
            attributes=json.dumps(brief.attributes),
            brand=brief.brand,
            preferred_brand=brief.preferred_brand,
            soft_prefs=brief.soft_prefs,
            products=json.dumps(payload, ensure_ascii=False),
        )
        js = chat_json(prompt, expect="dict")
        best = js.get("best")
        if best is None:
            return None
        try:
            bid = int(best)
        except (TypeError, ValueError):
            m = re.search(r"(\d+)", str(best))
            bid = int(m.group(1)) if m else 0
        if 1 <= bid <= len(batch):
            return batch[bid - 1]
        return None


class CloseEngine:
    """first cart (FirstCartLadder) → market events → recover → place_order."""

    def __init__(self, agent: ElonAgent):
        self.agent = agent

    def seal(self, choice: Listing) -> None:
        brief = self.agent.brief
        arena = self.agent.arena

        if (
            brief.family in {"intent_decomposition", "constraint_satisfaction"}
            and brief.attributes
            and arena.steps_left() > 8
            and not firm_ok(brief, choice)
        ):
            self.agent._spec_requery()
            alts = [v for v in self.agent.pool.values() if firm_ok(brief, v)]
            if brief.family == "intent_decomposition" and brief.preferred_brand:
                branded = [
                    v for v in alts if brand_hits(v, brief.preferred_brand)
                ]
                if branded:
                    alts = branded
            if alts:
                alts.sort(
                    key=lambda v: FirstCartLadder.ladder_score(brief, v), reverse=True
                )
                choice = alts[0]

        self.agent._absorb_shopper(pending=choice)

        prepared = self._prepare_commit(choice)
        if prepared is None:
            alts = [v for v in self.agent.pool.values() if firm_ok(brief, v)]
            if brief.family == "justification":
                want = brief.brand or brief.preferred_brand
                branded = [v for v in alts if want and brand_hits(v, want)]
                if branded:
                    alts = branded
                claimable = [v for v in alts if ProofLedger.claimable(brief, v)]
                if claimable:
                    alts = claimable
            if brief.family == "intent_decomposition" and brief.preferred_brand:
                branded = [
                    v for v in alts if brand_hits(v, brief.preferred_brand)
                ]
                if branded:
                    alts = branded
            alts.sort(
                key=lambda v: FirstCartLadder.ladder_score(brief, v), reverse=True
            )
            if not alts:
                return
            choice = alts[0]
            if not choice.viewed and arena.steps_left() > 3:
                self.agent.view(choice.product_id)
                choice = (
                    FirstCartLadder.climb(
                        brief,
                        [
                            v
                            for v in self.agent.pool.values()
                            if v.product_id == choice.product_id
                        ],
                    )
                    or choice
                )
            if brief.family == "justification" and not ProofLedger.claimable(
                brief, choice
            ):
                if arena.steps_left() > 6:
                    self.agent._spec_requery()
                    alts = [
                        v
                        for v in self.agent.pool.values()
                        if firm_ok(brief, v) and ProofLedger.claimable(brief, v)
                    ]
                    alts.sort(
                        key=lambda v: FirstCartLadder.ladder_score(brief, v),
                        reverse=True,
                    )
                    if not alts:
                        return
                    choice = alts[0]
                else:
                    return
        else:
            choice = prepared

        if brief.family == "ranking":
            if not choice.viewed and arena.steps_left() > 3:
                variants = self.agent.view(choice.product_id)
                if variants:
                    variants.sort(key=ranking_tuple, reverse=True)
                    choice = variants[0]
            if arena.steps_left() > 3:
                choice = self.agent.inspect_stock(choice.product_id, choice.sku)
                self.agent._absorb_shopper(pending=choice)
        elif brief.family not in {
            "intent_decomposition",
            "constraint_satisfaction",
            "justification",
        }:
            if not choice.viewed and arena.steps_left() > 3:
                variants = self.agent.view(choice.product_id)
                self.agent._absorb_shopper(pending=choice)
                if brief.family == "preference_reasoning":
                    if variants:
                        variants.sort(
                            key=lambda v: preference_rank_key(brief, v), reverse=True
                        )
                        choice = variants[0]
                else:
                    picked = FirstCartLadder.climb(brief, variants)
                    if picked is not None:
                        choice = picked
                    elif variants:
                        variants.sort(
                            key=lambda v: composite_utility(brief, v), reverse=True
                        )
                        choice = variants[0]
            if arena.steps_left() > 3:
                choice = self.agent.inspect_stock(choice.product_id, choice.sku)
                self.agent._absorb_shopper(pending=choice)

        if arena.steps_left() < 2:
            return

        add_obs = self.agent.add_to_cart(choice.product_id, choice.sku)
        if arena.finished:
            return

        live = self.agent.pool.get(choice.key) or choice
        live = MarketLedger.apply_add(live, add_obs or {})
        self.agent.pool[live.key] = live

        if arena.steps_left() > 0:
            live = self.agent.inspect_stock(choice.product_id, choice.sku)
            live = MarketLedger.apply_add(live, add_obs or {})
            self.agent.pool[live.key] = live
            self.agent._absorb_shopper(pending=live)
        else:
            self.agent._absorb_shopper(pending=live)

        if not needs_recover(brief, live):
            if brief.family == "ranking":
                replacement = self._recover_ranking(
                    exclude_keys=set(), previous=live
                )
                if replacement is not None and replacement.key != live.key:
                    self.agent.remove_from_cart(choice.product_id, choice.sku)
                    if arena.steps_left() > 1:
                        if not replacement.viewed:
                            self.agent.view(replacement.product_id)
                        self.agent.inspect_stock(
                            replacement.product_id, replacement.sku
                        )
                    self.agent.add_to_cart(
                        replacement.product_id, replacement.sku
                    )
                    if arena.finished:
                        return
                    if arena.steps_left() > 0:
                        replacement = self.agent.inspect_stock(
                            replacement.product_id, replacement.sku
                        )
                    if arena.steps_left() > 0 and buyable(brief, replacement):
                        self.agent.place_order(replacement)
                    return
            if buyable(brief, live):
                self.agent.place_order(live)
            return

        self.agent.remove_from_cart(choice.product_id, choice.sku)
        if brief.family == "ranking":
            replacement = self._recover_ranking(
                exclude_keys={choice.key}, previous=choice
            )
        else:
            replacement = self.recover(exclude_keys={choice.key}, previous=choice)
        if replacement is None:
            if buyable(brief, live):
                self.agent.add_to_cart(live.product_id, live.sku)
                if not arena.finished and buyable(brief, live):
                    self.agent.place_order(live)
            elif arena.steps_left() > 0:
                self.agent._force_done()
            return

        if arena.steps_left() > 1:
            if not replacement.viewed:
                self.agent.view(replacement.product_id)
            self.agent.inspect_stock(replacement.product_id, replacement.sku)
            replacement = self.agent.pool.get(replacement.key, replacement)
        if not buyable(brief, replacement):
            alt = self.recover(
                exclude_keys={choice.key, replacement.key}, previous=replacement
            )
            if alt is None or not buyable(brief, alt):
                if arena.steps_left() > 0:
                    self.agent._force_done()
                return
            replacement = alt
        self.agent.add_to_cart(replacement.product_id, replacement.sku)
        if arena.finished:
            return
        if arena.steps_left() > 0:
            replacement = self.agent.inspect_stock(
                replacement.product_id, replacement.sku
            )
            self.agent.pool[replacement.key] = replacement
        if arena.steps_left() > 0 and buyable(brief, replacement):
            self.agent.place_order(replacement)
        elif arena.steps_left() > 0:
            self.agent._force_done()

    def _prepare_commit(self, choice: Listing) -> Optional[Listing]:
        brief = self.agent.brief
        arena = self.agent.arena
        if brief.family not in FirstCartLadder.EVENT_FAMILIES:
            return choice

        def pick_from_product(pid: str) -> Optional[Listing]:
            variants = self.agent.view(pid) if arena.steps_left() > 2 else []
            self.agent._absorb_shopper()
            best = FirstCartLadder.climb(brief, variants)
            if best is None:
                return None
            if arena.steps_left() > 2:
                best = self.agent.inspect_stock(best.product_id, best.sku)
                self.agent._absorb_shopper(pending=best)
            if not (firm_ok(brief, best) and buyable(brief, best)):
                return None
            if brief.family == "justification" and not ProofLedger.claimable(
                brief, best
            ):
                return None
            return best

        brand = FirstCartLadder.soft_brand(brief)
        if brand and brief.family == "intent_decomposition":
            branded_hits = [v for v in self.agent.pool.values() if brand_hits(v, brand)]
            if not branded_hits:
                self._brand_hunt()

        ranked = FirstCartLadder.shortlist(
            brief, list(self.agent.pool.values()), limit=6
        )
        try_pids = [v.product_id for v in ranked]
        if choice.product_id not in try_pids:
            try_pids.append(choice.product_id)

        picked_ok: list[Listing] = []
        for pid in try_pids:
            if arena.steps_left() <= 5:
                break
            picked = pick_from_product(pid)
            if picked is None:
                continue
            picked_ok.append(picked)
            if (
                FirstCartLadder.ambiguity_penalty(picked) >= -0.5
                and (
                    brief.family != "justification"
                    or ProofLedger.claimable(brief, picked)
                )
                and (not brand or brand_force(picked, brand) >= 2)
            ):
                if len(picked_ok) >= 2 or arena.steps_left() <= 8:
                    break

        if not picked_ok:
            return choice if firm_ok(brief, choice) else None

        picked_ok.sort(
            key=lambda v: FirstCartLadder.ladder_score(brief, v), reverse=True
        )
        best = picked_ok[0]
        if brief.family == "justification" and not ProofLedger.claimable(brief, best):
            return None
        return best

    def _brand_hunt(self) -> None:
        brand = FirstCartLadder.soft_brand(self.agent.brief)
        if not brand or self.agent.arena.steps_left() <= 5:
            return
        for q in self.agent._firm_queries()[:2]:
            if self.agent.arena.steps_left() <= 4:
                break
            self.agent.search(q, k=10, in_stock=True)
        bits = [
            brand,
            self.agent.brief.category or "",
            " ".join(str(v) for v in self.agent.brief.attributes.values()),
        ]
        q = " ".join(b for b in bits if b)
        self.agent.search(q, k=10, in_stock=True)
        if self.agent.arena.steps_left() > 4:
            self.agent.filter_catalog(
                category=self.agent.brief.category,
                brand=brand,
                max_price=self.agent.brief.budget,
                k=10,
                in_stock=True,
            )
        self.agent._absorb_shopper()

    def _recover_ranking(
        self, *, exclude_keys: set[str], previous: Listing
    ) -> Optional[Listing]:
        keys = self.agent.brief.candidate_keys or []
        pool: list[Listing] = []
        for key in keys:
            if key in exclude_keys:
                continue
            v = self.agent.pool.get(key)
            if v is None:
                pid, _, sku = key.partition("::")
                v = Listing(product_id=pid, sku=sku or "default")
                self.agent.pool[key] = v
            if not v.viewed and self.agent.arena.steps_left() > 3:
                self.agent.view(v.product_id)
                v = self.agent.pool.get(key, v)
            if self.agent.arena.steps_left() > 2:
                v = self.agent.inspect_stock(v.product_id, v.sku)
            if v.in_stock and budget_ok(v, self.agent.brief.budget):
                pool.append(v)
        if not pool:
            return None
        pool.sort(key=ranking_tuple, reverse=True)
        return pool[0]

    def recover(
        self, *, exclude_keys: set[str], previous: Listing
    ) -> Optional[Listing]:
        brief = self.agent.brief
        arena = self.agent.arena

        if previous.product_id and arena.steps_left() > 2 and brief.family not in {
            "preference_reasoning",
            "retrieval_recall",
        }:
            variants = self.agent.view(previous.product_id)
            siblings = [
                v
                for v in variants
                if v.key not in exclude_keys
                and firm_ok(brief, v)
                and buyable(brief, v)
            ]
            if siblings:
                siblings.sort(
                    key=lambda v: composite_utility(brief, v), reverse=True
                )
                return siblings[0]
            for v in variants:
                if v.key in exclude_keys or not firm_ok(brief, v):
                    continue
                if arena.steps_left() <= 2:
                    break
                live = self.agent.inspect_stock(v.product_id, v.sku)
                if buyable(brief, live) and firm_ok(brief, live):
                    return live

        if arena.steps_left() > 2:
            if brief.family == "recovery" and brief.preferred_model:
                self.agent.search(brief.preferred_model, k=10, in_stock=True)
                if brief.preferred_color and arena.steps_left() > 2:
                    self.agent.search(
                        f"{brief.preferred_color} {brief.preferred_model}",
                        k=10,
                        in_stock=True,
                    )
            elif brief.family == "preference_reasoning":
                for q in soft_search_queries(brief)[:3]:
                    if arena.steps_left() <= 2:
                        break
                    self.agent.search(q, k=10, in_stock=True)
                self.agent._absorb_shopper()
            elif brief.family == "retrieval_recall":
                phrase = _recall_phrase(brief.goal)
                if phrase:
                    self.agent.search(phrase, k=10, in_stock=True)
                    toks = [t for t in phrase.split() if t]
                    if len(toks) > 1 and arena.steps_left() > 2:
                        self.agent.search(
                            " ".join(reversed(toks)), k=10, in_stock=True
                        )
                self.agent._absorb_shopper()
            elif brief.family in {
                "justification",
                "constraint_satisfaction",
                "intent_decomposition",
            }:
                self.agent._spec_requery()
            else:
                q = " ".join(brief.keywords[:6]) or brief.goal[:60]
                self.agent.search(q, k=10, in_stock=True)
                if arena.steps_left() > 2:
                    self.agent.filter_catalog(
                        category=brief.category,
                        brand=brief.brand,
                        max_price=brief.budget,
                        k=10,
                        in_stock=True,
                    )
                self.agent._absorb_shopper()

        candidates = [
            v
            for v in self.agent.pool.values()
            if v.key not in exclude_keys
            and firm_ok(brief, v)
            and buyable(brief, v)
        ]
        if brief.family == "recovery":
            same_model = []
            if brief.preferred_model:
                same_model = [
                    v
                    for v in candidates
                    if _canon(brief.preferred_model)
                    in _canon(v.blob() + " " + v.options)
                ]
            if same_model and brief.preferred_color:
                other_color = [
                    v
                    for v in same_model
                    if _canon(brief.preferred_color)
                    not in _canon(v.blob() + " " + v.options)
                ]
                pool = other_color or same_model
            else:
                pool = same_model or candidates
        elif brief.family == "preference_reasoning":
            prev_listing = previous.listing_id
            cand = [
                v
                for v in candidates
                if (not prev_listing or v.listing_id != prev_listing)
                and v.product_id != previous.product_id
            ] or [
                v for v in candidates if v.product_id != previous.product_id
            ] or candidates
            pool = sorted(
                cand, key=lambda v: preference_rank_key(brief, v), reverse=True
            )
            return pool[0] if pool else None
        elif brief.family == "retrieval_recall":
            phrase = _recall_phrase(brief.goal)
            matched = [
                v
                for v in candidates
                if title_phrase_match(v, phrase)
                and v.product_id != previous.product_id
            ] or [v for v in candidates if title_phrase_match(v, phrase)]
            if not matched:
                return None
            matched.sort(
                key=lambda v: (
                    1 if v.viewed else 0,
                    composite_utility(brief, v),
                ),
                reverse=True,
            )
            return matched[0]
        elif brief.family == "justification":
            want = brief.brand or brief.preferred_brand
            branded = [v for v in candidates if want and brand_hits(v, want)]
            pool = branded or candidates
        elif brief.family == "intent_decomposition":
            if brief.preferred_brand:
                branded = [
                    v for v in candidates if brand_hits(v, brief.preferred_brand)
                ]
                pool = branded or candidates
            else:
                pool = candidates
        else:
            same = [v for v in candidates if v.product_id == previous.product_id]
            pool = same or candidates

        if not pool:
            if brief.family in {
                "constraint_satisfaction",
                "justification",
                "intent_decomposition",
                "retrieval_recall",
                "preference_reasoning",
            }:
                return None
            pool = [
                v
                for v in self.agent.pool.values()
                if v.key not in exclude_keys and buyable(brief, v)
            ]
        if not pool:
            return None

        if brief.family in FirstCartLadder.EVENT_FAMILIES:
            pool.sort(
                key=lambda v: FirstCartLadder.ladder_score(brief, v), reverse=True
            )
        else:
            pool.sort(key=lambda v: composite_utility(brief, v), reverse=True)
        best = pool[0]
        if not best.viewed and arena.steps_left() > 2:
            variants = self.agent.view(best.product_id)
            variants = [
                v
                for v in variants
                if (firm_ok(brief, v) or buyable(brief, v))
                and v.key not in exclude_keys
            ]
            if variants:
                if brief.family in FirstCartLadder.EVENT_FAMILIES:
                    variants.sort(
                        key=lambda v: FirstCartLadder.ladder_score(brief, v),
                        reverse=True,
                    )
                else:
                    variants.sort(
                        key=lambda v: composite_utility(brief, v), reverse=True
                    )
                best = variants[0]
        if not buyable(brief, best):
            return None
        return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """ORO Bench entrypoint: EnvPack shopping episode."""
    arena = Arena.open(problem_data)
    agent = ElonAgent(arena)
    return agent.run()


__all__ = ["agent_main"]
