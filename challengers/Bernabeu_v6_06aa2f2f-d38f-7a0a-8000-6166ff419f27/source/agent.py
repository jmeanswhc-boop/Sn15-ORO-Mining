"""ORO Bench shopping agent for generated environment sessions."""

from __future__ import annotations

import json
import re
from os import getenv
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
# Official allowlist pairs. Local Chutes uses the *-TEE id; the race funds
# OpenRouter and requires the paired slug. The proxy also rewrites these, but
# SANDBOX_MODEL is often the Chutes id even when INFERENCE_PROVIDER=openrouter.
_MODEL_PAIRS = (
    ("deepseek-ai/DeepSeek-V3.2-TEE", "deepseek/deepseek-v3.2"),
    ("deepseek-ai/DeepSeek-V3.1-TEE", "deepseek/deepseek-chat-v3.1"),
    ("deepseek-ai/DeepSeek-V3-0324-TEE", "deepseek/deepseek-chat-v3-0324"),
    ("deepseek-ai/DeepSeek-R1-0528-TEE", "deepseek/deepseek-r1-0528"),
    ("Qwen/Qwen3-32B-TEE", "qwen/qwen3-32b"),
    ("Qwen/Qwen3.5-397B-A17B-TEE", "qwen/qwen3.5-397b-a17b"),
    ("Qwen/Qwen3.6-27B-TEE", "qwen/qwen3.6-27b"),
    ("google/gemma-4-31B-turbo-TEE", "google/gemma-4-31b-it"),
    ("zai-org/GLM-5.1-TEE", "z-ai/glm-5.1"),
    ("zai-org/GLM-5.2-TEE", "z-ai/glm-5.2"),
    ("moonshotai/Kimi-K2.6-TEE", "moonshotai/kimi-k2.6"),
    ("unsloth/Mistral-Nemo-Instruct-2407-TEE", "mistralai/mistral-small-2603"),
    ("XiaomiMiMo/MiMo-V2-Flash-TEE", "xiaomi/mimo-v2-flash"),
    ("openai/gpt-oss-120b-TEE", "openai/gpt-oss-120b"),
)
_CHUTES_TO_OPENROUTER = {chutes: openrouter for chutes, openrouter in _MODEL_PAIRS}
_OPENROUTER_TO_CHUTES = {openrouter: chutes for chutes, openrouter in _MODEL_PAIRS}
# Race SIGKILLs agent_main at 300s. Leave headroom for the last env call.
# ProxyClient max_retries is attempt count (1 = no extra try).
# Inference: 2 absorbs a fast 429/503; a 40s hang still fits one turn.
# Environment: 1 — a second POST reuses the turn's idempotency key and
# can quarantine the session.
_PROBLEM_BUDGET_S = 270.0
_INFERENCE_TIMEOUT_S = 40
_INFERENCE_ATTEMPTS = 2
_ENVIRONMENT_TIMEOUT_S = 50
_ENVIRONMENT_ATTEMPTS = 3
_HURRY_REMAINING_S = float(_INFERENCE_TIMEOUT_S * _INFERENCE_ATTEMPTS + 10)
# Policy max_steps is 30. The 270s wall clock is the real cap; cutting
# turns at 10 left in-cart SKUs unordered.
_MAX_TURNS = 30
_inference = ProxyClient(
    timeout=_INFERENCE_TIMEOUT_S, max_retries=_INFERENCE_ATTEMPTS
)
_environment = ProxyClient(
    timeout=_ENVIRONMENT_TIMEOUT_S, max_retries=_ENVIRONMENT_ATTEMPTS
)
_VARIANT_RE = re.compile(r"(\d{6,})::([A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)")
_BUDGET_RE = re.compile(
    r"(?:spend up to|go up to|up to|at most|at or under|at or below|under|below|"
    r"no more than)\s+"
    r"[`$]?(\d+(?:\.\d+)?)",
    re.I,
)
_CATEGORY_RE = re.compile(r"in the (.+?) category", re.I)
_PREFER_RE = re.compile(
    r"(?:i prefer|ideally,?\s+i prefer|strongly prefer(?: the)?)\s+([^.,]+)",
    re.I,
)
_BRAND_DEVICE_RE = re.compile(
    r"\b(?:need|want|get me|find me)\s+an?\s+([A-Za-z][A-Za-z0-9.-]+)\s+"
    r"(smartphone|tablet|monitor|headphone|headphones)\b",
    re.I,
)
_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*inch", re.I)
_RAM_RE = re.compile(r"(\d+)\s*gb\s*(?:of\s+)?(?:ram|memory)\b", re.I)
_HZ_RE = re.compile(r"(\d+)\s*(?:-\s*)?(?:hz|hertz)\b", re.I)
_RES_RE = re.compile(
    r"\b(\d{3,4}\s*[x×]\s*\d{3,4}|720p|1080p|1440p|2160p|4k|8k)\b",
    re.I,
)
_COLOR_WORDS = {
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
}
_PREFER_COLOR_MODEL_RE = re.compile(
    r"(?:strongly )?prefer(?: the)?\s+("
    + "|".join(sorted(_COLOR_WORDS, key=len, reverse=True))
    + r")\s+([A-Za-z0-9._-]{4,})",
    re.I,
)
_SEARCH_FILLER = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "for",
    "to",
    "of",
    "in",
    "on",
    "with",
    "from",
    "please",
    "need",
    "want",
    "looking",
    "find",
    "get",
    "buy",
    "me",
    "my",
    "i",
    "that",
    "this",
    "must",
    "be",
    "is",
    "are",
    "it",
    "at",
    "as",
    "if",
    "we",
    "under",
    "below",
    "up",
    "than",
    "more",
    "less",
    "also",
    "still",
    "can",
    "you",
    "your",
    "our",
    "any",
    "all",
    "one",
    "available",
    "stock",
    "category",
    "spend",
    "php",
    "usd",
    "ideally",
    "prefer",
    "preferred",
    "strongly",
    "firm",
    "constraint",
    "constraints",
    "budget",
    "order",
    "shopper",
    "tell",
    "ask",
    "just",
    "really",
    "something",
    "about",
    "like",
    "would",
    "could",
    "should",
    "within",
    "keep",
    "staying",
    "stay",
    "current",
    "price",
    "priced",
    "because",
    "while",
    "after",
    "before",
    "between",
    "same",
    "other",
    "then",
    "first",
    "only",
    "not",
    "no",
    "yes",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "will",
    "there",
    "here",
    "them",
    "they",
    "those",
    "these",
    "into",
    "over",
    "per",
    "via",
    "using",
    "used",
    "use",
    "make",
    "made",
    "sure",
    "let",
    "know",
    "thanks",
    "hello",
    "hi",
}
_PERMISSION_RE = re.compile(
    r"\b(is this acceptable|does that work|should i|do you want|would you like|"
    r"ok to (?:proceed|buy|order)|want me to|shall i)\b",
    re.I,
)
_ACCESSORY_RE = re.compile(
    r"\b(case|cover|charger|cable|screen only|replacement screen|tempered glass|"
    r"drain cleaner|clog remover|screen protector|ear tips|stand)\b",
    re.I,
)
_PHONE_JUNK_RE = re.compile(
    r"\b(holder|clamp|gamepad|joystick|lens|mount|crossbody bag|"
    r"organizer bag|cleaning kit|phone speaker and port)\b",
    re.I,
)
_SMARTPHONE_OS = {
    "android",
    "ios",
    "iphone",
    "blackberry os",
    "windows phone",
    "symbian",
    "harmonyos",
    "hyperos",
    "sailfish",
    "ubuntu touch",
    "tizen",
}
_CATEGORY_ALIAS = {
    "Smartphone": "Smartphones",
    "Headphone": "Headphones",
    "Tablet": "Tablets",
}


def _provider() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw in _DEFAULT_MODELS:
        return raw
    override = (getenv("SANDBOX_MODEL") or "").strip()
    if override.endswith("-TEE") or override in _CHUTES_TO_OPENROUTER:
        return "chutes"
    if override in _OPENROUTER_TO_CHUTES:
        return "openrouter"
    return "openrouter"


def _for_provider(model: str, provider: str) -> str:
    name = (model or "").strip()
    if provider == "openrouter":
        return _CHUTES_TO_OPENROUTER.get(name) or name or _DEFAULT_MODELS["openrouter"]
    return _OPENROUTER_TO_CHUTES.get(name) or name or _DEFAULT_MODELS["chutes"]


def _model() -> str:
    provider = _provider()
    requested = (getenv("SANDBOX_MODEL") or "").strip() or _DEFAULT_MODELS[provider]
    return _for_provider(requested, provider)


def _normalize(text: str) -> str:
    value = str(text).lower()
    value = re.sub(r"(\d+(?:\.\d+)?)([a-z]+)", r"\1 \2", value)
    value = re.sub(r"([a-z]+)(\d+(?:\.\d+)?)", r"\1 \2", value)
    value = re.sub(r"[()\[\]{}+*_:;,\-~/<>=|]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    raw = fn.get("arguments", "{}")
    if raw in (None, ""):
        raw = "{}"
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _function_spec(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        return {}
    fn = spec.get("function") if isinstance(spec.get("function"), dict) else spec
    return fn if isinstance(fn, dict) else {}


def _tool_names(policy: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for spec in policy.get("tools") or []:
        name = _function_spec(spec).get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _tool_spec(policy: dict[str, Any], name: str) -> dict[str, Any]:
    for spec in policy.get("tools") or []:
        fn = _function_spec(spec)
        if fn.get("name") == name:
            return fn
    return {}


def _claim_fields(policy: dict[str, Any]) -> set[str]:
    params = (_tool_spec(policy, "place_test_order").get("parameters") or {}).get(
        "properties"
    ) or {}
    claims = ((params.get("justification") or {}).get("properties") or {}).get("claims") or {}
    item = (claims.get("items") or {}).get("properties") or {}
    field = item.get("field") or {}
    enum = field.get("enum")
    if isinstance(enum, list) and enum:
        return {str(value) for value in enum}
    return {
        "brand",
        "category",
        "in_stock",
        "price",
        "refresh_rate",
        "resolution",
        "screen_size",
    }


def _justification_required(policy: dict[str, Any], query: str) -> bool:
    required = (_tool_spec(policy, "place_test_order").get("parameters") or {}).get(
        "required"
    ) or []
    if "justification" in required:
        return True
    return bool(
        re.search(
            r"trustworthy reason|walk me through why|tell me why|give me a reason",
            query,
            re.I,
        )
    )


def _goal_text(query: str) -> str:
    head, _, _ = query.partition("\n\nTask rules:")
    return head.strip() or query.strip()


def _rules_text(query: str) -> str:
    _, _, rules = query.partition("\n\nTask rules:")
    return rules.strip()


def _budget(text: str) -> float | None:
    match = _BUDGET_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "yes", "y", "in stock", "available", "1"}


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", str(value).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


_RES_ALIASES = {
    "720p": ("720p", "hd", "1280x720", "1280×720"),
    "1080p": ("1080p", "fhd", "fullhd", "full hd", "1920x1080", "1920×1080"),
    "1440p": ("1440p", "qhd", "wqhd", "2560x1440", "2560×1440"),
    "2160p": ("2160p", "4k", "uhd", "3840x2160", "3840×2160"),
    "4k": ("2160p", "4k", "uhd", "3840x2160", "3840×2160"),
    "8k": ("8k", "7680x4320"),
}


def _resolution_matches(wanted: str, blob: str) -> bool:
    if not wanted:
        return True
    compact = _normalize(blob).replace(" ", "").replace("×", "x")
    want = wanted.lower().replace(" ", "").replace("×", "x")
    if want and (want in compact or want.replace("x", "") in compact):
        return True
    aliases = _RES_ALIASES.get(want, ())
    if any(alias.replace(" ", "").replace("×", "x") in compact for alias in aliases):
        return True
    parts = [part for part in re.sub(r"[x×]", " ", want).split() if part]
    tokens = set(_tokens(blob))
    return bool(parts) and all(part in tokens for part in parts)


def _screen_from_text(text: str) -> float | None:
    match = _INCH_RE.search(text or "")
    if match:
        try:
            value = float(match.group(1))
            if 1.0 <= value <= 100.0:
                return value
        except ValueError:
            pass
    quoted = re.search(r'(\d+(?:\.\d+)?)\s*"', str(text or ""))
    if quoted:
        try:
            value = float(quoted.group(1))
            if 1.0 <= value <= 100.0:
                return value
        except ValueError:
            return None
    return None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _contains_term(term: str, blob: str) -> bool:
    if not term:
        return False
    if re.search(rf"\b{re.escape(term.lower())}\b", blob):
        return True
    return term.lower() in _tokens(blob)


def _fact_number(facts: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key, value in facts.items():
        lowered = str(key).lower()
        if any(token in lowered for token in keys):
            number = _number(value)
            if number is not None:
                return number
    return None


def _refresh_value(facts: dict[str, Any]) -> float:
    values: list[float] = []
    for key, value in facts.items():
        lowered = str(key).lower()
        text = str(value)
        if re.search(r"refresh|overclock|hz", lowered) or re.search(r"\d+\s*hz", text, re.I):
            number = _number(value)
            if number is not None:
                values.append(number)
    return max(values) if values else 0.0


def _options_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [_options_text(item) for item in value.values()]
        parts.extend(str(key) for key in value)
        return " ".join(part for part in parts if part)
    if isinstance(value, list):
        return " ".join(_options_text(item) for item in value)
    return str(value)


def _hz_from_text(text: str) -> float:
    match = _HZ_RE.search(text or "")
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def _sku_screen(
    row: dict[str, Any], viewed: dict[str, Any] | None, facts: dict[str, Any] | None
) -> float | None:
    viewed = viewed or {}
    facts = facts or {}
    options = _options_text(row.get("options") or viewed.get("options"))
    size = _screen_from_text(options)
    if size is not None:
        return size
    size = _number(facts.get("screen_size") or facts.get("size"))
    if size is not None:
        return size
    title = str(row.get("title") or viewed.get("title") or "")
    return _screen_from_text(f"{title} {options}")


def _sku_refresh(
    row: dict[str, Any], viewed: dict[str, Any] | None, facts: dict[str, Any] | None
) -> float:
    viewed = viewed or {}
    facts = facts or {}
    options = _options_text(row.get("options") or viewed.get("options"))
    hz = _hz_from_text(options)
    if hz:
        return hz
    title = str(row.get("title") or viewed.get("title") or "")
    return max(_refresh_value(facts), _hz_from_text(title))


def _ipx_score(blob: str) -> int:
    nums = [int(match) for match in re.findall(r"ipx\s*(\d+)", blob or "", re.I)]
    if nums:
        return 10 + max(nums)
    text = str(blob or "").lower()
    if re.search(r"waterproof|water.?resist", text):
        return 6
    if "sweat" in text:
        return 4
    if "resist" in text:
        return 2
    return 0


_BRAND_ALIASES = {
    "apple": ("iphone", "ipad", "ios"),
    "samsung": ("galaxy",),
}
_SCREEN_TOLERANCE = 0.05
_TERM_ALIASES = {
    "resistant": ("resist", "resistance"),
    "resist": ("resistant", "resistance"),
    "resistance": ("resist", "resistant"),
    "waterproof": ("waterresistant", "ipx"),
    "water": ("waterproof", "waterresistant"),
    "ipx": ("waterproof", "waterresistant"),
    "sweat": ("sweatproof", "sweatresistant"),
}


def _brand_matches(wanted: str, blob: str) -> bool:
    if not wanted:
        return True
    lowered = str(blob or "").lower()
    want = wanted.lower()
    if want in lowered:
        return True
    tokens = set(_tokens(lowered))
    if want and all(token in tokens for token in _tokens(want)):
        return True
    aliases = _BRAND_ALIASES.get(want, ())
    return any(alias in lowered or alias in tokens for alias in aliases)


def _unwrap(call_result: dict[str, Any]) -> dict[str, Any]:
    block = call_result.get("observation")
    if not isinstance(block, dict):
        return {}
    inner = block.get("observation")
    if isinstance(inner, dict):
        return inner
    return block


def _is_done(call_result: dict[str, Any]) -> bool:
    if not isinstance(call_result, dict):
        return False
    if call_result.get("done") is True:
        return True
    block = call_result.get("observation")
    if isinstance(block, dict) and block.get("done") is True:
        return True
    inner = block.get("observation") if isinstance(block, dict) else None
    return isinstance(inner, dict) and inner.get("done") is True


def _system_prompt(policy: dict[str, Any], allowed: set[str]) -> str:
    lines = [
        "You are a shopping agent in a live catalog session.",
        "The Task rules in the shopper message are binding. Follow them over any default habit.",
        "Use only the supplied tools. Continue until an observation reports done=true.",
        "Never invent product_id or sku. Never order an accessory when a device is requested.",
        "A title match is not enough: check category_path, raw facts, the selected SKU options, current price, and current stock.",
        "inspect_stock the exact product_id + sku before the first add_to_cart and again after any cart change.",
        "The first add_to_cart is your real intended purchase, not a placeholder.",
        "If a shopper message arrives, treat new requirements as binding, answer in one short sentence, and act in the SAME turn.",
        "Once the chosen SKU is viewed, in-stock, and within budget, add_to_cart then place_test_order immediately.",
        "The sandbox kills this episode at 5 minutes. One search, batched views, inspect_stock, cart, order. Do not keep browsing after a valid SKU exists.",
        "Do not ask permission, do not list options and wait, do not say 'is this acceptable'. Buy.",
        "Never place_test_order unless that exact SKU is in the cart. Never order above the current budget. If price or stock changes after cart, recover to another valid SKU.",
        "Use message only to (1) ask for a hidden spec the shopper offered to tell you, or (2) answer a direct shopper question once. Never confirm a purchase.",
        "If a preferred color+model is unavailable, keep the same model in another color before switching models. Never relax budget, stock, or category.",
        "Search with distinctive values: brand, model, color, and hardware specs. Drop filler. Reformulate if hits are off-category.",
        "If the shopper listed candidate variants as product_id::sku, view/compare those identities first and pick by the stated ranking rule.",
        "If rules require several distinct matching products before ordering, keep searching and viewing until that count is met, then order.",
        "Batch independent views and inspects in one turn when max_calls_per_turn allows.",
        "place_test_order justification, when required or requested, is claims as field/value pairs only. No summary prose.",
        "Claims must come from view/compare facts plus current price/stock, using only allowed claim fields.",
        "Reject listings whose category_path is wrong, whose options are screen-only/replacement parts, or whose device OS is unresolved when a smartphone is required.",
    ]
    catalog: list[str] = []
    for spec in policy.get("tools") or []:
        fn = _function_spec(spec)
        name = fn.get("name")
        if name not in allowed:
            continue
        desc = str(fn.get("description") or "").strip()
        catalog.append(f"- {name}: {desc}" if desc else f"- {name}")
    if catalog:
        lines.append("Available tools:")
        lines.extend(catalog)
    return "\n".join(lines)


class Episode:
    def __init__(self, query: str, allowed: set[str], policy: dict[str, Any]) -> None:
        self.query = query
        self.goal = _goal_text(query)
        self.rules = _rules_text(query)
        self.budget = _budget(query)
        self.allowed = allowed
        self.policy = policy
        self.claim_fields = _claim_fields(policy)
        self.needs_justification = _justification_required(policy, query)
        self.needs_ask = bool(re.search(r"tell you if you ask", query, re.I))
        self.needs_survey = bool(
            re.search(r"at least 5 distinct matching|several distinct products", self.rules, re.I)
        )
        self.needs_clarify = bool(
            re.search(r"clarify the user's priorities", self.rules, re.I)
        )
        self.wants_smartphone = bool(re.search(r"\bsmartphone", self.goal, re.I))
        self.search_count = 0
        self.message_count = 0
        self.compared = False
        self.viewed: dict[str, dict[str, Any]] = {}
        self.candidates: dict[str, dict[str, Any]] = {}
        self.cart: list[dict[str, Any]] = []
        self.last_stock: dict[str, dict[str, Any]] = {}
        self.asked = False
        self.last_shopper = ""
        self.pending_question = False
        self.hurry = False
        self.cart_inspected = False
        self.stock_epoch = 0
        self.inspected_at: dict[str, int] = {}
        self.rejected: set[str] = set()
        self.listed = [
            {"product_id": pid, "sku": sku.rstrip(".")}
            for pid, sku in _VARIANT_RE.findall(query)
        ]
        self.category = self._parse_category()
        self.preferred_color, self.preferred_model = self._parse_color_model()
        self.required_brand = self._parse_required_brand()
        self.preferred_brand = self._parse_brand() or self.required_brand
        self.match_terms = self._parse_match_terms()
        self.screen_inch = self._parse_screen()
        self.ram_gb = self._parse_ram()
        self.refresh_hz = self._parse_refresh()
        self.resolution = self._parse_resolution()
        self.value_terms = self._value_terms()
        self.search_queries = self._search_queries()

    def note_shopper(self, content: str) -> None:
        text = str(content or "").strip()
        if not text:
            return
        self.last_shopper = text
        if "?" in text:
            self.pending_question = True
        self._apply_constraints(text)

    def _apply_constraints(self, text: str) -> None:
        tighter = _budget(text)
        if tighter is not None:
            self.budget = tighter
        screen = _screen_from_text(text)
        if screen is not None:
            self.screen_inch = screen
        ram = _RAM_RE.search(text)
        if ram:
            try:
                self.ram_gb = float(ram.group(1))
            except ValueError:
                pass
        hz = _HZ_RE.search(text)
        if hz:
            try:
                self.refresh_hz = float(hz.group(1))
            except ValueError:
                pass
        resolution = _RES_RE.search(text)
        if resolution:
            self.resolution = re.sub(r"\s+", "", resolution.group(1).lower())
        color, model = None, None
        prefer = _PREFER_COLOR_MODEL_RE.search(text)
        if prefer:
            color = prefer.group(1).lower()
            model = prefer.group(2).strip(" .,'\"")
            if model.lower() in _SEARCH_FILLER or model.lower() in _COLOR_WORDS:
                model = None
        if color:
            self.preferred_color = color
        if model:
            self.preferred_model = model

    def waiting_for_spec(self) -> bool:
        if not self.needs_ask or self.asked:
            return False
        return (
            self.screen_inch is None
            and self.ram_gb is None
            and self.refresh_hz is None
            and not self.resolution
        )

    def should_skip_model(self) -> bool:
        # Race OpenRouter tool-calls 402/timeout; empty model turns became
        # agent_error. Shopping is deterministic from catalog tools.
        return True

    def _parse_screen(self) -> float | None:
        return _screen_from_text(self.goal)

    def _parse_category(self) -> str | None:
        match = _CATEGORY_RE.search(self.goal)
        if match:
            return match.group(1).strip()
        for token in (
            "smartphone",
            "tablet",
            "monitor",
            "headphone",
            "headphones",
            "webcam",
        ):
            if re.search(rf"\b{token}\b", self.goal, re.I):
                raw = "Headphones" if token.startswith("headphone") else token.title()
                return _CATEGORY_ALIAS.get(raw, raw)
        return None

    def _parse_color_model(self) -> tuple[str | None, str | None]:
        match = _PREFER_COLOR_MODEL_RE.search(self.goal)
        if not match:
            return None, None
        color = match.group(1).lower()
        model = match.group(2).strip(" .,'\"")
        if model.lower() in _SEARCH_FILLER or model.lower() in _COLOR_WORDS:
            return color, None
        return color, model

    def _parse_required_brand(self) -> str | None:
        match = re.search(
            r"\b(?:need|want|get me|find me)\s+an?\s+(.+?)\s+"
            r"(smartphone|tablet|monitor|headphones?)\b",
            self.goal,
            re.I,
        )
        if not match:
            return None
        brand_words: list[str] = []
        spec_stop = re.compile(
            r"^(\d|720p|1080p|1440p|2160p|4k|8k|hd|fhd|qhd|uhd|full|gaming)",
            re.I,
        )
        for word in match.group(1).split():
            cleaned = word.strip(" ,.")
            if (
                not cleaned
                or spec_stop.search(cleaned)
                or "inch" in cleaned.lower()
            ):
                break
            lowered = cleaned.lower()
            if lowered in _SEARCH_FILLER or lowered in _COLOR_WORDS:
                if brand_words:
                    break
                continue
            if not brand_words:
                if not cleaned[:1].isupper():
                    break
                brand_words.append(cleaned)
                continue
            if re.match(r"^[A-Za-z][A-Za-z0-9.&'-]*$", cleaned):
                brand_words.append(cleaned)
            else:
                break
        if not brand_words:
            return None
        brand = " ".join(brand_words)
        if brand.lower() in _SEARCH_FILLER:
            return None
        return brand

    def _parse_brand(self) -> str | None:
        if self.preferred_color and self.preferred_model:
            return self.required_brand
        prefer = _PREFER_RE.search(self.goal)
        if prefer:
            text = prefer.group(1).strip()
            text = re.sub(r"\bbut the firm.*$", "", text, flags=re.I).strip()
            words = [word for word in re.split(r"\s+", text) if word]
            if words and words[0].lower() in _COLOR_WORDS:
                pass
            elif text and not re.search(r"\d{6,}", text):
                first = words[0] if words else ""
                if first.lower() not in _SEARCH_FILLER:
                    if " " in text and len(text) > 24:
                        return first
                    return text
        return self.required_brand

    def _parse_ram(self) -> float | None:
        match = _RAM_RE.search(self.goal)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _parse_refresh(self) -> float | None:
        match = _HZ_RE.search(self.goal)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _parse_resolution(self) -> str | None:
        match = _RES_RE.search(self.goal)
        if not match:
            return None
        return re.sub(r"\s+", "", match.group(1).lower())

    def _parse_match_terms(self) -> list[str]:
        quoted = re.findall(r'"([^"]+)"', self.goal)
        if quoted:
            return _tokens(quoted[0])
        match = re.search(r"matches the phrase\s+(.+?)(?:,| at or| and buy|$)", self.goal, re.I)
        if match:
            return _tokens(match.group(1))
        return []

    def _value_terms(self) -> list[str]:
        budget_token = (
            str(int(self.budget))
            if self.budget is not None and self.budget == int(self.budget)
            else None
        )
        terms: list[str] = []
        for token in _tokens(_normalize(self.goal)):
            if token in _SEARCH_FILLER:
                continue
            if budget_token and token == budget_token:
                continue
            if token not in terms:
                terms.append(token)
        return terms

    def _search_queries(self) -> list[str]:
        queries: list[str] = []
        if self.match_terms:
            phrase = " ".join(self.match_terms)
            queries.append(phrase)
            if self.category:
                queries.append(f"{self.category} {phrase}")
            return self._dedupe_queries(queries)
        for row in self.listed[:4]:
            queries.append(row["product_id"])
            queries.append(row["sku"])
        if self.preferred_color and self.preferred_model:
            queries.append(self.preferred_model)
            queries.append(f"{self.preferred_color} {self.preferred_model}")
            queries.append(self.preferred_color)
        spec_bits: list[str] = []
        if self.screen_inch is not None:
            spec_bits.append(str(self.screen_inch).rstrip("0").rstrip("."))
            spec_bits.append("inch")
        if self.ram_gb is not None:
            spec_bits.extend([str(int(self.ram_gb)), "gb"])
        if self.refresh_hz is not None:
            spec_bits.extend([str(int(self.refresh_hz)), "hz"])
        if self.resolution:
            spec_bits.append(self.resolution)
        if self.wants_smartphone:
            queries.append(
                " ".join(
                    part
                    for part in (self.preferred_brand, "smartphone", *spec_bits)
                    if part
                )
            )
        if self.preferred_brand and self.category:
            queries.append(
                " ".join(
                    part
                    for part in (self.preferred_brand, self.category, *spec_bits)
                    if part
                )
            )
        if self.needs_clarify and self.category:
            queries.append(f"{self.category} waterproof IPX sweat")
            queries.append(f"{self.category} IPX")
        if self.category:
            queries.append(" ".join(part for part in (self.category, *spec_bits) if part))
        if self.preferred_brand:
            queries.append(self.preferred_brand)
        values = " ".join(self.value_terms[:10])
        if values:
            queries.append(values)
        if spec_bits:
            queries.append(" ".join(spec_bits))
        return self._dedupe_queries(queries)

    def _dedupe_queries(self, queries: list[str]) -> list[str]:
        seen: list[str] = []
        for query in queries:
            query = re.sub(r"\s+", " ", str(query or "")).strip()
            if query and query not in seen:
                seen.append(query)
        fallback = (self.goal or self.category or "product").strip()[:180]
        return seen or ([fallback] if fallback else ["product"])

    def ingest(self, action_name: str, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            payload = {}
        if action_name == "message":
            self.message_count += 1
            self.asked = True
            self.pending_question = False
        results = payload.get("results")
        if not isinstance(results, list):
            results = []
        if action_name == "search":
            self.search_count += 1
            for row in results:
                self._ingest_hit(row)
        elif action_name == "filter":
            for row in results:
                self._ingest_hit(row)
        elif action_name == "view" and payload.get("product_id"):
            pid = str(payload["product_id"])
            self.viewed[pid] = payload
            self._ingest_product(payload)
        elif action_name == "compare":
            self.compared = True
            compared = payload.get("compared") or payload.get("products") or []
            if isinstance(compared, list):
                for item in compared:
                    if isinstance(item, dict) and item.get("product_id"):
                        pid = str(item["product_id"])
                        self.viewed[pid] = item
                        self._ingest_product(item)
        elif action_name == "inspect_stock" and payload.get("product_id"):
            key = f"{payload.get('product_id')}::{payload.get('sku') or ''}"
            self.last_stock[key] = payload
            self.inspected_at[key] = self.stock_epoch
            if self.cart:
                self.cart_inspected = True
        if action_name == "add_to_cart":
            added = payload.get("added") if isinstance(payload.get("added"), dict) else {}
            pid = added.get("product_id") or payload.get("product_id")
            sku = added.get("sku") or payload.get("sku")
            if pid and sku:
                row = {
                    "product_id": str(pid),
                    "sku": str(sku),
                    "price": added.get("price", payload.get("price")),
                    "in_stock": added.get("in_stock", payload.get("in_stock")),
                }
                key = f"{pid}::{sku}"
                self.last_stock[key] = {**row}
                if not _truthy(row.get("in_stock")):
                    self.rejected.add(key)
                price = _number(row.get("price"))
                if self.budget is not None and price is not None and price > self.budget + 0.009:
                    self.rejected.add(key)
                if not any(
                    str(item.get("product_id")) == str(pid)
                    and str(item.get("sku")) == str(sku)
                    for item in self.cart
                ):
                    self.cart.append(row)
                self.cart_inspected = False
        if action_name == "remove_from_cart":
            self.cart_inspected = False
        if isinstance(payload.get("cart"), list):
            self.cart = [row for row in payload["cart"] if isinstance(row, dict)]
            for row in self.cart:
                pid, sku = row.get("product_id"), row.get("sku")
                if not pid or not sku:
                    continue
                key = f"{pid}::{sku}"
                price = row.get("current_price", row.get("price", row.get("price_at_add")))
                stock = {
                    **(self.last_stock.get(key) or {}),
                    "product_id": pid,
                    "sku": sku,
                    "price": price,
                    "in_stock": row.get("in_stock", (self.last_stock.get(key) or {}).get("in_stock")),
                }
                self.last_stock[key] = stock
        added = payload.get("added")
        if isinstance(added, dict) and added.get("product_id"):
            self.last_stock[f"{added.get('product_id')}::{added.get('sku') or ''}"] = added

    def _store_candidate(self, incoming: dict[str, Any]) -> None:
        pid = str(incoming.get("product_id") or "")
        sku = str(incoming.get("sku") or "")
        if not pid:
            return
        key = f"{pid}::{sku}"
        existing = self.candidates.get(key) or {}
        merged = {**existing, **incoming, "product_id": pid, "sku": sku}
        for field in ("facts", "options", "category_path", "title", "brand"):
            if not merged.get(field) and existing.get(field):
                merged[field] = existing[field]
        self.candidates[key] = merged

    def _ingest_hit(self, row: Any) -> None:
        if not isinstance(row, dict) or not row.get("product_id"):
            return
        self._store_candidate(
            {
                "product_id": str(row["product_id"]),
                "sku": str(row.get("sku") or ""),
                "title": row.get("title"),
                "brand": row.get("brand"),
                "price": row.get("price"),
                "in_stock": row.get("in_stock"),
            }
        )

    def _ingest_product(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        pid = str(payload.get("product_id") or "")
        variants = payload.get("variants") or []
        if not isinstance(variants, list):
            variants = []
        if not variants and payload.get("sku"):
            variants = [payload]
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            sku = str(variant.get("sku") or "")
            if not pid or not sku:
                continue
            self._store_candidate(
                {
                    "product_id": pid,
                    "sku": sku,
                    "title": payload.get("title"),
                    "brand": payload.get("brand"),
                    "price": variant.get("price", payload.get("price")),
                    "in_stock": variant.get("in_stock", payload.get("in_stock")),
                    "options": variant.get("options"),
                    "listing_id": payload.get("listing_id")
                    or payload.get("source_listing_id"),
                    "category_path": payload.get("category_path"),
                    "facts": payload.get("facts")
                    if isinstance(payload.get("facts"), dict)
                    else {},
                }
            )

    def _title_matches(self, title: str) -> bool:
        if not self.match_terms:
            return bool(title)
        lowered = str(title or "").lower()
        if not lowered:
            return False
        phrase = " ".join(self.match_terms)
        if phrase and phrase in lowered:
            return True
        tokens = set(_tokens(lowered))
        return all(term in tokens for term in self.match_terms)

    def title_match_count(self) -> int:
        if not self.match_terms:
            if self.needs_survey:
                return 0
            return len(
                {
                    row["product_id"]
                    for row in self.candidates.values()
                    if row.get("product_id")
                }
            )
        matched: set[str] = set()
        for row in list(self.candidates.values()) + list(self.viewed.values()):
            pid = str(row.get("product_id") or "")
            title = str(row.get("title") or "")
            if pid and self._title_matches(title):
                matched.add(pid)
        return len(matched)

    def fallback_actions(self, max_calls: int) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []

        def add(name: str, args: dict[str, Any]) -> None:
            if name in self.allowed and len(actions) < max_calls:
                actions.append({"name": name, "args": args})

        def search_now() -> None:
            if "search" not in self.allowed or not self.search_queries:
                return
            index = self.search_count % len(self.search_queries)
            add(
                "search",
                {
                    "query": self.search_queries[index],
                    "k": 10,
                    "in_stock": True,
                },
            )
            if "filter" in self.allowed and not self.match_terms:
                filt: dict[str, Any] = {"in_stock": True, "k": 10}
                if self.category:
                    filt["category"] = _CATEGORY_ALIAS.get(self.category, self.category)
                if self.preferred_brand and not re.search(r"\d", self.preferred_brand):
                    filt["brand"] = self.preferred_brand
                if self.budget is not None:
                    filt["max_price"] = self.budget
                add("filter", filt)

        if self.waiting_for_spec() and "message" in self.allowed:
            add(
                "message",
                {
                    "content": (
                        "What is the firm hardware spec I must meet "
                        "(RAM, screen size, resolution, or refresh rate)?"
                    )
                },
            )
            return actions

        if self.needs_clarify and self.message_count == 0 and "message" in self.allowed:
            add(
                "message",
                {
                    "content": (
                        "I'll pick in-stock headphones in budget with the strongest "
                        "sweat/water resistance, then order that one."
                    )
                },
            )

        unseen_listed = [
            row["product_id"]
            for row in self.listed
            if row["product_id"] not in self.viewed
        ]
        if unseen_listed and "view" in self.allowed:
            for pid in list(dict.fromkeys(unseen_listed)):
                add("view", {"product_id": pid})
            if actions:
                return actions

        if self.listed:
            pending_inspect = [
                row
                for row in self.listed
                if row["product_id"] in self.viewed
                and f"{row['product_id']}::{row['sku']}" not in self.last_stock
            ]
            if pending_inspect and "inspect_stock" in self.allowed:
                for row in pending_inspect:
                    add(
                        "inspect_stock",
                        {"product_id": row["product_id"], "sku": row["sku"]},
                    )
                if actions:
                    return actions

        commit = self.commit_actions()
        if commit and commit[0]["name"] in {
            "add_to_cart",
            "place_test_order",
            "inspect_stock",
            "remove_from_cart",
        }:
            for action in commit:
                add(action["name"], action["args"])
            return actions

        if (
            self.needs_survey
            and self.title_match_count() < 5
        ):
            unseen_matches = [
                str(row.get("product_id"))
                for row in self.candidates.values()
                if row.get("product_id")
                and str(row.get("product_id")) not in self.viewed
                and self._title_matches(str(row.get("title") or ""))
            ]
            if unseen_matches and "view" in self.allowed:
                for pid in list(dict.fromkeys(unseen_matches)):
                    add("view", {"product_id": pid})
                if actions:
                    return actions
            search_now()
            return actions

        if not self.listed and "search" in self.allowed and self.search_count == 0:
            search_now()
            return actions

        unseen = [
            str(row.get("product_id"))
            for row in self.candidates.values()
            if row.get("product_id") and str(row.get("product_id")) not in self.viewed
        ]
        if not self.listed and "view" in self.allowed and unseen and not self.hurry:
            limit = 10
            for pid in list(dict.fromkeys(unseen))[:limit]:
                add("view", {"product_id": pid})
            if actions:
                return actions

        usable_viewed = [
            row
            for row in self.candidates.values()
            if self._usable(row) and str(row.get("product_id")) in self.viewed
        ]
        if usable_viewed:
            target = self._rank(usable_viewed)
            for action in self.commit_actions(target):
                add(action["name"], action["args"])
            if actions:
                return actions

        if "view" in self.allowed and unseen:
            limit = 8 if self.hurry else 10
            for pid in list(dict.fromkeys(unseen))[:limit]:
                add("view", {"product_id": pid})
            if actions:
                return actions

        if (
            "compare" in self.allowed
            and not self.compared
            and len(self.viewed) >= 2
            and self.needs_clarify
        ):
            add("compare", {"product_ids": list(self.viewed.keys())[:5]})
            if actions:
                return actions

        target = self._best_target()
        if target is not None:
            for action in self.commit_actions(target):
                add(action["name"], action["args"])
            if actions:
                return actions

        if not self.listed:
            search_now()
        if not actions:
            for action in self._terminal_actions():
                add(action["name"], action["args"])
        return actions

    def _terminal_actions(self) -> list[dict[str, Any]]:
        if self.cart and "place_test_order" in self.allowed:
            row = self.cart[-1]
            pid, sku = str(row.get("product_id") or ""), str(row.get("sku") or "")
            if pid and sku:
                args: dict[str, Any] = {"product_id": pid, "sku": sku}
                if self.needs_justification:
                    claims = self._claims(pid, sku)
                    if claims:
                        args["justification"] = {"claims": claims}
                return [{"name": "place_test_order", "args": args}]
        target = self._best_target()
        if target is None and self.listed:
            listed_rows = [
                self.candidates[f"{row['product_id']}::{row['sku']}"]
                for row in self.listed
                if f"{row['product_id']}::{row['sku']}" in self.candidates
            ]
            target = self._rank(listed_rows) if listed_rows else None
            if target is None:
                row = self.listed[0]
                target = {"product_id": row["product_id"], "sku": row["sku"]}
        if target is None:
            for row in self.candidates.values():
                pid, sku = row.get("product_id"), row.get("sku")
                if pid and sku:
                    target = {"product_id": str(pid), "sku": str(sku)}
                    break
        if target:
            in_cart = any(
                str(row.get("product_id")) == target["product_id"]
                and str(row.get("sku")) == target["sku"]
                for row in self.cart
            )
            if not in_cart and "add_to_cart" in self.allowed:
                return [{"name": "add_to_cart", "args": target}]
            if in_cart and "place_test_order" in self.allowed:
                args = dict(target)
                if self.needs_justification:
                    claims = self._claims(target["product_id"], target["sku"])
                    if claims:
                        args["justification"] = {"claims": claims}
                return [{"name": "place_test_order", "args": args}]
        if "search" in self.allowed and self.search_queries:
            index = self.search_count % len(self.search_queries)
            return [
                {
                    "name": "search",
                    "args": {
                        "query": self.search_queries[index],
                        "k": 10,
                        "in_stock": True,
                    },
                }
            ]
        return []

    def shopper_asked(self) -> bool:
        return self.pending_question

    def pick_message(self, target: dict[str, str]) -> str:
        viewed = self.viewed.get(target["product_id"]) or {}
        title = str(viewed.get("title") or "this option")[:80]
        facts = viewed.get("facts") if isinstance(viewed.get("facts"), dict) else {}
        size = _sku_screen(
            self.candidates.get(f"{target['product_id']}::{target['sku']}") or {},
            viewed,
            facts,
        )
        extra = f" facts show {size} inches." if size else " it meets the firm requirements."
        return f"going with {title}.{extra} ordering it now."

    def commit_actions(self, target: dict[str, str] | None = None) -> list[dict[str, Any]]:
        if self.waiting_for_spec() and not self.hurry:
            return []
        if self.needs_survey and self.title_match_count() < 5 and not self.hurry:
            return []
        bad_cart = [
            row
            for row in self.cart
            if row.get("product_id") and row.get("sku") and not self._cart_row_ok(row)
        ]
        if bad_cart and "remove_from_cart" in self.allowed:
            row = bad_cart[0]
            return [
                {
                    "name": "remove_from_cart",
                    "args": {
                        "product_id": str(row.get("product_id")),
                        "sku": str(row.get("sku")),
                    },
                }
            ]
        if self.listed:
            unseen = [
                row["product_id"]
                for row in self.listed
                if row["product_id"] not in self.viewed
            ]
            if unseen and "view" in self.allowed:
                return [
                    {"name": "view", "args": {"product_id": pid}}
                    for pid in list(dict.fromkeys(unseen))
                ]
            pending_inspect = [
                row
                for row in self.listed
                if row["product_id"] in self.viewed
                and f"{row['product_id']}::{row['sku']}" not in self.last_stock
            ]
            if pending_inspect and "inspect_stock" in self.allowed:
                return [
                    {
                        "name": "inspect_stock",
                        "args": {
                            "product_id": row["product_id"],
                            "sku": row["sku"],
                        },
                    }
                    for row in pending_inspect
                ]
        target = target or self._best_target()
        if target is None:
            return []
        pid, sku = target["product_id"], target["sku"]
        key = f"{pid}::{sku}"
        in_cart = any(
            str(row.get("product_id")) == str(pid) and str(row.get("sku")) == str(sku)
            for row in self.cart
        )
        extras = [
            row
            for row in self.cart
            if not (
                str(row.get("product_id")) == str(pid) and str(row.get("sku")) == str(sku)
            )
        ]
        if extras and "remove_from_cart" in self.allowed:
            row = extras[0]
            return [
                {
                    "name": "remove_from_cart",
                    "args": {
                        "product_id": str(row.get("product_id")),
                        "sku": str(row.get("sku")),
                    },
                }
            ]
        stock = self.last_stock.get(key)
        if pid not in self.viewed and "view" in self.allowed:
            return [{"name": "view", "args": {"product_id": pid}}]
        if self.listed:
            pending_inspect = [
                row
                for row in self.listed
                if row["product_id"] in self.viewed
                and f"{row['product_id']}::{row['sku']}" not in self.last_stock
            ]
            if pending_inspect and "inspect_stock" in self.allowed:
                return [
                    {
                        "name": "inspect_stock",
                        "args": {
                            "product_id": row["product_id"],
                            "sku": row["sku"],
                        },
                    }
                    for row in pending_inspect
                ]
        if "inspect_stock" in self.allowed and stock is None:
            return [{"name": "inspect_stock", "args": {"product_id": pid, "sku": sku}}]
        first_listed = bool(self.listed and not self.cart)
        if self._over_budget(pid, sku) and not first_listed:
            if in_cart and "remove_from_cart" in self.allowed:
                return [
                    {
                        "name": "remove_from_cart",
                        "args": {"product_id": pid, "sku": sku},
                    }
                ]
            return []
        if not in_cart and "add_to_cart" in self.allowed:
            return [{"name": "add_to_cart", "args": {"product_id": pid, "sku": sku}}]
        if in_cart and "inspect_stock" in self.allowed and not self.cart_inspected:
            return [{"name": "inspect_stock", "args": {"product_id": pid, "sku": sku}}]
        if in_cart and "place_test_order" in self.allowed:
            if self._over_budget(pid, sku) and not first_listed:
                if "remove_from_cart" in self.allowed:
                    return [
                        {
                            "name": "remove_from_cart",
                            "args": {"product_id": pid, "sku": sku},
                        }
                    ]
                return []
            args: dict[str, Any] = {"product_id": pid, "sku": sku}
            if self.needs_justification:
                claims = self._claims(pid, sku)
                if claims:
                    args["justification"] = {"claims": claims}
            return [{"name": "place_test_order", "args": args}]
        return []

    def _over_budget(self, pid: str, sku: str) -> bool:
        stock = self.last_stock.get(f"{pid}::{sku}") or {}
        candidate = self.candidates.get(f"{pid}::{sku}") or {}
        price = _number(stock.get("price", candidate.get("price")))
        return (
            self.budget is not None
            and price is not None
            and price > self.budget + 0.009
        )

    def _cart_row_ok(self, row: dict[str, Any]) -> bool:
        pid, sku = row.get("product_id"), row.get("sku")
        if not pid or not sku:
            return False
        candidate = dict(self.candidates.get(f"{pid}::{sku}") or {})
        candidate["product_id"] = str(pid)
        candidate["sku"] = str(sku)
        price = row.get("current_price", row.get("price", candidate.get("price")))
        if price is not None:
            candidate["price"] = price
        if "in_stock" in row:
            candidate["in_stock"] = row.get("in_stock")
        return self._usable(candidate)

    def _best_target(self) -> dict[str, str] | None:
        if self.listed:
            rows = [
                self.candidates[f"{row['product_id']}::{row['sku']}"]
                for row in self.listed
                if f"{row['product_id']}::{row['sku']}" in self.candidates
            ]
            if self.cart:
                usable = [row for row in rows if self._usable(row)]
                picked = self._rank(usable) or self._rank(rows)
            else:
                picked = self._rank(rows)
            if picked:
                return picked
        if self.cart:
            for row in reversed(self.cart):
                pid, sku = row.get("product_id"), row.get("sku")
                if pid and sku and self._cart_row_ok(row):
                    return {"product_id": str(pid), "sku": str(sku)}
        usable = [row for row in self.candidates.values() if self._usable(row)]
        picked = self._rank(usable)
        if picked:
            return picked
        return None

    def _usable(self, row: dict[str, Any]) -> bool:
        pid, sku = row.get("product_id"), row.get("sku")
        if not pid or not sku:
            return False
        if f"{pid}::{sku}" in self.rejected:
            return False
        viewed = self.viewed.get(str(pid)) or {}
        if not isinstance(viewed, dict):
            viewed = {}
        title = str(row.get("title") or viewed.get("title") or "")
        options = _options_text(row.get("options") or viewed.get("options"))
        blob = f"{title} {options}"
        if _ACCESSORY_RE.search(blob):
            return False
        if self.wants_smartphone and _PHONE_JUNK_RE.search(blob):
            return False
        if row.get("in_stock") is False:
            return False
        stock = self.last_stock.get(f"{pid}::{sku}")
        if stock and stock.get("in_stock") is False:
            return False
        price = _number((stock or {}).get("price", row.get("price")))
        if self.budget is not None and price is not None and price > self.budget + 0.009:
            return False
        if self.match_terms:
            if not self._title_matches(title):
                return False
        path = viewed.get("category_path") or row.get("category_path") or []
        if self.category and path:
            joined = " ".join(str(part) for part in path).lower()
            want = self.category.lower().rstrip("s")
            if want not in joined:
                return False
        if self.required_brand and (viewed or title or row.get("brand")):
            brand_blob = " ".join(
                str(part or "")
                for part in (row.get("brand"), viewed.get("brand"), title, options)
            )
            if not _brand_matches(self.required_brand, brand_blob):
                return False
        facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
        viewed_facts = viewed.get("facts") if isinstance(viewed.get("facts"), dict) else {}
        merged = {**viewed_facts, **facts}
        fact_blob = f"{title} {options} " + " ".join(str(value) for value in merged.values())
        if self.screen_inch is not None and viewed:
            size = _sku_screen(row, viewed, merged)
            if size is None or abs(size - self.screen_inch) > _SCREEN_TOLERANCE:
                return False
        if self.ram_gb is not None and viewed:
            ram = _fact_number(merged, ("ram", "memory"))
            if ram is not None and ram + 0.01 < self.ram_gb:
                return False
        if self.refresh_hz is not None and viewed:
            refresh = _sku_refresh(row, viewed, merged)
            if refresh and refresh + 0.01 < self.refresh_hz:
                return False
        if self.resolution and viewed:
            if not _resolution_matches(self.resolution, fact_blob):
                return False
        if self.wants_smartphone and viewed:
            os_name = str(
                merged.get("device_operating_system") or merged.get("operating_system") or ""
            ).lower()
            type_blob = " ".join(
                str(merged.get(key) or "")
                for key in ("phone_type", "device_type", "type")
            ).lower()
            title_ok = "smartphone" in title.lower()
            os_ok = any(token in os_name for token in _SMARTPHONE_OS)
            type_ok = "smartphone" in type_blob
            if os_name in {"others", "other"}:
                os_ok = False
            if not (title_ok or os_ok or type_ok):
                return False
            if re.search(r"\b(tablet|feature phone|basic phone|mystery box)\b", blob, re.I):
                return False
        return True

    def _text_blob(self, row: dict[str, Any], viewed: dict[str, Any], facts: dict[str, Any]) -> str:
        parts = [
            row.get("title"),
            viewed.get("title"),
            row.get("brand"),
            viewed.get("brand"),
            _options_text(row.get("options")),
            _options_text(viewed.get("options")),
            " ".join(str(value) for value in facts.values()),
            " ".join(
                str(part)
                for part in (viewed.get("category_path") or row.get("category_path") or [])
            ),
        ]
        return _normalize(" ".join(str(part) for part in parts if part))

    def _keyword_hits(self, blob: str) -> int:
        tokens = set(_tokens(blob))
        hits = 0
        for term in self.value_terms:
            if term in tokens or any(
                alias in tokens for alias in _TERM_ALIASES.get(term, ())
            ):
                hits += 1
        return hits

    def _rank(self, rows: list[dict[str, Any]]) -> dict[str, str] | None:
        scored: list[tuple[Any, dict[str, str]]] = []
        for row in rows:
            pid, sku = str(row.get("product_id") or ""), str(row.get("sku") or "")
            if not pid or not sku:
                continue
            facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
            viewed = self.viewed.get(pid) or {}
            viewed_facts = viewed.get("facts") if isinstance(viewed.get("facts"), dict) else {}
            merged = {**viewed_facts, **facts}
            brand = str(row.get("brand") or viewed.get("brand") or "")
            blob = self._text_blob(row, viewed, merged)
            options = _options_text(row.get("options") or viewed.get("options"))
            refresh = _sku_refresh(row, viewed, merged)
            screen = _sku_screen(row, viewed, merged) or 0.0
            spec_hit = 0
            if self.screen_inch is not None and screen and abs(screen - self.screen_inch) <= _SCREEN_TOLERANCE:
                spec_hit += 1
            if self.ram_gb is not None:
                ram = _fact_number(merged, ("ram", "memory"))
                if ram is not None and ram + 0.01 >= self.ram_gb:
                    spec_hit += 1
            if self.refresh_hz is not None and refresh + 0.01 >= self.refresh_hz:
                spec_hit += 1
            if self.resolution and (
                _resolution_matches(self.resolution, blob)
                or _resolution_matches(self.resolution, options)
            ):
                spec_hit += 1
            price = _number(
                (self.last_stock.get(f"{pid}::{sku}") or {}).get("price", row.get("price"))
            )
            price_key = price if price is not None else 1e12
            brand_hit = 0
            if self.preferred_brand and _brand_matches(
                self.preferred_brand,
                f"{brand} {blob}",
            ):
                brand_hit = 1
            model_hit = 0
            if self.preferred_model and _contains_term(self.preferred_model, blob):
                model_hit = 1
            color_hit = 0
            if self.preferred_color and _contains_term(self.preferred_color, blob):
                color_hit = 1
            resist = _ipx_score(blob)
            kw_hits = self._keyword_hits(blob)
            option_hits = self._keyword_hits(_normalize(options))
            pick = {"product_id": pid, "sku": sku}
            if self.listed:
                scored.append((-refresh, -screen, price_key, pick))
            elif self.needs_clarify:
                scored.append((-resist, -brand_hit, -option_hits, -kw_hits, price_key, pick))
            elif self.preferred_model or self.preferred_color:
                scored.append(
                    (-model_hit, -color_hit, -brand_hit, -spec_hit, price_key, pick)
                )
            else:
                scored.append(
                    (-brand_hit, -spec_hit, -option_hits, -kw_hits, price_key, pick)
                )
        if not scored:
            return None
        scored.sort(key=lambda item: item[:-1])
        return scored[0][-1]

    def _claims(self, product_id: str, sku: str) -> list[dict[str, str]]:
        viewed = self.viewed.get(product_id) or {}
        facts = viewed.get("facts") if isinstance(viewed.get("facts"), dict) else {}
        stock = self.last_stock.get(f"{product_id}::{sku}") or {}
        candidate = self.candidates.get(f"{product_id}::{sku}") or {}
        merged = {
            **facts,
            **(candidate.get("facts") if isinstance(candidate.get("facts"), dict) else {}),
        }
        claims: list[dict[str, str]] = []

        def add(field: str, value: Any) -> None:
            if field not in self.claim_fields:
                return
            if value is None or value == "":
                return
            text = str(value).strip()
            if not text:
                return
            if any(row["field"] == field for row in claims):
                return
            claims.append({"field": field, "value": text})

        # Required specs, then live price/stock, then one visible soft claim
        # (brand). Extra fields from category_path fail faithfulness.
        if self.resolution:
            add(
                "resolution",
                merged.get("resolution") or merged.get("display"),
            )
        if self.refresh_hz is not None:
            refresh = _sku_refresh(candidate, viewed, merged)
            add(
                "refresh_rate",
                merged.get("refresh_rate")
                or merged.get("max_refresh_rate")
                or (f"{int(refresh)} Hz" if refresh else None),
            )
        if self.screen_inch is not None:
            sku_size = _sku_screen(candidate, viewed, merged)
            fact_raw = merged.get("screen_size") or merged.get("size")
            if (
                fact_raw
                and sku_size is not None
                and abs((_number(fact_raw) or -1) - sku_size) <= _SCREEN_TOLERANCE
            ):
                add("screen_size", fact_raw)
            elif sku_size is not None:
                add("screen_size", f"{sku_size:g} inches")
            else:
                add("screen_size", fact_raw)
        price = stock.get("price", candidate.get("price"))
        if price is not None:
            parsed = _number(price)
            add("price", f"{parsed:.2f}" if parsed is not None else str(price))
        if "in_stock" in stock or "in_stock" in candidate:
            add(
                "in_stock",
                "true"
                if _truthy(stock.get("in_stock", candidate.get("in_stock")))
                else "false",
            )
        if self.needs_justification:
            add("brand", viewed.get("brand") or candidate.get("brand"))
            if self.preferred_brand:
                add("brand", self.preferred_brand)
        if self.needs_justification and not claims:
            return []
        return claims


def _filter_actions(
    actions: list[dict[str, Any]],
    allowed: set[str],
    max_calls: int,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for action in actions:
        name = str(action.get("name") or "")
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        if name not in allowed:
            continue
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        kept.append({"name": name, "args": args})
        if len(kept) >= max_calls:
            break
    return kept


def _model_actions(
    tool_calls: list[dict[str, Any]], allowed: set[str], max_calls: int
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for call in tool_calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(fn.get("name") or "")
        if name not in allowed:
            continue
        args = _arguments(call)
        actions.append({"name": name, "args": args})
        if len(actions) >= max_calls:
            break
    return actions


def _rewrite_message(content: str, episode: Episode) -> str | None:
    text = str(content or "").strip()
    if not text:
        return None
    if _PERMISSION_RE.search(text) or text.endswith("?"):
        target = episode._best_target()
        if target:
            return episode.pick_message(target)
        return None
    return text


def _augment(
    actions: list[dict[str, Any]],
    episode: Episode,
    allowed: set[str],
    max_calls: int,
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("name") != "message":
            cleaned.append(action)
            continue
        if episode.waiting_for_spec():
            cleaned.append(action)
            continue
        if not episode.pending_question and not (
            episode.needs_clarify and episode.message_count == 0
        ):
            continue
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        rewritten = _rewrite_message(str(args.get("content") or ""), episode)
        if rewritten:
            cleaned.append({"name": "message", "args": {"content": rewritten}})

    names = {action["name"] for action in cleaned}
    extra: list[dict[str, Any]] = []
    if episode.waiting_for_spec() and not episode.hurry and "message" not in names:
        extra.append(
            {
                "name": "message",
                "args": {
                    "content": (
                        "What is the firm hardware spec I must meet "
                        "(RAM, screen size, resolution, or refresh rate)?"
                    )
                },
            }
        )
    target = episode._best_target()
    if (
        episode.pending_question
        and "message" not in names
        and not episode.waiting_for_spec()
        and target is not None
        and "message" in allowed
    ):
        extra.append({"name": "message", "args": {"content": episode.pick_message(target)}})
    if (
        episode.needs_clarify
        and episode.message_count == 0
        and "message" not in names
        and "message" in allowed
    ):
        extra.append(
            {
                "name": "message",
                "args": {
                    "content": (
                        "I'll pick in-stock headphones in budget with the strongest "
                        "sweat/water resistance, then order that one."
                    )
                },
            }
        )

    commit = episode.commit_actions(target)
    commit_names = {action["name"] for action in commit}
    if commit:
        kept: list[dict[str, Any]] = []
        for action in extra + cleaned:
            name = action.get("name")
            args = action.get("args") if isinstance(action.get("args"), dict) else {}
            if name == "message":
                kept.append(action)
            elif name in {"view", "inspect_stock"} and target:
                if str(args.get("product_id")) == str(target["product_id"]):
                    kept.append(action)
        merged = _filter_actions(kept + commit, allowed, max_calls)
        return merged or commit

    unseen_listed = [
        row["product_id"]
        for row in episode.listed
        if row["product_id"] not in episode.viewed
    ]
    if unseen_listed and "view" in allowed:
        viewed_now = {
            str((action.get("args") or {}).get("product_id"))
            for action in cleaned
            if action.get("name") == "view"
        }
        for pid in list(dict.fromkeys(unseen_listed)):
            if pid not in viewed_now:
                extra.append({"name": "view", "args": {"product_id": pid}})
        kept = [action for action in cleaned if action["name"] in {"view", "compare"}]
        merged = _filter_actions(extra + kept, allowed, max_calls)
        return merged or extra

    kept = cleaned
    if episode.listed:
        kept = [
            action
            for action in cleaned
            if action["name"] not in {"search", "filter", "place_test_order"}
        ]
    if episode.hurry and commit:
        kept = [
            action
            for action in kept
            if action["name"] not in {"search", "filter", "compare"}
        ]
    merged = _filter_actions(extra + kept + commit, allowed, max_calls)
    return merged or kept or commit


def _as_tool_calls(
    actions: list[dict[str, Any]], problem_id: str, turn: int
) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{problem_id}-act-{turn}-{index}",
            "type": "function",
            "function": {
                "name": action["name"],
                "arguments": json.dumps(action["args"]),
            },
        }
        for index, action in enumerate(actions, start=1)
    ]


def _env_call(envelope: dict[str, Any], remaining: float) -> dict[str, Any] | None:
    timeout = max(8, min(_ENVIRONMENT_TIMEOUT_S, int(remaining) - 2))
    client = (
        _environment
        if timeout >= _ENVIRONMENT_TIMEOUT_S
        else ProxyClient(timeout=timeout, max_retries=_ENVIRONMENT_ATTEMPTS)
    )
    try:
        result = client.post("/environment/call", json_data=envelope)
    except Exception:
        return None
    return result if isinstance(result, dict) else None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Let the model choose actions from the environment's dynamic tool list."""

    environment = problem_data.get("environment") if isinstance(problem_data, dict) else {}
    if not isinstance(environment, dict):
        return []
    binding = environment.get("binding") if isinstance(environment.get("binding"), dict) else {}
    policy = environment.get("policy_view") if isinstance(environment.get("policy_view"), dict) else {}
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    try:
        max_calls = max(1, int(policy.get("max_calls_per_turn") or 1))
    except (TypeError, ValueError):
        max_calls = 1
    try:
        max_steps = max(1, int(policy.get("max_steps") or 1))
    except (TypeError, ValueError):
        max_steps = 1
    allowed = _tool_names(policy)
    if not allowed:
        return []
    episode = Episode(str(policy.get("query") or ""), allowed, policy)
    deadline = monotonic() + _PROBLEM_BUDGET_S

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(policy, allowed)},
        {"role": "user", "content": str(policy.get("query") or "")},
    ]
    dialogue: list[dict[str, Any]] = []

    for turn in range(1, min(max_steps, _MAX_TURNS) + 1):
        remaining = deadline - monotonic()
        if remaining <= 8:
            break
        episode.hurry = remaining < _HURRY_REMAINING_S
        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        skip_model = True
        if not skip_model:
            try:
                inference = _inference.post(
                    "/inference/chat/completions",
                    json_data={
                        "model": _model(),
                        "messages": messages,
                        "tools": policy.get("tools") or [],
                        "tool_choice": "required",
                        "temperature": 0,
                    },
                )
            except Exception:
                inference = None
            if isinstance(inference, dict):
                choices = inference.get("choices")
                first = choices[0] if isinstance(choices, list) and choices else {}
                if not isinstance(first, dict):
                    first = {}
                assistant = (
                    first.get("message") if isinstance(first.get("message"), dict) else {}
                )
                assistant_content = str(assistant.get("content") or "")
                raw_calls = assistant.get("tool_calls") or []
                tool_calls = raw_calls[:max_calls] if isinstance(raw_calls, list) else []

        actions = _filter_actions(
            _augment(
                _model_actions(tool_calls, allowed, max_calls),
                episode,
                allowed,
                max_calls,
            ),
            allowed,
            max_calls,
        )
        if not actions:
            actions = _filter_actions(episode.fallback_actions(max_calls), allowed, max_calls)
        if not actions:
            actions = _filter_actions(episode._terminal_actions(), allowed, max_calls)
        if actions:
            tool_calls = _as_tool_calls(actions, problem_id, turn)
        if not tool_calls or not actions:
            break

        remaining = deadline - monotonic()
        if remaining <= 8:
            break

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        group_id = f"{problem_id}-t{turn}-{len(dialogue)}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {
                        "name": str((call.get("function") or {}).get("name") or ""),
                        "args": _arguments(call),
                    },
                }
                for index, call in enumerate(tool_calls, start=1)
            ],
        }
        sleep(0.12)
        result = _env_call(envelope, remaining)
        if result is None:
            dialogue.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                    "error": "environment call failed",
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

        call_rows = result.get("calls") if isinstance(result.get("calls"), list) else []
        for tool_call, call_result in zip(tool_calls, call_rows):
            observation = _unwrap(call_result) if isinstance(call_result, dict) else {}
            if not isinstance(observation, dict):
                observation = {}
            fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
            try:
                episode.ingest(str(fn.get("name") or ""), observation)
            except Exception:
                pass
            payload = (
                call_result.get("observation")
                if isinstance(call_result, dict)
                else observation
            )
            try:
                content = json.dumps(payload, default=str)
            except (TypeError, ValueError):
                content = "{}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": content,
                }
            )
        user_message = result.get("user_message")
        if isinstance(user_message, dict):
            content = user_message.get("content")
            if content:
                episode.note_shopper(str(content))
                episode.search_queries = episode._search_queries()
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"{content}\n\n"
                            "Treat new budget, spec, and stock as binding. "
                            "If the cart SKU is now over budget or out of stock, "
                            "remove it and add_to_cart / place_test_order a valid "
                            "replacement this turn. Do not ask permission."
                        ),
                    }
                )
        if any(
            _is_done(call_result)
            for call_result in call_rows
            if isinstance(call_result, dict)
        ):
            break

    return dialogue


__all__ = ["agent_main"]
