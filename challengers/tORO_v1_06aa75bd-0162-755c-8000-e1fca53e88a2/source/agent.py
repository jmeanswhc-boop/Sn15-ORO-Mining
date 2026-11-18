"""ORO-Bench shopping agent.

Design: a language model proposes one tool call per turn; a deterministic
controller owns every decision the runtime actually grades. The controller

  * reads product evidence the way the environment does (labelled raw facts
    first, then the selected variant's option text, then the title),
  * only ever commits a fully verified candidate to the cart, because the
    first cart addition is the public commitment the scorer checks,
  * re-inspects that item straight after the addition, since the market moves
    the moment something is carted,
  * recovers to a different, still-valid product when the carted one dies, and
  * orders as soon as a verified item is in the cart.

The model is left in charge of exploration (search wording, which products to
look at, how to answer the shopper); the controller takes over the endgame.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_MODELS = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_PROBLEM_BUDGET_S = 270.0
_OBS_CHARS = 7000
_MAX_REPROMPTS = 2
_EPS = 1e-9

_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _NUM
    + r"\s*(PHP|php|₱|pesos?)?",
    re.I,
)
_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTED_RE = re.compile(r"\"([^\"]{2,80})\"")
_PHRASE_RE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_SURFACE_RE = re.compile(r"surface at least (\d+) distinct", re.I)
_PREF_ITEM_RE = re.compile(r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
_PREF_BRAND_RE = re.compile(
    r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)",
    re.I,
)
_CAT_PHRASE_RE = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
# wording that describes a specification rather than a maker, dropped before a brand is read
_SPEC_WORD_RE = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b",
    re.I,
)
_REASON_RE = re.compile(
    r"\b(?:why|reason|explain|justif)", re.I
)
_CLAIM_FIELDS = ("resolution", "refresh_rate", "screen_size", "ram", "price", "in_stock", "brand")
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)

# ---------------------------------------------------------------- attributes
# The environment resolves a gated specification from public evidence only: the
# labelled raw facts a product page exposes, the selected variant's option text
# and the listing title. A number counts only when its unit or concept word sits
# next to it, and evidence that disagrees with itself resolves to nothing. These
# helpers mirror that reading so the controller can predict the verdict.
_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_PAIR_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_PAIR_HINT_RE = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
_MEM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_QUARTER_RE = re.compile(r"\bqHD\b")

# gated concept -> the display label its evidence arrives under in product facts
_FACT_LABEL = {"screen_size": "size", "resolution": "display", "refresh_rate": "refresh_rate", "ram": "storage"}


def _inch_vals(text: str) -> set:
    return {"%gin" % float(m.group(1)) for m in _INCH_RE.finditer(text or "")}


def _hz_vals(text: str) -> set:
    return {"%ghz" % float(m.group(1)) for m in _HZ_RE.finditer(text or "")}


def _gb_vals(text: str) -> set:
    return {
        "%ggb" % (float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1))
        for m in _GB_RE.finditer(text or "")
    }


def _pixel_class(long_side: int, short_side: int) -> str | None:
    if long_side >= 3840 and short_side >= 2160:
        return "4k"
    if long_side >= 2560 and short_side >= 1440:
        return "qhd"
    if long_side >= 1920 and short_side >= 1080:
        return "fhd"
    if long_side >= 1280 and short_side >= 720:
        return "hd"
    return None


def _res_vals(text: str) -> set:
    """Display classes asserted by a text: pixel pairs and the usual shorthands."""
    text = text or ""
    if _QUARTER_RE.search(text):
        return set()  # quarter HD is a different, much smaller panel
    low = text.lower()
    out = set()
    for m in _WXH_RE.finditer(low):
        pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        cls = _pixel_class(pair[0], pair[1])
        if cls:
            out.add(cls)
    if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
        out.add("4k")
    if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
        out.add("qhd")
    if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
        out.add("fhd")
    if re.search(r"\b(?:hd|720p?)\b", re.sub(r"\b(?:full|ultra) hd\b", "", low)):
        out.add("hd")
    return out


_VALUE_SETS = {"screen_size": _inch_vals, "refresh_rate": _hz_vals, "resolution": _res_vals, "ram": _gb_vals}


def _adjacent_vals(field: str, text: str) -> set:
    """Values a free-form string asserts, requiring the number's own concept word.

    Titles and variant option strings mix several capacities, so a memory figure
    only counts when the word for it follows the number; storage totals and a
    graphics chip's own memory must never be read as system memory.
    """
    text = text or ""
    if field != "ram":
        return _VALUE_SETS[field](text)
    paired = {"%ggb" % float(a) for a, _b in _PAIR_RE.findall(text)}
    return paired or {"%ggb" % float(n) for n in _MEM_RE.findall(text)}


def _mentions(field: str, text: str) -> bool:
    """Whether an option string talks about the concept at all (even ambiguously)."""
    text = text or ""
    if field == "ram":
        return bool(_PAIR_HINT_RE.search(text) or _MEM_RE.search(text))
    if field == "screen_size":
        return bool(_INCH_RE.search(text))
    if field == "refresh_rate":
        return bool(_HZ_RE.search(text))
    return bool(_WXH_RE.search(text) or re.search(r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", text, re.I))


def _one(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def _wanted_value(field: str | None, text: str) -> str | None:
    """The single value a request states for a concept, or nothing when unclear."""
    if not field:
        return None
    return _one(_VALUE_SETS[field](text or ""))


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _same_word(a: str, b: str) -> bool:
    """Loose noun equality: ignores case, spacing and a trailing plural."""
    x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
    y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
    return bool(x) and x == y


# ------------------------------------------------------------ device identity
# Some categories are graded on what the listing IS, not what it is named after:
# a case, a spare panel or a keypad handset are not the device a shopper asked
# for. The runtime decides that from the same public evidence the agent sees.
_OS_RE = re.compile(
    r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian|harmony\s*os|"
    r"hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
    re.I,
)
_IS_PHONE_RE = re.compile(r"\bsmart\s*phones?\b", re.I)
_NOT_PHONE_RE = re.compile(
    r"\btablets?\b|\b(?:feature|basic)\s*phones?\b|\b(?:mystery|blind|lucky)\s*box(?:es)?\b|"
    r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
    r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b",
    re.I,
)
_PART_ONLY_RE = re.compile(
    r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
    r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
    r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b",
    re.I,
)
_NEGATED_RE = re.compile(
    r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b",
    re.I,
)
_BUNDLE_RE = re.compile(
    r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s*protector|charger|accessor(?:y|ies))\b))",
    re.I,
)


def _identity_text(text: str, title: bool = False) -> str:
    """Strip claims that describe what a listing is NOT, or what ships beside it."""
    text = _NEGATED_RE.sub("", text or "")
    return _BUNDLE_RE.split(text, maxsplit=1)[0] if title else text


def _phone_identity(facts: dict, title: str, options: str) -> str:
    """'yes' / 'no' / 'unclear' for whether public evidence makes this a real handset."""
    if _PART_ONLY_RE.search(options or ""):
        return "no"
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or "")
    head = _identity_text(raw_title, title=True)
    kind = " ".join(_identity_text(str(facts.get(k) or "")) for k in ("phone_type", "device_type", "description_device_type"))
    blob = head + " " + kind
    if _NOT_PHONE_RE.search(blob):
        return "no"
    if any("[truncated]" in str(facts.get(k) or "") for k in ("phone_type", "device_type", "device_operating_system")):
        return "unclear"
    if _IS_PHONE_RE.search(blob):
        return "yes"
    os_blob = " ".join(
        _identity_text(str(facts.get(k) or ""))
        for k in ("device_operating_system", "description_operating_system")
        if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
    )
    if _OS_RE.search(os_blob):
        return "yes"
    if _OS_RE.search(head) and not re.search(r"\bcompatib", raw_title, re.I):
        return "yes"
    return "unclear"


# ---------------------------------------------------------- shared capabilities
# Selective donor behavior from MY_AGENT, adapted to HIORO's controller. These
# types are the single canonical TF1–TF7 vocabulary; Brief/Ledger/Agent own the
# loop, and these helpers evaluate evidence without inventing a second pipeline.

CHECKOUT_RESERVE = 3
RECOVERY_RESERVE = 3

_NEG_SEARCH_RE = re.compile(
    r"\b(?:reject|exclude|not)\s+[\w\s-]{0,40}?"
    r"(?:accessor(?:y|ies)|non[- ]devices?|device listings?)\b"
    r"|\bnon[- ]devices?\b"
    r"|\breject\s+accessor(?:y|ies)\b",
    re.I,
)
_QTY_RE = re.compile(
    r"\b(?:buy|order|need|want|get|purchase)\s+(\d{1,3})\s+"
    r"(?:items?|units?|pieces?|pcs|qty)\b",
    re.I,
)
_QTY_OF_RE = re.compile(r"\bquantity(?:\s+of)?\s+(\d{1,3})\b", re.I)
_OBJ_CHEAP_RE = re.compile(
    r"\b(?:cheapest|lowest(?:\s+possible)?\s+price|minimize(?:\s+the)?\s+price|"
    r"least\s+expensive|lowest[- ]cost)\b",
    re.I,
)
_OBJ_RATING_RE = re.compile(r"\b(?:highest[- ]rated|highest\s+rating|best\s+rating)\b", re.I)
_OBJ_STORAGE_RE = re.compile(r"\b(?:largest|most|highest)\s+storage\b", re.I)
_OBJ_DELIVERY_RE = re.compile(r"\b(?:fastest|earliest)\s+delivery\b", re.I)
_BRAND_UPDATE_RE = re.compile(
    r"\b(?:required brand(?:\s+is)?|brand (?:is|must be)|only(?:\s+buy)?)\s+"
    r"([A-Z][\w&.'\-]{1,32})",
)
_ACCESSORY_PATH = (
    "accessory", "accessories", "parts & accessories", "cases & covers",
    "screen protector", "replacement parts", "cables & interconnects",
    "phone charms", "case", "cover", "charger", "protector",
)


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class PrefVerdict(str, Enum):
    MATCH = "match"
    NO_MATCH = "no_match"
    UNKNOWN = "unknown"


class ToolRole(str, Enum):
    SEARCH = "search"
    FILTER = "filter"
    DETAIL = "detail"
    COMPARE = "compare"
    CURRENT_STATE = "current_state"
    CART = "cart"
    FINALIZE = "finalize"
    MESSAGE = "message"
    OTHER = "other"


class FailureType(str, Enum):
    NO_RESULTS = "no_results"
    TOOL_ERROR = "tool_error"
    INVALID_ARGUMENT = "invalid_argument"
    CONSTRAINT_FAILURE = "constraint_failure"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ACTION_REJECTED = "action_rejected"
    PRICE_CHANGED = "price_changed"
    OUT_OF_STOCK = "out_of_stock"
    SHOPPER_UPDATE = "shopper_update"
    PARTIAL_RESULT = "partial_result"


@dataclass
class Constraint:
    type: str
    operator: str | None
    value: Any
    unit: str | None
    source_text: str
    origin: str = "shopper"


@dataclass
class Preference:
    type: str
    value: Any
    priority: int | None
    source_text: str


@dataclass
class PolicyRule:
    type: str
    operator: str | None
    value: Any
    source_text: str
    origin: str = "task_rule"


@dataclass
class CoverageRequirement:
    kind: str
    minimum: int


@dataclass
class Fact:
    name: str
    value: Any
    unit: str | None
    source_tool: str
    source_turn: int
    source_role: ToolRole = ToolRole.OTHER
    confidence: float = 1.0
    volatile: bool = False


@dataclass
class RankingState:
    ordered_candidate_ids: list[str]
    selected_candidate_id: str | None


_VOLATILE = frozenset({"price", "availability", "in_stock", "stock", "current_price"})
_CURRENT_TYPES = frozenset({"price", "availability", "stock", "in_stock"})

_ROLE_BY_NAME = {
    "compare": ToolRole.COMPARE,
    "view": ToolRole.DETAIL,
    "get_details": ToolRole.DETAIL,
    "product_info": ToolRole.DETAIL,
    "inspect_stock": ToolRole.CURRENT_STATE,
    "check_stock": ToolRole.CURRENT_STATE,
    "current_stock": ToolRole.CURRENT_STATE,
    "inventory_status": ToolRole.CURRENT_STATE,
    "search": ToolRole.SEARCH,
    "filter": ToolRole.FILTER,
    "inspect_cart": ToolRole.CART,
    "add_to_cart": ToolRole.CART,
    "remove_from_cart": ToolRole.CART,
    "place_test_order": ToolRole.FINALIZE,
    "message": ToolRole.MESSAGE,
}


def _tool_fn_name(tool: dict) -> str | None:
    if not isinstance(tool, dict):
        return None
    name = (tool.get("function") or {}).get("name") or tool.get("name")
    return str(name) if name else None


def _tool_properties(tool: dict) -> dict:
    fn = tool.get("function") if isinstance(tool, dict) else None
    params = (fn or tool or {}).get("parameters") if isinstance(fn or tool, dict) else None
    if not isinstance(params, dict):
        return {}
    props = params.get("properties")
    return props if isinstance(props, dict) else {}


def infer_tool_role(tool: dict) -> ToolRole:
    name = (_tool_fn_name(tool) or "").lower().strip()
    if name in _ROLE_BY_NAME:
        return _ROLE_BY_NAME[name]
    props = _tool_properties(tool)
    if name == "compare" or any(tok in name for tok in ("compare", "diff", "versus")):
        return ToolRole.COMPARE
    if "cart" not in name and any(
        tok in name for tok in ("inspect_stock", "check_stock", "current_stock", "inventory")
    ):
        return ToolRole.CURRENT_STATE
    if name.startswith("inspect") and "cart" not in name and any(
        tok in name for tok in ("stock", "price", "inventory")
    ):
        return ToolRole.CURRENT_STATE
    if any(tok in name for tok in ("view", "detail", "product_info", "get_details")):
        return ToolRole.DETAIL
    if "filter" in name:
        return ToolRole.FILTER
    if "search" in name or name == "find" or name.startswith("find"):
        return ToolRole.SEARCH
    if any(tok in name for tok in ("cart", "basket")):
        return ToolRole.CART
    if any(tok in name for tok in ("order", "checkout", "finalize", "purchase")):
        return ToolRole.FINALIZE
    if any(tok in name for tok in ("message", "reply", "respond")):
        return ToolRole.MESSAGE
    if "product_ids" in props or "ids" in props:
        return ToolRole.COMPARE
    if "sku" in props and "product_id" in props and "inspect" in name:
        return ToolRole.CURRENT_STATE
    if "product_id" in props or "sku" in props:
        return ToolRole.DETAIL
    return ToolRole.OTHER


def build_tool_role_map(tools: list) -> dict[str, ToolRole]:
    out: dict[str, ToolRole] = {}
    for tool in tools or []:
        name = _tool_fn_name(tool)
        if name:
            out[name] = infer_tool_role(tool)
    return out


def role_to_name(role_map: dict[str, ToolRole], role: ToolRole, *preferred: str) -> str | None:
    for name in preferred:
        if name in role_map:
            return name
    for name, found in role_map.items():
        if found == role:
            return name
    return None


def fit_tool_args(tool: dict | None, args: dict) -> dict:
    if not tool or not isinstance(args, dict):
        return dict(args or {})
    props = _tool_properties(tool)
    if not props:
        return dict(args)
    return {k: v for k, v in args.items() if k in props}


def sanitize_search_query(query: str) -> str:
    """Drop negative verification rules that must never become search text."""
    cleaned = _NEG_SEARCH_RE.sub(" ", query or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def unwrap_observation(observation: Any) -> Any:
    current = observation
    seen: set[int] = set()
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        inner = current.get("observation")
        if isinstance(inner, dict) and (
            any(
                key in inner
                for key in ("category_path", "facts", "compared", "variants", "product_id", "results", "added", "cart", "product")
            )
        ):
            current = inner
            continue
        product = current.get("product")
        if isinstance(product, dict) and any(
            key in product for key in ("category_path", "facts", "product_id")
        ):
            current = product
            continue
        break
    return current


def iter_nested_dicts(value: Any):
    stack = [value]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            ident = id(current)
            if ident in seen:
                continue
            seen.add(ident)
            yield current
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def extract_market_event(result: Any) -> dict | None:
    for node in iter_nested_dicts(result):
        event = node.get("market_event")
        if isinstance(event, dict) and (event.get("kind") or event.get("new_price") is not None):
            return event
    return None


def _shopper_text(value: Any) -> str | None:
    if isinstance(value, dict):
        content = str(value.get("content") or "").strip()
        return content or None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def extract_shopper_update(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    for key in ("user_message", "shopper_message", "shopper_update"):
        found = _shopper_text(result.get(key))
        if found:
            return found
    for node in iter_nested_dicts(result):
        for key in ("user_message", "shopper_message", "shopper_update"):
            found = _shopper_text(node.get(key))
            if found:
                return found
    return None


def category_path_parts(observed: Any) -> list[str]:
    if isinstance(observed, list):
        return [str(part).strip().lower() for part in observed if str(part).strip()]
    if observed is None:
        return []
    return [part.strip().lower() for part in re.split(r"[>/|,]", str(observed)) if part.strip()]


def required_category_present(parts: list[str], required: str) -> bool:
    want = (required or "").strip().lower()
    if not want:
        return False
    want_stem = re.sub(r"[^a-z0-9]+", "", want).rstrip("s")
    for part in parts:
        compact = re.sub(r"[^a-z0-9]+", "", part)
        if want in part or part in want:
            return True
        if want_stem and want_stem in compact:
            return True
    return False


def category_path_is_accessory(parts: list[str]) -> bool:
    return any(any(marker in part for marker in _ACCESSORY_PATH) for part in parts)


def verify_title_tokens(title: str, required: list[str]) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", (title or "").lower()))
    return all(re.sub(r"[^a-z0-9]+", "", term.lower()) in tokens for term in required if term)


def parse_explicit_objective(text: str) -> tuple[str, str] | None:
    """Only an objective the shopper or task actually stated."""
    blob = text or ""
    if _OBJ_CHEAP_RE.search(blob):
        return ("price", "minimize")
    if _OBJ_RATING_RE.search(blob):
        return ("rating", "maximize")
    if _OBJ_STORAGE_RE.search(blob):
        return ("storage", "maximize")
    if _OBJ_DELIVERY_RE.search(blob):
        return ("delivery", "minimize")
    return None


def parse_quantity(text: str) -> int | None:
    m = _QTY_RE.search(text or "") or _QTY_OF_RE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def classify_failure(
    *,
    tool_error: bool = False,
    invalid_args: bool = False,
    no_results: bool = False,
    price_changed: bool = False,
    out_of_stock: bool = False,
    shopper_update: bool = False,
    constraint_fail: bool = False,
    missing_evidence: bool = False,
    rejected: bool = False,
) -> FailureType:
    if shopper_update:
        return FailureType.SHOPPER_UPDATE
    if price_changed:
        return FailureType.PRICE_CHANGED
    if out_of_stock:
        return FailureType.OUT_OF_STOCK
    if tool_error:
        return FailureType.TOOL_ERROR
    if invalid_args:
        return FailureType.INVALID_ARGUMENT
    if no_results:
        return FailureType.NO_RESULTS
    if constraint_fail:
        return FailureType.CONSTRAINT_FAILURE
    if missing_evidence:
        return FailureType.INSUFFICIENT_EVIDENCE
    if rejected:
        return FailureType.ACTION_REJECTED
    return FailureType.PARTIAL_RESULT


class IdentityIndex:
    """Union-find over product identities: id, model, or brand+title+close price."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.meta: dict[str, dict] = {}

    def _find(self, key: str) -> str:
        while self.parent.get(key, key) != key:
            self.parent[key] = self.parent.get(self.parent[key], self.parent[key])
            key = self.parent[key]
        return key

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self.parent[rb] = ra

    def add(
        self,
        pid: str,
        *,
        title: str = "",
        brand: str = "",
        model: str = "",
        price: float | None = None,
        matching: bool = True,
    ) -> None:
        if not matching or not pid:
            return
        pid = str(pid)
        self.parent.setdefault(pid, pid)
        tokens = frozenset(re.findall(r"[a-z0-9]+", (title or "").lower()))
        model_n = re.sub(r"[^a-z0-9]+", "", (model or "").lower())
        brand_n = re.sub(r"[^a-z0-9]+", "", (brand or "").lower())
        self.meta[pid] = {
            "title": title, "brand": brand_n, "model": model_n,
            "tokens": tokens, "price": price,
        }
        for other, info in self.meta.items():
            if other == pid:
                continue
            if model_n and info["model"] and model_n == info["model"]:
                self._union(pid, other)
                continue
            if brand_n and info["brand"] == brand_n and tokens and tokens == info["tokens"]:
                op, tp = info["price"], price
                if op is None or tp is None or abs(op - tp) <= max(1.0, 0.05 * max(op, tp, 1.0)):
                    self._union(pid, other)

    def count(self) -> int:
        return len({self._find(k) for k in self.parent})


# Which shape of task this is. Structure decides it wherever structure can: named
# candidate variants in the goal, a stated number of identities to surface, or a
# stated exact model and color. Only the rest fall back to a phrase in the rules.
_FAMILY_SIGNS = (
    ("preference", "listing_id"),
    ("constraint", "exact variant"),
    ("justification", "trustworthy reason"),
    ("intent", "nice-to-haves"),
)

_PLAYBOOK = {
    "intent": (
        "Firm requirements: the category, the named hardware specification, the budget and stock. A "
        "named brand is only a preference: choose it when a fully compliant item of that brand exists, "
        "otherwise any compliant brand. Use filter(category, brand, max_price, in_stock=true), then view "
        "or compare to read the exact variant's specification from raw facts. Answer the shopper with "
        "message; a budget the shopper lowers later applies to every later choice."
    ),
    "recall": (
        "Run several differently worded searches (k=10, in_stock=true) built from the requested words and "
        "close synonyms until enough distinct matching products have appeared; every requested word must "
        "appear as a whole word of the title. Then view the strongest candidates so the controller can "
        "verify one and buy it."
    ),
    "constraint": (
        "The shopper hides one firm hardware requirement and answers it when asked; obey it exactly for "
        "the specific variant, reading the variant's own option text before any product-wide value. Use "
        "filter(category, brand, max_price, in_stock=true) and view candidates. If the carted variant "
        "sells out, move to a different product that still meets every requirement."
    ),
    "preference": (
        "You already asked which preference matters most; use the answer. Search or filter the category "
        "within budget, then compare candidates with different listing_id values, reading raw facts. The "
        "first cart item is graded, so make it the best genuine match. If it is repriced above budget, "
        "recover to a different listing_id."
    ),
    "ranking": (
        "Only the named candidate variants count, and the controller ranks them for you: compare all of "
        "them first so their facts are on the record. Highest advertised maximum refresh rate wins, then "
        "the largest screen, then the lowest price, reading the selected variant's own value before a "
        "product-wide one. After the market moves, the remaining candidates are re-ranked."
    ),
    "recovery": (
        "Find the preferred model and color first: search the model code, then view it so every variant "
        "and color is on the record. The first cart item must be exactly that model and color. If it "
        "disappears, relax as little as possible - the same model in another color before any other "
        "model - and never relax the category, budget or stock."
    ),
    "justification": (
        "Find a product whose own raw facts state the requested specification, in the requested category "
        "and brand, in stock and within budget. View or compare it so its facts are on the record, then "
        "the controller inspects it, carts it and orders it with claims taken only from that evidence."
    ),
    "generic": (
        "Understand the firm requirements, filter or search the catalog, view candidates, and let the "
        "controller verify and buy the best one."
    ),
}

_CORE_RULES = (
    "You are a careful shopping agent working a catalog through tools. Exactly one tool call per turn, at "
    "most 30 turns, so be efficient. A deterministic controller checks every proposal and takes over the "
    "cart and the order once a candidate is fully verified, so your job is to surface and read candidates: "
    "search or filter, then view or compare so their raw facts, variant options and prices are on record. "
    "Never add an item to the cart as a placeholder - the first cart addition is graded. Prices and stock "
    "change while you shop; the controller re-checks the carted item and tells you when to move to a "
    "different product. Answer every direct question from the shopper with one short message before "
    "anything else, and never send progress or status updates. Never ask the shopper to raise the budget. "
    "Do not repeat a tool call with identical arguments; one compare of up to five products beats several "
    "view calls."
)


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


class Brief:
    """What the deterministic side needs, parsed from the public request text."""

    def __init__(self, query: str) -> None:
        self.raw = query or ""
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self._budget(self.goal)
        self.currency = "PHP"
        m = re.search(r"\b([A-Z]{3})\b", self.goal)
        if m and m.group(1) in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
            self.currency = m.group(1)
        self.candidate_keys = [(pid, sku.rstrip(".,;:")) for pid, sku in _KEY_RE.findall(self.goal)]
        phrase = _PHRASE_RE.search(self.goal)
        quoted = _QUOTED_RE.findall(self.goal)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        self.theme = re.findall(r"[a-z0-9]+", source.lower())
        sm = _SURFACE_RE.search(self.rules)
        self.required_identities = int(sm.group(1)) if sm else 0
        pm = _PREF_ITEM_RE.search(self.goal)
        self.pref_color = pm.group(1).lower() if pm else None
        self.pref_model = pm.group(2) if pm else None
        self.category = self._category(self.goal)
        self.brand = self._brand(self.goal, self.category)
        self.pref_brand = None if pm else self._pref_brand(self.goal)
        self.cat_qualifier = self._qualifier(self.goal)
        self.specs = self._specs(self.goal)
        self.spec_field, self.spec_value = self._spec(self.goal)
        # A hidden requirement is stated only when the shopper is asked for it.
        self.spec_sealed = self.spec_field is None and "firm requirement" in self.goal.lower()
        self.quantity = parse_quantity(self.goal)
        self.objective = parse_explicit_objective(self.goal + "\n" + self.rules)
        self.constraints: list[Constraint] = []
        self.preferences: list[Preference] = []
        self.policy_rules: list[PolicyRule] = []
        self.coverage: CoverageRequirement | None = None
        self.needs_justification = bool(_REASON_RE.search(self.goal) or "trustworthy reason" in self.rules)
        self.honor_pref_brand = False
        self.reject_accessories = False
        self.family = self._family()
        self._build_structure()

    def _family(self) -> str:
        if self.candidate_keys:
            return "ranking"
        if self.required_identities:
            return "recall"
        if self.pref_model:
            return "recovery"
        for name, sign in _FAMILY_SIGNS:
            if sign in self.rules:
                return name
        # The rules text is the primary signal. If it ever reads differently, fall back
        # on what the shopper themself asked for rather than giving up on the task shape.
        if self.theme:
            return "recall"
        if _REASON_RE.search(self.goal):
            return "justification"
        if self.pref_brand:
            return "intent"
        if self.spec_sealed or self.spec_field or self.brand:
            return "constraint"
        return "preference"

    @staticmethod
    def _budget(text: str) -> float | None:
        best = None
        for m in _BUDGET_RE.finditer(text or ""):
            whole = m.group(1).replace(",", "")
            frac = m.group(2) or "0"
            try:
                val = float(whole + "." + frac)
            except ValueError:
                continue
            if best is None or val < best:
                best = val
        return best

    @staticmethod
    def _category(goal: str) -> str | None:
        m = _CAT_PHRASE_RE.search(goal or "")
        if m:
            return m.group(1).strip()
        m = re.search(
            r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
            goal or "",
            re.I,
        )
        return m.group(1) if m else None

    @staticmethod
    def _brand(goal: str, category: str | None) -> str | None:
        """A brand named as part of the requested item, e.g. 'a <Brand> <item>'."""
        if not category:
            return None
        m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category), goal or "", re.I)
        if not m:
            return None
        words = [w for w in re.findall(r"[\w&.'\-]+", _SPEC_WORD_RE.sub(" ", m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not _res_vals(w)]
        return " ".join(keep) or None

    @staticmethod
    def _pref_brand(goal: str) -> str | None:
        m = _PREF_BRAND_RE.search(goal or "")
        if not m:
            return None
        name = m.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return None
        return name

    @staticmethod
    def _qualifier(goal: str) -> str | None:
        """A product shape that no shelf name captures, so the title must show it."""
        if re.search(r"\bgaming\s+monitors?\b", goal or "", re.I):
            return "gaming"
        if re.search(r"\bover[\s-]?ear\b", goal or "", re.I):
            return "over ear"
        return None

    @staticmethod
    def _specs(goal: str) -> dict:
        """Every hardware value the request states unambiguously.

        A request can name more than one ("a 27-inch 144Hz monitor") while only
        one of them is the gated requirement, so all of them are kept: the gate is
        honored whichever it turns out to be, and a reason can cite each of them.
        """
        out = {}
        for field in ("screen_size", "refresh_rate", "resolution", "ram"):
            value = _wanted_value(field, goal)
            if value:
                out[field] = value
        return out

    def _spec(self, goal: str) -> tuple:
        """The requirement the controller gates on: the first one stated."""
        for field, value in self.specs.items():
            return field, value
        return None, None

    def adopt_spec(self, text: str) -> str | None:
        """Record a requirement the shopper reveals later; returns its description."""
        for field in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = _wanted_value(field, text)
            if value:
                self.spec_field, self.spec_value = field, value
                self.specs[field] = value
                self.spec_sealed = False
                self._upsert_constraint(field, "=", value, "in" if field == "screen_size" else None, text, "shopper_update")
                return field + " = " + value
        return None

    def _upsert_constraint(
        self,
        ctype: str,
        operator: str | None,
        value: Any,
        unit: str | None,
        source_text: str,
        origin: str,
    ) -> None:
        for i, row in enumerate(self.constraints):
            if row.type == ctype:
                self.constraints[i] = Constraint(ctype, operator, value, unit, source_text, origin)
                return
        self.constraints.append(Constraint(ctype, operator, value, unit, source_text, origin))

    def _build_structure(self) -> None:
        """Canonical TF1 objects from the same public text Brief already parsed."""
        if self.category:
            self.constraints.append(
                Constraint("category", "=", self.category, None, self.category, "shopper")
            )
        if self.brand:
            self.constraints.append(
                Constraint("brand", "=", self.brand, None, self.brand, "shopper")
            )
        if self.budget is not None:
            self.constraints.append(
                Constraint("price", "<=", self.budget, self.currency, "budget", "shopper")
            )
        self.constraints.append(
            Constraint("availability", "=", True, None, "in stock", "shopper")
        )
        for field, value in self.specs.items():
            unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}.get(field)
            self.constraints.append(
                Constraint(field, "=", value, unit, value, "shopper")
            )
        if self.theme:
            self.constraints.append(
                Constraint("title_match", "contains", list(self.theme), None, " ".join(self.theme), "task_rule")
            )
        if self.candidate_keys:
            self.constraints.append(
                Constraint(
                    "candidate_variants",
                    "in",
                    [pid + "::" + sku for pid, sku in self.candidate_keys],
                    None,
                    "named variants",
                    "task_rule",
                )
            )
        if self.pref_brand:
            self.preferences.append(Preference("brand", self.pref_brand, 1, self.pref_brand))
        if self.pref_model:
            self.preferences.append(Preference("model", self.pref_model, 1, self.pref_model))
        if self.pref_color:
            self.preferences.append(Preference("color", self.pref_color, 2, self.pref_color))
        self._parse_policy()

    def _parse_policy(self) -> None:
        rules = self.rules
        if "reject" in rules and "accessor" in rules:
            self.reject_accessories = True
            self.policy_rules.append(
                PolicyRule("reject_accessories", "excludes", "accessories", "reject accessories")
            )
        if "nice-to-have" in rules or "prefer" in rules:
            if self.pref_brand:
                self.honor_pref_brand = True
                self.policy_rules.append(
                    PolicyRule("honor_brand_preference", "=", self.pref_brand, "prefer brand when compliant")
                )
        if self.pref_brand and self.family == "intent":
            self.honor_pref_brand = True
        if "listing_id" in rules:
            self.policy_rules.append(
                PolicyRule("inspect_distinct_listings", ">=", 2, "inspect distinct listing_id")
            )
            if self.coverage is None:
                self.coverage = CoverageRequirement("listing_id", 2)
        if self.required_identities:
            self.coverage = CoverageRequirement("product_identity", self.required_identities)
            self.policy_rules.append(
                PolicyRule("surface_identities", ">=", self.required_identities, "distinct identities")
            )
        if "exact variant" in rules or "firm requirement" in self.goal.lower():
            self.policy_rules.append(
                PolicyRule("sku_option_verification", "=", True, "verify the selected variant")
            )
        if self.needs_justification:
            self.policy_rules.append(
                PolicyRule("justification", "=", True, "trustworthy reason")
            )
        cat = (self.category or "").lower()
        if "smartphone" in cat or "phone" in cat:
            self.policy_rules.append(
                PolicyRule("device_verification", "=", "smartphone", "device identity")
            )
            if "accessor" in rules or "device" in rules:
                self.reject_accessories = True

    def apply_shopper_update(self, text: str) -> list[str]:
        """Replace only the requirements the shopper changed; keep the rest."""
        changes: list[str] = []
        new_budget = Brief._budget(text)
        if new_budget is not None:
            self.budget = new_budget
            self._upsert_constraint("price", "<=", new_budget, self.currency, text, "shopper_update")
            changes.append("budget=" + ("%.2f" % new_budget))
        stated = self.adopt_spec(text)
        if stated:
            changes.append(stated)
        qty = parse_quantity(text)
        if qty is not None:
            self.quantity = qty
            changes.append("quantity=" + str(qty))
        brand_m = _BRAND_UPDATE_RE.search(text or "")
        if brand_m:
            name = brand_m.group(1).strip(" .,")
            if "prefer" in (text or "").lower():
                self.pref_brand = name
                self.brand = None
                self.honor_pref_brand = True
                self.preferences = [p for p in self.preferences if p.type != "brand"]
                self.preferences.append(Preference("brand", name, 1, text))
                self.constraints = [c for c in self.constraints if c.type != "brand"]
                changes.append("pref_brand=" + name)
            else:
                self.brand = name
                self._upsert_constraint("brand", "=", name, None, text, "shopper_update")
                changes.append("brand=" + name)
        obj = parse_explicit_objective(text)
        if obj:
            self.objective = obj
            changes.append("objective=" + obj[0])
        return changes


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


class Ledger:
    """What the agent has observed for itself, keyed by product and variant."""

    def __init__(self) -> None:
        self.cart: list = []
        self.checked: dict = {}        # variant key -> latest price/stock
        self.checked_seq: dict = {}    # variant key -> when that was learned
        self.inspected: set = set()    # variants confirmed by a stock inspection
        self.products: dict = {}       # product id -> view/compare record
        self.rows: dict = {}           # variant key -> summary row from a listing page
        self.options: dict = {}        # variant key -> selected option text
        self.listing_ids: set = set()
        self.listing_of: dict = {}   # product id -> the listing it belongs to
        self.identities: set = set()
        self.identity_index = IdentityIndex()
        self.facts: dict[str, list[Fact]] = {}
        self.seq = 0
        self.first_add_seq: int | None = None
        self.last_add_seq: int | None = None
        self.adds = 0
        self.event_target: str | None = None
        self.dead_keys: set = set()
        self.dead_products: set = set()
        self.calls_seen: set = set()
        self.last_market_event: dict | None = None
        self.last_failure: FailureType | None = None

    @staticmethod
    def key(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def record(self, name: str, args: dict, obs: dict, brief: Brief, role: ToolRole | None = None) -> None:
        self.seq += 1
        inner = unwrap_observation(obs) if isinstance(obs, dict) else obs
        if not isinstance(inner, dict):
            return
        if inner.get("error") and not inner.get("results") and not inner.get("added"):
            return
        role = role or _ROLE_BY_NAME.get(name, ToolRole.OTHER)
        if role in (ToolRole.SEARCH, ToolRole.FILTER) or name in ("search", "filter"):
            for row in inner.get("results") or inner.get("products") or []:
                if isinstance(row, dict):
                    self._note_row(row, brief, name, role)
        elif role == ToolRole.DETAIL or name == "view":
            self._note_product(inner, brief, name, role)
        elif role == ToolRole.COMPARE or name == "compare":
            for prod in inner.get("compared") or inner.get("products") or []:
                if isinstance(prod, dict):
                    self._note_product(prod, brief, name, role)
        elif role == ToolRole.CURRENT_STATE or name == "inspect_stock":
            self._note_current_state(inner, name, role)
        elif name == "add_to_cart" or (role == ToolRole.CART and "product_id" in (args or {}) and "remove" not in name):
            added = inner.get("added") or {}
            pid, sku = added.get("product_id") or (args or {}).get("product_id"), added.get("sku") or (args or {}).get("sku")
            if pid and sku:
                self.adds += 1
                if (pid, sku) not in self.cart:
                    self.cart.append((pid, sku))
                if self.first_add_seq is None:
                    self.first_add_seq = self.seq
                self.last_add_seq = self.seq
            self._sync_cart(inner)
            self._apply_market_event(inner, name)
        elif name == "remove_from_cart":
            self._sync_cart(inner)
            removed = inner.get("removed") or {}
            pair = (removed.get("product_id"), removed.get("sku"))
            if pair in self.cart:
                self.cart.remove(pair)
        elif name == "inspect_cart" or role == ToolRole.CART:
            self._sync_cart(inner)
            self._apply_market_event(inner, name)

    def _sync_cart(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [(r.get("product_id"), r.get("sku")) for r in rows if isinstance(r, dict) and r.get("product_id")]
        if pairs or rows == []:
            self.cart = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            k = self.key(r.get("product_id"), r.get("sku"))
            price = _num(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.checked.get(k, {})
            self.checked[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
                "auth": True,
            }
            self.checked_seq[k] = self.seq
            self._add_fact(
                str(r.get("product_id")),
                "price",
                price,
                None,
                "cart",
                ToolRole.CURRENT_STATE,
                volatile=True,
            )

    def _note_current_state(self, obs: dict, name: str, role: ToolRole) -> None:
        k = self.key(obs.get("product_id"), obs.get("sku"))
        self.checked[k] = {
            "in_stock": bool(obs.get("in_stock")),
            "price": _num(obs.get("price")),
            "auth": True,
        }
        self.checked_seq[k] = self.seq
        self.inspected.add(k)
        pid = str(obs.get("product_id") or "")
        if pid:
            self._add_fact(pid, "price", _num(obs.get("price")), None, name, role, volatile=True)
            self._add_fact(pid, "availability", bool(obs.get("in_stock")), None, name, role, volatile=True)

    def _apply_market_event(self, obs: dict, name: str) -> None:
        event = extract_market_event(obs)
        if not event:
            return
        self.last_market_event = event
        pid = event.get("product_id") or (obs.get("added") or {}).get("product_id")
        sku = event.get("sku") or (obs.get("added") or {}).get("sku")
        if not pid:
            return
        k = self.key(pid, sku)
        prev = self.checked.get(k, {})
        price = _num(event.get("new_price"))
        if price is None:
            price = _num(event.get("price"))
        stock = event.get("in_stock_after")
        if stock is None:
            stock = event.get("in_stock")
        self.checked[k] = {
            "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
            "price": price if price is not None else prev.get("price"),
            "auth": True,
        }
        self.checked_seq[k] = self.seq
        self.inspected.add(k)
        if price is not None:
            self._add_fact(str(pid), "price", price, None, name, ToolRole.CURRENT_STATE, volatile=True)
        if stock is not None:
            self._add_fact(str(pid), "availability", bool(stock), None, name, ToolRole.CURRENT_STATE, volatile=True)

    def _add_fact(
        self,
        pid: str,
        name: str,
        value: Any,
        unit: str | None,
        source_tool: str,
        source_role: ToolRole,
        *,
        volatile: bool = False,
        confidence: float = 1.0,
    ) -> None:
        if value is None or not pid:
            return
        key = pid + "::" + name
        rows = self.facts.setdefault(key, [])
        fact = Fact(
            name=name,
            value=value,
            unit=unit,
            source_tool=source_tool,
            source_turn=self.seq,
            source_role=source_role,
            confidence=confidence,
            volatile=volatile or name in _VOLATILE,
        )
        rows.append(fact)

    def latest_fact(self, pid: str, name: str, *, authoritative: bool = False) -> Fact | None:
        rows = self.facts.get(str(pid) + "::" + name) or []
        if authoritative:
            rows = [f for f in rows if f.source_role == ToolRole.CURRENT_STATE]
        return rows[-1] if rows else None

    def _note_row(self, row: dict, brief: Brief, name: str = "search", role: ToolRole = ToolRole.SEARCH) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        matching = (not brief.theme) or verify_title_tokens(title, brief.theme)
        if matching:
            self.identities.add(str(pid))
            self.identity_index.add(
                str(pid),
                title=title,
                brand=str(row.get("brand") or ""),
                model=str(row.get("model") or ""),
                price=_num(row.get("price")),
                matching=True,
            )
        if not row.get("sku"):
            return
        k = self.key(pid, row.get("sku"))
        self.rows[k] = row
        if row.get("price") is not None:
            self.checked.setdefault(
                k,
                {"in_stock": bool(row.get("in_stock", True)), "price": _num(row.get("price")), "auth": False},
            )

    def _note_product(self, prod: dict, brief: Brief, name: str = "view", role: ToolRole = ToolRole.DETAIL) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.products[str(pid)] = prod
        if prod.get("listing_id"):
            self.listing_ids.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        title = str(prod.get("title") or "")
        facts = prod.get("facts") if isinstance(prod.get("facts"), dict) else {}
        matching = (not brief.theme) or verify_title_tokens(title, brief.theme)
        if matching:
            self.identities.add(str(pid))
            self.identity_index.add(
                str(pid),
                title=title,
                brand=str(prod.get("brand") or ""),
                model=str(facts.get("model") or ""),
                price=_num(prod.get("price")),
                matching=True,
            )
        path = prod.get("category_path")
        if path:
            self._add_fact(str(pid), "category", path, None, name, role)
        if prod.get("brand"):
            self._add_fact(str(pid), "brand", prod.get("brand"), None, name, role)
        if title:
            self._add_fact(str(pid), "title", title, None, name, role, confidence=0.2)
        for label, value in facts.items():
            if value is None:
                continue
            self._add_fact(str(pid), str(label), value, None, name, role)
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            k = self.key(pid, v.get("sku"))
            prev = self.checked.get(k) or {}
            if prev.get("auth"):
                self.options[k] = str(v.get("options") or "")
                continue
            self.checked[k] = {
                "in_stock": bool(v.get("in_stock")),
                "price": _num(v.get("price")),
                "auth": False,
            }
            self.checked_seq[k] = self.seq
            self.options[k] = str(v.get("options") or "")

    def status(self, pid: str, sku: str) -> dict | None:
        return self.checked.get(self.key(pid, sku))


def _arguments(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + ' ... (truncated by agent)"}'


class Agent:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_by_name = {n: t for t in self.tools if (n := _tool_fn_name(t))}
        self.tool_names = set(self.tool_by_name)
        self.tool_roles = build_tool_role_map(self.tools)
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.brief = Brief(str(self.policy.get("query") or ""))
        self.budget = self.brief.budget
        self.ledger = Ledger()
        self.clock = Clock(_PROBLEM_BUDGET_S)
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.turn = 0
        self.asked_shopper = False
        self.shopper_question_pending = False
        self.shopper_updated = False
        self.rejected_streak = 0
        self.pending_envelope: dict | None = None
        self.search_calls = 0
        self.cart_inspections = 0
        self.pending_recheck: tuple | None = None
        self.forced_compares = 0
        self.hinted_exact = False
        self.tried: dict = {}
        self.questions_asked = 0
        self.searched_siblings = False
        self.hinted_brand = False
        self._wishes: set | None = None
        self.probes: set = set()
        self.searched_twins = False
        self.reads_done = False
        self.event_turn: int | None = None
        self.unreadable: set = set()
        self.question_waits = 0
        self.done = False
        self.order_outcome: dict | None = None
        self.model_index = 0
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    # ------------------------------------------------------------------ prompts
    def _system_prompt(self) -> str:
        play = _PLAYBOOK.get(self.brief.family, _PLAYBOOK["generic"])
        extra = ""
        if self.brief.family == "recall" and self.brief.required_identities:
            extra = (
                " You must surface at least %d distinct matching products in search or filter "
                "results before ordering; the controller reports the count after each search."
                % self.brief.required_identities
            )
        return _CORE_RULES + "\n\nFamily playbook: " + play + extra

    def _opening_question(self) -> str | None:
        if self.brief.family == "constraint" or self.brief.spec_sealed:
            return (
                "Before I search: which hardware specification is your firm requirement "
                "(screen size, resolution class, refresh rate, or memory), and what value do you need?"
            )
        if self.brief.family == "preference":
            return (
                "Quick question so I choose well: which of your preferences counts most, and in what "
                "order should I weigh the others? Let me know your priority."
            )
        return None

    def _note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    def _tool(self, role: ToolRole, *preferred: str) -> str | None:
        return role_to_name(self.tool_roles, role, *preferred)

    def _has(self, role: ToolRole, *preferred: str) -> bool:
        return self._tool(role, *preferred) is not None

    def _is_add(self, name: str) -> bool:
        if name == "add_to_cart":
            return True
        return bool(name) and self.tool_roles.get(name) == ToolRole.CART and "add" in name

    def _cart_add_name(self) -> str | None:
        if "add_to_cart" in self.tool_names:
            return "add_to_cart"
        return next((n for n, r in self.tool_roles.items() if r == ToolRole.CART and "add" in n), None)

    def _cart_remove_name(self) -> str | None:
        if "remove_from_cart" in self.tool_names:
            return "remove_from_cart"
        return next((n for n, r in self.tool_roles.items() if r == ToolRole.CART and "remove" in n), None)

    def _fit(self, name: str, args: dict) -> dict:
        fitted = fit_tool_args(self.tool_by_name.get(name), args)
        if name and self.tool_roles.get(name) == ToolRole.SEARCH and "query" in fitted:
            fitted["query"] = sanitize_search_query(str(fitted.get("query") or ""))
        return fitted

    def _steps_left(self) -> int:
        return self.max_steps - self.turn

    def _in_reserve(self) -> bool:
        need = CHECKOUT_RESERVE + (RECOVERY_RESERVE if not self.ledger.adds else 0)
        return self._steps_left() <= need

    def _identity_count(self) -> int:
        return max(self.ledger.identity_index.count(), len(self.ledger.identities))

    # ------------------------------------------------------------------ evidence
    def _prod(self, pid: str) -> dict:
        return self.ledger.products.get(str(pid)) or {}

    def _facts(self, pid: str) -> dict:
        f = self._prod(pid).get("facts")
        return f if isinstance(f, dict) else {}

    def _title(self, pid: str) -> str:
        title = str(self._prod(pid).get("title") or "")
        if title:
            return title
        for key, row in self.ledger.rows.items():
            if key.startswith(str(pid) + "::"):
                return str(row.get("title") or "")
        return ""

    def _brand_of(self, pid: str) -> str:
        brand = self._prod(pid).get("brand")
        if brand:
            return str(brand)
        for key, row in self.ledger.rows.items():
            if key.startswith(str(pid) + "::") and row.get("brand"):
                return str(row["brand"])
        return ""

    def _cat_path(self, pid: str) -> list:
        path = self._prod(pid).get("category_path")
        return list(path) if isinstance(path, list) else []

    def _options(self, pid: str, sku: str) -> str:
        return self.ledger.options.get(Ledger.key(pid, sku), "")

    def _strip_models(self, pid: str, title: str) -> str:
        """Drop model codes from a title so a product name is not read as a panel class."""
        for token in re.split(r"[;,]", str(self._facts(pid).get("model") or "")):
            token = token.strip()
            stem = re.split(r"[-_]", token)[0] if token else ""
            if (
                token
                and re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)+", token, re.I)
                and re.search(r"[a-z]", stem, re.I)
                and re.search(r"\d", stem)
                and not re.fullmatch(r"(?:[48]k|[ufq]?hd|fullhd|ultrahd|\d+p)\d*", stem, re.I)
            ):
                title = re.sub(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", "", title, flags=re.I)
        return title

    def _fact_value(self, pid: str, field: str) -> str | None:
        """The value the product's own labelled facts state for a concept.

        A reason quoted back to the shopper reads these facts leniently: a fact
        printed as a bare number under a unit-bearing label carries that label's
        unit, so "27" under the size label is 27 inches.
        """
        raw = self._facts(pid).get(_FACT_LABEL[field])
        if not raw:
            return None
        text = str(raw).strip()
        found = _VALUE_SETS[field](text)
        if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text):
            unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[field]
            return "%g" % float(text) + unit
        return _one(found)

    def _product_value(self, pid: str, field: str) -> str | None:
        """Product-wide value: labelled facts plus the title, and nothing when they disagree."""
        vals: set = set()
        conflict = False
        raw = self._facts(pid).get(_FACT_LABEL[field])
        if raw:
            found = _VALUE_SETS[field](str(raw))
            conflict = conflict or len(found) > 1
            if field == "resolution" and _QUARTER_RE.search(str(raw)):
                conflict = True
            vals |= found
        title = self._title(pid)
        if field == "resolution":
            title = self._strip_models(pid, title)
            if _QUARTER_RE.search(title):
                conflict = True
        vals |= _adjacent_vals(field, title)
        return None if conflict else _one(vals)

    def _variant_value(self, pid: str, sku: str, field: str) -> str | None:
        """The value that holds for this exact variant: its own option text wins."""
        opt = self._options(pid, sku)
        vals = _adjacent_vals(field, opt)
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if _mentions(field, opt):
            return None
        return self._product_value(pid, field)

    def _measure(self, pid: str, sku: str, field: str) -> float | None:
        value = self._variant_value(pid, sku, field)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    @staticmethod
    def _stems(text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

    def _category_verdict(self, pid: str) -> Verdict:
        """PASS / FAIL / UNKNOWN for the requested shelf, from the category path."""
        want = self.brief.category
        path = self._cat_path(pid)
        if not want:
            return Verdict.PASS
        if not path:
            return Verdict.UNKNOWN
        parts = category_path_parts(path)
        placed = required_category_present(parts, want)
        if not placed:
            wanted = self._stems(want)
            labels = set()
            for label in path:
                labels |= self._stems(str(label))
            placed = bool(wanted & labels) or any(
                self._stems("".join(re.findall(r"[a-z]+", str(label).lower()))) == wanted for label in path
            )
            if not placed and "smartphone" in wanted:
                placed = any(
                    re.fullmatch(r"(?:smart|cell|mobile|unlocked)?\s*phones?", str(label).strip(), re.I)
                    for label in path
                )
        if not placed:
            return Verdict.FAIL
        qualifier = self.brief.cat_qualifier
        if not qualifier:
            return Verdict.PASS
        title = self._title(pid)
        words = self._stems(title) | _tokens(title)
        if qualifier == "gaming":
            fast = any(float(v[:-2]) >= 120 for v in _hz_vals(title))
            return Verdict.PASS if "monitor" in words and ("gaming" in words or fast) else Verdict.FAIL
        ok = "over" in words and "ear" in words or "overear" in words or "circumaural" in words
        return Verdict.PASS if ok else Verdict.FAIL

    def _category_ok(self, pid: str) -> bool:
        return self._category_verdict(pid) != Verdict.FAIL

    def _brand_ok(self, pid: str) -> bool:
        want = self.brief.brand
        if not want:
            return True
        wanted = _tokens(want)
        return bool(wanted <= _tokens(self._brand_of(pid)) or wanted <= _tokens(self._title(pid)))

    def _device_verdict(self, pid: str, sku: str) -> Verdict:
        """Device/accessory policy: category-path accessories fail; missing evidence is UNKNOWN."""
        if self.brief.reject_accessories or _same_word(self.brief.category or "", "smartphone"):
            path = category_path_parts(self._cat_path(pid))
            if path and category_path_is_accessory(path) and not required_category_present(
                path, self.brief.category or "smartphone"
            ):
                return Verdict.FAIL
        if not _same_word(self.brief.category or "", "smartphone"):
            return Verdict.PASS
        verdict = _phone_identity(self._facts(pid), self._title(pid), self._options(pid, sku))
        if verdict == "yes":
            return Verdict.PASS
        if verdict == "no":
            return Verdict.FAIL
        if self.brief.family in ("constraint", "justification", "recovery"):
            return Verdict.UNKNOWN
        return Verdict.PASS

    def _device_ok(self, pid: str, sku: str) -> bool:
        return self._device_verdict(pid, sku) != Verdict.FAIL

    def _resale_title(self, pid: str) -> bool:
        """Listings that describe someone else's listing, or a used unit, are never the answer."""
        return bool(re.search(r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed", self._title(pid), re.I))

    def _relax(self, pid: str, sku: str) -> int:
        """How far a candidate sits from the shopper's exact preferred model and color."""
        model, color = self.brief.pref_model, self.brief.pref_color
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self._title(pid) + " " + " ".join(str(v) for v in self._facts(pid).values()))
        same_model = squash(model) in blob
        shade = (self._options(pid, sku) + " " + str(self._facts(pid).get("color") or "") + " " + self._title(pid)).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def _pref_verdict(self, pid: str, sku: str, pref: Preference) -> PrefVerdict:
        viewed = str(pid) in self.ledger.products
        if pref.type == "brand":
            wanted = _tokens(str(pref.value or ""))
            if not wanted:
                return PrefVerdict.UNKNOWN
            have = _tokens(self._brand_of(pid)) or _tokens(self._title(pid))
            if wanted <= have:
                return PrefVerdict.MATCH
            return PrefVerdict.UNKNOWN if not viewed or not self._brand_of(pid) else PrefVerdict.NO_MATCH
        if pref.type == "color":
            shade = (self._options(pid, sku) + " " + str(self._facts(pid).get("color") or "") + " " + self._title(pid)).lower()
            if not str(pref.value or "").strip():
                return PrefVerdict.UNKNOWN
            if re.search(r"(?<![a-z])" + re.escape(str(pref.value).lower()) + r"(?![a-z])", shade):
                return PrefVerdict.MATCH
            return PrefVerdict.UNKNOWN if not viewed else PrefVerdict.NO_MATCH
        if pref.type == "model":
            blob = re.sub(r"[^a-z0-9]+", "", (self._title(pid) + " " + " ".join(str(v) for v in self._facts(pid).values())).lower())
            if re.sub(r"[^a-z0-9]+", "", str(pref.value or "").lower()) in blob:
                return PrefVerdict.MATCH
            return PrefVerdict.UNKNOWN if not viewed else PrefVerdict.NO_MATCH
        return PrefVerdict.UNKNOWN

    def _pref_score(self, pid: str, sku: str) -> tuple:
        """Known preference matches only; UNKNOWN never counts as NO_MATCH or MATCH."""
        hits = 0
        known = 0
        for pref in self.brief.preferences:
            verdict = self._pref_verdict(pid, sku, pref)
            if verdict == PrefVerdict.UNKNOWN:
                continue
            known += 1
            if verdict == PrefVerdict.MATCH:
                hits += 1
        hits += self._priority_hits(pid, sku)
        return (hits, known)

    def _assess(self, pid: str, sku: str) -> dict:
        """Score one variant against every hard gate. Soft prefs never affect validity."""
        fam = self.brief.family
        key = Ledger.key(pid, sku)
        st = self.ledger.checked.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.ledger.dead_keys or (str(pid) in self.ledger.dead_products and fam != "recovery")
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and not dead
        viewed = str(pid) in self.ledger.products
        field, want = self.brief.spec_field, self.brief.spec_value
        spec = 2
        hard_fail = False
        unknown = False
        if field and want:
            got = self._variant_value(pid, sku, field)
            spec = 2 if got == want else (-1 if got else 0)
            if spec == -1:
                hard_fail = True
            elif spec == 0:
                unknown = True
        if not self._brand_ok(pid):
            hard_fail = True
        device = self._device_verdict(pid, sku)
        if device == Verdict.FAIL:
            hard_fail = True
        elif device == Verdict.UNKNOWN:
            unknown = True
        cat = self._category_verdict(pid)
        cat_ok = cat != Verdict.FAIL
        if cat == Verdict.FAIL:
            hard_fail = True
        elif cat == Verdict.UNKNOWN:
            unknown = True
        if fam == "justification":
            if self._resale_title(pid) or (viewed and not self._brand_of(pid)):
                hard_fail = True
            if field and want:
                fact_hit = self._fact_value(pid, field) == want and self._product_value(pid, field) == want
                if viewed and not fact_hit:
                    hard_fail = True
        if fam == "preference" and self.ledger.event_target:
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split("::", 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                hard_fail = True
        if fam == "recall" and self.brief.theme:
            title = self._title(pid)
            if title and not verify_title_tokens(title, self.brief.theme):
                hard_fail = True
            elif not title:
                unknown = True
        if fam == "ranking":
            if (str(pid), str(sku)) not in self.brief.candidate_keys:
                hard_fail = True
            if viewed and (
                self._measure(pid, sku, "refresh_rate") is None
                or self._measure(pid, sku, "screen_size") is None
            ):
                unknown = True
        if not viewed:
            unknown = True
        if hard_fail:
            verdict = Verdict.FAIL
        elif unknown:
            verdict = Verdict.UNKNOWN
        else:
            verdict = Verdict.PASS
        ok = verdict == Verdict.PASS and live
        return {
            "pid": str(pid), "sku": str(sku), "key": key, "price": price, "live": live,
            "viewed": viewed, "spec": spec, "cat_ok": cat_ok, "ok": ok,
            "verdict": verdict, "hard_fail": hard_fail, "unknown": unknown,
            "relax": self._relax(pid, sku) if fam == "recovery" else 0,
            "pref": self._pref_score(pid, sku),
        }

    def _rank_of(self, c: dict, first: bool) -> tuple:
        """TF5: rank only hard-valid candidates. Never invent an objective."""
        fam = self.brief.family
        pref_hits, pref_known = c.get("pref") or (0, 0)
        coverage = 0 if c.get("viewed") else 1
        stable = c["key"]
        if fam == "ranking":
            hz = self._measure(c["pid"], c["sku"], "refresh_rate") or 0.0
            inch = self._measure(c["pid"], c["sku"], "screen_size") or 0.0
            price = c["price"] if c["price"] is not None else 1e12
            return (-hz, -inch, price, stable)
        if fam == "recovery":
            return (c["relax"], -pref_hits, coverage, stable)
        extra = sum(
            1 for f, v in self.brief.specs.items()
            if f != self.brief.spec_field and self._variant_value(c["pid"], c["sku"], f) != v
        )
        proven = 0 if _phone_identity(self._facts(c["pid"]), self._title(c["pid"]), self._options(c["pid"], c["sku"])) == "yes" else 1
        near = 0
        if first and self.brief.honor_pref_brand and self.brief.pref_brand:
            wanted = _tokens(self.brief.pref_brand)
            near = 0 if (wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))) else 1
        objective = self.brief.objective
        obj_key = 0.0
        if objective:
            field, direction = objective
            raw = c["price"] if field == "price" else self._measure(c["pid"], c["sku"], field)
            if raw is None:
                obj_key = 1e12 if direction == "minimize" else -1e12
            else:
                obj_key = raw if direction == "minimize" else -raw
        return (near, extra, proven, -pref_hits, -pref_known, coverage, obj_key, stable)

    def _ranking_state(self, first: bool | None = None) -> RankingState:
        if first is None:
            first = self.ledger.adds == 0
        good = [c for c in self._candidates() if c["ok"]]
        good.sort(key=lambda c: self._rank_of(c, first))
        ids = [c["key"] for c in good]
        return RankingState(ordered_candidate_ids=ids, selected_candidate_id=ids[0] if ids else None)

    def _candidates(self) -> list:
        seen = set()
        out = []
        for key in list(self.ledger.checked):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self._assess(pid, sku))
        return out

    def _target(self) -> dict | None:
        """The best fully verified candidate, or nothing while evidence is missing."""
        first = self.ledger.adds == 0
        fam = self.brief.family
        good = [c for c in self._candidates() if c["ok"]]
        if fam == "preference":
            carted = [c for c in good if (c["pid"], c["sku"]) in self.ledger.cart]
            if carted:
                return carted[0]
            if good and (self.turn >= self.max_steps - 8 or self._in_reserve()):
                good.sort(key=lambda c: self._rank_of(c, first))
                return good[0]
            return None
        if fam == "intent" and first and self.brief.pref_brand:
            wanted = _tokens(self.brief.pref_brand)
            named = [c for c in good if wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))]
            if named:
                good = named
            elif self.turn < self.max_steps * 0.6:
                self._preferred_brand_hint()
                return None
        if fam == "recovery" and first:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif self.turn < self.max_steps * 0.6:
                self._exact_match_hint()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self._rank_of(c, first))
        return good[0]

    def _answer_text(self) -> str:
        """A short, factual answer to the shopper, drawn only from observed evidence."""
        target = self._target() or (self._candidates() or [None])[0]
        if not target:
            return "I am still checking the options against your requirements; nothing is decided yet."
        pid, sku = target["pid"], target["sku"]
        parts = ["I am going with " + (self._title(pid) or pid)[:90]]
        field, want = self.brief.spec_field, self.brief.spec_value
        if field and want and self._variant_value(pid, sku, field) == want:
            parts.append("its listed " + field.replace("_", " ") + " is " + want)
        if target.get("price") is not None:
            parts.append("it is %.2f %s and in stock" % (target["price"], self.brief.currency))
        return ", ".join(parts) + "."

    def _read_more(self) -> tuple | None:
        """Turn the search results already in hand into read product pages, then widen.

        Reading pages is what turns rows into candidates the controller can verify,
        and a second verified product is what makes the market change survivable.
        """
        fam = self.brief.family
        if fam == "recall" and len(self.ledger.identities) < self._recall_need():
            return None
        twin = self._twin_search()
        if twin is not None:
            return twin
        leads = self._unseen_leads()
        if leads and self.forced_compares < 8 and self.turn >= 2:
            self.forced_compares += 1
            compare = self._tool(ToolRole.COMPARE, "compare")
            detail = self._tool(ToolRole.DETAIL, "view")
            if len(leads) > 1 and compare:
                return (compare, {"product_ids": leads[:5]})
            if detail:
                return (detail, {"product_id": leads[0]})
        probe = (self.brief.brand or "", "%.2f" % (self.budget or 0.0), str(self.brief.spec_value))
        if self.turn < 2:
            return None
        if probe in self.probes:
            self.reads_done = True
            return None
        self.probes.add(probe)
        filt = self._tool(ToolRole.FILTER, "filter")
        search = self._tool(ToolRole.SEARCH, "search")
        if filt and self.brief.category:
            args = {"category": self.brief.category, "in_stock": True, "k": 10}
            if self.budget is not None:
                args["max_price"] = self.budget
            if self.brief.brand:
                args["brand"] = self.brief.brand
            return (filt, args)
        if search:
            words = [w for w in (self.brief.brand, self.brief.spec_value, self.brief.category) if w]
            if words:
                return (search, {"query": sanitize_search_query(" ".join(str(w) for w in words)), "in_stock": True, "k": 10})
        self.reads_done = True
        return None

    def _twin_search(self) -> tuple | None:
        """After the market kills the carted item, its own twins are the nearest replacements.

        A listing's other colors and sizes are separate products carrying the same
        title, so searching that title surfaces exactly the items that still meet
        every requirement the dead one met.
        """
        dead = self.ledger.event_target
        if self.searched_twins or not dead or not self._tool(ToolRole.SEARCH, "search"):
            return None
        title = self._title(dead.split("::", 1)[0])
        if not title:
            return None
        self.searched_twins = True
        self._note(
            "The item that died is one of several listings sharing a title; searching that title "
            "surfaces its other colors and sizes, which meet the same requirements."
        )
        search = self._tool(ToolRole.SEARCH, "search")
        return (search, {"query": sanitize_search_query(" ".join(title.split()[:8])), "k": 10, "in_stock": True})

    def _preferred_brand_hint(self) -> None:
        if self.hinted_brand:
            return
        self.hinted_brand = True
        self._note(
            "No %s product that meets every firm requirement has been read yet. The shopper's first choice is "
            "graded on that brand, so filter by brand=%s within the budget and read those products before any "
            "other." % (self.brief.pref_brand, self.brief.pref_brand)
        )

    def _sibling_search(self) -> tuple | None:
        """A found model in the wrong color: its other colors share the same title."""
        if self.searched_siblings:
            return None
        rows = [c for c in self._candidates() if c["live"] and c["viewed"] and c["relax"] == 0]
        if rows:
            return None
        near = [c for c in self._candidates() if c["viewed"] and c["relax"] == 1]
        if not near:
            return None
        title = self._title(near[0]["pid"])
        if not title:
            return None
        self.searched_siblings = True
        search = self._tool(ToolRole.SEARCH, "search")
        if not search:
            return None
        return (search, {"query": sanitize_search_query(" ".join(title.split()[:8])), "k": 10, "in_stock": True})

    def _exact_match_hint(self) -> None:
        if self.hinted_exact or not self.brief.pref_model:
            return
        self.hinted_exact = True
        self._note(
            "The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own "
            "and view the results so every color variant is on record; only that exact pairing may be the "
            "first cart item." % (self.brief.pref_model, ", " + self.brief.pref_color if self.brief.pref_color else "")
        )

    def _unseen_leads(self) -> list:
        """Promising variants whose product page has not been read yet."""
        leads = []
        for key, row in self.ledger.rows.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.ledger.products or str(pid) in self.ledger.dead_products:
                continue
            st = self.ledger.checked.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.budget is not None and price > self.budget + _EPS):
                continue
            title = str(row.get("title") or "")
            if self.brief.theme and not all(t in _tokens(title) for t in self.brief.theme):
                continue
            if self.brief.pref_model and self.ledger.adds == 0 and not self.ledger.event_target:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.brief.pref_model.lower()) not in squashed:
                    continue
            if self.brief.brand:
                wanted = _tokens(self.brief.brand)
                if not (wanted <= _tokens(str(row.get("brand") or "")) or wanted <= _tokens(title)):
                    continue
            leads.append((price, str(pid)))
        leads.sort()
        out = []
        for _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    # ------------------------------------------------------------------ guidance
    def _shortlist_note(self) -> None:
        """Tell the model what the controller can and cannot yet verify."""
        cands = self._candidates()
        if self.brief.family == "preference" and not self.ledger.cart:
            good = [c for c in cands if c["ok"]]
            good.sort(key=lambda c: self._rank_of(c, True))
            if good:
                self._note(
                    "Products that meet every firm requirement so far: "
                    + "; ".join("%s::%s at %.2f" % (c["pid"], c["sku"], c["price"] or 0.0) for c in good[:6])
                    + ". Pick the one that best fits the shopper's stated priorities, inspect it, then add it."
                )
                return
        best = self._target()
        if best:
            self._note(
                "Verified candidate: %s::%s at %.2f. The controller will inspect, cart and order it; "
                "read other candidates only if one could rank better."
                % (best["pid"], best["sku"], best["price"] or 0.0)
            )
            return
        field, want = self.brief.spec_field, self.brief.spec_value
        blocked = [c for c in cands if c["viewed"] and c["live"] and not c["ok"]]
        if blocked and field:
            sample = blocked[:3]
            self._note(
                "None of the products read so far can be verified. For each, the evidence for %s must "
                "resolve to exactly %s from the variant's own option text or the product's raw facts: %s. "
                "Look at different products." % (field, want, ", ".join(c["pid"] + "::" + c["sku"] for c in sample))
            )
        elif blocked:
            self._note(
                "The products read so far fail a firm requirement (category, brand, stock, budget or the "
                "kind of device requested). Search for different products."
            )

    def _priority_hits(self, pid: str, sku: str) -> int:
        """How much of what the shopper actually asked for this product's own facts mention."""
        wants = self._wish_words()
        if not wants:
            return 0
        have = _tokens(" ".join(str(v) for v in self._facts(pid).values()) + " " + self._title(pid) + " " + self._options(pid, sku))
        return len(wants & have)

    def _wish_words(self) -> set:
        """Content words from the shopper's own wording, gathered once."""
        if self._wishes is None:
            said = self.brief.goal + " " + " ".join(
                str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
            )
            drop = _tokens(self.brief.category or "") | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
            }
            self._wishes = {w for w in _tokens(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._wishes

    def _budget_from_message(self, text: str) -> None:
        self._apply_shopper_text(text)

    def _apply_shopper_text(self, text: str) -> None:
        """TF6 shopper update: replace only changed requirements, then re-evaluate."""
        changes = self.brief.apply_shopper_update(text)
        if not changes:
            return
        self.shopper_updated = True
        self.ledger.last_failure = FailureType.SHOPPER_UPDATE
        self.budget = self.brief.budget
        self._note("Shopper update applied: " + ", ".join(changes) + ". Unchanged requirements still hold.")
        for pid, sku in list(self.ledger.cart):
            c = self._assess(pid, sku)
            if not c["ok"]:
                self.ledger.dead_keys.add(c["key"])
                self.ledger.last_failure = (
                    FailureType.PRICE_CHANGED if c["live"] is False and c.get("price") is not None
                    else FailureType.CONSTRAINT_FAILURE
                )
                self._note(
                    "The carted item %s no longer meets the active requirements; the controller will replace it."
                    % c["key"]
                )

    def _claims_for(self, pid: str, sku: str) -> list:
        """Claims proven by what the agent has already observed about this exact item."""
        out = []
        field, want = self.brief.spec_field, self.brief.spec_value
        if not field and self.brief.family == "justification":
            field, want = "resolution", self._fact_value(pid, "resolution")
        if not field or not want:
            return []
        stated = dict(self.brief.specs) or {field: want}
        stated.setdefault(field, want)
        for name, value in stated.items():
            if self._fact_value(pid, name) == value and self._product_value(pid, name) == value:
                out.append({"field": name, "value": value})
        if not out:
            return []  # an unprovable claim is worse than none at all
        st = self.ledger.status(pid, sku) or {}
        live_price = None
        auth = self.ledger.latest_fact(pid, "price", authoritative=True)
        if auth is not None and auth.value is not None:
            live_price = _num(auth.value)
        if live_price is None:
            live_price = st.get("price")
        if live_price is not None:
            out.append({"field": "price", "value": "%.2f" % live_price})
        out.append({"field": "in_stock", "value": "true"})
        brand = self._brand_of(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out

    def _order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.brief.quantity and self.brief.quantity > 1:
            out["quantity"] = self.brief.quantity
        if self.brief.needs_justification or self.brief.family == "justification":
            claims = self._claims_for(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def _valid_now(self, pid: str, sku: str) -> tuple:
        st = self.ledger.status(pid, sku)
        k = Ledger.key(pid, sku)
        if st is None:
            return False, "inspect current price/stock for the exact product_id and sku first"
        if (pid, sku) not in self.ledger.cart:
            return False, "add that exact variant to the cart first"
        if (
            self._tool(ToolRole.CURRENT_STATE, "inspect_stock")
            and self.ledger.last_add_seq is not None
            and self.ledger.checked_seq.get(k, 0) <= self.ledger.last_add_seq
            and not (st.get("auth") and self.ledger.last_market_event)
        ):
            return False, "re-check the variant with current-state evidence after the cart addition"
        if not st.get("in_stock"):
            return False, "that variant is out of stock; choose a different product"
        price = st.get("price")
        if self.budget is not None and price is not None and price > self.budget + _EPS:
            return False, "that variant costs %.2f, above the budget of %.2f" % (price, self.budget)
        if k in self.ledger.dead_keys:
            return False, "that variant was invalidated by the market change; choose a different product"
        return True, ""

    # ------------------------------------------------------------------ planning
    def _recall_need(self) -> int:
        """How many distinct matching products to surface before buying.

        Surfacing more than the stated minimum is worth turns early on, but the
        catalog may simply not hold that many, so the margin lapses.
        """
        need = self.brief.required_identities
        if self.brief.coverage and self.brief.coverage.kind == "product_identity":
            need = max(need, self.brief.coverage.minimum)
        early = self.search_calls < 8 and self.turn < self.max_steps * 0.5
        return need + (2 if early else 0)

    def _ready(self) -> bool:
        """Whether a first commitment can be made, or the shopper still owes information."""
        fam = self.brief.family
        if self.ledger.adds:
            return True
        if self.brief.spec_sealed:
            if self.turn < self.max_steps * 0.6:
                return False
            self.brief.spec_sealed = False  # it was never stated; buy the best of the rest
            self._note("The firm requirement was never stated, so the remaining requirements decide the choice.")
        if fam in ("intent", "constraint") and self.turn < 4 and not self.shopper_updated:
            return False  # a later message can still tighten the budget
        if fam == "preference" and len(self.ledger.listing_ids) < 2:
            return False
        roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
        if fam == "recall" and roomy:
            # The score counts how much of the market the search surfaced, so a second
            # differently worded search is worth its turn.
            if self._identity_count() < self._recall_need() or self.search_calls < 2:
                return False
        if roomy and not self.reads_done and len({c["pid"] for c in self._candidates() if c["ok"]}) < 2:
            # The market moves on the first cart action, so line up a fallback before
            # committing: hunting for one afterwards costs many extra turns. Once no
            # unread candidate remains, waiting only burns turns.
            return False
        return True

    def _fresh(self, pid: str, sku: str) -> bool:
        k = Ledger.key(pid, sku)
        if k not in self.ledger.inspected:
            return False
        return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq

    def _plan(self) -> tuple | None:
        """The action the controller takes itself, or nothing to let the model explore."""
        if self.done:
            return None
        if self.shopper_question_pending:
            msg = self._tool(ToolRole.MESSAGE, "message")
            if self.question_waits < 2 or not msg:
                self.question_waits += 1
                return None
            return (msg, {"content": self._answer_text()})
        fam = self.brief.family
        compare = self._tool(ToolRole.COMPARE, "compare")
        if fam == "ranking" and compare:
            missing = [p for p, _s in self.brief.candidate_keys
                       if p not in self.ledger.products and p not in self.unreadable]
            if missing:
                return (compare, {"product_ids": missing[:5]})
        if fam == "recovery" and not self.ledger.adds and self.brief.pref_model and self._tool(ToolRole.SEARCH, "search"):
            forced = self._sibling_search()
            if forced is not None:
                return forced
        target = self._target()
        if (target is None or not self._ready()) and not self._in_reserve():
            forced = self._read_more()
            if forced is not None:
                return forced
            if target is None:
                return None
        if target is None:
            return None
        pid, sku = target["pid"], target["sku"]
        current = self._tool(ToolRole.CURRENT_STATE, "inspect_stock")
        finalize = self._tool(ToolRole.FINALIZE, "place_test_order")
        cart_add = self._cart_add_name()
        remove = self._cart_remove_name()
        if (pid, sku) in self.ledger.cart:
            if not self._fresh(pid, sku) and current:
                return (current, {"product_id": pid, "sku": sku})
            if finalize:
                return (finalize, self._order_args(pid, sku))
            return None
        if not self._ready() and not self._in_reserve():
            return None
        for cpid, csku in self.ledger.cart:
            if Ledger.key(cpid, csku) in self.ledger.dead_keys and remove:
                return (remove, {"product_id": cpid, "sku": csku})
        if not self._fresh(pid, sku) and current:
            return (current, {"product_id": pid, "sku": sku})
        if cart_add:
            args = {"product_id": pid, "sku": sku}
            if self.brief.quantity and self.brief.quantity > 1:
                args["quantity"] = self.brief.quantity
            return (cart_add, args)
        return None

    # ------------------------------------------------------------------ inference
    def _ask_model(self) -> dict | None:
        for attempt in range(3):
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            payload = {
                "model": model,
                "messages": self.messages,
                "tools": self.tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            resp = _PROXY.post("/inference/chat/completions", json_data=payload)
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                self._note("No tool call was made. Choose exactly one of the supplied tools now.")
                continue
            # The proxy applies backpressure; back off rather than hammering it.
            if self.clock.remaining() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    # ------------------------------------------------------------------ environment
    def _send(self, envelope: dict) -> dict | None:
        """POST one solver-turn envelope; an identical envelope replays idempotently."""
        for attempt in range(4):
            result = _PROXY.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.clock.remaining() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def _act(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        args = self._fit(name, args)
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self._send(pend["envelope"])
            if replay is None:
                self._note("The environment is not answering; the previous action is still pending and will be retried.")
                return None
            self.pending_envelope = None
            self.turn = pend["turn"]
            self._absorb(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
            if self.done:
                return None
        turn = self.turn + 1
        group = self.problem_id + "-turn-" + str(turn)
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [{"call_id": group + "-1", "action": {"name": name, "args": args}}],
        }
        result = self._send(envelope)
        if result is None:
            self.pending_envelope = {
                "envelope": envelope, "turn": turn, "name": name, "args": args,
                "content": content, "call_id": tool_call_id or ("agent-" + str(turn)),
            }
        else:
            self.turn = turn
        return self._absorb(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.ledger.calls_seen.discard(name + json.dumps(args, sort_keys=True))
            self.messages.append(
                {"role": "tool", "tool_call_id": call_id,
                 "content": json.dumps({"error": "environment call failed; it will be retried automatically"})}
            )
            return None
        calls = result.get("calls") or []
        wrap = (calls[0].get("observation") if calls and isinstance(calls[0], dict) else None) or {}
        obs = wrap.get("observation") if isinstance(wrap.get("observation"), dict) else wrap
        if isinstance(obs, dict) and wrap.get("error") and not obs.get("error"):
            obs = dict(obs)
            obs["error"] = wrap.get("error")
        if wrap.get("done"):
            self.done = True
        self.ledger.record(name, args, obs, self.brief, self.tool_roles.get(name))
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _trim(json.dumps(obs), _OBS_CHARS)})
        self._note_terminal(result, name, obs)
        text = extract_shopper_update(result)
        if text:
            self.messages.append({"role": "user", "content": text})
            self._apply_shopper_text(text)
            if _QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self._note("The shopper asked a question: answer it with message in your next turn, then continue.")
        if name in ("add_to_cart",) or self.tool_roles.get(name) == ToolRole.CART:
            event = extract_market_event(result) or extract_market_event(obs)
            live = obs if isinstance(obs, dict) else {}
            if event:
                live = dict(live)
                live.setdefault("market_event", event)
                self.ledger._apply_market_event(live, name)
            self._invalidate_from_live_cart(live)
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        return obs

    def _note_terminal(self, result: dict | None, name: str, obs: dict | None) -> None:
        if not isinstance(result, dict):
            return
        wrap = ((result.get("calls") or [{}])[0] or {}).get("observation") if result.get("calls") else result
        payload = wrap if isinstance(wrap, dict) else {}
        done = bool(payload.get("done") or (isinstance(obs, dict) and obs.get("done")))
        if done:
            self.done = True
        if self.tool_roles.get(name) == ToolRole.FINALIZE or name == "place_test_order":
            self.order_outcome = {
                "done": done,
                "error": payload.get("error") or (obs or {}).get("error") if isinstance(obs, dict) else None,
                "outcome": payload.get("outcome") or (obs or {}).get("outcome") if isinstance(obs, dict) else None,
                "verdict": payload.get("verdict") or (obs or {}).get("verdict") if isinstance(obs, dict) else None,
            }
            self._note(
                "Finalize observation: done=%s error=%s outcome=%s"
                % (done, self.order_outcome.get("error"), self.order_outcome.get("outcome"))
            )
            if not done:
                self._note("place_test_order returned without terminal state; the session is not complete.")

    def _invalidate_from_live_cart(self, obs: dict | None) -> None:
        """Cart-mutating observations beat any earlier inspect_stock snapshot."""
        if not isinstance(obs, dict):
            return
        event = extract_market_event(obs)
        added = obs.get("added") or {}
        pid = str(added.get("product_id") or (event or {}).get("product_id") or "")
        sku = str(added.get("sku") or (event or {}).get("sku") or "")
        if not pid:
            return
        key = Ledger.key(pid, sku)
        st = self.ledger.checked.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        if stock is False:
            self.ledger.dead_keys.add(key)
            self.ledger.dead_products.add(pid)
            self.ledger.last_failure = FailureType.OUT_OF_STOCK
            self._note("Live cart state: %s is out of stock after the cart mutation." % key)
        elif self.budget is not None and price is not None and price > self.budget + _EPS:
            self.ledger.dead_keys.add(key)
            self.ledger.dead_products.add(pid)
            self.ledger.last_failure = FailureType.PRICE_CHANGED
            self._note(
                "Live cart state: %s is now %.2f, above the budget of %.2f. Previous inspect_stock is stale."
                % (key, price, self.budget)
            )

    # ------------------------------------------------------------------ controller steps
    def _post_add_recheck(self) -> None:
        """Re-validate from live cart state, then current-state evidence, then recover."""
        pid, sku = self.pending_recheck
        self.pending_recheck = None
        key = Ledger.key(pid, sku)
        if key in self.ledger.dead_keys:
            self._begin_recovery(pid, sku, "already invalidated by the cart observation")
            return
        current = self._tool(ToolRole.CURRENT_STATE, "inspect_stock")
        obs = self._act(current, {"product_id": pid, "sku": sku}) if current else None
        st = self.ledger.checked.get(key) or {}
        stock = bool(st.get("in_stock")) if st else (bool(obs.get("in_stock")) if isinstance(obs, dict) else False)
        price = st.get("price") if st else (_num(obs.get("price")) if isinstance(obs, dict) else None)
        if isinstance(obs, dict) and not obs.get("error"):
            stock = bool(obs.get("in_stock"))
            price = _num(obs.get("price"))
        over = self.budget is not None and price is not None and price > self.budget + _EPS
        if stock and not over:
            self._note(
                "Re-check after the cart addition: %s is in stock at %s, still valid. The controller will order it."
                % (key, price)
            )
            return
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        self.ledger.last_failure = FailureType.OUT_OF_STOCK if not stock else FailureType.PRICE_CHANGED
        self._begin_recovery(pid, sku, reason)

    def _begin_recovery(self, pid: str, sku: str, reason: str) -> None:
        self.ledger.event_target = Ledger.key(pid, sku)
        self.event_turn = self.turn
        self.ledger.dead_keys.add(Ledger.key(pid, sku))
        self.ledger.dead_products.add(str(pid))
        self.forced_compares = 0
        guidance = {
            "recovery": "Recover with the smallest possible change: the same model in another color before any other model.",
            "ranking": "The remaining named candidates are re-ranked; the controller takes the new best.",
            "preference": "Recover to a product with a different listing_id that still fits the shopper's priorities.",
            "constraint": "Choose a different product that still meets every requirement, not another variant of the same one.",
        }.get(self.brief.family, "Choose a different product that meets every firm requirement.")
        self._note("Market change: the carted item %s %s. " % (Ledger.key(pid, sku), reason) + guidance)
        self._shortlist_note()

    def _finish_fast(self) -> None:
        """Out of time or turns: reach a valid order in as few actions as possible."""
        for pid, sku in reversed(self.ledger.cart):
            ok, _why = self._valid_now(pid, sku)
            if ok and not self.done:
                finalize = self._tool(ToolRole.FINALIZE, "place_test_order")
                if finalize:
                    self._act(finalize, self._order_args(pid, sku))
                return
        if self.done or self.turn > self.max_steps - 3:
            return
        good = [c for c in self._candidates() if c["ok"] and c.get("verdict") == Verdict.PASS]
        if not good:
            return
        good.sort(key=lambda c: self._rank_of(c, self.ledger.adds == 0))
        pid, sku = good[0]["pid"], good[0]["sku"]
        add = self._cart_add_name()
        current = self._tool(ToolRole.CURRENT_STATE, "inspect_stock")
        finalize = self._tool(ToolRole.FINALIZE, "place_test_order")
        if (pid, sku) not in self.ledger.cart and add:
            obs = self._act(add, {"product_id": pid, "sku": sku})
            if self.done or not isinstance(obs, dict) or obs.get("error"):
                return
            self._invalidate_from_live_cart(obs)
            if Ledger.key(pid, sku) in self.ledger.dead_keys:
                return
        if current:
            obs = self._act(current, {"product_id": pid, "sku": sku})
            if self.done or not isinstance(obs, dict) or not obs.get("in_stock"):
                return
            price = _num(obs.get("price"))
            if self.budget is not None and price is not None and price > self.budget + _EPS:
                return
        elif not self._fresh(pid, sku):
            return
        if finalize:
            self._act(finalize, self._order_args(pid, sku))

    # ------------------------------------------------------------------ main loop
    def run(self) -> list:
        question = self._opening_question()
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            if self.clock.remaining() < 25 or self.turn >= self.max_steps - 3:
                self._finish_fast()
                break
            if question and not self.asked_shopper and self._tool(ToolRole.MESSAGE, "message"):
                self.asked_shopper = True
                self._act(self._tool(ToolRole.MESSAGE, "message"), {"content": question})
                continue
            if self.pending_recheck:
                if self._tool(ToolRole.CURRENT_STATE, "inspect_stock") or self.ledger.last_market_event:
                    self._post_add_recheck()
                    continue
            plan = self._plan()
            if plan is not None:
                name, args = plan
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    # The step is not moving the state on; abandon that candidate.
                    if name in ("inspect_stock", "add_to_cart"):
                        self.ledger.dead_keys.add(Ledger.key(args.get("product_id"), args.get("sku")))
                    if name == "compare":
                        self.unreadable.update(str(p) for p in args.get("product_ids") or [])
                    if name == "message":
                        self.shopper_question_pending = False
                    last_forced, repeats = None, 0
                    self._note("That step changed nothing, so the controller moved on to other candidates.")
                    continue
                obs = self._act(name, args)
                if name == "message":
                    self.shopper_question_pending = False
                    self.question_waits = 0
                if self._is_add(name) and isinstance(obs, dict) and not obs.get("error"):
                    added = obs.get("added") or {}
                    if added.get("product_id"):
                        self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                continue
            last_forced, repeats = None, 0
            proposal = self._ask_model()
            if proposal is None:
                self._finish_fast()
                break
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = _arguments(call)
            content = proposal.get("content") or ""
            reprompts = 0
            while True:
                verdict = self._validate(name, args)
                if verdict is None:
                    break
                if reprompts >= _MAX_REPROMPTS:
                    self.rejected_streak += 1
                    self._note("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if self.rejected_streak >= 3:
                        self._shortlist_note()
                        self.rejected_streak = 0
                    verdict = "skip"
                    break
                reprompts += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": verdict})}
                )
                self.dialogue.append({"role": "assistant", "content": "[rejected " + name + "] " + verdict[:200]})
                again = self._ask_model()
                if again is None:
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = _arguments(call)
                content = again.get("content") or ""
            if verdict == "skip":
                continue
            self.rejected_streak = 0
            if name not in self.tool_names:
                continue
            if name == "place_test_order" or self.tool_roles.get(name) == ToolRole.FINALIZE:
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.ledger.cart:
                    pid, sku = self.ledger.cart[-1]
                args = self._order_args(pid, sku)
            if name == "message":
                if not self.shopper_question_pending:
                    self.questions_asked += 1
                self.shopper_question_pending = False
                self.question_waits = 0
            obs = self._act(name, args, content, call.get("id"))
            if self._is_add(name) and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") or {}
                if added.get("product_id"):
                    self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
            if self.tool_roles.get(name) in (ToolRole.SEARCH, ToolRole.FILTER) or name in ("search", "filter"):
                self.search_calls += 1
                if self.brief.family == "recall" and self.brief.required_identities:
                    have, need = self._identity_count(), self._recall_need()
                    self._note(
                        "Distinct matching product identities surfaced so far: %d of the required %d." % (have, need)
                        + ("" if have >= need else " Keep searching with different wording.")
                    )
            if name == "inspect_cart":
                self.cart_inspections += 1
            if name in ("view", "compare") or self.tool_roles.get(name) in (ToolRole.DETAIL, ToolRole.COMPARE):
                self._shortlist_note()
        return self.dialogue

    def _validate(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
        if name == "place_test_order" or self.tool_roles.get(name) == ToolRole.FINALIZE:
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.ledger.cart:
                pid, sku = self.ledger.cart[-1]
            ok, why = self._valid_now(pid, sku)
            if not ok:
                return "order refused: " + why
            if self.brief.family == "recall" and self._identity_count() < self._recall_need():
                return (
                    "order refused: only %d distinct matching product identities have been surfaced; search with different "
                    "wording until at least %d appear" % (self._identity_count(), self._recall_need())
                )
            listing_need = 2
            if self.brief.coverage and self.brief.coverage.kind == "listing_id":
                listing_need = self.brief.coverage.minimum
            if self.brief.family == "preference" and len(self.ledger.listing_ids) < listing_need:
                return "order refused: compare at least two products with different listing_id values first"
        if self._is_add(name):
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            c = self._assess(pid, sku)
            key = Ledger.key(pid, sku)
            if c.get("verdict") == Verdict.FAIL or c.get("hard_fail"):
                return "that product fails a firm requirement (category, brand, or the kind of device requested)"
            if not c["live"]:
                return "that variant is out of stock, over budget, or already invalidated"
            if not c["viewed"]:
                return "view or compare that product first so its raw facts are on record"
            if key not in self.ledger.inspected and self._tool(ToolRole.CURRENT_STATE, "inspect_stock"):
                return "inspect current price/stock for that exact variant first: retrieval metadata is not final"
            if c["spec"] != 2 and self.brief.spec_field:
                return (
                    "that variant does not resolve to %s = %s from its own option text or raw facts"
                    % (self.brief.spec_field, self.brief.spec_value)
                )
            if c.get("verdict") == Verdict.UNKNOWN:
                return "missing evidence for a firm requirement; gather more facts rather than treating it as a failure"
            if not c["ok"]:
                return "that product fails a firm requirement (category, brand, or the kind of device requested)"
            if self.brief.family == "preference" and len(self.ledger.listing_ids) < 2:
                return "compare two products with different listing_id values before the first cart addition"
            if self.brief.family == "recovery" and not self.ledger.adds and self.brief.pref_model and self._relax(pid, sku):
                return (
                    "the first cart item must be the exact model and color the shopper named; the other colors are "
                    "separate listings sharing the same title, so search that title and read them first"
                )
            if not self._ready():
                return (
                    "hold: read one more product that also meets every requirement first, so there is a "
                    "fallback when the market moves right after the cart addition"
                )
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.shopper_question_pending and "?" not in text:
                return "do not send status updates; use message only to answer the shopper's question or to ask one"
            if not self.shopper_question_pending and self.questions_asked >= 2:
                return (
                    "the shopper has already answered; decide from the evidence you have rather than asking again"
                )
        if name == "inspect_cart" and self.cart_inspections >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.ledger.calls_seen and (
            name in ("search", "filter", "view", "compare")
            or self.tool_roles.get(name) in (ToolRole.SEARCH, ToolRole.FILTER, ToolRole.DETAIL, ToolRole.COMPARE)
        ):
            return "identical call already made; change the arguments"
        self.ledger.calls_seen.add(key)
        return None


def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = [
    "agent_main",
]
