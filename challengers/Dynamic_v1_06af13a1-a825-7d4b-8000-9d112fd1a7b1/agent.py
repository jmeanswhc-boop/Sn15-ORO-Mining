
"""
Dynamic Shopping Agent - Single File
=====================================

Workflow (one phase pipeline, the same strategy for every task):
1. Analyze the shopper request: items, hard requirements, budget, pick rule,
   message / step budgets, decline rules and the role of every world tool.
2. Create and execute a multi-step search plan (with automatic paging).
3. Collect tool observations into a candidate ledger keyed by product ID.
4. Screen candidates, open every plausible one in full, then ask the
   shopper one focused question per message about properties they may hold
   back.  After the questions only a small, targeted search runs.
5. Decide: judge every opened candidate against all requirements and pick
   the right one (cheapest / extreme / any) - or decline when nothing fits.
6. Execute cart and purchase steps, react to store notices (sold out, price
   or budget changes, refused checkout) by swapping and checking out again.

Adapters:
    llm_step(messages, tools, role, max_tokens)
    tool_executor(name, arguments)

All adapters are asynchronous.  The agent always finishes the episode with an
order or a decline while it still has steps.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

from src.agent.proxy_client import ProxyClient


# ============================================================
# 1. CONFIGURATION
# ============================================================

MAX_AGENT_STEPS = 60          # fallback when the world states no step limit
MAX_SEARCH_ROUNDS = 3         # planned search rounds before giving up
MAX_REFINEMENT_ROUNDS = 6     # detail batches opened while looking for a fit
MAX_SHOPPER_INTERACTIONS = 5  # never send more questions than this
MAX_TOOL_CALLS_PER_LOOP = 4   # search calls accepted from one plan
MAX_CANDIDATES = 250
MAX_ERRORS = 30
MAX_EXTRA_PAGES = 2           # follow-up pages read per search call
DETAIL_BATCH = 3              # listings opened per refinement round
MAX_SWAPS = 4                 # replacements after sell-outs / refusals
FINISH_RESERVE = 3            # steps kept for cart + checkout
STOCKOUT_RESERVE = 3          # extra steps kept, once a pick exists, to replace it after a sell-out
MAX_MAP_CALLS = 8             # model calls that learn observation layouts per task
DECLINE_RESERVE = 4           # extra steps kept while nothing fits, in case the decline is refused
EXTRA_SWAP_RESERVE = 3        # extra steps kept while researching, so a second sell-out can be replaced
POST_QUESTION_STEPS = 5       # extra steps kept while researching, for the small search after the questions
AFTER_SEARCH_RESERVE = 7      # steps the small search must leave: cart + checkout + one replacement
MAX_LLM_CALLS = 45
COST_LIMIT = 1.00             # USD of inference per task
TASK_SECONDS = 300.0
EXPLORE_UNTIL = 205.0         # after this, stop exploring and finish
LAST_MODEL_CALL = 250.0       # after this, finish without the model
LLM_TIMEOUT = 30              # a stuck call costs 30 s, not 60 s
LLM_HARD_STOP = 270.0         # no model request starts after this
MAX_DECLINE_REFUSALS = 2      # re-plans after the store refuses a decline
RATE_LIMIT_BASE = 4.0         # first wait after a 429, doubled each time (seconds)
RATE_LIMIT_MAX_DELAY = 20.0   # longest single wait after a 429
RATE_LIMIT_CALL_WAIT = 45.0   # one model call waits this long at most, then the fallback answers it
RATE_LIMIT_TASK_WAIT = 90.0   # a whole task waits this long at most; it must still finish in time
RESCUE_BATCH = 6              # screened-out rows given a second look
WIDEN_SHARE = 0.35            # share of the spare steps spent on more result rows before the questions
REFINE_BELOW = 4              # fewer plausible rows than this triggers the search refinement
CHEAPER_ROUNDS = 2            # times to look below the pick's price while the pick keeps getting cheaper
AUDIT_LISTINGS = 6            # rejected listings given a second opinion before a decline
APPROVAL_NEEDS_YES = False    # True: a non-answer ("no opinion") also blocks the order
CLIP_CHARS = 400
# Hidden reasoning: "low" keeps DeepSeek from reasoning until the token limit
# (17% of replies came back empty that way); an empty reply is retried with
# reasoning off.  Dropped automatically if the proxy refuses the field.
REASONING_DEFAULT = {"effort": "low"}
REASONING_OFF = {"enabled": False}

_DEFAULT_MODELS = {
    "reasoning": "deepseek/deepseek-v4.1-flash",
    "agent": "deepseek/deepseek-v4.1-flash",
}
# Opus 5.5 rejects temperature, forced tool choice and switching reasoning off.
CLAUDE_PREFIX = "anthropic/"
CLAUDE_MIN_TOKENS = 4096      # its reasoning counts toward max_tokens
# Fallbacks are tried only when a model errors (allowed: inference fallback).
_FALLBACK_MODELS = [
    "deepseek/deepseek-v4.1-flash",
    "deepseek/deepseek-v4.1-flash",
]

_proxy = ProxyClient(timeout=LLM_TIMEOUT, max_retries=1)
_env_proxy = ProxyClient(timeout=90, max_retries=2)

logger = logging.getLogger("shopping_agent")


class BudgetExhausted(RuntimeError):
    """Raised when model calls, cost or time no longer allow thinking."""


# ============================================================
# 2. AGENT STATES
# ============================================================

class AgentState(str, Enum):

    ANALYZE = "analyze"
    PLAN = "plan"
    SEARCH = "search"
    ASK_SHOPPER = "ask_shopper"
    EVALUATE = "evaluate"
    PURCHASE = "purchase"
    DECLINE = "decline"
    FINISHED = "finished"
    FAILED = "failed"


# ============================================================
# 3. DATA STRUCTURES
# ============================================================

@dataclass
class ProductCandidate:

    candidate_id: str

    title: str

    item_index: int = 0

    brand: str | None = None

    category: str | None = None

    price: float | None = None

    seller: str | None = None

    description: str | None = None

    in_stock: bool | None = None

    attributes: dict = field(default_factory=dict)

    source: str | None = None

    raw_data: dict = field(default_factory=dict)

    screened: bool | None = None       # None: not screened yet

    detail_loaded: bool = False

    fits: bool | None = None           # verdict against all requirements

    measure: float | None = None       # for "most"/"least" pick rules

    unavailable: bool = False

    rescued: bool = False              # screened out once, given a second look

    value_hits: int = 0                # required values named by a keyword search that returned it

    rejection_reasons: list = field(default_factory=list)


@dataclass
class ShoppingItem:

    name: str

    category_hint: str = ""

    quantity: int = 1

    requirements: list = field(default_factory=list)

    preferences: list = field(default_factory=list)

    dont_mind: list = field(default_factory=list)

    chosen_id: str | None = None

    kind_labels: list | None = None    # store categories that are this kind; [] = no check


@dataclass
class ShoppingContext:

    request_id: str

    original_request: str

    state: AgentState = AgentState.ANALYZE

    analysis: dict = field(default_factory=dict)

    roles: dict = field(default_factory=dict)

    items: list[ShoppingItem] = field(default_factory=list)

    budget: float | None = None

    budget_scope: str = "total"

    selection: str = "any"

    selection_measure: str = ""

    in_stock_only: bool = True

    message_limit: int = 0

    messages_sent: int = 0

    approval_messages: int = 0

    approval_steps: list = field(default_factory=list)

    approval_status: str = ""      # "", "approved", "refused" or "no_answer"

    approval_lines: dict = field(default_factory=dict)   # the order the status is about

    quiet_decline: bool = True

    decline_note: str = ""

    plan: list = field(default_factory=list)

    candidates: dict[str, ProductCandidate] = field(default_factory=dict)

    product_ids: list = field(default_factory=list)

    shopper_answers: list = field(default_factory=list)

    search_history: list = field(default_factory=list)

    notices: list = field(default_factory=list)

    cart: dict = field(default_factory=dict)

    cart_result: Any = None

    cart_observations: list = field(default_factory=list)

    tool_observations: list = field(default_factory=list)

    order_result: dict = field(default_factory=dict)

    episode_finished: bool = False

    finish_kind: str | None = None

    errors: list = field(default_factory=list)

    step_count: int = 0

    step_limit: int = MAX_AGENT_STEPS

    swaps: int = 0

    requirements_version: int = 0

    cap_ignored: set = field(default_factory=set)   # search modes that ignore the price cap

    decline_refusals: int = 0
    cart_dirty: bool = True        # the cart changed since it was last viewed
    search_closed: bool = False    # the first search's candidates are the final pool
    details_closed: bool = False   # every detail was read before the interview
    unread_notes: list = field(default_factory=list)   # store notes that came with research replies
    refined: bool = False          # the one search refinement has been spent
    audited: bool = False          # AUDIT has challenged the decline
    cross_requirements: list = field(default_factory=list)   # requirements relating the items to each other
    trail_lines: dict = field(default_factory=dict)          # the order TRAIL last passed
    checks: list = field(default_factory=list)               # CHECK / AUDIT / TRAIL / GATE / SWAP results


# ============================================================
# 4. TOOL REGISTRY
# ============================================================

ROLE_NAMES = ["search", "details", "listing", "compare", "cart_edit", "cart_view",
              "checkout", "decline", "message"]


class DynamicToolRegistry:
    """
    Supports variable tool-call sets.

    Accepts common function-tool schemas:

    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "...",
            "parameters": {}
        }
    }

    Also accepts simplified schemas:

    {
        "name": "search_products",
        "description": "...",
        "parameters": {}
    }
    """

    def __init__(self, tool_definitions: list[dict], store_schema: "StoreSchema"):

        self.tools = {}
        self.store_schema = store_schema

        for definition in tool_definitions or []:

            if not isinstance(definition, dict):
                continue

            name = self.get_tool_name(definition)

            if not name:
                continue

            self.tools[name] = definition

    @staticmethod
    def get_tool_name(tool: dict) -> str | None:

        if "name" in tool and isinstance(tool["name"], str):
            return tool["name"]

        return (tool.get("function") or {}).get("name")

    @staticmethod
    def get_description(tool: dict) -> str:

        if "description" in tool and "function" not in tool:
            return str(tool["description"] or "")

        return str((tool.get("function") or tool).get("description", "") or "")

    def get_tools(self):

        return list(self.tools.values())

    def get_tool(self, name: str):

        return self.tools.get(name)

    def contains(self, name: str) -> bool:

        return name in self.tools

    def schema(self, name: str) -> dict:

        tool = self.tools.get(name) or {}
        function = tool.get("function", tool)
        schema = function.get("parameters")
        return schema if isinstance(schema, dict) else {"type": "object", "properties": {}}

    def properties(self, name: str) -> dict:

        props = self.schema(name).get("properties")
        return props if isinstance(props, dict) else {}

    def required(self, name: str) -> list:

        return [r for r in self.schema(name).get("required") or [] if isinstance(r, str)]

    def openai_tool(self, name: str) -> dict:
        """Tool definition in OpenAI function format (what the model expects)."""

        tool = self.tools[name]
        if "function" in tool:
            return {"type": "function", "function": tool["function"]}
        return {"type": "function", "function": {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": self.schema(name),
        }}

    def describe(self) -> str:

        lines = []
        for name in self.tools:
            lines.append(
                f"- {name}: {self.get_description(self.tools[name])} | "
                f"parameters: {compact(self.schema(name))}"
            )
        return "\n".join(lines)

    # --------------------------------------------------------
    # Argument helpers: the learned tool map first, then the schema's shape
    # --------------------------------------------------------

    def param(self, name: str, slot: str) -> str | None:
        """The parameter the model mapped to `slot` ("id", "text", "page" ...)."""

        found = self.store_schema.param(name, slot)
        return found if isinstance(found, str) and found in self.properties(name) else None

    def id_param(self, name: str) -> str | None:

        found = self.param(name, "id")
        if found:
            return found
        strings = [
            key for key in self.required(name)
            if schema_kind(self.properties(name).get(key) or {}) == "string"
        ]
        return strings[0] if len(strings) == 1 else None

    def text_param(self, name: str) -> str | None:

        found = self.param(name, "text")
        if found:
            return found
        props = self.properties(name)
        strings = [k for k, s in props.items() if isinstance(s, dict) and schema_kind(s) == "string"
                   and not s.get("enum")]
        strings.sort(key=lambda k: -int((props[k] or {}).get("maxLength") or 0))
        return strings[0] if strings else None

    def quantity_param(self, name: str) -> str | None:

        found = self.param(name, "quantity")
        if found and schema_kind(self.properties(name).get(found) or {}) in ("integer", "number"):
            return found
        return None

    def mode_value(self, name: str, remove: bool) -> tuple[str, Any] | None:
        """(mode parameter, its value for add / remove), as the tool map read it."""

        key = self.param(name, "mode")
        value = self.store_schema.param(name, "remove" if remove else "add")
        if not key or value is None:
            return None
        options = (self.properties(name).get(key) or {}).get("enum")
        if isinstance(options, list) and value not in options:
            return None
        return key, value

    def mode_param(self, name: str) -> tuple[str, list] | None:

        for key, spec in self.properties(name).items():
            if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and spec["enum"]:
                return key, spec["enum"]
        return None


# ============================================================
# 5. RESPONSE NORMALIZATION
# ============================================================

def normalize_llm_response(response: Any) -> dict:

    """
    Expected adapter response:

    {
        "content": "...",
        "tool_calls": [
            {
                "id": "...",
                "name": "...",
                "arguments": {}
            }
        ]
    }
    """

    if isinstance(response, dict):

        content = response.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )

        return {
            "content": content or "",
            "tool_calls": response.get("tool_calls") or []
        }

    raise TypeError("Invalid LLM response")


def normalize_tool_call(call: dict) -> dict:

    """
    Normalize simplified and provider-style tool calls.
    """

    if not isinstance(call, dict):
        return {"id": None, "name": None, "arguments": {}}

    if "name" in call:
        name = call["name"]
        arguments = call.get("arguments", {})
    else:
        function = call.get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments", {})

    if isinstance(arguments, str):
        arguments = parse_json(arguments) if arguments.strip() else {}

    return {
        "id": call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
        "name": name,
        "arguments": arguments if isinstance(arguments, dict) else {}
    }


# ============================================================
# 6. GENERAL JSON / TEXT UTILITIES
# ============================================================

def parse_json(content: Any) -> dict:

    if isinstance(content, dict):
        return content

    if not isinstance(content, str):
        return {}

    content = content.strip()

    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    try:

        result = json.loads(content)

        return result if isinstance(result, dict) else {}

    except json.JSONDecodeError:

        start = content.find("{")
        end = content.rfind("}")

        if start >= 0 and end > start:

            try:

                result = json.loads(content[start:end + 1])

                return result if isinstance(result, dict) else {}

            except json.JSONDecodeError:
                pass

    return {}


def screen_json(reply: Any, ids_by_key: dict[str, str], item_count: int) -> dict:
    """
    Build the screening result from the model's row keys ("1: R3 R7" / "2: none").
    Returns {"items": [{"item": n, "ids": [...]}]}, or {} when no line can be read.
    A line holding anything other than keys (prose) is ignored.
    """

    lookup = {key.lower(): ident for key, ident in ids_by_key.items()}
    picked: dict[int, list[str]] = {}
    understood = False
    for raw in str(reply or "").splitlines():
        line = re.sub(r"[*`#]", "", raw).strip()
        head = re.match(r"(?:item\s*)?(\d+)\s*[:.)=-]\s*(.*)$", line, re.I)
        if head:
            number, rest = int(head.group(1)), head.group(2)
        elif item_count == 1 and line:
            number, rest = 1, line   # a single item may come without its number
        else:
            continue
        tokens = re.findall(r"[a-z0-9]+", rest.lower())
        if not all(t == "none" or re.fullmatch(r"r\d+", t) for t in tokens):
            continue
        understood = True
        chosen = picked.setdefault(number, [])
        for token in tokens:
            ident = lookup.get(token)
            if ident and ident not in chosen:
                chosen.append(ident)
    if not understood:
        return {}
    return {"items": [{"item": n, "ids": ids} for n, ids in picked.items()]}


def kind_json(reply: Any, labels_by_key: dict[str, str]) -> dict:
    """Build the category choice from the model's label keys ("C1 C3"); {} when unreadable."""

    lookup = {key.lower(): label for key, label in labels_by_key.items()}
    tokens = re.findall(r"[a-z0-9]+", re.sub(r"[*`#]", "", str(reply or "")).lower())
    if not tokens or not all(t in lookup for t in tokens):
        return {}
    return {"labels": list(dict.fromkeys(lookup[t] for t in tokens))}


def matrix_json(reply: Any, listing_keys: set, count: int) -> dict:
    """
    Build the judgement from lines like "L1: Y N" or "L1: Y Y = 42".
    Returns {"verdicts": {key: {"marks": [...], "measure": n}}}, or {} when no
    line can be read. A line with the wrong number of marks is ignored, so that
    listing is simply judged again.
    """

    rows: dict[str, dict] = {}
    for raw in str(reply or "").splitlines():
        head = re.match(r"(L\d+)\s*[:.)-]\s*(.*)$", re.sub(r"[*`#]", "", raw).strip(), re.I)
        if not head or head.group(1).upper() not in listing_keys:
            continue
        marks_text, _, measure = head.group(2).partition("=")
        marks = re.findall(r"[a-z]+", marks_text.lower())
        if len(marks) == count and all(m in ("y", "s", "n") for m in marks):
            rows[head.group(1).upper()] = {"marks": marks, "measure": as_number(measure)}
    return {"verdicts": rows} if rows else {}


def audit_json(reply: Any, listing_keys: set) -> dict:
    """Build the second opinion from the model's listing keys or "none"; {} when unreadable."""

    tokens = re.findall(r"[a-z0-9]+", re.sub(r"[*`#]", "", str(reply or "")).lower())
    known = {key.lower(): key for key in listing_keys}
    if tokens == ["none"]:
        return {"keys": []}
    if not tokens or not all(t in known for t in tokens):
        return {}
    return {"keys": [known[t] for t in tokens]}


def trail_json(reply: Any) -> dict:
    """Build the TRAIL result from "OK" or "FAIL 2"; {} when unreadable."""

    found = re.fullmatch(r"\W*(ok|fail)(?:\s+(?:item\s*)?(\d+))?\W*", str(reply or ""), re.I)
    if not found:
        return {}
    return {"ok": found.group(1).lower() == "ok", "item": as_int(found.group(2))}


def approval_json(reply: Any) -> dict:
    """Build the approval result from the model's one-word reply; {} when unreadable."""

    found = re.fullmatch(r"\W*(approved|refused|no_answer)\W*", str(reply or ""), re.I)
    return {"approval": found.group(1).lower()} if found else {}


def compact(value: Any) -> str:

    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(value)


def clip(value: Any, limit: int = CLIP_CHARS, depth: int = 0) -> Any:

    if depth > 6:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: clip(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [clip(v, limit, depth + 1) for v in value[:40]]
    return value


def normalize_text(value: str) -> str:

    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def attribute_name(key: Any) -> str:
    """A property key as plain words: 'fa_pattern' / 'fa.pattern' -> 'pattern'.

    Only the key's shape is used: separators become spaces, short abbreviation
    prefixes (one or two letters) are dropped and plurals made singular.
    """

    words = words_of(key).split()
    while len(words) > 1 and len(words[0]) <= 2:
        words = words[1:]
    words = [w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w for w in words]
    return " ".join(words)


def words_of(key: Any) -> str:
    """'traits_map' / 'traitsMap' / 'traits-map' -> 'traits map'."""

    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", text).strip().lower()


def as_int(value: Any) -> int | None:

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(?:\.0+)?\s*", value):
        return int(float(value))
    return None


def as_number(value: Any) -> float | None:

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        found = re.search(r"-?\d+(?:[.,]\d+)?", value.replace(",", ""))
        if found:
            try:
                return float(found.group(0))
            except ValueError:
                return None
    return None


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20,
}


def stated_count(text: str, nouns: str) -> int | None:
    """Smallest count the request states for one of the nouns ("3 messages")."""

    found = []
    pattern = r"\b(\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")\s+(?:\w+\s+){0,2}?(?:" + nouns + r")\b"
    for hit in re.finditer(pattern, text, re.I):
        token = hit.group(1).lower()
        found.append(int(token) if token.isdigit() else _NUMBER_WORDS[token])
    return min(found) if found else None


def quantity_stated(text: str, quantity: int) -> bool:
    """True when the request names this number at all (digits or a word)."""

    words = [w for w, n in _NUMBER_WORDS.items() if n == quantity]
    return bool(re.search(r"\b(?:" + "|".join([str(quantity), *words]) + r")\b", text, re.I))


def stated_budget(value: Any, text: str) -> float | None:
    """
    The budget the text really states. The model sometimes moves the decimal
    point ("1004 cts" read as 10.04), so the number is taken from the text;
    None when the text holds no such number.
    """

    number = as_number(value)
    if number is None:
        return None
    found = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]
    for guess in (number, number * 100, number / 100):
        for written in found:
            if abs(guess - written) < 1e-6:
                return written
    return None


def words_close(a: str, b: str) -> bool:
    """Same word, or one is the start of the other and at least four letters long."""

    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 4 and long_.startswith(short)


# A value like "not X", "at least 5" or "under 40" cannot be matched word for word.
NOT_PLAIN_VALUE = (r"\d|\bnot\b|\bno\b|\bnon\b|\bat least\b|\bunder\b|\bover\b|"
                   r"\bwithout\b|\bany\b|\bless\b|\bmore\b|\bfree\b")


def strip_ids(text: str, ids: list[str]) -> str:
    """Remove product identifiers from free text (decline remarks)."""

    for ident in sorted(ids, key=len, reverse=True):
        if ident:
            text = text.replace(ident, "")
    text = re.sub(r"\b[a-z]{1,3}-[0-9a-f]{6,}\b", "", text, flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip()


# ============================================================
# 7. SCHEMA VALIDATION
# ============================================================

def schema_kind(spec: dict) -> str:

    kind = spec.get("type") if isinstance(spec, dict) else None
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    return kind if isinstance(kind, str) else ""


def coerce_value(value: Any, spec: dict, where: str, problems: list) -> Any:

    if not isinstance(spec, dict):
        return value
    kind = schema_kind(spec)

    if kind == "integer" and not (isinstance(value, int) and not isinstance(value, bool)):
        number = as_number(value) if not isinstance(value, bool) else None
        if number is not None and float(number).is_integer():
            value = int(number)
        else:
            problems.append(f"{where} must be an integer")
            return value
    elif kind == "number" and not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        number = as_number(value) if not isinstance(value, bool) else None
        if number is None:
            problems.append(f"{where} must be a number")
            return value
        value = number
    elif kind == "string" and not isinstance(value, str):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        else:
            problems.append(f"{where} must be a string")
            return value
    elif kind == "boolean" and not isinstance(value, bool):
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            value = value.strip().lower() == "true"
        else:
            problems.append(f"{where} must be true or false")
            return value
    elif kind == "array" and not isinstance(value, list):
        value = [value]

    options = spec.get("enum")
    if isinstance(options, list) and value not in options:
        lowered = {str(o).lower(): o for o in options}
        if str(value).strip().lower() in lowered:
            value = lowered[str(value).strip().lower()]
        else:
            problems.append(f"{where} must be one of {options}")

    if isinstance(value, str):
        if isinstance(spec.get("maxLength"), int) and len(value) > spec["maxLength"]:
            value = value[:spec["maxLength"]].rstrip()
        if isinstance(spec.get("minLength"), int) and len(value) < spec["minLength"]:
            problems.append(f"{where} is too short")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low, high = spec.get("minimum"), spec.get("maximum")
        if isinstance(low, (int, float)) and value < low:
            problems.append(f"{where} must be at least {low}")
        if isinstance(high, (int, float)) and value > high:
            problems.append(f"{where} must be at most {high}")

    if isinstance(value, list) and isinstance(spec.get("items"), dict):
        value = [coerce_value(v, spec["items"], f"{where}[{i}]", problems)
                 for i, v in enumerate(value)]

    return value


def validate_arguments(arguments: dict, schema: dict) -> tuple[dict, list]:
    """
    Check arguments against the tool schema before spending a step.
    Returns repaired arguments and a list of problems (empty when valid).
    """

    problems: list = []
    if not isinstance(arguments, dict):
        return {}, ["arguments must be an object"]

    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    fixed = {}
    for key, value in arguments.items():
        if key not in props:
            if schema.get("additionalProperties") is False:
                problems.append(
                    f"'{key}' is not an argument of this tool (allowed: {sorted(props)})"
                )
                continue
            fixed[key] = value
            continue
        if value is None and key not in (schema.get("required") or []):
            continue
        fixed[key] = coerce_value(value, props[key], key, problems)

    for key in schema.get("required") or []:
        if key not in fixed:
            problems.append(f"missing required argument '{key}'")

    return fixed, problems


# ============================================================
# 8. OBSERVATION PARSING
# ============================================================

class StoreSchema:
    """
    How this store's tools and observations are laid out, learned by the model
    from the tool schemas and from the first observation of each shape.
    Nothing here is a fixed keyword: field and parameter names differ between
    stores (and may be made-up words), so they are read, not guessed.
    """

    SLOTS = ("id", "title", "price", "category", "brand", "description",
             "in_stock", "attributes", "quantity")
    PROSE = ("title", "category", "brand", "description")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.params: dict[str, dict] = {}          # tool -> slot -> parameter name / value
        self.roles: dict[str, str] = {}            # role -> tool
        self.fields: dict[str, set] = {s: set() for s in self.SLOTS}
        self.attr_name_keys: set = set()
        self.attr_value_keys: set = set()
        self.total_keys: set = set()
        self.next_keys: set = set()
        self.note_keys: set = set()
        self.stock_yes: set = set()
        self.stock_no: set = set()
        self.stock_true_means_unavailable: set = set()   # boolean keys meaning "sold out"
        self.known_keys: set = set()
        self.map_calls = 0

    def param(self, tool: str, slot: str) -> Any:
        return (self.params.get(tool) or {}).get(slot)

    def learn_observation(self, mapping: dict, shape: set) -> None:
        """Store one observation map returned by the model."""

        def key(name: str) -> str | None:
            value = mapping.get(name)
            return value if isinstance(value, str) and value in shape else None

        for slot, answer in (("id", "product_id"), ("title", "title"), ("price", "price"),
                             ("category", "category"), ("brand", "brand"),
                             ("description", "description"), ("in_stock", "in_stock"),
                             ("attributes", "attributes"), ("quantity", "quantity")):
            found = key(answer)
            if found:
                self.fields[slot].add(found)
        for answer, target in (("attribute_name_key", self.attr_name_keys),
                               ("attribute_value_key", self.attr_value_keys),
                               ("total_matches", self.total_keys),
                               ("next_page", self.next_keys)):
            found = key(answer)
            if found:
                target.add(found)
        for note in mapping.get("notes") or []:
            if isinstance(note, str) and note in shape:
                self.note_keys.add(note)
        stock = key("in_stock")
        if stock and str(mapping.get("in_stock_true_means") or "").lower().startswith("unavail"):
            self.stock_true_means_unavailable.add(stock)
        self.stock_yes |= {str(v).strip().lower() for v in mapping.get("available_values") or []}
        self.stock_no |= {str(v).strip().lower() for v in mapping.get("unavailable_values") or []}

    def shape_of(self, observation: Any) -> set:
        """Every key in an observation, except the per-product property keys."""

        keys: set = set()

        def walk(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                for k, v in value.items():
                    keys.add(k)
                    if k not in self.fields["attributes"]:
                        walk(v, depth + 1)
            elif isinstance(value, list):
                for v in value[:5]:
                    walk(v, depth + 1)

        walk(observation, 0)
        return keys

    def id_key_of(self, record: dict) -> str | None:

        for key in self.fields["id"]:
            value = record.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                return key
        return None

    def records_in(self, observation: Any, depth: int = 0) -> list[dict]:
        """Every dict in the observation that is a product record (it carries a product id)."""

        if depth > 6:
            return []
        found = []
        if isinstance(observation, dict):
            if self.id_key_of(observation):
                found.append(observation)
            for key, value in observation.items():
                if isinstance(value, (dict, list)) and key not in self.fields["attributes"]:
                    found.extend(self.records_in(value, depth + 1))
        elif isinstance(observation, list):
            for value in observation[:100]:
                found.extend(self.records_in(value, depth + 1))
        return found

    def record_field(self, record: dict, slot: str) -> Any:
        """The record's value for a learned slot (title, price, ...)."""

        for key in self.fields[slot]:
            if key in record:
                value = record[key]
                if slot == "attributes" or not isinstance(value, (dict, list)):
                    return value
        return None

    def stock_value(self, record: dict) -> bool | None:

        for key in self.fields["in_stock"]:
            if key not in record:
                continue
            value = record[key]
            if isinstance(value, bool):
                return (not value) if key in self.stock_true_means_unavailable else value
            if isinstance(value, (int, float)):
                return value > 0
            if isinstance(value, str):
                low = value.strip().lower()
                if low in self.stock_no:
                    return False
                if low in self.stock_yes:
                    return True
        return None

    def facts_of(self, record: dict) -> dict:
        """The product's properties: the learned attributes field (object or name/value list)."""

        facts: dict = {}
        container = self.record_field(record, "attributes")
        containers = [container] if container is not None else [
            # not learned yet: nested objects of plain values are the usual shape
            v for v in record.values() if isinstance(v, dict)
        ]
        for value in containers:
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool)) or (
                        isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v)
                    ):
                        facts[str(k)] = v
            elif isinstance(value, list):
                for entry in value:
                    if not isinstance(entry, dict):
                        continue
                    name = next((entry[k] for k in self.attr_name_keys if k in entry), None)
                    val = next((entry[k] for k in self.attr_value_keys if k in entry), None)
                    if isinstance(name, str) and val is not None:
                        facts[name] = val
        return facts

    def notices_in(self, observation: Any, depth: int = 0, inside_record: bool = False) -> list[str]:
        """
        Sentences in an observation that are not product data: store or shopper
        notes about changes, refusals and errors.
        """

        if depth > 6:
            return []
        found: list[str] = []
        prose = set().union(*(self.fields[s] for s in self.PROSE))
        if isinstance(observation, str):
            if not inside_record and len(observation.split()) >= 4:
                found.append(observation.strip())
        elif isinstance(observation, list):
            for value in observation[:30]:
                found.extend(self.notices_in(value, depth + 1, inside_record))
        elif isinstance(observation, dict):
            record = inside_record or bool(self.id_key_of(observation))
            for key, value in observation.items():
                if record and isinstance(value, str):
                    continue
                if isinstance(value, str) and key in prose:
                    continue
                if key in self.fields["attributes"]:
                    continue
                found.extend(self.notices_in(value, depth + 1, record))
        return found

    def error_in(self, observation: Any) -> str | None:

        if isinstance(observation, dict):
            for key, value in observation.items():
                # a true/false field is a status flag, not a message from the store
                if key in self.note_keys and not isinstance(value, bool) and value not in (None, "", [], {}):
                    return compact(value)[:300]
        return None

    def page_body(self, observation: Any) -> Any:
        """The object holding the paging fields; a reply may wrap it beside a notes field."""

        if isinstance(observation, dict):
            paging = self.total_keys | self.next_keys
            if paging & observation.keys():
                return observation
            for value in observation.values():
                if isinstance(value, dict) and paging & value.keys():
                    return value
        return observation

    def page_count(self, observation: Any) -> tuple[int | None, int]:
        """(total matches if reported, rows shown)."""

        if not isinstance(observation, dict):
            return None, len(self.records_in(observation))
        total = None
        body = self.page_body(observation)
        for key in self.total_keys:
            if key in body and as_int(body[key]) is not None:
                total = as_int(body[key])
                break
        return total, len(self.records_in(observation))


def sample_of(value: Any, depth: int = 0) -> Any:
    """A short copy of an observation for the model: two list entries, short strings."""

    if depth > 6:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= 80 else value[:80] + "..."
    if isinstance(value, dict):
        return {k: sample_of(v, depth + 1) for k, v in list(value.items())[:40]}
    if isinstance(value, list):
        return [sample_of(v, depth + 1) for v in value[:2]] + (["..."] if len(value) > 2 else [])
    return value


# ============================================================
# 9. PROMPTS
# ============================================================

SYSTEM_PROMPT = (
    "You are the reasoning core of an autonomous shopping agent that serves one "
    "shopper in an online store. A task is scored 1 only when the finish is "
    "exactly right: the order holds every requested item at the requested "
    "quantity, each meeting every stated and discovered requirement, in stock, "
    "inside the budget, and the cheapest or best one when the shopper asks for "
    "that; or, when truly nothing fits, the task ends with the no-purchase tool "
    "as the shopper asked. Never invent product facts, IDs, tool names or "
    "arguments. When asked for JSON, return one JSON object and nothing else."
)

TOOL_MAP_PROMPT = """Here are an online store's tools. Tool and parameter names may be unusual or
made-up words, so decide by the descriptions and parameter types.

TOOLS:
{tools}

Return one JSON object:
{{"roles": {{
    "search": tool that finds products,
    "details": tool that shows one product's full properties / datasheet,
    "listing": tool that shows one product's listing or offer (null if same as details),
    "compare": tool that compares products, or null,
    "cart_edit": tool that adds or removes cart items,
    "cart_view": tool that shows the cart,
    "checkout": tool that places the order,
    "decline": tool that ends the task without buying,
    "message": tool that sends the shopper a message}},
  "params": {{ "<tool name>": {{
    "id": parameter that takes a product id,
    "quantity": parameter for how many units,
    "text": parameter that takes free text to send or a remark,
    "keywords": search parameter for free-text keywords,
    "category": search parameter naming a category / department,
    "brand": search parameter naming a brand / maker,
    "price_cap": parameter for the maximum price,
    "page": parameter that selects the result page,
    "mode": parameter that chooses between adding and removing,
    "add": the value of that mode parameter that adds,
    "remove": the value of that mode parameter that removes}} }}}}
Use the exact tool and parameter names; leave out keys that do not apply.
"""

OBSERVATION_MAP_PROMPT = """The store tool "{tool}" returned the JSON below (lists cut to two
entries). Field names may be unusual or made-up words, so read each field by its
values and where it sits.

{sample}

Return one JSON object with the exact field names as they appear (null when absent):
{{"product_id": field holding a product's id inside each product object,
  "title": field with the product's name / title,
  "price": field with the product's price,
  "category": field with the product's category / department,
  "brand": field with the brand / maker / seller,
  "description": field with a longer free-text description,
  "in_stock": field telling whether the product can be bought now,
  "in_stock_true_means": "available" or "unavailable" (only for a true/false field),
  "available_values": string values of in_stock meaning it can be bought,
  "unavailable_values": string values of in_stock meaning it cannot,
  "attributes": field holding the product's properties (an object or a list of name/value objects),
  "attribute_name_key": inside such a list, the field with the property name,
  "attribute_value_key": inside such a list, the field with the value,
  "quantity": field with the units of a product in a cart line,
  "total_matches": field with the total number of matching products,
  "next_page": field with the next page number / token (also when it is null now),
  "notes": [fields whose values are messages, notices, refusals or errors from the store or shopper]}}
"""

ANALYSIS_PROMPT = """Read the shopper's request and the store's tools, and describe the task.

Return one JSON object with these keys:
"items": one entry per distinct product to buy:
    {{"name": the kind of product in the request's words,
      "category_hint": the category or department the request names for it, else "",
      "quantity": integer units (1 when not stated; "a pair" of shoes or jeans is 1),
      "requirements": firm requirements as "property: value" strings using the
                      request's own words (material, type, shape, size, colour,
                      occasion the product is made for, exclusions ...),
      "preferences": nice-to-haves only}}
    A category or department the request names ("fall/winter clothing", "bomber
    jackets") goes in category_hint only; never turn its words into a requirement
    such as "season: fall/winter" - the category itself already settles that.
    Price, budget and stock never go into requirements. Story details (who it is
    for, why, where) are context, not requirements, unless they state a value
    of the product itself.
"budget": the maximum spend as a number in the request's unit, or null.
"budget_scope": "total" or "each".
"selection": "cheapest" when the shopper wants the lowest price, "most" or
    "least" for the extreme of a measure, otherwise "any".
"selection_measure": the measure for most/least, else "".
"in_stock_only": true unless the shopper accepts unavailable items.
"message_limit": how many messages the agent may send the shopper (integer), or null.
"step_limit": how many moves / steps / tool calls are allowed (integer), or null.
"quiet_decline": true when the shopper says not to offer substitutes or
    alternatives if nothing fits.
"decline_note": what the shopper wants said when nothing fits, else "".
"approval_steps": things the request requires the agent to do with the shopper
    before ordering (e.g. "get approval before buying"), each
    {{"step": short text, "messages": messages it needs}}; [] when none.
"cross_requirements": requirements that relate the items to each other ("same brand",
    "matching colour", "from one seller"), as short strings; [] when none.
"roles": the exact tool name for each job, or null:
    "search", "details" (one listing's attributes / properties),
    "listing" (one listing's offer: price and stock, when separate from details),
    "compare" (several listings at once), "cart_edit", "cart_view",
    "checkout", "decline" (end with no purchase), "message" (talk to the shopper).

TOOLS:
{tools}

REQUEST:
{request}
"""

PLAN_PROMPT = """Plan product searches for this shopping task. Call the search tool
{count} times, each with a different, valid arguments object.

TASK:
{task}

SEARCHES ALREADY RUN (do not repeat them):
{history}

CATEGORY LABELS SEEN IN RESULTS SO FAR:
{labels}

Guidance:
- Read the search tool's schema; use only its argument names and allowed values.
- Every search needs keywords or a category / brand filter; a search with only a
  mode and a price cap returns nothing.
- Keyword searches work best short: the required value plus the product noun
  (e.g. the style word + the product). They find the listings that name the value,
  even when the title does not show it, and result pages are short. So first run
  one such search for each required value, then one with all the values together.
- A structured / filter search on the exact category label seen in results, with
  the price cap set to the budget, lists that category; use it once a label is known.
- If an earlier search found nothing, change the words, the mode or the filter;
  try a synonym of the product noun or the required value on its own.
"""

FOCUSED_SEARCH_PROMPT = """Run a small follow-up product search. Call the search tool up to {count} times,
each with a different, valid arguments object.

PRODUCT: {item}{hint}
VALUES THE SHOPPER REQUIRES: {values}
BUDGET: {budget}

SEARCHES ALREADY RUN (do not repeat them):
{history}

Guidance:
- Use short keyword searches: the required value plus the product noun. They find the
  listings that name the value even when the title does not show it.
- Use only the search tool's argument names and allowed values, and set its price cap
  to the budget when it has one.
"""

REFINE_SEARCH_PROMPT = """The first product searches came back thin. Call the search tool up to 2 times,
each with a different, valid arguments object.

TASK:
{task}

SEARCHES THAT FOUND NOTHING:
{failed}

ALL SEARCHES RUN (do not repeat them):
{history}

CATEGORY LABELS SEEN IN RESULTS SO FAR:
{labels}

Guidance:
- Change what failed: use fewer words, a synonym of the product noun, or a required
  value on its own.
- A category filter must use a label exactly as listed above; when none is listed, use
  keywords.
- Use only the search tool's argument names and allowed values.
"""

TRAIL_PROMPT = """These products were picked together for one order. Check only the requirements
that relate them to each other.

REQUIREMENTS:
{requirements}

PICKS:
{picks}

Reply with exactly one line and nothing else: OK when every requirement holds, or FAIL
and the item number that should be replaced, for example: FAIL 2
"""

AUDIT_PROMPT = """A shopping agent rejected every listing and is about to tell the shopper that
nothing fits. Check its work: a wrong "nothing fits" costs the sale, and so does buying
something that does not fit.

ITEM: {item}
REQUIREMENTS (all must hold):
{requirements}

REJECTED LISTINGS (key: data, then why it was rejected):
{listings}

Name a listing only when an attribute, the title or the description states every
required value outright. Similar words, guesses and silence do not count.

Reply with the keys of those listings and nothing else, or the word none.
"""

CHEAPER_SEARCH_PROMPT = """Call the search tool once to list the store category "{label}" with the price
cap set to {below}. Use the tool's filter on the category label (not keywords) and only
its argument names and allowed values.
"""

SCREEN_PROMPT = """Screen search results for a shopper.

ITEMS WANTED:
{items}

RESULT ROWS (key | title | category | brand | price | in stock | description start):
{rows}

For each item, pick the keys of rows that ARE the requested kind of product (the
category and title must fit the item as a whole, not a neighbouring kind). Judge
only the kind: properties such as style, colour or material are checked later on
the full datasheet, and a row rarely shows them. Leave a row out only when its
category or title clearly shows a different kind of product; when unsure, keep it.

Reply with one line per item and nothing else: the item number, a colon, then the
row keys. Write none when no row fits. Do not write JSON or explanations. Example:
1: R3 R7 R12
2: none
"""

QUESTION_PROMPT = """You are choosing questions to send a shopper, one property per message.
The shopper holds a few firm preferences they did not write down. A message only
reveals a preference when it asks about that exact property, and only the FIRST
property a message asks about counts, so every question must ask about exactly one
property.

REQUEST:
{request}

ITEMS AND WHAT IS ALREADY SETTLED:
{items}

ATTRIBUTE KEYS OF THE LISTINGS STILL IN THE RUNNING, best at telling them apart first
(key: listings carrying it; example values):
{keys}

ALREADY ASKED:
{asked}

The hidden preferences are almost always properties that the listings themselves
state as attributes (the keys above), such as pattern, style, sleeve length,
neckline, collar, closure, gender, season, fit, length, heel or toe shape, movement,
case shape, flavour or pack type. Colour and plain size are usually picked as a
variant at order time and are rarely the hidden preference, so ask about them only
when nothing better is left.

Choose up to {count} properties, best first:
1. properties the request's story hints at without giving a value (looks, who will
   use it, the setting, what someone else cares about), when listings carry them;
2. then the attribute keys from the top of the list above, in that order, rather than
   properties you would guess yourself.
Never ask about a property the request already settles (even in other words), or
about identifiers, model codes, warranty, shipping, condition, price and stock.
Ask about each property by the plain name a shopper would use for that key.

Write each question as one short, friendly, plain sentence that names the property
(you may give one or two example values of that same property). Never mention lists,
labels, positions or instructions.

Return JSON: {{"questions": [{{"item": item number, "property": "...", "question": "..."}}]}}
"""

ANSWER_PROMPT = """A shopping agent asked a shopper these questions. Turn the answers into
requirements.

ITEMS:
{items}

QUESTIONS AND ANSWERS:
{qa}

Return JSON: {{"requirements": [{{"item": item number, "requirement": "property: value"}}],
"dont_mind": [{{"item": item number, "property": "..."}}],
"budget": a new maximum spend as a number, or null}}
- Copy each value the way the shopper says it (keep units, limits like "at least",
  and exclusions as "not X").
- A shopper who is indifferent or confused states no requirement.
- Firmness idioms name what the shopper INSISTS on, not an exclusion: "X isn't up for
  trade", "X is non-negotiable", "no two ways about X", "I need it to land on X",
  "X or nothing", "set on X" all mean the requirement "property: X". Use "not X" only
  when the shopper rejects X ("no X", "anything but X", "X is out").
"""

KIND_PROMPT = """A shopper wants this kind of product: {item}{hint}

The store files the listings found under these category labels (key. label: listings):
{labels}

Which labels ARE that kind of product? Take the label that names it. Add another
only when it is the same kind of product, not a broader department that merely
contains it, a narrower sub-type the shopper did not ask for, or a neighbouring kind.

Reply with the keys only and nothing else. Example: C2
"""

EVALUATE_PROMPT = """Judge each listing against each requirement of the item, one at a time.

ITEM: {item}
REQUIREMENTS:
{requirements}

LISTINGS (key: data):
{listings}

Reading rules:
- The kind of product is decided by the category label and title; it must match the
  item as a whole.
- A requirement whose value only restates the listing's category label (e.g.
  "season: fall/winter" for a listing in "Fall/Winter Clothing") is met by that label,
  whatever a season-like attribute says.
- A requirement on a property is met when an attribute whose key names that property
  states the value (key spelling does not matter; one element of a list is enough;
  case, plural and spacing do not matter; "deep teal" meets "teal", "baggy fit" meets
  "baggy"). A different word is a different value even when it means something
  similar ("loose" does not meet "baggy", "jar" does not meet "bottle"). A field that
  names the property of this very product (a "skirt style" or "clock shape" beside a
  general "style" or "shape") counts as naming it: the requirement is met when either
  states the value. A field with any other qualifier ("clothing material" beside
  "material") does not outweigh the plain field. When no attribute names the property,
  the value stated in another attribute, the title or the description counts. When
  nothing states the value, it is not met.
- The one exception to "a different word": an attribute naming that property with
  words that mean exactly the same thing ("full sleeve" for "long sleeve", "no
  sleeves" for "sleeveless"). Mark that S. Merely similar values stay N.
- Units must be converted; limits include the edge value.
{measure_rule}
Reply with one line per listing and nothing else: the listing key, a colon, then one
mark per requirement in the order given: Y when met, S when met by the exception
above, N when not. Do not write JSON or explanations. Example for two requirements:
L1: Y N
L2: Y S
"""

NOTICE_PROMPT = """While shopping, the store returned the note(s) below after the agent's
last action. Decide what changed.

NOTES:
{notes}

LAST ACTION: {action}
CART (product ids): {cart}
ITEMS:
{items}
CURRENT BUDGET: {budget}

Return JSON: {{"kind": "change" | "error" | "info",
"unavailable_ids": [ids the note says can no longer be bought],
"cart_item_unavailable": true when it says a chosen / carted product is gone even
without naming it,
"budget": the new maximum spend as a number, or null,
"add": [{{"item": item number, "requirement": "property: value"}}],
"drop": [{{"item": item number, "requirement": "..."}}],
"summary": "one short sentence"}}
Reading notes:
- "Never mind about X", "forget X", "X was more of a wish than a need", "X doesn't
  matter any more" mean the shopper no longer requires property X: put the matching
  requirement from ITEMS in "drop" (copy its text).
- "I'm working with N now", "my budget is N", "I can spend N" give the new budget N.
- "Your pick / suggestion is gone / ran dry / shows zero left / sold out" means the
  chosen product is unavailable (set cart_item_unavailable).
"""

APPROVAL_PROMPT = """A shopping agent asked the shopper for approval before placing an order.

QUESTION: {question}
SHOPPER'S REPLY: {reply}

Answer with exactly one word and nothing else:
APPROVED - the shopper agrees to this purchase
REFUSED - the shopper says not to buy this
NO_ANSWER - anything else (indifferent, off-topic or unclear)
"""

CART_PROMPT = """Call the cart tool once to {action} product "{product_id}"{quantity}.
Use the tool's exact argument names and allowed values."""

FINISH_PROMPT = """Call the {tool} tool once to {action}. Use its exact argument names
and allowed values.{extra}"""


# ============================================================
# 11. SHOPPING AGENT
# ============================================================

class ShoppingAgent:

    def __init__(
        self,

        llm_step: Callable[..., Awaitable[dict]],

        tool_executor: Callable[..., Awaitable[Any]],

        tool_definitions: list[dict],

        max_steps: int = MAX_AGENT_STEPS,

        clock: Callable[[], float] | None = None,

    ):
        self.tool_definitions = tool_definitions
        self.llm_step = llm_step

        self.tool_executor = tool_executor

        self.store_schema = StoreSchema()   # every task learns its own store layout

        self.registry = DynamicToolRegistry(
            tool_definitions,
            self.store_schema
        )

        self.max_steps = max_steps
        self.clock = clock or time.monotonic
        self.started = self.clock()

        self.messages = []
        self.llm_calls = 0
        self.llm_cost = 0.0
        self.current_context: ShoppingContext | None = None

    # ========================================================
    # 12. LLM INTERFACE
    # ========================================================

    def elapsed(self) -> float:

        return self.clock() - self.started

    def can_think(self) -> bool:

        return (
            self.llm_calls < MAX_LLM_CALLS
            and self.llm_cost < COST_LIMIT
            and self.elapsed() < LAST_MODEL_CALL
        )

    def exploring_allowed(self, context: ShoppingContext) -> bool:

        reserve = FINISH_RESERVE
        if context.items and all(self.best_fit(context, i) for i in range(len(context.items))):
            # with a pick in hand keep room to swap it if it sells out at checkout
            reserve += STOCKOUT_RESERVE
        return (
            self.can_think()
            and self.elapsed() < EXPLORE_UNTIL
            and self.steps_left(context) > reserve
            and not context.episode_finished
        )

    async def think(
        self,
        user_prompt: str,
        not_use_tools: bool = False,
        tools: list[dict] | None = None,
        role: str = "agent",
        max_tokens: int = 2500,
    ) -> dict:
        """
        One stateless model call: system prompt + this prompt (+ tools).
        Each phase passes everything it needs in the prompt, so no stale or
        malformed conversation history ever reaches the provider.
        """

        if not self.can_think():
            raise BudgetExhausted("model budget used up")

        offered = [] if not_use_tools else (tools if tools is not None else [
            self.registry.openai_tool(n) for n in self.registry.tools
        ])
        self.llm_calls += 1
        messages = [{"role": "user", "content": user_prompt}]
        response = await self.llm_step(
            messages=messages,
            tools=offered,
            role=role,
            max_tokens=max_tokens,
        )

        if isinstance(response, dict):
            cost = response.pop("_cost", None)
            if isinstance(cost, (int, float)):
                self.llm_cost += float(cost)

        normalized = normalize_llm_response(response)
        self.messages.append({
            "role": "assistant",
            "prompt": user_prompt[:300],
            "content": str(normalized.get("content") or "")[:600],
            "tool_calls": [normalize_tool_call(c) for c in normalized["tool_calls"]],
        })
        return normalized

    async def think_json(self, prompt: str, role: str = "reasoning",
                         max_tokens: int = 4096,
                         parser: Callable[[Any], dict] = parse_json) -> dict:

        for _ in range(2):
            try:
                response = await self.think(prompt, not_use_tools=True, role=role,
                                            max_tokens=max_tokens)
            except BudgetExhausted:
                raise
            except Exception as exc:
                # A failed model call must not abort the whole episode: this
                # phase degrades to "no answer" and the pipeline continues.
                logger.warning("model call failed: %s", exc)
                continue
            parsed = parser(response.get("content", ""))
            if parsed:
                return parsed
            if not self.can_think():
                break
        return {}

    async def tool_call_for(self, tool_name: str, prompt: str) -> dict | None:
        """Ask the model for one call of a single offered tool, validated."""

        if not self.registry.contains(tool_name) or not self.can_think():
            return None
        for _ in range(2):
            response = await self.think(
                prompt, tools=[self.registry.openai_tool(tool_name)], role="agent",
                max_tokens=800,
            )
            for raw in response.get("tool_calls", []):
                call = normalize_tool_call(raw)
                if call.get("name") != tool_name:
                    continue
                fixed, problems = validate_arguments(
                    call["arguments"], self.registry.schema(tool_name)
                )
                if not problems:
                    call["arguments"] = fixed
                    return call
                prompt += f"\nYour last arguments were invalid: {problems}. Fix them."
            if not self.can_think():
                break
        return None

    # ========================================================
    # 13. TOOL EXECUTION
    # ========================================================

    def steps_left(self, context: ShoppingContext) -> int:

        return min(context.step_limit, self.max_steps) - context.step_count

    async def execute_tool(
        self,
        tool_call: dict,
        context: ShoppingContext
    ) -> Any:
        """
        Validate and run one tool call (one environment step).
        Invalid calls are refused here, before they cost a step.
        """

        call = normalize_tool_call(tool_call)

        name = call.get("name")

        arguments = call.get("arguments", {})

        if not name or not self.registry.contains(name):

            raise ValueError(
                f"Unknown tool: {name}"
            )

        fixed, problems = validate_arguments(arguments, self.registry.schema(name))

        if problems:

            raise ValueError(
                f"Invalid arguments for {name}: {problems}"
            )

        if context.episode_finished:
            raise RuntimeError("the episode has finished")

        if self.steps_left(context) <= 0:
            raise RuntimeError("maximum agent steps reached")

        if name == context.roles.get("message"):
            if context.messages_sent >= context.message_limit:
                raise RuntimeError("shopper message budget used up")
            context.messages_sent += 1

        context.step_count += 1

        result = await self.tool_executor(

            name=name,

            arguments=fixed

        )

        if isinstance(result, dict) and "observation" in result:
            if isinstance(result.get("step"), int):
                context.step_count = max(context.step_count, result["step"])
            if result.get("finished"):
                context.episode_finished = True
            observation = result.get("observation")
        else:
            observation = result

        context.tool_observations.append({
            "tool_call_id": call.get("id"),
            "tool": name,
            "arguments": fixed,
            "result": clip(observation, 300),
        })
        await self.learn_observation(name, observation)
        self.absorb(observation, name, context)
        for note in self.side_notes(name, observation, context):
            if note not in context.notices and note not in context.unread_notes:
                context.unread_notes.append(note)
        return observation

    def side_notes(self, name: str, observation: Any, context: ShoppingContext) -> list[str]:
        """Store notes riding on a research reply; cart and finish replies are read where they happen."""

        roles = context.roles
        if name == roles.get("message"):
            # the reply itself is the answer; notes come as a list beside it
            if not isinstance(observation, dict):
                return []
            return [s.strip() for value in observation.values() if isinstance(value, list)
                    for s in value if isinstance(s, str) and s.strip()]
        if name in (roles.get("search"), roles.get("details"), roles.get("listing"), roles.get("compare")):
            return self.store_schema.notices_in(observation)
        return []

    async def read_side_notes(self, context: ShoppingContext, action: str) -> None:
        """Apply the notes that came with research replies (a new budget, a dropped requirement)."""

        notes, context.unread_notes = context.unread_notes, []
        if notes:
            context.notices.extend(n for n in notes if n not in context.notices)
            await self.handle_notices(context, notes, action)

    async def act(self, role: str, arguments: dict, context: ShoppingContext) -> tuple[Any, list[str]]:
        """Run the tool that plays `role`; returns (observation, notices)."""

        name = context.roles.get(role)
        if not name:
            raise ValueError(f"no tool for role {role}")
        observation = await self.execute_tool(
            {"name": name, "arguments": arguments}, context
        )
        notes = [n for n in self.store_schema.notices_in(observation) if n not in context.notices]
        if role != "message":
            context.notices.extend(notes)
        return observation, notes

    # ========================================================
    # 14. DYNAMIC TOOL LOOP
    # ========================================================

    async def run_tool_loop(
        self,

        user_prompt: str,

        context: ShoppingContext,

        tools: list[str],

        max_calls: int = MAX_TOOL_CALLS_PER_LOOP

    ) -> list[Any]:
        """
        Ask the model for up to `max_calls` calls of the offered tools and run
        them; invalid calls are fed back once for repair without a step.
        """

        observations = []
        offered = [self.registry.openai_tool(n) for n in tools if self.registry.contains(n)]
        if not offered:
            return observations

        prompt = user_prompt
        calls_used = 0
        wanted = min(2, max_calls)
        for _attempt in range(3):
            if calls_used >= max_calls or not self.exploring_allowed(context):
                break
            response = await self.think(prompt, tools=offered, role="agent", max_tokens=1500)
            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                break
            feedback = []
            for original_call in tool_calls[:max_calls - calls_used]:
                call = normalize_tool_call(original_call)
                if call.get("name") not in tools:
                    # only the offered tools may run in this phase (never a
                    # cart, checkout or decline call during research)
                    feedback.append(f"{call.get('name')}: not available now; use {tools}")
                    continue
                key = (call.get("name"), compact(call.get("arguments")).lower())
                if key in [(h["tool"], compact(h["arguments"]).lower()) for h in context.search_history]:
                    # same search, maybe in other letter case: it would only repeat old rows
                    feedback.append(f"{call.get('name')} {compact(call.get('arguments'))}: already run")
                    continue
                if self.search_target_missing(call["name"], call.get("arguments") or {}):
                    feedback.append(
                        f"{call.get('name')} {compact(call.get('arguments'))}: no keywords and no "
                        "category / brand filter, so it lists nothing useful; add one"
                    )
                    continue
                if not self.exploring_allowed(context):
                    break
                try:
                    result = await self.execute_tool(call, context)
                    calls_used += 1
                    self.record_search(call["name"], call["arguments"], result, context)
                    observations.append(result)
                    observations.extend(await self.follow_pages(call, result, context))
                except Exception as exc:
                    context.errors.append({"tool": call.get("name"), "error": str(exc)})
                    context.errors = context.errors[-MAX_ERRORS:]
                    feedback.append(f"{call.get('name')} {compact(call.get('arguments'))}: {exc}")
            if not feedback and calls_used >= wanted:
                break
            done = "\n".join(
                f"{h['tool']} {compact(h['arguments'])} -> {h['summary']}"
                for h in context.search_history[-8:]
            )
            prompt = user_prompt + "\n\nSearches run so far:\n" + (done or "none")
            if feedback:
                prompt += "\n\nThese calls were rejected (no step used):\n" + "\n".join(feedback)
            else:
                prompt += "\n\nMake one more, different search."

        return observations

    async def follow_pages(self, call: dict, observation: Any, context: ShoppingContext) -> list:
        """Read further pages of a search when the store reports more results."""

        pages = []
        limit = MAX_EXTRA_PAGES
        name, args = call["name"], dict(call["arguments"])
        total, shown = self.store_schema.page_count(observation)
        if total is not None and shown:
            # Page on only when the pages we may read can show every match;
            # otherwise a narrower search is the better use of steps.
            reachable = shown * (1 + MAX_EXTRA_PAGES)
            page_param = self.registry.param(name, "page")
            spec = self.registry.properties(name).get(page_param or "") or {}
            if isinstance(spec.get("maximum"), int):
                reachable = min(reachable, shown * spec["maximum"])
            if total > reachable:
                # too many to read them all, but a search naming a required value
                # is the best source there is: take one more page of it
                if not self.search_value_hits(name, args, context):
                    return pages
                limit = 1
        for _ in range(limit):
            next_args = self.next_page_args(name, args, observation)
            if not next_args or not self.exploring_allowed(context):
                break
            if self.steps_left(context) <= FINISH_RESERVE + 4:
                break
            try:
                observation = await self.execute_tool({"name": name, "arguments": next_args}, context)
            except Exception as exc:
                context.errors.append({"tool": name, "error": str(exc)})
                break
            self.record_search(name, next_args, observation, context)
            pages.append(observation)
            args = next_args
        return pages

    # ---- search bookkeeping -------------------------------------------

    def search_target_missing(self, name: str, args: dict) -> bool:
        """True for a search holding only a mode / price cap / page: it lists nothing."""

        targets = [p for p in (self.registry.param(name, slot) for slot in ("keywords", "category", "brand")) if p]
        if not targets:
            # no map: any free-text parameter other than the page one may carry a target
            props = self.registry.properties(name)
            page = self.registry.param(name, "page")
            targets = [k for k, spec in props.items()
                       if schema_kind(spec) == "string" and not spec.get("enum") and k != page]
        if not targets:
            return False
        return not any(isinstance(args.get(k), str) and args[k].strip() for k in targets)

    def cap_param(self, name: str) -> str | None:

        cap = self.registry.param(name, "price_cap")
        if cap and schema_kind(self.registry.properties(name).get(cap) or {}) in ("number", "integer"):
            return cap
        return None

    def mode_signature(self, name: str, args: dict) -> str:
        """The search mode: the tool plus its enum-valued arguments."""

        props = self.registry.properties(name)
        return name + compact({k: v for k, v in sorted(args.items()) if (props.get(k) or {}).get("enum")})

    def record_search(self, name: str, args: dict, observation: Any, context: ShoppingContext) -> None:
        """Log a search, remember its next page, and learn whether its mode obeys the price cap."""

        context.search_history.append({
            "tool": name, "arguments": args,
            "summary": self.search_summary(observation),
            "next_args": self.next_page_args(name, args, observation),
        })
        hits = self.search_value_hits(name, args, context)
        for record in self.store_schema.records_in(observation) if hits else []:
            candidate = context.candidates.get(str(record.get(self.store_schema.id_key_of(record)) or "").strip())
            if candidate is not None:
                candidate.value_hits = max(candidate.value_hits, hits)
        cap = self.cap_param(name)
        limit = as_number(args.get(cap)) if cap else None
        if limit is None:
            return
        prices = [
            as_number(self.store_schema.record_field(r, "price"))
            for r in self.store_schema.records_in(observation)
        ]
        if any(p is not None and p > limit + 1e-9 for p in prices):
            context.cap_ignored.add(self.mode_signature(name, args))

    def search_value_hits(self, name: str, args: dict, context: ShoppingContext) -> int:
        """How many required values a search's keywords name."""

        keywords = args.get(self.registry.param(name, "keywords") or "")
        if not isinstance(keywords, str):
            return 0
        asked = self._stems(keywords)
        return max((len(self.required_values(i) & asked) for i in context.items), default=0)

    def next_page_args(self, name: str, args: dict, observation: Any) -> dict | None:

        if not isinstance(observation, dict) or not self.store_schema.records_in(observation):
            return None
        page_param = self.registry.param(name, "page")
        if not page_param:
            return None
        spec = self.registry.properties(name).get(page_param) or {}
        token = None
        said_no_more = False
        body = self.store_schema.page_body(observation)
        for key in self.store_schema.next_keys:
            if key not in body:
                continue
            value = body[key]
            if value not in (None, False, "", [], {}):
                token = value
                break
            said_no_more = True
        if token is None and said_no_more:
            return None   # the store says this was the last page
        if token is None or token is True:
            total, shown = self.store_schema.page_count(observation)
            current = as_int(args.get(page_param)) or 1
            if total is None or schema_kind(spec) != "integer" or total <= shown * current:
                return None
            token = current + 1
        next_args = dict(args)
        next_args[page_param] = token
        fixed, problems = validate_arguments(next_args, self.registry.schema(name))
        return None if problems or fixed == args else fixed

    def search_summary(self, observation: Any) -> str:

        total, shown = self.store_schema.page_count(observation)
        error = self.store_schema.error_in(observation)
        if error:
            return f"error {error[:120]}"
        return f"{shown} rows" + (f" of {total}" if total is not None else "")

    # ========================================================
    # 15. REQUEST ANALYSIS
    # ========================================================

    async def learn_tools(self) -> None:
        """Ask the model which tool plays which role and what each parameter means."""

        if not self.can_think():
            return
        try:
            mapping = await self.think_json(
                TOOL_MAP_PROMPT.format(tools=self.registry.describe()), role="mapper", max_tokens=2000
            )
        except BudgetExhausted:
            return
        roles = mapping.get("roles") if isinstance(mapping.get("roles"), dict) else {}
        self.store_schema.roles = {
            role: name for role, name in roles.items()
            if role in ROLE_NAMES and isinstance(name, str) and self.registry.contains(name)
        }
        params = mapping.get("params") if isinstance(mapping.get("params"), dict) else {}
        self.store_schema.params = {
            tool: dict(slots) for tool, slots in params.items()
            if self.registry.contains(tool) and isinstance(slots, dict)
        }

    async def learn_observation(self, tool: str, observation: Any) -> None:
        """Map an observation's fields the first time a new layout shows up."""

        shape = self.store_schema.shape_of(observation)
        if not shape or shape <= self.store_schema.known_keys:
            return
        if self.store_schema.map_calls >= MAX_MAP_CALLS or not self.can_think():
            return
        self.store_schema.map_calls += 1
        try:
            mapping = await self.think_json(OBSERVATION_MAP_PROMPT.format(
                tool=tool, sample=compact(sample_of(observation))[:6000],
            ), role="mapper", max_tokens=1500)
        except BudgetExhausted:
            return
        if mapping:
            self.store_schema.learn_observation(mapping, shape)
            # recomputed now that the properties field is known: its per-product
            # keys are not part of the layout (no answer: try again next time)
            self.store_schema.known_keys |= self.store_schema.shape_of(observation)

    async def analyze_request(
        self,
        context: ShoppingContext
    ):

        context.state = AgentState.ANALYZE

        # The model reads the tool schemas once: which tool plays which role and
        # which parameter carries the id, text, page, price cap ...
        await self.learn_tools()
        context.roles = dict(self.store_schema.roles)

        prompt = ANALYSIS_PROMPT.format(
            tools=self.registry.describe(),
            request=context.original_request,
        )
        try:
            analysis = await self.think_json(prompt, role="reasoning", max_tokens=3000)
        except Exception as exc:
            context.errors.append({"phase": "analysis", "error": str(exc)[:300]})
            analysis = {}
        context.analysis = analysis

        # ---- tool roles: the analysis, then the tool map fills the gaps ----
        roles = {}
        given = analysis.get("roles") if isinstance(analysis.get("roles"), dict) else {}
        for role in ROLE_NAMES:
            name = given.get(role)
            if isinstance(name, str) and self.registry.contains(name):
                roles[role] = name
        for role, name in self.store_schema.roles.items():
            if role not in roles and name not in roles.values():
                roles[role] = name
        context.roles = roles

        # ---- items ----
        for entry in analysis.get("items") or []:
            if not isinstance(entry, dict):
                continue
            quantity = as_int(entry.get("quantity")) or 1
            if quantity > 1 and not quantity_stated(context.original_request, quantity):
                quantity = 1   # "a pair" read as two: the request never says that number
            hint = str(entry.get("category_hint") or "")
            hint_stems = self._stems(hint)
            requirements = []
            for r in entry.get("requirements") or []:
                if not r:
                    continue
                value = str(r).split(":", 1)[-1]
                if hint_stems and self._stems(value) and self._stems(value) <= hint_stems:
                    continue   # only restates the category ("season: fall/winter")
                requirements.append(str(r))
            context.items.append(ShoppingItem(
                name=str(entry.get("name") or "the requested product"),
                category_hint=hint,
                quantity=max(1, quantity),
                requirements=requirements,
                preferences=[str(p) for p in entry.get("preferences") or [] if p],
            ))
        if not context.items:
            context.items.append(ShoppingItem(name=context.original_request[:120]))

        # ---- budget and pick rule ----
        context.budget = stated_budget(analysis.get("budget"), context.original_request)
        if context.budget is None:
            context.budget = as_number(analysis.get("budget"))
        if context.budget is not None and context.budget <= 0:
            context.budget = None
        context.budget_scope = "each" if analysis.get("budget_scope") == "each" else "total"
        selection = str(analysis.get("selection") or "any").lower()
        context.selection = selection if selection in ("cheapest", "most", "least") else "any"
        context.selection_measure = str(analysis.get("selection_measure") or "")
        context.in_stock_only = analysis.get("in_stock_only") is not False

        # ---- message and step budgets (request text is the authority) ----
        stated_messages = stated_count(context.original_request, r"messages?|questions?|texts?|notes?")
        model_messages = as_int(analysis.get("message_limit"))
        candidates = [v for v in (stated_messages, model_messages) if v is not None and v >= 0]
        if not context.roles.get("message"):
            context.message_limit = 0
        elif candidates:
            context.message_limit = min(candidates)
        else:
            context.message_limit = 2
        stated_steps = stated_count(context.original_request, r"moves?|steps?|tool calls?|actions?|calls?")
        model_steps = as_int(analysis.get("step_limit"))
        for limit in (stated_steps, model_steps):
            if limit is not None and 5 <= limit < context.step_limit:
                context.step_limit = limit

        context.quiet_decline = analysis.get("quiet_decline") is not False
        context.decline_note = str(analysis.get("decline_note") or "")
        for entry in analysis.get("approval_steps") or []:
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                context.approval_steps.append(str(entry["step"]).strip())
                context.approval_messages += max(0, as_int(entry.get("messages")) or 1)
        context.approval_messages = min(context.approval_messages, context.message_limit)
        context.cross_requirements = [str(r) for r in analysis.get("cross_requirements") or [] if r]

        return context.analysis

    def task_text(self, context: ShoppingContext) -> str:

        lines = []
        for number, item in enumerate(context.items, 1):
            lines.append(
                f"Item {number}: {item.name} x{item.quantity}"
                + (f" (category: {item.category_hint})" if item.category_hint else "")
                + f"\n  requirements: {item.requirements or 'none stated'}"
                + (f"\n  shopper does not mind: {item.dont_mind}" if item.dont_mind else "")
            )
        lines.append(f"Budget: {context.budget} ({context.budget_scope})")
        lines.append(f"Pick rule: {context.selection} {context.selection_measure}".strip())
        return "\n".join(lines)

    # ========================================================
    # 16. EXECUTION PLANNING
    # ========================================================

    async def create_plan(
        self,
        context: ShoppingContext
    ):

        context.state = AgentState.PLAN

        search = context.roles.get("search")
        if not search:
            return []
        labels = sorted({c.category for c in context.candidates.values() if c.category})[:40]
        history = [
            f"{h['tool']} {compact(h['arguments'])} -> {h['summary']}"
            for h in context.search_history
        ] or ["none"]
        values = max((len(i.requirements) for i in context.items), default=0)
        count = min(MAX_TOOL_CALLS_PER_LOOP, 2 + values) if not context.search_history else 2
        prompt = PLAN_PROMPT.format(
            count=count,
            task=self.task_text(context),
            history="\n".join(history[-12:]),
            labels=", ".join(labels) or "none yet",
        )
        context.plan.append({"round": len(context.plan) + 1, "prompt_labels": labels})
        return prompt

    # ========================================================
    # 17. PRODUCT SEARCH
    # ========================================================

    async def search_products(
        self,
        context: ShoppingContext
    ):

        context.state = AgentState.SEARCH

        prompt = await self.create_plan(context)
        if not prompt:
            return []
        search_tools = [context.roles["search"]]
        observations = await self.run_tool_loop(prompt, context, search_tools)
        return observations

    def absorb(self, observation: Any, tool: str, context: ShoppingContext) -> None:
        """Merge every product record of an observation into the ledger."""

        detail_tool = tool in (context.roles.get("details"), context.roles.get("compare"))
        for record in self.store_schema.records_in(observation):
            ident = str(record.get(self.store_schema.id_key_of(record)) or "").strip()
            if not ident:
                continue
            candidate = context.candidates.get(ident)
            if candidate is None:
                if len(context.candidates) >= MAX_CANDIDATES:
                    continue
                candidate = ProductCandidate(candidate_id=ident, title="", source=tool)
                context.candidates[ident] = candidate
            title = self.store_schema.record_field(record, "title")
            if isinstance(title, str) and title:
                candidate.title = title
            price = as_number(self.store_schema.record_field(record, "price"))
            if price is not None:
                candidate.price = price
            stock = self.store_schema.stock_value(record)
            if stock is not None:
                candidate.in_stock = stock
                if stock is False:
                    candidate.unavailable = True
            category = self.store_schema.record_field(record, "category")
            if isinstance(category, str) and category:
                candidate.category = category
            brand = self.store_schema.record_field(record, "brand")
            if isinstance(brand, str) and brand:
                candidate.brand = brand
            description = self.store_schema.record_field(record, "description")
            if isinstance(description, str) and description:
                candidate.description = description
            facts = self.store_schema.facts_of(record)
            if facts:
                candidate.attributes.update(facts)
            if detail_tool or facts:
                candidate.detail_loaded = True
            candidate.raw_data = clip(record, 300)

    # ========================================================
    # 18. CANDIDATE SCREENING
    # ========================================================

    def within_budget(self, candidate: ProductCandidate, item: ShoppingItem,
                      context: ShoppingContext) -> bool:

        if context.budget is None or candidate.price is None:
            return True
        if context.budget_scope == "each":
            return candidate.price <= context.budget + 1e-9
        others = sum(
            (context.candidates[i.chosen_id].price or 0) * i.quantity
            for i in context.items
            if i is not item and i.chosen_id and i.chosen_id in context.candidates
        )
        return others + candidate.price * item.quantity <= context.budget + 1e-9

    def buyable(self, candidate: ProductCandidate, item: ShoppingItem,
                context: ShoppingContext) -> bool:

        if candidate.unavailable:
            return False
        if context.in_stock_only and candidate.in_stock is False:
            return False
        return self.within_budget(candidate, item, context)

    async def screen_candidates(self, context: ShoppingContext) -> None:

        pending = [c for c in context.candidates.values() if c.screened is None]
        if not pending:
            return
        batch = pending[:80]
        ids_by_key = {f"R{n}": c.candidate_id for n, c in enumerate(batch, 1)}
        rows = "\n".join(
            f"{key} | {c.title[:90]} | {c.category or ''} | {c.brand or ''} | "
            f"{c.price if c.price is not None else '?'} | "
            f"{'?' if c.in_stock is None else ('yes' if c.in_stock else 'no')}"
            + (f" | {c.description[:120]}" if c.description else "")
            for key, c in zip(ids_by_key, batch)
        )
        items = "\n".join(
            f"{n}. {i.name}" + (f" (department: {i.category_hint})" if i.category_hint else "")
            for n, i in enumerate(context.items, 1)
        )
        # The model only names row keys; Python builds the JSON with the real ids.
        result = await self.think_json(
            SCREEN_PROMPT.format(items=items, rows=rows), role='agent',
            parser=lambda reply: screen_json(reply, ids_by_key, len(context.items)),
        )
        if not result:
            return  # no usable answer: leave these rows unscreened for the next pass
        kept: dict[str, int] = {}
        for entry in result.get("items") or []:
            if not isinstance(entry, dict):
                continue
            number = as_int(entry.get("item")) or 1
            for ident in entry.get("ids") or []:
                if isinstance(ident, str) and ident in context.candidates:
                    kept.setdefault(ident, number - 1)
        for candidate in batch:
            if candidate.candidate_id in kept:
                candidate.screened = True
                candidate.item_index = min(max(kept[candidate.candidate_id], 0), len(context.items) - 1)
                continue
            candidate.screened = False
            # A row whose own text names the product and every required value is
            # a plausible fit whatever the screener said: keep it for the datasheet check.
            for index, item in enumerate(context.items):
                if self.names_everything(candidate, item):
                    candidate.screened = True
                    candidate.rescued = True
                    candidate.item_index = index
                    break

    @staticmethod
    def _stems(text: str) -> set:

        stop = {"the", "and", "for", "with", "not", "any", "type", "style", "kind", "made",
                "item", "product", "one", "that", "this", "from", "has", "have", "must"}
        out = set()
        for word in re.findall(r"[a-z0-9]+", str(text).lower().replace("colour", "color")):
            if len(word) < 3 or word in stop:
                continue
            out.add(word[:-1] if len(word) > 3 and word.endswith("s") else word)
        return out

    def required_values(self, item: ShoppingItem) -> set:
        """Word stems of the values the item requires (exclusions left out)."""

        wanted = set()
        for requirement in item.requirements:
            req = str(requirement)
            if re.search(r"\bnot\b|\bno\b|\bwithout\b|\bexcept\b", req.lower()):
                continue
            wanted |= self._stems(req.split(":", 1)[-1])
        return wanted

    def relevance(self, candidate: ProductCandidate, item: ShoppingItem) -> int:
        """How many required values (x2) and product-name words the listing's text shows."""

        text = " ".join([
            candidate.title or "", candidate.description or "", candidate.category or "",
            " ".join(str(v) for v in candidate.attributes.values()),
        ])
        have = self._stems(text)
        wanted = self.required_values(item)
        name = self._stems(item.name) - wanted
        return 2 * len(wanted & have) + len(name & have)

    def names_everything(self, candidate: ProductCandidate, item: ShoppingItem) -> bool:
        """Title / description show the product noun and every required value."""

        wanted = self.required_values(item)
        if not wanted:
            return False
        have = self._stems(f"{candidate.title or ''} {candidate.description or ''} {candidate.category or ''}")
        name = self._stems(item.name) - wanted
        return wanted <= have and (not name or bool(name & have))

    def shortlist(self, context: ShoppingContext, index: int) -> list[ProductCandidate]:
        """Plausible, buyable candidates of one item, cheapest first."""

        item = context.items[index]
        pool = [
            c for c in context.candidates.values()
            if c.screened and c.item_index == index and c.fits is not False
            and self.buyable(c, item, context)
        ]
        pool.sort(key=lambda c: (c.price is None, c.price or 0, c.candidate_id))
        context.product_ids = [c.candidate_id for c in pool]
        return pool

    # ========================================================
    # 19. SHOPPER INTERACTION
    # ========================================================

    def question_budget(self, context: ShoppingContext) -> int:

        free = context.message_limit - context.messages_sent - context.approval_messages
        return max(0, min(free, MAX_SHOPPER_INTERACTIONS))

    async def ask_shopper(
        self,
        context: ShoppingContext,
        questions: list[str] | str,
        reserve: int = FINISH_RESERVE,
    ) -> list[str]:
        """Send each question as its own message; returns the replies."""

        context.state = AgentState.ASK_SHOPPER

        if isinstance(questions, str):
            questions = [questions]

        tool = context.roles.get("message")
        if not tool:
            return []
        param = self.registry.text_param(tool)

        answers = []
        for question in questions:
            if context.messages_sent >= context.message_limit:
                break
            if self.steps_left(context) <= reserve or context.episode_finished:
                break
            if param:
                arguments = {param: question}
            else:
                call = await self.tool_call_for(
                    tool, f"Send the shopper exactly this message: {question}"
                )
                if call is None:
                    break
                arguments = call["arguments"]
            try:
                observation, _ = await self.act("message", arguments, context)
            except Exception as exc:
                context.errors.append({"tool": tool, "error": str(exc)})
                break
            answer = observation if isinstance(observation, str) else compact(observation)
            answers.append(answer)
            context.shopper_answers.append({"question": question, "answer": answer})

        return answers

    def attribute_keys(self, context: ShoppingContext) -> str:

        lines = []
        for index, item in enumerate(context.items):
            stats: dict[str, dict[str, list]] = {}
            opened = [c for c in context.candidates.values() if c.item_index == index and c.attributes]
            running = [c for c in opened if c.fits is True]
            for c in running if len(running) >= 3 else opened:
                seen_here = set()
                for key, value in c.attributes.items():
                    name = attribute_name(key)
                    if not name or name in seen_here or key in self.store_schema.known_keys:
                        # a key of the record itself (id, price, stock ...), not a property
                        continue
                    seen_here.add(name)
                    stats.setdefault(name, {}).setdefault(c.candidate_id, []).append(str(value)[:30])
            # a key with one value tells nothing; otherwise widespread and varied first
            def split_power(entry: tuple) -> tuple:
                listings, values = len(entry[1]), len({v[0].lower() for v in entry[1].values()})
                return (values < 2, -listings * min(values, 4), -listings)

            ranked = sorted(stats.items(), key=split_power)[:20]
            lines.append(f"Item {index + 1} ({item.name}):")
            for name, per_listing in ranked:
                examples = list(dict.fromkeys(v[0] for v in per_listing.values()))[:4]
                lines.append(f"  {name}: on {len(per_listing)} listings; e.g. {examples}")
        return "\n".join(lines)

    async def interview(self, context: ShoppingContext) -> None:
        """Ask one property per message, then turn the replies into requirements."""

        budget = self.question_budget(context)
        if budget <= 0 or not self.exploring_allowed(context):
            return
        # keep enough steps to finish after the questions
        budget = min(budget, max(0, self.steps_left(context) - FINISH_RESERVE - 4))
        if budget <= 0:
            return

        asked = [a["question"] for a in context.shopper_answers]
        result = await self.think_json(QUESTION_PROMPT.format(
            request=context.original_request,
            items=self.task_text(context),
            keys=self.attribute_keys(context) or "none",
            asked="\n".join(asked) or "none",
            count=budget,
        ))
        planned = []
        seen_props = set()
        for entry in result.get("questions") or []:
            if not isinstance(entry, dict):
                continue
            question = str(entry.get("question") or "").strip()
            prop = normalize_text(entry.get("property") or question)
            words = self._words(entry.get("property") or "")
            if not question or prop in seen_props or (words and any(
                words <= self._words(q) for q in asked + planned
            )):
                continue   # same property again ("style" after "pullover or cardigan style")
            seen_props.add(prop)
            planned.append(question)
        planned = planned[:budget]
        if not planned:
            return

        before = len(context.shopper_answers)
        await self.ask_shopper(context, planned)
        new = context.shopper_answers[before:]
        if not new or not self.can_think():
            return

        qa = "\n".join(f"Q: {a['question']}\nA: {a['answer']}" for a in new)
        items = "\n".join(f"{n}. {i.name}" for n, i in enumerate(context.items, 1))
        parsed = await self.think_json(ANSWER_PROMPT.format(items=items, qa=qa))
        self.apply_changes(context, parsed.get("requirements"), [], stated_budget(parsed.get("budget"), qa))
        for entry in parsed.get("dont_mind") or []:
            if isinstance(entry, dict):
                index = self.item_index(context, entry.get("item"))
                context.items[index].dont_mind.append(str(entry.get("property") or ""))

    @staticmethod
    def item_index(context: ShoppingContext, value: Any) -> int:

        number = as_int(value) or 1
        return min(max(number - 1, 0), len(context.items) - 1)

    def apply_changes(self, context: ShoppingContext, add: Any, drop: Any, budget: Any) -> bool:

        changed = False
        for entry in add or []:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("requirement") or entry.get("need") or "").strip()
            item = context.items[self.item_index(context, entry.get("item"))]
            if text and text not in item.requirements:
                item.requirements.append(text)
                changed = True
        for entry in drop or []:
            if not isinstance(entry, dict):
                continue
            text = normalize_text(entry.get("requirement") or "")
            if not text:
                continue
            item = context.items[self.item_index(context, entry.get("item"))]
            raw = str(entry.get("requirement") or "")
            dropped = self._stems(raw.split(":", 1)[0] if ":" in raw else raw)

            def same(requirement: str) -> bool:
                # "color family" drops "colour family: green"; "style" drops "style: trench coat"
                req = normalize_text(requirement)
                if req == text or req.startswith(text):
                    return True
                if ":" not in str(requirement) or not dropped:
                    return False
                prop = self._stems(str(requirement).split(":", 1)[0])
                return bool(prop) and (prop <= dropped or dropped <= prop)

            kept = [r for r in item.requirements if not same(r)]
            if len(kept) != len(item.requirements):
                item.requirements = kept
                changed = True
        number = as_number(budget)
        if number is not None and number > 0 and number != context.budget:
            context.budget = number
            changed = True
        if changed:
            context.requirements_version += 1
            for candidate in context.candidates.values():
                if candidate.fits is not None:
                    candidate.fits = None
        return changed

    # ========================================================
    # 20. CANDIDATE DETAILS
    # ========================================================

    async def get_candidate_details(
        self,
        context: ShoppingContext,
        index: int,
        batch: int = DETAIL_BATCH,
    ) -> list[str]:
        """Open the cheapest not-yet-opened shortlisted listings of one item."""

        if context.details_closed:
            return []
        tool_role = "details" if context.roles.get("details") else (
            "listing" if context.roles.get("listing") else None
        )
        item = context.items[index]
        unopened = [c for c in self.shortlist(context, index) if not c.detail_loaded]
        # Rows a keyword search found for the required values first, then those
        # whose text names them (a broad search floods the cheap end), then cheapest.
        unopened.sort(key=lambda c: (-c.value_hits, -self.relevance(c, item), c.price is None, c.price or 0))
        if context.selection == "cheapest":
            # the answer is the lowest price that fits: read in price order
            unopened.sort(key=lambda c: (c.value_hits == 0, c.price is None, c.price or 0))
        pending = unopened[:batch]
        if not tool_role or not pending:
            for c in pending:
                c.detail_loaded = True
            return [c.candidate_id for c in pending]

        tool = context.roles[tool_role]
        param = self.registry.id_param(tool)
        opened = []
        for candidate in pending:
            if not self.exploring_allowed(context):
                break
            if param:
                arguments = {param: candidate.candidate_id}
            else:
                call = await self.tool_call_for(
                    tool, f'Open product "{candidate.candidate_id}" in full.'
                )
                if call is None:
                    break
                arguments = call["arguments"]
            try:
                await self.act(tool_role, arguments, context)
            except Exception as exc:
                context.errors.append({"tool": tool, "error": str(exc)})
                continue
            candidate.detail_loaded = True
            opened.append(candidate.candidate_id)
        return opened

    def steps_to_keep(self, context: ShoppingContext) -> int:
        """Steps the questions, the small search, the order and two replacements still need."""

        return (self.question_budget(context) + context.approval_messages + POST_QUESTION_STEPS
                + FINISH_RESERVE + DECLINE_RESERVE + EXTRA_SWAP_RESERVE)

    def small_search_room(self, context: ShoppingContext) -> int:

        return self.steps_left(context) - AFTER_SEARCH_RESERVE

    async def open_new_rows(self, context: ShoppingContext, index: int) -> None:
        """Screen what the small search found and read as many listings as the steps allow."""

        if self.can_think():
            await self.screen_candidates(context)
        await self.keep_kind_rows(context, index)
        if self.small_search_room(context) > 0:
            await self.get_candidate_details(context, index, self.small_search_room(context))
        await self.read_side_notes(context, "small search")

    async def focused_search(self, context: ShoppingContext, index: int,
                             values: list[str]) -> ProductCandidate | None:
        """A small search for required values the first searches could not know about."""

        item = context.items[index]
        room = self.small_search_room(context)
        search = context.roles.get("search")
        if room < 1 or not search:
            return self.best_fit(context, index)
        prompt = FOCUSED_SEARCH_PROMPT.format(
            count=min(2, room),
            item=item.name,
            hint=f" (department: {item.category_hint})" if item.category_hint else "",
            values="; ".join(str(v) for v in values) or item.name,
            budget=context.budget,
            history="\n".join(
                f"{h['tool']} {compact(h['arguments'])} -> {h['summary']}" for h in context.search_history[-12:]
            ) or "none",
        )
        context.search_closed = context.details_closed = False
        try:
            await self.run_tool_loop(prompt, context, [search], max_calls=min(2, room))
            await self.open_new_rows(context, index)
        finally:
            context.search_closed = context.details_closed = True
        return await self.decide_item(context, index)

    async def sweep_cheaper(self, context: ShoppingContext, index: int) -> ProductCandidate | None:
        """Pick rule "cheapest": look below the current pick's price, while steps allow."""

        for _ in range(CHEAPER_ROUNDS):
            pick = self.best_fit(context, index)
            if context.selection != "cheapest" or pick is None or self.small_search_room(context) <= 0:
                break
            context.search_closed = context.details_closed = False
            try:
                found = await self.category_below_pick(context, index)
                if self.small_search_room(context) > 0 and await self.cheaper_sweep(context, index):
                    found = True
                if found:
                    await self.open_new_rows(context, index)
            finally:
                context.search_closed = context.details_closed = True
            await self.decide_item(context, index)
            if not found or self.best_fit(context, index) is pick:
                break   # nothing cheaper turned up: another look would find the same
        return self.best_fit(context, index)

    def price_below(self, search: str, price: float) -> float | None:
        """The highest price cap under `price` that the search tool accepts."""

        spec = self.registry.properties(search).get(self.cap_param(search) or "") or {}
        below = price - (1 if schema_kind(spec) == "integer" else 0.01)
        if below < 0:
            return None
        return int(math.floor(below)) if schema_kind(spec) == "integer" else below

    async def category_below_pick(self, context: ShoppingContext, index: int) -> bool:
        """List the product's whole category under the pick's price; True when new rows came."""

        best = self.best_fit(context, index)
        search = context.roles.get("search")
        if best is None or best.price is None or not search or not self.cap_param(search) \
                or not self.registry.param(search, "category"):
            return False
        label = (context.items[index].kind_labels or [best.category])[0]
        below = self.price_below(search, best.price)
        if not label or below is None:
            return False
        before = len(context.candidates)
        await self.run_tool_loop(CHEAPER_SEARCH_PROMPT.format(label=label, below=below),
                                 context, [search], max_calls=1)
        return len(context.candidates) > before

    async def search_after_questions(self, context: ShoppingContext, before: list[list]) -> None:
        """
        The small search once the shopper has answered: for preferences the first
        searches could not know, for an item with nothing left that fits, and
        for a cheaper listing when the shopper wants the cheapest.
        """

        for index, item in enumerate(context.items):
            new = [r for r in item.requirements if r not in before[index]]
            if new or self.best_fit(context, index) is None:
                await self.focused_search(context, index, new or item.requirements)
            await self.sweep_cheaper(context, index)

    async def refine_search(self, context: ShoppingContext) -> None:
        """Once per task: when searches found nothing or the pool is thin, search differently."""

        search = context.roles.get("search")
        failed = [h for h in context.search_history if h["summary"].startswith(("0 rows", "error"))]
        thin = any(len(self.shortlist(context, i)) < REFINE_BELOW for i in range(len(context.items)))
        if context.refined or not search or not (failed or thin) \
                or self.steps_left(context) <= self.steps_to_keep(context) + 2:
            return
        context.refined = True

        def line(entry: dict) -> str:
            return f"{entry['tool']} {compact(entry['arguments'])} -> {entry['summary']}"

        labels = sorted({c.category for c in context.candidates.values() if c.category})[:40]
        await self.run_tool_loop(REFINE_SEARCH_PROMPT.format(
            task=self.task_text(context),
            failed="\n".join(line(h) for h in failed) or "none, but too few fitting rows came back",
            history="\n".join(line(h) for h in context.search_history[-12:]),
            labels=", ".join(labels) or "none yet",
        ), context, [search], max_calls=2)
        if self.can_think():
            await self.screen_candidates(context)

    async def widen_search(self, context: ShoppingContext) -> None:
        """
        Result pages are short, so spend part of the spare steps on more rows
        (further pages first, then new searches). The rest is left for reading
        the listings, and all of this happens before the shopper is asked.
        """

        await self.refine_search(context)
        stop_at = context.step_count + int(
            max(0, self.steps_left(context) - self.steps_to_keep(context)) * WIDEN_SHARE
        )
        stalls = 0
        while stalls < 2 and context.step_count < stop_at and self.exploring_allowed(context):
            index = context.step_count % len(context.items)
            if await self.page_more(context, index):
                stalls = 0
                continue
            if stop_at - context.step_count < 3:
                break   # a planned search round can take several steps
            before = len(context.candidates)
            await self.search_products(context)
            stalls = 0 if len(context.candidates) > before else stalls + 1

    async def open_all_details(self, context: ShoppingContext) -> None:
        """
        Read every plausible listing before the shopper is asked anything,
        keeping steps for the questions, the order and one replacement.
        Later phases judge only what was read here.
        """

        def unscreened() -> int:
            return sum(1 for c in context.candidates.values() if c.screened is None)

        while self.can_think() and unscreened():
            before = unscreened()
            await self.screen_candidates(context)
            if unscreened() == before:
                break
        for index in range(len(context.items)):
            self.rescue_screened_out(context, index)
            await self.keep_kind_rows(context, index)

        keep = self.steps_to_keep(context)
        while True:
            opened = []
            for index in range(len(context.items)):
                room = self.steps_left(context) - keep
                if room > 0:
                    opened += await self.get_candidate_details(context, index, min(DETAIL_BATCH, room))
            if not opened:
                break
        context.details_closed = True

    # ========================================================
    # 21. EVALUATION
    # ========================================================

    async def evaluate(self, context: ShoppingContext, index: int) -> None:
        """Judge opened, unjudged candidates of one item against all requirements."""

        context.state = AgentState.EVALUATE
        item = context.items[index]
        await self.learn_kind(context, index)
        unjudged = [c for c in self.shortlist(context, index) if c.detail_loaded and c.fits is None]
        for candidate in unjudged:
            if item.kind_labels and candidate.category and candidate.category not in item.kind_labels:
                candidate.fits = False
                candidate.rejection_reasons.append(f"category {candidate.category!r} is another kind")
        pending = [c for c in unjudged if c.fits is None][:8]
        if not pending:
            return
        requirements = list(item.requirements)
        if not requirements and context.selection not in ("most", "least"):
            # nothing beyond the kind of product, which the category check settled
            for c in pending:
                c.fits = not self.category_conflict(context, c, item)
            return
        keys = {f"L{n}": c for n, c in enumerate(pending, 1)}
        listings = "\n".join(f"{key}: " + compact({
            "title": c.title,
            "category": c.category,
            "brand": c.brand,
            "price": c.price,
            "attributes": clip(c.attributes, 120),
            "description": (c.description or "")[:500],
        }) for key, c in keys.items())
        measure_rule = ""
        if context.selection in ("most", "least"):
            measure_rule = (
                f'- After the marks write "= " and the listing\'s value of '
                f'"{context.selection_measure}" as a number ("= ?" if not stated), '
                'e.g. "L1: Y Y = 42".\n'
            )
        result = await self.think_json(EVALUATE_PROMPT.format(
            item=item.name,
            requirements="\n".join(f"Q{n}. {r}" for n, r in enumerate(requirements, 1)) or "(none)",
            listings=listings,
            measure_rule=measure_rule,
        ), parser=lambda reply: matrix_json(reply, set(keys), len(requirements)))
        verdicts = result.get("verdicts") or {}
        for key, candidate in keys.items():
            verdict = verdicts.get(key)
            if verdict is None:
                continue   # not read: stays unjudged for the next call
            candidate.measure = verdict["measure"]
            # Python combines the marks: one unmet requirement rejects the listing
            failed = [r for r, mark in zip(requirements, verdict["marks"]) if mark == "n"]
            # the model sometimes trusts a title word over the datasheet; an S mark
            # already read the datasheet and found the same meaning in other words
            literal = [r for r, mark in zip(requirements, verdict["marks"]) if mark == "y"]
            clash = self.category_conflict(context, candidate, item) or next(
                (r for r in literal if self.attribute_conflict(candidate, r, item)), None
            )
            candidate.fits = not failed and not clash
            if not candidate.fits:
                reason = f"datasheet contradicts: {clash}" if clash else f"not met: {failed}"
                candidate.rejection_reasons.append(reason[:200])

    async def keep_kind_rows(self, context: ShoppingContext, index: int) -> None:
        """
        The store's category beats a title: a listing filed under the requested
        kind stays in the pool even when its title calls it something else.
        With no requirement left to check on the datasheet the screener still decides.
        """

        item = context.items[index]
        await self.learn_kind(context, index)
        if not item.kind_labels or not item.requirements:
            return
        for candidate in context.candidates.values():
            if candidate.screened is False and candidate.category in item.kind_labels:
                candidate.screened = True
                candidate.rescued = True
                candidate.item_index = index

    async def learn_kind(self, context: ShoppingContext, index: int) -> None:
        """Find once which store categories are the requested kind of product."""

        item = context.items[index]
        if item.kind_labels is not None:
            return
        # a label that says exactly what the shopper asked for settles it
        target = (self._words(item.category_hint) or self._words(item.name)) - {"or"}
        exact = sorted({str(c.category) for c in context.candidates.values()
                        if c.category and self._words(c.category) == target})
        if target and exact:
            item.kind_labels = exact
            return
        if not self.can_think():
            return
        counts: dict[str, int] = {}
        for c in context.candidates.values():
            if c.screened and c.item_index == index and c.category:
                counts[c.category] = counts.get(c.category, 0) + 1
        if len(counts) < 2:
            item.kind_labels = []   # nothing to tell apart
            return
        ordered = sorted(counts, key=lambda label: -counts[label])
        labels_by_key = {f"C{n}": label for n, label in enumerate(ordered, 1)}
        result = await self.think_json(KIND_PROMPT.format(
            item=item.name,
            hint=f' (the shopper calls the department "{item.category_hint}")' if item.category_hint else "",
            labels="\n".join(f"{key}. {label}: {counts[label]}" for key, label in labels_by_key.items()),
        ), max_tokens=800, parser=lambda reply: kind_json(reply, labels_by_key))
        item.kind_labels = result.get("labels") or []

    @staticmethod
    def _words(text: Any) -> set:
        """Plain word stems, keeping 'style' / 'type' (they name properties)."""

        out = set()
        for word in re.findall(r"[a-z0-9]+", str(text).lower().replace("colour", "color")):
            if len(word) < 3 or word in ("the", "and", "for", "with", "of"):
                continue
            out.add(word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word)
        return out

    def attribute_conflict(self, candidate: ProductCandidate, requirement: str, item: ShoppingItem) -> bool:
        """
        True when the datasheet has an attribute for the requirement's property
        and none of those attributes states the required value.
        "style: military" fails a listing whose style attribute says "casual",
        whatever its title says.
        """

        text = str(requirement)
        if ":" not in text or not candidate.attributes:
            return False
        prop, value = text.split(":", 1)
        value = value.strip().lower()
        if not value or re.search(NOT_PLAIN_VALUE, value):
            return False
        prop_words = self._words(prop)
        if not prop_words:
            return False
        common = prop_words | self._words(item.name) | self._words(item.category_hint)
        alternatives = []
        for part in re.split(r"\bor\b|,|/|\beither\b", value):
            words = self._words(part) - prop_words
            # drop the product-kind words ("coat" of "trench coat") unless nothing else is left
            distinct = (words - common) or words
            if distinct:
                alternatives.append(distinct)
        if not alternatives:
            return False   # the value only repeats the property or the product kind
        own = self._words(item.name) | self._words(item.category_hint)
        named = False    # a field whose whole name is the property: "milk type" is not "type"
        stated = []      # also this product's own field: "skirt style" beside "style"
        for key, val in candidate.attributes.items():
            key_words = self._words(attribute_name(key))
            if key_words == prop_words:
                named = True
            if key_words == prop_words or (prop_words < key_words and key_words - prop_words <= own):
                stated.append(str(val).lower().replace("colour", "color"))
        if not named:
            return False   # no attribute for this property: the model's reading stands

        for val in stated:
            words = re.findall(r"[a-z0-9]+", val)
            for alt in alternatives:
                if any(any(words_close(s, w) for w in words) for s in alt):
                    return False
        return True

    def evidence(self, candidate: ProductCandidate, item: ShoppingItem) -> int:
        """
        How firmly the listing backs its match: datasheet fields on a required
        property that state the value, minus those on it that state something
        else, plus one when the title names the value.
        """

        def tokens(text: Any) -> list[str]:
            return re.findall(r"[a-z0-9]+", str(text).lower().replace("colour", "color"))

        score = 0
        title = tokens(candidate.title or "")
        for requirement in item.requirements:
            prop, separator, value = str(requirement).partition(":")
            if not separator or re.search(NOT_PLAIN_VALUE, value.lower()):
                continue
            prop_words = self._words(prop)
            wanted = self._words(value) - prop_words
            if not wanted or not prop_words:
                continue
            seen = set()
            for key, stated in candidate.attributes.items():
                name = normalize_text(attribute_name(key))
                if name in seen or not any(p in normalize_text(key) for p in prop_words):
                    continue
                seen.add(name)
                words = tokens(stated)
                agrees = all(any(words_close(w, s) for s in words) for w in wanted)
                score += 1 if agrees else -1
            if all(any(words_close(w, t) for t in title) for w in wanted):
                score += 1
        return score

    def category_conflict(self, context: ShoppingContext, candidate: ProductCandidate,
                          item: ShoppingItem) -> str:
        """
        When a store category label says exactly what the shopper asked for
        ("pants or jeans" -> "Pants & Jeans"), listings filed elsewhere ("Jeans")
        are a different kind of product.
        """

        target = self._words(item.category_hint) or self._words(item.name)
        target -= {"or"}
        if not target or not candidate.category:
            return ""
        labels = {str(c.category) for c in context.candidates.values() if c.category}
        exact = [label for label in labels if self._words(label) == target]
        if not exact:
            return ""
        if self._words(candidate.category) == target:
            return ""
        return f"category {candidate.category!r} is not {exact[0]!r}"

    def best_fit(self, context: ShoppingContext, index: int) -> ProductCandidate | None:

        fits = [c for c in self.shortlist(context, index) if c.fits is True]
        if not fits:
            return None
        if context.selection in ("most", "least"):
            measured = [c for c in fits if c.measure is not None]
            if measured:
                pick = max if context.selection == "most" else min
                best = pick(c.measure for c in measured)
                tied = [c for c in measured if c.measure == best]
                return min(tied, key=lambda c: (c.price is None, c.price or 0))
        if context.selection == "cheapest":
            return fits[0]  # shortlist is cheapest-first
        # No price rule: the listing whose datasheet backs the match most firmly
        # (the first of equals, so the cheapest).
        item = context.items[index]
        return max(fits, key=lambda c: self.evidence(c, item))

    def needs_more_looking(self, context: ShoppingContext, index: int) -> bool:
        """True while a cheaper or better fit could still be among unopened listings."""

        pool = self.shortlist(context, index)
        best = self.best_fit(context, index)
        unopened = [c for c in pool if not c.detail_loaded]
        if best is None:
            return bool(unopened)
        if context.selection in ("most", "least"):
            return bool(unopened)
        if context.selection == "cheapest" and best.price is not None:
            return any(c.price is not None and c.price < best.price for c in unopened)
        return False

    async def decide_item(self, context: ShoppingContext, index: int) -> ProductCandidate | None:
        """Open, judge and repeat until the item's pick is settled."""

        for _ in range(MAX_REFINEMENT_ROUNDS):
            if self.can_think():
                await self.evaluate(context, index)
            if not self.needs_more_looking(context, index):
                break
            if not self.exploring_allowed(context):
                break
            opened = await self.get_candidate_details(context, index)
            if not opened:
                break

        # evaluate() takes 8 listings a call: go on until every opened one is judged
        def unjudged() -> int:
            return sum(1 for c in self.shortlist(context, index) if c.detail_loaded and c.fits is None)

        while self.can_think() and unjudged():
            before = unjudged()
            await self.evaluate(context, index)
            if unjudged() == before:
                break
        return self.best_fit(context, index)

    def rescue_screened_out(self, context: ShoppingContext, index: int) -> int:
        """Give buyable rows the screener dropped, but whose text names the required values, a second look."""

        item = context.items[index]
        need = 2 if any(":" in str(r) for r in item.requirements) else 1
        pool = [
            c for c in context.candidates.values()
            if c.screened is False and not c.rescued and c.fits is not False
            and self.buyable(c, item, context) and self.relevance(c, item) >= need
        ]
        pool.sort(key=lambda c: (-self.relevance(c, item), c.price is None, c.price or 0))
        for candidate in pool[:RESCUE_BATCH]:
            candidate.rescued = True
            candidate.screened = True
            candidate.item_index = index
        return len(pool[:RESCUE_BATCH])

    async def page_more(self, context: ShoppingContext, index: int) -> bool:
        """Read the next page of the most relevant search that still has one."""

        item = context.items[index]
        wanted = set()
        for requirement in item.requirements:
            wanted |= self._stems(str(requirement).split(":", 1)[-1])
        wanted |= self._stems(item.name)
        run = {(h["tool"], compact(h["arguments"]).lower()) for h in context.search_history}
        options = [
            h for h in context.search_history
            if h.get("next_args") and (h["tool"], compact(h["next_args"]).lower()) not in run
            and "error" not in h["summary"]
        ]
        if not options:
            return False
        options.sort(key=lambda h: -len(self._stems(compact(h["arguments"])) & wanted))
        entry = options[0]
        before = len(context.candidates)
        try:
            observation = await self.execute_tool(
                {"name": entry["tool"], "arguments": entry["next_args"]}, context
            )
        except Exception as exc:
            context.errors.append({"tool": entry["tool"], "error": str(exc)})
            entry["next_args"] = None
            return False
        self.record_search(entry["tool"], entry["next_args"], observation, context)
        return len(context.candidates) > before

    async def keep_looking(self, context: ShoppingContext, index: int) -> ProductCandidate | None:
        """
        Nothing fits yet. Declining is right only when nothing in the store
        fits, so spend the remaining steps first: open more listings and give
        screened-out rows a second look. Once the details are closed, only
        the listings already read are judged again.
        """

        if context.details_closed:
            return await self.decide_item(context, index)
        stalls = 0
        # Leave room for a refused decline: the store often answers it by
        # relaxing a rule, and then cart + checkout are still needed.
        floor = 0 if context.decline_refusals else DECLINE_RESERVE
        while stalls < 3 and self.exploring_allowed(context) and self.steps_left(context) > FINISH_RESERVE + floor:
            self.rescue_screened_out(context, index)   # no step; opened first (relevance order)
            best = await self.decide_item(context, index)
            if best is not None:
                return best
            if not self.exploring_allowed(context):
                break
            if any(c.screened is None for c in context.candidates.values()):
                before = sum(1 for c in context.candidates.values() if c.screened is None)
                await self.screen_candidates(context)
                after = sum(1 for c in context.candidates.values() if c.screened is None)
                if after < before:
                    continue
            if self.rescue_screened_out(context, index):
                continue
            if await self.page_more(context, index):
                stalls = 0
                continue
            before = len(context.candidates)
            await self.search_products(context)
            if len(context.candidates) == before:
                stalls += 1
        return self.best_fit(context, index)

    async def cheaper_sweep(self, context: ShoppingContext, index: int) -> bool:
        """
        Pick rule "cheapest": search results are often truncated and in no price
        order.  Re-run each truncated search with its price cap just below the
        current pick, so every cheaper listing of that search becomes visible.
        Returns True when new rows were found.
        """

        best = self.best_fit(context, index)
        search = context.roles.get("search")
        if context.search_closed or context.selection != "cheapest" or best is None \
                or best.price is None or not search:
            return False
        cap = self.cap_param(search)
        if not cap:
            return False
        page = self.registry.param(search, "page")
        spec = self.registry.properties(search).get(cap) or {}
        below = best.price - (1 if schema_kind(spec) == "integer" else 0.01)
        if below < 0:
            return False
        if schema_kind(spec) == "integer":
            below = int(math.floor(below))

        before = len(context.candidates)
        swept = 0
        tried = {compact(h["arguments"]) for h in context.search_history}
        for entry in list(context.search_history):
            if swept >= 2 or not self.exploring_allowed(context):
                break
            if entry["tool"] != search or "error" in entry["summary"] or page in entry["arguments"]:
                continue
            if self.mode_signature(search, entry["arguments"]) in context.cap_ignored:
                continue   # this search mode ignores the price cap: sweeping it is wasted
            found = re.match(r"(\d+) rows of (\d+)", entry["summary"])
            base = {k: v for k, v in entry["arguments"].items() if k != page}
            pages_read = sum(
                1 for h in context.search_history
                if h["tool"] == search and {k: v for k, v in h["arguments"].items() if k != page} == base
            )
            if not found or int(found.group(2)) <= int(found.group(1)) * pages_read:
                continue   # the search showed everything already
            arguments = dict(base)
            arguments[cap] = below
            fixed, problems = validate_arguments(arguments, self.registry.schema(search))
            if problems or compact(fixed) in tried:
                continue
            tried.add(compact(fixed))
            swept += 1
            try:
                observation = await self.execute_tool({"name": search, "arguments": fixed}, context)
            except Exception as exc:
                context.errors.append({"tool": search, "error": str(exc)})
                continue
            self.record_search(search, fixed, observation, context)
            prices = [
                as_number(self.store_schema.record_field(r, "price"))
                for r in self.store_schema.records_in(observation)
            ]
            if any(p is not None and p > below for p in prices):
                continue   # this search mode ignores the cap: paging it is wasted
            await self.follow_pages({"name": search, "arguments": fixed}, observation, context)
        return len(context.candidates) > before

    async def shopper_decision_loop(
        self,
        context: ShoppingContext
    ) -> dict:
        """
        Search until every item has candidates, read all their details, then
        interview and judge. After the questions only a small, targeted search runs.
        """

        interviewed = False
        for round_number in range(MAX_SEARCH_ROUNDS):
            missing = [i for i, item in enumerate(context.items) if not self.best_fit(context, i)]
            if not missing or not self.exploring_allowed(context):
                break
            await self.search_products(context)
            if self.can_think():
                await self.screen_candidates(context)
            if not all(self.shortlist(context, i) for i in missing) and round_number + 1 < MAX_SEARCH_ROUNDS:
                continue
            await self.widen_search(context)
            context.search_closed = True
            await self.open_all_details(context)
            await self.read_side_notes(context, "product research")
            before = [list(item.requirements) for item in context.items]
            if not interviewed and self.question_budget(context) > 0:
                # judge first, so the questions split the listings still in the running
                for index in missing:
                    await self.decide_item(context, index)
                interviewed = True
                await self.interview(context)
                await self.read_side_notes(context, "shopper questions")
            for index in missing:
                await self.decide_item(context, index)
            await self.search_after_questions(context, before)
            break

        # nothing fits yet: use the remaining steps before settling for a decline
        for index in range(len(context.items)):
            if self.best_fit(context, index) is None:
                await self.keep_looking(context, index)

        if not interviewed and self.question_budget(context) > 0 and any(
            self.best_fit(context, i) for i in range(len(context.items))
        ):
            await self.interview(context)
            for index in range(len(context.items)):
                await self.decide_item(context, index)
                if self.best_fit(context, index) is None:
                    # a requirement learned from the shopper ruled the pick out
                    await self.keep_looking(context, index)

        for index, item in enumerate(context.items):
            best = self.best_fit(context, index)
            item.chosen_id = best.candidate_id if best else None
        return {
            "chosen": [item.chosen_id for item in context.items],
            "reason": "fits found" if all(i.chosen_id for i in context.items) else "no fit",
        }

    # ========================================================
    # 22. CART MANAGEMENT
    # ========================================================

    def cart_args(self, context: ShoppingContext, product_id: str, quantity: int,
                  remove: bool = False) -> dict | None:

        tool = context.roles.get("cart_edit")
        if not tool:
            return None
        id_param = self.registry.id_param(tool)
        if not id_param:
            return None
        arguments: dict = {id_param: product_id}
        if self.registry.mode_param(tool):
            chosen = self.registry.mode_value(tool, remove)
            if chosen is None:
                return None   # the model builds the call instead
            key, value = chosen
            arguments[key] = value
        qty_param = self.registry.quantity_param(tool)
        if qty_param:
            arguments[qty_param] = quantity
        elif quantity != 1 and not remove:
            return None
        fixed, problems = validate_arguments(arguments, self.registry.schema(tool))
        return None if problems else fixed

    async def change_cart(self, context: ShoppingContext, product_id: str, quantity: int,
                          remove: bool = False) -> list[str]:

        arguments = self.cart_args(context, product_id, quantity, remove)
        if arguments is None:
            call = await self.tool_call_for(context.roles["cart_edit"], CART_PROMPT.format(
                action="remove" if remove else "add",
                product_id=product_id,
                quantity="" if remove else f" with quantity {quantity}",
            ))
            if call is None or product_id not in compact(call["arguments"]):
                raise RuntimeError("could not build a cart call")
            arguments = call["arguments"]
        observation, notes = await self.act("cart_edit", arguments, context)
        context.cart_dirty = True
        context.cart_result = observation
        context.cart_observations.append({"action": "remove" if remove else "add",
                                          "product_id": product_id, "result": observation})
        problem = self.store_schema.error_in(observation)
        if problem and not remove:
            notes = notes + [f"cart refused: {problem}"]
        elif remove:
            context.cart.pop(product_id, None)
        else:
            context.cart[product_id] = context.cart.get(product_id, 0) + quantity
        return notes

    async def view_cart(self, context: ShoppingContext) -> list[str]:

        if not context.roles.get("cart_view") or self.steps_left(context) <= 2:
            return []
        observation, notes = await self.act("cart_view", {}, context)
        context.cart_dirty = False
        context.cart_result = observation
        context.cart_observations.append({"action": "view", "result": observation})
        lines = {}
        for record in self.store_schema.records_in(observation):
            ident = str(record.get(self.store_schema.id_key_of(record)) or "")
            qty = as_int(self.store_schema.record_field(record, "quantity")) or 1
            if ident:
                lines[ident] = lines.get(ident, 0) + qty
        # An empty view of a layout we can read means the add did not land; the
        # order check must see that instead of trusting the earlier add.
        if lines or self.store_schema.shape_of(observation) <= self.store_schema.known_keys:
            context.cart = lines
        return notes

    async def manage_cart(self, context: ShoppingContext) -> list[str]:
        """Make the cart hold exactly the chosen lines; returns new notices."""

        context.state = AgentState.PURCHASE
        notes: list[str] = []
        wanted = {item.chosen_id: item.quantity for item in context.items if item.chosen_id}
        for product_id in list(context.cart):
            if product_id not in wanted and self.steps_left(context) > 2:
                notes += await self.change_cart(context, product_id, context.cart[product_id], remove=True)
        for product_id, quantity in wanted.items():
            have = context.cart.get(product_id, 0)
            if have == quantity or self.steps_left(context) <= 1:
                continue
            if have > quantity:
                notes += await self.change_cart(context, product_id, have - quantity, remove=True)
            else:
                notes += await self.change_cart(context, product_id, quantity - have)
        if self.steps_left(context) > 2 and context.cart_dirty:
            notes += await self.view_cart(context)   # once per change, not on every retry
        return notes

    # ========================================================
    # 23. NOTICES AND CHANGES
    # ========================================================

    async def handle_notices(self, context: ShoppingContext, notes: list[str], action: str) -> bool:
        """
        Apply store / shopper notes. Returns True when the cart choice changed
        (a sold-out pick, a new budget or new requirements).
        """

        if not notes:
            return False
        changed = False
        parsed: dict = {}
        if self.can_think():
            parsed = await self.think_json(NOTICE_PROMPT.format(
                notes="\n".join(f"- {n[:400]}" for n in notes[-6:]),
                action=action,
                cart=list(context.cart) or [i.chosen_id for i in context.items if i.chosen_id],
                items=self.task_text(context),
                budget=context.budget,
            ))
        else:
            # no model left to read the note: re-check the cart's stock instead of guessing
            parsed = {"cart_item_unavailable": bool(notes)}

        for ident in parsed.get("unavailable_ids") or []:
            if isinstance(ident, str) and ident in context.candidates:
                context.candidates[ident].unavailable = True
                context.candidates[ident].in_stock = False
                changed = True
        if parsed.get("cart_item_unavailable"):
            gone = await self.verify_cart_stock(context)
            changed = changed or gone
        budget = stated_budget(parsed.get("budget"), "\n".join(notes))
        if self.apply_changes(context, parsed.get("add"), parsed.get("drop"), budget):
            changed = True
        return changed

    async def verify_cart_stock(self, context: ShoppingContext) -> bool:
        """Find which chosen product went away; re-open it when the store can show stock."""

        chosen = [i.chosen_id for i in context.items if i.chosen_id]
        if len(chosen) == 1:
            context.candidates[chosen[0]].unavailable = True
            return True
        role = "listing" if context.roles.get("listing") else None
        found = False
        for ident in chosen:
            if not role or self.steps_left(context) <= FINISH_RESERVE:
                break
            param = self.registry.id_param(context.roles[role])
            if not param:
                break
            await self.act(role, {param: ident}, context)
            if context.candidates[ident].in_stock is False:
                context.candidates[ident].unavailable = True
                found = True
        if not found and chosen:
            # cannot tell which one: drop the most recently added line
            context.candidates[chosen[-1]].unavailable = True
            found = True
        return found

    async def rechoose(self, context: ShoppingContext) -> None:
        """Pick again after a change, opening more listings when needed."""

        for index, item in enumerate(context.items):
            current = context.candidates.get(item.chosen_id or "")
            if current is not None and current.fits and self.buyable(current, item, context) \
                    and self.best_fit(context, index) is current:
                continue
            replaced = item.chosen_id
            item.chosen_id = None   # so the budget check does not count the old pick
            best = await self.decide_item(context, index)
            if best is None:
                # the pick sold out / a rule changed: look further before giving up
                best = await self.keep_looking(context, index)
            if context.details_closed:
                # SWAP: no replacement, or one the datasheet does not back, is worth a small search
                if best is None or (item.requirements and self.evidence(best, item) <= 0):
                    best = await self.focused_search(context, index, item.requirements)
                if best is not None:
                    best = await self.sweep_cheaper(context, index)
            item.chosen_id = best.candidate_id if best else None
            if replaced and replaced != item.chosen_id:
                self.record_check(context, "SWAP", best is not None, f"{replaced} -> {item.chosen_id}")

    # ========================================================
    # 26. PURCHASE EXECUTION
    # ========================================================

    def fixed_finish_args(self, context: ShoppingContext, role: str, remark: str = "") -> dict | None:

        tool = context.roles.get(role)
        if not tool:
            return None
        arguments: dict = {}
        if remark:
            param = self.registry.text_param(tool)
            if param:
                arguments[param] = remark
        fixed, problems = validate_arguments(arguments, self.registry.schema(tool))
        return None if problems else fixed

    async def finish_call(self, context: ShoppingContext, role: str, remark: str = "") -> Any:

        tool = context.roles.get(role)
        arguments = self.fixed_finish_args(context, role, remark)
        if arguments is None:
            call = await self.tool_call_for(tool, FINISH_PROMPT.format(
                tool=tool,
                action="check out the cart" if role == "checkout" else "end without a purchase",
                extra=f' Remark: "{remark}"' if remark else "",
            ))
            if call is None:
                raise RuntimeError(f"could not build the {role} call")
            arguments = call["arguments"]
        observation, notes = await self.act(role, arguments, context)
        return observation, notes

    async def request_approval(self, context: ShoppingContext) -> list[str]:
        """Ask for the approval the request requires, once per order that needs it."""

        wanted = {item.chosen_id: item.quantity for item in context.items if item.chosen_id}
        if not context.approval_steps or context.messages_sent >= context.message_limit:
            return []
        if context.approval_status and wanted == context.approval_lines:
            return []
        lines = []
        for item in context.items:
            c = context.candidates.get(item.chosen_id or "")
            if c:
                lines.append(f"{c.title} ({item.quantity} x {c.price})")
        message = (
            "Before I place the order: I found " + "; ".join(lines)
            + ". Is it okay for me to go ahead and buy this?"
        )
        answers = await self.ask_shopper(context, [message], reserve=1)   # 1 step for checkout
        if answers:
            context.approval_lines = wanted
        return answers

    async def obtain_approval(self, context: ShoppingContext) -> bool:
        """Get and read the shopper's approval. True when the pick must be chosen again."""

        answers = await self.request_approval(context)
        if not answers:
            return False
        question = context.shopper_answers[-1]["question"]
        context.approval_status = "no_answer"
        if not self.can_think():
            return False
        parsed = await self.think_json(
            APPROVAL_PROMPT.format(question=question, reply=answers[0]), role="agent",
            parser=approval_json,
        )
        context.approval_status = parsed.get("approval", "no_answer")
        changed = await self.handle_notices(context, answers, "approval question")
        if context.approval_status == "refused":
            for item in context.items:
                candidate = context.candidates.get(item.chosen_id or "")
                if candidate is not None:
                    candidate.fits = False
                    candidate.rejection_reasons.append("shopper refused it at approval")
            changed = True
        return changed

    @staticmethod
    def record_check(context: ShoppingContext, name: str, ok: bool, detail: Any = "") -> None:
        """Log one CHECK / AUDIT / TRAIL / GATE / SWAP outcome for the final result."""

        context.checks.append({"check": name, "ok": ok, "step": context.step_count,
                               "detail": str(detail)[:300]})

    def check_cart(self, context: ShoppingContext) -> list[str]:
        """CHECK: is the cart exactly the right lines, confirmed, in stock and in budget?"""

        problems: list[str] = []
        chosen = {item.chosen_id for item in context.items if item.chosen_id}
        total = 0.0
        for item in context.items:
            candidate = context.candidates.get(item.chosen_id or "")
            if candidate is None:
                problems.append(f"no product chosen for {item.name}")
                continue
            have = context.cart.get(candidate.candidate_id, 0)
            if have == 0:
                problems.append(f"{item.name} is not in the cart")
            elif have != item.quantity:
                problems.append(f"{item.name}: cart has {have}, requested {item.quantity}")
            if candidate.fits is not True:
                problems.append(f"{item.name}: pick is not confirmed against the requirements")
            if candidate.unavailable or (context.in_stock_only and candidate.in_stock is False):
                problems.append(f"{item.name}: pick is unavailable")
            if item.kind_labels and candidate.category and candidate.category not in item.kind_labels:
                problems.append(f"{item.name}: pick is filed under {candidate.category!r}")
            if candidate.price is not None:
                total += candidate.price * item.quantity
                if context.budget is not None and context.budget_scope == "each" \
                        and candidate.price > context.budget + 1e-9:
                    problems.append(f"{item.name}: price {candidate.price} is over budget {context.budget}")
        for product_id in context.cart:
            if product_id not in chosen:
                problems.append(f"cart holds an unrequested product {product_id}")
        if context.budget is not None and context.budget_scope == "total" \
                and total > context.budget + 1e-9:
            problems.append(f"order total {total} is over budget {context.budget}")
        return problems

    def gate(self, context: ShoppingContext) -> list[str]:
        """GATE: did every interaction the request requires happen, for this very order?"""

        problems: list[str] = []
        if context.approval_steps and context.message_limit > 0:
            wanted = {item.chosen_id: item.quantity for item in context.items if item.chosen_id}
            if not context.approval_status:
                problems.append("shopper approval was not requested")
            elif context.approval_status == "refused":
                problems.append("shopper refused the order")
            elif APPROVAL_NEEDS_YES and context.approval_status != "approved":
                problems.append("shopper did not approve the order")
            elif wanted != context.approval_lines and context.messages_sent < context.message_limit \
                    and self.steps_left(context) > 1:
                problems.append("shopper approved a different order")
        return problems

    def verify_order(self, context: ShoppingContext) -> list[str]:
        """CHECK and GATE together; empty means ready for checkout."""

        cart, shopper = self.check_cart(context), self.gate(context)
        self.record_check(context, "CHECK", not cart, cart)
        if context.approval_steps:
            self.record_check(context, "GATE", not shopper, shopper)
        return cart + shopper

    async def trail(self, context: ShoppingContext) -> bool:
        """TRAIL: do the picks hold together? True when one of them must be chosen again."""

        lines = {item.chosen_id: item.quantity for item in context.items if item.chosen_id}
        if len(context.items) < 2 or lines == context.trail_lines:
            return False
        chosen = [item.chosen_id for item in context.items]
        repeated = next((i for i in range(1, len(chosen)) if chosen[i] in chosen[:i]), None)
        item_number = None if repeated is None else repeated + 1
        passed = repeated is None
        if passed and context.cross_requirements and self.can_think():
            picks = []
            for number, item in enumerate(context.items, 1):
                c = context.candidates.get(item.chosen_id or "")
                if c is not None:
                    picks.append(f"Item {number} ({item.name}): " + compact({
                        "title": c.title, "brand": c.brand, "category": c.category,
                        "price": c.price, "attributes": clip(c.attributes, 120),
                    }))
            parsed = await self.think_json(TRAIL_PROMPT.format(
                requirements="\n".join(f"- {r}" for r in context.cross_requirements),
                picks="\n".join(picks),
            ), parser=trail_json)
            passed = parsed.get("ok", True)
            item_number = parsed.get("item")
        self.record_check(context, "TRAIL", passed, f"replace item {item_number}" if not passed else "")
        if passed:
            context.trail_lines = lines
            return False
        index = self.item_index(context, item_number or len(context.items))
        candidate = context.candidates.get(context.items[index].chosen_id or "")
        if candidate is not None:
            candidate.fits = False
            candidate.rejection_reasons.append("does not go with the other picks")
        return True

    async def second_opinion(self, context: ShoppingContext, index: int) -> None:
        """AUDIT: reread the near misses; a listing that truly meets everything is reinstated."""

        item = context.items[index]
        settled = ("datasheet contradicts", "another kind", "shopper refused", "go with the other")
        rejected = [
            c for c in context.candidates.values()
            if c.item_index == index and c.detail_loaded and c.fits is False
            and self.buyable(c, item, context)
            and not any(s in reason for reason in c.rejection_reasons for s in settled)
        ]
        rejected.sort(key=lambda c: (-self.evidence(c, item), c.price is None, c.price or 0))
        keys = {f"L{n}": c for n, c in enumerate(rejected[:AUDIT_LISTINGS], 1)}
        if not keys or not item.requirements or not self.can_think():
            return
        listings = "\n".join(f"{key}: " + compact({
            "title": c.title, "category": c.category,
            "attributes": clip(c.attributes, 120), "description": (c.description or "")[:300],
        }) + f" | rejected: {c.rejection_reasons[-1] if c.rejection_reasons else 'unknown'}"
            for key, c in keys.items())
        parsed = await self.think_json(AUDIT_PROMPT.format(
            item=item.name,
            requirements="\n".join(f"- {r}" for r in item.requirements),
            listings=listings,
        ), parser=lambda reply: audit_json(reply, set(keys)))
        for key in parsed.get("keys") or []:
            keys[key].fits = True

    async def audit_decline(self, context: ShoppingContext) -> bool:
        """
        AUDIT: before ending without a purchase, challenge that decision once:
        judge what is unjudged, search a little, and reread the near misses.
        True when every item now has a pick.
        """

        if context.audited or self.steps_left(context) < 3:
            return False
        context.audited = True
        for index, item in enumerate(context.items):
            best = await self.decide_item(context, index)
            if best is None:
                best = await self.focused_search(context, index, item.requirements)
            if best is None:
                await self.second_opinion(context, index)
                best = self.best_fit(context, index)
            item.chosen_id = best.candidate_id if best else None
        overturned = all(item.chosen_id for item in context.items)
        self.record_check(context, "AUDIT", overturned, [i.chosen_id for i in context.items])
        return overturned

    async def purchase(
        self,
        context: ShoppingContext
    ):
        """Cart -> (approval) -> checkout, recovering from sell-outs and refusals."""

        context.state = AgentState.PURCHASE

        while not context.episode_finished and self.steps_left(context) > 0:
            if not all(item.chosen_id for item in context.items):
                return await self.decline(context)

            if await self.trail(context):
                if context.swaps >= MAX_SWAPS:
                    break
                context.swaps += 1
                await self.rechoose(context)
                continue

            notes = await self.manage_cart(context)
            if notes and await self.handle_notices(context, notes, "cart update"):
                if context.swaps >= MAX_SWAPS:
                    break
                context.swaps += 1
                await self.rechoose(context)
                continue

            # Last gate before checkout: approval, then the whole order is verified.
            changed = await self.obtain_approval(context)
            problems = [] if changed else self.verify_order(context)
            if changed or problems:
                if problems:
                    context.errors.append({"phase": "verify_order", "problems": problems})
                if context.swaps >= MAX_SWAPS:
                    break
                context.swaps += 1
                await self.rechoose(context)
                continue

            if self.steps_left(context) <= 0 or context.episode_finished:
                break
            observation, notes = await self.finish_call(context, "checkout")
            context.order_result = {"checkout": observation}
            if context.episode_finished:
                context.finish_kind = "order"
                return observation

            # refused: learn why, fix the cart and try again
            reasons = notes or [f"checkout refused: {compact(observation)[:300]}"]
            await self.handle_notices(context, reasons, "checkout")
            if context.swaps >= MAX_SWAPS:
                break
            context.swaps += 1
            await self.rechoose(context)

        return await self.ensure_finished(context)

    async def decline(self, context: ShoppingContext) -> Any:

        context.state = AgentState.DECLINE
        if context.episode_finished or not context.roles.get("decline"):
            return None
        if await self.audit_decline(context):
            return await self.purchase(context)
        for product_id in list(context.cart):
            if self.steps_left(context) > 1:
                try:
                    await self.change_cart(context, product_id, context.cart[product_id], remove=True)
                except Exception as exc:
                    context.errors.append({"tool": "cart", "error": str(exc)})
        names = ", ".join(item.name for item in context.items)
        remark = context.decline_note or (
            f"Sorry, I could not find a {names} in stock that meets every requirement within the budget."
        )
        remark = strip_ids(remark, list(context.candidates))
        observation, notes = await self.finish_call(context, "decline", remark)
        if context.episode_finished:
            context.finish_kind = "decline"
            return observation

        # Refused: the store / shopper changed something (a requirement dropped,
        # a new budget). Apply it, choose again and buy when something now fits.
        context.decline_refusals += 1
        reasons = notes or [f"decline refused: {compact(observation)[:300]}"]
        await self.handle_notices(context, reasons, "decline (end without purchase)")
        # cart + checkout need 2 steps (the cart view is skipped when steps are short)
        if context.decline_refusals <= MAX_DECLINE_REFUSALS and self.steps_left(context) >= 2:
            await self.rechoose(context)
            if all(item.chosen_id for item in context.items):
                return await self.purchase(context)
        if not context.episode_finished and self.steps_left(context) > 0:
            observation, _ = await self.finish_call(context, "decline", remark)
            if context.episode_finished:
                context.finish_kind = "decline"
        return observation

    async def ensure_finished(self, context: ShoppingContext) -> Any:
        """Last resort: check out a cart that passes verification, otherwise decline."""

        if context.episode_finished or self.steps_left(context) <= 0:
            return None
        try:
            if context.cart and context.roles.get("checkout") and not self.verify_order(context):
                observation, _ = await self.finish_call(context, "checkout")
                if context.episode_finished:
                    context.finish_kind = "order"
                    return observation
                if self.steps_left(context) <= 0:
                    return observation
            return await self.decline(context)
        except Exception as exc:
            context.errors.append({"phase": "ensure_finished", "error": str(exc)})
            return None

    async def run(
        self,
        user_request: str,
        opening: Any = None,
    ) -> dict:

        self.messages = []
        if opening is not None:
            user_request += "\n\nSTORE OPENING NOTE:\n" + compact(clip(opening, 1500))
        context = ShoppingContext(

            request_id=str(uuid.uuid4()),

            original_request=user_request,

            step_limit=self.max_steps,

        )
        self.current_context = context

        try:

            # --------------------------------------------
            # STEP 1: Analyze request
            # --------------------------------------------

            await self.analyze_request(context)
            if opening is not None:
                self.absorb(opening, "opening", context)
                notes = self.store_schema.notices_in(opening)
                if notes:
                    context.notices.extend(notes)
                    await self.handle_notices(context, notes, "task opening")

            # --------------------------------------------
            # STEP 2-5: search, open, interview, decide
            # --------------------------------------------

            await self.shopper_decision_loop(context)

            # --------------------------------------------
            # STEP 6: order (or decline when nothing fits)
            # --------------------------------------------

            if all(item.chosen_id for item in context.items):
                await self.purchase(context)
            else:
                await self.decline(context)

        except Exception as exc:

            logger.exception(
                "Shopping agent execution failed"
            )

            context.errors.append(f"{type(exc).__name__}: {exc}"[:400])

        finally:

            if not context.episode_finished:
                await self.ensure_finished(context)

        context.state = AgentState.FINISHED if context.episode_finished else AgentState.FAILED
        return self.final_result(context)

    # ========================================================
    # 30. FINAL RESPONSE
    # ========================================================

    def final_result(
        self,
        context: ShoppingContext,
    ) -> dict:

        return {

            "request_id": context.request_id,

            "status": context.state.value,

            "finish": context.finish_kind,

            "analysis": context.analysis,

            "roles": context.roles,

            "items": [vars(item) for item in context.items],

            "budget": context.budget,

            "selection": context.selection,

            "plan": context.plan,

            "search_history": context.search_history,

            "shopper_answers": context.shopper_answers,

            "notices": context.notices,

            "cart": context.cart,

            "cart_observations": clip(context.cart_observations, 200),

            "steps": context.step_count,

            "llm_calls": self.llm_calls,

            "llm_cost": round(self.llm_cost, 5),

            "checks": context.checks,

            "errors": context.errors[-MAX_ERRORS:]

        }


# ============================================================
# 31. ADAPTERS AND ENTRY POINT
# ============================================================

def _model(role: str) -> list[str]:

    primary = _DEFAULT_MODELS.get(role) or _DEFAULT_MODELS["agent"]
    chain = [primary, _DEFAULT_MODELS["agent"], *_FALLBACK_MODELS]
    return list(dict.fromkeys(chain))


def _post(client: Any, path: str, body: dict) -> tuple[Any, Any, str]:
    """POST through the proxy: (data, http status on failure, error body)."""

    verbose = getattr(client, "post_verbose", None)
    if callable(verbose):
        result = verbose(path, body)
        error = getattr(result, "error", None) or {}
        return getattr(result, "data", None), error.get("status"), str(error.get("body") or "")
    return client.post(path, json_data=body), None, ""


def rate_limit_delay(waits: int, error_body: str) -> float:
    """Seconds to wait after a 429. Random spread, because many tasks share one key and would retry together."""

    stated = re.search(r"retry[-_ ]?after\D{0,6}(\d+(?:\.\d+)?)", error_body, re.I)
    if stated:
        return float(stated.group(1)) + random.uniform(0, 2)
    return min(RATE_LIMIT_BASE * 2 ** waits, RATE_LIMIT_MAX_DELAY) * random.uniform(1.0, 1.5)


def agent_main(
    problem_data: dict[str, Any]
) -> list[dict[str, Any]]:

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    tool_definitions = policy.get("tools") or []

    request = str(policy.get("query") or "")

    max_steps = as_int(policy.get("max_steps")) or MAX_AGENT_STEPS

    state = {"step": 0, "finished": bool(policy.get("finished"))}
    dialogue: list[dict[str, Any]] = []
    disabled_models: set = set()
    llm_flags = {"reasoning": True}
    started = time.monotonic()
    rate_wait = {"used": 0.0}

    async def post_inference(body: dict) -> tuple[Any, Any, str]:
        """One model request; a 429 is waited out while this call and this task have time for it."""

        waits, waited = 0, 0.0
        while True:
            reply = _post(_proxy, "/inference/chat/completions", body)
            if reply[1] != 429:
                return reply
            delay = rate_limit_delay(waits, reply[2])
            room = min(RATE_LIMIT_CALL_WAIT - waited, RATE_LIMIT_TASK_WAIT - rate_wait["used"],
                       LLM_HARD_STOP - (time.monotonic() - started))
            if delay > room:
                return reply
            waits += 1
            waited += delay
            rate_wait["used"] += delay
            await asyncio.sleep(delay)

    async def llm_step(
        messages: list[dict],
        tools: list[dict],
        role: str = "agent",
        max_tokens: int = 8192,
    ) -> dict:

        request_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *messages
        ]
        last_error = "no model available"
        for model in _model(role):
            if model in disabled_models:
                continue
            claude = model.startswith(CLAUDE_PREFIX)
            quick = REASONING_DEFAULT if claude else REASONING_OFF
            body = {
                "model": model,
                "messages": request_messages,
                "max_tokens": max(max_tokens, CLAUDE_MIN_TOKENS) if claude else max_tokens,
            }
            if not claude:
                body["temperature"] = 0
            if llm_flags["reasoning"]:
                # reading a schema needs no deliberation: answer fast
                body["reasoning"] = dict(quick if role == "mapper" else REASONING_DEFAULT)
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto" if claude else "required"
            for attempt in range(3):
                if time.monotonic() - started > LLM_HARD_STOP:
                    raise RuntimeError("inference skipped: task time nearly used up")
                data, status, error_body = await post_inference(body)

                choices = data.get("choices") if isinstance(data, dict) else None
                if choices and isinstance(choices[0], dict) and isinstance(choices[0].get("message"), dict):
                    message = dict(choices[0]["message"])
                    usage = data.get("usage") or {}
                    if isinstance(usage.get("cost"), (int, float)):
                        message["_cost"] = float(usage["cost"])
                    if message.get("content") or message.get("tool_calls"):
                        return message
                    # Empty reply: the model's hidden reasoning used up the token
                    # limit. More tokens do not help (it reasons on); retry with
                    # reasoning switched off.
                    last_error = "empty reply"
                    if llm_flags["reasoning"]:
                        body["reasoning"] = dict(quick)
                    continue
                last_error = f"status {status}: {error_body[:120]}"
                if status in (400, 422) and "reasoning" in body:
                    # this proxy / provider does not take the reasoning field
                    llm_flags["reasoning"] = False
                    body.pop("reasoning", None)
                    continue
                if status in (400, 401, 403, 404, 422) and tools and "tool_choice" in body \
                        and body["tool_choice"] == "required" and attempt == 0:
                    body["tool_choice"] = "auto"   # some providers reject "required"
                    continue
                if status in (401, 403, 404):
                    disabled_models.add(model)
                    break
                if status is None and attempt < 1:
                    # timed out: one quick retry with as little reasoning as the model allows
                    if llm_flags["reasoning"]:
                        body["reasoning"] = dict(quick)
                    continue
                if status in (500, 502, 503, 504) and attempt < 2:
                    await asyncio.sleep(2 + 3 * attempt)
                    continue
                break
        raise RuntimeError(f"inference failed: {last_error}")

    async def tool_executor(
        name: str,
        arguments: dict
    ) -> Any:

        if state["finished"]:
            raise RuntimeError("the episode has finished")

        if state["step"] >= max_steps:
            raise RuntimeError("environment step limit reached")

        payload = {
            **binding,
            "step": state["step"] + 1,
            "action": {
                "name": name,
                "args": arguments
            }
        }
        result = None
        for _ in range(3):
            data, status, error_body = _post(_env_proxy, "/environment/call", payload)
            if isinstance(data, dict):
                result = data
                break
            if status == 409:
                low = error_body.lower()
                if "finish" in low or "finaliz" in low:
                    state["finished"] = True
                    raise RuntimeError("the episode has finished")
                expected = re.search(r"expected (\d+)", low)
                if expected:
                    state["step"] = int(expected.group(1)) - 1
                    payload["step"] = state["step"] + 1
                    continue
            if status in (400, 401, 403, 404, 422):
                break

        if result is None:
            raise RuntimeError("environment call failed")

        state["step"] = result["step"] if isinstance(result.get("step"), int) else payload["step"]
        state["finished"] = bool(result.get("finished"))
        dialogue.append({
            "step": state["step"],
            "action": {"name": name, "args": arguments},
            "environment_result": result
        })

        return result

    try:
        if state["finished"]:
            return [{"kind": "finished"}]
        agent = ShoppingAgent(
            llm_step=llm_step,
            tool_executor=tool_executor,
            tool_definitions=tool_definitions,
            max_steps=max_steps,
        )
        result = asyncio.run(agent.run(request, policy.get("opening")))
        dialogue.append({"kind": "result", "result": result})
        return dialogue

    except Exception as exc:
        logger.exception("Shopping agent entry point failed")
        dialogue.append({
            "kind": "crash",
            "error": f"{type(exc).__name__}: {exc}"[:400]
        })
        return dialogue


__all__ = ["agent_main"]
