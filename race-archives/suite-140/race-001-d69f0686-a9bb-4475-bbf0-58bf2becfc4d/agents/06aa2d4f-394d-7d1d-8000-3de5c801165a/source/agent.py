"""Hotkey 03 agent: plan first, prove second, commit last.

Workflow is the ORO Bench generated-environment contract, unchanged:

    policy_view -> inference over the dynamic tools -> /environment/call
                -> observation -> repeat until done=true

The strategy is a four-phase machine. A dedicated planning call turns the
shopper goal into a requirement checklist before any action is taken, and the
session-ending tool is withheld from the model's tool list until the cart holds
an item whose identity, current price and current stock have all been observed.
The gate is deterministic and reads only returned observations, so the model
cannot talk its way past it -- but a deadline override always unlocks it, since
an unsold episode scores nothing at all.
"""

from __future__ import annotations

import json
import re
import time
from enum import Enum
from os import getenv
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

from src.agent.proxy_client import ProxyClient


# Model ladder per provider: preferred first, then a widely-served fallback
# used when the preferred id is unavailable or refused.
MODEL_LADDER = {
    "chutes": ("Qwen/Qwen3.5-397B-A17B-TEE", "deepseek-ai/DeepSeek-V3.2-TEE"),
    "openrouter": ("qwen/qwen3.5-397b-a17b", "deepseek/deepseek-v3.2"),
}

RULE_MARKER = "Task rules:"

# Turns held back for the deadline override.
RESERVE_TURNS = 4

# Every episode must reach a terminal state well inside this wall clock. Past
# the soft share the phases stop mattering and the cart gets settled; past the
# hard share the only remaining goal is to finish at all.
TIME_BUDGET = 300.0
SOFT_SHARE = 0.60
HARD_SHARE = 0.86

# Discovery calls that reveal no new variant are stagnant. Enough in a row and
# searching has stopped working, so the turns are better spent committing.
DISCOVERY = frozenset({"search", "filter"})
STAGNANT_LIMIT = 6

# Ceiling on how much of the planning reply is carried into later turns.
PLAN_BUDGET_CHARS = 1200

# Turn blocks retained when a request is retried with less context.
WINDOW_BLOCKS = 7

MONEY_PATTERN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*[A-Z]{3}\b")
LIMIT_PATTERN = re.compile(
    r"(under|below|up to|at most|no more than|within|max|maximum|budget)\W{0,24}$",
    re.IGNORECASE,
)


class Phase(Enum):
    """Where the episode is in its plan-prove-commit progression."""

    PLAN = "plan"
    GATHER = "gather"
    VERIFY = "verify"
    COMMIT = "commit"


def note(text: str) -> None:
    """Diagnostics only; never affects the dialogue that is returned."""
    try:
        print(f"[hk03] {text}", flush=True)
    except Exception:
        pass


def pluck(source: Any, *path: Any, default: Any = None) -> Any:
    """Walk nested mappings and sequences, yielding default on any miss."""
    cursor = source
    for step in path:
        try:
            if isinstance(step, int) and isinstance(cursor, (list, tuple)):
                cursor = cursor[step]
            elif isinstance(cursor, dict):
                cursor = cursor[step]
            else:
                return default
        except (KeyError, IndexError, TypeError):
            return default
    return default if cursor is None else cursor


def to_number(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        return float(value)
    except (TypeError, ValueError):
        return None


def positive_int(value: Any, fallback: int) -> int:
    try:
        number = int(value)
        return number if number > 0 else fallback
    except (TypeError, ValueError):
        return fallback


class Observation(NamedTuple):
    """One call result, flattened out of the runtime's double wrapper."""

    done: bool
    error: Any
    body: Dict[str, Any]

    @classmethod
    def parse(cls, call_result: Any) -> "Observation":
        outer = pluck(call_result, "observation", default={})
        body = pluck(outer, "observation", default={})
        return cls(
            done=bool(pluck(outer, "done", default=False)),
            error=pluck(outer, "error"),
            body=body if isinstance(body, dict) else {},
        )


class Move(NamedTuple):
    """An action to send, plus the tool_call id it must be answered against."""

    name: str
    args: Dict[str, Any]
    reply_to: Optional[str]

    @classmethod
    def from_tool_call(cls, tool_call: Any) -> Optional["Move"]:
        name = pluck(tool_call, "function", "name", default="")
        if not isinstance(name, str) or not name:
            return None
        raw = pluck(tool_call, "function", "arguments", default="")
        args: Dict[str, Any] = {}
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str) and raw.strip():
            try:
                decoded = json.loads(raw)
                args = decoded if isinstance(decoded, dict) else {}
            except ValueError:
                args = {}
        identifier = pluck(tool_call, "id")
        return cls(name, args, identifier if isinstance(identifier, str) else None)

    @classmethod
    def forced(cls, name: str, **args: Any) -> "Move":
        return cls(name, {key: value for key, value in args.items() if value is not None}, None)


class Requirement(NamedTuple):
    """One line of the checklist drafted during the planning phase."""

    text: str
    firm: bool


class Checklist:
    """Requirements drafted from the goal, used to steer every later turn."""

    def __init__(self) -> None:
        self.items: List[Requirement] = []
        self.freeform = ""

    def adopt(self, reply: str) -> None:
        try:
            block = self._json_block(reply)
            if block is not None:
                for text in block.get("firm") or []:
                    self._add(text, firm=True)
                for text in block.get("soft") or []:
                    self._add(text, firm=False)
            if not self.items:
                self.freeform = (reply or "").strip()[:PLAN_BUDGET_CHARS]
        except Exception:
            self.freeform = (reply or "").strip()[:PLAN_BUDGET_CHARS]

    def _add(self, text: Any, firm: bool) -> None:
        if isinstance(text, str) and text.strip():
            self.items.append(Requirement(text.strip()[:200], firm))

    def _json_block(self, reply: str) -> Optional[Dict[str, Any]]:
        if not reply:
            return None
        start = reply.find("{")
        end = reply.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            decoded = json.loads(reply[start : end + 1])
            return decoded if isinstance(decoded, dict) else None
        except ValueError:
            return None

    def render(self) -> str:
        if self.items:
            firm = [item.text for item in self.items if item.firm]
            soft = [item.text for item in self.items if not item.firm]
            lines = []
            if firm:
                lines.append("Firm requirements (all must hold for the exact variant):")
                lines.extend(f"  - {text}" for text in firm)
            if soft:
                lines.append("Soft preferences (honour only after every firm one holds):")
                lines.extend(f"  - {text}" for text in soft)
            return "\n".join(lines)
        if self.freeform:
            return f"Working plan:\n{self.freeform}"
        return ""

    @property
    def drafted(self) -> bool:
        return bool(self.items or self.freeform)


class Wire:
    """Proxy access with a model ladder and a shared retry budget."""

    def __init__(self) -> None:
        self._client = ProxyClient(
            timeout=60,
            max_retries=4,
            retry_delay=2,
            rate_limit_retry_delay=8,
            api_key=getenv("INFERENCE_ACCESS_TOKEN") or getenv("CHUTES_ACCESS_TOKEN"),
        )
        provider = getenv("INFERENCE_PROVIDER", "chutes")
        ladder = list(MODEL_LADDER.get(provider, MODEL_LADDER["chutes"]))
        # SANDBOX_MODEL is documented as an override for the reference agent, so
        # it sits at the end of the ladder rather than displacing our choice.
        late = getenv("SANDBOX_MODEL")
        if late and late not in ladder:
            ladder.append(late)
        self._ladder = ladder
        self._rung = 0

    @property
    def model(self) -> str:
        return self._ladder[min(self._rung, len(self._ladder) - 1)]

    def _demote(self) -> bool:
        if self._rung + 1 < len(self._ladder):
            self._rung += 1
            note(f"falling back to model {self.model}")
            return True
        return False

    def ask(
        self, messages: Sequence[Any], tools: Optional[Sequence[Any]]
    ) -> Optional[Dict[str, Any]]:
        """One chat completion. Tools omitted means a prose answer is wanted."""
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0,
        }
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = "required"
        for _try in range(2):
            try:
                reply = self._client.post("/inference/chat/completions", json_data=body)
            except Exception as failure:
                note(f"inference raised {type(failure).__name__}: {failure}")
                reply = None
            message = pluck(reply, "choices", 0, "message")
            if isinstance(message, dict):
                return message
            if not self._demote():
                return None
            body["model"] = self.model
        return None

    def send(self, envelope: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            reply = self._client.post("/environment/call", json_data=envelope)
            return reply if isinstance(reply, dict) else None
        except Exception as failure:
            note(f"environment call raised {type(failure).__name__}: {failure}")
            return None


class Conversation:
    """Message log with turn anchors, so context can be shed in whole blocks."""

    def __init__(self, system: str, goal: str) -> None:
        self._head: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": goal},
        ]
        self._tail: List[Dict[str, Any]] = []
        self._anchors: List[int] = []

    def open_block(self) -> None:
        self._anchors.append(len(self._tail))

    def add(self, message: Dict[str, Any]) -> None:
        self._tail.append(message)

    def full(self) -> List[Dict[str, Any]]:
        return self._head + self._tail

    def window(self) -> List[Dict[str, Any]]:
        if len(self._anchors) <= WINDOW_BLOCKS:
            return self.full()
        cut = self._anchors[-WINDOW_BLOCKS]
        return self._head + self._tail[cut:]


class Evidence:
    """What the environment has actually shown us, per variant."""

    def __init__(self) -> None:
        self.catalog: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Inspections are stamped with the cart generation they were taken in.
        # A cart mutation moves the generation on, which retires every earlier
        # price and stock reading -- the environment is free to reprice between
        # the add and the order.
        self.generation = 0
        self.inspected: Dict[Tuple[str, str], int] = {}
        self.examined: Set[str] = set()
        self.cart: List[Tuple[str, str]] = []
        self.ordered = False

    def ingest(self, action: str, body: Dict[str, Any]) -> None:
        try:
            if action in ("add_to_cart", "remove_from_cart"):
                self.generation += 1
            if action in ("view", "compare"):
                self._mark_examined(body)
            for row in pluck(body, "results", default=[]) or []:
                self._merge(row)
            for row in pluck(body, "products", default=[]) or []:
                self._merge(row)
            for row in pluck(body, "variants", default=[]) or []:
                self._merge(row, parent=body.get("product_id"))
            self._merge(body)
            added = pluck(body, "added")
            if isinstance(added, dict):
                self._merge(added)
            if action == "inspect_stock":
                self._note_inspection(body)
            cart = body.get("cart")
            if isinstance(cart, list):
                self.cart = []
                for row in cart:
                    self._merge(row)
                    key = self._identify(row)
                    if key:
                        self.cart.append(key)
            if body.get("order"):
                self.ordered = True
        except Exception:
            pass

    def _mark_examined(self, body: Dict[str, Any]) -> None:
        product = body.get("product_id")
        if product:
            self.examined.add(str(product))
        for row in pluck(body, "products", default=[]) or []:
            product = pluck(row, "product_id")
            if product:
                self.examined.add(str(product))

    def _identify(self, row: Any) -> Optional[Tuple[str, str]]:
        product = pluck(row, "product_id")
        sku = pluck(row, "sku")
        if not product or not sku:
            return None
        return str(product), str(sku)

    def _merge(self, row: Any, parent: Any = None) -> None:
        if not isinstance(row, dict):
            return
        if parent and not row.get("product_id"):
            row = dict(row, product_id=parent)
        key = self._identify(row)
        if key is None:
            return
        record = self.catalog.setdefault(key, {"product_id": key[0], "sku": key[1]})
        for field in ("title", "brand", "currency", "category_path"):
            if row.get(field) is not None:
                record[field] = row[field]
        for field in ("price", "price_at_add", "price_at_order"):
            amount = to_number(row.get(field))
            if amount is not None:
                record["price"] = amount
        if isinstance(row.get("in_stock"), bool):
            record["in_stock"] = row["in_stock"]

    def _note_inspection(self, row: Any) -> None:
        key = self._identify(row)
        if key is not None:
            self.inspected[key] = self.generation

    def is_current(self, key: Optional[Tuple[str, str]]) -> bool:
        """True when this SKU was read back in the present cart generation."""
        return key is not None and self.inspected.get(key) == self.generation

    def ruled_out(self, record: Any, ceiling: Optional[float]) -> bool:
        """Observed facts already disqualify this record."""
        try:
            if not isinstance(record, dict):
                return False
            if record.get("in_stock") is False:
                return True
            price = to_number(record.get("price"))
            return ceiling is not None and price is not None and price > ceiling
        except Exception:
            return False

    def cart_head(self) -> Optional[Dict[str, Any]]:
        for key in self.cart:
            record = self.catalog.get(key)
            if record:
                return record
        return None

    def proven(self, ceiling: Optional[float]) -> bool:
        """True when the cart's item is identified, in stock and affordable."""
        record = self.cart_head()
        if not record:
            return False
        key = (str(record.get("product_id")), str(record.get("sku")))
        if not self.is_current(key) or key[0] not in self.examined:
            return False
        if record.get("in_stock") is False:
            return False
        price = to_number(record.get("price"))
        if ceiling is not None and price is not None and price > ceiling:
            return False
        return True

    def shortlist(self, ceiling: Optional[float]) -> List[Dict[str, Any]]:
        pool: List[Tuple[float, Dict[str, Any]]] = []
        for key, record in self.catalog.items():
            if key in self.cart or record.get("in_stock") is False:
                continue
            price = to_number(record.get("price"))
            if ceiling is not None and price is not None and price > ceiling:
                continue
            pool.append((price if price is not None else float("inf"), record))
        pool.sort(key=lambda pair: pair[0])
        return [record for _price, record in pool]

    def size(self) -> int:
        return len(self.catalog)

    def any_record(self) -> Optional[Dict[str, Any]]:
        """Cheapest variant known, ceiling ignored, in stock preferred.

        A last resort only. An episode with no terminal state scores nothing,
        so an imperfect purchase is worth more than an unfinished one.
        """
        pool: List[Tuple[int, float, Dict[str, Any]]] = []
        for key, record in self.catalog.items():
            if key in self.cart:
                continue
            price = to_number(record.get("price"))
            pool.append(
                (
                    0 if record.get("in_stock") is not False else 1,
                    price if price is not None else float("inf"),
                    record,
                )
            )
        if not pool:
            return None
        pool.sort(key=lambda row: (row[0], row[1]))
        return pool[0][2]


class Solver:
    """Drives one episode through the plan-prove-commit phases."""

    def __init__(self, problem_data: Dict[str, Any]) -> None:
        self.binding = pluck(problem_data, "environment", "binding", default={}) or {}
        policy = pluck(problem_data, "environment", "policy_view", default={}) or {}
        self.tools = list(pluck(policy, "tools", default=[]) or [])
        self.goal = str(
            pluck(policy, "query", default="") or pluck(problem_data, "query", default="")
        )
        self.max_steps = positive_int(policy.get("max_steps"), 20)
        self.max_calls = positive_int(policy.get("max_calls_per_turn"), 1)
        self.task_id = str(
            pluck(problem_data, "problem_id", default="")
            or pluck(problem_data, "id", default="")
            or "task"
        )

        self.ceiling = self._read_ceiling()
        self.finisher, self.claim_fields = self._read_finisher()
        self.tool_names = [
            str(pluck(tool, "function", "name", default=""))
            for tool in self.tools
            if pluck(tool, "function", "name")
        ]

        self.wire = Wire()
        self.evidence = Evidence()
        self.checklist = Checklist()
        self.phase = Phase.PLAN
        self.chat: Optional[Conversation] = None
        self.transcript: List[Dict[str, Any]] = []
        self.spent = 0
        self.stalls = 0
        self.withheld = ""
        self.started = time.monotonic()
        self.stagnant = 0

    # ------------------------------------------------------------- reading

    def _read_ceiling(self) -> Optional[float]:
        try:
            head = self.goal.split(RULE_MARKER, 1)[0]
            anchored: List[float] = []
            loose: List[float] = []
            for hit in MONEY_PATTERN.finditer(head):
                amount = to_number(hit.group(1))
                if amount is None:
                    continue
                loose.append(amount)
                if LIMIT_PATTERN.search(head[max(0, hit.start() - 40) : hit.start()]):
                    anchored.append(amount)
            if anchored:
                return min(anchored)
            return loose[0] if len(loose) == 1 else (min(loose) if loose else None)
        except Exception:
            return None

    def _read_finisher(self) -> Tuple[str, List[str]]:
        """Session-ending tool name and its claim vocabulary, from the schema."""
        guess = ""
        try:
            for tool in self.tools:
                name = str(pluck(tool, "function", "name", default=""))
                enum = pluck(
                    tool,
                    "function",
                    "parameters",
                    "properties",
                    "justification",
                    "properties",
                    "claims",
                    "items",
                    "properties",
                    "field",
                    "enum",
                )
                if isinstance(enum, list) and enum:
                    return name, [str(value) for value in enum]
                if not guess and "order" in name.lower():
                    guess = name
        except Exception:
            pass
        return guess, []

    def _rules(self) -> str:
        try:
            if RULE_MARKER in self.goal:
                return self.goal.split(RULE_MARKER, 1)[1].strip()
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------ planning

    def _plan_prompt(self) -> List[Dict[str, Any]]:
        instruction = (
            "Read the shopper request below and separate what it demands from "
            "what it merely prefers. Reply with one JSON object and nothing "
            'else, shaped {"firm": [...], "soft": [...]}, where each entry is a '
            "short 'aspect: value' string. Firm entries are the non-negotiables "
            "such as product category, a named hardware specification, the "
            "budget and stock availability. Soft entries are relaxable tastes "
            "such as a preferred brand, colour or model. Do not add commentary."
        )
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": self.goal},
        ]

    def _draft_plan(self) -> None:
        try:
            reply = self.wire.ask(self._plan_prompt(), None)
            content = str(pluck(reply, "content", default="") or "")
            if content:
                self.checklist.adopt(content)
            if self.checklist.drafted:
                note(f"plan drafted: {len(self.checklist.items)} requirement lines")
            else:
                note("planning call produced nothing usable; proceeding unplanned")
        except Exception as failure:
            note(f"planning skipped after {type(failure).__name__}")
        self.phase = Phase.GATHER

    def _charter(self) -> str:
        lines = [
            "You are a shopping agent operating a generated catalogue environment.",
            "Only the supplied tools exist. Never invent a name or an argument.",
            f"Tools this task: {', '.join(self.tool_names) or 'none listed'}.",
            f"Turn allowance: {self.max_steps}.",
            "",
            "Sequence you must follow:",
            "1. Establish which listings could satisfy the firm requirements.",
            "2. Open the candidate itself and read its structured classification"
            " and raw facts. A promising title proves nothing.",
            "3. Confirm the exact variant's current price and stock.",
            "4. Only then commit.",
            "",
            "Standing cautions:",
            "- One title often sells a compliant and a non-compliant variant."
            " Judge the specific SKU you intend to buy.",
            "- Stock and prices move while you shop. A reading taken before a"
            " cart action is stale afterwards.",
            "- An error or an empty result means change approach, not repeat.",
            "- A shopper message overrides your earlier assumptions.",
            "- Your first cart choice is scored in its own right. It is not a"
            " scratchpad.",
        ]
        if self.ceiling is not None:
            lines.append(
                f"- The goal states a ceiling of {self.ceiling}. Treat it as firm"
                " and compare it against the live variant price."
            )
        if self.finisher:
            lines.append(
                f"- {self.finisher} ends the session. It becomes available once"
                " your cart item has been examined and re-checked; until then it"
                " is deliberately absent from your tool list."
            )
        if self.claim_fields:
            lines.append(
                "- Should the shopper want reasons, attach claims drawn only from"
                f" these fields: {', '.join(self.claim_fields)}. Each must be true"
                " against something you observed, expressed as field and value,"
                " with no prose."
            )
        rules = self._rules()
        if rules:
            lines.extend(["", "This task's own rules:", rules])
        return "\n".join(lines)

    # -------------------------------------------------------------- gating

    def _clock(self) -> float:
        try:
            return time.monotonic() - self.started
        except Exception:
            return 0.0

    def _pressed(self) -> bool:
        """Turns, wall clock or fruitless searching say stop gathering."""
        return (
            (self.max_steps - self.spent) <= RESERVE_TURNS
            or self._clock() >= TIME_BUDGET * SOFT_SHARE
            or self.stagnant >= STAGNANT_LIMIT
        )

    def _spent_out(self) -> bool:
        return (self.max_steps - self.spent) <= 2 or self._clock() >= (
            TIME_BUDGET * HARD_SHARE
        )

    def _last_ditch(self) -> List[Move]:
        """One envelope that reaches a terminal state, proven or not.

        The add and the order travel together because a single round trip is
        all the remaining budget can pay for.
        """
        try:
            head = self.evidence.cart_head()
            if head and self.finisher:
                return [
                    Move.forced(
                        self.finisher,
                        product_id=head.get("product_id"),
                        sku=head.get("sku"),
                    )
                ]
            shortlist = self.evidence.shortlist(self.ceiling)
            choice = shortlist[0] if shortlist else self.evidence.any_record()
            if choice is None or "add_to_cart" not in self.tool_names:
                return []
            filling = Move.forced(
                "add_to_cart",
                product_id=choice.get("product_id"),
                sku=choice.get("sku"),
            )
            if not self.finisher:
                return [filling]
            return [
                filling,
                Move.forced(
                    self.finisher,
                    product_id=choice.get("product_id"),
                    sku=choice.get("sku"),
                ),
            ]
        except Exception:
            return []

    def _offered_tools(self) -> List[Any]:
        """Withhold the finisher until the cart is proven, or time runs short."""
        if not self.finisher or self.phase is Phase.COMMIT or self._pressed():
            self.withheld = ""
            return self.tools
        self.withheld = self.finisher
        return [
            tool
            for tool in self.tools
            if str(pluck(tool, "function", "name", default="")) != self.finisher
        ]

    def _gate_step(self) -> Optional[Move]:
        """What to do instead of ordering: re-read the cart item, or drop it."""
        head = self.evidence.cart_head()
        if not head:
            return None
        key = (str(head.get("product_id")), str(head.get("sku")))
        if self.evidence.ruled_out(head, self.ceiling):
            if "remove_from_cart" in self.tool_names:
                return Move.forced(
                    "remove_from_cart", product_id=key[0], sku=key[1]
                )
            return None
        if not self.evidence.is_current(key) and "inspect_stock" in self.tool_names:
            return Move.forced("inspect_stock", product_id=key[0], sku=key[1])
        return None

    def _refresh_phase(self) -> None:
        if self.evidence.proven(self.ceiling):
            self.phase = Phase.COMMIT
        elif self.evidence.cart:
            self.phase = Phase.VERIFY
        else:
            self.phase = Phase.GATHER

    def _steer(self) -> str:
        remaining = self.max_steps - self.spent
        plan = self.checklist.render()
        if self.phase is Phase.COMMIT:
            head = (
                "Your cart item is identified, in stock and inside the budget. "
                f"Call {self.finisher or 'the finishing tool'} now. Do not seek "
                "further confirmation."
            )
        elif self.phase is Phase.VERIFY:
            head = (
                "Something is in the cart but not yet proven. Open it with view "
                "or compare, then re-check that exact SKU's current price and "
                "stock. Replace it if it fails a firm requirement."
            )
        else:
            head = (
                "No candidate is committed. Widen or sharpen the search, then "
                "examine the most promising listing's real facts."
            )
        tail = f"Turn {self.spent + 1} of {self.max_steps}; {remaining} remain."
        return "\n\n".join(part for part in (head, plan, tail) if part)

    # -------------------------------------------------------------- acting

    def _package(self, turn: int, moves: Sequence[Move]) -> Dict[str, Any]:
        stamp = f"{self.task_id}-turn-{turn}"
        return {
            **self.binding,
            "call_id": stamp,
            "idempotency_key": stamp,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{stamp}-{index}",
                    "action": {"name": move.name, "args": move.args},
                }
                for index, move in enumerate(moves, start=1)
            ],
        }

    def _override(self) -> Optional[Move]:
        """Deterministic path to a terminal state, used under time pressure."""
        try:
            head = self.evidence.cart_head()
            shortlist = self.evidence.shortlist(self.ceiling)
            if head and self.evidence.ruled_out(head, self.ceiling):
                if shortlist:
                    # Ordering a variant known to be out of stock or over
                    # budget scores nothing, and there is something better to
                    # swap to, so spend the reserve replacing it.
                    dropped = self._gate_step()
                    if dropped is not None:
                        return dropped
                elif self.finisher and head.get("in_stock") is not False:
                    # Nothing affordable is left to swap to, so churning the
                    # cart only burns turns. Take what is already in it.
                    return Move.forced(
                        self.finisher,
                        product_id=head.get("product_id"),
                        sku=head.get("sku"),
                    )
            elif head and self.finisher:
                return Move.forced(
                    self.finisher,
                    product_id=head.get("product_id"),
                    sku=head.get("sku"),
                )
            choice = shortlist[0] if shortlist else self.evidence.any_record()
            if choice is not None and "add_to_cart" in self.tool_names:
                return Move.forced(
                    "add_to_cart",
                    product_id=choice.get("product_id"),
                    sku=choice.get("sku"),
                )
            if "search" in self.tool_names and self.stalls < 3:
                return Move.forced("search", query=self._keywords(), in_stock=True)
            if "filter" in self.tool_names:
                return Move.forced("filter", in_stock=True, max_price=self.ceiling)
            return None
        except Exception:
            return None

    def _keywords(self) -> str:
        try:
            head = self.goal.split(RULE_MARKER, 1)[0]
            words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", head)[:10]
            return " ".join(words) if words else "in stock"
        except Exception:
            return "in stock"

    def _collect(self, tool_calls: Sequence[Any]) -> List[Move]:
        """One state-changing move, or a short run of information gathering."""
        reading = {"search", "filter", "view", "compare", "inspect_stock", "inspect_cart"}
        batch: List[Move] = []
        room = max(1, min(self.max_calls, 3))
        for tool_call in tool_calls:
            move = Move.from_tool_call(tool_call)
            if move is None:
                continue
            if self.withheld and move.name == self.withheld:
                # The model reached for a tool this turn's policy did not offer.
                note(f"{move.name} was not on offer; settling the cart first")
                stand_in = self._gate_step()
                return [stand_in] if stand_in is not None else batch
            if move.name not in reading:
                return batch if batch else [move]
            batch.append(move)
            if len(batch) >= room:
                break
        return batch

    def _decide(self, turn: int) -> List[Move]:
        assert self.chat is not None
        self._refresh_phase()

        if self._spent_out():
            final = self._last_ditch()
            if final:
                note(
                    f"turn {turn}: {self._clock():.0f}s gone, finishing with "
                    f"{', '.join(move.name for move in final)}"
                )
                self.chat.open_block()
                return final

        if self._pressed():
            forced = self._override()
            if forced is not None:
                note(
                    f"turn {turn}: closing out -> {forced.name} "
                    f"({self._clock():.0f}s, stagnant={self.stagnant})"
                )
                self.chat.open_block()
                return [forced]

        self.chat.open_block()
        self.chat.add({"role": "user", "content": self._steer()})

        offered = self._offered_tools()
        reply = self.wire.ask(self.chat.full(), offered)
        if reply is None:
            note(f"turn {turn}: retrying against a shorter window")
            reply = self.wire.ask(self.chat.window(), offered)
        if reply is None:
            self.stalls += 1
            forced = self._override()
            if forced is None:
                return []
            note(f"turn {turn}: no inference, forcing {forced.name}")
            return [forced]

        spoken = str(pluck(reply, "content", default="") or "")
        raw_calls = pluck(reply, "tool_calls", default=[]) or []
        moves = self._collect(raw_calls if isinstance(raw_calls, list) else [])
        if not moves:
            self.stalls += 1
            self.chat.add({"role": "assistant", "content": spoken})
            self.chat.add(
                {
                    "role": "user",
                    "content": (
                        "That reply contained no tool call. Choose the single "
                        "action that best advances the plan and call it now."
                    ),
                }
            )
            self.transcript.append({"role": "assistant", "content": spoken})
            forced = self._override() if self.stalls >= 2 else None
            return [forced] if forced is not None else []

        answered = {move.reply_to for move in moves if move.reply_to}
        self.chat.add(
            {
                "role": "assistant",
                "content": spoken,
                "tool_calls": [
                    call for call in raw_calls if pluck(call, "id") in answered
                ],
            }
        )
        return moves

    # ----------------------------------------------------------- observing

    def _digest(self, moves: Sequence[Move], reply: Dict[str, Any]) -> bool:
        assert self.chat is not None
        results = pluck(reply, "calls", default=[]) or []
        known = self.evidence.size()
        complete = False
        for index, move in enumerate(moves):
            slot = results[index] if index < len(results) else {}
            seen = Observation.parse(slot)
            self.evidence.ingest(move.name, seen.body)
            complete = complete or seen.done
            rendered = json.dumps(
                {"result": seen.body, "error": seen.error, "done": seen.done},
                default=str,
            )
            if move.reply_to:
                self.chat.add(
                    {"role": "tool", "tool_call_id": move.reply_to, "content": rendered}
                )
            else:
                self.chat.add(
                    {"role": "user", "content": f"{move.name} returned {rendered}"}
                )
            if seen.error:
                self.chat.add(
                    {
                        "role": "user",
                        "content": (
                            f"{move.name} failed with an error. Fix the arguments "
                            "or pick another tool; repeating it unchanged wastes a turn."
                        ),
                    }
                )
        if any(move.name in DISCOVERY for move in moves):
            if self.evidence.size() > known:
                self.stagnant = 0
            else:
                self.stagnant += 1
                note(f"discovery revealed nothing new ({self.stagnant} in a row)")

        spoken = pluck(reply, "user_message", "content")
        if spoken:
            self.chat.add({"role": "user", "content": str(spoken)})
            self.stagnant = 0
        return complete

    def _charge(self, moves: Sequence[Move], reply: Dict[str, Any]) -> None:
        """Advance the turn budget using the runtime's own counters."""
        try:
            previous = self.spent
            claimed = [
                positive_int(reply.get("action_count"), 0),
                positive_int(reply.get("solver_turn_count"), 0),
            ]
            self.spent = max([previous + 1] + claimed)
            if len(moves) > 1 and self.spent - previous >= len(moves):
                note("budget is charged per action; gathering one at a time")
                self.max_calls = 1
        except Exception:
            self.spent += 1

    def _record(self, turn: int, moves: Sequence[Move], reply: Dict[str, Any]) -> None:
        self.transcript.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": move.reply_to or f"{self.task_id}-{turn}-{index}",
                        "type": "function",
                        "function": {
                            "name": move.name,
                            "arguments": json.dumps(move.args, default=str),
                        },
                    }
                    for index, move in enumerate(moves, start=1)
                ],
                "environment_result": reply,
            }
        )

    # ------------------------------------------------------------ main run

    def solve(self) -> List[Dict[str, Any]]:
        self._draft_plan()
        self.chat = Conversation(self._charter(), self.goal)
        note(
            f"task {self.task_id}: steps={self.max_steps} calls={self.max_calls} "
            f"ceiling={self.ceiling} finisher={self.finisher or 'unknown'}"
        )

        for turn in range(1, self.max_steps + 1):
            if self.spent >= self.max_steps or self.evidence.ordered:
                break

            moves = self._decide(turn)
            if not moves:
                if self.stalls >= 3:
                    note(f"turn {turn}: stalled with no viable move; stopping")
                    break
                continue

            reply = self.wire.send(self._package(turn, moves))
            if reply is None and len(moves) > 1:
                note(f"turn {turn}: batch rejected, resending the first move alone")
                moves = moves[:1]
                self.max_calls = 1
                reply = self.wire.send(self._package(turn, moves))
            if reply is None:
                note(f"turn {turn}: environment unreachable; stopping")
                break

            self._record(turn, moves, reply)
            complete = self._digest(moves, reply)
            self._charge(moves, reply)
            note(
                f"turn {turn}: {', '.join(move.name for move in moves)} "
                f"phase={self.phase.value} done={complete} "
                f"spent={self.spent}/{self.max_steps}"
            )
            if complete:
                break

        return self.transcript


def agent_main(problem_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Entry point: run one generated-environment episode."""
    solver: Optional[Solver] = None
    try:
        solver = Solver(problem_data)
        return solver.solve()
    except Exception as failure:
        note(f"episode ended by {type(failure).__name__}: {failure}")
        try:
            return solver.transcript if solver is not None else []
        except Exception:
            return []


__all__ = ["agent_main"]
