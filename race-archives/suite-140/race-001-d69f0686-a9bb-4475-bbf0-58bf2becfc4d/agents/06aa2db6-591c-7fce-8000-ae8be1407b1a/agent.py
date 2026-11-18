"""Hotkey 06 agent: the environment moves, so the agent reacts to the movement.

Workflow is the ORO Bench generated-environment contract, unchanged:

    policy_view -> inference over the dynamic tools -> /environment/call
                -> observation -> repeat until done=true

The strategy is event driven. World state is snapshotted before each turn and
diffed against the state after it, and every difference becomes an event: a
price moved, stock disappeared, the cart changed, a call failed, a search came
back empty, the shopper spoke. Events are dispatched through a registry of
handlers, in priority order. A handler may narrate the event for the model, or
claim the next turn outright when the correct response is not a matter of
judgement -- a carted item that has just gone out of stock is dropped, not
discussed.

Handlers are registered with a decorator, which keeps each reaction next to the
condition that triggers it.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from src.agent.proxy_client import ProxyClient


MODEL_TIERS = {
    "chutes": ("moonshotai/Kimi-K2.6-TEE", "deepseek-ai/DeepSeek-V3.2-TEE"),
    "openrouter": ("moonshotai/kimi-k2.6", "deepseek/deepseek-v3.2"),
}

RULE_SPLIT = "Task rules:"
LATE_TURNS = 4

# Every episode must reach a terminal state well inside this wall clock. Past
# the soft share the agent stops reacting and settles the cart; past the hard
# share the only remaining goal is to finish at all.
TIME_BUDGET = 300.0
SOFT_SHARE = 0.60
HARD_SHARE = 0.86

# Discovery calls that surface no new variant are stagnant. Enough in a row and
# searching has stopped paying, so the turns are better spent committing.
DISCOVERY = frozenset({"search", "filter"})
STAGNANT_LIMIT = 6
WINDOW = 12
GATHERING = frozenset(
    {"search", "filter", "view", "compare", "inspect_stock", "inspect_cart"}
)

WORD = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
MONEY = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*[A-Z]{3}\b")
LIMIT_WORDS = re.compile(
    r"(under|below|up to|at most|no more than|within|max|maximum|budget|spend)\W{0,24}$",
    re.IGNORECASE,
)


def trace(message: str) -> None:
    """Instrumentation only; the returned dialogue is unaffected."""
    try:
        print(f"[hk06] {message}", flush=True)
    except Exception:
        pass


def as_map(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_seq(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def as_money(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        return float(value)
    except (TypeError, ValueError):
        return None


def as_span(value: Any, default: int) -> int:
    try:
        span = int(value)
        return span if span > 0 else default
    except (TypeError, ValueError):
        return default


class Event(NamedTuple):
    """One observed change in the world, or one thing the runtime reported."""

    kind: str
    key: Tuple[str, str]
    before: Any
    after: Any
    detail: str


class Intent:
    """An action this agent has decided to send on its own initiative."""

    def __init__(self, name: str, args: Dict[str, Any], answers: Optional[str] = None):
        self.name = name
        self.args = {key: value for key, value in args.items() if value is not None}
        self.answers = answers

    @classmethod
    def on(cls, name: str, key: Tuple[str, str]) -> "Intent":
        return cls(name, {"product_id": key[0], "sku": key[1]})


Handler = Callable[[Event, "Solver"], Optional[Intent]]

# (priority, event kind, handler). Lower priority number reacts first.
REGISTRY: List[Tuple[int, str, Handler]] = []


def reacts_to(kind: str, priority: int = 50) -> Callable[[Handler], Handler]:
    """Register a reaction to one kind of event."""

    def keep(handler: Handler) -> Handler:
        REGISTRY.append((priority, kind, handler))
        REGISTRY.sort(key=lambda row: row[0])
        return handler

    return keep


# ------------------------------------------------------------------- reactions


@reacts_to("order_placed", priority=1)
def _settled(event: Event, agent: "Solver") -> Optional[Intent]:
    agent.finished = True
    return None


@reacts_to("shopper_spoke", priority=5)
def _shopper(event: Event, agent: "Solver") -> Optional[Intent]:
    """A new instruction outranks any reaction already queued."""
    agent.bulletin.append(
        f"The shopper said: {event.detail} Apply this to every later choice, "
        "including anything already in the cart."
    )
    agent.claimed = None
    return None


@reacts_to("stock_lost", priority=10)
def _sold_out(event: Event, agent: "Solver") -> Optional[Intent]:
    if event.key in agent.ledger.cart and "remove_from_cart" in agent.tool_names:
        agent.bulletin.append(
            f"{event.key[0]} {event.key[1]} sold out while it was in the cart, so "
            "it is being removed. Choose a different variant."
        )
        return Intent.on("remove_from_cart", event.key)
    agent.bulletin.append(f"{event.key[0]} {event.key[1]} is now out of stock.")
    return None


@reacts_to("price_moved", priority=15)
def _repriced(event: Event, agent: "Solver") -> Optional[Intent]:
    after = as_money(event.after)
    over = agent.ceiling is not None and after is not None and after > agent.ceiling
    if event.key in agent.ledger.cart and over:
        agent.bulletin.append(
            f"{event.key[0]} {event.key[1]} moved from {event.before} to {after}, "
            f"past the {agent.ceiling} ceiling, so it is being removed."
        )
        if "remove_from_cart" in agent.tool_names:
            return Intent.on("remove_from_cart", event.key)
        return None
    agent.bulletin.append(
        f"{event.key[0]} {event.key[1]} moved from {event.before} to {after}."
    )
    return None


@reacts_to("cart_gained", priority=20)
def _carted(event: Event, agent: "Solver") -> Optional[Intent]:
    """A cart action retires the reading it was based on, so read again.

    The catalogue is free to reprice or sell out between the add and the order,
    which makes the price quoted in the add response the wrong one to commit on.
    """
    agent.unverified = event.key
    if "inspect_stock" not in agent.tool_names:
        return None
    agent.bulletin.append(
        f"{event.key[0]} {event.key[1]} is in the cart. Its current price and "
        "stock are being read back before anything is ordered."
    )
    return Intent.on("inspect_stock", event.key)


@reacts_to("cart_lost", priority=25)
def _uncarted(event: Event, agent: "Solver") -> Optional[Intent]:
    if agent.unverified == event.key:
        agent.unverified = None
    agent.bulletin.append(f"{event.key[0]} {event.key[1]} left the cart.")
    return None


@reacts_to("call_failed", priority=30)
def _failed(event: Event, agent: "Solver") -> Optional[Intent]:
    agent.misfires[event.detail] = agent.misfires.get(event.detail, 0) + 1
    agent.bulletin.append(
        f"The {event.detail} call was rejected: {event.after}. Change the "
        "arguments or the tool; repeating it unchanged will fail again."
    )
    if agent.misfires.get(event.detail, 0) >= 2:
        return agent.sweep_intent()
    return None


@reacts_to("no_matches", priority=35)
def _empty(event: Event, agent: "Solver") -> Optional[Intent]:
    agent.bulletin.append(
        f"{event.detail} returned nothing. Loosen one constraint at a time, "
        "keeping every firm requirement."
    )
    return agent.sweep_intent()


@reacts_to("found", priority=60)
def _appeared(event: Event, agent: "Solver") -> Optional[Intent]:
    agent.discovered += 1
    return None


# ----------------------------------------------------------------- world state


class Ledger:
    """Variant facts, the cart, and the ability to snapshot both."""

    def __init__(self) -> None:
        self.rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.cart: List[Tuple[str, str]] = []
        self.ordered = False

    def key_of(self, row: Any, parent: Any = None) -> Optional[Tuple[str, str]]:
        entry = as_map(row)
        product = entry.get("product_id") or parent
        sku = entry.get("sku")
        if not product or not sku:
            return None
        return str(product), str(sku)

    def note(self, row: Any, parent: Any = None, examined: bool = False) -> None:
        entry = as_map(row)
        key = self.key_of(entry, parent)
        if key is None:
            return
        record = self.rows.setdefault(key, {"product_id": key[0], "sku": key[1]})
        for field in ("title", "brand", "currency", "category_path"):
            if entry.get(field) is not None:
                record[field] = entry[field]
        for field in ("price", "price_at_add", "price_at_order"):
            amount = as_money(entry.get(field))
            if amount is not None:
                record["price"] = amount
        if isinstance(entry.get("in_stock"), bool):
            record["in_stock"] = entry["in_stock"]
        if examined:
            record["examined"] = True

    def absorb(self, action: str, body: Dict[str, Any]) -> None:
        try:
            opened = action in ("view", "compare")
            for row in as_seq(body.get("results")):
                self.note(row)
            for row in as_seq(body.get("products")):
                self.note(row, examined=True)
            for row in as_seq(body.get("variants")):
                self.note(row, parent=body.get("product_id"), examined=opened)
            self.note(body, examined=opened)
            self.note(body.get("added"))
            if isinstance(body.get("cart"), list):
                seen: List[Tuple[str, str]] = []
                for row in as_seq(body.get("cart")):
                    self.note(row)
                    key = self.key_of(row)
                    if key is not None:
                        seen.append(key)
                self.cart = seen
            if body.get("order"):
                self.ordered = True
        except Exception:
            pass

    def snapshot(self) -> Dict[Tuple[str, str], Tuple[Any, Any]]:
        return {
            key: (row.get("price"), row.get("in_stock")) for key, row in self.rows.items()
        }

    def changes(
        self,
        before: Dict[Tuple[str, str], Tuple[Any, Any]],
        cart_before: Sequence[Tuple[str, str]],
    ) -> List[Event]:
        """Every difference between a past snapshot and the present one."""
        found: List[Event] = []
        try:
            after = self.snapshot()
            for key, (price, stock) in after.items():
                if key not in before:
                    found.append(Event("found", key, None, price, ""))
                    continue
                was_price, was_stock = before[key]
                if price is not None and was_price is not None and price != was_price:
                    found.append(Event("price_moved", key, was_price, price, ""))
                if stock is False and was_stock is not False:
                    found.append(Event("stock_lost", key, was_stock, stock, ""))
                elif stock is True and was_stock is False:
                    found.append(Event("stock_gained", key, was_stock, stock, ""))
            held = set(self.cart)
            past = set(cart_before)
            for key in held - past:
                found.append(Event("cart_gained", key, None, None, ""))
            for key in past - held:
                found.append(Event("cart_lost", key, None, None, ""))
        except Exception:
            pass
        return found

    def head(self) -> Optional[Tuple[str, str]]:
        return self.cart[0] if self.cart else None

    def row(self, key: Optional[Tuple[str, str]]) -> Dict[str, Any]:
        return dict(self.rows.get(key) or {}) if key is not None else {}

    def unusable(self, key: Optional[Tuple[str, str]], ceiling: Optional[float]) -> bool:
        record = self.row(key)
        if not record:
            return False
        if record.get("in_stock") is False:
            return True
        price = as_money(record.get("price"))
        return ceiling is not None and price is not None and price > ceiling

    def size(self) -> int:
        return len(self.rows)

    def any_pick(self) -> Optional[Tuple[str, str]]:
        """Cheapest variant known, ceiling ignored, in stock preferred.

        A last resort only. An episode that never reaches a terminal state
        scores nothing, so an imperfect purchase is worth more.
        """
        try:
            pool = []
            for key, row in self.rows.items():
                if key in self.cart:
                    continue
                price = as_money(row.get("price"))
                pool.append(
                    (
                        0 if row.get("in_stock") is not False else 1,
                        price if price is not None else float("inf"),
                        key,
                    )
                )
            if not pool:
                return None
            pool.sort(key=lambda row: (row[0], row[1]))
            return pool[0][2]
        except Exception:
            return None

    def pick(self, ceiling: Optional[float]) -> Optional[Tuple[str, str]]:
        """Cheapest variant outside the cart that nothing observed rules out."""
        try:
            best: Optional[Tuple[float, Tuple[str, str]]] = None
            for key, row in self.rows.items():
                if key in self.cart or row.get("in_stock") is False:
                    continue
                price = as_money(row.get("price"))
                if ceiling is not None and price is not None and price > ceiling:
                    continue
                rank = price if price is not None else float("inf")
                if best is None or rank < best[0]:
                    best = (rank, key)
            return best[1] if best else None
        except Exception:
            return None


# -------------------------------------------------------------------- transport


class Link:
    """Proxy access with a model tier list and 429-aware retries."""

    def __init__(self) -> None:
        self.client = ProxyClient(
            timeout=40,
            max_retries=2,
            retry_delay=2,
            rate_limit_retry_delay=8,
            api_key=getenv("INFERENCE_ACCESS_TOKEN") or getenv("CHUTES_ACCESS_TOKEN"),
        )
        self._opened = time.monotonic()
        self._misses = 0
        self.muted = False
        provider = getenv("INFERENCE_PROVIDER", "chutes")
        self.tiers = list(MODEL_TIERS.get(provider, MODEL_TIERS["chutes"]))
        spare = getenv("SANDBOX_MODEL")
        if spare and spare not in self.tiers:
            self.tiers.append(spare)
        self.tier = 0

    def model(self) -> str:
        return self.tiers[min(self.tier, len(self.tiers) - 1)]


    def _pace(self) -> None:
        """Size the timeout and retry count to the time actually left.

        The client applies its timeout per attempt, so a stalled endpoint can
        otherwise burn retries x timeout seconds inside one call and take the
        whole episode with it.
        """
        try:
            left = max(
                0.0, TIME_BUDGET * HARD_SHARE - (time.monotonic() - self._opened)
            )
            window = max(10.0, min(40.0, left / 5.0))
            self.client.timeout = int(window)
            self.client.max_retries = 2 if left > 3 * window else 1
        except Exception:
            pass

    def _missed(self, answered: bool) -> None:
        """Give up on inference once it has stalled twice in a row.

        An endpoint that has already failed twice will not come back inside
        this budget, and the deterministic path still reaches a terminal state.
        """
        if answered:
            self._misses = 0
            return
        self._misses += 1
        if self._misses >= 2 and not self.muted:
            self.muted = True
            trace("inference stalled twice; running deterministically")

    def ask(
        self, messages: Sequence[Any], tools: Sequence[Any]
    ) -> Optional[Dict[str, Any]]:
        request = {
            "model": self.model(),
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "required",
            "temperature": 0,
        }
        if self.muted:
            return None
        for _try in range(2):
            self._pace()
            try:
                answer = self.client.post(
                    "/inference/chat/completions", json_data=request
                )
            except Exception as failure:
                trace(f"inference raised {type(failure).__name__}: {failure}")
                answer = None
            choices = as_seq(as_map(answer).get("choices"))
            self._missed(bool(choices))
            if choices:
                return as_map(as_map(choices[0]).get("message"))
            if self.tier + 1 >= len(self.tiers):
                return None
            self.tier += 1
            request["model"] = self.model()
            trace(f"dropping to model tier {request['model']}")
        return None

    def send(self, envelope: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._pace()
        try:
            answer = self.client.post("/environment/call", json_data=envelope)
            return answer if isinstance(answer, dict) else None
        except Exception as failure:
            trace(f"environment call raised {type(failure).__name__}: {failure}")
            return None


# ----------------------------------------------------------------------- solver


class Solver:
    """One episode: act, diff the world, react, repeat."""

    def __init__(self, problem_data: Dict[str, Any]) -> None:
        source = as_map(problem_data)
        environment = as_map(source.get("environment"))
        policy = as_map(environment.get("policy_view"))

        self.binding = as_map(environment.get("binding"))
        self.tools = as_seq(policy.get("tools"))
        self.tool_names = [
            str(as_map(as_map(tool).get("function")).get("name") or "")
            for tool in self.tools
        ]
        self.goal = str(policy.get("query") or source.get("query") or "")
        self.opening = self.goal.split(RULE_SPLIT, 1)[0]
        self.rules = (
            self.goal.split(RULE_SPLIT, 1)[1].strip() if RULE_SPLIT in self.goal else ""
        )
        self.limit = as_span(policy.get("max_steps"), 20)
        self.per_turn = as_span(policy.get("max_calls_per_turn"), 1)
        self.task = str(source.get("problem_id") or source.get("id") or "task")
        self.ceiling = self._ceiling()
        self.finisher, self.claim_fields = self._contract()

        self.link = Link()
        self.ledger = Ledger()
        self.history: List[Dict[str, Any]] = []
        self.dialogue: List[Dict[str, Any]] = []
        self.bulletin: List[str] = []
        self.misfires: Dict[str, int] = {}
        self.claimed: Optional[Intent] = None
        self.unverified: Optional[Tuple[str, str]] = None
        self.spent = 0
        self.idle = 0
        self.started = time.monotonic()
        self.stagnant = 0
        self.probes = 0
        self.discovered = 0
        self.finished = False

    # ------------------------------------------------------------- goal input

    def _ceiling(self) -> Optional[float]:
        try:
            anchored: List[float] = []
            every: List[float] = []
            for hit in MONEY.finditer(self.opening):
                amount = as_money(hit.group(1))
                if amount is None:
                    continue
                every.append(amount)
                if LIMIT_WORDS.search(self.opening[max(0, hit.start() - 40) : hit.start()]):
                    anchored.append(amount)
            if anchored:
                return min(anchored)
            if len(every) == 1:
                return every[0]
            return min(every) if every else None
        except Exception:
            return None

    def _contract(self) -> Tuple[str, List[str]]:
        """Session-ending tool and its claim vocabulary, read from the schema."""
        spare = ""
        try:
            for tool in self.tools:
                function = as_map(as_map(tool).get("function"))
                name = str(function.get("name") or "")
                properties = as_map(as_map(function.get("parameters")).get("properties"))
                justification = as_map(properties.get("justification"))
                if not justification:
                    if not spare and "order" in name.lower():
                        spare = name
                    continue
                claims = as_map(as_map(justification.get("properties")).get("claims"))
                field = as_map(
                    as_map(as_map(claims.get("items")).get("properties")).get("field")
                )
                return name, [str(value) for value in as_seq(field.get("enum"))]
        except Exception:
            pass
        return spare, []

    # -------------------------------------------------------------- briefing

    def _charter(self) -> str:
        try:
            lines = [
                "You are a shopping agent inside a generated catalogue environment.",
                "Call only the listed tools; never invent a name or an argument.",
                f"Tools: {', '.join(name for name in self.tool_names if name) or 'none'}.",
                f"You have {self.limit} turns. Gather evidence, then commit.",
                "",
                "This environment changes underneath you. Every turn you are told "
                "what changed since the last one, and reactions that need no "
                "judgement have already been applied for you.",
                "- Firm requirements outrank preferences; never trade one away.",
                "- Judge the exact variant, not the parent listing: one title often "
                "covers a compliant and a non-compliant SKU.",
                "- Prefer structured catalogue facts over words in a title.",
                "- A reading taken before a cart action does not describe the world "
                "after it.",
                "- An error or an empty result means change approach, not repeat.",
                "- The first item you cart is graded on its own merits.",
            ]
            if self.ceiling is not None:
                lines.append(f"- Stated ceiling: {self.ceiling}. Treat it as firm.")
            if self.finisher:
                lines.append(
                    f"- {self.finisher} ends the session. Call it once your pick is "
                    "confirmed in stock and inside budget; do not ask permission."
                )
            if self.claim_fields:
                lines.append(
                    "- If the shopper wants reasons, attach claims using only these "
                    f"fields: {', '.join(self.claim_fields)}. Each must be true "
                    "against an observed value, as field and value, without prose."
                )
            if self.rules:
                lines.extend(["", "Rules for this task:", self.rules])
            return "\n".join(lines)
        except Exception:
            return "Use the supplied tools until an observation reports done=true."

    def _dispatch(self, events: Sequence[Event]) -> None:
        """Narrate every event and let the first willing handler claim a turn."""
        for priority, kind, handler in REGISTRY:
            for event in events:
                if event.kind != kind:
                    continue
                try:
                    intent = handler(event, self)
                except Exception as failure:
                    trace(f"handler for {kind} raised {type(failure).__name__}: {failure}")
                    continue
                if intent is not None and self.claimed is None:
                    trace(f"{kind} on {event.key[0]}::{event.key[1]} claims {intent.name}")
                    self.claimed = intent

    def _digest_note(self) -> str:
        left = self.limit - self.spent
        lines: List[str] = []
        if self.bulletin:
            lines.append("Since your last turn:")
            lines.extend(f"  - {item}" for item in self.bulletin[-6:])
        elif self.spent:
            lines.append("Nothing changed since your last turn.")
        head = self.ledger.head()
        if head is not None:
            record = self.ledger.row(head)
            state = "verified" if self.unverified != head else "not re-read yet"
            lines.append(
                f"In the cart: {head[0]} {head[1]} at {record.get('price')} ({state})."
            )
        elif self.discovered:
            lines.append(f"{self.discovered} variants seen so far, cart empty.")
        lines.append(f"{left} turns remain.")
        self.bulletin = []
        return "\n".join(lines)

    # ---------------------------------------------------------------- acting

    def _envelope(self, turn: int, intents: Sequence[Intent]) -> Dict[str, Any]:
        stamp = f"{self.task}-turn-{turn}"
        return {
            **self.binding,
            "call_id": stamp,
            "idempotency_key": stamp,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{stamp}-{index}",
                    "action": {"name": intent.name, "args": intent.args},
                }
                for index, intent in enumerate(intents, start=1)
            ],
        }

    def _from_tool_call(self, call: Any) -> Optional[Intent]:
        entry = as_map(call)
        function = as_map(entry.get("function"))
        name = str(function.get("name") or "")
        if not name:
            return None
        raw = function.get("arguments")
        args: Dict[str, Any] = {}
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                args = parsed if isinstance(parsed, dict) else {}
            except ValueError:
                args = {}
        answers = entry.get("id")
        return Intent(name, args, answers if isinstance(answers, str) else None)

    def _order_intent(self, key: Tuple[str, str]) -> Intent:
        record = self.ledger.row(key)
        args: Dict[str, Any] = {"product_id": key[0], "sku": key[1]}
        allowed = set(self.claim_fields)
        claims: List[Dict[str, str]] = []
        price = as_money(record.get("price"))
        if "price" in allowed and price is not None:
            claims.append({"field": "price", "value": f"{price}"})
        if "in_stock" in allowed and record.get("in_stock") is True:
            claims.append({"field": "in_stock", "value": "true"})
        if "brand" in allowed and record.get("brand"):
            claims.append({"field": "brand", "value": str(record["brand"])})
        if claims:
            args["justification"] = {"claims": claims}
        return Intent(self.finisher, args)

    def sweep_intent(self) -> Optional[Intent]:
        """A broader look, used when a call fails or comes back empty."""
        try:
            self.probes += 1
            if "filter" in self.tool_names and self.probes % 2 == 0:
                args: Dict[str, Any] = {"in_stock": True}
                if self.ceiling is not None:
                    args["max_price"] = self.ceiling
                return Intent("filter", args)
            if "search" in self.tool_names:
                words = WORD.findall(self.opening)[: max(2, 8 - self.probes)]
                return Intent(
                    "search", {"query": " ".join(words) or "in stock", "in_stock": True}
                )
            return None
        except Exception:
            return None

    def _unaided(self) -> Optional[Intent]:
        """Deterministic next move when inference is unusable."""
        try:
            self.idle += 1
            head = self.ledger.head()
            if head is not None:
                if self.ledger.unusable(head, self.ceiling):
                    if "remove_from_cart" in self.tool_names:
                        return Intent.on("remove_from_cart", head)
                elif self.unverified == head and "inspect_stock" in self.tool_names:
                    return Intent.on("inspect_stock", head)
                elif self.finisher:
                    return self._order_intent(head)
            choice = self.ledger.pick(self.ceiling)
            if choice is not None:
                record = self.ledger.row(choice)
                if not record.get("examined") and "view" in self.tool_names:
                    return Intent("view", {"product_id": choice[0]})
                if "add_to_cart" in self.tool_names:
                    return Intent.on("add_to_cart", choice)
            return self.sweep_intent()
        except Exception:
            return None

    def _clock(self) -> float:
        try:
            return time.monotonic() - self.started
        except Exception:
            return 0.0

    def _late(self) -> bool:
        """Turns, wall clock or fruitless searching say stop gathering."""
        return (
            (self.limit - self.spent) <= LATE_TURNS
            or self._clock() >= TIME_BUDGET * SOFT_SHARE
            or self.stagnant >= STAGNANT_LIMIT
        )

    def _spent_out(self) -> bool:
        return (self.limit - self.spent) <= 2 or self._clock() >= (
            TIME_BUDGET * HARD_SHARE
        )

    def _last_ditch(self) -> List[Intent]:
        """One envelope that reaches a terminal state, verified or not.

        The add and the order travel together because a single round trip is
        all the remaining budget can pay for.
        """
        try:
            head = self.ledger.head()
            if head is not None and self.finisher:
                return [self._order_intent(head)]
            choice = self.ledger.pick(self.ceiling) or self.ledger.any_pick()
            if choice is None or "add_to_cart" not in self.tool_names:
                return []
            filling = Intent.on("add_to_cart", choice)
            if not self.finisher:
                return [filling]
            return [filling, self._order_intent(choice)]
        except Exception:
            return []

    def _vet(self, intents: List[Intent]) -> Optional[Intent]:
        """Stop an order that the world no longer supports.

        Under deadline pressure the order stands: an unfinished episode scores
        nothing at all, which is worse than an imperfect one.
        """
        try:
            if not self.finisher or self._late():
                return None
            if not any(intent.name == self.finisher for intent in intents):
                return None
            head = self.ledger.head()
            if head is None:
                choice = self.ledger.pick(self.ceiling)
                if choice is not None and "add_to_cart" in self.tool_names:
                    return Intent.on("add_to_cart", choice)
                return None
            if self.ledger.unusable(head, self.ceiling):
                if "remove_from_cart" in self.tool_names:
                    return Intent.on("remove_from_cart", head)
                return None
            if self.unverified == head and "inspect_stock" in self.tool_names:
                return Intent.on("inspect_stock", head)
            return None
        except Exception:
            return None

    def _plan(self, turn: int) -> List[Intent]:
        if self._spent_out():
            final = self._last_ditch()
            if final:
                self.claimed = None
                trace(
                    f"turn {turn}: {self._clock():.0f}s gone, finishing with "
                    f"{', '.join(intent.name for intent in final)}"
                )
                return final

        if self.claimed is not None:
            intent, self.claimed = self.claimed, None
            trace(f"turn {turn}: reacting with {intent.name}")
            return [intent]

        if self._late():
            forced = self._closing()
            if forced is not None:
                trace(f"turn {turn}: deadline, forcing {forced.name}")
                return [forced]

        self.history.append({"role": "user", "content": self._digest_note()})
        thread = [
            {"role": "system", "content": self._charter()},
            {"role": "user", "content": self.goal},
        ] + self.history

        reply = self.link.ask(thread, self.tools)
        if reply is None:
            trace(f"turn {turn}: retrying on a shorter thread")
            reply = self.link.ask(thread[:2] + self.history[-WINDOW:], self.tools)
        if reply is None:
            fallback = self._unaided()
            if fallback is None:
                return []
            trace(f"turn {turn}: no inference, falling back to {fallback.name}")
            return [fallback]

        spoken = str(reply.get("content") or "")
        raw = as_seq(reply.get("tool_calls"))
        chosen: List[Intent] = []
        room = max(1, min(self.per_turn, 3))
        for call in raw:
            intent = self._from_tool_call(call)
            if intent is None:
                continue
            if intent.name not in GATHERING:
                chosen = chosen or [intent]
                break
            chosen.append(intent)
            if len(chosen) >= room:
                break

        if not chosen:
            self.idle += 1
            self.history.append({"role": "assistant", "content": spoken})
            self.history.append(
                {
                    "role": "user",
                    "content": (
                        "That reply carried no tool call. Send one concrete tool "
                        "call now, chosen from what changed above."
                    ),
                }
            )
            self.dialogue.append({"role": "assistant", "content": spoken})
            fallback = self._unaided() if self.idle >= 2 else None
            return [fallback] if fallback is not None else []

        vetoed = self._vet(chosen)
        if vetoed is not None:
            trace(f"turn {turn}: order vetoed, running {vetoed.name} first")
            self.history.append({"role": "assistant", "content": spoken})
            self.history.append(
                {
                    "role": "user",
                    "content": (
                        f"That order was not sent; {vetoed.name} is running first "
                        "because the world moved since your evidence was gathered. "
                        "Decide again from its result."
                    ),
                }
            )
            return [vetoed]

        answered = {intent.answers for intent in chosen if intent.answers}
        self.history.append(
            {
                "role": "assistant",
                "content": spoken,
                "tool_calls": [
                    call for call in raw if as_map(call).get("id") in answered
                ],
            }
        )
        return chosen

    def _closing(self) -> Optional[Intent]:
        """Reach a terminal state with the best item the facts still allow."""
        try:
            head = self.ledger.head()
            spare = self.ledger.pick(self.ceiling)
            if head is not None:
                if self.ledger.unusable(head, self.ceiling) and spare is not None and (
                    "remove_from_cart" in self.tool_names
                ):
                    return Intent.on("remove_from_cart", head)
                # With nothing better to swap to, churning the cart only burns
                # turns, so take what is already in it.
                if self.finisher and self.ledger.row(head).get("in_stock") is not False:
                    return self._order_intent(head)
            choice = spare or self.ledger.any_pick()
            if choice is not None and "add_to_cart" in self.tool_names:
                return Intent.on("add_to_cart", choice)
            return self.sweep_intent()
        except Exception:
            return None

    # -------------------------------------------------------------- observing

    def _ingest(self, intents: Sequence[Intent], reply: Dict[str, Any]) -> List[Event]:
        """Record observations and collect the events the runtime reported."""
        results = as_seq(reply.get("calls"))
        known = self.ledger.size()
        events: List[Event] = []
        blank = ("", "")
        for index, intent in enumerate(intents):
            slot = results[index] if index < len(results) else {}
            outer = as_map(as_map(slot).get("observation"))
            body = as_map(outer.get("observation"))
            done = bool(outer.get("done"))
            fault = outer.get("error")
            self.ledger.absorb(intent.name, body)
            if intent.name == "inspect_stock" and self.unverified is not None:
                if self.ledger.key_of(body) == self.unverified or not fault:
                    self.unverified = None
            rendered = json.dumps(
                {"result": body, "error": fault, "done": done}, default=str
            )
            if intent.answers:
                self.history.append(
                    {"role": "tool", "tool_call_id": intent.answers, "content": rendered}
                )
            else:
                self.history.append(
                    {"role": "user", "content": f"{intent.name} returned {rendered}"}
                )
            if fault:
                events.append(Event("call_failed", blank, None, fault, intent.name))
            elif intent.name in ("search", "filter") and not as_seq(body.get("results")):
                events.append(Event("no_matches", blank, None, None, intent.name))
            if done:
                events.append(Event("order_placed", blank, None, None, intent.name))
            if body.get("rejected"):
                events.append(
                    Event("call_failed", blank, None, body["rejected"], intent.name)
                )
        if any(intent.name in DISCOVERY for intent in intents):
            if self.ledger.size() > known:
                self.stagnant = 0
            else:
                self.stagnant += 1
                trace(f"discovery surfaced nothing new ({self.stagnant} in a row)")

        spoken = as_map(reply.get("user_message")).get("content")
        if spoken:
            text = str(spoken)
            self.history.append({"role": "user", "content": text})
            events.append(Event("shopper_spoke", blank, None, None, text))
            self.stagnant = 0
        return events

    def _charge(self, intents: Sequence[Intent], reply: Dict[str, Any]) -> None:
        try:
            before = self.spent
            claimed = [
                as_span(reply.get("action_count"), 0),
                as_span(reply.get("solver_turn_count"), 0),
            ]
            self.spent = max([before + 1] + claimed)
            if len(intents) > 1 and self.spent - before >= len(intents):
                trace("budget is charged per action; gathering one at a time")
                self.per_turn = 1
        except Exception:
            self.spent += 1

    def _log_turn(
        self, turn: int, intents: Sequence[Intent], reply: Dict[str, Any]
    ) -> None:
        self.dialogue.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": intent.answers or f"{self.task}-{turn}-{index}",
                        "type": "function",
                        "function": {
                            "name": intent.name,
                            "arguments": json.dumps(intent.args, default=str),
                        },
                    }
                    for index, intent in enumerate(intents, start=1)
                ],
                "environment_result": reply,
            }
        )

    # ------------------------------------------------------------------ loop

    def run(self) -> List[Dict[str, Any]]:
        trace(
            f"task {self.task}: limit={self.limit} per_turn={self.per_turn} "
            f"ceiling={self.ceiling} finisher={self.finisher or 'unknown'}"
        )
        for turn in range(1, self.limit + 1):
            if self.spent >= self.limit or self.ledger.ordered or self.finished:
                break

            intents = self._plan(turn)
            if not intents:
                self.idle += 1
                if self.idle >= 4:
                    trace(f"turn {turn}: out of options, stopping")
                    break
                continue

            before = self.ledger.snapshot()
            cart_before = list(self.ledger.cart)

            reply = self.link.send(self._envelope(turn, intents))
            if reply is None and len(intents) > 1:
                trace(f"turn {turn}: batch refused, resending one action")
                intents = intents[:1]
                self.per_turn = 1
                reply = self.link.send(self._envelope(turn, intents))
            if reply is None:
                trace(f"turn {turn}: environment unreachable, stopping")
                break

            self._log_turn(turn, intents, reply)
            events = self._ingest(intents, reply)
            events.extend(self.ledger.changes(before, cart_before))
            self._dispatch(events)
            self._charge(intents, reply)
            trace(
                f"turn {turn}: {', '.join(intent.name for intent in intents)} -> "
                f"{len(events)} events "
                f"[{', '.join(sorted({event.kind for event in events})) or 'none'}] "
                f"spent={self.spent}/{self.limit}"
            )
            if self.ledger.ordered or self.finished:
                break

        return self.dialogue


def agent_main(problem_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run one generated-environment episode."""
    solver: Optional[Solver] = None
    try:
        solver = Solver(problem_data)
        return solver.run()
    except Exception as failure:
        trace(f"episode ended by {type(failure).__name__}: {failure}")
        if solver is not None:
            try:
                return solver.dialogue
            except Exception:
                return []
        return []


__all__ = ["agent_main"]
