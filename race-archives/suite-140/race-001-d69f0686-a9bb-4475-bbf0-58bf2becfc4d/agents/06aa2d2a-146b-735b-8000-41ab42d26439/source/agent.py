"""Leona agent for ORO Bench validator-owned generated environments.

Same interaction workflow as src/agent/environment_agent.py:

    policy_view -> inference over dynamic tools -> /environment/call
                -> observe -> repeat until an observation reports done=true

The structure is our own. An _Episode owns the message ledger as discrete turn
groups (so a refused request can be retried with less context), a _Facts table
built only from returned observations, and a close-out policy that converts a
verified cart into an order before the public step limit expires.

Nothing in here may raise out of agent_main: an episode that dies early is
recorded as agent_error and scores zero, which costs more than any single
bad action.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.agent.proxy_client import ProxyClient


# Fallback model per provider, used when SANDBOX_MODEL is unset.
_PROVIDER_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}

# Generated goals carry their own rubric after this marker.
_RULES_MARKER = "Task rules:"

# Actions that only read state, so several may share one envelope.
_READ_ACTIONS = frozenset(
    {"search", "filter", "view", "compare", "inspect_stock", "inspect_cart"}
)

# Turns held in reserve so a verified cart can still become an order. Enough
# for the full close-out sequence: add, re-read the price and stock, order.
_CLOSE_OUT_RESERVE = 4

# Every episode must reach a terminal state well inside this wall clock. The
# soft share switches from browsing to closing out; past the hard share the
# only goal left is to finish.
_TIME_BUDGET = 300.0
_SOFT_SHARE = 0.60
_HARD_SHARE = 0.86

# Discovery calls that add no variant the agent had not already seen are
# counted as stagnant. Enough of them in a row means searching is not working
# and the turns are better spent committing.
_DISCOVERY = frozenset({"search", "filter"})
_STAGNANT_LIMIT = 6

# Turn groups kept when an oversized request is retried smaller.
_TRIM_KEEP_GROUPS = 6

# Upper bound on batched read actions, well under max_calls_per_turn.
_MAX_BATCH = 4

_AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:[A-Z]{3})\b")
_CEILING = re.compile(
    r"(?:under|below|up to|at most|no more than|max(?:imum)?|budget|spend)\W{0,20}$",
    re.IGNORECASE,
)


def _log(message: str) -> None:
    """Instrumentation only; never changes the returned dialogue."""
    try:
        print(f"[leona] {message}", flush=True)
    except Exception:
        pass


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, bool) or value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _unwrap(call_result: Any) -> Tuple[bool, Any, Dict[str, Any]]:
    """Split one call result into (done, error, payload).

    The runtime double-wraps: calls[i]["observation"] holds done and error
    alongside the real tool payload under a second "observation" key.
    """
    try:
        wrapper = _as_dict(_as_dict(call_result).get("observation"))
        payload = _as_dict(wrapper.get("observation"))
        return bool(wrapper.get("done")), wrapper.get("error"), payload
    except Exception:
        return False, None, {}


def _call_name(tool_call: Any) -> str:
    try:
        function = _as_dict(_as_dict(tool_call).get("function"))
        return str(function.get("name") or "")
    except Exception:
        return ""


def _call_args(tool_call: Any) -> Dict[str, Any]:
    """Parse model-supplied arguments without ever failing the turn."""
    try:
        function = _as_dict(_as_dict(tool_call).get("function"))
        raw = function.get("arguments")
        if raw is None or raw == "":
            return {}
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _tool_names(tools: Sequence[Any]) -> List[str]:
    names: List[str] = []
    try:
        for tool in _as_list(tools):
            name = _as_dict(_as_dict(tool).get("function")).get("name")
            if isinstance(name, str) and name:
                names.append(name)
    except Exception:
        pass
    return names


def _tool_properties(tool: Any) -> Dict[str, Any]:
    try:
        function = _as_dict(_as_dict(tool).get("function"))
        parameters = _as_dict(function.get("parameters"))
        return _as_dict(parameters.get("properties"))
    except Exception:
        return {}


def _terminal_contract(tools: Sequence[Any]) -> Tuple[str, List[str]]:
    """Find the session-ending tool and its allowed justification fields.

    Both are read from the task's own schema because the claim vocabulary
    differs between tasks; a fixed list breaks on the other variant.
    """
    fallback = ""
    try:
        for tool in _as_list(tools):
            name = str(_as_dict(_as_dict(tool).get("function")).get("name") or "")
            properties = _tool_properties(tool)
            justification = _as_dict(properties.get("justification"))
            if not justification:
                if not fallback and "order" in name.lower():
                    fallback = name
                continue
            claims = _as_dict(_as_dict(justification.get("properties")).get("claims"))
            items = _as_dict(claims.get("items"))
            field = _as_dict(_as_dict(items.get("properties")).get("field"))
            allowed = [str(value) for value in _as_list(field.get("enum"))]
            return name, allowed
    except Exception:
        pass
    return fallback, []


def _spend_ceiling(query: str) -> Optional[float]:
    """Largest budget the goal states, preferring an explicit ceiling phrase."""
    try:
        text = query or ""
        anchored: List[float] = []
        every: List[float] = []
        for match in _AMOUNT.finditer(text):
            amount = _as_float(match.group(1).replace(",", ""))
            if amount is None:
                continue
            every.append(amount)
            if _CEILING.search(text[max(0, match.start() - 40) : match.start()]):
                anchored.append(amount)
        if anchored:
            return min(anchored)
        if len(every) == 1:
            return every[0]
        return min(every) if every else None
    except Exception:
        return None


def _rules_text(query: str) -> str:
    try:
        if _RULES_MARKER in (query or ""):
            return query.split(_RULES_MARKER, 1)[1].strip()
    except Exception:
        pass
    return ""


class _Transport:
    """Proxy access with a retry budget sized for 429 backpressure."""

    def __init__(self) -> None:
        self._client = ProxyClient(
            timeout=60,
            max_retries=4,
            retry_delay=2,
            rate_limit_retry_delay=8,
            api_key=getenv("INFERENCE_ACCESS_TOKEN") or getenv("CHUTES_ACCESS_TOKEN"),
        )
        provider = getenv("INFERENCE_PROVIDER", "chutes")
        self.model = getenv("SANDBOX_MODEL") or _PROVIDER_MODELS.get(
            provider, _PROVIDER_MODELS["chutes"]
        )

    def choose(
        self, messages: Sequence[Any], tools: Sequence[Any]
    ) -> Optional[Dict[str, Any]]:
        """Return the assistant message, or None when inference is unusable."""
        try:
            response = self._client.post(
                "/inference/chat/completions",
                json_data={
                    "model": self.model,
                    "messages": list(messages),
                    "tools": list(tools),
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            choices = _as_list(_as_dict(response).get("choices"))
            if not choices:
                return None
            return _as_dict(_as_dict(choices[0]).get("message"))
        except Exception as error:
            _log(f"inference raised {type(error).__name__}: {error}")
            return None

    def execute(self, envelope: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            response = self._client.post("/environment/call", json_data=envelope)
            return _as_dict(response) if response is not None else None
        except Exception as error:
            _log(f"environment call raised {type(error).__name__}: {error}")
            return None


class _Facts:
    """Variant facts and cart contents, assembled only from observations."""

    def __init__(self) -> None:
        self._rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._cart: List[Tuple[str, str]] = []
        self._order_placed = False
        # A cart mutation invalidates every earlier price/stock reading, so
        # freshness is tracked per cart generation rather than as a flag.
        self._epoch = 0
        self._checked: Dict[Tuple[str, str], int] = {}

    def absorb(self, action: str, payload: Dict[str, Any]) -> None:
        try:
            if action in ("add_to_cart", "remove_from_cart"):
                self._epoch += 1
            for row in _as_list(payload.get("results")):
                self._record(_as_dict(row))
            self._record(payload)
            self._record(_as_dict(payload.get("added")))
            for row in _as_list(payload.get("variants")):
                self._record(_as_dict(row), parent=payload.get("product_id"))
            for row in _as_list(payload.get("products")):
                self._record(_as_dict(row))
            if isinstance(payload.get("cart"), list):
                self._absorb_cart(_as_list(payload.get("cart")))
            if payload.get("order"):
                self._order_placed = True
            if action == "inspect_stock":
                key = self._key(payload)
                if key:
                    self._checked[key] = self._epoch
        except Exception:
            pass

    def checked(self, key: Optional[Tuple[str, str]]) -> bool:
        """True when this variant was re-read since the cart last changed."""
        return key is not None and self._checked.get(key) == self._epoch

    def cart_head(self) -> Optional[Tuple[str, str]]:
        return self._cart[0] if self._cart else None

    def row(self, key: Optional[Tuple[str, str]]) -> Dict[str, Any]:
        return dict(self._rows.get(key) or {}) if key is not None else {}

    def ruled_out(self, key: Optional[Tuple[str, str]], ceiling: Optional[float]) -> bool:
        """Current facts say this variant cannot be bought as it stands."""
        try:
            row = self.row(key)
            if not row:
                return False
            if row.get("in_stock") is False:
                return True
            price = _as_float(row.get("price"))
            return ceiling is not None and price is not None and price > ceiling
        except Exception:
            return False

    def _absorb_cart(self, rows: Sequence[Any]) -> None:
        cart: List[Tuple[str, str]] = []
        for row in rows:
            entry = _as_dict(row)
            self._record(entry)
            key = self._key(entry)
            if key:
                cart.append(key)
        self._cart = cart

    def _key(self, row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        product_id = row.get("product_id")
        sku = row.get("sku")
        if not product_id or not sku:
            return None
        return str(product_id), str(sku)

    def _record(self, row: Dict[str, Any], parent: Any = None) -> None:
        if not row:
            return
        if parent is not None and not row.get("product_id"):
            row = dict(row)
            row["product_id"] = parent
        key = self._key(row)
        if key is None:
            return
        current = self._rows.setdefault(
            key, {"product_id": key[0], "sku": key[1], "seen": 0}
        )
        current["seen"] = int(current.get("seen") or 0) + 1
        for field in ("title", "brand", "currency"):
            if row.get(field) is not None:
                current[field] = row[field]
        for field in ("price", "price_at_add", "price_at_order"):
            amount = _as_float(row.get(field))
            if amount is not None:
                current["price"] = amount
        if isinstance(row.get("in_stock"), bool):
            current["in_stock"] = row["in_stock"]

    @property
    def order_placed(self) -> bool:
        return self._order_placed

    def cart_keys(self) -> List[Tuple[str, str]]:
        return list(self._cart)

    def size(self) -> int:
        return len(self._rows)

    def any_candidate(self) -> Optional[Dict[str, Any]]:
        """Cheapest variant known, ceiling ignored, in stock preferred.

        Used only as a last resort. An episode that ends without a terminal
        state scores nothing at all, so an imperfect order beats no order.
        """
        try:
            pool = []
            for key, row in self._rows.items():
                if key in self._cart:
                    continue
                price = _as_float(row.get("price"))
                pool.append(
                    (
                        0 if row.get("in_stock") is not False else 1,
                        price if price is not None else float("inf"),
                        row,
                    )
                )
            if not pool:
                return None
            pool.sort(key=lambda item: (item[0], item[1]))
            return pool[0][2]
        except Exception:
            return None

    def viable_cart_entry(self, ceiling: Optional[float]) -> Optional[Dict[str, Any]]:
        """Cart entry that current facts say is orderable right now."""
        try:
            for key in self._cart:
                row = self._rows.get(key)
                if not row or row.get("in_stock") is False:
                    continue
                if not self.checked(key):
                    continue
                price = _as_float(row.get("price"))
                if ceiling is not None and price is not None and price > ceiling:
                    continue
                return row
            return None
        except Exception:
            return None

    def best_candidate(self, ceiling: Optional[float]) -> Optional[Dict[str, Any]]:
        """Cheapest variant outside the cart that current facts allow."""
        try:
            pool = []
            for key, row in self._rows.items():
                if key in self._cart or row.get("in_stock") is False:
                    continue
                price = _as_float(row.get("price"))
                if ceiling is not None and price is not None and price > ceiling:
                    continue
                pool.append((price if price is not None else float("inf"), row))
            if not pool:
                return None
            pool.sort(key=lambda item: item[0])
            return pool[0][1]
        except Exception:
            return None


class _Episode:
    """One task: drives the turn loop and accumulates the returned dialogue."""

    def __init__(self, problem_data: Dict[str, Any]) -> None:
        environment = _as_dict(_as_dict(problem_data).get("environment"))
        policy = _as_dict(environment.get("policy_view"))

        self.binding = _as_dict(environment.get("binding"))
        self.tools = _as_list(policy.get("tools"))
        self.query = str(policy.get("query") or _as_dict(problem_data).get("query") or "")
        self.max_steps = _as_int(policy.get("max_steps"), 20)
        self.max_calls = _as_int(policy.get("max_calls_per_turn"), 1)
        self.problem_id = str(
            _as_dict(problem_data).get("problem_id")
            or _as_dict(problem_data).get("id")
            or "problem"
        )

        self.ceiling = _spend_ceiling(self.query)
        self.terminal_tool, self.claim_fields = _terminal_contract(self.tools)
        self.tool_names = _tool_names(self.tools)

        self._transport = _Transport()
        self._facts = _Facts()
        self._prelude: List[Dict[str, Any]] = []
        self._groups: List[List[Dict[str, Any]]] = []
        self._dialogue: List[Dict[str, Any]] = []
        self._consumed = 0
        self._batching = self.max_calls > 1
        self._blind_turns = 0
        self._started = time.monotonic()
        self._stagnant = 0

    # ---------------------------------------------------------------- prompt

    def _system_prompt(self) -> str:
        try:
            rules = _rules_text(self.query)
            lines = [
                "You are a shopping agent working inside a generated environment.",
                "Call only the supplied tools; never invent a tool name or argument.",
                f"Tools available this task: {', '.join(self.tool_names) or 'none listed'}.",
                f"You have at most {self.max_steps} turns. Spend them on evidence, then commit.",
            ]
            if self.ceiling is not None:
                lines.append(
                    f"A spend ceiling of {self.ceiling} appears in the goal; treat the "
                    "shopper's stated budget as firm and check the exact variant's "
                    "current price against it."
                )
            lines.extend(
                [
                    "",
                    "How to work:",
                    "- Separate firm requirements from soft preferences, and never trade a"
                    " firm requirement away for a nicer soft one.",
                    "- Verify the specific variant you intend to buy, not the parent"
                    " listing: one title often sells both the right and the wrong spec.",
                    "- Trust structured catalog classification and raw product facts over"
                    " marketing words in a title.",
                    "- Prices and stock change while you shop. Re-check the current price"
                    " and stock of your pick after any cart action.",
                    "- If an observation carries an error or an empty result set, change"
                    " approach instead of repeating the same call.",
                    "- Treat a shopper message as high priority and apply it to every"
                    " later choice.",
                    "- Your first cart choice is graded on its own. Do not add a"
                    " placeholder you intend to replace.",
                ]
            )
            if self.terminal_tool:
                lines.append(
                    f"- Finish by calling {self.terminal_tool} once your pick is verified"
                    " in stock and within budget. Do not wait for permission."
                )
            if self.claim_fields:
                lines.append(
                    "- If the shopper asked why, attach justification claims using only"
                    f" these fields: {', '.join(self.claim_fields)}. Every claim must be"
                    " true against facts you actually observed, as field and value only,"
                    " with no prose."
                )
            if self._batching:
                lines.append(
                    f"- You may request up to {min(self.max_calls, _MAX_BATCH)}"
                    " information-gathering calls in one turn, but send any cart or order"
                    " action on its own."
                )
            if rules:
                lines.extend(["", "Rules stated by this task:", rules])
            return "\n".join(lines)
        except Exception:
            return "Use the supplied shopping tools until an observation reports done=true."

    def _messages(self) -> List[Dict[str, Any]]:
        flat = list(self._prelude)
        for group in self._groups:
            flat.extend(group)
        return flat

    def _trimmed_messages(self) -> List[Dict[str, Any]]:
        """Same conversation with the oldest whole turn groups dropped."""
        flat = list(self._prelude)
        for group in self._groups[-_TRIM_KEEP_GROUPS:]:
            flat.extend(group)
        return flat

    # ---------------------------------------------------------------- acting

    def _envelope(self, turn: int, actions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        group_id = f"{self.problem_id}-turn-{turn}"
        return {
            **self.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {"name": action["name"], "args": action["args"]},
                }
                for index, action in enumerate(actions, start=1)
            ],
        }

    def _batch(self, tool_calls: Sequence[Any]) -> List[Dict[str, Any]]:
        """Keep a run of read-only calls, or a single state-changing call."""
        actions: List[Dict[str, Any]] = []
        limit = min(self.max_calls, _MAX_BATCH) if self._batching else 1
        for tool_call in tool_calls:
            name = _call_name(tool_call)
            if not name:
                continue
            action = {
                "name": name,
                "args": _call_args(tool_call),
                "id": _as_dict(tool_call).get("id"),
            }
            if name not in _READ_ACTIONS:
                if not actions:
                    actions.append(action)
                break
            actions.append(action)
            if len(actions) >= limit:
                break
        return actions

    def _elapsed(self) -> float:
        try:
            return time.monotonic() - self._started
        except Exception:
            return 0.0

    def _pressed(self, remaining: int) -> bool:
        """Time to stop gathering: turns, wall clock or fruitless searching."""
        return (
            remaining <= _CLOSE_OUT_RESERVE
            or self._elapsed() >= _TIME_BUDGET * _SOFT_SHARE
            or self._stagnant >= _STAGNANT_LIMIT
        )

    def _spent_out(self, remaining: int) -> bool:
        """Nothing matters now except reaching a terminal state."""
        return remaining <= 2 or self._elapsed() >= _TIME_BUDGET * _HARD_SHARE

    def _order_action(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": self.terminal_tool,
            "args": {"product_id": row.get("product_id"), "sku": row.get("sku")},
            "id": None,
        }

    def _last_ditch(self) -> List[Dict[str, Any]]:
        """One envelope that reaches a terminal state, verified or not.

        Adding and ordering travel together here: a single round trip is all
        the remaining budget can pay for, and an unfinished episode is worth
        less than an imperfect purchase.
        """
        try:
            names = set(self.tool_names)
            head = self._facts.cart_head()
            if head is not None and self.terminal_tool:
                return [self._order_action(self._facts.row(head))]
            choice = self._facts.best_candidate(self.ceiling) or (
                self._facts.any_candidate()
            )
            if choice is None or "add_to_cart" not in names:
                return []
            filling = {
                "name": "add_to_cart",
                "args": {
                    "product_id": choice.get("product_id"),
                    "sku": choice.get("sku"),
                },
                "id": None,
            }
            if not self.terminal_tool:
                return [filling]
            return [filling, self._order_action(choice)]
        except Exception:
            return []

    def _close_out_action(self) -> Optional[Dict[str, Any]]:
        """Drive an episode to a terminal state without consulting the model.

        Two stages, because a browse-only episode reaches the reserve with an
        empty cart: fill the cart from the best known candidate, then order it
        on the following turn. Justification is omitted, since it is optional
        by contract and an unverified claim costs more than a missing one.
        """
        entry = self._facts.viable_cart_entry(self.ceiling)
        if entry is not None and self.terminal_tool:
            return {
                "name": self.terminal_tool,
                "args": {
                    "product_id": entry.get("product_id"),
                    "sku": entry.get("sku"),
                },
                "id": None,
            }
        stale = self._verify_action()
        if stale is not None:
            return stale
        head = self._facts.cart_head()
        affordable = self._facts.best_candidate(self.ceiling)
        if head is not None and affordable is None and self.terminal_tool:
            # Nothing affordable is left to swap to, so churning the cart only
            # burns turns. Take what is already in it.
            row = self._facts.row(head)
            if row.get("in_stock") is not False:
                return self._order_action(row)
        candidate = affordable or self._facts.any_candidate()
        if candidate is not None and "add_to_cart" in set(self.tool_names):
            return {
                "name": "add_to_cart",
                "args": {
                    "product_id": candidate.get("product_id"),
                    "sku": candidate.get("sku"),
                },
                "id": None,
            }
        return None

    def _verify_action(self) -> Optional[Dict[str, Any]]:
        """Re-read the cart item when the cart changed after its last reading."""
        head = self._facts.cart_head()
        if head is None or self._facts.checked(head):
            return None
        if "inspect_stock" not in set(self.tool_names):
            return None
        return {
            "name": "inspect_stock",
            "args": {"product_id": head[0], "sku": head[1]},
            "id": None,
        }

    def _hold_back(
        self, actions: Sequence[Dict[str, Any]], remaining: int
    ) -> Optional[Dict[str, Any]]:
        """Intercept an order placed on a reading taken before the cart changed.

        Under deadline pressure the order goes through regardless: reaching a
        terminal state beats leaving the episode unfinished.
        """
        try:
            if remaining <= _CLOSE_OUT_RESERVE or not self.terminal_tool:
                return None
            if not any(action["name"] == self.terminal_tool for action in actions):
                return None
            head = self._facts.cart_head()
            if head is None:
                return None
            if not self._facts.checked(head):
                return self._verify_action()
            if self._facts.ruled_out(head, self.ceiling) and (
                "remove_from_cart" in set(self.tool_names)
            ):
                return {
                    "name": "remove_from_cart",
                    "args": {"product_id": head[0], "sku": head[1]},
                    "id": None,
                }
            return None
        except Exception:
            return None

    def _blind_action(self) -> Optional[Dict[str, Any]]:
        """Deterministic next move when inference is unavailable."""
        try:
            self._blind_turns += 1
            closing = self._close_out_action()
            if closing is not None:
                return closing
            names = set(self.tool_names)
            if "search" in names and self._blind_turns <= 3:
                return {
                    "name": "search",
                    "args": {"query": self._seed_query(), "in_stock": True},
                    "id": None,
                }
            if "filter" in names:
                args: Dict[str, Any] = {"in_stock": True}
                if self.ceiling is not None:
                    args["max_price"] = self.ceiling
                return {"name": "filter", "args": args, "id": None}
            return None
        except Exception:
            return None

    def _seed_query(self) -> str:
        """Search text derived from the goal's own opening sentence."""
        try:
            head = (self.query.split(_RULES_MARKER, 1)[0]).strip()
            words = re.findall(r"[A-Za-z][A-Za-z\-]+", head)[:12]
            return " ".join(words) if words else "in stock item"
        except Exception:
            return "in stock item"

    def _decide(self, turn: int) -> List[Dict[str, Any]]:
        remaining = self.max_steps - self._consumed
        late = self._spent_out(remaining)
        if late:
            final = self._last_ditch()
            if final:
                _log(
                    f"turn {turn}: {remaining} turns and "
                    f"{self._elapsed():.0f}s gone, finishing with "
                    f"{', '.join(action['name'] for action in final)}"
                )
                return final
        if self._pressed(remaining):
            closing = self._close_out_action()
            if closing is not None:
                _log(
                    f"turn {turn}: closing out ({remaining} turns left, "
                    f"{self._elapsed():.0f}s, stagnant={self._stagnant})"
                )
                return [closing]

        assistant = self._transport.choose(self._messages(), self.tools)
        if assistant is None and not late:
            _log(f"turn {turn}: retrying inference with a trimmed conversation")
            assistant = self._transport.choose(self._trimmed_messages(), self.tools)

        if assistant is None:
            blind = self._blind_action()
            if blind is None:
                return []
            _log(f"turn {turn}: inference unavailable, acting on {blind['name']}")
            return [blind]

        content = str(assistant.get("content") or "")
        tool_calls = _as_list(assistant.get("tool_calls"))
        if not tool_calls:
            self._nudge(content, remaining)
            blind = self._blind_action()
            return [blind] if blind is not None else []

        actions = self._batch(tool_calls)
        if not actions:
            blind = self._blind_action()
            return [blind] if blind is not None else []

        held = self._hold_back(actions, remaining)
        if held is not None:
            _log(f"turn {turn}: order held back, running {held['name']} first")
            self._groups.append(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "That order was not sent. The cart item's price and stock "
                            "have not been read back since the cart last changed, so "
                            f"{held['name']} is running first. Decide again from the "
                            "result."
                        ),
                    },
                ]
            )
            return [held]

        self._groups.append(
            [
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        tool_call
                        for tool_call in tool_calls
                        if _as_dict(tool_call).get("id")
                        in {action.get("id") for action in actions}
                    ],
                }
            ]
        )
        return actions

    def _nudge(self, content: str, remaining: int) -> None:
        """Record a prose-only reply and ask again for a concrete action."""
        try:
            self._dialogue.append({"role": "assistant", "content": content})
            self._groups.append(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "No observation has reported done=true yet, and "
                            f"{remaining} turns remain. Call one of the supplied "
                            "tools now, choosing the action that most directly "
                            "advances the shopper's goal."
                        ),
                    },
                ]
            )
        except Exception:
            pass

    # ------------------------------------------------------------ observing

    def _absorb(
        self, actions: Sequence[Dict[str, Any]], response: Dict[str, Any]
    ) -> bool:
        """Feed observations back to the model. Returns True when done."""
        results = _as_list(response.get("calls"))
        group: List[Dict[str, Any]] = []
        finished = False
        known = self._facts.size()

        for index, action in enumerate(actions):
            result = results[index] if index < len(results) else {}
            done, error, payload = _unwrap(result)
            self._facts.absorb(str(action.get("name") or ""), payload)
            finished = finished or done
            body = json.dumps(
                {"observation": payload, "error": error, "done": done}, default=str
            )
            if action.get("id"):
                group.append(
                    {"role": "tool", "tool_call_id": action["id"], "content": body}
                )
            else:
                group.append(
                    {
                        "role": "user",
                        "content": f"Result of {action['name']}: {body}",
                    }
                )
            if error:
                group.append(
                    {
                        "role": "user",
                        "content": (
                            f"The {action['name']} call reported an error. Correct the "
                            "arguments or choose a different tool; do not repeat it "
                            "unchanged."
                        ),
                    }
                )

        if any(action["name"] in _DISCOVERY for action in actions):
            if self._facts.size() > known:
                self._stagnant = 0
            else:
                self._stagnant += 1
                _log(f"discovery added nothing new ({self._stagnant} in a row)")

        shopper = _as_dict(response.get("user_message"))
        if shopper.get("content"):
            group.append({"role": "user", "content": str(shopper["content"])})
            self._stagnant = 0

        if group and self._groups and any(
            message.get("role") == "assistant" for message in self._groups[-1]
        ):
            self._groups[-1].extend(group)
        elif group:
            self._groups.append(group)

        return finished

    def _account(self, actions: Sequence[Dict[str, Any]], response: Dict[str, Any]) -> None:
        """Track the runtime's own budget counters and stop batching if needed."""
        try:
            before = self._consumed
            reported = [
                _as_int(response.get("action_count"), 0),
                _as_int(response.get("solver_turn_count"), 0),
            ]
            self._consumed = max([before + 1] + reported)
            if len(actions) > 1 and self._consumed - before >= len(actions):
                _log("runtime charges per action; disabling batched reads")
                self._batching = False
        except Exception:
            self._consumed += 1

    # ----------------------------------------------------------------- loop

    def run(self) -> List[Dict[str, Any]]:
        self._prelude = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self.query},
        ]
        _log(
            f"task {self.problem_id}: max_steps={self.max_steps} "
            f"max_calls={self.max_calls} ceiling={self.ceiling} "
            f"terminal={self.terminal_tool or 'unknown'} tools={len(self.tools)}"
        )

        for turn in range(1, self.max_steps + 1):
            if self._consumed >= self.max_steps or self._facts.order_placed:
                break

            actions = self._decide(turn)
            if not actions:
                _log(f"turn {turn}: no action available, ending episode")
                break

            response = self._transport.execute(self._envelope(turn, actions))
            if response is None and len(actions) > 1:
                _log(f"turn {turn}: batch refused, retrying the first action alone")
                actions = actions[:1]
                self._batching = False
                response = self._transport.execute(self._envelope(turn, actions))
            if response is None:
                _log(f"turn {turn}: environment call failed, ending episode")
                break

            self._dialogue.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": action.get("id") or f"{self.problem_id}-{turn}-{index}",
                            "type": "function",
                            "function": {
                                "name": action["name"],
                                "arguments": json.dumps(action["args"], default=str),
                            },
                        }
                        for index, action in enumerate(actions, start=1)
                    ],
                    "environment_result": response,
                }
            )

            finished = self._absorb(actions, response)
            self._account(actions, response)
            names = ", ".join(action["name"] for action in actions)
            _log(
                f"turn {turn}: {names} -> done={finished} "
                f"consumed={self._consumed}/{self.max_steps}"
            )
            if finished:
                break

        return self._dialogue


def agent_main(problem_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run one ORO Bench episode over the environment's dynamic tool list."""
    episode: Optional[_Episode] = None
    try:
        episode = _Episode(problem_data)
        return episode.run()
    except Exception as error:
        _log(f"episode aborted by {type(error).__name__}: {error}")
        if episode is not None:
            try:
                return episode._dialogue
            except Exception:
                return []
        return []


__all__ = ["agent_main"]
