"""ORO Bench agent v2: an LLM shopper behind a mechanics-only harness.

The model does the shopping. It reads the shopper's request and the task rules exactly as the
environment hands them over, decides what to search, what to read, what to ask, what to buy and
how to justify it. The code around it never interprets the goal text; it only

  * speaks the environment protocol (envelope, turn numbers, idempotent replay, pacing),
  * remembers what the environment has actually returned (prices, stock, cart, market changes,
    shopper messages),
  * enforces invariants any careful shopper obeys and that can be checked with those numbers:
    read a product before committing to it, re-check an item after the market moved, answer a
    question before ordering, verify current price and stock right before the order,
  * finishes the session safely when turns or time run out.

No table in this file names a category, brand, colour, unit, currency, rule sentence or
shopper phrase; every judgement about the merchandise comes from the model's reading of the
observations.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# ------------------------------------------------------------------------------------ config
_ENV = ProxyClient(timeout=60, max_retries=1, retry_delay=1.0)
_LLM = ProxyClient(timeout=45, max_retries=1, retry_delay=1.0)
_TIME_BUDGET_S = 262.0            # the sandbox kills a task at 300 s
_CALL_GAP_S = 0.6                 # environment calls share one proxy address across many workers
_LAST_CALL = 0.0
_OBS_CHARS = 6000
_MAX_REPROMPTS = 2
_READS = {"search", "filter", "view", "compare", "inspect_stock", "inspect_cart"}
# A shopper line that asks something or asks to be answered: a question mark, or a sentence that opens
# with a request to confirm / reply / respond / answer / tell them / let them know.
_ASKS_RE = re.compile(r"[?？]|(?:^|[.!?;]\s*)(?:please\s+)?(?:confirm|reply|respond|answer|tell me|let me know)\b", re.I)
_MODELS = {
    "openrouter": ["deepseek/deepseek-v3.2", "qwen/qwen3-32b"],
    "chutes": ["deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"],
}


def _roster() -> list[str]:
    provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    roster = list(_MODELS.get(provider) or _MODELS["openrouter"])
    pinned = (getenv("SANDBOX_MODEL") or "").strip()
    if pinned and pinned not in roster:
        roster.insert(0, pinned)
    return roster


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 18] + " ...[truncated]"


def _key(pid: Any, sku: Any) -> str:
    return f"{pid}::{sku}"


def _shrink(payload: dict, limit: int) -> tuple[dict, int]:
    """Keep a tool result valid JSON under `limit` chars by dropping whole rows from the end of its lists."""
    dropped = 0
    out = json.loads(json.dumps(payload))
    while len(json.dumps(out)) > limit:
        lists = [(k, v) for k, v in out.items() if isinstance(v, list) and v]
        for row in out.get("compared") or []:
            if isinstance(row, dict) and isinstance(row.get("variants"), list) and len(row["variants"]) > 1:
                lists.append(("compared.variants", row["variants"]))
        if not lists:
            break
        k, v = max(lists, key=lambda kv: len(json.dumps(kv[1])))
        v.pop()
        dropped += 1
        if k in ("results", "compared", "variants"):
            tr = out.setdefault("truncated", {})
            tr[k] = tr.get(k, 0) + 1
    return out, dropped


# ------------------------------------------------------------------------------------ session
class Obs:
    __slots__ = ("name", "args", "data", "error", "user", "done", "turn")

    def __init__(self, name: str, args: dict, data: dict, error: Any, user: str | None, done: bool, turn: int):
        self.name, self.args, self.data, self.error, self.user, self.done, self.turn = name, args, data, error, user, done, turn

    @property
    def ok(self) -> bool:
        return not self.error


class Session:
    """The environment protocol: one action per solver turn, idempotent replay on transport loss."""

    def __init__(self, problem: dict, clock: Clock) -> None:
        env = problem["environment"]
        self.binding = dict(env.get("binding") or {})
        self.policy = env.get("policy_view") or {}
        self.pid = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.tools = self.policy.get("tools") or []
        self.tool_names = {t.get("function", {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.clock = clock
        self.turn = 0
        self.done = False
        self.broken = False
        self.pending: dict | None = None
        self.dialogue: list[dict] = []
        self.jitter = (sum(ord(c) for c in self.pid) % 40) / 100.0

    @property
    def steps_left(self) -> int:
        return self.max_steps - self.turn

    def _pace(self) -> None:
        global _LAST_CALL
        wait = _CALL_GAP_S + self.jitter - (time.monotonic() - _LAST_CALL)
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL = time.monotonic()

    def _post(self, envelope: dict) -> dict | None:
        for attempt in range(14):
            left = self.clock.remaining()
            if left < 12:
                break
            _ENV.timeout = max(8, min(60, int(left - 4)))
            self._pace()
            try:
                res = _ENV.post_verbose("/environment/call", json_data=envelope)
                data, err = res.data, res.error
            except Exception:
                data, err = None, {"status": None}
            if isinstance(data, dict) and data.get("calls"):
                return data
            status = err.get("status") if isinstance(err, dict) else None
            try:
                status = int(status) if status is not None else None
            except (TypeError, ValueError):
                status = None
            if status is not None and 400 <= status < 500 and status != 429:
                return None
            if self.clock.remaining() < 15:
                break
            time.sleep(min(1.5 * (1.6 ** attempt), 12.0) + self.jitter)
        return None

    def call(self, name: str, args: dict, content: str = "", call_id: str | None = None) -> Obs | None:
        got = self.group([(name, args, call_id)], content)
        return got[0] if got else None

    def group(self, actions: list[tuple[str, dict, str | None]], content: str = "") -> list[Obs]:
        """One solver turn carrying one or more actions; the environment runs them in order."""
        if self.done or self.broken or self.turn >= self.max_steps or not actions:
            return []
        if self.pending is not None:
            replay = self._post(self.pending["envelope"])
            if replay is None:
                self.broken = True
                return []
            pend = self.pending
            self.pending = None
            self.turn = pend["turn"]
            self._absorb(pend["actions"], pend["content"], replay)
            if self.done or self.turn >= self.max_steps:
                return []
        turn = self.turn + 1
        gid = f"{self.pid}-turn-{turn}"
        envelope = {
            **self.binding,
            "call_id": gid,
            "idempotency_key": gid,
            "turn": turn,
            "calls": [{"call_id": f"{gid}-{i}", "action": {"name": n, "args": a}} for i, (n, a, _c) in enumerate(actions, start=1)],
        }
        result = self._post(envelope)
        if result is None:
            self.pending = {"envelope": envelope, "turn": turn, "actions": actions, "content": content}
            self.broken = True
            return []
        self.turn = turn
        return self._absorb(actions, content, result)

    def _absorb(self, actions: list[tuple[str, dict, str | None]], content: str, result: dict) -> list[Obs]:
        calls = result.get("calls") or []
        um = result.get("user_message")
        user = str(um.get("content")) if isinstance(um, dict) and um.get("content") else None
        out: list[Obs] = []
        for i, (name, args, _cid) in enumerate(actions):
            c = calls[i] if i < len(calls) and isinstance(calls[i], dict) else {}
            wrap = c.get("observation") if isinstance(c.get("observation"), dict) else {}
            data = wrap.get("observation") if isinstance(wrap.get("observation"), dict) else wrap
            error = wrap.get("error") or (data.get("error") if isinstance(data, dict) else None)
            done = bool(wrap.get("done"))
            if done:
                self.done = True
            out.append(Obs(name, args, data if isinstance(data, dict) else {}, error, user if i == len(actions) - 1 else None, done, self.turn))
        slim = {k: v for k, v in result.items() if k != "calls"}
        slim["calls"] = [{"call_id": c.get("call_id"), "observation": json.loads(_trim(json.dumps(c.get("observation")), 3000)) if len(json.dumps(c.get("observation"))) <= 3000 else {"truncated": _trim(json.dumps(c.get("observation")), 3000)}} for c in calls if isinstance(c, dict)]
        self.dialogue.append({"role": "assistant", "content": content or "",
                              "tool_calls": [{"id": cid or f"{self.pid}-turn-{self.turn}-{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a, cid) in enumerate(actions, start=1)],
                              "environment_result": slim})
        if user:
            self.dialogue.append({"role": "user", "content": user})
        return out


# ------------------------------------------------------------------------------------ ledger
class Ledger:
    """What the environment has said so far, keyed the way the environment keys it."""

    def __init__(self) -> None:
        self.products: dict[str, dict] = {}          # pid -> {title, brand, category_path, facts, variants, listing_id, read}
        self.stock: dict[str, dict] = {}             # key -> {price, in_stock, turn, source}
        self.cart: list[tuple[str, str]] = []
        self.adds: list[tuple[str, str, int]] = []
        self.events: list[dict] = []                 # {turn, kind, targets:[key], old_price, new_price}
        self.last_event_turn = 0
        self.questions: list[tuple[int, str]] = []   # shopper messages that still need a reply
        self.last_agent_message_turn = 0
        self.replies_sent = 0
        self.last_claims: list[dict] = []
        self.calls_seen: set[str] = set()
        self.search_strings: list[str] = []
        self.reads = 0
        self.dead: set[str] = set()
        self.last_question_turn = 0
        self.candidate_missing: set[str] = set()

    def product(self, pid: str) -> dict:
        p = self.products.get(pid)
        if p is None:
            p = {"title": "", "brand": None, "category_path": None, "facts": {}, "variants": {}, "listing_id": None, "read": False}
            self.products[pid] = p
        return p

    def note_stock(self, pid: Any, sku: Any, price: Any, in_stock: Any, turn: int, source: str) -> None:
        if pid is None or sku is None:
            return
        self.stock[_key(pid, sku)] = {"price": _num(price), "in_stock": in_stock, "turn": turn, "source": source}

    def note_product(self, d: dict, turn: int) -> None:
        pid = str(d.get("product_id") or "")
        if not pid:
            return
        p = self.product(pid)
        p["title"] = str(d.get("title") or p["title"])
        p["brand"] = d.get("brand") if d.get("brand") is not None else p["brand"]
        if isinstance(d.get("category_path"), list):
            p["category_path"] = d["category_path"]
        if isinstance(d.get("facts"), dict):
            p["facts"] = d["facts"]
        if d.get("listing_id"):
            p["listing_id"] = d["listing_id"]
        for v in d.get("variants") or []:
            if isinstance(v, dict) and v.get("sku") is not None:
                p["variants"][str(v["sku"])] = {"options": v.get("options"), "price": _num(v.get("price")), "in_stock": v.get("in_stock")}
                self.note_stock(pid, v["sku"], v.get("price"), v.get("in_stock"), turn, "view")
        p["read"] = True

    def ingest(self, obs: Obs) -> None:
        d = obs.data or {}
        if not obs.ok:
            return
        if obs.name == "search":
            q = " ".join(str(obs.args.get("query") or "").lower().split())
            if q and q not in self.search_strings:
                self.search_strings.append(q)
        if obs.name in ("view", "compare"):
            self.reads += 1
        if obs.name in ("search", "filter"):
            for row in d.get("results") or []:
                if isinstance(row, dict) and row.get("product_id"):
                    p = self.product(str(row["product_id"]))
                    p["title"] = p["title"] or str(row.get("title") or "")
                    if row.get("brand") and not p["brand"]:
                        p["brand"] = row.get("brand")
                    if row.get("sku") is not None:
                        p["variants"].setdefault(str(row["sku"]), {"options": None, "price": _num(row.get("price")), "in_stock": row.get("in_stock")})
                        self.note_stock(row["product_id"], row["sku"], row.get("price"), row.get("in_stock"), obs.turn, "search")
        elif obs.name == "view":
            self.note_product(d, obs.turn)
        elif obs.name == "compare":
            for row in d.get("compared") or []:
                if isinstance(row, dict):
                    self.note_product(row, obs.turn)
        elif obs.name == "inspect_stock" and d.get("product_id") is not None:
            self.note_stock(d.get("product_id"), d.get("sku"), d.get("price"), d.get("in_stock"), obs.turn, "inspect_stock")
        elif obs.name in ("add_to_cart", "remove_from_cart", "inspect_cart"):
            cart = d.get("cart")
            if isinstance(cart, list):
                self.cart = [(str(r.get("product_id")), str(r.get("sku"))) for r in cart if isinstance(r, dict)]
            added = d.get("added") or {}
            if obs.name == "add_to_cart" and added.get("product_id") is not None:
                self.adds.append((str(added["product_id"]), str(added.get("sku")), obs.turn))
                self.note_stock(added["product_id"], added.get("sku"), added.get("price"), added.get("in_stock"), obs.turn, "add_to_cart")
            ev = d.get("market_event")
            if isinstance(ev, dict) and isinstance(ev.get("target"), dict):
                keys = [_key(ev["target"].get("product_id"), ev["target"].get("sku"))]
                for t in ev.get("targets") or []:
                    if isinstance(t, str) and "::" in t:
                        keys.append(t)
                    elif isinstance(t, dict):
                        keys.append(_key(t.get("product_id"), t.get("sku")))
                keys = list(dict.fromkeys(keys))
                self.events.append({"turn": obs.turn, "kind": ev.get("kind"), "targets": keys,
                                    "old_price": ev.get("old_price"), "new_price": ev.get("new_price")})
                self.last_event_turn = obs.turn
                for k in keys:
                    kp, _, ks = k.partition("::")
                    if ev.get("kind") == "stockout":
                        self.note_stock(kp, ks, ev.get("old_price"), False, obs.turn, "market_event")
                    elif ev.get("new_price") is not None:
                        self.note_stock(kp, ks, ev.get("new_price"), ev.get("in_stock_after", True), obs.turn, "market_event")
                    self.dead.add(k)
        if obs.name in ("view", "compare") and obs.error:
            for pid in ([str(obs.args.get("product_id"))] if obs.name == "view" else [str(x) for x in (obs.args.get("product_ids") or [])]):
                self.candidate_missing.add(pid)
        if obs.name == "message":
            self.last_agent_message_turn = obs.turn
            self.replies_sent += 1
            if _ASKS_RE.search(str(obs.args.get("content") or "")):
                self.last_question_turn = obs.turn
        if obs.user and _ASKS_RE.search(obs.user):
            self.questions.append((obs.turn, obs.user))

    def unanswered_question(self) -> str | None:
        """A shopper question the agent has not replied to since it arrived (only `message` reaches the shopper)."""
        if self.replies_sent >= 6:
            return None
        for turn, text in self.questions:
            if turn >= self.last_agent_message_turn:
                return text
        return None

    def refresh_dead(self, budget: float | None) -> None:
        """Carted variants the store's latest reading shows as gone or over budget."""
        for pid, sku, _t in self.adds:
            st = self.stock.get(_key(pid, sku)) or {}
            if st.get("in_stock") is False or (budget is not None and st.get("price") is not None and st["price"] > budget + 1e-6):
                self.dead.add(_key(pid, sku))

    def fresh_reading(self, pid: str, sku: str) -> dict | None:
        """The latest price/stock reading of a variant taken after the last market change and after it was carted."""
        s = self.stock.get(_key(pid, sku))
        if not s or s["source"] not in ("inspect_stock", "view"):
            return None
        last_add = max([t for p, k, t in self.adds if p == pid and k == sku] or [0])
        if s["turn"] <= max(self.last_event_turn, last_add):
            return None
        return s


# ------------------------------------------------------------------------------------ agent
_SYSTEM = """You are a shopping agent operating an online store through tools on behalf of a shopper.
The shopper's message below contains the request and the task rules; treat both as the specification and follow them literally.

How to work:
1. Requirements come only from the shopper's words and the rules. If the shopper says a requirement or priority will be given when asked, or the rules tell you to clarify before your first cart action, send ONE message before your first add_to_cart that asks directly for their firm requirements, specs and priorities, as a question ending in "?". Only text sent through the message tool reaches the shopper; anything written outside it is discarded. A silent or generic reply means nothing changed. Otherwise do not ask for permission, confirmation or approval: the shopper cannot approve a choice.
2. Read a product with view or compare before you rely on it: check the category path, the raw facts and the exact variant's options, price and stock against every requirement. Titles can be misleading; facts and variant options decide.
3. The first item you add to the cart is treated as your real choice, not a placeholder. Only add an item you have read and that meets every stated requirement (and the stated preferences when the rules make them matter); a later correction does not repair a careless first choice. Read a focused shortlist (you may request up to four view calls in one reply), pick the best, add it; do not keep exploring once you have a candidate that meets everything. If the shopper lists specific candidates, work only with those. In the reply that adds your first item, write out the shortlist with each candidate's decisive values as you read them (a value stated in the selected variant's option text counts before a product-wide fact, which counts before the title) and the rule you applied, then call add_to_cart.
4. Prices and stock change while you shop, typically right after a cart action, and a change is not always announced. After any cart action whose result did not announce a change, re-read the item with inspect_stock in its own turn before believing its price or stock. If it no longer fits the budget or is out of stock, move to a different product that meets everything (a different product, and where the rules define listing identity a different listing), respecting any rule about which replacements are acceptable.
5. When a shopper message asks you something or asks you to confirm, answer or let them know, answer it with one short message call placed FIRST in your next reply, followed by your next read, cart action or order; they run together in one turn, so answering never costs a turn. Then apply any changed requirement it states (a new budget or spec replaces the old one).
6. To commit: add_to_cart the exact variant; in the very next turn inspect_stock that same variant, unless the add result itself announced a change on it (then it is gone: skip the re-check and add your replacement). Only a fresh reading counts; if it is in stock and within budget, place_test_order it, otherwise treat it as gone and pick a different product. One test order ends the session.
7. Supply justification claims only when the order tool requires them. Claims are field/value pairs only, using the field names the tool describes; a specification claim uses the tool's standard attribute name (screen_size, resolution, refresh_rate or ram), never the store's fact label or a made-up name; copy each value exactly from the store's observation of that item (its facts, variant options, current price, stock and brand fields); never claim a feature the store's text does not state, and no summary text.
8. When a specification matters, confirm the exact variant's value from the facts and the variant option text; if the sources disagree or the value is missing, prefer a product whose facts state it plainly.
9. When the rules ask you to survey several distinct matching options, find them through search or filter results, count identities the way the rules define them, read only listings whose title and facts actually match, and do not repeat searches you already made.
10. Turns are limited. You may request several read calls (search, filter, view, compare, inspect_stock) in one reply and they run together in one turn; cart actions, messages and the order run alone. Make your first add_to_cart no later than the turn the [status] line names; time, not only turns, is limited. A dead item can stay in the cart: do not spend a turn removing it, just add the replacement and name it in place_test_order. Do not repeat reads you already have; keep messages short."""


class Agent:
    def __init__(self, problem: dict) -> None:
        self.clock = Clock(_TIME_BUDGET_S)
        self.session = Session(problem, self.clock)
        self.ledger = Ledger()
        self.query = str(self.session.policy.get("query") or problem.get("query") or "")
        self.roster = _roster()
        self.model_index = 0
        self.messages: list[dict] = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": self.query}]
        self.brief: dict = {}
        self.notes: list[str] = []
        self.rejections = 0
        self._pushed = False
        self._warned: set[str] = set()
        self.turn_secs: list[float] = []
        self._late_add_tried = False
        self._pending_content = ""
        self._turn_started = time.monotonic()
        self.commit_by = 10 ** 6

    # ------------------------------------------------------------------ model calls
    def _chat(self, messages: list[dict], tools: list | None, max_tokens: int = 700) -> dict | None:
        fails = 0
        for attempt in range(3):
            left = self.clock.remaining()
            if left < 45:
                return None
            _LLM.timeout = max(10, min(45, int(left - 25)))   # a stalled provider must not eat the task budget
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            payload: dict = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "required"
            resp = _LLM.post("/inference/chat/completions", json_data=payload)
            try:
                msg = resp["choices"][0]["message"]
                if isinstance(msg, dict):
                    return msg
            except Exception:
                pass
            fails += 1
            if fails >= 2:
                self.model_index = min(self.model_index + 1, len(self.roster) - 1)
            time.sleep(1.0 + attempt)
        return None

    def _json_call(self, instruction: str, text: str) -> dict:
        msg = self._chat([{"role": "system", "content": instruction}, {"role": "user", "content": text}], None, 500)
        content = (msg or {}).get("content") or ""
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end < 0:
            return {}
        try:
            out = json.loads(content[start:end + 1])
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}

    def extract_brief(self) -> None:
        """The model reads the request once and returns the few numbers the harness needs for its checks."""
        self.brief = self._json_call(
            "Read the shopper's request and rules. Reply with one JSON object only: "
            '{"budget": <maximum price the shopper will pay, as a number, or null>, '
            '"ask_first": <true if the shopper says a requirement or priority will be given only when asked, or the rules require you to clarify with the shopper before your first cart action, else false>, '
            '"candidate_ids": <identifiers the shopper explicitly lists as the only acceptable products, quoted verbatim, else []>, '
            '"firm_brand": <the brand name if the shopper requires a specific brand as a firm requirement (not a mere preference), else null>}',
            self.query,
        ) or {}
        b = _num(self.brief.get("budget"))
        self.brief["budget"] = b if (b is not None and b > 0 and self._number_in(b, self.query)) else None
        low = self.query.lower()
        cands = self.brief.get("candidate_ids") if isinstance(self.brief.get("candidate_ids"), list) else []
        pairs = []
        for x in cands:
            x = str(x).strip()
            pid_, _, sku_ = x.partition("::")
            pid_, sku_ = pid_.strip(), sku_.strip()
            if pid_ and pid_ in self.query and (not sku_ or f"{pid_}::{sku_}" in self.query):
                pairs.append((pid_, sku_ or None))
        self.brief["candidate_pairs"] = pairs if len(pairs) >= 2 else []
        self.brief["candidate_ids"] = [pp for pp, _ in self.brief["candidate_pairs"]]
        fb = self.brief.get("firm_brand")
        fb = str(fb).strip() if isinstance(fb, str) else ""
        self.brief["firm_brand"] = fb if fb and fb.lower() not in ("null", "none") and fb.lower() in low else None
        self.brief["required_title_words"] = []

    @staticmethod
    def _number_in(value: float, text: str) -> bool:
        """The store renders money with two decimals; accept a number only if the text really states it."""
        for form in ("%.2f" % value, "%g" % value):
            if re.search(r"(?<![\d.])" + re.escape(form) + r"(?![\d])", text):
                return True
        return False

    def update_brief(self, text: str, turn: int) -> None:
        if not re.search(r"\d", text):
            return
        upd = self._json_call(
            "A shopper sent this message during shopping. Reply with one JSON object only: "
            '{"budget": <a number only if the shopper sets a new spending limit for the purchase; '
            'a product price, a price change, an old or new price of an item, or a restated unchanged limit is null>, '
            '"market_change": <true if the message reports that an item changed price or went out of stock, else false>}',
            text,
        )
        if not isinstance(upd, dict):
            return
        b = _num(upd.get("budget"))
        old = _num(self.brief.get("budget"))
        if (b is not None and b > 0 and (old is None or b < old) and upd.get("market_change") is not True
                and self._number_in(b, text)):
            self.brief["budget"] = b
        if upd.get("market_change") is True:
            self.ledger.last_event_turn = max(self.ledger.last_event_turn, turn)
            self.ledger.refresh_dead(_num(self.brief.get("budget")))

    # ------------------------------------------------------------------ environment
    def act(self, name: str, args: dict, content: str = "", call_id: str | None = None) -> Obs | None:
        got = self.act_group([(name, args, call_id)], content)
        return got[0] if got else None

    def act_group(self, actions: list[tuple[str, dict, str | None]], content: str = "") -> list[Obs]:
        obs_list = self.session.group(actions, content)
        if not obs_list:
            return []
        tool_calls = []
        for i, (name, args, cid) in enumerate(actions, start=1):
            tool_calls.append({"id": cid or f"{self.session.pid}-turn-{obs_list[0].turn}-{i}", "type": "function",
                               "function": {"name": name, "arguments": json.dumps(args)}})
        self.messages.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls})
        hidden = []
        for tc, obs in zip(tool_calls, obs_list):
            self.ledger.ingest(obs)
            payload = dict(obs.data) if isinstance(obs.data, dict) else {}
            if obs.error and "error" not in payload:
                payload["error"] = obs.error
            limit = 4500 if obs.name in ("view", "compare") else 2500
            payload, dropped = _shrink(payload, max(limit, _OBS_CHARS // max(1, len(obs_list))))
            if dropped:
                hidden.append("%d rows hidden in %s (read them individually)" % (dropped, obs.name))
            self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": _trim(json.dumps(payload), 12000)})
        last = obs_list[-1]
        L = self.ledger
        budget = _num(self.brief.get("budget"))
        L.refresh_dead(budget)
        now = time.monotonic()
        self.turn_secs.append(now - self._turn_started)
        self._turn_started = now
        recent = self.turn_secs[-5:]
        spt = max(6.0, sum(recent) / max(1, len(recent)))
        by_time = int(self.clock.remaining() / spt)
        left = min(self.session.steps_left, by_time)
        self.commit_by = min(max(4, self.session.max_steps - 10), last.turn + max(0, by_time - 7))
        status = "turn %d of %d used; about %d turns of time left (%.0f s)" % (last.turn, self.session.max_steps, left, max(0.0, self.clock.remaining()))
        if not L.adds:
            status += "; make your first add_to_cart by turn %d at the latest: if the store changes that item you need about 7 turns to add a replacement, verify it and order" % self.commit_by
        else:
            status += "; finishing needs inspect_stock (in its own turn), add_to_cart if not carted, and place_test_order"
            live = any((L.fresh_reading(p, s_) or {}).get("in_stock") is True for p, s_ in L.cart)
            if L.dead and not live:
                status += "; your carted item is gone, so add a different product that meets everything and order it by turn %d; if nothing better turns up, order the best compliant in-stock item you have already verified" % max(last.turn + 1, self.session.max_steps - 4)
        if L.cart:
            status += "; cart: " + ", ".join("%s::%s" % c for c in L.cart)
        for obs in obs_list:
            if obs.name == "add_to_cart" and obs.ok:
                added = (obs.data or {}).get("added") or {}
                ev = (obs.data or {}).get("market_event") if isinstance(obs.data, dict) else None
                key = _key(added.get("product_id"), added.get("sku")) if added.get("product_id") is not None else None
                announced = isinstance(ev, dict) and key is not None and key in L.dead
                if announced:
                    twins = [k for k in L.dead if k != key and k in {t for e in L.events for t in e["targets"]}]
                    status += "; the store announced %s on the item you just added, so it is gone: do not order it; add a different product that meets everything (one you have already read), then inspect_stock that replacement in its own turn" % ev.get("kind")
                    if twins:
                        status += "; the same change also hit %s, so do not add those either" % ", ".join(twins)
                elif key is not None:
                    status += "; you just added %s: inspect_stock it next, before any other action, because the add result can be stale" % key
        if hidden:
            status += "; " + ", ".join(hidden)
        if left <= 6 and L.cart and not any(L.fresh_reading(p, s) for p, s in L.cart):
            status += "; few turns left: verify and order now"
        if left <= 6 and not L.cart:
            status += "; few turns left: add the best candidate you have fully read now"
        if last.user:
            self.messages.append({"role": "user", "content": last.user})
            self.update_brief(last.user, last.turn)
            if L.unanswered_question():
                status += "; the shopper asked you something: put one short message call first in your next reply (nothing you write outside the message tool reaches them), followed by your next action"
        elif last.name == "message":
            status += "; the shopper did not reply: they do not approve choices, so continue"
        self.messages.append({"role": "user", "content": "[status] " + status})
        return obs_list

    # ------------------------------------------------------------------ invariants
    def check(self, name: str, args: dict, answering: bool = False) -> str | None:
        """Reject a proposed action when the environment's own records show it is unsafe or wasteful."""
        L = self.ledger
        if name not in self.session.tool_names:
            return "that tool does not exist here; use one of " + ", ".join(sorted(n for n in self.session.tool_names if n))
        if name == "message" and not str(args.get("content") or "").strip():
            return "message needs content"
        cands = [c for c in (self.brief.get("candidate_ids") or []) if c not in L.candidate_missing]
        confirmed = any((L.products.get(c) or {}).get("read") for c in cands)
        if cands and confirmed:
            if name in ("view", "add_to_cart", "inspect_stock") and str(args.get("product_id") or "") not in cands:
                return "product %s is not one of the candidates the shopper listed (%s); stay with those" % (args.get("product_id"), ", ".join(cands))
            if name == "compare" and any(str(x) not in cands for x in (args.get("product_ids") or [])):
                return "compare only the candidates the shopper listed (%s)" % ", ".join(cands)
            if name == "add_to_cart":
                listed = {pp: sk for pp, sk in (self.brief.get("candidate_pairs") or []) if sk}
                want_sku = listed.get(str(args.get("product_id") or ""))
                if want_sku and str(args.get("sku") or "") != want_sku:
                    return "the shopper listed variant %s::%s, not sku %s" % (args.get("product_id"), want_sku, args.get("sku"))
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs product_id and sku"
            if not L.adds and not str(getattr(self, "_pending_content", "") or "").strip() and "shortlist:" not in self._warned:
                self._warned.add("shortlist:")
                return "before your first add_to_cart, write out in the reply text your shortlist with each candidate's decisive values as you read them and the rule you applied, then call add_to_cart in the same reply"
            if self.brief.get("ask_first") and not L.adds and L.last_question_turn == 0:
                return "the shopper said a requirement or priority will be given on request (or the rules ask you to clarify first): send one message asking directly for their firm requirements, specs and priorities, as a question ending in '?', before your first add_to_cart"
            p = L.products.get(pid)
            if not p or not p.get("read"):
                return "read product %s with view or compare before adding it, so its facts and variant options are on record" % pid
            fb = self.brief.get("firm_brand")
            if fb and ("brand:" + pid) not in self._warned:
                have = set(re.findall(r"[a-z0-9]+", str(p.get("brand") or "").lower()))
                want = set(re.findall(r"[a-z0-9]+", str(fb).lower()))
                if p.get("brand") is None or not (want <= have):
                    self._warned.add("brand:" + pid)
                    return "check before adding: the store lists product %s under brand '%s' while the shopper requires '%s'; a title mentioning the brand is not the store's brand. Add it again only if you are sure it qualifies, otherwise pick a listing whose brand field matches" % (pid, p.get("brand"), fb)
            words = self.brief.get("required_title_words") or []
            if words and pid not in self._warned:
                toks = set(re.findall(r"[a-z0-9]+", str(p.get("title") or "").lower()))
                missing = [w for w in words if w not in toks]
                if missing:
                    self._warned.add(pid)
                    return "check before adding: the title of product %s does not contain the whole word(s) %s that you listed as required by the rules; add it again only if you are sure the rules allow it" % (pid, ", ".join(missing))
            dead_pids = {k.split("::")[0] for k in L.dead}
            for dp in dead_pids:
                tp = L.products.get(dp) or {}
                if dp != pid and tp.get("listing_id") and p.get("listing_id") == tp.get("listing_id"):
                    return "product %s is another listing of the same source as the changed item %s; choose a product with a different listing" % (pid, dp)
            s = L.stock.get(_key(pid, sku))
            if s and s.get("in_stock") is False:
                return "variant %s was last seen out of stock; pick an in-stock variant or another product" % _key(pid, sku)
            budget = _num(self.brief.get("budget"))
            if s and budget is not None and s.get("price") is not None and s["price"] > budget + 1e-6:
                return "variant %s was last seen at %s, above the budget %s you extracted; choose one within budget" % (_key(pid, sku), s["price"], budget)
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and L.cart:
                pid, sku = L.cart[-1]
            if (pid, sku) not in L.cart:
                return "the order target must be in the cart: add_to_cart %s first" % (_key(pid, sku) if pid else "the chosen variant")
            q = L.unanswered_question()
            if q and not answering:
                return "the shopper asked something you have not answered yet (%s). Put one short message call first in the same reply, then the order." % _trim(q, 160)
            fresh = L.fresh_reading(pid, sku)
            if fresh is None:
                return "verify the current price and stock of %s with inspect_stock (after the latest cart action or market change) before ordering" % _key(pid, sku)
            if fresh.get("in_stock") is not True:
                return "%s is out of stock now; choose a different product" % _key(pid, sku)
            budget = _num(self.brief.get("budget"))
            if budget is not None and fresh.get("price") is not None and fresh["price"] > budget + 1e-6:
                return "%s now costs %s, above the budget %s; choose a different product" % (_key(pid, sku), fresh["price"], budget)
            if self.justification_required():
                just = args.get("justification")
                claims = just.get("claims") if isinstance(just, dict) else None
                allowed, desc = self._claim_field_spec()
                if not isinstance(claims, list) or not claims:
                    return "this order must carry justification claims: field/value pairs only. Field guidance from the tool: %s" % (desc or "see the tool schema")
                bad = [str(c.get("field")) for c in claims if isinstance(c, dict) and allowed and str(c.get("field") or "").lower() not in allowed]
                if bad:
                    return "claim field(s) %s are not accepted. Allowed field names: %s. Guidance: %s" % (", ".join(bad), ", ".join(sorted(allowed)), desc)
                names = set(re.findall(r"[a-z][a-z_]+", desc.lower())) | allowed
                labels = {str(k).lower() for k in ((L.products.get(pid) or {}).get("facts") or {}).keys()} - names
                lab = [str(c.get("field")) for c in claims if isinstance(c, dict) and str(c.get("field") or "").lower() in labels]
                if lab:
                    return ("claim field(s) %s are the store's fact labels, not claim field names. The order tool's standard field names for a "
                            "specification are screen_size, resolution, refresh_rate and ram; pick the one the shopper's requirement is about and copy "
                            "the value as the facts show it. Guidance: %s" % (", ".join(lab), desc or ", ".join(sorted(allowed)) or "see the tool schema"))
                kept = self.sanitize_order(args).get("justification", {}).get("claims", [])
                state = {"price", "in_stock", "brand", "category"}
                if not kept:
                    return "no claim survived grounding; each value must be copied from the store's facts, variant options or the price/stock reading of %s. Guidance: %s" % (_key(pid, sku), desc)
                if not [k for k in kept if k["field"] not in state]:
                    return ("the claims must include the specification the shopper asked about, under the order tool's standard field name for it "
                            "(screen_size, resolution, refresh_rate or ram), with the value copied from the item's facts or variant option. Guidance: %s" % desc)
        if name in ("search", "filter", "view", "compare", "inspect_stock", "inspect_cart"):
            sig = name + json.dumps(args, sort_keys=True)
            if sig in L.calls_seen:
                return "you already made exactly this call; use the result you have or change the arguments"
        return None

    def sanitize_order(self, args: dict) -> dict:
        """Keep the order well-formed: known target, claims as field/value pairs with observed price and stock."""
        L = self.ledger
        pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
        if not pid and L.cart:
            pid, sku = L.cart[-1]
        out: dict = {"product_id": pid, "sku": sku}
        just = args.get("justification")
        if self.justification_required():
            raw = just.get("claims") if isinstance(just, dict) and isinstance(just.get("claims"), list) else []
            if raw:
                L.last_claims = [c for c in raw if isinstance(c, dict)]
            allowed = self._claim_fields()
            reading = L.stock.get(_key(pid, sku)) or {}
            product = L.products.get(pid) or {}
            claims = []
            for c in raw or L.last_claims:
                if not isinstance(c, dict):
                    continue
                field = str(c.get("field") or "").strip().lower()
                value = c.get("value")
                if not field or value is None or (allowed and field not in allowed):
                    continue
                value = str(value).strip()
                if field == "price":
                    if reading.get("price") is None:
                        continue
                    value = str(reading["price"])          # the price string the store itself reported
                elif field == "in_stock":
                    if reading.get("in_stock") is not True:
                        continue
                    value = "true"
                elif field == "brand":
                    if not product.get("brand"):
                        continue
                    value = str(product["brand"])
                elif not self._grounded(pid, field, value):
                    continue
                if value and not any(k["field"] == field for k in claims):
                    claims.append({"field": field, "value": value})
            state = ("price", "in_stock", "brand")
            others = [k for k in claims if k["field"] not in state]
            if len(others) > 1 or (len(others) > 0 and any(k["field"] == "brand" for k in claims) and len(others) > 1):
                keep_spec = others[0]
                reason = None if any(k["field"] == "brand" for k in claims) else (others[1] if len(others) > 1 else None)
                claims = [k for k in claims if k["field"] in state] + [keep_spec] + ([reason] if reason else [])
            if claims and reading.get("price") is not None and not any(k["field"] == "price" for k in claims):
                claims.append({"field": "price", "value": str(reading["price"])})
            if claims and reading.get("in_stock") is True and not any(k["field"] == "in_stock" for k in claims):
                claims.append({"field": "in_stock", "value": "true"})
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def _order_schema(self) -> dict:
        for t in self.session.tools:
            fn = t.get("function") or {}
            if fn.get("name") == "place_test_order":
                return fn.get("parameters") or {}
        return {}

    def justification_required(self) -> bool:
        """The environment's own tool schema says whether an order must carry claims."""
        return "justification" in (self._order_schema().get("required") or [])

    def _claim_field_spec(self) -> tuple[set[str], str]:
        """Allowed claim field names: the schema's enum, else the names listed in its description."""
        try:
            field = self._order_schema()["properties"]["justification"]["properties"]["claims"]["items"]["properties"]["field"]
        except Exception:
            return set(), ""
        enum = field.get("enum")
        desc = str(field.get("description") or "")
        return (set(enum) if enum else set()), desc

    def _claim_fields(self) -> set[str]:
        return self._claim_field_spec()[0]

    def _grounded(self, pid: str, field: str, value: str) -> bool:
        """A claim is kept only if the store's own text about the product states it."""
        p = self.ledger.products.get(pid) or {}
        texts = " ".join([str(p.get("title") or "")] + [str(v) for v in (p.get("facts") or {}).values()]
                         + [str((v or {}).get("options") or "") for v in (p.get("variants") or {}).values()]).lower()
        flat = re.sub(r"[^a-z0-9]", "", texts)
        if value.lower() in ("true", "yes", "1"):
            parts = [re.escape(w) for w in field.lower().split("_") if w]
            return re.search(r"(?<![a-z0-9])" + r"[-\s]?".join(parts) + r"(?![a-z0-9])", texts) is not None
        nums = re.findall(r"\d+(?:\.\d+)?", value.lower())
        fnums = set(re.findall(r"\d+(?:\.\d+)?", texts))
        if nums and not all(n in fnums for n in nums):
            return False
        return re.sub(r"[^a-z0-9]", "", value.lower()) in flat or bool(nums)

    # ------------------------------------------------------------------ helpers for the model
    def _prerequisite(self, name: str, args: dict, reason: str) -> tuple[str, dict] | None:
        """When the model keeps proposing the same blocked step, do the obvious prerequisite for it."""
        L = self.ledger
        if name == "add_to_cart" and "before adding" in reason:
            return "view", {"product_id": str(args.get("product_id"))}
        if name == "add_to_cart" and "given on request" in reason and self.clock.remaining() >= 60:
            msg = self._chat(self.messages + [{"role": "user", "content": "Ask the shopper, in one question ending with '?', for their firm requirements, specs and priorities. Output the question only."}], None, 120)
            text = ((msg or {}).get("content") or "").strip()
            if text:
                return "message", {"content": text[:400]}
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and L.cart:
                pid, sku = L.cart[-1]
            if "must be in the cart" in reason and pid and sku:
                return "add_to_cart", {"product_id": pid, "sku": sku}
            if "verify the current price" in reason and pid and sku:
                return "inspect_stock", {"product_id": pid, "sku": sku}
        return None

    def _ask(self) -> dict | None:
        msg = self._chat(self.messages, self.session.tools)
        if not msg:
            return None
        calls = msg.get("tool_calls") or []
        if not calls:
            self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
            self.messages.append({"role": "user", "content": "Choose exactly one tool now."})
            msg = self._chat(self.messages, self.session.tools)
            calls = (msg or {}).get("tool_calls") or []
            if not calls:
                return None
        props = []
        for j, call in enumerate(calls[:4]):
            if not call.get("id"):
                call["id"] = "call%d" % j
            fn = call.get("function") or {}
            raw = fn.get("arguments", "{}")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}
            props.append({"id": call.get("id") or "call%d" % j, "name": fn.get("name") or "", "args": args, "content": msg.get("content") or "", "raw": call})
        first = props[0]
        if first["name"] == "message" and len(props) > 1:
            first["extra"] = props[1:2]
        else:
            first["extra"] = [p for p in props[1:] if p["name"] in _READS and first["name"] in _READS]
        return first

    # ------------------------------------------------------------------ safety net
    def finish(self) -> None:
        """Turns or time are nearly gone: order the carted item the records show as valid, else stop."""
        L = self.ledger
        budget = _num(self.brief.get("budget"))
        if not L.cart and self.session.steps_left >= 6 and self.clock.remaining() >= 60:
            self.messages.append({"role": "user", "content": "Turns are almost gone. Call add_to_cart now for the best variant you have read and verified."})
            prop = self._ask()
            if prop and prop["name"] == "add_to_cart" and self.check("add_to_cart", prop["args"]) is None:
                self.act("add_to_cart", prop["args"], prop.get("content") or "", prop["id"])
        reply = None
        if L.unanswered_question() and self.clock.remaining() >= 60:
            msg = self._chat(self.messages + [{"role": "user", "content": "Answer the shopper's last message in one sentence using only what the store showed you. Output the sentence only."}], None, 120)
            text = ((msg or {}).get("content") or "").strip()
            if text:
                reply = ("message", {"content": text[:600]}, None)
        for pid, sku in reversed(L.cart):
            if self.session.done or self.session.steps_left < 1:
                return
            fresh = L.fresh_reading(pid, sku)
            if fresh is None and self.session.steps_left >= 2:
                if reply:
                    self.act_group([reply, ("inspect_stock", {"product_id": pid, "sku": sku}, None)])
                    reply = None
                else:
                    self.act("inspect_stock", {"product_id": pid, "sku": sku})
                fresh = L.fresh_reading(pid, sku)
            if not fresh or fresh.get("in_stock") is not True:
                self.notes.append("finish skip %s: no fresh in-stock reading" % _key(pid, sku))
                continue
            if budget is not None and fresh.get("price") is not None and fresh["price"] > budget + 1e-6:
                self.notes.append("finish skip %s: price %s over budget %s" % (_key(pid, sku), fresh.get("price"), budget))
                continue
            args: dict = {"product_id": pid, "sku": sku}
            if self.justification_required():
                args = self.sanitize_order({"product_id": pid, "sku": sku, "justification": {"claims": L.last_claims}})
                spec = [k for k in args.get("justification", {}).get("claims", []) if k["field"] not in ("price", "in_stock", "brand")]
                if not spec and self.clock.remaining() >= 90:
                    self.messages.append({"role": "user", "content": "Time is up. Call place_test_order for %s now with the justification claims the tool requires." % _key(pid, sku)})
                    prop = self._ask()
                    if prop and prop["name"] == "place_test_order":
                        args = self.sanitize_order({**prop["args"], "product_id": pid, "sku": sku})
            if reply:
                self.act_group([reply, ("place_test_order", args, None)])
            else:
                self.act("place_test_order", args)
            return

    # ------------------------------------------------------------------ main loop
    def run(self) -> list[dict]:
        self.extract_brief()
        blocked_sig, blocked_n = None, 0
        while not self.session.done and not self.session.broken and self.session.steps_left > 0:
            if self.clock.remaining() < 60 or self.session.steps_left <= 2:
                self.notes.append("finish: remaining=%.0f steps_left=%d" % (self.clock.remaining(), self.session.steps_left))
                self.finish()
                break
            if self.session.steps_left <= 5 and not self.ledger.cart and not self._late_add_tried:
                self._late_add_tried = True
                self.notes.append("late add attempt at turn %d" % self.session.turn)
                self.finish()
                if self.session.done:
                    break
            if not self.ledger.adds and self.session.turn >= self.commit_by and not self._pushed:
                self._pushed = True
                self.messages.append({"role": "user", "content": "The commit deadline is here. If the rules require a clarifying question before the first cart action and you have not sent one, send it now as one question ending in '?'; otherwise choose the best variant you have already read that meets every requirement and call add_to_cart for it now; if the store then changes it you still have turns to recover. Do not start new searches first."})
            prop = self._ask()
            if prop is None:
                self.notes.append("model gave no action at turn %d (remaining %.0f)" % (self.session.turn, self.clock.remaining()))
                self.finish()
                break
            name, args, content = prop["name"], prop["args"], prop["content"]
            self._pending_content = content
            lead = name == "message" and bool(prop.get("extra"))
            if lead:
                follow = prop["extra"][0]
                reason = self.check("message", args) or self.check(follow["name"], follow["args"], answering=True)
            else:
                reason = self.check(name, args)
            reprompts = 0
            limit = 1 if self.clock.remaining() < 90 else _MAX_REPROMPTS
            while reason and reprompts < limit:
                reprompts += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [prop["raw"]]})
                self.messages.append({"role": "tool", "tool_call_id": prop["id"], "content": json.dumps({"rejected": reason})})
                again = self._ask()
                if again is None:
                    break
                prop = again
                name, args, content = prop["name"], prop["args"], prop["content"]
                self._pending_content = content
                lead = name == "message" and bool(prop.get("extra"))
                if lead:
                    follow = prop["extra"][0]
                    reason = self.check("message", args) or self.check(follow["name"], follow["args"], answering=True)
                else:
                    reason = self.check(name, args)
            if reason:
                sig = name + json.dumps(args, sort_keys=True)
                blocked_n = blocked_n + 1 if sig == blocked_sig else 1
                blocked_sig = sig
                pre = self._prerequisite(name, args, reason)
                if pre and self.session.steps_left > 2:
                    why = self.check(pre[0], pre[1])
                    if why and pre[0] == "add_to_cart" and "before adding" in why:
                        pre, why = ("view", {"product_id": pre[1]["product_id"]}), None
                    if why is None:
                        self.act(pre[0], pre[1], "doing the prerequisite step first")
                        continue
                    reason = why
                self.messages.append({"role": "user", "content": "That action was not taken: %s. Choose a different action." % reason})
                if blocked_n >= 3:
                    self.notes.append("blocked 3x on %s at turn %d" % (name, self.session.turn))
                    self.finish()
                    break
                continue
            blocked_sig, blocked_n = None, 0
            if name == "place_test_order":
                args = self.sanitize_order(args)
            if lead:
                f = prop["extra"][0]
                fargs = self.sanitize_order(f["args"]) if f["name"] == "place_test_order" else f["args"]
                self.act_group([(name, args, prop["id"]), (f["name"], fargs, f["id"])], content)
                continue
            extra = [(p["name"], p["args"], p["id"]) for p in prop.get("extra") or [] if not self.check(p["name"], p["args"])]
            if extra and name in _READS:
                self.act_group([(name, args, prop["id"])] + extra, content)
            else:
                self.act(name, args, content, prop["id"])
        self.notes.append("loop end: done=%s broken=%s steps_left=%d remaining=%.0f cart=%d" % (self.session.done, self.session.broken, self.session.steps_left, self.clock.remaining(), len(self.ledger.cart)))
        dialogue = self.session.dialogue
        dialogue.append({"role": "assistant", "content": "harness notes: brief=%s; %s" % (json.dumps(self.brief)[:300], "; ".join(self.notes)[:1200])})
        return dialogue


def agent_main(problem_data: dict) -> list[dict]:
    agent = None
    try:
        agent = Agent(problem_data)
        return agent.run()
    except Exception as exc:  # never raise out of the harness; still try to close the session sensibly
        note = "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]
        if agent is not None:
            try:
                agent.notes.append(note)
                if not agent.session.done:
                    agent.finish()
                agent.session.dialogue.append({"role": "assistant", "content": note})
                return agent.session.dialogue
            except Exception:
                pass
        return [{"role": "assistant", "content": note}]


__all__ = ["agent_main"]
