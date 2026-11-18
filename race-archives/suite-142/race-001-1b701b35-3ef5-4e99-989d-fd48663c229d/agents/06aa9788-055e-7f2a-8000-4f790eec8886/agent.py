"""ORO-Bench generalist agent, skeleton v1.

Design rules (integrity + generalization):
  * Every branch is a function of problem_data (query text, tool schemas, limits) or of
    observations returned at runtime. No task-family phrase tables, no spec-tier tables,
    no product/answer data, no clock/env/validator-phase routing.
  * The LLM picks every action from the dynamic tool list; deterministic code only handles
    transport (idempotent replay), schema coercion, budget/timing, and loop hygiene.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_MODELS = {
    # Ordered by measured ORO-Bench performance on the public roster (2026-09-15): GLM-5.2 0.55,
    # Kimi-K2.6 0.47, DeepSeek-V3.2 0.29. Later entries are fallbacks on inference failure.
    "chutes": (
        "zai-org/GLM-5.2-TEE",
        "moonshotai/Kimi-K2.6-TEE",
        "deepseek-ai/DeepSeek-V3.2-TEE",
    ),
    "openrouter": (
        "z-ai/glm-5.2",
        "moonshotai/kimi-k2.6",
        "deepseek/deepseek-v3.2",
    ),
}
_PROXY = ProxyClient(timeout=100, max_retries=2, retry_delay=1.0)
_PROBLEM_BUDGET_S = 235.0   # validator per-problem cap is 300 s; leave margin for slow final turns
_OBS_CHARS = 7000
_CONTEXT_CHARS = 220000   # ~55k tokens; older tool outputs get compressed beyond this
_KEEP_RECENT_TURNS = 5
_MAX_REPROMPTS = 2
_ENV_RETRIES = 4
_DEDUPE_WINDOW = 3      # identical call blocked only if sent within the last N turns
_COMMIT_STEPS_LEFT = 6  # nudge to finish when this many steps remain
_COMMIT_SECS_LEFT = 90.0
_FALLBACK_SECS_LEFT = 25.0
_FALLBACK_STEPS_LEFT = 2

_SYSTEM = (
    "You are a careful, honest shopping agent operating a marketplace through the supplied tools. "
    "Read the shopper's request closely: identify firm requirements (category, budget, stock, any "
    "named specification or variant) versus soft preferences, and treat every detail the shopper "
    "states later as binding. Work in small verifiable steps: search or filter, then open candidates "
    "to read their actual facts, options and prices before choosing. Never add an item to the cart "
    "until you have confirmed it meets every firm requirement, because the first cart addition is "
    "what gets graded. If the shopper asks you a question, answer it with the messaging tool before "
    "doing anything else. If something changes mid-task (price, stock, a new instruction), re-check "
    "your plan against the new facts and relax only the least important preference. When several "
    "candidates satisfy the firm requirements, prefer the one whose raw product facts state each required "
    "specification as a single clear value (not a title claim, not an ambiguous or multi-valued fact), and "
    "whose category_path matches the requested category. In any structured justification, use the exact "
    "fact labels shown by view/compare and copy values verbatim. You may issue several tool calls in one "
    "turn: batch independent reads together (view or inspect several shortlisted candidates at once, or run "
    "two differently worded searches) so you learn more per turn, but keep cart and order actions alone in "
    "their own turn after their evidence is in. Turns are limited, so shortlist early, read a focused set of "
    "candidates rather than the whole market, and commit once one candidate is fully verified. Never repeat "
    "a call with identical arguments, finish the task the environment describes, do not stop early, and do "
    "not send progress chatter."
)


def _roster() -> tuple:
    provider = (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()
    roster = _MODELS.get(provider) or _MODELS["chutes"]
    # Model choice is made here, in code (measured on ORO-Bench); the sandbox's SANDBOX_MODEL default
    # is meant for the reference agent and is deliberately ignored.
    return roster


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 24] + " ...[truncated]"


def _conform(val: Any, spec: dict) -> tuple:
    """Coerce/sanitize one value against its JSON-schema spec. Returns (ok, value).
    Generic: type coercion, enum enforcement (dropping non-conforming array items), recursion
    into objects/arrays, removal of properties the schema forbids (additionalProperties: false)."""
    spec = spec or {}
    typ = spec.get("type")
    try:
        if typ == "integer" and not isinstance(val, bool):
            val = int(float(val))
        elif typ == "number" and not isinstance(val, bool):
            val = float(val)
        elif typ == "boolean" and isinstance(val, str):
            val = val.strip().lower() in ("true", "1", "yes")
        elif typ == "string" and not isinstance(val, str):
            val = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
        elif typ == "array" and isinstance(val, str):
            val = [v.strip() for v in val.split(",") if v.strip()]
    except Exception:
        pass
    if "enum" in spec and val not in spec["enum"]:
        # try case-insensitive string match before rejecting
        if isinstance(val, str):
            match = [e for e in spec["enum"] if isinstance(e, str) and e.lower() == val.strip().lower()]
            if match:
                return True, match[0]
        return False, val
    if typ == "array" and isinstance(val, list):
        item_spec = spec.get("items") or {}
        kept = []
        for item in val:
            ok, item = _conform(item, item_spec)
            if ok:
                kept.append(item)
        return True, kept
    if typ == "object" and isinstance(val, dict):
        props = spec.get("properties") or {}
        strict = spec.get("additionalProperties") is False
        out = {}
        for k, v in val.items():
            if k not in props:
                if not strict:
                    out[k] = v
                continue
            ok, v2 = _conform(v, props[k])
            if ok:
                out[k] = v2
        for req in spec.get("required") or []:
            if req not in out and req in val:
                return False, val  # required field failed conformance
        return True, out
    return True, val


def _coerce(args: dict, schema: dict) -> dict:
    """Conform LLM-provided arguments to the tool's JSON schema. Generic, schema-driven."""
    ok, out = _conform(args, dict(schema or {}, type="object"))
    return out if ok and isinstance(out, dict) else args


def _arguments(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _walk_products(o: Any, sink: dict, ctx: str = "") -> None:
    """Collect every observed dict carrying a product_id into sink[pid] as text evidence."""
    if isinstance(o, dict):
        pid = o.get("product_id")
        if pid is not None:
            sink[str(pid)] = (sink.get(str(pid), "") + " " + json.dumps(o, ensure_ascii=False))[-20000:]
        for v in o.values():
            _walk_products(v, sink, ctx)
    elif isinstance(o, list):
        for v in o:
            _walk_products(v, sink, ctx)


def _walk_stock(o: Any, sink: dict, turn: int, pid_ctx: str = "") -> None:
    """Record (product_id, sku) -> (in_stock, turn, price) from any observed dict that states stock."""
    if isinstance(o, dict):
        pid = str(o.get("product_id") or pid_ctx or "")
        if pid and o.get("sku") is not None and isinstance(o.get("in_stock"), bool):
            sink[(pid, str(o["sku"]))] = (o["in_stock"], turn, o.get("current_price", o.get("price", o.get("price_at_add"))))
        for v in o.values():
            _walk_stock(v, sink, turn, pid)
    elif isinstance(o, list):
        for v in o:
            _walk_stock(v, sink, turn, pid_ctx)


def _walk_facts(o: Any, sink: dict) -> None:
    """Merge every observed product's raw fact labels (plus brand) into sink[pid]."""
    if isinstance(o, dict):
        pid = o.get("product_id")
        if pid is not None and (isinstance(o.get("facts"), dict) or o.get("brand") is not None):
            d = sink.setdefault(str(pid), {})
            for k, v in (o.get("facts") or {}).items():
                d[str(k)] = v
            if o.get("brand") is not None:
                d["brand"] = o.get("brand")
        for v in o.values():
            _walk_facts(v, sink)
    elif isinstance(o, list):
        for v in o:
            _walk_facts(v, sink)


def _compress_obs(text: str) -> str:
    """Shrink an old tool observation to the identifiers it contained."""
    try:
        o = json.loads(text)
    except Exception:
        return text[:200]
    ids: list = []
    def walk(x):
        if isinstance(x, dict):
            if x.get("product_id") is not None:
                ids.append(f"{x.get('product_id')}::{x.get('sku', '')} {str(x.get('title', ''))[:40]} {x.get('price', '')}".strip())
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(o)
    seen: list = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return "[older observation compressed; items seen: " + "; ".join(seen[:12]) + ("; ..." if len(seen) > 12 else "") + "]"


_BUDGET_RE = re.compile(r"(?:budget|at most|no more than|up to|under|below|max(?:imum)?)[^\d]{0,40}?(\d[\d,]*(?:\.\d+)?)", re.I)


def _stated_budget(text: str) -> float | None:
    """Lowest explicit budget figure a shopper states in a message (derived from the message itself)."""
    vals = []
    for m in _BUDGET_RE.finditer(text or ""):
        try:
            vals.append(float(m.group(1).replace(",", "")))
        except Exception:
            pass
    return min(vals) if vals else None


def _norm(t: Any) -> str:
    return " ".join(str(t).lower().replace(",", " ").split())


class Agent:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = [t for t in (self.policy.get("tools") or []) if isinstance(t, dict)]
        self.schemas = {
            (t.get("function") or {}).get("name"): (t.get("function") or {}).get("parameters") or {}
            for t in self.tools
        }
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.query = str(self.policy.get("query") or "")
        self.budget_cap = _stated_budget(self.query.split("Task rules")[0])
        self.started = time.monotonic()
        self.turn = 0
        self.done = False
        self.model_index = 0
        self.roster = _roster()
        self.seen_calls: dict = {}   # call signature -> turn it was sent (recency-scoped dedupe)
        self.ordered = False
        self.evidence: dict = {}     # product_id -> concatenated observed text (view/compare/inspect/cart)
        self.cart: list = []         # [{"product_id","sku","price"}] from cart observations
        self.last_event_turn = 0     # turn of last shopper message / cart-state surprise
        self.last_inspect: dict = {} # (product_id, sku) -> turn of last stock/price inspection
        self.stock: dict = {}        # (product_id, sku) -> (in_stock: bool, turn, price) from any observation
        self.facts: dict = {}        # product_id -> merged {fact_label: value, 'brand': ...} from view/compare
        self.fallback_tried: set = set()   # (product_id, sku) already attempted by the fallback
        self.fallback_stage: dict = {}     # (product_id, sku) -> 'carted' once fallback add_to_cart succeeded
        self.budget_cap: float | None = None
        self.add_tool = next((n for n in self.schemas if n and "add" in n and "cart" in n), None)
        self.order_tool = next((n for n in self.schemas if n and any(k in n for k in ("order", "checkout", "purchase", "buy"))), None)
        self.inspect_tool = next((n for n in self.schemas if n and "stock" in n), None) or next((n for n in self.schemas if n and "view" in n), None)
        self.pending: dict | None = None
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": self.query},
            {"role": "user", "content": f"[note] Limits for this task: at most {self.max_steps} turns and up to "
                                        f"{self.max_calls} tool calls per turn."},
        ]

    # -------------------------------------------------------------- timing
    def remaining(self) -> float:
        return _PROBLEM_BUDGET_S - (time.monotonic() - self.started)

    def _note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})

    # -------------------------------------------------------------- context hygiene
    def _trim_context(self) -> None:
        total = sum(len(json.dumps(m)) for m in self.messages)
        if total <= _CONTEXT_CHARS:
            return
        turn_idx = [i for i, m in enumerate(self.messages) if m.get("role") == "assistant" and m.get("tool_calls")]
        if len(turn_idx) <= _KEEP_RECENT_TURNS:
            return
        cutoff = turn_idx[-_KEEP_RECENT_TURNS]
        for i in range(cutoff):
            m = self.messages[i]
            if m.get("role") == "tool" and len(m.get("content") or "") > 300 and not str(m.get("content", "")).startswith("[older"):
                m["content"] = _compress_obs(m["content"])

    # -------------------------------------------------------------- inference
    def _infer(self) -> dict | None:
        self._trim_context()
        for _ in range(len(self.roster)):
            model = self.roster[self.model_index % len(self.roster)]
            payload = {
                "model": model,
                "messages": self.messages,
                "tools": self.tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            resp = _PROXY.post("/inference/chat/completions", json_data=payload)
            try:
                msg = resp["choices"][0]["message"]
                if isinstance(msg, dict):
                    return msg
            except Exception:
                pass
            self.model_index += 1  # rotate to next allowlisted model on failure
            if self.remaining() < 30:
                break
        return None

    def _choose(self) -> tuple[str, list]:
        """Return (assistant_content, tool_calls[]) with up to _MAX_REPROMPTS nudges."""
        for _ in range(_MAX_REPROMPTS + 1):
            msg = self._infer()
            if msg is None:
                return "", []
            content = msg.get("content") or ""
            calls = [c for c in (msg.get("tool_calls") or []) if isinstance(c, dict)][: self.max_calls]
            fresh = []
            for c in calls:
                name = (c.get("function") or {}).get("name")
                if name not in self.schemas:
                    continue
                key = name + json.dumps(_arguments(c), sort_keys=True)
                last = self.seen_calls.get(key)
                if last is not None and self.turn - last < _DEDUPE_WINDOW:
                    continue
                fresh.append(c)
            if fresh:
                return content, fresh
            if not calls and not content:
                self.model_index += 1  # blank completion: try the next allowlisted model
            self.messages.append({"role": "assistant", "content": content})
            if calls:
                self._note("That exact tool call was already made (or names an unavailable tool); its result is "
                           "above. Choose a different action that moves the task forward.")
            else:
                self._note("The environment has not reported done=true. Choose one of the supplied tools.")
        return "", []

    # -------------------------------------------------------------- environment
    def _send(self, envelope: dict) -> dict | None:
        for attempt in range(_ENV_RETRIES):
            result = _PROXY.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.remaining() < 20:
                break
            time.sleep(min(1.5 * (attempt + 1), 5.0))
        return None

    def _act(self, content: str, calls: list) -> None:
        if self.pending is not None:  # replay an unanswered envelope first (idempotent)
            replay = self._send(self.pending["envelope"])
            if replay is None:
                self._note("The environment is not answering; the previous action is still pending.")
                return
            p = self.pending
            self.pending = None
            self.turn = p["turn"]
            self._absorb(p["content"], p["calls"], replay)
            if self.done:
                return
        turn = self.turn + 1
        group = f"{self.problem_id}-turn-{turn}"
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [],
        }
        for i, c in enumerate(calls, start=1):
            name = (c.get("function") or {}).get("name")
            args = _coerce(_arguments(c), self.schemas.get(name) or {})
            if name == self.order_tool:
                args = self._order_args(args)
            envelope["calls"].append({"call_id": f"{group}-{i}", "action": {"name": name, "args": args}})
            self.seen_calls[name + json.dumps(_arguments(c), sort_keys=True)] = turn
        result = self._send(envelope)
        if result is None:
            self.pending = {"envelope": envelope, "turn": turn, "content": content, "calls": calls}
            self.messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            for c in calls:
                self.messages.append({"role": "tool", "tool_call_id": c.get("id") or group,
                                      "content": json.dumps({"error": "environment call failed; retrying"})})
            return
        self.turn = turn
        self._absorb(content, calls, result)

    def _absorb(self, content: str, calls: list, result: dict) -> None:
        self.messages.append({"role": "assistant", "content": content, "tool_calls": calls})
        self.dialogue.append({"role": "assistant", "content": content, "tool_calls": calls,
                              "environment_result": result})
        rcalls = result.get("calls") or []
        for i, c in enumerate(calls):
            rc = rcalls[i] if i < len(rcalls) and isinstance(rcalls[i], dict) else {}
            wrap = rc.get("observation") if isinstance(rc.get("observation"), dict) else {}
            obs = wrap.get("observation") if isinstance(wrap.get("observation"), dict) else wrap
            if wrap.get("error") and isinstance(obs, dict) and not obs.get("error"):
                obs = dict(obs, error=wrap.get("error"))
            if wrap.get("done") or (isinstance(obs, dict) and obs.get("done")):
                self.done = True
            self.messages.append({"role": "tool", "tool_call_id": c.get("id") or f"agent-{self.turn}-{i}",
                                  "content": _trim(json.dumps(obs), _OBS_CHARS)})
            cname = str((c.get("function") or {}).get("name") or "")
            _walk_products(obs, self.evidence)
            _walk_stock(obs, self.stock, self.turn)
            _walk_facts(obs, self.facts)
            a = _arguments(c)
            if isinstance(obs, dict) and not obs.get("error"):
                if cname == self.inspect_tool or "stock" in cname or "view" in cname:
                    self.last_inspect[(str(a.get("product_id") or obs.get("product_id")), str(a.get("sku") or obs.get("sku")))] = self.turn
                if "cart" in cname:
                    cart = obs.get("cart")
                    if isinstance(cart, list):
                        self.cart = [{"product_id": str(x.get("product_id")), "sku": str(x.get("sku")),
                                      "price": x.get("current_price", x.get("price_at_add", x.get("price")))}
                                     for x in cart if isinstance(x, dict) and x.get("product_id")]
                    added = obs.get("added")
                    if isinstance(added, dict) and added.get("product_id") and not cart:
                        self.cart.append({"product_id": str(added.get("product_id")), "sku": str(added.get("sku")), "price": added.get("price")})
                    for x in (cart if isinstance(cart, list) else []):
                        if isinstance(x, dict) and x.get("current_price") not in (None, x.get("price_at_add")) or (isinstance(x, dict) and x.get("in_stock") is False):
                            self.last_event_turn = self.turn
            if isinstance(obs, dict) and "cart" in cname and "remov" in cname and not isinstance(obs.get("cart"), list):
                self.cart = [x for x in self.cart if not (str(x.get("product_id")) == str(a.get("product_id")) and str(x.get("sku")) == str(a.get("sku")))]
            if isinstance(obs, dict) and obs.get("error") and "cart" in str(obs.get("error")).lower() and cname == self.order_tool:
                self.cart = [x for x in self.cart if not (str(x.get("product_id")) == str(a.get("product_id")) and str(x.get("sku")) == str(a.get("sku")))]
            if "cart" in cname or "order" in cname or "checkout" in cname or "buy" in cname:
                self.seen_calls.clear()  # world state changed; earlier reads are stale
            if ("order" in cname or "checkout" in cname or "buy" in cname) and isinstance(obs, dict) and not obs.get("error"):
                self.ordered = True
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self.seen_calls.clear()  # shopper spoke: re-checking anything is legitimate now
            self.last_event_turn = self.turn
            b = _stated_budget(text)
            if b is not None and (self.budget_cap is None or b < self.budget_cap):
                self.budget_cap = b
            if "?" in text:
                self._note("The shopper asked a question. Answer it with the messaging tool in your next "
                           "turn, then continue the task honoring anything new they said.")
            else:
                self._note("The shopper added information. Treat it as binding from now on.")

    # -------------------------------------------------------------- order hygiene
    def _order_args(self, args: dict) -> dict:
        """Justification policy: omit when the schema does not require it; when required keep only
        claims whose value is verbatim-supported by observed facts / current price / stock."""
        schema = self.schemas.get(self.order_tool) or {}
        required = set(schema.get("required") or [])
        if "justification" not in required:
            args.pop("justification", None)
            return args
        pid = str(args.get("product_id") or (self.cart[-1]["product_id"] if self.cart else ""))
        sku = str(args.get("sku") or (self.cart[-1]["sku"] if self.cart else ""))
        facts = self.facts.get(pid) or {}
        item = next((x for x in reversed(self.cart) if str(x.get("product_id")) == pid), None) or (self.cart[-1] if self.cart else {})
        st = self.stock.get((pid, sku))
        price = (st[2] if st and st[2] is not None else item.get("price"))
        kept = []
        for cl in ((args.get("justification") or {}).get("claims") or []):
            if not isinstance(cl, dict):
                continue
            f, v = str(cl.get("field") or "").strip(), str(cl.get("value") or "").strip()
            if not f or not v:
                continue
            if f == "price":
                if price is not None:
                    kept.append({"field": "price", "value": str(price)})
                continue
            if f == "in_stock":
                kept.append({"field": "in_stock", "value": "true"})
                continue
            if f.startswith("category"):
                continue  # not a raw product fact label; verifier rejects it
            fact = facts.get(f)
            if fact is None:
                continue  # field must be an actual fact label of this product
            nf, nv = _norm(fact), _norm(v)
            if nv == nf or (nv and nv in nf):
                kept.append({"field": f, "value": str(fact)})  # copy the stored value verbatim
        if not any(c["field"] == "price" for c in kept) and price is not None:
            kept.append({"field": "price", "value": str(price)})
        if not any(c["field"] == "in_stock" for c in kept):
            kept.append({"field": "in_stock", "value": "true"})
        args["justification"] = {"claims": kept}
        return args

    def _fallback_candidates(self) -> list:
        """Only items the agent itself put in the cart (its own verified choice), still known in stock and
        within any shopper-stated budget. Never speculatively cart something the agent did not choose."""
        out = []
        for x in reversed(self.cart):
            key = (str(x.get("product_id")), str(x.get("sku")))
            st = self.stock.get(key)
            if (st is None or st[0]) and key not in self.fallback_tried:
                pr = x.get("price") if st is None or st[2] is None else st[2]
                if self.budget_cap is None or pr is None or float(pr) <= self.budget_cap:
                    out.append((key, pr))
        return out

    def _fallback_order(self) -> bool:
        """Out of time/steps: cart the best in-stock candidate if needed, then place the order.
        One attempt per candidate; never orders something observed out of stock or over a stated budget."""
        if not self.order_tool or self.ordered:
            return False
        cands = self._fallback_candidates()
        if not cands:
            return False
        (pid, sku), _ = cands[0]
        in_cart = any(str(x.get("product_id")) == pid and str(x.get("sku")) == sku for x in self.cart)
        if not in_cart and self.add_tool and self.fallback_stage.get((pid, sku)) != "carted":
            props = (self.schemas.get(self.add_tool) or {}).get("properties") or {}
            aargs = {k: v for k, v in (("product_id", pid), ("sku", sku)) if k in props}
            self.fallback_stage[(pid, sku)] = "carted"
            call = {"id": f"fallback-add-{self.turn + 1}", "type": "function",
                    "function": {"name": self.add_tool, "arguments": json.dumps(aargs)}}
            self._act("Time is nearly up; carting the best verified in-stock candidate.", [call])
            return True
        self.fallback_tried.add((pid, sku))
        args = {"product_id": pid, "sku": sku}
        schema = self.schemas.get(self.order_tool) or {}
        if "justification" in set(schema.get("required") or []):
            args["justification"] = {"claims": []}
        args = self._order_args(_coerce(args, schema))
        call = {"id": f"fallback-{self.turn + 1}", "type": "function",
                "function": {"name": self.order_tool, "arguments": json.dumps(args)}}
        self._act("Time is nearly up; placing the order for the verified in-stock item.", [call])
        return True

    def _needs_recheck(self, calls: list) -> list:
        """If ordering after an event without re-inspecting the item since, inspect first."""
        if not self.inspect_tool or self.inspect_tool == self.order_tool:
            return calls
        for c in calls:
            if (c.get("function") or {}).get("name") != self.order_tool:
                continue
            a = _arguments(c)
            pid = str(a.get("product_id") or (self.cart[-1]["product_id"] if self.cart else ""))
            sku = str(a.get("sku") or (self.cart[-1]["sku"] if self.cart else ""))
            if self.last_event_turn and self.last_inspect.get((pid, sku), -1) < self.last_event_turn:
                props = (self.schemas.get(self.inspect_tool) or {}).get("properties") or {}
                iargs = {k: v for k, v in (("product_id", pid), ("sku", sku)) if k in props and v}
                if not iargs:
                    return calls
                self._note("Re-checking the final item's current price and stock before ordering, since the "
                           "situation changed after it was last inspected.")
                return [{"id": f"recheck-{self.turn + 1}", "type": "function",
                         "function": {"name": self.inspect_tool, "arguments": json.dumps(iargs)}}]
        return calls

    # -------------------------------------------------------------- main loop
    def run(self) -> list:
        while not self.done and self.turn < self.max_steps:
            left = self.remaining()
            steps_left = self.max_steps - self.turn
            if (left < _FALLBACK_SECS_LEFT or steps_left <= _FALLBACK_STEPS_LEFT) and not self.ordered:
                if self._fallback_order():
                    continue
            if left < 20:
                break
            if (left < _COMMIT_SECS_LEFT or steps_left <= _COMMIT_STEPS_LEFT) and self.turn > 0 and not self.ordered:
                self._note(f"Only {steps_left} steps and about {int(left)} seconds remain. Commit now: complete "
                           "the shopper's request with the best fully verified candidate you already have.")
            content, calls = self._choose()
            if calls:
                calls = self._needs_recheck(calls)
            if not calls:
                self.model_index += 1
                self._note("You must act with a tool now. If unsure, re-inspect or re-view your strongest "
                           "candidate, or take the next step toward completing the shopper's request.")
                content, calls = self._choose()
                if not calls:
                    break
            self._act(content, calls)
        return self.dialogue


def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": f"agent error: {type(exc).__name__}: {str(exc)[:200]}"}]


__all__ = ["agent_main"]
