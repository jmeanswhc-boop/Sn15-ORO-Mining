"""Model-driven shopping agent for ORO generated environments.

The language model decides everything about the purchase: which queries to run, which
products to read, how to judge them against the shopper's request, what to add to the
cart, when to answer the shopper, and what to order. The code around it is plumbing:

* it hands the model the environment's own goal text and tool contract, unchanged;
* it keeps a notebook of everything observed so trimmed context never loses a product;
* it batches the model's calls into one environment turn and returns the results;
* it enforces the tool contract (argument shapes, one commitment per turn, a fresh
  stock check before an order) and refuses an order the shopper's own budget rules out;
* it keeps the model aware of the remaining turn and time budget.

There is no product knowledge, attribute parser, or ranking rule in this file.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_PROXY = ProxyClient(timeout=75, max_retries=1, retry_delay=1.0)
_CHAT = ProxyClient(timeout=45, max_retries=1, retry_delay=1.0)

_FALLBACK_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}

_EPISODE_BUDGET_S = 270.0   # elapsed seconds since agent_main started; the sandbox kills at 300
_TOOL_RESULT_CHARS = 7000   # per tool result shown to the model
_TURN_RESULT_CHARS = 28000  # all results of one turn together
_CONTEXT_MESSAGES = 24      # recent messages kept verbatim besides the fixed head
_DIGEST_ROWS = 60         # products kept in the digest
_TURN_RESERVE = 3           # turns kept for check + order
_MONEY_EPS = 1e-6


# --------------------------------------------------------------------------- helpers


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:[.,]\d+)*", str(value or ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None

def _sku_key(pid: Any, sku: Any) -> str:
    return "%s::%s" % (pid, "" if sku is None else sku)

def _dict_nodes(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _dict_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _dict_nodes(value)

def _parse_json(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return None
    return None

def _resolve_model() -> str:
    override = getenv("SANDBOX_MODEL")
    if override:
        return override
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    preferred = _FALLBACK_MODELS.get(provider, _FALLBACK_MODELS["chutes"])
    try:
        allow = _PROXY.get("/inference/models")
    except Exception:
        return preferred
    ids: list[str] = []
    raw = allow.get("models") or allow.get("data") or [] if isinstance(allow, dict) else allow
    for entry in raw if isinstance(raw, list) else []:
        name = entry.get("id") if isinstance(entry, dict) else entry
        if isinstance(name, str):
            ids.append(name)
    if not ids or preferred in ids:
        return preferred
    for other in _FALLBACK_MODELS.values():
        if other in ids:
            return other
    return preferred


class TurnPacer:
    """Turn and wall-clock accounting for a session that must end with an order.

    The loop stops on two plain conditions: the step count, and a floor on the
    seconds needed for another model round trip. This is the fuller picture
    behind those two -- how fast turns have actually been going, how many the
    reserve leaves for the closing check-and-order pair, and which of the two
    budgets is the binding one. It reads counters and elapsed time only, and
    never chooses an action.
    """

    __slots__ = ("max_steps", "reserve", "budget_s", "_spans")

    def __init__(self, max_steps: int, reserve: int = _TURN_RESERVE,
                 budget_s: float = _EPISODE_BUDGET_S) -> None:
        self.max_steps = max(1, int(max_steps))
        self.reserve = max(0, int(reserve))
        self.budget_s = float(budget_s)
        self._spans: list[float] = []

    def note_turn(self, seconds: float) -> None:
        if seconds > 0:
            self._spans.append(float(seconds))

    def mean_turn(self) -> float:
        if not self._spans:
            return 0.0
        return round(sum(self._spans) / len(self._spans), 2)

    def turns_affordable(self, seconds_left: float) -> int:
        """How many more turns the remaining wall clock can pay for."""
        pace = self.mean_turn()
        if pace <= 0:
            return self.max_steps
        return max(0, int(seconds_left // pace))

    def binding_budget(self, turn: int, seconds_left: float) -> str:
        by_steps = self.max_steps - int(turn)
        by_clock = self.turns_affordable(seconds_left)
        if by_clock < by_steps:
            return "time"
        if by_steps < by_clock:
            return "turns"
        return "either"

    def report(self, turn: int, seconds_left: float) -> dict:
        return {
            "turn": int(turn),
            "max_steps": self.max_steps,
            "mean_turn_s": self.mean_turn(),
            "turns_affordable": self.turns_affordable(seconds_left),
            "binding": self.binding_budget(turn, seconds_left),
        }


# --------------------------------------------------------------------------- tool contract

class ToolSpec:
    """The environment's tool list, used only to keep calls well-formed."""

    def __init__(self, tools: list[dict]) -> None:
        self.tools = tools
        self.params: dict[str, dict] = {}
        self.required: dict[str, list] = {}
        for tool in tools:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            params = fn.get("parameters") or {}
            self.params[fn["name"]] = params.get("properties") or {}
            self.required[fn["name"]] = list(params.get("required") or [])

    @property
    def tool_names(self) -> list[str]:
        return list(self.params)

    def sanitise(self, name: str, args: Any) -> tuple[dict | None, str | None]:
        """Drop unknown arguments and fix scalar types; report what is missing."""
        if name not in self.params:
            return None, "unknown tool %s" % name
        if not isinstance(args, dict):
            args = {}
        props = self.params[name]
        out: dict[str, Any] = {}
        for key, value in args.items():
            if key not in props:
                continue
            kind = props[key].get("type")
            if kind == "integer":
                number = _to_float(value)
                if number is None:
                    continue
                value = int(number)
                if props[key].get("maximum") is not None:
                    value = min(value, int(props[key]["maximum"]))
                if props[key].get("minimum") is not None:
                    value = max(value, int(props[key]["minimum"]))
            elif kind == "number":
                number = _to_float(value)
                if number is None:
                    continue
                value = number
            elif kind == "boolean":
                value = str(value).lower() in ("true", "1", "yes") if not isinstance(value, bool) else value
            elif kind == "string":
                if value is None:
                    continue
                value = str(value)
            elif kind == "array":
                if isinstance(value, str):
                    value = [v.strip() for v in re.split(r"[,\s]+", value) if v.strip()]
                if not isinstance(value, list):
                    continue
                value = [str(v) for v in value]
            out[key] = value
        missing = [r for r in self.required.get(name, []) if r not in out]
        if missing:
            return None, "%s is missing required argument(s): %s" % (name, ", ".join(missing))
        return out, None

# --------------------------------------------------------------------------- notebook

class Ledger:
    """Everything observed about every product, as the environment reported it."""

    def __init__(self) -> None:
        self.products: dict[str, dict] = {}
        self.variants: dict[str, dict] = {}
        self.order: list[str] = []
        self.cart: dict[str, dict] = {}
        self.seq = 0
        self.checked: dict[str, int] = {}
        self.cart_changed_at = 0
        self.currency: str | None = None

    def entry(self, pid: Any) -> dict:
        pid = str(pid)
        if pid not in self.products:
            self.products[pid] = {"product_id": pid, "facts": {}, "variants": []}
            self.order.append(pid)
        return self.products[pid]

    def observe(self, name: str, obs: Any) -> None:
        self.seq += 1
        if not isinstance(obs, (dict, list)):
            return
        for node in _dict_nodes(obs):
            pid = node.get("product_id") or node.get("id")
            if not pid:
                continue
            prod = self.entry(pid)
            for field in ("title", "brand", "category_path", "listing_id"):
                if node.get(field) is not None:
                    prod[field] = node[field]
            facts = node.get("facts")
            if isinstance(facts, dict):
                prod["facts"].update(facts)
            variants = node.get("variants")
            if isinstance(variants, list):
                for variant in variants:
                    if isinstance(variant, dict) and variant.get("sku") is not None:
                        self._note_sku(pid, variant.get("sku"), variant, name)
            if node.get("sku") is not None and (node.get("price") is not None or node.get("in_stock") is not None):
                self._note_sku(pid, node.get("sku"), node, name)
            if node.get("currency") and not self.currency:
                self.currency = str(node["currency"])
        for node in _dict_nodes(obs):
            added = node.get("added")
            if isinstance(added, dict) and added.get("product_id"):
                self.cart[_sku_key(added["product_id"], added.get("sku"))] = dict(added)
                self.cart_changed_at = self.seq
            removed = node.get("removed")
            if isinstance(removed, dict) and removed.get("product_id"):
                self.cart.pop(_sku_key(removed["product_id"], removed.get("sku")), None)
                self.cart_changed_at = self.seq
            lines = node.get("cart")
            if isinstance(lines, list) and lines and all(isinstance(x, dict) for x in lines):
                if any(x.get("product_id") for x in lines):
                    self.cart = {_sku_key(x["product_id"], x.get("sku")): dict(x) for x in lines if x.get("product_id")}
            event = node.get("market_event")
            if isinstance(event, dict):
                target = event.get("target") or {}
                if target.get("product_id"):
                    key = _sku_key(target["product_id"], target.get("sku"))
                    variant = self.variants.get(key)
                    if variant is not None:
                        if event.get("kind") == "stockout":
                            variant["in_stock"] = False
                        elif _to_float(event.get("new_price")) is not None:
                            variant["price"] = _to_float(event["new_price"])
                        variant["event"] = str(event.get("kind") or "market_event")

    def _note_sku(self, pid: Any, sku: Any, node: dict, name: str) -> None:
        key = _sku_key(pid, sku)
        variant = self.variants.setdefault(key, {"product_id": str(pid), "sku": str(sku)})
        for field in ("price", "current_price", "in_stock", "options", "currency"):
            if node.get(field) is not None:
                variant[field] = node[field]
        if node.get("current_price") is not None:
            variant["price"] = node["current_price"]
        if node.get("price_at_add") is not None:
            variant["price_at_add"] = node["price_at_add"]
        variant["seen"] = self.seq
        if "stock" in (name or "").lower() or name in ("view", "compare"):
            self.checked[key] = self.seq
        prod = self.entry(pid)
        if key not in prod["variants"]:
            prod["variants"].append(key)

    def table(self, limit: int = _DIGEST_ROWS) -> str:
        """Compact table of what has been observed, most recently read products last."""
        rows = []
        read = [pid for pid in self.order if self.products[pid].get("category_path") or self.products[pid].get("facts")]
        unread = [pid for pid in self.order if pid not in set(read)]
        for pid in unread[-40:]:
            prod = self.products[pid]
            v = self.variants.get((prod.get("variants") or [""])[0]) or {}
            rows.append("- %s | %s | %s | unread" % (pid, str(prod.get("title") or "")[:70], v.get("price")))
        for pid in read[-limit:]:
            prod = self.products[pid]
            path = " > ".join(str(x) for x in (prod.get("category_path") or [])) or "category unread"
            facts = prod.get("facts") or {}
            fact_text = "; ".join("%s=%s" % (k, str(v)[:36]) for k, v in list(facts.items())[:6])
            variants = []
            for key in prod.get("variants") or []:
                v = self.variants.get(key) or {}
                variants.append("%s %s %s %s%s" % (
                    v.get("sku"), v.get("options") or "-", v.get("price"),
                    "in-stock" if v.get("in_stock") is True else ("OUT" if v.get("in_stock") is False else "stock?"),
                    " [" + str(v["event"]) + "]" if v.get("event") else ""))
            rows.append("- %s | %s | brand=%s | %s | %s | facts: %s" % (
                pid, str(prod.get("title") or "")[:90], prod.get("brand"), path,
                " / ".join(variants[:6]) or "no variant read", fact_text or "none read"))
        cart = ", ".join("%s (added at %s)" % (k, v.get("price_at_add", v.get("price"))) for k, v in self.cart.items()) or "empty"
        return "Cart: %s\nProducts observed so far (%d):\n%s" % (cart, len(self.order), "\n".join(rows))


class LedgerStats:
    """Counts over what a :class:`Ledger` has actually seen this episode.

    The table the model reads is a list of rows, which is the right shape for a
    prompt and the wrong shape for a question like "how much of this session
    went on listings that were never read". These are those questions, answered
    from the ledger alone: read against unread, how many variants carry a
    price, how many are confirmed in stock, and what the observed prices span.
    Nothing here reaches the model or the environment.
    """

    def __init__(self, notebook: "Ledger") -> None:
        self._nb = notebook

    def read_products(self) -> list[str]:
        return [pid for pid, prod in self._nb.products.items()
                if prod.get("category_path") or prod.get("facts")]

    def coverage(self) -> float:
        """Fraction of the listings observed that were actually read."""
        if not self._nb.order:
            return 0.0
        return round(len(self.read_products()) / len(self._nb.order), 3)

    def priced_variants(self) -> int:
        return sum(1 for v in self._nb.variants.values() if v.get("price") is not None)

    def in_stock_variants(self) -> int:
        return sum(1 for v in self._nb.variants.values() if v.get("in_stock") is True)

    def repriced_variants(self) -> list[str]:
        return [k for k, v in self._nb.variants.items() if v.get("event")]

    def report(self) -> dict:
        return {
            "products": len(self._nb.order),
            "read": len(self.read_products()),
            "coverage": self.coverage(),
            "variants": len(self._nb.variants),
            "priced": self.priced_variants(),
            "in_stock": self.in_stock_variants(),
            "repriced": self.repriced_variants(),
            "cart": list(self._nb.cart),
        }


# --------------------------------------------------------------------------- agent

_AGENT_BRIEF = """You are an autonomous shopping agent working a product catalog through tools on behalf of one shopper. Read the shopper's request and the task rules that follow it carefully; they are the whole specification. You decide everything: what to search, what to read, which product satisfies the shopper, what to add to the cart, how to answer the shopper, and what to order.

How turns work:
- Each reply is one turn. You may return several tool calls in one reply; they run in order as a batch and all their results come back together. Batch generously: in your first turn issue at least three searches with different wording and at least three filter calls; when reading, compare every plausible candidate in the same turn (several compare calls of up to 5 products each). Turns are limited, calls within a turn are cheap, and a turn with a single call is usually a wasted turn.
- search returns 10 products per query, so vary the wording to reach more. filter is a cheap structured lookup: use the requested category, try it with and without a brand, and with a few different max_price levels, to see the catalog's own view of the category.
- Judge a product only from what view or compare shows: its category_path, its facts, its variants' options, price and stock, and its title. Prefer listings whose own facts state the requirement over listings where only the title hints at it. Reject accessories, spare parts, cases, bundles of something else, and listings that are not the kind of device the shopper asked for.
- Read only what could qualify: when a quoted phrase is required, compare only listings whose title contains it; when a category is required, compare only listings that could be that kind of item. Every product you view or compare is treated as part of your shortlist.
- A stated measurement is exact: a 120 Hz request means 120 Hz, not 144 Hz; 8 GB means 8 GB. A quoted phrase must appear literally in the listing title. When two listings both look like the requested product, prefer the one whose title, brand and model facts agree with each other.
- When the shopper lists candidate variants, the choice is among those only: read them, rank them by the shopper's rules, and buy from them. Do not search the catalog for others.
- Variants matter: the sku you add must itself have the requested colour, size or capacity and be in stock.
- The category_path is where the catalog shelved the listing; the listing's own title, facts and options say what it is. When the shelf label and the listing's own description disagree, the listing's own words win in both directions: a title that says in-ear is an in-ear product even when shelved under on-ear, and a title that says on-ear is on-ear even when shelved under earbuds. Still reject anything that is plainly an accessory or part.
- Prefer listings that state a brand and whose brand, title and model fact spell the same name, for the first pick and for every replacement after a price change or stockout; a listing with no brand, or whose title spells the brand with look-alike characters, is a copy of the real product, not the product.
- When the shopper names a preferred product, brand, model or colour, your first cart addition should be that exact pick whenever it is in stock and within budget. Fall back to alternatives only after it proves unavailable, following the fallback order the shopper gave.
- Removing items from the cart is never needed; an unavailable item can simply be left there while you add and order a different one.
- Adding to the cart is a commitment. Before you add, have read the product with view or compare. After you add, prices and stock can change and the shopper may send a message; always inspect_stock the added variant before ordering, and read any shopper message. If the added variant went out of stock or above budget, choose the next best product the same way and add it; do not order it.
- When the shopper asks you a question, answer it briefly with the message tool, in the same turn as your next actions. Do not send status updates, and never wait for the shopper: they only speak when they have something to say.
- When the item you committed to becomes unavailable or overpriced, the same product is often listed again under another product_id with the same title: filter the category at its former price (filters are free) and, if the task does not limit distinct queries, search its title words; then choose among what you find by the shopper's rules.
- When the shopper asks for a reason, prefer a listing that states a brand (a brand claim is one of the checkable ones), and your claims are checked one by one against the product's own fact list, so pick a product whose facts (the labelled fields shown by view/compare), not just its title, state the requested attribute, and copy the fact label and value verbatim into the claim. A product with no fact for the attribute cannot be justified; choose another.
- Before your first add_to_cart, write in "thinking" which listing is the shopper's exact preferred pick and why its title, brand, model fact and variant options agree with each other; if two listings claim the same model, the one whose title, brand and facts are consistent is the real one.
- place_test_order only for a variant whose most recent inspect_stock shows it in stock and within the shopper's budget, and only on its own after that check. When the order tool requires justification, give claims as field/value pairs whose field is one of the fact labels shown by view/compare for that product (plus price, in_stock and brand) and whose value is copied from those facts; when the shopper wants a reason, choose a product whose own facts state the requested attribute, so the claim can be checked. Do not send justification when the tool does not require it.
- Never repeat a call with identical arguments, and never remove and re-add items to probe the environment.
- Keep going until the order is placed; an episode without an order is worth nothing.

Reply format, every turn, JSON only and nothing else:
{"thinking": "<at most 40 words: what you learned and what this turn does>",
 "calls": [{"name": "<tool name>", "args": {...}}, ...]}
The calls run in order as one batch (at most the per-turn limit). Tool schemas are given below; use exactly those argument names."""


class ClaimReview:
    """Re-check a justification payload against the product record behind it.

    ``Shopper._verify_claims`` already drops any claim the product's own
    reading cannot carry, so on a clean run this finds nothing. It is for the
    opposite case: when the verifier rejects a justification anyway, putting
    the submitted payload back through an independent reading of the same
    record names the one field that disagreed, instead of leaving the whole
    justification under suspicion.
    """

    CHECKABLE = ("price", "in_stock", "brand")

    def __init__(self, product: dict, variant: dict | None) -> None:
        self.product = product or {}
        self.variant = variant or {}

    def _labels(self) -> dict:
        return {str(k).strip().lower(): str(k) for k in (self.product.get("facts") or {})}

    def supports(self, field: str, value: str) -> bool:
        low = str(field).strip().lower().replace(" ", "_")
        if low in ("price", "current_price"):
            price = _to_float(self.variant.get("price"))
            return price is not None and abs(price - (_to_float(value) or -1.0)) < _MONEY_EPS
        if low in ("in_stock", "stock", "availability"):
            return self.variant.get("in_stock") is True and str(value).lower() == "true"
        if low == "brand":
            return bool(self.product.get("brand")) and str(value).strip().lower() \
                in str(self.product["brand"]).strip().lower()
        labels = self._labels()
        label = labels.get(low) or labels.get(low.replace("_", " "))
        if not label:
            return False
        facts = self.product.get("facts") or {}
        return " ".join(str(facts[label]).split()) == " ".join(str(value).split())

    def audit(self, justification: Any) -> dict:
        claims = justification.get("claims") if isinstance(justification, dict) else justification
        unsupported, fields = [], []
        for claim in claims if isinstance(claims, list) else []:
            if not isinstance(claim, dict):
                continue
            field = str(claim.get("field") or "")
            fields.append(field)
            if not self.supports(field, str(claim.get("value") or "")):
                unsupported.append(field)
        fact_claims = [f for f in fields if f not in self.CHECKABLE]
        return {
            "claims": len(fields),
            "fields": fields,
            "fact_claims": fact_claims,
            "unsupported": unsupported,
            "faithful": not unsupported and bool(fact_claims),
        }


class Shopper:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"] or {}
        self.tools = list(self.policy.get("tools") or [])
        self.contract = ToolSpec(self.tools)
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 16))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.query = str(self.policy.get("query") or problem.get("query") or "")
        self.model = _resolve_model()
        self.notebook = Ledger()
        self.started = time.monotonic()
        self.turn = 0
        self.done = False
        self.ordered = False
        self.tried: set = set()
        self.budget: float | None = None
        self.brief: dict = {}
        contract_text = "\n\nTOOL SCHEMAS (per-turn call limit %d):\n%s" % (
            self.max_calls, json.dumps([t.get("function", t) for t in self.tools], ensure_ascii=False))
        self.messages: list[dict] = [
            {"role": "system", "content": _AGENT_BRIEF + contract_text},
            {"role": "user", "content": self.query},
        ]
        self.dialogue: list[dict] = []
        self.pending_envelope: dict | None = None
        self.last_shopper_message: str | None = None
        self.verdicts: dict[str, tuple[bool, str]] = {}
        self.brand_warned: set = set()

    # -- budgets: turns and seconds -----------------------------------------
    def time_left(self) -> float:
        return _EPISODE_BUDGET_S - (time.monotonic() - self.started)

    def _turn_floor(self) -> float:
        """Seconds that must remain before another model turn is started: enough for a call as
        slow as the recent ones plus the environment round trip."""
        recent = getattr(self, "latencies", []) or [12.0]
        return max(25.0, 1.6 * max(recent) + 6.0)

    # -- model transport -----------------------------------------------------
    def _call_model(self, messages: list[dict], **extra) -> dict:
        started = time.monotonic()
        try:
            reply = _CHAT.post("/inference/chat/completions", json_data={
                "model": self.model, "messages": messages, "temperature": 0, **extra})
        except Exception:
            reply = None
        message = ((reply or {}).get("choices") or [{}])[0].get("message") or {}
        self.last_call_failed = not message
        if message:
            self.latencies = (getattr(self, "latencies", []) + [time.monotonic() - started])[-4:]
        return message

    def _prompt_window(self) -> list[dict]:
        head, tail = self.messages[:2], self.messages[2:]
        if len(tail) > _CONTEXT_MESSAGES:
            cut = len(tail) - _CONTEXT_MESSAGES
            while cut < len(tail) and tail[cut].get("role") == "tool":
                cut += 1
            tail = tail[cut:]
        status = ("Turn %d of %d used; about %d seconds of working time remain. %s"
                  % (self.turn, self.max_steps, max(0, int(self.time_left())),
                     "" if not self.budget else "Budget you stated: %s %s." % (self.budget, self.notebook.currency or "")))
        notebook = {"role": "system", "content": status + "\n\n" + self.notebook.table()}
        return head + [notebook] + tail

    def _read_brief(self) -> None:
        """Ask the model to restate the request; the budget it names is the only value the
        code uses, as a safety check on orders."""
        message = self._call_model([
            {"role": "system", "content": (
                "Restate a shopper's request as JSON with keys: budget (number or null), currency (string or null), "
                "category (string), firm_requirements (list of strings), preferences (list of strings, in the order "
                "the shopper would give them up, first to give up first), shopper_fallback_rule (string or null), "
                "reason_required (boolean: does the shopper ask for a reason or justification), quoted_phrase (the exact phrase "
                "in quotation marks that listing titles must contain, or null), plan (one paragraph: "
                "how you will search, read, choose, and verify). JSON only.")},
            {"role": "user", "content": self.query},
        ])
        parsed = _parse_json(str(message.get("content") or ""))
        if isinstance(parsed, dict):
            self.brief = parsed
            self.budget = _to_float(parsed.get("budget"))
            plan = {"role": "assistant", "content": "My reading of the request and plan:\n" + json.dumps(parsed, ensure_ascii=False)}
            self.messages.append(plan)
            self.dialogue.append(dict(plan))

    # -- environment ---------------------------------------------------------
    def _post_envelope(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            try:
                result = _PROXY.post("/environment/call", json_data=envelope)
            except Exception:
                result = None
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.time_left() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def _run_batch(self, actions: list[tuple[str, dict]], content: str = "") -> None:
        if not actions:
            return
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self._post_envelope(pend["envelope"])
            self.pending_envelope = None
            if replay is not None:
                self.turn = pend["turn"]
                self._ingest_result(pend["actions"], pend["content"], replay)
                if self.done:
                    return
            else:
                self.messages.append({"role": "user", "content": "The environment did not answer; that action is still pending."})
                return
        turn = self.turn + 1
        group = "%s-turn-%d" % (self.problem_id, turn)
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [{"call_id": "%s-%d" % (group, i), "action": {"name": name, "args": args}}
                      for i, (name, args) in enumerate(actions, start=1)],
        }
        for name, args in actions:
            self.tried.add(name + json.dumps(args, sort_keys=True, default=str))
        result = self._post_envelope(envelope)
        if result is None:
            self.pending_envelope = {"envelope": envelope, "turn": turn, "actions": actions, "content": content}
            return
        self.turn = turn
        self._ingest_result(actions, content, result)
        counted = result.get("solver_turn_count")
        if isinstance(counted, int) and counted > 0:
            self.turn = counted

    def _ingest_result(self, actions: list[tuple[str, dict]], content: str, result: dict) -> None:
        calls = list(result.get("calls") or [])
        tool_calls = [{"id": "call-%d-%d" % (self.turn, i), "type": "function",
                       "function": {"name": name, "arguments": json.dumps(args, default=str)}}
                      for i, (name, args) in enumerate(actions, start=1)]
        self.messages.append({"role": "assistant", "content": json.dumps(
            {"thinking": content or "", "calls": [{"name": n, "args": a} for n, a in actions]}, default=str, ensure_ascii=False)})
        self.dialogue.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls,
                              "environment_result": result})
        parts = []
        for index, (name, args) in enumerate(actions):
            payload = (calls[index] if index < len(calls) else {}).get("observation") or {}
            obs = payload.get("observation")
            error = payload.get("error")
            if obs is not None:
                self.notebook.observe(name, obs)
            if name == "place_test_order" and not error:
                self.ordered = True
            body = json.dumps({"error": error} if error else obs, default=str, ensure_ascii=False)
            if len(body) > _TOOL_RESULT_CHARS:
                body = body[:_TOOL_RESULT_CHARS] + " ...[truncated; the notebook keeps every product observed]"
            parts.append("Result of %s %s:\n%s" % (name, json.dumps(args, default=str), body))
            if payload.get("done"):
                self.done = True
        text = "Turn %d results:\n\n" % self.turn + "\n\n".join(parts)
        if len(text) > _TURN_RESULT_CHARS:
            text = text[:_TURN_RESULT_CHARS] + " ...[remaining results truncated; every product observed is in the notebook]"
        user_message = result.get("user_message")
        if isinstance(user_message, dict) and user_message.get("content"):
            shopper = str(user_message["content"])
            self.last_shopper_message = shopper
            text += "\n\nMessage from the shopper: " + shopper
            self._budget_from_note(shopper)
        self.messages.append({"role": "user", "content": text})

    def _budget_from_note(self, text: str) -> None:
        message = self._call_model([
            {"role": "system", "content": (
                "A shopper sent this message during a purchase. If it states a new maximum budget, reply with JSON "
                "{\"budget\": number}. Otherwise reply {\"budget\": null}. JSON only.")},
            {"role": "user", "content": text},
        ])
        parsed = _parse_json(str(message.get("content") or ""))
        value = _to_float(parsed.get("budget")) if isinstance(parsed, dict) else None
        if value is not None and (self.budget is None or value < self.budget):
            self.budget = value

    # -- the model's turn ----------------------------------------------------
    def _ask_for_actions(self) -> tuple[list[tuple[str, dict]], str, list[str]]:
        message = self._call_model(self._prompt_window(), response_format={"type": "json_object"})
        content = str(message.get("content") or "")
        parsed = _parse_json(content)
        actions: list[tuple[str, dict]] = []
        problems: list[str] = []
        calls = parsed.get("calls") if isinstance(parsed, dict) else None
        if isinstance(parsed, dict) and not calls and isinstance(parsed.get("name"), str):
            calls = [parsed]
        if not isinstance(calls, list):
            problems.append("The reply was not a JSON object with a non-empty \"calls\" list.")
            calls = []
        for call in calls[: self.max_calls]:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or call.get("tool") or "")
            args = call.get("args", call.get("arguments", {}))
            if isinstance(args, str):
                args = _parse_json(args) or {}
            coerced, problem = self.contract.sanitise(name, args)
            if coerced is None:
                problems.append(problem or name)
                continue
            actions.append((name, coerced))
        thinking = str(parsed.get("thinking") or "") if isinstance(parsed, dict) else ""
        return actions, thinking or content[:400], problems

    def _verify_claims(self, justification: Any, pid: str, variant: dict | None) -> tuple[list[dict], list[str]]:
        """Keep the model's claims that name something the product's own reading carries: one of
        its fact labels, its brand, its current price, or its stock. Values stay the model's,
        except price and stock, which are copied from the latest reading."""
        claims = justification.get("claims") if isinstance(justification, dict) else justification
        prod = self.notebook.products.get(pid) or {}
        facts = prod.get("facts") or {}
        labels = {str(k).strip().lower(): str(k) for k in facts}
        out, dropped = [], []
        for claim in claims if isinstance(claims, list) else []:
            if not isinstance(claim, dict) or claim.get("field") is None or claim.get("value") is None:
                continue
            field = str(claim["field"]).strip()
            low = field.lower().replace(" ", "_")
            if low in ("price", "current_price"):
                price = _to_float((variant or {}).get("price"))
                if price is not None:
                    out.append({"field": "price", "value": "%.2f" % price})
                continue
            if low in ("in_stock", "stock", "availability"):
                if (variant or {}).get("in_stock") is True:
                    out.append({"field": "in_stock", "value": "true"})
                continue
            if low == "brand":
                if prod.get("brand"):
                    out.append({"field": "brand", "value": str(prod["brand"])})
                continue
            label = labels.get(low) or labels.get(low.replace("_", " "))
            if label:
                # The verifier compares a fact claim to the fact itself, so the value is the
                # fact's own text, not the model's paraphrase of it.
                out.append({"field": label, "value": " ".join(str(facts[label]).split())})
                continue
            dropped.append(field)
        # Every claim beyond what the contract asks for is one more thing that can be judged
        # wrong: keep the first fact claim (the requested attribute), price, stock, and one
        # supporting reason, the structured brand when the listing states one.
        facts_claims = [c for c in out if c["field"] not in ("price", "in_stock", "brand")]
        others = [c for c in out if c["field"] in ("price", "in_stock")]
        brand_claim = next((c for c in out if c["field"] == "brand"), None)
        support = brand_claim or (facts_claims[1] if len(facts_claims) > 1 else None)
        out = facts_claims[:1] + others + ([support] if support else [])
        return out, dropped

    def _title_carries_phrase(self, pid: str) -> bool:
        """The task's own rule for phrase tasks: the quoted phrase must sit in the listing title."""
        phrase = str(self.brief.get("quoted_phrase") or "").strip().strip('"').strip("\u201c\u201d").lower()
        if not phrase:
            return True
        title = str((self.notebook.products.get(pid) or {}).get("title") or "").lower()
        if not title:
            return True  # unseen title cannot be judged; the model chose it from a result row
        words = re.findall(r"[a-z0-9]+", phrase)
        title_words = set(re.findall(r"[a-z0-9]+", title))
        return all(w in title_words for w in words)

    def _apply_contract(self, actions: list[tuple[str, dict]]) -> tuple[list[tuple[str, dict]], list[str]]:
        """Keep the batch inside the tool contract; the model keeps the choices."""
        kept: list[tuple[str, dict]] = []
        notes: list[str] = []
        seen_mutation = False
        for name, args in actions:
            if name in ("view", "compare") and self.brief.get("quoted_phrase"):
                pids = [str(args.get("product_id"))] if name == "view" else [str(p) for p in args.get("product_ids") or []]
                skipped = [p for p in pids if not self._title_carries_phrase(p)]
                if skipped:
                    notes.append("%s skipped %s: their titles do not contain the required phrase, and every product you read counts as shortlisted." % (name, ", ".join(skipped)))
                    pids = [p for p in pids if p not in skipped]
                    if not pids:
                        continue
                    args = {"product_id": pids[0]} if name == "view" else dict(args, product_ids=pids)
            sig = name + json.dumps(args, sort_keys=True, default=str)
            if sig in self.tried and name not in ("inspect_stock", "inspect_cart"):
                notes.append("%s with those exact arguments was already made; its result is in the notebook." % name)
                continue
            if name in ("add_to_cart", "place_test_order", "remove_from_cart"):
                pid = str(args.get("product_id") or "")
                key = _sku_key(pid, args.get("sku"))
                variant = self.notebook.variants.get(key)
                if pid not in self.notebook.products:
                    notes.append("%s refused: product %s has not been observed in this episode." % (name, pid))
                    continue
                if name != "remove_from_cart" and variant is None and args.get("sku") is not None:
                    notes.append("%s refused: variant %s has not been observed; view the product first." % (name, key))
                    continue
                if name != "remove_from_cart" and variant is not None:
                    if variant.get("in_stock") is False:
                        notes.append("%s refused: variant %s was last seen out of stock." % (name, key))
                        continue
                    price = _to_float(variant.get("price"))
                    if self.budget is not None and price is not None and price > self.budget + _MONEY_EPS:
                        notes.append("%s refused: variant %s was last seen at %s, above the shopper's budget of %s."
                                     % (name, key, price, self.budget))
                        continue
                if seen_mutation:
                    notes.append("%s deferred: one cart change or order per turn, so its result can be read first." % name)
                    continue
                if name == "add_to_cart" and "justification" in self.contract.required.get("place_test_order", []) \
                        and not str((self.notebook.products.get(pid) or {}).get("brand") or "").strip() \
                        and pid not in self.brand_warned:
                    self.brand_warned.add(pid)
                    notes.append("add_to_cart %s held for one turn: this listing states no brand, and the shopper wants a "
                                 "reason they can check; a brand is one of the checkable claims. Add it again if it is "
                                 "still your choice, or pick a listing that states its brand." % pid)
                    continue
                if name == "place_test_order":
                    if not self.notebook.cart:
                        notes.append("place_test_order refused: nothing is in the cart yet; add the item first.")
                        continue
                    if variant is not None and self.notebook.checked.get(key, 0) <= self.notebook.cart_changed_at:
                        notes.append("place_test_order deferred: inspect_stock %s first, then order in the next turn." % key)
                        kept.append(("inspect_stock", {"product_id": pid, "sku": str(args.get("sku"))}))
                        seen_mutation = True
                        continue
                    if "justification" in self.contract.required.get(name, []):
                        claims, dropped = self._verify_claims(args.get("justification"), pid, variant)
                        if dropped:
                            notes.append("Claims dropped because the product's read facts, price or stock do not carry "
                                         "that field: %s." % ", ".join(dropped))
                        if not any(c["field"] not in ("price", "in_stock", "brand") for c in claims):
                            notes.append("place_test_order refused: the justification names no fact of the product. "
                                         "Claim the requested attribute using one of this product's own fact labels, "
                                         "or choose a product whose facts state it.")
                            continue
                        args = dict(args, justification={"claims": claims})
                    else:
                        # optional justification can only be judged wrong; the contract says it is optional
                        args = {k: v for k, v in args.items() if k != "justification"}
                seen_mutation = True
            kept.append((name, args))
        return kept[: self.max_calls], notes

    def _fallback_order(self) -> tuple[str, dict] | None:
        """With no time left for another model turn, order the most recently added cart variant
        that the model itself verified in stock and within budget after adding it."""
        if "place_test_order" not in self.contract.params or "justification" in self.contract.required.get("place_test_order", []):
            return None
        best = None
        for key, line in self.notebook.cart.items():
            variant = self.notebook.variants.get(key) or {}
            if self.notebook.checked.get(key, 0) <= self.notebook.cart_changed_at:
                continue
            if variant.get("in_stock") is not True:
                continue
            price = _to_float(variant.get("price"))
            if self.budget is not None and price is not None and price > self.budget + _MONEY_EPS:
                continue
            if best is None or variant.get("seen", 0) > best[1].get("seen", 0):
                best = (key, variant)
        if best is None:
            return None
        return "place_test_order", {"product_id": best[1]["product_id"], "sku": best[1]["sku"]}

    def _review_cart_add(self, pid: str, key: str) -> tuple[bool, str]:
        """Ask the model to re-read the product record against the request before the cart
        changes. The model judges; the code only relays its verdict."""
        if key in self.verdicts:
            return self.verdicts[key]
        prod = self.notebook.products.get(pid) or {}
        record = {k: v for k, v in prod.items() if k != "variants"}
        record["variants"] = [self.notebook.variants.get(k) for k in prod.get("variants") or []]
        message = self._call_model([
            {"role": "system", "content": (
                "You audit a shopping agent's cart decision. You get the shopper's request, the agent's reading of it, "
                "the full record of the product as the catalog reported it, and the exact variant about to be added. "
                "Decide from the record alone whether this variant is what the shopper asked for: it must be the kind of "
                "item requested (not an accessory or part), meet every firm requirement by its own facts, options or title, "
                "be the variant with the requested colour/size/capacity, be in stock, and be within the budget in force. "
                "If the shopper named a preferred product, brand, model or colour, the first cart addition should be that "
                "exact pick when it is available. Reply with JSON only: {\"approve\": true|false, \"reason\": \"one sentence\"}.")},
            {"role": "user", "content": json.dumps({
                "request": self.query.split("\n\nTask rules:")[0],
                "agent_reading": self.brief,
                "budget_in_force": self.budget,
                "last_shopper_message": self.last_shopper_message,
                "cart_so_far": list(self.notebook.cart),
                "product_record": record,
                "variant_to_add": key,
            }, ensure_ascii=False, default=str)},
        ])
        parsed = _parse_json(str(message.get("content") or ""))
        if not isinstance(parsed, dict) or "approve" not in parsed:
            verdict = (True, "")
        else:
            verdict = (bool(parsed.get("approve")), str(parsed.get("reason") or ""))
        self.verdicts[key] = verdict
        return verdict

    # -- loop ----------------------------------------------------------------
    def run(self) -> list[dict]:
        self._read_brief()
        idle = 0
        while not self.done and self.turn < self.max_steps and self.time_left() > self._turn_floor():
            left = self.max_steps - self.turn
            if left <= _TURN_RESERVE and not self.ordered:
                self.messages.append({"role": "user", "content": (
                    "Only %d turn(s) remain. Place the order now for the best variant you have verified in stock "
                    "and within budget; if it still needs a stock check, do that check this turn and order next turn."
                    % left)})
            elif self.time_left() < 60 and not self.ordered:
                self.messages.append({"role": "user", "content": "Working time is nearly up; verify and order your best candidate now."})
            if not self.ordered and self.time_left() < self._turn_floor() + 30:
                fallback = self._fallback_order()
                if fallback:
                    self._run_batch([fallback], "Time is up; ordering the verified item already in the cart.")
                    continue
            actions, content, problems = self._ask_for_actions()
            if getattr(self, "last_call_failed", False):
                continue  # the model did not answer in time; try again while time remains
            kept, notes = self._apply_contract(actions)
            if problems or notes:
                note = {"role": "user", "content": "Note from the tool layer: " + " ".join(problems + notes)}
                self.messages.append(note)
                self.dialogue.append(dict(note))
            if not kept:
                idle += 1
                if idle >= 4:
                    break
                continue
            idle = 0
            self._run_batch(kept, content)
        return self.dialogue


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    shopper = None
    try:
        shopper = Shopper(problem_data)
        return shopper.run()
    except Exception:
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return shopper.dialogue if shopper is not None else []