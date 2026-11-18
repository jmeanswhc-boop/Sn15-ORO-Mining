"""Tallyman: a four-stage shopping agent for ORO store worlds.

Stage 1 (scout): a cheap model browses with the world's read-only tools and the
code files every record it sees. Stage 2 (interview): code, not the model,
spends the shopper's message allowance one property at a time, choosing the
properties from the attributes the scout actually saw varying. Stage 3 (judge):
a stronger model picks the lines (or the decline) from the evidence on file.
Stage 4 (closer): code builds the cart, checks it, orders, and feeds any
refusal back into the ledger for another round.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

FAST = "anthropic/claude-haiku-5.5"
DEEP = "anthropic/claude-sonnet-5.5"
BACKUP = "openai/gpt-6-luna"

# The network gives a task 300 s; the agent measures its own elapsed time from the first line of agent_main.
CLOCK_TOTAL = 255.0         # seconds the agent plans to use
CLOCK_CLOSE_IN = 80.0       # below this much time left, scouting and interviewing stop and the closer takes over
CLOCK_NO_MODEL = 24.0       # below this much time left, no model call starts
SPEND_CEIL = 0.085          # one run key pays for every task: no model call above this spend
SPEND_DEEP = 0.045          # above this spend, optional checks use the cheap model
BACKOFF = (3, 8, 16, 28)
BACKOFF_TOTAL = 40
SCOUT_TURNS = 40
CLIP_RESULT = 3600
CLIP_FIELD = 70

_infer = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_world = ProxyClient(timeout=60, max_retries=3, retry_delay=1)


# --------------------------------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------------------------------
SCOUT_SYSTEM = """You are the scout on a shopping team. A shopper wants goods from an online store. Your tools are read-only: search, open pages, compare. You never buy, never write to the shopper and never change the cart; teammates handle those parts after you.

What you produce is evidence. For every wanted product, track down listings that might meet all of its conditions and read their attribute pages, so that the judge decides from facts.

How to work:
- Each tool call spends one of a limited number of steps; reasoning is free. Copy tool and argument names exactly from the schemas. If a tool can show several listings in one call, use it instead of many single calls.
- Search rows are only summaries. Whether a condition on some property holds can only be read from the listing's attributes, so open the pages of the listings you are considering.
- Search for the KIND of product first, with the words of its name and of its store category rather than attribute values; colour, pattern, material and similar conditions are checked on the attribute pages later. A query is the kind plus at most one more word. Never substitute a neighbouring kind of product to get hits.
- An empty result usually means a word or filter is wrong, not that the product is missing: cut the query down to the product noun, or browse the category with the store's category filter.
- If a result reports more rows than it displayed and the decision needs all of them, request the next page using exactly the token or offset the store provides.
- If a condition takes its value from another listing, locate that listing, open its full page, and keep following pointers to further listings until a page states the value.
- If the wanted product has to work with another product (an add-on for a device), look for the pair using what the listings themselves say about compatibility.
- Do not reopen a page you have already read unless something changed.
- When you have collected enough for the goal you were given, reply in two plain lines summarising what you found, and call no tool."""

BRIEF = """TASK:BRIEF
You will read a shopper's request together with the tools of an online store. Answer with a single compact JSON object and no other text, using these keys.
"products": one entry for each distinct thing to be bought: {"label": the name used in the request, "aisle": the store's category name for it if a tool lists category names (copy it exactly), otherwise "", "count": number of units of that single listing (1 if not stated; a pair or set sold as one listing is 1), "requirements": [{"text": "property: value or limit", "points_elsewhere": true if the value has to be taken from another listing the request describes}]}. Requirements are the hard conditions on the product: exact values, limits, exclusions, closed lists. Use the request's own noun for each property. Background (who it is for, where it will go, what went wrong before) is not a requirement unless it plainly states a value. Wording that is just the product's name, or a season, audience or use already stated by its name or store category, is not a requirement. If a requirement takes its value from another listing, put the entire description of that listing inside this one requirement; that listing's shop, kind and features are not conditions on this product. Never list price, budget, stock or count as requirements. A single phrase under one quantity ("X and Y") is one product; separate products only when each has its own quantity or conditions.
"limit": {"amount": number or null, "unit": the unit as the request words it, or null, "per": "order" or "unit" or null}
"in_stock_only": true except when the request accepts items that cannot be bought now
"rank": "cheapest" if it asks for the lowest price, "most" or "least" if it asks for the extreme of a measure, otherwise "any" (worry about money alone is not a rank)
"rank_measure": the measure for most/least, else ""
"message_allowance": the number of messages the request lets the agent send to the shopper, integer or null
"step_allowance": the number of steps or tool calls the request permits, integer or null
"instructions": each instruction about how to work or how to end, at most 12 words apiece
"nice_to_have": things the shopper only wishes for
"check_with_shopper": things the request says to confirm with the shopper before choosing
"procedure": ONLY actions the request explicitly orders the agent to carry out besides finding and buying (get approval, tell or show the shopper something, read something first). A remark that the shopper has unstated tastes or "won't guess" is NOT an action; message and step allowances are not actions. Format: [{"do": at most 15 words, "before": "choosing" or "ordering" or "declining", "messages": messages to the shopper needed, 0 if none}]
"no_alternatives": true if the request forbids suggesting alternatives when nothing fits, false if it asks for them, otherwise null
"tool_map": the tool that does each job, null if none: "search", "open" (one listing's attributes; if two tools open one listing, take the one that advertises attributes, properties or specs), "offer" (live price and stock, when that is another tool than open), "compare" (several listings in full), "ask" (message the shopper), "cart" (modify the cart), "cart_view", "buy" (check out), "decline" (finish without buying)
"order_args": fixed arguments that turn the buy tool into a checkout (a mode value, say), else {}
"decline_args": fixed arguments that turn the decline tool into ending without buying, else {}
"""

SHAPE = """TASK:SHAPE
The result below came from a store tool. Identify its parts using the exact keys that occur in it, and answer with a single JSON object only; use null for anything missing:
{"id": key within one listing record that holds its identifier, "title": key of its name, "price": key of its price, "stock": key telling whether it can be bought now, "category": key of its store category label, "facts": key of the nested object or list that carries its attributes/properties, "total": top-level key giving the overall number of matches, "next": top-level key holding the value that asks for the next page, "next_param": name of the tool parameter that receives that value}"""

PLAN = """TASK:PLAN
You decide what to ask a shopper. The shopper has a few firm tastes that were never written down; each concerns ONE property of one wanted product, and a message uncovers it only when it names exactly that property. A message gets answered about the first property it mentions, so a message equals a property.

You are given the request, each product with what is already settled, and a table of the attributes found on opened listings of that product's category: how many listings carry the attribute, how many distinct values show up, and samples. A hidden taste may lie on a property that no table row names.

Put the properties to ask in this order, best first:
1. Anything the request says to confirm with the shopper.
2. Properties the story only points at (a sentence about what the shopper leaves to you, about taste, appearance, surroundings, the person who will use it, what goes in it). Name the property plainly and tie it to the right product. A hint about how it looks, matches or clashes covers EVERY appearance property still open (colour, pattern, finish and similar): give each its own entry ahead of any other kind of property.
3. Then table rows that separate listings of this product: attributes whose values vary between listings and that most listings have, those specific to the product ahead of broad ones every product has. Leave out attributes with one value throughout, identifiers, logistics or selling terms, bundles of purchase options, and whatever the request already settles.
A label is the everyday name of a product attribute (the way an attribute key would be named), at most four words, one property, no sample values, never a person's name or a phrase from the story.
Give COUNT entries, as a single JSON object only: {"ask": [{"product": number, "label": "..."}]}"""

DIGEST = """TASK:DIGEST
A shopping agent was handed the text below by the shopper or the store. Answer with a single JSON object only:
{"firm": [{"product": number, "need": "property: exact value or limit as stated", "points": true if the value is taken from another listing}] (only needs the TEXT itself states; never restate the requirements listed above it), "indifferent": [{"product": number, "property": "..."}], "dropped": [{"product": number, "need": "a requirement that is no longer wanted"}], "limit": a new overall budget as a number, or null, "gone": [ids of listings reported unavailable], "sold_out": true if a chosen listing is reported gone without being named, "approved": true or false if the text answers a request for approval, else null, "kind": "change" if the task itself shifted (budget, requirement, price, availability, order refused), "error" if it merely reports a malformed or rejected call, else "other", "summary": one short sentence}
Name a need with the property that the text itself uses, even if the agent asked about another one. Copy values word for word with their units. Indirect or folksy phrasing that still gives a value counts as stating it. "Don't mind" or "whatever" is indifferent, not firm. A requirement that is swapped counts as one dropped plus one firm."""

JUDGE = """TASK:JUDGE
You are the judge on a shopping team. From the evidence on file choose what to buy, or conclude nothing fits. The episode earns 1 only when the final order is exactly right: each product present in its count, each satisfying every requirement now in force, within budget, following the rank rule; or, when the store truly cannot satisfy everything, a decline.

Reading a listing:
- Its category label has to match the wanted product as a whole; a nearby category is not a match. A requirement whose value is already spelled out by the category's own name (a season or audience, say) is satisfied by that category.
- A condition on a property holds when the listing's attribute for that same property states the value. Spelling of the key, letter case, plurals and one entry of a list do not matter, and a value that includes every word of the wanted one is accepted. A value that uses a different word fails, even a near-synonym. Convert units, treat limits as inclusive, and require a range to sit completely inside a limit.
- When an attribute is keyed by that very property it is decisive: a different value there fails the listing no matter what the title says. Attributes about other properties, even close relatives, settle nothing.
- Only if no attribute names the property may the title, description or some other attribute that states the value be used; a lone number with no stated meaning never counts.
- A condition that takes its value from another listing holds only if that other listing's page was opened (see records) and the chosen listing agrees with it (or, for "unlike", does not share the value).
- Add-on and device pairs have to be tied together by what the listings themselves say.
- Listings marked SOLD-OUT cannot be bought, and "news" lines among the requirements are facts from the store or shopper that outrank older records. Price x count summed over all lines must stay within the budget. Wishes only break ties.
- Rank "cheapest": the lowest price among ALL listings that fit; "most"/"least": the extreme of the measure among fitting listings, where a listing not stating the measure cannot win. If the fitting set may be incomplete, say what is missing in "search_more".
- Records marked ROW come from search rows or description views only: they can neither pass nor fail on any property the row leaves out. Prefer to choose OPENED listings; choose a ROW listing only when no OPENED one fits, and then also put it in "open" so its attributes get read.
Answer with a single JSON object only: {"lines": [{"product": number, "id": "listing id", "count": units}], "decline": true only if no listing can satisfy everything, "open": [up to 6 ROW ids worth opening before deciding], "search_more": "what else to look for, or empty", "why": "one sentence with the deciding facts"}. Provide "lines" and leave out "decline" whenever any combination fits."""

CART_PLAN = """TASK:CART
Work out store calls that leave the cart containing exactly the DESIRED lines and nothing more. Use only the cart tool shown, with its precise argument names and permitted values; an add may raise an existing count and a removal may require a mode or a negative count, so study the schema and the present cart. Answer with a single JSON object only: {"calls": [{"args": {...}}], "note": "one sentence"}. Give an empty list if the cart already matches."""

COMPOSE = """TASK:COMPOSE
Produce the arguments for ONE call to the store tool below so that the call accomplishes the INTENT. Take argument names, permitted values and types from the schema, and include any fixed arguments given. Answer with a single JSON object containing only the arguments."""

CART_CHECK = """TASK:CARTCHECK
Does the cart below hold exactly the DESIRED lines (each listing in its count, nothing extra, nothing sold out), with the total inside the budget wherever prices are visible? Answer with a single JSON object only: {"ok": true or false, "problem": "short and concrete, or empty"}"""

MESSAGE = """TASK:MESSAGE
Write the message the agent will send to the shopper to carry out the step below: one or two plain, polite sentences about ONE thing only (no lists, no examples, no several questions rolled together). Reply with the message text alone."""


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------
def jdump(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def jparse(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    fenced = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)
    for chunk in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        if chunk:
            try:
                got = json.loads(chunk)
            except ValueError:
                continue
            return got if isinstance(got, dict) else None
    return None


def clip(value: Any, width: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= width else value[:width] + "..."
    if isinstance(value, dict):
        return {k: clip(v, width, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [clip(v, width, depth + 1) for v in value]
    return value


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, (str, dict)) and value else []


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
        return int(float(value))
    return None


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        found = re.fullmatch(r"\s*[^\d\-]{0,3}(-?\d+(?:[.,]\d+)?)\s*", value)
        if found:
            return float(found.group(1).replace(",", "."))
    if isinstance(value, dict):
        for inner in value.values():
            number = as_number(inner)
            if number is not None:
                return number
    return None


def plain(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    return re.sub(r"[\s_.\-]+", " ", text).strip().lower()


def stems(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in plain(text).split())


def conform(args: Any, schema: Any, path: str = "args") -> tuple[Any, list[str]]:
    """Coerce a value toward a JSON schema (types, enum case, unknown keys dropped); list what stays wrong."""
    faults: list[str] = []
    if not isinstance(schema, dict):
        return args, faults
    kinds = schema.get("type")
    kinds = [kinds] if isinstance(kinds, str) else [k for k in kinds or [] if isinstance(k, str)]

    def ok(v: Any, k: str) -> bool:
        return {"string": isinstance(v, str), "integer": isinstance(v, int) and not isinstance(v, bool),
                "number": isinstance(v, (int, float)) and not isinstance(v, bool), "boolean": isinstance(v, bool),
                "array": isinstance(v, list), "object": isinstance(v, dict), "null": v is None}.get(k, True)

    if kinds and not any(ok(args, k) for k in kinds):
        for k in kinds:
            if k == "integer" and as_int(args) is not None:
                args = as_int(args)
            elif k == "number" and as_number(args) is not None and not isinstance(args, (list, dict)):
                args = as_number(args)
            elif k == "string" and isinstance(args, (int, float)) and not isinstance(args, bool):
                args = str(args)
            elif k == "boolean" and isinstance(args, str) and args.strip().lower() in ("true", "false"):
                args = args.strip().lower() == "true"
            elif k == "array" and args is not None and not isinstance(args, list):
                args = [args]
            if any(ok(args, kk) for kk in kinds):
                break
        if not any(ok(args, k) for k in kinds):
            return args, [f"{path} must be {'/'.join(kinds)}"]
    options = schema.get("enum")
    if isinstance(options, list) and args not in options:
        low = str(args).strip().lower()
        same = [o for o in options if str(o).lower() == low]
        if same:
            args = same[0]
        else:
            faults.append(f"{path} must be one of {options}")
    if isinstance(args, str) and isinstance(schema.get("maxLength"), int):
        args = args[: schema["maxLength"]]
    if isinstance(args, (int, float)) and not isinstance(args, bool):
        for bound, bad in (("minimum", lambda a, b: a < b), ("maximum", lambda a, b: a > b)):
            if isinstance(schema.get(bound), (int, float)) and bad(args, schema[bound]):
                faults.append(f"{path} violates {bound} {schema[bound]}")
    if isinstance(args, list):
        if isinstance(schema.get("maxItems"), int) and len(args) > schema["maxItems"]:
            args = args[: schema["maxItems"]]
        if isinstance(schema.get("items"), dict):
            fixed = [conform(v, schema["items"], f"{path}[{i}]") for i, v in enumerate(args)]
            args = [v for v, _ in fixed]
            faults += [f for _, fs in fixed for f in fs]
    if isinstance(args, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if props and schema.get("additionalProperties") is False:
            args = {k: v for k, v in args.items() if k in props}
        for need in schema.get("required") or []:
            if need not in args:
                faults.append(f"{path}.{need} is required")
        merged = {}
        for key, value in args.items():
            if key in props:
                merged[key], fs = conform(value, props[key], f"{path}.{key}")
                faults += fs
            else:
                merged[key] = value
        args = merged
    return args, faults


# --------------------------------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------------------------------
class Product:
    def __init__(self, label: str, aisle: str, count: int, reqs: list[tuple[str, bool]]):
        self.label = label
        self.aisle = aisle
        self.count = count
        self.reqs = reqs                      # (text, points_elsewhere)
        self.learned: list[tuple[str, bool]] = []
        self.dropped: list[str] = []
        self.indifferent: list[str] = []
        self.firm = False                     # a shopper reply stated a firm preference for it
        self.asked: list[str] = []

    def live(self) -> list[tuple[str, bool]]:
        return [(t, p) for t, p in self.reqs + self.learned if t not in self.dropped]


class Buyer:
    def __init__(self, problem: dict[str, Any]):
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.view = env["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)
                      and isinstance(t["function"].get("name"), str)]
        self.schema = {t["function"]["name"]: (t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict)
                                                else {"type": "object", "properties": {}}) for t in self.tools}
        cap = as_int(self.view.get("max_steps"))
        self.cap = cap if cap and cap > 0 else 30
        self.step = 0
        self.over = bool(self.view.get("finished"))
        self.t0 = time.monotonic()
        self.spent = 0.0
        self.napped = 0
        self.broke = False
        self.log: list[dict[str, Any]] = []
        # the brief
        self.brief: dict[str, Any] = {}
        self.role: dict[str, str] = {}
        self.products: list[Product] = []
        self.budget: float | None = None
        self.msg_limit = 0
        self.msg_used = 0
        self.proc: list[dict[str, Any]] = []
        self.proc_done: set[str] = set()
        self.notes: list[str] = []
        # evidence
        self.cat: dict[str, dict[str, Any]] = {}       # id -> {"row": {...}, "facts": {...}|None, "opened": bool}
        self.keys: dict[str, str | None] = {}
        self.shaped: set[str] = set()
        self.cache: dict[str, str] = {}
        self.gone: set[str] = set()
        self.digests = 0
        self.noted = 0
        self.cart_problem = ""
        self.verdict: dict[str, Any] = {}
        self.cart_text = ""
        self.stock_tools: list[str] = []
        self.cart_calls_made = False
        self.refusals = 0
        self.deep_search = False
        self.cart_known: list[tuple[str, int]] | None = None

    # ---------------------------------------------------------------- clocks and budgets
    def secs(self) -> float:
        return CLOCK_TOTAL - (time.monotonic() - self.t0)

    def left(self) -> int:
        return self.cap - self.step

    def reserve(self) -> int:
        """Steps kept for the closer: re-read and cart per line, a cart view, the checkout and any ordering messages."""
        if self.deep_search:
            return 2           # a decline needs no cart: every spare step goes to searching
        lines = max(1, len(self.products))
        talk = sum(max(1, p["messages"]) for p in self.proc if p["before"] == "ordering" and p["messages"])
        # stores usually push back at the first checkout (a pick sells out, a requirement changes): keep steps to swap and retry
        recovery = min(9, max(2, self.cap // 5)) if self.refusals == 0 else 3
        return 3 + 2 * lines + talk + recovery

    def free_msgs(self) -> int:
        held = sum(max(1, p["messages"]) for p in self.proc if p["before"] in ("ordering", "declining") and p["messages"]
                   and p["do"] not in self.proc_done)
        return self.msg_limit - self.msg_used - held

    def deciders(self) -> list[str]:
        return [DEEP, FAST] if self.spent < SPEND_DEEP else [FAST, DEEP]

    # ---------------------------------------------------------------- model calls
    def llm(self, models: list[str], messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            for attempt in range(len(BACKOFF) + 1):
                if self.spent >= SPEND_CEIL or self.secs() < CLOCK_NO_MODEL or self.broke:
                    return None
                msg, status = self.llm_once(model, messages, max_tokens, tools)
                if msg is not None:
                    return msg
                if status == 402:
                    self.broke = True
                    return None
                transient = status in (None, 0, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not transient or attempt >= len(BACKOFF) or self.napped + BACKOFF[attempt] > BACKOFF_TOTAL \
                        or self.secs() - BACKOFF[attempt] < CLOCK_NO_MODEL:
                    break
                self.napped += BACKOFF[attempt]
                time.sleep(BACKOFF[attempt])
        return None

    def llm_once(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """One inference request. Returns (message, 200) or (None, failure status; 0 for an empty reply)."""
        outgoing = [dict(m) for m in messages]
        if len(outgoing) > 1 and model.split("/")[0] == "anthropic":
            # mark the stable head and the newest turn as cacheable
            for slot in sorted({0, len(outgoing) - 1}):
                text = outgoing[slot].get("content")
                if text and isinstance(text, str):
                    outgoing[slot]["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": outgoing, "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        res = _infer.post_verbose("/inference/chat/completions", body)
        payload = res.data
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if choices:
            price = (payload.get("usage") or {}).get("cost")
            if isinstance(price, (int, float)):
                self.spent += float(price)
            msg = choices[0].get("message") or {}
            return (msg, 200) if (msg.get("content") or msg.get("tool_calls")) else (None, 0)
        err = res.error or {}
        status = err.get("status")
        inner = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(inner, dict):
            status = as_int(inner.get("code"))
        self.log.append({"kind": "llm_fail", "model": model, "status": status, "body": str(err.get("body") or "")[:160]})
        return None, status

    def ask_json(self, tag_prompt: str, models: list[str], max_tokens: int = 1500) -> dict[str, Any] | None:
        for model in models:
            msg = self.llm([model], [{"role": "user", "content": tag_prompt}], max_tokens)
            got = jparse((msg or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.spent >= SPEND_CEIL:
                break
        return None

    # ---------------------------------------------------------------- the world
    def act(self, name: str, args: dict[str, Any]) -> Any:
        """Make one environment step. Returns the observation, or None when the call could not be made."""
        packet = {**self.binding, "step": self.step + 1, "action": {"name": name, "args": args}}
        for _ in range(4):
            res = _world.post_verbose("/environment/call", packet)
            body = res.data
            if isinstance(body, dict):
                reported = body.get("step")
                self.step = reported if isinstance(reported, int) else self.step + 1
                self.over = self.over or bool(body.get("finished"))
                obs = body.get("observation")
                self.log.append({"kind": "call", "step": self.step, "tool": name, "args": clip(args, 80), "result": clip(obs, 160)})
                self.absorb(name, obs)
                sk = self.keys.get("stock")
                if sk and isinstance(obs, dict) and sk in obs and name not in self.stock_tools and self.single_param(name):
                    self.stock_tools.append(name)
                return obs
            err = res.error or {}
            status, text = err.get("status"), str(err.get("body") or "")
            self.log.append({"kind": "world_fail", "status": status, "body": text[:160]})
            if status == 409:
                lowered = text.lower()
                if any(w in lowered for w in ("finish", "quarantin", "finaliz")):
                    self.over = True
                    return None
                expected = re.search(r"expected (\d+)", text)
                if expected:
                    self.step = int(expected.group(1)) - 1
                elif "reuse" in lowered or "different action" in lowered:
                    self.step += 1
                else:
                    return None
                packet["step"] = self.step + 1
                continue
            if status in (400, 401, 402, 403, 404, 422):
                self.broke = self.broke or status == 402
                return None
        return None

    def tool_call(self, name: str, intent: str, fixed: dict[str, Any] | None = None, note: str = "") -> Any:
        """Have the model write the arguments for `name` that do `intent`, check them against the schema, make the call."""
        schema = self.schema.get(name) or {}
        args: dict[str, Any] = dict(fixed or {})
        _, faults = conform(args, schema)
        if not (schema.get("properties") or {}):
            return self.act(name, args)
        if faults or not fixed:
            prompt = (COMPOSE + "\n\nTOOL: " + name + " | schema: " + jdump(schema) + "\nINTENT: " + intent
                      + ("\nFIXED ARGUMENTS: " + jdump(fixed) if fixed else "") + ("\nNOTE: " + note if note else ""))
            for attempt in range(2):
                got = self.ask_json(prompt, [FAST, DEEP], 800)
                if isinstance(got, dict):
                    args = {**got, **(fixed or {})}
                    args, faults = conform(args, schema)
                    if not faults:
                        break
                    prompt += "\nYour last arguments were wrong: " + "; ".join(faults[:4])
        args, faults = conform(args, schema)
        if faults:
            self.log.append({"kind": "compose_fail", "tool": name, "faults": faults[:4]})
            return None
        return self.act(name, args)

    def single_param(self, tool: str | None) -> str | None:
        schema = self.schema.get(tool or "") or {}
        props = schema.get("properties") or {}
        strings = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v]
        need = schema.get("required") or []
        if tool and len(strings) == 1 and all(k == strings[0] for k in need):
            return strings[0]
        return None

    def list_param(self, tool: str | None) -> str | None:
        schema = self.schema.get(tool or "") or {}
        props = schema.get("properties") or {}
        arrays = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array"]
        need = schema.get("required") or []
        if tool and len(arrays) == 1 and all(k == arrays[0] for k in need):
            return arrays[0]
        return None

    # ---------------------------------------------------------------- evidence
    def row_id(self, node: dict[str, Any]) -> str | None:
        key = self.keys.get("id")
        if key and key in node and isinstance(node[key], (str, int)) and not isinstance(node[key], bool):
            return str(node[key]).strip() or None
        for k, v in node.items():
            low = str(k).lower()
            if (low == "id" or low.endswith("_id") or low.endswith("id") and len(low) <= 12) and isinstance(v, (str, int)) \
                    and not isinstance(v, bool) and 1 <= len(str(v)) <= 64 and " " not in str(v):
                return str(v)
        return None

    def facts_of(self, node: dict[str, Any]) -> dict[str, str] | None:
        pool: Any = node.get(self.keys.get("facts") or "") if self.keys.get("facts") else None
        if pool is None:
            best = None
            for value in node.values():
                if isinstance(value, dict):
                    scalars = [k for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
                    if len(scalars) >= 3 and (best is None or len(scalars) > best[0]):
                        best = (len(scalars), value)
            pool = best[1] if best else None
        out: dict[str, str] = {}
        if isinstance(pool, dict):
            for k, v in pool.items():
                if isinstance(v, (str, int, float, bool)):
                    out[plain(k)] = str(v)
                elif isinstance(v, list) and v and all(isinstance(x, (str, int, float)) for x in v):
                    out[plain(k)] = ", ".join(str(x) for x in v)
        elif isinstance(pool, list):
            for pair in pool:
                if isinstance(pair, dict) and len(pair) >= 2:
                    vals = list(pair.values())
                    if isinstance(vals[0], str) and isinstance(vals[1], (str, int, float)):
                        out[plain(vals[0])] = str(vals[1])
        return out if len(out) >= 2 else None

    def absorb(self, tool: str, obs: Any) -> None:
        if tool not in self.shaped and tool in self.schema and tool not in {self.role.get(r) for r in ("ask", "buy", "decline", "cart", "cart_view")}:
            self.shaped.add(tool)
            self.learn_shape(tool, obs)

        def walk(node: Any, depth: int, in_list: bool) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                wraps = any(isinstance(v, list) and any(isinstance(x, dict) and self.row_id(x) for x in v) for v in node.values())
                ident = None if (wraps and not in_list) else self.row_id(node)
                if ident and (in_list or len(node) >= 3):
                    entry = self.cat.setdefault(ident, {"row": {}, "facts": None, "opened": False})
                    for k, v in node.items():
                        if isinstance(v, (str, int, float, bool)):
                            if entry["row"].get(k) is True and v is False:
                                self.gone.add(ident)
                            entry["row"][k] = v
                    facts = self.facts_of(node)
                    if facts:
                        entry["facts"] = facts
                        entry["opened"] = True
                    return
                for v in node.values():
                    walk(v, depth + 1, False)
            elif isinstance(node, list):
                for v in node:
                    walk(v, depth + 1, True)

        walk(obs, 0, False)

    def learn_shape(self, tool: str, obs: Any) -> None:
        if not isinstance(obs, (dict, list)) or len(self.shaped) > 6:
            return
        sample = jdump(clip(obs, 90))[:3500]
        got = self.ask_json(SHAPE + "\n\nTOOL: " + tool + " | parameters: " + jdump(self.schema.get(tool) or {}) + "\n\nRESULT:\n" + sample, [FAST], 500)
        for k, v in (got or {}).items():
            if isinstance(v, str) and v and not self.keys.get(k):
                self.keys[k] = v
        self.log.append({"kind": "shape", "tool": tool, "keys": dict(self.keys)})
        # the first result was read before the keys were known: read it once more with them
        self.absorb("", obs)

    def price(self, ident: str) -> float | None:
        key = self.keys.get("price")
        return as_number((self.cat.get(ident, {}).get("row") or {}).get(key)) if key else None

    def sellable(self, ident: str) -> bool:
        if ident in self.gone:
            return False
        key = self.keys.get("stock")
        value = (self.cat.get(ident, {}).get("row") or {}).get(key) if key else None
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0)
                    or (isinstance(value, str) and value.strip().lower() in ("out of stock", "sold out", "unavailable", "no", "false")))

    def category_of(self, ident: str) -> str:
        key = self.keys.get("category")
        return str((self.cat.get(ident, {}).get("row") or {}).get(key) or "").strip() if key else ""

    def fits_label(self, label: str, product: Product) -> bool:
        mine = stems(product.aisle) | stems(product.label)
        theirs = stems(label)
        if not theirs:
            return False
        if theirs <= mine or plain(label) in (plain(product.aisle), plain(product.label)):
            return True
        return bool({w for w in theirs if len(w) >= 4} & {w for w in mine if len(w) >= 4})

    def record_line(self, ident: str, full: bool = True) -> str:
        entry = self.cat[ident]
        row = {k: v for k, v in entry["row"].items() if not isinstance(v, (dict, list))}
        head = jdump(clip(row, CLIP_FIELD))
        mark = "SOLD-OUT " if not self.sellable(ident) else ""
        if entry["facts"] and full:
            facts = "; ".join(f"{k}={v[:CLIP_FIELD]}" for k, v in list(entry["facts"].items())[:30])
            return f"OPENED {mark}{ident}: {head[:600]} | ATTRIBUTES: {facts}"
        return f"ROW {mark}{ident}: {head[:600]}"

    # ---------------------------------------------------------------- ledger
    def ledger(self) -> str:
        out = []
        many = len(self.products) > 1
        for n, p in enumerate(self.products, 1):
            reqs = [t + (" (value comes from another listing: open its page)" if pt else "") for t, pt in p.live()]
            tag = f"product {n}: {p.label}" if many else f"product: {p.label}"
            if p.aisle:
                tag += f" (store category: {p.aisle})"
            line = f"{tag}, count {p.count}; requirements: " + ("; ".join(reqs) if reqs else "only the kind of product")
            if p.indifferent:
                line += "; shopper does not mind: " + ", ".join(p.indifferent)
            out.append(line)
        lim = self.brief.get("limit") if isinstance(self.brief.get("limit"), dict) else {}
        if self.budget is not None:
            out.append(f"budget: at most {self.budget:g} {lim.get('unit') or ''} " + ("per unit" if lim.get("per") == "unit" else "for the whole order"))
        rank = self.brief.get("rank")
        out.append("rank rule: " + (f"{rank} {self.brief.get('rank_measure') or ''}".strip() if rank in ("cheapest", "most", "least") else "any listing meeting everything"))
        if self.brief.get("in_stock_only", True) is not False:
            out.append("in stock only")
        for key, label in (("instructions", "instruction"), ("nice_to_have", "wish (tie-breaker)")):
            for text in as_list(self.brief.get(key)):
                out.append(f"{label}: {text}")
        for note in self.notes[-5:]:
            out.append("news: " + note)
        return "\n".join("- " + line for line in out)

    # ---------------------------------------------------------------- reading the request
    def read_request(self) -> bool:
        lines = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {jdump(t['function'].get('parameters') or {})}" for t in self.tools]
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING OBSERVATION:\n" + jdump(clip(self.view["opening"], 1200))
        brief = self.ask_json(prompt, [FAST], 3500) or {}
        tools = brief.get("tool_map") if isinstance(brief.get("tool_map"), dict) else {}
        if not as_list(brief.get("products")) or not tools.get("buy"):
            brief = self.ask_json(prompt, [DEEP, FAST], 2500) or brief
        self.brief = brief
        tools = brief.get("tool_map") if isinstance(brief.get("tool_map"), dict) else {}
        self.role = {k: v for k, v in tools.items() if isinstance(v, str) and v in self.schema}
        for key in ("order_args", "decline_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, decline = self.role.get("buy"), self.role.get("decline")
        if buy and buy == decline and (not brief["order_args"] or not brief["decline_args"] or brief["order_args"] == brief["decline_args"]):
            got = self.ask_json("TASK:MODES\nThe store tool " + buy + " both checks out and ends without buying, depending on arguments. Schema: "
                                + jdump(self.schema[buy]) + '\nReply with ONE JSON object: {"order_args": only the arguments that choose checkout, "decline_args": only those that choose ending without buying}', [FAST, DEEP], 500) or {}
            if isinstance(got.get("order_args"), dict) and isinstance(got.get("decline_args"), dict) and got["order_args"] != got["decline_args"]:
                brief["order_args"], brief["decline_args"] = got["order_args"], got["decline_args"]
        for entry in as_list(brief.get("products")):
            if not isinstance(entry, dict):
                continue
            reqs = []
            for r in as_list(entry.get("requirements")):
                if isinstance(r, dict) and str(r.get("text") or "").strip():
                    reqs.append((str(r["text"]).strip(), r.get("points_elsewhere") is True))
                elif isinstance(r, str) and r.strip():
                    reqs.append((r.strip(), False))
            count = as_int(entry.get("count"))
            self.products.append(Product(str(entry.get("label") or "the product"), str(entry.get("aisle") or ""), count if count and count > 0 else 1, reqs))
        if not self.products:
            self.products = [Product("the requested product", "", 1, [])]
        lim = brief.get("limit") if isinstance(brief.get("limit"), dict) else {}
        amount = as_number(lim.get("amount"))
        self.budget = amount if amount and amount > 0 else None
        allowance = as_int(brief.get("message_allowance"))
        self.msg_limit = allowance if allowance is not None and allowance >= 0 else (3 if self.role.get("ask") else 0)
        if not self.role.get("ask"):
            self.msg_limit = 0
        steps = as_int(brief.get("step_allowance"))
        if steps is not None and 10 <= steps < self.cap:
            self.cap = steps
        for entry in as_list(brief.get("procedure")):
            if isinstance(entry, dict) and str(entry.get("do") or "").strip():
                before = str(entry.get("before") or "ordering").lower()
                self.proc.append({"do": str(entry["do"]).strip(), "before": before if before in ("choosing", "ordering", "declining") else "ordering",
                                  "messages": max(0, as_int(entry.get("messages")) or 0)})
        return bool(self.role.get("buy") or self.role.get("decline")) and bool(self.role.get("search"))

    # ---------------------------------------------------------------- stage 1: scout
    def scout_tools(self) -> list[dict[str, Any]]:
        barred = {self.role.get(r) for r in ("ask", "cart", "cart_view", "buy", "decline")} - {None}
        return [t for t in self.tools if t["function"]["name"] not in barred]

    def page_hint(self, tool: str, obs: Any) -> str:
        total, nxt, param = self.keys.get("total"), self.keys.get("next"), self.keys.get("next_param")
        if not isinstance(obs, dict) or not total or not nxt or not param or total not in obs:
            return ""
        shown = max((len(v) for v in obs.values() if isinstance(v, list)), default=0)
        count = as_int(obs.get(total))
        token = obs.get(nxt)
        if count is None or shown >= count or token in (None, "", 0):
            return ""
        return f"\n[more rows exist: {shown} of {count} shown; to see the next page pass {param}={jdump(token)}]"

    def role_note(self) -> str:
        parts = []
        if self.role.get("search"):
            parts.append(f"search: {self.role['search']}")
        if self.role.get("open"):
            parts.append(f"attribute pages (the only place a listing's attributes appear): {self.role['open']}")
        barred = {self.role.get(r) for r in ("ask", "cart", "cart_view", "buy", "decline", "search", "compare", "open")} - {None}
        light = [t["function"]["name"] for t in self.tools if t["function"]["name"] not in barred and self.single_param(t["function"]["name"])]
        if light:
            parts.append("single-listing tools that show description, price and stock but NOT attributes: " + ", ".join(light))
        if self.role.get("compare"):
            parts.append(f"several listings at once: {self.role['compare']}")
        return "; ".join(parts)

    def look(self, goal: str, step_cap: int, turns: int = SCOUT_TURNS) -> None:
        """One scouting leg: the cheap model browses read-only until it has what the goal asks, or a limit is hit."""
        tools = self.scout_tools()
        if not tools:
            return
        floor = self.step + max(0, step_cap)
        chat = [{"role": "system", "content": SCOUT_SYSTEM},
                {"role": "user", "content": "SHOPPER'S REQUEST:\n" + self.request + "\n\nWHAT IS KNOWN:\n" + self.ledger()
                 + "\n\nTOOLS BY JOB: " + self.role_note() + "\n\nYOUR GOAL THIS LEG:\n" + goal + f"\n\nYou may use about {max(0, step_cap)} steps."}]
        idle = 0
        for _ in range(turns):
            budget = min(floor, self.cap - self.reserve() - 1) - self.step
            if self.over or budget <= 0 or self.secs() < CLOCK_CLOSE_IN or self.broke:
                break
            msg = self.llm([FAST, BACKUP], chat, 2500, tools)
            if msg is None:
                break
            calls = [c for c in msg.get("tool_calls") or [] if isinstance(c, dict) and isinstance(c.get("function"), dict)]
            if not calls:
                idle += 1
                if idle >= 2 or msg.get("content"):
                    break
                chat.append({"role": "user", "content": "Call a tool, or reply with your summary if you have enough."})
                continue
            chat.append({"role": "assistant", "content": msg.get("content") or None, "tool_calls": [
                {"id": str(c.get("id") or f"c{len(chat)}_{i}"), "type": "function", "function": c["function"]} for i, c in enumerate(calls)]})
            for i, c in enumerate(calls):
                cid = str(c.get("id") or f"c{len(chat) - 1}_{i}")
                chat.append({"role": "tool", "tool_call_id": cid, "content": self.scout_call(c["function"], min(floor, self.cap - self.reserve() - 1))})
            chat.append({"role": "user", "content": f"[{self.left()} steps left in the whole task; about {max(0, min(floor, self.cap - self.reserve() - 1) - self.step)} for this leg; {len([1 for e in self.cat.values() if e['opened']])} listings opened so far]"})

    def scout_call(self, fn: dict[str, Any], ceiling: int) -> str:
        name = fn.get("name")
        allowed = {t["function"]["name"] for t in self.scout_tools()}
        if name not in allowed:
            return f"Skipped, no step spent: {name!r} is not a tool you may use. You may use: {', '.join(sorted(allowed))}."
        raw = fn.get("arguments")
        args = raw if isinstance(raw, dict) else (jparse(raw) if isinstance(raw, str) and raw.strip() else {})
        if args is None:
            return "Skipped, no step spent: the arguments were not one JSON object."
        args, faults = conform(args, self.schema[name])
        if faults:
            return "Skipped, no step spent: " + "; ".join(faults[:5])
        mark = name + jdump(args)
        if mark in self.cache:
            return "Skipped, no step spent: same call as before, same result:\n" + self.cache[mark]
        if self.step >= ceiling or self.over:
            return "Skipped, no step spent: the step allowance for this leg is spent. Reply with your summary."
        obs = self.act(name, args)
        if obs is None:
            return "The call failed. You may try again."
        text = jdump(clip(obs, 400))
        if len(text) > CLIP_RESULT:
            text = text[:CLIP_RESULT] + "...(cut)"
        text += self.page_hint(name, obs)
        self.cache[mark] = text
        self.note_free_text(obs)
        return text

    def note_free_text(self, obs: Any) -> None:
        """Prose inside a result (not a listing record) may carry a change from the shopper or the store."""
        lines: list[str] = []
        if isinstance(obs, dict) and not self.row_id(obs):
            for v in obs.values():
                if isinstance(v, str) and len(v.split()) >= 6:
                    lines.append(v)
        if lines and self.noted < 4:
            self.noted += 1
            self.digest("\n".join(lines))

    # ---------------------------------------------------------------- reading replies and notes
    def digest(self, text: str, product_hint: str = "") -> dict[str, Any]:
        self.digests += 1
        many = "\n".join(f"product {n}: {p.label}" for n, p in enumerate(self.products, 1))
        got = self.ask_json(DIGEST + "\n\nPRODUCTS:\n" + many + "\nREQUIREMENTS NOW:\n" + self.ledger() + (("\nCONTEXT: " + product_hint) if product_hint else "")
                            + "\n\nTEXT:\n" + text[:2500], [FAST, DEEP], 1200) or {}
        self.apply(got)
        return got

    def apply(self, got: dict[str, Any]) -> list[str]:
        told: list[str] = []

        def prod(entry: dict[str, Any]) -> Product:
            return self.products[min(max(1, as_int(entry.get("product")) or 1), len(self.products)) - 1]

        for e in as_list(got.get("dropped")):
            if isinstance(e, dict) and str(e.get("need") or "").strip():
                p, target = prod(e), str(e["need"]).strip()
                hit = [t for t, _ in p.live() if t == target or plain(t.split(":")[0]) == plain(target.split(":")[0])][:1]
                p.dropped += hit or [target]
                told.append("dropped " + target)
        for e in as_list(got.get("firm")):
            if isinstance(e, dict) and str(e.get("need") or "").strip():
                p, need = prod(e), str(e["need"]).strip()
                if need in p.dropped:
                    p.dropped.remove(need)
                if need not in [t for t, _ in p.reqs + p.learned]:
                    p.learned.append((need, e.get("points") is True))
                    told.append(need)
                    p.firm = True
        for e in as_list(got.get("indifferent")):
            if isinstance(e, dict) and str(e.get("property") or "").strip():
                prod(e).indifferent.append(str(e["property"]).strip())
        lim = as_number(got.get("limit"))
        if lim and lim > 0 and lim != self.budget:
            self.budget = lim
            told.append(f"budget now {lim:g}")
        for g in as_list(got.get("gone")):
            if isinstance(g, (str, int)) and str(g) in self.cat:
                self.gone.add(str(g))
                told.append(f"{g} unavailable")
        if got.get("kind") == "change" or told:
            summary = str(got.get("summary") or "; ".join(told)).strip()
            if summary:
                self.notes.append(summary)
        return told

    # ---------------------------------------------------------------- stage 2: interview
    def table(self, product: Product) -> list[tuple[str, int, int, list[str]]]:
        pool = [i for i, e in self.cat.items() if e["facts"]]
        near = [i for i in pool if self.category_of(i) and self.fits_label(self.category_of(i), product)]
        pool = near or pool
        stats: dict[str, dict[str, Any]] = {}
        for ident in pool:
            for k, v in self.cat[ident]["facts"].items():
                slot = stats.setdefault(k, {"n": 0, "vals": []})
                slot["n"] += 1
                if v.lower() not in slot["vals"]:
                    slot["vals"].append(v.lower())
        rows = []
        for k, slot in stats.items():
            n, distinct = slot["n"], len(slot["vals"])
            mean = sum(len(v) for v in slot["vals"]) / max(1, distinct)
            if n < 2 or distinct < 2 or (n >= 4 and distinct == n and mean > 12) or mean > 40:
                continue
            rows.append((k, n, distinct, [v[:40] for v in slot["vals"][:4]]))
        return sorted(rows, key=lambda r: (-r[1], r[2]))[:14]

    def interview(self) -> str:
        ask = self.role.get("ask")
        if not ask or self.free_msgs() <= 0 or self.secs() < CLOCK_CLOSE_IN + 40:
            return ""
        tables = {n: self.table(p) for n, p in enumerate(self.products, 1)}
        parts = []
        for n, p in enumerate(self.products, 1):
            rows = "\n".join(f"  - {k} (carried by {c}, {d} different values): {' | '.join(vals)}" for k, c, d, vals in tables[n]) or "  (no attributes seen)"
            parts.append(f"product {n}: {p.label}; settled by the request: {'; '.join(t for t, _ in p.live()) or 'nothing'}\n{rows}")
        want = self.free_msgs() + len(self.products) + 2
        first = [str(x) for x in as_list(self.brief.get("check_with_shopper"))]
        prompt = PLAN.replace("COUNT", str(want)) + "\n\nREQUEST:\n" + self.request + ("\n\nCONFIRM FIRST:\n" + "\n".join(first) if first else "") + "\n\nPRODUCTS AND ATTRIBUTES SEEN:\n" + "\n".join(parts)
        plan = self.ask_json(prompt, [DEEP, FAST], 1500) or {}
        queues: dict[int, list[str]] = {n: [] for n in tables}
        for e in as_list(plan.get("ask")):
            if isinstance(e, dict):
                n = min(max(1, as_int(e.get("product")) or 1), len(self.products))
                label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(e.get("label") or ""))[0].strip().rstrip("?.")
                if label and len(label.split()) <= 5 and all(plain(label) != plain(q) for q in queues[n]):
                    queues[n].append(label)
        for n, rows in tables.items():
            # keep every product supplied with candidates: the table's own rows fill what the plan left out
            for k, *_ in rows:
                if all(plain(k) != plain(q) for q in queues[n]) and not any(stems(k) <= stems(t) for t, _ in self.products[n - 1].live()):
                    queues[n].append(k)
        heard: list[str] = []
        turn = 0
        while self.free_msgs() > 0 and not self.over and self.secs() > CLOCK_CLOSE_IN + 40 and self.left() > self.reserve() + 2:
            pick = self.next_question(queues, turn)
            if pick is None:
                break
            n, label = pick
            turn = n
            product = self.products[n - 1]
            product.asked.append(label)
            text = f"Which {label} would you like for the {product.label}?"
            reply = self.send(text)
            if reply is None:
                break
            got = self.digest(f"Question: {text}\nAnswer: {reply}", f"the question was about product {n} ({product.label}), property {label}")
            for e in as_list(got.get("firm")):
                if isinstance(e, dict):
                    heard.append(f"product {n}: {e.get('need')}")
        if not self.msg_used:
            return ""
        return "; ".join(heard) or "no new firm preferences"

    def next_question(self, queues: dict[int, list[str]], last: int) -> tuple[int, str] | None:
        """Round robin over products that have not yet revealed a firm preference."""
        for n, q in queues.items():
            p = self.products[n - 1]
            q[:] = [lab for lab in q if not any(stems(lab) <= stems(a) or stems(a) <= stems(lab) for a in p.asked)
                    and not any(stems(t.split(":")[0]) <= stems(lab) for t, _ in p.learned)]
        live = [n for n, q in queues.items() if q]
        if not live:
            return None
        pool = [n for n in live if not self.products[n - 1].firm]
        if not pool:
            return None            # every product already gave its hidden preference: more questions only cost steps
        order = sorted(pool, key=lambda n: (n <= last, n))
        n = order[0]
        return n, queues[n].pop(0)

    def send(self, text: str) -> str | None:
        ask = self.role.get("ask")
        schema = self.schema.get(ask or "") or {}
        if not ask or self.msg_used >= self.msg_limit or self.left() <= 1:
            return None
        props = schema.get("properties") or {}
        target = next((k for k in schema.get("required") or [] if isinstance(props.get(k), dict) and props[k].get("type") == "string"), None)
        target = target or next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v), None)
        if not target:
            return None
        args, faults = conform({target: text}, schema)
        if faults:
            obs = self.tool_call(ask, "send this message to the shopper: " + text)
        else:
            obs = self.act(ask, args)
        if obs is None:
            return None
        self.msg_used += 1
        words = self.spoken(obs)
        refused = isinstance(obs, dict) and bool(obs.get("error"))
        self.log.append({"kind": "ask", "q": text, "a": words[:300]})
        if refused:
            self.msg_used -= 1
            self.msg_limit = self.msg_used
            return None
        return words

    @staticmethod
    def spoken(obs: Any) -> str:
        if isinstance(obs, str):
            return obs
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, str) and len(node.split()) >= 2:
                found.append(node)
            elif isinstance(node, dict):
                for v in node.values():
                    walk(v, depth + 1)
            elif isinstance(node, list):
                for v in node:
                    walk(v, depth + 1)

        walk(obs, 0)
        return " ".join(found)[:800] if found else jdump(obs)[:400]

    # ---------------------------------------------------------------- stage 3: judge
    def evidence(self) -> str:
        opened = [i for i, e in self.cat.items() if e["facts"]]
        rows = [i for i, e in self.cat.items() if not e["facts"]]
        if self.brief.get("rank") == "cheapest":
            key = lambda i: (self.price(i) is None, self.price(i) or 0.0)
            opened.sort(key=key)
            rows.sort(key=key)
        else:
            opened.reverse()
            rows.reverse()
        lines = [self.record_line(i) for i in opened[:26]] + [self.record_line(i, False) for i in rows[:24]]
        return "\n".join(lines) or "(no listings seen)"

    def judge(self) -> dict[str, Any]:
        cats = []
        key = self.keys.get("category")
        if key:
            labels = []
            for i in self.cat:
                lab = self.category_of(i)
                if lab and lab not in labels:
                    labels.append(lab)
            cats = ["store categories seen: " + ", ".join(labels[:25])]
        prompt = (JUDGE + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
                  + ("\n\n" + cats[0] if cats else "") + "\n\nRECORDS ON FILE (newest first unless the rank rule sorts them):\n" + self.evidence())
        got = self.ask_json(prompt, self.deciders(), 1600) or {}
        self.log.append({"kind": "judge", "got": clip(got, 200)})
        return got

    def open_listings(self, ids: list[str]) -> bool:
        """Read pages for listings seen only as rows: side by side when one call does it, else one by one."""
        ids = [i for i in ids if i in self.cat and not self.cat[i]["facts"]][:6]
        if not ids:
            return False
        before = sum(1 for e in self.cat.values() if e["facts"])
        compare, many, one = self.role.get("compare"), self.list_param(self.role.get("compare")), self.single_param(self.role.get("open"))
        if compare and many and len(ids) > 1 and self.left() > self.reserve() + 2:
            self.act(compare, conform({many: ids}, self.schema[compare])[0])
        if sum(1 for e in self.cat.values() if e["facts"]) == before and one:
            for ident in ids[:5]:
                if self.left() <= self.reserve() + 1:
                    break
                self.act(self.role["open"], {one: ident})
        return sum(1 for e in self.cat.values() if e["facts"]) > before

    # ---------------------------------------------------------------- stage 4: closer
    def procedure(self, stage: str, cart_note: str = "") -> str:
        """Do the request's own steps due at this stage; returns a reason to go back and rethink, or an empty string."""
        for p in self.proc:
            if p["before"] != stage or p["do"] in self.proc_done:
                continue
            self.proc_done.add(p["do"])
            if p["messages"] <= 0 or not self.role.get("ask") or self.msg_used >= self.msg_limit or self.left() <= 2:
                continue
            msg = self.llm([FAST, DEEP], [{"role": "user", "content": MESSAGE + "\n\nSTEP: " + p["do"] + "\nREQUEST:\n" + self.request
                                           + ("\nCART: " + cart_note if cart_note else "")}], 300)
            text = str((msg or {}).get("content") or "").strip() or p["do"]
            reply = self.send(text)
            if reply is None:
                continue
            got = self.digest(f"Message sent: {text}\nShopper's reply: {reply}", "the shopper answering a step the agent had to do")
            if got.get("approved") is False or got.get("kind") == "change" or got.get("firm"):
                self.proc_done.discard(p["do"])
                return got.get("summary") or "the shopper did not approve"
        return ""

    def stock_probe(self) -> str | None:
        """The single-listing tool that shows live stock: one seen to carry the stock field, else another single-id
        read tool next to the attributes tool, else the attributes tool itself."""
        for t in self.stock_tools:
            if self.single_param(t):
                return t
        barred = {self.role.get(r) for r in ("ask", "cart", "cart_view", "buy", "decline", "search", "compare")} - {None}
        other = [t["function"]["name"] for t in self.tools if t["function"]["name"] not in barred
                 and t["function"]["name"] != self.role.get("open") and self.single_param(t["function"]["name"])]
        return self.role.get("offer") or (other[0] if other else None) or self.role.get("open")

    def refresh(self, ids: list[str]) -> list[str]:
        """Re-read each chosen listing just before ordering; returns ids whose price or stock moved."""
        tool = self.stock_probe()
        one = self.single_param(tool)
        moved = []
        if not tool or not one or self.left() < len(ids) + self.reserve() - 1:
            return moved
        for ident in ids:
            before = (self.price(ident), self.sellable(ident))
            self.act(tool, {one: ident})
            if (self.price(ident), self.sellable(ident)) != before:
                moved.append(ident)
        return moved

    def candidates(self, product: Product) -> list[str]:
        """Sellable listings of this product's category that were seen only as rows, within budget."""
        rows = [i for i, e in self.cat.items() if not e["facts"] and self.sellable(i)
                and (self.budget is None or self.price(i) is None or self.price(i) <= self.budget)]
        near = [i for i in rows if self.category_of(i) and self.fits_label(self.category_of(i), product)]
        pool = near or ([] if self.keys.get("category") else rows)
        if self.brief.get("rank") == "cheapest":
            pool.sort(key=lambda i: (self.price(i) is None, self.price(i) or 0.0))
        return pool

    def fill_facts(self, room: int) -> int:
        """Code reads the attribute pages of promising row-only listings (side by side when one call does it)."""
        tool = self.role.get("open")
        one = self.single_param(tool)
        many_tool, many = self.role.get("compare"), self.list_param(self.role.get("compare"))
        room = min(room, self.left() - self.reserve() - 2)
        if room <= 0 or not (one or many):
            return 0
        lanes = [self.candidates(p) for p in self.products]
        picks: list[str] = []
        for k in range(room):
            for lane in lanes:
                if k < len(lane) and lane[k] not in picks:
                    picks.append(lane[k])
        picks = picks[:room]
        before = sum(1 for e in self.cat.values() if e["facts"])
        if many_tool and many and len(picks) > 1 and self.role.get("compare") != tool:
            self.act(many_tool, conform({many: picks[:6]}, self.schema[many_tool])[0])
            picks = [i for i in picks if not self.cat[i]["facts"]]
        if one:
            for ident in picks:
                if self.left() <= self.reserve() + 2 or self.over:
                    break
                if not self.cat[ident]["facts"]:
                    self.act(tool, {one: ident})
        return sum(1 for e in self.cat.values() if e["facts"]) - before

    def build_cart(self, lines: list[tuple[str, int]]) -> bool:
        cart = self.role.get("cart")
        view = self.role.get("cart_view")
        if not cart:
            return False
        desired = "; ".join(f"{i} x{c}" for i, c in lines)
        for attempt in range(2):
            current = ""
            if not self.cart_calls_made and attempt == 0:
                current = "[]  (nothing has been added yet)"
            elif self.cart_known is not None and attempt == 0:
                current = "believed from the agent's own earlier calls: " + "; ".join(f"{i} x{c}" for i, c in self.cart_known)
            elif view and self.left() > 3:
                obs = self.tool_call(view, "show the cart")
                current = jdump(clip(obs, 200))
                self.cart_text = current
            note = "" if attempt == 0 else " Previous attempt left problem: " + self.cart_problem
            plan = self.ask_json(CART_PLAN + "\n\nCART TOOL: " + cart + " | schema: " + jdump(self.schema[cart]) + "\nCURRENT CART: " + (current or "(unknown; assume empty)")
                                 + "\nDESIRED: " + desired + note, [FAST, DEEP], 1200) or {}
            calls = [c for c in as_list(plan.get("calls")) if isinstance(c, dict)][:6]
            for c in calls:
                if self.left() <= 2 or self.over:
                    break
                args, faults = conform(c.get("args") if isinstance(c.get("args"), dict) else {}, self.schema[cart])
                if faults:
                    continue
                self.cart_calls_made = True
                self.act(cart, args)
            if view and self.left() > self.reserve() - 7:
                obs = self.tool_call(view, "show the cart")
                self.cart_text = jdump(clip(obs, 200))
                if self.cart_matches(lines):
                    self.cart_known = list(lines)
                    return True
            else:
                self.cart_known = list(lines)
                return True
        self.cart_known = None
        return False

    def cart_matches(self, lines: list[tuple[str, int]]) -> bool:
        total = 0.0
        known = True
        for i, c in lines:
            p = self.price(i)
            known = known and p is not None
            total += (p or 0.0) * c
        got = self.ask_json(CART_CHECK + "\n\nDESIRED: " + "; ".join(f"{i} x{c}" for i, c in lines) + "\nBUDGET: " + (f"{self.budget:g}" if self.budget else "none")
                            + ("\nCOMPUTED TOTAL FROM LISTING PRICES: " + f"{total:g}" if known else "") + "\nCART:\n" + self.cart_text[:3000], [FAST, DEEP], 400) or {}
        self.cart_problem = str(got.get("problem") or "")
        return got.get("ok") is not False

    def checkout(self, lines: list[tuple[str, int]] | None = None) -> bool:
        buy = self.role.get("buy")
        if not buy:
            return False
        obs = self.tool_call(buy, "check out the cart now", self.brief.get("order_args") or {})
        if obs is None or self.over:
            return self.over
        self.refusals += 1
        got = self.digest(jdump(clip(obs, 400)), "the store's reply to an attempted checkout, which did not finish the episode")
        ids = [i for i, _ in lines or []]
        named = [g for g in as_list(got.get("gone")) if str(g) in self.cat]
        if got.get("sold_out") is True and not named and ids:
            # "your pick is gone" without a name: a lone line is that pick, several lines are re-read to see which
            if len(ids) == 1:
                self.gone.add(ids[0])
            else:
                self.refresh(ids)
            self.notes.append("a chosen listing sold out at checkout: " + ", ".join(i for i in ids if i in self.gone or len(ids) > 1))
        return False

    def decline(self) -> None:
        end = self.role.get("decline")
        if not end or self.over or self.left() <= 0:
            return
        quiet = self.brief.get("no_alternatives") is not False
        self.procedure("declining")
        if self.over:
            return
        note = "Keep any text short; name no listing or alternative." if quiet else "Keep any text short."
        self.tool_call(end, "end the episode without buying anything, because no listing meets every requirement", self.brief.get("decline_args") or {}, note)

    def lines_from(self, verdict: dict[str, Any]) -> list[tuple[str, int]]:
        lines: list[tuple[str, int]] = []
        seen: set[int] = set()
        for e in as_list(verdict.get("lines")):
            if not isinstance(e, dict):
                continue
            ident = str(e.get("id") or "").strip()
            n = min(max(1, as_int(e.get("product")) or 1), len(self.products))
            if ident in self.cat and self.sellable(ident) and n not in seen:
                seen.add(n)
                lines.append((ident, self.products[n - 1].count))
        return lines if len(seen) >= len(self.products) else []

    # ---------------------------------------------------------------- the run
    def go(self) -> list[dict[str, Any]]:
        if self.over:
            return self.log
        if not self.read_request():
            self.fallback()
            return self.log
        opening = self.view.get("opening")
        if opening is not None:
            self.absorb("", opening)
        n = len(self.products)
        # stage 1: survey
        self.procedure("choosing")
        survey = min(8 + 3 * n, max(4, (self.cap - self.reserve()) // 2))
        self.look("Search for each wanted product in its category. Open the ATTRIBUTE pages (not just the description view) of about four promising listings of each, so the questions to the shopper can target attributes that really vary. Then stop.", survey)
        self.fill_facts(min(2 + 2 * n, 8, max(0, self.left() - self.reserve() - self.free_msgs() - 8)))
        # stage 2: interview
        heard = self.interview() if self.role.get("ask") else ""
        # stages 3 and 4, repeated when the store or the shopper pushes back
        pending_goal = "Using everything known (including new preferences from the shopper: " + (heard or "none") + "), find for each product the listings that meet ALL requirements, and read their attribute pages. " \
                       "Apply the rank rule fully (for cheapest: look at the whole fitting set, cheapest first; for most/least: compare every fitting listing). Resolve any requirement that takes its value from another listing by opening that listing's page chain."
        for rnd in range(4):
            if self.over:
                break
            if self.secs() > CLOCK_CLOSE_IN and self.left() > self.reserve() + 2:
                self.look(pending_goal, self.left() - self.reserve() - 1)
            self.fill_facts(min(8, max(0, self.left() - self.reserve() - 4)))
            verdict = self.judge()
            for _ in range(2):
                wanted = [str(i) for i in as_list(verdict.get("open")) if str(i) in self.cat]
                if not wanted or self.left() <= self.reserve() + 2 or not self.open_listings(wanted):
                    break
                verdict = self.judge()
            lines = self.lines_from(verdict)
            if not lines and verdict.get("decline") is True and rnd < 2 and self.left() > 8 and self.secs() > CLOCK_CLOSE_IN + 20:
                # most declines are mistakes: search harder (twice at most) before giving up
                self.deep_search = True
                pending_goal = ("Nothing that fits has been found yet, but the store usually has it. Search in ways not tried yet: the product's name plus ONE requirement word, "
                                "other names for the product, the whole store category paged through, then read the attribute pages of every listing of the right kind. " + str(verdict.get("search_more") or ""))
                continue
            self.deep_search = False
            if not lines and verdict.get("decline") is not True and verdict.get("lines") and rnd < 3 and self.left() > self.reserve() + 4:
                pending_goal = "The judge could not cover every product yet (" + str(verdict.get("why") or "")[:200] + "). Find and open the missing listings. " + str(verdict.get("search_more") or "")
                continue
            if not lines:
                if verdict.get("decline") is True or not self.verdict.get("lines"):
                    self.decline()
                    break
                lines = self.lines_from(self.verdict)
            unread = [i for i, _ in lines if not self.cat[i]["facts"]]
            if unread and self.left() > self.reserve() + len(unread) + 1 and self.open_listings(unread):
                # never order a listing judged from a description alone: read its attributes, then judge again
                again = self.judge()
                redo = self.lines_from(again)
                if redo:
                    verdict, lines = again, redo
            self.verdict = verdict if verdict.get("lines") else self.verdict
            moved = self.refresh([i for i, _ in lines])
            if moved:
                self.notes.append("listings changed or sold out on re-reading: " + ", ".join(moved))
                pending_goal = "Listings " + ", ".join(moved) + " changed price or stock. Re-check the fitting set and find replacements where needed."
                continue
            why = self.procedure("ordering", "; ".join(f"{i} x{c}" for i, c in lines))
            if why:
                self.notes.append(why)
                pending_goal = "The shopper's reply changed the task: " + why + ". Find listings that meet the updated requirements."
                continue
            if not self.build_cart(lines):
                self.log.append({"kind": "cart_unsure", "problem": self.cart_problem})
            if self.checkout(lines):
                break
            pending_goal = "The store did not accept the order. Latest news: " + (self.notes[-1] if self.notes else "unknown") + ". Adjust to the updated requirements and find listings that satisfy them."
        if not self.over:
            self.finish_anyway()
        self.log.append({"kind": "end", "finished": self.over, "steps": self.step, "cost": round(self.spent, 5)})
        return self.log

    def finish_anyway(self) -> None:
        """Out of rounds, steps or time: order what the judge last chose, else end without buying."""
        lines = self.lines_from(self.verdict) if self.verdict else []
        if lines and self.left() >= 2:
            if self.left() >= 3 and self.role.get("cart"):
                self.build_cart(lines)
            if not self.over and self.checkout(lines):
                return
        end = self.role.get("decline")
        if not self.over and self.left() >= 1 and end:
            self.tool_call(end, "end the episode without buying anything", self.brief.get("decline_args") or {}, "Keep any text short.")

    def fallback(self) -> None:
        """The request could not be read into roles: let the cheap model drive every tool directly, reference style."""
        chat = [{"role": "system", "content": "Serve the shopper with the supplied tools. Reply with exactly one tool call each time; message the shopper only when a tool exists for it; finish by ordering the right items or declining."},
                {"role": "user", "content": self.request}]
        idle = 0
        while not self.over and self.left() > 0 and self.secs() > CLOCK_NO_MODEL + 5 and not self.broke:
            msg = self.llm([FAST, DEEP], chat, 2500, self.tools)
            if msg is None:
                break
            calls = [c for c in msg.get("tool_calls") or [] if isinstance(c, dict) and isinstance(c.get("function"), dict)]
            if not calls:
                idle += 1
                if idle > 2:
                    break
                chat += [{"role": "assistant", "content": msg.get("content") or ""}, {"role": "user", "content": "Choose a tool."}]
                continue
            idle = 0
            chat.append({"role": "assistant", "content": msg.get("content") or None, "tool_calls": [
                {"id": str(c.get("id") or f"f{len(chat)}_{i}"), "type": "function", "function": c["function"]} for i, c in enumerate(calls)]})
            for i, c in enumerate(calls):
                fn = c["function"]
                raw = fn.get("arguments")
                args = raw if isinstance(raw, dict) else (jparse(raw) or {})
                obs = self.act(str(fn.get("name")), args) if not self.over and self.left() > 0 else {"error": "no steps left"}
                chat.append({"role": "tool", "tool_call_id": str(c.get("id") or f"f{len(chat) - 1}_{i}"), "content": jdump(clip(obs, 400))[:CLIP_RESULT]})


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        run = Buyer(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:300]}]
    try:
        return run.go()
    except Exception as exc:
        run.log.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:300]})
        try:
            if not run.over:
                run.finish_anyway()
        except Exception as again:
            run.log.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:300]})
        return run.log


__all__ = ["agent_main"]
