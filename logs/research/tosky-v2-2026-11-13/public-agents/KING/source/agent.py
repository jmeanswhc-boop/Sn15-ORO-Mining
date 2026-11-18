"""k2: a shopping agent for ORO daily worlds.

Every world brings its own catalog, tools and wording, so nothing here depends on a tool's name, an argument's name
or a phrase of the practice world. The agent:
  - reads the world's tools and works out each one's role from its description and schema (search, item view,
    property sheet, cart change, cart review, shopper message, checkout, decline), with a model check of the guess;
  - reads the request once with a model into a plan: what is wanted, the stated requirements, the price cap, the
    question and step budgets, the stock and decline rules, any procedure, what ranks the options, and the
    reasons that may hide a need;
  - then lets a strong tool-calling model act, one environment step per call, while code keeps the contract:
    consecutive steps with identical re-sends only, arguments fitted to each schema before sending, no question
    past the budget, steps and time held back for the finish, every carted item inspected before checkout, and
    refused checkouts or declines carried on from;
  - finishes by itself (checkout of a verified cart, else a decline) when the steps or the time run short.
"""
from __future__ import annotations

import copy
import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

# tool-calling models (OpenRouter ids), tried in order; the first that answers is kept for the episode
MODELS = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "google/gemini-3.8-flash"]
JUDGES = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "google/gemini-3.8-flash"]
# reading the request and planning the questions decide most of an episode: a stronger model, once each per task
PLANNERS = ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5", "openai/gpt-6-luna"]
TIME_LIMIT = 300.0           # seconds a task may take on the network
FINISH_BY = 200.0            # past this, the agent finishes by itself (a model call may still take up to a minute)
STEP_RESERVE = 4             # steps kept back for cart, checkout and one more checkout attempt (fewer on a small budget)
STEP_RESERVE_LONG = 6        # in worlds of 20 steps or more: an order and one refused order's repair (look, order again)
ERRORS_IN_A_ROW = 5          # world errors in a row after which the agent finishes with what it knows
OBS_CHARS = 1900             # how much of one observation the model sees
OLD_CHARS = 260              # how much of an older observation stays in the conversation
KEEP_RECENT = 4              # the latest observations kept whole
SWEEP_PAGES = 4             # more pages of the narrow search read at most when the cheapest (or an affordable) fit is wanted
SWEEP_ROOM = 14             # steps a sweep leaves past the reserve: opening candidates, questions, the order and one repair
NEXT_KEY = r"next|further|more|cursor|token|continu|following"   # how a search result names where its next page starts
SPEND_LIMIT = 0.05           # the episode's own model spend (USD) after which it finishes by itself; a race run's key stops at about USD 5
CATEGORY_FIELD = r"categ|depart|aisle|section"   # how a result row or a search argument names the store category
FIELD_GUIDE = """You read a few result rows from an online store's product search. Say which key of a row holds each of
these, using only keys that appear in the rows: "id" (the product's unique id, the value other tools take), "title" (its
name or heading), "price", "stock" (whether, or how many, are in stock), "category" (the store's category, department or
shelf label). Use null for any the rows do not have. Answer as JSON only:
{"id": ..., "title": ..., "price": ..., "stock": ..., "category": ...}"""
MAX_TOKENS = 900

_proxy = ProxyClient(timeout=60, max_retries=1)          # model calls
_env = ProxyClient(timeout=15, max_retries=3)            # environment calls: rate limits get backed-off retries


# ---------------------------------------------------------------- model access
class Reader:
    """Chat completions through the proxy, over the provider's models in order."""

    def __init__(self, models: list[str] | None = None) -> None:
        self.models = list(models or MODELS)
        self.calls = 0
        self.spent = 0.0
        self.deadline: float | None = None     # time.monotonic() past which no model call may still be running

    def client(self) -> Any:
        """The model client for one call, its timeout cut to the time left (a try and one retry must both fit),
        or None when too little time is left to start a call. The shared client is copied, never changed."""
        if self.deadline is None:
            return _proxy
        left = self.deadline - time.monotonic()
        if left < 6:
            return None
        if not isinstance(getattr(_proxy, "timeout", None), (int, float)):
            return _proxy                      # a stand-in client without a timeout setting
        one = copy.copy(_proxy)
        one.timeout = int(max(3, min(60, (left - 2) / 2)))
        return one

    def chat(self, messages: list[dict], tools: list[dict] | None = None, force: str | None = None,
             max_tokens: int = MAX_TOKENS, json_out: bool = False) -> dict | None:
        for model in list(self.models):
            body: dict[str, Any] = {"model": model, "messages": cached(messages) if tools and model.startswith("anthropic/") else messages,
                                    "max_tokens": max_tokens, "temperature": 0}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = {"type": "function", "function": {"name": force}} if force else "auto"
            client = self.client()
            if client is None:
                return None                            # the task's time is nearly up: no call that could outlast it
            self.calls += 1
            reply = client.post("/inference/chat/completions", json_data=body)
            if isinstance(reply, dict):
                try:
                    self.spent += float(((reply.get("usage") or {}).get("cost")) or 0.0)
                except (TypeError, ValueError):
                    pass
            message = ((reply or {}).get("choices") or [{}])[0].get("message") if isinstance(reply, dict) else None
            if message and not (message.get("content") or message.get("tool_calls")) \
                    and ((reply.get("choices") or [{}])[0].get("finish_reason") == "length"):
                client = self.client()
                if client is None:
                    return None
                self.calls += 1
                reply = client.post("/inference/chat/completions", json_data=dict(body, max_tokens=max_tokens * 2))
                message = ((reply or {}).get("choices") or [{}])[0].get("message") if isinstance(reply, dict) else None
            if message:
                if self.models[0] != model:
                    self.models.remove(model)
                    self.models.insert(0, model)      # keep the model that answered first in line
                return message
        return None

    def json(self, system: str, user: str, max_tokens: int = 1200) -> dict:
        message = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                            max_tokens=max_tokens, json_out=True)
        return loads_loose(str((message or {}).get("content") or ""))


def cached(messages: list[dict]) -> list[dict]:
    """The messages with prompt-cache marks: one after the system text (tools and system stay the same for the whole
    episode) and one on the latest message, so the next call reads everything before it from the cache."""
    out = [dict(m) for m in messages]
    marks = [i for i, m in enumerate(out) if m.get("role") == "system"][:1]
    last = next((i for i in range(len(out) - 1, -1, -1) if isinstance(out[i].get("content"), str) and out[i]["content"]), None)
    if last is not None and last not in marks:
        marks.append(last)
    for i in marks:
        if isinstance(out[i].get("content"), str) and out[i]["content"]:
            out[i]["content"] = [{"type": "text", "text": out[i]["content"], "cache_control": {"type": "ephemeral"}}]
    return out


def loads_loose(text: str) -> dict:
    """The JSON object in a reply: the last complete one when the model wrote several (a reply that corrects itself);
    a reply cut off by the token limit keeps every field it completed."""
    start = text.find("{")
    if start < 0:
        return {}
    decoder, found, at = json.JSONDecoder(), None, start
    while at >= 0:
        try:
            value, end = decoder.raw_decode(text, at)
        except ValueError:
            break
        if isinstance(value, dict):
            found = value
        at = text.find("{", end)
    if found is not None:
        return found
    stack, cuts, quoted, escaped = [], [], False, False
    for i, ch in enumerate(text[start:], start):
        if quoted:
            escaped, quoted = (not escaped and ch == "\\"), quoted and not (ch == '"' and not escaped)
            continue
        if ch == '"':
            quoted = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
        elif ch == ",":
            cuts.append((i, "".join(reversed(stack))))
    for i, closers in reversed(cuts):
        try:
            value = json.loads(text[start:i] + closers)
            return value if isinstance(value, dict) else {}
        except ValueError:
            continue
    return {}


# ---------------------------------------------------------------- the world's tools
ROLES = ("finder", "item_page", "properties", "cart_change", "cart_review", "ask_shopper", "checkout", "decline", "other")
ROLE_GUIDE = """You label the tools of an online store's agent interface by role. Roles:
- finder: finds products (by words and/or by structured fields such as price, brand, category); returns many rows.
- item_page: shows one product in full by its id.
- properties: returns the property / attribute / specification list of one product by its id.
- cart_change: adds a product to the cart or removes it (may also set a quantity).
- cart_review: shows what is in the cart now.
- ask_shopper: sends a question or message to the shopper (the customer); the shopper's reply comes back.
- checkout: buys what is in the cart and ends the session (final).
- decline: ends the session without buying (gives up, declines, reports that nothing fits) (final).
- other: anything else.
Then say how this store's cart and finishing tools are used, with only tool names, argument names and values that
appear in the schemas given (read every enum's values; never invent one):
- "cart": the tool that adds a product to the cart or removes it: {"tool", "product" (its product-id argument),
  "quantity" (its quantity argument or null), "action" (its add-or-remove argument or null), "add" and "remove" (that
  argument's values that add and that remove, or null), "remove_tool" and "remove_product" (a separate tool that
  removes a product, and its product-id argument, when removing is not done by the same tool; else null)}.
- "order": the tool that buys what the cart holds: {"tool", "args": {argument: value} for every choice argument that
  must be set to buy}.
- "decline": the tool that ends without buying: {"tool", "args": {argument: value} for every choice argument that must
  be set to decline, "remark": its free-text argument for the explanation, or null}.
One tool may both buy and decline through a choice argument: then name it for both, with the two different values.
- "search": the finder tool: {"tool", "words" (its free-text argument), "mode" (an argument choosing how it searches, or
  null) with "word_mode" (that argument's value for a free-text search) and "browse_mode" (its value for listing one
  category or shelf, or null), "category" (an argument taking a category or shelf label, or null), "page" (an argument
  for later results, or null) with "page_unit" ("page" when it counts pages, "offset" when it counts rows),
  "price_limit" (a highest-price argument, or null), "sort" (an ordering argument, or null) with "cheapest_first" (its
  value that lists the cheapest first, or null)}.
Answer as JSON: {"roles": {"<tool name>": "<role>", ...}, "cart": {...}, "order": {...}, "decline": {...},
"search": {...}} with every tool named once in roles."""


def schema_of(tool: dict) -> dict:
    return (tool.get("function") or {}).get("parameters") or {}


def guess_role(tool: dict) -> str:
    fn = tool.get("function") or {}
    text = f"{fn.get('name', '')} {fn.get('description', '')}".lower().replace("_", " ")
    props = schema_of(tool).get("properties") or {}
    enums = " ".join(str(v.get("enum")) for v in props.values() if isinstance(v, dict) and v.get("enum")).lower()
    final = bool(re.search(r"\bfinal\b|no more actions|ends? the (chat|session|conversation|task)|close", text))
    if final and re.search(r"no purchase|without (a )?(purchase|buying|order)|give up|decline|nothing (fits|found)|can'?t be met", text):
        return "decline"
    if final and re.search(r"check ?out|purchase|place (the |an )?order|\bbuy\b|settle|pay", text):
        return "checkout"
    if re.search(r"shopper|customer|buyer|client", text) and re.search(r"ask|question|message|write|send|pose|reply|contact", text):
        return "ask_shopper"
    if re.search(r"\b(cart|bag|basket|trolley)\b", text):
        return "cart_change" if (props or re.search(r"add|remove|modify|change|include|discard", text + enums)) else "cart_review"
    if re.search(r"search|find|look ?up|browse|query|filter", text) and len(props) >= 2:
        return "finder"
    if len([k for k in props]) == 1 and re.search(r"propert|attribute|spec|datasheet|traits|details? sheet|facts", text):
        return "properties"
    if len([k for k in props]) == 1 and re.search(r"single|one (article|product|item|listing)|in full|view|show|open|listing", text):
        return "item_page"
    return "other"


def classify(tools: list[dict], reader: Reader) -> tuple[dict[str, str], dict]:
    """Each tool's role, and how this store's cart and finishing tools are used (read from their schemas)."""
    roles = {(t.get("function") or {}).get("name"): guess_role(t) for t in tools}
    brief = [{"name": (t.get("function") or {}).get("name"), "description": (t.get("function") or {}).get("description"),
              "parameters": schema_of(t)} for t in tools]
    answer = reader.json(ROLE_GUIDE, json.dumps({"tools": brief, "first_guess": roles}, ensure_ascii=False), max_tokens=3000)
    named = answer.get("roles") if isinstance(answer.get("roles"), dict) else {}
    for name, role in named.items():
        if isinstance(name, str) and name in roles and isinstance(role, str) and role in ROLES:
            roles[name] = role
    return roles, interface_map(tools, answer)


def interface_map(tools: list[dict], answer: dict) -> dict:
    """The model's reading of how to add to and remove from the cart, order and decline, kept only where every tool,
    argument and value it names exists in the schemas."""
    by_name = {(t.get("function") or {}).get("name"): t for t in tools}

    def plain(value: Any) -> str | None:
        return value if isinstance(value, str) and value else None

    def props(name: str) -> dict:
        return schema_of(by_name[name]).get("properties") or {}

    out: dict = {}
    cart = answer.get("cart") if isinstance(answer.get("cart"), dict) else {}
    tool = plain(cart.get("tool"))
    if tool in by_name:
        p = props(tool)
        product, quantity, action = plain(cart.get("product")), plain(cart.get("quantity")), plain(cart.get("action"))
        add, remove = plain(cart.get("add")), plain(cart.get("remove"))
        enum = [o for o in ((p.get(action) or {}).get("enum") or [])] if action in p else []
        acts_ok = action is None or (action in p and (not enum or (add in enum and (remove is None or remove in enum))))
        if product in p and (quantity is None or quantity in p) and acts_ok:
            out["cart"] = {"tool": tool, "id": product, "qty": quantity, "act": action,
                           "add": add if action else None, "remove": remove if action else None}
            drop, drop_id = plain(cart.get("remove_tool")), plain(cart.get("remove_product"))
            if drop in by_name and drop != tool and drop_id in props(drop):
                out["cart"]["remove_tool"], out["cart"]["remove_id"] = drop, drop_id
    for purpose in ("order", "decline"):
        spec = answer.get(purpose) if isinstance(answer.get(purpose), dict) else {}
        tool = plain(spec.get("tool"))
        if tool not in by_name:
            continue
        p = props(tool)
        fixed = spec.get("args") if isinstance(spec.get("args"), dict) else {}
        if all(isinstance(k, str) and k in p and (not (p[k] or {}).get("enum") or v in p[k]["enum"]) for k, v in fixed.items()):
            remark = plain(spec.get("remark"))
            out[purpose] = {"tool": tool, "args": dict(fixed), "remark": remark if remark in p else None}
    search = answer.get("search") if isinstance(answer.get("search"), dict) else {}
    tool = plain(search.get("tool"))
    if tool in by_name:
        p = props(tool)

        def arg(key: str, kinds: tuple) -> str | None:
            name = plain(search.get(key))
            return name if name in p and (p[name] or {}).get("type") in kinds else None

        def value(arg_name: str | None, key: str) -> str | None:
            v = plain(search.get(key))
            return v if arg_name and v in ((p[arg_name] or {}).get("enum") or []) else None

        words, mode, sort = arg("words", ("string", "array")), arg("mode", ("string",)), arg("sort", ("string",))
        page = arg("page", ("integer", "number", "string"))
        if words and not (p[words] or {}).get("enum") and (mode is None or value(mode, "word_mode")):
            out["search"] = {"tool": tool, "words": words, "mode": mode, "word_mode": value(mode, "word_mode"),
                             "browse_mode": value(mode, "browse_mode"), "category": arg("category", ("string",)),
                             "page": page, "page_unit": "cursor" if page and (p[page] or {}).get("type") == "string"
                             else "offset" if search.get("page_unit") == "offset" else "page",
                             "price_limit": arg("price_limit", ("number", "integer")), "sort": sort,
                             "cheapest_first": value(sort, "cheapest_first")}
    # one tool that both buys and declines must be told apart by its values, or neither reading can be trusted
    if out.get("order") and out.get("decline") and out["order"]["tool"] == out["decline"]["tool"] \
            and out["order"]["args"] == out["decline"]["args"]:
        out.pop("order")
        out.pop("decline")
    return out


def schema_problems(tool: dict, args: dict) -> list[str]:
    """What keeps a call from fitting its tool's schema: missing required arguments, arguments the schema does not
    allow, values outside an enum, wrong basic types. A call with problems is never sent: it would use a step."""
    schema = schema_of(tool)
    props = schema.get("properties") or {}
    problems = [f"{k}: required" for k in schema.get("required") or [] if k not in args]
    kinds = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
    for key, value in args.items():
        spec = props.get(key)
        if not isinstance(spec, dict):
            if schema.get("additionalProperties") is False:
                problems.append(f"{key}: not an argument of this tool")
            continue
        if isinstance(value, list):
            low, high = spec.get("minItems"), spec.get("maxItems")
            if isinstance(low, int) and len(value) < low:
                problems.append(f"{key}: needs at least {low} items")
            if isinstance(high, int) and len(value) > high:
                problems.append(f"{key}: at most {high} items")
        if spec.get("enum") and value not in spec["enum"]:
            problems.append(f"{key}: must be one of {spec['enum']}")
        kind = spec.get("type")
        want = kinds.get(kind) if isinstance(kind, str) else None
        if want and (not isinstance(value, want) or (kind in ("integer", "number") and isinstance(value, bool))):
            problems.append(f"{key}: must be {kind}")
    return problems


def fit_args(tool: dict, args: Any) -> dict:
    """Make arguments fit the tool's schema before a step is spent on them: unknown keys dropped, strings cut to
    their length limits, numbers clamped, enums matched loosely."""
    if not isinstance(args, dict):
        args = {}
    schema = schema_of(tool)
    props = schema.get("properties") or {}
    out: dict[str, Any] = {}
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            if schema.get("additionalProperties") is False or props:
                continue
            out[key] = value
            continue
        kind = spec.get("type")
        if kind == "array" and not isinstance(value, list):
            value = [value]                            # one value for a list parameter: a one-item list
        if kind == "array" and isinstance(spec.get("maxItems"), int):
            value = value[: spec["maxItems"]]
        if spec.get("enum"):
            options = spec["enum"]
            if value not in options:
                low = str(value).lower()
                value = next((o for o in options if str(o).lower() == low), None) or next(
                    (o for o in options if low and (low in str(o).lower() or str(o).lower() in low)), None)
                if value is None:
                    continue
        if kind == "integer":
            try:
                value = int(round(float(value)))
            except (TypeError, ValueError):
                continue
        elif kind == "number":
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
        elif kind == "string":
            value = str(value)
            if spec.get("maxLength"):
                value = value[: int(spec["maxLength"])]
            if spec.get("minLength") and len(value) < int(spec["minLength"]):
                continue
        elif kind == "boolean":
            value = value if isinstance(value, bool) else str(value).lower() in ("true", "1", "yes")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in spec:
                value = max(value, spec["minimum"])
            if "maximum" in spec:
                value = min(value, spec["maximum"])
        out[key] = value
    return out


def cart_shape(tools: list[dict], roles: dict[str, str], iface: dict | None = None) -> dict:
    """How to add and remove with this world's cart tools: the tool(s), the product-id argument, the action argument
    and its add / remove values, and the quantity argument. The interface map's reading comes first; the schema-shape
    reading below is only the fallback, and a call it cannot complete is never sent (see cart_call)."""
    if iface and iface.get("cart"):
        entry = dict(iface["cart"])
        if entry.get("remove_tool"):               # a separate tool removes: never remove through the adding tool
            return {"add": entry, "remove": {"tool": entry["remove_tool"], "id": entry["remove_id"], "qty": None,
                                             "act": None, "add": None, "remove": None}}
        if entry.get("act") and not entry.get("remove"):
            return {"add": entry}                  # no known way to remove: the model removes, never a guess
        return {"add": entry, "remove": entry}
    shape: dict[str, Any] = {}
    for tool in tools:
        fn = tool.get("function") or {}
        if roles.get(fn.get("name")) != "cart_change":
            continue
        props = schema_of(tool).get("properties") or {}
        text = f"{fn.get('name', '')} {fn.get('description', '')}".lower()
        id_arg = next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and not v.get("enum")
                       and re.search(r"id|product|item|sku|target|chosen|article|listing", k.lower())), None) \
            or next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and not v.get("enum")), None)
        qty_arg = next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "integer"), None)
        act_arg, add_val, remove_val = None, None, None
        for k, v in props.items():
            options = [str(o) for o in (v.get("enum") or [])] if isinstance(v, dict) else []
            if options:
                add_val = next((o for o in options if re.search(r"add|include|put|insert|keep|plus", o.lower())), None)
                remove_val = next((o for o in options if re.search(r"remov|discard|delet|drop|take|minus", o.lower())), None)
                if add_val or remove_val:
                    act_arg = k
                    break
        entry = {"tool": fn.get("name"), "id": id_arg, "qty": qty_arg, "act": act_arg, "add": add_val, "remove": remove_val}
        if act_arg:
            shape.setdefault("add", entry)
            shape.setdefault("remove", entry)
        elif re.search(r"remov|discard|delet|drop|take out", text):
            shape.setdefault("remove", entry)
        else:
            shape.setdefault("add", entry)
    return shape


# ---------------------------------------------------------------- the request
PLAN_GUIDE = """You read one shopper's request to an online store assistant and write down, as JSON, everything the
assistant must respect. The request is written in the shopper's own voice; budgets and rules can be worded in any
way. Fields:
- "item": what kind of product is wanted, in plain words.
- "quantity": how many units of the product to buy (1 if not said); never the number of messages, steps or days.
  Something sold as one item, such as a pair of shoes or a set, is quantity 1.
- "requirements": list of {"property", "value"} the product must have, as stated.
- "price_cap": the highest acceptable price as a number in the request's own currency unit, or null; "price_unit": the unit word.
- "price_rule": how the price cap applies (per unit, total, each), in a few words, or "".
- "question_budget": how many messages/questions the assistant may send the shopper (a number), or null if not stated;
  "question_budget_quote": the exact words of the request that state it, or "".
- "step_budget": how many actions/moves/steps the assistant may take (a number), or null if not stated;
  "step_budget_quote": the exact words of the request that state it, or "".
- "stock_rule": what the request says about stock or availability, or "".
- "decline_rule": what the shopper wants when nothing fits (for example: say so and do not offer a substitute), or "".
- "procedure": a list of steps or checks the request says to follow before ordering, or [].
- "rank_by": what makes one acceptable product better than another if the request asks for the cheapest / best /
  most / largest of something, else "".
- "reasons": the reasons or situations the request gives for the purchase (who it is for, what it is used with,
  where), each in a few words.
- "likely_hidden": up to 3 product properties the reasons suggest the shopper may care about but did not state
  (for example a gift for someone with particular taste: colour; sending a parcel that is weighed: weight;
  filling a space: size or length; sockets or chargers: voltage; a room where it runs: power), most likely first.
- "ask_who_for": true when this kind of product is made differently for different people (by sex or age group) and
  the request does not say who will use it; else false.
- "notes": anything else that decides whether an order is correct.
Answer with the JSON object only."""


def read_plan(query: str, max_steps: int | None, reader: Reader) -> dict:
    plan = reader.json(PLAN_GUIDE, query, max_tokens=2600)
    numbers = set(re.findall(r"\d+(?:[.,]\d+)?", query))

    def stated(value: Any, quote: Any = "") -> float | None:
        """A number the request really states: written in it as digits, or read from words the request contains."""
        try:
            number = float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None
        written = {n.replace(",", "") for n in numbers}
        if any(abs(float(n) - number) < 1e-9 for n in written):
            return number
        quoted = str(quote or "").strip().lower()
        return number if len(quoted) >= 3 and quoted in query.lower() else None

    plan["question_budget"] = stated(plan.get("question_budget"), plan.get("question_budget_quote"))
    plan["step_budget"] = stated(plan.get("step_budget"), plan.get("step_budget_quote"))
    plan["price_cap"] = stated(plan.get("price_cap"))
    present = [int(v) for v in (plan["step_budget"], max_steps) if v]
    plan["step_budget"] = min(present) if present else 20
    return plan


# ---------------------------------------------------------------- the environment
class World:
    """One episode's environment session: consecutive steps, identical re-sends on a failed transport."""

    def __init__(self, binding: dict, max_steps: int | None) -> None:
        self.started = time.monotonic()
        self.binding = binding
        self.max_steps = max_steps              # None when the world does not say
        self.step = 0
        self.finished = False
        self.trace: list[dict] = []

    def left(self) -> int:
        return self.max_steps - self.step if self.max_steps else 10**6

    def call(self, name: str, args: dict) -> dict:
        if self.finished:
            return {"observation": {"error": "the episode has finished"}, "finished": True}
        self.step += 1
        envelope = {**self.binding, "step": self.step, "action": {"name": name, "args": args}}
        result = None
        for wait in (0.0, 2.0, 5.0):
            if wait and time.monotonic() - self.started > 230:
                break                                  # near the task's time limit: no more re-sends
            time.sleep(wait)
            result = _env.post("/environment/call", json_data=envelope)       # the same envelope each time
            if isinstance(result, dict) and "observation" in result:
                break
        if not isinstance(result, dict) or "observation" not in result:
            # the step may or may not have counted, so no later step can be numbered safely: stop here
            self.finished = True
            result = {"observation": {"error": "no response from the environment"}, "finished": True}
        if isinstance(result.get("step"), int) and not isinstance(result.get("step"), bool):
            self.step = result["step"]             # the world's own count of this step
        self.finished = self.finished or bool(result.get("finished"))
        self.trace.append({"step": self.step, "action": {"name": name, "args": args}, "result": result})
        return result


def compact(observation: Any, long: bool = False) -> str:
    """The observation as the model sees it: JSON, long texts trimmed, within a size limit. Inside lists (result
    rows) a long free text is dropped and other texts are cut short, so many rows fit in few tokens. The long form
    (for the judge) keeps more of a product's own texts."""
    def trim(value: Any, in_row: bool = False) -> Any:
        if isinstance(value, str):
            limit = 110 if in_row else (900 if long else 240)
            return value if len(value) <= limit else value[:limit] + "..."
        if isinstance(value, list):
            return [trim(v, True) for v in value[:9]]
        if isinstance(value, dict):
            return {k: trim(v, in_row) for k, v in value.items()
                    if not (in_row and isinstance(v, str) and len(v) > 160 and not re.search(r"title|name", str(k).lower()))}
        return value
    text = json.dumps(trim(observation), ensure_ascii=False, separators=(",", ":"))
    size = OBS_CHARS * 2 if long else OBS_CHARS
    return text if len(text) <= size else text[:size] + "...(cut)"


def property_map(observation: Any) -> dict[str, str]:
    """The listed properties in a property-list observation, with each key written as words ('fa.general.styles' ->
    'general styles'): a block named like a property list ('attributes', 'facts', 'specs') when the page has one,
    else the largest dict of plain values. Free texts (descriptions) and the listing's own fields (title, price,
    stock...) are not properties, whatever a store calls them."""
    best: dict = {}
    named: dict = {}

    def walk(value: Any) -> None:
        nonlocal best, named
        if isinstance(value, dict):
            for key, inner in value.items():
                if isinstance(inner, dict) and re.search(r"attribut|fact|spec|propert|detail|feature|trait|characteristic", str(key).lower()):
                    plain_inner = {k: v for k, v in inner.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
                    if len(plain_inner) > len(named):
                        named = plain_inner
            plain = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(plain) > len(best) and len(plain) >= 3:
                best = plain
            for v in value.values():
                walk(v)
    walk(observation)
    if len(named) >= 2:
        best = named
    found: dict[str, str] = {}
    for key, value in best.items():
        words = [w for w in re.split(r"[._\s]+", str(key).lower()) if w]
        while len(words) > 1 and len(words[0]) <= 2:
            words.pop(0)                       # short prefixes such as 'fa' or 'm'
        name = " ".join(words)
        if name and not re.search(r"(^|\s)(id|pid|sku|title|name|price|stock|options?|variants?)$", name) \
                and not (isinstance(value, str) and len(value) > 80):          # options list purchase variants; long texts describe
            text = str(value).lower()
            if name in found and text not in found[name]:
                found[name] += ", " + text             # one property under two keys ('dietary needs', 'dietary_needs'): keep both
            else:
                found.setdefault(name, text)
    return found


def listed_properties(observation: Any, role: str) -> list[dict[str, str]]:
    """The property list of each product an opened page shows. A property-list page is read whole; a full product
    page only by the dicts inside it (its own top-level fields are the name, price and the like); a page showing
    several products side by side (a list of them) one product at a time."""
    body = core(observation)
    rows = next((v for v in body.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), None) \
        if isinstance(body, dict) else None
    if rows is None and role == "properties":
        return [property_map(observation)]
    out = []
    for product in rows or [body]:
        if isinstance(product, dict):
            found = property_map({k: v for k, v in product.items() if isinstance(v, dict)})
            if found:
                out.append(found)
    return out


def product_properties(observation: Any, role: str, asked: list[str], id_key: str | None) -> dict[str, dict[str, str]]:
    """Each opened product's listed properties by its id: a page about one asked product is that product's; a page
    showing several gives each row to the id the row carries."""
    body = core(observation)
    rows = next((v for v in body.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), None) \
        if isinstance(body, dict) else None
    if rows is None:
        found = listed_properties(observation, role)
        return {asked[0]: found[0]} if len(asked) == 1 and len(found) == 1 and found[0] else {}
    out = {}
    for row in rows:
        own = one_id(row, ids_in(row, id_key), id_key)
        found = property_map({k: v for k, v in row.items() if isinstance(v, dict)}) or property_map(row)
        if own and found:
            out[own] = found
    return out


SPELLINGS = {"colour": "color", "colours": "colors", "grey": "gray", "jewellery": "jewelry"}
NOT_STATED = {"", "not specified", "none", "n a", "na", "other", "others", "default", "unknown"}


def term_words(text: Any) -> list[str]:
    return [SPELLINGS.get(w, w) for w in re.findall(r"[a-z0-9]+", str(text).lower().replace("_", " "))]


def stem_word(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "y"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def holds(value: Any, wanted: str) -> bool:
    """True when the wanted words appear in the value, in order (light stemming: embroidered holds embroidery)."""
    have = [stem_word(w) for w in term_words(value)]
    want = [stem_word(w) for w in term_words(wanted)]
    return bool(want) and any(have[i:i + len(want)] == want for i in range(len(have) - len(want) + 1))


EXCLUSION = re.compile(r"^\s*(?:not|no|non|never|without|avoid|except|excluding|exclude|anything but)\b-?\s*(.+)$")


def excluded_phrases(prop: str, value: str) -> list[str]:
    """The excluded words of a requirement that is wholly an exclusion ('not leather', 'no short sleeves', 'not black
    or navy'), one phrase per alternative; [] for anything else. A value mixing wanted and excluded parts ('formal,
    not casual') and availability or price entries are left to the judge."""
    if re.search(r"availab|stock|ship|shelf|price|budget|cost|quantity", prop.lower()):
        return []
    match = EXCLUSION.match(re.sub(r"\([^)]*\)", " ", value.lower()).strip())
    if not match or re.search(r"[,;]|\bbut\b|\bonly\b", match.group(1)):
        return []
    own = {stem_word(w) for w in term_words(prop)}
    phrases = []
    for part in re.split(r"\bor\b|\band\b|/", match.group(1)):
        words = [w for w in term_words(part) if stem_word(w) not in own and w not in ("a", "an", "the", "any")]
        if words:
            phrases.append(" ".join(words))
    return phrases if phrases and all(len(p.split()) <= 3 for p in phrases) else []


def search_note(tool: dict | None, observation: Any) -> str:
    """A note when a search matched more products than its pages can show."""
    observation = core(observation)
    if not isinstance(observation, dict):
        return ""
    rows = result_rows(observation) if any(isinstance(v, list) for v in observation.values()) else None
    count = next((v for k, v in observation.items() if isinstance(v, int) and not isinstance(v, bool)
                  and re.search(r"match|total|count|hits|found|number", str(k).lower())), None)
    if rows is None or count is None or not rows:
        return ""
    pages = next((int(p["maximum"]) for k, p in schema_of(tool or {}).get("properties", {}).items()
                  if re.search(r"page", k.lower()) and isinstance(p, dict) and isinstance(p.get("maximum"), int)), 1)
    reachable = len(rows) * max(1, pages)
    if count <= reachable:
        return ""
    return (f"\n[note] {count} matches, but only about {reachable} can ever be seen and they may not be ordered by "
            "relevance: narrow the search with more words (the item word plus required values; property values a "
            "listing states, such as a pattern or material, are often searchable) until the matches fit.")


def property_head(name: str, item: str = "") -> str:
    """The word that names a property: its last real word, past the item's own words ('skirt length' -> 'length')."""
    skip = {w[:5] for w in re.findall(r"[a-z]+", item.lower()) if len(w) >= 4} | {"gener", "type", "types", "kind", "class"}
    words = [w for w in re.findall(r"[a-z]+", name.lower()) if len(w) >= 3 and w[:5] not in skip]
    return words[-1] if words else ""


def names_property(text: str, name: str, item: str = "") -> bool:
    """True when a text names a property by its head word (stemmed to five letters)."""
    head = property_head(name, item)
    return bool(head) and head[:5] in {w[:5] for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3}


def is_fault(observation: Any) -> bool:
    """True when a tool result reports a failure, in any of the shapes stores use: an error or fault field, ok false,
    an outcome or status saying fault / error / failed, or an error code ('UNKNOWN_SKU', 'E12')."""
    body = core(observation) if isinstance(observation, dict) else observation
    for o in (observation, body):
        if not isinstance(o, dict):
            continue
        if o.get("error") or o.get("call_fault") or o.get("fault") or o.get("ok") is False or o.get("success") is False:
            return True
        if re.search(r"^(fault|error|failed|failure|invalid|rejected)$", str(o.get("outcome") or o.get("status") or "").lower()):
            return True
        if isinstance(o.get("code"), str) and re.search(r"unknown|invalid|not.?found|error|fault|^e\d+$", o["code"].lower()):
            return True
    return False


def notices(observation: Any) -> list[str]:
    """Lines the store or the shopper attach to a result (a refusal, a change of budget, stock or wishes): a list of
    sentences beside the result itself."""
    if not isinstance(observation, dict) or len(observation) < 2:
        return []
    found: list[str] = []
    wrapper = len(observation) <= 3                    # {result, [lines]}; a product page with bullet lists is not one
    for key, value in observation.items():
        named = re.search(r"refus|notice|alert|warn|news", str(key).lower())
        lines = (value if wrapper or named else []) if isinstance(value, list) else [value] if isinstance(value, str) and named else []
        if lines and all(isinstance(x, str) and " " in x.strip() for x in lines):
            found += [x for x in lines if x not in found]
    return found


def core(observation: Any) -> Any:
    """The result itself when notices came wrapped around it ({result, [notice lines]})."""
    if notices(observation):
        inner = [v for v in observation.values() if isinstance(v, dict)]
        if len(inner) == 1:
            return inner[0]
    return observation


def reply_text(observation: Any) -> Any:
    """A shopper's reply as text, also when notices came wrapped around it ({reply, [notice lines]})."""
    if isinstance(observation, dict):
        lines = notices(observation)
        return next((v for v in observation.values() if isinstance(v, str) and v not in lines), observation)
    return observation


def shown(observation: Any, long: bool = False) -> str:
    """The observation as the model sees it, any attached notices first so a long result never hides them."""
    lines = notices(observation)
    return "".join(f"[notice from the shopper or store] {x}\n" for x in lines) + compact(observation, long)


def stem(word: str) -> str:
    """A word without a plural ending, cut to five letters, for loose matching ('watches' ~ 'watch')."""
    for end in ("ches", "shes", "xes", "ies", "s"):
        if word.endswith(end) and len(word) > len(end) + 2:
            word = word[: -len(end)] + ("y" if end == "ies" else "ch" if end == "ches" else "sh" if end == "shes" else "x" if end == "xes" else "")
            break
    return word[:5]


QUERY_STOP = {"and", "or", "the", "to", "for", "of", "with", "a", "an", "in", "on", "at", "by"}
VALUE_FILLER = {"only", "must", "least", "more", "than", "less", "usual", "very", "really", "something", "one"}


def query_words(text: Any) -> list[str]:
    """Search words of a text: lower-case words of two or more letters, without joining words, in order, once each."""
    out: list[str] = []
    for w in re.findall(r"[a-z0-9]+", re.sub(r"\([^)]*\)", " ", str(text or "")).lower()):
        if len(w) >= 2 and w not in QUERY_STOP and w not in out:
            out.append(w)
    return out


def value_words(requirements: Any, base: list[str]) -> list[str]:
    """The words of stated requirement values worth searching with: short positive values, without the words that
    only name the property or restate the item; availability, price and purpose entries are left to the judge."""
    out: list[str] = []
    for r in requirements if isinstance(requirements, list) else []:
        if not isinstance(r, dict):
            continue
        prop = str(r.get("property") or "").lower()
        value = re.sub(r"\([^)]*\)", " ", str(r.get("value") or "")).lower()
        if re.search(r"availab|stock|ship|shelf|on hand|price|budget|cost|placement|quantity|amount", prop):
            continue
        # "lace-up only, no zips, no velcro": search with the positive part, leave the exclusions to the judge
        positive = [seg for seg in re.split(r"[,;]|\bbut\b", value) if seg.strip() and not re.match(r"\s*(not|no|never|without|avoid|except)\b", seg)]
        value = positive[0] if positive else value
        if re.match(r"\s*(not|no|never|without|avoid|except)\b", value) or len(query_words(value)) > 4:
            continue
        words = query_words(value)
        if len(words) < 2:
            words = [w for w in words if w not in set(query_words(prop))]
        if re.search(r"categ|type|style|kind|item", prop) and set(words) <= set(base):
            continue
        out += [w for w in words if w not in out and w not in base and w not in VALUE_FILLER]
    return out


def ids_in(value: Any, id_key: str | None = None) -> set[str]:
    """Product ids an observation mentions: values of the store's own id field (read from its results) and of keys
    that name an id."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (str, int)) and (key == id_key or re.search(r"(^|_)(id|pid|sku)$|_id$|^id", str(key).lower())):
                found.add(str(item))
            else:
                found |= ids_in(item, id_key)
    elif isinstance(value, list):
        for item in value:
            found |= ids_in(item, id_key)
    return found


def one_id(row: Any, found: set[str], id_key: str | None) -> str | None:
    """A row's own id: the one id it mentions, else (several id-like keys, such as a brand_id beside the sku) the value
    of the store's id field when that is among them."""
    if len(found) == 1:
        return next(iter(found))
    value = row.get(id_key) if id_key and isinstance(row, dict) else None
    return str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value) in found else None


def result_rows(result: Any) -> list:
    """The product rows of a result: its first list of objects, else its first list (a list of facet names before the rows
    is not the rows)."""
    if not isinstance(result, dict):
        return []
    lists = [v for v in result.values() if isinstance(v, list)]
    return next((v for v in lists if v and all(isinstance(x, dict) for x in v)), lists[0] if lists else [])


# ---------------------------------------------------------------- the episode
SYSTEM = """You are a careful shopping assistant working inside an online store through tools. Each tool call is one
step and steps are limited; questions to the shopper are limited too. Score: the task counts only if your finish is
exactly right: every stated requirement and every requirement the shopper holds back is met, the price and stock
rules hold, the cheapest or best option is chosen when the shopper asks for it, any stated procedure is followed,
and you decline (with the decline tool) when nothing in the store can meet the request, without offering a
substitute. Anything less scores nothing, and so does running out of steps.

How to work:
1. Questions: the shopper tells a held-back requirement only when asked about that property by name, and a missed
   one makes the order wrong. A shopper holds back only properties that listings state, so first open the property
   lists of a few candidates; the status line then names the properties they state. Questions within the budget
   cost nothing but a step: before choosing, use them on those properties, the ones the request hints at first and
   then the ones the candidates differ in. One property per message, named plainly; never exceed the budget; never
   ask about something already stated or already answered. A reply such as "I don't mind" means that property is free.
2. Search with product words (the kind of item plus required values), never with the shopper's chatty phrases.
   Results may not be ordered by relevance and only a few pages may be reachable, so make each search specific:
   combine the item word with the required values (a colour, a material, a pattern, a feature; keyword search often
   matches the property values a listing states, not only its title) until the matches fit in the reachable pages,
   and use structured filters (an exact category or brand as the store writes it, a price limit) when the tool
   offers them. Put the most specific words first (the required values, then the item word): some stores read only
   the first two or three words of a query. If a specific search returns nothing, drop one word and try again.
3. Before choosing, open the property list (or full view) of each serious candidate and check every requirement
   against the product's own properties, title and options. A requirement counts only when the product clearly
   states it (a longer value that contains the wanted words counts); a different value is a mismatch. Respect the
   price cap and stock.
4. If the shopper wants the cheapest/best/most of something, compare all acceptable candidates you found and pick
   that one.
5. Put exactly the chosen product (and the asked quantity) in the cart, remove anything else, then check out.
6. A checkout or decline can be refused (for example the item sold out, or the shopper drops a requirement).
   Read the refusal, update what you know (a dropped requirement no longer applies; a sold-out item can no longer
   be bought), and continue: pick another acceptable product, or decline if none. Things can also change while you
   work (price, stock): re-check the chosen product just before checkout when steps allow. A notice from the
   shopper or store can arrive with any result (a new budget, a dropped or added wish, a stock change): the latest
   notice wins over the request.
7. When nothing acceptable exists after a fair search, decline with a short plain remark that names no product
   and no product id (offering a substitute breaks the request).
Answer with tool calls only. You may send up to three searches or product views/property reads in one reply;
send a question, a cart change or a finish alone. Watch the status line after each result."""


JUDGE_GUIDE = """You check which product a shopping assistant should buy. You get the shopper's request, the shopper's
answers to the assistant's questions (these add requirements), any refusal lines from the store or the shopper (a
requirement may have been dropped, an item may have sold out, the budget may have changed), and the products the
assistant has opened, each with its own listed properties, title, price and stock.
Decide strictly:
- A product is acceptable only if it meets EVERY requirement: what the request states, what the shopper's answers
  add (minus anything a refusal line dropped), the price cap (and any lowered budget), the stock rule, and the kind
  of item asked for. A requirement counts as met only when the product's own properties, title or options clearly
  state it; a different stated value is a mismatch; silence is not a match for a stated requirement.
- An item a refusal said sold out or can no longer be ordered is not acceptable.
- The kind of item: when a STORE CATEGORY FOR THE ITEM is given (the store's category whose name matches how the
  request names the item), a product in another category is not the kind asked for, whatever its title says.
- Which text decides: the product's own listed properties for a requirement decide it. A listed value meets it when
  it is the wanted value, a list of several values that includes it, or a longer value that contains the wanted
  words ('light sky blue' meets blue, '80% cotton blend' meets cotton: the store counts these). The words themselves
  must appear: an abbreviation or code does not state them (a size list 'S, M, L, XL' does not state large; 'Large
  (L)' or 'extra large' does). If the
  product lists such a property and none of its values is or contains the wanted one, the requirement is not met,
  whatever the title says. Only when the product
  lists no property for it can its title, options or description show that it meets the requirement.
- Exclusions ('not leather', 'no short sleeves', 'anything but black'): met only when the product's own property for
  it states a value that does not contain the excluded words ('canvas' meets not leather; 'faux leather' does not); a
  product that does not state that property does not meet an exclusion.
- When several listed properties could speak to a requirement, the one named exactly like it decides ('material'
  over 'material filter' or 'material type').
- A request's own explanation of a requirement is not a further requirement: 'lace-up only, no zips, no velcro'
  asks for a lace-up closure, and a product whose closure is lace-up meets all of it.
- Options: a product's options (variant) value names only the variant the store shows first, such as one colour,
  size or set of several on offer; it never conflicts with, limits or overrides the product's other listed
  properties. Ignore it for a requirement that another listed property states; use it only when nothing else in the
  product speaks to that requirement.
- Consistency: a product whose own texts disagree about a required property (its title, another listed property or
  its description names a different value for it; its options value does not count) is weak. Choose among products
  whose texts all agree with every requirement; choose a weak one only when no consistent product is acceptable.
  This never outweighs price when the request asks for the cheapest: there the listed properties decide, and the
  cheapest acceptable product is the choice, consistent or not.
- If the request asks for the cheapest (or the most / best of something), choose that one among the acceptable
  (the consistent ones first). Otherwise choose the cheapest acceptable one: prices rise and budgets are cut while
  the shopper waits, and the cheapest keeps the most room.
Answer as JSON only: {"choice": "<product id or null>", "acceptable": ["<ids>"], "rejected": {"<id>": "<the failed
requirement in at most 8 words>"}, "missing": "<if none is acceptable: what to look for next, else empty>"}"""


PLAUSIBLE_GUIDE = """You read search-result rows for a shopper's request and pick the rows that could be the product asked for:
the right kind of item, with nothing in the title contradicting a stated requirement. Answer as JSON only:
{"plausible": ["<ids>"]}"""


PLANNER_GUIDE = """A shopper asked a store assistant for a product. The shopper holds back one or two requirements and reveals each
one only when a message asks about that property by name. The held-back requirements are among this kind of product's
main filters: the properties a store lets shoppers filter this kind of product by, which most of its listings state and
whose values differ between listings. Below are the request and the properties the opened listings state (name, how
many of those listings state it, example values).
The store writes each held-back requirement into the request's story as a hint that never names the property or the
value: who the item is for or who will wear or use it (a relative, a partner, a child, a present for someone) points
to gender or age group; how it must look to others, in photos or next to things they own (an event, an outfit, a
room) points to pattern, colour, style or shape; what they will read or feel on it (the label, the fabric) points to
material or care; where or how it will be used points to the property that
decides that use (size, fit, capacity, compatibility). Not every hint hides a requirement, but every held-back
requirement is hinted.
Choose up to N properties to ask about, the likeliest first:
- first the properties the story hints at this way, the clearest hint first, named as the listings name them;
- then main filters of this kind of product that most listings state with differing values;
- never an identifier (a model name, code or a value that differs on almost every listing), never a property with one
  value on every listing, never logistics (warranty, condition, stock, variant or option lists, storage);
- never a property the request already settles, nor details that merely follow from something it states;
- never two names for the same property.
Use the property names exactly as listed. Answer as JSON only: {"ask": ["<property name>", ...]}"""


ANSWER_GUIDE = """A store assistant asked a shopper one question and got a reply in the shopper's own colloquial voice.
Say plainly what the reply means for the product. Shoppers use idioms and figures of speech: a reply that names a
value as their pick, their rule or their line in the sand means that value is required; a reply that names a value
to avoid means it is excluded; a reply saying the topic doesn't matter or they have no opinion means no preference.
Negative words often describe how firm the shopper is, not a rejection of the value: "X isn't negotiable", "I won't
budge on X", "X isn't up for discussion" all mean X is required. Use excluded only when the shopper does not want X.
If the reply speaks of a different property than the one asked about, report the property the reply names.
Answer as JSON only: {"property": "<the property the reply is about>", "meaning": "required" | "excluded" |
"no preference" | "unclear", "value": "<the value the reply names, else empty>"}"""


CHANGE_GUIDE = """During a shopping task the store or the shopper sent the lines below (a refused checkout, or a note attached
to a result). The ids are products in play: in the cart or being considered. Say what the lines change. Answer as JSON
only: {"unavailable": ["<ids among those given that can no longer be bought: sold out, none left, withdrawn, gone, in any
wording>"], "recheck": ["<ids among those given whose price or stock the lines say has changed, without saying they are
gone>"], "budget": <the shopper's new maximum total price as a number, or null when unchanged>, "dropped": ["<requirements
the shopper no longer needs, in their words>"], "added": ["<new firm requirements, as 'property: value'>"], "summary":
"<one plain sentence of what changed>"}
A line that only changes a wish or the budget does not make an item unbuyable. When a line speaks of "your pick" or "the
item" without an id, it means the first id given."""


REFUSAL_GUIDE = """A store refused a checkout and sent the lines below, from the store or the shopper. The ids are what was in
the cart. Say which of those ids the lines mean can no longer be bought (sold out, unavailable, none left, withdrawn,
in any wording; a line that only changes a wish or the budget does not make an item unbuyable). Answer as JSON only:
{"unavailable": ["<ids>"]}"""


class Episode:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.by_name = {(t.get("function") or {}).get("name"): t for t in self.tools}
        stated = self.policy.get("max_steps")
        self.world = World(env["binding"], int(stated) if isinstance(stated, (int, float)) and stated > 0 else None)
        self.reader = Reader()
        self.started = time.monotonic()
        self.roles: dict[str, str] = {}
        self.plan: dict = {}
        self.questions = 0
        self.inspected: set[str] = set()       # ids whose property list or full view was read
        self.cart: dict[str, int] = {}         # id -> quantity, as the agent's own adds and removes leave it
        self.refusals: list[str] = []
        self.log: list[dict] = []
        self.checkout_guarded: set[str] = set()
        self.opened: dict[tuple[str, str], str] = {}   # (tool, args) of a view or property read -> what it returned
        self.judge = Reader(JUDGES)
        self.planner = Reader(PLANNERS)
        for reader in (self.reader, self.judge, self.planner):
            reader.deadline = self.started + TIME_LIMIT - 25    # no model call may still run when the task's time is up
        self.answers: list[tuple[str, str]] = []       # (question, the shopper's reply)
        self.meanings: list[str] = []                  # what each reply means, read plainly ('pattern: required stripe')
        self.verdict: dict | None = None
        self.verdict_at = -1
        self.cart_tools: dict = {}
        self.iface: dict = {}                  # how to use the cart and finishing tools, read from their schemas
        self.fields: dict[str, str] = {}       # which key of this store's result rows holds id / title / price / stock / category
        self.fields_tried = False
        self.last_call: tuple[dict, Any] = ({}, None)   # the scout's last search: its arguments and result
        self.narrow: tuple | None = None       # the scout's narrow search: (shape, rows read, its last call, matches)
        self.looked_further = 0                # rounds of looking further done (at most two)
        self.dropped = False                   # a change line dropped a requirement: more products may fit now
        self.explored = False
        self.searched: set[str] = set()        # word searches already sent by code
        self.widened = False
        self.cap_now: float | None = None      # the budget a change line set, over the request's own cap
        self.at_checkout: list[str] = []       # what the cart held at the latest checkout: the ids its refusal lines are about
        self.errors_in_row = 0                 # world calls in a row that came back as an error
        self.tool_errors: dict[str, int] = {}  # tool -> errors so far
        self.alarms: list[str] = []            # repeated failures to tell the model about once
        self.commit_failed = False             # a code-built cart or finish call could not be made: the model makes its own
        self.refused_ids: set[str] = set()     # items a refused checkout said can no longer be bought
        self.read_lines: set[str] = set()      # change lines already read into the changes below
        self.changes: list[str] = []           # what the change lines changed, read plainly (budget, needs dropped or added)
        self.recheck: set[str] = set()         # items whose price or stock changed: look again before the next order
        self.checkouts: dict[str, int] = {}    # item -> checkout attempts while it was the pick
        self.cheaper_checked: set[str] = set() # picks whose cheaper seen rows were already opened
        self.traits: dict[str, set[str]] = {}  # property name the opened listings state -> the values seen
        self.props: dict[str, dict[str, str]] = {}   # product id -> the properties its opened page lists
        self.trait_count: dict[str, int] = {}  # property name -> how many opened listings state it
        self.pages_read = 0                    # opened product pages that stated properties
        self.planned: list[str] | None = None  # the question planner's order of properties (None until it has run)
        self.redirected: set[str] = set()      # questions the guard already sent back once
        self.rows: dict[str, str] = {}         # id -> its search-result row (price, stock, category) as last seen

    # -- helpers
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def role(self, name: str) -> str:
        return self.roles.get(name, "other")

    def tool_named(self, role: str) -> str | None:
        mapped = {"checkout": "order", "decline": "decline"}.get(role)
        if mapped and (self.iface.get(mapped) or {}).get("tool"):
            return self.iface[mapped]["tool"]
        return next((n for n, r in self.roles.items() if r == role), None)

    def finish_purpose(self, name: str, args: dict) -> str | None:
        """Whether a call to a finishing tool orders or declines: by the interface map's values when one tool does
        both, else by the tool's role."""
        order, decline = self.iface.get("order") or {}, self.iface.get("decline") or {}
        if order.get("tool") == name and decline.get("tool") == name:
            if all(args.get(k) == v for k, v in decline["args"].items()):
                return "decline"
            if all(args.get(k) == v for k, v in order["args"].items()):
                return "order"
            return None
        if order.get("tool") == name:
            return "order"
        if decline.get("tool") == name:
            return "decline"
        return {"checkout": "order", "decline": "decline"}.get(self.role(name))

    def budget_q(self) -> int | None:
        q = self.plan.get("question_budget")
        return int(q) if q is not None else None

    def status(self) -> str:
        q = self.budget_q()
        steps_left = min(self.world.left(), int(self.plan.get("step_budget") or self.world.max_steps or 20) - self.world.step)
        alarms, self.alarms = self.alarms, []
        parts = ([f"[watchdog] {'; '.join(alarms)}. Read that tool's schema again and use exactly its argument names and "
                  "allowed values; if it keeps failing, use another tool or finish."] if alarms else []) + [
                 f"steps used {self.world.step}, left {steps_left} (keep {self.reserve()} for cart and checkout)",
                 f"questions used {self.questions}" + (f" of {q}" if q is not None else ""),
                 f"cart {json.dumps(self.cart)}" if self.cart else "cart empty"]
        kind = self.item_category()
        if kind:
            parts.append(f"store category matching the request: {kind!r} (only products in it are the kind asked for; "
                         "use it in the category filter)")
        if self.traits and (q is None or self.questions < q):
            parts.append("properties the listings state, not asked yet: " + ", ".join(self.open_traits()[:10]))
        if self.elapsed() > FINISH_BY - 60:
            parts.append("time is short: finish soon")
        return "[status] " + "; ".join(parts)

    def item_category(self) -> str | None:
        """The store category, among those seen in results, that best matches how the plan names the item (most of
        its words in the item's words, then the larger share of its words): the store's own name for the kind of item
        asked for. None when nothing matches or two categories match equally."""
        item = {stem(w) for w in re.findall(r"[a-z]+", str(self.plan.get("item") or "").lower())}
        asked = {stem(w) for w in re.findall(r"[a-z]+", str(self.policy.get("query") or "").lower())}
        scored = []
        for name in {str(self.row_field(i, CATEGORY_FIELD) or "") for i in self.rows}:
            words = {stem(w) for w in re.findall(r"[a-z]+", name.lower()) if w not in ("and", "or", "the", "to", "for", "of", "with")}
            hit = len(words & item)
            if hit:                                       # the request's own words only break ties
                scored.append(((hit, len(words & asked), hit / len(words)), name))
        if not scored:
            # the item's words name no category: a category of two or more words, all of which the request uses
            full = [(len(words), name) for name in {str(self.row_field(i, CATEGORY_FIELD) or "") for i in self.rows}
                    for words in [{stem(w) for w in re.findall(r"[a-z]+", name.lower()) if w not in ("and", "or", "the", "to", "for", "of", "with")}]
                    if len(words) >= 2 and words <= asked]               # one-word names match by accident too easily
            full.sort(reverse=True)
            return full[0][1] if full and (len(full) == 1 or full[0][0] > full[1][0]) else None
        scored.sort(reverse=True)
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return None
        return scored[0][1]

    def row_field(self, product: str, pattern: str) -> Any:
        try:
            row = json.loads(self.rows.get(product) or "{}")
        except ValueError:
            return None
        key = self.fields.get({r"title|name": "title", r"price|cost": "price", r"stock|avail": "stock", CATEGORY_FIELD: "category"}.get(pattern, ""))
        if key and key in row:
            return row[key]
        return next((v for k, v in row.items() if re.search(pattern, str(k).lower())), None)

    def map_fields(self, rows: list[dict]) -> None:
        """Read once, from the first search results, which key of this store's rows holds the id, title, price,
        stock and category; kept only for keys the rows really have."""
        self.fields_tried = True
        sample = [{k: (v[:80] if isinstance(v, str) else v) for k, v in r.items()} for r in rows[:3] if isinstance(r, dict)]
        if not sample:
            return
        answer = self.reader.json(FIELD_GUIDE, json.dumps(sample, ensure_ascii=False), max_tokens=1200)
        keys = set().union(*(set(r) for r in sample))
        self.fields = {role: answer[role] for role in ("id", "title", "price", "stock", "category")
                       if isinstance(answer.get(role), str) and answer[role] in keys}

    # -- the scout: a planned search before the model's own turns
    def search_shape(self) -> dict | None:
        """How the search tool takes words, a mode, a category, a price limit and a page number (None if no search)."""
        name = self.tool_named("finder")
        tool = self.by_name.get(name or "")
        if not tool:
            return None
        props = schema_of(tool).get("properties") or {}
        mapped = self.iface.get("search") or {}
        if mapped.get("tool") == name:
            page = props.get(mapped["page"]) if mapped.get("page") and mapped.get("page_unit") == "page" else None
            return {"tool": name, "text": mapped["words"], "category": mapped.get("category"), "cap": mapped.get("price_limit"),
                    "page": mapped.get("page") if mapped.get("page_unit") == "page" else None,
                    "typed": mapped.get("word_mode"), "fields": mapped.get("browse_mode") if mapped.get("category") else None,
                    "mode": mapped.get("mode"), "page_arg": mapped.get("page"),
                    "page_unit": mapped.get("page_unit") or "page",
                    "pages": int(page["maximum"]) if isinstance(page, dict) and isinstance(page.get("maximum"), int) else 1,
                    "sort": (mapped["sort"], mapped["cheapest_first"]) if mapped.get("sort") and mapped.get("cheapest_first") else None}
        def find(pattern: str, types: tuple) -> str | None:
            return next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") in types and re.search(pattern, k.lower())), None)
        shape = {"tool": name, "text": find(r"text|query|keyword|word|free|term|search|phrase", ("string", "array")),
                 "category": find(CATEGORY_FIELD, ("string",)), "cap": find(r"cap|max|budget|spend|price|limit|under", ("number", "integer")),
                 "page": find(r"page", ("integer", "number")), "typed": None, "fields": None, "mode": None}
        for k, v in props.items():
            enum = [str(e) for e in (v.get("enum") or [])] if isinstance(v, dict) else []
            typed = next((e for e in enum if re.search(r"typ|keyword|text|word|free|search|query", e.lower())), None)
            fields = next((e for e in enum if re.search(r"field|filter|struct|facet|attrib", e.lower())), None)
            if typed or fields:
                shape.update(mode=k, typed=typed, fields=fields)
                break
        shape["page_arg"], shape["page_unit"] = shape["page"], "page"
        if not shape["page"]:
            offset = find(r"offset|skip|start|first|from", ("integer",))
            cursor = find(r"cursor|token|after|next|mark|continu", ("string",))
            shape["page_arg"], shape["page_unit"] = (offset, "offset") if offset else (cursor, "cursor")
        page = props.get(shape["page"]) if shape["page"] else None
        shape["pages"] = int(page["maximum"]) if isinstance(page, dict) and isinstance(page.get("maximum"), int) else 1
        shape["sort"] = None
        for k, v in props.items():
            enum = [str(e) for e in (v.get("enum") or [])] if isinstance(v, dict) else []
            low = next((e for e in enum if re.search(r"price", e.lower()) and re.search(r"asc|low|cheap|up|inc", e.lower())), None) \
                or next((e for e in enum if re.search(r"cheap|lowest|low.?to.?high", e.lower())), None)
            if re.search(r"sort|order|rank", k.lower()) and low:
                shape["sort"] = (k, low)                    # (argument, its value for lowest price first)
                break
        return shape

    def scout_call(self, shape: dict, args: dict) -> tuple[int, list[dict]]:
        tool = self.by_name[shape["tool"]]
        args = fit_args(tool, args)
        if schema_problems(tool, args):
            return 0, []
        result = self.world.call(shape["tool"], args)
        self.note(shape["tool"], args, result)
        self.log.append({"step": self.world.step, "call": {"name": shape["tool"], "args": args}, "result": result, "by": "scout"})
        found = core(result.get("observation"))
        self.last_call = (args, found)
        rows = list(result_rows(found))
        rows = [r for r in rows if isinstance(r, dict)]
        count = next((v for k, v in found.items() if isinstance(v, int) and not isinstance(v, bool)
                      and re.search(r"match|total|count|hits|found|number", str(k).lower())), len(rows)) if isinstance(found, dict) else 0
        return int(count), rows

    def scout_words(self, shape: dict, words: list[str], read_all: bool = True) -> tuple[int, list[dict]]:
        self.searched.add(" ".join(words))
        args = {shape["text"]: " ".join(words)}
        if shape["mode"] and shape["typed"]:
            args[shape["mode"]] = shape["typed"]
        if shape.get("sort") and self.wants_cheapest():
            args[shape["sort"][0]] = shape["sort"][1]
        count, rows = self.scout_call(shape, args)
        if read_all and shape["page"] and len(rows) < count:
            for number in range(2, shape["pages"] + 1):
                if len(rows) >= count or self.steps_left() <= self.reserve() + 8:
                    break
                rows += self.scout_call(shape, {**args, shape["page"]: number})[1]
        return count, rows

    def scout(self) -> str:
        """Before the model's turns: find the store category from the plan's item, search it with the stated values
        (widening only when nothing matches), read its next pages when the cheapest or an affordable fit is wanted,
        sweep the category by price for a cheapest-wanted request, and open
        the best candidates' property lists. Returns a note for the model; every call is a normal step."""
        shape = self.search_shape()
        budget = int(self.plan.get("step_budget") or self.world.max_steps or 20)
        if not shape or not shape["text"] or self.steps_left() <= self.reserve() + min(14, max(8, budget // 3)):
            return ""
        notes: list[str] = []
        item = query_words(self.plan.get("item"))
        for words in (item[:3], item[:1]):
            if words:
                count, _ = self.scout_words(shape, words, read_all=False)
                notes.append(f"searched {' '.join(words)!r}: {count} matches")
                if count:
                    break
        kind = self.item_category()
        if not kind:
            return "\n".join(notes)
        base = query_words(kind)
        values = value_words(self.plan.get("requirements"), base)
        # the required values first, then the kind: some stores read only the first words of a query (race 1's world
        # read two), and a store that reads them all is indifferent to their order
        count, rows = self.scout_words(shape, (values + base)[:6])
        notes.append(f"searched {' '.join((values + base)[:6])!r}: {count} matches")
        while count == 0 and values and self.steps_left() > self.reserve() + 10:
            values = values[:-1]
            count, rows = self.scout_words(shape, (values + base)[:6])
            notes.append(f"searched {' '.join((values + base)[:6])!r}: {count} matches")
        if count == 0 and self.steps_left() > self.reserve() + 10:
            # this store's word search may not cover category names: the item's own words with the values instead
            values = value_words(self.plan.get("requirements"), base)
            words = ([w for w in values if w not in item[:2]] + item[:2])[:6]
            count, rows = self.scout_words(shape, words)
            notes.append(f"searched {' '.join(words)!r}: {count} matches")
        if count > len(rows) and self.wants_sweep(rows):
            # the cheapest (or an affordable) fit is often past the first page: read on while the steps allow
            per = max(1, len(next((v for v in self.last_call[1].values() if isinstance(v, list)), [])) if isinstance(self.last_call[1], dict) else 1)
            most = min(SWEEP_PAGES, self.steps_left() - self.reserve() - SWEEP_ROOM, -(-(count - len(rows)) // per))
            if self.price_ordered(shape, rows):
                most = min(most, 1)
            if most > 0:
                before = len(rows)
                rows = self.sweep(shape, rows, most)
                notes.append(f"read {len(rows) - before} more rows of that search from its next pages")
        if count:
            self.narrow = (shape, list(rows), self.last_call, count)
        matched = {i for r in rows for i in list(ids_in(r, self.fields.get("id")))[:1]} if values else set()     # rows that carry a stated value
        if self.wants_cheapest() and not shape["cap"] and self.steps_left() > self.reserve() + SWEEP_ROOM:
            # no price filter: other listings of the kind show other rows, and the cheapest fit is often among them
            if values and len(values) < len((values + base)[:6]):
                more_count, more = self.scout_words(shape, values[:2], read_all=False)
                notes.append(f"searched {' '.join(values[:2])!r}: {more_count} matches")
            if shape["category"] and shape["mode"] and shape["fields"]:
                label = self.shelf_label(shape, kind)
                listed, shelf = self.scout_call(shape, {shape["mode"]: shape["fields"], shape["category"]: label})
                if listed > len(shelf) and shelf:
                    shelf = self.sweep(shape, shelf, SWEEP_PAGES)
                notes.append(f"listed the shelf {label!r}: {len(shelf)} rows read")
        cap = self.plan.get("price_cap")
        if self.wants_cheapest() and shape["category"] and shape["cap"] and shape["mode"] and shape["fields"] and self.steps_left() > self.reserve() + 12:
            fields = {shape["mode"]: shape["fields"], shape["category"]: kind}
            if shape.get("sort"):
                fields[shape["sort"][0]] = shape["sort"][1]        # lowest prices first: one listing shows them
            seen: list[tuple[float, int]] = []                    # (price limit, items listed)
            best, reach, limit = None, None, float(cap) if isinstance(cap, (int, float)) else None
            for _ in range(6):
                if not limit or limit <= 0 or self.steps_left() <= self.reserve() + 8:
                    break
                listed, listed_rows = self.scout_call(shape, {**fields, shape["cap"]: round(limit, 2)})
                reach = reach or max(1, len(listed_rows)) * shape["pages"]
                seen.append((limit, listed))
                fits = [(c, n) for c, n in seen if n <= reach]
                over = [(c, n) for c, n in seen if n > reach]
                if fits:
                    best = max(fits)
                if not over or (best and best[1] >= reach * 0.6) or shape.get("sort"):
                    break
                hi = min(over)
                if not fits:
                    # first guess: prices spread evenly; later, halve on the log scale
                    limit = hi[0] * reach / max(1, hi[1]) * 0.8 if len(seen) == 1 else hi[0] / 2
                else:
                    lo = max(fits)
                    limit = (lo[0] * hi[0]) ** 0.5 if lo[0] > 0 else hi[0] / 2
                    if hi[0] - lo[0] < max(1.0, hi[0] * 0.02):
                        break
            if shape.get("sort") and best is None and seen:
                best = seen[0]                                    # sorted by price: the first pages are the cheapest
            if best:
                notes.append(f"listed the category {kind!r} up to price {best[0]:.2f}: {best[1]} items")
                if shape["page"] and best[1] > 7:
                    for number in range(2, shape["pages"] + 1):
                        if self.steps_left() <= self.reserve() + 8:
                            break
                        self.scout_call(shape, {**fields, shape["cap"]: round(best[0], 2), shape["page"]: number})
        reader = self.tool_named("properties") or self.tool_named("item_page")
        candidates = [i for i in self.rows if self.row_field(i, CATEGORY_FIELD) == kind and i not in self.inspected
                      and self.row_field(i, r"stock|avail") is not False
                      and not (isinstance(cap, (int, float)) and isinstance(self.row_field(i, r"price|cost"), (int, float)) and self.row_field(i, r"price|cost") > cap)]
        wanted = set(value_words(self.plan.get("requirements"), base))
        def price_of(i: str) -> float:
            price = self.row_field(i, r"price|cost")
            return float(price) if isinstance(price, (int, float)) else 1e12
        def titled(i: str) -> bool:
            return bool(set(query_words(self.row_field(i, r"title|name"))) & wanted)
        if self.wants_cheapest():
            # the cheapest rows that carry a stated value, then the cheapest listed row whose title has one, then the
            # cheapest listed row of all: the cheapest fit is usually among these
            a = sorted((i for i in candidates if i in matched), key=price_of)
            b = sorted((i for i in candidates if i not in matched and titled(i)), key=price_of)
            c = sorted((i for i in candidates if i not in matched and not titled(i)), key=price_of)
            order = a[:2] + b[:1] + c[:1] + a[2:] + b[1:] + c[1:]
        else:
            order = sorted(candidates, key=lambda i: (i not in matched, not titled(i), price_of(i)))
        candidates = list(dict.fromkeys(order))
        opened = []
        if reader:
            tool = self.by_name[reader]
            key = next(iter(schema_of(tool).get("required") or list((schema_of(tool).get("properties") or {}))), None)
            for product in candidates[:4]:
                if key is None or self.steps_left() <= self.reserve() + 6:
                    break
                args = fit_args(tool, {key: product})
                if schema_problems(tool, args):
                    break
                result = self.world.call(reader, args)
                self.note(reader, args, result)
                self.log.append({"step": self.world.step, "call": {"name": reader, "args": args}, "result": result, "by": "scout"})
                if not is_fault(result.get("observation")):
                    self.opened[(reader, json.dumps(args, sort_keys=True))] = shown(result.get("observation"), long=True)
                    opened.append(f"{product}: {shown(result.get('observation'))[:600]}")
        asked = self.scout_questions(min(3, self.budget_q()) if self.budget_q() else 2)
        listing = [self.rows[i][:220] for i in candidates[:14]]
        return ("\n".join(notes) + f"\nproducts of the category {kind!r} found so far (cheapest first)" * self.wants_cheapest()
                + f"\nCANDIDATE ROWS in {kind!r}:\n" + "\n".join(listing)
                + ("\nPROPERTY LISTS OPENED (the judge has them too):\n" + "\n".join(opened) if opened else "")
                + ("\nQUESTIONS ALREADY ASKED (they used questions from the budget):\n" + "\n".join(asked) if asked else ""))

    def shelf_label(self, shape: dict, kind: str) -> str:
        """The category argument's value for a store category: its enum entry when the schema lists them (matched
        without case), else the name in lower case (shelf labels are usually written that way)."""
        spec = (schema_of(self.by_name[shape["tool"]]).get("properties") or {}).get(shape["category"]) or {}
        enum = [str(e) for e in spec.get("enum") or []]
        return next((e for e in enum if e.lower() == kind.lower()), kind if enum else kind.lower())

    def row_value(self, row: dict, role: str, pattern: str) -> Any:
        key = self.fields.get(role)
        if key and key in row:
            return row[key]
        return next((v for k, v in row.items() if re.search(pattern, str(k).lower())), None)

    def wants_sweep(self, rows: list[dict]) -> bool:
        """Whether to read past the first page of the narrow search: the request wants the cheapest fit, or fewer
        than two of the rows seen are within its price cap."""
        if self.wants_cheapest():
            return True
        cap = self.plan.get("price_cap")
        if not isinstance(cap, (int, float)) or isinstance(cap, bool):
            return False
        prices = [self.row_value(r, "price", r"price|cost") for r in rows]
        return sum(1 for p in prices if isinstance(p, (int, float)) and p <= cap) < 2

    def price_ordered(self, shape: dict, rows: list[dict]) -> bool:
        """True when this search lists the cheapest first: it was asked to, its description says so, or the rows
        read come in rising price order."""
        if shape.get("sort") and self.wants_cheapest():
            return True
        text = str((self.by_name[shape["tool"]].get("function") or {}).get("description") or "").lower()
        if re.search(r"cheapest first|lowest price first|price ascending|low to high", text):
            return True
        prices = [p for p in (self.row_value(r, "price", r"price|cost") for r in rows) if isinstance(p, (int, float))]
        return len(prices) >= 4 and prices == sorted(prices)

    def next_page(self, shape: dict, args: dict, found: Any, seen: int) -> dict | None:
        """The arguments for the page after the one `args` fetched: where the result says the next page starts when it
        names that (an empty marker means this was the last page), else the next page number or row offset (`seen`
        rows read so far); None when the tool cannot page."""
        key = shape.get("page_arg")
        spec = (schema_of(self.by_name[shape["tool"]]).get("properties") or {}).get(key or "")
        if not key or not isinstance(spec, dict):
            return None
        marks = [v for k, v in found.items() if re.search(NEXT_KEY, str(k).lower()) and not isinstance(v, (bool, list, dict))] \
            if isinstance(found, dict) else []
        if marks:
            value = marks[0]
            if value in (None, ""):
                return None
        elif shape.get("page_unit") == "offset":
            value = seen
        elif shape.get("page_unit") == "page":
            value = int(args.get(key) or 1) + 1 if str(args.get(key) or 1).isdigit() else None
        else:
            value = None
        if value is None:
            return None
        if spec.get("type") == "integer" and not isinstance(value, int):
            digits = re.search(r"\d+", str(value))
            if not digits:
                return None
            value = int(digits.group())
        elif spec.get("type") == "string":
            value = str(value)
        return {**args, key: value}

    def sweep(self, shape: dict, rows: list[dict], most: int, keep: int = SWEEP_ROOM) -> list[dict]:
        """Read up to `most` more pages of the last search, from where it stopped; ends at its last page, at a page
        that adds no new product, or when the steps run short. Returns every row read."""
        args, found = self.last_call
        key = self.fields.get("id")
        seen = {i for r in rows for i in ids_in(r, key)}
        for _ in range(most):
            if self.steps_left() <= self.reserve() + keep:
                break
            nxt = self.next_page(shape, args, found, len(rows))
            if nxt is None:
                break
            _, more = self.scout_call(shape, nxt)
            args, found = self.last_call
            fresh = [r for r in more if ids_in(r, key) - seen]
            if not fresh:
                break
            seen |= {i for r in fresh for i in ids_in(r, key)}
            rows = rows + fresh
        return rows

    def unit_cap(self) -> float | None:
        """The highest price one unit may have now: a changed budget over the request's cap, split over the units
        when the cap is for the whole order."""
        cap = self.cap_now if self.cap_now is not None else self.plan.get("price_cap")
        if not isinstance(cap, (int, float)) or isinstance(cap, bool):
            return None
        units = self.plan.get("quantity")
        units = int(units) if isinstance(units, (int, float)) and not isinstance(units, bool) and units > 1 else 1
        return float(cap) / units if units > 1 and not re.search(r"each|per unit|per item|a piece", str(self.plan.get("price_rule") or "").lower()) else float(cap)

    def look_further(self, by_price: bool = False, extra: bool = False) -> bool:
        """When no opened product fits (after a refused order, or before ordering or declining): read on in the
        scout's narrow search, then open the cheapest unopened rows within the price cap that a short reading finds
        plausible, the narrow search's own rows first. A decline is rarely right (in race 1 every passing run ended in
        an order), so this runs up to twice; the second round needs a few steps more. True if a product was opened."""
        reader = self.tool_named("properties") or self.tool_named("item_page")
        if not reader or self.elapsed() > FINISH_BY - 40 or (not extra and (self.looked_further >= 2
                or (self.looked_further and self.steps_left() <= self.reserve() + 3))):
            return False
        self.looked_further += 0 if extra else 1
        rows: list[dict] = []
        if self.narrow and not by_price:
            shape, rows, last, count = self.narrow
            if len(rows) < count:
                self.last_call = last
                rows = self.sweep(shape, rows, 2 * SWEEP_PAGES, keep=4)    # leaves steps to open three and order
                self.narrow = (shape, list(rows), self.last_call, count)  # a second round reads on from here
        narrow = {i for r in rows for i in ids_in(r, self.fields.get("id"))}
        cap, kind = self.unit_cap(), self.item_category()
        fresh = sorted((i not in narrow and not by_price, p, i) for i in self.rows if i not in self.inspected and i not in self.refused_ids
                       for p in [self.row_field(i, r"price|cost")] if isinstance(p, (int, float)) and (cap is None or p <= cap)
                       and (kind is None or self.row_field(i, CATEGORY_FIELD) in (None, kind))
                       and self.row_field(i, r"stock|avail") is not False)
        ids = [i for _, _, i in fresh]
        if len(ids) > 3:
            listing = [{"id": i, "title": self.row_field(i, r"title|name"), "price": self.row_field(i, r"price|cost")} for i in ids[:40]]
            verdict = self.reader.json(PLAUSIBLE_GUIDE, json.dumps({"request": self.policy.get("query"), "changes": self.changes,
                                                                    "rows": listing}, ensure_ascii=False), max_tokens=1500)
            keep = {str(i) for i in verdict.get("plausible") or []}
            ids = [i for i in ids if i in keep] or ids[:2]
        tool = self.by_name[reader]
        key = next(iter(schema_of(tool).get("required") or list((schema_of(tool).get("properties") or {}))), None)
        opened = False
        for product in ids[:3]:
            if key is None or self.steps_left() <= self.reserve() + 1 or self.elapsed() > FINISH_BY - 30:
                break
            args = fit_args(tool, {key: product})
            if schema_problems(tool, args):
                break
            result = self.world.call(reader, args)
            self.note(reader, args, result)
            self.log.append({"step": self.world.step, "call": {"name": reader, "args": args}, "result": result, "by": "further"})
            if not is_fault(result.get("observation")):
                self.opened[(reader, json.dumps(args, sort_keys=True))] = shown(result.get("observation"), long=True)
                opened = True
        return opened

    def explore(self) -> bool:
        """Before a cheapest-wanted order, with steps to spare: run the searches not sent yet (the values alone, each
        value with the kind, the kind alone), then open the cheapest unopened rows within the cap. The cheapest fit
        is often outside the rows the first searches showed. Once; True if a product was opened."""
        self.explored = True
        shape, kind = self.search_shape(), self.item_category()
        if not shape or not shape["text"] or not kind or self.elapsed() > FINISH_BY - 60:
            return False
        base = query_words(kind)
        values = value_words(self.plan.get("requirements"), base)
        plans = ([values[:2]] if values else []) + [[v] + base[:1] for v in values[:3]] + [base[:3]]
        for words in [w for w in plans if w and " ".join(w) not in self.searched][:3]:
            if self.steps_left() <= self.reserve() + 6:
                break
            self.scout_words(shape, words, read_all=False)
        return self.look_further(by_price=True, extra=True)

    def widen(self) -> bool:
        """After a requirement is dropped on a cheapest-wanted request: search with the kind's own words only (no
        values), then open the cheapest unopened rows within the cap. Once; True if a product was opened."""
        self.widened = True
        shape = self.search_shape()
        kind = self.item_category()
        if not shape or not shape["text"] or not kind or self.steps_left() <= self.reserve() + 5 or self.elapsed() > FINISH_BY - 40:
            return False
        self.scout_words(shape, query_words(kind)[:3], read_all=False)
        return self.look_further(by_price=True, extra=True)

    def scout_questions(self, most: int) -> list[str]:
        """Ask the shopper about the likeliest held-back properties (the planner's order of the properties the opened
        listings state), one per message, each named with one example value the listings show."""
        tool = self.tool_named("ask_shopper")
        q = self.budget_q()
        if not tool or not self.traits:
            return []
        schema = schema_of(self.by_name[tool])
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and not v.get("enum")]
        key = next((k for k in texts if re.search(r"text|message|body|content|question|note|say", k.lower())), None) \
            or max(texts, key=lambda k: int(props[k].get("maxLength") or 10 ** 6), default=None)
        others = [k for k in schema.get("required") or [] if k in texts and k != key]   # a topic or subject line
        out = []
        for name in self.open_traits()[:most]:
            if key is None or (q is not None and self.questions >= min(q, most)) or self.steps_left() <= self.reserve() + 6:
                break
            # one example value: a message offering two values ("A or B") can read as two questions and be refused
            examples = [v for v in sorted(self.traits.get(name) or [], key=len) if 0 < len(v) <= 30][:1]
            question = f"Do you have a preference for the {name}" + (f" (for example {examples[0]})" if examples else "") + "?"
            args = fit_args(self.by_name[tool], {key: question, **{k: name[: int(props[k].get("maxLength") or 60)] for k in others}})
            if schema_problems(self.by_name[tool], args):
                break                                  # the ask tool wants more than one text: the model asks instead
            result = self.world.call(tool, args)
            self.note(tool, args, result)
            self.log.append({"step": self.world.step, "call": {"name": tool, "args": args}, "result": result, "by": "scout"})
            meaning = self.read_answer(question, reply_text(result.get("observation")))
            out.append(f"Q: {question}\nA: {shown(result.get('observation'))[:300]}" + (f"\n[reading] {meaning}" if meaning else ""))
        return out

    def wants_cheapest(self) -> bool:
        text = f"{self.plan.get('rank_by') or ''} {self.policy.get('query') or ''}".lower()
        return bool(re.search(r"cheap|costs? the least|least expensive|lowest price|lowest cost|lowest-priced|most affordable|least money", text))

    def check_cheaper(self, choice: str) -> bool:
        """For a cheapest-wanted request, before ordering: open the property lists of products a search showed cheaper
        than the pick, in the same category, not out of stock and never opened; the cheapest three (when more, those a
        short reading finds plausible). True if any was opened."""
        self.cheaper_checked.add(choice)
        price = self.row_field(choice, r"price|cost")
        reader = self.tool_named("properties") or self.tool_named("item_page")
        if not isinstance(price, (int, float)) or not reader:
            return False
        category = self.row_field(choice, CATEGORY_FIELD)
        cheaper = sorted((p, i) for i in self.rows if i not in self.inspected and i not in self.refused_ids
                         for p in [self.row_field(i, r"price|cost")] if isinstance(p, (int, float)) and p < price
                         and (category is None or self.row_field(i, CATEGORY_FIELD) == category)
                         and self.row_field(i, r"stock|avail") is not False)
        if len(cheaper) > 3:
            listing = [{"id": i, "title": self.row_field(i, r"title|name"), "price": p} for p, i in cheaper[:40]]
            verdict = self.reader.json(PLAUSIBLE_GUIDE, json.dumps({"request": self.policy.get("query"), "rows": listing}, ensure_ascii=False), max_tokens=1500)
            keep = {str(i) for i in verdict.get("plausible") or []}
            cheaper = [(p, i) for p, i in cheaper if i in keep] or cheaper[:2]
        tool = self.by_name[reader]
        key = next(iter(schema_of(tool).get("required") or list((schema_of(tool).get("properties") or {}))), None)
        opened = False
        for _, product in cheaper[:3]:
            if key is None or self.steps_left() <= self.reserve() + 1 or self.elapsed() > FINISH_BY - 30:
                break
            args = fit_args(tool, {key: product})
            if schema_problems(tool, args):
                break
            result = self.world.call(reader, args)
            self.note(reader, args, result)
            self.log.append({"step": self.world.step, "call": {"name": reader, "args": args}, "result": result, "by": "cheaper"})
            if not is_fault(result.get("observation")):
                self.opened[(reader, json.dumps(args, sort_keys=True))] = shown(result.get("observation"), long=True)
                opened = True
        return opened

    def answers_said(self) -> list[str]:
        """The questions asked so far, lower-cased."""
        return [q.lower() for q, _ in self.answers]

    def ranked_traits(self) -> list[str]:
        """Properties the opened listings state that neither the request nor a question has named yet, the ones most
        listings state first (a shopper's held-back properties are ones most listings of the kind state); one name
        per head word."""
        said = str(self.policy.get("query") or "") + " " + " ".join(q for q, _ in self.answers)
        item, heads, left = str(self.plan.get("item") or ""), set(), []
        kept: list[set[str]] = []
        for name in sorted(self.traits, key=lambda n: (-self.trait_count.get(n, 0), -len(self.traits[n]))):
            head = property_head(name, item)
            words = {w for w in re.findall(r"[a-z]+", name.lower()) if len(w) >= 3}
            if any(words <= k or k <= words for k in kept):
                continue                               # one property under two names ('speed' / 'speed ghz')
            if head and head not in heads and not names_property(said, name, item) and not self.settled(name):
                heads.add(head)
                kept.append(words)
                left.append(name)
        return left

    def settled(self, name: str) -> bool:
        """True when the request names, as whole words, a value the opened listings give this property and no other
        property: the request already settles it ('ash for the body' settles the body material)."""
        text = " " + re.sub(r"[^a-z0-9]+", " ", str(self.policy.get("query") or "").lower()) + " "
        def flat(value: Any) -> str:
            return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
        others = {flat(v) for n, values in self.traits.items() if n != name for v in values}
        return any(len(v) >= 3 and re.search(r"[a-z]", v) and f" {v} " in text and v not in others
                   for v in map(flat, self.traits.get(name) or []))

    def planned_order(self) -> list[str] | None:
        """The question planner's order of the properties to ask about: one model call, made once three product pages
        have stated properties, over each property's name, how many opened listings state it and example values.
        None before then, or when the planner names nothing usable."""
        if self.planned is None and self.pages_read >= 3 and self.traits:
            self.planned = []
            left = self.ranked_traits()
            menu = [{"property": n, "listings": self.trait_count.get(n, 0), "examples": sorted(self.traits[n])[:4]} for n in left]
            most = self.budget_q() or 3
            hints = {"reasons": self.plan.get("reasons"), "likely_hidden": self.plan.get("likely_hidden")}
            answer = self.planner.json(PLANNER_GUIDE.replace("up to N", f"up to {most}"),
                                      json.dumps({"request": self.policy.get("query"), "story_reading": hints, "properties": menu},
                                                 ensure_ascii=False), max_tokens=1500)
            names = set(left)
            self.planned = [n for n in answer.get("ask") or [] if isinstance(n, str) and n in names]
        return self.planned or None

    def open_traits(self) -> list[str]:
        """The properties to ask about, in order: the question planner's choice first, once it has run, then the rest
        by how many listings state them; before the planner runs (or when it names nothing), by that count alone."""
        left = self.ranked_traits()
        plan = self.planned_order()
        if plan:
            return [n for n in plan if n in left] + [n for n in left if n not in plan]
        # who it is for: when the plan finds that this kind of product differs by who uses it and the request does not
        # say, that is a property shoppers often keep back; it goes second, after the most stated one
        if self.plan.get("ask_who_for") is True and not any("gender" in n for n in self.answers_said()):
            who = next((n for n in left if "gender" in n), "gender")
            left = [n for n in left if n != who]
            left.insert(min(1, len(left)), who)
        return left

    def watch(self, name: str, result: dict) -> None:
        """Count world errors: a tool that fails twice raises an alarm for the model (and a cart or finishing tool is
        no longer built by code), and too many errors in a row end the episode with what is known."""
        observation = result.get("observation") if isinstance(result, dict) else None
        if is_fault(observation):
            self.errors_in_row += 1
            self.tool_errors[name] = self.tool_errors.get(name, 0) + 1
            if self.tool_errors[name] >= 2:
                self.alarms.append(f"{name} has failed {self.tool_errors[name]} times, last: {compact(observation)[:200]}")
                if name in ((self.cart_tools.get("add") or {}).get("tool"), (self.cart_tools.get("remove") or {}).get("tool"),
                            self.tool_named("checkout"), self.tool_named("decline")):
                    self.commit_failed = True
        else:
            self.errors_in_row = 0

    def endgame_tools(self) -> list[dict]:
        """Inside the step reserve only the tools that finish remain on offer: cart, views (to re-check price and
        stock) and the finishing tools."""
        if self.steps_left() > self.reserve():
            return self.tools
        keep = {"cart_change", "cart_review", "item_page", "properties", "checkout", "decline"}
        names = {self.tool_named("checkout"), self.tool_named("decline")}
        narrowed = [t for t in self.tools if self.role((t.get("function") or {}).get("name") or "") in keep
                    or (t.get("function") or {}).get("name") in names]
        return narrowed or self.tools

    def reserve(self) -> int:
        """Steps kept back for cart and checkout: four, fewer when the whole budget is small."""
        budget = int(self.plan.get("step_budget") or self.world.max_steps or 20)
        return min(STEP_RESERVE_LONG if budget >= 20 else STEP_RESERVE, budget // 4 + 1)

    def steps_left(self) -> int:
        return min(self.world.left(), int(self.plan.get("step_budget") or self.world.max_steps or 20) - self.world.step)

    # -- the judge: which opened product meets every requirement
    def decide(self) -> dict:
        """The judge's reading of the opened products, re-asked only when something new was opened or heard."""
        self.read_changes(list(dict.fromkeys(list(self.cart) + [str(self.verdict.get("choice"))] if self.verdict else list(self.cart))))
        stamp = len(self.opened) * 1000 + len(self.answers) * 10 + len(self.refusals) + 100000 * len(self.changes)
        if self.verdict is not None and (self.verdict_at == stamp or self.elapsed() > FINISH_BY + 40):
            return self.verdict                        # nothing new, or too late for another judge call
        products = "\n".join(f"- {text}" for text in self.opened.values())[:9000]
        rows = "\n".join(f"- {self.rows[i]}" for i in sorted(self.inspected) if i in self.rows)[:3000]
        if rows:
            products += "\n\nSEARCH-RESULT ROWS OF THOSE PRODUCTS (price, stock, category as the search showed them):\n" + rows
        said = "\n".join(f"Q: {q}\nA: {a}" for q, a in self.answers) or "(none)"
        if self.meanings:
            said += "\nWHAT THE REPLIES MEAN (read plainly; use this reading):\n" + "\n".join(self.meanings)
        refused = "\n".join(self.refusals) or "(none)"
        if self.changes:
            refused += "\nWHAT THESE LINES CHANGED (read plainly; the latest budget counts):\n" + "\n".join(self.changes)
        if self.refused_ids:
            refused += "\nThese ids were in the cart when a checkout was refused for being sold out or unavailable, so they can no longer be bought: " + ", ".join(sorted(self.refused_ids))
        kind = self.item_category()
        user = (f"REQUEST:\n{self.policy.get('query')}\n\n" + (f"STORE CATEGORY FOR THE ITEM: {kind}\n\n" if kind else "")
                + f"SHOPPER'S ANSWERS:\n{said}\n\nREFUSAL LINES:\n{refused}\n\nPRODUCTS OPENED:\n{products or '(none)'}")
        verdict = self.judge.json(JUDGE_GUIDE, user, max_tokens=3200) or {}
        if "choice" not in verdict and len(self.judge.models) > 1:
            first = self.judge.models.pop(0)
            self.judge.models.append(first)            # the next judge model gets the next try
            verdict = self.judge.json(JUDGE_GUIDE, user, max_tokens=3200) or {}
        if verdict.get("choice") not in (None, "", "null"):
            verdict["choice"] = self.known_id(str(verdict["choice"]))
        verdict["acceptable"] = [self.known_id(str(i)) for i in verdict.get("acceptable") or [] if i not in (None, "", "null")]
        vetoed = {str(i): why for i in [verdict.get("choice")] + list(verdict.get("acceptable") or [])
                  if i not in (None, "", "null") for why in [self.contradicted(str(i))] if why}
        if vetoed:
            # the store's own listing contradicts a stated requirement: the store will not count that product
            self.log.append({"veto": vetoed})
            verdict["acceptable"] = [i for i in verdict.get("acceptable") or [] if str(i) not in vetoed]
            verdict.setdefault("rejected", {}).update(vetoed) if isinstance(verdict.get("rejected"), dict) else None
            if str(verdict.get("choice")) in vetoed:
                left = [str(i) for i in verdict["acceptable"]]
                priced = sorted((p, i) for i in left for p in [self.row_field(i, r"price|cost")] if isinstance(p, (int, float)))
                verdict["choice"] = (priced[0][1] if priced and self.wants_cheapest() else (left[0] if left else None))
        choice = str(verdict.get("choice")) if verdict.get("choice") not in (None, "", "null") else None
        if choice and not str(self.plan.get("rank_by") or "").strip() and not self.wants_cheapest() and not self.lists_all(choice):
            # the store counts a listed property first: among the acceptable products, take one that states every
            # requirement in its own properties over one that shows it only in its title (no ranking asked for, so the
            # cheapest such one keeps the most room)
            listed = [str(i) for i in verdict.get("acceptable") or [] if str(i) != choice and str(i) not in self.refused_ids
                      and not self.over_budget(str(i)) and self.row_field(str(i), r"stock|avail") is not False and self.lists_all(str(i))]
            if listed:
                priced = sorted((p, i) for i in listed for p in [self.row_field(i, r"price|cost")] if isinstance(p, (int, float)))
                verdict["choice"] = priced[0][1] if priced else listed[0]
                self.log.append({"listed_over_title": [choice, verdict["choice"]]})
        if str(verdict.get("choice")) in self.refused_ids:
            verdict["choice"] = next((str(i) for i in verdict.get("acceptable") or [] if str(i) not in self.refused_ids), None)
        if verdict.get("choice") not in (None, "", "null") and self.over_budget(str(verdict["choice"])):
            verdict["choice"] = next((str(i) for i in verdict.get("acceptable") or []
                                      if str(i) not in self.refused_ids and not self.over_budget(str(i))), None)
        self.verdict = verdict
        self.verdict_at = stamp
        self.log.append({"judge": self.verdict})
        return self.verdict

    def contradicted(self, product: str) -> str | None:
        """Why the store would not count this product, read from its own listed properties, or None. For each plain
        stated requirement (no number, not a restatement of the category): when a listed property speaks
        to it (its name holds every word of the requirement's property) and none of those properties holds the wanted
        words, the listing contradicts it. A requirement no property speaks to is left to the judge (the title). A
        requirement that is wholly an exclusion needs the property named like it, stated without the excluded words."""
        props = self.props.get(product)
        if not props:
            return None
        kind = {stem_word(w) for w in term_words(self.row_field(product, CATEGORY_FIELD) or "")}
        reqs = [r for r in self.plan.get("requirements") or [] if isinstance(r, dict)]
        for req in reqs:
            prop, value = str(req.get("property") or ""), str(req.get("value") or "")
            excluded = excluded_phrases(prop, value) if prop and value else []
            named = {stem_word(w) for w in term_words(prop)} - {"type", "of", "the"}
            if excluded and any(r is not req and str(r.get("value") or "").strip()
                                and not excluded_phrases(str(r.get("property") or ""), str(r.get("value") or ""))
                                and named & ({stem_word(w) for w in term_words(r.get("property") or "")} - {"type", "of", "the"})
                                for r in reqs):
                excluded = []                          # it explains a wanted value of the same property ('baggy, not slim'): the judge reads it
            if excluded:
                # the store judges an exclusion on the property named like it, and only a product that states it
                name = " ".join(term_words(prop))
                need = {stem_word(w) for w in term_words(prop)} - {"type", "of", "the"}
                speak = [v for k, v in props.items() if " ".join(term_words(k)) == name] \
                    or [v for k, v in props.items() if need and need <= {stem_word(w) for w in term_words(k)}][:1]
                if not speak:
                    return f"it does not state its {prop}, so it does not show it is {value!r}"
                if any(holds(speak[0], x) for x in excluded):
                    return f"its listed {prop} is {speak[0]!r}, not {value!r}"
                continue
            if not prop or not value or re.search(r"\d", value) \
                    or re.search(r"(^|[\s(])(not|no|non|never|without|avoid|except|anything but)\b|\bnon-|-free\b|\bfree of\b", value.lower()):
                continue                               # numbers and exclusions in any wording are the judge's
            need = set(term_words(prop)) - {"type", "of", "the"}
            own = {stem_word(w) for w in term_words(prop)}
            # each alternative ('basic, plain' / 'red or blue') without the property's or the category's own words
            alternatives = []
            for part in re.split(r"\bor\b|/|,|;", value.lower()):
                words = [w for w in term_words(part) if stem_word(w) not in own and stem_word(w) not in kind and w not in ("a", "an", "the")]
                if words:
                    alternatives.append(" ".join(words))
            if not need or not alternatives or any(len(a.split()) > 3 for a in alternatives):
                continue                               # restates the category, or a long phrase: the judge reads it
            speak = [v for k, v in props.items() if need <= set(term_words(k)) and " ".join(term_words(v)) not in NOT_STATED]
            if not speak:
                continue
            if not any(holds(v, a) for v in speak for a in alternatives):
                return f"its listed {prop} is {speak[0]!r}, not {value!r}"
        return None

    def lists_all(self, product: str) -> bool:
        """True when the product's own listed properties state every plain stated requirement (no number, no
        exclusion, not a restatement of the category): a property speaking to it holds one of the wanted values."""
        props = self.props.get(product)
        if not props:
            return False
        kind = {stem_word(w) for w in term_words(self.row_field(product, CATEGORY_FIELD) or "")}
        for req in self.plan.get("requirements") or []:
            if not isinstance(req, dict):
                continue
            prop, value = str(req.get("property") or ""), str(req.get("value") or "")
            if not prop or not value or re.search(r"\d", value) or re.search(r"availab|stock|ship|shelf|price|budget|cost|quantity", prop.lower()) \
                    or re.search(r"(^|[\s(])(not|no|non|never|without|avoid|except|anything but)\b|\bnon-|-free\b|\bfree of\b", value.lower()):
                continue
            need = set(term_words(prop)) - {"type", "of", "the"}
            own = {stem_word(w) for w in term_words(prop)}
            alternatives = [" ".join(words) for part in re.split(r"\bor\b|/|,|;", value.lower())
                            for words in [[w for w in term_words(part) if stem_word(w) not in own and stem_word(w) not in kind and w not in ("a", "an", "the")]] if words]
            if not need or not alternatives or any(len(a.split()) > 3 for a in alternatives):
                continue
            speak = [v for k, v in props.items() if need <= set(term_words(k)) and " ".join(term_words(v)) not in NOT_STATED]
            if not any(holds(v, a) for v in speak for a in alternatives):
                return False
        return True

    def known_id(self, product: str) -> str:
        """The store's own id for one a model wrote: as written when the store showed it, else the one shown id that
        it matches without a prefix, separators or case ('cfcf0350c74f' -> 'p-cfcf0350c74f')."""
        known = set(self.rows) | set(self.inspected)
        if product in known:
            return product
        flat = re.sub(r"[^a-z0-9]", "", product.lower())
        same = [k for k in known if flat and (re.sub(r"[^a-z0-9]", "", k.lower()) == flat or k.lower().endswith(product.lower()))]
        return same[0] if len(same) == 1 else product

    def over_budget(self, product: str) -> bool:
        """True when the product's latest known price times the wanted quantity is over the latest budget. Used only
        when the budget is plainly in the rows' own price unit (some row seen is priced within it)."""
        cap = self.unit_cap()                  # the latest budget, per unit when it is for the whole order
        price = self.row_field(product, r"price|cost")
        if cap is None or not isinstance(price, (int, float)) or isinstance(price, bool):
            return False
        prices = [p for i in self.rows for p in [self.row_field(i, r"price|cost")] if isinstance(p, (int, float)) and not isinstance(p, bool)]
        if not any(p <= cap for p in prices):
            return False
        return price > cap + 1e-9

    def read_answer(self, question: str, reply: Any) -> str:
        """One plain line for what a shopper's reply means, kept for the judge and shown to the loop."""
        if not isinstance(reply, str) or not reply.strip():
            return ""
        read = self.reader.json(ANSWER_GUIDE, json.dumps({"question": question, "reply": reply}, ensure_ascii=False), max_tokens=1500)
        meaning = str(read.get("meaning") or "unclear")
        line = f"{read.get('property') or question[:60]}: {meaning}" + (f" {read.get('value')}" if read.get("value") and meaning in ("required", "excluded") else "")
        self.meanings.append(line)
        return line

    def finish_args(self, name: str, purpose: str = "order") -> dict | None:
        """Arguments for a finishing call (purpose 'order' or 'decline'): the values the interface map gives for that
        purpose, a plain remark in text arguments, and None when a required choice is still unknown. A choice is never
        guessed: one tool may buy or decline by its value, and a wrong guess ends the task the wrong way."""
        tool = self.by_name.get(name)
        if tool is None:
            return None
        schema = schema_of(tool)
        props = schema.get("properties") or {}
        spec = self.iface.get(purpose) or {}
        mine = spec.get("tool") == name
        args: dict[str, Any] = dict(spec.get("args") or {}) if mine else {}
        text = "Placing the order for the chosen product." if purpose == "order" else "I could not find a product that meets every requirement."
        if mine and spec.get("remark") and spec["remark"] not in args:
            args[spec["remark"]] = text
        for key in schema.get("required") or []:
            if key in args:
                continue
            p = props.get(key) or {}
            if p.get("enum"):
                if len(p["enum"]) != 1:
                    return None
                args[key] = p["enum"][0]
            elif p.get("type") == "boolean":
                args[key] = True
            elif p.get("type") in ("integer", "number"):
                args[key] = 1
            else:
                args[key] = text
        args = fit_args(tool, args)
        return None if schema_problems(tool, args) else args

    def cannot_buy(self, lines: list[str], ids: list[str]) -> set[str]:
        """The ids among those given that a refused checkout's lines say can no longer be bought."""
        if not lines or not ids:
            return set()
        if self.elapsed() > FINISH_BY + 40:
            return set(ids)                            # too late for a model call: treat them as gone
        for line in lines:
            if line not in self.refusals:
                self.refusals.append(line)
        self.read_changes(ids)
        return {i for i in ids if i in self.refused_ids}

    def read_changes(self, ids: list[str]) -> None:
        """Read the change lines not read yet (one cheap call): items gone, items whose price or stock changed, a new
        budget, needs dropped or added. Kept for the judge and for the recovery after a refused checkout."""
        fresh = [line for line in self.refusals if line not in self.read_lines]
        ids = [i for i in dict.fromkeys(self.at_checkout + list(ids)) if i and i not in ("None", "null")]
        if not fresh or self.elapsed() > FINISH_BY + 40:
            return
        self.read_lines.update(fresh)
        read = self.reader.json(CHANGE_GUIDE, json.dumps({"lines": fresh, "ids": ids}, ensure_ascii=False), max_tokens=1500)
        gone = {str(i) for i in read.get("unavailable") or [] if str(i) in ids}
        self.refused_ids |= gone
        self.recheck |= {str(i) for i in read.get("recheck") or [] if str(i) in ids and str(i) not in gone}
        budget = read.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.cap_now = float(budget)
            self.changes.append(f"the budget is now at most {budget}")
        for need in read.get("dropped") or []:
            if isinstance(need, str) and need.strip():
                self.changes.append(f"no longer required: {need}")
                self.dropped = True
        for need in read.get("added") or []:
            if isinstance(need, str) and need.strip():
                self.changes.append(f"now required: {need}")
        if gone:
            self.changes.append("can no longer be bought: " + ", ".join(sorted(gone)))
        self.log.append({"changes": read})

    def look_again(self, product: str) -> None:
        """Open a product again after its price or stock changed (one step), so the change is seen before the next
        order and the judge reads the current price."""
        reader = self.tool_named("item_page") or self.tool_named("properties")
        self.recheck.discard(product)
        if not reader or self.world.finished:
            return
        tool = self.by_name[reader]
        key = next(iter(schema_of(tool).get("required") or list((schema_of(tool).get("properties") or {}))), None)
        if key is None:
            return
        args = fit_args(tool, {key: product})
        if schema_problems(tool, args):
            return
        result = self.world.call(reader, args)
        self.note(reader, args, result)
        self.log.append({"step": self.world.step, "call": {"name": reader, "args": args}, "result": result, "by": "recover"})
        if not is_fault(result.get("observation")):
            self.opened[(reader, json.dumps(args, sort_keys=True))] = shown(result.get("observation"), long=True)

    # -- bookkeeping of what a call did
    def note(self, name: str, args: dict, result: dict) -> None:
        self.watch(name, result)
        role = self.role(name)
        if (role == "checkout" or name == (self.iface.get("order") or {}).get("tool")) and self.cart:
            self.at_checkout = list(self.cart)
        observation = result.get("observation")
        if role == "item_page" and isinstance(core(observation), dict):
            own = one_id(core(observation), ids_in(core(observation), self.fields.get("id")), self.fields.get("id"))
            if own:                                                       # a full view carries the row fields too
                self.rows[own] = json.dumps({k: v for k, v in core(observation).items() if not (isinstance(v, str) and len(v) > 160)},
                                                    ensure_ascii=False, separators=(",", ":"))
        if role == "finder":
            result = core(observation)
            listed = result_rows(result)
            if listed and not self.fields_tried:
                self.map_fields(listed)
            for row in listed:
                own = one_id(row, ids_in(row, self.fields.get("id")), self.fields.get("id")) if isinstance(row, dict) else None
                if own:
                    self.rows[own] = json.dumps({k: v for k, v in row.items() if not (isinstance(v, str) and len(v) > 160)},
                                                        ensure_ascii=False, separators=(",", ":"))
        if role == "ask_shopper":
            self.questions += 1
            question = next((str(v) for v in args.values() if isinstance(v, str)), "")
            self.answers.append((question, observation if isinstance(observation, str) else json.dumps(observation)[:300]))
        if role in ("item_page", "properties") and not is_fault(observation):
            self.recheck -= {str(x) for v in args.values() for x in (v if isinstance(v, list) else [v]) if isinstance(x, (str, int))}   # seen now
            self.inspected |= {str(v) for v in args.values() if isinstance(v, (str, int))}
            self.inspected |= {str(x) for v in args.values() if isinstance(v, list) for x in v if isinstance(x, (str, int))}
            asked = [str(x) for v in args.values() for x in (v if isinstance(v, list) else [v]) if isinstance(x, (str, int))]
            self.props.update(product_properties(observation, role, asked, self.fields.get("id")))
            own = {" ".join(w for w in re.split(r"[._\s]+", str(k).lower()) if w) for k in self.fields.values()}
            for found in listed_properties(observation, role):
                self.pages_read += 1
                for name, value in found.items():
                    if name in own:
                        continue                       # the store's title / price / stock / id / category field
                    self.traits.setdefault(name, set()).add(value)
                    self.trait_count[name] = self.trait_count.get(name, 0) + 1
        if role == "cart_change" and not is_fault(observation):
            entry = self.cart_tools.get("add") or {}
            mapped = name == entry.get("tool") and entry.get("id") in args
            product = str(args[entry["id"]]) if mapped else next((str(v) for k, v in args.items() if isinstance(v, str) and re.search(r"id|product|item|sku|target|chosen", k.lower())), None)
            if mapped and entry.get("act") and entry["act"] in args:
                removing = args.get(entry["act"]) != entry.get("add")
            elif mapped and entry.get("qty") in args and args.get(entry["qty"]) == 0:
                removing = True                        # a quantity set to 0 takes the line out
            else:
                removing = any(re.search(r"remove|discard|delete|drop|take", str(v).lower()) for v in args.values() if isinstance(v, str))
            quantity = next((int(v) for k, v in args.items() if isinstance(v, int) and not isinstance(v, bool)), 1)
            if product:
                if removing:
                    self.cart.pop(product, None)
                else:
                    self.cart[product] = quantity
        if role == "cart_review" and isinstance(observation, dict):
            seen = ids_in(observation, self.fields.get("id"))
            if seen or not self.cart:
                self.cart = {i: self.cart.get(i, 1) for i in seen} if seen else {}
        for line in notices(observation):
            if line not in self.refusals:
                self.refusals.append(line)

    # -- guards: return a note to the model instead of spending a step, or None to let the call through
    def guard(self, name: str, args: dict) -> str | None:
        role = self.role(name)
        tool = self.by_name.get(name)
        if tool is None:
            return f"[guard] there is no tool named {name!r}; use one of: {', '.join(self.by_name)}"
        problems = schema_problems(tool, args)
        if problems:
            return (f"[guard] {name} was not sent (no step used): {'; '.join(problems)}. Use exactly the argument names "
                    "and allowed values of its schema.")
        purpose = self.finish_purpose(name, args) if role in ("checkout", "decline") or name in (
            (self.iface.get("order") or {}).get("tool"), (self.iface.get("decline") or {}).get("tool")) else None
        q = self.budget_q()
        if role == "ask_shopper" and q is not None and self.questions >= q:
            return "[guard] the question budget is used up; do not message the shopper again. Decide from what you know."
        if role == "ask_shopper":
            question = next((str(v) for v in args.values() if isinstance(v, str)), "")
            if question not in self.redirected:
                if not self.traits and self.tool_named("properties") and self.steps_left() > self.reserve() + 4:
                    self.redirected.add(question)
                    return ("[guard] open the property lists of a few candidates before asking: the requirements a shopper "
                            "holds back are among the properties listings state, so the property names tell you what to ask.")
                left = self.open_traits()
                if left and not any(names_property(question, n, str(self.plan.get("item") or "")) for n in set(self.traits) | set(left)):
                    self.redirected.add(question)
                    return ("[guard] that question names no property the listings state. A shopper holds back only "
                            "properties listings have, so ask about one of these by its name instead (one per message, "
                            "the ones the request hints at first): " + ", ".join(left[:10]))
        if role in ("item_page", "properties"):
            seen = self.opened.get((name, json.dumps(args, sort_keys=True)))
            if seen is not None and not ({str(v) for v in args.values() if isinstance(v, (str, int))} & self.recheck):
                return "[already opened earlier, no step used] " + seen
        if role in ("finder", "item_page", "properties", "ask_shopper", "other") and self.steps_left() <= self.reserve():
            return ("[guard] only " + str(self.steps_left()) + " steps are left: finish now. If an acceptable product is "
                    "known, put it in the cart and check out; otherwise decline.")
        entry = self.cart_tools.get("add") or {}
        if role == "cart_change" and entry.get("act") and name == entry.get("tool") and entry["act"] in args:
            adding = args.get(entry["act"]) == entry.get("add")
        else:
            adding = role == "cart_change" and not any(re.search(r"remove|discard|delete|drop", str(v).lower()) for v in args.values() if isinstance(v, str))
        if (adding or purpose in ("order", "decline")) and self.opened and self.cart_tools.get("add") and not self.commit_failed:
            verdict = self.decide()
            choice = verdict.get("choice")
            choice = str(choice) if choice not in (None, "", "null") else None
            for _ in range(3):
                if not (choice and self.wants_cheapest() and choice not in self.cheaper_checked and self.check_cheaper(choice)):
                    break
                verdict = self.decide()                 # cheaper candidates were opened: judge again (and check below it)
                choice = verdict.get("choice")
                choice = str(choice) if choice not in (None, "", "null") else None
            if choice and self.wants_cheapest() and not self.explored and self.steps_left() > self.reserve() + 8 and self.explore():
                verdict = self.decide()                 # other searches found cheaper candidates: judge again
                choice = verdict.get("choice")
                choice = str(choice) if choice not in (None, "", "null") else None
            if not choice and self.look_further():
                verdict = self.decide()                 # more of the narrow search was read and opened: judge again
                choice = verdict.get("choice")
                choice = str(choice) if choice not in (None, "", "null") else None
            if choice:
                return self.commit(choice)
            if purpose != "decline" and self.steps_left() > self.reserve():
                rejected = verdict.get("rejected") or {}
                return ("[judge] no opened product meets every requirement" + (f" ({'; '.join(f'{k}: {v}' for k, v in list(rejected.items())[:4])})" if rejected else "")
                        + f". Look further: {verdict.get('missing') or 'search with other product words'}. Decline only if nothing fits.")
        if purpose == "order":
            ruled_out = {i: why for i in self.cart for why in [self.contradicted(i)] if why}
            if ruled_out:
                return ("[guard] not ordered (no step used): the store will not count " + "; ".join(f"{i} ({why})" for i, why in ruled_out.items())
                        + ". Take it out and order a product whose own listing states every requirement, or decline if none does.")
            unseen = [i for i in self.cart if i not in self.inspected]
            if unseen and name + ",".join(unseen) not in self.checkout_guarded and self.steps_left() > self.reserve() - 1:
                self.checkout_guarded.add(name + ",".join(unseen))
                return (f"[guard] the cart holds {unseen}, whose properties you have not opened. Open its property list "
                        "(or full view) and confirm every requirement first, then check out.")
            if not self.cart and self.tool_named("cart_change"):
                return "[guard] the cart is empty: add the chosen product first."
        return None

    @staticmethod
    def pruned(messages: list[dict]) -> list[dict]:
        """The conversation with all but the latest observations cut short, to keep each call small and cheap. The cut
        moves forward four observations at a time, so the start of the conversation stays the same for several calls
        and the provider's prompt cache keeps serving it."""
        tool_at = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        old = set(tool_at[:max(0, len(tool_at) - KEEP_RECENT) // 4 * 4])
        out = []
        for i, m in enumerate(messages):
            if i in old and len(str(m.get("content") or "")) > OLD_CHARS:
                m = dict(m, content=str(m["content"])[:OLD_CHARS] + "...(older result cut)")
            out.append(m)
        return out

    def resync_cart(self) -> None:
        """Read the real cart with the cart-view tool (one step), so the record matches it after a removal that did
        not go through."""
        name = self.tool_named("cart_review")
        tool = self.by_name.get(name or "")
        if not tool or self.steps_left() <= 2 or schema_problems(tool, {}):
            return
        result = self.world.call(name, {})
        self.note(name, {}, result)
        self.log.append({"step": self.world.step, "call": {"name": name, "args": {}}, "result": result, "by": "resync"})

    def sets_quantity(self, entry: dict) -> bool:
        """True when a cart tool sets a line's quantity outright rather than adding: no add/remove switch, and its
        quantity may be 0 or its description says it sets the quantity. Removing then means quantity 0."""
        tool = self.by_name.get(entry.get("tool") or "")
        if not tool or entry.get("act") or not entry.get("qty"):
            return False
        spec = (schema_of(tool).get("properties") or {}).get(entry["qty"]) or {}
        text = str((tool.get("function") or {}).get("description") or "").lower()
        return spec.get("minimum") == 0 or bool(re.search(r"\bsets?\b.*\bquantit|\bto 0\b|zero (removes|takes)", text))

    def cart_call(self, kind: str, product: str, quantity: int = 1) -> dict | None:
        entry = self.cart_tools.get(kind)
        if not entry or not entry.get("id"):
            return None
        args: dict[str, Any] = {entry["id"]: product}
        if entry.get("act"):
            args[entry["act"]] = entry["add"] if kind == "add" else entry["remove"]
        if kind == "remove" and self.sets_quantity(entry):
            args[entry["qty"]] = 0                     # this tool sets the quantity outright: 0 takes the line out
        elif entry.get("qty") and (kind == "add" or entry["qty"] in (schema_of(self.by_name[entry["tool"]]).get("required") or [])):
            args[entry["qty"]] = quantity if kind == "add" else max(1, int(self.cart.get(product) or 1))
        tool = self.by_name[entry["tool"]]
        args = fit_args(tool, args)
        problems = schema_problems(tool, args)
        if problems:
            self.commit_failed = True
            self.log.append({"guard": f"[commit] no valid {entry['tool']} call could be built: {problems}", "call": {"name": entry["tool"], "args": args}})
            return {"observation": {"error": "not sent", "detail": problems}}
        result = self.world.call(entry["tool"], args)
        self.note(entry["tool"], args, result)
        self.log.append({"step": self.world.step, "call": {"name": entry["tool"], "args": args}, "result": result, "by": "commit"})
        if is_fault(result.get("observation")):
            self.commit_failed = True          # the world refused a code-built call: the model makes its own from now on
        return result

    def commit(self, choice: str) -> str:
        """Order exactly the chosen product; after a refused checkout, read what changed, look again at an item whose
        price or stock changed, judge again under the changes and order the pick (the same item once seen again, or
        the next acceptable one), at most three checkouts in one go and never the same item more than twice."""
        message = ""
        for _ in range(3):
            message, retry = self.commit_once(choice)
            if not retry:
                return message
            nxt = self.after_refusal(choice)
            if not nxt:
                return message
            choice = nxt
        return message

    def after_refusal(self, refused: str) -> str | None:
        """The item to order next after a refused checkout, or None to hand back to the model: the lines are read, a
        changed item is opened again, and the judge picks under the changes."""
        if self.world.finished or self.steps_left() < 2 or self.elapsed() > FINISH_BY:
            return None
        self.read_changes(list(dict.fromkeys([refused] + list(self.cart))))
        if refused in self.recheck and refused not in self.refused_ids:
            self.look_again(refused)
        if self.dropped and self.wants_cheapest() and not self.widened:
            self.widen()                               # products the dropped requirement kept out may now be cheapest
        verdict = self.decide()
        first = str(verdict.get("choice")) if verdict.get("choice") not in (None, "", "null") else None
        if first and first not in self.refused_ids and self.wants_cheapest() and first not in self.cheaper_checked \
                and self.steps_left() > self.reserve() + 4 and self.check_cheaper(first):
            verdict = self.decide()                    # a cheaper fit may be among rows not opened yet: judge again
        picks = [str(i) for i in [verdict.get("choice")] + list(verdict.get("acceptable") or []) if i not in (None, "", "null")]
        if not [i for i in picks if i not in self.refused_ids and self.checkouts.get(i, 0) < 2] and self.look_further():
            verdict = self.decide()                    # more of the narrow search was read and opened: judge again
            picks = [str(i) for i in [verdict.get("choice")] + list(verdict.get("acceptable") or []) if i not in (None, "", "null")]
        for pick in dict.fromkeys(picks):
            if pick in self.refused_ids or self.checkouts.get(pick, 0) >= 2:
                continue
            if pick in self.recheck and self.steps_left() >= 4:
                self.look_again(pick)
            if self.steps_left() < (1 if pick in self.cart else 3):
                return None
            return pick
        return None

    def commit_once(self, choice: str) -> tuple[str, bool]:
        """One order attempt: other cart lines out, the product in at the wanted quantity, checkout. Returns what
        happened and whether the checkout was refused (so a recovery may follow)."""
        checkout = self.tool_named("checkout")
        if not checkout:
            return "[commit] this world has no checkout tool", False
        quantity = int(self.plan.get("quantity") or 1) if str(self.plan.get("quantity") or "1").isdigit() else 1
        for other in [i for i in self.cart if i != choice]:
            if self.steps_left() <= 1:
                break
            removed = self.cart_call("remove", other)
            if removed is None or is_fault(removed.get("observation")):
                # no way to remove by code: ordering now would buy the old line too, so the model fixes the cart
                self.commit_failed = True
                return (f"[commit] the cart still holds {other}, which must come out before ordering {choice}. "
                        "Remove it with the cart tool exactly as its schema asks, add the pick if needed, then order."), False
        if choice not in self.cart and self.steps_left() > 1:
            result = self.cart_call("add", choice, quantity) or {}
            observation = result.get("observation")
            if is_fault(observation):
                return (f"[commit] adding {choice} failed: {compact(observation)}. Make the cart call yourself, with "
                        "exactly the argument names and allowed values of the cart tool's schema, then order."), False
        if self.world.finished or self.steps_left() < 1:
            return "[commit] no steps left", False
        args = self.finish_args(checkout, "order")
        if args is None:
            self.commit_failed = True
            return (f"[commit] {choice} is in the cart; the order call needs a choice I cannot read from its schema. "
                    "Make the order call yourself, exactly as the schema asks."), False
        self.checkouts[choice] = self.checkouts.get(choice, 0) + 1
        result = self.world.call(checkout, args)
        self.note(checkout, args, result)
        self.log.append({"step": self.world.step, "call": {"name": checkout, "args": args}, "result": result, "by": "commit"})
        if self.world.finished:
            return "[commit] the order went through", False
        if choice in self.cannot_buy(notices(result.get("observation")) or self.refusals[-3:], list(dict.fromkeys([choice] + list(self.cart)))):
            self.refused_ids.add(choice)
            if self.steps_left() > 2:
                removed = self.cart_call("remove", choice) or {}
                if not isinstance(removed.get("observation"), dict) or is_fault(removed.get("observation")):
                    self.resync_cart()             # the item may still be in the real cart: never assume it left
        return ("[commit] the checkout was refused: " + compact(result.get("observation"))
                + (" What changed: " + "; ".join(self.changes[-4:]) + "." if self.changes else "")
                + " Update what is wanted from these lines and continue; a refused item can no longer be bought."), True

    # -- the finish the agent makes by itself when steps or time run short
    def finish_now(self) -> None:
        """Finish by itself: check out the cart if it holds something, else decline; a refused finish (an event
        fires at the first attempt) is tried again while steps remain."""
        checkout, decline = self.tool_named("checkout"), self.tool_named("decline")
        if self.opened and self.cart_tools.get("add") and self.steps_left() >= 2 and not self.world.finished:
            # the judge's pick of the opened products is ordered before any decline
            verdict = self.decide()
            picks = [str(i) for i in [verdict.get("choice")] + list(verdict.get("acceptable") or []) if i not in (None, "", "null")]
            for choice in dict.fromkeys(picks):        # the pick first, then the other acceptable ones in turn
                if self.world.finished or self.steps_left() < 2:
                    break
                if choice in self.refused_ids:
                    continue
                self.log.append({"finish_now": "commit", "choice": choice})
                self.commit(choice)                    # a refused checkout is recovered inside, a bounded number of times
                if self.world.finished:
                    return
        for _ in range(3):
            if self.world.finished or self.steps_left() <= 0:
                return
            for item in [i for i in self.cart if i in self.recheck]:
                if self.steps_left() >= 2:
                    self.look_again(item)                  # a changed item is seen before the next order
            stuck = [i for i in self.cart if i in self.refused_ids or self.contradicted(i)]
            if stuck and len(stuck) < len(self.cart) and self.steps_left() >= 1 + len(stuck) and self.cart_tools.get("remove"):
                for item in stuck:
                    self.cart_call("remove", item)     # out first: an order holding a refused item is refused again
                continue
            args = self.finish_args(checkout, "order") if self.cart and checkout and not stuck else None
            if args is not None:
                result = self.world.call(checkout, args)
                self.note(checkout, args, result)
                self.log.append({"step": self.world.step, "call": {"name": checkout, "args": args}, "result": result, "by": "finish_now"})
                gone = set() if self.world.finished else self.cannot_buy(notices(result.get("observation")), list(self.cart))
                if gone:
                    self.refused_ids |= gone
                    self.cart = {k: v for k, v in self.cart.items() if k not in gone}   # those can no longer be bought
                continue
            args = self.finish_args(decline, "decline") if decline else None
            if args is not None:
                result = self.world.call(decline, args)
                self.note(decline, args, result)
                self.log.append({"step": self.world.step, "call": {"name": decline, "args": args}, "result": result, "by": "finish_now"})
            else:
                return

    # -- the episode
    def run(self) -> list[dict]:
        query = str(self.policy.get("query") or "")
        if self.policy.get("finished"):
            return []
        self.roles, self.iface = classify(self.tools, self.reader)
        self.cart_tools = cart_shape(self.tools, self.roles, self.iface)
        self.plan = read_plan(query, self.world.max_steps, self.planner)
        opening = self.policy.get("opening")
        brief = {k: self.plan.get(k) for k in ("item", "quantity", "requirements", "price_cap", "price_unit", "price_rule",
                                               "question_budget", "step_budget", "stock_rule", "decline_rule", "procedure",
                                               "rank_by", "reasons", "likely_hidden", "notes")}
        roles = {n: r for n, r in self.roles.items()}
        scouted = self.scout()
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"SHOPPER'S REQUEST:\n{query}\n\n"
                                        + (f"OPENING OBSERVATION:\n{compact(opening)}\n\n" if opening is not None else "")
                                        + f"PLAN (read from the request; check it against the request):\n{json.dumps(brief, ensure_ascii=False)}\n\n"
                                        + f"TOOL ROLES: {json.dumps(roles)}\n\n"
                                        + (f"HOW THIS STORE'S CART AND FINISH WORK (read from the schemas): {json.dumps(self.iface, ensure_ascii=False)}\n\n" if self.iface else "")
                                        + (f"SEARCH ALREADY DONE FOR YOU (these calls used steps):\n{scouted}\n\n" if scouted else "")
                                        + f"{self.status()}\nStart."},
        ]
        idle = blocked = 0                                # blocked: guard notes in a row with no step taken
        while not self.world.finished:
            if (self.elapsed() > FINISH_BY or self.steps_left() <= 2 or (self.reader.spent + self.judge.spent) > SPEND_LIMIT
                    or blocked >= 4 or self.errors_in_row >= ERRORS_IN_A_ROW):
                self.finish_now()
                break
            reply = self.reader.chat(self.pruned(messages), tools=self.endgame_tools())
            if reply is None:
                self.finish_now()
                break
            calls = reply.get("tool_calls") or []
            content = reply.get("content") or ""
            if not calls:
                idle += 1
                messages += [{"role": "assistant", "content": content or "(no tool call)"},
                             {"role": "user", "content": "Answer with exactly one tool call. " + self.status()}]
                if idle > 2:
                    self.finish_now()
                    break
                continue
            idle = 0
            # one reply may carry several read-only calls (searches, views, property lists); they run back to back,
            # each its own step. A question, a cart change or a finish runs alone.
            readonly = [c for c in calls[:3] if self.role((c.get("function") or {}).get("name") or "") in ("finder", "item_page", "properties")]
            batch = readonly if len(readonly) == len(calls[:3]) and len(readonly) > 1 else [calls[0]]
            if len(batch) > 1:
                messages.append({"role": "assistant", "content": content, "tool_calls": batch})
                went, refused = False, False
                for call in batch:
                    name = (call.get("function") or {}).get("name") or ""
                    try:
                        raw = (call.get("function") or {}).get("arguments") or "{}"
                        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    except (ValueError, TypeError):
                        args = {}
                    tool = self.by_name.get(name)
                    args = fit_args(tool, args) if tool else args
                    stop = self.guard(name, args)
                    if stop:
                        messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": stop})
                        refused = refused or stop.startswith("[guard]")
                        continue
                    went = True
                    result = self.world.call(name, args)
                    self.note(name, args, result)
                    self.log.append({"step": self.world.step, "call": {"name": name, "args": args}, "result": result})
                    text = shown(result.get("observation"))
                    if self.role(name) in ("item_page", "properties") and not is_fault(result.get("observation")):
                        self.opened[(name, json.dumps(args, sort_keys=True))] = shown(result.get("observation"), long=True)
                    if self.role(name) == "finder":
                        text += search_note(tool, result.get("observation"))
                    messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": text})
                    if self.world.finished:
                        break
                messages[-1]["content"] += "\n" + self.status()
                blocked = 0 if went else blocked + int(refused)  # a batch counts once, and only for real blocks
                continue
            call = calls[0]                                     # one step per turn
            name = (call.get("function") or {}).get("name") or ""
            try:
                raw = (call.get("function") or {}).get("arguments") or "{}"
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (ValueError, TypeError):
                args = {}
            tool = self.by_name.get(name)
            args = fit_args(tool, args) if tool else args
            messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
            stop = self.guard(name, args)
            if stop:
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": stop + "\n" + self.status()})
                self.log.append({"guard": stop, "call": {"name": name, "args": args}})
                blocked = blocked + 1 if stop.startswith("[guard]") else 0     # real blocks only
                continue
            blocked = 0
            result = self.world.call(name, args)
            self.note(name, args, result)
            self.log.append({"step": self.world.step, "call": {"name": name, "args": args}, "result": result})
            text = shown(result.get("observation"))
            if self.role(name) in ("item_page", "properties") and not is_fault(result.get("observation")):
                self.opened[(name, json.dumps(args, sort_keys=True))] = shown(result.get("observation"), long=True)
            if self.role(name) == "finder":
                text += search_note(tool, result.get("observation"))
            if self.role(name) == "ask_shopper":
                meaning = self.read_answer(next((str(v) for v in args.values() if isinstance(v, str)), ""), reply_text(result.get("observation")))
                if meaning:
                    text += f"\n[reading of that reply] {meaning}"
            if isinstance(result.get("observation"), dict) and result["observation"].get("refusal"):
                text += "\n[note] that finishing call was refused and the session goes on: read the refusal lines above (a requirement may have changed, or the item can no longer be bought) and continue."
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": text + "\n" + self.status()})
        self.log.append({"totals": {"seconds": round(self.elapsed(), 1), "loop_usd": round(self.reader.spent, 4), "judge_usd": round(self.judge.spent, 4),
                                    "steps": self.world.step, "questions": self.questions, "finished": self.world.finished}})
        return self.log


def agent_main(problem_data: dict) -> list[dict]:
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception as exc:  # noqa: BLE001 - an error must not leave the episode unfinished when a finish is still possible
        try:
            if not episode.world.finished:
                episode.finish_now()
        except Exception:  # noqa: BLE001
            pass
        return episode.log + [{"error": f"{type(exc).__name__}: {exc}"}]


__all__ = ["agent_main"]
