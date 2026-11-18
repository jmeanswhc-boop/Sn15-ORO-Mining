"""Terry: a shopping assistant for ORO Bench worlds.

A strong model reads the request once and writes a brief (items, limits, what to ask).
A light model works the world's own tools step by step, guided by the brief.
Before the episode is closed, the strong model checks the cart against everything
learned so far, and the light model fixes what it finds.
"""

from __future__ import annotations

import copy
import json
import random
import re
import threading
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
LIGHT_SPARE = "openai/gpt-6-luna"
HEAVY = "anthropic/claude-sonnet-5.5"
HEAVY_SPARE = "openai/gpt-6-sol"

_proxy = ProxyClient(timeout=150, max_retries=2)

MAX_REVIEWS = 2
MAX_WAITS = 3
END_RESERVE = 3
SOFT_SPEND = 0.038
HARD_SPEND = 0.055

_spend = threading.local()


def _spent() -> float:
    return getattr(_spend, "usd", 0.0)


def _charge(model: str, data: dict) -> None:
    usage = data.get("usage") or {}
    cost = usage.get("cost")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        heavy = model not in (LIGHT, LIGHT_SPARE)
        cost = ((usage.get("prompt_tokens") or 0) * (2e-6 if heavy else 1e-7) +
                (usage.get("completion_tokens") or 0) * (1e-5 if heavy else 5e-7))
    _spend.usd = _spent() + float(cost)


# ---------------------------------------------------------------- inference

def _cacheable(messages: list[dict]) -> list[dict]:
    """Mark the system prompt and the newest message as cache breakpoints."""
    out = copy.deepcopy(messages)
    marks = [0] if out and out[0]["role"] == "system" else []
    if len(out) > 1:
        marks.append(len(out) - 1)
    for i in marks:
        m = out[i]
        c = m.get("content")
        if isinstance(c, str) and c:
            m["content"] = [{"type": "text", "text": c, "cache_control": {"type": "ephemeral"}}]
    return out


def _chat(models: list[str], messages: list[dict], *, tools=None, tool_choice=None,
          max_tokens: int = 1500, tries: int = 3) -> dict | None:
    for model in models:
        body: dict[str, Any] = {
            "model": model,
            "messages": _cacheable(messages) if model.startswith("anthropic/") else messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        thinking = True
        body["reasoning"] = {"effort": "low"}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto" if thinking and model.startswith("anthropic/") else (tool_choice or "required")
        attempt = 0
        while attempt < tries:
            res = _proxy.post_verbose("/inference/chat/completions", json_data=body)
            data = res.data
            if data and data.get("choices"):
                _charge(model, data)
                return data["choices"][0].get("message") or {}
            if _settling(res.error):
                # the account is busy, not spent: the same model works again once requests settle; every model
                # draws on the same account, so when the task's waits are used up there is no point in the spare
                if not _wait_settle():
                    return None
                continue
            status = (res.error or {}).get("status")
            if status in (400, 401, 402, 403, 404):
                break
            attempt += 1
            if attempt < tries:
                time.sleep(2 + 3 * attempt + random.uniform(0, 2))
    return None


def _settling(error: Any) -> bool:
    """A refusal that asks to retry once in-flight requests settle (a busy account, not a spent one)."""
    text = str(error or "").lower()
    return "in_flight" in text or "in-flight" in text


def _wait_settle() -> bool:
    """Wait for a busy account, at most MAX_WAITS times per task (a task has a few minutes in all)."""
    waits = getattr(_spend, "waits", 0)
    if waits >= MAX_WAITS:
        return False
    _spend.waits = waits + 1
    time.sleep(5 + 5 * _spend.waits + random.uniform(0, 2))
    return True


def _chat_forced(messages: list[dict], tools: list) -> dict | None:
    """A tool call with no thinking, for when the light model keeps answering in prose."""
    body = {"model": LIGHT, "messages": _cacheable(messages), "max_tokens": 1200, "temperature": 0,
            "reasoning": {"enabled": False}, "tools": tools, "tool_choice": "required"}
    for attempt in range(2):
        res = _proxy.post_verbose("/inference/chat/completions", json_data=body)
        if res.data and res.data.get("choices"):
            _charge(LIGHT, res.data)
            return res.data["choices"][0].get("message") or {}
        if attempt == 1:
            break
        if _settling(res.error):
            if not _wait_settle():
                break
        else:
            time.sleep(2 + random.uniform(0, 2))
    return None


def _json_from(text: str) -> dict | None:
    """The last complete JSON object in a reply (a model that corrects itself writes the right one last)."""
    if not text:
        return None
    found = None
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            break
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                found = obj
        except ValueError:
            pass
        start = text.find("{", end + 1)
    return found


def _ask_json(models: list[str], system: str, user: str, max_tokens: int = 2500) -> dict | None:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for _ in range(2):
        reply = _chat(models, msgs, max_tokens=max_tokens)
        if reply is None:
            return None
        parsed = _json_from(reply.get("content") or "")
        if parsed is not None:
            return parsed
        msgs = msgs + [{"role": "assistant", "content": reply.get("content") or "(empty)"},
                       {"role": "user", "content": "Reply with the JSON object only."}]
    return None


# ---------------------------------------------------------------- tool schemas

def _schema(tool: dict) -> dict:
    return (tool.get("function") or tool).get("parameters") or {}


def _tool_name(tool: dict) -> str:
    return (tool.get("function") or tool).get("name", "")


def _closest(value: Any, options: list) -> Any:
    """The schema option meant by value, or None when nothing resembles it."""
    low = str(value).strip().lower()
    for o in options:
        if str(o).lower() == low:
            return o
    for o in options:
        ol = str(o).lower()
        if low and ol and (low in ol or ol in low):
            return o
    words = set(re.findall(r"[a-z0-9]+", low))
    best, score = None, 0
    for o in options:
        sc = len(words & set(re.findall(r"[a-z0-9]+", str(o).lower())))
        if sc > score:
            best, score = o, sc
    return best


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value == value and abs(value) < 1e15 else None
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(value or ""))
    return float(m.group(0).replace(",", ".")) if m else None


def _coerce(value: Any, spec: Any) -> Any:
    """Fit a value to its schema; None when it cannot be made to fit."""
    if not isinstance(spec, dict):
        return value
    kind = spec.get("type")
    if isinstance(spec.get("enum"), list) and spec["enum"]:
        return _closest(value, spec["enum"])
    if kind in ("integer", "number"):
        v = _number(value)
        if v is None:
            return None
        if "minimum" in spec and isinstance(spec["minimum"], (int, float)):
            v = max(v, spec["minimum"])
        if "maximum" in spec and isinstance(spec["maximum"], (int, float)):
            v = min(v, spec["maximum"])
        return int(round(v)) if kind == "integer" else v
    if kind == "boolean":
        return bool(value) if not isinstance(value, str) else value.strip().lower() in ("true", "yes", "1")
    if kind == "string":
        s = value if isinstance(value, str) else json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        if isinstance(spec.get("maxLength"), int):
            s = s[: spec["maxLength"]]
        if spec.get("minLength") and not s:
            s = "-"
        return s
    if kind == "array":
        if isinstance(value, str) and value.strip().startswith("["):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        items = value if isinstance(value, list) else [value]
        sub = spec.get("items") or {}
        out = [x for x in (_coerce(v, sub) for v in items) if x is not None]
        if isinstance(spec.get("maxItems"), int):
            out = out[: spec["maxItems"]]
        return out
    if kind == "object":
        return _coerce_args(value if isinstance(value, dict) else {}, spec)
    return value


def _coerce_args(args: Any, schema: Any) -> dict:
    if not isinstance(schema, dict):
        return dict(args) if isinstance(args, dict) else {}
    if not isinstance(args, dict):
        args = {}
    props = schema.get("properties") or {}
    if not isinstance(props, dict) or not props:
        return {} if schema.get("additionalProperties") is False else dict(args)
    required = schema.get("required") or []
    out = {}
    for k, v in args.items():
        if k in props:
            if v is None:
                continue
            c = _coerce(v, props[k])
            if c is None:
                spec = props[k] if isinstance(props[k], dict) else {}
                if k in required and isinstance(spec.get("enum"), list) and spec["enum"]:
                    c = spec["enum"][0]
                else:
                    continue
            out[k] = c
        elif schema.get("additionalProperties") is not False:
            out[k] = v
    return out


def _parse_args(call: dict) -> dict:
    raw = (call.get("function") or {}).get("arguments") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except ValueError:
        parsed = _json_from(raw)
        return parsed or {}


# ---------------------------------------------------------------- observations

def _shrink(obj: Any, limit: int, depth: int = 0) -> Any:
    if isinstance(obj, str):
        return obj if len(obj) <= limit else obj[:limit] + "…"
    if isinstance(obj, list):
        return [_shrink(x, limit, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {k: _shrink(v, limit, depth + 1) for k, v in obj.items() if v not in (None, "", [], {})}
    return obj


def _id_of(d: dict) -> str | None:
    """The record's id: a field named id, or the only field whose name ends in id."""
    v = d.get("id")
    if isinstance(v, (str, int)) and not isinstance(v, bool) and len(str(v)) >= 1:
        return str(v)
    cands = [v for k, v in d.items() if isinstance(k, str) and k.lower().endswith("id")
             and isinstance(v, str) and len(v) >= 3]
    return cands[0] if len(cands) == 1 else None


def _code_like(v: Any) -> bool:
    return isinstance(v, str) and 4 <= len(v) <= 64 and " " not in v and bool(re.search(r"\d", v))


def _list_key(items: list) -> str | None:
    """In a list of records, the field that tells them apart: distinct id-named or code-like values in all."""
    if len(items) < 2 or not all(isinstance(x, dict) for x in items):
        return None
    keys = list(items[0])
    named = [k for k in keys if isinstance(k, str) and (k.lower() == "id" or k.lower().endswith("id"))]
    for k in named + [k for k in keys if k not in named]:
        vals = [x.get(k) for x in items]
        if len(set(map(str, vals))) != len(vals):
            continue
        if k in named and all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in vals):
            return k
        if all(_code_like(v) for v in vals):
            return k
    return None


def _listings_in(obj: Any, found: list, asked: set, page: bool = False) -> None:
    """Collect (listing id, record) pairs, whatever the world calls its fields."""
    if isinstance(obj, dict):
        lid = _id_of(obj)
        if not lid and page:
            lid = next((str(v) for v in obj.values() if isinstance(v, (str, int)) and str(v) in asked), None)
        if lid and len(obj) >= 2:
            found.append((lid, obj))
        for v in obj.values():
            if isinstance(v, (dict, list)):
                _listings_in(v, found, asked, page)
    elif isinstance(obj, list):
        key = _list_key(obj)
        for v in obj:
            if key and isinstance(v, dict):
                found.append((str(v[key]), v))
                for w in v.values():
                    if isinstance(w, (dict, list)):
                        _listings_in(w, found, asked, page)
            else:
                _listings_in(v, found, asked, page)


def _strings_in(obj: Any, out: set) -> None:
    if isinstance(obj, str):
        if len(obj) <= 80:
            out.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _strings_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _strings_in(v, out)


def _order_note(obs: Any) -> str:
    """Say when a result list is sorted by one of its numeric fields."""
    if not isinstance(obs, dict):
        return ""
    for v in obs.values():
        if isinstance(v, list) and len(v) >= 3 and all(isinstance(x, dict) for x in v):
            keys = [k for k in v[0] if all(isinstance(x.get(k), (int, float)) and not isinstance(x.get(k), bool)
                                           for x in v)]
            for k in keys:
                vals = [x[k] for x in v]
                if len(set(vals)) > 1 and all(a <= b for a, b in zip(vals, vals[1:])):
                    return f"(These results are sorted by {k}, lowest first.)"
    return ""


def _notes_in(obs: Any, listing_ids: set) -> list[str]:
    """Sentences a result carries outside any listing record, such as a word from the shopper."""
    out: list[str] = []

    def walk(o: Any, depth: int) -> None:
        if depth > 3 or not isinstance(o, dict):
            return
        if any(isinstance(v, (str, int)) and not isinstance(v, bool) and str(v) in listing_ids for v in o.values()):
            return
        for v in o.values():
            if isinstance(v, str) and len(v.split()) >= 5:
                out.append(v)
            elif isinstance(v, dict):
                walk(v, depth + 1)

    walk(obs, 0)
    return out


def _is_full_page(d: dict) -> bool:
    return any(isinstance(v, dict) and len(v) >= 2 for v in d.values())


def _compact_obs(obs: Any) -> str:
    """Short JSON for the light model: long text cut, empty fields dropped."""
    if isinstance(obs, dict):
        lists = [v for v in obs.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if lists:
            return json.dumps(_shrink(obs, 140), ensure_ascii=False)
    return json.dumps(_shrink(obs, 700), ensure_ascii=False)


# ---------------------------------------------------------------- prompts

BRIEF_SYSTEM = """You prepare a shopping assistant for one task in an online shop simulation.
You get the shopper's request and the shop's tools. Read the request with great care: it mixes small talk
with real requirements, limits and hints. Write a JSON brief, nothing else.

JSON fields:
- "analysis": 1-2 sentences on what the shopper really needs.
- "items": list, one per product to buy, each {"what": the kind of product in the shop's words, "category": the matching
  value from a tool's category list if the tools offer one (else null), "qty": integer count the shopper wants,
  "musts": every stated requirement for this product as short "property: wanted value" strings (keep numbers,
  units, ranges, "not X", "either A or B" exactly as stated, and use the shopper's own word for the value, not a
  paraphrase; the product category itself is not a must), "reference": a requirement whose value must be read
  from another listing's page (who sells it, which kind of product, which details identify it, which property to copy,
  and how it must relate: same, within N, at least...), or null}.
- "budget": {"amount": number or null, "unit": the unit the request uses, "applies_to": "total" or "each"}.
- "objective": how to choose among listings that meet everything (e.g. "cheapest total", "lowest unit price",
  "best match", or "any that fits").
- "skip_sold_out": true if out-of-stock listings are unacceptable (assume true unless the shopper says otherwise).
- "message_limit": the most messages the shopper will accept from the assistant, as an integer, or null if unstated.
- "question_cost": text of any stated cost per question or message, else null.
- "step_limit": the most tool calls allowed if the request states one, else null.
- "procedures": steps the shopper says must happen (e.g. confirm the total before ordering, tell them something),
  each with "needs_message": true if it means sending the shopper a message.
- "questions": the properties the request hints at without stating a value (an indirect remark on what they would
  notice, what is most important to them, or how they plan to use it), most telling first, as [{"item": product,
  "property": property name}, ...] with "hinted": true on each. Empty if there are no such hints.
- "tool_roles": map each tool name to one role: search, view, compare, message, cart_change, cart_set, cart_view,
  checkout, decline, finish (one tool that both orders and declines), other.
- "finish_modes": for each finish tool, {"<tool>": {"order": the argument values (for its choice-type arguments)
  that place the order, "decline": the values that close with no purchase}}, read from that tool's schema; {} if
  there is no finish tool.
- "probes": one first search per item, [{"item": product, "tool": search tool name, "args": arguments for that tool
  (use its category filter when it has one, and a short keyword: the product noun plus at most one stated
  value; leave out paging or page-token arguments)}].
- "plan": 2-3 short steps on how to find the right listing(s) with these tools."""

LOOP_SYSTEM = """You are a careful shop assistant working inside an online shop through its tools.
Every tool call is one step and steps are limited. Call exactly one tool per turn and write one short line
of reasoning before it.

How to work:
1. Requirements = everything the request states + every need the shopper told you in a reply. A listing qualifies
   only if its own page facts meet ALL of them. Read the attribute for the SAME property (same name or an obvious
   alias of it); a different property never stands in for it (if the wish is finish "matte", a texture of "smooth"
   says nothing). Values match by meaning (synonyms, unit conversions). Only when the page has no such attribute may
   the title or description decide. The listing must also be in the product category the shopper names: the
   listing's own category field (whatever the shop calls it) decides what kind of product it is, never its title
   or your own sense of it.
   Values must be the same value (same word, a true synonym, a spelling variant or a unit conversion); a value that
   is merely compatible, broader, stronger, weaker or related is not the same (wanted "rustic" is not met by
   "vintage"; wanted "lightly salted" is not met by "unsalted"). Assume listing
   prices are in the same unit as the shopper's budget unless the shop says otherwise. An exclusion ("not X") removes
   only listings whose value is X itself or a true synonym, not values that merely contain the word.
2. If a requirement refers to another listing ("the same X as the one sold by Y..."), find that listing, open its
   page, and if its page points you to another listing, follow it and use the final page's value.
3. Some shops split a listing over several page tools (for example one for its properties, one for price, stock and
   notes): before you choose a listing, look at it with each of them.
   Search with short queries (product noun + one or two wanted values). When a search matches very many listings,
   narrow the query with another wanted value instead of paging on; page at most a few times per query.
   When the shopper wants the cheapest, search with a few different wordings (the category's other names too), then
   open candidates in price order, cheapest first, until one meets everything. Use the category filter of a search
   tool when there is one. Results show summaries only: open the full page of promising candidates (or compare
   several at once if a tool allows) before deciding. Search more widely when the shopper wants the cheapest:
   check other queries or more result pages so you do not miss a cheaper qualifying listing. Notice how results
   are ordered (for example by price) and how many matched. A paging token must be copied exactly from
   an earlier result; never invent one.
4. Never pick a sold-out listing when the shopper rejects those. Respect the budget across all lines
   (price x quantity summed) and choose by the shopper's stated objective.
5. Put exactly the wanted quantity of each chosen listing in the cart, view the cart, check prices and stock there
   (things can change after you add), then close the episode with the purchase tool. If nothing in the shop can meet
   every requirement, close with no purchase and say briefly which requirement could not be met (no listing ids).
6. Questions to the shopper were already sent for you; their replies are in the notes. A reply that names a product
   and property applies to that product, even if the question named another. Only message the shopper
   again if a stated procedure requires it or a reply was unclear, and never beyond the message limit.
7. A thorough search usually takes most of the step budget: several searches (the category filter and more result
   pages too) and the property pages of many candidates. Keep the last 6 steps for cart, a last check and closing.
   Do not repeat a search or page you already have.
8. If a chosen listing sells out, or its price changes, first search for the same product by its title
   words: other sellers often list the identical item. Prefer such a twin when it meets everything; otherwise take
   the next best listing that meets everything.
9. A tool result may carry a sentence from the shopper (a new budget, a changed, added, loosened or dropped
   requirement). It is a real update and overrides the request: re-check your choice against it and change the cart
   if needed. A loosened or dropped requirement no longer has to be met.
10. Any price limit you pass to a tool uses the same unit as the prices the shop shows (if a listing shows 1440 for
   an item, a limit of nineteen and a half dollars is 1950, not 19.5)."""

REVIEW_SYSTEM = """You check a shop assistant's work before the episode is closed in an online shop simulation.
You get the shopper's request, a brief, the shopper's replies, the current cart, and the listings seen
(full pages for the ones opened). Decide whether closing now is right. Write JSON only:
{"table": one row per OPENED listing that could serve a requested product: {"item": the requested product it
   could serve, "id": listing id, "price": its price as a number, "in_stock": true/false/null, "meets_all": true only
   if its page facts meet every requirement and told need for that product (by the rules below), "why": the first
   unmet requirement or "ok"},
 "cheapest_wins": true if the shopper wants the lowest price or total,
 "verdict": "close" or "fix",
 "close_as": "order" or "decline",
 "fix": if verdict is fix: exact instructions (which listing ids to open, replace, remove, or which quantity to set),
 "reason": one sentence}
Rules: a shopper reply that is garbled, off the subject or not an answer adds no requirement. New words from the
shopper that came in tool results (a new budget, a changed, loosened or dropped
requirement) are real and override the request; a loosened or dropped requirement no longer has to be met. When a shopper's reply names a product and property, the need applies to the product the reply names, even if
the question named another. Every stated requirement and every need the shopper told must be met by page facts: the attribute for the
same property (same name or an obvious alias) must carry the wanted value (synonyms and unit conversions count); a
different, related property does not count; title or description decide only when the page lacks that attribute.
Each line belongs in the product category the shopper names: the listing's own category field (whatever the shop calls it) decides what
kind of product it is, never its title or your own sense of what it is. A value must be the same value (same word,
true synonym, spelling variant or unit conversion); merely compatible, broader or related values do not count
(wanted "rustic" is not met by "vintage"; wanted "lightly salted" is not met by "unsalted": a stronger, weaker or
neighbouring value is a different value). A page value that holds the wanted value as a whole word with a modifier
does meet it (wanted "green" is met by "dark green" or "green leather"). An exclusion ("not X") removes only listings whose value is X itself or a
true synonym; a different value that merely contains the word (such as "faux X" or "X-look") is not excluded.
When the listing's page has no attribute for a required property, its title and description decide; if they say
nothing about it either, that requirement does not rule the listing out (shops often show no such attribute at all).
When a property is unknown only because one of the shop's page tools was not used on that listing ("opened_with"),
ask in the fix to open it with that tool. Listing prices are in
the same unit as the budget unless the shop says otherwise. Judge only by the shop's data and the shopper's words,
not by what the product seems to be for. The cart must hold exactly the products asked for, one line per
requested product with the requested quantity; an extra line is wrong. Stock and price count as of the freshest
page or result seen after the last cart change. For an out-of-stock line, the fix should first look for the same
product from another seller (search by its title words). Out-of-stock lines are
wrong when the shopper rejects them. The order total must respect the budget. When the shopper wants the cheapest,
a cheaper listing that meets everything and was seen in results is a reason to fix (say which id). A decline is
right only if no listing seen can meet everything and a reasonable search was made. Ask for a fix only when it is
clearly needed and fits in the remaining steps; otherwise say close."""


# ---------------------------------------------------------------- environment

class Episode:
    def __init__(self, env: dict):
        self.binding = env["binding"]
        self.policy = env["policy_view"]
        self.tools = self.policy.get("tools") or []
        self.by_name = {_tool_name(t): t for t in self.tools}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dialogue: list[dict] = []
        self.listings: dict[str, dict] = {}
        self.viewed: set[str] = set()
        self.qa: list[tuple[str, str]] = []
        self.cart_obs: Any = None
        self.cart_dirty = False
        self.seen_strings: set[str] = set()
        self.notes: list[str] = []
        self.message_tools: set[str] = set()
        self.page_tools: set[str] = set()
        self.memo: dict[str, Any] = {}
        self.carted = False
        self.fresh: list[str] = []
        self.bool_pages: list[str] = []
        self.opened_by: dict[str, set] = {}

    def call(self, name: str, args: dict) -> Any:
        """One environment step; returns the observation."""
        self.fresh = []
        if self.finished:
            return {"error": "the episode has finished"}
        self.step += 1
        payload = {**self.binding, "step": self.step, "action": {"name": name, "args": args}}
        result = None
        refused = None
        for attempt in range(5):
            res = _proxy.post_verbose("/environment/call", json_data=payload)
            if res.data is not None:
                result = res.data
                break
            status = (res.error or {}).get("status")
            if status == 409:
                if attempt == 0:
                    self.step += 1
                    payload = {**payload, "step": self.step}
                    continue
                self.finished = True
                return {"error": "the episode refused further steps"}
            if status in (400, 401, 403, 404, 413, 422):
                refused = (res.error or {}).get("body")
                break
            time.sleep(2 + 2 * attempt + random.uniform(0, 2))
        if result is None:
            # the step was not taken: reuse its number for the next call
            self.step -= 1
            return {"error": "the shop did not accept that call", "detail": str(refused or "no answer")[:300]}
        self.finished = bool(result.get("finished"))
        obs = result.get("observation")
        self.dialogue.append({"role": "assistant", "tool_call": {"name": name, "args": args},
                              "environment_result": result})
        self._learn(obs, name, args)
        return obs

    def _learn(self, obs: Any, name: str = "", args: dict | None = None) -> None:
        _strings_in(obs, self.seen_strings)
        asked: set = set()
        _strings_in(args or {}, asked)
        is_page = name in self.page_tools
        if (is_page and isinstance(obs, dict) and name not in self.bool_pages
                and any(isinstance(v, bool) for v in obs.values())):
            self.bool_pages.append(name)
        found: list = []
        _listings_in(obs, found, asked, is_page)
        for lid, d in found:
            if self.page_tools:
                full = is_page and lid in asked
            else:
                full = _is_full_page(d)
            old = self.listings.get(lid)
            if old is None or full or not old.get("_full"):
                merged = dict(old or {})
                merged.update(d)
                merged["_full"] = full or bool(old and old.get("_full"))
                self.listings[lid] = merged
            if full:
                self.viewed.add(lid)
                self.opened_by.setdefault(lid, set()).add(name)
        if name not in self.message_tools:
            for n in _notes_in(obs, set(self.listings) | asked):
                if n not in self.notes:
                    self.notes.append(n)
                    self.fresh.append(n)

    @property
    def left(self) -> int:
        return self.max_steps - self.step


def _role_tools(brief: dict | None, ep: Episode) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    raw = (brief or {}).get("tool_roles") or {}
    if isinstance(raw, list):
        raw = {str(x.get("tool") or x.get("name")): x.get("role") for x in raw if isinstance(x, dict)}
    if not isinstance(raw, dict):
        return roles
    for name, role in raw.items():
        if name in ep.by_name and isinstance(role, str):
            roles.setdefault(role.strip().lower(), []).append(name)
    return roles


def _text_arg(tool: dict) -> str | None:
    props = _schema(tool).get("properties") or {}
    req = _schema(tool).get("required") or []
    strings = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v]
    for k in req:
        if k in strings:
            return k
    return strings[0] if strings else None


# ---------------------------------------------------------------- brief

def _tools_digest(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        f = t.get("function") or t
        props = (f.get("parameters") or {}).get("properties") or {}
        args = []
        for k, v in props.items():
            if v.get("enum"):
                args.append(f"{k} (one of: {', '.join(map(str, v['enum']))})")
            else:
                args.append(f"{k} ({v.get('type', 'any')})")
        lines.append(f"- {f.get('name')}: {f.get('description', '')} Args: {'; '.join(args) or 'none'}")
    return "\n".join(lines)


def _make_brief(ep: Episode, request: str) -> dict | None:
    user = (f"SHOPPER REQUEST:\n{request}\n\nSTEP LIMIT FROM THE SHOP: {ep.max_steps}\n\n"
            f"TOOLS:\n{_tools_digest(ep.tools)}")
    if ep.policy.get("opening") is not None:
        user += "\n\nOPENING NOTE FROM THE SHOP:\n" + json.dumps(ep.policy["opening"], ensure_ascii=False)
    return _ask_json([HEAVY, HEAVY_SPARE], BRIEF_SYSTEM, user, max_tokens=4000)


def _brief_text(brief: dict) -> str:
    keep = {k: brief.get(k) for k in ("items", "budget", "objective", "skip_sold_out", "message_limit",
                                       "question_cost", "procedures", "plan") if brief.get(k) is not None}
    return json.dumps(keep, ensure_ascii=False, indent=1)


def _message_limit(brief: dict | None) -> int:
    lim = (brief or {}).get("message_limit")
    try:
        return max(0, int(lim))
    except (TypeError, ValueError):
        return 3


def _reserved_messages(brief: dict | None) -> int:
    n = 0
    procs = (brief or {}).get("procedures")
    for p in procs if isinstance(procs, list) else []:
        if isinstance(p, dict) and p.get("needs_message"):
            n += 1
    return n


# ---------------------------------------------------------------- asking

QUESTION_SYSTEM = """You pick the NEXT property a shop assistant asks a shopper about. The shopper has wishes about
some properties of the products that are left unsaid; each message may ask about a single property, and messages are
limited. You see the request, the brief, hints from the request, the questions already sent with the shopper's
replies, and the properties the shop's own pages record. Write JSON only:
{"item": the product, named the way the request names it, "property": the property's name exactly as the shop's
 pages write it (one of their property keys, with underscores read as spaces), "why": a few words}
Messages that are not used are wasted, so always give a property while any unasked one is left.

How to read the replies so far:
- A reply that states a value settled that property: it is a real wish. Do not ask it again.
- A reply saying any choice is fine means that property is free. Do not ask it again.
- A reply saying they do not follow means the shopper has no wish about that property. Do not ask it again, under
  any name; pick a different property.
How to choose the next property:
- Choose among the properties the shop's pages record for that product: the shopper's unsaid wishes are about
  those. Among them, first one the request hints at (an indirect remark on what they would notice, what is most
  important to them, or how they plan to use it), then ones whose values differ between listings (they decide which
  listing is right) and that people usually choose that kind of product by. Name it exactly as the pages do, never
  in the shopper's own words.
- With several products, go to the product whose properties have been asked about least so far.
- Never a property whose value the request already states, nothing already asked (by meaning, not wording), and
  only properties of the product itself, not facts about the offer or the shop."""


def _has_props(obs: Any) -> bool:
    """A page that records a listing's properties: some field holds a map of several values."""
    if isinstance(obs, dict):
        return any(isinstance(v, dict) and len(v) >= 2 for v in obs.values()) or \
            any(_has_props(v) for v in obs.values() if isinstance(v, list))
    if isinstance(obs, list):
        return any(_has_props(x) for x in obs)
    return False


def _probe(ep: Episode, brief: dict | None, roles: dict, per_item: int = 3) -> list[str]:
    """One search per item, then the property pages of a few results, so the shop's own property names are known
    before any question is sent."""
    notes: list[str] = []
    if not brief:
        return notes
    views = [t for t in (roles.get("compare") or []) + (roles.get("view") or []) if t in ep.by_name]
    prop_tool = None
    probes = brief.get("probes") if isinstance(brief.get("probes"), list) else []
    for pr in probes[:3]:
        if not isinstance(pr, dict) or ep.finished or ep.left < END_RESERVE + 10:
            break
        name = pr.get("tool")
        if name not in ep.by_name:
            continue
        args = _coerce_args(pr.get("args") if isinstance(pr.get("args"), dict) else {}, _schema(ep.by_name[name]))
        obs = ep.call(name, args)
        notes.append(f"{name} {json.dumps(args, ensure_ascii=False)} -> {_compact_obs(obs)}")
        found: list = []
        _listings_in(obs, found, set())
        ids = [i for i in dict.fromkeys(i for i, _ in found) if i not in ep.viewed][:per_item]
        for n, lid in enumerate(ids):
            if ep.finished or ep.left < END_RESERVE + 10:
                break
            for t in ([prop_tool] if prop_tool else views[:2]):
                tool = ep.by_name[t]
                props = _schema(tool).get("properties") or {}
                arr = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array"]
                if arr:
                    vargs = _coerce_args({arr[0]: ids}, _schema(tool))
                else:
                    key = _text_arg(tool)
                    if not key:
                        continue
                    vargs = _coerce_args({key: lid}, _schema(tool))
                obs = ep.call(t, vargs)
                notes.append(f"{t} {json.dumps(vargs, ensure_ascii=False)} -> {_compact_obs(obs)}")
                if _has_props(obs):
                    prop_tool = t
                    break
            if prop_tool and arr:
                break  # a page tool that takes several ids opened them all at once
    return notes


def _page_digest(ep: Episode, limit: int = 14) -> str:
    """Each opened page as: its short fields, then its recorded properties."""
    rows = []
    for lid in list(ep.viewed)[-limit:]:
        d = ep.listings.get(lid) or {}
        short, props = [], []
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                props += [f"{pk}={str(pv)[:28]}" for pk, pv in v.items() if pv not in (None, "")]
            elif isinstance(v, str) and v and len(v) <= 90:
                short.append(f"{k}={v[:90]}")
        rows.append("- " + "; ".join(short)[:260] + " || " + "; ".join(props)[:700])
    return "\n".join(rows)


def _next_question(ep: Episode, brief: dict, request: str, stated: set, left: int) -> dict | None:
    history = "\n".join(f"Q: {q}\nA: {a}" for q, a in ep.qa) or "(none yet)"
    hints = [{"item": q.get("item"), "property": q.get("property")} for q in brief.get("questions") or []
             if isinstance(q, dict) and q.get("property")]
    user = (f"SHOPPER REQUEST:\n{request}\n\nBRIEF:\n{_brief_text(brief)}\n\nHINTS FROM THE REQUEST (in the "
            f"shopper's words): {json.dumps(hints, ensure_ascii=False)}\n\nALREADY STATED BY THE SHOPPER "
            f"(never ask): {json.dumps(sorted(stated), ensure_ascii=False)}\n\nQUESTIONS SENT AND REPLIES:\n"
            f"{history}\n\nSHOP PAGES SEEN:\n{_page_digest(ep)}\n\nMessages left after this one: {left - 1}.")
    got = _ask_json([LIGHT, HEAVY], QUESTION_SYSTEM, user, max_tokens=2500)
    prop = " ".join(str((got or {}).get("property") or "").replace("_", " ").split())
    if not prop:
        return None
    items = [str(i.get("what")) for i in brief.get("items") or [] if isinstance(i, dict) and i.get("what")]
    item = " ".join(str(got.get("item") or (items[0] if items else "item")).split())
    # one plain form for every question: the shop's own name for the property, and the product it is for
    got.update(property=prop, item=item, text=f"Which {prop} would you like the {item} to have?")
    return got


def _ask(ep: Episode, brief: dict | None, roles: dict, request: str) -> None:
    """Questions are sent singly, each chosen after reading the replies so far."""
    tools = roles.get("message") or []
    if not tools or not brief:
        return
    tool = ep.by_name[tools[0]]
    arg = _text_arg(tool)
    if not arg:
        return
    budget = _message_limit(brief) - _reserved_messages(brief)
    qs = brief.get("questions") if isinstance(brief.get("questions"), list) else []
    hinted = [q for q in qs if isinstance(q, dict) and q.get("hinted") and q.get("property")]
    if brief.get("question_cost"):
        budget = min(budget, len(hinted))
    budget = min(budget, ep.left - END_RESERVE - 6)
    stated = _stated_properties(brief)
    asked: set = set()
    tries = 0
    while len(ep.qa) < budget and not ep.finished and tries < budget + 3:
        tries += 1
        q = _next_question(ep, brief, request, stated, budget - len(ep.qa))
        if not q:
            break
        text = str(q.get("text")).strip()
        key = (str(q.get("item", "")).lower(), str(q.get("property", "")).lower())
        if not text or key in asked or _overlaps(key[1], stated):
            if key[1]:
                stated.add(key[1])
            continue
        asked.add(key)
        obs = ep.call(tools[0], _coerce_args({arg: text}, _schema(tool)))
        ep.qa.append((text, _reply_text(obs)))


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if len(w) > 2}


def _stated_properties(brief: dict | None) -> set[str]:
    out = set()
    items = (brief or {}).get("items")
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        musts = it.get("musts")
        for m in musts if isinstance(musts, list) else []:
            prop = str(m).split(":", 1)[0].strip().lower()
            if prop:
                out.add(prop)
    return out


def _overlaps(prop: str, stated: set[str]) -> bool:
    """True when the property is one the request already settles (same name, or a vaguer form of it)."""
    pw = " ".join(sorted(_words(prop)))
    for x in stated:
        xw = " ".join(sorted(_words(x)))
        if pw and xw and (pw == xw or set(pw.split()) <= set(xw.split())):
            return True
    return False


def _reply_text(obs: Any) -> str:
    if isinstance(obs, str):
        return obs
    if isinstance(obs, dict):
        strs = [v for v in obs.values() if isinstance(v, str)]
        if len(strs) == 1:
            return strs[0]
    return json.dumps(obs, ensure_ascii=False)


# ---------------------------------------------------------------- review

def _dossier(ep: Episode, request: str, brief: dict | None) -> str:
    full, brief_rows = [], []
    in_cart = set(_cart_ids(ep))
    for lid, d in ep.listings.items():
        clean = {k: v for k, v in d.items() if not k.startswith("_")}
        if d.get("_full"):
            if ep.opened_by.get(lid):
                clean["opened_with"] = sorted(ep.opened_by[lid])
            full.append((lid in in_cart, json.dumps(_shrink(clean, 180), ensure_ascii=False)))
        else:
            brief_rows.append(json.dumps(_shrink(clean, 60), ensure_ascii=False))
    # the cart's own pages always, then the most recently opened ones
    room = max(0, 10 - len(in_cart))
    others = [t for c, t in full if not c]
    full = [t for c, t in full if c] + (others[-room:] if room else [])
    parts = [f"SHOPPER REQUEST:\n{request}"]
    if brief:
        parts.append("BRIEF:\n" + _brief_text(brief))
    if ep.qa:
        parts.append("SHOPPER Q&A:\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in ep.qa))
    if ep.notes:
        parts.append("NEW WORDS FROM THE SHOPPER THAT CAME IN TOOL RESULTS (newest last; they override the request):\n" +
                     "\n".join(ep.notes))
    parts.append("CURRENT CART (latest view):\n" + json.dumps(ep.cart_obs, ensure_ascii=False))
    if len(ep.page_tools) > 1:
        parts.append("PAGE TOOLS IN THIS SHOP (each may show different parts of a listing): " +
                     ", ".join(sorted(ep.page_tools)))
    parts.append("OPENED LISTINGS (full pages):\n" + "\n".join(full))
    parts.append("OTHER LISTINGS SEEN IN RESULTS:\n" + "\n".join(brief_rows[-15:]))
    parts.append(f"STEPS LEFT: {ep.left}")
    return "\n\n".join(parts)


def _review(ep: Episode, request: str, brief: dict | None, closing: str) -> dict | None:
    user = _dossier(ep, request, brief) + f"\n\nTHE ASSISTANT NOW WANTS TO CLOSE AS: {closing}"
    return _ask_json([HEAVY, HEAVY_SPARE], REVIEW_SYSTEM, user, max_tokens=6000)


def _table_fix(verdict: dict, cart: list[str]) -> str | None:
    """When the shopper wants the lowest price: per product, the cheapest qualifying listing must be the one carted."""
    if not verdict.get("cheapest_wins") or not isinstance(verdict.get("table"), list):
        return None
    best: dict = {}
    item_of: dict = {}
    for row in verdict["table"]:
        if not isinstance(row, dict):
            continue
        lid, item = str(row.get("id") or ""), str(row.get("item") or "").strip().lower()
        price = _number(row.get("price"))
        item_of.setdefault(lid, item)
        if row.get("meets_all") is True and row.get("in_stock") is not False and price is not None and lid:
            if item not in best or price < best[item][1]:
                best[item] = (lid, price)
    moves = []
    for lid in cart:
        item = item_of.get(lid)
        if item in best and best[item][0] != lid:
            mine = next((_number(r.get("price")) for r in verdict["table"] if isinstance(r, dict)
                         and str(r.get("id")) == lid), None)
            if mine is None or best[item][1] < mine:
                moves.append(f"replace {lid} with {best[item][0]} (it meets everything and costs {best[item][1]:g})")
    for item, (lid, _) in best.items():
        if not any(item_of.get(c) == item for c in cart) and lid not in cart and cart:
            moves.append(f"the cart has no line for {item}: add {lid}")
    return "; ".join(moves) or None


DECIDE_SYSTEM = """You make the final choice for a shop assistant whose steps are almost used up, in an online
shop simulation. You get the shopper's request, a brief, the shopper's replies and every listing seen (full pages
for the ones opened). Choose from listings seen only. Write JSON only:
{"checks": short notes, "action": "order" or "decline", "lines": [{"id": listing id, "qty": count}],
 "reason": one sentence}
A need told in a reply applies to the product the reply names, even if the question named another. New words from the
shopper that came in tool results (new budget, changed, loosened or dropped requirement) are real and override the
request; a loosened or dropped requirement no longer has to be met.
Order only listings whose page facts meet every stated requirement and every need the shopper told (the attribute
for the same property must carry the wanted value; a related property does not count), in the category the shopper
names (the listing's own category field decides the kind of product, not its title), in stock when that
matters, within the budget, chosen by the shopper's objective. A value must be the same value (true synonyms and
unit conversions count; compatible, stronger, weaker or related values do not); a requirement the page contradicts is
not met; when the page has no attribute for it, the title or description decide, and if they say nothing it does not
rule the listing out (if a page tool was not yet used on a listing, list it in "open_first").
Prices are in the budget's unit unless the shop says otherwise. Decline only if every listing seen clearly fails a
requirement by its own page facts (a different value, another category, over the budget, sold out when that matters);
a requirement that is only unconfirmed (the page does not show it) is no reason to decline when a listing otherwise
fits: order the best such listing then.
If the shop refused a close (for example a listing went out of stock), choose the best remaining listing by the
shopper's objective; the same product from another seller (same title) is often listed, so put such twins seen in
results first, or name the search to run for them in "search_first" (a few title words). Add "open_first": up to 4 ids of listings seen in results that could beat your choice (for
example cheaper) but whose pages were not opened, so they can be checked before the order."""


def _decide(ep: Episode, request: str, brief: dict | None, why: str = "") -> dict | None:
    user = _dossier(ep, request, brief) + (f"\n\nWHY A DECISION IS NEEDED NOW: {why}" if why else "")
    return _ask_json([HEAVY, HEAVY_SPARE], DECIDE_SYSTEM, user, max_tokens=4000)


def _cart_steps(ep: Episode, pick: dict) -> str:
    """A decision as exact cart operations against the latest cart view."""
    if str(pick.get("action", "")).lower() == "decline":
        return "Exact steps: close the episode with no purchase."
    want: dict = {}
    for ln in pick.get("lines") or []:
        if isinstance(ln, dict) and ln.get("id"):
            want[str(ln["id"])] = ln.get("qty") or 1
    if not want:
        return ""
    now = _cart_ids(ep)
    ops = [f"remove {i}" for i in now if i not in want]
    ops += [f"put {i} x{q}" for i, q in want.items()]
    return ("Exact steps: " + "; ".join(ops) + " (the cart must end up holding only these lines); then view the "
            "cart and close the episode with the purchase.")


def _cart_ids(ep: Episode) -> list[str]:
    """Listings the latest cart view mentions: any of its values that is a listing seen before."""
    vals: list = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (str, int)) and not isinstance(o, bool) and str(o) in ep.listings:
            vals.append(str(o))

    walk(ep.cart_obs)
    return list(dict.fromkeys(vals))


def _refresh_cart(ep: Episode, roles: dict, view_cart: str | None) -> list[str]:
    """After the last cart change, look again at the cart and at each line's page (stock and price can change)."""
    if view_cart and (ep.cart_dirty or ep.cart_obs is None):
        ep.cart_obs = ep.call(view_cart, _coerce_args({}, _schema(ep.by_name[view_cart])))
        ep.cart_dirty = False
    ids = _cart_ids(ep)[:4]
    pages = [t for t in roles.get("view", []) if t in ep.by_name]
    stock_page = (ep.bool_pages or pages or [None])[0]
    if not ids or not stock_page:
        return []
    plan = []
    for i in ids:
        plan.append((stock_page, i))  # stock and price may have changed since the cart was set
        plan += [(t, i) for t in pages if t != stock_page and t not in ep.opened_by.get(i, set())]
    for name, i in plan:
        if ep.finished or ep.left <= 3:
            break
        tool = ep.by_name[name]
        key = _text_arg(tool)
        if key:
            ep.call(name, _coerce_args({key: i}, _schema(tool)))
    return ids


# ---------------------------------------------------------------- main loop

def _state_line(ep: Episode, brief: dict | None, msgs_sent: int) -> str:
    lim = _message_limit(brief)
    return f"[step {ep.step} of {ep.max_steps} used; {ep.left} left; messages sent {msgs_sent} of {lim}]"


def _finish_modes(brief: dict | None) -> dict:
    raw = (brief or {}).get("finish_modes")
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def _fallback_close(ep: Episode, roles: dict, brief: dict | None) -> None:
    """Leave no episode open: order what the cart holds, else close with no purchase."""
    if ep.finished or ep.left <= 0:
        return
    modes = _finish_modes(brief)
    plans = []
    if ep.carted:
        plans += [(t, {}) for t in roles.get("checkout", [])]
        plans += [(t, _mode_args(ep, t, modes.get(t, {}).get("order"))) for t in roles.get("finish", [])]
    plans += [(t, {}) for t in roles.get("decline", [])]
    plans += [(t, _mode_args(ep, t, modes.get(t, {}).get("decline"))) for t in roles.get("finish", [])]
    for name, args in plans:
        if ep.finished or ep.left <= 0 or name not in ep.by_name:
            continue
        schema = _schema(ep.by_name[name])
        props = schema.get("properties") or {}
        for k in schema.get("required") or []:
            spec = props.get(k) if isinstance(props.get(k), dict) else {}
            if k not in args and spec.get("type") == "string" and not spec.get("enum"):
                args[k] = "Closing: the requirements could not all be confirmed."
        ep.call(name, _coerce_args(args, schema))


def _mode_values(want: Any) -> list[str]:
    vals = want if isinstance(want, list) else [want] if isinstance(want, str) else []
    return [str(v).lower() for v in vals if isinstance(v, (str, int)) and not isinstance(v, bool)]


def _mode_args(ep: Episode, name: str, want: Any) -> dict:
    """A finish tool's arguments for one way of closing, whether the brief gave them as arguments or as values."""
    if isinstance(want, dict):
        return dict(want)
    vals = _mode_values(want)
    props = _schema(ep.by_name.get(name) or {}).get("properties") or {}
    out = {}
    for k, spec in props.items():
        opts = spec.get("enum") if isinstance(spec, dict) else None
        hit = next((o for o in opts if str(o).lower() in vals), None) if isinstance(opts, list) else None
        if hit is not None:
            out[k] = hit
    return out


def _closing_kind(roles: dict, name: str, args: dict, brief: dict | None = None) -> str | None:
    if name in roles.get("checkout", []):
        return "order"
    if name in roles.get("decline", []):
        return "decline"
    if name in roles.get("finish", []):
        modes = _finish_modes(brief).get(name) or {}
        for kind in ("order", "decline"):
            want = modes.get(kind)
            if isinstance(want, dict) and want and all(str(args.get(k)).lower() == str(v).lower()
                                                       for k, v in want.items()):
                return kind
            vals = _mode_values(want)
            if vals and any(isinstance(a, (str, int)) and str(a).lower() in vals for a in args.values()):
                return kind
        return "close with arguments " + json.dumps(args, ensure_ascii=False)
    return None


def agent_main(problem_data: dict) -> list[dict]:
    _spend.usd = 0.0
    _spend.waits = 0
    ep = Episode(problem_data["environment"])
    if ep.finished:
        return []
    state: dict = {"roles": {}, "brief": None}
    try:
        _run(ep, state)
    except Exception:  # never leave the episode open because of a bug
        pass
    try:
        _fallback_close(ep, state["roles"], state["brief"])
    except Exception:
        pass
    return ep.dialogue


def _run(ep: Episode, state: dict) -> None:
    request = ep.policy.get("query") or ""

    brief = _make_brief(ep, request)
    if not isinstance(brief, dict):
        brief = None
    roles = _role_tools(brief, ep)
    state["roles"], state["brief"] = roles, brief
    ep.message_tools = set(roles.get("message", []))
    ep.page_tools = set(roles.get("view", []) + roles.get("compare", []))
    try:
        stated = int((brief or {}).get("step_limit") or 0)
    except (TypeError, ValueError):
        stated = 0
    if 0 < stated < ep.max_steps:
        ep.max_steps = stated
    probe_notes = _probe(ep, brief, roles)
    _ask(ep, brief, roles, request)
    msgs_sent = len(ep.qa)
    msg_limit = _message_limit(brief)

    intro = [f"SHOPPER REQUEST:\n{request}"]
    if ep.policy.get("opening") is not None:
        intro.append("OPENING NOTE:\n" + json.dumps(ep.policy["opening"], ensure_ascii=False))
    if brief:
        intro.append("BRIEF (from a careful reading of the request):\n" + _brief_text(brief))
        if roles:
            intro.append("TOOL ROLES: " + json.dumps(roles))
    if probe_notes:
        intro.append("FIRST LOOKS AT THE SHOP (already done, steps used):\n" + "\n".join(probe_notes))
    if ep.qa:
        intro.append("QUESTIONS ALREADY SENT AND THE SHOPPER'S REPLIES:\n" +
                     "\n".join(f"Q: {q}\nA: {a}" for q, a in ep.qa) +
                     "\n(A reply that does not understand the question means that property is not something "
                     "they have a wish about.)")
    intro.append(_state_line(ep, brief, msgs_sent) + " Begin.")
    messages: list[dict] = [{"role": "system", "content": LOOP_SYSTEM},
                            {"role": "user", "content": "\n\n".join(intro)}]

    reviews = 0
    idle = 0
    idle_steps = 0
    decided = False
    widened = False
    decline_pushed = False
    over_turns = 0
    refreshed = False
    redecides = 0
    wrap_told = False
    pages: dict[str, int] = {}
    streak = 0
    lookups = set(roles.get("search", []) + roles.get("view", []) + roles.get("compare", []))
    cart_tools = set(roles.get("cart_change", []) + roles.get("cart_set", []))
    view_cart = (roles.get("cart_view") or [None])[0]

    while not ep.finished and ep.step < ep.max_steps:
        forced = ep.left <= 1
        closers = [t for t in ep.tools if _tool_name(t) in
                   set(roles.get("checkout", []) + roles.get("decline", []) + roles.get("finish", []))]
        offer = closers if forced and closers else ep.tools
        if forced and closers:
            messages.append({"role": "user", "content": "Final step now: close the episode, with the "
                             "purchase if the cart holds your choice, otherwise with no purchase."})
        elif _spent() >= HARD_SPEND and closers and not wrap_told:
            wrap_told = True
            keep = set(_tool_name(t) for t in closers) | cart_tools | ({view_cart} if view_cart else set())
            offer = [t for t in ep.tools if _tool_name(t) in keep]
            messages.append({"role": "user", "content": "Time to finish: put your best choice in the cart if it is "
                             "not there yet (the listing you found that meets the most requirements, in the product "
                             "category asked for and within the budget), then close the episode with the purchase. "
                             "Close with no purchase only if every listing you saw clearly fails a requirement by its "
                             "own page facts (a different value, another category, over the budget)."})
        elif idle_steps >= 4 and closers and not wrap_told:
            wrap_told = True
            keep = set(_tool_name(t) for t in closers) | cart_tools | ({view_cart} if view_cart else set())
            offer = [t for t in ep.tools if _tool_name(t) in keep]
            messages.append({"role": "user", "content": "You keep making calls that do nothing. Put your best "
                             "qualifying choice in the cart if it is not there yet, then close the episode."})
        elif wrap_told and closers:
            keep = set(_tool_name(t) for t in closers) | cart_tools | ({view_cart} if view_cart else set())
            offer = [t for t in ep.tools if _tool_name(t) in keep]
        if _spent() >= 1.35 * HARD_SPEND and not ep.carted:
            over_turns += 1  # a few light turns to put the best find in the cart, rather than leave with nothing
        if idle_steps >= 8 or (_spent() >= 1.35 * HARD_SPEND and (ep.carted or over_turns > 4)):
            break
        reply = _chat([LIGHT, LIGHT_SPARE], messages, tools=offer, max_tokens=3000)
        if reply is None:
            idle += 1
            if idle > 3:
                break
            time.sleep(3)
            continue
        calls = reply.get("tool_calls") or []
        content = reply.get("content") or ""
        if not calls:
            idle += 1
            if idle > 4:
                break
            if idle >= 2:
                reply = _chat_forced(messages + [{"role": "assistant", "content": content or "(no tool call)"},
                                                 {"role": "user", "content": "Call one of the tools now."}], offer)
                calls = (reply or {}).get("tool_calls") or []
                content = (reply or {}).get("content") or ""
        if not calls:
            messages += [{"role": "assistant", "content": content or "(no tool call)"},
                         {"role": "user", "content": "Call one of the tools now."}]
            continue
        idle = 0
        idle_steps += 1
        call = calls[0] if isinstance(calls[0], dict) else {}
        if not call.get("id"):
            call["id"] = f"call_{ep.step}_{idle_steps}"
        name = (call.get("function") or {}).get("name", "")
        args = _parse_args(call)
        messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
        if name not in ep.by_name:
            messages.append({"role": "tool", "tool_call_id": call.get("id"),
                             "content": f"Unknown tool {name}. Use one of: {', '.join(ep.by_name)}"})
            continue
        args = _coerce_args(args, _schema(ep.by_name[name]))
        memo_key = name + json.dumps(args, sort_keys=True)
        qkey = name + "|" + json.dumps(sorted((k, str(v).strip().lower()) for k, v in args.items()
                                              if isinstance(v, str) and v not in ep.seen_strings))
        unopened = [i for i in ep.listings if i not in ep.viewed]
        if name in roles.get("search", []) and streak >= 5 and unopened and memo_key not in ep.memo:
            messages.append({"role": "tool", "tool_call_id": call.get("id"),
                             "content": "Not run (no step used): you searched several times in a row. Open the most "
                             "promising listings you already found (full pages or a side-by-side compare) before "
                             "searching again.\n" + _state_line(ep, brief, msgs_sent)})
            streak = 0
            continue
        if name in roles.get("search", []) and pages.get(qkey, 0) >= 6 and memo_key not in ep.memo:
            messages.append({"role": "tool", "tool_call_id": call.get("id"),
                             "content": "Not run (no step used): you have paged this query enough. Narrow the "
                             "query with another wanted value, open candidates, or decide.\n" +
                             _state_line(ep, brief, msgs_sent)})
            continue
        if name in lookups and memo_key in ep.memo:
            messages.append({"role": "tool", "tool_call_id": call.get("id"),
                             "content": "You already made this exact call (no step used). Its result was:\n" +
                             ep.memo[memo_key] + "\nDo something new.\n" + _state_line(ep, brief, msgs_sent)})
            continue

        # running out of steps with nothing chosen: the strong model decides
        if (not decided and not ep.carted and _spent() < HARD_SPEND
                and (ep.left <= END_RESERVE + 3 or (_spent() >= SOFT_SPEND and not wrap_told))
                and not _closing_kind(roles, name, args, brief)):
            decided = True
            pick = _decide(ep, request, brief)
            if pick:
                if str(pick.get("action", "")).lower() == "decline":
                    decline_pushed = True  # the strong model chose this decline: do not argue with it
                note = ("Not done: it is time to choose, so a decision was made from what you found:\n" +
                        json.dumps(pick, ensure_ascii=False) + "\n" + _cart_steps(ep, pick) +
                        "\nCarry it out now: put exactly these lines in the cart (if ordering), then close the "
                        "episode as decided.\n" + _state_line(ep, brief, msgs_sent))
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": note})
                reviews = MAX_REVIEWS
                continue

        # messages to the shopper: keep within the limit
        if name in roles.get("message", []):
            if msgs_sent >= msg_limit:
                messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                 "content": "Not sent: the message limit is reached. Continue without asking."})
                continue

        # closing: let the strong model check first
        kind = _closing_kind(roles, name, args, brief)
        objective = str((brief or {}).get("objective") or "").lower()
        early = ep.step < 0.55 * ep.max_steps and ep.left > 10
        if kind and early and not widened and (kind == "decline" or "cheap" in objective or "low" in objective):
            widened = True
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
            messages.append({"role": "user", "content": (
                "From me, your supervisor: you still have " + str(ep.left) + " steps, so search more widely "
                "before you " + ("decline" if kind == "decline" else "settle") + ": use each item's category "
                "filter search if a tool offers one and look at more of its result pages, try other wordings, and "
                "open the properties of the cheapest candidates you have not opened yet. Keep about 6 steps for "
                "the cart, a last check and closing. Then decide.")})
            continue
        if kind == "decline" and not decline_pushed and not forced and not wrap_told and ep.left > 8:
            decline_pushed = True
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
            messages.append({"role": "user", "content": (
                "From me, your supervisor: closing with no purchase is rarely right, and you still have " +
                str(ep.left) + " steps. First look again at the listings you opened: a requirement is met when the "
                "same property carries the wanted value under another name, as a synonym or in other units. If one "
                "meets everything, order it. Otherwise make two or three more searches (the category filter with a "
                "wanted value as the keyword, another wording, the next result page), open the most promising "
                "pages, and then decide. Keep about 6 steps for the cart and closing.")})
            continue
        decline_check = kind == "decline" and reviews <= MAX_REVIEWS and _spent() < HARD_SPEND
        if (kind and not forced and ep.left > 2
                and (decline_check or (reviews < MAX_REVIEWS and _spent() < SOFT_SPEND))):
            if kind != "decline" and not refreshed:
                refreshed = bool(_refresh_cart(ep, roles, view_cart))
            reviews += 1
            verdict = _review(ep, request, brief, kind)
            if verdict and kind != "decline" and str(verdict.get("close_as", "order")).lower() != "decline":
                extra = _table_fix(verdict, _cart_ids(ep))
                if extra:
                    verdict["verdict"] = "fix"
                    verdict["fix"] = (str(verdict.get("fix") or "") + " " + extra +
                                      " (keep the requested quantities)").strip()
            if verdict and str(verdict.get("verdict", "")).lower() == "fix" and verdict.get("fix"):
                note = ("Not closed yet. A senior check of your work decided this (carry it out; it is not a "
                        "suggestion):\n" + str(verdict.get("fix")) +
                        ("\nReason: " + str(verdict.get("reason")) if verdict.get("reason") else "") +
                        "\nLatest cart: " + json.dumps(ep.cart_obs, ensure_ascii=False) +
                        "\n" + _state_line(ep, brief, msgs_sent))
                messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                 "content": "Not executed: held for a check."})
                messages.append({"role": "user", "content": "From me, your supervisor. " + note})
                continue
            other = str((verdict or {}).get("close_as", "")).lower()
            if verdict and other in ("order", "decline") and kind in ("order", "decline") and other != kind:
                note = (f"Not closed yet. A check of your work says the right close is: {other}.\n"
                        f"Reason: {verdict.get('reason', '')}\n" + _state_line(ep, brief, msgs_sent))
                messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                 "content": "Not executed: held for a check."})
                messages.append({"role": "user", "content": "From me, your supervisor. " + note})
                continue

        step_before = ep.step
        obs = ep.call(name, args)
        if ep.step > step_before:
            idle_steps = 0
        if name in roles.get("message", []):
            msgs_sent += 1
            ep.qa.append((str(next(iter(args.values()), "")), _reply_text(obs)))
        if name in cart_tools and ep.step > step_before:
            ep.cart_dirty = True
            ep.carted = True
            refreshed = False
            ep.memo.clear()  # pages and results may change once something is in the cart
        if name == view_cart:
            ep.cart_obs = obs
            ep.cart_dirty = False
        text = _compact_obs(obs)
        fresh = list(ep.fresh)
        if name in lookups:
            ep.memo[memo_key] = text
        streak = streak + 1 if name in roles.get("search", []) else 0
        if name in roles.get("search", []):
            pages[qkey] = pages.get(qkey, 0) + 1
            note = _order_note(obs)
            if note:
                text = note + "\n" + text
        if ep.left <= END_RESERVE and not ep.finished:
            text += ("\nONLY A FEW STEPS LEFT: put your choice in the cart if it is not there, "
                     "then close the episode now.")
        messages.append({"role": "tool", "tool_call_id": call.get("id"),
                         "content": text + "\n" + _state_line(ep, brief, msgs_sent)})
        if kind and not ep.finished:
            pick = None
            if redecides < 2 and _spent() < HARD_SPEND and ep.left > 3:
                redecides += 1
                pick = _decide(ep, request, brief, "the shop did not accept the close; its reply was: " +
                               _compact_obs(obs)[:400])
            note = ("From me, your supervisor: the shop system reports the episode is still open after that call. "
                    "Read its result: if something changed (price, stock, the shopper's wishes), fix the cart and "
                    "close again; if it says the order or decline was taken, call the tool that ends the episode.")
            if pick:
                note += ("\nA senior check made this decision from everything seen so far (carry it out): " +
                         json.dumps(pick, ensure_ascii=False) + "\n" + _cart_steps(ep, pick) +
                         "\nIf it gives search_first, run that search to find the same product from another seller. "
                         "If it lists open_first, open those pages first; if one of them meets every requirement "
                         "and suits the shopper's objective better, use it instead. Then set the cart and close.")
                reviews = max(reviews, MAX_REVIEWS - 1)
            messages.append({"role": "user", "content": note})
        if fresh:
            messages.append({"role": "user", "content": (
                "Relayed by me, your supervisor: in this shop the shopper's changes of mind reach you through tool "
                "results, and this result carried a sentence outside any listing:\n" +
                "\n".join(f'"{n}"' for n in fresh) +
                "\nIf it is the shopper speaking (a new budget, a changed or added wish), it is genuine and is now "
                "what they want: check your cart, spending limit and requirements against it again and change the "
                "cart if needed. If it is only a shop notice (an error or status line), carry on.")})

    return


__all__ = ["agent_main"]
