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
SOFT_SPEND = 0.034
HARD_SPEND = 0.049
MAX_TURNS = 32
WORK_WRAP = 18000
WORK_STOP = 24000

_spend = threading.local()


def _spent() -> float:
    return getattr(_spend, "usd", 0.0)


def _written() -> int:
    """Tokens the light model has written so far (its reasoning included): what a task's run time grows with."""
    return getattr(_spend, "light_out", 0)


def _charge(model: str, data: dict) -> None:
    usage = data.get("usage") or {}
    cost = usage.get("cost")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        heavy = model not in (LIGHT, LIGHT_SPARE)
        cost = ((usage.get("prompt_tokens") or 0) * (2e-6 if heavy else 1e-7) +
                (usage.get("completion_tokens") or 0) * (1e-5 if heavy else 5e-7))
    _spend.usd = _spent() + float(cost)
    if model in (LIGHT, LIGHT_SPARE):
        out = usage.get("completion_tokens")
        _spend.light_out = _written() + (out if isinstance(out, int) and not isinstance(out, bool) else 0)


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
            elif isinstance(v, list):  # some shops send such sentences as a list of lines
                out.extend(x for x in v if isinstance(x, str) and len(x.split()) >= 5)

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
- "price_filter": a search tool that can limit results to a highest price, as {"tool": its name, "max_price_arg": the
  argument that sets the highest price, "category_arg": its argument that limits results to one product category (or
  null), "page_arg": its paging-token argument (or null)}, read from the tools' arguments; null if no tool has one.
- "probes": one first search per item, [{"item": product, "tool": search tool name, "args": arguments for that tool
  (use its category filter when it has one, and a short keyword: the product noun plus at most one stated
  value; leave out paging or page-token arguments)}].
- "plan": 2-3 short steps on how to find the right listing(s) with these tools."""

LOOP_SYSTEM = """You are a careful shop assistant working inside an online shop through its tools.
Every tool call is one step and steps are limited. Call one tool per turn and write one short line of reasoning
before it; only to open the full pages of several listings may you put several page calls in one turn (they run in
order, one step each).

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
   "vintage"; wanted "lightly salted" is not met by "unsalted"). Two different words the shop's pages both use as
   values of the same property are different values, never synonyms (a value that holds the wanted word with a
   modifier still meets it). Assume listing
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
   check other queries or more result pages so you do not miss a cheaper qualifying listing; when your most
   specific query (stated values and told needs as keywords) matches up to about 50 listings, page through all of
   its results. Notice how results
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
neighbouring value is a different value; two different words the shop's pages both use as values of the same
property are different values, never synonyms). A page value that holds the wanted value as a whole word with a modifier
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
        self.probe_of: dict[str, str] = {}
        self.cart_set: dict[str, float] = {}     # listing -> count, from the cart calls this agent made

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
- Choose from the PROPERTY TABLE: the properties the shop's pages record for that product, best ranked first (the
  rank grows with how many of the opened pages record it and how much its values differ between listings). The
  shopper's unsaid wishes are about those. Look at the four best-ranked properties of that product not asked yet and
  take the one people most often choose that kind of product by (how it looks, what it is made of, how it fits or
  works, what it holds or runs on, the person meant to use it); never a code or model number, a field the table marks as several
  choices in one value, or a fact about the offer or the shop rather than the product. A remark in the request
  moves a property up only when it clearly points at one in the table. Name the property exactly as the pages do,
  never in the shopper's own words.
- With several products, ask about the PRODUCT TO ASK ABOUT NOW when the input names one: a property of that
  product, named as its pages name it.
- Never a property whose value the request already states, nothing already asked (by meaning, not wording), never a
  field the table marks as several choices in one value, and only properties of the product itself, not codes,
  model numbers or facts about the offer or the shop."""


def _has_props(obs: Any) -> bool:
    """A page that records a listing's properties: some field holds a map of several values."""
    if isinstance(obs, dict):
        return any(isinstance(v, dict) and len(v) >= 2 for v in obs.values()) or \
            any(_has_props(v) for v in obs.values() if isinstance(v, list))
    if isinstance(obs, list):
        return any(_has_props(x) for x in obs)
    return False


def _probe(ep: Episode, brief: dict | None, roles: dict, per_item: int | None = None, per_call: int = 6) -> list[str]:
    """One search per item, then the property pages of some results (a page tool that takes several ids opens up to
    per_call of them in its one step; one that takes a single id opens per_item: 6 for one product, 4 each for several),
    so the shop's own property names, and which of them most pages record, are known before any question is sent."""
    notes: list[str] = []
    if not brief:
        return notes
    views = [t for t in (roles.get("compare") or []) + (roles.get("view") or []) if t in ep.by_name]
    prop_tool = None
    probes = brief.get("probes") if isinstance(brief.get("probes"), list) else []
    if per_item is None:
        per_item = 6 if len(probes) <= 1 else 4
    for pr in probes[:4]:
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
        pool = [i for i in dict.fromkeys(i for i, _ in found) if i not in ep.viewed]
        alt = next((t for t in roles.get("search") or [] if t != name and t in ep.by_name
                    and _text_arg(ep.by_name[t])), None)
        if not pool and alt and pr.get("item") and ep.left >= END_RESERVE + 10:
            # the first search found nothing (often a filter value the shop words otherwise): try plain keywords
            args = _coerce_args({_text_arg(ep.by_name[alt]): str(pr["item"])}, _schema(ep.by_name[alt]))
            obs = ep.call(alt, args)
            notes.append(f"{alt} {json.dumps(args, ensure_ascii=False)} -> {_compact_obs(obs)}")
            found = []
            _listings_in(obs, found, set())
            pool = [i for i in dict.fromkeys(i for i, _ in found) if i not in ep.viewed]
        for i in pool[:per_call]:
            ep.probe_of.setdefault(i, str(pr.get("item") or ""))
        ids = pool[:per_item]
        for n, lid in enumerate(ids):
            if ep.finished or ep.left < END_RESERVE + 10:
                break
            for t in ([prop_tool] if prop_tool else views[:2]):
                tool = ep.by_name[t]
                props = _schema(tool).get("properties") or {}
                arr = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array"]
                if arr:
                    vargs = _coerce_args({arr[0]: pool[:per_call]}, _schema(tool))
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


def _prop_key(k: Any) -> str:
    return " ".join(str(k).replace("_", " ").lower().split())


def _prop_table(ep: Episode) -> dict[str, dict[str, dict]]:
    """Per probed product: each recorded property -> {pages: how many opened pages record it, values: some values,
    mixed: most values join several choices, rank: pages' share plus how much the values differ}."""
    out: dict[str, dict[str, dict]] = {}
    for lid in ep.viewed:
        d = ep.listings.get(lid) or {}
        item = ep.probe_of.get(lid, "")
        table = out.setdefault(item, {})
        for v in d.values():
            if not isinstance(v, dict):
                continue
            for pk, pv in v.items():
                if pv in (None, "", [], {}):
                    continue
                row = table.setdefault(_prop_key(pk), {"pages": 0, "values": [], "joined": 0, "lids": set()})
                if lid in row["lids"]:
                    continue  # the same property in two of the page's fields counts once
                row["lids"].add(lid)
                row["pages"] += 1
                text = str(pv)[:30]
                if text not in row["values"]:
                    row["values"].append(text)
                row["joined"] += " / " in str(pv)
    for item, table in out.items():
        n = max(1, sum(1 for lid in ep.viewed if ep.probe_of.get(lid, "") == item))
        for row in table.values():
            row["mixed"] = row["joined"] * 2 > row["pages"]
            row["rank"] = round(min(1.0, row["pages"] / n) + 0.5 * min(len(row["values"]), 3) / 3, 2)
    return out


def _table_text(ep: Episode, table: dict[str, dict[str, dict]]) -> str:
    lines = []
    for item, rows in table.items():
        n = sum(1 for lid in ep.viewed if ep.probe_of.get(lid, "") == item)
        order = sorted(rows.items(), key=lambda kv: (-kv[1]["rank"], -kv[1]["pages"]))
        cells = [f"{k}: rank {r['rank']:.2f}, {r['pages']}/{n} pages" +
                 (" [several choices in one value: never ask]" if r["mixed"] else f" ({', '.join(r['values'][:4])})")
                 for k, r in order[:24]]
        lines.append(f"{item or 'pages opened'}:\n  " + "\n  ".join(cells))
    return "\n".join(lines)


def _next_question(ep: Episode, brief: dict, request: str, stated: dict, left: int,
                   focus: str | None = None) -> dict | None:
    history = "\n".join(f"Q: {q}\nA: {a}" for q, a in ep.qa) or "(none yet)"
    hints = [{"item": q.get("item"), "property": q.get("property")} for q in brief.get("questions") or []
             if isinstance(q, dict) and q.get("property")]
    user = (f"SHOPPER REQUEST:\n{request}\n\nBRIEF:\n{_brief_text(brief)}\n\nHINTS FROM THE REQUEST (in the "
            f"shopper's words): {json.dumps(hints, ensure_ascii=False)}\n\nALREADY STATED BY THE SHOPPER, BY "
            f"PRODUCT (never ask these for that product): "
            f"{json.dumps({k: sorted(v) for k, v in stated.items()}, ensure_ascii=False)}\n\nQUESTIONS SENT AND REPLIES:\n"
            f"{history}\n\nPROPERTY TABLE (from the pages opened):\n{_table_text(ep, _prop_table(ep))}\n\n"
            f"SHOP PAGES SEEN:\n{_page_digest(ep)}\n\nMessages left after this one: {left - 1}." +
            (f"\n\nPRODUCT TO ASK ABOUT NOW: {focus}" if focus else ""))
    got = _ask_json([LIGHT, HEAVY], QUESTION_SYSTEM, user, max_tokens=2500)
    prop = " ".join(str((got or {}).get("property") or "").replace("_", " ").split())
    if not prop:
        return None
    items = [str(i.get("what")) for i in brief.get("items") or [] if isinstance(i, dict) and i.get("what")]
    item = " ".join(str(got.get("item") or focus or (items[0] if items else "item")).split())
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
    stated = _stated_by_item(brief)
    items = list(stated)
    count = {it: 0 for it in items}
    told: set = set()
    asked: set = set()
    tries = 0
    while len(ep.qa) < budget and not ep.finished and tries < budget + 3:
        tries += 1
        # several products: the next question goes to the one asked least, a product whose wish was told last
        focus = min(items, key=lambda it: (it in told, count[it], items.index(it))) if len(items) > 1 else None
        q = _next_question(ep, brief, request, stated, budget - len(ep.qa), focus)
        if not q:
            break
        text = str(q.get("text")).strip()
        item = _match_item(str(q.get("item", "")), items) or focus or (items[0] if items else "")
        key = (item.lower(), str(q.get("property", "")).lower())
        mixed = any(r.get(_prop_key(key[1]), {}).get("mixed") for r in _prop_table(ep).values())
        if not text or key in asked or _overlaps(key[1], stated.get(item, set())) or mixed:
            if key[1]:
                stated.setdefault(item, set()).add(key[1])
            continue
        asked.add(key)
        obs = ep.call(tools[0], _coerce_args({arg: text}, _schema(tool)))
        reply = _reply_text(obs)
        ep.qa.append((text, reply))
        if item in count:
            count[item] += 1
            if _words(key[1]) and _words(key[1]) <= _words(reply):
                told.add(item)  # a wish names its property ("where X is concerned, only Y")


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if len(w) > 2}


def _stated_by_item(brief: dict | None) -> dict[str, set[str]]:
    """Each requested product (as the brief names it) -> the properties the request states for it."""
    out: dict[str, set[str]] = {}
    items = (brief or {}).get("items")
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not it.get("what"):
            continue
        props = out.setdefault(" ".join(str(it["what"]).split()), set())
        musts = it.get("musts")
        for m in musts if isinstance(musts, list) else []:
            prop = str(m).split(":", 1)[0].strip().lower()
            if prop:
                props.add(prop)
    return out


def _match_item(name: str, items: list[str]) -> str | None:
    """The requested product a name means: the one sharing the most words with it."""
    w = _words(name)
    best = max(items, key=lambda it: len(w & _words(it)), default=None)
    return best if best and w & _words(best) else None


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
unit conversions count; compatible, stronger, weaker or related values do not, nor two different words the shop's pages
both use as values of the same property); a requirement the page contradicts is
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


def _removal_hint(ep: Episode, roles: dict) -> str:
    """How a cart line comes out when no tool says it removes: a cart tool whose count may be set to 0."""
    for name in roles.get("cart_set", []) + roles.get("cart_change", []):
        tool = ep.by_name.get(name)
        props = _schema(tool or {}).get("properties") or {}
        key = _text_arg(tool) if tool else None
        qty = next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") in ("integer", "number")
                    and v.get("minimum") == 0), None)
        if key and qty:
            return (f"{name} sets a listing's count in the cart: calling it with {key} = <listing id> and {qty} = 0 "
                    f"takes that line out of the cart.")
    return ""


def _pick_lines(pick: dict) -> dict[str, float] | None:
    """A decision's order as listing -> count; None for a decline."""
    if str(pick.get("action", "")).lower() != "order":
        return None
    out = {}
    for ln in pick.get("lines") or []:
        if isinstance(ln, dict) and ln.get("id"):
            out[str(ln["id"])] = _number(ln.get("qty")) or 1.0
    return out or None


def _verdict_lines(ep: Episode, brief: dict | None, verdict: dict) -> dict[str, float] | None:
    """A review's order as listing -> count: per requested product the cheapest (when the shopper wants the lowest
    price) or first in-stock listing it judged to meet everything; None when it judged a decline right."""
    if str(verdict.get("close_as", "")).lower() == "decline":
        return None
    rows = [r for r in verdict.get("table") or [] if isinstance(r, dict) and r.get("meets_all") is True
            and r.get("in_stock") is not False and r.get("id") and str(r["id"]) in ep.listings]
    if not rows:
        return None
    qty = {" ".join(str(i.get("what") or "").lower().split()): _number(i.get("qty")) or 1.0
           for i in (brief or {}).get("items") or [] if isinstance(i, dict)}
    best: dict[str, dict] = {}
    for r in rows:
        item = " ".join(str(r.get("item") or "").lower().split())
        price = _number(r.get("price"))
        if item not in best or (verdict.get("cheapest_wins") and price is not None
                                and price < (_number(best[item].get("price")) or float("inf"))):
            best[item] = r
    out = {}
    for item, r in best.items():
        match = _match_item(item, list(qty)) if qty else None
        out[str(r["id"])] = qty.get(match, 1.0) if match else (list(qty.values())[0] if len(qty) == 1 else 1.0)
    return out


def _place(ep: Episode, roles: dict, brief: dict | None, lines: dict[str, float]) -> Any:
    """Set the cart to exactly these lines with the cart tool and close with the purchase; returns the shop's last
    answer."""
    tool = next((t for t in roles.get("cart_set", []) + roles.get("cart_change", []) if t in ep.by_name), None)
    if not tool:
        return None
    schema = _schema(ep.by_name[tool])
    props = schema.get("properties") or {}
    key = _text_arg(ep.by_name[tool])
    qty = next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") in ("integer", "number")), None)
    if not key:
        return None
    have = {i: n for i, n in ep.cart_set.items() if n > 0} or {i: 1.0 for i in _cart_ids(ep)}
    got: Any = None
    for i in have:
        if i not in lines and qty and (props[qty] or {}).get("minimum") == 0 and ep.left > len(lines) + 1:
            got = ep.call(tool, _coerce_args({key: i, qty: 0}, schema))
            ep.cart_set[i] = 0.0
    for i, n in lines.items():
        if have.get(i) != n and ep.left > 1:
            got = ep.call(tool, _coerce_args({key: i, **({qty: int(n)} if qty else {})}, schema))
            ep.cart_set[i] = n
            ep.carted = True
    for t in roles.get("checkout", []):
        if t in ep.by_name and not ep.finished and ep.left > 0:
            return ep.call(t, _coerce_args({}, _schema(ep.by_name[t])))
    modes = _finish_modes(brief)
    for t in roles.get("finish", []):
        if t in ep.by_name and not ep.finished and ep.left > 0:
            return ep.call(t, _coerce_args(_mode_args(ep, t, modes.get(t, {}).get("order")), _schema(ep.by_name[t])))
    return got


def _unchecked_cheaper(ep: Episode, brief: dict | None, limit: int = 5) -> list[tuple[str, float]]:
    """For a lowest-price request: listings seen in results below the carted one's price, in its category, not sold
    out and within the budget, whose pages were never opened (cheapest first)."""
    cart = _cart_ids(ep) or [i for i, n in ep.cart_set.items() if n > 0]
    if len(cart) != 1:
        return []
    held = ep.listings.get(cart[0]) or {}
    price = _listing_price(held)
    if price is None:
        return []
    cat = _category_value(ep, cart[0], brief)
    cat_key = next((k for k, v in held.items() if cat and v == cat), None)
    budget = (brief or {}).get("budget") if isinstance((brief or {}).get("budget"), dict) else {}
    cap = _number(budget.get("amount")) if str(budget.get("applies_to") or "total") == "total" else None
    out = []
    for lid, d in ep.listings.items():
        if lid in ep.viewed or lid == cart[0]:
            continue
        p = _listing_price(d)
        if p is None or p >= price or (cap is not None and p > cap):
            continue
        if (brief or {}).get("skip_sold_out", True) and _sold_out(d):
            continue
        if cat_key and d.get(cat_key) is not None and d.get(cat_key) != held.get(cat_key):
            continue
        out.append((p, lid))
    return [(lid, p) for p, lid in sorted(out)[:limit]]


def _ids_for_titles(ep: Episode, args: dict) -> dict:
    """A cart call that names a listing by the start of its title instead of its id gets that listing's id."""
    out = dict(args)
    for k, v in args.items():
        if not isinstance(v, str) or v in ep.listings or len(v) < 12:
            continue
        hits = [lid for lid, d in ep.listings.items()
                if any(isinstance(x, str) and len(x) >= len(v) and x.startswith(v) for x in d.values())]
        if len(hits) == 1:
            out[k] = hits[0]
    return out


def _listing_price(d: dict) -> float | None:
    for k, v in d.items():
        if isinstance(k, str) and re.search(r"price|cost|amount", k.lower()):
            n = _number(v) if not isinstance(v, bool) else None
            if n is not None:
                return n
    return None


def _sold_out(d: dict) -> bool:
    for k, v in d.items():
        if not isinstance(v, bool) or not isinstance(k, str):
            continue
        name = k.lower()
        if re.search(r"sold|out_of|outof", name):
            return v
        if re.search(r"stock|avail|buy|purchas", name):
            return not v
    return False


def _category_value(ep: Episode, lid: str, brief: dict | None) -> str | None:
    """The carted listing's category as the shop writes it: its short text field that most listings seen share,
    favouring one whose words match a requested product."""
    d = ep.listings.get(lid) or {}
    wanted = set()
    for it in (brief or {}).get("items") or []:
        if isinstance(it, dict):
            wanted |= _words(f"{it.get('what') or ''} {it.get('category') or ''}")
    best, score = None, 0.0
    for k, v in d.items():
        if k.startswith("_") or not isinstance(v, str) or not v or len(v) > 80 or _code_like(v) or v == lid:
            continue
        having = [x.get(k) for x in ep.listings.values() if isinstance(x.get(k), str)]
        if len(having) < 3:
            continue
        share = sum(1 for x in having if x == v) / len(having)
        s = share + (0.5 if _words(v) & wanted else 0.0)
        if share >= 0.3 and s > score:
            best, score = v, s
    return best


def _page_token(obs: Any) -> str | None:
    """A result's paging token: its only top-level short text value."""
    if not isinstance(obs, dict):
        return None
    vals = [v for v in obs.values() if isinstance(v, str) and 0 < len(v) <= 40]
    return vals[0] if len(vals) == 1 else None


def _cheaper_sweep(ep: Episode, brief: dict | None, roles: dict, view_cart: str | None) -> str | None:
    """The shopper wants the lowest price: list the carted line's category below its price with the shop's price
    filter (the code sets the filter, which the light model often misreads) and open the cheaper in-stock listings,
    so the check before the order can compare them."""
    pf = (brief or {}).get("price_filter")
    if not isinstance(pf, dict) or pf.get("tool") not in ep.by_name:
        return None
    tool = ep.by_name[pf["tool"]]
    schema = _schema(tool)
    props = schema.get("properties") or {}
    max_arg, cat_arg, page_arg = pf.get("max_price_arg"), pf.get("category_arg"), pf.get("page_arg")
    if max_arg not in props or (props[max_arg] or {}).get("type") not in ("number", "integer"):
        return None
    cat_arg = cat_arg if cat_arg in props else None
    page_arg = page_arg if page_arg in props else None
    if view_cart and (ep.cart_dirty or ep.cart_obs is None):
        ep.cart_obs = ep.call(view_cart, _coerce_args({}, _schema(ep.by_name[view_cart])))
        ep.cart_dirty = False
    lines = _cart_ids(ep)
    if not lines or len(lines) > 2:
        return None
    wanted = set()
    for it in (brief or {}).get("items") or []:
        if isinstance(it, dict):
            wanted |= _words(" ".join(str(m).split(":", 1)[-1] for m in it.get("musts") or []))
    for _, a in ep.qa:
        wanted |= _words(a)
    cands: dict[str, tuple[int, float]] = {}
    for lid in lines:
        price = _listing_price(ep.listings.get(lid) or {})
        cat = _category_value(ep, lid, brief) if cat_arg else None
        if price is None or (cat_arg and not cat):
            continue
        args: dict = {max_arg: price - 1 if float(price).is_integer() and price > 1 else price * 0.999}
        if cat:
            args[cat_arg] = cat
        for _ in range(3):
            if ep.finished or ep.left < END_RESERVE + 8:
                break
            obs = ep.call(pf["tool"], _coerce_args(args, schema))
            found: list = []
            _listings_in(obs, found, set())
            for i, r in found:
                p = _listing_price(r)
                if p is None or p >= price or i in ep.viewed or i in lines:
                    continue
                if (brief or {}).get("skip_sold_out", True) and _sold_out(r):
                    continue
                text = " ".join(str(v) for v in r.values() if isinstance(v, str))
                cands[i] = (len(_words(text) & wanted), p)
            token = _page_token(obs)
            if not found or not page_arg or not token:
                break
            args = {**args, page_arg: token}
    if not cands:
        return None
    order = sorted(cands, key=lambda i: (-cands[i][0], cands[i][1]))
    views = [t for t in (roles.get("compare") or []) + (roles.get("view") or []) if t in ep.by_name]
    opened, shown = 0, []
    for t in views[:1]:
        page = ep.by_name[t]
        arr = [k for k, v in (_schema(page).get("properties") or {}).items()
               if isinstance(v, dict) and v.get("type") == "array"]
        batches = [order[:6], order[6:12]] if arr else [[i] for i in order[:3]]
        for b in batches:
            if not b or ep.finished or ep.left < END_RESERVE + 6:
                break
            key = arr[0] if arr else _text_arg(page)
            if not key:
                break
            vargs = _coerce_args({key: b if arr else b[0]}, _schema(page))
            shown.append(f"{t} {json.dumps(vargs, ensure_ascii=False)} -> {_compact_obs(ep.call(t, vargs))}")
            opened += len(b)
    rest = [f"{i} ({cands[i][1]:g})" for i in order[opened:opened + 12]]
    return (f"The shopper wants the lowest price, so the shop's price filter was run (steps used) for the cart's "
            f"category below the carted price: {len(cands)} cheaper listings in stock. The {opened} likeliest were "
            f"opened:\n" + "\n".join(shown) + ("\nOthers not opened (id, price): " + ", ".join(rest) if rest else "") +
            "\nIf one of them meets every stated requirement and every need the shopper told by its page facts, put "
            "the cheapest such listing in the cart instead (same quantity, and take the old line out); otherwise "
            "keep your choice. Then close.")


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
    _spend.light_out = 0
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
    removal = _removal_hint(ep, roles)
    if removal:
        intro.append("CART: " + removal)
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
    swept = False
    pick_lines: dict[str, float] | None = None   # the strong model's latest order: listing -> count
    placed = False
    cheaper_checks = 0
    cart_fix_told = False
    turns = 0
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
        elif (_spent() >= HARD_SPEND or turns >= MAX_TURNS or _written() >= WORK_WRAP) and closers and not wrap_told:
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
        if idle_steps >= 8 or turns >= MAX_TURNS + 6 or _written() >= WORK_STOP or (_spent() >= 1.35 * HARD_SPEND and (ep.carted or over_turns > 4)):
            break
        turns += 1
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
        pagers = set(roles.get("view", []) + roles.get("compare", []))
        batch = [c for c in calls if isinstance(c, dict) and (c.get("function") or {}).get("name") in pagers]
        if len(calls) > 1 and len(batch) == len(calls) and not wrap_told:
            # several page openings in one turn: each runs in order, one step each
            for n, c in enumerate(batch):
                c["id"] = c.get("id") or f"call_{ep.step}_{idle_steps}_{n}"
            messages.append({"role": "assistant", "content": content, "tool_calls": batch})
            fresh_all: list[str] = []
            for c in batch:
                nm = c["function"]["name"]
                a = _coerce_args(_parse_args(c), _schema(ep.by_name[nm]))
                key = nm + json.dumps(a, sort_keys=True)
                if key in ep.memo:
                    out = "You already made this exact call (no step used). Its result was:\n" + ep.memo[key]
                elif ep.finished or ep.left <= END_RESERVE + 1:
                    out = "Not run: too few steps are left to open more pages."
                else:
                    before = ep.step
                    out = _compact_obs(ep.call(nm, a))
                    fresh_all += ep.fresh
                    if ep.step > before:
                        idle_steps = 0
                        ep.memo[key] = out
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
            messages[-1]["content"] += "\n" + _state_line(ep, brief, msgs_sent)
            streak = 0
            if fresh_all:
                messages.append(_relay(fresh_all))
            continue
        dropped = len(calls) > 1
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
        if name in cart_tools:
            args = _ids_for_titles(ep, args)
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
                pick_lines = _pick_lines(pick)
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
        if kind == "decline" and pick_lines and not placed and ep.left >= len(pick_lines) + 2:
            placed = True
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed."})
            got = _place(ep, roles, brief, pick_lines)
            if ep.finished:
                break
            messages.append({"role": "user", "content": (
                "From me, your supervisor: the senior check found listings that meet everything, so I put them in "
                "the cart and placed the order for you: " + ", ".join(f"{i} x{n:g}" for i, n in pick_lines.items()) +
                ". The shop answered: " + _compact_obs(got)[:600] + "\nRead it: fix what it says, then close again.\n" +
                _state_line(ep, brief, msgs_sent))})
            continue
        if kind == "decline" and ep.carted and removal and not cart_fix_told and not forced and ep.left > 2:
            cart_fix_told = True
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
            messages.append({"role": "user", "content": (
                "From me, your supervisor: " + removal + " If the cart holds a line you do not want, take it out "
                "that way, so the cart holds exactly the listings to buy, then close with the purchase. Close with no "
                "purchase only if no listing you saw meets every requirement.\n" + _state_line(ep, brief, msgs_sent))})
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
        cheap = "cheap" in objective or "low" in objective
        if kind == "order" and cheap and not swept and not forced and ep.left >= END_RESERVE + 10:
            swept = True
            note = _cheaper_sweep(ep, brief, roles, view_cart)
            if note:
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
                messages.append({"role": "user", "content": "From me, your supervisor: " + note + "\n" +
                                 _state_line(ep, brief, msgs_sent)})
                continue
        if kind == "order" and cheap and cheaper_checks < 2 and not forced and ep.left > END_RESERVE + 5:
            seen = _unchecked_cheaper(ep, brief)
            if seen:
                cheaper_checks += 1
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
                messages.append({"role": "user", "content": (
                    "From me, your supervisor: the shopper wants the lowest price, and these listings you saw in "
                    "results cost less than your choice but were never opened: " +
                    ", ".join(f"{i} ({p:g})" for i, p in seen) + ". Open their pages (several page calls in one "
                    "turn); if one meets every stated requirement and every need the shopper told, put the cheapest "
                    "such listing in the cart instead (same quantity, and take the old line out); if none does, close "
                    "again.\n" + _state_line(ep, brief, msgs_sent))})
                continue
        decline_check = kind == "decline" and reviews <= MAX_REVIEWS and _spent() < HARD_SPEND
        if (kind and not forced and ep.left > 2
                and (decline_check or (reviews < MAX_REVIEWS and _spent() < SOFT_SPEND))):
            if kind != "decline" and not refreshed:
                refreshed = bool(_refresh_cart(ep, roles, view_cart))
            reviews += 1
            verdict = _review(ep, request, brief, kind)
            if verdict:
                pick_lines = _verdict_lines(ep, brief, verdict)
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
            if not view_cart and isinstance(obs, (dict, list)) and not (isinstance(obs, dict) and obs.get("error")):
                ep.cart_obs = obs  # no cart view in this shop: the cart tool's own reply is the latest cart
                ep.cart_dirty = False
            if not (isinstance(obs, dict) and obs.get("error")):
                lid = next((v for v in args.values() if isinstance(v, str) and v in ep.listings), None)
                n = next((v for v in args.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), 1)
                if lid:
                    ep.cart_set[lid] = float(n)
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
        if dropped:
            text += "\n(Only this call ran: one tool per turn, except several page openings at once.)"
        messages.append({"role": "tool", "tool_call_id": call.get("id"),
                         "content": text + "\n" + _state_line(ep, brief, msgs_sent)})
        if kind and not ep.finished:
            pick = None
            if redecides < 2 and _spent() < HARD_SPEND and ep.left > 3:
                redecides += 1
                pick = _decide(ep, request, brief, "the shop did not accept the close; its reply was: " +
                               _compact_obs(obs)[:400])
                if pick:
                    pick_lines = _pick_lines(pick)
            note = ("From me, your supervisor: the shop system reports the episode is still open after that call. "
                    "Read its result: if something changed (price, stock, the shopper's wishes), fix the cart and "
                    "close again; if it says the order or decline was taken, call the tool that ends the episode.")
            if cheap and kind == "order":
                note += (" The shopper wants the lowest price: before you order a backup, look again for cheaper "
                         "listings that fit now (more result pages, other wordings, a price limit when a tool has one).")
            if pick:
                note += ("\nA senior check made this decision from everything seen so far (carry it out): " +
                         json.dumps(pick, ensure_ascii=False) + "\n" + _cart_steps(ep, pick) +
                         "\nIf it gives search_first, run that search to find the same product from another seller. "
                         "If it lists open_first, open those pages first; if one of them meets every requirement "
                         "and suits the shopper's objective better, use it instead. Then set the cart and close.")
                reviews = max(reviews, MAX_REVIEWS - 1)
            messages.append({"role": "user", "content": note})
        if fresh:
            messages.append(_relay(fresh))

    return


def _relay(fresh: list[str]) -> dict:
    return {"role": "user", "content": (
        "Relayed by me, your supervisor: in this shop the shopper's changes of mind reach you through tool "
        "results, and this result carried a sentence outside any listing:\n" +
        "\n".join(f'"{n}"' for n in fresh) +
        "\nIf it is the shopper speaking (a new budget, a changed or added wish), it is genuine and is now "
        "what they want: check your cart, spending limit and requirements against it again and change the "
        "cart if needed. If it only says a price has changed, open the carted listing's page to read what it "
        "costs now, and keep it while the order total is within the spending limit and it is still the best choice by the "
        "shopper's objective, then close again. If it is only a shop notice (an error or status line), carry on.")}


__all__ = ["agent_main"]
