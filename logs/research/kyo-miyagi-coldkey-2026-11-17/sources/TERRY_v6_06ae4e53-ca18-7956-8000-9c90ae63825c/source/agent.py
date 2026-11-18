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
FIRST_BATCH = 5  # ids per call to a list-taking page tool until the shop shows what it takes
FIX_TURNS = 6  # light turns with the search and page tools back after a review asks for more looking late on

_spend = threading.local()


def _spent() -> float:
    return getattr(_spend, "usd", 0.0)


def _soft() -> float:
    return SOFT_SPEND * getattr(_spend, "scale", 1.0)


def _hard() -> float:
    return HARD_SPEND * getattr(_spend, "scale", 1.0)


def _written() -> int:
    """Tokens the light model has written so far (its reasoning included): what a task's run time grows with."""
    return getattr(_spend, "light_out", 0)


def _charge(model: str, data: dict) -> None:
    usage = data.get("usage") or {}
    cost = usage.get("cost")
    prompt = usage.get("prompt_tokens") or 0
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or (cost <= 0 and prompt):
        # no price given (or a zero one for real tokens): count the tokens at the provider's list price; a growing
        # conversation's earlier part counts as cached, as the provider caches it (the prompt sent before for this
        # model, when the new one is longer)
        heavy = model not in (LIGHT, LIGHT_SPARE)
        seen = getattr(_spend, "prompts", {})
        before = seen.get(model, 0)
        cached = min(prompt, (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        if before and prompt >= before and not heavy:  # the light model's tool loop is one growing conversation
            cached = max(cached, before)
        seen[model] = prompt
        _spend.prompts = seen
        cost = ((prompt - cached) * (2e-6 if heavy else 1e-7) + cached * (1e-7 if heavy else 1e-8) +
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


def _answer_tool(fields: dict[str, str]) -> dict:
    return {"type": "function", "function": {
        "name": "answer", "description": "Give your answer.",
        "parameters": {"type": "object", "properties": {k: {"type": "string", "description": d}
                                                        for k, d in fields.items()},
                       "required": list(fields)}}}


def _answer_fields(system: str) -> dict[str, str]:
    """The answer tool's fields, read off the JSON a prompt asks for: a line that opens with a key starts a field, and
    what follows the key, with the indented lines under it, describes that field."""
    fields: dict[str, str] = {}
    key = None
    for line in system.splitlines():
        m = re.match(r'\s*\{?"(\w+)":\s*(.*)', line)
        if m:
            key = m.group(1)
            fields[key] = m.group(2)
        elif key and line[:1].isspace():
            fields[key] += " " + line.strip()
        else:
            key = None
    return {k: re.sub(r"[,}]$", "", d.strip()).strip() for k, d in fields.items()}


def _ask_json(models: list[str], system: str, user: str, max_tokens: int = 2500,
              fields: dict[str, str] | None = None) -> dict | None:
    """Each model in turn gets two tries at the JSON; one that answers without it hands over to the next. With
    `fields` the answer may also come as a call of an answer tool that takes them."""
    tools = [_answer_tool(fields)] if fields else None
    for n, model in enumerate(models):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(2):
            reply = _chat(models[n:], msgs, tools=tools, max_tokens=max_tokens)
            if reply is None:
                return None
            for c in reply.get("tool_calls") or []:
                got = _parse_args(c) if isinstance(c, dict) else {}
                if got:
                    return got
            parsed = _json_from(reply.get("content") or "")
            if parsed is not None:
                return parsed
            if not str(reply.get("content") or "").strip():
                break  # an empty answer is not worth a second try: the next model
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


def _has_note_text(obs: Any) -> bool:
    """A page result with free text beyond a title: a string of whole sentences (notes, a description)."""
    if isinstance(obs, dict):
        return any(isinstance(v, str) and len(v.split()) >= 10 and re.search(r"[.!?]\s+[A-Z]", v)
                   for v in obs.values()) or \
            any(_has_note_text(v) for v in obs.values() if isinstance(v, (list, dict)))
    if isinstance(obs, list):
        return any(_has_note_text(x) for x in obs[:6])
    return False


def _is_full_page(d: dict) -> bool:
    return any(isinstance(v, dict) and len(v) >= 2 for v in d.values())


def _compact_obs(obs: Any, pages: bool = False) -> str:
    """Short JSON for the light model: long text cut, empty fields dropped. Rows of a page tool keep more of their
    text: a note often says only at its end which other listing to look at."""
    if isinstance(obs, dict):
        lists = [v for v in obs.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if lists:
            return json.dumps(_shrink(obs, 420 if pages else 140), ensure_ascii=False)
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
  and how it must relate: same, different, within N, at least, the value that page pairs it with when it must work
  with or suit what the page shows, or that pointed-to listing itself as the purchase), or null}.
- "budget": {"amount": number or null, "unit": the unit the request uses, "applies_to": "total" or "each"}.
- "objective": how to choose among listings that meet everything (e.g. "cheapest total", "lowest unit price",
  "best match", or "any that fits").
- "skip_sold_out": true if out-of-stock listings are unacceptable (assume true unless the shopper says otherwise).
- "message_limit": the most messages the shopper will accept from the assistant, as an integer, or null if unstated.
- "question_cost": text of any stated cost per question or message, else null.
- "step_limit": the most tool calls allowed if the request states one, else null.
- "procedures": every step the shopper says must happen besides finding and buying (for example: show them something
  or get their agreement before ordering, tell them something, read or check something first), each as {"step": what
  must happen, "when": the moment it applies (for example "just before ordering, with the final basket", "before
  anything goes in the cart", "at the start"), "needs_message": true if it means sending the shopper a message, "at_close": true if it must hold for the
  final basket or the order itself (so it is done once the cart is final, right before closing)}; steps due at the
  same moment get the same "when" wording; [] if none.
- "relations": rules that tie one requested product to another one picked (it must work with, fit, match or go
  with it), as [{"item": product, "rule": the rule in the request's words, "with": the other product}]; [] if none.
- "questions": the properties the request hints at without stating a value (an indirect remark on what they would
  notice, what they value above all, their intended use or setting, the person meant to use it, or what they
  offer to check or measure for you), most telling first, as [{"item": product,
  "property": property name}, ...] with "hinted": true on each. Empty if there are no such hints.
- "tool_roles": map each tool name to one role: search, view, compare, message, cart_change, cart_set, cart_view,
  checkout, decline, finish (one tool that both orders and declines), other.
- "finish_modes": for each finish tool, {"<tool>": {"order": the argument values (for its choice-type arguments)
  that place the order, "decline": the values that close with no purchase}}, read from that tool's schema; {} if
  there is no finish tool.
- "price_filter": a search tool that can limit results to a highest price, as {"tool": its name, "max_price_arg": the
  argument that sets the highest price, "category_arg": its argument that limits results to one product category (or
  null), "page_arg": its paging-token argument (or null)}, read from the tools' arguments; null if no tool has one.
- "paging": {"<search tool>": its argument that takes a paging token from an earlier result} for each search tool
  that has one, read from the tools' arguments; {} if none.
- "probes": one first search per item, [{"item": product, "tool": search tool name, "args": arguments for that tool
  (use its category filter when it has one, with the category as the shop writes it when the tools or request show
  that, and a short keyword: the product noun plus at most one stated value; a probe looks for the product
  to buy itself, never for a listing a requirement refers to, so no seller filter here; leave out paging or
  page-token arguments)}].
- "plan": 2-3 short steps on how to find the right listing(s) with these tools (never make closing with no purchase
  depend on a reply being worded in a way you expect)."""

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
   is merely compatible, broader, stronger, weaker or related is not the same (a value to be paired with what
   another listing shows is rule 2; wanted "rustic" is not met by
   "vintage"; wanted "lightly salted" is not met by "unsalted"). Two different words the shop's pages both use as
   values of the same property are different values, never synonyms (a value that holds the wanted word with a
   modifier still meets it). Assume listing
   prices are in the same unit as the shopper's budget unless the shop says otherwise. An exclusion ("not X") removes
   only listings whose value is X itself or a true synonym, not values that merely contain the word. A property the
   request only hints at (no value given) or one the shopper left open is no requirement and rules nothing out.
2. If a requirement or a reply refers to another listing ("the same X as the one sold by Y..."), find that listing: a
   keyword search with the seller's name and two or three of the values that describe it (no category words), the
   next result pages too when the first lacks it (pages are often short), and all rows from that seller opened
   in one page call so their notes show. If its page points you to another listing, find that one the same way and
   follow on until a page gives the value itself. Open each listing on that path with a page tool that shows the
   page's notes and description (a property list alone does not show where a page points). A filter that returned
   nothing for a seller or a value is no way to find it: use keywords.
   How the wanted value relates to the value found: the same value when it must match or equal it; any other
   value when it must differ; a bound when it must be at least or at most; and when it must work with, suit, go
   with, or be a counterpart to, what the page shows, the value that page says its own goes or pairs with (the
   same value only when the page names none). When the request wants the pointed-to listing itself, buy that one.
   A listing that only served as a reference is not bought unless it meets every requirement itself.
   Then look for that value in several ways before deciding no listing has it: the product noun with the value as
   keywords, the category filter, other words or spellings for the value, and the property's values on the pages
   you open (the same value may be written differently).
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
   again when a step the shopper requires calls for it, and never beyond the message limit.
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
   an item, a limit of nineteen and a half dollars is 1950, not 19.5).
11. When the request ties products together (one must work with, fit or match another you pick): choose the
   product it depends on first, then pick the other from listings whose own text says they work with, fit or match
   the one chosen (result rows often show that text already). When a page says more about it is on another listing,
   look that listing up (open it, or search the code it gives) before choosing.
12. The request may set steps of its own besides buying (the brief lists them as procedures). Do each at the moment
   it applies, so that it holds for the final state: a step about the final basket or the order comes after the cart
   holds exactly what you will buy, right before you close; if the cart changes after it, the step is due again.
   Steps due at the same moment go in one message. Any figure on the basket you pass on (its total, its lines, their
   stock) comes from a cart view made after the last cart change, as the shop shows it.
   Shopper replies seldom use the words you expect: a reply that raises no objection and asks for no change is
   agreement. Never send again a message the shopper has already answered. If a required step can no longer be done
   (for example no message is left), still close with the purchase when the cart meets the requirements: an order
   that misses a step may still be accepted, while closing with no purchase leaves the shopper with nothing."""

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
 "steps": one row per step the shopper requires (the brief's procedures): {"step": text, "done": true/false, "why": a
   few words},
 "cart_check": one row per stated requirement and told need of each CARTED line: {"id": listing id, "requirement":
   in the request's or reply's words, "page_value": what that listing's page says for the same property (or "not
   shown"), "met": true, false (the page shows another value), or null (the page shows nothing on it)},
 "blocker": if close_as is decline: the stated requirement or told need (in the request's, a reply's or an update's
   words) that every listing seen for that product fails by its own page facts, else null,
 "reason": one sentence}
Rules: required steps are judged on the ACTION TIMELINE: a step about the final basket or the order counts only if it
happened after the last cart change (a message the assistant is about to send, given below, counts when its text
does the step and the cart is final); a shopper reply that raises no objection and asks for no change is agreement. When
a step is not done and can still be done, the verdict is fix, with exact instructions for it (what to send or do), and
then the close. Once the shopper has agreed to the cart and no message is left to show them a change, ask for no cart
change unless a line clearly fails a stated requirement or a need they told. When the request ties products together (one must work with, fit or match another), each such line must
meet that rule with the carted line of the other product, by what their pages say. A shopper reply that is garbled, off the subject or not an answer adds no requirement. New words from the
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
When the request or a shopper reply refers to another listing, that listing and each one its notes point to must
have been opened with a page tool that shows notes ("opened_with"); if not, the fix opens them with it.
When a property is unknown only because one of the shop's page tools was not used on that listing ("opened_with"),
ask in the fix to open it with that tool. Listing prices are in
the same unit as the budget unless the shop says otherwise. Judge only by the shop's data and the shopper's words,
not by what the product seems to be for. The cart must hold exactly the products asked for, one line per
requested product with the requested quantity; an extra line is wrong. Stock and price count as of the freshest
page or result seen after the last cart change. For an out-of-stock line, the fix should first look for the same
product from another seller (search by its title words). Out-of-stock lines are
wrong when the shopper rejects them. The order total must respect the budget. When the shopper wants the cheapest,
a cheaper listing that meets everything and was seen in results is a reason to fix (say which id). A decline is
right only if no listing seen can meet everything and a reasonable search was made; its blocker is a stated
requirement or told need that each such listing's page contradicts (another value, another category, over the
budget, sold out when that matters). A value read off another listing's page is no blocker until it was searched
for in more than one way, and it must relate as the request asks: the same value, another value, a bound, the value
that page says its own pairs or goes with, or that pointed-to listing as the purchase. A cart line that only served as such a
reference (a page on the way to the value) is wrong unless the request wants that very listing or its own page
meets every requirement. A fact a page only does not show, a property the request only hints at or the
shopper left open, and a step that can no longer be done are never a blocker. Ask for a fix only when it is
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
        self.timeline: list[tuple] = []          # (step, tool, args, result) of every call
        self.lookup_tools: set[str] = set()
        self.id_cap: dict[str, int] = {}         # list-taking page tool -> most ids per call (learned when refused)
        self.id_fails: dict[str, int] = {}
        self.id_ok: dict[str, int] = {}          # list-taking page tool -> most ids a call has had accepted
        self.prop_tool: str | None = None        # the page tool whose pages record a listing's properties
        self.note_tools: set[str] = set()        # page tools whose pages carry free text (notes, description)

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
        self.timeline.append((self.step, name, json.dumps(args, ensure_ascii=False)[:200], _compact_obs(obs)[:220]))
        return obs

    def _learn(self, obs: Any, name: str = "", args: dict | None = None) -> None:
        _strings_in(obs, self.seen_strings)
        asked: set = set()
        _strings_in(args or {}, asked)
        is_page = name in self.page_tools
        if is_page and name not in self.note_tools and _has_note_text(obs):
            self.note_tools.add(name)
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


def _brief_text(ep: Episode, brief: dict) -> str:
    """The brief as the models read it: what the request asks, without the tool set-up the code uses itself (a part
    that names a tool) or the brief's own working notes."""
    tools = [json.dumps(t) for t in ep.by_name]
    keep = {k: v for k, v in brief.items() if v not in (None, [], {}) and k not in ("analysis", "questions", "step_limit")
            and not any(t in json.dumps(v, ensure_ascii=False) for t in tools)}
    first = ("items", "told_needs", "updates")      # the products, then what the shopper said since
    keep = {k: keep[k] for k in first if k in keep} | {k: v for k, v in keep.items() if k not in first}
    return json.dumps(keep, ensure_ascii=False, indent=1)


def _message_limit(brief: dict | None) -> int:
    lim = (brief or {}).get("message_limit")
    try:
        return max(0, int(lim))
    except (TypeError, ValueError):
        return 3


def _close_steps(brief: dict | None) -> list[str]:
    """The shopper's own steps that take a message once the cart is final, right before the order (from the brief)."""
    procs = (brief or {}).get("procedures")
    return [str(p["step"]) for p in (procs if isinstance(procs, list) else [])
            if isinstance(p, dict) and p.get("needs_message") and p.get("at_close") and p.get("step")]


def _step_due(ep: Episode, cart_tools: set, message_tools: set) -> bool:
    """No message has gone to the shopper since the cart last changed (or the cart was never set)."""
    last_cart = max((n for n, t, _, _ in ep.timeline if t in cart_tools), default=0)
    last_msg = max((n for n, t, _, _ in ep.timeline if t in message_tools), default=0)
    return last_msg <= last_cart


def _reserved_messages(brief: dict | None) -> int:
    """Messages kept back for the shopper's own steps: one per moment (steps due together share a message)."""
    moments = set()
    procs = (brief or {}).get("procedures")
    for p in procs if isinstance(procs, list) else []:
        if isinstance(p, dict) and p.get("needs_message"):
            moments.add(" ".join(sorted(_words(p.get("when") or p.get("step") or ""))))
    return len(moments)


# ---------------------------------------------------------------- asking

QUESTION_SYSTEM = """You pick the NEXT property a shop assistant asks a shopper about. The shopper has wishes about
some properties of the products that are left unsaid; each message may ask about a single property, and messages are
limited. You see the request, the brief, hints from the request, the questions already sent with the shopper's
replies, and the properties the shop's own pages record. Answer with the answer tool, or as JSON only:
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


def _number_twin(text: str) -> str:
    """The text with its last word in the other grammatical number (box -> boxes, glasses -> glass): shops name their
    shelves one way, and a category filter matches only the shop's own spelling."""
    head, _, last = str(text).rstrip().rpartition(" ")
    low = last.lower()
    if low.endswith("ies") and len(low) > 4:
        twin = last[:-3] + "y"
    elif low.endswith(("ses", "xes", "ches", "shes")):
        twin = last[:-2]
    elif low.endswith("s") and not low.endswith("ss"):
        twin = last[:-1]
    elif low.endswith("y") and len(low) > 2 and low[-2] not in "aeiou":
        twin = last[:-1] + "ies"
    elif low.endswith(("s", "x", "ch", "sh")):
        twin = last + "es"
    else:
        twin = last + "s"
    return (head + " " if head else "") + twin


def _names_item(text: str, words: set[str]) -> bool:
    """Whether a text (a listing's shelf, a filter value) shares a word with the product's name, in either number."""
    return any(w == u or _number_twin(w) == u for w in _words(text) for u in words)


def _phrase_words(text: str) -> set[str]:
    """A product phrase's naming words: its last word (the noun, however short) and its longer words, so the short
    joining words between them never make two names look alike."""
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return {w for w in words if len(w) > 3} | set(words[-1:])


def _item_words(brief: dict | None, item: str) -> set[str]:
    """The words naming a product: those of the brief's name for it and of its category."""
    words = _phrase_words(item)
    for it in (brief or {}).get("items") or []:
        if isinstance(it, dict) and _phrase_words(it.get("what") or "") & _phrase_words(item):
            words |= _phrase_words(it.get("what") or "") | _phrase_words(it.get("category") or "")
    return words


def _shelf_of(d: dict) -> str | None:
    """A listing's own category field, when the shop shows one."""
    for k, v in d.items():
        if isinstance(v, str) and v and re.search(r"categor|shelf|department|aisle", str(k), re.I):
            return v
    return None


def _probe(ep: Episode, brief: dict | None, roles: dict, per_item: int | None = None, per_call: int = 5) -> list[str]:
    """One search per item, then the property pages of some results (a page tool that takes several ids opens up to
    per_call of them in its one step; one that takes a single id opens per_item: 6 for one product, 4 each for several),
    so the shop's own property names, and which of them most pages record, are known before any question is sent."""
    notes: list[str] = []
    if not brief:
        return notes
    views = [t for t in (roles.get("compare") or []) + (roles.get("view") or []) if t in ep.by_name]
    keep = END_RESERVE + 10
    objective = str(brief.get("objective") or "").lower()
    cheap = "cheap" in objective or "low" in objective
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
        words = _item_words(brief, str(pr.get("item") or ""))
        named = [k for k, v in args.items() if isinstance(v, str) and _names_item(v, words)]
        if not pool and named and ep.left >= END_RESERVE + 10:
            # a category filter that matched nothing: the shop may name the shelf in the other number
            args = dict(args, **{k: _number_twin(args[k]) for k in named})
            obs = ep.call(name, args)
            notes.append(f"{name} {json.dumps(args, ensure_ascii=False)} -> {_compact_obs(obs)}")
            found = []
            _listings_in(obs, found, set())
            pool = [i for i in dict.fromkeys(i for i, _ in found) if i not in ep.viewed]
        if cheap and pool:
            # every page of a short result list, and its cheapest listings opened first
            more, extra = _more_pages(ep, brief, name, args, obs, keep, pages=4 if len(probes) <= 1 else 2)
            notes += more
            pool = [i for i in dict.fromkeys(pool + extra) if i not in ep.viewed]
            if brief.get("skip_sold_out", True):
                pool = [i for i in pool if not _sold_out(ep.listings.get(i) or {})] or pool
            pool.sort(key=lambda i: _listing_price(ep.listings.get(i) or {}) or float("inf"))
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
        pool = _on_shelf(ep, brief, pr, name, args, obs, pool, words, keep, notes)
        for i in pool[:per_call]:
            ep.probe_of.setdefault(i, str(pr.get("item") or ""))
        ids = pool[:per_item]
        # until a page tool shows properties, each one is tried on a first look (a list-taking one on all of them)
        for t in ([ep.prop_tool] if ep.prop_tool else views):
            if ep.finished or ep.left <= keep or not ids:
                break
            first = _open_pages(ep, t, pool[:per_call] if _list_arg(ep.by_name[t]) else ids[:1], keep, once=True)
            notes += [f"{t} {json.dumps(a, ensure_ascii=False)} -> {_compact_obs(o, True)}" for a, o in first]
            if any(_has_props(o) for _, o in first):
                ep.prop_tool = t
                break
        if ep.prop_tool and not _list_arg(ep.by_name[ep.prop_tool]):
            rest = [i for i in ids if not (ep.opened_by.get(i) or set()) & {ep.prop_tool}]
            notes += [f"{ep.prop_tool} {json.dumps(a, ensure_ascii=False)} -> {_compact_obs(o, True)}"
                      for a, o in _open_pages(ep, ep.prop_tool, rest, keep)]
    return notes


def _on_shelf(ep: Episode, brief: dict | None, pr: dict, name: str, args: dict, obs: Any, pool: list[str],
              words: set[str], keep: int, notes: list[str]) -> list[str]:
    """The probe's listings that the shop shelves as the product itself: a search for one product also returns
    others that only mention it (a lamp named after the furniture it lights), and questions planned from their pages ask about the wrong
    things. When none of the rows is on the product's shelf, the next result page is read once; if that finds none
    either, the rows are kept as they were (the shop may shelve the product under another name)."""
    def fit(ids: list[str]) -> list[str]:
        shelves = {i: _shelf_of(ep.listings.get(i) or {}) for i in ids}
        if not any(shelves.values()) or not words:
            return ids
        return [i for i in ids if shelves[i] and _names_item(shelves[i], words)]
    got = fit(pool)
    if got or not pool:
        return got
    arg, token = _paging_arg(ep, brief, name), _page_token(obs)
    if arg and token and ep.left > keep:
        a = _coerce_args({**args, arg: token}, _schema(ep.by_name[name]))
        o = ep.call(name, a)
        notes.append(f"{name} {json.dumps(a, ensure_ascii=False)} -> {_compact_obs(o)}")
        more: list = []
        _listings_in(o, more, set())
        got = fit([i for i in dict.fromkeys(i for i, _ in more) if i not in ep.viewed])
    return got or pool


def _empty_filter_note(ep: Episode, name: str, args: dict, obs: Any) -> str:
    """A search filter set to a seller or a shelf that matched nothing: the shop keeps or spells that value otherwise,
    so the light model is pointed to keywords (a seller's name and a listing's values work as keywords)."""
    found: list = []
    _listings_in(obs, found, set())
    props = _schema(ep.by_name[name]).get("properties") or {} if name in ep.by_name else {}
    fields = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string"
              and not re.search(r"next|token|cursor|after|onward|marker|page", k, re.I)]
    filters = [k for k, v in (args or {}).items() if k in fields and isinstance(v, str) and v.strip()]
    if found or len(fields) < 2 or not filters or _result_count(obs) not in (0, None):
        return ""
    return ("\nNothing matched " + ", ".join(f"{k}={args[k]!r}" for k in filters) + ": the shop writes or keeps "
            "that value otherwise. Search with keywords instead (a seller's name plus a few of the listing's values), "
            "or write a shelf name as the shop's result rows show it.")


def _failed(obs: Any) -> bool:
    """A call the shop turned down: an error or issue reply that holds no listing, or a bare status code."""
    if not isinstance(obs, dict) or not obs:
        return False
    found: list = []
    _listings_in(obs, found, set())
    if found:
        return False
    return (any(re.search(r"err|issue|fail|refus|invalid|denied", str(k), re.I) for k in obs)
            or all(isinstance(v, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", v) for v in obs.values()))


def _first_batch(ep: Episode, name: str) -> int:
    """How many ids a list-taking page tool gets per call: the size the shop taught (a refusal), else five or the
    declared most if fewer; shops often declare more than they answer."""
    if name in ep.id_cap:
        return ep.id_cap[name]
    arr = _list_arg(ep.by_name[name]) if name in ep.by_name else None
    most = ((_schema(ep.by_name[name]).get("properties") or {}).get(arr) or {}).get("maxItems") if arr else None
    return min(FIRST_BATCH, most) if isinstance(most, int) and most > 0 else FIRST_BATCH


def _list_arg(tool: dict) -> str | None:
    props = _schema(tool).get("properties") or {}
    return next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array"), None)


def _open_pages(ep: Episode, name: str, ids: list[str], keep: int, once: bool = False) -> list[tuple[dict, Any]]:
    """Open listings with a page tool while more than `keep` steps are left: a tool that takes a list gets them in
    batches, made smaller when the shop turns one down (the size that works is kept for the episode); one that takes
    a single id gets them singly. Stops at a refusal it cannot get round, or after one accepted call (once)."""
    tool = ep.by_name[name]
    arr = _list_arg(tool)
    key = arr or _text_arg(tool)
    out: list[tuple[dict, Any]] = []
    todo = list(dict.fromkeys(ids))
    while key and todo and not ep.finished and ep.left > keep:
        batch = todo[:max(1, min(_first_batch(ep, name), len(todo)))] if arr else todo[:1]
        vargs = _coerce_args({key: batch if arr else batch[0]}, _schema(tool))
        obs = ep.call(name, vargs)
        out.append((vargs, obs))
        if not _failed(obs) and arr:
            ep.id_ok[name] = max(ep.id_ok.get(name, 0), len(batch))
        if _failed(obs):
            ep.id_fails[name] = ep.id_fails.get(name, 0) + 1
            if arr and len(batch) > 1 and ep.id_fails[name] < 3:
                # perhaps too many at once: one fewer, then half as many
                ep.id_cap[name] = len(batch) - 1 if ep.id_fails[name] == 1 else max(1, len(batch) // 2)
                continue
            break
        todo = [] if once else todo[len(batch):]
    return out


def _fit_list(ep: Episode, name: str, args: dict) -> tuple[dict, str]:
    """Trim a list of ids to the most this tool is known to take per call; the note says so."""
    arr = _list_arg(ep.by_name[name]) if name in ep.by_name else None
    cap = _first_batch(ep, name) if arr else None
    if not arr or not cap or not isinstance(args.get(arr), list) or len(args[arr]) <= cap:
        return args, ""
    return ({**args, arr: args[arr][:cap]},
            f"\n(This tool takes at most {cap} ids per call here: only the first {cap} were sent.)")


def _refusal_note(ep: Episode, name: str, args: dict, obs: Any) -> str:
    """After a refused call with a list of ids: remember that fewer must be sent, and say so (an accepted one: how
    many it took)."""
    arr = _list_arg(ep.by_name[name]) if name in ep.by_name else None
    if arr and isinstance(args.get(arr), list) and not _failed(obs):
        ep.id_ok[name] = max(ep.id_ok.get(name, 0), len(args[arr]))
    if not arr or not _failed(obs) or not isinstance(args.get(arr), list) or len(args[arr]) < 2:
        return ""
    n = len(args[arr])
    ep.id_cap[name] = min(ep.id_cap.get(name, 99), max(ep.id_ok.get(name, 1), n - 1 if n <= 4 else n // 2))
    return (f"\n(The shop turned down {len(args[arr])} ids in one call: send at most {ep.id_cap[name]} per call to "
            f"this tool.)")


def _paging_arg(ep: Episode, brief: dict | None, name: str) -> str | None:
    """A search tool's paging-token argument: as the brief read it, else a text argument named for paging."""
    props = _schema(ep.by_name[name]).get("properties") or {}
    pg = (brief or {}).get("paging")
    if isinstance(pg, dict) and pg.get(name) in props:
        return pg[name]
    return next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string"
                 and re.search(r"next|token|cursor|after|onward|marker|page", k, re.I)), None)


def _result_count(obs: Any) -> int | None:
    """How many listings a search says it matched (a top-level count field), if it says."""
    if not isinstance(obs, dict):
        return None
    for k, v in obs.items():
        if isinstance(v, int) and not isinstance(v, bool) and re.search(r"count|found|total|match|hits", k, re.I):
            return v
    return None


def _more_pages(ep: Episode, brief: dict | None, name: str, args: dict, obs: Any, keep: int,
                pages: int = 2) -> tuple[list[str], list[str]]:
    """For a lowest-price request: when a search matched only a few pages more than it showed, read those pages too
    (the cheapest listing is often on one of them when results are not sorted by price)."""
    found: list = []
    _listings_in(obs, found, set())
    rows, total, token = len(found), _result_count(obs), _page_token(obs)
    arg = _paging_arg(ep, brief, name) if name in ep.by_name else None
    if not (rows and total and token and arg) or total <= rows or total > (pages + 1) * rows or _order_note(obs):
        return [], []
    out, ids = [], []
    while token and pages > 0 and not ep.finished and ep.left > keep:
        pages -= 1
        a = _coerce_args({**args, arg: token}, _schema(ep.by_name[name]))
        o = ep.call(name, a)
        text = _compact_obs(o)
        ep.memo[name + json.dumps(a, sort_keys=True)] = text
        out.append(f"{name} {json.dumps(a, ensure_ascii=False)} -> {text}")
        more: list = []
        _listings_in(o, more, set())
        ids += [i for i, _ in more]
        nxt = _page_token(o)
        token = nxt if nxt and nxt != token and not _failed(o) else None
    return out, ids


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
    mixed: many values join several choices, rank: pages' share plus how much the values differ}."""
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
            row["mixed"] = row["joined"] >= 2 and row["joined"] * 3 >= row["pages"]
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
    user = (f"SHOPPER REQUEST:\n{request}\n\nBRIEF:\n{_brief_text(ep, brief)}\n\nHINTS FROM THE REQUEST (in the "
            f"shopper's words): {json.dumps(hints, ensure_ascii=False)}\n\nALREADY STATED BY THE SHOPPER, BY "
            f"PRODUCT (never ask these for that product): "
            f"{json.dumps({k: sorted(v) for k, v in stated.items()}, ensure_ascii=False)}\n\nQUESTIONS SENT AND REPLIES:\n"
            f"{history}\n\nPROPERTY TABLE (from the pages opened):\n{_table_text(ep, _prop_table(ep))}\n\n"
            f"SHOP PAGES SEEN:\n{_page_digest(ep)}\n\nMessages left after this one: {left - 1}." +
            (f"\n\nPRODUCT TO ASK ABOUT NOW: {focus}" if focus else ""))
    got = _ask_json([LIGHT, HEAVY], QUESTION_SYSTEM, user, max_tokens=2500,
                    fields={"item": "the product", "property": "the property's name as the pages write it",
                            "why": "a few words"})
    prop = " ".join(str((got or {}).get("property") or "").replace("_", " ").split())
    if not prop:
        return None
    items = [str(i.get("what")) for i in brief.get("items") or [] if isinstance(i, dict) and i.get("what")]
    item = " ".join(str(got.get("item") or focus or (items[0] if items else "item")).split())
    # one plain form for every question: the shop's own name for the property, and the product it concerns (a name
    # given as alternatives, "A or B", reads as two questions: the first one names it)
    noun = re.split(r"\s+or\s+", item)[0] or item
    got.update(property=prop, item=item, text=f"Which {prop} would you like the {noun} to have?")
    return got


POINTER_SYSTEM = """You help an assistant in an online shop simulation reach the listing a shopper's requirement points
to and read one property off it. You get the requirement, the listing wanted and, after a search, the pages opened.
Answer with the answer tool, or as JSON only:
{"property": the property whose value is wanted (as the requirement names it),
 "listing": the id of the opened page that is the listing wanted (its seller and the details given), or null,
 "value": the value that page itself gives for the property, or null,
 "pairs_with": the value that page says its own value goes or pairs with, or null,
 "next": null, or, when the page names a further listing for the property instead of giving it (or when no
   page is opened yet: the listing the requirement points to), that listing as {"seller": the seller's name only
   (a note or requirement that names a listing by its maker or seller gives it), or null if none is named, "kind":
   which kind of product it is, "details": a list of a few of its values as written, without property names}}
A page's note that sends you to another listing for this property outweighs the value the page itself lists: then
give that listing in "next" and no value. Never guess a value: read it from the page text."""


REPLIES_SYSTEM = """You read a shopper's replies to an assistant's questions in an online shop simulation and list
the wishes they told. A reply that only says they have no preference, did not follow the question, or wants it put
again tells nothing. Answer with the answer tool, or as JSON only:
{"needs": [{"item": the product it is about, "need": "property: wanted value" in the shopper's words (keep numbers,
units and bounds such as at least or at most), "reference": null, or, when that value is to be read off a
different listing's page: who sells that listing, which kind of product it is, which details identify it, which of its
properties to read, and how the wanted value must relate to it (same, different, at least, at most, or the value
that page pairs it with)}]}"""


def _as_dict(value: Any) -> dict | None:
    """An answer field that should hold an object: as given, or parsed from its JSON text."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) and value else None


def _keyword_tool(ep: Episode, roles: dict) -> tuple[str, str] | None:
    """A search tool that takes free keywords: the one whose only text argument (paging aside) is the query."""
    fallback = None
    for name in roles.get("search") or []:
        if name not in ep.by_name:
            continue
        props = _schema(ep.by_name[name]).get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string"
                 and not re.search(r"next|token|cursor|after|onward|marker|page", k, re.I)]
        if len(texts) == 1:
            return name, texts[0]
        if texts and not fallback:
            fallback = (name, texts[0])
    return fallback


def _detail_values(details: Any) -> list[str]:
    """A listing's identifying values as the model gave them (a list, an object of property: value, or text)."""
    if isinstance(details, dict):
        details = [v for v in details.values() if not isinstance(v, (dict, list))]
    elif isinstance(details, str):
        details = re.split(r"[;,]", details)
    return [str(d).split(":", 1)[-1].strip() for d in details or [] if str(d).strip()] if isinstance(details, list) else []


def _seller_name(target: dict) -> str:
    """The seller's name if the model gave one (a name is a word or two, not a sentence)."""
    seller = str(target.get("seller") or "").strip()
    return seller if seller and len(seller.split()) <= 3 and seller.lower() not in ("null", "none", "unknown") else ""


def _given(value: Any) -> Any:
    return None if value is None or str(value).strip().lower() in ("", "null", "none", "n/a") else value


def _sold_by(row: dict, seller: str) -> bool:
    low = seller.strip().lower()
    return bool(low) and any(isinstance(v, str) and v.strip().lower() == low for v in row.values())


def _pages_text(ep: Episode, ids: list[str]) -> str:
    """The opened pages of these listings, their texts in full (a note names the next listing at its end)."""
    rows = []
    for lid in ids:
        d = ep.listings.get(lid) or {}
        rows.append(json.dumps({k: v for k, v in d.items() if not str(k).startswith("_")}, ensure_ascii=False)[:2500])
    return "\n".join(rows)


def _seller_batches(ep: Episode, brief: dict | None, search: tuple[str, str], target: dict, keep: int):
    """Listings of the seller named, page by page. Keywords: the seller's name with two short values of the listing,
    then the seller with the kind of product, then the seller alone (long keyword lists drift to other sellers). Each
    result page's rows from that seller come as one batch; the caller asks for the next page only when the listing
    wanted was not among them, and a page without the seller's rows ends that query."""
    name, arg = search
    seller = _seller_name(target)
    short = [d for d in _detail_values(target.get("details")) if len(d.split()) <= 3 and "(" not in d][:2]
    kind = " ".join(str(target.get("kind") or "").split()[:4])
    queries = [" ".join([seller] + short), " ".join([seller, kind]), seller] if seller else [" ".join(short + [kind])]
    page_arg = _paging_arg(ep, brief, name)
    limit = ((_schema(ep.by_name[name]).get("properties") or {}).get(arg) or {}).get("maxLength") or 200
    given: set = set()
    for q in dict.fromkeys(x.strip() for x in queries if x.strip()):
        args = {arg: q[:limit]}
        for _ in range(3):
            if ep.finished or ep.left <= keep:
                return
            obs = ep.call(name, _coerce_args(args, _schema(ep.by_name[name])))
            found: list = []
            _listings_in(obs, found, set())
            rows = [i for i, r in found if (_sold_by(r, seller) if seller else True)]
            mine = [i for i in rows if i not in given]
            given |= set(mine)
            if mine:
                yield mine
            token = _page_token(obs)
            if not rows or not page_arg or not token or not seller:
                break
            args = {**args, page_arg: token}


def _follow_references(ep: Episode, brief: dict | None, roles: dict, told_refs: list[dict], keep: int) -> list[str]:
    """Pointers to other listings, followed by code: the light model says which listing a requirement (or a told
    wish) points to and, on each page opened, whether it gives the value or names a further listing; the code
    searches, pages and opens the seller's listings in batches the shop accepts, until a page gives the value."""
    if not brief:
        return []
    refs = [(it.get("what"), it.get("reference")) for it in brief.get("items") or []
            if isinstance(it, dict) and it.get("reference")]
    refs += [(n.get("item"), n.get("reference")) for n in told_refs]
    search = _keyword_tool(ep, roles)
    views = [t for t in sorted(ep.note_tools) if t in ep.by_name] or \
            [t for t in (roles.get("compare") or []) + (roles.get("view") or []) if t in ep.by_name]
    if not refs or not search or not views:
        return []
    view = next((t for t in views if _list_arg(ep.by_name[t])), views[0])
    fields = _answer_fields(POINTER_SYSTEM)
    notes = []
    for item, ref in refs[:3]:
        if ep.finished or ep.left <= keep:
            break
        need = f"REQUIREMENT ON {item}: {ref if isinstance(ref, str) else json.dumps(ref, ensure_ascii=False)}"
        got = _ask_json([LIGHT, HEAVY], POINTER_SYSTEM, need + "\n\nPAGES OPENED: none yet", max_tokens=1200,
                        fields=fields) or {}
        prop, target = got.get("property"), _as_dict(got.get("next"))
        path, value, pairs, lost = [], None, None, None
        for _ in range(6):
            if not target or ep.finished or ep.left <= keep:
                break
            matched = False
            for ids in _seller_batches(ep, brief, search, target, keep):
                ids = [i for i in ids if i not in path]
                if not ids:
                    continue
                _open_pages(ep, view, ids[:2 * _first_batch(ep, view)], keep)
                got = _ask_json([LIGHT, HEAVY], POINTER_SYSTEM,
                                f"{need}\nPROPERTY: {prop}\nLISTING WANTED: {json.dumps(target, ensure_ascii=False)}"
                                + (f"\nALREADY READ ON THE WAY (not the listing wanted): {', '.join(path)}" if path else "")
                                + f"\n\nPAGES OPENED:\n{_pages_text(ep, ids)}", max_tokens=1200, fields=fields) or {}
                lid = got.get("listing")
                if isinstance(lid, str) and lid in ep.listings and lid not in path:
                    matched = True
                    path.append(lid)
                    break
            if not matched:
                lost = target
                break
            value, pairs, target = _given(got.get("value")), _given(got.get("pairs_with")), _as_dict(got.get("next"))
            if value:
                break
        text = f"For {item} ({prop}): pages read in order {' -> '.join(path) or '(none)'}"
        if value:
            text += f"; the last page gives {value!r}" + (f" and says it goes with {pairs!r}" if pairs else "")
        elif lost:
            text += ("; then this listing was not found by its seller's name: " + json.dumps(lost, ensure_ascii=False)
                     + " (try other words from the note that names it)")
        elif target:
            text += "; not finished, it points on to: " + json.dumps(target, ensure_ascii=False)
        notes.append(text)
    return notes


def _reply_needs(ep: Episode, brief: dict | None, request: str) -> list[dict]:
    """The wishes the shopper told in replies, read once after the questions and kept on the brief, so the loop and
    the checks see them as requirements (a told value may itself have to be read off another listing's page)."""
    if not brief or not ep.qa:
        return []
    user = (f"SHOPPER REQUEST:\n{request}\n\nPRODUCTS: " +
            json.dumps([it.get("what") for it in brief.get("items") or [] if isinstance(it, dict)], ensure_ascii=False) +
            "\n\nQUESTIONS AND REPLIES:\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in ep.qa))
    got = _ask_json([LIGHT, HEAVY], REPLIES_SYSTEM, user, max_tokens=1500,
                    fields={"needs": "the told wishes as a JSON list of {item, need, reference} objects"})
    needs = (got or {}).get("needs")
    if isinstance(needs, str):
        try:
            needs = json.loads(needs)
        except ValueError:
            needs = None
    needs = [n for n in needs if isinstance(n, dict) and n.get("need")] if isinstance(needs, list) else []
    if needs:
        brief["told_needs"] = needs
    return needs


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
        parts.append("BRIEF:\n" + _brief_text(ep, brief))
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
    acts = [f"step {n}: {t} {a} -> {o}" for n, t, a, o in ep.timeline if t not in ep.lookup_tools]
    if acts:
        parts.append("ACTION TIMELINE (every call besides searches and page views, oldest first):\n" +
                     "\n".join(acts[-25:]))
    parts.append(f"STEPS LEFT: {ep.left}")
    return "\n\n".join(parts)


def _review(ep: Episode, request: str, brief: dict | None, closing: str) -> dict | None:
    user = _dossier(ep, request, brief) + f"\n\nTHE ASSISTANT NOW WANTS TO CLOSE AS: {closing}"
    return _ask_json([HEAVY, HEAVY_SPARE], REVIEW_SYSTEM, user, max_tokens=6000)


def _cart_misses(verdict: dict, cart: list[str]) -> str | None:
    """Carted lines the check found failing a requirement by their own page (another value shown), with a listing
    the same check found meeting everything for that product when there is one."""
    rows = [r for r in verdict.get("cart_check") or [] if isinstance(r, dict) and r.get("met") is False
            and str(r.get("id") or "") in cart]
    if not rows:
        return None
    item_of = {str(r.get("id")): str(r.get("item") or "").strip().lower()
               for r in verdict.get("table") or [] if isinstance(r, dict)}
    good = [r for r in verdict.get("table") or [] if isinstance(r, dict) and r.get("meets_all") is True
            and r.get("in_stock") is not False and str(r.get("id") or "") not in cart]
    out = []
    for lid in dict.fromkeys(str(r["id"]) for r in rows):
        why = "; ".join(f'{r.get("requirement")} (page: {r.get("page_value")})' for r in rows if str(r["id"]) == lid)
        alt = [r for r in good if item_of.get(str(r.get("id"))) == item_of.get(lid)]
        if verdict.get("cheapest_wins"):
            alt.sort(key=lambda r: _number(r.get("price")) or float("inf"))
        out.append(f"{lid} fails {why}: " + (f"replace it with {alt[0]['id']}, which meets everything" if alt else
                                             "find a listing for that product that meets it and replace the line"))
    return "; ".join(out)


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
A property the request only hints at (no value given) or one the shopper left open rules nothing out.
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


COUNT_KEY = re.compile(r"qty|quant|count|piece|units?$|number", re.I)
MONEY_KEY = re.compile(r"price|cost|total|amount|sum", re.I)


def _count_in(obs: Any, lid: str) -> float | None:
    """The count a cart reply gives for one listing (its line's numeric count field), or None."""
    found: list[float] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            if lid in [str(v) for v in o.values() if isinstance(v, (str, int))]:
                for k, v in o.items():
                    if (isinstance(v, (int, float)) and not isinstance(v, bool) and COUNT_KEY.search(str(k))
                            and not MONEY_KEY.search(str(k))):
                        found.append(float(v))
                        return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obs)
    return found[0] if found else None


def _cart_now(ep: Episode) -> dict[str, float]:
    """The cart as listing -> count: the latest cart view when nothing changed since, else the cart calls made."""
    if ep.cart_obs is not None and not ep.cart_dirty:
        lines = {i: _count_in(ep.cart_obs, i) for i in _cart_ids(ep)}
        if lines and all(n is not None for n in lines.values()):
            return {i: n for i, n in lines.items() if n and n > 0}
    return {i: n for i, n in ep.cart_set.items() if n > 0}


def _cart_sig(ep: Episode) -> tuple:
    return tuple(sorted(_cart_now(ep).items()))


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


def _unchecked_cheaper(ep: Episode, brief: dict | None, limit: int = 5) -> list[tuple[str, float]]:
    """For a lowest-price request: per carted line, listings seen in results below that line's price, in its
    category, not sold out (and within the budget for a one-line cart), whose pages were never opened (cheapest
    first; with several lines, a few for each)."""
    cart = _cart_ids(ep) or [i for i, n in ep.cart_set.items() if n > 0]
    if not cart:
        return []
    budget = (brief or {}).get("budget") if isinstance((brief or {}).get("budget"), dict) else {}
    cap = _number(budget.get("amount")) if str(budget.get("applies_to") or "total") == "total" and len(cart) == 1 \
        else None
    each = limit if len(cart) == 1 else max(2, limit // len(cart) + 1)
    found: dict[str, float] = {}
    for line in cart:
        held = ep.listings.get(line) or {}
        price = _listing_price(held)
        if price is None:
            continue
        cat = _category_value(ep, line, brief)
        cat_key = next((k for k, v in held.items() if cat and v == cat), None)
        if len(cart) > 1 and not cat_key:
            continue  # with several products, only a shared category says a listing could take this line's place
        out = []
        for lid, d in ep.listings.items():
            if lid in ep.viewed or lid in cart or lid in found:
                continue
            p = _listing_price(d)
            if p is None or p >= price or (cap is not None and p > cap):
                continue
            if (brief or {}).get("skip_sold_out", True) and _sold_out(d):
                continue
            if cat_key and d.get(cat_key) is not None and d.get(cat_key) != held.get(cat_key):
                continue
            if len(cart) > 1 and d.get(cat_key) is None:
                continue
            out.append((p, lid))
        for p, lid in sorted(out)[:each]:
            found[lid] = p
    return sorted(found.items(), key=lambda kv: kv[1])[:max(limit, 2 * len(cart))]


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
    items = [it for it in (brief or {}).get("items") or [] if isinstance(it, dict)]
    for it in items:
        wanted |= _words(f"{it.get('what') or ''} {it.get('category') or ''}")
    floor = 0.3 / max(1, len(items))  # several products: each one's category is a smaller share of what was seen
    best, score = None, 0.0
    for k, v in d.items():
        if k.startswith("_") or not isinstance(v, str) or not v or len(v) > 80 or _code_like(v) or v == lid:
            continue
        having = [x.get(k) for x in ep.listings.values() if isinstance(x.get(k), str)]
        if len(having) < 3:
            continue
        share = sum(1 for x in having if x == v) / len(having)
        s = share + (0.5 if _words(v) & wanted else 0.0)
        if share >= floor and s > score:
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
        lowered, before = 0, None
        for _ in range(5):
            if ep.finished or ep.left < END_RESERVE + 8:
                break
            obs = ep.call(pf["tool"], _coerce_args(args, schema))
            found: list = []
            _listings_in(obs, found, set())
            if not found and before:
                # the lower cap left nothing: page on under the cap before it
                args, obs = before
                before, lowered = None, 9
                token = _page_token(obs)
                if not page_arg or not token:
                    break
                args = {**args, page_arg: token}
                continue
            total = _result_count(obs)
            if found and total and total > 3 * len(found) and lowered < 2 and page_arg not in args:
                # more cheaper listings than three short pages show: the cheapest are found sooner under a lower
                # cap (the rows seen so far stay candidates)
                before = (args, obs)
                lowered += 1
                cap = args[max_arg] / 2
                args = {**args, max_arg: int(cap) if float(price).is_integer() else round(cap, 2)}
            else:
                lowered, before = 9, None
            for i, r in found:
                p = _listing_price(r)
                if p is None or p >= price or i in ep.viewed or i in lines:
                    continue
                if (brief or {}).get("skip_sold_out", True) and _sold_out(r):
                    continue
                text = " ".join(str(v) for v in r.values() if isinstance(v, str))
                cands[i] = (len(_words(text) & wanted), p)
            if lowered < 9:
                continue
            token = _page_token(obs)
            if not found or not page_arg or not token:
                break
            args = {**args, page_arg: token}
    if not cands:
        return None
    order = sorted(cands, key=lambda i: (-cands[i][0], cands[i][1]))
    views = [t for t in (roles.get("compare") or []) + (roles.get("view") or []) if t in ep.by_name]
    opened, shown = 0, []
    for t in ([ep.prop_tool] if ep.prop_tool else views[:1]):
        want = order[:12] if _list_arg(ep.by_name[t]) else order[:3]
        for a, o in _open_pages(ep, t, want, END_RESERVE + 5):
            shown.append(f"{t} {json.dumps(a, ensure_ascii=False)} -> {_compact_obs(o)}")
            if not _failed(o):
                got = next(iter(a.values()), [])
                opened += len(got) if isinstance(got, list) else 1
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
    return {k: {kind: _mode_pairs(want) for kind, want in v.items()} for k, v in raw.items() if isinstance(v, dict)}


def _mode_pairs(want: Any) -> Any:
    """A way of closing the brief wrote as text ("Arg=value", "Arg: value", several joined by commas), read as the
    arguments it names; any other form is kept as written."""
    if isinstance(want, str):
        pairs = re.findall(r"([A-Za-z_][\w-]*)\s*[=:]\s*['\"]?([^,;'\"]+?)['\"]?\s*(?:[,;]|$)", want)
        if pairs:
            return {k: v.strip() for k, v in pairs}
    return want


CLOSING_SYSTEM = """You write the one message a shop assistant sends a shopper just ahead of the purchase in an
online shop simulation. It must do what the shopper asked to have done at this moment (the steps listed), using only
the figures and lines of the cart view given, in plain short words. Answer with the answer tool, or as JSON only:
{"message": the text to send}"""


def _closing_message(ep: Episode, roles: dict, brief: dict | None) -> None:
    """Before a closing order the code makes itself: send the message a final-cart step needs, if one is left and
    none went out since the cart last changed (the cart viewed first when steps allow)."""
    steps = _close_steps(brief)
    tools = [t for t in roles.get("message") or [] if t in ep.by_name]
    cart_tools = set(roles.get("cart_change", []) + roles.get("cart_set", []))
    if not steps or not tools or not ep.carted or ep.finished or ep.left < 2:
        return
    sent = sum(1 for _, t, _, _ in ep.timeline if t in ep.message_tools)
    if sent >= _message_limit(brief) or not _step_due(ep, cart_tools, ep.message_tools):
        return
    view = (roles.get("cart_view") or [None])[0]
    if view in ep.by_name and ep.left >= 3:
        ep.cart_obs = ep.call(view, _coerce_args({}, _schema(ep.by_name[view])))
    user = (f"SHOPPER REQUEST:\n{ep.policy.get('query') or ''}\n\nSTEPS TO DO NOW:\n" + "\n".join(steps) +
            "\n\nCART VIEW:\n" + json.dumps(ep.cart_obs, ensure_ascii=False)[:3000])
    got = _ask_json([LIGHT, HEAVY], CLOSING_SYSTEM, user, max_tokens=1200, fields={"message": "the text to send"})
    text = str((got or {}).get("message") or "").strip()
    arg = _text_arg(ep.by_name[tools[0]])
    if text and arg and not ep.finished and ep.left >= 2:
        ep.call(tools[0], _coerce_args({arg: text}, _schema(ep.by_name[tools[0]])))


def _fallback_close(ep: Episode, roles: dict, brief: dict | None) -> None:
    """Leave no episode open: order what the cart holds, else close with no purchase."""
    if ep.finished or ep.left <= 0:
        return
    try:
        _closing_message(ep, roles, brief)
    except Exception:
        pass
    modes = _finish_modes(brief)
    plans = []
    if ep.carted:
        plans += [(t, {}) for t in roles.get("checkout", [])]
        plans += [(t, _mode_args(ep, t, modes.get(t, {}).get("order"))) for t in roles.get("finish", [])]
    plans += [(t, {}) for t in roles.get("decline", [])]
    plans += [(t, _mode_args(ep, t, modes.get(t, {}).get("decline"))) for t in roles.get("finish", [])]
    for _ in range(3):  # a shop may turn a close down once (it brings news at checkout): try again
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
        if ep.finished or ep.left <= 0:
            break


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
    _spend.scale = 1.0
    _spend.prompts = {}
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
    told_refs = [n for n in _reply_needs(ep, brief, request) if n.get("reference")]
    ref_notes = _follow_references(ep, brief, roles, told_refs, END_RESERVE + 10)
    msgs_sent = len(ep.qa)
    msg_limit = _message_limit(brief)

    intro = [f"SHOPPER REQUEST:\n{request}"]
    if ep.policy.get("opening") is not None:
        intro.append("OPENING NOTE:\n" + json.dumps(ep.policy["opening"], ensure_ascii=False))
    if brief:
        intro.append("BRIEF (from a careful reading of the request):\n" + _brief_text(ep, brief))
        if roles:
            intro.append("TOOL ROLES: " + json.dumps(roles))
    if probe_notes:
        intro.append("FIRST LOOKS AT THE SHOP (already done, steps used):\n" + "\n".join(probe_notes))
    if ep.qa:
        intro.append("QUESTIONS ALREADY SENT AND THE SHOPPER'S REPLIES:\n" +
                     "\n".join(f"Q: {q}\nA: {a}" for q, a in ep.qa) +
                     "\n(A reply that does not understand the question means that property is not something "
                     "they have a wish about.)")
    if ref_notes:
        intro.append("POINTERS FOLLOWED (already done, steps used; the pages are open):\n" + "\n".join(ref_notes) +
                     "\nApply each value as its requirement says it must relate (same, different, a bound, the value "
                     "it goes with, or that very listing as the purchase), then search for the listing to buy.")
    if told_refs:
        intro.append("A TOLD WISH NAMES A VALUE ON A DIFFERENT LISTING'S PAGE: " + json.dumps(told_refs, ensure_ascii=False) +
                     "\nTreat it like the request's own requirements: find that page early (the seller's name and "
                     "its details as keywords), follow where it points, and use the value as it must relate.")
    removal = _removal_hint(ep, roles)
    if removal:
        intro.append("CART: " + removal)
    if ep.note_tools and len(ep.page_tools) > 1:
        intro.append("PAGES WITH NOTES: in this shop " + ", ".join(sorted(ep.note_tools)) + " show a listing's notes "
                     "and description, where a page says which other listing to look at. When the request or a reply "
                     "refers to another listing's page, open it, and each one its notes point to, with one of them.")
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
    keep_told = False
    checked_sig: tuple | None = None    # the cart a strong review last let through to the order
    step_pushes = 0
    heavy_turn = False
    fix_turns = 0     # turns left in which a review's fix may still search and open pages after the wrap
    reopened = 0
    empty_turns = 0   # wrap turns with nothing in the cart, in which searching is still allowed
    decline_seen = False   # a decline went through the strong check once (a wrong decline loses the whole task)
    close_steps = _close_steps(brief)
    turns = 0
    pages: dict[str, int] = {}
    streak = 0
    lookups = set(roles.get("search", []) + roles.get("view", []) + roles.get("compare", []))
    ep.lookup_tools = lookups
    cart_tools = set(roles.get("cart_change", []) + roles.get("cart_set", []))
    view_cart = (roles.get("cart_view") or [None])[0]

    # a task with a large step allowance (several products, rules between them) gets somewhat more light-model work
    grow = 1.25 if ep.max_steps > 60 else 1.0
    # the light model's written tokens are what a task's run time grows with (validators stop a task at 300 s), so
    # its token lines stay the same for every task; only the turn line grows
    max_turns, work_wrap, work_stop = int(MAX_TURNS * grow), WORK_WRAP, WORK_STOP
    _spend.scale = 1.4 if ep.max_steps > 60 else 1.0  # and a larger spend line (such tasks are a small share)
    while not ep.finished and ep.step < ep.max_steps:
        forced = ep.left <= 2  # one step to close, one more if the shop turns the close down
        closers = [t for t in ep.tools if _tool_name(t) in
                   set(roles.get("checkout", []) + roles.get("decline", []) + roles.get("finish", []))]
        offer = closers if forced and closers else ep.tools
        if forced and closers:
            messages.append({"role": "user", "content": (
                "Final step now: close the episode with the purchase: the cart holds your best find, and an order "
                "is the shopper's only chance of getting it." if _cart_now(ep) else
                "Final step now: close the episode (the cart is empty, so with no purchase).")})
        elif (_spent() >= _hard() or turns >= max_turns or _written() >= work_wrap) and closers and not wrap_told:
            wrap_told = True
            keep = set(_tool_name(t) for t in closers) | cart_tools | ({view_cart} if view_cart else set()) | \
                (ep.message_tools if close_steps and msgs_sent < msg_limit else set())
            offer = [t for t in ep.tools if _tool_name(t) in keep]
            messages.append({"role": "user", "content": "Time to finish: put your best choice in the cart if it is "
                             "not there yet (the listing you found that meets the most requirements, in the product "
                             "category asked for and within the budget), do any step the shopper requires with the "
                             "final cart, then close the episode with the purchase. "
                             "Close with no purchase only if every listing you saw clearly fails a requirement by its "
                             "own page facts (a different value, another category, over the budget)."})
        elif idle_steps >= 4 and closers and not wrap_told:
            wrap_told = True
            keep = set(_tool_name(t) for t in closers) | cart_tools | ({view_cart} if view_cart else set()) | \
                (ep.message_tools if close_steps and msgs_sent < msg_limit else set())
            offer = [t for t in ep.tools if _tool_name(t) in keep]
            messages.append({"role": "user", "content": "You keep making calls that do nothing. Put your best "
                             "qualifying choice in the cart if it is not there yet, then close the episode."})
        elif wrap_told and closers:
            keep = set(_tool_name(t) for t in closers) | cart_tools | ({view_cart} if view_cart else set()) | \
                (ep.message_tools if close_steps and msgs_sent < msg_limit else set())
            offer = [t for t in ep.tools if _tool_name(t) in keep]
        fixing = fix_turns > 0 and not forced
        if fixing:
            fix_turns -= 1
            offer = ep.tools  # a review asked for more looking: it can only be done with the lookup tools
        elif (wrap_told and closers and not forced and empty_turns < FIX_TURNS and ep.left > END_RESERVE + 4
              and not _cart_now(ep)):
            # nothing chosen yet: with only the closing tools, the one close left would be no purchase
            empty_turns += 1
            fixing = True
            offer = ep.tools
        if _spent() >= 1.35 * _hard() and not fixing:
            over_turns += 1  # a few light turns to finish (cart, a step the shopper requires, the close), no more
        if idle_steps >= 8 or turns >= max_turns + 6 or _written() >= work_stop or over_turns > 4:
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
            if not calls and idle >= 3 and not heavy_turn:
                heavy_turn = True  # the light model keeps saying what it would do without doing it: one strong turn
                reply = _chat([HEAVY], messages + [{"role": "user", "content": "Call one of the tools now."}],
                              tools=offer, max_tokens=3000)
                calls = (reply or {}).get("tool_calls") or []
                content = (reply or {}).get("content") or content
        if not calls:
            messages += [{"role": "assistant", "content": content or "(no tool call)"},
                         {"role": "user", "content": "Call one of the tools now."}]
            continue
        idle = 0
        idle_steps += 1
        pagers = set(roles.get("view", []) + roles.get("compare", []))
        batch = [c for c in calls if isinstance(c, dict) and (c.get("function") or {}).get("name") in pagers]
        if len(calls) > 1 and len(batch) == len(calls) and (not wrap_told or fixing):
            # several page openings in one turn: each runs in order, one step each
            for n, c in enumerate(batch):
                c["id"] = c.get("id") or f"call_{ep.step}_{idle_steps}_{n}"
            messages.append({"role": "assistant", "content": content, "tool_calls": batch})
            fresh_all: list[str] = []
            for c in batch:
                nm = c["function"]["name"]
                a, trim = _fit_list(ep, nm, _coerce_args(_parse_args(c), _schema(ep.by_name[nm])))
                key = nm + json.dumps(a, sort_keys=True)
                if key in ep.memo:
                    out = "You already made this exact call (no step used). Its result was:\n" + ep.memo[key]
                elif ep.finished or ep.left <= END_RESERVE + 1:
                    out = "Not run: too few steps are left to open more pages."
                else:
                    before = ep.step
                    got = ep.call(nm, a)
                    out = _compact_obs(got, nm in ep.page_tools) + trim + _refusal_note(ep, nm, a, got)
                    fresh_all += ep.fresh
                    if ep.step > before:
                        idle_steps = 0
                        ep.memo[key] = out
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
            messages[-1]["content"] += "\n" + _state_line(ep, brief, msgs_sent)
            streak = 0
            if fresh_all:
                messages.append(_relay(fresh_all))
                _note_updates(ep, brief, fresh_all)
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
        args, trim = _fit_list(ep, name, _coerce_args(args, _schema(ep.by_name[name])))
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
        if (not decided and not ep.carted and _spent() < _hard()
                and (ep.left <= END_RESERVE + 3 or (_spent() >= _soft() and not wrap_told))
                and not _closing_kind(roles, name, args, brief)):
            decided = True
            pick = _decide(ep, request, brief)
            if pick:
                pick_lines = _pick_lines(pick)
                if str(pick.get("action", "")).lower() == "decline" and ep.left > END_RESERVE + 10:
                    # chosen early (spend, not steps): the light model may still search, and is pushed once more
                    note = ("Not done: a first decision from what you found so far is to close with no purchase:\n" +
                            json.dumps(pick, ensure_ascii=False) + f"\nYou still have {ep.left} steps: first make "
                            "a few more searches (other wordings, the category filter, the next result page) and open "
                            "the most promising pages; order a listing that meets every requirement if you find one, "
                            "otherwise close with no purchase.\n" + _state_line(ep, brief, msgs_sent))
                    messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": note})
                    reviews = MAX_REVIEWS
                    continue
                if str(pick.get("action", "")).lower() == "decline":
                    decline_pushed = True  # the strong model chose this decline: do not argue with it
                note = ("Not done: it is time to choose, so a decision was made from what you found:\n" +
                        json.dumps(pick, ensure_ascii=False) + "\n" + _cart_steps(ep, pick) +
                        "\nCarry it out now: put exactly these lines in the cart (if ordering), do any step the "
                        "shopper requires before closing, then close the episode as decided.\n" +
                        _state_line(ep, brief, msgs_sent))
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": note})
                reviews = MAX_REVIEWS
                continue

        # messages to the shopper: keep within the limit
        if name in roles.get("message", []):
            if msgs_sent >= msg_limit:
                messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                 "content": "Not sent: the message limit is reached. Continue without asking."})
                continue
            if close_steps and msg_limit - msgs_sent <= 1 and not _cart_now(ep):
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": (
                    "Not sent (no step used): this is your last message, and the shopper asked for this with the "
                    "final cart: " + "; ".join(close_steps) + ". Keep it for that: first put your choice into "
                    "the cart.\n" + _state_line(ep, brief, msgs_sent))})
                continue

        # closing: let the strong model check first
        kind = _closing_kind(roles, name, args, brief)
        # the last message, sent for a final-cart step, gets the checks an order gets first: the cart must not change
        # after it, since no message is left to repeat the step
        final_msg = (name in ep.message_tools and bool(close_steps) and msg_limit - msgs_sent <= 1
                     and bool(_cart_now(ep)) and not forced)
        gate = "order" if final_msg else kind
        objective = str((brief or {}).get("objective") or "").lower()
        early = ep.left > 12 if ("cheap" in objective or "low" in objective) else (ep.step < 0.55 * ep.max_steps and ep.left > 10)
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
            messages.append({"role": "user", "content": (
                "From me, your supervisor: the senior check found listings that meet everything: " +
                ", ".join(f"{i} x{n:g}" for i, n in pick_lines.items()) + ". Set the cart to exactly these lines, "
                "do any step the shopper requires before the order, then close with the purchase.\n" +
                _state_line(ep, brief, msgs_sent))})
            continue
        if kind == "decline" and _cart_now(ep) and not keep_told and not forced and ep.left > 1:
            keep_told = True
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
            messages.append({"role": "user", "content": (
                "From me, your supervisor: the cart holds listings you chose. Closing with no purchase is right only "
                "if no line can meet the request; a step that can no longer be done is no reason to leave the shopper "
                "with nothing. Close with the purchase unless a line clearly fails a requirement (then fix or remove "
                "that line first).\n" + _state_line(ep, brief, msgs_sent))})
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
        if gate == "order" and cheap and not swept and not forced and ep.left >= END_RESERVE + 10:
            swept = True
            note = _cheaper_sweep(ep, brief, roles, view_cart)
            if note:
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
                messages.append({"role": "user", "content": "From me, your supervisor: " + note + "\n" +
                                 _state_line(ep, brief, msgs_sent)})
                continue
        if gate == "order" and cheap and cheaper_checks < 2 and not forced and ep.left > END_RESERVE + 5:
            seen = _unchecked_cheaper(ep, brief)
            if seen:
                cheaper_checks += 1
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
                messages.append({"role": "user", "content": (
                    "From me, your supervisor: the shopper wants the lowest price, and these listings you saw in "
                    "results cost less than your choice but were never opened: " +
                    ", ".join(f"{i} ({p:g})" for i, p in seen) + ". Open their pages (several page calls in one "
                    "turn); if one meets every stated requirement and every need the shopper told for the same "
                    "product, put the cheapest such listing in the cart instead of that product's line (same "
                    "quantity, and take the old line out); if none does, close again.\n" +
                    _state_line(ep, brief, msgs_sent))})
                continue
        decline_check = kind == "decline" and ((reviews <= MAX_REVIEWS and _spent() < _hard()) or not decline_seen)
        if (gate and not forced and ep.left > 2 and not (gate == "order" and checked_sig == _cart_sig(ep))
                and (decline_check or (reviews < MAX_REVIEWS and _spent() < _soft()))):
            if gate != "decline" and not refreshed:
                refreshed = bool(_refresh_cart(ep, roles, view_cart))
            reviews += 1
            decline_seen = decline_seen or kind == "decline"
            verdict = _review(ep, request, brief, gate if kind else
                              "order, right after sending the shopper this last message (its text: " +
                              json.dumps(str(next(iter(args.values()), "")), ensure_ascii=False)[:600] + ")")
            if verdict and kind == "decline" and str(verdict.get("close_as", "")).lower() == "decline" \
                    and not str(verdict.get("blocker") or "").strip():
                verdict["close_as"] = "order"   # no requirement blocks every listing: the best one is ordered
                if not verdict.get("reason"):
                    verdict["reason"] = "no stated requirement or told need rules out every listing seen"
            if verdict:
                pick_lines = _verdict_lines(ep, brief, verdict)
            if verdict and gate != "decline" and str(verdict.get("close_as", "order")).lower() != "decline":
                extra = _table_fix(verdict, _cart_ids(ep))
                miss = _cart_misses(verdict, _cart_ids(ep))
                if miss and ep.left >= 6 and not (close_steps and msgs_sent >= msg_limit
                                                  and not _step_due(ep, cart_tools, ep.message_tools)):
                    extra = (extra + "; " if extra else "") + miss
                if extra:
                    verdict["verdict"] = "fix"
                    verdict["fix"] = (str(verdict.get("fix") or "") + " " + extra +
                                      " (keep the requested quantities)").strip()
            if verdict and str(verdict.get("verdict", "")).lower() == "fix" and verdict.get("fix"):
                reopen = wrap_told and reopened < 2 and ep.left > END_RESERVE + 6
                if reopen:
                    reopened += 1
                    fix_turns = FIX_TURNS
                note = ("Not closed yet. A senior check of your work decided this (carry it out; it is not a "
                        "suggestion):\n" + str(verdict.get("fix")) +
                        (f"\nFor this the search and page tools are open again for {FIX_TURNS} turns; open several "
                         "pages in one turn where you can, then set the cart and close." if reopen else "") +
                        ("\nReason: " + str(verdict.get("reason")) if verdict.get("reason") else "") +
                        "\nLatest cart: " + json.dumps(ep.cart_obs, ensure_ascii=False) +
                        "\n" + _state_line(ep, brief, msgs_sent))
                messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                 "content": "Not executed: held for a check."})
                messages.append({"role": "user", "content": "From me, your supervisor. " + note})
                continue
            other = str((verdict or {}).get("close_as", "")).lower()
            if verdict and gate == "order" and other != "decline":
                checked_sig = _cart_sig(ep)
            if verdict and other in ("order", "decline") and gate in ("order", "decline") and other != gate:
                note = (f"Not closed yet. A check of your work says the right close is: {other}.\n"
                        f"Reason: {verdict.get('reason', '')}\n")
                if other == "order":
                    note += (("Set the cart to exactly these lines: " +
                              ", ".join(f"{i} x{n:g}" for i, n in pick_lines.items()) + ". ") if pick_lines else
                             "Put the listing that meets the most requirements in the cart (in the category asked "
                             "for and within the budget). ") + "Do any step the shopper requires before the order, " \
                            "then close with the purchase.\n"
                note += _state_line(ep, brief, msgs_sent)
                messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                 "content": "Not executed: held for a check."})
                messages.append({"role": "user", "content": "From me, your supervisor. " + note})
                continue

        if final_msg:
            swept, cheaper_checks = True, 2  # the cart passed the order's checks: no cart change after the message
        if (kind == "order" and close_steps and not forced and msgs_sent < msg_limit and step_pushes < 2
                and ep.left > 2 and _step_due(ep, cart_tools, ep.message_tools)):
            step_pushes += 1
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
            messages.append({"role": "user", "content": (
                "From me, your supervisor: the shopper asked for this with the final cart, right before the order: " +
                "; ".join(close_steps) + ". Nothing has gone to the shopper since the cart last changed. " +
                ("View the cart first, then send" if view_cart and ep.cart_dirty and ep.left > 3 else "Send") +
                " one message that does it (any figure as the latest cart view shows it), then close with the "
                "purchase.\n" + _state_line(ep, brief, msgs_sent))})
            continue
        if kind == "order":
            checked_sig = _cart_sig(ep)
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
                    got = _count_in(obs, lid)  # the shop's own count for the line (a removal reports 0)
                    ep.cart_set[lid] = float(n) if got is None else got
        if name == view_cart:
            ep.cart_obs = obs
            ep.cart_dirty = False
        text = _compact_obs(obs, name in ep.page_tools) + trim + _refusal_note(ep, name, args, obs)
        fresh = list(ep.fresh)
        if name in lookups:
            ep.memo[memo_key] = text
        streak = streak + 1 if name in roles.get("search", []) else 0
        if name in roles.get("search", []):
            pages[qkey] = pages.get(qkey, 0) + 1
            note = _order_note(obs)
            if note:
                text = note + "\n" + text
            text += _empty_filter_note(ep, name, args, obs)
        if ep.left <= END_RESERVE and not ep.finished:
            text += ("\nONLY A FEW STEPS LEFT: put your choice in the cart if it is not there, "
                     "then close the episode now.")
        if dropped:
            text += "\n(Only this call ran: one tool per turn, except several page openings at once.)"
        messages.append({"role": "tool", "tool_call_id": call.get("id"),
                         "content": text + "\n" + _state_line(ep, brief, msgs_sent)})
        if kind and not ep.finished:
            pick = None
            if redecides < 2 and _spent() < _hard() and ep.left > 3:
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
            _note_updates(ep, brief, fresh)

    return


UPDATE_SYSTEM = """A shopper's request in an online shop simulation was summed up in a brief; now a sentence came in
during the task. Say what it changes. Answer with the answer tool, or as JSON only:
{"said_by": "shopper" if the shopper says it, "shop" if it is a shop notice (a price, stock or status line),
 "new_budget": the new total spending limit as a number if it sets one, else null,
 "dropped": requirements the shopper no longer holds to, as "property: value" strings ([] if none),
 "added": new or changed wishes as "property: value" strings ([] if none),
 "gone": true if it says the pick in the cart (or another named listing) can no longer be had, else false,
 "summary": every change it brings, in a few words}
Several sentences may come at once: cover all of them."""


def _note_updates(ep: Episode, brief: dict | None, fresh: list[str]) -> None:
    """A change of mind (or a shop notice) that came during the task, put on the brief: the code's budget checks and
    the strong model's reviews read the brief, so they would otherwise hold the shopper to what no longer stands."""
    if not brief or not fresh:
        return
    user = "BRIEF:\n" + _brief_text(ep, brief) + "\n\nSENTENCE THAT CAME IN:\n" + "\n".join(f'"{n}"' for n in fresh)
    got = _ask_json([LIGHT, HEAVY], UPDATE_SYSTEM, user, max_tokens=800, fields=_answer_fields(UPDATE_SYSTEM)) or {}
    def listed(v):
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                v = [v] if v.strip() and v.strip().lower() not in ("[]", "none", "null") else []
        return [str(x) for x in v if str(x).strip()] if isinstance(v, list) else []
    updates = brief.setdefault("updates", [])
    summary = str(got.get("summary") or "").strip()
    if summary:
        updates.append(summary + (" (shop notice)" if str(got.get("said_by")).lower() == "shop" else ""))
    amount = _number(got.get("new_budget"))
    if amount and str(got.get("said_by")).lower() != "shop":
        budget = brief.get("budget") if isinstance(brief.get("budget"), dict) else {}
        brief["budget"] = {**budget, "amount": amount, "applies_to": budget.get("applies_to") or "total"}
    dropped = listed(got.get("dropped"))
    for it in brief.get("items") or []:
        if isinstance(it, dict) and isinstance(it.get("musts"), list) and dropped:
            it["musts"] = [m for m in it["musts"] if not any(_words(d) and _words(d) <= _words(m) for d in dropped)]
    added = listed(got.get("added"))
    if added:
        brief.setdefault("told_needs", []).extend({"item": None, "need": a, "reference": None} for a in added)
    if str(got.get("gone")).lower() == "true":
        # "that pick is unavailable": the listing last put in the cart is marked sold out, so the checks see it
        cart = _cart_now(ep)
        lid = next((i for _, _, a, _ in reversed(ep.timeline) for i in cart if i in a), None)
        if lid:
            d = ep.listings.setdefault(lid, {})
            keys = [k for k, v in d.items() if isinstance(v, bool) and isinstance(k, str)]
            for k in keys:
                if re.search(r"sold|out_of|outof", k.lower()):
                    d[k] = True
                elif re.search(r"stock|avail|buy|purchas", k.lower()):
                    d[k] = False
            if not keys:
                d["sold_out"] = True
            updates.append(f"{lid} in the cart can no longer be had: replace it")


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
