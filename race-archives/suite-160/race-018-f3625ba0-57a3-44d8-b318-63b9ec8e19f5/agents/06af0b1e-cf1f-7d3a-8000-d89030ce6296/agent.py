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
HEAVY_CALL = 0.02  # what one strong-model check usually costs: one starts only if it still fits under the line
REVIEW_LINE = 0.8  # an order check starts only below this share of the soft line (a refused close may need the rest)
MAX_WAITS = 3
END_RESERVE = 3
CLOSE_RESERVE = 10  # steps still left at the first close: the shop may answer it with a change to act on
SOFT_SPEND = 0.027
HARD_SPEND = 0.039
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


def _charge(model: str, data: dict, one_off: bool = False) -> None:
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
        if before and prompt >= before and not heavy and not one_off:  # the light model's tool loop is one growing
            cached = max(cached, before)                                 # conversation
        if not one_off:
            seen[model] = prompt  # a one-off call in between (a planner, a reader) is no part of that conversation
        _spend.prompts = seen
        cost = ((prompt - cached) * (2e-6 if heavy else 1e-7) + cached * (1e-7 if heavy else 1e-8) +
                (usage.get("completion_tokens") or 0) * (1e-5 if heavy else 5e-7))
    _spend.usd = _spent() + float(cost)
    if model in (LIGHT, LIGHT_SPARE):
        out = usage.get("completion_tokens")
        _spend.light_out = _written() + (out if isinstance(out, int) and not isinstance(out, bool) else 0)


# ---------------------------------------------------------------- inference

def _cacheable(messages: list[dict], newest: bool = True) -> list[dict]:
    """Mark the system prompt and, in a conversation that goes on, the newest message as cache breakpoints (a one-off
    call's own text is never read again, and writing it to the cache costs a quarter more than plain input)."""
    out = copy.deepcopy(messages)
    marks = [0] if out and out[0]["role"] == "system" else []
    if newest and len(out) > 1:
        marks.append(len(out) - 1)
    for i in marks:
        m = out[i]
        c = m.get("content")
        if isinstance(c, str) and c:
            m["content"] = [{"type": "text", "text": c, "cache_control": {"type": "ephemeral"}}]
    return out


def _chat(models: list[str], messages: list[dict], *, tools=None, tool_choice=None,
          max_tokens: int = 1500, tries: int = 3, one_off: bool = False) -> dict | None:
    for model in models:
        body: dict[str, Any] = {
            "model": model,
            "messages": _cacheable(messages, not one_off) if model.startswith("anthropic/") else messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        # every model thinks a little: a light model that answers at once opens far more pages before it settles, and
        # each page lengthens every later turn
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
                _charge(model, data, one_off)
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
            reply = _chat(models[n:], msgs, tools=tools, max_tokens=max_tokens, one_off=True)
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
        if isinstance(spec.get("maxLength"), int) and len(s) > spec["maxLength"]:
            cut = s[: spec["maxLength"]]
            # keywords cut mid-word match nothing: end at the last whole word when one fits
            s = cut.rsplit(" ", 1)[0] if " " in cut and s[spec["maxLength"]] != " " else cut
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
    # a required choice the call left out (a search's mode, say): the keyword option when keywords are sent, else
    # another option (a shelf, price or maker filter alone is a filter search; the keyword mode may ignore filters)
    query = _query_arg({"parameters": schema})
    texts = bool(query and isinstance(out.get(query), str) and out[query].strip())
    for k in required:
        spec = props.get(k) if isinstance(props.get(k), dict) else {}
        if k not in out and isinstance(spec.get("enum"), list) and spec["enum"]:
            word = next((o for o in spec["enum"] if re.search(r"text|keyword|word|query|search", str(o), re.I)), None)
            other = next((o for o in spec["enum"] if o != word), None)
            out[k] = word if texts and word is not None else (other if word is not None and other is not None
                                                                else spec["enum"][0])
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
    if len(cands) == 1:
        return cands[0]
    # else a code-like field named for a reference or code (ref, code, sku, key, number), when only one is
    refs = [v for k, v in d.items() if isinstance(k, str) and re.search(r"(^|_)(ref|code|sku|key|number)s?$", k.lower())
            and _code_like(v)]
    return refs[0] if len(refs) == 1 and not cands else None


_ROW_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,40}")
_PAIR_KEY = re.compile(r"[\w ./()-]{1,40}")


def _row_value(text: str) -> Any:
    """One value of a record written as text: true / false / none / a number as such, "a=b; c=d" as a map."""
    v = text.strip()
    if v in ("True", "False", "true", "false"):
        return v.lower() == "true"
    if v in ("None", "null"):
        return None
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    pairs = [x.partition("=") for x in v.split("; ")]
    if pairs and all(sep and _PAIR_KEY.fullmatch(k.strip()) for k, sep, _ in pairs):
        return {k.strip(): val.strip() for k, _, val in pairs}
    return v


def _row_fields(text: str) -> dict | None:
    """A record the shop wrote as one text ("key: value | key: value | ..."), as its fields; None for other text."""
    if not isinstance(text, str) or text.count(" | ") < 2:
        return None
    out: dict[str, str] = {}
    last = None
    for seg in text.split(" | "):
        key, sep, val = seg.partition(": ")
        key = key.strip()
        if sep and _ROW_KEY.fullmatch(key) and key not in out:
            last = key
            out[key] = val
        elif last is not None:
            out[last] += " | " + seg  # a value that itself holds the separator
        else:
            return None
    if len(out) < 3:
        return None
    return {k: _row_value(v) for k, v in out.items()}


def _decode_rows(obj: Any) -> Any:
    """Every record in a tool result that the shop wrote as one text, read into fields (the rest unchanged)."""
    if isinstance(obj, str):
        got = _row_fields(obj)
        return obj if got is None else got
    if isinstance(obj, list):
        return [_decode_rows(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _decode_rows(v) for k, v in obj.items()}
    return obj


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
  paraphrase; the product category itself is not a must, nor is a value that only describes a listing a reference
  points to), "reference": a requirement whose value must be read
  from another listing's page (who sells it, which kind of product, which details identify it: the values written
  with that listing, after its seller up to where the request says the value may also be reached by way of a
  further listing, are its details and never musts of the purchase, even when they read like requirements; which property to copy,
  and how it must relate: same, different, within N, at least, the value that page pairs it with when it must work
  with or suit what the page shows, or that pointed-to listing itself as the purchase), or null}.
- "budget": {"amount": number or null, "unit": the unit the request uses, "applies_to": "total" or "each"}.
- "objective": how to choose among listings that meet everything (e.g. "cheapest total", "lowest unit price",
  "best match", or "any that fits"; when the request ranks by a property, that property and which way, as
  "least <property>" or "most <property>").
- "lowest_price": true when the shopper wants to pay as little as possible, however the request puts it, else false.
- "skip_sold_out": true if listings out of stock must not be bought (assume true unless the shopper says otherwise).
- "message_limit": the most messages the shopper will accept from the assistant, as an integer, or null if unstated.
- "question_cost": text of any stated price, fee or penalty charged per question or message, else null (a limit on
  how many messages may be sent, and what happens to messages beyond it, is "message_limit", never a cost).
- "step_limit": the most tool calls allowed if the request states one, else null.
- "procedures": every step the shopper says must happen besides finding and buying (for example: show them something
  or get their agreement before ordering, tell them something, read or check something first), each as {"step": what
  must happen, "when": the moment it applies (for example "just before ordering, with the final basket", "before
  anything goes in the cart", "at the start"), "needs_message": true if it means sending the shopper a message, "at_close": true if it must hold for the
  final basket or the order itself (so it is done once the cart is final, right before closing)}; steps due at the
  same moment get the same "when" wording; [] if none. A step asked only for a case that may not arise (such as when
  no listing meets everything) is no procedure: it belongs in "if_nothing_fits".
- "if_nothing_fits": what the request says to do when no listing meets every requirement, as {"says": its words,
  "buy_closest": true if it asks for a listing to be bought anyway, "tell": true if the shopper is
  to be told so, "rule": how that listing is
  chosen (what counts as falling short, any price margin)}; null when it is silent on that.
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
   says nothing). Values match by their words (spelling variants and unit conversions count). A model code in a fit statement may
   stand for a series, with x for any digit and named exceptions ("AB12xx, except AB1250" covers AB1210 but not AB1250):
   check the shopper's own model against it. When one requested item must fit another, a part fits a device when the
   part's own text names the device's model code, or when the device's text names the series the part's code starts
   with (a device that takes series CD4 fits part CD4-1234): pick the two together. Only when the page has no such attribute may
   the title or description decide; a requirement the page does not state at all is unconfirmed, and the order is
   judged only by what each line's page states, so keep looking for a listing whose page states it and settle for an
   unconfirmed listing if the search finds none. The listing must also be in the product category the shopper names: the
   listing's own category field (whatever the shop calls it) decides what kind of product it is, never its title
   or your own sense of it; the words of the product kind or category name are no requirement of their own (a
   listing in a category named for two things need not be both).
   Values must be the same value (same word, a spelling variant or a unit conversion; a word that means nearly the
   same is another value: wanted "matte" is not met by "satin"); a value that
   is merely compatible, broader, stronger, weaker or related does not count (a value to be paired with what
   another listing shows is rule 2; wanted "rustic" is not met by
   "vintage"; wanted "lightly salted" is not met by "unsalted"). Two different words the shop's pages both use as
   values of the same property are different values, never synonyms, except that a value holding the wanted value as
   whole words with other words around it meets it: the shop's wording decides, not what the thing really is
   (wanted "steel" is met by "stainless steel" or "steel-look", wanted "oak" by "white oak finish"). Assume listing
   prices are in the same unit as the shopper's budget unless the shop says otherwise. An exclusion ("not X") removes
   only listings whose value is X itself or a true synonym, not values that merely contain the word. A property the
   request only hints at (no value given) or one the shopper left open is no requirement and rules nothing out.
2. If a requirement or a reply refers to another listing ("the same X as the one sold by Y..."), find that listing: a
   keyword search with the seller's name and two or three of the values that describe it (no category words), the
   next result pages too when the first lacks it (pages are often short), and all rows from that seller opened
   in one page call so their notes show. If its page points at a further listing, find that one the same way and
   follow on until a page gives the value itself. Open each listing on that path with a page tool that shows the
   page's notes and description (a property list alone does not show where a page points). A filter that returned
   nothing for a seller or a value is no way to find it: use keywords.
   How the wanted value relates to the value found: the same value when it must match or equal it; any other
   value when it must differ; a bound when it must be at least or at most; and when it must work with, suit, go
   with, or be a counterpart to, what the page shows, the value that page says its own goes or pairs with (the
   same value only when the page names none). When the request wants the pointed-to listing itself, buy that one.
   A listing read just for a reference is not bought unless it meets every requirement itself.
   Then look for that value in several ways before deciding no listing has it: the product noun with the value as
   keywords, the category filter, other words or spellings for the value, and the property's values on the pages
   you open (the same value may be written differently).
3. Some shops split a listing over several page tools (for example one for its properties, one for price, stock and
   notes): before you choose a listing, look at it with each of them.
   Search with short queries of a few words (two to four), the wanted value first and then the product noun (a
   keyword search may read only the first words of a query). Notes marked STORE NOTE and the SEARCH PLAN say what the searches showed about how this
   shop reads a query: follow them. When a search matches very many listings,
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
4. Never pick a listing the shop cannot supply now when the shopper rejects those. Respect the budget across all lines
   (price x quantity summed) and choose by the shopper's stated objective: when it ranks by a property (the least
   or most of some measure), read that property on each fitting listing's page and take the best by it; the price
   decides only when the shopper wants to pay least.
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
same property (same name or an obvious alias) must carry the wanted value (spelling variants and unit conversions count); a
different, related property does not count; title or description decide only when the page lacks that attribute.
Each line belongs in the product category the shopper names: the listing's own category field (whatever the shop calls it) decides what
kind of product it is, never its title or your own sense of what it is; the words of the product kind or category name
are no requirement of their own (a listing in a category named for two things need not be both). A value must be the same value (same word,
spelling variant or unit conversion; a word that means nearly the same is another value); merely compatible, broader or related values do not count
(wanted "rustic" is not met by "vintage"; wanted "lightly salted" is not met by "unsalted": a stronger, weaker or
neighbouring value is a different value; two different words the shop's pages both use as values of the same
property are different values, never synonyms). A page value that holds the wanted value as whole words with other words
around it does meet it, even when the shop's pages also use the plain value: the shop's wording decides, not what the
thing really is (wanted "green" is met by "dark green" or "green leather", wanted "steel" by "stainless steel" or
"steel-look"). An exclusion ("not X") removes only listings whose value is X itself or a
true synonym; a different value that merely contains the word (such as "faux X" or "X-look") is not excluded.
When the listing's page has no attribute for a required property, its title and description decide; if they say
nothing about it either, the requirement is unconfirmed, and the order is judged only by what each line's own page
states: while steps allow, an unconfirmed line is a reason to fix (name listings seen whose pages state the wanted
value, or the search for one). It is never a reason to decline, and when no listing seen states it after a reasonable
search, the best unconfirmed one stays.
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
a cheaper listing that meets everything and was seen in results is a reason to fix (say which id). When the brief's
"if_nothing_fits" asks for a listing to be bought anyway, a decline is wrong: the fix carts, for each product no
listing fully meets, the listing seen that misses the fewest requirements by that rule. Otherwise a decline is
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

PAGING_KEY = r"page|offset|start|begin|skip|token|cursor|next|after|onward|marker|from"


class StoreSense:
    """What a shop's keyword search shows about itself, learned from the searches made: how many leading words of a
    query it reads, whether a listing must carry every word read or any of them, and whether a mode reads the words at
    all. First-page searches of one tool that share every other argument are compared: two queries agreeing on their
    first k words and differing after them that give the same rows show the later words are not read; a pair that
    differs at or before word k and gives other rows shows that word is read. The read depth is claimed when those
    bounds meet, or when two pairs (or one deliberate test) show it. What is learned holds for the tool in that mode
    (its choice-list arguments), whatever its other filters."""

    def __init__(self) -> None:
        self.seen: dict[tuple, list[tuple[tuple, tuple]]] = {}
        self.low: dict[tuple, int] = {}              # per mode: a word past this many leading ones is read
        self.same: dict[tuple, dict[int, int]] = {}  # per mode: leading words k -> pairs showing nothing after is read
        self.lead: dict[tuple, int] = {}
        self.deaf: dict[tuple, int] = {}             # per mode: pairs with no word in common that gave the same rows
        self.rule: dict[tuple, str] = {}             # "all": every word read must match; "any": any one may
        self.pair: dict[tuple, tuple[str, str]] = {}  # the two queries that showed the read depth
        self.told: set[str] = set()
        self.unfiltered: set[tuple] = set()          # (tool, filter, mode) whose rows showed the filter not applied

    @staticmethod
    def words(text: str) -> tuple[str, ...]:
        return tuple(w for w in re.split(r"[\s(),./:;|]+", str(text).lower()) if w)

    @staticmethod
    def mode(tool: str, arg: str, args: dict, spec: dict | None = None) -> tuple:
        props = (spec or {}).get("properties") or {}
        # a mode is a short choice list (a long one, such as the shop's shelves, is a filter)
        modes = {k: v for k, v in args.items() if k != arg and isinstance(props.get(k), dict)
                 and 1 < len(props[k].get("enum") or []) <= 4}
        return (tool, arg, json.dumps(modes, sort_keys=True))

    def record(self, tool: str, arg: str, args: dict, obs: Any, first: bool, spec: dict | None = None) -> None:
        query = args.get(arg)
        if not first or not isinstance(query, str) or not query.strip() or _failed(obs):
            return
        found: list = []
        _listings_in(obs, found, set())
        rows = tuple(i for i, _ in found)
        total = _result_count(obs)
        sig = (total if total is not None else len(rows), rows)
        ctx = (tool, arg, json.dumps({k: v for k, v in args.items() if k != arg}, sort_keys=True))
        md = self.mode(tool, arg, args, spec)
        mine = self.words(query)
        for theirs, other in self.seen.get(ctx, []):
            if theirs == mine:
                continue
            k = 0
            while k < min(len(mine), len(theirs)) and mine[k] == theirs[k]:
                k += 1
            if sig != other:
                self.low[md] = max(self.low.get(md, 0), k)
                self.deaf.pop(md, None)
                short, long_ = (mine, theirs) if len(mine) < len(theirs) else (theirs, mine)
                if k == len(short) and sig[0] != other[0] and (not self.lead.get(md) or len(long_) <= self.lead[md]):
                    self.rule[md] = "any" if (sig[0] > other[0]) == (mine is long_) else "all"
            elif len(set(rows)) >= 3 and max(len(mine), len(theirs)) > k:
                if k == 0:
                    if not self.low.get(md):
                        self.deaf[md] = self.deaf.get(md, 0) + 1
                else:
                    self.same.setdefault(md, {})[k] = self.same.setdefault(md, {}).get(k, 0) + 1
                    self.pair.setdefault((md, k), (" ".join(theirs), " ".join(mine)))
        self.seen.setdefault(ctx, []).append((mine, sig))
        self._claim(md)

    def _claim(self, md: tuple) -> None:
        low = self.low.get(md, 0)
        claims = sorted(k for k, n in self.same.get(md, {}).items() if k > low and (k == low + 1 or n >= 2))
        self.lead[md] = claims[0] if claims and md not in self.deaf else 0

    def trust(self, md: tuple, k: int) -> None:
        """A deliberate test (a query against its own extension or cut) showed no word after the k-th is read."""
        same = self.same.get(md, {})
        if k in same and k > self.low.get(md, 0):
            same[k] = max(same[k], 2)
            self._claim(md)

    def depth(self, md: tuple) -> int:
        """The leading words a search mode is known to read (0 while unknown)."""
        return self.lead.get(md, 0)

    def news(self, md: tuple) -> str:
        """Findings about this search mode not yet told to the model, as notes for it."""
        out = []
        k = self.lead.get(md, 0)
        if k and f"lead{md}" not in self.told:
            self.told.add(f"lead{md}")
            a, b = self.pair.get((md, k), ("", ""))
            out.append(f"STORE NOTE: this keyword search reads only the first {k} word{'s' if k > 1 else ''} of a "
                       f"query" + (f" ('{a}' and '{b}' gave the same rows)" if a else "") + ": any later word changes "
                       f"nothing. Put the most telling wanted value first (the value, then the product noun), keep each "
                       f"query to {k} word{'s' if k > 1 else ''}, and search another value with a new query that starts "
                       "with it.")
        if self.deaf.get(md, 0) >= 2 and f"deaf{md}" not in self.told:
            self.told.add(f"deaf{md}")
            out.append("STORE NOTE: in this mode the search does not read its words (queries with nothing in common "
                       "gave the same rows): use the other mode for keywords, and this one for its filters.")
        rule = self.rule.get(md)
        if rule and f"rule{md}" not in self.told:
            self.told.add(f"rule{md}")
            out.append("STORE NOTE: " + (
                "every word this search reads must be on a listing (a longer query gave fewer rows): a wanted value "
                "with the product noun lists just the listings that carry it." if rule == "all" else
                "any one word this search reads may match (a longer query gave more rows): common words match most "
                "of the shelf, so a wanted value's own telling word alone lists the listings carrying it."))
        return "\n".join(out)

    def summary(self) -> str:
        parts = []
        for md in dict.fromkeys(list(self.lead) + list(self.rule) + list(self.deaf)):
            modes = json.loads(md[2])
            where = f"{md[0]}" + (f" ({', '.join(f'{k}={v}' for k, v in modes.items())})" if modes else "")
            k = self.lead.get(md, 0)
            bits = ([f"reads only the first {k} word{'s' if k > 1 else ''} of a query"] if k else []) + \
                   ([{"all": "needs every word it reads", "any": "matches any word it reads"}[self.rule[md]]]
                    if self.rule.get(md) else []) + \
                   (["does not read the words at all"] if self.deaf.get(md, 0) >= 2 else [])
            if bits:
                parts.append(where + " " + " and ".join(bits))
        for tool, k, modes in sorted(self.unfiltered):
            mm = json.loads(modes)
            parts.append(f"{tool}" + (f" ({', '.join(f'{a}={b}' for a, b in mm.items())})" if mm else "") +
                         f" does not apply its {k} filter (rows from other shelves came back)")
        return "; ".join(parts)


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
        self.cart_adds: bool | None = None       # the cart tool adds to a line's count (True) or sets it (False)
        self.pick: dict[str, float] | None = None  # the strong model's latest order (listing -> count), if any
        self.timeline: list[tuple] = []          # (step, tool, args, result) of every call
        self.lookup_tools: set[str] = set()
        self.id_cap: dict[str, int] = {}         # list-taking page tool -> most ids per call (learned when refused)
        self.id_fails: dict[str, int] = {}
        self.id_ok: dict[str, int] = {}          # list-taking page tool -> most ids a call has had accepted
        self.prop_tool: str | None = None        # the page tool whose pages record a listing's properties
        self.note_tools: set[str] = set()        # page tools whose pages carry free text (notes, description)
        self.search_tools: set[str] = set()
        self.store = StoreSense()                # what the searches show about how this shop reads a query

    def call(self, name: str, args: dict) -> Any:
        """One environment step; returns the observation."""
        if name in self.search_tools:
            args = _short_query(self, name, args)[0]  # a search made by code keeps to a few words too
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
        obs = _decode_rows(result.get("observation"))  # records written as one text each are read into fields
        self.dialogue.append({"role": "assistant", "tool_call": {"name": name, "args": args},
                              "environment_result": result})
        self._learn(obs, name, args)
        if name in self.search_tools:
            arg = _query_arg(self.by_name[name])
            first = not any(re.search(PAGING_KEY, k, re.I) and v not in (None, "", 0, 1, "0", "1")
                            for k, v in args.items())
            if arg:
                self.store.record(name, arg, args, obs, first, _schema(self.by_name[name]))
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
        # a message's reply is read elsewhere, but a sentence sent beside it (a change of mind) is a note like any other
        beside = [x for v in obs.values() if isinstance(v, list) for x in v if isinstance(x, str)
                  and len(x.split()) >= 3] if isinstance(obs, dict) else []
        for n in beside if name in self.message_tools else _notes_in(obs, set(self.listings) | asked):
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


def _told_cart(ep: Episode, close_steps: list, msgs_sent: int, msg_limit: int, cart_tools: set) -> bool:
    """No message is left and the last one went out after the cart last changed: a required step told the shopper
    this very cart, and no further message could tell them another."""
    return (bool(close_steps) and msgs_sent >= msg_limit and bool(_cart_now(ep))
            and not _step_due(ep, cart_tools, ep.message_tools))


def _closing_words(ep: Episode, brief: dict | None) -> bool:
    """A tool that closes with no purchase also takes free words (a reason or justification) the shopper reads."""
    names = set(_role_tools(brief, ep).get("decline", []) + _role_tools(brief, ep).get("finish", []))
    return any(isinstance(v, dict) and v.get("type") == "string" and not v.get("enum")
               for n in names if n in ep.by_name for v in (_schema(ep.by_name[n]).get("properties") or {}).values())


def _reserved_messages(brief: dict | None, ep: Episode | None = None) -> int:
    """Messages kept back for the shopper's own steps: one per moment (steps due together share a message)."""
    moments = set()
    fb = (brief or {}).get("if_nothing_fits")
    if (isinstance(fb, dict) and fb.get("tell") and not fb.get("buy_closest")
            and not (ep is not None and _closing_words(ep, brief))):
        moments.add("nothing fits")  # a close with no purchase has to tell the shopper first (unless its own words can)
    procs = (brief or {}).get("procedures")
    for p in procs if isinstance(procs, list) else []:
        if isinstance(p, dict) and p.get("needs_message"):
            moments.add(" ".join(sorted(_words(p.get("when") or p.get("step") or ""))))
    return len(moments)


# ---------------------------------------------------------------- asking

QUESTION_SYSTEM = """You help a shop assistant prepare its next message to a shopper. The shopper holds a few wishes
they left unsaid, each about one property of one requested product. A message names one to three properties of one
product, likeliest first, and messages are few, so the order matters. A wish is told only when a message names that
exact property; one the shopper has no view on gets a "don't mind", and one they never think about gets a puzzled
reply, so every property named spends part of a message. You see the request, the brief, hints from the request, the messages sent with the
shopper's replies, and CANDIDATES: the properties the shop's own pages record for the product to ask about now, the
most widely recorded first. Answer with the answer tool, or as JSON only:
{"item": the product, named the way the request names it, "first": up to 2 candidates that a remark in the request
 points at without giving their value, likeliest first ([] when no remark clearly does), "skip": candidates that are
 no property of the product itself (codes, model or part numbers, facts about the offer, stock, delivery, warranty or
 the shop), "rank": up to 5 of the first 10 CANDIDATES, in the order a buyer of this kind of product chooses between
 them (its type or style first, then form or fit, size or capacity, material, colour; a detail most buyers never
 think about last or not at all), "extra": when CANDIDATES name fewer than 6 properties of the product itself, up to 4
 more that people choose this kind of product by and CANDIDATES lack, each in a shop's usual short wording, else [],
 "why": a few words}
- A remark points at a property when it is clearly about it: how the thing looks, what it is made of, how it fits or
  works, what it holds or runs on, the person or place it serves. Background with no such link points at
  nothing.
- Write every property exactly as CANDIDATES write it. Never put in "first" a property the request already gives a
  value for or one that was already asked."""


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
    cheap = _cheap(brief)
    probes = brief.get("probes") if isinstance(brief.get("probes"), list) else []
    probes = [pr for pr in probes if isinstance(pr, dict)]
    if len(probes) == 1 and probes[0].get("item") and ep.max_steps >= 30:
        # one product, one first search: a second one through the other search tool (plain keywords against a
        # filter, or the other way round) shows more kinds of listing, so the property table the questions are
        # chosen from is broader (questions are few; the pages behind them matter)
        used = probes[0].get("tool")
        alt = next((t for t in roles.get("search") or [] if t != used and t in ep.by_name
                    and _query_arg(ep.by_name[t])), None)
        if alt:
            it = next((x for x in brief.get("items") or [] if isinstance(x, dict) and x.get("what")), {})
            vals = [str(m).split(":", 1)[-1].strip() for m in it.get("musts") or [] if ":" in str(m)]
            short = next((v for v in vals if len(v.split()) <= 2 and not re.search(r"\d", v)), "")
            probes.append({"item": probes[0]["item"], "tool": alt,
                           "args": {_query_arg(ep.by_name[alt]): " ".join(x for x in (short, str(it.get("what") or probes[0]["item"])) if x)}})
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
        allowed = _schema(ep.by_name[name]).get("properties") or {}
        named = [k for k in named if not isinstance((allowed.get(k) or {}).get("enum"), list)
                 or _number_twin(args[k]) in allowed[k]["enum"]]  # a choice list takes only its own values
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
                    and _query_arg(ep.by_name[t])), None)
        if not pool and alt and pr.get("item") and ep.left >= END_RESERVE + 10:
            # the first search found nothing (often a filter value the shop words otherwise): try plain keywords
            args = _coerce_args({_query_arg(ep.by_name[alt]): str(pr["item"])}, _schema(ep.by_name[alt]))
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
    # a product whose first search opened fewer than three property pages (results from other shelves, or a filter
    # word the shop writes otherwise): plain keywords for it through a search tool that takes them, a few pages more,
    # so the questions are chosen from more than one or two listings of it
    tool = _keyword_tool(ep, roles)
    if tool and ep.prop_tool and not _list_arg(ep.by_name[ep.prop_tool]):
        for pr in probes[:4]:
            item = str(pr.get("item") or "")
            have = [i for i, it in ep.probe_of.items() if it == item and i in ep.viewed]
            if not item or len(have) >= 3 or ep.finished or ep.left < END_RESERVE + 14:
                continue
            name, arg = tool
            args = _coerce_args({arg: item}, _schema(ep.by_name[name]))
            if args == pr.get("args"):
                continue
            obs = ep.call(name, args)
            notes.append(f"{name} {json.dumps(args, ensure_ascii=False)} -> {_compact_obs(obs)}")
            found: list = []
            _listings_in(obs, found, set())
            words = _item_words(brief, item)
            pool = [i for i in dict.fromkeys(i for i, _ in found) if i not in ep.viewed]
            pool = _on_shelf(ep, brief, pr, name, args, obs, pool, words, keep, notes)
            for i in pool[:3]:
                ep.probe_of.setdefault(i, item)
            notes += [f"{ep.prop_tool} {json.dumps(a, ensure_ascii=False)} -> {_compact_obs(o, True)}"
                      for a, o in _open_pages(ep, ep.prop_tool, pool[:3], keep)]
    # each other page tool that takes one listing, once on a listing already opened: a shop may show a listing's facts
    # with one tool and its notes (where a page points to another listing) with another
    seen = next((i for i in ep.viewed if i in ep.listings), None)
    for t in views:
        if seen and t != ep.prop_tool and not _list_arg(ep.by_name[t]) and ep.left > keep and not ep.finished:
            notes += [f"{t} {json.dumps(a, ensure_ascii=False)} -> {_compact_obs(o, True)}"
                      for a, o in _open_pages(ep, t, [seen], keep, once=True)]
    _test_search(ep, brief, notes)
    return notes


def _test_search(ep: Episode, brief: dict | None, notes: list[str]) -> None:
    """One deliberate step that shows how this shop reads a keyword query, so every later query can be written for it:
    a first-page keyword search already made is run again cut to its first two words (when it had three or more) or
    with one more wanted value (when it had two). The same rows show the shop does not read past the second word;
    other rows show it does, and whether a longer query narrows (every word must match) or widens (any may)."""
    if ep.max_steps < 30 or ep.left < END_RESERVE + 20 or ep.finished:
        return
    values = [w for it in (brief or {}).get("items") or [] if isinstance(it, dict) for m in it.get("musts") or []
              for w in StoreSense.words(str(m).split(":", 1)[-1]) if len(w) > 3 and not re.search(r"\d", w)]
    for ctx, runs in ep.store.seen.items():
        tool, arg, other = ctx
        if tool not in ep.by_name:
            continue
        for words, sig in runs:
            if len(set(sig[1])) < 3 or len(words) < 2:
                continue
            if len(words) >= 3:
                query = " ".join(words[:2])
            else:
                extra = next((w for w in values if w not in words), None)
                if not extra:
                    continue
                query = " ".join(words + (extra,))
            args = {**json.loads(other), arg: query}
            md = StoreSense.mode(tool, arg, args, _schema(ep.by_name[tool]))
            before = dict(ep.store.same.get(md, {}))
            obs = ep.call(tool, args)
            notes.append(f"{tool} {json.dumps(args, ensure_ascii=False)} -> {_compact_obs(obs)}")
            if ep.store.same.get(md, {}).get(2, 0) > before.get(2, 0):
                ep.store.trust(md, 2)
            news = ep.store.news(md)
            if news:
                notes.append(news)
            return


SEARCH_PLAN_SYSTEM = """You plan how a shop assistant searches one online shop. You get the products the shopper wants
with every requirement (stated in the request, or a wish the shopper told), the shop's search tools, the searches
already made with what they returned, and STORE NOTES: what those searches showed about how this shop reads a query.
Write the searches most likely to show, for each product, every listing that fits all of it (the cheapest of them
when the shopper wants the lowest price). Answer with the answer tool, or as JSON only:
{"plan": [{"item": the product, "tool": a search tool's name, "args": its exact arguments, "why": a few words}] (up to
 3 per product, best first), "advice": one or two sentences on how to search this shop}
- Follow the STORE NOTES: when the shop reads only a query's first words, put the most telling wanted value first and
  write no more words than it reads; when every word must match, a wanted value with the product noun lists just the
  listings that carry it; when any word may match, a wanted value's own telling word alone does.
- The best lead is a value a fitting listing must carry, a rare one before a common one. A category filter, in a
  mode that applies it, keeps a search on the product's own shelf.
- Keep each query to a few words (at most four).
- Never repeat a search already made. Use only argument names and allowed values from the tool's schema."""


def _search_plan(ep: Episode, brief: dict | None, roles: dict, request: str, made: list[str]) -> str:
    """The light model's search plan for this shop, written once the first searches have shown how it reads a query
    (the loop's model runs it and adapts it to what the results show)."""
    tools = [t for t in roles.get("search") or [] if t in ep.by_name]
    if not brief or not tools:
        return ""
    items = [{"item": it.get("what"), "category": it.get("category"), "musts": it.get("musts")}
             for it in brief.get("items") or [] if isinstance(it, dict)]
    told = [n for n in brief.get("told_needs") or [] if isinstance(n, dict)]
    schemas = {t: _schema(ep.by_name[t]) for t in tools}
    store = ep.store.summary()
    user = (f"SHOPPER REQUEST:\n{request}\n\nPRODUCTS AND STATED REQUIREMENTS: {json.dumps(items, ensure_ascii=False)}\n"
            f"WISHES THE SHOPPER TOLD: {json.dumps(told, ensure_ascii=False)}\nOBJECTIVE: {brief.get('objective')}\n\n"
            f"SEARCH TOOLS: {json.dumps(schemas, ensure_ascii=False)[:3000]}\n\nSEARCHES ALREADY MADE:\n" +
            "\n".join(m[:300] for m in made[-8:]) +
            f"\n\nSTORE NOTES: {store or 'nothing shown yet about how a query is read'}")
    got = _ask_json([LIGHT, HEAVY], SEARCH_PLAN_SYSTEM, user, max_tokens=4000,
                    fields={"plan": "a JSON list of {item, tool, args, why}", "advice": "one or two sentences"})
    plan = (got or {}).get("plan")
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except ValueError:
            plan = None
    rows = []
    for p in plan if isinstance(plan, list) else []:
        if isinstance(p, dict) and p.get("tool") in ep.by_name and isinstance(p.get("args"), dict):
            args = _coerce_args(p["args"], _schema(ep.by_name[p["tool"]]))
            rows.append(f"- {p.get('item')}: {p['tool']} {json.dumps(args, ensure_ascii=False)} ({p.get('why') or ''})")
    advice = str((got or {}).get("advice") or "").strip()
    if not rows and not advice:
        return ""
    return ("SEARCH PLAN (written for this shop from the first searches; run these first, then adapt to what the "
            "results show):\n" + "\n".join(rows[:9]) + (f"\nHow to search here: {advice}" if advice else "") +
            (f"\nSTORE NOTES: {store}." if store else ""))


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


def _read_key(ep: Episode, name: str, args: dict) -> str:
    """A search's memo key as the shop reads it: its query cut to the leading words the search is known to read
    ("" while that is unknown or the call has no query)."""
    arg = _query_arg(ep.by_name[name]) if name in ep.by_name else None
    k = ep.store.depth(StoreSense.mode(name, arg, args, _schema(ep.by_name[name]))) if arg else 0
    if not k or not isinstance(args.get(arg), str):
        return ""
    return name + json.dumps({**args, arg: " ".join(StoreSense.words(args[arg])[:k])}, sort_keys=True) + "|read"


def _store_note(ep: Episode, name: str, args: dict) -> str:
    """What the searches so far showed about how this shop reads a query (told once), and how this query was read."""
    arg = _query_arg(ep.by_name[name]) if name in ep.by_name else None
    if not arg:
        return ""
    md = StoreSense.mode(name, arg, args, _schema(ep.by_name[name]))
    out, k = ep.store.news(md), ep.store.depth(md)
    words = StoreSense.words(args.get(arg) or "") if arg else ()
    if k and len(words) > k:
        out += ("\n" if out else "") + f"(The shop read this query as '{' '.join(words[:k])}'; the rest was not read.)"
    return "\n" + out if out else ""


def _ignored_filter_note(ep: Episode, name: str, args: dict, obs: Any) -> str:
    """A search given a highest price or a filter value (a shelf, a maker) whose rows show it was not applied: some
    shops apply filters in one search mode only, and rows taken as filtered lead to an over-budget or off-shelf pick."""
    found: list = []
    _listings_in(obs, found, set())
    if not found or name not in ep.by_name:
        return ""
    props = _schema(ep.by_name[name]).get("properties") or {}
    query = _query_arg(ep.by_name[name])
    out = []
    for k, v in (args or {}).items():
        spec = props.get(k) if isinstance(props.get(k), dict) else {}
        if (spec.get("type") in ("number", "integer") and isinstance(v, (int, float)) and not isinstance(v, bool)
                and re.search(r"max|cap|ceil|most|upto|limit|price|cost", k, re.I)
                and not re.search(r"min|floor|least|page|start|offset|begin|from", k, re.I)):
            over = [i for i, r in found if (_listing_price(r) or 0) > v]
            if over:
                out.append(f"{len(over)} of the {len(found)} rows cost more than {k}={v:g}")
        elif (spec.get("type") == "string" and not spec.get("enum") and k != query and isinstance(v, str) and v.strip()
              and not re.search(r"next|token|cursor|after|onward|marker|page", k, re.I)):
            if not any(v.strip().lower() in json.dumps(r, ensure_ascii=False).lower() for _, r in found):
                out.append(f"no row shows {k}={v!r}")
        elif len(spec.get("enum") or []) > 4 and isinstance(v, str) and v.strip() and len(found) >= 3:
            # a shelf chosen from the shop's own list: rows from other shelves show this mode does not apply it
            off = [i for i, r in found if v.strip().lower() not in json.dumps(r, ensure_ascii=False).lower()]
            if len(off) * 2 >= len(found):
                out.append(f"{len(off)} of the {len(found)} rows are not on {k}={v!r}")
                ep.store.unfiltered.add((name, k, StoreSense.mode(name, query or "", args, _schema(ep.by_name[name]))[2]))
    if not out:
        return ""
    modes = [f"{k}={args[k]!r}" for k in args if (props.get(k) or {}).get("enum") and k != query
             and len((props.get(k) or {}).get("enum") or []) <= 4]
    return ("\nThese rows were not filtered: " + "; ".join(out) + "." +
            (f" The search ran with {', '.join(modes)}; this shop may apply filters only with another option there."
             if modes else ""))


def _failed(obs: Any) -> bool:
    """A call the shop turned down: a reply that holds no listing and only says why (an error or issue field, a bare
    status code, a line such as "couldn't locate that listing")."""
    if not isinstance(obs, dict) or not obs:
        return False
    found: list = []
    _listings_in(obs, found, set())
    if found:
        return False
    return (any(re.search(r"err|issue|fail|refus|invalid|denied|problem", str(k), re.I) for k in obs)
            or all(isinstance(v, str) for v in obs.values()))


def _first_batch(ep: Episode, name: str) -> int:
    """How many ids a list-taking page tool gets per call: the size the shop taught (a refusal), else five or half the
    declared most if fewer; shops often declare more than they answer (twice as many, so far)."""
    if name in ep.id_cap:
        return ep.id_cap[name]
    arr = _list_arg(ep.by_name[name]) if name in ep.by_name else None
    most = ((_schema(ep.by_name[name]).get("properties") or {}).get(arr) or {}).get("maxItems") if arr else None
    return min(FIRST_BATCH, max(1, most // 2)) if isinstance(most, int) and most > 0 else FIRST_BATCH


def _list_arg(tool: dict) -> str | None:
    """The argument that takes several listing ids at once: a list of plain values, never a list of cart lines."""
    props = _schema(tool).get("properties") or {}
    return next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array"
                 and (v.get("items") or {}).get("type") != "object"), None)


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
                # perhaps too many at once: half as many (shops often answer half of what they declare)
                ep.id_cap[name] = max(1, len(batch) // 2)
                continue
            break
        todo = [] if once else todo[len(batch):]
    return out


def _fit_list(ep: Episode, name: str, args: dict) -> tuple[dict, str]:
    """Trim a list of ids to the most this tool is known to take per call; the note says so."""
    arr = _list_arg(ep.by_name[name]) if name in ep.by_name and name in ep.page_tools else None
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
    ep.id_cap[name] = min(ep.id_cap.get(name, 99), max(ep.id_ok.get(name, 1), n // 2))
    return (f"\n(The shop turned down {len(args[arr])} ids in one call: send at most {ep.id_cap[name]} per call to "
            f"this tool.)")


def _paging_arg(ep: Episode, brief: dict | None, name: str) -> str | None:
    """A search tool's paging-token argument: as the brief read it, else a text argument named for paging."""
    props = _schema(ep.by_name[name]).get("properties") or {}
    pg = (brief or {}).get("paging")
    if isinstance(pg, dict) and pg.get(name) in props:
        return pg[name]
    return next((k for k, v in props.items() if isinstance(v, dict) and (
        v.get("type") == "string" and re.search(r"next|token|cursor|after|onward|marker|page", k, re.I)
        or v.get("type") == "integer" and re.search(r"page|start|offset|begin|skip|from", k, re.I))), None)


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


def _prop_key(k: Any) -> str:
    return " ".join(str(k).replace("_", " ").lower().split())


def _candidates(ep: Episode, item: str) -> list[str]:
    """The properties the opened pages of one product record, the most widely recorded first (then the ones whose
    values differ most): a property the shop spells two ways with the same values counts once, and a field holding
    several choices in one value, a long text or a value unique to every page (an identifier) is no candidate."""
    groups: dict[str, list[str]] = {}
    for lid in ep.viewed:
        groups.setdefault(ep.probe_of.get(lid, ""), []).append(lid)
    key = item if item in groups else (_match_item(item, [g for g in groups if g]) or "")
    lids = groups.get(key) or ([lid for g in groups.values() for lid in g] if len(groups) <= 1 else [])
    per: dict[str, dict[str, str]] = {}
    for lid in lids:
        for v in (ep.listings.get(lid) or {}).values():
            if not isinstance(v, dict):
                continue
            for pk, pv in v.items():
                name = _prop_key(pk)
                if name and pv not in (None, "", [], {}) and not isinstance(pv, dict):
                    per.setdefault(name, {}).setdefault(lid, " ".join(str(pv).lower().split())[:80])
    rows = []
    for name, vals in per.items():
        texts = list(vals.values())
        joined = sum(" / " in t for t in texts)
        mean = sum(len(t) for t in texts) / len(texts)
        if (joined >= 2 and joined * 3 >= len(texts)) or mean > 40 or \
                (len(texts) >= 3 and len(set(texts)) == len(texts) and mean > 12):
            continue
        rows.append((name, vals))
    rows.sort(key=lambda r: (-len(r[1]), -len(set(r[1].values())), len(r[0].split()), r[0]))
    kept: list[tuple[str, dict]] = []
    for name, vals in rows:
        w = _words(name)
        # one property spelt two ways (the same words but for a short code: "fa pattern", "pattern"): once. Names
        # with a further real word are kept apart even when their values agree ("power" is not "power consumption")
        twin = any(w and w == _words(other) for other, _ in kept)
        if not twin:
            kept.append((name, vals))
    return [n for n, _ in kept]


def _plan_props(ep: Episode, brief: dict, request: str, stated: dict, focus: str, cands: list[str],
                left: int) -> dict:
    """The model's reading of one product's candidates: the ones a remark in the request points at (asked first) and
    any that is no property of the product (never asked)."""
    history = "\n".join(f"Q: {q}\nA: {a}" for q, a in ep.qa) or "(none yet)"
    hints = [{"item": q.get("item"), "property": q.get("property")} for q in brief.get("questions") or []
             if isinstance(q, dict) and q.get("property")]
    user = (f"SHOPPER REQUEST:\n{request}\n\nBRIEF:\n{_brief_text(ep, brief)}\n\nHINTS FROM THE REQUEST (in the "
            f"shopper's words): {json.dumps(hints, ensure_ascii=False)}\n\nALREADY STATED BY THE SHOPPER, BY "
            f"PRODUCT: {json.dumps({k: sorted(v) for k, v in stated.items()}, ensure_ascii=False)}\n\n"
            f"MESSAGES SENT AND REPLIES:\n{history}\n\nMessages left: {left}.\n\nPRODUCT TO ASK ABOUT NOW: {focus}\n"
            f"CANDIDATES: {json.dumps(cands[:24], ensure_ascii=False)}")
    got = _ask_json([LIGHT, HEAVY], QUESTION_SYSTEM, user, max_tokens=2000,
                    fields={"first": "a JSON list of up to 2 candidates",
                            "skip": "a JSON list of candidates", "rank": "a JSON list of up to 5 candidates",
                            "extra": "a JSON list of up to 4 more properties"})
    out = {}
    for k in ("first", "skip", "rank", "extra"):
        v = (got or {}).get(k)
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                v = [v] if v.strip() else []
        out[k] = [_prop_key(x) for x in v if isinstance(x, str) and x.strip()] if isinstance(v, list) else []
    return out


def _question_text(item: str, props: list[str]) -> str:
    """One plain form for every message: the product as the request names it (a name given as alternatives, "A or B",
    reads as two products: the first names it; a bracketed gloss is no part of it), then the shop's names for the
    properties, likeliest first."""
    plain = " ".join(re.sub(r"\([^)]*\)?", " ", item).split()) or item
    noun = re.split(r"\s+or\s+", plain)[0] or plain or "item"
    named = props[0] if len(props) == 1 else ", ".join(props[:-1]) + " or " + props[-1]
    return f"For the {noun}, what {named} would you like?"


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
A page's note that points to a different listing for this property outweighs the value the page itself lists: then
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


QUERY_KEY = re.compile(r"(^|_)q($|_)|quer|phrase|keyword|search|text|words?$|term", re.I)


def _query_arg(tool: dict) -> str | None:
    """A search tool's free-keyword argument: its only free text field (paging aside), else the one named for a query.
    A choice list (a search mode, say) is never it, nor a filter field such as a shelf or a maker."""
    props = _schema(tool).get("properties") or {}
    texts = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and not v.get("enum")
             and not re.search(r"next|token|cursor|after|onward|marker|page", k, re.I)]
    if len(texts) == 1:
        return texts[0]
    return next((k for k in texts if QUERY_KEY.search(k)), None)


SEARCH_WORDS = 4  # words a search query keeps


def _short_query(ep: Episode, name: str, args: dict) -> tuple[dict, bool]:
    """A search's free-keyword query kept to its first SEARCH_WORDS words (a long query matches little)."""
    tool = ep.by_name.get(name)
    key = _query_arg(tool) if tool else None
    words = str(args.get(key) or "").split() if key else []
    if len(words) <= SEARCH_WORDS:
        return args, False
    return {**args, key: " ".join(words[:SEARCH_WORDS])}, True


def _keyword_tool(ep: Episode, roles: dict) -> tuple[str, str] | None:
    """A search tool that takes free keywords, with that argument: one whose only text field is the query first."""
    fallback = None
    for name in roles.get("search") or []:
        if name not in ep.by_name:
            continue
        arg = _query_arg(ep.by_name[name])
        props = _schema(ep.by_name[name]).get("properties") or {}
        alone = sum(1 for v in props.values() if isinstance(v, dict) and v.get("type") == "string"
                    and not v.get("enum")) == 1
        if arg and alone:
            return name, arg
        if arg and not fallback:
            fallback = (name, arg)
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


LINKS_SYSTEM = """You read the pages of the listings in a shopper's cart in an online shop simulation. A page may
send the reader to a further listing for more about the item, naming it by a part code, a model or a name; a list
of things the item works with or also fits is no such pointer. Answer with the answer tool, or as JSON only:
{"links": a JSON list of {"id": the carted listing's id, "points_to": the other listing's identifier exactly as the
 page writes it}, [] when no page sends the reader on}"""


def _cart_links(ep: Episode, roles: dict, keep: int) -> list[str]:
    """Before the order: each carted line's page opened, and each listing a carted page sends the reader to for more
    about the item found and opened (such a page belongs to the checks of the purchase)."""
    cart = list(_cart_now(ep))
    pages = [t for t in (roles.get("view") or []) + (roles.get("compare") or []) if t in ep.by_name]
    if not cart or not pages or ep.finished:
        return []
    notes = []
    # every carted line opened with each page tool that takes one listing (a shop may give a listing's facts with one
    # and its page, notes included, with another: both belong to the checks of a purchase)
    for t in [t for t in dict.fromkeys(pages) if not _list_arg(ep.by_name[t])] or pages[:1]:
        unread = [i for i in cart if t not in ep.opened_by.get(i, set())]
        for a, o in _open_pages(ep, t, unread, keep):
            notes.append(f"{t} {json.dumps(a, ensure_ascii=False)} -> {_compact_obs(o, pages=True)}")
    views = [t for t in sorted(ep.note_tools) if t in ep.by_name] or pages
    view = next((t for t in views if not _list_arg(ep.by_name[t])), views[0])
    got = _ask_json([LIGHT, HEAVY], LINKS_SYSTEM, "CART PAGES:\n" + _pages_text(ep, cart), max_tokens=600,
                    fields=_answer_fields(LINKS_SYSTEM)) or {}
    links = got.get("links")
    if isinstance(links, str):
        try:
            links = json.loads(links)
        except ValueError:
            links = []
    search = _keyword_tool(ep, roles)
    for ln in (links if isinstance(links, list) else [])[:4]:
        mark = str((ln or {}).get("points_to") or "").strip() if isinstance(ln, dict) else ""
        if not mark or ln.get("id") not in cart or ep.finished or ep.left <= keep + 1:
            continue
        def named(i):
            """2: the listing's own name carries the mark; 1: its page only mentions it (maybe pointing on itself)."""
            d = ep.listings.get(i) or {}
            if i in cart or mark.lower() not in json.dumps(d, ensure_ascii=False).lower():
                return 0
            return 2 if any(re.search(r"name|title|headline|label", str(k), re.I) and mark.lower() in str(v).lower()
                            for k, v in d.items()) else 1
        hit = max(ep.listings, key=named, default=None)
        if (hit is None or named(hit) < 2) and search:
            name, arg = search
            obs = ep.call(name, _coerce_args({arg: mark}, _schema(ep.by_name[name])))
            notes.append(f"{name} {json.dumps({arg: mark}, ensure_ascii=False)} -> {_compact_obs(obs)}")
            hit = max(ep.listings, key=named, default=None)
        if hit is not None and not named(hit):
            hit = None
        if hit and hit not in ep.viewed:
            for a, o in _open_pages(ep, view, [hit], keep):
                notes.append(f"{view} {json.dumps(a, ensure_ascii=False)} -> {_compact_obs(o, pages=True)}")
        if hit:
            notes.append(f"The page of {ln['id']} points to {mark} for more: that is {hit}, now shown.")
    return notes


def _cut_words(text: str, limit: int) -> str:
    """Keywords within a length limit, ended at the last whole word (a word cut in half matches nothing)."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut.rsplit(" ", 1)[0] if " " in cut and text[limit] != " " else cut


def _shelf_choice(props: dict, kind: str) -> dict:
    """A search tool's shelf argument set to the option naming this kind of product (the same name in either number
    first, else the shortest that shares a naming word), or nothing when no option names it."""
    for k, v in props.items():
        opts = [o for o in (v.get("enum") or [])[:200] if isinstance(o, str)] if isinstance(v, dict) else []
        hits = [o for o in opts if kind and _names_item(o, _phrase_words(kind))]
        if hits:
            same = {kind.lower(), _number_twin(kind).lower()}
            return {k: min(hits, key=lambda o: (o.lower() not in same, len(o)))}
    return {}


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
    props = _schema(ep.by_name[name]).get("properties") or {}
    limit = (props.get(arg) or {}).get("maxLength") or 200
    # the shop's own shelf for that kind of product, when the search takes one (else a seller's listings from every
    # shelf crowd the two or three result pages)
    shelf = _shelf_choice(props, kind)
    given: set = set()
    plans = [{arg: _cut_words(q, limit), **shelf} for q in dict.fromkeys(x.strip() for x in queries if x.strip())]
    if shelf and seller:
        plans.append({arg: seller[:limit]})  # the kind may sit on another shelf than the one guessed
    for k, args in enumerate(plans):
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
                # a query that carries the listing's own values comes back in the shop's relevance order: keep it
                yield mine, k == 0 and bool(seller and short)
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
        musts = next((it.get("musts") for it in brief.get("items") or [] if isinstance(it, dict)
                      and it.get("what") == item and it.get("musts")), None)
        if musts:
            need += (f"\nOTHER VALUES THE REQUEST GIVES FOR {item} (some may only describe the listing pointed to): "
                     + json.dumps(musts, ensure_ascii=False))
        got = _ask_json([LIGHT, HEAVY], POINTER_SYSTEM, need + "\n\nPAGES OPENED: none yet", max_tokens=1200,
                        fields=fields) or {}
        prop, target = got.get("property"), _as_dict(got.get("next"))
        path, value, pairs, lost = [], None, None, None
        for _ in range(6):
            if not target or ep.finished or ep.left <= keep:
                break
            matched = False
            for ids, ranked in _seller_batches(ep, brief, search, target, keep):
                ids = [i for i in ids if i not in path]
                if not ids:
                    continue
                # rows of a query without the listing's values: the seller's rows that share the most words with
                # the listing wanted first, a few at a time (a query with them keeps the shop's own order)
                want = _words(" ".join(_detail_values(target.get("details"))) + " " + str(target.get("kind") or ""))
                if not ranked:
                    ids.sort(key=lambda i: -len(want & _words(json.dumps(ep.listings.get(i) or {}, ensure_ascii=False))))
                size = 2 * _first_batch(ep, view)
                for chunk in [ids[k:k + size] for k in range(0, min(len(ids), 8), size)]:
                    if ep.finished or ep.left <= keep:
                        break
                    _open_pages(ep, view, chunk, keep)
                    got = _ask_json([LIGHT, HEAVY], POINTER_SYSTEM,
                                    f"{need}\nPROPERTY: {prop}\nLISTING WANTED: {json.dumps(target, ensure_ascii=False)}"
                                    + (f"\nALREADY READ ON THE WAY (not the listing wanted): {', '.join(path)}"
                                       if path else "")
                                    + f"\n\nPAGES OPENED:\n{_pages_text(ep, chunk)}", max_tokens=1200,
                                    fields=fields) or {}
                    lid, nxt = got.get("listing"), _as_dict(got.get("next"))
                    if isinstance(lid, str) and lid in chunk or nxt and _seller_name(nxt) != _seller_name(target):
                        break
                lid = got.get("listing")
                nxt = _as_dict(got.get("next"))
                if not (isinstance(lid, str) and lid in ep.listings and lid not in path) and nxt and \
                        _seller_name(nxt) and _seller_name(nxt) != _seller_name(target):
                    # a pointer on to another seller was read without naming the page that holds it: the opened
                    # page whose text names that seller (and is not theirs) is the one
                    who = str(_seller_name(nxt)).lower()
                    lid = next((i for i in ids if who in json.dumps(ep.listings.get(i) or {}, ensure_ascii=False).lower()
                                and not _sold_by(ep.listings.get(i) or {}, _seller_name(nxt))), None)
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
    got = _ask_json([LIGHT, HEAVY], REPLIES_SYSTEM, user, max_tokens=2500,
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


QUESTION_WIDTH = 1  # properties named in one message once the shopper has asked for fewer at a time
LIST_STEPS_ONE = (1, 3)   # properties named in the 1st, 2nd (and later) messages when one product is asked about
LIST_STEPS_MANY = (2, 3)  # the same when several are: a short list where lists are welcome costs little where they
#                           are not, and a reply asking for fewer switches every later message to a single property


def _ask(ep: Episode, brief: dict | None, roles: dict, request: str) -> None:
    """Each message names a few properties of one product (LIST_STEPS_*, QUESTION_WIDTH once the shopper asks for
    fewer), likeliest first; a reply settles one property at
    most (a told wish rules out the ones named before it, any other reply the first one named), so the next message
    about that product starts one further on. Products take turns, an untold one first."""
    tools = roles.get("message") or []
    if not tools or not brief:
        return
    tool = ep.by_name[tools[0]]
    arg = _text_arg(tool)
    if not arg:
        return
    budget = _message_limit(brief) - _reserved_messages(brief, ep)
    qs = brief.get("questions") if isinstance(brief.get("questions"), list) else []
    hinted = [q for q in qs if isinstance(q, dict) and q.get("hinted") and q.get("property")]
    if brief.get("question_cost"):
        budget = min(budget, len(hinted))
    budget = min(budget, ep.left - END_RESERVE - 6)
    stated = _stated_by_item(brief)
    items = list(stated) or [""]
    count = {it: 0 for it in items}
    told: set = set()
    done: dict[str, list[str]] = {it: [] for it in items}  # properties settled or passed over, per product
    plans: dict[str, dict] = {}
    spent: set = set()  # products with nothing left to ask
    lead: dict[str, list[str]] = {it: [] for it in items}  # what leads a product's next list (after its last reply)
    tried: dict[str, set] = {it: set() for it in items}    # properties already named for a product
    misses: dict[str, int] = {it: 0 for it in items}       # replies in a row that told nothing for a product
    steps = LIST_STEPS_ONE if len(items) == 1 else LIST_STEPS_MANY
    lists_on, sent = True, 0
    while len(ep.qa) < budget and not ep.finished and len(spent) < len(items):
        width = steps[min(sent, len(steps) - 1)] if lists_on else QUESTION_WIDTH
        focus = min((it for it in items if it not in spent), key=lambda it: (it in told, count[it], items.index(it)))

        def fresh(p: str) -> bool:
            return bool(p) and not NOT_ASKED.search(p) and not _overlaps(p, stated.get(focus, set())) \
                and not any(_words(p) == _words(x) for x in done[focus])
        cands = [c for c in _candidates(ep, focus) if fresh(c)]
        if focus not in plans:
            plans[focus] = _plan_props(ep, brief, request, stated, focus, cands, budget - len(ep.qa))
        plan = plans[focus]
        skip = set(plan.get("skip") or [])
        first = [p for p in plan.get("first") or [] if fresh(p) and p not in skip]
        if not cands:  # no page records this product's properties: the request's own hints, if any
            first += [_prop_key(q["property"]) for q in hinted if fresh(_prop_key(q["property"]))
                      and _match_item(str(q.get("item") or ""), items) in (focus, None)]
        extra = [p for p in plan.get("extra") or [] if fresh(p) and p not in skip][:4]
        recorded = [c for c in cands if c not in skip]
        # the planner's buyer's order over the most recorded candidates leads the rest (the pages' counts break ties)
        ranked_first = [p for p in plan.get("rank") or [] if p in recorded[:10]]
        recorded = list(dict.fromkeys(ranked_first + recorded))
        if len(recorded) >= 3:
            # a hinted property no opened page records is one the shopper may never think about: the pages' own lead
            first = [p for p in first if p in recorded]
        order = list(dict.fromkeys(first + recorded + extra))
        # a reply that told nothing settles only the first property named: one the shopper does not mind may have hidden
        # the ones after it, so those lead the next list in the reverse order; after two such replies in a row, the
        # properties not named yet lead. After a told wish, the ones named after it lead.
        if misses[focus] >= 2:
            order = [p for p in order if p not in tried[focus]] + [p for p in order if p in tried[focus]]
        elif lead[focus]:
            order = [p for p in lead[focus] if p in order] + [p for p in order if p not in lead[focus]]
        props = order[:width]
        if not props:
            spent.add(focus)
            continue
        text = _question_text(focus or next((str(i.get("what")) for i in brief.get("items") or []
                                             if isinstance(i, dict) and i.get("what")), "item"), props)
        obs = ep.call(tools[0], _coerce_args({arg: text}, _schema(tool)))
        reply = _reply_text(obs)
        ep.qa.append((text, reply))
        if isinstance(obs, dict) and obs.get("error"):
            break  # the shop took no more messages
        count[focus] += 1
        sent += 1
        # a wish names its property ("where X is concerned, only Y"); the longest name that fits wins
        hits = [k for k, p in enumerate(props) if _words(p) and _words(p) <= _words(reply)]
        hit = max(hits, key=lambda k: len(_words(props[k])), default=None)
        if hit is None and len(props) > 1 and _crowded(reply):
            # the shopper wants one property per message: nothing was asked, the first one named goes alone next
            lists_on, lead[focus] = False, list(props)
            continue
        tried[focus] |= set(props)
        if hit is not None:
            done[focus] += props[:hit + 1]
            told.add(focus)
            misses[focus], lead[focus] = 0, props[hit + 1:]
        else:
            done[focus].append(props[0])
            misses[focus] += 1
            lead[focus] = list(reversed(props[1:]))
        if CROWDED.search(reply):
            lists_on = False  # this shopper takes one property per message


# details never asked about: a listing's choice of variants and facts about the offer rather than the product
NOT_ASKED = re.compile(r"^(options?|variants?|variations?|warranty( \w+)?|shipping( \w+)?)$", re.I)
# a reply asking for one thing at a time
CROWDED = re.compile(r"\bone (?:thing|question|subject|topic|item|prompt)\b|\bat a time\b|\btoo many\b|\bone by one\b|"
                     r"\bsingle file\b|\bin (?:turn|sequence|a queue)\b|\b(?:just|only) (?:the )?(?:first|top|opening)\b|"
                     r"\bper turn\b|\bseparately\b|\btrickle\b", re.I)
CROWDED_SYSTEM = """You read a shopper's reply to an assistant's message in an online shop simulation. That message
named a few properties together. Answer with the answer tool, or as JSON only:
{"fewer": true if the reply wants smaller messages (a single property each, just the first, in turn) and answers
none of them, else false; a request to say it again or more plainly is not this}"""


def _crowded(reply: str) -> bool:
    """Whether a reply to a message that named several properties asks for one property per message."""
    if CROWDED.search(reply):
        return True
    got = _ask_json([LIGHT], CROWDED_SYSTEM, f"REPLY: {reply}", max_tokens=200,
                    fields={"fewer": "true or false"})
    v = (got or {}).get("fewer")
    return v is True or str(v).strip().lower() == "true"


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
    room = max(0, 8 - len(in_cart))
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
    store = ep.store.summary()
    if store:
        parts.append("HOW THIS SHOP'S SEARCH READS A QUERY (learned from the searches made): " + store)
    parts.append("CURRENT CART (latest view):\n" + json.dumps(ep.cart_obs, ensure_ascii=False))
    if len(ep.page_tools) > 1:
        parts.append("PAGE TOOLS IN THIS SHOP (each may show different parts of a listing): " +
                     ", ".join(sorted(ep.page_tools)))
    parts.append("OPENED LISTINGS (full pages):\n" + "\n".join(full))
    parts.append("OTHER LISTINGS SEEN IN RESULTS:\n" + "\n".join(brief_rows[-10:]))
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
names (the listing's own category field decides the kind of product, not its title, and the words of that kind are no
requirement of their own), in stock when that
matters, within the budget, chosen by the shopper's objective. A value must be the same value (spelling variants and
unit conversions count, a word that means nearly the same does not; compatible, stronger, weaker or related values do not, nor two different words the shop's pages
both use as values of the same property, though a value holding the wanted value as whole words with other words around
it meets it: the shop's wording decides, as "dark green" meets "green"); a requirement the page contradicts is
not met; when the page has no attribute for it, the title or description decide, and if they say nothing it does not
rule the listing out (if a page tool was not yet used on a listing, list it in "open_first").
A property the request only hints at (no value given) or one the shopper left open rules nothing out.
Prices are in the budget's unit unless the shop says otherwise. Decline only if every listing seen clearly fails a
requirement by its own page facts (a different value, another category, over the budget, sold out when that matters);
a requirement that is only unconfirmed (the page does not show it) is no reason to decline when a listing otherwise
fits: order the best such listing then. When the brief's "if_nothing_fits" asks for a listing to be bought anyway,
never decline: for each product no listing fully meets, order the listing seen that misses the fewest requirements
by that rule.
If the shop refused a close (for example a listing went out of stock), choose the best remaining listing by the
shopper's objective; the same product from another seller (same title) is often listed, so put such twins seen in
results first, or name the search to run for them in "search_first" (a few title words). Add "open_first": up to 4 ids of listings seen in results that could beat your choice (for
example cheaper) but whose pages were not opened, so they can be checked before the order."""


def _fallback(brief: dict | None) -> dict | None:
    """The request's rule for when no listing meets everything, when it asks for one to be bought anyway."""
    fb = (brief or {}).get("if_nothing_fits")
    return fb if isinstance(fb, dict) and fb.get("buy_closest") else None


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


def _line_items(ep: Episode, brief: dict | None) -> tuple[dict[str, int | None], list[dict]]:
    """Each cart line -> the requested product it serves (by the line's own category field: the brief's category or
    name for it in either number first, else the one product whose naming words it shares), or None when unclear."""
    items = [it for it in (brief or {}).get("items") or [] if isinstance(it, dict)]

    def same(a: str, b: str) -> bool:
        a, b = a.lower().strip(), b.lower().strip()
        return bool(a) and (a == b or _number_twin(a) == b or a == _number_twin(b))
    out: dict[str, int | None] = {}
    for lid in _cart_now(ep):
        out[lid] = _item_for(ep, brief, items, lid, same)
    return out, items


def _item_for(ep: Episode, brief: dict | None, items: list[dict], lid: str, same) -> int | None:
    """The requested product a listing serves, by its own category field (else its name), or None when unclear."""
    d = ep.listings.get(lid) or {}
    shelf = _shelf_of(d) or next((v for k, v in d.items() if isinstance(v, str)
                                  and re.search(r"name|title|headline", str(k), re.I)), "")
    exact = [k for k, it in enumerate(items)
             if shelf and any(same(shelf, str(it.get(f) or "")) for f in ("category", "what"))]
    loose = [k for k, it in enumerate(items)
             if shelf and _names_item(shelf, _item_words(brief, str(it.get("what") or "")))]
    hits = exact if len(exact) == 1 else loose if len(loose) == 1 else []
    return hits[0] if hits else None


def _unopened_items(ep: Episode, brief: dict | None) -> list[str]:
    """Requested products not one listing of which has had its page opened (by its category field or name; an opened
    page that cannot be placed counts for none of them)."""
    items = [it for it in (brief or {}).get("items") or [] if isinstance(it, dict)]
    if len(items) < 1:
        return []

    def same(a: str, b: str) -> bool:
        a, b = a.lower().strip(), b.lower().strip()
        return bool(a) and (a == b or _number_twin(a) == b or a == _number_twin(b))
    served = {_item_for(ep, brief, items, lid, same) for lid in ep.viewed if lid in ep.listings}
    return [str(it.get("what") or it.get("category") or "") for k, it in enumerate(items) if k not in served]


def _cheap(brief: dict | None) -> bool:
    """The shopper wants the lowest price: the brief model's own judgement of the request (any wording), or its
    objective naming a cheapest or lowest choice."""
    b = brief or {}
    return b.get("lowest_price") is True or str(b.get("lowest_price")).lower() == "true" or \
        bool(re.search(r"\bcheap|\b(low|least|small|minim)\w* (price|total|cost|spend)",
                       str(b.get("objective") or "").lower()))


def _qty_gaps(ep: Episode, brief: dict | None) -> list[str]:
    """Cart lines whose count differs from the count asked for their product, and requested products the cart has no
    line for (lines matched to products by their own category field; unclear ones are left to the review)."""
    if (brief or {}).get("counts_changed"):
        return []  # a change during the task set other counts
    served, items = _line_items(ep, brief)
    cart = _cart_now(ep)
    out = []
    for lid, k in served.items():
        q = items[k].get("qty") if k is not None else None
        if isinstance(q, (int, float)) and not isinstance(q, bool) and float(q) != float(cart.get(lid, 0)):
            out.append(f"{lid} is in the cart {cart.get(lid, 0):g} times; the shopper asked for {q:g} "
                       f"{items[k].get('what')}")
    if served and None not in served.values():
        out += [f"no line for the {it.get('what')} asked for" for k, it in enumerate(items)
                if k not in served.values()]
    return out


_BOUNDS = re.compile(r"\d|\b(least|most|not|no|non|under|over|less|more|than|between|same|like|as|any|except|other|"
                     r"others|without|up|from|match|matching|whatever|page|listing|near|unlike|different)\b")


def _plain_need(need: Any) -> tuple[str, list[set[str]], str] | None:
    """A requirement "property: value" whose value is plain words (no figure, bound, exclusion or pointer to another
    listing): (property, the words of each alternative it allows, the value)."""
    prop, sep, val = str(need or "").partition(":")
    val = " ".join(val.lower().split())
    if not sep or not val or _BOUNDS.search(val):
        return None
    alts = [{w for w in re.findall(r"[a-z]+", a) if not _FILLER.fullmatch(w)}
            for a in re.split(r"\bor\b|/|,|;", val)]
    alts = [a for a in alts if a]
    return (_prop_key(prop), alts, val) if alts and _words(prop) else None


def _word_in(w: str, words: set[str]) -> bool:
    """A word, or another form of it (a shared stem: stripe / striped, knee / knees), among the words."""
    stem = w[:max(4, len(w) - 2)]
    return any(x == w or (len(x) >= 4 and len(w) >= 4 and (x.startswith(stem) or w.startswith(x[:max(4, len(x) - 2)])))
               for x in words)


def _value_conflicts(ep: Episode, brief: dict | None) -> list[str]:
    """Cart lines whose own page states another value for a plain-valued requirement of their product: an attribute
    named for that very property whose value holds none of the wanted words in any allowed alternative (a related
    property, or no attribute for it, decides nothing here)."""
    served, items = _line_items(ep, brief)
    told = [n for n in (brief or {}).get("told_needs") or [] if isinstance(n, dict) and n.get("need")
            and not n.get("reference")]
    names = [str(it.get("what") or "") for it in items]
    out = []

    def page_attrs(d: dict) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for key, v in d.items():
            if isinstance(v, dict):
                for pk, pv in v.items():
                    if pv not in (None, "", [], {}) and not isinstance(pv, dict):
                        found.setdefault(_prop_key(pk), []).append(" ".join(str(pv).lower().split()))
            elif isinstance(v, (int, float, bool)) and not isinstance(v, str):
                found.setdefault(_prop_key(key), []).append(str(v).lower())  # a top-level fact (a price, a flag)
        return found
    # the properties some opened page of this shop records: a requirement none of them records cannot be read off
    # a page here (a fact about the offer, or a wording the shop never uses), so it is no gap on a line
    recorded = {key for d in ep.listings.values() if isinstance(d, dict) for key in page_attrs(d)}
    # what each opened page says, in its own words: a wanted value that some page states can be stated on a page here
    corpora = [(" ".join(str(v) for v in d.values() if isinstance(v, str)) + " " +
                " ".join(x for vals in page_attrs(d).values() for x in vals)).lower()
               for d in ep.listings.values() if isinstance(d, dict) and any(isinstance(v, dict) for v in d.values())]

    def kept(prop_words: set, value_words: set | None = None) -> bool:
        return any(_words(key) and (_words(key) <= prop_words or prop_words <= _words(key)) for key in recorded) or \
            bool(value_words) and any(all(_word_in(w, set(re.findall(r"[a-z]+", c))) for w in value_words)
                                      for c in corpora)
    for lid, k in served.items():
        if k is None:
            continue
        needs = list(items[k].get("musts") or [])
        needs += [n["need"] for n in told if len(items) == 1 or _match_item(str(n.get("item") or ""), names) == names[k]]
        d = ep.listings.get(lid) or {}
        attrs = page_attrs(d)
        text = " ".join(str(v) for v in d.values() if isinstance(v, str)).lower()
        corpus = text + " " + " ".join(x for vals in attrs.values() for x in vals)
        for need in needs:
            plain = _plain_need(need)
            if not plain:
                prop, _, val = str(need or "").partition(":")
                pw = _words(prop)
                if not pw or _POINTER.search(val) or not val.strip():
                    continue
                said = any(_words(key) and (_words(key) <= pw or pw <= _words(key)) for key in attrs)
                vw = {w for w in re.findall(r"[a-z]+", val.lower()) if not _FILLER.fullmatch(w)
                      and not _BOUNDS.fullmatch(w)}
                if not kept(pw, vw):
                    continue
                figures = set(re.findall(r"\d+(?:\.\d+)?", val))
                if not said and not any(w in text for w in pw | vw if len(w) > 2) \
                        and not figures & set(re.findall(r"\d+(?:\.\d+)?", corpus)):
                    out.append(f"{lid}: its page does not state the {_prop_key(prop)} (wanted: {val.strip()[:50]})")
                continue
            prop, alts, val = plain
            same = [vals for key, vals in attrs.items() if _words(key) == _words(prop)]
            if not same:
                if any(kept(_words(prop), alt) for alt in alts) and \
                        not any(_words(key) and _words(key) <= _words(prop) for key in attrs) and \
                        not any(all(_word_in(w, set(re.findall(r"[a-z]+", text))) for w in alt) for alt in alts):
                    out.append(f"{lid}: its page does not state the {prop} (wanted: {val[:50]})")
                continue
            words = {w for vals in same for x in vals for w in re.findall(r"[a-z]+", x)}
            if not any(all(_word_in(w, words) for w in alt) for alt in alts):
                out.append(f"{lid}: the {prop} asked for is \"{val}\", its page says \"{'; '.join(same[0])[:60]}\"")
    return out


_POINTER = re.compile(r"\b(same|like|unlike|match|matching|whatever|page|listing|different|goes|pairs?)\b", re.I)
_FILLER = re.compile(r"a|an|the|and|or|with|of|in|at|to|for")  # words that join a value, not part of it


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


def _whole_cart(ep: Episode, obs: Any) -> bool:
    """A reply that lists the cart's lines (records of listings seen) beside a total: the whole cart, not one line."""
    if not isinstance(obs, dict):
        return False
    total = any(isinstance(v, (int, float)) and not isinstance(v, bool) and MONEY_KEY.search(str(k))
                for k, v in obs.items())
    lines = any(isinstance(v, list) and v and all(isinstance(x, dict) and any(str(y) in ep.listings
                                                                              for y in x.values()) for x in v)
                for v in obs.values())
    return total and lines


def _shows_cart(ep: Episode, obs: Any) -> bool:
    """A reply that shows cart lines (a list, maybe empty, or a listing seen before), not just that a call worked."""
    if isinstance(obs, list):
        return True
    if isinstance(obs, dict):
        return any(str(k) in ep.listings or _shows_cart(ep, v) for k, v in obs.items())
    return isinstance(obs, str) and obs in ep.listings


COUNT_KEY = re.compile(r"qty|quant|count|piece|units?$|number", re.I)
MONEY_KEY = re.compile(r"price|cost|total|amount|sum", re.I)
PRICE_KEY = re.compile(r"price|cost|due|fee|tax|charge|pay|sub|grand|weight|size", re.I)


def _count_in(obs: Any, lid: str) -> float | None:
    """The count a cart reply gives for one listing (its line's numeric count field), or None."""
    found: list[float] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            if lid in [str(v) for v in o.values() if isinstance(v, (str, int))]:
                nums = [(str(k), v) for k, v in o.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
                for k, v in nums:
                    if COUNT_KEY.search(k) and not MONEY_KEY.search(k):
                        found.append(float(v))
                        return
                # no field named for a count: the line's one small whole number outside its price fields
                small = [v for k, v in nums if not PRICE_KEY.search(k) and float(v).is_integer() and 0 <= v < 100]
                if len(small) == 1:
                    found.append(float(small[0]))
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


def _page_token(obs: Any) -> Any:
    """A result's paging token: its only top-level short text value, else a next page's number the shop names."""
    if not isinstance(obs, dict):
        return None
    vals = [v for v in obs.values() if isinstance(v, str) and 0 < len(v) <= 40]
    if len(vals) == 1:
        return vals[0]
    nums = [v for k, v in obs.items() if isinstance(v, int) and not isinstance(v, bool) and v > 1
            and re.search(r"next|follow|cursor|onward|page", str(k), re.I)
            and not re.search(r"count|total|hits|size|per|num_", str(k), re.I)]
    return nums[0] if len(nums) == 1 else None


def _keyword_sweep(ep: Episode, brief: dict | None, roles: dict, view_cart: str | None) -> str | None:
    """The shopper wants the lowest price and the shop has no price filter: a few more searches on the carted line's
    shelf, keyed by the product's wanted values (one value per search, then all of them), each read to its last page,
    so listings cheaper than the carted one turn up; the cheapest likely ones are opened for the check before the
    order."""
    search = _keyword_tool(ep, roles)
    if not search or ep.left < END_RESERVE + 12:
        return None
    if view_cart and (ep.cart_dirty or ep.cart_obs is None):
        ep.cart_obs = ep.call(view_cart, _coerce_args({}, _schema(ep.by_name[view_cart])))
        ep.cart_dirty = False
    lines = _cart_ids(ep)
    if not lines or len(lines) > 4:
        return None
    name, arg = search
    props = _schema(ep.by_name[name]).get("properties") or {}
    page_arg = _paging_arg(ep, brief, name)
    items = [it for it in (brief or {}).get("items") or [] if isinstance(it, dict)]
    told = [str(n.get("need") or "") for n in (brief or {}).get("told_needs") or [] if isinstance(n, dict)]
    cands: dict[str, tuple[int, float]] = {}
    for lid in lines:
        d = ep.listings.get(lid) or {}
        price, shelf = _listing_price(d), _shelf_of(d) or ""
        it = next((x for x in items if shelf and _names_item(shelf, _item_words(brief, str(x.get("what") or "")))),
                  items[0] if len(items) == 1 else None)
        if price is None or not it:
            continue
        # only the values of properties the listing's own property table records (not price, stock or delivery)
        keys = set()
        for v in d.values():
            if isinstance(v, dict):
                keys |= _words(" ".join(str(k).replace("_", " ") for k in v))
        values = [str(m).split(":", 1)[-1].strip() for m in (it.get("musts") or []) + told
                  if str(m).strip() and (not keys or _words(str(m).split(":", 1)[0]) & keys)]
        wanted = _words(" ".join(values))
        noun = str(it.get("what") or "")
        # the wanted value first: a keyword search may read only its first words
        queries = [f"{v} {noun}" for v in values[:2]] + [" ".join(values + [noun])]
        where = _shelf_choice(props, shelf or noun)
        limit = (props.get(arg) or {}).get("maxLength") or 200
        for q in dict.fromkeys(queries):
            args = {arg: _cut_words(q, limit), **where}
            for _ in range(2):
                if ep.finished or ep.left < END_RESERVE + 8:
                    break
                obs = ep.call(name, _coerce_args(args, _schema(ep.by_name[name])))
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
    shown: list[str] = []
    opened = 0
    for t in views[:1]:
        for a, o in _open_pages(ep, t, order[:6], END_RESERVE + 4):
            shown.append(f"{t} {json.dumps(a, ensure_ascii=False)} -> {_compact_obs(o, pages=True)}")
            if not _failed(o):
                got = next(iter(a.values()), [])
                opened += len(got) if isinstance(got, list) else 1
    rest = [f"{i} ({cands[i][1]:g})" for i in order[opened:opened + 12]]
    return (f"The shopper wants the lowest price, so more searches were run on the cart's shelf (steps used): "
            f"{len(cands)} listings in stock cost less than the carted one. The {opened} likeliest were opened:\n" +
            "\n".join(shown) + ("\nOthers not opened (id, price): " + ", ".join(rest) if rest else "") +
            "\nIf one of them meets every stated requirement and every need the shopper told by its page facts, put "
            "the cheapest such listing in the cart instead (same quantity, and take the old line out); otherwise "
            "keep your choice. Then close.")


def _cheaper_sweep(ep: Episode, brief: dict | None, roles: dict, view_cart: str | None) -> str | None:
    """The shopper wants the lowest price: list the carted line's category below its price with the shop's price
    filter (the code sets the filter, which the light model often misreads) and open the cheaper in-stock listings,
    so the check before the order can compare them."""
    pf = (brief or {}).get("price_filter")
    if not isinstance(pf, dict) or pf.get("tool") not in ep.by_name:
        return _keyword_sweep(ep, brief, roles, view_cart)
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
    if not lines or len(lines) > 4:
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

def _budget_cap(brief: dict | None) -> float | None:
    """The spending limit for the whole order as the brief now holds it (a change during the task included)."""
    b = (brief or {}).get("budget") if isinstance((brief or {}).get("budget"), dict) else {}
    amount = _number(b.get("amount"))
    return amount if amount and str(b.get("applies_to") or "total").lower() == "total" else None


def _cart_total(ep: Episode) -> float | None:
    """The cart's total: the shop's own total field in the latest cart reply, else line prices times counts."""
    obs = ep.cart_obs.get("guseri", ep.cart_obs) if isinstance(ep.cart_obs, dict) and not ep.cart_dirty else None
    if isinstance(obs, dict):
        for k, v in obs.items():
            if (isinstance(v, (int, float)) and not isinstance(v, bool) and re.search(r"total|sum", str(k), re.I)
                    and not COUNT_KEY.search(str(k))):
                return float(v)
    cart = _cart_now(ep)
    prices = [_listing_price(ep.listings.get(i) or {}) for i in cart]
    if not cart or any(p is None for p in prices):
        return None
    return sum(p * n for p, n in zip(prices, cart.values()))


def _state_line(ep: Episode, brief: dict | None, msgs_sent: int) -> str:
    lim = _message_limit(brief)
    cap = _budget_cap(brief)
    changed = any(re.search(r"budget|spend|limit|price|cost|\d", str(u), re.I) for u in (brief or {}).get("updates") or [])
    money = f"; whole-order budget {cap:g}" + (" (as changed during the task)" if changed else "") if cap else ""
    return f"[step {ep.step} of {ep.max_steps} used; {ep.left} left; messages sent {msgs_sent} of {lim}{money}]"


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


def _learn_adds(ep: Episode, before: float, n: Any, got: float | None) -> None:
    """Whether the cart tool adds the count it is given to a line already held, or sets the line to it: seen from a
    call on a line the cart held before (the shop's own count after it)."""
    if not before or got is None or not isinstance(n, (int, float)) or isinstance(n, bool) or n <= 0:
        return
    if abs(got - (before + n)) < 1e-9:
        ep.cart_adds = True
    elif abs(got - n) < 1e-9:
        ep.cart_adds = False


def _cart_put(ep: Episode, roles: dict, lines: dict[str, float]) -> bool:
    """Put chosen lines in the cart by code, reading the cart tool's own arguments (the listing, the count, a choice
    such as add or set)."""
    name = next((t for t in roles.get("cart_set", []) + roles.get("cart_change", []) if t in ep.by_name), None)
    if not name:
        return False
    schema = _schema(ep.by_name[name])
    props = schema.get("properties") or {}

    def keys_of(fields: dict) -> tuple[str | None, str | None]:
        texts = [k for k, v in fields.items() if isinstance(v, dict) and v.get("type") == "string" and not v.get("enum")]
        return (next((k for k in texts if re.search(r"id|item|listing|product|ref|sku", k, re.I)),
                     texts[0] if texts else None),
                next((k for k, v in fields.items() if isinstance(v, dict) and v.get("type") in ("integer", "number")),
                     None))
    arr = next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array"
                and (v.get("items") or {}).get("type") == "object"), None)
    if arr:
        # one call with a list of lines; a tool whose lines replace the cart gets what it already holds too
        idk, numk = keys_of((props[arr].get("items") or {}).get("properties") or {})
        if not idk or ep.finished or ep.left <= 2:
            return False
        replaces = re.search(r"replac|set the cart|overwrit", str((ep.by_name[name].get("function") or ep.by_name[name])
                                                                  .get("description")), re.I)
        wanted = {**_cart_now(ep), **lines} if replaces else lines
        obs = ep.call(name, _coerce_args({arr: [{idk: lid, **({numk: int(n)} if numk else {})}
                                                for lid, n in wanted.items()]}, schema))
        done = not (isinstance(obs, dict) and obs.get("error"))
        if done:
            ep.carted, ep.cart_dirty = True, True
            ep.cart_set.update({i: float(n) for i, n in wanted.items()})
        return done
    idk, numk = keys_of(props)
    done = False
    view = any(t in ep.by_name for t in roles.get("cart_view", []))
    for lid, n in lines.items():
        if not idk or ep.finished or ep.left <= 2:
            break
        have = _cart_now(ep).get(lid, 0)
        if have >= n:
            continue  # already there (a second call would add to the line in a shop whose cart tool adds)
        args: dict = {idk: lid, **({numk: int(n - have if ep.cart_adds else n)} if numk else {})}
        for k, v in props.items():
            if isinstance(v, dict) and isinstance(v.get("enum"), list) and v["enum"] and k not in args:
                args[k] = next((o for o in v["enum"] if re.search(r"add|put|set|hold|insert", str(o), re.I)),
                               v["enum"][0])
        obs = ep.call(name, _coerce_args(args, schema))
        ok = not (isinstance(obs, dict) and obs.get("error")) and not _failed(obs)
        done = done or ok
        if ok:
            got = _count_in(obs, lid)
            _learn_adds(ep, have, args.get(numk) if numk else 1, got)
            ep.cart_set[lid] = float(n) if got is None else got
            if not view and isinstance(obs, (dict, list)):
                ep.cart_obs = obs  # the cart tool's reply is the cart in a shop with no cart view
    if done:
        ep.carted = True
        ep.cart_dirty = view
    return done


def _fallback_close(ep: Episode, roles: dict, brief: dict | None) -> None:
    """Leave no episode open: order what the cart holds, else close with no purchase (or, when the request asks for
    the closest listing to be bought when nothing fits, first cart that choice)."""
    if ep.finished or ep.left <= 0:
        return
    items = [i for i in (brief or {}).get("items") or [] if isinstance(i, dict)]
    if not _cart_now(ep) and ep.pick and len(ep.pick) >= len(items) and ep.left >= len(ep.pick) + 1:
        _cart_put(ep, roles, ep.pick)  # the strong check's choice, when the loop ended with nothing carted
    fb = _fallback(brief)
    if fb and not _cart_now(ep) and ep.left >= 3:
        pick = _decide(ep, ep.policy.get("query") or "", brief, "the episode must end now and the cart is empty; "
                       "the request says what to do when nothing meets everything: " + str(fb.get("says") or ""))
        lines = _pick_lines(pick) if pick else None
        if lines:
            _cart_put(ep, roles, lines)
    if _cart_now(ep) and ep.left >= 4:  # the checks of the purchase the loop did not reach
        try:
            _cart_links(ep, roles, 2)
        except Exception:
            pass
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
    ep.search_tools = set(roles.get("search", []))
    try:
        stated = int((brief or {}).get("step_limit") or 0)
    except (TypeError, ValueError):
        stated = 0
    if 0 < stated < ep.max_steps:
        ep.max_steps = stated
    probe_notes = _probe(ep, brief, roles)
    _ask(ep, brief, roles, request)
    told_refs = [n for n in _reply_needs(ep, brief, request) if n.get("reference")]
    early_notes = list(ep.notes)  # changes that came with the first looks and the replies: on the brief before the loop
    _note_updates(ep, brief, early_notes)
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
    if early_notes:
        intro.append("NEW WORDS THAT CAME IN WHILE YOU ASKED (the shopper's changes override the request; the brief "
                     "above already holds them):\n" + "\n".join(f'"{n}"' for n in early_notes))
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
    plan = _search_plan(ep, brief, roles, request, probe_notes + ref_notes)
    if plan:
        intro.append(plan)
    intro.append(_state_line(ep, brief, msgs_sent) + " Begin.")
    messages: list[dict] = [{"role": "system", "content": LOOP_SYSTEM},
                            {"role": "user", "content": "\n\n".join(intro)}]

    reviews = 0
    idle = 0
    idle_steps = 0
    decided = False
    widened = False
    decline_pushed = False
    strong_decline = False  # the strong model itself chose to close with no purchase
    fallback_told = False  # the request's own rule for when nothing fits was put to the model
    nofit_told = False     # the shopper is to be told before a close with no purchase
    offer_told = False     # a close with no purchase names the listings that come closest
    links_sig = None       # the cart whose pages (and the listings they point to) were last checked
    qty_sig = None         # the cart whose counts were last checked against the request
    value_sig = None       # the cart whose lines were last checked against plain-valued requirements
    value_notes = 0
    said_state = None      # notices and updates known when the last message went out
    lock_off = False       # a close was turned down: a cart told to the shopper may change after all
    order_said = False     # the light model was told which checked lines to order
    code_ordered = False   # the code ordered those lines itself after the light model still declined
    over_told = 0          # orders held back because the cart total was over the whole-order budget
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
    unseen_pushes = 0  # declines turned back because a requested product had no listing opened
    queued: list = []  # cart changes the model sent together in one turn, still to run (one step each)
    close_tries = 0
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
    _spend.scale = 1.2 if ep.max_steps > 60 else 1.0  # and a larger spend line (such tasks are a third or less)
    while not ep.finished and ep.step < ep.max_steps:
        forced = ep.left <= 2  # one step to close, one more if the shop turns the close down
        closers = [t for t in ep.tools if _tool_name(t) in
                   set(roles.get("checkout", []) + roles.get("decline", []) + roles.get("finish", []))]
        offer = closers if forced and closers else ep.tools
        buy = [t for t in closers if _tool_name(t) in roles.get("checkout", [])]
        if forced and buy and _cart_now(ep) and not strong_decline:
            # the last step with lines in the cart: a close with no purchase is paid only where nothing fits, and
            # the strong model never said so here, so the cart is ordered (a change the shop just sent included)
            offer = buy
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
        fixing = fix_turns > 0 and not forced and _spent() < 1.5 * _hard()  # a fix's looking stops well past the line
        if fixing:
            fix_turns -= 1
            offer = ep.tools  # a review asked for more looking: it can only be done with the lookup tools
        elif (wrap_told and closers and not forced and empty_turns < FIX_TURNS and ep.left > END_RESERVE + 4
              and not _cart_now(ep)):
            # nothing chosen yet: with only the closing tools, the one close left would be no purchase
            empty_turns += 1
            fixing = True
            offer = ep.tools
        if closers and not forced and not lock_off and _told_cart(ep, close_steps, msgs_sent, msg_limit, cart_tools) \
                and said_state == (len(ep.notes), len((brief or {}).get("updates") or [])):
            offer = closers
        if _spent() >= 1.25 * _hard() and not fixing:
            over_turns += 1  # a few light turns to finish (cart, a step the shopper requires, the close), no more
        if queued and not forced and not ep.finished:
            reply = {"tool_calls": [queued.pop(0)], "content": ""}  # the next cart change of the same turn
        else:
            queued = []
            if idle_steps >= 8 or turns >= max_turns + 6 or _written() >= work_stop or over_turns > 3:
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
            # a reply in prose only: the same turn again at once with a call required (no thinking), so a model
            # route that lets prose through costs one call, not a turn of the loop
            reply = _chat_forced(messages + [{"role": "assistant", "content": content or "(no tool call)"},
                                             {"role": "user", "content": "Call one of the tools now."}], offer)
            calls = (reply or {}).get("tool_calls") or []
            content = (reply or {}).get("content") or content
            if not calls and idle >= 2 and not heavy_turn:
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
        names = [(c.get("function") or {}).get("name") if isinstance(c, dict) else None for c in calls]
        closers_now = set(roles.get("checkout", []) + roles.get("decline", []) + roles.get("finish", []))
        if (len(calls) > 1 and any(n in cart_tools for n in names[:-1]) and all(
                n in cart_tools or n == view_cart for n in names[:-1]) and (
                names[-1] in cart_tools or names[-1] == view_cart or names[-1] in closers_now)):
            # several cart changes in one turn (a view or the close may follow): each runs in order, one step
            # each, through the same checks as a call of its own
            queued = list(calls[1:])
            calls = calls[:1]
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
        if name in roles.get("search", []):
            args, cut = _short_query(ep, name, args)
            if cut:
                trim += (f"\n(The query was cut to its first {SEARCH_WORDS} words: keep each search to a few words, "
                         "the most telling wanted value first.)")
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
        read_key = _read_key(ep, name, args) if name in roles.get("search", []) else ""
        if read_key and read_key in ep.memo and memo_key not in ep.memo:
            k = len(json.loads(read_key[len(name):-5]).get(_query_arg(ep.by_name[name]), "").split())
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": (
                f"Not run (no step used): this shop's keyword search reads only the first {k} word"
                f"{'s' if k > 1 else ''} of a query, so this one reads the same as a search you already made. Its "
                "result was:\n" + ep.memo[read_key] + "\nTo search for another value, start the query with it.\n" +
                _state_line(ep, brief, msgs_sent))})
            continue
        if name in lookups and memo_key in ep.memo:
            messages.append({"role": "tool", "tool_call_id": call.get("id"),
                             "content": "You already made this exact call (no step used). Its result was:\n" +
                             ep.memo[memo_key] + "\nDo something new.\n" + _state_line(ep, brief, msgs_sent)})
            continue

        # running out of steps with nothing chosen: the strong model decides
        if (not decided and not ep.carted and _spent() < _hard()
                and (ep.left <= END_RESERVE + 5 or (_spent() >= _soft() and not wrap_told))
                and not _closing_kind(roles, name, args, brief)):
            decided = True
            pick = _decide(ep, request, brief)
            if pick:
                pick_lines = ep.pick = _pick_lines(pick)
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
                    decline_pushed = strong_decline = True  # the strong model chose this decline: do not argue with it
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
        # the last message told the shopper this cart and nothing has changed since: it is ordered as it stands
        locked = _told_cart(ep, close_steps, msgs_sent, msg_limit, cart_tools) and not lock_off \
            and said_state == (len(ep.notes), len((brief or {}).get("updates") or []))
        gate = "order" if final_msg else None if locked and kind == "order" else kind
        if kind == "decline" and not forced and unseen_pushes < 2 and ep.left > END_RESERVE + 6:
            missing = [m for m in _unopened_items(ep, brief) if m]
            if missing:
                # nothing fits is known only once each product's listings were looked at
                unseen_pushes += 1
                if wrap_told:
                    fix_turns = FIX_TURNS
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
                messages.append({"role": "user", "content": (
                    "From me, your supervisor: close with no purchase only when nothing fits, and no listing of the " +
                    " / ".join(missing) + " has had its page opened yet. Search for it in a few words (its most telling "
                    "wanted value first), open the best results, put the best fit for every product in the cart and "
                    "close with the purchase." + (f" The search and page tools are open again for {FIX_TURNS} turns."
                                                  if wrap_told else "") + "\n" + _state_line(ep, brief, msgs_sent))})
                continue
        objective = str((brief or {}).get("objective") or "").lower()
        early = ep.left > 12 if _cheap(brief) else (ep.step < 0.55 * ep.max_steps and ep.left > 10)
        if gate and early and not widened and not wrap_told and (gate == "decline" or _cheap(brief)):
            widened = True
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
            messages.append({"role": "user", "content": (
                "From me, your supervisor: you still have " + str(ep.left) + " steps, so search more widely "
                "before you " + ("decline" if gate == "decline" else "settle") + ": use each item's category "
                "filter search if a tool offers one and look at more of its result pages, try other wordings, and "
                "open the properties of the cheapest candidates you have not opened yet. Keep about 6 steps for "
                "the cart, a last check and closing. Then decide.")})
            continue
        n_items = len([i for i in (brief or {}).get("items") or [] if isinstance(i, dict)])
        held = _cart_now(ep)
        cap, total = _budget_cap(brief), _cart_total(ep) if kind == "order" else None
        if (kind == "order" and cap and total and cap < total <= 20 * cap and over_told < 2 and not forced
                and ep.left > 3):
            # an order over the shopper's limit (often one lowered during the task) cannot be paid: say so first
            over_told += 1
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
            messages.append({"role": "user", "content": (
                f"From me, your supervisor: the cart totals {total:g}, over the shopper's whole-order budget of "
                f"{cap:g} (the latest figure the shopper gave). Replace a line with a listing that fits every "
                "requirement and brings the total within the budget (a search with the price limit finds them), "
                "then close. If no such listing exists, or the total is in another unit than the budget, close "
                "again as you see fit.\n" + _state_line(ep, brief, msgs_sent))})
            continue
        if (kind == "decline" and pick_lines and order_said and not code_ordered and not forced
                and ep.left >= len(pick_lines) + 2 and len(pick_lines) >= n_items and set(held) <= set(pick_lines)
                and all(held[i] == n for i, n in pick_lines.items() if i in held)):
            # the strong check found listings meeting everything and the light model was told so, yet it closes
            # with no purchase: the code puts those lines in the cart and orders them
            code_ordered = True
            todo = {i: n for i, n in pick_lines.items() if i not in held}
            if not todo or _cart_put(ep, roles, todo):
                _fallback_close(ep, roles, brief)
                break
        if kind == "decline" and pick_lines and not placed and ep.left >= len(pick_lines) + 2:
            placed = True
            order_said = True
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
        fallback = _fallback(brief)
        if kind == "decline" and fallback and not fallback_told and not forced and ep.left > 2:
            fallback_told = True
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed."})
            messages.append({"role": "user", "content": (
                "From me, your supervisor: the request says what to do when nothing meets everything: \"" +
                str(fallback.get("says") or "") + "\". So close with a purchase, not without one: for each requested "
                "product that no listing fully meets, cart the listing seen that misses the fewest requirements (" +
                str(fallback.get("rule") or "as the request counts them") + "); every other line as usual. If a "
                "message is left and the request asks for it, tell the shopper so in one short message. Then close "
                "with the purchase.\n" + _state_line(ep, brief, msgs_sent))})
            continue
        report = (brief or {}).get("if_nothing_fits")
        if (kind == "decline" and isinstance(report, dict) and (report.get("offer") or report.get("tell"))
                and not report.get("buy_closest") and not offer_told and not forced and ep.left > 1):
            offer_told = True
            words = [k for k, v in (_schema(ep.by_name[name]).get("properties") or {}).items()
                     if isinstance(v, dict) and v.get("type") == "string" and not v.get("enum")]
            if words:
                # the request wants the listings that come closest put forward, not bought: the close's own words
                # carry them (code makes sure the close says which ones; the model picks them by the request's rule)
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
                messages.append({"role": "user", "content": (
                    "From me, your supervisor: the request says, for when nothing meets everything: \"" +
                    str(report.get("says") or "") + "\". That case is rare: a listing whose page says nothing about a "
                    "requirement is not short on it, so if any listing seen meets every requirement by its page facts "
                    "or is only unconfirmed on some, order the best such listing instead. Only when every listing "
                    "clearly fails a requirement by its own page facts: close with no purchase, and in the close's " +
                    " / ".join(words) + " say that nothing meets everything and name the id of each listing that "
                    "comes closest by that rule (for each product with no full fit), with what it falls short on.\n" + _state_line(ep, brief, msgs_sent))})
                continue
        if (kind == "decline" and isinstance(report, dict) and report.get("tell") and not nofit_told and not forced
                and msgs_sent < msg_limit and ep.left > 1 and not _closing_words(ep, brief)):
            nofit_told = True
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
            messages.append({"role": "user", "content": (
                "From me, your supervisor: the request says what to do when nothing meets everything: \"" +
                str(report.get("says") or "") + "\". Do that first, in one message in your own words, then close "
                "as it says.\n" + _state_line(ep, brief, msgs_sent))})
            continue
        cheap = _cheap(brief)
        if (gate == "order" and cheap and not swept and not forced
                and ep.left >= END_RESERVE + 10 + 8 * max(0, len(_cart_ids(ep)) - 1)):
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
                if wrap_told:
                    fix_turns = max(fix_turns, 2)  # opening them needs the page tools, withdrawn at the wrap
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
        if gate == "order" and not forced and qty_sig != _cart_sig(ep) and ep.left > END_RESERVE:
            qty_sig = _cart_sig(ep)
            gaps = _qty_gaps(ep, brief)
            if gaps:
                # a requested product with no line, once the wrap took the search tools away: they come back for a
                # few turns, as for a review asking for more looking (only listings seen could be carted without them)
                look = (wrap_told and reopened < 2 and ep.left > END_RESERVE + 6
                        and any(g.startswith("no line for") for g in gaps))
                if look:
                    reopened += 1
                    fix_turns = FIX_TURNS
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
                messages.append({"role": "user", "content": (
                    "From me, your supervisor: the cart does not match what the shopper asked: " + "; ".join(gaps) +
                    ". A cart tool's number may set a line's count rather than add to it (look at the cart view after "
                    "each change). Set each count as asked; a product with no line needs the best listing you found "
                    "for it (when none fits, do what the request says for that case). Then view the cart and close.\n" +
                    (f"For a product you have not found yet, the search and page tools are open again for {FIX_TURNS} "
                     "turns: search for it (its wanted value first, then its noun), open the best results, put the "
                     "best fit in the cart.\n" if look else "") +
                    _state_line(ep, brief, msgs_sent))})
                continue
        if gate == "order" and not forced and value_sig != _cart_sig(ep) and value_notes < 2 \
                and ep.left > END_RESERVE + 4:
            value_sig = _cart_sig(ep)
            clash = _value_conflicts(ep, brief)
            if clash:
                value_notes += 1
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
                messages.append({"role": "user", "content": (
                    "From me, your supervisor: the order is judged only by what each line's own page states, and "
                    "these lines miss: " + "; ".join(clash) + ". A value meets a requirement only when it holds the "
                    "wanted words (another spelling of the same word counts; a value that means something similar "
                    "does not). Look for a listing whose page states the wanted value (search with that value first, "
                    "then the product noun; open the pages) and put it in place of the line, same quantity, then view "
                    "the cart and close again. If no listing you find states it, keep the best line you have and "
                    "close.\n" + _state_line(ep, brief, msgs_sent))})
                continue
        if gate == "order" and not forced and links_sig != _cart_sig(ep) and ep.left > END_RESERVE + 1:
            links_sig = _cart_sig(ep)
            shown = _cart_links(ep, roles, END_RESERVE)
            if shown:
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not executed yet."})
                messages.append({"role": "user", "content": (
                    "From me, your supervisor: before the order, the cart's pages were checked (steps used):\n" +
                    "\n".join(shown) + "\nIf a page shown here changes your choice, fix the cart; else close again.\n" +
                    _state_line(ep, brief, msgs_sent))})
                continue
        decline_check = kind == "decline" and ((reviews <= MAX_REVIEWS and _spent() + HEAVY_CALL <= _hard())
                                               or (not decline_seen and _spent() < 1.35 * _hard()))
        if (gate and not forced and ep.left > 2 and not (gate == "order" and checked_sig == _cart_sig(ep))
                and (decline_check or (reviews < MAX_REVIEWS and _spent() < REVIEW_LINE * _soft()))):
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
                pick_lines = ep.pick = _verdict_lines(ep, brief, verdict)
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
            strong_decline = strong_decline or (bool(verdict) and other == "decline")
            if verdict and gate == "order" and other != "decline":
                checked_sig = _cart_sig(ep)
            if verdict and other in ("order", "decline") and gate in ("order", "decline") and other != gate:
                note = (f"Not closed yet. A check of your work says the right close is: {other}.\n"
                        f"Reason: {verdict.get('reason', '')}\n")
                if other == "order":
                    order_said = order_said or bool(pick_lines)
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
        said_now = (len(ep.notes), len((brief or {}).get("updates") or []))
        cart_before = _cart_now(ep) if name in cart_tools else {}
        adds_before = ep.cart_adds
        cart_pages: list[str] = []
        obs = ep.call(name, args)
        if ep.step > step_before:
            idle_steps = 0
        if name in roles.get("message", []):
            msgs_sent += 1
            ep.qa.append((str(next(iter(args.values()), "")), _reply_text(obs)))
            said_state = said_now
        if name in cart_tools and ep.step > step_before:
            ep.cart_dirty = True
            ep.carted = True
            refreshed = False
            ep.memo.clear()  # pages and results may change once something is in the cart
            if isinstance(obs, (dict, list)) and not (isinstance(obs, dict) and obs.get("error")) and (
                    not view_cart or _whole_cart(ep, obs)):
                # no cart view in this shop, or the cart tool answers with the whole cart: its reply is the latest
                ep.cart_obs = obs
                ep.cart_dirty = False
                if view_cart:
                    ep.cart_set = dict(_cart_now(ep))
            if not (isinstance(obs, dict) and obs.get("error")):
                lid = next((v for v in args.values() if isinstance(v, str) and v in ep.listings), None)
                n = next((v for v in args.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), 1)
                if lid:
                    got = _count_in(obs, lid)  # the shop's own count for the line (a removal reports 0)
                    if got is None and lid not in json.dumps(obs) and _shows_cart(ep, obs):
                        got = 0.0  # the cart shown has no such line any more (a bare "done" shows no cart)
                    _learn_adds(ep, cart_before.get(lid, 0), n, got)
                    ep.cart_set[lid] = float(n) if got is None else got
                    if ep.cart_set[lid] > 0:
                        # a line put in the cart is opened at once with each page tool that takes one listing and has
                        # not shown it yet (its facts and its page belong to the checks of a purchase; steps run out
                        # by the order)
                        for t in dict.fromkeys(roles.get("view", []) + roles.get("compare", [])):
                            if t in ep.by_name and not _list_arg(ep.by_name[t]) and t not in ep.opened_by.get(lid, set()):
                                cart_pages += [f"{t} {json.dumps(a, ensure_ascii=False)} -> {_compact_obs(o, True)}"
                                               for a, o in _open_pages(ep, t, [lid], END_RESERVE + 2, once=True)]
        if name == view_cart:
            ep.cart_obs = obs
            ep.cart_dirty = False
        text = _compact_obs(obs, name in ep.page_tools) + trim + _refusal_note(ep, name, args, obs)
        if cart_pages:
            text += "\nThe carted listing, opened for the purchase checks (steps used):\n" + "\n".join(cart_pages)
        if name in cart_tools and ep.cart_adds and not adds_before:
            text += ("\n(This cart tool adds the count you send to a line already in the cart; it does not set the "
                     "line to it. Check each line's count against what the shopper asked for.)")
        fresh = list(ep.fresh)
        fresh_done = False
        if name in lookups:
            ep.memo[memo_key] = text
        streak = streak + 1 if name in roles.get("search", []) else 0
        if name in roles.get("search", []):
            pages[qkey] = pages.get(qkey, 0) + 1
            note = _order_note(obs)
            if note:
                text = note + "\n" + text
            text += _empty_filter_note(ep, name, args, obs) + _ignored_filter_note(ep, name, args, obs)
            text += _store_note(ep, name, args)
            if ep.step > step_before and _read_key(ep, name, args):
                ep.memo[_read_key(ep, name, args)] = text
                for key in [x for x in ep.memo if x.startswith(name + "{")]:  # searches made before it was known
                    try:
                        rk = _read_key(ep, name, json.loads(key[len(name):]))
                    except ValueError:
                        rk = ""
                    if rk:
                        ep.memo.setdefault(rk, ep.memo[key])
        if kind and ep.step > step_before:
            close_tries += 1
        if ep.left <= END_RESERVE and not ep.finished:
            text += ("\nONLY A FEW STEPS LEFT: put your choice in the cart if it is not there, "
                     "then close the episode now.")
        elif ep.carted and not close_tries and ep.left <= CLOSE_RESERVE and not ep.finished:
            text += (f"\nSTEPS ARE RUNNING LOW ({ep.left} left): once the cart holds your choice for every product, "
                     "close now; the shop may answer a close with a change (stock, price, budget) that takes a few "
                     "steps to act on.")
        if dropped:
            text += "\n(Only this call ran: one tool per turn, except several page openings at once.)"
        messages.append({"role": "tool", "tool_call_id": call.get("id"),
                         "content": text + "\n" + _state_line(ep, brief, msgs_sent)})
        if kind and not ep.finished:
            lock_off = True  # the shop turned the close down: the cart may have to change after all
            pick = None
            if fresh:
                # the change that came with the refusal (a new budget, a dropped wish, a sold-out pick) goes on the
                # brief before anything is decided again
                _note_updates(ep, brief, fresh)
                fresh_done = True
            if redecides < 2 and (_spent() + HEAVY_CALL <= _hard() or fresh and _spent() < 1.3 * _hard()) and ep.left > 3:
                redecides += 1
                pick = _decide(ep, request, brief, "the shop did not accept the close; its reply was: " +
                               _compact_obs(obs)[:400])
                if pick:
                    pick_lines = ep.pick = _pick_lines(pick)
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
            if not fresh_done:
                _note_updates(ep, brief, fresh)
        if brief and brief.pop("recheck", False):
            # a dropped requirement or a raised limit: the cheaper checks and the order's check run again on the new
            # terms (code safeguards; the light model was told why in the relayed note)
            swept, cheaper_checks, checked_sig = False, 0, None
            reviews = min(reviews, MAX_REVIEWS - 1)

    return


UPDATE_SYSTEM = """A shopper's request in an online shop simulation was summed up in a brief; now a sentence came in
during the task. Say what it changes. Answer with the answer tool, or as JSON only:
{"said_by": "shopper" if the shopper says it, "shop" if it is a shop notice (a price, stock or status line),
 "new_budget": the new total spending limit as a number if it sets one, else null,
 "dropped": requirements the shopper no longer holds to, as "property: value" strings ([] if none),
 "added": new or changed wishes as "property: value" strings ([] if none),
 "gone": true if it says the pick in the cart (or another named listing) can no longer be had, else false,
 "new_counts": true if it changes how many of a product to buy, else false,
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
    if dropped or amount:
        brief["recheck"] = True  # listings set aside for what no longer holds may now be the better choice
    for it in brief.get("items") or []:
        if isinstance(it, dict) and isinstance(it.get("musts"), list) and dropped:
            it["musts"] = [m for m in it["musts"] if not any(_words(d) and _words(d) <= _words(m) for d in dropped)]
    if dropped and isinstance(brief.get("told_needs"), list):
        brief["told_needs"] = [n for n in brief["told_needs"] if not isinstance(n, dict) or not any(
            _words(d) and _words(d) <= _words(n.get("need") or "") for d in dropped)]
    if str(got.get("new_counts")).lower() == "true":
        brief["counts_changed"] = True
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
        "cart if needed; when a requirement was dropped or the limit raised, listings you set aside only for that "
        "may now be the better choice by the shopper's objective, so look at them again (and search once more "
        "without it) before closing. If it only says a price has changed, open the carted listing's page to read what it "
        "costs now, and keep it while the order total is within the spending limit and it is still the best choice by the "
        "shopper's objective, then close again. If it is only a shop notice (an error or status line), carry on.")}


__all__ = ["agent_main"]
