"""Terry: a shopping assistant for ORO Bench worlds.

How a task runs: the strong model studies the request once and writes a brief (what to buy, the limits, what is
worth asking). The light model then drives the world's own tools one step at a time with that brief in hand. Just
before the episode closes, the strong model compares the cart with everything found so far, and the light model
repairs whatever that check points out.

This first part holds the plumbing: pure builders and readers for model requests, JSON text, tool arguments and
world observations, plus the few functions that actually talk to the proxy.
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

# model seats: the light pair drives the tools, the heavy pair writes briefs, reviews and decisions
LIGHT, LIGHT_SPARE = "anthropic/claude-haiku-5.5", "openai/gpt-6-luna"
HEAVY, HEAVY_SPARE = "anthropic/claude-sonnet-5.5", "openai/gpt-6-sol"

_proxy = ProxyClient(max_retries=2, timeout=150)

# per-task limits: reviews, busy-account waits, steps kept for closing, dollars, loop turns, light tokens written,
# seconds since the task began (wrap: cart and close only, no strong-model checks; stop: no more model turns)
MAX_REVIEWS, MAX_WAITS, END_RESERVE = 2, 3, 3
# one more step behind the closing reserve: the shop may turn the first close down (with a shopper update) and the
# episode then needs a second close; lookups and the late senior choice leave it free
CLOSE_SPARE = 1
SOFT_SPEND, HARD_SPEND = 0.031, 0.045
MAX_TURNS = 32
WORK_WRAP, WORK_STOP = 18000, 24000
CLOCK_WRAP, CLOCK_STOP = 190.0, 250.0

_FCA_LIGHT_SEATS = (LIGHT, LIGHT_SPARE)
# per-token dollar rates (prompt, completion) used when a reply does not report its cost; key: is a light seat
_FCA_RATES = {True: (1e-7, 5e-7), False: (2e-6, 1e-5)}
# proxy statuses that rule a seat out at once (bad request, auth, credit, not found): try the next seat
_FCA_SEAT_DEAD = frozenset((400, 401, 402, 403, 404))
_FCA_BUSY = re.compile(r"in[_-]flight")
_FCA_NUMERIC = (int, float)
_FCA_NONE = object()  # "no value" marker where None itself is a legal value

# ---------------------------------------------------------------- spend ledger (one per worker thread)

_spend = threading.local()


def _spent():
    """Dollars this task's model calls have cost so far."""
    return vars(_spend).get("usd", 0.0)


def _written():
    """Completion tokens the light seats produced so far, reasoning included; run time grows with this."""
    return vars(_spend).get("light_out", 0)


def _elapsed():
    """Seconds since this task began (0 before agent_main set the start)."""
    began = vars(_spend).get("t0")
    return 0.0 if began is None else time.monotonic() - began


def _caps():
    """This task's (loop turns, soft $, hard $): set from the brief's product count, the defaults before that."""
    return vars(_spend).get("caps") or (MAX_TURNS, SOFT_SPEND, HARD_SPEND)


def _lane_caps(count):
    """Pure: caps by the number of requested products. Each product past the first adds loop turns; four or more
    also get more dollars, so every line can be searched, opened and carted before the turns run out."""
    extra = max(0, count - 1)
    if count >= 4:
        return MAX_TURNS + 5 * extra, 0.040, 0.060
    return MAX_TURNS + 4 * extra, SOFT_SPEND + 0.002 * min(extra, 1), HARD_SPEND + 0.003 * min(extra, 1)


def _work_caps():
    """Light tokens written before the wrap-up and before the stop, in step with this task's turn cap."""
    scale = _caps()[0] / MAX_TURNS
    return WORK_WRAP * scale, WORK_STOP * scale


def _fca_count(n):
    return isinstance(n, int) and not isinstance(n, bool)


def _fca_price(seat, usage):
    """(dollars, light tokens added) for one reply's usage record; heavy seats add no tokens."""
    light = seat in _FCA_LIGHT_SEATS
    usd = usage.get("cost")
    if isinstance(usd, bool) or not isinstance(usd, _FCA_NUMERIC):
        per_in, per_out = _FCA_RATES[light]
        usd = (usage.get("prompt_tokens") or 0) * per_in + (usage.get("completion_tokens") or 0) * per_out
    produced = usage.get("completion_tokens") if light else 0
    return float(usd), produced if _fca_count(produced) else 0


def _charge(seat, payload):
    usd, produced = _fca_price(seat, payload.get("usage") or {})
    _spend.usd = _spent() + usd
    if produced:
        _spend.light_out = _written() + produced


# ---------------------------------------------------------------- request builders (pure)

def _cacheable(messages):
    """A copy of messages whose system prompt and newest message carry cache breakpoints."""
    marked = copy.deepcopy(messages)
    last = len(marked) - 1
    for pos, entry in enumerate(marked):
        wanted = (pos == 0 and entry["role"] == "system") or (pos == last > 0)
        words = entry.get("content") if wanted else None
        if words and isinstance(words, str):
            entry["content"] = [dict(type="text", text=words, cache_control=dict(type="ephemeral"))]
    return marked


def _fca_body(seat, convo, budget, reasoning, cached):
    return {"model": seat, "messages": _cacheable(convo) if cached else convo,
            "max_tokens": budget, "temperature": 0, "reasoning": reasoning}


def _fca_chat_body(seat, convo, tools, choice, budget, think=True):
    """Request for one seat; Anthropic seats think, so they get cache marks and a free tool choice. think=False
    turns thinking off (a seat that spent its turn thinking and wrote no text)."""
    claude = seat.startswith("anthropic/")
    request = _fca_body(seat, convo, budget, {"effort": "low"} if think else {"enabled": False}, claude)
    if tools:
        request.update(tools=tools, tool_choice="auto" if claude else (choice or "required"))
    return request


def _fca_forced_body(convo, tools):
    """Request that must come back as a tool call: light seat, thinking off."""
    request = _fca_body(LIGHT, convo, 1200, {"enabled": False}, True)
    request.update(tools=tools, tool_choice="required")
    return request


def _fca_reask(convo, said):
    """The conversation extended by an unusable reply and a request to send the object alone."""
    return convo + [dict(role="assistant", content=said or "(empty)"),
                    dict(role="user", content="Answer again with nothing but the JSON object.")]


# ---------------------------------------------------------------- reply readers and retry policy (pure)

def _settling(error):
    """True for a refusal that means "busy, retry once in-flight requests settle" (not "out of credit")."""
    return _FCA_BUSY.search(str(error or "").lower()) is not None


def _fca_next_move(error, misses, tries):
    """After a call with no answer: ("wait", 0) for a busy account, ("seat", 0) to give up on this seat,
    or ("again", base pause) to retry the same seat."""
    if _settling(error):
        return "wait", 0
    if (error or {}).get("status") in _FCA_SEAT_DEAD or misses + 1 >= tries:
        return "seat", 0
    return "again", 3 * (misses + 1) + 2


# ---------------------------------------------------------------- shell: the calls that touch the proxy

_FCA_CHAT_PATH = "/inference/chat/completions"


def _fca_post(seat, request):
    """Send one completion request. Answered: charge it and return (message, None); else (None, error)."""
    got = _proxy.post_verbose(_FCA_CHAT_PATH, request)
    payload = got.data
    if not (payload and payload.get("choices")):
        return None, got.error
    _charge(seat, payload)
    first = payload["choices"][0]
    return first.get("message") or {}, None


def _wait_settle():
    """Sleep off a busy account; each task may do so MAX_WAITS times (it only has a few minutes)."""
    turn = vars(_spend).get("waits", 0) + 1
    if turn > MAX_WAITS:
        return False
    _spend.waits = turn
    time.sleep(5 * turn + 5 + random.uniform(0, 2))
    return True


def _chat(models, messages, *, max_tokens=1500, tools=None, tool_choice=None, tries=3, think=True):
    """First answer from the seats in order, or None. A busy account is waited out on the same seat."""
    for seat in models:
        request = _fca_chat_body(seat, messages, tools, tool_choice, max_tokens, think)
        misses = 0
        while misses < tries:
            message, error = _fca_post(seat, request)
            if message is not None:
                return message
            move, pause = _fca_next_move(error, misses, tries)
            if move == "seat":
                break
            if move == "wait":
                # all seats share one account: once the waits are spent, the spare seat would be busy too
                if _wait_settle():
                    continue
                return None
            misses += 1
            time.sleep(pause + random.uniform(0, 2))
    return None


def _chat_forced(messages, tools):
    """One tool call without thinking, for a light model that keeps replying in prose."""
    request = _fca_forced_body(messages, tools)
    for final in (False, True):
        message, error = _fca_post(LIGHT, request)
        if message is not None:
            return message
        if final:
            break
        if not _settling(error):
            time.sleep(2 + random.uniform(0, 2))
        elif not _wait_settle():
            break
    return None


def _ask_json(models, system, user, max_tokens=2500):
    """A JSON object from the seats; one re-ask when the first answer holds none. A reply with no text at all
    (the turn went to thinking) is re-asked with thinking off."""
    convo = [dict(role="system", content=system), dict(role="user", content=user)]
    think = True
    for _round in range(2):
        answer = _chat(models, convo, max_tokens=max_tokens, think=think)
        if answer is None:
            break
        said = answer.get("content") or ""
        parsed = _json_from(said)
        if parsed is not None:
            return parsed
        think = think and bool(str(said).strip())
        convo = _fca_reask(convo, said)
    return None


# ---------------------------------------------------------------- JSON in free text (pure)

_FCA_DEPTH = {"{": 1, "}": -1}


def _fca_close_of(blob, opener):
    """Index of the brace closing the object that opens at opener, or -1; braces inside strings do not count."""
    depth, quoted, escaped = 0, False, False
    for pos in range(opener, len(blob)):
        ch = blob[pos]
        if escaped:
            escaped = False
        elif quoted:
            escaped = ch == "\\"
            quoted = ch != '"'
        elif ch == '"':
            quoted = True
        else:
            depth += _FCA_DEPTH.get(ch, 0)
            if not depth:
                return pos
    return -1


def _json_from(blob):
    """The last whole JSON object in a reply: a model that corrects itself writes the good one last."""
    last = None
    opener = blob.find("{") if blob else -1
    while opener >= 0:
        closer = _fca_close_of(blob, opener)
        if closer < 0:
            break
        try:
            parsed = json.loads(blob[opener:closer + 1])
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            last = parsed
        opener = blob.find("{", closer + 1)
    return last


# ---------------------------------------------------------------- tool schemas and arguments (pure)

def _fca_spec(entry):
    return entry.get("function") or entry


def _schema(entry):
    return _fca_spec(entry).get("parameters") or {}


def _tool_name(entry):
    return _fca_spec(entry).get("name", "")


def _fca_words(phrase):
    return set(re.findall(r"[a-z0-9]+", phrase))


def _closest(given, options):
    """The option given most likely means: same text, then containment, then most shared words; else None."""
    want = str(given).strip().lower()
    named = [(opt, str(opt).lower()) for opt in options]
    tests = (lambda low: low == want, lambda low: bool(want and low) and (want in low or low in want))
    for test in tests:
        hit = next((opt for opt, low in named if test(low)), _FCA_NONE)
        if hit is not _FCA_NONE:
            return hit
    wanted = _fca_words(want)
    scores = [(len(wanted & _fca_words(low)), opt) for opt, low in named]
    top = max((s for s, _ in scores), default=0)
    return next(opt for s, opt in scores if s == top) if top else None


_FCA_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def _number(amount):
    """amount as a float (first number in its text when it is not numeric), or None."""
    if isinstance(amount, bool):
        return None
    if not isinstance(amount, _FCA_NUMERIC):
        hit = _FCA_NUM.search(str(amount or ""))
        return float(hit.group(0).replace(",", ".")) if hit else None
    finite = amount == amount and abs(amount) < 1e15
    return float(amount) if finite else None


def _fca_fit_number(given, rule, kind):
    got = _number(given)
    if got is None:
        return None
    for bound, clamp in (("minimum", max), ("maximum", min)):
        edge = rule.get(bound)
        if isinstance(edge, _FCA_NUMERIC):
            got = clamp(got, edge)
    return int(round(got)) if kind == "integer" else got


def _fca_fit_flag(given, rule, kind):
    if not isinstance(given, str):
        return bool(given)
    return given.strip().lower() in ("true", "yes", "1")


def _fca_fit_text(given, rule, kind):
    if isinstance(given, str):
        text = given
    else:
        text = json.dumps(given) if isinstance(given, (dict, list)) else str(given)
    cap = rule.get("maxLength")
    if isinstance(cap, int):
        text = text[:cap]
    if not text and rule.get("minLength"):
        text = "-"
    return text


def _fca_fit_list(given, rule, kind):
    if isinstance(given, str) and given.strip()[:1] == "[":
        try:
            given = json.loads(given)
        except ValueError:
            pass
    each = rule.get("items") or {}
    fitted = [_coerce(one, each) for one in (given if isinstance(given, list) else [given])]
    kept = [one for one in fitted if one is not None]
    cap = rule.get("maxItems")
    return kept[:cap] if isinstance(cap, int) else kept


def _fca_fit_record(given, rule, kind):
    return _coerce_args(given if isinstance(given, dict) else {}, rule)


_FCA_FITTERS = {"integer": _fca_fit_number, "number": _fca_fit_number, "boolean": _fca_fit_flag,
                "string": _fca_fit_text, "array": _fca_fit_list, "object": _fca_fit_record}


def _coerce(given, rule):
    """given reshaped to its schema entry, or None when it cannot be made to fit."""
    if not isinstance(rule, dict):
        return given
    choices = rule.get("enum")
    if isinstance(choices, list) and choices:
        return _closest(given, choices)
    kind = rule.get("type")
    fit = _FCA_FITTERS.get(kind) if isinstance(kind, str) else None
    return fit(given, rule, kind) if fit else given


def _fca_keep(name, given, rule, needed):
    """What to store for one declared argument; _FCA_NONE drops it. A required enum falls back to its first choice."""
    if given is None:
        return _FCA_NONE
    fitted = _coerce(given, rule)
    if fitted is not None:
        return fitted
    choices = rule.get("enum") if isinstance(rule, dict) else None
    if name in needed and isinstance(choices, list) and choices:
        return choices[0]
    return _FCA_NONE


def _coerce_args(given, shape):
    """given trimmed and fitted to a tool's parameter schema."""
    if not isinstance(shape, dict):
        return dict(given) if isinstance(given, dict) else {}
    pairs = given if isinstance(given, dict) else {}
    fields = shape.get("properties") or {}
    extras_ok = shape.get("additionalProperties") is not False
    if not (isinstance(fields, dict) and fields):
        return dict(pairs) if extras_ok else {}
    needed = shape.get("required") or []
    fitted = {}
    for name, val in pairs.items():
        if name in fields:
            val = _fca_keep(name, val, fields[name], needed)
        elif not extras_ok:
            continue
        if val is not _FCA_NONE:
            fitted[name] = val
    return fitted


def _parse_args(tool_call):
    """The arguments of a model tool call as a dict ({} when they cannot be read)."""
    fn = tool_call.get("function") or {}
    packed = fn.get("arguments") or "{}"
    if isinstance(packed, dict):
        return packed
    try:
        parsed = json.loads(packed)
    except ValueError:
        return _json_from(packed) or {}
    return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------- observations (pure)

_FCA_BLANK = (None, "", [], {})


def _shrink(node, cap):
    """node with strings over cap characters cut short and empty fields dropped."""
    if isinstance(node, dict):
        return {key: _shrink(sub, cap) for key, sub in node.items() if sub not in _FCA_BLANK}
    if isinstance(node, list):
        return [_shrink(sub, cap) for sub in node]
    if isinstance(node, str) and len(node) > cap:
        return node[:cap] + "\u2026"
    return node


def _fca_plain(n):
    return isinstance(n, (str, int)) and not isinstance(n, bool)


def _fca_id_named(field):
    return isinstance(field, str) and field.lower().endswith("id")


def _id_of(record):
    """The record's id: its "id" field, else the one field named ...id that holds a string of 3+ characters."""
    own = record.get("id")
    if _fca_plain(own) and str(own):
        return str(own)
    tails = [sub for field, sub in record.items() if _fca_id_named(field) and isinstance(sub, str) and len(sub) > 2]
    return tails[0] if len(tails) == 1 else None


_FCA_DIGIT = re.compile(r"\d")


def _code_like(token):
    if not isinstance(token, str) or " " in token:
        return False
    return 4 <= len(token) <= 64 and _FCA_DIGIT.search(token) is not None


def _list_key(rows):
    """The field that tells a list's records apart: unique in every record, and id-named or code-like."""
    if len(rows) < 2 or any(not isinstance(row, dict) for row in rows):
        return None
    fields = list(rows[0])
    ordered = [f for f in fields if _fca_id_named(f)] + [f for f in fields if not _fca_id_named(f)]
    for field in ordered:
        column = [row.get(field) for row in rows]
        if len(set(map(str, column))) < len(column):
            continue
        if (_fca_id_named(field) and all(map(_fca_plain, column))) or all(map(_code_like, column)):
            return field
    return None


def _fca_page_ref(record, asked_ids):
    """On a page view: the first plain field value that is an id the agent asked for."""
    refs = (str(sub) for sub in record.values() if isinstance(sub, (str, int)))
    return next((ref for ref in refs if ref in asked_ids), None)


def _listings_in(node, sink, asked_ids, page_view=False):
    """Append (listing id, record) pairs found anywhere in node, whatever the world names its fields."""
    if isinstance(node, list):
        key = _list_key(node)
        for entry in node:
            if not key:
                _listings_in(entry, sink, asked_ids, page_view)
                continue
            sink.append((str(entry[key]), entry))
            for inner in entry.values():
                _listings_in(inner, sink, asked_ids, page_view)
        return
    if not isinstance(node, dict):
        return
    ident = _id_of(node) or (_fca_page_ref(node, asked_ids) if page_view else None)
    if ident and len(node) > 1:
        sink.append((ident, node))
    for inner in node.values():
        _listings_in(inner, sink, asked_ids, page_view)


def _fca_children(node):
    if isinstance(node, dict):
        return node.values()
    return node if isinstance(node, list) else ()


def _strings_in(node, bag):
    """Add every string of 80 characters or fewer found in node to bag."""
    if not isinstance(node, str):
        for child in _fca_children(node):
            _strings_in(child, bag)
    elif len(node) <= 80:
        bag.add(node)


def _fca_number(n):
    return isinstance(n, _FCA_NUMERIC) and not isinstance(n, bool)


def _fca_rises(rows, field):
    """True when field is numeric in every row, not constant, and never goes down."""
    column = [row.get(field) for row in rows]
    if not all(map(_fca_number, column)) or len(set(column)) < 2:
        return False
    return all(lo <= hi for lo, hi in zip(column, column[1:]))


def _fca_tables(result):
    """Lists of three or more records directly inside a result."""
    for sub in result.values():
        if isinstance(sub, list) and len(sub) > 2 and all(isinstance(row, dict) for row in sub):
            yield sub


def _order_note(result):
    """A note for the model when a result list comes sorted by one of its numeric fields."""
    if isinstance(result, dict):
        for rows in _fca_tables(result):
            for field in rows[0]:
                if _fca_rises(rows, field):
                    return f"(These results are sorted by {field}, lowest first.)"
    return ""


def _fca_sentence(line):
    return isinstance(line, str) and len(line.split()) >= 5


def _fca_loose_lines(node, depth, ids):
    """Sentences in node (dicts down to depth 3, lists one level) that sit outside any listing record."""
    if depth > 3 or not isinstance(node, dict):
        return
    if any(_fca_plain(sub) and str(sub) in ids for sub in node.values()):
        return
    for sub in node.values():
        if isinstance(sub, dict):
            yield from _fca_loose_lines(sub, depth + 1, ids)
        elif isinstance(sub, list):  # some shops send such sentences as a list of lines
            yield from filter(_fca_sentence, sub)
        elif _fca_sentence(sub):
            yield sub


def _notes_in(result, known_ids):
    """Free sentences a result carries outside its listing records, such as a word from the shopper."""
    return list(_fca_loose_lines(result, 0, known_ids))


def _is_full_page(record):
    """True when some field of record is itself a record with two or more fields."""
    return any(isinstance(sub, dict) and len(sub) > 1 for sub in record.values())

def _fcb_holds_records(obs: Any) -> bool:
    """True for a result that carries a list of records (a search hit list and the like)."""
    if not isinstance(obs, dict):
        return False
    return any(isinstance(v, list) and isinstance(next(iter(v), None), dict) for v in obs.values())


def _compact_obs(obs: Any) -> str:
    """Observation as short JSON for the light model: record lists cut harder, empty fields gone."""
    cut = 140 if _fcb_holds_records(obs) else 700
    return json.dumps(_shrink(obs, cut), ensure_ascii=False)


# ---------------------------------------------------------------- prompts

BRIEF_SYSTEM = """Return exactly one JSON object and nothing else. It is a planning sheet for an assistant who will
buy on a customer's behalf in a simulated web store. You receive the customer's text and the store's tool list.
Small talk, firm conditions, caps and hints are mixed together in that text; sort them out with care.

== Fields about the purchase ==
"analysis" -> one or two sentences: what does the customer really need?
"items" -> a list with an entry for every product to buy. Entry fields:
   "what": product kind, worded the way the store would word it
   "names": 2 to 4 names a store may list this kind of product under, most likely first: synonyms, the word
      written joined and split (for example a two-word noun also as one word), the shelf's own name
   "value_words": for each entry of "musts" that a listing's text could show, its wanted value in at most 4
      words as a listing would write it; leave out money limits, counts, conditions that only exclude a value, and
      bounds on a figure (a minimum, a maximum or a range: a listing shows its own figure, never the bound); [] when
      nothing is left
   "category": if some tool lists categories, pick the fitting value of that list; otherwise null
   "qty": how many the customer wants (integer)
   "musts": each condition set for this product, as a brief text of the form "property: wanted value". Figures,
      units, ranges, negations ("not X") and alternatives ("either A or B") stay verbatim, and the value keeps the
      customer's wording (no paraphrase). The product kind itself does not go here.
   "reference": null unless a condition takes its value from a different listing's page; then describe it: seller,
      kind of product, identifying details, the property whose value is copied, and the relation needed (same,
      within N, at least ...)
"budget" -> {"amount": number or null, "unit": unit as written by the customer, "applies_to": "total" or "each"}
"objective" -> how to pick among listings that pass everything, for instance "cheapest total" or
   "lowest unit price" or "best match" or "any that fits"
"skip_sold_out" -> true by default; false only if the customer accepts items that are out of stock

== Fields about the store's tools ==
"tool_roles" -> each tool's name -> one role from this set: search, view, compare, message, cart_change, cart_set,
   cart_view, checkout, decline, finish (one tool that does both ordering and declining), other
"finish_modes" -> {"<tool>": {"order": values for its choice-type arguments that buy, "decline": values that end
   without buying}} for every finish tool, as its schema shows; {} when no finish tool exists
"search_page_args" -> {"<search tool>": the argument that takes the token for the next page of results} for
   each search tool that has one; {} when none does
"price_filter" -> null unless some search tool can restrict results by a maximum price; then {"tool": that
   search tool, "max_price_arg": which argument holds the maximum price, "category_arg": which argument restricts
   results to a single category of product, or null, "page_arg": which argument takes a paging token, or null},
   taken from the arguments
"probes" -> opening searches, one per item: [{"item": product, "tool": name of a search tool, "args": the
   arguments to send it}]. Fill in the category filter when the tool has one, plus a brief keyword (product noun
   and no more than one stated value); omit paging and page-token arguments.
"cart_write" -> null when no tool changes the cart; else how that tool is driven, read off its schema:
   {"tool": its name, "id_arg": the argument holding the listing id, "count_arg": the argument holding a count, or
   null, "count_means": "set" when the count becomes the line's quantity or "add" when it is added to the line,
   "add_args": {any other fixed arguments that put items in, such as a mode value}, "remove_args": {the fixed
   arguments that take items out (the count then says how many), or null when a count of 0 removes the line}}

== Fields about limits and asking ==
"message_limit" -> integer: how many messages the customer accepts from the assistant at most; null if never said
"question_cost" -> any stated price for each question or message, as text; else null
"step_limit" -> integer cap on tool calls if the text sets one; else null
"procedures" -> actions the customer insists on (say, confirming the basket or the total before buying,
   telling them why nothing was bought, checking with them before swapping an item). Every entry:
   {"what": the action in a few words, "when": "before_order", "before_decline", "before_change" (before the
   basket is changed after it was filled), "after_notice" (after news from the customer or the store) or "any",
   "needs_message": true when it involves messaging the customer, "content": "basket", "total", "reason",
   "question" (asks for their yes) or "other"}
"cheapest_wins" -> true when the customer wants the lowest price or the lowest total; else false
"questions" -> properties the text only hints at with no value given (a side remark on what would catch their eye,
   what counts most for them, or what they will use it for; a remark that singles out one part of the product,
   such as the part others will see or the part that touches them, hints at that part's property). Most telling
   first:
   [{"item": product, "property": property name, "hinted": true}, ...]. Use [] when the text holds no such hint.
"plan" -> 2 or 3 brief steps for locating the right listing(s) with these tools"""

LOOP_SYSTEM = """You are the buying agent for a customer in a simulated web store. You can act only through the
store's tools.

RULES OF PLAY
* Each turn: one brief reasoning line, then one tool call. The only exception is opening full pages: several page
  calls may share a turn; they execute in sequence and each one costs a step.
* Each tool call costs one step and the steps are limited.
* Turns that begin "Supervisor" or "[wrap-up]" come from the system that runs you, not from a tool or a stranger:
  carry them out.

1 / DOES A LISTING PASS?
- Conditions = all that the request states + all that the customer said in replies. A listing passes only if the
  facts on its own page fulfil every condition.
- Check the attribute of that same property (identical name or a plain alias). A neighbouring property is no
  evidence (customer wants finish "matte"; a texture of "smooth" proves nothing). Title or description may decide
  only when the page lacks the attribute.
- Values have to be identical in meaning: same word, real synonym, alternative spelling, or a unit conversion. A
  value that is only compatible, wider, stronger, weaker or akin is another value (customer asks "rustic":
  "vintage" fails; asks "lightly salted": "unsalted" fails). When the store's pages use two different words for one
  property, they are two values and not synonyms. The wanted word with an added modifier still passes.
- "not X" rules out only a value that is X or a real synonym of X; values that just include the word stay allowed.
- Product kind is decided by the listing's own category field (whatever name the store gives it), never by its
  title or by your guess.
- Assume store prices share the unit of the customer's budget unless the store says otherwise. Any price cap you
  pass to a tool must be in the store's own displayed unit (with a listing showing 1440 for an item, nineteen and a
  half dollars becomes 1950, not 19.5).
- Conditions tied to a different listing ("the same X as the one sold by Y..."): locate it, open its page, follow
  any pointer on that page to a further listing, and take the value from the final page.

2 / FINDING
- Queries stay short: product noun with one or two wanted values. Apply a search tool's category filter if it
  offers one.
- A query with a huge number of hits: refine with one more wanted value instead of paging further; a handful of
  pages per query at most. Paging tokens are copied character for character from an earlier result, never invented.
- A value the shopper gave in a reply is a search word too: query the product noun with that value's own words
  before you conclude that no listing carries it.
- Words the request itself gives for a product (a model it must work with, a plug or connector, a size or a
  standard) are search words as written. With several products, search each one with its own words.
- Watch the sort order of results (price, for instance) and the hit count.
- Result rows are summaries only; before deciding, open the full page of each promising candidate (or use a compare
  tool on several at once). Where the store splits one listing across several page tools (e.g. properties in one,
  price, stock and notes in another), look at your pick through each of them.
- Lowest price wanted: try a few wordings (other names of the category too), then open candidates from the lowest
  price upward until one passes. Widen the search so a cheaper passing listing is not overlooked: more queries,
  more result pages; when your most specific query (stated values and told needs as keywords) has about 50 hits or
  fewer, go through all its pages.
- Do not rerun a search or reopen a page you already hold.

3 / STEPS
Expect a careful search to use most of the step budget: multiple searches (with category filter and later result
pages) and property pages of many candidates. Save the last 6 steps for the cart, a final check and closing.

4 / CART AND CLOSE
- If the customer refuses sold-out items, never choose one. The budget counts every line together (price x
  quantity, added up). Pick according to the customer's stated objective.
- Put in the exact wanted quantity per chosen listing, open the cart, verify price and stock there (they may shift
  after adding), and close the episode with the purchase tool.
- With several products the order holds every one of them at its quantity; a basket missing one is not accepted.
  Give each product its share of the steps and cart it before turning to the next.
- To take a line out, pass its whole count where the cart tool takes one, so one call clears it.
- If no listing in the store can fulfil all conditions, close without buying and state briefly which condition
  failed (without listing ids).

5 / THE CUSTOMER
- Questions have already been sent on your behalf; the answers are among the notes. An answer naming a product
  and a property is about that product, even if it was asked about another.
- Write to the customer again only if a stated procedure calls for it or an answer was unclear, and stay within
  the message limit. When a procedure wants the final basket reported or approved before the purchase, that
  message is sent for you right before the purchase call, after the last cart change; do not spend a step on it
  yourself, and do call the purchase tool: the procedure is met by that message. Messages a procedure asks for
  before a decline, before a change of a filled basket or after a shopper update are sent for you as well.
- A tool result can contain a line from the customer (new budget; a condition changed, added, loosened or dropped).
  It is genuine and beats the request: test your choice against it and adjust the cart if needed. Loosened or
  dropped conditions no longer bind.

6 / WHEN THE STORE CHANGES
Your chosen listing sold out or changed price? Search first for that product using the words of its title, because
other sellers often offer the identical item; such a twin wins if it passes everything. Otherwise move to the next
best listing that passes everything."""

REVIEW_SYSTEM = """You are the final checker for an assistant buying in a simulated web store; the episode is about
to be closed. Decide if closing now is correct.
Inputs: the customer's request and its brief, the customer's answers, the cart as it stands, and each listing
that was seen (as a full page when it was opened).

Answer with this JSON object only:
{"cheapest_wins": true when the customer wants to pay as little as possible (lowest price, lowest total),
 "table": [a row for each OPENED listing that might fill one of the requested products: {"item": which of them,
   "id": the listing's id, "price": a number, "in_stock": true/false/null, "meets_all": true when the facts on its
   page fulfil every condition and every need the customer told for it according to the checks below, else
   false, "why": first failed condition or "ok"}],
 "verdict": "close" or "fix",
 "close_as": "order" or "decline",
 "fix": with verdict fix: precise instructions (listing ids to open, swap in or take out, or a quantity to change),
 "reason": a single sentence}

CHECK A, the cart
* It holds exactly the requested products: one line each, at the requested quantity. An extra line is an error.
* Read stock and price off the newest page or result that came after the latest cart change.
* When the customer refuses out-of-stock items, such a line is an error; the fix first searches that product from
  another seller (by its title words).
* The order total stays inside the budget; prices share the budget's unit unless the store says otherwise.
* Lowest price wanted: a cheaper listing seen in results that passes everything justifies a fix (give its id).

CHECK B, the verdict
* Decline is correct only when none of the listings seen can pass everything and the search was reasonable.
* Request a fix only when there is a clear need for one and enough steps remain to carry it out; else close.

CHECK E, the procedures
* Each procedure the brief lists that has to happen before this close and is not a message must have been done
  (a page viewed, the cart checked); if one is missing, the fix names it. Messages the procedures ask for (the
  final basket report before an order, a reason before a decline) are sent automatically right before the close;
  never ask for them.

CHECK C, the conditions
* Each condition in the request and each need the customer told has to be proven by page facts.
* Answers that are garbled, beside the point or no real answer bring no condition.
* An answer naming a product and a property attaches that need to the named product, whatever product the
  question was about.
* Customer lines that arrived inside tool results (new budget; condition changed, loosened or dropped) are genuine
  and beat the request; a condition that was loosened or dropped no longer binds.
* Use only the store's data and the customer's words, never a guess about what the product is for.

CHECK D, page facts versus conditions
* Look up the very property in question on the page (same name or a plain alias); it has to carry the wanted
  value. Synonyms and unit conversions are fine; a related but different property is no proof.
* Identical value means: same word, real synonym, alternative spelling, unit conversion. Values that are only
  compatible, wider or akin fail (asked "rustic": "vintage" fails; asked "lightly salted": "unsalted" fails, as
  stronger, weaker or adjacent values are other values; two different words that the store's pages use for one
  property are two values, not synonyms).
* A page value holding the wanted value as a complete word next to a modifier passes (asked "green": both
  "green leather" and "dark green" pass).
* "not X" excludes X itself or a real synonym and nothing else; a different value merely containing the word
  (such as "X-look" or "faux X") remains allowed.
* When the page lacks any attribute for a needed property, its title and description settle it. If they too say
  nothing, see whether other opened pages of the same category record that property as an attribute: if one does,
  the store tracks it, so the silent listing leaves the condition unproven and does not meet it (why: "not shown").
  Only when no opened page of that category records the property can silence not rule the listing out (plenty of
  stores show no such attribute).
* If a property is unknown merely because the listing was never opened with one of the store's page tools
  ("opened_with"), the fix should open it that way.
* What kind of product a line is comes from the category field on the listing itself (under whatever name the
  store uses), not its title and not your impression; each line must belong to the category the customer names."""


# ---------------------------------------------------------------- environment

_FCB_TRIES = 5
_FCB_REJECTED = frozenset((422, 413, 404, 403, 401, 400))  # the shop turned the call down


def _fcb_step_outcome(res: Any, attempt: int) -> str:
    """What one proxy reply means for the step: ok, bump (stale step number, try the next one), stop (the episode
    refuses), refused (the shop rejected the call), or wait (try again later)."""
    if res.data is not None:
        return "ok"
    code = res.error.get("status") if res.error else None
    if code == 409:
        return "stop" if attempt else "bump"
    return "refused" if code in _FCB_REJECTED else "wait"


def _fcb_merge_listing(old: dict | None, rec: dict, full: bool) -> dict | None:
    """The stored record after seeing rec, or None to keep the old one (a full page is never replaced by a summary)."""
    if old is not None and not full and old.get("_full"):
        return None
    return {**(old or {}), **rec, "_full": full}


def _fcb_fold_hits(stored: dict, hits: list, named: set, page_call: bool, split_pages: bool) -> tuple[dict, list]:
    """Fold the listing records an observation shows, in order, into the records to write and the ids opened in
    full. With page tools known, a record is a full page only when a page tool was called for its id."""
    writes: dict = {}
    opened: list = []
    for lid, rec in hits:
        full = (page_call and lid in named) if split_pages else _is_full_page(rec)
        merged = _fcb_merge_listing(writes[lid] if lid in writes else stored.get(lid), rec, full)
        if merged is not None:
            writes[lid] = merged
        if full:
            opened.append(lid)
    return writes, opened


_FCB_GAP = 0.3  # seconds between two world calls of one task: many tasks start together and the shop rate-limits


def _fcb_pace():
    """Keep a short gap since this task's previous world call (calls issued by code come back to back)."""
    last = vars(_spend).get("last_call")
    if last is not None:
        wait = _FCB_GAP - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
    _spend.last_call = time.monotonic()


class Episode:
    """Everything one episode knows: the world binding, tools, step count, and what the world has shown."""

    def __init__(self, env: dict):
        pv = env["policy_view"]
        self.binding, self.policy = env["binding"], pv
        self.tools = pv.get("tools") or []
        self.by_name = dict((_tool_name(t), t) for t in self.tools)
        self.max_steps = int(pv.get("max_steps") or 30)
        self.step, self.finished = 0, bool(pv.get("finished"))
        # world record: the raw dialogue, listings by id, opened pages, stray sentences
        self.dialogue, self.notes, self.fresh, self.bool_pages = [], [], [], []
        self.listings, self.opened_by = {}, {}
        self.viewed, self.seen_strings = set(), set()
        # filled in by the driver
        self.message_tools, self.page_tools = set(), set()
        self.probe_of, self.memo = {}, {}
        self.searched = set()  # code-run search calls; kept when a cart edit clears the memo
        self.qa = []
        self.ref_notes = []  # pages a condition points to, found and opened by code
        self.told = []  # (product, property, reply) for each reply that named the asked property
        self.told_values = []  # (product, value) for each value a reply to a question insisted on
        # cart bookkeeping; cart_set = listing -> count from this agent's own cart calls; cart_rev counts cart edits
        # and stated_rev is the edit the last message to the shopper came after
        self.cart_obs, self.cart_dirty, self.carted = None, False, False
        self.cart_set = {}
        self.cart_rev = self.stated_rev = 0
        # procedure moments whose message went out; the shopper's reply to the last basket report
        self.fired, self.statement_reply = set(), None
        # the store's search: words and hits of the first probe search; "all" when every query word must match,
        # "any" when results are ranked by matching words, None until measured
        self.probe_hits, self.search_mode = None, None

    def _fcb_send(self, name: str, args: dict) -> tuple[Any, Any]:
        """Post one action; returns (result, None) or (None, error observation). Keeps self.step in line."""
        action = {"name": name, "args": args}
        detail = None
        for attempt in range(_FCB_TRIES):
            res = _proxy.post_verbose("/environment/call",
                                      json_data=dict(self.binding, step=self.step, action=action))
            kind = _fcb_step_outcome(res, attempt)
            if kind == "ok":
                return res.data, None
            if kind == "stop":
                self.finished = True
                return None, {"error": "the episode does not take more steps"}
            if kind == "refused":
                detail = (res.error or {}).get("body")
                break
            if kind == "bump":
                self.step += 1
            else:
                time.sleep(2.0 * (attempt + 1) + random.uniform(0.0, 2.0))
        self.step -= 1  # nothing was taken, so the number is free again
        return None, {"error": "the shop rejected that call", "detail": str(detail or "no answer")[:300]}

    def call(self, name, args) -> Any:
        """Take one step in the world and return what it shows."""
        self.fresh = []
        if self.finished:
            return {"error": "the episode is already over"}
        self.step += 1
        _fcb_pace()
        result, failed = self._fcb_send(name, args)
        if result is None:
            return failed
        self.finished = bool(result.get("finished"))
        self.dialogue.append(dict(role="assistant", tool_call=dict(name=name, args=args),
                                  environment_result=result))
        seen = result.get("observation")
        self._learn(seen, name, args)
        return seen

    def _learn(self, obs, name="", args=None) -> None:
        """File what an observation shows: strings, listings (full pages apart), and stray sentences."""
        _strings_in(obs, self.seen_strings)
        named: set = set()
        _strings_in(args or {}, named)
        page_call = name in self.page_tools
        if (page_call and name not in self.bool_pages and isinstance(obs, dict)
                and any(type(v) is bool for v in obs.values())):
            self.bool_pages.append(name)
        hits: list = []
        _listings_in(obs, hits, named, page_call)
        writes, opened = _fcb_fold_hits(self.listings, hits, named, page_call, bool(self.page_tools))
        self.listings.update(writes)
        for lid in opened:
            self.viewed.add(lid)
            self.opened_by[lid] = self.opened_by.get(lid, set()) | {name}
        if name in self.message_tools:
            return
        for line in _notes_in(obs, self.listings.keys() | named):
            if line not in self.notes:
                self.notes.append(line)
                self.fresh.append(line)

    @property
    def left(self) -> int:
        """Steps still to spend."""
        return self.max_steps - self.step


def _fcb_role_pairs(given: Any) -> list:
    """(tool, role) pairs from the brief's tool_roles, sent either as a map or as a list of rows."""
    if isinstance(given, dict):
        return list(given.items())
    if isinstance(given, list):
        rows = {str(r.get("tool") or r.get("name")): r.get("role") for r in given if isinstance(r, dict)}
        return list(rows.items())
    return []


def _role_tools(brief, ep) -> dict:
    """Role -> the known tools the brief gave that role."""
    out: dict[str, list[str]] = {}
    for tool, role in _fcb_role_pairs((brief or {}).get("tool_roles") or {}):
        if isinstance(role, str) and tool in ep.by_name:
            out.setdefault(role.strip().lower(), []).append(tool)
    return out


def _fcb_arg_specs(tool: dict) -> list[tuple[str, dict]]:
    """A tool's arguments whose schema is a proper map, in schema order."""
    return [(k, v) for k, v in (_schema(tool).get("properties") or {}).items() if isinstance(v, dict)]


def _fcb_free_text_keys(tool: dict) -> list[str]:
    """Arguments that take free text: string type, no fixed choices."""
    return [k for k, v in _fcb_arg_specs(tool) if "enum" not in v and v.get("type") == "string"]


def _text_arg(tool: dict) -> str | None:
    """The free-text argument to fill: the first required one, else the first one there is."""
    keys = _fcb_free_text_keys(tool)
    must = _schema(tool).get("required") or []
    return next((k for k in must if k in keys), keys[0] if keys else None)


def _fcb_array_arg(tool: dict) -> str | None:
    """A page tool's list-typed argument (it opens several ids in one step), if it has one."""
    lists = [k for k, v in _fcb_arg_specs(tool) if v.get("type") == "array"]
    return lists[0] if lists else None


# ---------------------------------------------------------------- brief

def _fcb_arg_label(key: str, spec: dict) -> str:
    choices = spec.get("enum")
    if choices:
        return "%s (one of: %s)" % (key, ", ".join(str(c) for c in choices))
    return "%s (%s)" % (key, spec.get("type", "any"))


def _tools_digest(tools: list[dict]) -> str:
    """One line per tool: name, description, arguments (choices spelled out)."""
    out = []
    for tool in tools:
        fn = tool.get("function") or tool
        spec = (fn.get("parameters") or {}).get("properties") or {}
        labels = "; ".join(_fcb_arg_label(k, v) for k, v in spec.items())
        out.append("- %s: %s Args: %s" % (fn.get("name"), fn.get("description", ""), labels or "none"))
    return "\n".join(out)


def _fcb_brief_request(request: str, max_steps: int, tools: list[dict], opening: Any) -> str:
    """The user turn of the brief call."""
    parts = [f"SHOPPER REQUEST:\n{request}", f"STEP LIMIT FROM THE SHOP: {max_steps}",
             "TOOLS:\n" + _tools_digest(tools)]
    if opening is not None:
        parts.append("OPENING NOTE FROM THE SHOP:\n" + json.dumps(opening, ensure_ascii=False))
    return "\n\n".join(parts)


def _make_brief(ep, request) -> dict | None:
    """The strong-model call that reads the request and the tools. When both strong seats fail (a key out of
    credit refuses the larger reply first), the light seats read it with less room: without a brief the episode
    loses its roles, procedures and basket report."""
    user = _fcb_brief_request(request, ep.max_steps, ep.tools, ep.policy.get("opening"))
    return (_ask_json([HEAVY, HEAVY_SPARE], BRIEF_SYSTEM, user, 4000)
            or _ask_json([LIGHT, LIGHT_SPARE], BRIEF_SYSTEM, user, 2500))


_FCB_BRIEF_SHOWN = ("items", "budget", "objective", "skip_sold_out", "message_limit", "question_cost",
                    "procedures", "plan")


def _brief_text(brief: dict) -> str:
    """The parts of the brief later prompts show, as indented JSON."""
    shown = dict((k, brief.get(k)) for k in _FCB_BRIEF_SHOWN if brief.get(k) is not None)
    return json.dumps(shown, ensure_ascii=False, indent=1)


def _message_limit(brief) -> int:
    """How many messages the shopper allows; 3 when the brief gives no usable number."""
    try:
        given = int((brief or {}).get("message_limit"))
    except (TypeError, ValueError):
        given = 3
    else:
        given = given if given > 0 else 0
    return given


def _reserved_messages(brief) -> int:
    """Messages kept back for procedures that need one: one for the close (an episode ends by ordering or by
    declining, never both) and one for each other moment a procedure names."""
    moments = {m for m in _FCF_MOMENTS if _procedures(brief, m)}
    closing = {"before_order", "before_decline"}
    return int(bool(moments & closing)) + len(moments - closing)


# ---------------------------------------------------------------- asking

QUESTION_SYSTEM = """Answer only with JSON of the form {"item": product name as the request words it, "property":
the property's name spelled as the store's pages spell it (one of their property keys; treat underscores as
spaces), "why": short reason}.

Context. An assistant may send a customer messages, one property per message, and only a few of them. The
customer has silent preferences about some product properties. You decide which property gets asked NEXT. Inputs:
the request and brief, hints taken out of the request, earlier questions together with their answers, and which
properties the store's own pages record.
A message left unused is lost, so name a property as long as an unasked one exists.

NEVER select
 - a property whose value the request already gives;
 - something asked before (compare meaning, not wording);
 - a field marked in the table as several choices in one value;
 - a code or model number;
 - a fact about the offer or the store instead of the product itself;
 - a field whose name is a plural noun for choices in general (the customer takes a question on it as several
   questions and answers nothing); a singular property name is fine. Where the table marks two properties as
   sharing values, ask only the one whose name says plainly what those values are: it takes the other's place
   and rank among the candidates of step 2, wherever it stands in the table.

HOW TO SELECT
 1. Candidates come from the PROPERTY TABLE: per product, the properties its store pages record, highest rank on
    top (the rank grows with the number of opened pages recording it and with the spread of its values across
    listings). The silent preferences concern these.
 2. Of that product's four top-ranked properties not yet asked, choose the one buyers most often decide that kind
    of product on: appearance, material, fit or function, what goes in it or what powers it, who it is for. A field
    marked as listing several values on each page covers many choices in one listing: it comes last among them.
 3. A remark in the request (or a listed hint) that singles out one part of the product, such as the part others
    will see or the part that touches them, promotes the table property that describes that part, ahead of the
    choice of step 2 and wherever it stands in the table. The pages may call that part by another name: match the
    part's meaning, not the customer's word. Any other remark promotes a property only if it clearly refers to one
    in the table.
 4. Several products and the input names a PRODUCT TO ASK ABOUT NOW: ask a property of that product, under the
    name its pages use.
 5. Always the pages' name for the property, never the customer's phrasing.

ANSWERS RECEIVED SO FAR
 - Stating a value fixes that property as a genuine preference: do not repeat it.
 - Saying any option is fine leaves that property open: do not repeat it.
 - Saying they do not understand means no preference on that property: do not repeat it under any name, and turn
   to a different property.
 - Asking for one thing at a time means the question was read as about several properties and nothing came out:
   do not repeat it; if the table marks it as sharing values with another property, ask that property next."""


def _has_props(obs: Any) -> bool:
    """Whether a page records properties: some field (here or in a nested list) is a map of two or more values."""
    if isinstance(obs, list):
        nested = obs
    elif isinstance(obs, dict):
        if any(isinstance(v, dict) and len(v) > 1 for v in obs.values()):
            return True
        nested = [v for v in obs.values() if isinstance(v, list)]
    else:
        return False
    return any(map(_has_props, nested))


def _fcb_unopened(obs: Any, viewed: set) -> list[str]:
    """Listing ids a result shows, first-seen order, without the ones already opened."""
    hits: list = []
    _listings_in(obs, hits, set())
    return [lid for lid in dict.fromkeys(lid for lid, _ in hits) if lid not in viewed]


def _fcb_page_args(tool: dict, lid: str, batch: list[str]) -> dict | None:
    """Arguments that open lid with a page tool: the whole batch for a list argument, else lid as text."""
    many = _fcb_array_arg(tool)
    if many:
        return _coerce_args({many: list(batch)}, _schema(tool))
    one = _text_arg(tool)
    if not one:
        return None
    return _coerce_args({one: lid}, _schema(tool))


def _fcb_keyword_tool(ep: Episode, roles: dict, used: str) -> str | None:
    """Another search tool that takes a free-text query."""
    for cand in roles.get("search") or []:
        spec = ep.by_name.get(cand)
        if cand == used or spec is None:
            continue
        if _text_arg(spec):
            return cand
    return None


def _probe(ep, brief, roles, per_item=None, per_call=6) -> list:
    """Before any question: run the brief's first search per item and open some hits with the page tools, so the
    shop's own property names (and how many pages carry each) are known. A page tool with a list argument opens up
    to per_call ids in one step; otherwise per_item pages are opened one by one (6 for one product, 4 each for two
    or three, 3 each for up to six).
    Returns one log line per call."""
    log: list[str] = []
    if not brief:
        return log

    def fit(tool: str, raw: dict) -> dict:
        return _coerce_args(raw, _schema(ep.by_name[tool]))

    def run(tool: str, args: dict) -> Any:
        seen = ep.call(tool, args)
        log.append(" ".join((tool, json.dumps(args, ensure_ascii=False), "->", _compact_obs(seen))))
        return seen

    def room() -> bool:
        return ep.left >= END_RESERVE + 10

    pagers = [t for role in ("compare", "view") for t in roles.get(role) or [] if t in ep.by_name]
    plan = brief.get("probes")
    if not isinstance(plan, list):
        plan = []
    if per_item is None:
        per_item = 6 if len(plan) < 2 else 4 if len(plan) < 4 else 3
    shows_props = None  # the page tool found to show properties
    for spec in plan[:6]:
        if not isinstance(spec, dict) or ep.finished or not room():
            break
        tool = spec.get("tool")
        if tool not in ep.by_name:
            continue
        given = spec.get("args")
        sent = fit(tool, given if isinstance(given, dict) else {})
        seen = run(tool, sent)
        fresh = _fcb_unopened(seen, ep.viewed)
        if ep.probe_hits is None and _text_arg(ep.by_name[tool]):
            words = len(str(sent.get(_text_arg(ep.by_name[tool])) or "").split())
            ep.probe_hits = (words, _hit_count(seen, len(_fcg_rows(seen))))
        other = _fcb_keyword_tool(ep, roles, tool)
        if not fresh and other and spec.get("item") and room():
            # an empty first search often means a filter value worded otherwise: retry with the plain product name
            plain = {_text_arg(ep.by_name[other]): str(spec["item"])}
            fresh = _fcb_unopened(run(other, fit(other, plain)), ep.viewed)
        batch = fresh[:per_call]
        label = str(spec.get("item") or "")
        for lid in batch:
            ep.probe_of.setdefault(lid, label)
        for lid in fresh[:per_item]:
            if ep.finished or not room():
                break
            for page in [shows_props] if shows_props else pagers[:2]:
                vargs = _fcb_page_args(ep.by_name[page], lid, batch)
                if vargs is not None and _has_props(run(page, vargs)):
                    shows_props = page
                    break
            if shows_props and _fcb_array_arg(ep.by_name[shows_props]):
                break  # that one call already opened the whole batch
    return log


def _fcb_page_row(rec: dict) -> str:
    """One digest line: short text fields, then the recorded properties."""
    shown = [(k, v) for k, v in rec.items() if k[:1] != "_"]
    texts = ["%s=%s" % (k, v) for k, v in shown if isinstance(v, str) and 0 < len(v) <= 90]
    facts = ["%s=%.28s" % (pk, pv) for _, sub in shown if isinstance(sub, dict)
             for pk, pv in sub.items() if pv is not None and pv != ""]
    return "- %s || %s" % ("; ".join(texts)[:260], "; ".join(facts)[:700])


def _page_digest(ep, limit=14) -> str:
    """The latest opened pages, one line each."""
    latest = [*ep.viewed][-limit:]
    return "\n".join(map(_fcb_page_row, (ep.listings.get(lid) or {} for lid in latest)))


_FCB_UNDERSCORE = str.maketrans("_", " ")


def _prop_key(k: Any) -> str:
    """Property name in one plain form: lower case, underscores as spaces, single spaces."""
    words = str(k).lower().translate(_FCB_UNDERSCORE).split()
    return " ".join(words)


def _fcb_probed_count(ep: Episode, item: str) -> int:
    return sum(ep.probe_of.get(lid, "") == item for lid in ep.viewed)


def _fcb_empty(v: Any) -> bool:
    return v in (None, "", [], {})


def _fcb_page_facts(rec: dict):
    """(normalised property key, value) for every non-empty property a stored page records."""
    for field in rec.values():
        if isinstance(field, dict):
            for name, val in field.items():
                if not _fcb_empty(val):
                    yield _prop_key(name), val


def _fcb_first_values(ep: Episode) -> dict[str, dict[str, dict]]:
    """Product -> property key -> {listing id: the first value that listing's page gives}; every opened listing's
    product is present, even one whose page records nothing."""
    grid: dict[str, dict[str, dict]] = {}
    for lid in ep.viewed:
        keys = grid.setdefault(ep.probe_of.get(lid, ""), {})
        for key, val in _fcb_page_facts(ep.listings.get(lid) or {}):
            keys.setdefault(key, {}).setdefault(lid, val)
    return grid


def _fcb_prop_row(firsts: dict, opened: int) -> dict:
    """One table row from a property's first values per page (opened = pages opened for that product, at least 1)."""
    texts = [str(v) for v in firsts.values()]
    row = {"pages": len(firsts), "values": list(dict.fromkeys(t[:30] for t in texts)),
           "joined": sum(" / " in t for t in texts), "lids": set(firsts)}
    # most pages give a comma list: each list holds the values one listing offers, so its items are no spread
    # between listings
    row["listed"] = row["pages"] < 2 * sum(", " in t for t in texts)
    spread = 0.0 if row["listed"] else 0.5 * min(len(row["values"]), 3) / 3
    row["mixed"] = row["pages"] < 2 * row["joined"]
    row["rank"] = round(min(1.0, row["pages"] / opened) + spread, 2)
    return row


def _prop_table(ep) -> dict:
    """For each probed product, every property its opened pages record, as a row: pages (how many record it), values
    (a few of them), joined (pages whose value lists several choices), lids, mixed (most do), and rank (share of
    pages plus how much the values differ)."""
    table: dict[str, dict[str, dict]] = {}
    for item, keys in _fcb_first_values(ep).items():
        opened = max(1, _fcb_probed_count(ep, item))
        table[item] = {key: _fcb_prop_row(firsts, opened) for key, firsts in keys.items()}
    return table


def _fcb_shared(rows: dict, key: str) -> list[str]:
    """Other properties of the same product whose recorded values meet this one's, leaving out names sharing a word
    with it (one property recorded under several names): a field whose values belong to another property is read
    by the customer as being about both."""
    def plain(values):  # figures coincide across measures (two sizes of one listing) without naming one subject
        return {" ".join(str(v).lower().split()) for v in values or [] if not any(c.isdigit() for c in str(v))}
    row = rows.get(key) or {}
    mine = plain(row.get("values"))
    own = set(key.split())
    out = []
    for other, orow in rows.items():
        if other == key or orow.get("mixed") or own & set(other.split()):
            continue
        if mine & plain(orow.get("values")):
            out.append(other)
    return out


def _fcb_cell(key: str, row: dict, total: int, shared: list | None = None) -> str:
    tail = (" [several choices in one value: never ask]" if row["mixed"]
            else " (%s)" % ", ".join(row["values"][:4]))
    if row.get("listed") and not row["mixed"]:
        tail += " [each page lists several values]"
    if shared and not row["mixed"]:
        tail += " [values also under: %s]" % ", ".join(shared[:3])
    return "%s: rank %.2f, %d/%d pages%s" % (key, row["rank"], row["pages"], total, tail)


def _table_text(ep, table) -> str:
    """The property table as text: per product, its 24 best-ranked properties."""
    blocks = []
    for item, rows in table.items():
        total = _fcb_probed_count(ep, item)
        best = sorted(rows, key=lambda k: (-rows[k]["rank"], -rows[k]["pages"]))[:24]
        body = "\n  ".join(_fcb_cell(k, rows[k], total, _fcb_shared(rows, k)) for k in best)
        blocks.append("%s:\n  %s" % (item or "pages opened", body))
    return "\n".join(blocks)

# ---------------------------------------------------------------- shared small helpers

_FCC_ASK_TEXT = "Which {prop} would you like the {item} to have?"
_FCC_STRONG = (HEAVY, HEAVY_SPARE)
_FCC_WORD = re.compile(r"[a-z0-9]+")


def _fcc_json(obj):
    return json.dumps(obj, ensure_ascii=False)


def _fcc_qa_lines(qa):
    return "\n".join("Q: %s\nA: %s" % pair for pair in qa)


def _fcc_blocks(blocks):
    """Labelled sections of a model request, one blank line apart."""
    return "\n\n".join("%s:\n%s" % pair for pair in blocks)


def _fcc_entries(brief):
    """The brief's requested products that are written as objects."""
    return [entry for entry in (brief or {}).get("items") or [] if isinstance(entry, dict)]


def _fcc_lone_string(obs, ok):
    """The single string among a dict result's own fields that ok() accepts; None when there is not exactly one."""
    if not isinstance(obs, dict):
        return None
    hits = [v for v in obs.values() if isinstance(v, str) and ok(v)]
    return hits[0] if len(hits) == 1 else None


def _fcc_run(ep, name, args):
    """One world call with arguments shaped to the tool's schema."""
    return ep.call(name, _coerce_args(args, _schema(ep.by_name[name])))


def _words(text):
    return {w for w in _FCC_WORD.findall(str(text).lower()) if len(w) > 2}


# ---------------------------------------------------------------- asking: pure choices

def _fcc_question_user(ep, brief, request, stated, left, focus):
    """The question model's input, built from a snapshot of what is known so far."""
    hints = [{"item": ask.get("item"), "property": ask.get("property")}
             for ask in brief.get("questions") or [] if isinstance(ask, dict) and ask.get("property")]
    settled = {}
    for name, props in stated.items():
        settled[name] = sorted(props)
    blocks = [
        ("Messages still available after this one", str(left - 1)),
        ("Request, as the shopper wrote it", request),
        ("Brief", _brief_text(brief)),
        ("Properties the request hints at (shopper's wording)", _fcc_json(hints)),
        ("Settled per product by the request (do not ask these for that product)", _fcc_json(settled)),
        ("Questions sent so far, each with its reply", _fcc_qa_lines(ep.qa) or "(nothing asked yet)"),
        ("PROPERTY TABLE (built from the opened pages)", _table_text(ep, _prop_table(ep))),
        ("Shop pages looked at", _page_digest(ep)),
    ]
    if focus:
        blocks.append(("PRODUCT TO ASK ABOUT NOW", focus))
    return _fcc_blocks(blocks)


def _fcc_shape_question(got, brief, focus):
    """Turn the model's pick into a question record; None when it named no property."""
    raw = (got or {}).get("property") or ""
    prop = " ".join(str(raw).replace("_", " ").split())
    if not prop:
        return None
    named = [str(entry["what"]) for entry in brief.get("items") or []
             if isinstance(entry, dict) and entry.get("what")]
    label = got.get("item") or focus or (named[0] if named else "item")
    item = " ".join(str(label).split())
    # every question uses one fixed form: the shop's own property name and the product it is for
    got["property"], got["item"] = prop, item
    got["text"] = _FCC_ASK_TEXT.format(prop=prop, item=item)
    return got


def _fcc_examples(table, item, prop, most=2):
    """Up to `most` short values the opened pages give for this property (the asked product's pages first): a
    question that shows the property's own values is read as being about that property."""
    key = _prop_key(prop)
    home = _fcc_match(item, list(table))
    rows = [table[home]] if home in table else []
    rows += [table[p] for p in table if p != home]
    picked: list[str] = []
    for rows_of in rows:
        row = rows_of.get(key)
        # values that another property also records would point the question at two properties
        if not row or row["mixed"] or _fcb_shared(rows_of, key):
            continue
        for val in row["values"]:
            text = " ".join(str(val).split())
            # very short words (flags, letters) say little about the property; sizes and counts keep their digits
            sized = len(text) >= 4 or _FCA_DIGIT.search(text) is not None
            plain = sized and len(text) <= 24 and not _code_like(text) and " / " not in text
            # a value whose digits are all zero is a blank left by the seller, not a value to show
            plain = plain and not (_FCA_DIGIT.search(text) and not re.sub(r"\D", "", text).strip("0"))
            if plain and text.lower() not in {p.lower() for p in picked}:
                picked.append(text)
        if picked:
            break
    return picked[:most]


def _fcc_with_examples(text, values):
    """The question with its example values added inside it, or as it was."""
    if not values or not text.endswith("?"):
        return text
    return "%s, for example %s?" % (text[:-1], " or ".join(values))


def _fcc_question_budget(brief, steps_left):
    """How many questions may be sent: the message limit less reserved messages, then the step margin."""
    cap = _message_limit(brief) - _reserved_messages(brief)
    if brief.get("question_cost"):
        listed = brief.get("questions")
        listed = listed if isinstance(listed, list) else []
        cap = min(cap, sum(1 for a in listed if isinstance(a, dict) and a.get("hinted") and a.get("property")))
    return min(cap, steps_left - END_RESERVE - 6)


class _FccAsking:
    """Bookkeeping of the question round: per product how often asked, whose wish came out, what was asked."""

    def __init__(self, stated):
        self.stated = stated
        self.items = list(stated)
        self.count = dict.fromkeys(self.items, 0)
        self.told = set()
        self.asked = set()

    def focus(self):
        """With several products: the least-asked one, preferring products whose wish has not come out yet."""
        if len(self.items) < 2:
            return None
        order = sorted(range(len(self.items)),
                       key=lambda n: (self.items[n] in self.told, self.count[self.items[n]], n))
        return self.items[order[0]]

    def screen(self, q, focus, table):
        """Whether a proposed question may go out: not a repeat, not settled by the request, not a mixed field."""
        fallback = self.items[0] if self.items else ""
        item = _fcc_match(str(q.get("item", "")), self.items) or focus or fallback
        key = (item.lower(), str(q.get("property", "")).lower())
        text = str(q.get("text")).strip()
        mixed = any(rows.get(_prop_key(key[1]), {}).get("mixed") for rows in table.values())
        ok = bool(text) and key not in self.asked and not mixed \
            and not _fcc_settled(key[1], self.stated.get(item, set()))
        return {"text": text, "item": item, "key": key, "send": ok}

    def skip(self, item, prop):
        if prop:
            self.stated.setdefault(item, set()).add(prop)

    def record(self, item, prop, reply):
        if item not in self.count:
            return
        self.count[item] += 1
        # a reply that carries a wish repeats the property's words ("where X is concerned, only Y")
        need = _words(prop)
        if need and need <= _words(reply):
            self.told.add(item)


# ---------------------------------------------------------------- asking: effects

def _fcc_table_pick(table, item, stated, asked, request):
    """Code's own pick when the question model gives none: the best-ranked property of the product's table row that
    is not mixed, has at least two values, is not settled or asked already (same words either way), and none of
    whose values the request already writes. None when no such property is left."""
    home = _fcc_match(item, list(table)) if item else None
    rows = table.get(home) if home in table else (next(iter(table.values())) if len(table) == 1 else None)
    if not rows:
        return None
    said = " ".join(str(request or "").lower().split())
    done = [_words(prop) for name, prop in asked if name == str(item).lower()]
    for key in sorted(rows, key=lambda k: (-rows[k]["rank"], -rows[k]["pages"], k)):
        row, need = rows[key], _words(key)
        if row["mixed"] or len(row["values"]) < 2 or not need or _fcc_settled(key, stated):
            continue
        if any(need <= old or old <= need for old in done if old):
            continue
        if any(len(v) >= 4 and " ".join(v.lower().split()) in said for v in row["values"]):
            continue
        return {"item": item, "property": key, "why": "table pick"}
    return None


def _fcc_next_question(ep, brief, request, stated, left, focus=None):
    user = _fcc_question_user(ep, brief, request, stated, left, focus)
    return _fcc_shape_question(_ask_json([LIGHT, HEAVY], QUESTION_SYSTEM, user, max_tokens=2500), brief, focus)


def _ask(ep, brief, roles, request):
    """Send questions one at a time, choosing each after the replies so far."""
    senders = roles.get("message") or []
    if not senders or not brief:
        return
    sender = senders[0]
    field = _text_arg(ep.by_name[sender])
    if not field:
        return
    limit = _fcc_question_budget(brief, ep.left)
    book = _FccAsking(_fcc_stated(brief))
    heard = []
    for _attempt in range(limit + 3):
        if ep.finished or len(ep.qa) >= limit:
            break
        focus = book.focus()
        q = _fcc_next_question(ep, brief, request, book.stated, limit - len(ep.qa), focus)
        if not q:
            # the question model gave nothing usable: a message left unused is lost, so code picks from the table
            item = focus or (book.items[0] if book.items else "")
            pick = _fcc_table_pick(_prop_table(ep), item, book.stated.get(item, set()), book.asked, request)
            q = _fcc_shape_question(pick, brief, focus) if pick else None
        if not q:
            break
        table = _prop_table(ep)
        check = book.screen(q, focus, table)
        item, prop = check["item"], check["key"][1]
        if not check["send"]:
            book.skip(item, prop)
            continue
        book.asked.add(check["key"])
        text = _fcc_with_examples(check["text"], _fcc_examples(table, item, prop))
        reply = _reply_text(_fcc_run(ep, sender, {field: text}))
        ep.qa.append((text, reply))
        heard.append((item, prop, reply))
        book.record(item, prop, reply)
        if _words(prop) and _words(prop) <= _words(reply):
            ep.told.append((item, prop, reply))
    _fcg_told_values(ep, heard)


def _fcc_stated(brief):
    """Requested product (as the brief names it) -> properties the request already fixes for it."""
    found = {}
    goods = (brief or {}).get("items")
    for entry in goods if isinstance(goods, list) else []:
        if not (isinstance(entry, dict) and entry.get("what")):
            continue
        musts = entry.get("musts")
        heads = {str(m).split(":", 1)[0].strip().lower() for m in (musts if isinstance(musts, list) else [])}
        found.setdefault(" ".join(str(entry["what"]).split()), set()).update(heads - {""})
    return found


def _fcc_match(name, items):
    """The requested product a name refers to: the first one sharing the most words with it, if any."""
    want = _words(name)
    best, most = None, 0
    for candidate in items:
        shared = len(want & _words(candidate))
        if shared > most:
            best, most = candidate, shared
    return best


def _fcc_settled(prop, stated):
    """The request already settles this property: its words all appear in one stated property."""
    need = _words(prop)
    return bool(need) and any(need <= _words(s) for s in stated)


def _reply_text(obs):
    if isinstance(obs, str):
        return obs
    only = _fcc_lone_string(obs, lambda v: True)
    return _fcc_json(obs) if only is None else only


# ---------------------------------------------------------------- strong-model review and decision

def _fcc_told_fit(ep, brief, lid):
    """A page that shows, under the asked property, a value the shopper's reply named for its product."""
    facts = _fce_facts(ep.listings.get(lid) or {})
    return any(_fce_fits_told(facts, p, r) for p, r in _fcf_told_for(ep, brief, lid))


def _fcc_listing_texts(ep, carted, brief=None):
    """Opened pages (cart lines first, then pages showing a value the shopper told, then the latest others, ten in
    all, kept in the order they were opened) and the last result rows."""
    mine, others, glimpses = [], [], []
    for lid, rec in ep.listings.items():
        shown = {k: v for k, v in rec.items() if not k.startswith("_")}
        if not rec.get("_full"):
            glimpses.append(_fcc_json(_shrink(shown, 60)))
            continue
        tools_used = ep.opened_by.get(lid)
        if tools_used:
            shown["opened_with"] = sorted(tools_used)
        if lid in carted:
            mine.append(_fcc_json(_shrink(shown, 180)))
        else:
            others.append((lid, _fcc_json(_shrink(shown, 180))))
    keep = min(max(0, 10 - len(carted)), len(others))
    # a told need is proven only by a page that shows it: such pages are never pushed out by later look-ups
    fitting = [n for n, (lid, _text) in enumerate(others) if ep.told and _fcc_told_fit(ep, brief, lid)]
    chosen = set(fitting[len(fitting) - min(keep, len(fitting)):])
    for n in reversed(range(len(others))):
        if len(chosen) >= keep:
            break
        chosen.add(n)
    return mine + [others[n][1] for n in sorted(chosen)], glimpses[-15:]


def _fcc_dossier(ep, request, brief):
    """Everything the strong model needs to judge the cart, as labelled blocks."""
    pages, glimpses = _fcc_listing_texts(ep, set(_cart_ids(ep)), brief)
    blocks = [("REQUEST FROM THE SHOPPER", request)]
    if brief:
        blocks.append(("BRIEF", _brief_text(brief)))
    if ep.qa:
        blocks.append(("QUESTIONS PUT TO THE SHOPPER, WITH ANSWERS", _fcc_qa_lines(ep.qa)))
    if ep.notes:
        blocks.append(("LATER SHOPPER WORDS FOUND IN TOOL RESULTS (oldest first; they take precedence over the "
                       "request)", "\n".join(ep.notes)))
    if ep.ref_notes:
        blocks.append(("LISTINGS THAT CONDITIONS POINT TO", "\n".join(ep.ref_notes)))
    blocks.append(("CART AS LAST VIEWED", _fcc_json(ep.cart_obs)))
    if len(ep.page_tools) > 1:
        blocks.append(("THIS SHOP'S PAGE TOOLS (each can show other parts of a listing)",
                       ", ".join(sorted(ep.page_tools))))
    blocks += [("OPENED LISTINGS (full pages)", "\n".join(pages)),
               ("LISTINGS ONLY SEEN IN RESULT LISTS", "\n".join(glimpses)),
               ("REMAINING STEPS", str(ep.left))]
    return _fcc_blocks(blocks)


def _review(ep, request, brief, closing):
    user = _fcc_dossier(ep, request, brief) + "\n\nCLOSE THE ASSISTANT IS ABOUT TO MAKE: " + str(closing)
    return _ask_json(list(_FCC_STRONG), REVIEW_SYSTEM, user, max_tokens=6000)


def _table_fix(verdict, cart):
    """Lowest-price requests: per product, the cheapest row judged to meet everything should be the carted one."""
    table = verdict.get("table")
    if not verdict.get("cheapest_wins") or not isinstance(table, list):
        return None
    rows = [r for r in table if isinstance(r, dict)]
    product_of, fitting = {}, {}
    for r in rows:
        lid = str(r.get("id") or "")
        product = str(r.get("item") or "").strip().lower()
        cost = _number(r.get("price"))
        product_of.setdefault(lid, product)
        if lid and cost is not None and r.get("meets_all") is True and r.get("in_stock") is not False:
            fitting.setdefault(product, []).append((lid, cost))
    # first-seen wins among equal prices
    cheapest = {product: min(group, key=lambda pair: pair[1]) for product, group in fitting.items()}
    moves = []
    for lid in cart:
        alt = cheapest.get(product_of.get(lid))
        if not alt or alt[0] == lid:
            continue
        own = next((_number(r.get("price")) for r in rows if str(r.get("id")) == lid), None)
        if own is None or alt[1] < own:
            moves.append("swap %s for %s, which meets every requirement at %s" % (lid, alt[0], format(alt[1], "g")))
    covered = {product_of.get(lid) for lid in cart}
    for product, (alt_id, _cost) in cheapest.items():
        if cart and product not in covered and alt_id not in cart:
            moves.append("no cart line covers %s yet: put %s in" % (product, alt_id))
    return "; ".join(moves) or None


_FCC_DECIDE_SYSTEM = """Output contract. Reply with a single JSON object, no prose around it:
{"checks": what you verified, briefly,
 "action": "order" or "decline",
 "lines": [{"id": listing id, "qty": how many}],
 "reason": one sentence,
 "open_first": at most 4 ids from result lists whose pages nobody opened yet and which could do better than your
   pick (a lower price, for instance); they are inspected before buying,
 "search_first": a few title words to search with, when another seller may offer the identical product}

Situation. You settle the purchase for an assistant in a simulated web store; it has almost no steps left. Your
input: what the shopper asked for, a brief, the answers the shopper gave, and all listings met so far (complete
pages where opened). Pick nothing outside those listings.

1. Whose wishes count
 a. An answer that names a product attaches its need to that product, whichever product the question was about.
 b. Lines the shopper wrote that showed up inside tool output (a fresh budget, or a requirement altered, eased or
    withdrawn) are authentic and outrank the original request; an eased or withdrawn requirement is void.

2. Testing a listing
 a. Page facts must satisfy all stated requirements plus all needs revealed in answers, through the attribute of
    exactly that property carrying the desired value; some neighbouring property proves nothing.
 b. Product kind follows the category field of the listing itself, never its title, and has to be the kind the
    shopper names.
 c. Availability counts when the shopper cares about it, the total may not go over the budget, and the shopper's
    objective ranks whatever passes.
 d. Identity of values: genuine synonyms or converted units match; anything merely compatible, more or less
    intense, or akin does not, nor do two separate words the store employs as different values of a single property.
 e. A page contradicting a requirement fails it. Lacking an attribute, description or title rule; if both
    stay silent, the listing fails that requirement when another opened page of its category records the
    property as an attribute (the store tracks it), and stays eligible only when none does (an id whose page some
    page tool has not shown yet goes in "open_first").
 f. Unless the store states a unit, prices use the same unit as the budget.

3. Choosing between order and decline
 a. Declining is allowed only when each listing met clearly breaks some requirement judging by that listing's
    page (value differs, category differs, too costly, unavailable where that counts, or silent on a property
    its category's other opened pages record, as in 2e).
 b. A page that omits a property no opened page of its category records never justifies a decline when the rest
    fits; buy the strongest of those listings.
 c. When the store rejected a close (say a listing ran out), go for the strongest listing still on
    offer under the shopper's objective. Identical titles from other sellers are common: rank such twins from result
    lists ahead, or put a lookup for them into "search_first"."""


def _decide(ep, request, brief, why=""):
    parts = [_fcc_dossier(ep, request, brief)]
    if why:
        parts.append("REASON FOR DECIDING NOW: %s" % (why,))
    return _ask_json(list(_FCC_STRONG), _FCC_DECIDE_SYSTEM, "\n\n".join(parts), max_tokens=4000)


# ---------------------------------------------------------------- cart: pure reading of decisions and views

def _fcc_order_lines(pick, count):
    """A decision's lines as listing id -> count(qty), later lines for the same id winning."""
    out = {}
    for ln in pick.get("lines") or []:
        if isinstance(ln, dict) and ln.get("id"):
            out[str(ln["id"])] = count(ln.get("qty"))
    return out


def _fcc_action(pick):
    return str(pick.get("action", "")).lower()


def _cart_steps(ep, pick):
    """Spell a decision out as cart operations against the latest cart view."""
    if _fcc_action(pick) == "decline":
        return "To do: end the episode without buying anything."
    target = _fcc_order_lines(pick, lambda q: q or 1)
    if not target:
        return ""
    ops = ["take %s out" % lid for lid in _cart_ids(ep) if lid not in target]
    ops += ["set %s to quantity %s" % pair for pair in target.items()]
    return ("To do, in this order: " + "; ".join(ops) + ". Afterwards the cart holds these lines and nothing "
            "else; view it, then close the episode with the purchase.")


def _cart_ids(ep):
    """Listing ids named anywhere in the latest cart view, in reading order."""
    found = []
    stack = [ep.cart_obs]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            stack += reversed(list(node.values()))
        elif isinstance(node, list):
            stack += reversed(node)
        elif not isinstance(node, bool) and isinstance(node, (str, int)) and str(node) in ep.listings:
            found.append(str(node))
    return list(dict.fromkeys(found))


def _fcc_recheck_plan(ep, roles):
    """Cart lines to look at again and the page calls for it: the stock page first, then unused page tools."""
    lids = _cart_ids(ep)[:4]
    pages = [t for t in roles.get("view", []) if t in ep.by_name]
    first = (ep.bool_pages or pages or [None])[0]
    if not lids or not first:
        return [], []
    calls = []
    for lid in lids:
        used = ep.opened_by.get(lid, set())
        calls.append((first, lid))
        calls += [(t, lid) for t in pages if t != first and t not in used]
    return lids, calls


def _fcc_count_field(props, zero=False):
    """First numeric argument; with zero=True the first one that also accepts 0."""
    for name, spec in props.items():
        numeric = isinstance(spec, dict) and spec.get("type") in ("integer", "number")
        if numeric and (not zero or spec.get("minimum") == 0):
            return name
    return None


def _fcc_cart_tools(roles):
    return roles.get("cart_set", []) + roles.get("cart_change", [])


def _fcc_cart_writer(ep, roles):
    """The cart tool _place uses and how its arguments are named."""
    name = next((t for t in _fcc_cart_tools(roles) if t in ep.by_name), None)
    if not name:
        return None
    props = _schema(ep.by_name[name]).get("properties") or {}
    count = _fcc_count_field(props)
    field = _text_arg(ep.by_name[name])
    if not field:
        return None
    return {"name": name, "field": field, "count": count,
            "zero": bool(count) and props[count].get("minimum") == 0}


def _fcc_purchase_call(ep, roles, brief):
    """The closing call that buys: a checkout tool, else a finish tool in its order mode."""
    if ep.finished or ep.left <= 0:
        return None
    payer = next((t for t in roles.get("checkout", []) if t in ep.by_name), None)
    if payer:
        return payer, {}
    modes = _finish_modes(brief)
    closer = next((t for t in roles.get("finish", []) if t in ep.by_name), None)
    if closer:
        return closer, _mode_args(ep, closer, modes.get(closer, {}).get("order"))
    return None


def _pick_lines(pick):
    """An order decision as listing -> count; None for anything else."""
    if _fcc_action(pick) == "order":
        return _fcc_order_lines(pick, lambda q: _number(q) or 1.0) or None
    return None


def _fcc_norm(text):
    return " ".join(str(text or "").lower().split())


def _fcc_cheaper(row, held):
    cost = _number(row.get("price"))
    return cost is not None and cost < (_number(held.get("price")) or float("inf"))


def _fcc_review_rows(ep, verdict):
    """Rows the review judged to meet everything, in stock, naming a listing seen."""
    keep = []
    for r in verdict.get("table") or []:
        if not isinstance(r, dict) or r.get("meets_all") is not True or r.get("in_stock") is False:
            continue
        if r.get("id") and str(r["id"]) in ep.listings:
            keep.append(r)
    return keep


def _verdict_lines(ep, brief, verdict):
    """A review's order as listing -> count: per product the first in-stock row judged to meet everything (the
    cheapest one when the shopper wants the lowest price); None when the review backs a decline."""
    if str(verdict.get("close_as", "")).lower() == "decline":
        return None
    rows = _fcc_review_rows(ep, verdict)
    if not rows:
        return None
    wanted = {_fcc_norm(entry.get("what")): _number(entry.get("qty")) or 1.0 for entry in _fcc_entries(brief)}
    groups = {}
    for r in rows:
        groups.setdefault(_fcc_norm(r.get("item")), []).append(r)
    lone = next(iter(wanted.values())) if len(wanted) == 1 else 1.0
    lines = {}
    for product, group in groups.items():
        held = group[0]
        if verdict.get("cheapest_wins"):
            for r in group[1:]:
                if _fcc_cheaper(r, held):
                    held = r
        match = _fcc_match(product, list(wanted)) if wanted else None
        lines[str(held["id"])] = wanted[match] if match else lone
    return lines


# ---------------------------------------------------------------- cart: effects

def _refresh_cart(ep, roles, view_cart):
    """After the last cart change, view the cart again and re-open each line's page (stock and price can move)."""
    if view_cart and (ep.cart_obs is None or ep.cart_dirty):
        ep.cart_obs = _fcc_run(ep, view_cart, {})
        ep.cart_dirty = False
    lids, calls = _fcc_recheck_plan(ep, roles)
    for name, lid in calls:
        if ep.finished or ep.left <= 3:
            break
        field = _text_arg(ep.by_name[name])
        if field:
            _fcc_run(ep, name, {field: lid})
    return lids


def _removal_hint(ep, roles):
    """When no tool removes lines: a cart tool whose count can be set to 0, described for the model."""
    for name in _fcc_cart_tools(roles):
        tool = ep.by_name.get(name)
        if not tool:
            continue
        count = _fcc_count_field(_schema(tool).get("properties") or {}, zero=True)
        field = _text_arg(tool)
        if field and count:
            return ("to drop a line, call %s with %s set to the listing id and %s set to 0 (%s sets how many of a "
                    "listing the cart holds)." % (name, field, count, name))
    return ""


def _fcc_held_lines(ep, lines=None):
    """The basket about to be bought: the given lines, else this agent's own cart calls, else the cart view."""
    held = {lid: n for lid, n in (lines or {}).items() if n}
    if not held and ep.cart_obs is not None and not ep.cart_dirty:
        held = {lid: float(n) for lid, n in (_cart_counts(ep) or {}).items() if n > 0}  # the current cart view
    held = held or {lid: n for lid, n in ep.cart_set.items() if n > 0}
    return held or dict.fromkeys(_cart_ids(ep), 1.0)


_FCF_MOMENTS = ("before_order", "before_decline", "before_change", "after_notice", "any")


def _procedures(brief, moment):
    """The brief's procedures that need a message at this moment; a message procedure without a known moment
    counts as one before the order."""
    found = []
    for entry in (brief or {}).get("procedures") or []:
        if isinstance(entry, dict) and entry.get("needs_message"):
            when = entry.get("when") if entry.get("when") in _FCF_MOMENTS else "before_order"
            if when == moment:
                found.append(entry)
    return found


def _fcc_basket_text(ep, held, ask=False):
    """The final-basket message: each line with its count and unit price, then the total when every price is
    known; ask=True ends it with a request for the go-ahead."""
    parts, total = [], 0.0
    for lid, n in held.items():
        price = _listing_price(ep.listings.get(lid) or {})
        count = format(float(n), "g")
        if price is None:
            total = None
            parts.append("%s x%s" % (lid, count))
            continue
        parts.append("%s x%s at %s each" % (lid, count, format(price, "g")))
        if total is not None:
            total += price * float(n)
    text = "Before I check out, here is the final basket: " + "; ".join(parts)
    if total is not None:
        text += "; total %s" % format(total, "g")
    if ask:
        return text + ". Shall I place the order with exactly these lines?"
    return text + ". I will place the order with exactly these lines."


def _fcf_sender(ep, roles):
    """The message tool and its text argument, or (None, None)."""
    sender = next((t for t in roles.get("message") or [] if t in ep.by_name), None)
    return (sender, _text_arg(ep.by_name[sender])) if sender else (None, None)


def _fcc_statement_due(ep, roles, brief, room=2):
    """A procedure wants a message before the purchase, one is still allowed, the cart holds something, no message
    went out since its last change, and the steps cover the message plus the purchase."""
    if not (_procedures(brief, "before_order") and ep.carted and ep.stated_rev != ep.cart_rev):
        return False
    if len(ep.qa) >= _message_limit(brief) or not _fcd_room(ep, room):
        return False
    return _fcf_sender(ep, roles)[1] is not None


def _state_basket(ep, roles, brief, lines=None):
    """Send the final-basket message when it is due and keep the shopper's reply; returns True when it went out."""
    if not _fcc_statement_due(ep, roles, brief):
        return False
    held = _fcc_held_lines(ep, lines)
    sender, field = _fcf_sender(ep, roles)
    if not held:
        return False
    ask = any(p.get("content") == "question" for p in _procedures(brief, "before_order"))
    text = _fcc_basket_text(ep, held, ask)
    reply = _reply_text(_fcc_run(ep, sender, {field: text}))
    ep.qa.append((text, reply))
    ep.stated_rev = ep.cart_rev
    ep.statement_reply = (text, reply)
    return True


def _fcf_cart_locked(ep, roles, brief):
    """The basket report a procedure wants before the order went out after the cart's last change, and no message
    is left to report another basket: any cart edit now would leave the purchase unreported."""
    return (bool(_procedures(brief, "before_order")) and ep.carted and ep.stated_rev == ep.cart_rev
            and len(ep.qa) >= _message_limit(brief))


def _procedure_message(ep, roles, brief, moment, text):
    """Send the message a procedure asks for at this moment (once per moment) when a message and two steps are
    left; returns True when it went out."""
    if moment in ep.fired or not _procedures(brief, moment):
        return False
    if len(ep.qa) >= _message_limit(brief) or not _fcd_room(ep, 2):
        return False
    sender, field = _fcf_sender(ep, roles)
    if field is None:
        return False
    reply = _reply_text(_fcc_run(ep, sender, {field: text}))
    ep.qa.append((text, reply))
    ep.fired.add(moment)
    return True


REPLY_SYSTEM = """Answer only with JSON of the form {"kind": "yes" | "ack" | "change" | "no"}: how the customer
answered the assistant's report of the basket it is about to buy. "yes" approves it; "ack" only acknowledges it or
says nothing about it; "change" asks for something different (another budget, product, quantity or condition);
"no" refuses the order."""


def _reply_kind(text, reply):
    got = _ask_json([LIGHT, LIGHT_SPARE], REPLY_SYSTEM, "ASSISTANT: %s\nCUSTOMER: %s" % (text, reply), 200)
    kind = _fcd_low((got or {}).get("kind", "")) if isinstance(got, dict) else ""
    return kind if kind in ("yes", "ack", "change", "no") else "ack"


def _hear_statement(ep, roles, brief):
    """Read the reply to the basket report just sent: 'order' to buy (the reported basket, or a new one the strong
    model chose, carted and reported again), or 'decline' when the shopper refused and the strong model agrees.
    A change with no message left keeps the reported basket: a cart edited after the report goes unconfirmed."""
    said = ep.statement_reply
    ep.statement_reply = None
    if not said or not str(said[1]).strip():
        return "order"
    if _reply_kind(*said) not in ("change", "no") or not _fcd_room(ep, 4) or _spent() >= _caps()[2]:
        return "order"
    pick = _decide(ep, ep.policy.get("query") or "", brief, "the shopper answered the basket report with: " +
                   str(said[1])[:300])
    if not pick:
        return "order"
    if _fcc_action(pick) == "decline":
        return "decline"
    lines = _pick_lines(pick)
    if not lines or lines == _fcc_held_lines(ep) or len(ep.qa) >= _message_limit(brief):
        return "order"
    if _commit_lines(ep, roles, brief, lines):
        _state_basket(ep, roles, brief, lines)
    return "order"


def _decline_call(ep, roles, brief):
    """The closing call that ends without buying: a decline tool, else a finish tool in its decline mode."""
    modes = _finish_modes(brief)
    for tool in roles.get("decline", []) + roles.get("finish", []):
        if tool in ep.by_name:
            raw = {} if tool in roles.get("decline", []) else _mode_args(ep, tool, modes.get(tool, {}).get("decline"))
            schema = _schema(ep.by_name[tool])
            return tool, _coerce_args(_fcd_with_required(raw, schema), schema)
    return None


# ---------------------------------------------------------------- cart: commits in the shop's own semantics

def _fcf_covers(ep, spec):
    """Pure-ish: the spec fills every argument its tool requires (else a call would be refused or misread)."""
    required = set(_schema(ep.by_name[spec["name"]]).get("required") or [])
    given = {spec["field"], spec["count"]} | set(spec["add_args"])
    return required <= given and (spec["remove_args"] is None or required <= given | set(spec["remove_args"]))


def _cart_spec(ep, roles, brief):
    """How to write the cart: the brief's cart_write when it names a real tool and id argument, else the first
    cart tool read by its schema (set semantics, 0 removes when the count allows it); None when the spec would
    leave a required argument empty."""
    given = (brief or {}).get("cart_write")
    spec = None
    if isinstance(given, dict) and given.get("tool") in ep.by_name:
        props = _schema(ep.by_name[given["tool"]]).get("properties") or {}
        if given.get("id_arg") in props and (props[given["id_arg"]] or {}).get("type") in (None, "string"):
            extra = given.get("add_args")
            out = given.get("remove_args")
            spec = {"name": given["tool"], "field": given["id_arg"],
                    "count": given.get("count_arg") if given.get("count_arg") in props else None,
                    "adds": given.get("count_means") == "add",
                    "add_args": dict(extra) if isinstance(extra, dict) else {},
                    "remove_args": dict(out) if isinstance(out, dict) else None, "zero": True}
    if spec is None:
        old = _fcc_cart_writer(ep, roles)
        if old is not None:
            spec = {"name": old["name"], "field": old["field"], "count": old["count"], "adds": False,
                    "add_args": {}, "remove_args": None, "zero": old["zero"]}
    return spec if spec and _fcf_covers(ep, spec) else None


def _fcf_take(spec, lid, k):
    """Pure: calls that take k pieces of lid out (none when the shop offers no way)."""
    field, count = spec["field"], spec["count"]
    if spec["remove_args"] is None:
        zero = count and not spec["adds"] and spec["zero"]
        return [dict(spec["add_args"], **{field: lid, count: 0})] if zero else []
    if count:
        return [dict(spec["remove_args"], **{field: lid, count: k})]
    return [dict(spec["remove_args"], **{field: lid}) for _ in range(min(k, 6))]


def _fcf_put(spec, lid, k, total):
    """Pure: calls that bring lid from total-k to total pieces."""
    field, count = spec["field"], spec["count"]
    if count:
        return [dict(spec["add_args"], **{field: lid, count: k if spec["adds"] else total})]
    return [dict(spec["add_args"], **{field: lid}) for _ in range(min(k, 6))]


def _cart_moves(spec, have, target):
    """Pure: the cart calls that turn the held counts into the target counts; None when a line must come out and
    the shop offers no way to take it out."""
    moves = []
    for lid, n in have.items():
        if lid not in target and n > 0:
            taken = _fcf_take(spec, lid, int(n))
            if not taken:
                return None
            moves += taken
    for lid, n in target.items():
        held, want = int(have.get(lid, 0)), int(n)
        if held < want:
            moves += _fcf_put(spec, lid, want - held, want)
        elif held > want:
            if spec["count"] and not spec["adds"]:
                moves.append(dict(spec["add_args"], **{spec["field"]: lid, spec["count"]: want}))
            else:
                taken = _fcf_take(spec, lid, held - want)
                if not taken:
                    return None
                moves += taken
    return moves


def _fcf_line_dicts(node, ids, sink):
    """Cart lines: the innermost records that name a known listing id among their text values."""
    if isinstance(node, list):
        for item in node:
            _fcf_line_dicts(item, ids, sink)
    elif isinstance(node, dict):
        named = [v for v in node.values() if isinstance(v, str) and v in ids]
        nested = [v for v in node.values() if isinstance(v, (dict, list))]
        if len(named) == 1 and not any(_fcf_names_ids(v, ids) for v in nested):
            sink.append((named[0], node))
        else:
            for v in nested:
                _fcf_line_dicts(v, ids, sink)


def _fcf_names_ids(node, ids):
    found = []
    _fcf_line_dicts(node, ids, found)
    return bool(found)


def _cart_counts(ep):
    """Listing id -> quantity from the latest cart view: in each line, the one whole-number field below the
    listing's price; None when some line has no such single field."""
    found = []
    _fcf_line_dicts(ep.cart_obs, ep.listings, found)
    counts = {}
    for lid, rec in found:
        price = _listing_price(ep.listings.get(lid) or {})
        nums = [v for v in rec.values()
                if isinstance(v, int) and not isinstance(v, bool) and (price is None or v < price)]
        if len(nums) != 1:
            return None
        counts[lid] = counts.get(lid, 0) + nums[0]
    return counts


def _fcf_have(ep, view):
    """What the cart holds now: the counts of a fresh cart view when the shop has one, else this agent's own
    record."""
    if not ep.carted:
        return {}
    if view and (ep.cart_dirty or ep.cart_obs is None) and _fcd_room(ep, 3):
        ep.cart_obs, ep.cart_dirty = _fcc_run(ep, view, {}), False
    own = {lid: n for lid, n in ep.cart_set.items() if n > 0}
    counts = _cart_counts(ep) if view and not ep.cart_dirty else None
    if counts is not None and (counts or not own):
        return {lid: n for lid, n in counts.items() if n > 0}
    return own


def _commit_lines(ep, roles, brief, lines):
    """Make the cart hold exactly these lines in the shop's own cart semantics, then check the cart view and
    correct once; True when the cart matches. A procedure wanting word before a filled basket changes is
    served first."""
    spec = _cart_spec(ep, roles, brief)
    view = next((t for t in roles.get("cart_view") or [] if t in ep.by_name), None)
    if spec is None:
        return False
    target = {lid: float(n) for lid, n in lines.items() if n}
    for _round in range(2):
        have = _fcf_have(ep, view)
        moves = _cart_moves(spec, have, target)
        if moves == []:
            return True  # the cart already matches: no step is needed, whatever is left
        if moves is None or not _fcd_room(ep, len(moves) + 3):
            return False
        if _fcf_cart_locked(ep, roles, brief):
            return False  # the reported basket stands: no message is left to report a changed one
        if any(lid not in target for lid in have):
            _procedure_message(ep, roles, brief, "before_change", "I am changing the basket: taking out %s and "
                               "putting in %s." % (", ".join(x for x in have if x not in target),
                                                   ", ".join(target)))
        for args in moves:
            _fcc_run(ep, spec["name"], args)
            ep.cart_rev += 1
        ep.carted, ep.cart_dirty = True, True
        ep.cart_set = dict(dict.fromkeys(ep.cart_set, 0.0), **target)
        if not view:
            return True
    return False


def _place(ep, roles, brief, lines):
    """Make the cart hold exactly these lines, then finish the purchase; None when the cart could not be set."""
    return _finish_basket(ep, roles, brief, lines) if _commit_lines(ep, roles, brief, lines) else None


def _finish_basket(ep, roles, brief, lines):
    """Report the basket when a procedure wants it, hear the reply, then close; returns the shop's last answer."""
    heard = _hear_statement(ep, roles, brief) if _state_basket(ep, roles, brief, lines) else "order"
    closing = _decline_call(ep, roles, brief) if heard == "decline" else _fcc_purchase_call(ep, roles, brief)
    return None if closing is None or ep.finished else _fcc_run(ep, *closing)


# ---------------------------------------------------------------- listing facts

def _fcc_total_cap(brief):
    budget = (brief or {}).get("budget")
    if not isinstance(budget, dict):
        budget = {}
    if str(budget.get("applies_to") or "total") != "total":
        return None
    return _number(budget.get("amount"))


def _unchecked_cheaper(ep, brief, limit=5):
    """Lowest-price requests: unopened listings from results priced below the single carted line, in its category,
    not sold out and within a total budget; cheapest first."""
    held_ids = _cart_ids(ep) or [lid for lid, n in ep.cart_set.items() if n > 0]
    if len(held_ids) != 1:
        return []
    mine = held_ids[0]
    held = ep.listings.get(mine) or {}
    ceiling = _listing_price(held)
    if ceiling is None:
        return []
    cat = _category_value(ep, mine, brief)
    cat_key = next((k for k, v in held.items() if cat and v == cat), None)
    cap = _fcc_total_cap(brief)
    skip_gone = (brief or {}).get("skip_sold_out", True)

    def other_kind(rec):
        return bool(cat_key) and rec.get(cat_key) is not None and rec.get(cat_key) != held.get(cat_key)

    found = []
    for lid, rec in ep.listings.items():
        if lid == mine or lid in ep.viewed:
            continue
        cost = _listing_price(rec)
        too_dear = cost is None or cost >= ceiling or (cap is not None and cost > cap)
        if too_dear or (skip_gone and _sold_out(rec)) or other_kind(rec):
            continue
        found.append((cost, lid))
    return [(lid, cost) for cost, lid in sorted(found)[:limit]]


def _ids_for_titles(ep, args):
    """Cart arguments that give the start of a title instead of an id get that listing's id (when unique)."""
    fixed = dict(args)
    for k, v in args.items():
        if isinstance(v, str) and len(v) >= 12 and v not in ep.listings:
            owners = [lid for lid, rec in ep.listings.items()
                      if any(isinstance(x, str) and x.startswith(v) for x in rec.values())]
            if len(owners) == 1:
                fixed[k] = owners[0]
    return fixed


FIELDS_SYSTEM = """Answer only with JSON of the form {"price": name of the field holding the listing's price,
"stock": name of the field telling whether the listing can be bought now, or null, "in_stock": the value of that
field that means it can be bought (true, false, a word as written, or "above 0" for a count)}. Use the field names
exactly as the records spell them; the records are listings from one store."""


def _fields():
    """This task's listing field roles: price, stock, in_stock (empty until the probe is read)."""
    return vars(_spend).get("fields") or {}


def _fcf_median(values):
    ranked = sorted(values)
    return ranked[len(ranked) // 2]


def _fields_by_shape(records):
    """Pure: field roles from the records' shape alone: the price is the numeric field with the largest median, the
    stock the true/false field (one whose value varies first) and in stock its more common value."""
    numbers, flags = {}, {}
    for rec in records:
        for k, v in rec.items():
            if k.startswith("_"):
                continue
            if isinstance(v, bool):
                flags.setdefault(k, []).append(v)
            elif isinstance(v, (int, float)):
                numbers.setdefault(k, []).append(v)
    roles = {}
    if numbers:
        roles["price"] = max(numbers, key=lambda k: (_fcf_median(numbers[k]), len(numbers[k])))
    if flags:
        stock = max(flags, key=lambda k: (len(set(flags[k])), len(flags[k])))
        seen = flags[stock]
        roles.update(stock=stock, in_stock=max((True, False), key=seen.count))
    return roles


def _fcf_plain(records):
    """Pure: the shape leaves no doubt: exactly one numeric field and one true/false field at the top level."""
    numeric = {k for r in records for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    flags = {k for r in records for k, v in r.items() if isinstance(v, bool)}
    return len(numeric) == 1 and len(flags) == 1


def _read_fields(ep):
    """The price and stock fields of this store's listings: from the records' shape, and from one light call when
    the shape leaves a doubt (several numbers, no true/false field or several)."""
    opened = [lid for lid in ep.viewed if lid in ep.listings][:3]
    records = [{k: v for k, v in ep.listings[lid].items() if not k.startswith("_")} for lid in opened]
    roles = _fields_by_shape(list(ep.listings.values()))
    if records and not _fcf_plain(records):
        got = _ask_json([LIGHT, LIGHT_SPARE], FIELDS_SYSTEM, _fcc_json(_shrink(records, 120)), max_tokens=400)
        got = got if isinstance(got, dict) else {}
        keys = set().union(*records)
        if got.get("price") in keys:
            roles["price"] = got["price"]
        if got.get("stock") in keys:
            roles.update(stock=got["stock"], in_stock=got.get("in_stock"))
    _spend.fields = roles


def _listing_price(d):
    key = _fields().get("price")
    return _number(d.get(key)) if key in d and not isinstance(d.get(key), bool) else None


def _sold_out(d):
    """Whether the stock field says the listing cannot be bought now: a count at or below 0, or a value other than
    the in-stock one."""
    key = _fields().get("stock")
    if key not in d:
        return False
    value, wanted = d[key], _fields().get("in_stock")
    if isinstance(value, bool):
        return value != (wanted if isinstance(wanted, bool) else True)
    if isinstance(value, (int, float)):
        return value <= 0
    if isinstance(wanted, str) and isinstance(value, str):
        return _fcd_low(value).strip() != _fcd_low(wanted).strip()
    return False


def _fcc_label_like(key, value, lid):
    return (not key.startswith("_") and isinstance(value, str) and 0 < len(value) <= 80
            and not _code_like(value) and value != lid)


def _category_value(ep, lid, brief):
    """The listing's category as the shop writes it: a short text field shared by many listings seen, with a bonus
    when its words match a requested product."""
    wanted = set()
    for entry in _fcc_entries(brief):
        wanted |= _words(" ".join((str(entry.get("what") or ""), str(entry.get("category") or ""))))
    scored = []
    for k, v in (ep.listings.get(lid) or {}).items():
        if not _fcc_label_like(k, v, lid):
            continue
        peers = [other.get(k) for other in ep.listings.values() if isinstance(other.get(k), str)]
        share = peers.count(v) / len(peers) if len(peers) >= 3 else 0.0
        if share >= 0.3:
            scored.append((share + (0.5 if _words(v) & wanted else 0.0), v))
    # the first best-scoring field wins a tie
    return max(scored, key=lambda pair: pair[0])[1] if scored else None


def _page_token(obs):
    """A result's paging token: its one short top-level text value."""
    return _fcc_lone_string(obs, lambda v: 0 < len(v) <= 40)

# ---------------------------------------------------------------- small shared pieces

def _fcd_js(value):
    """JSON text with non-ASCII kept as is (how the transcript quotes objects)."""
    return json.dumps(value, ensure_ascii=False)


def _fcd_msg(role, text, ref=None, calls=None):
    """One transcript message; a tool reply carries the id of the call it answers."""
    msg = dict(role=role)
    if role == "tool":
        msg.update(tool_call_id=ref)
    msg.update(content=text)
    if calls is not None:
        msg.update(tool_calls=calls)
    return msg


def _fcd_shaped(ep, tool_name, raw):
    """Arguments fitted to a world tool's schema (the tool must exist)."""
    spec = ep.by_name[tool_name]
    return _coerce_args(raw, _schema(spec))


def _fcd_room(ep, need):
    """Pure: is the episode open with at least `need` steps left?"""
    return not ep.finished and ep.left >= need


# ---------------------------------------------------------------- lowest-price sweep

def _fcd_sweep_setup(ep, brief):
    """Pure: (tool, schema, limit arg, category arg, page arg) of the brief's price filter, or None if unusable."""
    spec = (brief or {}).get("price_filter")
    if not isinstance(spec, dict) or spec.get("tool") not in ep.by_name:
        return None
    tool = spec["tool"]
    schema = _schema(ep.by_name[tool])
    fields = schema.get("properties") or {}
    limit_key = spec.get("max_price_arg")
    if limit_key not in fields:
        return None
    if (fields[limit_key] or {}).get("type") not in ("number", "integer"):
        return None
    optional = []
    for slot in ("category_arg", "page_arg"):
        given = spec.get(slot)
        optional.append(given if given in fields else None)
    return tool, schema, limit_key, optional[0], optional[1]


def _fcd_wanted(brief, qa):
    """Pure: words of every stated must (value part) and of every shopper reply."""
    texts = [" ".join(str(m).split(":", 1)[-1] for m in item.get("musts") or [])
             for item in (brief or {}).get("items") or [] if isinstance(item, dict)]
    texts.extend(reply for _q, reply in qa)
    return set().union(*map(_words, texts))


def _fcd_ceiling(price):
    """Pure: the filter limit just under a price (one unit less for whole prices above 1)."""
    whole = float(price).is_integer() and price > 1
    return price - 1 if whole else price * 0.999


def _fcd_cheaper_rows(found, price, skip, drop_sold, wanted):
    """Pure: (id, (word overlap, price)) for each row under the price that is neither skipped nor sold out."""
    rows = []
    for lid, row in found:
        cost = _listing_price(row)
        if cost is None or cost >= price or lid in skip:
            continue
        if drop_sold and _sold_out(row):
            continue
        blob = " ".join(v for v in row.values() if isinstance(v, str))
        rows.append((lid, (len(_words(blob) & wanted), cost)))
    return rows


def _fcd_filter_pages(ep, setup, price, cat, held, drop_sold, wanted, pool):
    """Run the price filter for one carted line, up to three result pages, adding cheaper rows to the pool."""
    tool, schema, limit_key, cat_key, page_key = setup
    query = {limit_key: _fcd_ceiling(price)}
    if cat:
        query[cat_key] = cat
    for _page in range(3):
        if not _fcd_room(ep, END_RESERVE + 8):
            return
        obs = ep.call(tool, _coerce_args(query, schema))
        found = []
        _listings_in(obs, found, set())
        pool.update(_fcd_cheaper_rows(found, price, ep.viewed.union(held), drop_sold, wanted))
        token = _page_token(obs)
        if not (found and page_key and token):
            return
        query = dict(query)
        query[page_key] = token


def _fcd_open_cheaper(ep, viewer, ranked):
    """Open the best-ranked cheaper listings with one page tool; returns (count opened, result lines)."""
    page = ep.by_name[viewer]
    fields = _schema(page).get("properties") or {}
    arrays = [k for k in fields if isinstance(fields[k], dict) and fields[k].get("type") == "array"]
    if arrays:
        key, groups = arrays[0], [ranked[:6], ranked[6:12]]
    else:
        key, groups = _text_arg(page), [ranked[n:n + 1] for n in range(min(3, len(ranked)))]
    opened, lines = 0, []
    for group in groups:
        if not (group and key and _fcd_room(ep, END_RESERVE + 6)):
            break
        sent = _fcd_shaped(ep, viewer, {key: group if arrays else group[0]})
        seen = _compact_obs(ep.call(viewer, sent))
        lines.append("%s %s -> %s" % (viewer, _fcd_js(sent), seen))
        opened += len(group)
    return opened, lines


def _fcd_sweep_note(total, opened, lines, rest):
    """Pure: the model-facing summary of a price sweep."""
    parts = [f"Price check for a lowest-price shopper. I ran the shop's own price filter on the cart's category, "
             f"capped just under what the cart pays now (this used steps). Found: {total} cheaper available "
             f"listings. Opened for you, best fit first: {opened}."]
    parts.extend(lines)
    if rest:
        parts.append("Still unopened, as id (price): " + ", ".join(rest))
    parts.append("Your move: test them by their page facts against the request's requirements and the needs "
                 "the shopper shared. When some pass, put the cheapest passing one in place of the current line "
                 "(same quantity, old line out); when none pass, the cart stays. Close afterwards.")
    return "\n".join(parts)


def _cheaper_sweep(ep: Episode, brief: dict | None, roles: dict, cart_view: str | None) -> str | None:
    """Before a lowest-price order: list the carted line's category under its price with the shop's price filter
    (set here in code, since the light model often gets the limit wrong), open the likeliest cheaper listings and
    return a note, or None when there is nothing to show."""
    setup = _fcd_sweep_setup(ep, brief)
    if setup is None:
        return None
    if cart_view and (ep.cart_dirty or ep.cart_obs is None):
        ep.cart_obs = _fcd_look_at_cart(ep, cart_view)
    held = _cart_ids(ep)
    if len(held) not in (1, 2):
        return None
    wanted = _fcd_wanted(brief, ep.qa)
    drop_sold = bool((brief or {}).get("skip_sold_out", True))
    by_cat = bool(setup[3])
    pool = {}
    for lid in held:
        price = _listing_price(ep.listings.get(lid) or {})
        cat = _category_value(ep, lid, brief) if by_cat else None
        if price is not None and (cat or not by_cat):
            _fcd_filter_pages(ep, setup, price, cat, held, drop_sold, wanted, pool)
    if not pool:
        return None
    ranked = sorted(pool, key=lambda i: (-pool[i][0], pool[i][1]))
    viewer = next((t for t in (roles.get("compare") or []) + (roles.get("view") or []) if t in ep.by_name), None)
    opened, lines = (0, []) if viewer is None else _fcd_open_cheaper(ep, viewer, ranked)
    rest = ["%s (%s)" % (i, format(pool[i][1], "g")) for i in ranked[opened:opened + 12]]
    return _fcd_sweep_note(len(pool), opened, lines, rest)


def _fcd_title_of(ep, lid, brief):
    """The listing's own name as text: the shortest of its multi-word text fields once its category is left out
    (other sellers often list the identical item under the same name)."""
    rec = ep.listings.get(lid) or {}
    cat = _category_value(ep, lid, brief)
    texts = [v for k, v in rec.items() if _fcc_label_like(k, v, lid) and v != cat and len(v.split()) >= 3]
    return min(texts, key=len) if texts else None


def _fcd_twin_query(title, most=6):
    """Pure: the first words of a listing name, without bracketed parts, code-like tokens and bare figures."""
    head = re.sub(r"\([^)]*\)", " ", title)
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9&'-]*", head)
             if not _code_like(w) and sum(c.isalpha() for c in w) >= 2]
    return " ".join(words[:most])


def _fcd_twin_search(ep, brief, roles, lid):
    """Search a carted listing's name with the search its product was probed with (same filters), else with any
    free-text search tool; one step. True when it ran."""
    title = _fcd_title_of(ep, lid, brief)
    query = _fcd_twin_query(title) if title else ""
    if len(query.split()) < 2:
        return False
    label = ep.probe_of.get(lid)
    plan = [p for p in (brief or {}).get("probes") or [] if isinstance(p, dict)]
    spec = next((p for p in plan if label and p.get("item") == label and p.get("tool") in ep.by_name), None)
    if spec and _text_arg(ep.by_name[spec["tool"]]):
        tool, args = spec["tool"], dict(spec.get("args") if isinstance(spec.get("args"), dict) else {})
    else:
        tool, args = _fcb_keyword_tool(ep, roles, ""), {}
    if not tool:
        return False
    args[_text_arg(ep.by_name[tool])] = query
    key = tool + json.dumps(_fcd_shaped(ep, tool, args), sort_keys=True)
    if key in ep.memo:
        return False
    ep.memo[key] = _compact_obs(_fcc_run(ep, tool, args))
    return True


def _fcd_word_sweep(ep, brief, roles, cart_view):
    """Lowest-price shopper and no price filter: search each carted line's name (one or two lines, one step each)
    so cheaper copies from other sellers enter the results; the unopened-cheaper check then lists them."""
    if cart_view and (ep.cart_dirty or ep.cart_obs is None):
        ep.cart_obs = _fcd_look_at_cart(ep, cart_view)
    held = _cart_ids(ep) or [lid for lid, n in ep.cart_set.items() if n > 0]
    if len(held) not in (1, 2):
        return
    for lid in held:
        if not _fcd_room(ep, END_RESERVE + 10):
            break
        _fcd_twin_search(ep, brief, roles, lid)


def _fcd_look_at_cart(ep, cart_view):
    """View the cart (one step) and mark the stored view fresh; returns the observation."""
    obs = ep.call(cart_view, _fcd_shaped(ep, cart_view, {}))
    ep.cart_dirty = False
    return obs


# ---------------------------------------------------------------- closing decisions (pure)

_FCD_REMARK = "Closing: the requirements could not all be confirmed."
_FCD_STATE = "[{0}/{1} steps used, {2} to go; {3}/{4} shopper messages sent]"


def _state_line(ep: Episode, brief: dict | None, sent: int) -> str:
    return _FCD_STATE.format(ep.step, ep.max_steps, ep.left, sent, _message_limit(brief))


def _finish_modes(brief: dict | None) -> dict:
    table = (brief or {}).get("finish_modes")
    if not isinstance(table, dict):
        return {}
    return {tool: how for tool, how in table.items() if isinstance(how, dict)}


def _fcd_close_plan(ep, roles, brief):
    """Pure: closing calls to try in turn, the purchase first when something is in the cart."""
    modes = _finish_modes(brief)
    kinds = ("order", "decline") if ep.carted else ("decline",)
    plan = []
    for kind in kinds:
        direct = "checkout" if kind == "order" else "decline"
        plan += [(t, {}) for t in roles.get(direct, [])]
        plan += [(t, _mode_args(ep, t, modes.get(t, {}).get(kind))) for t in roles.get("finish", [])]
    return plan


def _fcd_with_required(args, schema):
    """Pure: the arguments plus the fixed remark for each required free-text field left empty."""
    fields = schema.get("properties") or {}
    filled = dict(args)
    for key in schema.get("required") or []:
        spec = fields.get(key)
        free_text = isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")
        if free_text and key not in filled:
            filled[key] = _FCD_REMARK
    return filled


def _fallback_close(ep: Episode, roles: dict, brief: dict | None) -> None:
    """Never leave the episode open: buy what the cart holds, otherwise close without a purchase."""
    if not _fcd_room(ep, 1):
        return
    if ep.carted:
        _state_basket(ep, roles, brief)
    else:
        _procedure_message(ep, roles, brief, "before_decline", _FCF_DECLINE % _FCF_NO_FIT)
    for tool, raw in _fcd_close_plan(ep, roles, brief):
        if not _fcd_room(ep, 1):
            break
        if tool in ep.by_name:
            schema = _schema(ep.by_name[tool])
            ep.call(tool, _coerce_args(_fcd_with_required(raw, schema), schema))


def _fcd_low(value):
    return str(value).lower()


def _mode_values(want: Any) -> list[str]:
    if isinstance(want, str):
        want = [want]
    elif not isinstance(want, list):
        return []
    plain = [v for v in want if not isinstance(v, bool) and isinstance(v, (str, int))]
    return list(map(_fcd_low, plain))


def _mode_args(ep: Episode, name: str, want: Any) -> dict:
    """Arguments of a finish tool for one way of closing; the brief gives either the arguments or bare values."""
    if isinstance(want, dict):
        return dict(want)
    wanted = set(_mode_values(want))
    fields = _schema(ep.by_name.get(name) or {}).get("properties") or {}
    chosen = {}
    for field in fields:
        spec = fields[field]
        options = spec.get("enum") if isinstance(spec, dict) else None
        if isinstance(options, list):
            hits = [o for o in options if _fcd_low(o) in wanted]
            if hits and hits[0] is not None:
                chosen[field] = hits[0]
    return chosen


def _fcd_mode_hit(want, args):
    """Pure: do these finish arguments carry the values one closing mode is given by?"""
    if isinstance(want, dict) and want:
        if all(_fcd_low(args.get(k)) == _fcd_low(want[k]) for k in want):
            return True
    accepted = _mode_values(want)
    if not accepted:
        return False
    return any(_fcd_low(a) in accepted for a in args.values() if isinstance(a, (str, int)))


def _closing_kind(roles: dict, tool: str, args: dict, brief: dict | None = None) -> str | None:
    if tool in roles.get("checkout", []):
        return "order"
    if tool in roles.get("decline", []):
        return "decline"
    if tool not in roles.get("finish", []):
        return None
    modes = _finish_modes(brief).get(tool) or {}
    hit = [k for k in ("order", "decline") if _fcd_mode_hit(modes.get(k), args)]
    return hit[0] if hit else "close with arguments " + _fcd_js(args)


def agent_main(problem_data: dict) -> list[dict]:
    """Run one episode and return its dialogue; whatever the loop does, a closing attempt follows."""
    for field, start in (("usd", 0.0), ("waits", 0), ("light_out", 0), ("t0", time.monotonic()), ("caps", None),
                         ("fields", None), ("last_call", None)):
        setattr(_spend, field, start)
    time.sleep(random.uniform(0.0, 3.0))  # tasks start together: spread the first calls to the proxy
    episode = Episode(problem_data["environment"])
    if episode.finished:
        return []
    held = dict(roles={}, brief=None)
    for phase in (lambda: _run(episode, held), lambda: _fallback_close(episode, held["roles"], held["brief"])):
        try:
            phase()
        except Exception:  # a bug must not leave the episode open
            pass
    return episode.dialogue


# ---------------------------------------------------------------- turn loop: state and pure decisions

class _fcd_Ctx:
    """What stays fixed for the whole turn loop: the episode, the brief and the tool sets derived from the roles."""

    def __init__(self, ep, request, brief, roles):
        self.ep, self.request, self.brief, self.roles = ep, request, brief, roles
        named = lambda *keys: {t for k in keys for t in roles.get(k, [])}  # noqa: E731
        self.searchers = named("search")
        self.messengers = named("message")
        self.pagers = named("view", "compare")
        self.lookups = self.searchers | self.pagers
        self.cart_tools = named("cart_change", "cart_set")
        self.view_cart = (roles.get("cart_view") or [None])[0]
        enders = named("checkout", "decline", "finish")
        self.closers = [t for t in ep.tools if _tool_name(t) in enders]
        narrow = {_tool_name(t) for t in self.closers} | self.cart_tools
        if self.view_cart:
            narrow.add(self.view_cart)
        self.narrow = [t for t in ep.tools if _tool_name(t) in narrow]
        self.msg_limit = _message_limit(brief)
        self.cheap = (brief or {}).get("cheapest_wins") is True
        self.removal = ""

    def state(self, lp):
        return _state_line(self.ep, self.brief, lp.msgs)


class _fcd_Loop:
    """Counters and flags the turn loop carries from one turn to the next."""

    def __init__(self, msgs):
        self.msgs = msgs
        self.turns = self.idle = self.idle_steps = self.streak = 0
        self.reviews = self.redecides = self.cheaper_checks = self.over = 0
        self.decided = self.widened = self.decline_pushed = self.refreshed = False
        self.wrap_told = self.swept = self.placed = self.cart_fix_told = False
        self.proof_told = self.backed_decline = self.previewed = False
        self.code_places = 0
        self.relocked = False
        self.notice_searched = False
        self.vets = 0
        self.vetoed = {}
        self.reason = ""
        self.start = self.paced = 0
        self.pick_lines = None
        self.pages = {}


_FCD_NUDGE = "Your next reply must be a tool call."
_FCD_SILENT = "(answered without a tool call)"
_FCD_HOLD = "Paused by the supervisor; this call did not run."
_FCD_HOLD_CHECK = "Paused for a review; this call did not run."
_FCD_BOSS = "Supervisor here: "
_FCD_OFFER_NOTES = {
    "final": ("[last step] Use this turn to end the episode: buy when your choice sits in the cart, "
              "otherwise finish with nothing bought."),
    "wrap": ("[wrap-up] Searching is over. Two moves remain. (1) Cart check: your strongest find must be in the "
             "cart, meaning the seen listing that satisfies most requirements, is of the product type requested "
             "and keeps to the money limit; add it if absent. (2) End the episode by buying it. Finishing with "
             "nothing bought is right only if each seen listing visibly fails some requirement by its own page "
             "data (a different value, a different category, a price above the limit)."),
    "idle": ("[no progress] Nothing has come of the recent calls. Cart the top qualifying listing if that has "
             "not happened, and finish."),
}


def _fcd_step_cap(brief, max_steps):
    """Pure: a lower step budget the brief states, if any."""
    raw = (brief or {}).get("step_limit") or 0
    try:
        cap = int(raw)
    except (TypeError, ValueError):
        return None
    return cap if 0 < cap < max_steps else None


def _fcd_intro(cx, probe_notes, state):
    """Pure: the first user message of the tool loop."""
    ep, brief = cx.ep, cx.brief
    opening = ep.policy.get("opening")
    blocks = ["SHOPPER REQUEST:\n%s" % (cx.request,)]
    if opening is not None:
        blocks.append("OPENING NOTE:\n" + _fcd_js(opening))
    if brief:
        blocks.append("EXPERT BRIEF ON THE REQUEST:\n" + _brief_text(brief))
    if brief and cx.roles:
        blocks.append("TOOL ROLES: " + json.dumps(cx.roles))
    if probe_notes:
        blocks.append("EARLY SHOP LOOKUPS (done for you; their steps count):\n" + "\n".join(probe_notes))
    if ep.qa:
        pairs = ["Q: %s\nA: %s" % pair for pair in ep.qa]
        pairs.append("(An answer showing confusion about a question means the shopper holds no wish on that "
                     "property.)")
        blocks.append("SHOPPER Q&A SO FAR:\n" + "\n".join(pairs))
    if cx.removal:
        blocks.append("CART: " + cx.removal)
    blocks.append(state + " Go ahead.")
    return "\n\n".join(blocks)


def _fcd_offer_mode(forced, has_closers, over_limits, idle_steps, wrap_told):
    """Pure: which tools to offer this turn: 'final' (closers only), 'wrap'/'idle' (narrow set, first notice),
    'narrow' (narrow set again) or None (every tool)."""
    if not has_closers:
        return None
    if forced:
        return "final"
    if wrap_told:
        return "narrow"
    if over_limits:
        return "wrap"
    return "idle" if idle_steps >= 4 else None


def _fcd_halt(lp, written, blown, carted):
    """Pure: stop the loop before the next model turn?"""
    return (lp.idle_steps >= 8 or lp.turns >= _caps()[0] + 6 or written >= _work_caps()[1]
            or (blown and (carted or lp.over > 4)) or _elapsed() >= CLOCK_STOP)


def _fcd_call_name(call):
    return (call.get("function") or {}).get("name")


def _fcd_page_batch(calls, pagers, wrap_told):
    """Pure: the calls as a batch of page openings, when every one of several calls opens a page."""
    if wrap_told or len(calls) < 2:
        return None
    if all(isinstance(c, dict) and _fcd_call_name(c) in pagers for c in calls):
        return list(calls)
    return None


def _fcd_batch_step(key, ep):
    """Pure: 'memo' (answer from memory), 'full' (too few steps) or 'run' for one batched page call."""
    if key in ep.memo:
        return "memo"
    return "run" if _fcd_room(ep, END_RESERVE + 2 + CLOSE_SPARE) else "full"


def _fcd_query_key(tool, args, seen):
    """Pure: a search's identity across pages: its free-text values the shop never showed, normalised."""
    typed = []
    for field, value in args.items():
        if isinstance(value, str) and value not in seen:
            typed.append((field, value.strip().lower()))
    typed.sort()
    return tool + "|" + json.dumps(typed)


def _fcd_gate(cx, lp, tool, memo_key, qkey):
    """Pure: why a lookup should not run ('streak', 'paged', 'memo', 'full'), or None."""
    ep = cx.ep
    remembered = memo_key in ep.memo
    if remembered:
        return "memo" if tool in cx.lookups else None
    if tool in cx.lookups and (ep.carted or lp.decided) and not _fcd_room(ep, END_RESERVE + 2 + CLOSE_SPARE):
        return "full"  # the steps left belong to the cart, the basket report and the close (plus one retry)
    if tool not in cx.searchers:
        return None
    if lp.streak >= 5 and any(i not in ep.viewed for i in ep.listings):
        return "streak"
    return "paged" if lp.pages.get(qkey, 0) >= 6 else None


def _fcd_gate_text(code, earlier, state):
    """Pure: the tool reply for a gated lookup."""
    if code == "streak":
        body = ("Refused without using a step - too many searches back to back. Open your strongest found "
                "listings first (their full pages, or a side-by-side view where a tool offers it), then search.")
    elif code == "full":
        body = ("Refused without using a step - the remaining steps are kept for the cart, the basket report and "
                "the close, which the shop may turn down once. Cart your pick if it is absent, then close.")
    elif code == "paged":
        body = ("Refused without using a step - enough pages of this query were read. Options: tighten it "
                "with one more wanted value, open candidates, or choose.")
    else:
        body = "Repeat of an earlier call (no step used); it returned:\n" + earlier + "\nMake a different move."
    return body + "\n" + state


def _fcd_decide_due(lp, ep, spent, closing):
    """Pure: should the strong model choose now (few steps or spend high, nothing carted, not closing)?"""
    _turns, soft, hard = _caps()
    if lp.decided or ep.carted or spent >= hard or closing or _elapsed() >= CLOCK_WRAP:
        return False
    return ep.left <= END_RESERVE + 3 + CLOSE_SPARE or (spent >= soft and not lp.wrap_told)


def _fcd_pick_note(pick, steps, state):
    return ("Paused: choosing time has come, and a senior decision was taken from your findings:\n" +
            _fcd_js(pick) + "\n" + steps + "\nAct on it at once. For an order the cart must contain exactly those "
            "lines; then finish the episode in the manner decided.\n" + state)


def _fcd_review_hold(verdict, kind, extra, cart_obs, state):
    """Pure: the hold a review verdict calls for (a fix to carry out or the other kind of close), or None."""
    if not verdict:
        return None
    flag, fix = verdict.get("verdict", ""), verdict.get("fix")
    if extra:
        flag, fix = "fix", (str(fix or "") + " " + extra + " (same quantities as requested)").strip()
    if _fcd_low(flag) == "fix" and fix:
        why = verdict.get("reason")
        lines = ["Close paused. Senior review result (an order to follow, not advice):", str(fix)]
        if why:
            lines.append("Grounds: " + str(why))
        lines += ["Cart in its newest view: " + _fcd_js(cart_obs), state]
        return "\n".join(lines)
    other = _fcd_low(verdict.get("close_as", ""))
    if other != kind and {other, kind} <= {"order", "decline"}:
        return "Close paused. Review outcome: the fitting way to end is %s.\nGrounds: %s\n%s" % (
            other, verdict.get("reason", ""), state)
    return None


def _fcd_tool_text(text, order_note, ep, dropped, state):
    """Pure: the tool reply for a call that ran."""
    parts = [order_note, text] if order_note else [text]
    if ep.left <= END_RESERVE + CLOSE_SPARE and not ep.finished:
        parts[-1] += "\nSTEPS NEARLY GONE: cart your pick if it is absent, and end the episode straight away."
    if dropped:
        parts[-1] += "\n(Extra calls were dropped: a turn runs one tool, unless every call opens a page.)"
    parts.append(state)
    return "\n".join(parts)


def _fcd_cart_line(args, listings):
    """Pure: the listing and count a cart call names."""
    lid = count = None
    for value in args.values():
        if lid is None and isinstance(value, str) and value in listings:
            lid = value
        if count is None and isinstance(value, (int, float)) and not isinstance(value, bool):
            count = value
    return lid, 1 if count is None else count


def _fcd_still_open_note(cheap_order, pick, steps):
    note = (_FCD_BOSS + "according to the shop, that call left the episode open. Check its result. Did "
            "anything move (a price, the stock, or what the shopper asks for)? Repair the cart, then close once "
            "more. Does it confirm the order or the decline? Then use whichever tool finishes the episode.")
    if cheap_order:
        note += (" Since the goal is the lowest price, hunt once more for cheaper fitting listings before falling "
                 "back to another order: later result pages, new phrasings, a price cap where a tool accepts one.")
    if pick:
        note += ("\nSenior decision based on all findings so far (follow it): " + _fcd_js(pick) + "\n" + steps +
                 "\nField search_first (when given) is a query to run, so the identical item turns up from a "
                 "second seller. Field open_first (when given) lists pages to view before anything else; prefer one "
                 "of them if it passes all requirements and serves the shopper's goal better. Afterwards fix the "
                 "cart and close.")
    return note


# ---------------------------------------------------------------- reach: searches that adapt to the store

def _fcg_scalars(obs):
    """Pure: a result's top-level plain values (no lists, maps, flags or nulls)."""
    if not isinstance(obs, dict):
        return {}
    return {k: v for k, v in obs.items() if v is not None and not isinstance(v, (bool, dict, list))}


def _fcg_rows(obs):
    """Pure: listing ids a result shows, in its order."""
    hits = []
    _listings_in(obs, hits, set())
    return list(dict.fromkeys(lid for lid, _ in hits))


def _hit_count(obs, shown):
    """Pure: the result's total number of hits: its largest top-level whole number, when that is at least the rows
    shown; else None."""
    whole = [v for v in _fcg_scalars(obs).values() if isinstance(v, int)]
    most = max(whole, default=None)
    return most if most is not None and most >= shown and most > 0 else None


def _next_token(obs, hits):
    """Pure: the token for the next page: the one top-level plain value besides the hit count."""
    others = [v for v in _fcg_scalars(obs).values() if v != hits]
    return others[0] if len(others) == 1 else None


def _fcg_mode(probe, words, hits):
    """Pure: 'all' when a query with more words returned fewer hits than the probe (every word must match), 'any'
    when it returned as many or more (results ranked by matching words), None when it cannot be told."""
    if not probe or probe[1] is None or hits is None or words <= probe[0]:
        return None
    return "all" if hits < probe[1] else "any"


def _fcg_names(entry):
    """Pure: names to search a product by: the brief's store wordings, then its kind as the brief wrote it."""
    out = []
    for name in list(entry.get("names") or []) + [entry.get("what")]:
        text = " ".join(str(name or "").split())
        if text and _fcd_low(text) not in map(_fcd_low, out):
            out.append(text)
    return out[:4]


def _fcg_values(entry, dropped=()):
    """Pure: a product's stated values in the store's words (the brief's value_words; only when it gave no list,
    each must's text after its colon), without those sharing a word with a dropped condition."""
    given = [str(v).strip() for v in entry.get("value_words") or [] if str(v or "").strip()]
    if not given and not isinstance(entry.get("value_words"), list):
        given = [str(m).partition(":")[2].strip() for m in entry.get("musts") or [] if ":" in str(m)]
    gone = set().union(*map(_words, dropped)) if dropped else set()
    return [v for v in given if v and not (_words(v) & gone)][:4]


def _fcg_queries(mode, names, values, wide=False):
    """Pure: query texts in the order to try. Every-word stores: the first name with every value, then each value
    left out; ranked stores: each name with every value. wide=True adds the left-out forms for both and each name
    alone."""
    join = lambda name, vals: " ".join([name] + list(vals))  # noqa: E731
    if not names:
        return []
    drop_one = [join(names[0], values[:i] + values[i + 1:]) for i in range(len(values))] if len(values) > 1 else []
    if mode == "all":
        plan = [join(names[0], values)] + drop_one + [join(n, values) for n in names[1:]]
    else:
        plan = [join(n, values) for n in names]
    if wide:
        plan += drop_one + list(names)
    return list(dict.fromkeys(plan))


def _fcg_fit(text, limit):
    """Pure: the query cut at a word boundary to the tool's text limit."""
    if not limit or len(text) <= limit:
        return text
    head = text[:limit + 1]
    return head.rsplit(" ", 1)[0] if " " in head else text[:limit]


def _fcg_score(rec, values):
    """Pure: how many stated values a result row's own text shows in full."""
    have = _words(" ".join(str(v) for v in rec.values() if isinstance(v, (str, int, float))))
    return sum(1 for v in values if _words(v) and _words(v) <= have)


def _fcg_digest(seen):
    """A search result for the loop: its plain top-level values, then one short row per listing with the price and
    stock fields first and long texts left out, so every row stays readable (a cut JSON dump showed two rows)."""
    found = []
    _listings_in(seen, found, set())
    first = [k for k in (_fields().get("price"), _fields().get("stock")) if k]
    rows, done = [], set()
    for lid, rec in found:
        if lid in done:
            continue
        done.add(lid)
        keys = first + [k for k in rec if k not in first]
        parts = ["%s=%s" % (k, rec[k]) for k in keys
                 if k in rec and not str(k).startswith("_") and not isinstance(rec[k], (dict, list))
                 and rec[k] is not None and rec[k] != "" and rec[k] != lid
                 and not (isinstance(rec[k], str) and len(rec[k]) > 120)]
        rows.append("  %s: %s" % (lid, "; ".join(parts)[:300]))
    if not rows:
        return _compact_obs(seen)[:900]
    return _fcd_js(_fcg_scalars(seen)) + "\n" + "\n".join(rows)


TOLD_SYSTEM = """Answer only with JSON of the form {"values": [one entry per numbered reply, in order]}. Each entry
is the value the customer insists on in that reply (for the asked property or any other one), in at most 4 words as
a store listing would write it, using the reply's own words; null when the reply leaves the choice open."""


def _fcg_told_values(ep, heard):
    """One light call after the question round: the value each reply insisted on, kept per product when the reply
    itself holds its words (a told value is then searched for like a stated one)."""
    if not heard:
        return
    rows = ["%d. QUESTION ABOUT: %s of the %s\nREPLY: %s" % (n + 1, prop, item, reply)
            for n, (item, prop, reply) in enumerate(heard)]
    got = _ask_json([LIGHT, LIGHT_SPARE], TOLD_SYSTEM, "\n".join(rows), 400)
    values = got.get("values") if isinstance(got, dict) else None
    for (item, _prop, reply), value in zip(heard, values if isinstance(values, list) else []):
        text = " ".join(str(value or "").split())
        said = str(reply or "").lower()
        if text and len(text.split()) <= 4 and all(token in said for token in text.lower().split()):
            ep.told_values.append((item, text))


def _fcg_told(ep, brief, entry, dropped=()):
    """Pure on the episode record: values the shopper insisted on in replies for this product (with one requested
    product, every reply serves it), without those sharing a word with a dropped condition."""
    names = [" ".join(str(e["what"]).split()) for e in _fcc_entries(brief) if e.get("what")]
    mine = " ".join(str(entry.get("what") or "").split())
    gone = set().union(*map(_words, dropped)) if dropped else set()
    out = []
    for item, value in ep.told_values:
        owner = names[0] if len(names) == 1 else _fcc_match(item, names)
        if owner == mine and not (_words(value) & gone) and value not in out:
            out.append(value)
    return out[:3]


def _fcg_spec(ep, brief, roles, entry):
    """How to search for one product: the brief's probe for it (its filters kept, text and paging left out), else
    any free-text search; (tool, base args, text argument, paging argument, text limit) or None."""
    plan = [p for p in (brief or {}).get("probes") or [] if isinstance(p, dict) and p.get("tool") in ep.by_name]
    what = str(entry.get("what") or "")
    mine = next((p for p in plan if _fcc_match(str(p.get("item") or ""), [what])), None)
    tool = mine["tool"] if mine and _text_arg(ep.by_name[mine["tool"]]) else _fcb_keyword_tool(ep, roles, "")
    if not tool:
        return None
    props = _schema(ep.by_name[tool]).get("properties") or {}
    filled = (mine or {}).get("args") if mine and mine["tool"] == tool else None
    texts = _fcb_free_text_keys(ep.by_name[tool])
    # the free-text argument the brief's own probe put the product words in, else the schema's first one
    text = next((k for k in texts if isinstance(filled, dict) and isinstance(filled.get(k), str)
                 and _words(filled[k]) & _words(what)), None) or _text_arg(ep.by_name[tool])
    paging = ((brief or {}).get("search_page_args") or {})
    page = paging.get(tool) if isinstance(paging, dict) and paging.get(tool) in props else None
    base = dict(mine.get("args") or {}) if mine and mine["tool"] == tool and isinstance(mine.get("args"), dict) else {}
    base = {k: v for k, v in base.items() if k not in (text, page)}
    limit = (props.get(text) or {}).get("maxLength")
    return tool, base, text, page, limit if isinstance(limit, int) and limit > 8 else None


def _fcg_run(ep, tool, args):
    """One search step, remembered so the loop's identical call answers from memory; None when it ran before (in
    the memo, or as a code search earlier: a later code search under the same words would only repeat its rows)."""
    sent = _fcd_shaped(ep, tool, args)
    key = tool + json.dumps(sent, sort_keys=True)
    if key in ep.memo or key in ep.searched:
        return None
    ep.searched.add(key)
    seen = ep.call(tool, sent)
    ep.memo[key] = _compact_obs(seen)
    return seen


def _fcg_pager(ep, roles):
    """The page tool that has opened the most listings so far, else the first one there is."""
    pagers = [t for role in ("view", "compare") for t in roles.get(role) or [] if t in ep.by_name]
    used = [t for tools in ep.opened_by.values() for t in tools if t in pagers]
    return max(pagers, key=used.count) if pagers else None


def _fcg_open(ep, roles, lids, until):
    """Open these listings' pages (one call for all when the page tool takes a list) while steps last."""
    pager = _fcg_pager(ep, roles)
    if not pager or not lids:
        return []
    done = []
    many = _fcb_array_arg(ep.by_name[pager])
    for lid in ([lids[0]] if many else lids):
        if ep.finished or ep.step >= until or not _fcd_room(ep, END_RESERVE + 6):
            break
        args = _fcb_page_args(ep.by_name[pager], lid, lids)
        if args is not None:
            ep.call(pager, args)
            done += lids if many else [lid]
    return done


def _fcg_budget(ep):
    """Steps the opening searches may use: about a seventh of the task's steps, never eating into what the loop
    needs to compare, cart and close."""
    return max(0, min(max(4, round(ep.max_steps / 7)), ep.left - END_RESERVE - 14))


def _fcg_search_product(ep, brief, roles, entry, until, dropped, wide):
    """Run one product's queries (and their next pages) until the step mark; returns log lines and the new rows,
    best first."""
    spec = _fcg_spec(ep, brief, roles, entry)
    if spec is None:
        return [], []
    tool, base, text, page, limit = spec
    told = _fcg_told(ep, brief, entry, dropped)
    values = told + [v for v in _fcg_values(entry, dropped) if v not in told]
    names = _fcg_names(entry)
    # a value the shopper insisted on is searched first with the product's name alone, so other request words
    # cannot crowd it out of a ranked store's results
    lead = [" ".join([names[0]] + told)] if told and names else []
    cheap = (brief or {}).get("cheapest_wins") is True
    log, rows, records = [], [], {}
    for query in list(dict.fromkeys(lead + _fcg_queries(ep.search_mode, names, values, wide))):
        token, pages = None, 0
        while ep.step < until and _fcd_room(ep, END_RESERVE + 8) and pages < 3:
            args = dict(base, **{text: _fcg_fit(query, limit)})
            if token is not None:
                args[page] = token
            seen = _fcg_run(ep, tool, args)
            if seen is None:
                break
            shown = _fcg_rows(seen)
            hits = _hit_count(seen, len(shown))
            if ep.search_mode is None and token is None:
                ep.search_mode = _fcg_mode(ep.probe_hits, len(query.split()), hits)
            log.append(" ".join((tool, _fcd_js(args), "->", _fcg_digest(seen))))
            found = []
            _listings_in(seen, found, set())
            records.update((lid, rec) for lid, rec in found)
            rows += [lid for lid in shown if lid not in rows]
            pages += 1
            token = _next_token(seen, hits) if page else None
            short = hits is not None and hits <= 3 * max(1, len(shown))
            # lowest price wins: the cheapest passing row may sit on a later page of a plain name query, so page
            # on unless the store is known to require every word
            deep = cheap and ep.search_mode != "all"
            if token is None or not (ep.search_mode == "any" or short or deep):
                break
    fresh = [lid for lid in rows if lid not in ep.viewed]
    order = {lid: n for n, lid in enumerate(fresh)}
    ranked = sorted(fresh, key=lambda lid: (-_fcg_score(records.get(lid) or {}, values), order[lid]))
    return log, ranked


def _reach(ep, brief, roles, steps, dropped=(), wide=False):
    """Code-run searches per product that adapt to how the store matches words, then the most promising new rows
    opened; returns (log lines for the loop, opened ids)."""
    entries = [e for e in _fcc_entries(brief) if e.get("what")]
    if not entries or steps < 2:
        return [], []
    share = max(2, steps // len(entries))
    log, opened = [], []
    for entry in entries:
        until = min(ep.step + share, ep.max_steps - END_RESERVE - 6)
        lines, ranked = _fcg_search_product(ep, brief, roles, entry, until - 1, dropped, wide)
        log += lines
        take = 3 if len(entries) == 1 else 2
        opened += _fcg_open(ep, roles, ranked[:take], until + take)
    if opened:
        log.append("Pages opened after these searches: " + ", ".join(opened))
    return log, opened


# ---------------------------------------------------------------- listings a condition points to

REF_SYSTEM = """Answer only with JSON of the form {"refs": [{"property": the property whose value is to be taken
from another listing, "seller": that other listing's seller or shop name as written, or null, "kind": its kind of
product as written, or null, "details": [each identifying value the text gives for that other listing, a few words
each, as written, without its property name]}]}. The text is either a customer's words to a shopping assistant or one listing page's own
notes. List an entry only where the text points at a different listing (one described by its seller, its kind or
its features) to take a property's value from or to compare a property with. Use {"refs": []} when the text points
at no other listing."""

# references followed per task, pages opened per reference, hops from a page to the listing it passes a property on to
_FCH_REFS, _FCH_OPENS, _FCH_HOPS = 2, 3, 2


def _fch_text_of(rec):
    """Pure: one listing's words: every string and number it holds, nested property names and values included."""
    bag = []
    for k, v in rec.items():
        if str(k).startswith("_"):
            continue
        if isinstance(v, dict):
            bag += ["%s %s" % (_prop_key(n), x) for n, x in v.items() if not _fcb_empty(x)]
        elif isinstance(v, (str, int, float)) and not isinstance(v, bool):
            bag.append(str(v))
    return _words(" ".join(bag))


def _fch_seller_of(rec, seller):
    """Pure: the listing names this seller in one of its short text fields (a whole field's words)."""
    want = _words(seller)
    return bool(want) and any(isinstance(v, str) and len(v) <= 60 and want <= _words(v)
                              for k, v in rec.items() if not str(k).startswith("_"))


def _fch_value_part(detail, keys):
    """Pure: a detail without a leading property name the shop uses ("<property> <value>" -> "<value>")."""
    low = " ".join(_prop_key(detail).split())
    for key in sorted(keys, key=len, reverse=True):
        if low.startswith(key + " ") and _words(low[len(key) + 1:]):
            return low[len(key) + 1:]
    return low


def _fch_fit(rec, ref, keys=()):
    """Pure: (seller named, identifying details shown, kind words shown) for one stored listing; a detail is shown
    when its words are, or its value words once a leading property name of the shop (keys) is left out."""
    have = _fch_text_of(rec)
    details = [d for d in ref.get("details") or [] if _words(d)]
    shown = sum(1 for d in details if _words(d) <= have or _words(_fch_value_part(d, keys)) <= have)
    return (_fch_seller_of(rec, ref.get("seller") or ""), shown, len(_words(ref.get("kind") or "") & have))


def _fch_keys(ep):
    """The property names the shop's opened pages use, in plain form."""
    return {key for rec in ep.listings.values() for key, _v in _fcb_page_facts(rec)}


def _fch_clean(got):
    """Pure: the usable references of a REF_SYSTEM answer: a seller or at least two details to find them by."""
    out = []
    for r in (got or {}).get("refs") or [] if isinstance(got, dict) else []:
        if not isinstance(r, dict):
            continue
        details = [str(d) for d in r.get("details") or [] if isinstance(d, (str, int, float)) and _words(d)][:5]
        ref = dict(property=str(r.get("property") or ""), seller=str(r.get("seller") or ""),
                   kind=str(r.get("kind") or ""), details=details)
        if _words(ref["seller"]) or len(details) >= 2:
            out.append(ref)
    return out


def _fch_rank(ep, ref, skip):
    """Stored listings that could be the one meant, best first: the seller's own (when one is named), most details
    shown, then most kind words; ones in skip left out."""
    need_seller = bool(_words(ref.get("seller") or ""))
    keys = _fch_keys(ep)
    rows = []
    for n, (lid, rec) in enumerate(ep.listings.items()):
        if lid in skip:
            continue
        seller, shown, kind = _fch_fit(rec, ref, keys)
        if need_seller and not seller:
            continue
        if not need_seller and shown < 2:
            continue
        rows.append((-shown, -kind, n, lid))
    return [row[3] for row in sorted(rows)]


def _fch_match(ep, lid, ref):
    """The opened page shows the seller (when named) and every identifying detail."""
    rec = ep.listings.get(lid) or {}
    seller, shown, _kind = _fch_fit(rec, ref, _fch_keys(ep))
    named = bool(_words(ref.get("seller") or ""))
    return (seller or not named) and shown >= len(ref.get("details") or [])


def _fch_search(ep, brief, roles, ref):
    """One search for the listing meant: its seller with its details, else with its kind (the shop's free-text
    search, with only the arguments it requires besides the text)."""
    entries = _fcc_entries(brief)
    entry = next((e for e in entries if _fcc_match(ref.get("kind") or "", [str(e.get("what") or "")])), None)
    spec = _fcg_spec(ep, brief, roles, entry or (entries[0] if entries else {}))
    if spec is None:
        return
    tool, base, text, _page, limit = spec
    must = set(_schema(ep.by_name[tool]).get("required") or [])
    base = {k: v for k, v in base.items() if k in must}
    lead = " ".join(w for w in (ref.get("seller") or "").split())
    for tail in (ref.get("details") or [], [ref.get("kind") or ""]):
        query = " ".join([lead] + [str(t) for t in tail if t]).strip()
        if not query or not _fcd_room(ep, END_RESERVE + CLOSE_SPARE + 6):
            return
        seen = _fcg_run(ep, tool, dict(base, **{text: _fcg_fit(query, limit)}))
        if seen is not None and _fch_rank(ep, ref, ep.viewed):
            return


def _fch_locate(ep, brief, roles, ref):
    """Find and open the listing a reference means; the id of the opened page that shows it, or None."""
    for lid in _fch_rank(ep, ref, set()):
        if lid in ep.viewed and _fch_match(ep, lid, ref):
            return lid
    searched = False
    tries = 0
    while tries < _FCH_OPENS and _fcd_room(ep, END_RESERVE + CLOSE_SPARE + 4):
        fresh = _fch_rank(ep, ref, ep.viewed)
        if not fresh or (not searched and _fch_fit(ep.listings.get(fresh[0]) or {}, ref, _fch_keys(ep))[1] <
                         len(ref.get("details") or [])):
            if not searched:
                searched = True
                _fch_search(ep, brief, roles, ref)
                continue
            if not fresh:
                return None
        tries += 1
        _fcg_open(ep, roles, fresh[:1], ep.max_steps)
        if fresh[0] in ep.viewed and _fch_match(ep, fresh[0], ref):
            return fresh[0]
    return None


def _fch_notes_text(ep, lid):
    """Pure: the page's own longer free texts (not its title), where a pointer to another listing would stand."""
    rec = ep.listings.get(lid) or {}
    texts = [v for k, v in rec.items() if not str(k).startswith("_") and isinstance(v, str) and len(v) > 60]
    return "\n".join(texts)


def _fch_read(text):
    got = _ask_json([LIGHT, LIGHT_SPARE], REF_SYSTEM, text, max_tokens=900)
    return _fch_clean(got)


def _follow_refs(ep, brief, roles, request):
    """A condition may take its value from a different listing's page (named by seller and features), and that
    page may pass the property on to yet another listing. Code finds and opens each such page, so the value is
    read where it lives; returns note lines for the loop."""
    if not (roles.get("view") or roles.get("compare")):
        return []
    said = ["REQUEST: " + request] + ["ANSWER: " + str(a) for _q, a in ep.qa]
    refs = _fch_read("\n".join(said))[:_FCH_REFS]
    notes = []
    for ref in refs:
        chain, hop = [], ref
        for _depth in range(_FCH_HOPS):
            lid = _fch_locate(ep, brief, roles, hop)
            if lid is None or lid in chain:
                break
            chain.append(lid)
            pointer = _fch_notes_text(ep, lid)
            prop = _words(ref.get("property") or "")
            if not pointer or (prop and not prop & _words(pointer)):
                break
            onward = [r for r in _fch_read("PAGE NOTES OF LISTING %s: %s" % (lid, pointer))
                      if not prop or prop & _words(r.get("property") or "")]
            if not onward:
                break
            hop = onward[0]
        if chain:
            line = ("Listing %s is the page the shopper's condition on %s points to (seller %s; %s)"
                    % (chain[0], ref.get("property") or "a property", ref.get("seller") or "not named",
                       ", ".join(ref.get("details") or [])))
            if len(chain) > 1:
                line += "; that page passes the property on to " + " then ".join(chain[1:]) + \
                        ", so the value is taken from the last of these pages"
            notes.append(line + ". Its page (already opened): " + _fcb_page_row(ep.listings.get(chain[-1]) or {}))
        else:
            notes.append("A condition on %s points to another listing (seller %s; %s) that was not found yet: "
                         "search for it and open its page." % (ref.get("property") or "a property",
                                                              ref.get("seller") or "not named",
                                                              ", ".join(ref.get("details") or [])))
    ep.ref_notes = notes
    return notes


NOTICE_SYSTEM = """Answer only with JSON of the form {"product": the requested product the news is about, or null,
"dropped": [each wanted value the customer no longer requires, as the request words it], "budget": the new money
limit as a number, or null}. The news reached a shopping assistant during the purchase; the requested products
and their conditions are given."""


def _read_notice(brief, note):
    """One light call: which product a shopper update concerns, which values it drops, any new money limit."""
    items = [{"what": e.get("what"), "musts": e.get("musts")} for e in _fcc_entries(brief)]
    got = _ask_json([LIGHT, LIGHT_SPARE], NOTICE_SYSTEM, "PRODUCTS: %s\nNEWS: %s" % (_fcc_json(items), note), 300)
    return got if isinstance(got, dict) else {}


# ---------------------------------------------------------------- several products and the decline proof

_FCE_PACE = ("Supervisor here: {n} products are requested and the cart holds {held} of them after {used} steps of "
             "this search; at this pace the last ones get no steps. Put your best find so far for a product not "
             "yet in the cart into it now (requested quantity), then turn to the next product. Every product needs "
             "a line before the order.\n{state}")


def _fce_held(ep):
    """Distinct listings in the cart: the cart view when it is current, else this agent's own cart calls."""
    if ep.cart_obs is not None and not ep.cart_dirty:
        return len(_cart_ids(ep))
    return len([lid for lid, n in ep.cart_set.items() if n > 0])


def _fce_behind(count, held, used, room):
    """Pure: with room steps shared evenly by count products, more of them should be carted after used steps than
    are held."""
    if count < 2 or held >= count or room <= 0:
        return False
    return used * count // room > held


def _fce_pace_note(cx, lp):
    """Several products: a note when the cart falls behind an even share of steps per product; once per cart count."""
    ep = cx.ep
    count = len(_fcc_entries(cx.brief))
    held = _fce_held(ep)
    used = ep.step - lp.start
    room = ep.max_steps - lp.start - END_RESERVE - 4
    if lp.wrap_told or lp.paced > held or not _fce_behind(count, held, used, room):
        return None
    lp.paced = held + 1
    return _fcd_msg("user", _FCE_PACE.format(n=count, held=held, used=used, state=cx.state(lp)))


def _fce_conditions(entry):
    """Pure: a product's stated musts as (property words, value words); None when one is not "property: value"."""
    conds = []
    for must in entry.get("musts") or []:
        prop, sep, value = str(must).partition(":")
        if not sep or not _words(prop) or not _words(value):
            return None
        conds.append((_words(prop), _words(value)))
    return conds


def _fce_facts(rec):
    """Pure: normalised property key -> the words of its value, for every property a stored page records."""
    facts = {}
    for key, val in _fcb_page_facts(rec):
        facts.setdefault(key, set()).update(_words(val))
    return facts


def _fce_shows(facts, prop, value):
    """Pure: a page property sharing a word with the property name carries every value word, and they make at least
    half of its words (a value listing several options proves none of them)."""
    return any(prop & _words(key) and value <= words and 2 * len(value) >= len(words)
               for key, words in facts.items())


def _fce_fits_told(facts, prop, reply):
    """Pure: the page's value for the asked property is made of words the shopper's reply used."""
    want = _words(prop)
    return any(want <= _words(key) and words and words <= _words(reply) for key, words in facts.items())


def _fce_item_of(ep, lid, brief, names):
    """The requested product an opened listing belongs to: by the probe that found it, else by its category."""
    label = ep.probe_of.get(lid) or _category_value(ep, lid, brief)
    return _fcc_match(label, names) if label else None


def _decline_proof(ep, brief):
    """Lines (listing -> count) when, for every requested product, an opened page in stock shows each stated must
    and each told need word for word (the cheapest such page per product) and a total budget holds; else None."""
    entries = [e for e in _fcc_entries(brief) if e.get("what")]
    names = [" ".join(str(e["what"]).split()) for e in entries]
    skip_gone = (brief or {}).get("skip_sold_out", True)
    lines, total = {}, 0.0
    for name, entry in zip(names, entries):
        conds = _fce_conditions(entry)
        if conds is None:
            return None
        told = [(prop, reply) for item, prop, reply in ep.told if _fcc_match(item, names) == name]
        best = None
        for lid in ep.viewed:
            rec = ep.listings.get(lid) or {}
            price = _listing_price(rec)
            if price is None or (skip_gone and _sold_out(rec)) or _fce_item_of(ep, lid, brief, names) != name:
                continue
            facts = _fce_facts(rec)
            fits = all(_fce_shows(facts, p, v) for p, v in conds) and all(_fce_fits_told(facts, p, r) for p, r in told)
            if fits and (best is None or price < best[0]):
                best = (price, lid)
        if best is None:
            return None
        qty = _number(entry.get("qty")) or 1.0
        lines[best[1]] = qty
        total += best[0] * qty
    cap = _fcc_total_cap(brief)
    return lines if lines and (cap is None or total <= cap) else None


def _fce_keep_cart(cx, lp, kind):
    """A decline no strong-model check backed, while the cart holds a line for every requested product."""
    return (kind == "decline" and not lp.backed_decline and cx.ep.carted
            and _fce_held(cx.ep) >= max(1, len(_fcc_entries(cx.brief))))


# ---------------------------------------------------------------- closing holds: (applies, act) in order

def _fcd_tell(talk, cid, hold, note):
    talk += [_fcd_msg("tool", hold, cid), _fcd_msg("user", note)]
    return "next"


def _fcd_is_widen(cx, lp, kind, forced):
    ep = cx.ep
    early = ep.step < 0.55 * ep.max_steps and ep.left > 10
    return bool(kind) and early and not lp.widened and (kind == "decline" or cx.cheap)


def _fcd_do_widen(cx, lp, talk, cid, kind):
    lp.widened = True
    verb = "decline" if kind == "decline" else "settle"
    return _fcd_tell(talk, cid, _FCD_HOLD, (
        _FCD_BOSS + "with " + str(cx.ep.left) + " steps still available, broaden the hunt first, and only then "
        + verb + ". Ideas: category-filtered searching where some tool supports it (read deeper result pages "
        "too), reworded queries, and unopened property pages of the lowest-priced candidates. Hold back about 6 "
        "steps for cart, a final check and closing; decide after that."))


def _fcd_is_place(cx, lp, kind, forced):
    return (kind == "decline" and bool(lp.pick_lines) and not lp.placed
            and cx.ep.left >= len(lp.pick_lines) + 2)


def _fce_full(lines, brief):
    """Pure: an order that has a line for every requested product."""
    return bool(lines) and len(lines) >= max(1, len(_fcc_entries(brief)))


def _fcd_can_place(cx, lp):
    """A strong-model order covering every product, not yet carried out twice, with steps for it, whose lines'
    pages contradict no wish the shopper told (such a pick goes to the model, where the cart check sees it)."""
    return (_fce_full(lp.pick_lines, cx.brief) and lp.code_places < 2
            and _fcd_room(cx.ep, len(lp.pick_lines) + 3) and not _fcf_cart_locked(cx.ep, cx.roles, cx.brief)
            and not any(_fcf_contradicts(_fce_facts(cx.ep.listings.get(lid) or {}), p, r)
                        for lid in lp.pick_lines for p, r in _fcf_told_for(cx.ep, cx.brief, lid)))


def _fcd_code_place(cx, lp, talk, why):
    """Carry a strong-model order out in code (cart, basket report, purchase). When the episode stays open, the
    model gets the shop's answer and any shopper update that came with it. Returns 'stop' or 'next'."""
    ep = cx.ep
    lp.code_places += 1
    seen = len(ep.notes)
    lines = ", ".join("%s x%s" % (i, format(n, "g")) for i, n in lp.pick_lines.items())
    if not _commit_lines(ep, cx.roles, cx.brief, lp.pick_lines):
        talk.append(_fcd_msg("user", (
            _FCD_BOSS + why + ". Make the cart hold exactly these lines and nothing else: " + lines + "; view it, "
            "then close with the purchase.\n" + cx.state(lp))))
        return "next"
    reply = _finish_basket(ep, cx.roles, cx.brief, lp.pick_lines)
    lp.msgs = max(lp.msgs, len(ep.qa))  # the finish may have sent messages
    if ep.finished:
        return "stop"
    talk.append(_fcd_msg("user", (
        _FCD_BOSS + why + ", so I carted them and submitted the order on your behalf: " + lines +
        ". The shop answered: " + _compact_obs(reply)[:600] +
        "\nGo through that answer, repair anything it flags, and close once more.\n" + cx.state(lp))))
    if ep.notes[seen:]:
        talk.append(_relay(ep.notes[seen:]))
    return "next"


def _fcd_do_place(cx, lp, talk, cid, kind):
    lp.placed = True
    talk.append(_fcd_msg("tool", "Held back: not run.", cid))
    return _fcd_code_place(cx, lp, talk, "the senior review had fully qualifying listings")


def _fcd_is_proof(cx, lp, kind, forced):
    return (kind == "decline" and not lp.proof_told and not forced
            and cx.ep.left >= len(_fcc_entries(cx.brief)) + 3)


def _fcd_do_proof(cx, lp, talk, cid, kind):
    lines = _decline_proof(cx.ep, cx.brief)
    if not lines:
        return None
    lp.proof_told = True
    shown = ", ".join("%s x%s at %s each" % (lid, format(n, "g"), format(_listing_price(cx.ep.listings[lid]), "g"))
                      for lid, n in lines.items())
    return _fcd_tell(talk, cid, _FCD_HOLD, (
        _FCD_BOSS + "before ending without a purchase: these opened listings show every stated requirement, and "
        "every need the shopper told, on their own pages: " + shown + ". Unless a later shopper line rules one "
        "out, make the cart hold exactly these lines and buy.\n" + cx.state(lp)))


def _fcd_is_cart_fix(cx, lp, kind, forced):
    return (kind == "decline" and cx.ep.carted and bool(cx.removal) and not lp.cart_fix_told
            and not forced and cx.ep.left > 2)


def _fcd_do_cart_fix(cx, lp, talk, cid, kind):
    lp.cart_fix_told = True
    return _fcd_tell(talk, cid, _FCD_HOLD, (
        _FCD_BOSS + cx.removal + " Unwanted lines come out this way; after that the cart lists only what is to "
        "be bought, and you finish by buying. Finishing with nothing bought fits only if none of the seen listings "
        "passes every requirement.\n" + cx.state(lp)))


def _fcd_is_push(cx, lp, kind, forced):
    return kind == "decline" and not lp.decline_pushed and not forced and not lp.wrap_told and cx.ep.left > 8


def _fcd_do_push(cx, lp, talk, cid, kind):
    lp.decline_pushed = True
    _log, opened = _reach(cx.ep, cx.brief, cx.roles, min(6, cx.ep.left - END_RESERVE - 8), wide=True)
    ran = (" I just ran more searches (other names, each stated value left out in turn) and opened: " +
           ", ".join(opened) + "; look at those pages first.") if opened else ""
    return _fcd_tell(talk, cid, _FCD_HOLD, (
        _FCD_BOSS + "a no-purchase ending is usually the wrong call." + ran + " You have " + str(cx.ep.left) +
        " steps. Step A - revisit what you opened: a match also counts when that very property shows the value "
        "under a different label, through a synonym, or converted to other units; buy whatever passes every "
        "check. Step B - if nothing does, try two or three fresh searches: the category filter keyed on a wanted "
        "value, a reworded query, a later result page. Open the strongest pages and decide, holding back about "
        "6 steps for cart and close."))


def _fcd_is_sweep(cx, lp, kind, forced):
    return kind == "order" and cx.cheap and not lp.swept and not forced and cx.ep.left >= END_RESERVE + 10


def _fcd_do_sweep(cx, lp, talk, cid, kind):
    lp.swept = True
    if _fcd_sweep_setup(cx.ep, cx.brief) is None:
        # no price filter in this shop: search the carted names; the unopened-cheaper check runs next
        _fcd_word_sweep(cx.ep, cx.brief, cx.roles, cx.view_cart)
        return None
    note = _cheaper_sweep(cx.ep, cx.brief, cx.roles, cx.view_cart)
    if not note:
        return None
    return _fcd_tell(talk, cid, _FCD_HOLD, _FCD_BOSS + note + "\n" + cx.state(lp))


def _fcd_is_cheaper(cx, lp, kind, forced):
    return (kind == "order" and cx.cheap and lp.cheaper_checks < 2 and not forced
            and cx.ep.left > END_RESERVE + 5)


def _fcd_do_cheaper(cx, lp, talk, cid, kind):
    seen = _unchecked_cheaper(cx.ep, cx.brief)
    if not seen:
        return None
    lp.cheaper_checks += 1
    listed = ", ".join("%s (%s)" % (i, format(p, "g")) for i, p in seen)
    return _fcd_tell(talk, cid, _FCD_HOLD, (
        _FCD_BOSS + "the goal is the lowest price. Your results showed these listings under your pick's price, "
        "none opened so far: " + listed + ". Open them (one turn may hold several page calls). Should any satisfy "
        "each stated requirement and each need the shopper mentioned, replace the current line with the cheapest "
        "of those (same quantity, old line removed); should none, close once more.\n" + cx.state(lp)))


def _fcd_is_review(cx, lp, kind, forced):
    if not kind or forced or cx.ep.left <= 2 or _elapsed() >= CLOCK_WRAP:
        return False
    spent, (_turns, soft, hard) = _spent(), _caps()
    if kind == "decline" and lp.reviews <= MAX_REVIEWS and spent < hard:
        return True
    return lp.reviews < MAX_REVIEWS and spent < soft


def _fcd_do_review(cx, lp, talk, cid, kind):
    ep = cx.ep
    if kind != "decline" and not lp.refreshed:
        lp.refreshed = bool(_refresh_cart(ep, cx.roles, cx.view_cart))
    lp.reviews += 1
    verdict = _review(ep, cx.request, cx.brief, kind)
    if verdict:
        lp.pick_lines = _verdict_lines(ep, cx.brief, verdict)
        lp.backed_decline = lp.backed_decline or _fcd_low(verdict.get("close_as", "")) == "decline"
        lp.reason = str(verdict.get("reason") or lp.reason)
        better = lp.pick_lines and lp.pick_lines != _fcc_held_lines(ep)
        if kind != "decline" and _fcd_low(verdict.get("verdict", "")) == "fix" and better and _fcd_can_place(cx, lp):
            talk.append(_fcd_msg("tool", _FCD_HOLD_CHECK, cid))
            return _fcd_code_place(cx, lp, talk, "the senior review chose a better basket")
    extra = None
    if verdict and kind != "decline" and _fcd_low(verdict.get("close_as", "order")) != "decline":
        extra = _table_fix(verdict, _cart_ids(ep))
    note = _fcd_review_hold(verdict, kind, extra, ep.cart_obs, cx.state(lp))
    if not note:
        return None
    return _fcd_tell(talk, cid, _FCD_HOLD_CHECK, "Supervisor note. " + note)


_FCD_HOLDS = (
    (_fcd_is_widen, _fcd_do_widen),
    (_fcd_is_place, _fcd_do_place),
    (_fcd_is_proof, _fcd_do_proof),
    (_fcd_is_cart_fix, _fcd_do_cart_fix),
    (_fcd_is_push, _fcd_do_push),
    (_fcd_is_sweep, _fcd_do_sweep),
    (_fcd_is_cheaper, _fcd_do_cheaper),
    (_fcd_is_review, _fcd_do_review),
)


# ---------------------------------------------------------------- turn loop: the shell

def _fcd_run_batch(cx, lp, talk, batch, content):
    """Several page openings in one turn: each runs in order, one step each."""
    ep = cx.ep
    for n, c in enumerate(batch):
        if not c.get("id"):
            c["id"] = "call_%s_%s_%s" % (ep.step, lp.idle_steps, n)
    talk.append(_fcd_msg("assistant", content, calls=batch))
    relayed = []
    for c in batch:
        tool = c["function"]["name"]
        sent = _fcd_shaped(ep, tool, _parse_args(c))
        key = tool + json.dumps(sent, sort_keys=True)
        how = _fcd_batch_step(key, ep)
        if how == "run":
            before = ep.step
            out = _compact_obs(ep.call(tool, sent))
            relayed.extend(ep.fresh)
            if ep.step > before:
                lp.idle_steps = 0
                ep.memo[key] = out
        elif how == "memo":
            out = "Repeat of an earlier call (no step used); it returned:\n" + ep.memo[key]
        else:
            out = "Refused: the remaining steps are kept for the cart and closing, not more pages."
        talk.append(_fcd_msg("tool", out, c["id"]))
    talk[-1]["content"] = talk[-1]["content"] + "\n" + cx.state(lp)
    lp.streak = 0
    if relayed:
        talk.append(_relay(relayed))
        _fcd_after_notice(cx, lp, relayed, talk)


def _fcd_after_cart(cx, lp, args, obs):
    """Bookkeeping after a cart call that took a step."""
    ep = cx.ep
    ep.cart_dirty = ep.carted = True
    ep.cart_rev += 1
    lp.refreshed = False
    ep.memo.clear()  # results and pages may differ once the cart holds something
    failed = isinstance(obs, dict) and obs.get("error")
    if failed:
        return
    if not cx.view_cart and isinstance(obs, (dict, list)):
        ep.cart_obs, ep.cart_dirty = obs, False  # the shop has no cart view: the cart tool's reply is the newest cart
    lid, count = _fcd_cart_line(args, ep.listings)
    if lid:
        ep.cart_set[lid] = _fcf_count_after(_cart_spec(ep, cx.roles, cx.brief), args, ep.cart_set.get(lid, 0.0),
                                            float(count))


def _fcf_count_after(spec, args, before, count):
    """Pure: a line's count after a cart call, in the shop's own semantics: the call carries the remove arguments
    (take away), the count adds (add), else the count is the new quantity."""
    if spec and spec["remove_args"] and all(args.get(k) == v for k, v in spec["remove_args"].items()):
        return max(0.0, before - count)
    if spec and spec["adds"]:
        return before + count
    return count


def _fcd_after_close(cx, lp, talk, kind, obs):
    """A close the shop did not end the episode on: maybe let the strong model choose again, then explain. True
    when a strong-model decision was taken on the shop's answer."""
    ep = cx.ep
    if _fcf_cart_locked(ep, cx.roles, cx.brief):
        closing = _fcc_purchase_call(ep, cx.roles, cx.brief) if kind != "decline" else None
        if closing and not lp.relocked:
            lp.relocked = True  # one more try of the reported basket; shopper words on a refusal often only ease
            obs = _fcc_run(ep, *closing)
            if ep.finished:
                return
        talk.append(_fcd_msg("user", _FCD_BOSS + _FCD_LOCKED + " The shop answered: " + _compact_obs(obs)[:400]
                             + "\n" + cx.state(lp)))
        return
    pick = None
    if lp.redecides < 2 and _spent() < _caps()[2] and ep.left > 1 and _elapsed() < CLOCK_WRAP:
        lp.redecides += 1
        why = "a closing call was turned down by the shop, whose answer reads: " + _compact_obs(obs)[:400]
        pick = _decide(ep, cx.request, cx.brief, why)
        if pick:
            lp.pick_lines = _pick_lines(pick)
            lp.backed_decline = lp.backed_decline or _fcc_action(pick) == "decline"
            if _fcd_can_place(cx, lp):
                _fcd_code_place(cx, lp, talk, "after the shop turned the close down, a senior decision chose these "
                                "lines")
                return True
    steps = _cart_steps(ep, pick) if pick else ""
    if pick:
        lp.reviews = max(lp.reviews, MAX_REVIEWS - 1)
    talk.append(_fcd_msg("user", _fcd_still_open_note(cx.cheap and kind == "order", pick, steps)))
    return bool(pick)


def _fcd_record(cx, lp, tool, args, obs, moved):
    """State updates after a call ran, in the base order: messages, cart, cart view."""
    ep = cx.ep
    if moved:
        lp.idle_steps = 0
    if tool in cx.messengers:
        lp.msgs += 1
        asked = str(next(iter(args.values()), ""))
        ep.qa.append((asked, _reply_text(obs)))
        held = _fcc_held_lines(ep) if ep.carted else {}
        if held and all(lid in json.dumps(args) for lid in held):
            ep.stated_rev = ep.cart_rev  # the model's own message names every cart line: its basket report
    if moved and tool in cx.cart_tools:
        _fcd_after_cart(cx, lp, args, obs)
    if tool == cx.view_cart:
        ep.cart_obs, ep.cart_dirty = obs, False


VETO_SYSTEM = """Answer only with JSON of the form {"keep": true or false, "swap_to": a listing id or null, "reason":
one sentence}. An assistant buying in a simulated web store is about to put a listing in the cart for one requested
product. Keep its pick unless a fact on that listing's own page contradicts a condition of the request or a wish the
customer told, or, when the customer wants the lowest price, a cheaper OPENED listing of the same kind passes every
condition and wish. A listing whose page, title and description say nothing about a property that other opened pages
of its kind record does not pass that condition. Swap only to an opened listing named in the input."""


def _fcf_contradicts(facts, prop, reply):
    """Pure: the page records the asked property and none of its values is made of the reply's words."""
    want = _words(prop)
    held = [words for key, words in facts.items() if want <= _words(key) and words]
    return bool(held) and not any(words <= _words(reply) for words in held)


def _fcf_told_for(ep, brief, lid):
    """(property, reply) for each wish the shopper told about the requested product this listing serves."""
    names = [" ".join(str(e["what"]).split()) for e in _fcc_entries(brief) if e.get("what")]
    item = _fce_item_of(ep, lid, brief, names) if names else None
    if item is None and len(names) == 1:
        item = names[0]  # a single requested product: every listing serves it
    return [(prop, reply) for owner, prop, reply in ep.told if not names or _fcc_match(owner, names) == item]


def _vet_gate(ep, brief, lid, cheap):
    """Code gate for a senior look at a new cart line: its page contradicts a wish the shopper told for that
    product, or (lowest price wanted) a cheaper opened, in-stock listing of the same category contradicts none."""
    told = _fcf_told_for(ep, brief, lid)
    if any(_fcf_contradicts(_fce_facts(ep.listings.get(lid) or {}), p, r) for p, r in told):
        return True
    price = _listing_price(ep.listings.get(lid) or {})
    if not cheap or price is None:
        return False
    kind = _category_value(ep, lid, brief)
    for other in ep.viewed:
        rec = ep.listings.get(other) or {}
        cost = _listing_price(rec)
        if other == lid or cost is None or cost >= price or _sold_out(rec):
            continue
        if kind is not None and _category_value(ep, other, brief) != kind:
            continue
        if not any(_fcf_contradicts(_fce_facts(rec), p, r) for p, r in told):
            return True
    return False


def _fcd_vet(cx, lp, args):
    """Before a new listing enters the cart: when the code gate sees a reason, the strong model may swap it for an
    opened listing (same count); the light pick stands otherwise. At most twice per episode and only while spend is
    under the soft cap. Returns the arguments to run and a note for the model."""
    ep = cx.ep
    lid, count = _fcd_cart_line(args, ep.listings)
    held = set(_cart_ids(ep)) | {k for k, n in ep.cart_set.items() if n > 0}
    if not lid or lid in held or not count or count <= 0:
        return args, ""
    if lid in lp.vetoed and lp.vetoed[lid] not in held:
        # the senior check already replaced this listing: keep its choice
        new = lp.vetoed[lid]
        return {k: (new if v == lid else v) for k, v in args.items()}, (
            "Supervisor note: %s was replaced by %s in an earlier senior check; %s goes in." % (lid, new, new))
    if lp.vets >= 2:
        return args, ""
    if _spent() >= _caps()[1] or _elapsed() >= CLOCK_WRAP - 60 or not _fcd_room(ep, 6):
        return args, ""  # a slow strong call must not eat the minute before the wrap-up
    if not _vet_gate(ep, cx.brief, lid, cx.cheap):
        return args, ""
    lp.vets += 1
    user = (_fcc_dossier(ep, cx.request, cx.brief) + "\n\nTHE ASSISTANT IS ABOUT TO CART: %s x%s" %
            (lid, format(count, "g")))
    got = _ask_json(list(_FCC_STRONG), VETO_SYSTEM, user, max_tokens=1500)
    if not isinstance(got, dict) or got.get("keep") is not False:
        return args, ""
    new = str(got.get("swap_to") or "")
    if new == lid or new not in ep.viewed:
        return args, ""
    swapped = {k: (new if v == lid else v) for k, v in args.items()}
    lp.vetoed[lid] = new
    return swapped, ("Supervisor note: a senior check put %s in the cart instead of %s (%s)." %
                     (new, lid, str(got.get("reason") or "")[:200]))


_FCD_LOCKED = ("the basket was reported to the shopper after its last change and no message is left, so a cart "
              "change now would leave the purchase unreported. Keep the cart exactly as it is and close again.")
_FCF_DECLINE = "Before I close without buying: %s I am ending without placing an order."
_FCF_NO_FIT = "none of the listings I checked meets every requirement."


def _fcd_pre_view(cx, lp, talk, cid):
    """Before the basket report: one look at the cart after its last change, so a shopper update riding on that
    reply shows up while the cart can still follow it (once per episode). True when the close was held."""
    ep = cx.ep
    if lp.previewed or not cx.view_cart or not ep.cart_dirty:
        return False
    if not _fcc_statement_due(ep, cx.roles, cx.brief, room=4):
        return False
    lp.previewed = True
    ep.cart_obs, ep.cart_dirty = _fcc_run(ep, cx.view_cart, {}), False
    fresh = list(ep.fresh)
    if not fresh:
        return False
    talk.append(_fcd_msg("tool", _FCD_HOLD + " The cart was viewed once more before buying, and the shop's reply "
                         "carries a shopper update. Check the cart against it, adjust it if needed, then close "
                         "again.\nCart: " + _compact_obs(ep.cart_obs)[:600] + "\n" + cx.state(lp), cid))
    talk.append(_relay(fresh))
    _fcd_after_notice(cx, lp, fresh, talk)
    return True


def _fcd_after_notice(cx, lp, notes, talk=None):
    """News from the shopper or the store: acknowledge it when a procedure wants word, and once per episode search
    again under the changed conditions (without dropped values) and open the best new rows."""
    text = "Noted: %s I will check the basket against it." % str(notes[0])[:200]
    if _procedure_message(cx.ep, cx.roles, cx.brief, "after_notice", text):
        lp.msgs = max(lp.msgs, len(cx.ep.qa))
    ep = cx.ep
    if talk is None or lp.notice_searched or not _fcd_room(ep, END_RESERVE + 10) or _elapsed() >= CLOCK_WRAP:
        return
    news = _read_notice(cx.brief, " ".join(map(str, notes))[:600])
    dropped = [str(d) for d in news.get("dropped") or [] if str(d or "").strip()]
    if not dropped and _number(news.get("budget")) is None:
        return
    lp.notice_searched = True
    _log, opened = _reach(ep, cx.brief, cx.roles, min(6, ep.left - END_RESERVE - 8), dropped, wide=bool(dropped))
    if opened:
        shown = ", ".join("%s (%s)" % (lid, format(_listing_price(ep.listings.get(lid) or {}) or 0, "g"))
                          for lid in opened)
        talk.append(_fcd_msg("user", _FCD_BOSS + "after that update I searched again under the changed conditions "
                             "and opened: " + shown + ". Weigh them against the cart under the new conditions; "
                             "when the lowest price is the goal, the cheapest listing that passes everything "
                             "wins.\n" + cx.state(lp)))


def _fcd_turn(cx, lp, talk, call, content, dropped, forced):
    """One tool call from the model: gate it, maybe hold it, else run it. Returns 'stop' to end the loop."""
    ep = cx.ep
    call["id"] = call.get("id") or "call_%s_%s" % (ep.step, lp.idle_steps)
    cid = call["id"]
    tool = (call.get("function") or {}).get("name", "")
    raw = _parse_args(call)
    talk.append(_fcd_msg("assistant", content, calls=[call]))
    if tool not in ep.by_name:
        talk.append(_fcd_msg("tool", "There is no %s tool here. Available tools: %s" % (tool, ", ".join(ep.by_name)), cid))
        return None
    args = _fcd_shaped(ep, tool, raw)
    vetted = ""
    if tool in cx.cart_tools and _fcf_cart_locked(ep, cx.roles, cx.brief):
        talk.append(_fcd_msg("tool", "Refused without using a step: " + _FCD_LOCKED + "\n" + cx.state(lp), cid))
        return None
    if tool in cx.cart_tools:
        args, vetted = _fcd_vet(cx, lp, _ids_for_titles(ep, args))
    memo_key = tool + json.dumps(args, sort_keys=True)
    qkey = _fcd_query_key(tool, args, ep.seen_strings)
    gate = _fcd_gate(cx, lp, tool, memo_key, qkey)
    if gate is not None:
        if gate == "streak":
            lp.streak = 0
        talk.append(_fcd_msg("tool", _fcd_gate_text(gate, ep.memo.get(memo_key), cx.state(lp)), cid))
        return None

    kind = _closing_kind(cx.roles, tool, args, cx.brief)
    locked = kind and kind != "decline" and _fcf_cart_locked(ep, cx.roles, cx.brief)  # nothing left to repair
    if not locked and _fcd_decide_due(lp, ep, _spent(), kind):
        lp.decided = True
        pick = _decide(ep, cx.request, cx.brief)
        if pick:
            lp.pick_lines = _pick_lines(pick)
            lp.reason = str(pick.get("reason") or lp.reason)
            if _fcd_low(pick.get("action", "")) == "decline":
                lp.decline_pushed = lp.backed_decline = True  # the strong model chose to decline: no argument
            elif _fcd_can_place(cx, lp):
                talk.append(_fcd_msg("tool", "Held back: not run.", cid))
                out = _fcd_code_place(cx, lp, talk, "choosing time had come and a senior decision chose these lines")
                return "stop" if out == "stop" else None
            talk.append(_fcd_msg("tool", _fcd_pick_note(pick, _cart_steps(ep, pick), cx.state(lp)), cid))
            lp.reviews = MAX_REVIEWS
            return None

    if tool in cx.messengers and lp.msgs >= cx.msg_limit:
        talk.append(_fcd_msg("tool", "Message blocked: the allowance for shopper messages is used up, so proceed without asking.", cid))
        return None

    for applies, act in () if locked else _FCD_HOLDS:
        outcome = act(cx, lp, talk, cid, kind) if applies(cx, lp, kind, forced) else None
        if outcome is not None:
            return outcome if outcome == "stop" else None

    paying = _fcc_purchase_call(ep, cx.roles, cx.brief) if _fce_keep_cart(cx, lp, kind) else None
    if paying:
        # nothing backs this decline and the cart covers every product: buy it
        tool, args, kind = paying[0], _coerce_args(paying[1], _schema(ep.by_name[paying[0]])), "order"
    if kind and kind != "decline":  # an order, or a close the brief could not label
        if _fcd_pre_view(cx, lp, talk, cid):
            return None
        if _state_basket(ep, cx.roles, cx.brief):
            heard = _hear_statement(ep, cx.roles, cx.brief)
            lp.msgs = max(lp.msgs, len(ep.qa))
            swap = _decline_call(ep, cx.roles, cx.brief) if heard == "decline" else None
            if swap:
                tool, args, kind = swap[0], swap[1], "decline"
    if kind == "decline" and _procedure_message(ep, cx.roles, cx.brief, "before_decline", _FCF_DECLINE % (
            lp.reason or _FCF_NO_FIT)):
        lp.msgs = max(lp.msgs, len(ep.qa))
    before = ep.step
    obs = ep.call(tool, args)
    _fcd_record(cx, lp, tool, args, obs, ep.step > before)
    text = _compact_obs(obs)
    relayed = list(ep.fresh)
    if tool in cx.lookups:
        ep.memo[memo_key] = text
    order_note = vetted
    if tool in cx.searchers:
        lp.streak += 1
        lp.pages[qkey] = 1 + lp.pages.get(qkey, 0)
        order_note = _order_note(obs)
    else:
        lp.streak = 0
    talk.append(_fcd_msg("tool", _fcd_tool_text(text, order_note, ep, dropped, cx.state(lp)), cid))
    weighed = bool(kind and not ep.finished and _fcd_after_close(cx, lp, talk, kind, obs))
    if relayed:
        if not weighed:  # a senior decision taken on the shop's answer has already weighed that news
            talk.append(_relay(relayed))
        _fcd_after_notice(cx, lp, relayed, talk)
    return None


def _fcd_settle_offer(cx, lp, talk, forced, spent):
    """Pick the tools to offer this turn and post the matching notice; returns the tool list."""
    turns, _soft, hard = _caps()
    over = (spent >= hard or lp.turns >= turns or _written() >= _work_caps()[0]
            or _elapsed() >= CLOCK_WRAP)
    mode = _fcd_offer_mode(forced, bool(cx.closers), over, lp.idle_steps, lp.wrap_told)
    if mode in ("wrap", "idle"):
        lp.wrap_told = True
    if mode in _FCD_OFFER_NOTES:
        talk.append(_fcd_msg("user", _FCD_OFFER_NOTES[mode]))
    if mode is None:
        return cx.ep.tools
    return cx.closers if mode == "final" else cx.narrow


def _fcd_tool_calls(lp, talk, reply, offer):
    """The calls of a model reply, retrying once with tool_choice forced after a repeat; None ends the loop."""
    calls = reply.get("tool_calls") or []
    said = reply.get("content") or ""
    if calls:
        return calls, said
    lp.idle += 1
    if lp.idle > 4:
        return None
    if lp.idle >= 2:
        nudge = [_fcd_msg("assistant", said or _FCD_SILENT), _fcd_msg("user", _FCD_NUDGE)]
        again = _chat_forced(talk + nudge, offer) or {}
        calls, said = again.get("tool_calls") or [], again.get("content") or ""
    if not calls:
        talk += [_fcd_msg("assistant", said or _FCD_SILENT), _fcd_msg("user", _FCD_NUDGE)]
    return calls, said


def _run(ep: Episode, state: dict) -> None:
    request = ep.policy.get("query") or ""
    brief = _make_brief(ep, request)
    if not isinstance(brief, dict):
        brief = None
    roles = _role_tools(brief, ep)
    state.update(roles=roles, brief=brief)
    _spend.caps = _lane_caps(len(_fcc_entries(brief)) or 1)
    ep.message_tools = set(roles.get("message", []))
    ep.page_tools = set(roles.get("view", []) + roles.get("compare", []))
    ep.max_steps = _fcd_step_cap(brief, ep.max_steps) or ep.max_steps
    probe_notes = _probe(ep, brief, roles)
    _read_fields(ep)
    _ask(ep, brief, roles, request)
    found, _opened = _reach(ep, brief, roles, _fcg_budget(ep))
    probe_notes = probe_notes + found
    probe_notes = probe_notes + _follow_refs(ep, brief, roles, request)
    cx = _fcd_Ctx(ep, request, brief, roles)
    lp = _fcd_Loop(len(ep.qa))
    lp.start = ep.step
    cx.removal = _removal_hint(ep, roles)
    talk = [_fcd_msg("system", LOOP_SYSTEM), _fcd_msg("user", _fcd_intro(cx, probe_notes, cx.state(lp)))]

    while not ep.finished and ep.step < ep.max_steps:
        forced = ep.left <= 1
        spent = _spent()
        pace = _fce_pace_note(cx, lp)
        if pace:
            talk.append(pace)
        offer = _fcd_settle_offer(cx, lp, talk, forced, spent)
        blown = spent >= 1.15 * _caps()[2]
        if blown and not ep.carted:
            lp.over += 1  # a few cheap turns to cart the best find rather than leave empty-handed
        if _fcd_halt(lp, _written(), blown, ep.carted):
            break
        lp.turns += 1
        reply = _chat([LIGHT, LIGHT_SPARE], talk, tools=offer, max_tokens=3000)
        if reply is None:
            lp.idle += 1
            if lp.idle > 3:
                break
            time.sleep(3)
            continue
        got = _fcd_tool_calls(lp, talk, reply, offer)
        if got is None:
            break
        calls, content = got
        if not calls:
            continue
        lp.idle = 0
        lp.idle_steps += 1
        batch = _fcd_page_batch(calls, cx.pagers, lp.wrap_told)
        if batch is not None:
            _fcd_run_batch(cx, lp, talk, batch, content)
            continue
        first = calls[0] if isinstance(calls[0], dict) else {}
        if _fcd_turn(cx, lp, talk, first, content, len(calls) > 1, forced) == "stop":
            break


_FCD_RELAY = (
    "Supervisor relay. Shops in this task deliver the shopper's updates inside tool output; the last output "
    "contained this text, outside every listing:\n{0}\nWork out which case applies.\n"
    "(a) Shopper update - a new money limit, a wish changed or added. Treat it as genuine and binding from now "
    "on: compare cart, limit and wishes with it, and edit the cart where they no longer agree. A wish that was "
    "dropped or loosened can let a listing you set aside pass now: when the lowest price is the goal, compare "
    "the cheaper listings you have seen again.\n"
    "(b) Price notice only - read the current figure on the carted product's page; the product may stay while the "
    "order sum still respects the limit and nothing beats it for the stated goal; then close once more.\n"
    "(c) Plain system or error line - ignore it and go on.")


def _relay(fresh: list[str]) -> dict:
    quoted = "\n".join('"%s"' % n for n in fresh)
    return _fcd_msg("user", _FCD_RELAY.format(quoted))


__all__ = ["agent_main"]
