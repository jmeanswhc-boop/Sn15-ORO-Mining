"""Tern: a shopping agent for ORO store worlds.

A quick model drives the store through the world's own tools. Around it, code keeps the books: what was asked,
which listings were seen and read, what the cart holds, how many steps, seconds and dollars are left. A careful
model reads the request, chooses the questions for the shopper and reviews the cart before an order and the
evidence before an ending without a purchase.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.agent.proxy_client import ProxyClient

QUICK = "anthropic/claude-haiku-5.5"
CAREFUL = "anthropic/claude-sonnet-5.5"
BACKUP = "openai/gpt-6-luna"

# model turns of the store loop and model calls in all; a task with a longer step allowance gets more of both
TURNS_BASE = 42
CALLS_BASE = 60
# dollars per task: above SOFT the reviews become selective, at LOOP_STOP the store loop ends, at HARD_STOP no
# model call goes out at all (every task of a run is paid from one key)
SPEND_SOFT = 0.055
SPEND_LOOP_STOP = 0.078
SPEND_HARD_STOP = 0.085
SPEND_QUICK_FIRST = 0.035
# one run key pays for all 90 race episodes and for the shopper's own replies: an episode's purse is sized from its
# task (more for several items and a long step allowance), so the run's mean stays well under the key (race 13: Tern
# averaged about 0.065 USD an episode under the flat stops, the key ran dry with a quarter of the tasks still running,
# and the validator stopped the sandbox)
PURSE_BASE = 0.048
PURSE_TOP = 0.058
# seconds: the network ends a task at 300 s whatever its state, so the agent keeps its own clock
PLAN_SECONDS = 250.0
WRAP_UP = 70.0
NO_NEW_CALL = 28.0
ASK_WINDOW = 45.0
# properties one message names, best first. Race 14: the customer answers about the property it holds back wherever a short
# list names it (the first names most surely), so three messages cover a dozen names instead of three. A puzzled or
# crowded reply halves that item's next list.
NAMES_PER_ASK = 4
RETRY_PAUSES = (4, 10, 20, 35)
RETRY_BUDGET = 45
ROW_CHARS = 450
REPLY_CHARS = 15000
CHAT_CHARS = 110000
GATE_TRIES = 3
SELF_PAGES = 6
CHEAPER_READS = 3

_model_line = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_store_line = ProxyClient(timeout=50, max_retries=3, retry_delay=1)


# ======================================================================== small tools for text, numbers and schemas


def flat(val: Any) -> str:
    """Compact JSON text of a value (plain str() when it cannot be encoded)."""
    try:
        return json.dumps(val, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(val)


def shorten(val: Any, room: int, level: int = 0) -> Any:
    """The same structure with every long string cut to `room` characters (and very deep parts dropped)."""
    if level >= 8:
        return "..."
    if isinstance(val, str):
        return val[:room] + "..." if len(val) > room else val
    if isinstance(val, list):
        return [shorten(part, room, level + 1) for part in val]
    if isinstance(val, dict):
        return {key: shorten(part, room, level + 1) for key, part in val.items()}
    return val


def read_object(text: Any) -> dict[str, Any] | None:
    """The JSON object a model reply holds: the whole reply, a fenced block, or the outermost braces."""
    if not isinstance(text, str) or not text.strip():
        return None
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    candidates = [body]
    start, stop = body.find("{"), body.rfind("}")
    if start >= 0 and stop > start:
        candidates.append(body[start:stop + 1])
    for chunk in candidates:
        try:
            parsed = json.loads(chunk)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def plain(key: Any) -> str:
    """A key or label in plain lower-case words: camelCase, snake_case and dotted keys come apart."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(key))
    return " ".join(re.split(r"[\s_.\-]+", spaced)).strip().lower()


def roots(text: Any) -> frozenset[str]:
    """The words of a text with a plural s dropped from longer words."""
    out = set()
    for word in plain(text).split():
        out.add(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return frozenset(out)


def same_property(asked: frozenset[str], key: frozenset[str]) -> bool:
    """Whether a key names a property: the same words, or one word set holds the other (two words at least for the
    key inside the property)."""
    if not asked or not key:
        return False
    if asked <= key:
        return True
    return len(key) >= 2 and key <= asked


def as_list(val: Any) -> list[Any]:
    if isinstance(val, list):
        return val
    return [val] if val and isinstance(val, (str, dict)) else []


def whole(val: Any) -> int | None:
    """A whole number written as an int, an integral float or digits; None otherwise (booleans are not numbers)."""
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val) if val.is_integer() else None
    if isinstance(val, str):
        hit = re.fullmatch(r"\s*(\d+)(?:\.0+)?\s*", val)
        return int(hit.group(1)) if hit else None
    return None


def amount(val: Any) -> float | None:
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str) and re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", val):
        return float(val)
    return None


def kinds_of(sch: dict[str, Any]) -> list[str]:
    declared = sch.get("type")
    if isinstance(declared, str):
        return [declared]
    return [k for k in (declared or []) if isinstance(k, str)]


_KIND_TEST = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: is_num(v),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def kind_ok(val: Any, kind: str) -> bool:
    test = _KIND_TEST.get(kind)
    return True if test is None else bool(test(val))


def _coerce(val: Any, kind: str) -> tuple[bool, Any]:
    """Try to turn a value into a schema kind; (changed, new value)."""
    if kind == "integer":
        number = amount(val)
        if number is not None and number.is_integer():
            return True, int(number)
    elif kind == "number":
        number = amount(val)
        if number is not None:
            return True, int(number) if number.is_integer() and isinstance(val, str) and "." not in val else number
    elif kind == "string" and isinstance(val, (int, float)) and not isinstance(val, bool):
        return True, str(val)
    elif kind == "boolean" and isinstance(val, str) and val.strip().lower() in ("true", "false"):
        return True, val.strip().lower() == "true"
    elif kind == "array" and val is not None and not isinstance(val, list):
        return True, [val]
    return False, val


def conform(val: Any, sch: Any, where: str, problems: list[str]) -> Any:
    """Bend a value toward a JSON schema (types, enum spelling, length) and note what cannot be bent."""
    if not isinstance(sch, dict):
        return val
    target_lines = kinds_of(sch)
    if target_lines and not any(kind_ok(val, k) for k in target_lines):
        for kind in target_lines:
            changed, val = _coerce(val, kind)
            if changed:
                break
        if not any(kind_ok(val, k) for k in target_lines):
            problems.append(f"{where} must be {' or '.join(target_lines)}")
            return val
    allowed = sch.get("enum")
    if isinstance(allowed, list) and val not in allowed:
        if isinstance(val, str):
            val = next((o for o in allowed if isinstance(o, str) and o.lower() == val.strip().lower()), val)
        if val not in allowed:
            problems.append(f"{where} must be one of {allowed}")
    if isinstance(val, str):
        top, low = sch.get("maxLength"), sch.get("minLength")
        if isinstance(top, int):
            val = val[:top]
        if isinstance(low, int) and len(val) < low:
            problems.append(f"{where} is shorter than {low} characters")
    elif isinstance(val, (int, float)) and not isinstance(val, bool):
        floor, ceiling = sch.get("minimum"), sch.get("maximum")
        if isinstance(floor, (int, float)) and val < floor:
            problems.append(f"{where} may not be under {floor}")
        if isinstance(ceiling, (int, float)) and val > ceiling:
            problems.append(f"{where} may not be over {ceiling}")
    elif isinstance(val, list):
        most, least = sch.get("maxItems"), sch.get("minItems")
        if isinstance(most, int) and len(val) > most:
            problems.append(f"{where} takes {most} entries at most")
        if isinstance(least, int) and len(val) < least:
            problems.append(f"{where} takes {least} entries at least")
        inner = sch.get("items")
        if isinstance(inner, dict):
            val = [conform(part, inner, f"{where}[{n}]", problems) for n, part in enumerate(val)]
    elif isinstance(val, dict):
        params = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        problems.extend(f"{where}.{name} is required" for name in sch.get("required") or [] if name not in val)
        if sch.get("additionalProperties") is False:
            problems.extend(f"{where}.{name} is not a parameter" for name in val if name not in params)
        val = {name: conform(part, params[name], f"{where}.{name}", problems) if name in params else part
                 for name, part in val.items()}
    return val


def fill_args(args: dict[str, Any], form: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    fixed = conform(dict(args), form or {}, "args", problems)
    return (fixed if isinstance(fixed, dict) else dict(args)), problems


def looks_like_id(val: Any, keyed: bool = False) -> bool:
    """A listing id: a short token without spaces; a value under a key that is not named as an id needs a digit."""
    if not isinstance(val, str):
        return False
    token = val.strip()
    if " " in token or not 3 <= len(token) <= 64:
        return False
    return keyed or any(ch.isdigit() for ch in token)


def id_field(cards: list[dict[str, Any]]) -> str | None:
    """Which key identifies the rows: unique id-like strings, a key named as an id first, then the longest values."""
    ranked: list[tuple[tuple[bool, float], str]] = []
    for key in cards[0]:
        named = "id" in str(key).lower()
        vals = [row.get(key) for row in cards]
        if not all(looks_like_id(v, named) for v in vals):
            continue
        tokens = [v.strip() for v in vals]
        if len(tokens) > 1 and len(set(tokens)) != len(tokens):
            continue
        ranked.append(((named, sum(map(len, tokens)) / len(tokens)), key))
    return max(ranked)[1] if ranked else None


def holds_id(piece: Any, lid: str, level: int = 0) -> bool:
    """Whether some list inside a result carries a record with this id among its string values."""
    if level > 5:
        return False
    if isinstance(piece, dict):
        return any(holds_id(v, lid, level + 1) for v in piece.values())
    if isinstance(piece, list):
        for part in piece:
            if isinstance(part, dict) and lid in {str(x).strip() for x in part.values() if isinstance(x, str)}:
                return True
        return any(holds_id(part, lid, level + 1) for part in piece)
    return False


def attribute_block(record: Any, least: int = 2) -> dict[str, Any] | None:
    """The largest nested mapping of plain values in a listing record (at least `least` of them): its attributes."""
    if not isinstance(record, dict):
        return None
    blocks = []
    for part in record.values():
        if isinstance(part, dict):
            scalars = {k: v for k, v in part.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= least:
                blocks.append(scalars)
    return max(blocks, key=len) if blocks else None


def walk_strings(piece: Any, minimum_words: int = 2, level: int = 0) -> list[str]:
    """Every string of several words inside a result, in order."""
    if level > 4:
        return []
    if isinstance(piece, str):
        return [piece] if len(piece.split()) >= minimum_words else []
    parts = piece.values() if isinstance(piece, dict) else piece if isinstance(piece, list) else []
    collected: list[str] = []
    for part in parts:
        collected += walk_strings(part, minimum_words, level + 1)
    return collected


def is_num(v: Any) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float))


def free_text_params(params: dict[str, Any]) -> list[str]:
    """Parameters that take free text (strings without a fixed option list)."""
    return [name for name, sch in params.items() if isinstance(sch, dict) and "enum" not in sch and "string" in kinds_of(sch)]


def first_param(params: dict[str, Any], kind: str) -> str | None:
    return next(iter(name for name, sch in params.items() if isinstance(sch, dict) and kind in kinds_of(sch)), None)


def switch_param(params: dict[str, Any]) -> str | None:
    """A parameter with two or more fixed options (an add/remove mode, say)."""
    for name, sch in params.items():
        options = sch.get("enum") if isinstance(sch, dict) else None
        if isinstance(options, list) and len(options) > 1:
            return name
    return None


def tool_line(tool: dict[str, Any]) -> str:
    fn = tool["function"]
    return "- " + fn["name"] + ": " + str(fn.get("description", "")) + " | parameters: " + flat(fn.get("parameters") or {})


# ======================================================================== the two lines out: the store and the models


class StoreLink:
    """Environment calls for one episode. Every call carries the whole binding and the next step number; the same
    step with the same action may be resent safely, and a 409 that names the expected step re-synchronises."""

    def __init__(self, binding: dict[str, Any], log: list[dict[str, Any]]):
        self.binding = dict(binding)
        self.log = log
        self.step = 0
        self.finished = False
        self.unpaid = False  # the store's side refused for want of credit

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        body = {**self.binding, "step": self.step + 1, "action": {"name": name, "args": args}}
        for _ in range(4):
            result = _store_line.post_verbose("/environment/call", body)
            data = result.data
            if isinstance(data, dict):
                taken = data.get("step")
                self.step = taken if isinstance(taken, int) and not isinstance(taken, bool) else self.step + 1
                self.finished = self.finished or bool(data.get("finished"))
                return data
            fault = result.error or {}
            status, text = fault.get("status"), str(fault.get("body") or "")
            self.log.append({"kind": "store_fault", "status": status, "body": text[:200]})
            if status == 409:
                lowered = text.lower()
                if any(word in lowered for word in ("finish", "finaliz", "quarantin")):
                    self.finished = True
                    return None
                expected = re.search(r"expected (\d+)", text)
                if expected:
                    self.step = int(expected.group(1)) - 1
                elif "reuse" in lowered or "different action" in lowered:
                    self.step += 1
                else:
                    return None
                body["step"] = self.step + 1
                continue
            if status == 402:
                self.unpaid = True
            if status in (400, 401, 402, 403, 404, 422):
                return None
        return None


class ModelDesk:
    """Inference through the proxy for one episode, with the episode's own budgets: model calls, dollars and
    seconds. 402 stops all calls, 408/425/429/5xx are retried after a jittered pause (45 s of pauses at most)."""

    def __init__(self, log: list[dict[str, Any]]):
        self.log = log
        self.started = time.monotonic()
        self.spent = 0.0
        self.call_list = 0
        self.call_cap = CALLS_BASE
        self.paused = 0
        self.no_credit = False
        self.late = False
        self.purse = SPEND_LOOP_STOP

    def seconds_left(self) -> float:
        return PLAN_SECONDS - (time.monotonic() - self.started)

    def closed(self) -> bool:
        """No further model call may start: credit gone, call cap or dollar cap reached, or too little time left."""
        if self.seconds_left() < NO_NEW_CALL:
            self.late = True
        return self.no_credit or self.late or self.call_list >= self.call_cap or self.spent >= min(SPEND_HARD_STOP, self.purse + 0.007)

    def careful_first(self) -> list[str]:
        """Model order for a review: the careful model while this task is cheap so far, the quick one first after."""
        return [CAREFUL, QUICK] if self.spent < min(SPEND_QUICK_FIRST, 0.5 * self.purse) else [QUICK, CAREFUL]

    @staticmethod
    def guess(model: str, messages: list[dict[str, Any]], limit: int, tools: list[dict[str, Any]] | None) -> float:
        """Estimated USD for one call, made before sending it. Input tokens are taken as characters / 3.5; a multi-turn
        history is assumed cached except its newest 6000 characters (cache reads at a tenth); the reply is taken as a
        third of max_tokens, 300 at least. Prices per million tokens: QUICK 0.10 in / 0.50 out, anything else 2 / 10."""
        size = len(flat(tools)) if tools else 0
        for m in messages:
            size += len(flat(m.get("content")))
        per_in, per_out = {QUICK: (0.10, 0.50)}.get(model, (2.0, 10.0))
        uncached = size if len(messages) <= 2 else min(size, 6000)
        tokens_in = (uncached + (size - uncached) / 10) / 3.5
        tokens_out = max(300, limit // 3)
        return (tokens_in * per_in + tokens_out * per_out) / 1e6

    def size_purse(self, items: int, steps: int) -> None:
        """Sets this episode's spend stop: PURSE_BASE, plus 0.004 per extra item and a little per step allowed beyond 40."""
        self.purse = min(PURSE_TOP, PURSE_BASE + 0.004 * max(0, items - 1) + 0.00006 * max(0, min(steps, 110) - 40))

    def _send(self, model: str, messages: list[dict[str, Any]], limit: int, tools: list[dict[str, Any]] | None,
              effort: str | None) -> tuple[dict[str, Any] | None, Any]:
        outgoing = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            # cache marks on the system text and the newest message: the long loop history is re-read cheaply
            outgoing = [dict(m) for m in messages]
            for index in {0, len(outgoing) - 1}:
                text = outgoing[index].get("content")
                if isinstance(text, str) and text:
                    outgoing[index]["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": outgoing, "max_tokens": limit, "temperature": 0}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        if effort:
            body["reasoning"] = {"effort": effort}
        result = _model_line.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)) and not isinstance(usage.get("cost"), bool):
                self.spent += float(usage["cost"])
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            self.log.append({"kind": "model_empty", "model": model})
            return None, 0
        fault = result.error or {}
        status = fault.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            status = whole(data["error"].get("code"))
            fault = {"body": data["error"].get("message")}
        self.log.append({"kind": "model_fault", "model": model, "status": status, "body": str(fault.get("body") or "")[:200]})
        return None, status

    def talk(self, messages: list[dict[str, Any]], models: list[str], limit: int,
             tools: list[dict[str, Any]] | None = None, effort: str | None = None) -> dict[str, Any] | None:
        """The first model in `models` that answers; each is retried on passing faults within the pause budget."""
        for model in models:
            if self.closed():
                return None
            if self.spent + self.guess(model, messages, limit, tools) > min(SPEND_HARD_STOP, self.purse + 0.007):
                # this call would carry the episode past its stop: a cheaper model in the list may still fit
                self.log.append({"kind": "spend_skip", "model": model, "spent": round(self.spent, 5)})
                continue
            self.call_list += 1
            attempt = 0
            while True:
                message, status = self._send(model, messages, limit, tools, effort)
                if message is not None:
                    return message
                if status == 402:
                    self.no_credit = True
                    return None
                retryable = status in (None, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                tries_allowed = 1 if status in (None, 408) else len(RETRY_PAUSES)
                if not retryable or attempt >= tries_allowed:
                    break
                pause = RETRY_PAUSES[attempt]
                if self.paused + pause > RETRY_BUDGET or self.seconds_left() - pause < NO_NEW_CALL:
                    break
                self.paused += pause
                attempt += 1
                time.sleep(pause * random.uniform(0.8, 1.2))
        return None

    def verdict(self, ask_text: str, models: list[str], limit: int = 1800, effort: str | None = None) -> dict[str, Any] | None:
        """A one-message question whose answer is a JSON object; the next model is tried when a reply does not parse."""
        for model in models:
            message = self.talk([{"role": "user", "content": ask_text}], [model], limit, None, effort)
            parsed = read_object((message or {}).get("content"))
            if parsed is not None:
                return parsed
            if self.closed():
                return None
        return None


# ======================================================================== instructions for the models

LOOP_RULES = """Role: you are Tern's hands in an online shop, buying for one customer. You act only through the shop's tools.

Scoring. An episode is worth 1 point or nothing. It earns the point when it ends one of two ways:
(a) an order in which every product the customer listed appears in the asked number of units, each one satisfying every condition that applies at checkout, at a total the customer can afford; or
(b) when the shop genuinely holds nothing that satisfies everything, the no-purchase ending, carried out as the customer instructed.

1. Working in this shop
- A tool call costs one step and the step allowance is finite; reasoning is free. Tool names, parameter names and result layouts differ from shop to shop, so read each tool's description and schema and pass exactly the parameter names and values it defines.
- Finding things can be fiddly: a category filter here, a keyword box that only looks at the first few words there, small pages, a cursor (token or offset) to fetch the next page. If an answer depends on the complete set of matches, compare the match total the shop reports with what you have actually seen and keep fetching pages with the exact cursor it hands back. Keyword queries do best when short: the wanted value first, then the product noun. A filter that returns nothing usually has a wrong value or the wrong parameter; try the label spelled exactly as rows print it, or a different parameter.
- Rows in a result are previews only. Shops expose a product in different views: one may list its attributes (properties, specifications), another only price, stock and blurb, and a multi-product view may show either. Once Tern knows which view carries attributes, the notes say so. Every condition about a property is judged on the attributes, so read them for anything you may buy; a single call that shows several products saves steps when it shows what you need. Never reopen a product that has not changed since you read it.
- Never write to the customer yourself: Tern's interviewer does all questioning, and the replies show up in the conversation and in the notes.
- The request may also require actions along the way, such as an approval, a message, or something to read first. The notes list them. Those due just before the order or the no-purchase ending are carried out by Tern at the moment you call that tool, so do not wait for them and do not abandon buying because of them; any other such action you perform with the shop's tools (product questions still go through the interviewer).

2. Deciding whether a product qualifies
- What kind of product something is comes from its category label, and only a label naming the requested product as a whole counts; a neighbouring category does not.
- A condition on a property holds when the product's attribute for that property gives the value. Spelling of the attribute name is irrelevant (spacing, underscores, capitals, prefixes); for list values one option suffices, whatever separates the options (comma, slash, "and", "or", "&"). Values are compared word for word, ignoring case and singular/plural and accepting plain synonyms; a value that contains every word of the wanted one also holds. Different wording does not: a code is not the word it stands for, and colours that merely share a word are different colours; but the initials of a multi-word value state it (SUV for sport utility vehicle). Convert units before comparing figures; a limit includes its boundary; a range has to fit entirely inside a limit.
- The attribute named after the condition's property is decisive: when it gives something else, the product fails no matter what its title or other attributes suggest. When several attributes carry the property's name, the notes may mark a condition "[judged on attribute 'X']": then attribute X decides it, and the other attributes carrying that name say nothing about it. An attribute about a different property, however close (another component of the same product, say), says nothing about this condition. After opened products Tern prints [evidence] lines listing the attributes whose names match a condition.
- With no attribute named after the property, the wanted value stated in some other attribute (a feature list, for example) or in the title or blurb is acceptable, unless the other read listings of that kind record the property as an attribute: then the shop judges the condition on that attribute, and a listing without it is not proven; prefer one whose attribute states the value. A figure that measures something else is not, and neither is a number whose text never says what it measures.
- If the category label itself names a wanted value, the category settles that condition.
- Some conditions borrow their value from another product the request describes ("the same as", "different from", "close to" what that product has). Resolve those first: locate that product (search by its category, brand, seller or words), open its full page with a single or multi-product view (a search preview does not count), and if that page points to yet another product for the value, locate and open that one as well; the final page supplies the value. Every page in that chain must be open before checkout. A condition other than "the same as" only rules out products carrying that exact value, and never on its own justifies ending without an order. Where the request links the value to a rule the shop publishes, the value produced by that rule is the one that counts.
- If a product you are about to buy refers you elsewhere for details about itself, read that page before buying.
- If one requested product has to work with another (an accessory and its device), pick the two together using what the pages themselves say (a compatibility list, a model or series code on both). If no accessory meeting its own conditions works with the chosen device, change to a different qualifying device that such an accessory works with; do not give up on ordering while that change has not been tried.
- Exclusions and closed lists are binding. Mere preferences only break ties.
- Do not buy anything the shop marks as sold out. The sum of price times units across all lines may not go above the budget; hitting it exactly is fine.

3. Choosing among qualifying products
- "Cheapest" (lowest price or total) means the order must hold the lowest-priced product of the ENTIRE qualifying set. Collect that set (page through it: rows are rarely sorted by price, and Tern reads short leftovers itself), test candidates from the lowest price upward and take the first that fully qualifies. A search that accepts a price ceiling lets you list the category below your current best, leaving only cheaper candidates.
- "Most" or "least" of some measure: compare every qualifying product on it and take the extreme; a product silent about the measure cannot win.
- Any other request: the first product that satisfies everything is the right one, so do not hunt for a better one.
- As soon as you have opened a product that satisfies all conditions of an item (after comparing the qualifying set for cheapest/most/least), add it to the cart and move on: carted lines often sell out soon after, and carting early leaves steps to replace one.
- Several different products mean one cart line each, each with its own unit count.

4. When something changes
- The customer or the shop may change the job midway: a new budget, a condition added or withdrawn, a price change, a product selling out. Such news arrives inside tool results, frequently as a refused checkout; it applies wherever it appears, and the notes show the conditions currently in force. A refused checkout does not end the episode: apply every change, repair the cart, and check out again.
- Re-read a chosen product shortly before checkout, since price and stock may shift without warning.

5. Ending
- Make the cart hold exactly the chosen lines (remove anything else), view the cart once, then send the checkout as a call by itself.
- Choose the no-purchase ending only when a genuine search turned up nothing that satisfies everything. Being short of steps proves nothing of the sort; in that case check out the best cart you have. Phrase the ending as the request asks: start with the words it gives for that case, and when it wants the closest option offered or named, name the nearest listing with its id, price and what it lacks; when it forbids suggestions, name none. Tern completes that text when you leave it out.
- Each reply has to contain at least one tool call. Keep your words few: what you know, then your next move."""


READ_REQUEST = """Your job: convert a customer's shopping request into a structured brief for a buying agent. Input: the request text and the shop's tool list. Output: exactly one JSON object, compact (no indentation), with nothing before or after it, using these keys.

items — array; one element per separate product to be bought. Element fields:
  name: the kind of product, worded as the request words it.
  shelf: the shop's category label for it, copied exactly, if some tool enumerates such labels; otherwise "".
  quantity: integer count of units of that single listing; 1 when unspecified. A pair or set sold as a single listing is 1. People who will share the product, or earlier purchases mentioned in passing, do not change the count.
  musts: array of strict conditions on this product, each formatted "property: constraint" and carrying the exact value, limit, exclusion or allowed set.
    Use the request's own word for each property and keep the condition on that property rather than a related one.
    If a condition's value is to be taken from a different listing, make it a single must that preserves the full description of that other listing together with the request's exact wording of how the two values relate. The words describing that other listing (seller, kind, trailing "with ..." details) remain inside that single must and never become separate musts of this product.
    Never place price, budget, availability or unit count in musts. Background in the request (who it is for, where it will be used, earlier troubles, the occasion) is not a must unless it spells out a property value or limit; words forming part of the product's name (a season, an audience or a use stated with it) go into name.
  A product mentioned as one phrase joined by "and" with a single quantity (frequently the name of one shop department) is a single item; split into several items only if each product has its own quantity or its own conditions.
budget — highest total the customer will pay, as a number in the request's currency unit; null if none.
budget_unit — that unit as worded in the request; null if none.
budget_scope — "total" if the cap is for the whole order, "each" if per item, null if there is no cap.
in_stock_only — true unless the request clearly accepts products that are unavailable right now.
pick — "cheapest" if the request asks for the lowest price or total; "most" / "least" if it asks for the extreme of some measure; "any" otherwise. Saying money is short is not a pick rule.
pick_measure — the measure behind "most"/"least"; "" otherwise.
messages — integer limit on messages the agent may send the customer; null when the request sets none.
steps — integer limit on steps, moves or tool calls stated in the request; null otherwise.
lookups — the musts (verbatim) whose value has to be read off another listing the request refers to; [] if none. A condition that one requested item must suit or fit another item of this same order is NOT a lookup.
fits — array of {"item": n, "with": m} for every requested item that the request says must work with, fit or suit another requested item; [] if none.
rules — each instruction on how to proceed or how to finish, at most 12 words apiece.
wishes — things the customer would merely like.
ask_first — matters the request explicitly says to confirm with the customer before choosing; [] if none.
process — actions the request requires beyond locating and buying the right products (obtain approval, tell or show the customer something, confirm something with them, read something first, use a specific tool). Each element: {"step": the action in at most 15 words, "when": "choose" (before products are chosen) | "order" (just before checkout) | "end" (just before finishing without a purchase), "messages": number of customer messages it needs, 0 if none}. [] when there are none. List an action only when the request tells the agent to do it (an instruction such as "check with me before you pay" or "tell me which one you picked"); a description of how the product should be made, look or behave is a condition on the product, never an action. Conditions on products, the budget, availability, the message and step limits, and the wording or alternative wanted for a finish without a purchase are not actions (those belong to quiet_decline and decline_words).
quiet_decline — true if the request forbids suggesting or naming alternatives when nothing qualifies; false if it wants the closest option (or some alternative) named, shown or proposed in that case; null if it is silent on this.
decline_words — any exact phrase the request sets for a finish without a purchase (the words to use or open with when nothing qualifies), without quotation marks; "" when it sets none.
roles — which exact tool performs each job, or null where none does: "search"; "open" (displays one listing's attributes/properties — if two single-listing tools exist, take the one whose description mentions attributes, properties or specifications); "offer" (displays one listing's current price and stock, when that is a separate tool from "open"); "compare" (several listings in full); "ask" (messages the customer); "cart" (modifies the cart); "cart_view" (displays the cart); "buy" (checkout); "refuse" (finish without a purchase).
buy_args — the fixed argument values that make the "buy" tool perform a checkout (for example a mode value); {} if none.
refuse_args — the fixed argument values that make the "refuse" tool finish without a purchase; {} if none.
"""


SPLIT_MODES = """Tool {tool} performs either a checkout or a finish-without-purchase, chosen by its arguments. Schema: {schema}
Output one JSON object only: {{"buy_args": arguments selecting checkout, "refuse_args": arguments selecting the no-purchase finish}} — include only the mode-selecting arguments."""


SEPARATE = """Background: a customer's request has been broken into strict conditions on one product. Some conditions take their value from a different listing that the request refers to. The words identifying that other listing (its kind, its seller, the details following "with") sometimes get misread as conditions on the product being bought.
Task: judging by the request's wording, name which of the OTHER REQUIREMENTS listed below merely describe that other listing.
Output one JSON object only: {"describe_other": [such requirements, copied verbatim; [] if every one is a genuine condition on the product]}"""


QUESTION_PLAN = """Task: choose which product properties a buying agent should ask its customer about, and in which order.

How the customer behaves: they hold a few strict preferences they never wrote down, each tied to a single property of a single requested product; when several products are requested, most of them hold none and one or two hold them all. A question reveals such a preference only if it names that very property. On a property they have no preference about they reply "don't mind"; a property that means nothing to them leaves them puzzled. Either way the message is used up, and there are only a few messages; each message names a short list of one item's properties (the first ones are answered most surely), so the order matters within an item as well as across items.

Input: the request; each item with what the request already pins down for it; and, per item category, the attribute names found on its opened listings, how many listings carry each name, and some example values. A hidden preference may also concern a property none of these names covers.

Rank the (item, property) pairs over ALL items together, the pair most likely to hold an unstated preference first:
(0) Anything the request itself says to confirm with the customer (shown under ASK FIRST), named after the property it concerns.
(1) Hinted properties: a sentence of the request that implies an unstated preference without its value — something the customer cannot judge or leaves open, a person whose taste decides, how the product should look or the setting it must suit, the person who will use or wear it (look for an attribute telling men's from women's or children's versions), what it contains or is made from. Name the single property each such sentence implies, using a listed attribute name of that item when one matches the hint. A hint that no listed attribute of the item matches ranks after that item's best-carried listed attribute. The item the strongest hint is about goes first. Where accessories or parts are bought for a main product, expect the preferences on that main product.
(2) After those, each item's listed attribute names, the ones carried by more of its listings first: first a name sorting the product into its varieties (a type, style, build or sub-kind whose values read like words from the listing titles and vary from listing to listing), then names whose values distinguish variants in words (fastening, power source, placement, packaging, pattern, finish, colour, material, the group it is made for). Names whose values are bare measurements, sizes, weights or capacities rank after those. Pack sizes, counts and per-pack quantities are ordinary properties.
Leave out: properties the request already pins down for that item (an attribute whose values only echo a required value is NOT pinned down by it), synonyms of a property already chosen, a store's option, filter or variant field whose values bundle several other properties or purchase choices, identifiers, model names, warranty and terms of sale.
label = the property's plain name. For a listed attribute name, all of its words in plain spelling, never cut down to one of them (two names that share a word are two properties); only the name, without examples, brackets or a second property.
Return at most COUNT entries per item, all in ONE list in the order the questions should be asked across items, as one JSON object only: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 words of the request it rests on, or an empty string"}]}."""


READ_ANSWERS = """Below are questions a buying agent put to its customer, with the customer's replies. Output one JSON object only, with these fields:
needs: [{"item": n, "need": "property: constraint", "points": true if the value must be taken from another listing, else false}] — every strict preference a reply expresses, with its exact value or limit and its unit. Use the property the reply itself names (it may differ from the property asked about); several replies about one property give one need. Copy the value in the reply's own words, nouns and units included.
  A reply can supply the value by referring to another listing (like, unlike, or near the value of some seller's product, singled out by a few of its properties). That is ONE need on the property the reply names, with points true, whose value is the reply's wording copied exactly from the comparison word to the end of that description; the properties used to describe the other product are not needs of this item.
drop: [{"item": n, "property": "..."}] — conditions the customer now abandons.
free: [{"item": n, "property": "..."}] — each property a question named that the customer said they don't mind about; a reply waving off a question that named several properties frees all of them, a reply stating a preference about one of them frees none of the others.
crowded: [numbers of questions whose reply complains the message held too much or asks for one question at a time — not replies that simply failed to understand].
puzzled: [numbers of questions whose reply says the customer did not follow it, or asks for it again, simpler or in other words]. Such a reply states nothing: no needs and nothing free from it.
budget: a new total cap as a number, or null.
Quaint or indirect phrasing that names a value still counts as stating it."""


READ_NOTE = """A note appeared in the shop while a buying agent was working (it may come from the customer or from the shop). Output one JSON object only:
{"kind": "change" if the note alters the job (budget, conditions, a price, availability, a refused order) | "error" if it only reports a malformed or rejected call | "other", "budget": the new total cap as a number in the request's unit, or null, "add": [{"item": n, "need": "property: constraint"}], "drop": [{"item": n, "need": "the condition no longer wanted"}], "gone": [ids of listings said to be unavailable], "sold_out": true if it states or asks whether a chosen listing is gone, taken, sold out, unavailable or could not be added (even without naming it), "price_moved": true if it reports a price change, "summary": "one short sentence"}
A condition that is replaced counts as a drop of the old plus an add of the new. Include every change the note makes."""


FIT_RULES = ("the attribute named after a condition's property is decisive, and when a condition is marked [judged on attribute 'X'], "
             "attribute X decides it and other attributes carrying that name say nothing (attribute-name spelling is irrelevant; for a list one option "
             "suffices, whatever joins the options; case, singular/plural, plain synonyms and unit conversion are fine; a value containing "
             "every wanted word holds; a code is not the word it stands for, though the initials of a multi-word value state it (SUV for sport utility vehicle); limits include their boundary); an attribute "
             "about some other property, even a close one such as another component, says nothing; with no attribute named after the "
             "property, the wanted value stated in another attribute or in the title or blurb is acceptable unless other listings of "
             "that kind record the property as an attribute (then a listing without it is not proven) (a number that never says what "
             "it measures is worth nothing); a value named by the category label is settled by the category; the category label has to "
             "name the item; a 'different from that listing' condition holds unless the records show the very same value; when the request "
             "links a value to a rule the shop publishes, the value produced by that rule counts")

CART_REVIEW = ("Role: last reviewer before a buying agent's checkout. Using only the records below, decide whether the cart is exactly "
               "right.\nJudging: " + FIT_RULES + ".\nCart requirements: one separate line per requested item at its requested unit "
               "count; no other lines; no line shown sold out; total of price times units across lines within the budget. A listing's "
               "most recent record overrides older ones. Where one item has to work with, fit or suit another requested item, the two "
               "listings' own text must tie them together (a code or name on both, or a statement on one covering the other).\n"
               "With a cheapest, most or least pick, also consider the other listings shown: if one plainly satisfies everything and "
               "beats the cart's choice under that rule, the cart is wrong.\n"
               "Output one JSON object only: {\"ok\": true|false, \"problems\": [\"brief, specific problem\", ...], \"better\": "
               "[ids of listings that should replace a cart line, best first]}. Use ok=false only for a problem the records show plainly "
               "or for a condition no record addresses.")

DECLINE_REVIEW = ("Situation: a buying agent is about to finish without ordering, believing no listing qualifies. Test the listing records "
                  "below against the conditions.\nJudging: " + FIT_RULES + "; a listing shown sold out or above the budget does not "
                  "qualify.\nA record tagged ROW is only a search preview: its page and attributes are unread, so a property the preview "
                  "does not show neither passes nor fails it.\n"
                  "Output one JSON object only: {\"fits\": {\"<item number>\": [ids that seem to satisfy every condition, cheapest first, "
                  "no more than 4]}, \"open\": [ids of ROW previews of a requested kind, inside the budget, whose page might show they "
                  "qualify, most promising first, no more than 6], \"nearest\": {\"<item number>\": the id of the OPENED record of that kind that comes "
                  "closest to qualifying, or \"\"}, \"reason\": \"one sentence\"}")

SOLD_OUT_SWAP = ("Situation: a line in a buying agent's cart has just sold out. Among the other listing records shown, identify those that "
                 "could replace it.\nJudging: " + FIT_RULES + "; a listing shown sold out or above the budget does not qualify.\n"
                 "Output one JSON object only: {\"fits\": [{\"id\": listing id, \"price\": its price as a number, as recorded} — every "
                 "listing that seems to satisfy every condition of the sold-out line's item, no more than 6], \"reason\": \"one sentence\"}")

CHAIN_READ = """Situation: a buying agent wants to check out. Certain conditions take their value from another listing, whose page may in turn refer the reader to a further listing for that property. The order is valid only if the agent has fully opened every listing along each such chain.
Task: trace each condition's chain starting from its own description, one link at a time through the pages' own words, using the records below (OPENED = the agent opened that page; ROW = only a search preview, page unread).
A record counts as a link only if it matches that link's entire description: seller or brand, product kind and every value the description mentions. If nothing matches all of it, the link is still unfound, even when the same seller offers other listings of that kind.
Output one JSON object only: {"chains": [{"need": the condition verbatim, "links": [{"id": the listing id if a record below is this link, else "", "about": how the condition or the previous page describes it, "find": "for an unfound link: a few search words (seller plus the description's values)"}], "done": true if the last link is an OPENED page that itself states the property's value (also when the condition asks for a value different from it), else false, "value": that page's value for the property or "", "rule_value": if that page publishes a rule yielding a value for this condition, the value it yields, else ""}]}
Give links in chain order and stop at the first one that is unopened or unfound, because its words are not known yet."""

STEP_CHECK = """Situation: a buying agent is about to STAGE. The customer's request requires the actions below along the way.
Task: from the numbered RECORD of the agent's actions, decide for each action due before this STAGE whether it is done for the cart in its current state. An action is done only when a RECORD line shows it and, for an action concerning the cart or the order, that line is later than the cart's last change and the reply to it grants it. Questions about product properties never perform an action.
If an action is still missing, give the one shop call that performs it next — a tool from TOOLS plus arguments. Messages to the customer go through the messaging tool: one short plain sentence or question about that action only, without product properties or values the action does not require. Give no call if every due action is done, or if the remaining messages or steps cannot perform the missing one.
Output one JSON object only: {"steps": [{"step": the action, "done": true|false, "lines": [RECORD line numbers proving it]}], "call": {"tool": "...", "args": {...}} or null, "hold": anything a reply asks to change before proceeding, or "", "needs": [{"item": n, "need": "property: constraint"} for every strict condition such a reply states]}"""

RESULT_PARTS = """Below is a result returned by a shop tool. Identify its parts using the exact key names from the result and from the tool's parameters. Output one JSON object only: {"rows": key of the list of listings, "count": key giving how many listings matched in total (not how many appear here) or null, "next": key whose value requests the next page or null, "next_arg": the tool parameter that accepts that value or null, "price": key inside a row holding the listing's price or null, "stock": key inside a row saying whether the listing can be bought now or null}"""


# ======================================================================== what the shopper asked for


@dataclass
class Want:
    """A product on the customer's list, plus everything learned about it while shopping."""
    name: str
    shelf: str
    units: int
    stated: list[str]
    heard: list[str] = field(default_factory=list)
    given_up: list[str] = field(default_factory=list)
    open_on: list[str] = field(default_factory=list)

    def live(self) -> list[str]:
        return [need for need in self.stated + self.heard if need not in self.given_up]


class TaskSheet:
    """Structured view of the customer's request, updated whenever answers or shop notes change it."""

    JOBS = ("search", "open", "offer", "compare", "ask", "cart", "cart_view", "buy", "refuse")

    def __init__(self) -> None:
        self.raw: dict[str, Any] = {}
        self.wants: list[Want] = []
        self.tool_for: dict[str, str] = {}
        self.budget: float | None = None
        self.pick = "any"
        self.measure = ""
        self.message_cap = 0
        self.step_cap: int | None = None
        self.steps_asked: list[dict[str, Any]] = []
        self.pointers: list[tuple[int, str]] = []  # (item number, requirement whose value sits on another listing)
        self.pointer_values: dict[str, str] = {}
        self.pointer_rules: dict[str, str] = {}
        self.pairs: list[tuple[int, int]] = []  # (part item, device item)
        self.buy_args: dict[str, Any] = {}
        self.refuse_args: dict[str, Any] = {}

    def take(self, raw: dict[str, Any], tool_names: set[str]) -> None:
        self.raw = raw
        given = raw.get("roles") if isinstance(raw.get("roles"), dict) else {}
        self.tool_for = {job: tool for job, tool in given.items() if job in self.JOBS and isinstance(tool, str) and tool in tool_names}
        self.buy_args = raw.get("buy_args") if isinstance(raw.get("buy_args"), dict) else {}
        self.refuse_args = raw.get("refuse_args") if isinstance(raw.get("refuse_args"), dict) else {}
        for bit in as_list(raw.get("items")):
            if isinstance(bit, dict):
                units = whole(bit.get("quantity")) or 1
                self.wants.append(Want(str(bit.get("name") or "the product"), str(bit.get("shelf") or ""), max(1, units),
                                       [str(m) for m in as_list(bit.get("musts")) if m]))
        if not self.wants:
            self.wants.append(Want("the requested product", "", 1, []))
        for link in as_list(raw.get("fits")):
            if isinstance(link, dict):
                part, device = whole(link.get("item")) or 0, whole(link.get("with")) or 0
                if part != device and 1 <= part <= len(self.wants) and 1 <= device <= len(self.wants) and (part, device) not in self.pairs:
                    self.pairs.append((part, device))
        for text in (str(x).strip() for x in as_list(raw.get("lookups"))):
            if not text:
                continue
            owner = self.owner_of_must(text)
            if owner is None:
                head = plain(text.split(":")[0])
                match = next(((n, m) for n, w in enumerate(self.wants, 1) for m in w.stated if plain(m.split(":")[0]) == head), None)
                owner, text = match if match else (1, text)
            if (owner, text) not in self.pointers:
                self.pointers.append((owner, text))
        number = amount(raw.get("budget"))
        self.budget = number if number is not None and number > 0 else None
        self.pick = raw.get("pick") if raw.get("pick") in ("cheapest", "most", "least") else "any"
        self.measure = str(raw.get("pick_measure") or "")
        for bit in as_list(raw.get("process")):
            if isinstance(bit, dict) and str(bit.get("step") or "").strip():
                when = str(bit.get("when") or "").strip().lower()
                self.steps_asked.append({"step": str(bit["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                         "messages": max(0, whole(bit.get("messages")) or 0)})

    def owner_of_must(self, text: str) -> int | None:
        return next((n for n, want in enumerate(self.wants, 1) if text in want.stated), None)

    def want(self, number: Any) -> Want:
        index = whole(number) or 1
        return self.wants[min(max(1, index), len(self.wants)) - 1]

    def index_of(self, number: Any) -> int:
        return min(max(1, whole(number) or 1), len(self.wants))

    def live_needs(self) -> list[tuple[int, str]]:
        return [(n, need) for n, want in enumerate(self.wants, 1) for need in want.live()]

    def open_pointers(self) -> list[str]:
        live = {need for _, need in self.live_needs()}
        return [need for _, need in self.pointers if need in live and need not in self.pointer_values]

    def quiet(self) -> bool | None:
        val = self.raw.get("quiet_decline")
        return val if isinstance(val, bool) else None

    def in_stock_only(self) -> bool:
        return self.raw.get("in_stock_only", True) is not False

    def summary(self, read_from: str = "", judged: dict[str, str] | None = None) -> str:
        """Bullet list of everything currently in force; `judged` names the attribute that decides a condition when
        several attributes carry its property's name."""
        several = len(self.wants) > 1
        text_lines = []
        for number, want in enumerate(self.wants, 1):
            need_list = []
            for need in want.live():
                if judged and need in judged:
                    need_list.append(f"{need} [judged on attribute '{judged[need]}']")
                    continue
                if need in self.pointer_values:
                    extra = f"linked page says: {self.pointer_values[need]}"
                    if need in self.pointer_rules:
                        extra += f"; by the rule published there: {self.pointer_rules[need]}"
                    need = f"{need} ({extra})"
                need_list.append(need)
            head = (f"item {number}: " if several else "item: ") + want.name
            if want.shelf:
                head += f" [shop category {want.shelf}]"
            text = f"{head} x{want.units} | conditions: " + (" ; ".join(need_list) or "only the product kind")
            if want.open_on:
                text += " | customer indifferent about: " + ", ".join(want.open_on)
            text_lines.append(text)
        for part, device in self.pairs:
            text_lines.append(f"item {part} has to be compatible with whichever item {device} gets chosen (both pages must share a code or name)")
        if self.budget is not None:
            scope = "per item" if self.raw.get("budget_scope") == "each" else "for the whole order"
            unit = str(self.raw.get("budget_unit") or "").strip()
            text_lines.append(f"spending cap: {self.budget:g} {unit} {scope}".replace("  ", " "))
        rule = {"cheapest": "choose: lowest price among qualifying products", "most": f"choose: highest {self.measure} among qualifying products",
                "least": f"choose: lowest {self.measure} among qualifying products"}
        text_lines.append(rule.get(self.pick, "choose: any product meeting every condition"))
        if self.in_stock_only():
            text_lines.append("only products available now")
        text_lines += [f"nice to have (breaks ties only): {w}" for w in as_list(self.raw.get("wishes"))]
        text_lines += [f"instruction in the request: {r}" for r in as_list(self.raw.get("rules"))]
        if self.quiet() is True:
            text_lines.append("no qualifying product: finish without mentioning or proposing any product")
        stage_words = {"choose": "before you pick products", "order": "at checkout (handled by Tern when you call it)",
                       "end": "when finishing without an order (handled by Tern then)"}
        text_lines += [f"required action {stage_words[s['when']]}: {s['step']}" for s in self.steps_asked]
        if read_from:
            text_lines.append(read_from)
        return "\n".join("- " + line for line in text_lines)


# ======================================================================== everything the store has shown about listings


class Catalog:
    """Memory of the shop's listings: latest record per id, opened pages, attribute-bearing records, result layouts, sell-outs."""

    def __init__(self, sheet: TaskSheet, schemas: dict[str, dict[str, Any]]):
        self.sheet = sheet
        self.schemas = schemas
        self.cards: dict[str, dict[str, Any]] = {}
        self.read: dict[str, dict[str, Any]] = {}  # records that showed an attribute block
        self.order: list[str] = []
        self.visited: set[str] = set()
        self.off_sale: set[str] = set()
        self.retired: dict[str, int] = {}
        self.key: str | None = None
        self.price_field: str | None = None
        self.stock_field: str | None = None
        self.layouts: dict[str, dict[str, Any] | None] = {}
        self.view_fields: dict[str, set[str]] = {}
        self.seen_fields: dict[str, tuple[int, set[str]]] = {}
        self._labels: set[str] | None = None
        self._marks: dict[tuple[str, bool, frozenset[str]], tuple[frozenset[str], frozenset[str], tuple[tuple[str, frozenset[str]], ...], frozenset[str], frozenset[str]]] = {}
        self._heads: tuple[int, frozenset[str]] = (-1, frozenset())
        self.seen_at: dict[str, int] = {}   # listing -> number of the latest result that showed it
        self._shelf: tuple[int, str | None] = (-1, None)
        self.results = 0

    # ---- ids and records
    def lid(self, piece: Any, in_rows: bool = False, level: int = 0) -> str | None:
        if not isinstance(piece, dict) or len(piece) < 3:
            return None
        if self.key and looks_like_id(piece.get(self.key), True):
            return piece[self.key].strip()
        if not (in_rows or level == 0 or any(isinstance(v, dict) for v in piece.values())):
            return None
        collected = id_field([piece])
        if collected and ("id" in str(collected).lower() or piece[collected].strip() in self.cards):
            return piece[collected].strip()
        return None

    def _learn_rows(self, cards: list[Any]) -> bool:
        dicts = [r for r in cards if isinstance(r, dict) and len(r) >= 3]
        if not dicts or 2 * len(dicts) < len(cards):
            return False
        if self.key and all(looks_like_id(r.get(self.key), True) for r in dicts):
            return True
        collected = id_field(dicts)
        if collected:
            self.key = collected
        return bool(collected)

    def take(self, obs: Any) -> None:
        """Merge each listing record found anywhere in a result."""
        self.results += 1
        pending: list[tuple[Any, int, bool]] = [(obs, 0, False)]
        while pending:
            piece, level, in_rows = pending.pop()
            if level > 6:
                continue
            if isinstance(piece, list):
                cards = self._learn_rows(piece)
                pending.extend((part, level + 1, cards) for part in reversed(piece))
                continue
            if not isinstance(piece, dict):
                continue
            lid = self.lid(piece, in_rows, level)
            if not lid:
                pending.extend((part, level + 1, False) for part in reversed(list(piece.values())))
                continue
            if lid not in self.cards:
                self.cards[lid] = {}
                self.order.append(lid)
            self.seen_at[lid] = self.results
            before = self.cards[lid]
            # a boolean on the record flipping true->false: treated as no longer purchasable
            if any(before.get(k) is True and v is False for k, v in piece.items()):
                self.off_sale.add(lid)
            elif any(before.get(k) is False and v is True for k, v in piece.items()):
                self.off_sale.discard(lid)
            before.update(piece)
            if attribute_block(piece):
                self.read[lid] = dict(before)
            elif lid in self.read:
                self.read[lid].update(piece)

    def listings_in(self, obs: Any) -> list[tuple[str, dict[str, Any]]]:
        """(id, record) pairs in a result: the result itself or rows nested at most two deep."""
        if isinstance(obs, list):
            return [(i, r) for r in obs if isinstance(r, dict) for i in [self.lid(r, True)] if i]
        if not isinstance(obs, dict):
            return []
        own = self.lid(obs)
        if own:
            return [(own, obs)]
        collected: list[tuple[str, dict[str, Any]]] = []
        for part in obs.values():
            lists = [part] if isinstance(part, list) else [v for v in part.values() if isinstance(v, list)] if isinstance(part, dict) else []
            for cards in lists:
                collected += [(i, r) for r in cards if isinstance(r, dict) for i in [self.lid(r, True)] if i]
        return collected

    def record(self, lid: str, room: int = 400, cap: int = 3000) -> str:
        collected = self.read.get(lid) or self.cards.get(lid)
        return flat(shorten(collected, room))[:cap] if collected else "(no record seen)"

    # ---- labels, categories and kinds
    def schema_labels(self) -> set[str]:
        if self._labels is None:
            collected: set[str] = set()
            stack = [(s, 0) for s in self.schemas.values()]
            while stack:
                sch, level = stack.pop()
                if level > 6 or not isinstance(sch, dict):
                    continue
                collected.update(o.strip().lower() for o in sch.get("enum") or [] if isinstance(o, str))
                stack.extend((v, level + 1) for v in sch.values() if isinstance(v, dict))
            self._labels = collected
        return self._labels

    def shelf_field(self) -> str | None:
        """Guess which row field holds the category: short text present on most rows with few distinct values, favouring
        values that tool schemas offer as options (worked out once per number of known listings)."""
        if self._shelf[0] == len(self.cards):
            return self._shelf[1]
        found = self._shelf_guess()
        self._shelf = (len(self.cards), found)
        return found

    def _shelf_guess(self) -> str | None:
        vals: dict[str, set[str]] = {}
        carried: dict[str, int] = {}
        for row in self.cards.values():
            for key, val in row.items():
                if key == self.key or not isinstance(val, str) or not 0 < len(val) <= 40 or re.search(r"\d{3}", val):
                    continue
                vals.setdefault(key, set()).add(val)
                carried[key] = carried.get(key, 0) + 1
        everyone = max(1, len(self.cards))
        listed = self.schema_labels()
        scored = []
        for key, shown_obs in vals.items():
            if carried[key] < 0.8 * everyone or len(shown_obs) > 40:
                continue
            share = sum(v.strip().lower() in listed for v in shown_obs) / len(shown_obs)
            scored.append(((-share, len(shown_obs), -carried[key], sum(map(len, shown_obs)) / len(shown_obs)), key))
        return min(scored)[1] if scored else None

    def kind(self, lid: str) -> str:
        key = self.shelf_field()
        return str((self.cards.get(lid) or {}).get(key) or "").strip() if key else ""

    def label_score(self, label: str, want: Want) -> int:
        """Match strength between a category label and an item: 3 exact, 2 label words inside the item's, 1 one long word in common."""
        theirs = roots(label)
        if not theirs:
            return 0
        if plain(label) in (plain(want.shelf), plain(want.name)):
            return 3
        mine = roots(want.shelf) | roots(want.name)
        if theirs <= mine:
            return 2
        return 1 if {w for w in theirs if len(w) >= 4} & {w for w in mine if len(w) >= 4} else 0

    def owner(self, label: str) -> int:
        """Item number whose name best matches the label (0 when none does)."""
        best, top = 0, 0
        for number, want in enumerate(self.sheet.wants, 1):
            score = self.label_score(label, want)
            if score > top:
                best, top = number, score
        return best

    def ids_of(self, number: int) -> list[str]:
        return [i for i in self.order if self.owner(self.kind(i)) == number]

    def groups(self) -> list[list[str]]:
        """Ids grouped by matching category, biggest group first, as many groups as items."""
        by_label: dict[str, list[str]] = {}
        for lid in self.order:
            by_label.setdefault(self.kind(lid), []).append(lid)
        mine = [ids for label, ids in by_label.items() if any(self.label_score(label, w) for w in self.sheet.wants)]
        chosen = mine or list(by_label.values())
        chosen.sort(key=len, reverse=True)
        return chosen[: max(1, len(self.sheet.wants))]

    # ---- which attribute decides a condition
    def deciding_key(self, need: str) -> str | None:
        """When several attributes carry a condition's property name, the one that decides is the one that holds that
        kind of value: the attribute on which the wanted value appears for the most read listings (the plainly named one
        on a tie). None when there is no such choice to make or the condition is not a plain value."""
        prop, _, want = need.partition(":")
        want = want.strip()
        if not want or re.search(r"\d", want) or re.search(r"\b(?:not|no|than|least|most|under|over|between|within|unlike|same|or)\b", want, re.I):
            return None
        def tokens(text: Any) -> set[str]:
            return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"\w+", str(text).lower())}

        target_lines = tokens(want)
        asked = roots(prop)
        hits: dict[str, int] = {}
        for record in self.read.values():
            block = attribute_block(record) or {}
            matched: set[str] = set()
            for key, val in block.items():
                if not same_property(asked, roots(key)):
                    continue
                name = plain(key)
                hits.setdefault(name, 0)
                if target_lines and target_lines <= tokens(val) and name not in matched:
                    matched.add(name)
                    hits[name] += 1
        if len(hits) < 2:
            return None
        best = max(hits, key=lambda name: (hits[name], roots(name) == asked, -len(name)))
        if hits[best] == 0:
            return None
        exact = next((name for name in hits if roots(name) == asked), None)
        # the plainly named attribute keeps the decision unless another one clearly holds this kind of value more often
        if exact and best != exact and hits[exact] > 0 and hits[best] < hits[exact] + 2:
            return exact
        return best

    # ---- likeness between two listings
    def name_field(self) -> str | None:
        """The row field that names listings: the multi-word text field most rows carry, the shortest on average when
        several are carried equally often (a name line is shorter than a description)."""
        shelf = self.shelf_field()
        shown_obs: dict[str, list[int]] = {}
        for row in self.cards.values():
            for key, val in row.items():
                if key not in (self.key, shelf) and isinstance(val, str) and len(val.split()) >= 2:
                    shown_obs.setdefault(key, []).append(len(val))
        if not shown_obs:
            return None
        # carried by (nearly) the most rows, then the shortest: one one-word title must not hand the name to the description
        top = max(len(v) for v in shown_obs.values())
        return max(shown_obs, key=lambda k: (len(shown_obs[k]) >= 0.9 * top, -sum(shown_obs[k]) / len(shown_obs[k])))

    def title_words(self, lid: str, field_name: str | None = None) -> list[str]:
        """Distinct words (three letters or more) of the listing's name line, in order; long names give their first words."""
        key = field_name or self.name_field()
        text = (self.cards.get(lid) or {}).get(key) if key else None
        if not isinstance(text, str):
            return []
        words: list[str] = []
        for word in re.findall(r"\w+", text.lower()):
            if len(word) > 2 and word not in words:
                words.append(word)
            if len(words) >= 16:
                break
        return words

    def likeness(self, a: str, b: str) -> tuple[int, int]:
        """(attribute values the two read records share under the same key, name words they share)."""
        def facts(lid: str) -> dict[str, str]:
            block = attribute_block(self.read.get(lid) or {}) or {}
            return {re.sub(r"[^a-z]", "", str(k).lower()): str(v).strip().lower() for k, v in block.items()}
        fa, fb = facts(a), facts(b)
        same = sum(1 for k, v in fa.items() if k and fb.get(k) == v)
        key = self.name_field()
        return same, len(set(self.title_words(a, key)) & set(self.title_words(b, key)))

    # ---- prices and stock
    def price(self, lid: str) -> float | None:
        val = (self.cards.get(lid) or {}).get(self.price_field) if self.price_field else None
        if isinstance(val, dict):
            val = next((v for v in val.values() if is_num(v)), None)
        return amount(val)

    # ---- model codes: a device's own code against what an accessory's own page says it fits
    CODE = r"[A-Z]{2,5}-?\d{2,6}[0-9xX]*"

    def part_heads(self) -> frozenset[str]:
        """Letter heads of codes that two or more listings carry in their own name: the shop's part numbers. A head met only
        in short code sentences ("Goes with AB120.") belongs to the models parts go with."""
        if self._heads[0] == len(self.cards):
            return self._heads[1]
        field = self.name_field()
        seen: dict[str, set[str]] = {}
        for lid, card in self.cards.items():
            for raw in re.findall(self.CODE, str(card.get(field) or "") if field else ""):
                seen.setdefault(re.match(r"[A-Z]+", raw).group(), set()).add(lid)
        self._heads = (len(self.cards), frozenset(h for h, ids in seen.items() if len(ids) >= 2))
        return self._heads[1]

    def code_marks(self, lid: str) -> tuple[frozenset[str], frozenset[str], tuple[tuple[str, frozenset[str]], ...], frozenset[str], frozenset[str]]:
        """What a listing's own wording says about model codes: (codes it is, single codes it works with, code ranges it works
        with as (leading part, codes left out), other part numbers it refers the reader to, codes its short code sentences
        pair it by). Wording that talks about some other part number is that part's business and is passed over (race 13
        read such a line as the accessory's own). Race 14 pages speak in bare code sentences ("Model ref. AB120.", "Goes
        with AB120, AB130.", "More on part XY123."): a sentence of codes with at most four other words gives pairing codes,
        unless it names another part number, which is either a reference to look up (only part numbers in it) or a claim
        about that other part (passed over)."""
        cache_key = (lid, lid in self.read, self.part_heads())
        hit = self._marks.get(cache_key)
        if hit is not None:
            return hit
        merged = {**(self.cards.get(lid) or {}), **(self.read.get(lid) or {})}
        lines = walk_strings(merged, 1)
        lines += [plain(k) + ": " + str(v) for k, v in (attribute_block(merged) or {}).items()]
        is_codes: set[str] = set()
        works: set[str] = set()
        ranges: dict[str, frozenset[str]] = {}
        elsewhere: set[str] = set()
        norm = lambda raw: re.sub("-", "", raw.upper())
        for clause in (m.group(0) for line in lines for m in re.finditer(r"[^.;!?\n]+", line)):
            lowered = clause.lower()
            other_part = re.search(r"\bpart\s+([A-Z]{1,5}-?\d{3,8})", clause, re.I)
            if other_part:
                if re.search(r"\brest\b|\bsee\b", lowered):
                    elsewhere.add(norm(other_part.group(1)))
                continue
            if re.search(r"compatib|\bfits?\b", lowered):
                split_at = lowered.find(" except ")
                named = clause if split_at < 0 else clause[:split_at]
                left_out = frozenset(norm(c) for c in re.findall(self.CODE, clause[split_at:])) if split_at >= 0 else frozenset()
                for raw in re.findall(self.CODE, named):
                    code = norm(raw)
                    if code[-1] == "X":
                        ranges[code.rstrip("X")] = ranges.get(code.rstrip("X"), frozenset()) | left_out
                    else:
                        works.add(code)
                continue
            own = re.search(r"\bmodel\s*(?:no\.?|number)?\s*[:#]\s*(" + self.CODE + ")", clause, re.I)
            if own and norm(own.group(1))[-1] != "X":
                is_codes.add(norm(own.group(1)))
        heads = self.part_heads()
        field = self.name_field()
        named = {norm(c) for c in re.findall(self.CODE, str(merged.get(field) or "") if field else "")}
        texts = [v for k, v in merged.items() if isinstance(v, str) and k != field]
        named |= {norm(c) for t in texts for c in re.findall(r"\(\s*(?:[A-Za-z]+\s+)?(" + self.CODE + r")\s*\)", t)}
        pairing: set[str] = set()
        for sentence in (bit for t in texts for bit in re.split(r"(?<=[.;!?])\s+", t)):
            codes = [norm(c) for c in re.findall(self.CODE, sentence)]
            if not codes or len(re.findall(r"[A-Za-z]{2,}", re.sub(self.CODE, " ", sentence))) > 4:
                continue
            mentioned = re.search(r"\bpart\s+([A-Z]{1,5}-?\d{3,8})", sentence, re.I)
            if mentioned and norm(mentioned.group(1)) not in named:
                # a sentence about another part number: a pointer to it ("More on part X") or that part's own claim
                if not re.search(r"compatib|\bfits?\b|\bgoes\s+with\b", sentence, re.I):
                    elsewhere.add(norm(mentioned.group(1)))
                continue
            others = [c for c in codes if re.match(r"[A-Z]+", c).group() in heads and c not in named]
            if others:
                if all(re.match(r"[A-Z]+", c).group() in heads for c in codes):
                    elsewhere.update(others)
                continue
            pairing.update(c for c in codes if c not in named)
        result = (frozenset(is_codes), frozenset(works), tuple(sorted(ranges.items())), frozenset(elsewhere), frozenset(pairing))
        self._marks[cache_key] = result
        return result

    def paired(self, a: str, b: str) -> bool:
        """Two listings' pages pair them by code: one's code sentences name a code the other names or carries."""
        ma, mb = self.code_marks(a), self.code_marks(b)
        # either page's own or pairing codes, fitted by the other's single codes, ranges or pairing codes
        return any(self.fits_code(a, c) for c in mb[0] | mb[4]) or any(self.fits_code(b, c) for c in ma[0] | ma[4])

    def fits_code(self, lid: str, code: str) -> bool:
        marks = self.code_marks(lid)
        if code in marks[1] or code in marks[4]:
            return True
        return any(code[: len(lead)] == lead and code not in out for lead, out in marks[2])

    def is_gone(self, lid: str) -> bool:
        return lid in self.off_sale or lid in self.retired

    def can_buy(self, lid: str) -> bool:
        if self.is_gone(lid):
            return False
        flag = (self.cards.get(lid) or {}).get(self.stock_field) if self.stock_field else None
        if flag is False:
            return False
        return not (isinstance(flag, (int, float)) and not isinstance(flag, bool) and flag <= 0)

    def cheap_order(self, ids: list[str]) -> list[str]:
        return sorted(ids, key=lambda i: self.price(i) if self.price(i) is not None else float("inf"))

    # ---- what each view shows
    def note_view(self, tool: str, obs: Any, epoch: int) -> None:
        for lid, record in self.listings_in(obs):
            params = set(record)
            self.view_fields.setdefault(tool, set()).update(params)
            then, had = self.seen_fields.get(lid, (-1, set()))
            self.seen_fields[lid] = (epoch, (had if then == epoch else set()) | params)

    def nothing_new(self, tool: str, lid: str, epoch: int) -> bool:
        then, had = self.seen_fields.get(lid, (-1, set()))
        params = self.view_fields.get(tool)
        return bool(params) and then == epoch and params <= had and lid not in self.retired

    # ---- attribute keys across opened listings
    @staticmethod
    def _merge(keys: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Unify differently spelled attribute names that hold the same property (agreeing values on shared listings)."""
        def squash(text: str) -> str:
            return re.sub(r"[^a-z0-9]", "", text)

        folded: dict[str, dict[str, Any]] = {}
        members: dict[str, list[str]] = {}
        for name in sorted(keys, key=lambda n: (-len(n.split()), -keys[n]["n"], n)):
            tally = keys[name]
            into = None
            for host, sch in folded.items():
                common = [i for i in tally["per"] if i in sch["per"]]
                agree = sum(squash(tally["per"][i]) == squash(sch["per"][i]) for i in common)
                if agree != len(common):
                    continue
                inside = roots(name) <= roots(host) or roots(host) <= roots(name)
                shared = bool({squash(v) for v in tally["values"]} & {squash(v) for v in sch["values"]})
                if (inside and (agree or shared)) or (roots(name) & roots(host) and len(common) >= 3):
                    into = host
                    break
            if into is None:
                folded[name] = {"n": tally["n"], "values": list(tally["values"]), "per": dict(tally["per"])}
                members[name] = [name]
                continue
            sch = folded[into]
            members[into].append(name)
            for lid, text in tally["per"].items():
                sch["per"].setdefault(lid, text)
            sch["values"] += [v for v in tally["values"] if v not in sch["values"]]
            sch["n"] = len(sch["per"])
        # a folded group goes by the spelling most of its listings carry, the shorter on a tie (race 13: "style" on seven
        # jackets was asked as "general styles", the store's longer key on one of them, and drew nothing)
        named: dict[str, dict[str, Any]] = {}
        for host, sch in folded.items():
            best = min(members[host], key=lambda n: (-keys[n]["n"], len(n.split()), len(n), n))
            named[best if best not in named else host] = sch
        return named

    @staticmethod
    def _recurring(vals: list[str]) -> bool:
        """Half the values or more contain a word that some other value also contains ("modern farmhouse", "modern classic")."""
        tally: dict[str, int] = {}
        word_sets = [frozenset(re.findall(r"[a-z]{3,}", v)) for v in vals]
        for words in word_sets:
            for w in words:
                tally[w] = tally.get(w, 0) + 1
        echoed = [words for words in word_sets if any(tally[w] > 1 for w in words)]
        return len(echoed) * 2 >= len(vals)

    def key_table(self) -> list[dict[str, Any]]:
        """Attribute names per matching category with coverage counts, distinct-value counts and examples."""
        per_label: dict[str, dict[str, dict[str, Any]]] = {}
        for lid in self.order:
            record = self.read.get(lid)
            block = attribute_block(record) if record else None
            if not block:
                continue
            label = self.kind(lid) or "listings"
            keys = per_label.setdefault(label, {})
            for raw_key, val in block.items():
                name = plain(raw_key)
                if not name:
                    continue
                tally = keys.setdefault(name, {"n": 0, "values": [], "per": {}})
                if lid in tally["per"]:
                    continue
                text = str(val).strip().lower()
                tally["per"][lid] = text
                tally["n"] += 1
                if text not in tally["values"]:
                    tally["values"].append(text)
        relevant = {label for label in per_label if any(self.label_score(label, w) for w in self.sheet.wants)}
        if relevant:
            per_label = {label: keys for label, keys in per_label.items() if label in relevant}
        table: list[dict[str, Any]] = []
        for label, keys in sorted(per_label.items(), key=lambda kv: -max((s["n"] for s in kv[1].values()), default=0)):
            kept = 0
            for name, tally in sorted(self._merge(keys).items(), key=lambda kv: -kv[1]["n"]):
                vals = list(tally["per"].values())
                mean = sum(map(len, vals)) / len(vals)
                if mean > 40 or 2 * sum(" / " in v for v in vals) >= len(vals):
                    continue  # prose, or option lists
                if len(vals) >= 4 and len(set(vals)) == len(vals) and mean > 12 and not self._recurring(vals):
                    continue  # unique free text per listing (a style or kind differs on every page too, but its words recur)
                table.append({"label": label, "key": name, "carried": tally["n"], "distinct": len(set(vals)),
                              "samples": [v[:40] for v in tally["values"][:4]]})
                kept += 1
                if kept >= 22:
                    break
        return table

    def key_hints(self, obs: Any) -> str:
        """Per opened listing in a result: attributes whose names match a condition."""
        need_list = self.sheet.live_needs()
        if not need_list:
            return ""
        text_lines = []
        several = len(self.sheet.wants) > 1
        for lid, _ in self.listings_in(obs)[:12]:
            block = attribute_block(self.read.get(lid) or {}) or {}
            if not block:
                continue
            parts = []
            for number, need in need_list[:8]:
                prop = need.split(":", 1)[0].strip()
                hits = [f"{k}={str(v)[:50]}" for k, v in block.items() if same_property(roots(prop), roots(k))]
                if hits:
                    parts.append(("(item %d) " % number if several else "") + f"{prop} -> " + ", ".join(hits[:3]))
            if parts:
                text_lines.append(f"{lid}: " + "; ".join(parts))
        if not text_lines:
            return ""
        return ("[evidence | attribute names matching a condition; an attribute on another component settles nothing, and a "
                "differently named attribute can still carry the property] " + " | ".join(text_lines))


# ======================================================================== the cart


class Basket:
    """Cart state: tallied from cart calls, overridden by any line list a cart reply includes or by a fresh cart view."""

    def __init__(self, catalog: Catalog, schemas: dict[str, dict[str, Any]], tool_for: dict[str, str]):
        self.catalog = catalog
        self.schemas = schemas
        self.tool_for = tool_for
        self.text_lines: dict[str, int] = {}
        self.view: Any = None
        self.viewed_at = -1
        self.changed_at = -1
        self.touched = False
        self.reply_text = ""
        self.reply_lists = False  # the cart tool's replies list the cart's lines
        self.count_field: str | None = None
        self.best_build: list[str] = []
        self.taken_off: set[str] = set()
        self.id_params: set[tuple[str, str]] = set()

    # ---- reading the cart
    def fresh_view(self) -> bool:
        return self.view is not None and self.viewed_at >= self.changed_at

    def ids(self) -> list[str]:
        if self.fresh_view():
            shown = flat(self.view)
            return [i for i in self.catalog.cards if i in shown]
        return list(self.text_lines)

    def holding(self) -> bool:
        return bool(self.ids()) if self.fresh_view() else bool(self.text_lines)

    def mark(self) -> str:
        return flat(sorted(self.text_lines.items()))

    def text(self) -> str:
        if self.fresh_view():
            return flat(shorten(self.view, 300))[:3000]
        return "cart as tallied from the cart calls made (id: units): " + flat(self.text_lines)

    def units(self, lid: str) -> int:
        """Unit count for one line, from a fresh cart view if it shows a single count, else from the tally."""
        if self.fresh_view():
            stack: list[Any] = [self.view]
            while stack:
                piece = stack.pop()
                if isinstance(piece, list):
                    stack.extend(piece)
                elif isinstance(piece, dict):
                    if lid in [v.strip() for v in piece.values() if isinstance(v, str)]:
                        counts = [whole(v) for k, v in piece.items() if not isinstance(v, str) and whole(v) is not None
                                  and re.search(r"\b(?:qty|quantity|count|units?|amount)\b", plain(k))]
                        if len(counts) == 1:
                            return counts[0] or 0
                    stack.extend(v for v in piece.values() if isinstance(v, (dict, list)))
        return self.text_lines.get(lid, 1)

    def saw_view(self, obs: Any, step: int) -> None:
        self.view = obs
        self.viewed_at = step

    # ---- the cart tool's schema
    def sets_counts(self, tool: str) -> bool:
        """True when the cart tool assigns a quantity (min 0, no mode option) rather than adding to it."""
        params = (self.schemas.get(tool) or {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in params.values()):
            return False
        return any(isinstance(s, dict) and "integer" in kinds_of(s) and s.get("minimum") == 0 for s in params.values())

    def _is_removal(self, tool: str, args: dict[str, Any]) -> bool:
        params = (self.schemas.get(tool) or {}).get("properties") or {}
        for key, val in args.items():
            options = (params.get(key) or {}).get("enum")
            if isinstance(options, list) and len(options) >= 2 and val in options[1:]:
                return True
        return False

    # ---- recording a cart call
    def _arithmetic(self, tool: str, args: dict[str, Any]) -> dict[str, int]:
        params = (self.schemas.get(tool) or {}).get("properties") or {}
        loose = next((whole(v) for v in args.values() if not isinstance(v, str) and whole(v) is not None), None)
        touched: list[tuple[str, int | None]] = []
        replaces_all = False
        for key, val in args.items():
            if isinstance(val, str):
                if val in self.catalog.cards:
                    touched.append((val, loose))
                continue
            if not isinstance(val, list):
                continue
            inner = (params.get(key) or {}).get("items") or {}
            if not val and isinstance(inner, dict) and (inner.get("type") == "object" or "properties" in inner):
                replaces_all = True
            for bit in val:
                if isinstance(bit, dict):
                    replaces_all = True
                    lid = next((v for v in bit.values() if isinstance(v, str) and v in self.catalog.cards), None)
                    count = next((whole(v) for v in bit.values() if not isinstance(v, str) and whole(v) is not None), 1)
                    if lid:
                        touched.append((lid, count))
                elif isinstance(bit, str) and bit in self.catalog.cards:
                    touched.append((bit, loose))
        if replaces_all:
            return {i: c for i, c in touched if c and c > 0}
        after = dict(self.text_lines)
        removal, setting = self._is_removal(tool, args), self.sets_counts(tool)
        for lid, count in touched:
            have = after.get(lid, 0)
            if removal:
                have = have - count if count else 0
            elif setting:
                have = 1 if count is None else count
            else:
                have += 1 if count is None else count
            if have > 0:
                after[lid] = have
            else:
                after.pop(lid, None)
        return after

    def _listed(self, obs: Any, tool: str, args: dict[str, Any]) -> dict[str, int] | None:
        """Lines listed in a cart reply, or None if it has no usable list."""
        if not isinstance(obs, dict) or self.catalog.lid(obs):
            return None
        lists: list[list[Any]] = []
        stack: list[tuple[Any, int]] = [(obs, 0)]
        while stack:
            piece, level = stack.pop()
            if level > 3:
                continue
            if isinstance(piece, list):
                if all(isinstance(r, dict) for r in piece):
                    lists.append(piece)
            elif isinstance(piece, dict):
                stack.extend((v, level + 1) for v in piece.values())
        touched = {v for val in args.values() for v in (val if isinstance(val, list) else [val])
                   if isinstance(v, str) and v in self.catalog.cards}
        with_ids = [cards for cards in lists if cards and any(self.catalog.lid(r, True) for r in cards)]
        cards = next((r for r in with_ids if any(self.catalog.lid(x, True) in touched for x in r)), with_ids[0] if with_ids else None)
        if cards is None:
            return {} if len(lists) == 1 and not lists[0] else None
        idents = [self.catalog.lid(r, True) for r in cards]
        if not all(idents):
            return None
        key = self.count_field
        if key is None:
            options = [k for k in cards[0] if k != self.catalog.price_field
                       and all(isinstance(r.get(k), int) and not isinstance(r.get(k), bool) and 0 <= r[k] <= 99 for r in cards)]
            if len(options) > 1:
                expected = self._arithmetic(tool, args)
                options = [k for k in options if all(r.get(k) == expected.get(i, 0) for r, i in zip(cards, idents) if i in touched)]
            if len(options) != 1:
                return None
            key = self.count_field = options[0]
        collected: dict[str, int] = {}
        for row, lid in zip(cards, idents):
            count = row.get(key)
            if not isinstance(count, int) or isinstance(count, bool):
                return None
            if count > 0:
                collected[str(lid)] = count
        return collected

    def record_call(self, tool: str, args: dict[str, Any], obs: Any, step: int) -> None:
        self.touched = True
        self.changed_at = step
        listed = self._listed(obs, tool, args)
        if listed is not None:
            self.text_lines = listed
            self.view, self.viewed_at, self.reply_lists = obs, step, True
        elif not self.reply_lists:
            self.text_lines = self._arithmetic(tool, args)
        self.reply_text = flat(obs)
        counts = [whole(v) for v in args.values() if not isinstance(v, str) and whole(v) is not None]
        for val in args.values():
            if counts and isinstance(val, str) and val in self.catalog.cards:
                if counts[0] <= 0 and not holds_id(obs, val):
                    self.taken_off.add(val)
                elif counts[0] > 0:
                    self.taken_off.discard(val)
        kept = [i for i in self.ids() if not self.catalog.is_gone(i) and i not in self.taken_off]
        if len(kept) >= len(self.best_build):
            self.best_build = kept

    # ---- building cart calls
    def calls_for(self, target_lines: dict[str, int], remove: list[str]) -> list[dict[str, Any]] | None:
        """Argument sets for the cart tool that produce `wanted` and clear `remove` (None if the schema gives no way)."""
        tool = self.tool_for.get("cart") or ""
        form = self.schemas.get(tool) or {}
        params = form.get("properties") or {}
        for key, sch in params.items():
            inner = sch.get("items") if isinstance(sch, dict) and "array" in kinds_of(sch) else None
            if isinstance(inner, dict) and isinstance(inner.get("properties"), dict):
                parts = inner["properties"]
                id_key = next(iter(free_text_params(parts)), None)
                count_key = first_param(parts, "integer")
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in target_lines.items()]}]
        text_fields = free_text_params(params)
        id_param = (next((k for t, k in self.id_params if t == tool and k in text_fields), None)
                  or next((k for k in form.get("required") or [] if k in text_fields), None) or (text_fields[0] if len(text_fields) == 1 else None))
        count = first_param(params, "integer")
        mode = switch_param(params)
        if not id_param:
            return None
        call_list: list[dict[str, Any]] = []
        for lid in remove:
            have = self.text_lines.get(lid, 1)
            if mode:
                call_list.append({id_param: lid, mode: params[mode]["enum"][1], **({count: have} if count else {})})
            elif count and self.sets_counts(tool):
                call_list.append({id_param: lid, count: 0})
            elif count and isinstance(params[count].get("minimum"), int) and params[count]["minimum"] <= -have:
                call_list.append({id_param: lid, count: -have})
            else:
                return None
        present = self.ids()
        for lid, want in target_lines.items():
            have = self.text_lines.get(lid, 0) if lid in present else 0
            if have == want:
                continue
            if have and not self.sets_counts(tool):
                if not mode or not count:
                    return None
                call_list.append({id_param: lid, mode: params[mode]["enum"][1], count: have})
            call_list.append({id_param: lid, **({mode: params[mode]["enum"][0]} if mode else {}), **({count: want} if count else {})})
        return call_list

    # ---- checks in code before a checkout
    def faults(self, sheet: TaskSheet) -> list[str]:
        """Plain cart errors visible in the records (missing/duplicate lines, wrong units, stray kinds, sold out, over budget)."""
        catalog = self.catalog
        carried = [i for i in self.ids() if self.units(i) > 0]
        if not carried or not catalog.shelf_field():
            return []
        owner = {i: catalog.owner(catalog.kind(i)) for i in carried}
        known = {n for n in range(1, len(sheet.wants) + 1) if any(catalog.owner(catalog.kind(i)) == n for i in catalog.order)}
        collected: list[str] = []
        for number, want in enumerate(sheet.wants, 1):
            if number not in known:
                continue
            mine = [i for i in carried if owner[i] == number]
            if not mine:
                collected.append(f"nothing in the cart for item {number} ({want.name})")
            elif len(mine) > 1:
                collected.append(f"item {number} ({want.name}) appears {len(mine)} times ({', '.join(mine)}); leave a single line")
            for lid in mine:
                if self.units(lid) != want.units:
                    collected.append(f"{lid}: {self.units(lid)} unit(s) in the cart but item {number} ({want.name}) needs {want.units}")
        if len(known) == len(sheet.wants):
            collected += [f"remove {i} ({catalog.kind(i) or 'unknown kind'}): the customer did not ask for that kind" for i in carried if not owner[i]]
        collected += [f"{i} can no longer be bought" for i in carried if catalog.is_gone(i)]
        prices = {i: catalog.price(i) for i in carried}
        if sheet.budget is not None and None not in prices.values():
            if sheet.raw.get("budget_scope") == "each":
                collected += [f"{i} at {p:g} breaks the per-item cap of {sheet.budget:g}" for i, p in prices.items()
                          if sheet.budget < p <= 10 * sheet.budget]
            else:
                total = sum(p * self.units(i) for i, p in prices.items())
                if sheet.budget + 1e-9 < total <= 10 * sheet.budget:
                    collected.append(f"cart total {total:g} exceeds the cap of {sheet.budget:g}")
        return collected


# ======================================================================== questions to the shopper


def speakable(label: str) -> str:
    """A property name a person can read: leading one- or two-letter code fragments of a store key are dropped."""
    words = plain(label).split()
    while len(words) > 1 and len(words[0]) <= 2 and words[0] not in ("of", "on", "in", "to", "by", "up", "no"):
        words = words[1:]
    return " ".join(words)


def _measured(samples: list[str]) -> bool:
    """True when the examples are mostly plain numbers with a short unit."""
    figures = sum(bool(re.fullmatch(r"[\d.,/x ×-]+\s*[a-z\"']{0,6}", s.strip())) for s in samples)
    return 2 * figures > len(samples) if samples else False


class Interviewer:
    """Single-property questions, one per message, each chosen after reading the previous reply.

    One plan ranks the (item, property) pairs over all items; each item keeps a queue (explicit confirmations, the
    plan's picks, then the item's attribute names by coverage). The next question goes to the best-ranked item that has
    not yielded a preference yet and has been asked least; a confused reply means the property was wrong, so the next
    one is tried rather than a rephrase. Queues are kept, so a message left over at the checkout can still ask one."""

    def __init__(self, shop: "Shopper"):
        self.shop = shop
        self.done = False
        self.used: dict[int, set[frozenset[str]]] = {}
        self.revealed: dict[int, int] = {}
        self.asked_count: dict[int, int] = {}
        self.exchanges: list[tuple[int, str, str]] = []
        self.queues: dict[int, list[str]] = {}
        self.item_rank: dict[int, int] = {}
        self.last: int | None = None
        self.room: dict[int, int] = {}

    # ---- choosing what to ask
    def plan(self, room: int) -> dict[int, list[str]]:
        shop, sheet = self.shop, self.shop.sheet
        table = shop.catalog.key_table()
        fixed = [f"item {n}: {w.name} - settled by the request: " + ("; ".join(w.stated) or "nothing") for n, w in enumerate(sheet.wants, 1)]
        listed, last = [], None
        for row in table:
            if row["label"] != last:
                listed.append(f"category {row['label']!r}:")
                last = row["label"]
            listed.append(f"  - {row['key']} (on {row['carried']}): " + " | ".join(row["samples"]))
        ask_text = (QUESTION_PLAN.replace("COUNT", str(room)) + "\n\nREQUEST:\n" + shop.request + "\n\nITEMS:\n" + "\n".join(fixed))
        first = [str(x) for x in as_list(sheet.raw.get("ask_first")) if str(x).strip()]
        if first:
            ask_text += "\n\nASK FIRST (to confirm with the customer, per the request):\n" + "\n".join(first)
        ask_text += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listed) or "(no full listing records yet)")
        answer = shop.desk.verdict(ask_text, [CAREFUL, QUICK], 2400) or {}
        picked: dict[int, list[tuple[str, bool]]] = {n: [] for n in range(1, len(sheet.wants) + 1)}
        for place, bit in enumerate(as_list(answer.get("ask"))):
            if not isinstance(bit, dict):
                continue
            number = sheet.index_of(bit.get("item"))
            # a label naming two properties is cut back to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(bit.get("label") or ""))[0].strip().rstrip("?.")
            if label and len(label.split()) <= 5 and all(plain(label) != plain(l) for l, _ in picked[number]):
                picked[number].append((label, bool(str(bit.get("hint") or "").strip())))
                # items are ordered by where their first pick sits in the plan
                self.item_rank.setdefault(number, place)
        queues = {n: self._rank(n, picked[n], table) for n in picked}
        shop.log.append({"kind": "ask_plan", "plan": shorten(answer.get("ask"), 100), "rank": dict(self.item_rank),
                         "queues": {n: q[:8] for n, q in queues.items()}})
        return queues

    @staticmethod
    def _id_like(samples: list[str]) -> bool:
        """Values that label each listing instead of offering a choice: a third of their characters figures, or long
        phrases whose words never come back across the listings."""
        total = sum(len(v) for v in samples)
        if total == 0 or len(re.findall(r"\d", "".join(samples))) * 3 >= total:
            return True
        return total > 12 * len(samples) and not Catalog._recurring(samples)

    def _rank(self, number: int, chosen: list[tuple[str, bool]], table: list[dict[str, Any]]) -> list[str]:
        """The plan's picks in its order, hinted first; then the item's attribute names by coverage, worded values ahead of
        bare numbers and a one-word name that ends a longer name of the item after that longer one. The best-carried worded
        name keeps one of the first two places, and a pick no read listing carries waits behind it."""
        catalog, want = self.shop.catalog, self.shop.sheet.wants[number - 1]
        cards = [r for r in table if catalog.owner(r["label"]) == number]
        if not cards and len(self.shop.sheet.wants) == 1:
            cards = table
        read = max([r["carried"] for r in cards] + [1])
        settled = [roots(m.split(":")[0]) for m in want.stated]
        names = [r["key"] for r in cards]
        share: dict[str, float] = {}
        order: dict[str, tuple] = {}
        worded: dict[str, bool] = {}
        values = self._values(number)
        for r in cards:
            key = r["key"]
            if any(roots(key) <= s or (same_property(s, roots(key)) and self._settled_by(number, key, need, values))
                   for s, need in zip(settled, want.stated)):
                continue
            # a different value on every listing is only an identifier when the values look like one (a colour differs on
            # every page of a small sample too: race 13 lost colour questions to this rule)
            if r["carried"] >= 3 and r["distinct"] == r["carried"] and not _measured(r["samples"]) and (
                    self._id_like(r["samples"]) or (r["carried"] >= 5 and not catalog._recurring(r["samples"]))):
                continue
            reserve_steps = len(key.split()) == 1 and any(other != key and other.endswith(" " + key) for other in names)
            share[key] = r["carried"] / read
            # a choice a buyer makes: values in words that repeat across listings
            worded[key] = not _measured(r["samples"]) and 2 <= r["distinct"] < r["carried"]
            order[key] = (-(round(share[key] * 4) / 4), _measured(r["samples"]), reserve_steps, -r["carried"])

        code = sorted(share, key=lambda k: order[k])
        queue: list[str] = []
        own = roots(want.name) | roots(want.shelf)

        def share_of(label: str) -> float:
            return next((share[k] for k in share if roots(label) in (roots(k), roots(speakable(k)), roots(spoken(k)))), 0.0)

        def spoken(label: str) -> str:
            # the product's own name inside a long store key goes ("jackets closure type" is asked as "closure type": race
            # 13's shopper took the plain property and never the store's long name)
            words = speakable(label).split()
            kept = [w for w in words if not (roots(w) & own)]
            return " ".join(kept) if len(words) >= 3 and len(kept) >= 2 else " ".join(words)

        def add(label: str) -> None:
            twin = any(self._twins(values, speakable(label), other) for other in queue)
            label = spoken(label)
            if label and not twin and all(roots(label) != roots(other) for other in queue):
                queue.append(label)

        confirm = [roots(x) for x in as_list(self.shop.sheet.raw.get("ask_first")) if str(x).strip()]

        def asked_for(label: str) -> bool:
            # a property the request says to confirm (its words inside one of those matters) keeps its place
            return any(roots(label) and roots(label) <= c for c in confirm)
        hinted = [speakable(label) for label, flagged in chosen if flagged]
        for label in hinted + [speakable(label) for label, flagged in chosen if not flagged]:
            add(label)
        if read >= 4:
            # a pick no read listing carries (the plan's own words for a hint) waits behind the item's carried picks
            queue = [q for q in queue if share_of(q) > 0 or asked_for(q)] + [q for q in queue if share_of(q) == 0 and not asked_for(q)]
            best = next((k for k in code if worded[k] and share[k] >= 0.6), None)
            label = spoken(best) if best else ""
            weak = [k for k, o in enumerate(queue[:2]) if share_of(o) < 0.5 and not asked_for(o)]
            if label and weak and not any(roots(label) == roots(o) for o in queue[:2]):
                # one of the plan's first two picks is carried by under half the listings read: the best-carried worded
                # name takes its place, and the weak pick waits behind it
                queue = [o for o in queue if roots(o) != roots(label)]
                queue.insert(min(weak[0], len(queue)), label)
        for key in code:
            add(speakable(key))
        return queue

    def _values(self, number: int) -> dict[str, dict[str, str]]:
        """Per attribute name (as a person reads it), the value each read listing of the item shows."""
        catalog = self.shop.catalog
        out: dict[str, dict[str, str]] = {}
        for lid in catalog.ids_of(number) or (catalog.order if len(self.shop.sheet.wants) == 1 else []):
            for k, v in (attribute_block(catalog.read.get(lid)) or {}).items():
                text = str(v).strip().lower()
                for name in {plain(k), speakable(k)}:
                    out.setdefault(name, {})[lid] = text
        return out

    @staticmethod
    def _twins(values: dict[str, dict[str, str]], a: str, b: str) -> bool:
        """Two nested names ("material", "clothing material") for one property: on two or more read listings carrying both,
        three in four show the same value."""
        if not (roots(a) <= roots(b) or roots(b) <= roots(a)) or roots(a) == roots(b):
            return False
        va, vb = values.get(plain(a), {}), values.get(plain(b), {})
        both = [lid for lid in va if lid in vb and va[lid] and vb[lid]]
        return len(both) >= 2 and 4 * sum(va[lid] == vb[lid] for lid in both) >= 3 * len(both)

    def _settled_by(self, number: int, key: str, need: str, values: dict[str, dict[str, str]] | None = None) -> bool:
        """A longer key holding a settled property's words is that property when the item's read listings say so: the key
        shows the settled value, or shows the same value as the settled property's own key on one listing ("clothing
        material" and "material" both "cotton"). Otherwise it is another property ("memory type" beside "memory: 16 gb")."""
        prop, _, value = need.partition(":")
        wanted = roots(value)
        values = self._values(number) if values is None else values
        if wanted and any(wanted <= roots(v) for v in values.get(plain(key), {}).values()):
            return True
        return any(self._twins(values, key, name) for name in values if roots(name) == roots(prop))

    def _next(self, queues: dict[int, list[str]], last: int | None) -> tuple[int, list[str]] | None:
        sheet = self.shop.sheet
        for number, queue in queues.items():
            want = sheet.wants[number - 1]
            learned = [roots(t.split(":")[0]) for t in want.heard]
            asked = self.used.setdefault(number, set())
            queue[:] = [label for label in queue
                        if not any(t and t <= roots(label) for t in learned) and not any(roots(label) <= a for a in asked)]
        live = [n for n in queues if queues[n]]
        if not live:
            return None
        # the plan's best-ranked item first; an item that yielded a preference, or was asked more, waits for the others
        def turn(n: int) -> tuple[bool, int, int]:
            return self.revealed.get(n, 0) > 0, self.asked_count.get(n, 0), self.item_rank.get(n, 1000 + n)
        chosen = min(live, key=turn)
        room = max(1, self.room.get(chosen, NAMES_PER_ASK))
        labels = queues[chosen][:room]
        del queues[chosen][:room]
        return chosen, labels

    def phrase(self, number: int, labels: list[str] | str) -> str:
        # the property framed as the product's own (race 13: "...would you like for the Y?" drew a puzzled or
        # don't-mind reply where "...should the Y have?" named the same property and drew the preference)
        names = [labels] if isinstance(labels, str) else list(labels)
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
        return f"Which {listed} should the {self.shop.sheet.wants[number - 1].name} have?"

    # ---- running the interview
    def run(self, spare_only: bool = False) -> str:
        """Ask while messages last (all of them before the first cart change; with `spare_only`, at most one more, later)."""
        shop = self.shop
        first_round = not self.done
        self.done = True
        if shop.messages_free() <= 0 or shop.desk.seconds_left() <= WRAP_UP + ASK_WINDOW or not shop.sheet.tool_for.get("ask"):
            return ""
        if first_round:
            shop.cover_unread()
            shop.widen()
            per_item = -(-max(1, shop.messages_free()) // max(1, len(shop.sheet.wants)))
            self.queues = self.plan(min(8, max(4, per_item * NAMES_PER_ASK)))
        queues = self.queues
        learned: list[str] = []
        sent: list[tuple[int, str, str]] = []
        while shop.messages_free() > 0 and not shop.store.finished and shop.desk.seconds_left() > WRAP_UP + ASK_WINDOW:
            if spare_only and sent:
                break
            choice = self._next(queues, self.last)
            if choice is None:
                break
            number, labels = choice
            text = self.phrase(number, labels)
            resp = shop.message(text)
            if resp is None:
                break
            self.last = number
            self.used.setdefault(number, set()).update(roots(label) for label in labels)
            self.asked_count[number] = self.asked_count.get(number, 0) + 1
            self.exchanges.append((number, text, resp))
            sent.append((number, text, resp))
            before = len(learned)
            heard = self.hear([(number, text, resp)])
            if not as_list(heard.get("needs")) and any(roots(l) and roots(l) <= roots(resp) for l in labels) and not as_list(heard.get("free")):
                # the reply names a property it was asked about yet nothing was read from it: the careful model reads it
                # again (local: "pattern coming in as stripe isn't up for trade" went unread and a plain skirt was bought)
                heard = self.hear([(number, text, resp)], [CAREFUL]) or heard
            shop.apply_answers(heard, learned)
            if len(learned) > before:
                self.revealed[number] = self.revealed.get(number, 0) + 1
                # a reply states one preference as a rule: the list's other names were not answered and go back to the head
                # of the item's queue (local: "gender ... or case shape" drew gender only, and case shape was never asked)
                want = self.shop.sheet.wants[number - 1]
                told = [roots(t.split(":")[0]) for t in want.heard] + [roots(f) for f in want.open_on]
                rest = [l for l in labels if not any(t and (t <= roots(l) or roots(l) <= t) for t in told)]
                if rest and len(rest) < len(labels):
                    self.used[number] -= {roots(l) for l in rest}
                    self.queues[number][:0] = rest
            elif len(labels) > 1 and (as_list(heard.get("puzzled")) or as_list(heard.get("crowded"))):
                # the list did not get through: its names go back to the head of the item's queue, the next list half as long
                self.used[number] -= {roots(label) for label in labels}
                self.queues[number][:0] = labels
                self.room[number] = max(1, len(labels) // 2)
                shop.log.append({"kind": "ask_shorter", "item": number, "room": self.room[number]})
        if not sent:
            return ""
        if learned:
            shop.epoch += 1
        text_lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Interview held (one step per message):\n" + "\n".join(text_lines)
        summary += "\nConditions added from the replies: " + ("; ".join(learned) if learned else "none")
        summary += ". They bind every choice from now on; search again with the new values if needed."
        pointing = [need for _, need in shop.sheet.pointers if any(need in t for t in learned)]
        if pointing:
            summary += (" Value to be read off another listing: " + " | ".join(pointing) + ". Locate it, open its full page and any page it"
                        " refers on to; the final page supplies the value, and checkout is valid only once all of them have been opened.")
        return summary + shop.drain_notes()

    def hear(self, exchanges: list[tuple[int, str, str]], models: list[str] | None = None) -> dict[str, Any]:
        sheet = self.shop.sheet
        ask_text = READ_ANSWERS + "\n\nITEMS:\n" + "\n".join(f"item {n}: {w.name}" for n, w in enumerate(sheet.wants, 1))
        ask_text += "\n\nQ&A:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(exchanges, 1))
        return self.shop.desk.verdict(ask_text, models or [QUICK, CAREFUL], 1500) or {}


# ======================================================================== reviews by a second model


class Reviews:
    """Second opinions: cart before checkout, evidence before a no-order finish, stand-ins for sold-out lines, linked
    pages behind conditions, and shop notes."""

    def __init__(self, shop: "Shopper"):
        self.shop = shop
        self.passed: dict[str, bool] = {}
        self.asked: set[str] = set()
        self.declines_reviewed = False
        self.decline_looks = 0
        self.hopefuls: list[str] = []
        self.hopeful_items: dict[str, list[str]] = {}
        self.vouched: str | None = None
        self.chain_rounds = 0
        self.chain_holds = 0
        self.notes_heard: set[str] = set()
        self.sold_words: set[str] = set()
        self.note_said_sold = False
        self.nearest: dict[str, str] = {}
        self.better: list[str] = []
        self.better_mark = ""
        self.swap_memory: dict[str, list[str]] = {}

    def key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.shop.epoch}"

    # ---- the cart
    def cart(self, planned: list[str] | None = None, ceiling: float | None = None) -> str:
        """Empty string if the cart (or a planned set) passes, otherwise the fix needed."""
        shop = self.shop
        ids = planned if planned is not None else shop.basket.ids()
        key = self.key(ids)
        ceiling = 0.75 * shop.desk.purse if ceiling is None else ceiling
        if not ids or key in self.asked or shop.desk.spent > ceiling:
            return ""
        self.asked.add(key)
        sheet, catalog = shop.sheet, shop.catalog
        ask_text = CART_REVIEW + "\n\nCUSTOMER REQUEST:\n" + shop.request + "\n\nCONDITIONS NOW:\n" + shop.requirements()
        if planned is not None:
            units = "; ".join(f"item {n}: quantity {w.units}" for n, w in enumerate(sheet.wants, 1))
            ask_text += f"\n\nPROPOSED CART (assume each at its requested units: {units}):\n" + ", ".join(ids)
        else:
            ask_text += "\n\nCART CONTENTS:\n" + shop.basket.text()
        gone_tag = "[SOLD OUT per the shop] "
        ask_text += "\n\nCART LISTING RECORDS:\n" + "\n".join(f"{i}: {gone_tag if catalog.is_gone(i) else ''}{catalog.record(i)}" for i in ids)
        ranked = sheet.pick in ("cheapest", "most", "least")
        if ranked:
            ask_text += "\n\nOTHER RECORDS (latest first):\n" + shop.other_records(ids, 12)
        if planned is None:
            # the quick model reads the cart first (for a ranked pick too: race 13's careful reads of every ranked cart cost
            # as much as the rest of the episode); its pass lets the checkout go without being kept as a verdict, and a
            # fault it names goes to the careful model to decide
            quick = shop.desk.verdict(ask_text, [QUICK], 2000, "low")
            shop.log.append({"kind": "cart_quick", "cart": ids, "got": quick})
            if quick and quick.get("ok") is not False:
                self.asked.discard(key)
                self.better = []
                return ""
        judged = shop.desk.verdict(ask_text, shop.desk.careful_first(), 1500)
        shop.log.append({"kind": "cart_review", "cart": ids, "got": judged})
        self.passed[key] = bool(judged) and judged.get("ok") is not False
        if not judged or judged.get("ok") is not False:
            if planned is None:
                self.better = []
            return ""
        self.hopefuls = [i for i in self.hopefuls if i not in ids]
        problems = [str(p) for p in as_list(judged.get("problems"))][:5]
        better = [str(b) for b in as_list(judged.get("better")) if str(b) in catalog.cards and not catalog.is_gone(str(b))][:4]
        if planned is None:
            self.better = better
            self.better_mark = key
        text = "Cart review failed: " + "; ".join(problems or ["check the conditions"])
        if better:
            text += ". Stronger candidates: " + ", ".join(better)
        return text + ". Correct the cart, or re-read the records if you think the review is mistaken, and then check out."

    # ---- an ending without a purchase
    def _decline_look(self) -> dict[str, Any] | None:
        shop, catalog = self.shop, self.shop.catalog
        text_lines = []
        # the requested kinds first, newest first; shorter records keep the careful model's read affordable (race 13: a
        # review of 30 full records cost as much as the rest of the episode)
        # latest sighting first (race 14: the fitting set had been first seen early and sat 46th-111th by first sighting),
        # with rows a search for a required value returned ahead of the rest
        leads = set(shop.open_leads())
        order = [lid for lid in catalog.order if not catalog.is_gone(lid)]
        order.sort(key=lambda lid: (not catalog.owner(catalog.kind(lid)), lid not in leads, -catalog.seen_at.get(lid, -1)))
        for lid in order[:24]:
            if lid in catalog.read:
                text_lines.append("OPENED " + flat(shorten(catalog.read[lid], 160))[:900])
            else:
                text_lines.append("ROW " + flat(shorten(catalog.cards.get(lid), 100))[:380])
        if not text_lines:
            return None
        ask_text = (DECLINE_REVIEW + "\n\nCUSTOMER REQUEST:\n" + shop.request + "\n\nCONDITIONS NOW:\n" + shop.requirements()
                  + "\n\nRECORDS SEEN (latest first):\n" + "\n".join(text_lines))
        # QUICK reads first and CAREFUL only when QUICK names no fit: in race 13 all 44 of Tern's no-order finishes were
        # graded wrong, so a second, stronger read before one is worth its cost
        judged = shop.desk.verdict(ask_text, [QUICK], 1800, "low")
        fits = (judged or {}).get("fits")
        if not (isinstance(fits, dict) and any(as_list(v) for v in fits.values())) and shop.desk.spent < 0.6 * shop.desk.purse:
            judged = shop.desk.verdict(ask_text, [CAREFUL], 1500) or judged
        shop.log.append({"kind": "decline_review", "got": judged})
        near = (judged or {}).get("nearest")
        if isinstance(near, dict):
            for item, lid in near.items():
                if str(lid) in catalog.read:
                    self.nearest[str(item)] = str(lid)
        return judged

    def _named(self, judged: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (judged or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        catalog = self.shop.catalog
        named = {str(k): [str(i) for i in as_list(v) if str(i) in catalog.cards and not catalog.is_gone(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def decline(self) -> str:
        """One look per task version for overlooked qualifying listings; empty string if none."""
        shop = self.shop
        if self.declines_reviewed or self.decline_looks >= 2 or shop.desk.spent > shop.desk.purse:
            return ""
        self.declines_reviewed = True
        self.decline_looks += 1
        judged = self._decline_look()
        named = self._named(judged)
        if not named:
            cards = [str(i) for i in as_list((judged or {}).get("open"))
                    if str(i) in shop.catalog.cards and str(i) not in shop.catalog.visited and not shop.catalog.is_gone(str(i))]
            if cards and shop.open_rows(cards, 3):
                judged = self._decline_look()
                named = self._named(judged)
        if not named:
            return ""
        self.hopeful_items = named
        self.hopefuls = [i for ids in named.values() for i in ids]
        reason = f" ({judged.get('reason')})" if judged and judged.get("reason") else ""
        listed = "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items())
        return ("Possible fits spotted by a second review: " + listed + reason
                + ". Open these and order whichever really qualifies; finish without an order only if all of them fail.")

    # ---- sold-out lines
    def stand_ins(self, lost: list[str], twins: list[str] | None = None) -> list[str]:
        shop, catalog = self.shop, self.shop.catalog
        key = "|".join(sorted(lost))
        kept = [i for i in self.swap_memory.get(key, []) if not catalog.is_gone(i) and i not in shop.basket.ids()]
        if shop.desk.closed():
            # no model call is left: the stand-ins an earlier read found still serve (local: the spend stop came between the
            # swap read and the code-made finish, and the sold-out line was ordered)
            return kept[:4]
        # candidates: the near-twins just found, then every listing whose attributes were read, then search previews
        newest = list(reversed(catalog.order))
        ranked = list(twins or []) + [i for i in newest if i in catalog.read] + [i for i in newest if i not in catalog.read]
        text_lines, used = [], set()
        for lid in ranked:
            if lid in used or lid in lost or catalog.is_gone(lid) or lid not in catalog.cards:
                continue
            used.add(lid)
            text_lines.append(flat(shorten(catalog.read.get(lid) or catalog.cards.get(lid), 200))[:1200])
            if len(text_lines) >= 25:
                break
        if not text_lines:
            return []
        ask_text = (SOLD_OUT_SWAP + "\n\nCUSTOMER REQUEST:\n" + shop.request + "\n\nCONDITIONS NOW:\n" + shop.requirements()
                  + "\n\nLINE THAT SOLD OUT:\n" + "\n".join(f"{i}: {catalog.record(i)}" for i in lost)
                  + "\n\nCART CONTENTS:\n" + shop.basket.text() + "\n\nCANDIDATE RECORDS:\n" + "\n".join(text_lines))
        # a filtering job: quick model, low reasoning effort
        judged = shop.desk.verdict(ask_text, [QUICK, CAREFUL], 2000, "low")
        carried = shop.basket.ids()
        fits: list[tuple[float, str]] = []
        for bit in as_list((judged or {}).get("fits")):
            lid = str(bit.get("id") if isinstance(bit, dict) else bit)
            if lid in catalog.cards and lid not in lost and not catalog.is_gone(lid) and lid not in carried and lid not in [i for _, i in fits]:
                price = amount(bit.get("price")) if isinstance(bit, dict) else None
                fits.append((price if price is not None else float("inf"), lid))
        shop.log.append({"kind": "swap", "lost": lost, "fits": fits})
        if judged is None:
            return kept[:4]
        if shop.sheet.pick == "cheapest":
            fits.sort(key=lambda f: f[0])
        else:
            # the sold-out pick is the best evidence of what the shopper wants: the stand-in closest to it comes first
            fits.sort(key=lambda f: tuple(-n for n in catalog.likeness(lost[0], f[1])))
        self.swap_memory[key] = [i for _, i in fits]
        return [i for _, i in fits][:4]

    def twin_search(self, lost: list[str]) -> list[str]:
        """Look for near-twins of a sold-out line (the same product from another seller, a sibling variant): one search with
        the lost listing's own name words, then the closest unread results of its kind get their attributes read."""
        shop, catalog, sheet = self.shop, self.shop.catalog, self.shop.sheet
        twins: list[str] = []
        # the replacement review must be able to run afterwards, or the steps are wasted
        if (shop.steps_left() < shop.reserve_steps() + 9 or shop.desk.seconds_left() < WRAP_UP + 40
                or shop.desk.spent > shop.desk.purse or shop.desk.closed()):
            return twins
        for lost_id in lost[:2]:
            words = catalog.title_words(lost_id)[:6]
            if len(words) < 2:
                continue
            collected = shop.search_for(" ".join(words), limit=40)
            if not collected and len(words) > 3 and shop.steps_left() >= shop.reserve_steps() + 9:
                # a long name can over-narrow a keyword search: once more with its first words
                collected = shop.search_for(" ".join(words[:3]), limit=40)
            kind = catalog.kind(lost_id)
            fitting = [i for i in collected if i != lost_id and catalog.can_buy(i) and (not kind or catalog.kind(i) == kind)
                       and (sheet.budget is None or (catalog.price(i) or 0) <= sheet.budget)]
            fitting.sort(key=lambda i: tuple(-n for n in catalog.likeness(lost_id, i)))
            unread = [i for i in fitting if i not in catalog.read][:2]
            shop.log.append({"kind": "twin_search", "lost": lost_id, "words": words, "found": len(collected), "open": unread})
            shop.fill_facts(unread, shop.reserve_steps() + 6)
            twins += [i for i in fitting[:4] if i not in twins]
        return twins

    def sold_hint(self) -> str:
        """Newly sold-out cart lines: say so and offer already-read stand-ins (once per line)."""
        shop, catalog = self.shop, self.shop.catalog
        lost = [i for i in shop.basket.ids() if i not in catalog.retired and catalog.is_gone(i)]
        if not lost:
            return ""
        for lid in lost:
            catalog.retired[lid] = shop.store.step
        self.hopefuls = [i for i in self.hopefuls if i not in lost]
        shop.epoch += 1
        twins = self.twin_search(lost)
        chosen = self.stand_ins(lost, twins)
        text = f"[Tern] {', '.join(lost)} just sold out for good; skip it from now on."
        if chosen and shop.steps_left() <= 6:
            spare = ", ".join(chosen[1:]) or "none"
            return text + (f" Steps are nearly gone: right now, add {chosen[0]} with the requested units and drop the dead line (two calls), "
                           f"then check out. Other read listings that look suitable: {spare}.")
        if chosen and shop.sheet.pick == "cheapest":
            return text + (f" Read listings that look suitable, lowest price first: {', '.join(chosen)}. Because the lowest price wins, do one "
                           f"search capped below {chosen[0]}'s price (with the search's price ceiling if it offers one) unless you already know "
                           "the full qualifying set, then replace the dead line with the cheapest qualifying listing.")
        if chosen:
            return text + (f" Read listings that look suitable, closest to the sold-out one first: {', '.join(chosen)}. Now remove the "
                           f"dead line, add {chosen[0]} with the requested units, and check out.")
        visited = [i for i in twins if i in catalog.read]
        hint = f" Tern already opened close relatives of it: {', '.join(visited)}; check them first." if visited else ""
        return text + " Take it out and put in another qualifying listing, searching for one if nothing read so far qualifies." + hint

    # ---- chains of pages
    def chains(self, ceiling: float | None = None, floor: float = WRAP_UP) -> tuple[str, bool]:
        """Make sure linked pages behind conditions are open before checkout; returns (report, hold?). A code-made finish
        passes its own spend ceiling and time floor (the loop's have run out by then)."""
        shop, catalog, sheet = self.shop, self.shop.catalog, self.shop.sheet
        ceiling = shop.desk.purse if ceiling is None else ceiling
        pending = sheet.open_pointers()
        tool = sheet.tool_for.get("offer") or sheet.tool_for.get("open")
        id_param = shop.lone_param(tool)
        shown_obs: list[str] = []
        lost: list[str] = []
        while pending and self.chain_rounds < 6 and shop.desk.spent < ceiling and shop.desk.seconds_left() > floor:
            self.chain_rounds += 1
            text_lines = []
            for lid in reversed(catalog.order):
                collected = (catalog.read.get(lid) or catalog.cards.get(lid)) if lid in catalog.visited else None
                text_lines.append("OPENED " + flat(shorten(collected, 300))[:1400] if collected else "ROW " + flat(shorten(catalog.cards.get(lid), 80))[:300])
                if len(text_lines) >= 60:
                    break
            ask_text = (CHAIN_READ + "\n\nCONDITIONS WITH A LINKED VALUE:\n" + "\n".join(pending)
                      + "\n\nRECORDS (latest first):\n" + "\n".join(text_lines))
            judged = shop.desk.verdict(ask_text, [QUICK, CAREFUL], 1800) or {}
            shop.log.append({"kind": "chains", "pending": pending, "got": shorten(judged, 150)})
            reach: list[str] = []
            lost = []
            for chain in as_list(judged.get("chains")):
                if not isinstance(chain, dict):
                    continue
                said = str(chain.get("need") or "").strip()
                need = said if said in pending else (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in as_list(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                val = str(chain.get("value") or "").strip()
                if chain.get("done") is True and links and all(i in catalog.visited for i in ids) and val:
                    sheet.pointer_values[need] = val
                    rule = str(chain.get("rule_value") or "").strip()
                    if rule:
                        sheet.pointer_rules[need] = rule
                    shop.epoch += 1
                    continue
                unopened = [i for i in ids if i in catalog.cards and i not in catalog.visited]
                if unopened:
                    reach += [i for i in unopened if i not in reach]
                    continue
                gap = next((l for l, i in zip(links, ids) if i not in catalog.cards), {})
                about = str(gap.get("about") or "") or "the listing it points at"
                words = str(gap.get("find") or "").strip()
                collected = shop.search_for(words) if words else []
                if collected:
                    reach += [i for i in collected if i not in reach]
                else:
                    lost.append(f"{need} -> find {about}" + (f" (search words: {words})" if words else ""))
            pending = [p for p in pending if p not in sheet.pointer_values]
            if not reach or not tool or not id_param or shop.steps_left() < len(reach) + 4 or shop.store.finished:
                break
            for lid in reach[: max(0, min(3, 6 - len(shown_obs)))]:
                resp = shop.store_call(tool, {id_param: lid})
                if resp is None or shop.store.finished:
                    break
                shown_obs.append(f"{lid}: " + shop.show(resp.get("observation"))[:1500])
        text = ""
        if shown_obs:
            text += "[linked pages opened by Tern]\n" + "\n".join(shown_obs) + "\n"
        if sheet.pointer_values:
            text += "[linked values] " + "; ".join(f"{k} => {v}" for k, v in sheet.pointer_values.items()) + "\n"
        hold = bool(lost) and self.chain_holds < 3 and shop.steps_left() > 5 and shop.desk.seconds_left() > WRAP_UP
        if hold:
            self.chain_holds += 1
            text += ("Checkout is valid only after every linked page behind a condition is fully open (previews do not count), including "
                     "pages those pages point on to. Not yet located: " + " | ".join(lost[:3])
                     + ". Search by seller, brand or wording, open it, use the value it gives, and then check out.")
        return text, hold

    # ---- notes inside results
    def loose_text(self, obs: Any) -> list[str]:
        catalog = self.shop.catalog
        collected: list[str] = []
        if isinstance(obs, dict) and not catalog.lid(obs):
            for val in obs.values():
                if isinstance(val, str) and len(val.split()) >= 5:
                    collected.append(val)
                elif isinstance(val, list) and val and all(isinstance(v, str) for v in val):
                    collected += [v for v in val if len(v.split()) >= 5]
                elif isinstance(val, dict) and not catalog.lid(val, False, 1):
                    collected += [v for v in val.values() if isinstance(v, str) and len(v.split()) >= 6]
        return [line for line in collected if line not in self.notes_heard]

    def note(self, text_lines: list[str]) -> str:
        """Interpret a shop note, apply its effects, and describe them (empty string if none)."""
        shop, sheet, catalog = self.shop, self.shop.sheet, self.shop.catalog
        self.notes_heard.update(text_lines)
        judged = shop.desk.verdict(READ_NOTE + "\n\nCURRENT ITEMS AND CONDITIONS:\n" + shop.requirements() + "\n\nNOTE TEXT:\n" + "\n".join(text_lines),
                                [QUICK, BACKUP], 1200) or {}
        retired = [str(g) for g in as_list(judged.get("gone")) if isinstance(g, (str, int))]
        catalog.off_sale.update(g for g in retired if g in catalog.cards)
        self.note_said_sold = bool(retired) or judged.get("sold_out") is True
        if self.note_said_sold:
            # stores reuse their sell-out sentences: a later refusal made only of sentences already heard is still a sell-out
            self.sold_words.update(text_lines)
        if judged.get("kind") not in (None, "change"):
            return ""
        changed: list[str] = []
        budget = amount(judged.get("budget"))
        if budget is not None and budget > 0 and budget != sheet.budget:
            sheet.budget = budget
            changed.append(f"budget now {budget:g}")
        for bit in as_list(judged.get("drop")):
            if isinstance(bit, dict) and str(bit.get("need") or "").strip():
                want = sheet.want(bit.get("item"))
                id_param = str(bit["need"]).strip()
                hit = [m for m in want.live() if m == id_param] or [m for m in want.live() if plain(m.split(":")[0]) == plain(id_param.split(":")[0])][:1]
                want.given_up.extend(hit or [id_param])
                changed.append(f"dropped {id_param}")
        for bit in as_list(judged.get("add")):
            if isinstance(bit, dict) and str(bit.get("need") or "").strip():
                want = sheet.want(bit.get("item"))
                need = str(bit["need"]).strip()
                if need in want.given_up:
                    want.given_up.remove(need)
                if need not in want.stated + want.heard:
                    want.heard.append(need)
                changed.append(f"added {need}")
        if retired:
            changed.append("unavailable: " + ", ".join(retired))
        if judged.get("price_moved") is True:
            changed.append("a price changed")
        summary = str(judged.get("summary") or "").strip()
        if changed or summary:
            shop.epoch += 1
            shop.last_calls = []
            shop.changes.append(summary or "; ".join(changed))
            shop.changed_step = shop.store.step
            shop.reopen_checks()
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()


class StepGate:
    """Carries out the request's required actions due at checkout or at a no-order finish."""

    def __init__(self, shop: "Shopper"):
        self.shop = shop
        self.cleared: set[tuple[str, str]] = set()
        self.messages = 0
        self.record: list[str] = []
        self.record_steps: list[int] = []

    def stamp(self) -> str:
        return self.shop.basket.mark() + f"#{self.shop.epoch}"

    def due(self, stage: str) -> bool:
        return any(s["when"] == stage for s in self.shop.sheet.steps_asked) and (stage, self.stamp()) not in self.cleared

    def log(self, line: str, step: int) -> None:
        self.record.append(line)
        self.record_steps.append(step)

    def run(self, stage: str) -> tuple[bool, str]:
        shop, sheet = self.shop, self.shop.sheet
        steps = [s for s in sheet.steps_asked if s["when"] == stage]
        if not steps or shop.store.finished or not self.due(stage):
            return True, ""
        enders = {sheet.tool_for.get("buy"), sheet.tool_for.get("refuse")} - {None}
        tools = [tool_line(t) for t in shop.tools if t["function"]["name"] not in enders]
        done_now: list[str] = []
        nudged = False
        for _ in range(GATE_TRIES):
            ask_text = (STEP_CHECK.replace("STAGE", "place the order" if stage == "order" else "end without a purchase")
                      + "\n\nCUSTOMER REQUEST:\n" + shop.request + "\n\nACTIONS DUE:\n" + "\n".join("- " + s["step"] for s in steps)
                      + "\n\nCONDITIONS NOW:\n" + shop.requirements() + "\n\nCART CONTENTS: " + shop.basket.text()
                      + "\n\nRECORD:\n" + ("\n".join(f"{n}. {line}" for n, line in enumerate(self.record, 1)) or "(empty)")
                      + ("\n\nSHOP NOTES:\n" + "\n".join(shop.changes[-4:]) if shop.changes else "")
                      + f"\n\nREMAINING: {max(0, shop.message_cap - shop.messages_used)} messages, {shop.steps_left()} steps "
                      + "(one of them is needed for the final call)" + "\n\nTOOLS:\n" + "\n".join(tools))
            judged = shop.desk.verdict(ask_text, [CAREFUL, QUICK] if nudged else [QUICK, CAREFUL], 2000)
            if judged is None:
                break
            learned: list[str] = []
            hold = ""
            if done_now:
                # conditions stated in replies to the gate's own calls
                shop.apply_answers({"needs": judged.get("needs")}, learned)
                hold = str(judged.get("hold") or "").strip()
            call = judged.get("call") if isinstance(judged.get("call"), dict) else None
            shop.log.append({"kind": "gate", "stage": stage, "call": shorten(call, 200), "hold": hold[:200], "told": learned})
            if learned or hold:
                shop.epoch += 1
                head = "Required actions performed by Tern:\n" + "\n".join(done_now) + "\n" if done_now else ""
                return False, head + "The customer's reply alters the job: " + "; ".join(learned + ([hold] if hold else [])) + ". Adjust the cart accordingly and check out again."
            name = str((call or {}).get("tool") or "")
            reported = [e for e in as_list(judged.get("steps")) if isinstance(e, dict)]
            talking = stage == "order" and any(s["messages"] > 0 for s in steps)
            valid = {n for n in range(1, len(self.record) + 1)
                     if not talking or (n <= len(self.record_steps) and self.record_steps[n - 1] > shop.basket.changed_at)}
            unproven = [e for e in reported if e.get("done") is not True or not any((whole(x) or 0) in valid for x in as_list(e.get("lines")))]
            if not call and unproven and not done_now and not nudged and sheet.tool_for.get("ask") and shop.messages_used < shop.message_cap and shop.steps_left() >= 2:
                nudged = True
                missing = "; ".join(str(e.get("step") or "") for e in unproven)[:200]
                self.log(f"(Tern: since the cart changed at step {shop.basket.changed_at}, nothing in the record performs: {missing}. "
                         f"{shop.message_cap - shop.messages_used} message(s), {shop.steps_left()} steps remain; name the call that performs it)", -1)
                continue
            if not call or name not in shop.schemas or name in enders:
                if not call:
                    self.cleared.add((stage, self.stamp()))
                break
            if shop.steps_left() < 2 or (name == sheet.tool_for.get("ask") and shop.messages_used >= shop.message_cap):
                break
            args, problems = fill_args(call.get("args") if isinstance(call.get("args"), dict) else {}, shop.schemas[name])
            if problems:
                break
            if name == sheet.tool_for.get("ask"):
                text = next((v for v in args.values() if isinstance(v, str)), "")
                resp = shop.message(text, floor=1)
                if resp is None:
                    break
                self.messages += 1
                done_now.append(f"Q: {text}\nA: {resp}")
                continue
            resp = shop.store_call(name, args, fold=True)
            if resp is None:
                break
            shown_obs = resp.get("observation")
            self.log(f"step {shop.store.step}: {name} {flat(args)} -> {shop.show(shown_obs)[:300]} (cart then: {shop.basket.mark()})", shop.store.step)
            done_now.append(f"{name} {flat(args)} -> {shop.show(shown_obs)[:400]}")
        return True, ("Required actions performed by Tern:\n" + "\n".join(done_now)) if done_now else ""


# ======================================================================== one episode


class Shopper:
    def __init__(self, problem: dict[str, Any]):
        world = problem["environment"]
        self.view = world["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or []
                      if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.schemas: dict[str, dict[str, Any]] = {}
        for tool in self.tools:
            params = tool["function"].get("parameters")
            self.schemas[tool["function"]["name"]] = params if isinstance(params, dict) else {"type": "object", "properties": {}}
        cap = whole(self.view.get("max_steps"))
        if cap is None and isinstance(self.view.get("max_steps"), float):
            cap = int(self.view["max_steps"])
        self.step_limit = cap if cap and cap > 0 else 30
        self.log: list[dict[str, Any]] = []
        self.store = StoreLink(world["binding"], self.log)
        self.store.finished = bool(self.view.get("finished"))
        self.desk = ModelDesk(self.log)
        self.sheet = TaskSheet()
        self.catalog = Catalog(self.sheet, self.schemas)
        self.basket = Basket(self.catalog, self.schemas, self.sheet.tool_for)
        self.reviews = Reviews(self)
        self.gate = StepGate(self)
        self.asker = Interviewer(self)
        self.epoch = 0
        self.turns = 0
        self.turn_cap = TURNS_BASE
        self.blanks = 0
        self.message_cap = 0
        self.messages_used = 0
        self.held_back = 0
        self.message_refused = False
        self.changes: list[str] = []
        self.changed_step = -99
        self.pending_notes: list[str] = []
        self.last_calls: list[str] = []
        self.cache: dict[str, tuple[int, int, str]] = {}
        self.attrs_from: dict[str, bool] = {}
        self.compare_good = 0
        self.compare_fail = 0
        self.compare_missed = False
        self.pages_done = 0
        self.paged: list[Any] = []
        self.offer_read_at: dict[str, int] = {}
        self.refusal_step = -1
        self.orders_tried = 0
        self.decline_pushes = 0
        self.decline_holds = 0
        self.late_hold_used = False
        self.wide_held = False
        self.cheaper_done: set[str] = set()
        self.value_leads: list[str] = []
        self.row_hits: dict[str, int] = {}
        self.lead_holds = 0
        self.proof_holds = 0
        self.cut_short = ""
        self.parts_searched: set[str] = set()
        self.sold_holds = 0
        self.line_hold_count = 0
        self.filled_in = False
        self.declines_refused = 0
        self.decline_chained = False
        self.compat_holds = 0
        self.set_holds = 0
        self.history: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []

    # ---- counts and clocks
    def steps_left(self) -> int:
        return self.step_limit - self.store.step

    def messages_free(self) -> int:
        """Questions still allowed, after setting aside messages that required actions will need."""
        return self.message_cap - self.messages_used - max(0, self.held_back - self.gate.messages)

    def reserve_steps(self) -> int:
        """Steps held back for finishing (two while a checkout action still needs a message)."""
        talk = any(s["when"] == "order" and s["messages"] > 0 for s in self.sheet.steps_asked)
        return 2 if talk and self.messages_used < self.message_cap and self.gate.due("order") else 1

    def reopen_checks(self) -> None:
        """After the job changes, the one-shot safeguards become available again."""
        self.reviews.declines_reviewed = False
        self.decline_pushes = 0
        self.decline_holds = 0
        self.lead_holds = 0
        self.late_hold_used = False
        self.wide_held = False

    # ---- tool roles
    def role(self, job: str) -> str | None:
        return self.sheet.tool_for.get(job)

    def lone_param(self, tool: str | None) -> str | None:
        """Name of the sole id parameter of a one-listing tool, else None."""
        form = self.schemas.get(tool or "") or {}
        params = form.get("properties") or {}
        text_fields = free_text_params(params)
        needed = form.get("required") or []
        id_param = next((k for k in needed if k in text_fields), None) or (text_fields[0] if len(text_fields) == 1 else None)
        if not tool or not id_param or any(k != id_param for k in needed):
            return None
        return id_param

    def batch_param(self) -> str | None:
        form = self.schemas.get(self.role("compare") or "") or {}
        id_param = next((k for k, s in (form.get("properties") or {}).items() if isinstance(s, dict) and "array" in kinds_of(s)), None)
        if not id_param or any(k != id_param for k in form.get("required") or []):
            return None
        return id_param

    def batch_size(self) -> int:
        """Ids per multi-listing call: the schema's cap until a call fails, then a smaller size learned from the shop."""
        sch = ((self.schemas.get(self.role("compare") or "") or {}).get("properties") or {}).get(self.batch_param() or "") or {}
        stated = sch.get("maxItems") if isinstance(sch.get("maxItems"), int) else None
        if self.compare_fail:
            room = max(2, self.compare_good, min(self.compare_fail - 1, max(2, self.compare_fail // 2)))
            return min(room, stated) if stated else room
        return stated if stated else max(4, self.compare_good)

    def learn_compare(self, ids: list[Any], obs: Any) -> bool:
        """Learn from a multi-listing call whether that many ids work (unknown ids do not count as a size failure)."""
        if len(ids) < 2:
            return True
        shown = flat(obs)
        worked = any(str(i) in shown for i in ids)
        if worked:
            self.compare_good = max(self.compare_good, len(ids))
        elif all(str(i) in self.catalog.cards for i in ids):
            self.compare_fail = min(self.compare_fail or 999, len(ids))
        return worked

    def attr_view(self) -> str | None:
        """One-listing tool known (or presumed) to display attributes."""
        for tool, showed in self.attrs_from.items():
            if showed and self.lone_param(tool):
                return tool
        tool = self.role("open")
        return tool if self.attrs_from.get(tool or "") is not False and self.lone_param(tool) else None

    def batch_blind(self) -> bool:
        """True when the multi-listing view is not the way to read attributes here."""
        shown_obs = self.attrs_from.get(self.role("compare") or "")
        if shown_obs is False:
            return True
        return shown_obs is None and (self.attrs_from.get(self.attr_view() or "") is True or bool(self.role("offer")))

    def page_views(self) -> set[str]:
        page = self.role("offer") or self.role("open")
        return {t for t in (page, self.role("compare")) if t}

    def read_from_note(self) -> str:
        facts = self.attr_view()
        if not facts or not (self.attrs_from.get(facts) is True or self.role("offer")):
            return ""
        bare = sorted(t for t in self.page_views() | {self.role("offer") or ""} if t and self.attrs_from.get(t) is False)
        pages = sorted(t for t in self.page_views() if t != facts)
        text = f"attributes are shown by {facts}"
        if bare:
            text += f"; {', '.join(bare)} omit them"
        if pages:
            text += f"; full product pages (for linked-page chains and page checks) come from {' or '.join(pages)}"
        return text

    def requirements(self) -> str:
        judged = {}
        for _, need in self.sheet.live_needs():
            key = self.catalog.deciding_key(need)
            if key:
                judged[need] = key
        return self.sheet.summary(self.read_from_note(), judged)

    # ---- the request
    def read_request(self) -> None:
        catalogue = "\n".join(map(tool_line, self.tools))
        ask_text = READ_REQUEST + "\nSHOP TOOLS:\n" + catalogue + "\n\nCUSTOMER REQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            ask_text += "\n\nSHOP'S OPENING MESSAGE:\n" + flat(shorten(self.view.get("opening"), 1500))
        # quick model first; careful model only if the first reading lacks items or a checkout tool
        raw = self.desk.verdict(ask_text, [QUICK], 4000) or {}
        usable = as_list(raw.get("items")) and isinstance(raw.get("roles"), dict) and raw["roles"].get("buy")
        if not usable and not self.desk.no_credit:
            raw = self.desk.verdict(ask_text, [CAREFUL, QUICK], 2500) or raw
        self.sheet.take(raw, set(self.schemas))
        self.basket.tool_for = self.sheet.tool_for
        buy, refuse = self.role("buy"), self.role("refuse")
        if buy and buy == refuse and (not self.sheet.buy_args or not self.sheet.refuse_args or self.sheet.buy_args == self.sheet.refuse_args):
            split = self.desk.verdict(SPLIT_MODES.format(tool=buy, form=flat(self.schemas.get(buy))), [QUICK, CAREFUL], 600) or {}
            if isinstance(split.get("buy_args"), dict) and isinstance(split.get("refuse_args"), dict) and split["buy_args"] != split["refuse_args"]:
                self.sheet.buy_args, self.sheet.refuse_args = split["buy_args"], split["refuse_args"]
        self.separate_pointers()
        limit = whole(raw.get("messages"))
        if limit is not None and limit >= 0:
            self.message_cap = limit
        elif self.role("ask"):
            self.message_cap = 3
        steps = whole(raw.get("steps"))
        if steps is not None and 10 <= steps < self.step_limit:
            self.step_limit = steps
        if self.role("ask"):
            # only an action due at the checkout keeps a message back. What the request wants said or offered when nothing
            # qualifies goes into the finish's own text (race 13: a message held for "may I offer the closest one?" left the
            # interview one question short in most episodes, and every such offer drew a puzzled reply)
            talk = [s["messages"] for s in self.sheet.steps_asked if s["when"] == "order" and s["messages"]]
            self.held_back = min(max(talk), self.message_cap) if talk else 0
        # more steps allowed -> more turns allowed (time and money caps still apply)
        self.turn_cap = max(TURNS_BASE, min(110, self.step_limit))
        self.desk.call_cap = max(CALLS_BASE, self.turn_cap + 25)
        self.desk.size_purse(len(self.sheet.wants), self.step_limit)

    def separate_pointers(self) -> None:
        """Fold conditions that merely describe a linked listing into the linked condition."""
        sheet = self.sheet
        for number, want in enumerate(sheet.wants, 1):
            pointing = [need for n, need in sheet.pointers if n == number and need in want.stated]
            others = [m for m in want.stated if m not in pointing]
            if not pointing or not others:
                continue
            ask_text = (SEPARATE + "\n\nCUSTOMER REQUEST:\n" + self.request + f"\n\nPRODUCT: {want.name}\nLINKED REQUIREMENTS:\n"
                      + "\n".join(pointing) + "\nOTHER REQUIREMENTS:\n" + "\n".join(others))
            judged = self.desk.verdict(ask_text, self.desk.careful_first(), 500) or {}
            describing = {str(x).strip() for x in as_list(judged.get("describe_other"))}
            moved = [m for m in others if m in describing]
            if not moved:
                continue
            folded = pointing[0] + " (linked listing described as: " + "; ".join(moved) + ")"
            want.stated = [folded if m == pointing[0] else m for m in want.stated if m not in moved]
            sheet.pointers = [(n, folded if (n, need) == (number, pointing[0]) else need) for n, need in sheet.pointers]
            self.log.append({"kind": "separate", "item": number, "moved": moved})

    # ---- talking to the shopper
    def message_args(self, text: str) -> dict[str, Any] | None:
        form = self.schemas.get(self.role("ask") or "") or {}
        params = form.get("properties") or {}
        text_fields = free_text_params(params)
        id_param = next((k for k in form.get("required") or [] if k in text_fields), None) or (text_fields[0] if text_fields else None)
        if not id_param:
            return None
        args: dict[str, Any] = {id_param: text}
        for name in form.get("required") or []:
            if name in args:
                continue
            options = (params.get(name) or {}).get("enum")
            if not isinstance(options, list) or not options:
                return None
            args[name] = options[0]
        args, problems = fill_args(args, form)
        return None if problems else args

    @staticmethod
    def spoken(obs: Any) -> str:
        if isinstance(obs, str):
            return obs
        collected = walk_strings(obs)
        return " ".join(collected)[:800] if collected else flat(obs)[:400]

    def message(self, text: str, floor: int = 6) -> str | None:
        """Message the customer once; returns the reply text or None."""
        tool = self.role("ask")
        if not tool or self.store.finished or self.steps_left() <= floor or self.messages_used >= self.message_cap or self.message_refused:
            return None
        args = self.message_args(text)
        if not args:
            return None
        resp = self.store.call(tool, args)
        if resp is None:
            return None
        self.messages_used += 1
        answer = resp.get("observation")
        words = self.spoken(answer)
        self.gate.log(f"step {self.store.step} | asked customer {text!r} | reply {words[:300]!r} | cart {self.basket.mark()}", self.store.step)
        self.log.append({"kind": "message", "q": text, "a": words[:300]})
        if isinstance(answer, dict) and answer.get("error"):
            # refused: the true message limit is lower than assumed
            self.message_refused = True
            self.message_cap = self.messages_used
        return words

    def apply_answers(self, heard: dict[str, Any], learned: list[str]) -> None:
        sheet = self.sheet
        several = len(sheet.wants) > 1
        for bit in as_list(heard.get("needs")):
            if not isinstance(bit, dict) or not str(bit.get("need") or "").strip():
                continue
            number = sheet.index_of(bit.get("item"))
            need = str(bit["need"]).strip()
            want = sheet.wants[number - 1]
            if need not in want.heard:
                want.heard.append(need)
                learned.append(f"item {number}: {need}" if several else need)
            if bit.get("points") is True and (number, need) not in sheet.pointers:
                sheet.pointers.append((number, need))
        for bit in as_list(heard.get("drop")):
            if isinstance(bit, dict) and str(bit.get("property") or "").strip():
                want = sheet.want(bit.get("item"))
                head = plain(str(bit["property"]).split(":")[0])
                hit = [m for m in want.live() if plain(m.split(":")[0]) == head]
                want.given_up.extend(hit)
                learned.extend(f"dropped {m}" for m in hit)
        for bit in as_list(heard.get("free")):
            if isinstance(bit, dict) and bit.get("property"):
                want = sheet.want(bit.get("item"))
                said = roots(str(bit["property"]))
                # a "don't mind" about a property the request or a reply states leaves that condition standing
                if any(same_property(said, roots(m.split(":")[0])) or same_property(roots(m.split(":")[0]), said) for m in want.live()):
                    continue
                want.open_on.append(str(bit["property"]))
        budget = amount(heard.get("budget"))
        if budget is not None and budget > 0 and not isinstance(heard.get("budget"), str):
            sheet.budget = budget
        if learned:
            self.reopen_checks()

    def drain_notes(self) -> str:
        if not self.pending_notes:
            return ""
        text = "\n[shop notes meanwhile] " + " | ".join(self.pending_notes)
        self.pending_notes = []
        return text

    # ---- the store
    def show(self, obs: Any) -> str:
        text = flat(shorten(obs, ROW_CHARS))
        return text if len(text) <= REPLY_CHARS else text[:REPLY_CHARS] + "...(cut)"

    def fold(self, tool: str, obs: Any) -> str:
        """Record a result of Tern's own call and interpret any note inside it."""
        self.catalog.take(obs)
        text_lines = self.reviews.loose_text(obs)
        if not text_lines:
            return ""
        meaning = self.reviews.note(text_lines)
        if meaning:
            self.pending_notes.append(f"(seen on {tool}) {meaning}")
        return meaning

    def store_call(self, tool: str, args: dict[str, Any], fold: bool = True) -> dict[str, Any] | None:
        """A store step taken by Tern directly (schema-shaped first)."""
        args, problems = fill_args(args, self.schemas.get(tool))
        if problems or self.store.finished:
            return None
        resp = self.store.call(tool, args)
        if resp is None:
            return None
        shown_obs = resp.get("observation")
        if tool in self.page_views() and not (isinstance(shown_obs, dict) and shown_obs.get("error")):
            for val in args.values():
                self.catalog.visited.update(v.strip() for v in (val if isinstance(val, list) else [val]) if isinstance(v, str))
        if fold:
            self.fold(tool, shown_obs)
        return resp

    def note_attrs(self, tool: str, obs: Any) -> None:
        cards = self.catalog.listings_in(obs)
        if cards:
            self.attrs_from[tool] = self.attrs_from.get(tool, False) or any(attribute_block(r) for _, r in cards)

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Open unread listings so their attributes can be judged, never going below `keep` remaining steps."""
        tool = self.attr_view()
        id_param = self.lone_param(tool)
        if not tool or not id_param:
            return []
        shown = []
        for lid in [i for i in ids if i in self.catalog.cards and i not in self.catalog.read and not self.catalog.is_gone(i)]:
            if self.steps_left() <= keep or self.store.finished or self.attrs_from.get(tool) is False:
                break
            resp = self.store_call(tool, {id_param: lid})
            if resp is None:
                break
            shown_obs = resp.get("observation")
            self.catalog.note_view(tool, shown_obs, self.epoch)
            if isinstance(shown_obs, dict) and self.catalog.lid(shown_obs):
                self.attrs_from[tool] = self.attrs_from.get(tool, False) or bool(attribute_block(shown_obs))
                if lid not in self.catalog.read and attribute_block(shown_obs, 1):
                    self.catalog.read[lid] = dict(self.catalog.cards.get(lid) or {})
            shown.append(f"{lid}: " + self.show(shown_obs)[:1500])
        if shown:
            self.log.append({"kind": "fill_facts", "n": len(shown)})
        return shown

    def widen(self) -> None:
        """Open more listings per item before the interview so coverage counts mean something."""
        tool, id_param = self.role("compare"), self.batch_param()
        if not tool or not id_param or self.batch_blind():
            self.widen_singly()
            return
        if self.steps_left() <= 12:
            return
        room = self.batch_size()
        if room < 2:
            self.widen_singly()
            return
        groups = self.catalog.groups()
        share = max(1, room // max(1, len(groups)))
        chosen: list[str] = []
        for ids in groups:
            chosen += [i for i in ids if i not in self.catalog.read][:share]
        chosen = chosen[:room]
        for attempt in range(2):
            if len(chosen) < 2 or self.steps_left() <= 12:
                return
            before = len(self.catalog.read)
            resp = self.store_call(tool, {id_param: chosen})
            if resp is None:
                return
            shown_obs = resp.get("observation")
            self.catalog.note_view(tool, shown_obs, self.epoch)
            self.note_attrs(tool, shown_obs)
            worked = self.learn_compare(chosen, shown_obs)
            self.log.append({"kind": "widen", "ids": chosen, "opened": len(self.catalog.read) - before})
            if self.catalog.listings_in(shown_obs) and self.attrs_from.get(tool) is False:
                self.widen_singly()  # no attributes in that view
                return
            if worked or len(self.catalog.read) > before or len(chosen) <= 2:
                return
            # empty result at that size: retry smaller
            chosen = chosen[: min(len(chosen) - 1, self.batch_size())]

    def widen_singly(self) -> None:
        tool = self.attr_view()
        if not tool or not self.lone_param(tool):
            return
        lanes: list[list[str]] = []
        single = len(self.sheet.wants) == 1
        for ids in self.catalog.groups():
            done = sum(1 for i in ids if i in self.catalog.read)
            pool = [i for i in ids if i not in self.catalog.read and self.catalog.can_buy(i)]
            if self.sheet.pick == "cheapest":
                pool = self.catalog.cheap_order(pool)
            lanes.append(pool[: max(0, min(4 if single else 2, 8 - done))])
        queue = [lane[k] for k in range(4) for lane in lanes if k < len(lane)]
        self.fill_facts(queue[: min(8, max(4, 2 * len(lanes)))], 22)

    def cover_unread(self) -> None:
        """Before the question plan, on a many-item request: an item none of whose listings has been read gets one search by
        its name and two of its rows read, so the plan sees that item's attribute names too (race 14: a quarter of the
        held-back properties sat on items with no listing read before the first question). Only while 20 steps remain."""
        if len(self.sheet.wants) < 2:
            return
        for number, want in enumerate(self.sheet.wants, 1):
            if self.steps_left() < 20 or self.store.finished:
                return
            if any(lid in self.catalog.read for lid in self.catalog.ids_of(number)):
                continue
            rows = [lid for lid in self.catalog.ids_of(number) if lid not in self.catalog.read and self.catalog.can_buy(lid)]
            if not rows:
                rows = [lid for lid in self.search_for(want.name, 12) if self.catalog.owner(self.catalog.kind(lid)) == number]
            self.log.append({"kind": "cover_unread", "item": number, "rows": len(rows)})
            self.fill_facts(rows[:2], 18)

    def note_leads(self, args: dict[str, Any], obs: Any) -> None:
        """Rows a search for a required value returned. The shop's word search also reads attributes a row does not show
        (race 14: the passing listing came back for "low fat single" with neither word in its title), so these unopened rows
        are the ones worth opening first."""
        several = len(self.sheet.wants) > 1
        for lid, _ in self.catalog.listings_in(obs):
            # a row that keeps coming back across searches is worth opening before yet another query variant (race 14:
            # fitting rows topped three or four searches and were never opened)
            self.row_hits[lid] = self.row_hits.get(lid, 0) + 1
            if self.row_hits[lid] == 2 and lid not in self.catalog.read and lid not in self.value_leads and self.catalog.can_buy(lid) \
                    and (not several or self.catalog.owner(self.catalog.kind(lid))):
                self.value_leads.append(lid)
        form = (self.schemas.get(self.role("search") or "") or {}).get("properties") or {}
        labels = {self.catalog.kind(i).lower() for i in self.catalog.order} - {""}
        # the words of the search itself: free-text values, not a category label or an option
        said = roots(" ".join(str(args[k]) for k in free_text_params(form) if isinstance(args.get(k), str)
                              and args[k].strip().lower() not in labels))
        wanted: set[str] = set()
        for _, need in self.sheet.live_needs():
            prop, _, value = need.partition(":")
            wanted |= {w for w in roots(value) - roots(prop) if len(w) >= 3 or w.isdigit()}
        if not said & wanted:
            self.value_leads = self.value_leads[-12:]
            return
        for lid, _ in self.catalog.listings_in(obs):
            mine = self.catalog.owner(self.catalog.kind(lid)) if several else 1
            if mine and lid not in self.catalog.read and lid not in self.value_leads and self.catalog.can_buy(lid):
                self.value_leads.append(lid)
        self.value_leads = self.value_leads[-12:]

    def open_leads(self) -> list[str]:
        return [lid for lid in self.value_leads if lid not in self.catalog.read and lid not in self.catalog.visited and self.catalog.can_buy(lid)]

    def open_rows(self, cards: list[str], spare: int = 5) -> bool:
        """Open preview-only listings: together when possible, otherwise a few one at a time; `spare` steps stay free."""
        joiner, many = self.role("compare"), self.batch_param()
        tool = self.attr_view()
        id_param = self.lone_param(tool)
        if joiner and many and not self.batch_blind():
            plan = [(joiner, {many: cards[: self.batch_size()]})]
        elif tool and id_param:
            room = 4 if self.steps_left() >= 15 and self.desk.seconds_left() > WRAP_UP + 30 else 2 if self.steps_left() >= spare + 2 else 1
            plan = [(tool, {id_param: i}) for i in cards[:room]]
        else:
            return False
        if self.steps_left() < len(plan) + spare:
            return False
        before = len(self.catalog.read)
        for name, args in plan:
            resp = self.store_call(name, args)
            if resp is None or self.store.finished:
                break
            if name == joiner:
                self.learn_compare(args[many], resp.get("observation"))
        self.log.append({"kind": "decline_open", "rows": cards, "opened": len(self.catalog.read) - before})
        return len(self.catalog.read) > before

    def search_for(self, words: str, limit: int = 4) -> list[str]:
        """Search these words through the free-text parameter an earlier search used for words (not for a category
        label), with that search's fixed options; returns ids found."""
        tool = self.role("search")
        form = self.schemas.get(tool or "") or {}
        params = form.get("properties") or {}
        free = free_text_params(params)
        if not tool or not free or self.steps_left() < 8 or not words:
            return []
        labels = {self.catalog.kind(i).lower() for i in self.catalog.order} - {""}

        def labelish(val: str) -> bool:
            low = val.strip().lower()
            return low in labels or any(low and low in label for label in labels)

        def score(name: str) -> tuple[int, int, int]:
            sch = params.get(name) or {}
            hinted = bool(re.search(r"query|text|keyword|term|search|words|phrase|^q$", name + " " + str(sch.get("description") or ""), re.I))
            used = [p[name] for p in self.searches if isinstance(p.get(name), str) and p[name].strip()]
            worded = sum(1 for v in used if len(v.split()) >= 2 and not labelish(v))
            return int(hinted), worded, sum(1 for v in used if not labelish(v))

        field_name = max(free, key=score)
        origin = next((p for p in reversed(self.searches) if isinstance(p.get(field_name), str) and p[field_name].strip()), {})
        base = {k: v for k, v in origin.items()
                if k != field_name and (k in (form.get("required") or []) or isinstance((params.get(k) or {}).get("enum"), list))}
        resp = self.store_call(tool, {**base, field_name: words})
        if resp is None:
            return []
        return [i for i, _ in self.catalog.listings_in(resp.get("observation"))][:limit]

    # ---- result layout and paging
    def layout(self, tool: str, obs: Any) -> dict[str, Any] | None:
        """Per tool, learn (once, via the quick model) where rows, totals, paging, price and stock sit in its results."""
        catalog = self.catalog
        if tool in catalog.layouts:
            return catalog.layouts[tool]
        if not isinstance(obs, dict):
            return None
        lists = [k for k, v in obs.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        catalog.layouts[tool] = None
        sample = {k: (v[:3] if k in lists else v) for k, v in obs.items()}
        ask_text = (RESULT_PARTS + "\n\nTOOL: " + tool + " | parameters: " + flat(self.schemas.get(tool) or {})
                  + "\n\nRESULT (first rows only):\n" + flat(shorten(sample, 120))[:4000])
        judged = self.desk.verdict(ask_text, [QUICK, CAREFUL], 1500) or {}
        cards = judged.get("rows") if judged.get("rows") in lists else None
        if not cards:
            return None
        first = obs[cards][0]
        params = (self.schemas.get(tool) or {}).get("properties") or {}

        def present(name: Any, where: dict[str, Any]) -> str | None:
            return name if isinstance(name, str) and name in where else None

        parts = {"rows": cards, "count": present(judged.get("count"), obs), "next": present(judged.get("next"), obs),
                 "next_arg": present(judged.get("next_arg"), params), "price": present(judged.get("price"), first), "stock": present(judged.get("stock"), first)}
        if parts["count"] and whole(obs[parts["count"]]) is None:
            parts["count"] = None
        catalog.layouts[tool] = parts
        catalog.price_field = catalog.price_field or parts["price"]
        catalog.stock_field = catalog.stock_field or parts["stock"]
        self.log.append({"kind": "layout", "tool": tool, "parts": parts})
        return parts

    def more_pages(self, tool: str, args: dict[str, Any], obs: Any) -> str:
        """For ranked picks, fetch short remainders of a search directly; report long ones so the loop narrows down."""
        self.paged = []
        if self.sheet.pick not in ("cheapest", "most", "least"):
            return ""
        parts = self.layout(tool, obs)
        if not parts or not parts["count"] or not parts["next"] or not parts["next_arg"] or not isinstance(obs, dict):
            return ""
        cards, total, token = obs.get(parts["rows"]), whole(obs.get(parts["count"])), obs.get(parts["next"])
        if not isinstance(cards, list) or not cards or total is None:
            return ""
        if token is not None and (isinstance(token, bool) or not isinstance(token, (str, int))):
            return ""
        param = parts["next_arg"]
        given = args.get(param)
        paging_already = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        shown = len(cards)
        if (token is None and isinstance(given, int) and not isinstance(given, bool) and given >= shown
                and given + shown < total and not self.cut_short):
            # race 14: a category listing ended at row 96 of the 222 it reported; the cheapest fit sat past that row
            self.cut_short = f"{given + shown} of {total}"
            return (f"\n[Tern] The shop stopped this listing at row {given + shown} of the {total} it reports: the rest is reached only by "
                    "other searches (each condition's value by itself, the product's plain name, words from qualifying titles), and a "
                    "ranked pick counts them too.")
        if token is None or paging_already or total <= shown:
            return ""
        prices = [self.catalog.price(i) for i, _ in self.catalog.listings_in(obs)]
        if self.sheet.pick == "cheapest" and len(prices) >= 3 and None not in prices and prices == sorted(prices):
            return ""
        pages = -(-(total - shown) // shown)
        if pages > 2 or self.pages_done + pages > SELF_PAGES or self.steps_left() - self.reserve_steps() - pages < 12:
            return (f"\n[Tern] {total} matches, {shown} shown, unsorted by price. A ranked pick must consider all qualifying listings: "
                    f"search each condition's value by itself (one or two words, no other filters), or continue with {param}={flat(token)}.")
        text_fields = []
        for _ in range(pages):
            call, problems = fill_args({**args, param: token}, self.schemas.get(tool))
            if problems:
                break
            resp = self.store_call(tool, call)
            if resp is None or self.store.finished:
                break
            self.pages_done += 1
            shown_obs = resp.get("observation")
            self.paged.append(shown_obs)
            text = self.show(shown_obs)
            self.cache[tool + flat(call)] = (self.epoch, self.store.step, text)
            text_fields.append(f"{param}={flat(token)}: {text}")
            after = self.layout(tool, shown_obs)
            token = shown_obs.get(after["next"]) if after and after["next"] and isinstance(shown_obs, dict) else None
            if token is None:
                break
        self.log.append({"kind": "pages", "tool": tool, "read": len(text_fields)})
        return ("\n[remaining pages fetched by Tern, one step each]\n" + "\n".join(text_fields)) if text_fields else ""

    def cheapest_rows(self, observations: list[Any]) -> str:
        if self.sheet.pick != "cheapest" or not self.catalog.price_field:
            return ""
        collected = {i for shown_obs in observations for i, _ in self.catalog.listings_in(shown_obs)}
        priced = sorted((self.catalog.price(i), i) for i in collected if self.catalog.can_buy(i) and self.catalog.price(i) is not None)
        if len(priced) < 2:
            return ""
        return "\n[Tern] available here by price: " + ", ".join(f"{i} ({p:g})" for p, i in priced[:6])

    def other_records(self, skip: list[str], limit: int = 30) -> str:
        catalog = self.catalog
        order = list(reversed(catalog.order))
        if self.sheet.pick == "cheapest":
            order.sort(key=lambda i: (i not in catalog.read, catalog.price(i) if catalog.price(i) is not None else float("inf")))
        text_lines = []
        for lid in order:
            if lid in skip or catalog.is_gone(lid):
                continue
            text_lines.append(flat(shorten(catalog.read.get(lid) or catalog.cards.get(lid), 160))[:900])
            if len(text_lines) >= limit:
                break
        return "\n".join(text_lines)


# ======================================================================== the store loop


class Episode(Shopper):
    """Tool loop driven by the quick model, with Tern vetting calls and guarding both kinds of ending."""

    NOT_RUN = "Skipped (no step spent): "
    HELD = "On hold (no step spent). "

    # ---- what the loop sees
    def status(self) -> str:
        left = self.steps_left()
        text = f"[Tern] {self.store.step}/{self.step_limit} steps used, {left} remaining."
        if left <= self.reserve_steps() + 1:
            text += (" LAST STEPS: if every item has a cart line, check out now. A no-order finish is right only if searching proved some "
                     "item has no qualifying listing; a lack of steps proves nothing.")
        elif left <= 6 or (left <= len(self.sheet.wants) + 5 and self.orders_tried == 0):
            text += " Almost out of steps: put the chosen lines in the cart and check out."
        elif self.orders_tried == 0 and left <= 12:
            text += " Steps getting scarce: decide, fill the cart and check out soon."
        elif self.orders_tried == 0 and not self.basket.touched and self.store.step >= self.step_limit // 2:
            text += " Half the steps are gone with an empty cart: add a qualifying listing for each item now (sell-outs happen)."
        seconds = self.desk.seconds_left()
        if seconds < WRAP_UP + 50:
            text += f" Clock nearly out (about {max(0, int(seconds))} s left): cart what qualifies and check out."
        waiting = self.sheet.open_pointers()
        if waiting and self.orders_tried == 0:
            text += ("\n[unresolved linked values: open the referenced listing and any listing its page sends you to; the final page "
                     "decides] " + " | ".join(waiting[:3]))
        if self.reviews.hopefuls:
            text += "\n[flagged by review] " + ", ".join(self.reviews.hopefuls)
        leads = self.open_leads()
        if leads and self.orders_tried == 0:
            text += ("\n[rows returned by a search for a required value or by several searches, not opened yet: open these before "
                     "trying another query] " + ", ".join(leads[:4]))
        if self.changes:
            text += "\n[recent changes] " + " | ".join(self.changes[-4:])
        return text + "\n[in force]\n" + self.requirements()

    def offered(self) -> list[dict[str, Any]]:
        """Tools available to the loop right now (never the message tool; fewer as steps run out)."""
        left = self.steps_left()
        enders = {self.role("buy"), self.role("refuse")} - {None}
        allowed: set[str | None] | None = None
        if left <= self.reserve_steps():
            allowed = set(enders)
        elif left <= self.reserve_steps() + 3:
            allowed = set(enders) | {self.role("cart"), self.role("cart_view")}
        elif left <= len(self.sheet.wants) + 5 and self.orders_tried == 0 and self.changed_step < self.store.step - 2:
            # (a note that just changed the job keeps searching open: race 13's budget cut five steps before the end left
            # no search to find the cheaper listing it made the right one)
            allowed = set(enders) | {self.role("cart"), self.role("cart_view"), self.role("offer")}
        tools = [t for t in self.tools if t["function"]["name"] != self.role("ask")]
        if allowed is not None and enders:
            tools = [t for t in tools if t["function"]["name"] in allowed] or tools
        return tools

    def is_order(self, name: str, args: dict[str, Any]) -> bool:
        return name == self.role("buy") and all(args.get(k) == v for k, v in self.sheet.buy_args.items())

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:
        return name == self.role("refuse") and all(args.get(k) == v for k, v in self.sheet.refuse_args.items())

    def quiet_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Fills the text fields of a finish without an order (see closing_note)."""
        form = self.schemas.get(name) or {}
        props = form.get("properties") or {}
        mode_keys = set(self.sheet.refuse_args) | set(self.sheet.buy_args)
        writable = [k for k, sch in props.items()
                    if isinstance(sch, dict) and "string" in kinds_of(sch) and "enum" not in sch and k not in mode_keys]
        out = dict(args)
        for key in writable:
            wanted = key in (form.get("required") or []) or out.get(key) is not None or len(writable) == 1
            if wanted:
                out[key] = self.closing_note(str(out.get(key) or ""))[: (props.get(key) or {}).get("maxLength", 700)]
        return out

    def closing_note(self, said: str) -> str:
        """The no-order finish in the request's terms. Race 13's correct finishes of this kind started with the wording the
        request set for that case and pointed to the closest listing by id, and Tern's fixed sentence failed both of them;
        so the loop's own sentence is kept, led by that wording, with the closest listing added unless the request bars
        suggestions (then sentences naming a listing are dropped)."""
        cards = self.catalog.cards
        names_one = lambda part: any(lid in part for lid in cards)
        sentences = [x for x in re.split(r"(?<=[.!?;])\s+", " ".join(said.split())) if x]
        barred = self.sheet.quiet() is True
        if barred:
            sentences = [x for x in sentences if not names_one(x)]
        note = " ".join(sentences) or "I could not find anything here that meets every one of your requirements, so no order was placed."
        if not barred and not names_one(note):
            closest = self.closest_line()
            note = f"{note} {closest}" if closest else note
        lead = str(self.sheet.raw.get("decline_words") or "").strip(" \"'")
        if lead and plain(lead) not in plain(note[: len(lead) + 40]):
            note = lead.rstrip(".:!") + ". " + note
        return note

    def closest_line(self) -> str:
        """'Closest listing: <title> (<id>, <price>)' for each requested item: the decline review's choice when it made one,
        otherwise the latest opened listing of that item that can still be bought."""
        catalog = self.catalog
        title_key = catalog.name_field()
        found = []
        for number in range(1, len(self.sheet.wants) + 1):
            lid = self.reviews.nearest.get(str(number)) or next(
                (i for i in reversed(catalog.order) if i in catalog.read and catalog.can_buy(i)
                 and (len(self.sheet.wants) == 1 or catalog.owner(catalog.kind(i)) == number)), "")
            if lid:
                title = " ".join(str((catalog.cards.get(lid) or {}).get(title_key) or "").split())[:90] if title_key else ""
                cost = catalog.price(lid)
                found.append(f"{title} ({lid}{'' if cost is None else ', ' + format(cost, 'g')})")
        return "Closest listing: " + " | ".join(found) + "." if found else ""

    def by_title(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Map a title given where an id belongs to the matching id, or explain the problem."""
        if name not in {self.role(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.catalog.key:
            return args, ""
        params = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, s in params.items() if isinstance(s, dict) and ("string" in kinds_of(s) or "array" in kinds_of(s)) and "enum" not in s]
        if len(free) != 1 or (name != self.role("cart") and (name, free[0]) not in self.basket.id_params):
            return args, ""
        key = free[0]
        if key not in args:
            return args, ""

        def resolve(val: Any) -> tuple[Any, str]:
            if isinstance(val, str) and val.strip() in self.catalog.cards:
                return val.strip(), ""
            if not isinstance(val, str) or looks_like_id(val, True) or len(val.strip()) < 8:
                return val, ""
            start = val.strip().lower()
            hits = [i for i, row in self.catalog.cards.items()
                    if any(k != self.catalog.key and isinstance(v, str) and v.strip().lower().startswith(start) for k, v in row.items())]
            if len(hits) == 1:
                return hits[0], ""
            if hits:
                return val, f"{val!r} matches more than one listing ({', '.join(hits[:6])}); give a single id"
            return val, f"{val!r} is a title, not an id; use the {self.catalog.key} value shown in result rows"

        vals = args[key] if isinstance(args[key], list) else [args[key]]
        resolved = [resolve(v) for v in vals]
        trouble = next((p for _, p in resolved if p), "")
        if trouble:
            return args, trouble
        fixed = [v for v, _ in resolved]
        return {**args, key: fixed if isinstance(args[key], list) else fixed[0]}, ""

    def join_opens(self, call_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge several single opens in one turn into one multi-listing call."""
        opener, joiner = self.role("open"), self.role("compare")
        id_param, many = self.lone_param(opener), self.batch_param()
        if not opener or not joiner or opener == joiner or not id_param or not many or self.batch_blind():
            return call_list
        chosen: list[tuple[int, str]] = []
        for index, call in enumerate(call_list):
            fn = call.get("function") or {}
            args = read_object(fn.get("arguments")) if isinstance(fn.get("arguments"), str) else None
            lid = str((args or {}).get(id_param) or "").strip()
            if fn.get("name") == opener and args and set(args) == {id_param} and lid in self.catalog.cards and lid not in self.catalog.retired:
                chosen.append((index, lid))
        chosen = chosen[: self.batch_size()]
        if len(chosen) < 2:
            return call_list
        joined = {i for i, _ in chosen}
        out = []
        for index, call in enumerate(call_list):
            if index == chosen[0][0]:
                out.append({"id": call["id"], "type": "function", "function": {"name": joiner, "arguments": flat({many: [i for _, i in chosen]})}})
            elif index in joined:
                out.append({"id": call["id"], "joined": True})
            else:
                out.append(call)
        self.log.append({"kind": "joined", "ids": [i for _, i in chosen]})
        return out

    # ---- before an order
    def reread_offers(self) -> str:
        """Refresh the live offer of each cart line just before checkout."""
        tool = self.role("offer") or self.role("open")
        id_param = self.lone_param(tool)
        if not tool or not id_param:
            return ""
        since = max(self.store.step - 2, self.refusal_step)
        stale = [i for i in self.basket.ids() if self.offer_read_at.get(i, -99) < since and i not in self.catalog.retired]
        if not stale or self.steps_left() < len(stale) + 3:
            return ""
        changed = []
        for lid in stale:
            before = dict(self.catalog.cards.get(lid) or {})
            resp = self.store_call(tool, {id_param: lid})
            if resp is None or self.store.finished:
                break
            self.offer_read_at[lid] = self.store.step
            after = self.catalog.cards.get(lid) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                changed.append(f"{lid} changed ({', '.join(moved[:4])}): {flat(shorten(resp.get('observation'), 300))[:900]}")
        self.log.append({"kind": "reread", "ids": stale, "changed": len(changed)})
        if changed:
            self.epoch += 1
        return "\n".join(changed)

    def look_at_cart(self) -> str:
        """Look at the cart if the last view predates a change; report any change it shows."""
        view = self.role("cart_view")
        if not (self.basket.touched and not self.basket.fresh_view() and view and self.steps_left() > 3):
            return ""
        resp = self.store_call(view, {}, fold=False)
        if resp is None:
            return ""
        self.basket.saw_view(resp.get("observation"), self.store.step)
        change = self.fold(view, self.basket.view)
        if change:
            self.pending_notes = []
        return change

    def wider_hold(self) -> str:
        """One extra sweep before a lowest-price checkout when steps allow."""
        if self.wide_held or self.sheet.pick != "cheapest" or self.steps_left() < (8 if self.cut_short else 14 + 2 * len(self.sheet.wants)):
            return ""
        if self.role("search") not in {t["function"]["name"] for t in self.offered()}:
            return ""
        self.wide_held = True
        text_lines = [f"{i} at {self.catalog.price(i):g}" for i in self.basket.ids() if self.catalog.price(i) is not None]
        current = f" (now: {'; '.join(text_lines)})" if text_lines else ""
        return (f"Lowest price wins and {self.steps_left()} steps remain. Keyword search mostly matches titles, while condition values "
                "often live only in attributes, so many qualifying listings never show up, and results are not price-sorted. Try a few "
                "more queries per item using other product names (a bare one-word name, synonyms, the initials of a multi-word name or "
                f"value, words from titles of qualifying listings) and look only at available rows cheaper than the cart line{current}."
                + (f" The shop's category listing stopped at row {self.cut_short}, so most of it is unseen." if self.cut_short else "") + " Read attributes of the cheapest of those, "
                "swap in any that qualifies, then check out.")

    def cheaper_reads(self) -> list[str]:
        """Read a few cheaper unread rows of each cart line's kind before a lowest-price checkout."""
        catalog = self.catalog
        stamp = self.basket.mark() + f"#{self.epoch}"
        if self.sheet.pick != "cheapest" or stamp in self.cheaper_done or not catalog.shelf_field():
            return []
        self.cheaper_done.add(stamp)
        carried = self.basket.ids()
        chosen: list[str] = []
        for line in carried:
            price, kind = catalog.price(line), catalog.kind(line).lower()
            if price is None or not kind:
                continue
            pool = [i for i in catalog.order if i not in carried and i not in catalog.read and catalog.can_buy(i)
                    and catalog.kind(i).lower() == kind and (catalog.price(i) if catalog.price(i) is not None else float("inf")) < price]
            chosen += [i for i in catalog.cheap_order(pool)[:2] if i not in chosen]
        return self.fill_facts(chosen[:CHEAPER_READS], self.reserve_steps() + 4)

    def hold_order(self) -> str:
        """Checkout safeguards; empty string means proceed."""
        if (self.asker.done and self.messages_free() > 0 and self.steps_left() > self.reserve_steps() + 6
                and self.desk.seconds_left() > WRAP_UP + ASK_WINDOW and not self.desk.closed()):
            # a message the interview left unused is a question not asked: one more before the checkout (its answer can
            # still change the pick)
            extra = self.asker.run(spare_only=True)
            if extra:
                return self.HELD + extra + "\nRe-check the cart against any new condition, then check out."
        wider = self.wider_hold()
        if wider:
            return self.HELD + wider
        reread = self.reread_offers()
        change = self.look_at_cart()
        if change:
            return self.HELD + "The cart view revealed a change: " + change + "\nTake it into account, repair the cart if necessary, and check out."
        swap = self.reviews.sold_hint()
        stuck = [i for i in self.basket.ids() if i in self.catalog.retired] if self.sold_holds < 2 else []
        if swap or stuck:
            self.sold_holds += 0 if swap else 1
            return self.HELD + (swap or f"Cart line {', '.join(stuck)} is sold out; swap it before checking out.")
        chain, hold = self.reviews.chains()
        if hold:
            return self.HELD + chain
        faults = self.basket.faults(self.sheet) if self.line_hold_count < 2 and self.steps_left() > self.reserve_steps() + 2 else []
        if faults:
            self.line_hold_count += 1
            self.log.append({"kind": "line_hold", "faults": faults[:6]})
            return self.HELD + "Cart problems found: " + "; ".join(faults[:6]) + ". Repair them with the cart tool and check out."
        if self.proof_holds < 1 and self.steps_left() > self.reserve_steps() + 3:
            unproven = self.unproven_lines()
            if unproven:
                self.proof_holds += 1
                self.log.append({"kind": "proof_hold", "faults": unproven[:4]})
                return (self.HELD + "Not proven by the page: " + "; ".join(unproven[:4]) + ". The shop judges such a condition on the "
                        "attribute, so swap in a listing whose attribute states the value, or check out if none exists.")
        if self.set_holds < 1 and self.steps_left() > self.reserve_steps() + 5:
            cheaper = self.cheaper_set()
            if cheaper:
                self.set_holds += 1
                self.log.append({"kind": "set_hold", "said": cheaper[:300]})
                return self.HELD + cheaper
        if self.compat_holds < 2 and self.steps_left() > self.reserve_steps() + 3:
            # the lines must fit one another by their own pages' model codes (race 13: code-made and model-made carts mixed
            # families, and a "part ... also fits" line was taken for the accessory's own list)
            mismatched = self.compat_faults() or self.pair_faults()
            if mismatched:
                self.compat_holds += 1
                self.log.append({"kind": "compat_hold", "faults": mismatched[:4]})
                return self.HELD + "Model-code check: " + "; ".join(mismatched[:4]) + ". Repair the cart so every line fits one device, then check out."
        self.part_lookups()
        read = self.fill_facts(self.basket.ids(), self.reserve_steps() + 3) + self.cheaper_reads()
        verdict = self.reviews.cart()
        if verdict:
            shown_obs = "Attributes Tern read for the review (one step each):\n" + "\n".join(read) + "\n" if read else ""
            return "Not run yet. " + chain + shown_obs + verdict
        if reread and self.desk.spent > 0.75 * self.desk.purse:
            return (self.HELD + "Fresh reads of the cart lines:\n" + reread + "\nIf every line is still in stock and affordable, check out "
                    "again; otherwise replace the offending line first.")
        return ""

    # ---- before an ending without a purchase
    def finish_confirmed(self) -> str:
        """When the loop declines despite a vouched listing, build that cart directly and check out if it passes review."""
        reviews, catalog = self.reviews, self.catalog
        cart_tool, buy = self.role("cart"), self.role("buy")
        if self.filled_in or not reviews.vouched or catalog.is_gone(reviews.vouched) or not cart_tool or not buy:
            return ""
        self.filled_in = True
        planned: dict[str, int] = {}
        for number, ids in reviews.hopeful_items.items():
            index = whole(number)
            pick = next((i for i in ids if not catalog.is_gone(i)), None)
            if pick and index and 1 <= index <= len(self.sheet.wants) and catalog.kind(pick) not in {catalog.kind(p) for p in planned}:
                planned[pick] = self.sheet.wants[index - 1].units
        if reviews.vouched not in planned:
            return ""
        now = self.basket.ids()
        kind_list = {catalog.kind(p) for p in planned}
        keep = {i: self.basket.text_lines.get(i, 1) for i in now if i not in planned and not catalog.is_gone(i) and catalog.kind(i) not in kind_list}
        drop = [i for i in now if i not in planned and i not in keep]
        if len(planned) + len({catalog.kind(i) for i in keep}) < len(self.sheet.wants):
            return ""
        text_lines = {**keep, **planned}
        call_list = self.basket.calls_for(text_lines, drop)
        if not call_list or len(call_list) + 1 > self.steps_left():
            return ""
        self.fill_facts(list(text_lines), len(call_list) + 3)
        reviews.cart(list(text_lines), self.desk.purse)
        if not reviews.passed.get(reviews.key(list(text_lines))):
            return ""
        made = []
        for args in call_list:
            resp = self.store_call(cart_tool, args, fold=False)
            if resp is None or self.store.finished:
                return ""
            shaped, _ = fill_args(args, self.schemas.get(cart_tool))
            self.basket.record_call(cart_tool, shaped, resp.get("observation"), self.store.step)
            self.fold(cart_tool, resp.get("observation"))
            made.append(flat(args))
        replies = []
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                reviews.cart(list(text_lines), self.desk.purse)
                if not reviews.passed.get(reviews.key(list(text_lines))) or self.steps_left() < 1 or self.store.finished or any(catalog.is_gone(i) for i in text_lines):
                    break
            go, said = self.gate.run("order")
            if said:
                replies.append(said)
            if not go:
                break
            resp = self.store_call(buy, dict(self.sheet.buy_args), fold=False)
            self.log.append({"kind": "filled_in", "lines": text_lines, "drop": drop, "reply": shorten(resp, 200)})
            if resp is None:
                break
            self.orders_tried += 1
            if not resp.get("finished"):
                self.refusal_step = self.store.step
            self.fold(buy, resp.get("observation"))
            replies.append(self.show(resp.get("observation")))
            if self.store.finished or resp.get("finished"):
                break
        if not replies:
            return ""
        return ("Not executed; Tern stepped in because a separate review vouched for " + reviews.vouched + ". Cart calls: "
                + "; ".join(made) + ". Checkout reply: " + " | then: ".join(replies))

    def hold_decline(self) -> str:
        """No-order safeguards; empty string means proceed."""
        reviews = self.reviews
        enough_time = self.desk.seconds_left() > WRAP_UP + 15
        leads = self.open_leads()
        can_open = (self.attr_view() or self.role("open")) in {t["function"]["name"] for t in self.offered()}
        if leads and can_open and self.lead_holds == 0 and self.steps_left() >= 3 and self.desk.seconds_left() > WRAP_UP:
            # race 14: declines came with the passing listing shown by a value search and never opened
            self.lead_holds = 1
            self.log.append({"kind": "lead_hold", "leads": leads[:3]})
            return (self.HELD + "Searches returned rows never opened: " + ", ".join(leads[:3]) + " (a search for a required value "
                    "matched their attributes, or several searches showed them). Open them, and order the one that meets every condition. "
                    "Finish without an order only if none does.")
        if self.sheet.open_pointers() and not self.decline_chained and self.steps_left() > 6 and enough_time:
            # a condition that takes its value from another listing is resolved before nothing is said to qualify (race
            # 13: declines with a chain's first page unopened, where the chain's value made a read listing qualify)
            self.decline_chained = True
            chain, _ = reviews.chains()
            if chain:
                return (self.HELD + chain + "\nWith these linked values, check the listings already read against every condition and "
                        "order the one that qualifies; finish without an order only if none does.")
        if self.steps_left() > 3 and enough_time:
            review = reviews.decline()
            if review:
                return self.HELD + review
            if reviews.hopefuls and self.steps_left() > 4 and self.decline_holds == 0:
                self.decline_holds = 1
                self.log.append({"kind": "decline_held", "leads": list(reviews.hopefuls)})
                return (self.HELD + "Review candidates that look qualifying: " + ", ".join(reviews.hopefuls) + ". Open what you must, cart the "
                        "best qualifying one (mind the pick rule) and check out. If every one plainly breaks a condition, say which in "
                        "your text and repeat the no-order call.")
            if reviews.hopefuls and self.steps_left() > 4 and self.decline_holds == 1:
                self.decline_holds = 2
                first = reviews.hopefuls[0]
                reviews.cart([first], self.desk.purse)
                if reviews.passed.get(reviews.key([first])) and first in reviews.hopefuls:
                    reviews.vouched = first
                    self.log.append({"kind": "decline_held", "confirmed": first})
                    return (self.HELD + f"A separate review of {first} finds every condition met. Cart it with the requested units and "
                            "check out, unless its record plainly breaks a condition.")
        if (self.steps_left() <= 5 and not self.late_hold_used and self.steps_left() >= len(self.sheet.wants) + 2
                and len([i for i in self.basket.best_build if not self.catalog.is_gone(i)]) < len(self.sheet.wants)):
            # race 14: late declines with a partial or empty cart while read listings proved every simple condition
            plan = self.proven_cart()
            if plan and set(plan) != set(self.basket.ids()) and self.put_in_cart(plan):
                self.late_hold_used = True
                return (self.HELD + "Tern put listings whose pages state every condition in the cart (" + ", ".join(plan) + "). Check "
                        "out now; finish without an order only if one of them plainly breaks a condition, then resend the call.")
        if self.steps_left() <= 5 and not self.late_hold_used and enough_time:
            # late in the episode: a decline here usually discards a working cart
            built = [i for i in self.basket.best_build if not self.catalog.is_gone(i)]
            missing = [i for i in built if i not in self.basket.ids() or (i in self.basket.taken_off and not self.basket.fresh_view())]
            lost = [i for i in self.basket.best_build if self.catalog.is_gone(i)]
            if len(built) >= len(self.sheet.wants) and len(missing) + 1 <= self.steps_left():
                self.late_hold_used = True
                self.log.append({"kind": "late_hold", "lines": built, "missing": missing})
                redo = f"Return {', '.join(missing)} to the cart with the requested units and check out." if missing else "Check out the current cart."
                return (self.HELD + "A no-order finish scores zero whenever a qualifying listing exists, and low steps are no evidence that "
                        "none does. " + redo + " Decline only if searching showed some item has no qualifying listing, then resend the call.")
            if lost and len(built) + len(lost) >= len(self.sheet.wants) and len(missing) + len(lost) + 1 <= self.steps_left():
                chosen = reviews.stand_ins(lost)
                if chosen:
                    self.late_hold_used = True
                    self.log.append({"kind": "late_swap", "lost": lost, "picks": chosen})
                    back = f" Also return {', '.join(missing)} with the requested units." if missing else ""
                    return (self.HELD + f"{', '.join(lost)} sold out, but read listings look like valid stand-ins: {', '.join(chosen)}. "
                            f"Cart {chosen[0]} with the requested units (one per dead line, same kind).{back} Then check out. Decline "
                            "only if each stand-in plainly breaks a condition, then resend the call.")
        if not reviews.hopefuls and self.decline_pushes == 0 and self.steps_left() > self.step_limit // 3 and enough_time:
            self.decline_pushes = 1
            return (self.HELD + f"With {self.steps_left()} steps remaining, widen the search first: browse each item's category and page "
                    "on with the cursor the shop returns, try one-word value queries and alternative names for the property, and open "
                    "plausible listings. Decline only after that.")
        return ""


    def code_roles(self) -> tuple[int, set[int]]:
        """(the device item, the accessory items) when the request's items split that way by their listings' model codes:
        most seen listings of one item state an own code, most of the others' state what they fit; (0, set()) otherwise."""
        catalog, wants = self.catalog, self.sheet.wants
        if len(wants) < 2:
            return 0, set()
        devices, parts = [], set()
        for number in range(1, len(wants) + 1):
            ids = [i for i in catalog.order if catalog.owner(catalog.kind(i)) == number][:40]
            if len(ids) < 2:
                continue
            fitting = sum(1 for i in ids if catalog.code_marks(i)[1] or catalog.code_marks(i)[2])
            owning = sum(1 for i in ids if catalog.code_marks(i)[0])
            if 2 * fitting >= len(ids):
                parts.add(number)
            elif 2 * owning >= len(ids):
                devices.append(number)
        return (devices[0], parts) if len(devices) == 1 and parts else (0, set())

    def compat_faults(self) -> list[str]:
        """Accessory lines whose own page does not say they fit the carted device's model code, with the read listings that
        would: a part of the same kind that fits that code, or a device code every carted accessory fits."""
        catalog = self.catalog
        device, part_items = self.code_roles()
        if not device:
            return []
        carried = [i for i in self.basket.ids() if not catalog.is_gone(i)]
        lines = [i for i in carried if catalog.owner(catalog.kind(i)) == device and catalog.code_marks(i)[0]]
        parts = [i for i in carried if catalog.owner(catalog.kind(i)) in part_items]
        if len(lines) != 1 or not parts:
            return []
        codes = sorted(catalog.code_marks(lines[0])[0])
        faults = []
        for lid in parts:
            if any(catalog.fits_code(lid, code) for code in codes):
                continue
            kind = catalog.kind(lid)
            same = [i for i in reversed(catalog.order) if i != lid and catalog.kind(i) == kind and catalog.can_buy(i)
                    and any(catalog.fits_code(i, code) for code in codes)][:3]
            refer = sorted(catalog.code_marks(lid)[3])
            text = f"{lid} does not say it fits model {', '.join(codes)} of the carted device {lines[0]}"
            if refer:
                text += f" (its page sends the reader to part {', '.join(refer)} for the rest of its list: look that part up)"
            if same:
                text += f"; read listings of its kind that do: {', '.join(same)}"
            faults.append(text)
        if faults:
            spread = sorted({c for i in catalog.order if catalog.owner(catalog.kind(i)) == device for c in catalog.code_marks(i)[0]
                             if all(catalog.fits_code(p, c) for p in parts)})
            if spread:
                faults.append("device codes every carted accessory fits: " + ", ".join(spread[:4]))
        return faults

    def pair_faults(self) -> list[str]:
        """The request's pairs (an item that must work with another), checked by the carted pages' code sentences: two lines
        go together when one names a code the other names or carries. Pages that speak in no codes are left to the review."""
        catalog = self.catalog
        held: dict[int, list[str]] = {}
        for lid in self.basket.ids():
            if not catalog.is_gone(lid):
                held.setdefault(catalog.owner(catalog.kind(lid)), []).append(lid)
        faults = []
        for part, device in self.sheet.pairs:
            if not held.get(part) or not held.get(device):
                continue
            a, b = held[part][0], held[device][0]
            if not (catalog.code_marks(a)[4] or catalog.code_marks(b)[4]) or catalog.paired(a, b):
                continue
            kind = catalog.kind(a)
            instead = [i for i in reversed(catalog.order) if i != a and catalog.kind(i) == kind and catalog.can_buy(i) and catalog.paired(i, b)][:3]
            text = f"{a} and {b} name no code in common, so their pages do not show them working together"
            if instead:
                text += f"; listings of {a}'s kind that pair with {b}: {', '.join(instead)}"
            faults.append(text)
        return faults

    def part_lookups(self) -> None:
        """Before a checkout: a cart line whose page sends the reader to another part number ("More on part XY123") has that
        number searched (race 14: every multi-item pass with such a line had looked the part up; 72 of 76 runs without did
        not pass). Four look-ups at most, never into the steps the checkout and its re-reads need."""
        field = self.catalog.name_field()
        for lid in self.basket.ids():
            own = set(re.findall(self.catalog.CODE, str((self.catalog.cards.get(lid) or {}).get(field) or "") if field else ""))
            for code in sorted(self.catalog.code_marks(lid)[3] - {c.replace("-", "") for c in own}):
                if code in self.parts_searched or len(self.parts_searched) >= 4 or self.store.finished:
                    continue
                if self.steps_left() < self.reserve_steps() + 4 + len(self.basket.ids()):
                    return
                self.parts_searched.add(code)
                found = self.search_for(code)
                self.log.append({"kind": "part_lookup", "line": lid, "code": code, "rows": len(found)})

    def proven(self, lid: str, number: int, kin: list[str] | None = None) -> tuple[bool, list[str]]:
        """Whether a read listing's own page states each simple word condition of its item, and the conditions it leaves
        unproven while two or more other read listings of the kind record that property as an attribute (the shop judges
        such a condition on the attribute: race 14 lost orders of listings that named the value only in their title)."""
        catalog, sheet = self.catalog, self.sheet
        record = {**(catalog.cards.get(lid) or {}), **(catalog.read.get(lid) or {})}
        block = attribute_block(record, 1) or {}
        others = [attribute_block(catalog.read.get(i), 1) or {} for i in (catalog.ids_of(number) if kin is None else kin) if i != lid and i in catalog.read]
        stated, missing, checked = True, [], 0
        for need in sheet.wants[number - 1].live():
            prop, _, value = need.partition(":")
            value = value.strip().lower()
            words = re.findall(r"[a-z]{3,}", value)
            if not words or re.search(r"\d|\b(?:not|no|at|least|most|under|over|than|like|same|unlike)\b", value) or len(words) > 3:
                continue
            naming = lambda k: same_property(roots(prop), roots(k)) or same_property(roots(k), roots(prop))
            named = [str(v).lower() for k, v in block.items() if naming(k)]
            checked += 1
            if not named or not set(words) <= set(re.findall(r"[a-z]{3,}", " ".join(named))):
                stated = False
                if not named and sum(1 for b in others if any(naming(k) for k in b)) >= 2:
                    missing.append(need)
        return stated and checked > 0, missing

    def put_in_cart(self, plan: dict[str, int]) -> bool:
        """Make the cart hold exactly `plan` (lines and units) with the cart tool, keeping a step for the checkout."""
        cart_tool = self.role("cart")
        drop = [i for i in self.basket.ids() if i not in plan]
        call_list = self.basket.calls_for(plan, drop) if plan and cart_tool else None
        if not call_list or len(call_list) + 1 > self.steps_left():
            return False
        for args in call_list:
            resp = self.store_call(cart_tool, args, fold=False)
            if resp is None or self.store.finished:
                return False
            shaped, _ = fill_args(args, self.schemas.get(cart_tool))
            self.basket.record_call(cart_tool, shaped, resp.get("observation"), self.store.step)
            self.fold(cart_tool, resp.get("observation"))
        self.log.append({"kind": "proven_cart", "lines": plan})
        return True

    def unproven_lines(self) -> list[str]:
        """Cart lines whose page has no attribute for a condition that other read listings of the kind record, with the read
        listings whose attribute states the value."""
        catalog = self.catalog
        faults = []
        for lid in self.basket.ids():
            number = catalog.owner(catalog.kind(lid)) or (1 if len(self.sheet.wants) == 1 else 0)
            if not number or lid not in catalog.read:
                continue
            kin = catalog.ids_of(number)
            _, missing = self.proven(lid, number, kin)
            if missing:
                better = [i for i in reversed(kin) if i != lid and i in catalog.read and catalog.can_buy(i) and self.proven(i, number, kin)[0]][:3]
                faults.append(f"{lid} has no attribute for {'; '.join(missing)}, which other listings of its kind record"
                              + (f" (read listings that state it: {', '.join(better)})" if better else ""))
        return faults

    def proven_cart(self) -> dict[str, int]:
        """Per requested item, the read, buyable listing whose page states every simple condition (the cheapest under a
        lowest-price pick, else the newest read); {} unless every item has one and the total fits the budget."""
        catalog, sheet = self.catalog, self.sheet
        plan: dict[str, int] = {}
        for number, want in enumerate(sheet.wants, 1):
            kin = catalog.ids_of(number)
            pool = [i for i in reversed(kin) if i in catalog.read and catalog.can_buy(i)
                    and catalog.label_score(catalog.kind(i), want) >= 2 and self.proven(i, number, kin)[0]]
            if sheet.pick == "cheapest":
                pool.sort(key=lambda i: catalog.price(i) if catalog.price(i) is not None else float("inf"))
            if not pool:
                return {}
            plan[pool[0]] = want.units
        prices = [catalog.price(i) for i in plan]
        if sheet.budget is not None and None not in prices and sum(p * plan[i] for i, p in zip(plan, prices)) > sheet.budget + 1e-9:
            return {}
        return plan

    def cheaper_set(self) -> str:
        """Under a lowest-price pick, when the lines must fit one device by model code: the cheapest complete set among the
        listings seen (per device code, the cheapest device carrying it and the cheapest listing of each other item that
        fits it), when it costs less than the cart (race 13: Tern fixed the device by its own price first and ordered a
        dearer whole basket while a cheaper compatible family was in its rows)."""
        catalog, sheet = self.catalog, self.sheet
        if sheet.pick != "cheapest" or len(sheet.wants) < 2:
            return ""
        carried = [i for i in self.basket.ids() if not catalog.is_gone(i)]
        prices = [catalog.price(i) for i in carried]
        if not carried or None in prices:
            return ""
        now = sum(p * self.basket.units(i) for i, p in zip(carried, prices))
        owner = {i: catalog.owner(catalog.kind(i)) for i in catalog.order}
        device, parts = self.code_roles()
        if not device:
            return ""

        def stated(lid: str, number: int) -> bool:
            # each plain word value the item requires is stated by the attribute named for its property on the listing's
            # opened page (limits and links aside): only proven lines make a suggested set
            record = {**(catalog.cards.get(lid) or {}), **(catalog.read.get(lid) or {})}
            block = attribute_block(record) or {}
            for need in sheet.wants[number - 1].live():
                prop, _, value = need.partition(":")
                value = value.strip().lower()
                words = re.findall(r"[a-z]{3,}", value)
                if not words or re.search(r"\d|\b(?:not|no|at|least|most|under|over|than|like|same|unlike)\b", value) or len(words) > 3:
                    continue
                named = [str(v).lower() for k, v in block.items() if same_property(roots(prop), roots(k)) or same_property(roots(k), roots(prop))]
                if not named or not all(w in " ".join(named) for w in words):
                    return False
            return True
        best: tuple[float, dict[int, str]] | None = None
        for code in sorted({c for i, n in owner.items() if n == device for c in catalog.code_marks(i)[0]}):
            plan: dict[int, str] = {}
            for number in range(1, len(sheet.wants) + 1):
                pool = [i for i, n in owner.items() if n == number and i in catalog.read and catalog.can_buy(i) and catalog.price(i) is not None and stated(i, number)
                        and (code in catalog.code_marks(i)[0] if number == device else catalog.fits_code(i, code) if number in parts else True)]
                if not pool:
                    break
                plan[number] = min(pool, key=lambda i: (catalog.price(i), catalog.order.index(i)))
            if len(plan) == len(sheet.wants):
                total = sum(catalog.price(i) * sheet.wants[n - 1].units for n, i in plan.items())
                if best is None or total < best[0]:
                    best = (total, plan)
        if not best or best[0] >= now - 1e-9 or set(best[1].values()) == set(carried):
            return ""
        lines = ", ".join(f"item {n} {i} ({catalog.price(i):g})" for n, i in sorted(best[1].items()))
        return (f"A cheaper complete set whose lines all fit one model code and whose opened pages state the required values: {lines}, "
                f"total {best[0]:g} against the cart's {now:g}. Check each line against every condition, and if each meets them, cart that set instead.")

    # ---- one turn's calls
    def screen(self, name: str, raw: Any, calls_in_turn: int) -> tuple[dict[str, Any] | None, str, str]:
        """Vet a call before it costs a step: (args, note, mark) to run it, or (None, reason, "")."""
        allowed = {t["function"]["name"] for t in self.offered()}
        if self.store.finished:
            return None, "Skipped: the episode has ended.", ""
        if name == self.role("ask"):
            return None, self.NOT_RUN + "only Tern's interviewer writes to the customer.", ""
        if name not in self.schemas:
            return None, self.NOT_RUN + f"no such tool; available: {', '.join(sorted(allowed))}.", ""
        if name not in allowed:
            return None, self.NOT_RUN + "not enough steps left for that; allowed now: " + ", ".join(sorted(allowed)) + ".", ""
        args = raw if isinstance(raw, dict) else (read_object(raw) if str(raw or "").strip() else {})
        if args is None:
            return None, self.NOT_RUN + "arguments must form a single JSON object.", ""
        args, problems = fill_args(args, self.schemas[name])
        if problems:
            return None, self.NOT_RUN + "; ".join(problems[:6]), ""
        args, wrong = self.by_title(name, args)
        if wrong:
            return None, self.NOT_RUN + wrong + ".", ""
        extra = ""
        many = self.batch_param() if name == self.role("compare") else None
        if many and isinstance(args.get(many), list) and self.compare_fail and len(args[many]) > self.batch_size():
            room = self.batch_size()
            rest = ", ".join(map(str, args[many][room:]))
            extra += f"\n[Tern] This shop handles {room} ids per call, so only the first {room} were sent; still unseen: {rest}."
            args = {**args, many: args[many][:room]}
        id_param = self.lone_param(name)
        if id_param and name in (self.role("open"), self.role("offer")) and str(args.get(id_param)) in self.catalog.retired:
            gone_at = self.catalog.retired[str(args[id_param])]
            return None, self.NOT_RUN + f"{args[id_param]} sold out at step {gone_at} for good; use another qualifying listing for its line.", ""
        if name == self.role("cart") and any((whole(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
            view = flat(self.basket.view) if self.basket.fresh_view() else ""
            absent = [v for v in args.values() if isinstance(v, str) and v in self.catalog.cards and v not in self.basket.text_lines
                      and v not in view and v not in self.basket.reply_text]
            if absent:
                return None, self.NOT_RUN + f"{absent[0]} is not a cart line, so nothing can be removed. Cart: " + self.basket.text(), ""
        ending = self.is_order(name, args) or self.is_decline(name, args)
        if ending and calls_in_turn > 1:
            return None, self.NOT_RUN + "a finishing call must be the only call in its turn; read your other results first.", ""
        if not ending and self.steps_left() <= self.reserve_steps() and (self.role("buy") or self.role("refuse")):
            return None, self.NOT_RUN + "the final steps are reserved for checkout (plus any required approval) or a no-order finish.", ""
        mark = name + flat(args)
        if not ending and name != self.role("cart") and mark in self.last_calls:
            return None, self.NOT_RUN + "identical to a call you just made, with nothing changed since; see its result above.", ""
        readers = {self.role(r) for r in ("search", "open", "offer", "compare")} - {None}
        cached = self.cache.get(mark)
        if name in readers and cached and cached[0] == self.epoch:
            return None, self.NOT_RUN + f"step {cached[1]} made this exact call and nothing has changed; its result:\n" + cached[2][:3000], ""
        if id_param and name in readers and name != self.role("compare") and self.catalog.nothing_new(name, str(args.get(id_param)), self.epoch):
            return None, self.NOT_RUN + f"this view has nothing new on {args[id_param]}; stored record: " + self.catalog.record(str(args[id_param])), ""
        if many and isinstance(args.get(many), list) and args[many]:
            fresh = [i for i in args[many] if not self.catalog.nothing_new(name, str(i), self.epoch)]
            if not fresh:
                stored = "\n".join(f"{i}: {self.catalog.record(str(i))}" for i in args[many])[:4000]
                return None, self.NOT_RUN + "this view has nothing new on these listings; stored records:\n" + stored, ""
            if len(fresh) < len(args[many]):
                extra += f"\n[Tern] Dropped already-complete ids: {', '.join(str(i) for i in args[many] if i not in fresh)}."
                args = {**args, many: fresh}
        if ending and self.desk.seconds_left() > WRAP_UP - 25:
            carried = ""
            if self.is_order(name, args) and self.steps_left() > 2:
                carried = self.hold_order()
            elif self.is_decline(name, args):
                carried = self.hold_decline()
            if carried:
                return None, carried + "\nCart: " + self.basket.text(), ""
        if self.is_decline(name, args):
            done = self.finish_confirmed()
            if done:
                return None, done + (" [episode finished]" if self.store.finished else ""), ""
            args = self.quiet_args(name, args)
        if ending and self.steps_left() >= 2 and self.desk.seconds_left() > NO_NEW_CALL + 12:
            go, said = self.gate.run("order" if self.is_order(name, args) else "end")
            if not go:
                return None, "On hold. " + said + "\nCart: " + self.basket.text(), ""
            if said:
                extra = "\n[Tern] " + said + extra
        return args, extra, mark

    def describe(self, name: str, args: dict[str, Any], resp: dict[str, Any], extra: str, mark: str) -> str:
        """Record a shop reply and render it for the loop."""
        shown_obs = resp.get("observation")
        many = self.batch_param() if name == self.role("compare") else None
        if many and isinstance(args.get(many), list) and not self.learn_compare(args[many], shown_obs):
            self.compare_missed = True
            extra += f"\n[Tern] Empty result for {len(args[many])} ids at once; send no more than {self.batch_size()} per call."
        self.last_calls = (self.last_calls + [mark])[-3:]
        for key, val in args.items():
            vals = val if isinstance(val, list) else [val]
            if vals and all(isinstance(x, str) and x in self.catalog.cards for x in vals):
                self.basket.id_params.add((name, key))
        self.catalog.take(shown_obs)
        if name == self.role("compare"):
            self.note_attrs(name, shown_obs)
        readers = {self.role(r) for r in ("search", "open", "offer", "compare")} - {None}
        if name in readers - {self.role("search")}:
            self.catalog.note_view(name, shown_obs, self.epoch)
        if name == self.role("search"):
            self.searches.append(dict(args))
            self.note_leads(args, shown_obs)
        ending = self.is_order(name, args) or self.is_decline(name, args)
        not_search = {tool for job, tool in self.sheet.tool_for.items() if job != "search"}
        paged = self.more_pages(name, args, shown_obs) if name not in not_search and not ending else ""
        enders = {self.role(j) for j in ("cart", "buy", "refuse", "ask")}
        if self.lone_param(name) and name not in enders:
            listing = self.catalog.lid(shown_obs) if isinstance(shown_obs, dict) else None
            if listing:
                self.attrs_from[name] = self.attrs_from.get(name, False) or bool(attribute_block(shown_obs))
                if name == self.role("open") and listing not in self.catalog.read and attribute_block(shown_obs, 1):
                    # a page whose attributes hold a single value is read all the same: opening it again shows nothing new
                    self.catalog.read[listing] = dict(self.catalog.cards.get(listing) or {})
                if name == self.role("open") and not self.attrs_from[name]:
                    other = next((t for t in self.schemas if t != name and self.lone_param(t) and self.attrs_from.get(t) is not False
                                  and t not in enders | {self.role("search")}), None)
                    if other:
                        self.sheet.tool_for["open"] = other
                if not self.attrs_from[name] and name != self.role("open") and not self.role("offer"):
                    self.sheet.tool_for["offer"] = name
                if name == (self.role("offer") or self.role("open")):
                    self.offer_read_at[listing] = self.store.step
        if name == self.role("cart"):
            self.basket.record_call(name, args, shown_obs, self.store.step)
            self.last_calls = []
        if name == self.role("cart_view"):
            self.basket.saw_view(shown_obs, self.store.step)
        if self.is_order(name, args):
            self.orders_tried += 1
            if not resp.get("finished"):
                self.refusal_step = self.store.step
                self.reopen_checks()
        elif self.is_decline(name, args) and not resp.get("finished"):
            self.declines_refused += 1
            self.reopen_checks()
        text = self.show(shown_obs) + paged + extra
        if name not in not_search and not ending:
            text += self.cheapest_rows([shown_obs] + self.paged)
        plain_text = text
        proof = self.catalog.key_hints(shown_obs)
        if proof:
            text += "\n" + proof
        if name == self.role("cart") and any(whole(v) == 0 for v in args.values() if not isinstance(v, str)):
            if any(isinstance(v, str) and v in self.catalog.cards and holds_id(shown_obs, v) for v in args.values()):
                text += "\n[Tern] Quantity 0 did not remove the line: this tool adds the given count, so subtract with a negative count."
        self.reviews.note_said_sold = False
        text_lines = self.reviews.loose_text(shown_obs)
        if text_lines:
            meaning = self.reviews.note(text_lines)
            if meaning:
                text += "\n[shop note] " + meaning + ". Reflect this in your choices before checkout."
        if self.is_order(name, args) and not resp.get("finished"):
            said = self.spoken(shown_obs)
            if self.reviews.note_said_sold or any(line in said for line in self.reviews.sold_words):
                collected = self.find_sold()
                if collected:
                    text += "\n[cart lines re-read after the refused checkout]\n" + collected
        swap = self.reviews.sold_hint()
        if swap:
            text += "\n" + swap
        text += self.drain_notes()
        if name in readers:
            self.cache[mark] = (self.epoch, self.store.step, plain_text)
        if resp.get("finished"):
            text += " [episode finished]"
        return text

    def find_sold(self) -> str:
        """Work out which line a refusal meant by an unnamed sell-out."""
        ids = [i for i in self.basket.ids() if i not in self.catalog.retired]
        if not ids or any(self.catalog.is_gone(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.catalog.off_sale.add(ids[0])
            return ""
        return self.reread_offers()

    def handle(self, call_list: list[dict[str, Any]]) -> list[tuple[str, str]]:
        answers: list[tuple[str, str]] = []
        call_list = self.join_opens(call_list)
        real = sum(1 for c in call_list if not c.get("joined"))
        for call in call_list:
            cid = call["id"]
            if call.get("joined"):
                if self.compare_missed:
                    answers.append((cid, "Missing: the merged call above came back empty (this shop takes fewer ids at once). Open this listing separately."))
                else:
                    answers.append((cid, "See the merged result above: Tern fetched these listings in a single step."))
                continue
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            self.compare_missed = False
            args, extra, mark = self.screen(name, fn.get("arguments"), real)
            if args is None:
                answers.append((cid, extra))
                continue
            resp = self.store.call(name, args)
            if resp is None:
                answers.append((cid, "No response (network or server trouble); retrying is fine."))
                continue
            shown_obs = resp.get("observation")
            if name in self.page_views() and not (isinstance(shown_obs, dict) and shown_obs.get("error")):
                for val in args.values():
                    self.catalog.visited.update(v.strip() for v in (val if isinstance(val, list) else [val]) if isinstance(v, str))
            answers.append((cid, self.describe(name, args, resp, extra, mark)))
        return answers

    # ---- finishing
    def finish(self, report_refusal: bool = False) -> str:
        """Close the episode: checkout if the cart holds lines, otherwise a no-order finish. With `report_refusal` a refused
        checkout is returned as text for another loop round."""
        if self.store.finished or self.store.unpaid or self.steps_left() <= 0:
            return ""
        buy, refuse = self.role("buy"), self.role("refuse")
        self.revive_dead_line()
        if buy and self.basket.holding():
            self.take_better()
        if buy and self.basket.holding() and self.sheet.open_pointers() and not self.desk.closed() and self.steps_left() > 5:
            # a code-made checkout counts only when every page behind a linked condition was opened: open them now
            self.reviews.chains(self.desk.purse + 0.007, NO_NEW_CALL + 10)
        if buy and self.basket.holding():
            self.part_lookups()
        if buy and not self.basket.holding() and not self.declines_refused and self.steps_left() >= len(self.sheet.wants) + 2:
            # race 14: the loop stopped on its budget or clock with steps left and an empty cart, and the finish declined;
            # when every item has a read listing whose page states each simple condition, that cart is ordered instead
            plan = self.proven_cart()
            if plan:
                self.put_in_cart(plan)
        if buy and not self.basket.holding() and self.declines_refused and self.steps_left() >= 3:
            # a no-order finish the shop already turned down (a note changed the job) is not sent again unchanged: the
            # fullest cart built so far is put back and ordered
            self.rebuild_cart()
        if buy and (self.basket.holding() or (self.basket.touched and not self.basket.fresh_view() and self.steps_left() >= 2)):
            args, problems = fill_args(dict(self.sheet.buy_args), self.schemas.get(buy))
            if not problems:
                if self.steps_left() >= 2 and self.desk.seconds_left() > NO_NEW_CALL + 12:
                    self.gate.run("order")
                if self.store.finished or self.steps_left() <= 0:
                    return ""
                resp = self.store.call(buy, args)
                self.log.append({"kind": "forced_order", "reply": shorten(resp, 200)})
                if self.store.finished or self.steps_left() <= 0:
                    return ""
                if resp is not None and report_refusal:
                    self.orders_tried += 1
                    self.refusal_step = self.store.step
                    self.reviews.note_said_sold = False
                    self.fold(buy, resp.get("observation"))
                    words = self.spoken(resp.get("observation"))
                    if self.reviews.note_said_sold or any(line in words for line in self.reviews.sold_words):
                        self.find_sold()
                    return "the shop refused the checkout: " + self.show(resp.get("observation"))
        if refuse:
            args, problems = fill_args(self.quiet_args(refuse, dict(self.sheet.refuse_args)), self.schemas.get(refuse))
            if not problems:
                if self.steps_left() >= 2 and self.desk.seconds_left() > NO_NEW_CALL + 12:
                    self.gate.run("end")
                if self.store.finished or self.steps_left() <= 0:
                    return ""
                resp = self.store.call(refuse, args)
                self.log.append({"kind": "forced_decline", "reply": shorten(resp, 200)})
                if resp is not None and not self.store.finished:
                    self.declines_refused += 1
                    self.fold(refuse, resp.get("observation"))
                    return "the shop refused the no-order finish: " + self.show(resp.get("observation"))
        return ""

    def take_better(self) -> None:
        """At a code-made checkout, a better listing the last cart review named takes its kind's line (race 13 and local:
        the review found the cheaper fit, the spend stop came before the loop swapped it, and the dearer line was ordered)."""
        catalog, basket = self.catalog, self.basket
        cart_tool = self.role("cart")
        # only the cart the review judged, unchanged since (a line already swapped in is the loop's own fix)
        if self.reviews.better_mark != self.reviews.key(basket.ids()) or any(b in basket.ids() for b in self.reviews.better):
            return
        pick = next((b for b in self.reviews.better if catalog.can_buy(b)), None)
        if not pick or not cart_tool or self.steps_left() < 3:
            return
        same = [i for i in basket.ids() if catalog.kind(i) and catalog.kind(i) == catalog.kind(pick)]
        if len(same) != 1:
            return
        lines = {i: basket.text_lines.get(i, 1) for i in basket.ids() if i != same[0]}
        lines[pick] = basket.text_lines.get(same[0], 1)
        call_list = basket.calls_for(lines, same)
        if not call_list or len(call_list) + 1 > self.steps_left():
            return
        for args in call_list:
            resp = self.store_call(cart_tool, args, fold=False)
            if resp is None or self.store.finished:
                return
            shaped, _ = fill_args(args, self.schemas.get(cart_tool))
            basket.record_call(cart_tool, shaped, resp.get("observation"), self.store.step)
            self.fold(cart_tool, resp.get("observation"))
        self.reviews.better = []
        self.log.append({"kind": "took_better", "out": same[0], "in": pick})

    def rebuild_cart(self) -> None:
        """Put the fullest cart built so far back (lines still buyable, one per requested item) for a code-made checkout."""
        catalog, basket = self.catalog, self.basket
        cart_tool = self.role("cart")
        lines = [i for i in basket.best_build if not catalog.is_gone(i)]
        owners = {catalog.owner(catalog.kind(i)) for i in lines}
        if not cart_tool or not lines or (len(self.sheet.wants) > 1 and len(owners - {0}) < len(self.sheet.wants)):
            return
        units = {i: (self.sheet.wants[catalog.owner(catalog.kind(i)) - 1].units if catalog.owner(catalog.kind(i)) else 1) for i in lines}
        call_list = basket.calls_for(units, [])
        if not call_list or len(call_list) + 1 > self.steps_left():
            return
        for args in call_list:
            resp = self.store_call(cart_tool, args, fold=False)
            if resp is None or self.store.finished:
                return
            shaped, _ = fill_args(args, self.schemas.get(cart_tool))
            basket.record_call(cart_tool, shaped, resp.get("observation"), self.store.step)
            self.fold(cart_tool, resp.get("observation"))
        self.log.append({"kind": "rebuilt", "lines": units})

    def revive_dead_line(self) -> None:
        """At the forced finish, a cart line known to be sold out would be ordered anyway: swap in the best stand-in
        already read (the closest to it, or the cheapest under a lowest-price pick) when the steps allow."""
        catalog, basket = self.catalog, self.basket
        carried = basket.ids()
        dead = [i for i in carried if catalog.is_gone(i)]
        cart_tool = self.role("cart")
        if len(dead) != 1 or not cart_tool or self.steps_left() < 3 or self.desk.seconds_left() < NO_NEW_CALL + 20:
            return
        chosen = [p for p in self.reviews.stand_ins(dead) if p not in carried]
        if not chosen:
            return
        owner = catalog.owner(catalog.kind(dead[0]))
        units = self.sheet.wants[owner - 1].units if owner else basket.units(dead[0])
        text_lines = {i: basket.text_lines.get(i, 1) for i in carried if i not in dead}
        text_lines[chosen[0]] = units
        call_list = basket.calls_for(text_lines, dead)
        if not call_list or len(call_list) + 1 > self.steps_left():
            return
        for args in call_list:
            resp = self.store_call(cart_tool, args, fold=False)
            if resp is None or self.store.finished:
                return
            shaped, _ = fill_args(args, self.schemas.get(cart_tool))
            basket.record_call(cart_tool, shaped, resp.get("observation"), self.store.step)
            self.fold(cart_tool, resp.get("observation"))
        self.log.append({"kind": "revived", "dead": dead, "now": chosen[0]})

    def trim(self) -> None:
        """Shorten the older half of tool results once the history passes its size cap."""
        if sum(len(flat(m.get("content"))) for m in self.history) <= CHAT_CHARS:
            return
        results = [i for i, m in enumerate(self.history) if m.get("role") == "tool"]
        for index in results[: max(1, len(results) // 2)]:
            content = self.history[index].get("content") or ""
            if len(content) > 600:
                named = [i for i in dict.fromkeys(re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)) if i in self.catalog.cards][:20]
                self.history[index]["content"] = content[:400] + f"...[trimmed; ids inside: {', '.join(named)}]"

    def ask_now(self) -> bool:
        if self.asker.done or not self.role("ask") or self.messages_free() <= 0 or self.steps_left() <= 8:
            return False
        return len(self.catalog.read) >= 3 * len(self.sheet.wants) + 1 or self.store.step >= max(8, self.step_limit // 4)

    def loop(self, turn_cap: int, spend_cap: float) -> None:
        idle = 0
        while (not self.store.finished and not self.store.unpaid and self.steps_left() > 0 and self.turns < turn_cap
               and self.desk.spent < spend_cap and self.desk.seconds_left() > WRAP_UP):
            if self.ask_now():
                summary = self.asker.run()
                if summary:
                    self.history.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            self.trim()
            self.turns += 1
            message = self.desk.talk(self.history, [QUICK, BACKUP], 3000, self.offered())
            if message is None:
                self.blanks += 1
                if self.desk.closed() or self.blanks > 2:
                    break
                continue
            call_list = []
            for number, call in enumerate(message.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                given = fn.get("arguments")
                parsed = given if isinstance(given, dict) else (read_object(given) if isinstance(given, str) and given.strip() else {})
                call_list.append({"id": str(call.get("id") or f"t{self.turns}_{number}"), "type": "function",
                              "function": {"name": str(fn.get("name") or ""), "arguments": flat(parsed if isinstance(parsed, dict) else {})}})
            content = message.get("content")
            content = content if isinstance(content, str) else (flat(content) if content else "")
            if not call_list:
                idle += 1
                self.history += [{"role": "assistant", "content": content or "(no call)"},
                              {"role": "user", "content": "Reply with a tool call.\n" + self.status()}]
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.role("cart"), self.role("buy"), self.role("refuse")} - {None}
            if (not self.asker.done and self.role("ask") and self.messages_free() > 0 and self.steps_left() > 8
                    and any(c["function"]["name"] in committing for c in call_list)):
                # interview before the first cart change
                self.history.append({"role": "assistant", "content": content or None, "tool_calls": call_list})
                self.history += [{"role": "tool", "tool_call_id": c["id"], "content": "Skipped (no step spent): the interview comes first."} for c in call_list]
                summary = self.asker.run()
                self.history.append({"role": "user", "content": (summary or "No questions were needed.") + "\n" + self.status()})
                continue
            self.history.append({"role": "assistant", "content": content or None, "tool_calls": call_list})
            self.log.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in call_list]})
            answers = self.handle(call_list)
            for index, (cid, text) in enumerate(answers):
                if index == len(answers) - 1:
                    text += "\n" + self.status()
                self.history.append({"role": "tool", "tool_call_id": cid, "content": text})

    def run(self) -> list[dict[str, Any]]:
        if self.store.finished:
            return self.log
        self.read_request()
        self.log.append({"kind": "brief", "brief": self.sheet.raw})
        opening = self.view.get("opening")
        if opening is not None:
            self.catalog.take(opening)
        intro = "CUSTOMER REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nSHOP'S OPENING MESSAGE:\n" + self.show(opening)
        intro += ("\n\nFirst search each item's category and open several listings fully; the interviewer uses their attributes to "
                  "choose questions.\n\n" + self.status())
        self.history = [{"role": "system", "content": LOOP_RULES}, {"role": "user", "content": intro}]
        self.loop(self.turn_cap, self.desk.purse)
        if not self.store.finished:
            refused = self.finish(report_refusal=True)
            # a refused ending (changed conditions, a sell-out) gets a short extra round
            if refused and not self.store.finished and not self.store.unpaid and not self.desk.closed() and self.steps_left() > 1 and self.desk.seconds_left() > WRAP_UP:
                self.history.append({"role": "user", "content": "[Tern] Finishing failed: " + refused + "\nAct on the shop's reply, then finish.\n" + self.status()})
                self.loop(self.turns + 10, self.desk.purse + 0.005)
            if not self.store.finished:
                self.finish()
        self.log.append({"kind": "end", "finished": self.store.finished, "steps": self.store.step, "cost": round(self.desk.spent, 5), "calls": self.desk.call_list})
        return self.log


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Episode(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.log.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.store.finished:
                episode.finish()
        except Exception as again:
            episode.log.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.log


__all__ = ["agent_main"]
