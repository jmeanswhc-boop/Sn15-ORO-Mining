"""ORO Bench shopping agent: a contract-driven shopper.

How it works, in order:

1. READ.   One strong-model call compiles the shopper's request into a contract:
           every product to buy with its own requirements, exclusions and quantity,
           the price cap, stock rule, objective, message and step budgets, the
           procedures the shopper stated, the decline rule, and the properties the
           shopper left open but seems to care about. A requirement whose value lives
           on another listing ("the same colour as the lamp Lumora sells") is marked
           as a pointer. A cheap second reading cross-checks budgets and procedures.
2. ROLES.  The world's tool schemas are mapped to roles (search, open, cart, order,
           decline, message, compare) by a cheap model call, validated by code against
           the schemas, with code heuristics as the fallback.
3. DISCOVER. For each item, code runs the searches, pages, price-filter walks and
           listing opens, and keeps an evidence table: candidate x requirement, each
           cell a verdict with the listing text that proves it. Code decides the clear
           cells; a cheap model judges the rest. Pointer requirements are resolved by
           finding and opening the listing they point at and reading its value.
4. ASK.    Messages are reserved for stated procedures first. The rest go to hidden
           needs, item by item: one property per message, the shopper's hinted
           properties first, then the properties on which fitting candidates differ.
5. CHOOSE. Per item, a listing that proves every requirement wins (the cheapest or
           best when asked). Failing that, after a wider search, the listing that
           contradicts nothing and proves the most is taken: race data shows a decline
           is almost never the right ending. A decline needs every item to have no
           such listing, and follows the stated form.
6. PROCEDURES. Every stated step due before the finish is performed while budget
           allows, with proof after the last cart change; steps on a change run before
           any swap. A step is skipped only when the budget makes it impossible.
7. FINISH. Cart (one line per item at its quantity), then order; or a decline note
           that names no product.

Robustness: per-task elapsed-time deadlines (never time of day), per-task spend caps,
retries with deterministic jitter, a model ladder with fallbacks, one action per
environment call, 409 step resync, and a code-only finish when the models are gone.

The only shared code is the public protocol (ProxyClient, one action per step, the
finished check). Everything else was written for this agent.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from typing import Any

try:  # the sandbox provides this; offline tests inject a transport instead
    from src.agent.proxy_client import ProxyClient
except Exception:  # pragma: no cover - import only fails outside the sandbox
    ProxyClient = None

# ----------------------------------------------------------------------------- settings

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

LADDERS = {
    "read": [HEAVY, SPARE, LIGHT],
    "check": [HEAVY, LIGHT, SPARE],
    "judge": [LIGHT, SPARE, HEAVY],
    "write": [LIGHT, SPARE],
}
# USD per million tokens (input, output), used only when the proxy returns no cost.
PRICES = {LIGHT: (0.10, 0.50), HEAVY: (2.0, 10.0), SPARE: (1.0, 4.0)}

TASK_SECONDS = 300.0  # the network's per-task limit
SOFT_DEADLINE = 200.0  # stop exploring, head for the finish
HARD_DEADLINE = 250.0  # finish by code now
LAST_MODEL_CALL = 235.0  # no model call starts after this many seconds
SOFT_SPEND = 0.035  # USD per task: the heavy model is dropped past this
HARD_SPEND = 0.050  # USD per task: no model calls past this
SPEND_PER_EXTRA_ITEM = 0.010  # each further item raises both caps by this much: it needs its own judging
MAX_MODEL_CALLS = 48
CALLS_PER_EXTRA_ITEM = 8
DEFAULT_MESSAGES = 3  # questions when a message tool exists but no budget is stated (race worlds took three without refusing)
QUESTION_PROPS = 12  # properties one question names once this shopper has answered a list: a shopper states a held-back preference only when a question names it (race 15 took lists of eight)
FIRST_LIST_PROPS = 4  # names in a list before this shopper has answered one (race 16 refused 68% of lists naming five or more and 17% of shorter ones; race 17 answered lists of two to four, with two messages a task)
PLAN_PROPS = 20  # property names an item's questions draw on, in a stable order; each question names up to QUESTION_PROPS of them
FOLLOW_UP_ROOM = 12  # spare steps beyond the finish reserve before an item that stated a preference is asked again
FINISH_RESERVE = 6  # steps kept for cart + order, and for one swap after a pick sells out at the finish (remove, add, order)
MAX_OPENS = 16  # listing opens per task before the finish (one item, normal pace)
MAX_SEARCH_CALLS = 12
OPENS_PER_EXTRA_ITEM = 5  # each further item gets this many more opens
SEARCHES_PER_EXTRA_ITEM = 4
EARLY_SECONDS = 120.0  # while this early and well under the soft spend, the agent may look wider
EXTRA_OPENS_EARLY = 6
EXTRA_SEARCHES_EARLY = 4
MAX_ITEMS = 4
# a reply asking for one question at a time ("could we go one by one", "smaller bites", "split them up")
ONE_AT_A_TIME = re.compile(r"\bone (?:by one|at a time|thing at a time|question|subject|topic|item|prompt|query|property)\b|\bsmaller "
                           r"(?:bites|pieces|chunks|parts)\b|\bseparately\b|\bsplit (?:them|it|those) up\b|\bin turn\b|\btrickle\b|\bfirst thing\b")
POINTER_HOPS = 3  # a pointed-at listing may itself point further
MAX_ROUNDS = 4  # re-plans after notices or refused finishes (a sold-out pick at the finish is common)
TEXT_CAP = 320  # characters of any one text field shown to a model
OBS_CAP = 9000  # characters of an observation shown to a model

JSON_ONLY = "Reply with one JSON object and nothing else."


# ----------------------------------------------------------------------------- small helpers


def now() -> float:
    return time.monotonic()


def seed_from(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:12], 16)


def clip(text: Any, cap: int = TEXT_CAP) -> str:
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    return s if len(s) <= cap else s[: cap - 1] + "…"


def norm(text: Any) -> str:
    """Lowercase, punctuation to spaces, single spaces."""
    s = str(text).lower()
    s = re.sub(r"[^0-9a-zÀ-ɏ]+", " ", s)
    return " ".join(s.split())


def words(text: Any) -> list[str]:
    return norm(text).split()


def humanize(key: str) -> str:
    """A listing field name as words: dots, underscores and dashes to spaces. The whole name is kept:
    "cord.type" and "warranty.type" are different properties, and cutting both to "type" lets one overwrite the other."""
    return " ".join(str(key).replace(".", " ").replace("_", " ").replace("-", " ").split())


def plain_property(key: str, keep_code: bool = False) -> str:
    """A listing field name in words: "lamp's power source" -> "lamp power source", "screen size (e.g. 15.6 inch)" ->
    "screen size"; "xq.finish" -> "finish" unless `keep_code` keeps a catalog's short code in front ("xq finish": a
    question keeps it, since a judge reading the field's own name may need every word of it)."""
    ws = [w for w in norm(humanize(key)).split() if w != "s"]
    for i in range(len(ws) - 1):
        if ws[i] == "e" and ws[i + 1] == "g":
            ws = ws[:i]
            break
    if not keep_code and len(ws) > 1 and len(ws[0]) <= 2 and any(len(w) >= 3 and w.isalpha() for w in ws[1:]):
        ws = ws[1:]
    return " ".join(ws)


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:[.,]\d+)?", value.replace(",", ""))
        if m:
            try:
                return float(m.group(0).replace(",", "."))
            except ValueError:
                return None
    return None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "y", "1", "in stock", "available"):
            return True
        if v in ("false", "no", "n", "0", "out of stock", "unavailable", "sold out"):
            return False
    return None


def deep_find(obj: Any, pred, path: tuple = ()) -> list[tuple[tuple, Any]]:
    """All (path, value) pairs in a JSON structure for which pred(value) holds."""
    out = []
    if pred(obj):
        out.append((path, obj))
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(deep_find(v, pred, path + (k,)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(deep_find(v, pred, path + (i,)))
    return out


def get_path(obj: Any, path) -> Any:
    cur = obj
    for p in path:
        try:
            cur = cur[p]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


class TextRecord(dict):
    """A listing a store wrote as text, read into fields."""


REC_FIELD = re.compile(r"([A-Za-z][A-Za-z0-9_ .-]{0,30}?):(?:\s+|$)(.*)", re.S)
PAIR = re.compile(r"\s*([A-Za-z][A-Za-z0-9_ ./&()'-]{0,40}?)\s*[=:]\s*(.*)", re.S)


def _scalar(text: str) -> Any:
    s = text.strip()
    if s in ("None", "none", "null", "NULL", ""):
        return None
    if s in ("True", "true", "False", "false"):
        return s.lower() == "true"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return float(s) if "." in s else int(s)
    return s


def _pairs(text: str) -> dict | None:
    """'colour=black; height=40 cm' as fields, or None when the text is not such a list."""
    parts = [p for p in text.split(";") if p.strip()]
    if len(parts) < 2:
        return None
    out: dict = {}
    last = None
    for p in parts:
        m = PAIR.fullmatch(p)
        if m and m.group(1).strip() not in out:
            last = m.group(1).strip()
            out[last] = m.group(2).strip()
        elif last is not None:
            out[last] = f"{out[last]}; {p.strip()}"  # a value can hold the separator
        else:
            return None
    return {k: _scalar(v) for k, v in out.items()}


def parse_record(text: Any) -> TextRecord | None:
    """A listing written as text ("id: p-1 | title: Desk lamp | price: 1999 | facts: colour=black; height=40 cm"): its
    fields, or None when the text is not such a record. A piece without a field name continues the field before it
    (titles can hold the separator); a field holding "name=value" pairs becomes its own property sheet."""
    if not isinstance(text, str) or text.count(" | ") < 2:
        return None
    out = TextRecord()
    last = None
    for piece in text.split(" | "):
        m = REC_FIELD.fullmatch(piece.strip())
        if m and m.group(1).strip() not in out:
            last = m.group(1).strip()
            out[last] = m.group(2).strip()
        elif last is not None:
            out[last] += " | " + piece.strip()
        else:
            return None
    if len(out) < 3:
        return None
    for k, v in list(out.items()):
        sheet = _pairs(v) if ";" in v and ("=" in v or ":" in v) else None
        out[k] = sheet if sheet else _scalar(v)
    return out


def records_from_text(obs: Any) -> Any:
    """Every listing written as text inside a result, read into fields."""
    if isinstance(obs, str):
        rec = parse_record(obs)
        return rec if rec is not None else obs
    if isinstance(obs, list):
        return [records_from_text(x) for x in obs]
    if isinstance(obs, dict):
        return {k: records_from_text(v) for k, v in obs.items()}
    return obs


def parse_json(text: str) -> Any:
    """The first JSON object or array in a model reply, tolerant of fences and prose."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(s)):
                c = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                elif c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        chunk = s[start : i + 1]
                        try:
                            return json.loads(chunk)
                        except ValueError:
                            try:
                                return json.loads(re.sub(r",\s*([}\]])", r"\1", chunk))
                            except ValueError:
                                break
            start = s.find(opener, start + 1)
    return None


def spoken_number(text: str) -> int | None:
    """A small count written in words or digits ("three", "3")."""
    t = norm(text)
    m = re.search(r"\b(\d{1,3})\b", t)
    if m:
        return int(m.group(1))
    small = ("zero one two three four five six seven eight nine ten eleven twelve "
             "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()
    for i, w in enumerate(small):
        if re.search(rf"\b{w}\b", t):
            return i
    return None


# ----------------------------------------------------------------------------- schema fitting


def fit_args(schema: dict, args: dict) -> dict:
    """Coerce arguments to a tool's JSON schema so the world accepts the call."""
    props = schema.get("properties") or {}
    extra_ok = schema.get("additionalProperties", True) is not False
    out: dict = {}
    for key, value in (args or {}).items():
        if key not in props:
            if extra_ok:
                out[key] = value
            continue
        fitted = fit_value(props[key], value)
        if fitted is not None:
            out[key] = fitted
    for key in schema.get("required") or []:
        if key not in out and key in props:
            filled = fit_value(props[key], None, force=True)
            if filled is not None:
                out[key] = filled
    return out


def fit_value(spec: dict, value: Any, force: bool = False) -> Any:
    kinds = spec.get("type")
    kinds = kinds if isinstance(kinds, list) else [kinds] if kinds else []
    enum = spec.get("enum")
    if enum:
        if value in enum:
            return value
        if value is not None:
            low = norm(value)
            for e in enum:
                if norm(e) == low:
                    return e
            for e in enum:
                if low and (low in norm(e) or norm(e) in low):
                    return e
        return enum[0] if force or value is not None else None
    if "integer" in kinds or "number" in kinds:
        n = as_number(value)
        if n is None:
            if not force:
                return None
            n = float(spec.get("minimum", 1))
        if "minimum" in spec:
            n = max(n, float(spec["minimum"]))
        if "maximum" in spec:
            n = min(n, float(spec["maximum"]))
        return int(round(n)) if "integer" in kinds else (int(n) if float(n).is_integer() else n)
    if "boolean" in kinds:
        b = as_bool(value)
        return b if b is not None else (False if force else None)
    if "array" in kinds:
        if value is None:
            return [] if force else None
        items = list(value) if isinstance(value, (list, tuple, set)) else [value]
        if "maxItems" in spec:
            items = items[: int(spec["maxItems"])]
        sub = spec.get("items") or {}
        return [fit_value(sub, v) if sub else v for v in items]
    if "object" in kinds:
        return fit_args(spec, value if isinstance(value, dict) else {})
    # string or untyped
    if value is None:
        return "" if force else None
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    if "maxLength" in spec and len(s) > int(spec["maxLength"]):
        cap = int(spec["maxLength"])
        cut = s[:cap]
        if " " in cut[cap // 2 :]:
            cut = cut[: cut.rfind(" ")]
        s = cut
    if "minLength" in spec and len(s) < int(spec["minLength"]):
        s = s + " " * (int(spec["minLength"]) - len(s)) if s else "-" * int(spec["minLength"])
    return s


# ----------------------------------------------------------------------------- transport


class Wire:
    """One action per environment call, step numbering, 409 resync, retries."""

    def __init__(self, binding: dict, log: list, clock: "Clock", transport=None):
        self.binding = dict(binding or {})
        self.log = log
        self.clock = clock
        self.step = 0
        self.finished = False
        self.dead = False  # the run can no longer act (key exhausted, session gone)
        self.transport = transport
        self.calls = 0

    def post(self, path: str, body: dict, slow: bool = False):
        """Returns (data, error) like ProxyClient.post_verbose."""
        if self.transport is not None:
            return self.transport(path, body)
        client = _clients(slow)
        if client is None:
            return None, {"kind": "network", "status": None, "body": "no proxy client"}
        res = client.post_verbose(path, body)
        return res.data, res.error

    def act(self, name: str, args: dict, note: str = "") -> dict | None:
        """Send one action. Returns the observation, or None if the episode cannot continue."""
        if self.finished or self.dead:
            return None
        target = self.step + 1
        action = {"name": name, "args": args if isinstance(args, dict) else {}}
        attempts = 0
        while attempts < 4:
            attempts += 1
            data, err = self.post("/environment/call", {**self.binding, "step": target, "action": action})
            if data is not None and isinstance(data, dict) and "observation" in data:
                self.calls += 1
                self.step = int(data.get("step") or target)
                self.finished = bool(data.get("finished"))
                obs = data.get("observation")
                self.log.append({"role": "assistant", "content": note or name, "tool_call": action,
                                 "environment_result": {"step": self.step, "finished": self.finished,
                                                        "observation": _shrink(obs)}})
                return obs if obs is not None else {}
            status = (err or {}).get("status")
            body = str((err or {}).get("body") or "")
            if status == 409:
                if "finished" in body:
                    self.finished = True
                    return None
                m = re.search(r"expected (\d+)", body)
                if m:
                    expected = int(m.group(1))
                    if expected == target:  # a replay mismatch: give up on this step
                        self.dead = True
                        return None
                    self.step = expected - 1
                    target = expected
                    continue
                if "different action" in body:
                    self.step = target  # that step is spent; move on
                    target = self.step + 1
                    continue
                self.dead = True
                return None
            if status == 402:
                self.dead = True
                return None
            if status == 422:
                return {"error": "rejected_by_proxy", "detail": body[:300]}
            if self.clock.remaining() < 8:
                self.dead = True
                return None
            time.sleep(min(1.5 * attempts, 4.0))
        self.dead = True
        return None


_CLIENTS: dict = {}


def _clients(slow: bool):
    key = "slow" if slow else "fast"
    if key not in _CLIENTS:
        if ProxyClient is None:
            _CLIENTS[key] = None
        else:
            _CLIENTS[key] = ProxyClient(timeout=70 if slow else 28, max_retries=1)
    return _CLIENTS[key]


def _shrink(obj: Any, limit: int = 4000) -> Any:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) <= limit:
        return obj
    return {"truncated": s[:limit]}


class Clock:
    def __init__(self):
        self.t0 = now()

    def elapsed(self) -> float:
        return now() - self.t0

    def remaining(self) -> float:
        return TASK_SECONDS - self.elapsed()

    def past(self, mark: float) -> bool:
        return self.elapsed() >= mark


# ----------------------------------------------------------------------------- models


class Brain:
    """Model calls with a ladder per role, spend and time gates, JSON parsing."""

    def __init__(self, wire: Wire, clock: Clock, rng: random.Random, log: list):
        self.wire = wire
        self.clock = clock
        self.rng = rng
        self.log = log
        self.spend = 0.0
        self.calls = 0
        self.failed = 0
        self.soft, self.hard, self.max_calls = SOFT_SPEND, HARD_SPEND, MAX_MODEL_CALLS
        self.off = False  # no more model calls (key cap or hard spend)
        self.banned: set[str] = set()
        self.plain_content: set[str] = set()  # models that rejected multipart content
        self.no_reasoning: set[str] = set()

    def available(self) -> bool:
        return not self.off and self.calls < self.max_calls and self.spend < self.hard and not self.clock.past(LAST_MODEL_CALL)

    def ask(self, role: str, system: str, user: str, max_tokens: int = 900, effort: str = "low") -> Any:
        """Returns the parsed JSON reply, or None when no model could answer."""
        if not self.available():
            return None
        ladder = [m for m in LADDERS[role] if m not in self.banned]
        if self.spend >= self.soft:
            ladder = [m for m in ladder if m != HEAVY] or ladder
        tries = 0
        for model in ladder:
            for attempt in range(2):
                if not self.available():
                    return None
                tries += 1
                text, status = self._complete(model, system, user, max_tokens, effort, attempt)
                if text is None:
                    if status in (402,):
                        self.off = True
                        return None
                    if status in (403, 404):
                        self.banned.add(model)
                        break
                    if status == 400:  # an unsupported field: drop the reasoning setting, then multipart content
                        if model not in self.no_reasoning:
                            self.no_reasoning.add(model)
                        elif model not in self.plain_content:
                            self.plain_content.add(model)
                        else:
                            self.banned.add(model)
                            break
                        continue
                    self.failed += 1
                    if self.clock.remaining() < 30:
                        return None
                    time.sleep(min(0.8 * (2 ** attempt) + self.rng.random() * 0.7, 6.0))
                    continue
                parsed = parse_json(text)
                if parsed is not None:
                    return parsed
                user = user + "\n\n" + JSON_ONLY  # one retry with a sharper instruction
            if tries >= 5:
                break
        return None

    def _complete(self, model: str, system: str, user: str, max_tokens: int, effort: str, attempt: int):
        if model in self.plain_content:
            sys_msg = {"role": "system", "content": system}
        else:
            sys_msg = {"role": "system", "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]}
        body: dict = {"model": model, "messages": [sys_msg, {"role": "user", "content": user}], "max_tokens": max_tokens}
        if model.startswith("anthropic/") and model not in self.no_reasoning:
            body["reasoning"] = {"effort": effort}
        slow = self.clock.remaining() > 90
        data, err = self.wire.post("/inference/chat/completions", body, slow=slow)
        self.calls += 1
        if data is None:
            return None, (err or {}).get("status")
        usage = data.get("usage") if isinstance(data, dict) else None
        self.spend += self._cost(model, usage)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            if isinstance(data, dict) and "error" in data:
                code = (data.get("error") or {}).get("code") if isinstance(data.get("error"), dict) else None
                return None, code if isinstance(code, int) else 500
            return None, 500
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not isinstance(content, str) or not content.strip():
            # some providers put the answer in a tool call or a refusal field
            calls = msg.get("tool_calls") or []
            if calls:
                try:
                    content = calls[0]["function"]["arguments"]
                except (KeyError, IndexError, TypeError):
                    content = ""
        return (content if isinstance(content, str) else ""), 200

    def _cost(self, model: str, usage: Any) -> float:
        if isinstance(usage, dict):
            c = usage.get("cost")
            if isinstance(c, (int, float)) and not isinstance(c, bool) and c >= 0:
                return float(c)
            pin, pout = PRICES.get(model, (2.0, 10.0))
            pt = usage.get("prompt_tokens") or 0
            ct = usage.get("completion_tokens") or 0
            if isinstance(pt, int) and isinstance(ct, int):
                return (pt * pin + ct * pout) / 1e6
        return 0.004  # unknown: assume a typical small call


# ----------------------------------------------------------------------------- prompts

READ_SYSTEM = """You compile a shopper's request to a store assistant into a contract the assistant's code will execute. The code can only do what the contract says, so read the whole request twice: shoppers mix the order of things, wrap them in stories about their lives, and state the important parts in passing.

WHAT TO EXTRACT
1. items: every product the shopper wants to buy, one entry each ("a mattress and a headboard that works with it" is two items). "name" is the plain product noun a store would use (singular, no adjectives: "lamp", "backpack", "kettle"). "quantity" is 1 unless the shopper states a count of units of that product ("two of them" = 2; "a pair of shoes" = 1). "pairs_with" names another item this one must work with or fit ("a headboard that works with the mattress" -> "mattress"); null otherwise.
2. requirements, per item: facts a listing must state: a property value ("material: cotton", "finish: matte", "capacity: 1.7 l", "battery life: 10 hours or more"), a brand, a seller, a size, an origin, a feature, a model it must work with. "property" as a catalog would name it, "value" as the shopper wants it, keeping comparison words ("at least 10 hours", "under 500 g"). "text" keeps the shopper's own words. A condition stays a requirement even when the shopper says they will check it themselves.
   OCCASION: an occasion, event or use is a requirement (kind "soft", property "occasion") only when the shopper states it as a condition the product must suit, the way they state their other requirements ("they're for my brother's graduation", "a bag that will work at a formal dinner"): a listing that states a different occasion breaks it; one that states none does not. An occasion that only sets the scene is no requirement: where or when the shopper will use it in passing, why they are shopping, who recommended it, or an event they will be at ("I need something for the commute this winter", "a coworker says these are easy to carry", "the trip is next month, so it should arrive before then", "I have a conference dinner where I'll be giving a toast"). An event the shopper says the product is for stays a requirement however much scene they set around it ("they're for my brother's graduation, and the whole family will be there").
   POINTER: when the shopper gives a value by pointing at another listing instead of saying it ("the colour just the same as whatever's on the page for the lamps sold by Dovaro, the one with cord length 2 m, or wherever that page leads"; "anything that goes with just what the listing for Kovira's desk lamps, the one with cord length 2 m and finish matte gives, direct or via another listing it names"), set value to "" and fill "pointer": seller (the shop or seller named), kind (that listing's product kind), details (the facts given to pick out that listing, each "property: value"), relation ("same"; "compatible" for goes with / works alongside / mates up with / compatible with; "different" for other than / definitely not / different from; "at_least" for that value or above; "at_most" for no higher than / that value tops; "within" for N either way of, with tolerance N). The facts that pick out the listing can run together ("the one with cord length 2 m and finish matte"): split them into "cord length: 2 m" and "finish: matte". Otherwise pointer is null.
3. exclusions, per item: values the shopper rejects ("not glossy", "no hood", "nothing from brand X").
4. price_cap: a stated maximum price. Copy the number as written and the unit as written (cents, cts, pesos, dollars, or a bare number). applies_to is "total" unless the shopper prices each unit. A cap is not a requirement and does not by itself make the objective "cheapest".
5. stock_rule: "in_stock_only" when the shopper wants only what the store has on hand or available now, not merely listed somewhere; else "none".
6. objective: "cheapest" when the shopper asks for the least expensive among the fitting options ("pick whichever fitting one costs the least"); "best" when they ask for the most or least of a property ("the longest battery life", "the lightest"), then fill best_property and best_direction; "any" otherwise.
7. message_budget and step_budget: limits the shopper states on messages or questions to them, and on moves, steps or actions. Digits or words ("three messages at most" = 3, "ask me at most twice" = 2). null when not stated.
8. procedures: things the shopper tells the assistant to DO as part of the service, apart from choosing the product. Stage:
   - before_order: confirm before buying, tell the total or the chosen item first, wait for a yes. "Don't check out until I've said yes" needs_reply true; "tell me the total before you pay" needs_reply false.
   - before_choose: show or list the options found before picking; ask before narrowing down.
   - before_search: ask something before starting.
   - on_change: ask before replacing or swapping an item or changing the quantity.
   - at_decline: how to end when nothing fits ("if you come up empty, say so and don't suggest anything else").
   action is "message" when the step is a message to the shopper, "tool" when it is a store action (review the cart, check stock again), "none" when it only constrains how something else is done. "content" says what the message or step must contain. A product requirement, a budget and a price cap are never procedures.
9. decline_rule: decline_if_none true when the shopper says to give up rather than substitute when nothing fits; forbid_substitutes true when they do not want alternatives offered; "note" says what a decline must say or avoid, in the shopper's terms.
10. open_properties, per item: properties the shopper shows they care about without naming a value: what they will judge, what another person is particular about, what the use implies ("how it looks across the surface" -> pattern or colour; "what the fabric feels like" -> material; "it has to fit my desk" -> size). Name properties a listing would carry, most important first. The assistant will ask the shopper about each one. Empty when none.
11. search_terms, per item: 2 to 4 short store search queries: the plainest product noun first, then the noun plus the key stated value, then a synonym a store might use. No filler words.

RULES
- Every condition the shopper states is a requirement or an exclusion, however casually it is said; stories about their life are not.
- Do not invent values, budgets or procedures the request does not state, and do not drop any that it states.
- Numbers and units stay as written. Property and value strings stay short.

OUTPUT: JSON only, exactly these keys
{
 "items": [ {"name": "...", "quantity": 1, "pairs_with": null, "search_terms": ["...", "..."],
             "requirements": [ {"text": "...", "property": "...", "value": "...", "kind": "must" | "soft", "pointer": null} ],
             "exclusions":   [ {"text": "...", "property": "...", "value": "...", "kind": "exclude"} ],
             "open_properties": ["..."] } ],
 "price_cap": {"amount": 0, "unit": "...", "applies_to": "total"} | null,
 "stock_rule": "in_stock_only" | "none",
 "objective": "cheapest" | "best" | "any",
 "best_property": "..." | null, "best_direction": "max" | "min" | null,
 "message_budget": 0 | null, "step_budget": 0 | null,
 "procedures": [ {"text": "...", "stage": "before_search" | "before_choose" | "before_order" | "on_change" | "at_decline", "action": "message" | "tool" | "none", "content": "...", "needs_reply": true} ],
 "decline_rule": {"decline_if_none": true, "forbid_substitutes": true, "note": "..."},
 "summary": "one sentence"
}
A pointer, when present: {"seller": "...", "kind": "...", "details": ["property: value"], "relation": "same" | "compatible" | "different" | "at_least" | "at_most" | "within", "tolerance": "..." | null}

EXAMPLE 1 (invented). Request: "Hi! I need a kettle, the stainless steel kind, for my mum. Under 40 dollars and only if it's actually in stock. Pick the cheapest that fits. She cares a lot about how big it is, so ask me before you decide. Before paying, tell me the total and wait for my okay. If there isn't one, just say so, no alternatives. You can message me twice at most."
Contract: items [{"name": "kettle", "quantity": 1, "pairs_with": null, "search_terms": ["kettle", "stainless steel kettle", "electric kettle"], "requirements": [{"text": "the stainless steel kind", "property": "material", "value": "stainless steel", "kind": "must", "pointer": null}], "exclusions": [], "open_properties": ["capacity"]}]; price_cap {"amount": 40, "unit": "dollars", "applies_to": "total"}; stock_rule "in_stock_only"; objective "cheapest"; message_budget 2; step_budget null; procedures [{"text": "Before paying, tell me the total and wait for my okay", "stage": "before_order", "action": "message", "content": "the total price; wait for the shopper's agreement", "needs_reply": true}, {"text": "If there isn't one, just say so, no alternatives", "stage": "at_decline", "action": "none", "content": "say that nothing fits; offer no alternative", "needs_reply": false}]; decline_rule {"decline_if_none": true, "forbid_substitutes": true, "note": "say that nothing fits; no alternatives"}; summary "a stainless steel kettle under 40 dollars, in stock, the cheapest, confirmed before paying".

EXAMPLE 2 (invented). Request: "Two foam pillows and a pillow cover that fits them, please. For the cover, the colour has to be other than what's on the page for the throws sold by Dovaro, the one with material wool (or wherever that page points you). Ask me before swapping anything in the basket."
Contract: items [{"name": "pillow", "quantity": 2, "pairs_with": null, "search_terms": ["pillow", "foam pillow"], "requirements": [{"text": "foam", "property": "filling", "value": "foam", "kind": "must", "pointer": null}], "exclusions": [], "open_properties": []}, {"name": "pillow cover", "quantity": 1, "pairs_with": "pillow", "search_terms": ["pillow cover", "pillowcase"], "requirements": [{"text": "colour other than the Dovaro throw with material wool", "property": "colour", "value": "", "kind": "must", "pointer": {"seller": "Dovaro", "kind": "throw", "details": ["material: wool"], "relation": "different", "tolerance": null}}], "exclusions": [], "open_properties": []}]; procedures [{"text": "Ask me before swapping anything in the basket", "stage": "on_change", "action": "message", "content": "what would be replaced and by what; wait for a yes", "needs_reply": true}]; the other fields as the request states them.
""" + JSON_ONLY

RECHECK_SYSTEM = """You verify a reading of a shopper's request. You get the request and the reading's budgets and procedures. Check three things against the request's own words:
- message_budget: a stated limit on messages or questions to the shopper, in digits or words; null when none is stated.
- step_budget: a stated limit on moves, steps or actions; null when none is stated.
- procedures: every step the shopper tells the assistant to perform as part of the service: confirm before ordering and wait for a yes, tell the total or the chosen item first, show the options found, ask before swapping, how to end when nothing fits. List each one the request states and none it does not. A product requirement, a budget or a price cap is not a procedure. Stages: before_search, before_choose, before_order, on_change, at_decline. action: message, tool or none. needs_reply is true only when the shopper must answer before the assistant may go on.
Output JSON only:
{"message_budget": 0 | null, "step_budget": 0 | null, "procedures": [ {"text": "...", "stage": "...", "action": "...", "content": "...", "needs_reply": true} ], "agree": true, "note": "what you changed, if anything"}
""" + JSON_ONLY

ROLES_SYSTEM = """You map a store's tools to the roles a shopping assistant needs. You get the tool schemas: names, descriptions, argument names and types. Decide from the descriptions and the argument names together. Never invent a tool or an argument; use null for anything the store does not offer.
Roles:
- search: finds listings. query_arg takes free text; page_arg selects a result page; max_price_arg is an upper price bound; mode_arg switches keyword search and filtered search (give keyword_mode_value and filter_mode_value); category_arg and brand_arg filter by department or maker.
- open: returns ONE listing with its properties, specifications or attributes (the property sheet), taking the listing id in id_arg. When two tools return one listing, open is the one whose result includes properties; the other goes in open_alternatives.
- compare: returns several listings from a list of ids (ids_arg).
- cart_add: puts a listing in the cart: id_arg, quantity_arg, and mode_arg with add_value and remove_value when one tool both adds and removes. When the tool takes a list of cart lines instead (each line an object holding a listing id and a count), give the list argument in lines_arg, the line's id field in line_id_key and its count field in line_qty_key, and lines_replace: true when the list becomes the whole cart (lines left out are removed), false when its lines are added.
- cart_view: shows the cart.
- order: places the order, checks out, pays. decline: ends without buying; note_arg is a note to the shopper. When one tool does both through an argument value, give that tool for both roles with "args" holding the argument and its value for each.
- message: sends a message to the shopper; text_arg carries the text.
Output JSON only:
{
 "search":   {"tool": null, "query_arg": null, "page_arg": null, "max_price_arg": null, "mode_arg": null, "keyword_mode_value": null, "filter_mode_value": null, "category_arg": null, "brand_arg": null},
 "open":     {"tool": null, "id_arg": null},
 "open_alternatives": [],
 "compare":  {"tool": null, "ids_arg": null},
 "cart_add": {"tool": null, "id_arg": null, "quantity_arg": null, "mode_arg": null, "add_value": null, "remove_value": null, "lines_arg": null, "line_id_key": null, "line_qty_key": null, "lines_replace": null},
 "cart_view":{"tool": null},
 "order":    {"tool": null, "args": {}},
 "decline":  {"tool": null, "note_arg": null, "args": {}},
 "message":  {"tool": null, "text_arg": null}
}
""" + JSON_ONLY

FIELDS_SYSTEM = """You name the parts of a store tool's result so code can read it. You get one result.
- list_path: the keys from the top of the result down to the list of listings; [] when the result is a single listing.
- id_key: the field that identifies a listing (the value other tools take as the listing id). title_key: its name. price_key: its price, a number. stock_key: whether it is in stock or available. category_key: the field naming the listing's department or category. props_key: the field whose value is a dictionary of properties or attributes; null when there is none.
- next_page_key: the top-level field that says whether or which page comes next (a number, a token or null). total_key: the top-level count of all matches.
- notes: top-level strings that read as messages or notices to the assistant rather than listing data.
Use the exact key names from the result and null for anything absent. Output JSON only:
{"list_path": [], "id_key": null, "title_key": null, "price_key": null, "stock_key": null, "category_key": null, "props_key": null, "next_page_key": null, "total_key": null, "notes": []}
""" + JSON_ONLY

JUDGE_SYSTEM = """You check store listings against a shopper's requirements. You get numbered requirements (property: value) and listings with title, price, stock and stated properties. Give one verdict per requirement per listing.
- yes: a stated property, or failing that the title, states the required value. The same meaning counts whatever the spelling, spacing, case or unit style ("1 l" = "1L" = "1 liter"; "pull over" = "pullover"). A value listed among several counts ("halal, caffeine free" has "caffeine free"). A more specific value that includes the required one counts ("100% cotton" for "cotton"; "dark blue" for "blue"). Numbers compare as quantities with their units, honouring "at least", "at most", "under", "or less" and ranges.
- Which property: one whose name holds the required property's words ("frame material" or "size group" for "material" or "size") or is a shorter form of it ("type" for "type of lamp"). A property that only shares a word with it is a different property ("desk lamp style" is not "type of lamp"): the same value there does not count. A requirement marked ONLY THE VALUE WAS GIVEN is met by any property that states the value ("handle style: long handle" for "long handles").
- no: the listing states a conflicting value for that property ("round" for "square"; "5 ms" for "1 ms or less"), or it is a different kind of product.
- unclear: nothing shown settles it (the property is absent and the title does not say).
- unclear, not yes, when the listing contradicts itself: its title, or the property named plainly for the requirement, states a different value beside or instead of the required one ("up to 4K" or "1080p" in the title beside a stated resolution of 2560 x 1080; "32/40/43 inch" in the title for "40 inch"; "1.5 l" or "2 l" in the title beside a stated capacity of 1.7 l; "resolution: 720p" beside "screen resolution: 2560 x 1080"). A field that lists many features or tags ("features: hdr, 4k, 1080p, cordless, auto shut-off") is not such a statement.
- A requirement marked EXCLUDED VALUE: yes when the listing states that value, no when it states a different value for that property, unclear otherwise.
- The product kind: yes when the title or a category field names it or a plain synonym; no when it names a different kind, including a part or accessory for it ("phone case" is not "phone").
- What a listing works with or fits is read from its text when one is shown ("works with every AB21xx model, except AB2130" covers AB2110 but not AB2130 or CD4410).
Judge only from what is shown. Never infer from the brand, the price or the kind of product. Output JSON only:
{"listings": [ {"id": "...", "verdicts": {"1": "yes" | "no" | "unclear", "2": "..."}, "note": "short"} ]}
""" + JSON_ONLY

RANK_SYSTEM = """You rank listings that already meet a shopper's requirements by one stated property. You get the property, the direction (max or min) and the listings with their properties. Read the property's value from each listing as a quantity with its unit; a listing without a value for the property goes last; ties keep the given order. Output JSON only: {"order": ["best id", "next", "..."], "note": "short"}
""" + JSON_ONLY

REPLY_SYSTEM = """You read a shopper's reply to a store assistant's message. You get the message that was sent and the reply. Output JSON only:
{"kind": "preference" | "no_preference" | "confused" | "one_at_a_time" | "yes" | "no" | "change" | "other", "property": "..." | null, "value": "..." | null,
 "pointer": {"seller": "...", "kind": "...", "details": ["property: value"], "relation": "same" | "compatible" | "different" | "at_least" | "at_most" | "within", "tolerance": "..." | null} | null,
 "item": "..." | null, "new_requirements": [ {"property": "...", "value": "...", "kind": "must" | "exclude", "item": "..." | null} ], "price_cap": 0 | null, "quantity": 0 | null, "note": "short"}
- preference: the shopper says what they want for an asked property (a message can ask about several). value is their words; property is the one the answer is about, as the message named it, unless they name another.
- A preference can point at another listing instead of stating the value ("just the same as whatever's on the page for the tablets sold by Belkara that has display size 10.3 inch, or wherever that page points you"; "owt other than just what the listing for Kovira's desk lamps, the one with cord length 2 m and finish matte gives, direct or via another listing it names"). Then value is null and pointer holds: seller (the shop or seller named), kind (that listing's product kind), details (the facts that pick out that listing, each "property: value"), relation ("same" for same as / matching / equal to; "compatible" for goes with / works alongside / mates up with / compatible with; "different" for other than / definitely not / different from; "at_least" for that value or above; "at_most" for no higher than / that value tops; "within" for N either way of, with tolerance N). The facts that pick out the listing can run together ("the one with cord length 2 m and finish matte"): split them into "cord length: 2 m" and "finish: matte". Shoppers wrap this in odd phrasing; read past it.
- Emphatic framing states the value as wanted, not excluded: "rules out anything but X", "nothing else will do", "I won't budge on X" all mean X. A value is excluded only when the reply rejects that value itself ("anything other than X", "definitely not X"). The value is what a listing could state, without the framing around it: "at most 16 gb" from "memory matching at most 16 gb", "oval" from "settled on oval".
- item: the product the reply is about when several are being bought and the reply names one; null otherwise.
- no_preference: they do not mind, any option is fine, they leave it to the assistant.
- one_at_a_time: they ask for one question or one property at a time, or for the questions in smaller pieces ("could we go one by one", "just ask me the first thing", "split them up"); they state no preference.
- confused: they did not understand and want it said again more plainly.
- yes / no: an answer to a yes-or-no question (a confirmation or a refusal).
- change: they change the request: a new or dropped requirement, a new budget or price cap, a different quantity. List what changed in new_requirements / price_cap; a price cap is the number exactly as the reply writes it, in its own unit (never convert cents or cts to dollars). "Yes, but make it blue" is a change, not a yes.
- other: anything else (an acknowledgment, a question back).
A reply can carry a preference and a change at once: then kind is "change" and both parts are filled.
""" + JSON_ONLY

NOTICE_SYSTEM = """You read notices that a store or shopper sent to a shopping assistant during a task: a note attached to a tool result, a message that arrived unprompted, or the text the store returned when it refused to complete an order or a decline. You get the assistant's current contract (with what is in the cart) and the notices. Decide what changes:
- a new or changed requirement or exclusion -> new_requirements
- a requirement or exclusion the shopper drops ("never mind about X", "forget X", "skip X", "X isn't worth the trouble") -> dropped, copying each one exactly as it is written in the contract
- a new price limit (higher or lower), quantity, message budget or step budget -> the matching field; a price limit is the number exactly as the notice writes it, in the notice's own unit (never convert cents or cts to dollars)
- the chosen item in the cart can no longer be bought (sold out, gone, none left, unavailable, can't be ordered) -> cart_unavailable true, and item names the product when several are being bought
- any other change to a listing in the cart (its price or properties) -> recheck_cart true
- a step the shopper now wants performed (confirm first, tell the total, show the options, ask before swapping) -> procedure, with its stage
- a refused finish usually means one of the above happened just now; read it that way before assuming a skipped step
kind is "change" when the contract changes, "procedure" when only a step is added, "info" when nothing changes. Combine several notices into one answer. When several products are being bought, "item" on a requirement or on the quantity names the product it is about.
A requirement can point at another listing instead of stating its value ("just the same as whatever's on the page for the jackets sold by Voshan, the one with pattern stripe"; "anything that goes with just what the listing for Voshan's jackets, the one with pattern stripe and lining fleece gives, direct or via another listing it names"): then value is "" and pointer holds seller, kind, details (each "property: value"), relation (same, compatible, different, at_least, at_most, within) and tolerance. Output JSON only:
{"kind": "change" | "procedure" | "info", "new_requirements": [ {"property": "...", "value": "...", "kind": "must" | "exclude", "item": "..." | null, "pointer": null} ], "dropped": ["property: value"],
 "price_cap": 0 | null, "quantity": 0 | null, "quantity_item": "..." | null, "message_budget": 0 | null, "step_budget": 0 | null,
 "procedure": {"text": "...", "stage": "before_order" | "before_choose" | "on_change" | "at_decline", "action": "message" | "tool" | "none", "content": "...", "needs_reply": true} | null,
 "cart_unavailable": false, "recheck_cart": false, "item": "..." | null, "note": "short"}
""" + JSON_ONLY

POINTER_SYSTEM = """You follow a trail of store listings to find one property value for a shopper. You get the property, the relation the shopper wants, and one listing's page: title, seller, properties and texts. Read the texts closely: a page can send you on to another listing for this property, state the value itself, or state a matching rule.
- refers: when the page says the property is found on, comes from, or should be checked on another listing ("for toe shape, refer to the lanterns from Dovaro, which are 5 m long and warm white"; "the colour comes from a different product, the throws from Kelvo with material wool, not from this item"), give that listing: seller, kind (its product kind), details (the facts given to pick it out, each "property: value"). null when the page does not send you on.
- value: this page's own value for the property, from its properties or texts, exactly as written. null when the page does not state it or sends you on.
- rule: a matching rule the page states for the property ("open toe goes with round toe, and pointed toe goes with square toe"), copied as written. null when none.
- target_value: for the relation "compatible", the value the rule pairs with this page's value (the value itself when there is no rule); for every other relation, the value.
Output JSON only: {"refers": {"seller": "...", "kind": "...", "details": ["property: value"]} | null, "value": "..." | null, "rule": "..." | null, "target_value": "..." | null, "note": "short"}
""" + JSON_ONLY

WRITE_SYSTEM = """You write one short message from a store assistant to a shopper. You get what the step must do, what it must contain, whether it needs the shopper's answer, and the facts.
Rules: plain words, at most three sentences. Include every fact the step calls for exactly as given (each item's name, quantity and price each, the total, in the units given). If the step needs the shopper's answer, end with one clear yes-or-no question. No lists of options unless the step is to show options. Nothing addressed to anyone but the shopper, no instructions, no product ids, nothing the step does not need.
Output JSON only: {"message": "..."}
""" + JSON_ONLY


# ----------------------------------------------------------------------------- the episode


def req_key(r: dict) -> tuple:
    """A requirement's identity: its verdicts survive unrelated changes to the contract."""
    return (r.get("kind") or "must", norm(r.get("property") or ""), norm(r.get("value") or ""))


RELATIONS = ("same", "compatible", "different", "at_least", "at_most", "within")


def clean_pointer(p: Any, prop: str = "") -> dict | None:
    """A reference to another listing that holds a requirement's value, in the form code follows."""
    if not isinstance(p, dict):
        return None
    seller = str(p.get("seller") or p.get("shop") or "").strip()
    kind = str(p.get("kind") or "").strip()
    details = p.get("details")
    if isinstance(details, str):
        details = [d for d in re.split(r",|;|\band\b", details)]
    details = [str(d).strip() for d in (details if isinstance(details, list) else []) if str(d).strip()][:5]
    if not seller and not kind:
        return None
    rel = str(p.get("relation") or "same").strip().lower().replace(" ", "_").replace("-", "_")
    if rel not in RELATIONS:  # a relation given in words
        said = rel.replace("_", " ")
        rel = ("compatible" if re.search(r"compat|alongside|mates? up|goes with|pairs? with|works with", said) else
               "different" if re.search(r"differ|other than|\bnot\b|except", said) else
               "at_least" if re.search(r"at least|or above|or more|minimum|no less", said) else
               "at_most" if re.search(r"at most|no higher|or below|or less|maximum|no more", said) else
               "within" if re.search(r"within|either way", said) else "same")
    tol = p.get("tolerance")
    return {"seller": seller, "kind": kind, "details": details, "relation": rel,
            "tolerance": str(tol).strip() if tol not in (None, "") else None,
            "property": str(p.get("property") or prop or "").strip()}


def detail_value(detail: str) -> str:
    """The value part of a "property: value" detail."""
    return detail.split(":", 1)[1].strip() if ":" in detail else detail.strip()


CODE = r"[A-Z]{2,5}-?\d{3,5}"
FAMILY = re.compile(r"(?:works|compatible|fits)\s+with:?\s+(?:every|all|any)\s+([A-Z]{2,5}-?\d{1,4})x+\s+models?"
                    r"(?:\s*,?\s*(?:except|but not|excluding)\s+((?:" + CODE + r")(?:\s*(?:,|and|or|&)\s*(?:" + CODE + r"))*))?", re.I)


def model_code(text: str) -> str | None:
    """A device's model number from its text ("Model no. AB2130")."""
    m = re.search(r"\bmodel\s*(?:no|number|#)?\s*\.?\s*:?\s*(" + CODE + r")\b", text or "", re.I)
    return m.group(1).upper() if m else None


def works_with(text: str, code: str) -> bool | None:
    """Whether a part's text says it works with a device model: True, False, or None when it says nothing about models."""
    said = None
    for m in FAMILY.finditer(text or ""):
        fam = m.group(1).upper()
        excepted = {x.upper() for x in re.findall(CODE, m.group(2) or "", re.I)}
        if code.upper().startswith(fam) and code.upper() not in excepted:
            return True
        said = False
    return said


def code_family(code: str) -> str:
    return re.sub(r"\d{2}$", "xx", code)


class Item:
    """One product the shopper wants: its own requirements, exclusions, quantity, pick and search state."""

    def __init__(self, idx: int, name: str, quantity: int = 1):
        self.idx = idx
        self.name = name
        self.quantity = quantity
        self.terms: list[str] = []
        self.reqs: list[dict] = [{"text": name, "property": "product type", "value": name, "kind": "must"}]
        self.excls: list[dict] = []
        self.open_props: list[str] = []
        self.pointers: list[dict] = []  # requirements whose value is on another listing, not yet read
        self.unresolved: list[dict] = []
        self.pairs_with: str | None = None
        self.heard = 0  # answers that stated a preference for this item
        self.questions = 0  # questions sent about this item
        self.plan: list[str] = []  # the property names its questions draw on, in a stable order
        self.answered: list[str] = []  # properties the shopper stated a preference for
        self.sent: list[tuple] = []  # the name lists its questions named
        self.refused: set = set()  # properties asked alone that drew only a one-at-a-time refusal (asked once more later)
        self.follow_up = False  # an answer stated a preference for one of several named properties: ask again at once
        self.order = 0  # which order of its property list the next question uses (moves on after an answer that stated nothing)
        self.rejected: set[str] = set()
        self.search_exhausted = False
        self.cheaper_walk_done = False
        self.first_fit_open: int | None = None  # the open count when the first proven listing appeared
        self.dept_cache: tuple | None = None  # (rows seen, home department, how sure)
        self.pick: "Candidate | None" = None


class Candidate:
    def __init__(self, cid: str, row: dict | None = None):
        self.id = cid
        self.title = ""
        self.price: float | None = None
        self.stock: bool | None = None
        self.props: dict = {}
        self.opened = False
        # req_key -> yes (a property states it) / title (only the title says so: weak) / no / unclear;
        # for an exclusion, yes or title means the listing has the excluded value
        self.verdicts: dict[tuple, str] = {}
        self.deferred: set[tuple] = set()  # cells a property settles but the title may contradict: left to the model
        self.row = row or {}
        self.seen_order = 0
        self.items: set[int] = set()  # the items whose searches returned this listing
        self.text = ""  # free text: description, fine print
        self.seller = ""
        self.dept = ""  # the listing's department or category, when the store names one
        self.found_with: set[str] = set()  # stated values in the queries that returned this row

    def fits(self, reqs: list[dict], excls: list[dict]) -> bool:
        """Every requirement is proven by the listing's properties (a title alone is weak evidence: those listings are
        only the fallback, so the search goes on for a proven one) and no excluded value shows. A soft requirement (the
        occasion it is for) only must not be contradicted."""
        return (all(self.verdicts.get(req_key(r)) != "no" if r.get("kind") == "soft" else self.verdicts.get(req_key(r)) == "yes" for r in reqs)
                and not any(self.verdicts.get(req_key(e)) in ("yes", "title") for e in excls))

    def strong(self, reqs: list[dict], excls: list[dict]) -> bool:
        return self.fits(reqs, excls)

    def ruled_out(self, reqs: list[dict], excls: list[dict]) -> bool:
        return any(self.verdicts.get(req_key(r)) == "no" for r in reqs) or any(self.verdicts.get(req_key(e)) in ("yes", "title") for e in excls)

    def undecided(self, reqs: list[dict], excls: list[dict]) -> list[dict]:
        return [r for r in reqs + excls if self.verdicts.get(req_key(r)) is None]


class Episode:
    def __init__(self, problem: dict, transport=None):
        self.clock = Clock()
        self.log: list = []
        env = problem.get("environment") or {}
        self.binding = env.get("binding") or {}
        self.view = env.get("policy_view") or {}
        self.request: str = str(self.view.get("query") or problem.get("query") or "")
        self.opening = self.view.get("opening")
        self.tools: list[dict] = [self._flat(t) for t in (self.view.get("tools") or []) if isinstance(t, dict)]
        self.tool_by_name = {t["name"]: t for t in self.tools if t.get("name")}
        self.max_steps = int(self.view.get("max_steps") or 30)
        self.rng = random.Random(seed_from(self.request))
        self.wire = Wire(self.binding, self.log, self.clock, transport)
        self.wire.finished = bool(self.view.get("finished"))
        self.brain = Brain(self.wire, self.clock, self.rng, self.log)
        # contract
        self.contract: dict = {}
        self.items: list[Item] = [Item(0, "item")]
        self.cur: Item = self.items[0]  # the item that discovery, judging and asking work on
        self.item_step_cap: int | None = None  # the step at which the current item's share of exploring ends
        self.price_cap: float | None = None
        self.cap_total = True
        self.stock_only = False
        self.objective = "any"
        self.best_property = None
        self.best_direction = "max"
        self.procedures: list[dict] = []
        self.decline_rule: dict = {}
        self.msg_budget: int | None = None
        # roles and world knowledge
        self.roles: dict = {}
        self.fields: dict[str, dict] = {}  # tool name -> field map
        self.field_tries: dict[str, int] = {}
        self.shapes: dict[str, set] = {}  # tool name -> top-level keys of its plain results
        self.wrappers: dict[str, tuple] = {}  # tool name -> (key set of a notice wrapper, key of the wrapped result)
        self.search_kind: str | None = None  # 'all' or 'any' words
        # progress
        self.cands: dict[str, Candidate] = {}
        self.seen_counter = 0
        self.messages_sent = 0
        self.message_refused = False
        self.question_steps = 0  # steps the exploring keeps for the second round of questions
        self.opens = 0
        self.searches = 0
        self.notices: list[str] = []
        self.notice_seen: dict[str, int] = {}  # a notice's words -> the finish attempt they arrived at
        self.finish_attempts = 0
        self.proofs: dict[int, int] = {}  # procedure index -> step of proof
        self.last_cart_change = -1
        self.cart: dict[str, int] = {}
        self.single_questions = False  # the shopper asked for one question at a time: one property per question from then on
        self.lists_taken = False  # a question naming several properties drew an answer rather than a refusal: full lists from then on
        self.heard_queue: list = []  # (item, value) of preferences heard in replies, each owed a search of its own
        self.search_bonus = 0  # searches allowed past the cap for those
        self.zero_checked = False  # a count-0 removal was checked with a cart view
        self.zero_fails = False  # that check showed a count of 0 leaves the line in
        self.cart_known_lines: list = []
        self.picks: dict[int, Candidate] = {}  # item index -> the listing chosen for it
        self.finished_by: str | None = None
        self.rounds = 0
        self.notes: list[str] = []
        self.page_size: int | None = None
        self.queried: set = set()
        self.text_tool: str | None = None  # the open tool whose page carries free text (fine print)
        self.pointer_rows: dict[str, list[str]] = {}  # pointer query -> listing ids it returned
        self.ceiling: float | None = None  # a temporary price limit while looking for a cheaper line
        self.keyword_ignores_cap = False  # the store's keyword search returned rows above the price cap it was given
        self.cart_checked = -2  # the cart change last looked at with the cart view

    # the current item's state: the discovery, judging and asking code reads one item at a time
    reqs = property(lambda self: self.cur.reqs, lambda self, v: setattr(self.cur, "reqs", v))
    excls = property(lambda self: self.cur.excls, lambda self, v: setattr(self.cur, "excls", v))
    quantity = property(lambda self: self.cur.quantity, lambda self, v: setattr(self.cur, "quantity", v))
    search_exhausted = property(lambda self: self.cur.search_exhausted, lambda self, v: setattr(self.cur, "search_exhausted", v))
    cheaper_walk_done = property(lambda self: self.cur.cheaper_walk_done, lambda self, v: setattr(self.cur, "cheaper_walk_done", v))

    def focus(self, item: Item, share: int | None = None) -> None:
        """Work on one item. With `share`, it gets 1/share of the steps left for exploring, so one item cannot use them all."""
        self.cur = item
        self.item_step_cap = None
        if share and len(self.items) > 1:
            spare = self.steps_left() - self.finish_reserve() - 1
            self.item_step_cap = self.wire.step + max(3, spare // share)

    def open_cap(self) -> int:
        """Listing opens allowed: more for more items, and more while the run is early and cheap."""
        cap = MAX_OPENS + OPENS_PER_EXTRA_ITEM * (len(self.items) - 1)
        if self.clock.elapsed() < EARLY_SECONDS and self.brain.spend < 0.7 * self.brain.soft:
            cap += EXTRA_OPENS_EARLY
        return cap

    def search_cap(self) -> int:
        cap = MAX_SEARCH_CALLS + SEARCHES_PER_EXTRA_ITEM * (len(self.items) - 1) + self.search_bonus
        if self.clock.elapsed() < EARLY_SECONDS and self.brain.spend < 0.7 * self.brain.soft:
            cap += EXTRA_SEARCHES_EARLY
        return cap

    # ------------------------------------------------------------------ utilities

    @staticmethod
    def _flat(tool: dict) -> dict:
        f = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        return {"name": f.get("name"), "description": f.get("description") or "", "parameters": f.get("parameters") or {"type": "object", "properties": {}}}

    def diag(self, text: str) -> None:
        self.notes.append(text)
        self.log.append({"role": "assistant", "content": clip(text, 600)})

    def steps_left(self) -> int:
        return self.max_steps - self.wire.step

    def exploring_ok(self) -> bool:
        return (not self.wire.finished and not self.wire.dead and self.steps_left() > self.finish_reserve() + 1 + self.question_steps
                and not self.clock.past(SOFT_DEADLINE) and (self.item_step_cap is None or self.wire.step < self.item_step_cap))

    def finish_reserve(self) -> int:
        """Steps kept for the finish: a cart line per item, the order, one repair, and the stated steps before ordering."""
        tool_steps = sum(1 for p in self.procedures if p.get("action") == "tool" and p.get("stage") in ("before_order", "before_choose"))
        msg_steps = sum(1 for p in self.procedures if p.get("action") == "message" and p.get("stage") in ("before_order", "before_choose"))
        return FINISH_RESERVE + (len(self.items) - 1) + tool_steps + msg_steps

    def req_count(self) -> int:
        return len(self.reqs)

    def tool(self, role: str) -> dict | None:
        name = (self.roles.get(role) or {}).get("tool")
        return self.tool_by_name.get(name) if name else None

    def role_arg(self, role: str, key: str) -> str | None:
        v = (self.roles.get(role) or {}).get(key)
        return v if isinstance(v, str) and v else None

    def call(self, role: str, args: dict, note: str = "") -> dict | None:
        """Call the tool bound to a role with schema-fitted arguments; absorb notices."""
        tool = self.tool(role)
        if tool is None:
            return None
        fitted = fit_args(tool["parameters"], args)
        obs = self.wire.act(tool["name"], fitted, note or role)
        if obs is None:
            return None
        return self.absorb(tool["name"], obs)

    # ------------------------------------------------------------------ observation handling

    def absorb(self, tool_name: str, obs: Any) -> Any:
        """Unwrap notice wrappers, collect notices, remember each tool's plain result shape."""
        # the runtime's envelope for a finish the world refused
        if isinstance(obs, dict) and "refusal" in obs and "observation" in obs:
            self._collect_notice_strings(obs["refusal"], refusal=True)
            obs = obs.get("observation")
        # listings written as text become fields before anything reads them (a list of them is not a list of notices)
        obs = records_from_text(obs)
        if isinstance(obs, dict) and len(obs) == 1 and isinstance(next(iter(obs.values())), TextRecord):
            obs = next(iter(obs.values()))  # one listing under a single key is the listing
        if not isinstance(obs, dict) or ("error" in obs and len(obs) <= 2):
            return obs
        keys = frozenset(obs.keys())
        known = self.shapes.get(tool_name)
        wrap = self.wrappers.get(tool_name)
        if wrap and keys == wrap[0]:
            for k, v in obs.items():
                if k != wrap[1]:
                    self._collect_notice_strings(v)
            return obs.get(wrap[1])
        note_lists = [(k, v) for k, v in obs.items() if isinstance(v, list) and self._is_note(v)]
        rest = [(k, v) for k, v in obs.items() if not (isinstance(v, list) and self._is_note(v))]
        if len(note_lists) >= 1 and len(rest) == 1:
            # a wrapper: the plain result (a record, a list of records, or a reply string) under one key, notices
            # as lists of sentences under the others
            dk, dv = rest[0]
            inner_keys = set(dv.keys()) if isinstance(dv, dict) else (set().union(*(set(x.keys()) for x in dv if isinstance(x, dict))) if isinstance(dv, list) else set())
            if known is None or (inner_keys & known) or not isinstance(dv, (dict, list)):
                for _, v in note_lists:
                    self._collect_notice_strings(v)
                self.wrappers[tool_name] = (keys, dk)
                if known is None and inner_keys:
                    self.shapes[tool_name] = set(inner_keys)
                return dv
        data = [(k, v) for k, v in obs.items() if isinstance(v, dict) or (isinstance(v, list) and v and all(isinstance(x, dict) for x in v))]
        texts = [(k, v) for k, v in obs.items() if self._is_note(v)]
        if known is not None and len(data) == 1 and texts and len(data) + len(texts) == len(obs) and not (keys & known):
            dk, dv = data[0]
            inner_keys = set(dv.keys()) if isinstance(dv, dict) else set().union(*(set(x.keys()) for x in dv))
            if inner_keys & known:  # a wrapper with single-string notices, recognized by the known shape inside
                for _, v in texts:
                    self._collect_notice_strings(v)
                self.wrappers[tool_name] = (keys, dk)
                return dv
        if known is None:
            self.shapes[tool_name] = set(keys)
        elif keys & known:
            for k in keys - known:
                if self._is_note(obs[k]):
                    self._collect_notice_strings(obs[k])
        else:  # a different variant of this tool's result; learn it, say nothing
            self.shapes[tool_name] = known | set(keys)
        return obs

    @staticmethod
    def _is_note(v: Any) -> bool:
        if isinstance(v, str):
            return len(v.strip()) >= 16 and len(v.split()) >= 3
        if isinstance(v, list) and v:
            return all(isinstance(x, str) for x in v) and any(len(x.strip()) >= 16 and len(x.split()) >= 3 for x in v)
        return False

    def _collect_notice_strings(self, v: Any, refusal: bool = False) -> None:
        """Queue each notice once. A store hands a refused finish's notes over in the refusal and again with the next
        tool result: the same words again are that second delivery, not news, unless a later finish is refused with them."""
        items = v if isinstance(v, list) else [v]
        for x in items:
            if not isinstance(x, str) or len(x.strip()) < 12:
                continue
            seen = self.notice_seen.get(norm(x))
            if seen is not None and not (refusal and seen < self.finish_attempts):
                continue
            self.notice_seen[norm(x)] = self.finish_attempts
            if x.strip() not in self.notices:
                self.notices.append(x.strip())

    # ------------------------------------------------------------------ phase 1: read

    def read_request(self) -> bool:
        user = "REQUEST:\n" + self.request
        if self.opening is not None:
            user += "\n\nWHAT THE STORE SHOWS FIRST:\n" + clip(self.opening, 1500)
        user += "\n\nTOOLS THE STORE OFFERS (names and descriptions):\n" + "\n".join(
            f"- {t['name']}: {clip(t['description'], 200)}" for t in self.tools)
        data = self.brain.ask("read", READ_SYSTEM, user, max_tokens=3000, effort="medium")
        if not isinstance(data, dict) or not isinstance(data.get("items"), list) or not data["items"]:
            data = self.brain.ask("read", READ_SYSTEM, user, max_tokens=3000, effort="medium")
        if not isinstance(data, dict) or not isinstance(data.get("items"), list) or not data["items"]:
            self.diag("read: no contract from the models; code-only fallback")
            self.contract = {}
            self._fallback_contract()
            return False
        self.contract = data
        self._apply_contract(data)
        extra = len(self.items) - 1
        self.brain.soft = SOFT_SPEND + SPEND_PER_EXTRA_ITEM * extra
        self.brain.hard = HARD_SPEND + SPEND_PER_EXTRA_ITEM * extra
        self.brain.max_calls = MAX_MODEL_CALLS + CALLS_PER_EXTRA_ITEM * extra
        # cheap cross-check of budgets and procedures
        check_user = ("REQUEST:\n" + self.request + "\n\nREADING:\n" + json.dumps(
            {"message_budget": self.msg_budget, "step_budget": self.contract.get("step_budget"),
             "procedures": self.procedures}, ensure_ascii=False))
        check = self.brain.ask("judge", RECHECK_SYSTEM, check_user, max_tokens=1200)
        if isinstance(check, dict):
            self._merge_recheck(check)
        items = "; ".join(f"{it.name} x{it.quantity} reqs={len(it.reqs)} excl={len(it.excls)} pointers={len(it.pointers)}"
                          + (f" pairs_with={it.pairs_with}" if it.pairs_with else "") + (f" open={it.open_props}" if it.open_props else "") for it in self.items)
        self.diag(f"contract: {clip(self.contract.get('summary', ''), 200)} | {items} | cap={self.price_cap} "
                  f"stock_only={self.stock_only} objective={self.objective} msgs={self.msg_budget} steps={self.max_steps} procs={len(self.procedures)}")
        return True

    def _make_item(self, idx: int, spec: dict) -> Item:
        name = str(spec.get("name") or "item").strip() or "item"
        q = as_number(spec.get("quantity"))
        it = Item(idx, name, int(q) if q and 1 <= q <= 50 else 1)
        it.terms = [str(t) for t in (spec.get("search_terms") or []) if isinstance(t, (str, int, float)) and str(t).strip()][:6]
        for r in spec.get("requirements") or []:
            if not isinstance(r, dict):
                continue
            ptr = clean_pointer(r.get("pointer"), str(r.get("property") or ""))
            if ptr and ptr["property"]:
                ptr["text"] = str(r.get("text") or "")
                it.pointers.append(ptr)
            elif r.get("value") or r.get("text"):
                it.reqs.append(self._named({"text": str(r.get("text") or ""), "property": str(r.get("property") or ""), "value": str(r.get("value") or r.get("text") or ""),
                                            "kind": "soft" if r.get("kind") == "soft" else "must"}))
        it.excls = [self._named({"text": str(r.get("text") or ""), "property": str(r.get("property") or ""), "value": str(r.get("value") or r.get("text") or ""), "kind": "exclude"})
                    for r in spec.get("exclusions") or [] if isinstance(r, dict) and (r.get("value") or r.get("text"))]
        it.open_props = [str(x) for x in (spec.get("open_properties") or []) if isinstance(x, (str, int, float))][:6]
        pw = spec.get("pairs_with")
        it.pairs_with = pw.strip() if isinstance(pw, str) and pw.strip() and norm(pw) != norm(name) else None
        return it

    def _apply_contract(self, data: dict) -> None:
        specs = [i for i in data.get("items") or [] if isinstance(i, dict) and str(i.get("name") or "").strip()][:MAX_ITEMS]
        items: list[Item] = []
        for spec in specs or [{"name": "item"}]:
            it = self._make_item(len(items), spec)
            twin = next((x for x in items if norm(x.name) == norm(it.name)
                         and sorted(map(req_key, x.reqs + x.excls)) == sorted(map(req_key, it.reqs + it.excls)) and not it.pointers and not x.pointers), None)
            if twin is not None:  # the same product listed twice: one line with both quantities
                twin.quantity = min(50, twin.quantity + it.quantity)
                continue
            items.append(it)
        for hint in data.get("open_properties") or []:  # older contract form: hints for the whole request
            if isinstance(hint, (str, int, float)) and str(hint) not in items[0].open_props:
                items[0].open_props.append(str(hint))
        self.items = items
        self.cur = items[0]
        cap = data.get("price_cap")
        if isinstance(cap, dict):
            self.price_cap = as_number(cap.get("amount"))
            self.cap_total = str(cap.get("applies_to") or "total") != "each"
        elif cap is not None:
            self.price_cap = as_number(cap)
        self.stock_only = str(data.get("stock_rule") or "") == "in_stock_only"
        self.objective = data.get("objective") if data.get("objective") in ("cheapest", "best", "any") else "any"
        self.best_property = data.get("best_property") if isinstance(data.get("best_property"), str) else None
        self.best_direction = "min" if data.get("best_direction") == "min" else "max"
        mb = as_number(data.get("message_budget"))
        self.msg_budget = int(mb) if mb is not None and mb >= 0 else None
        sb = as_number(data.get("step_budget"))
        if sb is not None and 3 <= sb < self.max_steps:
            self.max_steps = int(sb)
        self.procedures = []
        for p in data.get("procedures") or []:
            if isinstance(p, dict) and (p.get("text") or p.get("content")):
                self.procedures.append({"text": str(p.get("text") or ""), "stage": p.get("stage") if p.get("stage") in ("before_search", "before_choose", "before_order", "at_decline", "on_change") else "before_order",
                                        "action": p.get("action") if p.get("action") in ("message", "tool", "none") else "message",
                                        "content": str(p.get("content") or ""), "needs_reply": bool(p.get("needs_reply"))})
        dr = data.get("decline_rule")
        self.decline_rule = dr if isinstance(dr, dict) else {}

    def _merge_recheck(self, check: dict) -> None:
        mb = as_number(check.get("message_budget"))
        if mb is not None and mb >= 0 and self.msg_budget is None:
            self.msg_budget = int(mb)
        sb = as_number(check.get("step_budget"))
        if sb is not None and 3 <= sb < self.max_steps:
            self.max_steps = int(sb)
        procs = check.get("procedures")
        if isinstance(procs, list):
            have = {norm(p["text"])[:40] for p in self.procedures}
            for p in procs:
                if isinstance(p, dict) and (p.get("text") or p.get("content")) and norm(p.get("text", ""))[:40] not in have:
                    if p.get("action") == "none" and p.get("stage") not in ("at_decline",):
                        continue
                    self.procedures.append({"text": str(p.get("text") or ""), "stage": p.get("stage") if p.get("stage") in ("before_search", "before_choose", "before_order", "at_decline", "on_change") else "before_order",
                                            "action": p.get("action") if p.get("action") in ("message", "tool", "none") else "message",
                                            "content": str(p.get("content") or ""), "needs_reply": bool(p.get("needs_reply"))})

    def _fallback_contract(self) -> None:
        """No model: a rough contract from the request text alone."""
        text = self.request
        stop = set("a an the and or but if so to of for in on at by with from as is are be was were i you we they it my your our me "
                   "this that these those not no nor do does did have has had will would can could should shall may might must "
                   "please just only want need like get give find pick something anything one ones some any all every it's i'm im "
                   "y'all yall got gotta ain't aint over under than then there here where when what which who whom how why about into onto "
                   "up down out off new good nice little big few bit lot really very".split())
        # "find me a lamp", "I want some gloves": the noun after an article is usually the product
        name = ""
        for m in re.finditer(r"\b(?:a|an|some|any|the)\s+([a-z][a-z-]{2,})", text.lower()):
            w = m.group(1)
            if w not in stop and not w.endswith("ly") and w not in ("cheap", "cheaper", "cheapest", "pair", "set", "couple", "kind", "sort", "way"):
                name = w
                break
        toks = [w for w in words(text) if w not in stop and len(w) > 2 and not w.isdigit()]
        counts: dict[str, int] = {}
        for i, w in enumerate(toks):
            counts[w] = counts.get(w, 0) + (2 if i < 40 else 1)
        top = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:3]]
        if not name:
            name = " ".join(top[:2]) if top else text[:30]
        item = Item(0, name)
        item.terms = [name] + ([top[0]] if top and top[0] != name else [])
        self.items = [item]
        self.cur = item
        nums = [as_number(m) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]
        nums = [n for n in nums if n is not None and n >= 5]
        self.price_cap = max(nums) if nums else None
        self.msg_budget = 0
        self.objective = "cheapest"

    # ------------------------------------------------------------------ phase 2: roles

    def map_roles(self) -> None:
        user = "TOOLS:\n" + json.dumps(self.tools, ensure_ascii=False)[:12000]
        data = self.brain.ask("judge", ROLES_SYSTEM, user, max_tokens=1500)
        roles = data if isinstance(data, dict) else {}
        self.roles = self._validate_roles(roles)
        missing = [r for r in ("search", "open", "order") if not self.tool(r)]
        if missing:
            self._heuristic_roles(missing)
        self._cart_lines()
        self.diag("roles: " + ", ".join(f"{k}={ (v or {}).get('tool') }" for k, v in self.roles.items() if isinstance(v, dict)))

    def _validate_roles(self, roles: dict) -> dict:
        out: dict = {}
        for role in ("search", "open", "compare", "cart_add", "cart_view", "order", "decline", "message"):
            spec = roles.get(role) if isinstance(roles.get(role), dict) else {}
            name = spec.get("tool")
            if not isinstance(name, str) or name not in self.tool_by_name:
                out[role] = {}
                continue
            props = (self.tool_by_name[name]["parameters"].get("properties") or {})
            clean: dict = {"tool": name}
            for key, val in spec.items():
                if key == "tool":
                    continue
                if key == "args":
                    if isinstance(val, dict):
                        clean["args"] = {k: v for k, v in val.items() if k in props}
                    continue
                if key.endswith("_arg"):
                    clean[key] = val if isinstance(val, str) and val in props else None
                else:
                    clean[key] = val
            out[role] = clean
        alts = roles.get("open_alternatives")
        out["open_alternatives"] = [a for a in alts if isinstance(a, str) and a in self.tool_by_name] if isinstance(alts, list) else []
        self._split_finish(out)
        return out

    def _cart_lines(self) -> None:
        """A cart tool that takes a list of lines (objects holding a listing id and a count): the line's fields, and
        whether the list becomes the whole cart. Read from the schema where the mapping left them out."""
        spec = self.roles.get("cart_add") or {}
        tool = self.tool_by_name.get(spec.get("tool") or "")
        if not tool:
            return
        props = tool["parameters"].get("properties") or {}
        lists = [a for a, p in props.items() if isinstance(p, dict) and p.get("type") == "array" and isinstance(p.get("items"), dict)
                 and (p["items"].get("type") == "object" or p["items"].get("properties"))]
        arg = spec.get("lines_arg") if spec.get("lines_arg") in lists else (lists[0] if lists and not spec.get("id_arg") else None)
        fields = (props[arg]["items"].get("properties") or {}) if arg else {}
        strs = [k for k, p in fields.items() if isinstance(p, dict) and p.get("type") == "string"]
        nums = [k for k, p in fields.items() if isinstance(p, dict) and p.get("type") in ("integer", "number")]
        spaced = lambda k: norm(re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(k)))  # noqa: E731
        idk = spec.get("line_id_key") if spec.get("line_id_key") in strs else next(
            (k for k in strs if re.search(r"\b(id|code|product|listing|item|sku|ref|article|number)\b", spaced(k))), strs[0] if strs else None)
        qk = spec.get("line_qty_key") if spec.get("line_qty_key") in nums else next(
            (k for k in nums if re.search(r"\b(amount|quantity|qty|count|units|number|pieces)\b", spaced(k))), nums[0] if nums else None)
        if not arg or not idk:
            for k in ("lines_arg", "line_id_key", "line_qty_key", "lines_replace"):
                spec.pop(k, None)
            return
        replace = spec.get("lines_replace")
        if not isinstance(replace, bool):
            text = " ".join(spaced(x) for x in (tool.get("description") or "", arg, props[arg].get("description") or ""))
            # a list without an add or remove switch can only take lines out by leaving them out
            replace = bool(re.search(r"\b(replace|replaces|whole|entire|becomes|overwrite|overwrites|exactly|sets|set (?:the|your))\b", text)) or not re.search(
                r"\b(add|adds|append|appends|plus|extra)\b", text)
        spec.update({"lines_arg": arg, "line_id_key": idk, "line_qty_key": qk, "lines_replace": replace, "id_arg": None, "quantity_arg": None})
        self.diag(f"cart lines: {arg} [{idk}, {qk}], the list {'becomes the cart' if replace else 'is added'}")

    def _split_finish(self, roles: dict) -> None:
        """One tool that both orders and declines is told apart by an enum argument."""
        order, decline = roles.get("order") or {}, roles.get("decline") or {}
        if not order.get("tool") or decline.get("tool") != order["tool"]:
            return
        if order.get("args") and decline.get("args") and order["args"] != decline["args"]:
            return
        tool = self.tool_by_name[order["tool"]]
        for aname, aspec in (tool["parameters"].get("properties") or {}).items():
            enum = aspec.get("enum") if isinstance(aspec, dict) else None
            if enum and len(enum) >= 2:
                buy = [e for e in enum if re.search(r"\b(buy|order|check ?out|purchase|pay|confirm|settle|place|complete|commit|submit|finali[sz]e|proceed|accept|execute)", norm(e))]
                quit_ = [e for e in enum if e not in buy and re.search(
                    r"\b(decline|cancel|abandon|give|quit|abstain|stop|none|nothing|leave|empty|exit|without|no|skip|pass|forgo|reject|refuse|drop)\b", norm(e))]
                buy_value = buy[0] if buy else next((e for e in enum if e not in quit_), enum[0])
                quit_value = quit_[0] if quit_ else next((e for e in enum if e != buy_value), enum[-1])
                order["args"] = {aname: buy_value}
                decline["args"] = {aname: quit_value}
                decline["mode_arg"] = aname
                break

    def _heuristic_roles(self, missing: list[str]) -> None:
        """Generic words in tool names and descriptions, as a last resort."""
        patterns = {
            "search": r"search|find|look|browse|query|list products|catalog",
            "open": r"detail|propert|spec|sheet|attribute|full|view|show|listing|single|one article|one item",
            "cart_add": r"cart|bag|basket",
            "cart_view": r"cart|bag|basket",
            "order": r"check ?out|purchase|buy|order|pay|settle|complete",
            "decline": r"decline|give up|abandon|cancel|no purchase|without buying|quit|close out",
            "message": r"message|ask|shopper|customer|question|chat|write|reply",
            "compare": r"compare|several|multiple|batch",
        }
        for role in missing + [r for r in patterns if not self.tool(r)]:
            best, score = None, 0
            for t in self.tools:
                text = norm(t["name"]) + " " + norm(t["description"])
                s = len(re.findall(patterns[role], text))
                props = t["parameters"].get("properties") or {}
                if role == "open" and not any("string" in str(p.get("type")) for p in props.values()):
                    s = 0
                if role in ("order", "decline") and props and all(p.get("type") in ("integer", "number") for p in props.values()):
                    s = 0
                if role == "cart_view" and props:
                    s = 0
                if role == "cart_add" and not props:
                    s = 0
                if s > score:
                    best, score = t, s
            if best is None or not score:
                continue
            props = best["parameters"].get("properties") or {}
            spec: dict = {"tool": best["name"]}
            if role == "search":
                for a, p in props.items():
                    if p.get("type") == "string" and not p.get("enum") and re.search(r"text|query|term|keyword|search|name|word", norm(a)):
                        spec["query_arg"] = a
                    if p.get("type") in ("integer", "number") and re.search(r"page", norm(a)):
                        spec["page_arg"] = a
                    if p.get("type") in ("integer", "number") and re.search(r"price|cost|cap|budget|spend|max", norm(a)):
                        spec["max_price_arg"] = a
                    if p.get("enum") and re.search(r"mode|using|type|kind|by", norm(a)):
                        spec["mode_arg"] = a
                        kw = [e for e in p["enum"] if re.search(r"text|typed|keyword|word|free|query", norm(e))]
                        spec["keyword_mode_value"] = kw[0] if kw else p["enum"][0]
                        spec["filter_mode_value"] = next((e for e in p["enum"] if e != spec["keyword_mode_value"]), None)
                    if p.get("type") == "string" and re.search(r"categor|department|section|type", norm(a)):
                        spec["category_arg"] = a
                    if p.get("type") == "string" and re.search(r"brand|maker|manufactur|vendor", norm(a)):
                        spec["brand_arg"] = a
                if not spec.get("query_arg"):
                    strs = [a for a, p in props.items() if p.get("type") == "string" and not p.get("enum")]
                    if strs:
                        spec["query_arg"] = strs[0]
            elif role in ("open",):
                ids = [a for a, p in props.items() if p.get("type") == "string"]
                spec["id_arg"] = next((a for a in ids if re.search(r"id|sku|code|key|product|item|target", norm(a))), ids[0] if ids else None)
            elif role == "compare":
                arrs = [a for a, p in props.items() if p.get("type") == "array"]
                spec["ids_arg"] = arrs[0] if arrs else None
            elif role == "cart_add":
                for a, p in props.items():
                    if p.get("enum") and len(p["enum"]) >= 2:
                        spec["mode_arg"] = a
                        add = [e for e in p["enum"] if re.search(r"add|include|put|insert|keep", norm(e))]
                        rem = [e for e in p["enum"] if re.search(r"remove|discard|delete|drop|take", norm(e))]
                        spec["add_value"] = add[0] if add else p["enum"][0]
                        spec["remove_value"] = rem[0] if rem else (p["enum"][1] if len(p["enum"]) > 1 else None)
                    elif p.get("type") == "string":
                        spec["id_arg"] = a
                    elif p.get("type") in ("integer", "number"):
                        spec["quantity_arg"] = a
            elif role == "decline":
                strs = [a for a, p in props.items() if p.get("type") == "string" and not p.get("enum")]
                spec["note_arg"] = strs[0] if strs else None
            elif role == "message":
                strs = [a for a, p in props.items() if p.get("type") == "string" and not p.get("enum")]
                spec["text_arg"] = strs[0] if strs else None
            self.roles[role] = spec
        self._split_finish(self.roles)

    # ------------------------------------------------------------------ phase 3: discover

    def discover_first(self) -> None:
        """The plain product query and a few opens per item, so questions to the shopper can name example values."""
        if not self.tool("search"):
            self.diag("discover: no search tool")
            return
        multi = len(self.items) > 1
        for item in self.items:
            self.focus(item)
            terms = self._search_terms()
            # a store's search reads property values too: the noun with the stated values finds fitting rows first
            narrow = next((t for t in terms[1:] if len(words(t)) > len(words(item.name))), terms[0])
            self._search_all_pages(narrow, first_page_only=multi)
            self._open_candidates(limit=2 if multi else 3)
        self.focus(self.items[0])

    def discover_all(self) -> None:
        """Each item in turn gets a fair share of the exploring steps; a device is settled before the part that must fit it."""
        order = self._item_order()
        for i, item in enumerate(order):
            self.focus(item, share=len(order) - i)
            self._pair_requirement(item)
            self.resolve_pointers()
            self.discover_more()
            if any(other.pairs_with and self._find_item(other.pairs_with) is item for other in self.items if other is not item):
                item.pick = self.choose() or self.choose_plausible()  # provisional: the part's search needs the device's model
        self.focus(self.items[0])

    def discover_more(self) -> None:
        """Searches first (a page of rows costs one step; an open costs one step for one listing), then opens by priority."""
        if not self.tool("search"):
            return
        terms = self._search_terms()
        multi = len(self.items) > 1
        if len(terms) > 1:
            terms = terms[1:] + terms[:1]  # the narrow queries first; the bare noun last (its matches are many and unsorted)
        full = 1 if multi else 2  # read every page of the sharpest queries; one page of the rest
        for i, term in enumerate(terms[:4] if multi else terms):
            if not self.exploring_ok() or self.searches >= self.search_cap():
                break
            if len(self._fitting()) >= 2 and self.objective == "any":
                break
            self._search_all_pages(term, first_page_only=(i >= full))
        self._open_candidates()
        if not self._fitting():
            self._widen()
            self._open_candidates()
        if self.objective == "cheapest":
            self._cheaper_walk(terms)

    def _widen(self) -> None:
        """Nothing fits yet: run the queries not run so far, then lower price windows of the plain query."""
        terms = self._search_terms()
        for term in terms:
            if self._fitting() or not self.exploring_ok() or self.searches >= self.search_cap():
                break
            if self._search_all_pages(term, first_page_only=True):
                self._open_candidates(limit=4)
        if not self._fitting():
            self._price_windows(terms[0])

    def _price_windows(self, term: str) -> None:
        """A broad query shows only its first page or three. Lower price caps show other rows of the same query."""
        cap_arg = self.role_arg("search", "max_price_arg")
        if not cap_arg or self.search_exhausted or self._fitting():
            return
        prices = sorted(c.price for c in self.cands.values() if c.price is not None and self._in_scope(c))
        if len(prices) < 8:
            return
        for q in (0.4, 0.15):
            if self._fitting() or not self.exploring_ok() or self.searches >= self.search_cap():
                return
            cap = prices[int(len(prices) * q)]
            below = cap - (1 if float(cap).is_integer() else 0.01)
            if below <= 0:
                return
            if self._bounded_search(term, below, first_page_only=True) == 0:
                return
            self._open_candidates(limit=4)

    def _bounded_search(self, term: str, max_price: float, first_page_only: bool = True) -> int:
        """Rows under a price: keyword search when it honours the cap, else the store's filter search by department."""
        cat_arg = self.role_arg("search", "category_arg")
        if not self.role_arg("search", "max_price_arg"):
            return 0  # the store cannot bound a search by price
        filter_mode = (self.roles.get("search") or {}).get("filter_mode_value") is not None or not self.role_arg("search", "mode_arg")
        if self.keyword_ignores_cap and cat_arg and filter_mode:
            home, level = self.home_department()
            if home:
                price = int(max_price) if float(max_price).is_integer() else round(max_price, 2)
                return self._search_all_pages(home, max_price=price, filters={cat_arg: home}, first_page_only=False)
        return self._search_all_pages(term, max_price=max_price, first_page_only=first_page_only)

    def _search_terms(self) -> list[str]:
        """The model's queries, then code-made ones: noun + each stated value, with number-unit forms joined ("1 l" -> "1l")."""
        terms = list(self.cur.terms)
        name = self.cur.name.strip()
        values = [v for v in (self._search_value(r) for r in self.reqs) if v]
        values = [v for i, v in enumerate(values) if norm(v) not in [norm(x) for x in values[:i]]]
        if name:
            terms.insert(0, name)
            if len(values) >= 2 and len(words(name + " " + " ".join(values))) <= 6:
                terms.insert(1, name + " " + " ".join(values))  # all stated values at once: the narrowest query
            home, level = self.home_department() if self.cands else ("", 0)
            dept = " ".join(w for w in norm(home).split() if w not in ("and",))
            if home and norm(dept) != norm(name) and values:
                # a store's search reads department names too: the stated value with the department finds that product's rows
                for v in reversed(values[:2]):
                    if len(words(f"{v} {dept}")) <= 6:
                        terms.insert(1, f"{v} {dept}")
        for r in self.reqs:
            val = self._search_value(r)
            if val and len(val) <= 30 and name:
                if r.get("pair_code"):
                    terms.insert(1, f"{name} {val}")  # the model family is the sharpest query a part has
                    continue
                terms.append(f"{name} {val}")
                joined = re.sub(r"(\d)\s+([a-zA-Z]{1,3})\b", r"\1\2", val)
                if joined != val:
                    terms.append(f"{name} {joined}")
        seen, out = set(), []
        for t in terms:
            key = norm(t)
            if key and key not in seen:
                seen.add(key)
                out.append(t)
        return out[:6] or [self.request[:40]]

    @staticmethod
    def _search_value(r: dict) -> str:
        """A requirement's value as search words: comparison words dropped ("at least 12 gb" -> "12 gb"), yes/no as the property."""
        if r.get("property") == "product type" or not r.get("value"):
            return ""
        if r.get("pair_code"):
            return code_family(r["pair_code"])  # "headboard SIR21xx": parts name the model family they fit
        val = str(r["value"]).strip()
        if norm(val) in ("yes", "true", "no", "false"):
            return str(r.get("property") or "").strip()  # "organic: yes" searches as "organic"
        val = re.sub(r"(?i)\b(at least|at most|or less|or more|or above|or below|less than|more than|no more than|no less than|"
                     r"up to|under|over|above|below|minimum|maximum|max|min|within|either way of|of)\b", " ", val)
        return " ".join(val.split())

    def _search_args(self, term: str, page: int | None = None, max_price: float | None = None, filters: dict | None = None) -> dict:
        args: dict = {}
        qa, pa, ma, mode = (self.role_arg("search", k) for k in ("query_arg", "page_arg", "max_price_arg", "mode_arg"))
        if filters:
            args.update(filters)
            if mode and self.roles["search"].get("filter_mode_value"):
                args[mode] = self.roles["search"]["filter_mode_value"]
        else:
            if qa:
                args[qa] = term
            if mode and self.roles["search"].get("keyword_mode_value"):
                args[mode] = self.roles["search"]["keyword_mode_value"]
        if pa and page is not None:
            args[pa] = page
        if ma and max_price is not None:
            args[ma] = max_price
        return args

    def _search_all_pages(self, term: str, max_price: float | None = None, filters: dict | None = None, first_page_only: bool = False) -> int:
        """Run one query through its pages (a refinement query pages on only when its matches are few). Returns the new candidates."""
        new = 0
        page = 1
        page_val = None  # the first page is asked for without a paging value: some stores page by row offset
        seen_total = 0
        pa = self.role_arg("search", "page_arg")
        cap_arg = self.role_arg("search", "max_price_arg")
        explicit = max_price is not None and bool(cap_arg)  # a store with no price argument gets the same query either way
        if max_price is None and cap_arg and self.price_cap is not None:
            max_price = self.line_budget()
            if max_price is not None and float(max_price).is_integer():
                max_price = int(max_price)
        # a query under the running budget counts as the same query when the budget moves a little
        signature = (norm(term), max_price if explicit else "budget", json.dumps(filters or {}, sort_keys=True))
        if signature in self.queried:
            return 0
        self.queried.add(signature)
        page_spec = ((self.tool("search") or {}).get("parameters", {}).get("properties") or {}).get(pa or "", {})
        max_val = as_number(page_spec.get("maximum")) if isinstance(page_spec, dict) else None
        max_page = int(max_val) if max_val is not None and 1 <= max_val <= 10 else 5
        while self.exploring_ok() and self.searches < self.search_cap():
            self.searches += 1
            obs = self.call("search", self._search_args(term, page_val if pa else None, max_price, filters), f"search {clip(term, 40)} p{page}")
            if obs is None:
                break
            rows, meta = self._rows(self.tool("search")["name"], obs)
            if not rows:
                if page == 1:
                    self._learn_search_kind(term, 0)
                break
            if page == 1:
                self._learn_search_kind(term, len(rows))
                if self.page_size is None or len(rows) > self.page_size:
                    self.page_size = len(rows)
            hit_values = {norm(v) for v in (self._search_value(q) for q in self.reqs) if v and self._stem_in(v, term)} if not filters else set()
            for r in rows:
                if self._register(r):
                    new += 1
                if hit_values:
                    cid = self._row_id(self.fields.get(self.tool("search")["name"]) or self._guess_fields(r), r)
                    if cid in self.cands:
                        self.cands[cid].found_with |= hit_values
            if max_price is not None and not filters and not self.keyword_ignores_cap:
                fm = self.fields.get(self.tool("search")["name"]) or {}
                over = [as_number(r.get(fm.get("price_key") or "")) for r in rows]
                if any(p is not None and p > max_price + 1e-9 for p in over):
                    self.keyword_ignores_cap = True  # this store applies a price cap only in its filter search
                    self.diag("keyword search ignores the price cap; price-bounded searches use the department filter")
            seen_total += len(rows)
            total = meta.get("total")
            nxt = meta.get("next_page")
            if "next_page" in meta:  # the store says whether a next page exists: believe it
                more = nxt not in (None, False, 0, "")
            else:
                more = total is not None and total > seen_total
            small = total is not None and total <= max_page * max(len(rows), 1)
            if pa and more and page < max_page and (self.page_size is None or len(rows) >= self.page_size) and (not first_page_only or small):
                page += 1
                given = as_number(nxt) if not isinstance(nxt, bool) else None
                page_val = int(given) if given is not None and given > 0 else page  # the store's own next value: a page or an offset
                if max_val is not None and page_val > max_val:
                    break
                continue
            # exhausted: every match was shown (no further page and no count beyond what we saw)
            self.search_exhausted = not more or (total is not None and total <= seen_total)
            if total is not None and total > seen_total:
                self.search_exhausted = False
            break
        return new

    def _learn_search_kind(self, term: str, hits: int) -> None:
        if self.search_kind is None and len(words(term)) >= 2:
            self.search_kind = "any" if hits >= (self.page_size or 7) else "all"

    def _rows(self, tool_name: str, obs: Any) -> tuple[list[dict], dict]:
        """Listing rows and paging meta from a search observation."""
        fm = self._field_map(tool_name, obs)
        rows: list[Any] = []
        lp = fm.get("list_path") or []
        lst = get_path(obs, lp) if lp else None
        if isinstance(lst, list):
            rows = [r for r in lst if isinstance(r, dict)]
        if not rows:
            for _, v in deep_find(obs, lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(y, dict) for y in x)):
                if len(v) > len(rows):
                    rows = v
        meta: dict = {"seen": 0}
        if isinstance(obs, dict):
            if fm.get("next_page_key") in obs:
                meta["next_page"] = obs[fm["next_page_key"]]
            if fm.get("total_key") in obs:
                meta["total"] = as_number(obs[fm["total_key"]])
        for r in rows:
            cid = self._row_id(fm, r)
            if cid:
                meta["seen"] += 1
        meta["seen"] = len(rows)
        if meta.get("total") is not None:
            meta["total"] = int(meta["total"])
        return rows, meta

    def _field_map(self, tool_name: str, obs: Any) -> dict:
        if tool_name in self.fields:
            return self.fields[tool_name]
        self.field_tries[tool_name] = self.field_tries.get(tool_name, 0) + 1
        fm = self._guess_fields(obs)
        if self.brain.available():
            sample = _shrink(obs, OBS_CAP)
            data = self.brain.ask("judge", FIELDS_SYSTEM, "RESULT:\n" + json.dumps(sample, ensure_ascii=False, default=str), max_tokens=900)
            if isinstance(data, dict):
                rows = get_path(obs, data.get("list_path") or []) if isinstance(data.get("list_path"), list) else None
                probe = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else (obs if isinstance(obs, dict) else {})
                for key in ("id_key", "title_key", "price_key", "stock_key", "category_key", "props_key"):
                    v = data.get(key)
                    if isinstance(v, str) and v in probe:
                        fm[key] = v
                if isinstance(data.get("list_path"), list) and isinstance(rows, list):
                    fm["list_path"] = data["list_path"]
                for key in ("next_page_key", "total_key"):
                    v = data.get(key)
                    if isinstance(v, str) and isinstance(obs, dict) and v in obs:
                        fm[key] = v
                for n in data.get("notes") or []:
                    if isinstance(n, str) and len(n) > 12:
                        self.notices.append(n)
        useful = sum(1 for k in ("id_key", "title_key", "props_key", "price_key") if fm.get(k))
        if useful >= 2 or self.field_tries[tool_name] >= 3:  # a thin map (empty or odd first result) is retried next time
            self.fields[tool_name] = fm
        return fm

    @staticmethod
    def _guess_fields(obs: Any) -> dict:
        fm: dict = {"list_path": []}
        probe: dict = {}
        if isinstance(obs, dict):
            lists = [(p, v) for p, v in deep_find(obs, lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(y, dict) for y in x))]
            if lists:
                p, v = max(lists, key=lambda pv: len(pv[1]))
                fm["list_path"] = list(p)
                probe = v[0]
            else:
                probe = obs
            for k, v in obs.items():
                nk = norm(k)
                if re.search(r"next|continue|more|cursor", nk) and not isinstance(v, (dict, list)):
                    fm["next_page_key"] = k
                if re.search(r"total|count|matched|number|hits", nk) and isinstance(v, (int, float)) and not isinstance(v, bool):
                    fm["total_key"] = k
        elif isinstance(obs, list) and obs and isinstance(obs[0], dict):
            probe = obs[0]
        for k, v in probe.items():
            nk = norm(k)
            if isinstance(v, dict) and v and all(isinstance(x, (str, int, float, bool)) for x in v.values()):
                fm.setdefault("props_key", k)
            elif isinstance(v, bool) or (isinstance(v, str) and as_bool(v) is not None and re.search(r"stock|avail|hand", nk)):
                if re.search(r"stock|avail|hand", nk):
                    fm.setdefault("stock_key", k)
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and re.search(r"price|cost|amount|cts|cents|usd|total", nk):
                fm.setdefault("price_key", k)
            elif isinstance(v, str):
                if re.search(r"\bid\b|pid|sku|code|identifier", nk) or re.fullmatch(r"[a-z]*-?[0-9a-f]{6,}", v.strip().lower()):
                    fm.setdefault("id_key", k)
                elif re.search(r"title|name|label", nk):
                    fm.setdefault("title_key", k)
        if "title_key" not in fm:
            strs = [(k, v) for k, v in probe.items() if isinstance(v, str) and k != fm.get("id_key")]
            if strs:
                fm["title_key"] = max(strs, key=lambda kv: len(kv[1]))[0]
        return fm

    def _row_id(self, fm: dict, row: dict) -> str | None:
        v = row.get(fm.get("id_key") or "")
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v)
        for k, x in row.items():
            if isinstance(x, str) and re.fullmatch(r"[a-z]*-?[0-9a-f]{6,}", x.strip().lower()):
                return x
        return None

    def _register(self, row: dict, fm: dict | None = None, opened: bool = False, tag: bool = True) -> bool:
        fm = fm or self.fields.get((self.tool("search") or {}).get("name", ""), {}) or self._guess_fields(row)
        cid = self._row_id(fm, row)
        if not cid:
            return False
        new = cid not in self.cands
        c = self.cands.get(cid) or Candidate(cid, row)
        if new:
            self.seen_counter += 1
            c.seen_order = self.seen_counter
            self.cands[cid] = c
        if tag:
            c.items.add(self.cur.idx)
        self._fill(c, row, fm, opened)
        return new

    def _fill(self, c: Candidate, row: dict, fm: dict, opened: bool) -> None:
        t = row.get(fm.get("title_key") or "")
        if isinstance(t, str) and t.strip():
            c.title = t.strip()
        p = as_number(row.get(fm.get("price_key") or ""))
        if p is not None:
            c.price = p
        s = row.get(fm.get("stock_key") or "")
        b = as_bool(s)
        if b is not None:
            c.stock = b
        dept = row.get(fm.get("category_key") or "")
        if isinstance(dept, str) and dept.strip():
            c.dept = dept.strip()
        props = row.get(fm.get("props_key") or "")
        if isinstance(props, dict):
            c.props.update({str(k): v for k, v in props.items() if isinstance(v, (str, int, float, bool))})
        # loose extras: short scalar fields other than the known ones; free text is kept apart for pairing and pointers
        for k, v in row.items():
            if k in (fm.get("title_key"), fm.get("price_key"), fm.get("stock_key"), fm.get("props_key"), fm.get("id_key")):
                continue
            nk = norm(k)
            if isinstance(v, str) and re.search(r"seller|shop|store|vendor|merchant|sold by", nk) and len(v) <= 60:
                c.seller = v.strip()
            text_key = re.search(r"descr|blurb|summary|text|fine|print|note|about", nk)
            long_text = isinstance(v, str) and (len(v) > 60 or len(v.split()) > 8)  # a department name is short; a description is not
            if isinstance(v, str) and v.strip() and (text_key or long_text) and not re.search(r"url|image|photo|link|title|name|\bid\b|sku", nk):
                if v.strip()[:80] not in c.text:
                    c.text = (c.text + " | " + v.strip()).strip(" |")[:2500]
                continue
            if re.search(r"descr|blurb|summary|text|url|image|photo|link|title|name|\bid\b|sku|fine|print|note", nk):
                continue
            if isinstance(v, (str, int, float, bool)) and len(str(v)) <= 60 and len(str(v).split()) <= 8:
                c.props.setdefault(str(k), v)
        if opened:
            c.opened = True
            c.verdicts = {}  # fresh properties: judge again
            c.deferred = set()

    def _in_scope(self, c: Candidate) -> bool:
        """A listing is weighed for the current item when that item's searches returned it or its title names the item."""
        return self.cur.idx in c.items or self._names_product(c)

    def line_budget(self) -> float | None:
        """The most one unit of the current item can cost: a cap per unit as stated; a total cap less the cheapest price
        seen so far for every other item (an optimistic estimate, so no listing that could fit is cut)."""
        if self.price_cap is None:
            return None
        if not self.cap_total:
            return self.price_cap
        others = 0.0
        for it in self.items:
            if it is self.cur:
                continue
            seen = [c.price for c in self.cands.values() if c.price is not None and it.idx in c.items]
            if seen:
                others += min(seen) * it.quantity
        return max(0.0, self.price_cap - others) / max(1, self.quantity)

    def _row_ok(self, c: Candidate) -> bool:
        """Not ruled out by the row alone: price under the cap, stock if stated, not turned down for this item."""
        if c.id in self.cur.rejected:
            return False
        if self.ceiling is not None and c.price is not None and c.price > self.ceiling + 1e-9:
            return False
        if c.price is not None and self.price_cap is not None:
            each = self.price_cap / self.quantity if self.cap_total else self.price_cap
            if c.price > each + 1e-9:
                return False
        if self.stock_only and c.stock is False:
            return False
        return not c.ruled_out(self.reqs, self.excls)

    def _title_score(self, c: Candidate) -> int:
        """How many stated values the row's title (or its row-level fields) already shows; word endings may differ
        ("long handle" in "Long Handled"). Used only to order opens, never as evidence."""
        title = norm(c.title) + " " + " ".join(norm(v) for v in c.props.values())
        score = 0
        for r in self.reqs:
            if r.get("property") == "product type" or r.get("pair_code"):
                continue
            value = self._search_value(r) or norm(r["value"])
            if self._stem_in(value, title) or norm(value) in c.found_with:
                score += 1  # the title shows it, or the store's own search matched it on this row
        for r in self.excls:
            if self._stem_in(norm(r["value"]), norm(c.title)):
                score -= 2
        return score

    @staticmethod
    def _stem_in(phrase: str, hay: str) -> bool:
        """Every word of the phrase is in the text, allowing a different ending (handle / handled, strap / straps)."""
        ws = norm(phrase).split()
        if not ws:
            return False
        have = set(norm(hay).split())
        stems = {w[: max(4, len(w) - 2)] for w in have if len(w) > 3}
        plural = {w + "s" for w in ws} | {w + "es" for w in ws}
        return all(w in have or w + "s" in have or w + "es" in have or (len(w) > 3 and w[: max(4, len(w) - 2)] in stems)
                   or any(h.startswith(w) and len(h) - len(w) <= 3 for h in have if len(w) > 3) for w in ws) or bool(plural & have and len(ws) == 1)

    @staticmethod
    def _singular(w: str) -> str:
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith(("ches", "shes", "xes", "sses")):
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            return w[:-1]
        return w

    @staticmethod
    def _dept_of(c: Candidate) -> str:
        return c.dept or next((str(v) for k, v in c.props.items() if isinstance(v, str) and re.search(r"categor|department|section|aisle", norm(k))), "")

    def home_department(self) -> tuple[str, int]:
        """The department that is the current product: ("Bags & Wallets", 2) for "bags or wallets", ("Wallets", 2) for "wallets".
        2: the department's name and the shopper's words correspond; 1: the department's name holds the product; 0: none known."""
        rows = [c for c in self.cands.values() if self.cur.idx in c.items]
        cached = self.cur.dept_cache
        if cached and cached[0] == len(rows):
            return cached[1], cached[2]
        alts = [[self._singular(w) for w in a.split() if len(w) > 2] for a in self._alternatives(self.cur.name)]
        name_words = {w for a in alts for w in a}
        scores: dict[str, tuple[int, int]] = {}
        for c in rows:
            dept = self._dept_of(c)
            if not dept:
                continue
            dw = {self._singular(w) for w in norm(dept).split() if len(w) > 2 and w != "and"}
            if not dw or not all(a and set(a) <= dw for a in alts) and not any(a and set(a) <= dw for a in alts):
                continue
            exact = all(a and set(a) <= dw for a in alts) and dw <= name_words
            score = 2 if exact else 1
            seen = scores.get(dept, (0, 0))
            scores[dept] = (score, seen[1] + 1)
        best = sorted(scores.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
        home, level = "", 0
        if best and (best[0][1][0] == 2 or len([b for b in best if b[1][0] == best[0][1][0]]) == 1):
            home, level = best[0][0], best[0][1][0]
        self.cur.dept_cache = (len(rows), home, level)
        return home, level

    def _dept_names_product(self, c: Candidate) -> bool:
        """The row's department or category names the product ("Lamps" for "lamp")."""
        noun = next((r["value"] for r in self.reqs if r.get("property") == "product type"), "")
        dept = norm(self._dept_of(c))
        return bool(dept) and any(self._stem_in(alt, dept) for alt in self._alternatives(noun))

    @staticmethod
    def _alternatives(noun: str) -> list[str]:
        """"rucksack or backpack" is two ways to name one product."""
        return [a.strip() for a in re.split(r"\bor\b", norm(noun)) if a.strip()] or [norm(noun)]

    def _names_product(self, c: Candidate) -> bool:
        """The row's title or its department carries the product noun ("kettle" in "Electric Kettle (Cordless)")."""
        noun = next((r["value"] for r in self.reqs if r.get("property") == "product type"), "")
        ws = [w for w in words(noun) if len(w) > 2]
        if not ws:
            return True
        hay = norm(c.title) + " " + norm(self._dept_of(c))
        return any(self._word_in(w, hay) or self._word_in(w.rstrip("s"), hay) or self._word_in(w + "s", hay) for w in ws)

    def _open_order(self) -> list[Candidate]:
        """Rows naming the product and carrying the stated values first, then cheaper first. Ahead of all, a row titled
        like a proven listing that sold out or was turned down: most likely the same product from another seller."""
        gone = {norm(c.title) for c in self.cands.values()
                if c.title and c.opened and (c.id in self.cur.rejected or c.stock is False) and c.fits(self.reqs, self.excls)}
        pool = [c for c in self.cands.values() if not c.opened and self._in_scope(c) and self._row_ok(c)]
        pool.sort(key=lambda c: (0 if gone and norm(c.title) in gone else 1, 0 if self._dept_names_product(c) else 1 if self._names_product(c) else 2,
                                 -self._title_score(c), c.price if c.price is not None else 1e12, c.seen_order))
        return pool

    def _open_candidates(self, limit: int | None = None) -> None:
        """Open listings in priority order until enough fitting candidates are proven."""
        if not self.tool("open") and not self.roles.get("open_alternatives"):
            self._judge_pending()
            return
        start = self.opens
        limit = self.open_cap() if limit is None else limit
        while self.exploring_ok() and self.opens < self.open_cap() and self.opens - start < limit:
            fitting = self._fitting()
            strong = [c for c in fitting if c.strong(self.reqs, self.excls)]
            pool = self._open_order()
            if not pool:
                break
            if fitting:
                basis = strong or fitting
                # a pick can sell out at the finish; a second proven listing makes the swap cheap
                backup_needed = len(basis) < 2
                if self.cur.first_fit_open is None:
                    self.cur.first_fit_open = self.opens
                cheapest_fit = min((c.price for c in basis if c.price is not None), default=None)
                if self.objective == "cheapest" and cheapest_fit is not None:
                    cheaper = [c for c in pool if c.price is None or c.price < cheapest_fit]
                    if cheaper:
                        pool = cheaper
                    elif not backup_needed or self.opens - self.cur.first_fit_open >= 5:
                        break
                    else:  # the backup is the next cheapest row naming the product: it is what a sold-out pick is swapped for
                        pool.sort(key=lambda c: (0 if self._dept_names_product(c) or self._names_product(c) else 1,
                                                 c.price if c.price is not None else 1e12, -self._title_score(c), c.seen_order))
                elif self.objective == "any":
                    if not backup_needed or self.opens - self.cur.first_fit_open >= 5:
                        break
                elif self.objective == "best" and len(strong) >= 4:
                    break
            batch = pool[:3]
            if self.tool("compare") and len(batch) > 1:
                if not self._open_many(batch):
                    self._open_one(batch[0])
            else:
                self._open_one(batch[0])
            self._judge_pending()

    def _open_one(self, c: Candidate, force: bool = False) -> bool:
        """Open one listing. `force` opens it after the exploring has ended (the finish's own steps are checked by the caller)."""
        tool = self.tool("open")
        id_arg = self.role_arg("open", "id_arg")
        names = ([tool["name"]] if tool else []) + [a for a in self.roles.get("open_alternatives", []) if not tool or a != tool["name"]]
        for name in names:
            t = self.tool_by_name.get(name)
            if not t or not (force or self.exploring_ok()):
                return False
            props = t["parameters"].get("properties") or {}
            arg = id_arg if (name == (tool or {}).get("name") and id_arg in props) else next((a for a, p in props.items() if p.get("type") == "string"), None)
            if not arg:
                continue
            self.opens += 1
            obs = self.wire.act(name, fit_args(t["parameters"], {arg: c.id}), f"open {c.id}")
            if obs is None:
                return False
            obs = self.absorb(name, obs)
            if not isinstance(obs, dict) or not obs or "error" in obs:
                continue
            fm = self._field_map(name, obs)
            row = get_path(obs, fm.get("list_path") or []) if fm.get("list_path") else obs
            if isinstance(row, list):
                row = row[0] if row and isinstance(row[0], dict) else {}
            if not isinstance(row, dict):
                continue
            before = len(c.props)
            self._fill(c, row, fm, opened=True)
            if len(c.props) > before or fm.get("props_key"):
                # this tool reveals properties: make it the open tool
                if not tool or name != tool["name"]:
                    self.roles["open"] = {"tool": name, "id_arg": arg}
                return True
            # the tool returned a summary without properties; try the next one
            tool = None
        return c.opened

    def _open_many(self, batch: list[Candidate]) -> bool:
        tool = self.tool("compare")
        ids_arg = self.role_arg("compare", "ids_arg")
        if not tool or not ids_arg:
            return False
        spec = (tool["parameters"].get("properties") or {}).get(ids_arg, {})
        cap = int(spec.get("maxItems", 3)) if isinstance(spec, dict) else 3
        batch = batch[: max(1, min(cap, 3))]
        self.opens += 1
        obs = self.call("compare", {ids_arg: [c.id for c in batch]}, "compare")
        if not isinstance(obs, (dict, list)) or (isinstance(obs, dict) and "error" in obs):
            return False
        fm = self._field_map(tool["name"], obs)
        rows: list = []
        lp = fm.get("list_path") or []
        lst = get_path(obs, lp) if lp else obs
        if isinstance(lst, list):
            rows = [r for r in lst if isinstance(r, dict)]
        elif isinstance(lst, dict):
            rows = [v for v in lst.values() if isinstance(v, dict)] or [lst]
        matched = 0
        for r in rows:
            cid = self._row_id(fm, r)
            c = self.cands.get(cid or "")
            if c is None:
                for b in batch:
                    if b.id in json.dumps(r, default=str):
                        c = b
                        break
            if c is not None:
                self._fill(c, r, fm, opened=True)
                matched += 1
        return matched > 0

    # ------------------------------------------------------------------ verification

    def _judge_pending(self) -> None:
        """Code decides the clear requirement cells; a model judges the rest, in batches."""
        pending = []
        for c in self.cands.values():
            if not c.opened or not self._in_scope(c) or not self._row_ok(c):
                continue
            self._code_verdicts(c)
            if c.undecided(self.reqs, self.excls) and not c.ruled_out(self.reqs, self.excls):
                pending.append(c)
        if not pending:
            return
        pending.sort(key=lambda c: (0 if self._names_product(c) else 1, c.price if c.price is not None else 1e12, c.seen_order))
        for start in range(0, min(len(pending), 12), 4):
            batch = pending[start : start + 4]
            if not self.brain.available():
                break
            self._model_verdicts(batch)
        for c in pending:  # whatever stayed open is unclear (an exclusion that cannot be shown is clear)
            for r in self.reqs:
                c.verdicts.setdefault(req_key(r), "yes" if req_key(r) in c.deferred else "unclear")  # no model looked: the property stands
            for e in self.excls:
                c.verdicts.setdefault(req_key(e), "no")

    def _code_verdicts(self, c: Candidate) -> None:
        prop_text = {norm(humanize(k)): norm(v) for k, v in c.props.items()}
        title = norm(c.title)
        for r in self.reqs + self.excls:
            key = req_key(r)
            if c.verdicts.get(key) is not None:
                continue
            want = norm(r["value"])
            prop = norm(r.get("property") or "")
            if not want:
                c.verdicts[key] = "yes" if r.get("kind") != "exclude" else "no"
                continue
            if prop == "product type":
                # the product noun in the title, or a department that holds one kind of product ("Lamps", not "Bags & Wallets");
                # anything less clear goes to the model
                raw_dept = self._dept_of(c)
                mixed = re.search(r"&|,|\band\b|\+", raw_dept.lower()) is not None
                dept = norm(raw_dept)
                if not raw_dept:  # no field is known to name the department: any field may carry the product's name
                    dept, mixed = " ".join(prop_text.values()), False

                def names(alt: str, hay: str) -> bool:
                    ws = [w for w in alt.split() if len(w) > 2]
                    return bool(ws) and all(self._word_in(w, hay) or self._word_in(w.rstrip("s"), hay) or self._word_in(w + "s", hay) for w in ws)

                alts = self._alternatives(r["value"])
                home, level = self.home_department() if norm(r["value"]) == norm(self.cur.name) else ("", 0)
                if home and raw_dept:
                    if norm(raw_dept) == norm(home):
                        c.verdicts[key] = "yes"
                        continue
                    if level == 2:
                        c.verdicts[key] = "no"  # the store files this product under another department
                        continue
                if any(names(a, title) for a in alts) or (not mixed and any(names(a, dept) for a in alts)):
                    # "pillow" is in "pillow cover": when another item's name holds this one, the model decides
                    longer = [norm(x.name) for x in self.items if norm(x.name) != want and want in norm(x.name)]
                    if not any(self._word_in(n, title) for n in longer):
                        c.verdicts[key] = "yes"
                continue
            if r.get("pair_code"):
                said = works_with(c.text + " " + " ".join(str(v) for v in c.props.values()), r["pair_code"])
                if said is not None:
                    c.verdicts[key] = "yes" if said else "no"
                continue
            if r.get("named", True):
                matched_prop = [k for k in prop_text if prop and self._same_property(prop, k)]
            else:
                matched_prop = [k for k in prop_text if self._same_property(prop, k) or self._related_property(prop, want, k)]
            vals = [prop_text[k] for k in matched_prop]
            if any(v == want or self._word_in(want, v) for v in vals):
                if self._overruled(r, prop, want, matched_prop, prop_text):
                    c.verdicts[key] = "no"
                elif r.get("kind") == "exclude" or not self.brain.available() or not (self._title_may_differ(want, title) or self._title_names_other(prop, want, title)):
                    c.verdicts[key] = "yes"  # for an exclusion: it has the value
                else:  # the title may state another value for this property (other figures, or other values of it): the model decides
                    c.deferred.add(key)
            elif r.get("kind") == "exclude" and any(self._word_in(want, v) for v in prop_text.values()):
                c.verdicts[key] = "yes"  # a ruled-out value under any name ("hood: detachable hood" for "no hood")
            elif not matched_prop and self._word_in(want, title):
                c.verdicts[key] = "title"  # the title says so but no property does: weak evidence
            # a stated but different value is left to the model: synonyms are possible

    @staticmethod
    def _title_may_differ(want: str, title: str) -> bool:
        """A figure in the title that is not part of a wanted value with figures ("1080p/4K" or "6/8/10 cup" beside a
        resolution or a capacity): the title may state another value for the property, and a listing that contradicts itself
        does not prove it, so a matching property alone does not settle the requirement."""
        if not re.search(r"\d", want):
            return False
        own = set(re.findall(r"\d+(?:\.\d+)?", want))
        return any(n not in own for n in re.findall(r"\d+(?:\.\d+)?", title))

    def _title_names_other(self, prop: str, want: str, title: str) -> bool:
        """A wanted value without figures that the title does not name, where the title names other values this property
        takes on the item's listings ("Waterproof ... Lightweight" for a feature of "foldable"): the title states
        another value instead, the listing contradicts itself, and the property alone does not settle it. A title naming
        the wanted value among others ("Waterproof Foldable Lightweight") is no contradiction. Answer keys leave such a
        listing out even when its property states the wanted value."""
        if not prop or re.search(r"\d", want):
            return False
        key_words = sorted((w for w in norm(want).split() if len(w) > 3), key=len, reverse=True)
        if not key_words or self._word_in(key_words[0], title) or self._word_in(key_words[0].rstrip("s"), title):
            return False
        others = set()
        for c in self.cands.values():
            if not c.opened:
                continue
            for k, v in c.props.items():
                if self._same_property(prop, k):
                    for part in re.split(r"[,;/|]", str(v)):
                        part = norm(part)
                        if len(part) > 3 and part != norm(want) and not self._word_in(key_words[0], part):
                            others.add(part)
        return any(self._word_in(o, title) for o in others)

    def _overruled(self, r: dict, prop: str, want: str, matched: list[str], prop_text: dict) -> bool:
        """The value shows only under a variant name ("frame material: oak") while the property under its own name states
        another ("material: pine"): the own name decides. A name for the product itself ("lamp style: arc" beside
        "style: modern") describes the product and stands."""
        if r.get("kind") in ("exclude", "soft"):
            return False
        has = lambda v: v == want or self._word_in(want, v)
        own = [k for k in matched if plain_property(k) == plain_property(prop)]
        if not own or any(has(prop_text[k]) for k in own):
            return False
        noun = {self._singular(w) for w in words(self.cur.name)}
        base = {self._singular(w) for w in words(plain_property(prop))}
        return not any(has(prop_text[k]) and ({self._singular(w) for w in words(plain_property(k))} - base) <= noun
                       for k in matched if k not in own)

    def _named(self, r: dict) -> dict:
        """Mark whether the shopper named the property ("the type of lamp is a floor lamp") or gave only a value ("a lamp
        with a long arm"): a named property is that property; for a bare value any related property stating it will do."""
        stop = {"of", "the", "a", "an", "and", "for", "in", "on", "to", "with"}
        prop = {self._singular(w) for w in norm(r.get("property") or "").split() if w not in stop}
        said = {self._singular(w) for w in norm(r.get("text") or "").split()}
        r["named"] = bool(prop) and prop <= said if r.get("text") else True
        return r

    def _related_property(self, want: str, value: str, key: str) -> bool:
        """For a bare value: the key shares a word stem with the property the reader chose or with the value itself
        ("handle style", "bag handlestyle" for "handle length: long handle")."""
        generic = {"type", "style", "kind", "class", "spec", "feature", "detail", "info", "other", "general"}

        def stems(text: str) -> set:
            return {w[:4] for w in norm(text).split() if len(w) >= 3 and w not in generic}

        k = stems(key)
        return bool(k & (stems(want) | stems(value))) or any(s in w for s in stems(want) | stems(value) for w in norm(key).split() if len(s) >= 4)

    def _same_property(self, want: str, key: str) -> bool:
        """One property name holds the other's words ("frame material" for "material", "lamp type" or "type" for
        "type of lamp"). Names that only share a word ("desk lamp style" for "type of lamp") are different properties."""
        stop = {"of", "the", "a", "an", "and", "for", "in", "on", "to", "with"}
        a = {self._singular(w) for w in norm(want).split() if w not in stop}
        b = {self._singular(w) for w in norm(key).split() if w not in stop}
        return bool(a) and bool(b) and (a <= b or b <= a)

    @staticmethod
    def _word_in(needle: str, hay: str) -> bool:
        if not needle or not hay:
            return False
        if re.search(r"(?<![0-9a-z])" + re.escape(needle) + r"(?![0-9a-z])", hay) is not None:
            return True
        if re.search(r"\d", needle):  # "1 l" and "1l", "1920 x 1080" and "1920x1080" are the same value
            tight = needle.replace(" ", "")
            return re.search(r"(?<![0-9a-z])" + re.escape(tight) + r"(?![0-9a-z])", hay.replace(" ", "")) is not None
        return False

    def _model_verdicts(self, batch: list[Candidate]) -> None:
        asked = [r for r in self.reqs + self.excls if any(c.verdicts.get(req_key(r)) is None for c in batch)]
        if not asked:
            return
        lines = []
        for i, r in enumerate(asked):
            tag = " [EXCLUDED VALUE: answer yes if the listing states this value]" if r.get("kind") == "exclude" else ""
            if r.get("kind") != "exclude" and not r.get("named", True) and r.get("property") != "product type":
                tag = " [ONLY THE VALUE WAS GIVEN: any property stating it counts]"
            if r.get("kind") == "soft":
                tag = " [OCCASION OR USE: no only when the listing states a different occasion or use; unclear when it states none]"
            lines.append(f"{i + 1}. {r['property'] or 'property'}: {r['value']}{tag}  ({clip(r['text'], 120)})")
        listings = []
        fit_reqs = any(r.get("pair_code") or re.search(r"compat|works with|fits|model", norm(r.get("property") or "")) for r in asked)
        for c in batch:
            entry = {"id": c.id, "title": clip(c.title, 160), "price": c.price, "in_stock": c.stock,
                     "properties": {humanize(k): clip(v, 80) for k, v in list(c.props.items())[:40]}}
            if fit_reqs and c.text:
                entry["text"] = clip(c.text, 300)  # what a part works with is stated in its text
            listings.append(entry)
        user = "REQUIREMENTS:\n" + "\n".join(lines) + f"\n\nLISTINGS:\n{json.dumps(listings, ensure_ascii=False, default=str)}"
        data = self.brain.ask("judge", JUDGE_SYSTEM, user, max_tokens=1400)
        if not isinstance(data, dict):
            return
        for entry in data.get("listings") or []:
            if not isinstance(entry, dict):
                continue
            c = self.cands.get(str(entry.get("id")))
            if c is None:
                continue
            verdicts = entry.get("verdicts") if isinstance(entry.get("verdicts"), dict) else {}
            for key, val in verdicts.items():
                idx = as_number(key)
                if idx is None or not (1 <= int(idx) <= len(asked)) or val not in ("yes", "no", "unclear"):
                    continue
                k = req_key(asked[int(idx) - 1])
                if c.verdicts.get(k) is None:  # code's exact matches stand
                    c.verdicts[k] = val

    def _fitting(self) -> list[Candidate]:
        return [c for c in self.cands.values() if c.opened and self._in_scope(c) and self._row_ok(c) and c.fits(self.reqs, self.excls) and (c.stock is not False)]

    def _strong_fits(self) -> list[Candidate]:
        return [c for c in self._fitting() if c.strong(self.reqs, self.excls)]

    def _cheaper_walk(self, terms: list[str]) -> None:
        """Look below the cheapest fit when the search was not exhausted and a price filter exists."""
        cap_arg = self.role_arg("search", "max_price_arg")
        if not cap_arg or self.cheaper_walk_done:
            return
        fractions = [1.0, 0.6, 0.35]  # just under the cheapest fit, then well under it: other rows of the same query
        for frac in fractions:
            fitting = self._fitting()
            if not fitting or not self.exploring_ok() or self.searches >= self.search_cap():
                break
            floor = min(c.price for c in fitting if c.price is not None) if any(c.price is not None for c in fitting) else None
            if floor is None or self.search_exhausted:
                break
            below = floor * frac - (1 if float(floor).is_integer() else 0.01) if frac == 1.0 else floor * frac
            if below <= 0 or any(c.price is not None and c.price <= below and not c.opened and self._in_scope(c) and self._row_ok(c) for c in self.cands.values()):
                continue  # cheaper unopened rows are already known; opening them comes first
            if self._bounded_search(terms[0], below, first_page_only=(frac < 1.0)) == 0:
                continue
            self._open_candidates()
        self.cheaper_walk_done = True

    # ------------------------------------------------------------------ phase 4: ask

    def messages_left(self) -> int:
        budget = self.msg_budget if self.msg_budget is not None else (DEFAULT_MESSAGES if self.tool("message") else 0)
        return max(0, budget - self.messages_sent)

    def reserved_messages(self, stage_filter=None) -> int:
        n = 0
        for i, p in enumerate(self.procedures):
            if p.get("action") != "message" or i in self.proofs:
                continue
            if stage_filter and p.get("stage") not in stage_filter:
                continue
            n += 1
        return n

    def ask_questions(self, first: bool = False) -> None:
        """Hidden needs. A shopper states a preference they held back only when a question names that property, so each
        message names several of one item's properties (see _next_properties for their order). The first round sends one
        message per item; the exploring then keeps steps for the second round, which spends what the budget has left on
        items whose answers stated nothing yet. When an answer in the first round states a preference for one of several
        named properties, a follow-up goes out at once (the others may hold a second preference). An item that stated a
        preference is asked again later only while steps are plentiful."""
        self.question_steps = 0
        if not self.tool("message") or self.message_refused:
            return
        free = self.messages_left() - self.reserved_messages(("before_choose", "before_order", "at_decline", "on_change"))  # a swap is likely: keep its message
        limit = 3 + (len(self.items) - 1)
        asked = 0
        this_round: set[int] = set()
        while free > 0 and asked < limit and self.exploring_ok() and not self.message_refused:
            props: list[str] = []
            # items never asked first, then items whose answers stated nothing yet (a product rarely holds back two
            # preferences), then the items asked least
            for item in sorted(self.items, key=lambda it: (it.questions > 0, it.heard > 0, not it.follow_up, it.questions, it.idx)):
                if first and item.idx in this_round and not item.follow_up:
                    continue  # the first round: one question per item, and a follow-up at once after a stated preference
                if item.heard and self.steps_left() <= self.finish_reserve() + FOLLOW_UP_ROOM:
                    continue  # a follow-up costs a step the search may need more
                self.focus(item)
                props = self._next_properties(self._list_size())
                if props:
                    break
            if not props:
                break
            this_round.add(self.cur.idx)
            used = self._ask_and_take(props, free)
            asked += used
            free -= used
        self.focus(self.items[0])
        if first and free > 0 and not self.message_refused:
            self.question_steps = min(free, 2 * sum(1 for it in self.items if not it.heard))

    def _ask_and_take(self, props: list[str], free: int) -> int:
        """Ask the current item's shopper about several properties at once and apply the answer. Returns the messages used."""
        self.cur.sent.append(tuple(props))
        self.cur.follow_up = False
        self.cur.questions += 1
        reply = self._ask(self._question(props))
        used = 1
        if reply is None:
            return used
        if self._wants_one_at_a_time(reply):
            # the shopper takes one question at a time: every later question names one property, and this one's first
            # property goes out alone at once (some shoppers say so even to a one-property question: it goes out again)
            if not self.single_questions:
                self.single_questions = True
                self.diag("the shopper asked for one question at a time: one property per question from now on")
            if free <= 1 or self.messages_left() <= 0:
                return used
            props = props[:1]
            self.cur.sent.append(tuple(props))
            again = self._ask(self._question(props))
            used += 1
            if again is None:
                return used
            reply = again
        elif reply.get("kind") == "confused" and free > 1 and self.messages_left() > 0:
            again = self._ask(self._question(props, plain=True))
            used += 1
            if again is not None:
                reply = again
        if len(props) == 1 and props[0] not in self.cur.refused and self._wants_one_at_a_time(reply):
            # refused again, so nothing was learned about it: a later question may name it alone once more
            self.cur.refused.add(props[0])
            self.cur.sent = [t for t in self.cur.sent if t != tuple(props)]
            return used
        if len(props) > 1 and reply.get("kind") in ("preference", "no_preference", "change", "other"):
            self.lists_taken = True
        self._take_answer(props, reply)
        return used

    def _list_size(self) -> int:
        """Properties the next question names: one after the shopper asked for one at a time; a short list until this
        shopper has answered a list, since a long first list some shoppers refuse costs a message that may be one of two."""
        if self.single_questions:
            return 1
        return QUESTION_PROPS if self.lists_taken else min(FIRST_LIST_PROPS, QUESTION_PROPS)

    @staticmethod
    def _wants_one_at_a_time(reply: dict) -> bool:
        if reply.get("kind") == "one_at_a_time":
            return True
        return reply.get("kind") in ("confused", "other", "no_preference") and bool(ONE_AT_A_TIME.search(norm(reply.get("reply_text") or "")))

    def _take_answer(self, props: list[str], reply: dict) -> None:
        kind = reply.get("kind")
        given = reply.get("property")
        said = given.strip() if isinstance(given, str) and given.strip() else self._answered(props, str(reply.get("reply_text") or ""))
        ptr = clean_pointer(reply.get("pointer"), said)
        if (ptr and kind in ("preference", "change", "other")) or (kind in ("preference", "change") and reply.get("value")):
            # the property the reply spoke to is settled: later questions leave it and names that hold it out
            spoken = set(words(said))
            matched = max(props, key=lambda p: (len(set(words(p)) & spoken), -len(words(p)))) if props else None
            for name in [said] + ([matched] if matched and set(words(matched)) & spoken else []):
                if norm(name) and norm(name) not in self.cur.answered:
                    self.cur.answered.append(norm(name))
            self.cur.follow_up = len(props) > 1
        else:
            self.cur.order += 1  # the first property it named that the shopper has a view on holds nothing: try another order
        if ptr and kind in ("preference", "change", "other"):
            ptr["text"] = f"answer about {said}"
            item = self._target_item(reply.get("item"))
            item.pointers.append(ptr)
            item.heard += 1
            self.diag(f"need: {said} {ptr['relation']} the {ptr['kind']} from {ptr['seller']} {ptr['details']}")
            self.focus(item)
            self.resolve_pointers()
        elif kind in ("preference", "change") and reply.get("value"):
            self.reqs.append({"text": f"asked about {said}", "property": said, "value": str(reply["value"]), "kind": "must"})
            self.heard_queue.append((self.cur, str(reply["value"])))
            self.cur.heard += 1
            self.diag(f"need: {said} = {reply['value']}")
            self._judge_pending()  # opening more rows is the discovery phase's job
        if kind == "change":
            if reply.get("price_cap") is not None:
                reply["price_cap"] = self._stated_amount(as_number(reply.get("price_cap")), [str(reply.get("reply_text") or "")])
            self._apply_change(reply, self.cur)

    @staticmethod
    def _answered(props: list[str], reply: str) -> str:
        """Which asked property a reply is about when the reader did not say: the one whose words the reply holds."""
        said = set(words(reply))
        return max(props, key=lambda p: (all(w in said for w in words(p)), sum(1 for w in words(p) if w in said), len(words(p))))

    def _next_properties(self, n: int = QUESTION_PROPS) -> list[str]:
        """The names the current item's next question gives. A shopper speaks to one property per reply, and when a
        message names several properties the shopper has a view on (some held back, some they do not care about), the
        reply is about the first of them. So the questions name one stable list: as ranked (the hinted properties
        first) and, after each answer that stated nothing, in a new order: reversed, then from the middle out. After a
        stated preference the order stays, without the answered property. A property the shopper answered, a stated
        requirement's property and the names that hold one of them are left out. Empty when no new list is left."""
        names = self._property_list()
        if not names:
            return []
        sent = {tuple(x) for x in self.cur.sent}
        if n == 1:  # one property per question: the best-ranked one not yet asked alone
            alone = {t[0] for t in sent if len(t) == 1}
            return next(([x] for x in names if x not in alone), [])
        mid = len(names) // 2
        middle_out = [names[i] for d in range(len(names)) for i in ((mid - d, mid + d) if d else (mid,)) if 0 <= i < len(names)]
        orders = [names, names[::-1], list(dict.fromkeys(middle_out))]
        orders += [names[k:] + names[:k] for k in range(1, len(names))]
        k = self.cur.order % len(orders)
        for order in orders[k:] + orders[:k]:
            pick = order[:n]
            if tuple(pick) not in sent:
                return pick
        return []

    def _property_list(self) -> list[str]:
        """The current item's property names in a stable order: the shopper's hinted ones first, then those its listings
        carry (the fitting rows weigh double, the other opened rows of the product once, even those a requirement rules
        out; names whose values differ first, the most carried first; who makes or sells it is never asked). Names found later join at
        the end. Names that hold one another count once, as the longer name at the first one's place."""
        fixed = [set(words(x.get("property") or "")) for x in self.reqs + self.excls + self.cur.pointers + self.cur.unresolved
                 if norm(x.get("property") or "") not in ("", "product type")]
        answered = [set(a.split()) for a in self.cur.answered]
        fitting = self._fitting()
        pt = next((req_key(r) for r in self.reqs if r.get("property") == "product type"), None)
        rows = fitting + [c for c in self.cands.values() if c.opened and c not in fitting and self._in_scope(c) and c.verdicts.get(pt) != "no"
                          and (self._dept_names_product(c) or self._names_product(c))]
        values: dict[str, set] = {}
        carried: dict[str, int] = {}
        for c in rows:
            weight = 2 if c in fitting else 1
            for k, v in c.props.items():
                hk = plain_property(k, keep_code=True)
                # details a question never asks about: the listing's own fields, warranty, and who makes or sells it
                if hk and len(hk) <= 40 and not re.search(r"\bid\b|price|stock|title|name|url|image|sku|description|warranty|option|"
                                                        r"categor|department|section|aisle|maker|brand|seller|company|vendor|shop|store", hk):
                    values.setdefault(hk, set()).add(norm(v))
                    carried[hk] = carried.get(hk, 0) + weight
        ranked = sorted(values, key=lambda k: (len(values[k]) < 2, -carried[k], k))
        plan = list(self.cur.plan)
        for name in [plain_property(h, keep_code=True) for h in self.cur.open_props] + ranked:
            ws = set(name.split())
            if not ws:
                continue
            at = next((i for i, p in enumerate(plan) if ws <= set(p.split()) or set(p.split()) <= ws), None)
            if at is None:
                if len(plan) < PLAN_PROPS:
                    plan.append(name)
            elif ws > set(plan[at].split()):
                plan[at] = name  # the longer name ("lamp shade style") asks for the shorter ("style") too, in its place
        self.cur.plan = plan
        out = []
        for name in plan:
            ws = set(name.split())
            if any(ws <= f or (len(f) > 1 and f <= ws) for f in fixed if f):
                continue  # a stated requirement's property
            if any(ws <= a or a <= ws for a in answered if a):
                continue  # answered already ("recommended gender" once "gender" is answered: a judge reads it as the same)
            out.append(name)
        return out

    def _question(self, props: list[str], plain: bool = False) -> str:
        item = self.cur.name or "item"
        listed = props[0] if len(props) == 1 else ", ".join(props[:-1]) + " or " + props[-1]
        if plain:
            return f"Sorry, let me ask more simply: for the {item}, do you want a particular {listed}?"
        examples = self._examples(props[0]) if len(props) == 1 else []
        if examples:
            return f"For the {item}, which {listed} would you like? For example {', '.join(examples[:-1]) + ' or ' + examples[-1] if len(examples) > 1 else examples[0]}."
        return f"For the {item}, which {listed} would you like?"

    def _examples(self, prop: str) -> list[str]:
        """Up to three values the current item's listings give for a property."""
        examples: list[str] = []
        p = norm(prop)
        for c in self._fitting() or [c for c in self.cands.values() if self._in_scope(c) and (self._dept_names_product(c) or self._names_product(c))]:
            for k, v in c.props.items():
                hk = norm(humanize(k))
                if hk == p or hk.endswith(" " + p) or hk.startswith(p + " "):
                    sv = str(v).strip()
                    if sv and sv.lower() not in [e.lower() for e in examples]:
                        examples.append(sv)
        return examples[:3]

    def _ask(self, text: str) -> dict | None:
        """Send one question and read the reply; the reply's own words stay with the reading."""
        reply = self.send_message(text)
        if reply is None:
            return None
        return {**self._read_reply(text, reply), "reply_text": reply}

    def send_message(self, text: str) -> str | None:
        """Send one message; returns the shopper's reply text, or None if it could not be sent."""
        tool = self.tool("message")
        arg = self.role_arg("message", "text_arg")
        if not tool or not arg or self.messages_left() <= 0 or not self.steps_left() > 1:
            return None
        obs = self.call("message", {arg: text[:800]}, "message")
        if obs is None:
            return None
        if isinstance(obs, dict) and obs.get("error"):
            if "budget" in str(obs.get("error")):
                self.message_refused = True
                self.msg_budget = self.messages_sent
            return None
        self.messages_sent += 1
        return self._reply_text(obs)

    @staticmethod
    def _reply_text(obs: Any) -> str:
        if isinstance(obs, str):
            return obs
        if isinstance(obs, dict):
            strs = [v for v in obs.values() if isinstance(v, str)]
            if strs:
                return max(strs, key=len)
            nested = deep_find(obs, lambda x: isinstance(x, str) and len(x) > 0)
            if nested:
                return max((v for _, v in nested), key=len)
        if isinstance(obs, list):
            strs = [v for v in obs if isinstance(v, str)]
            if strs:
                return " ".join(strs)
        return json.dumps(obs, ensure_ascii=False, default=str)

    def _read_reply(self, question: str, reply: str) -> dict:
        data = self.brain.ask("judge", REPLY_SYSTEM, f"MESSAGE SENT:\n{question}\n\nSHOPPER'S REPLY:\n{clip(reply, 800)}", max_tokens=800)
        if isinstance(data, dict) and data.get("kind"):
            return data
        # code reading when no model answered
        r = norm(reply)
        if re.search(r"\b(yes|yep|sure|go ahead|okay|ok|fine|please do|proceed)\b", r) and "?" not in reply:
            return {"kind": "yes"}
        if re.search(r"\b(no|don t|do not|stop|wait|cancel)\b", r):
            return {"kind": "no"}
        if re.search(r"(don t mind|not picky|whatever|no preference|any of them|indifferent|no opinion|not fussy)", r):
            return {"kind": "no_preference"}
        if "?" in reply or re.search(r"(lost me|didn t catch|come again|what do you mean|explain|plain)", r):
            return {"kind": "confused"}
        return {"kind": "other"}

    # ------------------------------------------------------------------ pointers to other listings

    def resolve_pointers(self) -> None:
        """Read the value each of the current item's pointer requirements stands for, following the pages it leads through."""
        item = self.cur
        while item.pointers and not self.wire.finished and not self.wire.dead and self.steps_left() > self.finish_reserve() + 2:
            ptr = item.pointers.pop(0)
            req = self._resolve(ptr)
            if req is None:
                item.unresolved.append(ptr)  # an unread value blocks nothing: the listing is judged on the rest
                self.diag(f"pointer not resolved: {ptr['property']} via the {ptr['kind']} from {ptr['seller']}")
                continue
            target = item.excls if req["kind"] == "exclude" else item.reqs
            if not any(req_key(x) == req_key(req) for x in target):
                target.append(req)
            self.diag(f"pointer resolved: {req['property']} {'not ' if req['kind'] == 'exclude' else ''}{req['value']}")
        self._judge_pending()

    def _resolve(self, ptr: dict) -> dict | None:
        prop, rel = ptr.get("property") or "", ptr.get("relation") or "same"
        target = {"seller": ptr.get("seller") or "", "kind": ptr.get("kind") or "", "details": list(ptr.get("details") or [])}
        seen: set[str] = set()
        for hop in range(POINTER_HOPS + 1):
            if not self.exploring_ok():
                return None
            c = self._find_listing(target, seen)
            if c is None:
                return None
            seen.add(c.id)
            texts = self._open_texts(c)
            ans = self._pointer_answer(prop, rel, c, texts)
            ref = clean_pointer(ans.get("refers")) if isinstance(ans.get("refers"), dict) else None
            if ref is not None and hop < POINTER_HOPS:
                target = ref  # this page sends us on
                continue
            value = ans.get("value")
            if not value and not c.opened and self.exploring_ok():
                self._open_one(c)  # the property sheet holds the value; the text page held the rule
                code = self._pointer_by_code(prop, rel, c, texts)
                rule_needed = rel == "compatible" and re.search(r"goes with|pairs with|matches with|go with", " ".join(texts), re.I)
                ans = code if code.get("value") and not rule_needed else self._pointer_answer(prop, rel, c, texts)
                value = ans.get("value")
            if not value:
                return None
            return self._pointer_requirement(ptr, str(value), str(ans.get("target_value") or value))
        return None

    def _find_listing(self, target: dict, seen: set) -> Candidate | None:
        """The listing a pointer names: the seller's listing of that kind that carries the given facts."""
        seller, kind = target.get("seller") or "", target.get("kind") or ""
        vals = [detail_value(d) for d in target.get("details") or [] if detail_value(d)]
        queries = []
        if seller and kind:
            queries.append(" ".join([seller, kind] + vals))
            queries.append(f"{seller} {kind}")
        if kind and vals:
            queries.append(" ".join([kind] + vals))
        queries.append(seller or kind)
        need = (4 if seller else 0) + (2 if kind else 0)
        scored: dict[str, float] = {}
        for q in dict.fromkeys(" ".join(q.split()) for q in queries if q and q.strip()):
            for c in self._pointer_search(q):
                if c.id not in seen:
                    scored[c.id] = self._pointer_score(c, seller, kind, vals)
            top = sorted((s for s in scored.values() if s >= need), reverse=True)
            if top and (len(top) == 1 or top[0] > top[1] or not vals):
                break
        ranked = sorted((cid for cid, s in scored.items() if s >= need), key=lambda cid: -scored[cid])
        if len(ranked) > 1 and vals and scored[ranked[0]] == scored[ranked[1]]:
            # the seller has several listings of that kind: the facts on their property sheets tell them apart
            for cid in ranked[:2]:
                c = self.cands[cid]
                if not c.opened and self.exploring_ok():
                    self._open_one(c)
                scored[cid] = self._pointer_score(c, seller, kind, vals)
            ranked.sort(key=lambda cid: -scored[cid])
        return self.cands[ranked[0]] if ranked else None

    def _pointer_search(self, query: str) -> list[Candidate]:
        """One page of results for a pointer query; rows are remembered but not tagged as candidates for any item."""
        if not self.tool("search") or not self.exploring_ok():
            return []
        key = norm(query)
        if key in self.pointer_rows:
            return [self.cands[cid] for cid in self.pointer_rows[key] if cid in self.cands]
        self.searches += 1
        obs = self.call("search", self._search_args(query), f"find page: {clip(query, 40)}")
        if obs is None:
            return []
        rows, _ = self._rows(self.tool("search")["name"], obs)
        fm = self.fields.get(self.tool("search")["name"]) or {}
        out = []
        for r in rows:
            self._register(r, tag=False)
            cid = self._row_id(fm or self._guess_fields(r), r)
            if cid and cid in self.cands:
                out.append(self.cands[cid])
        self.pointer_rows[key] = [c.id for c in out]
        return out

    def _pointer_score(self, c: Candidate, seller: str, kind: str, vals: list[str]) -> float:
        hay = norm(" ".join([c.title, c.seller, c.text[:800]] + [f"{humanize(k)} {v}" for k, v in c.props.items()]))
        score = 0.0
        if seller:
            s = norm(seller)
            if norm(c.seller) == s or any(isinstance(v, str) and norm(v) == s for v in c.props.values()):
                score += 4
            elif self._word_in(s, hay):
                score += 3
            else:
                return -1.0  # a seller was named and this is not their listing
        kw = [w for w in words(kind) if len(w) > 2]
        if kw:
            hits = sum(1 for w in kw if self._word_in(w, hay) or self._word_in(w.rstrip("s"), hay) or self._word_in(w.rstrip("es"), hay))
            score += 2 if hits == len(kw) else (1 if hits else 0)
        for v in vals:
            if self._word_in(norm(v), hay):
                score += 1
        return score

    def _open_texts(self, c: Candidate) -> list[str]:
        """Open a listing's page with the tool that shows its free text (fine print, notes); returns those texts."""
        main = (self.tool("open") or {}).get("name")
        names = [n for n in [self.text_tool] + list(self.roles.get("open_alternatives") or []) + [main] if n]
        names = list(dict.fromkeys(names))
        for name in names[:3]:
            t = self.tool_by_name.get(name)
            if not t or not self.exploring_ok():
                break
            props = t["parameters"].get("properties") or {}
            id_arg = self.role_arg("open", "id_arg")
            arg = id_arg if name == main and id_arg in props else next((a for a, p in props.items() if p.get("type") == "string"), None)
            if not arg:
                continue
            self.opens += 1
            obs = self.wire.act(name, fit_args(t["parameters"], {arg: c.id}), f"read page {c.id}")
            if obs is None:
                return []
            obs = self.absorb(name, obs)
            if not isinstance(obs, dict) or not obs or "error" in obs:
                continue
            fm = self._field_map(name, obs)
            row = get_path(obs, fm.get("list_path") or []) if fm.get("list_path") else obs
            if isinstance(row, list):
                row = row[0] if row and isinstance(row[0], dict) else {}
            if not isinstance(row, dict):
                continue
            self._fill(c, row, fm, opened=bool(fm.get("props_key")) and name == main)
            texts = [v.strip() for k, v in row.items() if isinstance(v, str) and len(v.split()) >= 5 and k != fm.get("title_key")]
            text_keys = any(re.search(r"fine|print|note|remark|caveat|small", norm(k)) for k in row)
            if texts or text_keys:
                self.text_tool = name
                return texts
        return []

    def _pointer_answer(self, prop: str, rel: str, c: Candidate, texts: list[str]) -> dict:
        page = {"title": clip(c.title, 160), "seller": c.seller or None,
                "properties": {humanize(k): clip(v, 80) for k, v in list(c.props.items())[:40]},
                "texts": [clip(t, 700) for t in texts[:4]]}
        data = self.brain.ask("judge", POINTER_SYSTEM, f"PROPERTY: {prop}\nRELATION: {rel}\n\nPAGE:\n{json.dumps(page, ensure_ascii=False, default=str)}", max_tokens=700)
        if isinstance(data, dict) and any(data.get(k) for k in ("refers", "value", "target_value")):
            return data
        return self._pointer_by_code(prop, rel, c, texts)

    def _pointer_by_code(self, prop: str, rel: str, c: Candidate, texts: list[str]) -> dict:
        """The page read without a model: a reference to another listing, else the property's value and any pairing rule."""
        joined = " ".join(texts)
        out: dict = {"refers": None, "value": None, "rule": None, "target_value": None}
        m = re.search(r"(?:refer to|comes? from|check|see|look at)\s+(?:a different product,?\s*)?(?:the\s+)?(.{3,60}?)\s+(?:product\s+)?"
                      r"(?:from|by|sold by)\s+([A-Z][\w'-]+?)(?:'s)?[,.]?\s+(?:which|that|with|who)?\s*(?:is|are|has|have)?\s*([^.]*)", joined)
        if m:
            details = [d.strip() for d in re.split(r",|\band\b", m.group(3)) if d.strip()
                       and not re.search(r"\b(not from|this item|nothing here|speaks to)\b", d)][:4]
            out["refers"] = {"seller": m.group(2), "kind": m.group(1), "details": details}
            return out
        p = norm(prop)
        for k, v in c.props.items():
            hk = norm(humanize(k))
            if p and (hk == p or p in hk or hk in p):
                out["value"] = str(v)
                break
        if out["value"]:
            out["target_value"] = out["value"]
            rule = re.search(r"rule for [^:]+:\s*([^.]+)", joined, re.I)
            out["rule"] = rule.group(1).strip() if rule else None
            if rel == "compatible":
                for a, b in re.findall(r"([\w][\w ()./-]*?)\s+goes? with\s+([\w][\w ()./-]*?)(?=,|\.|;|\band\b|$)", joined):
                    a = re.sub(r"^(?:and|or)\s+", "", a.strip())
                    if norm(a) == norm(out["value"]):
                        out["target_value"] = b.strip()
                        break
                    if norm(b) == norm(out["value"]):
                        out["target_value"] = a.strip()
                        break
        return out

    @staticmethod
    def _pointer_requirement(ptr: dict, value: str, target_value: str) -> dict:
        prop, rel = ptr["property"], ptr["relation"]
        text = f"{prop} {rel.replace('_', ' ')} the {ptr['kind']} from {ptr['seller']}".strip()
        if rel == "different":
            return {"text": text, "property": prop, "value": value, "kind": "exclude"}
        v = {"at_least": f"at least {value}", "at_most": f"at most {value}", "compatible": target_value}.get(rel, value)
        if rel == "within" and ptr.get("tolerance"):
            v = f"within {ptr['tolerance']} of {value}"
        return {"text": text, "property": prop, "value": v, "kind": "must"}

    # ------------------------------------------------------------------ items

    def _find_item(self, name: Any) -> Item | None:
        if isinstance(name, str) and name.strip():
            n = norm(name)
            for it in self.items:
                if norm(it.name) == n:
                    return it
            for it in self.items:
                if n and (n in norm(it.name) or norm(it.name) in n):
                    return it
        return None

    def _target_item(self, name: Any) -> Item:
        """The item a reply or notice names; the current item when it names none."""
        return self._find_item(name) or self.cur

    def _item_order(self) -> list[Item]:
        """Devices before the parts that must fit them; otherwise the request's order."""
        order = list(self.items)
        for it in list(order):
            dev = self._find_item(it.pairs_with) if it.pairs_with else None
            if dev is not None and dev is not it and order.index(dev) > order.index(it):
                order.remove(dev)
                order.insert(order.index(it), dev)
        return order

    def _pair_requirement(self, item: Item) -> None:
        """A part that must fit another item gets that item's model number as a requirement once the device is picked."""
        dev = self._find_item(item.pairs_with) if item.pairs_with else None
        pick = dev.pick if dev is not None and dev is not item else None
        if pick is None:
            return
        code = model_code(" ".join([pick.text, pick.title] + [f"{humanize(k)} {v}" for k, v in pick.props.items()]))
        if not code or any(r.get("pair_code") == code for r in item.reqs):
            return
        item.reqs = [r for r in item.reqs if not r.get("pair_code")]  # another device was picked: its model replaces the old one
        item.reqs.append({"text": f"works with the {dev.name} ({code})", "property": "compatible with", "value": code, "kind": "must", "pair_code": code})
        self.diag(f"{item.name} must work with model {code}")

    # ------------------------------------------------------------------ notices and changes

    def handle_notices(self) -> bool:
        """Interpret queued notices (one model call for up to four). Returns True when the plan must be redone."""
        if not self.notices:
            return False
        batch, self.notices = self.notices[:4], self.notices[4:]
        for text in batch:
            self.diag("notice: " + clip(text, 200))
        joined = "\n".join(f"- {clip(t, 600)}" for t in batch)
        data = self.brain.ask("judge", NOTICE_SYSTEM, "CONTRACT:\n" + json.dumps(self._contract_brief(), ensure_ascii=False) + "\n\nNOTICES:\n" + joined, max_tokens=1000)
        if not isinstance(data, dict):
            data = {}
        if data.get("price_cap") is not None:
            data["price_cap"] = self._stated_amount(as_number(data.get("price_cap")), batch)
        if not data.get("cart_unavailable") and self.cart and any(self._says_unavailable(t) for t in batch):
            data["cart_unavailable"] = True  # read by code too: a gone item must never be ordered again
        if not data:
            return False
        redo = False
        p = data.get("procedure")
        if isinstance(p, dict) and (p.get("text") or p.get("content")):
            self.procedures.append({"text": str(p.get("text") or batch[0]), "stage": p.get("stage") if p.get("stage") in ("before_choose", "before_order", "on_change", "at_decline") else "before_order",
                                    "action": p.get("action") if p.get("action") in ("message", "tool", "none") else "message",
                                    "content": str(p.get("content") or ""), "needs_reply": bool(p.get("needs_reply"))})
            redo = True
        if self._apply_change(data):
            redo = True
        if data.get("cart_unavailable"):
            redo = True
            self._drop_unavailable(self._find_item(data.get("item")))
        elif data.get("recheck_cart"):
            redo = True
            self.refresh_cart()
        return redo

    @staticmethod
    def _stated_amount(value: float | None, texts: list[str]) -> float | None:
        """A new price limit in the units its text uses. A model may convert ("540 cts" read as 5.40 dollars) while the
        store's prices and the contract's cap keep the written units: the number the text states wins."""
        if value is None:
            return None
        nums = [n for n in (as_number(m) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", " ".join(texts))) if n is not None and n > 0]
        if not nums or any(abs(n - value) <= 1e-9 * max(1.0, n) for n in nums):
            return value
        for factor in (100.0, 0.01, 1000.0, 0.001):
            hit = next((n for n in nums if abs(n - value * factor) <= 1e-6 * max(1.0, n)), None)
            if hit is not None:
                return hit
        return value

    @staticmethod
    def _says_unavailable(text: str) -> bool:
        return re.search(r"\b(sold out|out of stock|no longer available|unavailable|none left|zero left|nobody has any|all gone|gone from|"
                         r"can t be ordered|cannot be ordered|can no longer|ran dry|dried up|no stock|not available)\b", norm(text)) is not None

    def _drop_unavailable(self, item: Item | None) -> None:
        """The cart's listing for an item can no longer be bought: never pick it again; the next round swaps it."""
        lines = [cid for cid in self.cart]  # in the order they went in: a store names "your pick" for the first
        targets = [(i, c) for i, c in self.picks.items() if c.id in self.cart and (item is None or i == item.idx)]
        if item is None and len(targets) > 1:
            targets.sort(key=lambda ic: lines.index(ic[1].id))
            targets = targets[:1]
        if not targets and item is not None and item.pick is not None:
            targets = [(item.idx, item.pick)]
        for i, c in targets:
            c.stock = False
            self.items[i].rejected.add(c.id)
            self.diag(f"{self.items[i].name}: {c.id} can no longer be bought; swapping it")

    def refresh_cart(self) -> None:
        """A listing in the cart changed: read its price and stock again, and judge its properties again."""
        for cid in list(self.cart):
            c = self.cands.get(cid)
            if c is None or self.steps_left() <= self.finish_reserve():
                continue
            if self.roles.get("open_alternatives") or self.text_tool:
                self._open_texts(c)  # a card shows price and stock
            c.opened = False

    def _contract_brief(self) -> dict:
        cart = [{"item": self.items[i].name, "listing": clip(c.title, 80), "price": c.price} for i, c in sorted(self.picks.items()) if c.id in self.cart]
        return {"items": [{"name": it.name, "quantity": it.quantity, "requirements": [f"{r['property']}: {r['value']}" for r in it.reqs if r.get("property") != "product type"],
                           "exclusions": [f"{e['property']}: {e['value']}" for e in it.excls]} for it in self.items], "cart": cart,
                "price_cap": self.price_cap, "objective": self.objective, "procedures": [p["text"] for p in self.procedures], "message_budget": self.msg_budget}

    def _apply_change(self, data: dict, item: Item | None = None) -> bool:
        """Apply a change from a notice or a reply. Requirements go to the item they name, else to `item`, else the current one."""
        changed = False
        base = self._find_item(data.get("item")) or item or self.cur
        for r in data.get("new_requirements") or []:
            if not isinstance(r, dict):
                continue
            it = self._find_item(r.get("item")) or base
            ptr = clean_pointer(r.get("pointer"), str(r.get("property") or ""))
            if ptr and ptr["property"]:
                ptr["text"] = "changed during the task"
                it.pointers.append(ptr)
                changed = True
                continue
            if r.get("value"):
                entry = {"text": "changed during the task", "property": str(r.get("property") or ""), "value": str(r["value"]), "kind": "exclude" if r.get("kind") == "exclude" else "must"}
                target = it.excls if entry["kind"] == "exclude" else it.reqs
                if not any(norm(x["property"]) == norm(entry["property"]) and norm(x["value"]) == norm(entry["value"]) for x in target):
                    target.append(entry)
                    changed = True
        scope = [self._find_item(data.get("item"))] if self._find_item(data.get("item")) else self.items
        for entry in data.get("dropped") or []:  # as written in the contract: "property: value"
            want = norm(entry)
            prop = norm(str(entry).split(":")[0])
            for it in scope:
                before = len(it.reqs) + len(it.excls) + len(it.pointers)
                keep = lambda r: r.get("property") == "product type" or (norm(f"{r['property']}: {r['value']}") != want and norm(r.get("property") or "") != prop)
                it.reqs = [r for r in it.reqs if keep(r)]
                it.excls = [e for e in it.excls if keep(e)]
                it.pointers = [p for p in it.pointers if norm(p.get("property") or "") != prop]
                if len(it.reqs) + len(it.excls) + len(it.pointers) != before:
                    changed = True
                    self.diag(f"{it.name}: dropped {entry}")
        for prop in data.get("dropped_properties") or []:
            for it in scope:
                before = len(it.reqs) + len(it.excls) + len(it.pointers)
                same = lambda r: norm(r.get("property") or "") == norm(prop) and r.get("property") != "product type"
                it.reqs = [r for r in it.reqs if not same(r)]
                it.excls = [e for e in it.excls if not same(e)]
                it.pointers = [p for p in it.pointers if norm(p.get("property") or "") != norm(prop)]
                changed = changed or len(it.reqs) + len(it.excls) + len(it.pointers) != before
        cap = as_number(data.get("price_cap"))
        if cap is not None and cap > 0 and cap != self.price_cap:
            self.price_cap = cap
            changed = True
        q = as_number(data.get("quantity"))
        q_item = self._find_item(data.get("quantity_item")) or base
        if q is not None and 1 <= q <= 50 and int(q) != q_item.quantity:
            q_item.quantity = int(q)
            changed = True
        mb = as_number(data.get("message_budget"))
        if mb is not None and mb >= 0:
            self.msg_budget = int(mb)
        sb = as_number(data.get("step_budget"))
        if sb is not None and 3 <= sb < self.max_steps:
            self.max_steps = int(sb)
        return changed

    # ------------------------------------------------------------------ phase 5: choose

    def pick_all(self) -> dict[int, Candidate]:
        """One listing per item: every requirement proven if possible, after a wider search; else one that contradicts nothing."""
        picks: dict[int, Candidate] = {}
        order = self._item_order()
        for i, item in enumerate(order):
            self.focus(item, share=len(order) - i)
            self._pair_requirement(item)
            if item.pointers:
                self.resolve_pointers()
            c = self.choose()
            if (c is not None and self.objective == "cheapest" and c.price is not None and self.exploring_ok()
                    and any(x.price is not None and x.price < c.price for x in self._open_order())):
                self._open_candidates()  # a cheaper row seen but never opened may fit (the cheapest fit may have sold out)
                c = self.choose()
            if c is None and self.exploring_ok():
                if self.searches < self.search_cap():
                    self._widen()
                self._open_candidates()  # rows already seen are worth opening even when no search is left
                c = self.choose()
            if c is None and (self.steps_left() <= self.finish_reserve() + 1 + self.question_steps or self.clock.past(SOFT_DEADLINE)
                              or self.opens >= self.open_cap()):
                self._last_chance_opens()  # the exploring is over for the whole task, not only this item's share
                c = self.choose()
            if c is None:
                c = self.choose_plausible()
                if c is not None:
                    self.diag(f"{item.name}: nothing proves every requirement; taking {c.id}, which contradicts none")
            item.pick = c
            if c is not None:
                picks[item.idx] = c
        self.focus(self.items[0])
        self._fit_budget(picks)
        return picks

    def _last_chance_opens(self, most: int = 3) -> None:
        """Nothing fits the current item once the exploring is over (its pick sold out at the finish, say): open the
        likeliest rows already seen while the steps left still cover the open, the cart changes and the order."""
        if not self.tool("open") and not self.roles.get("open_alternatives"):
            return
        keep = {c.id for i, c in self.picks.items() if i != self.cur.idx}
        removes = sum(1 for cid in self.cart if cid not in keep)
        adds = 1 + sum(1 for i, c in self.picks.items() if i != self.cur.idx and c.id not in self.cart)
        steps = sum(1 for i, p in enumerate(self.procedures) if p.get("stage") in ("before_choose", "before_order") and p.get("action") in ("message", "tool"))
        need = 1 + removes + adds + 1 + steps  # this open, the cart changes, the order, the stated steps before it
        opened = 0
        while (opened < most and self.steps_left() >= need and not self.clock.past(HARD_DEADLINE - 30)
               and not self.wire.finished and not self.wire.dead):
            pool = self._open_order()
            if not pool or not self._open_one(pool[0], force=True):
                break
            opened += 1
            self._judge_pending()
            if self._fitting():
                break
        if opened:
            self.diag(f"{self.cur.name}: opened {opened} more after the exploring ended" + ("; one fits" if self._fitting() else ""))

    def choose_plausible(self) -> Candidate | None:
        """The current item's listing that names the product, breaks no requirement or cap, and proves the most."""
        self._judge_pending()
        pt = next((r for r in self.reqs if r.get("property") == "product type"), None)
        stated = [r for r in self.reqs if r.get("property") != "product type"]
        pool = []
        for c in self.cands.values():
            if not c.opened or not self._in_scope(c) or not self._row_ok(c) or (self.stock_only and c.stock is False):
                continue
            if pt is not None and c.verdicts.get(req_key(pt)) not in ("yes", "title"):
                continue
            proven = sum(2 if c.verdicts.get(req_key(r)) == "yes" else 1 if c.verdicts.get(req_key(r)) == "title" else 0 for r in stated)
            own = [r for r in stated if not r.get("pair_code") and r.get("kind") != "soft"]  # fitting the device is common: it is not proof on its own
            if own and not any(c.verdicts.get(req_key(r)) in ("yes", "title") for r in own):
                continue  # it proves none of the properties the shopper asked for: no better than a guess
            if any(r.get("named", True) and c.verdicts.get(req_key(r)) not in ("yes", "title") for r in own):
                continue  # a property the shopper named is checked by name: a listing that does not state it cannot pass
            pool.append((proven, c))
        if not pool:
            return None
        best = max(p for p, _ in pool)
        top = [c for p, c in pool if p == best]
        if self.objective == "best" and self.best_property and len(top) > 1:
            ranked = self._rank_best(top)
            if ranked:
                return ranked[0]
        top.sort(key=lambda c: (0 if c.stock is not False else 1, c.price if c.price is not None else 1e12, c.seen_order))
        return top[0]

    def _fit_budget(self, picks: dict[int, Candidate]) -> None:
        """Several lines must fit one total cap: swap in each item's cheaper fitting listing until they do."""
        if self.price_cap is None or not self.cap_total or len(picks) < 2:
            return

        def total() -> float:
            return sum((c.price or 0) * self.items[i].quantity for i, c in picks.items())

        def line(i: int) -> float:
            return (picks[i].price or 0) * self.items[i].quantity

        for i in sorted(picks, key=lambda i: -line(i)):
            if total() <= self.price_cap + 1e-9:
                break
            self.focus(self.items[i])
            cheaper = [c for c in self._fitting() if c.price is not None and picks[i].price is not None and c.price < picks[i].price]
            if cheaper:
                picks[i] = min(cheaper, key=lambda c: c.price)
                self.items[i].pick = picks[i]
        # still over: look for a cheaper listing under what the other lines leave, unproven lines first, then the dearest;
        # a proven line is only ever replaced by another proven one
        proven = {}
        for i, c in picks.items():
            self.focus(self.items[i])
            proven[i] = c.fits(self.reqs, self.excls)
        for i in sorted(picks, key=lambda i: (proven[i], -line(i))):
            if total() <= self.price_cap + 1e-9 or not self.tool("search"):
                break
            item = self.items[i]
            allowed = (self.price_cap - (total() - line(i))) / item.quantity
            if allowed <= 0:
                continue
            self.focus(item, share=1)
            self.ceiling = allowed
            try:
                for term in self._search_terms()[:3]:
                    if not self.exploring_ok():
                        break
                    self._bounded_search(term, allowed, first_page_only=True)
                    self._open_candidates(limit=4)
                    if self._fitting():
                        break
                found = self.choose() or (None if proven[i] else self.choose_plausible())
            finally:
                self.ceiling = None
            if found is not None and found.price is not None and found.price <= allowed + 1e-9:
                self.diag(f"{item.name}: {found.id} at {found.price} keeps the order under the cap")
                picks[i] = found
                item.pick = found
        self.focus(self.items[0])
        if total() > self.price_cap + 1e-9:
            self.diag(f"the picks total {total()} against a cap of {self.price_cap}")

    def choose(self) -> Candidate | None:
        self._judge_pending()
        fitting = self._fitting()
        if not fitting:
            return None
        strong = [c for c in fitting if c.strong(self.reqs, self.excls)]
        pool = strong or fitting  # a title-only fit is taken only when nothing is proven by properties
        in_stock = [c for c in pool if c.stock is not False]
        pool = in_stock or pool
        if self.objective == "best" and self.best_property and len(pool) > 1:
            ranked = self._rank_best(pool)
            if ranked:
                return ranked[0]
        pool.sort(key=lambda c: (c.price if c.price is not None else 1e12, c.seen_order))
        return pool[0]

    def _rank_best(self, pool: list[Candidate]) -> list[Candidate]:
        # code first: a numeric property
        key = norm(self.best_property or "")
        scored = []
        for c in pool:
            for k, v in c.props.items():
                if key and (key == norm(humanize(k)) or key in norm(humanize(k))):
                    n = as_number(v)
                    if n is not None:
                        scored.append((n, c))
                        break
        if len(scored) == len(pool) and scored:
            scored.sort(key=lambda t: t[0], reverse=(self.best_direction == "max"))
            return [c for _, c in scored]
        listings = [{"id": c.id, "title": clip(c.title, 120), "price": c.price, "properties": {humanize(k): clip(v, 60) for k, v in list(c.props.items())[:30]}} for c in pool[:6]]
        data = self.brain.ask("judge", RANK_SYSTEM, f"PROPERTY: {self.best_property} ({self.best_direction})\n\nLISTINGS:\n{json.dumps(listings, ensure_ascii=False, default=str)}", max_tokens=600)
        if isinstance(data, dict) and isinstance(data.get("order"), list):
            order = [self.cands[str(i)] for i in data["order"] if str(i) in self.cands]
            if order:
                return order
        return pool

    # ------------------------------------------------------------------ phase 6: procedures

    def run_procedures(self, stage: str, picks: dict | None = None, repeat: bool = False, extra: dict | None = None) -> str:
        """Perform every stated step of a stage. Returns 'ok', 'no' (shopper refused) or 'change'.
        With `repeat`, steps already done run again (a step due on each change)."""
        outcome = "ok"
        for i, p in enumerate(self.procedures):
            if p.get("stage") != stage:
                continue
            if not repeat and i in self.proofs and (stage != "before_order" or self.proofs[i] > self.last_cart_change):
                continue
            if p.get("action") == "none":
                self.proofs[i] = self.wire.step
                continue
            if p.get("action") == "tool":
                role = "cart_view" if re.search(r"cart|basket|bag|total|review", norm(p["text"] + " " + p["content"])) else None
                if role and self.tool(role) and self.steps_left() > 2:
                    obs = self.call(role, {}, "procedure: " + clip(p["text"], 40))
                    if obs is not None:
                        self._note_cart(obs)
                        self.proofs[i] = self.wire.step
                        continue
                # no tool fits: a message carries out the step when one is allowed
                if not self.tool("message"):
                    self.proofs[i] = self.wire.step
                    continue
            if self.messages_left() <= 0 or self.steps_left() <= 1:
                self.diag(f"procedure impossible (no messages/steps left): {clip(p['text'], 80)}")
                continue
            text = self._compose(p, picks, extra=extra)
            reply = self.send_message(text)
            if reply is None:
                continue
            self.proofs[i] = self.wire.step
            if p.get("needs_reply"):
                read = self._read_reply(text, reply)
                kind = read.get("kind")
                if kind == "confused" and self.messages_left() > 0 and self.steps_left() > 2:
                    text2 = self._compose(p, picks, plain=True, extra=extra)
                    reply2 = self.send_message(text2)
                    if reply2 is not None:
                        self.proofs[i] = self.wire.step
                        read = self._read_reply(text2, reply2)
                        kind = read.get("kind")
                if kind == "no":
                    outcome = "no"
                elif kind == "change":
                    self._apply_change(read)
                    outcome = "change"
        return outcome

    def _lines(self, picks: dict | None) -> list[dict]:
        out = []
        for idx, c in sorted((picks or {}).items()):
            q = self.items[idx].quantity if 0 <= idx < len(self.items) else 1
            out.append({"item": clip(c.title or c.id, 100), "quantity": q, "price_each": c.price,
                        "line_total": round(c.price * q, 2) if c.price is not None else None})
        return out

    def _compose(self, p: dict, picks: dict | None, plain: bool = False, extra: dict | None = None) -> str:
        facts: dict = {}
        lines = self._lines(picks)
        if len(lines) == 1:
            line = lines[0]
            facts.update({"chosen_item": line["item"], "price_each": line["price_each"], "quantity": line["quantity"]})
            if line["line_total"] is not None:
                facts["total"] = line["line_total"]
        elif lines:
            facts["items"] = lines
            if all(line["line_total"] is not None for line in lines):
                facts["total"] = round(sum(line["line_total"] for line in lines), 2)
        if self.cart_known_lines:
            facts["cart"] = clip(self.cart_known_lines, 600)
        options: dict = {}
        current = self.cur
        for item in self.items:
            self.cur = item
            found = [f"{clip(c.title, 60)} at {c.price}" for c in self._fitting()[:4]]
            if found:
                options[item.name] = found
        self.cur = current
        if options:
            facts["fitting_options"] = options if len(self.items) > 1 else next(iter(options.values()))
        if extra:
            facts.update(extra)
        if plain:
            facts["note"] = "the shopper did not understand the last message; say it again in the simplest words"
        data = self.brain.ask("write", WRITE_SYSTEM, f"STEP THE SHOPPER ASKED FOR:\n{p['text']}\nWHAT IT MUST CONTAIN: {p.get('content') or '-'}\nNEEDS THE SHOPPER'S ANSWER: {p.get('needs_reply')}\n\nFACTS:\n{json.dumps(facts, ensure_ascii=False, default=str)}", max_tokens=600)
        msg = data.get("message") if isinstance(data, dict) else None
        if isinstance(msg, str) and 3 <= len(msg.strip()) <= 800:
            return msg.strip()
        # template when no model answered
        ask = " Shall I go ahead? Please answer yes or no." if p.get("needs_reply") else ""
        if extra and extra.get("replacing"):
            return f"I would like to replace {', '.join(extra['replacing'])} with {', '.join(extra.get('with') or [])}." + ask
        if lines:
            parts = " and ".join(f"{line['quantity']} x {line['item']}" for line in lines)
            total = facts.get("total")
            return f"I plan to order {parts}" + (f" for a total of {total}" if total is not None else "") + "." + ask
        return clip(p.get("content") or p["text"], 300) + (" Is that all right? Please answer yes or no." if p.get("needs_reply") else "")

    # ------------------------------------------------------------------ phase 7: finish

    def _note_cart(self, obs: Any) -> None:
        rows = deep_find(obs, lambda x: isinstance(x, list) and x and all(isinstance(y, dict) for y in x))
        if rows:
            self.cart_known_lines = rows[0][1][:10]
        elif isinstance(obs, dict):
            self.cart_known_lines = [obs]

    def fill_cart(self, picks: dict[int, Candidate]) -> tuple[str, list[int]]:
        """Make the cart hold one line per picked item at its quantity, and nothing else.
        Steps the shopper wants on a change run before any line is replaced. Returns (outcome, items whose add failed):
        outcome is 'ok', or 'kept' when the shopper turned the change down and the cart stays as it was."""
        if not self.tool("cart_add"):
            return "ok", []  # some stores order by id directly; the order step carries the id
        want: dict[str, int] = {}
        for idx, c in picks.items():
            want[c.id] = want.get(c.id, 0) + self.items[idx].quantity
        changes = [cid for cid, q in self.cart.items() if want.get(cid) != q]
        if changes and any(p.get("stage") == "on_change" for p in self.procedures):
            titles = {cid: clip((self.cands.get(cid).title if self.cands.get(cid) else cid) or cid, 80) for cid in set(changes) | set(want)}
            outcome = self.run_procedures("on_change", picks, repeat=True,
                                          extra={"replacing": [titles[cid] for cid in changes],
                                                 "with": [titles[cid] for cid in want if self.cart.get(cid) != want[cid]]})
            if outcome == "no" and self.cart:
                self.diag("the shopper turned down the change: the cart stays as it is")
                return "kept", []
        if self._lines_mode() == "replace":
            return self._set_cart(picks, want)
        by_mode = bool(self.role_arg("cart_add", "mode_arg") and self.roles["cart_add"].get("remove_value") is not None)
        can_remove = by_mode or self._zero_removes()
        stale = [cid for cid, qty in self.cart.items() if want.get(cid) != qty]
        adds = [c for _, c in sorted(picks.items()) if self.cart.get(c.id) != want[c.id]]
        check = 0 if by_mode or self.zero_checked else 1  # the first count-0 removal is checked with a cart view
        if stale and (not can_remove or self.steps_left() < len(stale) + len(adds) + 1 + check):
            # a swap that cannot finish (no remove, or too few steps for remove, add and order) would leave two lines
            self.diag("not enough steps or no remove action for the swap: the cart stays as it is")
            return "kept", []
        for cid in stale:
            if by_mode:
                self.call("cart_add", self._cart_args(cid, self.cart[cid], remove=True), "cart remove")
            else:
                self.call("cart_add", self._cart_args(cid, 0), "cart remove (count 0)")
                if not self.zero_checked and self.tool("cart_view"):
                    self.zero_checked = True
                    got = self._lines_in(self.call("cart_view", {}, "cart view after a count-0 removal"))
                    if got is not None and cid in got:
                        self.zero_fails = True
                        self.diag("a count of 0 does not take a line out in this store: the cart stays as it is")
                        return "kept", []
            self.cart.pop(cid, None)
            self.last_cart_change = self.wire.step
        failed = []
        for idx, c in sorted(picks.items()):
            if self.cart.get(c.id) == want[c.id]:
                continue
            if not self._cart_add(c, want[c.id] - self.cart.get(c.id, 0)):
                failed.append(idx)
        return "ok", failed

    def _lines_mode(self) -> str | None:
        """'replace' when the cart tool takes the whole cart as a list of lines, 'add' when it adds a list, else None."""
        spec = self.roles.get("cart_add") or {}
        if not spec.get("lines_arg"):
            return None
        return "replace" if spec.get("lines_replace") else "add"

    def _lines_args(self, lines: dict[str, int]) -> dict:
        spec = self.roles["cart_add"]
        rows = []
        for cid, qty in lines.items():
            row: dict = {spec["line_id_key"]: cid}
            if spec.get("line_qty_key"):
                row[spec["line_qty_key"]] = qty
            rows.append(row)
        return {spec["lines_arg"]: rows}

    def _lines_in(self, obs: Any) -> dict[str, int | None] | None:
        """The cart's lines as a store's reply lists them (listing id -> count, None where no count field is recognized),
        or None when the reply lists none."""
        best = None
        for path, lst in deep_find(obs, lambda x: isinstance(x, list) and all(isinstance(y, dict) for y in x)):
            key = next((p for p in reversed(path) if isinstance(p, str)), "")
            if not lst and not re.search(r"cart|bag|basket|line|selection|stash|hold|content", norm(key)):
                continue
            lines: dict[str, int | None] = {}
            for row in lst:
                cid = next((str(v) for v in row.values() if isinstance(v, str) and v in self.cands), None)
                if cid is None:
                    break
                q = next((as_number(v) for k, v in row.items() if re.search(r"amount|quant|qty|count|units|held|number", norm(k))
                          and not re.search(r"price|cost|total|due", norm(k)) and as_number(v) is not None), None)
                lines[cid] = None if q is None else (lines.get(cid) or 0) + int(q)
            else:
                if best is None or len(lines) > len(best):
                    best = lines
        return best

    def _set_cart(self, picks: dict[int, Candidate], want: dict[str, int]) -> tuple[str, list[int]]:
        """A cart tool that takes the whole cart: one call sets every line. When the store refuses the set, the lines go
        in one at a time so one refused listing does not keep the others out."""
        if self.cart == want:
            return "ok", []
        before = dict(self.cart)
        obs = self.call("cart_add", self._lines_args(want), "cart set")
        if obs is not None and not self._refused(obs):
            self._took_lines(obs, before, want)
            return "ok", []
        self.diag("cart set refused: " + clip(obs, 200))
        held = {cid: q for cid, q in self.cart.items() if want.get(cid) == q}
        failed = []
        for idx, c in sorted(picks.items()):
            if held.get(c.id) == want[c.id]:
                continue
            if self.steps_left() <= 2:
                failed.append(idx)
                continue
            trial = {**held, c.id: want[c.id]}
            obs = self.call("cart_add", self._lines_args(trial), "cart set (line by line)")
            if obs is None or self._refused(obs):
                failed.append(idx)
                continue
            held = trial
            self._took_lines(obs, dict(self.cart), trial)
        return "ok", failed

    def _took_lines(self, obs: Any, before: dict[str, int], sent: dict[str, int]) -> None:
        """Record a cart set the store accepted. The reply's lines say which listings the cart holds, and their counts
        where it names them; a line that came back with its old count added means the list is added, not set."""
        got = self._lines_in(obs)
        if got is None:
            self.cart = dict(sent)
        else:
            if any(got.get(cid) is not None and got[cid] == q + before.get(cid, 0) for cid, q in sent.items() if before.get(cid)):
                self.roles["cart_add"]["lines_replace"] = False
                self.diag("cart lines: the store adds the list to the cart")
            self.cart = {cid: (q if q is not None else sent.get(cid, before.get(cid, 1))) for cid, q in got.items()}
            if set(self.cart) != set(sent):
                self.diag(f"cart set: the store holds {sorted(self.cart)} for {sorted(sent)}")
        self.last_cart_change = self.wire.step

    def _zero_removes(self) -> bool:
        """A cart tool that takes a listing and a count but has no add or remove switch: a count of 0 takes the line out
        (race 16's store worked this way), unless the schema rules 0 out or a cart view showed the line still there."""
        spec = self.roles.get("cart_add") or {}
        if self.zero_fails or spec.get("mode_arg") or spec.get("lines_arg") or not spec.get("quantity_arg") or not spec.get("id_arg"):
            return False
        tool = self.tool_by_name.get(spec.get("tool") or "") or {}
        p = (tool.get("parameters", {}).get("properties") or {}).get(spec["quantity_arg"]) or {}
        if p.get("type") not in ("integer", "number"):
            return False
        if "exclusiveMinimum" in p and as_number(p["exclusiveMinimum"]) is not None and as_number(p["exclusiveMinimum"]) >= 0:
            return False
        return not ("minimum" in p and as_number(p["minimum"]) is not None and as_number(p["minimum"]) > 0)

    def _cart_args(self, cid: str, qty: int, remove: bool = False) -> dict:
        if self._lines_mode() == "add":
            return self._lines_args({cid: qty})
        args: dict = {}
        if self.role_arg("cart_add", "id_arg"):
            args[self.role_arg("cart_add", "id_arg")] = cid
        if self.role_arg("cart_add", "quantity_arg"):
            args[self.role_arg("cart_add", "quantity_arg")] = qty
        mode = self.role_arg("cart_add", "mode_arg")
        value = self.roles["cart_add"].get("remove_value" if remove else "add_value")
        if mode and value is not None:
            args[mode] = value
        return args

    @staticmethod
    def _refused(obs: Any) -> bool:
        if not isinstance(obs, dict):
            return False
        if obs.get("error") or obs.get("issue") or obs.get("problem"):
            return True
        return any(isinstance(v, str) and re.search(r"\b(too many|not enough|refused|cannot|can t|unable|invalid|no such|not found|"
                                                    r"out of stock|sold out|exceeds|limit reached|not available)\b", norm(v))
                   for v in obs.values())

    def _cart_add(self, c: Candidate, qty: int) -> bool:
        """Add one line; a refused add is retried one unit at a time."""
        if self._lines_mode() == "replace":
            sent = {**self.cart, c.id: self.cart.get(c.id, 0) + qty}
            before = dict(self.cart)
            obs = self.call("cart_add", self._lines_args(sent), "cart add")
            if obs is None or self._refused(obs):
                self.diag("cart add refused: " + clip(obs, 200))
                return False
            self._took_lines(obs, before, sent)
            return True
        obs = self.call("cart_add", self._cart_args(c.id, qty), "cart add")
        if obs is None:
            return False
        if not self._refused(obs):
            self.cart[c.id] = self.cart.get(c.id, 0) + qty
            self.last_cart_change = self.wire.step
            return True
        self.diag("cart add refused: " + clip(obs, 200))
        if qty == 1 or not self.role_arg("cart_add", "quantity_arg"):
            return False
        got = 0
        while got < qty and self.steps_left() > 2:
            obs = self.call("cart_add", self._cart_args(c.id, 1), "cart add (one unit)")
            if obs is None or self._refused(obs):
                break
            got += 1
            self.last_cart_change = self.wire.step
        if got:
            self.cart[c.id] = self.cart.get(c.id, 0) + got
        return got > 0

    def check_cart(self, picks: dict[int, Candidate]) -> bool:
        """Look at the cart once before ordering: a note a store sends when the cart fills rides on this reply, prices
        may have moved, a line may be missing. Returns True when the plan must be redone (a price now breaks the cap)."""
        if not self.tool("cart_view") or self.cart_checked == self.last_cart_change or self.steps_left() <= 3:
            return False
        self.cart_checked = self.last_cart_change
        obs = self.call("cart_view", {}, "cart check")
        if obs is None:
            return False
        self._note_cart(obs)
        rows, _ = self._rows(self.tool("cart_view")["name"], obs) if isinstance(obs, (dict, list)) else ([], {})
        fm = self.fields.get(self.tool("cart_view")["name"]) or {}
        prices = {}
        for row in rows:
            cid = self._row_id(fm, row) if fm else None
            p = as_number(row.get(fm.get("price_key") or "")) if fm else None
            if cid and p is not None:
                prices[cid] = p
        moved = False
        for idx, c in sorted(picks.items()):
            if c.id in prices and c.price is not None and abs(prices[c.id] - c.price) > 1e-9:
                self.diag(f"cart check: {c.id} now costs {prices[c.id]} (was {c.price})")
                c.price = prices[c.id]
                moved = True
        seen = json.dumps(obs, ensure_ascii=False, default=str)
        for idx, c in sorted(picks.items()):
            if c.id not in seen and self.steps_left() > 2:
                self.diag(f"cart check: {c.id} missing; adding it again")
                self.cart.pop(c.id, None)
                self._cart_add(c, self.items[idx].quantity)
        if moved and self.price_cap is not None:
            total = sum((c.price or 0) * self.items[i].quantity for i, c in picks.items())
            each_ok = all((c.price or 0) <= (self.price_cap if not self.cap_total else self.price_cap / max(1, self.items[i].quantity)) for i, c in picks.items())
            if total > self.price_cap + 1e-9 or not each_ok:
                return True
        return moved and self.objective == "cheapest"

    def place_order(self, picks: dict[int, Candidate] | None) -> bool:
        tool = self.tool("order")
        if not tool or self.wire.finished or self.wire.dead:
            return False
        args = dict(self.roles["order"].get("args") or {})
        props = tool["parameters"].get("properties") or {}
        first = min(picks.items())[1] if picks else None
        if first is not None and not self.tool("cart_add"):
            for a, p in props.items():
                if p.get("type") == "string" and not p.get("enum") and re.search(r"id|product|item|sku", norm(a)):
                    args[a] = first.id
                if p.get("type") in ("integer", "number") and re.search(r"quant|qty|count|how", norm(a)):
                    args[a] = self.items[min(picks)].quantity
        self.finish_attempts += 1
        obs = self.call("order", args, "order")
        if obs is None:
            return self.wire.finished
        if self.wire.finished:
            self.finished_by = "order"
            return True
        self.diag("order did not finish: " + clip(obs, 300))
        return False

    def decline(self) -> bool:
        tool = self.tool("decline")
        if not tool or self.wire.finished or self.wire.dead:
            return False
        args = dict(self.roles["decline"].get("args") or {})
        note_arg = self.role_arg("decline", "note_arg")
        if note_arg:
            args[note_arg] = self._decline_note()
        self.finish_attempts += 1
        obs = self.call("decline", args, "decline")
        if self.wire.finished:
            self.finished_by = "decline"
            return True
        if obs is None or self.wire.dead:
            return False
        # not finished: a shared finish tool may need another enum value, or the note may have been refused
        mode = self.role_arg("decline", "mode_arg")
        props = tool["parameters"].get("properties") or {}
        tried = {json.dumps(args, sort_keys=True)}
        options: list[dict] = []
        if mode and isinstance(props.get(mode), dict) and props[mode].get("enum"):
            used = (self.roles.get("order") or {}).get("args", {}).get(mode)
            for e in props[mode]["enum"]:
                if e != used and e != args.get(mode):
                    options.append({**args, mode: e})
        if note_arg:
            options.append({k: v for k, v in args.items() if k != note_arg})
        for alt in options:
            if self.wire.finished or self.wire.dead or self.steps_left() < 1:
                break
            key = json.dumps(alt, sort_keys=True)
            if key in tried:
                continue
            tried.add(key)
            self.finish_attempts += 1
            self.call("decline", alt, "decline (retry)")
        self.finished_by = "decline" if self.wire.finished else None
        return self.wire.finished

    def _decline_note(self) -> str:
        """A plain note that names no product: what was looked for and that nothing fit."""
        item = " and ".join(it.name for it in self.items) or "item"
        reasons = []
        if self.price_cap is not None:
            reasons.append(f"within {int(self.price_cap) if float(self.price_cap).is_integer() else self.price_cap}")
        if self.stock_only:
            reasons.append("in stock")
        want = [f"{r['property']} {r['value']}" for it in self.items for r in it.reqs if r.get("property") != "product type" and not r.get("pair_code")][:3]
        text = f"I could not find a {item}" + (" with " + ", ".join(want) if want else "") + (" that is " + " and ".join(reasons) if reasons else "") + ", so I am not placing an order."
        hint = self.decline_rule.get("note") if isinstance(self.decline_rule, dict) else None
        if isinstance(hint, str) and hint.strip() and self.brain.available():
            data = self.brain.ask("write", WRITE_SYSTEM, "STEP THE SHOPPER ASKED FOR:\nend without an order, telling the shopper that nothing fit\n"
                                  f"WHAT IT MUST CONTAIN: {clip(hint, 200)}; never name or suggest any product\nNEEDS THE SHOPPER'S ANSWER: False\n\nFACTS:\n"
                                  + json.dumps({"looked_for": item, "requirements": want, "conditions": reasons}, ensure_ascii=False), max_tokens=500)
            msg = data.get("message") if isinstance(data, dict) else None
            if isinstance(msg, str) and 10 <= len(msg) <= 390 and not re.search(r"\b[a-z]{1,3}-[0-9a-f]{6,}\b", msg.lower()):
                return msg.strip()
        return text[:390]

    # ------------------------------------------------------------------ driver

    def run(self) -> None:
        if self.wire.finished:
            return
        self.read_request()
        self.map_roles()
        self.run_procedures("before_search")
        self.discover_first()
        self.ask_questions(first=True)  # early, one message per item: the answers narrow the search and the judging
        self.handle_notices()
        self.discover_all()
        self.ask_questions()  # what the budget has left, for items whose answers gave nothing yet
        self.handle_notices()
        self.search_heard()
        while self.rounds < MAX_ROUNDS and not self.wire.finished and not self.wire.dead:
            self.rounds += 1
            if self.notices:
                self.handle_notices()
            picks = self.pick_all()
            if not picks:
                self.diag("no listing fits any item, even loosely: declining")
                self.run_procedures("at_decline")
                if self.decline():
                    return
                if self.wire.dead or self.wire.finished:
                    return
                continue
            before, self.picks = dict(self.picks), picks
            missing = [it.name for it in self.items if it.idx not in picks]
            self.diag("picks: " + "; ".join(f"{self.items[i].name} -> {c.id} {clip(c.title, 60)} at {c.price}" for i, c in sorted(picks.items()))
                      + (f" | nothing for {missing}" if missing else ""))
            outcome = self.run_procedures("before_choose", picks)
            if outcome == "no":
                self._reject(picks)
                continue
            if outcome == "change":
                continue
            kept, failed = self.fill_cart(picks)
            if failed:
                self._reject({i: picks[i] for i in failed})
                continue
            if kept == "ok":
                if self.check_cart(picks):
                    continue  # a price moved: choose again with the prices as they are now
            else:  # the shopper kept the cart as it was: the steps that follow describe that cart
                picks = self.picks = {i: c for i, c in before.items() if self.cart.get(c.id)}
            if self.notices and self.handle_notices():
                continue
            outcome = self.run_procedures("before_order", picks)
            if outcome == "no":
                self._reject(picks)
                continue
            if outcome == "change":
                continue
            if self.place_order(picks):
                return
            if self.notices and self.handle_notices():
                continue
            if self.wire.dead or self.wire.finished:
                return
        self.finish_by_code()

    def search_heard(self) -> None:
        """Each preference heard in a reply gets a search of its own, "<value> <product>", even when the search allowance is
        spent: the earlier searches ran before it was known, and a store that shows a few pages of rows sorted by id can
        hold the one fitting listing past every earlier query's pages, and a query holding the heard value reaches it. Its new rows that carry the value are opened while steps are left beyond the finish.
        Only after the last questions, and only when the allowance is spent: otherwise the usual searches use the value."""
        if self.searches < self.search_cap():
            self.heard_queue.clear()
            return
        while self.heard_queue and not self.wire.finished and not self.wire.dead:
            item, value = self.heard_queue.pop(0)
            if self.steps_left() <= self.finish_reserve() + 2 or self.clock.past(SOFT_DEADLINE) or not self.tool("search"):
                break
            name_words = [w for w in norm(item.name).split() if w not in set(norm(value).split())]
            term = " ".join([value.strip()] + name_words)
            if not term.strip() or len(words(term)) > 6:
                continue
            self.focus(item)
            self.search_bonus += 1
            if self._search_all_pages(term, first_page_only=True):
                self._open_candidates(limit=2)
        self.focus(self.items[0])

    def _reject(self, picks: dict[int, Candidate]) -> None:
        """The shopper or the store turned these listings down: never pick them again for those items."""
        for idx, c in picks.items():
            self.items[idx].rejected.add(c.id)
            if self.items[idx].pick is c:
                self.items[idx].pick = None

    def finish_by_code(self) -> None:
        """Last resort: order the best known listing for each item, else decline."""
        if self.wire.finished or self.wire.dead:
            return
        picks = {i: c for i, c in self.picks.items() if c.id not in self.items[i].rejected}
        for item in self.items:
            if item.idx in picks:
                continue
            self.focus(item)
            c = self.choose() or self.choose_plausible()
            if c is None:
                pool = [x for x in self.cands.values() if self._in_scope(x) and self._row_ok(x)]
                pool.sort(key=lambda x: (x.price if x.price is not None else 1e12, -self._title_score(x)))
                c = pool[0] if pool and self._title_score(pool[0]) > 0 else None
            if c is not None:
                picks[item.idx] = c
        self.focus(self.items[0])
        if picks and self.steps_left() >= 2:
            kept, failed = self.fill_cart(picks)
            if (len(failed) < len(picks) or self.cart) and self.place_order(picks):
                return
        if not self.wire.finished:
            self.decline()


# ----------------------------------------------------------------------------- entry point


LAST_EPISODE: list = []  # offline tests look at the finished episode's state


def agent_main(problem_data: dict, transport=None) -> list[dict]:
    """ORO Bench entry point. `transport` is for offline tests only."""
    ep = Episode(problem_data if isinstance(problem_data, dict) else {}, transport)
    LAST_EPISODE[:] = [ep]
    try:
        ep.run()
    except Exception as exc:  # never leave the episode without a finish attempt
        ep.diag(f"error: {type(exc).__name__}: {clip(str(exc), 200)}")
        try:
            if not ep.clock.past(HARD_DEADLINE + 30):
                ep.finish_by_code()
        except Exception as exc2:
            ep.diag(f"finish error: {type(exc2).__name__}")
    ep.log.append({"role": "assistant", "content": f"summary: finished_by={ep.finished_by} steps={ep.wire.step} model_calls={ep.brain.calls} "
                                                   f"failed_calls={ep.brain.failed} spend={ep.brain.spend:.4f} seconds={ep.clock.elapsed():.1f} "
                                                   f"candidates={len(ep.cands)} opened={ep.opens} messages={ep.messages_sent}"})
    return ep.log


__all__ = ["agent_main"]
