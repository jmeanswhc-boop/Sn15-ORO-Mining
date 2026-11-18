"""Race agent for ORO store worlds.

Code owns the parts a wrong guess makes fatal: which tool does what, the ledger
of requirements and listings, the questions to the shopper, the reaction to
changes during the task, and the checks before an order or a decline. A light
model walks the store with the world's own tools; a heavy model reads the
request once and checks the finish.

Run locally with::

    docker compose run test --agent-file src/agent/race_agent.py
"""

from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"
_BACKUP = {"chutes": "deepseek-ai/DeepSeek-V3.2-TEE", "openrouter": "deepseek/deepseek-v3.2"}

TASK_SECONDS = 300
WRAP_AT = 235          # elapsed seconds after which the agent heads for a finish
FORCE_AT = 268         # elapsed seconds after which it finishes with what it has
ASK_UNTIL = 190        # no new shopper question after this
SPEND_SOFT = 0.05      # heavy work moves to the light model past this
SPEND_HARD = 0.09      # only finish-critical light calls past this
LOOP_TURNS = 45
OBS_CHARS = 5000
ROW_DESC = 160
DETAIL_DESC = 700
AUTO_PAGES = 3
CHEAPER_CHECKS = 4

ASK_WIDTH = 3          # properties named in the last question message, while the shopper answers such lists
MIN_READ = 3           # in-category listings read before an order, when steps and time allow

ENV_TIMEOUT = 30       # seconds one store call may take while time is plentiful
LATE_OPTIONAL = 55     # seconds left below which only finish-critical model calls are made
SELLOUT_ROOM = 4       # steps held back for a swap and a second checkout until the first checkout is tried
CART_TOUCHES = 3       # cart calls on one listing before further ones are refused
MULTI_CALLS = 3        # independent read-only calls run for one model reply

def _backup_model() -> str:
    return _BACKUP.get(getenv("INFERENCE_PROVIDER", "openrouter"), _BACKUP["openrouter"])


# ----------------------------------------------------------------- helpers


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-/]+", " ", str(text or ""))).strip().lower()


def _prop_key(key: str) -> str:
    """Key spelling differs between worlds: 'fa_season', 'm_top_neckline', 'Neckline'."""
    words = _norm(re.sub(r"([a-z])([A-Z])", r"\1 \2", key).replace(".", " ")).split()
    while len(words) > 1 and len(words[0]) <= 2:
        words = words[1:]
    return " ".join(words)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:[.,]\d+)?", value.replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def _json_from(text: str | None) -> Any:
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except ValueError:
                    return None
    return None


def _short(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + " …"


_STOPWORDS = {"a", "an", "the", "of", "and", "or", "for", "with", "in", "on", "to", "is", "are", "be", "it", "by",
              "at", "as", "no", "not"}


def _stem(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 4 and word.endswith("ie"):
        word = word[:-2] + "y"          # hoodie / hoodies
    if len(word) > 5 and word.endswith("ing"):
        base = word[:-3]
        word = base[:-1] if len(base) > 2 and base[-1] == base[-2] else base
    return word


def _words(text: Any) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if w not in _STOPWORDS}


def _near_words(wanted: set[str], found: set[str]) -> bool:
    """Some word of `found` is a close spelling of a wanted word (rectangle/rectangular, stripe/striped)."""
    for a in wanted:
        for b in found:
            if a == b:
                return True
            if len(a) >= 4 and len(b) >= 4 and (a[:5] == b[:5] or SequenceMatcher(None, a, b).ratio() >= 0.8):
                return True
    return False


class Clock:
    def __init__(self) -> None:
        self.start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def left(self) -> float:
        return TASK_SECONDS - self.elapsed()


# ----------------------------------------------------------------- models


class Brain:
    """Model calls with fallbacks, a spend ledger and the episode clock."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.spent = 0.0
        self.calls = 0
        self.failures: dict[str, int] = {}
        # episodes run side by side and each call sets its own timeout, so every episode needs its own client;
        # one attempt per model: a failed request moves on to the next model instead of waiting on the same provider
        self.llm = ProxyClient(timeout=45, max_retries=1, retry_delay=1, rate_limit_retry_delay=2)

    def _chain(self, role: str, critical: bool) -> list[str]:
        if self.spent >= SPEND_HARD and not critical:
            return []
        if role == "heavy" and self.spent < SPEND_SOFT:
            chain = [HEAVY, SPARE, LIGHT]
        else:
            chain = [LIGHT, SPARE]
        chain.append(_backup_model())
        # provider errors come and go: a model that keeps failing moves to the back, but is never dropped for the episode
        return sorted(chain, key=lambda m: self.failures.get(m, 0) >= 2)

    def chat(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1200,
        critical: bool = False,
        cap: float = 45,
    ) -> dict[str, Any] | None:
        for attempt in range(2):
            if attempt:
                if self.clock.left() < 60:
                    return None
                time.sleep(1.5)
            message = self._chat_once(role, messages, tools, max_tokens, critical, cap)
            if message is not None:
                return message
        return None

    def _chat_once(self, role: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                   max_tokens: int, critical: bool, cap: float) -> dict[str, Any] | None:
        if not critical and self.clock.left() < LATE_OPTIONAL:
            return None
        for model in self._chain(role, critical):
            room = min(cap, self.clock.left() - 12)
            if room < 6:
                return None
            body: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            self.calls += 1
            self.llm.timeout = max(5, int(room))
            reply = self.llm.post("/inference/chat/completions", json_data=body)
            if not reply or not reply.get("choices"):
                self.failures[model] = self.failures.get(model, 0) + 1
                continue
            self.failures[model] = 0
            usage = reply.get("usage") or {}
            cost = usage.get("cost") if isinstance(usage, dict) else None
            if isinstance(cost, (int, float)):
                self.spent += float(cost)
            choice = reply["choices"][0]
            message = choice.get("message") or {}
            if not message.get("content") and not message.get("tool_calls") and choice.get("finish_reason") == "length" \
                    and max_tokens < 4000 and self.clock.left() > 25:
                # the budget went on hidden reasoning and nothing was written: ask again with more room
                body["max_tokens"] = min(4000, max_tokens * 3)
                self.calls += 1
                self.llm.timeout = max(5, int(min(cap * 1.5, self.clock.left() - 12)))
                again = self.llm.post("/inference/chat/completions", json_data=body)
                if again and again.get("choices"):
                    usage = again.get("usage") or {}
                    cost = usage.get("cost") if isinstance(usage, dict) else None
                    if isinstance(cost, (int, float)):
                        self.spent += float(cost)
                    message = again["choices"][0].get("message") or message
            return message
        return None

    def ask_json(self, role: str, system: str, user: str, *, max_tokens: int = 1500,
                 critical: bool = False, cap: float = 45) -> Any:
        msg = self.chat(
            role,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, critical=critical, cap=cap,
        )
        return _json_from((msg or {}).get("content"))


# ----------------------------------------------------------------- tools


_FINAL_RE = re.compile(r"\bfinal\b|no (?:more|further) actions|ends? the (?:chat|episode|session|task|conversation)|"
                       r"nothing (?:more|further)|accepts? no(?:thing)? (?:more|further)|terminal|closes? the (?:chat|session|request)", re.I)
_DECLINE_RE = re.compile(
    r"no purchase|without (?:a )?(?:buy|purchas|order)|give up|abandon|declin|refus|reject|cancel|"
    r"nothing (?:to|was) (?:buy|bought)|close out|walk away|end without|no order|cannot be met", re.I)
_ORDER_RE = re.compile(r"check ?out|purchase|place (?:an |the )?order|\border\b|\bbuy\b|settle|pay", re.I)
_MSG_RE = re.compile(r"shopper|customer|buyer|client|user", re.I)
_ASK_RE = re.compile(r"ask|question|message|write|send|note|chat|contact|reply|pose", re.I)
_CART_RE = re.compile(r"cart|bag|basket|trolley|order list", re.I)
_ADD_VALS = re.compile(r"^(?:add|include|put|insert|keep|plus|increase|set|append|attach|stash)", re.I)
_DEL_VALS = re.compile(r"^(?:remove|discard|drop|delete|take|minus|decrease|clear|erase|pull|strip|exclude)", re.I)


class Tool:
    def __init__(self, spec: dict[str, Any]) -> None:
        fn = spec.get("function", spec)
        self.spec = spec
        self.name: str = fn.get("name", "")
        self.desc: str = fn.get("description", "") or ""
        self.params: dict[str, Any] = fn.get("parameters") or {}
        self.props: dict[str, Any] = self.params.get("properties") or {}
        self.required: list[str] = list(self.params.get("required") or [])
        self.text = f"{self.name.replace('_', ' ')} {self.desc}"

    def string_params(self) -> list[str]:
        return [k for k, v in self.props.items() if (v or {}).get("type") == "string" and not v.get("enum")]

    def enum_params(self) -> dict[str, list[Any]]:
        return {k: list(v["enum"]) for k, v in self.props.items() if isinstance(v, dict) and v.get("enum")}

    def check(self, args: dict[str, Any]) -> str | None:
        """Return why the store would refuse these arguments, so no step is wasted."""
        if not isinstance(args, dict):
            return "arguments must be an object"
        for key in self.required:
            if key not in args or args[key] in (None, ""):
                return f"missing required argument '{key}'"
        if self.params.get("additionalProperties") is False:
            extra = [k for k in args if k not in self.props]
            if extra:
                return f"unknown argument(s) {extra}; allowed: {sorted(self.props)}"
        for key, value in args.items():
            schema = self.props.get(key) or {}
            if schema.get("enum") and value not in schema["enum"]:
                return f"'{key}' must be one of {schema['enum']}"
            kind = schema.get("type")
            if kind == "integer" and not (isinstance(value, int) and not isinstance(value, bool)):
                return f"'{key}' must be an integer"
            if kind == "number" and not isinstance(value, (int, float)):
                return f"'{key}' must be a number"
            if kind == "string" and not isinstance(value, str):
                return f"'{key}' must be a string"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "minimum" in schema and value < schema["minimum"]:
                    return f"'{key}' must be at least {schema['minimum']}"
                if "maximum" in schema and value > schema["maximum"]:
                    return f"'{key}' must be at most {schema['maximum']}"
            if isinstance(value, str):
                if "maxLength" in schema and len(value) > schema["maxLength"]:
                    return f"'{key}' is longer than {schema['maxLength']} characters"
                if schema.get("minLength") and len(value) < schema["minLength"]:
                    return f"'{key}' is too short"
        return None


class Roles:
    """Which tool does what in this world, worked out from names, descriptions and schemas."""

    def __init__(self, tools: list[Tool]) -> None:
        self.tools = {t.name: t for t in tools}
        self.message: tuple[str, str] | None = None
        self.order: tuple[str, dict[str, Any]] | None = None
        self.decline: tuple[str, dict[str, Any], str | None] | None = None
        self.cart_change: dict[str, dict[str, Any]] = {}
        self.cart_view: str | None = None
        self.details: list[str] = []
        self.search: list[str] = []
        self._guess()

    def _guess(self) -> None:
        for t in self.tools.values():
            text = t.text
            strings = t.string_params()
            if _FINAL_RE.search(text) or (_DECLINE_RE.search(text) and not t.required and not self.decline):
                enums = t.enum_params()
                flags = [k for k, v in t.props.items() if (v or {}).get("type") == "boolean"]
                if _DECLINE_RE.search(text) and not _ORDER_RE.search(t.name):
                    note = next((p for p in strings), None)
                    self.decline = self.decline or (t.name, {}, note)
                elif enums or flags:
                    self._shared_final(t, enums, flags, strings)
                else:
                    self.order = self.order or (t.name, {})
                continue
            if self.message is None and _MSG_RE.search(text) and _ASK_RE.search(text) and strings \
                    and not _CART_RE.search(t.name):
                text_arg = max(strings, key=lambda p: int((t.props[p] or {}).get("maxLength", 0) or 0))
                self.message = (t.name, text_arg)
                continue
            if _CART_RE.search(text):
                mode = self._cart_mode(t)
                if mode:
                    self.cart_change[t.name] = mode
                    continue
                if not t.required:
                    self.cart_view = self.cart_view or t.name
                    continue
            id_like = [p for p in t.required if self._is_id(p) or self._idish(t, p)]
            if len(t.required) == 1 and id_like:
                self.details.append(t.name)
            elif t.props:
                self.search.append(t.name)

    def _shared_final(self, t: Tool, enums: dict[str, list[Any]], flags: list[str], strings: list[str]) -> None:
        note = next((p for p in strings), None)
        for key, values in enums.items():
            buy = next((v for v in values if _ORDER_RE.search(str(v))), None)
            stop = next((v for v in values if _DECLINE_RE.search(str(v)) or re.search(r"abstain|none|skip", str(v), re.I)), None)
            if buy is not None and stop is not None:
                self.order = self.order or (t.name, {key: buy})
                self.decline = self.decline or (t.name, {key: stop}, note)
                return
        for key in flags:
            if re.search(r"abstain|declin|refus|no.?purchase|give.?up|skip", key, re.I):
                self.order = self.order or (t.name, {key: False})
                self.decline = self.decline or (t.name, {key: True}, note)
                return
        self.order = self.order or (t.name, {})

    @staticmethod
    def _idish(t: "Tool", param: str) -> bool:
        """A short free-form string argument that is not prose: a listing identifier whatever it is called."""
        spec = t.props.get(param) or {}
        return (spec.get("type") == "string" and not spec.get("enum") and int(spec.get("maxLength") or 0) <= 128
                and not re.search(r"text|query|note|message|remark|reason|comment|body|prompt|question|phrase|word|term|^q$|label|name$",
                                  param, re.I))

    @staticmethod
    def _is_id(param: str) -> bool:
        return bool(re.search(r"id$|^id|pid|sku|code|target|listing|product|item|article|chosen|which", param, re.I))

    def _cart_mode(self, t: Tool) -> dict[str, Any] | None:
        id_arg = next((p for p in t.props if self._is_id(p) and (t.props[p] or {}).get("type") == "string"), None)
        if not id_arg:
            id_arg = next((p for p in t.required if self._idish(t, p)), None)
        if not id_arg:
            return None
        qty = next((p for p, v in t.props.items()
                    if (v or {}).get("type") in ("integer", "number")
                    and re.search(r"qty|quant|how.?many|count|units|number|amount|pieces|copies|times|pcs", p, re.I)), None)
        for key, values in t.enum_params().items():
            add = next((v for v in values if _ADD_VALS.search(str(v))), None)
            rem = next((v for v in values if _DEL_VALS.search(str(v))), None)
            if add is not None or rem is not None:
                return {"id": id_arg, "qty": qty, "op": key, "add": add, "remove": rem}
        if re.search(r"remove|discard|delete|drop|take out", t.text, re.I):
            return {"id": id_arg, "qty": qty, "op": None, "add": None, "remove": True}
        return {"id": id_arg, "qty": qty, "op": None, "add": True, "remove": None}

    def apply_brief(self, roles: dict[str, Any]) -> None:
        """The brief model may correct the guesses; only real tool names are accepted."""
        if not isinstance(roles, dict):
            return
        msg = roles.get("message") or {}
        if isinstance(msg, dict) and msg.get("tool") in self.tools:
            t = self.tools[msg["tool"]]
            arg = msg.get("text_arg") if msg.get("text_arg") in t.props else (t.string_params() or [None])[0]
            if arg:
                self.message = (t.name, arg)
        for key in ("order", "decline"):
            spec = roles.get(key) or {}
            if not isinstance(spec, dict) or spec.get("tool") not in self.tools:
                continue
            t = self.tools[spec["tool"]]
            fixed = {k: v for k, v in (spec.get("args") or {}).items()
                     if k in t.props and v in ((t.props[k] or {}).get("enum") or [v])}
            if key == "order":
                self.order = (t.name, fixed)
            else:
                note = spec.get("text_arg") if spec.get("text_arg") in t.props else (t.string_params() or [None])[0]
                self.decline = (t.name, fixed, note)

    def finish_tools(self) -> set[str]:
        names = set()
        if self.order:
            names.add(self.order[0])
        if self.decline:
            names.add(self.decline[0])
        if self.message:
            names.add(self.message[0])
        return names

    def describe(self) -> str:
        return json.dumps({
            "message": self.message, "order": self.order, "decline": self.decline,
            "cart_change": self.cart_change, "cart_view": self.cart_view,
            "details": self.details, "search": self.search,
        })


# ----------------------------------------------------------------- observations


_TITLE_KEYS = re.compile(r"^(?:title|name|product.?name|item.?name|label|heading|headline|caption|designation|product.?title|listing.?name)$", re.I)
_PRICE_KEYS = re.compile(r"price|cost|amount|cts|cents|fee", re.I)
_STOCK_KEYS = re.compile(r"stock|avail|in.?store|on.?hand|sold.?out|orderable|buyable|inventory|purchasab|can.?buy", re.I)
_CAT_KEYS = re.compile(r"categ|department|aisle|section|kind|type$|class|shelf|group", re.I)
_BRAND_KEYS = re.compile(r"brand|maker|company|manufacturer|vendor|seller|shop|store", re.I)
_QTY_KEYS = re.compile(r"qty|quant|count|units|how.?many|number|copies|pieces|pcs", re.I)
_PRICE_WORDS = re.compile(r"price|cost|cts|cents|fee|due|total|charge", re.I)
_ID_KEY = re.compile(r"(?:^|_)(?:id|pid|sku|ref|uid|ident|handle|article)$|^id$|_id$|pid$|^sku|listing|item.?code|product.?code|_ref$|^ref", re.I)
_ID_VALUE = re.compile(r"^(?=[^0-9]*[0-9])[A-Za-z]{1,4}[-_][0-9A-Za-z]{6,}$")   # needs a digit: "high-waisted" is not an id


def _qty_of(node: dict[str, Any]) -> int:
    """A cart line's quantity: a quantity-named number, else a small whole number under a bare amount key."""
    for k, v in node.items():
        if _QTY_KEYS.search(k) and not _PRICE_WORDS.search(k) and _num(v):
            return max(1, int(_num(v)))
    for k, v in node.items():
        if re.search(r"amount", k, re.I) and not _PRICE_WORDS.search(k) and isinstance(v, int) and 0 < v <= 5:
            return v
    return 1


def _id_of(d: dict[str, Any]) -> str | None:
    best = None
    for key, value in d.items():
        if isinstance(value, (str, int)) and not isinstance(value, bool) and _ID_KEY.search(key):
            if re.search(r"session|call|order|cart|shop|store|seller|category", key, re.I):
                continue
            best = best or str(value)
            if re.search(r"product|item|pid|listing", key, re.I):
                return str(value)
    if best is None:
        for key, value in d.items():
            if isinstance(value, str) and _ID_VALUE.match(value) and not re.search(r"session|order|cart|shop|store|seller|categ", key, re.I):
                return value
    return best


def _is_listing(d: Any) -> bool:
    """A nested mapping is a listing of its own only when it has an id and a title or a price; an attribute list
    may hold a value that merely looks like an id."""
    return isinstance(d, dict) and bool(_id_of(d)) and any(_TITLE_KEYS.match(k) or _PRICE_KEYS.search(k) for k in d)


def _split_notes(obs: Any) -> tuple[Any, list[str]]:
    """Shopper updates ride along with an observation under keys that change per world."""
    if not isinstance(obs, dict) or not 1 < len(obs) <= 3:
        return obs, []
    notes: list[str] = []
    rest: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, list) and value and all(isinstance(x, str) and " " in x and len(x) > 15 for x in value):
            notes.extend(value)
        elif isinstance(value, str) and len(value.split()) >= 5 and len(value) > 20 \
                and re.search(r"refus|note|update|shopper|customer", key, re.I):
            notes.append(value)     # a refusal or update written as one string
        else:
            rest[key] = value
    if not notes:
        return obs, []
    inner = next(iter(rest.values())) if len(rest) == 1 else rest
    return inner, notes


def _is_error(obs: Any) -> bool:
    if isinstance(obs, dict):
        keys = " ".join(obs.keys()).lower()
        if re.search(r"error|issue|problem|fault|denied|refus|fail", keys):
            return True
        for value in obs.values():
            if value is False and len(obs) == 1:
                return True
        return False
    if isinstance(obs, str):
        return bool(re.match(r"\s*(?:error|invalid|unknown|not (?:found|allowed)|cannot|refused|denied)", obs, re.I))
    return False


class Listing:
    def __init__(self, lid: str) -> None:
        self.id = lid
        self.title = ""
        self.price: float | None = None
        self.stock: bool | None = None
        self.category = ""
        self.brand = ""
        self.desc = ""
        self.traits: dict[str, Any] = {}
        self.extra: dict[str, Any] = {}
        self.detailed = False   # its attributes have been read
        self.viewed = False
        self.seen_step = 0
        self.refreshed_step = 0
        self.stock_step = 0     # step at which a record last showed this listing's stock

    def card(self, traits: bool = True, evidence: Any = None) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "title": self.title[:140], "price": self.price,
                               "in_stock": self.stock, "category": self.category}
        if evidence:
            out["evidence"] = evidence
        if self.brand:
            out["brand"] = self.brand
        if traits and self.traits:
            out["attributes"] = self.traits
        if traits and self.extra:
            out["other"] = {k: _short(v, 200) for k, v in list(self.extra.items())[:8]}
        if traits and self.desc:
            out["description"] = self.desc[:DETAIL_DESC]
        return out


class Catalog:
    def __init__(self) -> None:
        self.items: dict[str, Listing] = {}

    def absorb(self, obs: Any, step: int, detail_tool: bool) -> list[str]:
        found: list[str] = []
        self._walk(obs, step, detail_tool, found, 0)
        return found

    def _walk(self, node: Any, step: int, detail_tool: bool, found: list[str], depth: int) -> None:
        if depth > 6:
            return
        if isinstance(node, list):
            for x in node:
                self._walk(x, step, detail_tool, found, depth + 1)
            return
        if not isinstance(node, dict):
            return
        lid = _id_of(node)
        looks = lid and any(_TITLE_KEYS.match(k) or _PRICE_KEYS.search(k) or isinstance(v, dict) for k, v in node.items())
        if looks:
            self._record(lid, node, step, detail_tool)
            found.append(lid)
        for key, value in node.items():
            if isinstance(value, (list, dict)) and not (looks and isinstance(value, dict) and not _is_listing(value)):
                self._walk(value, step, detail_tool, found, depth + 1)

    def _record(self, lid: str, node: dict[str, Any], step: int, detail_tool: bool) -> None:
        item = self.items.setdefault(lid, Listing(lid))
        item.seen_step = step
        for key, value in node.items():
            if value is None:
                continue
            if _TITLE_KEYS.match(key) and isinstance(value, str):
                item.title = value
            elif _PRICE_KEYS.search(key) and _num(value) is not None and not isinstance(value, dict):
                if item.price is None or re.match(r"^(?:price|unit.?price|cost)$", key, re.I) or detail_tool:
                    item.price = _num(value)
            elif _STOCK_KEYS.search(key) and isinstance(value, (bool, int, str)):
                item.stock = self._stock(key, value)
                item.stock_step = step
            elif _CAT_KEYS.search(key) and isinstance(value, str):
                item.category = value
            elif _BRAND_KEYS.search(key) and isinstance(value, str):
                item.brand = value
            elif re.search(r"desc|detail|about|summary|blurb|text", key, re.I) and isinstance(value, str):
                if len(value) > len(item.desc) or detail_tool:
                    item.desc = value
            elif isinstance(value, dict) and not _is_listing(value):
                item.traits.update({str(k): v for k, v in value.items()})
                item.detailed = True
            elif not _ID_KEY.search(key) and isinstance(value, (str, int, float, list)):
                item.extra[key] = value
        if detail_tool:
            item.viewed = True
            item.refreshed_step = step

    @staticmethod
    def _stock(key: str, value: Any) -> bool | None:
        negative = bool(re.search(r"sold.?out|out.?of", key, re.I))
        if isinstance(value, bool):
            return not value if negative else value
        if isinstance(value, (int, float)):
            return (value <= 0) if negative else value > 0
        text = str(value).lower()
        if re.search(r"out of stock|sold out|unavailable|none|^no$|^0$|gone", text):
            return negative
        if re.search(r"in stock|available|yes|^true$|on hand|\d", text):
            return not negative
        return None


def _compact(obs: Any, limit: int = OBS_CHARS) -> str:
    """Shorten what the model reads: row descriptions are mostly seller boilerplate."""

    def trim(node: Any, depth: int) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if isinstance(v, str) and re.search(r"desc", k, re.I):
                    out[k] = v[: (ROW_DESC if depth > 1 else DETAIL_DESC)]
                else:
                    out[k] = trim(v, depth + 1)
            return out
        if isinstance(node, list):
            return [trim(x, depth + 1) for x in node]
        return node

    return _short(trim(obs, 0), limit)


# ----------------------------------------------------------------- ledger


class Ledger:
    def __init__(self, request: str, max_steps: int) -> None:
        self.request = request
        self.max_steps = max_steps
        self.items: list[dict[str, Any]] = []
        self.budget: float | None = None
        self.budget_each = False
        self.optimize = "none"
        self.optimize_detail = ""
        self.need_stock = True
        self.question_budget: int | None = None
        self.questions_used = 0
        self.questions_refused = False
        self.decline_rule = ""
        self.procedure: list[dict[str, Any]] = []
        self.hint_props: list[str] = []
        self.stated_props: set[str] = set()
        self.answers: list[dict[str, Any]] = []
        self.asked: list[str] = []
        self.dropped: list[str] = []
        self.notes: list[str] = []
        self.unavailable: set[str] = set()
        self.cart: dict[str, int] = {}
        self.verdicts: dict[str, dict[str, Any]] = {}
        self.verdict_key = ""
        self.searches = 0
        self.done_procedure: set[int] = set()

    def requirements(self) -> list[dict[str, Any]]:
        reqs: list[dict[str, Any]] = []
        for i, item in enumerate(self.items):
            for r in item.get("requirements") or []:
                prop = _norm(r.get("property"))
                if prop and any(_norm(d) in prop or prop in _norm(d) for d in self.dropped):
                    continue
                reqs.append({"item": i, **r})
        for a in self.answers:
            if a.get("kind") == "answer":
                reqs.append({"item": a.get("item", 0), "property": a["property"], "value": a.get("value"),
                             "kind": "must", "from": "shopper reply"})
        return reqs

    def key(self) -> str:
        return json.dumps([self.requirements(), self.budget, sorted(self.unavailable)], sort_keys=True, default=str)

    def summary(self) -> dict[str, Any]:
        return {
            "items": [{"what": it.get("what"), "quantity": it.get("quantity", 1),
                       "category_hint": it.get("category_hint")} for it in self.items],
            "requirements": self.requirements(),
            "dropped_by_shopper": self.dropped,
            "budget_total": self.budget, "budget_applies_to": "each item" if self.budget_each else "the whole purchase",
            "optimize": self.optimize, "optimize_detail": self.optimize_detail,
            "must_be_in_stock": self.need_stock,
            "decline_rule": self.decline_rule,
            "procedure": self.procedure,
            "shopper_updates": self.notes[-6:],
        }


# ----------------------------------------------------------------- prompts


BRIEF_SYSTEM = """You read one shopping request and the store's tool list, and return a JSON brief. Return only JSON, no prose.

Schema:
{
 "items": [{"what": "product noun as the shopper names it", "quantity": 1, "category_hint": "store category or department words if stated",
            "requirements": [{"property": "attribute name", "value": "required value", "kind": "must|exclude|max|min|one_of|same_as_listing"}],
            "wishes": ["soft preferences only"]}],
 "budget_total": number or null,     // the most the whole purchase may cost, in the unit the request uses
 "budget_scope": "total|each",       // "each" only when the limit applies to every item or unit separately
 "optimize": "cheapest|best|most|least|none", "optimize_detail": "what is minimised or maximised",
 "must_be_in_stock": true|false,
 "question_budget": integer or null, // messages the shopper allows; 0 if they forbid questions
 "decline_rule": "what the request says to do when nothing fits, verbatim-ish",
 "procedure": [{"when": "before_order|before_decline|first|anytime", "action": "a step the request requires, e.g. tell the shopper the total before buying"}],
 "hint_properties": ["product properties the story suggests the shopper cares about but did not state, most likely first"],
 "roles": {"message": {"tool": "...", "text_arg": "..."}, "order": {"tool": "...", "args": {}},
           "decline": {"tool": "...", "args": {}, "text_arg": "..."}}
}

Rules:
- Requirements are only what the shopper states as needed. Product kind itself goes in "what", not in requirements.
- The requests follow a pattern. A stated requirement is a trait of the product that the shopper asserts: it is phrased as needed ("has to be", "got to be", "should be", "must", "nothing else", "the X is Y") and usually followed by a remark that they will check it themselves in parentheses ("I'll check the label myself", "I'll double-check whatever you turn up", "don't worry"). Such a trait is a requirement even when it is a purpose ("they're for a wedding (I'll double-check ...)" means occasion = wedding).
- The rest of the story (who the item is for, what the shopper does, a cousin's errand, an event they attend, their feelings, a remark that someone is "particular" or that they "won't guess") is NOT a requirement. Never turn a story detail into a requirement: it only feeds hint_properties. Test: if the sentence has no assertion that the product must be so, and no "I'll check" remark about it, leave it out.
- Budget, message limits, step limits and "on hand / in stock" are never requirements.
- Words that only describe the kind of product itself ("UHT milk, the ready to drink kind", "a skirt for women") belong to "what" and category_hint, not to requirements: a store category label often carries them, e.g. "UHT Milk(Ready to Drink)".
- category_hint: the label the store most likely uses for the kind, written like a store label ("Pants & Jeans" for "pants or jeans", "Hoodies & Sweatshirts" for "a hoodie or sweatshirt").
- "whichever fitting one costs the least" or "the cheapest" means optimize = "cheapest".
- hint_properties: read the story for concerns such as how something looks up close (pattern, texture), fit problems last time (size, fit, length), use cases (capacity, resolution), and list concrete attribute names a store would record. Do not repeat stated requirements.
- procedure: only extra steps the request demands along the way. Telling the shopper that nothing was found is what the decline itself does; never list it as a procedure step.
- roles: use only exact tool names from the list. "order" is the tool that buys the cart; "decline" ends with no purchase; when one tool does both, give the fixed args that select each.
"""

LOOP_SYSTEM = """You are the shopping hands of an agent serving one shopper in an online store. You act only through the store's tools. Every store call uses one of a limited number of steps; thinking is free. The task scores 1 only if the finish is exactly right: every requirement met (stated ones and ones the shopper revealed), within budget, in stock, the cheapest or best fitting choice when asked, and the request's procedure followed. Otherwise it scores 0.

What the code around you does, so you do not:
- It sends every message to the shopper. You never message the shopper; their answers are added to the status.
- It owns finishing. When the cart is right, call finish_order. When the store truly has nothing that meets everything, call finish_decline. The code checks your finish and may send you back with reasons.

How to work:
- Read each tool's schema; use exact argument names and allowed values. Search queries work best short: the product noun plus the key value.
- Search rows are summaries. Stores split a listing over several views: one may show its offer (price, stock), another its attribute list. The status names the attribute tool once it is known. Read a listing's attributes with that tool before you judge or buy it. Do not reopen a listing that has not changed.
- A requirement on a property is met when the listing's attribute for that property states the value (spelling, case, plural and plain synonyms are fine; different words are not). If no attribute names the property, the title or description stating it counts. Respect exclusions. Budget limits include the edge value.
- The kind of product must match the listing's category, not a neighbouring category.
- For "cheapest" (or least/most): page through all results under the budget, then open candidates in price order and take the first that meets every requirement. Use a price-cap filter when the search tool has one, and sort by price from low to high when it can sort.
- Search tools often have two modes. The keyword mode reads your words: every word must appear in a title or description, so keep the query to two or three words (the key value and the product noun), and try other wordings when it finds little. The filter mode ignores any words: it reads only its filter arguments (category label, brand, price ceiling), so never put words in a filter search. A result page holds only a few rows in no useful order: read the further pages the result reports. Start with a keyword search of the key value plus the product noun ("baggy pants", "wrap skirt"), because a listing often states the value in its title; browsing a whole category by filter only shows an arbitrary slice, so use the filter mode to lower prices or to list a small category.
- Most requests can be met. Do not end without a purchase until you have searched with several wordings and opened many listings; a listing that merely looks different from the wording (a synonym, a plural) may still state the value.
- Add exactly the requested quantity of each item to the cart with the cart tool. Remove anything that should not be bought.
- Shopper updates (a budget change, a dropped need, an item gone) can arrive at any time; they appear in the status. Act on them at once: replace a gone item, recheck the budget.
- When a requirement points at another listing ("the same colour as the one sold by X"), find and open that listing first.
- Keep going until finish_order or finish_decline. When you need to read several listings, ask for up to 3 of them (or a search and a listing) in one reply: reads that do not depend on each other. Any cart change, message or finish call goes alone."""

JUDGE_SYSTEM = """You check store listings against a shopper's requirements. Return only JSON.
For each listing return {"id": ..., "fit": "yes|no|unknown", "why": "short reason naming the deciding attribute"}.
How the store decides a property requirement:
- The attribute whose name is that property decides (spelling variants of a key, such as 'sleeve length', 'sleeve_length' and 'm.sleeve.length', are one attribute). If it states a different value, the listing fails whatever the title says. Only when the listing has no attribute for that property does the title, the description or an attribute of another name that states the value count.
- A listing may carry "evidence", computed by the code, one entry per requirement: "states" = keys whose value states the required value; "other_value" = keys known to carry this kind of value that say something different here; "title_has" / "description_has" = the value words appear there. Follow it: "states" non-empty and "other_value" empty -> yes (for that requirement); "states" empty and "other_value" non-empty -> no, unless "literal_peers" is false (no listing writes the wanted value itself in that field) or "near_spelling" is true: then a close wording of the same value counts as yes, a different value as no; both non-empty -> unknown; both empty -> yes only if the title or description states the value, else no.
- Matching is by the meaning of the words: the product noun inside a value may be left out ("trench coat" is stated by style = trench; "wrap skirt" by skirt style = wrap skirts), and case, spacing, plural, word order and plain variants are fine ("circle skirts" meets circle, "anti-acne" and "acne & blemishes" meet acne care, "knit" meets knits). A value with different words does not count. When the listing's category label itself states the required value (a season or audience in the label), the category meets that requirement. Numbers are compared after converting units; limits include the edge.
- Category must be the requested kind of product as a whole (a neighbouring or partial label such as 'Jeans' for 'pants or jeans' does not count). Price must be within budget (equal is fine). Out of stock fails when stock is required.
- Answer "unknown" only when nothing can be told from the listing. Check every requirement, including ones the shopper stated in replies.
Return {"verdicts": [...]}."""

REPLY_SYSTEM = """A store assistant asked a shopper about one or more product properties. Read the shopper's reply and report, for every property asked, what the shopper said about it. Return only JSON:
{"answers": [{"property": "a property exactly as asked", "kind": "answer|indifferent|unclear", "value": "the wanted value when kind is answer, else null"}],
 "one_at_a_time": false, "extra": "any other need or change the reply mentions, else null"}
answer = states what they want for that property (or don't want; then put e.g. "not floral"). indifferent = says they have no preference on it, or that anything / the rest is fine.
A blanket reply that names no property ("I don't mind", "surprise me", "no opinion on that") is indifferent for every property asked.
A reply that names some properties and passes over the rest without a blanket remark: the passed-over ones are indifferent too, unless the shopper says they will come back to them.
unclear = only when the shopper did not understand or asks to rephrase. one_at_a_time = true only when the shopper objects to several questions at once or asks to go one by one.
Keep values short and in the shopper's words; a value that names several options keeps them all (e.g. "red or blue")."""

NOTE_SYSTEM = """A shopper sent updates during a shopping task. Return only JSON:
{"budget_total": new total budget number or null, "dropped_properties": ["requirements the shopper no longer needs"],
 "item_gone": true|false (the assistant's current pick or a cart item is no longer available),
 "gone_ids": ["ids from cart_ids or last_opened_ids that the updates say are gone, else empty"],
 "added_requirements": [{"item": index of the requested item, "property": "attribute name", "value": "the value now wanted, in the shopper's words", "replaces_property": "a requirement property this one replaces, else null"}],
 "new_quantity": {"item": index, "quantity": number} or null,
 "other": "anything else that changes the task, else null"}
Use added_requirements when the shopper now wants something new or different about the product (a new colour, a different size, an extra feature) and new_quantity when the amount changes. Leave both empty for updates that only say an item is gone or change the budget."""

PLAN_SYSTEM = """Choose which product properties to ask a shopper about. The shopper holds back a few needs, each about a property the store records on its listings, and reveals one only when asked about that property by name. Each question costs one of very few messages. Return only JSON: {"ask": [{"item": index, "property": "name exactly as in attributes_seen_on_candidates"}]} ordered most valuable first.
Pick from attributes_seen_on_candidates when it is not empty; otherwise name properties a store would record for this product. Prefer properties that (1) the story hints at, (2) describe what the product is or looks like, or who it is for - things a person holds a standing preference on - rather than per-order picks such as a size or a selectable option (variant_share near 1 means the value is chosen from the listing's options), logistics, or marketing claims, (3) many candidates record, (4) take a few different values across candidates so the answer changes the pick. Never list a stated requirement or a near-duplicate of one, and never two names for the same property."""

CATEGORY_SYSTEM = """Match each requested item to the store's category labels. Return only JSON: {"items": [{"item": index, "categories": ["label exactly as given"], "proposed_label": "label or null"}]}.
Choose the label whose name covers the WHOLE requested kind of product as the shopper phrases it ('Pants & Jeans' for 'pants or jeans'). A label that names only a part of the requested kind ('Jeans' for 'pants or jeans') or a neighbouring kind does not count. Give more than one label only when each equally names the whole requested kind. When no listed label names the whole kind, give an empty list and put in proposed_label the label the store most likely uses (store labels look like 'Pants & Jeans', 'Hoodies & Sweatshirts', 'Fall/Winter Clothing')."""


# ----------------------------------------------------------------- agent


class Episode:
    def __init__(self, problem_data: dict[str, Any]) -> None:
        env = problem_data["environment"]
        self.binding = env["binding"]
        policy = env["policy_view"]
        self.policy = policy
        self.clock = Clock()
        self.env = ProxyClient(timeout=ENV_TIMEOUT, max_retries=1, retry_delay=1)
        self.brain = Brain(self.clock)
        self.tools = [Tool(t) for t in policy.get("tools") or []]
        self.roles = Roles(self.tools)
        self.max_steps = int(policy.get("max_steps") or 30)
        request = policy.get("query") or problem_data.get("query") or ""
        if policy.get("opening") is not None:
            request += "\n\nOpening: " + _short(policy["opening"], 3000)
        self.ledger = Ledger(request, self.max_steps)
        self.catalog = Catalog()
        self.step = 0
        self.finished = bool(policy.get("finished"))
        self.dialogue: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.calls_seen: dict[str, int] = {}
        self.change_tick = 0
        self.inbox: list[str] = []
        self.pending: list[str] = []
        self.detail_stock: dict[str, bool] = {}
        self.detail_tried: set[str] = set()
        self.attr_tools: set[str] = set()
        self.target_cats: dict[int, set[str]] = {}
        self.labels_shown: set[str] = set()
        self.category_calls = 0
        self.judge_tried: set[tuple[str, str]] = set()
        self.gate_rejections = 0
        self.decline_rejections = 0
        self.cheaper_checked: set[str] = set()
        self.question_plan: list[dict[str, Any]] = []
        self.planned = False
        self.replan = False
        self.last_note_step = 0
        self.cart_step: dict[str, int] = {}
        self.target_labels: list[str] = []
        self.auto_opened = 0
        self.weak_gate = 0
        self.sellout_save = 0
        self.swept: set[tuple[str, float]] = set()
        self.label_probed = False
        self.sweeps = 0
        self.blind_looks = 0
        self.twin_done: set[str] = set()
        self.late_asked = False
        self.explored = False
        self.converted = 0
        self.plan_opened = 0
        self.plans = 0
        self.unchecked_gate = 0
        self.ask_width = ASK_WIDTH
        self.refusals = 0
        self.rescuing = False
        self.cart_touches: dict[str, int] = {}

    # ---- store

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def call(self, name: str, args: dict[str, Any]) -> tuple[Any, list[str]]:
        if self.finished:
            return {"error": "episode finished"}, []
        self.step += 1
        result = None
        for _ in range(3):
            # a store call must return while there is still time to finish; the step number makes a retry safe
            self.env.timeout = max(4, int(min(ENV_TIMEOUT, self.clock.left() - 6)))
            payload = {**self.binding, "step": self.step, "action": {"name": name, "args": args}}
            verbose = getattr(self.env, "post_verbose", None)
            if verbose is not None:
                reply = verbose("/environment/call", json_data=payload)
                result, err_info = reply.data, getattr(reply, "error", None)
            else:
                result, err_info = self.env.post("/environment/call", json_data=payload), None
            if result is not None:
                break
            if self.clock.left() < 15:
                break
            err = err_info or {}
            # the store says which step number it expects next: use it instead of guessing
            expected = re.search(r"expected\D{0,12}(\d+)", str(err.get("body") or ""), re.I)
            if err.get("status") == 409 and expected:
                self.step = int(expected.group(1))
                continue
            time.sleep(3 if err.get("status") in (429, 502, 503, 504) else 1)
        if result is None:
            self.dialogue.append({"role": "environment", "action": name, "args": args, "error": "no response"})
            return {"error": "the store did not respond"}, []
        if isinstance(result.get("step"), int):
            self.step = max(self.step, result["step"])
        self.finished = bool(result.get("finished"))
        self.dialogue.append({"role": "environment", "action": {"name": name, "args": args}, "environment_result": result})
        obs, notes = _split_notes(result.get("observation"))
        fresh = [n for n in notes if n not in self.ledger.notes]
        if fresh:
            self.ledger.notes.extend(fresh)
            self.pending.extend(fresh)
            self.inbox.extend("Shopper update: " + n for n in fresh)
            self.change_tick += 1
            self.last_note_step = self.step
        detail = name in self.roles.details
        if detail and not _is_error(obs):
            shows = any(_STOCK_KEYS.search(k) for k in re.findall(r'"([^"]{1,40})"\s*:', json.dumps(obs, default=str)[:4000]))
            self.detail_stock[name] = self.detail_stock.get(name, False) or shows
        if not _is_error(obs):
            before = sum(len(it.traits) for it in self.catalog.items.values())
            self.catalog.absorb(obs, self.step, detail)
            self._track_cart(name, args, obs)
            if detail:
                self.detail_tried.add(name)
                if sum(len(it.traits) for it in self.catalog.items.values()) > before:
                    self.attr_tools.add(name)
                if not self.attr_tools and set(self.roles.details) <= self.detail_tried:
                    for it in self.catalog.items.values():
                        it.detailed = it.detailed or it.viewed
        return obs, fresh

    def _track_cart(self, name: str, args: dict[str, Any], obs: Any) -> None:
        mode = self.roles.cart_change.get(name)
        if mode:
            lid = str(args.get(mode["id"], ""))
            qty = int(args.get(mode["qty"]) or 1) if mode.get("qty") else 1
            adding = mode["add"] is True or (mode["op"] and args.get(mode["op"]) == mode["add"])
            removing = mode["remove"] is True or (mode["op"] and args.get(mode["op"]) == mode["remove"])
            if adding and lid:
                self.ledger.cart[lid] = self.ledger.cart.get(lid, 0) + qty
                self.cart_step[lid] = self.step
            elif removing and lid:
                left = self.ledger.cart.get(lid, 0) - (qty if mode.get("qty") and mode["qty"] in args else 10 ** 6)
                if left > 0:
                    self.ledger.cart[lid] = left
                else:
                    self.ledger.cart.pop(lid, None)
            return
        if name == self.roles.cart_view:
            lines = self._cart_lines(obs)
            if lines is not None:
                self.ledger.cart = lines

    def _cart_lines(self, obs: Any) -> dict[str, int] | None:
        lines: dict[str, int] = {}
        lists = {"empty": 0, "full": 0}

        def walk(node: Any, depth: int) -> None:
            if depth > 5:
                return
            if isinstance(node, list):
                lists["full" if node else "empty"] += 1
                for x in node:
                    walk(x, depth + 1)
            elif isinstance(node, dict):
                lid = _id_of(node)
                if lid:
                    qty = _qty_of(node)
                    lines[lid] = lines.get(lid, 0) + qty
                    return
                for v in node.values():
                    walk(v, depth + 1)

        walk(obs, 0)
        if lines:
            return lines
        if lists["empty"] and not lists["full"]:
            return {}
        return None

    # ---- brief

    def brief(self) -> None:
        tool_text = json.dumps([t.spec for t in self.tools], ensure_ascii=False)[:12000]
        user = f"REQUEST:\n{self.ledger.request}\n\nmax_steps: {self.max_steps}\n\nTOOLS:\n{tool_text}"
        data = self.brain.ask_json("heavy", BRIEF_SYSTEM, user, max_tokens=1800, critical=True, cap=50)
        if not isinstance(data, dict) or not data.get("items"):
            data = self.brain.ask_json("light", BRIEF_SYSTEM, user, max_tokens=1800, critical=True, cap=40) or {}
        led = self.ledger
        items = data.get("items") if isinstance(data, dict) else None
        led.items = [i for i in (items or []) if isinstance(i, dict)] or [{"what": "the requested product", "quantity": 1, "requirements": []}]
        for it in led.items:
            try:
                it["quantity"] = max(1, int(it.get("quantity") or 1))
            except (TypeError, ValueError):
                it["quantity"] = 1
            for r in it.get("requirements") or []:
                if isinstance(r, dict) and r.get("property"):
                    led.stated_props.add(_norm(r["property"]))
        led.budget = _num(data.get("budget_total")) if isinstance(data, dict) else None
        led.budget_each = isinstance(data, dict) and str(data.get("budget_scope") or "").lower() == "each"
        if led.budget is None:
            led.budget = self._budget_from_text()
        led.optimize = str(data.get("optimize") or "none").lower() if isinstance(data, dict) else "none"
        if led.optimize == "none" and re.search(r"cheapest|costs? the least|lowest price|least expensive", led.request, re.I):
            led.optimize = "cheapest"
        led.optimize_detail = str(data.get("optimize_detail") or "")
        led.need_stock = bool(data.get("must_be_in_stock", True)) if isinstance(data, dict) else True
        qb = data.get("question_budget") if isinstance(data, dict) else None
        led.question_budget = int(qb) if isinstance(qb, (int, float)) else self._questions_from_text()
        led.decline_rule = str(data.get("decline_rule") or "")
        led.procedure = [p for p in (data.get("procedure") or []) if isinstance(p, dict) and p.get("action")
                         and not re.search(r"come up empty|nothing (?:was )?(?:found|fits|matches)|no substitute|say so",
                                           str(p["action"]), re.I)]
        led.hint_props = [str(p) for p in (data.get("hint_properties") or []) if p]
        if re.search(r"not fussy about|all I asked for|is all I (?:care|need|want)|nothing else matters|any (?:pair|one|[a-z]+) that [^.]{0,40}will do",
                     led.request, re.I):
            led.question_budget = 0     # the shopper says nothing else matters: no message can reveal a need
        self.roles.apply_brief(data.get("roles") or {})

    def _budget_from_text(self) -> float | None:
        m = re.search(r"(?:over|above|exceed|past|more than|max(?:imum)?|ceiling|cap|budget|tops out at|up to|spend)\D{0,25}(\d[\d,]*(?:\.\d+)?)",
                      self.ledger.request, re.I)
        return _num(m.group(1)) if m else None

    def _questions_from_text(self) -> int | None:
        m = re.search(r"(\d+)\s+(?:messages|questions|texts|notes)", self.ledger.request, re.I)
        if m:
            return int(m.group(1))
        if re.search(r"(?:no|don't|do not) (?:ask|message|questions)", self.ledger.request, re.I):
            return 0
        return None

    # ---- shopper

    def questions_left(self) -> int:
        led = self.ledger
        if not self.roles.message or led.questions_refused:
            return 0
        budget = led.question_budget if led.question_budget is not None else 3
        reserve = sum(1 for p in led.procedure if re.search(r"message|tell|inform|confirm|let me know|ask", p["action"], re.I)
                      and p.get("when") in ("before_order", "before_decline"))
        return max(0, budget - led.questions_used - reserve)

    def candidate_props(self) -> list[dict[str, Any]]:
        values: dict[str, set[str]] = {}
        listings: dict[str, int] = {}
        variant: dict[str, int] = {}
        pool = [it for it in self.catalog.items.values() if it.traits and not self.category_fail(it)]
        for item in pool:
            for prop in {_prop_key(k) for k in item.traits}:
                listings[prop] = listings.get(prop, 0) + 1
            # a value that also sits in the listing's selectable options is picked per purchase, not a standing taste
            options = " ".join(_norm(v) for k, v in item.traits.items() if re.search(r"^options?$|variant", _prop_key(k)))
            for key, value in item.traits.items():
                prop = _prop_key(key)
                if prop and not re.search(r"^(?:condition|options?|id|sku|price|stock|warranty.*)$|deliver|ship|dispatch|return",
                                          prop):
                    norm = _norm(value)[:40]
                    values.setdefault(prop, set()).add(norm)
                    if options and len(norm) > 1 and re.search(rf"(?<!\w){re.escape(norm)}(?!\w)", options):
                        variant[prop] = variant.get(prop, 0) + 1
        stated = self.ledger.stated_props | {_norm(a) for a in self.ledger.asked}
        rows = [{"property": p, "listings_recording_it": listings.get(p, 0), "distinct_values": len(v),
                 "examples": sorted(v)[:5], "variant_share": variant.get(p, 0) / max(listings.get(p, 0), 1)}
                for p, v in values.items() if not any(p == s or p in s or s in p for s in stated if s)]
        rows.sort(key=lambda r: -(r["listings_recording_it"] * min(r["distinct_values"], 4)))
        return rows[:30]

    def lock_categories(self) -> None:
        """Pick the category label that names the requested kind, once labels are visible."""
        labels: dict[str, int] = {}
        for it in self.catalog.items.values():
            if it.category:
                labels[it.category] = labels.get(it.category, 0) + 1
        fresh = set(labels) - self.labels_shown
        if not fresh or self.category_calls >= 3:
            return
        self.category_calls += 1
        self.labels_shown |= set(labels)
        data = self.brain.ask_json("heavy", CATEGORY_SYSTEM, json.dumps({
            "request": self.ledger.request,
            "items": [{"item": i, "what": it.get("what"), "category_hint": it.get("category_hint")}
                      for i, it in enumerate(self.ledger.items)],
            "labels_with_listing_counts": labels}, ensure_ascii=False), max_tokens=900, cap=25)
        proposed = None
        for row in (data or {}).get("items", []) if isinstance(data, dict) else []:
            if not isinstance(row, dict):
                continue
            chosen = {c for c in row.get("categories") or [] if c in labels}
            if chosen:
                self.target_cats[int(row.get("item") or 0)] = {_norm(c) for c in chosen}
                self.target_labels.extend(c for c in sorted(chosen) if c not in self.target_labels)
            elif row.get("proposed_label") and not proposed:
                proposed = str(row["proposed_label"]).strip()
        if not self.target_cats and proposed:
            self.probe_label(proposed)
        if self.target_cats:
            self.change_tick += 1

    def probe_label(self, label: str) -> None:
        """No label seen so far names the whole kind: try the label the store most likely uses (one step)."""
        found = self.filter_search()
        if self.label_probed or not found or self.steps_left() <= self.reserve() + 4:
            return
        self.label_probed = True
        t, mode_p, fv, cat_p, cap_p = found
        args: dict[str, Any] = {cat_p: label}
        if mode_p:
            args[mode_p] = fv
        if self.ledger.budget:
            args[cap_p] = int(self.ledger.budget) if (t.props[cap_p] or {}).get("type") == "integer" else self.ledger.budget
        if t.check(args):
            return
        obs, _ = self.call(t.name, args)
        self.ledger.searches += 1
        rows = [it for it in self.catalog.items.values() if _norm(it.category) == _norm(label)]
        if rows:
            self.target_cats[0] = {_norm(label)}
            self.target_labels.append(rows[0].category)
            self.labels_shown.add(rows[0].category)

    def category_fail(self, item: Listing) -> bool:
        if not self.target_cats or not item.category:
            return False
        allowed = set().union(*self.target_cats.values())
        return _norm(item.category) not in allowed

    def plan_questions(self) -> None:
        self.planned = True
        led = self.ledger
        cands = self.candidate_props()
        if cands:
            # Ask first about properties candidates actually record, that differ between them,
            # and that the story already hinted at (hints come from the request, not a fixed table).
            pool = max(c["listings_recording_it"] for c in cands) or 1
            hint_norms = [_norm(h) for h in led.hint_props if h]
            hint_words = {w for h in hint_norms for w in h.split() if len(w) > 2}

            def hint_boost(prop: str) -> float:
                pw = set(prop.split())
                best = 0.0
                for h in hint_norms:
                    hw = set(h.split())
                    if not hw:
                        continue
                    if hw == pw or hw <= pw or pw <= hw:
                        best = max(best, 1.2)
                    elif pw & hw:
                        best = max(best, 0.6)
                if not best and pw & hint_words:
                    best = 0.4
                return best

            def rank(c: dict[str, Any]) -> float:
                score = 0.4 * c["listings_recording_it"] / pool + hint_boost(c["property"]) - 0.5 * c.get("variant_share", 0)
                return score - (0.8 if c["distinct_values"] < 2 else 0)

            ranked = sorted(cands, key=rank, reverse=True)
            data = self.brain.ask_json("heavy", PLAN_SYSTEM, json.dumps({
                "request": led.request, "items": [it.get("what") for it in led.items],
                "stated_requirements": led.requirements(), "story_hints": led.hint_props,
                "attributes_seen_on_candidates": ranked[:20], "questions_available": self.questions_left(),
            }, ensure_ascii=False, default=str), max_tokens=1000, cap=30)
            known = {c["property"] for c in cands}
            picked = [q for q in (data.get("ask") or [] if isinstance(data, dict) else [])
                      if isinstance(q, dict) and _norm(q.get("property")) in known]
            plan = picked + [{"item": 0, "property": c["property"]} for c in ranked]
        else:
            self.replan = True
            user = json.dumps({
                "request": led.request, "items": [it.get("what") for it in led.items],
                "stated_requirements": led.requirements(), "story_hints": led.hint_props,
                "attributes_seen_on_candidates": [], "questions_available": self.questions_left(),
            }, ensure_ascii=False)
            data = self.brain.ask_json("heavy", PLAN_SYSTEM, user, max_tokens=1200, cap=30)
            plan = data.get("ask") if isinstance(data, dict) else None
            plan = [q for q in plan or [] if isinstance(q, dict)]
        # Story-derived hints first when listings record them: a held-back need is about a recorded property,
        # so a hint no listing records is asked only after the recorded ones.
        # with listings to compare, the ranking above already weighed the hints; without them, hints lead
        hints = [{"item": 0, "property": p} for p in led.hint_props if p]
        plan = plan + hints if cands else hints + plan
        if led.budget is None and "budget" not in {_norm(a) for a in led.asked}:
            plan = [{"item": 0, "property": "budget"}] + plan     # no spending limit was stated: that is asked first
        generic = {"type", "types", "kind", "general", "class", "value", "filter", "of", "the", "for"}

        def words(p: str) -> set[str]:
            return {w for w in p.split() if w not in generic} or set(p.split())

        names = [c["property"] for c in cands]
        asked = {_norm(a) for a in led.asked}
        seen: list[set[str]] = [words(a) for a in asked]
        self.question_plan = []
        for q in plan:
            if not isinstance(q, dict) or not q.get("property"):
                continue
            prop = _norm(q["property"])
            if prop in led.stated_props or prop in asked or any(words(prop) & s for s in seen):
                continue
            seen.append(words(prop))
            prop = min((n for n in names if words(n) <= words(prop)), key=len, default=prop)
            self.question_plan.append({"item": int(q.get("item") or 0) if str(q.get("item", "0")).isdigit() else 0,
                                       "property": prop})

    def ask_next(self) -> bool:
        """Send one planned question. Return True when a step was used."""
        if not self.question_plan or self.questions_left() <= 0 or self.clock.elapsed() > ASK_UNTIL:
            return False
        if self.steps_left() <= self.reserve() + 1:
            return False
        first = self.question_plan.pop(0)
        idx = first["item"] if first["item"] < len(self.ledger.items) else 0
        # a held-back need shows only when its property is named, so one message names several properties of one item
        # the shopper reads each message as being about one property, so a list can hide the one that matters;
        # only the last message, which has nothing left to lose, names several
        width = self.ask_width if self.questions_left() <= 1 else 1
        group = [first]
        for q in list(self.question_plan):
            if len(group) >= width or first["property"] == "budget" or q["property"] == "budget":
                break
            if (q["item"] if q["item"] < len(self.ledger.items) else 0) == idx:
                group.append(q)
                self.question_plan.remove(q)
        props = [q["property"] for q in group]
        what = re.sub(r"\s+and\s+", " or ", str(self.ledger.items[idx].get("what") or "item"))   # "and" draws one-at-a-time replies
        if props == ["budget"]:
            text = f"What is the most you would like to spend on the {what}?"
        elif len(props) == 1:
            text = f"What {props[0]} would you like for the {what}?"
        else:
            text = f"For the {what}, what would you like for each of these: {', '.join(props[:-1])} and {props[-1]}?"
        tool, arg = self.roles.message  # type: ignore[misc]
        limit = int((self.roles.tools[tool].props.get(arg) or {}).get("maxLength") or 600)
        obs, _ = self.call(tool, {arg: text[:limit]})
        self.ledger.asked.extend(props)
        reply = obs if isinstance(obs, str) else _short(obs, 600)
        if _is_error(obs) or re.search(r"budget|limit|no more (?:messages|questions)|refus", reply, re.I) and not isinstance(obs, str):
            self.ledger.questions_refused = True
            self.inbox.append(f"(Your question about {', '.join(props)} was refused by the store: {reply[:200]})")
            return True
        self.ledger.questions_used += 1
        if props == ["budget"]:
            figure = self._note_budget([reply + " budget"]) or _num((re.search(r"\d[\d,]*(?:\.\d+)?", reply) or [None])[0])
            if figure and figure > 0:
                self.ledger.budget = figure
                self.change_tick += 1
                self.inbox.append(f"The shopper's budget is {figure:g}.")
            return True
        verdict = self.brain.ask_json("light", REPLY_SYSTEM,
                                      f"Properties asked (for the {what}): {json.dumps(props)}\nShopper reply: {reply}",
                                      max_tokens=900, critical=True, cap=20)
        if not isinstance(verdict, dict):
            self.inbox.append(f"Shopper replied about {', '.join(props)} for the {what}: \"{reply[:300]}\" (read it yourself: "
                              "a stated preference is a requirement)")
            return True
        rows = {_norm(r.get("property")): r for r in verdict.get("answers") or [] if isinstance(r, dict)}
        unclear: list[dict[str, Any]] = []
        for q in group:
            row = rows.get(_norm(q["property"])) or (next(iter(rows.values())) if len(group) == 1 and rows else {})
            kind = str(row.get("kind") or "unclear")
            if kind not in ("answer", "indifferent"):
                unclear.append(q)
                continue
            record = {"item": idx, "property": q["property"], "kind": kind, "value": row.get("value"), "reply": reply[:300]}
            self.ledger.answers.append(record)
            if kind == "answer":
                self.change_tick += 1
                self.inbox.append(f"Shopper said about {q['property']} for the {what}: \"{reply[:300]}\" -> requirement: "
                                  f"{record['value']}. Search that value as a short keyword next.")
            else:
                self.inbox.append(f"Shopper has no preference on {q['property']}.")
        refused_list = bool(verdict.get("one_at_a_time"))
        if refused_list:
            self.ask_width = 1
        if unclear and len(group) > 1:
            # a shopper who wants one question at a time gets these again first; one who passed over them likely does not mind
            if refused_list:
                self.question_plan[:0] = unclear
            else:
                self.question_plan.extend(unclear)
            for q in unclear:
                if q["property"] in self.ledger.asked:
                    self.ledger.asked.remove(q["property"])
        if verdict.get("extra"):
            self.inbox.append(f"Shopper also said: {verdict['extra']}")
        return True

    @staticmethod
    def _note_budget(notes: list[str]) -> float | None:
        """A budget change names the new limit as a plain number; read it from the text, not from a model."""
        for note in notes:
            if not re.search(r"spend|afford|budget|work(?:ing)? with|limit|cap\b|tight|money|allowance|cts|cents|can only|all I (?:can|have)", note, re.I):
                continue
            if re.search(r"gone|zero left|unavailable|sold out|ran dry|all gone|no longer|grayed|closed on", note, re.I):
                continue
            m = re.search(r"(\d[\d,]*)(?:\.\d+)?\s*(?:cts|ct|cents|c\b)?", note)
            if m:
                value = _num(m.group(1))
                if value and value > 0:
                    return value
        return None

    def flush_notes(self) -> None:
        notes, self.pending = self.pending, []
        if not notes:
            return
        led = self.ledger
        recent = sorted((it for it in self.catalog.items.values() if it.detailed),
                        key=lambda it: -it.refreshed_step)[:3]
        data = self.brain.ask_json("light", NOTE_SYSTEM, json.dumps({
            "updates": notes, "current_budget": led.budget, "requirements": led.requirements(),
            "cart_ids": list(led.cart), "last_opened_ids": [it.id for it in recent]},
            ensure_ascii=False), max_tokens=900, critical=True, cap=20)
        if not isinstance(data, dict):
            return
        nb = _num(data.get("budget_total"))
        stated = self._note_budget(notes)
        if stated is not None:
            nb = stated      # the model misreads "507 cts. is" as 5.07; the figure in the text is the truth
        if nb is not None and nb > 0:
            led.budget = nb
        for prop in data.get("dropped_properties") or []:
            if prop and prop not in led.dropped:
                led.dropped.append(str(prop))
        for row in data.get("added_requirements") or []:
            if not isinstance(row, dict) or not row.get("property") or not row.get("value"):
                continue
            idx = int(row["item"]) if str(row.get("item", "0")).isdigit() and int(row["item"]) < len(led.items) else 0
            old = row.get("replaces_property")
            if old and str(old) not in led.dropped:
                led.dropped.append(str(old))
            led.answers.append({"item": idx, "property": str(row["property"]), "kind": "answer",
                                "value": str(row["value"]), "reply": "shopper update"})
            self.inbox.append(f"Shopper now also wants {row['property']}: {row['value']}"
                              + (f" (instead of {old})" if old else "") + ". Check the cart against it.")
        nq = data.get("new_quantity")
        if isinstance(nq, dict) and led.items:
            q = _num(nq.get("quantity"))
            idx = int(nq["item"]) if str(nq.get("item", "0")).isdigit() and int(nq["item"]) < len(led.items) else 0
            if q and 1 <= q <= 99:
                led.items[idx]["quantity"] = int(q)
                self.inbox.append(f"The shopper now wants quantity {int(q)} of the {led.items[idx].get('what') or 'item'}: "
                                  "set the cart line to that amount.")
        gone = [str(g) for g in data.get("gone_ids") or [] if str(g) in self.catalog.items or str(g) in led.cart]
        if not gone and data.get("item_gone") and len(led.cart) == 1:
            gone = list(led.cart)
        for lid in gone:
            if lid in self.catalog.items:
                self.catalog.items[lid].stock = False
            led.unavailable.add(lid)
            self.inbox.append(f"{lid} is no longer available: remove it from the cart and choose another.")
        self.change_tick += 1

    # ---- evidence: what the listing's own attributes say about each stated requirement

    _GENERIC = {"type", "style", "kind", "general", "class", "value", "filter", "name", "feature", "option"}
    _COMPLEX = re.compile(r"\b(?:at (?:most|least)|under|over|below|above|less|more|max|min|up to|between|within|"
                          r"same as|like|unlike|not|no|except|without)\b|[<>]", re.I)

    def split_reqs(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Requirements the code can check by words, and the rest (limits, ranges, references) for the model."""
        simple: list[dict[str, Any]] = []
        rest: list[dict[str, Any]] = []
        for r in self.ledger.requirements():
            value = r.get("value")
            kind = str(r.get("kind") or "must").lower()
            words = _words(value) if isinstance(value, str) else set()
            if len(words) > 1:
                # "trench coat" is stated by an attribute that says "trench": the product noun may be left out
                words = (words - self._kind_words()) or (words - self._KIND_NOUNS) or words
            if kind in ("must", "exclude") and words and not self._COMPLEX.search(str(value)):
                simple.append({**r, "words": words})
            else:
                rest.append(r)
        return simple, rest

    _KIND_NOUNS = {"coat", "jacket", "skirt", "pant", "jean", "top", "shirt", "dress", "watch", "clock", "shoe", "sandal", "boot",
                   "bag", "milk", "sweater", "hoodie", "short", "set", "cloth", "clothing", "wear", "item", "product", "hoody"}

    def _kind_words(self, fixed: bool = True) -> set[str]:
        out = set(self._KIND_NOUNS) if fixed else set()
        for it in self.ledger.items:
            out |= _words(it.get("what")) | _words(it.get("category_hint"))
        return out

    def _domain(self, words: set[str]) -> set[str]:
        """Attribute names that, on some listing read so far, state this value: the keys that can carry it."""
        keys: set[str] = set()
        for it in self.catalog.items.values():
            for k, v in it.traits.items():
                if words <= _words(v):
                    keys.add(_prop_key(k))
        return keys

    def evidence(self, item: Listing, reqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for r in reqs:
            w = r["words"]
            dom = self._domain(w)
            name_words = (_words(r.get("property")) - self._GENERIC) or _words(r.get("property"))
            states: list[str] = []
            other: list[tuple[str, Any]] = []
            for k, v in item.traits.items():
                pk = _prop_key(k)
                if w <= _words(v):
                    states.append(k)
                elif (pk in dom) if dom else bool(_words(pk) & name_words):
                    other.append((k, v))
            stated_names = {_prop_key(k) for k in states}
            other = [(k, v) for k, v in other if _prop_key(k) not in stated_names]
            # a key that joins the product kind and the property ("skirt style" for a skirt's style) outranks a bare "style"
            kind = self._kind_words(fixed=False)
            pw_all = _words(r.get("property"))

            def specific(k: str) -> bool:
                kw = _words(_prop_key(k))
                return bool(kw & kind) and bool(pw_all) and pw_all <= kw
            spec_states = [k for k in states if specific(k)]
            spec_other = [(k, v) for k, v in other if specific(k)]
            if spec_states and not spec_other:
                other = []
            elif spec_other and not spec_states:
                states = []
            exact = any(w == _words(part) for k in states for part in re.split(r"[,;/]", str(item.traits.get(k, ""))))
            pwords = _words(r.get("property"))
            if exact and pwords and not any(_words(_prop_key(k)) == pwords or _words(_prop_key(k)) >= (pwords - self._GENERIC) > set()
                                            for k in states if _words(_prop_key(k))):
                exact = False      # only a differently named attribute states it: a weaker match
            if w <= _words(item.category):
                states, other, exact = ["category"] + states, [], True   # the category label states it (the kind itself)
            close = any(_near_words(w, _words(v)) for k, v in other)
            out.append({
                "requirement": f"{r.get('property')}: {r.get('value')}" + (" (must NOT have)" if r.get("kind") == "exclude" else ""),
                "states": states[:4],
                "exact": exact,
                # a different value only rules a listing out when other listings of the store write the wanted value
                # itself in that field, and this one is not a near spelling of it
                "literal_peers": bool(dom),
                "near_spelling": close,
                "other_value": [f"{k}={_short(v, 40)}" for k, v in other[:4]],
                "title_has": w <= _words(item.title),
                "description_has": w <= _words(item.desc),
            })
        return out

    def code_verdict(self, item: Listing) -> dict[str, Any] | None:
        """A fit decided by the listing's attributes alone; None when a model has to read it."""
        if not item.detailed:
            return None
        simple, rest = self.split_reqs()
        if rest:
            return None
        if not simple:
            return {"fit": "yes", "why": "no property requirement to check", "by": "code", "strength": 2}
        strength = 2
        notes = []
        for r, e in zip(simple, self.evidence(item, simple)):
            if r.get("kind") == "exclude":
                if e["states"]:
                    return {"fit": "no", "why": f"{e['requirement']} but {e['states'][0]} states it", "by": "code"}
                continue
            if e["states"] and not e["other_value"]:
                clash = self._shape_clash(item, r["words"])
                if clash:
                    return {"fit": "no", "why": f"the description states the shape as {clash}, not {r.get('value')}", "by": "code"}
                notes.append(f"{e['states'][0]} states {r.get('value')}")
                if not e["exact"]:
                    strength = min(strength, 1.4)     # the value is only part of a longer phrase: a weaker match
            elif e["states"]:
                return None
            elif e["other_value"]:
                if not e["literal_peers"] or e["near_spelling"]:
                    return None      # a close wording may still be the same value: a model reads it
                return {"fit": "no", "why": f"{e['other_value'][0]} instead of {r.get('value')}", "by": "code"}
            elif e["title_has"] or e["description_has"]:
                strength = 1
                notes.append(f"title/description states {r.get('value')}, no attribute for it")
            else:
                return None
        return {"fit": "yes", "why": "; ".join(notes), "by": "code", "strength": strength}

    _SHAPES = {"square", "rectangle", "rectangular", "round", "circular", "oval", "diamond"}

    def _shape_clash(self, item: Listing, words: set[str]) -> str | None:
        """The description names a shape the attributes never state, against a shape requirement."""
        if not (words & self._SHAPES):
            return None
        blob = f"{item.title} {item.desc}".lower()
        attr = set().union(*(_words(v) for v in item.traits.values())) if item.traits else set()
        for m in re.finditer(r"shape\s*[:\-]\s*([a-z][a-z \-]{1,40})", blob):
            said = {s for tok in re.findall(r"[a-z]+", m.group(1)) for s in self._SHAPES if tok.startswith(s)}
            if said and said.isdisjoint(words) and said.isdisjoint(attr):
                return m.group(1).strip()[:24]
        return None

    # ---- candidates: rows read from search results, ranked by what their words already show

    def screen(self, it: Listing) -> float:
        simple, _ = self.split_reqs()
        text = _words(it.title) | _words(it.desc)
        score = 0.0
        for r in simple:
            if r.get("kind") == "exclude":
                continue
            w = r["words"]
            if w <= text:
                score += 1
            elif len(w & text) * 2 >= len(w):
                score += 0.4
        return score

    def candidates(self, n: int = 6, below: float | None = None) -> list[Listing]:
        pool = [it for it in self.catalog.items.values()
                if not it.detailed and not self.hard_fail(it) and it.price is not None
                and (below is None or it.price < below)]
        cheapest = self.ledger.optimize == "cheapest" or below is not None
        pool.sort(key=lambda it: (-self.screen(it), it.price or 0) if not cheapest else (-(self.screen(it) >= 1), it.price or 0))
        return pool[:n]

    def attr_tool(self) -> Tool | None:
        for name in sorted(self.attr_tools):
            t = self.roles.tools.get(name)
            if t and t.required:
                return t
        return None

    def open_listings(self, ids: list[str]) -> int:
        """Read the attribute list of each listing; return how many steps were used."""
        used = 0
        for lid in ids:
            if self.finished or self.steps_left() <= self.reserve() + 1 or self.clock.elapsed() > WRAP_AT - 20:
                break
            tool = self.attr_tool()
            if tool is None:
                # which single-listing tool lists attributes is not known yet: try each once
                untried = [n for n in self.roles.details if n not in self.detail_tried and self.roles.tools[n].required]
                if not untried:
                    break
                tool = self.roles.tools[untried[0]]
            self.call(tool.name, {tool.required[0]: lid})
            self.auto_opened += 1
            used += 1
        return used

    def auto_open(self, limit: int = 3) -> None:
        """After a search, read the best-matching unread rows so the model starts from checked facts."""
        if self.auto_opened >= 9 or self.finished:
            return
        if self.ledger.cart:
            return
        key = self.sync_verdicts()
        have_fit = any(v.get("fit") == "yes" for lid, v in self.ledger.verdicts.items()
                       if lid in self.catalog.items and not self.hard_fail(self.catalog.items[lid]))
        if have_fit:
            return
        picks = [it.id for it in self.candidates(limit) if self.screen(it) >= 0.4 or self.ledger.optimize == "cheapest"]
        if picks:
            self.open_listings(picks[:limit])

    # ---- fit checks

    def sync_verdicts(self) -> str:
        """Verdicts hold only for the ledger they were made under."""
        key = self.ledger.key()
        if key != self.ledger.verdict_key:
            self.ledger.verdicts = {}
            self.ledger.verdict_key = key
        # what the attributes say can change as more listings reveal which keys carry a value
        for lid, v in list(self.ledger.verdicts.items()):
            if v.get("by") != "code" or lid not in self.catalog.items:
                continue
            fresh = self.code_verdict(self.catalog.items[lid])
            if fresh is None:
                del self.ledger.verdicts[lid]
                self.judge_tried.discard((lid, key))
            elif fresh["fit"] != v.get("fit") or fresh.get("strength") != v.get("strength"):
                self.ledger.verdicts[lid] = {"id": lid, **fresh}
        return key

    def judge(self, ids: list[str], role: str = "light") -> dict[str, dict[str, Any]]:
        self.sync_verdicts()
        todo = [i for i in ids if i in self.catalog.items and (i not in self.ledger.verdicts or role == "heavy")]
        if todo:
            coded = []
            for i in todo:
                known = self.ledger.verdicts.get(i)
                if known and known.get("by") == "code":
                    continue
                v = self.code_verdict(self.catalog.items[i])
                if v:
                    self.ledger.verdicts[i] = {"id": i, **v}
                    coded.append(i)
            if coded:
                self.dialogue.append({"role": "agent", "judge": "code", "verdicts": [
                    {k: self.ledger.verdicts[i].get(k) for k in ("id", "fit", "why")} for i in coded]})
            todo = [i for i in todo if i not in coded and self.ledger.verdicts.get(i, {}).get("by") != "code"]
        if todo:
            simple, _ = self.split_reqs()
            cards = [self.catalog.items[i].card(evidence=self.evidence(self.catalog.items[i], simple) if simple else None)
                     for i in todo[:10]]
            data = self.brain.ask_json(role, JUDGE_SYSTEM, json.dumps({
                "shopper": self.ledger.summary(), "listings": cards}, ensure_ascii=False, default=str),
                max_tokens=1200, critical=role == "light", cap=35)
            for v in (data or {}).get("verdicts", []) if isinstance(data, dict) else []:
                if isinstance(v, dict) and str(v.get("id")) in self.catalog.items:
                    v["strength"] = 1.5 if v.get("fit") == "yes" else 0
                    self.ledger.verdicts[str(v["id"])] = v
            self.dialogue.append({"role": "agent", "judge": role, "verdicts": [
                {k: self.ledger.verdicts[i].get(k) for k in ("id", "fit", "why")}
                for i in todo[:10] if i in self.ledger.verdicts]})
        return {i: self.ledger.verdicts[i] for i in ids if i in self.ledger.verdicts}

    def hard_fail(self, item: Listing) -> str | None:
        led = self.ledger
        if item.id in led.unavailable or (led.need_stock and item.stock is False):
            return "not in stock"
        if led.budget is not None and item.price is not None and item.price > led.budget + 1e-9:
            return f"price {item.price} is over the budget {led.budget}"
        if self.category_fail(item):
            wanted = sorted(set().union(*self.target_cats.values()))
            return f"category '{item.category}' is not the requested kind {wanted}"
        return None

    def refresh_tool(self) -> Tool | None:
        """The single-listing tool that reports stock, so a re-read catches a sell-out."""
        names = [n for n in self.roles.details if self.roles.tools[n].required]
        if not names:
            return None
        names.sort(key=lambda n: (not self.detail_stock.get(n, False),
                                  not re.search(r"full|view|page|offer|stock", self.roles.tools[n].text, re.I)))
        return self.roles.tools[names[0]]

    def reserve(self) -> int:
        procedure_msgs = sum(1 for p in self.ledger.procedure if p.get("when") in ("before_order", "before_decline"))
        # the first checkout often finds a cart line sold out: keep the steps a swap and a second checkout take
        swap_room = SELLOUT_ROOM if self.refusals == 0 and self.roles.cart_change and not self.rescuing else 0
        return 2 + min(2, len(self.ledger.cart)) + procedure_msgs + swap_room

    # ---- finishing

    def ask_before_finish(self) -> str | None:
        """Messages the request allows are free of steps' worth only up to the reserve: a held-back need is invisible until
        asked, so use the unused questions before the order or the decline is sent (once)."""
        if self.late_asked or not self.roles.message or self.questions_left() <= 0 or self.finished:
            return None
        if self.steps_left() <= self.reserve() + 1 or self.clock.elapsed() > ASK_UNTIL:
            return None
        self.late_asked = True
        if not self.planned:
            self.plan_questions()
        used_before = len(self.ledger.answers)
        while self.question_plan and self.questions_left() > 0 and not self.finished and self.ask_next():
            pass
        fresh = self.ledger.answers[used_before:]
        if any(a.get("kind") == "answer" for a in fresh) or any("also said" in m for m in self.inbox):
            self.change_tick += 1
            return ("the shopper answered questions that were still open (see the status). Check the cart against every "
                    "requirement again; a new need may call for another listing")
        return None

    def try_order(self, reason: str) -> str | None:
        """Return None when the order went through, or why it should not be placed yet."""
        led = self.ledger
        if not self.roles.order:
            return "no order tool was found; call finish_decline if nothing can be bought"
        late = self.ask_before_finish()
        if late:
            return late
        if self.roles.cart_view and self.steps_left() > max(self.reserve(), 14) and self.gate_rejections < 4:
            self.call(self.roles.cart_view, {})
            if self.finished:
                return None
            self.flush_notes()
        if not led.cart and self.roles.cart_change:
            return "the cart is empty: add the chosen listing with the cart tool first"
        problems = self._cart_problems()
        if problems and self.gate_rejections < 4 and self.steps_left() > 2:
            self.gate_rejections += 1
            return "; ".join(problems)
        self._tidy_cart()
        if self.finished:
            return None
        if not led.cart:
            pick = self._best_known(loose=False)
            if not pick or not self._cart_add(pick):
                return "the cart is empty after taking out lines that cannot be bought: add a listing that fits"
        self._run_procedure("before_order")
        if self.finished:
            return None
        tool, args = self.roles.order
        obs, notes = self.call(tool, dict(args))
        if self.finished:
            return None
        self.flush_notes()
        # a refusal often means a cart line sold out: re-read every line with the tool that shows stock
        self.last_note_step = max(self.last_note_step, self.step)
        self.refusals += 1
        self.refresh_cart()
        gone = self._gone_lines()
        if gone and self.recover_sellout(gone) is None:
            return None
        gone = self._gone_lines()
        tail = f" These cart lines are sold out now: {gone}; take them out and choose other listings." if gone else ""
        return f"the store did not accept the order: {_compact(obs, 600)}.{tail}"

    def _gone_lines(self) -> list[str]:
        led = self.ledger
        gone = [lid for lid in led.cart if lid in led.unavailable
                or (lid in self.catalog.items and self.catalog.items[lid].stock is False)]
        for lid in gone:
            led.unavailable.add(lid)
        return gone

    def _swap_pick(self, out: str, same_shelf: bool = True) -> str | None:
        """The listing to put in place of a sold-out cart line: a confirmed fit, on the same shelf when the
        request has several items."""
        led = self.ledger
        base = self.catalog.items.get(out)
        fits = [i for i, it in self.catalog.items.items()
                if led.verdicts.get(i, {}).get("fit") == "yes" and not self.hard_fail(it) and i not in led.cart]
        if base and base.category:
            # the shopper's unstated needs held for the sold-out listing: stay on its shelf when anything there fits
            fits = [i for i in fits if self.catalog.items[i].category == base.category] or (
                [] if same_shelf or len(led.items) > 1 else fits)
        if not fits:
            return None
        if led.optimize == "cheapest":
            return min(fits, key=lambda i: self.catalog.items[i].price or 0)
        return max(fits, key=lambda i: (led.verdicts[i].get("strength", 1), self._likeness(base, self.catalog.items[i]),
                                        -(self.catalog.items[i].price or 0)))

    @staticmethod
    def _likeness(base: Listing | None, it: Listing) -> float:
        """How alike two listings look: a close title, the same shelf, a close price and shared attribute values."""
        if base is None:
            return 0.0
        score = SequenceMatcher(None, (base.title or "").lower(), (it.title or "").lower()).ratio()
        if base.category and it.category == base.category:
            score += 0.5
        if base.price and it.price and abs(it.price - base.price) <= 0.05 * base.price:
            score += 0.3
        shared = [k for k in base.traits if k in it.traits]
        if shared:
            score += sum(str(base.traits[k]).lower() == str(it.traits[k]).lower() for k in shared) / len(shared)
        return score

    def _twins(self, out: str, n: int) -> list[str]:
        """Unread listings most like the sold-out one: the store often carries the same product several times."""
        base = self.catalog.items.get(out)
        if base is None:
            return [it.id for it in self.candidates(n)]
        pool = [it for it in self.catalog.items.values()
                if not it.detailed and not self.hard_fail(it) and it.id not in self.ledger.cart
                and (not base.category or it.category == base.category)]
        pool.sort(key=lambda it: -(self._likeness(base, it) + 0.5 * self.screen(it)))
        return [it.id for it in pool[:n]]

    def recover_sellout(self, gone: list[str]) -> str | None:
        """After a checkout found cart lines sold out: swap each for the best other fit and check out again,
        in code, so the swap does not wait on model turns. None when the order went through."""
        self.rescuing = True
        try:
            for _ in range(2):
                if self.finished or not gone or self.steps_left() < 3 or not self.roles.order:
                    break
                out = gone[0]
                rest = [lid for lid in self.ledger.cart if lid not in gone]
                if rest and len(self.ledger.items) <= 1:
                    self._tidy_cart()
                    if self.finished or not self.ledger.cart:
                        return None if self.finished else "not recovered"
                    tool, args = self.roles.order
                    self.call(tool, dict(args))
                    if self.finished:
                        return None
                    self.flush_notes()
                    self.refresh_cart()
                    gone = self._gone_lines()
                    continue
                pick = self._swap_pick(out)
                if pick is None and self.steps_left() >= 5 and self.clock.elapsed() < FORCE_AT - 15:
                    look = self._twins(out, 2 if self.steps_left() >= 6 else 1)
                    self.open_listings(look)
                    self.judge([i for i in look if i in self.catalog.items and self.catalog.items[i].detailed])
                    pick = self._swap_pick(out)
                if pick is None:
                    pick = self._swap_pick(out, same_shelf=False)
                if pick is None:
                    break
                price = self.catalog.items[pick].price
                if self.ledger.optimize == "cheapest" and price and self.steps_left() >= 7 \
                        and self.clock.elapsed() < FORCE_AT - 20:
                    shelf = self.catalog.items[out].category if out in self.catalog.items else ""
                    cheaper = [it.id for it in self.candidates(6, below=price)
                               if self.screen(it) >= 1 and (not shelf or it.category == shelf)][:2]
                    if cheaper:
                        self.open_listings(cheaper)
                        self.judge([i for i in cheaper if self.catalog.items[i].detailed])
                        pick = self._swap_pick(out) or pick
                self.dialogue.append({"role": "agent", "sellout_swap": [out, pick]})
                if not self._swap(out, pick):
                    break
                tool, args = self.roles.order
                self.call(tool, dict(args))
                if self.finished:
                    return None
                self.flush_notes()
                self.last_note_step = max(self.last_note_step, self.step)
                self.refresh_cart()
                gone = self._gone_lines()
        finally:
            self.rescuing = False
        return "not recovered"

    def refresh_cart(self) -> None:
        """Right before checkout, and after any refusal: re-read each cart line with the tool that shows stock."""
        led = self.ledger
        tool = self.refresh_tool()
        if tool is None or not led.cart:
            return
        for lid in list(led.cart):
            item = self.catalog.items.get(lid)
            if self.finished or self.steps_left() <= 2:
                break
            if item and item.stock_step and item.stock_step >= self.last_note_step:
                continue
            self.call(tool.name, {tool.required[0]: lid})
        self.flush_notes()

    def _line_faults(self) -> list[str]:
        """With several requested items: each item needs its own cart line(s) in its own category and amount."""
        led = self.ledger
        if len(led.items) < 2 or not self.target_cats:
            return []
        shelves = list(self.target_cats.values())
        if any(a & b for i, a in enumerate(shelves) for b in shelves[i + 1:]):
            return []       # two items share a shelf: a line cannot be assigned to one of them from its category
        faults: list[str] = []
        claimed: set[str] = set()
        for idx, item in enumerate(led.items):
            cats = self.target_cats.get(idx)
            if not cats:
                continue
            lines = {lid: q for lid, q in led.cart.items()
                     if lid in self.catalog.items and _norm(self.catalog.items[lid].category) in cats}
            claimed |= set(lines)
            wanted = int(item.get("quantity") or 1)
            what = item.get("what") or f"item {idx + 1}"
            if not lines:
                faults.append(f"nothing in the cart is a {what} (category {sorted(cats)[0]}); add one")
            elif sum(lines.values()) != wanted:
                faults.append(f"the cart holds {sum(lines.values())} unit(s) of the {what} but the shopper wants {wanted}")
        stray = [lid for lid in led.cart if lid not in claimed and lid in self.catalog.items
                 and len(self.target_cats) >= len(led.items)]
        if stray:
            faults.append(f"cart line(s) {stray} belong to none of the requested items; remove them")
        return faults

    def _cart_problems(self) -> list[str]:
        led = self.ledger
        problems: list[str] = []
        self.refresh_cart()
        wanted = sum(int(it.get("quantity") or 1) for it in led.items)
        have = sum(led.cart.values())
        if len(led.items) > 1 and len(led.cart) < len(led.items):
            problems.append(f"the request has {len(led.items)} items but the cart has {len(led.cart)} line(s)")
        elif len(led.items) == 1 and have != wanted and led.cart:
            problems.append(f"the cart holds {have} unit(s) but the shopper wants {wanted}")
        total = 0.0
        for lid, qty in led.cart.items():
            item = self.catalog.items.get(lid)
            if not item:
                continue
            why = self.hard_fail(item)
            if why:
                problems.append(f"{lid}: {why}; remove it and pick another")
            total += (item.price or 0) * qty
        if led.budget is not None and not led.budget_each and total > led.budget + 1e-9:
            problems.append(f"cart total {total} is over the budget {led.budget}")
        problems.extend(self._line_faults())
        if problems:
            return problems
        # a cart line whose attribute list was never read has only been judged by its title: read it first
        unread = [lid for lid in led.cart if lid in self.catalog.items and not self.catalog.items[lid].detailed]
        if unread and self.steps_left() > self.reserve() + 1:
            self.open_listings(unread)
            self.flush_notes()
            if self.finished:
                return []
            self.sync_verdicts()
            for lid in unread:
                self.ledger.verdicts.pop(lid, None)
                self.judge_tried.discard((lid, self.ledger.key()))
        # one listing read tells little about which attribute carries a value, or what else the shelf holds
        read = [it for it in self.catalog.items.values()
                if it.detailed and not self.category_fail(it) and not self.hard_fail(it)]
        if len(read) < MIN_READ and not self.explored and self.steps_left() > self.reserve() + 8 \
                and self.clock.elapsed() < WRAP_AT - 60:
            self.explored = True
            more = [it.id for it in self.candidates(MIN_READ - len(read)) if self.screen(it) >= 0.4]
            if more:
                self.open_listings(more)
                self.flush_notes()
                if self.finished:
                    return []
                self.judge([i for i in more if self.catalog.items[i].detailed])
        verdicts = self.judge(list(led.cart), role="heavy")
        unchecked = [lid for lid in led.cart if lid not in verdicts]
        if unchecked:
            verdicts.update(self.judge(unchecked))
        for lid in led.cart:
            v = verdicts.get(lid)
            if v and v.get("fit") == "no":
                problems.append(f"{lid} does not fit: {v.get('why')}")
            elif v is None and self.unchecked_gate < 1 and self.steps_left() > self.reserve() + 2:
                self.unchecked_gate += 1
                problems.append(f"{lid} could not be checked against the requirements yet; compare its attributes with "
                                "every requirement in the status, then call finish_order again")
        weak = [lid for lid in led.cart if (led.verdicts.get(lid) or {}).get("strength") == 1.4]
        if weak and self.weak_gate < 2 and self.steps_left() > self.reserve() + 10 and self.candidates(1) \
                and led.optimize in ("none", "best", ""):
            self.weak_gate += 1
            return [f"{weak[0]} fits only through an attribute that is not named like the property (or a longer phrase). "
                    "A listing whose attribute named like the property states the value exactly fits more surely: "
                    "open a few more unread rows (UNREAD ROWS) and prefer such a listing; if none turns up, buy this one"]
        if problems or led.optimize not in ("cheapest", "least", "most", "best"):
            return problems
        if led.optimize == "cheapest" and len(led.cart) == 1:
            problems.extend(self.cheaper_check(next(iter(led.cart))))
        return problems

    # ---- the cheapest pick: look below its price

    @staticmethod
    def search_modes(t: Tool) -> tuple[str | None, str | None, str | None]:
        """(mode argument, keyword value, filter value) of a search tool that has a keyword and a filter mode."""
        kw_name = r"typ|text|keyword|query|search|word|phrase|free|term|string|match"
        fl_name = r"field|filter|facet|struct|narrow|attrib|refine|criteri|categor|browse|column"
        kw_near = r"word|keyword|text|phrase|quer|term|search by|free"
        fl_near = r"filter|facet|attribute|narrow|structur|categor|section|vendor|brand|price|criteri|field|department"
        for p, vals in t.enum_params().items():
            vals = [str(v) for v in vals]
            if len(vals) < 2:
                continue
            kw = re.search(r"(\w+) for (?:a )?(?:keyword|text|query|typed)", t.desc, re.I)
            fl = re.search(r"(\w+) for (?:a )?(?:filter|field|facet|structured)", t.desc, re.I)
            score: dict[str, tuple[int, int]] = {}
            for v in vals:
                k = 2 * bool(re.search(kw_name, v, re.I)) + 3 * bool(kw and kw.group(1) == v)
                f = 2 * bool(re.search(fl_name, v, re.I)) + 3 * bool(fl and fl.group(1) == v)
                for m in re.finditer(r"\b" + re.escape(v) + r"\b", t.desc):
                    window = t.desc[m.end(): m.end() + 45]
                    k += bool(re.search(kw_near, window, re.I))
                    f += bool(re.search(fl_near, window, re.I))
                score[v] = (k, f)
            best = None
            for a in vals:
                for b in vals:
                    if a == b:
                        continue
                    total = score[a][0] - score[a][1] + score[b][1] - score[b][0]
                    if score[a][0] > 0 and score[b][1] > 0 and (best is None or total > best[0]):
                        best = (total, a, b)
            if best:
                return p, best[1], best[2]
        return None, None, None

    def filter_search(self) -> tuple[Tool, str | None, str | None, str, str] | None:
        """A search tool that can list one category under a price ceiling."""
        for name in self.roles.search:
            t = self.roles.tools[name]
            mode_p, _kv, fv = self.search_modes(t)
            cat_p = next((p for p in t.string_params() if re.search(r"depart|categ|section|aisle|label|kind|shelf|group", p, re.I)), None)
            cap_p = next((p for p, v in t.props.items() if (v or {}).get("type") in ("number", "integer")
                          and re.search(r"cap|max|limit|under|ceil|spend|price|budget", p, re.I)), None)
            if cat_p and cap_p and (mode_p is None or fv is not None):
                return t, mode_p, fv, cat_p, cap_p
        return None

    @staticmethod
    def price_order(t: Tool) -> dict[str, Any]:
        """Arguments that sort a search's results from the lowest price up, when the tool can sort."""
        for p, vals in t.enum_params().items():
            if not re.search(r"sort|order|rank|arrange", p, re.I):
                continue
            for v in vals:
                s = str(v)
                if re.search(r"price.{0,6}(?:asc|low|up|inc)|(?:low|cheap)\w*.{0,6}(?:high|first|price)|^price$|cheapest", s, re.I):
                    return {p: v}
        return {}

    def keyword_search_capped(self) -> tuple[Tool, str | None, str | None, str, str] | None:
        """A search tool that takes words and a price ceiling, for stores whose filters lack a category."""
        for name in self.roles.search:
            t = self.roles.tools[name]
            mode_p, kw, _fv = self.search_modes(t)
            words_p = next((p for p in t.string_params() if re.search(r"text|query|keyword|term|search|word|phrase|^q$", p, re.I)), None)
            cap_p = next((p for p, v in t.props.items() if (v or {}).get("type") in ("number", "integer")
                          and re.search(r"cap|max|limit|under|ceil|spend|price|budget", p, re.I)), None)
            if words_p and cap_p and (mode_p is None or kw is not None):
                return t, mode_p, kw, words_p, cap_p
        return None

    def sweep_below(self, cap: float, label: str | None = None) -> int:
        """List the requested category under `cap` (its newest rows join the catalog); return how many rows were new."""
        found = self.filter_search()
        label = label or (self.target_labels[0] if self.target_labels else None)
        if not found or not label:
            return self.keyword_below(cap)
        t, mode_p, fv, cat_p, cap_p = found
        if (label, float(cap)) in self.swept or self.steps_left() <= self.reserve() + 2 or cap <= 0:
            return 0
        self.swept.add((label, float(cap)))
        args: dict[str, Any] = {cat_p: label, cap_p: int(cap) if (t.props[cap_p] or {}).get("type") == "integer" else cap}
        if mode_p:
            args[mode_p] = fv
        args.update(self.price_order(t))
        if t.check(args):
            return 0
        before = set(self.catalog.items)
        obs, _ = self.call(t.name, args)
        self.ledger.searches += 1
        self.sweeps += 1
        self._auto_page(t, args, obs, force=True)
        return len(set(self.catalog.items) - before)

    def keyword_below(self, cap: float) -> int:
        """The product noun searched under `cap`, lowest price first when the store sorts; return how many rows were new."""
        found = self.keyword_search_capped()
        what = str((self.ledger.items[0] if self.ledger.items else {}).get("what") or "").strip()
        if not found or not what or cap <= 0:
            return 0
        t, mode_p, kw, words_p, cap_p = found
        if (what, float(cap)) in self.swept or self.steps_left() <= self.reserve() + 2:
            return 0
        self.swept.add((what, float(cap)))
        limit = int((t.props.get(words_p) or {}).get("maxLength") or 80)
        args: dict[str, Any] = {words_p: what[:limit],
                                cap_p: int(cap) if (t.props[cap_p] or {}).get("type") == "integer" else cap}
        if mode_p:
            args[mode_p] = kw
        args.update(self.price_order(t))
        sig = json.dumps([t.name, args], sort_keys=True)
        if t.check(args) or sig in self.calls_seen:
            return 0
        self.calls_seen[sig] = self.change_tick
        before = set(self.catalog.items)
        obs, _ = self.call(t.name, args)
        self.ledger.searches += 1
        self.sweeps += 1
        self._auto_page(t, args, obs, force=True)
        return len(set(self.catalog.items) - before)

    def twin_search(self, pick: Listing) -> int:
        """Sellers list the same product several times at different prices: a keyword search on the pick's own title
        words finds those twins, which a category listing (a few rows in an arbitrary order) mostly hides."""
        if pick.id in self.twin_done or not pick.title:
            return 0
        self.twin_done.add(pick.id)
        for name in self.roles.search:
            t = self.roles.tools[name]
            mode_p, kw, _fv = self.search_modes(t)
            words_p = next((p for p in t.string_params() if re.search(r"text|query|keyword|term|search|word|phrase|^q$", p, re.I)), None)
            if not words_p or (mode_p and kw is None):
                continue
            limit = int((t.props.get(words_p) or {}).get("maxLength") or 80)
            tokens: list[str] = []
            for w in re.findall(r"[a-z0-9]+", pick.title.lower()):
                if len(w) >= 2 and w not in _STOPWORDS and w not in tokens:
                    tokens.append(w)
            before = set(self.catalog.items)
            for n in (4, 6):
                if self.steps_left() <= self.reserve() + 4 or self.clock.elapsed() > WRAP_AT - 30:
                    break
                query = " ".join(tokens[:n])[:limit]
                args: dict[str, Any] = {words_p: query}
                if mode_p:
                    args[mode_p] = kw
                if t.check(args):
                    break
                sig = json.dumps([name, args], sort_keys=True)
                if sig in self.calls_seen:
                    continue
                self.calls_seen[sig] = self.change_tick
                obs, _ = self.call(name, args)
                self.ledger.searches += 1
                obs = self._auto_page(t, args, obs, force=True)
                total = None
                if isinstance(obs, dict):
                    pages = obs.get("pages") or [obs]
                    first = pages[0] if isinstance(pages[0], dict) else {}
                    total = next((_num(v) for k, v in first.items() if re.search(r"total|count|match|hits", k, re.I)), None)
                if total is None or total <= 21 or len(tokens) <= n:
                    break
            return len(set(self.catalog.items) - before)
        return 0

    def cheaper_check(self, chosen: str) -> list[str]:
        """Before buying a cheapest pick: find listings below its price and read the likely ones."""
        pick = self.catalog.items.get(chosen)
        if not pick or pick.price is None:
            return []
        for _ in range(4):
            if self.finished:
                break
            self.sync_verdicts()
            rivals = [it for it in self.catalog.items.values()
                      if it.id != chosen and it.price is not None and it.price < pick.price
                      and not self.hard_fail(it) and (not pick.category or _norm(it.category) == _norm(pick.category))]
            unjudged = [r.id for r in rivals if r.detailed and r.id not in self.ledger.verdicts]
            if unjudged and self.clock.elapsed() < WRAP_AT:
                self.judge(unjudged[:10])
            fits = [r for r in rivals if self.ledger.verdicts.get(r.id, {}).get("fit") == "yes"]
            if not fits and (self.steps_left() <= self.reserve() + 3 or self.clock.elapsed() > WRAP_AT - 25):
                break
            if not fits and pick.id not in self.twin_done and self.twin_search(pick):
                continue
            if fits:
                best = min(fits, key=lambda r: r.price or 0)
                why = self.ledger.verdicts[best.id].get("why")
                return [f"{best.id} costs {best.price}, less than your pick {chosen} ({pick.price}), and fits ({why}); "
                        f"the request wants the cheapest fit, so take {chosen} out of the cart and buy {best.id} instead"]
            unread = [r for r in rivals if not r.detailed and r.id not in self.cheaper_checked]
            unread.sort(key=lambda r: (-(self.screen(r) >= 1), r.price or 0))
            if unread and self.screen(unread[0]) < 0.4:
                # nothing left mentions the required value: one blind look, then stop spending steps on rows that cannot match
                if self.blind_looks >= 1:
                    unread = []
                else:
                    self.blind_looks += 1
            if unread:
                batch = unread[:CHEAPER_CHECKS]
                self.cheaper_checked.update(r.id for r in batch)
                self.open_listings([r.id for r in batch])
                self.judge([r.id for r in batch if self.catalog.items[r.id].detailed])
                continue
            prices = sorted(r.price or 0 for r in rivals if r.price)
            cap = pick.price - 1
            if self.sweeps >= 1 and len(prices) >= 8:
                cap = min(cap, prices[len(prices) // 2])    # the first listing was a sample: look under its median
            if self.sweeps >= 3 or self.sweep_below(cap, pick.category or None) == 0:
                break
        return []

    def try_decline(self, reason: str) -> str | None:
        led = self.ledger
        if not self.roles.decline:
            return "no decline tool was found; keep searching or order the best fit"
        if self.sellout_save < 1 and self.last_note_step and self.step - self.last_note_step <= 3 and self.steps_left() > 1:
            self.sellout_save += 1
            nxt = [it.id for it in self.candidates(4)]
            if nxt:
                self.open_listings(nxt[:1])
                got = self.judge([i for i in nxt[:1] if i in self.catalog.items and self.catalog.items[i].detailed])
                yes = [i for i, v in got.items() if v.get("fit") == "yes" and not self.hard_fail(self.catalog.items[i])]
                if yes:
                    return f"the cart listing sold out, but {yes[0]} fits and is in stock. Put it in the cart and call finish_order."
                return "the listing that sold out is not the only result. Open more unread rows before declining."
        if self.decline_rejections < 4 and self.steps_left() > self.reserve() + 3 and self.clock.elapsed() < WRAP_AT:
            seen = [it.id for it in self.catalog.items.values() if not self.hard_fail(it)]
            if led.searches < 3:
                self.decline_rejections += 1
                return ("only a few searches were made; most requests can be met. Try other wordings of the key value "
                        "(a synonym, a plural, the product noun alone) and read the further result pages first")
            verdicts = self.judge([i for i in seen if self.catalog.items[i].detailed])
            fits = [i for i, v in verdicts.items() if v.get("fit") == "yes"]
            if fits:
                self.decline_rejections += 1
                return f"these opened listings look like fits: {fits[:4]}; check them before declining"
            read = [i for i in seen if self.catalog.items[i].detailed]
            unread = [i for i in seen if not self.catalog.items[i].detailed][:6]
            unsure = [i for i, v in verdicts.items() if v.get("fit") == "unknown"][:4]
            if not unread and not unsure and len(read) < 10 and self.steps_left() > self.reserve() + 6:
                self.decline_rejections += 1
                return (f"only {len(read)} in-stock listings within budget were read. A listing that looks different may still "
                        f"state the value: search again with other wordings and open more listings before declining")
            if unread or unsure:
                self.decline_rejections += 1
                self.open_listings(unread[:3])
                how = ", ".join(sorted(self.attr_tools)) or "each single-listing tool"
                return (f"in-stock listings within budget have unread attributes {unread} or unclear fit {unsure}; "
                        f"read them with {how} before declining")
        converted = self._decline_to_order()
        if converted != "skip":
            return converted
        self._run_procedure("before_decline")
        if self.finished:
            return None
        tool, args, note = self.roles.decline
        call_args = dict(args)
        if note:
            t = self.roles.tools[tool]
            limit = int((t.props.get(note) or {}).get("maxLength") or 300)
            call_args[note] = (f"Sorry, nothing in stock meets all of your requirements"
                               f"{' within ' + str(int(led.budget)) if led.budget else ''}, so I am not buying a substitute.")[:limit]
        obs, notes = self.call(tool, call_args)
        if self.finished:
            return None
        self.flush_notes()
        return f"the store did not accept the decline: {_compact(obs, 600)}"

    def _run_procedure(self, when: str) -> None:
        for i, p in enumerate(self.ledger.procedure):
            if i in self.ledger.done_procedure or p.get("when") != when or self.finished:
                continue
            if not self.roles.message or not re.search(r"message|tell|inform|confirm|let|ask|send|note|share|text",
                                                       p["action"], re.I):
                continue
            if self.steps_left() <= 1:
                return
            self.ledger.done_procedure.add(i)
            cart = [self.catalog.items[l].card(traits=False) for l in self.ledger.cart if l in self.catalog.items]
            msg = self.brain.chat("light", [
                {"role": "system", "content": "Write the one short, plain message to the shopper that this required step asks for. Output only the message text."},
                {"role": "user", "content": json.dumps({"step": p["action"], "cart": cart, "budget": self.ledger.budget})}],
                max_tokens=800, critical=True, cap=20)
            text = ((msg or {}).get("content") or "").strip() or f"Before I finish: {p['action']}"
            tool, arg = self.roles.message
            limit = int((self.roles.tools[tool].props.get(arg) or {}).get("maxLength") or 600)
            self.call(tool, {arg: text[:limit]})
            self.ledger.questions_used += 1

    def force_finish(self) -> None:
        """Out of time or steps: order a cart that passes the hard checks, else decline."""
        if self.finished:
            return
        led = self.ledger
        if led.cart and self.roles.cart_change and self.steps_left() >= 3:
            self._tidy_cart()
            if self.finished:
                return
        ok_cart = led.cart and all(
            lid in self.catalog.items and not self.hard_fail(self.catalog.items[lid]) for lid in led.cart)
        if not led.cart and self.roles.cart_change and self.steps_left() >= 2:
            pick = self._best_known(loose=True)
            if pick:
                ok_cart = self._cart_add(pick)
        if ok_cart and self.roles.order and not self.finished:
            tool, args = self.roles.order
            self.call(tool, dict(args))
        for _ in range(2):
            if self.finished or not ok_cart or self.steps_left() < 4 or not self.roles.cart_change:
                break
            self.flush_notes()
            gone = [lid for lid in led.cart if lid in self.catalog.items and self.hard_fail(self.catalog.items[lid])]
            pick = self._best_known(loose=True)
            if not gone or not pick:
                break
            if not self._swap(gone[0], pick):
                break
            tool, args = self.roles.order
            self.call(tool, dict(args))
        if not self.finished and self.roles.decline:
            tool, args, note = self.roles.decline
            call_args = dict(args)
            if note:
                call_args[note] = "Sorry, I could not find an in-stock option that meets everything you asked for."
            self.call(tool, call_args)

    def _cart_add(self, pick: str) -> bool:
        """Put `pick` in the cart with the store's add call; True when it shows in the cart afterwards."""
        mode = next((m for m in self.roles.cart_change.values() if m["add"] is not None), None)
        name = next((n for n, m in self.roles.cart_change.items() if m is mode), None)
        if mode is None or name is None or self.steps_left() < 2:
            return False
        args = {mode["id"]: pick}
        if mode["op"]:
            args[mode["op"]] = mode["add"]
        if mode.get("qty"):
            args[mode["qty"]] = int(self.ledger.items[0].get("quantity") or 1) if self.ledger.items else 1
        self.call(name, args)
        return pick in self.ledger.cart

    def _decline_to_order(self) -> str | None:
        """A decline that is about to go through while a listing fits: buy that listing instead, since a decline
        scores only when nothing at all fits. Returns None when the episode ended, else a note for the model."""
        if self.converted >= 2 or not self.roles.cart_change or not self.roles.order \
                or self.steps_left() < 5 or self.clock.elapsed() > WRAP_AT:
            return "skip"
        held = [lid for lid in self.ledger.cart if lid in self.catalog.items
                and not self.hard_fail(self.catalog.items[lid])
                and self.ledger.verdicts.get(lid, {}).get("fit") != "no"]
        if held:
            # the cart already holds a listing that may fit: buy it rather than decline
            self.converted += 1
            self.dialogue.append({"role": "agent", "decline_to_order": held})
            return self.try_order("the cart holds a fitting listing, so the code bought it instead of declining") or None
        if self.ledger.cart:
            self._tidy_cart()
            if self.finished:
                return None
        pick = self._best_known(loose=False)
        if not pick:
            # no listing is confirmed, but one that states every wanted word and has no failing attribute is a fair try
            near = [i for i, it in self.catalog.items.items()
                    if it.detailed and not self.hard_fail(it) and self.screen(it) >= 1
                    and self.ledger.verdicts.get(i, {}).get("fit") != "no"]
            if near:
                pick = min(near, key=lambda i: self.catalog.items[i].price or 0)
        if not pick:
            return "skip"
        self.converted += 1
        self.dialogue.append({"role": "agent", "decline_to_order": pick})
        if not self._cart_add(pick):
            return "skip"
        return self.try_order("a listing fits, so the code bought it instead of declining") or None

    def _cart_remove(self, lid: str) -> bool:
        remover = next(((n, m) for n, m in self.roles.cart_change.items() if m["remove"] is not None), None)
        if not remover or self.steps_left() < 2:
            return False
        name, mode = remover
        args = {mode["id"]: lid}
        if mode["op"]:
            args[mode["op"]] = mode["remove"]
        if mode.get("qty"):
            args[mode["qty"]] = self.ledger.cart.get(lid, 1)
        self.call(name, args)
        return lid not in self.ledger.cart

    def _tidy_cart(self) -> None:
        """Before a checkout: take out lines that can no longer be bought and, when one item is wanted, every
        line but the best, since a cart with a stray line never scores."""
        led = self.ledger
        bad = [lid for lid in led.cart if lid in self.catalog.items and self.hard_fail(self.catalog.items[lid])]
        if len(led.items) <= 1:
            good = [lid for lid in led.cart if lid not in bad]
            if len(good) > 1:
                def rank(lid: str) -> tuple:
                    v = led.verdicts.get(lid, {})
                    price = (self.catalog.items[lid].price or 0) if lid in self.catalog.items else 0
                    return (v.get("fit") == "yes", -price if led.optimize == "cheapest" else v.get("strength", 0),
                            self.cart_step.get(lid, 0))
                keep = max(good, key=rank)
                bad += [g for g in good if g != keep]
        for lid in bad:
            if self.finished or not self._cart_remove(lid):
                break

    def _swap(self, out: str, pick: str) -> bool:
        """Replace a cart line that can no longer be bought with the next best fit."""
        led = self.ledger
        qty = led.cart.get(out, 1)
        remover = next(((n, m) for n, m in self.roles.cart_change.items() if m["remove"] is not None), None)
        adder = next(((n, m) for n, m in self.roles.cart_change.items() if m["add"] is not None), None)
        if not remover or not adder:
            return False
        for (name, mode), lid, op in ((remover, out, "remove"), (adder, pick, "add")):
            args = {mode["id"]: lid}
            if mode["op"]:
                args[mode["op"]] = mode[op]
            if mode.get("qty"):
                args[mode["qty"]] = qty
            self.call(name, args)
            if self.finished:
                return False
        return pick in led.cart and out not in led.cart

    def _best_known(self, loose: bool = False) -> str | None:
        """The listing to buy when time or steps run out. `loose` also accepts unclear fits and unread rows,
        because most requests can be met and a guess beats ending without a purchase."""
        verdicts = self.ledger.verdicts
        usable = [i for i, it in self.catalog.items.items() if not self.hard_fail(it) and i not in self.ledger.cart]
        fits = [i for i in usable if verdicts.get(i, {}).get("fit") == "yes"]
        if fits:
            if self.ledger.optimize == "cheapest":
                return min(fits, key=lambda i: self.catalog.items[i].price or 0)
            return max(fits, key=lambda i: (verdicts[i].get("strength", 1), -(self.catalog.items[i].price or 0)))
        if not loose:
            return None
        unknown = [i for i in usable if verdicts.get(i, {}).get("fit") == "unknown"]
        if unknown:
            return min(unknown, key=lambda i: self.catalog.items[i].price or 0)
        rows = [it for it in self.candidates(1) if self.screen(it) >= 0.4]
        return rows[0].id if rows else None

    # ---- loop

    def loop_tools(self) -> list[dict[str, Any]]:
        hidden = self.roles.finish_tools()
        tools = [t.spec for t in self.tools if t.name not in hidden]
        tools.append({"type": "function", "function": {
            "name": "finish_order", "description": "Buy the cart now. The code checks it first.",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": []}}})
        tools.append({"type": "function", "function": {
            "name": "finish_decline", "description": "End with no purchase because nothing in the store meets every requirement.",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": []}}})
        return tools

    def status(self) -> str:
        led = self.ledger
        table = []
        self.sync_verdicts()
        for it in sorted(self.catalog.items.values(), key=lambda x: (x.price is None, x.price or 0))[:18]:
            v = led.verdicts.get(it.id)
            table.append({"id": it.id, "title": it.title[:70], "price": it.price, "in_stock": it.stock,
                          "category": it.category, "attributes_read": it.detailed,
                          "fit": (v or {}).get("fit", "?"), "strength": (v or {}).get("strength"), "why": (v or {}).get("why", "")[:80],
                          "blocked": self.hard_fail(it)})
        cart = {lid: {"qty": q, "price": getattr(self.catalog.items.get(lid), "price", None)} for lid, q in led.cart.items()}
        lines = [
            "STATUS (from the code; trust it over your memory)",
            f"steps used {self.step} of {self.max_steps}; {self.steps_left()} left; time left about {int(self.clock.left())} s",
            ("attribute tool (shows a listing's attribute list): " + ", ".join(sorted(self.attr_tools)))
            if self.attr_tools else "attribute tool: not known yet; try each single-listing tool once",
            "task: " + json.dumps(led.summary(), ensure_ascii=False, default=str),
            "cart: " + json.dumps(cart),
            "listings seen (cheapest first): " + json.dumps(table, ensure_ascii=False, default=str),
        ]
        if any((v or {}).get("fit") == "yes" for v in led.verdicts.values()) and led.optimize in ("none", "best", ""):
            lines.append("When several listings fit, buy the one with the highest strength (2 = the attribute states the value exactly).")
        nxt = self.candidates(6)
        simple, _ = self.split_reqs()
        unseen = [r["value"] for r in simple if r.get("kind") != "exclude" and r.get("value")
                  and not any(r["words"] <= (_words(it.title) | _words(it.desc)) for it in self.catalog.items.values())]
        if unseen and self.catalog.items and led.searches < 6:
            # rows read in price order rarely carry a value no visible title or description mentions
            lines.append(f"No listing seen so far mentions {', '.join(map(str, unseen[:2]))}: run a keyword search for it "
                         "(with the product noun) before opening more rows.")
        if nxt:
            lines.append("UNREAD ROWS TO OPEN NEXT (best match of the request's words first; cheapest first when the request wants the cheapest): "
                         + json.dumps([{"id": it.id, "title": it.title[:60], "price": it.price} for it in nxt], ensure_ascii=False))
        if self.inbox:
            lines.append("NEW FROM THE SHOPPER OR THE CODE:\n- " + "\n- ".join(self.inbox))
            self.inbox = []
        if self.steps_left() <= self.reserve() + 2 or self.clock.elapsed() > WRAP_AT:
            lines.append("FINISH NOW: put the best fitting listing in the cart and call finish_order, or call finish_decline.")
        return "\n".join(lines)

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.dialogue
        self.brief()
        tools = self.loop_tools()
        self.history = [{"role": "system", "content": LOOP_SYSTEM},
                        {"role": "user", "content": "SHOPPER REQUEST:\n" + self.ledger.request}]
        idle = dead = 0
        for _turn in range(LOOP_TURNS):
            if self.finished:
                break
            if self.steps_left() <= 1 or self.clock.elapsed() > FORCE_AT:
                break
            self.flush_notes()
            if self.clock.elapsed() < WRAP_AT:
                self.lock_categories()
            opened = sum(1 for it in self.catalog.items.values() if it.traits and not self.category_fail(it))
            if self.questions_left() > 0 and (
                    (not self.planned and (opened >= 3 or self.step >= 8)) or (self.replan and opened >= 3)
                    or (self.planned and self.plans < 3 and opened >= self.plan_opened + 2
                        and self.clock.elapsed() < ASK_UNTIL - 20)):
                self.replan = False
                self.plan_opened = opened
                self.plans += 1
                self.plan_questions()
            if self.planned and self.ask_next():
                continue
            key = self.sync_verdicts()
            fresh = [it.id for it in self.catalog.items.values()
                     if it.detailed and it.id not in self.ledger.verdicts and (it.id, key) not in self.judge_tried]
            if fresh and self.clock.elapsed() < WRAP_AT:
                self.judge_tried.update((i, key) for i in fresh[:10])
                self.judge(fresh[:10])
            self.history.append({"role": "user", "content": self.status()})
            # leave the closing calls (cart, order, a swap after a sell-out) about half a minute
            msg = self.brain.chat("light", self.history, tools=tools, max_tokens=1600, critical=True,
                                  cap=max(8, min(40, self.clock.left() - 30)))
            self.history.pop()
            if msg is None:
                # every model failed this turn; the providers often recover, so only give up when time is short
                dead += 1
                if dead >= 3 or self.clock.elapsed() > WRAP_AT:
                    break
                time.sleep(2)
                continue
            dead = 0
            calls = msg.get("tool_calls") or []
            content = msg.get("content") or ""
            if not calls:
                idle += 1
                self.history.append({"role": "assistant", "content": content or "(no call)"})
                if idle >= 2 and self.ledger.cart:
                    why = self.try_order("the model stopped calling tools")
                    if why is None:
                        break
                    self.history.append({"role": "user", "content": f"The code tried to check out: {why}. Reply with exactly one tool call."})
                else:
                    self.history.append({"role": "user", "content": "Reply with exactly one tool call."})
                if idle > 4:
                    break
                continue
            idle = 0
            batch = self._batch(calls)
            self.history.append({"role": "assistant", "content": content, "tool_calls": batch})
            for call in batch:
                result = self.act(call)
                self.history.append({"role": "tool", "tool_call_id": call.get("id", "call"), "content": result})
                if self.finished:
                    break
            self._trim_history()
        self.force_finish()
        return self.dialogue

    def _read_only(self, call: dict[str, Any]) -> bool:
        name = (call.get("function") or {}).get("name", "")
        return name in self.roles.search or name in self.roles.details or name == self.roles.cart_view

    def _batch(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The calls to run this turn: the first alone, unless it and the next ones only read the store (a search or a
        listing view), which cost no more steps run together than one by one but save a model turn each."""
        first = calls[0]
        if not self._read_only(first) or self.steps_left() <= self.reserve() + 6:
            return [first]
        room = min(MULTI_CALLS, self.steps_left() - self.reserve() - 4)
        batch = [first]
        for c in calls[1:]:
            if len(batch) >= room:
                break
            if self._read_only(c):
                batch.append(c)
        return batch

    def act(self, call: dict[str, Any]) -> str:
        fn = call.get("function") or {}
        name = fn.get("name", "")
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            return "Not run: arguments were not valid JSON."
        if name == "finish_order":
            why = self.try_order(str(args.get("reason", "")))
            return "Order placed." if why is None else f"Not finished: {why}"
        if name == "finish_decline":
            why = self.try_decline(str(args.get("reason", "")))
            return "Ended without purchase." if why is None else f"Not finished: {why}"
        tool = self.roles.tools.get(name)
        if tool is None or name in self.roles.finish_tools():
            return f"Not run: '{name}' is not available to you."
        problem = tool.check(args)
        if problem:
            return f"Not run (no step used): {problem}."
        mode_p, _kv, fv = self.search_modes(tool) if name in self.roles.search else (None, None, None)
        if mode_p and fv is not None and args.get(mode_p) == fv:
            words_p = [p for p in tool.string_params() if re.search(r"text|query|keyword|term|search|word|phrase|^q$", p, re.I)]
            if any(args.get(p) for p in words_p):
                return (f"Not run (no step used): with {mode_p}={fv} the store ignores words and reads only its filter "
                        f"arguments. Use the keyword mode for words, or drop {words_p} and filter by category, brand or price.")
        if name in self.roles.cart_change:
            lid = str(args.get(self.roles.cart_change[name]["id"]) or "")
            if self.cart_touches.get(lid, 0) >= CART_TOUCHES and lid not in self.ledger.unavailable:
                return (f"Not run (no step used): {lid} was already put in and taken out of the cart several times. "
                        "Keep the cart as it is and call finish_order, or add a different listing.")
            self.cart_touches[lid] = self.cart_touches.get(lid, 0) + 1
        sig = json.dumps([name, args], sort_keys=True)
        if self.calls_seen.get(sig) == self.change_tick and name not in self.roles.cart_change:
            return "Not run (no step used): you already made this exact call and nothing has changed since."
        self.calls_seen[sig] = self.change_tick
        obs, notes = self.call(name, args)
        if name in self.roles.search:
            self.ledger.searches += 1
            obs = self._auto_page(tool, args, obs)
            self.auto_open()
        text = _compact(obs)
        if name in self.roles.details and tool.required:
            item = self.catalog.items.get(str(args.get(tool.required[0])))
            if item and not item.detailed:
                others = sorted(self.attr_tools) or [d for d in self.roles.details if d not in self.detail_tried]
                if others:
                    text += (f"\nNOTE: this view shows no attribute list for {item.id}. "
                             f"Open it with {others[0]} to read its attributes before judging it.")
        if notes:
            text += "\nSHOPPER UPDATE: " + " | ".join(notes)
        return text

    def _auto_page(self, tool: Tool, args: dict[str, Any], obs: Any, force: bool = False) -> Any:
        """Read the further result pages a search reports: always for the first searches (a page holds few rows)
        and for cheapest-style tasks."""
        wanted = force or self.ledger.optimize in ("cheapest", "least", "most") or (
            self.ledger.searches <= 2 and self.steps_left() > 20)
        if not wanted or not isinstance(obs, dict):
            return obs
        page_arg = next((p for p in tool.props if re.search(r"page|offset|cursor|token|next|continue|start", p, re.I)), None)
        if not page_arg:
            return obs
        pages = [obs]
        for _ in range(AUTO_PAGES):
            nxt_key = next((k for k in obs if re.search(r"next|continue|cursor|token|more|after", k, re.I)), None)
            nxt = obs.get(nxt_key) if nxt_key else None
            if nxt in (None, "", False) or self.steps_left() <= self.reserve() + 4 or self.clock.elapsed() > WRAP_AT - 30:
                break
            schema = tool.props.get(page_arg) or {}
            value: Any = nxt
            if schema.get("type") == "integer":
                value = int(_num(nxt) or 0)
                if not value:
                    break
            nargs = {**args, page_arg: value}
            sig = json.dumps([tool.name, nargs], sort_keys=True)
            if tool.check(nargs) or sig in self.calls_seen:
                break
            self.calls_seen[sig] = self.change_tick
            obs, _ = self.call(tool.name, nargs)
            if not isinstance(obs, dict) or _is_error(obs):
                break
            pages.append(obs)
        return pages[0] if len(pages) == 1 else {"pages_read": len(pages), "pages": pages}

    def _trim_history(self) -> None:
        size = sum(len(json.dumps(m, default=str)) for m in self.history)
        i = 2
        while size > 90000 and i < len(self.history) - 4:
            m = self.history[i]
            if m.get("role") == "tool" and len(m.get("content", "")) > 400:
                size -= len(m["content"]) - 120
                m["content"] = m["content"][:120] + " … (older result shortened; see STATUS)"
            i += 1


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception as exc:  # an unfinished episode scores 0, so always try to finish
        episode.dialogue.append({"role": "agent", "error": f"{type(exc).__name__}: {exc}"})
        try:
            episode.force_finish()
        except Exception as inner:
            episode.dialogue.append({"role": "agent", "error": f"finish failed: {inner}"})
        return episode.dialogue


__all__ = ["agent_main"]
