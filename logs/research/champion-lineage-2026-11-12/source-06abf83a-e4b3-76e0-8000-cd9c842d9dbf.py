from __future__ import annotations

from os import getenv
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic
import json
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from src.agent.proxy_client import ProxyClient

SEARCH, FILTER, VIEW, COMPARE = "search", "filter", "view", "compare"
INSPECT_STOCK, INSPECT_CART = "inspect_stock", "inspect_cart"
ADD, MESSAGE, ORDER = "add_to_cart", "message", "place_test_order"

GLM = ("z-ai/glm-5.2", "zai-org/GLM-5.2-TEE")
KIMI = ("moonshotai/kimi-k2.6", "moonshotai/Kimi-K2.6-TEE")
GPT_LUNA = ("openai/gpt-6-luna", "zai-org/GLM-5.2-TEE", {"reasoning": {"enabled": False}})
MODELS = (GLM, GLM, KIMI)
RECORD_MODELS = SHORTLIST_MODELS = CLAIMS_MODELS = MODELS
EXTRACT_MODELS = (GPT_LUNA, GLM, KIMI)
JUDGE_MODELS = MODELS

TIME_LIMIT, STORE_LIMIT, WRAP_UP_AT, HURRY_AT, CLIENT_LIMIT = 270, 290, 235, 180, 60
REQUESTS_AT_ONCE, EXTRACTS_AT_ONCE, HEDGE_AT, RETRIES = 3, 3, 0.75, 3
OPEN_ROWS, EXTRACT_PAGES, REPLACEMENT_OPENS, TABLE_ROWS, PAGE_ROWS, SISTER_PRICE_ROOM, SHELF_ROWS = 10, 6, 12, 24, 6, 1.1, 4


_proxy = ProxyClient(timeout=90, max_retries=1)


class Tool:
    def __init__(self, spec: Dict[str, Any], glm_names: Optional[Dict[str, str]] = None) -> None:
        function = spec["function"]
        self.name, self.description, self.parameters = function["name"], function["description"], function["parameters"]
        self.glm_names = dict(glm_names or {})
        self.code_names = {seen: own for own, seen in self.glm_names.items()}

    def spec(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": Tool.renamed_schema(self.parameters, self.glm_names)}}

    def read(self, args: Any) -> Any:
        return Tool.renamed(args, self.code_names)

    def show(self, args: Any) -> Any:
        return Tool.renamed(args, self.glm_names)

    def variant(self, parameters: Dict[str, Any]) -> "Tool":
        return Tool({"function": {"name": self.name, "description": self.description, "parameters": parameters}}, self.glm_names)

    @staticmethod
    def renamed(value: Any, names: Dict[str, str]) -> Any:
        if isinstance(value, dict):
            return {names.get(key, key): Tool.renamed(item, names) for key, item in value.items()}
        if isinstance(value, list):
            return [Tool.renamed(item, names) for item in value]
        return value

    @staticmethod
    def renamed_schema(schema: Any, names: Dict[str, str]) -> Any:
        if isinstance(schema, list):
            return [Tool.renamed_schema(item, names) for item in schema]
        if not isinstance(schema, dict):
            return schema
        return {key: ({names.get(name, name): Tool.renamed_schema(shape, names) for name, shape in sorted(item.items(), key=lambda pair: names.get(pair[0], pair[0]))}
                      if key == "properties" and isinstance(item, dict) else
                      [names.get(name, name) for name in item] if key == "required" and isinstance(item, list) else Tool.renamed_schema(item, names))
                for key, item in schema.items()}

class Seat:
    def __init__(self, guide: str, tool: Tool, tokens: int, wait: float, models: Tuple[Tuple[Any, ...], ...] = MODELS) -> None:
        self.guide, self.tool, self.tokens, self.wait = guide, tool, tokens, wait
        self.name = tool.name
        self.models = models

    def messages(self, user: str) -> List[Dict[str, Any]]:
        return [{"role": "system", "content": self.guide}, {"role": "user", "content": user}]

class Text:
    _TYPOGRAPHY = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2033": '"', "\u2018": "'", "\u2019": "'", "\u2032": "'", "\u2013": "-", "\u2014": "-", "\\": None, "\u200b": None, "\u200c": None, "\u200d": None, "\ufeff": None})

    _ESCAPED = re.compile(r"\\u([0-9a-f]{4})")

    @staticmethod
    def squash(text: Any) -> str:
        return " ".join(str(text).casefold().split())

    @staticmethod
    def plain(text: Any) -> str:
        return Text._ESCAPED.sub(lambda match: chr(int(match.group(1), 16)), Text.squash(text)).translate(Text._TYPOGRAPHY)

    @staticmethod
    def unlabelled(quote: Any) -> str:
        return re.sub(r"^\s*(?:title|options?|facts?)\s*:\s*", "", str(quote or ""), flags=re.I)

    @staticmethod
    def tag(text: Any) -> str:
        return "".join(re.findall(r"[^\W_]+", Text.squash(text)))

    @staticmethod
    def words(text: Any) -> set:
        text = re.sub(r"[()\[\]{}\"/\u3010\u3011\u300c\u300d\u300e\u300f\uff08\uff09\u3008\u3009\u300a\u300b\u3014\u3015]", " ", str(text))
        return ({word.strip("'.,;:!?") for word in Text.squash(text).split()} | set(re.findall(r"[^\W_]+", Text.squash(text)))) - {""}

    @staticmethod
    def tokens(text: Any) -> set:
        return set(re.findall(r"[^\W_]+", Text.squash(text)))

    @staticmethod
    def bag(text: Any) -> frozenset:
        return frozenset(word[:-1] if word.endswith("s") and len(word) > 3 else word for word in Text.tokens(text))

    @staticmethod
    def within(needle: Any, haystack: Any) -> bool:
        one, other = Text.plain(needle), Text.plain(haystack)
        return bool(one) and (one in other or one.replace(" ", "") in other.replace(" ", ""))

    @staticmethod
    def label_words(text: Any) -> str:
        return " ".join(re.findall(r"[^\W_]+", Text.squash(text)))

    @staticmethod
    def same_label(named: Any, label: Any) -> bool:
        marks = lambda text: re.findall(r"[()\[\]]", str(text))
        return Text.label_words(named) == Text.label_words(label) and (not marks(named) or marks(named) == marks(label))

    @staticmethod
    def dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    @staticmethod
    def stock_word(in_stock: Any) -> str:
        return "in stock" if in_stock else "SOLD OUT"

class Figure:
    @staticmethod
    def number(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).replace(",", "").strip()
        try:
            return float(text)
        except ValueError:
            match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*[^\d]*", text)
            return float(match.group(1)) if match else None

    @staticmethod
    def figures(text: Any) -> List[Tuple[float, str]]:
        found = []
        for match in re.finditer(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*-?\s*([\u00b0]?[^\W\d_]*[\"']?)", Text.plain(text)):
            value = Figure.number(match.group(1))
            if value is not None:
                found.append((value, match.group(2).lower()))
        return [(value, unit or next((later for figure, later in found[index + 1:] if figure == value and later), "")) for index, (value, unit) in enumerate(found)]

    @staticmethod
    def unit_like(one: str, other: str) -> bool:
        grab = lambda unit: "".join(char for char in str(unit).lower() if char.isalpha() or char == "\u00b0")
        a, b = grab(one), grab(other)
        if not a or not b:
            return True
        a, b = (a[:-1] if len(a) > 2 and a.endswith("s") else a), (b[:-1] if len(b) > 2 and b.endswith("s") else b)
        if a[0] != b[0]:
            return False
        short, rest = (a, iter(b)) if len(a) <= len(b) else (b, iter(a))
        return all(char in rest for char in short)

    @staticmethod
    def unit_stated(written: str, wanted: str) -> bool:
        return bool(written) or not wanted

    @staticmethod
    def states(text: Any, value: float, unit: str, relation: str) -> bool:
        for found, written in Figure.figures(text):
            if not Figure.unit_like(written, unit) or not Figure.unit_stated(written, unit):
                continue
            if (relation == "at least" and found >= value) or (relation == "at most" and found <= value) or found == value:
                return True
        return False

    @staticmethod
    def ranges(text: Any) -> List[Tuple[float, float, str]]:
        found = []
        for low, high, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(?:-|~|\u2013|to)\s*(\d+(?:\.\d+)?)\s*([^\W\d_]+)", Text.plain(text)):
            if float(low) < float(high):
                found.append((float(low), float(high), unit.lower()))
        return found

    @staticmethod
    def another_figure(text: Any, value: float, unit: str) -> Optional[float]:
        for found, written in Figure.figures(text):
            if unit and written and Figure.unit_like(written, unit) and found != value:
                return found
        return None

    @staticmethod
    def money(text: str) -> List[float]:
        return [value for value in (Figure.number(figure.replace(",", "")) for figure in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*PHP\b", str(text))) if value is not None]

    @staticmethod
    def over(price: Any, budget: Optional[float]) -> bool:
        return budget is not None and Figure.number(price) is not None and Figure.number(price) > budget

class Naming:
    @staticmethod
    def stem(sku: Any) -> str:
        return str(sku).rsplit("-", 1)[0]

    @staticmethod
    def is_key(item: Any) -> bool:
        return "::" in str(item) and str(item).split("::")[0].strip().isdigit()

    @staticmethod
    def find_key(text: Any, keys: List["Key"]) -> Optional["Key"]:
        text = Text.squash(text).strip(" .")
        if "::" in text:
            product_id, _, sku = text.partition("::")
            key = (product_id.strip(), sku.strip())
            exact = next((candidate for candidate in keys if Text.squash(candidate[0]) == key[0] and Text.squash(candidate[1]) == key[1]), None)
            same = [candidate for candidate in keys if Text.squash(candidate[0]) == key[0]]
            by_sku = [candidate for candidate in keys if Text.squash(candidate[1]) == key[1]]
            return exact or (same[0] if len(same) == 1 else None) or (by_sku[0] if len(by_sku) == 1 else None)
        matches = [candidate for candidate in keys if Text.squash(candidate[0]) == text]
        return matches[0] if len(matches) == 1 else None

class Wire:
    @staticmethod
    def openrouter() -> bool:
        return getenv("INFERENCE_PROVIDER", "openrouter") != "chutes"

    @staticmethod
    def native(model: Tuple[Any, ...]) -> str:
        return model[0] if Wire.openrouter() else model[1]

    @staticmethod
    def fields(model: Tuple[Any, ...]) -> Dict[str, Any]:
        if not Wire.openrouter():
            return {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}, "temperature": 0}
        return dict(model[2]) if len(model) > 2 else {"reasoning": {"enabled": False}, "temperature": 0}

    @staticmethod
    def post(path: str, body: Dict[str, Any], client: Any = _proxy) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        for attempt in range(RETRIES + 1):
            result = client.post_verbose(path, json_data=body)
            error = result.error or {}
            if error.get("status") == 429 and "capacity" in str(error.get("body") or "").lower():
                return None, dict(error, kind="capacity")
            if result.data is not None or error.get("status") != 429 or attempt == RETRIES:
                return result.data, error
            Wire.pause(2 ** attempt * (1 + sum(map(ord, Text.dumps(body.get("messages") or body.get("calls"))[-512:])) % 100 / 100))
        return None, {}

    @staticmethod
    def pause(seconds: float) -> None:
        box: Queue = Queue()
        try:
            box.get(timeout=max(0.0, seconds))
        except Empty:
            pass

    @staticmethod
    def fit(value: Any, schema: Dict[str, Any]) -> Tuple[bool, Any]:
        kind = schema.get("type")
        if isinstance(kind, list):
            kind = next((option for option in kind if option != "null"), None)
        if kind == "object":
            if not isinstance(value, dict):
                return False, value
            declared = schema.get("properties") or {}
            fitted = {}
            for name, given in value.items():
                if name not in declared or given is None and "null" not in (declared[name].get("type") if isinstance(declared[name].get("type"), list) else []):
                    continue
                ok, fitted[name] = (True, None) if given is None else Wire.fit(given, declared[name])
                if not ok:
                    fitted.pop(name)
            return all(name in fitted for name in schema.get("required") or []), fitted
        if kind == "array":
            if isinstance(value, str):
                value = [part.strip() for part in value.split(",") if part.strip()]
            if not isinstance(value, list):
                return False, value
            items = [Wire.fit(item, schema.get("items") or {}) for item in value]
            return True, [item for ok, item in items if ok]
        if kind == "string":
            if isinstance(value, bool):
                value = "true" if value else "false"
            elif isinstance(value, (int, float)):
                value = "%g" % value
            options = schema.get("enum")
            if isinstance(value, str) and isinstance(options, list) and value not in options:
                same = [option for option in options if Text.squash(option) == Text.squash(value)]
                return bool(same), same[0] if same else value
            return isinstance(value, str), value
        if kind in ("number", "integer"):
            found = Figure.number(value)
            if found is None and isinstance(value, str):
                written = Figure.figures(value)
                found = written[0][0] if written else None
            if found is None:
                return False, value
            return True, int(found) if kind == "integer" else found
        if kind == "boolean":
            if isinstance(value, str) and Text.squash(value) in ("true", "false"):
                return True, Text.squash(value) == "true"
            return isinstance(value, bool), value
        return True, value

    @staticmethod
    def cut_short(reply: Any) -> bool:
        first = (reply.get("choices") or [None])[0] if isinstance(reply, dict) else None
        return isinstance(first, dict) and "length" in (first.get("finish_reason"), first.get("native_finish_reason"))

    @staticmethod
    def tool_args(reply: Any, name: str) -> Optional[Dict[str, Any]]:
        first = (reply.get("choices") or [None])[0] if isinstance(reply, dict) else None
        message = first.get("message") if isinstance(first, dict) else None
        if not isinstance(message, dict):
            return None
        if Wire.cut_short(reply) and name == SHORTLIST_TOOL.name:
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                if isinstance(function, dict) and function.get("name") == name:
                    ids = list(dict.fromkeys(re.findall(r'"product_id"\s*:\s*"(\d+)"', str(function.get("arguments") or ""))))
                    if ids:
                        return {"open": [{"product_id": product_id, "why": "named before the answer was cut"} for product_id in ids[:OPEN_ROWS]], "search_instead": []}
            return None
        found = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or function.get("name") != name:
                continue
            args = function.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    continue
            if isinstance(args, dict):
                if name != EXTRACT_TOOL.name:
                    return args
                found.append(args)
        rows = [row for args in found for row in args.get("rows") or []]
        rows = rows[:-1] if Wire.cut_short(reply) else rows
        return {"rows": rows} if found and (rows or not Wire.cut_short(reply)) else None

class Key(NamedTuple):
    product_id: str
    sku: str

class Page(dict):
    brand = property(lambda self: str(self.get("brand") or ""))
    title = property(lambda self: str(self.get("title") or ""))
    path = property(lambda self: [str(label) for label in self.get("category_path") or []])
    shelf = property(lambda self: " > ".join(self.path))
    facts = property(lambda self: self.get("facts") or {})
    model_line = property(lambda self: str(self.facts.get("model") or ""))
    variants = property(lambda self: [variant for variant in self.get("variants") or [] if isinstance(variant, dict)])

    def variant(self, sku: Any) -> Dict[str, Any]:
        return next((variant for variant in self.variants if str(variant.get("sku")) == str(sku)), {})

    def options(self, sku: Any) -> str:
        return str(self.variant(sku).get("options") or "")

    def facts_text(self, model_line: bool = True) -> str:
        return "; ".join("%s = %s" % item for item in self.facts.items() if model_line or Text.tag(item[0]) != "model")

    def render(self) -> str:
        lines = ["product_id=%s | brand field: %s | title: %s" % (self.get("product_id"), self.get("brand") or "(empty)", self.get("title")),
                 "  category_path: %s" % " > ".join(str(label) for label in self.get("category_path") or []),
                 "  facts: %s" % ("; ".join("%s = %s" % (label, value) for label, value in (self.get("facts") or {}).items()) or "(none)")]
        for variant in self.get("variants") or []:
            if isinstance(variant, dict):
                lines.append("  variant sku=%s | options: %s | %s PHP | %s" % (variant.get("sku"), variant.get("options") or "(none)",
                                                                             variant.get("price"), Text.stock_word(variant.get("in_stock"))))
        return "\n".join(lines)

class Must(dict):
    text = property(lambda self: self.get("text"))
    value = property(lambda self: Figure.number(self.get("value")))
    figure = property(lambda self: float(Figure.number(self.get("value")) or 0))
    unit = property(lambda self: Text.squash(self.get("unit")))
    relation = property(lambda self: str(self.get("relation") or "exact"))
    words = property(lambda self: Text.squash(str(self.get("words") or self.get("text") or "")))
    also = property(lambda self: [Text.squash(item) for item in self.get("also") or [] if isinstance(item, str) and item.strip()][:2])

    def worded_in(self, text: Any) -> bool:
        return any(Text.words(value) and Text.words(value) <= Text.words(text) for value in [self.words] + self.also)

    def stated_in(self, text: Any) -> bool:
        return Figure.states(text, self.figure, self.unit, self.relation)

    def other_in(self, text: Any) -> Optional[float]:
        return None if self.covered_in(text) else Figure.another_figure(text, self.figure, self.unit)

    def covered_in(self, text: Any) -> bool:
        return any(low <= self.figure <= high and Figure.unit_like(unit, self.unit) for low, high, unit in Figure.ranges(text))

class Assessment(dict):
    doubts = property(lambda self: self.get("doubts") if isinstance(self.get("doubts"), dict) else {})
    fits = property(lambda self: bool(self.get("fits")))
    ranks = property(lambda self: [line for line in self.get("ranks") or [] if isinstance(line, dict)])

    def line(self, must: Dict[str, Any]) -> Dict[str, Any]:
        for line in self.get("musts") or []:
            if isinstance(line, dict) and (Text.squash(line.get("must")) == Text.squash(must.get("text")) or Text.words(must.get("text")) <= Text.words(line.get("must"))):
                return line
        return {}

class Session:
    def __init__(self, binding: Dict[str, Any], problem_id: str, max_steps: int, clock: "Clock") -> None:
        self.binding = binding
        self.problem_id = problem_id
        self.max_steps = max_steps
        self.clock = clock
        self.turn = 1
        self._generation = 0
        self.dead = False
        self.done = False
        self.dialogue: List[Dict[str, Any]] = []

    def turns_left(self) -> int:
        return self.max_steps - self.turn + 1

    def send(self, actions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for attempt, turn in enumerate((self.turn, self.turn, self.turn + 1)):
            left = STORE_LIMIT - self.clock.elapsed()
            if self.dead or (attempt and left < 12):
                return None
            base = "%s-t%d-g%d" % (self.problem_id, turn, self._generation)
            envelope = {**self.binding, "call_id": base, "idempotency_key": base, "turn": turn,
                        "calls": [{"call_id": "%s-c%d" % (base, i), "action": {"name": a["name"], "args": a["args"]}} for i, a in enumerate(actions, 1)]}
            try:
                result, error = Wire.post("/environment/call", envelope, _proxy if left > 95 else ProxyClient(timeout=max(5, int(left) - 2), max_retries=1))
            except Exception:
                result, error = None, {}
            status = error.get("status")
            if status == 402 or (status in (409, 500, 504) and "quarantined" in str(error.get("body") or "").lower()):
                self.dead = True
                return None
            if isinstance(result, dict) and isinstance(result.get("calls"), list):
                count = result.get("solver_turn_count")
                self.turn = (count if isinstance(count, int) and not isinstance(count, bool) else turn) + 1
                return result
            if attempt:
                self._generation += 1
            if attempt < 2:
                Event().wait(1.0)
        return None


class Clock:
    def __init__(self) -> None:
        self.started = monotonic()

    def elapsed(self) -> float:
        return monotonic() - self.started

class Models:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.dollars = 0.0
        self.flying = 0
        self.lock = Lock()
        self.awaited: set = set()

    def release(self, tokens: List[Any]) -> None:
        with self.lock:
            for token in tokens:
                if token in self.awaited:
                    self.awaited.discard(token)
                    self.flying -= 1

    def spend(self, reply: Any) -> None:
        usage = reply.get("usage") if isinstance(reply, dict) else None
        if not isinstance(usage, dict):
            return
        cost = usage.get("cost")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool):
            cost = (Figure.number(usage.get("prompt_tokens")) or 0) * 0.643e-6 + (Figure.number(usage.get("completion_tokens")) or 0) * 2.02e-6
        with self.lock:
            self.dollars += float(cost)

    def attempt(self, model: Tuple[Any, ...], messages: List[Dict[str, Any]], tool: Tool, seat: Seat, wait: float, bound: bool = False) -> Optional[Dict[str, Any]]:
        name, spec = tool.name, tool.spec()
        body = {"model": Wire.native(model), "messages": messages, "tools": [spec], "tool_choice": {"type": "function", "function": {"name": name}},
                "max_tokens": seat.tokens, **Wire.fields(model)}
        client = ProxyClient(timeout=max(3, int(wait) + 1 if bound else int(max(wait, CLIENT_LIMIT))), max_retries=1)
        error: Dict[str, Any] = {}
        try:
            reply, error = Wire.post("/inference/chat/completions", body, client)
            if reply is None and error.get("status") in (400, 422) and "tool_choice" in str(error.get("body") or ""):
                reply, error = Wire.post("/inference/chat/completions", dict(body, tool_choice="required"), client)
        except Exception:
            reply = None
        self.spend(reply)
        args = Wire.tool_args(reply, name)
        if args is not None and Wire.cut_short(reply) and not Wire.fit(args, spec["function"]["parameters"])[0]:
            args = None
        if args is None and (error.get("kind") == "capacity" or Figure.number(error.get("status")) in (500, 502, 503, 504)):
            Wire.pause(1.5)
        return tool.read(args) if args is not None else None

    def ask(self, seat: Seat, messages: List[Dict[str, Any]], tool: Optional[Tool] = None, deadline: Optional[float] = None, inline: bool = False) -> Optional[Dict[str, Any]]:
        tool = tool or seat.tool
        chain = list(seat.models)
        limit = deadline if deadline is not None else TIME_LIMIT - 10
        while chain:
            left = limit - self.clock.elapsed()
            if left < 5:
                return None
            wait = min(seat.wait, left - 2)
            first = chain.pop(0)
            if inline:
                answer = self.attempt(first, messages, tool, seat, wait * HEDGE_AT if chain else wait, bound=True)
                if answer is not None:
                    return answer
                continue
            partner = chain[0] if chain else None
            box: Queue = Queue()
            started = 0

            outstanding: List[Any] = []

            def launch(model: Tuple[str, str], seconds: float) -> bool:
                token = object()
                with self.lock:
                    if self.flying >= REQUESTS_AT_ONCE:
                        return False
                    self.flying += 1
                    self.awaited.add(token)

                def run() -> None:
                    try:
                        box.put(self.attempt(model, messages, tool, seat, seconds))
                    except Exception:
                        box.put(None)
                    finally:
                        self.release([token])
                try:
                    Thread(target=run, daemon=True).start()
                except RuntimeError:
                    self.release([token])
                    return False
                outstanding.append(token)
                return True

            if launch(first, wait):
                started = 1
            else:
                answer = self.attempt(first, messages, tool, seat, wait, bound=True)
                if answer is not None:
                    return answer
                continue
            begun, answers, until = monotonic(), 0, wait
            while answers < started:
                now = monotonic() - begun
                if partner is not None and started == 1 and now >= wait * HEDGE_AT:
                    spare = min(wait, limit - self.clock.elapsed() - 2)
                    if spare > 3 and launch(partner, spare):
                        started = 2
                        until = now + spare
                    partner = None
                    continue
                timeout = (wait * HEDGE_AT if partner is not None and started == 1 else until) - now + 0.5
                try:
                    answer = box.get(timeout=max(0.05, timeout))
                except Empty:
                    if partner is not None:
                        continue
                    break
                answers += 1
                if answer is not None:
                    self.release(outstanding)
                    return answer
                if partner is not None and started == 1:
                    break
            self.release(outstanding)
            if started == 2 and chain:
                chain.pop(0)
        return None

class Part:
    def __init__(self, episode: "Episode") -> None:
        self.ep = episode

    conditions = property(lambda self: self.ep.conditions)
    store = property(lambda self: self.ep.store)
    queries = property(lambda self: self.ep.queries)
    scout = property(lambda self: self.ep.scout)
    reader = property(lambda self: self.ep.reader)
    judge = property(lambda self: self.ep.judge)
    checkout = property(lambda self: self.ep.checkout)
    session = property(lambda self: self.ep.session)
    models = property(lambda self: self.ep.models)
    clock = property(lambda self: self.ep.clock)
    schemas = property(lambda self: self.ep.schemas)
    per_turn = property(lambda self: self.ep.per_turn)
    request = property(lambda self: self.ep.request)

class Storefront(Part):
    def __init__(self, episode: "Episode") -> None:
        super().__init__(episode)
        self.pages: Dict[str, Page] = {}
        self.opened_order: List[str] = []
        self.live: Dict[Key, Tuple[Any, Any]] = {}
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.row_order: List[str] = []
        self.search_rows: List[Tuple[str, List[str]]] = []
        self.cart: List[Key] = []
        self.first_add: Optional[Key] = None
        self.lost: set = set()
        self.found_by: Dict[str, List[str]] = {}
        self.opened_before_add = 0
        self.list_open = False
        self.read_at: Dict[Key, int] = {}
        self.reading = 0
        self.last_errors: List[Any] = []

    def page(self, product_id: Any) -> Page:
        return self.pages.get(str(product_id)) or Page()

    def in_stock(self, key: Tuple[str, str]) -> Any:
        return self.live.get(key, (None, None))[0]

    def price(self, key: Tuple[str, str]) -> Any:
        return self.live.get(key, (None, None))[1]

    def orderable(self, key: Tuple[str, str]) -> bool:
        named = self.conditions.shortlist_keys()
        return self.in_stock(key) is not False and not Figure.over(self.price(key), self.conditions.budget()) and (not named or "%s::%s" % key in named or self.list_open)

    def sold_out(self) -> List[Key]:
        return [lost for lost in self.lost if self.in_stock(lost) is False]

    def lost_families(self) -> set:
        return {Naming.stem(lost[1]) for lost in self.lost}

    def lost_titles(self) -> set:
        return {Text.squash(self.page(key[0]).title) for key in self.lost}

    def shelf_label(self) -> str:
        named = Text.label_words(self.conditions.record.get("category") or self.conditions.record.get("product") or "")
        if not named:
            return ""
        plain = lambda words: [re.sub(r"s$", "", word) if len(word) > 3 else word for word in words.split()]
        paths = [row.get("category_path") or [] for row in self.rows.values()] + [page.path for page in self.pages.values()]
        for path in paths:
            for label in path[-1:]:
                words = Text.label_words(label)
                if words == named or plain(words) == plain(named):
                    return str(label)
        if len(plain(named)) >= 2:
            for path in paths:
                for label in path[-1:]:
                    if set(plain(named)) <= set(plain(Text.label_words(label))):
                        return str(label)
        return ""

    def shelf_path(self) -> List[str]:
        shelf = self.shelf_label()
        for page in self.pages.values():
            if shelf and page.path and page.path[-1] == shelf:
                return [str(label) for label in page.path]
        return []

    def shelf_parent(self) -> str:
        shelf = self.shelf_label()
        for page in self.pages.values():
            if shelf and len(page.path) >= 2 and page.path[-1] == shelf:
                return str(page.path[-2])
        return ""

    def on_shelf(self, product_id: str) -> bool:
        shelf = self.shelf_label() or str(self.conditions.record.get("category") or "")
        return not Text.label_words(shelf) or any(Text.same_label(shelf, label) for label in self.page(product_id).path)

    def fresh_row(self, product_id: str) -> bool:
        row = self.rows.get(product_id)
        return row is not None and product_id not in self.pages and row.get("in_stock") is not False and not Figure.over(row.get("price"), self.conditions.budget())

    def learn(self, call: Dict[str, Any], obs: Dict[str, Any]) -> None:
        name, args = call["name"], call["args"]
        key = Key(str(args.get("product_id") or ""), str(args.get("sku") or ""))
        if name in (SEARCH, FILTER):
            rows = [row for row in obs.get("results") or [] if isinstance(row, dict)]
            source = "search %s" % Text.dumps(str(args.get("query") or "")) if name == SEARCH else "filter %s" % Text.dumps({k: args[k] for k in ("category", "brand") if args.get(k)})
            for row in rows:
                product_id = str(row.get("product_id"))
                if product_id not in self.rows:
                    self.rows[product_id] = row
                    self.row_order.append(product_id)
                if source not in self.found_by.setdefault(product_id, []):
                    self.found_by[product_id].append(source)
                self.note(product_id, row.get("sku"), row.get("in_stock"), row.get("price"))
            if name == SEARCH:
                self.search_rows.append((str(args.get("query") or ""), [str(row.get("product_id")) for row in rows]))
        elif name in (VIEW, COMPARE):
            for page in ([obs] if name == VIEW else obs.get("compared") or []):
                if isinstance(page, dict) and page.get("product_id"):
                    product_id = str(page["product_id"])
                    if product_id not in self.pages:
                        self.opened_order.append(product_id)
                    self.pages[product_id] = Page(page)
                    for variant in self.pages[product_id].variants:
                        self.note(product_id, variant.get("sku"), variant.get("in_stock"), variant.get("price"))
        elif name == INSPECT_STOCK:
            self.note(key[0], key[1], obs.get("in_stock"), obs.get("price"))
        elif name == ADD:
            added = obs.get("added") if isinstance(obs.get("added"), dict) else {}
            if key not in self.cart:
                self.cart.append(key)
            if self.first_add is None:
                self.opened_before_add = len(self.opened_order)
            self.first_add = self.first_add or key
            self.note(key[0], key[1], added.get("in_stock"), added.get("price"))
        elif name == MESSAGE:
            self.conditions.question_open = False

    def note(self, product_id: Any, sku: Any, in_stock: Any, price: Any) -> None:
        if product_id and sku and isinstance(in_stock, bool):
            self.live[Key(str(product_id), str(sku))] = (in_stock, price)
            self.read_at[Key(str(product_id), str(sku))] = self.reading

    def listing(self, rows: List[Dict[str, Any]]) -> str:
        shown = [row for row in rows if not Figure.over(row.get("price"), self.conditions.budget())]
        lines = ["product_id=%s | %s PHP | %s | brand field: %s | %s | found by: %s" % (
            row.get("product_id"), row.get("price"), Text.stock_word(row.get("in_stock")), row.get("brand") or "(empty)", str(row.get("title") or "")[:160],
            "; ".join((self.found_by.get(str(row.get("product_id"))) or [])[:3]) or "-") for row in shown]
        return "\n".join(lines) or "(no rows within the budget)"

    def constrained(self, name: str, args: Dict[str, Any], cap: Optional[float] = None) -> Dict[str, Any]:
        declared = self.schemas.get(name, {}).get("properties") or {}
        args = {key: value for key, value in args.items() if key in declared and value not in (None, "")}
        if "in_stock" in declared:
            args["in_stock"] = True
        if "k" in declared:
            args["k"] = 10
        if name == FILTER and "max_price" in declared:
            ceiling = [value for value in (cap, self.conditions.budget()) if value is not None]
            if ceiling:
                args["max_price"] = round(min(ceiling), 2)
        return {"name": name, "args": args}

    def search_call(self, query: str) -> Dict[str, Any]:
        return self.constrained(SEARCH, {"query": query})

    def filter_call(self, category: str = "", brand: str = "", cap: Optional[float] = None) -> Dict[str, Any]:
        return self.constrained(FILTER, {"category": category, "brand": brand}, cap)

    def send(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        calls = [call for call in calls if call["name"] in self.schemas][:self.per_turn]
        if not calls or self.session.done or self.session.turns_left() <= 0:
            return []
        turn = self.reading = self.session.turn
        result = self.session.send(calls)
        if result is None:
            raise RuntimeError("the store did not answer")
        self.session.dialogue.append({"role": "environment", "turn": turn, "request": calls, "result": result})
        observations = []
        self.last_errors = []
        for index, call in enumerate(calls):
            outcome = result["calls"][index] if index < len(result["calls"]) and isinstance(result["calls"][index], dict) else {}
            step = outcome.get("observation") if isinstance(outcome.get("observation"), dict) else {}
            if step.get("done") is True:
                self.session.done = True
            obs = step.get("observation") if isinstance(step.get("observation"), dict) else {}
            self.last_errors.append(step.get("error") or obs.get("error"))
            if not step.get("error") and not obs.get("error"):
                self.learn(call, obs)
            observations.append(obs)
        said = result.get("user_message")
        text = str(said.get("content") or "").strip() if isinstance(said, dict) else ""
        if text:
            self.conditions.said.append(text)
            self.conditions.question_open = "?" in text or not self.conditions.pending_question
            self.conditions.pending_question = False
        self.adopt_shelf()
        return observations

    def adopt_shelf(self) -> None:
        label = self.shelf_label()
        if label and label != self.conditions.record.get("category"):
            self.conditions.record["category"] = label


BRIEF_GUIDE = """You keep one shopper's order conditions for a shopping agent that buys one item in an online store; you read the shopper's request and every later line of theirs and write the record the agent shops by, whole, after every line.
- Write what they said, in their words, and nothing they did not say. A later line changes the record: a new ceiling replaces the old budget; a requirement they changed replaces the old one (the old value is gone); a requirement they dropped is gone; a requirement they added is a new must. Their lines come in order; the latest state is the record.
- category: every word of their phrase for the kind of thing, written as a store labels a shelf - capital initials, the plural the store would use, every word kept ('a garden hose' is 'Garden Hoses'; 'an other camera accessory' is 'Other Camera Accessories'; 'a jacket/coat' is 'Jackets/Coats'; 'pens & pencils' is 'Pens & Pencils'). What the item is for or must fit is not part of the shelf's name ('a holder for a road bicycle' names the shelf 'Holders'; the fit is a must). When they name the exact label ('the required catalog category is X') write X exactly as they wrote it. Empty only when they named no kind of thing.
- product: the item noun as they said it, without the article, colour, figure or the use it is for.
- musts: each firm requirement of the item itself, one entry each: those the request states, those they add or change later, a value a hint of theirs pins (what the item must fit or match), and, when they weighed two things and then told which matters more, the one that matters more. kind is figure for a number with a unit, standard for a named standard (a resolution name, a plug or voltage standard written as a name), property for words (a colour, a material, a pattern, a shape, a style, a flavour, a closure type, the make or model it is made for, a piece count written in words). text is their words for it; value is the bare number (null for a property or a standard); unit is the unit word as they wrote it ('l', 'kg', 'inch', 'W'); words is the value alone as a seller writes it in a title or an option, without the property's name or their other words ('in the colour navy' is 'navy', 'bamboo as its material' is 'bamboo'); also is up to two other ways sellers write that same value - another spelling, another common name for it, the same amount in another unit ('gray' for 'grey', 'pu leather' for 'faux leather', '0.5 kg' for '500 g') - and empty when sellers write it one way. A kind or type they ask for in words ('the kind that runs on batteries', 'the type for oily skin') is a must of kind property. A use, an occasion or a user the item itself is made for, named with the item ('a dress for a wedding', 'boots for hiking', 'a bowl for a dog'), is a must of kind property with that word as its value, since sellers title such items by it. Wrong: 'a dress for a wedding' recorded with no must, the wedding taken for the purpose of the purchase. Right: a must 'wedding'. The purpose of the purchase is another thing: a separate statement of whom it is for or why they buy it ('it is a present for a colleague', 'we are moving into a new flat') is no must, and is an open item when they say it may change what the item needs. A requirement stated twice is one entry. The category, the budget and being in stock are not musts; a preference is never a must. The musts you wrote before their latest line are listed below it: write every one of them again unless their latest line removed or replaced it, and list what you removed or replaced in dropped; what their line added comes in beside them.
- budget: the latest ceiling they gave, as a number in PHP; null when none.
- prefer: what they would like but do not require ('ideally', 'preferably', 'I'd love', a brand they lean to), each in their words; empty when none.
- shortlist: the candidates they limited the choice to, exactly as written (product ids with skus, or names); empty when the choice is open.
- ranking: the steps of a ranking rule they gave, in order and in their words ('the largest capacity', 'then the lightest', 'then the lowest price'); empty when none.
- survey: how many options they asked you to look at before ordering; 0 when they named no number.
- reason: true when they ask why the pick is right, for a reason, or to justify it.
- open: what they left open and told you to settle with them before buying, one entry each, with the one question that settles it, written in their frame and in the second person:
  - They name what the purchase is for - a use, an occasion, a person it is for, something it replaces or must match, a plan it belongs to - and say this may change what the item needs (that someone told them, or that they know, exactly what is needed does not make it a withheld requirement while they name what the purchase is for): the question is exactly 'What will you be using the <product> for?' with the product named. Wrong: 'What were you told it needs?', 'What do you normally get?', 'What should it be like?', 'Which colour, size or material do you need?' - a question about what they were told, what they normally get, what the item should be, or a list of properties does not reach what they hold back.
  - They weigh two things and say to check which matters more when both cannot be had: 'Which matters more to you, <A> or <B>?' - one question naming the two.
  - They say they have a requirement in mind they will tell when asked, and name no purpose: 'Before I choose: what is the requirement you have in mind for it?'
  - A hint of theirs fixes a property of the item but you cannot tell its value: one question about that one property.
  - Nothing else is open: not being in stock, not the budget, not a property they stated. Empty when nothing is open. An entry their answer has settled is dropped.
- searches: three searches as a seller would write the item's title: the item word with the stated values as sellers write them, two or four words each; no stock, money or purpose words, no article. When a value has more than one common spelling, one search uses each spelling: the store finds only the spelling the seller wrote. When sellers commonly write that kind of figure in another unit, one search writes the same amount in that unit, worked out by you.
- The record is for the agent alone: write nothing to the shopper here."""

BRIEF_TOOL = Tool({"type": "function", "function": {
    "name": "record", "description": "The shopper's order conditions, whole, as they stand after their latest line.",
    "parameters": {"type": "object", "properties": {
        "budget": {"type": ["number", "null"], "description": "the latest ceiling in PHP as a number; null when none"},
        "category": {"type": "string", "description": "the shelf: every word of their phrase for the kind of thing, as a store labels a shelf; empty only when no kind is named"},
        "dropped": {"type": "array", "items": {"type": "string"}, "description": "the musts of your earlier record that their latest line removed or replaced with another value, each exactly as written there; empty otherwise"},
        "musts": {"type": "array", "description": "each firm requirement of the item itself; a preference is never one", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["figure", "standard", "property"], "description": "figure for a number with a unit, standard for a named standard, property for words"},
            "text": {"type": "string", "description": "the requirement in their words"},
            "unit": {"type": "string", "description": "the unit word as they wrote it; empty for a property or a standard"},
            "value": {"type": ["number", "null"], "description": "the bare number of a figure; null otherwise"},
            "words": {"type": "string", "description": "the value alone as a seller writes it in a title or an option, without the property's name or the shopper's other words ('navy', 'bamboo', '5 m', 'zip closure')"},
            "also": {"type": "array", "maxItems": 2, "items": {"type": "string"}, "description": "up to two other ways sellers write the same value: another spelling, another common name for it, the same amount in another unit; empty when none"}},
            "required": ["also", "kind", "text", "unit", "value", "words"]}},
        "open": {"type": "array", "description": "what they left open and told you to settle with them, one entry each; empty when nothing is open", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["purpose", "priority", "withheld", "hinted"], "description": "purpose: they named what the purchase is for and said it may change what the item needs; priority: they weigh two things and said to check which matters more; withheld: they hold a requirement they will tell when asked; hinted: a hint fixes a property whose value you cannot tell"},
            "question": {"type": "string", "description": "the one question that settles it, in the second person, in the shape the guide gives"},
            "what": {"type": "string", "description": "what is open, in a few words (the purpose they named; which of two matters more; a requirement they hold back; a hinted property)"}},
            "required": ["kind", "question", "what"]}},
        "prefer": {"type": "array", "items": {"type": "string"}, "description": "what they would like but do not require, each in their words"},
        "product": {"type": "string", "description": "the item noun as they said it"},
        "ranking": {"type": "array", "items": {"type": "string"}, "description": "the steps of a ranking rule they gave, in order, in their words; empty when none"},
        "reason": {"type": "boolean", "description": "true when they ask why the pick is right or to justify it"},
        "searches": {"type": "array", "items": {"type": "string"}, "description": "three searches as a seller would write the title"},
        "shortlist": {"type": "array", "items": {"type": "string"}, "description": "the candidates they limited the choice to, exactly as written; empty when the choice is open"},
        "survey": {"type": "integer", "description": "how many options they asked you to look at before ordering; 0 when none"}},
        "required": ["budget", "category", "dropped", "musts", "open", "prefer", "product", "ranking", "reason", "searches", "shortlist", "survey"]}}})

SHORTLIST_GUIDE = """You pick the rows worth opening for a shopping agent buying one item in an online store for one shopper: the store returned rows (a title, a price, stock, a brand field, and which searches or filters returned each), and the agent can open only a few pages per turn.
- Open rows that could be the kind of thing the shopper asked for and could carry every requirement they stated: first the rows the filter on the shelf they named returned ('found by: filter' - the shelf decides what kind of thing they mean, whatever a title calls it), then rows whose titles write the shopper's values (a figure with its unit, a property word, a model or make), then other rows that are that kind of thing. A row's title shows little: a page may state a value its title does not, and a title listing several sizes or values one of which is the shopper's ('5/10/15m' for 10 m) may hold the shopper's variant among its options - open it.
- Leave out rows of another kind of thing. When the summary gives the shelf's place in the store (its path), that path says what kind of thing the shelf holds ('Home > Kitchen > Organizers' holds kitchen organizers, not tool cases): a row of another kind that shares a word with the shelf's name is not the item.
- Pick at most the number of rows the agent asked for. When fewer than three rows could be the item, write search_instead: three or four words each, as a seller titles the item with the shopper's values ('raincoat hood zip'), one of them with another word sellers use for the item; otherwise leave it empty."""

SHORTLIST_TOOL = Tool({"type": "function", "function": {
    "name": "shortlist", "description": "The rows to open now and, when none could be the item, what to search for instead.",
    "parameters": {"type": "object", "properties": {
        "open": {"type": "array", "description": "the rows to open, best first", "items": {"type": "object", "properties": {
            "product_id": {"type": "string", "description": "the row's product_id as written"},
            "why": {"type": "string", "description": "why it could be the item, in a few words"}},
            "required": ["product_id", "why"]}},
        "search_instead": {"type": "array", "items": {"type": "string"}, "description": "only when fewer than three rows could be the item: what to search for, three or four words each; empty otherwise"}},
        "required": ["open", "search_instead"]}}})

EXTRACT_GUIDE = """You copy what shop pages state, for a shopping agent that decides which variant to buy: for each page and each requirement named, the words of the page that state its value, from three places, copied exactly and never inferred.
- facts: the fact under the property's own label, as 'label = value'; when the page has no fact under that label, a fact under a related label; a fact listing several values is copied whole. Empty when no fact states it.
- options: this variant's options string when it states the value (a colour, a size, a capacity, a model with a value); empty otherwise.
- title: the words of the title that state the value, as the title runs them - a figure with its unit, the property word, and when that word stands inside a blend, a list or a composite, the whole of it ('wool silk' for wool, not 'wool'; '5/10/15m' for 10 m); empty when the title says nothing about it.
- Copy; do not judge. Whether what you copied meets the requirement is another reader's work."""

EXTRACT_TOOL = Tool({"type": "function", "function": {
    "name": "extract", "description": "What each page states for each requirement, copied as written.",
    "parameters": {"type": "object", "properties": {
        "rows": {"type": "array", "description": "one row per variant of the pages given", "items": {"type": "object", "properties": {
            "product_id": {"type": "string", "description": "the page's product_id"},
            "sku": {"type": "string", "description": "this variant's sku"},
            "values": {"type": "array", "description": "one entry per requirement named", "items": {"type": "object", "properties": {
                "facts": {"type": "string", "description": "the fact stating it, as 'label = value', copied; empty when none"},
                "options": {"type": "string", "description": "this variant's options when they state it; empty otherwise"},
                "requirement": {"type": "string", "description": "the requirement, exactly as named"},
                "title": {"type": "string", "description": "the title's words that state it, or that name a blend, a list or a composite containing it; empty when none"}},
                "required": ["facts", "options", "requirement", "title"]}}},
            "required": ["product_id", "sku", "values"]}}},
        "required": ["rows"]}}})

JUDGE_GUIDE = """You are a shopping agent buying one item in an online store for one shopper, and you decide which variant to buy: every product decision is yours, made from the pages as they are written. The shopper may send further lines while you shop - an answer, a new budget, a changed requirement, a notice that the item changed - and you follow them before you order.
- Below are the shopper's request, their lines since, the record of their conditions, what has happened (a loss, a changed condition), and a table of candidate variants, each in stock and within the budget. Each row gives the page's brand field, shelf, title, facts and this variant's own options; then, for each requirement, what the page states for it, copied as written; and your earlier judgement of that variant when you judged it before.
- A variant serves the shopper only when its own page states every must with the shopper's value: treat every candidate as wrong until its page shows that.

# Reading a page

## Where a value is stated
- A page states a value in three places: this variant's options, which say which variant arrives; a fact under the property's own label, or under a label naming the part the property belongs to; and the title.
- When this variant's options state the property, they decide it. A title may advertise the listing's other variants: 'Garden Hose 5/10/15 m' above this variant's options '10 m' is a 10 m hose.
- A value only the title states, once and plainly for the item, meets the must: quote the title. It is weaker than the options or a fact, but it is a statement.
- When a fact under the property's own label states one single value and the title another single value, and the options do not settle it, the page contradicts itself: met from the plainer statement, the conflict written in trouble, the page ranked after the clean ones. A title naming a blend that holds the shopper's value is no such conflict: it says what the item is made of, and the blend rule below decides.
- These state nothing for a requirement: a figure inside a model name or code; a figure under a label describing another part; a bare number with no unit, unless its label names the unit.
- What no place states is unknown - never met, never 'probably'.
- The copies under a row point at where the page states each requirement, and may keep only part of it: read the row's own title, facts and options too. A title running the shopper's word into a blend or a list says so even when a copy shows the word alone.

## Several values on one page: four different things
- A list of the values a listing comes in, one per variant ('colour = blue, grey, teal'; 'S/M/L/XL'; '5/10/15 m'), says what the listing offers, not what this variant is: unknown for this variant unless its options pin one. Wrong: a variant with no colour in its options judged teal because its colour fact lists teal among three. Right: unknown - a variant whose options say teal states it.
- A range the item itself covers - a size band, a fit range, an adjustable span ('fits 60-80 kg', 'adjustable 20-40 cm', 'for ages 3-6 years') - holds every value inside it, its ends included: the shopper's value inside it is met. Wrong: a helmet whose options say 'L (head 58-60 cm)' judged unknown for a 60 cm head because the page gives a range. Right: met - the range holds 60 cm.
- A list of the materials of an object's parts ('material = iron, glass, mdf' for a desk; 'frame = oak, seat = linen') says the object has each of them: a shopper asking for a glass desk is met by that desk.
- A blend or a composite - one material, flavour or finish named by words that name two things ('wool silk', 'wool-acrylic', 'wool blend'; a flavour 'garlic butter'; 'faux suede') - is neither thing alone: unmet for wool, for the flavour butter, for suede, however plainly a fact names one of its parts. Judge whether the words name one thing or two. Wrong: a 'Wool Silk Scarf' whose fact says fabric = wool judged wool. Right: unmet - the title sells a blend, and the fact names one of its fibres. A look or feel word is no material either ('suede-look', 'stone effect').
- A title stringing several kinds or styles together ('Sofa Bed Futon Recliner') names none of them for the item: unknown, unless a fact states the type.

## The value they named
- The value the shopper names is exactly the value they want. A nearer, larger or 'better' value is another value unless their own words make a range: 1500 W does not meet 1250 W.
- The same amount in another unit or notation is the same value: work the amount out yourself and ask whether it is the very amount they named (0.5 kg is 500 g, 1.5 l is 1500 ml; a weight never meets a volume).
- A shade, tint or finish of the colour they named is that colour: dark blue, light grey and matte brown are blue, grey and brown. Two colours joined ('blue-orange', 'grey/pink') name neither unless this variant's options pin one.
- A part's value is not the item's: a 'titanium tip' on a brass pen is not a titanium pen, and a 'wool lining' is not a wool coat. When the shopper's words name the part ('a titanium tip'), the part's value states it.

## The shelf, the candidates, the maker, the price
- The shelf the shopper named decides what kind of thing they mean: a listing on that shelf is that kind, whatever its title calls it, and a listing off it is not what they asked for while any candidate on the shelf fits. Never doubt a listing on the named shelf because of what you expected the word to mean.
- A listing is on the named shelf only when one label of its shelf path is that name word for word; a label holding the name inside more words is another shelf.
- When the shopper named the candidates, only those count, each judged on its own page; one that is gone is gone, and the next in their order comes next.
- The brand field says whose listing it is; the title counts only when the field is empty, and a generic seller word in it (OEM, no brand) names no maker. When the shopper requires a maker, only that maker's listings fit; a maker they prefer ranks first among the usable.
- The price is the price: the budget is a ceiling, never a wish for the cheapest. The lower price decides only when nothing else separates two candidates, or where the shopper's ranking rule puts it.

# How to decide
Your judge call writes its fields in this order, and you decide in the same order: assessments, none_fit, ranking, reply, searches, why, winner, and winner_claims when a reason is required. Write the winner only after the ranking and the why.

## assessments - each candidate that could win
- One assessment for each candidate that could win: your likely winner first, then its closest rivals, eight at most - or every candidate the shopper named. Each writes, in this order:
  - key: its key as the table writes it.
  - lines: one line for each must and one for the shelf they named: must (their words), page_value (what the page says the item's value is, read as above: one value, a range the item covers, a blend or a list named as such, a part's value marked as the part's, or 'not stated'), quote (the page's words you rely on, copied exactly, ten words at most; empty when nothing states it) and verdict (met, unmet or unknown). No line for the budget or the stock. A ranking step or a preference is no must: a page stating nothing for it still fits.
  - ranks: only when the shopper gave a ranking rule: for each step, the quote and the number this variant's page states for it, in one unit for every candidate; 0 with an empty quote when it states none.
  - trouble: the strongest reason it may be the wrong item, twelve words at most - another kind of thing, the page contradicting itself, a value stated only in a list; empty when none. A low price is no trouble.
  - usable: true only when every must is met by its own page and its trouble names nothing that makes it another item than the shopper asked for.
- Your earlier judgement of a candidate stands on the table: a must you found unmet there is unmet now unless the page shows you something you missed, and you say what in why.

## none_fit
- true when no candidate is usable. Then write searches, and still name the closest candidate as the winner: the most musts met and none unmet, on the shelf, else off it. When the table is empty, none_fit true and no winner.

## ranking - the usable candidates, best first
- Order them by these criteria, each one only separating candidates that every earlier one left tied:
  1. The shopper's ranking rule, when they gave one: the best on its first step wins, later steps only break ties, the price counts only where they put it.
  2. After a loss (the carted variant sold out or rose above the budget): the same product in another listing, then another product.
  3. What they prefer, given up only in the order they gave.
  4. A page stating each must without contradicting itself before one that contradicts itself.
  5. The plainer statement: this variant's options or a fact under the property's label before the title alone.
  6. The lower price; then the one listed earlier on the table.
- Then the closest of the others, in the same order.

## reply
- When the shopper's latest line asked something or announced a change (a new budget, a changed requirement, a notice that the item sold out or rose in price): one or two plain sentences that answer it in its own terms and say what you now do - the figure their line named that you now keep to, or the item you switch to with its price and stock as you re-checked them. Say only that: never list their requirements, the values you look for or the searches you run - a line naming several requirements reads to them as asking about each of them, and they answer 'one thing at a time'. When nothing fits, name the item you switch to with its price and stock and say nothing about what it lacks: a line telling them a requirement is not met reads to them as asking them to decide about it. A statement that asks nothing back: no question mark, no request to confirm. Empty when their latest line was only an answer to your question.

## searches
- Only when none fit: three searches of three or four words each for other listings that could meet every must, written as sellers title such an item - the kind of thing the closest candidate is, in its title's own words, with the values still unmet as sellers write them; one of them with another word sellers use for the item.

## why
- Why the first of your ranking and no other, in one or two sentences, from the pages' words.

## winner
- The key of the first of your ranking - the variant to buy. When none fit, the closest candidate; when the table is empty, nothing.

## winner_claims - only when a reason is required
- Claims about the winner, each true and copied from its page: the property they asked about under the label of the fact that states it, with the value exactly as that fact writes it (field 'title' or 'options' and the words copied when only they state it); the price as a bare number; in_stock true. Nothing more.

# Answer
- Answer with one call of the judge tool, or of the pick tool when a message asks for it."""

JUDGE_TOOL = Tool({"type": "function", "function": {
    "name": "judge", "description": "The choice among the candidates: each one assessed requirement by requirement from the page's words, then the ranking, the reply and the winner.",
    "parameters": {"type": "object", "properties": {
        "assessments": {"type": "array", "maxItems": 8, "description": "one per candidate that could win, the likely winner first, eight at most; every candidate the shopper named", "items": {"type": "object", "properties": {
            "key": {"type": "string", "description": "product_id::sku as the table writes it"},
            "musts": {"type": "array", "description": "one line for each must and for the shelf they named", "items": {"type": "object", "properties": {
                "must": {"type": "string", "description": "the must in the shopper's words"},
                "page_value": {"type": "string", "description": "what the page says the item's value for this property is, as a buyer reads the whole page: one value, a range the item covers, a blend or a list named as such, a part's value marked as the part's, or 'not stated'"},
                "quote": {"type": "string", "description": "the page's words you rely on, copied exactly, ten words at most; empty when nothing states it"},
                "verdict": {"type": "string", "enum": ["met", "unmet", "unknown"], "description": "met only when page_value is the shopper's value itself for the item, or a range the item covers that holds it; unmet when the page makes the item another value; unknown when nothing states it"}},
                "required": ["must", "page_value", "quote", "verdict"]}},
            "ranks": {"type": "array", "description": "only when the shopper gave a ranking rule: each step's value as the page states it, quoted and as a number in one unit for every candidate", "items": {"type": "object", "properties": {
                "number": {"type": "number", "description": "that value as a number, in one unit for every candidate; 0 when the page states none"},
                "quote": {"type": "string", "description": "the page's words stating this variant's value for the step, copied exactly; empty when none"},
                "step": {"type": "string", "description": "the step of the ranking rule, in the shopper's words"}},
                "required": ["number", "quote", "step"]}},
            "against": {"type": "string", "description": "the strongest reason this may be the wrong item, twelve words at most; empty when none"},
            "fits": {"type": "boolean", "description": "true when every must is met and nothing in trouble makes it another item"}},
            "required": ["key", "musts", "ranks", "against", "fits"]}},
        "none_fit": {"type": "boolean", "description": "true when no candidate is usable; the winner is then the closest, or none when the table is empty"},
        "ranking": {"type": "array", "items": {"type": "string"}, "description": "the candidates' keys best first: the usable ones, then the closest others"},
        "reply": {"type": "string", "description": "one or two sentences to the shopper: a statement, no question mark, answering their latest line in its own terms; empty when they said nothing new"},
        "searches": {"type": "array", "items": {"type": "string"}, "description": "only when none fits: what to search for next, three or four words each, as sellers title such an item with the unmet values"},
        "why": {"type": "string", "description": "why the winner, in one or two sentences, from the page's words"},
        "winner": {"type": "string", "description": "the key of the chosen candidate as the table writes it; empty when the table is empty"},
        "winner_claims": {"type": "array", "description": "only when the shopper asked for a reason: the claims of the order", "items": {"type": "object", "properties": {
            "field": {"type": "string", "description": "the page's own fact label, or title, options, price, in_stock"},
            "value": {"type": ["string", "number", "boolean"], "description": "the fact's value or the words copied exactly; the price as a number; in_stock true"}},
            "required": ["field", "value"]}}},
        "required": ["assessments", "none_fit", "ranking", "reply", "searches", "why", "winner", "winner_claims"]}}},
    glm_names={"against": "trouble", "fits": "usable", "musts": "lines"})

PICK_TOOL = Tool({"type": "function", "function": {
    "name": "pick", "description": "The choice alone, when the full judgement did not come.",
    "parameters": {"type": "object", "properties": {
        "why": {"type": "string", "description": "why, in a few words"},
        "winner": {"type": "string", "description": "the key of the chosen candidate as the table writes it"}},
        "required": ["why", "winner"]}}})
PICK_NOTE = "Answer now with one call of the pick tool: why in a few words, then the winner - the key of the candidate that meets the most of the shopper's requirements, on the shelf they named."

CLAIMS_GUIDE = """You write the justification of one order that a shopping agent places for one shopper in an online store, when the shopper asked for a reason: short claims about the ordered variant that the page and the store's latest reading state.
- One claim for each requirement the shopper stated: the page's own fact label as field and that fact's value copied exactly as value (when only the title or this variant's options state it, field 'title' or 'options' and the words copied). Then price with the latest price as a number and in_stock with true.
- Nothing the page does not state, nothing beyond these: every extra claim is a risk and adds nothing."""

CLAIMS_TOOL = Tool({"type": "function", "function": {
    "name": "claims", "description": "The claims of the order, each one the page or the latest reading states.",
    "parameters": {"type": "object", "properties": {
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "field": {"type": "string", "description": "the page's own fact label, or title, options, price, in_stock"},
            "value": {"type": ["string", "number", "boolean"], "description": "the value as the page or the latest reading states it"}},
            "required": ["field", "value"]}}},
        "required": ["claims"]}}})

BRIEF = Seat(BRIEF_GUIDE, BRIEF_TOOL, tokens=1536, wait=45, models=RECORD_MODELS)
SHORTLIST = Seat(SHORTLIST_GUIDE, SHORTLIST_TOOL, tokens=768, wait=30, models=SHORTLIST_MODELS)
EXTRACT = Seat(EXTRACT_GUIDE, EXTRACT_TOOL, tokens=1792, wait=60, models=EXTRACT_MODELS)
JUDGE = Seat(JUDGE_GUIDE, JUDGE_TOOL, tokens=3072, wait=60, models=JUDGE_MODELS)
PICK = Seat(JUDGE_GUIDE, PICK_TOOL, tokens=256, wait=30, models=JUDGE_MODELS)
CLAIMS = Seat(CLAIMS_GUIDE, CLAIMS_TOOL, tokens=512, wait=30, models=CLAIMS_MODELS)


class Conditions(Part):
    def __init__(self, episode: "Episode") -> None:
        super().__init__(episode)
        self.record: Dict[str, Any] = {}
        self.said: List[str] = []
        self.heard = 0
        self.asked_texts: List[str] = []
        self.pending_question = False
        self.unsettled = False
        self.question_open = False
        self.changed_musts: List[Must] = []
        self.dropped_musts: List[Must] = []
        self.budget_dropped = False
        self.settled: List[str] = []

    def budget(self) -> Optional[float]:
        return Figure.number(self.record.get("budget"))

    def musts(self) -> List[Must]:
        return [Must(dict(item, relation="exact")) for item in self.record.get("musts") or [] if isinstance(item, dict) and re.search(r"[a-z0-9]", str(item.get("text") or ""), re.I)]

    def figure_musts(self) -> List[Must]:
        return [must for must in self.musts() if must.get("kind") == "figure" and must.value is not None and must.unit]

    def prefer(self) -> List[str]:
        return [str(item) for item in self.record.get("prefer") or [] if str(item).strip()]

    def shortlist_keys(self) -> set:
        return {item for item in (str(item) for item in self.record.get("shortlist") or []) if Naming.is_key(item)}

    def shortlist_names(self) -> List[str]:
        return [item for item in (str(item).strip() for item in self.record.get("shortlist") or []) if item and not Naming.is_key(item)]

    def steps(self) -> List[str]:
        return [str(item) for item in self.record.get("ranking") or [] if str(item).strip()]

    def survey(self) -> int:
        return int(Figure.number(self.record.get("survey")) or 0)

    def reason(self) -> bool:
        return bool(self.record.get("reason"))

    def product(self) -> str:
        return str(self.record.get("product") or "").strip()

    def requirement_names(self) -> List[str]:
        names = [str(must.get("text")) for must in self.musts()] + self.steps()
        return list(dict.fromkeys(name for name in names if name.strip()))[:12]

    def open_items(self) -> List[Dict[str, Any]]:
        return [item for item in self.record.get("open") or [] if isinstance(item, dict) and str(item.get("question") or "").strip()]

    def question(self) -> str:
        if len(self.asked_texts) >= 2:
            return ""
        asked = {Text.squash(text) for text in self.asked_texts}
        settled = {Text.squash(text) for text in self.settled}
        for item in self.open_items():
            text = str(item.get("question") or "").strip()
            if item.get("kind") == "purpose":
                text = "What will you be using the %s for?" % (self.product() or "item")
            elif item.get("kind") == "withheld":
                text = "Before I choose: what is the requirement you have in mind for it?"
            if Text.squash(text) in asked or Text.squash(str(item.get("what") or "")) in settled:
                continue
            if "?" not in text:
                text = text.rstrip(".") + "?"
            return text
        if len(self.asked_texts) == 1 and (self.pending_question or self.unsettled) and self.open_items():
            for text in ("What will you be using the %s for?" % (self.product() or "item"), "Before I choose: what is the requirement you have in mind for it?"):
                if Text.squash(text) not in asked:
                    return text
        return ""

    def write_conditions(self) -> None:
        lines = "\n".join("- %s" % text for text in self.said) or "(nothing yet)"
        earlier = "\n\nThe musts you wrote before their latest line (each stays unless that line removed or replaced it): %s" % ("; ".join(str(must.get("text")) for must in self.musts()) or "(none)") if self.record and self.said else ""
        messages = BRIEF.messages("The shopper's request:\n%s\n\nWhat the shopper has said since, in order:\n%s%s" % (self.request, lines, earlier))
        answer = self.models.ask(BRIEF, messages, deadline=WRAP_UP_AT)
        before = dict(self.record)
        self.heard = len(self.said)
        if answer is None:
            if not self.record:
                self.record = {"budget": Figure.money(self.request)[-1] if Figure.money(self.request) else None, "category": "", "musts": [], "open": [],
                               "prefer": [], "product": " ".join(self.request.split()[:6]), "ranking": [], "reason": False, "searches": [self.request[:80]], "shortlist": [], "survey": 0}
            return
        record = {key: answer.get(key) for key in BRIEF_TOOL.parameters["properties"]}
        record["musts"] = [item for item in record.get("musts") or [] if isinstance(item, dict) and re.search(r"[a-z0-9]", str(item.get("text") or ""), re.I)]
        if before.get("musts"):
            dropped = {Text.squash(text) for text in record.get("dropped") or [] if str(text).strip()}
            kept_texts = {Text.squash(item.get("text")) for item in record["musts"]}
            new_units = {(Text.squash(item.get("unit")), Figure.number(item.get("value"))) for item in record["musts"] if item.get("kind") == "figure" and item.get("unit")}
            for item in before["musts"]:
                text = Text.squash(item.get("text"))
                replaced = item.get("kind") == "figure" and item.get("unit") and any(unit == Text.squash(item.get("unit")) and value != Figure.number(item.get("value")) for unit, value in new_units)
                if text and text not in kept_texts and text not in dropped and not any(text in Text.squash(d) or Text.squash(d) in text for d in dropped) and not replaced:
                    record["musts"].append(item)
        if before.get("category") and self.store.shelf_label() == before.get("category"):
            record["category"] = before["category"]
        self.record = record
        self.mark_changes(before)

    def mark_changes(self, before: Dict[str, Any]) -> None:
        old = {Text.squash(item.get("text")): Must(dict(item, relation="exact")) for item in before.get("musts") or [] if isinstance(item, dict)}
        new = {Text.squash(must.get("text")): must for must in self.musts()}
        same = lambda one, other: one.value == other.value and one.unit == other.unit and Text.squash(one.get("text")) == Text.squash(other.get("text"))
        self.changed_musts = [must for key, must in new.items() if key not in old or not same(must, old[key])] if before else []
        self.dropped_musts = [must for key, must in old.items() if key not in new] if before else []
        old_budget, new_budget = Figure.number(before.get("budget")), self.budget()
        if before and old_budget is not None and new_budget is not None and new_budget < old_budget:
            self.budget_dropped = True
        if self.pending_question:
            if self.changed_musts or self.dropped_musts or (before and new_budget != old_budget):
                asked = Text.squash(self.asked_texts[-1]) if self.asked_texts else ""
                for item in (before.get("open") or []):
                    if isinstance(item, dict) and Text.squash(str(item.get("question") or "")) == asked:
                        self.settled.append(str(item.get("what") or ""))
            self.pending_question = False

    def budget_cut(self, text: str, current: Optional[float]) -> Optional[float]:
        prices = {Figure.number(self.store.price(key)) for key in self.store.cart}
        below = {amount for amount in Figure.money(text) if current is not None and amount < current and amount not in prices}
        return below.pop() if len(below) == 1 else None

    def hear(self) -> bool:
        if self.heard >= len(self.said) or self.clock.elapsed() > WRAP_UP_AT:
            return False
        watched = ("musts", "budget", "category", "prefer", "shortlist", "ranking", "survey")
        before = Text.dumps({key: self.record.get(key) for key in watched})
        cut = self.budget_cut(self.said[-1], self.budget())
        self.write_conditions()
        if cut is not None and (self.budget() is None or self.budget() > cut):
            self.record["budget"] = cut
            self.budget_dropped = True
        return before != Text.dumps({key: self.record.get(key) for key in watched})

    def catch_up(self) -> None:
        self.hear()

    def changed(self) -> bool:
        return bool(self.changed_musts or self.dropped_musts or self.budget_dropped)

    def settle_changes(self) -> None:
        self.changed_musts, self.dropped_musts, self.budget_dropped = [], [], False


class Queries(Part):
    @staticmethod
    def spelling(must: Must) -> Tuple[str, str]:
        number = Figure.number(must.get("value"))
        text = ("%g" % number) if number is not None else str(must.get("value") or "")
        unit = str(must.get("unit") or "").strip()
        return ("%s%s" % (text, unit)).strip(), ("%s %s" % (text, unit)).strip()

    def value_of(self, must: Must) -> str:
        if must.get("kind") == "figure" and must.value is not None:
            return self.spelling(must)[0]
        return must.words

    def values(self) -> List[str]:
        return [value for value in (self.value_of(must) for must in self.conditions.musts()) if value]

    def other_values(self) -> List[str]:
        musts = self.conditions.musts()
        return [" ".join((self.joined(other) if changed.get("kind") == "figure" else other) if index == place else self.value_of(must) for index, must in enumerate(musts))
                for place, changed in enumerate(musts) for other in changed.also]

    def shelf_queries(self, limit: int) -> List[str]:
        label = self.store.shelf_label() or str(self.conditions.record.get("category") or "")
        words = Text.label_words(label)
        values = " ".join(dict.fromkeys(" ".join(self.values()).split()))
        product = Text.squash(self.conditions.product())
        out: List[str] = []
        others = self.other_values()
        if words and values:
            out.append("%s %s" % (words, values))
            out += ["%s %s" % (words, other) for other in others]
        parent = self.store.shelf_parent()
        if parent and values:
            out.append("%s %s %s" % (Text.label_words(parent), words, values))
        label_words = [word for word in words.split() if len(word) > 2]
        if values and len(label_words) > 1:
            out += ["%s %s" % (word, values) for word in (label_words[0], label_words[-1])]
        if product and values and Text.bag("%s %s" % (product, values)) not in {Text.bag(query) for query in out}:
            out.append("%s %s" % (product, values))
        for value in self.values():
            if words:
                out.append("%s %s" % (words, value))
        for must in self.conditions.figure_musts():
            joined, apart = self.spelling(must)
            if product:
                out.append("%s %s" % (product, apart))
        if values:
            out.append(values)
        seen: set = set()
        kept = []
        for query in out:
            bag = Text.bag(query)
            if query.strip() and bag not in seen:
                seen.add(bag)
                kept.append(" ".join(query.split()))
        return kept[:limit + len(others)]

    @staticmethod
    def label_forms(label: str) -> List[str]:
        forms = [""]
        for part in re.split(r"(\s*[&/,]\s*)", label.strip()):
            words = part.split()
            if not words or re.fullmatch(r"\s*[&/,]\s*", part):
                options = [part]
            else:
                last = words[-1]
                options = [part, " ".join(words[:-1] + [last[:-1] if last.lower().endswith("s") and len(last) > 3 else last + "s"])]
            forms = [form + option for form in forms for option in options]
        return list(dict.fromkeys(forms))[:4]

    @staticmethod
    def joined(text: str) -> str:
        return re.sub(r"(\d)\s+([^\W\d_]+)", r"\1\2", text)

    def model_queries(self) -> List[str]:
        searches = [str(query) for query in self.conditions.record.get("searches") or [] if str(query).strip()][:3]
        return list(dict.fromkeys(searches + [self.joined(query) for query in searches]))

    def sister(self, key: Key) -> str:
        title = self.store.page(key[0]).title
        return " ".join((" ".join(Text.plain(title).split()[:8]) + " " + " ".join(self.values())).split())

    def siblings(self, limit: int) -> List[str]:
        musts = self.conditions.musts()
        out: List[str] = []
        for key, entry in self.judge.assessed.items():
            if key[0] not in self.store.pages or not self.store.on_shelf(key[0]):
                continue
            short = [must for must in musts if entry.line(must).get("verdict") != "met"]
            if len(short) != 1:
                continue
            must = short[0]
            value = self.value_of(must)
            out.append(" ".join(("%s %s" % (" ".join(Text.plain(self.store.page(key[0]).title).split()[:6]), value)).split()))
        return list(dict.fromkeys(out))[:limit]


class Scout(Part):
    def __init__(self, episode: "Episode") -> None:
        super().__init__(episode)
        self.instead: List[str] = []
        self.shelf_rows: List[str] = []
        self.sister_query = ""
        self.sent: set = set()

    def ask(self, calls: List[Dict[str, Any]]) -> None:
        question = self.conditions.question()
        if not question or MESSAGE not in self.schemas or any(call["name"] in (ADD, ORDER, MESSAGE) for call in calls):
            return
        if self.session.turn == 1 and self.conditions.record.get("open") == [] and not self.conditions.open_items():
            return
        calls.insert(0, {"name": MESSAGE, "args": {"content": question}})
        self.conditions.asked_texts.append(question)
        self.conditions.pending_question = True

    def fresh(self, queries: List[str]) -> List[Dict[str, Any]]:
        calls = []
        for query in queries:
            bag = Text.bag(query)
            if query.strip() and bag not in self.sent:
                self.sent.add(bag)
                calls.append(self.store.search_call(" ".join(query.split())[:120]))
        return calls

    def filters(self, cap: Optional[float] = None) -> List[Dict[str, Any]]:
        adopted = self.store.shelf_label()
        shelf = adopted or str(self.conditions.record.get("category") or "").strip()
        if not shelf or FILTER not in self.schemas:
            return []
        return [dict(self.store.filter_call(category=label, cap=cap), shelf=True) for label in ([shelf] if adopted else self.queries.label_forms(shelf))]

    @staticmethod
    def lists_figure(title: str, must: Must) -> bool:
        if not must.unit or must.value is None:
            return False
        for group, unit in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?(?:\s*[/\-]\s*\d+(?:\.\d+)?)+)\s*([a-z]+)", Text.plain(title)):
            numbers = {float(number) for number in re.findall(r"\d+(?:\.\d+)?", group)}
            if must.figure in numbers and Figure.unit_like(Text.squash(unit), must.unit):
                return True
        return False

    def collect(self, calls: List[Dict[str, Any]], observations: List[Dict[str, Any]]) -> None:
        for call, obs in zip(calls, observations):
            if call.get("shelf"):
                for row in obs.get("results") or []:
                    product_id = str(row.get("product_id") or "")
                    if product_id and product_id not in self.shelf_rows:
                        self.shelf_rows.append(product_id)

    def first_look(self) -> None:
        conditions = self.conditions
        calls: List[Dict[str, Any]] = []
        keyed = sorted({key.split("::")[0] for key in conditions.shortlist_keys()})
        if keyed and COMPARE in self.schemas:
            calls += [{"name": COMPARE, "args": {"product_ids": keyed[i:i + 5]}} for i in range(0, len(keyed), 5)]
        queries = self.queries.model_queries() + self.queries.shelf_queries(5) + [name for name in conditions.shortlist_names()[:4]]
        if not queries:
            queries = [conditions.product() or self.request[:80]]
        calls += self.filters() + self.fresh(queries)
        self.ask(calls)
        calls = calls[:self.per_turn]
        before = set(self.store.rows)
        self.collect(calls, self.store.send([{"name": call["name"], "args": call["args"]} for call in calls]))

    def shortlist(self, limit: int = OPEN_ROWS, lead: Optional[List[str]] = None, note: str = "") -> List[str]:
        conditions = self.conditions
        unopened = [product_id for product_id in self.store.row_order if product_id not in self.store.pages and self.store.fresh_row(product_id)]
        if not unopened:
            return []
        rows = [self.store.rows[product_id] for product_id in unopened[:80]]
        summary = "Kind of thing: %s (shelf %s). Requirements: %s. Budget: %s PHP. Candidates the shopper named: %s. Preferences: %s." % (
            conditions.product() or "(not named)", " > ".join(self.store.shelf_path()) or conditions.record.get("category") or "(not named)", "; ".join(str(must.get("text")) for must in conditions.musts()) or "(none)",
            conditions.budget() if conditions.budget() is not None else "no limit", ", ".join(sorted(conditions.shortlist_keys()) + conditions.shortlist_names()) or "(none)", "; ".join(conditions.prefer()) or "(none)")
        messages = SHORTLIST.messages("%s\n%sOpen at most %d rows.\n\nRows:\n%s" % (summary, note + "\n" if note else "", limit, self.store.listing(rows)))
        answer = self.models.ask(SHORTLIST, messages, deadline=WRAP_UP_AT)
        picked = [str(item.get("product_id")) for item in (answer or {}).get("open") or [] if isinstance(item, dict) and str(item.get("product_id")) in self.store.rows]
        self.instead = [str(query) for query in (answer or {}).get("search_instead") or [] if str(query).strip()][:3]
        title_of = lambda product_id: str(self.store.rows[product_id].get("title") or "")
        figured = [product_id for product_id in unopened if any(must.stated_in(title_of(product_id)) or self.lists_figure(title_of(product_id), must) for must in conditions.figure_musts())
                   or any(must.worded_in(title_of(product_id)) for must in conditions.musts() if must.get("kind") == "property")]
        front = [product_id for product_id in (lead or []) + self.shelf_rows if product_id in unopened][:SHELF_ROWS + len(lead or [])]
        ordered = list(dict.fromkeys(front + picked + figured + ([product_id for product_id in unopened if product_id in self.shelf_rows] if not picked else [])))
        if not ordered:
            ordered = unopened[:limit]
        return ordered[:limit]

    def value_rows(self, limit: int = 6) -> List[str]:
        conditions = self.conditions
        title_of = lambda product_id: str(self.store.rows[product_id].get("title") or "")
        named = Text.words(conditions.product()) | Text.words(str(conditions.record.get("category") or ""))
        spellings = [value for must in conditions.musts() for value in ([*self.queries.spelling(must)] if must.get("kind") == "figure" and must.value is not None else [self.queries.value_of(must)]) + must.also]
        values = [words for words in (Text.words(value) - named for value in spellings) if words]

        def evidence(product_id: str) -> Tuple[int, int]:
            title = title_of(product_id)
            searched = sum(1 for query, found in self.store.search_rows if product_id in found and any(words <= Text.words(query) for words in values))
            titled = sum(1 for must in conditions.figure_musts() if must.stated_in(title) or self.lists_figure(title, must)) + sum(
                1 for must in conditions.musts() if must.get("kind") == "property" and must.worded_in(title))
            return searched, titled

        unopened = [product_id for product_id in self.store.row_order if product_id not in self.store.pages and self.store.fresh_row(product_id)]
        scored = [(evidence(product_id), product_id) for product_id in unopened]
        ranked = sorted((pair for pair in scored if pair[0][0] or pair[0][1]), key=lambda pair: (-pair[0][0] - 2 * pair[0][1], -pair[0][1]))
        return [product_id for _, product_id in ranked][:limit]

    def open_values(self) -> List[str]:
        if self.clock.elapsed() > HURRY_AT or self.session.turns_left() < 6:
            return []
        return self.open(self.value_rows())

    def open(self, ids: List[str]) -> List[str]:
        ids = [product_id for product_id in dict.fromkeys(ids) if product_id not in self.store.pages][:OPEN_ROWS + 2]
        if not ids or self.clock.elapsed() > HURRY_AT:
            return []
        if COMPARE in self.schemas:
            calls = [{"name": COMPARE, "args": {"product_ids": ids[i:i + 5]}} for i in range(0, len(ids), 5)]
        elif VIEW in self.schemas:
            calls = [{"name": VIEW, "args": {"product_id": product_id}} for product_id in ids]
        else:
            return []
        self.store.send(calls)
        return [product_id for product_id in ids if product_id in self.store.pages]

    def research(self, queries: List[str], limit: int = OPEN_ROWS, note: str = "", lead_from: Optional[List[str]] = None) -> List[str]:
        if self.clock.elapsed() > HURRY_AT or self.session.turns_left() < 4:
            return []
        calls = self.filters() + self.fresh(queries + self.queries.shelf_queries(6))
        if calls:
            self.collect(calls, self.store.send([{"name": call["name"], "args": call["args"]} for call in calls[:self.per_turn]]))
        lead: List[str] = []
        for query in lead_from or []:
            found = next((rows for sent, rows in reversed(self.store.search_rows) if Text.bag(sent) == Text.bag(query)), [])
            lead += [product_id for product_id in found if product_id not in self.store.pages and self.store.fresh_row(product_id)][:2]
        opened = self.open(self.shortlist(limit=limit, lead=list(dict.fromkeys(lead))[:4], note=note))
        return opened + self.open_values()

    def after_answer(self) -> List[str]:
        return self.research(self.queries.model_queries(), note="The shopper has just told the agent a further requirement; rows that could carry it matter most.")

    def replacement_reads(self) -> List[str]:
        lost = [key for key in self.store.lost if key[0] in self.store.pages]
        queries: List[str] = []
        if lost:
            self.sister_query = self.queries.sister(lost[-1])
            queries.append(self.sister_query)
        queries += self.queries.model_queries() + self.queries.siblings(2)
        calls = self.fresh(queries + self.queries.shelf_queries(6)) + self.filters()
        if calls and self.clock.elapsed() < HURRY_AT:
            self.collect(calls, self.store.send([{"name": call["name"], "args": call["args"]} for call in calls[:self.per_turn]]))
        sisters = next((found for query, found in self.store.search_rows if query == self.sister_query), []) if self.sister_query else []
        titles, stems = self.store.lost_titles(), self.store.lost_families()
        kin = [product_id for product_id in self.store.row_order if product_id not in self.store.pages and self.store.fresh_row(product_id)
               and (Naming.stem(str(self.store.rows[product_id].get("sku") or "")) in stems or Text.squash(self.store.rows[product_id].get("title")) in titles)]
        lead = list(dict.fromkeys([product_id for product_id in sisters if product_id in kin] + kin))[:4]
        note = "The variant carted before is gone (sold out, or priced above the budget): the same product in another listing comes first when it still meets every requirement, then another product that does."
        opened = self.open(self.shortlist(limit=REPLACEMENT_OPENS, lead=lead, note=note))
        return opened + self.open_values()

    def survey_reads(self) -> None:
        wanted = self.conditions.survey()
        if wanted <= 0:
            return
        for _ in range(2):
            opened = [product_id for product_id in self.store.pages if self.store.on_shelf(product_id)]
            if len(opened) >= wanted or self.clock.elapsed() > HURRY_AT:
                return
            more = [product_id for product_id in self.shelf_rows + self.store.row_order if product_id not in self.store.pages and self.store.fresh_row(product_id)]
            if not more:
                more_calls = self.filters()
                if more_calls:
                    self.collect(more_calls, self.store.send([{"name": call["name"], "args": call["args"]} for call in more_calls]))
                more = [product_id for product_id in self.shelf_rows if product_id not in self.store.pages]
            if not more or not self.open(more[:min(OPEN_ROWS, wanted - len(opened) + 2)]):
                return


class Reader(Part):
    def __init__(self, episode: "Episode") -> None:
        super().__init__(episode)
        self.evidence: Dict[Key, Dict[str, Any]] = {}
        self.extracted: set = set()

    def extract(self, ids: List[str]) -> None:
        fresh = [product_id for product_id in ids if product_id in self.store.pages and product_id not in self.extracted]
        if not fresh:
            return
        self.extracted |= set(fresh)
        names = self.conditions.requirement_names()
        if not names:
            return
        chunks = [fresh[i:i + EXTRACT_PAGES] for i in range(0, len(fresh), EXTRACT_PAGES)][:EXTRACTS_AT_ONCE * 2]
        parameters = json.loads(json.dumps(EXTRACT.tool.parameters))
        parameters["properties"]["rows"]["items"]["properties"]["values"]["items"]["properties"]["requirement"]["enum"] = names
        tool = EXTRACT.tool.variant(parameters)

        def job(chunk: List[str]) -> Optional[Dict[str, Any]]:
            pages = "\n\n".join(self.store.pages[product_id].render() for product_id in chunk)
            messages = EXTRACT.messages("Requirements to copy: %s\n\nThe pages:\n\n%s" % ("; ".join(names), pages))
            return self.models.ask(EXTRACT, messages, tool, deadline=WRAP_UP_AT - 25, inline=True)

        results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)
        for start in range(0, len(chunks), EXTRACTS_AT_ONCE):
            batch = list(range(start, min(start + EXTRACTS_AT_ONCE, len(chunks))))
            threads = []
            for index in batch:
                def run(index: int = index) -> None:
                    try:
                        results[index] = job(chunks[index])
                    except Exception:
                        results[index] = None
                thread = Thread(target=run, daemon=True)
                try:
                    thread.start()
                    threads.append(thread)
                except RuntimeError:
                    run()
            until = self.clock.elapsed() + min(60.0, TIME_LIMIT - 5 - self.clock.elapsed())
            for thread in threads:
                thread.join(timeout=max(1.0, until - self.clock.elapsed()))
        for chunk, answer in zip(chunks, results):
            for row in (answer or {}).get("rows") or []:
                if not isinstance(row, dict):
                    continue
                key = Key(str(row.get("product_id")), str(row.get("sku")))
                if key[0] not in chunk or key not in self.store.live:
                    continue
                given = {Text.squash(value.get("requirement")): value for value in row.get("values") or [] if isinstance(value, dict)}
                self.evidence[key] = {"values": [{"requirement": name, **{place: str((given.get(Text.squash(name)) or {}).get(place) or "") for place in ("facts", "options", "title")}} for name in names]}

    def cell(self, key: Key, must: Dict[str, Any]) -> Optional[Dict[str, str]]:
        evidence = self.evidence.get(key)
        if not evidence:
            return None
        return next((value for value in evidence["values"] if Text.squash(value.get("requirement")) == Text.squash(must.get("text"))), None)

    def copied(self, key: Tuple[str, str], must: Must) -> Tuple[List[str], Optional[float]]:
        cell = self.cell(key, must)
        if cell is None:
            return [], None
        tail = lambda place: str(cell.get(place) or "").split("=", 1)[-1]
        present = [place for place in ("options", "facts", "title") if cell.get(place) and (must.stated_in(tail(place)) or must.covered_in(tail(place)))]
        raw = self.store.page(key[0]).options(key[1])
        clash = next((found for text in (tail("options"), raw) if text for found in [must.other_in(text)] if found is not None), None)
        return present, clash

    def figure_note(self, key: Tuple[str, str], must: Must) -> str:
        cell = self.cell(key, must)
        if cell is None:
            return ""
        said = ("%g %s" % (must.figure, must.get("unit") or "")).strip()
        present, clash = self.copied(key, must)
        tail = lambda place: str(cell.get(place) or "").split("=", 1)[-1]
        if present:
            inside = [place for place in present if not must.stated_in(tail(place))]
            return "the shopper's %s appears in: %s%s%s" % (said, ", ".join(present), " (inside a range in %s)" % ", ".join(inside) if inside else "",
                                                          " - but this variant's options write %g" % clash if clash is not None else "")
        other = next(((place, found) for place in ("options", "facts", "title") if cell.get(place) for found in [must.other_in(tail(place))] if found is not None), None)
        if other:
            return "the copies write %g, not the shopper's %s (in %s)" % (other[1], said, other[0])
        return "no copy writes %s as such - whether the page states that very amount another way is yours to read" % said

    def copies_prove(self, key: Tuple[str, str], must: Must, places: Tuple[str, ...] = ("options", "facts")) -> bool:
        present, clash = self.copied(key, must)
        return clash is None and bool(set(places) & set(present))

    def page_texts(self, key: Key) -> List[str]:
        page = self.store.page(key[0])
        return [page.title, page.facts_text(), page.options(key[1])]

    def candidates(self) -> List[Key]:
        store, conditions = self.store, self.conditions
        named = conditions.shortlist_keys()
        changed = store.first_add is not None and conditions.changed()
        order = store.opened_order[store.opened_before_add:] + store.opened_order[:store.opened_before_add] if changed else list(store.opened_order)
        keys: List[Key] = []
        for product_id in order:
            own = [Key(product_id, str(variant.get("sku"))) for variant in store.pages[product_id].variants]
            keys += [key for key in own if key not in store.lost and store.orderable(key) and (not named or "%s::%s" % key in named)][:PAGE_ROWS]
        if conditions.record.get("category") and not named:
            shelved = [key for key in keys if store.on_shelf(key[0])]
            keys = shelved or keys
        musts = conditions.figure_musts()
        if musts:
            keys.sort(key=lambda key: (not all(self.copies_prove(key, must) for must in musts),
                                       not all(self.copies_prove(key, must, ("options", "facts", "title")) for must in musts)))
        sold = store.sold_out()
        if sold and not named:
            stems = {Naming.stem(lost[1]) for lost in sold}
            titles = store.lost_titles()
            keys.sort(key=lambda key: not (Naming.stem(key[1]) in stems or Text.squash(store.page(key[0]).title) in titles))
        return keys

    def table(self, keys: List[Key]) -> str:
        store = self.store
        lost_titles, lost_stems = store.lost_titles(), store.lost_families()
        lines = []
        for index, key in enumerate(keys[:TABLE_ROWS], 1):
            page = store.pages[key[0]]
            marks = []
            if Text.squash(page.title) in lost_titles:
                marks.append("SAME TITLE AS THE LOST LISTING")
            if Naming.stem(key[1]) in lost_stems:
                marks.append("SAME SKU FAMILY AS THE LOST VARIANT")
            if self.conditions.record.get("category"):
                marks.append("ON THE SHELF THE SHOPPER NAMED" if store.on_shelf(key[0]) else "NOT ON THE SHELF THE SHOPPER NAMED")
            lines.append("[%d] key=%s::%s | %s PHP, in stock | brand field: %s | shelf: %s | this variant's options: %s%s\n    title: %s\n    facts: %s" % (
                index, key[0], key[1], store.price(key), page.brand or "(empty)", page.shelf, page.options(key[1]) or "(none)", "".join(" | %s" % mark for mark in marks),
                page.title[:200], (page.facts_text() or "(none)")[:500]))
            evidence = self.evidence.get(key)
            musts = {Text.squash(must.get("text")): must for must in self.conditions.figure_musts()}
            if evidence:
                for cell in evidence["values"]:
                    places = "; ".join("%s: %s" % (place, cell[place][:100]) for place in ("options", "facts", "title") if cell.get(place)) or "nothing stated"
                    must = musts.get(Text.squash(cell["requirement"]))
                    note = self.figure_note(key, must) if must is not None else ""
                    lines.append("    %s -> %s%s" % (cell["requirement"], places, " -> " + note if note else ""))
            earlier = self.judge.assessed.get(key)
            if earlier is not None:
                for line in earlier.get("musts") or []:
                    if isinstance(line, dict) and line.get("verdict") in ("unmet", "unknown"):
                        lines.append("    your earlier judgement: %s -> %s (%s)" % (line.get("must"), line.get("verdict"), str(line.get("page_value") or "")[:80]))
        more = len(keys) - TABLE_ROWS
        return "\n".join(lines) + ("\n(%d more variants of these pages not listed)" % more if more > 0 else "")


class Judge(Part):
    def __init__(self, episode: "Episode") -> None:
        super().__init__(episode)
        self.backups: List[Key] = []
        self.reply = ""
        self.claims: Dict[Key, List[Any]] = {}
        self.assessed: Dict[Key, Assessment] = {}

    def situation(self) -> str:
        conditions, store = self.conditions, self.store
        notes = []
        for must in conditions.changed_musts:
            notes.append("The shopper has just added or changed this requirement: %s." % Text.dumps(must.get("text")))
        for must in conditions.dropped_musts:
            notes.append("The shopper no longer requires: %s." % Text.dumps(must.get("text")))
        if conditions.budget_dropped:
            notes.append("The shopper lowered the budget to %s PHP; the table holds only variants within it." % conditions.budget())
        for key in sorted(store.lost):
            in_stock, price = store.live.get(key, (None, None))
            notes.append("Carted before and lost: %s::%s (%s)." % (key[0], key[1], "sold out" if in_stock is False else "now %s PHP, above the budget" % price))
        if conditions.shortlist_keys() or conditions.shortlist_names():
            notes.append("The shopper named the candidates: only those count.")
        if conditions.steps():
            notes.append("The shopper's ranking rule, in order: %s." % "; ".join(conditions.steps()))
        if conditions.reason():
            notes.append("The shopper asked for a reason: write winner_claims from the winner's page.")
        if conditions.said:
            notes.append("Their latest line: %s" % Text.dumps(conditions.said[-1]))
        return "\n".join(notes) or "Nothing has changed since the request."

    def key_of(self, text: Any, keys: List[Key]) -> Optional[Key]:
        return Naming.find_key(text, keys)

    def verify(self, answer: Dict[str, Any], keys: List[Key]) -> Tuple[Optional[Key], str]:
        key = self.key_of(answer.get("winner"), keys)
        if key is None:
            ranked = [found for found in (self.key_of(item, keys) for item in answer.get("ranking") or []) if found is not None]
            return (ranked[0] if ranked else None), ""
        entries = self.entries(answer, keys)
        mine = entries.get(key)
        if self.conditions.record.get("category") and not self.store.on_shelf(key[0]):
            on_shelf = [other for other, entry in entries.items() if self.store.on_shelf(other[0]) and entry.fits]
            if on_shelf:
                return key, ("Your winner %s::%s stands off the shelf the shopper named (%s), while %s::%s stands on it and your own lines call it usable. A listing off that shelf is not what they asked for while one on it fits: choose among the candidates on the shelf, or say in why what makes the one off it the shopper's item." % (
                    key[0], key[1], self.conditions.record.get("category"), on_shelf[0][0], on_shelf[0][1]))
        if mine is not None:
            texts = " ".join(self.reader.page_texts(key))
            for must in self.conditions.figure_musts():
                line = mine.line(must)
                quote = Text.plain(Text.unlabelled(line.get("quote") or ""))
                if line.get("verdict") == "met" and quote and not Text.within(quote, texts):
                    return key, ("Your line for %s quotes %s, which the page of %s::%s does not carry. Quote the page's own words that state the shopper's figure for this variant, or mark it unknown and judge again." % (
                        Text.dumps(must.get("text")), Text.dumps(quote), key[0], key[1]))
                clash = self.reader.copied(key, must)[1] if line.get("verdict") == "met" else None
                if clash is not None:
                    said = ("%g %s" % (must.figure, must.get("unit") or "")).strip()
                    return key, ("Your winner %s::%s: this variant's options write %g %s, not the shopper's %s - the options say which variant arrives. Choose a variant whose page states %s for itself, or say in why what makes this one the shopper's." % (
                        key[0], key[1], clash, must.get("unit") or "", said, said))
        return key, ""

    def entries(self, answer: Dict[str, Any], keys: List[Key]) -> Dict[Key, Assessment]:
        out: Dict[Key, Assessment] = {}
        for entry in answer.get("assessments") or []:
            found = self.key_of(entry.get("key"), keys) if isinstance(entry, dict) else None
            if found is not None:
                out[found] = Assessment(entry)
        return out

    def decide(self, keys: List[Key]) -> Tuple[Optional[Key], Optional[Dict[str, Any]]]:
        table = self.reader.table(keys)
        messages = JUDGE.messages("The shopper's request:\n%s\n\nWhat they said since, in order:\n%s\n\nThe record of their conditions: %s\n\n%s\n\nCandidates:\n%s" % (
            self.request, "\n".join("- " + text for text in self.conditions.said) or "(nothing)", Text.dumps(self.conditions.record), self.situation(), table))
        answer = self.models.ask(JUDGE, messages, deadline=WRAP_UP_AT + 10)
        if answer is None:
            picked = self.models.ask(PICK, messages + [{"role": "user", "content": PICK_NOTE}], deadline=WRAP_UP_AT + 10)
            key = self.key_of((picked or {}).get("winner"), keys)
            return key, ({"winner": picked.get("winner"), "why": picked.get("why"), "assessments": [], "ranking": []} if key is not None and picked else None)
        key, problem = self.verify(answer, keys)
        if problem and self.clock.elapsed() < WRAP_UP_AT - 20:
            messages.append({"role": "user", "content": "Your answer was: %s\n\n%s Answer again with the judge tool." % (Text.dumps(JUDGE.tool.show(answer))[:6000], problem)})
            second = self.models.ask(JUDGE, messages, deadline=WRAP_UP_AT + 10)
            if second is not None:
                answer = second
                key, _ = self.verify(second, keys)
        return key, answer

    def adopt(self, key: Optional[Key], answer: Optional[Dict[str, Any]], keys: List[Key]) -> None:
        if answer is None:
            return
        self.assessed.update(self.entries(answer, keys))
        ranked = [found for found in (self.key_of(item, keys) for item in answer.get("ranking") or []) if found is not None]
        self.backups = [other for other in ranked if other != key]
        reply = str(answer.get("reply") or "").strip()
        self.reply = reply if reply and "?" not in reply else ""
        if key is not None and self.conditions.reason():
            self.claims[key] = [claim for claim in answer.get("winner_claims") or [] if isinstance(claim, dict)]

    def fallback_key(self, keys: List[Key]) -> Optional[Key]:
        shelved = [key for key in keys if self.store.on_shelf(key[0])]
        return (shelved or keys)[0] if keys else None

    def choose(self) -> Optional[Key]:
        keys = self.reader.candidates()
        if not keys and self.clock.elapsed() < HURRY_AT:
            opened = self.scout.research(self.scout.instead + self.queries.model_queries(), note="Nothing opened so far could be sold to the shopper: the rows that could carry every requirement matter most.")
            self.reader.extract(opened)
            keys = self.reader.candidates()
        if not keys:
            return None
        key, answer = self.decide(keys)
        rounds = 0
        while answer is not None and answer.get("none_fit") and rounds < 2 and self.clock.elapsed() < HURRY_AT - 60 * rounds and self.session.turns_left() >= 5:
            rounds += 1
            searches = [str(query) for query in answer.get("searches") or [] if str(query).strip()][:3]
            opened = self.scout.research(searches + self.queries.siblings(2) + self.scout.instead, note="No candidate opened so far meets every requirement: open the rows that could state the values still unmet.", lead_from=searches)
            if not opened:
                break
            self.reader.extract(opened)
            wider = self.reader.candidates()
            if len(wider) <= len(keys):
                break
            key2, answer2 = self.decide(wider)
            if key2 is None:
                break
            key, answer, keys = key2, answer2, wider
        if key is None:
            key = self.fallback_key(keys)
        self.adopt(key, answer, keys)
        return key


class Checkout(Part):
    def add(self, key: Key) -> bool:
        if key in self.store.cart:
            return self.store.orderable(key)
        calls = [{"name": ADD, "args": {"product_id": key[0], "sku": key[1]}}]
        if INSPECT_STOCK in self.schemas:
            calls.append({"name": INSPECT_STOCK, "args": {"product_id": key[0], "sku": key[1]}})
        self.store.send(calls)
        if not self.store.orderable(key):
            self.store.lost.add(key)
            return False
        return True

    def shown_claims(self, key: Key, claims: List[Any]) -> List[Dict[str, Any]]:
        page = self.store.page(key[0])
        facts = page.facts
        in_stock, price = self.store.in_stock(key), self.store.price(key)
        kept = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            field, value = Text.squash(claim.get("field") or ""), Text.squash(claim.get("value") or "")
            if field == "price":
                amount = Figure.number("".join(ch for ch in value if ch.isdigit() or ch == "."))
                if amount is not None and amount == Figure.number(price):
                    kept.append({"field": "price", "value": Figure.number(price)})
            elif field == "in_stock":
                if value == Text.squash(in_stock) and in_stock is True:
                    kept.append({"field": "in_stock", "value": True})
            elif field == "brand":
                if value and Text.plain(value) == Text.plain(page.brand):
                    kept.append({"field": "brand", "value": page.brand})
            elif field in ("title", "options"):
                text = page.title if field == "title" else page.options(key[1])
                if value and Text.within(value, text):
                    kept.append({"field": field, "value": str(claim.get("value"))})
            else:
                label = next((label for label in facts if Text.tag(label) == Text.tag(field)), None)
                if label is not None and Text.within(value, str(facts[label])):
                    kept.append({"field": str(label), "value": str(facts[label])})
        return kept

    def claims_for(self, key: Key) -> List[Dict[str, Any]]:
        kept = self.shown_claims(key, self.judge.claims.get(key) or [])
        needed = [str(must.get("text")) for must in self.conditions.musts()]
        covered = all(any(Text.within(Text.squash(str(must.get("text"))).split()[-1], str(claim.get("value"))) or Text.within(str(claim.get("value")), " ".join(self.reader.page_texts(key))) for claim in kept if Text.squash(claim["field"]) not in ("price", "in_stock", "brand")) for must in self.conditions.musts()) if needed else True
        if (kept and covered) or self.clock.elapsed() > TIME_LIMIT - 25:
            return kept
        page = self.store.page(key[0])
        messages = CLAIMS.messages("The shopper's request: %s\nThe requirements to cite: %s\n\nThe ordered variant: sku=%s, latest price %s, in_stock %s\n%s" % (
            self.request, "; ".join(needed) or "(none)", key[1], self.store.price(key), Text.dumps(self.store.in_stock(key)), page.render()))
        answer = self.models.ask(CLAIMS, messages, deadline=TIME_LIMIT - 8)
        written = self.shown_claims(key, list((answer or {}).get("claims") or []))
        return written + [claim for claim in kept if Text.squash(claim["field"]) not in {Text.squash(c["field"]) for c in written}]

    def reply_text(self, key: Key) -> str:
        if self.judge.reply and "?" not in self.judge.reply:
            return self.judge.reply
        page = self.store.page(key[0])
        lost = "The one carted before is no longer available, so " if self.store.lost else ""
        return "%sI am ordering the %s at %s PHP, in stock, which meets the requirements as they stand." % (lost, " ".join(page.title.split()[:8]) or "chosen item", self.store.price(key))

    def pre_order(self, key: Key) -> bool:
        calls: List[Dict[str, Any]] = []
        if self.conditions.question_open and MESSAGE in self.schemas:
            calls.append({"name": MESSAGE, "args": {"content": self.reply_text(key)[:1500]}})
        if INSPECT_STOCK in self.schemas:
            calls.append({"name": INSPECT_STOCK, "args": {"product_id": key[0], "sku": key[1]}})
        if not calls:
            return self.store.orderable(key)
        self.store.send(calls)
        self.conditions.catch_up()
        if not self.store.orderable(key):
            self.store.lost.add(key)
            return False
        return True

    def order(self, key: Key) -> None:
        args: Dict[str, Any] = {"product_id": key[0], "sku": key[1]}
        if self.conditions.reason():
            claims = self.claims_for(key)
            in_stock, price = self.store.in_stock(key), self.store.price(key)
            if price is not None and not any(Text.squash(claim.get("field")) == "price" for claim in claims):
                claims.append({"field": "price", "value": Figure.number(price)})
            if in_stock is True and not any(Text.squash(claim.get("field")) == "in_stock" for claim in claims):
                claims.append({"field": "in_stock", "value": True})
            args["justification"] = {"claims": claims}
        self.store.send([{"name": ORDER, "args": args}])

    def bare_order(self) -> None:
        store = self.store
        if self.session.done or self.session.dead or ORDER not in self.schemas:
            return
        keys = [key for key in reversed(store.cart) if store.orderable(key)] + [key for key in reversed(store.cart)]
        for key in dict.fromkeys(keys):
            if self.session.done or self.session.turns_left() < 1:
                return
            store.send([{"name": ORDER, "args": {"product_id": key[0], "sku": key[1]}}])

    def close(self) -> None:
        store, conditions = self.store, self.conditions
        if self.session.done or self.session.dead:
            return
        if not store.rows and not store.pages and SEARCH in self.schemas and self.session.turns_left() >= 3:
            store.send([store.search_call(str(conditions.product() or self.request)[:120])])
        conditions.catch_up()
        target = next((key for key in reversed(store.cart) if store.orderable(key)), None)
        tried: set = set()
        while target is None and not self.session.done and self.session.turns_left() >= 2:
            keys = [key for key in self.reader.candidates() if key not in tried and key not in store.cart]
            if not keys:
                named = conditions.shortlist_keys()
                keys = [key for key in store.live if key not in tried and key not in store.cart and store.in_stock(key) is True and not Figure.over(store.price(key), conditions.budget())
                        and (not named or "%s::%s" % key in named or store.list_open)]
                if not keys and named and not store.list_open:
                    store.list_open = True
                    continue
            keys.sort(key=lambda key: (not (key[0] in store.pages and store.on_shelf(key[0])), -len(store.found_by.get(key[0], []))))
            pick = next((key for key in self.judge.backups if key in keys), None) or (self.judge.fallback_key(keys) if keys else None)
            if pick is None:
                break
            tried.add(pick)
            view = [{"name": VIEW, "args": {"product_id": pick[0]}}] if pick[0] not in store.pages and VIEW in self.schemas else []
            store.send(view + [{"name": ADD, "args": {"product_id": pick[0], "sku": pick[1]}}] + ([{"name": INSPECT_STOCK, "args": {"product_id": pick[0], "sku": pick[1]}}] if INSPECT_STOCK in self.schemas else []))
            conditions.catch_up()
            if pick in store.cart and store.orderable(pick):
                target = pick
        if target is None and store.cart:
            target = store.cart[-1]
        tries = 0
        while target is not None and not self.session.done and not self.session.dead and self.session.turns_left() > 0 and tries < 3:
            tries += 1
            if self.session.turns_left() >= 2 and not self.pre_order(target):
                other = next((key for key in reversed(store.cart) if key != target and store.orderable(key)), None)
                if other is None:
                    break
                target = other
                continue
            if conditions.changed() and self.session.turns_left() >= 3 and self.clock.elapsed() < WRAP_UP_AT:
                nxt = self.judge.choose()
                conditions.settle_changes()
                if nxt is not None and nxt != target and self.add(nxt):
                    target = nxt
                    continue
            self.order(target)
            if self.session.done:
                break
            conditions.catch_up()
            if conditions.changed() or not store.orderable(target):
                nxt = self.judge.choose() if conditions.changed() else None
                conditions.settle_changes()
                if nxt is not None and nxt != target and self.add(nxt):
                    target = nxt
                elif not store.orderable(target):
                    target = next((key for key in reversed(store.cart) if store.orderable(key)), None) or target


class Episode:
    def __init__(self, problem_data: Dict[str, Any]) -> None:
        environment = problem_data.get("environment") if isinstance(problem_data.get("environment"), dict) else {}
        view = environment.get("policy_view") if isinstance(environment.get("policy_view"), dict) else {}
        binding = environment.get("binding") if isinstance(environment.get("binding"), dict) else {}
        self.problem_id = str(problem_data.get("problem_id") or "problem")
        self.clock = Clock()
        self.models = Models(self.clock)
        self.request = str(view.get("query") or "")
        self.session = Session(dict(binding), self.problem_id, int(Figure.number(view.get("max_steps")) or 30), self.clock)
        self.schemas: Dict[str, Dict[str, Any]] = {}
        for tool in view.get("tools") or []:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                self.schemas[function["name"]] = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
        self.per_turn = max(1, min(int(Figure.number(view.get("max_calls_per_turn")) or 16), 16))
        self.conditions = Conditions(self)
        self.store = Storefront(self)
        self.queries = Queries(self)
        self.scout = Scout(self)
        self.reader = Reader(self)
        self.judge = Judge(self)
        self.checkout = Checkout(self)

    def late(self) -> bool:
        return self.clock.elapsed() > WRAP_UP_AT or self.session.turns_left() <= 4 or self.session.dead

    def run(self) -> None:
        conditions, store, scout, reader, judge, checkout = self.conditions, self.store, self.scout, self.reader, self.judge, self.checkout
        conditions.write_conditions()
        scout.first_look()
        heard = conditions.hear()
        conditions.unsettled = bool(conditions.asked_texts) and not conditions.changed_musts and bool(conditions.open_items())
        if heard or conditions.pending_question or conditions.unsettled:
            if (conditions.pending_question or conditions.unsettled) and conditions.question() and not self.late():
                scout.ask(calls := [])
                if calls:
                    store.send(calls)
                    conditions.hear()
            if conditions.changed():
                scout.after_answer()
        picked = scout.shortlist()
        scout.open(picked)
        scout.open_values()
        scout.survey_reads()
        conditions.catch_up()
        reader.extract(store.opened_order)
        choice = judge.choose()
        conditions.settle_changes()
        rounds = 0
        while choice is not None and not self.session.done and not self.late():
            checkout.add(choice)
            conditions.catch_up()
            if store.orderable(choice) and choice in store.cart and not conditions.changed():
                break
            if not store.orderable(choice):
                store.lost.add(choice)
            rounds += 1
            if rounds > 2:
                break
            opened = scout.replacement_reads()
            reader.extract(opened or store.opened_order)
            conditions.catch_up()
            nxt = judge.choose()
            conditions.settle_changes()
            if nxt is None or nxt == choice:
                break
            choice = nxt
        checkout.close()


def agent_main(problem_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    episode = Episode(problem_data)
    try:
        episode.run()
    except Exception as exc:
        episode.session.dialogue.append({"role": "assistant", "content": "stopped: %s" % type(exc).__name__})
        try:
            if not episode.session.done:
                episode.checkout.close()
        except Exception:
            try:
                episode.checkout.bare_order()
            except Exception:
                pass
    return episode.session.dialogue
