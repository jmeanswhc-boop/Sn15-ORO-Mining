from __future__ import annotations

import difflib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

MODELS = {
    "director": ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"],
    "judge": ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"],
    "reviewer": ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"],
    "reviewer_light": ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"],
    "chooser": ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"],
}

REASONING = {
    "director": {"effort": "low"},
    "judge": {"enabled": False},
    "reviewer": {"effort": "minimal"},
    "reviewer_light": {"effort": "low"},
    "chooser": {"effort": "low"},
}

PRICES = {
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "openai/gpt-6-luna": (0.10, 0.50),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
}

RUN_SPEND_CAP = 5.0
RUN_TASKS = 90
SPEND_BASE = 0.81 * RUN_SPEND_CAP / RUN_TASKS
STEP_NORM = 40.0
SPEND_SCALE_MAX = 2.0
FINAL_RESERVE = 0.004
COST_MARGIN = 1.12
COST_GUESS = {"director": 0.0025, "judge": 0.0015, "reviewer": 0.03, "reviewer_light": 0.002, "chooser": 0.0015}
STRONG_REVIEW_SHARE = 0.75

TIME_LIMIT = 268.0
END_RESERVE = 40.0
TOKEN_LIMIT = 450000
MODEL_TRIES = 2
WORLD_TRIES = 3
REQUEST_GAP = 0.12
MAX_ACTIONS = 8
JUDGE_WORKERS = 4
FINAL_STEPS = 5
MIN_STEP_LIMIT = 8
REVIEW_REJECTS = 2
HIST_VIEW_CHARS = 3600
RECORD_VIEW_CHARS = 3600
RECORD_TEXT_WIDTH = 600
CHAIN_TEXT_CHARS = 420
HIST_MAX_CHARS = 60000
NOTES_CHARS = 2200
LISTING_LINES = 24
SEARCH_LINES = 12
ASK_LINES = 16
CHOOSER_CALLS = 2
TABLES_TO_ASK = 3
ASSUMED_MESSAGES = 2


def now() -> float:
    return monotonic()


def clip(text: Any, n: int) -> str:
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    return text if len(text) <= n else text[: n - 3] + "..."


def dumps(o: Any, n: int = 0) -> str:
    s = json.dumps(o, ensure_ascii=False, default=str)
    return clip(s, n) if n else s


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def plain(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", str(text or "").lower())).strip()


def letters(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def find_json(text: str, outer: bool = False) -> Any:
    if not isinstance(text, str):
        return None
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fence:
        s = fence.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i = s.find(opener)
        while i != -1:
            depth = 0
            in_str = False
            esc = False
            for j in range(i, len(s)):
                c = s[j]
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
                        try:
                            return json.loads(s[i : j + 1])
                        except Exception:
                            break
            if outer:
                return None
            i = s.find(opener, i + 1)
    return None


def pick_json(text: str, keys: tuple[str, ...]) -> Any:
    if not isinstance(text, str):
        return None
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    for body in ([fence.group(1).strip()] if fence else []) + [s]:
        i = body.find("{")
        while i != -1:
            depth = 0
            in_str = False
            esc = False
            for j in range(i, len(body)):
                c = body[j]
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
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            o = json.loads(body[i : j + 1])
                        except Exception:
                            o = None
                        if isinstance(o, dict) and any(k in o for k in keys):
                            return o
                        break
            i = body.find("{", i + 1)
    pairs = re.findall(r'<parameter\s+name="([A-Za-z_]+)"\s*>(.*?)</parameter>', s, re.S)
    if pairs:
        out: dict[str, Any] = {}
        for k, v in pairs:
            try:
                out[k] = json.loads(v.strip())
            except Exception:
                out[k] = v.strip()
        if any(k in out for k in keys):
            return out
    return None


def content_of(content: Any, cache: bool) -> Any:
    if not isinstance(content, list):
        return content
    if not cache:
        return "".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    out = []
    for p in content:
        part = {"type": "text", "text": str(p.get("text") or "")}
        if p.get("cache"):
            part["cache_control"] = {"type": "ephemeral"}
        out.append(part)
    return out


def with_tail(content: Any, tail: str) -> Any:
    if isinstance(content, list):
        return list(content) + [{"text": tail}]
    return str(content) + tail


def shrink(o: Any, width: int = 220, items: int = 12, depth: int = 0) -> Any:
    if isinstance(o, str):
        return o if len(o) <= width else o[: width - 3] + "..."
    if isinstance(o, list):
        out = [shrink(x, width, items, depth + 1) for x in o[:items]]
        if len(o) > items:
            out.append("(%d more not shown)" % (len(o) - items))
        return out
    if isinstance(o, dict):
        inner = width if depth < 2 else max(60, width // 2)
        return {str(k): shrink(v, inner, items, depth + 1) for k, v in o.items()}
    return o


class Log:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, kind: str, **fields: Any) -> None:
        rec = {"kind": kind}
        rec.update(fields)
        self.items.append(rec)


class Wire:
    def __init__(self, binding: dict[str, Any], max_steps: int, log: Log) -> None:
        self.client = ProxyClient(timeout=50, max_retries=2, retry_delay=1.0, rate_limit_retry_delay=3.0)
        self.binding = dict(binding)
        self.max_steps = int(max_steps)
        self.log = log
        self.t0 = now()
        self.step = 0
        self.done = False
        self.calls = 0
        self.tokens = 0
        self.spend = 0.0
        self.review_spend = 0.0
        self.spend_seen = False
        self.failed_streak = 0
        self.dead = False
        self.plain: set[tuple[str, str]] = set()
        self.last_request = 0.0
        self.role_spend: dict[str, list[float]] = {}
        self.last_finish = ""
        self.nocache = False
        self.cached_tokens = 0
        self.lock = threading.Lock()
        self.set_steps(self.max_steps)

    def set_steps(self, n: int) -> None:
        self.max_steps = int(n)
        scale = min(SPEND_SCALE_MAX, max(1.0, self.max_steps / STEP_NORM))
        self.limit = SPEND_BASE * scale
        self.token_limit = TOKEN_LIMIT * scale

    def elapsed(self) -> float:
        return now() - self.t0

    def time_left(self) -> float:
        return TIME_LIMIT - self.elapsed()

    def steps_left(self) -> int:
        return max(0, self.max_steps - self.step)

    def over_budget(self) -> bool:
        return self.spend > self.limit - FINAL_RESERVE or self.tokens > self.token_limit

    def affords(self, role: str) -> bool:
        if role == "reviewer" and self.review_spend + COST_GUESS["reviewer"] > STRONG_REVIEW_SHARE * self.limit:
            return False
        return self.spend + COST_GUESS.get(role, 0.003) <= self.limit - FINAL_RESERVE

    def pace(self) -> None:
        with self.lock:
            at = max(now(), self.last_request + REQUEST_GAP)
            self.last_request = at
        gap = at - now()
        if gap > 0:
            sleep(gap)

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.done:
            return {"ok": False, "obs": None, "finished": True, "note": "the visit has already ended"}
        if self.steps_left() <= 0:
            self.done = True
            return {"ok": False, "obs": None, "finished": True, "note": "no store steps left"}
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        res = None
        for attempt in range(WORLD_TRIES):
            self.pace()
            res = self.client.post_verbose("/environment/call", payload)
            if res.ok:
                break
            status = (res.error or {}).get("status")
            body = str((res.error or {}).get("body") or "")
            m = re.search(r"expected (\d+)", body)
            if status == 409 and m:
                expected = int(m.group(1))
                if expected > payload["step"]:
                    self.step = expected - 1
                    payload["step"] = expected
                    continue
            if status == 409 and "finished" in body:
                self.done = True
                return {"ok": False, "obs": None, "finished": True, "note": "the visit has already ended"}
            if status in (400, 404, 422):
                return {"ok": False, "obs": None, "finished": False, "note": clip("the store rejected the call: %s" % body, 400)}
            sleep(0.6 * (attempt + 1))
        if res is None or not res.ok:
            self.log.add("world_error", name=name, args=args, error=(res.error if res else None))
            return {"ok": False, "obs": None, "finished": False, "note": "the store did not answer"}
        data = res.data or {}
        self.step = int(data.get("step") or (self.step + 1))
        obs = data.get("observation")
        finished = bool(data.get("finished"))
        self.done = self.done or finished
        self.log.add("call", step=self.step, name=name, args=args, observation=clip(dumps(obs), 4000), finished=finished)
        return {"ok": True, "obs": obs, "finished": finished}

    def count_usage(self, data: dict[str, Any], model: str) -> None:
        usage = data.get("usage") or {}
        costs = []
        try:
            pin, pout = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
            cached = min(pin, int(details.get("cached_tokens") or 0))
            written = min(pin - cached, int(details.get("cache_write_tokens") or 0))
            self.cached_tokens += cached
            self.tokens += pin - cached * 9 // 10 + pout
            price = PRICES.get(str(data.get("model") or model)) or PRICES.get(model)
            if price:
                costs.append(COST_MARGIN * ((pin - cached - written) * price[0] + cached * price[0] * 0.1 + written * price[0] * 1.25 + pout * price[1]) / 1e6)
        except Exception:
            pass
        for v in (usage.get("cost"), (usage.get("cost_details") or {}).get("upstream_inference_cost") if isinstance(usage.get("cost_details"), dict) else None):
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                costs.append(float(v))
        if costs:
            self.spend += max(costs)
            self.spend_seen = True
            return max(costs)
        return 0.0

    def infer(self, role: str, messages: list[dict[str, Any]], max_tokens: int) -> str | None:
        if self.dead or self.time_left() < 6:
            return None
        chain = MODELS.get(role) or MODELS["director"]
        for model in chain:
            for attempt in range(MODEL_TRIES):
                cache = not self.nocache and model.startswith("anthropic/") and isinstance(messages[0].get("content"), str)
                sent = [dict(m, content=content_of(m.get("content"), cache)) for m in messages]
                if cache:
                    sent = [{"role": "system", "content": [{"type": "text", "text": messages[0]["content"], "cache_control": {"type": "ephemeral"}}]}] + sent[1:]
                body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0.0, "max_tokens": max_tokens}
                if role in REASONING and (role, model) not in self.plain:
                    body["reasoning"] = REASONING[role]
                self.pace()
                res = self.client.post_verbose("/inference/chat/completions", body)
                with self.lock:
                    self.calls += 1
                if res.ok and isinstance(res.data, dict) and res.data.get("choices"):
                    with self.lock:
                        cost = self.count_usage(res.data, model)
                        if role == "reviewer":
                            self.review_spend += cost
                        self.role_spend.setdefault(role, []).append(cost)
                    choice = res.data["choices"][0] or {}
                    self.last_finish = str(choice.get("finish_reason") or "")
                    msg = choice.get("message") or {}
                    text = str(msg.get("content") or "")
                    if text.strip():
                        self.failed_streak = 0
                        return text
                    self.log.add("model_empty", role=role, model=model, finish=choice.get("finish_reason"))
                    if self.over_budget():
                        return None
                    break
                status = (res.error or {}).get("status") if res.error else None
                if status == 429:
                    if self.time_left() < 15:
                        return None
                    sleep(min(2.0 + 2.0 * attempt, 6.0))
                    continue
                if status in (400, 422) and cache:
                    self.nocache = True
                    self.log.add("cache_off", role=role, model=model, status=status)
                    continue
                if status in (400, 422) and "reasoning" in body:
                    self.plain.add((role, model))
                    continue
                self.failed_streak += 1
                self.log.add("model_error", role=role, model=model, status=status, body=clip(str((res.error or {}).get("body") or ""), 300))
                if self.failed_streak >= 8:
                    self.dead = True
                    return None
                if status in (400, 403, 404):
                    break
                if self.time_left() < 15:
                    return None
                sleep(min(0.8 * (attempt + 1), 1.5))
        return None

    def turns_left(self) -> int:
        costs = self.role_spend.get("director") or []
        per = max(sum(costs) / len(costs), 0.0008) if costs else COST_GUESS["director"]
        return max(0, int((self.limit - FINAL_RESERVE - self.spend) / per))

    def ask(self, role: str, system: str, user: Any, max_tokens: int, keys: tuple[str, ...] = ()) -> dict[str, Any] | None:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(2):
            text = self.infer(role, messages, max_tokens)
            if text is None:
                return None
            parsed = pick_json(text, keys) if keys else None
            if parsed is None:
                parsed = find_json(text, outer=bool(keys))
            if isinstance(parsed, dict) and (not keys or any(k in parsed for k in keys)):
                return parsed
            if self.last_finish == "length" or (keys and not isinstance(parsed, dict)):
                max_tokens = int(max_tokens * 1.5)
                messages = [{"role": "system", "content": system}, {"role": "user", "content": with_tail(user, "\n\nYOUR LAST REPLY WAS CUT OFF BEFORE ITS JSON WAS COMPLETE. Keep the notes shorter this time and reply with the complete JSON object only.")}]
                continue
            messages = messages + [{"role": "assistant", "content": text}, {"role": "user", "content": "Reply with the JSON object only."}]
        return None


def schema_problem(schema: Any, value: Any, path: str = "args") -> str | None:
    if not isinstance(schema, dict):
        return None
    t = schema.get("type")
    kinds = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
    if isinstance(t, str) and t in kinds:
        ok = isinstance(value, kinds[t]) and not (t in ("integer", "number") and isinstance(value, bool))
        if not ok:
            return "%s must be of type %s" % (path, t)
    if "enum" in schema and isinstance(schema["enum"], list) and value not in schema["enum"]:
        return "%s must be one of %s" % (path, dumps(schema["enum"], 300))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            return "%s must be at least %s" % (path, schema["minimum"])
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            return "%s must be at most %s" % (path, schema["maximum"])
    if isinstance(value, str):
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            return "%s must have at most %d characters" % (path, schema["maxLength"])
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            return "%s must have at least %d characters" % (path, schema["minLength"])
    if isinstance(value, list):
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            return "%s must have at most %d items" % (path, schema["maxItems"])
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            return "%s must have at least %d items" % (path, schema["minItems"])
        for i, x in enumerate(value):
            p = schema_problem(schema.get("items"), x, "%s[%d]" % (path, i))
            if p:
                return p
    if isinstance(value, dict) and (t == "object" or "properties" in schema):
        props = schema.get("properties") or {}
        for r in schema.get("required") or []:
            if r not in value:
                return "%s.%s is required" % (path, r)
        if schema.get("additionalProperties") is False:
            extra = [k for k in value if k not in props]
            if extra:
                return "%s has unknown fields %s" % (path, dumps(extra, 200))
        for k, v in value.items():
            p = schema_problem(props.get(k), v, "%s.%s" % (path, k))
            if p:
                return p
    return None


class Tools:
    def __init__(self, raw: Any) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        for t in raw or []:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            params = fn.get("parameters") or fn.get("input_schema") or fn.get("inputSchema") or {"type": "object", "properties": {}}
            self.items[name] = {"name": name, "description": str(fn.get("description") or ""), "parameters": params}

    def text(self) -> str:
        out = []
        for t in self.items.values():
            props = (t["parameters"] or {}).get("properties") or {}
            req = set((t["parameters"] or {}).get("required") or [])
            args = []
            for k, s in props.items():
                s = s if isinstance(s, dict) else {}
                bits = [str(s.get("type") or "any")]
                for key in ("enum", "minimum", "maximum", "maxItems", "minItems", "maxLength"):
                    if key in s:
                        bits.append("%s=%s" % (key, dumps(s[key], 600)))
                if isinstance(s.get("items"), dict) and s["items"].get("type"):
                    it = s["items"]
                    inner = it.get("properties") if isinstance(it.get("properties"), dict) else {}
                    if inner:
                        ireq = set(it.get("required") or [])
                        fields = []
                        for ik, iv in inner.items():
                            iv = iv if isinstance(iv, dict) else {}
                            extra = [str(iv.get("type") or "any")]
                            for key in ("enum", "minimum", "maximum", "maxLength"):
                                if key in iv:
                                    extra.append("%s=%s" % (key, dumps(iv[key], 200)))
                            fields.append("%s%s: %s" % (ik, "" if ik in ireq else "?", " ".join(extra)))
                        bits.append("items=object with fields {%s}" % ", ".join(fields))
                    else:
                        bits.append("items=%s" % it["type"])
                if s.get("description"):
                    bits.append("about: %s" % clip(s["description"], 160))
                args.append("%s%s (%s)" % (k, "" if k in req else "?", ", ".join(bits)))
            out.append("- %s: %s\n    args: %s" % (t["name"], t["description"], "; ".join(args) or "none"))
        return "\n".join(out)

    def fit_lengths(self, name: str, args: Any) -> Any:
        t = self.items.get(name)
        if t is None or not isinstance(args, dict):
            return args
        props = (t["parameters"] or {}).get("properties") or {}
        out = dict(args)
        for k, v in args.items():
            s = props.get(k) if isinstance(props.get(k), dict) else {}
            if isinstance(v, str) and isinstance(s.get("maxLength"), int) and len(v) > s["maxLength"]:
                out[k] = v[: max(0, s["maxLength"] - 3)].rstrip() + "..." if s["maxLength"] > 3 else v[: s["maxLength"]]
        return out

    def check(self, name: str, args: Any) -> str | None:
        t = self.items.get(name)
        if t is None:
            return "there is no store tool named %s" % name
        if not isinstance(args, dict):
            return "args must be an object"
        return schema_problem(t["parameters"], args)


def strings_in(x: Any) -> set[str]:
    out: set[str] = set()
    stack = [x]
    while stack:
        v = stack.pop()
        if isinstance(v, str):
            out.add(v.strip())
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out.add(str(v))
        elif isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
    return out


def price_num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"\d[\d,]*(?:\.\d+)?", v)
        if m:
            try:
                return float(m.group(0).replace(",", ""))
            except ValueError:
                return None
    return None


def fmt_price(p: float) -> str:
    return str(int(p)) if p == int(p) else ("%.2f" % p)


def ident_of(d: dict[str, Any]) -> str | None:
    named = [k for k in d if isinstance(k, str) and re.search(r"(^|_|\b)(id|code|ref|sku)$|^id|id$", k.lower())]
    for k in named + [k for k in d if k not in named]:
        v = d.get(k)
        if isinstance(v, str) and 3 <= len(v) <= 64 and " " not in v and re.search(r"\d", v) and re.search(r"[A-Za-z]", v):
            if k not in named and any(isinstance(u, list) and any(isinstance(y, dict) and ident_of(y) for y in u) for u in d.values()):
                return None
            return v
    return None


class Memory:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.entries: list[dict[str, Any]] = []
        self.notes = ""
        self.ending: dict[str, Any] = {}
        self.fallback: dict[str, Any] = {}
        self.calls_seen: dict[str, int] = {}
        self.last_sig = ""
        self.repeats = 0
        self.texts: list[str] = []
        self.sources: dict[str, set[str]] = {}
        self.candidates: dict[str, list[str]] = {}
        self.references: dict[str, list[dict[str, Any]]] = {}
        self.verdicts: dict[str, dict[str, list[str]]] = {}
        self.rich: set[str] = set()
        self.view_args: dict[str, str] = {}
        self.basket_ids: list[str] = []
        self.by_id: dict[str, set[str]] = {}
        self.msgplan = ""
        self.searches: list[dict[str, Any]] = []
        self.price_field = ""
        self.message_tool = ""
        self.allowed: int | None = None
        self.sent_hint = 0
        self.sent_texts: list[str] = []
        self.answers: list[tuple[str, str]] = []
        self.chosen: list[tuple[str, str]] = []
        self.plan_items: list[tuple[str, str]] = []
        self.chosen_key: Any = None
        self.asked_hint: list[str] = []
        self.hints: dict[str, str] = {}
        self.slot_names: dict[str, list[list[str]]] = {}
        self.thin_tables: dict[str, int] = {}
        self.blocks: list[str] = []
        self.rendered = 0
        self.dropped = 0
        self.lowest_price: list[str] = []
        self.limits: dict[str, Any] = {"order_total": None, "per_unit": {}}
        self.basket_lines: list[tuple[str, dict[str, Any]]] = []
        self.basket_fields: dict[str, Any] = {}
        self.step = 0
        self.opened_at: dict[tuple[str, str], int] = {}
        self.basket_change_step = 0
        self.written: set[str] = set()
        self.wants: dict[str, list[str]] = {}
        self.units: dict[str, int] = {}
        self.writes: list[dict[str, Any]] = []
        self.blobs: dict[str, tuple[int, str]] = {}
        self.text_views: set[str] = set()

    def collect_texts(self, obs: Any, tool: str, step: int) -> None:
        found: list[str] = []

        def walk(x: Any, in_record: bool) -> None:
            if isinstance(x, str):
                t = re.sub(r"\s+", " ", x).strip()
                if not in_record and len(t) >= 12 and " " in t:
                    found.append(t)
            elif isinstance(x, list):
                for v in x:
                    walk(v, in_record)
            elif isinstance(x, dict):
                rec = in_record or (ident_of(x) is not None and len(x) >= 3)
                for v in x.values():
                    walk(v, rec)

        walk(obs, False)
        for t in found:
            line = "[step %d] %s: %s" % (step, tool, clip(t, 400))
            if not any(line.split(": ", 1)[-1] == old.split(": ", 1)[-1] and old.startswith("[step %d]" % step) for old in self.texts):
                self.texts.append(line)
        self.texts = self.texts[-25:]

    def absorb(self, obs: Any, tool: str = "", asked: set[str] | None = None) -> None:
        stack = [(obs, "own result")]
        while stack:
            x, how = stack.pop()
            if isinstance(x, list):
                stack.extend((v, "in a list") for v in x)
            elif isinstance(x, dict):
                rid = ident_of(x)
                if rid and len(x) >= 3:
                    self.sources.setdefault(rid, set()).add("%s (%s)" % (tool or "?", how))
                    if asked and rid in asked and how != "in a list":
                        self.by_id.setdefault(tool or "?", set()).add(rid)
                        self.opened_at[(tool or "?", rid)] = self.step
                        if any((isinstance(v, (dict, list)) and v) or (isinstance(v, str) and len(v) > 120) for v in x.values()):
                            self.rich.add(tool or "?")
                        if any(isinstance(v, str) and len(v.strip()) >= 80 for v in x.values()):
                            self.text_views.add(tool or "?")
                    cur = self.records.get(rid)
                    if cur is None:
                        self.records[rid] = dict(x)
                    else:
                        for k, v in x.items():
                            if v in (None, "", [], {}) and cur.get(k) not in (None, "", [], {}):
                                continue
                            cur[k] = v
                    if rid in self.order:
                        self.order.remove(rid)
                    self.order.append(rid)
                for v in x.values():
                    if isinstance(v, list):
                        stack.append((v, "in a list"))
                    elif isinstance(v, dict) and any(isinstance(y, (dict, list)) for y in v.values()):
                        stack.append((v, "inside a result"))

    def single_views(self) -> list[str]:
        return sorted(self.by_id)

    def views_of(self, rid: str) -> tuple[list[str], list[str]]:
        opened = [t for t in self.single_views() if rid in self.by_id[t]]
        return opened, [t for t in self.single_views() if rid not in self.by_id[t]]

    def coverage_line(self, rid: str, basket: bool = False) -> str:
        if rid not in self.records:
            return "%s: not seen in any result" % rid
        opened, missing = self.views_of(rid)
        lists = sorted({s.split(" (")[0] for s in (self.sources.get(rid) or set())})
        head = ("opened through " + ", ".join(opened)) if opened else ("seen only in the results of " + ", ".join(lists))
        again = ""
        if basket and self.basket_change_step:
            since = [t for t in opened if self.opened_at.get((t, rid), 0) > self.basket_change_step]
            again = "; re-opened since the last basket change (step %d): %s" % (self.basket_change_step, ("yes, through " + ", ".join(since)) if since else "no")
        return "%s: %s%s%s" % (rid, head, ("; not yet opened through " + ", ".join(missing)) if missing else "", again)

    def long_texts(self, rid: str) -> list[tuple[str, str]]:
        out = []
        for k, v in (self.records.get(rid) or {}).items():
            if isinstance(v, str) and len(v.strip()) >= 80:
                out.append((str(k), clip(re.sub(r"\s+", " ", v.strip()), CHAIN_TEXT_CHARS)))
        return out

    def verdict_key(self, rid: str, desc: list[str]) -> str:
        return rid + "|" + "|".join(desc) + "|%d" % len(dumps(self.records.get(rid) or {}))

    def hop_state(self, hop: dict[str, Any]) -> tuple[str, str]:
        rid = hop.get("id")
        desc = [d for d in (hop.get("described_as") or []) if d]
        shown = ("described as: " + "; ".join(desc)) if desc else "described as: (not recorded: give the words that describe the listing to read)"
        if not rid:
            return "not_found", "(not found yet) | " + shown
        if rid not in self.records:
            return "unknown", "%s: not seen in any result | %s" % (rid, shown)
        opened, missing = self.views_of(rid)
        cover = ("opened through " + ", ".join(opened)) if opened else "seen only in a list row, not opened through any single-listing view"
        if missing:
            cover += "; not yet opened through " + ", ".join(missing)
        if not desc:
            return "unopened" if (missing or not opened) else "open", "%s | %s | %s" % (rid, shown, cover)
        if not opened:
            return "unopened", "%s | %s | its data shows: nothing checked yet (open it first) | %s" % (rid, shown, cover)
        verdict = self.verdicts.get(self.verdict_key(rid, desc))
        if verdict is None:
            return "pending", "%s | %s | its data shows: not checked yet | %s" % (rid, shown, cover)
        parts = []
        bad = []
        unsaid = []
        for d in desc:
            ans, quote = (verdict.get(d) or ["unsaid", ""])[:2]
            tag = {"met": "MET", "not_met": "NOT MET"}.get(ans, "UNSAID")
            parts.append("%s %s%s" % (d, tag, (' ("%s")' % clip(quote, 70)) if quote and ans in ("met", "not_met") else ""))
            if ans == "not_met":
                bad.append(d)
            elif ans != "met":
                unsaid.append(d)
        if bad:
            state, mark = "not_met", "=> NOT the listing described (%s): another listing of that seller; a value or pointer read from it is wrong" % ", ".join(bad)
        elif unsaid:
            state, mark = ("unopened" if missing else "unsaid"), "=> not identified yet: its data does not state %s%s" % (", ".join(unsaid), " (open the views not yet opened)" if missing else "")
        else:
            state, mark = ("unopened" if missing else "identified"), "=> identified" + ("; open the views not yet opened before you use its value or pointer" if missing else "")
        return state, "%s | %s | its data shows: %s %s | %s" % (rid, shown, "; ".join(parts), mark, cover)

    def reference_text(self) -> str:
        out = []
        for key, hops in self.references.items():
            if not hops:
                out.append("%s: no listing of its chain found or described yet" % key)
                continue
            lines = []
            last = None
            for k, hop in enumerate(hops):
                state, line = self.hop_state(hop)
                lines.append("%d. %s" % (k + 1, line))
                if state != "identified":
                    match = self.described_matches(hop)
                    if match:
                        lines.append("   " + clip(match, 700))
                if hop.get("id") in self.records:
                    last = hop["id"]
            if last:
                texts = self.long_texts(last)
                if texts:
                    lines.append("texts on the opened views of %s (does it state the value, or send you to another listing?): " % last + " | ".join('%s: "%s"' % (k, v) for k, v in texts))
            out.append("%s:\n  " % key + "\n  ".join(lines))
        return "\n".join(out)

    def chain_gaps(self) -> list[dict[str, Any]]:
        gaps = []
        for key, hops in self.references.items():
            if not hops:
                gaps.append({"kind": "not_found", "key": key, "id": None, "desc": []})
                continue
            for hop in hops:
                state, _ = self.hop_state(hop)
                rid = hop.get("id")
                desc = [d for d in (hop.get("described_as") or []) if d]
                if state == "not_found":
                    gaps.append({"kind": "not_found", "key": key, "id": None, "desc": desc})
                elif state in ("not_met", "unsaid"):
                    verdict = self.verdicts.get(self.verdict_key(rid, desc)) or {}
                    gaps.append({"kind": state, "key": key, "id": rid, "desc": [d for d in desc if (verdict.get(d) or [""])[0] == ("not_met" if state == "not_met" else "unsaid") or (state == "unsaid" and d not in verdict)]})
                elif state == "unopened" and rid in self.records:
                    opened, missing = self.views_of(rid)
                    views = [v for v in missing if v in self.rich] if opened else sorted(self.rich)
                    if views:
                        gaps.append({"kind": "unopened", "key": key, "id": rid, "desc": views})
        return gaps

    def answer_text(self) -> str:
        if not self.answers or not self.basket_ids:
            return ""
        tables = {i: {plain(k): v for k, v in self.tables_of(i).items()} for i in self.basket_ids}
        names = {plain(k) for rid in self.order for k in self.tables_of(rid)}
        kinds = sorted({self.kind_of(i) for i in self.basket_ids})
        out = []
        for q, a in self.answers[-8:]:
            pq = plain(q)
            hit = [nm for nm in names if len(nm) >= 3 and re.search(r"(?<![a-z0-9])" + re.escape(nm) + r"(?![a-z0-9])", pq)]
            hit = [nm for nm in hit if not any(nm != o and nm in o for o in hit)]
            ks = self.kinds_named(q, kinds) if len(kinds) > 1 else set(kinds)
            ids = [i for i in self.basket_ids if self.kind_of(i) in ks]
            if len(ks) > 1 or len(kinds) == 1:
                ids = [i for i in ids if any(nm in tables[i] for nm in hit[:2])] or ids
            vals = []
            for i in ids:
                if not tables[i]:
                    vals.append("%s: its property data not read" % i)
                    continue
                for nm in hit[:2]:
                    vals.append("%s %s: %s" % (i, nm, tables[i][nm]) if nm in tables[i] else "%s: no %s in its data" % (i, nm))
            out.append('"%s" -> "%s"%s' % (clip(q, 160), clip(a or "(no reply text)", 260), (" | basket: " + "; ".join(vals[:6])) if vals else (" | the property it names is in no table read" if not hit else "")))
        return "\n".join(out)

    def basket_coverage(self) -> str:
        out = [self.coverage_line(i, True) for i in self.basket_ids]
        kinds: dict[str, list[str]] = {}
        for i in self.basket_ids:
            kinds.setdefault(self.kind_of(i), []).append(i)
        shared = ["%s: %s" % (k, ", ".join(ids)) for k, ids in kinds.items() if len(ids) > 1 and k != "all"]
        if shared:
            out.append("lines whose listings share one store category (one product named by that category is one line): " + "; ".join(shared))
        return "\n".join(out)

    def category_lines(self, request: str) -> str:
        gf = self.group_field()
        if not gf:
            return ""
        req = plain(request)
        counts: dict[str, int] = {}
        for rid in self.order:
            v = self.records[rid].get(gf)
            if isinstance(v, str) and v:
                counts[v] = counts.get(v, 0) + 1
        out = []
        for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            words = [w for w in plain(cat).split() if len(w) >= 3]
            if len(words) >= 2 and all(re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", req) for w in words):
                out.append("%s (%d listings seen under it)" % (cat, n))
        return "\n".join(out[:6])

    def blob_of(self, rid: str) -> str:
        r = self.records.get(rid) or {}
        size = len(r)
        cached = self.blobs.get(rid)
        if cached and cached[0] == size:
            return cached[1]
        parts = []
        for v in r.values():
            vals = list(v.values()) if isinstance(v, dict) else [v]
            for x in vals:
                if isinstance(x, str) and x != rid and len(x) <= 120:
                    parts.append(plain(x))
        blob = " | ".join(parts)
        self.blobs[rid] = (size, blob)
        return blob

    def described_matches(self, hop: dict[str, Any]) -> str:
        desc = [d for d in (hop.get("described_as") or []) if d]
        if not desc:
            return ""
        scores: dict[str, list[str]] = {}
        for d in desc:
            words = plain(d).split()
            hit_ids: list[str] = []
            value = ""
            for k in range(1, len(words)):
                value = " ".join(words[k:])
                pat = re.compile(r"(?<![a-z0-9])" + re.escape(value) + r"(?![a-z0-9])")
                hit_ids = [rid for rid in self.order[-400:] if pat.search(self.blob_of(rid))]
                if hit_ids:
                    break
            for rid in hit_ids:
                scores.setdefault(rid, []).append(value)
        if not scores:
            return ""
        rid0 = hop.get("id")
        ranked = sorted(scores, key=lambda i: (-len(scores[i]), i != rid0, not self.views_of(i)[0]))
        parts = []
        for i in ranked[:8]:
            opened, missing = self.views_of(i)
            tag = ("opened: " + ", ".join(opened)) if opened else "not opened through any single-listing view"
            if missing and opened:
                tag += "; not opened: " + ", ".join(missing)
            parts.append("%s (carries %d of %d: %s; %s%s)" % (i, len(scores[i]), len(desc), ", ".join(scores[i]), "recorded above; " if i == rid0 else "", tag))
        more = (" (%d more)" % (len(ranked) - 8)) if len(ranked) > 8 else ""
        return "listings seen whose data carries any of the describing words (the one meant carries every one of them; open those not yet opened through the page view): %s%s" % (", ".join(parts), more)

    def writes_text(self) -> str:
        out = []
        for w in self.writes[-6:]:
            after = w.get("after")
            shown = "; ".join(("%s with %s" % (rid, ", ".join("%s=%s" % (k, v) for k, v in list(nums.items())[:4]) or "no figures")) if nums is not None else ("%s not in the basket" % rid) for rid, nums in (after or {}).items())
            out.append("step %d %s %s -> %s" % (w["step"], w["tool"], dumps(w["args"], 120), ("the store then showed " + shown) if shown else "no figures for it shown yet"))
        return "\n".join(out)

    def note_write_reply(self, obs: Any, step: int) -> None:
        if not self.writes or self.writes[-1]["step"] != step:
            return
        w = self.writes[-1]
        found: dict[str, dict[str, Any]] = {}

        def walk(x: Any) -> None:
            if isinstance(x, dict):
                rid = ident_of(x)
                if rid and rid in w["ids"] and rid not in found:
                    found[rid] = {str(k): v for k, v in x.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        walk(obs)
        if found:
            w["after"] = found

    def units_gaps(self) -> list[str]:
        if not self.units or not self.basket_lines:
            return []
        asked = {plain(k): v for k, v in self.units.items()}
        out = []
        for product, ids in self.candidates.items():
            want = asked.get(plain(product))
            if not want:
                continue
            held = [(rid, nums) for rid, nums in self.basket_lines if rid in ids]
            if len(held) != 1:
                continue
            rid, nums = held[0]
            counts = {k: int(v) for k, v in nums.items() if float(v).is_integer() and 0 < v < 100 and k != self.price_field}
            if not counts or want in counts.values():
                continue
            out.append("%s, bought as %s, shows %s on its line, while you recorded that the request asks for %d unit(s) of it" % (rid, product, ", ".join("%s %d" % (k, v) for k, v in counts.items()), want))
        return out

    def note_basket(self, obs: Any) -> None:
        found: list[str] = []
        lines: list[tuple[str, dict[str, Any]]] = []

        def walk(x: Any) -> None:
            if isinstance(x, dict):
                rid = ident_of(x)
                if rid and rid in self.records and rid not in found:
                    found.append(rid)
                    lines.append((rid, {str(k): v for k, v in x.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}))
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        walk(obs)
        self.basket_ids = found
        self.basket_lines = lines
        holder, _ = self.find_rows(obs)
        self.basket_fields = {str(k): v for k, v in (holder or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        self.written = set()
        for w in self.writes[-6:]:
            if w.get("after") is None:
                held = {rid: nums for rid, nums in lines if rid in w["ids"]}
                w["after"] = {rid: held.get(rid) for rid in w["ids"]}

    def note_write(self, tool: str, args: Any, step: int) -> None:
        ids = [i for i in strings_in(args) if i in self.records]
        if not ids or tool == self.message_tool or (self.searches and self.searches[-1]["step"] == step):
            return
        if any(i in self.by_id.get(tool, set()) for i in ids):
            return
        self.basket_change_step = step
        self.written.update(ids)
        self.writes.append({"step": step, "tool": tool, "args": args, "ids": ids, "after": None})
        self.writes = self.writes[-12:]

    def find_rows(self, obs: Any) -> tuple[dict[str, Any] | None, list[Any]]:
        stack = [(obs, 0)]
        while stack:
            x, depth = stack.pop()
            if isinstance(x, list):
                recs = [y for y in x if isinstance(y, dict) and ident_of(y) is not None and len(y) >= 3]
                if recs:
                    return {}, recs
            elif isinstance(x, dict) and depth < 3:
                for v in x.values():
                    if isinstance(v, list):
                        recs = [y for y in v if isinstance(y, dict) and ident_of(y) is not None and len(y) >= 3]
                        if recs:
                            return x, recs
                for v in x.values():
                    if isinstance(v, dict):
                        stack.append((v, depth + 1))
        return None, []

    def empty_holder(self, obs: Any) -> dict[str, Any] | None:
        best = None
        stack = [(obs, 0)]
        while stack:
            x, depth = stack.pop()
            if not isinstance(x, dict) or depth >= 3:
                continue
            if any(isinstance(v, list) and not v for v in x.values()):
                scalars = sum(1 for v in x.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool))
                if best is None or scalars > best[0]:
                    best = (scalars, x)
            stack.extend((v, depth + 1) for v in x.values() if isinstance(v, dict))
        return best[1] if best else None

    def note_search(self, tool: str, args: Any, obs: Any, step: int) -> None:
        args = args if isinstance(args, dict) else {}
        if not args or tool == self.ending.get("basket_view"):
            return
        vals: list[Any] = []
        for v in args.values():
            vals.extend(v if isinstance(v, list) else [v])
        if any(isinstance(v, str) and v in self.records for v in vals):
            return
        holder, rows = self.find_rows(obs)
        if holder is None:
            holder = self.empty_holder(obs)
            if holder is None and not any(s["tool"] == tool and s["rows"] for s in self.searches):
                return
        fields = {str(k): v for k, v in (holder or {}).items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
        prev = None
        via = ""
        for s in self.searches:
            if s["tool"] != tool or s["followed"]:
                continue
            for k, v in args.items():
                if isinstance(v, list):
                    continue
                hit = [fk for fk, fv in s["fields"].items() if fv not in ("", 0) and (v == fv or (isinstance(v, str) and v == str(fv)))]
                if not hit:
                    continue
                rest = {kk: vv for kk, vv in args.items() if kk != k}
                base = {kk: vv for kk, vv in s["args"].items() if kk != k}
                if rest == base:
                    s["followed"] = True
                    prev = s
                    via = hit[0]
        echo = strings_in(args)
        ended = prev is not None and fields.get(via) in (None, "", 0)
        if prev is None:
            more = any(isinstance(v, str) and v.strip() and v.strip() not in echo for v in fields.values()) or any(isinstance(v, (int, float)) and v > len(rows) for v in fields.values())
        else:
            more = not ended and any((k not in prev["fields"] or prev["fields"][k] != v) and (not isinstance(v, str) or (v.strip() and v.strip() not in echo)) and not (isinstance(v, (int, float)) and k not in prev["fields"]) for k, v in fields.items())
        ids = [i for i in (ident_of(r) for r in rows) if i]
        same = next((s["step"] for s in self.searches if s["tool"] == tool and ids and s["ids"] == ids), None)
        reply = "" if rows or holder is not None else clip(re.sub(r"\s+", " ", dumps(shrink(obs, 120, 6))), 140)
        self.searches.append({"step": step, "tool": tool, "args": args, "rows": len(rows), "fields": fields, "followed": False, "more": bool(more and rows), "ids": ids, "parent": prev["step"] if prev else None, "root": prev["root"] if prev else step, "ended": ended, "same_as": same, "reply": reply})

    def ranked_text(self) -> str:
        out = []
        for x in self.searches[-8:]:
            ids = [i for i in x["ids"] if i]
            if len(ids) < 2:
                continue
            lead = [ids[0]]
            for i in ids[1:]:
                if i > lead[-1]:
                    lead.append(i)
                else:
                    break
            if len(lead) == len(ids):
                lead = ids[:3]
            todo = [i for i in lead[:5] if i in self.records and not self.views_of(i)[0]]
            if not todo:
                continue
            names = []
            for i in todo:
                r = self.records.get(i) or {}
                texts = [v for k, v in r.items() if isinstance(v, str) and v != i and " " in v and len(v) <= 160]
                names.append("%s (%s)" % (i, clip(max(texts, key=len), 70) if texts else "no title"))
            out.append("step %s %s: %s" % (x["step"], dumps(x["args"], 70), ", ".join(names)))
        return "\n".join(out)

    def order_text(self) -> str:
        long_lists = [x for x in self.searches if x["rows"] >= 4 and self.count_of(x) > x["rows"] and len(x["ids"]) >= 4]
        if len(long_lists) < 3:
            return ""
        rising = [x for x in long_lists if x["ids"] == sorted(x["ids"])]
        if len(rising) < 0.8 * len(long_lists):
            return ""
        return "Every longer list this store returned (%d of %d) came in rising listing-id order, not by price or by how well a listing fits. A search phrase shows first the listings that carry every one of its words (then those carrying fewer), each group in id order. So a short phrase of the exact value words (the kind's own word plus each wanted value as listings write it) brings the listings holding them all to the top, while a long or loose phrase shows the same low ids as any other. Page such a search until a row of another kind, or an id lower than the one before it, appears: that ends the group holding every word. To see the cheapest listings of a kind, list its category (its exact name, copied from a row's category field) with a price ceiling low enough that the count found falls within the pages the store can show, and page it to its end; a category list without such a ceiling shows only the lowest ids." % (len(rising), len(long_lists))

    def count_of(self, s: dict[str, Any]) -> int:
        nums = [int(v) for v in s["fields"].values() if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= s["rows"] and v == int(v)]
        return max(nums) if nums else 0

    def search_text(self) -> str:
        roots = [s for s in self.searches if s["parent"] is None]
        counts = [self.count_of(s) for s in self.searches if s["rows"]]
        top = max(counts) if counts else 0
        ceiling = top if top and counts.count(top) >= 3 else 0
        out = []
        for s in roots[-SEARCH_LINES:]:
            pages = [s] + [p for p in self.searches if p["root"] == s["step"] and p["step"] != s["step"]]
            seen = sum(p["rows"] for p in pages)
            count = max(self.count_of(p) for p in pages)
            line = "step %s %s %s -> %d rows" % (s["step"], s["tool"], dumps(s["args"], 80), s["rows"])
            if not s["rows"]:
                line += " (store reply: %s)" % (s["reply"] or "; ".join("%s=%s" % (k, clip(str(v), 24)) for k, v in s["fields"].items()) or "nothing")
                out.append(line)
                continue
            if count > s["rows"]:
                line += " of %d found" % count
            if ceiling and count == ceiling:
                line += " (= the store's ceiling)"
            if len(pages) > 1:
                line += "; continued at step%s %s, %d rows seen in all" % ("s" if len(pages) > 2 else "", ", ".join(str(p["step"]) for p in pages[1:]), seen)
            if any(p["ended"] for p in pages):
                line += "; the store offers no further page" + ((" although %d rows remain" % (count - seen)) if count > seen else "")
            elif count > seen:
                line += "; %d rows not shown | continued: %s" % (count - seen, "yes, not to the end" if len(pages) > 1 else "NO")
            elif pages[-1]["more"]:
                line += "; a further page is offered | continued: %s" % ("yes, not to the end" if len(pages) > 1 else "NO")
            if s["same_as"] is not None:
                line += "; same rows as step %s" % s["same_as"]
            prices = [price_num((self.records.get(i) or {}).get(self.price_field)) for i in s["ids"]] if self.price_field else []
            prices = [p for p in prices if p is not None]
            if len(prices) >= 3 and all(prices[k] <= prices[k + 1] for k in range(len(prices) - 1)):
                line += "; rows rise in price"
            out.append(line)
        if len(roots) > SEARCH_LINES:
            out.insert(0, "(%d earlier searches not shown)" % (len(roots) - SEARCH_LINES))
        return "\n".join(out)

    def kind_of(self, rid: str) -> str:
        gf = self.group_field()
        r = self.records.get(rid) or {}
        return str(r.get(gf)) if gf and r.get(gf) else "all"

    def product_of(self, rid: str) -> str | None:
        for product, ids in self.candidates.items():
            if rid in ids:
                return product
        return None

    def limit_for(self, product: str | None) -> float | None:
        per = self.limits.get("per_unit") or {}
        p = plain(product)
        if not p:
            return None
        for k, v in per.items():
            kk = plain(k)
            if kk and (kk == p or kk in p or p in kk):
                return v
        return None

    def lowest_kinds(self) -> set[str]:
        kinds: set[str] = set()
        wanted = [plain(x) for x in self.lowest_price if plain(x)]
        for product, ids in self.candidates.items():
            p = plain(product)
            if any(w == p or w in p or p in w for w in wanted):
                kinds.update(self.kind_of(i) for i in ids if i in self.records)
        kinds.update(self.kind_of(i) for i in self.basket_ids)
        return kinds

    def pending_lists(self, kinds: set[str]) -> list[dict[str, Any]]:
        out = []
        for s in self.searches:
            if s["followed"] or not s.get("more"):
                continue
            ids = [i for i in s.get("ids") or [] if i in self.records]
            if kinds and not any(self.kind_of(i) in kinds for i in ids):
                continue
            prices = [price_num(self.records[i].get(self.price_field)) for i in ids] if self.price_field else []
            prices = [p for p in prices if p is not None]
            if len(prices) >= 3 and all(prices[k] <= prices[k + 1] for k in range(len(prices) - 1)):
                continue
            out.append(s)
        return out

    def basket_price_rows(self) -> tuple[list[str], float | None, list[str]]:
        lines: list[str] = []
        breaches: list[str] = []
        total = 0.0
        exact = bool(self.basket_lines)
        for rid, nums in self.basket_lines:
            price = price_num((self.records.get(rid) or {}).get(self.price_field)) if self.price_field else None
            product = self.product_of(rid)
            lim = self.limit_for(product)
            if price is None:
                exact = False
                lines.append("%s (%s): no price seen for it under the price field you named" % (rid, product or self.kind_of(rid)))
                continue
            qs = [v for v in nums.values() if v != price and float(v) == int(v) and 1 <= v <= 99]
            qty = int(qs[0]) if len(qs) == 1 else None
            if qty is None:
                exact = False
            total += price * (qty or 1)
            part = "%s (%s): unit price %s, units %s" % (rid, product or self.kind_of(rid), fmt_price(price), str(qty) if qty is not None else "not read from the basket view (1 assumed)")
            if lim is not None:
                part += ", per-unit limit %s: %s" % (fmt_price(lim), "OVER THE LIMIT" if price > lim else "within it")
                if price > lim:
                    breaches.append("%s costs %s a unit, above the %s per-unit limit you recorded for %s" % (rid, fmt_price(price), fmt_price(lim), product))
            lines.append(part)
        cap = self.limits.get("order_total")
        if exact and isinstance(cap, (int, float)) and total > cap:
            breaches.append("the basket's lines add up to %s, above the %s order total limit you recorded" % (fmt_price(total), fmt_price(cap)))
        return lines, (total if exact else None), breaches

    def price_rule_text(self) -> str:
        out = []
        if self.lowest_price:
            out.append("lowest price wanted for: " + ", ".join(self.lowest_price))
            pend = self.pending_lists(self.lowest_kinds())
            if pend:
                out.append("searches that showed these kinds, announced more rows than they showed and were never continued (a list in the store's own order can hold a cheaper fit on any page; only a list whose rows rise in price is complete at its first page):")
                for s in pend[-8:]:
                    f = "; ".join("%s=%s" % (k, clip(str(v), 24)) for k, v in s["fields"].items())
                    out.append("  step %s %s %s -> %d rows%s" % (s["step"], s["tool"], dumps(s["args"], 80), s["rows"], ("; " + f) if f else ""))
            elif self.searches:
                out.append("every search that showed these kinds has been continued to its end, or its rows rise in price")
        lim = self.limits
        per = lim.get("per_unit") or {}
        if isinstance(lim.get("order_total"), (int, float)) or per:
            out.append("limits you recorded: order total %s; per unit: %s" % (fmt_price(lim["order_total"]) if isinstance(lim.get("order_total"), (int, float)) else "none", ", ".join("%s %s" % (k, fmt_price(v)) for k, v in per.items()) or "none"))
            if self.basket_lines:
                lines, total, breaches = self.basket_price_rows()
                out.append("basket as the store last showed it, against them:\n  " + "\n  ".join(lines))
                if total is not None:
                    cap = lim.get("order_total")
                    out.append("sum of the lines (unit price x units): %s%s" % (fmt_price(total), (" against the order total limit %s: %s" % (fmt_price(cap), "OVER THE LIMIT" if total > cap else "within it")) if isinstance(cap, (int, float)) else ""))
                if self.basket_fields:
                    out.append("figures the basket view itself reports: " + dumps(self.basket_fields, 240))
                for b in breaches:
                    out.append("=> " + b)
        return "\n".join(out)

    def is_read(self, rid: str) -> bool:
        if self.tables_of(rid):
            return True
        return any(not s.endswith("(in a list)") for s in self.sources.get(rid) or ())

    def cheapest_text(self) -> str:
        if not self.price_field:
            return ""
        gf = self.group_field()
        groups: dict[str, list[tuple[float, str]]] = {}
        for rid in self.order:
            r = self.records[rid]
            p = price_num(r.get(self.price_field))
            if p is None:
                continue
            g = str(r.get(gf)) if gf and r.get(gf) else "all"
            groups.setdefault(g, []).append((p, rid))
        out = []
        for g, lst in groups.items():
            lst.sort()
            parts = ["%s %s %s" % (rid, fmt_price(p), "read" if self.is_read(rid) else "UNREAD") for p, rid in lst[:6]]
            unread = sum(1 for p, rid in lst if not self.is_read(rid))
            out.append("%s (%d seen, %d of them unread): %s" % (g, len(lst), unread, ", ".join(parts)))
        return "\n".join(out)

    def listing_lines(self) -> str:
        out = []
        views = self.single_views()
        for rid in self.order[-LISTING_LINES:]:
            r = self.records[rid]
            short = []
            for k, v in r.items():
                if isinstance(v, (str, int, float, bool)) and v != rid and len(str(v)) <= 90:
                    short.append("%s=%s" % (k, v))
            deep = [k for k, v in r.items() if isinstance(v, (dict, list)) and v]
            opened = [v for v in views if rid in self.by_id[v]]
            missing = [v for v in views if rid not in self.by_id[v]]
            tag = ("opened: " + ", ".join(opened)) if opened else "opened: none (list rows only)"
            out.append("%s | %s%s | %s%s" % (rid, clip("; ".join(short), 200), (" | also has: " + ", ".join(deep)) if deep else "", tag, (" | missing: " + ", ".join(missing)) if missing and opened else ""))
        return "\n".join(out)

    def group_field(self) -> str | None:
        recs = [self.records[r] for r in self.order]
        if len(recs) < 4:
            return None
        best, best_score = None, 0.0
        keys = {k for r in recs for k, v in r.items() if isinstance(v, str)}
        for k in keys:
            vals = [r.get(k) for r in recs if isinstance(r.get(k), str) and r.get(k)]
            if len(vals) < max(3, len(recs) // 2):
                continue
            distinct = len(set(vals))
            if distinct < 1 or distinct > max(2, len(vals) // 3) or max(len(v) for v in vals) > 80:
                continue
            score = len(vals) / distinct
            if score > best_score:
                best, best_score = k, score
        return best

    def variety_text(self) -> str:
        gf = self.group_field()
        groups: dict[str, dict[str, dict[str, int]]] = {}
        sizes: dict[str, int] = {}
        for rid in self.order:
            r = self.records[rid]
            g = str(r.get(gf)) if gf and r.get(gf) else "all"
            has_table = False
            for v in r.values():
                if isinstance(v, dict) and v and all(not isinstance(x, (dict, list)) for x in v.values()):
                    has_table = True
                    for pk, pv in v.items():
                        sv = str(pv).strip()
                        if not sv or len(sv) > 60:
                            continue
                        groups.setdefault(g, {}).setdefault(str(pk), {})
                        groups[g][str(pk)][sv.lower()] = groups[g][str(pk)].get(sv.lower(), 0) + 1
            if has_table:
                sizes[g] = sizes.get(g, 0) + 1
        out = []
        for g, props in groups.items():
            total = max(1, sizes.get(g, 1))
            ranked = sorted(props.items(), key=lambda kv: (-sum(kv[1].values()), -min(len(kv[1]), 6)))
            parts = []
            for pk, vals in ranked[:18]:
                top = sorted(vals.items(), key=lambda kv: -kv[1])[:5]
                parts.append("%s (carried by %d of %d, %s): %s" % (pk, sum(vals.values()), total, "values differ" if len(vals) > 1 else "one value so far", ", ".join(v for v, _ in top)))
            out.append("%s (%d listings with a property table):\n  " % (g, total) + "\n  ".join(parts))
        return "\n".join(out)

    def tables_of(self, rid: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for v in (self.records.get(rid) or {}).values():
            if isinstance(v, dict) and v and all(not isinstance(x, (dict, list)) for x in v.values()):
                for pk, pv in v.items():
                    sv = str(pv).strip()
                    if sv and len(sv) <= 60:
                        out[str(pk)] = sv
        return out

    def candidate_text(self) -> str:
        out = []
        for product, ids in self.candidates.items():
            known = [i for i in ids if i in self.records]
            if len(known) < 2:
                continue
            tables = {i: self.tables_of(i) for i in known}
            read = [i for i in known if tables[i]]
            if len(read) < 2:
                out.append("%s: %d candidates, but fewer than two have their property data read yet" % (product, len(known)))
                continue
            keys: dict[str, dict[str, int]] = {}
            for i in read:
                for k, v in tables[i].items():
                    keys.setdefault(k, {})
                    keys[k][v.lower()] = keys[k].get(v.lower(), 0) + 1
            differ = []
            for k, vals in keys.items():
                shown = sum(vals.values())
                if len(vals) >= 2 or shown < len(read):
                    differ.append((len(vals), shown, k, vals))
            differ.sort(key=lambda x: (-min(x[0], 6), -x[1]))
            parts = ["%s (%s%s)" % (k, ", ".join("%s x%d" % (v, c) for v, c in sorted(vals.items(), key=lambda kv: -kv[1])[:5]), "" if shown == len(read) else ", not shown on %d" % (len(read) - shown)) for _, shown, k, vals in differ[:14]]
            out.append("%s (%d candidates with property data): %s" % (product, len(read), "; ".join(parts) if parts else "they do not differ on any property shown"))
        return "\n".join(out)

    def want_pattern(self, want: str) -> re.Pattern[str] | None:
        toks = [t for t in re.split(r"[\s_\-]+", want.lower().strip()) if t]
        if not toks:
            return None
        return re.compile(r"(?<![a-z0-9])" + r"[\s_\-]+".join(re.escape(t) for t in toks) + r"(?![a-z0-9])")

    def home_keys(self, want: str) -> set[str]:
        pat = self.want_pattern(want)
        if pat is None:
            return set()
        counts: dict[str, int] = {}
        for rid in self.order:
            for k, v in self.tables_of(rid).items():
                if pat.search(v.lower()):
                    counts[plain(k)] = counts.get(plain(k), 0) + 1
        top = max(counts.values()) if counts else 0
        return {k for k, c in counts.items() if c == top} if top >= 2 else set()

    def own_says(self, rid: str, want: str) -> tuple[str, str] | None:
        pat = self.want_pattern(want)
        keys = self.home_keys(want)
        if pat is None or not keys:
            return None
        table = {plain(k): v for k, v in self.tables_of(rid).items()}
        own = [k for k in sorted(keys) if k in table]
        if not own or any(pat.search(table[k].lower()) for k in own):
            return None
        return own[0], table[own[0]]

    def want_in(self, rid: str, want: str) -> tuple[str, str] | None:
        pat = self.want_pattern(want)
        rec = self.records.get(rid) or {}
        if pat is None or not rec or self.own_says(rid, want):
            return None
        found: list[tuple[int, str, str]] = []

        def look(field: str, v: Any) -> None:
            if isinstance(v, str):
                text = v.lower()
                if "+" in text and len(text) <= 60 and re.sub(r"[\s_\-]+", " ", text.strip()) != re.sub(r"[\s_\-]+", " ", want.lower().strip()):
                    return
                m = pat.search(text)
                if m:
                    a, b = max(0, m.start() - 20), min(len(v), m.end() + 20)
                    found.append((len(v), field, clip(re.sub(r"\s+", " ", v[a:b]).strip(), 60)))
            elif isinstance(v, dict):
                for k, x in v.items():
                    look("%s.%s" % (field, k), x)
            elif isinstance(v, list):
                for x in v:
                    look(field, x)

        for k, v in rec.items():
            if v == rid:
                continue
            look(str(k), v)
        if not found:
            return None
        found.sort()
        return found[0][1], found[0][2]

    def wants_text(self) -> str:
        out = []
        for product, wants in self.wants.items():
            wants = [w for w in wants if w.strip()][:8]
            if not wants:
                continue
            named = [i for i in self.candidates.get(product, []) if i in self.records]
            for rid in self.basket_ids:
                if rid not in named and self.product_of(rid) in (product, None) and len(self.candidates) <= 1:
                    named.append(rid)
            kinds = {self.kind_of(i) for i in named} or {self.kind_of(i) for i in self.basket_ids}
            fits = []
            for rid in self.order:
                if kinds and self.kind_of(rid) not in kinds:
                    continue
                hits = [self.want_in(rid, w) for w in wants]
                if all(hits):
                    fields = sorted({h[0] for h in hits if h})
                    fits.append((price_num(self.records[rid].get(self.price_field)) if self.price_field else None, rid, fields))
            fits.sort(key=lambda x: (x[0] is None, x[0] or 0))
            lines = ["%s (wanted: %s):" % (product, "; ".join(wants))]
            if fits:
                lines.append("  listings seen of that kind whose stored data holds every wanted value, cheapest first, with the fields holding them: " + ", ".join("%s%s %s (%s)" % (rid, (" " + fmt_price(p)) if p is not None else "", "read" if self.is_read(rid) else "UNREAD", "; ".join(f)) for p, rid, f in fits[:8]) + ("" if len(fits) <= 8 else " (+%d more)" % (len(fits) - 8)))
            else:
                lines.append("  no listing seen of that kind holds every wanted value in its stored data (unread rows show little: a search naming the value, then their pages, may)")
            for w in wants:
                holders = []
                for rid in self.order:
                    if (kinds and self.kind_of(rid) not in kinds) or not self.want_in(rid, w):
                        continue
                    holders.append((price_num(self.records[rid].get(self.price_field)) if self.price_field else None, rid))
                holders.sort(key=lambda x: (x[0] is None, x[0] or 0))
                lines.append("  %s: held by %d listing(s) seen of that kind%s" % (w, len(holders), (", cheapest: " + ", ".join("%s%s %s" % (rid, (" " + fmt_price(p)) if p is not None else "", "read" if self.is_read(rid) else "UNREAD") for p, rid in holders[:4])) if holders else ""))
            for rid in named[:6]:
                parts = []
                for w in wants:
                    hit = self.want_in(rid, w)
                    own = None if hit else self.own_says(rid, w)
                    parts.append('%s: %s' % (w, ('in %s ("%s")' % (hit[0], hit[1])) if hit else (('NOT ITS VALUE: its own %s says "%s", the property under which other listings seen hold "%s", so the word elsewhere in its data does not count' % (own[0], clip(own[1], 50), w)) if own else "NOT IN ITS STORED DATA")))
                lines.append("  %s (%s): " % (rid, "read" if self.is_read(rid) else "UNREAD") + "; ".join(parts))
            out.append("\n".join(lines))
        return "\n".join(out)

    def messages_sent(self) -> int:
        return len(self.sent_texts) if self.message_tool else self.sent_hint

    def messages_left(self) -> int | None:
        if self.allowed is None:
            return max(0, ASSUMED_MESSAGES - self.messages_sent())
        return max(0, self.allowed - self.messages_sent())

    def same_property(self, a: str, b: str) -> bool:
        la, lb = letters(a).replace(" ", ""), letters(b).replace(" ", "")
        if not la or not lb:
            return False
        return la == lb or difflib.SequenceMatcher(None, la, lb).ratio() >= 0.85

    def hinted(self, names: list[str], kind: str | None = None, kinds: list[str] | None = None) -> int:
        best = 0
        for hk, h in self.hints.items():
            if kind is not None and kinds and kind not in self.kinds_named(str(hk), kinds):
                continue
            best = max(best, self.hint_score(h, names))
        return best

    def hint_score(self, h: str, names: list[str]) -> int:
        best = 0
        lh = letters(h).replace(" ", "")
        if len(lh) < 3:
            return 0
        for name in names:
            ln = letters(name).replace(" ", "")
            if ln == lh:
                score = 40
            elif (len(ln) >= 4 and ln in lh) or (len(lh) >= 4 and lh in ln):
                score = 30 - min(9, abs(len(ln) - len(lh)))
            elif difflib.SequenceMatcher(None, ln, lh).ratio() >= 0.8:
                score = 10
            else:
                score = 0
            best = max(best, score)
        return best


    def name_words(self, text: str) -> set[str]:
        out: set[str] = set()
        for w in letters(text).split():
            if len(w) >= 4:
                out.add(w[:-1] if len(w) > 4 and w.endswith("s") else w)
        return out

    def named_for(self, pats: list[str], req: str, kind: str, kinds: list[str]) -> bool:
        if not any(re.search(p, req) for p in pats):
            return False
        if len(kinds) <= 1:
            return True
        for sent in re.split(r"(?<=[.!?;:])\s+", req):
            clauses = re.split(r"[,;]", sent)
            for i, c in enumerate(clauses):
                if not any(re.search(p, c) for p in pats):
                    continue
                owner = None
                for j in range(i, -1, -1):
                    tw = self.name_words(clauses[j])
                    sc = {g: len(tw & self.name_words(g)) for g in kinds}
                    top = max(sc.values()) if sc else 0
                    if top:
                        owner = {g for g, v in sc.items() if v == top}
                        break
                if owner is None or kind in owner:
                    return True
        return False

    def plan_swap(self, args: dict[str, Any]) -> dict[str, Any] | None:
        plan = [(g, pk) for g, pk in self.plan_items if g != "all"]
        if not plan:
            return None
        keys = [k for k, v in args.items() if isinstance(v, str) and v.strip()]
        if not keys:
            return None
        key = max(keys, key=lambda k: len(args[k]))
        if "?" not in args[key] or len(args[key]) > 240:
            return None
        kinds = list(dict.fromkeys(g for g, _ in plan))

        def says(text: str, g: str, pk: str) -> bool:
            t = plain(text)
            if not any(re.search(r"(?<![a-z0-9])" + re.escape(plain(n)) + r"(?![a-z0-9])", t) for n in pk.split(" / ") if n.strip()):
                return False
            return len(kinds) == 1 or g in self.kinds_named(text, kinds)

        if any(says(args[key], g, pk) for g, pk in plan):
            return None
        todo = [(g, pk) for g, pk in plan if not any(says(x, g, pk) for x in self.sent_texts)]
        if not todo:
            return None
        g, pk = todo[0]
        out = dict(args)
        out[key] = "What %s would you like for the %s?" % (pk.split(" / ")[0].strip(), g.lower())
        return out

    def kinds_named(self, text: str, kinds: list[str]) -> set[str]:
        tw = self.name_words(text)
        scores = {g: len(tw & self.name_words(g)) for g in kinds}
        top = max(scores.values()) if scores else 0
        if top == 0:
            return set(kinds)
        return {g for g, sc in scores.items() if sc == top}

    def ask_rows(self, request: str) -> tuple[dict[str, list[tuple[str, int, int, list[str], str, int, int, str]]], dict[str, int], dict[str, int]]:
        gf = self.group_field()
        kinds: dict[str, list[dict[str, Any]]] = {}
        sizes: dict[str, int] = {}
        bare: dict[str, int] = {}
        thin: dict[str, int] = {}
        cands = {i for ids in self.candidates.values() for i in ids if i in self.records}
        csizes: dict[str, int] = {}
        widest: dict[str, int] = {}
        for rid in self.order:
            r = self.records[rid]
            g = str(r.get(gf)) if gf and r.get(gf) else "all"
            widest[g] = max(widest.get(g, 0), len(self.tables_of(rid) or {}))
        for rid in self.order:
            r = self.records[rid]
            g = str(r.get(gf)) if gf and r.get(gf) else "all"
            tab = self.tables_of(rid)
            if not tab:
                bare[g] = bare.get(g, 0) + 1
                continue
            if len(tab) <= 3 and len(tab) * 3 < widest.get(g, 0):
                thin[g] = thin.get(g, 0) + 1
                continue
            sizes[g] = sizes.get(g, 0) + 1
            cand = rid in cands
            if cand:
                csizes[g] = csizes.get(g, 0) + 1
            groups = kinds.setdefault(g, [])
            done: list[dict[str, Any]] = []
            for pk, pv in tab.items():
                name = plain(pk)
                if not name:
                    continue
                slot = None
                for s in groups:
                    if name in s["names"] or any(self.same_property(name, o) for o in s["names"]):
                        slot = s
                        break
                if slot is None:
                    slot = {"names": {}, "vals": [], "n": 0, "nc": 0, "by": {}, "cands": set()}
                    groups.append(slot)
                slot["names"][name] = slot["names"].get(name, 0) + 1
                slot["vals"].append(pv.lower())
                slot["by"].setdefault(rid, re.sub(r"\s+", " ", pv.lower()).strip())
                if slot not in done:
                    done.append(slot)
                    slot["n"] += 1
                    if cand:
                        slot["nc"] += 1
                        slot["cands"].add(rid)
        for g, groups in kinds.items():
            joined = True
            while joined:
                joined = False
                for a in groups:
                    for b in groups:
                        if a is b:
                            continue
                        both = [r for r in a["by"] if r in b["by"]]
                        if not both or any(a["by"][r] != b["by"][r] for r in both):
                            continue
                        if len(both) < 2 and len(a["by"][both[0]]) < 8:
                            continue
                        for nm, c in b["names"].items():
                            a["names"][nm] = a["names"].get(nm, 0) + c
                        for r, v in b["by"].items():
                            if r not in a["by"]:
                                a["by"][r] = v
                                a["vals"].append(v)
                        a["cands"] |= b["cands"]
                        a["n"] = len(a["by"])
                        a["nc"] = len(a["cands"])
                        groups.remove(b)
                        joined = True
                        break
                    if joined:
                        break
        self.slot_names = {g: [list(slot["names"]) for slot in groups] for g, groups in kinds.items()}
        req = plain(request)
        all_kinds = list(kinds) + [g for g in bare if g not in kinds]
        said = [(plain(t), self.kinds_named(t, all_kinds)) for t in self.sent_texts + self.asked_hint]
        rows: dict[str, list[tuple[str, int, int, list[str], str, int, int, str]]] = {}
        for g, groups in kinds.items():
            ranked = []
            for slot in groups:
                names = sorted(slot["names"], key=lambda k: -slot["names"][k])
                pats = [r"(?<![a-z0-9])" + re.escape(nm) + r"(?![a-z0-9])" for nm in names]
                if any(g in gs and re.search(p, t) for t, gs in said for p in pats):
                    continue
                vals = slot["vals"]
                n = slot["n"]
                nc = slot["nc"]
                tot = sizes[g]
                listy = sum(1 for v in vals if "," in v or ";" in v or " / " in v or len(v) > 40) * 2 > len(vals)
                longish = sum(1 for v in vals if len(v.split()) > 4 or len(v) > 30) * 2 > len(vals)
                named = self.named_for(pats, req, g, all_kinds)
                hint = self.hinted(names, g, all_kinds)
                counts: dict[str, int] = {}
                for v in vals:
                    counts[v] = counts.get(v, 0) + 1
                top = [v for v, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:3]]
                distinct = len({re.sub(r"\s+", "", v) for v in vals})
                closed = "" if hint else ("named in the request" if named else ("values are lists" if listy else ("values are long texts" if longish else "")))
                notes = []
                if hint:
                    notes.append("your HINTS point here")
                if n == tot and tot >= 2:
                    notes.append("every table read carries it")
                if distinct >= 2:
                    notes.append("values differ")
                elif n >= 2:
                    notes.append("one value so far")
                key = (bool(closed), -hint, n < tot, -n, -min(distinct, 2), -nc)
                ranked.append((key, " / ".join(names[:3]), n, tot, top, closed, nc, csizes.get(g, 0), ", ".join(notes)))
            ranked.sort(key=lambda x: x[0])
            rows[g] = [x[1:] for x in ranked]
        for g in bare:
            rows.setdefault(g, [])
        self.thin_tables = thin
        return rows, sizes, bare

    def choose_input(self, request: str) -> tuple[Any, str, dict[str, list[str]], int]:
        rows, sizes, bare = self.ask_rows(request)
        left = self.messages_left() or 0
        ready = {g: [it[0] for it in items if not it[4]] for g, items in rows.items() if sizes.get(g)}
        ready = {g: v for g, v in ready.items() if v}
        key = tuple(sorted((g, min(sizes.get(g, 0), 5)) for g in ready))
        text = []
        for g, items in rows.items():
            if sizes.get(g):
                text.append("%s (%d tables read): " % (g, sizes[g]) + "; ".join("%s (%d of %d; %s%s)" % (pk, n, tot, ", ".join(top), "; " + mark if mark else "") for pk, n, tot, top, mark, nc, _, note in items[:ASK_LINES]))
        user = "SHOPPER'S REQUEST:\n%s\n\nMESSAGES ALLOWED: %d\n\nPROPERTIES IN THE TABLES READ, PER KIND:\n%s" % (request, left, "\n".join(text))
        return key, user, ready, left

    def take_choice(self, out: Any, ready: dict[str, list[str]]) -> list[tuple[str, str]]:
        picks = out.get("ask") if isinstance(out, dict) else None
        if not isinstance(picks, list):
            return []
        kinds = list(ready)
        chosen: list[tuple[str, str]] = []
        for x in picks:
            if not isinstance(x, dict):
                continue
            prop = plain(re.sub(r"\s*\(.*$", "", str(x.get("property") or "")))
            if not prop:
                continue
            named = self.kinds_named(str(x.get("product") or ""), kinds)
            order = [g for g in kinds if g in named] + [g for g in kinds if g not in named]
            for g in order:
                hit = [pk for pk in ready[g] if prop == plain(pk) or prop in [plain(n) for n in pk.split(" / ")]]
                if hit and (g, hit[0]) not in chosen:
                    chosen.append((g, hit[0]))
                    break
        return chosen

    def ask_text(self, request: str, steps_left: int | None = None) -> str:
        rows, sizes, bare = self.ask_rows(request)
        left = self.messages_left()
        self.plan_items = []
        lines = ["allowed: %s | sent: %d | left: %s" % (self.allowed if self.allowed is not None else "not stated in the request, so taken as %d until the shopper or store says otherwise" % ASSUMED_MESSAGES, self.messages_sent(), left if left is not None else "unknown")]
        if self.hints:
            lines.append("your HINTS (the property the request's story points at, as you recorded it): " + "; ".join("%s: %s" % (k, v) for k, v in list(self.hints.items())[:8]))
            read_kinds = [g for g in rows if sizes.get(g)]
            unseen = []
            for prod, h in list(self.hints.items())[:8]:
                ks = self.kinds_named(prod, read_kinds) if read_kinds else set()
                if not ks or any(self.hint_score(h, names) for g in ks for names in self.slot_names.get(g, [])):
                    continue
                pat = r"(?<![a-z0-9])" + re.escape(plain(h)) + r"(?![a-z0-9])"
                if any(re.search(pat, plain(t)) for t in self.sent_texts + self.asked_hint):
                    continue
                unseen.append("%s: %s" % (", ".join(sorted(ks)), h))
            if unseen and self.messages_left():
                lines.append("=> hinted propert%s that no table read shows yet: %s; not to be asked in other words: read the tables, then ask what this section plans" % ("y" if len(unseen) == 1 else "ies", "; ".join(unseen)))
        open_count = 0
        unread: list[str] = []
        ready: dict[str, list[str]] = {}
        held: dict[str, int] = {}
        krows: list[str] = []
        all_kinds = list(rows)
        asked_kinds: dict[str, int] = {}
        for t in self.sent_texts + self.asked_hint:
            for g in self.kinds_named(t, all_kinds):
                asked_kinds[g] = asked_kinds.get(g, 0) + 1
        for g, items in rows.items():
            if not sizes.get(g):
                if g == "all" and len(rows) > 1:
                    continue
                if bare.get(g, 0) >= 3:
                    unread.append(g)
                krows.append("%s: no property table read yet (%d listings seen only through views without one), so its property names are still unknown: open the view that shows a listing's property table for %d of its cheapest listings that may fit the request, in one turn" % (g, bare.get(g, 0), TABLES_TO_ASK))
                continue
            items = items[:ASK_LINES]
            opens = [it for it in items if not it[4]]
            open_count += len(opens)
            if not items:
                krows.append("%s (%d tables read): every property seen has been asked about" % (g, sizes[g]))
                continue
            ctot = items[0][6]
            if sizes[g] < TABLES_TO_ASK:
                held[g] = TABLES_TO_ASK - sizes[g]
            if opens:
                ready[g] = [it[0] for it in opens]
            head = "%s (%d table%s read%s%s%s): " % (g, sizes[g], "" if sizes[g] == 1 else "s", (", %d more with only a few properties each, not counted" % self.thin_tables[g]) if self.thin_tables.get(g) else "", (", %d of them your candidates" % ctot) if ctot else "", ("; too few to tell the properties this kind carries from those one listing happens to have: open the table view of %d more of its cheapest listings that may fit the request, this turn, and ask about it next turn" % held[g]) if g in held else "")
            krows.append(head + "; ".join("%s (%d of %d%s; %s%s%s)" % (pk, n, tot, (", %d of %d candidates" % (nc, ctot)) if ctot else "", ", ".join(top), "; " + note if note else "", "; " + mark if mark else "") for pk, n, tot, top, mark, nc, _, note in items))
        if left and open_count:
            need = sum(held.values()) + TABLES_TO_ASK * len(unread)
            rush = steps_left is not None and steps_left < need + 2 * (len(held) + len(unread)) + 6
            now_kinds = [g for g in ready if rush or g not in held]
            waiting = [g for g in ready if g in held and not rush] + unread
            reserve = 0 if rush else min(len(waiting), max(0, left - 1))
            now_kinds.sort(key=lambda g: (asked_kinds.get(g, 0), all_kinds.index(g)))
            plan: list[str] = []
            budget = min(left - reserve, sum(len(ready[g]) for g in now_kinds))
            for g, pk in self.chosen:
                if g in now_kinds and pk in ready[g] and len(plan) < budget and "%s: %s" % (g, pk) not in plan:
                    plan.append("%s: %s" % (g, pk))
            i = 0
            while len(plan) < budget and any(i < len(ready[g]) for g in now_kinds):
                for g in now_kinds:
                    if i < len(ready[g]) and len(plan) < budget and "%s: %s" % (g, ready[g][i]) not in plan:
                        plan.append("%s: %s" % (g, ready[g][i]))
                i += 1
            self.plan_items = [(p.split(": ", 1)[0], p.split(": ", 1)[1]) for p in plan if ": " in p]
            if plan:
                lines.append("=> %d message(s) left and %d unasked properties above: this turn's actions should hold these %d question(s), one property per message and one message per action, in this order: %s%s%s (a property named in the request needs no question when the request settles it). The property goes into the question spelled as the table spells it, the product as the request names it." % (left, open_count, len(plan), "; ".join("%d. %s" % (j + 1, p) for j, p in enumerate(plan)), ("; %d message(s) kept for the kind(s) whose tables are too few yet (%s), to be asked next turn once their tables are read, if they are products to buy" % (reserve, ", ".join(waiting))) if reserve else "", " (the steps left cannot afford more table reads first, so ask now)" if rush else ""))
            else:
                lines.append("=> %d message(s) left, but the tables read are too few to choose the questions well: this turn open the table views named above (%s), and send the questions next turn in the order this section will give" % (left, ", ".join("%d more for %s" % (held[g], g) for g in held) + (", %d for %s" % (TABLES_TO_ASK, ", ".join(unread)) if unread else "")))
        elif left == 0:
            lines.append("=> no messages left")
        elif not sizes:
            lines.append("=> no property table read yet, so no property names to ask about: read %d first, in one turn, then ask in the next" % TABLES_TO_ASK)
        return "\n".join(lines + krows)

    def history_blocks(self) -> list[str]:
        fresh = self.entries[self.rendered :]
        if fresh:
            self.rendered = len(self.entries)
            self.blocks.append("\n".join("[%s] %s %s -> %s" % (e["label"], e["tool"], dumps(e["args"], 400), clip(e["view"], HIST_VIEW_CHARS)) for e in fresh) + "\n")
        if sum(len(b) for b in self.blocks) > HIST_MAX_CHARS:
            while self.blocks and sum(len(b) for b in self.blocks) > 0.6 * HIST_MAX_CHARS:
                self.dropped += self.blocks.pop(0).count("\n")
        head = "ACTIONS SO FAR AND THEIR RESULTS (oldest first%s):\n" % (", %d earlier ones no longer shown" % self.dropped if self.dropped else "")
        return [head] + self.blocks


DIRECTOR_SYSTEM = """You are a shopping assistant serving one shopper in an online store you have never used. Its tools, their behaviour, results and limits, and the shopper's rules are all new: learn them only from the shopper's request, the tool descriptions and schemas, and the results of your own calls. Nothing carries over from other stores.
Your job: get the shopper what the request asks for, following every rule, limit and instruction in it, and end the visit the way the request and the tools say.

TURNS
- Each turn you choose the next 1 to 8 actions: store tool calls with arguments that fit the schema, or helper actions (below). They run in order; you see all their results next turn. Put the action that ends the visit last in its turn, after everything it depends on has been seen; only the re-opens of the basket's listings (see THE BASKET AND THE ORDER) go before it in the same turn.
- Turns are the scarcest resource: each uses the visit's model budget (LIMITS GIVEN BY THE RUN); actions inside a turn cost only store steps. So do as much per turn as your results allow: open several promising pages together, run several different searches together, send a question in the same turn as reads, put several lines in the basket at once.
- Error texts, empty results, counts, paging fields and notes on listings teach you how the store works and what to do next. If a call fails, change it instead of repeating it. Tool and argument names can be odd or misleading: go by the descriptions, and when it matters look at the result (for example the basket) to learn what a call actually did.
- Every store view costs a step, and in a short visit the steps run out first: once the first two or three listings have taught you which view carries the property table and which the notes, read only the table view for the candidates you compare, and open the page view with the notes for the listings you will buy, for every page on a reference chain, and for a listing whose table leaves a requirement unsaid. Open each view of a listing once: LISTINGS SEEN SO FAR says which views it has been opened through, and an opened view does not change.
- Spend steps, messages and time so the visit ends properly before any limit runs out. A store can answer the ending call with changes instead of ending, so first try to end while a few steps are left to act on such an answer.

QUESTIONS TO THE SHOPPER
- Shoppers hold requirements they did not write down, and an item that misses one is rejected even if it meets every written one. Only the shopper can tell you, and only a question naming the property draws it out: spend every allowed message, and spend it early.
- The request itself points at the unwritten wish. Beside the requirements and limits, most requests carry a few sentences of the shopper's story: who will use the product and for what, where it will sit, what will be kept, shown or carried on it, what went wrong with the last one, what the shopper could not measure or check themselves. Each such sentence is there to point at one property of one product the shopper has not stated (a story about what will be kept on it points at how much it holds; about what its screen will show, at the part that draws it; about a person nobody ever measured, at the size; about something that must slot into another machine, at the connection; about what people touch or handle, at what it is made of; about how it looks to those who see it, or about one part of it, at that look or part (its finish, colour, pattern, decoration, or the shape or style of that part), not at its material, size or fit; about whom it is for, at whom it is made for; about what exactly they want, at which kind or type of the product it is; about where it gets propped, stood or carried, at its size; about what came in the box last time, at how many pieces the pack holds). A hint names the property that decides the outcome the sentence describes, not one that merely shares a word with it. On your first turn, before any listing is read, write each such sentence under HINTS with the product and the property it points at, and return them in "hints"; keep them for the whole visit, and once tables are read, move each hint to the property name they show that best names what its sentence describes. A hint is asked under the name a table read shows for that property, in the order MESSAGES TO THE SHOPPER plans; a property no table read shows is not asked in other words, since the shopper answers only a question that names a property as their listings do: read the tables first.
- Property names come from listing data: for each product, read the view that shows a listing's property table for three of its cheapest listings that may fit the request, in one turn (a list row or a page of notes may hold no table; MESSAGES TO THE SHOPPER says for which kinds none, or too few, have been read yet). One table alone shows a dozen properties, most of which that one listing happens to carry, and a question about such a property is a message thrown away; with three read, the properties every one of them carries stand out, and those are where unwritten wishes sit. A property only one listing carries, one whose values are long texts, codes or lists of options, one the request already states, or one that is the same property under another spelling (the section joins such spellings with "/"), is rarely one. The section lists the unasked properties of the tables read, marks the hinted ones and those every table carries, and names the best next question per product: an unwritten wish is about the product the shopper will end up with, so it sits on a property that the listings fitting the request so far (your candidates) carry, whether or not they differ on it (if the shopper wants another value, none of them fits and the search must change); a property carried only by listings that already fail the request is a poor question.
- One product gets each property asked once: when the request's list of items is the name of one store category, it is one product, and the answer about its material or size holds for the whole line.
- As soon as three of a product's tables have been read (fewer only when MESSAGES TO THE SHOPPER says the steps left cannot afford the reads), send its questions in that same turn, one message per action, in the order that section plans them, as many as the messages left and the unasked properties allow; several messages in one turn are right (each gets its own answer), two properties in one message is wrong. With several products, spread the messages over them: for each in the order MESSAGES TO THE SHOPPER plans; a message kept for a product whose tables are not read yet is sent the turn its tables are in. The search can wait a turn, the questions cannot: waiting until the search is finished usually means the budget, steps or time run out first, with messages unused and properties unasked.
- Ask about the product itself (what it is made of, its size, colour, kind, fit, power source and the like), not the listing or the sale (warranty, delivery, a video link, a description); a property whose values are lists of alternatives or options is a poor question and comes last in that section.
- How to ask: one short question (about twelve words at most) about one property of one product the shopper asked for, asking what they would like, for example "What <property> would you like for the <product>?"; the product as the request names it, the property copied letter for letter from the listing data (never your own spelling of it, a translation, or a broader or vaguer word: the shopper answers only a question that names the property as their listings do). Do not ask what the request already states (listing data settles that), do not ask them to judge a listing (they cannot see the listings), do not offer a list of options.
- An answer settles only the property asked about, nothing else. "Don't mind", "any will do", "just choose" settles that one property as free; a reply they did not follow (asking you to put it more simply, saying they are lost or confused) counts the same, asked and free, since rephrasing the same property almost never brings an answer and wastes a message. Neither kind says anything about the other properties: never read it as "the shopper is easy about everything" and stop asking; go on to the next unasked property or product.
- When an answer states a requirement, recheck your candidates against it and, in the same turn, run a new search whose phrase holds the product kind and the stated value: the listings you found before the answer were found without that word and rarely carry it, while the listings that carry it mostly appear only when the phrase names it. When the answer gives the requirement by reference to another listing's page ("the same as on the page for the <product> sold by <seller> that has <properties>"), it is a reference chain exactly like one in the request (REQUIREMENTS GIVEN BY REFERENCE): in that same turn record it in "references" with the words that describe the listing to read, and search the seller's name with the most telling describing word (the name on its own when that finds none of the seller's rows); the value is read on that page, never from a list row, a title or a listing you already hold.
- An answer that names a value or a bound for the property asked ("only X", "at least N", "it has to be X", "I'd rather wait than settle") is a requirement on that product as firm as any the request writes, even when the request also says any fit or anything is fine: those easy words cover only what was never asked. SHOPPER'S ANSWERS AGAINST THE BASKET sets each answer beside the basket listings' own values for the property it names; before you end, each such answer must be met there, and a basket listing whose value does not meet it is swapped for a listing whose own data does, found by search when none is held.
- Messages are for learning what the shopper needs, and for anything the request tells you to send them (exactly that, once, at the point the request names): not for progress reports, apologies, summaries of what you found or asking permission (for example to raise the budget), since the request already says what the shopper accepts and the ending call carries any explanation.

SEARCHING
- Learn the search tool from its schema and its results, and use everything it offers: every mode (a phrase search, a category or shelf listing, a maker or seller filter), every filter (a price ceiling, a sort) and its paging (a page number, an offset, or a cursor or token the result hands back).
- A result that reports more rows than it shows, or hands back a value for continuing, is one page of a longer list. A short list (two or three pages beyond the one seen) is read to its end before any rewording: the fit is as often on page 2 or 3 as on page 1, and a reworded search mostly repeats the first page; pages a number or offset addresses can be asked for several at a time in one turn. A list the store will not continue although rows remain shows only a sample of the kind: the rest needs another phrase.
- Build a phrase from the store's own words: the kind of product as the listings and categories name it, plus requirement values written as the listing data writes them. Leave out words listings would not carry (the shopper's story, their own car, device or room, budget figures, words like cheap or in stock): they match nothing or everything.
- How many requirement words belong in one phrase depends on how this store matches, and SEARCHES SO FAR tells you. Rows the same as an earlier search's mean the store ignored what you changed (words added or past the first few, an argument such as a price cap in that mode): do not add to such a phrase; put the requirement's own value first or alone, or use the mode that honours the argument. A phrase that returns nothing means the store wants every word of it: keep to the kind plus one requirement word at a time. A count that sits at the same ceiling for every phrase and never comes back empty means the store matches any of your words and ranks listings by how many they carry: there a one- or two-word phrase buries the listings you want under everything sharing the common word, while one phrase holding the kind and every requirement value (the stated ones and each value an answer adds) puts the listings carrying all of them first, though rarely all on page 1: page it two or three pages deep. A count below the ceiling is a complete list you can read to its end; the requirement's rarest value word alone (a seller, a colour, a model, a material) is what brings a capped list below it. With several values in one phrase, give each its single most distinctive word, the rarest first: a value written out in several words fills the few words the store weighs and pushes the next value out.
- A category or shelf listing shows the kind without depending on words. A price ceiling set low and raised step by step shows the cheapest listings of a kind whatever the result order; when results come cheapest first (rows rising in price), the first pages already hold the cheapest candidates, and a narrower phrase brings the cheapest fit into them. A search by a seller's or maker's name that returns only a handful of rows: open every one of them. An error or empty reply from a mode used for the first time means its arguments were wrong (a listing by category or maker needs that name, as the store writes it, in the mode's own argument), not that the mode is useless: fix the arguments once before you give it up.
- Never repeat a search you already made, nor one that returned nothing (see SEARCHES SO FAR).
- Read the shopper's words through the store's vocabulary: the category names, property names and values in the tool schemas and listing data are how this store expresses what the shopper asked for. One product usually maps to one store category; when several could hold it, the one whose name matches the shopper's own words most closely is the likely one. When the shopper uses everyday words for something the store records in a field of its own (availability, size, origin, condition and the like), that field is the evidence; a listing need not repeat the shopper's words.

READING A LISTING
- Base every claim on data you have seen, and use only listing ids that appeared in results. List data is often partial: a listing's own page or detail view may hold properties a list leaves out. "The data does not say" is not "it does not meet": find out before you rule a listing out.
- Never rule a listing out for a reason the shopper did not give (who it seems made for, how its title looks, whether you think it suits the story, "it does not look like a proper <product>"): only stated or revealed requirements and the category count, never your own taste; if unsure what the shopper would accept, ask them.
- Read a requirement against a listing like this. Where the listing's property data has the property the shopper names, only that value counts: it shows the requirement when the shopper's words appear in it in the same order (spaces, hyphens and underscores count alike; other words may come before or after, such as "dark green" for "green" or "sky blue" for "blue"), and in a list of values separated by commas, slashes, "and" or "or" any one item may hold them (a title or value naming several alternatives shows each of them). A value joined with "+" is one combined value, not a list ("ssd + hdd" is not "ssd"); a different word is a different value even if the meaning seems close (such as "navy" for "blue", "hot swappable" for "hot pluggable"). A property the data carries under more than one spelling (letters and digits alike; case, spaces, hyphens and underscores aside) is one property, and any one spelling that shows the value counts.
- Where the property data has no such property at all, the same words may be shown by the title, another property's value or the page's notes, read the same way, never by the description; a field of options or variants counts only in that case too. What a listing says it fits or works with (models, parts, series) counts wherever it is written, its description included. A requirement nothing in the listing states (for example a required condition when no condition is written anywhere) is not met.
- A listing marked NOT IN ITS STORED DATA for a wanted value (WANTED WORDS IN THE STORED DATA) does not show that requirement, whatever a near word, a wider word or a part of the value suggests: it is not a fit while any listing seen shows the value. When the cheapest listings read fail a requirement, the fit is dearer: read next the listings that section lists as holding every wanted value, before cheaper rows that hold none.
- A value word that can be written in several ways (joined, hyphenated or spaced) is searched in each way when the first finds no fit.
- An exclusion ("not X", "cross off X") rules out the value X itself, not other values that merely contain or resemble it; a listing whose deciding value is another value meets it. "A or B", "either A or B" and "A or B, just choose one" mean a listing fits when its data shows either value: it is not a question for the shopper, neither value is preferred, and a listing whose data shows neither does not fit.
- A requirement names its property first and then a figure or value. When a unit word after the figure does not fit that property (for example "calories per serving at least 160 servings"), the property named first decides: compare the figure with that property's value, and never read the stray unit word as another property.
- Figures are compared as numbers in the unit the listing states, converted when the requirement uses another unit: a bound such as at least, at most, up to or within is satisfied by equality too, "1.10" and "1.1" are the same figure, and a range the listing states satisfies a figure that lies inside it. A model or series pattern with placeholder characters (such as a code ending in "xx") covers every model whose other characters match, unless the listing names that model as an exception.
- What kind of product a listing is comes from the store's category for it: a listing filed under the category the shopper asked for counts as that kind even when its title names something else, so it stays in the running.
- Some wishes cannot be checked in this store at all: when no listing you have read carries any data on the point (for example a delivery date when no listing states delivery times), it cannot tell listings apart. Note it, choose on the requirements the data can show, and never end without a purchase only because of such a point.

RULES THAT BIND TWO PRODUCTS, AND PRICE LIMITS
- A conditional rule ("X's P must be V, except when Y's Q is W"; "should Y's Q turn out to be W, I need X's P to be V") is one requirement on X that switches on what you choose for Y: it is not a requirement on Y (Y's Q may be anything) and not an unconditional requirement on X. Keep it whole in NEED, condition and all, until the order is placed. Decide it from the own data of the listing you actually choose for Y: the condition holds only when that data shows the value W, read like any requirement; when Y's data does not state Q, the condition does not hold. When it holds, apply or lift X's requirement exactly as the rule says; when you change the Y listing, decide it again; and when the request wants the lowest total, the pair of Y and X listings with the lowest sum under the rule is the one to buy.
- A limit set on one product binds only listings of that product, never another product the request names, whatever words the two share.
- Price limits name what they bind. A limit on one product ("no more than N for a <product>") binds each unit of that product: compare the listing's own price with it. A limit on the whole ("N for the lot", "in total", "all told", "altogether") binds the order's total: compare the basket's total as the store shows it. Both kinds hold at once. A limit on one product tells you where its fit lies: among the cheapest listings of that kind, which often look plainer or odder than you picture the product; one filed under that product's category whose data shows the stated properties at a price within the limit is a fit. Record the limits in "limits" as numbers in the store's price unit: PRICE RULES YOU RECORDED then shows the basket's lines and total against them.

REQUIREMENTS GIVEN BY REFERENCE, AND COMPATIBILITY
- A requirement can be given by reference, in the request or in an answer: "the same <property> as the <product> from <seller>'s listing with <properties> (or wherever that page leads)". That listing is a place to read the value, not a pick: buy it only if it meets every requirement like any other. The seller's name and the properties that identify it describe the listing to read, not the product to buy: they are not requirements on your purchase; only the one value taken from the end of the chain is. Find it first, before you compare candidates on that property. The listing to read is often of a quite different kind from the product you buy: that kind is no product to buy, no question goes to the shopper about it, and the reference is never unrelated or off-topic; the product to buy stays the kind the request or the answer names.
- Which listing it is: the one from that seller whose own data (its page's notes, its table, its title or category) shows every property used to describe it. A seller has several listings of the kind and they look alike in a result list: a row shows no properties, so a row cannot identify it, and the first row or the cheapest is no more likely than the others. A listing from the named seller whose data lacks or contradicts one describing property is a different listing with a different chain: a value or a pointer read from it is wrong, whatever its notes say. So open the page view of every listing of the named kind that the seller's search shows, several in one turn, until one shows them all, and continue the seller's list when it announced more rows than it showed.
- A page on a chain that neither states the value nor sends you on is not the listing meant: open the seller's other listings of that kind, the ones carrying the describing words first.
- Such a chain often runs four or five pages before one states the value itself. Open every listing it names, through the page view that carries the notes, until a page states the value without sending you on; keep steps for it, since each hop costs a search (the seller's name with the kind's name) and the opening of the one or two seller's rows that search ranks first.
- Record each listing of the chain in "references" with the words that describe it (the shopper's words for the first, the sending page's words for the next). REFERENCE CHAINS YOU RECORDED then shows, for each, whether its own data shows every describing property (checked for you against its stored data, with the fragment relied on) and the texts on the last listing's page: a listing marked NOT MET is not the one, so keep looking among the seller's other listings before you go on; one marked UNSAID does not state that property, so open its views not yet opened, and when every view is open and it still does not state it, it is not shown to be the one either (the listing meant states every describing property).
- A seller's name is the rarest word in a store. First search it together with the name of the kind the listing is (its product word, such as "desk lamp") and at most one telling describing word, in a short phrase: a store that ranks the listings carrying more of the words first then shows the listing meant at the top, even when its row does not show those words. Open the page view of the one or two rows of that seller ranked first, in the same turn; the listing meant is almost always one of them, so open further rows only when neither page shows the describing properties. When that search returns none of the seller's rows, search the name on its own, which returns that seller's listings, usually a short list. Once the seller's rows are on screen, searching the name again with more words returns the same rows: open them instead.
- Open the PAGE view, the one with the seller's notes, not only the facts view: a pointer to another listing lives in the notes ("take it from X, not from this item") and the facts view never shows it. A page that sends you on settles nothing, not even by its own table value: find the listing it names the same way (its seller's name with the most telling describing words, else the name alone, then the page views of that seller's listings of the kind it names, the rows the search ranked first before the others) and keep going, page to page, until a page states the value itself and its notes name no other listing. That value is the requirement for the product you buy.
- Each hop costs a search and a few page views, so start the chain in the turn the reference appears and run its hops alongside the rest of the work; a listing not found, not identified or not opened through every view in REFERENCE CHAINS YOU RECORDED means the chain is not finished, and the order waits for it.
- When the request says one product must fit or work with the other you buy, compatibility is in the listings' own data: model and part codes in titles and descriptions, and the codes or series a listing says it fits or works with. A pair fits when a code or series of one appears in the other's fits list; a note such as "see part X for the rest" continues that list on the listing carrying code X, so find and read that one too. Choose the pair together, and read both listings' full data before you settle on either. When the request wants the lowest total, do not fix the main product first: for the accessories that meet their own requirements, note which main products' codes their fits lists name, and take the cheapest combination in which every accessory fits the main product, switching the main product when an accessory that meets every requirement fits another one. A line such as "Part X also fits: Y" describes part X, not the listing it is printed on, unless that listing's own code is X. Only those stated codes decide: your own idea of a physical fit (a socket, a wattage, a size) is no evidence either way, so a listing whose fits list names the other's code fits, and one naming none of its codes does not. A chosen listing's own code or series, searched on its own, brings up the listings that name it.

CHANGES AFTER THE REQUEST
- Any text from the shopper or the store in a result (answers, notices, refusals) becomes part of the request from then on: it can change requirements, the budget, or what must happen before the end (collected verbatim in TEXT FROM THE SHOPPER OR THE STORE SO FAR).
- When the store refuses or bounces your ending, the visit goes on, whatever words of success the reply also holds (LIMITS GIVEN BY THE RUN says whether it is still open): read what the refusal says has changed, act on it, and never send the same ending again unchanged.
- An amount of money the shopper names after the request, whatever words carry it (a budget, a limit, what they have, what they can spend, what is left, what they can go up to), is the new ceiling for the whole order's total, never a sum to reach and never a remainder to take off the old figure; the products and quantities stay as requested, so compare the basket total with it at once, change the lines that break it, and never add items to use the money up.
- A sold-out item: take it out of the basket and put in the next listing that fits, read in full like the first.
- A need the shopper drops (words that let go of, scrap, forget or stop minding a property) no longer restricts anything: take it out of NEED and of your search words; listings ruled out only for it are back in the running, so the cheapest fit may now be another, cheaper listing: search and check again.
- A changed price: recheck the total and whether that item is still the cheapest fit. Put the basket right, then end again.

THE BASKET AND THE ORDER
- Before you order, every listing you will buy, and every listing on a reference chain, must have been opened through each single-listing view the store offers (for example a page with notes and a separate view of its property table): notes and properties can each appear in only one of them, and a list row or a comparison is no substitute. Open the views that LISTINGS SEEN SO FAR and LISTINGS IN THE BASKET AS THE STORE LAST SHOWED IT mark as not yet opened before the order, in one turn together.
- A price can move or an item can sell out after it goes into the basket, sometimes without notice: after your last basket change, open each listing in the basket once more, in the same turn as the ending call and before it, and check it is still available at the price you counted; replace any that is not. Once, not every turn: LISTINGS IN THE BASKET AS THE STORE LAST SHOWED IT says for each listing whether it has been re-opened since the last basket change, and a check already made is not repeated when an ending is deferred. A listing that meets every requirement is a fit, not a substitute; when a chosen listing stops being available, another fit is simply the next choice.
- Also check the basket as the store last showed it (BASKET AS THE STORE LAST SHOWED IT, once you have named the tool that shows it): exactly one line for each product the request asks for, at the quantity asked for, and nothing else.
- A quantity is a number of units of one listing on one line: "3 headboards" means one headboard listing held three times or with quantity 3, never three different headboards; "2 x laptops" is one laptop listing with quantity 2; the only thing a second line of the same kind can serve is a second product the request names separately. One thing the request names is one product even when its words list several items, above all when they are the name of one store category ("pots, pans & lids" is one product and one line, not three). The quantity comes from the request alone: a minimum order, a pack size or a note in a listing never changes it, and a product the request mentions more than once is still one line at the one quantity stated. Take out alternatives or leftovers from earlier tries.
- An argument's name may not say what it does (a field called line, position or number can set the number of units), and a tool that adds units adds them again on every call: after each basket change, read from the store's reply or the basket view what the basket now holds, line by line and unit by unit against the request, and correct it before you end.

WHAT THE REQUEST WANTS LOWEST
- "The cheapest one that fits" for a product, or "the lowest total" for the order, means the cheapest fit of that kind in the whole store, not among the few you happened to see: the order is judged against the cheapest fitting listing the store holds. A result list in the store's own order says nothing about price unless the store sorts by it (rows in rising price order), so the cheapest fit is as likely on the last page as on the first.
- Under such a request the pages come before the candidate reads: build the narrowest list that still holds every fit of that kind (the kind's words plus the requirement's own value words, so that the count the store reports falls below its cap when it can), page it to its end, and only then read upwards from the lowest price; a list capped at a fixed count cannot be seen to the end, so narrow it or page it as far as the steps allow while keeping steps for the reads and the ending. Record in "lowest_price" the products this applies to (every product when the total is to be lowest): PRICE RULES YOU RECORDED then lists, per kind, the searches that showed it and still have pages not continued, and an order is held back while such lists remain and the steps allow.
- Work upwards from the lowest price: read and check the cheapest candidates of the right kind first; the first whose data shows every requirement is the answer, and dearer listings need no reading. A cheaper listing you have not read is the main risk to that answer: read it rather than settle on a dearer one you happened to read first. Using CHEAPEST LISTINGS SEEN OF EACH KIND, before you buy open every unread listing of that kind cheaper than your choice, unless its row already shows a value that breaks a requirement. When search results are ordered by price, the fit you want is the first in that order whose data shows every requirement.

BEFORE YOU CONCLUDE THAT NOTHING FITS
- Make sure you have really looked: each product's kind searched by its own category words and continued through the pages the store offers, every mode and filter of the search tool tried, other wordings, and the full pages of the cheapest and most promising listings of that kind (an unread listing is not evidence of anything).
- A requirement counts as met only when a listing's own data shows it, read as under READING A LISTING; a cheap listing whose property data shows another value, or that is filed in another category, is not a fit. When nothing in a listing states a point, read its full page or look for listings that show it.
- Nothing fits only when the data shows every candidate of the requested kind failing a requirement (another value, a price over the ceiling, sold out). A listing whose data does not state a point is not contradicted: it stays a candidate, and "no listing read shows it" is never "nothing fits"; with such candidates left, an order of the best of them serves the shopper better than no order.
- Some requests say what to do when nothing fits, such as "say so and put forward the listing that falls short on the fewest requirements". Record it in "if_nothing_fits". Such a store almost always holds a full fit for every product, so keep searching until you find it and buy it; a listing that misses a requirement is never the one to buy. Only when the search really shows no listing of a product's kind meeting every requirement, end without a purchase. The ending text then puts forward, by listing id, only the closest listing for that product, and names no other listing id. The closest is the listing of that kind whose own data fails the fewest of the request's requirements (being out of stock and a price above a limit each count as one; a point the data does not state is not a failure; use check to count), the cheapest of those or one within the percentage the request allows above it.

HELPER ACTIONS (they use no store steps)
- {"tool": "check", "args": {"ids": ["<listing id>", ...], "requirements": ["<requirement in the shopper's words>", ...]}}: reads the full stored data of up to 6 listings and reports, for each requirement, whether the data shows it met, not met or does not say, with the quoted fragment. Give one product's listings with that product's requirements per call (a requirement meant for another product is noise on these listings), and write each requirement with its property name and value, figure or bound as the shopper gave them.
- {"tool": "view", "args": {"ids": ["<listing id>", ...]}}: shows the full stored data of up to 4 listings.

YOUR NOTES
Keep short working notes (at most about 1800 characters; every character costs time) and rewrite them every turn, in these parts:
NEED: each requirement as the shopper stated it, grouped by product, and every rule and limit in the request, each written whole: a rule that holds only under a condition keeps its condition and the product the condition looks at; a price limit says what it binds (one unit of a product, or the whole order); what the request wants lowest (one product's price, or the order's total).
HINTS: each sentence of the request that states no requirement, limit or instruction but tells the shopper's story, with the product and the property it points at (written on the first turn, kept for the whole visit).
OPEN: per product, the properties its listings' property tables carry (under their exact names, from MESSAGES TO THE SHOPPER) that the request does not settle, each marked unasked, asked, or settled (with what the shopper said). A list of property names, not of things to verify; empty until you have read a property table, never filled from your own idea of what such a product has.
HEARD: what the shopper and the store have told you since the request (answers, notices, refusals, changes).
CANDIDATES: listing ids with the quoted evidence for each requirement.
USED: steps, messages and anything else limited: used and left.
PLAN: what you will do next and why.

OUTPUT
Return JSON only:
{"step_limit": <the number of store steps the request or the store allows for this visit, as a whole number, or null when none is stated>, "notes": "<your rewritten notes>", "candidates": {"<product>": ["<listing id>", ...]}, "wants": {"<product>": ["<each value a requirement on it names, stated in the request or given in an answer, in the words given: no property name, no figure or bound>", ...]}, "references": {"<product>: <property>": [{"id": "<listing id, or null while it is not found yet>", "described_as": ["<each property the shopper's words (for the first listing) or the sending page's notes (for the next) use to describe it: the seller as 'seller <name>', then each property with its value, such as 'position rear' or 'pack size 3 pieces'>", ...]}, ...]}, "messages": {"allowed": <how many messages the shopper allows, as a whole number, or null when the request states none>, "sent": <number sent so far>, "asked": ["<product>: <property>", ...], "next": "<product>: <property> you would ask about next, or none>"}, "hints": {"<product>": "<the property the request's story points at, in one or two words>"}, "lowest_price": ["<product the request wants at the lowest price, or every product when the order's total is to be lowest>", ...], "units": {"<product>": <the number of units of it the request asks for, 1 when it names no number>}, "if_nothing_fits": {"do": "<name_closest when the request asks you, should nothing fit, to say so and offer, propose or put forward the listing that falls short on the fewest requirements; decline when it asks you only to say so, or not to offer a substitute; unstated otherwise>", "band_percent": <the percentage above the cheapest such listing that the request accepts, as a number, or null>}, "limits": {"order_total": <the limit on the whole order's total as a number in the store's price unit, or null when none is stated>, "per_unit": {"<product>": <the limit on one unit of that product, as a number>}}, "message_tool": "<the store tool that sends a message to the shopper, or null if none>", "ending": {"order": {"tool": "<store tool>", "args": {}}, "decline": {"tool": "<store tool>", "args": {}}, "intended": "order|decline"}, "basket_view": "<the store tool whose result shows the basket's lines and quantities, or null if none seen yet>", "price_field": "<the field name under which listing rows and pages carry the price, exactly as the data spells it, or null if no listing seen yet>", "actions": [{"tool": "<name>", "args": {}, "why": "<short reason>"}]}
"step_limit": the limits shown use the smaller of it and the platform's figure.
"candidates": per product, the listings still in the running (meeting everything known so far, or not yet ruled out); see WHAT YOUR CURRENT CANDIDATES DIFFER ON next turn.
"wants": per product, the value words its requirements name (from the request and from answers; a value given by reference once read on its page), kept current. A value a requirement excludes ("other than X", "anything but X", "not X", whether given directly or by reference) is never a wanted value: leave it out of wants and keep the exclusion, with its value, in notes. WANTED WORDS IN THE STORED DATA then shows which listings seen of that kind hold every one of them in their stored data, read or not, and for each candidate and basket listing where each value is held or that it is NOT IN ITS STORED DATA.
"references": per requirement given by reference, its chain in order (the listing the request or the shopper describes first, then each listing a page's notes sent you to), each with the words that describe it; keep it current from the moment such a requirement appears: a listing described but not found yet has "id": null, and a listing whose check says NOT MET is replaced by the seller's listing that does show every describing property.
"messages": the product properties asked about, and the next worth asking (the most widely carried property of a product being bought that no message has named yet, as MESSAGES TO THE SHOPPER ranks them, preferring a product not asked about yet); "next" is none only when no message is left or no unasked property remains for any product to buy; otherwise this turn's actions hold the questions.
"messages": the product properties asked about, and the next worth asking (the hinted property of a product being bought, else its most widely carried unasked property, as MESSAGES TO THE SHOPPER ranks them, preferring a product not asked about yet); "next" is none only when no message is left or no unasked property remains for any product to buy; otherwise this turn's actions hold the questions.
"hints": per product, the property its story sentences point at (from HINTS; leave out a product whose story points at nothing, and give an empty object when the request tells no story); MESSAGES TO THE SHOPPER then marks that property in the tables read.
"lowest_price": the products whose price the request wants lowest (every product you buy when it wants the lowest total; an empty list when it names no such wish). "limits": the limits the request or the shopper states, as numbers in the store's price unit, "per_unit" keyed by the product as you name it in "candidates". Keep both current: a later notice from the shopper can change them.
"if_nothing_fits": what the request says to do when no listing fits, read from the request's own words (see BEFORE YOU CONCLUDE THAT NOTHING FITS).
"message_tool": as far as the tool descriptions say; messages sent are counted from your calls to it.
"ending": from the tools, how this store ends a visit with and without a purchase, and which you intend now; keep it current.
"basket_view": a tool that only views the basket, or the basket-changing tool when its reply shows the whole basket.
"price_field": lets the input list the cheapest listings of each kind for you."""

REVIEW_SYSTEM = """You review a shopping assistant's decision to end a visit now. You see what the assistant sees; the proposed ending is at the bottom (PROPOSED ENDING).
First list what is still unexplored or unsettled, as far as the remaining steps, messages and time allow: wordings, filters, categories or result pages not tried (see SEARCHES SO FAR; a search that announced more rows than it showed and was never continued is unfinished); promising listings whose own page was not read (marked UNREAD in CHEAPEST LISTINGS SEEN OF EACH KIND); open properties not asked about while messages remain; anything the request asks for before the end that has not happened.
Messages are the only way to learn requirements the shopper did not write, and an order that misses one fails. MESSAGES TO THE SHOPPER shows the messages left and, per kind of listing, the properties no message has named yet. If messages are left and that list holds properties of a product being bought (a property the request settles or whose values are lists does not count), ending now is premature unless the steps or time left cannot fit the questions and what follows from their answers: do not approve; give the questions as the actions, one short question per property naming the product, as many as the messages left allow. An earlier "don't mind" or confused reply settled only the property it answered, not the others. An answer that names a value or a bound is a requirement as firm as the request's own, whatever easy words the request also uses: when SHOPPER'S ANSWERS AGAINST THE BASKET shows a basket listing whose own value does not meet such an answer, do not approve; name the listing that meets it to swap in, or the search that finds one.
A conditional rule (a requirement on one product that holds except when, or only when, another product's property shows a value) is decided on the listing actually chosen for that other product, from its own data, and the requirement applied or lifted accordingly; it is never a requirement on the other product. Each price limit holds where it binds: a limit on one unit of a product against that listing's own price, a limit on the whole order against the basket's total (PRICE RULES THE ASSISTANT RECORDED shows both). A listing under the asked category whose data shows the stated properties within its limit is a fit however plain or odd it looks; the assistant's own idea of what the product should look like is not a reason to refuse it.
If the store refused an earlier ending, approve only when the basket now meets what the refusal changed: an amount of money the shopper named is the ceiling for the whole order's total whatever words carried it; a sold-out item must have left the basket; a dropped need lets cheaper listings fit. Repeating the same ending unchanged is not an answer to a refusal.
Approve only if the evidence supports the ending, ending now serves the shopper better than continuing, and all of these hold:
1. Every requirement of what is bought is shown by that listing's own data, read value by value against the request's words, or the shopper settled it. A wanted value marked NOT IN ITS STORED DATA for a bought listing (WANTED WORDS IN THE STORED DATA) is not shown by it; when that section lists another listing holding every wanted value, do not approve: give its reads and the basket swap as the actions.
2. A requirement naming a property is shown by that property's value when the listing's property data has it: the shopper's words in it in the same order, or in one item of a list separated by commas, slashes, "and" or "or", count (only "+" joins items into one value), any one spelling of the property that shows the value counts, and a different word does not; figures compare as numbers in the listing's unit, equality satisfying a bound, a stated range covering a figure inside it, and a model pattern with placeholder characters covering the models whose other characters match. Only when the property data has no such property may the title, another property's value or the page's notes show it; the description never does, except for what the listing says it fits or works with, which counts wherever it is written.
3. The listing's category is the kind of product asked for (the store's category decides, not the title). A cheap listing whose property data shows a different value, or whose data states the property nowhere, is not a fit.
4. Nothing the request asks for before the end is missing.
5. Every listing to be bought has been opened through each single-listing view this store offers (see LISTINGS IN THE BASKET AS LAST SHOWN); one view may hold the property table and another the notes, and a list row or a comparison is no substitute.
6. Every requirement given by reference to another listing's page (in the request or an answer) has a complete chain (REFERENCE CHAINS THE ASSISTANT RECORDED): each listing of it identified (its own data shows every property used to describe it, as the checks there show; one marked NOT MET is another listing of that seller, and everything read from it is wrong), each opened through the page view carrying the notes, the last page stating the value itself with notes that name no other listing (a page whose notes send on settles nothing, not even by its own table value), and the value used taken from that last page.
7. When the request says one bought product must fit another, both listings' own data (model or part codes and the codes or series they fit) show the match.
8. Each listing in the basket has been opened again after the last basket change and is still available (items can sell out or change price after being added, without notice).
9. The basket as the store last showed it (BASKET AS THE STORE LAST SHOWED IT; a basket not shown since its last change is unchecked) holds exactly one line per product the request asks for, at the quantity asked for, and nothing else. Count the products the request names and compare product by product and unit by unit ("3 headboards" is one headboard listing with quantity 3, never three different headboards; a product whose words match the name of one store category is one product and one line even when that name lists several items; a listing's minimum order or pack size never changes the quantity). More lines than products, or a line whose units differ from the quantity asked, fails the shopper: do not approve it.
10. If nothing is bought, the search really covered what the store offers under the request as it stands now (including any change the shopper or store announced).
When the request asks, should nothing fit, to say so and put forward the listing that falls short on the fewest requirements (IF NOTHING FITS shows it when the assistant recorded it), the store almost always holds a full fit for every product. An order still needs full fits: conditions 1 and 3 hold as written, and a listing that misses a requirement is never right to buy. A decline is right only when the search covered the kind thoroughly and found no full fit. Its text must then put forward, by listing id, only the closest listing: the fewest failed requirements, with out of stock and a price over a limit counting, and the cheapest of those or within the request's percentage. It must name no other listing. Otherwise do not approve: give the searches and reads that may find the full fit.
A decline claims the store has nothing that fits: approve it only when the listings' own data shows every candidate of the requested kind contradicts the request, not merely because no fit has been confirmed yet (a candidate whose data does not state a point is not contradicted). A narrow or unfinished search is not that evidence, nor is a listing nobody opened; while steps remain, a search for that kind whose further pages were never read, or a mode or filter never tried, means the search is unfinished: give the continuation as the action. A wish no listing in the store carries any data about is no grounds for a decline either (it cannot be checked here).
An order of a cheapest fit (or a lowest total) is premature while an unread listing of the same kind cheaper than the chosen one appears in CHEAPEST LISTINGS SEEN OF EACH KIND and the steps left can open it: give those reads as the actions. It is premature too while a search that showed that kind announced more rows than it showed and was never continued (PRICE RULES THE ASSISTANT RECORDED lists them; a list in the store's own order can hold a cheaper fit on any page, and only a list whose rows rise in price is exempt) and the steps left can continue it and still read and end: give those continuations as the actions.
Beyond these conditions, do not block an order only to re-check facts the data already shows or for a risk nothing in the data suggests. Weigh the time and model budget left (LIMITS GIVEN BY THE RUN): each extra round costs seconds and one of the assistant's remaining turns, so ask only for what they can finish, and when little is left, approve an order the data supports rather than start more searching.
If ending now is premature, give the next 1 to 5 actions (store tool calls with schema-valid arguments, or the helper actions check and view); when a bought listing has an unopened view or a reference chain is unfinished, those reads (or the search for the chain's next listing, by its seller's name with the most telling describing words, or the page views of that seller's other listings of the kind when the recorded one is marked NOT MET) are the actions to give.
Be brief: at most five short items in "unexplored", one sentence of reason, short "why" texts.
Return JSON only: {"unexplored": ["<item>", ...], "approve": true|false, "reason": "<one sentence>", "actions": [{"tool": "<name>", "args": {}, "why": "<short reason>"}]}"""

CHOOSE_SYSTEM = """You choose the questions a shopping assistant sends a shopper before buying. The shopper has written a request; besides what it says, the shopper may hold one firm wish about one property of a product that the request never mentions, and reveals it only to a question that names that exact property. Every message is precious: the shopper answers at most {n}.
Below are the properties found in the store's property tables for each kind of product to buy, with how many of the tables read carry each and some values. Choose {n} properties to ask about, one per message, best first, copied exactly as listed. Choose the properties that define what this kind of product is and that a buyer of it would most likely care about (what it is made of, its kind or type, its look, its size, how it is powered or connects, its main specification), not details of the listing or the sale (warranty, condition, delivery) and not a field naming the variants a listing is sold in (options and the like). Skip a property the request already settles. With several products, spread the questions over them.
Reply with JSON only: {{"ask": [{{"product": "...", "property": "..."}}]}}"""

CHECK_SYSTEM = """You compare store listing data with a shopper's requirements. For each listing and each requirement answer "met" if the data shows the requirement is satisfied, "not_met" if the data shows it is not, and "unsaid" if the data does not settle it. Read every part of the data, including tables and notes, and decide from the words and figures written there, never from the meaning you would give them, from synonyms or from what you know of such products: a word that is not written in the data is not shown by it, however close its sense.
Which part of the data decides. Property names match by their letters and digits alone (case, spaces, hyphens and underscores do not count). A requirement naming the seller or the category of the listing is decided by the field that carries the seller or the category, wherever it sits in the data. When a requirement names a property and the listing's property data has that property, decide it from that value only, under every spelling the data carries it (if any one of them shows the requirement, it is met); what kind of product the listing is comes from its category, whatever its title or style says. When the property data has no such property, the title, another property's value or the page's notes may show it the same way, while the description never counts and a field of options or variants counts only in this case; when none of them states it, answer "unsaid". One exception: a statement of which models, parts or series the listing fits or works with counts wherever it is written, description included. When the notes say that a property's value is to be taken from another listing rather than this one, this listing's own value does not settle that property: answer "unsaid" and quote the note.
How a value is compared. A requirement's words are shown when they appear in the value in the same order (spaces, hyphens and underscores alike, other words before or after allowed). A value that lists several items separated by commas, slashes, "and" or "or" shows each item; items joined by "+" form one combined value, not a list. An exclusion ("not X", "other than X", "anything but X") is met when the deciding value is something other than X itself, not met when it is X or a list holding X, and unsaid when nothing states the property. Figures are compared as numbers, not as strings: "at least", "at most", "or more", "or less", "up to", "within" and "between" are bounds that the listing's figure satisfies when it lies inside them, equality included, and figures that differ only in their written form are the same number; a listing figure given as a range satisfies a required figure that lies inside it and a bound that every end of the range satisfies; when the listing states its figure in another unit than the requirement, convert it before comparing and quote the listing's figure as written. A model or series pattern with placeholder characters (such as a code ending in "xx") covers every model whose other characters match, unless the data names that model as an exception; a listing's own model number or alias is covered by a pattern the same way.
Quote the exact fragment you relied on, copied from the data as written (quote marks and punctuation may be left out).
Return JSON only: {"<listing id>": {"<requirement number>": {"answer": "met|not_met|unsaid", "quote": "<exact fragment or empty>"}}}"""


class Episode:
    def __init__(self, pdata: dict[str, Any]) -> None:
        self.log = Log()
        env = pdata.get("environment") if isinstance(pdata.get("environment"), dict) else {}
        policy = env.get("policy_view") if isinstance(env.get("policy_view"), dict) else {}
        self.request = str(policy.get("query") or pdata.get("query") or "")
        self.opening = policy.get("opening")
        max_steps = policy.get("max_steps") if isinstance(policy.get("max_steps"), int) else 40
        self.wire = Wire(env.get("binding") if isinstance(env.get("binding"), dict) else {}, max_steps, self.log)
        self.tools = Tools(policy.get("tools") or [])
        self.memory = Memory()
        self.finished = bool(policy.get("finished"))
        self.turn = 0
        self.chooser_calls = 0
        self.fixed_text = ""
        self.failed_turns = 0
        self.idle_turns = 0
        self.empty_turns = 0
        self.bounced = ""
        self.asked_bounce = False
        self.rejections = 0
        self.cap_bounce = False
        self.closest_bounces = 0
        self.chain_bounces = 0
        self.list_bounces = 0
        self.followed_at_bounce = -1
        self.view_bounce = False
        self.bounce_step = 0
        self.units_bounces = 0
        self.units_note = ""
        self.units_held = False

    def limits_text(self, final: bool) -> str:
        w = self.wire
        lines = ["visit status: open (the store has not ended this visit; a visit ends only when a call result is marked finished by the platform)",
                 "store steps used %d of %d (%d left; the smaller of the platform's limit and any limit the request or store states)" % (w.step, w.max_steps, w.steps_left()),
                 self.ending_text(),
                 "time left about %d seconds" % max(0, int(w.time_left())),
                 "model budget left for this visit: about %d more turns of yours (every turn uses it, whatever number of actions it holds; checks and reviews use it too). When it runs out the visit must end at once, so this is often the limit that binds first" % w.turns_left()]
        if self.bounced:
            lines.append(self.bounced)
            figure = self.figure_after_bounce()
            if figure:
                lines.append(figure)
        if self.units_note:
            lines.append(self.units_note)
        if final:
            pend = self.unopened_views()
            reads = ("; listings on a reference chain or in the basket not opened through every single-listing view: %s" % ", ".join("%s (%s)" % (rid, v) for rid, v in pend[:6])) if pend else ""
            lines.append("FINAL TURN: a limit of the run is about to run out, so the visit must end now, before your search is finished. Running out is not evidence that nothing fits. Ending without a purchase is right only when the data you read shows every candidate of a requested kind failing a requirement (another value, a price over the ceiling, sold out); a listing whose data does not state a point is not contradicted, and \"no listing read shows it\" is not \"nothing fits\". So for each product not yet in the basket, put the best candidate in, at the quantity asked for, one line per product (a store category the request names in full is one product), and order. The best candidate is the cheapest of its kind whose stored data holds every wanted value as WANTED WORDS IN THE STORED DATA lists them, read or not; only when no listing seen holds them all, the cheapest whose data shows no contradiction. A basket listing whose data does not show a wanted value is swapped, in this turn, for a listing seen that holds it. When the steps left allow it, first open, in this same turn, the single-listing views not yet opened of each listing you buy and of each listing on a reference chain%s: an order of a listing never opened through the page or table view fails even when it fits. Give at most %d actions, the last of which ends the visit." % (reads, self.final_allowance()))
            if self.closest_rule():
                lines.append("FINAL TURN UNDER THIS REQUEST (see IF NOTHING FITS): buy the full fits you found. Only for a product with no full fit, end without a purchase instead, and put forward in the ending text only the closest listing's id for it, naming no other listing.")
        return "\n".join(lines)

    def unopened_views(self, basket_only: bool = False) -> list[tuple[str, str]]:
        m = self.memory
        ids: list[str] = []
        if not basket_only:
            for hops in m.references.values():
                for hop in hops:
                    if hop.get("id") in m.records:
                        ids.append(hop["id"])
        ids.extend(i for i in list(m.basket_ids) + sorted(m.written) if i in m.records)
        out = []
        for rid in dict.fromkeys(ids):
            opened, missing = m.views_of(rid)
            views = [v for v in missing if v in m.rich] if opened else sorted(m.rich)
            if rid not in m.basket_ids and rid not in m.written:
                views.sort(key=lambda v: v not in m.text_views)
            for v in views:
                if m.view_args.get(v):
                    out.append((rid, v))
        return out

    def figure_after_bounce(self) -> str:
        m = self.memory
        if not self.bounce_step or not m.basket_lines:
            return ""
        figures = []
        for t in m.texts:
            mt = re.match(r"\[step (\d+)\] [^:]*: (.*)$", t)
            if not mt or int(mt.group(1)) < self.bounce_step:
                continue
            for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", mt.group(2)):
                v = price_num(raw)
                if v is not None and v >= 10:
                    figures.append((int(mt.group(1)), v))
        if not figures:
            return ""
        lines, total, breaches = m.basket_price_rows()
        if total is None:
            return ""
        over = [(st, v) for st, v in figures if total > v]
        if not over:
            return ""
        return "after that ending call the shopper's text carried the figure %s (step %d); the basket's lines add up to %s, above it: if that figure is a new ceiling for the whole order, the basket must change before you end again" % (fmt_price(over[0][1]), over[0][0], fmt_price(total))

    def closest_rule(self) -> str:
        f = self.memory.fallback if isinstance(self.memory.fallback, dict) else {}
        if f.get("do") != "name_closest":
            return ""
        band = f.get("band_percent")
        within = (", or one costing up to %s%% more than the cheapest of them" % fmt_price(band)) if isinstance(band, (int, float)) and not isinstance(band, bool) and band > 0 else ""
        return "The request asks, should no listing fit, to say so and put forward the listing that falls short on the fewest requirements. A store under such a request almost always holds a listing that fits in full for every product, so a plan to end without a purchase means the search has not found that fit yet: search on (each kind by its own category words, every result page, the cheapest listings read in full) and buy the full fits. Never buy a listing that misses a requirement. Only when the search really shows no listing of a product's kind meeting every requirement is ending without a purchase right. Its text must then put forward, by listing id, only the closest listing for that product: the one whose own data fails the fewest requirements (being out of stock and a price above a limit each count as a failed requirement; a point its data does not state is not a failure), the cheapest of those%s. It must name no other listing id." % within

    def ending_needs(self) -> tuple[int, int]:
        m = self.memory
        products = max(1, len(m.candidates))
        held = len(set(m.basket_ids) | m.written)
        missing = products if not held else max(0, products - held)
        return missing, missing + 1

    def final_allowance(self) -> int:
        missing, need = self.ending_needs()
        reads = len(self.unopened_views()) + (len(self.memory.rich) if missing else 0)
        return min(MAX_ACTIONS, max(3, need + 1 + reads))

    def ending_text(self) -> str:
        missing, need = self.ending_needs()
        left = self.wire.steps_left()
        extra = ("; %d listing(s) written to the basket since its last view count as held: %s" % (len(self.memory.written), ", ".join(sorted(self.memory.written)))) if self.memory.written else ""
        return "ending with a purchase needs at least %d more store steps (%d basket write(s) for the product(s) not yet in the basket, then the ending call), so at most %d of the %d steps left can go to searches and reads; a store can answer the first ending call with changes instead of ending, which costs one more%s" % (need, missing, max(0, left - need), left, extra)

    def fixed_part(self) -> str:
        parts = ["SHOPPER'S REQUEST:\n" + self.request]
        if self.opening:
            parts.append("OPENING FROM THE STORE OR SHOPPER:\n" + clip(dumps(self.opening) if not isinstance(self.opening, str) else self.opening, 1500))
        parts.append("STORE TOOLS:\n" + self.tools.text())
        return "\n\n".join(parts)

    def prompt(self, final: bool) -> Any:
        parts = ["LIMITS GIVEN BY THE RUN:\n" + self.limits_text(final)]
        if self.memory.texts:
            parts.append("TEXT FROM THE SHOPPER OR THE STORE SO FAR (every sentence that came back outside listing data, verbatim, oldest first):\n" + "\n".join(self.memory.texts))
        parts.append("YOUR NOTES FROM LAST TURN:\n" + (self.memory.notes or "(none yet: this is your first turn)"))
        if self.memory.msgplan:
            parts.append("YOUR MESSAGE PLAN FROM LAST TURN:\n" + self.memory.msgplan)
        basket = self.basket_text(1000, True)
        if basket:
            parts.append("BASKET AS THE STORE LAST SHOWED IT (check its lines and units against the request before you end; calls after this view may have changed it):\n" + basket)
        writes = self.memory.writes_text()
        if writes:
            parts.append("BASKET WRITES SO FAR (each basket-changing call, then the figures the store showed for that listing right after it; an argument whose value comes back as the line's units is the one that sets units, whatever its name):\n" + writes)
        cats = self.memory.category_lines(self.request)
        if cats:
            parts.append("STORE CATEGORIES WHOSE EVERY WORD THE REQUEST USES (from the listing data seen; when the request asks for such a category as a product, it is one product and one basket line, whatever items the category's name lists):\n" + cats)
        ranked = self.memory.ranked_text() if self.memory.order_text() else ""
        if ranked:
            parts.append("ROWS EACH RECENT SEARCH RANKED FIRST, NOT OPENED YET (many stores show first the listings that carry most of the search's words, also words only their properties hold; open these before other rows of that search):\n" + clip(ranked, 1400))
        order = self.memory.order_text()
        if order:
            parts.append("RESULT ORDER IN THIS STORE (measured from its lists so far):\n" + order)
        closest = self.closest_rule()
        if closest:
            parts.append("IF NOTHING FITS (as you recorded it from the request):\n" + closest)
        rules = self.memory.price_rule_text()
        if rules:
            parts.append("PRICE RULES YOU RECORDED (what the request wants lowest, and the price limits, as you recorded them; against the searches and the basket as the store showed them):\n" + clip(rules, 2600))
        if self.memory.records:
            parts.append("LISTINGS SEEN SO FAR (id | short fields | the single-listing views it was opened through, and those still missing):\n" + self.memory.listing_lines())
            variety = self.memory.variety_text()
            parts.append("PROPERTIES OF THE LISTINGS SEEN SO FAR, per kind of listing, most widely carried first (property tables only, in the store's own words: property, how many of the listings with a table carry it, whether values differ, most common values):\n" + (clip(variety, 3000) if variety else "(none yet: no listing data read so far holds a table of properties)"))
            cand = self.memory.candidate_text()
            if cand:
                parts.append("WHAT YOUR CURRENT CANDIDATES DIFFER ON (from the stored data of the listings you named as candidates last turn; property: values with counts):\n" + clip(cand, 2500))
            wanted = self.memory.wants_text()
            if wanted:
                parts.append("WANTED WORDS IN THE STORED DATA (per product, the value words you recorded in wants, looked for letter by letter in the stored data of the listings seen, in that order, spaces, hyphens and underscores alike; the field that holds them is named, a description does not count, a listing whose own value for the property under which other listings hold the word says something else does not hold it, and figures and bounds are not compared here):\n" + clip(wanted, 2200))
            refs = self.memory.reference_text()
            if refs:
                parts.append("REFERENCE CHAINS YOU RECORDED (for each requirement given by reference, the listings of its chain in order; for each, the words that describe it and whether its own data shows each of them, checked against its stored data (UNSAID when the data does not state it), then the single-listing views it has been opened through and those not yet opened; last, the texts on the last listing's opened views):\n" + clip(refs, 3600))
            basket = self.memory.basket_coverage()
            if basket:
                parts.append("LISTINGS IN THE BASKET AS THE STORE LAST SHOWED IT, with the single-listing views each has been opened through and those not yet opened:\n" + clip(basket, 1200))
            answered = self.memory.answer_text()
            if answered:
                parts.append("SHOPPER'S ANSWERS AGAINST THE BASKET (each message sent, the reply verbatim, then each basket listing of the kind it names with its own value for the property it names):\n" + clip(answered, 2400))
            cheap = self.memory.cheapest_text()
            if cheap:
                parts.append("CHEAPEST LISTINGS SEEN OF EACH KIND (by the price field you named, lowest first; UNREAD = seen only in a list, its own data not opened yet):\n" + clip(cheap, 2200))
        if self.memory.searches:
            parts.append("SEARCHES SO FAR (per search: rows shown and found, the pages fetched by reusing its continuation value, rows not yet shown, whether the store offers a further page, rows the same as an earlier search's, rows rising in price, and the store's reply when nothing came back; the largest count this store ever returns is marked as its ceiling):\n" + clip(self.memory.search_text(), 4600))
            parts.append("MESSAGES TO THE SHOPPER (allowed as your plan read it from the request; sent as counted from your calls; your hints; then, per kind of listing, the properties its property tables carry that no message has named yet, the hinted one first, then those every table read carries, most widely carried first, with how many of the tables read carry each and its most common values; spellings of one property are joined with \"/\"):\n" + clip(self.memory.ask_text(self.request, self.wire.steps_left()), 2800))
        now_text = "CURRENT STATE (rewritten every turn):\n\n" + "\n\n".join(parts)
        if not self.memory.entries:
            return now_text
        hist = self.memory.history_blocks()
        out = [{"text": hist[0] + (hist[1] if len(hist) > 1 else "")}] + [{"text": b} for b in hist[2:]]
        out[-1]["cache"] = True
        out.append({"text": "\n" + now_text})
        return out

    def final_due(self) -> bool:
        return self.wire.steps_left() <= FINAL_STEPS or self.wire.time_left() < END_RESERVE or self.wire.over_budget()

    def verify_chains(self) -> None:
        m = self.memory
        jobs = []
        for hops in m.references.values():
            for hop in hops:
                rid, desc = hop.get("id"), [d for d in (hop.get("described_as") or []) if d]
                if not rid or rid not in m.records or not desc or not m.views_of(rid)[0]:
                    continue
                sig = m.verdict_key(rid, desc)
                if sig in m.verdicts or any(j[0] == sig for j in jobs):
                    continue
                jobs.append((sig, rid, desc))
        if not jobs or self.wire.time_left() < END_RESERVE + 15:
            return
        allowed = []
        spend = self.wire.spend
        for job in jobs[:8]:
            if spend + COST_GUESS["judge"] > self.wire.limit - FINAL_RESERVE:
                break
            spend += COST_GUESS["judge"]
            allowed.append(job)
        if not allowed:
            return
        with ThreadPoolExecutor(max_workers=min(JUDGE_WORKERS, len(allowed))) as pool:
            futures = [(sig, rid, pool.submit(self.judge, desc, [rid])) for sig, rid, desc in allowed]
            for sig, rid, fut in futures:
                try:
                    rows = fut.result()
                except Exception as err:
                    self.log.add("judge_error", error=clip(str(err), 200))
                    continue
                row = rows.get(rid) if isinstance(rows, dict) else None
                m.verdicts[sig] = {k: list(v) for k, v in (row or {}).items() if isinstance(v, list)}
        self.log.add("chain_check", checked=[{"id": rid, "described_as": desc, "verdict": m.verdicts.get(sig)} for sig, rid, desc in allowed])

    def choose(self) -> None:
        m = self.memory
        if self.chooser_calls >= CHOOSER_CALLS or not m.message_tool or not m.messages_left() or self.wire.time_left() < END_RESERVE + 60 or not self.wire.affords("chooser"):
            return
        if "this turn's actions should hold these" not in m.ask_text(self.request, self.wire.steps_left()):
            return
        key, user, ready, left = m.choose_input(self.request)
        if not ready or key == m.chosen_key:
            return
        self.chooser_calls += 1
        out = self.wire.ask("chooser", CHOOSE_SYSTEM.replace("{n}", str(left)).replace("{{", "{").replace("}}", "}"), user, 2000, keys=("ask",))
        picks = m.take_choice(out, ready)
        self.log.add("chooser", picks=["%s: %s" % x for x in picks])
        if picks:
            m.chosen = picks
            m.chosen_key = key

    def direct(self, final: bool) -> list[dict[str, Any]]:
        if not final:
            self.choose()
        if not self.fixed_text:
            self.fixed_text = DIRECTOR_SYSTEM + "\n\nTHIS VISIT (fixed for the whole visit):\n\n" + self.fixed_part()
        self.verify_chains()
        out = self.wire.ask("director", self.fixed_text, self.prompt(final), 3200, keys=("actions", "notes"))
        self.turn += 1
        self.units_note = ""
        if not isinstance(out, dict):
            self.failed_turns += 1
            self.log.add("director_failed", turn=self.turn)
            return []
        self.failed_turns = 0
        notes = out.get("notes")
        if isinstance(notes, (str, dict, list)) and notes:
            self.memory.notes = clip(notes if isinstance(notes, str) else dumps(notes), NOTES_CHARS)
        fb = out.get("if_nothing_fits")
        if isinstance(fb, dict) and fb.get("do") in ("name_closest", "decline", "unstated"):
            band = fb.get("band_percent")
            self.memory.fallback = {"do": fb["do"], "band_percent": band if isinstance(band, (int, float)) and not isinstance(band, bool) else None}
        if isinstance(out.get("ending"), dict):
            self.memory.ending = dict(out["ending"])
            if isinstance(out.get("basket_view"), str):
                self.memory.ending["basket_view"] = out["basket_view"]
        lim = out.get("step_limit")
        if isinstance(lim, str) and lim.strip().isdigit():
            lim = int(lim.strip())
        if isinstance(lim, int) and not isinstance(lim, bool) and MIN_STEP_LIMIT <= lim < self.wire.max_steps:
            self.log.add("step_limit", platform=self.wire.max_steps, stated=lim)
            self.wire.set_steps(lim)
        if isinstance(out.get("candidates"), dict):
            self.memory.candidates = {str(k): [str(i) for i in v if isinstance(i, (str, int))][:20] for k, v in out["candidates"].items() if isinstance(v, list)}
        if isinstance(out.get("wants"), dict):
            self.memory.wants = {clip(str(k), 60): [clip(re.sub(r"\s+", " ", str(w)).strip(), 60) for w in v if isinstance(w, (str, int, float)) and str(w).strip()][:8] for k, v in list(out["wants"].items())[:8] if isinstance(v, list)}
        if isinstance(out.get("references"), dict):
            refs: dict[str, list[dict[str, Any]]] = {}
            for k, v in list(out["references"].items())[:8]:
                if isinstance(v, dict):
                    v = v.get("chain") if isinstance(v.get("chain"), list) else [v]
                if not isinstance(v, list):
                    continue
                hops = []
                for h in v[:12]:
                    if isinstance(h, (str, int)) and not isinstance(h, bool):
                        rid, desc = str(h).strip(), []
                    elif isinstance(h, dict):
                        rid = str(h.get("id")).strip() if isinstance(h.get("id"), (str, int)) and not isinstance(h.get("id"), bool) else ""
                        raw = h.get("described_as")
                        raw = [raw] if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
                        desc = [clip(re.sub(r"\s+", " ", str(d)).strip(), 80) for d in raw if isinstance(d, (str, int, float)) and str(d).strip()][:8]
                    else:
                        continue
                    if rid and rid not in self.memory.records and " " in rid:
                        rid = ""
                    if rid or desc:
                        hops.append({"id": rid or None, "described_as": desc})
                refs[clip(str(k), 90)] = hops
            if refs or not self.memory.references:
                merged = dict(self.memory.references)
                for k, hops in refs.items():
                    prev = merged.get(k) or []
                    pids = [h.get("id") for h in prev]
                    nids = [h.get("id") for h in hops]
                    if len(prev) > len(hops) and all(a == b or b is None for a, b in zip(pids, nids)):
                        continue
                    merged[k] = hops
                self.memory.references = dict(list(merged.items())[-8:])
        if isinstance(out.get("messages"), dict):
            self.memory.msgplan = clip(dumps(out["messages"]), 600)
        if isinstance(out.get("hints"), dict):
            self.memory.hints = {clip(str(k), 60): clip(v.strip(), 60) for k, v in list(out["hints"].items())[:8] if isinstance(v, str) and v.strip()}
        if isinstance(out.get("price_field"), str) and out["price_field"].strip():
            self.memory.price_field = out["price_field"].strip()
        if isinstance(out.get("messages"), dict):
            mp = out["messages"]
            self.memory.msgplan = clip(dumps(mp), 600)
            allowed = mp.get("allowed")
            if isinstance(allowed, str) and allowed.strip().isdigit():
                allowed = int(allowed.strip())
            if isinstance(allowed, int) and not isinstance(allowed, bool) and 0 <= allowed <= 50:
                self.memory.allowed = allowed
            if isinstance(mp.get("sent"), int) and not isinstance(mp.get("sent"), bool):
                self.memory.sent_hint = max(0, mp["sent"])
            self.memory.asked_hint = [str(a) for a in (mp.get("asked") or []) if isinstance(a, str)][:20]
        if isinstance(out.get("message_tool"), str) and out["message_tool"] in self.tools.items:
            self.memory.message_tool = out["message_tool"]
        if isinstance(out.get("lowest_price"), list):
            self.memory.lowest_price = [clip(p.strip(), 60) for p in out["lowest_price"] if isinstance(p, str) and p.strip()][:8]
        if isinstance(out.get("units"), dict):
            self.memory.units = {clip(str(k), 60): int(v) for k, v in list(out["units"].items())[:8] if isinstance(v, (int, float)) and not isinstance(v, bool) and 1 <= v <= 999 and float(v).is_integer()}
        if isinstance(out.get("limits"), dict):
            lim = out["limits"]
            per = lim.get("per_unit") if isinstance(lim.get("per_unit"), dict) else {}
            self.memory.limits = {"order_total": price_num(lim.get("order_total")), "per_unit": {clip(str(k), 60): price_num(v) for k, v in list(per.items())[:8] if price_num(v) is not None}}
        acts = [a for a in (out.get("actions") or []) if isinstance(a, dict) and isinstance(a.get("tool"), str)]
        if not acts:
            self.empty_turns += 1
            self.record("none", {}, dumps({"no_actions": "your last turn gave no actions, but the visit is still open: the store has not ended it. Give the next actions, or end the visit."}), "turn %d" % self.turn)
        else:
            self.empty_turns = 0
        self.log.add("turn", n=self.turn, final=final, notes=clip(self.memory.notes, 1200), msgplan=self.memory.msgplan, actions=[{"tool": a.get("tool"), "args": a.get("args"), "why": a.get("why")} for a in acts])
        return acts[: (self.final_allowance() if final else MAX_ACTIONS)]

    def record(self, tool: str, args: Any, view: str, label: str) -> None:
        self.memory.entries.append({"tool": tool, "args": args, "view": view, "label": label})

    def helper(self, tool: str, args: dict[str, Any]) -> None:
        if tool == "check":
            self.checks([args])
            return
        ids = [i for i in (args.get("ids") or []) if isinstance(i, str)]
        known = [i for i in ids if i in self.memory.records]
        unknown = [i for i in ids if i not in self.memory.records]
        view = {i: shrink(self.memory.records[i], 700, 30) for i in known[:4]}
        if unknown:
            view["not_seen_ids"] = unknown
        self.record(tool, args, dumps(view, 4500), "helper")

    def judge(self, reqs: list[str], batch: list[str]) -> dict[str, Any]:
        user = "Requirements:\n%s\n\nListings:\n%s" % ("\n".join("%d. %s" % (i, r) for i, r in enumerate(reqs)),
                                                      "\n\n".join("[%s]\n%s" % (rid, clip(dumps(self.memory.records[rid]), 3500)) for rid in batch))
        out = self.wire.ask("judge", CHECK_SYSTEM, user, 1800) or {}
        rows: dict[str, Any] = {}
        for rid in batch:
            info = out.get(rid) if isinstance(out.get(rid), dict) else {}
            hay = letters(dumps(self.memory.records[rid]))
            row = {}
            for i, r in enumerate(reqs):
                v = info.get(str(i)) if isinstance(info.get(str(i)), dict) else {}
                ans = str(v.get("answer") or "unsaid")
                quote = str(v.get("quote") or "")
                if ans in ("met", "not_met") and (not letters(quote) or letters(quote) not in hay):
                    ans, quote = "unsaid", "the quoted fragment is not in the data, so this verdict does not count" if quote else ""
                row[r] = [ans, clip(quote, 160)]
            row["data seen via"] = ", ".join(sorted(self.memory.sources.get(rid) or []))
            rows[rid] = row
        return rows

    def checks(self, group: list[dict[str, Any]]) -> None:
        plans = []
        jobs = []
        for args in group:
            ids = [i for i in (args.get("ids") or []) if isinstance(i, str)]
            known = [i for i in ids if i in self.memory.records]
            unknown = [i for i in ids if i not in self.memory.records]
            reqs = [str(r) for r in (args.get("requirements") or []) if str(r).strip()][:10]
            plan = {"args": args, "known": known[:6], "unknown": unknown, "reqs": reqs, "problem": None}
            if not known or not reqs:
                plan["problem"] = {"problem": "give listing ids seen in results and at least one requirement", "not_seen_ids": unknown}
            else:
                for start in range(0, len(plan["known"]), 3):
                    jobs.append((len(plans), plan["known"][start : start + 3]))
            plans.append(plan)
        allowed = 0
        spend = self.wire.spend
        for _ in jobs:
            if spend + COST_GUESS["judge"] > self.wire.limit - FINAL_RESERVE:
                break
            spend += COST_GUESS["judge"]
            allowed += 1
        results: dict[int, dict[str, Any]] = {}
        if allowed:
            with ThreadPoolExecutor(max_workers=min(JUDGE_WORKERS, allowed)) as pool:
                futures = [(k, pool.submit(self.judge, plans[k]["reqs"], batch)) for k, batch in jobs[:allowed]]
                for k, fut in futures:
                    try:
                        results.setdefault(k, {}).update(fut.result())
                    except Exception as err:
                        self.log.add("judge_error", error=clip(str(err), 200))
        for k, plan in enumerate(plans):
            if plan["problem"] is not None:
                self.record("check", plan["args"], dumps(plan["problem"]), "helper")
                continue
            result = dict(results.get(k) or {})
            if not result:
                result = {"problem": "the model budget for checks is used up; decide from the data you have, using view if needed"}
            elif len(result) < len(plan["known"]):
                result["not_checked"] = [i for i in plan["known"] if i not in result]
            if plan["unknown"]:
                result["not_seen_ids"] = plan["unknown"]
            self.record("check", plan["args"], dumps(result, 4500), "helper")

    def execute(self, action: dict[str, Any]) -> bool:
        tool = action.get("tool")
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        if tool in ("check", "view") and tool not in self.tools.items:
            self.helper(tool, args)
            return True
        args = self.tools.fit_lengths(tool, args)
        problem = self.tools.check(tool, args)
        if problem:
            self.record(tool, args, dumps({"not_sent": problem}), "not sent")
            return False
        m = self.memory
        swapped = ""
        if tool == m.message_tool and m.messages_left() and not self.is_ending(action):
            new = m.plan_swap(args)
            if new is not None:
                swapped = " ".join(str(v) for v in args.values() if isinstance(v, str))
                args = new
                self.log.add("plan_swap", was=swapped, sent=new)
        sig = tool + "|" + json.dumps(args, sort_keys=True, default=str)
        m.repeats = m.repeats + 1 if sig == m.last_sig else 1
        m.last_sig = sig
        if m.repeats > 2:
            self.record(tool, args, dumps({"not_sent": "this exact call was just made twice in a row; its results are above"}), "not sent")
            return False
        res = self.wire.act(tool, args)
        if res.get("ok"):
            m.step = self.wire.step
            self.memory.absorb(res["obs"], tool, strings_in(args))
            for k, v in args.items():
                if isinstance(v, str) and v in m.by_id.get(tool, set()):
                    m.view_args[tool] = k
            self.memory.note_search(tool, args, res["obs"], self.wire.step)
            m.note_write(tool, args, self.wire.step)
            m.note_write_reply(res["obs"], self.wire.step)
            self.memory.collect_texts(res["obs"], tool, self.wire.step)
            if isinstance(m.ending, dict) and tool == m.ending.get("basket_view"):
                m.note_basket(res["obs"])
            if tool == m.message_tool:
                m.sent_texts.append(" ".join(str(v) for v in args.values() if isinstance(v, str)))
                head = "[step %d] %s: " % (self.wire.step, tool)
                m.answers.append((m.sent_texts[-1], " ".join(t[len(head):] for t in m.texts if t.startswith(head))))
            _, rows = m.find_rows(res["obs"])
            view = dumps(shrink(res["obs"]), 2600) if rows else dumps(shrink(res["obs"], RECORD_TEXT_WIDTH, 30), RECORD_VIEW_CHARS)
        else:
            view = dumps({"no_result": res.get("note")})
        self.record(tool, args, view, "step %d" % self.wire.step)
        if swapped:
            m.entries[-1]["label"] += " (sent by the code in place of your question \"%s\", which names no question of the plan in MESSAGES TO THE SHOPPER: the plan's next question was sent instead)" % clip(swapped, 120)
        if res.get("finished"):
            self.finished = True
        return bool(res.get("ok"))

    def is_ending(self, action: dict[str, Any]) -> bool:
        end = self.memory.ending if isinstance(self.memory.ending, dict) else {}
        names = {(end.get(k) or {}).get("tool") for k in ("order", "decline") if isinstance(end.get(k), dict)}
        return isinstance(action.get("tool"), str) and action.get("tool") in names

    def review_prompt(self, action: dict[str, Any]) -> str:
        m = self.memory
        recent = m.entries[-5:]
        lines = ["[%s] %s %s -> %s" % (e["label"], e["tool"], dumps(e["args"], 200), clip(e["view"], 450)) for e in recent]
        tools = "; ".join("%s(%s)" % (t["name"], ", ".join(((t["parameters"] or {}).get("properties") or {}).keys())) for t in self.tools.items.values())
        index = "\n".join(m.listing_lines().splitlines()[-12:])
        parts = ["SHOPPER'S REQUEST:\n" + self.request,
                 "STORE TOOLS (names and arguments; the assistant has their full descriptions):\n" + tools,
                 "LIMITS GIVEN BY THE RUN:\n" + self.limits_text(False),
                 "TEXT FROM THE SHOPPER OR THE STORE SO FAR (verbatim, oldest first):\n" + ("\n".join(m.texts) or "(none)"),
                 "ASSISTANT'S NOTES:\n" + (m.notes or "(none)"),
                 "ASSISTANT'S MESSAGE PLAN (messages allowed, sent, properties asked, next one):\n" + (m.msgplan or "(none)"),
                 "MESSAGES TO THE SHOPPER (allowed, sent, left; the assistant's hints; then per kind of listing the properties its tables carry that no message has named yet, the hinted one first, then those every table carries):\n" + clip(m.ask_text(self.request, self.wire.steps_left()), 2000),
                 "LAST ACTIONS AND RESULTS (%d actions in all so far):\n%s" % (len(m.entries), "\n".join(lines)),
                 "MOST RECENT LISTINGS SEEN (id | short fields):\n" + index,
                 "FULL STORED DATA OF THE LISTINGS THE ASSISTANT NAMES AS ITS CHOICES OR CANDIDATES:\n" + self.chosen_data(action),
                 "WANTED WORDS IN THE STORED DATA (the value words the assistant recorded per product, looked for in the stored data of the listings seen; NOT IN ITS STORED DATA means the listing does not show that value; NOT ITS VALUE means its own value for the property that holds that word elsewhere says something else, so it does not show it either):\n" + (clip(m.wants_text(), 1800) or "(no wanted values recorded)"),
                 "BASKET AS THE STORE LAST SHOWED IT:\n" + self.basket_text(),
                 "PRICE RULES THE ASSISTANT RECORDED (what the request wants lowest, with the searches of those kinds never continued; the price limits, with the basket's lines and total against them):\n" + (clip(m.price_rule_text(), 2000) or "(none recorded: the assistant read no lowest-price wish and no price limit in the request)"),
                 "IF NOTHING FITS (as the assistant recorded it from the request):\n" + (self.closest_rule() or "(no instruction recorded for when nothing fits)"),
                 "LISTINGS IN THE BASKET AS LAST SHOWN, with the single-listing views of this store each has been opened through and those not yet opened:\n" + (m.basket_coverage() or "(no basket view with listings seen yet)"),
                 "SHOPPER'S ANSWERS AGAINST THE BASKET (each message sent, the reply verbatim, then each basket listing of the kind it names with its own value for the property it names):\n" + (clip(m.answer_text(), 2400) or "(no answers, or no basket view with listings seen yet)"),
                 "BASKET WRITES SO FAR (each basket-changing call and the figures the store showed for that listing after it) AND STORE CATEGORIES WHOSE EVERY WORD THE REQUEST USES (one such category is one product and one line):\n" + ((m.writes_text() + "\n" + m.category_lines(self.request)).strip() or "(none)"),
                 "REFERENCE CHAINS THE ASSISTANT RECORDED (listings of each chain in order; for each, the words that describe it, whether its own data shows each of them as checked against its stored data, the views opened and not yet opened; then the texts on the last listing's opened views):\n" + (clip(m.reference_text(), 3600) or "(none recorded; check the request and the shopper's answers for a requirement given by reference to another listing's page)"),
                 "CHEAPEST LISTINGS SEEN OF EACH KIND (lowest first; UNREAD = its own data was never opened):\n" + (clip(m.cheapest_text(), 1500) or "(no price field named yet)"),
                 "SEARCHES SO FAR (rows shown, the result's count and continuation fields, whether the list was continued):\n" + (clip("\n".join(m.search_text().splitlines()[-14:]), 2200) or "(none)"),
                 "NAMED LISTINGS WHOSE PROPERTY TABLE HAS NOT BEEN READ (the data stored for them holds no table of properties):\n" + self.unread_named(action),
                 "PROPOSED ENDING:\n%s %s\nreason given: %s" % (action.get("tool"), dumps(action.get("args"), 600), clip(action.get("why") or "", 300))]
        return "\n\n".join(parts)

    def chosen_data(self, action: dict[str, Any]) -> str:
        m = self.memory
        named: list[str] = []
        for ids in m.candidates.values():
            named.extend(i for i in ids if i in m.records)
        text = dumps(action.get("args")) + " " + str(action.get("why") or "") + " " + m.notes
        named.extend(rid for rid in m.records if rid in text)
        seen: list[str] = []
        for rid in named:
            if rid not in seen:
                seen.append(rid)
        if not seen:
            return "(none named)"
        return "\n".join("[%s] %s | seen via: %s" % (rid, clip(dumps(m.records[rid]), 1400), ", ".join(sorted(m.sources.get(rid) or []))) for rid in seen[:5])

    def basket_text(self, width: int = 1800, quiet: bool = False) -> str:
        tool = self.memory.ending.get("basket_view") if isinstance(self.memory.ending, dict) else None
        if not isinstance(tool, str) or not tool:
            return "" if quiet else "(the assistant has not named a tool that shows the basket)"
        last = None
        later = 0
        for e in self.memory.entries:
            if e["tool"] == tool and str(e["label"]).startswith("step"):
                last = e
                later = 0
            elif str(e["label"]).startswith("step"):
                later += 1
        if last is None:
            return "" if quiet else "(%s has not been called yet)" % tool
        return "%s at %s (%d store calls since): %s" % (tool, last["label"], later, clip(last["view"], width))

    def unread_named(self, action: dict[str, Any]) -> str:
        m = self.memory
        text = dumps(action.get("args")) + " " + str(action.get("why") or "") + " " + m.notes
        named = [rid for rid in m.records if rid in text]
        for ids in m.candidates.values():
            named.extend(i for i in ids[:3] if i in m.records)
        out = []
        for rid in named:
            if rid not in out and not m.tables_of(rid):
                out.append(rid)
        return ", ".join(out[:12]) if out else "(none)"

    def is_decline(self, action: dict[str, Any]) -> bool:
        end = self.memory.ending if isinstance(self.memory.ending, dict) else {}
        d = end.get("decline") if isinstance(end.get("decline"), dict) else {}
        o = end.get("order") if isinstance(end.get("order"), dict) else {}
        if action.get("tool") != d.get("tool"):
            return False
        if o.get("tool") != d.get("tool"):
            return True
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        dargs = d.get("args") if isinstance(d.get("args"), dict) else {}
        oargs = o.get("args") if isinstance(o.get("args"), dict) else {}
        same = lambda a, b: json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
        if same(args, oargs) and not same(args, dargs):
            return False
        if same(args, dargs) and not same(args, oargs):
            return True
        return any(isinstance(v, str) and isinstance(dargs.get(k), str) and v == dargs.get(k) != oargs.get(k) for k, v in args.items())

    def review(self, action: dict[str, Any]) -> list[dict[str, Any]] | None:
        m = self.memory
        left = m.messages_left()
        if left and m.message_tool and not self.asked_bounce and self.wire.steps_left() > left + 2 and self.wire.time_left() > END_RESERVE + 25:
            rows, sizes, bare = m.ask_rows(self.request)
            open_count = sum(1 for g in rows for it in rows[g] if not it[4])
            if open_count:
                self.asked_bounce = True
                reason = "the shopper allows %d more message(s) and the property tables read show %d properties no message has named yet (see MESSAGES TO THE SHOPPER); each may hide a requirement the order misses: ask about them first, one property per message, then end" % (left, open_count)
                self.log.add("review", approve=False, reason=reason)
                self.record("review", {"proposed": action.get("tool")}, dumps({"not_yet": reason}), "reviewer")
                return []
        named_ids = [rid for rid in m.records if rid and rid in dumps(action.get("args"), 4000)]
        if self.is_decline(action) and self.closest_rule() and ((self.closest_bounces < 1 and self.wire.steps_left() > 8 and self.wire.time_left() > END_RESERVE + 40) or (not named_ids and self.closest_bounces < 2 and self.wire.steps_left() > 1)):
            self.closest_bounces += 1
            if named_ids or self.closest_bounces == 1:
                reason = "under this request (see IF NOTHING FITS) the store almost always holds a full fit for every product, so this decline means the search has not found it yet: for each product without a full fit, search its kind by its own category words, continue every result page, and read the cheapest listings in full; buy the full fits you find, never a listing that misses a requirement, and decline only if the kind really holds none, putting forward only the closest listing's id"
            else:
                reason = "under this request (see IF NOTHING FITS) a decline must put forward, by listing id, the closest listing for each product without a full fit (the fewest failed requirements, the cheapest of those) and name no other listing; this ending names none: give the ending again with those ids in its text"
            self.log.add("review", approve=False, reason=reason)
            self.record("review", {"proposed": action.get("tool")}, dumps({"not_yet": reason}), "reviewer")
            return []
        if self.rejections >= REVIEW_REJECTS:
            self.log.add("review", approve=True, reason="earlier reviews already sent the assistant back %d times; this ending stands" % self.rejections)
            return None
        if not self.is_decline(action) and self.wire.steps_left() > 3 and self.wire.time_left() > END_RESERVE + 15:
            lines, total, breaches = m.basket_price_rows()
            if breaches and not self.cap_bounce:
                self.cap_bounce = True
                reason = "the basket as the store last showed it breaks a price limit you recorded (see PRICE RULES): %s. Either the limit is misread (then correct it in limits) or the basket is wrong: replace the line that breaks it with a fitting listing within the limit, then end" % "; ".join(breaches)
                self.log.add("review", approve=False, reason=reason)
                self.record("review", {"proposed": action.get("tool")}, dumps({"not_yet": reason}), "reviewer")
                return []
            self.verify_chains()
            gaps = m.chain_gaps()
            if gaps and self.chain_bounces < 2 and self.wire.steps_left() > 6 + sum(len(g["desc"]) for g in gaps if g["kind"] == "unopened") and self.wire.time_left() > END_RESERVE + 30:
                self.chain_bounces += 1
                reasons = []
                acts = []
                for g in gaps[:6]:
                    if g["kind"] == "not_found":
                        reasons.append("the chain for %s has no listing found yet%s: search the seller's name with the most telling describing words (the name alone when that finds none of the seller's rows), continue that list when it announced more rows than it showed, and open the page views of that seller's listings of the named kind, the rows ranked first first" % (g["key"], (" for the one described as " + ", ".join(g["desc"])) if g["desc"] else ""))
                    elif g["kind"] == "unsaid":
                        reasons.append("%s on the chain for %s is not shown to be the listing described: its data, opened through every view, does not state %s (UNSAID in REFERENCE CHAINS), while the listing meant shows every describing property; open that seller's other listings of the kind through the page view and record the one whose data shows them all, or this one if no other does" % (g["id"], g["key"], ", ".join(g["desc"])))
                    elif g["kind"] == "not_met":
                        reasons.append("%s on the chain for %s does not show %s, which describes the listing to read (NOT MET in REFERENCE CHAINS): it is another listing of that seller, so a value or pointer read from it is wrong; open that seller's other listings of the kind through the page view and record the one whose data shows every describing property" % (g["id"], g["key"], ", ".join(g["desc"])))
                    else:
                        reasons.append("%s on the chain for %s has not been opened through %s: its notes may send you on, and a value counts only when read on the page" % (g["id"], g["key"], ", ".join(g["desc"])))
                        for v in g["desc"]:
                            arg = m.view_args.get(v)
                            if arg:
                                acts.append({"tool": v, "args": {arg: g["id"]}, "why": "open the chain listing through this view before the order"})
                reason = "a requirement given by reference is not settled yet: " + "; ".join(reasons)
                self.log.add("review", approve=False, reason=reason, actions=[{"tool": a.get("tool"), "args": a.get("args")} for a in acts])
                self.record("review", {"proposed": action.get("tool")}, dumps({"not_yet": clip(reason, 900)}), "reviewer")
                return acts[:MAX_ACTIONS]
            reads = self.unopened_views(True)
            if reads and not self.view_bounce and self.wire.steps_left() > 2 + len(reads) and self.wire.time_left() > END_RESERVE + 25:
                self.view_bounce = True
                acts = [{"tool": v, "args": {m.view_args[v]: rid}, "why": "open this basket listing through the view not yet opened before the order"} for rid, v in reads[:MAX_ACTIONS]]
                reason = "listings in the basket have not been opened through every single-listing view this store offers: %s. Their notes or property table may contradict the request, and an order of a listing never opened through a view fails: open them now, in the same turn as the ending call and before it" % ", ".join("%s through %s" % (rid, v) for rid, v in reads[:6])
                self.log.add("review", approve=False, reason=reason, actions=[{"tool": a.get("tool"), "args": a.get("args")} for a in acts])
                self.record("review", {"proposed": action.get("tool")}, dumps({"not_yet": reason}), "reviewer")
                return acts
            pend = m.pending_lists(m.lowest_kinds()) if m.lowest_price else []
            followed = sum(1 for s in m.searches if s["followed"])
            if pend and self.list_bounces < 2 and followed > self.followed_at_bounce and self.wire.steps_left() > 10 and self.wire.time_left() > END_RESERVE + 30:
                self.list_bounces += 1
                self.followed_at_bounce = followed
                named = "; ".join("step %s %s" % (s["step"], dumps(s["args"], 90)) for s in pend[-6:])
                reason = "you recorded that the shopper wants the lowest price for %s, but these searches showed that kind, announced more rows than they showed and were never continued: %s. A list in the store's own order can hold a cheaper fit on any page, so continue them now (several pages in one turn, keeping steps for the reads and the ending), read the cheapest fits they show, and only then end" % (", ".join(m.lowest_price), named)
                self.log.add("review", approve=False, reason=reason)
                self.record("review", {"proposed": action.get("tool")}, dumps({"not_yet": reason}), "reviewer")
                return []
        strong = self.is_decline(action) and self.wire.affords("reviewer")
        role = "reviewer" if strong else "reviewer_light" if self.wire.affords("reviewer_light") else None
        if role is None:
            self.log.add("review", approve=True, reason="model budget used up")
            return None
        out = self.wire.ask(role, REVIEW_SYSTEM, self.review_prompt(action), 1200 if role == "reviewer" else 2000)
        if not isinstance(out, dict) and self.wire.affords(role) and self.wire.time_left() > END_RESERVE + 20:
            out = self.wire.ask(role, REVIEW_SYSTEM, self.review_prompt(action), 1200 if role == "reviewer" else 2000)
        if not isinstance(out, dict) or out.get("approve") is not False:
            self.log.add("review", approve=True, reason=(out or {}).get("reason") if isinstance(out, dict) else None)
            return None
        self.rejections += 1
        acts = [a for a in (out.get("actions") or []) if isinstance(a, dict) and isinstance(a.get("tool"), str)][:MAX_ACTIONS]
        self.log.add("review", approve=False, reason=out.get("reason"), actions=[{"tool": a.get("tool"), "args": a.get("args")} for a in acts])
        self.record("review", {"proposed": action.get("tool")}, dumps({"not_yet": clip(out.get("reason") or "", 500)}), "reviewer")
        return acts

    def end_anyway(self) -> None:
        end = self.memory.ending if isinstance(self.memory.ending, dict) else {}
        intended = end.get("intended") if end.get("intended") in ("order", "decline") else "order"
        if intended == "order" and self.ending_needs()[0] > 0 and isinstance(end.get("decline"), dict):
            intended = "decline"
        for key in (intended, "order" if intended == "decline" else "decline"):
            choice = end.get(key)
            if not (isinstance(choice, dict) and isinstance(choice.get("tool"), str)):
                continue
            args = self.tools.fit_lengths(choice["tool"], choice.get("args") or {})
            if self.tools.check(choice["tool"], args):
                continue
            for _ in range(2):
                if self.finished or self.wire.done or self.wire.steps_left() <= 0:
                    return
                self.log.add("end_anyway", choice=choice)
                self.execute({"tool": choice["tool"], "args": args})
            other = end.get("order" if key == "decline" else "decline")
            if not self.finished and not self.wire.done and self.wire.steps_left() > 0 and isinstance(other, dict) and isinstance(other.get("tool"), str):
                oargs = self.tools.fit_lengths(other["tool"], other.get("args") or {})
                if not self.tools.check(other["tool"], oargs):
                    self.log.add("end_anyway", choice=other)
                    self.execute({"tool": other["tool"], "args": oargs})
            return

    def run(self) -> list[dict[str, Any]]:
        if self.finished or not self.tools.items:
            return self.log.items
        final_retry = 1
        bounce_retry = 1
        while not self.finished and not self.wire.done:
            final = self.final_due()
            acts = self.direct(final)
            if not acts:
                if self.failed_turns >= 3 or self.empty_turns >= 3 or final or self.wire.dead or self.wire.time_left() < 10:
                    break
                continue
            done_any = False
            queue = list(acts)
            while queue:
                a = queue.pop(0)
                if self.finished or self.wire.done:
                    break
                if not final and self.is_ending(a) and self.wire.steps_left() > 2 and self.wire.time_left() > END_RESERVE + 20 and not self.wire.over_budget():
                    instead = self.review(a)
                    if instead is not None:
                        queue = [x for x in instead if not self.is_ending(x)]
                        done_any = True
                        continue
                if a.get("tool") == "check" and "check" not in self.tools.items:
                    group = [a.get("args") if isinstance(a.get("args"), dict) else {}]
                    while queue and queue[0].get("tool") == "check":
                        nxt = queue.pop(0)
                        group.append(nxt.get("args") if isinstance(nxt.get("args"), dict) else {})
                    self.checks(group)
                    done_any = True
                    continue
                if self.is_ending(a) and not self.is_decline(a) and self.units_bounces < 2 and self.wire.steps_left() > 1 and self.wire.time_left() > 8:
                    gaps = self.memory.units_gaps()
                    if gaps:
                        self.units_bounces += 1
                        self.units_held = True
                        self.units_note = "YOUR ENDING CALL WAS HELD BACK, NOT SENT: the basket as the store last showed it does not hold the units the request asks for: %s. A whole-number argument of the basket tool sets how many units go on or come off a line, whatever its name; a second call for the same listing adds to its line. Correct the lines now, read the basket the store shows, and end in the same turn." % "; ".join(gaps)
                        self.record(a.get("tool") or "?", a.get("args") or {}, dumps({"not_sent": self.units_note}), "held back")
                        self.log.add("units_hold", gaps=gaps)
                        done_any = True
                        break
                if final and not self.is_ending(a) and self.wire.steps_left() <= 1:
                    self.record(a.get("tool") or "?", a.get("args") or {}, dumps({"not_sent": "only one store step is left, and it must end the visit"}), "not sent")
                    continue
                sent = self.execute(a)
                done_any = sent or done_any
                if sent and self.is_ending(a) and not self.finished and not self.wire.done:
                    self.bounce_step = self.wire.step
                    self.bounced = "your ending call %s at step %d did not end the visit: the platform did not mark it finished, so the visit is still open whatever its reply says. Act on what the reply changes before you end again, and do not send the same ending unchanged; when it says a listing in the basket is gone, sold out or cannot be bought, believe it even if your last look showed it available: in this same turn take that line out, put the next best fit in, then end (re-opening it does not bring it back). A listing still in the basket stays on its line through such a reply: never add it again, since every further add puts more units on its line. It returned: %s" % (a.get("tool"), self.wire.step, clip(self.memory.entries[-1]["view"], 900))
                if not self.finished and self.final_due() and not final:
                    break
            self.idle_turns = 0 if done_any else self.idle_turns + 1
            if self.idle_turns >= 3:
                break
            if final and not self.finished and self.units_held and not self.wire.done and self.wire.time_left() > 8 and self.wire.steps_left() > 1:
                self.units_held = False
                continue
            self.units_held = False
            if final and not self.finished:
                bounced = any(self.is_ending(x) for x in acts) and bool(self.bounced) and not self.wire.done
                if bounced and bounce_retry > 0 and self.wire.time_left() > 15 and self.wire.steps_left() > 1:
                    bounce_retry -= 1
                    continue
                unsent = any(e.get("label") == "not sent" for e in self.memory.entries[-len(acts):])
                if unsent and final_retry > 0 and self.wire.time_left() > 15 and self.wire.steps_left() > 0:
                    final_retry -= 1
                    continue
                break
        if not self.finished and not self.wire.done:
            self.end_anyway()
        self.log.add("usage", calls=self.wire.calls, tokens=self.wire.tokens, cached=self.wire.cached_tokens, spend=round(self.wire.spend, 5), turns=self.turn, seconds=round(self.wire.elapsed(), 1), steps=self.wire.step)
        return self.log.items


def agent_main(pdata: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Episode(pdata)
    try:
        return episode.run()
    except Exception as err:
        episode.log.add("crash", error=clip("%s: %s" % (type(err).__name__, err), 500))
        try:
            episode.end_anyway()
        except Exception as second:
            episode.log.add("crash", error=clip("%s: %s" % (type(second).__name__, second), 500))
        return episode.log.items


__all__ = ["agent_main"]
