from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from time import monotonic
from typing import Any

from src.agent.proxy_client import ProxyClient

DEADLINE = 250.0
QUERY_TOKENS = 6
CHECKOUT_RESERVE = 3
REPAIR_RESERVE = 4
VERIFY_ROUND = 3
ASK_DEFAULT = 5
ASK_TOGETHER = 8
ASK_ALONE = 2
SWEEP_TURNS = 5
DECOY_FRONT = 2
FIRST_WIDTH = 3
KIN_MERGE = True
GROUND_MIN = 4
COMPARE_START = 5
PREFIX_TRUST = 500
SORRY = "Apologies: none of the listings here matches everything you asked for, so this errand ends without a purchase."

class EnvDown(Exception):
    pass

class Finished(Exception):
    pass

def schema_problem(value: Any, schema: dict, where: str = "args") -> str | None:
    if not isinstance(schema, dict):
        return None
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind] if kind else []
    checks = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
              "string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
              "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool), "null": lambda v: v is None}
    if kinds and not any(checks.get(k, lambda v: True)(value) for k in kinds):
        return f"{where}: expected {kind}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where}: not one of {schema['enum']}"
    if "const" in schema and value != schema["const"]:
        return f"{where}: must be {schema['const']}"
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", 1 << 30):
            return f"{where}: length out of range"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"] or "maximum" in schema and value > schema["maximum"]:
            return f"{where}: out of range"
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 1 << 30):
            return f"{where}: wrong number of items"
        for i, item in enumerate(value):
            why = schema_problem(item, schema.get("items") or {}, f"{where}[{i}]")
            if why:
                return why
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                return f"{where}: missing {name}"
        for name, item in value.items():
            if name in props:
                why = schema_problem(item, props[name], f"{where}.{name}")
                if why:
                    return why
            elif schema.get("additionalProperties") is False:
                return f"{where}: unknown {name}"
    return None

@dataclass
class Reply:
    tool: str
    args: dict
    data: Any
    notices: list[str]
    refused: bool = False
    error: str | None = None
    raw: Any = None

class Env:
    def __init__(self, problem: dict, clock_start: float):
        env = problem["environment"]
        self.binding = env["binding"]
        view = env["policy_view"]
        self.request = view.get("query") or problem.get("query") or ""
        self.opening = view.get("opening")
        self.specs = {t["function"]["name"]: t["function"] for t in view.get("tools") or [] if "function" in t}
        self.max_steps = int(view.get("max_steps") or 38)
        self.step = 0
        self.finished = bool(view.get("finished"))
        self.started = clock_start
        self.client = ProxyClient(timeout=60, max_retries=3)
        self.log: list[Reply] = []

    def left(self) -> int:
        return self.max_steps - self.step

    def elapsed(self) -> float:
        return monotonic() - self.started

    def check(self, name: str, args: dict) -> str | None:
        spec = self.specs.get(name)
        if spec is None:
            return f"unknown tool {name}"
        return schema_problem(args, spec.get("parameters") or {"type": "object"})

    def call(self, name: str, args: dict) -> Reply:
        return self.batch([(name, args)])[0]

    def batch(self, actions: list[tuple[str, dict]]) -> list[Reply]:
        if self.finished:
            raise Finished("finished")
        actions = actions[: max(0, self.left())]
        if not actions:
            raise Finished("no steps left")
        for name, args in actions:
            why = self.check(name, args)
            if why:
                raise ValueError(f"engine built a broken call: {why}")
        envelope = {**self.binding, "step": self.step + 1}
        if len(actions) == 1:
            envelope["action"] = {"name": actions[0][0], "args": actions[0][1]}
        else:
            envelope["actions"] = [{"name": n, "args": a} for n, a in actions]
        result = self.client.post("/environment/call", json_data=envelope)
        if not isinstance(result, dict) or "observation" not in result:
            raise EnvDown("environment call failed")
        self.step = int(result.get("step") or self.step + len(actions))
        self.finished = bool(result.get("finished"))
        observations = result.get("observations") or [result["observation"]]
        replies = [self.unwrap(n, a, o) for (n, a), o in zip(actions, observations)]
        self.log.extend(replies)
        return replies

    def unwrap(self, name: str, args: dict, obs: Any) -> Reply:
        notices: list[str] = []
        refused = False
        if isinstance(obs, dict) and set(obs) == {"observation", "refusal"}:
            refused = True
            notices += [s for s in obs["refusal"] or [] if isinstance(s, str)] if isinstance(obs["refusal"], list) else \
                [str(obs["refusal"])] if obs["refusal"] else []
            obs = obs["observation"]
        for _ in range(2):
            if isinstance(obs, dict) and len(obs) == 2:
                keys = list(obs)
                lists = [k for k in keys if isinstance(obs[k], list) and obs[k] and all(isinstance(s, str) for s in obs[k])]
                if len(lists) == 1:
                    other = keys[0] if keys[1] == lists[0] else keys[1]
                    if isinstance(obs[other], (dict, list)) or obs[other] is None:
                        notices += obs[lists[0]]
                        obs = obs[other]
                        continue
            break
        raw = obs
        obs = structured(obs)
        while isinstance(obs, dict) and len(obs) == 1 and isinstance(next(iter(obs.values())), dict) and next(iter(obs.values())):
            obs = next(iter(obs.values()))
        error = obs["error"] if isinstance(obs, dict) and isinstance(obs.get("error"), str) and len(obs) <= 2 else None
        return Reply(name, args, obs, notices, refused, error, raw)

FIELD = re.compile(r"^\s*([A-Za-z_][\w .()/-]{0,40}?)\s*(?::\s|=)(.*)$", re.S)
PAIR = re.compile(r"^\s*([^=;|:\s][^=;|]{0,60}?)\s*=(.*)$", re.S)

def text_record(s: str) -> dict | None:
    if not isinstance(s, str) or len(s) < 3:
        return None
    best = None
    for sep in ("|", "\n", ";", ","):
        parts = s.split(sep)
        if len(parts) < 3:
            continue
        hits = [FIELD.match(p) for p in parts]
        n = sum(1 for m in hits if m)
        if n >= 3 and n * 2 >= len(parts):
            best = (sep, parts, hits)
            break
    if best is None:
        for sep in (";", ","):
            parts = [p for p in s.split(sep) if p.strip()]
            if parts and all(PAIR.match(p) for p in parts):
                best = (sep, parts, [PAIR.match(p) for p in parts])
                break
    if best is None:
        return None
    sep, parts, hits = best
    out: dict[str, str] = {}
    last = None
    for p, m in zip(parts, hits):
        if m and m.group(1).strip() not in out:
            last = m.group(1).strip()
            out[last] = m.group(2).strip()
        elif last is not None:
            out[last] += sep + p.rstrip()
    return {k: text_value(v) for k, v in out.items()}

def text_value(v: str) -> Any:
    t = v.strip()
    if t in ("True", "true", "False", "false"):
        return t.lower() == "true"
    if t in ("None", "null", ""):
        return None
    if re.fullmatch(r"-?\d{1,15}", t):
        return int(t)
    if re.fullmatch(r"-?\d+\.\d+", t):
        return float(t)
    for sep in (";", ",") if "=" in t else ():
        parts = [p for p in t.split(sep) if p.strip()]
        hits = [PAIR.match(p) for p in parts]
        if parts and hits[0] and sum(1 for h in hits if h) * 2 >= len(parts):
            out: dict[str, str] = {}
            last = None
            for p, m in zip(parts, hits):
                if m and m.group(1).strip() not in out:
                    last = m.group(1).strip()
                    out[last] = m.group(2).strip()
                elif last is not None:
                    out[last] += sep + p.rstrip()
            return {k: v if "=" in v else text_value(v) for k, v in out.items()}
    return t

LISTING_ID = re.compile(r"^[A-Za-z]{1,6}[-_][0-9A-Za-z]{6,}$")

def listing_line(x: str) -> dict | None:
    rec = text_record(x) if len(x) >= 12 else None
    if not rec:
        return None
    head = list(rec.items())[:2]
    return rec if any(isinstance(v, str) and LISTING_ID.match(v.strip()) for _, v in head) else None

def structured(x: Any, depth: int = 0) -> Any:
    if depth > 6:
        return x
    if isinstance(x, str):
        return listing_line(x) or x
    if isinstance(x, list):
        return [structured(v, depth + 1) for v in x]
    if isinstance(x, dict):
        return {k: structured(v, depth + 1) for k, v in x.items()}
    return x

@dataclass
class Tool:
    role: str
    name: str
    args: dict[str, str] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    style: str = ""
    limit: int = 0

@dataclass
class Fields:
    rows: str = ""
    total: str = ""
    next: str = ""
    id: str = ""
    title: str = ""
    price: str = ""
    stock: str = ""
    in_stock: list = field(default_factory=list)
    category: str = ""
    brand: str = ""
    facts: str = ""
    notes: str = ""
    reply: str = ""
    lines: str = ""
    line_id: str = ""
    line_qty: str = ""
    error: str = ""

@dataclass
class Req:
    label: str
    value: str
    exclude: bool = False
    source: str = "stated"
    said: str = ""

@dataclass
class Ref:
    slot: int
    label: str
    target: str
    hops: int = 0
    done: bool = False
    seller: str = ""
    relation: str = "same"
    margin: str = ""

@dataclass
class Cursor:
    role: str
    sem: dict
    label: str
    valued: bool = False
    evidence: bool = False
    whole: bool = False
    cap: float | None = None
    ids: set = field(default_factory=set)
    total: int | None = None
    page_size: int = 0
    done: bool = False
    ordered: bool = True
    last_id: str = ""
    priced: bool = True
    last_price: float | None = None
    by_rows: bool | None = None

@dataclass
class Slot:
    kind: str
    qty: int
    reqs: list[Req]
    category: str | None = None
    hints: list = field(default_factory=list)
    prefs: list = field(default_factory=list)
    asked: set = field(default_factory=set)
    free: set = field(default_factory=set)
    pool: dict = field(default_factory=dict)
    rank: dict = field(default_factory=dict)
    problems: dict = field(default_factory=dict)
    fits: list = field(default_factory=list)
    rejected: set = field(default_factory=set)
    gone: set = field(default_factory=set)
    supersets: list = field(default_factory=list)
    pick: str | None = None
    tried: list = field(default_factory=list)
    cursors: list = field(default_factory=list)
    totals: dict = field(default_factory=dict)
    prefixes: list = field(default_factory=list)

@dataclass
class Need:
    slots: list[Slot]
    budget: int | None
    cheapest: bool
    msg_budget: int | None
    refs: list[Ref] = field(default_factory=list)
    in_stock: bool = True
    no_names: bool = True
    step_limit: int | None = None

@dataclass
class Item:
    id: str
    title: str = ""
    price: float | None = None
    stock: bool | None = None
    category: str | None = None
    brand: str | None = None
    facts: dict | None = None
    notes: str | None = None
    opened: bool = False
    stale: bool = False
    raw: dict | None = None
    reads: int = 0
    text: str = ""

class Brain:

    def read_tools(self, request: str, specs: dict[str, dict]) -> dict[str, Tool]:
        raise NotImplementedError

    def read_fields(self, role: str, data: Any) -> Fields:
        raise NotImplementedError

    def read_request(self, request: str) -> Need:
        raise NotImplementedError

    def category(self, slot: Slot, labels: list[str]) -> str | None:
        raise NotImplementedError

    def judge(self, slot: Slot, item: Item) -> tuple[bool, list[str]]:
        raise NotImplementedError

    def judge_many(self, slot: Slot, items: list[Item]) -> dict[str, tuple[bool, list[str]]]:
        return {item.id: self.judge(slot, item) for item in items}

    def questions(self, need: Need, keys: dict[int, list[str]]) -> list[tuple[int, str]]:
        raise NotImplementedError

    def question_text(self, slot: Slot, label: str, attempt: int = 0) -> str:
        return f"Which {label} would you like for the {slot.kind}?" if attempt == 0 else f"What {label} do you need?"

    def hear(self, slot: Slot, label: str, reply: str) -> dict:
        raise NotImplementedError

    def read_notice(self, text: str, picks: dict[str, Item]) -> dict:
        raise NotImplementedError

    def read_note(self, ref: Ref, item: Item, target: str = "") -> dict:
        raise NotImplementedError

    def pick_reference(self, ref: Ref, rows: list[Item], target: str = "") -> list[str]:
        raise NotImplementedError

    def paired(self, prop: str, value: str, notes: list[str]) -> str | None:
        raise NotImplementedError

    def set_limits(self, max_steps: int) -> None:
        pass

LINKING = {"an", "and", "or", "the", "of", "for", "with", "in"}

def property_table(record: Any) -> dict | None:
    if not isinstance(record, dict):
        return None
    tables = [v for v in record.values() if isinstance(v, dict) and v and
              all(isinstance(x, (str, int, float, bool)) or x is None for x in v.values())]
    return tables[0] if len(tables) == 1 else None

BOUND = re.compile(r"^\s*(at most|at least|no (more|less|fewer) than|not (any )?(more|less|fewer) than|under|over|below|above|up to|"
                   r"minimum|maximum|min|max|more than|less than|greater than|fewer than)\b", re.I)

def checked_fields(f: Fields, data: Any) -> Fields:
    row = data
    if isinstance(data, dict) and f.rows:
        rows = data.get(f.rows)
        row = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None
    if not isinstance(row, dict):
        return f
    number = lambda k: isinstance(row.get(k), (int, float)) and not isinstance(row.get(k), bool)
    if not number(f.price):
        numbers = [k for k in row if number(k) and k not in (f.id, f.total, f.next)]
        if len(numbers) == 1:
            if f.stock == numbers[0]:
                f.stock = ""
            f.price = numbers[0]
    if f.stock not in row:
        buyable = [k for k, v in row.items() if k not in (f.id, f.price) and (isinstance(v, bool) or v in (f.in_stock or []))]
        f.stock = buyable[0] if len(buyable) == 1 else ""
        if f.stock and isinstance(row[f.stock], bool) and not any(isinstance(v, bool) for v in f.in_stock or []):
            f.in_stock = [True]
    return f

def holds_rows(data: Any) -> bool:
    stack = [data]
    while stack:
        piece = stack.pop()
        if isinstance(piece, list) and piece:
            return True
        if isinstance(piece, dict):
            stack.extend(piece.values())
    return False

COUNTED = re.compile(r"\d|\b(dozen|eight|eleven|five|four|nine|seven|six|ten|three|twelve|two)\b", re.I)

def which(group: list[str], named: str, label: str, reply: str = "") -> str:
    if named and named != label:
        for prop in group:
            if prop_name(prop) == prop_name(named) or names_prop(prop, named) or names_prop(named, prop):
                return prop
        return named
    said = set(query_words(reply))
    best = max(group, key=lambda g: len(set(query_words(g)) & said))
    return best if set(query_words(best)) & said else group[0]

def plain_values(reqs: list[Req]) -> list[str]:
    return [r.value for r in reqs if not r.exclude and not BOUND.search(r.value) and " or " not in f" {prop_name(r.value)} "]

def query_words(text: str) -> list[str]:
    out: list[str] = []
    for word in re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split():
        if (len(word) >= 2 or word.isdigit()) and word not in out and word not in LINKING:
            out.append(word)
    return out

class Engine:
    def __init__(self, problem: dict, brain: Brain, clock_start: float):
        self.env = Env(problem, clock_start)
        self.brain = brain
        self.items: dict[str, Item] = {}
        self.fields: dict[str, Fields] = {}
        self.tools: dict[str, Tool] = {}
        self.need: Need | None = None
        self.budget: int | None = None
        self.msgs_used = 0
        self.one_by_one = False
        self.shelf_arg: bool | None = None
        self.notes_log: list[str] = []
        self.trace: list[str] = []
        self.categories: list[str] | None = None
        self.cart: dict[str, int] = {}
        self.error_keys: set[str] = set()
        self.repairing = False
        self.order_tried = False
        self.guessed = False
        self.remapped = False
        self.word_rule: str | None = None
        self.seen_totals: dict = {}
        self.compare_bad = False

    def say(self, text: str) -> None:
        self.trace.append(f"[{self.env.step}] {text}")

    def tool(self, role: str) -> Tool | None:
        return self.tools.get(role)

    def fields_of(self, role: str, data: Any) -> Fields:
        if role in ("search", "filter") and role not in self.fields and not holds_rows(data):
            counts = [k for k, v in data.items() if isinstance(v, int) and not isinstance(v, bool)] if isinstance(data, dict) else []
            return Fields(total=counts[0] if len(counts) == 1 else "")
        if role not in self.fields and data is not None:
            self.fields[role] = checked_fields(self.brain.read_fields(role, data), data)
            if self.fields[role].error:
                self.error_keys.add(self.fields[role].error)
            f = self.fields[role]
            self.say(f"fields {role}: " + ", ".join(f"{k}={v}" for k, v in asdict(f).items() if v))
        return self.fields.get(role) or Fields()

    def error_of(self, role: str, data: Any) -> str | None:
        if not isinstance(data, dict) or len(data) != 1 or role == "message":
            return None
        key, value = next(iter(data.items()))
        if not isinstance(value, str):
            return None
        if key in self.error_keys or re.fullmatch(r"[A-Z][A-Z0-9_]{1,30}", value) or \
                re.search(r"error|invalid|unknown|not found|no such|too many|cannot|can't|empty|missing|exceed", value, re.I):
            return value
        return None

    def run_tool(self, role: str, sem_args: dict[str, Any]) -> Reply:
        tool = self.tools[role]
        args = {tool.args[k]: v for k, v in sem_args.items() if v is not None and k in tool.args}
        reply = self.env.call(tool.name, args)
        self.absorb(role, reply)
        return reply

    def run_batch(self, role: str, sem_list: list[dict[str, Any]]) -> list[Reply]:
        tool = self.tools[role]
        actions = [(tool.name, {tool.args[k]: v for k, v in sem.items() if v is not None and k in tool.args}) for sem in sem_list]
        replies = self.env.batch(actions)
        for reply in replies:
            self.absorb(role, reply)
        return replies

    def absorb(self, role: str, reply: Reply) -> None:
        reply.error = reply.error or self.error_of(role, reply.data)
        if reply.error:
            self.say(f"{role} error {reply.error}")
        for text in reply.notices:
            self.notice(text)

    def item_from(self, role: str, d: dict, opened: bool = False, asked: str | None = None) -> Item | None:
        f = self.fields_of(role, d)
        pid = d.get(f.id) if f.id else None
        if not isinstance(pid, str) or not pid:
            pid = asked
        if not isinstance(pid, str) or not pid:
            return None
        item = self.items.setdefault(pid, Item(pid))
        if f.title and isinstance(d.get(f.title), str):
            item.title = d[f.title]
        if f.price and isinstance(d.get(f.price), (int, float)) and not isinstance(d.get(f.price), bool):
            item.price = d[f.price]
        if f.stock and f.stock in d and f.in_stock:
            value = d[f.stock]
            item.stock = value in f.in_stock if isinstance(value, bool) else \
                str(value).strip().lower() in {str(v).strip().lower() for v in f.in_stock}
        if f.category and isinstance(d.get(f.category), str):
            item.category = d[f.category]
        shown = " ".join(v for k, v in d.items() if isinstance(v, str) and k != f.id)
        if shown:
            item.text = shown[:800]
        if f.brand and f.brand in d:
            item.brand = d[f.brand]
        facts = d.get(f.facts) if f.facts else None
        if not isinstance(facts, dict) and role in ("details", "specs", "compare"):
            facts = property_table(d)
        if isinstance(facts, dict):
            item.facts = facts
            item.stale = False
        if f.notes and isinstance(d.get(f.notes), str):
            item.notes = d[f.notes]
        if opened:
            item.opened = True
            item.raw = d
            if f.notes and f.notes not in d and item.notes is None:
                item.notes = ""
        return item

    def rows_of(self, role: str, data: Any) -> tuple[list[Item], int | None, Any]:
        f = self.fields_of(role, data)
        if not isinstance(data, dict):
            return [], None, None
        rows = data.get(f.rows) if f.rows else None
        items = [self.item_from(role, r) for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
        total = data.get(f.total) if f.total else None
        nxt = data.get(f.next) if f.next else None
        return [i for i in items if i], total if isinstance(total, int) else None, nxt

    def read_page(self, cur: Cursor, slot: Slot | None) -> list[Item]:
        try:
            reply = self.run_tool(cur.role, cur.sem)
        except ValueError:
            cur.done = True
            return []
        if reply.error:
            cur.done = True
            return []
        rows, total, nxt = self.rows_of(cur.role, reply.data)

        total = None if total is not None and nxt is not None and rows and total <= len(rows) else total
        cur.total = total if total is not None else cur.total
        if cur.role == "search" and total is not None and not cur.ids:
            self.learn_rule(query_words(cur.sem.get("query", "")), total)
        seq = ([cur.last_id] if cur.last_id else []) + [i.id for i in rows]
        cur.ordered = cur.ordered and all(a < b for a, b in zip(seq, seq[1:]))
        cur.last_id = max(seq) if seq else cur.last_id
        prices = [p for p in ([cur.last_price] + [i.price for i in rows]) if p is not None]
        cur.priced = cur.priced and all(a <= b for a, b in zip(prices, prices[1:]))
        cur.last_price = prices[-1] if prices else cur.last_price
        fresh = sum(1 for i in rows if i.id not in cur.ids)
        cur.ids |= {i.id for i in rows}
        cur.page_size = max(cur.page_size, len(rows))
        if nxt is None and rows and cur.total is not None and len(cur.ids) < cur.total and isinstance(cur.sem.get("page", 1), int):

            if cur.by_rows is None:
                cur.by_rows = self.offset_paging(cur.role)
            if not cur.by_rows and "page" in cur.sem and fresh < len(rows) / 2:
                cur.by_rows = True
            nxt = len(cur.ids) if cur.by_rows else cur.sem.get("page", 1) + 1
        if nxt is None or not rows or "page" not in self.tools[cur.role].args:
            cur.done = True
        else:
            cur.sem = dict(cur.sem, page=nxt)
        complete = cur.total is not None and len(cur.ids) >= cur.total
        if complete:
            cur.done = True
        self.shelve(rows, slot, 2 if cur.evidence else 0)
        if slot is not None and complete and cur.whole and self.on_shelf(slot, cur):
            slot.supersets.append((set(cur.ids), cur.cap))
        self.say(f"{cur.role} {cur.label}: {len(cur.ids)} rows of {cur.total}{' (all)' if complete else ''}")
        return rows

    def offset_paging(self, role: str) -> bool:
        tool = self.tools.get(role)
        schema = ((self.env.specs.get(tool.name) or {}).get("parameters") or {}).get("properties") or {} if tool else {}
        return (schema.get(tool.args.get("page", "")) or {}).get("minimum") == 0 if tool else False

    def learn_rule(self, words: list[str], total: int) -> None:
        key = frozenset(words)
        for other, n in self.seen_totals.items():
            more, fewer = (total, n) if other < key else (n, total) if key < other else (None, None)
            if more is None or more == fewer or not fewer:
                continue
            rule = "any" if more > fewer else "all"
            if rule != self.word_rule:
                self.word_rule = rule
                self.say(f"search rule: {rule} of the words")
        self.seen_totals[key] = total

    def on_shelf(self, slot: Slot, cur: Cursor) -> bool:
        if cur.role == "filter":
            return bool(cur.ids)
        return any(i in self.items and (slot.category is None or same_shelf(self.items[i].category, slot.category)) for i in cur.ids)

    def shelve(self, rows: list[Item], slot: Slot | None, evidence: int) -> None:
        for item in rows:
            for other in self.need.slots if self.need else []:
                if other is slot:
                    other.pool[item.id] = other.pool.get(item.id, 0) + evidence
                    other.rank.setdefault(item.id, len(other.rank))
                elif other.category and same_shelf(item.category, other.category):
                    other.pool.setdefault(item.id, 0)

    def can_finish_listing(self, cur: Cursor, pages: int = 4) -> bool:
        return not cur.done and cur.total is not None and cur.total - len(cur.ids) <= cur.page_size * pages

    def search_cursor(self, words: list[str], valued: bool, category: str | None = None) -> Cursor | None:
        tool = self.tool("search")
        if tool is None or not words:
            return None
        sem: dict[str, Any] = {"query": " ".join(words[:QUERY_TOKENS])}
        if category and "category" in tool.args and self.shelf_arg is not False:
            sem["category"] = category
        if "mode" in tool.args and "keyword" in tool.values:
            sem["mode"] = tool.values["keyword"]
        return Cursor("search", sem, repr(sem["query"]), valued=valued, whole=valued)

    def open_items(self, ids: list[str], details: bool = False) -> None:
        ids = [i for i in ids if i in self.items]
        if not ids:
            return
        for pid in ids:
            self.items[pid].reads += 1
        compare = self.tool("compare")
        if compare and not self.compare_bad and not details and len(ids) > 1:
            chunks = [ids[i:i + compare.limit] for i in range(0, len(ids), max(1, compare.limit))]
            again = []
            try:
                replies = self.run_batch("compare", [{"ids": c} for c in chunks])
            except ValueError as exc:
                self.say(f"compare unusable: {exc}")
                self.compare_bad = True
                for pid in ids:
                    self.items[pid].reads -= 1
                self.open_items(ids)
                return
            for chunk, reply in zip(chunks, replies):
                if reply.error:
                    if len(chunk) > 1:
                        compare.limit = max(2, len(chunk) - 2) if len(chunk) > 3 else len(chunk) - 1
                    again += chunk
                    self.say(f"compare of {len(chunk)} refused: limit now {compare.limit}")
                    continue
                self.rows_of("compare", reply.data)
                if all(self.items[p].facts is None for p in chunk):
                    self.compare_bad = True
                    again += chunk
            if again and self.spare() > 0:
                for pid in again:
                    self.items[pid].reads -= 1
                self.open_items(again)
            return
        if not details and "specs" in self.tools:
            self.read_specs(ids)
            return
        if "details" not in self.tools:
            if "specs" in self.tools:
                self.read_specs(ids)
            elif compare and not self.compare_bad:
                for reply in self.run_batch("compare", [{"ids": [i]} for i in ids]):
                    if not reply.error:
                        self.rows_of("compare", reply.data)
            return
        for pid, reply in zip(ids, self.run_batch("details", [{"id": i} for i in ids])):
            if isinstance(reply.data, dict) and not reply.error:
                self.item_from("details", reply.data, opened=True, asked=pid)

        missing = [pid for pid in ids if self.items[pid].facts is None]
        if missing and "specs" in self.tools and self.spare() > len(missing):
            self.read_specs(missing)

    def read_specs(self, ids: list[str]) -> None:
        for pid, reply in zip(ids, self.run_batch("specs", [{"id": i} for i in ids])):
            if isinstance(reply.data, dict) and not reply.error:
                self.item_from("specs", reply.data, asked=pid)

    def picks(self) -> dict[str, Item]:
        out = {}
        for slot in self.need.slots if self.need else []:
            for pid in ([slot.pick] if slot.pick else []) + slot.fits[:3]:
                if pid in self.items:
                    out[pid] = self.items[pid]
        for pid in self.cart:
            if pid in self.items:
                out[pid] = self.items[pid]
        return out

    def notice(self, text: str) -> None:
        if not text or text in self.notes_log[-6:]:
            return
        self.notes_log.append(text)

        in_cart = {pid: self.items[pid] for pid in self.cart if pid in self.items}
        news = self.brain.read_notice(text, in_cart or self.picks())
        self.say(f"notice {text!r} -> {news}")
        if news.get("budget") is not None:
            self.budget = news["budget"]
        for pid in news.get("gone") or []:
            for slot in self.need.slots:
                slot.gone.add(pid)
                if pid in slot.fits:
                    slot.fits.remove(pid)
        for pid in news.get("changed") or []:
            if pid in self.items:
                self.items[pid].stale = True
            for slot in self.need.slots:
                if pid in slot.fits:
                    slot.fits.remove(pid)

        dropped = {prop_name(d) for d in news.get("dropped") or []}
        added = news.get("added") or []
        if dropped or added:
            for slot in self.need.slots:
                slot.reqs = [r for r in slot.reqs if prop_name(r.label) not in dropped]
            for req in added:
                owner = next((s for s in self.need.slots if any(names_prop(r.label, req.label) or
                                                                names_prop(req.label, r.label) for r in s.reqs)),
                             self.need.slots[0])
                owner.reqs = [r for r in owner.reqs if prop_name(r.label) != prop_name(req.label)] + [req]
            for slot in self.need.slots:
                slot.rejected.clear()
                if dropped:
                    slot.supersets.clear()
                    slot.prefixes.clear()
                self.judge_known(slot)
            self.say(f"requirements now {[(s.kind, [(r.label, r.value, r.exclude) for r in s.reqs]) for s in self.need.slots]}")

    def reserve(self) -> int:
        n = len(self.need.slots) if self.need else 1
        cart = self.tool("cart")
        per_line = 0 if cart is not None and cart.style == "lines" else 1
        return CHECKOUT_RESERVE + per_line * max(0, n - 1) + (0 if self.order_tried else REPAIR_RESERVE)

    def spare(self) -> int:
        return self.env.left() - self.reserve()

    def late(self) -> bool:
        return self.env.elapsed() > DEADLINE

    def run(self) -> None:
        self.tools = self.brain.read_tools(self.env.request, self.env.specs)
        self.say("tools: " + "; ".join(f"{r}={t.name}{t.args}{t.values or ''}{t.style or ''}{t.limit or ''}"
                                        for r, t in sorted(self.tools.items())))
        self.brain.set_limits(self.env.max_steps)
        if "compare" in self.tools:
            self.tools["compare"].limit = min(self.tools["compare"].limit or COMPARE_START, COMPARE_START)
        self.need = self.brain.read_request(self.env.request)
        phrased(self.env.request, self.need.slots)
        self.budget = self.need.budget
        if self.need.step_limit and self.need.step_limit >= 10:
            self.env.max_steps = min(self.env.max_steps, self.need.step_limit)
        self.say(f"need: {[(s.kind, s.qty, s.category, [(r.label, r.value, r.exclude) for r in s.reqs]) for s in self.need.slots]} "
                 f"budget {self.budget} cheapest {self.need.cheapest} refs {[(r.label, r.target) for r in self.need.refs]}")
        if any(s.category is None for s in self.need.slots):
            self.schema_shelves()
        lister = self.tools.get("shelves")
        if lister and any(s.category is None for s in self.need.slots) and                 lister.name not in {t.name for r, t in self.tools.items() if r in ("search", "filter")}:

            self.read_shelves()
        for ref in self.need.refs:
            self.follow(ref)
        for slot in self.need.slots:
            self.gather(slot)
        self.verify_round(first=True)
        self.ask()
        self.work()
        self.checkout()

    def work(self) -> None:
        for _ in range(max(24, self.env.max_steps)):
            if all(self.settled(s) for s in self.need.slots) or self.spare() <= 0 or self.late():
                return
            empty = [s for s in self.need.slots if not s.fits]
            moved = False
            for slot in empty or [s for s in self.need.slots if not self.settled(s)]:
                moved |= self.verify(slot, VERIFY_ROUND)
                unsettled = not slot.fits if empty else not self.settled(slot)
                if unsettled and not self.verifiable(slot):
                    moved |= self.widen(slot)
            if not moved and empty:
                for slot in self.need.slots:
                    if slot.fits and not self.settled(slot):
                        moved |= self.verify(slot, VERIFY_ROUND)
                        if not self.settled(slot) and not self.verifiable(slot):
                            moved |= self.widen(slot)
            if not moved:
                return

    def schema_shelves(self) -> None:
        for role in ("filter", "search"):
            tool = self.tools.get(role)
            if tool is None or "category" not in tool.args:
                continue
            schema = ((self.env.specs.get(tool.name) or {}).get("parameters") or {}).get("properties") or {}
            names = (schema.get(tool.args["category"]) or {}).get("enum") or []
            if names and all(isinstance(n, str) for n in names):
                self.categories = list(names)
                for slot in self.need.slots:
                    if slot.category is None:
                        slot.category = self.brain.category(slot, self.categories)
                self.say(f"shelves from the {tool.name} schema {len(names)}: {[(s.kind, s.category) for s in self.need.slots]}")
                return

    def read_shelves(self) -> None:
        reply = self.run_tool("shelves", {})
        names: list[str] = []
        stack = [reply.data]
        while stack:
            piece = stack.pop()
            if isinstance(piece, dict):
                stack.extend(piece.values())
            elif isinstance(piece, list):
                if piece and all(isinstance(x, str) for x in piece):
                    names += piece
                else:
                    stack.extend(piece)
        if names and not reply.error:
            self.categories = names
            for slot in self.need.slots:
                if slot.category is None:
                    slot.category = self.brain.category(slot, names)
            self.say(f"shelves {len(names)}: {[(s.kind, s.category) for s in self.need.slots]}")

    def settled(self, slot: Slot) -> bool:
        if not slot.fits:
            return False
        if not self.need.cheapest and not self.over_budget():
            if strength(slot, self.items[slot.fits[0]]) < 2 and self.spare() > 6:
                return False

            wishful = slot.prefs and self.wishes_met(slot, self.items[slot.fits[0]]) == 0
            return not (wishful and self.spare() > 6 and self.verifiable(slot))
        best = self.items[slot.fits[0]].price or 0
        if not self.covered(slot, self.under(best)):
            return False
        return not any(self.candidate(slot, p) and p not in slot.fits and (self.items[p].price or 0) < best
                       for p in slot.pool)

    @staticmethod
    def under(price: float) -> float:
        return price - 1 if float(price).is_integer() else price * 0.999

    def covered(self, slot: Slot, price: float) -> bool:
        return any(cap is None or cap >= price for _, cap in slot.supersets)

    def reference_rows(self, target: str, seller: str, seen: set) -> list[Item]:
        rows: list[Item] = []

        labels = self.categories or sorted({i.category for i in self.items.values() if i.category})
        named = [c for c in labels if set(query_words(c)) and set(query_words(c)) <= set(query_words(target))]
        shelf = max(named, key=lambda c: len(query_words(c))) if named else None

        names = {w for i in self.items.values() for k in (i.facts or {}) for w in query_words(k)}
        skip = names | set(query_words(seller)) | set(query_words(shelf or "")) | set(query_words("sold by from with page"))
        values = [w for w in query_words(target) if w not in skip]
        tries = ([query_words(seller) + values] if seller and values else []) + \
            ([query_words(seller)] if seller else []) + ([values] if values else []) + [query_words(target)]
        for words in [t[:QUERY_TOKENS] for t in tries]:
            if not words or tuple(words) in seen or self.spare() <= 2:
                continue
            seen.add(tuple(words))
            cur = self.search_cursor(words, valued=False, category=shelf)
            if cur is None:
                break
            rows += self.read_page(cur, None)
            pages = 0
            while self.can_finish_listing(cur, 3) and self.spare() > 2 and pages < 3:
                rows += self.read_page(cur, None)
                pages += 1

            if seller and cur.done and any(set(query_words(seller)) <= set(query_words(i.text)) for i in rows):
                break
        return list({i.id: i for i in rows}.values())

    def settle_reference(self, ref: Ref, value: str) -> None:
        slot = self.need.slots[ref.slot]
        exclude = ref.relation == "different"
        values = [value]
        if ref.relation in ("at least", "at most"):
            values = [f"{ref.relation} {value}"]
        elif ref.relation == "pair":

            notes = [i.notes for i in self.items.values() if i.notes and set(query_words(ref.label)) <= set(query_words(i.notes))]
            paired = self.brain.paired(ref.label, value, notes) if notes else None
            values = [paired or value]
        elif ref.relation == "within":
            at, by = numbers_in(value), numbers_in(ref.margin)
            unit = re.sub(r"[-+]?\d+(?:\.\d+)?", "", value).strip()
            if at and by:
                lo, hi = round(at[0] - by[0], 2), round(at[0] + by[0], 2)
                values = [f"at least {lo:g} {unit}".strip(), f"at most {hi:g} {unit}".strip()]
        slot.reqs = [r for r in slot.reqs if prop_name(r.label) != prop_name(ref.label)] +             [Req(ref.label, v, exclude, source="reference") for v in values]
        value = " and ".join(values)
        ref.done = True
        self.say(f"reference resolved: {ref.label} {ref.relation} {value}")
        self.judge_known(slot)

    def follow(self, ref: Ref) -> None:
        target, seller = ref.target, ref.seller
        visited: set = set()
        seen: set = set()
        for hop in range(5):
            if self.spare() <= 2 or self.late() or tuple(query_words(target)) in visited:
                return
            visited.add(tuple(query_words(target)))
            rows = self.reference_rows(target, seller, seen)
            order = self.brain.pick_reference(replace(ref, seller=seller), rows, target)
            found, opened = None, []
            for pid in order[:3]:
                if self.spare() <= 1:
                    break
                if not self.items[pid].opened:
                    self.open_items([pid], details=True)
                opened.append(self.items[pid])
                found = self.brain.read_note(ref, self.items[pid], target)
                if found:
                    break
            if not found:

                want = set(query_words(target))
                carry = lambda i: len(want & set(query_words(" ".join([i.title or ""] + [f"{k} {v}" for k, v in (i.facts or {}).items()]))))
                first = max(opened, key=carry) if opened else None
                fact = next((v for k, v in (first.facts or {}).items() if names_prop(k, ref.label)), None) if first else None
                if fact is not None:
                    self.settle_reference(ref, str(fact))
                else:
                    self.say(f"reference {target!r}: no note found")
                return
            ref.hops += 1
            if found.get("value"):
                self.settle_reference(ref, found["value"])
                return
            target, seller = found.get("next") or "", found.get("seller") or ""
            if not target:
                return

    def value_words(self, slot: Slot) -> set[str]:
        return {w for v in plain_values(slot.reqs) for w in query_words(v)} | set(query_words(slot.category or ""))

    def shelf_words(self, slot: Slot) -> list[str]:
        return sorted(query_words(slot.category), key=lambda w: -len(w))[:2] if slot.category else []

    def query_plan(self, slot: Slot) -> list[list[str]]:
        values = sorted([query_words(v) for v in plain_values(slot.reqs) if query_words(v)], key=len)
        shelf = self.shelf_words(slot)
        kind = query_words(slot.kind)
        plans: list[list[str]] = []

        def add(words: list[str]) -> None:
            words = list(dict.fromkeys(words))[:QUERY_TOKENS]
            if words and words not in plans:
                plans.append(words)
        every = [w for v in values for w in v]
        wished = [w for v in plain_values(slot.prefs) for w in query_words(v) if w not in every]
        if wished:
            add(shelf + every + wished)
        add(shelf + every)
        if self.word_rule == "any":

            for v in values:
                add(v + query_words(slot.kind))
            for v in values:
                if len(v) == 1:
                    add(v)
        elif slot.totals.get(tuple((shelf + every)[:QUERY_TOKENS])) == 0 and len(values) > 1:
            for i in range(len(values)):
                add(shelf + [w for j, v in enumerate(values) if j != i for w in v])
            for v in values:
                add(shelf + v)
        if not shelf:
            add(every + kind)
            if not every:
                add(kind)
        return plans

    def read_listing(self, slot: Slot, cur: Cursor) -> list[Item]:
        rows = self.read_page(cur, slot)
        pages = 0
        while self.can_finish_listing(cur) and self.spare() > 0 and pages < 4:
            rows += self.read_page(cur, slot)
            pages += 1
        if not cur.done:
            slot.cursors.append(cur)
        return rows

    def run_query(self, slot: Slot, words: list[str], relax: int = 2) -> list[Item]:
        slot.tried.append(words)

        valued = set(words) <= self.value_words(slot)
        cur = self.search_cursor(words, valued=valued, category=slot.category)
        if cur is None:
            return []
        cur.evidence = bool(set(words) & {w for v in plain_values(slot.reqs) for w in query_words(v)})
        if valued:
            slot.prefixes.append(cur)
        rows = self.read_listing(slot, cur)
        if cur.sem.get("category"):
            if rows or cur.total:
                self.shelf_arg = True
            elif self.shelf_arg is None and self.spare() > 2:

                bare = self.search_cursor(words, valued=valued)
                found = self.read_listing(slot, bare) if bare is not None else []
                if any(same_shelf(i.category, slot.category) for i in found):
                    self.shelf_arg = False
                    rows = found
        slot.totals[tuple(words[:QUERY_TOKENS])] = cur.total
        self.learn_category(slot, rows)
        if valued and cur.total:

            for other in slot.cursors:
                if other is not cur and other.valued and other.role == "search" and \
                        set(query_words(other.sem.get("query", ""))) < set(words):
                    other.done = True

        if not rows and not cur.total and relax > 0 and len(words) > 1 and words[:-1] not in slot.tried \
                and self.spare() > 2 and not self.late():
            return self.run_query(slot, words[:-1], relax - 1)
        return rows

    def shelf(self, slot: Slot, cap: float | None, walk: int = 4) -> list[Item]:
        tool = self.tool("filter")
        if tool is None or not slot.category or ("shelf", cap) in slot.tried:
            return []
        slot.tried.append(("shelf", cap))
        sem: dict[str, Any] = {"category": slot.category}
        if cap is not None and "max_price" in tool.args:
            sem["max_price"] = int(cap) if float(cap).is_integer() else cap
        if "mode" in tool.args and "filter" in tool.values:
            sem["mode"] = tool.values["filter"]
        cur = Cursor("filter", sem, f"{slot.category!r} <= {cap}", whole=True, cap=cap if "max_price" in tool.args else None)
        rows = self.read_page(cur, slot)
        cheap = self.need.cheapest or self.over_budget()
        if cheap and cap and "max_price" in tool.args and walk > 0 and cap > 2 and self.spare() > 2 and \
                not self.can_finish_listing(cur, 3) and not cur.done:
            return rows + self.shelf(slot, cap // 2, walk - 1)
        pages = 0
        while self.can_finish_listing(cur) and self.spare() > 0 and pages < 4:
            rows += self.read_page(cur, slot)
            pages += 1
        if not cur.done:
            slot.cursors.append(cur)
        return rows

    def wide(self, slot: Slot) -> bool:
        return self.word_rule == "any"

    def walk_up(self, slot: Slot) -> bool:
        tool = self.tool("filter")
        caps = [c for _, c in slot.supersets if c is not None]
        top = self.unit_cap(slot)
        if tool is None or not caps or "max_price" not in tool.args:
            return False
        cap = max(caps) * 2
        cap = min(cap, top) if top else cap
        if cap <= max(caps) or ("shelf", cap) in slot.tried:
            return False
        self.shelf(slot, cap, walk=0)
        return True

    def more_rows(self, slot: Slot) -> bool:
        if self.spare() <= 0:
            return False

        def promise(c: Cursor) -> float:
            rows = [self.items[i] for i in c.ids if i in self.items]
            share = sum(1 for i in rows if not slot.category or same_shelf(i.category, slot.category)) / max(1, len(rows))
            return share * (2.0 if c.valued else 1.0)

        starved = not any(self.candidate(slot, p) and self.items[p].facts is None for p in slot.pool)
        open_ = [c for c in slot.cursors if not c.done and
                 (c.valued and promise(c) > 0 or promise(c) >= 0.3 or c.role == "filter" and starved)]
        if not open_:
            return False
        open_.sort(key=lambda c: -promise(c))
        self.read_page(open_[0], slot)
        return True

    def unit_cap(self, slot: Slot) -> float | None:
        return None if self.budget is None else self.budget / slot.qty

    def gather(self, slot: Slot) -> None:
        if slot.category is None and query_words(slot.kind) and self.spare() > 2:

            rows = self.run_query(slot, query_words(slot.kind)[:QUERY_TOKENS])
            head = query_words(re.split(r"[,(;]", slot.kind)[0])[:QUERY_TOKENS]
            if not rows and head and head not in slot.tried and self.spare() > 2:
                self.run_query(slot, head)
        for _ in range(3):
            plan = [w for w in self.query_plan(slot) if w not in slot.tried]
            if not plan or self.spare() <= 1 or self.late():
                return
            rows = self.run_query(slot, plan[0])
            if any(self.candidate(slot, i.id) for i in rows):
                break
        some = any(self.candidate(slot, p) for p in slot.pool)
        if self.tool("filter") and (self.need.cheapest and not self.covered(slot, self.unit_cap(slot) or float('inf')) and not (some and self.wide(slot))
                                    or not some):
            if self.spare() > 1:
                self.shelf(slot, self.unit_cap(slot))

    def widen(self, slot: Slot) -> bool:
        unread = lambda: sum(1 for p in slot.pool if self.candidate(slot, p) and self.items[p].facts is None)
        moved = False
        for _ in range(4):
            if self.spare() <= 0 or self.late() or not self.widen_once(slot):
                return moved
            moved = True
            if unread():
                return True
        return moved

    def widen_once(self, slot: Slot) -> bool:
        plans = self.query_plan(slot)
        valued = [w for w in plans if set(w) <= self.value_words(slot)]
        if (self.need.cheapest or self.over_budget()) and slot.fits and self.wide(slot):

            for words in valued:
                if words not in slot.tried:
                    self.run_query(slot, words)
                    return True
        if (self.need.cheapest or self.over_budget()) and self.tool("filter") and slot.fits:
            cap = self.under(self.items[slot.fits[0]].price or 0)
            if not self.covered(slot, cap) and ("shelf", cap) not in slot.tried:
                self.shelf(slot, cap, walk=1 if self.wide(slot) else 4)
                return True
        for words in valued:
            if words not in slot.tried:
                self.run_query(slot, words)
                return True
        cheapest = self.need.cheapest or self.over_budget()
        shelf_due = self.tool("filter") and slot.category and ("shelf", self.unit_cap(slot)) not in slot.tried
        if shelf_due and not slot.fits:

            self.shelf(slot, self.unit_cap(slot))
            return True
        if any(c.valued and not c.done for c in slot.cursors) and self.more_rows(slot):
            return True
        if shelf_due and cheapest:
            self.shelf(slot, self.unit_cap(slot))
            return True
        if cheapest and not self.queue(slot) and self.walk_up(slot):
            return True
        for words in plans:
            if words not in slot.tried:
                self.run_query(slot, words)
                return True
        if shelf_due:
            self.shelf(slot, self.unit_cap(slot))
            return True
        if self.more_rows(slot):
            return True
        for words in self.split_plans(slot):
            if words not in slot.tried:
                self.run_query(slot, words)
                return True
        return False

    def split_plans(self, slot: Slot) -> list[list[str]]:
        plans = self.query_plan(slot)
        plans = [w for w in plans if set(w) <= self.value_words(slot)] or plans
        if self.word_rule == "any" or not plans or not slot.totals.get(tuple(plans[0][:QUERY_TOKENS])):
            return []
        base = plans[0]
        titles = [set(query_words(self.items[p].title)) for p in slot.pool
                  if same_shelf(self.items[p].category, slot.category) and self.items[p].title]
        if len(titles) < 3:
            return []
        counts: dict[str, int] = {}
        for words in titles:
            for w in words - set(base):
                if not w.isdigit():
                    counts[w] = counts.get(w, 0) + 1

        useful = [w for w, n in counts.items() if 0.15 * len(titles) <= n <= 0.6 * len(titles)]
        useful.sort(key=lambda w: (-counts[w], w))
        return [[w] + base[: QUERY_TOKENS - 1] for w in useful[:4]]

    def learn_category(self, slot: Slot, rows: list[Item]) -> None:
        if slot.category is not None:
            return
        labels = self.categories or sorted({i.category for i in self.items.values() if i.category})
        if labels and tuple(labels) not in slot.tried:
            slot.tried.append(tuple(labels))
            slot.category = self.brain.category(slot, labels)
            self.say(f"category for {slot.kind}: {slot.category} (of {len(labels)})")

    def candidate(self, slot: Slot, pid: str) -> bool:
        item = self.items.get(pid)
        if item is None or pid in slot.gone or pid in slot.rejected:
            return False
        if slot.category and item.category and not same_shelf(item.category, slot.category):
            return False
        if self.need.in_stock and item.stock is False:
            return False
        if self.budget is not None and item.price is not None and item.price * slot.qty > self.budget:
            return False
        for ids, cap in slot.supersets:
            if pid not in ids and (cap is None or (item.price is not None and item.price <= cap)):
                return False
        for cur in slot.prefixes:
            if pid not in cur.ids and pid < cur.last_id and self.trusted_prefix(cur) and self.on_shelf(slot, cur):
                return False
        return True

    @staticmethod
    def trusted_prefix(cur: Cursor) -> bool:
        return cur.ordered and not cur.priced and len(cur.ids) >= 3 and cur.total is not None and cur.total < PREFIX_TRUST

    def ground(self, slot: Slot) -> None:
        read = [i for i in self.items.values() if i.facts and i.id in slot.pool and (not slot.category or same_shelf(i.category, slot.category))]
        for req in [r for r in slot.reqs if r.source == "phrase"]:

            if len(read) >= GROUND_MIN and not any(kindred(k, req.label) for i in read for k in i.facts):
                self.say(f"no property {req.label!r} on the shelf: {req.value!r} is a wish")
                slot.reqs.remove(req)
                slot.prefs.append(req)
                self.forget(slot)
        if slot.fits:
            return
        for req in slot.reqs:
            new = self.rewritten(slot, req, req.said or self.env.request, GROUND_MIN)
            if new:
                self.say(f"ground {req.label}: {req.value!r} -> {new!r}")
                req.value = new
                self.forget(slot)

    def rewritten(self, slot: Slot, req: Req, text: str, least: int) -> str | None:
        if req.exclude or re.search(r"\d", req.value) or BOUND.search(req.value) or " or " in f" {prop_name(req.value)} ":
            return None
        values, carriers = set(), 0
        for item in self.items.values():
            if item.facts and item.id in slot.pool and (not slot.category or same_shelf(item.category, slot.category)):
                vals = [str(v) for k, v in item.facts.items() if field_class(k, req.label) in ("own", "front")]
                carriers += bool(vals)
                values |= {prop_name(x) for v in vals for x in re.split(r"[,;/|]", v) if prop_name(x)}
        if carriers < least or any(holds(v, req.value) for v in values):
            return None

        found = [v for v in values if len(v.split()) <= 3 and holds(text, v) and not holds(req.value, v)]

        top = [v for v in found if all(holds(v, w) for w in found)]
        return top[0] if len(top) == 1 else None

    @staticmethod
    def forget(slot: Slot) -> None:
        slot.rejected.clear()
        slot.problems.clear()
        slot.supersets = []
        slot.prefixes = []

    def judge_known(self, slot: Slot) -> None:
        self.ground(slot)
        slot.fits = []
        ready = [self.items[p] for p in list(slot.pool)
                 if self.items[p].facts is not None and not self.items[p].stale and self.candidate(slot, p)]
        verdicts = self.brain.judge_many(slot, ready) if ready else {}
        for item in ready:
            if item.id not in verdicts:
                continue
            fits, problems = verdicts[item.id]
            if fits:
                slot.fits.append(item.id)
            else:
                self.say(f"  reject {item.id} {item.price}: {'; '.join(problems)[:160]}")
                slot.rejected.add(item.id)
                slot.problems[item.id] = len(problems) or 1
        by_price = self.need.cheapest or self.over_budget_of(slot)

        slot.fits.sort(key=lambda p: (0 if by_price else -strength(slot, self.items[p]), 0 if by_price else -shown(slot, self.items[p]),
                                      0 if by_price else -self.wishes_met(slot, self.items[p]), self.items[p].price or 0, p))

    def over_budget_of(self, slot: Slot) -> bool:
        cheapest = min((self.items[p].price or 0 for p in slot.fits), default=0)
        return self.budget is not None and cheapest * slot.qty > self.budget

    @staticmethod
    def wishes_met(slot: Slot, item: Item) -> int:
        met = 0
        for req in slot.prefs:
            named = [str(v) for k, v in (item.facts or {}).items() if names_prop(k, req.label)]
            has = any(holds(v, req.value) for v in named) or holds(item.title, req.value)
            met += has != req.exclude
        return met

    def queue(self, slot: Slot) -> list[str]:

        out = [p for p in slot.pool if self.candidate(slot, p) and p not in slot.fits
               and (self.items[p].facts is None and self.items[p].reads < 2 or self.items[p].stale)]
        price = lambda p: self.items[p].price if self.items[p].price is not None else float('inf')
        if self.need.cheapest or self.over_budget():

            out.sort(key=lambda p: (slot.pool.get(p, 0) == 0, price(p), p))
            if slot.fits:
                out = [p for p in out if price(p) < (self.items[slot.fits[0]].price or 0)]
            return out
        values = [query_words(v) for v in plain_values(slot.reqs + slot.prefs)]
        ranked = self.word_rule == "any"

        def likely(p: str) -> tuple:
            title = set(query_words(self.items[p].title))

            return (-slot.pool.get(p, 0), -sum(all(w in title for w in v) for v in values if v),
                    slot.rank.get(p, 1 << 30) if ranked else price(p), p)
        out.sort(key=likely)
        return out

    def verifiable(self, slot: Slot) -> bool:
        return bool(self.queue(slot))

    def verify(self, slot: Slot, n: int) -> bool:
        queue = self.queue(slot)
        compare = self.tool("compare")
        per_step = compare.limit if compare and not self.compare_bad else 1

        queue = queue[: max(0, min(max(n, per_step), self.spare() * per_step))]
        if not queue:
            return False
        self.say(f"verify {slot.kind}: open {queue}")
        self.open_items(queue)
        self.judge_known(slot)
        self.say(f"verify {slot.kind}: fits {slot.fits[:4]}")
        return True

    def verify_round(self, first: bool = False) -> None:
        for slot in self.need.slots:
            if self.spare() > 0 and not self.late():
                self.verify(slot, max(2, 5 - len(self.need.slots)) if first else VERIFY_ROUND)

    def ask(self) -> None:
        budget = self.need.msg_budget if self.need.msg_budget is not None else ASK_DEFAULT
        if budget <= 0:
            return

        unread = []
        for slot in self.need.slots:
            on_shelf = [p for p in slot.pool if slot.category is None or same_shelf(self.items[p].category, slot.category)]
            if not any(self.items[p].facts for p in on_shelf):
                unread += [p for p in on_shelf if p not in slot.gone][:2]
        if unread and self.spare() > 1:
            self.open_items(unread[: max(1, self.spare() - 1)])
        keys: dict[int, list[str]] = {}
        shelf_names: dict[int, set] = {}
        for k, slot in enumerate(self.need.slots):
            shelf = [self.items[p].facts or {} for p in slot.pool if not slot.category or same_shelf(self.items[p].category, slot.category)]
            names = {prop_name(key) for facts in shelf for key in facts}
            shelf_names[k] = names
            counts: dict[str, int] = {}
            values: dict[str, dict[str, int]] = {}
            spelled = field_spellings(shelf)
            for facts in shelf:
                for name in {spelled[prop_name(key)] for key in facts}:
                    counts[name] = counts.get(name, 0) + 1
                for key, val in facts.items():
                    text = prop_name(val)
                    if text and len(text) <= 24:
                        seen = values.setdefault(spelled[prop_name(key)], {})
                        seen[text] = seen.get(text, 0) + 1
            keys[k] = [f"{key} (on {n}): " + " | ".join(sorted(values.get(key, {}), key=lambda v: -values[key][v])[:3])
                       for key, n in sorted(counts.items(), key=lambda kv: -kv[1])]

        refs: list[Ref] = []
        width = max(1, ASK_TOGETHER) if budget > ASK_ALONE else 1
        plan = self.brain.questions(replace(self.need, msg_budget=budget * width), keys)
        props: dict[int, list[str]] = {}
        for k, prop in plan:

            known = props.get(k, []) + [r.label for r in self.need.slots[k].reqs] if KIN_MERGE else []
            if len(props.setdefault(k, [])) < (2 * width if width > 1 else budget) and not any(kindred(prop, q) for q in known):
                props[k].append(prop)

        target = 2 * width if width > 1 else budget
        for k, slot in enumerate(self.need.slots):
            known = props.get(k, []) + [r.label for r in slot.reqs]
            for entry in keys.get(k, []):
                if len(props.setdefault(k, [])) >= target:
                    break
                name = entry.split(" (on ")[0]
                if prop_name(name) in UNASKED or name.startswith("fa ") or any(kindred(name, q) for q in known):
                    continue
                props[k].append(name)
                known.append(name)
        turns = {k: 0 for k in props if props[k]}
        while any(props.values()):
            if self.msgs_used >= budget or self.spare() <= 1 or self.late():
                break
            k = min((j for j in props if props[j]), key=lambda j: (turns[j], j))
            if width == 1 or self.one_by_one:
                group = props[k][:1]
            elif turns[k] >= SWEEP_TURNS:
                props[k] = []
                continue
            else:

                group = props[k][:width if self.msgs_used else min(width, FIRST_WIDTH)]
            turns[k] += 1
            label = group[0] if len(group) == 1 else ", ".join(group[:-1]) + " or " + group[-1]
            slot = self.need.slots[k]
            reply = self.run_tool("message", {"text": self.brain.question_text(slot, label)})
            self.msgs_used += 1
            if reply.error:
                break
            text = self.reply_text(reply.raw if reply.raw is not None else reply.data)
            if re.fullmatch(r"[A-Za-z0-9_\-]+", text.strip()):
                self.say(f"message refused: {text}")
                break
            heard = self.brain.hear(slot, label, text)
            self.say(f"asked {label!r} -> {text!r} -> {heard}")
            if len(group) > 1 and heard.get("one_at_a_time"):
                self.one_by_one = True
                continue
            for item in (heard.get("needs") or []) + (heard.get("refs") or []):

                said = spoken_property(text, shelf_names.get(k, set()) | {prop_name(g) for g in group})
                if said and not holds(text, item.label):
                    item.label = said
                elif item.label == label:
                    item.label = which(group, item.label, label, text)
                if isinstance(item, Req):
                    item.said = text
                    new = self.rewritten(slot, item, text, 1)
                    if new:
                        self.say(f"heard {item.label}: {item.value!r} -> {new!r}")
                        item.value = new
            answered = [which(group, r.label, label, text) for r in (heard.get("needs") or []) + (heard.get("refs") or [])]
            refs += [replace(r, slot=k) for r in heard.get("refs") or []]
            slot.asked.update(group)
            if len(group) == 1 or self.one_by_one:

                props[k] = [q for q in props[k] if q not in group]
            elif answered:

                cut = max(group.index(a) for a in answered if a in group) if any(a in group for a in answered) else 0
                props[k] = [q for q in props[k] if q not in group[: cut + 1]]
            else:

                front = group[: min(DECOY_FRONT, len(group) - 1)]
                props[k] = [q for q in props[k] if q not in front] + front
            for req in heard.get("needs") or []:
                if any(r.label == req.label and prop_name(r.value) != prop_name(req.value) for r in slot.reqs):
                    self.forget(slot)
                slot.reqs = [r for r in slot.reqs if r.label != req.label] + [req]
                slot.prefs = [r for r in slot.prefs if not (names_prop(r.label, req.label) or names_prop(req.label, r.label) or prop_name(r.label) == prop_name(label))]
            for free in heard.get("free") or [] if len(group) == 1 else []:
                slot.free.add(free)
            if heard.get("budget") is not None:
                self.budget = heard["budget"]
            if heard.get("needs"):
                self.judge_known(slot)
        for ref in refs:
            self.need.refs.append(ref)
            self.follow(ref)

    def reply_text(self, data: Any) -> str:
        f = self.fields_of("message", data)
        if isinstance(data, dict) and f.reply and isinstance(data.get(f.reply), str):
            return data[f.reply]
        return data if isinstance(data, str) else json.dumps(data)[:500]

    def choose(self) -> bool:
        for slot in self.need.slots:
            slot.fits = [p for p in slot.fits if self.candidate(slot, p) and not self.items[p].stale]
            slot.pick = slot.fits[0] if slot.fits else None
        if any(s.pick is None for s in self.need.slots):
            return False
        total = sum((self.items[s.pick].price or 0) * s.qty for s in self.need.slots)
        return self.budget is None or total <= self.budget

    def checkout(self) -> None:
        for attempt in range(8):
            if self.env.finished or self.env.left() <= 0:
                return
            if self.env.left() == 1:
                self.say(f"last step: {'ordering the cart' if self.cart else 'declining'}")
                self.finish(order=bool(self.cart))
                return
            if self.repairing and not self.guessed and not all(self.settled(s) for s in self.need.slots):
                self.work()
            if not self.choose():
                self.repairing = True
                if not self.repair():
                    self.say(f"no basket: fits {[(s.kind, s.fits[:3]) for s in self.need.slots]} budget {self.budget} "
                             f"candidates {[sum(self.candidate(s, p) for p in s.pool) for s in self.need.slots]} "
                             f"unread {[len(self.queue(s)) for s in self.need.slots]} left {self.env.left()}")
                    if not self.guess():
                        self.decline()
                        return
                continue
            want = {s.pick: s.qty for s in self.need.slots}
            if want != self.cart:
                try:
                    self.fill_cart(want)
                except ValueError:
                    if self.remapped:
                        raise
                    self.remapped = True
                    self.say("cart call refused by its schema: reading the tools again")
                    self.tools = self.brain.read_tools(self.env.request, self.env.specs)
                continue
            self.order_tried = True
            reply = self.finish(order=True)
            if self.env.finished:
                return
            self.repairing = True
            self.say(f"order refused: {reply.notices}")
        if not self.env.finished:
            self.decline()

    def repair(self) -> bool:
        self.guessed = False
        for slot in self.need.slots:
            if slot.fits or self.spare() <= 0:
                continue
            stale = [p for p in slot.pool if self.items[p].stale and self.candidate(slot, p)]
            if stale:
                self.open_items(stale[:1])
                self.judge_known(slot)
        self.work()
        return self.choose()

    def guess(self) -> bool:
        guesses = {}
        for k, slot in enumerate(self.need.slots):
            if slot.fits:
                continue
            values = [query_words(v) for v in plain_values(slot.reqs)]
            title = lambda p: set(query_words(self.items[p].title))
            likely = lambda p: (-slot.pool.get(p, 0), -sum(all(w in title(p) for w in v) for v in values if v), self.items[p].price or 0, p)
            pool = sorted([p for p in slot.pool if self.candidate(slot, p) and self.items[p].facts is None], key=likely)
            near = sorted([p for p in slot.rejected if p not in slot.gone and self.buyable(slot, p)],
                          key=lambda p: (-strength(slot, self.items[p]), slot.problems.get(p, 9), self.items[p].price or 0, p))
            if not pool and not near:
                self.say(f"nothing to buy for {slot.kind}")
                return False

            inside = [p for p in near if all(p in ids for ids, _ in slot.supersets)]
            if not pool and slot.supersets and all(strength(slot, self.items[p]) == 0 for p in inside):
                self.say(f"proved empty: {slot.kind}")
                return False
            guesses[k] = pool[0] if pool else near[0]
        for k, pid in guesses.items():
            slot = self.need.slots[k]
            slot.rejected.discard(pid)
            slot.fits = [pid]
            slot.supersets, slot.prefixes = [], []
        self.guessed = True
        self.say(f"guessing {guesses}")
        return self.squeeze() and self.choose()

    def squeeze(self) -> bool:
        if self.budget is None:
            return True
        for _ in range(len(self.need.slots) + 1):
            if not self.over_budget():
                return True
            price = lambda p: self.items[p].price or 0
            swaps = []
            for slot in self.need.slots:
                cheaper = [p for p in slot.pool if self.buyable(slot, p) and p not in slot.gone and price(p) < price(slot.fits[0])]
                if cheaper:
                    best = min(cheaper, key=lambda p: (price(p), p))
                    swaps.append(((price(slot.fits[0]) - price(best)) * slot.qty, slot, best))
            if not swaps:
                return False
            _, slot, best = max(swaps, key=lambda s: s[0])
            slot.rejected.discard(best)
            slot.fits = [best]
            self.say(f"over budget: {slot.kind} falls back on {best} at {self.items[best].price}")
        return not self.over_budget()

    def buyable(self, slot: Slot, pid: str) -> bool:
        item = self.items.get(pid)
        return item is not None and item.stock is not False and (not slot.category or same_shelf(item.category, slot.category)) \
            and (self.budget is None or item.price is None or item.price * slot.qty <= self.budget)

    def over_budget(self) -> bool:
        if self.budget is None or not all(s.fits for s in self.need.slots):
            return False
        return sum((self.items[s.fits[0]].price or 0) * s.qty for s in self.need.slots) > self.budget

    def fill_cart(self, want: dict[str, int]) -> None:
        cart = self.tool("cart")
        if cart.style == "lines":
            if want != self.cart:
                self.run_tool("cart", {"lines": [{cart.args["line_id"]: p, cart.args["line_qty"]: q} for p, q in want.items()]})
                self.cart = dict(want)
            return
        ops = []
        for pid, held in list(self.cart.items()):
            if pid not in want:
                ops.append({"id": pid, "qty": held, "mode": cart.values.get("remove")})
        for pid, qty in want.items():
            held = self.cart.get(pid, 0)
            if held == qty:
                continue
            if cart.style == "set" and "set" in cart.values:
                ops.append({"id": pid, "qty": qty, "mode": cart.values["set"]})
            elif qty > held:
                ops.append({"id": pid, "qty": qty - held, "mode": cart.values.get("add")})
            else:
                ops.append({"id": pid, "qty": held - qty, "mode": cart.values.get("remove")})
        for op in ops:
            if self.env.left() <= 1:
                return
            reply = self.run_tool("cart", op)
            if reply.error:
                self.say(f"cart error {reply.error}")
                continue
            pid, mode = op["id"], op.get("mode")
            if mode == cart.values.get("remove"):
                self.cart[pid] = self.cart.get(pid, 0) - op["qty"]
            elif mode == cart.values.get("set"):
                self.cart[pid] = op["qty"]
            else:
                self.cart[pid] = self.cart.get(pid, 0) + op["qty"]
            if self.cart[pid] <= 0:
                self.cart.pop(pid)

    def finish(self, order: bool) -> Reply:
        if "finish" in self.tools:
            sem = {"mode": self.tools["finish"].values["order" if order else "decline"]}
            if not order:
                sem["note"] = SORRY
            return self.run_tool("finish", sem)
        role = "order" if order else "decline"
        if role not in self.tools:
            raise Finished(f"no {role} tool understood")
        return self.run_tool(role, {} if order else {"note": SORRY})

    def decline(self) -> None:
        for _ in range(3):
            if self.env.finished or self.env.left() <= 0:
                return
            self.say("decline")
            self.finish(order=False)

def unmapped_required(tools: dict[str, Tool], specs: dict[str, dict]) -> list[str]:
    out = []
    for tool in tools.values():
        schema = specs[tool.name].get("parameters") or {}
        for arg in schema.get("required") or []:
            if arg not in tool.args.values():
                values = (schema.get("properties") or {}).get(arg, {}).get("enum")
                out.append(f"{arg} of {tool.name}" + (f" (values {values})" if values else ""))
    return out

def guess_ends(specs: dict[str, dict]) -> dict[str, Tool]:
    quit_words = r"\b(decline|declining|refuse|abandon|give up|giving up|walk away|stand down|abstain|without (buying|a purchase|purchasing)|no purchase|buy(ing)? nothing|buys nothing)\b"
    soft_quit = r"\b(pass|skip|cancel|none|nothing|leave|quit|stop|no)\b"
    buy_words = r"\b(order|checkout|check out|purchase|buy|buys|settle)\b"
    changes = r"\b(add|adds|put|set|sets|remove|change|changes|update|modify|edit|adjust)\b"
    squash = lambda t: re.sub(r"[\s_\-]+", " ", str(t)).strip().lower()
    out: dict[str, Tool] = {}
    for name, spec in specs.items():
        words = squash(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)) + " " + squash(spec.get("description", ""))
        buyable = re.sub(r"\b(in|by|listing|price|sort|sorted|id|any|alphabetical) order\b|\border (by|of)\b", " ", words)
        props = (spec.get("parameters") or {}).get("properties") or {}
        texts = [p for p, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and not v.get("enum")]
        enums = {p: v["enum"] for p, v in props.items() if isinstance(v, dict) and isinstance(v.get("enum"), list)}
        quits, buys = re.search(quit_words, words), re.search(buy_words, buyable)
        two = {p: vals for p, vals in enums.items() if len(vals) == 2}
        merged = two and quits and (buys or re.search(r"\b(either|buying the|buy the)\b", words))
        if not (quits or buys and not re.search(changes + r"|\b(view|show|what is in|contents)\b", words)):
            continue
        if merged and "finish" not in out:
            for p, vals in two.items():
                said = {v: squash(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", v)) for v in vals if isinstance(v, str)}
                d = [v for v, w in said.items() if re.search(quit_words + "|" + soft_quit, w)]
                o = [v for v, w in said.items() if re.search(buy_words, w) and v not in d]
                if len(d) != 1 and len(o) == 1:
                    d = [v for v in vals if v != o[0]]
                if len(d) == 1 and len(said) == 2:
                    o = [v for v in vals if v != d[0]]
                    args = {"mode": p, **({"note": texts[0]} if texts else {})}
                    out["finish"] = Tool("finish", name, args, {"order": o[0], "decline": d[0]})
                    break
        elif quits and not merged and "decline" not in out:
            out["decline"] = Tool("decline", name, {"note": texts[0]} if texts else {})
        elif buys and not quits and "order" not in out:
            out["order"] = Tool("order", name, {})
    return out

MODELS = ["deepseek/deepseek-v4-flash"]
SPEND_CAP = 0.012

STEP_SCALE = 38
RECORD_CHARS = 2500
ROLE_ARGS = {"search": ("query", "page", "mode", "category", "max_price"), "filter": ("category", "brand", "max_price", "page", "mode"),
             "details": ("id",), "specs": ("id",), "compare": ("ids",), "cart": ("id", "qty", "mode", "lines"), "decline": ("note",),
             "finish": ("mode", "note"), "message": ("text",)}

TOOLS_PROMPT = """You map a web store's tools to the jobs of a shopping assistant. Names of tools and arguments are made up anew for every store, so judge every tool by its description and its JSON schema, and copy names and enum values exactly.
Answer with a single JSON object only. Each key is a job; its value is null when no tool does that job:
"search": {"tool", "query": argument for the words, "page": argument for further results or null, "mode": argument choosing how it searches or null, "mode_value": the value of that argument for a search by words or null, "category": argument or null, "max_price": argument or null}
"filter": a tool listing one category or shelf: {"tool", "category": argument, "brand": argument or null, "max_price": argument or null, "page": argument or null, "mode": argument or null, "mode_value": the value that makes it list a category or null}
"shelves": a tool naming the store's categories: {"tool"}
"details": a tool showing one listing in full: {"tool", "id": argument}
"specs": a tool showing the property list or datasheet of one listing, when that is not the "details" tool: {"tool", "id": argument}
"compare": a tool showing several listings' properties at once: {"tool", "ids": argument, "limit": the most ids one call takes, when the description or the schema says so, else null}
"cart": {"tool", "id": argument or null, "qty": argument or null, "mode": argument or null, "add": value or null, "remove": value or null, "set": value or null, "lines": array argument of lines or null, "line_id": key of a line's listing id or null, "line_qty": key of a line's quantity or null}
"cart_view": a tool showing the cart: {"tool"}
"order": a tool that buys the cart with no argument to say so: {"tool"}
"decline": a tool that ends without buying: {"tool", "note": argument for a message or null}
"finish": only when a single tool can either buy or decline depending on an argument, whatever that argument is called (then leave "order" and "decline" null): {"tool", "mode": argument, "order": its value for buying, "decline": its value for not buying, "note": argument or null}
"message": a tool that writes to the shopper: {"tool", "text": argument}
"status": a tool reporting usage: {"tool"}
One tool may do two jobs (a search whose mode also lists a category is both "search" and "filter")."""

FIELDS_PROMPT = """Below is a result of a store's "{role}" tool. Name the keys that hold each piece of information (keys exactly as written; null when absent). Answer with a single JSON object only:
{{"rows": key of the list of listings or null, "total": key of the number of matches or null, "next": key of the token that fetches the following page, or null, "id": key of a listing's id, "title": key of its name, "price": key of its price, "stock": key of its availability, "in_stock": [values of that key meaning it can be bought now], "category": key of its category or shelf, "brand": key of its brand or maker, "facts": key of its table of properties, "notes": key of notes or remarks from the seller, "reply": key of a person's reply text, "error": key of an error message}}
The keys of a listing are those inside one row of the list, or of the result itself when it shows one listing."""

REQUEST_PROMPT = """Read a shopper's request sent to a shopping assistant. Answer with a single JSON object only:
{"products": [{"kind": the product kind in the request's words, "quantity": units to buy (1 unless the request plainly asks for several units; alternatives joined by "or" are one unit), "requirements": [{"property": the property as the request names it, "value": the value it must have, in the request's words, "exclude": true when the value must NOT be present, "firm": false when this is only a wish}], "hints": [story details that point at a property without giving its value, each "property: detail"]}],
 "budget": the highest total price as the number written in the request, in the request's own unit and never converted ("2750 US cents" and "2750\u00a2" give 2750), or null, "cheapest": true only when the request asks for the cheapest or lowest-priced fitting item, "in_stock_only": true when only items available now are wanted, "message_limit": the most messages the assistant may send, or null, "step_limit": the most steps or moves allowed, or null,
 "references": [{"product": index of the product, "property": the property, "listing": the words the request uses to name the other listing whose value it must take, "seller": the shop or maker named as selling that listing or null, "relation": "same", "pair" (made to go with it: the store's pairing rule decides), "within" (inside some amount of it), "different", "at least" or "at most": how this property must relate to that listing's value, "margin": for "within" the amount with its unit, else null}],
 "no_substitutes": true when the request says not to offer substitutes or alternatives}
Read every sentence: requirements, references and rules can stand anywhere, the last sentence included. One entry in "products" per different product to buy.
A requirement names a property AND states a definite value for it: "the P has to be X", "only X for the P", "P: X", "make sure the P is X", "nothing with X as the P" / "any P except X" / "skip whatever has P X" (exclude; the property is P and the value X alone). A value said twice in other words is one requirement ("lidless please, no lid of any kind" is lid: lidless). Copy the property name exactly as written, all its words, however odd ("coats fastening kind", "units qz", "zz finish"), and the value exactly as written.
A requirement can also be worded as a description of the product: "bags sewn from canvas" is material: canvas; "printed with a floral motif" is pattern: floral; "with a paisley motif on it" is motif: paisley; "skip whatever has sole rubber" is sole: NOT rubber; "somewhere in the blue colour range" is color family: blue; "boots suited to a hike" is occasion: hike; "the cut must be narrow, never wide or regular" is cut: narrow (with wide and regular excluded). When the shopper says they will verify something on arrival, that thing is certainly a requirement.
A requirement is firm when it is laid down as a rule ("has to", "must", "should be", "only", "make sure", "nothing with", "P: X") or built into what the product is ("bags sewn from canvas", or something they say they will verify on arrival). It is only a wish ("firm": false) when the shopper merely says what they would like: "I want a crew neck that doesn't choke", "I'd like", "I prefer one without a front pocket", "would be great", "I'm looking for something in a cool tone".
Being available, in stock or on hand goes in "in_stock_only" and the price ceiling in "budget": neither is ever a requirement. The story is never a requirement: who it is for, what happened to the old one ("the zipper split"), where or how it will be used beyond what the product must be, what it should feel or look like without a stated value ("it shouldn't show dirt", "roomy enough for layering"). Such details go in "hints".
The product kind is the request's own words for what to buy, kept whole when it names a store category or alternatives ("garden/patio gear", "mugs or cups").
A reference tells you to take a property's value from another listing: "go by the 'X' listing for the P", "the P has to match what the 'X' listing says", "the same P as the one sold by X", "compatible with what the page of Y's sofas with depth 90 cm displays". It goes in "references" with the listing's naming words (the quoted words when quoted), never in "requirements".
Saying money is short does not ask for the cheapest."""

CHECK_PROMPT = """Read a shopper's request and answer two narrow questions, each with the request's own words as evidence. Answer with a single JSON object only:
{"cheapest": true when the shopper does ask for the lowest-priced of the fitting items (saying money is short, or naming a price ceiling, is not that), "cheapest_quote": the sentence that asks for it, copied exactly, or null,
 "products": [{"kind": the product, "units": how many units of it to buy, "units_quote": the words that say so, copied exactly, or null}]}
A product named once with "a" or "one", or with alternatives joined by "or", is one unit, and "a pair" of something worn in pairs is one unit too; several units need a number written beside it. Products in the order the request names them."""

CATEGORY_PROMPT = """A shopper wants: {kind}. Which of these store categories is the one for that kind of product? The category whose name matches the shopper's words as a whole wins: for "mugs", one named just "Mugs" beats a broader "Cups & Mugs"; for "cups or mugs", the category "Cups & Mugs" that names both beats "Mugs"; a broader category counts only when no closer one exists. A category for other things that are merely worn or used alongside it never counts. Reply with ONE JSON object: {{"category": the category exactly as written, or null when none plainly fits}}.
Categories: {labels}"""

JUDGE_PROMPT = """For each listing, settle whether EVERY requirement holds, reading the requirements against the listing's property table:
- Which field: one whose name is the property's, ignoring case and separators ("sleeve_length" = "sleeve length"); a name with one more word in front also counts ("sleeve.length" is a "length" field, "clothing.material" a "material" field). When the listing has such a field, it alone decides: other fields (an "is unisex" beside a "gender") do not. If no field of that name exists at all, a field with a shorter or longer form of the name decides ("material" for "clothing material", "fit" for "pants fit").
- Figures compare as amounts, converting units: "under 45 mm" is not met by 45 mm, "at least 4 gb" is met by 8 gb and by 4 gb; such a limit is met only by a figure, never by words alone.
- The value is met when that field holds the value's words, in any case, singular or plural: inside a longer, more specific value ("cotton spandex", "polyester cotton" and "80% cotton" all hold "cotton"; "forest green" holds "green") or among the items of a list. A different value does not meet it.
- "NOT value" is met unless such a field holds that value; a listing that does not state the property at all meets it.
- A required value with no field for its property at all is not met. The description and the title never count.
Answer with a single JSON object only: {"items": [{"id": that listing's id, "meets": true or false, "gaps": [a few words per requirement not met]}]}"""

QUESTIONS_PROMPT = """Part of what the shopper wants is left out of the request: each such want fixes one property of one product, and it comes out only if the shopper is asked about that property. The shopper's story drops hints: a concrete detail (who it is for, an age, the place or manner of use, what it must go with or survive) that a property of THAT product decides.
Each listing property shows how many of the product's listings carry it and a few values it takes. For each product below, go through its hints and pick, from THAT product's listing properties, the one property each hint is about (a hint that names or describes one of a property's values is about that property; an age or "for my daughter" points at an age or gender property, carrying or travel at a size, spills or dirt at a pattern or colour, cooking at a function, comfort or movement at a fit or sleeves, and so on). Then, if questions are left, the product's most common listing properties the request does not settle.
List up to {n} questions in all, best first, at most one per property, never one the request already settles, never price, stock, brand or quantity. Always answer, even with little to go on.
Answer with a single JSON object only: {{"questions": [{{"product": index, "property": a property exactly as that product's listings name it, "hint": the hint it answers or ""}}], "paperwork": [listing properties that are about the sale or the listing rather than the product itself (new or used, guarantees, codes)]}}"""

HEAR_PROMPT = """The assistant asked the shopper about the {prop} of the {kind}. The shopper replied: "{reply}"
The reply is wrapped in chatter about their day: go by what it requires of a property, if anything. It may speak of another property than the one asked about: go by the one it speaks of. Any way of fixing a property to a value counts, however colourful ("it is stuck on plaid", "needs to be teak, full stop", "digging my heels in on matte", "the settled pick is linen", "no budging on that"); "whatever else" and "relaxed on everything besides" do not make it less firm.
Answer with a single JSON object only: {{"property": the property the reply requires something of, as it names it, without "of the <product>", or null, "value": the value required for it, with any limit in its words ("at least 4 gb", "at most 3.5 ghz") but nothing else ("anything apart from linen gets thrown out" requires linen), or null, "exclude": true only when the shopper does not want that value itself ("anything but teak", "keep teak away", "teak is a dealbreaker", "no teak"), and false when that value is what they insist on, however negatively it is put ("teak is beyond bargaining", "the frame being teak is not for negotiation", "no moving off teak", "I won't swap teak for anything"), "no_preference": true when the shopper does not mind, "unclear": true when the reply does not answer (did not follow, did not catch it, asks to repeat or to ask one thing at a time), "one_at_a_time": true only when it asks for fewer things per message ("one question per message, please", "let's go through those separately"), "new_budget": a new highest total price stated, or null, "reference": null, or when the value is to be taken from another listing ("what the page of Orlo's sofas with depth 90 cm displays", "compatible with what the page of Orlo's sofas lists", "whatever the floor lamps sold by Orlo state", "the value shown for the rugs from Orlo that have a fringe", "or wherever it leads"): {{"listing": the words naming that listing, "seller": the shop named, "relation": "same" for matching or compatible with it, "pair" for made to go with it or paired with it (the store's own pairing rule then decides), "within" for inside some amount of it ("inside 3.77 inches of"), "different" for anything but it or never the same as it, "at least" for not less than it, "at most" for not more than it, "margin": for "within" the amount with its unit ("3.77 inches"), else null}}}}"""

NOTICE_PROMPT = """A shopper or store sent this notice while a shopping assistant worked: "{text}"
The assistant's candidate listings: {picks}
The notice is wrapped in chatter about their day and is often figurative. Answer with a single JSON object only: {{"budget": the new spending ceiling if the notice sets one, as the number written there and never converted ("640 cts" and "271 pennies" give those numbers), else null, "gone": [ids of listings above it names as no longer buyable], "changed": [ids of listings above it names as changed in price or properties], "pick_gone": true when, without naming it, it says what the assistant picked, suggested or recommended can no longer be bought ("went out like a candle", "the last unit walked off", "emptied out", "the buy link stopped working", "vanished from the racks", "none remaining", "no longer purchasable"), "pick_changed": true when, without naming it, it says the price or details of what the assistant picked or suggested are different now ("wears a new tag", "has been repriced", "the amount has moved", "the rate was altered", "reads otherwise today"; any remark that the amount, rate, tag or figure of what was suggested is now new or different counts), "dropped": [properties the shopper no longer requires ("forget about handle style", "strike finish type from the requirements", "ease up on lid size", "letting go of grip")], "added": [{{"property": a property the shopper now requires, "value": its value, "exclude": true when that value is now ruled out}}], "confirm": true when it only confirms something without changing it}}
A correction from a seller about a listing is a change to it, even when it does not say the new value. A new or different value for a property is added (and replaces the old one)."""

NOTE_PROMPT = """A shopping assistant needs the {prop} for a {kind}, and is following store listings that say what it must be. It opened the listing below, looking for the one named "{target}".
Read the listing's notes. When they state the {prop} needed (for something bought or paired with this listing), or what its {prop} "goes together with", give that as "value". When they instead send the reader on to another listing for the {prop} ("for the {prop}, refer to the Sofas product from Orlo, a grey fabric piece"), give the words naming that listing as "next" and its shop or maker as "seller". When they say nothing about the {prop}, give all as null.
Answer with a single JSON object only: {{"value": the value or null, "next": the naming words or null, "seller": the shop or maker or null}}"""

def seller_in(words: str) -> str:
    found = re.match(r"\W*(?:the page of\s+)?([A-Z][\w-]*(?:\s[A-Z][\w-]*)?)['\u2019]s\b", words) or \
        re.search(r"\b(?:sold by|from)\s+([A-Z][\w-]*(?:\s[A-Z][\w-]*)?)\b", words)
    return found.group(1) if found else ""

PAIR_PROMPT = """A store's notes may give a rule for which {prop} goes with which ("our {prop} pairings: A with B, C with C", "A goes together with B"). Notes: {notes}
Under that rule, what {prop} goes with {value}? Answer with a single JSON object only: {{"value": the {prop} that goes with {value}, or null when no note gives such a rule}}"""

def numbers_in(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text or "")]

def relation_of(value: Any) -> str:
    return value if value in ("same", "different", "at least", "at most", "pair", "within") else "same"

def json_object(text: Any) -> dict | None:
    if not isinstance(text, str):
        return None
    body = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    span = body[body.find("{"): body.rfind("}") + 1] if "{" in body else ""

    mended = re.sub(r"//[^\n\"]*\n", "\n", span)
    mended = re.sub(r",\s*([}\]])", r"\1", mended)
    mended = re.sub(r"\bNone\b", "null", re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", mended)))
    for candidate in (body, span, mended):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None

def written_amount(value: Any, text: str) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    figures = {float(f.replace(",", "")) for f in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
    if not figures or value in figures:
        return value
    for scaled in (value * 100, value / 100):
        if round(scaled, 2) in figures:
            return int(scaled) if float(scaled).is_integer() else scaled
    return value

def prop_name(key: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(key).lower()).split())

def kin(name: str) -> frozenset:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in prop_name(name).split())

def kindred(a: str, b: str) -> bool:
    x, y = kin(a), kin(b)
    return bool(x) and bool(y) and (x <= y or y <= x)

def names_prop(key: str, prop: str) -> bool:
    k, p = prop_name(key), prop_name(prop)
    return bool(p) and (k == p or k.endswith(" " + p) and len(k.split()) == len(p.split()) + 1)

def singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word

def holds(text: str, value: str) -> bool:
    if " or " in f" {prop_name(value)} ":
        return any(holds(text, part) for part in f" {prop_name(value)} ".split(" or ") if part.strip())
    v = " ".join(singular(w) for w in prop_name(value).split())
    t = " ".join(singular(w) for w in prop_name(text).split())
    return bool(v) and f" {v} " in f" {t} "

def phrased(request: str, slots: list[Slot]) -> None:
    words = prop_name(request)
    for slot in slots:
        m = re.match(r"(.+?)\s+in\s+(?:a|an|the)\s+((?:[\w'&+-]+\s+){0,3}?[\w'&+-]+)\s+([a-z]+)$", slot.kind.strip(), re.I)
        if m and m.group(3).lower() not in LINKING and not any(kindred(r.label, m.group(3)) for r in slot.reqs):
            slot.kind = m.group(1)
            slot.reqs.append(Req(m.group(3).lower(), m.group(2), source="phrase"))
        for req in slot.reqs:

            neg = re.match(r"(?:no|not|without) (.+)", prop_name(req.value))
            if req.exclude and neg:
                req.value = neg.group(1)
            if req.exclude or not prop_name(req.label):
                continue
            m = re.search(r"\bin (?:a|an|the) ((?:\S+ ){0,3}?\S+) " + re.escape(prop_name(req.label)) + r"\b", words)
            if m and m.group(1) not in LINKING and not holds(m.group(1), req.value) and not holds(req.value, m.group(1)):
                req.value = m.group(1)

def spoken_property(text: str, names: set, value: str = "") -> str:
    rest = re.sub(re.escape(value), " ", text, flags=re.I) if value else text
    found = [n for n in names if len(n) > 2 and not n.isdigit() and holds(rest, n)]
    return max(found, key=lambda n: (len(n.split()), len(n)), default="")

def field_spellings(shelf: list[dict]) -> dict[str, str]:
    names = {prop_name(key) for facts in shelf for key in facts}
    met: dict[tuple, list[int]] = {}
    seen: dict[str, int] = {}
    snake: set = set()
    for facts in shelf:
        for key, val in facts.items():
            name = prop_name(key)
            seen[name] = seen.get(name, 0) + 1
            if "_" in str(key):
                snake.add(name)
            for other, oval in facts.items():
                if prop_name(other) != name:
                    tally = met.setdefault((name, prop_name(other)), [0, 0])
                    tally[0] += prop_name(val) == prop_name(oval)
                    tally[1] += 1
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(merged_name(name, names), []).append(name)
    out: dict[str, str] = {}
    for merged, group in groups.items():
        rest = sorted(group, key=lambda o: (o.startswith("fa "), -seen.get(o, 0), o not in snake, o))
        while rest:
            head, alike = rest[0], [rest[0]]
            for o in rest[1:]:
                same, both = met.get((o, head), [0, 0])
                if both == 0 or same >= 0.8 * both:
                    alike.append(o)
            for o in alike:
                out[o] = head
            rest = [o for o in rest if o not in alike]
    return out

def fill_one(values: dict, needed: tuple, enum: list | None) -> None:
    if not enum:
        return
    missing = [k for k in needed if k not in values]
    free = [v for v in enum if v not in values.values()]
    if len(missing) == 1 and len(free) == 1 and len(enum) == len(values) + 1:
        values[missing[0]] = free[0]

def same_shelf(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and " ".join(a.lower().split()) == " ".join(b.lower().split())

def merged_name(key: str, names: set) -> str:
    words = prop_name(key).split()
    if len(words) > 1 and words[-1] == "type" and " ".join(words[:-1]) in names:
        words = words[:-1]
    for i in range(len(words) - 1, 0, -1):
        tail = " ".join(words[i:])
        if tail in names and tail not in LINKING:
            return tail
    return " ".join(words)

def field_class(key: str, prop: str) -> str:
    words = lambda text: {singular(w) for w in prop_name(text).split()} - LINKING
    k, p = words(key), words(prop)
    if k and k == p:
        return "own"
    if names_prop(key, prop):
        return "front"
    return "short" if k and k < p else "long" if p and p < k else ""

def akin(value: str, text: str, prop: str = "") -> bool:
    own = set(prop_name(prop).split())
    return any(v[:3] == t[:3] or v in t or t in v for v in prop_name(value).split() if len(v) > 2 and v not in own
               for t in prop_name(text).split() if len(t) > 2)

def evidence(req: Req, item: Item) -> int:
    own, front, short, long, other = [], [], [], [], []
    for key, val in (item.facts or {}).items():
        {"own": own, "front": front, "short": short, "long": long}.get(field_class(key, req.label), other).append(str(val))
    hit = lambda vals: any(holds(v, req.value) for v in vals)
    if hit(own) or (not own and hit(front)) or holds(item.category or "", req.value):
        return 3
    if hit(front) or (not own and not front and (hit(short) or hit(long))):
        return 2
    named = own + front + short + long
    if re.search(r"\d", req.value):
        return 2 if named else 1
    if not named:
        return 1
    if hit(other) or holds(item.title, req.value) or (own and front):
        return 0
    return 1 if any(akin(req.value, v, req.label) for v in named) else 0

def strength(slot: Slot, item: Item) -> int:
    return min((evidence(r, item) for r in slot.reqs if not r.exclude), default=3)

def shown(slot: Slot, item: Item) -> int:
    texts = [str(v) for v in (item.facts or {}).values()] + [item.title or ""]
    return sum(any(holds(t, r.value) for t in texts) for r in slot.reqs if not r.exclude)

class Models:

    def __init__(self, models: list[str]):
        self.models = list(models)
        self.client = ProxyClient(timeout=60, max_retries=2)
        self.spent = 0.0
        self.calls = 0
        self.cap = SPEND_CAP
        self.log: list[str] = []
        self.thinkers: set[str] = set()
        self.last_text = ""

    def ask(self, system: str, user: str, max_tokens: int = 600) -> dict | None:
        self.last_text = ""
        if self.spent >= self.cap:
            self.log.append("spend cap reached")
            return None

        for model in [m for m in self.models for _ in range(2)]:
            if self.spent >= self.cap:
                return None

            body = {"model": model, "temperature": 0, "max_tokens": max(max_tokens * 3, 1500),
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if model not in self.thinkers:

                body["reasoning"] = {"enabled": False}
            result = self.client.post_verbose("/inference/chat/completions", json_data=body)
            if result.data is None and model not in self.thinkers and (result.error or {}).get("status") == 400:
                self.thinkers.add(model)
                body.pop("reasoning")
                result = self.client.post_verbose("/inference/chat/completions", json_data=body)
            data = result.data
            if not isinstance(data, dict) or not data.get("choices"):
                self.log.append(f"{model}: no answer {str(result.error)[:160]}")
                continue
            self.calls += 1
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.spent += usage["cost"]
            choice = data["choices"][0]
            content = (choice.get("message") or {}).get("content")
            self.last_text = content if isinstance(content, str) else ""
            answer = json_object(content)
            if answer is not None:
                return answer
            self.log.append(f"{model}: unreadable ({choice.get('finish_reason')}) {str(content)[:200]!r} for {system[:50]!r}")
        return None

UNASKED = {"options", "option", "model", "title", "name", "description", "price", "brand", "condition"}

class ModelBrain(Brain):
    def __init__(self, models: list[str] | None = None):
        self.m = Models(models or MODELS)
        self.verdicts: dict[tuple, tuple[bool, list[str]]] = {}
        self.specs: dict[str, dict] = {}
        self.kinds: list[str] = []
        self.shelf_seen: dict[int, dict[str, Item]] = {}

    def set_limits(self, max_steps: int) -> None:
        self.m.cap = SPEND_CAP * max(1.0, max_steps / STEP_SCALE)

    def read_tools(self, request: str, specs: dict[str, dict]) -> dict[str, Tool]:
        self.specs = specs
        listing = json.dumps([{"name": s["name"], "description": s.get("description", ""), "parameters": s.get("parameters")}
                              for s in specs.values()], separators=(",", ":"))
        tools, gaps, complaint = {}, ["nothing read"], ""
        for _ in range(3):
            got = self.map_tools(self.m.ask(TOOLS_PROMPT, listing + complaint, max_tokens=900) or {}, specs)
            missing = unmapped_required(got, specs)
            if "finish" not in got and not ("order" in got and "decline" in got):
                missing.append("the tool that closes the errand, as \"order\" and \"decline\" or as \"finish\" with both of its values")
            if not ({"details", "specs", "compare"} & set(got)):
                missing.append("the tool that shows a listing's record, as \"details\" or \"specs\"")
            if not ({"search", "filter"} & set(got)):
                missing.append("the tool that finds listings, as \"search\"")
            if "cart" not in got:
                missing.append("the tool that puts a listing in the cart, as \"cart\"")
            used = {t.name for t in got.values()}

            if "message" not in got and any(name not in used and [p.get("type") for p in ((s.get("parameters") or {}).get("properties") or {}).values()] == ["string"]
                                            for name, s in specs.items()):
                missing.append("the tool that writes to the shopper, as \"message\", if one of them does")
            if got and (not tools or len(missing) < len(gaps)):
                tools, gaps = got, missing
            if tools and not gaps:
                break
            complaint = ("\n\nA previous answer was wrong, it left out these required arguments: " + "; ".join(missing or gaps)
                         + ". Name each one under its job. A tool that buys or declines according to an argument is the \"finish\" job.")
        if "finish" not in tools and not ("order" in tools and "decline" in tools):
            for role, tool in guess_ends(specs).items():
                if role not in tools and not ("finish" in tools and role in ("order", "decline")):
                    tools[role] = tool
            if "finish" in tools:
                tools.pop("order", None), tools.pop("decline", None)
        for role in ("shelves", "filter", "compare", "cart_view", "status"):
            if role in tools and unmapped_required({role: tools[role]}, specs):
                tools.pop(role)
        return tools

    def map_tools(self, got: dict, specs: dict[str, dict]) -> dict[str, Tool]:
        tools: dict[str, Tool] = {}
        for role, spec in got.items():
            if not isinstance(spec, dict) or spec.get("tool") not in specs:
                continue
            props = (specs[spec["tool"]].get("parameters") or {}).get("properties") or {}
            args = {k: v for k, v in spec.items() if k in ROLE_ARGS.get(role, ()) and isinstance(v, str) and v in props}
            tool = Tool(role, spec["tool"], args)
            if role in ("search", "filter") and spec.get("mode_value") is not None and "mode" in args:
                tool.values["keyword" if role == "search" else "filter"] = spec["mode_value"]
            if role == "compare":
                limit = spec.get("limit")
                cap = (props.get(args.get("ids", ""), {}) or {}).get("maxItems")
                tool.limit = max(1, min(int(limit) if isinstance(limit, int) else 5, cap or 99))
            if role == "cart":
                for k in ("add", "remove", "set"):
                    if spec.get(k) is not None:
                        tool.values[k] = spec[k]
                fill_one(tool.values, ("add", "remove"), (props.get(args.get("mode", ""), {}) or {}).get("enum"))
                if "lines" in args:
                    tool.args["line_id"], tool.args["line_qty"] = spec.get("line_id") or "id", spec.get("line_qty") or "qty"
                    tool.style = "lines"
                else:
                    tool.style = "set" if "set" in tool.values else "add"
            if role == "finish":
                ends = {k: spec[k] for k in ("order", "decline") if spec.get(k) is not None}
                fill_one(ends, ("order", "decline"), (props.get(args.get("mode", ""), {}) or {}).get("enum"))
                if len(ends) < 2:
                    continue
                tool.values = ends
            tools[role] = tool
        if "finish" in tools and "order" in tools:
            tools.pop("finish")
        return tools

    def read_fields(self, role: str, data: Any) -> Fields:
        if isinstance(data, dict) and len(data) == 1 and isinstance(next(iter(data.values())), str):
            key = next(iter(data))
            return Fields(reply=key) if role == "message" else Fields(error=key)
        sample = data
        if isinstance(data, dict):
            sample = {k: (v[:2] if isinstance(v, list) else v) for k, v in data.items()}
        got = self.m.ask(FIELDS_PROMPT.format(role=role), json.dumps(sample)[:RECORD_CHARS], max_tokens=400) or {}
        names = {k: v for k, v in got.items() if isinstance(v, str)}
        stock = got.get("in_stock") if isinstance(got.get("in_stock"), list) else [True]
        return Fields(**{k: names.get(k, "") for k in Fields.__dataclass_fields__ if k not in ("in_stock", "lines", "line_id", "line_qty")}, in_stock=stock)

    def category(self, slot: Slot, labels: list[str]) -> str | None:
        if len(labels) == 1:
            return labels[0]
        got = self.m.ask("You match a product to a store's categories.",
                         CATEGORY_PROMPT.format(kind=slot.kind, labels=json.dumps(labels[:300])), max_tokens=100) or {}
        return got.get("category") if got.get("category") in labels else None

    def read_request(self, request: str) -> Need:
        got = self.m.ask(REQUEST_PROMPT, request, max_tokens=900) or {}
        if not isinstance(got.get("products"), list) or not got["products"]:

            got = self.m.ask(REQUEST_PROMPT, request + "\n\n(Answer with one valid JSON object and nothing else.)", max_tokens=1200) or got
        same = [str(r.get("value")) for p in got.get("products") or [] if isinstance(p, dict) for r in p.get("requirements") or []
                if isinstance(r, dict) and r.get("value") and prop_name(r.get("property")) == prop_name(r.get("value"))]
        if same:
            got = self.m.ask(REQUEST_PROMPT, request + "\n\nA previous answer was wrong, it gave the same words as property and as value for: "
                             + "; ".join(same) + ". The property says what is described, the value says which one it is.", max_tokens=900) or got
        slots = []
        self.kinds = []
        for p in got.get("products") or []:
            if not isinstance(p, dict):
                continue
            reqs, prefs, hints = [], [], [str(h) for h in p.get("hints") or []]
            for r in p.get("requirements") or []:
                if not (isinstance(r, dict) and r.get("property") and r.get("value")):
                    continue
                prop, value = str(r["property"]), str(r["value"])
                if prop_name(prop) in ("availability", "stock", "stock status", "price", "budget", "cost", "total price"):
                    continue

                if len(prop_name(prop).split()) > 4 or prop_name(value) in ("true", "false", "yes", "no"):
                    hints.append(f"{prop}: {value}")
                    continue

                if r.get("firm") is False or len(prop_name(value).split()) > 4 or re.match(r"(for|to|that|so|with)\b", prop_name(value)):
                    prefs.append(Req(prop, value, bool(r.get("exclude"))))
                    hints.append(f"{prop}: {value}")
                else:
                    reqs.append(Req(prop, value, bool(r.get("exclude"))))

            for r in sorted([r for r in reqs if not r.exclude], key=lambda r: len(prop_name(r.value).split())):
                if any(q is not r and not q.exclude and prop_name(q.label) == prop_name(r.label)
                       and len(prop_name(q.value).split()) <= len(prop_name(r.value).split()) for q in reqs):
                    reqs.remove(r)
            qty = p.get("quantity") if isinstance(p.get("quantity"), int) and p.get("quantity") > 0 else 1
            slots.append(Slot(str(p.get("kind") or "item"), qty, reqs, hints=hints, prefs=prefs))
        if not slots:
            slots = [Slot("item", 1, [])]
        self.kinds = [s.kind for s in slots]
        refs = []
        for r in got.get("references") or []:
            if isinstance(r, dict) and r.get("property") and r.get("listing"):
                k = r.get("product") if isinstance(r.get("product"), int) and 0 <= r.get("product") < len(slots) else 0
                refs.append(Ref(k, str(r["property"]), str(r["listing"]), seller=str(r.get("seller") or seller_in(str(r["listing"]))), relation=relation_of(r.get("relation")), margin=str(r.get("margin") or "")))

                slots[k].reqs = [q for q in slots[k].reqs if prop_name(q.label) != prop_name(r["property"])]
        number = lambda v: v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        cheapest = bool(got.get("cheapest"))
        check = self.m.ask(CHECK_PROMPT, request, max_tokens=300) or {}
        quoted = lambda q: isinstance(q, str) and len(q) > 3 and prop_name(q) in prop_name(request)
        if check.get("cheapest") is True and quoted(check.get("cheapest_quote")):
            cheapest = True
        counted = check.get("products") if isinstance(check.get("products"), list) else []
        if len(counted) == len(slots):
            for slot, c in zip(slots, counted):
                units = c.get("units") if isinstance(c, dict) else None

                if isinstance(units, int) and units > 0 and units != slot.qty and (units == 1 or quoted(c.get("units_quote")) and COUNTED.search(str(c.get("units_quote")))):
                    slot.qty = units
        return Need(slots, written_amount(got.get("budget"), request), cheapest, number(got.get("message_limit")), refs,
                    in_stock=got.get("in_stock_only") is not False, no_names=got.get("no_substitutes") is not False,
                    step_limit=got.get("step_limit") if isinstance(got.get("step_limit"), int) else None)

    def judge(self, slot: Slot, item: Item) -> tuple[bool, list[str]]:
        return self.judge_many(slot, [item]).get(item.id, (False, ["not judged"]))

    def code_judge(self, slot: Slot, item: Item, written: set | None = None) -> tuple[bool, list[str]] | None:
        problems, unsure = [], False
        facts = item.facts or {}
        for req in slot.reqs:
            named = [str(v) for k, v in facts.items() if names_prop(k, req.label)]
            has = any(holds(v, req.value) for v in named)
            if req.exclude:
                if any(holds(str(v), req.value) for v in facts.values()):
                    problems.append(f"{req.label} is {req.value}")
                elif not named and written and (prop_name(req.label), "") in written:

                    problems.append(f"{req.label} not shown")
                elif not named:
                    unsure = True
            elif evidence(req, item) == 0:
                problems.append(f"{req.label} is {', '.join(named)[:60]}, not {req.value}")
            elif not has and holds(item.category or "", req.value):
                pass
            elif not has:
                if named and written and (prop_name(req.label), prop_name(req.value)) in written:
                    problems.append(f"{req.label} is {', '.join(named)[:60]}, not {req.value}")
                else:
                    unsure = True
        if problems:
            return (False, problems)
        return None if unsure else (True, [])

    def written_values(self, slot: Slot, items: list[Item]) -> set:
        out = set()
        for item in items:
            for req in slot.reqs:
                named = [str(v) for k, v in (item.facts or {}).items() if names_prop(k, req.label)]
                if req.exclude and named:
                    out.add((prop_name(req.label), ""))
                elif not req.exclude and any(holds(v, req.value) for v in named):
                    out.add((prop_name(req.label), prop_name(req.value)))
        return out

    def judge_many(self, slot: Slot, items: list[Item]) -> dict[str, tuple[bool, list[str]]]:
        sig = tuple((r.label, r.value, r.exclude) for r in slot.reqs)
        out, ask = {}, []
        seen = self.shelf_seen.setdefault(id(slot), {})
        seen.update({it.id: it for it in items})
        written = self.written_values(slot, list(seen.values()))
        for item in items:
            key = (item.id, sig, json.dumps(item.facts, sort_keys=True)[:400])
            verdict = self.code_judge(slot, item, written)
            if verdict is not None:
                self.verdicts[key] = out[item.id] = verdict
            elif key in self.verdicts:
                out[item.id] = self.verdicts[key]
            else:
                ask.append((key, item))
        chunks = [ask[i:i + 4] for i in range(0, len(ask), 4)]
        while chunks:
            chunk = chunks.pop(0)
            reqs = [f"{r.label}: {'NOT ' if r.exclude else ''}{r.value}" for r in slot.reqs]
            records = [{"id": it.id, "title": it.title, "category": it.category, "properties": it.facts} for _, it in chunk]
            got = self.m.ask(JUDGE_PROMPT, json.dumps({"requirements": reqs, "listings": records})[:RECORD_CHARS * 3],
                             max_tokens=150 + 120 * len(chunk)) or {}
            answers = {a.get("id"): a for a in got.get("items") or [] if isinstance(a, dict)}
            if not answers and self.m.last_text:
                for piece in re.findall(r"\{[^{}]*\}", self.m.last_text):
                    pid = re.search(r'"id"\s*:\s*"([^"]+)"', piece)
                    ok = re.search(r'"meets"\s*:\s*(true|false)', piece)
                    if pid and ok:
                        answers[pid.group(1)] = {"meets": ok.group(1) == "true", "gaps": ["(read from a broken answer)"]}
            missed = []
            for key, it in chunk:
                a = answers.get(it.id)
                if a is not None:
                    self.verdicts[key] = out[it.id] = (bool(a.get("meets")), [str(p) for p in a.get("gaps") or []])
                else:
                    missed.append((key, it))
            if missed and len(chunk) > 1:

                chunks += [[m] for m in missed]
        return out

    def questions(self, need: Need, keys: dict[int, list[str]]) -> list[tuple[int, str]]:
        n = need.msg_budget if need.msg_budget is not None else 2
        if n <= 0:
            return []
        products = [{"product": k, "kind": s.kind, "settled": [r.label for r in s.reqs], "hints": s.hints,
                     "listing_properties": keys.get(k, [])[:40]} for k, s in enumerate(need.slots)]
        got = self.m.ask(QUESTIONS_PROMPT.format(n=n), json.dumps(products), max_tokens=420) or {}
        paperwork = {prop_name(p) for p in got.get("paperwork") or [] if isinstance(p, str)}
        known = {k: [prop_name(re.sub(r" \(on \d+\)(?::.*)?$", "", key)) for key in keys.get(k, [])] for k in range(len(need.slots))}
        known = {k: [p for p in names if p not in paperwork] or names for k, names in known.items()}
        settled_all = {k: {prop_name(r.label) for r in s.reqs} for k, s in enumerate(need.slots)}
        out: list[tuple[int, str, bool]] = []
        for k, s in enumerate(need.slots):
            for pref in s.prefs:

                wish = prop_name(pref.label)
                prop = next((key for key in known[k] if names_prop(key, wish) or names_prop(wish, key)), None) or \
                    next((key for key in known[k] if wish in key or key in wish), None) or (None if known[k] else wish)
                if prop and prop not in settled_all[k] and all((k, prop) != (j, q) for j, q, _ in out):
                    out.append((k, prop, True))
        for q in got.get("questions") or []:
            if not (isinstance(q, dict) and isinstance(q.get("product"), int) and 0 <= q["product"] < len(need.slots)
                    and q.get("property")):
                continue
            k, prop = q["product"], prop_name(q["property"])

            if known[k] and prop not in known[k]:
                owner = next((j for j in known if prop in known[j]), None)
                if owner is None:
                    continue
                k = owner
            settled = {prop_name(r.label) for r in need.slots[k].reqs}

            if q.get("hint") and prop in known[k] and prop not in settled and all((k, prop) != (j, p) for j, p, _ in out) \
                    and sum(hinted for _, _, hinted in out) < max(1, n // 2):
                out.append((k, prop, True))

        for depth in range(40):
            if len(out) >= n:
                break
            for k in known:
                fresh = [p for p in known[k] if p not in settled_all[k] and p not in UNASKED and all((k, p) != (j, q) for j, q, _ in out)]
                if fresh and depth < len(known[k]) and len(out) < n:
                    out.append((k, fresh[0], False))

        order, rounds = [], {}
        for k, prop, hinted in out:
            rounds[(k, hinted)] = rounds.get((k, hinted), 0) + 1
            order.append((not hinted, rounds[(k, hinted)], len(order), k, prop))
        return [(k, prop) for _, _, _, k, prop in sorted(order)][:n]

    def hear(self, slot: Slot, label: str, reply: str) -> dict:
        got = self.m.ask("You read a shopper's reply.", HEAR_PROMPT.format(prop=label, kind=slot.kind, reply=reply[:600]),
                         max_tokens=150) or {}
        out: dict = {"unclear": bool(got.get("unclear")), "one_at_a_time": bool(got.get("one_at_a_time"))}

        named = re.split(r"\s+of\s+(?:the|your|my|our)\b", str(got.get("property") or ""))[0].strip()
        if not named or "'" in named or not set(query_words(named)) <= set(query_words(reply)):
            named = label
        ref = got.get("reference")
        if isinstance(ref, dict) and ref.get("listing"):
            out["refs"] = [Ref(0, named, str(ref["listing"]), seller=str(ref.get("seller") or seller_in(str(ref["listing"]))), relation=relation_of(ref.get("relation")), margin=str(ref.get("margin") or ""))]
        elif got.get("value") and not got.get("no_preference"):
            out["needs"] = [Req(named, str(got["value"]), bool(got.get("exclude")), source="revealed")]
        elif got.get("no_preference"):
            out["free"] = [label]
        if isinstance(got.get("new_budget"), (int, float)):
            out["budget"] = got["new_budget"]
        return out

    def read_notice(self, text: str, picks: dict[str, Item]) -> dict:
        listed = [{"id": pid, "title": item.title} for pid, item in picks.items()]
        first = next(iter(picks), None)
        named = [pid for pid, item in picks.items() if item.title and holds(text, item.title)]
        got = self.m.ask("You read store notices.", NOTICE_PROMPT.format(text=text[:600], picks=json.dumps(listed)),
                         max_tokens=200) or {}
        added = [Req(str(a["property"]), str(a["value"]), bool(a.get("exclude")), source="revealed")
                 for a in got.get("added") or [] if isinstance(a, dict) and a.get("property") and a.get("value")]
        return {"budget": written_amount(got.get("budget"), text),
                "gone": [p for p in got.get("gone") or [] if p in picks] or
                        (named or [first] if got.get("pick_gone") and first else []),
                "changed": [p for p in got.get("changed") or [] if p in picks] or
                           (named or [first] if got.get("pick_changed") and first else []) or
                           (named if not got.get("gone") and not got.get("pick_gone") and not got.get("confirm") else []),
                "confirm": bool(got.get("confirm")),

                "dropped": [str(d) for d in got.get("dropped") or [] if d and set(query_words(d)) <= set(query_words(text))], "added": added}

    def read_note(self, ref: Ref, item: Item, target: str = "") -> dict:
        if not item.raw and not item.notes and not item.facts:
            return {}

        facts_key = next((k for k, v in (item.raw or {}).items() if v is item.facts), None)
        texts = " ".join(str(v) for k, v in (item.raw or {}).items() if k != facts_key) + " " + (item.notes or "")
        if not set(query_words(ref.label)) <= set(query_words(texts)):
            self.m.log.append(f"note {item.id}: no mention of {ref.label!r} (raw keys {list((item.raw or {}))})")
            return {}

        record = {k: (v[:400] if isinstance(v, str) else v) for k, v in (item.raw or {}).items()} or \
            {"title": item.title, "notes": item.notes, "properties": item.facts}
        kind = self.kinds[ref.slot] if ref.slot < len(self.kinds) else "product"
        got = self.m.ask("You follow a reference between store listings.",
                         NOTE_PROMPT.format(prop=ref.label, kind=kind, target=target or ref.target) + "\n" +
                         json.dumps(record)[:RECORD_CHARS], max_tokens=150) or {}
        self.m.log.append(f"note {item.id}: model said {got}")
        if got.get("value"):
            return {"value": str(got["value"])}
        if got.get("next"):
            return {"next": str(got["next"]), "seller": str(got.get("seller") or "")}
        return {}

    def paired(self, prop: str, value: str, notes: list[str]) -> str | None:
        got = self.m.ask("You read a store's rules.", PAIR_PROMPT.format(prop=prop, value=value, notes=json.dumps(notes[:6])[:2400]),
                         max_tokens=120) or {}
        return str(got["value"]) if got.get("value") else None

    def pick_reference(self, ref: Ref, rows: list[Item], target: str = "") -> list[str]:
        words = query_words(target or ref.target)
        seller = set(query_words(ref.seller))

        def score(item: Item) -> tuple:
            text = query_words(f"{item.brand or ''} {item.title} {item.text}")
            hits = [text.index(w) for w in words if w in text]
            return (not (seller and seller <= set(text)), -len(hits), sum(hits), item.id)
        return [i.id for i in sorted(rows, key=score)]

def run(problem: dict, brain: Brain, clock_start: float | None = None) -> tuple[Engine, str | None]:
    engine = Engine(problem, brain, monotonic() if clock_start is None else clock_start)
    failure = None
    try:
        engine.run()
    except Finished:
        pass
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        engine.say(f"engine failure {failure}")
        try:
            if not engine.env.finished and engine.env.left() > 0 and engine.tools:
                engine.decline()
        except Exception:
            pass
    return engine, failure

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    brain = ModelBrain()
    engine, failure = run(problem_data, brain)
    return [{"steps": engine.env.step, "finished": engine.env.finished, "failure": failure,
             "model_calls": brain.m.calls, "model_spend": round(brain.m.spent, 5)}]
