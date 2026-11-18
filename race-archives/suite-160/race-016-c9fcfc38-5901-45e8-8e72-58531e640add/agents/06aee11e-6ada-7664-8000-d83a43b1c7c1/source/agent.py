import json
import random
import re
import time
from typing import Any
import math
from dataclasses import dataclass, field
import sys
import traceback
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
ENV_PATH = '/environment/call'
LLM_PATH = '/inference/chat/completions'
ENV_RESENDS = 2
LLM_PAUSES = (3, 8, 15)
LLM_PAUSE_BUDGET = 30
SOFT_CAP = 0.05
HARD_CAP = 0.065
HEAVY_FIRST_ROOM = True
HEAVY_ROOM = 0.02

def _ct_groups(text: str):
    depth, start, quoted, escaped = (0, -1, False, False)
    for at, ch in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                quoted = False
        elif ch == '"' and depth:
            quoted = True
        elif ch == '{':
            start = at if depth == 0 else start
            depth += 1
        elif ch == '}' and depth:
            depth -= 1
            if depth == 0:
                yield (start, at + 1)

def first_json(text: Any) -> dict | None:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    text = text.strip()
    try:
        whole = json.loads(text)
        if isinstance(whole, dict):
            return whole
    except ValueError:
        pass
    for start, end in _ct_groups(text):
        try:
            found = json.loads(text[start:end])
        except ValueError:
            continue
        if isinstance(found, dict):
            return found
    return None

def _ct_args_whole(call: Any) -> bool:
    fn = (call or {}).get('function') if isinstance(call, dict) else None
    if not isinstance(fn, dict) or not fn.get('name'):
        return False
    args = fn.get('arguments')
    if args is None or isinstance(args, dict) or (isinstance(args, str) and (not args.strip())):
        return True
    try:
        return isinstance(json.loads(args), dict) if isinstance(args, str) else False
    except ValueError:
        return False

def _ct_strings(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all((isinstance(v, str) for v in value))

def _ct_spoken(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v.strip()]
    if isinstance(value, dict):
        return [v for v in value.values() if isinstance(v, str) and len(v.split()) >= 2]
    return []
STRUCT_TEXT = True
FACT_CONT = True
_CT_FIELD = re.compile('(?:^|\\s\\|\\s)([a-z][a-z0-9_]{0,30}):\\s')
_CT_FACT = re.compile('^\\s*([^=;]{1,40}?)\\s*=\\s*([^;]*?)\\s*$')

def _ct_scalar(text: str) -> Any:
    if text in ('True', 'False'):
        return text == 'True'
    if text == 'None':
        return None
    if re.fullmatch('-?\\d+', text):
        return int(text)
    if re.fullmatch('-?\\d+\\.\\d+', text):
        return float(text)
    parts = text.split(';')
    facts = [_CT_FACT.match(x) for x in parts]
    if len(parts) >= 2 and all(facts):
        return {m.group(1): m.group(2) for m in facts}
    if FACT_CONT and len(parts) >= 2 and facts[0] and (sum((1 for m in facts if m)) >= 2):
        table, last = ({}, None)
        for x in parts:
            k, eq, v = x.partition('=')
            if eq and k.strip():
                last = k.strip()
                table[last] = v.strip()
            elif x.strip():
                table[last] = f'{table[last]}; {x.strip()}'
        return table
    return text

def _ct_record(text: Any) -> dict | None:
    if not isinstance(text, str) or ' | ' not in text:
        return None
    marks = list(_CT_FIELD.finditer(text))
    if len(marks) < 3 or marks[0].start() != 0:
        return None
    out = {}
    for n, m in enumerate(marks):
        end = marks[n + 1].start() if n + 1 < len(marks) else len(text)
        out[m.group(1)] = _ct_scalar(text[m.end():end].strip())
    return out

def _ct_struct(obs: Any) -> Any:
    if isinstance(obs, list):
        out = [_ct_record(v) or _ct_struct(v) for v in obs]
        return out if any((a is not b for a, b in zip(out, obs))) else obs
    if isinstance(obs, dict):
        out = {k: _ct_record(v) or _ct_struct(v) for k, v in obs.items()}
        if not any((out[k] is not v for k, v in obs.items())):
            return obs
        if len(obs) == 1 and isinstance(next(iter(obs.values())), str):
            return next(iter(out.values()))
        return out
    return obs

def unwrap(obs: Any) -> tuple[Any, list[str]]:
    if STRUCT_TEXT:
        obs = _ct_struct(obs)
    if not isinstance(obs, dict):
        return (obs, [])
    lists = [k for k, v in obs.items() if _ct_strings(v)]
    bodies = [k for k, v in obs.items() if isinstance(v, (dict, list)) and k not in lists]
    beside = len(obs) == 2 and len(lists) == 1 and (len(bodies) == 1) and any((len(v.split()) >= 2 for v in obs[lists[0]]))
    notices: list[str] = []
    rest: dict = {}
    refused = False
    for key, value in obs.items():
        if key == 'refusal':
            refused = True
            notices.extend(_ct_spoken(value))
        elif key in lists and (beside or sum((len(v.split()) for v in value)) / len(value) >= 4):
            notices.extend(value)
        else:
            rest[key] = value
    if not notices and (not refused):
        return (obs, [])
    if len(rest) == 1:
        only = next(iter(rest.values()))
        if isinstance(only, (dict, list)):
            return (only, notices)
    return (rest, notices)

def _ct_reply(result: Any) -> tuple[Any, int | None, str]:
    data = getattr(result, 'data', result)
    error = getattr(result, 'error', None) or {}
    status = error.get('status') if isinstance(error, dict) else None
    body = error.get('body') if isinstance(error, dict) else ''
    if not isinstance(body, str):
        try:
            body = json.dumps(body)
        except (TypeError, ValueError):
            body = str(body)
    return (data if isinstance(data, dict) else None, status, body or '')

def _ct_transient(status: Any) -> bool:
    if status is None:
        return True
    if not isinstance(status, int) or isinstance(status, bool):
        return False
    return status == 408 or status == 425 or status == 429 or (500 <= status <= 599)

def _ct_post(proxy: Any, path: str, payload: dict) -> tuple[Any, int | None, str]:
    try:
        if hasattr(proxy, 'post_verbose'):
            return _ct_reply(proxy.post_verbose(path, payload))
        return _ct_reply(proxy.post(path, payload))
    except Exception as exc:
        return (None, None, type(exc).__name__)

class Env:

    def __init__(self, binding: dict, max_steps: int, proxy=None, pause=None):
        if proxy is None:
            from src.agent.proxy_client import ProxyClient
            proxy = ProxyClient(timeout=95, max_retries=1)
        self.proxy = proxy
        self.binding = dict(binding or {})
        self.max_steps = max_steps if isinstance(max_steps, int) and max_steps > 0 else 30
        self.step = 0
        self.finished = False
        self.out_of_key = False
        self.history: list[dict] = []
        self.faults: list[dict] = []
        self.pause = pause or time.sleep

    def steps_left(self) -> int:
        return max(0, self.max_steps - self.step)

    def call(self, name: str, args: dict) -> Any:
        if self.finished or self.out_of_key or self.step >= self.max_steps:
            return None
        action = {'name': name, 'args': args if isinstance(args, dict) else {}}
        number = self.step + 1
        resends = resyncs = 0
        while True:
            payload = dict(self.binding)
            payload['step'] = number
            payload['action'] = action
            data, status, body = _ct_post(self.proxy, ENV_PATH, payload)
            if data is not None:
                got = data.get('step')
                self.step = got if isinstance(got, int) and (not isinstance(got, bool)) else number
                if data.get('finished'):
                    self.finished = True
                obs = data.get('observation')
                obs = {} if obs is None else obs
                self.history.append({'step': self.step, 'name': name, 'args': action['args'], 'obs': obs})
                return obs
            self.faults.append({'step': number, 'name': name, 'status': status, 'body': body[:300]})
            if status == 402:
                self.out_of_key = True
                return None
            if status == 409:
                low = body.lower()
                if 'finished' in low or 'finalized' in low or 'quarantined' in low:
                    self.finished = True
                    return None
                expected = re.search('expected\\s+(\\d+)', body)
                if expected and resyncs < 1:
                    resyncs += 1
                    number = int(expected.group(1))
                    self.step = number - 1
                    if self.step >= self.max_steps:
                        return None
                    continue
                if 'different action' in low and resyncs < 1:
                    resyncs += 1
                    number += 1
                    self.step = number - 1
                    continue
                return None
            if not _ct_transient(status) or resends >= ENV_RESENDS:
                return None
            resends += 1
            if status is not None:
                self.pause(resends * (0.8 + 0.4 * random.random()))

class Llm:

    def __init__(self, proxy=None, soft_cap=SOFT_CAP, hard_cap=HARD_CAP, max_calls=60, pause=None):
        if proxy is None:
            from src.agent.proxy_client import ProxyClient
            proxy = ProxyClient(timeout=50, max_retries=1)
        self.proxy = proxy
        self.soft_cap = soft_cap
        self.hard_cap = hard_cap
        self.max_calls = max_calls
        self.spent = 0.0
        self.calls = 0
        self.paused = 0.0
        self.broke = False
        self.faults: list[dict] = []
        self.pause = pause or time.sleep
        self.last_cut = False
        self.cuts = 0
        self.unread = 0

    def over_soft(self) -> bool:
        return self.spent >= self.soft_cap

    def over_hard(self) -> bool:
        return self.spent >= self.hard_cap

    def closed(self) -> bool:
        return self.broke or self.spent >= self.hard_cap or self.calls >= self.max_calls

    def heavy_room(self) -> bool:
        return self.spent + HEAVY_ROOM <= self.hard_cap

    def _cut(self, data: dict) -> bool:
        choices = data.get('choices')
        first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        self.last_cut = str(first.get('finish_reason') or '').lower() in ('length', 'max_tokens')
        self.cuts += self.last_cut
        return self.last_cut

    def chat(self, messages: list, tools: list, *, heavy=False, max_tokens=1500) -> dict | None:
        if self.closed():
            return None
        model = HEAVY if heavy else LIGHT
        sent = [dict(m) for m in messages]
        if model.startswith('anthropic/'):
            for i in {0, len(sent) - 1}:
                if isinstance(sent[i].get('content'), str) and sent[i]['content']:
                    sent[i]['content'] = [{'type': 'text', 'text': sent[i]['content'], 'cache_control': {'type': 'ephemeral'}}]
        body = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body['tools'], body['tool_choice'] = (tools, 'auto')
        self.calls += 1
        tries = 0
        while True:
            data, status, detail = _ct_post(self.proxy, LLM_PATH, body)
            if data is not None and isinstance(data.get('error'), dict):
                code = data['error'].get('code')
                status = code if isinstance(code, int) and (not isinstance(code, bool)) else 500
                data = None
            if data is not None:
                usage = data.get('usage') or {}
                if isinstance(usage, dict) and isinstance(usage.get('cost'), (int, float)):
                    self.spent += float(usage['cost'])
                choices = data.get('choices') or [{}]
                msg = dict((choices[0] or {}).get('message') or {})
                if self._cut(data) and msg.get('tool_calls'):
                    whole = [c for c in msg['tool_calls'] if _ct_args_whole(c)]
                    if len(whole) < len(msg['tool_calls']):
                        self.faults.append({'model': model, 'status': 'cut', 'body': 'tool call cut at max_tokens'})
                    msg['tool_calls'] = whole
                return msg if msg.get('content') or msg.get('tool_calls') else None
            self.faults.append({'model': model, 'status': status, 'body': str(detail)[:200]})
            if status == 402:
                self.broke = True
                return None
            if not _ct_transient(status) or tries >= len(LLM_PAUSES) - 1 or self.paused + LLM_PAUSES[tries] > LLM_PAUSE_BUDGET:
                return None
            self.paused += LLM_PAUSES[tries]
            self.pause(LLM_PAUSES[tries] * (0.8 + 0.4 * random.random()))
            tries += 1

    def ask(self, system: str, user: str, *, heavy=False, max_tokens=1800, want_json=True) -> Any:
        first = HEAVY if heavy and (not HEAVY_FIRST_ROOM or self.heavy_room()) else LIGHT
        max_tokens = max(max_tokens, 1200)
        for model in (first, LIGHT if heavy else HEAVY):
            if self.closed() or (model != first and model == HEAVY and (not self.heavy_room())):
                return None
            text, status = self._send(model, system, user, max_tokens)
            if text is not None:
                got = first_json(text) if want_json else text
                if got is not None or not want_json:
                    return got
                self.unread += 1
                continue
            if status == 200:
                self.unread += 1
                continue
            if status not in (403, 404):
                return None
        return None

    def _send(self, model: str, system: str, user: str, max_tokens: int) -> tuple[str | None, Any]:
        self.calls += 1
        head: Any = system
        if model.startswith('anthropic/') and system:
            head = [{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}]
        messages = ([{'role': 'system', 'content': head}] if system else []) + [{'role': 'user', 'content': user}]
        body = {'model': model, 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0}
        tries = 0
        while True:
            data, status, detail = _ct_post(self.proxy, LLM_PATH, body)
            if data is not None and isinstance(data.get('error'), dict):
                code = data['error'].get('code')
                status = code if isinstance(code, int) and (not isinstance(code, bool)) else 500
                detail = str(data['error'].get('message') or '')
                data = None
            if data is not None:
                usage = data.get('usage') or {}
                cost = usage.get('cost') if isinstance(usage, dict) else None
                if isinstance(cost, (int, float)) and (not isinstance(cost, bool)):
                    self.spent += float(cost)
                if self._cut(data):
                    self.faults.append({'model': model, 'status': 'cut', 'body': f'stopped at max_tokens {max_tokens}'})
                return (self._text(data), 200)
            self.faults.append({'model': model, 'status': status, 'body': detail[:200]})
            if status == 402:
                self.broke = True
                return (None, status)
            limit = 1 if status in (None, 408) else len(LLM_PAUSES)
            if not _ct_transient(status) or tries >= limit or self.paused + LLM_PAUSES[tries] > LLM_PAUSE_BUDGET:
                return (None, status)
            self.paused += LLM_PAUSES[tries]
            self.pause(LLM_PAUSES[tries] * (0.8 + 0.4 * random.random()))
            tries += 1

    @staticmethod
    def _text(data: dict) -> str | None:
        choices = data.get('choices')
        if not isinstance(choices, list) or not choices or (not isinstance(choices[0], dict)):
            return None
        message = choices[0].get('message') or {}
        content = message.get('content')
        if isinstance(content, list):
            content = ''.join((p.get('text', '') for p in content if isinstance(p, dict)))
        if isinstance(content, str) and content.strip():
            return content
        for call in message.get('tool_calls') or []:
            fn = (call or {}).get('function') or {}
            if fn.get('name'):
                args = fn.get('arguments')
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except ValueError:
                    args = {}
                return json.dumps({'tool': fn['name'], 'args': args if isinstance(args, dict) else {}})
        return None
ROLES = ('search', 'detail', 'compare', 'cart_set', 'cart_view', 'message', 'finish_order', 'finish_decline', 'other')
SEARCH_SLOTS = ('query', 'page', 'category', 'max_price')
TM_RECORDS = 400
JSON_TYPES = ('null', 'boolean', 'integer', 'number', 'string', 'array', 'object')
TM_ASK = 'You map the tools of an online store to the jobs a shopping program needs done. Tool names can be invented words: read each name, its description and its parameters together.\nJobs:\nsearch - looks listings up by words or by filters such as a shelf, a seller or a price cap\ndetail - opens the page of a single listing (properties, offer, notes)\ncompare - returns a chosen set of listings together\ncart_set - puts a listing in the cart, changes its quantity or takes it out\ncart_view - lists what the cart holds now\nmessage - writes to the shopper\nfinish_order - places the order for what the cart holds\nfinish_decline - closes the task with no purchase\nother - none of these\nReply with ONE JSON object and nothing else:\n{"roles": {"<tool>": "<job>"} for every tool marked OPEN,\n"order_args": {"<parameter>": <value>} constant values the finish_order tool needs so that it places the order, {} if it takes none,\n"decline_args": {"<parameter>": <value>} constant values a tool needs so that it closes the task with no purchase; when a single tool does both, the values that choose closing with no purchase; {} if none,\n"search": {"<search tool>": {"query": the parameter taking free search words or null, "page": the parameter taking the value that asks for the next page or null, "category": the parameter taking a shelf or category or null, "max_price": the parameter taking a price cap or null, "fixed": {"<parameter>": <value>} any other argument a plain word search needs}}}\nA tool marked OPEN takes one of the jobs listed after it.'

def _tm_types(spec: Any) -> list[str]:
    kind = spec.get('type') if isinstance(spec, dict) else None
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]

def _tm_kind(spec: Any) -> str:
    if not isinstance(spec, dict):
        return 'any'
    if isinstance(spec.get('enum'), list) and spec['enum']:
        return 'choice'
    types = _tm_types(spec)
    if 'array' in types:
        inner = _tm_types(spec.get('items') or {})
        return 'objs' if 'object' in inner else 'ids'
    if 'integer' in types or 'number' in types:
        return 'num'
    if 'boolean' in types:
        return 'flag'
    if 'object' in types:
        return 'obj'
    return 'text' if 'string' in types or not types else 'any'

def _tm_json_type(value: Any) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, int):
        return 'integer'
    if isinstance(value, float):
        return 'number'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, list):
        return 'array'
    return 'object' if isinstance(value, dict) else 'unknown'

def _tm_fits(value: Any, kind: str) -> bool:
    have = _tm_json_type(value)
    return have == kind or (kind == 'number' and have == 'integer') or kind not in JSON_TYPES

def _tm_convert(value: Any, kind: str) -> Any:
    have = _tm_json_type(value)
    if kind in ('integer', 'number') and have in ('integer', 'number', 'string'):
        number = _tm_number(value)
        if number is None or (kind == 'integer' and (not number.is_integer())):
            return None
        return int(number) if number.is_integer() else number
    if kind == 'string' and have in ('integer', 'number'):
        return str(value)
    if kind == 'array' and have not in ('array', 'null'):
        return [value]
    return None

def _tm_bend(value: Any, spec: Any) -> Any:
    if not isinstance(spec, dict):
        return value
    types = _tm_types(spec)
    if types and (not any((_tm_fits(value, t) for t in types))):
        value = next((c for c in (_tm_convert(value, t) for t in types) if c is not None), value)
    options = spec.get('enum')
    if isinstance(options, list) and isinstance(value, str) and (value not in options):
        value = next((o for o in options if isinstance(o, str) and o.lower() == value.strip().lower()), value)
    cap = spec.get('maxLength')
    if isinstance(value, str) and isinstance(cap, int):
        value = value[:cap]
    if isinstance(value, list):
        cap = spec.get('maxItems')
        value = [_tm_bend(v, spec.get('items')) for v in (value[:cap] if isinstance(cap, int) else value)]
    return value

def _tm_fault(value: Any, spec: Any, where: str) -> str | None:
    if not isinstance(spec, dict):
        return None
    types = _tm_types(spec)
    if types and (not any((_tm_fits(value, t) for t in types))):
        return f"{where} must be {'/'.join(types)}"
    options = spec.get('enum')
    if isinstance(options, list) and value not in options:
        return f'{where} must be one of {options}'
    if isinstance(value, str):
        if isinstance(spec.get('maxLength'), int) and len(value) > spec['maxLength']:
            return f"{where} is longer than {spec['maxLength']}"
        if isinstance(spec.get('minLength'), int) and len(value) < spec['minLength']:
            return f"{where} is shorter than {spec['minLength']}"
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        if isinstance(spec.get('minimum'), (int, float)) and value < spec['minimum']:
            return f"{where} is below {spec['minimum']}"
        if isinstance(spec.get('maximum'), (int, float)) and value > spec['maximum']:
            return f"{where} is above {spec['maximum']}"
    if isinstance(value, list):
        if isinstance(spec.get('maxItems'), int) and len(value) > spec['maxItems']:
            return f"{where} takes at most {spec['maxItems']} entries"
        if isinstance(spec.get('minItems'), int) and len(value) < spec['minItems']:
            return f"{where} needs at least {spec['minItems']} entries"
        for i, v in enumerate(value):
            fault = _tm_fault(v, spec.get('items'), f'{where}[{i}]')
            if fault:
                return fault
    if isinstance(value, dict):
        props = spec.get('properties') if isinstance(spec.get('properties'), dict) else {}
        for name in spec.get('required') or []:
            if name not in value:
                return f'{where}{name} is required'
        for name, v in value.items():
            if name not in props:
                if spec.get('additionalProperties') is False:
                    return f'{where}{name} is not an argument'
                continue
            fault = _tm_fault(v, props[name], f'{where}{name}')
            if fault:
                return fault
    return None

def _tm_words(text: Any) -> set[str]:
    spaced = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(text)).lower()
    return {re.sub('(?<=...)s$', '', w) for w in re.findall('[a-z0-9]{3,}', spaced)}

def _tm_number(value: Any) -> float | None:
    if isinstance(value, dict):
        return next((n for n in map(_tm_number, value.values()) if n is not None), None)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None

def _tm_token(value: Any) -> bool:
    return isinstance(value, str) and 3 <= len(value.strip()) <= 64 and (not re.search('\\s', value.strip()))

def _tm_row_lists(body: Any) -> list[tuple[str | None, list]]:

    def rowish(v: Any) -> bool:
        dicts = [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []
        return bool(dicts) and len(dicts) * 2 >= len(v) and (sum((len(d) for d in dicts)) >= 2 * len(dicts))
    if isinstance(body, list):
        return [(None, body)] if rowish(body) else []
    found = []
    if isinstance(body, dict):
        for key, value in body.items():
            if rowish(value):
                found.append((key, value))
            elif isinstance(value, dict):
                found += [(f'{key}.{k}', v) for k, v in value.items() if rowish(v)]
    return found

class ToolMap:

    def __init__(self, tools: list[dict], llm=None):
        self.tools = [t for t in tools or [] if isinstance(t, dict) and isinstance((t.get('function') or t).get('name'), str)]
        self.schemas: dict[str, dict] = {}
        self.about: dict[str, str] = {}
        for t in self.tools:
            fn = t.get('function') or t
            params = fn.get('parameters')
            self.schemas[fn['name']] = params if isinstance(params, dict) else {}
            self.about[fn['name']] = str(fn.get('description') or '')
        self.names = list(self.schemas)
        self.role: dict[str, str] = {}
        self.options: dict[str, set[str]] = {n: self._options(n) for n in self.names}
        self.slots: dict[str, dict[str, str]] = {}
        self.fixed: dict[str, dict] = {}
        self.order_args: dict = {}
        self.decline_args: dict | None = None
        self.keys: dict[str, str | None] = dict.fromkeys(('id', 'title', 'price', 'stock', 'category', 'props', 'notes', 'qty'))
        self.rows_key: dict[str, str | None] = {}
        self.next_key: dict[str, str] = {}
        self.count_key: dict[str, str] = {}
        self.page_seen: dict[str, dict] = {}
        self.shows_props: dict[str, bool] = {}
        self.asked = 0
        self.faults: list[str] = []
        self._records: list[dict] = []
        self._arg_values: set[str] = set()
        self._cart: dict[str, int] = {}
        self._cart_probe: tuple | None = None
        self._infer(llm)
        self.cart_semantics = self._cart_default()

    def _args(self, name: str) -> dict[str, Any]:
        props = self.schemas.get(name, {}).get('properties')
        return props if isinstance(props, dict) else {}

    def _required(self, name: str) -> list[str]:
        return [r for r in self.schemas.get(name, {}).get('required') or [] if r in self._args(name)]

    def _of(self, name: str, *kinds: str) -> list[str]:
        return [a for a, spec in self._args(name).items() if _tm_kind(spec) in kinds]

    def _options(self, name: str) -> set[str]:
        kinds = [_tm_kind(s) for s in self._args(name).values()]
        need = {_tm_kind(self._args(name)[r]) for r in self._required(name)}
        found = {'other'}
        if not kinds:
            found |= {'cart_view', 'finish_order', 'finish_decline'}
        if 'ids' in kinds:
            found.add('compare')
        if 'objs' in kinds:
            found.add('cart_set')
        text = 'text' in kinds and 'ids' not in kinds and ('objs' not in kinds)
        if text or (('choice' in kinds or 'num' in kinds) and 'ids' not in kinds):
            found.add('search')
        if text:
            found |= {'detail', 'message'}
        if text and ('num' in kinds or 'choice' in kinds):
            found.add('cart_set')
        if kinds and need <= {'text', 'choice'} and ('ids' not in kinds):
            found |= {'finish_order', 'finish_decline'}
        return found

    def _default(self, name: str) -> str:
        opts = self.options[name] - {'other'}
        if len(opts) == 1:
            return next(iter(opts))
        if not self._args(name):
            return 'other'
        texts = self._of(name, 'text')
        need = {_tm_kind(self._args(name)[r]) for r in self._required(name)}
        if 'cart_set' in opts and len(texts) == 1 and ('num' in need or 'choice' in need):
            return 'cart_set'
        if len(self._args(name)) == 1 and texts:
            return 'detail' if self._required(name) else 'finish_decline'
        return 'search' if 'search' in opts else 'other'

    def _line(self, name: str, settled: str | None) -> str:
        parts = []
        for arg, spec in self._args(name).items():
            kind = _tm_kind(spec)
            bits = [', '.join(_tm_types(spec)) or kind]
            if kind in ('ids', 'objs'):
                bits = [f"list of {', '.join(_tm_types((spec or {}).get('items') or {})) or 'values'}"]
            if arg in self._required(name):
                bits.append('required')
            if kind == 'choice':
                bits.append('one of ' + json.dumps(spec['enum']))
            parts.append(f"{arg} ({'; '.join(bits)})")
        about = self.about.get(name, '')
        same = _tm_words(about) <= _tm_words(name.replace('_', ' '))
        mark = f'settled: {settled}' if settled else 'OPEN: ' + ', '.join((r for r in ROLES if r in self.options[name]))
        text = f'- {name} [{mark}]' + ('' if same or not about else f' - {about[:300]}')
        return text + ': ' + ('; '.join(parts) if parts else 'no parameters')

    def _infer(self, llm) -> None:
        settled = {n: next(iter(self.options[n] - {'other'})) for n in self.names if len(self.options[n] - {'other'}) == 1}
        for name in self.names:
            self.role[name] = settled.get(name) or self._default(name)
        open_tools = [n for n in self.names if n not in settled]
        if llm is None or not open_tools:
            self._structural_slots()
            return
        user = 'TOOLS:\n' + '\n'.join((self._line(n, settled.get(n)) for n in self.names))
        got = None
        for heavy in (False, True):
            self.asked += 1
            got = llm.ask(TM_ASK, user, heavy=heavy, max_tokens=2000, want_json=True)
            if isinstance(got, dict) and isinstance(got.get('roles'), dict):
                break
            got = None
        if got:
            self._take(got, open_tools)
        self._structural_slots()

    def _take(self, got: dict, open_tools: list[str]) -> None:
        for name, job in got['roles'].items():
            if name in open_tools and isinstance(job, str) and (job.strip().lower() in self.options[name]):
                self.role[name] = job.strip().lower()
        finish = self.tool('finish_order')
        decline = next((n for n in self.names if self.role[n] == 'finish_decline'), None)
        if finish and self._fixed_ok(finish, got.get('order_args')):
            self.order_args = dict(got['order_args'])
        target = decline or finish
        wanted = got.get('decline_args')
        if target and self._fixed_ok(target, wanted) and (decline or (wanted and wanted != self.order_args)):
            self.decline_args = dict(wanted)
        for name, spec in (got.get('search') or {}).items():
            if name not in self.schemas or self.role.get(name) != 'search' or (not isinstance(spec, dict)):
                continue
            slots = {s: spec[s] for s in SEARCH_SLOTS if isinstance(spec.get(s), str) and spec[s] in self._args(name)}
            self.slots[name] = slots
            fixed = spec.get('fixed')
            if self._fixed_ok(name, fixed, partial=True):
                self.fixed[name] = {k: v for k, v in fixed.items() if k not in slots.values()}

    def _fixed_ok(self, name: str, args: Any, partial: bool=False) -> bool:
        if not isinstance(args, dict):
            return False
        props = self._args(name)
        for key, value in args.items():
            if key not in props or _tm_fault(_tm_bend(value, props[key]), props[key], key):
                return False
        return partial or not [r for r in self._required(name) if r not in args and _tm_kind(props[r]) != 'text']

    def _structural_slots(self) -> None:
        for name in self.names:
            if self.role.get(name) != 'search':
                continue
            slots = self.slots.setdefault(name, {})
            if 'query' not in slots:
                texts = self._of(name, 'text')
                need = [a for a in texts if a in self._required(name)]
                pick = need[0] if len(need) == 1 else texts[0] if len(texts) == 1 else None
                if pick and pick not in slots.values():
                    slots['query'] = pick

    def tool(self, role: str) -> str | None:
        found = [n for n in self.names if self.role.get(n) == role]
        if role == 'finish_decline' and (not found) and self.decline_args:
            return self.tool('finish_order')
        if role == 'detail' and found:
            found.sort(key=lambda n: {True: 0, None: 1, False: 2}[self.shows_props.get(n)])
        if role == 'search' and found:
            found.sort(key=lambda n: 'query' not in self.slots.get(n, {}))
        return found[0] if found else None

    def _page_arg(self, name: str) -> str | None:
        slots = self.slots.get(name, {})
        if slots.get('page'):
            return slots['page']
        taken = set(slots.values()) | set(self.fixed.get(name, {}))
        free = [a for a in self._args(name) if a not in taken and a not in self._required(name) and (_tm_kind(self._args(name)[a]) in ('text', 'num'))]
        seen = self.page_seen.get(name, {}).get('value')
        if seen is not None:
            want = 'num' if isinstance(seen, int) else 'text'
            free = [a for a in free if _tm_kind(self._args(name)[a]) == want]
        return free[0] if len(free) == 1 else None

    def args_for(self, role: str, **semantic) -> dict | None:
        decline = bool(semantic.get('decline')) or role == 'finish_decline'
        name = semantic.get('tool') or self.tool(role)
        if not name or name not in self.schemas:
            return None
        role = self.role.get(name, role)
        props = self._args(name)
        out: dict = {}
        texts = self._of(name, 'text')
        lead = next((a for a in texts if a in self._required(name)), texts[0] if texts else None)
        if role == 'search':
            slots = self.slots.get(name, {})
            page = semantic.get('page', semantic.get('token'))
            for slot, value in (('query', semantic.get('query')), ('category', semantic.get('category')), ('max_price', semantic.get('max_price')), ('page', page)):
                arg = self._page_arg(name) if slot == 'page' else slots.get(slot)
                if value is not None and (not arg):
                    return None
                if value is not None:
                    out[arg] = value
            for key, value in self.fixed.get(name, {}).items():
                out.setdefault(key, value)
        elif role == 'detail':
            if lead and semantic.get('product_id'):
                out[lead] = semantic['product_id']
        elif role == 'compare':
            arg = next(iter(self._of(name, 'ids')), None)
            if arg and semantic.get('product_ids'):
                out[arg] = list(semantic['product_ids'])
        elif role == 'cart_set':
            if not self._cart_args(name, out, semantic):
                return None
        elif role == 'message':
            if lead and semantic.get('text'):
                out[lead] = semantic['text']
        elif role in ('finish_order', 'finish_decline'):
            if self.role.get(name) == 'finish_order' and decline:
                if self.decline_args is None:
                    return None
                out.update(self.decline_args)
            elif self.role.get(name) == 'finish_order':
                out.update(self.order_args)
            elif self.decline_args:
                out.update(self.decline_args)
            note = next((a for a in texts if a not in out), None)
            if note and semantic.get('text'):
                out[note] = semantic['text']
        for arg in self._required(name):
            if arg not in out and _tm_kind(props[arg]) == 'choice':
                out[arg] = props[arg]['enum'][0]
        out = {k: _tm_bend(v, props.get(k)) for k, v in out.items()}
        return None if self.validate(name, out) else out

    def _cart_args(self, name: str, out: dict, semantic: dict) -> bool:
        props = self._args(name)
        lines = self._of(name, 'objs')
        ident, qty = (semantic.get('product_id'), semantic.get('qty'))
        remove = semantic.get('mode') == 'remove'
        if lines:
            inner = (props[lines[0]].get('items') or {}).get('properties') or {}
            id_arg = next((a for a, s in inner.items() if _tm_kind(s) == 'text'), None)
            qty_arg = next((a for a, s in inner.items() if _tm_kind(s) == 'num'), None)
            if not id_arg or not ident:
                return False
            line = {id_arg: ident}
            if qty_arg:
                line[qty_arg] = 0 if remove else qty if qty is not None else 1
            out[lines[0]] = [line]
            return True
        texts = self._of(name, 'text')
        id_arg = next((a for a in texts if a in self._required(name)), texts[0] if texts else None)
        if not id_arg or not ident:
            return False
        out[id_arg] = ident
        modes = self._of(name, 'choice')
        nums = self._of(name, 'num')
        if modes:
            values = props[modes[0]]['enum']
            if remove and len(values) < 2:
                return False
            out[modes[0]] = values[1] if remove else values[0]
        if nums:
            if remove and (not modes):
                out[nums[0]] = 0
            elif qty is not None:
                out[nums[0]] = qty
            elif nums[0] in self._required(name):
                out[nums[0]] = 1
        elif remove and (not modes):
            return False
        return True

    def validate(self, name: str, args: dict) -> str | None:
        if name not in self.schemas:
            return f'unknown tool {name!r}'
        if not isinstance(args, dict):
            return 'arguments must be an object'
        spec = dict(self.schemas[name])
        spec.setdefault('type', 'object')
        return _tm_fault(args, spec, '')

    def learn(self, name: str, args: dict, obs: Any) -> None:
        try:
            self._learn(name, args if isinstance(args, dict) else {}, obs)
        except Exception as exc:
            self.faults.append(f'learn {name}: {type(exc).__name__}')

    def _learn(self, name: str, args: dict, obs: Any) -> None:
        body, _ = unwrap(obs)
        values = [v for v in (args or {}).values()]
        flat = [x for v in values for x in (v if isinstance(v, list) else [v])]
        echoes = {x.strip() for x in flat if _tm_token(x)}
        words = set().union(*([_tm_words(x) for x in flat if isinstance(x, str) and (not _tm_token(x))] or [set()]))
        self._page_echo(name, args or {})
        kind = 'cart' if self.role.get(name) in ('cart_set', 'cart_view') else 'list'
        lists = _tm_row_lists(body)
        if lists:
            known = self.rows_key.get(name)
            lists.sort(key=lambda kv: (kv[0] != known, -len(kv[1])))
            self.rows_key[name] = lists[0][0]
            for _, rows in lists:
                for row in rows:
                    if isinstance(row, dict):
                        self._keep(row, kind, words, name)
            if kind == 'list' and isinstance(body, dict):
                self._paging(name, body, len(lists[0][1]))
        elif isinstance(body, dict) and len(body) >= 2 and (not ('error' in body and len(body) <= 2)):
            if echoes & {str(v).strip() for v in body.values() if isinstance(v, str)} or self._ident(body, strict=True):
                self._keep(body, 'cart' if kind == 'cart' else 'single', words, name, echoes)
        self._arg_values |= echoes
        self._settle()
        if self.role.get(name) == 'detail' and isinstance(body, dict) and self._ident(body):
            self.shows_props[name] = isinstance(body.get(self.keys['props'] or ''), dict)
        if kind == 'cart':
            self._cart_learn(name, args or {}, body)

    def _keep(self, row: dict, kind: str, words: set[str], tool: str, echoes: set[str]=frozenset()) -> None:
        hits = {k for k, v in row.items() if words and isinstance(v, str) and (len(v) <= 80) and words & _tm_words(v)}
        self._records.append({'row': row, 'kind': kind, 'hits': hits, 'tool': tool, 'echo': echoes})
        if len(self._records) > TM_RECORDS:
            del self._records[0]

    def _paging(self, name: str, body: dict, shown: int) -> None:
        scalars = {k: v for k, v in body.items() if v is None or (isinstance(v, (str, int)) and (not isinstance(v, bool)))}
        counts = [k for k, v in scalars.items() if isinstance(v, int) and v >= shown]
        if counts:
            self.count_key.setdefault(name, max(counts, key=lambda k: scalars[k]))
        rest = [k for k, v in scalars.items() if k != self.count_key.get(name) and (v is None or isinstance(v, int) or _tm_token(v))]
        known = self.next_key.get(name)
        pick = known if known in scalars else rest[0] if len(rest) == 1 else None
        if pick:
            self.next_key[name] = pick
            self.page_seen[name] = {'value': scalars[pick]}

    def _page_echo(self, name: str, args: dict) -> None:
        seen = self.page_seen.get(name, {}).get('value')
        if seen is None:
            return
        for arg, value in args.items():
            if value == seen and arg in self._args(name):
                self.slots.setdefault(name, {})['page'] = arg

    def next_page(self, name: str, obs: Any) -> Any:
        body, _ = unwrap(obs)
        key = self.next_key.get(name)
        return body.get(key) if isinstance(body, dict) and key else None

    def total(self, name: str, obs: Any) -> int | None:
        body, _ = unwrap(obs)
        key = self.count_key.get(name)
        value = body.get(key) if isinstance(body, dict) and key else None
        return value if isinstance(value, int) and (not isinstance(value, bool)) else None

    def _ident(self, row: Any, strict: bool=False) -> str | None:
        if not isinstance(row, dict):
            return None
        key = self.keys.get('id')
        if key and _tm_token(row.get(key)):
            return row[key].strip()
        if strict:
            return None
        tokens = [v.strip() for v in row.values() if _tm_token(v) and re.search('\\d', v)]
        return tokens[0] if len(tokens) == 1 else None

    def _settle(self) -> None:
        recs = self._records
        if not recs:
            return
        self.keys['id'] = self._learn_id(recs) or self.keys['id']
        ident = self.keys['id']
        unique: dict[str, dict] = {}
        for rec in recs:
            if rec['kind'] != 'cart':
                rid = rec['row'].get(ident) if ident else None
                key = rid if isinstance(rid, str) else id(rec['row'])
                merged = unique.setdefault(key, {'row': {}, 'hits': set(), 'kind': rec['kind']})
                merged['row'].update(rec['row'])
                merged['hits'] |= rec['hits']
        goods = list(unique.values())
        carts = [r['row'] for r in recs if r['kind'] == 'cart']
        self.keys['title'] = self._learn_title(goods, carts) or self.keys['title']
        self.keys['price'] = self._learn_price(goods, carts) or self.keys['price']
        self.keys['stock'] = self._learn_stock(goods) or self.keys['stock']
        self.keys['category'] = self._learn_category(goods) or self.keys['category']
        self.keys['props'] = self._learn_props(recs) or self.keys['props']
        self.keys['notes'] = self._learn_notes(recs) or self.keys['notes']
        self.keys['qty'] = self._learn_qty(carts) or self.keys['qty']

    def _learn_id(self, recs: list[dict]) -> str | None:
        best = None
        scores: dict[str, list] = {}
        for rec in recs:
            for key, value in rec['row'].items():
                slot = scores.setdefault(key, [0, 0, 0, 0, set()])
                slot[0] += 1
                if _tm_token(value):
                    slot[1] += 1
                    slot[2] += bool(re.search('\\d', value))
                    slot[3] += value.strip() in rec['echo'] or value.strip() in self._arg_values
                    slot[4].add(value.strip())
        for key, (present, tokens, digits, echoed, distinct) in scores.items():
            if tokens < 0.8 * len(recs) or tokens < present:
                continue
            score = (echoed, tokens, len(distinct) / tokens, digits / tokens)
            if best is None or score > best[0]:
                best = (score, key)
        return best[1] if best else None

    @staticmethod
    def _text_stats(goods: list[dict], skip: set) -> dict[str, dict]:
        stats: dict[str, dict] = {}
        for rec in goods:
            for key, value in rec['row'].items():
                if key in skip or not (isinstance(value, str) or value is None):
                    continue
                s = stats.setdefault(key, {'n': 0, 'full': 0, 'spaced': 0, 'chars': 0, 'distinct': set(), 'hit': 0})
                s['n'] += 1
                text = (value or '').strip()
                if text:
                    s['full'] += 1
                    s['spaced'] += ' ' in text
                    s['chars'] += len(text)
                    s['distinct'].add(text.lower())
                    s['hit'] += key in rec['hits']
        return stats

    def _learn_title(self, goods: list[dict], carts: list[dict]) -> str | None:
        if not goods:
            return None
        stats = self._text_stats(goods, {self.keys['id']})
        in_cart = {v.strip().lower() for row in carts for v in row.values() if isinstance(v, str)}
        best = None
        for key, s in stats.items():
            if s['full'] < 0.9 * len(goods) or s['full'] == 0 or s['spaced'] < 0.6 * s['full']:
                continue
            if len(goods) >= 3 and len(s['distinct']) < 0.6 * s['full']:
                continue
            echoed = sum((1 for v in s['distinct'] if v in in_cart))
            score = (echoed, -s['chars'] / s['full'])
            if best is None or score > best[0]:
                best = (score, key)
        return best[1] if best else None

    def _learn_price(self, goods: list[dict], carts: list[dict]) -> str | None:
        ident = self.keys['id']
        lines = {row.get(ident): row for row in carts if ident and isinstance(row.get(ident), str)}
        best = None
        keys = {k for rec in goods for k, v in rec['row'].items() if isinstance(v, (int, float)) and (not isinstance(v, bool))}
        for key in keys:
            vals = [rec['row'].get(key) for rec in goods]
            nums = [v for v in vals if isinstance(v, (int, float)) and (not isinstance(v, bool))]
            if len(nums) < 0.8 * len(goods) or min(nums) < 0:
                continue
            echoed = sum((1 for rec in goods if rec['row'].get(ident) in lines and rec['row'].get(key) in lines[rec['row'].get(ident)].values()))
            score = (echoed, len(set(nums)), sum(nums) / len(nums))
            if best is None or score > best[0]:
                best = (score, key)
        return best[1] if best else None

    @staticmethod
    def _learn_stock(goods: list[dict]) -> str | None:
        best = None
        keys = {k for rec in goods for k, v in rec['row'].items() if isinstance(v, bool)}
        for key in keys:
            flags = [rec['row'].get(key) for rec in goods if isinstance(rec['row'].get(key), bool)]
            if len(flags) < 0.8 * len(goods):
                continue
            score = (len(flags), sum(flags))
            if best is None or score > best[0]:
                best = (score, key)
        return best[1] if best else None

    def _learn_category(self, goods: list[dict]) -> str | None:
        stats = self._text_stats(goods, {self.keys['id'], self.keys['title']})
        best = None
        for key, s in stats.items():
            if s['full'] < 0.8 * len(goods) or s['full'] == 0 or s['chars'] / s['full'] > 40:
                continue
            spread = len(s['distinct']) / s['full']
            score = (s['hit'] / s['full'], -spread, s['full'])
            if score[0] == 0 and (len(goods) < 3 or spread > 0.5):
                continue
            if best is None or score > best[0]:
                best = (score, key)
        return best[1] if best else None

    @staticmethod
    def _learn_props(recs: list[dict]) -> str | None:
        counts: dict[str, int] = {}
        for rec in recs:
            for key, value in rec['row'].items():
                if isinstance(value, dict):
                    scalars = [v for v in value.values() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))]
                    if len(scalars) >= 2:
                        counts[key] = counts.get(key, 0) + 1
        return max(counts, key=counts.get) if counts else None

    def _learn_notes(self, recs: list[dict]) -> str | None:
        listed = {k for r in recs if r['kind'] == 'list' and self.role.get(r['tool']) != 'compare' for k in r['row']}
        singles = [r['row'] for r in recs if r['kind'] == 'single']
        if not listed or not singles:
            return None
        skip = listed | {self.keys[k] for k in ('id', 'title', 'category', 'price', 'stock', 'props')}
        best = None
        keys = {k for row in singles for k, v in row.items() if k not in skip and (v is None or isinstance(v, str))}
        for key in keys:
            vals = [row.get(key) for row in singles if key in row]
            if not all((v is None or isinstance(v, str) for v in vals)):
                continue
            blank = sum((1 for v in vals if not (v or '').strip()))
            said = sum((1 for v in vals if isinstance(v, str) and len(v.split()) >= 4))
            score = ((blank + said) / len(vals), blank / len(vals))
            if score[0] <= 0.5:
                continue
            if best is None or score > best[0]:
                best = (score, key)
        return best[1] if best else None

    def _learn_qty(self, carts: list[dict]) -> str | None:
        keys = {k for row in carts for k, v in row.items() if isinstance(v, int) and (not isinstance(v, bool)) and (k != self.keys['price'])}
        if not keys:
            return None
        if len(keys) == 1:
            return next(iter(keys))
        return min(keys, key=lambda k: sum((row.get(k, 0) for row in carts if isinstance(row.get(k), int))))

    def _cart_default(self) -> str:
        cart = self.tool('cart_set')
        if not cart:
            return 'add'
        nums = [self._args(cart)[a] for a in self._of(cart, 'num')]
        return 'set' if not self._of(cart, 'choice') and any((s.get('minimum') == 0 for s in nums)) else 'add'

    def _cart_learn(self, name: str, args: dict, body: Any) -> None:
        lines = None
        found = _tm_row_lists(body)
        if found:
            lines = {self._ident(r): self.qty(r) for r in found[0][1] if isinstance(r, dict) and self._ident(r)}
        elif isinstance(body, dict) and self._ident(body) and (self.qty(body) is not None):
            lines = {self._ident(body): self.qty(body)}
        elif self.role.get(name) == 'cart_view' and isinstance(body, dict) and ([] in body.values()):
            lines, found = ({}, True)
        if self.role.get(name) == 'cart_set':
            texts = self._of(name, 'text')
            id_arg = next((a for a in texts if a in self._required(name)), texts[0] if texts else None)
            ident = args.get(id_arg) if id_arg else None
            qty = next((args[a] for a in self._of(name, 'num') if isinstance(args.get(a), int)), None)
            modes = self._of(name, 'choice')
            adding = not modes or args.get(modes[0]) == self._args(name)[modes[0]]['enum'][0]
            if ident and qty and adding and self._cart.get(ident):
                self._cart_probe = (ident, self._cart[ident], qty)
        if lines is None:
            return
        probe, self._cart_probe = (self._cart_probe, None)
        if probe and probe[0] in lines and (lines[probe[0]] is not None):
            _, before, sent = probe
            if lines[probe[0]] == sent and sent != before + sent:
                self.cart_semantics = 'set'
            elif lines[probe[0]] == before + sent and before + sent != sent:
                self.cart_semantics = 'add'
        if found or self.role.get(name) == 'cart_view':
            self._cart = {k: v for k, v in lines.items() if v}
        else:
            self._cart.update({k: v for k, v in lines.items() if v is not None})

    def rows(self, obs: Any) -> list[dict]:
        body, _ = unwrap(obs)
        lists = _tm_row_lists(body)
        if not lists:
            return []
        known = set(self.rows_key.values())
        lists.sort(key=lambda kv: (kv[0] not in known, -len(kv[1])))
        return [r for r in lists[0][1] if isinstance(r, dict)]

    def record(self, obs: Any) -> dict | None:
        body, _ = unwrap(obs)
        return body if isinstance(body, dict) and self._ident(body, strict=True) else None

    def row_id(self, row: Any) -> str | None:
        return self._ident(row)

    def price(self, row: Any) -> float | None:
        key = self.keys['price']
        return _tm_number(row.get(key)) if isinstance(row, dict) and key else None

    def in_stock(self, row: Any) -> bool | None:
        key = self.keys['stock']
        value = row.get(key) if isinstance(row, dict) and key else None
        if isinstance(value, bool):
            return value
        number = _tm_number(value)
        return None if number is None else number > 0

    def _field(self, row: Any, what: str) -> Any:
        key = self.keys[what]
        return row.get(key) if isinstance(row, dict) and key else None

    def title(self, row: Any) -> str | None:
        value = self._field(row, 'title')
        return value if isinstance(value, str) else None

    def category(self, row: Any) -> str | None:
        value = self._field(row, 'category')
        return value if isinstance(value, str) else None

    def props(self, row: Any) -> dict:
        value = self._field(row, 'props')
        return value if isinstance(value, dict) else {}

    def notes(self, row: Any) -> str | None:
        value = self._field(row, 'notes')
        return value if isinstance(value, str) and value.strip() else None

    def qty(self, row: Any) -> int | None:
        value = self._field(row, 'qty')
        return value if isinstance(value, int) and (not isinstance(value, bool)) else None
NEED_KINDS = ('stated', 'hidden', 'ref', 'excluded')
REF_RELATIONS = ('same', 'max', 'min', 'near', 'not', 'fits')

@dataclass
class Need:
    item: int
    text: str
    kind: str
    value: str | None = None
    prop: str = ''
    when: dict | None = None

    def __post_init__(self):
        if not self.prop:
            self.prop = self.text.split(':', 1)[0].strip()

@dataclass
class Item:
    name: str
    qty: int
    needs: list[Need]
    shelf: str = ''
    cap_cents: float | None = None

@dataclass
class Ref:
    item: int
    prop: str
    relation: str
    start: str
    chain: list[dict] = field(default_factory=list)
    value: str | None = None
    done: bool = False
    tol: str = ''
    owner: str = ''

def twin_refs(a, b) -> bool:
    return a.item == b.item and a.owner == b.owner and (' '.join(a.prop.lower().split()) == ' '.join(b.prop.lower().split()))

def fold_ref(refs, ref):
    kept = next((r for r in refs if twin_refs(r, ref)), None)
    if kept is None:
        refs.append(ref)
        return ref
    if kept is ref:
        return kept
    if ref.done and (not kept.done) or (not kept.done and len(ref.chain) > len(kept.chain)):
        kept.chain, kept.done, kept.value = (ref.chain, ref.done, ref.value)
    if kept.relation == 'same' and ref.relation in REF_RELATIONS and (ref.relation != 'same'):
        kept.relation, kept.tol = (ref.relation, ref.tol or kept.tol)
    if not kept.start and ref.start:
        kept.start = ref.start
    return kept

def _lg_call(fn, *args, default=None):
    try:
        got = fn(*args)
    except Exception:
        return default
    return default if got is None else got

def _lg_id_like(value) -> bool:
    text = value.strip() if isinstance(value, str) else ''
    return 3 <= len(text) <= 64 and ' ' not in text and bool(re.search('\\d', text))

def _lg_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None

def _lg_arg_ids(args) -> list[str]:
    found = []
    for value in (args or {}).values():
        for one in value if isinstance(value, list) else [value]:
            if _lg_id_like(one) and one not in found:
                found.append(one)
    return found

def _lg_text(node) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return ' '.join((t for t in (_lg_text(v) for v in node.values()) if t))
    if isinstance(node, list):
        return ' '.join((t for t in (_lg_text(v) for v in node) if t))
    return ''

def _lg_ok(body) -> bool:
    if isinstance(body, list):
        return True
    if not isinstance(body, dict):
        return False
    return any((v is True or isinstance(v, (list, dict)) for v in body.values()))

class Ledger:

    def __init__(self):
        self.query = ''
        self.items: list[Item] = []
        self.budget_cents: float | None = None
        self.pick = 'any'
        self.pick_measure = ''
        self.msg_budget: int | None = None
        self.step_budget: int | None = None
        self.in_stock_only = True
        self.nearest: bool | None = None
        self.ask_first: list[str] = []
        self.process: list[str] = []
        self.flags: list[str] = []
        self.seen: dict[str, dict] = {}
        self.order: list[str] = []
        self.opened: set[str] = set()
        self.views: dict[tuple, int] = {}
        self.gone: set[str] = set()
        self.sold_unnamed = False
        self.sold_marked: set[str] = set()
        self.cart: dict[str, int] = {}
        self.cart_mode: str | None = None
        self.asked: list[tuple] = []
        self.revealed: list[Need] = []
        self.refs: list[Ref] = []
        self.notices: list[str] = []
        self.msgs_sent = 0
        self.talk: list[tuple[str, str]] = []
        self._pending = None

    def note_obs(self, toolmap, name, args, obs):
        role = getattr(toolmap, 'role', {}).get(name, 'other')
        body, notes = _lg_call(unwrap, obs, default=(obs, []))
        for note in notes or []:
            if isinstance(note, str) and note.strip() and (note not in self.notices):
                self.notices.append(note)
        if obs is None:
            return
        records = self._records(toolmap, body)
        for row in records:
            self._merge(toolmap, row)
        if role == 'detail' and (records or _lg_ok(body)):
            ids = [_lg_call(toolmap.row_id, r) for r in records] or _lg_arg_ids(args)[:1]
            for ident in ids:
                if ident:
                    self._open(ident)
                    self.views[name, ident] = len(self.notices)
        elif role == 'message':
            self.msgs_sent += 1
            sent = max((v for v in (args or {}).values() if isinstance(v, str)), key=len, default='')
            self.talk.append((sent, _lg_text(body)))
        elif role in ('cart_set', 'cart_view'):
            self._cart(toolmap, role, name, args or {}, body, records)

    def _records(self, toolmap, body) -> list[dict]:
        rows = [r for r in _lg_call(toolmap.rows, body, default=[]) if isinstance(r, dict)]
        if not rows and isinstance(body, dict) and _lg_call(toolmap.row_id, body):
            rows = [body]
        return rows

    def _merge(self, toolmap, row):
        ident = _lg_call(toolmap.row_id, row)
        if not ident:
            return
        if ident not in self.seen:
            self.seen[ident] = {}
            self.order.append(ident)
        self.seen[ident].update(row)
        stock = _lg_call(toolmap.in_stock, row)
        if stock is False:
            self.gone.add(ident)
        elif stock is True and ident not in self.sold_marked:
            self.gone.discard(ident)

    def mark_sold(self, ids):
        for ident in ids or []:
            if ident:
                self.gone.add(ident)
                self.sold_marked.add(ident)

    def _open(self, ident):
        self.opened.add(ident)
        for ref in self.refs:
            for link in ref.chain:
                if link.get('id') == ident:
                    link['opened'] = True

    def _cart(self, toolmap, role, name, args, body, records):
        if not records and (not _lg_ok(body)):
            return
        listed = isinstance(body, list) or (isinstance(body, dict) and any((isinstance(v, list) for v in body.values()))) or any((r is not body for r in records))
        if listed:
            view = {}
            for row in records:
                ident = _lg_call(toolmap.row_id, row)
                qty = self._row_qty(toolmap, row)
                if ident and qty > 0:
                    view[ident] = view.get(ident, 0) + qty
            if role == 'cart_set':
                self._learn_mode(toolmap, name, args, view)
            elif self._pending:
                ident, before, sent = self._pending
                self._infer(before, sent, view.get(ident, 0))
            self._pending = None
            self.cart = view
            return
        if role != 'cart_set':
            return
        if records:
            for row in records:
                ident = _lg_call(toolmap.row_id, row)
                if ident:
                    qty = self._row_qty(toolmap, row)
                    self._set_line(ident, qty)
            return
        ident, sent, remove = self._intent(toolmap, name, args)
        if not ident:
            return
        before = self.cart.get(ident, 0)
        if remove:
            self._set_line(ident, before - sent if sent else 0)
            return
        sent = sent if sent is not None else 1
        if self._mode(toolmap, name) == 'add':
            self._set_line(ident, before + sent)
        else:
            self._set_line(ident, sent)
        if not self.cart_mode and before > 0:
            self._pending = (ident, before, sent)

    def _set_line(self, ident, qty):
        if qty and qty > 0:
            self.cart[ident] = qty
        else:
            self.cart.pop(ident, None)

    def _learn_mode(self, toolmap, name, args, view):
        ident, sent, remove = self._intent(toolmap, name, args)
        if ident and (not remove) and sent:
            self._infer(self.cart.get(ident, 0), sent, view.get(ident, 0))

    def _infer(self, before, sent, after):
        if before <= 0 or not sent:
            return
        if after == before + sent:
            self.cart_mode = 'add'
        elif after == sent:
            self.cart_mode = 'set'

    def _mode(self, toolmap, name) -> str:
        if self.cart_mode:
            return self.cart_mode
        hint = getattr(toolmap, 'cart_semantics', None)
        if hint in ('set', 'add'):
            return hint
        props = self._props(toolmap, name)
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return 'add'
        if any((isinstance(s, dict) and s.get('type') == 'integer' and (s.get('minimum') == 0) for s in props.values())):
            return 'set'
        return 'add'

    def _props(self, toolmap, name) -> dict:
        for tool in getattr(toolmap, 'tools', None) or []:
            fn = tool.get('function', tool) if isinstance(tool, dict) else {}
            if fn.get('name') == name:
                return (fn.get('parameters') or {}).get('properties') or {}
        return {}

    def _intent(self, toolmap, name, args):
        ids = _lg_arg_ids(args)
        ident = next((i for i in ids if i in self.seen), ids[0] if ids else None)
        if not ident:
            return (None, None, False)
        build = getattr(toolmap, 'args_for', None)
        try:
            one = build('cart_set', product_id=ident, qty=1, mode='add') if build else None
            two = build('cart_set', product_id=ident, qty=2, mode='add') if build else None
            off = build('cart_set', product_id=ident, qty=1, mode='remove') if build else None
        except Exception:
            one = two = off = None
        one, two, off = (x if isinstance(x, dict) else {} for x in (one, two, off))
        qty_key = next((k for k in one if k in two and two[k] != one[k]), None)
        mode_key = next((k for k in one if k != qty_key and k in off and (off[k] != one[k])), None)
        sent = _lg_int(args.get(qty_key)) if qty_key in args else None
        if sent is None:
            sent = next((_lg_int(v) for v in args.values() if _lg_int(v) is not None), None)
        if mode_key and mode_key in args:
            remove = args[mode_key] == off[mode_key]
        elif one:
            remove = False
        else:
            props = self._props(toolmap, name)
            remove = any((isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list) and (len(props[k]['enum']) >= 2) and (v in props[k]['enum'][1:]) for k, v in args.items()))
        return (ident, sent, remove or sent == 0)

    def _row_qty(self, toolmap, row) -> int:
        reader = getattr(toolmap, 'qty', None)
        if callable(reader):
            got = _lg_int(_lg_call(reader, row))
            if got is not None:
                return got
        counts = {k: _lg_int(v) for k, v in row.items() if _lg_int(v) is not None}
        if not counts:
            return 1
        values = set(counts.values())
        if len(values) == 1:
            return values.pop()
        price = _lg_call(toolmap.price, row)
        rest = [v for v in counts.values() if v != price]
        return min(rest) if rest else min(values)

    def shown(self, name, ident) -> bool:
        return self.views.get((name, ident)) == len(self.notices) and ident not in self.gone

    def add_ref(self, ref: Ref) -> Ref:
        return fold_ref(self.refs, ref)

    def reveal(self, need: Need):
        if not any((n.item == need.item and n.text == need.text for n in self.revealed)):
            self.revealed.append(need)
        if 0 <= need.item < len(self.items) and (not any((n.text == need.text for n in self.items[need.item].needs))):
            self.items[need.item].needs.append(need)

    def needs_of(self, item: int) -> list[Need]:
        own = list(self.items[item].needs) if 0 <= item < len(self.items) else []
        return own + [n for n in self.revealed if n.item == item and (not any((m.text == n.text for m in own)))]

    def cart_total(self, toolmap):
        total = 0
        for ident, qty in self.cart.items():
            price = _lg_call(toolmap.price, self.seen.get(ident, {}))
            if price is None:
                return None
            total += price * qty
        return total

    def _need_line(self, need: Need) -> str:
        text = need.text
        if need.kind == 'ref':
            ref = next((r for r in self.refs if r.item == need.item and r.prop == need.prop), None)
            if ref and ref.value:
                text += f' (the page chain gives: {ref.value})'
        if need.kind == 'hidden':
            text += ' (told by the shopper)'
        return text

    def summary(self, limit_chars: int) -> str:
        many = len(self.items) > 1
        head = []
        for number, item in enumerate(self.items):
            label = f'item {number + 1}: {item.name}' if many else f'item: {item.name}'
            if item.shelf:
                label += f' [store label {item.shelf}]'
            label += f', quantity {item.qty}'
            if item.cap_cents is not None:
                label += f', at most {item.cap_cents:g} each'
            needs = [self._need_line(n) for n in self.needs_of(number)]
            head.append(label + '; needs: ' + ('; '.join(needs) if needs else 'only the kind of product'))
        if self.budget_cents is not None:
            head.append(f'budget: at most {self.budget_cents:g} in total')
        if self.pick == 'cheapest':
            head.append('pick: the cheapest listing that meets every need')
        elif self.pick == 'best':
            head.append(f"pick: the best {self.pick_measure or 'value'} among listings that meet every need")
        else:
            head.append('pick: any listing that meets every need')
        if self.in_stock_only:
            head.append('in stock only')
        if self.nearest is True:
            head.append('if no listing meets every need: order the nearest one, missing the fewest needs (the request asks for it)')
        limit = 'no limit stated' if self.msg_budget is None else f'{self.msg_budget} allowed'
        head.append(f'messages to the shopper: {self.msgs_sent} sent, {limit}')
        head += [f'check with the shopper first: {a}' for a in self.ask_first]
        head += [f'rule of the request: {p}' for p in self.process]
        for ref in self.refs:
            if not ref.done:
                opened = sum((1 for link in ref.chain if link.get('opened')))
                whose = f' of listing {ref.owner}' if ref.owner else ''
                head.append(f'open page chain for item {ref.item + 1} {ref.prop}{whose}: starts at {ref.start}; {opened} of {len(ref.chain)} known pages opened')
        if self.cart:
            head.append('cart: ' + ', '.join((f'{i} x{q}' + (' (sold out)' if i in self.gone else '') for i, q in self.cart.items())))
        else:
            head.append('cart: empty')
        head += [f'notice: {n}' for n in self.notices[-4:]]
        text = '\n'.join(('- ' + line for line in head))
        if len(text) >= limit_chars:
            return text[:max(0, limit_chars - 3)] + '...'
        lines = []
        for ident in reversed(self.order):
            row = self.seen[ident]
            title = max((v for v in row.values() if isinstance(v, str) and len(v) <= 200), key=lambda v: len(v.split()), default='')
            mark = 'opened' if ident in self.opened else 'row'
            stock = ', sold out' if ident in self.gone else ''
            lines.append(f'  {ident} [{mark}{stock}] {title[:70]}')
        room = limit_chars - len(text) - 20
        shown = []
        for line in lines:
            if room - len(line) - 1 < 0:
                break
            shown.append(line)
            room -= len(line) + 1
        if shown:
            text += f'\n- listings seen ({len(shown)} of {len(lines)}, newest first):\n' + '\n'.join(shown)
        return text[:limit_chars]

    def to_json(self) -> str:
        return json.dumps({'items': [{'name': i.name, 'qty': i.qty, 'shelf': i.shelf, 'cap': i.cap_cents, 'needs': [n.__dict__ for n in i.needs]} for i in self.items], 'budget': self.budget_cents, 'pick': self.pick, 'messages': self.msg_budget, 'steps': self.step_budget, 'refs': [r.__dict__ for r in self.refs], 'process': self.process, 'ask_first': self.ask_first, 'flags': self.flags}, ensure_ascii=False)
READ_ASK = 'You turn one shopper\'s request into a work order for an agent that shops in an online store for them. Answer with a single JSON object only (no prose, no code fence), shaped like this:\n{"items": [{"name": str, "shelf": str, "qty": int, "cap": number or null, "needs": [NEED, ...]}],\n "budget": number or null, "pick": "cheapest" | "best" | "any", "measure": str,\n "messages": int or null, "steps": int or null, "in_stock_only": true | false, "nearest": true | false | null,\n "ask_first": [str], "process": [STEP, ...]}\nNEED = {"prop": str, "kind": "stated" | "excluded" | "ref", "value": str,\n        "relation": "same" | "max" | "min" | "near" | "not" | "fits" | "", "start": str, "tol": str,\n        "when": null or {"item": int, "prop": str, "values": str, "unless": true | false}}\nSTEP = {"rule": str, "due": "choose" | "order" | "end" | "any", "messages": int}\n\nRead the whole request before answering. It is chatty: limits, goals and requirements are scattered between stories, may be put as questions or polite asks, and may come back later in other words. A sentence that asks something of the agent still sets a rule.\n\nITEMS\n- One entry for each product to buy that has a quantity or requirements of its own in the request. A product named in one phrase that contains "and" or "&" (often a store department\'s name) is one item.\n- "name": what sort of product it is, worded as the request words it. Words that belong to the product\'s name (a season or an audience named with the product) stay in the name and are not also needs.\n- "shelf": the entry of LABELS that names this kind of product, copied exactly; "" when none does or no LABELS are given.\n- "qty": units of that product; 1 when the request gives no number. A pair, set or pack sold as one listing is 1 unless the request wants several of them.\n- "cap": a price limit on one unit of this item alone (each, apiece, per unit), as a plain number; null when there is none.\n\nNEEDS (firm requirements on one item; each NEED is about exactly one property)\n- "prop": the property in the request\'s own words, never swapped for a related property. "value": the wanted value, limit, range or closed set of allowed values, in the request\'s words, keeping numbers and units ("at least <n> <unit>", "<a> or <b>").\n- kind "stated": the request gives the value, limit, range or allowed set.\n- kind "excluded": the request rules one value or several values out; "value" is only what is ruled out.\n- kind "ref": the value must be read from the page of another listing that the request refers to. "relation": "same" (equal to or matching that value), "max" (that value or less), "min" (that value or more), "near" (within a margin on both sides of it; put the margin, with its unit, in "tol"), "not" (any value except the one that page shows), "fits" (compatible or working together with that value). "start": the whole description of that other listing as the request gives it: its kind, its shop or seller, and every property and value used to pick it out; leave out the comparison words and any clause that lets that page hand you on to a further listing. Leave "value" "". Such a clause is part of the same ref, never a need of its own.\n- The other listing is often of a different kind from the item, and the words that pick it out describe THAT listing only. A need of the item on a property also named in that description (the item\'s own "<p> not <v>" next to a description that has "<p> <v>") is still a need of the item.\n- One clause can hold a ref and more needs of the same item ("<prop A> near the value on the page of <listing>, and <prop B> at least <m>"): give each its own NEED.\n- A requirement that an item must fit, suit or work with another requested item, or with the shopper\'s own equipment, is kind "stated" with "prop": "fits" and "value" naming the other item ("item <n>") or the equipment exactly as written.\n- A requirement that holds only when another requested item turns out to have some value of a property, or holds except in that case, goes on the item it constrains, with "when": {"item": the other item\'s number counted from 1, "prop": that property, "values": those values as written, "unless": true when the requirement holds except in that case, false when it holds only in that case}. Otherwise "when" is null.\n- A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a need on that property ("occasion": "business"), unless the store\'s label for the item already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay story.\n- Not needs: price, budget, stock, quantity, delivery or timing, the story (who it is for, where it goes, an earlier mishap, the event it is bought for), wishes, and sentences that point at a property without giving its value (those may be worth asking about, see "ask_first").\n- Give every requirement once. Fill "relation", "start" and "tol" only for kind "ref" (else "").\n\nLIMITS AND GOAL\n- "budget": the most the whole order may cost, as a plain number in the request\'s own money unit (no conversion, no separators); null when not given. It may be called a ceiling, a total, a pot or a spend, or be asked as whether some amount works. A limit on one unit\'s price is that item\'s "cap", not the budget.\n- "messages": the most messages the agent may send the shopper, however it is put: a count of messages, notes, questions or turns, a point after which the shopper stops replying, or messages beyond some number not going through. null when the request does not say. Messages are not steps.\n- "steps": the largest number of steps, moves, actions or tool calls allowed for the whole job, including a number to stop after or to finish within; null when the request does not say.\n- When the request states one limit twice with different numbers, give the smaller.\n- "pick": "cheapest" if the shopper wants the lowest price or lowest total, or says price settles the choice between fitting listings; "best" if the shopper wants the largest or smallest amount of some other measure (name it in "measure"); otherwise "any". Saying money is tight is not a pick.\n- "in_stock_only": false only when the request accepts listings that cannot be bought now; asking to skip, leave out or set aside sold-out listings means true.\n- "nearest": true when the request says that, if no listing meets every requirement, the agent should still offer or buy the nearest one (the one missing the smallest number of them); false when it says to buy nothing in that case; null when it says neither.\n\nRULES OF WORK\n- "ask_first": what the request explicitly wants the agent to ask or confirm with the shopper ahead of picking (a quality it hints matters, details it says the agent will need), each in at most 10 words naming the property when it names one; [] when none.\n- "process": every other instruction on the way of working or of finishing (what to do or say when nothing fits, whether to name alternatives, what to confirm, show or report to the shopper), each a STEP: "rule" in at most 20 words; "due" is "choose" (before any product is picked), "order" (just before the purchase is placed), "end" (when closing with no purchase) or "any"; "messages" is the number of messages to the shopper the step needs (0 if none). [] when none. Limits, stock and item requirements are not process rules.'
NOTE_ASK_PLAIN = 'A shopping agent is working on a shopper\'s order. A tool result just carried the note below, from the shopper or from the store. Say what it changes in the work order. Answer with a single JSON object only (no prose, no code fence):\n{"kind": "change" | "error" | "other", "budget": number or null, "add": [{"item": int, "prop": str, "value": str}], "drop": [{"item": int, "prop": str}], "gone": [str]}\n- "kind": "change" when the note alters the job (a new total limit, a requirement added, removed or replaced, a listing no longer for sale, a changed price, a refused checkout that reports any of these); "error" when all it reports is a faulty tool call; "other" otherwise.\n- "budget": the most the whole order may now cost, as the note\'s figure, a plain number in the request\'s money unit; null when the note sets no new total. Read the figure itself: friendly or worried wording does not tell whether it went up or down.\n- "add": each requirement the note adds; "item" is the item\'s number in the work order (from 1), "prop" the property and "value" the wanted value or limit in the note\'s words. When the note swaps one requirement for another, list the earlier one under "drop" and its successor under "add".\n- "drop": each requirement the note says is no longer wanted, with "prop" as the work order words it.\n- "gone": ids of listings the note says can no longer be bought, exactly as written; [] when it names none.'
NOTE_ASK = NOTE_ASK_PLAIN.replace('"gone": [str]}', '"gone": [str], "sold_out": true | false}') + '\n- "sold_out": true when the note says a listing in the cart or the order is gone, sold out or can no longer be bought, even without naming it (an unnamed "that pick", "it" or "the item" there means the cart\'s listing); a new money limit or a changed requirement alone is not that; false otherwise.'
UNMIX_ASK = 'A shopper\'s request was read into needs on one product. Some needs take their value from another listing the request points at, and the words that pick that other listing out (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as needs of the product being bought. Using the request\'s wording, say which of the numbered OTHER NEEDS only describe that other listing. Answer with a single JSON object only (no prose, no code fence): {"describe_other": [the numbers of those needs; [] when every one is a real need of the product]}'
PICKS = ('cheapest', 'best', 'any')
STEP_FLOOR = 10
DUES = ('choose', 'order', 'end', 'any')
_BF_STEP_TAG = re.compile('\\[due: (\\w+); messages: (\\d+)\\]$')

def _bf_json(got):
    if isinstance(got, dict) or got is None:
        return got
    if not isinstance(got, str):
        return None
    left, right = (got.find('{'), got.rfind('}'))
    if 0 <= left < right:
        try:
            got = json.loads(got[left:right + 1])
        except ValueError:
            return None
        return got if isinstance(got, dict) else None
    return None

def _bf_num(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and re.fullmatch('\\s*\\d[\\d,]*(\\.\\d+)?\\s*', value):
        return float(value.replace(',', ''))
    return None

def _bf_count(value):
    got = _bf_num(value)
    return int(got) if got is not None and float(got).is_integer() and (got >= 0) else None

def _bf_money(value):
    got = _bf_num(value)
    return got if got is not None and got > 0 else None

def _bf_step(raw):
    if isinstance(raw, str):
        return raw.strip() or None
    if not isinstance(raw, dict):
        return None
    rule = str(raw.get('rule') or '').strip()
    if not rule:
        return None
    due = raw.get('due') if raw.get('due') in DUES else 'any'
    count = _bf_count(raw.get('messages')) or 0
    return f'{rule} [due: {due}; messages: {count}]'

def step_messages(process, dues=DUES) -> int:
    total = 0
    for line in process or []:
        tag = _BF_STEP_TAG.search(str(line))
        if tag and tag.group(1) in dues:
            total += int(tag.group(2))
    return total

def step_messages_most(process, dues=DUES) -> int:
    most = 0
    for line in process or []:
        tag = _BF_STEP_TAG.search(str(line))
        if tag and tag.group(1) in dues:
            most = max(most, int(tag.group(2)))
    return most

def check_brief(got, query: str='') -> list[str]:
    if not isinstance(got, dict):
        return ['no JSON object']
    items = got.get('items')
    if not isinstance(items, list) or not items:
        return ['no items']
    faults = []
    for n, entry in enumerate(items, 1):
        if not isinstance(entry, dict) or not str(entry.get('name') or '').strip():
            faults.append(f'item {n} has no name')
            continue
        if entry.get('cap') is not None and _bf_money(entry.get('cap')) is None:
            faults.append(f"item {n} cap {entry.get('cap')!r} is not a positive number")
        needs = entry.get('needs') if entry.get('needs') is not None else []
        if not isinstance(needs, list):
            faults.append(f'item {n} needs is not a list')
            continue
        for need in needs:
            if not isinstance(need, dict) or not str(need.get('prop') or '').strip():
                faults.append(f'item {n} has a need without a property')
            elif need.get('kind') not in NEED_KINDS:
                faults.append(f"item {n} need kind {need.get('kind')!r}")
            elif need.get('kind') == 'ref' and (need.get('relation') not in REF_RELATIONS or not str(need.get('start') or '').strip()):
                faults.append(f"item {n} ref on {need.get('prop')} lacks relation or start")
            elif need.get('when') and (not (isinstance(need['when'], dict) and _bf_count(need['when'].get('item')) and str(need['when'].get('prop') or '').strip())):
                faults.append(f"item {n} need on {need.get('prop')} has a malformed condition")
    if got.get('budget') is not None and _bf_money(got.get('budget')) is None:
        faults.append(f"budget {got.get('budget')!r} is not a positive number")
    for key in ('messages', 'steps'):
        if got.get(key) is not None and _bf_count(got.get(key)) is None:
            faults.append(f'{key} is not a count')
    if got.get('pick') not in PICKS:
        faults.append(f"pick {got.get('pick')!r}")
    if got.get('process') is not None and (not isinstance(got.get('process'), list)):
        faults.append('process is not a list')
    return faults

def _bf_words(text) -> str:
    return ' '.join(re.findall('[a-z0-9]+', re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(text or '')).lower()))

def _bf_entries(got) -> list[dict]:
    return [e for e in (got or {}).get('items') or [] if isinstance(e, dict) and str(e.get('name') or '').strip()]

def _bf_twin_shelves(got) -> bool:
    shelves = [w for w in (_bf_words(e.get('shelf')) for e in _bf_entries(got)) if w]
    return len(set(shelves)) < len(shelves)

def _bf_needs(entry) -> list[dict]:
    raw = entry.get('needs')
    return [n for n in raw if isinstance(n, dict)] if isinstance(raw, list) else []

def _bf_clash(one: dict, two: dict) -> bool:

    def wants(entry):
        return {(_bf_words(n.get('prop')), n.get('kind'), _bf_words(n.get('value')), _bf_words(n.get('start'))) for n in _bf_needs(entry) if _bf_words(n.get('prop'))}
    mine, theirs = (wants(one), wants(two))
    return any((a[0] == b[0] and a != b for a in mine for b in theirs))

def _bf_fold(entries: list[dict]):
    kept, became = ([], {})
    for old, entry in enumerate(entries):
        shelf = _bf_words(entry.get('shelf'))
        into = next((k for k, other in enumerate(kept) if shelf and shelf == _bf_words(other.get('shelf')) and (not _bf_clash(other, entry))), None)
        if into is None:
            became[old] = len(kept)
            kept.append(dict(entry, needs=_bf_needs(entry)))
        else:
            became[old] = into
            kept[into]['needs'] += [n for n in _bf_needs(entry) if n not in kept[into]['needs']]
    return (kept, became)

def _bf_labels(toolmap) -> list[str]:
    labels = []
    for tool in getattr(toolmap, 'tools', None) or []:
        fn = tool.get('function', tool) if isinstance(tool, dict) else {}
        for spec in ((fn.get('parameters') or {}).get('properties') or {}).values():
            for value in spec.get('enum') or [] if isinstance(spec, dict) else []:
                if isinstance(value, str) and value not in labels:
                    labels.append(value)
    return labels[:120]

def _bf_need_text(prop, kind, value, relation='', start='', tol=''):
    if kind == 'excluded':
        return f'{prop}: not {value}'
    if kind == 'ref':
        margin = f' (within {tol})' if tol else ''
        return f"{prop}: '{relation}'{margin} compared with the value on the page of {start}, following that page's pointers to the last listing"
    return f'{prop}: {value}'

def _bf_build(got: dict, query: str, ledger: Ledger):
    read = _bf_entries(got)
    entries, became = _bf_fold(read)
    for old, new in became.items():
        if new < old:
            ledger.flags.append(f"item {old + 1} '{str(read[old]['name']).strip()}' folded into item {new + 1}: same store label")
    for entry in entries:
        qty = _bf_count(entry.get('qty'))
        ledger.items.append(Item(str(entry['name']).strip(), qty if qty else 1, [], str(entry.get('shelf') or '').strip(), _bf_money(entry.get('cap'))))
    count = len(ledger.items)
    for number, entry in enumerate(entries):
        item = ledger.items[number]
        for raw in entry['needs']:
            if raw.get('kind') not in NEED_KINDS or not str(raw.get('prop') or '').strip():
                continue
            prop, kind = (str(raw['prop']).strip(), raw['kind'])
            value = _bf_renumber(str(raw.get('value') or '').strip(), became)
            relation, start, tol = (str(raw.get('relation') or ''), str(raw.get('start') or '').strip(), str(raw.get('tol') or '').strip())
            if kind == 'ref' and (relation not in REF_RELATIONS or not start):
                continue
            when = raw.get('when') if isinstance(raw.get('when'), dict) else None
            if when:
                other = _bf_count(when.get('item'))
                other = became[other - 1] + 1 if other and 1 <= other <= len(read) else None
                when = {'item': other - 1, 'prop': str(when.get('prop') or '').strip(), 'values': str(when.get('values') or '').strip(), 'unless': when.get('unless') is True} if other and 1 <= other <= count and (other - 1 != number) else None
            text = _bf_need_text(prop, kind, value, relation, start, tol)
            if when:
                text += f" ({('unless' if when['unless'] else 'only if')} item {when['item'] + 1} {when['prop']} is {when['values']})"
            if any((n.text == text for n in item.needs)):
                continue
            item.needs.append(Need(number, text, kind, None if kind == 'ref' else value, prop, when))
            if kind == 'ref':
                add = getattr(ledger, 'add_ref', None)
                (add if callable(add) else ledger.refs.append)(Ref(number, prop, relation, start, tol=tol))
    _bf_self_fit(ledger)
    _bf_unmix(ledger)
    ledger.budget_cents = _bf_money(got.get('budget'))
    pick = got.get('pick')
    ledger.pick = pick if pick in PICKS else 'any'
    ledger.pick_measure = str(got.get('measure') or '').strip() if ledger.pick == 'best' else ''
    ledger.in_stock_only = got.get('in_stock_only') is not False
    near = got.get('nearest')
    ledger.nearest = near if isinstance(near, bool) else None
    ledger.ask_first = [str(a).strip() for a in got.get('ask_first') or [] if str(a).strip()] if isinstance(got.get('ask_first'), list) else []
    steps = got.get('process') if isinstance(got.get('process'), list) else []
    ledger.process = [s for s in (_bf_step(raw) for raw in steps) if s]
    ledger.msg_budget = _bf_count(got.get('messages'))
    steps = _bf_count(got.get('steps'))
    ledger.step_budget = steps if steps is not None and steps >= STEP_FLOOR else None

def _bf_renumber(value: str, became: dict) -> str:
    named = re.fullmatch('item (\\d+)', value)
    if not named or int(named.group(1)) - 1 not in became:
        return value
    return f'item {became[int(named.group(1)) - 1] + 1}'

def _bf_self_fit(ledger: Ledger):
    for number, item in enumerate(ledger.items):
        item.needs = [n for n in item.needs if not (n.prop == 'fits' and n.value == f'item {number + 1}')]

def _bf_unmix_ask(ledger: Ledger, llm):
    for number, item in enumerate(ledger.items):
        refs = [r for r in ledger.refs if r.item == number]
        rest = [n for n in item.needs if n.kind == 'stated' and n.prop != 'fits']
        if not refs or not rest:
            continue
        user = 'REQUEST:\n' + ledger.query + f'\n\nPRODUCT: {item.name}\nNEEDS POINTING AT ANOTHER LISTING:\n' + '\n'.join((f'- {r.prop}: from the listing {r.start}' for r in refs)) + '\nOTHER NEEDS:\n' + '\n'.join((f'{k}. {n.text}' for k, n in enumerate(rest, 1)))
        try:
            got = _bf_json(llm.ask(UNMIX_ASK, user, heavy=True, max_tokens=1500, want_json=True))
        except Exception:
            got = None
        named = got.get('describe_other') if isinstance(got, dict) else None
        picked = {_bf_count(k) for k in named} if isinstance(named, list) else set()
        moved = [n for k, n in enumerate(rest, 1) if k in picked]
        if not moved:
            continue
        ref = refs[0]
        start = ref.start + ' (that listing: ' + '; '.join((n.text for n in moved)) + ')'
        for need in item.needs:
            if need.kind == 'ref' and need.prop == ref.prop and (ref.start in need.text):
                need.text = need.text.replace(ref.start, start, 1)
        item.needs = [n for n in item.needs if n not in moved]
        ledger.flags.append(f'moved {len(moved)} need(s) of item {number + 1} into the listing its ref points at')
        ref.start = start

def _bf_unmix(ledger: Ledger):
    for number, item in enumerate(ledger.items):
        starts = [r.start.lower() for r in ledger.refs if r.item == number]
        if not starts:
            continue
        kept = []
        for need in item.needs:
            if need.kind == 'stated' and need.value and any((need.prop.lower() in s and need.value.lower() in s for s in starts)):
                ledger.flags.append(f"dropped '{need.text}' from item {number + 1}: it describes the referenced listing")
                continue
            kept.append(need)
        item.needs = kept

def read_notes(ledger: Ledger, llm, notes, stock=True) -> list[str]:
    text = '\n'.join((n for n in notes if isinstance(n, str) and n.strip()))
    if not text or not ledger.items:
        return []
    user = 'WORK ORDER NOW:\n' + ledger.summary(2500) + '\n\nTHE NOTE:\n' + text[:2000]
    try:
        got = _bf_json(llm.ask(NOTE_ASK if stock else NOTE_ASK_PLAIN, user, heavy=False, max_tokens=1200, want_json=True))
    except Exception:
        got = None
    changed = []
    if stock and isinstance(got, dict) and (got.get('sold_out') is True):
        named = [str(g).strip() for g in got.get('gone') or [] if isinstance(g, (str, int))] if isinstance(got.get('gone'), list) else []
        if not any((g in (getattr(ledger, 'cart', None) or {}) for g in named)):
            ledger.sold_unnamed = True
            changed.append('a listing in the cart can no longer be bought (the note does not say which)')
    if not isinstance(got, dict) or got.get('kind') != 'change':
        return changed
    budget = _bf_money(got.get('budget'))
    if budget is not None and budget != ledger.budget_cents:
        ledger.budget_cents = budget
        changed.append(f'budget now {budget:g}')
    count = len(ledger.items)
    for raw in got.get('drop') if isinstance(got.get('drop'), list) else []:
        number = _bf_count(raw.get('item')) if isinstance(raw, dict) else None
        prop = str(raw.get('prop') or '').strip().lower() if isinstance(raw, dict) else ''
        if not prop or not number or (not 1 <= number <= count):
            continue
        item = ledger.items[number - 1]
        kept = [n for n in item.needs if n.prop.strip().lower() != prop]
        if len(kept) < len(item.needs):
            item.needs = kept
            ledger.revealed = [n for n in ledger.revealed if not (n.item == number - 1 and n.prop.strip().lower() == prop)]
            changed.append(f'item {number}: {prop} no longer required')
    for raw in got.get('add') if isinstance(got.get('add'), list) else []:
        number = _bf_count(raw.get('item')) if isinstance(raw, dict) else None
        prop = str(raw.get('prop') or '').strip() if isinstance(raw, dict) else ''
        value = str(raw.get('value') or '').strip() if isinstance(raw, dict) else ''
        if not prop or not value or (not number) or (not 1 <= number <= count):
            continue
        item = ledger.items[number - 1]
        text = _bf_need_text(prop, 'stated', value)
        if not any((n.text == text for n in item.needs)):
            item.needs.append(Need(number - 1, text, 'stated', value, prop))
            changed.append(f'item {number}: now needs {text}')
    for ident in got.get('gone') if isinstance(got.get('gone'), list) else []:
        ident = str(ident).strip()
        if ident in ledger.seen and ident not in ledger.gone:
            ledger.gone.add(ident)
            changed.append(f'{ident} can no longer be bought')
    return changed

def read_request(query: str, llm, toolmap) -> Ledger:
    ledger = Ledger()
    ledger.query = query or ''
    labels = _bf_labels(toolmap)
    user = 'LABELS: ' + (json.dumps(labels, ensure_ascii=False) if labels else 'none') + '\n\nREQUEST:\n' + ledger.query
    best, best_faults = (None, ['no reading'])
    for heavy, tokens in ((False, 3000), (True, 2500)):
        try:
            got = _bf_json(llm.ask(READ_ASK, user, heavy=heavy, max_tokens=tokens, want_json=True))
        except Exception:
            got = None
        faults = check_brief(got, ledger.query)
        if got is not None and (best is None or len(faults) <= len(best_faults)):
            best, best_faults = (got, faults)
        if not faults and (heavy or not _bf_twin_shelves(got)):
            break
        if not faults:
            ledger.flags.append('light reading put two items on one store label: read again heavy')
            continue
        ledger.flags.append(('heavy' if heavy else 'light') + ' reading rejected: ' + '; '.join(faults[:4]))
    if best is not None:
        _bf_build(best, ledger.query, ledger)
        _bf_unmix_ask(ledger, llm)
    if not ledger.items:
        ledger.items = [Item('the requested product', 1, [])]
    return ledger
C_PLAN = 'You plan the questions a shopping assistant may send to the shopper before it buys. The shopper keeps a few unwritten wishes, each tied to a single property of a single requested item, and states one only when a question names that property for that item. A question that touches two properties or two items, asks something open, or asks about anything else is wasted, so every pick is one property of one item.\nRank the picks this way:\n1. Whatever the request itself says should be checked with the shopper (listed under CHECK FIRST when there is any).\n2. Properties the story points at without giving a value (any sentence of the request that points at one of the listed keys without saying its value): a remark about who will use the item, where it goes, what it has to look like or match, a choice the shopper hands over to someone else or feels unsure about, or an earlier problem. Tie each remark to the one item and the one property it is about; a remark usually concerns one item only.\n3. Then the keys listed under KEYS for each item, numbered in the order their chance of holding an unwritten wish was judged from the listings. Prefer the properties that someone buying that kind of product decides on before anything else (often its size, capacity or dimensions, its type, its material or its colour, whichever that kind of product is bought by), and keys most of the item\'s listings carry; when an item has five or more listings, skip keys that appear on just one or two listings.\nNever pick a property whose value the request already gives for that item, one asked before, a different wording of a property you already picked, a key whose values mix several purchase variants, identifiers, or logistics (seller, price, stock, delivery, warranty).\nThe name decides whether the shopper understands: the wishes carry the names the store gives its listing keys. When a pick is a listed key, write the key exactly as it stands under KEYS, every word and nothing added. Only a property the story points at that no listed key covers gets a name of your own, and then the two or three plain words a listing would use as its key. Never put values, a second property or the item\'s name into "property".\nReply with a single JSON object only, shaped {"ask": [{"item": <item number>, "property": "...", "hint": "the story words it comes from, or empty"}]}, holding at most COUNT picks, best first.'
C_READ = 'A shopping assistant sent the shopper one question and got the reply below. Reply with a single JSON object only, shaped\n{"class": "reveal" or "miss" or "multi" or "refused", "property": "...", "need": "property: constraint", "points": true or false, "relation": "...", "start": "..."}\n"class":\n- "reveal": the reply states a firm wish or limit (a value, a range, something to avoid, or a value to be read off another listing), whatever property it names. Homely or indirect phrasing that still carries a value counts as a reveal.\n- "miss": no value is given: the shopper does not mind, has no view, leaves the choice to the assistant, or says the message was unclear or asks to hear it again.\n- "multi": the shopper says the message was about too many things at once or asks for one thing at a time.\n- "refused": the text is not the shopper answering but the store rejecting the message (a limit reached, an error).\nOnly for "reveal":\n- "property": the property the reply itself talks about, in the reply\'s words (this can be another property than the one asked);\n- "need": "property: constraint" with the constraint copied word for word, keeping its nouns and units; a reply that turns down every value but one (anything except it is refused) wants that one value, so the constraint is that value or limit itself, never "other than" it;\n- "points": true if the wanted value has to be looked up on another listing; then "relation" says how the value compares with that listing\'s (same, max, min, near, not or fits) and "start" copies the reply\'s description of that listing exactly, starting at the comparing words and running to the end of the reply.'
C_REPICK = 'You decide which property to ask a shopper about, for one item of their order.\n\nThe shopper has hidden requirements on one or two properties of the items they want. A hidden requirement is revealed only when you ask about that exact property; a question about any other property is wasted.\n\nYou get the shopper\'s request, the item, the properties already asked about for this item without finding anything, and candidate properties seen on the item\'s listings with sample values.\n\nHow to judge:\n- A property the request already states for this item is not hidden. Rank it last.\n- A hidden requirement usually leaves an indirect trace in the request: why the shopper needs the item, where it will go or be used, what it must match, fit or work with, who it is for, what went wrong last time. Work out which property that trace constrains for this item.\n- When the request says little or nothing about this item beyond its name, the hidden requirement tends to be one of the item\'s main specifications, not a minor detail.\n- Between equally likely properties, prefer the one whose values differ in ways that would matter to this shopper.\n\nReply with JSON only: {"order": [candidate numbers, best first]}, listing every candidate.'
_IV_ONE = re.compile('\\s+(?:and|or)\\s+|[/,;(&]')
IV_ORDER = 'model'
IV_RANK_LAST = (3, 6)
IV_KEY_PRIOR = (0.24, 0.116, 0.07)
IV_KEY_PRIOR_SAID = (0.2, 0.058, 0.026)
IV_ITEM_PRIOR = (0.58, 0.72, 0.4, 0.22)
IV_POS_FACTOR = (1.0, 1.0, 0.79, 0.34)
IV_P_CAP = 0.95
IV_FILL = 'tuple'
IV_REPICK = True
IV_REPICK_K = 10
IV_REPICK_VALS = 6
IV_GUARD_KEYS = 3
IV_GUARD_SHARE = 0.0
IV_GUARD_GAP = 0.34
IV_PAIR = True
IV_WIDE = 8
IV_REORDER = None
IV_LONGER = True
SHELF_KEYS = False
KEY_CAP = 14
FIT_MESSAGE = True
IV_PLAN_EACH = True
IV_PLAN_TOKENS = 3500
IV_SETTLE = True
IV_SETTLE_KEEP = 3
IV_REFILL = True

def iv_plan_count(left, items):
    left = max(1, left)
    if not IV_PLAN_EACH or items <= 0:
        return left + 2
    return max(left + 2, min(IV_WIDE + left - 1, 2 * IV_WIDE))

def iv_key_prior(rank, said):
    bucket = next((b for b, last in enumerate(IV_RANK_LAST) if rank <= last), len(IV_RANK_LAST))
    return (IV_KEY_PRIOR_SAID if said else IV_KEY_PRIOR)[bucket]

def iv_item_priors(n):
    if n <= 0:
        return []
    row = IV_ITEM_PRIOR[min(n, len(IV_ITEM_PRIOR)) - 1]
    factors = [IV_POS_FACTOR[min(k, len(IV_POS_FACTOR) - 1)] for k in range(n)]
    mean = sum(factors) / n
    return [min(IV_P_CAP, row * f / mean) for f in factors]

def iv_any_hidden(scale, bases):
    stay = 1.0
    for b in bases:
        stay *= 1.0 - min(IV_P_CAP, scale * b)
    return 1.0 - stay

def iv_scale(bases, q):
    q = min(max(q, 0.0), IV_P_CAP)
    if not bases or q <= 0.0:
        return 0.0
    lo, hi = (0.0, IV_P_CAP / min(bases))
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if iv_any_hidden(mid, bases) < q else (lo, mid)
    return (lo + hi) / 2

def iv_gf_order(groups):
    out = []
    for item, q, keys in groups:
        bases = [iv_key_prior(rank, said) for rank, (_, said) in enumerate(keys, 1)]
        s = iv_scale(bases, q)
        out += [(min(IV_P_CAP, s * b), rank, item, key) for rank, ((key, _), b) in enumerate(zip(keys, bases), 1)]
    return sorted(out, key=lambda c: (-c[0], c[1], c[2]))

def _iv_words(text):
    text = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(text or ''))
    return re.sub('[^0-9a-z]+', ' ', text.lower()).strip()

def _iv_stems(text):
    return {w[:-1] if w.endswith('s') else w for w in _iv_words(text).split() if len(w) >= 4}

def _iv_wordset(text):
    return {w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w for w in _iv_words(text).split()}

def _iv_twins(one, other):
    a, b = (_iv_wordset(one), _iv_wordset(other))
    return bool(a and b) and (a <= b or b <= a)

def _iv_leaves(node, depth=0):
    if depth > 4:
        return
    if isinstance(node, dict):
        if depth and node and all((not isinstance(v, (dict, list)) for v in node.values())):
            yield node
        for v in node.values():
            yield from _iv_leaves(v, depth + 1)
    elif isinstance(node, list):
        for v in node:
            yield from _iv_leaves(v, depth + 1)

def _iv_facts(record, ident):
    best, where = ({}, None)
    for block in _iv_leaves(record):
        scal = {k: v for k, v in block.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
        if ident in [str(v) for v in scal.values()]:
            continue
        if len(scal) >= 2 and len(scal) > len(best):
            best, where = (scal, block)
    return (best, where)

def _iv_strings(node, skip, out, depth=0, longer_than=0):
    if depth > 4 or node is skip:
        return out
    if isinstance(node, str):
        if len(node.split()) > longer_than:
            out.append(node.strip())
    elif isinstance(node, dict):
        for v in node.values():
            _iv_strings(v, skip, out, depth + 1, longer_than)
    elif isinstance(node, list):
        for v in node:
            _iv_strings(v, skip, out, depth + 1, longer_than)
    return out

def _iv_shared_word(values):
    seen = set()
    for v in values:
        words = set(re.findall('[a-z]{3,}', str(v).lower()))
        if words & seen:
            return True
        seen |= words
    return False

def _iv_has(phrase, text):
    phrase = _iv_words(phrase)
    return bool(phrase) and f' {phrase} ' in f' {_iv_words(text)} '

def _iv_norm(text):
    return ' '.join((w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w for w in _iv_words(text).split()))

def _iv_label_fits(label, item):
    lab = _iv_norm(label)
    shelf, name = (_iv_norm(getattr(item, 'shelf', '') or ''), _iv_norm(getattr(item, 'name', '') or ''))
    return bool(lab) and (lab in (shelf, name) or bool(_iv_stems(label) & (_iv_stems(shelf) | _iv_stems(name))))

def _iv_names_item(sentence, name):
    want = _iv_stems(name)
    return bool(want) and want <= _iv_stems(sentence)

def iv_item_text(request, own=(), others=()):
    sents = [s for s in re.split('(?<=[.?!;])\\s+', str(request or '')) if s.strip()]
    keep = [s for s in sents if any((_iv_names_item(s, n) for n in own if n)) or not any((_iv_names_item(s, n) for n in others if n))]
    return ' '.join(keep)

def iv_fill_order(rows, request, own=(), others=(), claimed=()):
    text = f' {_iv_norm(iv_item_text(request, own, others))} '
    keyed = []
    for k, (name, n, vals, seen, digits) in enumerate(rows):
        taken = {_iv_norm(v) for key, v in claimed if _iv_norm(key) != _iv_norm(name)}
        named = f' {_iv_norm(name)} ' in text or any((len(v) >= 3 and _iv_norm(v) and (_iv_norm(v) not in taken) and (f' {_iv_norm(v)} ' in text) for v in vals))
        rank = 1 + sum((1 for other in rows if other[1] > n))
        bucket = next((b for b, last in enumerate(IV_RANK_LAST) if rank <= last), len(IV_RANK_LAST))
        keyed.append(((named, len(vals) <= 1, bucket, 2 * digits >= max(1, seen), -n, k), k))
    return [rows[k] for _, k in sorted(keyed)]

class Interview:

    def __init__(self, ledger, llm, toolmap, env, reserve=0):
        self.led, self.llm, self.tm, self.env = (ledger, llm, toolmap, env)
        self.reserve = reserve
        self.queue = []
        self.planned = False
        self.closed = False
        self.readings = {}
        self.narrowed = set()
        self.one_key = False
        self.transcript = []
        self.value_counts = {}
        self.fills = set()
        self.repicked = set()
        self.fill_table = ({}, {})
        self.held = False
        self.misses = {}
        self.refilled = False

    def needs(self, item):
        fn = getattr(self.led, 'needs_of', None)
        return fn(item) if callable(fn) else list(self.led.items[item].needs)

    def store_labels(self):
        if getattr(self, '_labels', None) is None:
            out = set()
            for tool in getattr(self.tm, 'tools', None) or []:
                fn = tool.get('function', tool) if isinstance(tool, dict) else {}
                props = (fn.get('parameters') or {}).get('properties') or {} if isinstance(fn, dict) else {}
                for spec in props.values():
                    for value in spec.get('enum') or [] if isinstance(spec, dict) else []:
                        if isinstance(value, str) and _iv_norm(value):
                            out.add(_iv_norm(value))
            self._labels = out
        return self._labels

    def labels_meet(self):
        size = len(self.led.seen or {})
        if getattr(self, '_meet', (None, None))[0] != size:
            meet = False
            for record in (self.led.seen or {}).values():
                try:
                    cat = self.tm.category(record)
                except Exception:
                    cat = None
                if isinstance(cat, str) and any((_iv_label_fits(cat, item) for item in self.led.items)):
                    meet = True
                    break
            self._meet = (size, meet)
        return self._meet[1]

    def by_label(self, category):
        cat = _iv_norm(category) if isinstance(category, str) else ''
        items = self.led.items
        labelled = [i for i, it in enumerate(items) if _iv_norm(getattr(it, 'shelf', '') or '')]
        if not cat or not labelled:
            return None
        if cat not in self.store_labels() and (not self.labels_meet()):
            return None
        exact = [i for i in labelled if _iv_norm(items[i].shelf) == cat]
        fits = exact or [i for i, it in enumerate(items) if _iv_label_fits(category, it)]
        if fits:
            return (fits, True)
        return ([i for i in range(len(items)) if i not in labelled], False)

    def owner_of(self, texts, strict=False, category=None):
        items = self.led.items
        found = self.by_label(category) if category else None
        if found is not None:
            pool, labelled = found
            if labelled and len(pool) == 1:
                return pool[0]
            if not pool:
                return -1
            flat = {_iv_words(t) for t in texts if t}
            for i in pool:
                item = items[i]
                if _iv_words(item.name) in flat or (getattr(item, 'shelf', '') and _iv_words(item.shelf) in flat):
                    return i
            bag = set().union(*(_iv_stems(t) for t in texts)) if texts else set()
            scores = [len((_iv_stems(items[i].name) | _iv_stems(getattr(items[i], 'shelf', ''))) & bag) for i in pool]
            top = max(scores)
            return pool[scores.index(top)] if top and scores.count(top) == 1 else -1
        if len(items) == 1:
            if not strict:
                return 0
            bag = set().union(*(_iv_stems(t) for t in texts)) if texts else set()
            return 0 if (_iv_stems(items[0].name) | _iv_stems(getattr(items[0], 'shelf', ''))) & bag else -2
        flat = {_iv_words(t) for t in texts if t}
        for i, item in enumerate(items):
            if _iv_words(item.name) in flat or (getattr(item, 'shelf', '') and _iv_words(item.shelf) in flat):
                return i
        bag = set().union(*(_iv_stems(t) for t in texts)) if texts else set()
        scores = [len((_iv_stems(item.name) | _iv_stems(getattr(item, 'shelf', ''))) & bag) for item in items]
        top = max(scores)
        return scores.index(top) if top and scores.count(top) == 1 else -1

    def catalogue(self):
        groups, rows = ({}, [])
        for ident, record in (self.led.seen or {}).items():
            if not isinstance(record, dict):
                continue
            facts, block = _iv_facts(record, str(ident))
            if facts:
                try:
                    category = self.tm.category(record)
                except Exception:
                    category = None
                owner = self.owner_of(_iv_strings(record, block, []), strict=True, category=category if SHELF_KEYS and isinstance(category, str) else None)
                rows.append((ident, facts, owner))
        matched = any((owner == 0 for _, _, owner in rows))
        shelves = set()
        for ident, _, owner in rows:
            if owner == -2:
                try:
                    shelves.add(str(self.tm.category(self.led.seen.get(ident)) or '').strip().lower())
                except Exception:
                    shelves.add('')
        one_shelf = len(shelves) <= 1
        labelled = len(self.led.items) == 1 and bool(str(getattr(self.led.items[0], 'shelf', '') or '').strip())
        for ident, facts, owner in rows:
            if owner == -2:
                owner = -1 if matched or not one_shelf or labelled else 0
            group = groups.setdefault(owner, {'n': 0, 'keys': {}})
            group['n'] += 1
            for key, value in facts.items():
                name = _iv_words(key)
                if name:
                    group['keys'].setdefault(name, {}).setdefault(str(ident), str(value).strip().lower())
        table = {}
        for owner, group in groups.items():
            kept = []
            for name, per in sorted(group['keys'].items(), key=lambda kv: -len(kv[1])):
                vals = [v for v in per.values() if v]
                if not vals:
                    continue
                mean = sum(map(len, vals)) / len(vals)
                if mean > 40 or 2 * sum((' / ' in v for v in vals)) >= len(vals):
                    continue
                if len(vals) >= 4 and len(set(vals)) == len(vals) and (mean > 12) and (not _iv_shared_word(vals)):
                    continue
                if any((per == other and set(name.split()) & set(prev.split()) for prev, other in kept)):
                    continue
                kept.append((name, per))
            table[owner] = [(name, len(per), list(dict.fromkeys((v for v in per.values() if v)))) for name, per in kept[:KEY_CAP]]
            for name, per in kept[:KEY_CAP]:
                vals = [v for v in per.values() if v]
                self.value_counts[owner, name] = (len(vals), sum((1 for v in vals if re.search('\\d', v))))
        return (table, {k: g['n'] for k, g in groups.items()})

    def fill_rows(self, table, sizes, item):
        thin = [r for r in table.get(item, []) if not (r[1] >= 2 or sizes.get(item, 0) <= 2)]
        rows = [r for r in table.get(item, []) if r[1] >= 2 or sizes.get(item, 0) <= 2]
        if IV_FILL != 'tuple':
            return rows + thin
        full = [(name, n, vals) + self.value_counts.get((item, name), (n, 0)) for name, n, vals in rows]
        items = self.led.items
        names = lambda x: (str(x.name or ''), str(getattr(x, 'shelf', '') or ''))
        others = [n for j, x in enumerate(items) if j != item for n in names(x)] if 0 <= item < len(items) else []
        own = names(items[item]) if 0 <= item < len(items) else ()
        claimed = [(getattr(n, 'prop', '') or '', str(getattr(n, 'value', '') or '')) for n in self.needs(item) if getattr(n, 'kind', '') == 'stated' and getattr(n, 'value', None)] if 0 <= item < len(items) else []
        return [r[:3] for r in iv_fill_order(full, self.request_text(), own, others, claimed)] + thin

    def settled(self, item, label):
        want = _iv_words(label)
        for need in self.needs(item):
            text = str(getattr(need, 'text', '') or '')
            prop = getattr(need, 'prop', '') or (text.split(':', 1)[0] if ':' in text else '')
            if want and (_iv_words(prop) == want or _iv_has(want, text.split(':', 1)[0] if ':' in text else text)):
                return True
            value = str(getattr(need, 'value', '') or '').strip().lower()
            if want and prop and _iv_twins(prop, label) and value and (getattr(need, 'kind', '') == 'stated') and any((value == str(v).strip().lower() or value in str(v).lower().split(', ') for v in self.values_for(self.fill_table[0], item, label))):
                return True
        return False

    def asked_before(self, item, label):
        want = _iv_wordset(label)
        return any((entry[0] == item and (_iv_wordset(entry[1]) == want or (_iv_twins(entry[1], label) and (len(entry) < 3 or entry[2] != 'multi'))) for entry in self.led.asked))

    @staticmethod
    def values_for(table, item, label):
        want = _iv_words(label)
        rows = table.get(item, []) + table.get(-1, [])
        exact = next((vals for name, _, vals in rows if name == want), None)
        if exact is not None:
            return exact
        return next((vals for name, _, vals in rows if _iv_twins(name, want)), [])

    def examples(self, item, label, observed):
        said = ' '.join((str(getattr(n, 'text', '')) for n in self.needs(item)))
        good = []
        for v in (str(x).strip() for x in observed):
            low = v.lower()
            if not v or len(v) > 30 or re.search('[,;/]', v) or _iv_has(low, said):
                continue
            if _iv_words(low) != _iv_words(label) and low not in (g.lower() for g in good):
                good.append(v)
        return good[:2]

    @staticmethod
    def one_property(label):
        label = _iv_words(_IV_ONE.split(str(label or ''))[0])
        return label if label and len(label.split()) <= 5 else ''

    @staticmethod
    def snap(table, item, label):
        rows = table.get(item, []) + table.get(-1, [])
        if not label or any((name == label for name, _, _ in rows)):
            return label
        near = {}
        for name, n, _ in rows:
            if _iv_twins(name, label):
                near[name] = max(n, near.get(name, 0))
        if not near:
            return label
        top = max(near.values())
        best = [name for name, n in near.items() if n == top]
        return best[0] if len(best) == 1 else label

    def distinct_name(self, item):
        items = self.led.items
        shared = set()
        for j, other in enumerate(items):
            if j != item:
                shared |= _iv_wordset(other.name)
        name = str(items[item].name or '')
        words = _iv_words(name).split()
        if not words or not shared & _iv_wordset(name) or _iv_wordset(words[-1]) & shared:
            return ''
        pieces = []
        for piece in _IV_ONE.split(name):
            kept = [w for w in _iv_words(piece).split() if not _iv_wordset(w) & shared]
            if kept:
                pieces.append(' '.join(kept))
        return ' and '.join(pieces)

    def request_text(self):
        text = getattr(self.led, 'query', '')
        return text if isinstance(text, str) and text.strip() else self.led.summary(3000)

    def plan(self):
        self.planned = True
        items = self.led.items
        self.queue = []
        if not items or not self.can_send():
            self.held = bool(items)
            return []
        self.held = False
        table, sizes = self.catalogue()
        self.fill_table = (table, sizes)
        if IV_ORDER == 'coverage':
            return self.by_coverage(table, sizes, [])
        count = iv_plan_count(self.budget() - self.reserve - self.used(), len(items))
        lines = []
        for i, item in enumerate(items):
            said = '; '.join((str(getattr(n, 'text', '')) for n in self.needs(i))) or 'nothing'
            lines.append(f'{i + 1}. {item.name} (quantity {item.qty}) - settled by the request: {said}')
        keys = []
        for owner in sorted(table, key=lambda k: (k < 0, k)):
            keys.append((f'item {owner + 1}' if owner >= 0 else 'listings not matched to one item') + f' ({sizes[owner]} listings):')
            keys += [f'  {k}. {name} (on {n}): ' + ' | '.join((v[:40] for v in vals[:4])) for k, (name, n, vals) in enumerate(self.lane_rows(table, sizes, owner), 1)]
        first = [str(a) for a in getattr(self.led, 'ask_first', None) or [] if str(a).strip()]
        user = f'REQUEST:\n{self.request_text()}\n\nITEMS:\n' + '\n'.join(lines)
        if first:
            user += '\n\nCHECK FIRST:\n' + '\n'.join(first)
        user += '\n\nKEYS:\n' + ('\n'.join(keys) or '(no listing properties seen yet)')
        user += '\n\nASKED ALREADY:\n' + ('\n'.join((f'item {e[0] + 1}: {e[1]}' for e in self.led.asked)) or 'nothing')
        system = C_PLAN
        if IV_PLAN_EACH:
            system = C_PLAN.replace('COUNT picks, best first', "COUNT picks for each item, each item's best first")
        tokens = IV_PLAN_TOKENS if IV_PLAN_EACH else 2000
        reply = self.llm.ask(system.replace('COUNT', str(count)), user, heavy=True, max_tokens=tokens)
        raw = reply.get('ask') if isinstance(reply, dict) else None
        picks, hinted = ([], set())
        for pick in raw if isinstance(raw, list) else []:
            if not isinstance(pick, dict):
                continue
            try:
                item = int(pick.get('item')) - 1
            except (TypeError, ValueError):
                continue
            if not 0 <= item < len(items):
                continue
            label = self.snap(table, item, self.one_property(pick.get('property')))
            if IV_ORDER in ('hint+coverage', 'gf') and (not str(pick.get('hint') or '').strip()):
                continue
            if label and self.open_for(item, label, picks):
                picks.append((item, label, self.examples(item, label, self.values_for(table, item, label))))
                if str(pick.get('hint') or '').strip():
                    hinted.add((item, label))
        if IV_ORDER == 'hint+coverage':
            return self.by_coverage(table, sizes, picks)
        if IV_ORDER == 'gf':
            return self.by_chance(table, sizes, picks)
        lanes, self.fills = ({}, set())
        for item in range(len(items)):
            rows = self.fill_rows(table, sizes, item)
            lane = self.guard(item, [p for p in picks if p[0] == item], rows, hinted, table, sizes, len(items) > 1)
            for name, n, vals in rows:
                if self.open_for(item, name, lane):
                    lane.append((item, name, self.examples(item, name, vals)))
                    self.fills.add((item, name))
            lane = [self.fullest(p, table) for p in lane] if IV_LONGER else lane
            listed = {_iv_words(r[0]) for r in table.get(item, [])}
            if lane and _iv_words(lane[0][1]) not in listed:
                k = next((n for n, p in enumerate(lane) if _iv_words(p[1]) in listed), None)
                if k:
                    lane.insert(0, lane.pop(k))
            lanes[item] = lane
        self.fill_table = (table, sizes)
        self.queue = self.fair([p for item in sorted(lanes, key=lambda i: self.lead_rank(i, picks)) for p in lanes[item]])
        return [p[:3] for p in self.queue]

    def fullest(self, pick, table):
        item, label = (pick[0], pick[1])
        words = _iv_wordset(label)
        longer = [r for r in table.get(item, []) if words and words < _iv_wordset(r[0]) and (not (self.settled(item, r[0]) or self.asked_before(item, r[0])))]
        if not longer or any((not _iv_twins(a[0], b[0]) for a in longer for b in longer)):
            return pick
        name, n, vals = max(longer, key=lambda r: (len(_iv_wordset(r[0])), r[1]))
        if (item, label) in self.fills:
            self.fills.add((item, name))
        return (item, name, self.examples(item, name, vals)) + tuple(pick[3:])

    @staticmethod
    def lead_rank(item, picks):
        return next((k for k, p in enumerate(picks) if p[0] == item), len(picks))

    def lane_rows(self, table, sizes, owner):
        rows = self.fill_rows(table, sizes, owner)
        return rows + [r for r in table.get(owner, []) if r not in rows]

    def guard(self, item, lane, rows, hinted, table, sizes, many):
        listings = max(1, sizes.get(item, 0))
        share = lambda label: self.row_of(table, item, label)[0] / listings
        lane = list(lane)
        head_n = IV_GUARD_KEYS
        guards = [(name, n, vals) for name, n, vals in rows if n / listings >= IV_GUARD_SHARE and (not (self.settled(item, name) or self.asked_before(item, name)))][:head_n]
        for name, n, vals in guards:
            head = lane[:head_n]
            if any((_iv_twins(p[1], name) for p in head)):
                continue
            weak = next((k for k, p in enumerate(head) if p[:2] not in hinted and share(p[1]) + IV_GUARD_GAP <= n / listings), None)
            if weak is None:
                continue
            entry = (item, name, self.examples(item, name, vals))
            placed = [p for p in lane if not _iv_twins(p[1], name)]
            if many:
                moved, placed[weak] = (placed[weak], entry)
                placed.insert(min(head_n, len(placed)), moved)
            else:
                placed.insert(weak, entry)
            if any((p[:2] in hinted and p not in placed[:head_n] for p in head)):
                continue
            lane = placed
        return lane

    def by_coverage(self, table, sizes, lead):
        said = set(_iv_words(self.request_text()).split())
        spare = []
        for item in range(len(self.led.items)):
            rows = [r for r in table.get(item, []) if r[1] >= 2 or sizes.get(item, 0) <= 2]
            rows = [r for r in rows if not set(_iv_words(r[0]).split()) & said] + [r for r in rows if set(_iv_words(r[0]).split()) & said]
            for name, n, vals in rows:
                if self.open_for(item, name, lead + spare):
                    spare.append((item, name, self.examples(item, name, vals)))
        self.queue = self.fair(lead) + self.fair(spare)
        return [p[:3] for p in self.queue]

    def by_chance(self, table, sizes, lead):
        said = {w for w in _iv_words(self.request_text()).split() if len(w) >= 3}
        items = self.led.items
        groups, rows_of = ([], {})
        for item, q in enumerate(iv_item_priors(len(items))):
            rows = [r for r in table.get(item, []) if (r[1] >= 2 or sizes.get(item, 0) <= 2) and (not self.settled(item, r[0]))]
            rows_of[item] = {r[0]: r for r in rows}
            groups.append((item, q, [(r[0], bool(set(_iv_words(r[0]).split()) & said)) for r in rows]))
        spare = []
        for _, _, item, name in iv_gf_order(groups):
            if self.open_for(item, name, lead + spare):
                spare.append((item, name, self.examples(item, name, rows_of[item][name][2])))
        self.queue = self.fair(lead) + spare
        return [p[:3] for p in self.queue]

    def open_for(self, item, label, taken):
        return not (self.settled(item, label) or self.asked_before(item, label) or any((p[0] == item and _iv_twins(p[1], label) for p in taken)))

    @staticmethod
    def fair(picks):
        lanes = {}
        for pick in picks:
            lanes.setdefault(pick[0], []).append(pick)
        out, depth = ([], 0)
        while any((len(lane) > depth for lane in lanes.values())):
            out += [lane[depth] for lane in lanes.values() if len(lane) > depth]
            depth += 1
        return out

    def budget(self):
        value = getattr(self.led, 'msg_budget', None)
        return value if isinstance(value, int) and (not isinstance(value, bool)) and (value >= 0) else 3

    def used(self):
        name = self.tm.tool('message')
        return sum((1 for h in self.env.history if h.get('name') == name)) if name else 0

    def can_send(self):
        return not self.closed and (not self.env.finished) and (not getattr(self.env, 'out_of_key', False)) and (self.env.steps_left() > 1) and (self.tm.tool('message') is not None) and (self.used() < self.budget() - self.reserve)

    def compose(self, item, label, naming=None):
        name = str(naming or self.led.items[item].name).strip()
        if name.lower().startswith('the '):
            name = name[4:]
        return f'Which {label} would you like for the {name}?'

    def compose_many(self, item, labels, naming=None):
        name = str(naming or self.led.items[item].name).strip()
        if name.lower().startswith('the '):
            name = name[4:]
        said = [str(getattr(n, 'value', '') or '').strip() for n in self.needs(item) if getattr(n, 'kind', '') == 'stated']
        said = [v for v in said if v and len(v) <= 30][:2]
        head = f'For the {name}' + (f" ({', '.join(said)})" if said else '')
        return f"{head}, what {', '.join(labels[:-1])} or {labels[-1]} would you like?"

    def told_items(self):
        return {entry[0] for entry in self.led.asked if len(entry) > 2 and entry[2] == 'reveal'}

    def next_index(self):
        if not self.queue or len(self.queue[0]) > 3:
            return 0
        told = self.told_items()
        return next((k for k, pick in enumerate(self.queue) if pick[0] not in told), 0)

    def ask_next(self):
        if not self.planned:
            self.plan()
        if not self.can_send():
            return False
        while self.queue:
            k = self.next_index()
            if self.wants_repick(self.queue[k]):
                self.repick(self.queue[k][0])
            pick = self.queue.pop(k)
            if len(pick) > 3 and pick[3] == '' and (pick[0] in self.told_items()):
                continue
            if len(pick) > 3 or not (self.asked_before(pick[0], pick[1]) or self.settled(pick[0], pick[1])):
                break
        else:
            if IV_REFILL and (not self.refilled) and self.can_send():
                self.refilled = True
                self.plan()
                return self.ask_next() if self.queue else False
            return False
        item, label = pick[:2]
        naming = pick[3] if len(pick) > 3 else None
        partners = self.partners(pick)
        keys = [label] + [p[1] for p in partners]
        question = self.compose_many(item, keys, naming) if partners else self.compose(item, label, naming)
        name = self.tm.tool('message')
        args = self.tm.args_for('message', text=question)
        if FIT_MESSAGE and partners:
            partners, question, args = self.fit(item, label, naming, partners, question, args)
        if not args or self.tm.validate(name, args):
            self.closed = True
            return False
        obs = self.env.call(name, args)
        self.led.note_obs(self.tm, name, args, obs)
        if obs is None:
            self.closed = True
            return False
        reply = self.reply_text(obs)
        cls = self.read(reply, item, label, question)['class']
        if partners:
            return self.after_bundle(pick, partners, question, reply, cls)
        self.led.asked.append((item, label, cls))
        self.transcript.append((item, question, reply, cls))
        if cls == 'refused':
            self.closed = True
            return False
        if cls == 'reveal':
            self.hear(reply, item, label)
        elif cls == 'multi' and (item, label) not in self.narrowed:
            self.narrowed.add((item, label))
            naming = self.distinct_name(item)
            near = None if naming or len(pick) > 3 else self.near_key(item, label)
            if naming:
                self.queue.insert(0, (item, label, pick[2], naming))
            elif near:
                self.queue.insert(0, near + (None,))
        return True

    def fit(self, item, label, naming, partners, question, args):
        kept = list(partners)
        while kept and args and (question not in args.values()):
            kept.pop()
            keys = [label] + [p[1] for p in kept]
            question = self.compose_many(item, keys, naming) if kept else self.compose(item, label, naming)
            args = self.tm.args_for('message', text=question)
        for p in reversed(partners[len(kept):]):
            self.queue.insert(0, p)
        return (kept, question, args)

    def open_keys(self):
        return any((len(p) > 3 or not (self.asked_before(p[0], p[1]) or self.settled(p[0], p[1])) for p in self.queue))

    def late_ask(self):
        kept, self.reserve = (self.reserve, 0)
        try:
            if not self.can_send():
                return False
            if not self.planned or self.held:
                self.plan()
            if not self.open_keys():
                return False
            return self.ask_next()
        finally:
            self.reserve = kept

    def width(self):
        return 1 if not IV_PAIR or self.one_key else IV_WIDE

    def partners(self, pick):
        want = self.width() - 1
        if want <= 0 or len(pick) > 3:
            return []
        item, chosen, out = (pick[0], [pick[1]], [])
        for p in list(self.queue):
            if len(out) >= want:
                break
            if p[0] == item and len(p) == 3 and (not any((_iv_twins(p[1], c) for c in chosen))) and (not (self.asked_before(item, p[1]) or self.settled(item, p[1]))):
                self.queue.remove(p)
                out.append(p)
                chosen.append(p[1])
        return out

    def after_bundle(self, pick, partners, question, reply, cls):
        item, keys = (pick[0], [pick] + list(partners))
        joined = ' and '.join((k[1] for k in keys))
        self.transcript.append((item, question, reply, cls))
        if cls == 'refused':
            self.led.asked.append((item, joined, cls))
            self.closed = True
            return False
        if cls == 'multi':
            self.one_key = True
            self.led.asked.append((item, joined, cls))
            for k in reversed(keys):
                self.queue.insert(0, (item, k[1], k[2], ''))
            return True
        if cls == 'reveal':
            need = self.hear(reply, item, pick[1])
            named = str(getattr(need, 'prop', '') or '')
            at = next((n for n, k in enumerate(keys) if named and (_iv_twins(named, k[1]) or self.names_property(reply, k[1]))), 0)
            self.led.asked += [(item, k[1], 'miss') for k in keys[:at]] + [(item, keys[at][1], 'reveal')]
            for k in reversed(keys[at + 1:]):
                self.queue.insert(0, k)
            self.misses[item] = 0
            return True
        self.led.asked.append((item, keys[0][1], cls))
        own = [p for p in self.queue if p[0] == item and len(p) == 3]
        if IV_REORDER == 'field':
            back = own[:1] + keys[2:] + own[1:] + keys[1:2]
        elif IV_REORDER == 'fresh':
            back = own + keys[1:]
        elif IV_REORDER:
            back = keys[IV_REORDER:] + own + keys[1:IV_REORDER]
        elif IV_SETTLE:
            self.misses[item] = self.misses.get(item, 0) + 1
            cut = 1 + IV_SETTLE_KEEP
            back = own + keys[1:] if self.misses[item] % 2 == 0 else keys[1:cut] + own + keys[cut:]
        else:
            back = keys[1:] + own
        self.queue = self.fair([p for p in self.queue if p not in own] + back)
        return True

    def near_key(self, item, label):
        if item in self.told_items():
            return None
        table, sizes = self.fill_table if self.fill_table[0] else self.catalogue()
        words = {w for w in _iv_wordset(label) if len(w) >= 3}
        rows = [r for r in self.fill_rows(table, sizes, item) if _iv_words(r[0]) != _iv_words(label) and (not (self.settled(item, r[0]) or self.asked_before(item, r[0])))]
        twin = sorted((r for r in rows if _iv_twins(r[0], label)), key=lambda r: not _iv_wordset(label) < _iv_wordset(r[0]))
        share = [r for r in rows if words & _iv_wordset(r[0])]
        hit = (twin or share or [None])[0]
        return (item, hit[0], self.examples(item, hit[0], hit[2])) if hit else None

    def wants_repick(self, pick):
        item = pick[0]
        return IV_REPICK and IV_ORDER == 'model' and (len(pick) == 3) and (tuple(pick[:2]) in self.fills) and (item not in self.repicked) and (item not in self.told_items()) and (not (self.asked_before(item, pick[1]) or self.settled(item, pick[1])))

    def repick_slots(self, item):
        return [k for k, p in enumerate(self.queue) if p[0] == item and len(p) == 3 and (tuple(p[:2]) in self.fills) and (not (self.asked_before(item, p[1]) or self.settled(item, p[1])))]

    def repick(self, item):
        self.repicked.add(item)
        slots = self.repick_slots(item)
        if len(slots) < 2:
            return
        entries = [self.queue[k] for k in slots]
        cands = entries[:IV_REPICK_K]
        table, sizes = self.fill_table if self.fill_table[0] else self.catalogue()
        listings = max(1, sizes.get(item, 0))
        messages = max(1, self.budget() - self.reserve - self.used())
        left = max(1, sum((1 for p in self.queue[:messages] if p[0] == item)))
        items = self.led.items
        others = [str(x.name) for j, x in enumerate(items) if j != item]
        asked = [str(e[1]) for e in self.led.asked if e[0] == item and (len(e) < 3 or e[2] == 'miss')]
        lines = [f'Request:\n{self.request_text()}', '', f'Item: {items[item].name}', f"Other items in the order: {(', '.join(others) if others else 'none')}", f"Already asked about this item (no hidden requirement found): {', '.join(asked) or 'none'}", f'Questions left for this item: {left}', 'Candidates:']
        for i, (_, name, _) in enumerate(cands, 1):
            n, vals = self.row_of(table, item, name)
            seen = '; '.join((str(v)[:60] for v in vals[:IV_REPICK_VALS])) or 'none'
            lines.append(f'{i}. {name} (on {round(100 * n / listings)}% of its listings; seen values: {seen})')
        got = self.llm.ask(C_REPICK, '\n'.join(lines), heavy=False, max_tokens=1200)
        raw = got.get('order') if isinstance(got, dict) else None
        order = []
        for x in raw if isinstance(raw, list) else []:
            try:
                k = int(x) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= k < len(cands) and k not in order:
                order.append(k)
        if not order:
            return
        order += [k for k in range(len(cands)) if k not in order]
        for k, entry in zip(slots, [cands[k] for k in order] + entries[len(cands):]):
            self.queue[k] = entry

    @staticmethod
    def row_of(table, item, label):
        rows = table.get(item, []) + table.get(-1, [])
        hit = next((r for r in rows if r[0] == label), None) or next((r for r in rows if _iv_twins(r[0], label)), None)
        return (hit[1], hit[2]) if hit else (0, [])

    def reply_text(self, obs):
        try:
            body, _ = unwrap(obs)
        except Exception:
            body = obs
        if isinstance(body, str):
            return body.strip()
        parts = _iv_strings(body, None, [], longer_than=1)
        return ' '.join(parts)[:800] if parts else str(body)[:400]

    @staticmethod
    def names_property(reply, label):
        want = _iv_stems(label)
        return bool(want) and want <= _iv_stems(reply)

    def read(self, reply, item, label, question):
        known = self.readings.get(reply)
        if known and known['class'] != 'reveal':
            return known
        user = f'Item: {self.led.items[item].name}\nQuestion: {question}\nReply: {reply}'
        got = self.llm.ask(C_READ, user, heavy=False, max_tokens=300) if reply else None
        got = got if isinstance(got, dict) else {}
        cls = str(got.get('class') or '').strip().lower()
        cls = cls if cls in ('reveal', 'miss', 'multi', 'refused') else ''
        if cls in ('', 'miss') and self.names_property(reply, label):
            cls = 'reveal'
        reading = dict(got)
        reading['class'] = cls or 'miss'
        if got:
            self.readings[reply] = reading
        return reading

    def hear(self, reply, item, prop):
        reading = self.readings.get(reply)
        if not reading or reading.get('class') != 'reveal':
            reading = self.read(reply, item, prop, prop)
        if reading.get('class') != 'reveal':
            return None
        named = self.one_property(reading.get('property')) or prop
        text = str(reading.get('need') or '').strip()
        if ':' not in text:
            text = f'{named}: {text or reply}'
        value = text.split(':', 1)[1].strip() or None
        points = reading.get('points') is True
        need = Need(item=item, text=text, kind='ref' if points else 'hidden', value=None if points else value, prop=named)
        if any((getattr(n, 'text', None) == text for n in self.needs(item))):
            return need
        add = getattr(self.led, 'reveal', None)
        if callable(add):
            add(need)
        else:
            self.led.revealed.append(need)
        if points:
            relation = str(reading.get('relation') or '').strip().lower()
            start = str(reading.get('start') or '').strip() or value or reply
            ref = Ref(item=item, prop=named, relation=relation if relation in REF_RELATIONS else 'same', start=start, chain=[], value=None, done=False)
            add = getattr(self.led, 'add_ref', None)
            if callable(add):
                add(ref)
            else:
                self.led.refs.append(ref)
        return need
CHAIN_FIRST = True
PART_TRACE = True
CHAIN_PAIRS = True
PAIR_CLAUSE = 'For "value": when the page states a rule that pairs values of PROPERTY with one another (one value goes with, goes together with or is paired with another), "value" is the value that rule pairs with this page\'s own value of PROPERTY, as the rule writes it, not the page\'s own value.'
D_LINK = 'You help a shopping agent find one particular store listing from a description of it.\nThe description gives the listing\'s kind of product and usually its seller or brand and a few of its properties.\nIt may still carry words about how a value compares with that listing\'s or a clause allowing a further listing:\nignore those, they do not describe the listing. The listing is often of a different kind from what the shopper buys.\nSEEN lists listings the agent has already come across. Name the ones that could be the described listing,\nbest first: a listing qualifies only when none of its fields contradicts the description (a value written in another\nunit, spelling or rounding is no contradiction; a different seller or kind of product is, but a seller or brand\nfield naming the described shop is enough even when the other one names someone else), and among those, the\nones whose fields already show more of the description\'s details come first.\nWhen none of them is clearly that listing, also write short texts to type into the store\'s search to find it,\nbest first: the seller or brand name with the one-word kind of product as titles write it (a store narrows such a\nsearch to that seller\'s few listings, which can then be read to the end), then that name with one value the\ndescription gives in a word the titles would carry, then the name alone; a value in another unit than the store\'s\nown often matches nothing. Keep each text to a few words; long texts often match nothing. When a\nline says some listings were opened and are not it, write different texts than before, for instance other values\nof the description with the seller.\nReply with one JSON object only: {"ids": [at most 3 listing ids], "queries": [at most 3 search texts]}'
D_READ = 'A shopping agent follows a chain of store pages to learn one property value for a product it will buy.\nPROPERTY is that property. DESCRIPTION picks out the listing the chain has reached. PAGES are full records of the\ncandidate listings just opened.\n1. "id": the page that is the described listing: same kind and seller or brand, and every detail of the description\n   the page shows agrees with it (another unit, spelling or rounding still agrees). The shop the description names\n   is there when either the seller or the brand field carries it; the other field naming someone else is no\n   contradiction. When several agree, the one that\n   matches the most details. "" when none is that listing.\n   "missing": the details of the description (each value it gives for some property) that this page does not show\n   anywhere, copied from the description; [] when the page shows every one. A page silent on a detail may still be\n   another listing from the same seller, so list every such detail.\n2. "next": on that page, any field (remarks, fine print, notes, annotations or similar) may tell the reader to take\n   this same PROPERTY from a different listing instead of this one; it may name the property in other words or\n   spelling. Copy that note\'s whole description of the other listing (its kind, seller or brand, and the properties\n   it lists). A note about a different property, a remark that some other version or variant exists elsewhere, and\n   sales talk do not send the reader anywhere for PROPERTY: then "next" is "".\n   "see": a text on that page that points at another listing by its code, part or model number or name (for more on\n   this product, a part of it or a version of it) is not a "next", yet it is no sales talk either: copy each code or\n   name such a text gives, exactly as written, at most 3; [] when there is none.\n3. "value": the PROPERTY\'s value as this page itself states it (attributes first, then title or text), copied as\n   written, or "" when the page does not state it. When "next" is set, this page\'s own value is not the answer.\nReply with one JSON object only: {"id": "...", "missing": ["..."], "next": "...", "value": "...", "see": ["..."]}'
NOTE_ROOM = True
NOTE_TOKENS = 3500
D_NOTE = 'A shopping agent has opened some store listings. REQUIREMENTS lists the shopper\'s items and what each must have.\nPAGES gives, for each listing, the longer texts its page shows. Any of them (remarks, fine print, notes, annotations or\nsimilar) may tell the reader to take one property\'s value from a different listing instead of this one; it may name the\nproperty in other words or spelling. Report each such text whose property is one the REQUIREMENTS name for an item:\nthe listing id, the item number, the property as the REQUIREMENTS name it, and the text\'s whole description of the\nother listing (its kind, seller or brand, and the properties it lists). A text about a property no requirement names,\na remark that some other version or variant exists elsewhere, and sales talk send the reader nowhere: leave them out\nof "sends".\nSeparately, under "looks", report each text that points the reader at another listing by its code, part or model\nnumber or name (for more on this product, a part of it or a version of it), whatever its wording and whatever kind\nthat other listing is: the listing id and the code or name exactly as the text gives it. Such a text is no sales talk.\nReply with one JSON object only: {"sends": [{"id": "...", "item": 1, "property": "...", "listing": "..."}], "looks": [{"id": "...", "code": "..."}]}'
D_READ_PLAIN = 'A shopping agent follows a chain of store pages to learn one property value for a product it will buy.\nPROPERTY is that property. DESCRIPTION picks out the listing the chain has reached. PAGES are full records of the\ncandidate listings just opened.\n1. "id": the page that is the described listing: same kind and seller or brand, and every detail of the description\n   the page shows agrees with it (another unit, spelling or rounding still agrees). The shop the description names\n   is there when either the seller or the brand field carries it; the other field naming someone else is no\n   contradiction. When several agree, the one that\n   matches the most details. "" when none is that listing.\n   "missing": the details of the description (each value it gives for some property) that this page does not show\n   anywhere, copied from the description; [] when the page shows every one. A page silent on a detail may still be\n   another listing from the same seller, so list every such detail.\n2. "next": on that page, any field (remarks, fine print, notes, annotations or similar) may tell the reader to take\n   this same PROPERTY from a different listing instead of this one; it may name the property in other words or\n   spelling. Copy that note\'s whole description of the other listing (its kind, seller or brand, and the properties\n   it lists). A note about a different property, a remark that some other version or variant exists elsewhere, and\n   sales talk do not send the reader anywhere: then "next" is "".\n3. "value": the PROPERTY\'s value as this page itself states it (attributes first, then title or text), copied as\n   written, or "" when the page does not state it. When "next" is set, this page\'s own value is not the answer.\nReply with one JSON object only: {"id": "...", "missing": ["..."], "next": "...", "value": "..."}'
D_NOTE_PLAIN = 'A shopping agent has opened some store listings. REQUIREMENTS lists the shopper\'s items and what each must have.\nPAGES gives, for each listing, the longer texts its page shows. Any of them (remarks, fine print, notes, annotations or\nsimilar) may tell the reader to take one property\'s value from a different listing instead of this one; it may name the\nproperty in other words or spelling. Report each such text whose property is one the REQUIREMENTS name for an item:\nthe listing id, the item number, the property as the REQUIREMENTS name it, and the text\'s whole description of the\nother listing (its kind, seller or brand, and the properties it lists). A text about a property no requirement names,\na remark that some other version or variant exists elsewhere, and sales talk send the reader nowhere: leave them out.\nReply with one JSON object only: {"sends": [{"id": "...", "item": 1, "property": "...", "listing": "..."}]}'

def _d_long_lines(obj, depth=0, words=5):
    if depth > 4:
        return []
    if isinstance(obj, str):
        return [obj] if len(obj.split()) >= words else []
    vals = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (list, tuple)) else []
    return [line for v in vals for line in _d_long_lines(v, depth + 1, words)]

def _d_flat(obj, out, cut=300):
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            _d_flat(v, out, cut)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _d_flat(v, out, cut)
    elif obj is not None:
        out.append(str(obj)[:cut])
    return out

def _d_tokens(text):
    return set(re.findall('[a-z0-9]+', str(text).lower()))

def _d_brief(rec, cut=160):

    def clip(v):
        if isinstance(v, dict):
            return {k: clip(x) for k, x in v.items() if x not in (None, '', [], {})}
        if isinstance(v, list):
            return [clip(x) for x in v[:12]]
        if isinstance(v, str) and len(v) > cut:
            return v[:cut] + '...'
        return v
    try:
        return json.dumps(clip(rec), ensure_ascii=False)
    except (TypeError, ValueError):
        return str(rec)[:cut * 8]

def _d_rank(desc, records, skip=(), limit=8):
    want = _d_tokens(desc)
    toks = {i: _d_tokens(' '.join(_d_flat(r, []))) for i, r in records.items() if i not in skip}
    if not want or not toks:
        return []
    df = {}
    for ts in toks.values():
        for t in ts & want:
            df[t] = df.get(t, 0) + 1
    n = len(toks) + 1
    scored = [(sum((math.log(n / df[t]) for t in ts & want)), i) for i, ts in toks.items()]
    scored = [(s, i) for s, i in scored if s > 0]
    scored.sort(key=lambda p: -p[0])
    return [i for _, i in scored[:limit]]

def _d_words(text):
    return ' ' + ' '.join(re.findall('[a-z0-9]+', str(text).lower())) + ' '

def _d_named_in(desc, rec):
    low = _d_words(desc)
    return {_d_words(v) for v in (rec or {}).values() if isinstance(v, str) and 3 <= len(v.strip()) <= 40 and _d_words(v).strip() and (_d_words(v) in low)}

def _d_named(desc, records, skip=()):
    low = _d_words(desc)
    carriers = {}
    for ident, rec in records.items():
        if not isinstance(rec, dict):
            continue
        for v in rec.values():
            if isinstance(v, str) and 3 <= len(v.strip()) <= 40 and re.search('[a-z]', v.lower()) and (v != ident):
                key = _d_words(v)
                if key.strip() and key in low:
                    carriers.setdefault(key, set()).add(ident)
    if not carriers:
        return []
    fewest = min((len(c) for c in carriers.values()))
    if fewest * 2 > len(records) and len(records) > 2:
        return []
    keep = set().union(*(c for c in carriers.values() if len(c) == fewest))
    return [i for i in records if i in keep and i not in skip]

def _d_stems(text):
    return {w[:-1] if len(w) > 3 and w.endswith('s') else w for w in re.findall('[a-z0-9]{3,}', str(text).lower())}

def _d_label(desc, labels):
    want = _d_stems(desc)
    best, score, tie = (None, (0, 0), False)
    for label in labels:
        stems = _d_stems(label)
        hit = len(stems & want)
        if not stems or not hit:
            continue
        mine = (hit / len(stems), hit)
        if mine > score:
            best, score, tie = (label, mine, False)
        elif mine == score and _d_words(label) != _d_words(best):
            tie = True
    return None if tie else best
_D_CODE = re.compile('[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*')

def _d_codeish(token):
    core = token.replace('-', '')
    digits = sum((c.isdigit() for c in core))
    if len(core) < 4 or digits < 4:
        return False
    return digits >= 5 if core.isdigit() else core[0].isalpha()

def _d_shape(token):
    return re.sub('\\d', '9', re.sub('[a-z]', 'a', token.lower()))

def _d_prose(text):
    t = text.strip()
    return len(t.split()) >= 3 and bool(re.search('[A-Za-z0-9)][.!?](\\s|$)', t))

def _d_fields(obj, path=(), out=None):
    out = [] if out is None else out
    if len(path) > 4:
        return out
    if isinstance(obj, str):
        out.append((path, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _d_fields(v, path + (str(k),), out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _d_fields(v, path + ('*',), out)
    return out

def _d_text_paths(records):
    votes = {}
    for rec in records:
        for path, s in _d_fields(rec):
            if not s.strip():
                continue
            n, k = votes.get(path, (0, 0))
            votes[path] = (n + _d_prose(s), k + 1)
    return {p for p, (n, k) in votes.items() if 2 * n > k}

def _d_name_codes(record, text_paths):
    found = [(p, s) for p, s in _d_fields(record) if p not in text_paths]
    top = min((len(p) for p, _ in found), default=1)
    return [t for p, s in found if len(p) == top for t in _D_CODE.findall(s) if _d_codeish(t)]

def part_pointers(record, seen):
    if not isinstance(record, dict):
        return []
    seen = seen if isinstance(seen, dict) else {}
    others = {str(k): r for k, r in seen.items() if isinstance(r, dict) and r is not record}
    own_ids = {k.lower() for k in seen if seen[k] is record}
    own_ids |= {s.lower() for p, s in _d_fields(record) if len(p) == 1 and s in seen}
    others = {k: r for k, r in others.items() if k.lower() not in own_ids}
    text_paths = _d_text_paths([record] + list(others.values()))
    mine = _d_name_codes(record, text_paths)
    own = {t.lower() for t in mine} | own_ids
    own_forms = {_d_shape(t) for t in mine}
    carriers = {}
    for ident, rec in others.items():
        for t in {t.lower() for t in _d_name_codes(rec, text_paths)} | {ident.lower()}:
            carriers.setdefault(t, set()).add(ident)
    first, then, done = ([], [], set())
    for path, text in _d_fields(record):
        if path not in text_paths:
            continue
        for token in _D_CODE.findall(text):
            low = token.lower()
            if low in own or low in done or (not _d_codeish(token)):
                continue
            named = len(carriers.get(low, ())) == 1 and (not token.replace('-', '').isdigit())
            if named or _d_shape(token) in own_forms:
                done.add(low)
                (first if _d_shape(token) in own_forms else then).append(token)
    return (first + then)[:2]

def _d_single_id_args(tool, ident):
    params = ((tool or {}).get('function') or {}).get('parameters') or {}
    props = params.get('properties') or {}
    names = params.get('required') or list(props)
    if len(names) == 1 and (props.get(names[0]) or {}).get('type', 'string') == 'string':
        return {names[0]: ident}
    return None

def _d_call(env, tm, led, name, args):
    if not name or not isinstance(args, dict) or tm.validate(name, args):
        return None
    if env.finished or env.out_of_key:
        return None
    obs = env.call(name, args)
    if obs is not None:
        tm.learn(name, args, obs)
        led.note_obs(tm, name, args, obs)
    return obs

def _d_rows(tm, obs):
    try:
        body = unwrap(obs)[0]
        rows = tm.rows(body)
    except Exception:
        return []
    return [r for r in rows or [] if isinstance(r, dict)]

def _d_mark_opened(tm, led, ident, obs):
    if isinstance(obs, dict) and tm.row_id(obs) == ident:
        led.opened.add(ident)

def _d_str(v):
    return v.strip() if isinstance(v, str) else ''

class ChainTracker:
    MAX_LINKS = 8
    ROUNDS = 4
    ROUNDS_LONG = 6
    LONG = 15
    PER_LINK = 3
    LOOK_ON = 5
    SWEEP = 12
    NOTE_PAGES = 8

    def __init__(self, ledger, llm, toolmap, env):
        self.led, self.llm, self.tm, self.env = (ledger, llm, toolmap, env)
        self.reserve = 2
        self.sent = set()
        self.attempts = {}
        self.heard = set()
        self.noted = {}
        self.looked = {}
        self.fresh = []

    def pending(self, lines=None):
        ids = [str(l['id']) for l in lines or [] if isinstance(l, dict) and l.get('id')]
        ids += [i for i, q in (getattr(self.led, 'cart', None) or {}).items() if q and i not in ids]
        if ids:
            self.heed(ids)
        self._fold()
        return [r for r in self.led.refs if not r.done]

    def report(self, cut=1500):
        lines = [f'{i}: {_d_brief(self.led.seen.get(i), 300)}' for i in dict.fromkeys(self.fresh) if i in self.led.seen]
        self.fresh = []
        text = ''
        for line in lines:
            if len(text) + len(line) > cut:
                break
            text += line + '\n'
        return text

    def heed(self, ids):
        if not self.env.finished and (not self.env.out_of_key):
            self._read_notes()
        added = []
        for owner in ids:
            for item, prop, desc in self.noted.get(owner, [])[:2]:
                if any((r.owner == owner and r.prop.lower() == prop.lower() for r in self.led.refs)):
                    continue
                ref = Ref(item=item, prop=prop, relation='same', start=desc, owner=owner)
                self.led.refs.append(ref)
                added.append(ref)
        return added

    def noted_text(self):
        lines = [f"listing {i}: its page takes item {item + 1}'s {prop} from {desc}" for i, sends in self.noted.items() for item, prop, desc in sends]
        lines += [f"listing {i}: its page points at another listing for more on it: {', '.join(codes)}" for i, codes in self.looked.items() if codes]
        return '\n'.join(lines)
    part_pointers = staticmethod(part_pointers)

    def look(self, ident, codes):
        rec = self.led.seen.get(ident) if isinstance(getattr(self.led, 'seen', None), dict) else None
        if not PART_TRACE or not isinstance(rec, dict) or (not isinstance(codes, list)):
            return
        text = _d_words(' '.join((s for _, s in _d_fields(rec))))
        mine = self.looked.setdefault(ident, [])
        for code in codes[:3]:
            code = _d_str(code)
            if code and _d_words(code).strip() and (_d_words(code) in text) and (code.lower() not in (c.lower() for c in mine)):
                mine.append(code)

    def pointers(self, ident, limit=2):
        rec = (getattr(self.led, 'seen', None) or {}).get(ident)
        out = part_pointers(rec, self.led.seen) if isinstance(rec, dict) else []
        for code in self.looked.get(ident, []):
            if code.lower() not in (c.lower() for c in out):
                out.append(code)
        return out[:limit]

    def leading(self):
        if not CHAIN_FIRST:
            return []
        return [r for r in self.pending() if not getattr(r, 'owner', '') and (not self.stuck(r))]

    def stuck(self, ref):
        if ref.done or not ref.chain:
            return False
        link = ref.chain[-1]
        return link.get('rounds', 0) >= self._rounds() and (not self._unopened(link)) and (not link.get('cands')) and (not link.get('partial'))

    def missing(self):
        return [r.chain[-1]['desc'] if r.chain else r.start for r in self.pending()]

    def chain_ids(self, ref):
        return [l['id'] for l in ref.chain if l.get('id')]

    def unopened_ids(self):
        return [i for r in self.led.refs for i in self.chain_ids(r) if i not in self.led.opened and self.attempts.get(i, 0) < 2]

    def values_text(self):
        lines = []
        for r in self.led.refs:
            if r.done and getattr(r, 'owner', ''):
                lines.append(f"item {r.item + 1}: listing {r.owner} has {r.prop} '{r.value or '(not stated)'}' (its page sends the reader on; read along pages {', '.join(self.chain_ids(r))})")
            elif r.done:
                said = self.need_words(r)
                if said:
                    lines.append(f"item {r.item + 1}: {said} (the page it points at gives: '{r.value or '(not stated)'}'; read along pages {', '.join(self.chain_ids(r))})")
                else:
                    lines.append(f"item {r.item + 1}: {r.prop} must be '{r.relation}' relative to '{r.value or '(not stated)'}' (read along pages {', '.join(self.chain_ids(r))})")
        return '\n'.join(lines)

    def need_words(self, ref):
        needs_of = getattr(self.led, 'needs_of', None)
        try:
            needs = needs_of(ref.item) if callable(needs_of) else []
        except Exception:
            return ''
        want = set(re.findall('[a-z0-9]+', str(ref.prop or '').lower()))
        for need in needs or []:
            if getattr(need, 'kind', '') != 'ref':
                continue
            prop = str(getattr(need, 'prop', '') or str(getattr(need, 'text', '')).split(':', 1)[0])
            if want and want <= set(re.findall('[a-z0-9]+', prop.lower())):
                return str(getattr(need, 'text', '') or '')
        return ''

    def advance(self, max_calls, model_calls=None, halt=None):
        used, model_room = (0, 3 + 3 * max(0, max_calls))
        if model_calls is not None:
            model_room = min(model_room, max(0, model_calls))
        stop = (lambda: bool(halt())) if callable(halt) else lambda: False
        for i in self.unopened_ids():
            if used >= self._room(max_calls) or stop():
                break
            used += self._open(i, self._room(max_calls) - used)
        moved = True
        while moved and (not self.env.finished) and (not self.env.out_of_key):
            moved = False
            for ref in self.pending():
                room = self._room(max_calls) - used
                if model_room <= 0 or stop():
                    return used
                n, asked, did = self._step(ref, room)
                used += n
                model_room -= asked
                moved = moved or did
        return used

    def _fold(self):
        kept = []
        for ref in list(self.led.refs):
            fold_ref(kept, ref)
        if len(kept) < len(self.led.refs):
            self.led.refs[:] = kept

    def _read_notes(self):
        items = getattr(self.led, 'items', None) or []
        if not items:
            return
        pages = []
        for ident in [i for i in self.led.seen if i in self.led.opened]:
            fresh = [t for t in dict.fromkeys(_d_long_lines(self.led.seen.get(ident))) if t not in self.heard]
            if fresh and len(pages) < self.NOTE_PAGES:
                self.heard.update(fresh)
                pages.append(f'{ident}: ' + json.dumps([t[:400] for t in fresh[:6]], ensure_ascii=False))
        if not pages:
            return
        wants = '\n'.join((f'item {n}: {it.name}; ' + '; '.join((x.text for x in getattr(it, 'needs', []) or [])) for n, it in enumerate(items, 1)))
        wants += ''.join((f'\nitem {r.item + 1}: {r.prop} (taken from another listing)' for r in self.led.refs if not r.owner))
        got = self.llm.ask(D_NOTE if PART_TRACE else D_NOTE_PLAIN, f'REQUIREMENTS:\n{wants}\n\nPAGES:\n' + '\n'.join(pages), max_tokens=NOTE_TOKENS if NOTE_ROOM else 2000)
        sends = got.get('sends') if isinstance(got, dict) else None
        for s in sends if isinstance(sends, list) else []:
            if not isinstance(s, dict):
                continue
            ident, prop, desc = (_d_str(str(s.get('id') or '')), _d_str(s.get('property')), _d_str(s.get('listing')))
            item = s.get('item') if isinstance(s.get('item'), int) and (not isinstance(s.get('item'), bool)) else 0
            if ident in self.led.opened and prop and desc and (1 <= item <= len(items)):
                mine = self.noted.setdefault(ident, [])
                if all((p.lower() != prop.lower() for _, p, _ in mine)):
                    mine.append((item - 1, prop, desc))
        looks = got.get('looks') if isinstance(got, dict) else None
        for s in looks if isinstance(looks, list) else []:
            ident = _d_str(str(s.get('id') or '')) if isinstance(s, dict) else ''
            if ident in self.led.opened:
                self.look(ident, [s.get('code')])

    def _rounds(self):
        return self.ROUNDS_LONG if self.env.steps_left() > self.LONG else self.ROUNDS

    def _room(self, max_calls):
        return max(0, min(max_calls, self.env.steps_left() - self.reserve))

    def _label(self, desc):
        tool = self.tm.tool('search')
        slot = ((getattr(self.tm, 'slots', None) or {}).get(tool) or {}).get('category') if tool else None
        if not slot:
            return None
        args = getattr(self.tm, '_args', None)
        try:
            spec = (args(tool) if callable(args) else {}).get(slot) or {}
        except Exception:
            spec = {}
        labels = [v for v in spec.get('enum') or [] if isinstance(v, str)] if isinstance(spec, dict) else []
        if not labels:
            cat = getattr(self.tm, 'category', None)
            labels = list(dict.fromkeys((c for c in (cat(r) if callable(cat) else None for r in self.led.seen.values() if isinstance(r, dict)) if isinstance(c, str) and c.strip())))
        return _d_label(desc, labels)

    def _of_kind(self, desc, ident, label):
        cat = getattr(self.tm, 'category', None)
        try:
            got = cat(self.led.seen.get(ident)) if callable(cat) else None
        except Exception:
            got = None
        if not isinstance(got, str) or not got.strip():
            return True
        if label:
            return _d_stems(got) == _d_stems(label)
        return bool(_d_stems(got) & _d_stems(desc))

    def _visible(self, ref, link):
        closed = getattr(self.llm, 'closed', None)
        if callable(closed) and closed():
            return []
        skip = set(link['tried']) | set(self.chain_ids(ref))
        named = _d_named(link['desc'], self.led.seen, skip)
        if not named:
            return []
        label = None if link.get('nocat') else self._label(link['desc'])
        named = [i for i in named if self._of_kind(link['desc'], i, label) and self.attempts.get(i, 0) < 2]
        found = link.get('found') or []
        named.sort(key=lambda i: (i not in self.led.opened, found.index(i) if i in found else len(found)))
        return named[:max(0, self.SWEEP - link.get('opens', 0))]

    def _new_link(self, desc):
        return {'desc': desc, 'id': None, 'opened': False, 'pointer': None, 'value': '', 'cands': [], 'tried': [], 'queries': [], 'rounds': 0, 'partial': None, 'looked': 0}

    def _unopened(self, link):
        if link.get('opens', 0) >= self.SWEEP:
            return []
        return [i for i in link.get('cands', []) if i not in self.led.opened and self.attempts.get(i, 0) < 2]

    def _detail_tools(self):
        first = self.tm.tool('detail')
        others = [n for n, r in self.tm.role.items() if r == 'detail' and n != first]
        return ([first] if first else []) + others

    def _views(self, ident):
        return {h.get('name') for h in getattr(self.env, 'history', None) or [] if ident in json.dumps(h.get('args'), ensure_ascii=False)}

    def _open(self, ident, room, every=False, skip=()):
        used = 0
        self.attempts[ident] = self.attempts.get(ident, 0) + 1
        tools = {t.get('function', {}).get('name'): t for t in getattr(self.tm, 'tools', []) or []}
        names = self._detail_tools()
        for k, name in enumerate(names if every else names[:1]):
            if used >= room:
                break
            if name in skip:
                continue
            args = self.tm.args_for('detail', product_id=ident) if k == 0 else _d_single_id_args(tools.get(name), ident)
            if not args:
                continue
            used += 1
            _d_mark_opened(self.tm, self.led, ident, _d_call(self.env, self.tm, self.led, name, args))
        if used and ident in self.led.opened:
            self.fresh.append(ident)
        return used

    def _step(self, ref, room):
        if not ref.chain:
            ref.chain.append(self._new_link(ref.start))
        link = ref.chain[-1]
        if link.get('cands') is None:
            link.update({k: v for k, v in self._new_link(link.get('desc', '')).items() if k not in link or link[k] is None})
            if link.get('id') and link['pointer'] is None:
                link['cands'] = [link['id']]
        link.setdefault('partial', None)
        link.setdefault('looked', 0)
        if link.get('held'):
            if room <= 0:
                return (0, 0, False)
            n, asked = self._accept(ref, link, link.pop('held'), room)
            return (n, asked, True)
        closed = self._unopened(link)
        if closed:
            if room <= 0:
                return (0, 0, False)
            ident = closed[0]
            main = (self._detail_tools() or [None])[0]
            every = getattr(self.tm, 'shows_props', {}).get(main) is not True
            if every and room < len(self._detail_tools()):
                return (0, 0, False)
            used = self._open(ident, room, every=every)
            link['opens'] = link.get('opens', 0) + 1
            if link['partial']:
                link['looked'] += 1
            if ident not in self.led.opened:
                return (used, 0, used > 0)
            n, asked, did = self._read(ref, link, [ident], room - used)
            return (used + n, asked, did)
        if link['cands']:
            return self._read(ref, link, list(link['cands']), room)
        if link['rounds'] >= self._rounds() or link['looked'] >= self.LOOK_ON or link.get('opens', 0) >= self.SWEEP:
            return self._settle(ref, link, room)
        near = self._visible(ref, link)
        if near:
            link['cands'] = near
            return (0, 0, True)
        link['rounds'] += 1
        got = self._find(ref, link)
        if got is None:
            link['rounds'] = self.ROUNDS_LONG
            return (0, 1, bool(link['partial']))
        if link['cands']:
            return (0, 1, True)
        while link['queries']:
            q = link['queries'].pop(0)
            label = None if link.get('nocat') else self._label(link['desc'])
            args = self.tm.args_for('search', query=q, category=label) if label else None
            if not args:
                label, args = (None, self.tm.args_for('search', query=q))
            key = q.lower() + (f' (in {label})' if label else '')
            if key in self.sent:
                continue
            if room <= 0:
                link['queries'].insert(0, q)
                link['rounds'] -= 1
                return (0, 1, False)
            self.sent.add(key)
            if args:
                obs = _d_call(self.env, self.tm, self.led, self.tm.tool('search'), args)
                found = [self.tm.row_id(r) for r in _d_rows(self.tm, obs)]
                done = set(link['tried']) | set(self.chain_ids(ref))
                if label and (not [i for i in found if i and i not in done]):
                    link['nocat'] = True
                link['found'] = [i for i in found if i] + [i for i in link.get('found', []) if i not in found]
                return (1, 1, True)
        return (0, 1, link['rounds'] < self._rounds() or bool(link['partial']))

    def _find(self, ref, link):
        skip = set(link['tried']) | set(self.chain_ids(ref))
        ranked = _d_rank(link['desc'], self.led.seen, skip)
        named = _d_named(link['desc'], self.led.seen, skip)
        found = link.get('found') or []
        named.sort(key=lambda i: (i not in self.led.opened, found.index(i) if i in found else len(found), ranked.index(i) if i in ranked else len(ranked)))
        ranked += [i for i in named if i not in ranked][:self.SWEEP]
        ranked = [i for i in found if i in named and i in ranked] + [i for i in ranked if i not in found or i not in named]
        seen = '\n'.join((f'{i}: {_d_brief(self.led.seen[i])}' for i in ranked)) or '(none)'
        user = f"DESCRIPTION: {link['desc']}\n\nSEEN:\n{seen}"
        if link['tried']:
            user += '\n\nOpened already and not it, or silent on some of its details: ' + ', '.join(link['tried'][-6:]) + '\nSearched already: ' + '; '.join(sorted(self.sent))[:400]
        got = self.llm.ask(D_LINK, user, max_tokens=2000)
        if not isinstance(got, dict):
            return None
        ids = [str(i) for i in got.get('ids') or [] if str(i) in ranked][:self.PER_LINK]
        if ids:
            like = _d_named_in(link['desc'], self.led.seen.get(ids[0]))
            ids += [i for i in named if i not in ids and i not in self.led.opened and (like <= _d_named_in(link['desc'], self.led.seen.get(i)))]
        link['cands'] = ids[:self.SWEEP]
        qs = [_d_str(q) for q in got.get('queries') or [] if _d_str(q)]
        if not ids and (not qs) and link['desc']:
            qs = [link['desc'][:80]]
        link['queries'] = [q for q in qs[:3] if q.lower() not in self.sent] + link['queries']
        return got

    def _ask_read(self, ref, link, ids):
        pages = '\n'.join((f'{i}: {_d_brief(self.led.seen.get(i), 400)}' for i in ids))
        rule = f'\nRULE: {PAIR_CLAUSE}' if CHAIN_PAIRS else ''
        got = self.llm.ask(D_READ if PART_TRACE else D_READ_PLAIN, f"PROPERTY: {ref.prop}\nDESCRIPTION: {link['desc']}{rule}\n\nPAGES:\n{pages}", max_tokens=2000)
        return got if isinstance(got, dict) else None

    def _read(self, ref, link, ids, room):
        got = self._ask_read(ref, link, ids)
        if got is None:
            link['rounds'] += 1
            return (0, 1, False)
        ident = _d_str(got.get('id'))
        for i in ids:
            if i != ident and i not in link['tried']:
                link['tried'].append(i)
        link['cands'] = [i for i in link['cands'] if i == ident or i not in ids]
        if ident not in ids:
            return (0, 1, True)
        self.look(ident, got.get('see'))
        missing = got.get('missing') if isinstance(got.get('missing'), list) else []
        missing = [m for m in (_d_str(x) for x in missing) if m]
        found = {'id': ident, 'missing': len(missing), 'next': _d_str(got.get('next')), 'value': _d_str(got.get('value'))}
        best = link.get('partial')
        if missing and self._more(link, ident):
            if not best or found['missing'] < best['missing']:
                link['partial'] = found
            link['cands'] = [i for i in link['cands'] if i != ident]
            if ident not in link['tried']:
                link['tried'].append(ident)
            return (0, 1, True)
        if missing and best and (best['missing'] <= found['missing']):
            found = best
        n, asked = self._accept(ref, link, found, room)
        return (n, 1 + asked, True)

    def _more(self, link, ident):
        if link.get('looked', 0) >= self.LOOK_ON or link.get('opens', 0) >= self.SWEEP:
            return False
        return bool([i for i in link['cands'] if i != ident]) or bool(self._unopened(link)) or link['rounds'] < self._rounds()

    def _settle(self, ref, link, room):
        best = link.get('partial')
        if not best:
            return (0, 0, False)
        n, asked = self._accept(ref, link, best, room)
        return (n, asked, True)

    def _accept(self, ref, link, found, room):
        ident, nxt, value = (found['id'], found['next'], found['value'])
        used = asked = 0
        views = self._views(ident)
        others = [n for n in self._detail_tools()[1:] if n not in views]
        if others and room <= 0:
            link['held'] = found
            return (used, asked)
        if others:
            used = self._open(ident, room, every=True, skip=views | set(self._detail_tools()[:1]))
            if used:
                asked = 1
                got = self._ask_read(ref, link, [ident])
                if got is not None and _d_str(got.get('id')) == ident:
                    nxt, value = (_d_str(got.get('next')), _d_str(got.get('value')))
                    self.look(ident, got.get('see'))
        link.update({'id': ident, 'opened': True, 'value': value, 'cands': [ident], 'partial': None})
        if nxt and len(ref.chain) < self.MAX_LINKS:
            link['pointer'] = nxt
            ref.chain.append(self._new_link(nxt))
            return (used, asked)
        link['pointer'] = ''
        ref.value = value or None
        ref.done = True
        return (used, asked)
COMPARE_WORDS = re.compile('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', re.I)
D_READING = 'The listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge, a range must lie wholly inside the limit), and when that attribute gives another value the listing does not meet the requirement, even if some other key or the title carries the wanted one; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure that does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole, and it alone decides the kind of product: a title or description that names some other kind of thing does not change it; a rule of being unlike another listing\'s value counts as met unless the records show that same value; a requirement that rules a value out is met only when the listing states some other value for that property, and a listing silent on it leaves it unknown; out-of-stock listings and listings that break the budget do not fit.'
D_REVIEW = 'You review the listings a shopping agent may order for a shopper. REQUIREMENTS lists each requested item\nwith everything it must satisfy. For each record below decide whether it satisfies EVERY requirement of one item.\nReading rules: ' + D_READING + '\nA LINE marked with an item number is meant for that item; judge it against that item. A property the record does\nnot show at all makes it "unsure", never "yes" and never "no"; give "no" only for a requirement the record clearly breaks.\nA record may say that for some property this listing is not the place to read it and send the reader to another\nlisting instead. For a LINE, when such a note is about a property that its item\'s requirements name, list it under\n"sends" with the requirement\'s property and the other listing exactly as the note describes it, and judge that\nproperty "unsure" until it is read there; leave out notes about properties no requirement names.\nReply with one JSON object only:\n{"rows": [{"id": "...", "item": item number it would fill (0 if none), "verdict": "yes" | "no" | "unsure", "why": "few words",\n"sends": [{"property": "...", "listing": "..."}]}]}'
D_AUDIT = "The shopping agent wants to stop without ordering, convinced that no listing satisfies the shopper. Test\nthat claim against the records. OPENED records show a listing's full page. ROW records are search rows only: their page\nhas not been read, so they can be neither passed nor failed on a property the row does not show. Reading rules: " + D_READING + '\nReply with one JSON object only:\n{"fits": {"item number": [ids of listings that appear to meet every requirement of that item, in the order the pick rule prefers (cheapest first for a lowest-price rule, the extreme of the measure first for a largest-or-smallest rule), at most 4]},\n "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show they fit, most promising first, at most 6],\n "reason": "one sentence"}'
D_STEPS = 'The shopper\'s request sets steps the agent must carry out along the way (PROCESS). RECORD lists what the\nagent has done, numbered. Decide which steps due before placing the order are not done for the current cart.\nA step counts as carried out if some RECORD line performs it after the cart\'s current lines were chosen and any reply\naccepts it; questions about product properties do not carry out a step. When a missing step is done by a message\nto the shopper, write that message: one short plain sentence about that step alone.\nReply with one JSON object only: {"missing": ["step", ...], "message": "text, or empty"}'
D_FINAL = 'A shopping program checks its order before placing it. REQUEST is the shopper\'s own words. NEEDS lists the requirements on record for each item (numbered from 1), including what the shopper told and what store notices changed; a requirement a notice dropped is no longer in force. Read them and answer four things.\n- "bounds": for each requirement in NEEDS that sets a number (a lower or upper limit, a range, an exact amount), the item number, the property as NEEDS names it, the lowest and the highest value it allows (null for an open side, both the same for an exact amount; limits include their edge) and the unit as written ("" when none).\n- "rank": when the pick rule ranks the fitting listings by a measure other than price, the item number (0 for every item), that measure\'s property, whether "more" or "less" of it is better, and its unit; null otherwise.\n- "missed": every firm requirement on a requested item that the REQUEST states and NEEDS does not hold, with the item number, the property and the wanted value in the request\'s words; [] when none. Words that only describe another listing the request points at are no requirement of the item.\n- "nearest": true only when the request says that, if no listing meets every requirement, the closest one should still be bought or offered; false otherwise.\nReply with one JSON object only:\n{"bounds": [{"item": 1, "property": "...", "low": number or null, "high": number or null, "unit": "..."}], "rank": {"item": 0, "property": "...", "better": "more" or "less", "unit": "..."} or null, "missed": [{"item": 1, "property": "...", "value": "..."}], "nearest": true or false}'
S1, S0, UNK, MODEL = ('S1', 'S0', 'U', 'M')

def _breaks(amount, cap):
    return cap + 1e-09 < amount <= 10 * cap

def _ck_word(word):
    return word[:-1] if len(word) > 3 and word.endswith('s') and (not word.endswith('ss')) else word

def _ck_tokens(text):
    return [_ck_word(w) for w in re.findall('[a-z0-9]+', str(text or '').lower())]

def _ck_same(a, b):
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))

def _ck_holds(words, tokens):
    return all((any((_ck_same(w, t) for t in tokens)) for w in words))

def _ck_numbers(text):
    text = re.sub('(?<=\\d),(?=\\d{3}\\b)', '', str(text or '').lower())
    return [(float(m.group(1)), _ck_word(m.group(2) or '')) for m in re.finditer('(?<![a-z0-9.])(\\d+(?:\\.\\d+)?)\\s*([a-z]*)', text)]

def _ck_unit_ok(have, want, key):
    want = _ck_word((str(want or '').lower().split() or [''])[0])
    have = _ck_word(have or '')
    if have == want or not want:
        return True
    if have:
        return min(len(have), len(want)) >= 2 and (have.startswith(want) or want.startswith(have))
    return any((_ck_same(want, t) for t in _ck_tokens(key)))

def _ck_float(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(str(value).replace(',', '').strip())
    except ValueError:
        return None

def _ck_code_tokens(text, ids):
    out = set()
    for tok in re.findall('[a-z][a-z0-9-]{3,}', str(text or '').lower()):
        tok = tok.strip('-')
        if tok in ids or sum((c.isdigit() for c in tok)) < 3 or sum((c.isalpha() for c in tok)) < 2:
            continue
        out.add(tok)
    return out

def _ck_key_kind(key, prop):
    have, want = (_iv_norm(key).split(), _iv_norm(prop).split())
    if not have or not want:
        return ''
    if set(have) == set(want):
        return 'same'
    n = len(want)
    if len(have) > n and have[-n:] == want:
        return 'own'
    if any((have[k:k + n] == want for k in range(len(have) - n))):
        return 'part'
    return ''

def _ck_rewording(have, want, prop, vocab=()):
    a, b, names = (_ck_tokens(have), _ck_tokens(want), _ck_tokens(prop))
    shared = [x for x in a if any((_ck_same(x, y) for y in b))]
    if not shared or not all((any((_ck_same(x, n) for n in names)) for x in shared)):
        return False
    mine = [w for w in a if not any((_ck_same(w, y) for y in b))]
    wanted = [w for w in b if not any((_ck_same(w, x) for x in a))]
    if not wanted or any((_ck_digits(w) for w in mine + wanted)):
        return False
    return not any((_ck_same(w, t) for w in wanted for t in vocab))

def _ck_digits(text):
    return any((ch.isdigit() for ch in str(text or '')))

def _ck_values(node, depth=0):
    if depth > 4:
        return []
    if isinstance(node, dict):
        return [x for v in node.values() for x in _ck_values(v, depth + 1)]
    if isinstance(node, (list, tuple)):
        return [x for v in node for x in _ck_values(v, depth + 1)]
    return [str(node)] if isinstance(node, (str, int, float)) and (not isinstance(node, bool)) else []
REVIEW_HEAVY = False
FINAL_CHECK = True
CLEAN_PICK = True
CHAIN_COMPLETE = True
CHAIN_HOLDS_LONG = 6
RANK_ROWS = True
FINAL_ROOM = True
FINAL_TOKENS = 3000
DECLINE_ON_BUY = True
RANK_KEEP_LINE = True
RANK_KEEP_STATED = True
NUM_MIXED_MODEL = True
CHEAP_KEEP_LINE = True
SWAP_ONLY_BROKEN = True
WORD_S0_MODEL = True

class CheckoutGate:
    CODE_HOLDS = 3
    CHAIN_HOLDS = 2
    STEP_SENDS = 2
    CHEAPER = 8
    REVIEW_HOLDS = 3
    OPEN_UNSURE = 3
    ALT_OPENED = 4

    def __init__(self, ledger, llm, toolmap, env, chain):
        self.led, self.llm, self.tm, self.env, self.chain = (ledger, llm, toolmap, env, chain)
        self.reserve = 2
        self.fresh = {}
        self.reviewed = {}
        self.code_holds = 0
        self.chain_holds = 0
        self.sends = 0
        self.review_holds = 0
        self.unstated_held = set()
        self.sends_of = {}
        self.leads = {}
        self.unnamed = set()
        self.audited = {}
        self.unread_audit = False
        self.readings = {}
        self.last_reading = {}
        self.read_calls = 0
        self.part_seen = set()
        self.part_faults = {}
        self.visits = []
        self.verdict = {}
        self._vocab_of = {}
        self._all_words = None
        self.chain_keep = None

    def ready(self):
        return self.ready_for([{'id': i, 'qty': q} for i, q in self.led.cart.items() if q])

    def ready_for(self, lines):
        lines = [dict({'id': str(l['id']), 'qty': int(l.get('qty') or 0)}, **{'item': l['item']} if isinstance(l.get('item'), int) else {}) for l in lines if l.get('id')]
        problem = self._shape(lines) or self._refresh(lines) or self._stock_budget(lines)
        if problem and self.code_holds < self.CODE_HOLDS:
            self.code_holds += 1
            return (False, problem)
        if self.chain.pending(lines) or self.chain.unopened_ids():
            room = min(12 if self.env.steps_left() > 15 else 4, self.env.steps_left() - self.reserve - len(lines) - 1)
            if room > 0:
                self.chain.advance(room)
            holds = self.CHAIN_HOLDS + (1 if self.env.steps_left() > 15 else 0)
            keep = getattr(self, 'chain_keep', None)
            if CHAIN_COMPLETE and keep is not None and (self.env.steps_left() >= keep) and any((not self.chain.stuck(r) for r in self.chain.pending(lines) if hasattr(self.chain, 'stuck'))):
                holds = max(holds, CHAIN_HOLDS_LONG)
            if self.chain.pending(lines) and self.chain_holds < holds:
                self.chain_holds += 1
                opened = self.chain.report() if hasattr(self.chain, 'report') else ''
                return (False, 'Hold the order: the pages a requirement refers to must all be opened first. Still to find and open: ' + ' | '.join(self.chain.missing()[:3]) + ('\nPages opened for it meanwhile (already read):\n' + opened if opened else ''))
        problem = self._review(lines)
        if problem:
            return (False, problem)
        problem = self._steps()
        if problem:
            return (False, problem)
        return (True, '')

    def recheck(self, lines):
        lines = [dict({'id': str(l['id']), 'qty': int(l.get('qty') or 0)}, **{'item': l['item']} if isinstance(l.get('item'), int) else {}) for l in lines if l.get('id')]
        for l in lines:
            self.fresh.pop(l['id'], None)
        self._refresh(lines)
        problem = self._shape(lines) or self._stock_budget(lines)
        if problem:
            return problem
        rows = self._judge(lines, [])
        return self._problems(lines, rows) if rows is not None else ''

    def find_sold(self, lines):
        ids = list(dict.fromkeys((str(l['id']) for l in lines or [] if l.get('id'))))
        cart = [i for i in getattr(self.led, 'cart', None) or {} if i in ids]
        ids = cart + [i for i in ids if i not in cart]
        if not ids:
            return []
        known = [i for i in ids if self._gone(i)]
        if known:
            return known
        if len(ids) == 1:
            return ids
        flags = {}
        if self.env.steps_left() >= len(ids) + self.reserve + 1:
            reader = getattr(self.tm, 'record', None)
            for ident in ids:
                obs = _d_call(self.env, self.tm, self.led, self.tm.tool('detail'), self.tm.args_for('detail', product_id=ident))
                self.fresh[ident] = self.env.step
                record = reader(obs) if callable(reader) else self.led.seen.get(ident)
                flags[ident] = self.tm.in_stock(record) if isinstance(record, dict) else None
            self.reviewed.clear()
        sold = [i for i in ids if flags.get(i) is False]
        if sold:
            return sold
        if len(flags) == len(ids) and all((f is True for f in flags.values())):
            return []
        return ids[:1]

    def should_decline(self):
        sig = (len(self.led.seen), len(self.led.opened), len(self.led.revealed), len(self.led.notices))
        if sig in self.audited:
            return self.audited[sig]
        got = self._audit()
        if got is not None and self._uncovered():
            rows = [str(i) for i in got.get('open') or [] if str(i) in self.led.seen and (not self._tabled(str(i)))]
            if self.led.pick == 'cheapest':
                rows.sort(key=self._price)
            room = min(self.AUDIT_OPENS, self.env.steps_left() - self.reserve - 3)
            if rows and room > 0 and self._open_some(rows, room):
                got = self._audit() or got
        self.unread_audit = got is None
        verdict = got is not None and (self._none_named() or self._over_budget())
        self.audited[sig] = verdict
        return verdict

    def process_now(self):
        return self._steps()

    def fallback_lines(self):
        if not self.leads:
            self.should_decline()
        lines = []
        for n, item in enumerate(self.led.items, 1):
            ids = self.leads.get(n) or []
            if not ids:
                return []
            lines.append({'id': ids[0], 'qty': item.qty})
        return lines

    def whole_cart(self, lines):
        name = self.tm.tool('cart_set')
        if not name or not lines:
            return None
        built = [self.tm.args_for('cart_set', product_id=str(l['id']), qty=int(l.get('qty') or 1), mode='add') for l in lines]
        if not all((isinstance(b, dict) and b for b in built)):
            return None
        key = next((k for k, v in built[0].items() if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict)), None)
        if key is None or not all((isinstance(b.get(key), list) and len(b[key]) == 1 for b in built)):
            return None
        args = dict(built[0])
        args[key] = [b[key][0] for b in built]
        return None if self.tm.validate(name, args) else (name, args)

    def cart_faults(self, lines):
        have = {str(i): int(q) for i, q in (self.led.cart or {}).items() if q}
        want = {str(l['id']): int(l.get('qty') or 0) for l in lines if l.get('id')}
        out = [f'{i} is not in the cart (wanted x{q})' for i, q in want.items() if i not in have]
        out += [f'{i} is held at {have[i]}, wanted x{q}' for i, q in want.items() if i in have and have[i] != q]
        out += [f'{i} x{q} is in the cart but not meant for the order' for i, q in have.items() if i not in want]
        return '; '.join(out)

    def _shape(self, lines):
        want = sorted((it.qty for it in self.led.items))
        if len(lines) != len(want):
            return f'Hold the order: the request has {len(want)} item(s) and the order must have exactly one line per item, but it has {len(lines)}. Lines now: {self._lines_text(lines)}.'
        if sorted((l['qty'] for l in lines)) != want:
            return 'Hold the order: line quantities must match the requested quantities ' + ', '.join((f"item {n} '{it.name}' x{it.qty}" for n, it in enumerate(self.led.items, 1))) + f'. Lines now: {self._lines_text(lines)}.'
        return ''

    def _refresh(self, lines):
        stale = [l['id'] for l in lines if self.fresh.get(l['id'], -99) < self.env.step - 2]
        if not stale or self.env.steps_left() < len(stale) + self.reserve + 1:
            return ''
        changed = False
        for ident in stale:
            before = self.led.seen.get(ident)
            old = (self.tm.price(before), self.tm.in_stock(before)) if isinstance(before, dict) else (None, None)
            _d_call(self.env, self.tm, self.led, self.tm.tool('detail'), self.tm.args_for('detail', product_id=ident))
            self.fresh[ident] = self.env.step
            after = self.led.seen.get(ident)
            new = (self.tm.price(after), self.tm.in_stock(after)) if isinstance(after, dict) else (None, None)
            changed = changed or old != new
        if changed:
            self.reviewed.clear()
        return ''

    def _stock_budget(self, lines):
        gone = [l['id'] for l in lines if self._gone(l['id'])]
        if gone:
            return f"Hold the order: {', '.join(gone)} cannot be bought now (out of stock); replace that line with another listing that satisfies the item."
        cap = self.led.budget_cents
        if cap:
            prices = [self.tm.price(self.led.seen.get(l['id']) or {}) for l in lines]
            if all((p is not None for p in prices)):
                total = sum((p * l['qty'] for p, l in zip(prices, lines)))
                if _breaks(total, cap):
                    return f'Hold the order: the total {total:g} is over the budget {cap:g}; choose cheaper listings that still satisfy the items.'
        return ''

    def _gone(self, ident):
        return ident in getattr(self.led, 'gone', ()) or self.tm.in_stock(self.led.seen.get(ident) or {}) is False

    def _over_budget(self):
        cap = self.led.budget_cents
        if not cap or not self.leads:
            return False
        total = 0
        for n, item in enumerate(self.led.items, 1):
            prices = [self.tm.price(self.led.seen.get(i) or {}) for i in self.leads.get(n) or []]
            prices = [p for p in prices if p is not None]
            if not prices:
                return False
            total += min(prices) * item.qty
        return total > cap

    def _requirements(self):
        text = self.led.summary(3000)
        values = self.chain.values_text()
        return text + ('\nValues read along reference pages:\n' + values if values else '')

    def _cheaper(self, lines):
        if self.led.pick != 'cheapest':
            return []
        mine = {l['id'] for l in lines}
        cat = (getattr(self.tm, 'keys', None) or {}).get('category')
        out = []
        for l in lines:
            rec = self.led.seen.get(l['id']) or {}
            price = self.tm.price(rec)
            if price is None:
                continue
            pool = {i: r for i, r in self.led.seen.items() if i not in mine and isinstance(r, dict) and (not self._gone(i)) and (self.tm.price(r) is not None and self.tm.price(r) < price)}
            kind = str(rec.get(cat) or '').strip().lower() if cat else ''
            if kind:
                same = sorted((i for i, r in pool.items() if str(r.get(cat) or '').strip().lower() == kind), key=self._lead_key)
                out += [i for i in same[:self.CHEAPER] if i not in out]
                continue
            title = ' '.join((str(v) for v in rec.values() if isinstance(v, str) and len(v) < 200))
            out += [i for i in _d_rank(title, pool, limit=self.CHEAPER) if i not in out]
        return out

    def _judge(self, lines, others):
        recs = [f"LINE {l['id']}" + (f" (item {l['item']})" if l.get('item') else '') + f" x{l['qty']}: {_d_brief(self.led.seen.get(l['id']), 300)}" for l in lines]
        recs += [f"OTHER {('OPENED' if self._tabled(i) else 'ROW')} {i}: {_d_brief(self.led.seen.get(i), 200)}" for i in others]
        got = self.llm.ask(D_REVIEW, f'REQUIREMENTS:\n{self._requirements()}\n\nRECORDS:\n' + '\n'.join(recs), heavy=REVIEW_HEAVY, max_tokens=3000)
        if not isinstance(got, dict):
            return None
        rows = {}
        mine = {l['id'] for l in lines}
        for r in got.get('rows') or []:
            if isinstance(r, dict) and str(r.get('id')) in self.led.seen:
                item = r.get('item')
                rows[str(r['id'])] = (item if isinstance(item, int) else 0, _d_str(r.get('verdict')).lower(), _d_str(r.get('why')))
                if str(r['id']) in mine and isinstance(r.get('sends'), list):
                    self.sends_of[str(r['id'])] = [(_d_str(s.get('property')), _d_str(s.get('listing'))) for s in r['sends'] if isinstance(s, dict) and _d_str(s.get('property')) and _d_str(s.get('listing'))]
        return rows

    def _note_refs(self, lines, rows):
        added = []
        for l in lines:
            item = l.get('item') or rows.get(l['id'], (0, '', ''))[0]
            if not 1 <= item <= len(self.led.items):
                continue
            for prop, desc in self.sends_of.get(l['id'], [])[:2]:
                if any((r.owner == l['id'] and r.prop.lower() == prop.lower() for r in self.led.refs)):
                    continue
                ref = Ref(item=item - 1, prop=prop, relation='same', start=desc, owner=l['id'])
                self.led.refs.append(ref)
                added.append(ref)
        return added

    def _review(self, lines):
        sig = json.dumps([sorted(((l['id'], l['qty']) for l in lines)), len(self.led.revealed), len(self.led.notices), [r.value for r in self.led.refs]])
        if sig in self.reviewed or self.review_holds >= self.REVIEW_HOLDS:
            return ''
        others = self._cheaper(lines)
        rows = self._judge(lines, others)
        if rows is None:
            return ''
        unsure = [i for i in others if rows.get(i, (0, 'unsure'))[1] == 'unsure' and (not self._tabled(i))]
        room = self.env.steps_left() - self.reserve - len(lines) - 1
        if unsure and room > 0 and self._open_some(sorted(unsure, key=self._lead_key)[:min(room, self.OPEN_UNSURE)], room):
            rows = self._judge(lines, others) or rows
        alts = self._alternatives(lines, rows, others)
        unread = [i for i in alts if not self._tabled(i)]
        room = self.env.steps_left() - self.reserve - len(lines) - 1
        if unread and room > 0:
            self._open_some(unread[:min(room, self.OPEN_UNSURE)], room)
        if alts:
            rows = self._judge(lines, others + alts) or rows
        for ident, why in (self.code_faults(lines) if FINAL_CHECK else {}).items():
            item = rows.get(ident, (0, '', ''))[0]
            rows[ident] = (item, 'no', why)
        added = self._note_refs(lines, rows)
        if added:
            return "Hold the order: a chosen listing's own page sends the reader to another listing for a required property, and that page must be opened first: " + ' | '.join((f'{r.owner} {r.prop}: {r.start}' for r in added[:3]))
        self.reviewed[sig] = problem = self._problems(lines, rows)
        if problem:
            self.review_holds += 1
        return problem

    def _lead_key(self, ident):
        words = set()
        for item in self.led.items:
            for need in item.needs:
                words |= set(re.findall('[a-z0-9]{3,}', str(getattr(need, 'value', '') or '').lower()))
        text = json.dumps(self.led.seen.get(ident) or {}, ensure_ascii=False).lower()
        return (-sum((1 for w in words if w in text)), self._price(ident))

    def _alternatives(self, lines, rows, others):
        mine = {l['id'] for l in lines}
        cat = (getattr(self.tm, 'keys', None) or {}).get('category')
        out = []
        for l in lines:
            if rows.get(l['id'], (0, '', ''))[1] != 'unsure' or not self._tabled(l['id']):
                continue
            kind = str((self.led.seen.get(l['id']) or {}).get(cat) or '').strip().lower() if cat else ''
            cap = self.led.budget_cents
            pool = [i for i, r in self.led.seen.items() if i not in mine and i not in others and (i not in out) and isinstance(r, dict) and (not self._gone(i)) and (not kind or str(r.get(cat) or '').strip().lower() == kind) and (not (cap and self._price(i) != float('inf') and _breaks(self._price(i), cap)))]
            out += sorted((i for i in pool if self._tabled(i)), key=self._price)[:self.ALT_OPENED]
            out += sorted((i for i in pool if not self._tabled(i)), key=self._price)[:self.OPEN_UNSURE]
        return out

    def _problems(self, lines, rows):
        out, filled = ([], {})
        for l in lines:
            item, verdict, why = rows.get(l['id'], (0, 'unsure', ''))
            item = l.get('item') or item
            if verdict == 'no':
                out.append(f"{l['id']} does not satisfy its item ({why or 'see the requirements'})")
            elif verdict == 'unsure' and self._tabled(l['id']):
                sure = sorted((i for i, (it, v, _) in rows.items() if v == 'yes' and i not in {x['id'] for x in lines} and (it == item or not item)), key=self._price)
                if sure:
                    out.append(f"{l['id']}'s page does not show that it meets every requirement ({why or 'a value is missing'}), while {sure[0]} ({self._price(sure[0]):g}) is shown to meet them all; order that one instead")
                elif l['id'] not in self.unstated_held:
                    self.unstated_held.add(l['id'])
                    out.append(f"{l['id']}'s page does not state every required value ({why or 'a value is missing'}); open listings of its kind whose page states it and order one of those, or order {l['id']} again if none does")
            if 1 <= item <= len(self.led.items):
                filled.setdefault(item, []).append(l)
                cap = getattr(self.led.items[item - 1], 'cap_cents', None)
                if cap is not None and self._price(l['id']) != float('inf') and _breaks(self._price(l['id']), cap):
                    out.append(f"{l['id']} costs {self._price(l['id']):g} each, over that item's limit {cap:g}")
        if len(filled) == len(lines):
            for item, ls in filled.items():
                want = self.led.items[item - 1].qty
                if len(ls) > 1:
                    out.append(f'item {item} has {len(ls)} lines; it needs one')
                elif ls[0]['qty'] != want:
                    out.append(f"{ls[0]['id']} must be quantity {want}")
        for l in lines:
            item, verdict, _ = rows.get(l['id'], (0, '', ''))
            item = l.get('item') or item
            price = self._price(l['id'])
            if self.led.pick != 'cheapest' or price == float('inf') or (not (item or len(self.led.items) == 1)):
                continue
            better = sorted((i for i, (it, v, _) in rows.items() if v == 'yes' and i not in {x['id'] for x in lines} and (it == item or not item) and (self._price(i) < price)), key=self._price)
            if better:
                out.append(f"{better[0]} at {self._price(better[0]):g} also satisfies the item of {l['id']} (at {price:g}) and is cheaper; the request wants the cheapest")
        if not out:
            return ''
        return 'Hold the order: ' + '; '.join(out[:5]) + '. Fix the lines, then try to order again.'

    def _tabled(self, ident):
        key = (getattr(self.tm, 'keys', None) or {}).get('props')
        record = self.led.seen.get(ident)
        if key and isinstance(record, dict):
            return isinstance(record.get(key), dict)
        return ident in self.led.opened

    def _price(self, ident):
        p = self.tm.price(self.led.seen.get(ident) or {})
        return p if p is not None else float('inf')

    def _messages_left(self):
        sent = getattr(self.led, 'msgs_sent', None)
        if sent is None:
            sent = sum((1 for h in self.env.history if self.tm.role.get(h.get('name')) == 'message'))
        budget = self.led.msg_budget if self.led.msg_budget is not None else 3
        return budget - sent

    def _process(self):
        return list(self.led.process or [])

    def _steps(self):
        steps = self._process()
        if not steps or self.sends >= self.STEP_SENDS:
            return ''
        record = '\n'.join((f"{n}. {h.get('name')} {json.dumps(h.get('args'), ensure_ascii=False)[:160]} -> {_d_brief(h.get('obs'), 120)[:300]}" for n, h in enumerate(self.env.history[-25:], 1)))
        got = self.llm.ask(D_STEPS, 'PROCESS:\n' + '\n'.join(steps) + '\n\nCART: ' + _d_brief(self.led.cart) + '\n\nRECORD:\n' + record, max_tokens=1500)
        if not isinstance(got, dict) or not got.get('missing'):
            return ''
        text = _d_str(got.get('message'))
        if not text or self._messages_left() <= 0 or self.env.steps_left() <= self.reserve + 1:
            return ''
        self.sends += 1
        obs = _d_call(self.env, self.tm, self.led, self.tm.tool('message'), self.tm.args_for('message', text=text))
        self.reviewed.clear()
        return 'A step the request asks for was just carried out by message; the reply: ' + _d_brief(obs, 400)[:600] + '. Apply anything it changes, then try to order again.'
    AUDIT_RECORDS = 30
    AUDIT_FITS = 4
    AUDIT_OPENS = 4

    def _audit(self):
        recs = []
        for i in reversed(list(self.led.seen)):
            r = self.led.seen[i]
            if self._gone(i):
                continue
            recs.append(f"{('OPENED' if self._tabled(i) else 'ROW')} {i}: {_d_brief(r, 200 if self._tabled(i) else 120)}")
            if len(recs) >= self.AUDIT_RECORDS:
                break
        if not recs:
            return None
        got = self.llm.ask(D_AUDIT, f'REQUIREMENTS:\n{self._requirements()}\n\nRECORDS:\n' + '\n'.join(recs), heavy=self._heavy_room(), max_tokens=4000)
        print('AUDIT', json.dumps(got)[:600], file=sys.stderr)
        if not isinstance(got, dict):
            return None
        if not isinstance(got.get('fits'), dict):
            return None
        self.leads, self.unnamed = ({}, set())
        for k, ids in got['fits'].items():
            number = re.search('\\d+', str(k))
            if not number and len(self.led.items) != 1:
                continue
            n = int(number.group()) if number else 1
            ids = ids if isinstance(ids, list) else [ids]
            ok = [str(i) for i in ids if str(i) in self.led.seen and (not self._gone(str(i)))]
            if ok:
                self.leads[n] = (sorted(ok, key=self._price) if self.led.pick == 'cheapest' else ok)[:self.AUDIT_FITS]
            elif any((str(i).strip() and str(i) not in self.led.seen for i in ids)):
                self.unnamed.add(n)
        return got

    def _heavy_room(self):
        room = getattr(self.llm, 'heavy_room', None)
        return bool(room()) if callable(room) else REVIEW_HEAVY

    def _none_named(self):
        return not self.unnamed and (not any((self.leads.get(n) for n in range(1, len(self.led.items) + 1))))

    def _uncovered(self):
        return [n for n in range(1, len(self.led.items) + 1) if not self.leads.get(n) and n not in self.unnamed]

    def _open_some(self, ids, room):
        used = 0
        for ident in ids:
            if used >= room or self.env.steps_left() <= self.reserve + 1:
                break
            used += 1
            obs = _d_call(self.env, self.tm, self.led, self.tm.tool('detail'), self.tm.args_for('detail', product_id=ident))
            _d_mark_opened(self.tm, self.led, ident, obs)
        return used > 0

    def _lines_text(self, lines):
        return ', '.join((f"{l['id']} x{l['qty']}" for l in lines)) or '(none)'
    VERIFY_OPENS = 3
    PART_POINTERS = 2
    FINAL_READS = 3

    def _needs(self, n):
        fn = getattr(self.led, 'needs_of', None)
        return fn(n) if callable(fn) else list(self.led.items[n].needs)

    @staticmethod
    def _twin(a, b):
        one, two = (_iv_wordset(a or ''), _iv_wordset(b or ''))
        return bool(one and two) and (one <= two or two <= one)

    def reading(self):
        sig = json.dumps([[str(getattr(x, 'text', x)) for x in self._needs(n)] for n in range(len(self.led.items))] + [[str(getattr(r, 'value', '')) for r in self.led.refs], len(self.led.notices)])
        if sig in self.readings:
            return self.readings[sig]
        closed = getattr(self.llm, 'closed', None)
        if self.llm is None or self.read_calls >= self.FINAL_READS or (callable(closed) and closed()):
            return self.last_reading
        self.read_calls += 1
        query = str(getattr(self.led, 'query', '') or '')
        got = self.llm.ask(D_FINAL, (f'REQUEST:\n{query}\n\n' if query else '') + 'NEEDS:\n' + self._requirements(), heavy=False, max_tokens=FINAL_TOKENS if FINAL_ROOM else 1500)
        got = got if isinstance(got, dict) else {}
        print('FINAL', json.dumps(got)[:600], file=sys.stderr)
        self.readings[sig] = self.last_reading = got
        return got

    @staticmethod
    def _item_no(value):
        number = _ck_float(value)
        return int(number) if number is not None and number.is_integer() else None

    def _test(self, n, prop, value, kind, bounds):
        for b in bounds:
            k = self._item_no(b.get('item'))
            if (k == n + 1 or len(self.led.items) == 1) and self._twin(_d_str(b.get('property')), prop):
                low, high = (_ck_float(b.get('low')), _ck_float(b.get('high')))
                if low is not None or high is not None:
                    return ('num', low, high, _d_str(b.get('unit')))
        if kind == 'excluded':
            return ('not', value)
        parts = [p.strip() for p in re.split('\\s*(?:,|/|\\bor\\b)\\s*', value, flags=re.I) if p.strip()]
        if not parts or any((COMPARE_WORDS.search(p) for p in parts)):
            return ('model',)
        if len(parts) == 1:
            figures = _ck_numbers(value)
            units = {u for _, u in figures}
            rest = [w for w in _ck_tokens(value) if not re.fullmatch('\\d+', w) and w not in units]
            if len(figures) == 1 and (not rest):
                return ('num', figures[0][0], figures[0][0], figures[0][1])
        return ('word', parts)

    def rows_for(self, reading):
        reading = reading if isinstance(reading, dict) else {}
        bounds = [b for b in reading.get('bounds') or [] if isinstance(b, dict)]
        out = {}
        for n in range(len(self.led.items)):
            rows = []
            for need in self._needs(n):
                prop = str(getattr(need, 'prop', '') or str(getattr(need, 'text', '')).split(':', 1)[0]).strip()
                if not prop:
                    continue
                kind, value = (getattr(need, 'kind', 'stated'), str(getattr(need, 'value', '') or '').strip())
                if kind == 'ref':
                    ref = next((r for r in self.led.refs if r.item == n and self._twin(r.prop, prop) and r.done and r.value and (getattr(r, 'relation', 'same') == 'same')), None)
                    if ref is None:
                        rows.append((prop, ('model',), 'need'))
                        continue
                    kind, value = ('stated', str(ref.value).strip())
                if getattr(need, 'when', None) or kind not in ('stated', 'hidden', 'excluded'):
                    rows.append((prop, ('model',), 'need'))
                    continue
                if value:
                    rows.append((prop, self._test(n, prop, value, kind, bounds), 'need'))
            for m in reading.get('missed') or []:
                if not isinstance(m, dict):
                    continue
                k = self._item_no(m.get('item'))
                if k != n + 1 and (not (k is None and len(self.led.items) == 1)):
                    continue
                prop, value = (_d_str(m.get('property')), _d_str(m.get('value')))
                if prop and value and (not any((self._twin(prop, p) for p, _, _ in rows))):
                    rows.append((prop, self._test(n, prop, value, 'stated', bounds), 'missed'))
            out[n] = rows
        return out

    def _vocab(self, prop):
        key = _iv_norm(prop)
        if key not in self._vocab_of:
            words = set()
            for ident, rec in self.led.seen.items():
                if isinstance(rec, dict):
                    for k, v in _iv_facts(rec, str(ident))[0].items():
                        if _ck_key_kind(k, prop):
                            words.update(_ck_tokens(v))
            self._vocab_of[key] = words
        return self._vocab_of[key]

    def _store_words(self):
        if self._all_words is None:
            self._all_words = set(_ck_tokens(json.dumps(list(self.led.seen.values()), ensure_ascii=False)))
        return self._all_words

    def _facts(self, ident):
        rec = self.led.seen.get(ident)
        return _iv_facts(rec, str(ident))[0] if isinstance(rec, dict) else {}

    def _keys_for(self, ident, prop):
        kinds = [(_ck_key_kind(k, prop), k, str(v)) for k, v in self._facts(ident).items()]
        same = [(k, v) for kind, k, v in kinds if kind == 'same']
        own = same or [(k, v) for kind, k, v in kinds if kind == 'own']
        return (own, own + [(k, v) for kind, k, v in kinds if kind == 'part'])

    def _own_key(self, ident, prop):
        return any((_ck_key_kind(k, prop) for k in self._facts(ident)))

    def grade(self, ident, prop, test):
        if test[0] == 'model':
            return MODEL
        rec = self.led.seen.get(ident)
        facts = self._facts(ident)
        own, evidence = self._keys_for(ident, prop)
        if test[0] == 'num':
            if not own:
                return UNK
            low, high, unit = (test[1], test[2], test[3])
            figures = [x for k, v in own for x, u in _ck_numbers(v) if _ck_unit_ok(u, unit, k)]
            if not figures:
                return MODEL
            inside = all(((low is None or x >= low - 1e-09) and (high is None or x <= high + 1e-09) for x in figures))
            if inside and NUM_MIXED_MODEL and any((u and (not _ck_unit_ok(u, unit, k)) for k, v in own for _, u in _ck_numbers(v))):
                return MODEL
            return S1 if inside else S0
        vocab = self._vocab(prop)
        decides = lambda w: len(w) >= 4 or any((ch.isdigit() for ch in w)) or any((_ck_same(w, t) for t in vocab))
        if test[0] == 'not':
            if not own:
                return UNK
            words = [w for w in _ck_tokens(test[1]) if decides(w)]
            if not words:
                return MODEL
            return S0 if any((_ck_holds(words, _ck_tokens(v)) for _, v in own)) else S1
        choices = [[w for w in _ck_tokens(c) if decides(w)] for c in test[1]]
        if any((c and _ck_holds(c, _ck_tokens(v)) for c in choices for _, v in evidence)):
            return S1
        if own:
            if not any(choices):
                return MODEL
            if all((_ck_digits(v) != _ck_digits(c) for _, v in own for c in test[1])):
                return MODEL
            if WORD_S0_MODEL and any((_ck_rewording(v, c, prop, vocab) for _, v in own for c in test[1])):
                return MODEL
            return S0
        alias = [str(v) for k, v in facts.items() if self._twin(k, prop)]
        if any((_ck_tokens(v) == _ck_tokens(c) for v in alias for c in test[1])):
            return S1
        if isinstance(rec, dict):
            text = set(_ck_tokens(' '.join(_ck_values(rec))))
            store = self._store_words()
            for c in test[1]:
                words = [w for w in _ck_tokens(c) if w in store]
                if words and _ck_holds(words, text):
                    return S1
        return UNK

    def code_faults(self, lines):
        out = {}
        self._vocab_of, self._all_words = ({}, None)
        rows = self.rows_for(self.last_reading)
        for k, l in enumerate(lines):
            n = self._line_item(l, k, len(lines))
            for prop, test, source in rows.get(n, []):
                if source == 'need' and self.grade(l['id'], prop, test) == S0:
                    out[l['id']] = f"its own '{prop}' attribute gives another value than the requirement"
                    break
        return out

    def _line_item(self, line, k, count):
        item = line.get('item')
        if isinstance(item, int) and (not isinstance(item, bool)) and (1 <= item <= len(self.led.items)):
            return item - 1
        return k if count == len(self.led.items) else 0

    def _pool(self, n, line_id):
        cat = (getattr(self.tm, 'keys', None) or {}).get('category')
        rec = self.led.seen.get(line_id)
        label = str(rec.get(cat) or '').strip().lower() if cat and isinstance(rec, dict) else ''
        if label:
            ids = [i for i, r in self.led.seen.items() if isinstance(r, dict) and str(r.get(cat) or '').strip().lower() == label]
        elif len(self.led.items) == 1:
            ids = [i for i, r in self.led.seen.items() if isinstance(r, dict)]
        else:
            ids = [line_id] + [i for i in self.leads.get(n + 1, []) if i != line_id]
        cap, budget = (getattr(self.led.items[n], 'cap_cents', None), self.led.budget_cents)
        qty = max(1, int(getattr(self.led.items[n], 'qty', 1) or 1))
        out = [line_id] if line_id in self.led.seen else []
        for i in ids:
            price = self._price(i)
            if i in out or self._gone(i):
                continue
            if price != float('inf') and (cap is not None and _breaks(price, cap) or (budget and _breaks(price * qty, budget))):
                continue
            out.append(i)
        return out

    def _measure(self, ident, rank, bare=False):
        own, _ = self._keys_for(ident, _d_str(rank.get('property')))
        figures = [x for k, v in own for x, u in _ck_numbers(v) if _ck_unit_ok(u, rank.get('unit'), k) or (bare and (not u))]
        if not figures:
            return None
        return max(figures) if rank.get('better') != 'less' else min(figures)

    def _rank_for(self, n, reading):
        rank = reading.get('rank') if isinstance(reading, dict) else None
        if self.led.pick != 'best' or not isinstance(rank, dict) or rank.get('better') not in ('more', 'less'):
            return None
        k = self._item_no(rank.get('item'))
        return rank if k in (None, 0, n + 1) else None

    def _judge_pool(self, pool, rows):
        out = {}
        for i in pool:
            grades = [(self.grade(i, prop, test), source) for prop, test, source in rows]
            fail = any((g == S0 and s == 'need' for g, s in grades)) or i in self.part_faults
            unknown = sum((1 for g, _ in grades if g != S1))
            out[i] = (self._tabled(i) and (not fail) and (unknown == 0), fail, unknown)
        return out

    def _footing(self, ident, rows):
        if not self._tabled(ident):
            return ''
        weak, clean = (ident in self.part_faults, True)
        for prop, test, source in rows:
            g = self.grade(ident, prop, test)
            keyed = bool(self._keys_for(ident, prop)[1]) or any((self._twin(k, prop) for k in self._facts(ident)))
            if g == S0 and source == 'need':
                weak = True
            if g == S1:
                continue
            if g == UNK or (g == MODEL and (not keyed)):
                if source == 'need':
                    weak = True
                clean = False
        return 'weak' if weak else 'clean' if clean else ''

    def _clean_choice(self, n, line_id, pool, rows, reading, place):
        if not rows or self._footing(line_id, rows) != 'weak':
            return None
        clean = [i for i in pool if i != line_id and self._footing(i, rows) == 'clean']
        if not clean:
            return None
        if self.led.pick == 'cheapest':
            best = min(clean, key=lambda i: (self._price(i), place(i)))
            return (best, 'the cheapest opened listing that states every required property and breaks none')
        rank = self._rank_for(n, reading)
        if rank:
            more = rank.get('better') != 'less'
            values = {i: self._measure(i, rank) for i in clean}
            best = sorted(clean, key=lambda i: (values[i] is None, -(values[i] or 0) if more else values[i] or 0, place(i)))[0]
            return (best, 'the opened listing that states every required property, by the rank measure')
        return (sorted(clean, key=place)[0], 'an opened listing that states every required property and breaks none')

    def _order_key(self, n):
        place = {}
        for k, i in enumerate(list(self.leads.get(n + 1, [])) + list(getattr(self.led, 'order', None) or self.led.seen)):
            place.setdefault(i, k)
        return lambda i: place.get(i, len(place))

    def _choose(self, n, line_id, pool, judged, reading, rows=None):
        place = self._order_key(n)
        verified = [i for i in pool if judged[i][0]]
        if not verified:
            got = self._clean_choice(n, line_id, pool, rows, reading, place) if CLEAN_PICK and rows else None
            return got or (line_id, '')
        if self.led.pick == 'cheapest':
            best = min(verified, key=lambda i: (self._price(i), place(i)))
            kept = CHEAP_KEEP_LINE and rows and (self._price(best) >= self._price(line_id) - 1e-09) and (self._footing(line_id, rows) == 'clean') and (not any((self.grade(line_id, p, t) == S0 for p, t, _ in rows)))
            if line_id not in verified and (not kept) or self._price(best) < self._price(line_id) - 1e-09:
                return (best, 'the cheapest listing shown to meet every requirement')
            return (line_id, '')
        rank = self._rank_for(n, reading)
        if rank:
            values = {i: self._measure(i, rank) for i in verified}
            values = {i: v for i, v in values.items() if v is not None}
            if values:
                more = rank.get('better') != 'less'
                best = sorted(values, key=lambda i: (-values[i] if more else values[i], place(i)))[0]
                mine = values.get(line_id)
                if RANK_KEEP_LINE and mine is None and (not judged[line_id][1]) and (not RANK_KEEP_STATED or not rows or self._footing(line_id, rows) == 'clean'):
                    mine = self._measure(line_id, rank, bare=True)
                if mine is None and line_id in verified and self._own_key(line_id, _d_str(rank.get('property'))):
                    return (line_id, '')
                if mine is None or (values[best] > mine + 1e-09 if more else values[best] < mine - 1e-09):
                    return (best, f"the {('largest' if more else 'smallest')} {rank.get('property')} shown to meet every requirement")
                return (line_id, '')
        if line_id in verified or (SWAP_ONLY_BROKEN and rows and (self._footing(line_id, rows) == 'clean') and (not any((self.grade(line_id, p, t) == S0 for p, t, _ in rows)))):
            return (line_id, '')
        return (sorted(verified, key=place)[0], 'shown under its own keys to meet every requirement')

    def _rows_to_open(self, n, line_id, pool, judged, reading, forced, judged_none):
        rows = [i for i in pool if not self._tabled(i) and i != line_id]
        if not rows:
            return []
        verified = [i for i in pool if judged[i][0]]
        if self.led.pick == 'cheapest':
            top = min([self._price(i) for i in verified] or [float('inf')])
            return sorted([i for i in rows if self._price(i) < top], key=self._lead_key)
        rank = self._rank_for(n, reading)
        if rank:
            more = rank.get('better') != 'less'

            def figure(i):
                text = json.dumps(self.led.seen.get(i) or {}, ensure_ascii=False)
                got = [x for x, u in _ck_numbers(text) if u and _ck_unit_ok(u, rank.get('unit'), '')] if rank.get('unit') else []
                if RANK_ROWS:
                    own = self._measure(i, rank)
                    if own is None:
                        rec, prop = (self.led.seen.get(i) or {}, _d_str(rank.get('property')))
                        own = [x for k, v in rec.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool)) and (_ck_key_kind(k, prop) in ('same', 'own')) for x, u in _ck_numbers(str(v)) if _ck_unit_ok(u, rank.get('unit'), k)]
                        own = (max(own) if more else min(own)) if own else None
                    got = [own] if own is not None else got
                return (0, -max(got) if more else min(got)) if got else (1, 0)
            order = sorted(rows, key=lambda i: (figure(i), self._lead_key(i)))
            if RANK_ROWS:
                tops = [m for m in (self._measure(i, rank) for i in verified) if m is not None]
                if tops:
                    top = max(tops) if more else min(tops)
                    beats = lambda f: f[0] == 1 or (-f[1] > top + 1e-09 if more else f[1] < top - 1e-09)
                    order = [i for i in order if beats(figure(i))]
            return order
        opened = [i for i in pool if self._tabled(i)]
        stuck = forced and judged_none and opened and all((judged[i][1] for i in opened))
        if line_id in verified and (not stuck):
            return []
        return sorted(rows, key=self._lead_key)

    def final_check(self, lines, room=0, forced=False, judged_none=False, uncertain=False):
        lines = [dict(l) for l in lines or [] if l.get('id')]
        if not lines or not FINAL_CHECK:
            return (lines, False)
        start = self.env.step
        reading = self.reading()
        rows = self.rows_for(reading)
        items = [self._line_item(l, k, len(lines)) for k, l in enumerate(lines)]

        def survey():
            self._vocab_of, self._all_words = ({}, None)
            got = {}
            for l, n in zip(lines, items):
                pool = self._pool(n, l['id'])
                got[l['id']] = (pool, self._judge_pool(pool, rows.get(n, [])))
            return got
        own = list(dict.fromkeys((l['id'] for l in lines if l['id'] in self.led.seen and (not self._tabled(l['id'])))))
        own = own[:max(0, room)]
        if own:
            self._open_some(own, len(own))
        seen = survey()
        wanted = [self._rows_to_open(n, l['id'], *seen[l['id']], reading, forced, judged_none or uncertain) for l, n in zip(lines, items)]
        queue = [lane[k] for k in range(max([len(w) for w in wanted] or [0])) for lane in wanted if k < len(lane)]
        queue = list(dict.fromkeys(queue))[:min(self.VERIFY_OPENS, max(0, room - (self.env.step - start)))]
        if queue and self._open_some(queue, len(queue)):
            seen = survey()
        new, notes = self._pick_lines(lines, items, seen, reading, rows)
        if self._over_cap(new) and (not self._over_cap(lines)):
            new, notes = ([dict(l) for l in lines], [])
        left = max(0, room - (self.env.step - start))
        if PART_TRACE and left > 0 and self.trace_parts(new, left) and any((l['id'] in self.part_faults for l in new)):
            more, extra = self._pick_lines(new, items, {}, reading, rows)
            new, notes = (more, notes + extra)
        decline = False
        broken = next((n for l, n in zip(lines, items) if [i for i in seen[l['id']][0] if self._tabled(i)] and all((seen[l['id']][1][i][1] for i in seen[l['id']][0] if self._tabled(i)))), None)
        if forced and (judged_none or uncertain) and (broken is not None):
            nearest = getattr(self.led, 'nearest', None)
            if nearest is None:
                nearest = isinstance(reading, dict) and reading.get('nearest') is True
            if not nearest:
                decline = True
                notes.append(f'item {broken + 1}: every opened listing of its kind breaks a requirement')
        self.verdict = {n + 1: l['id'] for l, n in zip(new, items)}
        print('VERIFY', json.dumps({'lines': [l['id'] for l in new], 'decline': decline, 'notes': notes})[:600], file=sys.stderr)
        return (new, decline)

    def evidence_lines(self):
        if len(self.led.items) != 1:
            return []
        opened = [i for i, r in self.led.seen.items() if isinstance(r, dict) and self._tabled(i) and (not self._gone(i))]
        if not opened:
            return []
        cat = (getattr(self.tm, 'keys', None) or {}).get('category')
        if cat:
            label = lambda i: str(self.led.seen[i].get(cat) or '').strip().lower()
            named = [label(i) for i in self.leads.get(1, []) if i in opened and label(i)]
            labels = [label(i) for i in opened if label(i)]
            kind = (named or sorted(labels, key=lambda x: -labels.count(x)) or [''])[0]
            opened = [i for i in opened if label(i) == kind] if kind else opened
        cap, budget = (getattr(self.led.items[0], 'cap_cents', None), self.led.budget_cents)
        qty = max(1, int(getattr(self.led.items[0], 'qty', 1) or 1))
        pool = [i for i in opened if self._price(i) == float('inf') or not (cap is not None and _breaks(self._price(i), cap) or (budget and _breaks(self._price(i) * qty, budget)))]
        self._vocab_of, self._all_words = ({}, None)
        judged = self._judge_pool(pool, self.rows_for(self.last_reading or {}).get(0, []))
        fit = [i for i in pool if not judged[i][1]]
        if not fit:
            return []
        place = self._order_key(0)
        price = self._price if self.led.pick == 'cheapest' else lambda i: 0
        best = sorted(fit, key=lambda i: (not judged[i][0], judged[i][2], price(i), place(i)))[0]
        return [{'item': 1, 'id': best, 'qty': qty}]

    def broken_item(self, lines):
        if not DECLINE_ON_BUY or not FINAL_CHECK:
            return None
        lines = [l for l in lines or [] if isinstance(l, dict) and l.get('id')]
        if not lines:
            return None
        reading = self.last_reading if isinstance(self.last_reading, dict) else {}
        nearest = getattr(self.led, 'nearest', None)
        if nearest is None:
            nearest = reading.get('nearest') is True
        if nearest:
            return None
        rows = self.rows_for(reading)
        self._vocab_of, self._all_words = ({}, None)
        cat = (getattr(self.tm, 'keys', None) or {}).get('category')
        for k, l in enumerate(lines):
            n, line_id = (self._line_item(l, k, len(lines)), str(l['id']))
            needs = [(p, t) for p, t, s in rows.get(n, []) if s == 'need' and t[0] != 'model']
            rec = self.led.seen.get(line_id)
            label = str(rec.get(cat) or '').strip().lower() if cat and isinstance(rec, dict) else ''
            if not needs or (not label and len(self.led.items) != 1):
                continue
            kind = lambda r: str(r.get(cat) or '').strip().lower() if cat else ''
            cap, budget = (getattr(self.led.items[n], 'cap_cents', None), self.led.budget_cents)
            qty = max(1, int(getattr(self.led.items[n], 'qty', 1) or 1))
            pool = []
            for i, r in self.led.seen.items():
                if not isinstance(r, dict) or self._gone(i) or (label and kind(r) not in (label, '')):
                    continue
                if label and kind(r) == '' and self._tabled(i):
                    continue
                price = self._price(i)
                if price != float('inf') and (cap is not None and _breaks(price, cap) or (budget and _breaks(price * qty, budget))):
                    continue
                pool.append(i)
            opened = [i for i in pool if self._tabled(i)]
            if not opened:
                continue
            breach = {}
            for i in pool:
                hit = next((p for p, t in needs if self.grade(i, p, t) == S0), None)
                if hit is not None:
                    breach[i] = hit
            if any((i not in breach for i in pool)):
                continue
            said = '; '.join((f'{i} ({breach[i]})' for i in opened[:6]))
            return (n + 1, f"every opened listing of item {n + 1}'s kind gives another value than a requirement under its own attribute: {said}, and no unread listing seen could fit")
        return None

    def _pick_lines(self, lines, items, seen, reading, rows):
        new, notes = ([], [])
        for l, n in zip(lines, items):
            entry = seen.get(l['id'])
            if entry is None:
                pool = self._pool(n, l['id'])
                entry = (pool, self._judge_pool(pool, rows.get(n, [])))
            pool, judged = entry
            chosen, why = self._choose(n, l['id'], pool, judged, reading, rows.get(n, []))
            if chosen != l['id']:
                notes.append(f"item {n + 1}: {l['id']} -> {chosen} ({why})")
            new.append(dict(l, id=chosen))
        return (new, notes)

    def _over_cap(self, lines):
        cap = self.led.budget_cents
        if not cap:
            return False
        prices = [self._price(l['id']) for l in lines]
        if any((p == float('inf') for p in prices)):
            return False
        return _breaks(sum((p * int(l.get('qty') or 1) for p, l in zip(prices, lines))), cap)

    def _part_pointers(self):
        mod = sys.modules.get(type(self.chain).__module__) if self.chain is not None else None
        fn = getattr(mod, 'part_pointers', None) or globals().get('part_pointers')
        return fn if callable(fn) else None

    def _holder(self, code, line_id):
        title = getattr(self.tm, 'title', None)
        low = code.lower()
        for ident, rec in self.led.seen.items():
            if ident == line_id or not isinstance(rec, dict):
                continue
            try:
                text = str(title(rec) or '') if callable(title) else ''
            except Exception:
                text = ''
            if low in text.lower():
                return ident
        return None

    def trace_parts(self, lines, room):
        fn = self._part_pointers()
        tracker = getattr(self.chain, 'pointers', None) if self.chain is not None else None
        if not fn and (not callable(tracker)) or room <= 0:
            return 0
        used = 0
        for l in lines:
            rec = self.led.seen.get(l['id'])
            if not isinstance(rec, dict):
                continue
            try:
                got = tracker(l['id']) if callable(tracker) else fn(rec, self.led.seen)
                codes = [str(c).strip() for c in got or [] if str(c).strip()]
            except Exception:
                continue
            for code in codes[:self.PART_POINTERS]:
                if used >= room or self.env.steps_left() <= self.reserve + 1:
                    return used
                if (l['id'], code.lower()) in self.part_seen:
                    continue
                self.part_seen.add((l['id'], code.lower()))
                found = self._holder(code, l['id'])
                tool = self.tm.tool('search')
                if found is None and tool:
                    args = self.tm.args_for('search', query=code)
                    before = self.env.step
                    obs = _d_call(self.env, self.tm, self.led, tool, args) if args else None
                    used += self.env.step - before
                    found = next((str(self.tm.row_id(r)) for r in _d_rows(self.tm, obs) if code.lower() in json.dumps(r, ensure_ascii=False).lower() and str(self.tm.row_id(r)) != l['id']), None)
                if found and (not self._tabled(found)) and (used < room) and self._open_some([found], 1):
                    used += 1
                self.visits.append((l['id'], code, found))
                fault = self._part_fault(l, found, code, lines) if found else ''
                if fault:
                    self.part_faults[l['id']] = fault
        return used

    def _part_fault(self, line, found, code, lines):
        ids = {str(i).lower() for i in self.led.seen}
        text = lambda i: json.dumps(self.led.seen.get(i) or {}, ensure_ascii=False)
        named = _ck_code_tokens(text(found), ids) - {code.lower()} - _ck_code_tokens(text(line['id']), ids)
        partners = set()
        for other in lines:
            if other['id'] != line['id']:
                partners |= _ck_code_tokens(text(other['id']), ids)
        if not named or not partners or named & partners:
            return ''
        chosen = {l['id'] for l in lines} | {found}
        elsewhere = set()
        for i in self.led.seen:
            if i not in chosen:
                elsewhere |= _ck_code_tokens(text(i), ids)
        hit = named & elsewhere
        return f"{found}, the part its page points at, names {', '.join(sorted(hit)[:3])} and none of the other chosen listings" if hit else ''
GUIDE = "You are the hands of a shopping program that buys on behalf of a single shopper from an online store, and you work only through tool calls. Grading is all or nothing per job, and full marks need the right ending. That is either one order with a line for every item the shopper wants, at the number of units asked for, where every line satisfies all requirements that still apply when the order is placed and the sum stays inside the money limit; or, if for at least one item the store really carries nothing acceptable, ending with no order.\n\nSTEPS AND TOOLS\n- Every call to a store tool spends a step and the steps run out; thinking spends nothing. Each store names its tools, arguments and result fields its own way: read the schemas and copy argument names and permitted values exactly.\n- Search tools are often limited: a category filter, a text field that may look at only the first few words, small pages, a page token or offset. If a result says there are more matches than you have seen and the answer may lie among them, ask for the next page, passing the token or offset unchanged.\n- Keep query text short and lead with the value you want, then the product kind in the request's own words; when that finds few listings of the kind, try the kind as the titles name one unit (often the singular). When a search with a filter comes back empty, suspect the filter: reuse the label as the result rows write it, or filter on a different argument.\n- A bare kind word usually matches hundreds of listings, so narrow: search the kind together with one or two of that item's required values (stated in the request or revealed by the shopper), then vary which values you combine. When a narrowed search reports more matches than its page shows, turn its pages until you have seen them all before trying other words. Every item needs its own searches; an item with no listing found yet comes before more work on items already covered.\n- A value the request rules out is never a search word: a search with it brings back the very listings that fail. Search that item by its kind alone or with its other required values, and open rows that show some other value.\n- The kind of product is decided by the listing's category label: only a label that matches the requested item as a whole counts, never a neighbouring one, and a title or description naming some other kind of thing does not change it.\n- A search row is only a summary. Properties often appear only when a listing is opened, or in a view that puts several listings side by side; requirements are decided on those properties, so look at them for any listing you might order. Prefer one call covering several listings over several calls. Reopening a listing teaches nothing unless something changed.\n- Questions to the shopper are asked by the program, never by you; the answers are merged into the needs you are shown. The request's own end-of-job steps (asking for approval, sending a note) are also done by the program at buy_now or nothing_fits: do not hold back for them, and they are no reason to end empty-handed.\n\nJUDGING A LISTING\n- " + D_READING + '\n- A pointer requirement borrows its value from another listing\'s page (equal to it, not above it, not below it, within a margin of it, or different from it). Locate that listing and open its page with a tool that shows one listing or a few side by side (a search row does not count). If the page names yet another listing as the source for that property, locate and open that one too, until a page states the value itself. If any page on that path stays unopened, the order fails. A pointer to be different excludes only listings carrying that exact value; it never justifies ending with no order. A listing\'s own page can also send you to another listing for one of its properties (read it there, not on this page): when the shopper\'s requirements name that property, find and open that other listing before ordering this one, and judge the property by what it says.\n- A requirement may be conditional on what another requested item turns out to be: fix that other item first, then apply it.\n- If the page of a listing you intend to order directs you to some other page about it, read that page first.\n- If one item has to work with another item or with something the shopper owns, pick the pair together using what the pages themselves say (lists of supported models, a series or model code that both carry). When no partner that meets its own needs suits your main choice, swap the main choice for another acceptable one before concluding nothing fits.\n- Values the request rules out and fixed sets of allowed values are binding; preferences only settle ties.\n\nPICKING AMONG FITS\n- Lowest price or total wanted: the order needs the cheapest of all listings that satisfy the item. Rows rarely arrive sorted by price, so collect every fitting candidate across pages, go through them from cheap to dear and stop at the first full fit. A maximum-price search argument, where offered, lets you look only below your current choice.\n- Largest or smallest of some measure wanted: compare that measure over all fitting listings and take the top or bottom one; listings silent on the measure are out.\n- No such rule: the first listing that satisfies everything will do; spend no steps looking for a better one.\n- You may cart a listing as soon as it is confirmed; a line that sells out early can then still be replaced.\n\nNEWS DURING THE JOB\n- Budgets, requirements, prices and stock can change while you work. Such news comes in tool results, frequently as a refused checkout. A refusal does not end the job: take every change into account and order again. When only prices moved or a requirement was dropped, and the same lines still meet every remaining requirement and the money limit, order those same lines again; replace a line only where the news makes it fail.\n\nHANDING OVER\n- buy_now: one line per requested item (item number and listing id). Use it when each item has a listing checked against all its needs and, if the lowest price is wanted, you have seen nothing cheaper that also fits. The program then fills the cart, rereads the listings and checks out; if it holds back it explains why, so repair that and call buy_now again.\n- nothing_fits: only after honest searching shows an item with no acceptable listing. Running low on steps is not evidence of that; then order the best lines found so far. Put in "remark" what the request asks to be told in that case, briefly, mentioning a listing only if the request wants one.\n- Every reply makes at least one tool call, with a few words of reasoning: what you found and what you do next. Calls that do not wait on each other\'s results can share a reply (a search for each item, or several listings to open); buy_now and nothing_fits always go alone.'
CONTROL = [{'type': 'function', 'function': {'name': 'buy_now', 'description': 'Order these listings: one line per requested item. The program fills the cart, checks the lines and checks out, or says why it holds the order.', 'parameters': {'type': 'object', 'properties': {'lines': {'type': 'array', 'items': {'type': 'object', 'properties': {'item': {'type': 'integer', 'description': 'requested item number, counted from 1'}, 'id': {'type': 'string', 'description': 'listing id exactly as the store shows it'}}, 'required': ['item', 'id']}}}, 'required': ['lines']}}}, {'type': 'function', 'function': {'name': 'nothing_fits', 'description': 'End with no order, because for at least one requested item the store has nothing acceptable.', 'parameters': {'type': 'object', 'properties': {'reason': {'type': 'string', 'description': 'which item has no fitting listing, and why'}, 'remark': {'type': 'string', 'description': "what to tell the shopper, following the request's own wording for this ending"}}, 'required': ['reason']}}}]
PICK_ASK = 'A shopping job must end now and you make its final choice. NEEDS gives every requested item (numbered from 1) with all requirements in force, whether stated in the request, told by the shopper or read along reference pages, plus the money limit and the pick rule. LISTINGS holds what the store returned: OPENED marks a full page, ROW only a search row, so a property missing from a ROW is unknown rather than failed.\n' + D_READING + '\nChoose one listing of the right kind for each item:\n- Best is a listing that satisfies all of the item\'s requirements, an OPENED one ahead of a ROW.\n- A listing that states the wanted value under the requirement\'s own key ranks above one that meets it only through another key or its title.\n- If none is certain, prefer one that breaks no requirement, where only unknowns remain, and the fewer unknowns the better. A value a reference chain has not resolved yet stays an unknown.\n- With a lowest-price rule take the cheapest among the listings on that same footing; with a largest-or-smallest rule take the extreme of that measure; otherwise any of them.\n- The lines together must stay within the money limit, and a cap on one unit\'s price applies to its item.\nGive no lines at all only if, for some item, every listing of its kind here plainly breaks a requirement, is sold out or cannot fit the money limit, or no listing of its kind appears.\nSet "certain" to true only when every chosen line is OPENED and its record shows each of its item\'s requirements met, with no unknown left; otherwise false.\nAnswer with a single JSON object: {"lines": [{"item": <item number>, "id": "<listing id>"}], "certain": true or false, "why": "<short>"}'
DECLINE_REMARK = 'None of the listings I found matches everything you asked for, so no order was placed.'
PUSH = "Not ended (no step used). Keep looking first: {left} steps are still left, so search in ways you have not tried: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, search every item again with other wording (a shorter query, the product kind alone, the wanted value first), and open the listings that might fit. End only when some item clearly has nothing that meets its needs."
_EX_COMPARE = COMPARE_WORDS
HISTORY_CHARS = 110000
SHOW_CUT = 400
SHOW_LIMIT = 12000
PUSHES = 3
TURNS = 70
REPAIR_TURNS = 10
CALL_RESERVE = 10
NO_STEP_HOLDS = 4
FINISH_TRIES = 6
RESUBMITS = 1
PICK_CHARS = 8000
TRIM_HALF = True
WIDEN_HEAD = 4
WIDEN_EACH = 6
WIDEN_EACH_MANY = 4
WIDEN_ROUND = 5
WIDEN_FIRST = 8
WIDEN_ALL = 10
WIDEN_ITEM_CAP = 8
WIDEN_KEEP = 22
WIDEN_KEEP_MORE = 30
WIDEN_COMPARE = 4
AUTO_PAGES = 6
PICK_HEAVY = True
VALUE_CALLS = 8
VALUE_ROOM = 22
VALUE_OPENS = 3
VALUE_KEEP = 14
VALUE_OPENS_CHEAP = 5
VALUE_CHEAP_EXTRA = 2
WIDER_ROOM = 14
LATE_ROOM = 5
CHAIN_ROOM = 3
WIDER_SEARCH = False
RESERVE_LATE = True
LATE_STEPS = 8
LIST_BELOW = True
LIST_BELOW_OPENS = 2
LIST_BELOW_DEEP = True
CLOSE_KEEP_CART = True
SWAP_LIGHT = True
DECLINE_EVIDENCE = True
DECLINE_ON_BUY = True
BROKEN_HOLDS = 2
VALUE_SHELF = True
STOCK_REFUSAL = True
ASK_AT_CLOSE = True

def _ex_int(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    return int(number) if number.is_integer() else None

def _ex_refused(obs):
    return isinstance(obs, dict) and bool(obs.get('error'))

def _ex_singular(word):
    if word.endswith('ies') and len(word) > 3:
        return word[:-3] + 'y'
    if word.endswith(('ches', 'shes', 'xes', 'sses')) and len(word) > 4:
        return word[:-2]
    if word.endswith('s') and (not word.endswith('ss')) and (len(word) > 3):
        return word[:-1]
    return word

def _ex_clip(node, cut):
    if isinstance(node, dict):
        return {k: _ex_clip(v, cut) for k, v in node.items()}
    if isinstance(node, list):
        return [_ex_clip(v, cut) for v in node]
    if isinstance(node, str) and len(node) > cut:
        return node[:cut] + '...'
    return node

def _ex_show(body):
    try:
        text = json.dumps(_ex_clip(body, SHOW_CUT), ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(body)
    return text if len(text) <= SHOW_LIMIT else text[:SHOW_LIMIT] + '...(cut)'

class Explorer:

    def __init__(self, problem_data, parts=None):
        parts = parts or {}
        env_in = (problem_data or {}).get('environment') or {}
        pv = env_in.get('policy_view') or {}
        self.query = str(pv.get('query') or '')
        self.opening = pv.get('opening')
        given = lambda key, build: parts[key] if parts.get(key) is not None else build()
        self.env = given('env', lambda: Env(env_in.get('binding') or {}, _ex_int(pv.get('max_steps')) or 30))
        self.llm = given('llm', Llm)
        self.tm = given('toolmap', lambda: self._toolmap(pv.get('tools') or []))
        self.led = given('ledger', self._ledger)
        self.iv = given('interview', lambda: Interview(self.led, self.llm, self.tm, self.env, reserve=self.message_reserve()))
        self.chain = given('chain', lambda: ChainTracker(self.led, self.llm, self.tm, self.env))
        self.gate = given('gate', lambda: CheckoutGate(self.led, self.llm, self.tm, self.env, self.chain))
        self.noted = 0
        self.changes = []
        self.views = {}
        self.view_keys = {}
        self.bare = set()
        self.compare_tables = None
        self.compare_ok = 0
        self.compare_bad = 0
        if self.opening is not None:
            self.absorb('', {}, self.opening)
        self.interviewed = False
        self.widen_stuck = False
        self.pushes = 0
        self.push_step = -1
        self.pages_read = 0
        self.lead_holds = 0
        self.no_step_holds = 0
        self.view_holds = 0
        self.wider_held = False
        self.late_held = False
        self.late_asked = False
        self.capped = set()
        self.pick_certain = False
        self.pick_none = False
        self.said_none = False
        self.close_uncertain = False
        self.broken_holds = 0
        self.soft_refused = set()
        self.finish_tries = 0
        self.planned = []
        self.passed = []
        self.refused = {}
        self.sold_sigs = set()
        self.remark = ''
        self.sent = set()
        self.read_texts = {}
        self.blind_calls = []
        self.unfound_tried = False
        self.lead_ordered = False
        self.chain_searched = set()
        self.deferred = []
        self.status_at = 0
        hidden = {'message', 'finish_order', 'finish_decline'}
        store = [t for t in self.tm.tools if self.tm.role.get((t.get('function') or t).get('name')) not in hidden]
        self.store_tools = {(t.get('function') or t).get('name') for t in store}
        self.offered = store + CONTROL
        first = 'What the request needs:\n' + self.led.summary(2500)
        if self.opening is not None:
            first += '\n\nWhat the store shows first:\n' + _ex_show(unwrap(self.opening)[0])
        first += '\n\nBegin with a search for every item by its kind (one reply can hold them all), then open a few of the listings in full, so that the properties this store records for them are known.'
        self.history = [{'role': 'system', 'content': GUIDE + "\n\nTHE SHOPPER'S REQUEST:\n" + self.query}, {'role': 'user', 'content': first}]

    def _toolmap(self, tools):
        try:
            return ToolMap(tools, self.llm)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return ToolMap(tools, None)

    def _ledger(self):
        try:
            return read_request(self.query, self.llm, self.tm)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            led = Ledger()
            led.query = self.query
            return led

    def message_reserve(self):
        if not RESERVE_LATE:
            return 0
        most = step_messages_most(getattr(self.led, 'process', None), ('order', 'end', 'any'))
        limit = getattr(self.led, 'msg_budget', None)
        limit = limit if isinstance(limit, int) and (not isinstance(limit, bool)) and (limit >= 0) else 3
        return min(most, limit)

    def absorb(self, name, args, obs):
        try:
            self.tm.learn(name, args, obs)
        except Exception:
            traceback.print_exc(file=sys.stderr)
        try:
            self.led.note_obs(self.tm, name, args, obs)
        except Exception:
            traceback.print_exc(file=sys.stderr)
        try:
            self.note_view(name, obs)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    def note_view(self, name, obs):
        if self.tm.role.get(name) not in ('detail', 'compare'):
            return
        body = unwrap(obs)[0]
        records = [r for r in self.tm.rows(body) or [] if isinstance(r, dict)]
        if not records and isinstance(body, dict) and self.tm.row_id(body):
            records = [body]
        key = (getattr(self.tm, 'keys', None) or {}).get('props')
        if self.tm.role.get(name) == 'compare' and records and key:
            self.compare_tables = bool(self.compare_tables) or any((isinstance(r.get(key), dict) for r in records))
        version = len(self.led.notices)
        for record in records:
            ident = self.tm.row_id(record)
            if not ident:
                continue
            keys = set(record)
            self.view_keys.setdefault(name, set()).update(keys)
            if getattr(self.tm, 'shows_props', {}).get(name) is False:
                if str(ident) not in self.views:
                    self.bare.add(str(ident))
            else:
                self.bare.discard(str(ident))
            at, have = self.views.get(str(ident), (-1, set()))
            self.views[str(ident)] = (version, (have if at == version else set()) | keys)

    def shown_already(self, name, ident):
        at, have = self.views.get(str(ident), (-1, set()))
        keys = self.view_keys.get(name)
        return bool(keys) and at == len(self.led.notices) and (keys <= have) and (ident not in self.led.gone)

    def record_text(self, ident):
        return _ex_show(self.led.seen.get(ident))[:3000]

    def has_table(self, ident):
        key = (getattr(self.tm, 'keys', None) or {}).get('props')
        record = self.led.seen.get(ident)
        if key and isinstance(record, dict):
            return isinstance(record.get(key), dict)
        return ident in self.led.opened and ident not in self.bare

    def compare_shows(self, tool):
        known = getattr(self.tm, 'shows_props', {}).get(tool)
        return known if known is not None else self.compare_tables

    def compare_room(self, tool):
        props = ((getattr(self.tm, 'schemas', None) or {}).get(tool) or {}).get('properties') or {}
        limit = next((v.get('maxItems') for v in props.values() if isinstance(v, dict) and isinstance(v.get('maxItems'), int)), None)
        if limit:
            return limit
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(WIDEN_COMPARE, self.compare_ok)

    def note_compare(self, name, args, obs):
        asked = next((v for v in args.values() if isinstance(v, list)), None)
        if self.tm.role.get(name) != 'compare' or not isinstance(asked, list) or len(asked) < 2:
            return ''
        shown = json.dumps(unwrap(obs)[0], ensure_ascii=False)
        if not any((str(i) in shown for i in asked)):
            self.compare_bad = min(self.compare_bad or len(asked), len(asked))
            return f'\n(The side-by-side view showed nothing for {len(asked)} listings: send at most {self.compare_room(name)} per call.)'
        self.compare_ok = max(self.compare_ok, len(asked))
        return ''

    def heed(self):
        new = self.led.notices[self.noted:]
        self.noted = len(self.led.notices)
        if not new or self.llm.closed():
            return
        try:
            self.changes += read_notes(self.led, self.llm, new, stock=STOCK_REFUSAL)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    def resolve_sold(self, lines=None):
        if not STOCK_REFUSAL or not getattr(self.led, 'sold_unnamed', False):
            return []
        self.led.sold_unnamed = False
        lines = [l for l in lines or self.cart_lines() if l.get('id')]
        if not lines:
            return []
        try:
            sold = self.gate.find_sold(lines)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            sold = []
        mark = getattr(self.led, 'mark_sold', None)
        if callable(mark):
            mark(sold)
        else:
            self.led.gone.update(sold)
        if sold:
            self.changes.append(f"{', '.join(sold)} can no longer be bought (the store's notice named no listing; read as the cart's {('line' if len(lines) == 1 else 'sold-out line')})")
        return sold

    def sold_hold(self, lines, key, refusal):
        self.resolve_sold(lines)
        gone = [l['id'] for l in lines if l['id'] in self.led.gone]
        if not gone:
            return None
        self.sold_sigs.add(key)
        return refusal + f"\nSold out now: {', '.join(gone)}. The same lines will not be ordered again: replace " + ('that line' if len(gone) == 1 else 'those lines') + ' with other listings that meet every requirement of the item. ' + self.swap_hint(lines, gone)

    def flush(self):
        if self.changes:
            self.say('Changes now applied to the needs: ' + '; '.join(self.changes) + '.\nNeeds and state now:\n' + self.led.summary(2000))
            self.changes = []

    def act(self, name, args):
        if not name or self.done() or self.tm.validate(name, args):
            return None
        obs = self.env.call(name, args)
        self.absorb(name, args, obs)
        return obs

    def blocked(self, name, args):
        fault = self.tm.validate(name, args)
        if fault:
            return fault
        if self.tm.role.get(name) == 'cart_view':
            return self.cart_unchanged()
        if self.tm.role.get(name) == 'cart_set':
            return None
        early = self.page_before_first(name, args) or self.page_after_short(name, args)
        if early:
            return early
        ident = self.page_id(name, args)
        odd = self.unlike_ids(name, args)
        if odd:
            return f'{json.dumps(odd)} is not a listing id: every listing seen so far has an id like {json.dumps(next(iter(self.led.seen)))}; use an id exactly as a result row shows it.'
        if ident and self.shown_already(name, ident):
            return f'{ident} has shown everything this view gives since the last change. Its record: ' + self.record_text(ident)
        key = self.mark(name, args)
        if key in self.sent:
            step, text = self.read_texts.get(key, (None, ''))
            if text:
                return f'the same call at step {step} returned this, and nothing has changed since:\n' + text[:3000]
            return 'this exact call was already made and nothing has changed since; use its earlier result or change the arguments.'
        self.sent.add(key)
        return None

    def page_before_first(self, name, args):
        if self.tm.role.get(name) != 'search' or not hasattr(self.tm, '_page_arg'):
            return ''
        param = self.tm._page_arg(name)
        if not param or (args or {}).get(param) in (None, '', 0, 1, '0', '1'):
            return ''
        base = {k: v for k, v in args.items() if k != param}
        if self.mark(name, base) in self.sent or any((h.get('name') == name and {k: v for k, v in (h.get('args') or {}).items() if k != param} == base for h in self.env.history)):
            return ''
        return f'the first page of this search has not been read, so {param}={json.dumps(args.get(param))} was not given for it: search without {param} first, then page on with the value the result returns.'

    def page_after_short(self, name, args):
        if self.tm.role.get(name) != 'search' or not hasattr(self.tm, '_page_arg'):
            return ''
        param = self.tm._page_arg(name)
        if not param or (args or {}).get(param) in (None, '', 0, 1, '0', '1'):
            return ''
        base = {k: v for k, v in args.items() if k != param}
        last, full = (None, 0)
        for h in self.env.history:
            if h.get('name') != name:
                continue
            try:
                shown = len(self.tm.rows(unwrap(h.get('obs'))[0]) or [])
            except Exception:
                continue
            full = max(full, shown)
            if {k: v for k, v in (h.get('args') or {}).items() if k != param} == base:
                last = shown
        if last is None or full < 2 or last >= full:
            return ''
        return f'the latest page of this search showed {last} rows, fewer than a full page ({full}): it has no further page. Search with other words instead.'

    def cart_unchanged(self):
        last_view = last_set = -1
        for k, h in enumerate(self.env.history):
            role = self.tm.role.get(h.get('name'))
            if role == 'cart_view':
                last_view = k
            elif role == 'cart_set':
                last_set = k
        if last_view >= 0 and last_view > last_set:
            return 'the cart has not changed since its last view; use that result.'
        return None

    def unlike_ids(self, name, args):
        if self.tm.role.get(name) != 'detail' or len(self.led.seen) < 3:
            return None
        ids = [str(i) for i in self.led.seen]
        head = ids[0]
        for i in ids[1:]:
            while head and (not i.startswith(head)):
                head = head[:-1]
        if len(head) < 2:
            return None
        try:
            probe = self.tm.args_for('detail', product_id='\x00probe')
        except Exception:
            return None
        keys = [k for k, v in (probe or {}).items() if v == '\x00probe']
        for k in keys:
            v = (args or {}).get(k)
            if isinstance(v, str) and v and (v not in self.led.seen) and (not v.startswith(head)):
                return v
        return None

    def page_id(self, name, args):
        if self.tm.role.get(name) != 'detail':
            return None
        return next((v for v in args.values() if isinstance(v, str) and v in self.led.seen), None)

    def trim_compare(self, name, args):
        key = next((k for k, v in args.items() if isinstance(v, list) and v), None)
        if self.tm.role.get(name) != 'compare' or key is None:
            return (args, '', '')
        asked = [str(i) for i in args[key]]
        room, cut = (self.compare_room(name), '')
        if self.compare_bad and len(asked) > room:
            cut = f"\n(Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(asked[room:])}.)"
            asked = asked[:room]
            args = dict(args, **{key: asked})
        keep = [i for i in asked if not self.shown_already(name, i)]
        if not keep:
            return (args, '', 'these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.record_text(i)[:1200]}' for i in asked))[:4000])
        if len(keep) == len(asked) or self.tm.validate(name, dict(args, **{key: keep})):
            return (args, cut, '')
        left_out = [i for i in asked if i not in keep]
        return (dict(args, **{key: keep}), cut + f"\n(Left out {', '.join(left_out)}: shown in full already, nothing changed since.)", '')

    def mark(self, name, args):
        return json.dumps([name, args, len(self.led.notices)], sort_keys=True)

    def done(self):
        return self.env.finished or self.env.out_of_key

    def left(self):
        cap = min(self.env.max_steps, self.led.step_budget or self.env.max_steps)
        return max(0, cap - self.env.step)

    def count(self):
        return max(1, len(self.led.items))

    def tail(self):
        return 2 * self.count() + 3

    def floor(self):
        return self.tail() + self.count() + 2

    def chain_first(self):
        lead = getattr(self.chain, 'leading', None)
        if not callable(lead):
            return False
        try:
            return bool(lead())
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return False

    def chain_floor(self, floor):
        return min(floor, self.tail() + CHAIN_ROOM) if self.chain_first() else floor

    def after_chain(self):
        if self.chain_first():
            return ''
        found = self.chain_values()
        found += [n for n in self.deferred if n not in found]
        self.deferred = []
        if not found:
            return ''
        try:
            return self.value_search(found)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return ''

    def searches(self):
        return sum((1 for h in self.env.history if self.tm.role.get(h.get('name')) == 'search'))

    def status(self):
        left, floor = (self.left(), self.floor())
        text = f'Steps: {self.env.step} used, {left} left.'
        ordered = bool(self.refused) or bool(self.planned)
        if left <= floor + 1:
            text += ' FINAL STEPS: call buy_now now with a line for each item; call nothing_fits only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= floor + 4 or (left <= floor + self.count() + 5 and (not ordered)):
            text += ' Few steps left: settle on one listing per item and call buy_now now.'
        elif not ordered and left <= floor + 10:
            text += ' Steps are running low: settle on your picks and call buy_now within the next few replies.'
        elif not ordered and (not self.led.cart) and (self.env.step >= self.env.max_steps // 2):
            text += " Half the steps are used and nothing is ordered: call buy_now with each item's fitting listing now (the store may still sell one out)."
        try:
            lookups = self.chain.missing() if self.chain.pending() else []
        except Exception:
            lookups = []
        if lookups and (not ordered):
            text += '\nResolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value: ' + ' | '.join((str(x) for x in lookups[:3]))
        return text

    def say(self, text):
        self.history.append({'role': 'user', 'content': text})

    def trim(self):
        size = sum((len(json.dumps(m, ensure_ascii=False)) for m in self.history))
        if TRIM_HALF:
            if size <= HISTORY_CHARS:
                return
            older = [k for k, m in enumerate(self.history) if m.get('role') == 'tool']
            older = set(older[:max(1, len(older) // 2)])
            size = 0
        for k, m in enumerate(self.history[2:-6], 2):
            if TRIM_HALF and k not in older:
                continue
            if size <= HISTORY_CHARS and (not TRIM_HALF):
                break
            text = m.get('content')
            if m.get('role') == 'tool' and isinstance(text, str) and (len(text) > 600) and ('...(shortened;' not in text):
                ids = [i for i in self.led.order if i in text][:20]
                short = text[:300] + '...(shortened; listings in it: ' + (', '.join(ids) or 'none') + ')'
                size -= len(text) - len(short)
                m['content'] = short

    def qty_of(self, item, asked=None):
        if 1 <= item <= len(self.led.items):
            return max(1, int(self.led.items[item - 1].qty or 1))
        got = _ex_int(asked)
        return got if got and got > 0 else 1

    def lines_from(self, raw):
        n = len(self.led.items)
        chosen, loose = ({}, [])
        for ln in raw if isinstance(raw, list) else []:
            if not isinstance(ln, dict) or not str(ln.get('id') or '').strip():
                continue
            ident = str(ln['id']).strip()
            item = _ex_int(ln.get('item'))
            if item and (n == 0 or 1 <= item <= n) and (item not in chosen):
                chosen[item] = (ident, ln.get('qty'))
            else:
                loose.append((ident, ln.get('qty')))
        for ident, qty in loose:
            free = next((k for k in range(1, max(n, len(chosen) + len(loose)) + 1) if k not in chosen), None)
            if free is None or (n and free > n):
                break
            chosen[free] = (ident, qty)
        return [{'item': k, 'id': v[0], 'qty': self.qty_of(k, v[1])} for k, v in sorted(chosen.items())]

    @staticmethod
    def sig(lines):
        return json.dumps(sorted(((l['id'], l['qty']) for l in lines)))

    def cart_lines(self):
        return [{'id': i, 'qty': q} for i, q in self.led.cart.items() if q]

    def fill(self, lines):
        want = {l['id']: l['qty'] for l in lines}
        whole = self.gate.whole_cart(lines)
        if whole and self.led.cart != want and (self.left() > 1):
            self.act(*whole)
            return
        cart = self.tm.tool('cart_set')
        for extra in [i for i in list(self.led.cart) if i not in want]:
            self.act(cart, self.tm.args_for('cart_set', product_id=extra, qty=self.led.cart[extra], mode='remove'))
        for ident, qty in want.items():
            have = self.led.cart.get(ident, 0)
            if have == qty or self.left() <= 1:
                continue
            if have:
                self.act(cart, self.tm.args_for('cart_set', product_id=ident, qty=have, mode='remove'))
            self.act(cart, self.tm.args_for('cart_set', product_id=ident, qty=qty, mode='add'))

    def look(self):
        name = self.tm.tool('cart_view')
        if not name or self.left() <= 2:
            return False
        args = self.tm.args_for('cart_view')
        return self.act(name, args if args is not None else {}) is not None

    def remark_for(self, text):
        text = str(text or '').strip()
        if not text or any((ident in text for ident in self.led.seen)):
            return DECLINE_REMARK
        return text[:400]

    def finish(self, kind, remark=''):
        decline = kind != 'order'
        role = 'finish_decline' if decline else 'finish_order'
        name = self.tm.tool(role) or (self.tm.tool('finish_order') if decline else None)
        args = None
        if decline:
            args = self.tm.args_for(role, decline=True, text=self.remark_for(remark or self.remark))
        if args is None:
            args = self.tm.args_for(role, decline=decline)
        if not name or args is None or self.left() <= 0:
            return 'This store gives no way to ' + ('end without an order.' if decline else 'check out.')
        self.finish_tries += 1
        if self.finish_tries > FINISH_TRIES:
            return 'Finishing keeps failing.'
        obs = self.act(name, args)
        if self.env.finished:
            return None
        body, notes = unwrap(obs)
        return 'The store did not accept ' + ('ending without an order' if decline else 'the checkout') + ': ' + (' '.join(notes) if notes else _ex_show(body)[:400])

    def place(self, lines, final=False):
        if STOCK_REFUSAL and self.sig(lines) in self.sold_sigs:
            gone = [l['id'] for l in lines if l['id'] in self.led.gone]
            return 'Not ordered (no step used): the store already refused these same lines because ' + (', '.join(gone) if gone else 'a line') + ' sold out. Replace the sold-out line with another listing that meets every requirement of its item, then call buy_now again.'
        notices = len(self.led.notices)
        self.fill(lines)
        want = {l['id']: l['qty'] for l in lines}
        if self.look():
            gone = [i for i in want if i in self.led.gone]
            news = self.led.notices[notices:]
            if (gone or news) and (not final) and (self.view_holds < 2):
                self.view_holds += 1
                return 'Not ordered yet: the cart now shows a change. ' + (f"Sold out: {', '.join(gone)}. " if gone else '') + (f"Notices: {' '.join(news)} " if news else '') + self.swap_hint(lines, gone) + 'Apply it, then call buy_now again.'
            if self.led.cart != want and self.left() > 2:
                self.fill(lines)
        faults = self.gate.cart_faults(lines)
        if faults and self.left() > 2:
            self.fill(lines)
            faults = self.gate.cart_faults(lines)
        if faults and (not final) and (self.view_holds < 2):
            self.view_holds += 1
            return "Not ordered yet: the cart does not hold the order's lines (" + faults + '). Call buy_now again.'
        refusal = self.finish('order')
        if refusal is not None:
            key = self.sig(lines)
            self.refused[key] = self.refused.get(key, 0) + 1
            had = len(self.changes)
            if STOCK_REFUSAL:
                try:
                    self.heed()
                    held = self.sold_hold(lines, key, refusal)
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    held = None
                if held is not None:
                    return held
            if self.refused[key] == 1 and self.left() >= 1:
                before = {l['id']: self.tm.price(self.led.seen.get(l['id']) or {}) for l in lines}
                try:
                    self.heed()
                    news = self.changes[had:]
                    roomy = not final and self.left() > 2
                    if all(('no longer required' in c for c in news)):
                        self.soft_refused.add(key)
                        self.gate._refresh(lines)
                        problem = self.gate._shape(lines) or self.gate._stock_budget(lines)
                    elif roomy:
                        problem = self.gate.recheck(lines)
                    else:
                        problem = 'news to check the lines against, with no room left to check them'
                    moved = any((self.tm.price(self.led.seen.get(i) or {}) != p for i, p in before.items()))
                    if not problem and moved and (self.led.pick == 'cheapest') and roomy:
                        problem = 'a price moved under a lowest-price rule'
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    problem = 'recheck failed'
                if not problem and (not self.done()) and (not self.gate.cart_faults(lines)) and (self.left() >= 1):
                    again = self.finish('order')
                    if again is None:
                        return None
                    self.refused[key] += 1
                    return refusal + '\n' + again
        return refusal

    def swap_hint(self, lines, gone):
        lost = {l.get('item') for l in lines if l['id'] in gone} - {None}
        if not lost or self.left() <= self.tail() + 2:
            return ''
        if SWAP_LIGHT:
            picks = [l for l in self.guarded(lambda: self.pick(exclude=set(gone), heavy=False)) if l['item'] in lost]
            if not picks and getattr(self.llm, 'spent', 0.0) < getattr(self.llm, 'soft_cap', float('inf')):
                picks = [l for l in self.guarded(lambda: self.pick(exclude=set(gone))) if l['item'] in lost]
        else:
            picks = [l for l in self.guarded(lambda: self.pick(exclude=set(gone))) if l['item'] in lost]
        if not picks:
            return 'Search for another listing of that item if none already read fits. '
        return 'Listings already seen that appear to fit instead: ' + ', '.join((f"item {l['item']}: {l['id']}" for l in picks)) + '; check them and swap the sold-out line. '

    def buy(self, lines):
        unknown = [l['id'] for l in lines if l['id'] not in self.led.seen]
        if unknown and self.no_step_holds < NO_STEP_HOLDS:
            self.no_step_holds += 1
            return 'Not ordered (no step used): the store has not shown these ids: ' + ', '.join(unknown) + '. Use listing ids exactly as the results give them.'
        self.planned = lines
        self.heed()
        self.resolve_sold()
        wider = ''
        if WIDER_SEARCH:
            try:
                wider = self.wider_search(lines)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                wider = ''
        else:
            hint = self.wider_hold(lines)
            if hint:
                return hint
        below = ''
        if LIST_BELOW:
            try:
                below = self.list_below(lines)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                below = ''
        if hasattr(self.gate, 'chain_keep'):
            self.gate.chain_keep = self.floor() + LATE_STEPS
        ok, why = self.gate.ready_for(lines)
        if self.done():
            return None
        if not ok:
            return 'Not ordered yet: ' + why + ('\n' + wider if wider else '') + ('\n' + below if below else '')
        if RESERVE_LATE:
            try:
                late = self.late_question()
            except Exception:
                traceback.print_exc(file=sys.stderr)
                late = ''
            if self.done():
                return None
            if late:
                return late
        lines, _ = self.verify(lines)
        if self.done():
            return None
        if DECLINE_ON_BUY:
            held = self.broken_hold(lines)
            if held is None or held:
                return held
        self.passed = lines
        refusal = self.place(lines)
        return refusal if refusal is None else refusal + '\nApply every change in it, then call buy_now again.'

    def broken_evidence(self, lines):
        if not DECLINE_ON_BUY or not lines or (not hasattr(self.gate, 'broken_item')):
            return None
        try:
            return self.gate.broken_item(lines)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return None

    def broken_hold(self, lines):
        got = self.broken_evidence(lines)
        if not got:
            return ''
        self.broken_holds += 1
        print('BROKEN', json.dumps({'item': got[0], 'holds': self.broken_holds})[:300], file=sys.stderr)
        if self.broken_holds > BROKEN_HOLDS:
            return self.finish('decline', self.remark)
        return "Not ordered yet: the program's check finds that " + got[1] + f'. Search for other listings of item {got[0]} that state the required value and open them; if the store has none, call nothing_fits and say which requirement no listing meets.'

    def broken_stands(self):
        return bool(self.broken_holds and self.broken_evidence(self.planned))

    def wider_hold(self, lines):
        if self.wider_held or self.led.pick != 'cheapest' or self.left() < WIDER_ROOM + 3 * self.count():
            return ''
        self.wider_held = True
        priced = [f"{l['id']} at {p:g}" for l in lines for p in [self.tm.price(self.led.seen.get(l['id']) or {})] if p is not None]
        return f"Not ordered yet (no step used): {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a required value (often only in the properties) finds few of the fitting listings, and rows come in no price order. First run a few more searches for each item under other names for the product (its plain one-word name, other words for it, words from fitting titles you have seen) and look only at buyable rows priced below your lines" + (f" ({'; '.join(priced)})" if priced else '') + '; open the cheapest such rows, swap in any that fits, then call buy_now again.'

    def wider_search(self, lines):
        if self.wider_held or self.led.pick not in ('cheapest', 'best') or self.left() < WIDER_ROOM + 3 * self.count():
            return ''
        self.wider_held = True
        tool = self.tm.tool('search')
        if not tool:
            return ''
        cheapest = self.led.pick == 'cheapest'
        budget, said = (VALUE_CALLS + (VALUE_CHEAP_EXTRA if cheapest else 0), [])
        for l in lines:
            item = (l.get('item') or 1) - 1
            if not 0 <= item < self.count() or budget <= 2:
                continue
            noun = self.title_noun(item)
            values, _ = self.value_words(item, [])
            values = [v for v in values if not set(re.findall('[a-z0-9]+', v.lower())) <= set(noun.lower().split())]
            head = noun.split()[-1] if noun else ''
            tries = []
            for words in ([values[0], noun] if values and noun else [], [noun] if noun else [], [head] if head and head != noun else []):
                if words and words not in tries:
                    tries.append(words)
            price = self.tm.price(self.led.seen.get(l['id']) or {})
            below = price if cheapest and price is not None else None
            budget = self.search_tries(tool, item, tries, budget, said, below=below, skip_sent=True)
        if not said:
            return ''
        return 'Before ordering, the program searched wider for the pick rule and read the fitting rows (a step a call):\n' + '\n'.join(said)

    def late_question(self):
        if self.late_asked or self.llm.closed():
            return ''
        if step_messages_most(getattr(self.led, 'process', None), ('order', 'any')) > 0:
            return ''
        if self.left() < LATE_STEPS or self.left() <= self.tail() + 2:
            return ''
        ask = getattr(self.iv, 'late_ask', None)
        if not callable(ask):
            return ''
        self.late_asked = True
        told, heard, had = (len(self.led.revealed), len(getattr(self.iv, 'transcript', None) or []), len(self.changes))
        if not ask():
            return ''
        self.heed()
        fresh = list(self.led.revealed[told:])
        news = self.changes[had:]
        del self.changes[had:]
        if not fresh and (not news):
            return ''
        talk = (getattr(self.iv, 'transcript', None) or [])[heard:]
        question, answer = (talk[-1][1], talk[-1][2]) if talk else ('', '')
        text = f'Not ordered yet: the program asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join([n.text for n in fresh] + news) + '. The order was not placed: check each line against them (search again with the new values where needed), fix the lines, then call buy_now again.'
        if any((n.kind == 'ref' for n in fresh)):
            text += ' Some take their value from another listing: find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        values = [n for n in fresh if n.kind == 'hidden' and n.value]
        held = bool(values) and self.chain_first()
        if held:
            self.deferred += [n for n in values if n not in self.deferred]
        try:
            searched = self.value_search(values) if values and (not held) else ''
        except Exception:
            traceback.print_exc(file=sys.stderr)
            searched = ''
        return text + ('\n' + searched if searched else '') + '\nNeeds and state now:\n' + self.led.summary(2000)

    def cap_tool(self):
        slots = getattr(self.tm, 'slots', None) or {}
        return next((n for n, s in slots.items() if self.tm.role.get(n) == 'search' and isinstance(s, dict) and s.get('category') and s.get('max_price')), None)

    def below_args(self, tool, label, cap):
        slots = (getattr(self.tm, 'slots', None) or {}).get(tool) or {}
        cat, top, query = (slots.get('category'), slots.get('max_price'), slots.get('query'))
        page = self.tm._page_arg(tool) if hasattr(self.tm, '_page_arg') else None
        args = None
        for h in reversed(self.env.history):
            old = h.get('args') or {}
            if h.get('name') == tool and old.get(cat) and (not str(old.get(query) or '').strip()) and (not _ex_refused(h.get('obs'))):
                args = dict({k: v for k, v in old.items() if k not in (cat, top, page, query)}, **{cat: label, top: cap})
                break
        if args is None or self.tm.validate(tool, args):
            args = self.tm.args_for('search', category=label, max_price=cap, tool=tool)
        return self.shelf_label(tool, args) if args else None

    def list_below(self, lines):
        if self.led.pick != 'cheapest':
            return ''
        tool = self.cap_tool()
        if not tool:
            return ''
        said = []
        for l in lines:
            rec = self.led.seen.get(l['id']) or {}
            price, label = (self.tm.price(rec), str(self.tm.category(rec) or '').strip())
            if price is None or price < 2 or (not label) or ((l['id'], price) in self.capped):
                continue
            if self.left() <= self.floor() + 2 or self.done():
                break
            self.capped.add((l['id'], price))
            cap = int(price) - 1 if float(price).is_integer() else int(price)
            args = self.below_args(tool, label, cap)
            if not args or self.blocked(tool, args):
                continue
            start = len(self.env.history)
            obs = self.act(tool, args)
            if obs is None:
                break
            self.read_texts[self.mark(tool, args)] = (self.env.step, _ex_show(unwrap(obs)[0]))
            try:
                if LIST_BELOW_DEEP:
                    self.deep_pages(tool, args, obs)
                else:
                    self.more_pages(tool, args, obs)
            except Exception:
                traceback.print_exc(file=sys.stderr)
            rows, cost = ([], {})
            for h in self.env.history[start:]:
                for r in self.tm.rows(h.get('obs')) or []:
                    ident = str(self.tm.row_id(r) or '')
                    if not ident or ident in rows or ident in self.led.gone or (self.tm.in_stock(r) is False) or (str(self.tm.category(r) or '').strip().lower() != label.lower()):
                        continue
                    p = self.tm.price(r)
                    if p is not None and p < price:
                        rows.append(ident)
                        cost[ident] = p
            rows.sort(key=lambda i: cost[i])
            said.append(f"item {l.get('item') or 1}, {label} below {cap:g}: " + (', '.join(rows[:6]) or 'no cheaper row'))
            pages = self.read_pages([i for i in rows if not self.has_table(i)][:LIST_BELOW_OPENS], self.floor() + 2)
            if pages:
                said.append('opened: ' + ' | '.join(pages)[:2400])
        if not said:
            return ''
        return "Before ordering, the program listed each line's own shelf below the line's price and read the cheapest rows (a step a call):\n" + '\n'.join(said)

    def deep_pages(self, name, args, obs, most=None):
        page_of = getattr(self.tm, '_page_arg', None)
        param = page_of(name) if callable(page_of) else None
        if not param or not callable(getattr(self.tm, 'next_page', None)):
            return []
        rows = self.tm.rows(obs) or []
        prices = [self.tm.price(r) for r in rows]
        if self.led.pick == 'cheapest' and len(rows) > 1 and (None not in prices) and (prices == sorted(prices)):
            return []
        value, texts = (self.tm.next_page(name, obs), [])
        limit = AUTO_PAGES if most is None else min(AUTO_PAGES, most)
        while value is not None and len(texts) < limit and (self.left() > self.floor() + LATE_STEPS):
            call = dict(args)
            call[param] = value
            if self.tm.validate(name, call) or self.done():
                break
            got = self.act(name, call)
            if got is None or self.done():
                break
            texts.append(f'{param}={json.dumps(value)}: ' + _ex_show(unwrap(got)[0]))
            value = self.tm.next_page(name, got)
        return texts

    def verify(self, lines, forced=False):
        if not lines or self.done():
            return (lines, False)
        keep = self.tail() if forced else self.floor()
        try:
            got, decline = self.gate.final_check(lines, room=max(0, self.left() - keep), forced=forced, judged_none=self.said_none or self.pick_none, uncertain=forced and self.close_uncertain)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return (lines, False)
        if len(got) != len(lines):
            return (lines, False)
        return ([dict(l, id=g['id']) for l, g in zip(lines, got)], bool(decline))

    def records(self, exclude=(), limit=PICK_CHARS):
        pool = {i: r for i, r in self.led.seen.items() if i not in self.led.gone and i not in exclude and isinstance(r, dict)}
        needs_of = getattr(self.led, 'needs_of', None)
        ranked = []
        for n, item in enumerate(self.led.items):
            needs = needs_of(n) if callable(needs_of) else item.needs
            read = [str(r.value) for r in getattr(self.led, 'refs', None) or [] if getattr(r, 'item', None) == n and getattr(r, 'done', False) and getattr(r, 'value', None)]
            said = [x.text if not _EX_COMPARE.search(str(getattr(x, 'value', '') or x.text.split(':', 1)[-1])) else str(getattr(x, 'prop', '') or x.text.split(':', 1)[0]) for x in needs]
            words = ' '.join([item.name, getattr(item, 'shelf', '') or ''] + said + read)
            got = _d_rank(words, pool, limit=25)
            ranked.append([i for i in got if self.has_table(i)] + [i for i in got if not self.has_table(i)])
        order = []
        for k in range(max([len(r) for r in ranked] or [0])):
            order += [r[k] for r in ranked if k < len(r) and r[k] not in order]
        order += [i for i in reversed(self.led.order) if i in pool and i not in order]
        out, size = ([], 0)
        for ident in order:
            line = f"{('OPENED' if self.has_table(ident) else 'ROW')} {ident}: {_d_brief(pool[ident], 140)}"
            if size + len(line) > limit:
                continue
            out.append(line)
            size += len(line) + 1
        return '\n'.join(out)

    def pick(self, exclude=(), heavy=None):
        self.pick_certain = False
        if self.llm.closed() or not self.led.seen:
            return []
        recs = self.records(exclude)
        if not recs:
            return []
        values = self.chain.values_text() if hasattr(self.chain, 'values_text') else ''
        noted = self.chain.noted_text() if hasattr(self.chain, 'noted_text') else ''
        values = '\n'.join((t for t in (values, noted) if t))
        user = f'REQUEST:\n{self.query}\n\nNEEDS:\n{self.led.summary(2500)}' + (f'\nValues read along reference pages:\n{values}' if values else '') + f'\n\nLISTINGS:\n{recs}'
        got = self.llm.ask(PICK_ASK, user, heavy=PICK_HEAVY if heavy is None else heavy, max_tokens=2500)
        raw = got.get('lines') if isinstance(got, dict) else got if isinstance(got, list) else None
        if raw is None and isinstance(got, dict) and got.get('id'):
            raw = [got]
        print('PICK', json.dumps(got)[:600], file=sys.stderr)
        if not exclude:
            self.pick_none = isinstance(raw, list) and (not raw)
        keep = [l for l in raw if isinstance(l, dict) and str(l.get('id') or '') in self.led.seen and (str(l.get('id')) not in self.led.gone) and (str(l.get('id')) not in exclude)] if isinstance(raw, list) else []
        lines = self.lines_from(keep)
        if len(lines) < self.count():
            return []
        self.pick_certain = isinstance(got, dict) and got.get('certain') is True and all((self.has_table(l['id']) for l in lines))
        return lines

    def leads(self):
        return self.lines_from([{'item': n, 'id': l['id']} for n, l in enumerate(self.gate.fallback_lines() or [], 1)])

    def unfound_items(self):
        if not self.led.seen:
            return []
        owners = set()
        for rec in self.led.seen.values():
            try:
                texts = [self.tm.title(rec) or '', self.tm.category(rec) or '']
                owners.add(self.iv.owner_of(texts, strict=True) if self.count() == 1 else self.iv.owner_of(texts))
            except TypeError:
                if self.count() == 1:
                    return []
                owners.add(self.iv.owner_of(texts))
            except Exception:
                return []
        return [i for i in range(self.count()) if i not in owners]

    def ready_to_ask(self):
        try:
            tables = self.shelf_tables() if SHELF_KEYS else self.iv.catalogue()[1]
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return True
        if sum((tables.get(i, 0) for i in range(self.count()))) >= 3 * self.count() + 1:
            return True
        return self.env.step >= max(8, self.env.max_steps // 4)

    def label_fits(self, label, item):
        words = lambda t: ' '.join(re.findall('[a-z0-9]+', str(t or '').lower()))
        stems = lambda t: {_ex_singular(w) for w in words(t).split() if len(w) >= 4}
        own = words(label)
        shelf = getattr(item, 'shelf', '') or ''
        if not own:
            return False
        return own in (words(shelf), words(item.name)) or bool(stems(label) & (stems(shelf) | stems(item.name)))

    def shelf_owners(self):
        category = getattr(self.tm, 'category', None)
        title = getattr(self.tm, 'title', None)
        out, labels, matched = ({}, {}, False)
        for ident, rec in self.led.seen.items():
            try:
                label = str(category(rec) or '') if callable(category) else ''
            except Exception:
                label = ''
            labels[ident] = label
            if label and any((self.label_fits(label, item) for item in self.led.items)):
                matched = True
        for ident, rec in self.led.seen.items():
            try:
                texts = [str(title(rec) or '') if callable(title) else '', labels[ident]]
            except Exception:
                texts = ['', labels[ident]]
            label = labels[ident]
            if not matched:
                out[ident] = self.iv.owner_of(texts)
                continue
            hits = [i for i, item in enumerate(self.led.items) if label and self.label_fits(label, item)]
            if len(hits) == 1:
                out[ident] = hits[0]
                continue
            if label and (not hits):
                out[ident] = -1
                continue
            try:
                owner = self.iv.owner_of(texts, strict=True)
            except TypeError:
                owner = self.iv.owner_of(texts)
            out[ident] = owner if owner >= 0 and (not hits or owner in hits) else hits[0] if hits else -1
        return out

    def shelf_tables(self, owners=None):
        owners = self.shelf_owners() if owners is None else owners
        out = {}
        for ident, owner in owners.items():
            if owner is not None and owner >= 0 and self.has_table(ident):
                out[owner] = out.get(owner, 0) + 1
        return out

    def wanted_words(self, item):
        needs_of = getattr(self.led, 'needs_of', None)
        needs = needs_of(item) if callable(needs_of) else self.led.items[item].needs
        return {w for x in needs for w in re.findall('[a-z0-9]{3,}', str(x.text).split(':', 1)[-1].lower())}

    def widen_lanes(self, sizes):
        each = WIDEN_EACH if self.count() <= 2 else WIDEN_EACH_MANY
        lanes, heads = ([], [])
        if SHELF_KEYS:
            owners = self.shelf_owners()
            tables = self.shelf_tables(owners)
            owner = lambda ident, rec: owners.get(ident)
        else:
            owner = lambda ident, rec: self.iv.owner_of([self.tm.title(rec) or '', self.tm.category(rec) or ''])
            tables = sizes
        place = {ident: k for k, ident in enumerate(list(dict.fromkeys(list(getattr(self.led, 'order', None) or []) + list(self.led.seen))))}
        price = lambda ident: self.tm.price(self.led.seen[ident]) if self.tm.price(self.led.seen[ident]) is not None else float('inf')
        for i in range(self.count()):
            open_rows = [(ident, rec) for ident, rec in self.led.seen.items() if not self.has_table(ident) and ident not in self.led.gone and (self.tm.in_stock(rec) is not False)]
            pool = [ident for ident, rec in open_rows if owner(ident, rec) == i]
            words = self.wanted_words(i)
            hits = lambda ident: len(words & set(re.findall('[a-z0-9]{3,}', json.dumps(self.led.seen[ident], ensure_ascii=False).lower())))
            pool.sort(key=lambda ident: (-hits(ident) if words else 0, price(ident) if self.led.pick == 'cheapest' else 0, place.get(ident, len(place))))
            have = tables.get(i, 0)
            lanes.append(pool[:max(0, min(each, WIDEN_ITEM_CAP) - have)])
            heads.append(max(0, WIDEN_HEAD - have))
        return (lanes, heads)

    def widen(self):
        detail = self.tm.tool('detail')
        if not detail:
            return 0
        try:
            table, sizes = self.iv.catalogue()
        except Exception:
            return 0
        lanes, heads = self.widen_lanes(sizes)
        depth = max([len(lane) for lane in lanes] or [0])
        first = [lane[k] for k in range(depth) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(depth) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(WIDEN_ROUND, 2 * len(lanes) + 2)
        wanted = first[:cap][:WIDEN_FIRST]
        later = (first[cap:] + more)[:max(0, WIDEN_ALL - len(wanted))]
        if not wanted:
            wanted, later = (later, [])
        if not wanted:
            return 0
        keep = max(WIDEN_KEEP, self.floor() + 2)
        compare = self.tm.tool('compare')
        if compare and len(wanted) > 1 and self.compare_shows(compare):
            room = self.compare_room(compare)
            args = self.tm.args_for('compare', product_ids=wanted[:room])
            if args and self.left() > keep:
                obs = self.act(compare, args)
                if obs is not None:
                    self.note_compare(compare, args, obs)
                    return len(wanted[:room])
        opened = 0
        plan = [(ident, keep) for ident in wanted] + [(ident, max(WIDEN_KEEP_MORE, self.floor() + 2)) for ident in later]
        for ident, room in plan:
            if self.left() <= room:
                break
            detail = self.tm.tool('detail')
            if opened and getattr(self.tm, 'shows_props', {}).get(detail) is False:
                break
            args = self.tm.args_for('detail', product_id=ident, tool=detail)
            if args and self.act(detail, args) is not None:
                opened += 1
        try:
            after = self.iv.catalogue()[1]
        except Exception:
            after = sizes
        if not any((after.get(i, 0) > sizes.get(i, 0) for i in range(self.count()))):
            self.widen_stuck = True
        return opened

    def interview(self):
        self.interviewed = True
        before, told, start = (len(self.led.asked), len(self.led.revealed), len(self.env.history))
        try:
            if not self.widen_stuck:
                self.widen()
        except Exception:
            traceback.print_exc(file=sys.stderr)
        opened = self.opened_since(start)
        try:
            self.iv.plan()
            while self.left() > self.tail() + 2 and self.iv.ask_next():
                pass
        except Exception:
            traceback.print_exc(file=sys.stderr)
        fresh = [n for n in self.led.revealed[told:] if n.kind == 'hidden' and n.value]
        held = bool(fresh) and self.chain_first()
        if held:
            self.deferred += [n for n in fresh if n not in self.deferred]
        try:
            searched = self.value_search(fresh) if fresh and (not held) else ''
        except Exception:
            traceback.print_exc(file=sys.stderr)
            searched = ''
        parts = []
        if len(self.led.asked) > before:
            parts.append('The shopper was asked; the needs are now:\n' + self.led.summary(2500))
        if opened:
            parts.append('Before asking, the program opened the pages of ' + ', '.join(opened) + ' (a call for one of them again costs no step and returns its record).')
        if searched:
            parts.append(searched)
        if held:
            parts.append('The program follows the reference pages first and then searches the new values together with the value those pages give.')
        return '\n'.join(parts)

    def opened_since(self, start):
        out = []
        for h in self.env.history[start:]:
            if self.tm.role.get(h.get('name')) in ('detail', 'compare'):
                for ident in [v for v in (h.get('args') or {}).values() for v in (v if isinstance(v, list) else [v])]:
                    if isinstance(ident, str) and ident in self.led.seen and (ident not in out):
                        out.append(ident)
        return out

    def value_words(self, item, fresh):
        needs_of = getattr(self.led, 'needs_of', None)
        needs = needs_of(item) if callable(needs_of) else self.led.items[item].needs
        records = ' '.join((json.dumps(r, ensure_ascii=False).lower() for r in self.led.seen.values()))
        heard = records + ' ' + ' '.join((str(reply).lower() for _, reply in getattr(self.led, 'talk', [])))
        values, new = ([], [])
        for need in needs:
            value = str(need.value or '').strip()
            kind = need.kind
            if kind == 'ref':
                ref = next((r for r in getattr(self.led, 'refs', None) or [] if r.item == need.item and r.prop == need.prop and getattr(r, 'done', False) and r.value and (getattr(r, 'relation', '') == 'same')), None)
                value, kind = (str(ref.value).strip(), 'stated') if ref else ('', kind)
            parts = [p.strip() for p in re.split('\\s*(?:,|/|\\bor\\b)\\s*', value, flags=re.I) if p.strip()]
            if len(parts) > 1 and (not any((_EX_COMPARE.search(p) for p in parts))):
                value = max(parts, key=lambda p: len(re.findall('(?<![a-z0-9])' + re.escape(p.lower()) + '(?![a-z0-9])', records)))
            low = value.lower()
            if not value or kind not in ('stated', 'hidden') or need.when or (low in [v.lower() for v in values]):
                continue
            words = re.findall('[a-z0-9]+', low)
            if _EX_COMPARE.search(value):
                continue
            if kind == 'stated' and low not in records and (not all((w.rstrip('s') in records for w in re.findall('[a-z0-9]+', low) if w not in ('the', 'a', 'an')))):
                continue
            if kind == 'hidden' and (not words or any((w not in heard for w in words))):
                continue
            values.append(value)
            if need in fresh:
                new.append(value)
        return (values, new)

    def chain_values(self):
        out = []
        for item in range(self.count()):
            needs_of = getattr(self.led, 'needs_of', None)
            for need in needs_of(item) if callable(needs_of) else self.led.items[item].needs:
                if need.kind != 'ref' or id(need) in self.chain_searched:
                    continue
                if any((r.item == need.item and r.prop == need.prop and getattr(r, 'done', False) and r.value and (getattr(r, 'relation', '') == 'same') for r in getattr(self.led, 'refs', None) or [])):
                    self.chain_searched.add(id(need))
                    out.append(need)
        return out

    def words_fit(self, lines):
        for l in lines:
            item = (l.get('item') or 1) - 1
            if not 0 <= item < self.count():
                continue
            facts, _ = _iv_facts(self.led.seen.get(l['id']) or {}, l['id'])
            needs_of = getattr(self.led, 'needs_of', None)
            for need in needs_of(item) if callable(needs_of) else self.led.items[item].needs:
                value = str(getattr(need, 'value', '') or '').strip().lower()
                prop = str(getattr(need, 'prop', '') or '')
                if getattr(need, 'kind', '') not in ('stated', 'hidden') or not value or (not prop) or re.search('\\d', value) or _EX_COMPARE.search(value):
                    continue
                want = _iv_wordset(prop)
                named = lambda k: _iv_wordset(k) and (want <= _iv_wordset(k) or _iv_wordset(k) <= want)
                held = ' '.join((str(v).lower() for k, v in facts.items() if named(k)))
                store = ' '.join((str(v).lower() for rec in self.led.seen.values() if isinstance(rec, dict) for k, v in _iv_facts(rec, '')[0].items() if named(k)))
                words = [w for w in re.findall('[a-z]{3,}', value) if w.rstrip('s') in store]
                if held and (not all((w.rstrip('s') in held for w in words))):
                    return False
        return True

    def title_noun(self, item):
        name = str(self.led.items[item].name or '').strip()
        titles = []
        for rec in self.led.seen.values():
            try:
                texts = [self.tm.title(rec) or '', self.tm.category(rec) or '']
                mine = (self.iv.owner_of(texts, strict=True) if self.count() == 1 else self.iv.owner_of(texts)) == item
            except Exception:
                mine = False
            if mine:
                titles.append(str(self.tm.title(rec) or '').lower())
        text = ' ' + ' '.join((re.sub('[^a-z0-9]+', ' ', t) for t in titles)) + ' '
        best, score = (name, 0)
        for part in re.split('\\s*(?:&|,|/|\\band\\b|\\bor\\b)\\s*', name.lower()):
            words = part.split()
            if not words:
                continue
            root = _ex_singular(words[-1])
            forms = {words[-1]} | {w for w in text.split() if _ex_singular(w) == root}
            for form in sorted(forms):
                phrase = ' '.join(words[:-1] + [form])
                got = 2 * text.count(f' {phrase} ') + text.count(f' {form} ')
                if got > score:
                    best, score = (phrase, got)
        return best

    def value_search(self, fresh):
        tool = self.tm.tool('search')
        if not tool or self.left() < VALUE_ROOM:
            return ''
        budget, said = (VALUE_CALLS, [])
        for item in sorted({n.item for n in fresh}):
            if not 0 <= item < self.count() or budget <= 2:
                continue
            values, new = self.value_words(item, fresh)
            noun = self.title_noun(item)
            if not new:
                if values and noun:
                    new = []
                    tries = [values + [noun]]
                else:
                    continue
            if new:
                lead = new + [v for v in values if v not in new]
                stated = [k for k, v in enumerate(lead) if v not in new]
                drop = stated[-1] if stated else None
                tries = []
                for words in (lead, new + ([noun] if noun else []), lead[:drop] + lead[drop + 1:] if drop is not None else lead, new):
                    if not words or words in tries:
                        continue
                    tries.append(words)
            if self.led.pick == 'cheapest':
                budget += VALUE_CHEAP_EXTRA
            budget = self.search_tries(tool, item, tries, budget, said)
        if not said:
            return ''
        return "The program searched each such item's required values together and read the first fitting rows of its kind (a step a call):\n" + '\n'.join(said)

    def search_tries(self, tool, item, tries, budget, said, below=None, skip_sent=False):
        cheapest = self.led.pick == 'cheapest'
        price = lambda i: self.tm.price(self.led.seen.get(i) or {})
        for words in tries:
            if budget <= 1 or self.left() <= self.floor() + 2:
                break
            args = self.unblind(tool, self.tm.args_for('search', query=' '.join(words))) if words else None
            ranked = VALUE_SHELF and self.led.pick in ('cheapest', 'best')
            if ranked and args and (0 <= item < self.count()):
                label = str(getattr(self.led.items[item], 'shelf', '') or '').strip()
                shelf = self.tm.args_for('search', query=' '.join(words), category=label) if label else None
                if shelf and (not self.tm.validate(tool, shelf)):
                    args = self.unblind(tool, shelf)
            if not args or self.done():
                break
            if self.blocked(tool, args):
                if skip_sent:
                    continue
                break
            start = len(self.env.history)
            obs = self.act(tool, args)
            budget -= 1
            if obs is None:
                break
            self.read_texts[self.mark(tool, args)] = (self.env.step, _ex_show(unwrap(obs)[0]))
            try:
                if ranked:
                    self.deep_pages(tool, args, obs, most=max(0, budget - 2))
                else:
                    self.more_pages(tool, args, obs)
            except Exception:
                traceback.print_exc(file=sys.stderr)
            budget -= max(0, len(self.env.history) - start - 1)
            rows = []
            for h in self.env.history[start:]:
                rows += [i for i in self.kind_rows(item, h.get('obs')) if i not in rows]
            if below is not None:
                rows = [i for i in rows if price(i) is not None and price(i) < below]
            if cheapest:
                rows.sort(key=lambda i: price(i) if price(i) is not None else float('inf'))
            said.append(f"{json.dumps(' '.join(words))}: " + (', '.join(rows[:6]) or 'no row of that kind'))
            if rows:
                opens = VALUE_OPENS_CHEAP if cheapest else VALUE_OPENS
                pages = self.read_pages([i for i in rows if not self.has_table(i)][:min(opens, budget)], self.floor() + VALUE_KEEP)
                budget -= len(pages)
                if pages:
                    said.append('opened: ' + ' | '.join(pages)[:2400])
                if not cheapest:
                    break
        return budget

    def search_unfound(self):
        tool = self.tm.tool('search')
        if self.unfound_tried or not tool or self.left() <= self.floor() + 2:
            return ''
        self.unfound_tried = True
        said = []
        for item in self.unfound_items():
            name = str(self.led.items[item].name or '').strip()
            args = self.unblind(tool, self.tm.args_for('search', query=name)) if name else None
            if not args or self.blocked(tool, args) or self.done() or (self.left() <= self.floor() + 2):
                continue
            obs = self.act(tool, args)
            if obs is None:
                break
            self.read_texts[self.mark(tool, args)] = (self.env.step, _ex_show(unwrap(obs)[0]))
            rows = self.kind_rows(item, obs)
            said.append(f'item {item + 1} ({name}): ' + (', '.join(rows[:6]) or 'no row of that kind'))
            pages = self.read_pages([i for i in rows if not self.has_table(i)][:VALUE_OPENS], self.floor() + 2)
            if pages:
                said.append('opened: ' + ' | '.join(pages)[:2400])
        if not said:
            return ''
        return 'The program searched the items no listing had been found for, by their kind (a step a call):\n' + '\n'.join(said)

    def kind_rows(self, item, obs):
        rows = [r for r in self.tm.rows(obs) or [] if isinstance(r, dict) and self.tm.row_id(r)]
        cap = self.led.budget_cents
        prices = [self.tm.price(r) for r in rows]
        if cap and prices and all((p is not None and p > 10 * cap for p in prices)):
            cap = None
        out, loose = ([], [])
        for r in rows:
            ident = str(self.tm.row_id(r))
            if ident in self.led.gone or self.tm.in_stock(r) is False:
                continue
            if cap and (self.tm.price(r) or 0) > cap:
                continue
            texts = [self.tm.title(r) or '', self.tm.category(r) or '']
            try:
                owner = self.iv.owner_of(texts, strict=True)
            except TypeError:
                owner = self.iv.owner_of(texts)
            except Exception:
                owner = item
            if owner == item and ident not in out:
                out.append(ident)
            elif owner == -2 and ident not in loose:
                loose.append(ident)
        labelled = bool(str(getattr(self.led.items[item], 'shelf', '') or '').strip()) if 0 <= item < self.count() else False
        out = out or ([] if labelled else loose)
        if self.led.pick == 'cheapest':
            price = {str(self.tm.row_id(r)): self.tm.price(r) for r in rows}
            out.sort(key=lambda i: price.get(i) if price.get(i) is not None else float('inf'))
        return out

    def read_pages(self, ids, keep):
        texts = []
        for ident in ids:
            tool = self.tm.tool('detail')
            if not tool or self.left() <= keep or self.done():
                break
            args = self.tm.args_for('detail', product_id=ident, tool=tool)
            if not args or self.blocked(tool, args):
                continue
            obs = self.act(tool, args)
            if obs is None:
                break
            shown = _ex_show(unwrap(obs)[0])
            self.read_texts[self.mark(tool, args)] = (self.env.step, shown)
            texts.append(f'{ident}: {shown[:1500]}')
        return texts

    def control(self, name, args):
        if not self.interviewed:
            said = self.interview()
            if said:
                return 'Not run yet (no step used). Before ordering or giving up:\n' + said + '\nRead this, then decide again.'
        if name == 'buy_now':
            lines = self.lines_from(args.get('lines'))
            return self.buy(lines) if lines else 'No lines given (no step used).'
        remark = str(args.get('remark') or '').strip()
        self.remark = remark or self.remark
        self.said_none = True
        held = self.late_hold()
        if held:
            return held
        said = self.search_unfound()
        if said:
            return 'Not ended (no step used). ' + said + '\nJudge these, then decide again.'
        cap = self.left() + self.env.step
        if self.pushes < PUSHES and (self.searches() < 2 * self.count() + 1 or self.left() > cap // 3):
            self.pushes += 1
            said = PUSH.format(left=self.left())
            if self.env.step == self.push_step:
                start = len(self.env.history)
                opened = self.widen()
                if opened:
                    pages = [_ex_show(unwrap(h.get('obs'))[0]) for h in self.env.history[start:]]
                    said += f' The program has just opened {opened} more listing page(s); judge them before ending:\n' + '\n'.join(pages)
            self.push_step = self.env.step
            return said
        if DECLINE_ON_BUY and self.broken_stands():
            return self.finish('decline', self.remark)
        lines = self.pick()
        if lines and self.pick_certain and self.words_fit(lines):
            return self.order_picked(lines, 'Before ending without an order, the program picked these lines, which it found to meet every requirement')
        if self.gate.should_decline():
            return self.finish('decline', self.remark)
        if getattr(self.gate, 'unread_audit', False) and self.lead_holds < 1 and (self.left() > self.floor()):
            self.lead_holds += 1
            return 'Not ended (no step used). The check of this ending could not confirm that nothing fits. Open the most promising listings of each item and order what fits; end only if each clearly fails.'
        leads = self.leads()
        lines = leads or lines
        if lines and self.lead_holds < 1:
            self.lead_holds += 1
            return 'Not ended (no step used). A review of the records finds listings that seem to fit: ' + ', '.join((f"item {l['item']}: {l['id']}" for l in lines)) + '. Open them if needed and order what truly fits with buy_now; end only if each clearly fails a requirement, then say which one and call nothing_fits again.'
        if leads and len({l['item'] for l in leads}) == self.count() and all((self.has_table(l['id']) for l in leads)) and (not self.lead_ordered) and (self.left() > self.floor()):
            self.lead_ordered = True
            if not self.words_fit(leads):
                return self.finish('decline', self.remark)
            said = self.order_picked(leads, 'The review found these listings to meet every requirement, so the program ordered them')
            if said is None or self.done():
                return None
            return 'Not ended (no step used). ' + said
        return self.finish('decline', self.remark)

    def late_hold(self):
        if self.late_held or self.left() > self.floor() + LATE_ROOM:
            return ''
        lines = next((l for l in (self.passed, self.planned) if self.usable(l)), None)
        if not lines:
            return ''
        self.late_held = True
        return 'Not ended (no step used). Ending without an order earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. Order ' + ', '.join((f"item {l['item']}: {l['id']}" for l in lines)) + ' with buy_now. End without an order only if your search showed that some requested item has no listing that meets its requirements; then call nothing_fits again.'

    def order_picked(self, lines, what):
        self.planned = lines
        ok, why = self.gate.ready_for(lines)
        if self.done():
            return None
        if not ok:
            return what + ' (' + ', '.join((f"item {l['item']}: {l['id']}" for l in lines)) + '), but the checks hold them: ' + why
        lines, _ = self.verify(lines)
        if self.done():
            return None
        self.passed = lines
        refusal = self.place(lines)
        return refusal if refusal is None else refusal + '\nApply every change in it, then call buy_now again.'

    def stop(self, floor, spend_cap):
        return self.done() or self.left() <= floor or self.llm.spent >= spend_cap or (self.llm.calls >= self.llm.max_calls - CALL_RESERVE)

    def loop(self, turns, floor, spend_cap):
        idle = 0
        for _ in range(turns):
            if self.stop(self.chain_floor(floor), spend_cap):
                return
            if not self.interviewed and self.left() > floor:
                budget = self.led.step_budget or self.env.max_steps
                late = self.left() <= budget * 3 // 5
                if late or (len(self.led.seen) >= 2 * self.count() + 1 and self.ready_to_ask()):
                    said = self.interview()
                    if said:
                        self.say(said + '\n' + self.status())
            low = self.chain_floor(floor)
            marks = (self.env.step, self.llm.calls)
            if self.chain.pending() and self.left() > low + 2:
                room = self.left() - low - 2
                if self.chain_first():
                    if not self.interviewed:
                        budget = self.led.step_budget or self.env.max_steps
                        room = max(min(4, room), min(room, self.left() - budget * 3 // 5))
                    used = self.chain.advance(max_calls=room, model_calls=self.llm.max_calls - CALL_RESERVE - self.llm.calls, halt=lambda: self.stop(low, spend_cap))
                else:
                    used = self.chain.advance(max_calls=min(4, room))
                if used:
                    said = 'Reference pages followed:\n' + str(self.chain.values_text() or '(no value read yet)')[:1500]
                    got = self.after_chain()
                    said += '\n' + got if got else ''
                    self.say(said)
            if self.deferred and (not self.chain_first()):
                got = self.after_chain()
                if got:
                    self.say(got)
            opened = self.chain.report() if hasattr(self.chain, 'report') else ''
            if opened:
                self.say('The program opened these pages (already read; no need to open them again):\n' + opened)
            if self.stop(self.chain_floor(floor), spend_cap):
                return
            if self.left() <= floor:
                if self.chain_first() and (self.env.step, self.llm.calls) != marks:
                    continue
                return
            self.heed()
            self.flush()
            self.trim()
            msg = self.llm.chat(self.history, self.offer_now(floor), max_tokens=3000)
            calls = (msg or {}).get('tool_calls') or []
            if not calls:
                idle += 1
                if msg:
                    self.history.append({'role': 'assistant', 'content': msg.get('content') or '...'})
                    self.say('Call one of the tools to continue.\n' + self.status())
                elif getattr(self.llm, 'last_cut', False):
                    self.say('Your last reply ran out of room before a whole tool call. Reply with one short tool call now.')
                if idle >= 3:
                    return
                continue
            idle = 0
            ran = self.run_turn(msg, calls, floor)
            if self.done():
                continue
            if not ran and self.left() - floor <= 1:
                return
            missing = self.unfound_items() if self.env.step - self.status_at >= 4 else []
            text = '\n' + self.status()
            if self.env.step - self.status_at >= 4 or missing:
                self.status_at = self.env.step
                text += '\nNeeds and state:\n' + self.led.summary(2000)
                if missing:
                    text += '\nNo listing of these items has been found yet; search for them now, by their kind: ' + '; '.join((f'item {i + 1} ({self.led.items[i].name})' for i in missing))
            if self.history and self.history[-1].get('role') == 'tool':
                self.history[-1]['content'] += text

    def offer_now(self, floor):
        if self.left() - floor <= 1:
            return list(CONTROL)
        return self.offered

    def run_turn(self, msg, calls, floor):
        turn = []
        for k, call in enumerate(calls):
            fn = (call or {}).get('function') or {}
            name = fn.get('name')
            try:
                args = json.loads(fn.get('arguments') or '{}')
            except (TypeError, ValueError):
                args = {}
            args = args if isinstance(args, dict) else {}
            turn.append(({'id': (call or {}).get('id') or f'call{len(self.history)}_{k}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}, name, args))
        self.history.append({'role': 'assistant', 'content': msg.get('content') or '', 'tool_calls': [t[0] for t in turn]})
        joined, lead, ran = (self.join_opens(turn), None, 0)
        offer = {(t.get('function') or t).get('name') for t in self.offer_now(floor)}
        for k, (call, name, args) in enumerate(turn):
            if k in joined and joined[k] is None:
                self.history.append({'role': 'tool', 'tool_call_id': call['id'], 'content': lead if lead.startswith(('Held', 'Not sent')) else 'Shown in the side-by-side result above: these listings were opened together in one step.'})
                continue
            if k in joined:
                name, args = joined[k]
            print('CALL', self.env.step, name, json.dumps(args)[:200], file=sys.stderr)
            if self.done():
                result = 'Held: the episode has ended.'
            elif name in ('buy_now', 'nothing_fits') and len(turn) > 1:
                result = f'Held (no step used): {name} must be the only call of its reply; read these results first.'
            elif name in self.store_tools and self.left() <= floor:
                result = 'Held (no step used): the remaining steps are reserved for the order.'
            elif name in self.store_tools and name not in offer:
                result = 'Held (no step used): too few steps are left for that; use: ' + ', '.join(sorted(offer)) + '.'
            else:
                result = self.run_call(name, args)
                ran += 1
            lead = str(result) if k in joined else lead
            self.history.append({'role': 'tool', 'tool_call_id': call['id'], 'content': str(result)})
        return ran

    def join_opens(self, turn):
        detail, compare = (self.tm.tool('detail'), self.tm.tool('compare'))
        if not detail or not compare or compare == detail or (not self.compare_shows(compare)):
            return {}
        picks = []
        for k, (_, name, args) in enumerate(turn):
            ident = next(iter(args.values()), None) if len(args) == 1 else None
            if name == detail and isinstance(ident, str) and (ident in self.led.seen) and (ident not in self.led.gone) and (ident not in [i for _, i in picks]) and (not self.shown_already(detail, ident)):
                picks.append((k, ident))
        picks = picks[:self.compare_room(compare)]
        if len(picks) < 2:
            return {}
        args = self.tm.args_for('compare', product_ids=[i for _, i in picks])
        if not args:
            return {}
        out = {k: None for k, _ in picks}
        out[picks[0][0]] = (compare, args)
        return out

    def run_call(self, name, args):
        if name in ('buy_now', 'nothing_fits'):
            result = self.control(name, args)
            return 'Done: the episode is finished.' if result is None or self.done() else result
        if name in self.store_tools:
            args = self.shelf_label(name, args)
            args, trimmed, none_left = self.trim_compare(name, args)
            if none_left:
                return 'Not sent (no step used): ' + none_left
            why = self.blocked(name, args)
            if why:
                return 'Not sent (no step used): ' + why
            before, start = (self.env.step, len(self.env.history))
            obs = self.act(name, args)
            if self.env.step == before:
                return 'The store did not answer this call.'
            body, notes = unwrap(obs)
            try:
                rest = self.more_pages(name, args, obs)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                rest = ''
            shown = _ex_show(body)
            if self.tm.role.get(name) not in ('cart_set', 'cart_view'):
                self.read_texts[self.mark(name, args)] = (self.env.step, shown)
            if self.tm.role.get(name) in ('search', None, 'other'):
                rest += self.cheapest_here([h.get('obs') for h in self.env.history[start:]])
                rest += self.cap_ignored(name, args, obs)
                rest += self.text_ignored(name, args, obs)
            rest += self.page_hint(name, args) + self.note_compare(name, args, obs)
            rest += self.evidence(name, args)
            return shown + trimmed + rest + (f"\nNotices from the shopper or store: {' '.join(notes)}" if notes else '')
        return 'Unknown tool; use one of the listed tools.'

    def cap_ignored(self, name, args, obs):
        arg = ((getattr(self.tm, 'slots', None) or {}).get(name) or {}).get('max_price')
        cap = args.get(arg) if arg else None
        if isinstance(cap, bool) or not isinstance(cap, (int, float)):
            return ''
        over = [r for r in self.tm.rows(obs) or [] if (self.tm.price(r) or 0) > cap]
        if not over:
            return ''
        props = ((getattr(self.tm, 'schemas', None) or {}).get(name) or {}).get('properties') or {}
        others = [f'{k}={json.dumps(o)}' for k, v in args.items() if isinstance(v, str) for o in [next((x for x in (props.get(k) or {}).get('enum') or [] if x != v), None)] if o is not None]
        hint = '; list the category with ' + ', '.join(others) + f' and the same {arg} to see only cheaper listings' if others else ''
        return f'\nNote: this search did not apply {arg}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def evidence(self, name, args):
        if self.tm.role.get(name) not in ('detail', 'compare'):
            return ''
        ids = [v for v in (args or {}).values() for v in (v if isinstance(v, list) else [v]) if isinstance(v, str) and v in self.led.seen]
        needs = []
        for item in range(self.count()):
            needs_of = getattr(self.led, 'needs_of', None)
            for need in needs_of(item) if callable(needs_of) else self.led.items[item].needs:
                prop = str(getattr(need, 'prop', '') or '').strip()
                if prop and (item, prop) not in needs:
                    needs.append((item, prop))
        if not ids or not needs:
            return ''
        lines = []
        for ident in list(dict.fromkeys(ids))[:12]:
            facts, _ = _iv_facts(self.led.seen.get(ident) or {}, ident)
            parts, missing = ([], [])
            for item, prop in needs[:8]:
                want = _iv_wordset(prop)
                hits = [f'{k}={str(v)[:50]}' for k, v in facts.items() if want and _iv_wordset(k) and (want <= _iv_wordset(k) or _iv_wordset(k) <= want)]
                tag = f'item {item + 1} ' if self.count() > 1 else ''
                if hits:
                    parts.append(f'{tag}{prop}: ' + ', '.join(hits[:3]))
                else:
                    missing.append(f'{tag}{prop}')
            if parts or missing:
                lines.append(f'{ident}: ' + '; '.join(parts) + (f" | no key names {', '.join(missing)}" if missing else ''))
        if not lines:
            return ''
        return "\nKeys on these pages named like a requirement's property (a key for another part or aspect decides nothing; a value may still sit under a key of another name): " + ' || '.join(lines)

    def shelf_label(self, name, args):
        arg = ((getattr(self.tm, 'slots', None) or {}).get(name) or {}).get('category')
        value = args.get(arg) if arg and isinstance(args, dict) else None
        if not isinstance(value, str) or not value.strip() or self.tm.role.get(name) != 'search':
            return args
        try:
            labels = {str(self.tm.category(r)).strip() for r in self.led.seen.values() if self.tm.category(r)}
        except Exception:
            return args
        new = value
        if not any((l.lower() == value.strip().lower() for l in labels)):
            words = set(re.findall('[a-z0-9]+', value.lower()))
            holders = [l for l in labels if words and words <= set(re.findall('[a-z0-9]+', l.lower()))]
            if len(holders) == 1:
                new = holders[0] if value != value.lower() else holders[0].lower()
        accepted = [str((h.get('args') or {}).get(arg)) for h in self.env.history if h.get('name') == name and (h.get('args') or {}).get(arg) and (not _ex_refused(h.get('obs')))]
        same = next((a for a in accepted if a.lower() == new.lower()), None)
        if same is not None:
            new = same
        elif accepted and all((a == a.lower() for a in accepted)):
            new = new.lower()
        return dict(args, **{arg: new}) if new != value else args

    def text_ignored(self, name, args, obs):
        arg = ((getattr(self.tm, 'slots', None) or {}).get(name) or {}).get('query')
        words = args.get(arg) if arg else None
        if not isinstance(words, str) or not words.strip():
            return ''
        ids = [self.tm.row_id(r) for r in self.tm.rows(obs) or []]
        if not ids or self.blind(name, args):
            return ''
        slots = (getattr(self.tm, 'slots', None) or {}).get(name) or {}
        loose = {arg, self.tm._page_arg(name) if hasattr(self.tm, '_page_arg') else None, slots.get('max_price')}
        same = {k: v for k, v in args.items() if k not in loose}
        for h in self.env.history[:-1]:
            old = h.get('args') or {}
            if h.get('name') != name or {k: v for k, v in old.items() if k not in loose} != same:
                continue
            if str(old.get(arg) or '').strip().lower() in ('', words.strip().lower()):
                continue
            if set(re.findall('[a-z0-9]{3,}', words.lower())) & set(re.findall('[a-z0-9]{3,}', str(old.get(arg)).lower())):
                continue
            if [self.tm.row_id(r) for r in self.tm.rows(h.get('obs')) or []] == ids:
                self.blind_calls.append((name, same))
                props = ((getattr(self.tm, 'schemas', None) or {}).get(name) or {}).get('properties') or {}
                others = [f'{k}={json.dumps(o)}' for k, v in same.items() if isinstance(v, str) for o in [next((x for x in (props.get(k) or {}).get('enum') or [] if x != v), None)] if o is not None]
                hint = '; search the words with ' + ', '.join(others) + ' instead' if others else ''
                return f'\nNote: these rows are the same as for {json.dumps(old.get(arg))}: with these other arguments the store does not read {arg}{hint}.'
        return ''

    def blind(self, name, args):
        slots = (getattr(self.tm, 'slots', None) or {}).get(name) or {}
        loose = {slots.get('query'), self.tm._page_arg(name) if hasattr(self.tm, '_page_arg') else None, slots.get('max_price')}
        same = {k: v for k, v in args.items() if k not in loose}
        return any((n == name and s == same for n, s in self.blind_calls))

    def unblind(self, name, args):
        if not args or not self.blind(name, args):
            return args
        props = ((getattr(self.tm, 'schemas', None) or {}).get(name) or {}).get('properties') or {}
        for k, v in args.items():
            other = next((x for x in (props.get(k) or {}).get('enum') or [] if x != v), None)
            if isinstance(v, str) and other is not None:
                swapped = dict(args, **{k: other})
                if not self.tm.validate(name, swapped):
                    return swapped
        return args

    def page_hint(self, name, args):
        if self.tm.role.get(name) != 'detail' or getattr(self.tm, 'shows_props', {}).get(name) is not False:
            return ''
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.led.seen), None)
        better = self.tm.tool('detail')
        if not ident or not better or better == name or (self.tm.shows_props.get(better) is False) or self.has_table(ident):
            return ''
        return f"\n(This view shows no property table; {better} shows {ident}'s properties.)"

    def cheapest_here(self, observations):
        if self.led.pick != 'cheapest':
            return ''
        priced = {}
        for obs in observations:
            for row in self.tm.rows(obs) or []:
                ident, cost = (self.tm.row_id(row), self.tm.price(row))
                if ident and cost is not None and (ident not in self.led.gone) and (self.tm.in_stock(row) is not False):
                    priced[str(ident)] = cost
        if len(priced) < 2:
            return ''
        ranked = sorted(priced.items(), key=lambda kv: (kv[1], kv[0]))[:6]
        return '\nBuyable rows here, cheapest first: ' + ', '.join((f'{i} ({c:g})' for i, c in ranked))

    def holds_value(self, name, args):
        arg = ((getattr(self.tm, 'slots', None) or {}).get(name) or {}).get('query')
        words = set(re.findall('[a-z0-9]{3,}', str(args.get(arg) or '').lower())) if arg else set()
        if not words:
            return False
        wanted = set()
        for item in range(self.count()):
            wanted |= self.wanted_words(item)
        return bool(words & wanted)

    def more_pages(self, name, args, obs):
        if self.tm.role.get(name) != 'search' or self.led.pick not in ('cheapest', 'best'):
            return ''
        if not all((callable(getattr(self.tm, f, None)) for f in ('next_page', 'total', 'rows', 'price'))):
            return ''
        page_of = getattr(self.tm, '_page_arg', None)
        param = page_of(name) if callable(page_of) else None
        value, count, rows = (self.tm.next_page(name, obs), self.tm.total(name, obs), self.tm.rows(obs))
        if not param or value is None or (not count) or (not rows) or (count <= len(rows)):
            return ''
        if args.get(param) not in (None, '', 0, 1):
            return ''
        prices = [self.tm.price(r) for r in rows]
        if self.led.pick == 'cheapest' and None not in prices and (prices == sorted(prices)):
            return ''
        pages = -(-(count - len(rows)) // len(rows))
        if pages > 2 and self.pages_read + 2 <= AUTO_PAGES and (self.left() - self.tail() - 2 >= 12) and self.holds_value(name, args):
            pages = 2
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
            return f'\nNote: this search matched {count} listings and showed {len(rows)}, in no price order, and the pick weighs every fitting listing: search with narrower words (a required value) or page on with {param}={json.dumps(value)}.'
        texts = []
        for _ in range(pages):
            call = dict(args)
            call[param] = value
            if self.tm.validate(name, call):
                break
            got = self.act(name, call)
            if got is None or self.done():
                break
            self.pages_read += 1
            texts.append(f'{param}={json.dumps(value)}: ' + _ex_show(unwrap(got)[0]))
            self.sent.add(self.mark(name, call))
            self.read_texts[self.mark(name, call)] = (self.env.step, texts[-1])
            value = self.tm.next_page(name, got)
            if value is None:
                break
        if not texts:
            return ''
        rest = f'\n{count} listings matched and the program has read {len(rows) * (len(texts) + 1)} rows of them; narrow the search with a required value or page on with {param}={json.dumps(value)} to see the rest.' if value is not None else ''
        return '\nThe program read the rest of this search itself, a step a page:\n' + '\n'.join(texts) + rest

    def run(self):
        self.loop(TURNS, self.floor(), self.llm.soft_cap)
        if ASK_AT_CLOSE and (not self.interviewed) and (not self.done()) and (self.left() > self.tail() + 2):
            try:
                self.interview()
            except Exception:
                traceback.print_exc(file=sys.stderr)
        self.close()

    def usable(self, lines):
        if not lines or len(lines) < self.count() or any((l['id'] in self.led.gone for l in lines)):
            return False
        key = self.sig(lines)
        if STOCK_REFUSAL and key in self.sold_sigs:
            return False
        times = self.refused.get(key, 0)
        if not times:
            return True
        if times > RESUBMITS:
            return False
        if key in self.soft_refused:
            return not (self.gate._shape(lines) or self.gate._stock_budget(lines))
        if self.left() <= self.tail():
            return False
        try:
            problem = self.gate.recheck(lines)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            problem = 'check failed'
        if problem:
            self.refused[key] = RESUBMITS + 1
            return False
        return True

    def final_lines(self):
        self.close_uncertain = False
        if self.usable(self.passed):
            return self.passed
        if CLOSE_KEEP_CART:
            kept = self.kept_lines()
            if kept:
                return kept
        lines = self.guarded(self.pick)
        if self.usable(lines):
            self.close_uncertain = not self.pick_certain
            return lines
        if self.usable(self.planned):
            return self.planned
        cart = self.cart_lines()
        want = sorted((it.qty for it in self.led.items))
        if self.usable(cart) and (not want or sorted((l['qty'] for l in cart)) == want):
            return cart
        lines = self.guarded(self.leads)
        return lines if self.usable(lines) else []

    def kept_lines(self):
        cart = self.cart_lines()
        want = sorted((it.qty for it in self.led.items))
        shaped = cart if cart and (not want or sorted((l['qty'] for l in cart)) == want) else []
        for lines in (self.planned, shaped):
            if not self.usable(lines):
                continue
            try:
                faults = self.gate.code_faults(lines) if hasattr(self.gate, 'code_faults') else {}
            except Exception:
                traceback.print_exc(file=sys.stderr)
                continue
            if not faults:
                return lines
        return []

    def guarded(self, make):
        if self.llm.closed():
            return []
        try:
            return make() or []
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return []

    def close(self):
        repaired = False
        for _ in range(4):
            if self.done() or self.left() <= 0:
                return
            try:
                refusal = self.close_once()
            except Exception:
                traceback.print_exc(file=sys.stderr)
                refusal = None if self.done() else self.finish('decline', self.remark)
            if refusal is None or self.done():
                return
            if not repaired and self.left() > self.count() + 2 and (not self.llm.closed()):
                repaired = True
                self.say('The program tried to end the episode. ' + refusal + '\nTake that into account, then call buy_now (or nothing_fits).')
                try:
                    self.loop(REPAIR_TURNS, self.count() + 1, (self.llm.soft_cap + self.llm.hard_cap) / 2)
                except Exception:
                    traceback.print_exc(file=sys.stderr)

    def close_once(self):
        self.heed()
        self.resolve_sold()
        if DECLINE_ON_BUY and self.broken_stands():
            return self.finish('decline', self.remark)
        lines = self.final_lines()
        if not lines and DECLINE_EVIDENCE and (self.left() > self.tail() + 2):
            try:
                lines = self.gate.evidence_lines() if hasattr(self.gate, 'evidence_lines') else []
            except Exception:
                traceback.print_exc(file=sys.stderr)
                lines = []
            lines = lines if self.usable(lines) else []
        if not lines:
            return self.finish('decline', self.remark)
        lines, decline = self.verify(lines, forced=True)
        if self.done():
            return None
        if decline or (STOCK_REFUSAL and self.sig(lines) in self.sold_sigs):
            return self.finish('decline', self.remark)
        try:
            self.gate.process_now()
        except Exception:
            traceback.print_exc(file=sys.stderr)
        if self.done():
            return None
        return self.place(lines, final=True)

def _ex_episode(ex):
    try:
        ex.run()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        try:
            ex.close()
        except Exception:
            traceback.print_exc(file=sys.stderr)
    return ex.env.history

def agent_main(problem_data):
    try:
        ex = Explorer(problem_data)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return []
    return _ex_episode(ex)
