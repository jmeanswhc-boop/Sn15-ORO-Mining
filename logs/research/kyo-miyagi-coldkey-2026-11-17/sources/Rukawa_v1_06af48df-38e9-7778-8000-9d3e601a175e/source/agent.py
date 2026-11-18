"""Rukawa: a shopping agent for ORO world tasks.

Every judgement is made by a language model: what the shopper asks for, which tool does which job, what to search
for, what to ask the shopper, whether a listing meets each requirement, and which basket to order. Code runs the
calls, keeps the step, message and time budgets, and checks the model's answers against the tools' schemas.
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.agent.proxy_client import ProxyClient

FAST = 'anthropic/claude-haiku-5.5'
SMART = 'anthropic/claude-sonnet-5.5'
TASK_SECONDS = 300.0
FINISH_BY = 262.0          # start the finishing moves by this many seconds into the task
LLM_TIMEOUT = 75
FAST_TOKENS = 6000
SMART_TOKENS = 4000
SPEND_SOFT = 0.045         # above this (when the provider reports cost) the agent stops adding new work
SMART_SPEND = 0.035        # above this the heavy model is no longer used
OPEN_PER_ITEM = 4
PAGES_CHEAPEST = 4
ROW_CHARS = 260
RECORD_CHARS = 2600
TRACE_CHARS = 1200

_env = ProxyClient(timeout=60, max_retries=1, retry_delay=1)
_llm_clients: dict[int, ProxyClient] = {}
_llm_lock = threading.Lock()


def _llm_client(seconds: float) -> ProxyClient:
    limit = max(8, min(LLM_TIMEOUT, int(seconds)))
    with _llm_lock:
        if limit not in _llm_clients:
            _llm_clients[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
        return _llm_clients[limit]


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + '...'


def _json_of(text: Any) -> Any:
    if not isinstance(text, str) or not text.strip():
        return None
    body = text.strip()
    body = re.sub(r'^```[a-zA-Z]*\s*', '', body)
    body = re.sub(r'\s*```\s*$', '', body)
    try:
        return json.loads(body)
    except ValueError:
        pass
    for opener, closer in (('{', '}'), ('[', ']')):
        start, end = body.find(opener), body.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(body[start:end + 1])
            except ValueError:
                continue
    return None


def _str_list(value: Any) -> list[str]:
    """A model's list field as a list of strings (a lone string becomes one entry)."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, '') and not isinstance(v, (dict, list))]
    return []


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        found = re.search(r'-?\d[\d,]*(?:\.\d+)?', value)
        if found:
            try:
                return float(found.group(0).replace(',', ''))
            except ValueError:
                return None
    return None


def _path_get(obj: Any, path: str | None) -> Any:
    if not path or obj is None:
        return None
    current = obj
    for part in str(path).split('.'):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _text_record(text: Any) -> dict | None:
    """A listing written as text, "key: value | key: value", read into a dict."""
    if not isinstance(text, str):
        return None
    parts = [p for p in re.split(r'\s+\|\s+|\n', text) if p.strip()]
    record: dict[str, Any] = {}
    for part in parts:
        key, sep, value = part.partition(':')
        if sep and 0 < len(key.strip()) <= 40 and '  ' not in key.strip():
            record[key.strip()] = value.strip()
        elif record:
            last = list(record)[-1]
            record[last] = f'{record[last]} | {part.strip()}'
    return record if len(record) >= 2 else None


def _dict_lists(obj: Any, depth: int = 0) -> list[list[dict]]:
    found: list[list[dict]] = []
    if depth > 3:
        return found
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        found.append(obj)
    elif isinstance(obj, list) and obj and all(isinstance(x, str) for x in obj):
        read = [_text_record(x) for x in obj]
        if all(r is not None for r in read):
            found.append(read)
    if isinstance(obj, dict):
        for value in obj.values():
            found.extend(_dict_lists(value, depth + 1))
    elif isinstance(obj, list):
        for value in obj[:3]:
            if isinstance(value, (dict, list)):
                found.extend(_dict_lists(value, depth + 1))
    return found


def _looks_like_id(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_\-:.#]{3,63}', value)) \
        and bool(re.search(r'\d', value))


class Episode:
    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.opening = self.view.get('opening')
        self.tools = []
        for tool in self.view.get('tools') or []:
            spec = tool.get('function', tool) if isinstance(tool, dict) else None
            if isinstance(spec, dict) and spec.get('name'):
                self.tools.append(spec)
        self.tool_by = {t['name']: t for t in self.tools}
        self.max_steps = int(self.view.get('max_steps') or 60)
        self.step = 0
        self.finished = bool(self.view.get('finished'))
        self.t0 = time.monotonic()
        self.trace: list[dict[str, Any]] = []
        self.cost = 0.0
        self.calls = 0
        self.lock = threading.Lock()
        self.roles: dict[str, Any] = {}
        self.plan: dict[str, Any] = {}
        self.items: list[dict[str, Any]] = []
        self.schema: dict[str, Any] = {}
        self.rows: dict[str, dict] = {}
        self.records: dict[str, dict] = {}
        self.seen_by_item: dict[str, list[str]] = {}
        self.queries: list[dict[str, Any]] = []
        self.verdicts: dict[str, dict[str, dict]] = {}
        self.events: list[dict[str, Any]] = []
        self.dialogue: list[dict[str, str]] = []
        self.told: list[dict[str, Any]] = []
        self.messages_used = 0
        self.message_limit: int | None = None
        self.obs_keys: dict[str, set] = {}
        self.cart_lines: dict[str, int] = {}
        self.outcome = ''
        self.problems: list[dict[str, Any]] = []
        self.price_seen: dict[str, float] = {}
        self.decline_checked = False
        self.no_think = False
        self.asked_items: list[str] = []
        self.single_mode = False
        self.answered_none: dict[str, list[str]] = {}
        self.paged: set = set()

    # ------------------------------------------------------------------ plumbing
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def left(self) -> float:
        return TASK_SECONDS - self.elapsed()

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def log(self, kind: str, **data: Any) -> None:
        entry = {'kind': kind, 'at': round(self.elapsed(), 1)}
        for key, value in data.items():
            entry[key] = value if isinstance(value, (int, float, bool)) or value is None else _clip(value, TRACE_CHARS)
        with self.lock:
            self.trace.append(entry)

    def llm(self, prompt: str, model: str = FAST, max_tokens: int | None = None, system: str | None = None,
            label: str = '', think: str | None = None) -> Any:
        """One JSON-answer model call with one fallback to the other model.

        think: None keeps the model's default reasoning, 'off' turns it off, 'low' asks for a little."""
        if model == SMART and self.cost > SMART_SPEND:
            model = FAST                       # past this spend the rest of the task runs on the light model
        order = [model, FAST]                  # a failed call is retried on the light model, never moved up
        with_think = think is not None and not self.no_think
        attempt, waits = 0, 0
        while attempt < len(order):
            name = order[attempt]
            room = self.left() - 12
            if room < 10:
                return None
            messages: list[dict[str, Any]] = []
            if system:
                messages.append({'role': 'system', 'content': [
                    {'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}]})
            messages.append({'role': 'user', 'content': prompt})
            body = {'model': name, 'messages': messages, 'temperature': 0,
                    'max_tokens': max_tokens or (FAST_TOKENS if name == FAST else SMART_TOKENS)}
            if with_think:
                body['reasoning'] = {'enabled': False} if think == 'off' else {'effort': think}
            began = time.monotonic()
            result = self.inference_post(min(room, 35 if name == FAST else 50), body)
            with self.lock:
                self.calls += 1
            data = result.data
            if isinstance(data, dict) and data.get('choices'):
                usage = data.get('usage') or {}
                if isinstance(usage.get('cost'), (int, float)):
                    with self.lock:
                        self.cost += float(usage['cost'])
                message = data['choices'][0].get('message') or {}
                parsed = _json_of(message.get('content'))
                if parsed is not None:
                    self.log('llm', label=label, model=name, secs=round(time.monotonic() - began, 1),
                             cost=usage.get('cost'), tokens=[usage.get('prompt_tokens'), usage.get('completion_tokens')],
                             end=data['choices'][0].get('finish_reason'))
                    return parsed
                self.log('llm_bad', label=label, model=name, text=message.get('content'))
            else:
                error = result.error or {}
                self.log('llm_fail', label=label, model=name, status=error.get('status'),
                         body=str(error.get('body') or '')[:200])
                if error.get('status') == 402:
                    return None
                if error.get('status') == 400 and with_think and 'reason' in str(error.get('body') or '').lower():
                    self.no_think = True       # the route refused the reasoning setting: never send it again
                    with_think = False
                    order[1] = name            # the same model again, without the setting
                if error.get('status') in (429, 500, 502, 503, 504) and waits < 3:
                    # a busy or briefly failing proxy: wait, then the same model again
                    pause = 2.0 * (2 ** waits) + (time.monotonic() * 7.0) % 1.0
                    if self.left() - pause > 50:
                        waits += 1
                        time.sleep(pause)
                        continue
            if attempt == 0:
                if self.left() < 60:
                    return None            # too late for a second model: the finishing moves need the time
                time.sleep(1.0)
            attempt += 1
        return None

    def inference_post(self, seconds: float, body: dict[str, Any]) -> Any:
        """One chat-completion request through the sandbox's model proxy."""
        return _llm_client(seconds).post_verbose('/inference/chat/completions', body)

    def parallel(self, jobs: list) -> list:
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
            futures = [pool.submit(job) for job in jobs]
            out = []
            for future in futures:
                try:
                    out.append(future.result())
                except Exception as exc:  # a failed helper must not end the episode
                    self.log('error', where='parallel', error=repr(exc))
                    out.append(None)
            return out

    def act(self, name: str, args: dict[str, Any]) -> Any:
        if self.finished or self.step >= self.max_steps or name not in self.tool_by:
            return None
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        data = None
        began = time.monotonic()
        for attempt in range(3):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                break
            if self.left() < 15:
                break
            time.sleep(1.0 + attempt)
        if data is None:
            self.log('env_fail', tool=name, args=args)
            return None
        self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
        if data.get('finished'):
            self.finished = True
        observation = data.get('observation')
        self.note_event(name, observation)
        self.log('act', step=self.step, tool=name, args=args, secs=round(time.monotonic() - began, 2),
                 obs=_clip(observation, 700))
        return observation

    def act_fixed(self, name: str, args: dict[str, Any]) -> Any:
        """A call whose arguments the harness turned down gets them rewritten once by the model, then is sent again."""
        observation = self.act(name, args)
        error = observation.get('error') if isinstance(observation, dict) else None
        if not error or 'argument' not in str(error).lower() or self.steps_left() < 4 or self.left() < 40:
            return observation
        spec = self.tool_by.get(name) or {}
        prompt = (FIX_PROMPT.replace('{TOOL}', json.dumps(spec, ensure_ascii=False))
                  .replace('{ARGS}', json.dumps(args, ensure_ascii=False))
                  .replace('{ERROR}', _clip(observation, 600)))
        got = self.llm(prompt, model=FAST, label='fix_args', think='off')
        fixed = got.get('args') if isinstance(got, dict) else None
        if isinstance(fixed, dict) and fixed != args:
            return self.act(name, fixed)
        return observation

    def note_event(self, tool: str, observation: Any) -> None:
        """Keep any observation part this tool did not show before: world notices, warnings, changes."""
        if not isinstance(observation, dict):
            return
        keys = set(observation.keys())
        if 'error' in keys:
            return                         # a turned-down call is not the tool's normal reply shape
        known = self.obs_keys.get(tool)
        if known is None:
            self.obs_keys[tool] = keys
            return
        extra = keys - known
        if extra:
            self.events.append({'tool': tool, 'step': self.step, 'extra': {k: observation[k] for k in extra}})

    # ------------------------------------------------------------------ setup
    def shared_system(self) -> str:
        """One system prompt for the planning calls of a task, so the provider can cache it between them."""
        if getattr(self, '_shared', None) is None:
            opening = '' if self.opening is None else '\n\nOPENING NOTE FROM THE STORE:\n' + _clip(self.opening, 3000)
            self._shared = (SHARED_SYSTEM + '\n\nSTORE TOOLS:\n' + self.tools_text() + '\n\nSHOPPER\'S REQUEST:\n'
                            + self.request + opening)
        return self._shared

    def tools_text(self) -> str:
        lines = []
        for tool in self.tools:
            params = json.dumps(tool.get('parameters') or {}, separators=(',', ':'))
            lines.append(f"- {tool['name']}: {tool.get('description') or ''} | parameters: {params}")
        return '\n'.join(lines)

    def map_roles(self) -> dict[str, Any] | None:
        prompt = ROLES_PROMPT.replace('{TOOLS}', self.tools_text())
        got = self.llm(prompt, model=FAST, label='roles', think='off')
        roles = self.check_roles(got)
        if roles:
            return roles
        # no answer, or one that does not fit the schemas: the task cannot go on without roles, ask the stronger model
        return self.check_roles(self.llm(prompt, model=SMART, label='roles', think='low'))

    def check_roles(self, got: Any) -> dict[str, Any] | None:
        if not isinstance(got, dict):
            return None

        def params_of(tool: str) -> dict:
            return ((self.tool_by.get(tool) or {}).get('parameters') or {}).get('properties') or {}

        def ok_args(tool: str, args: Any) -> dict:
            props = params_of(tool)
            clean = {}
            if isinstance(args, dict):
                for key, value in args.items():
                    if key in props:
                        enum = props[key].get('enum')
                        if enum and value not in enum:
                            continue
                        clean[key] = value
            return clean

        roles: dict[str, Any] = {}
        for job, spec in got.items():
            if job == 'extra':
                continue
            if not isinstance(spec, dict) or spec.get('tool') not in self.tool_by:
                roles[job] = None
                continue
            tool = spec['tool']
            props = params_of(tool)
            clean: dict[str, Any] = {'tool': tool}
            for arg in ('text_arg', 'category_arg', 'cursor_arg', 'id_arg', 'qty_arg', 'lines_arg'):
                value = spec.get(arg)
                clean[arg] = value if isinstance(value, str) and value in props else None
            for arg in ('keyword_args', 'browse_args', 'args', 'fixed_args'):
                clean[arg] = ok_args(tool, spec.get(arg))
            clean['qty_means'] = spec.get('qty_means') if spec.get('qty_means') in ('set', 'add', 'replace') else 'set'
            clean['style'] = 'line'
            if spec.get('style') == 'lines' and clean['lines_arg']:
                inner = ((props[clean['lines_arg']].get('items') or {}).get('properties') or {})
                clean['line_id_key'] = spec.get('line_id_key') if spec.get('line_id_key') in inner else None
                clean['line_qty_key'] = spec.get('line_qty_key') if spec.get('line_qty_key') in inner else None
                if clean['line_id_key']:
                    clean['style'] = 'lines'
            if clean['category_arg']:
                clean['categories'] = props[clean['category_arg']].get('enum') or []
            roles[job] = clean
        if not roles.get('search') or not roles.get('cart') or not roles.get('buy'):
            return None
        for job in ROLE_JOBS:
            roles.setdefault(job, None)
        if roles['search'].get('text_arg') is None and roles['search'].get('category_arg') is None:
            return None
        if roles.get('open') and not roles['open'].get('id_arg'):
            roles['open'] = None
        if roles['cart'].get('style') == 'line' and roles['cart'].get('id_arg') is None:
            return None
        roles['extra'] = got.get('extra') if isinstance(got.get('extra'), list) else []
        return roles

    def understand(self) -> dict[str, Any] | None:
        categories = []
        for tool in self.tools:
            for prop in ((tool.get('parameters') or {}).get('properties') or {}).values():
                if isinstance(prop, dict) and isinstance(prop.get('enum'), list) and len(prop['enum']) > 4:
                    categories = prop['enum']
        opening = '' if self.opening is None else '\nOPENING NOTE FROM THE STORE:\n' + _clip(self.opening, 3000)
        prompt = PLAN_PROMPT.replace('{CATS}', json.dumps(categories, ensure_ascii=False))
        got = self.llm(prompt, model=SMART, label='plan', think='low', system=self.shared_system())
        if not isinstance(got, dict) or not isinstance(got.get('items'), list) or not got['items']:
            got = self.llm(prompt, model=FAST, label='plan_retry', think='low', system=self.shared_system())
        if not isinstance(got, dict) or not isinstance(got.get('items'), list):
            return None
        items = []
        for index, item in enumerate(got['items']):
            if not isinstance(item, dict):
                continue
            key = chr(ord('A') + len(items))
            quantity = item.get('quantity')
            items.append({
                'key': key,
                'name': str(item.get('name') or f'item {index + 1}'),
                'quantity': int(quantity) if isinstance(quantity, (int, float)) and quantity >= 1 else 1,
                'category': item.get('category') if item.get('category') in categories else '',
                'shelf': _clip(item.get('shelf') or '', 80) if isinstance(item.get('shelf'), str) else '',
                'musts': _str_list(item.get('musts')),
                'wishes': _str_list(item.get('wishes')),
                'queries': _str_list(item.get('queries'))[:4],
                'depends_on': _str_list(item.get('depends_on')),
            })
        for item in items:
            for other in items:
                if other is item or other['key'] in item['depends_on']:
                    continue
                name = other['name'].lower()
                if len(name) > 3 and any(name in m.lower() for m in item['musts']):
                    item['depends_on'].append(other['key'])
        got['items'] = items
        for field in ('rules', 'process', 'references', 'conditions'):
            value = got.get(field)
            got[field] = [v for v in value if v] if isinstance(value, list) else ([value] if value else [])
        got['process'] = [p for p in got['process'] if isinstance(p, dict)]
        got['references'] = [r for r in got['references'] if isinstance(r, dict)]
        if not isinstance(got.get('budget'), dict):
            got['budget'] = {}
        limit = got.get('message_limit')
        self.message_limit = int(limit) if isinstance(limit, (int, float)) else None
        return got

    # ------------------------------------------------------------------ searching
    def search(self, item: dict[str, Any], words: str | None, cursor: Any = None, browse: bool = False) -> dict:
        role = self.roles['search']
        args: dict[str, Any] = {}
        if browse or not role.get('text_arg'):
            args.update(role.get('browse_args') or {})
        else:
            args.update(role.get('keyword_args') or {})
            args[role['text_arg']] = words or item['name']
        if role.get('category_arg') and item.get('category'):
            args[role['category_arg']] = item['category']
        elif browse and role.get('category_arg'):
            return {'rows': [], 'error': 'no category'}
        if cursor is not None and role.get('cursor_arg'):
            args[role['cursor_arg']] = cursor
            self.paged.add((words, browse, str(cursor)))
        props = ((self.tool_by[role['tool']].get('parameters') or {}).get('properties') or {})
        if role.get('text_arg') and role['text_arg'] in args:
            limit = props.get(role['text_arg'], {}).get('maxLength')
            if isinstance(limit, int):
                args[role['text_arg']] = str(args[role['text_arg']])[:limit]
        observation = self.act_fixed(role['tool'], args)
        rows = self.read_rows(observation)
        for row in rows:
            rid = row['id']
            self.rows[rid] = row
            self.seen_by_item.setdefault(item['key'], [])
            if rid not in self.seen_by_item[item['key']]:
                self.seen_by_item[item['key']].append(rid)
        total = _num(_path_get(observation, self.schema.get('total_key'))) if isinstance(observation, dict) else None
        nxt = _path_get(observation, self.schema.get('cursor_key')) if isinstance(observation, dict) else None
        entry = {'item': item['key'], 'words': words, 'category': item.get('category') or '', 'rows': len(rows),
                 'total': total, 'cursor': nxt, 'browse': browse, 'page': cursor is not None}
        if rows == [] and observation is not None and not isinstance(observation, dict):
            entry['note'] = _clip(observation, 200)
        elif isinstance(observation, dict) and not rows:
            entry['note'] = _clip(observation, 200)
        self.queries.append(entry)
        return {'rows': rows, 'total': total, 'cursor': nxt}

    def learn_schema(self, observation: Any) -> None:
        if self.schema.get('id_key') or not isinstance(observation, (dict, list)):
            return
        lists = _dict_lists(observation)
        if not lists:
            return
        sample = max(lists, key=len)[:3]
        prompt = SCHEMA_PROMPT.replace('{OBS}', _clip(observation if isinstance(observation, dict) else
                                                         {'rows': sample}, 3500))
        got = self.llm(prompt, model=FAST, label='schema', think='off')
        if isinstance(got, dict):
            for key in ('id_key', 'title_key', 'price_key', 'stock_key', 'cursor_key', 'total_key', 'rows_key'):
                value = got.get(key)
                self.schema[key] = value if isinstance(value, str) and value else None
        if not self.schema.get('id_key'):
            for key, value in sample[0].items():
                if _looks_like_id(value):
                    self.schema['id_key'] = key
                    break

    def read_rows(self, observation: Any) -> list[dict]:
        if not isinstance(observation, (dict, list)):
            return []
        if not self.schema.get('id_key'):
            self.learn_schema(observation)
        lists = []
        rows_key = self.schema.get('rows_key')
        if rows_key and isinstance(observation, dict):
            listed = _path_get(observation, rows_key)
            if isinstance(listed, list):
                listed = [_text_record(r) or {'text': r} if isinstance(r, str) else r for r in listed]
                lists = [listed]
        if not lists:
            lists = _dict_lists(observation)
        out = []
        id_key = self.schema.get('id_key')
        for listed in lists:
            for row in listed:
                if not isinstance(row, dict):
                    continue
                rid = _path_get(row, id_key) if id_key else None
                if not isinstance(rid, (str, int)):
                    rid = next((v for v in row.values() if _looks_like_id(v)), None)
                if rid is None:
                    continue
                out.append({'id': str(rid), 'price': _num(_path_get(row, self.schema.get('price_key'))),
                            'raw': row})
            if out:
                break
        return out

    def open_listing(self, rid: str) -> dict | None:
        role = self.roles.get('open')
        if not role:
            return None
        args = dict(role.get('fixed_args') or {})
        args[role['id_arg']] = rid
        observation = self.act_fixed(role['tool'], args)
        if observation is None:
            return None
        record = observation if isinstance(observation, dict) else (_text_record(observation) or {'text': observation})
        self.records[rid] = record
        fresh = _num(_path_get(record, self.schema.get('price_key'))) if self.schema.get('price_key') else None
        if fresh is not None:
            self.price_seen[rid] = fresh
        offer = self.roles.get('offer')
        if offer and offer.get('id_arg') and self.steps_left() > 12:
            extra = dict(offer.get('fixed_args') or {})
            extra[offer['id_arg']] = rid
            seen = self.act(offer['tool'], extra)
            if seen is not None:
                record = dict(record)
                record['_offer'] = seen
                self.records[rid] = record
        return record

    def row_text(self, rid: str) -> str:
        row = self.rows.get(rid) or {}
        return _clip(row.get('raw') or row, ROW_CHARS)

    def record_text(self, rid: str) -> str:
        record = self.records.get(rid)
        if record is None:
            return 'NOT OPENED; search row: ' + self.row_text(rid)
        return _clip(record, RECORD_CHARS)

    def absorb_prices(self, observation: Any) -> None:
        """Remember the newest price the store shows for a listing (cart lines, refused checkouts, pages)."""
        key = self.schema.get('price_key')
        if not key or not isinstance(observation, (dict, list)):
            return
        known = set(self.rows) | set(self.records)
        for listed in _dict_lists(observation):
            for row in listed:
                if not isinstance(row, dict):
                    continue
                rid = next((v for v in row.values() if isinstance(v, str) and v in known), None)
                price = _num(_path_get(row, key))
                if rid is not None and price is not None:
                    self.price_seen[rid] = price

    def price_of(self, rid: str) -> float | None:
        if rid in self.price_seen:
            return self.price_seen[rid]
        record = self.records.get(rid)
        price = None
        if isinstance(record, dict) and self.schema.get('price_key'):
            price = _num(_path_get(record, self.schema['price_key']))
            if price is None and isinstance(record.get('_offer'), dict):
                price = _num(_path_get(record['_offer'], self.schema['price_key']))
        if price is None:
            price = (self.rows.get(rid) or {}).get('price')
        verdict = self.best_verdict(rid)
        if price is None and verdict and isinstance(verdict.get('price'), (int, float)):
            price = float(verdict['price'])
        return price

    def best_verdict(self, rid: str) -> dict | None:
        for per_item in list(self.verdicts.values()):
            if rid in per_item:
                return per_item[rid]
        return None

    # ------------------------------------------------------------------ item briefs
    def needs_of(self, item: dict[str, Any]) -> list[str]:
        return list(item['musts'])

    def item_text(self, item: dict[str, Any]) -> str:
        parts = [f"[{item['key']}] {item['name']} x{item['quantity']}"]
        if item.get('category'):
            parts.append(f"category: {item['category']}")
        elif item.get('shelf'):
            parts.append(f"store category named by the shopper: {item['shelf']}")
        needs = self.needs_of(item)
        parts.append('requirements: ' + ('; '.join(needs) if needs else 'none stated'))
        if item.get('wishes'):
            parts.append('wishes: ' + '; '.join(item['wishes']))
        return ' | '.join(parts)

    def rules_text(self) -> str:
        rules = [str(r) for r in self.plan.get('rules') or []]
        return '\n'.join(f'- {r}' for r in rules) if rules else '- none'

    # ------------------------------------------------------------------ exploration
    def first_searches(self) -> None:
        for item in self.items:
            if self.steps_left() < self.reserve() + 2 or self.left() < 150:
                break
            queries = item['queries'] or [item['name']]
            result = self.search(item, queries[0])
            tries = 1
            while not result['rows'] and tries < len(queries) and tries < 3 and self.steps_left() > self.reserve() + 2:
                result = self.search(item, queries[tries])
                tries += 1
            # a roomy step budget: one more phrasing up front widens the shelf the screening sees
            roomy = self.max_steps >= 10 * len(self.items) + 30
            shelf = str(item.get('shelf') or '')
            if shelf and not item.get('category') and shelf.lower() not in [q.lower() for q in queries[:tries]] and \
                    self.steps_left() > self.reserve() + 10:
                # the shelf the shopper named, searched by its name, shows what the store files there
                self.search(item, shelf[:60])
            elif roomy and result['rows'] and tries < len(queries) and self.steps_left() > self.reserve() + 10:
                self.search(item, queries[tries])
            if not result['rows'] and self.roles['search'].get('category_arg') and item.get('category'):
                self.search(item, None, browse=True)

    def reserve(self) -> int:
        """Steps kept back for carting, the checks and the finish."""
        messages_left = self.messages_left()
        return len(self.items) * 2 + 4 + min(messages_left, 3)

    def messages_left(self) -> int:
        if not self.roles.get('ask'):
            return 0
        if self.message_limit is None:
            return max(0, 2 - self.messages_used)
        return max(0, self.message_limit - self.messages_used)

    def screen(self, item: dict[str, Any], limit: int) -> dict:
        """Model picks which listings to open next for one item, and new searches if the shelf looks wrong."""
        seen = self.seen_by_item.get(item['key'], [])
        opened = [rid for rid in seen if rid in self.records]
        fresh = [rid for rid in seen if rid not in self.records]
        cheapest = self.wants_cheapest()
        # cheaper rows first: the cheapest acceptable choice wins on such tasks, and budgets bind on the others
        fresh.sort(key=lambda r: (self.rows.get(r, {}).get('price') is None, self.rows.get(r, {}).get('price') or 0))
        lines = [f'{rid} | {self.row_text(rid)}' for rid in fresh[:40]]
        verdict_lines = []
        for rid in opened:
            verdict = (self.verdicts.get(item['key']) or {}).get(rid)
            if verdict:
                verdict_lines.append(f"{rid}: {verdict.get('verdict')} ({_clip(verdict.get('why') or '', 160)})")
        history = [f"{q['words']!r} in {q['category'] or 'all'} -> {q['rows']} rows (total {q['total']})"
                   for q in self.queries if q['item'] == item['key']]
        hint = f"\nWhy more listings are needed now: {item['_hint']}" if item.get('_hint') else ''
        prompt = (SCREEN_PROMPT.replace('{ITEM}', self.item_text(item) + hint)
                  .replace('{RULES}', self.rules_text())
                  .replace('{CHEAP}', 'yes: the cheapest acceptable choice wins' if cheapest else
                           'no, but the budget binds: cheaper listings matter' if self.budget_short() else 'no')
                  .replace('{ROWS}', '\n'.join(lines) or '(none)')
                  .replace('{CHECKED}', '\n'.join(verdict_lines) or '(none)')
                  .replace('{HISTORY}', '\n'.join(history) or '(none)')
                  .replace('{LIMIT}', str(limit))
                  .replace('{SEARCH_NOTE}', self.search_note()))
        got = self.llm(prompt, model=FAST, label=f'screen {item["key"]}', think='off')
        if not isinstance(got, dict):
            return {'open': fresh[:limit], 'queries': []}
        picks = [str(r) for r in got.get('open') or [] if str(r) in self.rows and str(r) not in self.records]
        shelf = got.get('shelf')
        if isinstance(shelf, str) and shelf.strip() and not item.get('category') and not item.get('shelf') and \
                any(shelf.strip().lower() in self.row_text(r).lower() for r in seen):
            item['shelf'] = _clip(shelf.strip(), 80)
            self.log('shelf', item=item['key'], shelf=item['shelf'])
        if self.price_first():
            # in price order, every listing the model did not rule out from its row comes first: a title that reads
            # oddly is no reason to pass over the cheapest listing of the right shelf
            skip = {str(r) for r in got.get('skip') or []}
            passing = [self.price_of(r) for r in self.passing(item)]
            best = min([p for p in passing if p is not None], default=None)
            ordered = [r for r in fresh if r not in skip and r in self.rows and
                       (best is None or (self.rows[r].get('price') or 1e18) < best)]
            picks = ordered[:limit] + [r for r in picks if r not in ordered[:limit]]
        queries = [str(q) for q in got.get('queries') or [] if isinstance(q, str) and q.strip()]
        return {'open': picks[:limit], 'queries': queries[:2], 'more_pages': bool(got.get('more_pages'))}

    def search_note(self) -> str:
        role = self.roles['search']
        tool = self.tool_by[role['tool']]
        note = f"Search tool {role['tool']}: {tool.get('description') or ''}"
        stats = [f"{q['words']!r} -> {q['total'] if q['total'] is not None else q['rows']}" for q in self.queries[-12:]
                 if not q.get('page')]
        if stats:
            note += '\nSearches so far (words -> matches): ' + '; '.join(stats)
        return note

    def wants_cheapest(self) -> bool:
        return str(self.plan.get('cheapest') or 'none').lower() not in ('none', 'no', 'false', '', 'null')

    def budget_short(self) -> bool:
        """The cheapest passing listings found so far break a budget stated in the prices' unit."""
        budget = self.plan.get('budget') or {}
        limit = _num(budget.get('limit'))
        if limit is None or 'cent' not in str(budget.get('unit') or '').lower():
            return False
        total = 0.0
        for item in self.items:
            prices = [p for p in (self.price_of(r) for r in self.passing(item)) if p is not None]
            if not prices:
                return False
            if budget.get('scope') == 'each' and min(prices) > limit:
                return True
            total += min(prices) * item['quantity']
        return budget.get('scope') != 'each' and total > limit

    def price_first(self) -> bool:
        """Open listings in price order: the cheapest acceptable choice wins, or the budget binds."""
        return self.wants_cheapest() or self.budget_short()

    def needs_more(self, item: dict[str, Any]) -> bool:
        verdicts = self.verdicts.get(item['key']) or {}
        passing = [rid for rid, v in verdicts.items() if v.get('verdict') == 'pass' and v.get('in_stock') is not False]
        if not passing:
            return True
        if self.price_first():
            # keep looking while unopened listings cheaper than the cheapest passing one remain
            best = min((self.price_of(r) or 1e18) for r in passing)
            cheaper = [r for r in self.seen_by_item.get(item['key'], []) if r not in self.records
                       and (self.rows.get(r, {}).get('price') or 1e18) < best]
            return bool(cheaper) and not item.get('_cheap_done')
        return False

    def passing(self, item: dict[str, Any]) -> list[str]:
        return [rid for rid, v in (self.verdicts.get(item['key']) or {}).items()
                if v.get('verdict') == 'pass' and v.get('in_stock') is not False]

    def explore_all(self, rounds: int, per_round: int, force: bool = False, floor: float | None = None) -> None:
        """Rounds over every item still lacking a good listing: screen in parallel, open, judge in parallel."""
        searched_only = False          # a last round that only searched gets one more round to open what it found
        for round_no in range(rounds + 1):
            if round_no >= rounds and not searched_only:
                return
            floor = floor if floor is not None else (95 if force else 135)
            if self.steps_left() < self.reserve() + 2 or self.left() < floor or self.over_spend():
                return
            active = [item for item in self.items if self.needs_more(item) or (force and len(self.passing(item)) < 3)]
            if not active:
                return
            picks = self.parallel([(lambda it=it: self.screen(it, per_round)) for it in active])
            searched = False
            for item, pick in zip(active, picks):
                pick = pick or {'open': [], 'queries': []}
                if self.price_first() and pick.get('more_pages') and self.steps_left() > self.reserve() + 6:
                    self.page_more(item)
                    searched = True
                elif self.price_first() and not pick.get('more_pages'):
                    item['_cheap_done'] = round_no >= 1
                if not pick['open'] and pick.get('queries'):
                    for words in pick['queries'][:2]:
                        if self.steps_left() < self.reserve() + 3:
                            break
                        self.search(item, words)
                        searched = True
                for rid in pick['open']:
                    if self.steps_left() < self.reserve() + 1 or self.left() < max(84, floor - 25):
                        break
                    self.open_listing(rid)
            self.parallel([(lambda it=it: self.judge(it)) for it in active])
            opened = any(p and p.get('open') for p in picks)
            searched_only = searched and not opened
            if not searched and not opened:
                return

    def page_small_sets(self) -> None:
        """When the cheapest acceptable choice wins, read every page of a short result list."""
        if not self.wants_cheapest():
            return
        for item in self.items:
            last = next((q for q in reversed(self.queries) if q['item'] == item['key'] and not q.get('page')), None)
            if not last or not last.get('cursor') or not isinstance(last.get('total'), (int, float)):
                continue
            if last['total'] <= 6 * max(1, last['rows']) and self.steps_left() > self.reserve() + 8:
                self.page_more(item)

    def page_more(self, item: dict[str, Any]) -> None:
        if not self.roles['search'].get('cursor_arg'):
            return
        def unread(q: dict) -> bool:
            # a result list whose next page has not been requested yet
            return q['item'] == item['key'] and q.get('cursor') not in (None, '', False) and \
                (q['words'], q.get('browse', False), str(q['cursor'])) not in self.paged
        last = next((q for q in reversed(self.queries) if unread(q)), None)
        pages = 0
        while last and pages < PAGES_CHEAPEST and self.steps_left() > self.reserve() + 4:
            result = self.search(item, last['words'], cursor=last['cursor'], browse=last.get('browse', False))
            pages += 1
            last = self.queries[-1]
            if not result['rows'] or not unread(last):
                break

    def over_spend(self) -> bool:
        return self.cost > SPEND_SOFT

    # ------------------------------------------------------------------ judging
    def judge(self, item: dict[str, Any], rids: list[str] | None = None) -> None:
        if self.left() < 80:
            return
        verdicts = self.verdicts.setdefault(item['key'], {})
        todo = rids or [rid for rid in self.seen_by_item.get(item['key'], []) if rid in self.records
                        and rid not in verdicts]
        if not todo:
            return
        blocks = [f'LISTING {rid}:\n{self.record_text(rid)}' for rid in todo[:6]]
        others = []
        for other in self.items:
            if other['key'] == item['key']:
                continue
            others.append(self.item_text(other))
        partners = []
        for key in item.get('depends_on') or []:
            per = self.verdicts.get(key) or {}
            ranked = sorted(per.items(), key=lambda kv: (kv[1].get('verdict') != 'pass', self.price_of(kv[0]) or 1e18))
            for rid, v in ranked[:4]:
                if v.get('verdict') != 'fail':
                    partners.append(f'[{key}] candidate {rid}: {_clip(self.records.get(rid) or {}, 900)}')
        if partners:
            others.append('CANDIDATES OF THE PRODUCTS THIS ONE MUST WORK WITH (say in "compatible_with" which of '
                          'them each listing works with; compatibility shows as matching model names, part series, '
                          'ranges or sizes named on both pages):\n' + '\n'.join(partners))
        prompt = (JUDGE_PROMPT.replace('{ITEM}', self.item_text(item))
                  .replace('{OTHERS}', '\n'.join(others) or '(none)')
                  .replace('{RULES}', self.rules_text())
                  .replace('{REQUEST}', self.request)
                  .replace('{LISTINGS}', '\n\n'.join(blocks)))
        got = self.llm(prompt, model=FAST, label=f'judge {item["key"]}')
        results = got.get('listings') if isinstance(got, dict) else got if isinstance(got, list) else None
        if not isinstance(results, list):
            return
        for entry in results:
            if not isinstance(entry, dict):
                continue
            rid = str(entry.get('id') or '')
            if rid not in todo:
                continue
            verdict = str(entry.get('verdict') or 'unknown').lower()
            entry['verdict'] = verdict if verdict in ('pass', 'fail', 'unknown') else 'unknown'
            verdicts[rid] = entry
        self.log('judged', item=item['key'], verdicts={r: (v.get('verdict'), _clip(v.get('why') or '', 140))
                                                        for r, v in verdicts.items() if r in todo})
        if len(todo) > 6:
            self.judge(item, todo[6:])

    def resolve_lookups(self, item: dict[str, Any]) -> None:
        """A listing whose value must be read from another listing: follow the pointer, then judge it again."""
        verdicts = self.verdicts.get(item['key']) or {}
        pending = [(rid, v) for rid, v in verdicts.items() if isinstance(v.get('lookup'), dict) and
                   v.get('verdict') != 'fail' and not v.get('_looked')]
        if self.wants_cheapest():
            pending.sort(key=lambda kv: self.price_of(kv[0]) or 1e18)
        for rid, verdict in pending[:2]:
            if self.steps_left() < self.reserve() + 4 or self.left() < 130:
                return
            verdict['_looked'] = True
            look = verdict['lookup']
            prop = str(look.get('property') or '')
            desc = ' '.join(str(look.get(k) or '') for k in ('hints', 'kind', 'category', 'words')).strip()
            found = self.follow_chain(prop, desc, max_hops=3)
            if found and found.get('value') is not None:
                holder = dict(self.records.get(rid) or {})
                holder['_value_from_other_listing'] = {'property': prop, 'value': found['value'],
                                                       'pages': found.get('pages')}
                self.records[rid] = holder
                self.judge(item, [rid])        # overwrites the verdict when the judge answers

    def resolve_references(self) -> None:
        """Requirements whose value is read from other listings' pages (a trail of pointers)."""
        for ref in (self.plan.get('references') or [])[:4]:
            if not isinstance(ref, dict) or ref.get('_done') or self.steps_left() < self.reserve() + 4 \
                    or self.left() < 120:
                continue
            ref['_done'] = True
            item = next((i for i in self.items if i['key'] == ref.get('item')), None)
            if item is None or not ref.get('start'):
                continue
            prop = str(ref.get('property') or '')
            found = self.follow_chain(prop, str(ref['start']), max_hops=5)
            if found and found.get('value') is not None:
                note = f"{prop}: {found['value']} (read from the trail of pages)"
                item['musts'] = [m for m in item['musts'] if not m.lower().startswith(prop.lower() + ':')] + [note]
                self.verdicts.pop(item['key'], None)
            self.log('reference', item=item['key'], prop=prop, found=found)

    def follow_chain(self, prop: str, desc: str, max_hops: int) -> dict | None:
        pages: list[str] = []
        for _ in range(max_hops):
            if self.steps_left() < self.reserve() + 3 or self.left() < 100:
                return None
            hit = self.find_described(desc, prop)
            if not hit:
                return {'value': None, 'pages': pages, 'lost_at': desc}
            rid, read = hit
            pages.append(rid)
            nxt = read.get('next')
            if isinstance(nxt, str) and nxt.strip():
                desc = nxt.strip()
                continue
            return {'value': read.get('value'), 'pages': pages}
        return {'value': None, 'pages': pages}

    def find_described(self, desc: str, prop: str) -> tuple[str, dict] | None:
        """Search for the listing a description names (seller, kind, attributes), open candidates, read the pages."""
        cats = self.roles['search'].get('categories') or []
        prompt = (LOCATE_PROMPT.replace('{DESC}', desc).replace('{CATS}', json.dumps(cats, ensure_ascii=False))
                  .replace('{SEARCH_NOTE}', self.search_note()))
        plan = self.llm(prompt, model=FAST, label='locate', think='off')
        if not isinstance(plan, dict):
            return None
        target = {'key': '_trail', 'name': desc[:60], 'category': plan.get('category') if plan.get('category') in cats
                  else ''}
        tried: set = set()
        for words in [str(q) for q in plan.get('queries') or [] if q][:3]:
            if self.steps_left() < self.reserve() + 3 or self.left() < 60:
                return None
            result = self.search(target, words)
            for _ in range(3):
                rows = [r for r in result['rows'] if r['id'] not in tried]
                if rows:
                    listing = '\n'.join(f"{r['id']} | {_clip(r['raw'], ROW_CHARS)}" for r in rows[:14])
                    got = self.llm(FIND_PROMPT.replace('{WANT}', desc).replace('{ROWS}', listing), model=FAST,
                                   label='pick', think='off')
                    ids = [str(i) for i in ((got or {}).get('ids') or []) if str(i) in self.rows and str(i) not in tried] \
                        if isinstance(got, dict) else []
                    pages = []
                    for rid in ids[:3]:
                        if self.steps_left() < self.reserve() + 2:
                            break
                        tried.add(rid)
                        record = self.records.get(rid) or self.open_listing(rid)
                        if record is not None:
                            pages.append(f'PAGE {rid}:\n{_clip(record, 1800)}')
                    if pages:
                        read = self.llm(POINTER_PROMPT.replace('{PROP}', prop).replace('{DESC}', desc)
                                        .replace('{PAGES}', '\n\n'.join(pages)), model=FAST, label='read_page',
                                        think='off')
                        if isinstance(read, dict) and str(read.get('id') or '') in tried:
                            self.log('trail', want=desc, id=read['id'], read=read)
                            return str(read['id']), read
                if not result.get('cursor') or self.steps_left() < self.reserve() + 4:
                    break
                result = self.search(target, words, cursor=result['cursor'])
        return None

    # ------------------------------------------------------------------ asking
    def ask_round(self) -> None:
        if not self.roles.get('ask') or not self.roles['ask'].get('text_arg'):
            return
        reserved = int(_num(self.plan.get('process_messages')) or 0)
        while self.messages_left() > reserved and self.steps_left() > self.reserve() + 2 and self.left() > 110:
            plan = self.plan_question()
            told = {t['item'] for t in self.told}
            if isinstance(plan, dict) and not plan.get('ask') and any(i['key'] not in told for i in self.items):
                # a product without a told requirement is left while messages remain: one more look
                plan = self.plan_question(note='\nSome products have had no requirement told yet and messages '
                                               'remain: plan a message about one of them, with properties not '
                                               'asked before.')
            if not isinstance(plan, dict) or not plan.get('ask') or \
                    not (str(plan.get('message') or '').strip() or plan.get('properties')):
                return
            item = next((i for i in self.items if i['key'] == plan.get('item')), None)
            props = self.fresh_properties(item, plan.get('properties'))
            if item is not None and props:
                plan['message'] = self.question_text(item, props[:1] if self.single_mode else props[:4])
            elif item is not None and plan.get('properties'):
                return                         # everything it would ask was already answered or stated
            if not str(plan.get('message') or '').strip():
                return
            reply = self.send_message(str(plan['message']))
            told_before = len(self.told)
            got = self.read_reply(str(plan['message']), reply, plan)
            if item is not None and props and len(self.told) == told_before and isinstance(got, dict) \
                    and got.get('no_preference') and not got.get('confused'):
                # "don't mind" answers the first property of the message the shopper tracks: most likely the first
                self.answered_none.setdefault(item['key'], []).append(props[0])

    def fresh_properties(self, item: dict[str, Any] | None, props: Any) -> list[str]:
        """The planner's properties minus those the shopper already answered or the item's requirements state."""
        if item is None or not isinstance(props, list):
            return []
        known = {t['property'].lower().strip() for t in self.told if t.get('item') == item['key'] and t.get('property')}
        known |= {m.split(':', 1)[0].lower().strip() for m in item['musts'] if ':' in m}
        asked = {p.lower().strip() for p in self.answered_none.get(item['key'], [])}
        out = []
        for prop in props:
            # a record path ("details.Frame material") names the property by its last part
            name = re.sub(r'\s+', ' ', str(prop).rsplit('.', 1)[-1].replace('_', ' ')).strip(' .?')
            low = name.lower()
            if name and low not in known and low not in asked and low not in [o.lower() for o in out]:
                out.append(name)
        return out

    @staticmethod
    def question_text(item: dict[str, Any], props: list[str]) -> str:
        first = f"For the {item['name']}, which {props[0]} do you need?"
        if len(props) == 1:
            return first
        rest = props[1:]
        tail = rest[0] if len(rest) == 1 else ', the '.join(rest[:-1]) + ' or the ' + rest[-1]
        return f"{first} Or does the {tail} matter?"

    def property_inventory(self, item: dict[str, Any]) -> str:
        values: dict[str, set] = {}
        carried: dict[str, int] = {}
        opened = 0
        for rid in self.seen_by_item.get(item['key'], []):
            record = self.records.get(rid)
            if not isinstance(record, dict):
                continue
            opened += 1
            for key, value in _flatten(record).items():
                if len(values) > 70 and key not in values:
                    continue
                values.setdefault(key, set()).add(_clip(value, 40))
                carried[key] = carried.get(key, 0) + 1
        if not values:
            return '(no listing opened yet)'
        lines = []
        for key in sorted(values, key=lambda k: -carried[k])[:18]:
            lines.append(f"{key} (on {carried[key]} of {opened} pages): {', '.join(sorted(values[key])[:4])}")
        return '\n'.join(lines)

    def ask_targets(self) -> list[str]:
        told = {t['item'] for t in self.told}
        asked = set(self.asked_items)
        fresh = [i['key'] for i in self.items if i['key'] not in asked and i['key'] not in told]
        if fresh:
            return fresh
        untold = [i['key'] for i in self.items if i['key'] not in told]
        return untold or [i['key'] for i in self.items]

    def plan_question(self, note: str = '') -> dict | None:
        blocks = []
        low = self.request.lower()
        for item in self.items:
            mentions = low.count(item['name'].lower()) if len(item['name']) > 3 else 0
            blocks.append(self.item_text(item) + f' | the request names it {mentions} time(s)'
                          + '\nproperties on its listings:\n' + self.property_inventory(item))
        convo = '\n'.join(f"{d['who']}: {d['text']}" for d in self.dialogue) or '(no messages yet)'
        prompt = (ASK_PROMPT
                  .replace('{ITEMS}', '\n\n'.join(blocks))
                  .replace('{CONVO}', convo)
                  .replace('{LEFT}', str(self.messages_left() - int(_num(self.plan.get('process_messages')) or 0)))
                  .replace('{TARGETS}', ', '.join(self.ask_targets()))
                  .replace('{SINGLE}', '\nThe shopper wants a single question per message: name exactly ONE property, your best '
                           'guess.' if self.single_mode else '')) + note
        model = SMART if self.messages_used == 0 else FAST
        got = self.llm(prompt, model=model, label='ask_plan', think='low' if model == SMART else None,
                       system=self.shared_system())
        if isinstance(got, dict) and got.get('item'):
            self.asked_items.append(str(got['item']))
        self.log('ask_plan', plan=got)
        return got if isinstance(got, dict) else None

    def send_message(self, text: str) -> str:
        role = self.roles['ask']
        args = dict(role.get('fixed_args') or {})
        limit = (((self.tool_by[role['tool']].get('parameters') or {}).get('properties') or {})
                 .get(role['text_arg']) or {}).get('maxLength')
        args[role['text_arg']] = text[:limit] if isinstance(limit, int) else text
        observation = self.act_fixed(role['tool'], args)
        self.messages_used += 1
        reply = observation if isinstance(observation, str) else _clip(observation, 1200)
        self.dialogue.append({'who': 'agent', 'text': text})
        self.dialogue.append({'who': 'shopper', 'text': str(reply)})
        return str(reply)

    def read_reply(self, question: str, reply: str, plan: dict) -> dict | None:
        items = '\n'.join(self.item_text(i) for i in self.items)
        prompt = (REPLY_PROMPT.replace('{ITEMS}', items).replace('{QUESTION}', question)
                  .replace('{REPLY}', reply).replace('{REQUEST}', self.request))
        got = self.llm(prompt, model=FAST, label='reply', think='off')
        if not isinstance(got, dict):
            return
        keys = {i['key'] for i in self.items}
        for told in got.get('told') or []:
            if isinstance(told, dict) and told.get('item') in keys and told.get('requirement'):
                entry = {'item': told['item'], 'property': str(told.get('property') or ''),
                         'requirement': str(told['requirement'])}
                self.told.append(entry)
                item = next(i for i in self.items if i['key'] == entry['item'])
                must = f"{entry['property']}: {entry['requirement']}" if entry['property'] else entry['requirement']
                if must not in item['musts']:
                    item['musts'].append(must)
                self.verdicts.pop(entry['item'], None)
        for rule in got.get('rules') or []:
            if isinstance(rule, str) and rule.strip():
                self.plan.setdefault('rules', []).append(rule.strip())
        if isinstance(got.get('message_limit'), (int, float)):
            self.message_limit = int(got['message_limit'])
        if got.get('single_only'):
            self.single_mode = True
        for ref in got.get('references') or []:
            if isinstance(ref, dict) and ref.get('item') in keys and ref.get('start'):
                self.plan.setdefault('references', []).append(dict(ref, _new=True))
        self.log('reply', reply=reply, read=got)
        return got

    # ------------------------------------------------------------------ choosing and finishing
    def candidates_text(self) -> str:
        blocks = []
        for item in self.items:
            verdicts = self.verdicts.get(item['key']) or {}
            lines = []
            ranked = sorted(verdicts.items(), key=lambda kv: (kv[1].get('verdict') != 'pass',
                                                              self.price_of(kv[0]) or 1e18))
            for rid, v in ranked[:5]:
                price = self.price_of(rid)
                compat = f" | works with {v.get('compatible_with')}" if v.get('compatible_with') else ''
                lines.append(f"  {rid} | price {price} | {v.get('verdict')} | stock {v.get('in_stock')} | "
                             f"{_clip(v.get('why') or '', 150)} | facts {_clip(v.get('facts') or {}, 150)}{compat}")
            blocks.append(self.item_text(item) + '\n' + ('\n'.join(lines) or '  (no listing checked)'))
        return '\n'.join(blocks)

    def last_resort(self) -> dict | None:
        """No time left for the chooser: the best listing seen for each product (pass, then unknown; cheapest)."""
        lines = []
        for item in self.items:
            per = self.verdicts.get(item['key']) or {}
            ranked = sorted((r for r, v in per.items() if v.get('verdict') != 'fail' and v.get('in_stock') is not False),
                            key=lambda r: (per[r].get('verdict') != 'pass', self.price_of(r) or 1e18))
            if not ranked:
                return None
            lines.append({'item': item['key'], 'id': ranked[0], 'quantity': item['quantity']})
        decision = {'action': 'order', 'lines': lines, 'why': 'last resort: best listing per product'}
        self.log('choose', decision=decision, by='last_resort')
        return decision

    def auto_choose(self) -> dict | None:
        """A plain basket (no rules between products, no conditions, no notices): the cheapest passing listing per
        product, when the total is clearly within a budget stated in the same unit as the prices."""
        if self.plan.get('conditions') or self.plan.get('process') or self.events or self.problems:
            return None
        if any(i.get('depends_on') for i in self.items):
            return None
        lines, total = [], 0.0
        for item in self.items:
            priced = [(self.price_of(r), r) for r in self.passing(item)]
            if not priced or any(p is None for p, _ in priced):
                return None
            price, rid = min(priced)
            lines.append({'item': item['key'], 'id': rid, 'quantity': item['quantity']})
            total += price * item['quantity']
        budget = self.plan.get('budget') or {}
        limit = budget.get('limit')
        if isinstance(limit, (int, float)):
            if budget.get('scope') == 'each' or 'cent' not in str(budget.get('unit') or '').lower():
                return None
            if total > limit:
                return None
        decision = {'action': 'order', 'lines': lines, 'total': total, 'why': 'cheapest passing listing per product'}
        self.log('choose', decision=decision, by='code')
        return decision

    def choose(self, final: bool = False) -> dict | None:
        events = _clip([{k: v for k, v in e.items() if k != '_read'} for e in self.events[-6:]] + self.problems[-3:],
                       1800) if (self.events or self.problems) else '(none)'
        convo = '\n'.join(f"{d['who']}: {d['text']}" for d in self.dialogue) or '(none)'
        prompt = (CHOOSE_PROMPT
                  .replace('{RULES}', self.rules_text())
                  .replace('{CANDIDATES}', self.candidates_text())
                  .replace('{CONVO}', convo)
                  .replace('{EVENTS}', events)
                  .replace('{STEPS}', str(self.steps_left()))
                  .replace('{TIME}', str(int(self.left()))))
        if final:
            prompt += '\nThis is the last decision: answer "order" or "decline" only.'
        got = self.llm(prompt, model=FAST, label='choose', system=self.shared_system())
        if isinstance(got, dict) and str(got.get('action') or '').lower() == 'decline' and self.left() > 45:
            # giving up loses the task whenever an order was possible: the stronger model looks again
            second = self.llm(prompt + '\nA first reading wanted to decline: ' + _clip(got.get('why') or '', 300)
                              + '\nCheck that reading; decline only when the request truly cannot be met.',
                              model=SMART, label='choose_check', think='low', system=self.shared_system())
            if isinstance(second, dict) and second.get('action'):
                got = second
        self.log('choose', decision=got)
        return got if isinstance(got, dict) else None

    def set_line(self, rid: str, quantity: int) -> Any:
        role = self.roles['cart']
        args = dict(role.get('fixed_args') or {})
        args[role['id_arg']] = rid
        if role.get('qty_arg'):
            props = ((self.tool_by[role['tool']].get('parameters') or {}).get('properties') or {})
            spec = props.get(role['qty_arg']) or {}
            have = self.cart_lines.get(rid, 0)
            value = quantity if role.get('qty_means') == 'set' else quantity - have
            if value == 0 and role.get('qty_means') != 'set':
                return None
            if isinstance(spec.get('maximum'), int):
                value = min(value, spec['maximum'])
            args[role['qty_arg']] = value
            observation = self.act_fixed(role['tool'], args)
            self.cart_lines[rid] = quantity
            return observation
        observation = None
        added = 0
        for _ in range(max(0, quantity - self.cart_lines.get(rid, 0))):
            if self.steps_left() < 3:
                break
            observation = self.act(role['tool'], args)
            added += 1
        self.cart_lines[rid] = self.cart_lines.get(rid, 0) + added
        return observation

    def remove_line(self, rid: str) -> Any:
        remove = self.roles.get('remove')
        if remove and remove.get('id_arg'):
            args = dict(remove.get('fixed_args') or {})
            args[remove['id_arg']] = rid
            observation = self.act(remove['tool'], args)
            self.cart_lines.pop(rid, None)
            return observation
        if self.roles['cart'].get('qty_arg') and self.roles['cart'].get('qty_means') == 'set':
            observation = self.set_line(rid, 0)
            self.cart_lines.pop(rid, None)
            return observation
        return None

    def view_cart(self) -> Any:
        role = self.roles.get('cart_view')
        if not role:
            return None
        observation = self.act(role['tool'], dict(role.get('fixed_args') or {}))
        self.absorb_prices(observation)
        return observation

    def fill_lines(self, wanted: dict[str, int]) -> list[dict]:
        """A cart tool that takes a list of lines in one call."""
        role = self.roles['cart']
        props = ((self.tool_by[role['tool']].get('parameters') or {}).get('properties') or {})
        inner = ((props.get(role['lines_arg']) or {}).get('items') or {}).get('properties') or {}
        qty_spec = inner.get(role.get('line_qty_key') or '') or {}
        batch: dict[str, int] = {}
        if role.get('qty_means') == 'replace':
            batch = dict(wanted)
        else:
            for rid, quantity in wanted.items():
                have = self.cart_lines.get(rid, 0)
                if have != quantity:
                    batch[rid] = quantity - have if role.get('qty_means') == 'add' else quantity
            for rid in self.cart_lines:
                if rid not in wanted and role.get('qty_means') == 'set':
                    batch[rid] = 0
        out_lines = []
        for rid, quantity in batch.items():
            line: dict[str, Any] = {role['line_id_key']: rid}
            if role.get('line_qty_key'):
                low, high = qty_spec.get('minimum'), qty_spec.get('maximum')
                if isinstance(low, int) and quantity < low:
                    if role.get('qty_means') != 'replace':
                        continue
                    quantity = low
                if isinstance(high, int):
                    quantity = min(quantity, high)
                line[role['line_qty_key']] = quantity
            out_lines.append(line)
        if not out_lines:
            return []
        args = dict(role.get('fixed_args') or {})
        args[role['lines_arg']] = out_lines
        observation = self.act_fixed(role['tool'], args)
        for rid in list(self.cart_lines):
            if rid not in wanted and role.get('qty_means') in ('replace', 'set'):
                self.cart_lines.pop(rid, None)
        self.cart_lines.update(wanted)
        return [{'lines': out_lines, 'reply': _clip(observation, 600)}]

    def fill_cart(self, lines: list[dict]) -> list[dict]:
        replies = []
        wanted = {str(l['id']): int(l.get('quantity') or 1) for l in lines}
        if self.roles['cart'].get('style') == 'lines':
            return self.fill_lines(wanted)
        for rid in list(self.cart_lines):
            if rid not in wanted:
                replies.append({'id': rid, 'removed': _clip(self.remove_line(rid), 300)})
        for rid, quantity in wanted.items():
            if self.cart_lines.get(rid) == quantity:
                continue
            if self.steps_left() < 2:
                break
            replies.append({'id': rid, 'quantity': quantity, 'reply': _clip(self.set_line(rid, quantity), 400)})
        return replies

    def check_cart(self, lines: list[dict], replies: list[dict]) -> dict | None:
        events_before = len(self.events)
        cart = self.view_cart() if self.steps_left() > 2 else None
        known = set(self.rows) | set(self.records) | {str(l['id']) for l in lines}
        shown = {v for v in _strings(cart) if v in known} if cart is not None else set()
        planned = {str(l['id']) for l in lines}
        if cart is not None and shown == planned and len(self.events) == events_before:
            self.log('cart_check', cart=_clip(cart, 600), check='ids match the plan')
            return {'ok': True}
        prompt = (CART_PROMPT.replace('{REQUEST}', self.request)
                  .replace('{RULES}', self.rules_text())
                  .replace('{LINES}', json.dumps(lines, ensure_ascii=False))
                  .replace('{REPLIES}', _clip(replies, 2500))
                  .replace('{CART}', _clip(cart, 3000))
                  .replace('{EVENTS}', _clip(self.events[-6:], 1500) if self.events else '(none)'))
        got = self.llm(prompt, model=FAST, label='cart_check', think='off')
        self.log('cart_check', cart=_clip(cart, 600), check=got)
        return got if isinstance(got, dict) else None

    def process_step(self, when: str) -> None:
        steps = [p for p in self.plan.get('process') or [] if isinstance(p, dict) and p.get('when') == when
                 and not p.get('_done')]
        for step in steps:
            step['_done'] = True
            if self.left() < 25 or self.steps_left() < 3:
                return
            if int(_num(step.get('messages')) or 0) > 0 and self.roles.get('ask') and self.messages_left() > 0:
                prompt = (PROCESS_PROMPT.replace('{REQUEST}', self.request).replace('{STEP}', _clip(step, 400))
                          .replace('{CART}', _clip(self.cart_summary(), 1500))
                          .replace('{CONVO}', '\n'.join(f"{d['who']}: {d['text']}" for d in self.dialogue)
                                   or '(none)'))
                got = self.llm(prompt, model=FAST, label='process', think='off')
                text = str((got or {}).get('message') or '') if isinstance(got, dict) else ''
                if text.strip():
                    reply = self.send_message(text)
                    self.log('process', step=step, reply=reply)
                    self.read_reply(text, reply, {})

    def cart_summary(self) -> list[dict]:
        out = []
        for rid, quantity in self.cart_lines.items():
            out.append({'id': rid, 'quantity': quantity, 'price': self.price_of(rid),
                        'listing': self.row_text(rid)})
        return out

    def finish(self, buy: bool) -> Any:
        role = self.roles['buy'] if buy else (self.roles.get('refuse') or None)
        if role is None:
            return None
        args = dict(role.get('args') or role.get('fixed_args') or {})
        observation = self.act_fixed(role['tool'], args)
        self.absorb_prices(observation)
        self.log('finish', buy=buy, obs=_clip(observation, 600), finished=self.finished)
        return observation

    # ------------------------------------------------------------------ main flow
    def run(self) -> None:
        roles, plan = self.parallel([self.map_roles, self.understand])
        if not roles or not plan:
            self.log('setup_failed', roles=roles, plan=plan)
            fallback_loop(self)
            return
        self.roles, self.plan, self.items = roles, plan, plan['items']
        self.log('setup', roles=roles, plan=plan)
        self.first_searches()
        self.page_small_sets()
        self.resolve_references()
        # a first look at each item so the questions can name the properties its listings carry
        look = 3 if len(self.items) <= 2 else 2
        picks = self.parallel([(lambda it=it: self.screen(it, look)) for it in self.items])
        for item, pick in zip(self.items, picks):
            for rid in (pick or {}).get('open', [])[:look]:
                if self.steps_left() < self.reserve() + 2 or self.left() < 140:
                    break
                self.open_listing(rid)
        self.ask_round()
        self.resolve_references()
        told_items = [item for item in self.items if any(t['item'] == item['key'] for t in self.told)]
        queries = self.parallel([(lambda it=it: self.requery(it)) for it in told_items])
        for item, words in zip(told_items, queries):
            for w in (words or [])[:1]:
                if self.steps_left() > self.reserve() + 3:
                    self.search(item, w)
        self.parallel([(lambda it=it: self.judge(it)) for it in self.items])
        self.explore_all(rounds=3, per_round=OPEN_PER_ITEM)
        for item in self.items:
            self.resolve_lookups(item)
        self.complete()

    def search_changed(self) -> None:
        """Products whose requirements changed: new searches, since earlier ones were shaped by the old wording."""
        changed = [i for i in self.items if i.pop('_requery', None)]
        if not changed or self.left() < 80:
            return
        queries = self.parallel([(lambda it=it: self.requery(it)) for it in changed])
        for item, words in zip(changed, queries):
            for w in (words or [])[:1]:
                if self.steps_left() > self.reserve() + 3:
                    self.search(item, w)
        self.page_small_sets()

    def requery(self, item: dict[str, Any]) -> list[str]:
        prompt = (REQUERY_PROMPT.replace('{ITEM}', self.item_text(item)).replace('{SEARCH_NOTE}', self.search_note()))
        got = self.llm(prompt, model=FAST, label='requery', think='off')
        if isinstance(got, dict):
            return [str(q) for q in got.get('queries') or [] if isinstance(q, str) and q.strip()]
        return []

    def read_notices(self) -> bool:
        """Read store notices and refusals: changed requirements, listings gone or repriced."""
        fresh = [e for e in self.events if not e.get('_read')]
        if not fresh:
            return False
        for event in fresh:
            event['_read'] = True
        items = '\n'.join(f"[{i['key']}] {i['name']} x{i['quantity']} requirements: "
                          + '; '.join(f'({n + 1}) {m}' for n, m in enumerate(i['musts'])) for i in self.items)
        prompt = (NOTICE_PROMPT.replace('{REQUEST}', self.request).replace('{ITEMS}', items)
                  .replace('{CART}', _clip(self.cart_summary(), 1500))
                  .replace('{NOTICES}', _clip([{k: v for k, v in e.items() if k != '_read'} for e in fresh], 2500)))
        got = self.llm(prompt, model=FAST, label='notices', think='off')
        self.log('notices', read=got)
        if not isinstance(got, dict):
            return False
        changed = False
        for change in got.get('changes') or []:
            if not isinstance(change, dict):
                continue
            item = next((i for i in self.items if i['key'] == change.get('item')), None)
            if item is None:
                continue
            drop = {int(n) - 1 for n in change.get('remove') or [] if isinstance(n, (int, float))}
            item['musts'] = [m for n, m in enumerate(item['musts']) if n not in drop]
            for must in change.get('add') or []:
                if isinstance(must, str) and must.strip():
                    item['musts'].append(must.strip())
            self.verdicts.pop(item['key'], None)
            item['_requery'] = True
            changed = True
        for rid in got.get('unavailable') or []:
            for per_item in self.verdicts.values():
                if str(rid) in per_item:
                    per_item[str(rid)]['verdict'] = 'fail'
                    per_item[str(rid)]['in_stock'] = False
                    per_item[str(rid)]['why'] = 'the store said it is no longer available'
                    changed = True
        for rule in got.get('rules') or []:
            if isinstance(rule, str) and rule.strip():
                self.plan.setdefault('rules', []).append(rule.strip())
        new_limit = _num(got.get('budget'))
        budget = self.plan.setdefault('budget', {})
        if new_limit is not None and new_limit > 0 and new_limit != _num(budget.get('limit')):
            budget['limit'] = new_limit
            for item in self.items:
                item['_cheap_done'] = False
                item['_hint'] = f'the budget is now {new_limit:g}: cheaper listings are needed'
            changed = True
        if changed:
            self.parallel([(lambda it=it: self.judge(it)) for it in self.items if it['key'] not in self.verdicts
                           or not self.verdicts[it['key']]])
        return changed

    def complete(self) -> None:
        dependent = [i for i in self.items if i.get('depends_on')]
        if dependent and self.left() > 110:
            jobs = []
            for item in dependent:
                old = dict(self.verdicts.get(item['key']) or {})
                ranked = sorted(old, key=lambda r: (old[r].get('verdict') == 'fail', self.price_of(r) or 1e18))
                keep = ranked[:6]
                jobs.append(lambda it=item, ids=keep: self.judge(it, ids) if ids else None)
            self.parallel(jobs)
        for attempt in range(4):
            if self.finished:
                return
            if self.read_notices() and self.left() > 70:
                self.search_changed()
                self.explore_all(rounds=1, per_round=3, floor=85)
            bare = [i for i in self.items if not self.verdicts.get(i['key']) and
                    any(r in self.records for r in self.seen_by_item.get(i['key'], []))]
            if bare:
                self.parallel([(lambda it=it: self.judge(it)) for it in bare])
            decision = (self.auto_choose() if attempt == 0 else None) or \
                (self.choose(final=attempt == 3 or self.left() < 60 or self.steps_left() < self.reserve())
                 if self.left() > 30 else None) or self.last_resort()
            if not decision:
                break
            action = str(decision.get('action') or '').lower()
            if action == 'search_more' and attempt < 3 and self.left() > 90 and \
                    self.steps_left() > self.reserve() + 3:
                for want in decision.get('search') or []:
                    if not isinstance(want, dict):
                        continue
                    named = str(want.get('item') or '').strip().lower()
                    item = next((i for i in self.items if i['key'].lower() == named or i['name'].lower() == named),
                                None)
                    if item is None:
                        continue
                    # phrases already searched give the same rows again: new ones only
                    used = {str(q['words'] or '').lower().strip() for q in self.queries
                            if q['item'] == item['key'] and not q.get('page')}
                    fresh = [str(w) for w in want.get('queries') or [] if str(w).lower().strip() not in used]
                    if not fresh:
                        fresh = [w for w in self.requery(item) if w.lower().strip() not in used]
                    for words in fresh[:2]:
                        if self.steps_left() > self.reserve() + 3:
                            self.search(item, words)
                    if item.get('_cheap_done'):
                        item['_cheap_done'] = False
                self.explore_all(rounds=1, per_round=3, floor=90)
                continue
            if action == 'search_more':
                decision = self.choose(final=True) or {}
                action = str(decision.get('action') or '').lower()
            if action == 'decline' and not self.decline_checked and self.left() > 70 and \
                    self.steps_left() > self.reserve() + 4:
                self.decline_checked = True
                for item in self.items:
                    item['_hint'] = _clip(decision.get('why') or '', 300)
                    if len(self.passing(item)) < 2 and self.steps_left() > self.reserve() + 8 and self.left() > 90:
                        self.page_more(item)
                self.explore_all(rounds=1, per_round=3, force=True)
                continue
            if action == 'decline':
                self.process_step('end')
                observation = self.finish(buy=False)
                if self.finished:
                    return
                # the store turned the decline down (often with a relaxed requirement): read it and carry on
                self.events.append({'decline_refused': _clip(observation, 1200)})
                self.decline_checked = False
                if self.read_notices() and self.left() > 60:
                    self.search_changed()
                    for item in self.items:
                        item['_hint'] = 'the shopper changed a requirement after a decline: ' + _clip(observation, 200)
                    self.explore_all(rounds=2, per_round=3, force=True, floor=85)
                continue
            lines = []
            for line in decision.get('lines') or []:
                if not isinstance(line, dict) or not line.get('id'):
                    continue
                item = next((i for i in self.items if i['key'] == line.get('item')), None)
                quantity = line.get('quantity')
                if not isinstance(quantity, (int, float)) or quantity < 1:
                    quantity = item['quantity'] if item else 1
                lines.append({'item': line.get('item'), 'id': str(line['id']), 'quantity': int(quantity)})
            if not lines:
                break
            for line in lines:
                rid = str(line['id'])
                if rid not in self.records and self.roles.get('open') and self.steps_left() > len(lines) + 3:
                    self.open_listing(rid)
            self.process_step('choose')
            replies = self.fill_cart(lines)
            check = self.check_cart(lines, replies)
            if check and check.get('ok') is False and attempt < 2 and self.left() > 40 and self.steps_left() > 4:
                self.problems.append({'cart_problem': check.get('problems')})
                for problem in check.get('drop') or []:
                    for per_item in self.verdicts.values():
                        if str(problem) in per_item:
                            per_item[str(problem)]['verdict'] = 'fail'
                            per_item[str(problem)]['why'] = 'cart check: ' + _clip(check.get('problems'), 200)
                continue
            self.process_step('order')
            observation = self.finish(buy=True)
            if self.finished:
                return
            self.events.append({'checkout_refused': _clip(observation, 1200)})
            if self.roles.get('cart_view') and self.steps_left() > 3:
                self.events.append({'cart_after_refusal': _clip(self.view_cart(), 1200)})
        if not self.finished:
            if self.cart_lines:
                self.finish(buy=True)
            if not self.finished:
                self.finish(buy=False)


def _strings(obj: Any, depth: int = 0) -> set:
    found: set = set()
    if depth > 6:
        return found
    if isinstance(obj, str):
        found.add(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            found |= _strings(value, depth + 1)
    elif isinstance(obj, list):
        for value in obj:
            found |= _strings(value, depth + 1)
    return found


def _flatten(record: dict, prefix: str = '', depth: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if str(key).startswith('_'):
            continue
        name = f'{prefix}{key}'
        if isinstance(value, dict) and depth < 2:
            out.update(_flatten(value, name + '.', depth + 1))
        elif isinstance(value, (str, int, float, bool)) and value != '':
            out[name] = value
        elif isinstance(value, list) and value and all(isinstance(v, (str, int, float)) for v in value):
            out[name] = ', '.join(str(v) for v in value[:6])
    return out


def fallback_loop(episode: Episode) -> None:
    """When the setup could not be read: let the model drive the tools directly, one call per step."""
    messages: list[dict[str, Any]] = [
        {'role': 'system', 'content': FALLBACK_SYSTEM},
        {'role': 'user', 'content': episode.request + ('' if episode.opening is None else
                                                      '\n\n' + _clip(episode.opening, 2000))}]
    tools = episode.view.get('tools') or []
    idle = 0
    while not episode.finished and episode.step < episode.max_steps and episode.left() > 20 and \
            episode.cost < SPEND_SOFT:
        recent = messages[:2] + messages[2:][-24:]
        while len(recent) > 2 and recent[2].get('role') == 'tool':
            recent.pop(2)                  # a tool reply needs the call before it
        body = {'model': FAST, 'messages': recent, 'tools': tools, 'tool_choice': 'auto', 'temperature': 0,
                'max_tokens': 3000}
        result = episode.inference_post(min(episode.left() - 10, 40), body)
        data = result.data
        if isinstance(data, dict) and isinstance((data.get('usage') or {}).get('cost'), (int, float)):
            episode.cost += float(data['usage']['cost'])
        if not isinstance(data, dict) or not data.get('choices'):
            idle += 1
            if idle > 3:
                break
            continue
        message = data['choices'][0].get('message') or {}
        calls = message.get('tool_calls') or []
        if not calls:
            idle += 1
            if idle > 3:
                break
            messages.append({'role': 'assistant', 'content': message.get('content') or ''})
            messages.append({'role': 'user', 'content': 'Call one of the tools.'})
            continue
        messages.append({'role': 'assistant', 'content': message.get('content') or '', 'tool_calls': calls[:1]})
        call = calls[0]
        raw = call['function'].get('arguments') or '{}'
        try:
            args = raw if isinstance(raw, dict) else json.loads(raw)
        except (ValueError, TypeError):
            args = {}
        observation = episode.act(call['function']['name'], args if isinstance(args, dict) else {})
        messages.append({'role': 'tool', 'tool_call_id': call.get('id'), 'content': _clip(observation, 4000)})


# ---------------------------------------------------------------------- prompts
SHARED_SYSTEM = """You are the planning and decision model of a shopping agent working in an online store for one
shopper. Each message below gives you one job and the JSON object to return. Answer with that JSON object only.

GUIDE ON ASKING THE SHOPPER (used when the job is to plan a message):
The shopper has hidden requirements: some products of the order have one property the shopper cares about but did not
write in the request (for example a phone case's material or a lamp's plug type). The shopper reveals such a
requirement only when a message asks about that property of that product. Each message gets one reply. How shoppers
usually answer:
- They answer about the FIRST property in the message that they care about; a property they have no view on may get
  "I don't mind", and that answer can end the reply even if a later property was the one they care about.
- A message that asks about several products at once often confuses them ("you've lost me"), and a long list of
  properties can be turned away (a request to ask fewer things per message). After such a reply ask about one product and one property.
- Asking about something the request already states wastes the message.
So: one product per message, its likeliest hidden property first, then 2 or 3 more candidates in order of
likelihood. Shoppers read the first few properties of a message, so keep it to 3 or 4 in all. Name each property
exactly as the listings' attribute names it ("storage type", not "storage capacity"), in plain words, with no example
values or explanations in brackets. The one property that nearly every
listing of that product carries is more often a "don't mind" than the hidden one.
After an "I don't mind" reply to a list, the answer was about the first property of that list the shopper keeps an
eye on, most likely the first one you named: never lead with it again; ask the others of that product in a new order,
or move to another product. After a reply that turned the list away, ask one property alone.
How to rank: a property the listings of that product carry as a named attribute (use that attribute's name, in
plain words), whose values differ between listings, that matters for how the product is used and that the request
leaves open. Never lead with a property no listing names. A message that names no property the shopper keeps an eye
on for that product is wasted (a shopper with several products then says "you've lost me"), so name properties
that most of the opened pages carry. Never ask again about a property the shopper already answered or the request states. Product-specific properties (a connector, a size class, a
form, a type) are often the hidden ones; broad looks (colour, material, pattern, style) are often "no preference" but
not always. The request's story can hint at a property: when it says the shopper has not decided or does not
know something about a product (its use, how it travels, where it is kept, who shares it, what role it plays), the hidden
requirement is most likely the listing property that open point is about, so ask that one first. Spread the messages over the products that most likely hold
a hidden requirement. Which products: a product the request mentions only once (no fit, condition, price cap, trail
or other rule refers to it) holds a hidden requirement far more often than a product that other rules already
constrain, so ask about those first; then products bought in larger quantities; and do not ask twice about a product whose hidden requirement was already told unless all
others were covered."""

ROLES_PROMPT = """You map the tools of an online store's API to shopping jobs. Read each tool's name, description and
parameters, then return ONE JSON object (no prose) with these keys. Use exact tool and parameter names; null when no
tool does the job.

"search": {"tool": the tool that lists products, "text_arg": its parameter for free search words or null,
  "category_arg": its parameter that takes a category, department or shelf name or null,
  "keyword_args": fixed arguments needed to search by words (for a tool with a mode argument, the word-search mode),
  "browse_args": fixed arguments to list a category without words (its browse mode), else {},
  "cursor_arg": the parameter that takes a continuation token or page for more results, or null}
"open": {"tool": the tool that shows ONE listing's full details (attributes, specifications, description), "id_arg": its id parameter, "fixed_args": {}}
"offer": {"tool", "id_arg", "fixed_args"} for a DIFFERENT tool that shows one listing's current price or stock, else null
"cart": {"tool": the tool that puts listings in the cart or changes quantities,
  "style": "line" when one call handles one listing, "lines" when one call takes a list of lines,
  "id_arg": (style line) its listing id parameter, "qty_arg": (style line) its quantity parameter or null,
  "lines_arg": (style lines) the parameter holding the list, "line_id_key" and "line_qty_key": the id and quantity
  keys inside one line object,
  "qty_means": "set" when a quantity replaces that line's quantity, "add" when it is added to what is there,
  "replace" when the list given replaces the whole cart,
  "fixed_args": arguments that make it add or set (for a tool with a mode argument)}
"remove": {"tool", "id_arg", "fixed_args"} for a way to take a line out of the cart (the cart tool itself with another
  value of its mode argument counts), else null
"cart_view": {"tool", "fixed_args"} that shows the cart, else null
"buy": {"tool": the tool that places the order or checks out, "args": the fixed arguments that make it buy}
"refuse": {"tool": the tool that ends the visit WITHOUT buying (decline), "args": the fixed arguments that make it decline}
"ask": {"tool": the tool that sends a message to the shopper or customer, "text_arg": its text parameter, "fixed_args": {}}
"extra": a list of {"tool", "purpose"} for every other tool, in a few words each

TOOLS:
{TOOLS}"""

ROLE_JOBS = [job for job in re.findall(r'^"(\w+)": ', ROLES_PROMPT, re.M) if job != 'extra']

PLAN_PROMPT = """You prepare a shopping task. Read the shopper's request and the store's tools and return ONE JSON object
(no prose) with these keys:

"items": one entry per distinct product to buy, in the request's order. A product named in one phrase like "cups and
saucers" with one quantity and one set of requirements is ONE item (often the name of a store category); make separate
items only when each product gets its own quantity or its own requirements. Each item is {
  "name": the kind of product in the request's words,
  "quantity": units of that product (1 when not stated),
  "category": the store's category label for it copied EXACTLY from this list when one fits: {CATS}, else "",
  "shelf": the store category, range or department the request itself names for this product, in the request's
     words ("from your Patio Heating range" -> "Patio Heating"), else "",
  "musts": every firm requirement on this product, each "property: requirement" keeping the exact values, limits,
     units, exclusions and either/or lists ("colour: navy or black", "screen size: at least 13 inch",
     "fits: the Pelton 220 bike the shopper owns", "seller: not Brinx"). A requirement on how the product relates to
     ANOTHER requested product ("must fit the stand you pick") is a must here too. A rule that applies only under a
     condition is NOT a must: put it in "rules". Never put price, budget, stock or the quantity here, nor the
     product's own kind ("type: lamp" for a lamp; the store category decides the kind). Worries,
     stories and things the shopper says they do not know are context, not musts; but when the shopper states a
     requirement through a situation ("it will sit out in the rain all winter"), write the listing property and
     the value that situation calls for ("weather resistance: waterproof"), not the story.
  "wishes": soft preferences ("ideally", "would be nice"), else [],
  "queries": 3 short search phrases for this product, most specific first: 2-4 words each, product words plus at most
     one or two required values; the store matches every word, so long phrases find nothing. For a fit to a model,
     listings usually name a model family pattern rather than the exact model ("every ZX40xx model"), so search with
     that pattern or leave the model out; when the request names the store category, one phrase is that name,
  "depends_on": keys of the other items this one must work with, fit or match ("B" when it must fit whichever item
     B is picked), else []
}
"budget": {"limit": the number or null, "unit": as written, "scope": "total" when it caps the sum of everything bought, "each" when it caps each product, or null},
"cheapest": "total" when the request asks for the lowest total or the cheapest acceptable choices, "each" when only some products must be the cheapest (say which in rules), "none" otherwise,
"rules": every other rule in short plain sentences: conditions ("if X then Y"), what to do when something is out of
   stock or nothing fits (order the rest, decline everything, ...), when to decline, rules about sellers or listings,
   rules about how products relate, and any step the shopper wants taken (approval, confirmation, telling them
   something). Keep the request's exact values.
"process": steps the shopper wants taken besides finding and buying (getting their approval, confirming the basket,
   telling them something), each {"step": in at most 15 words, "when": "choose" (before choosing), "order" (right before
   checking out) or "end" (before ending without a purchase), "messages": messages it takes}; [] when none.
"references": requirements whose value must be read from another listing's page, each {"item": item key,
   "property": the property, "start": the description of the first listing to open, copied with its seller, kind and
   attributes ("the Brinx kettle with a glass lid"), "follow": true when the request says to follow the listings that
   page names}; [] when none. Such a requirement also appears in musts as "property: same as the end of that trail".
"conditions": the rules that limit the choice beyond each product's own requirements and the total budget (a price
   cap on one product, a rule linking products, an if-then rule, a rule about sellers or listings), as short
   sentences; [] when there are none,
"process_messages": total messages the process steps need (0 when none),
"message_limit": how many messages the shopper will answer (a number) or null when not stated,
"step_limit": the step limit stated or null

The store's tools and the shopper's request are given above."""

SCHEMA_PROMPT = """Below is one reply from a store's search tool. Name the fields, as dotted paths inside ONE result
row (or inside the reply for cursor/total/rows). Return ONE JSON object:
{"rows_key": path in the reply to the list of results, "id_key": the listing id field in a row, "title_key": the
title or name field in a row, "price_key": the price field in a row (null when rows carry no price), "stock_key": the
in-stock or availability field in a row or null, "cursor_key": path in the reply to the token for the next page or
null, "total_key": path in the reply to the count of all matches or null}

REPLY:
{OBS}"""

SCREEN_PROMPT = """You help a shopping agent find a product in an online store.

PRODUCT WANTED:
{ITEM}

RULES OF THE ORDER:
{RULES}

Does the cheapest acceptable choice win: {CHEAP}

{SEARCH_NOTE}

Search history for this product:
{HISTORY}

LISTINGS ALREADY CHECKED (with the result):
{CHECKED}

LISTINGS FOUND BUT NOT OPENED YET (id | search row):
{ROWS}

Choose up to {LIMIT} listing ids to open next: those that could meet every requirement (the page shows details the
row may not). The store's category decides the kind of product: a listing filed in the requested category counts as
that product even when its title reads oddly. Skip only listings that clearly break a requirement or sit in another
category. When the cheapest choice wins, prefer the cheapest plausible ones. Also list in "skip" the listings whose
search row itself shows they cannot do: a stated value that breaks one of the requirements above, a store category
for another kind of product, or out of stock. A title is no reason to skip: when the row's store category fits the
product wanted, the listing counts as that product whatever its title calls it (a "tote" filed under Backpacks is a
backpack), and a row that does not mention a requirement is no reason to skip either. If none of the
listings fits, or too few look right, write up to 2 new short search phrases (2-4 words; the store matches every
word, so fewer words find more) that would find better listings. Set "more_pages" true when cheaper listings matter
and the cheaper plausible listings may be on later result pages. Set "shelf" to a store category shown on the rows
when the request names that category (its words match the category's name, not just the kind of product), copied as
the rows show it, else "".
Return ONE JSON object: {"open": [ids], "skip": [ids], "queries": [phrases], "more_pages": true or false,
"shelf": category or ""}"""

JUDGE_PROMPT = """You check store listings against what a shopper requires. Return ONE JSON object (no prose).

THE SHOPPER'S WHOLE REQUEST (for context):
{REQUEST}

PRODUCT BEING CHECKED:
{ITEM}

OTHER PRODUCTS IN THE SAME ORDER:
{OTHERS}

RULES OF THE ORDER:
{RULES}

How to read a listing:
- Judge only the product's own requirements. Price and budget are NOT judged here (another step adds up the order):
  never fail a listing for its price; only report its cost and stock state.
- The listing must be the requested kind of product: the store's category decides it (a listing filed in the
  requested category counts even when its title reads oddly; one filed in a clearly different category fails;
  plurals and small wording differences do not matter; with no category, the title decides). A requirement the store's
  category label itself states (a category named "Outdoor Lamps" states outdoor use) is met by every listing in it.
- An attribute whose name is that requirement's property (or a plain synonym) decides it; if it says something else,
  the requirement fails. A key for a different property, even a related one, decides nothing about it.
- With no such attribute, the value stated in the title or description counts. Units may be converted; limits include
  the edge; "either/or" lists accept any member; an exclusion fails when the page states the excluded value.
- Notes on the page can say a value is taken from ANOTHER listing ("for X, see the Y sold by Z"): then that other
  listing's value counts instead. If the value read there is attached under _value_from_other_listing, use
  it; if not, set "lookup" so the agent can fetch it.
- Fit requirements ("fits the shopper's car/device/model") must be shown by the listing's own text (model lists,
  ranges with exceptions, part series). Read ranges and exceptions literally.
- Stock: a listing marked unavailable/out of stock is in_stock false.
- A requirement on matching ANOTHER product of the order: say what this listing offers in "facts" and judge it as
  "unknown" only when it depends on which other product is chosen.

For each listing return {"id", "verdict": "pass" (meets every requirement of this product) | "fail" | "unknown"
(a requirement cannot be decided from the page), "in_stock": true/false/null, "price": number or null,
"failed": [requirements it breaks], "unclear": [requirements it cannot show], "compatible_with": [ids of the
other products' candidates it works with, when such candidates are listed], "facts": {property: value} for every
property named in the rules or requirements, "why": one short sentence, "lookup": null or {"property": which value,
"kind": the other listing's kind of product, "category": its store category when named, "words": 2-4 search words
that would find it (its seller or brand and kind), "hints": what identifies it}}.
Answer as {"listings": [...]}.

LISTINGS:
{LISTINGS}"""

FIND_PROMPT = """A shopping agent looks for one particular store listing, described like this:
{WANT}

Which of these search results could be it (the description may name its seller, kind and attributes; rows may not
show them all)? Return ONE JSON object {"ids": [up to 3 ids, likeliest first]}.

RESULTS:
{ROWS}"""

LOCATE_PROMPT = """A shopping agent must find one particular store listing, described like this:
{DESC}

{SEARCH_NOTE}
Store categories: {CATS}

Write search phrases that find it. The store matches every word of a phrase, and seller or brand names are
searchable. Return ONE JSON object {"category": its category copied exactly from the list or "", "queries": [up to 3
phrases: first the seller or brand name with the kind and one attribute value, then shorter ones such as the seller
name with the kind]}"""

POINTER_PROMPT = """A shopping agent follows store pages to learn the value of one property: {PROP}
It is looking for the listing described like this: {DESC}

OPENED PAGES:
{PAGES}

Return ONE JSON object:
{"id": the id of the page that is the listing described (seller, kind and the attributes named agree), or null when
none is; for that page:
"next": when that page says the value of THIS property ({PROP}) is to be taken from ANOTHER listing, that listing's
description copied from the page (seller, kind, attributes); a pointer for a different property does not count; else null,
"value": when the page does not send you elsewhere for this property, the value it gives for it; else null}"""

ASK_PROMPT = """You plan the next message from a shopping agent to the shopper.

Follow the guide on asking given above.

The shopper's request is given above.

PRODUCTS (with requirements known so far and the properties their listings show):
{ITEMS}

CONVERSATION SO FAR:
{CONVO}

Messages you may still send: {LEFT}
Ask about one of these products now (keys): {TARGETS}{SINGLE}

Return ONE JSON object: {"ask": true or false (false only when every product's hidden requirement was already told;
an unused message is wasted), "item": the product key,
"properties": [ordered property names], "message": the message to send, short and plain, e.g. "For the desk lamp,
which plug type do you need? Or does the shade colour or the bulb type matter?"}"""

REPLY_PROMPT = """A shopping agent sent the shopper a message and got a reply. Read what the reply tells.

REQUEST:
{REQUEST}

PRODUCTS:
{ITEMS}

MESSAGE: {QUESTION}
REPLY: {REPLY}

Return ONE JSON object:
{"told": [{"item": product key, "property": the property, "requirement": the exact requirement with its value or
limit, e.g. "at least 2 m", "red", "not leather"}] for every requirement the reply states,
"no_preference": [{"item", "property"}] for properties the shopper does not mind,
"confused": true when the shopper did not understand the message,
"single_only": true when the shopper asks for fewer questions per message (a single question each time),
"rules": [new rules about the order, in short sentences] or [],
"references": [{"item": product key, "property": the property, "start": the description of the first listing to open,
copied with its seller, kind and attributes}] when the reply says the value is to be read from another listing's page
(a trail of pages), else [],
"message_limit": a number if the reply states how many more messages they will answer, else null}"""

REQUERY_PROMPT = """Write up to 2 short search phrases (2-4 words, the store matches every word) that find listings meeting
this product's requirements as they stand now; do not repeat a search already made (listed below).

PRODUCT:
{ITEM}

{SEARCH_NOTE}

Return ONE JSON object: {"queries": [phrases]}"""

CHOOSE_PROMPT = """You decide how a shopping agent ends its visit. Return ONE JSON object (no prose). The shopper's
request is given above.

RULES (read from the request and the shopper's replies):
{RULES}

CONVERSATION WITH THE SHOPPER:
{CONVO}

CHECKED LISTINGS PER PRODUCT (id | price | verdict | stock | reason | facts):
{CANDIDATES}

NOTICES FROM THE STORE DURING THE VISIT:
{EVENTS}

Steps left: {STEPS}. Seconds left: {TIME}.

Decide:
- "order": pick one listing for each product: only listings with verdict pass and stock available, so that every
  rule holds (conditions between products, the budget over everything bought: price times quantity summed, the cheapest
  acceptable choice where asked). Prices in the list are the store's prices. When the budget is written in cents
  and the prices are whole numbers, the prices are cents too (a price of 9 is 9 cents); convert only when the store
  clearly shows another unit.
- "search_more": when a product has no acceptable listing yet but more searching could find one and steps and time
  allow it; give new short search phrases (2-4 words).
- "decline": when the request cannot be met as stated (listings of a product were checked and all fail, the budget
  cannot be met, or a stock rule says to give up) and the request never asks to order the rest anyway.
- A product with no "pass" but with "unknown" listings: pick the likeliest unknown one rather than decline.
- Store prices are what they are: a very low price is not an error. Do not reject a listing for reasons outside the
  requirements and rules.
Return {"action": "order" | "search_more" | "decline", "lines": [{"item": key, "id": listing id, "quantity": n}],
"search": [{"item": key, "queries": [phrases]}], "total": the order total, "why": one sentence}"""

CART_PROMPT = """A shopping agent filled the cart and is about to check out. Check the cart against the plan and the
request. Return ONE JSON object.

REQUEST:
{REQUEST}

RULES:
{RULES}

PLANNED LINES: {LINES}

REPLIES OF THE CART TOOL: {REPLIES}

CART NOW: {CART}

NOTICES DURING THE VISIT: {EVENTS}

Report a problem ONLY when the replies or the cart SHOW it: a planned line missing or with the wrong quantity, an
extra line, an error message from the store, a listing marked unavailable, a price that changed so the budget breaks,
a notice that changes what the shopper wants. Something merely not confirmed is not a problem. Store prices are what
they are, however low.
Return {"ok": true or false, "problems": [short descriptions], "drop": [listing ids that can no longer be ordered]}"""

PROCESS_PROMPT = """The shopper asked for a step before the order is finished. Write the message that performs it.

REQUEST:
{REQUEST}

STEP: {STEP}

CART: {CART}

CONVERSATION SO FAR:
{CONVO}

Return ONE JSON object: {"message": the message to send, plain and short}"""

NOTICE_PROMPT = """A shopping agent got notices from the store or the shopper during the visit (a refused checkout, a
change of mind, a listing that ran out or changed price). Work out what they change. Return ONE JSON object.

REQUEST:
{REQUEST}

PRODUCTS WITH THEIR NUMBERED REQUIREMENTS:
{ITEMS}

CART: {CART}

NOTICES:
{NOTICES}

Return {"changes": [{"item": product key, "remove": [numbers of requirements the shopper dropped or replaced],
"add": ["property: requirement" for new or replacing requirements]}], "unavailable": [listing ids that ran out or can no
longer be bought], "rules": [any new rule in a short sentence], "budget": the new budget limit as a number when a
notice changes it (same unit as before), else null, "summary": one sentence}"""

FIX_PROMPT = """A store tool turned down a call because of its arguments. Rewrite the arguments so they fit the
tool's parameter schema and still do what the call meant to do. Return ONE JSON object {"args": {...}}.

TOOL: {TOOL}
ARGUMENTS SENT: {ARGS}
REPLY: {ERROR}"""

FALLBACK_SYSTEM = ("You are a careful shopping agent. Use the tools to find listings that meet every requirement of "
                   "the request, ask the shopper only when the request allows it, put the right quantities in the "
                   "cart and finish with an order, or decline when the request cannot be met. Every reply must be "
                   "exactly one tool call.")


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Episode(problem_data)
    try:
        episode.run()
    except Exception as exc:  # keep the trace and try to end the visit cleanly
        episode.log('crash', error=repr(exc))
        try:
            if not episode.finished and episode.roles:
                if episode.cart_lines:
                    episode.finish(buy=True)
                if not episode.finished:
                    episode.finish(buy=False)
        except Exception as again:
            episode.log('crash_finish', error=repr(again))
    episode.log('end', steps=episode.step, calls=episode.calls, cost=round(episode.cost, 4),
                finished=episode.finished)
    return episode.trace


__all__ = ['agent_main']
