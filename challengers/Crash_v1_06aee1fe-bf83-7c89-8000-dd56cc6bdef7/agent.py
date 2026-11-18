from __future__ import annotations
import json
import random
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient

# 29-know: a shopping agent in its own structure. Wire talks to the store and the models, Shelf reads the store's
# tools and listings, Task holds what the shopper wants, Asker runs the questions to the shopper, Pilot drives the
# light model through the store, Closer guards the checkout and the ending without a purchase.

LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'

LOOP_TURNS = 44          # main-loop model turns per episode
CALL_CAP = 64            # model calls of any kind per episode
SOFT_SPEND = 0.040       # optional heavy calls only below this (USD, as the provider reports it)
HARD_SPEND = 0.078       # the main loop stops above this
HEAVY_ROOM = 0.02        # what one heavy call may need under the hard line

EPISODE_S = 290.0        # the store stops an episode at 300 s
WRAP_S = 70.0            # from here on: finish the cart and check out
FORCE_S = 18.0           # the controller ends the episode itself
FILL_S = 120.0           # a checkout from here on fills items that have no cart line

LIST_KEYS = 8            # properties named in one broad question
LIST_MIN_MESSAGES = 3    # broad questions only when the shopper allows this many messages
KEY_KEEP = 24            # attribute keys kept per item for the questions
PLAN_SHOW = 14           # keys per item shown to the planner
OPEN_BEFORE_ASK = 2      # opened listings per item before the questions start

TEXT_CLIP = 450          # long strings in a result shown to the model
ROW_TEXT = 260           # long strings inside search rows
RESULT_CAP = 15000       # characters of one result shown to the model
CUT_AT = 150000          # history size (chars) that triggers a deep cut
CUT_GAP = 60000          # then no cut until the history grew this much more
CUT_KEEP = 3             # model turns kept whole at a cut
DIGEST_ROW = 220
PAGES_CAP = 6            # result pages the controller reads itself in one episode
SKIP = 'Skipped (no step spent): '      # a call the controller did not send
HOLD = 'Held (no step spent): '            # an ending the controller holds back for now
UNSPENT = (SKIP, HOLD)
BACKOFF = (4, 10, 20, 35)
BACKOFF_TOTAL = 45

MODEL_TIMEOUT = 50
_llm = ProxyClient(max_retries=1, retry_delay=1, timeout=MODEL_TIMEOUT)
_env = ProxyClient(max_retries=3, retry_delay=1, timeout=95)


def _j(value: Any) -> str:
    try:
        return json.dumps(value, separators=(',', ':'), ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _clip(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value[:limit] + '...' if len(value) > limit else value
    if isinstance(value, dict):
        return {key: _clip(inner, limit, depth + 1) for key, inner in value.items()}
    if isinstance(value, list):
        return [_clip(v, limit, depth + 1) for v in value]
    return value


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r'\s*-?\d+\s*', value):
        return int(value)
    return None


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.fullmatch(r'\s*(-?\d+(?:\.\d+)?)\s*', value)
        if m:
            return float(m.group(1))
    if isinstance(value, dict):
        return next((float(v) for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
    return None


def _obj(text: Any) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    fence = re.match(r'^```[A-Za-z]*\s*(.*?)\s*```$', text, re.S)
    if fence:
        text = fence.group(1)
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else None
    except ValueError:
        pass
    start = text.find('{')
    while start >= 0:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        got = json.loads(text[start:end + 1])
                        if isinstance(got, dict):
                            return got
                    except ValueError:
                        pass
                    break
        start = text.find('{', start + 1)
    return None


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [] if value is None else [value]


def _words(text: Any) -> str:
    # a name in plain lower-case words: camelCase, snake_case and punctuation become spaces
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(text or ''))
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def _stems(text: Any) -> set[str]:
    out = set()
    for w in _words(text).split():
        out.add(w[:-1] if len(w) > 3 and w.endswith('s') else w)
    return out


def _kinds(schema: Any) -> list[str]:
    kind = schema.get('type') if isinstance(schema, dict) else None
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def _idlike(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_\-:.]{4,40}', value.strip())) and bool(re.search(r'\d', value))


def _fill(node: Any, ident: str, units: int) -> Any:
    """A call template with the listing and unit placeholders filled in, at any depth."""
    if node == '<ID>':
        return ident
    if node == '<QTY>':
        return units
    if isinstance(node, dict):
        return {k: _fill(v, ident, units) for k, v in node.items()}
    if isinstance(node, list):
        return [_fill(v, ident, units) for v in node]
    return node


def _line_list(template: dict[str, Any]) -> str | None:
    """The argument that holds a list of cart lines (a tool that sets the whole basket), when the template has one."""
    for key, value in template.items():
        if isinstance(value, list) and len(value) == 1 and '<ID>' in _j(value[0]):
            return key
    return None


def _holder(node: Any, key: str, depth: int = 0) -> dict[str, Any] | None:
    """The dict that carries `key`, looking into wrapped results."""
    if not isinstance(node, dict) or depth > 3:
        return None
    if key in node:
        return node
    for value in node.values():
        found = _holder(value, key, depth + 1)
        if found is not None:
            return found
    return None


def _has_rows(node: Any, depth: int = 0) -> bool:
    """True when a result holds listing-like records (a non-empty list of objects, or an object with an id-like value)."""
    if depth > 3:
        return False
    if isinstance(node, list):
        return any(isinstance(x, dict) and x for x in node) or any(_has_rows(x, depth + 1) for x in node[:3])
    if isinstance(node, dict):
        if depth <= 2 and len(node) >= 3 and any(_idlike(v) for v in node.values()):
            return True
        return any(_has_rows(v, depth + 1) for v in node.values() if isinstance(v, (dict, list)))
    return False


class Wire:
    """The two endpoints: the store (one call = one step) and the models. Counts steps, spend, calls and time."""

    def __init__(self, binding: dict[str, Any], trace: list[dict[str, Any]]):
        self.binding = binding
        self.trace = trace
        self.step = 0
        self.finished = False
        self.spend = 0.0
        self.calls = 0
        self.waited = 0.0
        self.no_credit = False
        self.started = time.monotonic()

    def seconds_left(self) -> float:
        return EPISODE_S - (time.monotonic() - self.started)

    def store(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        for attempt in range(4):
            result = _env.post_verbose('/environment/call', payload)
            data = result.data
            if isinstance(data, dict):
                step = data.get('step')
                self.step = step if isinstance(step, int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.trace.append({'kind': 'store_fail', 'status': status, 'body': body[:200]})
            if status == 409:
                said = re.search(r'expected (\d+)', body)
                if said:
                    self.step = int(said.group(1)) - 1
                    payload['step'] = self.step + 1
                    continue
                self.finished = self.finished or bool(re.search(r'finish|final|quarantin', body, re.I))
                return None
            if status is None or status == 429 or (isinstance(status, int) and status >= 500):
                time.sleep(1.5 * (attempt + 1) * (0.8 + 0.4 * random.random()))
                continue
            return None
        return None

    def model(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= CALL_CAP or self.seconds_left() < 12:
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self._post(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                if status == 402:
                    self.no_credit = True
                    return None
                again = status is None or status in (408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not again or tries >= len(BACKOFF) or self.waited + BACKOFF[tries] > BACKOFF_TOTAL or self.seconds_left() < BACKOFF[tries] + FORCE_S:
                    break
                self.waited += BACKOFF[tries]
                time.sleep(BACKOFF[tries] * (0.8 + 0.4 * random.random()))
                tries += 1
        return None

    def _post(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            # prompt cache: the fixed system text and the newest message carry the marks
            sent = [dict(m) for m in messages]
            for index in sorted({0, len(sent) - 1}):
                content = sent[index].get('content')
                if isinstance(content, str) and content:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        extra: dict[str, Any] = {'tools': tools, 'tool_choice': 'auto'} if tools else {}
        begun = time.monotonic()
        data = _llm.post('/inference/chat/completions', json_data={'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0, **extra})
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.spend += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return message, 200
            self.trace.append({'kind': 'model_empty', 'model': model})
            return None, 0
        # post() hands back no status: a failure that took the whole timeout is read as a timeout, any other one as an
        # upstream fault; both keep the bounded retry rules of model()
        status = 408 if time.monotonic() - begun >= 0.9 * MODEL_TIMEOUT else 502
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = _int(data['error'].get('code')) or status
        self.trace.append({'kind': 'model_fail', 'model': model, 'status': status})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        for model in models:
            message = self.model([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = _obj((message or {}).get('content'))
            if got is not None:
                return got
        return None


LAYOUT_SLOTS = {
    'rows': 'the key whose value is the list of listings (null if the result describes one listing)',
    'id': 'per listing, the key of the code other tools use to refer to it',
    'price': 'per listing, the price key (null if none)',
    'stock': 'per listing, the key saying if it is available to buy (null if none)',
    'category': 'per listing, the key of its shelf or category label (null if none)',
    'facts': 'per listing, the key of its nested attribute block (null if none)',
    'count': 'the key of the total number of matches over all pages, not only this page (null if none)',
    'next': 'the key whose value fetches the page after this one (null if none)',
    'next_arg': 'which tool parameter accepts that value (null if none)',
    'units': 'per cart line, the key of its unit count (null if none)',
}
LAYOUT = (
    "An online store's tool gave the result shown at the end; its parameters are listed too. Tell the controller "
    "where each part sits, using the result's own key names. Reply with ONE JSON object only:\n{"
    + ',\n '.join(f'"{slot}": {text}' for slot, text in LAYOUT_SLOTS.items()) + '}'
)


class Shelf:
    """The store as the controller sees it: tools, roles, schemas, and every listing read so far."""

    def __init__(self, view: dict[str, Any], wire: Wire, trace: list[dict[str, Any]]):
        self.wire = wire
        self.trace = trace
        self.tools = [t for t in view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas: dict[str, dict[str, Any]] = {}
        for tool in self.tools:
            params = tool['function'].get('parameters')
            self.schemas[tool['function']['name']] = params if isinstance(params, dict) else {'type': 'object', 'properties': {}}
        self.roles: dict[str, str] = {}
        self.layouts: dict[str, dict[str, Any] | None] = {}
        self.keys: dict[str, str | None] = {'id': None, 'price': None, 'stock': None, 'category': None, 'facts': None}
        self.rows: dict[str, dict[str, Any]] = {}       # every listing seen, merged
        self.opened: dict[str, dict[str, Any]] = {}     # listings whose attributes were read
        self.order: list[str] = []                       # ids in the order first seen
        self.gone: set[str] = set()                      # listings the store said are sold out
        self.pages_read = 0
        self.layout_tries: dict[str, int] = {}

    # ---------------------------------------------------------------- tools and arguments
    def params(self, name: str) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        return schema.get('properties') or {}

    def tool_list(self, skip: set[str] | None = None, only: set[str] | None = None) -> list[dict[str, Any]]:
        out = []
        for t in self.tools:
            name = t['function']['name']
            if (skip and name in skip) or (only is not None and name not in only):
                continue
            out.append({'type': 'function', 'function': {k: v for k, v in t['function'].items() if k in ('name', 'description', 'parameters')}})
        return out

    def describe(self) -> str:
        return '\n'.join(f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_j(t['function'].get('parameters') or {})}" for t in self.tools)

    def shape(self, name: str, args: Any) -> tuple[dict[str, Any], list[str]]:
        """Fit arguments to the tool's JSON schema: types coerced, enums matched, unknown keys dropped, missing ones named."""
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        faults: list[str] = []
        if not isinstance(args, dict):
            return {}, ['the arguments were not one JSON object']
        out: dict[str, Any] = {}
        for key, value in args.items():
            spec = props.get(key)
            if not isinstance(spec, dict):
                if props:
                    faults.append(f'{key} is not a parameter of {name} (parameters: {", ".join(props)})')
                continue
            fixed, fault = self._fit(key, value, spec)
            if fault:
                faults.append(fault)
            elif fixed is not None:
                out[key] = fixed
        for key in schema.get('required') or []:
            if key not in out:
                spec = props.get(key) or {}
                if isinstance(spec.get('enum'), list) and len(spec['enum']) == 1:
                    out[key] = spec['enum'][0]
                else:
                    faults.append(f'{name} needs {key}')
        return out, faults

    def _fit(self, key: str, value: Any, spec: dict[str, Any]) -> tuple[Any, str]:
        kinds = _kinds(spec)
        enum = spec.get('enum')
        if isinstance(enum, list) and enum:
            if value in enum:
                return value, ''
            low = {str(e).lower(): e for e in enum}
            if str(value).lower() in low:
                return low[str(value).lower()], ''
            return None, f'{key} must be one of {", ".join(map(str, enum))}'
        if 'array' in kinds:
            items = value if isinstance(value, list) else [value]
            inner = spec.get('items') if isinstance(spec.get('items'), dict) else {}
            fixed = []
            for v in items:
                f, fault = self._fit(key, v, inner) if inner else (v, '')
                if fault:
                    return None, fault
                fixed.append(f)
            cap = spec.get('maxItems')
            if isinstance(cap, int) and len(fixed) > cap:
                return None, f'{key} takes at most {cap} entries'
            return fixed, ''
        if 'integer' in kinds and not isinstance(value, bool):
            n = _int(value) if not isinstance(value, float) else (int(value) if value.is_integer() else None)
            if n is None and isinstance(value, float):
                n = int(round(value))
            if n is not None:
                return n, ''
        if 'number' in kinds and not isinstance(value, bool):
            n = _num(value)
            if n is not None:
                return (int(n) if n.is_integer() else n), ''
        if 'boolean' in kinds:
            if isinstance(value, bool):
                return value, ''
            if str(value).lower() in ('true', 'false'):
                return str(value).lower() == 'true', ''
        if 'string' in kinds:
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                return str(value), ''
        if 'object' in kinds and isinstance(value, dict):
            return value, ''
        if not kinds:
            return value, ''
        return None, f'{key} must be {"/".join(kinds)}'

    def text_param(self, name: str) -> str | None:
        """The one free-text parameter a tool takes (the message to the shopper, the words of a search)."""
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and not v.get('enum')]
        required = [k for k in schema.get('required') or [] if k in texts]
        return required[0] if len(required) == 1 else texts[0] if len(texts) == 1 else (required or texts or [None])[0]

    def single_param(self, name: str) -> str | None:
        """The parameter that names one listing, for a tool that reads one listing."""
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and not v.get('enum')]
        need = [k for k in schema.get('required') or [] if k in texts]
        return need[0] if need else texts[0] if len(texts) == 1 else None

    def list_param(self, name: str) -> str | None:
        props = self.params(name)
        return next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)

    # ---------------------------------------------------------------- reading results
    def learn_layout(self, name: str, observation: Any) -> None:
        have = self.layouts.get(name)
        if (have and have.get('id')) or self.layout_tries.get(name, 0) >= 2 or not _has_rows(observation):
            return
        self.layout_tries[name] = self.layout_tries.get(name, 0) + 1
        sample = observation
        if isinstance(observation, dict):
            sample = {k: (v[:3] if isinstance(v, list) else v) for k, v in observation.items()}
        prompt = LAYOUT + '\n\nTOOL PARAMETERS:\n' + _j((self.schemas.get(name) or {}).get('properties') or {}) + '\n\nRESULT:\n' + _j(_clip(sample, 200))[:6000]
        got = self.wire.ask_json(prompt, [LIGHT, HEAVY], 900) or {}
        layout = {k: got.get(k) if isinstance(got.get(k), str) and got.get(k) else None for k in LAYOUT_SLOTS}
        self.layouts[name] = layout
        for key in (slot for slot, text in LAYOUT_SLOTS.items() if text.startswith('per listing')):
            if layout.get(key) and not self.keys.get(key):
                self.keys[key] = layout[key]
        self.trace.append({'kind': 'layout', 'tool': name, 'layout': layout})

    def ident(self, node: Any, key: str | None = None) -> str | None:
        if not isinstance(node, dict) or len(node) < 2:
            return None
        key = key or self.keys.get('id')
        if key and _idlike(node.get(key)):
            return node[key].strip()
        if key:
            return None
        found = [v for v in node.values() if _idlike(v)]
        return found[0].strip() if len(found) == 1 and len(node) >= 3 else None

    def records(self, observation: Any, depth: int = 0, key: str | None = None) -> list[tuple[str, dict[str, Any]]]:
        if depth > 5:
            return []
        out: list[tuple[str, dict[str, Any]]] = []
        if isinstance(observation, dict):
            ident = self.ident(observation, key)
            if ident:
                return [(ident, observation)]
            for value in observation.values():
                out += self.records(value, depth + 1, key)
        elif isinstance(observation, list):
            for value in observation:
                out += self.records(value, depth + 1, key)
        return out

    def absorb(self, observation: Any) -> list[str]:
        seen = []
        facts_key = self.keys.get('facts')
        for ident, record in self.records(observation):
            if ident not in self.rows:
                self.rows[ident] = {}
                self.order.append(ident)
            old = self.rows[ident]
            old.update(record)
            if (facts_key and isinstance(record.get(facts_key), dict) and record.get(facts_key)) or (not facts_key and any(isinstance(v, dict) and len(v) >= 2 for v in record.values())):
                self.opened[ident] = dict(old)
            elif ident in self.opened:
                self.opened[ident].update(record)
            if self.in_stock(ident) is False:
                self.gone.add(ident)
            seen.append(ident)
        return seen

    def price(self, ident: str) -> float | None:
        key = self.keys.get('price')
        return _num((self.rows.get(ident) or {}).get(key)) if key else None

    def in_stock(self, ident: str) -> bool | None:
        key = self.keys.get('stock')
        value = (self.rows.get(ident) or {}).get(key) if key else None
        if isinstance(value, bool):
            return value
        n = _num(value)
        return None if n is None else n > 0

    def buyable(self, ident: str) -> bool:
        return ident not in self.gone and self.in_stock(ident) is not False

    def category(self, ident: str) -> str:
        key = self.keys.get('category')
        return str((self.rows.get(ident) or {}).get(key) or '').strip() if key else ''

    def facts(self, ident: str) -> dict[str, Any]:
        record = self.opened.get(ident) or self.rows.get(ident) or {}
        key = self.keys.get('facts')
        block = record.get(key) if key else next((v for v in record.values() if isinstance(v, dict)), None)
        return block if isinstance(block, dict) else {}

    def card(self, ident: str, limit: int = 160, cap: int = 900) -> str:
        record = self.opened.get(ident) or self.rows.get(ident)
        return (_j(_clip(record, limit))[:cap]) if record else '(not seen)'

    def page_info(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        layout = self.layouts.get(name) or {}
        if not isinstance(observation, dict) or not layout.get('count') or not layout.get('next_arg'):
            return None
        observation = _holder(observation, layout.get('rows') or layout['count']) or observation
        rows = observation.get(layout.get('rows') or '') if layout.get('rows') else None
        count = _int(observation.get(layout['count']))
        nxt = observation.get(layout['next']) if layout.get('next') else None
        if not isinstance(rows, list) or not rows or count is None:
            return None
        if nxt is None or isinstance(nxt, bool) or not isinstance(nxt, (str, int)):
            return None
        return count, len(rows), layout['next_arg'], nxt


READ = (
    "An agent is about to shop in an online store using the tools listed below. From the shopper's request and the "
    "tool list, prepare its job sheet. Reply with ONE JSON object only:\n"
    '{"items": [{"name": the product as the request words it, "shelf": its store category if the request or a tool '
    'names one, else "", "quantity": how many units (1 if unsaid), "musts": ["property: required value", ...], '
    '"points": [positions (1, 2, ...) of the musts whose value comes from some other listing]}],\n'
    ' "budget": the money cap as a number in the unit the request uses, or null, "budget_scope": "total" or "each",\n'
    ' "pick": "cheapest" if the lowest price wins, "most" or "least" if the highest or lowest of some other measure '
    'wins, otherwise "any", "pick_measure": the measure for most/least, else "",\n'
    ' "in_stock_only": false only when the request accepts sold-out listings,\n'
    ' "messages": the number of messages to the shopper the request permits, or null,\n'
    ' "steps": the number of tool calls the request permits, or null,\n'
    ' "wishes": [nice-to-have preferences, for breaking ties only], "rules": [any other conditions on the order],\n'
    ' "process": [{"when": "choose" | "order" | "end", "step": something the request wants done on the way, '
    'for example asking approval or messaging before the order}],\n'
    ' "quiet_decline": true if closing without a purchase must not mention any listing,\n'
    ' "fits": [{"item": item number, "with": item number}] for items that must go together,\n'
    ' "roles": {"search": the tool that returns listings, "open": the tool that returns one listing\'s full record '
    'and attributes, "compare": a tool returning several listings side by side, or null, "cart": the tool that edits '
    'the cart, "cart_view": a tool that shows the cart, or null, "buy": the tool that places the order, "refuse": the '
    'tool that closes the job with no purchase, "ask": a tool that messages the shopper, or null},\n'
    ' "buy_args": arguments for placing the order with the buy tool, "refuse_args": arguments for closing with no '
    'purchase, "cart_add": arguments that add some units of one listing with the cart tool, writing "<ID>" for the '
    'listing and "<QTY>" for the unit count (if that tool takes a list of lines, a list holding that single line)}\n'
    "How to fill it: a must is a hard condition the request sets on that item, as \"property: value\", the value in the "
    "request's wording with units and bounds kept (\"width: at most 101 cm\"). When a must's value depends on another "
    "listing (equal to, different from or close to what some other product's page says), keep it as a single must "
    "whose value is the request's own phrase. Price limits, stock, quantity and the pick rule are never musts. If one "
    "tool both orders and closes without buying, name it in both roles and separate the two uses by their arguments."
)

NOTE = (
    "A store tool's result included the text below, outside of any listing, while an agent was shopping. Work out "
    "whether it changes the order. Reply with ONE JSON object only:\n"
    '{"change": true if it alters what the shopper wants or what is for sale, otherwise false,\n'
    ' "budget": the new money cap as a number, or null,\n'
    ' "drop": [{"item": item number, "property": a condition the shopper has let go of}],\n'
    ' "add": [{"item": item number, "need": "property: value" newly required}],\n'
    ' "sold_out": [ids of listings it says are no longer for sale],\n'
    ' "summary": a short plain description of the change, or ""}\n'
    "When the text speaks of the agent's pick, choice or suggestion without naming a listing, it means the listings "
    "in CART."
)


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []
        self.pointing: list[str] = []

    def in_force(self) -> list[str]:
        return [m for m in self.musts + self.told if m not in self.dropped]


class Task:
    """What the shopper wants, as it stands now."""

    def __init__(self, request: str):
        self.request = request
        self.items: list[Item] = []
        self.budget: float | None = None
        self.scope = 'total'
        self.pick = 'cheapest'
        self.pick_said = False
        self.measure = ''
        self.in_stock_only = True
        self.wishes: list[str] = []
        self.rules: list[str] = []
        self.process: list[dict[str, str]] = []
        self.quiet = False
        self.msg_limit = 0
        self.step_limit: int | None = None
        self.fits: list[tuple[int, int]] = []
        self.notes: list[str] = []
        self.version = 0

    def take(self, brief: dict[str, Any]) -> None:
        for raw in _list(brief.get('items')):
            if not isinstance(raw, dict) or not str(raw.get('name') or '').strip():
                continue
            musts = [str(m).strip() for m in _list(raw.get('musts')) if str(m).strip()]
            qty = _int(raw.get('quantity'))
            self.items.append(Item(str(raw['name']).strip(), str(raw.get('shelf') or '').strip(), qty if qty and qty > 0 else 1, musts))
        if not self.items:
            self.items.append(Item('the requested product', '', 1, []))
        budget = _num(brief.get('budget'))
        if budget and budget > 0:
            self.budget = budget
        self.scope = 'each' if str(brief.get('budget_scope') or '').lower() == 'each' else 'total'
        pick = str(brief.get('pick') or 'any').lower()
        self.pick_said = pick in ('cheapest', 'most', 'least')
        self.pick = pick if self.pick_said else 'cheapest'
        self.measure = str(brief.get('pick_measure') or '')
        self.in_stock_only = brief.get('in_stock_only') is not False
        self.wishes = [str(w) for w in _list(brief.get('wishes')) if str(w).strip()]
        self.rules = [str(r) for r in _list(brief.get('rules')) if str(r).strip()]
        for entry in _list(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or 'order').lower()
                self.process.append({'when': when if when in ('choose', 'order', 'end') else 'order', 'step': str(entry['step']).strip(), 'done': ''})
        self.quiet = brief.get('quiet_decline') is True
        limit = _int(brief.get('messages'))
        self.msg_limit = limit if limit is not None and limit >= 0 else 3
        steps = _int(brief.get('steps'))
        self.step_limit = steps if steps and steps > 0 else None
        for pair in _list(brief.get('fits')):
            if isinstance(pair, dict):
                a, b = _int(pair.get('item')), _int(pair.get('with'))
                if a and b and a != b and a <= len(self.items) and b <= len(self.items):
                    self.fits.append((a, b))

    def devices(self) -> set[int]:
        return {b for _, b in self.fits}

    def ledger(self) -> str:
        lines = []
        many = len(self.items) > 1
        for n, item in enumerate(self.items, 1):
            head = (f'item {n}: ' if many else 'item: ') + item.name + (f' (store label: {item.shelf})' if item.shelf else '') + f', quantity {item.qty}'
            force = item.in_force()
            head += '; requirements: ' + ('; '.join(force) if force else 'none beyond the kind of product')
            if item.free:
                head += '; shopper does not mind: ' + ', '.join(item.free[:12])
            lines.append(head)
        if self.budget is not None:
            lines.append(f'budget: at most {self.budget:g} ' + ('for each item' if self.scope == 'each' else 'in total'))
        lines.append({'cheapest': 'pick: the cheapest listing that fits' + ('' if self.pick_said else ' (no other ranking was asked for)'), 'most': f'pick: the most {self.measure} among fitting listings', 'least': f'pick: the least {self.measure} among fitting listings'}.get(self.pick, 'pick: any listing that fits everything'))
        if self.in_stock_only:
            lines.append('in stock only')
        lines += [f'wish (tie-breaker only): {w}' for w in self.wishes]
        lines += [f'rule: {r}' for r in self.rules]
        for a, b in self.fits:
            lines.append(f'item {a} must work with item {b}')
        if self.quiet:
            lines.append('closing with no purchase: mention no listing at all')
        for entry in self.process:
            lines.append(f"step the request asks for ({entry['when']}): {entry['step']}" + (' [done]' if entry.get('done') else ''))
        if self.notes:
            lines.append('changes so far: ' + ' | '.join(self.notes[-4:]))
        return '\n'.join('- ' + line for line in lines)

    def apply_change(self, got: dict[str, Any]) -> str:
        said = []
        budget = _num(got.get('budget'))
        if budget and budget > 0 and budget != self.budget:
            self.budget = budget
            said.append(f'budget now {budget:g}')
        for entry in _list(got.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                n = min(max(1, _int(entry.get('item')) or 1), len(self.items))
                item = self.items[n - 1]
                gone = _stems(str(entry['property']).split(':')[0])
                hit = [m for m in item.in_force() if gone and gone <= _stems(m.split(':')[0])]
                item.dropped += hit
                said += [f'dropped {m}' for m in hit]
        for entry in _list(got.get('add')):
            if isinstance(entry, dict) and ':' in str(entry.get('need') or ''):
                n = min(max(1, _int(entry.get('item')) or 1), len(self.items))
                need = str(entry['need']).strip()
                if need not in self.items[n - 1].told:
                    self.items[n - 1].told.append(need)
                    said.append(f'item {n} now needs {need}')
        if said:
            self.version += 1
            self.notes.append('; '.join(said))
        return '; '.join(said)


PLAN = (
    "You pick the product properties a shopping agent asks a shopper about. The shopper holds a few firm preferences "
    "they did not write down, each on one property of one requested item, and tells one only when a question names that "
    "property. For each item below you get what the request already settles, and the attribute keys seen on the item's "
    "opened listings (how many listings carry each, a few example values).\n"
    "Rank, for each item, the properties most likely to hide such a preference: first what the request's story hints at "
    "without giving a value (who will use it, how it should look, where it goes, what the shopper cannot judge), named "
    "by the item's own key for it when one fits; then the keys a buyer of that kind of product chooses by, among those "
    "most listings carry. Leave out what the request settles, identifiers, model names and logistics, and keys that "
    "only list a listing's variants or bundle several properties at once (a question about such a key reads as many "
    "questions). A label is the property's plain name in the key's own words.\n"
    'Answer with ONE JSON object only: {"ask": [{"item": item number, "label": "...", "hint": true when the story hints '
    'at it}]}, up to COUNT labels for each item, best first.'
)

HEARD = (
    "Below are a message a shopping agent sent to the shopper and the shopper's answer. Read the answer and reply "
    "with ONE JSON object only:\n"
    '{"needs": [{"item": item number, "need": "property: value", "points": true if the value refers to some other '
    'listing}],\n'
    ' "free": [{"item": item number, "property": a property the shopper is indifferent to}],\n'
    ' "drop": [{"item": item number, "property": a condition the shopper lets go of}],\n'
    ' "budget": the new money cap as a number, or null,\n'
    ' "one_at_a_time": true if the answer asks to be asked about one thing per message, otherwise false,\n'
    ' "approves": true if the answer accepts what the message sought approval for, false if it declines, else null}\n'
    "Write a need under the property the answer itself names, copying the value or bound exactly and keeping units "
    "(\"at least 3 shelves\", \"not wicker\"). If the answer gives a value by comparison with some other listing "
    "(equal to, different from or close to another product's value), record one need holding the answer's words from "
    "the comparison onward, with \"points\": true. A value hinted at in indirect or colourful language still counts. "
    "An answer that ignores the question gives no need."
)


class Asker:
    """The interviewer: when to ask, which properties to name, how many at once, and what the replies mean."""

    def __init__(self, task: Task, shelf: Shelf, wire: Wire, trace: list[dict[str, Any]]):
        self.task = task
        self.shelf = shelf
        self.wire = wire
        self.trace = trace
        self.used = 0
        self.asked = False
        self.single = False
        self.sent: list[tuple[int, str, str]] = []
        self.named: dict[int, set[str]] = {}
        self.broad: set[int] = set()          # items that got the open question
        self.record: list[str] = []

    def tool(self) -> str | None:
        return self.shelf.roles.get('ask')

    def free(self) -> int:
        return max(0, self.task.msg_limit - self.used)

    def owner(self, label: str) -> int:
        """The requested item a category label belongs to (0 when none)."""
        if not label:
            return 0
        words = _words(label)
        best, top = 0, 0.0
        for n, item in enumerate(self.task.items, 1):
            if words and words == _words(item.shelf):
                score = 3.0
            elif words and words == _words(item.name):
                score = 2.0
            else:
                mine = _stems(item.shelf) | _stems(item.name)
                theirs = _stems(label)
                score = len(mine & theirs) / max(1, len(mine | theirs))
            if score > top:
                best, top = n, score
        return best if top > 0 else (1 if len(self.task.items) == 1 else 0)

    def opened_for(self, number: int) -> list[str]:
        return [i for i in self.shelf.order if i in self.shelf.opened and self.owner(self.shelf.category(i)) == number]

    def ready(self, step: int, max_steps: int) -> bool:
        if self.asked or not self.tool() or self.free() <= 0 or max_steps - step <= 6:
            return False
        if all(len(self.opened_for(n)) >= OPEN_BEFORE_ASK for n in range(1, len(self.task.items) + 1)):
            return True
        return step >= max(6, max_steps // 6)

    # ---------------------------------------------------------------- which properties
    def key_table(self, number: int) -> list[tuple[str, int, list[str]]]:
        ids = self.opened_for(number) or ([i for i in self.shelf.order if i in self.shelf.opened] if len(self.task.items) == 1 else [])
        stats: dict[str, dict[str, Any]] = {}
        for ident in ids:
            for key, value in self.shelf.facts(ident).items():
                name = _words(key)
                if not name or isinstance(value, (dict, list)):
                    continue
                slot = stats.setdefault(name, {'n': 0, 'values': []})
                slot['n'] += 1
                slot['values'].append(str(value).strip().lower())
        bundles = self.bundles(ids)
        settled = [_stems(m.split(':', 1)[0]) for m in self.task.items[number - 1].musts]
        table = []
        for name, slot in stats.items():
            values = slot['values']
            mean = sum(len(v) for v in values) / max(1, len(values))
            if mean > 40:
                continue                     # free text, not a property one chooses
            if not any(values):
                continue                     # a key the listings leave empty
            if name in bundles:
                continue                     # several properties under one key
            if len(values) >= 3 and len(set(values)) == len(values) and all(_idlike(v.replace(' ', '')) for v in values):
                continue                     # an identifier: every listing has its own code
            if any(s and (s <= _stems(name) or _stems(name) <= s) for s in settled):
                continue                     # the request settles it
            table.append((name, slot['n'], list(dict.fromkeys(values))[:4]))
        table.sort(key=lambda r: (-r[1], len(r[0])))
        return table[:KEY_KEEP]

    def bundles(self, ids: list[str]) -> set[str]:
        """Keys whose value repeats other properties' values (one listing its colour, the next its model): a question
        about such a key reads as many questions at once. Keys equal on every listing count as one property."""
        rows = [{_words(k): str(v).strip().lower() for k, v in self.shelf.facts(i).items() if not isinstance(v, (dict, list)) and str(v).strip()} for i in ids]
        keys = {k for r in rows for k in r}
        def twins(a: str, b: str) -> bool:
            both = [r for r in rows if a in r and b in r]
            return bool(both) and all(r[a] == r[b] for r in both)
        out = set()
        for key in keys:
            groups: list[set[str]] = []
            hits = 0
            for r in rows:
                value = r.get(key)
                if not value:
                    continue
                same = [k for k, v in r.items() if k != key and len(v) >= 2 and not twins(key, k) and (v == value or v in value.split() or v in value.split(' / '))]
                if same:
                    hits += 1
                for k in same:
                    if not any(twins(k, g) or k in grp for grp in groups for g in grp):
                        groups.append({k})
            if len(groups) >= 2 and hits * 2 >= max(1, len(rows)):
                out.add(key)
        return out

    def lanes(self) -> dict[int, list[str]]:
        tables = {n: self.key_table(n) for n in range(1, len(self.task.items) + 1)}
        lines = []
        for n, item in enumerate(self.task.items, 1):
            lines.append(f'item {n}: {item.name}' + (f' (store label: {item.shelf})' if item.shelf else '') + ' - settled by the request: ' + ('; '.join(item.musts) or 'nothing'))
            for name, count, values in tables[n][:PLAN_SHOW]:
                lines.append(f'  - {name} (on {count}): ' + ' | '.join(v[:30] for v in values[:3]))
        prompt = PLAN.replace('COUNT', str(LIST_KEYS)) + '\n\nREQUEST:\n' + self.task.request + '\n\nITEMS AND KEYS:\n' + '\n'.join(lines)
        # with two messages or fewer the first pick is nearly all there is: the stronger model picks it
        models = [HEAVY, LIGHT] if self.task.msg_limit <= 2 else [LIGHT, HEAVY]
        got = self.wire.ask_json(prompt, models, 3000) or {}
        lanes: dict[int, list[str]] = {n: [] for n in tables}
        for pick in _list(got.get('ask')):
            if not isinstance(pick, dict):
                continue
            n = min(max(1, _int(pick.get('item')) or 1), len(self.task.items))
            label = _words(pick.get('label'))
            if label and len(label.split()) <= 5 and label not in lanes[n]:
                lanes[n].append(label)
        for n, table in tables.items():
            for name, _, _ in table:
                if name not in lanes[n]:
                    lanes[n].append(name)
        self.trace.append({'kind': 'plan', 'lanes': {n: v[:LIST_KEYS] for n, v in lanes.items()}})
        return lanes

    # ---------------------------------------------------------------- the questions
    def question(self, number: int, labels: list[str]) -> str:
        name = self.task.items[number - 1].name
        if len(labels) == 1:
            return f'What {labels[0]} would you like for the {name}?'
        return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {name}?"

    def open_question(self, number: int) -> str:
        return f'What else would you like the {self.task.items[number - 1].name} to have?'

    def send(self, text: str) -> str | None:
        tool = self.tool()
        if not tool:
            return None
        param = self.shelf.text_param(tool)
        if not param:
            return None
        args, faults = self.shelf.shape(tool, {param: text})
        if faults:
            return None
        reply = self.wire.store(tool, args)
        if reply is None:
            return None
        self.used += 1
        said = self.spoken(reply.get('observation'))
        self.record.append(f'step {self.wire.step}: message {text!r} -> reply {said[:300]!r}')
        self.trace.append({'kind': 'ask', 'q': text, 'a': said[:300]})
        return said

    @staticmethod
    def spoken(observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, str) and len(node.split()) >= 2:
                found.append(node)
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        return ' '.join(found) if found else _j(observation)

    def hear(self, number: int, text: str, said: str) -> dict[str, Any]:
        items = '\n'.join(f'item {n}: {it.name}' for n, it in enumerate(self.task.items, 1))
        prompt = HEARD + f'\n\nITEMS:\n{items}\n\nMESSAGE (about item {number}):\n{text}\n\nREPLY:\n{said}'
        return self.wire.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}

    def take(self, heard: dict[str, Any]) -> list[str]:
        told = []
        for entry in _list(heard.get('needs')):
            if not isinstance(entry, dict) or ':' not in str(entry.get('need') or ''):
                continue
            n = min(max(1, _int(entry.get('item')) or 1), len(self.task.items))
            item = self.task.items[n - 1]
            need = str(entry['need']).strip()
            if need not in item.told:
                item.told.append(need)
                told.append(f'item {n}: {need}' if len(self.task.items) > 1 else need)
                if entry.get('points') is True:
                    item.pointing.append(need)
        for entry in _list(heard.get('free')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                n = min(max(1, _int(entry.get('item')) or 1), len(self.task.items))
                self.task.items[n - 1].free.append(_words(entry['property']))
        change = self.task.apply_change({'budget': heard.get('budget'), 'drop': heard.get('drop')})
        if change:
            told.append(change)
        if told:
            self.task.version += 1
        return told

    def interview(self, step: int, max_steps: int) -> str:
        self.asked = True
        if self.free() <= 0 or not self.tool():
            return ''
        if self.task.msg_limit < LIST_MIN_MESSAGES:
            self.single = True
        lanes = self.lanes()
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        devices = self.task.devices()
        told_all: list[str] = []
        lines = []
        # two messages for three or more products: the hidden needs sit mostly on the first one read (race-13 Ask
        # corpus: 23 of 35), so it keeps the second message too unless its first answer already told one
        focus = min(lanes) if self.task.msg_limit <= 2 and len(lanes) >= 3 else None
        while self.free() > 0 and not self.wire.finished and max_steps - self.wire.step > 6 and self.wire.seconds_left() > WRAP_S:
            order = sorted(lanes, key=lambda n: (spent[n] > 0, found[n] > 0, spent[n], n not in devices, n))
            if focus is not None and not found[focus]:
                order = [focus] + [n for n in order if n != focus]
            pick = None
            for n in order:
                left = [label for label in lanes[n] if label not in self.named.setdefault(n, set())]
                if left:
                    pick = (n, left)
                    break
            if pick:
                number, left = pick
                labels = left[:1 if self.single else LIST_KEYS]
                text = self.question(number, labels)
            else:
                # every named property was asked: the messages left go to one open question per item
                rest = sorted((n for n in lanes if n not in self.broad), key=lambda n: (spent[n] > 0, n))
                if not rest:
                    break
                number, labels = rest[0], []
                self.broad.add(number)
                text = self.open_question(number)
            said = self.send(text)
            if said is None:
                break
            spent[number] += 1
            self.named.setdefault(number, set()).update(labels)
            heard = self.hear(number, text, said)
            told = self.take(heard)
            lines.append(f'Q: {text}\nA: {said}')
            if told:
                told_all += told
                for t in told:
                    owner = number
                    m = re.match(r'item (\d+): ', t)
                    if m:
                        owner = int(m.group(1))
                    if owner in found:
                        found[owner] += 1
            elif heard.get('one_at_a_time') is True and len(labels) > 1 and not self.single:
                # this shopper wants one property per message: the rest go one by one, this list's ones again first
                self.single = True
                self.named[number].difference_update(labels[1:])
                self.trace.append({'kind': 'single', 'after': text[:80]})
        searched = self.value_search([n for n in found if found[n]])
        summary = '[ctl] Shopper interview (one step per message):\n' + '\n'.join(lines)
        summary += '\nRequirements added by these answers: ' + ('; '.join(told_all) if told_all else 'none')
        summary += '. Every pick has to meet them from now on; search again with the told values where that helps.'
        pointed = [p for item in self.task.items for p in item.pointing if any(p in t for t in told_all)]
        if pointed:
            summary += ' Their value is borrowed from another listing: ' + ' | '.join(pointed) + ". Open that listing's full page and each page it refers to; the value on the final page is the one to use."
        if searched:
            summary += '\n' + searched
        return summary

    # ---------------------------------------------------------------- one search with the told values
    def value_search(self, numbers: list[int]) -> str:
        tool = self.shelf.roles.get('search')
        plan = self.search_plan()
        if not tool or not plan or not numbers or self.wire.seconds_left() < WRAP_S + 40:
            return ''
        base, field = plan
        notes = []
        for number in numbers[:2]:
            item = self.task.items[number - 1]
            values = [t.split(':', 1)[1].strip() for t in item.told if ':' in t and t not in item.pointing]
            values = [v for v in values if v and not re.search(r'\d', v) and len(v.split()) <= 4]
            if not values:
                continue
            args, faults = self.shelf.shape(tool, {**base, field: ' '.join(values[:2] + [item.name.split(' and ')[0]])})
            if faults:
                continue
            reply = self.wire.store(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.shelf.learn_layout(tool, seen)
            found = self.shelf.absorb(seen)
            if self.task.pick == 'cheapest' and self.task.pick_said and found:
                found += self.more_pages(tool, args, seen)
            rows = [i for i in dict.fromkeys(found) if self.shelf.buyable(i) and self.owner(self.shelf.category(i)) in (number, 0)]
            if self.task.pick == 'cheapest':
                rows.sort(key=lambda i: self.shelf.price(i) if self.shelf.price(i) is not None else float('inf'))
            notes.append(f"{_j(args.get(field))}: " + (', '.join(rows[:6]) or 'no row of that kind'))
        if not notes:
            return ''
        return "[ctl] The controller searched each item's told values and found: " + ' | '.join(notes)

    def search_plan(self) -> tuple[dict[str, Any], str] | None:
        tool = self.shelf.roles.get('search')
        if not tool:
            return None
        field = self.shelf.text_param(tool)
        if not field:
            return None
        props = (self.shelf.schemas.get(tool) or {}).get('properties') or {}
        base = {}
        for key in (self.shelf.schemas.get(tool) or {}).get('required') or []:
            if key != field and isinstance((props.get(key) or {}).get('enum'), list):
                base[key] = props[key]['enum'][0]
        return base, field

    def more_pages(self, tool: str, args: dict[str, Any], observation: Any) -> list[str]:
        plan = self.shelf.page_info(tool, observation)
        if not plan:
            return []
        count, shown, param, value = plan
        found: list[str] = []
        for _ in range(2):
            if count <= shown or self.shelf.pages_read >= PAGES_CAP or value is None:
                break
            call, faults = self.shelf.shape(tool, {**args, param: value})
            if faults:
                break
            reply = self.wire.store(tool, call)
            if reply is None:
                break
            self.shelf.pages_read += 1
            seen = reply.get('observation')
            found += self.shelf.absorb(seen)
            after = self.shelf.page_info(tool, seen)
            value = after[3] if after else None
            shown += len(self.shelf.records(seen))
        return found


GUIDE = (
    "Your job: buy what one shopper asked for in an online store, working only through the store's tools. Each "
    "episode is graded pass or fail. It passes when you check out a cart with every requested item at the asked "
    "quantity, every line meeting all requirements that hold at checkout time and the total within the budget, or "
    "when, the store truly lacking a fitting listing, you close without a purchase in the way the request specifies.\n\n"
    "HOW THE STORE BEHAVES\n"
    "- A tool call costs one step and your steps are capped. Argument names and allowed values must match each "
    "tool's schema exactly.\n"
    "- Searching may be restricted: category filters, keyword modes that look only at the start of the text, small "
    "pages. The best queries are short and lead with the wanted value, then the product noun. An empty filtered "
    "search usually means a wrong filter value or parameter; copy labels exactly as result rows spell them.\n"
    "- Rows in search results only summarise. Decide properties from the listing's full record and its attribute "
    "block; open the full record of anything you might buy, and skip reopening a listing nothing has changed about.\n"
    "- Messages to the shopper are not yours to send. The controller's interviewer does that and folds the answers "
    "into the requirements.\n\n"
    "DECIDING WHETHER A LISTING FITS\n"
    "- A listing's category label tells what kind of product it is; only listings under the requested item's label qualify.\n"
    "- A property requirement holds when the attribute for that property gives the value. Spelling of the key, "
    "letter case, singular or plural and everyday synonyms are irrelevant; matching one entry of a list is enough, as "
    "is a value that contains the required words. Convert units first; a bound includes its own value; a range has "
    "to sit within a bound. The attribute for the property wins over the title; an attribute about a different part "
    "(the colour of a strap is not the colour of the bag) settles nothing. With no such attribute, a value stated in "
    "another attribute, the title or the description counts.\n"
    "- Some requirements borrow their value from another listing (equal to, different from or close to what a "
    "different product shows). Locate that listing, open its full page, follow every page it refers you to, and take "
    "the value from the final page; only an order placed after opening those pages counts.\n"
    "- Items that must work together are chosen as a pair, from evidence on both pages (a compatibility list, a "
    "model code printed on each).\n"
    "- Honour exclusions. Wishes matter only to break a tie. A listing marked sold out is never bought. Price times "
    "quantity, summed over the lines, must stay within the budget.\n\n"
    "PICKING AMONG FITTING LISTINGS\n"
    "- Cheapest: the winner is the lowest price among every fitting listing, so survey the whole fitting set (turn "
    "pages; a search with a maximum price lets you list only cheaper ones). Most or least of a measure: compare that "
    "measure across all fitting listings. Any other request: any fully fitting listing will do.\n"
    "- Put an item in the cart as soon as you have opened a listing that fits it (for cheapest / most / least, once "
    "the comparison is done) and go on to the next item; listings in a cart can sell out, and carting early keeps "
    "steps in hand to swap them. Each item gets one line at its own quantity.\n\n"
    "WHEN SOMETHING CHANGES\n"
    "- Any tool result, most often at checkout, can bring a new budget, a requirement removed or added, a new price or "
    "a sold-out notice. Take it into account, repair the cart, and check out once more.\n\n"
    "ENDING\n"
    "- The cart must hold just the chosen lines; view it one time, then send the checkout call alone.\n"
    "- Close without buying only when a genuine search proves some item has no fitting listing. Running low on steps "
    "proves nothing, so in that case check out the best cart you have.\n"
    "- Call a tool in every reply, and keep your words few: the facts you hold and your next action.\n"
    "- Text tagged [ctl] inside tool results is written by the controller, the program running this episode "
    "alongside you; it is not from the store, a listing or the shopper. It tells you the time left, the steps left "
    "and the results of its checks. Do as it says."
)

CHECK = (
    "Before a shopping agent's order goes through, review it. Using only the listing records given below, test the "
    "cart against the request and every requirement that currently applies. Reply with ONE JSON object only:\n"
    '{"ok": false when some line misfits or something is missing, otherwise true, "problems": ["brief plain '
    'findings"], "better": [ids from OTHER LISTINGS that fit where a cart line fails, or that fit and cost less when '
    "the request wants the cheapest]}\n"
    "Decide each requirement from the attribute for its property. A value that refers to another listing is met only "
    "if that listing's page is among the records and gives the value. Never fault a line over a property that no "
    "requirement mentions."
)

SCREEN = (
    "A shopping agent is choosing a listing for one item of a request. The rows below come from search results, so "
    "most of their details are missing. Name the rows worth opening in full: rows whose data contradicts no "
    "requirement and that could meet every one of them. WHY says what the agent is looking for. Reply with ONE JSON "
    'object only: {"open": [ids, most promising first, at most 4]}'
)

AUDIT = (
    "A shopping agent is about to close without buying, believing no listing fits. Go through the listings it has "
    "read for each requested item and sort them. Reply with ONE JSON object only:\n"
    '{"fits": {"<item number>": [ids that satisfy all of the item\'s requirements]}, '
    '"near": {"<item number>": [ids that can be bought within the budget and conflict with no requirement, even when '
    "their record is silent on some requirement, best first]}}\n"
    "A conflict means the record gives a different value for that property, the listing is sold out, or it breaks the "
    "budget. Count only listings of that item's kind."
)

STEPS = (
    "The agent is getting ready to STAGE. Along the way the request wants the steps listed below. Using the numbered "
    "RECORD of actions so far, decide if every step due at this point has been done for the current cart (meaning a "
    "message after the last cart change got a reply that grants it). If one is still open, write the single short "
    "plain message to the shopper that completes it, or leave the message empty when the remaining messages cannot. "
    'Reply with ONE JSON object only: {"done": true or false, "message": "text to send, or empty"}'
)


class Episode:
    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.view = environment['policy_view']
        self.trace: list[dict[str, Any]] = []
        self.wire = Wire(dict(environment['binding']), self.trace)
        self.wire.finished = bool(self.view.get('finished'))
        self.shelf = Shelf(self.view, self.wire, self.trace)
        self.task = Task(str(self.view.get('query') or ''))
        self.asker = Asker(self.task, self.shelf, self.wire, self.trace)
        steps = _int(self.view.get('max_steps'))
        self.max_steps = steps if steps and steps > 0 else 30
        self.chat: list[dict[str, Any]] = []
        self.turns = 0
        self.cut_to = 2
        self.cut_size = 0
        self.cart: dict[str, int] = {}        # cart lines as the controller tracks them
        self.cart_seen_at = -1                # step of the last cart view
        self.cart_changed_at = 0
        self.built: list[str] = []            # the last full set of lines the cart held
        self.cart_template: tuple[str, dict[str, Any]] | None = None
        self.cache: dict[str, tuple[int, int, str]] = {}
        self.recent: list[str] = []
        self.checks: set[str] = set()
        self.check_holds = 0
        self.line_holds = 0
        self.patched = False                  # the controller carted the missing items once
        self.buy_tried = False                # a checkout has been sent to the store
        self.value_swept: set[int] = set()      # items that had their one single-value search
        self.pending_notes: list[str] = []    # store notes met by the controller's own calls
        self.cart_remove: dict[str, Any] | None = None   # learned call that takes a line out
        self.unit_fixes = 0
        self.heavy_checks = 0
        self.decline_tries = 0
        self.pushed = False
        self.leads: list[str] = []
        self.seen_notes: set[str] = set()
        self.note_calls = 0
        self.process_rounds = 0
        self.searched: list[dict[str, Any]] = []
        self.buy_args: dict[str, Any] = {}
        self.refuse_args: dict[str, Any] = {}
        self.cart_add: dict[str, Any] = {}

    # ---------------------------------------------------------------- basics
    def left(self) -> int:
        return self.max_steps - self.wire.step

    def tail(self) -> int:
        return 2 + (1 if self.shelf.roles.get('cart_view') else 0)

    def late(self) -> bool:
        return self.wire.seconds_left() < WRAP_S

    def read_request(self) -> None:
        prompt = READ + '\n\nTOOLS:\n' + self.shelf.describe() + '\n\nREQUEST:\n' + self.task.request
        if self.view.get('opening') is not None:
            prompt += '\n\nTHE STORE SHOWS FIRST:\n' + _j(_clip(self.view.get('opening'), 1500))
        brief = self.wire.ask_json(prompt, [LIGHT, HEAVY], 4000) or {}
        names = set(self.shelf.schemas)
        roles = brief['roles'] if isinstance(brief.get('roles'), dict) else {}
        self.shelf.roles = {r: str(t) for r, t in roles.items() if isinstance(t, str) and t in names}
        if not self.shelf.roles.get('buy') or not self.shelf.roles.get('cart') or not self.shelf.roles.get('search'):
            again = self.wire.ask_json(prompt, [HEAVY], 3000) or {}
            if isinstance(again.get('roles'), dict):
                brief = again
                self.shelf.roles = {r: str(t) for r, t in again['roles'].items() if isinstance(t, str) and t in names}
        self.task.take(brief)
        for n, raw in enumerate([r for r in _list(brief.get('items')) if isinstance(r, dict) and str(r.get('name') or '').strip()], 1):
            if n <= len(self.task.items):
                marks = [str(p) for p in _list(raw.get('points'))]
                item = self.task.items[n - 1]
                item.pointing = [m for k, m in enumerate(item.musts) if str(k) in marks or str(k + 1) in marks or m in marks]
        if not self.shelf.roles.get('ask'):
            self.task.msg_limit = 0
        self.buy_args = brief.get('buy_args') if isinstance(brief.get('buy_args'), dict) else {}
        self.refuse_args = brief.get('refuse_args') if isinstance(brief.get('refuse_args'), dict) else {}
        add = brief.get('cart_add')
        self.cart_add = add if isinstance(add, dict) and '"<ID>"' in _j(add) else {}
        if self.task.step_limit and 10 <= self.task.step_limit < self.max_steps:
            self.max_steps = self.task.step_limit
        self.trace.append({'kind': 'brief', 'roles': self.shelf.roles, 'items': [(i.name, i.shelf, i.qty, i.musts) for i in self.task.items],
                           'budget': self.task.budget, 'pick': self.task.pick, 'messages': self.task.msg_limit})

    # ---------------------------------------------------------------- status and history
    def status(self) -> str:
        left = self.left()
        text = f'[ctl] step {self.wire.step} of {self.max_steps} used, {left} left.'
        if self.late():
            text += ' THE CLOCK IS ALMOST OUT: fill the cart with the chosen lines and check out at once.'
        elif left <= self.tail() + 1:
            text += ' LAST STEPS: once every item has its cart line, send the checkout.'
        elif left <= 6 or (left <= len(self.task.items) + 5 and not self.cart):
            text += ' Steps are running low: put the chosen lines in the cart, then check out.'
        elif not self.cart and self.wire.step >= self.max_steps // 2:
            text += " Half of the steps are gone with nothing in the cart: add a fitting listing for every item now."
        if self.leads:
            text += '\n[ctl] likely fits already read: ' + ', '.join(self.leads[:6])
        if self.cart:
            text += '\n[ctl] cart as tracked: ' + ', '.join(f'{i} x{u}' for i, u in self.cart.items())
        return text + '\n[ctl] what the shopper wants now:\n' + self.task.ledger()

    def show(self, observation: Any, rows_short: bool = True) -> str:
        if rows_short and isinstance(observation, dict):
            clipped = {k: ([_clip(r, ROW_TEXT) for r in v] if isinstance(v, list) and sum(isinstance(r, dict) for r in v) >= 2 else _clip(v, TEXT_CLIP)) for k, v in observation.items()}
        else:
            clipped = _clip(observation, TEXT_CLIP)
        text = _j(clipped)
        return text if len(text) <= RESULT_CAP else text[:RESULT_CAP] + '...(cut)'

    def size(self) -> int:
        return sum(len(_j(m.get('content'))) for m in self.chat)

    def cut(self) -> None:
        # the prompt cache reads an unchanged history cheaply, and a change to an older message makes the rest be
        # written again: cut rarely and deeply (older results become short digests), and never touch a cut message again
        size = self.size()
        if size <= max(CUT_AT, self.cut_size + CUT_GAP):
            return
        turns = [i for i, m in enumerate(self.chat) if m.get('role') == 'assistant']
        if len(turns) <= CUT_KEEP or turns[-CUT_KEEP] <= self.cut_to:
            return
        edge = turns[-CUT_KEEP]
        for index in range(max(2, self.cut_to), edge):
            message = self.chat[index]
            content = message.get('content')
            if not isinstance(content, str) or not content:
                continue
            if message.get('role') == 'tool':
                message['content'] = self.digest(content)
            elif message.get('role') == 'assistant' and len(content) > 400:
                message['content'] = content[:400] + '...'
            elif message.get('role') == 'user':
                mark = content.rfind('\n[ctl] step ')
                message['content'] = content[:mark] if mark > 0 else content
        self.cut_to = edge
        self.cut_size = self.size()
        self.trace.append({'kind': 'cut', 'before': size, 'after': self.cut_size})

    def digest(self, content: str) -> str:
        mark = content.rfind('\n[ctl] step ')
        body = content[:mark] if mark >= 0 else content
        if len(body) <= 600:
            return body
        ids = [i for i in dict.fromkeys(re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{4,40})"', body)) if i in self.shelf.rows][:20]
        notes = [line[:300] for line in body.split('\n')[1:] if line.startswith('[')][:5]
        if not ids:
            return body[:600] + '...'
        lines = ['[ctl] earlier result in short form; listings as last seen:']
        for ident in ids:
            row = self.shelf.rows.get(ident) or {}
            flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
            lines.append((ident + ' ' + _j(_clip(flat, 50)) + (' (sold out)' if ident in self.shelf.gone else ''))[:DIGEST_ROW])
        return '\n'.join(lines + notes)

    # ---------------------------------------------------------------- notes inside results
    def loose_text(self, observation: Any) -> list[str]:
        out: list[str] = []
        inside = {id(r) for _, r in self.shelf.records(observation)}

        def walk(node: Any, depth: int) -> None:
            if depth > 5 or id(node) in inside:
                return
            if isinstance(node, str) and len(node.split()) >= 5:
                out.append(node.strip())
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        return [t for t in out if t not in self.seen_notes]

    def read_notes(self, observation: Any) -> str:
        texts = self.loose_text(observation)
        if not texts or self.note_calls >= 6:
            return ''
        self.seen_notes.update(texts)
        self.note_calls += 1
        items = '\n'.join(f'item {n}: {it.name} - requirements: ' + '; '.join(it.in_force()) for n, it in enumerate(self.task.items, 1))
        cart = ', '.join(f'{i} ' + self.shelf.card(i, 60, 160) for i in self.cart) or 'empty'
        prompt = NOTE + f'\n\nITEMS:\n{items}\nBUDGET: {self.task.budget}\nCART: {cart}\n\nTEXT:\n' + '\n'.join(texts)[:3000]
        got = self.wire.ask_json(prompt, [LIGHT, SPARE], 900) or {}
        if got.get('change') is not True:
            return ''
        for ident in _list(got.get('sold_out')):
            if isinstance(ident, str) and ident in self.shelf.rows:
                self.shelf.gone.add(ident)
                if ident in self.cart:
                    self.cart_seen_at = -1
        said = self.task.apply_change(got)
        summary = str(got.get('summary') or '').strip() or said
        if summary and summary not in self.task.notes:
            self.task.notes.append(summary[:200])
        return summary

    # ---------------------------------------------------------------- the main loop
    def run(self) -> list[dict[str, Any]]:
        if self.wire.finished:
            return self.trace
        self.read_request()
        opening = self.view.get('opening')
        if opening is not None:
            self.shelf.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.task.request
        if opening is not None:
            intro += '\n\nTHE STORE SHOWS FIRST:\n' + self.show(opening)
        intro += ("\n\nFirst search every requested item's category and open a few of its listings in full: the interviewer "
                  "needs to see which attributes these listings carry.\n" + self.status())
        self.chat = [{'role': 'system', 'content': GUIDE}]
        self.say('user', intro)
        self.work(LOOP_TURNS, HARD_SPEND)
        if not self.wire.finished:
            said = self.finish(again=True)
            if said and not self.wire.finished and not self.wire.no_credit and self.left() > 1:
                self.chat.append({'role': 'user', 'content': '[ctl] The controller tried to end the episode and ' + said + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.turns + 10, HARD_SPEND + 0.006)
            if not self.wire.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.wire.finished, 'steps': self.wire.step, 'spend': round(self.wire.spend, 5), 'calls': self.wire.calls})
        return self.trace

    def choose_steps(self) -> str:
        """Steps the request wants done before listings are chosen (for example a question it asks to be asked)."""
        said = self.steps_due('choose')
        for p in self.task.process:
            if p['when'] == 'choose':
                p['done'] = p.get('done') or 'sent'
        return ('[ctl] Before choosing: ' + said + '\n') if said else ''

    def say(self, role: str, content: Any, **extra: Any) -> None:
        self.chat.append(dict(extra, role=role, content=content))

    def offered(self) -> list[dict[str, Any]]:
        skip = {self.shelf.roles.get('ask')} - {None}
        r = self.shelf.roles
        if self.late() or self.left() <= self.tail() + 1:
            only = {r.get('cart'), r.get('cart_view'), r.get('buy'), r.get('refuse'), r.get('open')} - {None}
            return self.shelf.tool_list(skip=skip, only=only)
        return self.shelf.tool_list(skip=skip)

    def work(self, turn_cap: int, spend_cap: float) -> None:
        idle = misses = dry = 0
        while not self.wire.finished and self.left() > 0 and self.turns < turn_cap and self.wire.spend < spend_cap and self.wire.seconds_left() > FORCE_S:
            if self.asker.ready(self.wire.step, self.max_steps):
                summary = self.choose_steps() + self.asker.interview(self.wire.step, self.max_steps)
                if summary:
                    self.say('user', summary + '\n' + self.status())
                continue
            self.cut()
            self.turns += 1
            message = self.wire.model(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                misses += 1
                if self.wire.no_credit or misses > 2:
                    break
                continue
            calls = []
            for number, call in enumerate(c for c in message.get('tool_calls') or [] if isinstance(c, dict)):
                fn = call['function'] if isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else _obj(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f'c{self.turns}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _j(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            content = content if isinstance(content, str) else (_j(content) if content else '')
            if not calls:
                idle += 1
                self.chat.append({'role': 'assistant', 'content': content or '-'})
                self.chat.append({'role': 'user', 'content': '[ctl] That reply called no tool; every reply needs a tool call.\n' + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            self.say('assistant', content or None, tool_calls=calls)
            self.trace.append({'kind': 'move', 'said': content[:300], 'tools': [c['function'] for c in calls]})
            replies = self.guard_calls(calls)
            dry = dry + 1 if replies and all(text.startswith(UNSPENT) for _, text in replies) else 0
            if dry >= 6:
                self.trace.append({'kind': 'dry_stop', 'turn': self.turns})
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.status()
                self.say('tool', text, tool_call_id=cid)
            if dry >= 6:
                break

    # ---------------------------------------------------------------- the guard layer
    def orders_now(self, name: str, args: dict[str, Any]) -> bool:
        r = self.shelf.roles
        if name != r.get('buy'):
            return False
        if r.get('buy') != r.get('refuse'):
            return True
        return all(args.get(k) == v for k, v in self.buy_args.items()) if self.buy_args else not all(args.get(k) == v for k, v in self.refuse_args.items())

    def closes_empty(self, name: str, args: dict[str, Any]) -> bool:
        r = self.shelf.roles
        return name == r.get('refuse') and not self.orders_now(name, args)

    def guard_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        r = self.shelf.roles
        names = set(self.shelf.schemas)
        ending_in_turn = any(c['function']['name'] in (r.get('buy'), r.get('refuse')) for c in calls) and len(calls) > 1
        for call in calls:
            cid, name = call['id'], call['function']['name']
            parsed = _obj(call['function'].get('arguments')) or {}
            if self.wire.finished:
                out.append((cid, SKIP + 'the episode has ended.'))
                continue
            if name not in names:
                out.append((cid, SKIP + f"no tool by that name. Tools: {', '.join(sorted(names))}."))
                continue
            if name == r.get('ask'):
                out.append((cid, SKIP + "shopper messages go through the controller's interviewer only."))
                continue
            args, faults = self.shelf.shape(name, parsed)
            if faults:
                out.append((cid, SKIP + '; '.join(faults[:6])))
                continue
            buying, refusing = self.orders_now(name, args), self.closes_empty(name, args)
            if (buying or refusing) and ending_in_turn:
                out.append((cid, SKIP + 'the closing call must be sent alone, once the other results are read.'))
                continue
            if not (buying or refusing) and name not in (r.get('cart'), r.get('cart_view')) and self.left() <= self.tail() + (0 if self.buy_tried else 3):
                out.append((cid, SKIP + 'the remaining steps are reserved for the cart and the checkout (send the checkout now: the store may answer it with a change that needs a few steps).'))
                continue
            commit = name == r.get('cart') or buying or refusing
            if commit and not self.asker.asked and self.asker.tool() and self.asker.free() > 0 and self.left() > 8 and self.wire.seconds_left() > WRAP_S + 30:
                summary = self.choose_steps() + self.asker.interview(self.wire.step, self.max_steps)
                out.append((cid, SKIP + 'the shopper was interviewed first. Repeat this call if it still suits the answers.\n' + summary))
                continue
            mark = name + _j(args)
            cached = self.cache.get(mark)
            if cached and cached[0] == self.task.version and cached[1] >= self.cart_changed_at and name not in (r.get('cart'), r.get('buy'), r.get('refuse')):
                out.append((cid, SKIP + f'this call already ran at step {cached[1]} with nothing changed since; its result is below. Choose another call.\n' + cached[2][:3000]))
                continue
            if buying:
                hold = self.checkout_gate()
                if hold:
                    out.append((cid, HOLD + hold))
                    continue
            if refusing:
                hold = self.decline_gate()
                if hold:
                    out.append((cid, hold if hold.startswith('[done]') else HOLD + hold))
                    if self.wire.finished:
                        break
                    continue
                args = self.quiet_args(name, args)
            if name == r.get('cart'):
                args = self.full_removal(args)
            reply = self.wire.store(name, args)
            if reply is None:
                out.append((cid, 'No answer came from the store (network or server trouble); the call can be repeated.'))
                continue
            out.append((cid, self.after_call(name, args, reply)))
        return out

    def after_call(self, name: str, args: dict[str, Any], reply: dict[str, Any]) -> str:
        r = self.shelf.roles
        observation = reply.get('observation')
        if self.shelf.records(observation) or name in (r.get('search'), r.get('open'), r.get('compare'), r.get('cart_view'), r.get('cart')):
            self.shelf.learn_layout(name, observation)
        found = self.shelf.absorb(observation)
        extra = ''
        if name == r.get('search') and found:
            self.searched = (self.searched + [dict(args)])[-8:]
            if (self.task.pick_said or self.max_steps >= 40) and self.left() - self.tail() >= 12:
                more = self.asker.more_pages(name, args, observation)
                if more:
                    extra += f'\n[ctl] The controller read the next pages of this search itself ({len(more)} more rows): ' + ', '.join(i for i in dict.fromkeys(more) if self.shelf.buyable(i))[:1500]
            if self.task.pick == 'cheapest':
                rows = sorted(((self.shelf.price(i), i) for i in found if self.shelf.buyable(i) and self.shelf.price(i) is not None))
                if len(rows) >= 2:
                    extra += '\n[ctl] buyable rows here, cheapest first: ' + ', '.join(f'{i} ({p:g})' for p, i in rows[:6])
        if name == r.get('cart') and not (isinstance(observation, dict) and observation.get('error')):
            self.track_cart(args, observation)
        if name == r.get('cart_view'):
            self.read_cart(observation)
        note = self.read_notes(observation)
        text = self.show(observation) + extra
        if note:
            text += '\n[ctl] note read: ' + note + '. Apply this change before checking out.'
        if self.orders_now(name, args):
            self.buy_tried = True
        if self.orders_now(name, args) and not reply.get('finished'):
            self.trace.append({'kind': 'buy_refused'})
            self.cart_seen_at = -1
            self.cart_changed_at = max(self.cart_changed_at, 0)
        if name in (r.get('search'), r.get('open'), r.get('compare'), r.get('cart_view')):
            self.cache[name + _j(args)] = (self.task.version, self.wire.step, text)
        if reply.get('finished'):
            text += ' [the episode is over]'
        return text

    def quiet_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not self.task.quiet:
            return args
        param = self.shelf.text_param(name)
        out = dict(args)
        if param and isinstance(out.get(param), str):
            out[param] = 'No listing in the store meets all the requirements, so no purchase is made.'
        return out

    # ---------------------------------------------------------------- the cart
    def track_cart(self, args: dict[str, Any], observation: Any) -> None:
        self.cart_changed_at = self.wire.step
        basket = self.basket_from(args)
        if basket is not None:
            lines = self.cart_lines(observation, self.shelf.roles.get('cart'))
            self.cart = lines if lines is not None else basket
            self.note_built()
            return
        ident = next((v for v in args.values() if isinstance(v, str) and (v in self.shelf.rows or _idlike(v))), None)
        units = next((_int(v) for v in args.values() if _int(v) is not None and not isinstance(v, str)), None)
        if ident:
            other = self.other_cart_op(args)
            if not other:
                template = {k: ('<ID>' if v == ident else '<QTY>' if (units is not None and _int(v) == units and not isinstance(v, str)) else v) for k, v in args.items()}
                if units is None or units > 0:
                    self.cart_template = (self.shelf.roles['cart'], template)
            if (units is not None and units <= 0) or (other and (units is None or units >= self.cart.get(ident, 0))):
                if other and not self.cart_remove:
                    self.cart_remove = {k: ('<ID>' if v == ident else '<QTY>' if (units is not None and _int(v) == units and not isinstance(v, str)) else v) for k, v in args.items()}
                self.cart.pop(ident, None)
            elif other:
                self.cart[ident] = units
            else:
                self.cart[ident] = units if units is not None else max(1, self.cart.get(ident, 0))
        lines = self.cart_lines(observation, self.shelf.roles.get('cart'), empty_ok=False)
        if lines is not None:
            self.cart = lines
        self.note_built()

    def basket_from(self, args: dict[str, Any]) -> dict[str, int] | None:
        """For a tool that takes the whole basket as a list of lines: learn its line template and return the lines sent."""
        for key, value in args.items():
            if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
                continue
            first = value[0] if value else None
            id_key = next((k for k, v in first.items() if isinstance(v, str) and (v in self.shelf.rows or _idlike(v))), None) if first else None
            if first and not id_key:
                continue
            if not first:
                known = self.cart_template[1] if self.cart_template else self.cart_add
                if not (known and _line_list(known) == key):
                    continue
                return {}
            unit_key = next((k for k, v in first.items() if k != id_key and _int(v) is not None and not isinstance(v, (str, bool))), None)
            line = {k: ('<ID>' if k == id_key else '<QTY>' if k == unit_key else v) for k, v in first.items()}
            self.cart_template = (self.shelf.roles['cart'], {k: ([line] if k == key else v) for k, v in args.items()})
            sent: dict[str, int] = {}
            for x in value:
                if isinstance(x.get(id_key), str):
                    units = _int(x.get(unit_key)) if unit_key else 1
                    if units is None or units > 0:
                        sent[x[id_key].strip()] = units if units is not None else 1
            return sent
        return None

    def full_removal(self, args: dict[str, Any]) -> dict[str, Any]:
        """A removal or change call that gives no unit count takes the line's whole count, so the line goes at once
        (a store can read a missing count as one unit and leave the rest of the line in the cart)."""
        template = self.cart_template[1] if self.cart_template else self.cart_add
        if not template or _line_list(template) or not self.other_cart_op(args):
            return args
        qty_key = next((k for k, v in template.items() if v == '<QTY>'), None)
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart), None)
        if not qty_key or not ident or args.get(qty_key) not in (None, '') or self.cart.get(ident, 0) < 1:
            return args
        out = dict(args)
        out[qty_key] = self.cart[ident]
        shaped, faults = self.shelf.shape(self.shelf.roles['cart'], out)
        if faults:
            return args
        self.trace.append({'kind': 'full_removal', 'line': ident, 'units': self.cart[ident]})
        return shaped

    def other_cart_op(self, args: dict[str, Any]) -> bool:
        """True when a cart call differs from the adding call on a fixed field (a removal or a change, not an add)."""
        learned = self.cart_template[1] if self.cart_template else None
        base = learned or self.cart_add or {}
        props = (self.shelf.schemas.get(self.shelf.roles.get('cart') or '') or {}).get('properties') or {}
        for key, value in base.items():
            if isinstance(value, (list, dict)) or value in ('<ID>', '<QTY>') or key not in args or args[key] == value:
                continue
            if learned:
                return True
            allowed = (props.get(key) or {}).get('enum') if isinstance(props.get(key), dict) else None
            if isinstance(allowed, list) and value in allowed:
                return True
        return False

    def cart_lines(self, observation: Any, tool: str | None = None, empty_ok: bool = True) -> dict[str, int] | None:
        layout = self.shelf.layouts.get(tool or '') or self.shelf.layouts.get(self.shelf.roles.get('cart_view') or '') or {}
        unit_key = layout.get('units')
        records = self.shelf.records(observation, key=layout.get('id'))
        if not records:
            if not isinstance(observation, dict):
                return None
            holder = _holder(observation, layout['rows']) if layout.get('rows') else None
            if holder is not None:
                return {} if holder.get(layout['rows']) == [] else None
            if not empty_ok:
                return None
            # a wrapped result keeps the notes beside it: look for the empty list inside the result only
            inner = [v for v in observation.values() if isinstance(v, dict)]
            pool = [v for w in inner for v in w.values()] if inner else list(observation.values())
            return {} if any(isinstance(v, list) and not v for v in pool) else None
        lines = {}
        for ident, record in records:
            units = _int(record.get(unit_key)) if unit_key else None
            if units is None:
                ints = [_int(v) for k, v in record.items() if k != self.shelf.keys.get('price') and _int(v) is not None and not isinstance(v, (str, bool))]
                units = ints[0] if len(ints) == 1 else 1
            if units > 0:
                lines[ident] = units
        return lines

    def read_cart(self, observation: Any) -> None:
        lines = self.cart_lines(observation, self.shelf.roles.get('cart_view'))
        if lines is not None:
            self.cart = lines
            self.cart_seen_at = self.wire.step
            self.note_built()

    def note_built(self) -> None:
        owners = {self.asker.owner(self.shelf.category(i)) for i in self.cart} - {0}
        if self.cart and len(owners) >= len(self.task.items):
            self.built = list(self.cart)

    def cart_call(self, ident: str, units: int) -> bool:
        tool = self.shelf.roles.get('cart')
        if not tool:
            return False
        template = self.cart_template[1] if self.cart_template else self.cart_add
        if not template:
            return False
        key = _line_list(template)
        if key:
            # this tool sets the whole basket: send every line the cart should hold
            target = dict(self.cart)
            target[ident] = units
            args = {k: ([_fill(v[0], i, u) for i, u in target.items() if u > 0] if k == key else v) for k, v in template.items()}
        else:
            args = _fill(template, ident, units)
        args, faults = self.shelf.shape(tool, args)
        if faults:
            return False
        reply = self.wire.store(tool, args)
        if reply is None or (isinstance(reply.get('observation'), dict) and reply['observation'].get('error')):
            return False
        self.track_cart(args, reply.get('observation'))
        note = self.read_notes(reply.get('observation'))
        if note:
            self.pending_notes.append(note)
        return True

    def view_cart(self) -> None:
        tool = self.shelf.roles.get('cart_view')
        if not tool or self.cart_seen_at >= self.cart_changed_at or self.left() < 2:
            return
        args, faults = self.shelf.shape(tool, {})
        if faults:
            return
        reply = self.wire.store(tool, args)
        if reply is not None:
            seen = reply.get('observation')
            self.shelf.learn_layout(tool, seen)
            self.shelf.absorb(seen)
            self.read_cart(seen)
            self.read_notes(seen)

    # ---------------------------------------------------------------- checkout gate
    def lines_by_item(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {n: [] for n in range(0, len(self.task.items) + 1)}
        for ident in self.cart:
            out.setdefault(self.asker.owner(self.shelf.category(ident)), []).append(ident)
        return out

    def line_faults(self) -> list[str]:
        faults = []
        by = self.lines_by_item()
        for n, item in enumerate(self.task.items, 1):
            lines = by.get(n) or []
            if not lines:
                faults.append(f'item {n} ({item.name}) has no cart line')
            elif len(lines) > 1:
                faults.append(f"item {n} ({item.name}) is split over {len(lines)} lines ({', '.join(lines)}); one line per item")
            for ident in lines:
                if self.cart.get(ident) != item.qty:
                    faults.append(f'{ident} has {self.cart.get(ident)} units; item {n} needs {item.qty}')
        for ident in by.get(0) or []:
            if len(self.task.items) > 1:
                faults.append(f'{ident} is not of a requested kind')
        for ident in self.cart:
            if not self.shelf.buyable(ident):
                faults.append(f'{ident} is sold out')
        if self.task.budget is not None:
            prices = {i: self.shelf.price(i) for i in self.cart}
            if all(p is not None for p in prices.values()):
                if self.task.scope == 'each':
                    faults += [f'{i} costs {prices[i]:g}, over the budget of {self.task.budget:g} for each item' for i in self.cart if prices[i] > self.task.budget]
                else:
                    total = sum(prices[i] * u for i, u in self.cart.items())
                    if total > self.task.budget + 1e-9:
                        faults.append(f'the cart totals {total:g}, over the budget of {self.task.budget:g}')
        return faults

    def missing(self) -> list[int]:
        by = self.lines_by_item()
        return [n for n in range(1, len(self.task.items) + 1) if not by.get(n)]

    def patch_cart(self) -> list[str]:
        """Give every item a cart line before an order: the last full cart again when the cart was emptied, else the best read listing."""
        done = []
        if not self.cart and self.built and self.refill():
            done.append('the last full cart again: ' + ', '.join(f'{i} x{u}' for i, u in self.cart.items()))
        return done + self.fill_missing()

    def removal(self) -> dict[str, Any] | None:
        """The call that takes a line out: learned from the model, else the add call with its two-valued switch flipped."""
        if self.cart_remove:
            return self.cart_remove
        template = self.cart_template[1] if self.cart_template else self.cart_add
        if not template or _line_list(template):
            return None
        props = (self.shelf.schemas.get(self.shelf.roles.get('cart') or '') or {}).get('properties') or {}
        for key, value in template.items():
            spec = props.get(key)
            allowed = spec.get('enum') if isinstance(spec, dict) else None
            if isinstance(allowed, list) and len(allowed) == 2 and value in allowed:
                return {**template, key: allowed[1] if allowed[0] == value else allowed[0]}
        return None

    def fix_units(self) -> list[str]:
        """Lines whose unit count (as the store shows it) differs from the item's quantity: out, then back in right."""
        tool = self.shelf.roles.get('cart')
        add = self.cart_template[1] if self.cart_template else self.cart_add
        out = self.removal()
        layout = self.shelf.layouts.get(self.shelf.roles.get('cart_view') or '') or self.shelf.layouts.get(tool or '') or {}
        if not tool or not add or not out or _line_list(add) or not layout.get('units') or self.cart_seen_at < self.cart_changed_at:
            return []
        per_call = '<QTY>' in _j(add)
        done = []
        by = self.lines_by_item()
        for n, item in enumerate(self.task.items, 1):
            lines = by.get(n) or []
            if len(lines) != 1 or self.cart.get(lines[0]) in (None, item.qty):
                continue
            ident, have = lines[0], self.cart[lines[0]]
            need_calls = 1 + (1 if per_call else item.qty)
            if self.left() < need_calls + self.tail():
                break
            args, faults = self.shelf.shape(tool, _fill(out, ident, have))
            if faults:
                break
            reply = self.wire.store(tool, args)
            if reply is None or (isinstance(reply.get('observation'), dict) and reply['observation'].get('error')):
                break
            self.track_cart(args, reply.get('observation'))
            self.cart.pop(ident, None)
            ok = self.cart_call(ident, item.qty) if per_call else all(self.cart_call(ident, 1) for _ in range(item.qty))
            if not per_call and ok:
                self.cart[ident] = item.qty
            done.append(f'{ident}: {have} -> {item.qty} units' + ('' if ok else ' (the re-add failed)'))
        if done:
            self.trace.append({'kind': 'unit_fix', 'lines': done})
        return done

    def fill_missing(self) -> list[str]:
        by = self.lines_by_item()
        done = []
        for n, item in enumerate(self.task.items, 1):
            if by.get(n) or self.left() < 2:
                continue
            pick = self.best_for(n)
            if pick and self.cart_call(pick, item.qty):
                done.append(f'item {n}: {pick} x{item.qty}')
        if done:
            self.trace.append({'kind': 'fill', 'lines': done})
        return done

    def best_for(self, number: int) -> str | None:
        item = self.task.items[number - 1]
        room = None
        if self.task.budget is not None:
            spent = sum((self.shelf.price(i) or 0) * u for i, u in self.cart.items())
            room = self.task.budget if self.task.scope == 'each' else self.task.budget - spent
        wanted = [_stems(m.split(':', 1)[-1]) for m in item.in_force() if m not in item.pointing]
        ranked = []
        for ident in self.shelf.order:
            if self.asker.owner(self.shelf.category(ident)) != number or not self.shelf.buyable(ident) or ident in self.cart:
                continue
            price = self.shelf.price(ident)
            if room is not None and price is not None and price * (item.qty if self.task.scope != 'each' else 1) > room:
                continue
            said = _stems(_j(self.shelf.opened.get(ident) or self.shelf.rows.get(ident)))
            hits = sum(1 for w in wanted if w and w <= said)
            ranked.append((-hits, ident not in self.shelf.opened, price if price is not None else float('inf'), ident))
        return min(ranked)[3] if ranked else None

    def screen(self, number: int, pool: list[str], why: str, cap: int = 4) -> list[str]:
        """The light model names the seen-but-unopened rows of item NUMBER that could fit; the controller opens them."""
        tool = self.shelf.roles.get('open')
        param = self.shelf.single_param(tool) if tool else None
        room = min(cap, self.left() - self.tail() - 3)
        if not tool or not param or not pool or room < 1 or self.late():
            return []
        item = self.task.items[number - 1]
        prompt = (SCREEN + f'\n\nWHY: {why}\n\nITEM: {item.name}\nREQUIREMENTS: ' + ('; '.join(item.in_force()) or 'none beyond the kind of product')
                  + '\n\nROWS:\n' + '\n'.join(f'{i}: ' + self.shelf.card(i, 160, 400) for i in pool[:30]))
        got = self.wire.ask_json(prompt, [LIGHT], 600) or {}
        chosen = [str(i) for i in _list(got.get('open')) if str(i) in pool][:room]
        opened = []
        for ident in chosen:
            args, faults = self.shelf.shape(tool, {param: ident})
            if faults:
                break
            reply = self.wire.store(tool, args)
            if reply is None:
                break
            self.shelf.absorb(reply.get('observation'))
            note = self.read_notes(reply.get('observation'))
            if note:
                self.pending_notes.append(note)
            opened.append(ident)
        self.trace.append({'kind': 'screen', 'item': number, 'pool': len(pool), 'opened': opened})
        return opened

    def unopened(self, number: int, cheaper_than: float | None = None) -> list[str]:
        room = None
        if self.task.budget is not None:
            room = self.task.budget / max(1, self.task.items[number - 1].qty if self.task.scope != 'each' else 1)
        pool = []
        for ident in self.shelf.order:
            if ident in self.cart or ident in self.shelf.opened or not self.shelf.buyable(ident) or self.asker.owner(self.shelf.category(ident)) != number:
                continue
            price = self.shelf.price(ident)
            if price is not None and ((room is not None and price > room) or (cheaper_than is not None and price >= cheaper_than)):
                continue
            if cheaper_than is not None and price is None:
                continue
            pool.append(ident)
        if self.task.pick == 'cheapest':
            pool.sort(key=lambda i: self.shelf.price(i) if self.shelf.price(i) is not None else float('inf'))
        return pool

    def value_sweep(self) -> list[str]:
        """Once per item, before the final check: one search with the item's most specific written value alone (a
        word value no search has used yet), plus one more page. A search with every value at once narrows the hits."""
        tool = self.shelf.roles.get('search')
        plan = self.asker.search_plan()
        if not tool or not plan:
            return []
        base, field = plan
        used = ' '.join(_j(a).lower() for a in self.searched)
        found: list[str] = []
        for line in list(self.cart):
            n = self.asker.owner(self.shelf.category(line))
            if not n or n in self.value_swept or self.left() - self.tail() < 9 or self.late():
                continue
            self.value_swept.add(n)
            item = self.task.items[n - 1]
            values = [m.split(':', 1)[1].strip() for m in item.in_force() if ':' in m and m not in item.pointing]
            values = [v for v in values if v and not re.search(r'\d', v) and len(v.split()) <= 3 and v.lower() not in used]
            if not values:
                continue
            last = next((dict(a) for a in reversed(self.searched) if any(isinstance(v, str) and _stems(v) & _stems(item.name + ' ' + item.shelf) for v in a.values())), {})
            args = {**{k: v for k, v in last.items() if k != field and isinstance(v, str) and isinstance(((self.shelf.schemas.get(tool) or {}).get('properties') or {}).get(k, {}).get('enum'), list)}, **base, field: min(values, key=len)}
            call, faults = self.shelf.shape(tool, args)
            if faults:
                continue
            reply = self.wire.store(tool, call)
            if reply is None:
                continue
            seen = reply.get('observation')
            self.shelf.learn_layout(tool, seen)
            got = self.shelf.absorb(seen)
            if got and self.left() - self.tail() >= 8:
                got += self.asker.more_pages(tool, call, seen)
            found += got
            note = self.read_notes(seen)
            if note:
                self.pending_notes.append(note)
        if found:
            self.trace.append({'kind': 'value_sweep', 'rows': len(found)})
        return found

    def open_cheaper(self) -> list[str]:
        opened: list[str] = []
        for line in list(self.cart):
            n = self.asker.owner(self.shelf.category(line))
            price = self.shelf.price(line)
            if not n or (self.task.pick == 'cheapest' and price is None):
                continue
            if self.task.pick == 'cheapest':
                why = f'the cart holds {line} at {price:g}; the lowest price among fitting listings wins, so name rows that cost less and may fit'
                pool = self.unopened(n, cheaper_than=price)
            else:
                why = f'the cart holds {line}; the request ranks fitting listings by {self.task.measure or "a property"} ({self.task.pick} first), so name rows that may fit and rank above it'
                pool = self.unopened(n)
            opened += self.screen(n, pool, why)
        return opened

    def judge(self) -> str:
        key = _j(sorted(self.cart.items())) + f'#{self.task.version}'
        if key in self.checks or self.check_holds >= 2:
            return ''
        self.checks.add(key)
        heavy = self.heavy_checks == 0 and self.wire.spend + HEAVY_ROOM <= HARD_SPEND
        models = [HEAVY, LIGHT] if heavy else [LIGHT]
        if not heavy and self.wire.spend > HARD_SPEND:
            return ''
        if heavy:
            self.heavy_checks += 1
        records = '\n'.join(f'{i}: ' + ('SOLD OUT. ' if not self.shelf.buyable(i) else '') + self.shelf.card(i, 400, 3000) for i in self.cart)
        others = ''
        if self.task.pick in ('cheapest', 'most', 'least'):
            pool = [i for i in self.shelf.order if i not in self.cart and self.shelf.buyable(i)]
            if self.task.pick == 'cheapest':
                pool.sort(key=lambda i: (i not in self.shelf.opened, self.shelf.price(i) if self.shelf.price(i) is not None else float('inf')))
            others = '\n\nOTHER LISTINGS:\n' + '\n'.join(self.shelf.card(i, 160, 500) for i in pool[:10])
        pointed = [p for item in self.task.items for p in item.pointing]
        opened = '\n\nOPENED PAGES OF OTHER LISTINGS (for requirements that point at another listing):\n' + '\n'.join(self.shelf.card(i, 200, 800) for i in list(self.shelf.opened)[-8:] if i not in self.cart) if pointed else ''
        prompt = CHECK + '\n\nREQUEST:\n' + self.task.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.task.ledger() + '\n\nCART:\n' + ', '.join(f'{i} x{u}' for i, u in self.cart.items()) + "\n\nRECORDS OF THE CART'S LISTINGS:\n" + records + others + opened
        got = self.wire.ask_json(prompt, models, 3000)
        self.trace.append({'kind': 'check', 'cart': dict(self.cart), 'got': got})
        if not got or got.get('ok') is not False:
            return ''
        self.check_holds += 1
        problems = [str(p) for p in _list(got.get('problems'))][:5]
        better = [str(b) for b in _list(got.get('better')) if str(b) in self.shelf.rows and self.shelf.buyable(str(b))][:4]
        text = 'The pre-order review found problems: ' + '; '.join(problems or ['compare the lines with the requirements'])
        if better:
            text += '. Stronger candidates: ' + ', '.join(better)
        return text + '. Fix the cart (or re-read the records if the review is wrong), then check out.'

    def steps_due(self, stage: str) -> str:
        due = [p for p in self.task.process if p['when'] == stage and not p.get('done')]
        if not due or self.process_rounds >= 3 or self.asker.free() <= 0 or self.left() <= self.tail() + 1:
            return ''
        self.process_rounds += 1
        record = '\n'.join(f'{i}. {line}' for i, line in enumerate(self.asker.record, 1)) or '(nothing yet)'
        prompt = STEPS.replace('STAGE', {'order': 'place the order', 'choose': 'choose which listings to buy', 'end': 'finish the job'}.get(stage, 'end without a purchase')) + '\n\nREQUEST:\n' + self.task.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join('- ' + p['step'] for p in due) + '\n\nCART: ' + (', '.join(f'{self.shelf.card(i, 60, 200)} x{u}' for i, u in self.cart.items()) or 'empty') + f'\n\nRECORD:\n{record}\n\nMESSAGES LEFT: {self.asker.free()}'
        got = self.wire.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        if got.get('done') is True or not str(got.get('message') or '').strip():
            for p in due:
                p['done'] = 'yes'
            return ''
        text = str(got['message']).strip()
        said = self.asker.send(text)
        if said is None:
            return ''
        heard = self.asker.hear(1, text, said)
        told = self.asker.take(heard)
        if heard.get('approves') is not False and not told:
            for p in due:
                p['done'] = 'yes'
            return ''
        return 'The answer alters the order: ' + ('; '.join(told) if told else said[:200]) + '. Adjust the cart to it and send the checkout again.'

    def checkout_gate(self) -> str:
        if self.wire.seconds_left() < FILL_S and self.left() > self.tail() + 2:
            filled = self.fill_missing()
            if filled:
                return '[ctl] Little time is left and some items lacked a cart line; the controller added the strongest listing read for each: ' + '; '.join(filled) + '. Send the checkout now.'
        self.view_cart()
        changed = self.steps_due('order') or self.steps_due('end')
        if changed:
            return changed
        faults = self.line_faults()
        if faults and self.unit_fixes < 2 and any(' units; item ' in f for f in faults):
            self.unit_fixes += 1
            fixed = self.fix_units()
            if fixed:
                return '[ctl] The controller set the unit counts the request asks for: ' + '; '.join(fixed) + '. Send the checkout again.'
        if faults and self.line_holds < 2 and self.left() > self.tail():
            self.line_holds += 1
            self.trace.append({'kind': 'held_lines', 'faults': faults[:6]})
            return 'The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines with the cart tool, then check out.\nCart now: " + (', '.join(f'{i} x{u}' for i, u in self.cart.items()) or 'empty')
        if self.missing() and not self.patched and self.left() >= 2:
            self.patched = True
            filled = self.patch_cart()
            if filled:
                return '[ctl] Items without a cart line were added by the controller: ' + '; '.join(filled) + '. Send the checkout now, unless a line is wrong and needs fixing first.'
        if self.late() or self.left() <= self.tail() + 1:
            return ''
        self.value_sweep()
        read = self.open_cheaper()
        if self.pending_notes:
            notes, self.pending_notes = self.pending_notes, []
            self.cart_seen_at = -1
            return 'A store note came in while the controller read listings: ' + '; '.join(notes) + '. Apply this change to the cart before checking out.'
        verdict = self.judge()
        if verdict:
            seen = ('The controller read these cheaper listings for the check: ' + ', '.join(read) + '\n') if read else ''
            return seen + verdict
        return ''

    # ---------------------------------------------------------------- decline gate
    def audit(self) -> dict[str, Any]:
        lines = []
        for n, item in enumerate(self.task.items, 1):
            ids = [i for i in self.shelf.order if self.asker.owner(self.shelf.category(i)) == n][-24:]
            lines.append(f'item {n}: {item.name} - requirements: ' + '; '.join(item.in_force()))
            lines += ['  ' + self.shelf.card(i, 160, 700) for i in ids]
        prompt = AUDIT + '\n\nREQUIREMENTS IN FORCE:\n' + self.task.ledger() + '\n\nLISTINGS READ, BY ITEM:\n' + '\n'.join(lines)
        models = [LIGHT, HEAVY] if self.wire.spend + HEAVY_ROOM <= HARD_SPEND else [LIGHT]
        got = self.wire.ask_json(prompt, models, 3000) or {}
        self.trace.append({'kind': 'audit', 'got': _clip(got, 100)})
        return got

    def clashes(self, ident: str, number: int) -> bool:
        """True when the opened page gives a property a requirement names, with a value sharing no word with it."""
        record = self.shelf.opened.get(ident)
        if not record:
            return False
        flat: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, dict):
                flat.update({str(k): v for k, v in value.items()})
            else:
                flat[str(key)] = value
        item = self.task.items[number - 1]
        for must in item.in_force():
            if ':' not in must or must in item.pointing:
                continue
            prop, want = (_stems(x) for x in must.split(':', 1))
            if not prop or not want:
                continue
            for key, value in flat.items():
                if isinstance(value, (str, int, float)) and prop <= _stems(key.replace('_', ' ')) and not (want & _stems(value)):
                    return True
        return False

    def decline_gate(self) -> str:
        self.decline_tries += 1
        if self.left() <= 2:
            return ''
        if self.decline_tries == 1:
            for n in range(1, min(3, len(self.task.items)) + 1):
                self.screen(n, self.unopened(n), 'no listing read so far meets every requirement; name rows that may still meet them all', cap=3)
        if self.pending_notes:
            notes, self.pending_notes = self.pending_notes, []
            return 'A store note came in while the controller read listings: ' + '; '.join(notes) + '. Take it into account before closing.'
        got = self.audit()
        fits = {n: [str(i) for i in _list((got.get('fits') or {}).get(str(n))) if str(i) in self.shelf.rows and self.shelf.buyable(str(i))] for n in range(1, len(self.task.items) + 1)}
        near = {n: [str(i) for i in _list((got.get('near') or {}).get(str(n))) if str(i) in self.shelf.opened and self.shelf.buyable(str(i)) and not self.clashes(str(i), n)] for n in range(1, len(self.task.items) + 1)}
        self.trace.append({'kind': 'near_kept', 'near': near})
        leads = [ids[0] for ids in fits.values() if ids]
        if self.decline_tries == 1 and leads and self.left() > self.tail() + 2:
            self.leads = leads
            return 'Listings already read that look like fits: ' + ', '.join(leads) + '. Open any you have not read in full, cart the best one per item under the pick rule, and check out. Close without buying only if every one of them plainly breaks a requirement; name that requirement in your text and repeat the closing call.'
        if not leads and not self.pushed and self.left() > self.max_steps // 3 and self.wire.seconds_left() > WRAP_S + 30:
            self.pushed = True
            return f"{self.left()} steps remain: run searches you have not tried yet (each item's category page by page, one-word value queries, other names for the property) and open whatever might fit. Close without buying only once that is done."
        # a decline is right in very few tasks: order the nearest fit for every item when one exists
        picks = {n: (fits.get(n) or near.get(n) or [None])[0] for n in fits}
        if all(picks.values()) and self.left() >= len(picks) + 2:
            for n, ident in picks.items():
                if ident not in self.cart and not self.cart_call(ident, self.task.items[n - 1].qty):
                    return ''
            done = self.buy_now()
            if done:
                return '[done] The controller placed an order with the closest fitting listings rather than closing empty-handed: ' + ', '.join(str(v) for v in picks.values()) + '. ' + done
        return ''

    # ---------------------------------------------------------------- ending
    def buy_now(self) -> str:
        tool = self.shelf.roles.get('buy')
        if not tool:
            return ''
        args, faults = self.shelf.shape(tool, dict(self.buy_args))
        if faults:
            return ''
        reply = self.wire.store(tool, args)
        if reply is None:
            return ''
        text = self.after_call(tool, args, reply)
        return text if self.wire.finished else 'the store turned the checkout down: ' + text[:1500]

    def refill(self) -> bool:
        lines = {}
        for ident in self.built:
            if self.shelf.buyable(ident):
                n = self.asker.owner(self.shelf.category(ident))
                lines[ident] = self.task.items[n - 1].qty if n else 1
        if not lines and len(self.task.items) == 1:
            pick = self.best_for(1)
            if pick:
                lines[pick] = self.task.items[0].qty
        if not lines or len(lines) + 1 > self.left():
            return False
        for ident, units in lines.items():
            if not self.cart_call(ident, units):
                return False
        self.trace.append({'kind': 'refill', 'lines': lines})
        return True

    def finish(self, again: bool = False) -> str:
        if self.wire.finished or self.left() <= 0:
            return ''
        r = self.shelf.roles
        if r.get('buy') and self.left() >= 2 and (not self.cart or self.missing()):
            self.patch_cart()
        if r.get('buy') and self.cart:
            said = self.buy_now()
            self.trace.append({'kind': 'forced_buy', 'said': said[:200]})
            if self.wire.finished or self.left() <= 0:
                return ''
            if again:
                return said
        tool = r.get('refuse')
        if tool and not self.wire.finished:
            args, faults = self.shelf.shape(tool, self.quiet_args(tool, dict(self.refuse_args)))
            if not faults:
                reply = self.wire.store(tool, args)
                self.trace.append({'kind': 'forced_refuse'})
                if reply is not None and not self.wire.finished:
                    return 'the store did not accept closing without a purchase: ' + self.show(reply.get('observation'))
        return ''


def _crash(exc: BaseException) -> dict[str, Any]:
    return {'kind': 'crash', 'error': (exc.__class__.__name__ + ': ' + str(exc))[:400]}


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Episode(problem_data)
    except Exception as exc:
        return [_crash(exc)]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append(_crash(exc))
        try:
            if not episode.wire.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append(_crash(again))
        return episode.trace


__all__ = ['agent_main']
