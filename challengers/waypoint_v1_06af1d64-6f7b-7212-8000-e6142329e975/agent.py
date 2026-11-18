"""Waypoint: a stage-driven shopping agent for ORO Bench.

Code owns the route through an episode: bind the tools, scout every line of the order, interview the shopper, grade listings, fill
the cart, check out and repair a refused checkout. Models only answer narrow questions inside a stage, so no free-running tool loop
can burn the step budget, and the same route runs for every task. Designed and written for this miner.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

FAST = 'anthropic/claude-haiku-5.5'
JUDGE = 'anthropic/claude-sonnet-5.5'
BACKUP = 'openai/gpt-6-luna'

TIME_LIMIT = 270.0
MODEL_CUTOFF = TIME_LIMIT - 45
STOP_SCOUTING_AT = 170.0
SOFT_SPEND = 0.045
HARD_SPEND = 0.09
READ_PER_LINE = 10
READ_CHEAPEST = 6
READ_AFTER_ANSWER = 5
LIST_ONE_LINE = 4
LIST_SINGLE_SHOT = 6
LIST_ACROSS = 5
MESSAGE_ROOM = 60
CHEAP_WALKS = 3
CHEAP_FAIR_FIRST = True
ASSUMED_MESSAGES = 3
BLANK_VALUES = ('none', 'null', 'n a', 'unknown')
VERDICT_RANK = {'strong': 0, 'fair': 1, 'thin': 2, 'none': 2, 'conflict': 3}
GRADE_BATCH = 14
FINALISTS = 3
RESERVE_PROVEN = 2
RESERVE_ROUNDS = 2
MAX_ROUNDS = 8
ID_KEY = r'(^| )(id|pid|uid|sku|code|ref)$'
NOTE_ECHO = 3
REFERENCE_HOPS = 6
REFERENCE_READS = 8
REFERENCE_LOOKS = 5
SELLER_WORD = r'seller|vendor|brand|maker|merchant|shop|producer|company|supplier'
REFERS_TO_PAGE = r'\b(?:page|listing)\b'
TOKEN = r'[a-z]+\d[a-z\d]*|\d+(?:\.\d+)?|[a-z]+'
BUY_WORDS = r'order|buy|purchase|place|check ?out|confirm|complete|submit|accept'
DECLINE_WORDS = r'declin|pass|cancel|abort|abandon|refus|stand ?down|give ?up|reject|stop|\bno\b'
NOT_WORTH_ASKING = r'\b(?:warranty|condition|country|origin|shipping)\b|^options?$'
LIMIT_WORD = r'\b(?:at (?:most|least)|under|over|below|above|between|up to|minimum|maximum|or (?:more|less)|no (?:more|less|fewer)|not|anything but)\b'
SEGMENT = r'\s\|\s(?=[A-Za-z_][\w /-]{0,38}:\s)'

_model_client = ProxyClient(timeout=45, max_retries=1, retry_delay=1)
_store_client = ProxyClient(timeout=40, max_retries=2, retry_delay=1)


# ---- text and shape helpers ---------------------------------------------------------------------------------------------------
def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)


def words_of(text: Any) -> str:
    spaced = re.sub('([a-z0-9])([A-Z])', r'\1 \2', str(text))
    return re.sub(r'[^a-z0-9]+', ' ', spaced.lower()).strip()


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def value_tokens(text: Any) -> list[str]:
    """The words and whole numbers of a value: 1.1v gives 1.1 and v, so one decimal can never match another by its parts."""
    return re.findall(TOKEN, re.sub(r'(\d)([a-z])', r'\1 \2', str(text).lower()))


def word_set(text: Any) -> set[str]:
    """The words of a key or value with plural endings folded."""
    return {w[:-1] if len(w) > 3 and w.endswith('s') else w for w in words_of(text).split()}


def clip(value: Any, size: int) -> str:
    text = ' '.join(str(value).split())
    return text if len(text) <= size else text[:size - 1] + '~'


def first_object(text: Any) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for mark in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[mark.start():])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def flat_objects(text: Any) -> list[dict[str, Any]]:
    """Recover whole inner objects from a reply that was cut off before its outer JSON closed."""
    if not isinstance(text, str):
        return []
    out = []
    for piece in re.findall(r'\{[^{}]*\}', text):
        try:
            value = json.loads(piece)
        except ValueError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def typed(raw: Any) -> Any:
    """A value read out of a text record, given back its likely type."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    low = text.lower()
    if low in ('true', 'false'):
        return low == 'true'
    if low in ('none', 'null', ''):
        return None
    if re.fullmatch(r'-?\d+(\.\d+)?', text):
        return float(text) if '.' in text else int(text)
    return text


def pairs_from(text: Any) -> dict[str, str]:
    """Turn 'a=b; c=d' or 'A:b;C:d' into a mapping when most of the segments really are pairs."""
    if not isinstance(text, str) or len(text) < 8:
        return {}
    chunks = [c for c in re.split(r'\s*;\s*', text) if c.strip()]
    if len(chunks) < 3:
        return {}
    for mark in ('=', ':'):
        parts = [c.split(mark, 1) for c in chunks if mark in c]
        if len(parts) >= 0.7 * len(chunks):
            return {a.strip(): b.strip() for a, b in parts if a.strip() and len(a) <= 40}
    return {}


def record_from_line(text: Any) -> dict[str, Any] | None:
    """A listing written as text: JSON, or 'key: value | key: value' segments."""
    if not isinstance(text, str):
        return None
    body = text.strip()
    if body.startswith('{'):
        return first_object(body)
    parts = re.split(SEGMENT, body)
    if len(parts) < 3:
        return None
    record: dict[str, Any] = {}
    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            record[key.strip()] = typed(value)
    return record if len(record) >= 3 else None


def plain_value(value: Any) -> Any:
    """An attribute's value as text or a number, or None when it holds nothing a requirement could be checked against."""
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, list):
        return ', '.join(str(v) for v in value if isinstance(v, (str, int, float))) or None
    return None


def as_record(node: Any) -> dict[str, Any] | None:
    if isinstance(node, dict):
        return node if len(node) >= 2 else None
    return record_from_line(node)


def has_id_key(record: dict[str, Any]) -> bool:
    return any(re.search(ID_KEY, words_of(k)) for k in record)


def rows_in(payload: Any, depth: int = 0) -> list[dict[str, Any]]:
    """Every listing record in a reply, whether the reply is one record, a list, or a list under any key."""
    if isinstance(payload, str):
        many = [r for r in (record_from_line(x) for x in payload.splitlines() if x.strip()) if r]
        if sum(1 for r in many if has_id_key(r)) >= 2:
            return many
        one = record_from_line(payload)
        return [one] if one else []
    if isinstance(payload, list):
        return [r for r in (as_record(x) for x in payload) if r]
    if not isinstance(payload, dict):
        return []
    best: list[dict[str, Any]] = []
    mark = (0, 0)
    for value in payload.values():
        if isinstance(value, list) and value:
            found = [r for r in (as_record(x) for x in value) if r]
            score = (sum(1 for r in found if has_id_key(r)), len(found))
            if score > mark:
                best, mark = found, score
    if has_id_key(payload) and not mark[0]:
        return [payload]
    if best:
        return best
    if has_id_key(payload):
        return [payload]
    nested = [v for v in payload.values() if isinstance(v, dict) and has_id_key(v)]
    if nested:
        return nested
    for value in payload.values():
        one = record_from_line(value)
        if one:
            return [one]
    if depth < 2:
        for value in payload.values():
            if isinstance(value, dict):
                deeper = rows_in(value, depth + 1)
                if deeper:
                    return deeper
    return []


def texts_in(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [t for v in value for t in texts_in(v)]
    if isinstance(value, dict):
        return [t for v in value.values() for t in texts_in(v)]
    return []


def split_reply(observation: Any, depth: int = 0) -> tuple[Any, list[str]]:
    """Separate a reply's payload from any shopper notices riding on it; apart from the runtime's refusal key only shapes are read, because worlds rename every key."""
    if not isinstance(observation, dict):
        return observation, []
    if depth < 1 and observation.get('refusal'):
        rest = {k: v for k, v in observation.items() if k != 'refusal'}
        body, more = split_reply(rest['observation'] if set(rest) == {'observation'} else rest, depth + 1)
        return body, texts_in(observation['refusal']) + more
    if depth < 1:
        body, found = split_reply(observation, depth + 1)
        if found and isinstance(body, dict):
            inner, more = split_reply(body, depth + 1)
            return inner, found + more
        return body, found
    notices: list[str] = []
    rest: dict[str, Any] = {}
    if len(observation) > 3:
        return observation, []
    for key, value in observation.items():
        if isinstance(value, list) and value and all(isinstance(v, str) and len(v.split()) >= 3 and not record_from_line(v) for v in value):
            notices += value
        else:
            rest[key] = value
    if notices and len(rest) == 1:
        return next(iter(rest.values())), notices
    if not notices and len(observation) == 2:
        texts = [(k, v) for k, v in observation.items() if isinstance(v, str) and len(v.split()) >= 6 and not record_from_line(v)]
        others = [v for v in observation.values() if isinstance(v, (dict, list))]
        if len(texts) == 1 and len(others) == 1:
            return others[0], [texts[0][1]]
    return (rest if notices else observation), notices


def is_problem(payload: Any) -> bool:
    return isinstance(payload, dict) and any(re.search('error|problem|trouble|invalid|issue|fail', words_of(k)) and isinstance(v, str) and bool(v.strip()) for k, v in payload.items())


def spoken(reply: Any) -> str:
    if isinstance(reply, str):
        return reply
    if isinstance(reply, dict):
        strings = [v for v in reply.values() if isinstance(v, str)]
        if strings:
            return ' '.join(strings)
    return '' if reply is None else dump(reply)


class Episode:
    def __init__(self, problem: dict[str, Any]):
        env = problem['environment']
        view = env['policy_view']
        self.binding = dict(env['binding'])
        self.request = str(view.get('query') or '')
        if view.get('opening') is not None:
            self.request += '\n\nOPENING NOTE FROM THE STORE: ' + dump(view['opening'])
        self.tools = [t['function'] for t in view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict)]
        self.schema = {t['name']: t.get('parameters') or {} for t in self.tools}
        self.max_steps = int(view.get('max_steps') or 30)
        self.began = time.monotonic()
        self.step = 0
        self.spend = 0.0
        self.done = bool(view.get('finished'))
        self.trace: list[dict[str, Any]] = []
        self.plan: dict[str, Any] = {}
        self.lines: list[dict[str, Any]] = []
        self.budget: float | None = None
        self.messages_left = 0
        self.listings: dict[str, dict[str, Any]] = {}
        self.basket: dict[str, int] = {}
        self.notes_waiting: list[str] = []
        self.revision = 0
        self.scores: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
        self.second_looks: dict[tuple[int, int, str], bool] = {}
        self.fields: dict[str, str | None] = {'id': None, 'price': None, 'stock': None, 'section': None, 'title': None, 'seller': None}
        self.shelf_names: set[str] = set()
        self.orders_tried = 0
        self.confirmed = ''
        self.last_order_text = ''
        self.one_at_a_time = False
        self.budget_checked = False
        self.budget_scale = 1
        self.notes_retried = False
        self.heard_at: dict[str, int] = {}
        self.cart_failed = False
        self.retry_fill = False
        self.declines_tried = 0

    # ---- plumbing ---------------------------------------------------------------------------------------------------------------
    def clock(self) -> float:
        return time.monotonic() - self.began

    def left(self) -> int:
        return self.max_steps - self.step

    def note(self, kind: str, **info: Any) -> None:
        self.trace.append({'kind': kind, 'at': round(self.clock(), 1), **info})

    def model(self, prompt: str, tier: str = FAST, tokens: int = 1800, salvage: str = '') -> dict[str, Any] | None:
        if self.spend >= HARD_SPEND or self.clock() > MODEL_CUTOFF:
            return None
        if tier == JUDGE and self.spend >= SOFT_SPEND:
            tier = FAST
        for name in (tier, BACKUP):
            for attempt in range(3):
                if self.clock() > MODEL_CUTOFF:
                    return None
                body = {'model': name, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': tokens, 'temperature': 0}
                result = _model_client.post_verbose('/inference/chat/completions', body)
                data = result.data
                if isinstance(data, dict) and data.get('choices'):
                    usage = data.get('usage') or {}
                    if isinstance(usage.get('cost'), (int, float)):
                        self.spend += float(usage['cost'])
                    content = (data['choices'][0].get('message') or {}).get('content') or ''
                    found = first_object(content)
                    if salvage and (found is None or salvage in found):
                        pieces = [p for p in flat_objects(content) if salvage in p]
                        if pieces:
                            return {'recovered': pieces}
                    if found is not None:
                        return found
                    break
                fault = (result.error if isinstance(result.error, dict) else {}) if not isinstance(data, dict) else data.get('error')
                status = (fault.get('status') or fault.get('code')) if isinstance(fault, dict) else None
                if status == 402:
                    return None
                if status != 429 and not (isinstance(status, int) and status >= 500):
                    break
                time.sleep((1.5 ** attempt) + random.random())
        return None

    def fit_args(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Fit arguments to the tool's own schema: unknown names dropped, numbers cast, long text cut, required enums filled."""
        schema = self.schema.get(tool) or {}
        props = schema.get('properties') or {}
        out: dict[str, Any] = {}
        for key, value in args.items():
            spec = props.get(key)
            if spec is None or value is None:
                continue
            kind = spec.get('type')
            if kind == 'integer':
                try:
                    value = int(float(value))
                except (TypeError, ValueError):
                    continue
                value = min(value, spec['maximum']) if 'maximum' in spec else value
            elif kind == 'number':
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
            elif kind == 'string':
                value = str(value)
                if 'maxLength' in spec:
                    value = value[:spec['maxLength']]
            if 'enum' in spec and value not in spec['enum']:
                continue
            out[key] = value
        finishing = tool in (self.role('order').get('tool'), self.role('decline').get('tool'))
        for key in schema.get('required') or []:
            if key not in out and props.get(key, {}).get('enum') and not finishing:
                out[key] = props[key]['enum'][0]
        return out

    def call(self, tool: str | None, args: dict[str, Any]) -> Any:
        if self.done or not tool or tool not in self.schema or self.left() <= 0:
            return None
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': tool, 'args': self.fit_args(tool, args)}
        for attempt in range(4):
            result = _store_client.post_verbose('/environment/call', payload)
            data = result.data
            if isinstance(data, dict) and 'observation' in data:
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                self.done = self.done or bool(data.get('finished'))
                body, notes = split_reply(data.get('observation'))
                self.notes_waiting += [n for n in notes if self.step - self.heard_at.get(n, -99) > NOTE_ECHO]
                self.heard_at.update((n, self.step) for n in notes)
                self.note('call', tool=tool, step=self.step, notes=len(notes))
                return body
            if (result.error or {}).get('status') != 429:
                break
            time.sleep(1.2 ** attempt + random.random())
        return None

    # ---- stage: bind the request and the store's tools ----------------------------------------------------------------------------
    def bind_prompt(self, complaint: str = '') -> str:
        tools = '\n'.join(f"- {t['name']}: {t.get('description', '')} | {dump(t.get('parameters') or {})}" for t in self.tools)
        extra = f'\nA FIRST ATTEMPT WAS WRONG: {complaint}\n' if complaint else ''
        return ('BIND. Below are a store\'s tools and a message from a shopper. Produce ONE JSON object: {"items":[{"name":bare kind of product,"label":store category the request '
                'names or "","qty":integer,"needs":["property: value"],"terms":[at most 3 short search phrases a shopper would type, most specific first, using the words a listing title would use]}],"budget":the number exactly as written in the request\'s own unit, or null,"budget_each":true when the limit is per item,"pick":"cheapest"|"most"|"least"|"any",'
                '"pick_on":measure or "","messages":integer or null,"steps":integer or null,"approval":true when the request requires showing or confirming the basket with the shopper '
                'before ordering,"tools":{"search":{"tool":name,"text":arg for free words,"label":arg for a category or null,"page":arg for paging or null,"cap":arg for a price limit or null,'
                '"fixed":{args that select word search}},"browse":null or {"tool":name,"label":arg for an exact category,"page":arg or null,"cap":arg for a price limit or null,"fixed":{args that select category listing}},"facts":{"tool":name of the tool returning '
                'attributes, preferring one that accepts a list of ids,"id":arg},"offer":null or {"tool":name returning one listing with price and stock,"id":arg},"ask":{"tool":name or null,"text":arg},"cart":{"tool":name,"id":arg or '
                'null,"qty":arg or null,"op":arg or null,"add":value or null,"remove":value or null,"remove_tool":name or null,"lines":arg holding the whole cart as a list or null,'
                '"line_id":field inside a line or null,"line_qty":field inside a line or null},"view":{"tool":name or null},"order":{"tool":name,"args":{}},"decline":{"tool":name,"args":{},'
                '"text":arg or null}}}\nRules: one tool with a mode argument can serve both order and decline, with the mode in args; a type, style or value stated about the product is a need, never part of the '
                "name; each need names the property in the request's own word and copies its value, limit or exclusion exactly, putting one required value in each need, values offered as alternatives with or together in one need, and any exclusion in a need of its own; "
                'price, stock and quantity are not needs; the story around a request is not a need; any clause followed by an aside that the shopper will verify it personally states a firm requirement, even one given in passing, so keep it as a need; a store '
                'category the request names goes in label and is not a need; every name you give must exist in the tool list.' + extra +
                f'\n\nTOOLS:\n{tools}\n\nREQUEST:\n{self.request}')

    def plan_problems(self, plan: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        tools = plan.get('tools') if isinstance(plan.get('tools'), dict) else {}
        if not plan.get('items'):
            problems.append('no items')

        def check(role: str, required: bool, args: tuple[str, ...]) -> None:
            spec = tools.get(role) if isinstance(tools.get(role), dict) else {}
            name = spec.get('tool')
            if not name:
                if required:
                    problems.append(f'{role}: no tool named')
                return
            if name not in self.schema:
                problems.append(f'{role}: {name} is not a tool')
                return
            props = (self.schema[name].get('properties') or {})
            for arg in args:
                if spec.get(arg) and spec[arg] not in props:
                    problems.append(f'{role}.{arg}: {spec[arg]} is not an argument of {name}')
        check('search', True, ('text', 'label', 'page', 'cap'))
        check('browse', False, ('label', 'page', 'cap'))
        check('facts', True, ('id',))
        check('offer', False, ('id',))
        check('ask', False, ('text',))
        check('cart', True, ('id', 'qty', 'op', 'lines'))
        cart = tools.get('cart') if isinstance(tools.get('cart'), dict) else {}
        if not (cart.get('id') or cart.get('lines')):
            problems.append('cart: give either the id argument or the lines argument')
        check('view', False, ())
        check('order', True, ())
        check('decline', True, ('text',))
        return problems

    def bind(self) -> bool:
        plan = self.model(self.bind_prompt(), JUDGE, 2600) or self.model(self.bind_prompt(), FAST, 2600)
        if not plan:
            return False
        trouble = self.plan_problems(plan)
        if trouble:
            self.note('bind_retry', problems=trouble[:5])
            second = self.model(self.bind_prompt('; '.join(trouble)), JUDGE, 2600)
            if second and not self.plan_problems(second):
                plan = second
        if not isinstance(plan.get('tools'), dict) or not plan.get('items'):
            return False
        self.plan = plan
        for raw in plan['items']:
            if isinstance(raw, dict) and raw.get('name'):
                try:
                    qty = max(1, int(raw.get('qty') or 1))
                except (TypeError, ValueError):
                    qty = 1
                self.lines.append({'name': str(raw['name']), 'label': str(raw.get('label') or ''), 'qty': qty,
                                   'needs': [str(n) for n in raw.get('needs') or []], 'dropped': [], 'used': set(), 'keys': [], 'labels': [], 'pages': {}, 'terms': [str(t) for t in raw.get('terms') or []][:3]})
        if isinstance(plan.get('budget'), (int, float)):
            self.budget = float(plan['budget'])
            written = {int(n) for n in re.findall(r'\b(\d{2,7})\b', self.request)}
            if self.budget < 100 and round(self.budget * 100) in written:
                self.budget = round(self.budget * 100)
        said = re.search(r'(\d+)\s*(?:more\s*)?messages?', self.request, re.I)
        count = plan.get('messages') if isinstance(plan.get('messages'), int) else (int(said.group(1)) if said else ASSUMED_MESSAGES)
        self.messages_left = int(count or 0) if (plan['tools'].get('ask') or {}).get('tool') else 0
        if isinstance(plan.get('steps'), int) and 8 <= plan['steps'] < self.max_steps and re.search(rf'\b{plan["steps"]}\b', self.request):
            self.max_steps = plan['steps']
        self.note('bind', lines=[(l['name'], l['needs']) for l in self.lines], budget=self.budget, messages=self.messages_left, pick=plan.get('pick'))
        return bool(self.lines)

    def role(self, name: str) -> dict[str, Any]:
        found = (self.plan.get('tools') or {}).get(name)
        return found if isinstance(found, dict) else {}

    # ---- reading the catalogue ----------------------------------------------------------------------------------------------------
    def learn_fields(self, rows: list[dict[str, Any]]) -> None:
        names = {k for row in rows for k in row}

        def pick(pattern: str, test: Any, avoid: str = '', distinct: bool = False) -> str | None:
            best, top = None, (0, 0)
            for key in sorted(names, key=lambda k: (len(k), k)):
                low = words_of(key)
                if not re.search(pattern, low) or (avoid and re.search(avoid, low)):
                    continue
                good = [row.get(key) for row in rows if test(row.get(key))]
                mark = (len(good), len({str(v) for v in good}) if distinct else 0)
                if mark > top:
                    best, top = key, mark
            return best
        fields = self.fields
        if not fields['id']:
            token = lambda v: isinstance(v, str) and 0 < len(v) <= 64 and ' ' not in v
            named_id = lambda v: token(v) or (isinstance(v, int) and not isinstance(v, bool))
            fields['id'] = pick(ID_KEY, named_id, distinct=True)
            if not fields['id']:
                for key in sorted(names):
                    values = [row.get(key) for row in rows]
                    if len(values) >= 2 and all(token(v) for v in values) and len(set(values)) == len(values):
                        fields['id'] = key
                        break
        number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        if not fields['price']:
            fields['price'] = pick('price|cost|amount', number, 'total|shipping')
        if not fields['stock']:
            fields['stock'] = pick('stock|orderable|buy|available|on hand', lambda v: isinstance(v, (bool, int)))
        if not fields['section']:
            fields['section'] = pick(r'categor|section|department|shelf|label', lambda v: isinstance(v, str) and bool(v))
        if not fields['title']:
            fields['title'] = pick(r'(^| )(title|name)$', lambda v: isinstance(v, str) and len(v) > 3, 'seller|vendor|brand|maker|merchant|shop|producer|company|section|categor')
        if not fields['seller']:
            fields['seller'] = pick(SELLER_WORD, lambda v: isinstance(v, str) and bool(v))

    def check_budget_unit(self, rows: list[dict[str, Any]]) -> None:
        """A limit stated in dollars can meet prices kept in cents; once the first prices are seen, put the limit on the same scale."""
        if self.budget_checked or self.budget is None or not self.fields['price']:
            return
        prices = sorted(r[self.fields['price']] for r in rows if isinstance(r.get(self.fields['price']), (int, float)) and not isinstance(r.get(self.fields['price']), bool))
        if not prices:
            return
        self.budget_checked = True
        if re.search(r'\$|dollar', self.request, re.I) and not re.search(r'\bcents?\b|¢', self.request, re.I) and prices[len(prices) // 2] > self.budget * 20:
            self.budget *= 100
            self.budget_scale = 100
            self.note('budget_scaled', budget=self.budget)

    def absorb(self, payload: Any, read: bool = False) -> list[str]:
        rows = rows_in(payload)
        if not rows:
            return []
        self.learn_fields(rows)
        self.check_budget_unit(rows)
        out: list[str] = []
        for row in rows:
            ident = row.get(self.fields['id'] or '')
            if isinstance(ident, int) and not isinstance(ident, bool):
                ident = str(ident)
            if not isinstance(ident, str) or not ident:
                continue
            entry = self.listings.setdefault(ident, {'id': ident, 'row': {}, 'facts': {}, 'read': False, 'gone': False})
            for key, value in row.items():
                if isinstance(value, dict):
                    self.merge_facts(entry, {k: t for k, t in ((k, plain_value(v)) for k, v in value.items()) if t is not None})
                elif isinstance(value, str) and pairs_from(value):
                    self.merge_facts(entry, pairs_from(value))
                    entry['row'][key] = value
                elif not isinstance(value, list):
                    entry['row'][key] = value
                elif value and all(isinstance(v, dict) for v in value):
                    named = {str(n): t for n, t in ((next((x for k, x in v.items() if words_of(k) in ('name', 'key', 'label', 'attribute')), None),
                                                     plain_value(next((x for k, x in v.items() if words_of(k) == 'value'), None))) for v in value) if n and t is not None}
                    self.merge_facts(entry, named)
            if read:
                entry['read'] = True
                if not entry['facts']:
                    known = {self.fields[r] for r in self.fields}
                    self.merge_facts(entry, {k: v for k, v in row.items() if k not in known and isinstance(v, (str, int, float)) and not isinstance(v, bool) and 0 < len(str(v)) <= 60})
            out.append(ident)
        return out

    @staticmethod
    def merge_facts(entry: dict[str, Any], incoming: dict[str, Any]) -> None:
        """Store attributes under one spelling of each key: variants of a key are one property, so their values are joined instead of reading as conflicts."""
        facts = entry['facts']
        for key, value in incoming.items():
            name = words_of(key)
            if not name:
                continue
            text = str(value)
            known = facts.get(name)
            if known is None:
                facts[name] = value
            elif words_of(text) not in words_of(known):
                facts[name] = f'{known}, {text}'

    def field(self, ident: str, role: str) -> Any:
        return self.listings[ident]['row'].get(self.fields[role] or '')

    def price(self, ident: str) -> float | None:
        value = self.field(ident, 'price')
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        stock = self.field(ident, 'stock')
        return not self.listings[ident]['gone'] and stock is not False and stock != 0

    def card(self, ident: str, text_size: int = 220) -> dict[str, Any]:
        entry = self.listings[ident]
        known = {self.fields[r] for r in self.fields}
        extra = {k: clip(v, text_size) for k, v in entry['row'].items() if k not in known and isinstance(v, str) and v and not pairs_from(v)}
        attributes = {k: clip(v, 80) for k, v in entry['facts'].items()}
        seller = self.field(ident, 'seller')
        if isinstance(seller, str) and seller:
            attributes.setdefault(words_of(self.fields['seller']), clip(seller, 60))
        return {'id': ident, 'title': clip(self.field(ident, 'title') or '', 150), 'price': self.price(ident), 'section': self.field(ident, 'section'),
                'attributes': attributes, 'text': extra}

    def belongs(self, ident: str, line: dict[str, Any]) -> bool:
        section = words_of(self.field(ident, 'section') or '')
        if not section:
            return True
        chosen = line.get('shelves')
        if chosen and section in self.shelf_names:
            named = line.get('named_shelf')
            if named and named in self.shelf_names and not line.get('named_failed'):
                return section == named
            return section in chosen
        wanted = set(words_of(line['label'] or line['name']).split())
        return bool(wanted & set(section.split())) or any(len(w) > 3 and w[:4] in section for w in wanted)

    def buyable_for(self, line: dict[str, Any]) -> list[str]:
        return [i for i in self.listings if self.belongs(i, line) and self.buyable(i)]

    def text_arg(self, tool_role: dict[str, Any], text: str) -> str:
        spec = (self.schema.get(tool_role.get('tool') or '', {}).get('properties') or {}).get(tool_role.get('text') or '', {})
        limit = spec.get('maxLength') or 80
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(' ', 1)[0]

    def search(self, line: dict[str, Any], text: str, more: bool = False, capped: bool = True) -> list[str]:
        spec = self.role('search')
        if not spec.get('tool') or not text.strip():
            return []
        args = dict(spec.get('fixed') or {})
        args[spec.get('text') or 'query'] = self.text_arg(spec, text)
        if capped and self.budget is not None and spec.get('cap'):
            args[spec['cap']] = self.budget
        key = (spec['tool'], text)
        if more:
            token = line['pages'].get(key)
            if token is None or not spec.get('page'):
                return []
            args[spec['page']] = token
        payload = self.call(spec['tool'], args)
        found = self.absorb(payload)
        line.setdefault('searched', set()).add(text)
        for ident in found:
            self.listings[ident].setdefault('via', set()).update(words_of(text).split())
        if isinstance(payload, dict) and spec.get('page'):
            onward = next((v for k, v in payload.items() if re.search('next|onward|continue|cursor|token|follow', words_of(k)) and isinstance(v, (int, str)) and v), None)
            if onward is True:
                onward = args[spec['page']] + 1 if isinstance(args.get(spec['page']), int) else 2
            line['pages'][key] = onward
        return found

    def browse(self, line: dict[str, Any]) -> list[str]:
        spec = self.role('browse')
        if not spec.get('tool') or not spec.get('label') or not line['label']:
            return []
        args = dict(spec.get('fixed') or {})
        args[spec['label']] = line['label']
        return self.absorb(self.call(spec['tool'], args))

    def read_listings(self, idents: list[str]) -> None:
        """Open listings for their attributes: several per step when the world's read tool takes a list, otherwise one at a time."""
        spec = self.role('facts')
        todo = [i for i in idents if not self.listings[i]['read']]
        if not spec.get('tool') or not todo:
            return
        arg = spec.get('id') or 'id'
        arg_spec = (self.schema.get(spec['tool'], {}).get('properties') or {}).get(arg, {})
        if arg_spec.get('type') == 'array':
            size = int(arg_spec.get('maxItems') or 6)
            start = 0
            while start < len(todo) and self.left() > 3:
                batch = todo[start:start + size]
                payload = self.call(spec['tool'], {arg: batch})
                if (is_problem(payload) or not rows_in(payload)) and size > 2:
                    size = max(2, size // 2)
                    continue
                self.absorb(payload, read=True)
                start += len(batch)
            return
        for ident in todo:
            if self.left() <= 3:
                return
            self.absorb(self.call(spec['tool'], {arg: ident}), read=True)

    def choose_shelves(self) -> None:
        shelves: dict[str, list[str]] = {}
        for ident, entry in self.listings.items():
            name = str(self.field(ident, 'section') or '')
            if name:
                shelves.setdefault(name, []).append(clip(self.field(ident, 'title') or '', 70))
        if not shelves:
            return
        names = [{'item': n, 'name': l['name'], 'category asked for': l['label']} for n, l in enumerate(self.lines, 1)]
        catalog = [{'shelf': s, 'examples': rows[:2]} for s, rows in shelves.items()]
        prompt = ('SHELF. These are the store shelves that came back when searching for each item. For each item list every shelf that could hold it; leave out shelves selling something '
                  'else. "named" is the one shelf whose name the shopper\'s wording directly matches, or null. Return ONE JSON object: {"items":[{"item":number,"shelves":[exact shelf names],"named":'
                  'shelf or null}]}.\n\nITEMS: ' + dump(names) + '\nSHELVES: ' + dump(catalog))
        got = self.model(prompt, FAST, 700) or {}
        self.shelf_names = {words_of(s) for s in shelves}
        for entry in as_list(got.get('items')):
            if isinstance(entry, dict) and isinstance(entry.get('item'), int) and 1 <= entry['item'] <= len(self.lines):
                line = self.lines[entry['item'] - 1]
                line['shelves'] = {words_of(s) for s in entry.get('shelves') or []}
                line['named_shelf'] = words_of(entry.get('named') or '')

    # ---- stage: scout ------------------------------------------------------------------------------------------------------------
    def need_values(self, line: dict[str, Any]) -> list[str]:
        """The positive part of each live requirement, usable as search words: limits and exclusions stay out of the query."""
        values = []
        for need in line['needs']:
            if need in line['dropped'] or ':' not in need or re.search(REFERS_TO_PAGE, need.split(':', 1)[1], re.I):
                continue
            text = re.split(r',\s*(?:not|no)\b|\bnot\b|\banything but\b|\bbut\b', need.split(':', 1)[1])[0].strip(' ,.;')
            if text and not re.search(LIMIT_WORD, text, re.I):
                values.append(re.split(r'\s+or\s+', text)[0].strip())
        return values

    def value_words(self, line: dict[str, Any]) -> str:
        return ' '.join(self.need_values(line))

    def search_phrases(self, line: dict[str, Any]) -> list[str]:
        name = line['name']
        values = self.need_values(line)
        phrases = [' '.join(values + [name])] if values else []
        phrases += list(line['terms'])
        if values:
            phrases += [f'{v} {name}' for v in values[:2]]
            phrases.append(' '.join(values))
        for need in self.live_needs(line):
            phrases += [f'{alt.strip()} {name}' for alt in re.split(r'\s+or\s+', need.split(':', 1)[-1])[1:] if ':' in need and not re.search(REFERS_TO_PAGE, need, re.I)]
        phrases.append(name)
        seen: list[str] = []
        for phrase in phrases:
            if phrase.strip() and phrase.lower() not in [p.lower() for p in seen]:
                seen.append(phrase)
        return seen

    def fit_score(self, ident: str, line: dict[str, Any]) -> int:
        entry = self.listings[ident]
        text = words_of(dump(entry['row']) + dump(entry['facts']))
        via = entry.get('via') or set()
        return sum(1 for word in words_of(self.value_words(line)).split() if f' {word} ' in f' {text} ' or word in via)

    def reading_order(self, line: dict[str, Any], idents: list[str]) -> list[str]:
        def cost(ident: str) -> float:
            value = self.price(ident)
            return value if value is not None else 1e12
        if self.plan.get('pick') == 'cheapest':
            return sorted(idents, key=lambda i: (self.fit_score(i, line) == 0, cost(i)))
        return sorted(idents, key=lambda i: -self.fit_score(i, line))

    def room_to_finish(self) -> int:
        return min(8 + 3 * len(self.lines) + self.messages_left, self.max_steps // 2)

    def can_read(self, tight: bool = False) -> bool:
        room = min(5 + 2 * len(self.lines), self.max_steps // 3) if tight else self.room_to_finish()
        return self.left() > room and self.clock() < STOP_SCOUTING_AT

    def first_look(self, line: dict[str, Any]) -> None:
        for phrase in self.search_phrases(line)[:(4 if len(self.lines) == 1 else 3)]:
            if self.can_read():
                self.search(line, phrase)

    def read_for(self, line: dict[str, Any], count: int) -> None:
        unread = [i for i in self.buyable_for(line) if not self.listings[i]['read']]
        chosen: list[str] = []
        for ident in sorted(self.reading_order(line, unread), key=lambda i: not self.within_budget(i, line)):
            if len(chosen) >= count:
                break
            chosen.append(ident)
        if chosen and self.can_read(self.repairing()):
            spare = max(0, self.left() - (min(5 + 2 * len(self.lines), self.max_steps // 3) if self.repairing() else self.room_to_finish()))
            spec = self.role('facts')
            arg_spec = (self.schema.get(spec.get('tool') or '', {}).get('properties') or {}).get(spec.get('id') or '', {})
            allowed = count if arg_spec.get('type') == 'array' else spare
            self.read_listings(chosen[:allowed])

    def scout(self) -> None:
        for line in self.lines:
            if self.can_read():
                self.first_look(line)
        self.choose_shelves()
        for number, line in enumerate(self.lines):
            lines_after = len(self.lines) - number
            spare = max(0, self.left() - self.room_to_finish())
            first_reads = READ_CHEAPEST if self.plan.get('pick') == 'cheapest' and self.role('browse').get('cap') and self.role('browse').get('label') else READ_PER_LINE
            self.read_for(line, min(first_reads, max(2, spare // lines_after)))

    # ---- stage: interview --------------------------------------------------------------------------------------------------------
    def key_table(self, line: dict[str, Any]) -> list[str]:
        counts: dict[str, int] = {}
        values: dict[str, set[str]] = {}
        for ident in self.buyable_for(line):
            for key, value in self.listings[ident]['facts'].items():
                name, said = words_of(key), words_of(value)
                if name and said and said not in BLANK_VALUES and not re.search(NOT_WORTH_ASKING, name):
                    counts[name] = counts.get(name, 0) + 1
                    values.setdefault(name, set()).add(said)
        settled = {words_of(n.split(':')[0]) for n in self.live_needs(line)}
        keys = [k for k in counts if k not in settled]
        # A shopper's firm preferences sit on properties most listings of the kind state; how varied a property is only separates ties.
        keys.sort(key=lambda k: (-counts[k], -min(len(values[k]), 6)))
        return keys[:30]

    def rank_properties(self) -> None:
        for line in self.lines:
            line['keys'] = self.key_table(line)
        tables = [{'item': n, 'name': l['name'], 'settled': l['needs'], 'attribute keys': l['keys']} for n, l in enumerate(self.lines, 1)]
        prompt = ('LABELS. Below are a shopper\'s request and, for each item, the attribute keys its listings carry. The shopper holds firm preferences that were not written out, and the story '
                  'around the request often gives them away: who the item is for, how it will look in a photo or next to other things, what they '
                  'inspect on a label or tag, how it will be used, worn or carried, or what it must fit or go with. For each item look only '
                  'at its attribute keys. In "hinted" list the keys the story points to, most direct first, and leave it empty when nothing points anywhere. In "labels" order all remaining keys '
                  'from most to least likely to matter to this shopper, favouring properties that tell products of that kind apart; leave out anything already settled and internal field names. '
                  'Return ONE JSON object: {"items":[{"item":number,"hinted":[keys],"labels":[keys]}]}.\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + dump(tables))
        got = self.model(prompt, JUDGE, 2600) or self.model(prompt, FAST, 2600) or {}
        ranked: dict[int, list[str]] = {}
        hinted: dict[int, list[str]] = {}
        for entry in as_list(got.get('items')):
            if isinstance(entry, dict) and isinstance(entry.get('item'), int):
                ranked[entry['item']] = [words_of(k) for k in as_list(entry.get('labels')) if k]
                hinted[entry['item']] = [words_of(k) for k in as_list(entry.get('hinted')) if k]
        for number, line in enumerate(self.lines, 1):
            opinion = {k: i for i, k in enumerate(k for k in ranked.get(number, []) if k in line['keys'])}
            tail = len(line['keys'])
            lead = [k for k in hinted.get(number, []) if k in line['keys']]
            rest = sorted([k for k in line['keys'] if k not in lead], key=lambda k: line['keys'].index(k) + opinion.get(k, tail))
            line['labels'] = lead + rest
            line['rounds'] = []
            line['dead'] = set()

    def settled_keys(self, line: dict[str, Any]) -> set[str]:
        return {words_of(n.split(':')[0]) for n in self.live_needs(line)} | set(line.get('untraced') or ())

    def message_keys(self, line: dict[str, Any], size: int) -> list[str]:
        """Choose the properties for a line's next message. A shopper answers for the first tracked property a message names, so the previous reply says what to do next."""
        rounds = line.setdefault('rounds', [])
        settled = self.settled_keys(line)
        dead = line.setdefault('dead', set()) | settled
        live = [k for k in line['labels'] if k not in dead and not any(self.same_property(line, k, s) for s in settled)]
        asked = {k for r in rounds for k in r['keys']}
        fresh = [k for k in live if k not in asked]
        if not rounds:
            return fresh[:size]
        last = rounds[-1]
        if last['kind'] == 'confused':
            order = fresh + [k for k in live if k not in fresh]
        elif last['kind'] == 'need':
            place = last.get('hit')
            after = [k for k in last['keys'][(place if place is not None else -1) + 1:] if k in live]
            order = after + [k for k in fresh if k not in after]
        else:
            flat = [r for r in rounds if r['kind'] == 'indifferent']
            if len(flat) == 1 and len(last['keys']) > 1:
                order = [k for k in reversed(last['keys']) if k in live]
                order += [k for k in fresh if k not in order]
            else:
                order = fresh + [k for k in flat[0]['keys'] if k in live and k not in fresh]
        return order[:size]

    def same_property(self, line: dict[str, Any], key: str, other: str) -> bool:
        """Whether two attribute keys are spellings of one property: one's words contain the other's, and listings carrying both give them the same values."""
        first, second = word_set(key), word_set(other)
        if not first or not second or not (first <= second or second <= first):
            return False
        same = total = 0
        for ident in self.buyable_for(line):
            facts = self.listings[ident]['facts']
            if key in facts and other in facts:
                total += 1
                one, two = words_of(facts[key]), words_of(facts[other])
                same += one in two or two in one
        return total == 0 or same * 2 >= total

    def compose(self, group: list[int], size: int) -> tuple[str, list[tuple[int, list[str]]]]:
        parts: list[tuple[int, list[str]]] = []
        for number in group:
            keys = self.message_keys(self.lines[number], size)
            if keys:
                parts.append((number, keys))
        if not parts:
            return '', []
        said = []
        for index, (number, keys) in enumerate(parts):
            names = keys[0] if len(keys) == 1 else f"{', '.join(keys[:-1])} or {keys[-1]}"
            said.append(f"for the {names} of your {self.lines[number]['name']}")
        return 'What do you want ' + ', and '.join(said) + '?', parts

    def text_room(self) -> int:
        spec = self.role('ask')
        limit = ((self.schema.get(spec.get('tool') or '', {}).get('properties') or {}).get(spec.get('text') or '', {})).get('maxLength') or 900
        return int(limit)

    def per_line(self, group: list[int]) -> int:
        """How many properties fit per line within the message length the world allows."""
        words = [sum(len(k) + 2 for k in self.lines[n]['labels'][:30]) / max(1, min(30, len(self.lines[n]['labels']))) for n in group]
        average = max(10.0, sum(words) / max(1, len(words)))
        room = max(1, self.text_room() - MESSAGE_ROOM * len(group))
        fit = int(room / (average * len(group)))
        limit = LIST_ACROSS
        if len(group) == 1:
            limit = LIST_ONE_LINE if self.messages_left > len(self.lines) else LIST_SINGLE_SHOT
        return max(2, min(limit, fit))

    def interview(self) -> None:
        spec = self.role('ask')
        if self.messages_left <= 0 or not spec.get('tool'):
            return
        self.rank_properties()
        self.note('labels', order=[k for l in self.lines for k in l['labels']][:40])
        keep = 1 if self.plan.get('approval') else 0
        groups: list[list[int]] = []
        refused = 0
        count = len(self.lines)
        if count > self.messages_left - keep:
            share = max(1, -(-count // max(1, self.messages_left - keep)))
            groups = [list(range(start, min(start + share, count))) for start in range(0, count, share)]
        while self.messages_left > keep and self.left() > 6 and not self.done and self.clock() < MODEL_CUTOFF - 15:
            if groups:
                group = groups.pop(0)
            else:
                group = sorted(range(count), key=lambda n: (self.lines[n].get('times', 0), n))[:1]
            size = 1 if self.one_at_a_time else self.per_line(group)
            text, parts = self.compose(group, size)
            if not text:
                others = [n for n in range(count) if n not in group]
                text, parts = self.compose(others, 1 if self.one_at_a_time else self.per_line(others)) if others else ('', [])
                if not text:
                    return
            reply = self.call(spec['tool'], {spec.get('text') or 'text': text})
            self.messages_left -= 1
            if reply is None or is_problem(reply):
                refused += 1
                if refused >= 2:
                    return
                continue
            answer = spoken(reply)
            before = {n: len(l['needs']) for n, l in enumerate(self.lines)}
            kind = self.hear(text, answer)
            self.note('asked', q=text, a=answer[:220], reply_kind=kind)
            if kind == 'too_many':
                self.one_at_a_time = True
                continue
            for number, keys in parts:
                line = self.lines[number]
                line['times'] = line.get('times', 0) + 1
                new = line['needs'][before[number]:] + list(line.pop('untraced_now', None) or ())
                hit = None
                if new:
                    named = words_of(new[0].split(':')[0])
                    hit = next((p for p, k in enumerate(keys) if k == named or named in k or k in named), None)
                    line['found'] = True
                outcome = 'need' if new else ('confused' if kind == 'confused' else 'indifferent')
                if outcome == 'confused':
                    line['dead'].update(keys)
                elif outcome == 'need' and hit is not None:
                    line['dead'].update(keys[:hit + 1])
                line['rounds'].append({'keys': list(keys), 'kind': outcome, 'hit': hit})
                line['used'].update(keys)
            self.take_notes(light=True)

    def hear(self, question: str, answer: str) -> str:
        if not answer:
            return 'empty'
        names = [f"{n}: {l['name']}" for n, l in enumerate(self.lines, 1)]
        prompt = ('HEAR. A shopping agent asked a shopper a question. Read the answer and return ONE JSON object: {"needs":[{"item":number,"property":the full name of the property word for word as the answer gives it without the item\'s name,"value":value or limit,'
                  '"negated":false,"relation":"same","points_to":null or {"section":shelf or kind of product named,"seller":seller named,"attributes":{"property":"value"}}}],"budget":new limit as a number or null,'
                  '"drop":[{"item":number,"property":name}],"kind":"need" when a need is stated, "too_many" when the answer asks for one question at a time, "confused" when the answer did not understand the question or asks for a '
                  'rewording, otherwise "indifferent"}. A need is a firm preference the answer states. Firm wording with no explicit limit states exactly that value; only words such as at least or at most make a limit; a '
                  "stated exclusion sets negated true. When the value is given by reference to another listing's page, possibly through whatever that page defers to, fill points_to with that listing's shelf or kind, its seller "
                  'and every detail given to identify it as attributes, and leave value empty. "relation" says how the wanted value relates to what that page gives: "same", "not" when anything other than it is wanted (also set '
                  'negated true), "pairs" when what is wanted is what is designed to pair or go with it, "at least" or "at most" when the wording sets a limit against it. A reply that is indifferent, confused or asks for a rewording states no need.\n\n'
                  f'ITEMS: {names}\nQUESTION: {question}\nANSWER: {answer}')
        got = self.model(prompt, FAST, 1000) or {}
        self.steady_relations(got, answer)
        self.apply_changes(got)
        kind = str(got.get('kind') or '')
        return kind if kind in ('need', 'confused', 'indifferent', 'too_many') else ('need' if got.get('needs') else 'indifferent')

    @staticmethod
    def steady_relations(got: dict[str, Any], answer: str) -> None:
        """Reading how a reference relates to its page can slip on the same wording from one call to the next, so clear words in the answer settle it."""
        low = answer.lower()
        for entry in as_list(got.get('needs')):
            if not isinstance(entry, dict) or not isinstance(entry.get('points_to'), dict):
                continue
            if re.search(r"\b(?:anything|something) (?:but|bar|apart from|other than|except)\b|\bother than\b", low):
                entry['relation'], entry['negated'] = 'not', True
            elif re.search(r"\bdesigned to pair\b|\bpairs? with\b|\bgoes? with\b|\bgo together\b", low):
                entry['relation'] = 'pairs'
            elif re.search(r"\bnot under\b|\bnothing under\b|\bno less than\b|\bat least\b", low):
                entry['relation'] = 'at least'
            elif re.search(r"\bnot over\b|\bnothing over\b|\bno more than\b|\bat most\b", low):
                entry['relation'] = 'at most'

    def line_at(self, raw: Any) -> dict[str, Any]:
        try:
            number = int(raw)
        except (TypeError, ValueError):
            number = 1
        return self.lines[max(1, min(number, len(self.lines))) - 1]

    # ---- references: a value given by pointing at another listing's page ---------------------------------------------------------------
    @staticmethod
    def agrees(wanted: Any, stated: Any) -> bool:
        """Whether some text carries a value: every number in it as a whole number, and every word of a short value."""
        want = value_tokens(wanted)
        have = set(value_tokens(stated))
        if not want:
            return False
        numbers = [w for w in want if any(c.isdigit() for c in w)]
        if numbers:
            return all(n in have for n in numbers)
        hits = sum(1 for w in want if w in have)
        return hits == len(want) if len(want) <= 3 else hits * 4 >= len(want) * 3

    def reference_score(self, ident: str, details: dict[str, Any]) -> int:
        """How well a listing fits the details a reference gives: a key stating the value counts most, a key stating something else counts against it."""
        entry = self.listings[ident]
        facts = {words_of(k): str(v) for k, v in entry['facts'].items()}
        loose = ' '.join(str(v) for v in entry['row'].values() if isinstance(v, str))
        score = 0
        for key, value in details.items():
            name = words_of(key)
            named = [text for k, text in facts.items() if name and (k == name or name in k or k in name)]
            if named:
                score += 3 if any(self.agrees(value, text) for text in named) else -2
            elif any(self.agrees(value, text) for text in facts.values()):
                score += 2
            elif self.agrees(value, loose):
                score += 1
        return score

    def sold_by(self, ident: str, seller: str) -> bool:
        row = self.listings[ident]['row']
        target = f' {words_of(seller)} '
        structural = {self.fields[r] for r in ('id', 'title', 'section')}
        named = [v for k, v in row.items() if isinstance(v, str) and len(v) <= 60 and k not in structural]
        return any(target in f' {words_of(v)} ' for v in named) if named else target in f' {words_of(dump(row))} '

    def find_reference(self, pointer: dict[str, Any], line: dict[str, Any], visited: set[str], memo: dict[tuple[str, int], list[str]]) -> str | None:
        """Find the listing a reference describes: search on its seller and details, then read candidates until one fits the details given."""
        details = dict(pointer.get('attributes')) if isinstance(pointer.get('attributes'), dict) else {}
        held_in = word_set(self.fields['seller'] or '')
        seller = str(pointer.get('seller') or next((details.pop(k) for k in list(details) if held_in and word_set(k) & held_in), '') or '').strip()
        section = str(pointer.get('section') or '').strip()
        values = ' '.join(str(v) for v in details.values())
        shelf = f'{seller} {section}'.strip()
        route: list[tuple[str, bool]] = []
        for text, more in ((f'{seller} {values}'.strip(), False), (shelf, False), (shelf, True), (shelf, True), (seller, False), (f'{section} {values}'.strip(), False), (section, False)):
            if text and (more or (text, False) not in route):
                route.append((text, more))
        enough = 2 * len(details) + 1 if details else 1
        best, top = None, None
        reads = looks = 0
        opened: list[str] = []
        pages_seen: dict[str, int] = {}
        for phrase, more in route:
            if looks >= REFERENCE_LOOKS or reads >= REFERENCE_READS or not self.can_read(True):
                break
            nth = pages_seen.get(phrase, 0)
            pages_seen[phrase] = nth + 1
            if (phrase, nth) in memo:
                found = memo[(phrase, nth)]
            else:
                found = self.search(line, phrase, more, capped=False)
                memo[(phrase, nth)] = found
                looks += 0 if more and not found else 1
            pool = [i for i in found if i not in visited and i not in opened and (not seller or self.sold_by(i, seller))]
            pool.sort(key=lambda i: -self.reference_score(i, details))
            for start in range(0, len(pool), 3):
                if reads >= REFERENCE_READS or not self.can_read(True):
                    break
                batch = pool[start:start + 3]
                self.read_listings(batch)
                reads += len(batch)
                opened += batch
                for ident in batch:
                    score = self.reference_score(ident, details)
                    if top is None or score > top:
                        best, top = ident, score
                if top is not None and top >= enough:
                    return best
        picked = self.choose_reference(pointer, opened)
        if picked is not False:
            return picked
        return best if best is not None and (not details or (top or 0) > 0) else None

    def choose_reference(self, pointer: dict[str, Any], opened: list[str]) -> Any:
        """No listing clearly matched the details, which often means the same fact written differently: a model picks among the opened listings. Gives the id, None when it says none fits, or False when no answer came."""
        if not opened:
            return False
        prompt = ('WHICH. A description of one store listing is given below, with the listings it could be. The description and a listing can write the same fact differently: other units, other number formats, '
                  'other word order. Pick the listing the description fits, and answer null when none of them fits. A remark on a listing that a version with those details exists elsewhere means that listing is not it. Return ONE JSON object: {"id":the id, or null}.'
                  '\n\nDESCRIPTION: ' + dump(pointer) + '\nLISTINGS: ' + dump([self.card(i, 300) for i in opened]))
        got = self.model(prompt, FAST, 800)
        if not isinstance(got, dict) or 'id' not in got:
            return False
        answer = str(got.get('id') or '')
        return answer if answer in opened else None

    def read_reference(self, ident: str, prop: str) -> dict[str, Any] | None:
        """Ask what a page says for a property: another listing it hands the property to, the value it states, and any rule pairing values of that property."""
        prompt = ('TRACE. A shopper wants what the product page below says for the property "' + prop + '". Read its attributes and remarks. "next": when a remark says this property must be taken from another '
                  'listing and not from this page, describe that listing as {"section":its shelf or kind of product,"seller":the seller or brand name the remark gives for it, also when it comes right before the kind of product,"attributes":{"property":"value" for each identifying detail given}}, otherwise null. '
                  'A remark about a different property does not count, and neither does a remark that some version exists elsewhere. "value": what this page itself states for the property when "next" is null, otherwise null. '
                  '"pairs": every pairing the page gives between values of this property, such as X goes with Y or a list of X with Y, written as [["X","Y"]], otherwise []. Return ONE JSON object: '
                  '{"next":null or {...},"value":text or null,"pairs":[]}.\n\nPAGE: ' + dump(self.card(ident, 700)))
        return self.model(prompt, FAST, 1000)

    def stated_on_page(self, ident: str, prop: str) -> Any:
        wanted = word_set(prop)
        return next((v for k, v in self.listings[ident]['facts'].items() if wanted and wanted <= word_set(k) and v), None)

    def follow_reference(self, prop: str, pointer: dict[str, Any], line: dict[str, Any]) -> tuple[str | None, list[list[str]]]:
        """Walk from the page a shopper points at to the page that really states the property; a page may hand the property on, and the next page may hand it on again."""
        visited: set[str] = set()
        memo: dict[tuple[str, int], list[str]] = {}
        current = pointer
        pairs: list[list[str]] = []
        again = False
        for _ in range(REFERENCE_HOPS):
            ident = self.find_reference(current, line, visited, memo)
            if ident is None:
                break
            visited.add(ident)
            reading = self.read_reference(ident, prop)
            if not reading:
                break
            pairs += [[str(p[0]), str(p[1])] for p in as_list(reading.get('pairs')) if isinstance(p, list) and len(p) == 2]
            onward = reading.get('next')
            if isinstance(onward, dict) and (onward.get('seller') or onward.get('section')):
                self.note('reference_hop', prop=prop, at=ident, onward=onward)
                current, again = onward, False
                memo = {}
                continue
            value = reading.get('value') or self.stated_on_page(ident, prop)
            if value:
                self.note('reference_end', prop=prop, at=ident, value=value, pairs=pairs[:6])
                return str(value), pairs
            if again:
                break
            again = True
        self.note('reference_lost', prop=prop, pages=len(visited))
        return None, pairs

    def settle_reference(self, prop: str, entry: dict[str, Any], line: dict[str, Any]) -> str | None:
        """The value a reference stands for, or None when the trail could not be followed to a page that states it."""
        value, pairs = self.follow_reference(prop, entry['points_to'], line)
        if value is None or not pairs or not str(entry.get('relation') or '').lower().startswith('pair'):
            return value
        wanted = word_set(value)
        for test in (lambda a: word_set(a) == wanted, lambda a: bool(word_set(a)) and (word_set(a) <= wanted or wanted <= word_set(a)), lambda a: self.agrees(a, value) or self.agrees(value, a)):
            partner = next((b for a, b in pairs if test(a)), None)
            if partner:
                return partner
        return pairs[0][1] if len(pairs) == 1 else None

    def settle_stated_references(self) -> None:
        """A requirement in the request can also give its value by pointing at a page; read it the way an answer is read."""
        for number, line in enumerate(self.lines, 1):
            for need in list(self.live_needs(line)):
                if ':' in need and re.search(REFERS_TO_PAGE, need.split(':', 1)[1], re.I):
                    self.hear(f'What do you want for item {number}?', need)
                    if need not in line['dropped']:
                        line['dropped'].append(need)
        for line in self.lines:
            line.pop('untraced_now', None)

    def apply_changes(self, got: dict[str, Any]) -> None:
        changed = False
        for entry in as_list(got.get('needs')):
            if not isinstance(entry, dict):
                continue
            line = self.line_at(entry.get('item'))
            prop = str(entry.get('property') or '').strip()
            value = str(entry.get('value') or '').strip()
            if entry.get('need') and not prop:
                text = str(entry['need'])
            else:
                pointer = entry.get('points_to')
                if isinstance(pointer, dict) and prop and (pointer.get('seller') or pointer.get('section')):
                    traced = self.settle_reference(prop, entry, line)
                    if traced:
                        value = traced
                    elif not value or re.search(REFERS_TO_PAGE, value, re.I):
                        value = ''
                        line.setdefault('untraced', set()).add(words_of(prop))
                        line.setdefault('untraced_now', []).append(prop)
                if not (prop and value):
                    continue
                relation = str(entry.get('relation') or '').lower()
                lead = 'anything but ' if entry.get('negated') or relation == 'not' else f'{relation} ' if relation in ('at least', 'at most') and not re.search(LIMIT_WORD, value, re.I) else ''
                text = f"{prop}: {lead}{value}"
            name = words_of(text.split(':')[0])
            same_value = any(old not in line['dropped'] and words_of(old.split(':', 1)[-1]) == words_of(text.split(':', 1)[-1]) and
                             (set(words_of(old.split(':')[0]).split()) & set(name.split())) for old in line['needs'])
            if same_value and text not in line['needs']:
                continue
            for old in line['needs']:
                if old not in line['dropped'] and words_of(old.split(':')[0]) == name and old != text:
                    line['dropped'].append(old)
            if text not in line['needs']:
                line['needs'].append(text)
                changed = True
            elif text in line['dropped']:
                line['dropped'].remove(text)
                changed = True
        for entry in as_list(got.get('drop')):
            if isinstance(entry, dict) and entry.get('property'):
                line = self.line_at(entry.get('item'))
                name = words_of(entry['property'])
                for need in line['needs']:
                    head = words_of(need.split(':')[0])
                    if need not in line['dropped'] and (head == name or name in head or head in name):
                        line['dropped'].append(need)
                        changed = True
        if isinstance(got.get('budget'), (int, float)) and not isinstance(got.get('budget'), bool) and got['budget'] > 0:
            limit = float(got['budget'])
            if self.budget_scale != 1 and limit * self.budget_scale <= (self.budget or 0) * 4 and (self.budget is None or limit * 20 < self.budget):
                limit *= self.budget_scale
            if limit != self.budget:
                self.budget = limit
                changed = True
        if changed:
            self.revision += 1

    def live_needs(self, line: dict[str, Any]) -> list[str]:
        return [n for n in line['needs'] if n not in line['dropped']]

    # ---- stage: grade -----------------------------------------------------------------------------------------------------------
    def grade_batch(self, number: int, idents: list[str], table: dict[str, dict[str, Any]], tier: str = FAST) -> None:
        line = self.lines[number]
        needs = self.live_needs(line) or [f"kind: is a {line['name']}"]
        records = [self.card(i) for i in idents]
        keyless = [self.keyless_need(line, need) for need in needs]
        measure = str(self.plan.get('pick_on') or '') if self.plan.get('pick') in ('most', 'least') else ''
        extra = f' Also give each listing "measure": the number it has for {measure}, or null.' if measure else ''
        prompt = ('GRADE. For every listing and every requirement give one verdict. Attribute keys differ between listings: a key can name a property in other words, with extra words, a '
                  'prefix or punctuation. Values may be lists and may carry extra words. "strong": a key named for the property states the value (word forms, spelling variants, words with the same meaning and unit '
                  'conversions count, limits include their edge, one value of a list counts), or a differently named key and the title both state it. "fair": the value is stated only by a '
                  'differently named key, only by the title, or by a related key while the key named exactly for the property says something else. "thin": only the free description states it. '
                  '"none": nothing states the property; a value such as other or n/a states nothing. "conflict": keys for the property state only other values, even similar ones (a word with the same meaning is the same value), or the '
                  'product is plainly another kind than the item. A requirement that excludes a value is strong when a key states a different value and conflict when a key states the excluded one.' + extra +
                  ' Return ONE JSON object: {"grades":[{"id":id,"checks":[one verdict per requirement, in order],"measure":number or null}]}.\n\nITEM: ' + line['name'] + '\nREQUIREMENTS: ' + dump(needs) + '\nLISTINGS: ' + dump(records))
        got = self.model(prompt, tier, 3500, salvage='checks') or {}
        rows = {str(e.get('id')): e for e in as_list(got.get('grades') or got.get('recovered')) if isinstance(e, dict)}
        for ident in idents:
            row = rows.get(ident)
            if not row:
                continue
            checks = [str(c).lower() for c in as_list(row.get('checks'))]
            if len(checks) != len(needs):
                checks = ['none'] * len(needs)
            checks = ['fair' if c == 'thin' and keyless[n] else c for n, c in enumerate(checks)]
            rank = max(VERDICT_RANK.get(c, 2) for c in checks)
            table[ident] = {'rank': rank, 'checks': checks, 'measure': row.get('measure')}

    def keyless_need(self, line: dict[str, Any], need: str) -> bool:
        """True when no listing of this kind carries an attribute key for the property, so free text is the only place it can be stated."""
        wanted = word_set(need.split(':')[0])
        return bool(wanted) and not any(wanted <= word_set(k) or word_set(k) <= wanted for i in self.buyable_for(line) for k in self.listings[i]['facts'])

    def graded(self, number: int) -> dict[str, dict[str, Any]]:
        line = self.lines[number]
        table = self.scores.setdefault((number, self.revision), {})
        affordable = [i for i in self.buyable_for(line) if self.within_budget(i, line)]
        pool = [i for i in affordable if self.listings[i]['read'] or self.listings[i]['facts'] or self.fit_score(i, line) > 0]
        if not pool:
            pool = affordable
        todo = [i for i in pool if i not in table]
        for start in range(0, min(len(todo), GRADE_BATCH * 3), GRADE_BATCH):
            self.grade_batch(number, todo[start:start + GRADE_BATCH], table)
        return {i: s for i, s in table.items() if i in pool}

    def ranked(self, number: int) -> list[str]:
        line = self.lines[number]
        table = self.graded(number)
        keep = [i for i, s in table.items() if s['rank'] < 3 and self.buyable(i) and self.within_budget(i, line) and self.second_looks.get((number, self.revision, i)) is not False]
        pick = self.plan.get('pick')

        def cost(ident: str) -> float:
            value = self.price(ident)
            return value if value is not None else 1e12
        if pick == 'cheapest':
            keep.sort(key=(lambda i: (table[i]['rank'] > 1, cost(i), table[i]['rank'])) if CHEAP_FAIR_FIRST else (lambda i: (table[i]['rank'], cost(i))))
        elif pick in ('most', 'least'):
            sign = -1 if pick == 'most' else 1
            keep.sort(key=lambda i: (table[i]['rank'] > 1, sign * float(table[i]['measure']) if isinstance(table[i]['measure'], (int, float)) else 1e12, cost(i)))
        else:
            keep.sort(key=lambda i: (table[i]['rank'], cost(i)))
        return keep

    def within_budget(self, ident: str, line: dict[str, Any]) -> bool:
        cost = self.price(ident)
        if self.budget is None or cost is None:
            return True
        return cost * (1 if self.plan.get('budget_each') else line['qty']) <= self.budget

    def second_look(self, number: int, options: list[str]) -> list[str]:
        """A stronger model re-checks the finalists on the fuller record; a listing it rejects is out for this revision."""
        fresh = [i for i in options[:FINALISTS] if (number, self.revision, i) not in self.second_looks]
        line = self.lines[number]
        if fresh and self.live_needs(line):
            records = [self.card(i, 500) for i in fresh]
            prompt = ('FINAL. Check these finalists for one item. A listing fails a requirement only when keys for that property state nothing but other values, or the product is plainly '
                      'another kind than the item; keys can name a property in other words, one value of a list counts, a similar but different value does not count while the same meaning in other words does, and missing information '
                      'alone is not a failure. Return ONE JSON object: {"verdicts":[{"id":id,"fits":true or false}]}.\n\nITEM: ' + line['name'] + '\nREQUIREMENTS: ' + dump(self.live_needs(line)) + '\nFINALISTS: ' + dump(records))
            got = self.model(prompt, JUDGE, 900) or {}
            said = {str(v.get('id')): v.get('fits') for v in as_list(got.get('verdicts')) if isinstance(v, dict)}
            for ident in fresh:
                self.second_looks[(number, self.revision, ident)] = said.get(ident) is not False
        return [i for i in options if self.second_looks.get((number, self.revision, i)) is not False]

    def repairing(self) -> bool:
        return self.orders_tried > 0

    def widen(self, number: int) -> bool:
        """One more look for a line short of fits: another page, then the kind alone, then the values alone, then the shelf."""
        line = self.lines[number]
        before = len(self.listings)
        joined = f"{self.value_words(line)} {line['name']}".strip()
        tool = self.role('search').get('tool')
        for text in (joined, self.value_words(line), line['name']):
            if not text:
                continue
            if not self.can_read(self.repairing()):
                break
            again = text in line.get('searched', ())
            if again and line['pages'].get((tool, text)) is None:
                continue
            if self.search(line, text, again):
                break
        if len(self.listings) == before and self.can_read(self.repairing()):
            self.browse(line)
        self.read_for(line, 5)
        return len(self.listings) > before or any(not self.listings[i]['read'] for i in self.buyable_for(line))

    def chase_answers(self) -> None:
        """After the interview, look again for lines whose requirements just changed."""
        for line in self.lines:
            if not line.get('found') or not self.can_read():
                continue
            values = self.need_values(line)
            fresh = [f'{v} {line["name"]}' for v in reversed(values)][:1] + [' '.join(values + [line['name']])]
            for phrase in dict.fromkeys(fresh):
                if self.can_read() and phrase not in line.get('searched', ()):
                    self.search(line, phrase)
            self.read_for(line, READ_AFTER_ANSWER)

    def cheaper_guard(self, number: int, pick: str) -> str:
        """For a cheapest pick, open unread in-budget listings priced under the pick so no cheaper fit goes unseen."""
        line = self.lines[number]
        for _ in range(3):
            floor = self.price(pick)
            under = [i for i in self.buyable_for(line) if not self.listings[i]['read'] and floor is not None and (self.price(i) if self.price(i) is not None else 1e12) < floor]
            if not under or not self.can_read(self.repairing()):
                break
            self.read_for(line, min(4, len(under)))
            options = self.ranked(number)
            if options:
                pick = options[0]
        return pick

    # ---- stage: basket and checkout ----------------------------------------------------------------------------------------------
    def set_cart(self, wanted: dict[str, int]) -> None:
        spec = self.role('cart')
        tool = spec.get('tool')
        if spec.get('lines'):
            line_id = spec.get('line_id') or 'id'
            line_qty = spec.get('line_qty') or 'qty'
            payload = self.call(tool, {spec['lines']: [{line_id: i, line_qty: q} for i, q in wanted.items()]})
            if payload is not None and not is_problem(payload):
                self.basket = dict(wanted)
            return
        for ident in [i for i in self.basket if i not in wanted]:
            self.change_line(ident, self.basket[ident], False)
        for ident, qty in wanted.items():
            if self.basket.get(ident) != qty:
                if ident in self.basket:
                    self.change_line(ident, self.basket[ident], False)
                self.change_line(ident, qty, True)

    def change_line(self, ident: str, qty: int, add: bool) -> None:
        spec = self.role('cart')
        tool = spec.get('tool') if add or not spec.get('remove_tool') else spec['remove_tool']
        name = spec.get('id') or 'id'
        if not add and spec.get('remove_tool'):
            schema = self.schema.get(spec['remove_tool']) or {}
            props = schema.get('properties') or {}
            if name not in props:
                name = next((k for k in list(schema.get('required') or []) + list(props) if props.get(k, {}).get('type') == 'string'), name)
        args: dict[str, Any] = {name: ident}
        if spec.get('qty'):
            args[spec['qty']] = qty
        if spec.get('op'):
            args[spec['op']] = spec.get('add') if add else spec.get('remove')
        if not add and spec.get('qty') and not spec.get('op') and not spec.get('remove_tool'):
            args[spec['qty']] = 0
        payload = self.call(tool, args)
        if payload is None or is_problem(payload):
            return
        if add:
            self.basket[ident] = qty
        else:
            self.basket.pop(ident, None)

    def sweep(self, number: int) -> None:
        """For a cheapest pick, walk further result pages so the price floor is measured on the whole shelf rather than the first page."""
        line = self.lines[number]
        for text in (f"{self.value_words(line)} {line['name']}".strip(), line['name']):
            for _ in range(2):
                if not self.can_read() or not self.search(line, text, True):
                    break

    def with_evidence(self, number: int, options: list[str]) -> list[str]:
        """A listing with nothing on it for a requirement almost never fits, so only listings with positive evidence on every requirement can be bought."""
        table = self.scores.get((number, self.revision), {})
        return [i for i in options if table.get(i, {}).get('rank', 3) <= 1]

    def needs_better(self, number: int, options: list[str]) -> bool:
        table = self.scores.get((number, self.revision), {})
        return not options or table.get(options[0], {}).get('rank', 3) > 0

    def short_of_reserve(self, number: int, options: list[str]) -> bool:
        """Before the first checkout: a first choice often sells out at the till, so hold a second fully proven listing while there is room to look for one."""
        if self.repairing() or not options:
            return False
        table = self.scores.get((number, self.revision), {})
        return sum(1 for i in options if table.get(i, {}).get('rank', 3) == 0) < RESERVE_PROVEN

    def second_opinion(self, number: int) -> list[str]:
        """Before giving up on a line, a stronger model looks again at the opened listings the first pass turned down, so one wrong reading cannot end the order."""
        line = self.lines[number]
        table = self.scores.get((number, self.revision), {})
        doubted = [i for i, s in table.items() if s['rank'] >= 2 and self.listings[i]['read'] and self.buyable(i) and self.within_budget(i, line)]
        if not doubted or line.get('reviewed_at') == self.revision:
            return []
        line['reviewed_at'] = self.revision
        doubted.sort(key=lambda i: -self.fit_score(i, line))
        self.grade_batch(number, doubted[:GRADE_BATCH], table, JUDGE)
        return self.with_evidence(number, self.ranked(number))

    def price_walk(self, number: int, pick: str) -> str:
        """Use a category listing with a price limit to look only below the current choice, stepping the limit down as cheaper fits appear."""
        spec = self.role('browse')
        shelf = self.field(pick, 'section')
        if not (spec.get('tool') and spec.get('cap') and spec.get('label') and shelf):
            return pick
        line = self.lines[number]
        for _ in range(CHEAP_WALKS):
            floor = self.price(pick)
            if floor is None or floor <= 0 or not self.can_read(True):
                break
            args = dict(spec.get('fixed') or {})
            args[spec['label']] = shelf
            args[spec['cap']] = floor - 1 if float(floor).is_integer() else floor * 0.99
            below = [i for i in self.absorb(self.call(spec['tool'], args)) if self.belongs(i, line) and self.buyable(i)]
            if not below:
                break
            unread = [i for i in below if not self.listings[i]['read']]
            self.read_listings(self.reading_order(line, unread)[:4])
            cheaper = [i for i in self.with_evidence(number, self.ranked(number)) if (self.price(i) if self.price(i) is not None else 1e12) < floor]
            if not cheaper:
                break
            pick = cheaper[0]
        return pick

    def pick_for(self, number: int) -> str | None:
        line = self.lines[number]
        cheapest = self.plan.get('pick') == 'cheapest'
        walkable = bool(self.role('browse').get('cap') and self.role('browse').get('label'))
        if cheapest and not walkable and line.get('swept_at') != self.revision and line.get('sweeps', 0) < 2:
            line['swept_at'] = self.revision
            line['sweeps'] = line.get('sweeps', 0) + 1
            self.sweep(number)
            self.read_for(line, 4)
        options = self.ranked(number)
        tries = 0
        while self.needs_better(number, options) and tries < 5 and self.can_read(self.repairing()):
            tries += 1
            if not self.widen(number):
                break
            options = self.ranked(number)
        rounds = 0
        while self.short_of_reserve(number, options) and rounds < RESERVE_ROUNDS and self.can_read():
            rounds += 1
            if not self.widen(number):
                break
            options = self.ranked(number)
        if not self.scores.get((number, self.revision)):
            options = self.ranked(number)
        fits = self.with_evidence(number, options)
        if not fits and line.get('named_shelf') and not line.get('named_failed'):
            line['named_failed'] = True
            self.revision += 1
            options = self.ranked(number)
            while not self.with_evidence(number, options) and tries < 7 and self.can_read(self.repairing()):
                tries += 1
                if not self.widen(number):
                    break
                options = self.ranked(number)
            fits = self.with_evidence(number, options)
        if not fits:
            fits = self.second_opinion(number)
        while not fits and tries < 9 and self.can_read(True):
            tries += 1
            if not self.widen(number):
                break
            fits = self.with_evidence(number, self.ranked(number))
        if not fits:
            self.note('no_fit', item=line['name'], seen=len(options))
            return None
        fits = self.second_look(number, fits)
        if not fits:
            self.note('no_fit', item=line['name'], seen=0)
            return None
        pick = fits[0]
        if not cheapest and self.plan.get('pick') not in ('most', 'least'):
            pick = next((i for i in fits if i in self.basket), pick)
        if cheapest:
            pick = self.cheaper_guard(number, pick)
            pick = self.price_walk(number, pick)
            confirmed = self.second_look(number, [pick])
            if not confirmed:
                remaining = [i for i in self.with_evidence(number, self.ranked(number)) if i != pick]
                if not remaining:
                    self.note('no_fit', item=line['name'], seen=0)
                    return None
                pick = remaining[0]
        score = self.scores[(number, self.revision)].get(pick, {})
        self.note('pick', line=line['name'], pick=pick, rank=score.get('rank'), checks=score.get('checks'), needs=self.live_needs(line), price=self.price(pick), shelf=self.field(pick, 'section'), options=len(fits))
        return pick

    def fill_basket(self) -> bool:
        self.cart_failed = False
        lines: dict[int, str] = {}
        for number in range(len(self.lines)):
            chosen = self.pick_for(number)
            if chosen is None:
                return False
            lines[number] = chosen
        wanted = {lines[n]: self.lines[n]['qty'] for n in lines}
        for _ in range(2):
            if wanted != self.basket and self.left() > 2:
                self.set_cart(wanted)
        if wanted != self.basket:
            self.note('cart_trouble', wanted=list(wanted), held=list(self.basket))
            self.cart_failed = True
            stubborn = next((i for i in wanted if self.basket.get(i) != wanted[i]), None)
            if stubborn:
                self.listings[stubborn]['gone'] = True
                self.revision += 1
                self.retry_fill = True
            return False
        if not self.fit_total(lines):
            self.note('over_budget', total=self.cart_total(), budget=self.budget)
            return False
        return True

    def cart_total(self) -> float:
        return sum((self.price(i) or 0) * q for i, q in self.basket.items())

    def fit_total(self, lines: dict[int, str]) -> bool:
        """Bring a cart over the shared limit back under it by swapping the dearest lines for their cheapest listing with evidence; False when it cannot be done."""
        if self.budget is None or self.plan.get('budget_each'):
            return True
        for _ in range(len(self.lines) + 1):
            if self.cart_total() <= self.budget:
                return True
            swapped = False
            for number in sorted(lines, key=lambda n: -(self.price(lines[n]) or 0)):
                current = lines[number]
                cheaper = [i for i in self.with_evidence(number, self.ranked(number)) if (self.price(i) or 1e12) < (self.price(current) or 0)]
                if cheaper:
                    lines[number] = min(cheaper, key=lambda i: self.price(i) or 1e12)
                    self.set_cart({lines[n]: self.lines[n]['qty'] for n in lines})
                    swapped = True
                    break
            if not swapped:
                return False
        return self.cart_total() <= self.budget

    def one_id(self, spec: dict[str, Any], ident: str) -> Any:
        """A single id shaped the way this read tool takes it."""
        kind = (self.schema.get(spec.get('tool') or '', {}).get('properties') or {}).get(spec.get('id') or 'id', {}).get('type')
        return [ident] if kind == 'array' else ident

    def refresh_cart(self) -> None:
        """Re-read what the cart costs now: the cart view when the world has one, otherwise each line's listing."""
        view = self.role('view')
        if view.get('tool') and self.left() > 2:
            self.absorb(self.call(view['tool'], {}))
            return
        spec = self.role('offer') if self.role('offer').get('tool') else self.role('facts')
        for ident in list(self.basket):
            if self.left() <= 3:
                return
            self.absorb(self.call(spec.get('tool'), {spec.get('id') or 'id': self.one_id(spec, ident)}))

    def take_notes(self, light: bool = False) -> str:
        notes = self.notes_waiting
        self.notes_waiting = []
        if not notes:
            return ''
        basket = dump([{'id': i, 'title': clip(self.field(i, 'title') or '', 90)} for i in self.basket])
        prompt = ('NOTE. The store showed these notes while a shopping agent worked. Return ONE JSON object: {"needs":[{"item":number,"property":name,"value":value}],"drop":[{"item":number,'
                  '"property":name}],"budget":new limit as a number or null,"sold_out":true when a chosen listing has run out or left the shelf,"gone":[ids],"price_changed":true when the note '
                  'says the price of a chosen listing moved}. A note that gives up on a property, says to forget it or that it no longer matters is a drop.\n\n'
                  f"ITEMS: {[l['name'] for l in self.lines]}\nBASKET: {basket}\nNOTES: {dump(notes)}")
        got = self.model(prompt, FAST, 700)
        if got is None:
            self.notes_retried = not self.notes_retried
            if self.notes_retried:
                self.notes_waiting = notes + self.notes_waiting
            return ''
        self.notes_retried = False
        self.apply_changes(got)
        gone = [str(g) for g in as_list(got.get('gone')) if str(g) in self.listings]
        if got.get('sold_out') and not gone and not light:
            gone = self.find_gone()
        for ident in gone:
            self.listings[ident]['gone'] = True
            self.revision += 1
        if got.get('price_changed') and not light:
            self.refresh_cart()
            self.revision += 1
        self.note('notes', n=len(notes), gone=gone, budget=self.budget, price=bool(got.get('price_changed')))
        return 'gone' if gone else 'changed'

    def find_gone(self) -> list[str]:
        if len(self.basket) == 1:
            return list(self.basket)
        spec = self.role('offer') if self.role('offer').get('tool') else self.role('facts')
        out: list[str] = []
        for ident in list(self.basket):
            if self.left() <= 3:
                break
            self.listings[ident]['read'] = False
            payload = self.call(spec.get('tool'), {spec.get('id') or 'id': self.one_id(spec, ident)})
            self.absorb(payload, read=True)
            if self.field(ident, 'stock') is False or self.field(ident, 'stock') == 0:
                out.append(ident)
        return out

    def confirm(self) -> bool:
        """Show the shopper the cart when the request asks for that. True when the answer changed anything, so the cart has to be worked out again."""
        spec = self.role('ask')
        current = dump(sorted(self.basket.items()))
        if not self.plan.get('approval') or self.confirmed == current or self.messages_left <= 0 or not spec.get('tool') or self.left() <= 3:
            return False
        lines = '; '.join(f"{q} x {clip(self.field(i, 'title') or i, 80)}" for i, q in self.basket.items())
        text = f'Before I pay, this is what is in the cart: {lines}. Is that all right with you?'
        reply = self.call(spec['tool'], {spec.get('text') or 'text': text})
        self.messages_left -= 1
        self.confirmed = current
        before = self.revision
        self.hear(text, spoken(reply))
        return self.revision != before

    def finish_args(self, tool: str | None, args: dict[str, Any], intent: str, sentence: str = '', avoid: str = '') -> dict[str, Any]:
        """Complete a finishing call's required arguments: a choice by what it means, and any required text with the given sentence."""
        schema = self.schema.get(tool or '') or {}
        props = schema.get('properties') or {}
        for key in schema.get('required') or []:
            if key in args:
                continue
            choices = props.get(key, {}).get('enum')
            if choices:
                match = next((v for v in choices if re.search(intent, words_of(v)) and not (avoid and re.search(avoid, words_of(v)))), None)
                if match is not None:
                    args[key] = match
            elif props.get(key, {}).get('type') == 'string' and sentence:
                args[key] = sentence
        return args

    def place_order(self) -> bool:
        spec = self.role('order')
        self.orders_tried += 1
        self.call(spec.get('tool'), self.finish_args(spec.get('tool'), dict(spec.get('args') or {}), BUY_WORDS, avoid=DECLINE_WORDS))
        return self.done

    def decline(self) -> None:
        spec = self.role('decline')
        args = dict(spec.get('args') or {})
        sentence = 'I looked through the shop and could not find a listing that fits everything you asked for, so I am stopping here without buying.'
        if spec.get('text'):
            args[spec['text']] = sentence
        self.call(spec.get('tool'), self.finish_args(spec.get('tool'), args, DECLINE_WORDS, sentence))

    def last_gasp(self) -> None:
        if self.done:
            return
        if self.basket and len(self.basket) >= len(self.lines):
            self.place_order()
        if not self.done:
            self.decline()

    def basket_still_good(self) -> bool:
        """Whether the cart held now still covers every line, nothing in it has gone, and it is within the current limit."""
        if len(self.basket) < len(self.lines) or any(self.listings[i]['gone'] for i in self.basket if i in self.listings):
            return False
        return self.budget is None or bool(self.plan.get('budget_each')) or self.cart_total() <= self.budget

    def checkout_loop(self) -> None:
        stuck = 0
        for _ in range(MAX_ROUNDS):
            if self.done:
                return
            heard = self.take_notes()
            if heard:
                stuck = 0
            if self.left() <= 2 or self.clock() > MODEL_CUTOFF:
                self.last_gasp()
                return
            filled = self.fill_basket()
            if self.done:
                return
            if self.notes_waiting:
                continue
            if not filled:
                if self.retry_fill:
                    self.retry_fill = False
                    continue
                if self.orders_tried == 0 and self.basket_still_good():
                    self.place_order()
                    if self.done:
                        return
                    continue
                self.note('declined', basket=list(self.basket))
                self.decline()
                if self.done:
                    return
                self.declines_tried += 1
                if self.declines_tried >= 3:
                    break
                continue
            if self.confirm() or self.notes_waiting:
                continue
            attempt = dump(sorted(self.basket.items()))
            if attempt == self.last_order_text and not heard:
                stuck += 1
                self.refresh_cart()
                if stuck >= 2:
                    self.note('stuck', basket=list(self.basket))
                    break
            self.last_order_text = attempt
            if self.place_order():
                return
        self.last_gasp()

    def run(self) -> list[dict[str, Any]]:
        if self.done:
            return self.trace
        if not self.bind():
            self.note('unbound')
            return self.trace
        self.scout()
        self.settle_stated_references()
        self.interview()
        self.take_notes(light=True)
        self.chase_answers()
        self.checkout_loop()
        self.note('end', finished=self.done, steps=self.step, cost=round(self.spend, 5), lines=len(self.lines))
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception as exc:
        episode.note('crash', error=f'{type(exc).__name__}: {exc}'[:300])
        try:
            episode.last_gasp()
        except Exception as again:
            episode.note('crash', error=f'{type(again).__name__}: {again}'[:300])
        return episode.trace


__all__ = ['agent_main']
