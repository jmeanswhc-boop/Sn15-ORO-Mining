from __future__ import annotations
import json
import random
import re
import time
from collections import Counter
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
MODEL_CHAIN = {'chutes': ['zai-org/GLM-5.2-TEE', 'deepseek-ai/DeepSeek-V3.2-TEE'], 'openrouter': ['z-ai/glm-5.2', 'deepseek/deepseek-v3.2']}
SEARCH_DEADLINE = 175.0
STRONG_MODEL = {'openrouter': {'request_record': 'anthropic/claude-opus-5.5', 'basket': 'anthropic/claude-opus-5.5', 'reference': 'anthropic/claude-opus-5.5', 'tool_roles': 'anthropic/claude-sonnet-5.5', 'cart_direction': 'anthropic/claude-sonnet-5.5', 'record_keys': 'anthropic/claude-sonnet-5.5'}}
STRONG_RATES = {'anthropic/claude-opus-5.5': (4.0, 20.0), 'anthropic/claude-sonnet-5.5': (2.0, 10.0)}
STRONG_SPEND = 0.045
MODEL_DEADLINE = 250.0
CLOSING_STEPS = 4
SEARCH_HOLDBACK = 9
PAGE_CAP = 3
SHEET_READS = 12
MEASURE_READS = 36
QUESTION_CAP = 4
JUDGE_BATCH = 12
MONEY_RE = '(?:[$€£¥₱₹]\\s*)(\\d[\\d,]*(?:\\.\\d+)?)|(\\d[\\d,]*(?:\\.\\d+)?)\\s*(?:us\\s+|u\\.s\\.\\s+)?(?:cents?|cts?|dollars?|usd|bucks|euros?|eur|pesos?|php|rupees?|pounds?|yen)\\b'
RULE_OUT_RE = "\\b(?:leave (?:out|off)|rule[sd]? out|steer clear|stay away|keep away|avoid|skip|don'?t want|do not want|no thanks)\\b"
INSIST_RE = '\\b(?:must|has to|have to|got to|gotta|needs? to|should|only|nothing else|anything but|insist)\\b'
NOT_RE = "\\b(?:no|not|never|without|out)\\b|n't\\b"
THRIFT_RE = 'cheap|\\bleast\\b|lowest|low[- ](?:cost|price)|bargain|inexpensive|economical|affordable|smallest (?:total|sum|bill|overall)'
LOWEST_RE = 'cheapest|least expensive|costs? (?:the )?least|lowest[- ]?pric|(?:smallest|lowest) (?:total|sum|bill)|least (?:in total|overall)'
TOP_RE = '\\b(?:greatest|largest|biggest|highest|longest|widest|tallest|heaviest|strongest|smallest|lowest|shortest|lightest|best)\\b|(?<!at )\\b(?:most|least)\\b'
STORY_CUES = [('photo|picture|\\blooks?\\b|\\bmatch|clash|closet|outfit|pretty|appearance', ('pattern', 'shape')), ('\\btags?\\b|fabric|\\bfeel\\b|\\bitch|\\bscratch|\\bsoft\\b', ('material', 'fabric'))]
held_1 = 'a an and are as at be but by for if in into is it no not of on or such that the their then there these they this to was will with'.split()
FILLER = frozenset(held_1)
_proxy = ProxyClient(max_retries=1, timeout=60)
_proxy_quick = ProxyClient(max_retries=1, timeout=25)
_proxy_env = ProxyClient(max_retries=1, timeout=30)

def low_text(value: Any) -> str:
    held_2 = str(value or '').lower()
    return held_2

def flat(value: Any) -> str:
    held_3 = low_text(value)
    text = re.sub('(?<=\\d)(?=[a-z])|(?<=[a-z])(?=\\d)', ' ', held_3)
    held_4 = ' '.join(re.findall('[a-z0-9]+', text))
    return held_4

def word_tokens(value: Any) -> list[str]:
    held_5 = re.findall('[a-z0-9]+', low_text(value))
    return held_5

def only_compared(value: str, request: str) -> bool:
    held_6 = low_text(value).strip()
    words = re.escape(held_6)
    uses = list((m.start() for m in re.finditer('(?<![\\w]){}(?![\\w])'.format(words), low_text(request))))
    held_7 = bool(uses) and len(flat(value).split()) >= 2 and all((re.search('\\b(?:like|as|than)\\s+(?:a|an|the|some)?\\s*$', low_text(request)[max(0, i - 25):i]) for i in uses))
    return held_7

def bound_check(value: str, quote: str, same_unit: bool=False) -> str | None:
    held_8 = low_text(value)
    bound = re.search('\\b(at most|at least|up to|under|over|less than|more than|no more than|no less than|maximum|minimum|max|min)\\b\\s*[^\\w\\s]?\\s*(\\d[\\d,]*(?:\\.\\d+)?)\\s*([a-z\\"\']*)', held_8)
    if bound:
        kind, limit, unit = (bound.group(1), float(bound.group(2).replace(',', '')), re.sub('[\\"”″]', 'in', bound.group(3))[:2])
        held_9 = low_text(quote)
        text = re.sub('[\\"”″]', ' in ', held_9)
        figures = list(((float(n.replace(',', '')), u) for n, u in re.findall("(\\d[\\d,]*(?:\\.\\d+)?)\\s*-?\\s*([a-z']*)", text)))
        held_10 = list((n for n, u in figures if unit and u[:2] == unit))
        if not held_10:
            if 1 == len(figures) and (not same_unit):
                held_10 = list((n for n, u in figures if not u))
            else:
                held_10 = []
        same = held_10
        if same:
            figure = same[0]
            if kind in ('at least', 'no less than', 'minimum', 'min'):
                ok = limit <= figure
            elif kind in ('at most', 'up to', 'no more than', 'maximum', 'max'):
                ok = limit >= figure
            elif not kind in ('over', 'more than'):
                ok = limit > figure
            else:
                ok = limit < figure
            if ok:
                return 'yes'
            return 'no'
        else:
            return None
    else:
        return None

def word_set(text: Any) -> set[str]:
    out = set()
    for word in flat(text).split():
        out.add(word)
        if 3 < len(word):
            if word.endswith('es'):
                out.add(word[:-2])
        if 3 < len(word):
            if word.endswith('s'):
                out.add(word[:-1])
    return out

def first_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    elif isinstance(value, (int, float)):
        held_11 = float(value)
        return held_11
    elif not isinstance(value, str):
        if isinstance(value, dict):
            for key in ('amount', 'value', 'price', 'current', 'sale'):
                if key in value:
                    held_13 = first_number(value[key])
                    return held_13
        return None
    else:
        found = re.search('\\d[\\d,]*(?:\\.\\d+)?', value)
        if found:
            held_12 = float(found.group(0).replace(',', ''))
            return held_12
        return None

def stock_of(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    elif not isinstance(value, (int, float)):
        if isinstance(value, str):
            if re.search('out of stock|sold out|unavailable|no stock|not available|none left|^no$|^false$|^0$', value.strip(), re.I):
                return False
            elif re.search('in stock|available|yes|true|^\\d+$', value.strip(), re.I):
                return True
        return None
    else:
        held_14 = 0 < value
        return held_14

def message_texts(value: Any) -> list[str]:
    if None is value:
        held_15 = []
        return held_15
    elif isinstance(value, str):
        if value.strip():
            held_16 = [value]
            return held_16
        held_17 = []
        return held_17
    elif not isinstance(value, dict):
        if isinstance(value, (list, tuple)):
            held_19 = list((t for v in value for t in message_texts(v)))
            return held_19
        held_20 = []
        return held_20
    else:
        held_18 = list((t for v in value.values() for t in message_texts(v)))
        return held_18
BOUND_RE = re.compile('\\b(at most|at least|up to|under|over|less than|more than|no more than|no less than|between|maximum|minimum|max|min)\\b\\s*[^\\w\\s]?\\s*\\d', re.I)

def search_key(text: str) -> str:
    held_21 = ' '.join(sorted(set((re.sub('(es|s)$', '', w) if 3 < len(w) else w for w in text.split()))))
    return held_21

def search_text(parts: list[str], limit: int | None) -> str:
    out: list[str] = []
    for word in re.findall('[a-z0-9]+', ' '.join((p for p in parts if not BOUND_RE.search(p))).lower()):
        if not (word in FILLER or (2 > len(word) and (not word.isdigit())) or word in out):
            out.append(word)
            if 6 == len(out):
                break
        else:
            continue
    text = ' '.join(out)
    while limit and limit < len(text) and out:
        out.pop()
        text = ' '.join(out)
    return text

def peel(obs: Any) -> tuple[Any, list[str], list[str]]:
    refusal: list[str] = []
    said: list[str] = []
    if isinstance(obs, dict):
        if 'refusal' in obs:
            if 'observation' in obs:
                held_22 = obs.get('refusal')
                refusal = message_texts(held_22)
                obs = obs.get('observation')
    for _ in range(3):
        if isinstance(obs, dict) and 2 == len(obs):
            talk = list((k for k, v in obs.items() if isinstance(v, list) and v and all((isinstance(x, str) and len(x.split()) >= 3 for x in v))))
            if not 1 != len(talk):
                said += obs[talk[0]]
                obs = next((v for k, v in obs.items() if talk[0] != k))
            else:
                break
        else:
            break
    held_23 = (obs, refusal, said)
    return held_23

def is_failure(obs: Any) -> bool:
    if isinstance(obs, dict):
        if any((re.search('error|issue|fail', low_text(k)) and v not in (None, False, '', [], {}, 0) for k, v in obs.items())):
            return True
        elif not any((re.search('^(ok|success|succeeded)$', low_text(k)) and False is v for k, v in obs.items())):
            held_24 = any((isinstance(v, str) and (re.fullmatch('[A-Z_]{4,}', v) and re.search('NO|FAIL|INVALID|OVER|DENIED|ERROR|BARE|EMPTY', v) or re.fullmatch('(error|failed|failure|invalid|denied|rejected)', v.strip(), re.I)) for v in obs.values()))
            return held_24
        else:
            return True
    elif not isinstance(obs, str):
        held_26 = None is obs
        return held_26
    else:
        held_25 = bool(re.search('\\b(error|invalid|not found|fail)', obs, re.I))
        return held_25

class Timer:

    def __init__(self) -> None:
        self.start = time.monotonic()

    def elapsed(self) -> float:
        held_27 = time.monotonic() - self.start
        return held_27

class Oracle:

    def __init__(self, clock: Timer) -> None:
        self.vendor = getenv('INFERENCE_PROVIDER', 'chutes')
        self.model_list = MODEL_CHAIN.get(self.vendor, MODEL_CHAIN['chutes'])
        self.strong_model = STRONG_MODEL.get(self.vendor) or {}
        self.elapsed = 0.0
        self.timer = clock
        self.call_count = 0

    def ask_model(self, name: str, guide: str, text: str, properties: dict, required: list[str], max_tokens: int=1500, think: bool=False, critical: bool=False) -> dict | None:
        tool = {'type': 'function', 'function': {'name': name, 'description': 'Write down the answer.', 'parameters': {'type': 'object', 'properties': properties, 'required': required}}}
        if STRONG_SPEND > self.elapsed:
            strong = self.strong_model.get(name)
        else:
            strong = None
        chain = ([strong] if strong else []) + self.model_list
        plan = [chain[0], chain[0], *chain[1:]]
        if critical:
            plan = plan + plan
        limited: set[str] = set()
        hits = 0
        for attempt, planned in enumerate(plan):
            if planned in limited:
                model = next((m for m in chain if m not in limited), planned)
            else:
                model = planned
            if not self.timer.elapsed() > MODEL_DEADLINE:
                held_29 = len(plan) - 1 == attempt
                if held_29:
                    held_29 = 0 < attempt
                held_28 = held_29
                if not held_28:
                    held_28 = model.startswith('anthropic/')
                last = held_28
                body: dict[str, Any] = {'model': model, 'temperature': 0 if 0 == attempt else 0.3, 'max_tokens': max_tokens + (3000 if think else 0), 'messages': [{'role': 'system', 'content': guide}, {'role': 'user', 'content': text}], 'tools': [tool], 'tool_choice': {'type': 'function', 'function': {'name': name}}}
                if last:
                    del body['tools'], body['tool_choice']
                    body['messages'][1]['content'] = text + '\n\nReply with one JSON object only, with these fields: ' + json.dumps(properties, ensure_ascii=False)[:3000]
                if 'chutes' == self.vendor:
                    body['chat_template_kwargs'] = {'thinking': think, 'enable_thinking': think}
                elif 'glm' in model:
                    body['reasoning'] = {'enabled': think}
                status = None
                try:
                    if self.timer.elapsed() < MODEL_DEADLINE - 70:
                        client = _proxy
                    else:
                        client = _proxy_quick
                    answer = client.post_verbose('/inference/chat/completions', json_data=body)
                    reply, status = (answer.data, (answer.error or {}).get('status'))
                except Exception:
                    reply = None
                self.call_count = self.call_count + 1
                if model in STRONG_RATES:
                    if isinstance(reply, dict):
                        if isinstance(reply.get('usage'), dict):
                            usage, rate = (reply['usage'], STRONG_RATES[model])
                            self.elapsed += ((usage.get('prompt_tokens') or 0) * rate[0] + (usage.get('completion_tokens') or 0) * rate[1]) / 1000000.0
                parsed = self.parse_reply(reply, name, properties)
                if isinstance(parsed, dict) and all((key in parsed for key in required)):
                    return parsed
                elif 429 == status:
                    hits = hits + 1
                    if 4 >= hits:
                        plan.append(model)
                    if self.timer.elapsed() < MODEL_DEADLINE - 30:
                        held_30 = min(20.0, random.uniform(2.0, 5.0) * 1.6 ** (hits - 1))
                        time.sleep(held_30)
                    continue
                elif self.timer.elapsed() < MODEL_DEADLINE - 30:
                    held_31 = min(15.0, (4.0 if 429 == status else 2.0) * (attempt + 1))
                    time.sleep(held_31)
            else:
                return None
        return None

    @staticmethod
    def parse_reply(reply: Any, name: str, properties: dict) -> dict | None:
        try:
            message = reply['choices'][0]['message']
        except (KeyError, IndexError, TypeError):
            return None
        raw = None
        for call in message.get('tool_calls') or []:
            held_32 = call.get('function')
            if not held_32:
                held_32 = {}
            function = held_32
            if function.get('name') in (name, None, ''):
                raw = function.get('arguments')
                break
        if None is raw:
            held_33 = message.get('content')
            if not held_33:
                held_33 = ''
            content = held_33
            found = re.search('\\{.*\\}', content, re.S)
            if found:
                raw = found.group(0)
            else:
                raw = None
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                raw = repaired_json(raw)
        if isinstance(raw, dict):
            for key, spec in properties.items():
                value = raw.get(key)
                if isinstance(value, str):
                    if spec.get('type') in ('array', 'object'):
                        if value.strip()[:1] in '[{':
                            try:
                                raw[key] = json.loads(value)
                            except ValueError:
                                raw[key] = [] if spec.get('type') == 'array' else {}
            return raw
        else:
            return None

def repaired_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith('{'):
        for end in range(len(text), 0, -1):
            if not text[end - 1] not in ',]}"0123456789eul':
                head = text[:end].rstrip(',')
                stack, quoted, escaped = ([], False, False)
                for ch in head:
                    if escaped:
                        escaped = False
                    elif '\\' == ch:
                        escaped = quoted
                    elif '"' == ch:
                        quoted = not quoted
                    elif not quoted and ch in '[{':
                        stack.append(']' if '[' == ch else '}')
                    elif not quoted:
                        if ch in ']}':
                            if stack:
                                stack.pop()
                if not quoted:
                    try:
                        value = json.loads(head + ''.join(reversed(stack)))
                    except ValueError:
                        continue
                    if isinstance(value, dict):
                        return value
                    return None
                else:
                    continue
            else:
                continue
        return None
    else:
        return None

class StepResult:

    def __init__(self, obs: Any, refusal: list[str], said: list[str], finished: bool) -> None:
        self.seen, self.refusals, self.spoken, self.done = (obs, refusal, said, finished)

class Session:

    def __init__(self, env: dict) -> None:
        self.bind = env.get('binding') or {}
        held_34 = env.get('policy_view')
        if not held_34:
            held_34 = {}
        view = held_34
        self.request_text = str(view.get('query') or '')
        self.first_view = view.get('opening')
        self.tool_list = list((t for t in view.get('tools') or [] if isinstance(t, dict)))
        self.step_cap = int(view.get('max_steps') or 38)
        held_35 = view.get('finished')
        self.done = bool(held_35)
        self.step = 0
        self.inbox_msgs: list[str] = []
        self.trail: list[dict] = []

    def steps_left(self) -> int:
        held_36 = self.step_cap - self.step
        return held_36

    def call(self, tool: str, args: dict) -> StepResult:
        self.step = self.step + 1
        payload = {**self.bind, 'step': self.step, 'action': {'name': tool, 'args': args}}
        result, problem, started = (None, '', time.monotonic())
        for attempt in range(8):
            try:
                answer = _proxy_env.post_verbose('/environment/call', json_data=payload)
                result, error = (answer.data, answer.error or {})
            except Exception as exc:
                result, error = (None, {'body': '{}: {}'.format(type(exc).__name__, exc)})
            if not (isinstance(result, dict) and ('observation' in result or 'finished' in result)):
                status, body = (error.get('status'), str(error.get('body') or ''))
                problem = '{} {}'.format(status, body[:160])
                if not (409 == status and 'finish' in body.lower()):
                    if 409 == status:
                        expected = re.search('expected (\\d+)', body)
                    else:
                        expected = None
                    if not expected:
                        if status in (400, 401, 403, 404, 422) or time.monotonic() - started > 50:
                            break
                        held_39 = min(2.0 + 2 * attempt, 8.0)
                        time.sleep(held_39)
                    else:
                        held_38 = expected.group(1)
                        self.step = int(held_38)
                        payload['step'] = self.step
                        continue
                else:
                    self.done = True
                    held_37 = StepResult({}, [], [], True)
                    return held_37
            else:
                break
        if isinstance(result, dict):
            held_40 = result.get('finished')
            self.done = bool(held_40)
            if isinstance(result.get('step'), int):
                self.step = max(self.step, result['step'])
            held_41 = result.get('observation')
            obs, refusal, said = peel(held_41)
            self.inbox_msgs += said
            self.trail.append({'step': self.step, 'tool': tool, 'args': args, 'observation': result.get('observation'), 'finished': self.done})
            held_42 = StepResult(obs, refusal, said, self.done)
            return held_42
        else:
            raise RuntimeError('environment call failed: {}'.format(problem))
ROLE_NAMES = ('search', 'filter', 'view', 'facts', 'cart_add', 'cart_remove', 'cart_view', 'checkout', 'order', 'confirm', 'decline', 'ask')
SLOTS = ('text', 'id', 'quantity', 'page', 'limit', 'category', 'brand', 'max_price', 'token')
NEEDED_SLOTS = {'search': ('text',), 'view': ('id',), 'facts': ('id',), 'cart_add': ('id',), 'cart_remove': ('id',), 'ask': ('text',), 'order': ('id',), 'confirm': ('token',)}
MODEL_ONLY_ROLES = ('order', 'confirm')
ROLE_PROMPT = "Match each job of a shop assistant to the interface tool that does it. The jobs:\n- search: find products by keywords (the tool takes free text);\n- filter: find products by department, maker or a price cap (sometimes the search tool in another mode);\n- view: open one product in full by its id;\n- facts: open one product's property sheet by its id;\n- cart_add: put a product in the cart; cart_remove: take one out (often one tool with a different action value);\n- cart_view: list the cart;\n- checkout: buy what is in the cart (ends the task, or starts a purchase that a confirm call finishes);\n- order: buy one product named in the call itself, by its id, without a cart;\n- confirm: the second step of a purchase: a call that takes the code or token the checkout or order call returned;\n- decline: end the task with no purchase (ends the task, a note may go with it);\n- ask: send the shopper a message or a question.\nFor every job name the tool, which of its arguments carries each slot (text, id, quantity, page, limit, category, brand, max_price, token - page: a page number, offset or next-page mark; limit: how many results one page shows; token: a code an earlier call returned; leave a slot empty when the tool has no such argument) and any fixed argument values the job needs, such as a mode or an action picked from an enum. For search and filter also put in cheapest_first the argument values that sort the lowest price first, when the tool can sort (else leave it empty). Use only the tool and argument names given. Skip a job that no tool does."
ROLE_WORDS = {'search': 'search|find|query|look|seek|browse|scan|locate|explore|catalog|hunt|discover', 'filter': 'filter|field|categor|department|brand|price|criteri', 'view': 'view|show|listing|detail|article|item in full|everything about|get|fetch|open|load|display|single|page', 'facts': 'datasheet|\\bpropert|\\bspec|\\battribute|\\btrait|\\bfacts?\\b|\\bfeature', 'cart_add': 'cart|bag|basket|trolley', 'cart_remove': 'cart|bag|basket|trolley', 'cart_view': 'cart|bag|basket|trolley', 'checkout': 'check ?out|purchase|order|buy|settle|pay|complete|submit|confirm', 'decline': 'no purchase|no order|give up|decline|close out|abandon|without (buying|purchase|a purchase|an order|ordering)|cancel|quit|walk away|skip|leave|end without|refuse|reject|stop', 'ask': 'question|ask|message|customer|shopper|client|buyer|contact|write', 'order': 'order|buy|purchase|place', 'confirm': 'confirm|finali[sz]e|complete|commit|second step'}
SLOT_WORDS = {'text': 'text|query|term|keyword|word|phrase|note|body|message|msg|remark|comment|reason|explan|content|q$|kw|lookup|search|keys', 'id': 'id$|_id|^id|pid|product|item|target|sku|article|ref$|_ref|listing', 'quantity': 'qty|quant|how_many|count|amount|units|number', 'page': 'page|^pg$|^p$|^pn$|token|cursor|offset|further|continu|after|first|start|skip', 'category': 'categ|depart|dept|section|aisle|label|group|class|shelf', 'brand': 'brand|maker|company|manufact|vendor|make$|producer|marque', 'max_price': 'cap|max|price|budget|spend|limit|ceiling|cost', 'limit': 'limit|per.?page|page.?size|size$|rows|max.?results|top$|^n$|how.?many.?results|batch', 'token': 'token|quote|reservation|confirm|ticket|draft|handle|code$|_code|key$'}
SLOT_NOT = {'max_price': 'min|low|from|floor|result|row|per.?page|page', 'page': 'size|per|limit|count|len', 'quantity': 'page|min|result|row', 'limit': 'price|cost|budget|spend|char|len|word', 'token': 'product|item|listing|sku|page|cursor|search|query', 'text': 'id$|_id|ref$|_ref|categ|brand|maker|manufact|group|section|aisle|token|cursor|page'}
ENUM_WORDS = {'search': 'key|text|term|word|free|typ|query|search|phrase|plain', 'filter': 'field|filter|criteri|struct|facet|attrib|categ|exact|param', 'cart_add': '^(add|include|put|insert|plus|raise|increase|incr|more|up|in|keep|set|append|place|push|load|attach|hold|reserve|grab|\\+)', 'cart_remove': '^(remove|discard|delete|drop|minus|lower|decrease|decr|less|down|out|take|clear|reduce|subtract|pull|unload|detach|pop|release|unhold|free|-)', 'checkout': '^(commit|order|buy|purchase|check ?out|checkout|pay|confirm|complete|place|accept|submit|settle)', 'decline': '^(refuse|decline|cancel|abandon|give ?up|quit|reject|none|nothing|skip|walk|leave|no\\b|no_)'}
COUNTER_ROLE = {'search': 'filter', 'filter': 'search', 'cart_add': 'cart_remove', 'cart_remove': 'cart_add', 'checkout': 'decline', 'decline': 'checkout'}
ROLE_ARGS = {'search': ('text', 'page', 'category', 'limit'), 'filter': ('category', 'brand', 'max_price', 'page', 'limit'), 'view': ('id',), 'facts': ('id',), 'cart_add': ('id', 'quantity'), 'cart_remove': ('id', 'quantity'), 'ask': ('text',), 'decline': ('text',), 'order': ('id', 'quantity'), 'confirm': ('token',)}
ADD_WORDS = ENUM_WORDS['cart_add']
REMOVE_WORDS = ENUM_WORDS['cart_remove']

class Toolkit:

    def __init__(self, tools: list[dict]) -> None:
        self.tool_specs: dict[str, dict] = {}
        for raw in tools:
            if isinstance(raw.get('function'), dict):
                spec = raw.get('function')
            else:
                spec = raw
            name = spec.get('name')
            if isinstance(name, str):
                if name:
                    held_43 = spec.get('parameters')
                    if not held_43:
                        held_43 = {}
                    params = held_43
                    self.tool_specs[name] = {'description': str(spec.get('description') or ''), 'properties': params.get('properties') or {}, 'required': list(params.get('required') or [])}
        self.role_map: dict[str, dict] = {}

    def assign_roles(self, brain: Oracle) -> list[str]:
        held_44 = list(({'name': n, 'description': s['description'], 'parameters': {'properties': s['properties'], 'required': s['required']}} for n, s in self.tool_specs.items()))
        listing = json.dumps(held_44, ensure_ascii=False)
        role_item = {'type': 'object', 'properties': {'role': {'type': 'string', 'enum': list(ROLE_NAMES)}, 'tool': {'type': 'string'}, **dict((('{}_arg'.format(slot), {'type': 'string'}) for slot in SLOTS)), 'fixed': {'type': 'array', 'items': {'type': 'object', 'properties': {'arg': {'type': 'string'}, 'value': {'type': 'string'}}, 'required': ['arg', 'value']}}, 'cheapest_first': {'type': 'array', 'items': {'type': 'object', 'properties': {'arg': {'type': 'string'}, 'value': {'type': 'string'}}, 'required': ['arg', 'value']}}}, 'required': ['role', 'tool']}
        shape = {'roles': {'type': 'array', 'items': role_item}}
        held_45 = brain.ask_model('tool_roles', ROLE_PROMPT, 'Tools:\n' + listing, shape, ['roles'], 1800)
        if not held_45:
            held_45 = {}
        answer = held_45
        notes = []
        problems = self.read_role_map(answer, notes)
        for role, needed in (('search', ('search',)), ('view', ('view', 'facts')), ('cart_add', ('cart_add', 'order')), ('checkout', ('checkout', 'order')), ('decline', ('decline',))):
            if not any((r in self.role_map for r in needed)):
                if not any((p.startswith(role + ':') for p in problems)):
                    held_46 = '{}: no tool was given for it'.format(role)
                    problems.append(held_46)
        if problems:
            if answer:
                text = 'Tools:\n{}\n\nYour earlier answer:\n{}\n\nWhat is wrong with it:\n'.format(listing, json.dumps(answer, ensure_ascii=False)[:3000]) + '\n'.join(('- {}'.format(p) for p in problems[:12])) + '\n\nGive the whole answer again with these fixed, using only tool names, argument names and values the tools list.'
                held_47 = 'tool map sent back: {}'.format(problems[:6])
                notes.append(held_47)
                held_48 = brain.ask_model('tool_roles', ROLE_PROMPT, text, shape, ['roles'], 1800)
                if not held_48:
                    held_48 = {}
                again = held_48
                self.read_role_map(again, notes)
        for role, entry in self.role_map.items():
            spec = self.tool_specs[entry['tool']]
            used = set(entry['slots'].values()) | set(entry['fixed']) | set(entry.get('cheap') or {})
            wanted = {'search': ('page', 'limit'), 'filter': ('page', 'max_price', 'category', 'brand', 'limit'), 'cart_add': ('quantity',)}.get(role, ())
            for slot in wanted:
                if not slot in entry['slots']:
                    numeric = slot in ('quantity', 'page', 'max_price', 'limit')
                    arg = next((a for a, prop in spec['properties'].items() if a not in used and (not prop.get('enum')) and re.search(SLOT_WORDS[slot], a.lower()) and (not re.search(SLOT_NOT.get(slot, '$^'), a.lower())) and ((prop.get('type') in ('integer', 'number')) == numeric)), None)
                    if arg:
                        entry['slots'][slot] = arg
                        used.add(arg)
                        held_49 = '{}: slot {} filled with {}'.format(role, slot, arg)
                        notes.append(held_49)
                else:
                    continue
        for role in ROLE_NAMES:
            if role not in self.role_map:
                if role not in MODEL_ONLY_ROLES:
                    guess = self.guess_role(role)
                    if guess:
                        self.role_map[role] = guess
                        held_50 = '{} guessed: {}'.format(role, guess)
                        notes.append(held_50)
        held_51 = self.settle_roles(brain, notes)
        return held_51

    def read_role_map(self, answer: dict, notes: list[str]) -> list[str]:
        problems = []
        for item in answer.get('roles') or []:
            if not (not isinstance(item, dict) or item.get('role') not in ROLE_NAMES or item.get('role') in self.role_map):
                entry = {'tool': str(item.get('tool') or ''), 'slots': dict(((slot, str(item.get('{}_arg'.format(slot)))) for slot in SLOTS if str(item.get('{}_arg'.format(slot)) or '').strip())), 'fixed': {}, 'cheap': {}}
                spec = self.tool_specs.get(entry['tool'])
                if spec:
                    for fixed in item.get('fixed') or []:
                        if isinstance(fixed, dict):
                            if str(fixed.get('arg') or '') in spec['properties']:
                                entry['fixed'][fixed['arg']] = self.cast_value(spec['properties'][fixed['arg']], fixed.get('value'))
                    for sort in item.get('cheapest_first') or []:
                        if isinstance(sort, dict):
                            arg = str(sort.get('arg') or '')
                        else:
                            arg = ''
                        prop = spec['properties'].get(arg)
                        if prop:
                            if arg not in entry['fixed']:
                                if arg not in entry['slots'].values():
                                    held_52 = sort.get('value')
                                    value = self.cast_value(prop, held_52)
                                    if not prop.get('enum') or value in prop['enum']:
                                        entry['cheap'][arg] = value
                raw = {'slots': dict(entry['slots']), 'fixed': dict(entry['fixed'])}
                entry = self.seen_whole(item['role'], entry)
                if not self.is_valid(item['role'], entry):
                    held_53 = 'model mapping of {} rejected: {}'.format(item['role'], entry)
                    notes.append(held_53)
                    problems += list(('{}: {}'.format(item['role'], p) for p in self.map_faults(item['role'], str(item.get('tool') or ''), raw)))
                else:
                    self.role_map[item['role']] = entry
            else:
                continue
        return problems

    def map_faults(self, role: str, tool: str, raw: dict) -> list[str]:
        spec = self.tool_specs.get(tool)
        if spec:
            props, out = (spec['properties'], [])
            out += list(('{} has no argument {!r} (for {})'.format(tool, arg, slot) for slot, arg in raw['slots'].items() if arg not in props))
            for arg, value in raw['fixed'].items():
                enum = (props.get(arg) or {}).get('enum')
                if arg not in props:
                    held_55 = '{} has no argument {!r}'.format(tool, arg)
                    out.append(held_55)
                elif enum:
                    if value not in enum:
                        held_56 = '{}.{} cannot be {!r}; it takes one of {}'.format(tool, arg, value, enum)
                        out.append(held_56)
            covered = set(raw['slots'].values()) | set(raw['fixed'])
            out += list(('{} requires {!r}, which no slot or fixed value fills'.format(tool, arg) for arg in spec['required'] if arg not in covered))
            out += list(('the {} slot is missing (which argument of {} carries it?)'.format(slot, tool) for slot in NEEDED_SLOTS.get(role, ()) if slot not in raw['slots']))
            held_57 = out or ["it did not fit the tool's arguments"]
            return held_57
        else:
            held_54 = ['there is no tool named {!r}'.format(tool)]
            return held_54

    def settle_roles(self, brain: Oracle, notes: list[str]) -> list[str]:
        if not {'cart_add', 'cart_remove'} <= set(self.role_map):
            for name, spec in self.tool_specs.items():
                enums = list(((a, list((e for e in p.get('enum') or [] if None is not e))) for a, p in spec['properties'].items() if len(p.get('enum') or []) >= 2))
                words = '{} {}'.format(name.replace('_', ' '), spec['description']).lower()
                if not (not enums or not re.search(ROLE_WORDS['cart_add'], words)):
                    arg, values = enums[0]
                    held_59 = 'Tool {}: {}\nArgument {} takes one of {}.'.format(name, spec['description'], arg, values)
                    held_58 = brain.ask_model('cart_direction', 'Say which argument value of this shop tool does each thing.', held_59, {'puts_in': {'type': 'string'}, 'takes_out': {'type': 'string'}}, ['puts_in', 'takes_out'], 200)
                    if not held_58:
                        held_58 = {}
                    answer = held_58
                    for role, key in (('cart_add', 'puts_in'), ('cart_remove', 'takes_out')):
                        value = next((v for v in values if low_text(v) == low_text(answer.get(key))), None)
                        if role not in self.role_map:
                            if None is not value:
                                if answer.get('puts_in') != answer.get('takes_out'):
                                    entry = self.seen_whole(role, {'tool': name, 'slots': {}, 'fixed': {arg: value}, 'cheap': {}})
                                    if self.is_valid(role, entry):
                                        self.role_map[role] = entry
                                        held_60 = '{}: {} with {}={} (asked)'.format(role, name, arg, value)
                                        notes.append(held_60)
                    break
                else:
                    continue
        for role, other in (('view', 'facts'), ('facts', 'view')):
            if role not in self.role_map:
                if other in self.role_map:
                    self.role_map[role] = dict(self.role_map[other])
                    held_61 = '{} served by the {} tool'.format(role, other)
                    notes.append(held_61)
        return notes

    @staticmethod
    def cast_value(prop: dict, value: Any) -> Any:
        kind = prop.get('type')
        enum = prop.get('enum')
        if not enum:
            try:
                if 'integer' == kind:
                    held_62 = int(float(value))
                    return held_62
                elif 'number' == kind:
                    held_63 = float(value)
                    return held_63
                elif 'boolean' == kind:
                    held_64 = low_text(value) in ('true', '1', 'yes')
                    return held_64
            except (TypeError, ValueError):
                return value
            return value
        else:
            match = next((e for e in enum if low_text(e) == low_text(value)), None)
            if None is not match:
                return match
            return value

    def is_valid(self, role: str, entry: dict) -> bool:
        spec = self.tool_specs.get(entry.get('tool') or '')
        if spec:
            props = spec['properties']
            if not any((arg not in props for arg in entry['slots'].values())):
                for arg, value in entry['fixed'].items():
                    enum = props[arg].get('enum')
                    if enum:
                        if value not in enum:
                            return False
                covered = set(entry['slots'].values()) | set(entry['fixed'])
                if not any((arg not in covered for arg in spec['required'])):
                    if any((slot not in entry['slots'] for slot in NEEDED_SLOTS.get(role, ()))):
                        return False
                    if 'filter' == role:
                        if not {'category', 'brand', 'max_price'} & set(entry['slots']):
                            return False
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def seen_whole(self, role: str, entry: dict) -> dict:
        spec = self.tool_specs.get(entry.get('tool') or '')
        if spec:
            props = spec['properties']
            entry['slots'] = dict(((slot, arg) for slot, arg in entry['slots'].items() if arg in props))
            entry['fixed'] = dict(((arg, value) for arg, value in entry['fixed'].items() if arg in props))
            used = set(entry['slots'].values()) | set(entry['fixed']) | set(entry.get('cheap') or {})
            for arg, prop in props.items():
                enum = list((e for e in prop.get('enum') or [] if None is not e))
                if not (not enum or arg in used or role not in ENUM_WORDS or (4 < len(enum) and re.search(SLOT_WORDS['category'], arg.lower()))):
                    hit = next((e for e in enum if re.search(ENUM_WORDS[role], low_text(e))), None)
                    if None is hit:
                        if 2 == len(enum):
                            other = list((e for e in enum if re.search(ENUM_WORDS[COUNTER_ROLE[role]], low_text(e))))
                            if 1 == len(other):
                                hit = next((e for e in enum if e not in other), None)
                            else:
                                hit = None
                    if None is not hit:
                        entry['fixed'][arg] = hit
                        used.add(arg)
                else:
                    continue
            for slot in ROLE_ARGS.get(role, ()):
                if not slot in entry['slots']:
                    arg = self.slot_argument(slot, props, used, spec['required'])
                    if arg:
                        entry['slots'][slot] = arg
                        used.add(arg)
                else:
                    continue
            return entry
        else:
            return entry

    @staticmethod
    def slot_argument(slot: str, props: dict, used: set, required: list) -> str | None:

        def kind_fits(prop: dict) -> bool:
            kind = prop.get('type')
            if slot in ('quantity', 'max_price'):
                held_65 = kind in ('integer', 'number')
                return held_65
            elif 'page' == slot:
                held_66 = kind in ('integer', 'number', 'string')
                return held_66
            elif not 'id' == slot:
                held_68 = kind in ('string', None)
                return held_68
            else:
                held_67 = kind in ('string', 'array', 'integer', None)
                return held_67
        free = list((a for a, prop in props.items() if a not in used and kind_fits(prop) and (not prop.get('enum') or ('category' == slot and 4 < len(prop['enum']) and re.search(SLOT_WORDS['category'], a.lower())))))
        named = list((a for a in free if re.search(SLOT_WORDS[slot], a.lower()) and (not re.search(SLOT_NOT.get(slot, '$^'), a.lower()))))
        if not named:
            others = list((a for a in free if not any((re.search(SLOT_WORDS[s], a.lower()) for s in SLOT_WORDS if slot != s))))
            if 'text' == slot and others:
                held_70 = max(others, key=lambda a: props[a].get('maxLength') or 0)
                return held_70
            elif not (slot in ('id', 'quantity') and 1 == len(others) and ('id' != slot or others[0] in required)):
                return None
            else:
                held_71 = others[0]
                return held_71
        else:
            held_69 = named[0]
            return held_69

    def guess_role(self, role: str) -> dict | None:
        best, best_score = (None, 0)
        for name, spec in self.tool_specs.items():
            words = '{} {}'.format(name.replace('_', ' '), spec['description']).lower()
            held_72 = re.findall(ROLE_WORDS[role], words)
            score = len(held_72)
            held_73 = re.search('final|no more actions|ends?\\b', words)
            final = bool(held_73)
            if role in ('checkout', 'decline'):
                if not final:
                    score = score - 1
            if role in ('checkout', 'decline'):
                if final:
                    score = score + 1
            held_74 = role in ('checkout', 'decline')
            if held_74:
                held_74 = bool(self.seen_whole(role, {'tool': name, 'slots': {}, 'fixed': {}, 'cheap': {}})['fixed'])
            moded = held_74
            if 'checkout' == role:
                if re.search(ROLE_WORDS['decline'], words):
                    if not moded:
                        score = score - 3
            if not ('decline' == role and re.search('\\b(buy|pay|purchase|place)\\b', words) and (not re.search(ROLE_WORDS['decline'], words)) and (not moded)):
                if role in ('checkout', 'decline'):
                    if final:
                        if not moded:
                            message_texts = any((p.get('type') == 'string' for p in spec['properties'].values()))
                            score += 1 if message_texts == ('decline' == role) else -1
                if 'view' == role:
                    if re.search(ROLE_WORDS['facts'], words):
                        if not re.search('in full|everything|all about', words):
                            score = score - 2
                if 'cart_view' == role:
                    if spec['required']:
                        score = score - 2
                if role in ('view', 'facts'):
                    if 1 != len(spec['properties']):
                        score = score - 1
                if role in ('cart_add', 'cart_remove'):
                    if not any((p.get('enum') for p in spec['properties'].values())):
                        held_75 = re.search('\\b(add|put|include|insert|place)\\b', words)
                        adds = bool(held_75)
                        held_76 = re.search('\\b(remove|delete|discard|drop|take out|clear)\\b', words)
                        takes = bool(held_76)
                        if not ('cart_add' == role and (takes or not adds) or ('cart_remove' == role and (adds or not takes))):
                            score = score + 2
                        else:
                            continue
                if not 0 >= score:
                    entry = self.seen_whole(role, {'tool': name, 'slots': {}, 'fixed': {}, 'cheap': {}})
                    if self.is_valid(role, entry):
                        if best_score < score:
                            best, best_score = (entry, score)
                else:
                    continue
            else:
                continue
        return best

    def arg_type(self, role: str, slot: str) -> str | None:
        held_77 = self.role_map.get(role)
        if not held_77:
            held_77 = {}
        entry = held_77
        arg = entry.get('slots', {}).get(slot)
        if arg:
            held_78 = (self.tool_specs.get(entry.get('tool') or '', {}).get('properties', {}).get(arg or '', {}) or {}).get('type')
            return held_78
        return None

    def page_limit(self, role: str) -> int:
        held_79 = self.role_map.get(role)
        if not held_79:
            held_79 = {}
        entry = held_79
        prop = self.tool_specs.get(entry.get('tool') or '', {}).get('properties', {}).get(entry.get('slots', {}).get('page') or '', {})
        maximum = prop.get('maximum')
        if not (isinstance(maximum, (int, float)) and 10 < maximum):
            if isinstance(maximum, (int, float)) and 1 <= maximum <= 10:
                held_80 = int(maximum)
                return held_80
            return PAGE_CAP
        else:
            return 10

    def build_call(self, role: str, cheap: bool=False, **values: Any) -> tuple[str, dict] | None:
        entry = self.role_map.get(role)
        if entry:
            props = self.tool_specs[entry['tool']]['properties']
            held_81 = props.get(entry['slots'].get('id') or '')
            if not held_81:
                held_81 = {}
            spec = held_81
            if spec.get('type') == 'array':
                if (spec.get('items') or {}).get('type') == 'object':
                    if 'id' in values:
                        if isinstance(values['id'], list):
                            ids = values['id']
                        else:
                            ids = [] if None is values['id'] else [values['id']]
                        quantity = values.pop('quantity', None)
                        values['id'] = list((self.basket_line(spec['items'], *i) if isinstance(i, tuple) else self.basket_line(spec['items'], i, quantity) for i in ids))
            if 'limit' in entry['slots']:
                if values.get('limit') is None:
                    held_82 = props.get(entry['slots']['limit'])
                    if not held_82:
                        held_82 = {}
                    prop = held_82
                    top = prop.get('maximum')
                    values['limit'] = int(top) if isinstance(top, (int, float)) and 1 <= top <= 100 else None
            args = dict(entry['fixed'])
            if cheap:
                args.update(entry.get('cheap') or {})
            for slot, value in values.items():
                arg = entry['slots'].get(slot)
                if None is not arg:
                    if None is not value:
                        fitted = self.fit_arg(props.get(arg) or {}, value)
                        if not None is fitted:
                            args[arg] = fitted
                        else:
                            return None
            if not any((arg not in args for arg in self.tool_specs[entry['tool']]['required'])):
                held_83 = (entry['tool'], args)
                return held_83
            else:
                return None
        else:
            return None

    @staticmethod
    def basket_line(item: dict, pid: Any, quantity: Any) -> Any:
        out = {}
        for name, prop in (item.get('properties') or {}).items():
            words = name.lower()
            if prop.get('type') in ('integer', 'number') and re.search('count|qty|quant|amount|number|units|how.?many', words):
                out[name] = Toolkit.fit_arg(prop, quantity or 1)
            elif prop.get('type') in (None, 'string'):
                if re.search('id|listing|product|item|sku|ref|code', words):
                    out[name] = Toolkit.fit_arg(prop, pid)
        if out:
            return out
        return pid

    def sets_whole_basket(self, role: str) -> bool:
        entry = self.role_map.get(role)
        held_84 = bool(entry) and re.search('replac\\w*\\s+(?:its|the|all|whole|entire)?\\s*(?:cart|basket|contents)|set the (?:cart|basket)\\b|overwrit\\w*\\s+the (?:cart|basket)', self.tool_specs[entry['tool']]['description'], re.I) is not None
        return held_84

    @staticmethod
    def fit_arg(prop: dict, value: Any) -> Any:
        kind = prop.get('type')
        enum = list((e for e in prop.get('enum') or [] if isinstance(e, str)))
        if enum and isinstance(value, str) and (value not in enum):
            exact = next((e for e in enum if low_text(e) == low_text(value)), None)
            close = max(enum, key=lambda e: len(roots(e) & roots(value)))
            if None is not exact:
                return exact
            if roots(close) & roots(value):
                return close
            return None
        elif 'array' == kind:
            if isinstance(value, list):
                return value
            held_85 = [value]
            return held_85
        elif not 'string' == kind:
            if kind in ('integer', 'number'):
                number = first_number(value)
                if not None is number:
                    if isinstance(prop.get('minimum'), (int, float)):
                        number = max(number, prop['minimum'])
                    if isinstance(prop.get('maximum'), (int, float)):
                        number = min(number, prop['maximum'])
                    if 'integer' == kind or number.is_integer():
                        held_87 = int(number)
                        return held_87
                    return number
                else:
                    return value
            return value
        else:
            text = str(value)
            limit = prop.get('maxLength')
            if isinstance(limit, int):
                held_86 = text[:limit]
                return held_86
            return text
ORDER_PROMPT = "Read what a shopper asks a shop assistant for and set it down field by field, quoting the shopper's own words in each quote.\n- item: the kind of product wanted, in a few words; noun: the single word a catalogue search should use for it.\n- category: the shop category or kind of product the shopper names ('hoodie or sweatshirt', 'fall/winter clothing', 'luxury & formal watches'), worded as they word it.\n- requirements: every property the product itself has to have (material, style, flavour, pack type, colour, size, occasion, ...) with the value as given; exclude=true when the shopper rules a value out ('leave out whatever has ...'). Leave out the category itself, price, stock and quantity rules, and wishes about the listing or the service (that it matches its photo, comes on time).\n- max_price: the top price allowed, as a number in the shop's price unit; -1 when the request gives none.\n- in_stock_only: true when the shopper wants only items the shop has right now.\n- priority: 'cheapest' only when the shopper asks for the lowest-priced item that fits ('the one that costs the least', 'the cheapest'); talk of a tight budget or a price cap is not that. 'other' when they want the best by some other measure (say which in priority_detail, and put just that measure in rank_by in a word or two: volume, length, pieces, power, screen size); otherwise 'none'.\n- worn: true when a person wears or carries the item (clothing, shoes, watches, jewellery, bags); for_whom: who it is for, in the shopper's words, when the request says so (else empty).\n- quantity: how many units to buy (1 unless stated).\n- decline_if_nothing: true when the shopper says to report back instead of buying something else if nothing fits; no_names_in_decline: true when that report may not suggest or name other products.\n- message_budget, step_budget: the stated caps on messages to the shopper and on moves or actions; -1 when none is stated.\n- unstated_preferences: 'yes' when the shopper hints that they, or whoever it is for, have wishes not written down (fussy, won't guess, will check something); 'no' when they say anything that fits will do; else 'unclear'.\n- searches: three to five short catalogue keyword searches of one to four words, the most specific first (the noun with the required values), then broader ones."
ORDER_SHAPE = {'item': {'type': 'string'}, 'noun': {'type': 'string'}, 'category': {'type': 'string'}, 'requirements': {'type': 'array', 'items': {'type': 'object', 'properties': {'property': {'type': 'string'}, 'value': {'type': 'string'}, 'exclude': {'type': 'boolean'}, 'quote': {'type': 'string'}}, 'required': ['property', 'value']}}, 'max_price': {'type': 'number'}, 'in_stock_only': {'type': 'boolean'}, 'priority': {'type': 'string', 'enum': ['cheapest', 'other', 'none']}, 'priority_detail': {'type': 'string'}, 'rank_by': {'type': 'string'}, 'quantity': {'type': 'integer'}, 'decline_if_nothing': {'type': 'boolean'}, 'no_names_in_decline': {'type': 'boolean'}, 'message_budget': {'type': 'integer'}, 'step_budget': {'type': 'integer'}, 'unstated_preferences': {'type': 'string', 'enum': ['yes', 'no', 'unclear']}, 'worn': {'type': 'boolean'}, 'for_whom': {'type': 'string'}, 'searches': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 5}}

class Brief:

    def __init__(self) -> None:
        self.item, self.noun, self.category = ('', '', '')
        self.needs: list[dict] = []
        self.ceiling: float | None = None
        self.stock_only = True
        self.priority, self.rank_words = ('none', '')
        self.units = 1
        self.may_refuse, self.quiet_refusal = (True, True)
        self.message_cap: int | None = None
        self.steps: int | None = None
        self.is_worn, self.wearer = (False, '')
        self.leanings = 'unclear'
        self.seed_searches: list[str] = []
        self.revision = 0
        self.corrections: list[str] = []
        self.measured, self.low_wins = (False, False)
        self.rank_on = ''

    @classmethod
    def read(cls, brain: Oracle, request: str) -> 'Record':
        record = cls()
        answer = brain.ask_model('request_record', ORDER_PROMPT, request, ORDER_SHAPE, ['item', 'noun', 'requirements', 'max_price', 'priority', 'searches'], 1600, critical=True)
        figures = set((float(x.replace(',', '')) for x in re.findall('\\d[\\d,]*(?:\\.\\d+)?', request)))
        if answer:
            record.item = str(answer.get('item') or '').strip()
            record.noun = str(answer.get('noun') or '').strip() or (record.item.split() or [''])[-1]
            record.category = str(answer.get('category') or '').strip()
            for item in answer.get('requirements') or []:
                if isinstance(item, dict):
                    if str(item.get('property') or '').strip():
                        if str(item.get('value') or '').strip():
                            record.add(item['property'], item['value'], bool(item.get('exclude')), str(item.get('quote') or ''), 'request')
            price = answer.get('max_price')
            money = set((float((a or b).replace(',', '')) for a, b in re.findall(MONEY_RE, request, re.I)))
            record.ceiling = next((p for p in ([float(price), round(price * 100, 2), round(price / 100, 4)] if isinstance(price, (int, float)) and 0 < price else []) if p in (money or figures)), None)
            record.stock_only = answer.get('in_stock_only') is not False
            if answer.get('priority') in ('cheapest', 'other', 'none'):
                record.priority = answer.get('priority')
            else:
                record.priority = 'none'
            record.rank_words = str(answer.get('priority_detail') or '')
            record.rank_on = str(answer.get('rank_by') or '').strip()
            quantity = answer.get('quantity')
            spelled = set((n for n, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve'.split()) if re.search('\\b{}\\b'.format(w), request, re.I))) | ({2} if re.search('\\b(pair|couple)\\b', request, re.I) else set())
            if isinstance(quantity, int) and 1 <= quantity <= 50 and (float(quantity) in figures | {1.0} or quantity in spelled):
                record.units = int(quantity)
            else:
                record.units = 1
            record.may_refuse = answer.get('decline_if_nothing') is not False
            record.quiet_refusal = answer.get('no_names_in_decline') is not False
            messages = answer.get('message_budget')
            if isinstance(messages, int) and 0 <= messages and (float(messages) in figures):
                record.message_cap = int(messages)
            else:
                record.message_cap = None
            steps = answer.get('step_budget')
            if isinstance(steps, int) and 0 < steps and (float(steps) in figures):
                record.steps = int(steps)
            else:
                record.steps = None
            if answer.get('unstated_preferences') in ('yes', 'no', 'unclear'):
                record.leanings = answer.get('unstated_preferences')
            else:
                record.leanings = 'unclear'
            record.seed_searches = list((s for s in answer.get('searches') or [] if isinstance(s, str) and s.strip()))[:5]
            record.is_worn = answer.get('worn') is True
            record.wearer = str(answer.get('for_whom') or '').strip()
        record.corrections = []
        for r in list((r for r in record.needs if not r['exclude'] and re.search(TOP_RE, r['value'], re.I) and (not BOUND_RE.search(r['value'])))):
            record.needs.remove(r)
            if 'cheapest' != record.priority:
                record.priority = 'other'
                record.rank_words = record.rank_words or '{} {}'.format(r['value'], r['property'])
            held_88 = '{}: {!r} ranks the items (priority), not a requirement'.format(r['property'], r['value'])
            record.corrections.append(held_88)
        record.seed_searches = list((' '.join((w for w in q.split() if not re.search(TOP_RE + '|\\b(?:among|eligible|options?|possible|available)\\b', w, re.I))) for q in record.seed_searches))
        record.seed_searches = list((q for q in dict.fromkeys(record.seed_searches) if q.strip()))
        for r in record.needs:
            want = set((w for w in flat(r['value']).split() if 3 <= len(w) or w.isdigit()))
            said = list((c for c in re.split('[.!?,;:()]|\\b(?:and|but)\\b', request) if want and want <= word_set(c)))
            if said and (not r['exclude']) and all((re.search(RULE_OUT_RE, s, re.I) for s in said)):
                r['exclude'] = True
                held_89 = '{}: rule out {}'.format(r['property'], r['value'])
                record.corrections.append(held_89)
            elif said and r['exclude'] and all((re.search(INSIST_RE, s, re.I) and (not re.search(RULE_OUT_RE + '|' + NOT_RE, s, re.I)) for s in said)):
                r['exclude'] = False
                held_90 = '{}: require {}'.format(r['property'], r['value'])
                record.corrections.append(held_90)
            elif said:
                if r['exclude']:
                    if not any((re.search(RULE_OUT_RE + '|' + NOT_RE, s, re.I) for s in said)):
                        r['drop'] = True
                        held_91 = '{}: {!r} was never ruled out'.format(r['property'], r['value'])
                        record.corrections.append(held_91)
        record.needs = list((r for r in record.needs if not r.pop('drop', False)))
        for r in record.needs:
            figure = re.search('\\d[\\d,]*(?:\\.\\d+)?', r['value'])
            if not (r['exclude'] or not figure or BOUND_RE.search(r['value'])):
                held_92 = figure.group(0)
                num = re.escape(held_92)
                held_93 = '(?:\\bnot (?:\\w+ )?(?:drop|go|fall|be|dip|come) (?:any )?(?:under|below|lower than|less than|smaller than)|\\bno (?:less|smaller|lower|shorter|fewer) than|\\bat least|\\bminimum(?: of)?|\\bnot (?:under|below))\\s+(?:\\w+\\s+){{0,2}}{}(?!\\d)|(?<!\\d){}\\s*\\S*\\s+(?:or (?:more|larger|bigger|higher|longer|above|over)|and (?:up|above|over))'.format(num, num)
                low = re.search(held_93, request, re.I)
                held_94 = '(?:\\bnot (?:\\w+ )?(?:go|be|exceed|come|run|rise) (?:over|above|more than|beyond|past)|\\bno (?:more|larger|bigger|higher|longer) than|\\bat most|\\bmaximum(?: of)?|\\bup to|\\bnot (?:over|above))\\s+(?:\\w+\\s+){{0,2}}{}(?!\\d)|(?<!\\d){}\\s*\\S*\\s+(?:or (?:less|smaller|lower|under|below|fewer))'.format(num, num)
                high = re.search(held_94, request, re.I)
                if bool(high) != bool(low):
                    r['value'] = ('at least ' if low else 'at most ') + r['value']
                    held_95 = '{}: {} (the request bounds the figure)'.format(r['property'], r['value'])
                    record.corrections.append(held_95)
            else:
                continue
        intent = '\\b(?:want|wish|purchase|buy|need|must|should|only|whose|sort|after|require|has to|have to|prefer|like|would|find|get)\\b'
        for r in list((r for r in record.needs if not r['exclude'])):
            want = set((w for w in flat(r['value']).split() if 3 <= len(w) or w.isdigit()))
            said = list((c for c in re.split('(?<=[.!?])\\s+', request) if want and want <= word_set(c)))
            if said:
                if all((re.search('\\b(?:is|are|was)\\s+(?:an?|the)\\b', c, re.I) and (not re.search(intent, c, re.I)) for c in said)):
                    r['drop'] = True
                    held_96 = '{}: {!r} is only named in a figure of speech'.format(r['property'], r['value'])
                    record.corrections.append(held_96)
        record.needs = list((r for r in record.needs if not r.pop('drop', False)))
        for r in list((r for r in record.needs if only_compared(r['value'], request))):
            record.needs.remove(r)
            held_97 = '{}: {!r} is a comparison in the story, not a property'.format(r['property'], r['value'])
            record.corrections.append(held_97)
        for r in list((r for r in record.needs if re.fullmatch('(?:intended )?(?:use|usage|purpose)', flat(r['property'])) and len(flat(r['value']).split()) >= 4)):
            record.needs.remove(r)
            held_98 = '{}: {!r} is what it is for, not a property'.format(r['property'], r['value'])
            record.corrections.append(held_98)
        if 'cheapest' == record.priority and (not re.search(THRIFT_RE, request, re.I)):
            record.priority = 'none'
            record.corrections.append('priority: none (no words ask for the lowest price)')
        elif 'none' == record.priority:
            if re.search(LOWEST_RE, request, re.I):
                record.priority = 'cheapest'
                record.corrections.append('priority: cheapest')
        if 'other' == record.priority:
            if record.rank_words.strip():
                record.measured = True
                held_100 = low_text(record.rank_words)
                held_99 = re.search('smallest|lowest|least|minimum|shortest|lightest|fewest', held_100)
                record.low_wins = bool(held_99)
        names = {flat(record.item), flat(record.category), flat(record.noun)} - {''}
        record.needs = list((r for r in record.needs if r['exclude'] or flat(r['value']) not in names))
        required = set((flat(r['property']) for r in record.needs if not r['exclude']))
        record.needs = list((r for r in record.needs if not (r['exclude'] and flat(r['property']) in required)))
        if None is record.ceiling:
            held_101 = re.search('(\\d[\\d,]*(?:\\.\\d+)?)\\s*(?:us\\s+|u\\.s\\.\\s+)?(?:cents?|cts?|dollars?|usd|bucks|euros?|pesos?|rupees?|pounds?|yen)\\b', request, re.I)
            if not held_101:
                held_101 = re.search('[$€£¥₱₹]\\s*(\\d[\\d,]*(?:\\.\\d+)?)', request)
            if not held_101:
                held_101 = re.search('(?:spend|budget|afford|over|above|past|more than|up to|under|at most|max(?:imum)?|tops? out at|ceiling)\\D{0,20}?(\\d[\\d,]*(?:\\.\\d+)?)(?!\\s*(?:messages?|questions?|moves?|steps?|actions?|turns?)\\b)', request, re.I)
            found = held_101
            if found:
                record.ceiling = float(found.group(1).replace(',', ''))
            else:
                record.ceiling = None
        if None is record.message_cap:
            found = re.search('(\\d+)\\s+(?:messages?|questions?)', request, re.I)
            if found:
                record.message_cap = int(found.group(1))
            else:
                record.message_cap = None
        if not answer:
            if re.search('cheapest|least|lowest price|lowest-priced', request, re.I):
                record.priority = 'cheapest'
        return record

    def add(self, prop: str, value: str, exclude: bool, quote: str, source: str) -> None:
        if not re.search('\\b(price|cost|budget|spend|cts|stock|availability|quantity|amount to buy|deliver\\w*|arriv\\w*|ship\\w*|dispatch)\\b', flat(prop)):
            if flat(value) in ('true', 'yes', 'required', 'needed', 'must'):
                value = prop
            key = (flat(prop), flat(value), exclude)
            if not (not key[0] or not key[1] or any(((flat(r['property']), flat(r['value']), r['exclude']) == key for r in self.needs))):
                same = list((r for r in self.needs if flat(r['property']) == key[0] and (not r['exclude']) and (not exclude)))
                if not (same and 'request' == source):
                    self.needs = list((r for r in self.needs if r not in same))
                    self.needs.append({'property': prop.strip(), 'value': value.strip(), 'exclude': exclude, 'quote': quote, 'source': source})
                    self.revision = self.revision + 1
                else:
                    if key[1] not in flat(same[0]['value']).split(' or '):
                        same[0]['value'] = '{} or {}'.format(same[0]['value'], value)
                    return
            else:
                return
        else:
            return

    def drop_need(self, prop: str) -> bool:
        held_102 = flat(prop).split()
        words = set(held_102)
        kept = list((r for r in self.needs if not (words and (words <= set(flat(r['property']).split()) or set(flat(r['property']).split()) <= words))))
        changed = len(self.needs) != len(kept)
        self.needs = kept
        self.revision += changed
        return changed

    def lines(self) -> str:
        held_103 = '\n'.join(('{}. {}: {}{}'.format(n, r['property'], 'rule out ' if r['exclude'] else '', r['value']) for n, r in enumerate(self.needs, 1))) or '(none)'
        return held_103

    def values(self) -> list[str]:
        held_104 = list((r['value'] for r in self.needs if not r['exclude']))
        return held_104

class Offer:

    def __init__(self, pid: str, order: int) -> None:
        self.id = pid
        self.order = order
        self.title = ''
        self.price: float | None = None
        self.stock: bool | None = None
        self.category = ''
        self.brand = ''
        self.text = ''
        self.sheet: dict | None = None
        self.log_lines = ''
        self.sold_out = False
        self.via: set[str] = set()

    def as_card(self, number: int) -> str:
        if not None is not self.sheet:
            if self.text:
                extra = ' | about: {}'.format(self.text[:160])
            else:
                extra = ''
            held_107 = '{}. DATASHEET: (not read) || title: {} | category: {} | brand: {}{}'.format(number, self.title[:220], self.category or '-', self.brand or '-', extra)
            return held_107
        else:
            held_105 = '; '.join(('{}: {}'.format(k, str(v)[:150]) for k, v in self.sheet.items()))[:1600]
            if not held_105:
                held_105 = '(empty)'
            facts = held_105
            if self.log_lines:
                notes = ' | page notes: {}'.format(self.log_lines[:400])
            else:
                notes = ''
            held_106 = '{}. DATASHEET: {} || title: {} | category: {} | brand: {}{}'.format(number, facts, self.title[:220], self.category or '-', self.brand or '-', notes)
            return held_106

    def words(self) -> str:
        held_108 = flat(' '.join([self.title, self.category, self.brand, self.text, self.sheet_words()]))
        return held_108

    def sheet_words(self) -> str:
        held_109 = flat(' '.join(('{} {}'.format(k, v) for k, v in (self.sheet or {}).items())))
        return held_109
COLUMNS = {'id': '(^|[^a-z])(id|pid|sku)$|_id$|^id|pid', 'title': 'title|name|head|caption|label', 'price': 'price|cost|amount|fee', 'stock': 'stock|avail|on_?hand', 'category': 'categ|depart|dept|section|aisle', 'brand': 'brand|make|company|manufact|vendor|producer', 'text': 'descri|about|summary|details'}
ROW_AVOID = {'title': 'brand|maker|company|categ|seller|shop|store', 'price': 'ship|deliver|tax|old|orig|list|was|before|msrp|compare|fee|discount', 'category': 'id$', 'text': 'short_?desc$'}
ROW_PREFER = {'id': '^(id|pid|item_?id|product_?id|item_?pid|sku)$', 'title': '^(title|name|product_?name|item_?name)$', 'price': '^(price|sale_?price|current_?price)$'}

def guess_columns(row: dict) -> dict[str, str | None]:
    keys = list(row)
    tests = {'id': lambda v: isinstance(v, (str, int)) and (not isinstance(v, bool)) and (str(v).strip() != ''), 'title': lambda v: isinstance(v, str), 'price': lambda v: first_number(v) is not None, 'stock': lambda v: stock_of(v) is not None, 'category': lambda v: isinstance(v, str) or None is v, 'brand': lambda v: isinstance(v, str) or None is v, 'text': lambda v: isinstance(v, str) or None is v}
    out: dict[str, str | None] = {}
    for field, pattern in COLUMNS.items():
        fits = list((k for k in keys if re.search(pattern, k.lower()) and tests[field](row[k]) and (not re.search(ROW_AVOID.get(field, '$^'), k.lower()))))
        held_110 = fits
        if not held_110:
            held_110 = list((k for k in keys if re.search(pattern, k.lower()) and tests[field](row[k])))
        fits = held_110
        preferred = list((k for k in fits if re.search(ROW_PREFER.get(field, '$^'), k.lower())))
        out[field] = (preferred or fits or [None])[0]
    if None is out['price']:
        numbers = list((k for k in keys if isinstance(row[k], (int, float)) and (not isinstance(row[k], bool))))
        out['price'] = numbers[0] if 1 == len(numbers) else None
    if None is out['stock']:
        flags = list((k for k in keys if isinstance(row[k], bool)))
        out['stock'] = flags[0] if 1 == len(flags) else None
    return out

def result_rows(obs: Any) -> list[dict]:
    best: list[dict] = []

    def visit(node: Any, depth: int) -> None:
        nonlocal best
        if 3 < depth:
            return
        elif isinstance(node, list):
            if node:
                if all((isinstance(x, dict) for x in node)):
                    if len(best) < len(node):
                        if 3 <= len(node[0]):
                            best = node
        elif isinstance(node, dict):
            for value in node.values():
                visit(value, depth + 1)
    visit(obs, 0)
    return best

def match_count(obs: Any) -> int | None:
    if isinstance(obs, dict):
        for key, value in obs.items():
            if isinstance(value, int):
                if not isinstance(value, bool):
                    if re.search('match|count|total|found|hits|num|results', key.lower()):
                        if not re.search('page|per|size|limit|price', key.lower()):
                            return value
    return None

def has_more(obs: Any) -> bool | None:
    if isinstance(obs, dict):
        for key, value in obs.items():
            if re.search('next|continu|more|cursor|further|after|onward|forward|resume|bookmark', key.lower()):
                held_111 = bool(value)
                return held_111
    return None

def cursor_of(obs: Any) -> Any:
    if isinstance(obs, dict):
        for key, value in obs.items():
            if re.search('next|continu|cursor|further|after|token|onward|forward|resume|bookmark', key.lower()):
                if isinstance(value, (str, int)):
                    if not isinstance(value, bool):
                        if str(value).strip():
                            return value
    return None

def code_of(obs: Any) -> Any:
    if isinstance(obs, dict):
        for key, value in obs.items():
            if re.search('token|quote|reserv|confirm|ticket|draft|handle|code$|_code|^key$|_key$|ref$|_ref', str(key).lower()):
                if isinstance(value, (str, int)):
                    if not isinstance(value, bool):
                        if str(value).strip():
                            return value
        for value in obs.values():
            found = code_of(value)
            if None is not found:
                return found
    return None

def sheet_of(obs: Any) -> dict:
    if isinstance(obs, dict):
        for value in obs.values():
            if isinstance(value, list):
                if value:
                    if all((isinstance(x, dict) for x in value)):
                        if not (any((isinstance(v, dict) for v in value[0].values())) or 5 <= len(value[0])):
                            pairs = {}
                            for x in value:
                                label = next((x[k] for k in x if re.search('name|key|label|attr|prop|field', k.lower()) and isinstance(x[k], str)), None)
                                entry = next((x[k] for k in x if re.search('value|val$|text|data', k.lower())), None)
                                if label:
                                    if entry not in (None, ''):
                                        pairs[label] = ', '.join(map(str, entry)) if isinstance(entry, list) else entry
                            if max(1, len(value) // 2) <= len(pairs):
                                return pairs
                        else:
                            held_113 = sheet_of(value[0])
                            return held_113
        nested = list((v for v in obs.values() if isinstance(v, dict) and v))
        if nested:
            source = max(nested, key=len)
        else:
            source = dict(((k, v) for k, v in obs.items() if not re.search('id$|pid|title|name', k.lower())))
        out = {}
        for key, value in source.items():
            if isinstance(value, (list, tuple)):
                value = ', '.join((str(x) for x in value))
            if isinstance(value, (str, int, float, bool)):
                if str(value).strip():
                    out[str(key)] = value
        return out
    else:
        held_112 = {}
        return held_112
LINES_PROMPT = "Some requests ask for several different products in a single order. List every separate product the request wants bought, in the order the request gives them:\n- item: what kind of product, in a few words (use the shop's own category name when the shopper does); noun: the single word a catalogue search should use for it; category: the shop category the shopper names for it.\n- quantity: how many units of it ('3 tablets' = 3, 'a pair of shoes' = 1; 'each' puts the count on every unit).\n- requirements: every property this product has to have, with the value as given, keeping the words that bound a figure ('at least 4 drawers' -> drawers: at least 4); exclude=true when the shopper rules a value out. Only what the request says about this product.\n- searches: two or three short catalogue searches of one to four words for it, the most specific first.\nSeveral units of the same product are one entry with a quantity, not several entries. A product the shopper already owns or only mentions along the way is not one to buy."
LINES_SHAPE = {'products': {'type': 'array', 'items': {'type': 'object', 'properties': {'item': {'type': 'string'}, 'noun': {'type': 'string'}, 'category': {'type': 'string'}, 'quantity': {'type': 'integer'}, 'requirements': {'type': 'array', 'items': {'type': 'object', 'properties': {'property': {'type': 'string'}, 'value': {'type': 'string'}, 'exclude': {'type': 'boolean'}}, 'required': ['property', 'value']}}, 'searches': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['item', 'quantity', 'requirements']}}}
LINES_RE = '\\b(?:\\d+|two|three|four|five|six)\\s+(?:x\\s+|pairs? of\\s+)?(?!(?:cents?|messages?|tool|calls?|steps?|moves?|actions?|cm|mm|inch|inches|gb|tb|mah|hz|drawers?|weeks?|days?|hours?|minutes?|years?|months?|miles?|boys?|girls?|kids?|children|times?|percent|us\\b|uk\\b|eu\\b)\\b)[a-z]'
LOOKUP_RE = 'shown on the page|on the page for|page for the|wherever that page|\\bsold by\\b|\\bthe one (?:shown|on|sold)\\b|\\bsame as the \\w+ (?:shown|on|of)'
LOOKUP_PROMPT = "The shopper wants one property copied from a different listing in the shop (for instance a kettle holding as much as the toaster one maker sells in red). Set down:\n- wanted: the property whose value is copied, named as the shopper names it;\n- relation: 'same' when the value has to match, 'within' when it may differ by some amount ('within 3 count above or below');\n- amount: for 'within', the allowed difference as a number; unit: the unit word that goes with it;\n- brand: the seller or maker of that other listing (the name after 'sold by', or the name in front of its category);\n- category: that other listing's shop category;\n- clues: the properties the shopper gives to single that listing out, each with its value."
LOOKUP_SHAPE = {'wanted': {'type': 'string'}, 'relation': {'type': 'string', 'enum': ['same', 'within']}, 'amount': {'type': 'number'}, 'unit': {'type': 'string'}, 'brand': {'type': 'string'}, 'category': {'type': 'string'}, 'clues': {'type': 'array', 'items': {'type': 'object', 'properties': {'property': {'type': 'string'}, 'value': {'type': 'string'}}}}}
CHECK_PROMPT = "Check shop listings against the shopper's numbered requirements, reading every part of a listing (title, category, datasheet, description). For each listing and requirement answer:\n- yes: the listing states the required value; quote the exact words that show it;\n- no: the listing states another value for that property, not the required one; quote those words;\n- unknown: the listing does not say.\nPage notes weigh like the datasheet; where a note says a property is to be judged on another product, this listing does not state that property (unknown). Give where your quote comes from ('datasheet' or 'title'), and set mixed=true when one part of the listing states the required value while another states a different value that rules it out (title lace but datasheet chiffon; title baggy but datasheet fit: loose). A datasheet entry listing several values with the required one among them ('rectangle, square') states the required value and is not mixed. Two datasheet entries for one property that disagree (material: chiffon next to material: net yarn) are mixed; entries about different aspects sharing a word (skirt style: wrap next to style: casual) are not. Go by the shop's own value words: a different word for a close idea (loose for baggy, round for square) is a different value; plural or compound forms of the same word count ('wrap skirts' for wrap). A requirement marked 'rule out' is met (yes) when the listing lacks that value and broken (no) when it has it. 'A or B' in the shopper's words accepts either one."
CHECK_SHAPE = {'listings': {'type': 'array', 'items': {'type': 'object', 'properties': {'n': {'type': 'integer'}, 'checks': {'type': 'array', 'items': {'type': 'object', 'properties': {'r': {'type': 'integer'}, 'verdict': {'type': 'string', 'enum': ['yes', 'no', 'unknown']}, 'quote': {'type': 'string'}, 'source': {'type': 'string', 'enum': ['datasheet', 'title', 'none']}, 'mixed': {'type': 'boolean'}}, 'required': ['r', 'verdict', 'quote']}}}, 'required': ['n', 'checks']}}}
LISTEN_PROMPT = "Keep a shop assistant's note of the shopper's wishes current. New messages have come in (the shopper answering, a note from the shop, or the shop's reason for turning an order down). Write down only what they change:\n- new_requirements: a property the product now has to have (or must not have: exclude=true), with the value as given;\n- dropped_properties: properties the shopper no longer cares about ('never mind about ...'), named as the note names them when they are in it;\n- max_price: the new top price when the shopper changes the budget, else -1;\n- carted_unavailable: true when a message says the item in the cart is out of stock, sold out or cannot be bought (its price rising or falling is not that);\nAn answer that only says the shopper has no preference changes nothing. Read which way a message points with care:\n- 'I won't take anything but stripe when it comes to pattern', 'X coming in as Y isn't up for trade', 'I'm only taking Y' = a requirement X: Y (exclude=false);\n- 'leave out whatever has Y', 'no Y please' = exclude=true;\n- 'never mind about X', 'forget X', 'X was more of a wish than a need', 'X doesn't matter now' = X goes into dropped_properties (it is no requirement)."
LISTEN_SHAPE = {'new_requirements': {'type': 'array', 'items': {'type': 'object', 'properties': {'property': {'type': 'string'}, 'value': {'type': 'string'}, 'exclude': {'type': 'boolean'}}, 'required': ['property', 'value']}}, 'dropped_properties': {'type': 'array', 'items': {'type': 'string'}}, 'max_price': {'type': 'number'}, 'carted_unavailable': {'type': 'boolean'}}
MUST_PROMPT = "A shop assistant noted a requirement from a shopper's request, yet no product of the wanted kind meets it. Answer demanded=true when the request puts it as something the product has to have or be (however worded), false when it is only background: the shopper's story, a season or event they mention, a question they ask themselves, or how they will use the item."
MEASURE_PROMPT = 'The shopper ranks the items that fit by one measure (a volume, a length, a count of pieces, a power, a screen size, ...). For every listing give that measure as a single number in one unit shared by all listings, read from its datasheet or title (when several sizes are offered, the one the listing is sold as). Leave the value out when the listing does not give it.'
MEASURE_SHAPE = {'figures': {'type': 'array', 'items': {'type': 'object', 'properties': {'n': {'type': 'integer'}, 'value': {'type': 'number'}}, 'required': ['n']}}}
QUESTION_PROMPT = "Before buying, a shop assistant may ask the shopper a few questions, one property each. The shopper keeps back a firm value for some property of the item and reveals it only to a question that names exactly that property. Pick the properties most likely to hide such a value, likeliest first:\n1. what the shopper's reasons for buying make matter for this item: stocking up for a while points to how many pieces one purchase holds and how much one holds or weighs; parcels that get weighed point to piece count and weight; a space to fill or fit into points to length and volume; a charger, sockets or a power source it has to work with point to voltage, then power; a room it runs in points to power; looks (photos, matching, clashing) point to pattern and colour; feel, tags or care point to material; a gift for someone with particular taste points to colour, then material; a gift for someone the shopper hardly knows points to who it is for (gender);\n2. next, among the properties the shop's datasheets record for this item and on which its listings differ, first the one that tells this kind of product's versions apart in use: a size, length or capacity figure; what it has to fit or work with (compatibility, memory type, form factor); the spec a user sees or feels in use (screen resolution, display size, refresh rate, the shape or finish of a shoe's toe, closure, cut); its flavour, roast or pack for food and drink. Its material only when the request points to feel, care or what it is made of. Prefer what a shop assistant would ask first for this kind of product and what most of its listings record; a property found on only one or two listings seldom. A property whose values the listings' own titles name is one sellers use to tell versions apart and is likelier kept back than one only the datasheets record. Colour, pattern and who wears it are rarely kept back.\nNever: a property the request already settles, price, stock, the category, the maker, the seller or the offer (warranty, condition, shipping), the measure the shopper ranks by. Name each property in one to three plain words the way a shopper would (voltage, not input voltage; material, not jug material), a bare 'size' only when the datasheets use that very label, and never two properties joined by 'and' or 'or'."
QUESTION_SHAPE = {'properties': {'type': 'array', 'items': {'type': 'string'}}}
COLUMN_PROMPT = 'For each field, name the key of this product record that holds it: id (the product id other tools use), title, price, stock (in stock or not), category, brand, text (a description). Leave a field empty when no key holds it.'

def roots(text: Any) -> set[str]:
    out = set()
    for word in flat(text).split():
        if not (3 > len(word) or word in FILLER):
            out.add(word[:5] if 5 < len(word) else word.rstrip('s'))
        else:
            continue
    return out

class Basics:

    def __init__(self, problem_data: dict) -> None:
        self.timer = Timer()
        self.oracle = Oracle(self.timer)
        self.session = Session(problem_data.get('environment') or {})
        self.kit = Toolkit(self.session.tool_list)
        self.log_lines: list[str] = []
        self.reset_state()

    def reset_state(self) -> None:
        self.brief = Brief()
        self.stop_step: int | None = None
        self.as_line = False
        self.fellow_lines: list['Agent'] = []
        self.lines: list[dict] = []
        self.offers: dict[str, Offer] = {}
        self.columns: dict[str, str | None] | None = None
        self.tried: set[str] = set()
        self.seen_whole: set[str] = set()
        self.category = ''
        self.checks: dict[tuple[str, str], tuple[str, str, bool]] = {}
        self.basket_now: dict[str, int] = {}
        self.heard_before: set[str] = set()
        self.questions_done = False
        self.search_log: dict[str, tuple] = {}
        self.cursors: dict[str, Any] = {}
        self.shelf_ok: bool | None = None
        self.named_shelf = ''
        self.no_more: set[str] = set()
        self.first_seen_whole = False
        self.rechecked_once = False
        self.basket_doubt = False
        self.rank_cache: dict[tuple, list[str]] = {}
        self.must_cache: dict[str, bool] = {}
        self.measure_cache: dict[tuple, float | None] = {}
        self.cannot_remove = False
        self.widened_once = False
        self.live_cache: tuple | None = None
        self.loose_yes: set[tuple[str, str]] = set()
        self.sheet_reads = 0
        self.rows_per_page = 0

    def jot(self, text: str) -> None:
        held_114 = '[s{} {:.0f}s] {}'.format(self.session.step, self.timer.elapsed(), text)
        self.log_lines.append(held_114)

    def has_room(self, steps: int) -> bool:
        if self.as_line:
            steps = max(0, steps - CLOSING_STEPS)
        held_115 = self.session.steps_left() > steps and (not self.session.done) and (None is self.stop_step or self.stop_step > self.session.step + steps)
        return held_115

    def call(self, tool: str, args: dict) -> StepResult:
        reply = self.session.call(tool, args)
        self.session.inbox_msgs = []
        fresh = self.unheard(reply.spoken)
        if fresh:
            held_116 = 'messages: {}'.format(fresh)
            self.jot(held_116)
            self.listen(fresh, 'The shopper wrote:')
        return reply

    def unheard(self, messages: list[str]) -> list[str]:
        out = []
        for message in messages:
            key = flat(message)
            if key:
                if key not in self.heard_before:
                    self.heard_before.add(key)
                    out.append(message)
        return out

    def run(self) -> None:
        if not self.session.done:
            self.prepare()
            if not (self.brief.item or self.brief.noun or self.brief.needs):
                self.jot('no record of the request: declining rather than buying blind')
                self.refuse_order()
                return
            elif not self.lines:
                self.resolve_lookups()
                self.discover()
                self.interview()
                self.confirm_pool()
                self.close_out()
            else:
                self.shop_lines()
                return
        else:
            return

class Catalog:

    def prepare(self) -> None:
        for line in self.kit.assign_roles(self.oracle):
            self.jot(line)
        self.jot('roles: ' + json.dumps(self.kit.role_map, ensure_ascii=False)[:900])
        request = self.session.request_text + ('\n\n' + json.dumps(self.session.first_view, ensure_ascii=False) if None is not self.session.first_view else '')
        self.brief = Brief.read(self.oracle, request)
        if not (self.brief.item or self.brief.noun):
            if self.timer.elapsed() < 90:
                self.jot('request record empty: reading it once more')
                time.sleep(20)
                self.brief = Brief.read(self.oracle, request)
        r = self.brief
        if r.steps:
            if self.session.step_cap > r.steps:
                self.session.step_cap = r.steps
        self.jot('record: item={!r} noun={!r} category={!r} reqs={!r} max={} stock={} priority={} qty={} messages={} prefs={} searches={}'.format(r.item, r.noun, r.category, r.lines(), r.ceiling, r.stock_only, r.priority, r.units, r.message_cap, r.leanings, r.seed_searches) + (" fixed by the request's words: {}".format(r.corrections) if r.corrections else ''))
        if len(re.findall(LINES_RE, request, re.I)) >= 2 or re.search('\\b(?:plus|along with|as well as|together with)\\s+(?:\\d+|an?|one|two|three)\\b|\\b(?:two|three|four) (?:things|separate items|items)\\b', request, re.I):
            held_117 = self.oracle.ask_model('basket', LINES_PROMPT, request, LINES_SHAPE, ['products'], 1500)
            if not held_117:
                held_117 = {}
            answer = held_117
            products, seen = ([], set())
            for x in answer.get('products') or []:
                if not (not isinstance(x, dict) or not str(x.get('item') or '').strip()):
                    key = (flat(x.get('item')), frozenset((flat(q.get('property')) for q in x.get('requirements') or [] if isinstance(q, dict))))
                    if key not in seen:
                        seen.add(key)
                        products.append(x)
                else:
                    continue
            if len(set((flat(x.get('item')) for x in products))) >= 2:
                self.lines = products
                held_119 = list(((x.get('item'), x.get('quantity'), list(((q.get('property'), q.get('value')) for q in x.get('requirements') or [] if isinstance(q, dict)))) for x in products))
                held_118 = 'basket: {}'.format(held_119)
                self.jot(held_118)

    def absorb(self, obs: Any) -> list[Offer]:
        rows = result_rows(obs)
        if rows:
            if None is self.columns or not self.columns.get('id') or self.columns['id'] not in rows[0]:
                self.columns = guess_columns(rows[0])
                if not self.columns.get('id') or not self.columns.get('price'):
                    self.columns = self.learn_columns(rows[0]) or self.columns
                held_121 = 'row schema: {}'.format(self.columns)
                self.jot(held_121)
            out = []
            for row in rows:
                listing = self.offer_from_row(row)
                if listing:
                    out.append(listing)
            return out
        else:
            held_120 = []
            return held_120

    def learn_columns(self, row: dict) -> dict | None:
        fields = dict(((f, {'type': 'string'}) for f in COLUMNS))
        answer = self.oracle.ask_model('record_keys', COLUMN_PROMPT, json.dumps(row, ensure_ascii=False)[:1500], fields, ['id'], 300)
        if answer:
            held_122 = dict(((f, answer.get(f) if answer.get(f) in row else None) for f in COLUMNS))
            return held_122
        else:
            return None

    def offer_from_row(self, row: dict) -> Offer | None:
        held_123 = self.columns
        if not held_123:
            held_123 = {}
        schema = held_123
        pid = row.get(schema.get('id') or '')
        if not (None is pid or str(pid).strip() == ''):
            pid = str(pid)
            listing = self.offers.get(pid)
            if None is listing:
                held_124 = len(self.offers)
                listing = self.offers[pid] = Offer(pid, held_124)
            for field in ('title', 'category', 'brand', 'text'):
                key = schema.get(field)
                if key:
                    if isinstance(row.get(key), str):
                        held_125 = row[key].strip()
                        setattr(listing, field, held_125)
            self.update_price_stock(listing, row)
            return listing
        else:
            return None

    def update_price_stock(self, listing: Offer, record: Any) -> None:
        if not (not isinstance(record, dict) or not self.columns):
            key = self.columns.get('price')
            if key:
                if key in record:
                    if first_number(record[key]) is not None:
                        listing.price = first_number(record[key])
            key = self.columns.get('stock')
            if key:
                if key in record:
                    if stock_of(record[key]) is not None:
                        listing.stock = stock_of(record[key])
        else:
            return

    def within_reach(self, listing: Offer) -> bool:
        if listing.sold_out or (self.brief.stock_only and False is listing.stock):
            return False
        elif None is not self.brief.ceiling and None is not listing.price and (self.brief.ceiling + 1e-09 < listing.price * self.brief.units):
            return False
        elif not (self.category and listing.category and (self.category != listing.category)):
            return True
        else:
            return False

    def has_compare_tool(self) -> bool:
        held_126 = self.kit.arg_type('facts', 'id') == 'array'
        return held_126

    def open_sheets(self, listings: list[Offer]) -> None:
        listings = list((l for l in listings if None is l.sheet))
        if 1 < len(listings):
            if self.has_compare_tool():
                if self.has_room(self.holdback()):
                    action = self.kit.build_call('facts', id=list((l.id for l in listings)))
                    if action:
                        reply = self.call(*action)
                    else:
                        reply = None
                    items = list((x for x in (result_rows(reply.seen) if reply else []) if isinstance(x, dict)))
                    held_127 = (self.columns or {}).get('id')
                    if not held_127:
                        held_127 = 'id'
                    key = held_127
                    by_id = dict(((str(x.get(key) or x.get('id')), x) for x in items))
                    skip = set((v for v in (self.columns or {}).values() if v))
                    for listing in listings:
                        page = by_id.get(listing.id)
                        if not None is page:
                            listing.sheet = sheet_of(page) or {}
                            listing.log_lines = ' | '.join(('{}: {}'.format(k, v.strip()) for k, v in page.items() if isinstance(v, str) and len(v.strip()) > 25 and (k not in skip) and ('id' != k)))[:600]
                            self.update_price_stock(listing, page)
                        else:
                            continue
                    self.sheet_reads = self.sheet_reads + 1
                    if not by_id:
                        held_128 = 'side-by-side read gave no pages: {}'.format(str(reply.seen)[:120] if reply else '')
                        self.jot(held_128)
                    else:
                        return
        for listing in listings:
            if None is listing.sheet:
                if self.has_room(self.holdback()):
                    self.open_sheet(listing)

    def open_sheet(self, listing: Offer) -> None:
        held_129 = self.kit.arg_type('facts', 'id') == 'array'
        if held_129:
            held_129 = 'view' in self.kit.role_map
        listed = held_129
        if listed:
            held_130 = None
        else:
            held_130 = self.kit.build_call('facts', id=listing.id)
        if not held_130:
            held_130 = self.kit.build_call('view', id=listing.id)
        if not held_130:
            held_130 = self.kit.build_call('facts', id=listing.id)
        action = held_130
        if not (not action or not self.has_room(CLOSING_STEPS + 1)):
            reply = self.call(*action)
            self.sheet_reads = self.sheet_reads + 1
            if is_failure(reply.seen):
                held_131 = 'datasheet of {} failed: {}'.format(listing.id, str(reply.seen)[:120])
                self.jot(held_131)
                return
            elif not isinstance(reply.seen, dict):
                listing.sheet = {}
            else:
                listing.sheet = sheet_of(reply.seen) or {}
                skip = set((v for v in (self.columns or {}).values() if v))
                listing.log_lines = ' | '.join(('{}: {}'.format(k, v.strip()) for k, v in reply.seen.items() if isinstance(v, str) and len(v.strip()) > 25 and (k not in skip) and ('id' != k)))[:600]
                self.update_price_stock(listing, reply.seen)
                key = (self.columns or {}).get('category')
                if key:
                    if isinstance(reply.seen.get(key), str):
                        if reply.seen[key].strip():
                            listing.category = reply.seen[key].strip()
        else:
            return

class Finder:

    def settle_shelf(self) -> None:
        held_132 = ' '.join([self.brief.category, self.brief.item, self.brief.noun])
        wanted = roots(held_132)
        counts = Counter((l.category for l in self.offers.values() if l.category))
        if not (not counts or not wanted):
            noun = roots(self.brief.noun)
            named = roots(self.brief.category)
            best = max(counts, key=lambda c: (bool(roots(c)) and roots(c) <= named, bool(roots(c) & noun), len(roots(c) & wanted), counts[c]))
            if roots(best) & wanted:
                if self.category != best:
                    held_135 = counts.most_common(5)
                    held_134 = dict(held_135)
                    held_133 = 'category: {!r} (rows by category: {})'.format(best, held_134)
                    self.jot(held_133)
                    self.category = best
        else:
            return

    def search(self, words: list[str], everything: bool=False) -> int | None:
        entry = self.kit.role_map.get('search')
        if entry:
            limit = self.kit.tool_specs[entry['tool']]['properties'].get(entry['slots']['text'], {}).get('maxLength')
            text = search_text(words, limit)
            key = search_key(text)
            if not (not text or key in self.tried or (not self.has_room(self.holdback()))):
                self.tried.add(key)
                held_136 = self.read_pages(text, key, 1, everything)
                return held_136
            else:
                return None
        else:
            return None

    def read_pages(self, text: str, key: str, page: int, everything: bool) -> int | None:
        entry = self.kit.role_map['search']
        cheap = 'cheapest' == self.brief.priority
        held_137 = cheap
        if held_137:
            held_138 = entry.get('cheap')
            held_137 = bool(held_138)
        sorted_cheap = held_137
        count, seen = self.search_log.get(key, (None, 0, 0, text))[:2]
        if not (1 < page and key in self.no_more):
            while self.has_room(self.holdback()):
                if 1 < page:
                    token = self.cursors.get(key)
                else:
                    token = None
                held_139 = self.category
                if not held_139:
                    held_139 = self.named_shelf
                shelf = held_139
                if False is not self.shelf_ok and shelf and ('category' in entry['slots']):
                    scope = shelf
                else:
                    scope = None
                if 1 < page:
                    where = token if None is not token else page
                else:
                    where = None
                if scope:
                    held_140 = self.kit.build_call('search', category=scope, page=where, text=text, cheap=cheap)
                else:
                    held_140 = None
                if not held_140:
                    held_140 = self.kit.build_call('search', page=where, text=text, cheap=cheap)
                action = held_140
                if action:
                    reply = self.call(*action)
                    rows = self.absorb(reply.seen)
                    if scope:
                        if None is self.shelf_ok:
                            if entry['slots']['category'] in action[1]:
                                if not (rows and (not is_failure(reply.seen))):
                                    self.shelf_ok = False
                                    held_141 = 'search {!r} within {!r} gave nothing: plain searches from now on'.format(text, scope)
                                    self.jot(held_141)
                                    continue
                                else:
                                    self.shelf_ok = True
                    for listing in rows:
                        listing.via.add(text)
                    if not (is_failure(reply.seen) and (not rows)):
                        if None is count:
                            count = match_count(reply.seen)
                        else:
                            count = count
                        self.cursors[key] = cursor_of(reply.seen)
                        if has_more(reply.seen) is False:
                            self.no_more.add(key)
                        seen += len(rows)
                        self.search_log[key] = (count, seen, page, text)
                        if has_more(reply.seen) or (None is not count and len(rows) < count):
                            self.rows_per_page = max(self.rows_per_page, len(rows))
                        held_144 = len(rows)
                        held_143 = 'search {!r} p{}: {} rows, {} matches'.format(text, page, held_144, count)
                        self.jot(held_143)
                        if not (not rows or has_more(reply.seen) is False or page >= self.kit.page_limit('search') or ('page' not in entry['slots'])):
                            if sorted_cheap or not (everything or self.too_few()):
                                break
                            page = page + 1
                        else:
                            break
                    else:
                        held_142 = 'search {!r} p{}: failed {}'.format(text, page, str(reply.seen)[:160])
                        self.jot(held_142)
                        break
                else:
                    break
            if None is not count:
                if seen >= count:
                    self.seen_whole.add(key)
            return count
        else:
            return count

    def shelf_sweep(self, cap: float | None, whole_only: bool=False, near: float=1.0) -> int | None:
        entry = self.kit.role_map.get('filter')
        if not (not entry or not self.category or (not self.has_room(self.holdback()))):
            key = 'filter {} {}'.format(self.category, cap)
            if not key in self.tried:
                self.tried.add(key)
                count, page, token = (None, 1, None)
                while self.has_room(self.holdback()):
                    action = self.kit.build_call('filter', cheap='cheapest' == self.brief.priority, category=self.category, max_price=cap, page=(token if None is not token else page) if 1 < page else None)
                    if action:
                        reply = self.call(*action)
                        rows = self.absorb(reply.seen)
                        if None is count:
                            count = match_count(reply.seen)
                        else:
                            count = count
                        token = cursor_of(reply.seen)
                        if has_more(reply.seen) or (None is not count and len(rows) < count):
                            self.rows_per_page = max(self.rows_per_page, len(rows))
                        held_145 = 'sweep {!r} <= {}: p{} {} rows, {} matches'.format(self.category, cap, page, len(rows), count)
                        self.jot(held_145)
                        if not (not rows or has_more(reply.seen) is False or page >= self.kit.page_limit('filter') or ('page' not in entry['slots'])):
                            held_146 = None is not count
                            if held_146:
                                held_146 = count <= near * self.rows_per_page * self.kit.page_limit('filter')
                            whole = held_146
                            if whole or (not whole_only and self.too_few()):
                                page = page + 1
                            else:
                                break
                        else:
                            break
                    else:
                        break
                return count
            else:
                return None
        else:
            return None

    def value_terms(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in self.brief.needs:
            if not (r['exclude'] or BOUND_RE.search(r['value'])):
                own = set((w.rstrip('s') for w in flat(r['property']).split()))
                options = []
                first = re.split('[,;:]', re.sub('\\([^)]*\\)', ' ', r['value']))[0]
                for option in re.split('\\s+or\\s+|/', first):
                    kept = ' '.join((w for w in word_tokens(option) if w and (len(word_tokens(option)) < 3 or w.rstrip('s') not in own)))
                    if len(kept.split()) > 1:
                        held_147 = ' '.join((w for w in kept.split() if w not in ('family', 'type', 'kind')))
                        if not held_147:
                            held_147 = kept
                        kept = held_147
                    if kept.strip() or option.strip():
                        options.append(kept.strip() or option.strip())
                if options:
                    out[r['value']] = options
            else:
                continue
        return out

    def names_all(self, words: list[str]) -> bool:
        held_148 = ' '.join(words)
        said = roots(held_148)
        held_149 = all((any((roots(o) and roots(o) <= said for o in options)) for options in self.value_terms().values()))
        return held_149

    def search_plan(self) -> list[list[str]]:
        r = self.brief
        noun = [r.noun or r.item]
        options = self.value_terms()
        values = list((opts[0] for opts in options.values()))
        words = list((w for v in values if len(flat(v).split()) > 1 for w in flat(v).split() if w not in FILLER and 2 < len(w)))
        heads = list((self.leading_words(v) for v in values if self.leading_words(v)))
        led = list((h[:1] + noun + h[1:] for h in heads))
        if heads:
            out = [heads[0][:1] + noun + list((w for h in heads[1:] for w in h[:1])) + list((w for h in heads for w in h[1:]))]
        else:
            out = []
        if heads:
            if self.category or self.named_shelf:
                if False is not self.shelf_ok:
                    out = [list((w for h in heads for w in h[:2]))[:3]] + out
        out += led + ([list((h[0] for h in heads[:2]))] if 1 < len(heads) else [])
        out += [noun + values] + list((noun + [o] for opts in options.values() for o in opts[1:]))
        out += list((noun + [w] for w in words)) + list(([s] for s in r.seed_searches[:3])) + [noun]
        seen, unique = (set(), [])
        for q in out:
            held_150 = search_text(q, None)
            key = search_key(held_150)
            if key:
                if key not in seen:
                    seen.add(key)
                    unique.append(q)
        return unique

    @staticmethod
    def leading_words(value: str) -> list[str]:
        ws = list((w for w in word_tokens(value) if w not in FILLER and (1 < len(w) or w.isdigit())))
        held_151 = sorted(ws, key=lambda w: (not any((c.isdigit() for c in w)), -len(w)))
        return held_151

    def split_plan(self) -> list[list[str]]:
        noun = [self.brief.noun or self.brief.item]
        values = list((o[0] for o in self.value_terms().values()))
        base = noun + values
        if not (not values or search_key(search_text(base, None)) not in self.search_log):
            count, seen = self.search_log[search_key(search_text(base, None))][:2]
            if not (not count or seen >= count):
                held_154 = search_text(base, None).split()
                have = set(held_154)
                tally: Counter = Counter()
                for listing in self.offers.values():
                    if all((self.holds_value({'value': v}, listing.words()) for v in values)):
                        held_155 = set((w for w in flat(listing.title).split() if 4 <= len(w) and (not w.isdigit()) and (w not in FILLER) and (w not in have) and (re.sub('(es|s)$', '', w) not in set((re.sub('(es|s)$', '', h) for h in have)))))
                        tally.update(held_155)
                held_156 = list((base + [w] for w, n in tally.most_common(4) if 2 <= n))
                return held_156
            else:
                held_153 = []
                return held_153
        else:
            held_152 = []
            return held_152

    def varied(self, words: list[str]) -> list[list[str]]:
        text = search_text(words, None)
        held_157 = text.split()
        have = set(held_157)
        tally: Counter = Counter()
        rows = list((l for l in self.offers.values() if text in l.via))
        for listing in rows:
            held_158 = set((w for w in flat(listing.title).split() if 4 <= len(w) and (not w.isdigit()) and (w not in FILLER) and (w not in have) and (w not in flat(listing.brand).split())))
            tally.update(held_158)
        held_159 = list((words + [w] for w, n in tally.most_common() if 2 <= n <= len(rows) // 2))[:2]
        return held_159

    def too_few(self) -> bool:
        held_160 = sum((1 for l in self.offers.values() if self.within_reach(l) and self.status(l) != 'no')) < 3
        return held_160

    def shelf_in_request(self) -> str:
        entry = self.kit.role_map.get('search')
        if not (not entry or 'category' not in entry['slots']):
            held_161 = self.kit.tool_specs[entry['tool']]['properties'].get(entry['slots']['category'])
            if not held_161:
                held_161 = {}
            prop = held_161
            held_162 = ' '.join([self.brief.category, self.brief.item])
            wanted = roots(held_162)
            named = list((e for e in prop.get('enum') or [] if isinstance(e, str) and roots(e) and (roots(e) <= wanted)))
            held_163 = max(named, key=lambda e: len(roots(e)), default='')
            return held_163
        else:
            return ''

    def discover(self) -> None:
        self.named_shelf = self.named_shelf or self.shelf_in_request()
        cheap = 'cheapest' == self.brief.priority
        searched = 0
        for words in self.search_plan():
            if not (self.timer.elapsed() > SEARCH_DEADLINE or not self.has_room(self.holdback() + 3) or 4 <= searched):
                first = not self.offers
                count = self.search(words, everything=(cheap or bool(self.brief.measured) or bool(self.category or self.named_shelf)) and first)
                if not None is count:
                    searched = searched + 1
                    held_164 = search_text(words, None)
                    key = search_key(held_164)
                    if first:
                        if count:
                            if key in self.no_more:
                                if count > self.search_log.get(key, (0, 0))[1]:
                                    for extra in self.varied(words)[:2]:
                                        if self.has_room(self.holdback() + 3):
                                            self.search(extra)
                    if first:
                        if count:
                            self.first_seen_whole = search_key(search_text(words, None)) in self.seen_whole
                    self.settle_shelf()
                    seen_whole = search_key(search_text(words, None)) in self.seen_whole
                    if count:
                        if not self.too_few():
                            if self.names_all(words) or 3 <= searched:
                                if self.category or 4 <= searched:
                                    if seen_whole or 3 <= searched:
                                        break
                else:
                    continue
            else:
                break
        self.settle_shelf()

    def confirm_pool(self) -> None:
        self.confirm_top()
        if 'cheapest' == self.brief.priority:
            self.probe_cheaper()

    def probe_cheaper(self) -> None:
        entry = self.kit.role_map.get('filter')
        if not (not entry or 'max_price' not in entry['slots'] or (not self.category)):
            best = self.ordered(confirmed=True)
            if not best or None is best[0].price:
                return
            elif not self.first_seen_whole:
                top = best[0].price - 0.01
                low, high, cap = (0.0, top, top)
                for _ in range(4):
                    if not (self.timer.elapsed() > SEARCH_DEADLINE or not self.has_room(self.holdback() + 2) or 0.02 > high - low):
                        cap = round(cap, 2)
                        count = self.shelf_sweep(cap, near=1.6, whole_only=True)
                        if not None is count:
                            visible = max(self.rows_per_page, 1) * self.kit.page_limit('filter')
                            if visible < count <= 1.6 * visible:
                                self.confirm_top()
                                found = self.ordered(confirmed=True)
                                if found:
                                    if None is not found[0].price:
                                        if cap >= found[0].price:
                                            break
                            if not visible < count:
                                low = cap
                                self.confirm_top()
                                found = self.ordered(confirmed=True)
                                if found and None is not found[0].price and (low >= found[0].price) or low + 0.01 >= high:
                                    break
                                cap = (low + high) / 2
                            else:
                                high = cap
                                seen = sorted((l.price for l in self.offers.values() if None is not l.price and self.category == l.category and (low < l.price <= cap)))
                                if not 5 <= len(seen):
                                    cap = low + (cap - low) * max(0.3, min(0.7, (0.8 * visible / count) ** 0.6))
                                else:
                                    held_165 = min(len(seen) - 1, int(len(seen) * 0.85 * visible / count) - 1)
                                    k = max(0, held_165)
                                    cap = max(low + 0.01, seen[k])
                                continue
                        else:
                            break
                    else:
                        break
                self.confirm_top()
            else:
                self.jot('cheapest: the first search was seen whole, no window probing')
                return
        else:
            return

    def widen_search(self) -> bool:
        before = len(self.offers)
        pages = self.kit.page_limit('search')
        named = lambda text: sum((1 for opts in self.value_terms().values() if any((roots(o) and roots(o) <= roots(text) for o in opts))))
        open_ = sorted(((c, s, p, t, k) for k, (c, s, p, t) in self.search_log.items() if c and s < c and (pages > p)), key=lambda x: (-named(x[3]), -x[0]))
        for count, seen, page, text, key in open_[:2]:
            self.read_pages(text, key, page + 1, True)
        if not before < len(self.offers):
            for words in self.split_plan()[:2]:
                self.search(words, everything=True)
            if not before < len(self.offers):
                for words in self.search_plan():
                    if self.search(words, everything=True):
                        break
                self.settle_shelf()
                self.shelf_sweep(self.brief.ceiling)
                self.confirm_top()
                held_166 = before < len(self.offers)
                return held_166
            else:
                self.confirm_top()
                return True
        else:
            self.confirm_top()
            return True

class Checker:

    def need_key(self, r: dict) -> str:
        held_167 = '{}|{}|{}'.format(flat(r['property']), flat(r['value']), r['exclude'])
        return held_167

    def live_needs(self) -> list[dict]:
        read = list((l for l in self.offers.values() if l.sheet))
        stamp = (self.brief.revision, len(self.brief.needs), len(self.checks), hash(tuple((v[:3] for v in self.checks.values()))), len(read))
        if not (self.live_cache and stamp == self.live_cache[0]):
            by_key: dict[str, list] = {}
            for (_, k), v in self.checks.items():
                if v[2]:
                    by_key.setdefault(k, []).append(v)
            out = []
            for r in self.brief.needs:
                held_169 = self.need_key(r)
                judged = by_key.get(held_169, [])
                if 4 <= len(judged) and (not any((v[0] in ('yes', 'no') for v in judged))):
                    continue
                elif r.get('source') == 'request' and (not r['exclude']) and (len(list((l for l in read if self.within_reach(l)))) >= 6) and (not any(('yes' == v[0] for v in judged))) and (not self.really_demanded(r)):
                    continue
                elif not (4 <= len(read) and len(self.value_terms_of(r)) >= 3 and (not any(('yes' == v[0] for v in judged))) and (not any((self.holds_value(r, l.words()) for l in read)))):
                    out.append(r)
                else:
                    continue
            self.live_cache = (stamp, out)
            return out
        else:
            held_168 = self.live_cache[1]
            return held_168

    def really_demanded(self, r: dict) -> bool:
        key = self.need_key(r)
        if key not in self.must_cache:
            if not self.timer.elapsed() > MODEL_DEADLINE:
                held_171 = 'Request:\n{}\n\nRecorded: {}: {}'.format(self.session.request_text, r['property'], r['value'])
                held_170 = self.oracle.ask_model('requirement_check', MUST_PROMPT, held_171, {'demanded': {'type': 'boolean'}}, ['demanded'], 200)
                if not held_170:
                    held_170 = {}
                answer = held_170
                self.must_cache[key] = answer.get('demanded') is not False
                if not self.must_cache[key]:
                    held_172 = '{}: {} is context in the request, not a demand: left out'.format(r['property'], r['value'])
                    self.jot(held_172)
            else:
                return True
        held_173 = self.must_cache[key]
        return held_173

    def marks_for(self, listing: Offer) -> list[str]:
        out = []
        literal = set((k for (_, k), v in self.checks.items() if 'yes' == v[0] and v[2]))
        for r in self.live_needs():
            key = self.need_key(r)
            verdict = self.checks.get((listing.id, key))
            if (not verdict or 'unknown' == verdict[0]) and (listing.id, key) in self.loose_yes and (None is not listing.sheet) and (key not in literal):
                out.append('mixed')
            elif not verdict or 'unknown' == verdict[0]:
                out.append('unknown')
            elif not verdict[2]:
                out.append('hint+' if 'yes' == verdict[0] else 'hint-')
            else:
                held_174 = 'yes' == verdict[0]
                if held_174:
                    held_174 = verdict[3]
                if held_174:
                    held_174 = not self.own_label_ok(listing, r)
                mixed = held_174
                out.append('mixed' if mixed else verdict[0])
        return out

    def own_label_ok(self, listing: Offer, r: dict) -> bool:
        prop = roots(r['property'])
        exact = list(('{} {}'.format(k, v) for k, v in (listing.sheet or {}).items() if prop and roots(re.sub('[._]+', ' ', str(k))) == prop))
        held_175 = bool(exact) and all((self.holds_value(r, entry) for entry in exact))
        return held_175

    def status(self, listing: Offer) -> str:
        if not listing.sold_out:
            marks = self.marks_for(listing)
            if 'no' in marks:
                return 'no'
            elif not all(('yes' == m for m in marks)):
                if all((m in ('yes', 'mixed') for m in marks)):
                    return 'mixed'
                return 'unknown'
            else:
                return 'yes'
        else:
            return 'no'

    def check_offers(self, listings: list[Offer]) -> None:
        reqs = self.brief.needs
        if reqs:
            todo = list((l for l in listings if any(((v := self.checks.get((l.id, self.need_key(r)))) is None or (None is not l.sheet and (not v[2])) for r in reqs))))
            for start in range(0, len(todo), JUDGE_BATCH):
                batch = todo[start:start + JUDGE_BATCH]
                text = 'Requirements:\n' + '\n'.join(('{}. {}: {}{}'.format(n, r['property'], 'rule out ' if r['exclude'] else '', r['value']) + ('  (shopper: "{}")'.format(r['quote']) if r.get('quote') else '') for n, r in enumerate(reqs, 1))) + '\n\nProduct wanted: {}\n\nOffers:\n'.format(self.brief.item) + '\n'.join((l.as_card(n) for n, l in enumerate(batch, 1)))
                answer = self.oracle.ask_model('listing_checks', CHECK_PROMPT, text, CHECK_SHAPE, ['listings'], 400 + 160 * len(batch) * len(reqs))
                if answer:
                    for item in answer.get('listings') or []:
                        if not (not isinstance(item, dict) or not isinstance(item.get('n'), int) or (not 1 <= item['n'] <= len(batch))):
                            listing = batch[item['n'] - 1]
                            in_sheet, anywhere = (listing.sheet_words(), listing.words())
                            for check in item.get('checks') or []:
                                if not (not isinstance(check, dict) or not isinstance(check.get('r'), int) or (not 1 <= check['r'] <= len(reqs))):
                                    r = reqs[check['r'] - 1]
                                    if check.get('verdict') in ('yes', 'no', 'unknown'):
                                        verdict = check.get('verdict')
                                    else:
                                        verdict = 'unknown'
                                    held_176 = check.get('quote')
                                    quote = flat(held_176)
                                    held_177 = check.get('mixed')
                                    mixed = bool(held_177)
                                    held_178 = BOUND_RE.search(r['value'])
                                    bounded = bool(held_178)
                                    if bounded:
                                        if verdict in ('yes', 'no'):
                                            if not r['exclude']:
                                                held_179 = bound_check(r['value'], str(check.get('quote') or ''))
                                                if not held_179:
                                                    held_179 = bound_check(r['value'], listing.title, True)
                                                if not held_179:
                                                    held_179 = verdict
                                                verdict = held_179
                                    if not r['exclude']:
                                        if self.listed_on_sheet(r, listing):
                                            mixed = False
                                            if 'no' == verdict:
                                                verdict, quote = ('yes', '')
                                    if r['exclude'] and 'yes' == verdict:
                                        if None is not listing.sheet:
                                            source = 'datasheet'
                                        else:
                                            source = 'title'
                                    elif 'yes' == verdict and (not bounded) and (not self.holds_value(r, anywhere)):
                                        if quote:
                                            if quote in anywhere:
                                                self.loose_yes.add((listing.id, self.need_key(r)))
                                        verdict, source = ('unknown', 'none')
                                    elif verdict in ('yes', 'no') and quote and (None is not listing.sheet) and (quote in in_sheet):
                                        source = 'datasheet'
                                    elif verdict in ('yes', 'no') and quote and (quote in anywhere):
                                        source = 'title'
                                    elif not ('yes' == verdict and (bounded or self.holds_value(r, anywhere))):
                                        verdict, source = ('unknown', 'none')
                                    elif None is not listing.sheet and self.holds_value(r, in_sheet):
                                        source = 'datasheet'
                                    else:
                                        source = 'title'
                                    if 'yes' == verdict:
                                        if not r['exclude']:
                                            if not mixed:
                                                if self.sheet_conflict(r, listing):
                                                    mixed = True
                                    self.checks[listing.id, self.need_key(r)] = (verdict, source, None is not listing.sheet, mixed)
                                else:
                                    continue
                        else:
                            continue
                else:
                    continue
        else:
            return

    @staticmethod
    def listed_on_sheet(r: dict, listing: Offer) -> bool:
        prop, want = (roots(r['property']), Assistant.value_terms_of(r))
        for label, value in (listing.sheet or {}).items():
            if not (not want or not roots(re.sub('[._]+', ' ', str(label))) & prop):
                parts = list((part for part in re.split('[,/;|]+', str(value)) if part.strip()))
                if 2 <= len(parts):
                    if any((all((Assistant.word_forms(w) & word_set(part) for w in want)) for part in parts)):
                        return True
            else:
                continue
        return False

    @staticmethod
    def sheet_conflict(r: dict, listing: Offer) -> bool:
        if not (not listing.sheet or BOUND_RE.search(r['value'])):
            prop = roots(r['property']) - {'type', 'kind'}
            if prop:
                related = list(((str(label), str(value)) for label, value in listing.sheet.items() if prop <= roots(re.sub('[._]+', ' ', str(label)))))
                if related:
                    holds = list((Assistant.holds_value(r, '{} {}'.format(label, value)) for label, value in related))
                    if not any(holds):
                        return True
                    return False
                    listed = any((h and len(list((p for p in re.split('[,/;|]+', value) if p.strip()))) >= 2 for h, (label, value) in zip(holds, related)))
                    if listed:
                        return False

                    def label_words(label: str) -> set[str]:
                        held_180 = set((w.rstrip('s') for w in flat(re.sub('[._]+', ' ', label)).split()))
                        return held_180
                    held = list((label_words(label) for h, (label, _) in zip(holds, related) if h))
                    held_181 = any((not h and any((other >= w or w >= other for other in held)) for h, (label, _) in zip(holds, related) for w in [label_words(label)]))
                    return held_181
                else:
                    return False
            else:
                return False
        else:
            return False

    @staticmethod
    def value_terms_of(r: dict) -> set[str]:
        held_182 = set((w for w in flat(r['value']).split() if (3 <= len(w) or w.isdigit()) and w not in FILLER))
        return held_182

    @staticmethod
    def is_grounded(phrase: str, heard: set[str]) -> bool:
        want = Assistant.value_terms_of({'value': phrase})
        held_183 = not want or any((Assistant.word_forms(w) & heard or (4 <= len(w) and any((h[:4] == w[:4] for h in heard))) for w in want))
        return held_183

    @staticmethod
    def word_forms(word: str) -> set[str]:
        out = {word}
        if 3 < len(word):
            if word.endswith('s'):
                out.add(word[:-1])
        if 3 < len(word):
            if word.endswith('es'):
                out.add(word[:-2])
        return out

    @staticmethod
    def holds_value(r: dict, words: str) -> bool:
        have = word_set(words)
        for option in re.split('\\s+or\\s+|/', r['value']):
            ratio = re.search('(\\d+(?:\\.\\d+)?)\\s*:\\s*(\\d+(?:\\.\\d+)?)', option)
            if not ratio:
                held_184 = dict(r, value=option)
                want = Assistant.value_terms_of(held_184)
                if want:
                    if all((Assistant.word_forms(w) & have for w in want)):
                        return True
            elif not re.search('(?<![\\d.]){}\\s*:\\s*{}(?![\\d.])'.format(re.escape(ratio.group(1)), re.escape(ratio.group(2))), low_text(words)):
                continue
            else:
                return True
        return False

    def ordered(self, confirmed: bool=False) -> list[Offer]:
        pool = list((l for l in self.offers.values() if self.within_reach(l)))
        state = dict(((l.id, self.status(l)) for l in pool))
        held_185 = self.live_needs()
        active = bool(held_185)
        if confirmed:
            pool = list((l for l in pool if state[l.id] == 'yes' and (None is not l.sheet or not active)))
        else:
            pool = list((l for l in pool if state[l.id] != 'no'))
        values = list((w for v in self.brief.values() for w in flat(v).split()))

        def doubt(l: Offer) -> int:
            marks = self.marks_for(l)
            held_186 = 2 * marks.count('hint-') + marks.count('mixed') - marks.count('hint+')
            return held_186

        def overlap(l: Offer) -> int:
            said = l.words().split()
            held_187 = sum((1 for w in values if any((s.startswith(w[:5]) for s in said))))
            return held_187
        price = lambda l: l.price if None is not l.price else 1e+18
        backed_by = set((text for text in set((t for l in pool for t in l.via)) if self.names_all(text.split())))

        def backed(l: Offer) -> bool:
            held_188 = bool(l.via & backed_by)
            return held_188
        title_reqs = list((r for r in self.live_needs() if not r['exclude']))

        def agree(l: Offer) -> float:
            have = word_set(l.title)
            total = 0.0
            for r in title_reqs:
                want = self.value_terms_of(r)
                if want:
                    total += sum((1 for w in want if self.word_forms(w) & have)) / len(want)
            return total
        owned = self.own_labelled()

        def aside(l: Offer) -> int:
            held_189 = sum((1 for r in owned if self.where_held(l, r) == 'other'))
            return held_189
        if not 'cheapest' == self.brief.priority:
            held_191 = flat(self.brief.item)
            alternatives = re.findall('([a-z]+) or ([a-z]+)', held_191)
            kinds = set((w for w in flat(self.brief.noun).split() + list((w for pair in alternatives for w in pair)) if 4 <= len(w) and w not in FILLER))

            def named(l: Offer) -> bool:
                have = word_set(l.title)
                held_192 = not kinds or any((self.word_forms(w) & have or (5 <= len(w) and any((h[:5] == w[:5] for h in have))) for w in kinds))
                return held_192

            def ranking(l: Offer) -> tuple:
                value = self.measured(l)
                if None is not value:
                    held_193 = (None is value, value if self.brief.low_wins else -value)
                    return held_193
                held_194 = (True, 0)
                return held_194
            held_195 = sorted(pool, key=lambda l: (state[l.id] != 'yes', doubt(l), aside(l), ranking(l), not named(l), -agree(l), not backed(l), None is l.sheet, -overlap(l), price(l), l.order))
            return held_195
        else:
            held_190 = sorted(pool, key=lambda l: (state[l.id] not in ('yes', 'mixed') and doubt(l) > 0, state[l.id] != 'yes' and (not backed(l)), aside(l), price(l), state[l.id] != 'yes', l.order))
            return held_190

    def measured(self, l: Offer) -> float | None:
        if self.brief.measured:
            found = self.measure_cache.get((l.id, True))
            if None is not found:
                return found
            held_196 = self.measure_cache.get((l.id, False))
            return held_196
        else:
            return None

    def read_measures(self, listings: list[Offer]) -> None:
        if not (not self.brief.measured or self.timer.elapsed() > MODEL_DEADLINE):
            todo = list((l for l in listings if (l.id, None is not l.sheet) not in self.measure_cache))[:MEASURE_READS]
            held_197 = self.brief.rank_on
            if not held_197:
                held_197 = self.brief.rank_words
            measure = held_197
            for start in range(0, len(todo), 12):
                if not self.timer.elapsed() > MODEL_DEADLINE:
                    part = todo[start:start + 12]
                    text = 'Measure: {} (the shopper is after {}; give this measure alone, no other figure)\nProduct: {}\n\nOffers:\n'.format(measure, self.brief.rank_words, self.brief.item) + '\n'.join((l.as_card(n) for n, l in enumerate(part, 1)))
                    held_198 = self.oracle.ask_model('measure_figures', MEASURE_PROMPT, text, MEASURE_SHAPE, ['figures'], 600)
                    if not held_198:
                        held_198 = {}
                    answer = held_198
                    got = dict(((f['n'], f.get('value')) for f in answer.get('figures') or [] if isinstance(f, dict) and isinstance(f.get('n'), int)))
                    for n, l in enumerate(part, 1):
                        value = got.get(n)
                        self.measure_cache[l.id, None is not l.sheet] = float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None
                else:
                    return
        else:
            return

    def confirm_top(self, want: int=2) -> None:
        if 'other' == self.brief.priority:
            want = max(want, 5)
            self.read_measures(self.ordered()[:MEASURE_READS])
        self.check_offers(self.ordered()[:24])
        for _ in range(6):
            leading = self.ordered()
            if self.has_compare_tool():
                batch = 5
            else:
                batch = 3
            if 'cheapest' == self.brief.priority:
                held_199 = self.ordered(confirmed=True)
                confirmed = self.clean_fits(held_199)
                if want <= len(confirmed):
                    cut = confirmed[min(len(confirmed), want) - 1].price
                else:
                    cut = None
                need = list((l for l in leading if None is l.sheet and (None is cut or (l.price or 0) <= cut)))[:batch]
            elif not len(self.clean_fits(self.ordered(confirmed=True))) >= want:
                need = list((l for l in leading if None is l.sheet))[:batch]
            else:
                return
            if not (not need or SHEET_READS <= self.sheet_reads or (not self.has_room(self.holdback()))):
                self.open_sheets(need)
                self.check_offers(need)
            else:
                return

    def where_held(self, l: Offer, r: dict) -> str:
        if not self.holds_value(r, l.title):
            labels = list((re.sub('[._]+', ' ', str(k)) for k, v in (l.sheet or {}).items() if self.holds_value(r, '{} {}'.format(re.sub('[._]+', ' ', str(k)), v))))
            prop = roots(r['property'])
            if prop and any((prop <= roots(k) for k in labels)):
                return 'own'
            if labels:
                return 'other'
            return ''
        else:
            return 'title'

    def own_labelled(self) -> list[dict]:
        read = list((l for l in self.offers.values() if l.sheet and (not (self.category and l.category and (self.category != l.category)))))
        held_200 = list((r for r in self.live_needs() if not r['exclude'] and (not BOUND_RE.search(r['value'])) and self.value_terms_of(r) and any((self.where_held(l, r) == 'own' for l in read))))
        return held_200

    def clean_fits(self, confirmed: list[Offer], reqs: list[dict] | None=None) -> list[Offer]:
        if None is reqs:
            reqs = list((r for r in self.live_needs() if not r['exclude'] and (not BOUND_RE.search(r['value'])) and self.value_terms_of(r)))
        held_201 = list((l for l in confirmed if not any((self.where_held(l, r) == 'other' for r in reqs))))
        return held_201

    def by_rank(self, confirmed: list[Offer]) -> list[Offer]:
        if not ('other' != self.brief.priority or 2 > len(confirmed) or (not self.brief.rank_words)):
            self.read_measures(confirmed[:10])
            if not (self.brief.measured and sum((1 for l in confirmed[:10] if self.measured(l) is not None)) >= 2):
                held_203 = sorted((l.id for l in confirmed[:10]))
                stamp = tuple(held_203)
                if stamp not in self.rank_cache:
                    text = 'What the shopper wants most: {}\nProduct: {}\n\nOffers:\n'.format(self.brief.rank_words, self.brief.item) + '\n'.join((l.as_card(n) for n, l in enumerate(confirmed[:10], 1)))
                    held_204 = self.oracle.ask_model('best_first', 'Rank the listings from the best match to the worst for what the shopper wants, going by their datasheets and titles. Answer with the listing numbers.', text, {'order': {'type': 'array', 'items': {'type': 'integer'}}}, ['order'], 300)
                    if not held_204:
                        held_204 = {}
                    answer = held_204
                    order = list((n for n in answer.get('order') or [] if isinstance(n, int) and 1 <= n <= len(confirmed[:10])))
                    self.rank_cache[stamp] = list((confirmed[n - 1].id for n in dict.fromkeys(order)))
                    held_205 = 'priority order: {}'.format(self.rank_cache[stamp])
                    self.jot(held_205)
                rank = dict(((pid, i) for i, pid in enumerate(self.rank_cache[stamp])))
                held_206 = sorted(confirmed, key=lambda l: rank.get(l.id, len(rank)))
                return held_206
            else:

                def figure(l: Offer) -> tuple:
                    value = self.measured(l)
                    held_202 = (None is value, 0 if None is value else value if self.brief.low_wins else -value)
                    return held_202
                ordered = sorted(confirmed, key=figure)
                self.jot('by the measure: ' + ', '.join(('{}={}'.format(l.id, self.measured(l)) for l in ordered[:4])))
                return ordered
        else:
            return confirmed

    def worth_reading(self, listing: Offer) -> bool:
        if not any((self.names_all(text.split()) for text in listing.via)):
            reqs = list((r for r in self.live_needs() if not r['exclude']))
            held_207 = bool(reqs) and all((self.holds_value(r, listing.title) for r in reqs))
            return held_207
        else:
            return True

class Interviewer:

    def holdback(self) -> int:
        if not (self.basket_now or self.as_line):
            held_209 = min(SEARCH_HOLDBACK, max(6, round(0.24 * self.session.step_cap))) + self.question_holdback()
            return held_209
        else:
            held_208 = CLOSING_STEPS + 1
            return held_208

    def question_holdback(self) -> int:
        if not (self.questions_done or 'ask' not in self.kit.role_map):
            if 30 >= self.session.step_cap:
                cap = 2
            else:
                cap = QUESTION_CAP
            held_210 = min(cap, self.brief.message_cap if None is not self.brief.message_cap else 2)
            return held_210
        else:
            return 0

    def listen(self, messages: list[str], context: str, answers: bool=False) -> None:
        messages = list((m for m in messages if m and m.strip()))
        if messages:
            text = 'Product wanted: {}\nCurrent note:\n{}\nTop price: {}\n\n{}\n'.format(self.brief.item, self.brief.lines(), self.brief.ceiling, context) + '\n'.join(('- {}'.format(m) for m in messages))
            answer = self.oracle.ask_model('record_update', LISTEN_PROMPT, text, LISTEN_SHAPE, ['new_requirements', 'dropped_properties'], 700)
            joined = ' '.join(messages)
            figures = set((float(x.replace(',', '')) for x in re.findall('\\d[\\d,]*(?:\\.\\d+)?', joined)))
            held_211 = flat(' '.join((re.sub('^\\(asked about [^)]*\\)\\s*', '', m) for m in messages))).split()
            heard = set(held_211)
            if None is answer:
                answer = {'new_requirements': [], 'dropped_properties': re.findall('never mind (?:about )?(?:the )?([a-z ]+?)[.,!]', joined, re.I), 'max_price': None, 'carted_unavailable': bool(re.search('stock|sold out|unavailable|no longer', joined, re.I))}
            required = list((dict(r) for r in self.brief.needs if not r['exclude']))
            for prop in answer.get('dropped_properties') or []:
                if isinstance(prop, str):
                    values = ' '.join((r['value'] for r in required if flat(r['property']) == flat(prop)))
                    if not (self.is_grounded(prop, heard) or (values and self.is_grounded(values, heard))):
                        held_212 = 'ignored drop {!r}: not named in {!r}'.format(prop, joined[:120])
                        self.jot(held_212)
                        continue
                    elif self.brief.drop_need(prop):
                        held_213 = 'dropped {!r}'.format(prop)
                        self.jot(held_213)
                else:
                    continue
            for item in answer.get('new_requirements') or []:
                if not isinstance(item, dict):
                    continue
                elif not (item.get('exclude') and any((flat(r['value']) == flat(item.get('value')) for r in required))):
                    if answers:
                        if str(item.get('value') or '').strip():
                            said = list((c for c in re.split('[.!?,;:()]|\\b(?:and|but)\\b', ' '.join(messages)) if self.is_grounded(str(item['value']), set(flat(c).split()))))
                            if item.get('exclude'):
                                if said:
                                    if not any((re.search(RULE_OUT_RE, c, re.I) for c in said)):
                                        item['exclude'] = False
                                        held_216 = item.get('property')
                                        held_217 = item.get('value')
                                        held_215 = 'reply read as requiring {}={}'.format(held_216, held_217)
                                        self.jot(held_215)
                            bound = next((b for c in said for b in [re.search('\\b(?:at most|at least|up to|under|over|less than|more than|no more than|no less than)\\s*\\d[\\d.,]*\\s*[a-z]*', c, re.I)] if b), None)
                            if bound:
                                if not BOUND_RE.search(str(item['value'])):
                                    if re.sub('\\D', '', str(item['value'])) in re.sub('\\D', '', bound.group(0)):
                                        item['value'] = bound.group(0).strip()
                    if str(item.get('value') or '').strip() and (not self.is_grounded(str(item['value']), heard)):
                        held_219 = item.get('property')
                        held_220 = item.get('value')
                        held_218 = 'ignored need {}={}: not named in {!r}'.format(held_219, held_220, joined[:120])
                        self.jot(held_218)
                        continue
                    elif str(item.get('property') or '').strip():
                        if str(item.get('value') or '').strip():
                            self.brief.add(item['property'], item['value'], bool(item.get('exclude')), joined[:200], 'told')
                            held_221 = 'new need {}={}{}'.format(item['property'], item['value'], ' (rule out)' if item.get('exclude') else '')
                            self.jot(held_221)
                else:
                    prop = next((r['property'] for r in required if flat(r['value']) == flat(item.get('value'))))
                    if self.is_grounded('{} {}'.format(prop, item.get('value')), heard):
                        if self.brief.drop_need(prop):
                            held_214 = 'dropped {!r} (read as ruling out its own value)'.format(prop)
                            self.jot(held_214)
                    continue
            price = answer.get('max_price')
            if isinstance(price, (int, float)) and (not isinstance(price, bool)) and (0 < price):
                price = next((p for p in [float(price), round(price * 100, 2), round(price / 100, 4)] if p in figures), None)
            else:
                price = None
            if None is not price:
                if self.brief.ceiling != price:
                    held_222 = 'budget {} -> {}'.format(self.brief.ceiling, price)
                    self.jot(held_222)
                    self.brief.ceiling = float(price)
            if answer.get('carted_unavailable'):
                if self.basket_now:
                    self.basket_doubt = True
                    held_224 = list(self.basket_now)
                    held_223 = 'carted maybe unavailable: {}'.format(held_224)
                    self.jot(held_223)
        else:
            return

    def interview(self) -> None:
        if not ('ask' not in self.kit.role_map or self.questions_done):
            limit = self.question_holdback()
            self.questions_done = True
            while 0 < limit and (not self.has_room(CLOSING_STEPS + limit + 2)):
                limit = limit - 1
            if not (0 >= limit or self.timer.elapsed() > SEARCH_DEADLINE):
                for listing in list((l for l in self.ordered() if None is l.sheet))[:max(0, 4 - sum((1 for l in self.offers.values() if l.sheet)))]:
                    if self.has_room(CLOSING_STEPS + limit + 4):
                        if SHEET_READS > self.sheet_reads:
                            self.open_sheet(listing)
                labels: dict[str, set] = {}
                carried: Counter = Counter()
                read = 0
                for listing in self.offers.values():
                    if listing.sheet:
                        if self.within_reach(listing):
                            read = read + 1
                            held_225 = set((re.sub('([a-z])([A-Z])', '\\1 \\2', re.sub('[._]+', ' ', str(k))).strip().lower() for k in listing.sheet))
                            carried.update(held_225)
                            for key, value in listing.sheet.items():
                                labels.setdefault(re.sub('([a-z])([A-Z])', '\\1 \\2', re.sub('[._]+', ' ', str(key))).strip().lower(), set()).add(str(value)[:40])
                stated = set((flat(r['property']) for r in self.brief.needs))
                titles = list((flat(l.title) for l in self.offers.values() if l.title))

                def in_titles(values: set) -> int:
                    words = list((flat(v) for v in values if len(flat(v)) > 2))
                    held_226 = sum((1 for t in titles if any((re.search('\\b{}\\b'.format(re.escape(w)), t) for w in words))))
                    return held_226
                shown = '\n'.join(('- {} (on {} of {} listings): {} (named in {} of {} titles)'.format(k, carried[k], read, ', '.join(sorted(v)[:6]), in_titles(v), len(titles)) for k, v in sorted(labels.items(), key=lambda kv: (-carried[kv[0]], -len(kv[1])))[:30]))
                text = "What the shopper asked for:\n{}\n\nProduct wanted: {}\nRequirements the request already sets:\n{}\n\nProperties the shop's datasheets give for this product (with the values seen):\n{}\n\nPick at most {}.".format(self.session.request_text, self.brief.item, self.brief.lines(), shown or '(no datasheet read yet)', 2 * limit)
                held_227 = self.oracle.ask_model('question_plan', QUESTION_PROMPT, text, QUESTION_SHAPE, ['properties'], 300)
                if not held_227:
                    held_227 = {}
                answer = held_227
                props = list((p for p in answer.get('properties') or [] if isinstance(p, str) and p.strip() and (flat(p) not in stated)))
                if not props:
                    props = list((k for k, v in sorted(labels.items(), key=lambda kv: -len(kv[1])) if 1 < len(v) and flat(k) not in stated))
                said = ' '.join([self.session.request_text] + list((r['property'] + ' ' + r['value'] for r in self.brief.needs))).lower()
                recorded = set((flat(k) for k in labels))
                held_228 = next((k for k in labels if re.search('\\b(gender|sex)\\b', flat(k))), None)
                if not held_228:
                    if self.brief.is_worn:
                        held_228 = 'gender'
                    else:
                        held_228 = None
                whom = held_228
                unknown = re.search("\\bgift|\\bpresent\\b|drew names|swap|secret santa|met (?:only )?(?:once|twice)|barely know|don'?t know (?:them|her|him)", self.session.request_text, re.I)
                if whom:
                    if unknown:
                        if not self.brief.wearer:
                            if not re.search('\\b(gender|women|woman|men|man|female|male|girl|girls|boy|boys|unisex|ladies|kids?)\\b', said):
                                props = [whom] + list((p for p in props if not re.search('gender|sex', flat(p))))
                for cue, kinds in STORY_CUES:
                    if re.search(cue, self.session.request_text, re.I):
                        for kind in kinds:
                            label = min((k for k in labels if kind in flat(k).split() and flat(k) not in stated), default=None, key=len)
                            if not (None is label or any((kind in flat(s).split() for s in stated))):
                                if not any((kind in flat(p).split() for p in props[:2])):
                                    props = props[:1] + [label] + list((p for p in props[1:] if kind not in flat(p).split()))
                                break
                            else:
                                continue
                    else:
                        continue
                props = list((p for p in dict.fromkeys(props) if flat(p) not in stated and (not re.search('price|cost|stock|availab|option|variant|quantit|deliver|shipping|^id$|sku|\\bmodel\\b|^condition$', flat(p)))))[:limit]
                held_230 = len(recorded)
                held_229 = 'question plan: {} (datasheet labels: {})'.format(props, held_230)
                self.jot(held_229)
                replies = []
                for first in range(len(props)):
                    pair = [re.sub('([a-z])([A-Z])', '\\1 \\2', re.sub('[._]+', ' ', props[first])).strip().lower()]
                    prop_words = ' and '.join(pair)
                    held_231 = self.brief.noun
                    if not held_231:
                        held_231 = self.brief.item
                    if not held_231:
                        held_231 = self.category
                    if not held_231:
                        held_231 = 'item'
                    noun = held_231
                    seen = next((v for k, v in labels.items() if flat(k) == flat(pair[0])), set())
                    shown = list((v for v in sorted(seen, key=len) if 0 < len(v) <= 25))[:2]
                    if 2 == len(pair):
                        question = 'What {} and what {} would you like the {} to have?'.format(pair[0], pair[1], noun)
                    elif 2 == len(shown):
                        question = 'Which {} would you like for the {}, for example {} or {}?'.format(pair[0], noun, shown[0], shown[1])
                    else:
                        question = 'What {} would you like the {} to have?'.format(pair[0], noun)
                    action = self.kit.build_call('ask', text=question)
                    if not (not action or not self.has_room(CLOSING_STEPS + 2)):
                        reply = self.call(*action)
                        said = message_texts(reply.seen)
                        held_232 = 'asked {!r}: {}'.format(prop_words, said[:2])
                        self.jot(held_232)
                        if not (isinstance(reply.seen, dict) and is_failure(reply.seen)):
                            replies += list(('(asked about {}) {}'.format(prop_words, t) for t in said))
                        else:
                            break
                    else:
                        break
                if replies:
                    if self.fellow_lines:
                        names = lambda part: set((n for n in (flat(part.brief.noun), flat(part.brief.item), flat(part.brief.category)) if n))
                        mine = names(self)
                        for reply in list(replies):
                            said = roots(reply)
                            if not any((roots(n) and roots(n) <= said for n in mine)):
                                other = next((p for p in self.fellow_lines if self is not p and any((roots(n) and roots(n) <= said for n in names(p)))), None)
                                if None is not other:
                                    replies.remove(reply)
                                    held_233 = 'reply names {!r}: heard there'.format(other.brief.item)
                                    self.jot(held_233)
                                    other.listen([reply], "The shopper's answers to the assistant's questions:", answers=True)
                            else:
                                continue
                if replies:
                    version = self.brief.revision
                    self.listen(replies, "The shopper's answers to the assistant's questions:", answers=True)
                    if version != self.brief.revision:
                        self.after_change()
            else:
                return
        else:
            return

    def after_change(self) -> None:
        self.check_offers(self.ordered()[:24])
        ranked_pick = self.brief.priority in ('cheapest', 'other')
        if ranked_pick or not self.ordered(confirmed=True):
            if self.timer.elapsed() < SEARCH_DEADLINE:
                noun = [self.brief.noun or self.brief.item]
                options = self.value_terms()
                told = list((options[r['value']][0] for r in self.brief.needs if 'told' == r['source'] and r['value'] in options))
                told += list((''.join(m) for r in self.brief.needs if 'told' == r['source'] and r['value'] not in options and (not r['exclude']) for m in re.findall('(\\d[\\d.,]*)\\s*([a-z]{1,4})\\b', low_text(r['value']))[:1]))
                told_lead = list((w for v in told for w in self.leading_words(v)[:1]))
                others = list((w for v in options for w in self.leading_words(options[v][0])[:1] if w not in told_lead))
                rest = list((w for v in told for w in self.leading_words(v)[1:]))
                if self.category or self.named_shelf:
                    if False is not self.shelf_ok:
                        whole = list((w for v in told for w in word_tokens(v) if w not in FILLER and (1 < len(w) or w.isdigit())))
                        self.search(whole + list((w for w in others if w not in whole)), everything=ranked_pick)
                self.search(told_lead + others + noun + rest)
                self.search(told_lead + noun + rest)
                self.settle_shelf()
        self.confirm_top()
        if 'cheapest' == self.brief.priority:
            self.probe_cheaper()

class Closer:

    def pick(self) -> Offer | None:
        confirmed: list[Offer] = []
        for _ in range(6):
            held_234 = self.ordered(confirmed=True)
            confirmed = self.by_rank(held_234)
            if confirmed:
                if 'cheapest' == self.brief.priority:
                    if None is not confirmed[0].price:
                        if self.has_room(CLOSING_STEPS + 2):
                            cheaper = list((l for l in self.ordered() if None is l.sheet and None is not l.price and (confirmed[0].price > l.price) and self.worth_reading(l)))
                            if cheaper:
                                self.open_sheet(cheaper[0])
                                if None is not cheaper[0].sheet:
                                    self.check_offers([cheaper[0]])
                                    continue
            if not (confirmed and self.clean_fits(confirmed[:1], self.own_labelled())):
                leading = list((l for l in self.ordered() if None is l.sheet))
                if confirmed:
                    if not leading:
                        if not self.widened_once:
                            if self.timer.elapsed() < SEARCH_DEADLINE:
                                if self.has_room(CLOSING_STEPS + 4):
                                    self.widened_once = True
                                    self.jot('the pick carries a value only under another label: widening the search first')
                                    if self.widen_search():
                                        continue
                if not (not leading or not self.has_room(CLOSING_STEPS + 1)):
                    self.open_sheet(leading[0])
                    if not None is leading[0].sheet:
                        self.check_offers([leading[0]])
                    else:
                        break
                else:
                    break
            else:
                held_235 = confirmed[0]
                return held_235
        held_236 = self.ordered(confirmed=True)
        confirmed = self.by_rank(held_236)
        if not confirmed:
            mixed = list((l for l in self.ordered() if None is not l.sheet and self.status(l) == 'mixed'))
            if mixed:
                if not self.widened_once:
                    if self.timer.elapsed() < SEARCH_DEADLINE:
                        if self.has_room(CLOSING_STEPS + 4):
                            self.widened_once = True
                            self.jot('only second-tier matches left: widening the search first')
                            if self.widen_search():
                                held_238 = self.ordered(confirmed=True)
                                confirmed = self.by_rank(held_238)
                                if confirmed:
                                    held_239 = confirmed[0]
                                    return held_239
                            mixed = list((l for l in self.ordered() if None is not l.sheet and self.status(l) == 'mixed'))
            if not mixed:
                unclear = list((l for l in self.ordered() if None is not l.sheet))
                if unclear:
                    held_241 = unclear[0]
                    return held_241
                return None
            else:
                held_240 = mixed[0]
                return held_240
        else:
            held_237 = confirmed[0]
            return held_237

    def second_step(self, reply: StepResult) -> StepResult:
        if not (self.session.done or 'confirm' not in self.kit.role_map or is_failure(reply.seen) or (not self.has_room(0))):
            token = code_of(reply.seen)
            if None is not token:
                action = self.kit.build_call('confirm', token=token)
            else:
                action = None
            if action:
                held_242 = 'confirming the purchase with {!r}'.format(token)
                self.jot(held_242)
                held_243 = self.call(*action)
                return held_243
            else:
                return reply
        else:
            return reply

    def put_in_basket(self, choice: Offer) -> bool:
        if not ('cart_add' not in self.kit.role_map and 'order' in self.kit.role_map):
            if self.kit.sets_whole_basket('cart_add'):
                others = []
            else:
                others = list((p for p in self.basket_now if choice.id != p))
            if self.kit.sets_whole_basket('cart_add'):
                if choice.id in self.basket_now:
                    if 1 < len(self.basket_now):
                        self.basket_now = {}
            for pid in others:
                action = self.kit.build_call('cart_remove', quantity=self.basket_now[pid], id=pid)
                if not action:
                    self.cannot_remove = True
                    return False
                elif self.has_room(2):
                    reply = self.call(*action)
                    if is_failure(reply.seen):
                        held_244 = 'remove {} failed: {}'.format(pid, str(reply.seen)[:120])
                        self.jot(held_244)
                    self.basket_now.pop(pid, None)
                else:
                    return False
            if not choice.id in self.basket_now:
                action = self.kit.build_call('cart_add', quantity=self.brief.units, id=choice.id)
                if action:
                    reply = self.call(*action)
                    if is_failure(reply.seen):
                        held_245 = 'add {} failed: {}'.format(choice.id, str(reply.seen)[:160])
                        self.jot(held_245)
                        choice.sold_out = True
                        return False
                    if self.kit.sets_whole_basket('cart_add'):
                        self.basket_now = {}
                    self.basket_now[choice.id] = self.brief.units
                    held_246 = 'carted {} {!r} at {}'.format(choice.id, choice.title[:60], choice.price)
                    self.jot(held_246)
                    held_247 = not choice.sold_out
                    return held_247
                else:
                    return False
            else:
                return True
        else:
            return True

    def reread(self, choice: Offer) -> bool:
        action = self.kit.build_call('view', id=choice.id)
        if not (not action or not self.has_room(2)):
            reply = self.call(*action)
            self.update_price_stock(choice, reply.seen)
            held_248 = self.within_reach(choice) and self.status(choice) != 'no'
            return held_248
        else:
            return True

    def close_out(self) -> None:
        for _ in range(16):
            if not (self.session.done or self.session.steps_left() <= 0):
                choice = self.pick()
                if self.basket_now:
                    held = self.offers.get(next(iter(self.basket_now)))
                else:
                    held = None
                if None is not held:
                    if choice is not held:
                        if self.brief.priority not in ('cheapest', 'other'):
                            if not held.sold_out:
                                if self.within_reach(held):
                                    if self.status(held) == 'yes':
                                        choice = held
                if self.cannot_remove:
                    if self.basket_now:
                        held_250 = iter(self.basket_now)
                        held_249 = next(held_250)
                        held = self.offers.get(held_249)
                        if held and self.within_reach(held) and (self.status(held) != 'no'):
                            choice = held
                        else:
                            choice = None
                        if None is choice:
                            self.refuse_order()
                            continue
                if None is choice:
                    if self.timer.elapsed() < SEARCH_DEADLINE and self.has_room(CLOSING_STEPS + 3) and self.widen_search():
                        continue
                    elif not 'decline' not in self.kit.role_map:
                        self.refuse_order()
                        continue
                    else:
                        return
                elif self.put_in_basket(choice):
                    if not self.rechecked_once or self.basket_doubt:
                        if self.has_room(4):
                            self.basket_doubt = False
                            self.rechecked_once = True
                            if not self.reread(choice):
                                held_251 = 'recheck: {} no longer fits (stock {}, price {})'.format(choice.id, choice.stock, choice.price)
                                self.jot(held_251)
                                continue
                    if not self.has_room(0):
                        return
                    held_252 = 'cart_add' not in self.kit.role_map
                    if held_252:
                        held_252 = 'order' in self.kit.role_map
                    direct = held_252
                    if direct:
                        action = self.kit.build_call('order', quantity=self.brief.units, id=choice.id)
                    else:
                        action = self.kit.build_call('checkout')
                    if not action:
                        return
                    held_253 = self.call(*action)
                    reply = self.second_step(held_253)
                    if self.session.done:
                        held_254 = 'ordered {} at {}'.format(choice.id, choice.price)
                        self.jot(held_254)
                        return
                    refusal = self.unheard(reply.refusals)
                    held_255 = 'checkout refused: {}'.format(reply.refusals or str(reply.seen)[:200])
                    self.jot(held_255)
                    if not reply.refusals:
                        if is_failure(reply.seen):
                            self.basket_now.clear()
                    version = self.brief.revision
                    self.listen(refusal, 'The shop turned the order down and said:')
                    if refusal:
                        if not choice.sold_out:
                            if version == self.brief.revision:
                                if re.search("out of stock|sold out|unavailable|no longer (?:available|in stock|for sale|sold)|none left|nothing left|zero left|nobody has any|gone from the shelves|all gone|can'?t be (?:ordered|bought)", ' '.join(refusal), re.I):
                                    choice.sold_out = True
                    if refusal:
                        if not choice.sold_out:
                            if self.has_room(4):
                                self.basket_doubt = False
                                if not self.reread(choice):
                                    held_256 = 'after the refusal: {} no longer fits (stock {}, price {})'.format(choice.id, choice.stock, choice.price)
                                    self.jot(held_256)
                    if version != self.brief.revision:
                        self.after_change()
                else:
                    continue
            else:
                return

    def refuse_order(self) -> None:
        read = list((l for l in self.offers.values() if None is not l.sheet))[:10]
        self.jot('declining; read listings: ' + '; '.join(('{} cat={!r} price={} stock={} afford={} status={} marks={}'.format(l.id, l.category, l.price, l.stock, self.within_reach(l), self.status(l), self.marks_for(l)) for l in read)))
        action = self.kit.build_call('decline', text='I looked through the shop and could not find an item that meets every one of your requirements within your budget, so I am not buying anything.')
        if action:
            reply = self.call(*action)
            if not self.session.done:
                held_257 = 'decline refused: {}'.format(reply.refusals or str(reply.seen)[:200])
                self.jot(held_257)
                held_258 = self.unheard(reply.refusals)
                self.listen(held_258, 'The shop would not close the task and said:')
                self.after_change()
            else:
                self.jot('declined')
                return
        else:
            return

class Lookups:

    def resolve_lookups(self) -> None:
        for r in list((r for r in self.brief.needs if re.search(LOOKUP_RE, r['value'], re.I))):
            if not (not self.has_room(self.holdback() + 4) or self.timer.elapsed() > SEARCH_DEADLINE):
                text = 'The requirement: {}: {}\n\nThe full request:\n{}'.format(r['property'], r['value'], self.session.request_text)
                held_259 = self.oracle.ask_model('reference', LOOKUP_PROMPT, text, LOOKUP_SHAPE, ['wanted', 'brand'], 500)
                if not held_259:
                    held_259 = {}
                ref = held_259
                brand = str(ref.get('brand') or '').strip()
                if brand:
                    held_260 = self.kit.role_map.get('search')
                    if not held_260:
                        held_260 = {}
                    entry = held_260
                    shelf = None
                    if 'category' in entry.get('slots', {}):
                        if ref.get('category'):
                            shelf = self.kit.fit_arg(self.kit.tool_specs[entry['tool']]['properties'].get(entry['slots']['category']) or {}, str(ref['category']))
                    rows: list[Offer] = []
                    for scope in [shelf, None] if shelf else [None]:
                        if scope:
                            held_261 = self.kit.build_call('search', category=scope, text=brand)
                        else:
                            held_261 = None
                        if not held_261:
                            held_261 = self.kit.build_call('search', text=brand)
                        action = held_261
                        if not (not action or not self.has_room(self.holdback() + 3)):
                            found = self.absorb(self.call(*action).seen)
                            held_262 = list((l for l in found if flat(brand) in flat('{} {}'.format(l.brand, l.title))))
                            if not held_262:
                                held_262 = found
                            rows = held_262
                            if rows:
                                break
                        else:
                            break
                    clues = list((flat(c.get('value')) for c in ref.get('clues') or [] if isinstance(c, dict) and flat(c.get('value'))))
                    best, best_score, best_raw = (None, -1, '')
                    for listing, facts, raw in self.read_lookup_pages(sorted(rows, key=lambda l: -sum((1 for c in clues if c in flat(l.title))))[:8]):
                        if not flat(brand) not in flat(raw):
                            score = sum((1 for c in clues if c in flat(' '.join(('{} {}'.format(k, v) for k, v in facts.items())) + ' ' + listing.title)))
                            if best_score < score:
                                best, best_score, best_raw = (facts, score, raw)
                        else:
                            continue
                    if best:
                        wanted = roots(ref.get('wanted') or r['property'])
                        has = lambda facts: any((roots(re.sub('([a-z])([A-Z])', '\\1 \\2', str(k))) & wanted for k in facts))
                        if not has(best):
                            note = next((n for n in re.findall('"[^"]*(?:refer|judged from|another product|look at|see the)[^"]*"', best_raw, re.I)), '')
                            maker = next((w for w in re.findall('\\b([A-Z][a-z]{4,})\\b', note) if flat(w) != flat(brand)), '')
                            if maker:
                                action = self.kit.build_call('search', text=maker)
                                if action and self.has_room(self.holdback() + 3):
                                    found = self.absorb(self.call(*action).seen)
                                else:
                                    found = []
                                held_264 = flat(note).split()
                                said = set(held_264)
                                for listing, facts, raw in self.read_lookup_pages(sorted(found, key=lambda l: -len(said & set(flat(l.title).split())))[:8]):
                                    if has(facts):
                                        if maker.lower() in raw.lower():
                                            overlap = len(said & set(flat(listing.title + ' ' + ' '.join(map(str, facts.values()))).split()))
                                            if not has(best) or best_score < overlap:
                                                best, best_score = (facts, overlap)
                                held_265 = 'reference {!r}: seller note sends {!r} to {!r}: {}'.format(brand, r['property'], maker, 'found' if has(best) else 'not found')
                                self.jot(held_265)
                        label = max(best, key=lambda k: (len(roots(re.sub('([a-z])([A-Z])', '\\1 \\2', str(k))) & wanted), -len(str(k))), default=None)
                        if None is label or not roots(re.sub('([a-z])([A-Z])', '\\1 \\2', str(label))) & wanted:
                            held_266 = 'reference {!r}: no {!r} on its page'.format(brand, r['property'])
                            self.jot(held_266)
                            self.drop_lookup(r, 'property not on its page')
                            continue
                        value = str(best[label]).strip()
                        if not (ref.get('relation') == 'within' and isinstance(ref.get('amount'), (int, float)) and (first_number(value) is not None)):
                            r['value'] = value
                        else:
                            centre, amount, unit = (first_number(value), float(ref['amount']), str(ref.get('unit') or '').strip())
                            r['value'] = 'at least {:g} {}'.format(centre - amount, unit).strip()
                            self.brief.needs.append({**r, 'value': 'at most {:g} {}'.format(centre + amount, unit).strip()})
                        self.brief.revision = self.brief.revision + 1
                        held_267 = 'reference {!r} ({} clues): {} = {!r}'.format(brand, best_score, r['property'], value)
                        self.jot(held_267)
                    else:
                        held_263 = 'reference {!r}: no listing found for {!r}'.format(brand, r['property'])
                        self.jot(held_263)
                        self.drop_lookup(r, 'no listing found')
                        continue
                else:
                    self.drop_lookup(r, 'no seller named')
                    continue
            else:
                self.drop_lookup(r, 'no steps left to look it up')
                continue

    def drop_lookup(self, r: dict, why: str) -> None:
        if r in self.brief.needs:
            self.brief.needs.remove(r)
            self.brief.revision = self.brief.revision + 1
            held_268 = 'reference requirement {!r} left out ({})'.format(r['property'], why)
            self.jot(held_268)

    def read_lookup_pages(self, listings: list[Offer]) -> list[tuple[Offer, dict, str]]:
        pages: list[tuple[Offer, Any]] = []
        if self.kit.arg_type('facts', 'id') == 'array':
            if listings:
                if self.has_room(self.holdback() + 2):
                    action = self.kit.build_call('facts', id=list((l.id for l in listings)))
                    if action:
                        reply = self.call(*action)
                    else:
                        reply = None
                    items = list((x for x in (result_rows(reply.seen) if reply else []) if isinstance(x, dict)))
                    by_id = dict(((str(x.get(self.columns.get('id') if self.columns else 'id') or x.get('id')), x) for x in items))
                    pages = list(((l, by_id[l.id]) for l in listings if l.id in by_id))
        if not pages:
            for listing in listings[:4]:
                held_269 = self.kit.build_call('view', id=listing.id)
                if not held_269:
                    held_269 = self.kit.build_call('facts', id=listing.id)
                action = held_269
                if not (not action or not self.has_room(self.holdback() + 2)):
                    pages.append((listing, self.call(*action).seen))
                else:
                    break
        out = []
        for listing, obs in pages:
            listing.sheet = sheet_of(obs) or {}
            if None is not obs:
                raw = json.dumps(obs, ensure_ascii=False)
            else:
                raw = ''
            other = list((i for i in re.findall('p-[0-9a-f]{6,}', raw) if listing.id != i))
            if other:
                if re.search('redirect|moved|replac|instead|now (?:at|listed)|see |superseded|go to', raw, re.I):
                    if self.has_room(self.holdback() + 2):
                        held_270 = self.kit.build_call('view', id=other[0])
                        if not held_270:
                            held_270 = self.kit.build_call('facts', id=other[0])
                        action = held_270
                        if action:
                            again = self.call(*action).seen
                            held_271 = 'reference page {} sends to {}'.format(listing.id, other[0])
                            self.jot(held_271)
                            out.append((listing, sheet_of(again) or {}, raw + json.dumps(again, ensure_ascii=False)))
                            continue
            out.append((listing, listing.sheet, raw))
        return out

class LineWork:

    def line_shopper(self, product: dict) -> 'Agent':
        sub = Assistant.__new__(Assistant)
        sub.timer, sub.oracle, sub.session, sub.kit, sub.log_lines = (self.timer, self.oracle, self.session, self.kit, self.log_lines)
        sub.reset_state()
        sub.columns, sub.heard_before, sub.as_line = (self.columns, self.heard_before, True)
        rec = sub.brief
        rec.item = str(product.get('item') or '').strip()
        rec.noun = str(product.get('noun') or '').strip() or (rec.item.split() or [''])[-1]
        rec.category = str(product.get('category') or '').strip() or rec.item
        for q in product.get('requirements') or []:
            if isinstance(q, dict):
                if str(q.get('property') or '').strip():
                    if str(q.get('value') or '').strip():
                        rec.add(q['property'], q['value'], bool(q.get('exclude')), '', 'request')
        request = self.session.request_text
        for r in rec.needs:
            held_272 = re.escape(r['value'])
            negated = "\\b(?:not|no|never|without|except|other than|anything but|isn'?t|aren'?t)\\s+(?:a |an |the |in |be |made (?:of|from) )?{}(?![\\w:])".format(held_272)
            if not r['exclude']:
                if re.search(negated, request, re.I):
                    r['exclude'] = True
                    held_273 = 'basket {!r}: {} rules out {}'.format(rec.item, r['property'], r['value'])
                    self.jot(held_273)
        rec.needs = list((r for r in rec.needs if not only_compared(r['value'], request)))
        count = product.get('quantity')
        if isinstance(count, int) and 1 <= count <= 50:
            rec.units = int(count)
        else:
            rec.units = 1
        rec.ceiling = self.brief.ceiling
        rec.stock_only = self.brief.stock_only
        if 'cheapest' == self.brief.priority:
            rec.priority = 'cheapest'
        else:
            rec.priority = 'none'
        rec.message_cap = 0
        rec.seed_searches = list((x for x in product.get('searches') or [] if isinstance(x, str) and x.strip()))[:4] or [rec.item]
        return sub

    def shop_lines(self) -> None:
        n = len(self.lines)
        parts: list[tuple[Assistant, Offer | None]] = []
        if None is not self.brief.message_cap:
            messages = self.brief.message_cap
        else:
            messages = n
        subs = list((self.line_shopper(product) for product in self.lines))
        for sub in subs:
            sub.fellow_lines = subs
        for i, product in enumerate(self.lines):
            sub = subs[i]
            if 0 < messages:
                sub.brief.message_cap = -(-messages // (n - i))
            else:
                sub.brief.message_cap = 0
            messages -= sub.brief.message_cap
            hold = n + CLOSING_STEPS + 4
            sub.stop_step = self.session.step + max(4, (self.session.steps_left() - hold) // (n - i))
            held_274 = 'basket part {}/{}: {!r} x{} reqs={!r} until step {}'.format(i + 1, n, sub.brief.item, sub.brief.units, sub.brief.lines(), sub.stop_step)
            self.jot(held_274)
            sub.resolve_lookups()
            sub.discover()
            sub.interview()
            sub.confirm_pool()
            pick = sub.pick()
            if None is pick:
                pick = next((l for l in sub.ordered() if None is not l.price), None)
                if None is not pick:
                    held_275 = 'basket part {}: nothing confirmed, taking {} (breaks nothing known)'.format(i + 1, pick.id)
                    self.jot(held_275)
            self.columns = sub.columns or self.columns
            held_276 = 'basket part {}: pick {} at {}'.format(i + 1, pick.id if pick else None, pick.price if pick else None)
            self.jot(held_276)
            parts.append((sub, pick))
            if None is pick:
                break
        if not (n > len(parts) or any((None is pick for _, pick in parts))):
            for attempt in range(3):
                if self.fit_budget(parts):
                    lines = list(((pick.id, sub.brief.units) for sub, pick in parts))
                    if self.fill_basket(lines):
                        action = self.kit.build_call('checkout')
                        if not (not action or not self.has_room(0)):
                            held_277 = self.call(*action)
                            reply = self.second_step(held_277)
                            if self.session.done:
                                held_278 = 'ordered basket {}'.format(lines)
                                self.jot(held_278)
                                return
                            held_279 = 'basket checkout refused: {}'.format(reply.refusals or str(reply.seen)[:200])
                            self.jot(held_279)
                            refusal = self.unheard(reply.refusals)
                            self.listen(refusal, 'The shop turned the order down and said:')
                            names_of = lambda part: set((x for x in (flat(part.brief.noun), flat(part.brief.item), flat(part.brief.category)) if x))
                            versions = list((sub.brief.revision for sub, _ in parts))
                            for message in refusal:
                                words = roots(message)
                                named = list((sub for sub, _ in parts if any((roots(x) and roots(x) <= words for x in names_of(sub)))))
                                for sub in named or list((sub for sub, _ in parts)):
                                    sub.listen([message], 'The shop turned the order down and said:')
                            for k, (sub, pick) in enumerate(parts):
                                sub.brief.ceiling = self.brief.ceiling
                                if versions[k] != sub.brief.revision:
                                    sub.check_offers([pick])
                                if versions[k] != sub.brief.revision and sub.status(pick) == 'no' or (self.has_room(3) and (not sub.reread(pick))):
                                    pick.sold_out = pick.sold_out or False is pick.stock
                                    again = sub.pick()
                                    held_280 = 'basket part {}: {} no longer fits, now {}'.format(k + 1, pick.id, again.id if again else None)
                                    self.jot(held_280)
                                    if not None is again:
                                        parts[k] = (sub, again)
                                    else:
                                        self.refuse_order()
                                        return
                        else:
                            return
                    else:
                        self.refuse_order()
                        return
                else:
                    self.jot('basket: no choice of fits stays within the total budget')
                    self.refuse_order()
                    return
        else:
            self.jot('basket: a product has no fitting listing')
            self.refuse_order()
            return

    def fit_budget(self, parts: list) -> bool:
        budget = self.brief.ceiling
        for _ in range(12):
            total = sum(((pick.price or 0) * sub.brief.units for sub, pick in parts))
            if not (None is budget or budget + 1e-09 >= total):
                best = None
                for k, (sub, pick) in enumerate(parts):
                    cheaper = list((l for l in sub.ordered(confirmed=True) if None is not l.price and None is not pick.price and (pick.price > l.price)))
                    if cheaper:
                        low = min(cheaper, key=lambda l: l.price)
                        saving = (pick.price - low.price) * sub.brief.units
                        if None is best or best[0] < saving:
                            best = (saving, k, low)
                if not None is best:
                    parts[best[1]] = (parts[best[1]][0], best[2])
                else:
                    return False
            else:
                return True
        return False

    def fill_basket(self, lines: list[tuple[str, int]]) -> bool:
        if not self.kit.sets_whole_basket('cart_add'):
            wanted = dict(lines)
            for pid in list((p for p in self.basket_now if p not in wanted)):
                action = self.kit.build_call('cart_remove', quantity=self.basket_now[pid], id=pid)
                if action:
                    if self.has_room(2):
                        self.call(*action)
                self.basket_now.pop(pid, None)
            for pid, count in lines:
                if not self.basket_now.get(pid) == count:
                    action = self.kit.build_call('cart_add', quantity=count, id=pid)
                    if not (not action or not self.has_room(1)):
                        reply = self.call(*action)
                        if not is_failure(reply.seen):
                            self.basket_now[pid] = count
                        else:
                            held_282 = 'basket add {} failed: {}'.format(pid, str(reply.seen)[:160])
                            self.jot(held_282)
                            return False
                    else:
                        return False
                else:
                    continue
            return True
        else:
            action = self.kit.build_call('cart_add', id=lines)
            if not (not action or not self.has_room(1)):
                reply = self.call(*action)
                if not is_failure(reply.seen):
                    self.basket_now = dict(lines)
                    return True
                else:
                    held_281 = 'basket cart failed: {}'.format(str(reply.seen)[:200])
                    self.jot(held_281)
                    return False
            else:
                return False

class Assistant(LineWork, Lookups, Closer, Interviewer, Checker, Finder, Catalog, Basics):
    pass

def salvage(agent: Assistant) -> None:
    world = agent.session
    if not (world.done or world.steps_left() <= 0):
        held = list((pid for pid in agent.basket_now if pid in agent.offers and agent.within_reach(agent.offers[pid]) and (agent.status(agent.offers[pid]) != 'no')))
        if not (1 == len(agent.basket_now) and held and agent.kit.build_call('checkout')):
            best = agent.ordered(confirmed=True)
            if best and (not agent.basket_now) and (world.steps_left() >= 2) and agent.kit.build_call('cart_add', quantity=agent.brief.units, id=best[0].id):
                agent.call(*agent.kit.build_call('cart_add', quantity=agent.brief.units, id=best[0].id))
                if agent.kit.build_call('checkout'):
                    agent.call(*agent.kit.build_call('checkout'))
                return
            elif agent.kit.build_call('decline', text='I could not find an item that meets every one of your requirements, so I am not buying anything.'):
                agent.call(*agent.kit.build_call('decline', text='I could not find an item that meets every one of your requirements, so I am not buying anything.'))
        else:
            agent.call(*agent.kit.build_call('checkout'))
            return
    else:
        return

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    agent = Assistant(problem_data)
    try:
        agent.run()
        if not agent.session.done:
            if agent.session.steps_left() > 0:
                agent.jot('the finish ended without an order or a decline: rescue')
                salvage(agent)
    except Exception as exc:
        held_283 = 'error: {}: {}'.format(type(exc).__name__, exc)
        agent.jot(held_283)
        try:
            salvage(agent)
        except Exception as again:
            held_284 = 'rescue error: {}: {}'.format(type(again).__name__, again)
            agent.jot(held_284)
    print(json.dumps({'s11v3_notes': agent.log_lines[-80:], 'model_calls': agent.oracle.call_count}, ensure_ascii=False)[:20000], flush=True)
    held_285 = list(({'note': n} for n in agent.log_lines)) + agent.session.trail
    return held_285
__all__ = ['agent_main']
