import json
import re
import threading
import time
from src.agent.proxy_client import ProxyClient
GLM = 'z-ai/glm-5.2'
LUNA = 'openai/gpt-6-luna'
KIMI = 'moonshotai/kimi-k2.6'
MODEL_LIMIT = 262.0
WRAP_AT = 222.0
HURRY_AT = 168.0
PARALLEL_CALLS = 3
OPEN_PER_TURN = 30
COMPARE_SIZE = 5
TABLE_LIMIT = 16
VARIANTS_PER_PAGE = 4
PRICE_BANDS = (1.0, 0.6, 0.35, 0.2, 0.12, 0.07)
QUIET_LINES = ('market_notice', 'acknowledge', 'other')
JOINS = '(\\s*(?:&|/|,| and )\\s*)'

def squash(text):
    return ' '.join(str(text or '').casefold().split())

def tokens(text):
    return set(re.findall('[^\\W_]+', squash(text)))

def number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search('-?\\d[\\d,]*(?:\\.\\d+)?', str(value or ''))
    if not match:
        return None
    try:
        return float(match.group(0).replace(',', ''))
    except ValueError:
        return None

def figures(text):
    found = []
    for match in re.finditer('(\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?|\\d+(?:\\.\\d+)?)\\s*-?\\s*([°]?[^\\W\\d_]*[\\"\']?)', squash(text)):
        value = number(match.group(1))
        if value is not None:
            found.append((value, match.group(2)))
    return found

def unit_letters(unit):
    return ''.join((ch for ch in str(unit or '').lower() if ch.isalpha()))

def unit_like(one, other):
    a, b = (unit_letters(one), unit_letters(other))
    if not a or not b:
        return not a and (not b)
    a, b = (a[:-1] if len(a) > 2 and a.endswith('s') else a, b[:-1] if len(b) > 2 and b.endswith('s') else b)
    if a[0] != b[0]:
        return False
    short, rest = (a, iter(b)) if len(a) <= len(b) else (b, iter(a))
    return all((ch in rest for ch in short))

def contains(needle, hay):
    one, other = (squash(needle), squash(hay))
    if not one:
        return False
    return one in other or one.replace(' ', '') in other.replace(' ', '')

def plain(text):
    return ' '.join(re.findall('[a-z0-9]+(?:\\.[0-9]+)?', squash(text)))

def fact_says(fact, value, drop):
    words = {singular_of(w) for w in tokens(fact)} - drop
    if words == value:
        return True
    text = str(fact or '')
    last = [singular_of(w) for w in re.findall('[^\\W_]+', squash(text))][-1:]
    return False

class Mailbox:

    def __init__(self):
        self.items = []
        self.cond = threading.Condition()

    def put(self, item):
        with self.cond:
            self.items.append(item)
            self.cond.notify()

    def get(self, timeout):
        with self.cond:
            if not self.items:
                self.cond.wait(timeout)
            return (True, self.items.pop(0)) if self.items else (False, None)

def similarity(a, b):
    pa = {a[i:i + 2] for i in range(len(a) - 1)} or {a}
    pb = {b[i:i + 2] for i in range(len(b) - 1)} or {b}
    return 2 * len(pa & pb) / (len(pa) + len(pb))

def quoted(quote, page):
    text = str(quote or '').strip()
    if not text:
        return True
    hay = plain(page)

    def one(segment):
        segment = segment.strip(' .')
        if not plain(segment):
            return True
        parts = [segment] + ([re.split('[:=]', segment, maxsplit=1)[1]] if re.search('[:=]', segment) else [])
        return any((plain(p) and plain(p) in hay for p in parts))
    if one(text):
        return True
    bare = re.sub('\\s*\\([^)]*\\)', '', text)
    if bare != text and plain(bare) and one(bare):
        return True
    segments = [x for x in re.split('\\s*;\\s*|\\.\\.\\.', text) if plain(x)]
    return len(segments) > 1 and all((one(x) for x in segments))

def plural_of(word):
    low = word.lower()
    if low.endswith('s') and (not low.endswith('ss')):
        return word
    if re.search('[^aeiou]y$', low):
        return word[:-1] + 'ies'
    return word + ('es' if low.endswith(('s', 'sh', 'ch', 'x', 'z')) else 's')

def singular_of(word):
    low = word.lower()
    if low.endswith('ies') and len(low) > 4:
        return word[:-3] + 'y'
    if low.endswith(('sses', 'shes', 'ches', 'xes', 'zes')):
        return word[:-2]
    return word[:-1] if low.endswith('s') and (not low.endswith('ss')) and (len(low) > 3) else word

def shelf_spellings(name, limit=6):
    text = ' '.join(str(name or '').split())
    if not text:
        return []
    pieces = re.split(JOINS, text)
    slots = [i for i, piece in enumerate(pieces) if piece.strip() and (not re.fullmatch(JOINS, piece))][:4]
    out = [text]
    for mode in (plural_of, singular_of):
        built = list(pieces)
        for i in slots:
            part = built[i].split(' ')
            part[-1] = mode(part[-1])
            built[i] = ' '.join(part)
        out.append(''.join(built))
    out.append(' '.join((word[:1].upper() + word[1:] for word in out[1].split(' '))))
    return list(dict.fromkeys(out))[:limit]

def same_shelf(name, label):
    return bool(squash(label)) and squash(label) in {squash(s) for s in shelf_spellings(name, 12)}

def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)

def tool(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description, 'parameters': {'type': 'object', 'properties': properties, 'required': required}}}

class Clock:

    def __init__(self):
        self.start = time.monotonic()

    def elapsed(self):
        return time.monotonic() - self.start

class Seat:

    def __init__(self, system, spec, models, tokens_out, wait, hedge, check=None):
        self.system, self.spec, self.models, self.tokens_out, self.wait, self.hedge, self.check = (system, spec, models, tokens_out, wait, hedge, check)
        self.name = spec['function']['name']

def tool_args(reply, name):
    choice = ((reply or {}).get('choices') or [None])[0] if isinstance(reply, dict) else None
    message = choice.get('message') if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        return None
    for call in message.get('tool_calls') or []:
        function = call.get('function') if isinstance(call, dict) else None
        if not isinstance(function, dict) or function.get('name') not in (name, None):
            continue
        raw = function.get('arguments')
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            continue
        if isinstance(args, dict) and args:
            return args
    return None

class Net:

    def __init__(self, clock):
        self.clock = clock
        self.gate = threading.BoundedSemaphore(PARALLEL_CALLS)
        self.lock = threading.Lock()
        self.spent = 0.0

    def post(self, path, body, timeout):
        client = ProxyClient(timeout=max(3, int(timeout)), max_retries=1)
        for attempt in range(3):
            try:
                if hasattr(client, 'post_verbose'):
                    result = client.post_verbose(path, json_data=body)
                    data, error = (getattr(result, 'data', None), dict(getattr(result, 'error', None) or {}))
                else:
                    data, error = (client.post(path, json_data=body), {})
            except Exception:
                data, error = (None, {})
            if data is not None:
                return (data, error)
            busy = error.get('status') == 429 and 'capacity' not in str(error.get('body') or '').lower()
            if busy and attempt < 2 and (MODEL_LIMIT - self.clock.elapsed() > 25):
                time.sleep(1.5 * (attempt + 1))
                continue
            return (None, error)
        return (None, {})

    def spend(self, reply):
        usage = reply.get('usage') if isinstance(reply, dict) else None
        cost = usage.get('cost') if isinstance(usage, dict) else None
        if isinstance(cost, (int, float)) and (not isinstance(cost, bool)):
            with self.lock:
                self.spent += float(cost)

    def model(self, model, seat, user, timeout):
        body = {'model': model, 'messages': [{'role': 'system', 'content': seat.system}, {'role': 'user', 'content': user}], 'tools': [seat.spec], 'tool_choice': {'type': 'function', 'function': {'name': seat.name}}, 'max_tokens': seat.tokens_out, 'temperature': 0, 'reasoning': {'enabled': False}}
        if not self.gate.acquire(timeout=max(0.5, timeout / 2)):
            return None
        try:
            reply, error = self.post('/inference/chat/completions', body, timeout)
            if reply is None and error.get('status') in (400, 422) and ('tool_choice' in str(error.get('body') or '')):
                reply, error = self.post('/inference/chat/completions', dict(body, tool_choice='required'), timeout)
        finally:
            self.gate.release()
        self.spend(reply)
        return tool_args(reply, seat.name)

    def ask(self, seat, user, deadline=None):
        limit = min(deadline or MODEL_LIMIT, MODEL_LIMIT)
        answers = Mailbox()

        def run(model, seconds):
            try:
                answer = self.model(model, seat, user, seconds)
                if answer is not None and seat.check is not None:
                    answer = seat.check(answer)
            except Exception:
                answer = None
            answers.put(answer)
        launched, pending, next_launch = (0, 0, 0.0)
        while True:
            left = limit - self.clock.elapsed()
            if left < 4:
                return None
            now = time.monotonic()
            if launched < len(seat.models) and (pending == 0 or now >= next_launch):
                seconds = min(seat.wait, left - 2)
                threading.Thread(target=run, args=(seat.models[launched], seconds), daemon=True).start()
                launched, pending, next_launch = (launched + 1, pending + 1, now + seat.hedge)
            if pending == 0:
                return None
            got, answer = answers.get(0.25)
            if not got:
                continue
            pending -= 1
            if answer is not None:
                return answer

class Session:

    def __init__(self, problem):
        env = problem.get('environment') if isinstance(problem.get('environment'), dict) else {}
        self.view = env.get('policy_view') if isinstance(env.get('policy_view'), dict) else {}
        self.binding = env.get('binding') if isinstance(env.get('binding'), dict) else {}
        self.label = str(problem.get('problem_id') or 'problem')
        self.max_turns = int(number(self.view.get('max_steps')) or 30)
        self.per_turn = max(1, min(int(number(self.view.get('max_calls_per_turn')) or 16), 16))
        self.schemas = {}
        for item in self.view.get('tools') or []:
            function = item.get('function') if isinstance(item, dict) else None
            if isinstance(function, dict) and isinstance(function.get('name'), str):
                self.schemas[function['name']] = function.get('parameters') or {}
        self.turn, self.gen, self.done, self.dead = (0, 0, False, False)
        self.client = ProxyClient(timeout=65, max_retries=2)

    def left(self):
        return self.max_turns - self.turn

    def fit(self, name, args):
        declared = (self.schemas.get(name) or {}).get('properties') or {}
        return {key: value for key, value in args.items() if key in declared and value not in (None, '')} if declared else args

    def send(self, calls):
        calls = [(name, self.fit(name, args)) for name, args in calls if name in self.schemas][:self.per_turn]
        if not calls or self.done or self.dead or (self.left() <= 0):
            return None
        for attempt, step in enumerate((1, 1, 2)):
            turn = self.turn + step
            group = f'{self.label}-t{turn}-g{self.gen}'
            envelope = {**self.binding, 'call_id': group, 'idempotency_key': group, 'turn': turn, 'calls': [{'call_id': f'{group}-c{i}', 'action': {'name': n, 'args': a}} for i, (n, a) in enumerate(calls, 1)]}
            try:
                reply = self.client.post('/environment/call', json_data=envelope)
            except Exception:
                reply = None
            if isinstance(reply, dict) and isinstance(reply.get('calls'), list):
                count = reply.get('solver_turn_count')
                self.turn = count if isinstance(count, int) and (not isinstance(count, bool)) else turn
                return reply
            self.gen += 1
            time.sleep(0.8)
        return None
MUST = {'type': 'object', 'properties': {'property': {'type': 'string', 'description': "what the requirement is about, in the shopper's word for it when they give one (colour, material, length, brand, scent...)"}, 'value': {'type': 'string', 'description': 'the value exactly as the shopper wrote it'}, 'kind': {'type': 'string', 'enum': ['figure', 'word'], 'description': 'figure for a number with a unit, word otherwise'}, 'unit': {'type': 'string', 'description': 'the unit as written for a figure; empty otherwise'}, 'words': {'type': 'string', 'description': "the value as a seller writes it in a title or option ('navy', '500ml', '12pcs')"}, 'also': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 2, 'description': 'other spellings of this very same value (grey/gray), and for a figure the same amount in the other unit sellers use (0.5 g/500mg, 40 cm/400mm, 25.4 cm/10 inch); never a nearby or related value; empty when none'}}, 'required': ['property', 'value', 'kind', 'unit', 'words', 'also']}
WISH = {'type': 'object', 'properties': {'property': {'type': 'string'}, 'value': {'type': 'string'}}, 'required': ['property', 'value']}
RECORD_SYSTEM = "You keep the order record for a shopping assistant that buys exactly one item for a shopper in an online store. You read the shopper's request and every line they sent since, and write the whole record as it stands after their latest line.\nWrite only what the shopper said. A later line overrides an earlier one: a new budget replaces the old; a requirement they change is replaced (the old value disappears); one they drop disappears; one they add is added. A line that only repeats an old requirement never undoes a change they made explicitly (red swapped for green, then a line naming red again): the changed value stays.\nFields:\n- product: the item noun as they wrote it, without colour, figures or purpose. category: the store shelf for it. When the shopper names the exact catalog category, copy that name exactly; otherwise write their item phrase the way a store names a shelf (capitalised, plural: 'a garden hose' -> 'Garden Hoses').\n- budget: the latest most they will pay, a bare number.\n- musts: every firm requirement on the item itself, one per entry (in '<item> with <value> as its <property>' the value is the part after 'with'): what they say it must / has to / needs to be, what they call non-negotiable, a value a later line says it has to have (an answer to your question), and the wish they say matters more once they say so. Never the budget, the stock, the price or the category. Never the objective (the most / greatest / largest of something goes in objective). Never a wish they only hope for.\n- wishes: when the shopper names two things and doubts one listing can have both (or a second thing they call a nice extra), the two things, each as property and value - even when they word them as needs (two firm-sounding needs followed by doubt that one item has both). They are not musts while the shopper has not said which matters more. Empty when there are no such two things. Wishes are properties of the item (colour, brand, material, form...); two purposes or uses are never wishes - they go in purpose. 'be a <Name>' or 'a <Name>' with a capitalised name is the brand.\n- doubt: the shopper's sentence that doubts one listing can have both of two things, copied exactly; empty when there is none.\n- wish_chosen: 'first' or 'second' once a line of theirs says which of the two wishes matters more (that one is then also in musts); otherwise empty.\n- prefer: a single maker or option they would favour 'if it fits' / 'first choice' - a soft preference, never a must.\n- purpose: what the purchase is for or why, in their words (a gift, replacing a broken one, restocking, matching something, a planned setup, hiding stains...); empty when not said. A purpose is not a requirement by itself.\n- purposes: each separate purpose or use they give, one entry each, copied as they wrote it (two purpose sentences give two entries); empty when none.\n- objective: the quantity they want the most (or least) of among qualifying items ('most pieces', 'largest volume', 'greatest length', 'longest warranty', 'highest power rating', 'most storage', 'most RAM', 'highest resolution') as property and direction; property empty when they gave none.\n- survey: how many listings they ask you to look at before choosing; 0 when they did not ask.\n- justify: the details they ask the order to be justified or explained with when it is placed (e.g. warranty length, price, stock, brand); empty when they did not ask for a justification. The objective is not a justification.\n- shortlist: product_id::sku keys they limit the choice to; empty when open.\n- latest_line: what their most recent line does: 'request' (no line yet), 'answer' (answers your question), 'budget_change', 'requirement_change' (replaces a value), 'requirement_add', 'requirement_drop', 'market_notice' (asks if an item sold out or got pricier), 'acknowledge' (thanks or restates), or 'other'. A line may do several things: pick the most specific change.\n- asks_confirmation: true when their latest line asks you to confirm something.\n- line_changes: what their most recent line itself changes, one entry each (a new budget, a requirement added, replaced or dropped, an answer to your question that adds a requirement or says which wish matters more), with its new value. Empty when that line changes nothing - a question or notice about an item's stock or price, a thanks, or a repeat of what they said before. Judge the most recent line alone, even when it also repeats old requirements.\n- dropped: the musts of the earlier record that the latest line removed or replaced, each as written there.\n- searches: three short store searches written as a seller titles the item: the item noun with the must values (two to five words each)."
RECORD_TOOL = tool('record', "The shopper's order record after their latest line.", {'product': {'type': 'string'}, 'category': {'type': 'string'}, 'budget': {'type': ['number', 'null']}, 'musts': {'type': 'array', 'items': MUST}, 'wishes': {'type': 'array', 'items': WISH, 'maxItems': 2}, 'doubt': {'type': 'string'}, 'wish_chosen': {'type': 'string', 'enum': ['', 'first', 'second']}, 'prefer': {'type': 'array', 'items': {'type': 'string'}}, 'purpose': {'type': 'string'}, 'purposes': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 3}, 'objective': {'type': 'object', 'properties': {'property': {'type': 'string'}, 'direction': {'type': 'string', 'enum': ['most', 'least']}}, 'required': ['property', 'direction']}, 'survey': {'type': 'integer'}, 'justify': {'type': 'array', 'items': {'type': 'string'}}, 'shortlist': {'type': 'array', 'items': {'type': 'string'}}, 'latest_line': {'type': 'string', 'enum': ['request', 'answer', 'budget_change', 'requirement_change', 'requirement_add', 'requirement_drop', 'market_notice', 'acknowledge', 'other']}, 'asks_confirmation': {'type': 'boolean'}, 'dropped': {'type': 'array', 'items': {'type': 'string'}}, 'line_changes': {'type': 'array', 'items': {'type': 'object', 'properties': {'what': {'type': 'string', 'enum': ['budget', 'requirement_add', 'requirement_replace', 'requirement_drop', 'answer']}, 'new_value': {'type': 'string'}}, 'required': ['what', 'new_value']}}, 'searches': {'type': 'array', 'items': {'type': 'string'}}}, ['product', 'category', 'budget', 'musts', 'wishes', 'doubt', 'wish_chosen', 'prefer', 'purpose', 'purposes', 'objective', 'survey', 'justify', 'shortlist', 'latest_line', 'asks_confirmation', 'line_changes', 'dropped', 'searches'])
JUDGE_SYSTEM = "You choose the one variant a shopping assistant buys for a shopper. You get the shopper's request, their lines since, the record of their requirements, what happened (a lost item, a changed requirement), and a table of candidate variants that are in stock and within the budget, each with its page's brand field, shelf path, title, facts and this variant's own options.\nA variant fits only when its own page states every must with the shopper's value. Treat each candidate as not fitting until its page shows that.\nReading a page:\n- This variant's options say which variant arrives; when they state the property, they decide it. A colour word written as its own word in this variant's options states that colour, even inside a variant name ('6 pcs red star' is red); a colour fused by hyphens into a code with a figure ('2m-red-usb') states no colour. When this variant's options name a colour, a shade or a colour combination ('royal blue', 'peach', 'two-tone'), that is this variant's colour, and a colour in the title describes the listing, not this variant. A title listing several values ('5/10/15m') advertises the listing's variants, not this one. In a dimension list such as '40*30*20cm' only the last figure, the one written against the unit, is a stated length (20 cm).\n- A fact whose value is 'not specified', 'none', 'n/a' or empty states nothing; the options and the title decide then.\n- A fact named for the property states it. A value only the title states, plainly and once, also states it (weaker than options or facts).\n- A list of the values a listing comes in does not state this variant's value. A range the item covers ('fits 20-40 cm') holds the values inside it.\n- A blend or two things joined ('wool silk', 'blue/orange') is neither part alone. A shade of the named colour is that colour (dark blue is blue). A figure inside a model code states nothing. A different value is a different value (1500 W is not 1250 W); the same amount in another unit is the same value. A material written in its common trade short form, or a specific kind of the named material, is that material ('PU' is PU leather; ABS or PVC is plastic).\n- The shelf the shopper named decides the kind of item: a listing whose shelf path has that label fits the kind; one off that shelf does not while any on-shelf candidate fits.\n- The brand field names the maker; a required maker must be in it (or in the title when the field is empty).\nFor the objective (the quantity the shopper wants the most or least of), read each candidate's value from its page: this variant's options, else a fact, else the title; give the quote and the number in the one unit named in the request. The number is the objective's own quantity (a volume in ml or l, a length in cm or m); a count or size that a must requires is not the objective. 0 with an empty quote when the page states none. When variants differ only by a size number and the objective is a length or a size, that size number is the measure.\nAnswer with the choose tool: assessments for the likely winner and its closest rivals (at most 10), each with a line per must (page value, exact quote of at most ten words, verdict met/unmet/unknown), the objective quote and number, and fits; then ranking (fitting candidates, best objective first, then the closest non-fitting), none_fit, searches (only when none fit: three searches of three or four words for listings that could meet every must) and winner."
JUDGE_TOOL = tool('choose', 'The choice among the candidates.', {'assessments': {'type': 'array', 'maxItems': 10, 'items': {'type': 'object', 'properties': {'key': {'type': 'string', 'description': 'product_id::sku as in the table'}, 'lines': {'type': 'array', 'items': {'type': 'object', 'properties': {'must': {'type': 'string'}, 'page_value': {'type': 'string'}, 'quote': {'type': 'string'}, 'verdict': {'type': 'string', 'enum': ['met', 'unmet', 'unknown']}}, 'required': ['must', 'page_value', 'quote', 'verdict']}}, 'objective_quote': {'type': 'string'}, 'objective_value': {'type': 'number'}, 'fits': {'type': 'boolean'}}, 'required': ['key', 'lines', 'objective_quote', 'objective_value', 'fits']}}, 'ranking': {'type': 'array', 'items': {'type': 'string'}}, 'none_fit': {'type': 'boolean'}, 'searches': {'type': 'array', 'items': {'type': 'string'}}, 'winner': {'type': 'string'}}, ['assessments', 'ranking', 'none_fit', 'searches', 'winner'])

def _record_check(answer):
    if not isinstance(answer.get('musts'), list) or 'category' not in answer:
        return None
    return answer

def _judge_check(answer):
    return answer if isinstance(answer.get('assessments'), list) else None
RECORD = Seat(RECORD_SYSTEM, RECORD_TOOL, (GLM, LUNA, GLM), 2400, 40.0, 14.0, _record_check)
DOUBT_SYSTEM = "A shopper wrote a request to a shopping assistant. Find the sentence where the shopper doubts that one listing can have both of two things they want - doubt that both fit the price, unsure the pair exists together - or where they add a second thing only as a bonus on top of a first (they want X, and Y too if at all possible). Those two things are often worded as firm needs.\nAnswer with the doubt tool: sentence (copied exactly; empty when there is no such sentence) and the two things, each as property and value copied exactly from the request (the value as written, e.g. 'gummy', 'BIOGESIC')."
DOUBT_TOOL = tool('doubt', "The shopper's doubt that one listing has both of two things.", {'sentence': {'type': 'string'}, 'things': {'type': 'array', 'maxItems': 2, 'items': WISH}}, ['sentence', 'things'])
DOUBT = Seat(DOUBT_SYSTEM, DOUBT_TOOL, (LUNA, GLM, LUNA), 400, 25.0, 10.0, lambda a: a if isinstance(a.get('things'), list) else None)
JUDGE = Seat(JUDGE_SYSTEM, JUDGE_TOOL, (GLM, LUNA, GLM), 4200, 60.0, 30.0, _judge_check)

class Conditions:

    def __init__(self, request):
        self.request = request
        self.record = {}
        self.lines = []
        self.heard = 0
        self.asked = []
        self.disclosed = set()
        self.retried = set()
        self.owed = []
        self.changed = False
        self.answer_index = None
        self.had_choice = False
        self.current = -1
        self.abandoned = set()

    def budget(self):
        return number(self.record.get('budget'))

    def musts(self):
        return [m for m in self.record.get('musts') or [] if isinstance(m, dict) and str(m.get('value') or '').strip()]

    def wishes(self):
        return [w for w in self.record.get('wishes') or [] if isinstance(w, dict) and str(w.get('value') or '').strip()][:2]

    def product(self):
        return str(self.record.get('product') or 'item').strip() or 'item'

    def category(self):
        return getattr(self, 'shelf_label', '') or str(self.record.get('category') or '').strip()

    def objective(self):
        obj = self.record.get('objective') if isinstance(self.record.get('objective'), dict) else {}
        return (str(obj.get('property') or '').strip(), str(obj.get('direction') or 'most'))

    def survey(self):
        return int(number(self.record.get('survey')) or 0)

    def justify(self):
        return [str(x) for x in self.record.get('justify') or [] if str(x).strip()]

    def shortlist(self):
        return {re.sub('\\s+', '', str(k)) for k in self.record.get('shortlist') or [] if '::' in str(k)}

class Store:

    def __init__(self):
        self.rows = {}
        self.order = []
        self.found_by = {}
        self.pages = {}
        self.opened = []
        self.live = {}
        self.cart = []
        self.lost = set()
        self.first_add = None
        self.searched = set()

    def note(self, pid, sku, in_stock, price):
        if pid and sku and isinstance(in_stock, bool):
            self.live[f'{pid}::{sku}'] = (in_stock, price)

    def page_text(self, key):
        pid, sku = key.split('::', 1)
        page = self.pages.get(pid) or {}
        variant = next((v for v in page.get('variants') or [] if str(v.get('sku')) == sku), {})
        return ' '.join([str(page.get('title') or ''), dumps(page.get('facts') or {}), str(variant.get('options') or ''), str(page.get('brand') or '')])

    def options(self, key):
        pid, sku = key.split('::', 1)
        page = self.pages.get(pid) or {}
        return str(next((v.get('options') for v in page.get('variants') or [] if str(v.get('sku')) == sku), '') or '')

class Episode:

    def __init__(self, problem):
        self.clock = Clock()
        self.net = Net(self.clock)
        self.session = Session(problem)
        self.request = str(self.session.view.get('query') or '')
        self.cond = Conditions(self.request)
        self.store = Store()
        self.dialogue = []
        self.judged = {}
        self.backups = []
        self.fit_count = 0
        self.closest = None
        self.evidence = {}

    def send(self, calls):
        calls = [(name, self.session.fit(name, args)) for name, args in calls if name in self.session.schemas][:self.session.per_turn]
        reply = self.session.send(calls)
        if reply is None:
            return []
        self.dialogue.append({'role': 'environment', 'turn': self.session.turn, 'calls': [{'name': n, 'args': a} for n, a in calls]})
        out = []
        for index, (name, args) in enumerate(calls[:self.session.per_turn]):
            item = reply['calls'][index] if index < len(reply['calls']) and isinstance(reply['calls'][index], dict) else {}
            step = item.get('observation') if isinstance(item.get('observation'), dict) else {}
            self.session.done = self.session.done or bool(step.get('done'))
            body = step.get('observation') if isinstance(step.get('observation'), dict) else {}
            if not step.get('error') and (not body.get('error')):
                self.learn(name, args, body)
            out.append(body if not (step.get('error') or body.get('error')) else {'error': step.get('error') or body.get('error')})
        said = (reply.get('user_message') or {}).get('content') if isinstance(reply.get('user_message'), dict) else None
        for line in str(said or '').split('\n'):
            if line.strip():
                self.cond.lines.append(line.strip())
        return out

    def learn(self, name, args, body):
        store = self.store
        if name in ('search', 'filter'):
            source = f"{name}:{args.get('query') or args.get('category') or ''}"
            for row in body.get('results') or []:
                if not isinstance(row, dict) or not row.get('product_id'):
                    continue
                pid = str(row['product_id'])
                if pid not in store.rows:
                    store.rows[pid] = row
                    store.order.append(pid)
                store.found_by.setdefault(pid, [])
                if source not in store.found_by[pid]:
                    store.found_by[pid].append(source)
                store.note(pid, row.get('sku'), row.get('in_stock'), row.get('price'))
        elif name in ('view', 'compare'):
            for page in [body] if name == 'view' else body.get('compared') or []:
                if isinstance(page, dict) and page.get('product_id'):
                    pid = str(page['product_id'])
                    if pid not in store.pages:
                        store.opened.append(pid)
                    store.pages[pid] = page
                    for variant in page.get('variants') or []:
                        if isinstance(variant, dict):
                            store.note(pid, variant.get('sku'), variant.get('in_stock'), variant.get('price'))
        elif name == 'inspect_stock':
            store.note(args.get('product_id'), args.get('sku'), body.get('in_stock'), body.get('price'))
        elif name == 'add_to_cart':
            key = f"{args.get('product_id')}::{args.get('sku')}"
            added = body.get('added') if isinstance(body.get('added'), dict) else {}
            if key not in store.cart:
                store.cart.append(key)
            store.first_add = store.first_add or key
            store.note(args.get('product_id'), args.get('sku'), added.get('in_stock'), added.get('price'))
        if isinstance(body.get('market_event'), dict):
            event = body['market_event']
            target = event.get('target') or {}
            store.note(target.get('product_id'), target.get('sku'), event.get('in_stock_after'), event.get('new_price') or event.get('old_price'))

    def write_record(self, upto=None):
        cond = self.cond
        upto = len(cond.lines) if upto is None else upto
        lines = '\n'.join((f'- {line}' for line in cond.lines[:upto])) or '(none yet)'
        earlier = f"\n\nThe record before their latest line: {dumps({k: cond.record.get(k) for k in ('budget', 'musts', 'wishes', 'wish_chosen', 'category')})}" if cond.record else ''
        user = f"The shopper's request:\n{self.request}\n\nTheir lines since, in order:\n{lines}{earlier}"
        before = dict(cond.record)
        answer = self.net.ask(RECORD, user, deadline=WRAP_AT)
        cond.heard = upto
        cond.current = upto - 1
        if answer is None:
            if not cond.record:
                budget = [number(x) for x in re.findall('(\\d[\\d,]*(?:\\.\\d+)?)\\s*PHP', self.request)]
                cond.record = {'product': ' '.join(self.request.split()[:5]), 'category': '', 'budget': budget[-1] if budget else None, 'musts': [], 'wishes': [], 'wish_chosen': '', 'prefer': [], 'purpose': '', 'objective': {'property': '', 'direction': 'most'}, 'survey': 0, 'justify': [], 'shortlist': [], 'latest_line': 'request', 'asks_confirmation': False, 'dropped': [], 'searches': [self.request[:60]]}
            return
        record = {key: answer.get(key) for key in RECORD_TOOL['function']['parameters']['properties']}
        record['musts'] = [m for m in record.get('musts') or [] if isinstance(m, dict) and str(m.get('value') or '').strip() and (not self.not_a_must(m))]
        for m in record['musts']:
            if ' with ' in f" {squash(m.get('value'))} ":
                m['value'] = re.split('\\s+with\\s+', str(m.get('value')), flags=re.I)[-1].strip() or m['value']
        if before.get('musts'):
            dropped = {squash(d) for d in record.get('dropped') or []}
            have = {(squash(m.get('property')), squash(m.get('value'))) for m in record['musts']}
            props = {squash(m.get('property')) for m in record['musts']}
            for old in before['musts']:
                ident = (squash(old.get('property')), squash(old.get('value')))
                gone = any((squash(old.get('value')) in d or squash(old.get('property')) in d for d in dropped if d))
                if ident not in have and (not gone) and (squash(old.get('property')) not in props):
                    record['musts'].append(old)
            if before.get('budget') is not None and record.get('budget') is None:
                record['budget'] = before['budget']
        if before.get('musts'):
            old_by_prop = {squash(m.get('property')): m for m in before.get('musts') or [] if isinstance(m, dict)}
            for i, must in enumerate(record['musts']):
                prev = old_by_prop.get(squash(must.get('property')))
                if prev and squash(must.get('value')) in cond.abandoned and (squash(must.get('value')) != squash(prev.get('value'))):
                    record['musts'][i] = prev
            for must in record['musts']:
                prev = old_by_prop.get(squash(must.get('property')))
                if prev and squash(prev.get('value')) != squash(must.get('value')):
                    cond.abandoned.add(squash(prev.get('value')))
        record['wishes'] = [w for w in record.get('wishes') or [] if isinstance(w, dict) and (not self.not_a_must(w))]
        request_words = re.findall("[A-Za-z][\\w&'.-]*", self.request)
        for wish in record['wishes']:
            value = str(wish.get('value') or '')
            if value and squash(value) not in squash(self.request):
                close = sorted(((similarity(squash(w), squash(value)), w) for w in set(request_words) if w[:1].isupper()), reverse=True)
                if close and close[0][0] >= 0.5:
                    wish['value'] = close[0][1]
        if len(record['wishes']) < 2:
            record['wishes'] = [] if not before.get('wishes') else record['wishes']
        if len(before.get('wishes') or []) == 2 and len(record.get('wishes') or []) < 2:
            record['wishes'] = before['wishes']
        if before.get('wish_chosen') and (not record.get('wish_chosen')):
            idx = {'first': 0, 'second': 1}.get(str(before['wish_chosen']))
            chosen = (before.get('wishes') or [None, None])[idx] if idx is not None and len(before.get('wishes') or []) == 2 else None
            dropped = {squash(d) for d in record.get('dropped') or []}
            if not (isinstance(chosen, dict) and any((squash(chosen.get('value')) in d for d in dropped if d))):
                record['wish_chosen'] = before['wish_chosen']
        wishes = [w for w in record.get('wishes') or [] if isinstance(w, dict)][:2]
        answered = any((isinstance(c, dict) and c.get('what') == 'answer' for c in record.get('line_changes') or [])) or bool(cond.asked and cond.asked[-1][0] == 'priority' and (cond.answer_index is not None) and (cond.current == cond.answer_index))
        if len(wishes) == 2 and (not record.get('wish_chosen')) and answered and (0 <= cond.current < len(cond.lines)):
            said = {singular_of(w) for w in tokens(cond.lines[cond.current])}
            named = [i for i, w in enumerate(wishes) if tokens(w.get('value')) and {singular_of(t) for t in tokens(w.get('value'))} <= said]
            if len(named) == 1:
                record['wish_chosen'] = ('first', 'second')[named[0]]
        if len(wishes) == 2:
            chosen = {'first': 0, 'second': 1}.get(str(record.get('wish_chosen') or ''))
            drop = [w for i, w in enumerate(wishes) if chosen is None or i != chosen]
            dropped_values = {squash(w.get('value')) for w in drop}
            record['musts'] = [m for m in record['musts'] if squash(m.get('value')) not in dropped_values]
            if chosen is not None and (not any((squash(m.get('value')) == squash(wishes[chosen].get('value')) or squash(m.get('property')) == squash(wishes[chosen].get('property')) for m in record['musts']))):
                w = wishes[chosen]
                record['musts'].append({'property': w.get('property'), 'value': w.get('value'), 'kind': 'word', 'unit': '', 'words': w.get('value'), 'also': []})
        if len(record.get('wishes') or []) == 2:
            cond.had_choice = True
        cond.record = record
        if cond.current < 0:
            cond.request_purposes = [str(x) for x in record.get('purposes') or [] if squash(x) and squash(x)[:12] in squash(self.request)]
        self.note_changes(before)

    def not_a_must(self, must):
        fields = set()
        for name in ('in_stock', 'max_price', 'category'):
            fields |= set(name.split('_')) - {'in', 'max'}
        words = {singular_of(w) for w in tokens(must.get('property')) | tokens(must.get('value'))}
        budget = number(self.cond.record.get('budget')) if self.cond.record else None
        value = squash(must.get('value'))
        shelf = {singular_of(w) for w in tokens(self.cond.record.get('category') if self.cond.record else '')}
        if words and {singular_of(w) for w in tokens(must.get('value'))} and ({singular_of(w) for w in tokens(must.get('value'))} <= shelf):
            return True
        return bool(words & fields) or (budget is not None and number(must.get('value')) == budget and ('php' in value)) or (bool(value) and value == squash(self.cond.record.get('category') if self.cond.record else ''))

    def note_changes(self, before):
        cond = self.cond
        if not before:
            return
        old = {squash(m.get('property')): m for m in before.get('musts') or [] if isinstance(m, dict)}
        new = {squash(m.get('property')): m for m in cond.musts()}
        notes = []
        for prop, must in new.items():
            if prop in old and squash(old[prop].get('value')) != squash(must.get('value')):
                notes.append(('switching to', str(must.get('value'))))
            elif prop not in old and cond.record.get('latest_line') in ('requirement_add', 'requirement_change'):
                notes.append(('adding', str(must.get('value'))))
        for prop, must in old.items():
            if prop not in new and cond.record.get('latest_line') in ('requirement_drop', 'requirement_change'):
                notes.append(('dropping', str(must.get('value'))))
        old_budget, new_budget = (number(before.get('budget')), cond.budget())
        if old_budget is not None and new_budget is not None and (new_budget != old_budget):
            notes.append(('budget', f'{new_budget:.2f}'))
        if before.get('wish_chosen') != cond.record.get('wish_chosen') and cond.record.get('wish_chosen'):
            cond.disclosed.add('priority')
        new_musts = set(new) - set(old)
        if new_musts and cond.asked and (cond.record.get('latest_line') == 'answer'):
            cond.disclosed.add(cond.asked[-1][0])
        if notes:
            cond.changed = True
        answering = cond.answer_index is not None and cond.current == cond.answer_index
        if notes and (not answering) and (cond.record.get('asks_confirmation') or cond.record.get('latest_line') in ('requirement_change', 'requirement_add', 'requirement_drop')):
            cond.owed.append(notes)

    def hear(self):
        if self.cond.heard >= len(self.cond.lines) or self.clock.elapsed() > WRAP_AT:
            return False
        watched = ('budget', 'musts', 'wish_chosen', 'category', 'shortlist', 'survey', 'justify')
        before = dumps({k: self.cond.record.get(k) for k in watched})
        while self.cond.heard < len(self.cond.lines) and self.clock.elapsed() <= WRAP_AT:
            kept = {k: self.cond.record.get(k) for k in ('budget', 'musts', 'wishes', 'wish_chosen', 'objective', 'survey', 'justify', 'shortlist')}
            owed, changed = (len(self.cond.owed), self.cond.changed)
            heard = self.cond.heard
            self.write_record(upto=heard + 1)
            self.cond.heard = max(self.cond.heard, heard + 1)
            if not self.cond.record.get('line_changes') and kept.get('musts') is not None and (not self.grounded(kept, heard)):
                self.cond.record.update(kept)
                del self.cond.owed[owed:]
                self.cond.changed = changed
        return before != dumps({k: self.cond.record.get(k) for k in watched})

    def wishes_from_doubt(self, answer):
        if not answer or not squash(answer.get('sentence')) or squash(answer.get('sentence'))[:30] not in squash(self.request):
            return
        purpose = squash(self.cond.record.get('purpose'))
        things = [t for t in answer.get('things') or [] if isinstance(t, dict) and squash(t.get('value')) and (squash(t.get('value')) in squash(self.request)) and (not (purpose and squash(t.get('value')) in purpose))]
        if len(things) != 2:
            return
        record = self.cond.record
        record['wishes'] = [{'property': t.get('property'), 'value': t.get('value')} for t in things]
        values = {squash(t.get('value')) for t in things}
        record['musts'] = [m for m in record.get('musts') or [] if squash(m.get('value')) not in values]
        self.cond.had_choice = True

    def doubts_both(self):
        return bool(squash(self.cond.record.get('doubt')))

    def fact_decides(self, key):
        pid, _ = key.split('::', 1)
        page = self.store.pages.get(pid) or {}
        facts = page.get('facts') or {}
        options = {singular_of(w) for w in tokens(self.store.options(key))}
        for must in self.cond.musts():
            prop = {singular_of(w) for w in tokens(must.get('property'))}
            if not prop or str(must.get('kind')) == 'figure' or re.search('\\d', str(must.get('value'))) or any((squash(k) == squash(must.get('property')) for k in page if k != 'facts')):
                continue
            value = {singular_of(w) for w in tokens(must.get('value'))} - prop
            if not value or value <= options:
                continue

            def named(k):
                have = {singular_of(w) for w in tokens(str(k).replace('_', ' '))}
                return bool(have) and (have <= prop or prop <= have)
            here = [v for k, v in facts.items() if named(k)]
            if len(here) > 1 and len(prop) == 1:
                continue
            if here:
                noun = {singular_of(w) for w in tokens(self.cond.product())}
                if not any((fact_says(v, value - noun, prop | noun) for v in here)):
                    return False
                continue
            shown = {singular_of(w) for w in tokens(str(page.get('title') or '') + ' ' + self.store.options(key))}
            if not value <= shown and sum((1 for p in self.store.pages.values() if any((named(k) for k in p.get('facts') or {})))) >= 3:
                return False
        return True

    def grounded(self, kept, index):
        line = self.cond.lines[index] if index < len(self.cond.lines) else ''
        said = {singular_of(w) for w in tokens(line)}
        old = {str(m.get('value')).lower() for m in kept.get('musts') or [] if isinstance(m, dict)}
        new = [str(m.get('value')) for m in self.cond.musts() if str(m.get('value')).lower() not in old]
        figures_said = {v for v, _ in figures(line)}
        budget_moved = number(self.cond.record.get('budget')) != number(kept.get('budget'))
        if not new and (not budget_moved):
            return False
        if budget_moved and number(self.cond.record.get('budget')) not in figures_said:
            return False
        return all(({singular_of(w) for w in tokens(v)} <= said and tokens(v) for v in new))

    def next_question(self):
        cond = self.cond
        asked_kinds = [kind for kind, _ in cond.asked]
        wishes = cond.wishes()
        if len(wishes) == 2 and (not cond.record.get('wish_chosen')) and ('priority' not in cond.disclosed):
            a, b = wishes
            if asked_kinds.count('priority') == 0:
                return ('priority', f"Which matters more to you: the {a['property']} {a['value']} or the {b['property']} {b['value']}?")
            if asked_kinds.count('priority') == 1:
                return ('priority', f"Would you prefer the {a['property']} {a['value']} or the {b['property']} {b['value']}?")
        if (wishes or cond.had_choice) and (not cond.record.get('wish_chosen')) and (asked_kinds.count('priority') >= 2) and str(cond.record.get('purpose') or '').strip() and (asked_kinds.count('purpose') == 0):
            return ('purpose', f'What will you be using the {cond.product()} for?')
        if wishes or cond.had_choice:
            return None
        if str(cond.record.get('purpose') or '').strip() and 'purpose' not in cond.disclosed and (asked_kinds.count('purpose') == 0):
            return ('purpose', f'What will you be using the {cond.product()} for?')
        purposes = [str(x).strip().rstrip('.') for x in getattr(cond, 'request_purposes', []) if str(x).strip()]
        if len(purposes) >= 2 and asked_kinds.count('purpose') == 1 and (asked_kinds.count('purpose2') == 0) and (cond.answer_index is not None) and (len(cond.lines) > cond.answer_index):
            answered = {singular_of(w) for w in tokens(cond.lines[cond.answer_index])}
            untold = [x for x in purposes if len({singular_of(w) for w in tokens(x)} - answered) > len(tokens(x)) / 2]
            if len(untold) == 1:
                second = untold[0][0].lower() + untold[0][1:]
                return ('purpose2', f'And for {second}, is there anything it must have?')
        return None

    def must_words(self):
        out = []
        for must in self.cond.musts():
            for word in [must.get('words'), must.get('value')] + list(must.get('also') or []):
                word = str(word or '').strip()
                if word:
                    out.append(word)
                    spaced = re.sub('(\\d)\\s*([^\\W\\d_])', '\\1 \\2', word)
                    if spaced != word:
                        out.append(spaced)
        return list(dict.fromkeys(out))

    def learn_shelf(self):
        cond, store = (self.cond, self.store)
        if getattr(cond, 'shelf_label', '') or not store.rows:
            return
        want = {singular_of(w) for w in tokens(cond.product()) | tokens(cond.record.get('category'))}
        joined = re.sub('[^a-z0-9]', '', squash(cond.product()) + squash(cond.record.get('category')))
        request = squash(self.request)
        scores = {}
        for row in list(store.rows.values()) + list(store.pages.values()):
            path = [str(x) for x in row.get('category_path') or [] if str(x).strip()]
            for depth, label in enumerate(path):
                have = {singular_of(w) for w in tokens(label)}
                letters = re.sub('[^a-z0-9]', '', squash(label))
                cover = len(want & have) / max(1, len(want))
                if letters and letters in joined:
                    cover = max(cover, 1.0)
                prev = scores.get(label)
                count = prev[4] + 1 if prev else 1
                verbatim = len(have) >= 2 and squash(label) in request
                scores[label] = (round(cover, 3), verbatim, len(want & have) / max(1, len(have)), depth, count)
        if not scores:
            return
        label, score = max(scores.items(), key=lambda kv: kv[1])
        shelf_rows = sum((1 for srcs in store.found_by.values() if any((src.startswith('filter:') for src in srcs))))
        exact = score[0] >= 1.0 and score[2] >= 1.0
        if exact or (score[1] and score[0] >= 1.0) or (shelf_rows < 3 and score[0] >= 0.5):
            cond.shelf_label = label

    def discovery_calls(self, extra=None, limit=None):
        cond, store = (self.cond, self.store)
        self.learn_shelf()
        budget = cond.budget()
        planned = []
        shelf = cond.category()
        working = [s for s in shelf_spellings(shelf, 4) if any((src == f'filter:{s}' for srcs in store.found_by.values() for src in srcs))]
        for spelling in working[:1] or shelf_spellings(shelf, 4):
            for band in PRICE_BANDS if working else (1.0,):
                args = {'category': spelling, 'in_stock': True, 'k': 10}
                if budget:
                    args['max_price'] = round(budget * band, 2)
                planned.append((f'filter:{squash(spelling)}:{band}', ('filter', args)))
        makers = [str(m.get('value')) for m in cond.musts() if squash(m.get('property')) == 'brand'] + [str(x) for x in cond.record.get('prefer') or []]
        for maker in dict.fromkeys((m for m in makers if m.strip())):
            args = {'category': shelf, 'brand': maker, 'in_stock': True, 'k': 10}
            if budget:
                args['max_price'] = budget
            planned.insert(0, (f'filter:brand:{squash(maker)}', ('filter', args)))
        queries = list(extra or []) + self.mined_queries() + [str(q) for q in cond.record.get('searches') or []]
        product = cond.product()
        for word in self.must_words()[:4]:
            queries.append(f'{word} {product}')
        queries.append(product)
        for query in queries:
            query = ' '.join(str(query).split())[:80]
            if query:
                planned.append((f'search:{squash(query)}', ('search', {'query': query, 'in_stock': True, 'k': 10})))
        calls, seen = ([], set())
        for mark, call in planned:
            if mark in store.searched or mark in seen:
                continue
            if len(calls) >= (limit or self.session.per_turn):
                break
            seen.add(mark)
            calls.append(call)
        store.searched |= seen
        return calls

    def must_hits(self, text):
        have = tokens(text)
        hits = 0
        for must in self.cond.musts():
            words = set()
            for value in [must.get('value'), must.get('words')] + list(must.get('also') or []):
                words |= {w for w in tokens(value) if len(w) >= 4 or any((ch.isdigit() for ch in w))}
            hits += bool(words & have)
        return hits

    def mined_queries(self, limit=4):
        store, cond = (self.store, self.cond)
        shelf_rows = [store.rows[pid] for pid in store.order if any((src.startswith('filter:') for src in store.found_by.get(pid, [])))]
        shelf_rows += [store.pages[pid] for pid in store.opened if self.on_shelf(pid)]
        if len(shelf_rows) < 5:
            return []
        skip = tokens(cond.product()) | tokens(cond.category()) | {w for m in cond.musts() for w in tokens(m.get('value'))}
        shelf_ids = {id(r) for r in shelf_rows}
        off_rows = [store.rows[pid] for pid in store.order if id(store.rows[pid]) not in shelf_ids and (not self.on_shelf(pid))]

        def shares(rows):
            counts = {}
            for row in rows:
                for word in {singular_of(w) for w in re.findall('[a-z]{4,}', squash(row.get('title')))}:
                    counts[word] = counts.get(word, 0) + 1
            return {w: n / max(1, len(rows)) for w, n in counts.items()}
        on, off = (shares(shelf_rows), shares(off_rows) if len(off_rows) >= 5 else {})
        ranked = sorted((w for w in on if w not in skip and on[w] * len(shelf_rows) >= 2 and (on[w] > off.get(w, 0.0) + 0.1)), key=lambda w: -(on[w] - off.get(w, 0.0)))
        common = ranked[:limit]
        values = ' '.join((str(m.get('words') or m.get('value')) for m in cond.musts()[:3]))
        return [f'{values} {word}'.strip() for word in common]

    def on_shelf(self, pid):
        shelf = self.cond.category()
        if not shelf:
            return True
        path = (self.store.pages.get(pid) or self.store.rows.get(pid) or {}).get('category_path') or []
        return any((same_shelf(shelf, label) for label in path))

    def open_calls(self, limit=OPEN_PER_TURN):
        store, cond = (self.store, self.cond)
        budget = cond.budget()

        def rank(pid):
            row = store.rows.get(pid) or {}
            by_filter = any((src.startswith('filter:') for src in store.found_by.get(pid, [])))
            hits = self.must_hits(row.get('title'))
            tier = 0 if hits and by_filter else 1 if by_filter else 2 if hits else 3
            return (tier, -hits, -len(store.found_by.get(pid, [])))
        fresh = [pid for pid in store.order if pid not in store.pages and store.rows[pid].get('in_stock') is not False and (not (budget and number(store.rows[pid].get('price')) and (number(store.rows[pid].get('price')) > budget + 1e-06)))]
        if cond.shortlist():
            named = {k.split('::')[0] for k in cond.shortlist()}
            fresh = [pid for pid in named if pid not in store.pages] + [pid for pid in fresh if pid in named]
        fresh = sorted(dict.fromkeys(fresh), key=rank)[:limit]
        return [('compare', {'product_ids': fresh[i:i + COMPARE_SIZE]}) for i in range(0, len(fresh), COMPARE_SIZE)]

    def orderable(self, key):
        in_stock, price = self.store.live.get(key, (None, None))
        budget = self.cond.budget()
        named = self.cond.shortlist()
        return in_stock is not False and (not (budget and number(price) is not None and (number(price) > budget + 1e-06))) and (not named or key in named)

    def candidates(self):
        store = self.store
        keys = []
        for pid in store.opened:
            page = store.pages.get(pid) or {}
            own = [f"{pid}::{v.get('sku')}" for v in page.get('variants') or [] if isinstance(v, dict)]
            keys += [k for k in own if k not in store.lost and self.orderable(k)][:VARIANTS_PER_PAGE]
        shelved = [k for k in keys if self.on_shelf(k.split('::')[0])]
        keys = shelved or keys

        def strong(k):
            return store.options(k) + ' ' + dumps((store.pages.get(k.split('::')[0]) or {}).get('facts') or {})
        keys.sort(key=lambda k: (-self.must_hits(strong(k)), -self.must_hits(store.page_text(k))))
        return keys

    def situation(self):
        cond, store = (self.cond, self.store)
        notes = []
        obj, direction = cond.objective()
        if obj:
            notes.append(f'Objective: the {direction} {obj}.')
        if cond.record.get('wish_chosen') and len(cond.wishes()) == 2:
            chosen = cond.wishes()[0 if cond.record['wish_chosen'] == 'first' else 1]
            notes.append(f"The shopper chose: {chosen['property']} {chosen['value']} matters more (it is a must); the other wish is not required.")
        if cond.record.get('prefer'):
            notes.append(f"Their pick if it fits (rank a fitting candidate with it first): {', '.join(map(str, cond.record['prefer']))}.")
        for key in sorted(store.lost):
            in_stock, price = store.live.get(key, (None, None))
            notes.append(f"Lost: {key} ({('sold out' if in_stock is False else 'now above the budget')}).")
        if cond.lines:
            notes.append(f'Their latest line: {cond.lines[-1]}')
        return '\n'.join(notes) or 'Nothing has changed.'

    def table(self, keys):
        store = self.store
        lines = []
        for i, key in enumerate(keys[:TABLE_LIMIT], 1):
            lines.append(f'[{i}] {self.row(key)}')
        return '\n'.join(lines)

    def row(self, key):
        store = self.store
        pid, sku = key.split('::', 1)
        page = store.pages.get(pid) or {}
        in_stock, price = store.live.get(key, (None, None))
        path = ' > '.join(map(str, page.get('category_path') or []))
        facts = '; '.join((f'{k} = {v}' for k, v in (page.get('facts') or {}).items()))[:600]
        return f"key={key} | {price} PHP | brand field: {page.get('brand') or '(empty)'} | shelf: {path} | this variant's options: {store.options(key) or '(none)'}\n    title: {str(page.get('title') or '')[:220]}\n    facts: {facts or '(none)'}"

    def judge(self, keys):
        cond = self.cond
        verdicts = {k: self.code_verdicts(k) for k in keys}
        open_keys = [k for k in keys if False not in verdicts[k]]
        keys = open_keys or keys
        self.code_fit = {k for k in keys if verdicts[k] and all((v is True for v in verdicts[k]))}
        musts = '; '.join((f"{m.get('property')}: {m.get('value')}" + (f" (the same as {', '.join(map(str, m.get('also')))})" if m.get('also') else '') for m in cond.musts())) or '(none beyond the shelf)'
        user = f"The shopper's request:\n{self.request}\n\nTheir lines since:\n" + ('\n'.join((f'- {l}' for l in cond.lines)) or '(none)') + f"\n\nMusts now: {musts}\nShelf: {cond.category() or '(not named)'}\nBudget: {cond.budget()} PHP\n{self.situation()}\n\nCandidates:\n{self.table(keys)}"
        answer = self.net.ask(JUDGE, user, deadline=WRAP_AT + 12)
        if answer is None:
            return (None, None)
        return (self.decide(answer, keys), answer)

    def decide(self, answer, keys):
        by_key = {re.sub('\\s+', '', str(k)): k for k in keys}
        objective, direction = self.cond.objective()
        fitting, dropped, now = ([], [], set())
        for entry in answer.get('assessments') or []:
            if not isinstance(entry, dict):
                continue
            key = by_key.get(re.sub('\\s+', '', str(entry.get('key') or '')))
            if key is None:
                continue
            self.judged[key] = entry
            now.add(key)
            if not entry.get('fits'):
                continue
            text = self.store.page_text(key) + ' ' + self.row(key)
            met = [line for line in entry.get('lines') or [] if isinstance(line, dict) and line.get('verdict') == 'met' and self.about_a_must(line)]
            if not all((quoted(line.get('quote'), text) for line in met)):
                dropped.append((key[-14:], 'unquoted', [str(line.get('quote'))[:60] for line in met if not quoted(line.get('quote'), text)][:2]))
                continue
            if any((self.list_quote(line, key) for line in met)) or not all((self.quote_names_value(line) for line in met)):
                dropped.append((key[-14:], 'list/value'))
                continue
            if self.glued_colour(key):
                dropped.append((key[-14:], 'glued'))
                continue
            if not self.fact_decides(key):
                dropped.append((key[-14:], 'fact'))
                continue
            if not self.figures_stated(key):
                dropped.append((key[-14:], 'figure'))
                continue
            value = number(entry.get('objective_value'))
            if not quoted(entry.get('objective_quote'), text):
                value = None
            elif value is not None and re.search('\\d\\s*[*x×]\\s*\\d', str(entry.get('objective_quote') or '')):
                against = re.search('\\d+(?:\\.\\d+)?(?:\\s*[*x×]\\s*\\d+(?:\\.\\d+)?)*\\s*[*x×]\\s*(\\d+(?:\\.\\d+)?)\\s*[a-z]', str(entry.get('objective_quote')).lower())
                value = number(against.group(1)) if against else value
            self.evidence[key] = self.evidence_strength(key, entry)
            fitting.append((key, value))
        for key in getattr(self, 'code_fit', set()):
            if key in by_key.values() and key not in {k for k, _ in fitting} and self.orderable(key):
                self.evidence[key] = len(self.cond.musts())
                fitting.append((key, None))
        ranking = [by_key.get(re.sub('\\s+', '', str(k))) for k in answer.get('ranking') or []]
        order = {k: i for i, k in enumerate([k for k in ranking if k])}
        sign = -1 if direction != 'least' else 1

        def option_figure(key):
            found = [v for v, _ in figures(self.store.options(key))]
            return max(found) if found else 0.0
        quote_units = []
        for k, _ in fitting:
            entry = self.judged.get(k) or {}
            said = number(entry.get('objective_value'))
            figs = [(v, u) for v, u in figures(re.sub('\\s*[*x×]\\s*', ' * ', str(entry.get('objective_quote') or ''))) if u]
            if figs:
                match = [u for v, u in figs if said is not None and abs(v - said) < 1e-09]
                quote_units.append(squash(match[0] if match else figs[-1][1]))
        objective_words = {singular_of(w) for w in tokens(objective)}
        must_units = {squash(u) for m in self.cond.musts() for _, u in figures(m.get('value')) if u and (not {singular_of(w) for w in tokens(m.get('property'))} & objective_words)}
        quote_units = [u for u in quote_units if u not in must_units]
        unit = max(set(quote_units), key=quote_units.count) if quote_units else ''
        measured = {k: self.measure(k, unit) for k, _ in fitting} if objective and unit else {}
        if sum((v is not None for v in measured.values())) >= 1:
            fitting = [(k, measured[k] if measured.get(k) is not None else v) for k, v in fitting]
        liked = [squash(x) for x in self.cond.record.get('prefer') or [] if squash(x)]

        def preferred(key):
            pid = key.split('::')[0]
            page = self.store.pages.get(pid) or {}
            where = squash(' '.join([str(page.get('brand') or ''), self.store.options(key), str(page.get('title') or '')]))
            return any((x in where for x in liked))
        fitting.sort(key=lambda kv: (not self.exact(kv[0]), -self.evidence.get(kv[0], 0), not preferred(kv[0]) if liked else False, kv[1] is None if objective else False, sign * (kv[1] or 0) if objective else 0, sign * option_figure(kv[0]) if objective else 0, order.get(kv[0], 99)))
        chosen = [k for k, _ in fitting]
        self.fit_count = len(chosen)
        self.backups = chosen[1:]
        self.closest = by_key.get(re.sub('\\s+', '', str(answer.get('winner') or '')))
        return chosen[0] if chosen else None

    def measure(self, key, unit):
        pid, _ = key.split('::', 1)
        page = self.store.pages.get(pid) or {}
        facts = ' '.join(map(str, (page.get('facts') or {}).values()))
        title = re.sub('\\d+(?:\\.\\d+)?(?:\\s*/\\s*\\d+(?:\\.\\d+)?)+\\s*[a-z]+', ' ', squash(page.get('title')))
        for text in (self.store.options(key), facts, title):
            found = [v for v, u in figures(text) if squash(u) == unit]
            if found:
                return max(found)
        return None

    def quote_names_value(self, line):

        def words(text):
            return {singular_of(w) for w in tokens(text) if len(w) >= 3}
        said = words(line.get('must')) | tokens(line.get('must'))
        for must in self.cond.musts():
            if str(must.get('kind')) == 'figure' or re.search('\\d', str(must.get('value'))):
                if (words(must.get('property')) & said or tokens(must.get('value')) & tokens(line.get('must'))) and (not re.search('\\d', str(line.get('quote') or ''))):
                    return False
                continue
            if 'brand' in tokens(must.get('property')):
                continue
            value_words = words(must.get('value'))
            if not value_words or not (value_words <= said or words(must.get('property')) & said):
                continue
            spellings = [must.get('value'), must.get('words')] + list(must.get('also') or [])
            quote = words(line.get('quote'))
            return any((words(s) <= quote for s in spellings if words(s)))
        return True

    def code_verdicts(self, key):
        pid, _ = key.split('::', 1)
        page = self.store.pages.get(pid) or {}
        facts = page.get('facts') or {}
        options = self.store.options(key)
        parts = [t for t in re.split('[\\s/,;()\\[\\]+|]+', options.casefold()) if t]
        alone = {singular_of(t) for t in parts}
        fused = {singular_of(w) for t in parts if '-' in t and re.search('\\d', t) for w in t.split('-')}
        text = ' '.join([options, ' '.join(map(str, facts.values())), str(page.get('title') or '')])
        stated = {(v, squash(u)[:1]) for v, u in figures(text) if u}
        noun = {singular_of(w) for w in tokens(self.cond.product())}
        out = []
        for must in self.cond.musts():
            prop = {singular_of(w) for w in tokens(must.get('property'))}
            value_raw = str(must.get('value') or '')
            if squash(must.get('property')) in page and squash(must.get('property')) != 'facts' and page.get(squash(must.get('property'))):
                out.append(squash(page.get(squash(must.get('property')))) == squash(value_raw))
                continue
            found = [(v, squash(u)) for v, u in figures(value_raw) if re.fullmatch('[a-z]+', squash(u) or '-')]
            if found:
                v, u = found[0]
                if (v, u[:1]) in stated or any((plain(a) and plain(a) in plain(text) for a in list(must.get('also') or []))):
                    out.append(True)
                else:
                    out.append(None if not any((su == u[:1] for _, su in stated)) else False)
                continue
            value = {singular_of(w) for w in tokens(value_raw)} - prop
            if not value or re.search('\\d', value_raw):
                out.append(None)
                continue
            named = [str(val) for k, val in facts.items() if (lambda h: bool(h) and (h <= prop or prop <= h))({singular_of(w) for w in tokens(str(k).replace('_', ' '))})]
            if len(named) > 1 and len(prop) == 1:
                out.append(None)
                continue
            if named:
                out.append(any((fact_says(val, value - noun, prop | noun) for val in named)))
                continue
            if value <= alone:
                out.append(True)
            elif value <= fused:
                out.append(False)
            else:
                out.append(None)
        return out

    def figures_stated(self, key):
        pid, _ = key.split('::', 1)
        page = self.store.pages.get(pid) or {}
        text = ' '.join([self.store.options(key), ' '.join(map(str, (page.get('facts') or {}).values())), str(page.get('title') or '')])
        stated = {(v, squash(u)[:1]) for v, u in figures(text) if u}
        plain_text = plain(text)
        for must in self.cond.musts():
            found = [(v, squash(u)) for v, u in figures(must.get('value')) if re.fullmatch('[a-z]+', squash(u) or '-')]
            if not found:
                continue
            value, unit = found[0]
            if (value, unit[:1]) in stated:
                continue
            if any((plain(a) and plain(a) in plain_text for a in [must.get('words')] + list(must.get('also') or []))):
                continue
            return False
        return True

    def glued_colour(self, key):
        parts = [t for t in re.split('[\\s/,;()\\[\\]+|]+', self.store.options(key).casefold()) if t]
        for must in self.cond.musts():
            if str(must.get('kind')) == 'figure' or re.search('\\d', str(must.get('value'))):
                continue
            words = [singular_of(w) for w in str(must.get('value') or '').casefold().split() if w]
            if not words:
                continue
            alone = {singular_of(t) for t in parts}
            fused = {singular_of(w) for t in parts if '-' in t and re.search('\\d', t) for w in t.split('-')}
            if not set(words) <= alone and set(words) <= fused:
                return True
        return False

    def list_quote(self, line, key):
        quote = str(line.get('quote') or '')
        if not re.search('\\d\\s*/\\s*\\d', quote):
            return False
        return not figures(self.store.options(key)) or not any((unit for _, unit in figures(self.store.options(key))))

    def evidence_strength(self, key, entry):
        pid, _ = key.split('::', 1)
        strong_text = self.store.options(key) + ' ' + dumps((self.store.pages.get(pid) or {}).get('facts') or {})
        return sum((1 for line in entry.get('lines') or [] if isinstance(line, dict) and line.get('verdict') == 'met' and line.get('quote') and self.about_a_must(line) and quoted(line.get('quote'), strong_text)))

    def about_a_must(self, line):
        said = {singular_of(w) for w in tokens(line.get('must'))}
        return any(({singular_of(w) for w in tokens(m.get('value'))} <= said or {singular_of(w) for w in tokens(m.get('property'))} & said for m in self.cond.musts() if tokens(m.get('value'))))

    def exact(self, key):
        text = squash(self.store.page_text(key))
        for must in self.cond.musts():
            if str(must.get('kind')) == 'figure' or re.search('\\d', str(must.get('value'))):
                continue
            words = [squash(w) for w in [must.get('value'), must.get('words')] + list(must.get('also') or []) if str(w or '').strip()]
            if words and (not any((re.search('(?<![a-z0-9])' + re.escape(singular_of(w)) + '(?:s|es)?(?![a-z0-9])', text) for w in words))):
                return False
        return True

    def choose(self):
        keys = self.candidates()
        if not keys:
            return None
        key, answer = self.judge(keys)
        rounds = 0
        while answer is not None and (answer.get('none_fit') or key is None) and (rounds < 2) and (self.clock.elapsed() < HURRY_AT) and (self.session.left() >= 6):
            rounds += 1
            calls = self.discovery_calls([str(q) for q in answer.get('searches') or []][:3])
            if calls:
                self.send(calls[:self.session.per_turn])
                self.react()
            opens = self.open_calls()
            if not opens:
                break
            self.send(opens[:self.session.per_turn])
            self.react()
            wider = self.candidates()
            if len(wider) <= len(keys):
                break
            keys = wider
            key, answer = self.judge(keys)
        if key and answer is not None and (key not in getattr(self, 'code_fit', set())) and (self.evidence.get(key, 0) < len(self.cond.musts())) and (self.clock.elapsed() < HURRY_AT):
            first = self.evidence.get(key, 0)
            again, second = self.judge(keys)
            if again and self.evidence.get(again, 0) > first:
                key, answer = (again, second)
            else:
                key = self.decide(answer, keys) or key
        return key or self.closest or (keys[0] if keys else None)

    def react(self):
        if self.hear():
            self.cond.changed = True

    def claims(self, key):
        pid, _ = key.split('::', 1)
        page = self.store.pages.get(pid) or {}
        facts = page.get('facts') or {}
        in_stock, price = self.store.live.get(key, (None, None))
        out = []
        wanted = self.cond.justify() + [str(m.get('property')) for m in self.cond.musts()]
        for field in wanted:
            low = squash(field)
            if 'price' in low:
                continue
            if 'stock' in low:
                continue
            if 'brand' in low and page.get('brand'):
                out.append({'field': 'brand', 'value': page.get('brand')})
                continue
            label = next((lab for lab in facts if tokens(lab.replace('_', ' ')) & tokens(low)), None)
            if label is not None:
                out.append({'field': label, 'value': facts[label]})
        if price is not None:
            out.append({'field': 'price', 'value': number(price)})
        if in_stock is True:
            out.append({'field': 'in_stock', 'value': True})
        seen, unique = (set(), [])
        for claim in out:
            if claim['field'] not in seen:
                seen.add(claim['field'])
                unique.append(claim)
        return unique

    def ask_turn(self):
        question = self.next_question()
        calls = []
        if question and 'message' in self.session.schemas:
            calls.append(('message', {'content': question[1]}))
            self.cond.asked.append(question)
            self.cond.answer_index = len(self.cond.lines)
        calls += self.discovery_calls(limit=self.session.per_turn - len(calls))
        if calls:
            self.send(calls[:self.session.per_turn])
            self.react()

    def owed_message(self, key=None):
        if not self.cond.owed or 'message' not in self.session.schemas:
            return []
        notes = [n for group in self.cond.owed for n in group]
        self.cond.owed = []
        actions = {}
        for verb, value in notes:
            if verb != 'budget':
                actions.setdefault(verb, []).append(value)
        parts = [f"{verb} {' and '.join(dict.fromkeys(values))}" for verb, values in actions.items()]
        budget = self.cond.budget()
        head = 'Confirmed: ' + (', '.join(parts) if parts else f'switching to your {budget:.2f} PHP budget' if budget else 'noted')
        price = number((self.store.live.get(key) or (None, None))[1]) if key else None
        tail = f', and the one I picked is {price:.2f} PHP and in stock' if price is not None else ''
        if budget and parts:
            tail += f', within your {budget:.2f} PHP budget'
        return [('message', {'content': (head + tail + '.')[:400]})]

    def add(self, key):
        pid, sku = key.split('::', 1)
        calls = [('add_to_cart', {'product_id': pid, 'sku': sku}), ('inspect_stock', {'product_id': pid, 'sku': sku})]
        self.send(calls)
        self.react()
        if not self.orderable(key):
            self.store.lost.add(key)
            return False
        return True

    def ensure_survey(self):
        need = self.cond.survey()
        tries = 0
        while need and tries < 2 and (sum((1 for pid in self.store.opened if self.on_shelf(pid))) < need + 1) and (self.session.left() > 6):
            tries += 1
            opens = self.open_calls(limit=max(need + 2, 10))
            if not opens:
                self.send(self.discovery_calls()[:self.session.per_turn])
                opens = self.open_calls(limit=max(need + 2, 10))
            if not opens:
                return
            self.send(opens[:self.session.per_turn])
            self.react()

    def run(self):
        self.write_record()
        if len(self.cond.wishes()) < 2 and (not self.cond.had_choice) and (self.clock.elapsed() < HURRY_AT):
            self.wishes_from_doubt(self.net.ask(DOUBT, f"The shopper's request:\n{self.request}", deadline=WRAP_AT))
        self.ask_turn()
        for _ in range(2):
            if self.next_question() is None or self.session.left() < 8 or self.clock.elapsed() > HURRY_AT:
                break
            self.ask_turn()
        calls = self.discovery_calls()
        if calls:
            self.send(calls)
            self.react()
        for _ in range(2):
            opens = self.open_calls()
            if not opens or self.session.left() < 10:
                break
            self.send(opens[:self.session.per_turn])
            self.react()
        self.ensure_survey()
        self.cond.changed = False
        choice = self.choose()
        if choice and self.cond.objective()[0] and self.sibling_round([choice] + self.backups[:1]):
            choice = self.choose() or choice
        if choice and (not self.store.cart) and (not self.exact(choice) or self.fit_count < 2 or (not self.on_shelf(choice.split('::')[0]))) and (self.clock.elapsed() < HURRY_AT) and (self.session.left() > 8):
            self.refresh_after_change()
            choice = self.choose() or choice
        if choice and (not self.store.cart) and (not self.on_shelf(choice.split('::')[0])):
            choice = next((k for k in [choice] + self.backups if self.on_shelf(k.split('::')[0]) and self.orderable(k)), choice)
        rounds = 0
        while choice is not None and rounds < 4 and (not self.session.done) and (self.session.left() > 3) and (self.clock.elapsed() < WRAP_AT):
            rounds += 1
            if choice not in self.store.cart and (not self.add(choice)):
                choice = next((k for k in self.backups if k not in self.store.lost and self.orderable(k)), None) or self.rechoose()
                continue
            if self.cond.changed:
                self.cond.changed = False
                self.refresh_after_change()
                fresh = self.choose()
                if fresh and fresh != choice:
                    choice = fresh
                    continue
            break
        self.close(choice)

    def sibling_round(self, keys):
        store = self.store
        if self.session.left() < 9 or self.clock.elapsed() > HURRY_AT:
            return False
        stems = {k.split('::')[1].rsplit('-', 1)[0] for k in keys if k}
        values = ' '.join((str(m.get('words') or m.get('value')) for m in self.cond.musts()[:2]))
        titles = []
        for k in keys:
            title = str((store.pages.get(k.split('::')[0]) or {}).get('title') or '')
            if title:
                words = re.findall('[^\\W_]+', title)
                titles.append(' '.join(words[:9]))
                titles.append(f"{values} {' '.join(words[:6])}".strip())
        brands = []
        for k in keys:
            page = store.pages.get(k.split('::')[0]) or {}
            words = re.findall('[^\\W_]+', str(page.get('title') or ''))
            option_words = [w for w in re.findall('[^\\W\\d_]+', store.options(k)) if len(w) > 1]
            if option_words:
                titles.append(f"{' '.join(option_words[:3])} {' '.join(words[:6])}".strip())
            if page.get('brand'):
                brands.append(str(page['brand']))
        calls = [('search', {'query': t, 'in_stock': True, 'k': 10}) for t in dict.fromkeys(titles) if f'search:{squash(t)}' not in store.searched]
        for t in titles:
            store.searched.add(f'search:{squash(t)}')
        shelf = self.cond.category()
        for brand in dict.fromkeys(brands):
            mark = f'filter:brand:{squash(brand)}'
            if shelf and mark not in store.searched:
                store.searched.add(mark)
                args = {'category': shelf, 'brand': brand, 'in_stock': True, 'k': 10}
                if self.cond.budget():
                    args['max_price'] = self.cond.budget()
                calls.append(('filter', args))
        if calls:
            self.send(calls[:self.session.per_turn])
            self.react()
        family = [(tokens((store.pages.get(k.split('::')[0]) or {}).get('title')), number(store.live.get(k, (None, None))[1])) for k in keys if k]

        def same_listing(row):
            words, price = (tokens(row.get('title')), number(row.get('price')))
            return any((words and t and (len(words & t) >= 0.8 * len(words | t)) and price and p and (abs(price - p) <= 0.01 * p) for t, p in family))
        siblings = [pid for pid in store.order if pid not in store.pages and (str(store.rows[pid].get('sku', '')).rsplit('-', 1)[0] in stems or same_listing(store.rows[pid])) and (not (self.cond.budget() and number(store.rows[pid].get('price')) and (number(store.rows[pid].get('price')) > self.cond.budget() + 1e-06)))]
        if not siblings:
            return False
        self.send([('compare', {'product_ids': siblings[i:i + COMPARE_SIZE]}) for i in range(0, len(siblings[:OPEN_PER_TURN]), COMPARE_SIZE)])
        self.react()
        return True

    def rechoose(self):
        if self.clock.elapsed() > WRAP_AT or self.session.left() < 5:
            return None
        self.refresh_after_change()
        return self.choose()

    def refresh_after_change(self):
        calls = self.discovery_calls()
        if calls and self.session.left() > 5:
            self.send(calls[:self.session.per_turn])
            self.react()
        opens = self.open_calls()
        if opens and self.session.left() > 5:
            self.send(opens[:self.session.per_turn])
            self.react()
        families = list(dict.fromkeys(list(reversed(self.store.cart)) + sorted(self.store.lost)))[:2]
        if families:
            self.sibling_round(families)

    def market_notice(self, target):
        if 'message' not in self.session.schemas:
            return []
        said = getattr(self, 'market_said', set())
        parts = []
        for key in self.store.cart:
            if key == target or key in said:
                continue
            pid, sku = key.split('::', 1)
            page = self.store.pages.get(pid) or {}
            seen = next((number(v.get('price')) for v in page.get('variants') or [] if isinstance(v, dict) and str(v.get('sku')) == sku), None)
            in_stock, price = self.store.live.get(key, (None, None))
            price = number(price)
            if in_stock is False:
                parts.append('an earlier pick sold out')
            elif seen is not None and price is not None and (abs(price - seen) > 0.009):
                parts.append(f'an earlier pick changed from {seen:.2f} to {price:.2f} PHP')
            else:
                continue
            said.add(key)
        self.market_said = said
        if not parts:
            return []
        parts = list(dict.fromkeys(parts))
        return [('message', {'content': f"Noted: {' and '.join(parts[:2])}, so I picked another one."})]

    def pre_order(self, target):
        pid, sku = target.split('::', 1)
        for _ in range(3):
            notes = [n for group in self.cond.owed for n in group]
            seen = len(self.cond.lines)
            owed = self.owed_message(target)
            self.send((owed or self.market_notice(target)) + [('inspect_stock', {'product_id': pid, 'sku': sku})])
            self.react()
            if not self.orderable(target):
                return False
            unread = notes and len(self.cond.lines) > seen and (self.cond.record.get('latest_line') != 'acknowledge')
            if not unread or self.session.left() < 3 or self.clock.elapsed() > WRAP_AT:
                return True
            self.cond.owed.insert(0, notes)
        return True

    def close(self, choice):
        store, session = (self.store, self.session)
        for _ in range(3):
            if session.done or session.dead or session.left() <= 0:
                return
            target = choice if choice in store.cart and self.orderable(choice) else next((k for k in reversed(store.cart) if self.orderable(k)), None)
            if target is None:
                pick = next((k for k in self.backups + self.candidates() if k not in store.lost and self.orderable(k)), None)
                if pick is None or session.left() < 2:
                    break
                if not self.add(pick):
                    choice = None
                    continue
                target = pick
            pid, sku = target.split('::', 1)
            if session.left() >= 2:
                if not self.pre_order(target):
                    store.lost.add(target)
                    choice = None
                    continue
                if self.cond.changed and session.left() >= 4 and (self.clock.elapsed() < WRAP_AT):
                    self.cond.changed = False
                    if session.left() >= 7:
                        self.refresh_after_change()
                    fresh = self.choose()
                    if fresh and fresh != target and self.add(fresh):
                        target = fresh
                        pid, sku = target.split('::', 1)
                    if self.cond.owed and session.left() >= 2 and (not self.pre_order(target)):
                        store.lost.add(target)
                        choice = None
                        continue
            args = {'product_id': pid, 'sku': sku}
            if self.cond.justify():
                args['justification'] = {'claims': self.claims(target)}
            result = self.send([('place_test_order', args)])
            if session.done:
                return
            self.react()
            if result and isinstance(result[0], dict) and result[0].get('error'):
                choice = target
                continue
            return
        if store.cart and (not session.done) and (session.left() > 0):
            pid, sku = store.cart[-1].split('::', 1)
            self.send([('place_test_order', {'product_id': pid, 'sku': sku})])

def agent_main(problem_data):
    episode = Episode(problem_data)
    try:
        episode.run()
    except Exception as exc:
        episode.dialogue.append({'role': 'assistant', 'content': f'stopped: {type(exc).__name__}'})
        try:
            if not episode.session.done:
                episode.close(None)
        except Exception:
            pass
    return episode.dialogue
