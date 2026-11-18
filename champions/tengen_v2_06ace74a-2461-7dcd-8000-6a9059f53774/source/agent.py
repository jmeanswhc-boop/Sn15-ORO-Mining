from __future__ import annotations
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from src.agent.proxy_client import ProxyClient
PARSERS = ('google/gemini-3.8-flash', 'moonshotai/kimi-k2.6')
READERS = ('google/gemini-3.5-flash-lite', 'moonshotai/kimi-k2.6')
FORM_MODELS = {'labels': READERS, 'values': READERS, 'doubts': READERS}
NO_THINKING = {'google/gemini-3.8-flash': {'effort': 'minimal'}, 'google/gemini-3.5-flash-lite': {'effort': 'minimal'}, 'moonshotai/kimi-k2.6': {'enabled': False}}
EPISODE_SECONDS = 205.0
SPEND_CAP = 0.08
TURN_RESERVE = 4
PROPERTY_QUESTIONS = 9
PURPOSE_AFTER = 2
LIST_FIRST = 2
ASK_FLOOR = 0.08
MAX_READ = 520
MAX_READ_BEFORE_CART = 300
STRETCH = 1.3
READ_MARGIN = 1.35
_infer = ProxyClient(timeout=35, max_retries=2)
_env = ProxyClient(timeout=90, max_retries=2)
FIRM, TITLE, SYN, WEAK, ABSENT, CONTRA = (3, 2, 1.5, 1, 0, -1)
KEEP = '\x00keep\x00'
LOW = WEAK
_TOKEN = re.compile('[a-z0-9]+(?:\\.[0-9]+)*')
_AMOUNT = re.compile('(?<![a-wyz0-9.])(\\d+(?:[.,]\\d+)?)\\s*(?:-|~|to)\\s*(\\d+(?:[.,]\\d+)?)\\s*([a-z]+|\\")|(?<![a-wyz0-9.])(\\d+(?:[.,]\\d+)?)\\s*([a-z]+(?![a-z0-9])|\\")')

def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith('ies'):
        return word[:-3] + 'y'
    return word[:-1] if len(word) > 3 and word.endswith('s') and (not word.endswith('ss')) else word

def tokens(text: Any) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(str(text or '').casefold())]

def has_phrase(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if not n or n > len(hay):
        return False
    return any((hay[i:i + n] == needle for i in range(len(hay) - n + 1)))

def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 4 and long_.startswith(short) and (long_[len(short):] in ('s', 'es', 'y')) and (not short[0].isdigit())

def has_form(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if not n or n > len(hay):
        return False
    return any((all((_same_word(h, w) for h, w in zip(hay[i:i + n], needle))) for i in range(len(hay) - n + 1)))

def running_text(value) -> bool:
    text = str(value or '')
    return bool(re.search('[<>]|=\\\\?\\"', text)) or len(text.split()) > 25

def _number(raw: str) -> float:
    if ',' in raw:
        head, _, tail = raw.partition(',')
        raw = head + tail if len(tail) == 3 else head + '.' + tail
    return float(raw)

def amounts(text: Any) -> list[tuple[float, float, str, int]]:
    found = []
    text = re.sub("(\\d)''", '\\1" ', str(text or '').casefold())
    text = re.sub('(\\d)(\\s*)[”″]', '\\1\\2"', text)
    text = re.sub('(\\d)-([a-z])', '\\1 \\2', text)
    for m in _AMOUNT.finditer(text):
        if m.group(1):
            lo, hi, unit = (_number(m.group(1)), _number(m.group(2)), m.group(3))
        else:
            lo = hi = _number(m.group(4))
            unit = m.group(5)
        unit = unit.rstrip('.')
        found.append((min(lo, hi), max(lo, hi), unit, m.start()))
    return found

def num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search('\\d[\\d,]*(?:\\.\\d+)?', value)
        return _number(m.group(0).replace(',', '')) if m else None
    return None

def entries(text: Any) -> list[str]:
    parts = [p.strip() for p in re.split('[,;/|]', str(text or ''))]
    return [p for p in parts if p]

def short(text: str, n: int=60) -> str:
    text = ' '.join(str(text or '').split())
    return text if len(text) <= n else text[:n - 1].rstrip() + '…'
_WISH = {'type': 'object', 'additionalProperties': False, 'required': ['property', 'value', 'kind', 'number', 'unit_forms', 'alt_amounts', 'alt_values', 'shade', 'made_of'], 'properties': {'property': {'type': 'string'}, 'value': {'type': 'string'}, 'kind': {'type': 'string', 'enum': ['amount', 'word', 'brand']}, 'number': {'type': ['number', 'null']}, 'unit_forms': {'type': 'array', 'items': {'type': 'string'}}, 'alt_amounts': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'required': ['number', 'unit_forms'], 'properties': {'number': {'type': 'number'}, 'unit_forms': {'type': 'array', 'items': {'type': 'string'}}}}}, 'alt_values': {'type': 'array', 'items': {'type': 'string'}}, 'shade': {'type': 'boolean'}, 'made_of': {'type': 'boolean'}}}
_WISH_RULES = "A requirement has: property (a short name), value (the whole phrase the product must match, as written: 'pale mint glaze' stays 'pale mint glaze'), kind ('amount' for a number with a unit, 'brand' for the maker or brand of the product itself, otherwise 'word': a maker named only for what the product fits or goes with is a 'word'). For an amount: number, unit_forms (every way sellers write that same unit: symbol or mark, abbreviation, full word, singular and plural) and alt_amounts (the same quantity written in each larger or smaller unit of the same system that sellers commonly use for it, and in the other measuring system (metric or imperial) whenever the conversion gives a whole number or a simple fraction: an amount with decimals is often a conversion of a round amount sellers write in the other system; otherwise empty). For a word or brand: number null, unit_forms and alt_amounts empty, and alt_values (at most 3 other words sellers commonly use for exactly this value; empty when there are none). shade: true when the value is a colour or a shade of one, else false. made_of: true only when the property names the material the product's own body is made of; a property that names anything else is false, even when its value is a substance."
REQUEST_FORM = {'name': 'request', 'rules': "You turn one online shopper's request into JSON for a shopping program. Copy values exactly as written; do not invent anything.\n- item: the product category as the shopper writes it, singular, kept whole when it joins kinds with '&' or 'and' (e.g. 'kettle & teapot set'); a type, style or other property named for it stays a wish.\n- budget: the most the shopper will pay, as a number (null if none); currency: its code.\n- wishes: every property the product must have. Leave out the price limit, being in stock or available (never a wish), a brand named only as an optional preference, not a requirement, and the two wishes of a doubted pair. A value may be very short (a code, a letter, one word): keep it.\n- doubted_pair: only when the shopper wants two things AND says in words that one listing may not have both, for any reason (the price included): side_a and side_b in the order written, and doubt: those words of doubt copied exactly from the request; otherwise null.\n- priority: the measurable quantity the shopper wants the most or the least of among the listings that fit: property, direction ('max' or 'min'), and units: never empty, every way sellers write a unit that measures that quantity, in both measuring systems (metric and imperial: listings write either, whatever the request uses; symbols, abbreviations, full words, singular and plural, larger and smaller units), each with its size relative to the first one. Null when there is none.\n- purposes: each reason, use or occasion the shopper gives, including anything the product must agree with. A sentence about why or how the product is wanted is a purpose even when worded as a must, never a wish. quote: the words of the request; you_form: the same purpose said back to the shopper, starting lowercase, in the second person (e.g. 'it is for your cousin's workshop'). Only the purpose itself, not other instructions to the assistant.\n- reason_asked: true only when the shopper asks to be told why the pick fits or for the facts behind it; reason_fields: the properties named for that, else empty.\n- searches: 4 short catalog search phrases a seller's listing title would match (the item with the values).\n" + _WISH_RULES, 'schema': {'type': 'object', 'additionalProperties': False, 'required': ['item', 'budget', 'currency', 'wishes', 'doubted_pair', 'priority', 'purposes', 'reason_asked', 'reason_fields', 'searches'], 'properties': {'item': {'type': 'string'}, 'budget': {'type': ['number', 'null']}, 'currency': {'type': 'string'}, 'wishes': {'type': 'array', 'items': _WISH}, 'doubted_pair': {'anyOf': [{'type': 'null'}, {'type': 'object', 'additionalProperties': False, 'required': ['side_a', 'side_b', 'doubt'], 'properties': {'side_a': _WISH, 'side_b': _WISH, 'doubt': {'type': 'string'}}}]}, 'priority': {'anyOf': [{'type': 'null'}, {'type': 'object', 'additionalProperties': False, 'required': ['property', 'direction', 'units'], 'properties': {'property': {'type': 'string'}, 'direction': {'type': 'string', 'enum': ['max', 'min']}, 'units': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'required': ['form', 'size'], 'properties': {'form': {'type': 'string'}, 'size': {'type': 'number'}}}}}}]}, 'purposes': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'required': ['quote', 'you_form'], 'properties': {'quote': {'type': 'string'}, 'you_form': {'type': 'string'}}}}, 'reason_asked': {'type': 'boolean'}, 'reason_fields': {'type': 'array', 'items': {'type': 'string'}}, 'searches': {'type': 'array', 'items': {'type': 'string'}}}}}
LINE_FORM = {'name': 'line', 'rules': "You read one message a shopper sent to a shopping assistant and report, as JSON, what it changes. You get the message, the assistant's last message, the requirements known so far (id, property, value) and, when still open, the two doubted wishes side_a and side_b.\n- kind: 'need' (tells something the item must have, answering the assistant), 'choice' (says which doubted wish matters more), 'change' (replaces, adds or drops a requirement or changes the budget on the shopper's own initiative), 'market' (asks about stock or a price change), 'progress' (asks how it is going), 'refusal' (declines to answer or objects to the question), 'restate' (only repeats a budget, category or value already in the known list), 'other'.\n- Being in stock or available, and the price, are never requirements (a new price limit goes in budget).\n- told: requirements the message names that are not in the known list, whatever the wording. A value already in the known list is not told again.\n- replaces: each known requirement the message replaces: its id and the new requirement.\n- drops: ids of known requirements the message says to forget with no replacement.\n- budget: the new price limit when the message writes a new number for it, else null.\n- chosen: 'a' or 'b' when the message says which doubted wish matters more, else null.\n- reply_expected: true when the message asks the assistant anything or asks it to confirm.\n- category: the store's own category name when the message names a store category to shop in, copied exactly; else null.\n- searches: when the message tells or changes a requirement, 3 short catalog search phrases for the item with that value, written the way sellers title such products (other names for the value, product types or well-known product names that have it); else empty.\nTake only values the message itself writes. A value is what the item must have or be (a word, a name, or a number with its unit), never the property's name and never a word that only says it is needed. The property is named as the message names it, not replaced by another one. The value may stand before, after or inside the property's name in the sentence: report the property and the value apart.\n" + _WISH_RULES, 'schema': {'type': 'object', 'additionalProperties': False, 'required': ['kind', 'told', 'replaces', 'drops', 'budget', 'chosen', 'reply_expected', 'category', 'searches'], 'properties': {'kind': {'type': 'string', 'enum': ['need', 'choice', 'change', 'market', 'progress', 'refusal', 'restate', 'other']}, 'told': {'type': 'array', 'items': _WISH}, 'replaces': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'required': ['id', 'new'], 'properties': {'id': {'type': 'string'}, 'new': _WISH}}}, 'drops': {'type': 'array', 'items': {'type': 'string'}}, 'budget': {'type': ['number', 'null']}, 'chosen': {'type': ['string', 'null'], 'enum': ['a', 'b', None]}, 'reply_expected': {'type': 'boolean'}, 'category': {'type': ['string', 'null']}, 'searches': {'type': 'array', 'items': {'type': 'string'}}}}}
LABEL_FORM = {'name': 'labels', 'rules': 'A shop describes its products with labelled facts. For each requirement, list the fact labels whose values state that same property (any spelling or wording of the property counts). Use only labels from the given list; an empty list when none does.', 'schema': {'type': 'object', 'additionalProperties': False, 'required': ['map'], 'properties': {'map': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'required': ['id', 'labels'], 'properties': {'id': {'type': 'string'}, 'labels': {'type': 'array', 'items': {'type': 'string'}}}}}}}}
DOUBT_FORM = {'name': 'doubts', 'rules': "You get one requirement a shopper stated (property and value, in the shopper's words) and numbered product listings (title, variant name, facts). For each listing say 'yes' when the listing itself states that it has the required value, possibly in the store's own words for it (a different spelling, the store's term for the same thing), and 'no' when it does not state it or states something else. Decide only from what is written.", 'schema': {'type': 'object', 'additionalProperties': False, 'required': ['verdicts'], 'properties': {'verdicts': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'required': ['n', 'verdict'], 'properties': {'n': {'type': 'integer'}, 'verdict': {'type': 'string', 'enum': ['yes', 'no']}}}}}}}
PROPERTY_FORM = {'name': 'properties', 'rules': "An online shopper wrote the order below and said why they want the product. Tied to that motive, the shopper keeps a further condition on the product to themselves: the order does not state it. You also get the conditions already read from the order ('known'). Rank six product attributes by how probably this unstated condition concerns each, most probable on top.\nMotives for buying point to kinds of attribute, and the hidden condition sits in the kind that motive indicates:\n- a motive about pleasing someone's eye or personal liking indicates appearance: the colour before anything else, then the material, the pattern or the style;\n- a motive about having enough of something or keeping supplies up indicates an amount: how many pieces first, then weight, volume or capacity;\n- something to be posted, carried or moved about: what it weighs and how many pieces come together;\n- something meant for a fixed spot: its measurements, the length ahead of the other dimensions;\n- something used together with other equipment: its power and connection specifications, the voltage ahead of the plug or connector;\n- a motive that says nothing of these: the attributes shoppers most often pick this sort of product by.\nWhatever the motive, the six include looks (colour, material) and amount (piece count, weight), ranked as the motive suggests, and close with one or two attributes particular to this sort of product.\nSkip attributes the known conditions already fix, and the price, brand, model, stock, delivery and warranty. Name each attribute in plain everyday words, a word or two; for a measurement name the exact measure (which dimension, which amount), not a vague word like 'size'.\nAlso say in 'motive_has_kind' whether the motive itself hints at any kind of attribute: false when it only says that another person settled the choice, or that the product should be just like one the shopper had before, or gives no reason at all; true otherwise.", 'schema': {'type': 'object', 'additionalProperties': False, 'required': ['properties', 'motive_has_kind'], 'properties': {'properties': {'type': 'array', 'items': {'type': 'string'}}, 'motive_has_kind': {'type': 'boolean'}}}}
PURPOSE_FORM = {'name': 'properties', 'rules': PROPERTY_FORM['rules'] + "\nThis order gives more than one motive, and the condition tied to the other one is already known. Work only from the motive quoted under 'purpose'.", 'schema': PROPERTY_FORM['schema']}
VALUE_FORM = {'name': 'values', 'rules': "You get one requirement (property and value) and numbered texts taken from product listings. For each text say: 'yes' when it names that value itself or a plain synonym of it; 'other' only when it names a different value of that same property (a related but different value is 'other'); 'silent' when it is about something else (another property, a code) or says nothing about that property; 'unsure' when it cannot be told; 'mixed' when it names the required value together with a second, different value of that same property (two values of that one property side by side). A text often joins values of different properties with '/', ',' or spaces: read only the part about this property; a size, a count, a code or another property's value beside it is not a second value, and neither is a lighter or darker form of the value itself.", 'schema': {'type': 'object', 'additionalProperties': False, 'required': ['verdicts'], 'properties': {'verdicts': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'required': ['n', 'verdict'], 'properties': {'n': {'type': 'integer'}, 'verdict': {'type': 'string', 'enum': ['yes', 'mixed', 'other', 'silent', 'unsure']}}}}}}}

class Wish:
    _ids = 0

    def __init__(self, raw: dict, origin: str):
        raw = raw if isinstance(raw, dict) else {}
        Wish._ids += 1
        self.id = f'w{Wish._ids}'
        self.origin = origin
        self.property = str(raw.get('property') or '').strip()
        self.value = str(raw.get('value') or '').strip()
        self.kind = raw.get('kind') if raw.get('kind') in ('amount', 'word', 'brand') else 'word'
        self.colour = raw.get('shade') is True
        self.material = raw.get('made_of') is True
        self.amounts: list[tuple[float, set[str]]] = []
        written = amounts(self.value)
        if self.kind != 'brand' and len(written) == 1 and (written[0][0] == written[0][1]):
            self.kind = 'amount'
        if self.kind == 'amount':
            forms = self.spellings(raw.get('unit_forms'))
            number = raw.get('number')
            if written:
                number = number if number is not None else written[0][0]
                forms |= self.spellings([written[0][2]])
            if number is not None and forms:
                if num(number) is not None:
                    self.amounts.append((num(number), forms))
            for alt in raw.get('alt_amounts') or []:
                if not isinstance(alt, dict):
                    continue
                alt_forms = self.spellings(alt.get('unit_forms'))
                if num(alt.get('number')) is not None and alt_forms:
                    self.amounts.append((num(alt['number']), alt_forms))
            if not self.amounts:
                self.kind = 'word'
        self.alts = [str(v) for v in raw.get('alt_values') or [] if isinstance(v, (str, int, float)) and str(v).strip()][:3]
        phrases = [self.value] + self.alts
        prop_t, value_t = (tokens(self.property), tokens(self.value))
        at = next((i for i in range(len(value_t) - len(prop_t)) if prop_t and value_t[i:i + len(prop_t)] == prop_t), None)
        if at is not None and (len(prop_t) >= 2 or at > 0 or len(value_t) - len(prop_t) >= 1):
            words = self.value.split()
            cut = next((j + 1 for j in range(len(words)) if tokens(' '.join(words[:j + 1]))[-len(prop_t):] == prop_t and len(tokens(' '.join(words[:j + 1]))) >= at + len(prop_t)), None)
            rest = ' '.join(words[cut:]) if cut is not None else ''
            if tokens(rest):
                phrases.insert(0, rest)
                self.stripped = tokens(rest)
        self.phrases = [tokens(p) for p in phrases if tokens(p)]
    stripped: list = []
    colour = False
    replaced: str | None = None
    material = False
    said = ''
    alts: list = []

    @staticmethod
    def spellings(forms: Any) -> set[str]:
        out = set()
        if isinstance(forms, str):
            forms = [forms]
        for form in forms if isinstance(forms, (list, tuple, set)) else []:
            word = str(form).casefold().strip().rstrip('.')
            if not word or ' ' in word:
                continue
            out |= {word, word + 's', word[:-1] if word.endswith('s') and len(word) > 1 else word}
        return out

    @property
    def units(self) -> set[str]:
        return {u for _, forms in self.amounts for u in forms}

    def public(self) -> dict:
        return {'id': self.id, 'property': self.property, 'value': self.value}

    def code_verdict(self, text: str) -> str | None:
        if self.kind == 'amount':
            hits = [a for a in amounts(text) if a[2] in self.units]
            if not hits:
                return None
            for lo, hi, unit, _ in hits:
                for number, forms in self.amounts:
                    if unit in forms and (abs(lo - number) < 1e-09 or abs(hi - number) < 1e-09):
                        return 'yes'
            return 'other'
        words = tokens(text)
        if any((has_form(words, p) for p in self.phrases)):
            return 'yes'
        return None

    def literal(self, text: str) -> bool:
        if self.kind == 'amount':
            return self.code_verdict(text) == 'yes'
        return bool(self.phrases) and has_form(tokens(text), self.phrases[0])

    def amount_count(self, text: str) -> int:
        if self.kind != 'amount':
            return 0
        return len({(lo, hi, u) for lo, hi, u, _ in amounts(text) if u in self.units})

    def range_only(self, text: str) -> bool:
        if self.kind != 'amount':
            return False
        hits = [(lo, hi) for lo, hi, u, _ in amounts(text) if u in self.units]
        return bool(hits) and all((lo < hi for lo, hi in hits))

class Episode:

    def __init__(self, problem: dict):
        Wish._ids = 0
        env = problem['environment']
        self.binding = env['binding']
        self.policy = env['policy_view']
        self.request = str(self.policy.get('query') or '')
        self.max_turns = int(self.policy.get('max_steps') or 30)
        self.max_calls = int(self.policy.get('max_calls_per_turn') or 16)
        self.tools = {t['function']['name'] for t in self.policy.get('tools') or []}
        self.tool_flags = {tuple(tokens(str(name).replace('_', ' '))) for t in self.policy.get('tools') or [] for name, spec in ((t.get('function') or {}).get('parameters') or {}).get('properties', {}).items() if isinstance(spec, dict) and spec.get('type') == 'boolean'}
        self.tool_fields = {w for t in self.policy.get('tools') or [] for name, spec in ((t.get('function') or {}).get('parameters') or {}).get('properties', {}).items() if isinstance(spec, dict) and spec.get('type') == 'string' and (name != 'query') for w in tokens(str(name).replace('_', ' '))}
        self.sid = str(self.binding.get('session_id') or problem.get('problem_id') or 's')
        self.start = time.monotonic()
        self.turn_no = 0
        self.spent = 0.0
        self.done = False
        self.notes: list[str] = []
        self.item = ''
        self.item_ref = 'it'
        self.currency = ''
        self.budget: float | None = None
        self.first_budget: float | None = None
        self.wishes: list[Wish] = []
        self.stated_ids: set[str] = set()
        self.pair: dict[str, Wish] | None = None
        self.pair_asked = 0
        self.priority: dict | None = None
        self.purposes: list[dict] = []
        self.reason_asked = False
        self.reason_fields: list[str] = []
        self.searches: list[str] = []
        self.line_searches: list[str] = []
        self.colour_words: set | None = None
        self.sibling_looked = False
        self.top_titles_looked = False
        self.colour_alts: dict[tuple, list[str]] = {}
        self.colour_seen = 0
        self.home_share: dict = {}
        self.twin_queries: list[str] = []
        self.last_message = ''
        self.last_question: tuple | None = None
        self.questions = 0
        self.use_door_closed = False
        self.props_asked: list[str] = []
        self.needs_drawn = 0
        self.ideas: list[str] | None = None
        self.asked_from: list[str] = []
        self.property_door = False
        self.ideas_for: set[int] = set()
        self.list_start = 0
        self.motive_open = False
        self.shopper_changed = False
        self.owed: list[dict] = []
        self.rows: dict[str, dict] = {}
        self.listings: dict[str, dict] = {}
        self.queried: set[str] = set()
        self.row_yes: set[tuple[str, str]] = set()
        self.row_asked: set[tuple[str, str]] = set()
        self.ceilings_done: set[float] = set()
        self.shelf: tuple[str, ...] | None = None
        self.shelf_label: str | None = None
        self.told_label: str | None = None
        self.shelf_rows: set[str] = set()
        self.value_rows: set[str] = set()
        self.first_key: str | None = None
        self.filtered_brands: set[str] = set()
        self.aligned: dict[str, set[str]] = {}
        self.align_seen: dict[str, set[str]] = {}
        self.align_calls: dict[str, int] = {}
        self.verdicts: dict[tuple[str, str], str] = {}
        self.pending: dict[str, set[str]] = {}
        self.cart: list[str] = []
        self.carts = 0
        self.dead: set[str] = set()
        self.events: list[dict] = []
    _label_n = -1
    _generic: set = set()

    def note(self, text: str) -> None:
        self.notes.append(f't{self.turn_no} {text}')

    def seconds_left(self) -> float:
        return EPISODE_SECONDS - (time.monotonic() - self.start)

    def turns_left(self) -> int:
        return self.max_turns - self.turn_no

    def parse(self, form: dict, payload: dict, max_tokens: int=900, models: tuple | None=None) -> dict | None:
        for model in models or FORM_MODELS.get(form['name'], PARSERS):
            if self.spent >= SPEND_CAP or self.seconds_left() < 15:
                return None
            body = {'model': model, 'messages': [{'role': 'system', 'content': form['rules']}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}], 'temperature': 0, 'max_tokens': max_tokens, 'response_format': {'type': 'json_schema', 'json_schema': {'name': form['name'], 'strict': True, 'schema': form['schema']}}}
            if model in NO_THINKING:
                body['reasoning'] = NO_THINKING[model]
            try:
                reply = _infer.post('/inference/chat/completions', json_data=body)
            except Exception:
                reply = None
            if not reply:
                continue
            try:
                self.spent += float((reply.get('usage') or {}).get('cost') or 0.0)
            except (TypeError, ValueError):
                pass
            try:
                text = reply['choices'][0]['message'].get('content') or ''
                data = json.loads(text[text.index('{'):text.rindex('}') + 1])
            except (KeyError, IndexError, ValueError, TypeError):
                continue
            if isinstance(data, dict) and all((k in data for k in form['schema']['required'])):
                if form is REQUEST_FORM and (not tokens(data.get('item'))):
                    continue
                return data
        return None

    def act(self, calls: list[tuple[str, dict]]) -> dict | None:
        calls = [c for c in calls if c[0] in self.tools][:self.max_calls]
        kept, n_search = ([], 0)
        for c in calls:
            if c[0] in ('search', 'filter'):
                n_search += 1
                if n_search > 10:
                    if c[0] == 'search':
                        self.queried.discard(' '.join(str(c[1].get('query', '')).split())[:120].casefold())
                    continue
            kept.append(c)
        calls = kept
        if not calls or self.done or self.turn_no >= self.max_turns:
            return None
        self.turn_no += 1
        gid = f'{self.sid}-tg{self.turn_no}'
        envelope = {**self.binding, 'call_id': gid, 'idempotency_key': gid, 'turn': self.turn_no, 'calls': [{'call_id': f'{gid}-{i}', 'action': {'name': name, 'args': args}} for i, (name, args) in enumerate(calls, 1)]}
        result = _env.post('/environment/call', json_data=envelope)
        if result is None:
            self.note('environment call failed')
            return None
        for (name, args), call in zip(calls, result.get('calls') or []):
            obs = call.get('observation') or {}
            if obs.get('done'):
                self.done = True
            self.absorb(name, args, obs.get('observation') or {}, obs.get('error'))
        asked, self.last_question = (self.last_question, None)
        line = result.get('user_message') or {}
        if line.get('content'):
            carted = any((name == 'add_to_cart' for name, _ in calls))
            self.hear(str(line['content']), asked if not carted else None, carted)
        return result

    def absorb(self, name: str, args: dict, payload: dict, error: Any) -> None:
        if error:
            self.note(f'{name} error {short(str(error), 80)}')
            if name == 'add_to_cart':
                self.dead.add(f"{args.get('product_id')}::{args.get('sku')}")
            return
        if payload.get('market_event'):
            self.market(payload['market_event'])
        if name == 'filter' and args.get('category') and payload.get('results') and (not self.shelf_label) and (str(args['category']).casefold() in self.category_names()):
            self.shelf_label = str(args['category']).casefold()
            self.note(f'category probe hit {self.shelf_label!r}')
        on_shelf = name == 'filter' and str(args.get('category') or '').casefold() in ([self.shelf_label] if self.shelf_label else []) + ([self.shelf[-1].casefold()] if self.shelf else [])
        q = set(tokens(str(args.get('query') or ''))) if name == 'search' else set()
        all_values = bool(q) and bool(self.wishes) and all((set(tokens(self.shelf_spelling(w))) <= q or set(tokens(w.value)) <= q for w in self.wishes))
        for rank_no, row in enumerate(payload.get('results') or []):
            pid = str(row.get('product_id') or '')
            if pid and all_values and (rank_no < 5):
                self.value_rows.add(pid)
            if pid and on_shelf:
                self.shelf_rows.add(pid)
            if pid:
                self.rows.setdefault(pid, dict(row)).update({k: row[k] for k in ('price', 'in_stock') if k in row})
                self.update_variant(pid, str(row.get('sku') or ''), row.get('price'), row.get('in_stock'))
        listings = list(payload.get('compared') or [])
        if name == 'view' and payload.get('product_id') and (payload.get('variants') is not None):
            listings.append(payload)
        for listing in listings:
            pid = str(listing.get('product_id') or '')
            if not pid:
                continue
            variants = {}
            for v in listing.get('variants') or []:
                variants[str(v.get('sku'))] = {'price': v.get('price'), 'in_stock': v.get('in_stock'), 'options': str(v.get('options') or '')}
            self.listings[pid] = {'title': str(listing.get('title') or ''), 'brand': str(listing.get('brand') or ''), 'path': tuple(listing.get('category_path') or ()), 'facts': {str(k): str(v) for k, v in (listing.get('facts') or {}).items() if not running_text(v)}, 'variants': variants}
        if name == 'inspect_stock' or (payload.get('product_id') and payload.get('sku') and ('in_stock' in payload)):
            self.update_variant(str(payload.get('product_id')), str(payload.get('sku')), payload.get('price'), payload.get('in_stock'))
        added = payload.get('added')
        if added:
            key = f"{added.get('product_id')}::{added.get('sku')}"
            self.carts += 1
            self.cart.append(key)
            self.update_variant(str(added.get('product_id')), str(added.get('sku')), added.get('price'), added.get('in_stock'))
        for row in payload.get('cart') or []:
            self.update_variant(str(row.get('product_id')), str(row.get('sku')), row.get('current_price', row.get('price')), row.get('in_stock'))

    def update_variant(self, pid: str, sku: str, price: Any, in_stock: Any) -> None:
        key = f'{pid}::{sku}'
        listing = self.listings.get(pid)
        if listing and sku in listing['variants']:
            v = listing['variants'][sku]
            if price is not None:
                v['price'] = price
            if in_stock is not None:
                v['in_stock'] = in_stock
        if in_stock is False:
            self.dead.add(key)
        elif in_stock is True:
            self.dead.discard(key)

    def market(self, event: dict) -> None:
        target = event.get('target') or {}
        key = f"{target.get('product_id')}::{target.get('sku')}"
        self.events.append(event)
        self.note(f"market {event.get('kind')} {key} new_price={event.get('new_price')}")
        self.update_variant(str(target.get('product_id')), str(target.get('sku')), event.get('new_price'), event.get('in_stock_after'))

    def read_request(self) -> None:
        got = self.parse(REQUEST_FORM, {'request': self.request}, max_tokens=1400) or {}
        if not got:
            got = self.parse(REQUEST_FORM, {'request': self.request}, max_tokens=1400) or {}
        if not (got.get('wishes') or got.get('doubted_pair')):
            other = self.parse(REQUEST_FORM, {'request': self.request}, max_tokens=1400, models=PARSERS[::-1]) or {}
            if other.get('wishes') or other.get('doubted_pair'):
                got = other
        self.item = str(got.get('item') or '').strip() or ' '.join(self.request.split()[:3])
        joined = any((sep in self.item for sep in ('&', '/', ' and ')))
        self.item_ref = 'it' if joined else f'the {self.item}'
        self.currency = str(got.get('currency') or '').strip()
        budget = got.get('budget')
        if budget is None:
            money = re.findall('(\\d[\\d,]*(?:\\.\\d+)?)\\s*[A-Z]{3}\\b', self.request)
            budget = _number(money[0]) if money else None
        self.budget = self.first_budget = num(budget) or None
        pair = got.get('doubted_pair')
        doubt = tokens(pair.get('doubt')) if isinstance(pair, dict) else []
        if isinstance(pair, dict) and pair.get('side_a') and pair.get('side_b') and (len(doubt) >= 2) and has_phrase(tokens(self.request), doubt):
            self.pair = {'a': Wish(pair['side_a'], 'pair'), 'b': Wish(pair['side_b'], 'pair')}
        sides = [tokens(w.value) for w in self.pair.values()] if self.pair else []
        quotes = [set(tokens(p.get('quote'))) for p in got.get('purposes') or []]
        for raw in got.get('wishes') or []:
            wish = Wish(raw, 'stated')
            if not wish.value or tokens(wish.value) in sides or self.tool_state(wish):
                continue
            value_words = set(tokens(wish.value))
            if len(value_words) >= 3 and any((value_words <= q for q in quotes)):
                continue
            self.wishes.append(wish)
            self.stated_ids.add(wish.id)
        if self.pair and isinstance(pair, dict):
            self.check_pair(str(pair.get('doubt') or ''))
        money = {round(float(m.replace(',', '')), 2) for m in re.findall('(\\d[\\d,]*(?:\\.\\d+)?)\\s*[A-Z]{3}\\b', self.request)}
        have = {round(n, 6) for w in self.all_wishes() for n, _ in w.amounts}
        prio_units = set()
        if isinstance(got.get('priority'), dict):
            prio_units = {str(u.get('form') or '').casefold() for u in got['priority'].get('units') or []}
        for lo, hi, unit, at in amounts(self.request):
            if lo != hi or not unit.isalpha() or unit in prio_units or (round(lo, 2) in money) or (round(lo, 6) in have):
                continue
            sentence = re.split('[.!?;]\\s', self.request[:at])[-1] + re.split('[.!?;]\\s', self.request[at:])[0]
            if self.tool_fields & set(tokens(sentence)):
                continue
            if unit.upper() == self.currency.upper() or len(unit) > 12:
                continue
            wish = Wish({'property': unit, 'value': f'{lo:g} {unit}', 'kind': 'amount', 'number': lo, 'unit_forms': [unit], 'alt_amounts': [], 'alt_values': []}, 'stated')
            self.wishes.append(wish)
            self.stated_ids.add(wish.id)
            have.add(round(lo, 6))
            self.note(f'amount the parser left out, kept from the request: {wish.value}')
        prio = got.get('priority')
        if isinstance(prio, dict) and prio.get('property') and (not prio.get('units')):
            self.priority = {'property': str(prio.get('property')), 'sign': 1.0 if prio.get('direction') != 'min' else -1.0, 'units': {}}
        if isinstance(prio, dict) and prio.get('units'):
            units = {}
            for u in prio['units']:
                form = str(u.get('form') or '').casefold().strip().rstrip('.')
                if form:
                    units[form] = num(u.get('size')) or 1.0
            if units:
                self.priority = {'property': str(prio.get('property') or ''), 'sign': 1.0 if prio.get('direction') != 'min' else -1.0, 'units': units}
        for p in got.get('purposes') or []:
            you = str(p.get('you_form') or '').strip().rstrip('.').rstrip(',')
            quote = str(p.get('quote') or '')
            if you:
                self.purposes.append({'quote': quote, 'you': you, 'asked': False, 'drew': None})
        self.reason_asked = bool(got.get('reason_asked'))
        self.reason_fields = [str(f) for f in got.get('reason_fields') or []]
        self.searches = [str(s) for s in got.get('searches') or [] if str(s).strip()][:5]
        self.note(f"request item={self.item!r} budget={self.budget} wishes={[w.value for w in self.wishes]} pair={([w.value for w in self.pair.values()] if self.pair else None)} priority={self.priority} purposes={[p['you'] for p in self.purposes]} reason={self.reason_asked}")

    @staticmethod
    def written_in(wish: Wish, line: str) -> bool:
        if wish.kind == 'amount' and wish.amounts:
            said = {float(x) for x in re.findall('\\d+(?:\\.\\d+)?', line.replace(',', ''))}
            return any((abs(n - x) < 1e-09 for n, _ in wish.amounts for x in said))
        words = set(tokens(line))
        return any((set(p) <= words for p in wish.phrases))

    def own_label(self, wish: Wish, label: str) -> bool:
        prop = [t[:4] for t in tokens(wish.property.replace('_', ' ')) if len(t) >= 3]
        if not prop:
            return True
        self.generic_label_words()
        own = [t[:4] for t in tokens(label.replace('_', ' '))]
        return any((t in prop and t not in self._generic for t in own))

    def plain_entry(self, wish: Wish, text: str, label: str) -> bool:
        if wish.kind == 'amount':
            return True
        known = set(wish.phrases[0] if wish.phrases else []) | set(tokens(self.item)) | set(tokens(wish.property))
        known |= set(tokens(label.replace('_', ' '))) | (set(tokens(self.shelf[-1])) if self.shelf else set())
        return not set(tokens(text)) - known

    def check_pair(self, doubt: str) -> None:
        sentences = re.split('(?<=[.!?])\\s+', self.request)
        doubt_words = tokens(doubt)
        at = next((i for i, x in enumerate(sentences) if has_phrase(tokens(x), doubt_words)), None)
        sentence = ' '.join(sentences[max(0, at - 1):at + 1]) if at is not None else ''
        words = set(tokens(sentence))
        named = lambda w: any((set(p) <= words for p in w.phrases))
        sides = list(self.pair.values())
        if all((named(w) for w in sides)):
            return
        pool = [w for w in sides + self.wishes if named(w)]
        if len(pool) == 2:
            for w in pool:
                if w in self.wishes:
                    self.wishes.remove(w)
                    self.stated_ids.discard(w.id)
            for w in sides:
                if w not in pool:
                    w.origin = 'stated'
                    self.wishes.append(w)
                    self.stated_ids.add(w.id)
            self.pair = {'a': pool[0], 'b': pool[1]}
            for w in pool:
                w.origin = 'pair'
            self.note(f'pair rebuilt from the doubting sentence: {pool[0].value} / {pool[1].value}')
        else:
            self.note('pair dropped: its sides are not the wishes the doubting sentence names')
            for w in sides:
                w.origin = 'stated'
                self.wishes.append(w)
                self.stated_ids.add(w.id)
            self.pair = None

    def tool_state(self, wish: Wish) -> bool:
        return tuple(tokens(wish.value)) in self.tool_flags

    def label_words(self, text: str) -> set[str]:
        self.generic_label_words()
        return {t for t in tokens(str(text or '').replace('_', ' ')) if len(t) >= 3 and t[:4] not in self._generic}

    def generic_label_words(self) -> set[str]:
        labels = {l for listing in self.listings.values() for l in listing['facts']}
        if len(labels) != self._label_n:
            seen: dict[str, int] = {}
            for l in labels:
                for t in {t[:4] for t in tokens(l.replace('_', ' '))}:
                    seen[t] = seen.get(t, 0) + 1
            self._label_n = len(labels)
            self._generic = {t for t, n in seen.items() if len(labels) >= 20 and n > 0.15 * len(labels)}
        return self._generic

    def wish_by_id(self, wid: str) -> Wish | None:
        return next((w for w in self.wishes if w.id == wid), None)

    def hear(self, line: str, asked: tuple | None, carted: bool=False) -> None:
        payload = {'message': line, 'assistant_last_message': self.last_message, 'known': [w.public() for w in self.wishes], 'doubted': {k: w.public() for k, w in self.pair.items()} if self.pair else None}
        got = self.parse(LINE_FORM, payload) or {}
        kind = got.get('kind') or 'other'
        self.note(f'heard [{kind}] {short(line, 120)}')
        answer = not carted and kind not in ('market', 'progress')
        said: list[str] = []
        doubted = [w.value for w in self.pair.values()] if self.pair else []
        if self.pair and got.get('chosen') in ('a', 'b'):
            side = self.pair[got['chosen']]
            side.origin = 'chosen'
            self.wishes.append(side)
            self.stated_ids.add(side.id)
            self.pair = None
            self.note(f'chose {side.value}')
        drew = False
        if kind == 'restate':
            got['replaces'], got['drops'] = ([], [])
        if answer:
            got['replaces'], got['drops'], got['budget'] = ([], [], None)
        replacing = [tokens((rep.get('new') or {}).get('value')) for rep in got.get('replaces') or []]
        budget_said = None
        budget_before = self.budget
        told_category = str(got.get('category') or '').strip()
        cat_words = {t.rstrip('s') for t in tokens(told_category)}
        part_of_item = bool(cat_words) and cat_words < {t.rstrip('s') for t in tokens(self.item)}
        if told_category and has_phrase(tokens(line), tokens(told_category)) and (not part_of_item):
            label = ' '.join(told_category.casefold().split())
            if label != self.shelf_label:
                self.shelf_label, self.told_label = (label, label)
                self.line_searches = [label] + self.line_searches
                self.note(f'shelf told by the shopper {label!r}')
        else:
            told_category = ''
        for raw in got.get('told') or []:
            wish = Wish(raw, 'told' if answer or asked else 'changed')
            wish.said = line
            if not wish.value or any((tokens(w.value) == tokens(wish.value) for w in self.wishes)):
                continue
            if wish.kind == 'amount' and self.currency and (self.currency.casefold() in wish.units):
                if budget_said is None:
                    budget_said = wish.amounts[0][0]
                continue
            if told_category and tokens(wish.value) == tokens(told_category):
                continue
            if tokens(wish.value) in replacing or self.tool_state(wish):
                continue
            if not self.written_in(wish, line):
                self.note(f'not taken (the line does not write it): {wish.value}')
                continue
            if any((tokens(v) == tokens(wish.value) for v in doubted)):
                continue
            self.wishes.append(wish)
            drew = True
            self.note(f'told {wish.property}={wish.value}')
            if not answer:
                self.shopper_changed = True
                prop = ' '.join((t for t in re.split('[_\\s]+', wish.property or '') if t))
                said.append(f'the {prop} is now {wish.value}{KEEP}' if prop else f'it must also have {wish.value}{KEEP}')
        for rep in got.get('replaces') or []:
            old = self.wish_by_id(str(rep.get('id')))
            raw_new = dict(rep.get('new') or {})
            old_guess = self.wish_by_id(str(rep.get('id')))
            if old_guess is not None and old_guess.kind == 'amount' and re.fullmatch('\\s*\\d+(?:\\.\\d+)?\\s*', str(raw_new.get('value') or '')):
                unit = sorted(old_guess.amounts[0][1], key=len)[0]
                raw_new.update({'kind': 'amount', 'number': float(raw_new['value']), 'value': f"{raw_new['value'].strip()} {unit}", 'unit_forms': sorted(old_guess.amounts[0][1])})
            new = Wish(raw_new, 'changed')
            new.said = line
            if not new.value or not self.written_in(new, line) or self.tool_state(new):
                continue
            if old is not None and (not self.written_in(old, line)) and (self.head(old.property) != self.head(new.property)):
                self.note(f'replacement of {old.value} not taken: an addition')
                old = None
            if old is not None:
                new.replaced = old.value
                self.wishes = [new if w is old else w for w in self.wishes]
                self.stated_ids.discard(old.id)
                self.note(f'replaced {old.value} -> {new.value}')
                said.append(f"the {new.property or old.property or 'requirement'} is now {new.value} instead of {old.value}{KEEP}")
            else:
                self.wishes.append(new)
                self.note(f'added {new.value}')
                said.append(f'it must now have {new.value}')
            self.shopper_changed = True
        for wid in got.get('drops') or []:
            old = self.wish_by_id(str(wid))
            if old is not None and (not self.written_in(old, line)):
                self.note(f'drop not taken (the line does not write it): {old.value}')
                continue
            if old is not None:
                self.wishes.remove(old)
                self.shopper_changed = True
                self.note(f'dropped {old.value}')
                said.append(f'{old.value} no longer applies')
        budget = got.get('budget')
        try:
            budget = num(budget)
        except (TypeError, ValueError):
            budget = None
        if budget is None and budget_said is not None and (not answer):
            budget = budget_said
        if budget is None and (not answer) and (kind == 'change') and self.currency:
            money = {round(_number(m), 2) for m in re.findall('(\\d[\\d,]*(?:\\.\\d+)?)\\s*' + re.escape(self.currency) + '\\b', line)}
            if len(money) == 1 and self.budget and (abs(next(iter(money)) - self.budget) > 0.004):
                budget = next(iter(money))
        if budget and self.budget and (abs(budget - self.budget) > 0.004) and re.search(re.escape(f'{budget:.2f}').replace('\\.00', '(?:\\.00)?'), line.replace(',', '')):
            self.note(f'budget {self.budget} -> {budget}')
            self.budget = budget
            self.shopper_changed = True
            said.append(f'your limit is now {budget:.2f} {self.currency}, so I will only order something in stock at or under it'.replace('  ', ' '))
        if got.get('told') or got.get('replaces'):
            anchor = set(tokens(self.item)) | {_stem(a) for a in self.anchors()}
            phrases = [str(q) for q in got.get('searches') or [] if set(tokens(q)) & anchor]
            self.line_searches = phrases[:3] + self.line_searches
        settled_before = sum((1 for q in self.purposes if q.get('drew')))
        if asked and asked[0] == 'purpose':
            self.purposes[asked[1]]['drew'] = drew
            restates = bool(self.budget) and f'{self.budget:.2f}' in line.replace(',', '')
            if not drew and kind in ('need', 'refusal') and (not restates):
                self.purposes[asked[1]]['redo'] = True
            elif not drew:
                self.use_door_closed = True
        elif drew and answer and self.purposes:
            said_words = set(tokens(line))
            p = max(self.purposes, key=lambda q: len(said_words & set(tokens(q['quote'] + ' ' + q['you']))))
            if not p['asked'] and said_words & set(tokens(p['quote'] + ' ' + p['you'])):
                p['asked'], p['drew'] = (True, True)
                self.note(f"purpose settled by this answer: {p['you']}")
        if asked and asked[0] == 'property' and drew:
            self.needs_drawn += 1
            self.property_door = True
            unsettled = [q for q in self.purposes if not q.get('drew')]
            if unsettled and sum((1 for q in self.purposes if q.get('drew'))) == settled_before:
                said_words = set(tokens(line))
                q = max(unsettled, key=lambda q: len(said_words & set(tokens(q['quote'] + ' ' + q['you']))))
                q['asked'], q['drew'] = (True, True)
            self.note(f'need drawn by the question about {asked[1]}')
        if asked and asked[0] == 'pair' and (self.pair is not None):
            self.repeat_pair = True
            self.note('the pair question was not settled: ask it once more')
        if kind == 'market':
            said.append(self.market_words())
        if kind == 'progress':
            said.append('I have found listings that fit and will order shortly')
        keep = self.keeping(with_limit=self.budget == budget_before)
        said = [part.replace(KEEP, keep) for part in said]
        if said or (not answer and got.get('reply_expected')):
            confirm = bool(got.get('reply_expected'))
            self.owed.append({'confirm': confirm, 'parts': said or ['I am keeping every requirement you gave and ordering an option that meets them']})

    def keeping(self, with_limit: bool=True) -> str:
        limit = f' and the {self.budget:.2f} {self.currency} limit'.replace('  ', ' ') if self.budget and with_limit else ''
        return f'; I am keeping your other requirements{limit} as they are'

    @staticmethod
    def head(name: str | None) -> str:
        words = tokens(str(name or '').replace('_', ' '))
        return max(reversed(words), key=len) if words else ''

    def market_words(self) -> str:
        event = self.events[-1] if self.events else {}
        if event.get('kind') == 'stockout' or event.get('in_stock_after') is False:
            return 'that item has just gone out of stock, so I am switching to another in-stock option that meets every requirement'
        if event.get('new_price') is not None:
            return f"its price changed to {float(event['new_price']):.2f} {self.currency}; I am checking it against your limit before ordering".replace('  ', ' ')
        return 'I am re-checking stock and price before ordering'

    def shelf_of(self, listing: dict) -> bool:
        if self.shelf_label:
            return tokens(self.shelf_label) in [tokens(p) for p in listing['path']]
        return self.shelf is None or listing['path'][-1:] == self.shelf[-1:]

    @staticmethod
    def plural(word: str) -> str:
        if re.search('[^aeiou]y$', word):
            return word[:-1] + 'ies'
        if re.search('(s|x|z|ch|sh)$', word):
            return word + 'es'
        return word + 's'

    def category_names(self) -> list[str]:
        item = ' '.join(self.item.casefold().split())
        names = [item]
        parts = re.split('(\\s*&\\s*|\\s+and\\s+)', item)
        if len(parts) > 1:
            names.append(''.join((self.plural(p) if i % 2 == 0 and p.strip() else p for i, p in enumerate(parts))))
        words = item.split()
        if words:
            names.append(' '.join(words[:-1] + [self.plural(words[-1])]))
        return list(dict.fromkeys((n for n in names if n)))[:3]

    def find_shelf(self) -> None:
        if self.shelf_label:
            paths = [l['path'] for l in self.listings.values() if self.shelf_of(l)]
            if paths:
                self.shelf = max(set(paths), key=paths.count)
                self.note(f"shelf by probe {self.shelf_label!r} {' > '.join(self.shelf)}")
                return
            if self.shelf_label == self.told_label:
                return
            self.note(f'probe {self.shelf_label!r} names no path of the listings read: shelf by their paths instead')
            self.shelf_label = None
        if not self.listings:
            return
        want = set(tokens(self.item))
        said = set(tokens(self.request))
        by_leaf: dict[tuple, tuple] = {}
        by_title: dict[tuple, float] = {}
        for listing in self.listings.values():
            path = listing['path']
            if not path:
                continue
            if want & set(tokens(listing['title'])):
                by_title[path] = by_title.get(path, 0.0) + 1.0
            leaf = set(tokens(path[-1]))
            if leaf & (want | said):
                by_leaf[path] = (len(leaf - said) == 0, len(leaf & said), len(leaf & want) / max(1, len(leaf | want)))
        if by_leaf:
            self.shelf = max(by_leaf, key=lambda p: (by_leaf[p], by_title.get(p, 0.0)))
        elif by_title:
            self.shelf = max(by_title, key=lambda p: by_title[p])
        if self.shelf:
            self.note(f"shelf {' > '.join(self.shelf)}")

    def anchors(self, n: int=4) -> list[str]:
        counts: dict[str, int] = {}
        titles = 0
        for listing in self.listings.values():
            if self.shelf and self.shelf_of(listing):
                titles += 1
                for t in set(_TOKEN.findall(listing['title'].casefold())):
                    if len(t) > 2 and (not t[0].isdigit()):
                        counts[t] = counts.get(t, 0) + 1
        said = set(_TOKEN.findall(self.request.casefold()))
        ranked = sorted((t for t in counts if counts[t] <= 0.6 * max(titles, 1) and t not in said), key=lambda t: -counts[t])
        head = self.head_word()
        return [head] + [t for t in ranked if t != head][:n - 1]

    def shelf_spelling(self, wish: Wish) -> str:
        forms = self.value_forms(wish)
        glued = forms[1] if wish.kind == 'amount' and len(forms) > 1 else wish.value
        if wish.kind != 'amount' or not wish.amounts:
            return glued
        number, units = wish.amounts[0]
        seen: dict[tuple, int] = {}
        anyn: dict[tuple, int] = {}
        for listing in self.listings.values():
            if not self.shelf_of(listing):
                continue
            text = ' '.join([listing['title']] + [str(v.get('options') or '') for v in listing['variants'].values()])
            for m in re.finditer('(\\d+(?:\\.\\d+)?)(\\s?)([a-z]+)', text.casefold()):
                if not any((m.group(3) == u or m.group(3).startswith(u) or (len(m.group(3)) > 1 and u.startswith(m.group(3))) for u in units if len(u) > 1 or m.group(3) == u)):
                    continue
                key = (m.group(3), m.group(2) == '')
                anyn[key] = anyn.get(key, 0) + 1
                try:
                    same = abs(float(m.group(1)) - number) < 1e-09
                except ValueError:
                    same = False
                if same:
                    seen[key] = seen.get(key, 0) + 1
        if not seen and anyn:
            seen = anyn
        if not seen:
            return glued
        unit, tight = max(seen, key=lambda k: seen[k])
        return f'{number:g}{unit}' if tight else f'{number:g} {unit}'

    def head_word(self) -> str:
        pool = set(tokens(self.item)) | (set(tokens(self.shelf[-1])) if self.shelf else set())
        counts: dict[str, int] = {}
        for listing in self.listings.values():
            if self.shelf and self.shelf_of(listing):
                for t in set(tokens(listing['title'])) & pool:
                    counts[t] = counts.get(t, 0) + 1
        if counts:
            return max(counts, key=lambda t: counts[t])
        words = self.item.split()
        return words[-1] if words else self.item

    def shelf_labels(self) -> dict[str, list[str]]:
        labels: dict[str, list[str]] = {}
        count: dict[str, int] = {}
        for listing in self.listings.values():
            if not self.shelf_of(listing):
                continue
            for label, value in listing['facts'].items():
                if len(label) > 60:
                    continue
                count[label] = count.get(label, 0) + 1
                seen = labels.setdefault(label, [])
                if len(seen) < 3 and short(value, 40) not in seen:
                    seen.append(short(value, 40))
        top = sorted(labels, key=lambda l: -count[l])[:120]
        return {l: labels[l] for l in top}

    def check_stripped(self) -> None:
        for w in self.all_wishes():
            if not w.stripped:
                continue
            alone = any((tokens(part) == w.stripped for listing in self.listings.values() for text in listing['facts'].values() for part in entries(text)))
            if not alone and len(self.listings) >= 20:
                w.phrases = [p for p in w.phrases if p != w.stripped] or [tokens(w.value)]
                self.note(f'value kept whole: {w.value}')
                w.stripped = []

    def align(self) -> None:
        self.check_stripped()
        labels = self.shelf_labels()
        if not labels:
            return
        fresh = []
        for w in self.all_wishes():
            if w.kind == 'brand':
                continue
            prop = self.label_words(w.property)
            self.aligned.setdefault(w.id, set()).update((l for l in labels if prop and (prop <= set(tokens(l)) or (self.label_words(l) and self.label_words(l) <= prop))))
            if w.kind == 'word' and w.phrases:
                counts: dict[str, int] = {}
                for listing in self.listings.values():
                    if not self.shelf_of(listing):
                        continue
                    for l, text in listing['facts'].items():
                        if any((tokens(part) == w.phrases[0] for part in entries(text))):
                            counts[l] = counts.get(l, 0) + 1
                self.aligned[w.id].update((l for l, n in counts.items() if n >= 2 and l in labels))
            said = tokens(w.said or (self.request if w.origin in ('stated', 'pair', 'chosen') else ''))
            value_at = [i for p in w.phrases for i in range(len(said) - len(p) + 1) if p and said[i:i + len(p)] == p]
            if said and value_at:
                for l in labels:
                    name = [t for t in tokens(l) if t in self.label_words(l)]
                    if len(name) < 2:
                        continue
                    name_at = [i for i in range(len(said) - len(name) + 1) if said[i:i + len(name)] == name]
                    if any((abs(i - j) <= len(name) + 4 for i in name_at for j in value_at)):
                        self.aligned[w.id].add(l)
            seen = self.align_seen.get(w.id)
            if seen is None or (len(set(labels) - seen) >= 5 and self.align_calls.get(w.id, 0) < 3):
                fresh.append(w)
        if not fresh:
            return
        got = self.parse(LABEL_FORM, {'requirements': [w.public() for w in fresh], 'labels': labels}) or {}
        for entry in got.get('map') or []:
            wid = str(entry.get('id'))
            if wid in self.aligned:
                self.aligned[wid] |= {str(l) for l in entry.get('labels') or [] if str(l) in labels}
        for w in fresh:
            self.align_seen[w.id] = set(labels)
            self.align_calls[w.id] = self.align_calls.get(w.id, 0) + 1
            self.note(f'aligned {w.value} -> {sorted(self.aligned[w.id])}')

    def all_wishes(self) -> list[Wish]:
        return self.wishes + (list(self.pair.values()) if self.pair else [])

    def verdict(self, wish: Wish, text: str, ask: bool, mixed_check: bool=False) -> str | None:
        got = wish.code_verdict(text)
        if got == 'yes' and mixed_check and (wish.kind == 'word'):
            extra = {t for t in tokens(text) if not t[0].isdigit()} - {t for p in wish.phrases for t in p}
            if extra:
                key = (wish.id, text)
                if key in self.verdicts:
                    return 'mixed' if self.verdicts[key] == 'mixed' else 'yes'
                if ask:
                    self.pending.setdefault(wish.id, set()).add(text)
        if got:
            return got
        if wish.kind == 'amount':
            return None
        key = (wish.id, text)
        if key in self.verdicts:
            return self.verdicts[key]
        if ask and text.strip():
            self.pending.setdefault(wish.id, set()).add(text)
        return None

    def amount_rule(self, wish: Wish, listing: dict, options: str) -> int | None:

        def figures(text: str) -> list[tuple[float, float, str]]:
            return [(lo, hi, u) for lo, hi, u, _ in amounts(str(text or '').casefold()) if u in wish.units]
        hits = figures(options)
        if not hits:
            hits = figures(listing['title']) + [h for v in listing['facts'].values() for h in figures(v)]
        if not hits:
            return None
        same = lambda lo, hi, u: any((u in forms and abs(hi - n) < 1e-09 for n, forms in wish.amounts))
        if not all((same(*h) for h in hits)):
            return CONTRA
        own = bool(figures(options))
        text = str(options if own else listing['title']).casefold()
        found = [start for lo, hi, u, start in amounts(text) if u in wish.units]
        pack = lambda start: re.search('(?<![\\d.])\\d+\\s*[x\\u00d7*]\\s*$', text[:start])
        sized = lambda start: own and re.search('\\d\\s*[x\\u00d7*]\\s*\\d+(?:\\.\\d+)?\\s*[x\\u00d7*]\\s*$', text[:start])
        if found and all((pack(start) and (not sized(start)) for start in found)):
            return TITLE
        return FIRM

    def status(self, wish: Wish, pid: str, sku: str, ask: bool=True) -> int:
        level = self.read_cell(wish, pid, sku, ask)
        if level > WEAK and self.lacks_home_label(wish, pid):
            return WEAK
        if level > CONTRA and wish.kind == 'word' and (not wish.colour) and (not wish.material):
            full = set(tokens((wish.property or '').replace('_', ' ')))
            if len(full) >= 2:
                facts = self.listings[pid]['facts']
                named = [t for l, t in facts.items() if t and full <= set(tokens(l.replace('_', ' ')))]
                if named:
                    got = [self.verdict(wish, part, ask) for t in named for part in entries(t)]
                    if 'yes' not in got and 'other' in got:
                        return CONTRA
                elif level > TITLE and self.full_label_share(full) >= 0.3:
                    return TITLE
        return level

    def full_label_share(self, full: set) -> float:
        key = (tuple(sorted(full)), len(self.listings))
        if key not in self.home_share:
            shelf = [l for l in self.listings.values() if self.shelf_of(l)]
            has = lambda l: any((full <= set(tokens(label.replace('_', ' '))) for label in l['facts']))
            self.home_share[key] = sum((has(l) for l in shelf)) / len(shelf) if len(shelf) >= 8 else 0.0
        return self.home_share[key]

    @staticmethod
    def options_other_amount(wish: Wish, options: str) -> bool:
        props = [t for t in tokens((wish.property or '').replace('_', ' ')) if len(t) >= 4]
        for m in re.finditer('([a-z]{4,})\\s*[:=]?\\s*(\\d+(?:\\.\\d+)?)\\s*([a-z]+)', str(options).casefold()):
            word, number, unit = (m.group(1), float(m.group(2)), m.group(3))
            if not any((word[:3] == p[:3] and abs(len(word) - len(p)) <= 2 for p in props)):
                continue
            if not any((unit in forms and abs(number - n) < 1e-09 for n, forms in wish.amounts)):
                return True
        return False

    @staticmethod
    def joined_entry(wish: Wish, text: str) -> bool:
        if wish.material:
            return False
        low = str(text).casefold()
        rest = low.replace(str(wish.value).casefold(), ' ')
        replaced = wish.replaced
        return bool(re.search('\\band\\b|&|\\+', rest)) or bool(replaced and str(replaced).casefold() in low)

    def lacks_home_label(self, wish: Wish, pid: str) -> bool:
        if wish.kind != 'word' or wish.colour or wish.material:
            return False
        own = self.label_words(wish.property)
        if not own:
            return False
        has = lambda l: any((own & set(tokens(label.replace('_', ' '))) for label in l['facts']))
        if has(self.listings[pid]):
            return False
        key = (wish.id, len(self.listings))
        if key not in self.home_share:
            shelf = [l for l in self.listings.values() if self.shelf_of(l)]
            self.home_share[key] = sum((has(l) for l in shelf)) / len(shelf) if len(shelf) >= 8 else 0.0
        return self.home_share[key] >= 0.3

    def read_cell(self, wish: Wish, pid: str, sku: str, ask: bool=True) -> int:
        listing = self.listings[pid]
        variant = listing['variants'].get(sku) or {}
        options = variant.get('options') or ''
        if wish.kind == 'amount' and wish.amounts and options and self.options_other_amount(wish, options):
            return CONTRA
        if wish.kind == 'amount' and wish.amounts:
            ruled = self.amount_rule(wish, listing, options)
            if ruled is not None:
                return ruled
        if wish.replaced and wish.kind == 'word':
            old_value = tokens(wish.replaced)
            if not (options and wish.code_verdict(options) == 'yes'):
                for label in self.aligned.get(wish.id, ()):
                    parts = entries(listing['facts'].get(label) or '')
                    if any((tokens(p) == old_value for p in parts)) and (not any((wish.literal(p) for p in parts))):
                        return CONTRA
            if options and tokens(options) == old_value:
                return CONTRA
        if wish.kind == 'brand':
            brand = tokens(listing['brand'])
            if any((has_phrase(brand, p) for p in wish.phrases)):
                return FIRM
            for label, text in listing['facts'].items():
                if self.own_label(wish, label) and any((has_phrase(tokens(text), p) for p in wish.phrases)):
                    return FIRM
            for label in self.aligned.get(wish.id, ()):
                if any((has_phrase(tokens(listing['facts'].get(label, '')), p) for p in wish.phrases)):
                    return FIRM
            if any((has_phrase(tokens(listing['title']), p) for p in wish.phrases)):
                title = listing['title'].casefold()
                if re.search('[/,]\\s*\\w+\\s*[/,]', title):
                    return WEAK
                return TITLE
            return ABSENT
        leaf_name = str(listing['path'][-1]) if listing.get('path') else ''
        leaf = tokens(leaf_name) if not re.search('&|,|/|\\band\\b', leaf_name, re.I) else []
        if wish.kind == 'word' and leaf and any((has_phrase(leaf, p) for p in wish.phrases)):
            own = self.label_words(wish.property)
            against = any((own & set(tokens(label.replace('_', ' '))) and text and (wish.code_verdict(text) != 'yes') for label, text in listing['facts'].items()))
            if not against:
                return FIRM
        fact_level, fact_other, other_named = (None, False, None)
        exact_level = None
        said_by = []
        colour = wish.colour
        own_words = self.label_words(wish.property)
        own_other = own_yes = False
        for label in self.aligned.get(wish.id, ()):
            text = listing['facts'].get(label)
            if not text:
                continue
            label_words = set(tokens(label.replace('_', ' ')))
            if colour and (not self.own_label(wish, label)):
                if wish.code_verdict(text) == 'yes':
                    other_named = WEAK if other_named is None else max(other_named, WEAK)
                continue
            parts = entries(text) if wish.kind != 'amount' else [text]
            got = [self.verdict(wish, part, ask) for part in parts]
            if wish.kind == 'amount' and 'yes' not in got and set(re.split('[_\\s]+', label.casefold())) & wish.units:
                bare = re.fullmatch('\\s*(\\d+(?:\\.\\d+)?)\\s*', text)
                if bare and any((abs(float(bare.group(1)) - n) < 1e-09 for n, forms in wish.amounts)):
                    got = ['yes']
            if 'yes' in got:
                if len(parts) == 1 and wish.amount_count(text) <= 1:
                    if wish.literal(text) and self.joined_entry(wish, text):
                        replaced = wish.replaced
                        level = WEAK if replaced and str(replaced).casefold() in str(text).casefold() else TITLE
                    elif wish.literal(text):
                        level = FIRM
                    else:
                        level = SYN
                    said_by.append((len(label), 'yes'))
                    own_yes = own_yes or bool(own_words & label_words)
                elif any((len(tokens(x)) > len(p) and has_form(tokens(x), p) for x in parts for p in wish.phrases[:1] if tokens(x) != p)):
                    level = WEAK
                elif wish.material:
                    level = TITLE if wish.code_verdict(listing['title']) == 'yes' else WEAK
                else:
                    long_list = len(parts) >= 8 or len(str(text)) >= 155
                    level = TITLE if not long_list and any((wish.literal(x) for x in parts)) and (not self.joined_entry(wish, text)) else WEAK
                fact_level = level if fact_level is None else max(fact_level, level)
                if own_words and self.label_words(label) == own_words:
                    exact_level = level if exact_level is None else max(exact_level, level)
            elif len(parts) == 1 and got[0] == 'other':
                fact_other = True
                said_by.append((len(label), 'other'))
                own_other = own_other or bool(own_words & label_words)
            elif len(parts) == 1 and wish.kind == 'word' and (got[0] in ('silent', 'unsure')) and own_words and self.label_words(label) and (self.label_words(label) <= own_words):
                fact_other = True
                said_by.append((len(label), 'other'))
                own_other = True
            elif 'other' in got and own_words & label_words:
                own_other = True
        if said_by and max(said_by)[1] == 'other' and any((v == 'yes' for _, v in said_by)) and (not (own_yes and (not own_other))):
            fact_level = None
        if exact_level is not None and fact_level is not None and (exact_level < fact_level):
            fact_level = exact_level
        if own_other and (not own_yes):
            fact_level = None
        word = wish.kind == 'word' and (not colour) and (not wish.material)
        if word and own_other and (not own_yes) and re.search('\\d', wish.value or '') and options and (wish.code_verdict(options) == 'yes'):
            own_other = False
        if word and own_other and (not own_yes):
            return CONTRA
        if own_other and (not own_yes) and wish.material and (fact_level is None) and (wish.code_verdict(listing['title']) == 'yes'):
            return TITLE
        v_opt = self.verdict(wish, options, ask, mixed_check=True) if options else None
        parts = [p.strip() for p in re.split('\\s/\\s|\\s/|/\\s', options or '') if p.strip()]
        if colour and len(parts) > 1:
            got = [self.verdict(wish, p, ask, mixed_check=True) for p in parts]
            whole = v_opt
            if 'yes' in got and ('other' in got or 'mixed' in got):
                v_opt = whole if whole in ('yes', 'mixed') else 'mixed'
            elif 'yes' in got:
                v_opt = 'yes'
            elif 'other' in got:
                v_opt = 'other'
            elif 'mixed' in got:
                v_opt = 'mixed'
            else:
                v_opt = None if any((g is None for g in got)) else v_opt
        if colour and v_opt == 'yes' and self.colour_beside(wish, options):
            return CONTRA
        if v_opt == 'other' and colour:
            return CONTRA
        opt_level = None
        if colour and v_opt in ('yes', 'mixed') and options:
            self.colour_beside(wish, '')
            if self.colour_words:
                want = {t for p in wish.phrases for t in p}
                found = {t for t in tokens(options) if t in self.colour_words} - want
                seen = {t: self.verdict(wish, t, ask) for t in found}
                if any((v == 'other' for v in seen.values())):
                    return CONTRA
                if all((v == 'yes' for v in seen.values())):
                    v_opt = 'yes'
        if v_opt == 'mixed' and colour:
            return CONTRA
        if v_opt == 'mixed':
            opt_level = WEAK
        elif v_opt == 'yes':
            if wish.amount_count(options) > 1:
                opt_level = WEAK
            elif wish.range_only(options):
                opt_level = TITLE
            else:
                opt_level = FIRM if wish.literal(options) else SYN
        if colour and opt_level is None and (fact_level is not None) and (fact_level > TITLE):
            fact_level = TITLE
        if fact_level is not None:
            best = max(fact_level, opt_level if opt_level is not None else ABSENT)
        elif opt_level is not None:
            best = opt_level
        elif fact_other or v_opt == 'other':
            return CONTRA
        else:
            best = other_named if other_named is not None else ABSENT
        if best >= TITLE or fact_level is not None:
            return best
        title_verdict = wish.code_verdict(listing['title'])
        if title_verdict == 'yes' and colour:
            return max(best, WEAK)
        if title_verdict == 'yes':
            if wish.amount_count(listing['title']) <= 1:
                if wish.kind == 'amount' and (not wish.range_only(listing['title'])):
                    return max(best, FIRM)
                if wish.kind == 'word' and (not wish.material):
                    return max(best, WEAK)
                return max(best, TITLE)
            number = wish.amounts[0][0] if wish.amounts else None
            bare = [float(x) for x in re.findall('\\d+(?:\\.\\d+)?', options)]
            return max(best, TITLE if number is not None and bare == [number] else WEAK)
        page = ' ; '.join([listing['title'], options] + list(listing['facts'].values()))
        for label, text in listing['facts'].items():
            if label not in self.aligned.get(wish.id, ()) and wish.code_verdict(text) == 'yes':
                if wish.kind == 'amount' and wish.amount_count(page) <= 1 and (not wish.range_only(text)):
                    return max(best, FIRM)
                return max(best, WEAK)
        return best

    def resolve_pending(self) -> None:
        jobs = []
        for wid, texts in self.pending.items():
            wish = next((w for w in self.all_wishes() if w.id == wid), None)
            if wish is None:
                continue
            todo = [t for t in texts if (wid, t) not in self.verdicts][:80]
            if todo:
                jobs.append((wish, todo))
        self.pending = {}
        if not jobs:
            return

        def run(job):
            wish, texts = job
            got = self.parse(VALUE_FORM, {'requirement': {'property': wish.property, 'value': wish.value}, 'texts': [{'n': i, 'text': short(t, 200)} for i, t in enumerate(texts)]}, max_tokens=40 + 18 * len(texts)) or {}
            return (wish, texts, got)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for wish, texts, got in pool.map(run, jobs):
                answered = {int(v.get('n', -1)): v.get('verdict') for v in got.get('verdicts') or []}
                for i, text in enumerate(texts):
                    self.verdicts[wish.id, text] = answered.get(i) or 'unsure'

    def price(self, pid: str, sku: str) -> float | None:
        v = self.listings[pid]['variants'].get(sku) or {}
        try:
            p = float(v.get('price'))
        except (TypeError, ValueError):
            return None
        return p if p > 0 else None

    def figure(self, pid: str, sku: str) -> float | None:
        if not self.priority:
            return None
        if not self.priority['units']:
            self.learn_priority_units()
        listing = self.listings[pid]
        units = self.priority['units']
        if not units:
            return None
        prop = self.label_words(self.priority['property'])
        own = [str(v) for k, v in listing['facts'].items() if prop & self.label_words(k)]
        rest = [str(v) for k, v in listing['facts'].items() if not prop & self.label_words(k)]
        sources = [(listing['variants'].get(sku) or {}).get('options') or '', ' ; '.join(own), listing['title'], ' ; '.join(rest)]
        for at, text in enumerate(sources):
            values, sides = (set(), set())
            vague: set[str] = set()
            plain: dict[float, str] = {}
            low = str(text).casefold()
            for lo, hi, unit, start in amounts(low):
                if unit not in units:
                    continue
                if re.match('[\\d.,]+\\s+' + re.escape(unit) + '\\s*1(?![\\d.])\\s*(?![a-z\\"]*\\b(?:' + '|'.join((re.escape(u) for u in units if u.isalpha())) + ')\\b)', low[start:]):
                    continue
                if start > 0 and low[start - 1].isalpha() and (low[start - 1] != 'x'):
                    continue
                if lo == 0 < hi and (not re.search('\\d\\s*/\\s*$', low[:start])):
                    lo = hi
                if lo != hi or re.search('\\d\\s*/\\s*$', low[:start]):
                    values.add(None)
                    vague.add(unit)
                elif start > 0 and low[start - 1] in 'x*×':
                    sides.add(round(lo * units[unit], 6))
                else:
                    value = round(lo * units[unit], 6)
                    values.add(value)
                    plain[value] = unit
            pool = values
            base = {u for u, size in units.items() if abs(size - 1.0) < 1e-12}
            if plain and sides and (not set(plain.values()) & base) and (len(sides) == 1):
                return self.counted(next(iter(sides)))
            if at == 0 and len(plain) == 1:
                tlow = str(listing['title']).casefold()
                hits = [(lo, hi, u, st) for lo, hi, u, st in amounts(tlow) if u in units]
                listed = any((lo != hi or re.search('\\d\\s*/\\s*$', tlow[:st]) for lo, hi, u, st in hits))
                title_vals = {round(lo * units[u], 6) for lo, hi, u, st in hits}
                if not listed and len(title_vals) == 1 and (next(iter(plain)) not in title_vals):
                    return None
            if len(plain) == 1 and next(iter(plain.values())) not in vague:
                return self.counted(next(iter(plain)))
            if pool:
                return self.counted(pool.pop() if len(pool) == 1 and None not in pool else None)
            if at == 0 and sides:
                return self.counted(sides.pop()) if len(sides) == 1 else None
        bare = set()
        for label, text in listing['facts'].items():
            unit = re.split('[_\\s]+', label.casefold())[-1]
            m = re.fullmatch('\\s*(\\d+(?:\\.\\d+)?)\\s*', str(text))
            if m and unit in units:
                bare.add(round(float(m.group(1)) * units[unit], 6))
        return self.counted(bare.pop() if len(bare) == 1 else None)

    def counted(self, figure: float | None) -> float | None:
        return figure

    def learn_priority_units(self) -> None:
        prop = self.label_words(self.priority['property'])
        seen: dict[str, int] = {}
        for listing in self.listings.values():
            if not self.shelf_of(listing):
                continue
            for label, text in listing['facts'].items():
                if prop & set(tokens(label.replace('_', ' '))):
                    for lo, hi, unit, _ in amounts(text):
                        if unit.isalpha() or unit == '"':
                            seen[unit] = seen.get(unit, 0) + 1
        if seen:
            top = max(seen, key=lambda u: seen[u])
            self.priority['units'] = {top: 1.0}
            self.note(f'priority unit learned from the shelf: {top}')

    def candidates(self, wishes: list[Wish], budget: float | None, ask: bool=False) -> list[dict]:
        out = []
        ask_cap = budget * STRETCH if budget else None
        for pid, listing in self.listings.items():
            if not self.shelf_of(listing):
                continue
            for sku, v in listing['variants'].items():
                key = f'{pid}::{sku}'
                if v.get('in_stock') is False or key in self.dead:
                    continue
                price = self.price(pid, sku)
                if price is None:
                    continue
                asking = ask and (ask_cap is None or price <= ask_cap + 1e-09)
                marks = [self.status(w, pid, sku, asking) for w in wishes]
                marks = [SYN if m == ABSENT and (w.id, key) in self.row_yes else m for w, m in zip(wishes, marks)]
                figure = self.figure(pid, sku)
                out.append({'key': key, 'pid': pid, 'sku': sku, 'price': price, 'marks': marks, 'figure': figure, 'within': budget is None or price <= budget + 1e-09})
        return out

    def solve(self, wishes: list[Wish] | None=None, budget: float | None=None) -> list[dict]:
        wishes = self.wishes if wishes is None else wishes
        budget = self.budget if budget is None else budget
        self.candidates(wishes, budget, ask=True)
        if self.pending and self.seconds_left() > 30:
            self.resolve_pending()
        return self.candidates(wishes, budget)

    @staticmethod
    def tier(marks: list[int]) -> int:
        if not marks:
            return 3
        low = min(marks)
        return low if low > ABSENT else 0 if low == ABSENT else -1

    def rank(self, rows: list[dict]) -> list[dict]:
        sign = self.priority['sign'] if self.priority else 1.0
        outlier = None

        def level(r):
            t = self.tier(r['marks'])
            if self.priority and t == TITLE and (len(r['marks']) == len(self.wishes)) and all((m != TITLE or w.material for w, m in zip(self.wishes, r['marks']))):
                return FIRM
            return t

        def key(r):
            fig = r['figure']
            support = sum((self.sources(w, r['pid'], r['sku']) for w in self.wishes))
            odd = int(fig is not None and outlier is not None and (fig * sign == outlier))
            own = sum((self.own_label_states(w, r['pid']) for w in self.wishes if w.kind == 'word'))
            near = sum((self.near_form(w, r['pid']) for w in self.wishes if w.kind == 'word'))
            known = 0 if fig is not None or not self.priority else 1
            syn_near = sum((self.near_form(w, r['pid']) for w, m in zip(self.wishes, r['marks']) if w.kind == 'word' and m == SYN))
            return (-level(r), known, sum((m < TITLE for m in r['marks'])), -syn_near, odd, 0 if fig is not None else 1, -(fig or 0.0) * sign, -self.tier(r['marks']), -own, -near, -support, -self.names_item(r['pid']), -self.sibling_of_first(r), -sum((m == FIRM for m in r['marks'])), r['price'])
        return sorted(rows, key=key)

    def shelf_writes_plain(self, wish: 'Wish', label: str) -> int:
        want = list(wish.phrases[0]) if wish.phrases else []
        if not want:
            return 0
        n = 0
        for listing in self.listings.values():
            if not self.shelf_of(listing):
                continue
            text = listing['facts'].get(label)
            if text and any((tokens(p) == want for p in entries(text))):
                n += 1
        return n

    def sibling_of_first(self, row: dict) -> int:
        if not self.first_key:
            return 0
        first_sku = self.first_key.partition('::')[2]
        return int(first_sku.split('-')[0] == row['sku'].split('-')[0] and row['key'] != self.first_key)

    def names_item(self, pid: str) -> int:
        words = set(tokens(self.item)) | (set(tokens(self.shelf[-1])) if self.shelf else set())
        return int(bool(words & set(tokens(self.listings[pid]['title']))))

    def fitting(self, rows: list[dict], floor: int=TITLE) -> list[dict]:
        return self.rank([r for r in rows if r['within'] and self.tier(r['marks']) >= floor])

    def value_forms(self, wish: Wish) -> list[str]:
        forms = [wish.value]
        if wish.kind == 'amount' and wish.amounts:
            number, units = wish.amounts[0]
            written = amounts(wish.value)
            unit = written[0][2] if written and written[0][2].isalpha() else None
            if unit is None:
                words = [u for u in units if u.isalpha()] or sorted(units)
                unit = min(words, key=lambda u: (abs(len(u) - 4), u))
            forms += [f'{number:g}{unit}', f'{number:g} {unit}']
            for n, us in wish.amounts[1:3]:
                u = min(us, key=len)
                if u.isalpha() and n >= 0.1:
                    forms.append(f'{n:g}{u}')
        return list(dict.fromkeys((f for f in forms if f)))

    def shelf_colour_alts(self, wish: Wish) -> list[str]:
        shelf = [l for l in self.listings.values() if self.shelf_of(l)]
        key = (wish.id, len(shelf) // 20)
        if key in self.colour_alts:
            return self.colour_alts[key]
        if len(shelf) < 20:
            return []
        own = set((t for p in wish.phrases for t in p))
        counts: dict[str, int] = {}
        for l in shelf:
            texts = [str(v.get('options') or '') for v in l['variants'].values()]
            texts += [str(t) for label, t in l['facts'].items() if self.own_label(wish, label)]
            for text in texts:
                for part in re.split('[/,;|()\\[\\]-]', text):
                    for t in tokens(part):
                        if len(t) > 2 and (not t[0].isdigit()) and (t not in own):
                            counts[t] = counts.get(t, 0) + 1
        words = sorted(counts, key=lambda t: -counts[t])[:80]
        for t in words:
            self.verdict(wish, t, True)
        self.resolve_pending()
        found = [t for t in words if self.verdicts.get((wish.id, t)) == 'yes'][:2]
        self.colour_alts[key] = found
        if found:
            self.note(f'shelf colour words for {wish.value}: {found}')
        return found

    @staticmethod
    def other_unit_forms(wish: Wish) -> list[str]:
        out = []
        for n, us in wish.amounts[1:4]:
            u = min((u for u in us if u.isalpha()), key=len, default=None)
            if u and n >= 0.1:
                out.append(f'{n:g}{u}')
        return list(dict.fromkeys(out))

    @staticmethod
    def spaced_unit_forms(wish: Wish) -> list[str]:
        out = []
        for n, us in wish.amounts[1:2]:
            letters = [u for u in us if u.isalpha()]
            words = [u for u in letters if len(u) > 2]
            if n == 1:
                words = [u for u in words if not u.endswith('s')] or words
            for u in (max(words, key=len, default=None), min(letters, key=len, default=None)):
                if u and n >= 0.1:
                    out.append(f'{n:g} {u}')
        return list(dict.fromkeys(out))

    def search_calls(self, queries: list[str], limit: int) -> list[tuple[str, dict]]:
        calls = []
        for q in queries:
            q = ' '.join(q.split())[:120]
            if q and q.casefold() not in self.queried and (len(calls) < limit):
                self.queried.add(q.casefold())
                calls.append(('search', {'query': q, 'in_stock': True, 'k': 10}))
        return calls

    def opening_queries(self) -> list[str]:
        wishes = self.all_wishes()
        parts = [p for p in re.split('\\s*(?:&|,|/|\\band\\b)\\s*', self.item) if p.strip()] or [self.item]
        glued = [self.value_forms(w)[1] if w.kind == 'amount' and len(self.value_forms(w)) > 1 else w.value for w in wishes]
        qs = [' '.join(glued + [parts[0]]), ' '.join(glued + [self.item])] if glued else []
        qs += list(self.searches)
        for w in wishes:
            if w.colour:
                qs += [' '.join([alt if x is w else g for x, g in zip(wishes, glued)] + [parts[0]]) for alt in w.alts[:2]]
        for w in wishes:
            for f in self.value_forms(w)[:2]:
                qs.append(f'{f} {self.item}')
        if wishes:
            qs.append(' '.join([w.value for w in wishes] + [self.item]))
        qs.append(self.item)
        return qs

    def value_queries(self, wishes: list[Wish]) -> list[str]:
        anchors = self.anchors(8)
        head = anchors[0]
        others = [w for w in self.wishes if w not in wishes]
        group = wishes + others
        parts = [p for p in re.split('\\s*(?:&|,|/|\\band\\b)\\s*', self.item) if p.strip()] or [self.item]
        glued_all = [self.value_forms(w)[1] if w.kind == 'amount' and len(self.value_forms(w)) > 1 else w.value for w in group]
        spelled = [self.shelf_spelling(w) for w in group]
        item_words = [t for t in tokens(self.item) if len(t) > 3]
        qs = [' '.join(spelled + [parts[0]]), ' '.join(spelled + [self.item])]
        qs += [' '.join(spelled + [t]) for t in item_words][:3]
        if len(group) > 1:
            qs += [' '.join(spelled), ' '.join(spelled + [head])]
        qs += [' '.join(spelled + [a]) for a in anchors[1:5]]
        written = [w.value for w in group]
        if written != spelled:
            qs += [' '.join(written + [a]) for a in anchors[1:3]]
            qs += [' '.join(written + [t]) for t in item_words][:2]
        shelf = [l for l in self.listings.values() if self.shelf_of(l)]
        if len(shelf) >= 20 and len(group) > 1:
            text = [tokens(' '.join([l['title']] + [str(v.get('options') or '') for v in l['variants'].values()])) for l in shelf]
            kept = [g for w, g in zip(group, spelled) if w.kind not in ('word', 'brand') or any((any((has_form(t, p) for p in w.phrases[:1])) for t in text))]
            if kept and len(kept) < len(group):
                qs += [' '.join(kept + [a]) for a in anchors[1:4]] + [' '.join(kept + [parts[0]])]
        for w in group:
            if w.colour:
                for alt in (self.shelf_colour_alts(w) + w.alts)[:3]:
                    qs.append(' '.join([alt if x is w else g for x, g in zip(group, spelled)] + [parts[0]]))
            elif w.kind == 'amount':
                for alt in self.other_unit_forms(w)[:2]:
                    qs.append(' '.join([alt if x is w else g for x, g in zip(group, spelled)] + [parts[0]]))
                for alt in self.spaced_unit_forms(w)[:2]:
                    qs.append(' '.join([alt if x is w else g for x, g in zip(group, spelled)] + [parts[0]]))
        if len(group) > 1:
            qs += [' '.join(glued_all + [parts[0]]), ' '.join([w.value for w in group] + [self.item]), ' '.join(spelled + [parts[-1]])]
        if len(group) > 2:
            qs += [' '.join(spelled[:j] + spelled[j + 1:] + [parts[0]]) for j in range(min(4, len(group)))]
        qs += [' '.join(spelled + [a]) for a in self.anchors(30)[5:30]]
        if len(wishes) + len(others) > 1:
            every = [w.value for w in wishes + others]
            glued = [self.value_forms(w)[1] if w.kind == 'amount' and len(self.value_forms(w)) > 1 else w.value for w in wishes + others]
            qs += [' '.join(glued + [self.item]), ' '.join(glued + [head]), ' '.join(glued)]
            last = self.item.split()[-1] if self.item.split() else head
            qs += [f'{g} {last}' for g, w in zip(glued, wishes + others) if w.kind == 'amount'][:2]
            if len(glued) > 2:
                qs += [f'{a} {b}' for i, a in enumerate(glued) for b in glued[i + 1:]][:3]
            parts = [p for p in re.split('\\s*(?:&|,|/|\\band\\b)\\s*', self.item) if p.strip()]
            qs += [' '.join(every + [self.item])] + [' '.join(every + [p]) for p in parts if p != self.item][:2]
            qs.append(' '.join(every + [head]))
            group = wishes + others
            for w in group:
                for alt in w.alts[:3]:
                    swapped = [alt if x is w else g for x, g in zip(group, glued_all)]
                    qs.append(' '.join(swapped + [parts[0]]))
                    qs.append(' '.join(swapped + [head]))
            titles = [tokens(l['title']) for l in self.listings.values() if self.shelf_of(l)]
            titled = [w.value for w in wishes + others if any((any((has_form(t, p) for p in w.phrases[:1])) for t in titles))]
            if titled:
                if titled != every:
                    qs.append(' '.join(titled + [self.item]))
                qs += [' '.join(titled + [a, head]) for a in anchors[1:6]]
            qs += [' '.join(every + [a]) for a in anchors[1:3]]
        qs += list(self.line_searches)
        for w in wishes:
            forms = self.value_forms(w)
            closed = forms[1] if len(forms) > 1 else forms[0]
            qs += [f'{closed} {head}', f'{forms[0]} {self.item}']
            for o in others:
                o_forms = self.value_forms(o)
                o_closed = o_forms[1] if o.kind == 'amount' and len(o_forms) > 1 else o_forms[0]
                qs += [f'{closed} {o_closed} {parts[0]}', f'{closed} {o_closed} {head}', f'{closed} {o_closed}']
        for a in anchors[1:]:
            for w in wishes[:2]:
                qs.append(f'{w.value} {a}')
        return qs

    def sweep_calls(self) -> list[tuple[str, dict]]:
        label = self.shelf_label or (self.shelf[-1] if self.shelf else None)
        if not label or not self.budget:
            return []
        calls = [('filter', {'category': label, 'max_price': round(self.budget * share, 2), 'in_stock': True, 'k': 10}) for share in (1.0, 0.45, 0.25, 0.12)]
        brands: dict[str, int] = {}
        for pid, row in self.rows.items():
            b = str(row.get('brand') or '').strip()
            listing = self.listings.get(pid)
            if b and (listing is None or self.shelf_of(listing)):
                brands[b] = brands.get(b, 0) + 1
        for b in sorted(brands, key=lambda x: -brands[x])[:4]:
            self.filtered_brands.add(b.casefold())
            calls.append(('filter', {'category': label, 'brand': b, 'max_price': round(self.budget, 2), 'in_stock': True, 'k': 10}))
        return calls

    def ceiling_calls(self, n: int) -> list[tuple[str, dict]]:
        label = self.shelf_label or (self.shelf[-1] if self.shelf else None)
        if not label or not self.budget:
            return []
        shares = [x for x in (0.35, 0.7, 0.18, 0.06) if x not in self.ceilings_done][:n]
        self.ceilings_done.update(shares)
        return [('filter', {'category': label, 'max_price': round(self.budget * x, 2), 'in_stock': True, 'k': 10}) for x in shares]

    def brand_calls(self, n: int) -> list[tuple[str, dict]]:
        label = self.shelf_label or (self.shelf[-1] if self.shelf else None)
        if not label or not self.budget:
            return []
        brands: dict[str, int] = {}
        for pid, row in self.rows.items():
            b = str(row.get('brand') or '').strip()
            listing = self.listings.get(pid)
            if b and b.casefold() not in self.filtered_brands and (listing is None or self.shelf_of(listing)):
                brands[b] = brands.get(b, 0) + 1
        calls = []
        for b in sorted(brands, key=lambda x: -brands[x])[:n]:
            self.filtered_brands.add(b.casefold())
            calls.append(('filter', {'category': label, 'brand': b, 'max_price': round(self.budget, 2), 'in_stock': True, 'k': 10}))
        return calls

    def probe_calls(self) -> list[tuple[str, dict]]:
        args = {'in_stock': True, 'k': 10}
        if self.budget:
            args['max_price'] = round(self.budget, 2)
        return [('filter', {**args, 'category': name}) for name in self.category_names()]

    def unread(self, limit: int, any_price: bool=False) -> list[str]:
        cap = MAX_READ if self.carts else MAX_READ_BEFORE_CART
        if len(self.listings) >= cap:
            return []
        limit = min(limit, cap - len(self.listings))
        cap = (self.budget or 0) * READ_MARGIN if self.budget and (not any_price) else None
        want = set(tokens(self.item)) | {t for w in self.all_wishes() for p in w.phrases for t in p}
        shelf_words = set(tokens(self.shelf[-1])) if self.shelf else set()
        scored = []
        for pid, row in self.rows.items():
            if pid in self.listings:
                continue
            try:
                price = float(row.get('price') or 0)
            except (TypeError, ValueError):
                price = 0
            if row.get('in_stock') is False or (cap and price > cap):
                continue
            title = set(tokens(row.get('title')))
            on_shelf = (2.0 if pid in self.shelf_rows else 0.0) + (4.0 if pid in self.value_rows else 0.0)
            scored.append((len(title & want) + 0.5 * len(title & shelf_words) + on_shelf, pid))
        scored.sort(key=lambda s: -s[0])
        return [pid for _, pid in scored[:min(limit, MAX_READ - len(self.listings))]]

    def compare_calls(self, pids: list[str]) -> list[tuple[str, dict]]:
        return [('compare', {'product_ids': pids[i:i + 5]}) for i in range(0, len(pids), 5)]

    def question(self) -> str | None:
        if self.pair is not None and self.pair_asked < 2 and (self.pair_asked == 0 or self.repeat_pair):
            self.pair_asked += 1
            self.repeat_pair = False
            a, b = (self.pair['a'], self.pair['b'])
            self.last_question = ('pair',)
            return f'Which matters more to you: {self.say(a)}, or {self.say(b)}?'
        self.property_ideas()
        after = 0 if self.motive_open else PURPOSE_AFTER
        if len(self.props_asked) - self.needs_drawn >= after and self.needs_open() > 0 and (not self.use_door_closed):
            ask = self.purpose_question()
            if ask:
                return ask
        ask = self.property_question()
        if ask:
            return ask
        if not self.ideas and self.shelf_unread() and (self.needs_open() > 0) and (len(self.props_asked) < PROPERTY_QUESTIONS):
            return None
        return None if self.use_door_closed else self.purpose_question()
    repeat_pair = False

    def purpose_question(self) -> str | None:
        for i, p in enumerate(self.purposes):
            if not p['asked'] and (not p.get('drew')):
                if self.pair_asked and (not self.worth_asking_purpose()):
                    continue
                p['asked'] = True
                self.last_question = ('purpose', i)
                return f"Since {p['you']}, what would you need {self.item_ref} to have, given how you'll use it?"
        for i, p in enumerate(self.purposes):
            if p.get('redo') and (not p.get('reasked')) and (self.turns_left() > TURN_RESERVE + 3):
                p['reasked'] = True
                self.last_question = ('purpose', i)
                return f"And since {p['you']}, what would you need {self.item_ref} to have for that use, given how you'll use it?"
        return None

    def shelf_unread(self) -> bool:
        return self.turn_no < 2 or (sum((1 for l in self.listings.values() if self.shelf_of(l) and l['facts'])) < 4 and self.turn_no < 4)

    def needs_open(self) -> int:
        if self.purposes:
            return sum((1 for p in self.purposes if not p.get('drew')))
        return 0

    def property_question(self) -> str | None:
        if self.needs_open() <= 0 or len(self.props_asked) >= PROPERTY_QUESTIONS:
            return None
        if self.turns_left() <= TURN_RESERVE + 2 or self.seconds_left() < 60:
            return None
        label = self.next_property()
        if label is None:
            return None
        self.props_asked.append(label)
        self.last_question = ('property', label)
        name = ' '.join((t for t in re.split('[_\\s]+', label) if len(t) > 2 or t == 'of')) or label
        return f'What {name} do you need?'

    def property_ideas(self) -> list[str]:
        if self.ideas is not None:
            return self.ideas
        payload = {'request': self.request, 'known': [w.public() for w in self.all_wishes()]}
        got = self.parse(PROPERTY_FORM, payload, max_tokens=300) or {}
        ideas: list[str] = []
        for raw in got.get('properties') or []:
            name = ' '.join(str(raw or '').replace('_', ' ').split()).strip(' ?.').casefold()
            if name and len(name) <= 40 and (name not in ideas):
                ideas.append(name)
        self.ideas = ideas[:6]
        self.motive_open = got.get('motive_has_kind') is False
        self.note(f'properties that may be unsaid: {self.ideas}')
        return self.ideas

    def purpose_ideas(self) -> None:
        if not self.property_door or self.ideas is None or len(self.purposes) < 2:
            return
        open_ = [i for i, p in enumerate(self.purposes) if not p.get('drew')]
        if not open_ or open_[0] in self.ideas_for:
            return
        i = open_[0]
        self.ideas_for.add(i)
        payload = {'request': self.request, 'known': [w.public() for w in self.all_wishes()], 'purpose': self.purposes[i]['quote']}
        got = self.parse(PURPOSE_FORM, payload, max_tokens=300) or {}
        ideas: list[str] = []
        for raw in got.get('properties') or []:
            name = ' '.join(str(raw or '').replace('_', ' ').split()).strip(' ?.').casefold()
            if name and len(name) <= 40 and (name not in ideas):
                ideas.append(name)
        if ideas:
            asked = set(self.props_asked)
            rest = [n for n in self.ideas if n not in asked]
            merged: list[str] = []
            queues = [list(ideas[:6]), rest]
            turn = 0
            while any(queues):
                queue = queues[turn % 2] or queues[(turn + 1) % 2]
                while queue and queue[0] in merged:
                    queue.pop(0)
                if queue:
                    merged.append(queue.pop(0))
                turn += 1
            self.ideas = merged
            self.list_start = len(self.asked_from)
            self.note(f"properties for the open purpose {self.purposes[i]['you']!r}: {self.ideas}")

    def spoken(self, name: str) -> bool:
        mine = [t for t in tokens(str(name).replace('_', ' ')) if len(t) >= 4]
        named = [w.property for w in self.all_wishes()] + self.props_asked
        theirs = [t for n in named for t in tokens(str(n or '').replace('_', ' ')) if len(t) >= 4]
        return any((a[:4] == b[:4] and abs(len(a) - len(b)) <= 2 for a in mine for b in theirs))

    def next_property(self) -> str | None:
        wished = [self.label_words(w.property) for w in self.all_wishes() if w.kind != 'brand']
        asked_words = [self.label_words(l) for l in self.props_asked]
        asked_plain = [set(tokens(l.replace('_', ' '))) for l in self.props_asked]
        taken: set[str] = set()
        for w in self.all_wishes():
            taken |= self.aligned.get(w.id, set())
        idea = None
        self.property_ideas()
        self.purpose_ideas()
        for name in self.property_ideas():
            words = self.label_words(name) or set(tokens(name))
            if words and (not any((words & other for other in wished + asked_words))) and (set(tokens(name)) not in asked_plain) and (not self.spoken(name)) and (not self.names_priority(name)):
                idea = name
                break
        from_list = sum((1 for source in self.asked_from[self.list_start:] if source == 'list'))
        if idea is not None and (from_list < LIST_FIRST or self.asked_from[-1:] == ['shelf']):
            self.asked_from.append('list')
            return idea
        pool = [l for l in self.listings.values() if self.shelf_of(l) and l['facts']]
        if len(pool) < 4:
            if idea is not None:
                self.asked_from.append('list')
            return idea
        fit_pids = {r['pid'] for r in self.fitting(self.solve(), LOW)}
        best, best_score = (None, 0.0)
        stats: dict[str, dict] = {}
        for pid, listing in self.listings.items():
            if not self.shelf_of(listing) or not listing['facts']:
                continue
            text = tokens(' '.join([listing['title']] + [str(v.get('options') or '') for v in listing['variants'].values()]))
            for label, value in listing['facts'].items():
                if len(label) > 60:
                    continue
                st = stats.setdefault(label, {'n': 0, 'values': {}, 'titled': 0, 'fit': set()})
                parts = [tuple(tokens(part)) for part in entries(value)]
                parts = [part for part in parts if part]
                if not parts:
                    continue
                st['n'] += 1
                st['values'][parts[0]] = st['values'].get(parts[0], 0) + 1
                st['titled'] += any((has_phrase(text, list(part)) for part in parts))
                if pid in fit_pids:
                    st['fit'].add(parts[0])
        makers = {}
        for listing in self.listings.values():
            if not self.shelf_of(listing) or not listing['facts']:
                continue
            maker = tokens(listing.get('brand') or '')
            for label, value in listing['facts'].items():
                first = (entries(value) or [''])[0]
                n_same = makers.setdefault(label, [0, 0])
                n_same[0] += 1
                n_same[1] += bool(maker) and tokens(first) == maker
        for label, st in stats.items():
            words = self.label_words(label)
            seen_n, same_n = makers.get(label, (0, 0))
            if seen_n and same_n * 2 >= seen_n:
                continue
            if label in taken or label in self.props_asked or (not words):
                continue
            if any((words & other for other in wished)) or self.names_priority(label):
                continue
            if any((words & other for other in asked_words)) or set(tokens(label.replace('_', ' '))) in asked_plain:
                continue
            n, distinct = (st['n'], len(st['values']))
            if n < 3 or distinct < 2:
                continue
            if n >= 6 and distinct > 0.75 * n:
                continue
            if sum((len(v) * c for v, c in st['values'].items())) > 3.0 * n:
                continue
            if self.spoken(label):
                continue
            share = n / len(pool)
            titled = st['titled'] / n
            score = share * (0.05 + titled) * (1.5 if len(st['fit']) >= 2 else 1.0)
            if score >= ASK_FLOOR and score > best_score:
                best, best_score = (label, score)
        if best is not None:
            self.note(f'property to ask: {best} score={best_score:.2f} of {len(stats)} labels, {len(fit_pids)} fitting')
            self.asked_from.append('shelf')
            return best
        if idea is not None:
            self.asked_from.append('list')
        return idea

    def names_priority(self, name: str) -> bool:
        if not self.priority:
            return False
        head = self.head(self.priority.get('property'))
        return bool(head) and head.rstrip('s') in {t.rstrip('s') for t in tokens(str(name).replace('_', ' '))}

    @staticmethod
    def say(wish: Wish) -> str:
        prop = ' '.join((t for t in re.split('[_\\s]+', wish.property) if t))
        if wish.kind == 'brand':
            return f'the brand {wish.value}'
        return f'{wish.value} as the {prop}' if prop else wish.value

    def message_call(self, text: str | None) -> list[tuple[str, dict]]:
        if not text:
            return []
        self.last_message = text
        self.questions += 1
        self.note(f'ask {text}')
        return [('message', {'content': text})]

    def reply_call(self) -> list[tuple[str, dict]]:
        if not self.owed:
            return []
        parts = list(dict.fromkeys((p for said in self.owed for p in said['parts'])))
        confirm = any((said['confirm'] for said in self.owed))
        self.owed = []
        if any((not p.startswith('your limit') for p in parts)):
            parts = [p for p in parts if not p.startswith('your limit')]
        text = ('Yes, confirmed: ' if confirm else 'Understood: ') + '. '.join((p[0].upper() + p[1:] if i else p for i, p in enumerate(parts))) + '.'
        self.last_message = text
        self.note(f'reply {text}')
        return [('message', {'content': text})]

    def explore(self) -> None:
        self.act(self.message_call(self.question()) + self.probe_calls() + self.search_calls(self.opening_queries(), 12))
        self.act(self.message_call(self.question()) + self.compare_calls(self.unread(75)))
        self.find_shelf()
        self.align()
        told = [w for w in self.wishes if w.origin == 'told']
        self.act(self.message_call(self.question()) + self.sweep_calls() + self.search_calls(self.value_queries(told or self.all_wishes()), 7))
        self.act(self.message_call(self.question()) + self.compare_calls(self.unread(75)))
        guard = 0
        while guard < (4 + PROPERTY_QUESTIONS if self.needs_open() > 0 else 4) and self.turns_left() > TURN_RESERVE + 2 and (self.seconds_left() > 60):
            guard += 1
            self.align()
            ask = self.question()
            fresh = [w for w in self.wishes if w.origin in ('told', 'chosen')]
            reads = self.compare_calls(self.unread(40)) or self.search_calls(self.value_queries(fresh), 6)
            if not ask and (not reads):
                break
            self.act(self.message_call(ask) + reads)
            if not ask and (not self.pending_questions()):
                break

    def pending_questions(self) -> bool:
        return self.pair is not None and self.pair_asked < 2 or any((not p['asked'] for p in self.purposes))

    def stated_now(self) -> list[Wish]:
        return [w for w in self.wishes if w.id in self.stated_ids]

    def siblings(self, rows: list[dict]) -> None:
        queries = []
        for r in rows[:4]:
            title = self.listings[r['pid']]['title']
            queries.append(' '.join(title.split()[:8]))
        calls = self.search_calls(queries, 6)
        if calls:
            self.act(calls)
            more = self.unread(30)
            if more:
                self.act(self.compare_calls(more))

    def partials(self, rows: list[dict], variants: bool=False) -> list[dict]:
        cap = (self.budget or 0) * STRETCH if self.budget else None
        named = lambda r: bool((self.listings[r['pid']]['variants'].get(r['sku']) or {}).get('options'))
        pool = [r for r in rows if (cap is None or r['price'] <= cap) and (min(r['marks'] or [0]) > CONTRA or (variants and named(r)))]
        if not pool:
            return []
        met = lambda r: sum((m >= LOW for m in r['marks']))
        best = max((met(r) for r in pool))
        return sorted([r for r in pool if met(r) == best and best > 0], key=lambda r: (not r['within'], r['price']))

    def bottleneck(self, rows: list[dict], wishes: list[Wish] | None=None) -> list[Wish]:
        wishes = self.wishes if wishes is None else wishes
        count = {w.id: sum((1 for r in rows if r['within'] and i < len(r['marks']) and (r['marks'][i] >= TITLE))) for i, w in enumerate(wishes)}
        return sorted(wishes, key=lambda w: count.get(w.id, 0))

    def complete(self, waves: int, wishes: list[Wish] | None=None) -> list[dict]:
        rows = self.solve(wishes)
        thin = 0
        for wave in range(waves):
            firm_sources = {r['sku'].split('-')[0] for r in self.fitting(rows, FIRM)}
            after = self.carts > 0
            need = (6 if self.priority else 3) if after else 2 if self.shopper_changed and self.priority else 0
            enough = 10 if after and self.priority else 5 if after else 3
            settled = len(firm_sources) >= enough and wave >= need or (wave >= max(6 if after else 2, need) and firm_sources) or (wave >= (7 if after else 3) and self.fitting(rows, TITLE))
            if settled or self.turns_left() <= TURN_RESERVE + 2 or self.seconds_left() < 70:
                break
            head = self.head_word()
            walk = [f'{a} {head}' for a in self.anchors(20)[1:]]
            scarce = self.bottleneck(rows, wishes)
            missing = scarce[0].value if scarce else ''
            seen, queries = (set(), [])
            for r in self.partials(rows, variants=True)[:6]:
                source = r['sku'].split('-')[0]
                if source not in seen:
                    seen.add(source)
                    title = ' '.join(self.listings[r['pid']]['title'].split()[:14])
                    queries.append(f'{missing} {title}'.strip())
            families = []
            if scarce:
                value = scarce[0].value
                for pid in list(self.shelf_rows)[:40]:
                    words = str(self.rows.get(pid, {}).get('title') or '').split()[:5]
                    q = f"{value} {' '.join(words)}"
                    if words and q not in families:
                        families.append(q)
            new = lambda qs: [q for q in qs if ' '.join(q.split())[:120].casefold() not in self.queried]
            calls = self.search_calls(new(self.twin_queries)[:2] + new(self.value_queries(scarce))[:8] + new(queries)[:4] + new(families)[:4] + walk, 12)
            calls += self.ceiling_calls(1) + self.brand_calls(2)
            if not calls:
                break
            known = len(self.rows)
            self.act(self.reply_call() + calls)
            new_rows = len(self.rows) - known
            more = self.unread(75)
            if more:
                self.act(self.compare_calls(more))
            self.align()
            rows = self.solve(wishes)
            thin = thin + 1 if new_rows < 3 else 0
            if thin >= 4:
                self.note(f'coverage saturated: two waves found almost nothing new (last {new_rows})')
                break
        return rows

    def worth_asking_purpose(self) -> bool:
        shelf_read = sum((1 for l in self.listings.values() if self.shelf_of(l)))
        if shelf_read < 25:
            return False
        rows = self.solve()
        if self.fitting(rows, LOW):
            return True
        near = [r for r in self.near_misses(rows) if self.tier(r['marks']) >= TITLE]
        if not near:
            return True
        self.note('purpose question held: nothing within the limit fits, a need cannot change the ending')
        return False

    def ask_held(self) -> None:
        if any((not p['asked'] for p in self.purposes)) and self.turns_left() > TURN_RESERVE + 2:
            self.pair_asked = max(self.pair_asked, 1)
            held = self.pair
            ask = self.question() if self.fitting(self.solve(), LOW) else None
            self.pair = held
            if ask:
                self.act(self.message_call(ask) + self.compare_calls(self.unread(40)))
                self.align()

    def first_cart(self, pick: dict) -> None:
        self.note(f"cart {pick['key']} marks={pick['marks']} figure={pick['figure']} price={pick['price']}")
        before = len(self.cart)
        self.act(self.reply_call() + [('add_to_cart', {'product_id': pick['pid'], 'sku': pick['sku']}), ('inspect_stock', {'product_id': pick['pid'], 'sku': pick['sku']})])
        if len(self.cart) == before:
            self.dead.add(pick['key'])
            return
        self.first_key = pick['key']
        if not self.top_titles_looked:
            self.top_titles_looked = True
            others = [r for r in self.fitting(self.solve(), floor=LOW) if r['pid'] != pick['pid']][:2]
            if others:
                self.siblings(others)
        plain = re.sub('[\\[(【][^\\])】]*[\\])】]', ' ', self.listings[pick['pid']]['title']).split()
        values = [self.value_forms(w)[1] if w.kind == 'amount' and len(self.value_forms(w)) > 1 else w.value for w in self.wishes]
        self.twin_queries = [' '.join(plain[:8] + values), ' '.join(plain[:12])]
        self.refresh_candidates()
        if self.owed:
            changed = [w for w in self.wishes if w.origin == 'changed']
            title = self.listings[pick['pid']]['title'].split()
            siblings = [' '.join(title[:8])] + [f"{w.value} {' '.join(title[:5])}" for w in changed[:2]]
            values = [self.value_forms(w)[1] if w.kind == 'amount' and len(self.value_forms(w)) > 1 else w.value for w in self.wishes]
            siblings.append(' '.join(plain[:8] + values))
            fresh = self.search_calls(siblings + self.value_queries(changed), 12)
            self.act(self.reply_call() + fresh)
            more = self.unread(75)
            if more:
                self.act(self.compare_calls(more))
            self.align()

    def refresh_candidates(self) -> None:
        rows = self.candidates(self.wishes, None)
        fit = [r for r in rows if self.tier(r['marks']) >= LOW]
        fit.sort(key=lambda r: (-self.tier(r['marks']), r['price']))
        pids = list(dict.fromkeys((r['pid'] for r in fit)))[:40]
        pids += [p for p in self.unread(30, any_price=True) if p not in pids]
        if pids and self.turns_left() > TURN_RESERVE + 2:
            self.note(f'market refresh: {len(pids)} listings read again')
            self.act(self.reply_call() + self.compare_calls(pids))

    def settle(self) -> None:
        self.align()
        if self.pair_asked and any((not p['asked'] for p in self.purposes)):
            self.complete(2)
            self.ask_held()
        if self.carts == 0 and (not self.done):
            rows = self.solve()
            n = len(self.stated_now())
            first = self.first_pick()
            if first is not None and first['within'] and (self.tier(first['marks'][:n]) >= FIRM) and (len(first['marks']) <= n or self.tier(first['marks'][n:]) >= SYN):
                self.first_cart(first)
                rows = self.complete(14)
                if self.try_abstain(rows):
                    return
                self.cart_loop()
                return
            if not self.fitting(rows, floor=LOW):
                rows = self.complete(3)
            first = self.first_pick()
            if not self.fitting(rows, floor=LOW) and (not (first is not None and first['within'] and (self.tier(first['marks'][:n]) >= FIRM))):
                if self.try_abstain(rows):
                    return
            if first is None or self.tier(first['marks'][:n]) < FIRM:
                self.complete(1 if first is not None else 4, wishes=self.stated_now())
                first = self.first_pick()
            if first is not None and (len(first['marks']) > n and self.tier(first['marks'][n:]) < SYN or self.tier(first['marks'][:n]) < FIRM) and (self.seconds_left() > 90) and (self.turn_no < 12):
                self.complete(2)
                first = self.first_pick() or first
            if first is not None:
                self.first_cart(first)
        rows = self.complete(14)
        if self.try_abstain(rows):
            return
        self.cart_loop()

    def try_abstain(self, rows: list[dict]) -> bool:
        fit = self.fitting(rows, floor=LOW)
        shelf_read = sum((1 for l in self.listings.values() if self.shelf_of(l)))
        if self.shopper_changed or shelf_read < 25 or self.done:
            return False
        if fit and max((self.tier(r['marks']) for r in fit)) < TITLE:
            firm_over = [r for r in self.near_misses(rows) if self.tier(r['marks']) >= FIRM]
            if firm_over:
                self.note(f"abstain: within the limit only weak rows (best {max((self.tier(r['marks']) for r in fit))}), over it a firm one")
                self.abstain(firm_over[0])
                return True
            return False
        if fit:
            return False
        near = [r for r in self.near_misses(rows) if self.tier(r['marks']) >= TITLE]
        if near:
            self.siblings(near)
            rows = self.solve()
            if self.fitting(rows, floor=LOW):
                return False
            near = [r for r in self.near_misses(rows) if self.tier(r['marks']) >= TITLE]
        if near:
            self.abstain(near[0])
            return True
        met = lambda r: sum((m >= LOW for m in r['marks']))
        over = sorted([r for r in rows if not r['within'] and r['price'] <= self.budget * STRETCH + 1e-09], key=lambda r: (-met(r), r['price'])) if self.budget else []
        inside = max((met(r) for r in rows if r['within']), default=0)
        if over and met(over[0]) > inside:
            self.abstain(over[0])
            return True
        return False

    def near_misses(self, rows: list[dict]) -> list[dict]:
        if not self.budget:
            return []
        over = [r for r in rows if not r['within'] and r['price'] <= self.budget * STRETCH + 1e-09 and (self.tier(r['marks']) >= LOW)]
        return sorted(over, key=lambda r: (-self.tier(r['marks']), r['price']))

    def first_pick(self) -> dict | None:
        stated = self.stated_now()
        others = [w for w in self.wishes if w.id not in self.stated_ids]
        rows = self.solve(stated + others)
        n = len(stated)
        pool = [r for r in rows if r['within'] and self.tier(r['marks'][:n]) >= TITLE]
        if not pool:
            pool = [r for r in rows if r['within'] and self.tier(r['marks'][:n]) >= LOW]
        sign = self.priority['sign'] if self.priority else 1.0
        support = {r['key']: min([self.sources(w, r['pid'], r['sku']) for w in stated] or [3]) for r in pool}
        pool.sort(key=lambda r: (-self.tier(r['marks'][:n]), -min(self.tier(r['marks'][n:]), TITLE), -support[r['key']], -self.names_item(r['pid']), -sum((m == FIRM for m in r['marks'][:n])), -self.tier(r['marks'][n:]), 0 if r['figure'] is not None else 1, -(r['figure'] or 0.0) * sign, r['price']))
        if not pool:
            return None
        best = self.final_pick()
        if best is not None and best['within'] and (self.tier(best['marks'][:n]) == FIRM):
            return best
        return pool[0]

    def sources(self, wish: Wish, pid: str, sku: str) -> int:
        listing = self.listings[pid]
        options = (listing['variants'].get(sku) or {}).get('options') or ''
        if wish.kind == 'brand':
            brand = tokens(listing['brand'])
            return int(any((has_phrase(brand, p) for p in wish.phrases))) + int(any((has_phrase(tokens(listing['title']), p) for p in wish.phrases)))
        count = int(bool(options) and self.verdict(wish, options, False) == 'yes')
        for label in self.aligned.get(wish.id, ()):
            text = listing['facts'].get(label)
            if text and len(entries(text)) == 1 and (self.verdict(wish, text, False) == 'yes'):
                count += 1
                break
        return count + int(wish.code_verdict(listing['title']) == 'yes')

    def colour_beside(self, wish: Wish, options: str) -> bool:
        if self.colour_words is None or len(self.listings) != self.colour_seen:
            words = set()
            for listing in self.listings.values():
                for label, text in listing['facts'].items():
                    if self.own_label(wish, label):
                        for part in entries(str(text)):
                            t = tokens(part)
                            if len(t) == 1 and (not t[0][0].isdigit()):
                                words.add(t[0])
            self.colour_words, self.colour_seen = (words, len(self.listings))
        want = set((t for p in wish.phrases for t in p))
        seq = tokens(options)
        for i, t in enumerate(seq[:-1]):
            if t in want and seq[i + 1] not in want and (seq[i + 1] in self.colour_words):
                return True
        return False

    def near_form(self, wish: Wish, pid: str) -> int:
        want = {t[:5] for p in wish.phrases[:1] for t in p if len(t) >= 5}
        if not want:
            return 0
        for label, text in self.listings[pid]['facts'].items():
            if label in self.aligned.get(wish.id, ()) and want & {t[:5] for t in tokens(text) if len(t) >= 5}:
                return 1
        return 0

    def own_label_states(self, wish: Wish, pid: str) -> int:
        own = self.label_words(wish.property)
        for label, text in self.listings[pid]['facts'].items():
            if own & set(tokens(label.replace('_', ' '))) and wish.code_verdict(text) == 'yes':
                return 1
        return 0

    def doubt_pass(self, rows: list[dict]) -> None:
        pool = [r for r in rows if r['within'] and CONTRA not in r['marks'] and (sum((m == ABSENT for m in r['marks'])) == 1) and (self.tier([m for m in r['marks'] if m != ABSENT]) >= LOW)]
        pool.sort(key=lambda r: (-sum((max(m, 0) for m in r['marks'])), r['price']))
        groups: dict[str, tuple[Wish, list[dict]]] = {}
        for r in pool[:12]:
            w = self.wishes[r['marks'].index(ABSENT)]
            if (w.id, r['key']) not in self.row_asked:
                groups.setdefault(w.id, (w, []))[1].append(r)
        if not groups:
            return

        def page(r: dict) -> str:
            listing = self.listings[r['pid']]
            opts = (listing['variants'].get(r['sku']) or {}).get('options') or ''
            facts = '; '.join((f'{k}: {v}' for k, v in listing['facts'].items()))
            return short(f"{listing['title']} | variant: {opts} | {facts}", 600)

        def run(job):
            w, rs = job
            got = self.parse(DOUBT_FORM, {'requirement': {'property': w.property, 'value': w.value}, 'listings': [{'n': i, 'text': page(r)} for i, r in enumerate(rs)]}, max_tokens=40 + 16 * len(rs)) or {}
            return (w, rs, got)
        with ThreadPoolExecutor(max_workers=3) as ex:
            for w, rs, got in ex.map(run, list(groups.values())):
                yes = {int(v.get('n', -1)) for v in got.get('verdicts') or [] if v.get('verdict') == 'yes'}
                for i, r in enumerate(rs):
                    self.row_asked.add((w.id, r['key']))
                    if i in yes:
                        self.row_yes.add((w.id, r['key']))
                self.note(f'doubt pass {w.value}: {len(yes)} of {len(rs)} rows state it')

    def final_pick(self) -> dict | None:
        rows = self.solve()
        if not self.fitting(rows, TITLE) and self.seconds_left() > 40:
            self.doubt_pass(rows)
            rows = self.candidates(self.wishes, self.budget)
        for floor in (TITLE, LOW):
            fit = self.fitting(rows, floor)
            if fit:
                top = fit[0]
                held = next((r for r in fit if r['key'] == self.first_key), None)
                if held is not None and held is not top and (held['figure'] == top['figure']) and all((h >= t for h, t in zip(held['marks'], top['marks']))):
                    return held
                return top
        return None

    def best_effort(self) -> dict | None:
        rows = [r for r in self.candidates(self.wishes, self.budget) if r['within']]
        sign = self.priority['sign'] if self.priority else 1.0
        stated = [i for i, w in enumerate(self.wishes) if w.id in self.stated_ids or w.origin in ('told', 'changed', 'chosen')]
        rows.sort(key=lambda r: (sum((r['marks'][i] <= CONTRA for i in stated if i < len(r['marks']))), -sum((m >= LOW for m in r['marks'])), -sum((max(m, 0) for m in r['marks'])), 0 if r['figure'] is not None else 1, -(r['figure'] or 0.0) * sign, r['price']))
        return rows[0] if rows else None

    def cart_loop(self) -> None:
        for _ in range(5):
            if self.done:
                return
            pick = self.first_pick() if self.carts == 0 else self.final_pick()
            if pick is not None and self.carts > 0 and self.priority and (not self.sibling_looked) and (self.turns_left() > TURN_RESERVE + 1) and (self.seconds_left() > 60):
                self.sibling_looked = True
                self.siblings([pick])
                pick = self.final_pick() or pick
            if pick is None and self.carts > 0 and (self.turns_left() > TURN_RESERVE):
                self.complete(4)
                pick = self.final_pick()
            if pick is None:
                pick = self.best_effort()
                if pick is None:
                    return self.last_resort()
                self.note(f"best effort {pick['key']} marks={pick['marks']}")
            key = pick['key']
            if self.carts > 0 and key in self.cart and (key not in self.dead):
                return self.order(pick)
            self.note(f"cart {key} marks={pick['marks']} figure={pick['figure']} price={pick['price']}")
            before = len(self.cart)
            self.act(self.reply_call() + [('add_to_cart', {'product_id': pick['pid'], 'sku': pick['sku']}), ('inspect_stock', {'product_id': pick['pid'], 'sku': pick['sku']})])
            if len(self.cart) == before:
                self.dead.add(key)
                continue
            if self.turns_left() <= 2 or self.seconds_left() < 25:
                return self.order(pick) if key not in self.dead else self.last_resort()
            if self.owed:
                fresh = self.search_calls(self.value_queries([w for w in self.wishes if w.origin == 'changed']), 12)
                self.act(self.reply_call() + fresh)
                more = self.unread(60)
                if more:
                    self.act(self.compare_calls(more))
                self.align()
        self.last_resort()

    def claims(self, pid: str, sku: str) -> list[dict]:
        price = self.price(pid, sku)
        claims: list[dict] = [{'field': 'price', 'value': price}, {'field': 'in_stock', 'value': True}]
        listing = self.listings[pid]
        pool = [w for w in self.wishes if self.reason_fields and any((set(tokens(f)) & set(tokens(w.property)) for f in self.reason_fields))] or self.wishes

        def holds(wish: Wish, text: str) -> bool:
            return wish.code_verdict(text) == 'yes' or self.verdicts.get((wish.id, text)) == 'yes'
        for wish in pool:
            for label in sorted(self.aligned.get(wish.id, ())):
                text = listing['facts'].get(label)
                if not text:
                    continue
                part = next((p for p in [text, *entries(text)] if holds(wish, p)), None)
                if part:
                    claims.append({'field': label, 'value': part})
                    return claims
        options = (listing['variants'].get(sku) or {}).get('options') or ''
        if options and any((holds(w, options) for w in pool)):
            claims.append({'field': 'options', 'value': options})
        return claims

    def order(self, pick: dict) -> None:
        args: dict[str, Any] = {'product_id': pick['pid'], 'sku': pick['sku']}
        if self.reason_asked:
            args['justification'] = {'claims': self.claims(pick['pid'], pick['sku'])}
        self.note(f"order {pick['key']} marks={pick['marks']} figure={pick['figure']}")
        try:
            self.note('wishes ' + ' | '.join((f'{w.property}={w.value}:{w.origin}' for w in self.wishes)))
            ranked = self.rank([r for r in self.candidates(self.wishes, self.budget) if r['within']])[:6]
            self.note('ranked ' + ' ; '.join((f"{r['key']} {r['marks']} f={r['figure']}" for r in ranked)))
        except Exception:
            pass
        self.act(self.reply_call() + [('place_test_order', args)])

    def abstain(self, row: dict) -> None:
        args = {'abstain': True, 'product_id': row['pid'], 'sku': row['sku'], 'justification': {'claims': self.claims(row['pid'], row['sku'])[:3 if self.reason_asked else 2]}}
        self.note(f"abstain naming {row['key']} price={row['price']} limit={self.budget}")
        self.act([('place_test_order', args)])

    def last_resort(self) -> None:
        if self.done:
            return

        def affordable(key: str) -> bool:
            pid, _, sku = key.partition('::')
            price = self.price(pid, sku) if pid in self.listings else None
            return price is None or not self.budget or price <= self.budget + 1e-09
        alive = [k for k in self.cart if k not in self.dead and affordable(k)]
        r = self.best_effort() if self.listings else None
        if alive and (r is None or r['key'] in alive or self.turns_left() < 3):
            pid, sku = alive[-1].split('::', 1)
            self.note('last resort: order the cart')
            self.act([('place_test_order', {'product_id': pid, 'sku': sku})])
            return
        if r is None:
            alive = [k for k in self.cart if k not in self.dead]
            if alive:
                pid, sku = alive[-1].split('::', 1)
                self.note('last resort: order the cart')
                self.act([('place_test_order', {'product_id': pid, 'sku': sku})])
                return
            seen = [row for row in self.rows.values() if row.get('in_stock') is not False and row.get('sku') and (not self.budget or float(row.get('price') or 0) <= self.budget)]
            if not seen:
                self.note('last resort: nothing to order')
                return
            row = min(seen, key=lambda x: float(x.get('price') or 0))
            r = {'key': f"{row['product_id']}::{row['sku']}", 'pid': str(row['product_id']), 'sku': str(row['sku'])}
        self.note(f"last resort: cart and order {r['key']}")
        self.act([('add_to_cart', {'product_id': r['pid'], 'sku': r['sku']})])
        if not self.done:
            self.act([('place_test_order', {'product_id': r['pid'], 'sku': r['sku']})])

    def run(self) -> list[dict]:
        try:
            self.read_request()
            self.explore()
            self.settle()
            if not self.done:
                self.last_resort()
        except Exception as exc:
            self.note(f'error {type(exc).__name__}: {short(str(exc), 160)}')
            try:
                self.last_resort()
            except Exception as again:
                self.note(f'last resort failed {type(again).__name__}')
        self.note(f'end turns={self.turn_no} questions={self.questions} spent={self.spent:.4f} seconds={time.monotonic() - self.start:.0f} read={len(self.listings)}')
        return [{'role': 'assistant', 'content': 'note: ' + n} for n in self.notes]

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    return Episode(problem_data).run()
__all__ = ['agent_main']
