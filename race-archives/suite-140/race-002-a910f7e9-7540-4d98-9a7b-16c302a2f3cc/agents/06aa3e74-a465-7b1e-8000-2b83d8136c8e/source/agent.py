from __future__ import annotations
import json
import re
import time
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
_MODELS = {'chutes': ('deepseek-ai/DeepSeek-V3.2-TEE', 'Qwen/Qwen3-32B-TEE'), 'openrouter': ('deepseek/deepseek-v3.2', 'qwen/qwen3-32b')}
_PROXY = ProxyClient(timeout=60, max_retries=2, retry_delay=1.0)
_PROBLEM_BUDGET_S = 270.0
_OBS_CHARS = 7000
_MAX_REPROMPTS = 2
_EPS = 1e-09
_MODEL_FAIL_LIMIT = 2
_BARREN_LIMIT = 3
_ENDGAME_TURNS = 4
_IDLE_LIMIT = 4
_NUM = '(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?'
_BUDGET_RE = re.compile('(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|within|budget (?:is|of)|only go up to|can spend up to|at most)\\s*' + _NUM, re.I)
_CUR_SYM = '[^\\w\\s\\d.,:;%+*/()\\[\\]<>=!?&|~^@#-]'
_CURRENCY_ADJ_RE = re.compile('(?:(' + _CUR_SYM + ')\\s*\\d)|(?:\\d[\\d,]*(?:\\.\\d+)?\\s*(' + _CUR_SYM + '))|(?:\\b([A-Z]{3})\\b\\s*\\d[\\d,]*(?:\\.\\d+)?)|(?:\\d[\\d,]*(?:\\.\\d+)?\\s+\\b([A-Z]{3})\\b)')
_KEY_RE = re.compile('(\\d{6,})::([A-Za-z0-9._\\-]+)')
_QUOTED_RE = re.compile('\\"([^\\"]{2,80})\\"')
_PHRASE_RE = re.compile('matches (?:the phrase )?\\"?([A-Za-z0-9][^\\",]{1,80}?)\\"?\\s*,', re.I)
_SURFACE_RE = re.compile('surface at least (\\d+) distinct', re.I)
_PREF_ITEM_RE = re.compile('prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\\-]*\\d[A-Za-z0-9\\-]*)\\b', re.I)
_PREF_BRAND_RE = re.compile("\\bprefer(?:ably)?\\s+(?:the\\s+)?([A-Za-z][\\w&.'\\- ]{1,38}?)\\s*(?:,|\\.|;|\\bbut\\b|\\bwhen\\b|\\bif\\b|$)")
_CAT_PHRASE_RE = re.compile("in the ([A-Za-z][\\w&'\\- ]{1,48}?) category", re.I)
_SPEC_WORD_RE = re.compile('(?<![\\w.])\\d+(?:\\.\\d+)?\\s*[-–]?\\s*[A-Za-z]{0,6}(?![\\w])')
_REASON_RE = re.compile('\\b(?:why|reason|explain|justif)', re.I)
_QUESTION_RE = re.compile('\\?|\\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)|tell me (?:which|whether|if)|confirm whether)\\b', re.I)
_MEASURE_RE = re.compile('(?<![\\w.])(?<!\\w-)(\\d+(?:\\.\\d+)?)\\s*[-–]?\\s*([A-Za-z]{1,6}|\'{1,2}|[\\"“”])(?![\\w])')
_FUNCTION_WORDS = frozenset('a an the this that these those my your our their its it i me we you they\n    is are was were be been being do does did doing have has had having\n    can could will would shall should may might must\n    get gets getting got find finds finding need needs needing want wants wanting\n    look looking looks buy buys buying order orders ordering purchase show shows\n    pick picks picking choose choosing prefer prefers recommend recommends\n    please just really some any something anything one another other same\n    and or but for with without from into onto than then there here about\n    under over below above within at most least around near upto up to of in on'.split())

def _unit(raw: str) -> str:

    def hio6__body_15(raw, *, hio6_z=0):
        return (raw or '').strip().lower()
    return hio6__body_15(raw)

def _named_unit(text: str) -> bool:

    def hio6__body_16(text, *, hio6_z=0):
        return bool(text) and text[:1].isalpha()
    return hio6__body_16(text)

def _unit_eq(a: str, b: str) -> bool:

    def hio6__body_17(a, b, *, hio6_z=0):
        try:
            x, y = (_unit(a), _unit(b))
            if not x or not y:
                return False
            if x == y:
                return True
            if not (_named_unit(x) and _named_unit(y)):
                return True
            return min(len(x), len(y)) >= 2 and (x.startswith(y) or y.startswith(x))
        except Exception:
            raise
    return hio6__body_17(a, b)

def _measures(text: str) -> list:

    def hio6__body_18(text, *, hio6_z=0):
        out = []
        for m in _MEASURE_RE.finditer(text or ''):
            unit = _unit(m.group(2))
            if unit in _FUNCTION_WORDS:
                continue
            try:
                out.append((float(m.group(1)), unit, m.start()))
            except ValueError:
                continue
        return out
    return hio6__body_18(text)

def _fmt(value: float, unit: str) -> str:

    def hio6__body_19(value, unit, *, hio6_z=0):
        return '%g%s' % (value, _unit(unit))
    return hio6__body_19(value, unit)

def _split_value(want: str) -> tuple:

    def hio6__body_20(want, *, hio6_z=0):
        m = re.fullmatch('(\\d+(?:\\.\\d+)?)([a-z]*)', (want or '').strip().lower())
        if m:
            return (float(m.group(1)), m.group(2))
        return (None, (want or '').strip().lower())
    return hio6__body_20(want)

def _tail(word: str) -> str:
    n = len(word or '')

    def hio6__ie_3():
        if n >= 3:
            return '[a-z]?'
        else:
            return ''
    if n >= 5:
        return '[a-z]{0,3}'
    else:
        return hio6__ie_3()

def _says(text: str, word: str) -> bool:

    def hio6__body_21(text, word, *, hio6_z=0):
        if not word:
            return False
        low = word.lower()
        return bool(re.search('(?<![a-z])' + re.escape(low) + _tail(low) + '(?![a-z])', (text or '').lower()))
    return hio6__body_21(text, word)

def _covers(text: str, word: str) -> bool:

    def hio6__body_22(text, word, *, hio6_z=0):
        if not word or len(word) < 4:
            return False
        flat = re.sub('[^a-z0-9]', '', (text or '').lower())
        return word.lower()[:5] in flat
    return hio6__body_22(text, word)

def _marks(text: str, word: str) -> list:

    def hio6__body_23(text, word, *, hio6_z=0):
        if not word:
            return []
        low = word.lower()
        return [m.start() for m in re.finditer('(?<![a-z])' + re.escape(low) + _tail(low) + '(?![a-z])', (text or '').lower())]
    return hio6__body_23(text, word)

def _gap(at: int, mark: int) -> float:
    if mark >= at:
        return float(mark - at)
    else:
        return (at - mark) * 3.0

def _values_for(field: str, want: str, text: str) -> set:

    def hio6__body_24(field, want, text, *, hio6_z=0):
        number, unit = _split_value(want)
        text = text or ''
        if number is None:
            if _says(text, unit):
                return {unit}
            else:
                return set()
        hits = []
        for v, u, at in _measures(text):
            if _unit_eq(u, unit):
                hits.append((v, at))
        if not hits and len(unit) == 1:
            hits = []
            for m in re.finditer('(?<![\\w.])\\d+(?:\\.\\d+)?(?![\\w.])', text):
                if float(m.group()) == number:
                    hits.append((float(m.group()), m.start()))
        if not hits:
            return set()
        if not field or _unit_eq(field, unit):
            marks = []
        else:
            marks = _marks(text, field)
        if marks:
            scored = []
            for v, at in hits:
                scored.append((min((_gap(at, p) for p in marks)), v))
            closest = min((s for s, _v in scored))
            if closest > 26.0:
                return set()
            return {_fmt(v, unit) for s, v in scored if s == closest}
        return {_fmt(v, unit) for v, _at in hits}
    return hio6__body_24(field, want, text)

def _mentions(field: str, want: str, text: str) -> bool:

    def hio6__body_25(field, want, text, *, hio6_z=0):
        number, unit = _split_value(want)
        if number is None:
            return _says(text, unit)
        return bool([1 for _v, u, _at in _measures(text or '') if _unit_eq(u, unit)]) or _says(text, field)
    return hio6__body_25(field, want, text)

def _one(values: set) -> str | None:
    if len(values) == 1:
        return next(iter(values))
    else:
        return None

def _tokens(text: str) -> set:

    def hio6__body_26(text, *, hio6_z=0):
        return set(re.findall('[a-z0-9]+', (text or '').lower()))
    return hio6__body_26(text)
_ABOUT_RE = re.compile('\\b(?:for|compatible\\s+with|fits|suitable\\s+for)\\b', re.I)
_NEGATED_RE = re.compile('\\bnot\\s+(?:an?\\s+)?[\\w\\s]{0,24}?(?=,|\\.|$)', re.I)
_BUNDLE_RE = re.compile('\\b(?:with\\s+(?:free|bundled)|bundled\\s+with|free|includes?)\\s+', re.I)
_KIND_KEY_RE = re.compile('type|categ|kind|class', re.I)

def _identity_text(text: str, title: bool=False) -> str:

    def hio6__body_27(text, title, *, hio6_z=0):
        text = _NEGATED_RE.sub('', text or '')
        if title:
            return _BUNDLE_RE.split(text, maxsplit=1)[0]
        else:
            return text
    return hio6__body_27(text, title)

def _other_kind(noun: str, facts: dict) -> bool:

    def hio6__body_28(noun, facts, *, hio6_z=0):
        try:
            tail = (noun or '')[-5:]
            if len(noun or '') < 7 or len(tail) < 5:
                return False
            for value in (facts or {}).values():
                for word in re.findall('[a-z]+', str(value or '').lower()):
                    if word != noun and len(word) > len(tail) + 1 and word.endswith(tail):
                        return True
            return False
        except Exception:
            raise
    return hio6__body_28(noun, facts)

def _identity_of(noun: str, facts: dict, title: str, options: str) -> str:

    def hio6__body_29(noun, facts, title, options, *, hio6_z=0):
        if not noun:
            return 'unclear'
        if isinstance(facts, dict):
            facts = facts
        else:
            facts = {}
        head = _identity_text(str(title or ''), title=True)
        kind = ' '.join((str(v) for k, v in facts.items() if _KIND_KEY_RE.search(str(k))))
        kind = _identity_text(kind)
        opts = str(options or '')
        about = _ABOUT_RE.search(head)
        if about and _says(head[about.end():], noun) and (not _says(head[:about.start()], noun)):
            return 'no'
        if '[truncated]' in kind:
            return 'unclear'
        if _says(kind, noun) or _says(head, noun) or _says(opts, noun):
            return 'yes'
        if _other_kind(noun, facts):
            return 'no'
        return 'unclear'
    return hio6__body_29(noun, facts, title, options)
_PLAYBOOK = {'intent': "Firm requirements: the category, the named hardware specification, the budget and stock. A named brand is only a preference: choose it when a fully compliant item of that brand exists, otherwise any compliant brand. Use filter(category, brand, max_price, in_stock=true), then view or compare to read the exact variant's specification from raw facts. Answer the shopper with message; a budget the shopper lowers later applies to every later choice.", 'recall': 'Run several differently worded searches (k=10, in_stock=true) built from the requested words and close synonyms until enough distinct matching products have appeared; every requested word must appear as a whole word of the title. Then view the strongest candidates so the controller can verify one and buy it.', 'constraint': "The shopper hides one firm hardware requirement and answers it when asked; obey it exactly for the specific variant, reading the variant's own option text before any product-wide value. Use filter(category, brand, max_price, in_stock=true) and view candidates. If the carted variant sells out, move to a different product that still meets every requirement.", 'preference': 'You already asked which preference matters most; use the answer. Search or filter the category within budget, then compare candidates with different listing_id values, reading raw facts. The first cart item is graded, so make it the best genuine match. If it is repriced above budget, recover to a different listing_id.', 'ranking': "Only the named candidate variants count, and the controller ranks them for you: compare all of them first so their facts are on the record. Highest advertised maximum refresh rate wins, then the largest screen, then the lowest price, reading the selected variant's own value before a product-wide one. After the market moves, the remaining candidates are re-ranked.", 'recovery': 'Find the preferred model and color first: search the model code, then view it so every variant and color is on the record. The first cart item must be exactly that model and color. If it disappears, relax as little as possible - the same model in another color before any other model - and never relax the category, budget or stock.', 'justification': 'Find a product whose own raw facts state the requested specification, in the requested category and brand, in stock and within budget. View or compare it so its facts are on the record, then the controller inspects it, carts it and orders it with claims taken only from that evidence.', 'generic': 'Understand the firm requirements, filter or search the catalog, view candidates, and let the controller verify and buy the best one.'}
_CORE_RULES = 'You are a careful shopping agent working a catalog through tools. Propose exactly one tool call per turn, and the turns are limited, so every turn must move the evidence on. A deterministic controller checks every proposal, groups its own follow-up actions into the same turn, and takes over the cart and the order once a candidate is fully verified, so your job is to surface and read candidates: search or filter, then view or compare so their raw facts, variant options and prices are on record. Never add an item to the cart as a placeholder - the first cart addition is graded. Prices and stock change while you shop; the controller re-checks the carted item and tells you when to move to a different product. Answer every direct question from the shopper with one short message before anything else, and never send progress or status updates. Never ask the shopper to raise the budget. Do not repeat a tool call with identical arguments; one compare of up to five products beats several view calls.'

class Clock:

    def __init__(self, budget: float) -> None:

        def hio6__body_30(self, budget, *, hio6_z=0):
            self.start = time.monotonic()
            self.budget = budget
        return hio6__body_30(self, budget)

    def remaining(self) -> float:

        def hio6__body_31(self, *, hio6_z=0):
            return self.budget - (time.monotonic() - self.start)
        return hio6__body_31(self)

class Brief:

    def __init__(self, query: str) -> None:
        self.raw = query or ''
        goal, _, rules = self.raw.partition('\n\nTask rules:')
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self._budget(self.goal)
        self.currency = ''
        m = _CURRENCY_ADJ_RE.search(self.goal)
        if m:
            self.currency = next((g for g in m.groups() if g), '')
        self.candidate_keys = [(pid, sku.rstrip('.,;:')) for pid, sku in _KEY_RE.findall(self.goal)]
        phrase = _PHRASE_RE.search(self.goal)
        quoted = _QUOTED_RE.findall(self.goal)

        def hio6__ie_4():
            if quoted:
                return quoted[0]
            else:
                return ''
        if phrase:
            source = phrase.group(1)
        else:
            source = hio6__ie_4()
        self.theme = re.findall('[a-z0-9]+', source.lower())
        sm = _SURFACE_RE.search(self.rules)
        if sm:
            self.required_identities = int(sm.group(1))
        else:
            self.required_identities = 0
        pm = _PREF_ITEM_RE.search(self.goal)
        if pm:
            self.pref_color = pm.group(1).lower()
        else:
            self.pref_color = None
        if pm:
            self.pref_model = pm.group(2)
        else:
            self.pref_model = None
        self.category = self._category(self.goal)
        self.brand = self._brand(self.goal, self.category)
        if pm:
            self.pref_brand = None
        else:
            self.pref_brand = self._pref_brand(self.goal)
        self.head_noun = self._head_noun(self.category)
        self.alt_of: dict = {}
        self.specs = self._specs(self.goal, self.currency, self.budget, self.alt_of)
        if self.brand:
            spec_words = {str(k).lower() for k in self.specs} | {str(v).lower() for v in self.specs.values()}
            kept = []
            for w in self.brand.split():
                if w.lower() not in spec_words:
                    kept.append(w)
            self.brand = ' '.join(kept) or None
        self.cat_qualifiers = self._qualifiers(self.goal, self.category, self.brand, self.pref_brand)
        self.spec_field, self.spec_value = self._spec(self.goal)
        self.order_by = self._order_by(self.raw)
        self.spec_sealed = self.spec_field is None and 'firm requirement' in self.goal.lower()
        self.family = self._family()

    def _family(self) -> str:

        def hio6__body_32(self, *, hio6_z=0):
            if self.candidate_keys:
                return 'ranking'
            if self.required_identities:
                return 'recall'
            if self.pref_model:
                return 'recovery'
            if self.theme:
                return 'recall'
            if _REASON_RE.search(self.goal):
                return 'justification'
            if self.pref_brand:
                return 'intent'
            if self.spec_sealed or self.spec_field or self.brand:
                return 'constraint'
            return 'preference'
        return hio6__body_32(self)

    @staticmethod
    def _budget(text: str) -> float | None:

        def hio6__body_33(text, *, hio6_z=0):
            best = None
            for m in _BUDGET_RE.finditer(text or ''):
                whole = m.group(1).replace(',', '')
                frac = m.group(2) or '0'
                try:
                    val = float(whole + '.' + frac)
                except ValueError:
                    continue
                if best is None or val < best:
                    best = val
            return best
        return hio6__body_33(text)

    @staticmethod
    def _category(goal: str) -> str | None:

        def hio6__body_34(goal, *, hio6_z=0):
            m = _CAT_PHRASE_RE.search(goal or '')
            if m:
                return m.group(1).strip()
            m = re.search('\\b([a-z][a-z\\-]{2,20})\\s*(?:,|\\s(?:at or (?:under|below)|at most|under|below|within|with)\\b)', goal or '', re.I)
            if m:
                return m.group(1)
            else:
                return None
        return hio6__body_34(goal)

    @staticmethod
    def _brand(goal: str, category: str | None) -> str | None:

        def hio6__body_35(goal, category, *, hio6_z=0):
            try:
                if not category:
                    return None
                m = re.search('\\b(?:an?|the)\\s+([^,.;]{0,60}?)\\b' + re.escape(category), goal or '', re.I)
                if not m:
                    return None
                words = []
                for w in re.findall("[\\w&.'\\-]+", _SPEC_WORD_RE.sub(' ', m.group(1))):
                    if w:
                        words.append(w)
                keep = []
                for w in words:
                    if w[:1].isupper() and (not any((c.isdigit() for c in w))):
                        keep.append(w)
                return ' '.join(keep) or None
            except Exception:
                raise
        return hio6__body_35(goal, category)

    @staticmethod
    def _pref_brand(goal: str) -> str | None:

        def hio6__body_36(goal, *, hio6_z=0):
            try:
                m = _PREF_BRAND_RE.search(goal or '')
                if not m:
                    return None
                name = m.group(1).strip(' .,')
                words = name.split()
                if not words or len(words) > 4 or (not any((w[:1].isupper() for w in words))):
                    return None
                return name
            except Exception:
                raise
        return hio6__body_36(goal)

    @staticmethod
    def _head_noun(category: str | None) -> str:

        def hio6__body_37(category, *, hio6_z=0):
            words = re.findall('[a-z]+', (category or '').lower())
            if words:
                return words[-1]
            else:
                return ''
        return hio6__body_37(category)

    @staticmethod
    def _qualifiers(goal: str, category: str | None, brand: str | None, pref_brand: str | None) -> tuple:

        def hio6__body_38(goal, category, brand, pref_brand, *, hio6_z=0):
            try:
                if not category:
                    return ()
                head = Brief._head_noun(category)
                if not head:
                    return ()
                m = re.search("((?:[\\w'-]+[\\s-]+){0,3})\\b" + re.escape(head) + 's?\\b', goal or '', re.I)
                if not m:
                    return ()
                skip = _tokens(category) | _tokens(brand or '') | _tokens(pref_brand or '') | _FUNCTION_WORDS
                out = []
                for word in reversed(re.findall("[\\w'-]+", _SPEC_WORD_RE.sub(' ', m.group(1)))):
                    low = word.lower().strip("-'")
                    if len(low) < 3 or any((ch.isdigit() for ch in low)) or low in skip:
                        break
                    out.append(low)
                return tuple(reversed(out))
            except Exception:
                raise
        return hio6__body_38(goal, category, brand, pref_brand)

    @staticmethod
    def _specs(goal: str, currency: str, budget: float | None, alts: dict | None=None) -> dict:

        def hio6__body_39(goal, currency, budget, alts, *, hio6_z=0):
            if alts is None:
                alts = {}
            else:
                alts = alts
            text = goal or ''
            spent = set()
            for m in _BUDGET_RE.finditer(text):
                spent.update(range(m.start(), m.end()))
            out: dict = {}
            for value, unit, at in _measures(text):
                if at in spent or _unit_eq(unit, currency) or unit.upper() == unit.lower() == '':
                    continue
                if budget is not None and abs(value - budget) < _EPS:
                    continue
                word = Brief._concept_word(text, at)
                key = word or unit
                if key not in out:
                    out[key] = _fmt(value, unit)
                alt = Brief._class_form(text, at, key)
                if alt and alt not in out:
                    out[alt] = alt
                    alts[alt] = key
            return out
        return hio6__body_39(goal, currency, budget, alts)

    @staticmethod
    def _class_form(text: str, at: int, taken: str='') -> str | None:

        def hio6__body_40(text, at, taken, *, hio6_z=0):
            try:
                m = re.match("\\s*[\\w.'-]+((?:\\s+[A-Za-z]{2,5}){1,2})(?![\\w])", (text or '')[at:])
                if not m:
                    return None
                for word in [w.lower() for w in m.group(1).split()]:
                    if word not in _FUNCTION_WORDS and word != (taken or '').lower():
                        return word
                return None
            except Exception:
                raise
        return hio6__body_40(text, at, taken)

    @staticmethod
    def _concept_word(text: str, at: int) -> str:

        def hio6__body_41(text, at, *, hio6_z=0):
            try:
                tail = re.match("\\s*[\\w.'-]+\\s*[-–]?\\s*((?:[a-z]+\\s+){0,2}[a-z]+)", (text or '')[at:], re.I)
                skip = {'and', 'or', 'with', 'the', 'a', 'an', 'at', 'under', 'over', 'for', 'of', 'in', 'to', 'that', 'is'}
                if tail:
                    for word in re.findall('[a-z]+', tail.group(1).lower()):
                        if word not in skip and len(word) > 2:
                            return word
                lead = re.findall('[a-z]+', (text or '')[max(0, at - 34):at].lower())
                for word in reversed(lead):
                    if word not in skip and len(word) > 2:
                        return word
                return ''
            except Exception:
                raise
        return hio6__body_41(text, at)

    @staticmethod
    def _order_by(text: str) -> tuple:

        def hio6__body_42(text, *, hio6_z=0):
            low = (text or '').lower()
            down = 'highest|larger|largest|biggest|greatest|most|max(?:imum)?|fastest|longest|top'
            up = 'lowest|smaller|smallest|least|min(?:imum)?|cheapest|slowest|shortest|fewest'
            money = {'price', 'cost', 'cheapest', 'priced', 'expensive'}
            out: list = []
            seen: set = set()
            for m in re.finditer('\\b(' + down + '|' + up + ')\\b((?:\\s+[\\w-]+){0,4})', low):
                if re.fullmatch(down, m.group(1)):
                    desc = -1.0
                else:
                    desc = 1.0
                if m.group(1) in money:
                    words: tuple = ('price',)
                else:
                    tail = []
                    for w in re.findall('[a-z]+', m.group(2)):
                        if len(w) > 2 and w not in _FUNCTION_WORDS:
                            tail.append(w)
                    words = tuple((w for w in tail[:3] if w not in ('wins', 'first', 'then', 'ties', 'tie')))
                    if words and words[0] in money:
                        words = ('price',)
                if not words or words in seen:
                    continue
                seen.add(words)
                out.append((desc, words))
            if not any((w == ('price',) for _d, w in out)):
                out.append((1.0, ('price',)))
            return tuple(out)
        return hio6__body_42(text)

    def _spec(self, goal: str) -> tuple:

        def hio6__body_43(self, goal, *, hio6_z=0):
            for field, value in self.specs.items():
                return (field, value)
            return (None, None)
        return hio6__body_43(self, goal)

    def adopt_spec(self, text: str) -> str | None:

        def hio6__body_44(self, text, *, hio6_z=0):
            found = self._specs(text, self.currency, None, self.alt_of)
            for field, value in found.items():
                self.spec_field, self.spec_value = (field, value)
                self.specs[field] = value
                self.spec_sealed = False
                return field + ' = ' + value
            return None
        return hio6__body_44(self, text)

def _num(value: Any) -> float | None:

    def hio6__body_45(value, *, hio6_z=0):
        try:
            if value is None:
                return None
            return float(str(value).replace(',', ''))
        except (TypeError, ValueError):
            return None
    return hio6__body_45(value)

class Ledger:

    def __init__(self) -> None:

        def hio6__body_46(self, *, hio6_z=0):
            self.cart: list = []
            self.checked: dict = {}
            self.checked_seq: dict = {}
            self.inspected: set = set()
            self.products: dict = {}
            self.rows: dict = {}
            self.options: dict = {}
            self.listing_ids: set = set()
            self.listing_of: dict = {}
            self.identities: set = set()
            self.seq = 0
            self.first_add_seq: int | None = None
            self.last_add_seq: int | None = None
            self.adds = 0
            self.event_target: str | None = None
            self.dead_keys: set = set()
            self.dead_products: set = set()
            self.calls_seen: set = set()
        return hio6__body_46(self)

    @staticmethod
    def key(pid: Any, sku: Any) -> str:

        def hio6__body_47(pid, sku, *, hio6_z=0):
            return str(pid) + '::' + str(sku or '')
        return hio6__body_47(pid, sku)

    def record(self, name: str, args: dict, obs: dict, brief: Brief) -> None:

        def hio6__body_48(self, name, args, obs, brief, *, hio6_z=0):
            self.seq += 1
            if not isinstance(obs, dict) or obs.get('error'):
                return
            if name in ('search', 'filter'):
                for row in obs.get('results') or []:
                    if isinstance(row, dict):
                        self._note_row(row, brief)
            elif name == 'view':
                self._note_product(obs, brief)
            elif name == 'compare':
                for prod in obs.get('compared') or []:
                    if isinstance(prod, dict):
                        self._note_product(prod, brief)
            elif name == 'inspect_stock':
                k = self.key(obs.get('product_id'), obs.get('sku'))
                self.checked[k] = {'in_stock': bool(obs.get('in_stock')), 'price': _num(obs.get('price'))}
                self.checked_seq[k] = self.seq
                self.inspected.add(k)
            elif name == 'add_to_cart':
                added = obs.get('added') or {}
                pid, sku = (added.get('product_id'), added.get('sku'))
                if pid and sku:
                    self.adds += 1
                    if (pid, sku) not in self.cart:
                        self.cart.append((pid, sku))
                    if self.first_add_seq is None:
                        self.first_add_seq = self.seq
                    self.last_add_seq = self.seq
                self._sync_cart(obs)
            elif name == 'remove_from_cart':
                self._sync_cart(obs)
                removed = obs.get('removed') or {}
                pair = (removed.get('product_id'), removed.get('sku'))
                if pair in self.cart:
                    self.cart.remove(pair)
            elif name == 'inspect_cart':
                self._sync_cart(obs)
        return hio6__body_48(self, name, args, obs, brief)

    def _sync_cart(self, obs: dict) -> None:

        def hio6__body_49(self, obs, *, hio6_z=0):
            rows = obs.get('cart')
            if not isinstance(rows, list):
                return
            pairs = []
            for r in rows:
                if isinstance(r, dict) and r.get('product_id'):
                    pairs.append((r.get('product_id'), r.get('sku')))
            if pairs or rows == []:
                self.cart = pairs
            for r in rows:
                if not isinstance(r, dict) or not r.get('product_id'):
                    continue
                k = self.key(r.get('product_id'), r.get('sku'))
                price = _num(r.get('current_price'))
                stock = r.get('current_in_stock')
                if price is None and stock is None:
                    continue
                prev = self.checked.get(k, {})
                self.checked[k] = {'in_stock': bool(stock) if stock is not None else prev.get('in_stock', True), 'price': price if price is not None else prev.get('price')}
                self.checked_seq[k] = self.seq
        return hio6__body_49(self, obs)

    def _note_row(self, row: dict, brief: Brief) -> None:

        def hio6__body_50(self, row, brief, *, hio6_z=0):
            try:
                pid = row.get('product_id')
                if not pid:
                    return
                title = str(row.get('title') or '')
                if brief.theme:
                    if all((t in _tokens(title) for t in brief.theme)):
                        self.identities.add(str(pid))
                else:
                    self.identities.add(str(pid))
                if not row.get('sku'):
                    return
                k = self.key(pid, row.get('sku'))
                self.rows[k] = row
                if row.get('price') is not None:
                    self.checked.setdefault(k, {'in_stock': bool(row.get('in_stock', True)), 'price': _num(row.get('price'))})
            except Exception:
                raise
        return hio6__body_50(self, row, brief)

    def _note_product(self, prod: dict, brief: Brief) -> None:

        def hio6__body_51(self, prod, brief, *, hio6_z=0):
            pid = prod.get('product_id')
            if not pid:
                return
            self.products[str(pid)] = prod
            if prod.get('listing_id'):
                self.listing_ids.add(str(prod['listing_id']))
                self.listing_of[str(pid)] = str(prod['listing_id'])
            if brief.theme and all((t in _tokens(str(prod.get('title') or '')) for t in brief.theme)):
                self.identities.add(str(pid))
            for v in prod.get('variants') or []:
                if not isinstance(v, dict) or not v.get('sku'):
                    continue
                k = self.key(pid, v.get('sku'))
                self.checked[k] = {'in_stock': bool(v.get('in_stock')), 'price': _num(v.get('price'))}
                self.checked_seq[k] = self.seq
                self.options[k] = str(v.get('options') or '')
        return hio6__body_51(self, prod, brief)

    def status(self, pid: str, sku: str) -> dict | None:

        def hio6__body_52(self, pid, sku, *, hio6_z=0):
            return self.checked.get(self.key(pid, sku))
        return hio6__body_52(self, pid, sku)

def _arguments(tool_call: dict) -> dict:

    def hio6__body_53(tool_call, *, hio6_z=0):
        raw = (tool_call.get('function') or {}).get('arguments', '{}')
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw or '{}')
        except (TypeError, ValueError):
            parsed = {}
        if isinstance(parsed, dict):
            return parsed
        else:
            return {}
    return hio6__body_53(tool_call)

def _trim(text: str, limit: int) -> str:

    def hio6__body_54(text, limit, *, hio6_z=0):
        if len(text) <= limit:
            return text
        return text[:limit - 40] + ' ... (truncated by agent)"}'
    return hio6__body_54(text, limit)

class Agent:

    def __init__(self, problem: dict) -> None:

        def hio6__body_55(self, problem, *, hio6_z=0):
            env = problem['environment']
            self.binding = dict(env['binding'])
            self.policy = env['policy_view']
            self.tools = list(self.policy.get('tools') or [])
            self.tool_names = {(t.get('function') or {}).get('name') for t in self.tools if isinstance(t, dict)}
            self.max_steps = int(self.policy.get('max_steps') or 30)
            self.max_calls = max(1, int(self.policy.get('max_calls_per_turn') or 1))
            self.problem_id = str(problem.get('problem_id') or problem.get('id') or 'problem')
            self.brief = Brief(str(self.policy.get('query') or ''))
            self.budget = self.brief.budget
            self.ledger = Ledger()
            self.clock = Clock(_PROBLEM_BUDGET_S)
            self.dialogue: list = []
            self.messages: list = [{'role': 'system', 'content': self._system_prompt()}, {'role': 'user', 'content': self.policy.get('query') or ''}]
            self.turn = 0
            self.asked_shopper = False
            self.shopper_question_pending = False
            self.shopper_updated = False
            self.rejected_streak = 0
            self.pending_envelope: dict | None = None
            self.search_calls = 0
            self.cart_inspections = 0
            self.pending_recheck: tuple | None = None
            self.forced_compares = 0
            self.hinted_exact = False
            self.tried: dict = {}
            self.questions_asked = 0
            self.searched_siblings = False
            self.hinted_brand = False
            self._wishes: set | None = None
            self._wish_ranked: list | None = None
            self.replies: list = []
            self._brand_seen: bool | None = None
            self._spec_claim: tuple | None = None
            self._spec_asked = False
            self._axis_unit: dict | None = None
            self._spoken_cache: dict = {}
            self._live_terms: dict = {}
            self.probes: set = set()
            self.searched_twins = False
            self.reads_done = False
            self.event_turn: int | None = None
            self.unreadable: set = set()
            self.question_waits = 0
            self.done = False
            self.model_index = 0
            self.actions = 0
            self.model_fails = 0
            self.model_down = False
            self.barren = 0
            self.pool_seen = 0
            self.explored: set = set()
            provider = (getenv('INFERENCE_PROVIDER') or '').strip().lower()
            self.roster = _MODELS.get(provider) or _MODELS['chutes']
            pinned = getenv('SANDBOX_MODEL')
            if pinned:
                self.roster = (pinned,) + tuple((m for m in self.roster if m != pinned))
        return hio6__body_55(self, problem)

    def _system_prompt(self) -> str:

        def hio6__body_56(self, *, hio6_z=0):
            play = _PLAYBOOK.get(self.brief.family, _PLAYBOOK['generic'])
            extra = ' You have at most %d turns.' % self.max_steps
            if self.brief.family == 'recall' and self.brief.required_identities:
                extra += ' You must surface at least %d distinct matching products in search or filter results before ordering; the controller reports the count after each search.' % self.brief.required_identities
            return _CORE_RULES + '\n\nFamily playbook: ' + play + extra
        return hio6__body_56(self)

    def _opening_question(self) -> str | None:

        def hio6__body_57(self, *, hio6_z=0):
            if self.brief.family == 'constraint' or self.brief.spec_sealed:
                return 'Before I search: which specification is your firm requirement, and what exact value does it have to be?'
            if self.brief.family == 'preference':
                return 'Quick question so I choose well: which of your preferences counts most, and in what order should I weigh the others? Let me know your priority.'
            return None
        return hio6__body_57(self)

    def _note(self, text: str) -> None:

        def hio6__body_58(self, text, *, hio6_z=0):
            self.messages.append({'role': 'user', 'content': '[note] ' + text})
            self.dialogue.append({'role': 'assistant', 'content': '[note] ' + text[:300]})
        return hio6__body_58(self, text)

    def _prod(self, pid: str) -> dict:

        def hio6__body_59(self, pid, *, hio6_z=0):
            return self.ledger.products.get(str(pid)) or {}
        return hio6__body_59(self, pid)

    def _facts(self, pid: str) -> dict:

        def hio6__body_60(self, pid, *, hio6_z=0):
            f = self._prod(pid).get('facts')
            if isinstance(f, dict):
                return f
            else:
                return {}
        return hio6__body_60(self, pid)

    def _title(self, pid: str) -> str:

        def hio6__body_61(self, pid, *, hio6_z=0):
            title = str(self._prod(pid).get('title') or '')
            if title:
                return title
            for key, row in self.ledger.rows.items():
                if key.startswith(str(pid) + '::'):
                    return str(row.get('title') or '')
            return ''
        return hio6__body_61(self, pid)

    def _brand_of(self, pid: str) -> str:

        def hio6__body_62(self, pid, *, hio6_z=0):
            try:
                brand = self._prod(pid).get('brand')
                if brand:
                    return str(brand)
                for key, row in self.ledger.rows.items():
                    if key.startswith(str(pid) + '::') and row.get('brand'):
                        return str(row['brand'])
                return ''
            except Exception:
                raise
        return hio6__body_62(self, pid)

    def _cat_path(self, pid: str) -> list:

        def hio6__body_63(self, pid, *, hio6_z=0):
            path = self._prod(pid).get('category_path')
            if isinstance(path, list):
                return list(path)
            else:
                return []
        return hio6__body_63(self, pid)

    def _options(self, pid: str, sku: str) -> str:

        def hio6__body_64(self, pid, sku, *, hio6_z=0):
            return self.ledger.options.get(Ledger.key(pid, sku), '')
        return hio6__body_64(self, pid, sku)

    def _strip_models(self, pid: str, title: str) -> str:

        def hio6__body_65(self, pid, title, *, hio6_z=0):
            try:
                claimed = set()
                for v in self.brief.specs.values():
                    claimed.add(v)
                for token in re.split('[;,]', str(self._facts(pid).get('model') or '')):
                    token = token.strip()
                    if token:
                        stem = re.split('[-_]', token)[0]
                    else:
                        stem = ''
                    if token and token.lower() not in claimed and re.fullmatch('[a-z0-9]+(?:[-_][a-z0-9]+)+', token, re.I) and re.search('[a-z]', stem, re.I) and re.search('\\d', stem):
                        title = re.sub('(?<![\\w-])' + re.escape(token) + '(?![\\w-])', '', title, flags=re.I)
                return title
            except Exception:
                raise
        return hio6__body_65(self, pid, title)

    def _spec_forms(self) -> list:

        def hio6__body_66(self, *, hio6_z=0):
            field, want = (self.brief.spec_field, self.brief.spec_value)
            if not (field and want):
                return []
            return [(field, want)] + [(k, v) for k, v in self.brief.specs.items() if self.brief.alt_of.get(k) == field]
        return hio6__body_66(self)

    def _want_of(self, field: str) -> str | None:

        def hio6__body_67(self, field, *, hio6_z=0):
            want = self.brief.specs.get(field)
            if want:
                return want
            if field and field == self.brief.spec_field:
                return self.brief.spec_value
            else:
                return None
        return hio6__body_67(self, field)

    def _facts_by_label(self, pid: str, field: str) -> tuple:

        def hio6__body_68(self, pid, field, *, hio6_z=0):
            named, rest = ([], [])
            facts = self._facts(pid)
            for key, raw in facts.items() if isinstance(facts, dict) else ():
                text = str(raw or '').strip()
                if not text or '[truncated]' in text:
                    continue
                label = str(key).replace('_', ' ')
                (named if _says(label, field) else rest).append((label, text))
            return (named, rest)
        return hio6__body_68(self, pid, field)

    def _fact_value(self, pid: str, field: str, want: str | None=None) -> str | None:

        def hio6__body_69(self, pid, field, want, *, hio6_z=0):
            want = want or self._want_of(field)
            if not want:
                return None
            number, unit = _split_value(want)
            for group in self._facts_by_label(pid, field):
                found: set = set()
                for label, text in group:
                    found |= _values_for(field, want, label + ' ' + text)
                    if number is not None and _says(label, field) and re.fullmatch('\\d+(?:\\.\\d+)?', text):
                        found.add(_fmt(float(text), unit))
                if found:
                    return _one(found)
            return None
        return hio6__body_69(self, pid, field, want)

    def _product_value(self, pid: str, field: str, want: str | None=None) -> str | None:

        def hio6__body_70(self, pid, field, want, *, hio6_z=0):
            try:
                want = want or self._want_of(field)
                if not want:
                    return None
                vals: set = set()
                fact = self._fact_value(pid, field, want)
                if fact:
                    vals.add(fact)
                number, unit = _split_value(want)
                if number is not None and len(unit) == 1:
                    return _one(vals)
                title = self._title(pid)
                if _split_value(want)[0] is None:
                    title = self._strip_models(pid, title)
                vals |= _values_for(field, want, title)
                return _one(vals)
            except Exception:
                raise
        return hio6__body_70(self, pid, field, want)

    def _variant_value(self, pid: str, sku: str, field: str) -> str | None:

        def hio6__body_71(self, pid, sku, field, *, hio6_z=0):
            try:
                want = self._want_of(field)
                if not want:
                    return None
                opt = self._options(pid, sku)
                vals = _values_for(field, want, opt)
                if len(vals) > 1:
                    return None
                if vals:
                    return next(iter(vals))
                if _mentions(field, want, opt):
                    return None
                return self._product_value(pid, field)
            except Exception:
                raise
        return hio6__body_71(self, pid, sku, field)

    def _axis_units(self) -> dict:

        def hio6__body_72(self, *, hio6_z=0):
            if self._axis_unit is None:
                self._axis_unit = {}
                axes = []
                for _desc, w in self.brief.order_by:
                    if w != ('price',):
                        axes.append(w)
                if axes:
                    answer = self._ask_text('A shopper is ranking products and asked to order them by these measures:\n%s\n\nFor each measure, name the unit that a product listing prints immediately after the figure. Reply with one line per measure and nothing else, in the form:  <measure>=<unit>' % '\n'.join((' '.join(w) for w in axes)))
                    for line in str(answer or '').splitlines():
                        found = re.match('\\s*[-*]?\\s*([A-Za-z][\\w ]{0,30}?)\\s*[=:]\\s*([A-Za-z]{1,6})\\b', line)
                        if not found:
                            continue
                        named, unit = (found.group(1).lower(), found.group(2).lower())
                        for words in axes:
                            if all((w in named for w in words)):
                                self._axis_unit[words] = unit
            return self._axis_unit
        return hio6__body_72(self)

    def _measure_in(self, pid: str, sku: str, unit: str) -> float | None:

        def hio6__body_73(self, pid, sku, unit, *, hio6_z=0):
            facts = self._facts(pid)
            sources = [self._options(pid, sku)]
            sources += [str(v or '') for _k, v in sorted((facts if isinstance(facts, dict) else {}).items())]
            sources += [self._title(pid)]
            for text in sources:
                if not text or '[truncated]' in text:
                    continue
                hits = set()
                for value, seen, _at in _measures(text):
                    if _unit_eq(seen, unit):
                        hits.add(value)
                if len(hits) == 1:
                    return hits.pop()
            return None
        return hio6__body_73(self, pid, sku, unit)

    def _measure(self, pid: str, sku: str, words: tuple) -> float | None:

        def hio6__body_74(self, pid, sku, words, *, hio6_z=0):
            unit = self._axis_units().get(words)
            if unit:
                got = self._measure_in(pid, sku, unit)
                if got is not None:
                    return got
            best: tuple | None = None
            for source in (self._options(pid, sku), self._title(pid)):
                for value, _unit_text, at in _measures(source):
                    gap = min((_gap(at, p) for w in words for p in _marks(source, w)), default=None)
                    if gap is not None and gap <= 26.0 and (best is None or gap < best[0]):
                        best = (gap, value)
                if best is not None:
                    return best[1]
            facts = self._facts(pid)
            for key, raw in facts.items() if isinstance(facts, dict) else ():
                label = str(key).replace('_', ' ')
                text = str(raw or '').strip()
                if not text or '[truncated]' in text or (not any((_says(label, w) for w in words))):
                    continue
                if re.fullmatch('\\d+(?:\\.\\d+)?', text):
                    return float(text)
                found = _measures(text)
                if len(found) == 1:
                    return found[0][0]
            return None
        return hio6__body_74(self, pid, sku, words)

    @staticmethod
    def _stems(text: str) -> set:

        def hio6__body_75(text, *, hio6_z=0):
            return {w.rstrip('s') for w in re.findall('[a-z]+', (text or '').lower())}
        return hio6__body_75(text)

    def _category_ok(self, pid: str) -> bool:

        def hio6__body_76(self, pid, *, hio6_z=0):
            try:
                want = self.brief.category
                path = self._cat_path(pid)
                if not want or not path:
                    return True
                wanted = self._stems(want)
                labels = set()
                for label in path:
                    labels |= self._stems(str(label))
                placed = bool(wanted & labels) or any((self._stems(''.join(re.findall('[a-z]+', str(label).lower()))) == wanted for label in path))
                if not placed and len(wanted) > 1:
                    joined = ''.join(sorted(wanted))
                    placed = any((''.join(sorted(self._stems(str(label)))) == joined or re.sub('[^a-z]', '', str(label).lower()).rstrip('s') in {w for w in wanted if len(w) > 4} for label in path))
                if not placed:
                    return False
                title = self._title(pid)
                for term in self.brief.cat_qualifiers:
                    if self._says_term(title, term):
                        continue
                    if self._term_in_catalog(term):
                        return False
                return True
            except Exception:
                raise
        return hio6__body_76(self, pid)

    def _says_term(self, text: str, term: str) -> bool:

        def hio6__body_77(self, text, term, *, hio6_z=0):
            try:
                parts = []
                for p in re.findall('[a-z]+', term.lower()):
                    if p:
                        parts.append(p)
                if not parts:
                    return True
                words = self._stems(text) | _tokens(text)
                if all((p in words or p.rstrip('s') in words for p in parts)):
                    return True
                return ''.join(parts) in re.sub('[^a-z]', '', (text or '').lower())
            except Exception:
                raise
        return hio6__body_77(self, text, term)

    def _term_in_catalog(self, term: str) -> bool:

        def hio6__body_78(self, term, *, hio6_z=0):
            if self._live_terms.get(term):
                return True
            titles = []
            for row in self.ledger.rows.values():
                titles.append(str(row.get('title') or ''))
            titles += [str(p.get('title') or '') for p in self.ledger.products.values()]
            live = any((self._says_term(t, term) for t in titles))
            if live:
                self._live_terms[term] = True
            return live
        return hio6__body_78(self, term)

    def _brand_ok(self, pid: str) -> bool:

        def hio6__body_79(self, pid, *, hio6_z=0):
            if not self._brand_live():
                return True
            wanted = _tokens(self.brief.brand)
            return bool(wanted <= _tokens(self._brand_of(pid) + ' ' + self._title(pid)))
        return hio6__body_79(self, pid)

    def _model_live(self) -> bool:

        def hio6__body_80(self, *, hio6_z=0):
            if not self.brief.pref_model:
                return False
            want = re.sub('[^a-z0-9]+', '', self.brief.pref_model.lower())
            titles = []
            for r in self.ledger.rows.values():
                titles.append(str(r.get('title') or ''))
            if any((want in re.sub('[^a-z0-9]+', '', t.lower()) for t in titles)):
                return True
            return len(titles) < 8
        return hio6__body_80(self)

    def _brand_live(self) -> bool:

        def hio6__body_81(self, *, hio6_z=0):
            if not self.brief.brand:
                return False
            if self._brand_seen is None:
                wanted = _tokens(self.brief.brand)
                noun = self.brief.head_noun
                rows = list(self.ledger.rows.values())
                seen = 0
                for row in rows:
                    title = str(row.get('title') or '')
                    if noun and (not _says(title, noun)) and (not _covers(title, noun)):
                        continue
                    seen += 1
                    if wanted <= _tokens(str(row.get('brand') or '') + ' ' + title):
                        self._brand_seen = True
                        return True
                return seen < 6
            return self._brand_seen
        return hio6__body_81(self)

    def _device_ok(self, pid: str, sku: str) -> bool:

        def hio6__body_82(self, pid, sku, *, hio6_z=0):
            return self._identity(pid, sku) != 'no'
        return hio6__body_82(self, pid, sku)

    def _identity(self, pid: str, sku: str) -> str:

        def hio6__body_83(self, pid, sku, *, hio6_z=0):
            return _identity_of(self.brief.head_noun, self._facts(pid), self._title(pid), self._options(pid, sku))
        return hio6__body_83(self, pid, sku)

    def _relax(self, pid: str, sku: str) -> int:
        model, color = (self.brief.pref_model, self.brief.pref_color)
        if not model:
            return 0

        def hio6__lam_2(t, *, hio6_z=0):

            def hio6__body_84(t, *, hio6_z=0):
                return re.sub('[^a-z0-9]+', '', (t or '').lower())
            return hio6__body_84(t)
        squash = hio6__lam_2
        blob = squash(self._title(pid) + ' ' + ' '.join((str(v) for v in self._facts(pid).values())))
        same_model = squash(model) in blob
        shade = (self._options(pid, sku) + ' ' + str(self._facts(pid).get('color') or '') + ' ' + self._title(pid)).lower()
        same_color = bool(color) and re.search('(?<![a-z])' + re.escape(color) + '(?![a-z])', shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        if same_color:
            return 2
        else:
            return 3

    def _assess(self, pid: str, sku: str) -> dict:
        fam = self.brief.family
        key = Ledger.key(pid, sku)
        st = self.ledger.checked.get(key) or {}
        price, stock = (st.get('price'), st.get('in_stock'))
        dead = key in self.ledger.dead_keys or (str(pid) in self.ledger.dead_products and fam != 'recovery')
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and (not dead)
        viewed = str(pid) in self.ledger.products
        field, want = (self.brief.spec_field, self.brief.spec_value)
        spec = 2
        if field and want:
            seen = []
            for f, w in self._spec_forms():
                seen.append((self._variant_value(pid, sku, f), w))
            if any((got == w for got, w in seen)):
                spec = 2
            elif any((got for got, _w in seen)):
                spec = -1
            else:
                spec = 0
        ok = bool(live and viewed and (spec == 2) and self._brand_ok(pid) and self._device_ok(pid, sku))
        cat_ok = self._category_ok(pid)
        ok = ok and cat_ok
        proven = True
        if fam == 'justification':
            ok = ok and bool(self._brand_of(pid))
            if field and want:
                proven = any((self._fact_value(pid, f, w) == w and self._product_value(pid, f, w) == w for f, w in self._spec_forms()))
                ok = ok and (proven or self.turn > self.max_steps - _ENDGAME_TURNS)
        if fam == 'preference' and self.ledger.event_target:
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split('::', 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if fam == 'recall' and self.brief.theme:
            ok = ok and all((t in _tokens(self._title(pid)) for t in self.brief.theme))
        if fam == 'ranking':
            ok = ok and (str(pid), str(sku)) in self.brief.candidate_keys
            ok = ok and all((self._measure(pid, sku, words) is not None for _desc, words in self.brief.order_by if words != ('price',)))

        def hio6__ie_5():
            if fam == 'recovery':
                return self._relax(pid, sku)
            else:
                return 0
        return {'pid': str(pid), 'sku': str(sku), 'key': key, 'price': price, 'live': live, 'viewed': viewed, 'spec': spec, 'cat_ok': cat_ok, 'ok': ok, 'proven': proven, 'relax': hio6__ie_5()}

    def _rank_of(self, c: dict, first: bool) -> tuple:

        def hio6__body_85(self, c, first, *, hio6_z=0):
            fam = self.brief.family
            if c['price'] is not None:
                price = c['price']
            else:
                price = 1000000000000.0
            if fam == 'ranking':
                keys = []
                for desc, words in self.brief.order_by:
                    if words == ('price',):
                        value = price
                    else:
                        value = self._measure(c['pid'], c['sku'], words)
                    keys.append(desc * (value if value is not None else 0.0))
                return (tuple(keys), price, c['key'])
            if fam == 'recovery':
                return (c['relax'], price, c['key'])
            if fam == 'preference':
                return (-self._priority_hits(c['pid'], c['sku']), price, c['key'])
            extra = sum((1 for f, v in self.brief.specs.items() if f != self.brief.spec_field and self._variant_value(c['pid'], c['sku'], f) != v))
            if self._identity(c['pid'], c['sku']) == 'yes':
                proven = 0
            else:
                proven = 1
            if fam == 'intent' and first and self.brief.pref_brand:
                wanted = _tokens(self.brief.pref_brand)
                if wanted <= _tokens(self._brand_of(c['pid'])) or wanted <= _tokens(self._title(c['pid'])):
                    near = 0
                else:
                    near = 1
                return (near, extra, proven, price, c['key'])
            return (extra, proven, price, c['key'])
        return hio6__body_85(self, c, first)

    def _candidates(self) -> list:

        def hio6__body_86(self, *, hio6_z=0):
            seen = set()
            out = []
            for key in list(self.ledger.checked):
                pid, _, sku = key.partition('::')
                if not sku or key in seen:
                    continue
                seen.add(key)
                out.append(self._assess(pid, sku))
            return out
        return hio6__body_86(self)

    def _prefer_backed(self, good: list) -> list:

        def hio6__body_87(self, good, *, hio6_z=0):
            if self.brief.family != 'justification':
                return good
            return [c for c in good if c['proven']] or good
        return hio6__body_87(self, good)

    def _target(self) -> dict | None:

        def hio6__body_91(self, *, hio6_z=0):
            try:
                first = self.ledger.adds == 0
                fam = self.brief.family
                good = self._prefer_backed([c for c in self._candidates() if c['ok']])

                def hio6__lam_6(c, *, hio6_z=0):

                    def hio6__body_88(c, *, hio6_z=0):
                        return -self._priority_hits(c['pid'], c['sku'])
                    return hio6__body_88(c)

                def hio6__lam_7(c, *, hio6_z=0):

                    def hio6__body_89(c, *, hio6_z=0):
                        return (-self._priority_hits(c['pid'], c['sku']), c['price'] if c['price'] is not None else 1000000000000.0)
                    return hio6__body_89(c)
                if fam == 'preference':
                    carted = []
                    for c in good:
                        if (c['pid'], c['sku']) in self.ledger.cart:
                            carted.append(c)
                    if carted:
                        carted.sort(key=hio6__lam_6)
                        return carted[0]
                    if good and (self.model_down or self.turn >= self.max_steps - 8):
                        good.sort(key=hio6__lam_7)
                        return good[0]
                    return None
                if fam == 'intent' and first and self.brief.pref_brand:
                    wanted = _tokens(self.brief.pref_brand)
                    named = []
                    for c in good:
                        if wanted <= _tokens(self._brand_of(c['pid'])) or wanted <= _tokens(self._title(c['pid'])):
                            named.append(c)
                    if named:
                        good = named
                    elif self.turn < self.max_steps * 0.6:
                        self._preferred_brand_hint()
                        return None
                if fam == 'recovery' and first:
                    exact = []
                    for c in good:
                        if c['relax'] == 0:
                            exact.append(c)
                    if exact:
                        good = exact
                    elif self.turn < self.max_steps * 0.6:
                        self._exact_match_hint()
                        return None
                if not good:
                    return None

                def hio6__lam_8(c, *, hio6_z=0):

                    def hio6__body_90(c, *, hio6_z=0):
                        return self._rank_of(c, first)
                    return hio6__body_90(c)
                good.sort(key=hio6__lam_8)
                return good[0]
            except Exception:
                raise
        return hio6__body_91(self)

    def _answer_text(self) -> str:

        def hio6__body_92(self, *, hio6_z=0):
            target = self._target() or (self._candidates() or [None])[0]
            if not target:
                return 'I am still checking the options against your requirements; nothing is decided yet.'
            pid, sku = (target['pid'], target['sku'])
            parts = ['I am going with ' + (self._title(pid) or pid)[:90]]
            field, want = (self.brief.spec_field, self.brief.spec_value)
            if field and want and (self._variant_value(pid, sku, field) == want):
                parts.append('its listed ' + field.replace('_', ' ') + ' is ' + want)
            if target.get('price') is not None:
                parts.append('it is %.2f %s and in stock' % (target['price'], self.brief.currency))
            return ', '.join(parts) + '.'
        return hio6__body_92(self)

    def _read_more(self) -> tuple | None:

        def hio6__body_93(self, *, hio6_z=0):
            fam = self.brief.family
            if fam == 'recall' and len(self.ledger.identities) < self._recall_need():
                return None
            twin = self._twin_search()
            if twin is not None:
                return twin
            leads = self._unseen_leads()
            if leads and self.forced_compares < 8 and (self.turn >= 2):
                self.forced_compares += 1
                if len(leads) > 1 and 'compare' in self.tool_names:
                    return ('compare', {'product_ids': leads[:5]})
                return ('view', {'product_id': leads[0]})
            probe = (self.brief.brand or '', '%.2f' % (self.budget or 0.0), str(self.brief.spec_value))
            if self.turn < 2:
                return None
            if probe in self.probes:
                self.reads_done = True
                return None
            self.probes.add(probe)
            if 'filter' in self.tool_names and self.brief.category:
                args = {'category': self.brief.category, 'in_stock': True, 'k': 10}
                if self.budget is not None:
                    args['max_price'] = self.budget
                if self.brief.brand:
                    args['brand'] = self.brief.brand
                return ('filter', args)
            if 'search' in self.tool_names:
                words = []
                for w in (self.brief.brand, self.brief.spec_value, self.brief.category):
                    if w:
                        words.append(w)
                if words:
                    return ('search', {'query': ' '.join(words), 'in_stock': True, 'k': 10})
            self.reads_done = True
            return None
        return hio6__body_93(self)

    def _explore(self) -> tuple | None:

        def hio6__body_94(self, *, hio6_z=0):
            try:
                if self.done:
                    return None
                if self.shopper_question_pending and 'message' in self.tool_names:
                    return ('message', {'content': self._answer_text()})
                forced = self._read_more()
                if forced is not None:
                    return forced
                leads = self._unseen_leads()
                if leads:
                    if len(leads) > 1 and 'compare' in self.tool_names:
                        return ('compare', {'product_ids': leads[:5]})
                    return ('view', {'product_id': leads[0]})
                if 'search' not in self.tool_names:
                    return None
                starved = not any((c['ok'] for c in self._candidates()))
                if (self.reads_done or self.barren >= _BARREN_LIMIT) and (not (starved and self.turn <= self.max_steps - _ENDGAME_TURNS - 2)):
                    return None
                query = self._next_query()
                if not query:
                    return None
                return ('search', {'query': query, 'in_stock': True, 'k': 10})
            except Exception:
                raise
        return hio6__body_94(self)

    def _next_query(self) -> str | None:
        ordered: list = []

        def push(word):
            if word and word.lower() not in [w.lower() for w in ordered]:
                ordered.append(word)
        if self.brief.theme:
            phrase = ' '.join(self.brief.theme)
            for extra in ('', ' ' + str(self.brief.category or '')):
                query = (phrase + extra).strip()
                if query and query not in self.explored:
                    self.explored.add(query)
                    return query
        push(self.brief.brand)
        for field, value in self.brief.specs.items():
            number, unit = _split_value(value)
            push('%g %s' % (number, unit) if number is not None else value)
            push(field)
        push(self.brief.category)
        for word in self._wish_order():
            push(word)
        for take in range(min(5, len(ordered)), 0, -1):
            query = ' '.join(ordered[:take])
            if query and query not in self.explored:
                self.explored.add(query)
                return query
        return None

    def _twin_search(self) -> tuple | None:

        def hio6__body_95(self, *, hio6_z=0):
            try:
                dead = self.ledger.event_target
                if self.searched_twins or not dead or 'search' not in self.tool_names:
                    return None
                title = self._title(dead.split('::', 1)[0])
                if not title:
                    return None
                self.searched_twins = True
                self._note('The item that died is one of several listings sharing a title; searching that title surfaces its other colors and sizes, which meet the same requirements.')
                return ('search', {'query': ' '.join(title.split()[:8]), 'k': 10, 'in_stock': True})
            except Exception:
                raise
        return hio6__body_95(self)

    def _preferred_brand_hint(self) -> None:

        def hio6__body_96(self, *, hio6_z=0):
            if self.hinted_brand:
                return
            self.hinted_brand = True
            self._note("No %s product that meets every firm requirement has been read yet. The shopper's first choice is graded on that brand, so filter by brand=%s within the budget and read those products before any other." % (self.brief.pref_brand, self.brief.pref_brand))
        return hio6__body_96(self)

    def _sibling_search(self) -> tuple | None:

        def hio6__body_97(self, *, hio6_z=0):
            try:
                if self.searched_siblings:
                    return None
                rows = []
                for c in self._candidates():
                    if c['live'] and c['viewed'] and (c['relax'] == 0):
                        rows.append(c)
                if rows:
                    return None
                near = []
                for c in self._candidates():
                    if c['viewed'] and c['relax'] == 1:
                        near.append(c)
                if not near:
                    return None
                title = self._title(near[0]['pid'])
                if not title:
                    return None
                self.searched_siblings = True
                return ('search', {'query': ' '.join(title.split()[:8]), 'k': 10, 'in_stock': True})
            except Exception:
                raise
        return hio6__body_97(self)

    def _exact_match_hint(self) -> None:

        def hio6__body_98(self, *, hio6_z=0):
            try:
                if self.hinted_exact or not self.brief.pref_model:
                    return
                self.hinted_exact = True

                def hio6__ie_9():
                    if self.brief.pref_color:
                        return ', ' + self.brief.pref_color
                    else:
                        return ''
                self._note("The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own and view the results so every color variant is on record; only that exact pairing may be the first cart item." % (self.brief.pref_model, hio6__ie_9()))
            except Exception:
                raise
        return hio6__body_98(self)

    def _unseen_leads(self) -> list:

        def hio6__body_99(self, *, hio6_z=0):
            leads = []
            for key, row in self.ledger.rows.items():
                pid, _, sku = key.partition('::')
                if not sku or str(pid) in self.ledger.products or str(pid) in self.ledger.dead_products:
                    continue
                st = self.ledger.checked.get(key) or {}
                price = st.get('price')
                if not st.get('in_stock') or price is None or (self.budget is not None and price > self.budget + _EPS):
                    continue
                title = str(row.get('title') or '')
                if self.brief.theme and (not all((t in _tokens(title) for t in self.brief.theme))):
                    continue
                if self.brief.pref_model and self.ledger.adds == 0 and (not self.ledger.event_target) and self._model_live():
                    squashed = re.sub('[^a-z0-9]+', '', title.lower())
                    if re.sub('[^a-z0-9]+', '', self.brief.pref_model.lower()) not in squashed:
                        continue
                if self._brand_live():
                    wanted = _tokens(self.brief.brand)
                    if not wanted <= _tokens(str(row.get('brand') or '') + ' ' + title):
                        continue
                leads.append((price, str(pid)))
            leads.sort()
            out = []
            for _price, pid in leads:
                if pid not in out:
                    out.append(pid)
            return out
        return hio6__body_99(self)

    def _shortlist_note(self) -> None:

        def hio6__body_101(self, *, hio6_z=0):
            try:
                cands = self._candidates()

                def hio6__lam_10(c, *, hio6_z=0):

                    def hio6__body_100(c, *, hio6_z=0):
                        return (-self._priority_hits(c['pid'], c['sku']), c['price'] if c['price'] is not None else 1000000000000.0)
                    return hio6__body_100(c)
                if self.brief.family == 'preference' and (not self.ledger.cart):
                    good = []
                    for c in cands:
                        if c['ok']:
                            good.append(c)
                    good.sort(key=hio6__lam_10)
                    if good:
                        self._note('Products that meet every firm requirement so far: ' + '; '.join(('%s::%s at %.2f' % (c['pid'], c['sku'], c['price'] or 0.0) for c in good[:6])) + ". Pick the one that best fits the shopper's stated priorities, inspect it, then add it.")
                        return
                best = self._target()
                if best:
                    self._note('Verified candidate: %s::%s at %.2f. The controller will inspect, cart and order it; read other candidates only if one could rank better.' % (best['pid'], best['sku'], best['price'] or 0.0))
                    return
                field, want = (self.brief.spec_field, self.brief.spec_value)
                blocked = []
                for c in cands:
                    if c['viewed'] and c['live'] and (not c['ok']):
                        blocked.append(c)
                if blocked and field:
                    sample = blocked[:3]
                    self._note("None of the products read so far can be verified. For each, the evidence for %s must resolve to exactly %s from the variant's own option text or the product's raw facts: %s. Look at different products." % (field, want, ', '.join((c['pid'] + '::' + c['sku'] for c in sample))))
                elif blocked:
                    self._note('The products read so far fail a firm requirement (category, brand, stock, budget or the kind of device requested). Search for different products.')
            except Exception:
                raise
        return hio6__body_101(self)

    def _priority_hits(self, pid: str, sku: str) -> float:

        def hio6__body_102(self, pid, sku, *, hio6_z=0):
            try:
                ranked = self._wish_order()
                if not ranked:
                    return 0.0
                blob = self._title(pid) + ' ' + self._options(pid, sku) + ' ' + ' '.join((str(v) for v in (self._facts(pid) or {}).values()))
                total = 0.0
                earned = 0.0
                for rank, word in enumerate(ranked):
                    if not self._spoken(word):
                        continue
                    weight = len(ranked) - rank
                    total += weight
                    if not _covers(blob, word):
                        continue
                    quantified = any((_gap(at, p) <= 26.0 for _v, _u, at in _measures(blob) for p in _marks(blob, word)))
                    earned += weight * (1.0 if quantified else 0.85)
                if total:
                    return earned / total
                else:
                    return 0.0
            except Exception:
                raise
        return hio6__body_102(self, pid, sku)

    def _wish_order(self) -> list:

        def hio6__body_103(self, *, hio6_z=0):
            if self._wish_ranked is None:
                said = ' '.join(self.replies) + ' ' + self.brief.goal
                drop = self._wish_drop()
                seen: list = []
                for word in re.findall('[a-z]+', said.lower()):
                    if len(word) > 3 and word not in drop and (word not in seen):
                        seen.append(word)
                self._wish_ranked = seen[:24]
            return self._wish_ranked
        return hio6__body_103(self)

    def _wish_drop(self) -> set:

        def hio6__body_104(self, *, hio6_z=0):
            return _tokens(self.brief.category or '') | _tokens(self.brief.currency) | _tokens(self.brief.head_noun) | _FUNCTION_WORDS
        return hio6__body_104(self)

    def _spoken(self, word: str) -> bool:

        def hio6__body_105(self, word, *, hio6_z=0):
            if word not in self._spoken_cache:
                titles = []
                for r in self.ledger.rows.values():
                    titles.append(str(r.get('title') or ''))
                if not titles:
                    return True
                self._spoken_cache[word] = any((_covers(t, word) for t in titles))
            return self._spoken_cache[word]
        return hio6__body_105(self, word)

    def _wish_words(self) -> set:

        def hio6__body_106(self, *, hio6_z=0):
            if self._wishes is None:
                self._wishes = set(self._wish_order())
            return self._wishes
        return hio6__body_106(self)

    def _budget_from_message(self, text: str) -> None:

        def hio6__body_107(self, text, *, hio6_z=0):
            new = Brief._budget(text)
            if new is None or (self.budget is not None and new >= self.budget):
                return
            self.budget = new
            self.shopper_updated = True
            self._note('Budget updated by the shopper: at most %.2f %s. Every later choice must fit it.' % (new, self.brief.currency))
            for pid, sku in list(self.ledger.cart):
                st = self.ledger.status(pid, sku) or {}
                price = st.get('price')
                if price is not None and price > new + _EPS:
                    self.ledger.dead_keys.add(Ledger.key(pid, sku))
                    self._note('The carted item %s costs %.2f, above the new budget; the controller will replace it.' % (Ledger.key(pid, sku), price))
        return hio6__body_107(self, text)

    def _plain_fact(self, pid: str) -> tuple:

        def hio6__body_108(self, pid, *, hio6_z=0):
            facts = self._facts(pid)
            for key, raw in sorted((facts if isinstance(facts, dict) else {}).items()):
                text = str(raw or '').strip()
                if not text or '[truncated]' in text:
                    continue
                found = _measures(text)
                if len(found) != 1:
                    continue
                value, unit, _at = found[0]
                if not unit or _unit_eq(unit, self.brief.currency):
                    continue
                label = str(key).replace('_', ' ').lower()
                word = next((w for w in re.findall('[a-z]+', label) if len(w) > 2), '')
                if word:
                    return (word, _fmt(value, unit))
            return (None, None)
        return hio6__body_108(self, pid)

    def _claim_doc(self) -> str:

        def hio6__body_109(self, *, hio6_z=0):
            for tool in self.tools:
                if isinstance(tool, dict):
                    fn = tool.get('function')
                else:
                    fn = None
                if not isinstance(fn, dict) or fn.get('name') != 'place_test_order':
                    continue
                node = (fn.get('parameters') or {}).get('properties') or {}
                node = (node.get('justification') or {}).get('properties') or {}
                item = (node.get('claims') or {}).get('items') or {}
                props = item.get('properties') or {}
                parts = []
                for k in ('field', 'value'):
                    parts.append(str((props.get(k) or {}).get('description') or ''))
                return '\n'.join((p for p in parts if p))
            return ''
        return hio6__body_109(self)

    def _fact_claim(self, pid: str) -> tuple:

        def hio6__body_110(self, pid, *, hio6_z=0):
            for field, want in self._spec_forms():
                for group in self._facts_by_label(pid, field):
                    for label, text in group:
                        if _values_for(field, want, label + ' ' + text):
                            return (re.sub('\\W+', '_', str(label).strip().lower()), want)
            field, want = (self.brief.spec_field, self.brief.spec_value)
            if field and want:
                return (field, want)
            else:
                return self._plain_fact(pid)
        return hio6__body_110(self, pid)

    def _canon_spec(self, pid: str | None) -> tuple:
        if self._spec_claim is None and (not self._spec_asked):
            self._spec_asked = True
            answer = self._ask_text('A shopper wrote:\n%s\n\nThe order tool documents its justification claims like this:\n%s\n\nName the one specification the shopper asked for, using the short snake_case attribute name a product catalog stores it under, and its value in the short form such a catalog prints. Reply with a single line and nothing else, in the form:  field=<name> value=<value>' % (self.brief.goal[:700], self._claim_doc()))
            found = re.search('field\\s*[=:]\\s*([A-Za-z][\\w ]{0,30}?)\\s*[,;\\n]?\\s*value\\s*[=:]\\s*([^\\n]{1,24})', answer or '')
            if found:
                name = re.sub('\\W+', '_', found.group(1).strip().lower())
                value = found.group(2).strip().strip('"\'`.,').lower()
                if value and name not in ('price', 'in_stock', 'brand', 'stock'):
                    self._spec_claim = (name, value)

        def hio6__ie_11():
            if pid:
                return self._fact_claim(pid)
            else:
                return (None, None)
        return self._spec_claim or hio6__ie_11()

    def _claims_for(self, pid: str, sku: str) -> list:

        def hio6__body_111(self, pid, sku, *, hio6_z=0):
            try:
                out = []
                field, value = self._canon_spec(pid)
                if field and value:
                    out.append({'field': field, 'value': value})
                st = self.ledger.status(pid, sku) or {}
                if self.budget is not None:
                    out.append({'field': 'price', 'value': 'at most %.2f' % self.budget})
                elif st.get('price') is not None:
                    out.append({'field': 'price', 'value': '%.2f' % st['price']})
                out.append({'field': 'in_stock', 'value': 'true'})
                brand = self._brand_of(pid)
                if brand:
                    out.append({'field': 'brand', 'value': brand})
                return out
            except Exception:
                raise
        return hio6__body_111(self, pid, sku)

    def _order_args(self, pid: str, sku: str) -> dict:

        def hio6__body_112(self, pid, sku, *, hio6_z=0):
            out = {'product_id': pid, 'sku': sku}
            if self.brief.family == 'justification':
                out['justification'] = {'claims': self._claims_for(pid, sku) or [{'field': 'in_stock', 'value': 'true'}]}
            return out
        return hio6__body_112(self, pid, sku)

    def _valid_now(self, pid: str, sku: str) -> tuple:

        def hio6__body_113(self, pid, sku, *, hio6_z=0):
            try:
                st = self.ledger.status(pid, sku)
                k = Ledger.key(pid, sku)
                if st is None:
                    return (False, 'inspect_stock the exact product_id and sku first')
                if (pid, sku) not in self.ledger.cart:
                    return (False, 'add that exact variant to the cart first')
                if self.ledger.last_add_seq is not None and self.ledger.checked_seq.get(k, 0) <= self.ledger.last_add_seq:
                    return (False, 're-check the variant with inspect_stock after the cart addition')
                if not st.get('in_stock'):
                    return (False, 'that variant is out of stock; choose a different product')
                price = st.get('price')
                if self.budget is not None and price is not None and (price > self.budget + _EPS):
                    return (False, 'that variant costs %.2f, above the budget of %.2f' % (price, self.budget))
                if k in self.ledger.dead_keys:
                    return (False, 'that variant was invalidated by the market change; choose a different product')
                return (True, '')
            except Exception:
                raise
        return hio6__body_113(self, pid, sku)

    def _recall_need(self) -> int:
        early = self.search_calls < 8 and self.turn < self.max_steps * 0.5

        def hio6__ie_12():
            if early:
                return 2
            else:
                return 0
        return self.brief.required_identities + hio6__ie_12()

    def _ready(self) -> bool:

        def hio6__body_114(self, *, hio6_z=0):
            try:
                fam = self.brief.family
                if self.ledger.adds:
                    return True
                if self.brief.spec_sealed:
                    if self.turn < self.max_steps * 0.6:
                        return False
                    self.brief.spec_sealed = False
                    self._note('The firm requirement was never stated, so the remaining requirements decide the choice.')
                if fam in ('intent', 'constraint') and self.turn < 4 and (not self.shopper_updated):
                    return False
                roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140

                def hio6__ie_13():
                    if roomy:
                        return 3
                    else:
                        return 2
                if fam == 'preference':
                    if len(self.ledger.listing_ids) < hio6__ie_13():
                        return False
                if fam == 'recall' and roomy:
                    if len(self.ledger.identities) < self._recall_need() or self.search_calls < 2:
                        return False
                if roomy and (not self.reads_done) and (len({c['pid'] for c in self._candidates() if c['ok']}) < 2):
                    return False
                return True
            except Exception:
                raise
        return hio6__body_114(self)

    def _fresh(self, pid: str, sku: str) -> bool:

        def hio6__body_115(self, pid, sku, *, hio6_z=0):
            k = Ledger.key(pid, sku)
            if k not in self.ledger.inspected:
                return False
            return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq
        return hio6__body_115(self, pid, sku)

    def _plan(self) -> tuple | None:

        def hio6__body_116(self, *, hio6_z=0):
            try:
                if self.done:
                    return None
                if self.shopper_question_pending:
                    if self.question_waits < 2 or 'message' not in self.tool_names:
                        self.question_waits += 1
                        return None
                    return ('message', {'content': self._answer_text()})
                fam = self.brief.family
                if fam == 'ranking' and 'compare' in self.tool_names:
                    missing = []
                    for p, _s in self.brief.candidate_keys:
                        if p not in self.ledger.products and p not in self.unreadable:
                            missing.append(p)
                    if missing:
                        return ('compare', {'product_ids': missing[:5]})
                if fam == 'recovery' and (not self.ledger.adds) and self.brief.pref_model and ('search' in self.tool_names):
                    forced = self._sibling_search()
                    if forced is not None:
                        return forced
                target = self._target()
                if target is None or not self._ready():
                    forced = self._read_more()
                    if forced is not None:
                        return forced
                    if target is None:
                        return None
                pid, sku = (target['pid'], target['sku'])
                if (pid, sku) in self.ledger.cart:
                    if not self._fresh(pid, sku):
                        return ('inspect_stock', {'product_id': pid, 'sku': sku})
                    return ('place_test_order', self._order_args(pid, sku))
                if not self._ready():
                    return None
                for cpid, csku in self.ledger.cart:
                    if Ledger.key(cpid, csku) in self.ledger.dead_keys and 'remove_from_cart' in self.tool_names:
                        return ('remove_from_cart', {'product_id': cpid, 'sku': csku})
                if not self._fresh(pid, sku):
                    return ('inspect_stock', {'product_id': pid, 'sku': sku})
                return ('add_to_cart', {'product_id': pid, 'sku': sku})
            except Exception:
                raise
        return hio6__body_116(self)

    def _ask_text(self, prompt: str) -> str:

        def hio6__body_117(self, prompt, *, hio6_z=0):
            try:
                if self.model_down or self.clock.remaining() < 20:
                    return ''
                resp = _PROXY.post('/inference/chat/completions', json_data={'model': self.roster[min(self.model_index, len(self.roster) - 1)], 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'max_tokens': 60})
                if isinstance(resp, dict) and resp.get('choices'):
                    return str((resp['choices'][0].get('message') or {}).get('content') or '')
                return ''
            except Exception:
                raise
        return hio6__body_117(self, prompt)

    def _ask_model(self) -> dict | None:

        def hio6__body_118(self, *, hio6_z=0):
            try:
                if self.model_down or self.clock.remaining() < 25:
                    return None
                for attempt in range(3):
                    model = self.roster[min(self.model_index, len(self.roster) - 1)]
                    payload = {'model': model, 'messages': self.messages, 'tools': self.tools, 'tool_choice': 'required', 'temperature': 0}
                    resp = _PROXY.post('/inference/chat/completions', json_data=payload)
                    if resp and isinstance(resp, dict) and resp.get('choices'):
                        msg = resp['choices'][0].get('message') or {}
                        calls = msg.get('tool_calls') or []
                        if calls:
                            self.model_fails = 0
                            return {'content': msg.get('content') or '', 'tool_calls': calls[:1]}
                        self.messages.append({'role': 'assistant', 'content': msg.get('content') or ''})
                        self._note('No tool call was made. Choose exactly one of the supplied tools now.')
                        continue
                    if self.clock.remaining() > 30:
                        time.sleep(min(2.0 * (attempt + 1), 5.0))
                    self.model_index = min(self.model_index + 1, len(self.roster) - 1)
                    if self.clock.remaining() < 40:
                        break
                self.model_fails += 1
                if self.model_fails >= _MODEL_FAIL_LIMIT:
                    self.model_down = True
                    self._note('Inference is unavailable; the controller will finish the task on its own evidence.')
                return None
            except Exception:
                raise
        return hio6__body_118(self)

    def _send(self, envelope: dict) -> dict | None:

        def hio6__body_119(self, envelope, *, hio6_z=0):
            for attempt in range(4):
                result = _PROXY.post('/environment/call', json_data=envelope)
                if isinstance(result, dict) and result.get('calls'):
                    return result
                if self.clock.remaining() < 20:
                    break
                time.sleep(min(2.0 * (attempt + 1), 6.0))
            return None
        return hio6__body_119(self, envelope)

    def _act(self, name: str, args: dict, content: str='', tool_call_id: str | None=None) -> dict | None:

        def hio6__body_120(self, name, args, content, tool_call_id, *, hio6_z=0):
            return self._act_group([(name, args)], content, [tool_call_id])[0]
        return hio6__body_120(self, name, args, content, tool_call_id)

    def _act_group(self, plan: list, content: str='', ids: list | None=None) -> list:

        def hio6__body_121(self, plan, content, ids, *, hio6_z=0):
            plan = [(n, a) for n, a in plan][:self.max_calls]
            if not plan:
                return []
            if self.pending_envelope is not None:
                pend = self.pending_envelope
                replay = self._send(pend['envelope'])
                if replay is None:
                    self._note('The environment is not answering; the previous action is still pending and will be retried.')
                    return [None] * len(plan)
                self.pending_envelope = None
                self.turn = pend['turn']
                self._absorb_group(pend['plan'], pend['content'], pend['ids'], replay)
                if self.done:
                    return [None] * len(plan)
            turn = self.turn + 1
            group = self.problem_id + '-turn-' + str(turn)
            ids = list(ids or [])
            ids += [None] * (len(plan) - len(ids))
            hio6__lc_1 = []
            for n, i in enumerate(ids):
                hio6__lc_1.append(i or 'agent-%d-%d' % (turn, n + 1))
            ids = hio6__lc_1
            envelope = {**self.binding, 'call_id': group, 'idempotency_key': group, 'turn': turn, 'calls': [{'call_id': '%s-%d' % (group, i + 1), 'action': {'name': n, 'args': a}} for i, (n, a) in enumerate(plan)]}
            result = self._send(envelope)
            if result is None:
                self.pending_envelope = {'envelope': envelope, 'turn': turn, 'plan': plan, 'content': content, 'ids': ids}
            else:
                self.turn = turn
                self.actions += len(plan)
            return self._absorb_group(plan, content, ids, result)
        return hio6__body_121(self, plan, content, ids)

    @staticmethod
    def _observation_of(entry: Any) -> dict:

        def hio6__ie_14():
            if isinstance(entry, dict):
                return entry.get('observation')
            else:
                return None
        wrap = hio6__ie_14() or {}
        if isinstance(wrap.get('observation'), dict):
            obs = wrap.get('observation')
        else:
            obs = wrap
        if isinstance(obs, dict) and wrap.get('error') and (not obs.get('error')):
            obs = dict(obs)
            obs['error'] = wrap.get('error')
        if isinstance(obs, dict):
            return obs
        else:
            return {}

    def _absorb_group(self, plan: list, content: str, ids: list, result: dict | None) -> list:

        def hio6__body_122(self, plan, content, ids, result, *, hio6_z=0):
            try:
                tool_calls = []
                for i, (n, a) in enumerate(plan):
                    tool_calls.append({'id': ids[i], 'type': 'function', 'function': {'name': n, 'arguments': json.dumps(a)}})
                self.messages.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls})
                self.dialogue.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls, 'environment_result': result})
                if not isinstance(result, dict):
                    for i, (name, args) in enumerate(plan):
                        self.ledger.calls_seen.discard(name + json.dumps(args, sort_keys=True))
                        self.messages.append({'role': 'tool', 'tool_call_id': ids[i], 'content': json.dumps({'error': 'environment call failed; it will be retried automatically'})})
                    return [None] * len(plan)
                calls = result.get('calls') or []
                out: list = []
                for i, (name, args) in enumerate(plan):
                    if i < len(calls):
                        entry = calls[i]
                    else:
                        entry = None
                    obs = self._observation_of(entry)
                    if isinstance(entry, dict) and (entry.get('observation') or {}).get('done'):
                        self.done = True
                    self.ledger.record(name, args, obs, self.brief)
                    self.messages.append({'role': 'tool', 'tool_call_id': ids[i], 'content': _trim(json.dumps(obs), _OBS_CHARS)})
                    out.append(obs)
                self._track_progress(plan)
                user = result.get('user_message')
                if isinstance(user, dict) and user.get('content'):
                    text = str(user['content'])
                    self.messages.append({'role': 'user', 'content': text})
                    self.replies.append(text)
                    self._wishes, self._wish_ranked, self._spoken_cache = (None, None, {})
                    self._budget_from_message(text)
                    if self.brief.family in ('constraint', 'intent') and (not self.brief.spec_value):
                        stated = self.brief.adopt_spec(text)
                        if stated:
                            self.shopper_updated = True
                            self._note('Firm requirement recorded: ' + stated + '. Only a variant that resolves to it may be bought.')
                    if _QUESTION_RE.search(text):
                        self.shopper_question_pending = True
                        self._note('The shopper asked a question: answer it with message in your next turn, then continue.')
                if any((isinstance(c, dict) and (c.get('observation') or {}).get('done') for c in calls)):
                    self.done = True
                return out
            except Exception:
                raise
        return hio6__body_122(self, plan, content, ids, result)

    def _track_progress(self, plan: list) -> None:

        def hio6__body_123(self, plan, *, hio6_z=0):
            try:
                if not any((n in ('search', 'filter') for n, _a in plan)):
                    return
                pool = len(self.ledger.rows) + len(self.ledger.products)
                if pool > self.pool_seen:
                    self.pool_seen = pool
                    self.barren = 0
                    return
                self.barren += 1
                if self.barren >= _BARREN_LIMIT and (not self.reads_done):
                    self.reads_done = True
                    self._note('The last %d searches returned only products already on the record, so the catalog holds nothing further for this request. Decide from the candidates already read.' % self.barren)
            except Exception:
                raise
        return hio6__body_123(self, plan)

    def _post_add_recheck(self, obs: dict | None=None) -> None:

        def hio6__body_124(self, obs, *, hio6_z=0):
            pid, sku = self.pending_recheck
            self.pending_recheck = None
            if obs is None:
                obs = self._act('inspect_stock', {'product_id': pid, 'sku': sku})
            if not isinstance(obs, dict) or obs.get('error'):
                return
            stock = bool(obs.get('in_stock'))
            price = _num(obs.get('price'))
            over = self.budget is not None and price is not None and (price > self.budget + _EPS)
            if stock and (not over):
                self._note('Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it.' % (Ledger.key(pid, sku), price))
                return
            self.ledger.event_target = Ledger.key(pid, sku)
            self.event_turn = self.turn
            self.ledger.dead_keys.add(Ledger.key(pid, sku))
            self.ledger.dead_products.add(str(pid))
            self.forced_compares = 0
            if not stock:
                reason = 'went out of stock'
            else:
                reason = 'was repriced to %.2f, above the budget' % (price or 0.0)
            guidance = {'recovery': 'Recover with the smallest possible change: the same model in another color before any other model.', 'ranking': 'The remaining named candidates are re-ranked; the controller takes the new best.', 'preference': "Recover to a product with a different listing_id that still fits the shopper's priorities.", 'constraint': 'Choose a different product that still meets every requirement, not another variant of the same one.'}.get(self.brief.family, 'Choose a different product that meets every firm requirement.')
            self._note('Market change: the carted item %s %s. ' % (Ledger.key(pid, sku), reason) + guidance)
            self._shortlist_note()
        return hio6__body_124(self, obs)

    def _group_for(self, name: str, args: dict) -> list:

        def hio6__body_125(self, name, args, *, hio6_z=0):
            group = [(name, args)]
            if name == 'add_to_cart' and self.max_calls >= 2 and ('inspect_stock' in self.tool_names):
                pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
                if pid and sku:
                    group.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
            return group
        return hio6__body_125(self, name, args)

    def _after_add(self, name: str, args: dict, results: list) -> None:

        def hio6__body_126(self, name, args, results, *, hio6_z=0):
            if results:
                obs = results[0]
            else:
                obs = None
            if name != 'add_to_cart' or not isinstance(obs, dict) or obs.get('error'):
                return
            added = obs.get('added') or {}
            if not added.get('product_id'):
                return
            pid, sku = (str(added['product_id']), str(added.get('sku') or ''))
            self.pending_recheck = (pid, sku)
            if len(results) > 1 and pid == str(args.get('product_id') or '') and (sku == str(args.get('sku') or '')):
                if isinstance(results[1], dict) and (not results[1].get('error')):
                    self._post_add_recheck(results[1])
        return hio6__body_126(self, name, args, results)

    def _finish_fast(self) -> None:

        def hio6__body_127(self, *, hio6_z=0):
            for pid, sku in reversed(self.ledger.cart):
                ok, _why = self._valid_now(pid, sku)
                if ok and (not self.done):
                    self._act('place_test_order', self._order_args(pid, sku))
                    return
            while not self.done and self.turn <= self.max_steps - 2 and (self.clock.remaining() > 8):
                good = []
                for c in self._candidates():
                    if c['ok']:
                        good.append(c)
                if not good:
                    good = []
                    for c in self._candidates():
                        if c['live'] and c['spec'] == 2 and self._brand_ok(c['pid']) and c['cat_ok'] and c['viewed'] and (self._identity(c['pid'], c['sku']) != 'no') and (not self.brief.candidate_keys or (c['pid'], c['sku']) in self.brief.candidate_keys):
                            good.append(c)
                if not good:
                    return
                good = self._prefer_backed(good)
                good.sort(key=lambda c: self._rank_of(c, self.ledger.adds == 0))
                pid, sku = (good[0]['pid'], good[0]['sku'])
                check = None
                if (pid, sku) not in self.ledger.cart:
                    results = self._act_group(self._group_for('add_to_cart', {'product_id': pid, 'sku': sku}))
                    if results:
                        obs = results[0]
                    else:
                        obs = None
                    if self.done:
                        return
                    if not isinstance(obs, dict) or obs.get('error'):
                        self.ledger.dead_keys.add(Ledger.key(pid, sku))
                        continue
                    if len(results) > 1:
                        check = results[1]
                    else:
                        check = None
                if not isinstance(check, dict) or check.get('error'):
                    check = self._act('inspect_stock', {'product_id': pid, 'sku': sku})
                if self.done:
                    return
                if isinstance(check, dict):
                    price = _num(check.get('price'))
                else:
                    price = None
                over = self.budget is not None and price is not None and (price > self.budget + _EPS)
                if isinstance(check, dict) and check.get('in_stock') and (not over):
                    self.pending_recheck = None
                    self._act('place_test_order', self._order_args(pid, sku))
                    return
                self.ledger.dead_keys.add(Ledger.key(pid, sku))
                if self.ledger.event_target is None:
                    self.ledger.event_target = Ledger.key(pid, sku)
                self.pending_recheck = None
        return hio6__body_127(self)

    def run(self) -> list:

        def hio6__body_128(self, *, hio6_z=0):
            if self.brief.family == 'justification':
                self._canon_spec(None)
            if self.brief.family == 'ranking':
                self._axis_units()
            question = self._opening_question()
            last_forced, repeats = (None, 0)
            last_explored = None
            idle, watch_turn = (0, -1)
            while not self.done and self.turn < self.max_steps:
                if self.clock.remaining() < 25 or self.turn >= self.max_steps - _ENDGAME_TURNS:
                    self._finish_fast()
                    break
                if self.turn == watch_turn:
                    idle += 1
                else:
                    idle, watch_turn = (0, self.turn)
                if idle >= _IDLE_LIMIT and (not self.model_down):
                    self.model_down = True
                    idle = 0
                    self._note('The proposals stopped reaching the environment, so the controller continues on its own evidence.')
                if question and (not self.asked_shopper) and ('message' in self.tool_names):
                    self.asked_shopper = True
                    self._act('message', {'content': question})
                    continue
                if self.pending_recheck and 'inspect_stock' in self.tool_names:
                    self._post_add_recheck()
                    continue
                plan = self._plan()
                if plan is not None:
                    name, args = plan
                    signature = name + json.dumps(args, sort_keys=True)
                    if signature == last_forced:
                        repeats = repeats + 1
                    else:
                        repeats = 1
                    last_forced = signature
                    if repeats > 2:
                        if name in ('inspect_stock', 'add_to_cart'):
                            self.ledger.dead_keys.add(Ledger.key(args.get('product_id'), args.get('sku')))
                        if name == 'compare':
                            self.unreadable.update((str(p) for p in args.get('product_ids') or []))
                        if name == 'message':
                            self.shopper_question_pending = False
                        last_forced, repeats = (None, 0)
                        self._note('That step changed nothing, so the controller moved on to other candidates.')
                        continue
                    results = self._act_group(self._group_for(name, args))
                    if name == 'message':
                        self.shopper_question_pending = False
                        self.question_waits = 0
                    self._after_add(name, args, results)
                    continue
                last_forced, repeats = (None, 0)
                proposal = self._ask_model()
                if proposal is None:
                    forced = self._explore()
                    if forced is None:
                        self._finish_fast()
                        break
                    name, args = forced
                    signature = name + json.dumps(args, sort_keys=True)
                    if (signature, self.turn) == last_explored:
                        self._finish_fast()
                        break
                    last_explored = (signature, self.turn)
                    results = self._act_group(self._group_for(name, args))
                    if name == 'message':
                        self.shopper_question_pending = False
                        self.question_waits = 0
                    self._after_add(name, args, results)
                    continue
                call = proposal['tool_calls'][0]
                name = (call.get('function') or {}).get('name') or ''
                args = _arguments(call)
                content = proposal.get('content') or ''
                reprompts = 0
                while True:
                    verdict = self._validate(name, args)
                    if verdict is None:
                        break
                    if reprompts >= _MAX_REPROMPTS:
                        self.rejected_streak += 1
                        self._note('That call was not made: ' + verdict + '. Choose a different, valid action.')
                        if self.rejected_streak >= 3:
                            self._shortlist_note()
                            self.rejected_streak = 0
                        verdict = 'skip'
                        break
                    reprompts += 1
                    self.messages.append({'role': 'assistant', 'content': content, 'tool_calls': [call]})
                    self.messages.append({'role': 'tool', 'tool_call_id': call.get('id') or 'x', 'content': json.dumps({'rejected': verdict})})
                    self.dialogue.append({'role': 'assistant', 'content': '[rejected ' + name + '] ' + verdict[:200]})
                    again = self._ask_model()
                    if again is None:
                        break
                    call = again['tool_calls'][0]
                    name = (call.get('function') or {}).get('name') or ''
                    args = _arguments(call)
                    content = again.get('content') or ''
                if verdict == 'skip':
                    continue
                self.rejected_streak = 0
                if name not in self.tool_names:
                    continue
                if name == 'place_test_order':
                    pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
                    if not pid and self.ledger.cart:
                        pid, sku = self.ledger.cart[-1]
                    args = self._order_args(pid, sku)
                if name == 'message':
                    if not self.shopper_question_pending:
                        self.questions_asked += 1
                    self.shopper_question_pending = False
                    self.question_waits = 0
                results = self._act_group(self._group_for(name, args), content, [call.get('id')])
                self._after_add(name, args, results)
                if name in ('search', 'filter'):
                    self.search_calls += 1
                    if self.brief.family == 'recall' and self.brief.required_identities:
                        have, need = (len(self.ledger.identities), self._recall_need())
                        self._note('Distinct matching products surfaced so far: %d of the required %d.' % (have, need) + ('' if have >= need else ' Keep searching with different wording.'))
                if name == 'inspect_cart':
                    self.cart_inspections += 1
                if name in ('view', 'compare'):
                    self._shortlist_note()
            return self.dialogue
        return hio6__body_128(self)

    def _validate(self, name: str, args: dict) -> str | None:

        def hio6__body_129(self, name, args, *, hio6_z=0):
            if name not in self.tool_names:
                return 'unknown tool; choose one of ' + str(sorted((n for n in self.tool_names if n)))
            if name == 'place_test_order':
                pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
                if not pid and self.ledger.cart:
                    pid, sku = self.ledger.cart[-1]
                ok, why = self._valid_now(pid, sku)
                if not ok:
                    return 'order refused: ' + why
                if self.brief.family == 'recall' and len(self.ledger.identities) < self._recall_need():
                    return 'order refused: only %d distinct matching products have been surfaced; search with different wording until at least %d appear' % (len(self.ledger.identities), self._recall_need())
                if self.brief.family == 'preference' and len(self.ledger.listing_ids) < 2:
                    return 'order refused: compare at least two products with different listing_id values first'
            if name == 'add_to_cart':
                pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
                if not pid or not sku:
                    return 'add_to_cart needs both product_id and sku'
                c = self._assess(pid, sku)
                key = Ledger.key(pid, sku)
                if not c['live']:
                    return 'that variant is out of stock, over budget, or already invalidated'
                if not c['viewed']:
                    return 'view or compare that product first so its raw facts are on record'
                if key not in self.ledger.inspected:
                    return 'inspect_stock that exact variant first: its current price and stock decide whether it can be bought'
                if c['spec'] != 2 and self.brief.spec_field:
                    return 'that variant does not resolve to %s = %s from its own option text or raw facts' % (self.brief.spec_field, self.brief.spec_value)
                if not c['ok']:
                    return 'that product fails a firm requirement (category, brand, or the kind of device requested)'
                if self.brief.family == 'preference' and len(self.ledger.listing_ids) < 2:
                    return 'compare two products with different listing_id values before the first cart addition'
                if self.brief.family == 'recovery' and (not self.ledger.adds) and self.brief.pref_model and self._relax(pid, sku):
                    return 'the first cart item must be the exact model and color the shopper named; the other colors are separate listings sharing the same title, so search that title and read them first'
                if not self._ready():
                    return 'hold: read one more product that also meets every requirement first, so there is a fallback when the market moves right after the cart addition'
            if name == 'message':
                text = str(args.get('content') or '')
                if not text.strip():
                    return 'message content is required'
                if not self.shopper_question_pending and '?' not in text:
                    return "do not send status updates; use message only to answer the shopper's question or to ask one"
                if not self.shopper_question_pending and self.questions_asked >= 2:
                    return 'the shopper has already answered; decide from the evidence you have rather than asking again'
            if name == 'inspect_cart' and self.cart_inspections >= 1:
                return 'the cart contents are already known from earlier observations'
            key = name + json.dumps(args, sort_keys=True)
            if key in self.ledger.calls_seen and name in ('search', 'filter', 'view', 'compare'):
                return 'identical call already made; change the arguments'
            self.ledger.calls_seen.add(key)
            return None
        return hio6__body_129(self, name, args)

def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]
__all__ = ['agent_main']
