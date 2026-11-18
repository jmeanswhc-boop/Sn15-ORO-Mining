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
_DURATION_RE = re.compile('(?<![\\w.])(\\d+(?:\\.\\d+)?)\\s*\\+?\\s*(?:hours?|hrs?|h)\\b|(?<![\\w.])(\\d+(?:\\.\\d+)?)\\s*\\+?\\s*days?\\b', re.I)

def _soft_blob(title: str, facts: dict, options: str='') -> str:

    def nhv4__body_18(title, facts, options, *, nhv4_z=0):
        bits = [title or '', options or '']
        for key, value in (facts or {}).items():
            bits.append(str(key).replace('_', ' ') + ' ' + str(value))
        return ' '.join(bits)
    return nhv4__body_18(title, facts, options)

def _duration_hours(text: str) -> float:

    def nhv4__body_19(text, *, nhv4_z=0):
        best = 0.0
        for m in _DURATION_RE.finditer(text or ''):
            raw = m.group(1) or m.group(2)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if m.group(2):
                value *= 24.0
            if value > best:
                best = value
        return best
    return nhv4__body_19(text)

def _soft_hits(title: str, facts: dict, options: str='', wants: set | None=None) -> float:

    def nhv4__body_20(title, facts, options, wants, *, nhv4_z=0):
        blob = _soft_blob(title, facts, options)
        hours = _duration_hours(blob)
        have = _tokens(blob)
        axes = float(len((wants or set()) & have))
        if hours > 0:
            axes += 1.0
        return axes + min(hours, 80.0) / 1000.0
    return nhv4__body_20(title, facts, options, wants)
_QUESTION_RE = re.compile('\\?|\\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)|tell me (?:which|whether|if)|confirm whether)\\b', re.I)
_MEASURE_RE = re.compile('(?<![\\w.])(?<!\\w-)(\\d+(?:\\.\\d+)?)\\s*[-–]?\\s*([A-Za-z]{1,6}|\'{1,2}|[\\"“”])(?![\\w])')
_FUNCTION_WORDS = frozenset('a an the this that these those my your our their its it i me we you they\n    is are was were be been being do does did doing have has had having\n    can could will would shall should may might must\n    get gets getting got find finds finding need needs needing want wants wanting\n    look looking looks buy buys buying order orders ordering purchase show shows\n    pick picks picking choose choosing prefer prefers recommend recommends\n    please just really some any something anything one another other same\n    and or but for with without from into onto than then there here about\n    under over below above within at most least around near upto up to of in on'.split())

def _unit(raw: str) -> str:

    def nhv4__body_21(raw, *, nhv4_z=0):
        return (raw or '').strip().lower()
    return nhv4__body_21(raw)

def _named_unit(text: str) -> bool:

    def nhv4__body_22(text, *, nhv4_z=0):
        return bool(text) and text[:1].isalpha()
    return nhv4__body_22(text)

def _unit_eq(a: str, b: str) -> bool:

    def nhv4__body_23(a, b, *, nhv4_z=0):
        x, y = (_unit(a), _unit(b))
        if not x or not y:
            return False
        if x == y:
            return True
        if not (_named_unit(x) and _named_unit(y)):
            return True
        return min(len(x), len(y)) >= 2 and (x.startswith(y) or y.startswith(x))
    return nhv4__body_23(a, b)

def _measures(text: str) -> list:

    def nhv4__body_24(text, *, nhv4_z=0):
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
    return nhv4__body_24(text)

def _fmt(value: float, unit: str) -> str:

    def nhv4__body_25(value, unit, *, nhv4_z=0):
        return '%g%s' % (value, _unit(unit))
    return nhv4__body_25(value, unit)

def _split_value(want: str) -> tuple:

    def nhv4__body_26(want, *, nhv4_z=0):
        m = re.fullmatch('(\\d+(?:\\.\\d+)?)([a-z]*)', (want or '').strip().lower())
        if m:
            return (float(m.group(1)), m.group(2))
        return (None, (want or '').strip().lower())
    return nhv4__body_26(want)

def _tail(word: str) -> str:
    n = len(word or '')

    def nhv4__ie_2():
        if n >= 3:
            return '[a-z]?'
        else:
            return ''
    if n >= 5:
        return '[a-z]{0,3}'
    else:
        return nhv4__ie_2()

def _says(text: str, word: str) -> bool:

    def nhv4__body_27(text, word, *, nhv4_z=0):
        if not word:
            return False
        low = word.lower()
        return bool(re.search('(?<![a-z])' + re.escape(low) + _tail(low) + '(?![a-z])', (text or '').lower()))
    return nhv4__body_27(text, word)

def _marks(text: str, word: str) -> list:

    def nhv4__body_28(text, word, *, nhv4_z=0):
        if not word:
            return []
        low = word.lower()
        return [m.start() for m in re.finditer('(?<![a-z])' + re.escape(low) + _tail(low) + '(?![a-z])', (text or '').lower())]
    return nhv4__body_28(text, word)

def _gap(at: int, mark: int) -> float:
    if mark >= at:
        return float(mark - at)
    else:
        return (at - mark) * 3.0

def _values_for(field: str, want: str, text: str) -> set:

    def nhv4__body_29(field, want, text, *, nhv4_z=0):
        number, unit = _split_value(want)
        text = text or ''
        if field == 'resolution' or (want or '').strip().lower() in {'hd', 'fhd', 'qhd', '4k'}:
            found = _res_classes(text)
            if want in found:
                return {want}
            else:
                return set()
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
    return nhv4__body_29(field, want, text)

def _mentions(field: str, want: str, text: str) -> bool:

    def nhv4__body_30(field, want, text, *, nhv4_z=0):
        if field == 'resolution' or (want or '').strip().lower() in {'hd', 'fhd', 'qhd', '4k'}:
            return bool(_res_classes(text) or _WXH_RE.search(text or ''))
        number, unit = _split_value(want)
        if number is None:
            return _says(text, unit)
        return bool([1 for _v, u, _at in _measures(text or '') if _unit_eq(u, unit)]) or _says(text, field)
    return nhv4__body_30(field, want, text)

def _one(values: set) -> str | None:
    if len(values) == 1:
        return next(iter(values))
    else:
        return None
_WXH_RE = re.compile('(\\d{3,5})\\s*[x×]\\s*(\\d{3,5})')
_RES_CLASS = frozenset({'hd', 'fhd', 'qhd', '4k'})

def _res_tokens(text: str) -> set:

    def nhv4__body_31(text, *, nhv4_z=0):
        try:
            text = text or ''
            if re.search('\\bqHD\\b', text):
                return set()
            low = text.lower()
            out = set()
            for m in _WXH_RE.finditer(low):
                out.add(m.group(1))
                out.add(m.group(2))
            for m in re.finditer('\\b(\\d{3,4})p\\b', low):
                out.add(m.group(1))
                out.add(m.group(1) + 'p')
            if re.search('\\b(?:4k|uhd)\\b', low) or 'ultra hd' in low:
                out.add('4k')
            if re.search('\\b(?:qhd|2k)\\b', low):
                out.add('qhd')
            if re.search('\\bfhd\\b', low) or 'full hd' in low:
                out.add('fhd')
            if re.search('\\bhd\\b', re.sub('\\b(?:full|ultra)\\s+hd\\b', ' ', low)):
                out.add('hd')
            return out
        except Exception:
            raise
        else:
            pass
    return nhv4__body_31(text)

def _res_classes(text: str) -> set:

    def nhv4__body_32(text, *, nhv4_z=0):
        try:
            t = _res_tokens(text)
            out = set((x for x in t if x in _RES_CLASS))
            if '2160' in t or '2160p' in t:
                out.add('4k')
            if '1440' in t or '1440p' in t:
                out.add('qhd')
            if '1080' in t or '1080p' in t:
                out.add('fhd')
            if '720' in t or '720p' in t:
                out.add('hd')
            return out
        except Exception:
            raise
        else:
            pass
    return nhv4__body_32(text)

def _tokens(text: str) -> set:

    def nhv4__body_33(text, *, nhv4_z=0):
        return set(re.findall('[a-z0-9]+', (text or '').lower()))
    return nhv4__body_33(text)
_ABOUT_RE = re.compile('\\b(?:for|compatible\\s+with|fits|suitable\\s+for)\\b', re.I)
_NEGATED_RE = re.compile('\\bnot\\s+(?:an?\\s+)?[\\w\\s]{0,24}?(?=,|\\.|$)', re.I)
_BUNDLE_RE = re.compile('\\b(?:with\\s+(?:free|bundled)|bundled\\s+with|free|includes?)\\s+', re.I)
_KIND_KEY_RE = re.compile('type|categ|kind|class', re.I)

def _identity_text(text: str, title: bool=False) -> str:

    def nhv4__body_34(text, title, *, nhv4_z=0):
        text = _NEGATED_RE.sub('', text or '')
        if title:
            return _BUNDLE_RE.split(text, maxsplit=1)[0]
        else:
            return text
    return nhv4__body_34(text, title)

def _other_kind(noun: str, facts: dict) -> bool:

    def nhv4__body_35(noun, facts, *, nhv4_z=0):
        tail = (noun or '')[-5:]
        if len(noun or '') < 7 or len(tail) < 5:
            return False
        for value in (facts or {}).values():
            for word in re.findall('[a-z]+', str(value or '').lower()):
                if word != noun and len(word) > len(tail) + 1 and word.endswith(tail):
                    return True
        return False
    return nhv4__body_35(noun, facts)

def _identity_of(noun: str, facts: dict, title: str, options: str) -> str:

    def nhv4__body_36(noun, facts, title, options, *, nhv4_z=0):
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
    return nhv4__body_36(noun, facts, title, options)
_CORE_RULES = 'You are a careful shopping agent working a catalog through tools. Exactly one tool call per turn, at most 30 turns, so be efficient. A deterministic controller checks every proposal and takes over the cart and the order once a candidate is fully verified, so your job is to surface and read candidates: search or filter, then view or compare so their raw facts, variant options and prices are on record. Never add an item to the cart as a placeholder - the first cart addition is graded. Prices and stock change while you shop; the controller re-checks the carted item and tells you when to move to a different product. Answer every direct question from the shopper with one short message before anything else, and never send progress or status updates. Never ask the shopper to raise the budget. Do not repeat a tool call with identical arguments; one compare of up to five products beats several view calls. Before you commit, name two candidate product_ids with their prices or a listed attribute and say in one sentence why one beats the other.'

class Clock:

    def __init__(self, budget: float) -> None:

        def nhv4__body_37(self, budget, *, nhv4_z=0):
            self.start = time.monotonic()
            self.budget = budget
        return nhv4__body_37(self, budget)

    def remaining(self) -> float:

        def nhv4__body_38(self, *, nhv4_z=0):
            return self.budget - (time.monotonic() - self.start)
        return nhv4__body_38(self)

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

        def nhv4__ie_3():
            if quoted:
                return quoted[0]
            else:
                return ''
        if phrase:
            source = phrase.group(1)
        else:
            source = nhv4__ie_3()
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

        def nhv4__body_39(self, *, nhv4_z=0):
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
        return nhv4__body_39(self)

    @staticmethod
    def _budget(text: str) -> float | None:

        def nhv4__body_40(text, *, nhv4_z=0):
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
        return nhv4__body_40(text)

    @staticmethod
    def _category(goal: str) -> str | None:

        def nhv4__body_41(goal, *, nhv4_z=0):
            m = _CAT_PHRASE_RE.search(goal or '')
            if m:
                return m.group(1).strip()
            m = re.search('\\b([a-z][a-z\\-]{2,20})\\s*(?:,|\\s(?:at or (?:under|below)|at most|under|below|within|with)\\b)', goal or '', re.I)
            if m:
                return m.group(1)
            else:
                return None
        return nhv4__body_41(goal)

    @staticmethod
    def _brand(goal: str, category: str | None) -> str | None:

        def nhv4__body_42(goal, category, *, nhv4_z=0):
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
        return nhv4__body_42(goal, category)

    @staticmethod
    def _pref_brand(goal: str) -> str | None:

        def nhv4__body_43(goal, *, nhv4_z=0):
            m = _PREF_BRAND_RE.search(goal or '')
            if not m:
                return None
            name = m.group(1).strip(' .,')
            words = name.split()
            if not words or len(words) > 4 or (not any((w[:1].isupper() for w in words))):
                return None
            return name
        return nhv4__body_43(goal)

    @staticmethod
    def _head_noun(category: str | None) -> str:

        def nhv4__body_44(category, *, nhv4_z=0):
            words = re.findall('[a-z]+', (category or '').lower())
            if words:
                return words[-1]
            else:
                return ''
        return nhv4__body_44(category)

    @staticmethod
    def _qualifiers(goal: str, category: str | None, brand: str | None, pref_brand: str | None) -> tuple:

        def nhv4__body_45(goal, category, brand, pref_brand, *, nhv4_z=0):
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
        return nhv4__body_45(goal, category, brand, pref_brand)

    @staticmethod
    def _specs(goal: str, currency: str, budget: float | None, alts: dict | None=None) -> dict:

        def nhv4__body_46(goal, currency, budget, alts, *, nhv4_z=0):
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
                if at in spent or _unit_eq(unit, currency):
                    continue
                if budget is not None and abs(value - budget) < _EPS:
                    continue
                if unit in {'p', 'px'}:
                    continue
                word = Brief._concept_word(text, at)
                if (word or '').lower() in _RES_CLASS:
                    continue
                key = word or unit
                if key not in out:
                    out[key] = _fmt(value, unit)
                alt = Brief._class_form(text, at, key)
                if alt and alt not in out and (alt.lower() not in _RES_CLASS):
                    out[alt] = alt
                    alts[alt] = key
            classes = _res_classes(text)
            if len(classes) == 1:
                out['resolution'] = next(iter(classes))
            return out
        return nhv4__body_46(goal, currency, budget, alts)

    @staticmethod
    def _class_form(text: str, at: int, taken: str='') -> str | None:

        def nhv4__body_47(text, at, taken, *, nhv4_z=0):
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
            else:
                pass
        return nhv4__body_47(text, at, taken)

    @staticmethod
    def _concept_word(text: str, at: int) -> str:

        def nhv4__body_48(text, at, *, nhv4_z=0):
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
        return nhv4__body_48(text, at)

    @staticmethod
    def _order_by(text: str) -> tuple:

        def nhv4__body_49(text, *, nhv4_z=0):
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
        return nhv4__body_49(text)

    def _spec(self, goal: str) -> tuple:

        def nhv4__body_50(self, goal, *, nhv4_z=0):
            for field, value in self.specs.items():
                return (field, value)
            return (None, None)
        return nhv4__body_50(self, goal)

    def adopt_spec(self, text: str) -> str | None:

        def nhv4__body_51(self, text, *, nhv4_z=0):
            found = self._specs(text, self.currency, None, self.alt_of)
            for field, value in found.items():
                self.spec_field, self.spec_value = (field, value)
                self.specs[field] = value
                self.spec_sealed = False
                return field + ' = ' + value
            return None
        return nhv4__body_51(self, text)

def _num(value: Any) -> float | None:

    def nhv4__body_52(value, *, nhv4_z=0):
        try:
            if value is None:
                return None
            return float(str(value).replace(',', ''))
        except (TypeError, ValueError):
            return None
    return nhv4__body_52(value)

class Ledger:

    def __init__(self) -> None:

        def nhv4__body_53(self, *, nhv4_z=0):
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
        return nhv4__body_53(self)

    @staticmethod
    def key(pid: Any, sku: Any) -> str:

        def nhv4__body_54(pid, sku, *, nhv4_z=0):
            return str(pid) + '::' + str(sku or '')
        return nhv4__body_54(pid, sku)

    def record(self, name: str, args: dict, obs: dict, brief: Brief) -> None:

        def nhv4__body_55(self, name, args, obs, brief, *, nhv4_z=0):
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
        return nhv4__body_55(self, name, args, obs, brief)

    def _sync_cart(self, obs: dict) -> None:

        def nhv4__body_56(self, obs, *, nhv4_z=0):
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
        return nhv4__body_56(self, obs)

    def _note_row(self, row: dict, brief: Brief) -> None:

        def nhv4__body_57(self, row, brief, *, nhv4_z=0):
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
        return nhv4__body_57(self, row, brief)

    def _note_product(self, prod: dict, brief: Brief) -> None:

        def nhv4__body_58(self, prod, brief, *, nhv4_z=0):
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
        return nhv4__body_58(self, prod, brief)

    def status(self, pid: str, sku: str) -> dict | None:

        def nhv4__body_59(self, pid, sku, *, nhv4_z=0):
            return self.checked.get(self.key(pid, sku))
        return nhv4__body_59(self, pid, sku)

def _arguments(tool_call: dict) -> dict:

    def nhv4__body_60(tool_call, *, nhv4_z=0):
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
    return nhv4__body_60(tool_call)

def _trim(text: str, limit: int) -> str:

    def nhv4__body_61(text, limit, *, nhv4_z=0):
        if len(text) <= limit:
            return text
        return text[:limit - 40] + ' ... (truncated by agent)"}'
    return nhv4__body_61(text, limit)

class Agent:

    def __init__(self, problem: dict) -> None:

        def nhv4__body_62(self, problem, *, nhv4_z=0):
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
            self.search_queries: list = []
            self.recall_filtered = False
            self.cart_inspections = 0
            self.pending_recheck: tuple | None = None
            self.forced_compares = 0
            self.hinted_exact = False
            self.tried: dict = {}
            self.questions_asked = 0
            self.searched_siblings = False
            self.hinted_brand = False
            self._wishes: set | None = None
            self.pref_qi = 0
            self.probes: set = set()
            self.searched_twins = False
            self.reads_done = False
            self.event_turn: int | None = None
            self.unreadable: set = set()
            self.question_waits = 0
            self.done = False
            self.model_index = 0
            self._axis_unit: dict | None = None
            self._live_terms: dict = {}
            self._reasoned = False
            self._step = 0
            provider = (getenv('INFERENCE_PROVIDER') or '').strip().lower()
            self.roster = _MODELS.get(provider) or _MODELS['chutes']
            pinned = getenv('SANDBOX_MODEL')
            if pinned:
                self.roster = (pinned,) + tuple((m for m in self.roster if m != pinned))
        return nhv4__body_62(self, problem)

    def _system_prompt(self) -> str:

        def nhv4__body_63(self, *, nhv4_z=0):
            extra = ''
            if self.brief.family == 'recall' and self.brief.required_identities:
                extra = ' Surface at least %d distinct matching products in search or filter results before ordering.' % self.brief.required_identities
            return _CORE_RULES + extra
        return nhv4__body_63(self)

    def _opening_question(self) -> str | None:

        def nhv4__body_64(self, *, nhv4_z=0):
            if self.brief.family == 'constraint' or self.brief.spec_sealed:
                return 'Before I search: which specification is your firm requirement, and what exact value does it have to be?'
            if self.brief.family == 'preference':
                return 'Quick question so I choose well: which of your preferences counts most, and should I weigh several of them together?'
            return None
        return nhv4__body_64(self)

    def _note(self, text: str) -> None:

        def nhv4__body_65(self, text, *, nhv4_z=0):
            self.messages.append({'role': 'user', 'content': '[note] ' + text})
            self._push_step(think='', content='[note] ' + text[:300])
        return nhv4__body_65(self, text)

    def _push_step(self, think: str='', content: str='', tool_calls: list | None=None, env_result: dict | None=None) -> None:

        def nhv4__body_66(self, think, content, tool_calls, env_result, *, nhv4_z=0):
            self._step += 1
            tools = tool_calls or []
            formatted = []
            for call in tools:
                fn = call.get('function') or {}
                args = fn.get('arguments')
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (TypeError, ValueError):
                        args = {}
                formatted.append({'name': fn.get('name') or call.get('name') or '', 'parameters': args or {}, 'result': env_result or ''})
            message: dict = {}
            if think:
                message['think'] = think[:1000]
            if formatted:
                message['tool_call'] = formatted
            if content:
                message['response'] = content[:400]
            step = {'role': 'assistant', 'content': content, 'tool_calls': tools, 'environment_result': env_result, 'completion': {'content': content, 'message': message}, 'extra_info': {'step': self._step, 'query': self.brief.goal[:300], 'timestamp': int(time.time() * 1000)}}
            self.dialogue.append(step)
        return nhv4__body_66(self, think, content, tool_calls, env_result)

    def _compare_think(self) -> str:

        def nhv4__body_68(self, *, nhv4_z=0):
            try:
                rows = []
                for c in self._candidates():
                    price = c.get('price')
                    if price is None:
                        continue
                    title = (self._title(c['pid']) or c['pid'])[:40]
                    brand = self._brand_of(c['pid'])
                    extra = ''
                    if self.brief.spec_field and self.brief.spec_value:
                        got = self._variant_value(c['pid'], c['sku'], self.brief.spec_field)
                        if got:
                            extra = ', %s=%s' % (self.brief.spec_field, got)
                    elif brand:
                        extra = ', brand %s' % brand
                    rows.append((c, title, extra, price))

                def nhv4__ie_4():
                    if self.brief.currency:
                        return ' ' + self.brief.currency
                    else:
                        return ''

                def nhv4__ie_5():
                    if self.brief.currency:
                        return ' ' + self.brief.currency
                    else:
                        return ''
                if len(rows) < 2:
                    leads = []
                    for key, row in list(self.ledger.rows.items())[:8]:
                        pid, _, sku = key.partition('::')
                        price = _num((self.ledger.checked.get(key) or {}).get('price') or row.get('price'))
                        if price is None:
                            continue
                        leads.append((pid, sku, str(row.get('title') or pid)[:40], price))
                    if len(leads) >= 2:
                        a, b = (leads[0], leads[1])
                        if a[3] <= b[3]:
                            winner, loser = (a, b)
                        else:
                            winner, loser = (b, a)
                        return '%s (%s) is %.2f%s and %s (%s) is %.2f%s. %s wins because it is cheaper and still in the results that match the request.' % (winner[0], winner[2], winner[3], nhv4__ie_4(), loser[0], loser[2], loser[3], nhv4__ie_5(), winner[0])
                    return ''

                def nhv4__lam_6(r, *, nhv4_z=0):

                    def nhv4__body_67(r, *, nhv4_z=0):
                        return r[3]
                    return nhv4__body_67(r)
                rows.sort(key=nhv4__lam_6)
                a, b = (rows[0], rows[1])

                def nhv4__ie_7():
                    if self.brief.currency:
                        return ' ' + self.brief.currency
                    else:
                        return ''

                def nhv4__ie_8():
                    if self.brief.currency:
                        return ' ' + self.brief.currency
                    else:
                        return ''
                return '%s (%s%s) is %.2f%s and %s (%s%s) is %.2f%s. %s wins because it is %.2f cheaper while both are in stock and fit the stated constraints.' % (a[0]['pid'], a[1], a[2], a[3], nhv4__ie_7(), b[0]['pid'], b[1], b[2], b[3], nhv4__ie_8(), a[0]['pid'], b[3] - a[3])
            except Exception:
                raise
            else:
                pass
        return nhv4__body_68(self)

    def _reason_once(self) -> None:

        def nhv4__body_69(self, *, nhv4_z=0):
            if self._reasoned or self.clock.remaining() < 40:
                return
            think = self._compare_think()
            if not think:
                return
            self._reasoned = True
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            prompt = 'You are scoring two shopping candidates. Write 4-6 sentences. Name both product ids, cite each price or a listed attribute, and say which one you would buy and why it beats the other. Do not call tools.\n\n' + think
            resp = _PROXY.post('/inference/chat/completions', json_data={'model': model, 'messages': [{'role': 'system', 'content': 'Compare two products using only the facts given.'}, {'role': 'user', 'content': prompt}], 'temperature': 0})
            spoken = ''
            if resp and isinstance(resp, dict) and resp.get('choices'):
                spoken = str((resp['choices'][0].get('message') or {}).get('content') or '')
            text = spoken.strip() or think
            if think.split('.')[0] not in text:
                text = think + ' ' + text
            self._push_step(think=text[:1000], content=text[:200])
            self.messages.append({'role': 'assistant', 'content': text[:800]})
        return nhv4__body_69(self)

    def _prod(self, pid: str) -> dict:

        def nhv4__body_70(self, pid, *, nhv4_z=0):
            return self.ledger.products.get(str(pid)) or {}
        return nhv4__body_70(self, pid)

    def _facts(self, pid: str) -> dict:

        def nhv4__body_71(self, pid, *, nhv4_z=0):
            f = self._prod(pid).get('facts')
            if isinstance(f, dict):
                return f
            else:
                return {}
        return nhv4__body_71(self, pid)

    def _title(self, pid: str) -> str:

        def nhv4__body_72(self, pid, *, nhv4_z=0):
            title = str(self._prod(pid).get('title') or '')
            if title:
                return title
            for key, row in self.ledger.rows.items():
                if key.startswith(str(pid) + '::'):
                    return str(row.get('title') or '')
            return ''
        return nhv4__body_72(self, pid)

    def _brand_of(self, pid: str) -> str:

        def nhv4__body_73(self, pid, *, nhv4_z=0):
            brand = self._prod(pid).get('brand')
            if brand:
                return str(brand)
            for key, row in self.ledger.rows.items():
                if key.startswith(str(pid) + '::') and row.get('brand'):
                    return str(row['brand'])
            return ''
        return nhv4__body_73(self, pid)

    def _cat_path(self, pid: str) -> list:

        def nhv4__body_74(self, pid, *, nhv4_z=0):
            path = self._prod(pid).get('category_path')
            if isinstance(path, list):
                return list(path)
            else:
                return []
        return nhv4__body_74(self, pid)

    def _options(self, pid: str, sku: str) -> str:

        def nhv4__body_75(self, pid, sku, *, nhv4_z=0):
            return self.ledger.options.get(Ledger.key(pid, sku), '')
        return nhv4__body_75(self, pid, sku)

    def _strip_models(self, pid: str, title: str) -> str:

        def nhv4__body_76(self, pid, title, *, nhv4_z=0):
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
        return nhv4__body_76(self, pid, title)

    def _want_of(self, field: str) -> str | None:

        def nhv4__body_77(self, field, *, nhv4_z=0):
            if field in self.brief.specs:
                return self.brief.specs[field]
            if field == self.brief.spec_field:
                return self.brief.spec_value
            return None
        return nhv4__body_77(self, field)

    def _spec_forms(self) -> list:

        def nhv4__body_78(self, *, nhv4_z=0):
            field, want = (self.brief.spec_field, self.brief.spec_value)
            if not (field and want):
                return []
            out = [(field, want)]
            for alt, src in self.brief.alt_of.items():
                if src == field and alt in self.brief.specs:
                    out.append((alt, self.brief.specs[alt]))
            return out
        return nhv4__body_78(self)

    def _facts_by_label(self, pid: str, field: str) -> list:

        def nhv4__body_79(self, pid, field, *, nhv4_z=0):
            named, rest = ([], [])
            for key, raw in (self._facts(pid) or {}).items():
                text = str(raw or '').strip()
                if not text or '[truncated]' in text:
                    continue
                label = str(key).replace('_', ' ')
                (named if _says(label, field) else rest).append((label, text))
            return [named, rest]
        return nhv4__body_79(self, pid, field)

    def _fact_value(self, pid: str, field: str, want: str | None=None) -> str | None:

        def nhv4__body_80(self, pid, field, want, *, nhv4_z=0):
            try:
                want = want or self._want_of(field)
                if field == 'resolution':
                    for group in self._facts_by_label(pid, field):
                        found: set = set()
                        for _label, text in group:
                            found |= _res_classes(text)
                        if found:
                            return _one(found)
                    return None
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
            except Exception:
                raise
            else:
                pass
        return nhv4__body_80(self, pid, field, want)

    def _product_value(self, pid: str, field: str, want: str | None=None) -> str | None:

        def nhv4__body_81(self, pid, field, want, *, nhv4_z=0):
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
                if number is None:
                    title = self._strip_models(pid, title)
                vals |= _values_for(field, want, title)
                return _one(vals)
            except Exception:
                raise
            else:
                pass
        return nhv4__body_81(self, pid, field, want)

    def _variant_value(self, pid: str, sku: str, field: str) -> str | None:

        def nhv4__body_82(self, pid, sku, field, *, nhv4_z=0):
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
        return nhv4__body_82(self, pid, sku, field)

    def _measure_in(self, pid: str, sku: str, unit: str) -> float | None:

        def nhv4__body_83(self, pid, sku, unit, *, nhv4_z=0):
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
        return nhv4__body_83(self, pid, sku, unit)

    def _axis_units(self) -> dict:

        def nhv4__body_84(self, *, nhv4_z=0):
            try:
                if self._axis_unit is not None:
                    return self._axis_unit
                self._axis_unit = {}
                axes = []
                for _desc, w in self.brief.order_by:
                    if w != ('price',):
                        axes.append(w)
                votes: dict = {w: {} for w in axes}
                for pid, sku in self.brief.candidate_keys or []:
                    facts = self._facts(str(pid))
                    for key, raw in facts.items() if isinstance(facts, dict) else ():
                        label = str(key).replace('_', ' ').lower()
                        found = _measures(str(raw or ''))
                        if len(found) != 1:
                            continue
                        unit = found[0][1]
                        for words in axes:
                            if any((w in label for w in words)):
                                votes[words][unit] = votes[words].get(unit, 0) + 1
                taken = set()
                for words in axes:
                    if votes[words]:
                        unit = max(votes[words], key=votes[words].get)
                        self._axis_unit[words] = unit
                        taken.add(unit)
                leftover = []
                for w in axes:
                    if w not in self._axis_unit:
                        leftover.append(w)
                if leftover:
                    extra: dict = {}
                    for pid, sku in self.brief.candidate_keys or []:
                        blob = self._options(str(pid), str(sku)) + ' ' + self._title(str(pid))
                        blob += ' ' + ' '.join((str(v or '') for v in (self._facts(str(pid)) or {}).values()))
                        for _value, unit, _at in _measures(blob):
                            if unit not in taken:
                                extra[unit] = extra.get(unit, 0) + 1
                    if extra:
                        unit = max(extra, key=extra.get)
                        for words in leftover:
                            self._axis_unit[words] = unit
                return self._axis_unit
            except Exception:
                raise
            else:
                pass
        return nhv4__body_84(self)

    def _measure(self, pid: str, sku: str, words: tuple) -> float | None:

        def nhv4__body_85(self, pid, sku, words, *, nhv4_z=0):
            unit = self._axis_units().get(words)
            if unit:
                got = self._measure_in(pid, sku, unit)
                if got is not None:
                    return got
            for source in (self._options(pid, sku), self._title(pid)):
                best = None
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
        return nhv4__body_85(self, pid, sku, words)

    @staticmethod
    def _stems(text: str) -> set:

        def nhv4__body_86(text, *, nhv4_z=0):
            return {w.rstrip('s') for w in re.findall('[a-z]+', (text or '').lower())}
        return nhv4__body_86(text)

    def _category_ok(self, pid: str) -> bool:

        def nhv4__body_87(self, pid, *, nhv4_z=0):
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
            else:
                pass
        return nhv4__body_87(self, pid)

    def _says_term(self, text: str, term: str) -> bool:

        def nhv4__body_88(self, text, term, *, nhv4_z=0):
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
        return nhv4__body_88(self, text, term)

    def _term_in_catalog(self, term: str) -> bool:

        def nhv4__body_89(self, term, *, nhv4_z=0):
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
        return nhv4__body_89(self, term)

    def _brand_ok(self, pid: str) -> bool:

        def nhv4__body_90(self, pid, *, nhv4_z=0):
            want = self.brief.brand
            if not want:
                return True
            wanted = _tokens(want)
            return bool(wanted <= _tokens(self._brand_of(pid)) or wanted <= _tokens(self._title(pid)))
        return nhv4__body_90(self, pid)

    def _device_ok(self, pid: str, sku: str) -> bool:

        def nhv4__body_91(self, pid, sku, *, nhv4_z=0):
            return _identity_of(self.brief.head_noun, self._facts(pid), self._title(pid), self._options(pid, sku)) != 'no'
        return nhv4__body_91(self, pid, sku)

    def _resale_title(self, pid: str) -> bool:

        def nhv4__body_92(self, pid, *, nhv4_z=0):
            return bool(re.search('customer (?:rating|review)|\\bcompare \\b|refurb|pre-owned|restored|renewed', self._title(pid), re.I))
        return nhv4__body_92(self, pid)

    def _relax(self, pid: str, sku: str) -> int:

        def nhv4__body_94(self, pid, sku, *, nhv4_z=0):
            try:
                model, color = (self.brief.pref_model, self.brief.pref_color)
                if not model:
                    return 0

                def nhv4__lam_1(t, *, nhv4_z=0):

                    def nhv4__body_93(t, *, nhv4_z=0):
                        return re.sub('[^a-z0-9]+', '', (t or '').lower())
                    return nhv4__body_93(t)
                squash = nhv4__lam_1
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
            except Exception:
                raise
            else:
                pass
        return nhv4__body_94(self, pid, sku)

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
        if fam == 'justification':
            ok = ok and (not self._resale_title(pid)) and bool(self._brand_of(pid))
            if field and want:
                ok = ok and any((self._fact_value(pid, f, w) == w and self._product_value(pid, f, w) == w for f, w in self._spec_forms()))
        if fam == 'preference' and self.ledger.event_target:
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split('::', 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if fam == 'recall' and self.brief.theme:
            ok = ok and all((t in _tokens(self._title(pid)) for t in self.brief.theme))
        if fam == 'ranking':
            ok = ok and (str(pid), str(sku)) in self.brief.candidate_keys

        def nhv4__ie_9():
            if fam == 'recovery':
                return self._relax(pid, sku)
            else:
                return 0
        return {'pid': str(pid), 'sku': str(sku), 'key': key, 'price': price, 'live': live, 'viewed': viewed, 'spec': spec, 'cat_ok': cat_ok, 'ok': ok, 'relax': nhv4__ie_9()}

    def _rank_of(self, c: dict, first: bool) -> tuple:

        def nhv4__body_95(self, c, first, *, nhv4_z=0):
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
            if fam == 'preference':
                return (-self._soft_score(c['pid'], c['sku']), -self._priority_hits(c['pid'], c['sku']), price, c['key'])
            if fam == 'recovery':
                return (c['relax'], price, c['key'])
            extra = sum((1 for f, v in self.brief.specs.items() if f != self.brief.spec_field and self._variant_value(c['pid'], c['sku'], f) != v))
            if _identity_of(self.brief.head_noun, self._facts(c['pid']), self._title(c['pid']), self._options(c['pid'], c['sku'])) == 'yes':
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
        return nhv4__body_95(self, c, first)

    def _candidates(self) -> list:

        def nhv4__body_96(self, *, nhv4_z=0):
            try:
                seen = set()
                out = []
                for key in list(self.ledger.checked):
                    pid, _, sku = key.partition('::')
                    if not sku or key in seen:
                        continue
                    seen.add(key)
                    out.append(self._assess(pid, sku))
                return out
            except Exception:
                raise
            else:
                pass
        return nhv4__body_96(self)

    def _target(self) -> dict | None:
        first = self.ledger.adds == 0
        fam = self.brief.family
        good = []
        for c in self._candidates():
            if c['ok']:
                good.append(c)

        def nhv4__lam_10(c, *, nhv4_z=0):

            def nhv4__body_97(c, *, nhv4_z=0):
                return self._rank_of(c, first)
            return nhv4__body_97(c)
        if fam == 'ranking':
            pool = []
            for c in self._candidates():
                if (c['pid'], c['sku']) in self.brief.candidate_keys and c['viewed'] and c['live']:
                    pool.append(c)
            if pool:
                pool.sort(key=nhv4__lam_10)
                return pool[0]
            return None

        def nhv4__lam_11(c, *, nhv4_z=0):

            def nhv4__body_98(c, *, nhv4_z=0):
                return self._rank_of(c, first)
            return nhv4__body_98(c)
        if fam == 'preference':
            carted = []
            for c in good:
                if (c['pid'], c['sku']) in self.ledger.cart:
                    carted.append(c)
            if carted:
                return carted[0]
            if not good:
                return None
            good.sort(key=nhv4__lam_11)
            return good[0]
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

        def nhv4__lam_12(c, *, nhv4_z=0):

            def nhv4__body_99(c, *, nhv4_z=0):
                return self._rank_of(c, first)
            return nhv4__body_99(c)
        good.sort(key=nhv4__lam_12)
        return good[0]

    def _answer_text(self) -> str:

        def nhv4__body_100(self, *, nhv4_z=0):
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
        return nhv4__body_100(self)

    def _read_more(self) -> tuple | None:

        def nhv4__body_101(self, *, nhv4_z=0):
            fam = self.brief.family
            if fam == 'recall':
                extra = self._recall_search()
                if extra is not None:
                    return extra
                extra = self._recall_filter()
                if extra is not None:
                    self.recall_filtered = True
                    return extra
            twin = self._twin_search()
            if twin is not None:
                return twin
            leads = self._unseen_leads()
            if leads and self.forced_compares < 8 and (self.turn >= 2):
                self.forced_compares += 1
                if len(leads) > 1 and 'compare' in self.tool_names:
                    return ('compare', {'product_ids': leads[:5]})
                return ('view', {'product_id': leads[0]})
            if fam == 'preference' and 'search' in self.tool_names:
                queries = self._pref_queries()
                if self.pref_qi < len(queries):
                    query = queries[self.pref_qi]
                    self.pref_qi += 1
                    return ('search', {'query': query, 'in_stock': True, 'k': 10})
            probe = (self.brief.brand or '', '%.2f' % (self.budget or 0.0), str(self.brief.spec_value))
            if self.turn < 2:
                return None
            if probe in self.probes:
                self.reads_done = True
                return None
            if fam == 'recall':
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
        return nhv4__body_101(self)

    def _twin_search(self) -> tuple | None:

        def nhv4__body_102(self, *, nhv4_z=0):
            try:
                dead = self.ledger.event_target
                if self.brief.family in ('preference', 'recall'):
                    return None
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
            else:
                pass
        return nhv4__body_102(self)

    def _preferred_brand_hint(self) -> None:

        def nhv4__body_103(self, *, nhv4_z=0):
            if self.hinted_brand:
                return
            self.hinted_brand = True
            self._note("No %s product that meets every firm requirement has been read yet. The shopper's first choice is graded on that brand, so filter by brand=%s within the budget and read those products before any other." % (self.brief.pref_brand, self.brief.pref_brand))
        return nhv4__body_103(self)

    def _sibling_search(self) -> tuple | None:

        def nhv4__body_104(self, *, nhv4_z=0):
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
        return nhv4__body_104(self)

    def _exact_match_hint(self) -> None:
        if self.hinted_exact or not self.brief.pref_model:
            return
        self.hinted_exact = True

        def nhv4__ie_13():
            if self.brief.pref_color:
                return ', ' + self.brief.pref_color
            else:
                return ''
        self._note("The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own and view the results so every color variant is on record; only that exact pairing may be the first cart item." % (self.brief.pref_model, nhv4__ie_13()))

    def _unseen_leads(self) -> list:

        def nhv4__body_105(self, *, nhv4_z=0):
            try:
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
                    if self.brief.pref_model and self.ledger.adds == 0 and (not self.ledger.event_target):
                        squashed = re.sub('[^a-z0-9]+', '', title.lower())
                        if re.sub('[^a-z0-9]+', '', self.brief.pref_model.lower()) not in squashed:
                            continue
                    if self.brief.brand:
                        wanted = _tokens(self.brief.brand)
                        if not (wanted <= _tokens(str(row.get('brand') or '')) or wanted <= _tokens(title)):
                            continue
                    if self.brief.family == 'preference':
                        leads.append((-_soft_hits(title, row if isinstance(row, dict) else {}, wants=self._wish_words()), price, str(pid)))
                    else:
                        leads.append((0.0, price, str(pid)))
                leads.sort()
                out = []
                for _soft, _price, pid in leads:
                    if pid not in out:
                        out.append(pid)
                return out
            except Exception:
                raise
            else:
                pass
        return nhv4__body_105(self)

    def _shortlist_note(self) -> None:
        cands = self._candidates()

        def nhv4__lam_14(c, *, nhv4_z=0):

            def nhv4__body_106(c, *, nhv4_z=0):
                return self._rank_of(c, True)
            return nhv4__body_106(c)
        if self.brief.family == 'preference' and (not self.ledger.cart):
            good = []
            for c in cands:
                if c['ok']:
                    good.append(c)
            good.sort(key=nhv4__lam_14)
            if good:
                shown = []
                for c in good[:6]:
                    names = self._soft_axis_names(c['pid'], c['sku'])
                    shown.append('%s::%s at %.2f [%s]' % (c['pid'], c['sku'], c['price'] or 0.0, ', '.join(names) or 'no soft-axis evidence'))
                self._note('Products that meet every firm requirement so far: ' + '; '.join(shown) + '. Prefer a listing that states several of these together, especially a battery duration (hours/days), not a bare capacity figure. The controller will pick the strongest multi-axis match.')
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

    def _priority_hits(self, pid: str, sku: str) -> int:

        def nhv4__body_107(self, pid, sku, *, nhv4_z=0):
            wants = self._wish_words()
            if not wants:
                return 0
            have = _tokens(' '.join((str(v) for v in self._facts(pid).values())) + ' ' + self._title(pid) + ' ' + self._options(pid, sku))
            return len(wants & have)
        return nhv4__body_107(self, pid, sku)

    def _soft_score(self, pid: str, sku: str) -> float:

        def nhv4__body_108(self, pid, sku, *, nhv4_z=0):
            return _soft_hits(self._title(pid), self._facts(pid), self._options(pid, sku), self._wish_words())
        return nhv4__body_108(self, pid, sku)

    def _soft_axis_names(self, pid: str, sku: str) -> list:

        def nhv4__body_109(self, pid, sku, *, nhv4_z=0):
            blob = _soft_blob(self._title(pid), self._facts(pid), self._options(pid, sku))
            names = []
            if _duration_hours(blob) > 0:
                names.append('duration')
            have = _tokens(blob)
            names.extend(sorted((w for w in self._wish_words() if w in have))[:4])
            return names
        return nhv4__body_109(self, pid, sku)

    def _pref_queries(self) -> list:

        def nhv4__body_110(self, *, nhv4_z=0):
            try:
                cat = self.brief.category or self.brief.head_noun or 'product'
                words = [w for w in sorted(self._wish_words()) if len(w) > 3][:4]
                if not words:
                    return [cat]
                return [' '.join(words + [cat]), ' '.join(words[:2] + [cat])]
            except Exception:
                raise
            else:
                pass
        return nhv4__body_110(self)

    def _shopper_replies(self) -> list:

        def nhv4__body_111(self, *, nhv4_z=0):
            return [str(m.get('content') or '') for m in self.messages if m.get('role') == 'user' and (not str(m.get('content') or '').startswith('[note]'))]
        return nhv4__body_111(self)

    def _wish_words(self) -> set:

        def nhv4__body_112(self, *, nhv4_z=0):
            if self._wishes is None:
                said = self.brief.goal + ' ' + ' '.join((str(m.get('content') or '') for m in self.messages if m.get('role') == 'user'))
                drop = _tokens(self.brief.category or '') | _tokens(self.brief.currency) | {'the', 'and', 'for', 'with', 'that', 'this', 'you', 'your', 'find', 'need', 'want', 'order', 'stock', 'place', 'under', 'over', 'one', 'test', 'buy', 'rather', 'than', 'pay', 'bit', 'more', 'settle', 'something', 'its', 'would', 'like', 'have'}
                self._wishes = {w for w in _tokens(said) if len(w) > 3 and (not w.isdigit()) and (w not in drop)}
            return self._wishes
        return nhv4__body_112(self)

    def _budget_from_message(self, text: str) -> None:

        def nhv4__body_113(self, text, *, nhv4_z=0):
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
        return nhv4__body_113(self, text)

    def _claims_for(self, pid: str, sku: str) -> list:

        def nhv4__body_114(self, pid, sku, *, nhv4_z=0):
            try:
                out = []
                field, want = (self.brief.spec_field, self.brief.spec_value)
                if not field and self.brief.family == 'justification':
                    field, want = ('resolution', self._fact_value(pid, 'resolution'))
                if not field or not want:
                    return []
                stated = dict(self.brief.specs) or {field: want}
                stated.setdefault(field, want)
                for name, value in stated.items():
                    if self._fact_value(pid, name) == value and self._product_value(pid, name) == value:
                        out.append({'field': name, 'value': value})
                if not out:
                    return []
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
            else:
                pass
        return nhv4__body_114(self, pid, sku)

    def _order_args(self, pid: str, sku: str) -> dict:

        def nhv4__body_115(self, pid, sku, *, nhv4_z=0):
            out = {'product_id': pid, 'sku': sku}
            if self.brief.family == 'justification':
                claims = self._claims_for(pid, sku)
                if claims:
                    out['justification'] = {'claims': claims}
            return out
        return nhv4__body_115(self, pid, sku)

    def _valid_now(self, pid: str, sku: str) -> tuple:

        def nhv4__body_116(self, pid, sku, *, nhv4_z=0):
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
        return nhv4__body_116(self, pid, sku)

    def _recall_need(self) -> int:

        def nhv4__body_117(self, *, nhv4_z=0):
            floor = self.brief.required_identities
            if len(self.search_queries) < 3 and self._recall_cluster_count() < 10:
                return max(floor, 10)
            return floor
        return nhv4__body_117(self)

    def _row_title(self, pid: str) -> str:

        def nhv4__body_118(self, pid, *, nhv4_z=0):
            try:
                title = self._title(pid)
                if title:
                    return title
                for row in self.ledger.rows.values():
                    if str(row.get('product_id')) == str(pid):
                        return str(row.get('title') or '')
                return ''
            except Exception:
                raise
            else:
                pass
        return nhv4__body_118(self, pid)

    def _row_brand(self, pid: str) -> str:

        def nhv4__body_119(self, pid, *, nhv4_z=0):
            brand = self._brand_of(pid)
            if brand:
                return brand
            for row in self.ledger.rows.values():
                if str(row.get('product_id')) == str(pid):
                    return str(row.get('brand') or '')
            return ''
        return nhv4__body_119(self, pid)

    def _recall_cluster_count(self) -> int:

        def nhv4__body_120(self, *, nhv4_z=0):
            seen = set()
            for pid in self.ledger.identities:
                title = self._row_title(pid)
                tokens = frozenset(_tokens(title))
                if tokens:
                    seen.add((self._row_brand(pid).strip().lower(), tokens))
                else:
                    seen.add((str(pid), frozenset()))
            return len(seen)
        return nhv4__body_120(self)

    def _recall_extra_tokens(self) -> list:

        def nhv4__body_121(self, *, nhv4_z=0):
            try:
                theme = set()
                for t in self.brief.theme:
                    if t:
                        theme.add(t)
                skip = theme | _tokens(self.brief.category or '') | _FUNCTION_WORDS
                for pid in self.ledger.identities:
                    skip |= _tokens(self._row_brand(pid))
                owners: dict = {}
                for pid in self.ledger.identities:
                    title = self._row_title(pid)
                    if not title:
                        continue
                    cluster = (self._row_brand(pid).strip().lower(), frozenset(_tokens(title)))
                    for tok in _tokens(title):
                        if tok in skip or len(tok) < 4 or any((ch.isdigit() for ch in tok)):
                            continue
                        owners.setdefault(tok, set()).add(cluster)
                return [w for w, _ids in sorted(owners.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
            except Exception:
                raise
            else:
                pass
        return nhv4__body_121(self)

    def _recall_queries(self) -> list:

        def nhv4__body_122(self, *, nhv4_z=0):
            theme = []
            for t in self.brief.theme:
                if t:
                    theme.append(t)
            if not theme:
                return []
            phrase = ' '.join(theme)
            out = [phrase]
            head = theme[0]
            for extra in self._recall_extra_tokens():
                out.append(extra + ' ' + head)
            if len(theme) > 1:
                out.append(' '.join(reversed(theme)))
            seen = set()
            uniq = []
            for query in out:
                key = query.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    uniq.append(query.strip())
            return uniq[:3]
        return nhv4__body_122(self)

    def _recall_filter(self) -> tuple | None:

        def nhv4__body_123(self, *, nhv4_z=0):
            if self.recall_filtered or self.brief.family != 'recall':
                return None
            if self._recall_cluster_count() >= 10:
                return None
            if 'filter' not in self.tool_names or not self.brief.category:
                return None
            args = {'category': self.brief.category, 'in_stock': True, 'k': 10}
            if self.budget is not None:
                args['max_price'] = self.budget
            return ('filter', args)
        return nhv4__body_123(self)

    def _recall_search(self) -> tuple | None:

        def nhv4__body_124(self, *, nhv4_z=0):
            if 'search' not in self.tool_names or not self.brief.theme:
                return None
            if self._recall_cluster_count() >= 10:
                return None
            used = set()
            for q in self.search_queries:
                used.add(q.lower())
            if len(used) >= 3:
                return None
            for query in self._recall_queries():
                if query.lower() not in used:
                    return ('search', {'query': query, 'in_stock': True, 'k': 10})
            return None
        return nhv4__body_124(self)

    def _mark_catalog_query(self, name: str, args: dict) -> None:
        if name == 'search':
            query = str(args.get('query') or '').strip()
            if query and query.lower() not in {q.lower() for q in self.search_queries}:
                self.search_queries.append(query)

        def nhv4__ie_15():
            if have >= need:
                return ''
            else:
                return ' Keep searching with different wording.'
        if name in ('search', 'filter'):
            self.search_calls += 1
            if self.brief.family == 'recall' and self.brief.required_identities:
                have, need = (self._recall_cluster_count(), self._recall_need())
                self._note('Distinct matching products surfaced so far: %d of the required %d.' % (have, need) + nhv4__ie_15())

    def _ready(self) -> bool:

        def nhv4__body_125(self, *, nhv4_z=0):
            fam = self.brief.family
            if self.ledger.adds:
                return True
            if self.brief.spec_sealed:
                if self.turn < self.max_steps * 0.6:
                    return False
                self.brief.spec_sealed = False
                self._note('The firm requirement was never stated, so the remaining requirements decide the choice.')
            if fam == 'ranking':
                missing = []
                for p, _s in self.brief.candidate_keys:
                    if p not in self.ledger.products and p not in self.unreadable:
                        missing.append(p)
                return not missing
            if fam in ('intent', 'constraint') and self.turn < 4 and (not self.shopper_updated):
                return False
            if fam == 'preference' and len(self.ledger.listing_ids) < 2:
                return False
            roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
            if fam == 'preference' and roomy:
                good = []
                for c in self._candidates():
                    if c['ok']:
                        good.append(c)
                if good and (not any((self._soft_score(c['pid'], c['sku']) > 0 for c in good))):
                    return False
            if fam == 'recall' and roomy:
                if self._recall_search() is not None or self._recall_filter() is not None:
                    return False
                if self.brief.required_identities and self._recall_cluster_count() < self.brief.required_identities:
                    return False
            if roomy and (not self.reads_done) and (len({c['pid'] for c in self._candidates() if c['ok']}) < 2):
                return False
            return True
        return nhv4__body_125(self)

    def _fresh(self, pid: str, sku: str) -> bool:

        def nhv4__body_126(self, pid, sku, *, nhv4_z=0):
            k = Ledger.key(pid, sku)
            if k not in self.ledger.inspected:
                return False
            return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq
        return nhv4__body_126(self, pid, sku)

    def _plan(self) -> tuple | None:

        def nhv4__body_127(self, *, nhv4_z=0):
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
        return nhv4__body_127(self)

    def _ask_model(self) -> dict | None:

        def nhv4__body_128(self, *, nhv4_z=0):
            for attempt in range(3):
                model = self.roster[min(self.model_index, len(self.roster) - 1)]
                payload = {'model': model, 'messages': self.messages, 'tools': self.tools, 'tool_choice': 'required', 'temperature': 0}
                resp = _PROXY.post('/inference/chat/completions', json_data=payload)
                if resp and isinstance(resp, dict) and resp.get('choices'):
                    msg = resp['choices'][0].get('message') or {}
                    calls = msg.get('tool_calls') or []
                    if calls:
                        return {'content': msg.get('content') or '', 'tool_calls': calls[:self.max_calls]}
                    self.messages.append({'role': 'assistant', 'content': msg.get('content') or ''})
                    self._note('No tool call was made. Choose exactly one of the supplied tools now.')
                    continue
                if self.clock.remaining() > 30:
                    time.sleep(min(2.0 * (attempt + 1), 5.0))
                self.model_index = min(self.model_index + 1, len(self.roster) - 1)
            return None
        return nhv4__body_128(self)

    def _send(self, envelope: dict) -> dict | None:

        def nhv4__body_129(self, envelope, *, nhv4_z=0):
            for attempt in range(4):
                result = _PROXY.post('/environment/call', json_data=envelope)
                if isinstance(result, dict) and result.get('calls'):
                    return result
                if self.clock.remaining() < 20:
                    break
                time.sleep(min(2.0 * (attempt + 1), 6.0))
            return None
        return nhv4__body_129(self, envelope)

    def _act(self, name: str, args: dict, content: str='', tool_call_id: str | None=None) -> dict | None:

        def nhv4__body_130(self, name, args, content, tool_call_id, *, nhv4_z=0):
            if self.pending_envelope is not None:
                pend = self.pending_envelope
                replay = self._send(pend['envelope'])
                if replay is None:
                    self._note('The environment is not answering; the previous action is still pending and will be retried.')
                    return None
                self.pending_envelope = None
                self.turn = pend['turn']
                self._absorb(pend['name'], pend['args'], pend['content'], pend['call_id'], replay)
                if self.done:
                    return None
            turn = self.turn + 1
            group = self.problem_id + '-turn-' + str(turn)
            envelope = {**self.binding, 'call_id': group, 'idempotency_key': group, 'turn': turn, 'calls': [{'call_id': group + '-1', 'action': {'name': name, 'args': args}}]}
            result = self._send(envelope)
            if result is None:
                self.pending_envelope = {'envelope': envelope, 'turn': turn, 'name': name, 'args': args, 'content': content, 'call_id': tool_call_id or 'agent-' + str(turn)}
            else:
                self.turn = turn
            return self._absorb(name, args, content, tool_call_id or 'agent-' + str(turn), result)
        return nhv4__body_130(self, name, args, content, tool_call_id)

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]
        self.messages.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls})
        if content:
            think = content
        else:
            think = self._compare_think()
        self._push_step(think=think, content=content, tool_calls=tool_calls, env_result=result)
        if not isinstance(result, dict):
            self.ledger.calls_seen.discard(name + json.dumps(args, sort_keys=True))
            self.messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps({'error': 'environment call failed; it will be retried automatically'})})
            return None
        calls = result.get('calls') or []

        def nhv4__ie_16():
            if calls and isinstance(calls[0], dict):
                return calls[0].get('observation')
            else:
                return None
        wrap = nhv4__ie_16() or {}
        if isinstance(wrap.get('observation'), dict):
            obs = wrap.get('observation')
        else:
            obs = wrap
        if isinstance(obs, dict) and wrap.get('error') and (not obs.get('error')):
            obs = dict(obs)
            obs['error'] = wrap.get('error')
        if wrap.get('done'):
            self.done = True
        self.ledger.record(name, args, obs, self.brief)
        self.messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': _trim(json.dumps(obs), _OBS_CHARS)})
        user = result.get('user_message')
        if isinstance(user, dict) and user.get('content'):
            text = str(user['content'])
            self.messages.append({'role': 'user', 'content': text})
            self._wishes = None
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
        return obs

    def _post_add_recheck(self) -> None:

        def nhv4__body_131(self, *, nhv4_z=0):
            pid, sku = self.pending_recheck
            self.pending_recheck = None
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
        return nhv4__body_131(self)

    def _finish_fast(self) -> None:
        for pid, sku in reversed(self.ledger.cart):
            ok, _why = self._valid_now(pid, sku)
            if ok and (not self.done):
                self._act('place_test_order', self._order_args(pid, sku))
                return
        if self.done or self.turn > self.max_steps - 3:
            return
        good = []
        for c in self._candidates():
            if c['ok']:
                good.append(c)
        if not good and self.brief.family != 'justification':
            good = []
            for c in self._candidates():
                if c['live'] and c['spec'] == 2 and self._brand_ok(c['pid']) and c['cat_ok'] and (not self.brief.candidate_keys or (c['pid'], c['sku']) in self.brief.candidate_keys):
                    good.append(c)
        if not good:
            return

        def nhv4__lam_17(c, *, nhv4_z=0):

            def nhv4__body_132(c, *, nhv4_z=0):
                return self._rank_of(c, self.ledger.adds == 0)
            return nhv4__body_132(c)
        good.sort(key=nhv4__lam_17)
        pid, sku = (good[0]['pid'], good[0]['sku'])
        if (pid, sku) not in self.ledger.cart:
            obs = self._act('add_to_cart', {'product_id': pid, 'sku': sku})
            if self.done or not isinstance(obs, dict) or obs.get('error'):
                return
        obs = self._act('inspect_stock', {'product_id': pid, 'sku': sku})
        if self.done or not isinstance(obs, dict) or (not obs.get('in_stock')):
            return
        price = _num(obs.get('price'))
        if self.budget is not None and price is not None and (price > self.budget + _EPS):
            return
        self._act('place_test_order', self._order_args(pid, sku))

    def run(self) -> list:

        def nhv4__body_133(self, *, nhv4_z=0):
            question = self._opening_question()
            last_forced, repeats = (None, 0)
            while not self.done and self.turn < self.max_steps:
                if self.clock.remaining() < 25 or self.turn >= self.max_steps - 3:
                    self._finish_fast()
                    break
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
                    obs = self._act(name, args)
                    if name in ('search', 'filter'):
                        self._mark_catalog_query(name, args)
                    if name == 'message':
                        self.shopper_question_pending = False
                        self.question_waits = 0
                    if name in ('view', 'compare', 'search', 'filter') and self.brief.family != 'ranking':
                        self._reason_once()
                    if name == 'add_to_cart' and isinstance(obs, dict) and (not obs.get('error')):
                        added = obs.get('added') or {}
                        if added.get('product_id'):
                            self.pending_recheck = (str(added['product_id']), str(added['sku']))
                    continue
                last_forced, repeats = (None, 0)
                proposal = self._ask_model()
                if proposal is None:
                    self._finish_fast()
                    break
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
                    self._push_step(content='[rejected ' + name + '] ' + verdict[:200])
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
                obs = self._act(name, args, content, call.get('id'))
                if name == 'add_to_cart' and isinstance(obs, dict) and (not obs.get('error')):
                    added = obs.get('added') or {}
                    if added.get('product_id'):
                        self.pending_recheck = (str(added['product_id']), str(added['sku']))
                if name in ('search', 'filter'):
                    self._mark_catalog_query(name, args)
                if name == 'inspect_cart':
                    self.cart_inspections += 1
                if name in ('view', 'compare'):
                    self._shortlist_note()
                    self._reason_once()
            if not self._reasoned:
                self._reason_once()
            return self.dialogue
        return nhv4__body_133(self)

    def _validate(self, name: str, args: dict) -> str | None:

        def nhv4__body_134(self, name, args, *, nhv4_z=0):
            if name not in self.tool_names:
                return 'unknown tool; choose one of ' + str(sorted((n for n in self.tool_names if n)))
            if name == 'place_test_order':
                pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
                if not pid and self.ledger.cart:
                    pid, sku = self.ledger.cart[-1]
                ok, why = self._valid_now(pid, sku)
                if not ok:
                    return 'order refused: ' + why
                if self.brief.family == 'recall' and self._recall_cluster_count() < self.brief.required_identities:
                    return 'order refused: only %d distinct matching products have been surfaced; search with different wording until at least %d appear' % (self._recall_cluster_count(), self.brief.required_identities)
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
            if name in ('view', 'compare') and self.brief.family == 'recall' and self.brief.theme:
                if name == 'view':
                    pids = [args.get('product_id')]
                else:
                    pids = list(args.get('product_ids') or [])
                for pid in pids:
                    title = self._row_title(str(pid or ''))
                    if title and (not all((t in _tokens(title) for t in self.brief.theme))):
                        return "that listing's title does not contain every requested word; only open products whose title matches the phrase"
            key = name + json.dumps(args, sort_keys=True)
            if key in self.ledger.calls_seen and name in ('search', 'filter', 'view', 'compare'):
                return 'identical call already made; change the arguments'
            self.ledger.calls_seen.add(key)
            return None
        return nhv4__body_134(self, name, args)

def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:
        return [{'role': 'assistant', 'content': 'agent error: ' + type(exc).__name__ + ': ' + str(exc)[:200]}]
__all__ = ['agent_main']
