from __future__ import annotations
import json
import re
import time
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
class Agent:
    """Catalog session: brief → search → first cart → recover → order."""

    # -- settings, prompts, patterns --
    _MODELS = {'chutes': ('deepseek-ai/DeepSeek-V3.2-TEE', 'Qwen/Qwen3-32B-TEE', 'zai-org/GLM-5.2-TEE'), 'openrouter': ('deepseek/deepseek-v3.2', 'qwen/qwen3-32b', 'z-ai/glm-5.2')}
    _PROXY = ProxyClient(timeout=70, max_retries=1, retry_delay=0.4)
    _PROBLEM_BUDGET_S = 255.0
    _FINISH_RESERVE_S = 65.0
    _OBS_CHARS = 5000
    _MAX_REPROMPTS = 1
    _EPS = 1e-09
    _REASON_RE = re.compile('\\b(walk me through|tell me why|explain why|justify|written reason|give me a reason|reason i can trust|trustworthy reason|why (?:this|that|you (?:picked|chose|selected)))\\b', re.I)
    _MODELISH_RE = re.compile('\\b(?=[a-z0-9\\-]*\\d)[a-z0-9][a-z0-9\\-]{2,}\\b', re.I)
    _PHRASE_RE = re.compile('matches(?: the phrase)?\\s+[\\"“]?([a-z0-9][a-z0-9 \\-]{1,40}?)[\\"”]?(?:,| at | and |\\.|$)', re.I)
    _PREFER_RE = re.compile('(?:strongly prefer|prefer(?:s|red)?(?: the)?)\\s+([a-z0-9][\\w\\- ]{0,40}?)(?:\\.|,|;| if | and | but |\\n|$)', re.I)
    _SOFT_PICK_RE = re.compile('(?:lean toward(?:s)?\\s+([A-Za-z][A-Za-z0-9\\-]{1,24}))|([A-Za-z][A-Za-z0-9\\-]{1,24})\\s+would be my pick', re.I)
    _EXACT_PAIR_RE = re.compile('strongly prefer|exact(?:ly)?(?: the)? (?:color|model|pair|combination)|if that exact|another color of the same model|version of the|ideally in\\s+[a-z]+|the one i like is', re.I)
    _PAIR_COLOR_RE = re.compile('(?:the one i like is the\\s+|prefer(?:s|red)?(?: the)?\\s+|the\\s+)?([a-z]{3,12})\\s+version of(?: the)?\\s+([a-z0-9\\-]+)|(?:strongly prefer|prefer(?:s|red)?)(?: the)?\\s+([a-z]{3,12})\\s+([a-z][a-z0-9]*\\d[a-z0-9]*)|ideally in\\s+([a-z]{3,12})', re.I)
    _TRADEOFF_RE = re.compile("\\b(i'?d rather|rather pay|cheapest isn'?t|don'?t just grab the cheapest|which of your preferences|weigh the others)\\b", re.I)
    _HIDDEN_SPEC_RE = re.compile('tell you if you ask|firm requirement on the spec|hidden (?:firm )?requirement', re.I)
    _NUM = '(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?'
    _BUDGET_RE = re.compile('(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|within|budget\\s*(?:is|of|:)?|only go up to|can (?:spend|go) up to|at most|my limit is|limit is|limit of)\\s*' + _NUM + '\\s*(PHP|php|₱|pesos?)?', re.I)
    _KEY_RE = re.compile('(\\d{6,})::([A-Za-z0-9._\\-]+)')
    _QUOTED_RE = re.compile('\\"([^\\"]{2,80})\\"')
    _QUESTION_RE = re.compile('\\?|\\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)|tell me (?:which|whether|if)|confirm whether)\\b', re.I)
    _CORE_RULES = "You are a careful shopping agent working a catalog through tools. Exactly one tool call per turn, at most 30 turns, so be efficient. Search or filter, then view or compare so raw facts, variant options and prices are on record. Never add a placeholder to the cart — the first addition is graded. Prices and stock change; after a cart addition the item is re-checked and you may need a different product. Answer every direct question from the shopper with one short message before anything else. Never send progress updates. Never ask the shopper to raise the budget. Do not repeat a tool call with identical arguments; one compare of up to five products beats several view calls. Read evidence the way a careful shopper would: labelled facts first, then the selected variant's option text, then the title. If those disagree, the specification is not proven. Every must-have the shopper stated that a product card can prove is a hard gate — any labelled fact, quantity, duration, count, class, or named pairing, in the shopper's own wording. Preferences only rank items that already pass. If the shopper would rather pay more than grab the cheapest, first-cart a listing whose own facts mention those priorities — price is the tie-break, not the goal. Search with the shopper's own words and quantities so catalog values can match. Buy the item that the request itself would select, not a near-miss, unless the request says a fallback is allowed. If a written reason is required, copy only field names and values from that product's labelled facts plus the live price and stock check, using only fields the order tool allows. Never invent a field name. If the carted item disappears, recover from the shopper's own words — same model, next acceptable option, or the next listing that still meets the request. Do not wait for extra coaching in the task notes."
    _JUDGE_SYS = "You classify a shopping task and its catalog evidence. Reply with a single JSON object, no markdown, no tools, no prose. You already know everyday retail language: you can tell when two phrases name the same quantity, when a listing is the requested kind of product rather than an accessory, spare part, or 'compatible with' item, and when a title is describing someone else's listing or a used unit. Use only the request and the evidence you are given. Do not invent facts. Firm requirements are whatever the shopper made must-have, in their own wording — including every labelled-fact kind of requirement a catalog card can prove. A variant is ok only when each of those is proven: labelled facts first, then that variant's option text, then the title. Everyday retail equivalents count; guesses do not. If a labelled fact exists and disagrees with the title, the labelled fact wins and the variant is not ok. Title-only proof is not enough when a written reason is required — the labelled facts themselves must state the specification. A named brand can be a soft preference: it ranks the first cart item but does not override a firm requirement. When the request lists product_id::sku pairs, only those pairs count. Extract the shopper's own tradeoff order as rank axes, then for each variant report a number for each axis taken from that variant's option text, then labelled facts, then the title (prefer a variant-specific maximum when the shopper asked for the highest advertised maximum). Claims must use only field names the order tool allows. If a labelled fact uses a different key than the allowed field, still emit the allowed field name and copy the value from that labelled fact. Never emit a labelled key that the order tool does not allow. Infer ranking, recovery, and whether a written reason is required from the shopper request itself — not from hidden task coaching."
    _COMPARE_AXIS_RE = re.compile('\\b(highest|largest|most|lowest|smallest|least|cheapest)\\s+([^.;,\\n]{2,48})', re.I)
    _FIELD_SNAKE_RE = re.compile('\\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\\b')
    _FIELD_LIST_RE = re.compile('(?:only:|fields?:|facts only:|claims? for)\\s*([^.]+)', re.I)
    @staticmethod
    def _num(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(str(value).replace(',', ''))
        except (TypeError, ValueError):
            return None
    @staticmethod
    def _tokens(text: str) -> set:
        return set(re.findall('[a-z0-9]+', (text or '').lower()))
    @staticmethod
    def _arguments(tool_call: dict) -> dict:
        raw = (tool_call.get('function') or {}).get('arguments', '{}')
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw or '{}')
        except (TypeError, ValueError):
            parsed = {}
        return parsed if isinstance(parsed, dict) else {}
    @staticmethod
    def _trim(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit - 40] + ' ... (truncated by agent)"}'
    @staticmethod
    def _parse_json(text: str) -> dict:
        raw = (text or '').strip()
        if raw.startswith('```'):
            raw = re.sub('^```(?:json)?\\s*|\\s*```$', '', raw, flags=re.I)
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            pass
        (start, end) = (raw.find('{'), raw.rfind('}'))
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except (TypeError, ValueError):
                return {}
        return {}
    @staticmethod
    def _budget_in(text: str) -> float | None:
        best = None
        for m in _BUDGET_RE.finditer(text or ''):
            whole = m.group(1).replace(',', '')
            frac = m.group(2) or '0'
            try:
                val = float(whole + '.' + frac)
            except ValueError:
                continue
            curr = (m.group(3) or '').strip() if m.lastindex and m.lastindex >= 3 else ''
            if val < 10 and not curr:
                continue
            if best is None or val < best:
                best = val
        return best
    @staticmethod
    def _exact_pair_ask(text: str) -> bool:
        return bool(_EXACT_PAIR_RE.search(text or ''))
    @staticmethod
    def _pairing_words(text: str) -> list:
        out = []
        for m in _PAIR_COLOR_RE.finditer(text or ''):
            out.extend((g.lower() for g in m.groups() if g))
        return out
    @staticmethod
    def _usable_brand(text: str) -> str:
        t = (text or '').strip()
        if not t or _MODELISH_RE.search(t) or len(t.split()) > 3:
            return ''
        return t
    @staticmethod
    def _norm_unit(unit: str) -> str:
        raw = (unit or '').strip().lower()
        if raw in {'"', '”', '″', "''"} or raw.endswith('"'):
            return 'in'
        u = re.sub('[^a-z]', '', raw)
        aliases = {'in': 'in', 'inch': 'in', 'inches': 'in', 'gb': 'gb', 'gbyte': 'gb', 'gigabyte': 'gb', 'gigabytes': 'gb', 'mb': 'mb', 'mbyte': 'mb', 'megabyte': 'mb', 'megabytes': 'mb', 'tb': 'tb', 'tbyte': 'tb', 'terabyte': 'tb', 'hz': 'hz', 'hertz': 'hz', 'khz': 'khz', 'mhz': 'mhz', 'ghz': 'ghz', 'mah': 'mah'}
        return aliases.get(u, u)
    @staticmethod
    def _measure_mentioned(blob: str, n: float, unit: str) -> bool:
        want = _norm_unit(unit)
        text = blob or ''
        if not want:
            return any((abs(n - h) < 0.0001 for (h, _neigh) in _numbers_from_text(text)))
        for m in re.finditer('(\\d+(?:\\.\\d+)?)\\s*[-–]?\\s*([A-Za-z]{1,12}|["”″])?', text):
            hn = _num(m.group(1))
            if hn is None or abs(hn - n) >= 0.0001:
                continue
            if _norm_unit(m.group(2) or '') == want:
                return True
        compact = re.sub('[^a-z0-9"]+', '', text.lower())
        nrep = str(int(n)) if abs(n - int(n)) < 1e-09 else str(n).rstrip('0').rstrip('.')
        if nrep + want in compact:
            return True
        if want == 'in' and (nrep + 'inch' in compact or nrep + '"' in compact):
            return True
        return False
    @staticmethod
    def _backoff(attempt: int, cap: float=1.2) -> None:
        time.sleep(min(0.25 * (attempt + 1), cap))
    @staticmethod
    def _tool_fn(tools: list, name: str) -> dict:
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get('function') or {}
            if isinstance(fn, dict) and fn.get('name') == name:
                return fn
        return {}
    @staticmethod
    def _fields_from_description(text: str) -> list:
        if not text:
            return []
        skip = {'only', 'field', 'fields', 'value', 'facts', 'from', 'the', 'and', 'for', 'with', 'that', 'this', 'optional', 'checkable', 'purchase', 'must', 'type', 'string', 'object', 'array', 'true', 'false', 'when', 'shopper', 'asked', 'include', 'structured', 'justification', 'grounded', 'viewed', 'current'}
        found: list = []
        seen: set = set()
        for tok in _FIELD_SNAKE_RE.findall((text or '').lower()):
            if tok in skip or tok in seen:
                continue
            seen.add(tok)
            found.append(tok)
        for chunk in _FIELD_LIST_RE.findall(text or ''):
            for tok in re.findall('\\b[a-z][a-z0-9_]{2,}\\b', chunk.lower()):
                if tok in skip or tok in seen:
                    continue
                seen.add(tok)
                found.append(tok)
        for chunk in re.findall('(?:[a-z][a-z0-9_]+,\\s*){2,}[a-z][a-z0-9_]+', (text or '').lower()):
            for tok in re.findall('\\b[a-z][a-z0-9_]{2,}\\b', chunk):
                if tok in skip or tok in seen:
                    continue
                seen.add(tok)
                found.append(tok)
        return found
    # Generic English filler — never product-class or catalog keys.
    _AXIS_STOP = {'highest', 'largest', 'most', 'lowest', 'smallest', 'least', 'cheapest', 'advertised', 'maximum', 'max', 'biggest', 'best', 'then', 'among', 'ties', 'choose', 'the', 'a', 'an', 'of', 'with', 'remaining', 'including', 'explicitly', 'mode', 'please', 'compare', 'tradeoffs', 'available'}
    _WISH_STOP = {'the', 'and', 'for', 'with', 'that', 'this', 'you', 'your', 'find', 'need', 'want', 'order', 'stock', 'place', 'under', 'over', 'one', 'php', 'test', 'buy', 'rather', 'than', 'pay', 'bit', 'more', 'settle', 'something', 'its', 'would', 'like', 'have', 'from', 'into', 'just', 'also', 'only', 'them', 'they', 'when', 'what', 'which', 'about', 'been', 'being', 'does', 'done', 'each', 'very', 'much', 'some', 'such', 'then', 'please', 'could', 'should', 'still', 'even', 'able'}
    @staticmethod
    def _axes_from_text(text: str) -> list:
        out = []
        seen: set = set()
        for m in _COMPARE_AXIS_RE.finditer(text or ''):
            word = m.group(1).lower()
            hint = m.group(0).strip()
            rest = (m.group(2) or '').strip().lower()
            if word == 'cheapest' and re.match("(?:isn'?t|is not|don'?t)\\b", rest):
                continue
            key = hint.lower()
            if key in seen:
                continue
            seen.add(key)
            direction = 'min' if word in ('lowest', 'smallest', 'least', 'cheapest') else 'max'
            out.append({'direction': direction, 'from': hint[:120]})
            if len(out) >= 4:
                break
        return out
    @staticmethod
    def _hint_tokens(text: str) -> set:
        return {t for t in _tokens(text) if t not in _AXIS_STOP and (not t.isdigit()) and (len(t) > 1)}
    @staticmethod
    def _ask_only(goal: str) -> str:
        return re.split('\\n\\s*task rules\\s*:', goal or '', maxsplit=1, flags=re.I)[0].strip()
    @staticmethod
    def _field_tokens(name: str) -> set:
        return {t for t in re.split('[_\\-\\s]+', (name or '').lower()) if t and t not in _AXIS_STOP}
    @staticmethod
    def _token_overlap(a: set, b: set) -> int:
        if not a or not b:
            return 0
        n = len(a & b)
        if n:
            return n
        extra = 0
        for x in a:
            for y in b:
                if len(x) >= 3 and len(y) >= 3 and (x.startswith(y) or y.startswith(x)):
                    extra += 1
        return extra
    @staticmethod
    def _numbers_from_text(text: str) -> list:
        out = []
        raw = text or ''
        for m in re.finditer('(?<![A-Za-z0-9])(\\d{1,4}(?:\\.\\d+)?)(?!\\d)', raw):
            val = _num(m.group(1))
            if val is None:
                continue
            window = raw[max(0, m.start() - 24):min(len(raw), m.end() + 24)]
            neigh = _tokens(window)
            suffix = re.match('([A-Za-z]{2,4})', raw[m.end():])
            if suffix:
                neigh.add(suffix.group(1).lower())
            out.append((val, neigh))
        return out
    @staticmethod
    def _suffix_numbers(text: str) -> list:
        out = []
        raw = text or ''
        for m in re.finditer('(?<![A-Za-z0-9])(\\d{1,4}(?:\\.\\d+)?)([A-Za-z]{2,4})(?![A-Za-z])', raw):
            val = _num(m.group(1))
            if val is not None:
                out.append(val)
        return out
    @staticmethod
    def _claim_schema(tools: list) -> dict:
        fn = _tool_fn(tools, 'place_test_order')
        params = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {}
        props = params.get('properties') if isinstance(params.get('properties'), dict) else {}
        just = props.get('justification')
        if not isinstance(just, dict):
            return {'present': False, 'required': False, 'allowed': [], 'description': ''}
        just_props = just.get('properties') if isinstance(just.get('properties'), dict) else {}
        claims = just_props.get('claims') if isinstance(just_props.get('claims'), dict) else {}
        items = claims.get('items') if isinstance(claims.get('items'), dict) else {}
        item_props = items.get('properties') if isinstance(items.get('properties'), dict) else {}
        field = item_props.get('field') if isinstance(item_props.get('field'), dict) else {}
        enum = field.get('enum')
        allowed = [str(x) for x in enum] if isinstance(enum, list) else []
        desc = ' '.join((str(part) for part in (just.get('description'), field.get('description'), fn.get('description')) if part))
        if not allowed:
            allowed = _fields_from_description(desc)
        required = 'justification' in (params.get('required') or [])
        return {'present': True, 'required': required, 'allowed': allowed, 'description': desc[:900]}

    def __init__(self, problem: dict) -> None:
        env = problem['environment']
        self.binding = dict(env['binding'])
        self.policy = env['policy_view']
        self.tools = list(self.policy.get('tools') or [])
        self.tool_names = {(t.get('function') or {}).get('name') for t in self.tools if isinstance(t, dict)}
        self.claim_schema = _claim_schema(self.tools)
        self.max_steps = int(self.policy.get('max_steps') or 30)
        self.max_calls = max(1, int(self.policy.get('max_calls_per_turn') or 1))
        self.problem_id = str(problem.get('problem_id') or problem.get('id') or 'problem')
        self.brief = Brief(str(self.policy.get('query') or ''))
        self.budget = self.brief.budget
        self.stated_budget = self.brief.budget
        self.ledger = Ledger()
        self.clock = Clock(_PROBLEM_BUDGET_S)
        self.dialogue: list = []
        self.messages: list = [{'role': 'system', 'content': _CORE_RULES}, {'role': 'user', 'content': self.policy.get('query') or ''}]
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
        self.questions_asked = 0
        self.searched_twins = False
        self.card_searches = 0
        self.reads_done = False
        self.unreadable: set = set()
        self.question_waits = 0
        self.done = False
        self.verdicts: dict = {}
        self.verdict_fp = ''
        self.judge_calls = 0
        self._probes: set = set()
        self._path_pref_recovery = None
        self.model_index = 0
        provider = (getenv('INFERENCE_PROVIDER') or '').strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS['chutes']
        pinned = getenv('SANDBOX_MODEL')
        if pinned:
            self.roster = (pinned,) + tuple((m for m in self.roster if m != pinned))
    # -- catalog evidence --
    def _note(self, text: str) -> None:
        self.messages.append({'role': 'user', 'content': '[note] ' + text})
        self.dialogue.append({'role': 'assistant', 'content': '[note] ' + text[:300]})
    def _prod(self, pid: str) -> dict:
        return self.ledger.products.get(str(pid)) or {}
    def _facts(self, pid: str) -> dict:
        f = self._prod(pid).get('facts')
        return f if isinstance(f, dict) else {}
    def _title(self, pid: str) -> str:
        title = str(self._prod(pid).get('title') or '')
        if title:
            return title
        for (key, row) in self.ledger.rows.items():
            if key.startswith(str(pid) + '::'):
                return str(row.get('title') or '')
        return ''
    def _brand_of(self, pid: str) -> str:
        brand = self._prod(pid).get('brand')
        if brand:
            return str(brand)
        for (key, row) in self.ledger.rows.items():
            if key.startswith(str(pid) + '::') and row.get('brand'):
                return str(row['brand'])
        return ''
    def _options(self, pid: str, sku: str) -> str:
        return self.ledger.options.get(Ledger.key(pid, sku), '')
    def _card(self, pid: str, sku: str) -> dict:
        st = self.ledger.checked.get(Ledger.key(pid, sku)) or {}
        facts = self._facts(pid)
        compact = {}
        for (key, val) in list(facts.items())[:36]:
            text = str(val)
            compact[str(key)] = text[:240]
        return {'product_id': str(pid), 'sku': str(sku), 'title': self._title(pid)[:180], 'brand': self._brand_of(pid), 'category_path': self._prod(pid).get('category_path') or [], 'facts': compact, 'options': self._options(pid, sku)[:240], 'price': st.get('price'), 'in_stock': st.get('in_stock')}
    # -- judge --
    def _consult(self, payload: dict) -> dict:
        if self.clock.closing(80) or self.judge_calls >= 8:
            return {}
        self.judge_calls += 1
        body = json.dumps(payload, ensure_ascii=False)
        if len(body) > 12000:
            body = body[:12000] + '..."'
        messages = [{'role': 'system', 'content': _JUDGE_SYS}, {'role': 'user', 'content': body}]
        for attempt in range(2):
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            resp = _PROXY.post('/inference/chat/completions', json_data={'model': model, 'messages': messages, 'temperature': TemperatureController.for_attempt('judge', attempt)})
            if resp and isinstance(resp, dict) and resp.get('choices'):
                msg = resp['choices'][0].get('message') or {}
                data = _parse_json(str(msg.get('content') or ''))
                if data:
                    return data
            if self.clock.remaining() > 90:
                _backoff(attempt)
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return {}
    def _judge_request(self) -> None:
        data = self._consult({'stage': 'request', 'request': self.brief.goal[:2500], 'environment_notes': (self.brief.raw[len(self.brief.goal):] or '').strip()[:700], 'named_pairs': [pid + '::' + sku for (pid, sku) in self.brief.candidate_keys], 'claim_schema': {'justification_present': self.claim_schema.get('present'), 'required': self.claim_schema.get('required'), 'allowed_fields': self.claim_schema.get('allowed') or [], 'description': self.claim_schema.get('description') or ''}, 'need': {'ask': 'question to send the shopper now, or null if nothing must be asked', 'must_surface': 'how many distinct matching products search must show before buying, else 0', 'named_only': 'true if only the named pairs may be bought', 'need_reason': 'true if the shopper asked for a written reason built from labelled facts', 'need_two_listings': 'true if at least two different listings must be compared first', 'wait_for_shopper': 'true if a hidden firm requirement is still missing', 'exact_first': 'true if the first cart item must be an exact named model/color pairing', 'category': 'shelf name to filter on, or empty', 'brand': 'required or preferred maker, or empty', 'search_query': "best first catalog query using the shopper's own words and quantities so catalog values can match", 'match_words': 'title words that identify a matching product, if any', 'firm_requirements': 'list of every must-have the shopper stated, in their own words. Include each labelled-fact style requirement a product card can prove. Empty list if none. Do not invent and do not copy hidden task notes.', 'rank_axes': [{'direction': 'max or min', 'from': "the shopper's own comparison phrase, in order"}], 'claim_fields': "allowed claim-schema field names (besides price, in_stock, brand) that a written reason must include, mapped from the shopper's stated specs. Empty if no written reason.", 'brand_is_preference': 'true if a named maker is a soft preference for the first cart item', 'notes': 'one short paragraph of strategy from the shopper request: what is firm vs preference, which labelled facts must be proven before buying, and how to recover after a market change'}})
        self.brief.apply_judge(data)
        allowed_map = {str(a).lower(): str(a) for a in self.claim_schema.get('allowed') or []}
        if self.brief.claim_fields:
            self.brief.claim_fields = [allowed_map.get(f.lower(), f) for f in self.brief.claim_fields if str(f).strip()]
        ask = _ask_only(self.brief.goal)
        if not self.brief.brand:
            pref = _PREFER_RE.search(self.brief.goal)
            if pref:
                cleaned = _usable_brand(pref.group(1))
                if cleaned:
                    self.brief.brand = cleaned
                    self.brief.brand_is_preference = True
            if not self.brief.brand:
                pick = _SOFT_PICK_RE.search(ask)
                if pick:
                    cleaned = _usable_brand(pick.group(1) or pick.group(2) or '')
                    if cleaned:
                        self.brief.brand = cleaned
                        self.brief.brand_is_preference = True
        if self.brief.brand and _SOFT_PICK_RE.search(ask):
            self.brief.brand_is_preference = True
        cleaned = _usable_brand(self.brief.brand)
        if self.brief.brand and not cleaned:
            self.brief.brand = ''
        else:
            self.brief.brand = cleaned
        if self.brief.brand and self.brief.brand.lower() not in (self.brief.search_query or '').lower():
            self.brief.search_query = (self.brief.brand + ' ' + (self.brief.search_query or '')).strip()
        self.brief.need_reason = bool(_REASON_RE.search(ask))
        self.brief.exact_first = bool(self.brief.candidate_keys) or _exact_pair_ask(ask)
        if self.brief.need_reason and self.brief.brand and (not re.search('\\bprefer\\b', ask, re.I)):
            self.brief.brand_is_preference = False
        if self.budget is None and data.get('budget') is not None:
            extra = _num(data.get('budget'))
            if extra is not None:
                self.budget = extra
        if self.brief.notes:
            self.messages[0] = {'role': 'system', 'content': _CORE_RULES + '\n\nHow to handle this request:\n' + self.brief.notes}
        if self._path_pref_recovery is None:
            self._path_pref_recovery = self._pref_gates()
        if self.brief.wait_for_shopper and (not self.brief.ask):
            self.brief.ask = 'Before I search: which hardware specification is your firm requirement, and what value do you need?'
        if self._shopper_choice() and (not self.brief.ask):
            self.brief.ask = 'Quick question so I choose well: which of your preferences counts most, and in what order should I weigh the others?'
    def _judge_products(self, force: bool=False) -> None:
        if self.brief.fingerprint != self.verdict_fp:
            self.verdicts.clear()
            self.verdict_fp = self.brief.fingerprint
        keys = []
        cards = []
        for key in list(self.ledger.checked):
            (pid, _, sku) = key.partition('::')
            if not sku or str(pid) not in self.ledger.products:
                continue
            if not force and key in self.verdicts:
                continue
            keys.append(key)
            cards.append(self._card(pid, sku))
        if not cards:
            return
        data = self._consult({'stage': 'products', 'request': self.brief.goal[:2000], 'notes': self.brief.notes, 'firm_requirements': self.brief.firm_requirements, 'dead': sorted(self.ledger.dead_keys), 'first_cart_already_made': self.ledger.adds > 0, 'claim_fields': self.brief.claim_fields or self.claim_schema.get('allowed') or [], 'claim_guidance': "Use only allowed order-tool field names. If a labelled fact uses a different key, still emit the allowed name and copy the value from that labelled fact. When the shopper asked for a written reason, emit ONLY the claim_fields the shopper's stated specs need, plus price, in_stock, and brand. Do not add extra allowed specifications the shopper did not ask to justify. Never emit a labelled key the order tool does not allow.", 'rank_axes': self.brief.rank_axes, 'rank_guidance': "If the request lists named product_id::sku pairs, only those pairs count. For every rank_axis, read a number from this variant's option text first, then labelled facts, then the title. Prefer a variant-specific advertised maximum when the shopper asked for the highest maximum. After a market change, dead pairs stay dead; remaining pairs are re-scored on current price and stock. Claims must use claim_fields names, even when the labelled fact key is different; copy the value from the labelled fact.", 'variants': cards[:12], 'need': {'variants': [{'product_id': '', 'sku': '', 'ok': 'true only if labelled facts (then option text) prove every firm_requirement for this exact variant. Title-only is false when a written reason is required.', 'rank': '1 = best among the ok ones on rank_axes', 'relax': '0 = preferred brand or exact named pairing, higher = more relaxed', 'axis_values': ['one number per rank_axis, or null'], 'claims': [{'field': 'a claim_fields / allowed field name, never a raw extra key', 'value': 'copied from labelled facts or live price/stock'}], 'why': 'one short sentence'}]}})
        rows = data.get('variants')
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            (pid, sku) = (str(row.get('product_id') or ''), str(row.get('sku') or ''))
            if not pid or not sku:
                continue
            claims = []
            for claim in row.get('claims') or []:
                if isinstance(claim, dict) and claim.get('field') and claim.get('value'):
                    (field, value) = (str(claim['field']), str(claim['value']))
                    if self._claim_allowed(field, pid) and self._grounded(pid, sku, field, value):
                        claims.append({'field': field, 'value': value})
            axis_values = []
            raw_axes = row.get('axis_values')
            if isinstance(raw_axes, list):
                for item in raw_axes[:6]:
                    axis_values.append(_num(item))
            self.verdicts[Ledger.key(pid, sku)] = {'ok': bool(row.get('ok')), 'rank': _num(row.get('rank')) if row.get('rank') is not None else 99.0, 'relax': int(_num(row.get('relax')) or 0), 'claims': claims, 'axis_values': axis_values, 'why': str(row.get('why') or '')[:240]}
    def _claim_allowed(self, field: str, pid: str | None=None) -> bool:
        field_l = str(field or '').strip().lower()
        if not field_l:
            return False
        if field_l in {'price', 'in_stock', 'brand'}:
            return True
        allowed = {str(a).lower() for a in self.claim_schema.get('allowed') or []}
        asked = {str(f).lower() for f in self.brief.claim_fields}
        if field_l in allowed or field_l in asked:
            return True
        desc = self.claim_schema.get('description') or ''
        return bool(re.search('\\b' + re.escape(field_l) + '\\b', desc, re.I))
    def _spec_blob(self, pid: str, sku: str, include_title: bool=False) -> str:
        parts = [str(v) for v in self._facts(pid).values()]
        parts.append(self._options(pid, sku))
        if include_title:
            parts.append(self._title(pid))
        return ' '.join(parts).lower()
    def _grounded(self, pid: str, sku: str, field: str, value: str) -> bool:
        field_l = str(field or '').strip().lower()
        value_s = str(value or '').strip()
        if not field_l or not value_s:
            return False
        st = self.ledger.status(pid, sku) or {}
        if field_l == 'price':
            if value_s.lower().startswith('at most'):
                cap_m = re.search('[\\d,.]+', value_s[7:])
                cap = _num(cap_m.group(0)) if cap_m else None
                return self.budget is not None and cap is not None and (abs(cap - self.budget) < 0.05)
            price = st.get('price')
            observed = _num(value_s)
            return price is not None and observed is not None and (abs(price - observed) < 0.05)
        if field_l == 'in_stock':
            want = value_s.lower() in ('true', '1', 'yes', 'in stock')
            have = bool(st.get('in_stock'))
            return have == want
        if field_l == 'brand':
            brand = self._brand_of(pid)
            return bool(brand) and value_s.lower() in brand.lower()
        if field_l == 'category':
            path = self._prod(pid).get('category_path') or []
            blob = ' '.join((str(x) for x in path)).lower() if isinstance(path, list) else str(path).lower()
            return bool(blob) and value_s.lower() in blob
        facts = self._facts(pid)
        value_l = value_s.lower()
        for (key, fact) in facts.items():
            fact_s = str(fact).lower()
            if str(key).lower() == field_l and (value_l in fact_s or fact_s in value_l):
                return True
        fact_blob = self._spec_blob(pid, sku, include_title=not self.brief.need_reason)
        if value_l in fact_blob:
            return True
        num_m = re.match('[\\d.]+', value_s)
        want_n = _num(num_m.group(0)) if num_m else None
        if want_n is None:
            return False
        want_toks = _field_tokens(field_l)
        for (key, fact) in facts.items():
            if want_toks and (not (_token_overlap(want_toks, _field_tokens(str(key))) or _token_overlap(want_toks, _tokens(str(fact))))):
                continue
            if any((abs(n - want_n) < 0.05 for (n, _neigh) in _numbers_from_text(str(fact)))):
                return True
            bare = _num(fact)
            if bare is not None and abs(bare - want_n) < 0.05:
                return True
        return False
    def _axis_is_price(self, axis: dict) -> bool:
        hint = str(axis.get('from') or '').lower()
        return any((w in hint for w in ('price', 'cost', 'cheap', 'budget')))
    def _axis_want_tokens(self, axis: dict) -> set:
        hint = str(axis.get('from') or '')
        want = _hint_tokens(hint)
        for fname in self.claim_schema.get('allowed') or []:
            ft = _field_tokens(str(fname))
            if _token_overlap(ft, want):
                want |= ft
        return want
    def _axis_number(self, pid: str, sku: str, axis: dict, idx: int | None=None) -> float | None:
        if self._axis_is_price(axis):
            st = self.ledger.checked.get(Ledger.key(pid, sku)) or {}
            return st.get('price')
        want = self._axis_want_tokens(axis)
        facts = self._facts(pid)
        (best_key, best_score) = (None, 0)
        for (key, val) in facts.items():
            score = _token_overlap(_field_tokens(str(key)), want)
            if score > best_score:
                (best_key, best_score) = (str(key), score)
        def pick(text: str, require_overlap: bool) -> float | None:
            nums = []
            for (val, neigh) in _numbers_from_text(text or ''):
                if require_overlap and want and (not neigh & want):
                    continue
                nums.append(val)
            if not nums:
                return None
            return max(nums) if str(axis.get('direction') or 'max') != 'min' else min(nums)
        opt = self._options(pid, sku)
        fact_text = str(facts.get(best_key) or '') if best_key else ''
        fact_n = pick(fact_text, require_overlap=False) if fact_text else None
        direction = str(axis.get('direction') or 'max')
        opt_overlap = pick(opt, require_overlap=True) if opt else None
        if opt_overlap is not None:
            return opt_overlap
        if opt and fact_n is not None:
            overrides = []
            for (val, _neigh) in _numbers_from_text(opt):
                (bigger, smaller) = (max(val, fact_n), min(val, fact_n))
                if smaller <= 0 or bigger / smaller > 4:
                    continue
                if direction == 'min' and val <= fact_n + _EPS:
                    overrides.append(val)
                elif direction != 'min' and val + _EPS >= fact_n:
                    overrides.append(val)
            if overrides:
                return max(overrides) if direction != 'min' else min(overrides)
        if fact_n is not None:
            return fact_n
        got = pick(self._title(pid), require_overlap=True)
        if got is not None:
            return got
        if not best_score:
            suffix = _suffix_numbers(opt) + _suffix_numbers(self._title(pid))
            if suffix:
                return max(suffix) if direction != 'min' else min(suffix)
        judged = self.verdicts.get(Ledger.key(pid, sku)) or {}
        av = judged.get('axis_values') or []
        if idx is not None and idx < len(av) and (av[idx] is not None) and (av[idx] < 1000000.0):
            return av[idx]
        return None
    def _axis_sort_tuple(self, row: dict) -> tuple | None:
        axes = self.brief.rank_axes
        if not axes:
            return None
        out = []
        for (i, axis) in enumerate(axes):
            n = self._axis_number(row['pid'], row['sku'], axis, i)
            if n is None:
                n = -1000000000000.0 if str(axis.get('direction') or 'max') != 'min' else 1000000000000.0
            out.append(-n if str(axis.get('direction') or 'max') != 'min' else n)
        return tuple(out)
    def _has_rank_signal(self, pid: str, sku: str) -> bool:
        axes = [ax for ax in self.brief.rank_axes if not self._axis_is_price(ax)]
        if not axes:
            return bool(self.brief.rank_axes)
        for (i, axis) in enumerate(self.brief.rank_axes):
            if self._axis_is_price(axis):
                continue
            if self._axis_number(pid, sku, axis, i) is not None:
                return True
        return False
    # -- isolation (shopper signals, not family names) --
    def _justification_mode(self) -> bool:
        if not self.brief.need_reason:
            return False
        if self.brief.named_only or self.brief.rank_axes:
            return False
        if self._shopper_choice():
            return False
        return True
    def _shopper_choice(self) -> bool:
        return bool(self.brief.shopper_choice) and (not (self.brief.named_only or self.brief.rank_axes))
    def _recovery_mode(self) -> bool:
        if self.brief.named_only or self.brief.rank_axes or self._shopper_choice():
            return False
        if not self.brief.exact_first:
            return False
        return bool(self._pairing_prefer())
    def _pairing_prefer(self) -> list:
        specs = set(self._shopper_spec_numbers())
        out = []
        for t in self.brief.prefer:
            if not any((ch.isdigit() for ch in t)):
                continue
            m = re.search('(\\d+(?:\\.\\d+)?)', t)
            if not m:
                continue
            n = _num(m.group(1))
            if t[:1].isalpha():
                out.append(t)
                continue
            if n is not None and any((abs(n - s) < 0.0001 for s in specs)):
                continue
            out.append(t)
        return out
    def _shopper_soft_specs(self) -> list:
        text = self._shopper_ask_text()
        out = []
        for m in re.finditer('\\bwith\\s+([A-Za-z][A-Za-z0-9 \\-]{1,24}?)(?:,|\\.| at | up to | under |$)', text or '', re.I):
            phrase = re.sub('\\s+', ' ', m.group(1).strip())
            if phrase:
                out.append(phrase)
        return out
    def _pref_gates(self) -> bool:
        spoken_axes = _axes_from_text(self._original_ask())
        return bool(self.brief.brand_is_preference and (self._shopper_measurements() or self.brief.firm_requirements or self._shopper_soft_specs()) and (not self.brief.named_only) and (not spoken_axes) and (not self._shopper_choice()) and (not self.brief.exact_first) and (not self.brief.need_reason) and (not self.brief.wait_for_shopper))
    def _pref_recovery(self) -> bool:
        if self._path_pref_recovery is not None:
            return self._path_pref_recovery
        return self._pref_gates()
    def _hold_pref_talk(self) -> bool:
        return bool(self._pref_recovery() and self.ledger.adds == 0 and (not self.ledger.event_target))
    def _original_ask(self) -> str:
        goal = self.brief.goal or ''
        return re.split('\\n\\s*task rules\\s*:', goal, maxsplit=1, flags=re.I)[0].strip()
    # -- preference / first-cart recovery --
    def _asked_category(self) -> str:
        text = self._original_ask()
        m = re.search('\\b(?:need|want|looking for|buy|get me|find me)\\s+(?:an?\\s+|the\\s+)?([A-Za-z][A-Za-z0-9\\- ]{2,40}?)(?=\\s*,|\\s+with\\b|\\s+at or under|\\s+at or below|\\s+up to\\b)', text or '', re.I)
        if m:
            cat = re.sub('\\s+', ' ', m.group(1)).strip()
            cat = re.sub('^(?:to\\s+)?(?:buy|get|find)\\s+(?:an?\\s+|the\\s+)?', '', cat, flags=re.I).strip()
            if cat:
                return cat
        m = re.search('\\b(?:need|want|looking for|buy)\\s+(?:an?\\s+|the\\s+)?([A-Za-z][A-Za-z\\-]{2,24})\\s+with\\b', text or '', re.I)
        if m:
            return m.group(1).strip()
        m = re.search('\\b(?:need|want|looking for|buy|get me|find me)\\b.*?\\b([A-Za-z][A-Za-z\\-]{2,24})\\s+(?:at or under|at or below|in stock)', text or '', re.I)
        return m.group(1).strip() if m else ''
    def _shopper_negatives(self) -> list:
        text = self._shopper_ask_text()
        phrases = []
        m = re.search('\\breject(?:ing)?\\s+(?:conflicting\\s+)?(.+?)(?:\\bevidence\\b|[.]|$)', text or '', re.I)
        blob = m.group(1) if m else ''
        for m in re.finditer('\\bnot\\s+(?:a\\s+|an\\s+)?([a-z][a-z0-9\\- ]{2,60}?)(?:\\.|,|;|$)', text or '', re.I):
            blob += ', ' + m.group(1)
        for part in re.split('\\s*,\\s*|\\s+or\\s+', blob, flags=re.I):
            for alt in part.split('/'):
                phrase = re.sub('\\s+', ' ', alt.strip().lower().replace('-', ' '))
                phrase = re.sub('^(?:or|the|a|an)\\s+', '', phrase)
                if len(phrase) >= 3:
                    phrases.append(phrase)
        return phrases
    def _shopper_category(self) -> str:
        if self._pref_recovery() or self._recovery_mode():
            asked = self._asked_category()
            if asked:
                return asked
        if self.brief.category:
            return self.brief.category
        return self._asked_category()
    def _device_hold(self, pid: str) -> bool:
        if not self._pref_recovery() or not self.ledger.event_target:
            return True
        refs: list = []
        refs.append(self.ledger.event_target.split('::', 1)[0])
        for (key, judged) in self.verdicts.items():
            if judged.get('ok'):
                refs.append(str(key).split('::', 1)[0])
        device_keys = {'battery', 'device_operating_system', 'operating_system'}
        must: set = set()
        for ref in refs:
            if ref == str(pid):
                continue
            must |= {str(k).lower() for k in self._facts(ref) if str(k).lower() in device_keys}
        if not must:
            return True
        have = {str(k).lower() for k in self._facts(pid)}
        return bool(must & have)
    def _pref_phrase_hold(self, blob: str) -> bool:
        skip = _tokens(self.brief.brand) | _tokens(self._shopper_category()) | {'php', 'usd', 'the', 'and', 'for', 'with'}
        phrases = self._shopper_soft_specs() + list(self.brief.firm_requirements)
        hay = re.sub('[^a-z0-9]+', ' ', (blob or '').lower()).strip()
        compact = re.sub('[^a-z0-9]+', '', hay)
        if not phrases:
            return True
        for req in phrases:
            pn = re.sub('[^a-z0-9]+', ' ', str(req).lower()).strip()
            pc = re.sub('[^a-z0-9]+', '', str(req).lower())
            if pn and pn in hay:
                return True
            if pc and len(pc) >= 5 and pc in compact:
                return True
            words = [w for w in pn.split() if w not in skip]
            if len(words) >= 2:
                for i in range(len(words) - 1):
                    for j in range(i + 2, len(words) + 1):
                        if ' '.join(words[i:j]) in hay:
                            return True
            elif len(words) == 1 and len(words[0]) > 3 and re.search('\\b' + re.escape(words[0]) + '\\b', hay):
                return True
        return False
    def _pref_spec_hold(self, pid: str, sku: str) -> bool:
        pairs = self._shopper_measure_pairs()
        blob = ' '.join([self._title(pid), self._options(pid, sku), ' '.join((str(v) for v in self._facts(pid).values()))])
        if pairs:
            return all((_measure_mentioned(blob, n, unit) for (n, unit) in pairs))
        return self._pref_phrase_hold(blob)
    def _pref_keep(self, pid: str, sku: str) -> bool:
        return bool(self._pref_spec_hold(pid, sku) and self._class_hold(pid) and self._device_hold(pid))
    def _pref_replacements(self) -> list:
        if not self._pref_recovery() or not self.ledger.event_target:
            return []
        out = []
        seen = set()
        for pid in list(self.ledger.products):
            if str(pid) in self.ledger.dead_products:
                continue
            for v in (self._prod(pid).get('variants') or []):
                sku = v.get('sku')
                if not sku:
                    continue
                key = Ledger.key(pid, sku)
                if key in seen or key in self.ledger.dead_keys:
                    continue
                seen.add(key)
                price = _num(v.get('price'))
                st = self.ledger.checked.get(key) or {}
                if price is None:
                    price = st.get('price')
                if self.budget is not None and price is not None and (price > self.budget + _EPS):
                    continue
                if not self._pref_keep(str(pid), str(sku)):
                    continue
                out.append({'pid': str(pid), 'sku': str(sku), 'key': key, 'price': price, 'ok': True, 'live': True, 'viewed': True})
        out.sort(key=lambda c: (0 if self._preferred_brand_hit(c['pid']) else 1, c['price'] if c['price'] is not None else 1000000000000.0, c['key']))
        return out
    def _pref_live(self) -> dict | None:
        if not self._pref_recovery() or not self.ledger.event_target:
            return None
        good = [c for c in self._candidates() if c['live'] and c['viewed'] and self._pref_keep(c['pid'], c['sku'])]
        good.sort(key=lambda c: (0 if self._preferred_brand_hit(c['pid']) else 1, c['price'] if c['price'] is not None else 1000000000000.0, c['key']))
        if good:
            return good[0]
        repl = self._pref_replacements()
        return repl[0] if repl else None
    def _pref_branded_pending(self) -> list:
        if not self.brief.brand:
            return []
        want = _tokens(self.brief.brand)
        keeper_pids = set()
        for key in self.ledger.inspected:
            (pid, _, sku) = key.partition('::')
            if not sku or str(pid) in self.ledger.dead_products or str(pid) not in self.ledger.products:
                continue
            st = self.ledger.checked.get(key) or {}
            if not st.get('in_stock'):
                continue
            if self._pref_keep(pid, sku) and self._preferred_brand_hit(pid):
                keeper_pids.add(str(pid))
        pending = []
        seen = set()
        for (key, row) in self.ledger.rows.items():
            (pid, _, sku) = key.partition('::')
            pid = str(pid)
            if not sku or pid in seen or pid in keeper_pids or pid in self.ledger.dead_products:
                continue
            title = str(row.get('title') or '')
            if not (want <= _tokens(title) or want <= _tokens(str(row.get('brand') or ''))):
                continue
            st = self.ledger.checked.get(key) or {}
            price = st.get('price')
            if price is None:
                price = _num(row.get('price'))
            if self.budget is not None and price is not None and (price > self.budget + _EPS):
                continue
            seen.add(pid)
            pending.append(pid)
        for pid in list(self.ledger.products):
            pid = str(pid)
            if pid in seen or pid in keeper_pids or pid in self.ledger.dead_products:
                continue
            if self._preferred_brand_hit(pid):
                seen.add(pid)
                pending.append(pid)
        for pid in self._pref_unread():
            if pid not in seen and pid not in keeper_pids:
                pending.append(pid)
        return pending
    def _pref_first_cart(self) -> dict | None:
        if not self._pref_recovery() or self.ledger.adds or self.ledger.event_target:
            return None
        keepers = []
        seen = set()
        for key in self.ledger.inspected:
            (pid, _, sku) = key.partition('::')
            if not sku or key in seen or key in self.ledger.dead_keys or str(pid) in self.ledger.dead_products:
                continue
            st = self.ledger.checked.get(key) or {}
            if not st.get('in_stock'):
                continue
            price = st.get('price')
            if self.budget is not None and price is not None and (price > self.budget + _EPS):
                continue
            if str(pid) not in self.ledger.products:
                continue
            if not self._pref_keep(pid, sku):
                continue
            if self.brief.brand and (not self._preferred_brand_hit(pid)):
                continue
            seen.add(key)
            keepers.append({'pid': str(pid), 'sku': str(sku), 'key': key, 'price': price, 'ok': True})
        if (not self._late()) and self.brief.brand and len(keepers) < 2 and self._pref_branded_pending():
            return None
        if not keepers:
            return None
        keepers.sort(key=lambda c: (-(c['price'] if c['price'] is not None else 0.0), c['key']))
        return keepers[0]
    def _measurement_phrase(self) -> str:
        text = self._original_ask()
        for m in re.finditer('(\\d+(?:\\.\\d+)?)\\s*[-–]?\\s*([A-Za-z]{2,8})', text or ''):
            n = _num(m.group(1))
            unit = (m.group(2) or '').lower()
            if n is None:
                continue
            if unit in {'php', 'usd', 'eur', 'sgd'}:
                continue
            if self.budget is not None and abs(n - self.budget) < 0.05:
                continue
            if n >= 1000 and unit not in {'p', 'k', 'hz'}:
                continue
            return m.group(0).strip()
        return ''
    def _inspect_measured(self) -> tuple | None:
        if not self._pref_recovery():
            return None
        if self.ledger.event_target and self._pref_live() is not None:
            return None
        if not self.ledger.event_target and (self.ledger.adds or self._pref_first_cart() is not None):
            return None
        keys = list(self.ledger.checked)
        for pid in list(self.ledger.products):
            for v in (self._prod(pid).get('variants') or []):
                sku = v.get('sku')
                if sku:
                    keys.append(Ledger.key(pid, sku))
        seen = set()
        for key in keys:
            if not key or key in seen:
                continue
            seen.add(key)
            (pid, _, sku) = key.partition('::')
            if not sku or str(pid) in self.ledger.dead_products:
                continue
            if key in self.ledger.dead_keys or str(pid) not in self.ledger.products:
                continue
            if key in self.ledger.inspected:
                continue
            st = self.ledger.checked.get(key) or {}
            price = st.get('price')
            if price is None:
                for v in (self._prod(pid).get('variants') or []):
                    if str(v.get('sku') or '') == sku:
                        price = _num(v.get('price'))
                        break
            if price is not None and self.budget is not None and (price > self.budget + _EPS):
                continue
            if not self._pref_keep(pid, sku):
                continue
            if (not self.ledger.event_target) and self.brief.brand and self._honor_brand_now() and (not self._preferred_brand_hit(pid)):
                continue
            args = {'product_id': str(pid), 'sku': str(sku)}
            if self._unused('inspect_stock', args):
                return ('inspect_stock', args)
        return None
    def _pair_relax(self, pid: str, sku: str) -> int:
        blob = _tokens(' '.join([self._title(pid), self._brand_of(pid), self._options(pid, sku), ' '.join((str(v) for v in self._facts(pid).values()))]))
        prefer = [t for t in self.brief.prefer if t not in {'php', 'the', 'and'}]
        digit = [t for t in prefer if any((ch.isdigit() for ch in t))]
        words = [t for t in prefer if t not in digit]
        same_model = bool(digit) and any((t in blob for t in digit))
        same_words = sum((1 for t in words if t in blob))
        if same_model and words and (same_words == len(words)):
            return 0
        if same_model and (not words or same_words):
            return 0 if not words else 1
        if same_model:
            return 1
        if words and same_words == len(words):
            return 2
        return 3
    def _preferred_brand_hit(self, pid: str) -> bool:
        want = _tokens(self.brief.brand)
        if not want:
            return False
        catalog = _tokens(self._brand_of(pid))
        if self._justification_mode() and catalog:
            return want <= catalog
        return bool(want <= catalog or want <= _tokens(self._title(pid)))
    def _honor_brand_now(self) -> bool:
        if self._pref_recovery():
            return self.ledger.adds == 0 and bool(self.brief.brand) and (not self._late())
        return self.ledger.adds == 0 and self.brief.brand_is_preference and bool(self.brief.brand) and (not self._late()) and (self.turn < self.max_steps * 0.6)
    def _shopper_measure_pairs(self) -> list:
        wants = []
        for m in re.finditer('(\\d+(?:\\.\\d+)?)\\s*[-–]?\\s*([A-Za-z]{2,8})?', self._original_ask() or ''):
            n = _num(m.group(1))
            unit = (m.group(2) or '').lower()
            if n is None or not unit:
                continue
            if unit in {'php', 'usd', 'eur', 'sgd'}:
                continue
            if self.budget is not None and abs(n - self.budget) < 0.05:
                continue
            if n >= 1000 and unit not in {'hz', 'khz', 'mhz', 'ghz'}:
                continue
            wants.append((n, unit))
        return wants
    def _shopper_measurements(self) -> list:
        return [n for (n, _unit) in self._shopper_measure_pairs()]
    def _measurement_grounded(self, pid: str, sku: str) -> bool:
        wants = self._shopper_measurements()
        if not wants:
            return True
        opt_nums = [n for (n, _neigh) in _numbers_from_text(self._options(pid, sku))]
        for n in wants:
            found = any((abs(n - h) < 0.0001 for h in opt_nums))
            for (_key, raw) in self._facts(pid).items():
                text = str(raw or '').strip()
                nums = [x for (x, _neigh) in _numbers_from_text(text)]
                bare = _num(text)
                if bare is not None:
                    nums.append(bare)
                if not any((abs(n - h) < 0.0001 for h in nums)):
                    continue
                if re.match('^\\d+(\\.\\d+)?(\\s*[a-z]+)?', text.lower()):
                    found = True
                    break
            if not found:
                return False
        return True
    def _kind_hold(self, pid: str) -> bool:
        skip = {'condition', 'warranty_type', 'warranty_duration', 'material', 'wattage', 'navigation', 'description_device_type'}
        refs: list = []
        if self.ledger.event_target:
            refs.append(self.ledger.event_target.split('::', 1)[0])
        for (key, judged) in self.verdicts.items():
            if judged.get('ok'):
                refs.append(str(key).split('::', 1)[0])
        must: set = set()
        for ref in refs:
            must |= {k.lower() for k in self._facts(ref) if str(k).lower() not in skip}
        must &= {'size', 'display', 'battery', 'device_operating_system', 'operating_system'}
        if not must:
            return True
        have = {k.lower() for k in self._facts(pid)}
        return bool(must & have)
    def _firm_numbers_hold(self, pid: str, sku: str) -> bool:
        if not self._measurement_grounded(pid, sku):
            return False
        wants = self._shopper_measurements()
        blob = ' '.join((str(v) for v in self._facts(pid).values())) + ' ' + self._options(pid, sku)
        blob_l = blob.lower()
        if wants:
            have = [n for (n, _neigh) in _numbers_from_text(blob)]
            bare = [_num(v) for v in self._facts(pid).values()]
            have.extend((n for n in bare if n is not None))
            if not all((any((abs(n - h) < 0.0001 for h in have)) for n in wants)):
                return False
        specish = []
        for text in [self.brief.spoken] + list(self.brief.firm_requirements):
            for tok in _tokens(text):
                if tok in {'php', 'usd', 'the', 'and', 'for', 'with', 'inch', 'inches'}:
                    continue
                if any((ch.isdigit() for ch in tok)) or 2 <= len(tok) <= 3:
                    specish.append(tok)
        specish = [t for t in specish if t not in _tokens(self.brief.brand) and t not in _tokens(self.brief.category)]
        if specish and (not any((t in blob_l or t.rstrip('p') in blob_l for t in specish if not t.isdigit()))):
            if any((not t.isdigit() and (not t.endswith('p')) for t in specish)):
                letters = [t for t in specish if t.isalpha()]
                if letters and (not any((t in blob_l for t in letters))):
                    return False
        return True
    def _shopper_ask_text(self) -> str:
        goal = self.brief.goal or ''
        ask = re.split('\\n\\s*task rules\\s*:', goal, maxsplit=1, flags=re.I)[0].strip()
        extra: list = []
        for m in self.messages:
            if m.get('role') != 'user':
                continue
            text = str(m.get('content') or '').strip()
            if not text or text.startswith('[note]'):
                continue
            if text == goal or (ask and text.startswith(ask[:60])):
                continue
            extra.append(text)
        firms: list = []
        for req in self.brief.firm_requirements:
            chunk = re.split('\\n\\s*task rules\\s*:', str(req), maxsplit=1, flags=re.I)[0].strip()
            if chunk:
                firms.append(chunk)
        return ' '.join([p for p in [ask] + extra + firms if p])
    def _shopper_spec_numbers(self) -> list:
        text = self._shopper_ask_text()
        out: list = []
        for m in re.finditer('(\\d+(?:\\.\\d+)?)([A-Za-z]{0,8})', text or ''):
            n = _num(m.group(1))
            unit = (m.group(2) or '').lower()
            if n is None:
                continue
            if unit in {'php', 'usd', 'eur', 'sgd'}:
                continue
            if self.budget is not None and abs(n - self.budget) < 0.05:
                continue
            if n >= 1000 and unit not in {'p', 'k', 'hz'}:
                continue
            out.append(n)
        return out
    def _asked_spec_names(self) -> list:
        skip = {'price', 'in_stock', 'brand', 'category'}
        names: list = []
        seen: set = set()
        for name in list(self.brief.claim_fields) + list(self.claim_schema.get('allowed') or []):
            n = str(name or '').strip()
            if not n or n.lower() in skip or n.lower() in seen:
                continue
            seen.add(n.lower())
            names.append(n)
        for tok in _fields_from_description(self.claim_schema.get('description') or ''):
            if tok in skip or tok in seen:
                continue
            seen.add(tok)
            names.append(tok)
        return names
    def _asked_spec_tokens(self) -> set:
        toks: set = set()
        for name in self._asked_spec_names():
            toks |= _field_tokens(name)
        return toks
    def _shopper_class_tokens(self) -> set:
        out: set = set()
        skip = _tokens(self.brief.brand) | _tokens(self._shopper_category()) | {'php', 'usd', 'the', 'and', 'for'}
        for tok in _tokens(self._shopper_ask_text()):
            if tok in skip:
                continue
            if any((ch.isdigit() for ch in tok)) or (2 <= len(tok) <= 3 and tok.isalpha()):
                out.add(tok)
        return out
    def _is_asked_spec_fact(self, key: str, raw: str='') -> bool:
        kl = str(key or '').strip().lower()
        if not kl or kl in {'price', 'in_stock', 'brand', 'category'}:
            return False
        asked = {n.lower() for n in self._asked_spec_names()}
        want = self._asked_spec_tokens()
        if kl in asked or (want and _token_overlap(_field_tokens(kl), want)):
            return True
        text = str(raw or '').strip().lower()
        if not text:
            return False
        spec_toks = self._shopper_class_tokens()
        if spec_toks and _token_overlap(_tokens(text), spec_toks):
            return True
        have_nums = [n for (n, _neigh) in _numbers_from_text(text)]
        return any((abs(n - s) < 0.0001 for n in have_nums for s in self._shopper_spec_numbers()))
    def _fact_is_res_class(self, raw: str) -> bool:
        text = str(raw or '').lower()
        return bool(re.search('\\b\\d{3,4}p\\b', text) or re.search('\\b\\d{3,4}\\s*[x×*]\\s*\\d{3,4}\\b', text))
    def _labelled_spec_conflict(self, pid: str, sku: str) -> bool:
        wants = self._shopper_spec_numbers()
        if not wants:
            return False
        allowed_toks: set = set()
        for name in self.claim_schema.get('allowed') or []:
            allowed_toks |= _field_tokens(str(name))
        spoken = self._shopper_ask_text()
        shopper_res = bool(self._justification_mode() and re.search('\\b\\d{3,4}p\\b', spoken or ''))
        for (key, raw) in self._facts(pid).items():
            if self._justification_mode():
                if not (self._is_asked_spec_fact(key, raw) or (shopper_res and self._fact_is_res_class(raw))):
                    continue
            else:
                kl = str(key).lower()
                if kl not in {'size', 'display'} and (not _token_overlap(_field_tokens(kl), allowed_toks)):
                    continue
            have = [n for (n, _neigh) in _numbers_from_text(str(raw))]
            bare = _num(raw)
            if bare is not None:
                have.append(bare)
            if not have:
                continue
            if self._justification_mode():
                small = [h for h in have if h < 100]
                want_small = [w for w in wants if w < 100]
                if want_small and len(small) >= 2 and re.search('\\d+(?:\\.\\d+)?\\s*[-–]\\s*\\d+', str(raw)) and (not all((any((abs(s - w) < 0.0001 for w in want_small)) for s in small))):
                    return True
            for sn in wants:
                band = [h for h in have if (sn < 100) == (h < 100)]
                if band and (not any((abs(sn - h) < 0.0001 for h in band))):
                    return True
        return False
    def _class_hold(self, pid: str) -> bool:
        path = self._prod(pid).get('category_path') or []
        leaf = str(path[-1]) if path else ''
        typed = []
        for (key, raw) in self._facts(pid).items():
            kl = str(key).lower()
            if 'device_type' in kl or kl in {'phone_type', 'product_type', 'item_type'}:
                typed.append(str(raw or ''))
        body = ' '.join([self._title(pid)] + typed).lower().replace('-', ' ')
        hay = ' '.join([body, leaf]).lower().replace('-', ' ')
        neg_hay = body if self._recovery_mode() else hay
        for phrase in self._shopper_negatives():
            stem = phrase[:-3] if phrase.endswith('ies') else (phrase[:-1] if phrase.endswith('s') and len(phrase) > 4 else phrase)
            if stem and re.search('\\b' + re.escape(stem) + '(?:s|ies)?\\b', neg_hay):
                return False
            if self._recovery_mode():
                bits = [b for b in stem.split() if b]
                if len(bits) >= 2:
                    head = ' '.join(bits[:2])
                    if len(head) >= 4 and re.search('\\b' + re.escape(head) + '\\b', neg_hay):
                        return False
        cat = _tokens(self._shopper_category())
        stems = set(cat)
        for tok in list(cat):
            if tok.endswith('s') and len(tok) > 4:
                stems.add(tok[:-1])
        type_vals = [t.lower() for t in typed if t.strip()]
        blob = _tokens(self._title(pid) + ' ' + ' '.join(type_vals) + ' ' + ' '.join((str(v) for v in self._facts(pid).values())) + ' ' + leaf)
        joined = hay
        if self._pref_recovery():
            if not cat:
                return True
            need = {(s[:-1] if s.endswith('s') and len(s) > 4 else s) for s in cat}
            return all((s in blob or s in joined for s in need))
        if self._recovery_mode():
            cleaned = re.sub('\\bnot\\s+(?:a\\s+|an\\s+)?[^.;]+', ' ', self._shopper_ask_text() or '', flags=re.I)
            forms = [re.sub('[\\s]+', '-', m.group(0).lower()) for m in re.finditer('\\b(?:over|on|in)[\\s-]+ears?\\b', cleaned, re.I)]
            if forms and (not any((f.replace('-', ' ') in hay or f in hay.replace(' ', '-') for f in forms))):
                return False
            if cat:
                path_joined = ' '.join((str(x) for x in path)).lower().replace('-', ' ')
                type_join = ' '.join(type_vals)
                need = {(s[:-1] if s.endswith('s') and len(s) > 4 else s) for s in cat}
                if need and (not all((s in path_joined or s in type_join for s in need))):
                    return False
            return True
        if not cat:
            return True
        if not type_vals:
            if not (self._justification_mode() or self._recovery_mode()):
                return True
            return bool(stems & blob) or any((s in joined for s in stems))
        blob = _tokens(' '.join(type_vals))
        joined = ' '.join(type_vals)
        return bool(stems & blob) or any((s in joined for s in stems))
    def _claim_value_ok(self, value: str) -> bool:
        wants = self._shopper_spec_numbers()
        have = [n for (n, _neigh) in _numbers_from_text(value)]
        bare = _num(value)
        if bare is not None:
            have.append(bare)
        if not wants or not have:
            return True
        for sn in wants:
            band = [h for h in have if (sn < 100) == (h < 100)]
            if band and (not any((abs(sn - h) < 0.0001 for h in band))):
                return False
        return True
    def _asked_spec_blob(self, pid: str) -> str:
        spec_vals = []
        shopper_res = bool(re.search('\\b\\d{3,4}p\\b', self._shopper_ask_text() or ''))
        for (key, raw) in self._facts(pid).items():
            if self._is_asked_spec_fact(key, raw) or (shopper_res and self._fact_is_res_class(raw)):
                spec_vals.append(str(raw or '').lower())
        return ' '.join(spec_vals)
    def _asked_spec_class_conflict(self, pid: str) -> bool:
        spoken = self._shopper_ask_text().lower()
        shopper_p = {int(m.group(1)) for m in re.finditer('\\b(\\d{3,4})p\\b', spoken)}
        if not shopper_p:
            return False
        text = self._asked_spec_blob(pid)
        card_p = {int(m.group(1)) for m in re.finditer('\\b(\\d{3,4})p\\b', text)}
        shopper_class = set(re.findall('\\b\\d{3,4}p\\b|\\b\\d+k\\b', spoken))
        card_class = set(re.findall('\\b\\d{3,4}p\\b|\\b\\d+k\\b', text))
        return bool(card_p - shopper_p) or bool(card_class - shopper_class)
    def _reason_spec_hold(self, pid: str, sku: str) -> bool:
        if self._labelled_spec_conflict(pid, sku):
            return False
        if self._asked_spec_class_conflict(pid):
            return False
        if self._firm_numbers_hold(pid, sku):
            return True
        blob = self._asked_spec_blob(pid)
        if not blob:
            return False
        spoken = self._shopper_ask_text().lower()
        for tok in self._shopper_class_tokens():
            if tok.isalpha() and re.search('\\b' + re.escape(tok) + '\\b', blob):
                return True
            if any((ch.isdigit() for ch in tok)) and (tok in blob.replace(' ', '') or tok in blob):
                return True
        for m in re.finditer('(\\d+(?:\\.\\d+)?)p\\b', spoken):
            token = m.group(0).lower()
            if token in blob.replace(' ', '') or token in blob:
                return True
        return False
    def _clean_claim_value(self, value: str) -> str:
        raw = str(value or '').strip()
        parts = [p.strip() for p in re.split('[,;/]', raw) if p.strip()]
        if len(parts) <= 1:
            return raw
        spoken_toks = _tokens(self._shopper_ask_text())
        spoken_nums = set(self._shopper_spec_numbers())
        best = None
        for part in parts:
            if not self._claim_value_ok(part):
                continue
            score = _token_overlap(_tokens(part), spoken_toks) + len({n for (n, _neigh) in _numbers_from_text(part)} & spoken_nums)
            if part.lower() in spoken_toks and (not any((ch.isdigit() for ch in part))):
                score += 3
            if best is None or score > best[0]:
                best = (score, part)
        return best[1] if best else parts[0]
    def _with_shopper_unit(self, value: str) -> str:
        raw = str(value or '').strip()
        if not re.fullmatch('\\d+(?:\\.\\d+)?', raw):
            return raw
        blob = self._shopper_ask_text() if self.brief.need_reason else ' '.join([self.brief.spoken, ' '.join((str(x) for x in self.brief.firm_requirements))])
        m = re.search(re.escape(raw) + '\\s*[-–]?\\s*([A-Za-z]{1,8})', blob)
        if not m:
            return raw
        unit = m.group(1).lower().rstrip('s')
        if unit in {'in', 'inch'}:
            return raw + ' inches'
        if unit:
            return raw + ' ' + unit
        return raw
    # -- score listings --
    def _assess(self, pid: str, sku: str) -> dict:
        key = Ledger.key(pid, sku)
        st = self.ledger.checked.get(key) or {}
        (price, stock) = (st.get('price'), st.get('in_stock'))
        dead = key in self.ledger.dead_keys or str(pid) in self.ledger.dead_products
        if self.brief.exact_first and self.ledger.adds and (str(pid) in self.ledger.dead_products):
            dead = key in self.ledger.dead_keys
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and (not dead)
        viewed = str(pid) in self.ledger.products
        judged = self.verdicts.get(key) or {}
        fit = bool(judged.get('ok')) if judged else False
        relax = int(judged.get('relax') or 0)
        if self.brief.named_only and self.brief.candidate_keys:
            named = (str(pid), str(sku)) in self.brief.candidate_keys
            if not named:
                fit = False
            elif self.brief.rank_axes:
                fit = viewed and live and self._has_rank_signal(pid, sku)
            else:
                fit = bool(fit and named)
        elif self._justification_mode() and viewed:
            if judged:
                fit = bool(judged.get('ok'))
            elif self.brief.firm_requirements:
                fit = self._firm_numbers_hold(pid, sku)
            if not self._reason_spec_hold(pid, sku):
                fit = False
            elif not fit and self.ledger.event_target:
                fit = True
            if fit and (not self._kind_hold(pid)):
                fit = False
            if fit and (not self._class_hold(pid)):
                fit = False
        elif self._shopper_choice():
            fit = viewed and self._kind_hold(pid) and self._measurement_grounded(pid, sku)
        elif self._recovery_mode() and viewed:
            fit = self._kind_hold(pid) and self._measurement_grounded(pid, sku) and self._class_hold(pid)
            relax = self._pair_relax(pid, sku)
        elif viewed:
            if judged:
                fit = bool(judged.get('ok'))
            elif self.brief.firm_requirements:
                fit = self._firm_numbers_hold(pid, sku)
            if fit and (not self._firm_numbers_hold(pid, sku)):
                fit = False
            if fit and (not self._kind_hold(pid)):
                fit = False
        if fit and viewed and self.brief.brand and (not self.brief.brand_is_preference) and (not self._shopper_choice()) and (not self._preferred_brand_hit(pid)):
            fit = False
        if self.brief.theme:
            fit = fit and all((t in _tokens(self._title(pid)) for t in self.brief.theme))
        if self.brief.need_two_listings and self.ledger.event_target:
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split('::', 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                fit = False
        recovering = bool(self.ledger.event_target or self.ledger.adds)
        if fit and viewed and (not (self.brief.named_only and self.brief.rank_axes)) and (self._justification_mode() or (recovering and self._pref_recovery())) and self._labelled_spec_conflict(pid, sku):
            fit = False
        if fit and viewed and self._pref_recovery() and (not self._pref_keep(pid, sku)):
            fit = False
        if self._pref_recovery() and viewed and self._pref_keep(pid, sku):
            if self.ledger.adds or (not self.brief.brand) or self._preferred_brand_hit(pid):
                fit = True
        ok = bool(live and viewed and fit)
        return {'pid': str(pid), 'sku': str(sku), 'key': key, 'price': price, 'live': live, 'viewed': viewed, 'ok': ok, 'rank': judged.get('rank', 99.0), 'relax': relax, 'claims': judged.get('claims') or [], 'why': judged.get('why') or ''}
    def _candidates(self) -> list:
        seen = set()
        out = []
        for key in list(self.ledger.checked):
            (pid, _, sku) = key.partition('::')
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self._assess(pid, sku))
        return out
    def _recovery_near(self, pid: str) -> bool:
        if not self.ledger.event_target:
            return True
        dead = self.ledger.event_target.split('::', 1)[0]
        if str(pid) == str(dead):
            return False
        return len(_tokens(self._title(dead)) & _tokens(self._title(pid))) >= 4
    def _kin(self, pid: str, sku: str) -> int:
        blob = _tokens(' '.join([self._title(pid), self._brand_of(pid), self._options(pid, sku), ' '.join((str(v) for v in self._facts(pid).values()))]))
        bits = set(self.brief.prefer)
        if self.ledger.event_target:
            dead_pid = self.ledger.event_target.split('::', 1)[0]
            bits |= {t for t in _tokens(self._title(dead_pid)) if any((ch.isdigit() for ch in t))}
        digit = {t for t in bits if any((ch.isdigit() for ch in t))}
        words = bits - digit
        return 3 * len(digit & blob) + len(words & blob)
    def _wish_words(self) -> set:
        parts = [self.brief.spoken, self.brief.goal]
        for m in self.messages:
            if m.get('role') != 'user':
                continue
            text = str(m.get('content') or '')
            if text.startswith('[note]'):
                continue
            parts.append(text)
        drop = _tokens(self.brief.category or '') | _WISH_STOP
        return {w for w in _tokens(' '.join(parts)) if len(w) > 3 and (not w.isdigit()) and (w not in drop)}
    def _wish_query_terms(self) -> list:
        wants = self._wish_words()
        out: list = []
        seen: set = set()
        for w in re.findall('[a-z0-9]+', (self.brief.spoken + ' ' + self.brief.goal).lower()):
            if w in wants and w not in seen:
                seen.add(w)
                out.append(w)
        return out
    def _priority_hits(self, pid: str, sku: str) -> int:
        wants = self._wish_words()
        if not wants:
            return 0
        have = _tokens(' '.join((str(v) for v in self._facts(pid).values())) + ' ' + self._title(pid) + ' ' + self._options(pid, sku))
        return len(wants & have)
    def _preference_key(self, row: dict) -> tuple:
        return (-self._priority_hits(row['pid'], row['sku']), row['price'] if row['price'] is not None else 1000000000000.0, row['key'])
    def _named_pair(self, pid: str, sku: str) -> bool:
        if not (self.brief.named_only and self.brief.candidate_keys):
            return True
        return (str(pid), str(sku)) in self.brief.candidate_keys
    def _order_pool(self, require_ok: bool=True) -> list:
        cands = self._candidates()
        named = bool(self.brief.named_only and self.brief.candidate_keys)
        def keep(row: dict, ok_only: bool) -> bool:
            if named and (not self._named_pair(row['pid'], row['sku'])):
                return False
            if ok_only:
                return row['ok']
            return bool(row['live'] and row['viewed'])
        good = [c for c in cands if keep(c, require_ok)]
        if not good and require_ok:
            return []
        if not good and named:
            good = [c for c in cands if keep(c, False)]
        if not good:
            if self.brief.firm_requirements and (not named) and (not self._shopper_choice()) and (not self._recovery_mode()) and (not self._justification_mode()):
                return []
            good = [c for c in cands if c['live'] and c['viewed']]
            if named:
                good = [c for c in good if self._named_pair(c['pid'], c['sku'])]
            if self.brief.theme:
                themed = [c for c in good if all((t in _tokens(self._title(c['pid'])) for t in self.brief.theme))]
                if themed:
                    good = themed
            if self.brief.brand:
                branded = [c for c in good if self._preferred_brand_hit(c['pid'])]
                if branded:
                    good = branded
            if self.ledger.event_target:
                kinned = [c for c in good if self._kin(c['pid'], c['sku']) > 0]
                if kinned:
                    good = kinned
        first = self.ledger.adds == 0
        use_axes = bool(self.brief.rank_axes)
        use_priority = self._shopper_choice()
        use_kin = bool(self.ledger.event_target) and (not named) and (not use_axes) and (not use_priority)
        use_relax = not use_axes and (not self._justification_mode()) and (first and self.brief.exact_first or bool(self.ledger.event_target))
        good.sort(key=lambda c: (self._axis_sort_tuple(c) or (), -self._priority_hits(c['pid'], c['sku']) if use_priority else 0, -self._kin(c['pid'], c['sku']) if use_kin else 0, c['relax'] if use_relax else 0, c['rank'] if not use_axes and c['rank'] is not None else 99.0, c['price'] if c['price'] is not None else 1000000000000.0, c['key']))
        return good
    def _last_resort_pool(self) -> list:
        good = [c for c in self._candidates() if c['live'] and c['viewed']]
        if self.brief.named_only and self.brief.candidate_keys:
            named = [c for c in good if self._named_pair(c['pid'], c['sku'])]
            if named:
                good = named
        if not good:
            return []
        if self.brief.brand and (self._justification_mode() or not self.brief.brand_is_preference):
            branded = [c for c in good if self._preferred_brand_hit(c['pid'])]
            if branded:
                good = branded
        if not self._justification_mode():
            if self._recovery_mode() and self.ledger.event_target:
                close = [c for c in good if (c.get('relax') or 9) <= 1 and self._class_hold(c['pid']) and self._recovery_near(c['pid'])]
                if close:
                    good = close
                else:
                    return []
            elif self._recovery_mode():
                held = [c for c in good if self._class_hold(c['pid'])]
                if held:
                    good = held
                elif not self._late():
                    return []
            elif self.ledger.event_target:
                kinned = [c for c in good if self._kin(c['pid'], c['sku']) > 0]
                if kinned:
                    good = kinned
        if not (self.brief.named_only and self.brief.rank_axes):
            if self._pref_recovery():
                held = [c for c in good if self._kind_hold(c['pid']) and self._pref_keep(c['pid'], c['sku'])]
                if held:
                    good = held
                else:
                    return self._pref_replacements()
            elif self._justification_mode():
                held = [c for c in good if self._kind_hold(c['pid']) and self._class_hold(c['pid']) and self._reason_spec_hold(c['pid'], c['sku'])]
                if held:
                    good = held
                else:
                    return []
            else:
                held = [c for c in good if self._kind_hold(c['pid']) and self._measurement_grounded(c['pid'], c['sku'])]
                if held:
                    good = held
        if self._shopper_choice():
            scored = [c for c in good if self._priority_hits(c['pid'], c['sku']) > 0]
            if scored:
                good = scored
            good.sort(key=self._preference_key)
        elif self._justification_mode():
            good.sort(key=lambda c: (c['price'] if c['price'] is not None else 1000000000000.0, c['key']))
        elif self.brief.rank_axes:
            good.sort(key=lambda c: (self._axis_sort_tuple(c) or (), c['price'] if c['price'] is not None else 1000000000000.0, c['key']))
        elif self._recovery_mode() or self.ledger.event_target:
            good.sort(key=lambda c: (c['relax'] if c.get('relax') is not None else 9, -self._kin(c['pid'], c['sku']), c['price'] if c['price'] is not None else 1000000000000.0, c['key']))
        else:
            good.sort(key=lambda c: (c['price'] if c['price'] is not None else 1000000000000.0, c['key']))
        return good
    def _target(self) -> dict | None:
        good = self._order_pool(require_ok=True)
        if not good:
            return None
        first = self.ledger.adds == 0
        if first and self.brief.wait_for_shopper and (not self.shopper_updated) and (not self._late()):
            return None
        if first and self.brief.named_only and self.brief.candidate_keys:
            missing = [p for (p, _s) in self.brief.candidate_keys if p not in self.ledger.products and p not in self.unreadable]
            if missing and (not self._late()):
                return None
        if self._shopper_choice():
            carted = [c for c in good if (c['pid'], c['sku']) in self.ledger.cart]
            if carted:
                return carted[0]
            scored = [c for c in good if self._priority_hits(c['pid'], c['sku']) > 0]
            pool = list(scored)
            if not pool and (self._late() or self.turn >= self.max_steps - 8):
                pool = list(good)
            if not pool:
                return None
            pool.sort(key=self._preference_key)
            if self.ledger.event_target or self._late() or self.turn >= self.max_steps - 8:
                return pool[0]
            return None
        if first and self._pref_recovery():
            return self._pref_first_cart()
        if first and self.brief.brand_is_preference and self.brief.brand and (not self._recovery_mode()):
            branded = [c for c in good if self._preferred_brand_hit(c['pid'])]
            if branded:
                good = branded
            elif not self._late() and self.turn < self.max_steps * 0.6:
                return None
        if first and self.brief.exact_first and (not self.brief.rank_axes):
            exact = [c for c in good if c['relax'] == 0]
            if self._recovery_mode():
                exact = [c for c in exact if self._class_hold(c['pid'])]
                if exact:
                    good = exact
                else:
                    return None
            elif exact:
                good = exact
            elif not self._late() and self.turn < self.max_steps * 0.6:
                return None
        if self.brief.need_two_listings and self.ledger.adds == 0 and (not self.ledger.cart) and (not self._late()) and (self.turn < self.max_steps - 8):
            return None
        if self._pref_recovery() and self.ledger.event_target:
            kept = [c for c in good if self._pref_keep(c['pid'], c['sku'])]
            if not kept:
                return None
            good = kept
        if self._justification_mode() and self.ledger.event_target:
            held = [c for c in good if self._reason_spec_hold(c['pid'], c['sku']) and self._class_hold(c['pid'])]
            if held:
                good = held
            elif not self._late() and self.turn < self.max_steps * 0.6:
                return None
        elif self._recovery_mode() and self.ledger.event_target:
            close = [c for c in good if (c.get('relax') or 9) <= 1 and self._class_hold(c['pid']) and self._recovery_near(c['pid'])]
            if close:
                close.sort(key=lambda c: (-self._kin(c['pid'], c['sku']), c.get('relax') or 9, c['price'] if c['price'] is not None else 1000000000000.0, c['key']))
                good = close
            else:
                return None
        return good[0]
    def _answer_text(self) -> str:
        target = self._target() or (self._candidates() or [None])[0]
        if not target:
            return 'I am still checking the options against your requirements; nothing is decided yet.'
        (pid, sku) = (target['pid'], target['sku'])
        parts = ['I am going with ' + (self._title(pid) or pid)[:90]]
        if target.get('why'):
            parts.append(target['why'])
        if target.get('price') is not None:
            parts.append('it is %.2f %s and in stock' % (target['price'], self.brief.currency))
        return ', '.join(parts) + '.'
    def _recall_need(self) -> int:
        if not self.brief.must_surface:
            return 0
        early = self.search_calls < 8 and self.turn < self.max_steps * 0.5
        return self.brief.must_surface + (2 if early else 0)
    def _late(self) -> bool:
        return self.turn >= self.max_steps - 5 or self.clock.closing(_FINISH_RESERVE_S)
    def _ready(self) -> bool:
        if self.ledger.adds or self._late() or self.ledger.event_target:
            return True
        if self.brief.wait_for_shopper and (not self.shopper_updated) and (not self._late()):
            return False
        if self.brief.need_two_listings and len(self.ledger.listing_ids) < 2 and (self.turn < self.max_steps - 8):
            return False
        if self._shopper_choice():
            hits = [c for c in self._candidates() if c['ok'] and self._priority_hits(c['pid'], c['sku']) > 0]
            if not hits and self.turn < self.max_steps - 8:
                return False
            ok_n = len({c['pid'] for c in self._candidates() if c['ok']})
            if self.turn < 10 and self.clock.remaining() > 170 and (ok_n < 2):
                return False
            return True
        if self.brief.named_only and self.brief.candidate_keys:
            missing = [p for (p, _s) in self.brief.candidate_keys if p not in self.ledger.products and p not in self.unreadable]
            return not missing
        if self._recovery_mode() and (not self._late()):
            exact = [c for c in self._candidates() if c['ok'] and (c.get('relax') or 9) == 0 and self._class_hold(c['pid'])]
            if not exact:
                return False
            return True
        ok_n = len({c['pid'] for c in self._candidates() if c['ok']})
        if ok_n >= 1 and self.turn >= 8:
            return True
        roomy = self.turn < 10 and self.clock.remaining() > 170
        need = self._recall_need()
        if need and roomy and (len(self.ledger.identities) < min(need, 3)):
            return False
        if roomy and (not self.reads_done) and (ok_n < 1):
            return False
        return True
    def _fresh(self, pid: str, sku: str) -> bool:
        k = Ledger.key(pid, sku)
        if k not in self.ledger.inspected:
            return False
        return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq
    # -- search and widen --
    def _unseen_leads(self, ignore_budget: bool=False) -> list:
        leads = []
        want_brand = _tokens(self.brief.brand)
        for (key, row) in self.ledger.rows.items():
            (pid, _, sku) = key.partition('::')
            if not sku or str(pid) in self.ledger.products or str(pid) in self.ledger.dead_products:
                continue
            if self.brief.named_only and (not self._named_pair(str(pid), str(sku))):
                continue
            st = self.ledger.checked.get(key) or {}
            price = st.get('price')
            if not st.get('in_stock') or price is None:
                continue
            if not ignore_budget and self.budget is not None and (price > self.budget + _EPS):
                continue
            title = str(row.get('title') or '')
            if self.brief.theme and (not all((t in _tokens(title) for t in self.brief.theme))):
                continue
            if want_brand and self._honor_brand_now() and (not (want_brand <= _tokens(str(row.get('brand') or '')) or want_brand <= _tokens(title))):
                continue
            leads.append((price, str(pid), title))
        if self._pref_recovery() and self.ledger.event_target:
            wants = self._shopper_measurements()
            cat = _tokens(self._shopper_category())
            stems = {(s[:-1] if s.endswith('s') and len(s) > 4 else s) for s in cat}
            measured = []
            for (price, pid, title) in leads:
                title_l = (title or '').lower()
                nums = [n for (n, _neigh) in _numbers_from_text(title)]
                if wants and (not any((any((abs(w - h) < 0.0001 for h in nums)) for w in wants))):
                    continue
                if stems and (not all((s in title_l for s in stems))):
                    continue
                measured.append((price, pid, title))
            if measured:
                leads = measured
            elif stems:
                kinned = [(price, pid, title) for (price, pid, title) in leads if all((s in (title or '').lower() for s in stems))]
                if kinned:
                    leads = kinned
        leads.sort()
        out = []
        for (_price, pid, _title) in leads:
            if pid not in out:
                out.append(pid)
        return out
    def _call_key(self, name: str, args: dict) -> str:
        return name + json.dumps(args, sort_keys=True)
    def _unused(self, name: str, args: dict) -> bool:
        return self._call_key(name, args) not in self.ledger.calls_seen
    def _pref_unread(self) -> list:
        if not self._pref_recovery():
            return []
        want = _tokens(self.brief.brand)
        branded = []
        other = []
        for pid in list(self.ledger.identities):
            sp = str(pid)
            if sp in self.ledger.products or sp in self.ledger.dead_products or sp in self.unreadable:
                continue
            title = self.ledger.titles.get(sp, '')
            if want and want <= _tokens(title):
                branded.append(sp)
            else:
                other.append(sp)
        return branded + other
    def _open_leads(self, leads: list) -> tuple | None:
        if not leads or self.forced_compares >= 10:
            return None
        self.forced_compares += 1
        if len(leads) > 1 and 'compare' in self.tool_names:
            args = {'product_ids': leads[:5]}
            if self._unused('compare', args):
                return ('compare', args)
        if 'view' in self.tool_names:
            args = {'product_id': leads[0]}
            if self._unused('view', args):
                return ('view', args)
        return None
    def _twin_search(self) -> tuple | None:
        dead = self.ledger.event_target
        if self.searched_twins or not dead or 'search' not in self.tool_names:
            return None
        if self.brief.named_only:
            return None
        if self._shopper_choice():
            query = ' '.join(self._wish_query_terms()[:8]) or (self.brief.search_query or '')
            hint = "The carted item is no longer valid. Search the shopper's own priority words and recover to a different listing whose facts still mention them."
        elif self._pref_recovery():
            phrase = self._measurement_phrase()
            query = ' '.join((p for p in [self.brief.brand, self._shopper_category(), phrase] if p)).strip()
            hint = "The carted item is no longer valid. Search the shopper's own labelled measurement and recover to a different listing whose facts state that value."
        elif self._recovery_mode():
            pair = ' '.join(self._pairing_prefer())
            shelf = self._shopper_category()
            query = ' '.join((p for p in [pair, shelf] if p)).strip() or ' '.join(self.brief.prefer[:6]).strip()
            hint = 'The carted item is no longer valid. Search the same model under the asked product kind and recover to another color if needed.'
        else:
            query = ' '.join(self.brief.prefer[:6]).strip()
            if not query:
                title = self._title(dead.split('::', 1)[0])
                query = ' '.join((w for w in title.split() if any((ch.isdigit() for ch in w)))) or ' '.join(title.split()[:5])
            hint = "The carted item is no longer valid. Search the shopper's own model or pairing words to recover with the smallest change the request allows."
        if not query:
            return None
        args = {'query': query, 'k': 10, 'in_stock': True}
        if not self._unused('search', args):
            return None
        self.searched_twins = True
        self._note(hint)
        return ('search', args)
    def _card_search(self) -> tuple | None:
        if not self.ledger.event_target:
            return None
        if not (self._pref_recovery() or self._recovery_mode() or self._justification_mode()):
            return None
        if self.card_searches >= 3:
            return None
        if self._pref_recovery() and (self._pref_live() is not None or self._unseen_leads()):
            return None
        title = self._title(self.ledger.event_target.split('::', 1)[0])
        queries = []
        if self._justification_mode():
            short = ' '.join(title.split()[:4]).strip()
            if short:
                queries.append(short)
            brand = self.brief.brand
            shelf = self._shopper_category()
            if brand and shelf:
                queries.append('%s %s' % (brand, shelf))
        elif self._recovery_mode():
            pair = ' '.join(self._pairing_prefer())
            if pair:
                queries.append(pair)
            words = title.split()
            for (i, w) in enumerate(words):
                if any((ch.isdigit() for ch in w)) and len(w) >= 2:
                    prev = words[i - 1] if i else ''
                    bit = ' '.join((x for x in (prev, w) if x)).strip()
                    if bit and bit.lower() not in {q.lower() for q in queries}:
                        queries.append(bit)
                    break
            short = ' '.join(words[:4]).strip()
            if short and short.lower() not in {q.lower() for q in queries}:
                queries.append(short)
            shelf = self._shopper_category()
            if pair and shelf:
                bit = ('%s %s' % (pair, shelf)).strip()
                if bit.lower() not in {q.lower() for q in queries}:
                    queries.append(bit)
        else:
            phrase = self._measurement_phrase()
            shelf = self._shopper_category()
            combo = ' '.join((p for p in [self.brief.brand, shelf, phrase] if p)).strip()
            if combo:
                queries.append(combo)
            if phrase and shelf:
                bit = ('%s %s' % (phrase, shelf)).strip()
                if bit.lower() not in {q.lower() for q in queries}:
                    queries.append(bit)
            if self.brief.search_query and self.brief.search_query.lower() not in {q.lower() for q in queries}:
                queries.append(self.brief.search_query)
        if self._justification_mode() and 'filter' in self.tool_names and (self.budget is not None):
            args = {'in_stock': True, 'k': 20, 'max_price': self.budget}
            if self.brief.brand:
                args['brand'] = self.brief.brand
            shelf = self._shopper_category()
            if shelf:
                args['category'] = shelf
            if self._unused('filter', args):
                self.card_searches += 1
                self._note('The carted listing is gone. Filter to the same maker, category, and live budget so the replacement still has the spec you must justify.')
                return ('filter', args)
        for query in queries:
            if not query or 'search' not in self.tool_names:
                continue
            args = {'query': query, 'k': 20, 'in_stock': True}
            if not self._unused('search', args):
                continue
            self.card_searches += 1
            self._note('The carted listing is gone. Search the words printed on that card to find another listing whose facts still match.')
            return ('search', args)
        if self._recovery_mode() and (not self._justification_mode()):
            if 'filter' in self.tool_names:
                args = {'in_stock': True, 'k': 20}
                if self.budget is not None:
                    args['max_price'] = self.budget
                shelf = self._shopper_category()
                if shelf:
                    args['category'] = shelf
                if self._unused('filter', args):
                    self.card_searches += 1
                    self._note('The carted listing is gone. Filter to the asked product kind under the live budget.')
                    return ('filter', args)
            return None
        if 'filter' in self.tool_names and self.brief.brand and (self.budget is not None):
            args = {'brand': self.brief.brand, 'in_stock': True, 'k': 20, 'max_price': self.budget}
            shelf = self.brief.category or self._shopper_category()
            if shelf:
                args['category'] = shelf
            if self._unused('filter', args):
                self.card_searches += 1
                self._note('The carted listing is gone. Filter to the preferred maker under the live budget to find another listing whose facts still match.')
                return ('filter', args)
        return None
    def _pref_shelf_widen(self) -> tuple | None:
        if not self._pref_recovery() or not self.ledger.event_target:
            return None
        if self._pref_live() is not None or self._pref_replacements():
            return None
        shelf = self._shopper_category()
        if not shelf or 'filter' not in self.tool_names:
            return None
        args = {'category': shelf, 'in_stock': True, 'k': 20}
        if self.budget is not None:
            args['max_price'] = self.budget
        if not self._unused('filter', args):
            return None
        return ('filter', args)
    def _hold_budget_dump(self) -> bool:
        if not self.ledger.event_target:
            return False
        if self._pref_recovery():
            return False
        if self._justification_mode() and self.card_searches < 2:
            return True
        if self._recovery_mode() and self.card_searches < 2:
            return True
        return False
    def _budget_narrow(self) -> tuple | None:
        if self.budget is None or not self.ledger.rows:
            return None
        if self._justification_mode():
            shelf = self._shopper_category()
        elif self._recovery_mode():
            shelf = self._shopper_category() or self.brief.category or ''
        elif self._pref_recovery():
            shelf = self.brief.category or self._shopper_category()
        else:
            shelf = self.brief.category or ''
        if 'filter' in self.tool_names and shelf:
            args = {'category': shelf, 'in_stock': True, 'k': 10, 'max_price': self.budget}
            if self.brief.brand and self._honor_brand_now():
                args['brand'] = self.brief.brand
            if self._unused('filter', args):
                return ('filter', args)
        if 'search' in self.tool_names:
            query = self.brief.search_query or ' '.join(self.brief.theme or ([self.brief.brand, self.brief.category] if self.brief.brand or self.brief.category else []))
            if query:
                args = {'query': query, 'k': 10, 'in_stock': True}
                if self._unused('search', args):
                    return ('search', args)
        return None
    def _read_more(self) -> tuple | None:
        if self._recovery_mode() and self.ledger.adds == 0 and 'search' in self.tool_names:
            color = [t for t in self.brief.prefer if t.isalpha() and len(t) >= 3 and t not in {'php', 'the', 'and', 'for'}]
            pair = ' '.join(list(self._pairing_prefer()) + color[:2]).strip()
            if pair:
                args = {'query': pair, 'k': 10, 'in_stock': True}
                if self._unused('search', args):
                    return ('search', args)
            shelf = self._shopper_category()
            if pair and shelf:
                args = {'query': (pair + ' ' + shelf).strip(), 'k': 10, 'in_stock': True}
                if self._unused('search', args):
                    return ('search', args)
        if self._recovery_mode() and self.ledger.adds == 0:
            digit = set(self._pairing_prefer())
            for pid in list(self.ledger.products):
                if str(pid) in self.ledger.dead_products or (not self._class_hold(str(pid))):
                    continue
                if digit and (not (digit & _tokens(self._title(pid)))):
                    continue
                for v in (self._prod(pid).get('variants') or []):
                    sku = str(v.get('sku') or '')
                    if not sku or Ledger.key(pid, sku) in self.ledger.inspected:
                        continue
                    args = {'product_id': str(pid), 'sku': sku}
                    if self._unused('inspect_stock', args):
                        return ('inspect_stock', args)
        if self.brief.named_only and self.brief.candidate_keys and ('compare' in self.tool_names):
            missing = [p for (p, _s) in self.brief.candidate_keys if p not in self.ledger.products and p not in self.unreadable]
            if missing:
                return ('compare', {'product_ids': missing[:5]})
            if self.ledger.event_target:
                stale = [(p, s) for (p, s) in self.brief.candidate_keys if Ledger.key(p, s) not in self.ledger.dead_keys and p in self.ledger.products and (Ledger.key(p, s) not in self.ledger.inspected)]
                if stale:
                    (p, s) = stale[0]
                    return ('inspect_stock', {'product_id': p, 'sku': s})
        if self._pref_recovery() and self.ledger.adds == 0:
            opened = self._open_leads(self._pref_unread())
            if opened is not None:
                return opened
        if self._pref_recovery() and self.ledger.event_target:
            if self._pref_live() is not None:
                return None
            widened = self._pref_shelf_widen()
            if widened is not None:
                return widened
            opened = self._open_leads(self._pref_unread() or self._unseen_leads())
            if opened is not None:
                return opened
            measured = self._inspect_measured()
            if measured is not None:
                return measured
            twin = self._twin_search()
            if twin is not None:
                return twin
            card = self._card_search()
            if card is not None:
                return card
        if self._justification_mode() and self.ledger.event_target:
            opened = self._open_leads(self._unseen_leads())
            if opened is not None:
                return opened
            card = self._card_search()
            if card is not None:
                return card
        if self._recovery_mode() and self.ledger.event_target:
            twin = self._twin_search()
            if twin is not None:
                return twin
            card = self._card_search()
            if card is not None:
                return card
        twin = self._twin_search()
        if twin is not None:
            return twin
        measured = self._inspect_measured()
        if measured is not None:
            return measured
        opened = self._open_leads(self._unseen_leads())
        if opened is not None:
            return opened
        if not self._hold_budget_dump():
            narrowed = self._budget_narrow()
            if narrowed is not None:
                return narrowed
        if self.brief.must_surface and (not self._late()) and (len(self.ledger.identities) < self._recall_need()):
            if self.ledger.products:
                return None
        if self._shopper_choice() and 'search' in self.tool_names:
            terms = self._wish_query_terms()
            if terms:
                args = {'query': ' '.join(terms[:8]), 'k': 10, 'in_stock': True}
                if self._unused('search', args):
                    return ('search', args)
        shelf = self._shopper_category() if (self._pref_recovery() or self._recovery_mode()) else self.brief.category
        query = self.brief.search_query or ' '.join((w for w in (self.brief.brand, shelf, ' '.join(self.brief.theme)) if w))
        if self._pref_recovery():
            phrase = self._measurement_phrase()
            if phrase and phrase.lower() not in (query or '').lower():
                query = (query + ' ' + phrase).strip()
        probe = (shelf, self.brief.brand if self._honor_brand_now() else '', '%.2f' % (self.budget or 0.0), query)
        if probe in self._probes:
            self.reads_done = True
            return None
        self._probes.add(probe)
        if 'filter' in self.tool_names and shelf:
            args = {'category': shelf, 'in_stock': True, 'k': 10}
            if self.budget is not None:
                args['max_price'] = self.budget
            if self.brief.brand and self._honor_brand_now():
                args['brand'] = self.brief.brand
            if self._unused('filter', args):
                return ('filter', args)
        if 'search' in self.tool_names:
            if query:
                args = {'query': query, 'in_stock': True, 'k': 10}
                if self._unused('search', args):
                    return ('search', args)
        self.reads_done = True
        return None
    def _unstick(self) -> tuple | None:
        opened = self._open_leads(self._unseen_leads())
        if opened is not None:
            return opened
        if not self._hold_budget_dump():
            narrowed = self._budget_narrow()
            if narrowed is not None:
                return narrowed
        more = self._read_more()
        if more is not None:
            return more
        return self._open_leads(self._unseen_leads(ignore_budget=True))
    def _spoken_claim_text(self) -> str:
        return self._shopper_ask_text()
    def _overlapping_fact_claim(self, pid: str, sku: str) -> tuple | None:
        skip = {'price', 'in_stock', 'brand', 'category', 'condition', 'navigation', 'model', 'warranty_type', 'warranty_duration'}
        spoken_text = self._spoken_claim_text()
        spoken_toks = _tokens(spoken_text)
        spoken_nums = {n for (n, _neigh) in _numbers_from_text(spoken_text)}
        spoken_nums |= set(self._shopper_spec_numbers())
        if self.budget is not None:
            spoken_nums = {n for n in spoken_nums if abs(n - self.budget) >= 0.05}
        allowed_toks: set = set()
        for name in self.claim_schema.get('allowed') or []:
            allowed_toks |= _field_tokens(str(name))
        best = None
        for (key, raw) in self._facts(pid).items():
            if str(key).lower() in skip:
                continue
            kl = str(key).lower()
            if self.brief.need_reason:
                shopper_res = bool(re.search('\\b\\d{3,4}p\\b', spoken_text or ''))
                if not (self._is_asked_spec_fact(key, raw) or (shopper_res and self._fact_is_res_class(raw))):
                    continue
            elif kl not in {'size', 'display'} and (not _token_overlap(_field_tokens(kl), allowed_toks)):
                continue
            text = str(raw).strip()
            if not text:
                continue
            score = _token_overlap(_tokens(text), spoken_toks) + 2 * len({n for (n, _neigh) in _numbers_from_text(text)} & spoken_nums)
            if score > 0 and (best is None or score > best[0]):
                best = (score, str(key), text.split(',')[0].strip())
        if not best:
            return None
        (_score, key, value) = best
        value = self._clean_claim_value(value)
        if self.brief.need_reason and (not self._claim_value_ok(value)):
            return None
        value = self._with_shopper_unit(value)
        for name in list(self.brief.claim_fields) + [str(a) for a in self.claim_schema.get('allowed') or []]:
            if _token_overlap(_field_tokens(name), _field_tokens(key)):
                return (name, value)
        return (key, value)
    def _spec_claim_fields(self, pid: str | None=None, sku: str | None=None) -> list:
        skip = {'price', 'in_stock', 'brand', 'category'}
        allowed_map = {str(a).lower(): str(a) for a in self.claim_schema.get('allowed') or []}
        named = []
        seen: set = set()
        for field in self.brief.claim_fields:
            fl = str(field).strip()
            if not fl or fl.lower() in skip or fl.lower() in seen:
                continue
            if not self.brief.need_reason and fl.lower() in {'display', 'size', 'wattage', 'condition', 'navigation'}:
                continue
            seen.add(fl.lower())
            named.append(allowed_map.get(fl.lower(), fl))
        if named:
            return named[:2]
        spoken_text = self._spoken_claim_text()
        spoken_toks = _tokens(spoken_text)
        spoken_nums = {n for (n, _neigh) in _numbers_from_text(spoken_text)}
        spoken_nums |= set(self._shopper_spec_numbers())
        if self.budget is not None:
            spoken_nums = {n for n in spoken_nums if abs(n - self.budget) >= 0.05}
        candidates = [str(a) for a in self.claim_schema.get('allowed') or [] if str(a).lower() not in skip]
        for tok in _fields_from_description(self.claim_schema.get('description') or ''):
            if tok not in skip and tok not in {c.lower() for c in candidates}:
                candidates.append(tok)
        scored: list = []
        for field in candidates:
            value = self._value_for_claim_field(pid or '', sku or '', field) if pid else None
            if not value:
                continue
            score = _token_overlap(_tokens(value), spoken_toks) + len({n for (n, _neigh) in _numbers_from_text(value)} & spoken_nums)
            if score > 0:
                scored.append((score, field))
        scored.sort(key=lambda row: row[0], reverse=True)
        return [field for (_score, field) in scored][:1]
    def _value_for_claim_field(self, pid: str, sku: str, field: str) -> str | None:
        judged = self.verdicts.get(Ledger.key(pid, sku)) or {}
        for claim in judged.get('claims') or []:
            if not isinstance(claim, dict):
                continue
            if str(claim.get('field') or '').lower() != field.lower():
                continue
            value = self._clean_claim_value(str(claim.get('value') or '').strip())
            if self.brief.need_reason and (not self._claim_value_ok(value)):
                continue
            value = self._with_shopper_unit(value)
            if value and self._grounded(pid, sku, field, value):
                return value
        want = _field_tokens(field)
        spoken_text = self._spoken_claim_text()
        spoken = _hint_tokens(spoken_text)
        spoken_toks = _tokens(spoken_text)
        spoken_nums = {n for (n, _neigh) in _numbers_from_text(spoken_text)}
        spoken_nums |= set(self._shopper_spec_numbers())
        if self.budget is not None:
            spoken_nums = {n for n in spoken_nums if abs(n - self.budget) >= 0.05}
        (best_val, best_score) = (None, 0)
        for (key, raw) in self._facts(pid).items():
            text = str(raw).strip()
            if not text:
                continue
            spoken_hit = _token_overlap(_tokens(text), spoken_toks) + len({n for (n, _neigh) in _numbers_from_text(text)} & spoken_nums)
            if not spoken_hit and (not _token_overlap(_field_tokens(str(key)), want)):
                continue
            if not spoken_hit:
                continue
            score = spoken_hit + _token_overlap(_field_tokens(str(key)), want)
            if score > best_score:
                (best_val, best_score) = (text.split(',')[0].strip(), score)
        if best_val:
            best_val = self._clean_claim_value(best_val)
            if self.brief.need_reason and (not self._claim_value_ok(best_val)):
                return None
            best_val = self._with_shopper_unit(best_val)
            if self._grounded(pid, sku, field, best_val):
                return best_val
        return None
    def _claims_for(self, pid: str, sku: str) -> list:
        out: list = []
        seen: set = set()
        def add(field: Any, value: Any, force: bool=False) -> None:
            (field_s, value_s) = (str(field or '').strip(), str(value or '').strip())
            if not field_s or not value_s:
                return
            if not force and (not self._claim_allowed(field_s, pid)):
                return
            if not self._grounded(pid, sku, field_s, value_s):
                return
            key = (field_s.lower(), value_s.lower())
            if key in seen:
                return
            seen.add(key)
            out.append({'field': field_s, 'value': value_s})
        if self.brief.need_reason:
            fact_claim = self._overlapping_fact_claim(pid, sku)
            if fact_claim:
                add(fact_claim[0], fact_claim[1], force=True)
        if not any((str(c.get('field') or '').lower() not in {'price', 'in_stock', 'brand', 'category'} for c in out)):
            for field in self._spec_claim_fields(pid, sku):
                value = self._value_for_claim_field(pid, sku, field)
                if value:
                    add(field, value, force=True)
        if not any((str(c.get('field') or '').lower() not in {'price', 'in_stock', 'brand', 'category'} for c in out)):
            fact_claim = self._overlapping_fact_claim(pid, sku)
            if fact_claim:
                add(fact_claim[0], fact_claim[1], force=True)
        st = self.ledger.status(pid, sku) or {}
        if self.budget is not None:
            add('price', 'at most %.2f' % self.budget)
        elif st.get('price') is not None:
            add('price', '%.2f' % st['price'])
        if st.get('in_stock') or (pid, sku) in self.ledger.cart:
            add('in_stock', 'true')
        brand = self._brand_of(pid)
        if brand:
            add('brand', brand)
        return out
    def _want_justification(self) -> bool:
        return bool(self.brief.need_reason or self.claim_schema.get('required'))
    def _order_args(self, pid: str, sku: str) -> dict:
        out = {'product_id': pid, 'sku': sku}
        if self._want_justification():
            claims = self._claims_for(pid, sku)
            if claims:
                out['justification'] = {'claims': claims}
        return out
    def _valid_now(self, pid: str, sku: str) -> tuple:
        st = self.ledger.status(pid, sku)
        k = Ledger.key(pid, sku)
        if st is None:
            return (False, 'inspect_stock the product_id and sku first')
        if (pid, sku) not in self.ledger.cart:
            return (False, 'add that variant to the cart first')
        if self.ledger.last_add_seq is not None and self.ledger.checked_seq.get(k, 0) <= self.ledger.last_add_seq:
            return (False, 're-check the variant with inspect_stock after the cart addition')
        if not st.get('in_stock'):
            return (False, 'that variant is out of stock; choose a different product')
        price = st.get('price')
        if self.budget is not None and price is not None and (price > self.budget + _EPS):
            return (False, 'that variant costs %.2f, above the budget of %.2f' % (price, self.budget))
        if k in self.ledger.dead_keys:
            return (False, 'that variant was invalidated; choose a different product')
        return (True, '')
    def _shortlist_note(self) -> None:
        if self.clock.closing(80):
            return
        self._judge_products()
        if self._shopper_choice() and (not self.ledger.cart):
            good = [c for c in self._candidates() if c['ok']]
            good.sort(key=self._preference_key)
            if good:
                self._note('Listings that already meet stock and budget: ' + '; '.join(('%s::%s at %.2f' % (c['pid'], c['sku'], c['price'] or 0.0) for c in good[:6])) + ". First-cart the one whose own facts mention the most of the shopper's stated priorities, not the cheapest.")
                return
        best = self._target()
        if best:
            self._note('A candidate now looks fully supported: %s at %s. %s The controller will inspect, cart and order once the usual stock check is fresh; keep reading only if something could rank better.' % (best['key'], best['price'], best.get('why') or ''))
            return
        blocked = [c for c in self._candidates() if c['viewed'] and c['live'] and (not c['ok'])]
        if blocked:
            sample = '; '.join(((c['key'] + ' ' + (c.get('why') or 'not yet supported'))[:80] for c in blocked[:3]))
            self._note('Read products are not yet a safe buy: ' + sample + '. Look at different products.')
    def _budget_from_message(self, text: str) -> None:
        new = _budget_in(text)
        if new is None or (self.budget is not None and new >= self.budget):
            return
        if self._pref_recovery():
            return
        self.budget = new
        self.shopper_updated = True
        self._note('Budget updated by the shopper: at most %.2f %s.' % (new, self.brief.currency))
        for (pid, sku) in list(self.ledger.cart):
            st = self.ledger.status(pid, sku) or {}
            price = st.get('price')
            if price is not None and price > new + _EPS:
                self.ledger.dead_keys.add(Ledger.key(pid, sku))
                self._note('The carted item is now above budget and will be replaced.')
    def _live_cart(self) -> tuple | None:
        for (pid, sku) in reversed(self.ledger.cart):
            key = Ledger.key(pid, sku)
            if key in self.ledger.dead_keys:
                continue
            if self.brief.named_only and (not self._named_pair(str(pid), str(sku))):
                continue
            st = self.ledger.status(pid, sku) or {}
            if not st.get('in_stock'):
                continue
            price = st.get('price')
            if self.budget is not None and price is not None and (price > self.budget + _EPS):
                continue
            return (str(pid), str(sku))
        return None
    # -- choose the next tool --
    def _plan(self) -> tuple | None:
        if self.done:
            return None
        if self.shopper_question_pending and 'message' in self.tool_names:
            if self._hold_pref_talk() or (self._pref_recovery() and self.ledger.event_target):
                pass
            elif self.turn <= self.max_steps - 4:
                self.shopper_question_pending = False
                self.question_waits = 0
                return ('message', {'content': self._answer_text()})
            else:
                self.shopper_question_pending = False
                self.question_waits = 0
        live = self._live_cart()
        if live is not None:
            (pid, sku) = live
            if not self._fresh(pid, sku):
                return ('inspect_stock', {'product_id': pid, 'sku': sku})
            return ('place_test_order', self._order_args(pid, sku))
        for (cpid, csku) in self.ledger.cart:
            if Ledger.key(cpid, csku) in self.ledger.dead_keys and 'remove_from_cart' in self.tool_names:
                return ('remove_from_cart', {'product_id': cpid, 'sku': csku})
        if self.brief.named_only and self.brief.candidate_keys and ('compare' in self.tool_names):
            missing = [p for (p, _s) in self.brief.candidate_keys if p not in self.ledger.products and p not in self.unreadable]
            if missing:
                return ('compare', {'product_ids': missing[:5]})
        target = self._target()
        first = self._pref_first_cart()
        if first is not None:
            target = first
        if target is None and self._pref_recovery() and self.ledger.event_target:
            target = self._pref_live()
        if first is not None:
            (pid, sku) = (first['pid'], first['sku'])
            if not self._fresh(pid, sku):
                return ('inspect_stock', {'product_id': pid, 'sku': sku})
            return ('add_to_cart', {'product_id': pid, 'sku': sku})
        if self._pref_recovery() and (not self.ledger.adds) and (not self.ledger.event_target):
            measured = self._inspect_measured()
            if measured is not None:
                return measured
        if self._pref_recovery() and self.ledger.event_target and (target is not None):
            (pid, sku) = (target['pid'], target['sku'])
            if not self._fresh(pid, sku):
                return ('inspect_stock', {'product_id': pid, 'sku': sku})
            return ('add_to_cart', {'product_id': pid, 'sku': sku})
        if target is None or not self._ready():
            forced = self._read_more()
            if forced is not None:
                return forced
            if target is None:
                if self._pref_recovery() and self.ledger.event_target and (not self._late()):
                    opened = self._open_leads(self._pref_unread() or self._unseen_leads())
                    if opened is not None:
                        return opened
                    measured = self._inspect_measured()
                    if measured is not None:
                        return measured
                    twin = self._twin_search()
                    if twin is not None:
                        return twin
                    card = self._card_search()
                    if card is not None:
                        return card
                if self._justification_mode() and self.ledger.event_target and (not self._late()):
                    opened = self._open_leads(self._unseen_leads())
                    if opened is not None:
                        return opened
                    card = self._card_search()
                    if card is not None:
                        return card
                    narrowed = self._budget_narrow()
                    if narrowed is not None:
                        return narrowed
                if self._recovery_mode() and self.ledger.event_target:
                    twin = self._twin_search()
                    if twin is not None:
                        return twin
                    card = self._card_search()
                    if card is not None:
                        return card
                if self.ledger.event_target and (self._late() or self.turn >= self.max_steps - 10 or self.search_calls >= 8):
                    fallback = self._last_resort_pool()
                    if fallback:
                        (pid, sku) = (fallback[0]['pid'], fallback[0]['sku'])
                        if not self._fresh(pid, sku):
                            return ('inspect_stock', {'product_id': pid, 'sku': sku})
                        return ('add_to_cart', {'product_id': pid, 'sku': sku})
                return None
        if not self._ready():
            return None
        (pid, sku) = (target['pid'], target['sku'])
        if not self._fresh(pid, sku):
            return ('inspect_stock', {'product_id': pid, 'sku': sku})
        return ('add_to_cart', {'product_id': pid, 'sku': sku})
    def _ask_model(self) -> dict | None:
        if self.clock.closing(_FINISH_RESERVE_S + 5):
            return None
        for attempt in range(2):
            if self.clock.closing(_FINISH_RESERVE_S + 5):
                return None
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            payload = {'model': model, 'messages': self.messages, 'tools': self.tools, 'tool_choice': 'required', 'temperature': TemperatureController.for_attempt('act', attempt)}
            resp = _PROXY.post('/inference/chat/completions', json_data=payload)
            if resp and isinstance(resp, dict) and resp.get('choices'):
                msg = resp['choices'][0].get('message') or {}
                calls = msg.get('tool_calls') or []
                if calls:
                    return {'content': msg.get('content') or '', 'tool_calls': calls[:self.max_calls]}
                self.messages.append({'role': 'assistant', 'content': msg.get('content') or ''})
                self._note('No tool call was made. Choose exactly one of the supplied tools now.')
                continue
            if self.clock.remaining() > 100:
                _backoff(attempt)
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None
    def _send(self, envelope: dict) -> dict | None:
        for attempt in range(2):
            result = _PROXY.post('/environment/call', json_data=envelope)
            if isinstance(result, dict) and result.get('calls'):
                return result
            if self.clock.remaining() < 12:
                break
            _backoff(attempt)
        return None
    def _act(self, name: str, args: dict, content: str='', tool_call_id: str | None=None) -> dict | None:
        self._mark_call(name, args)
        turn = self.turn + 1
        group = self.problem_id + '-turn-' + str(turn)
        envelope = {**self.binding, 'call_id': group, 'idempotency_key': group, 'turn': turn, 'calls': [{'call_id': group + '-1', 'action': {'name': name, 'args': args}}]}
        result = self._send(envelope)
        self.turn = turn
        self.pending_envelope = None
        return self._absorb(name, args, content, tool_call_id or 'agent-' + str(turn), result)
    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]
        self.messages.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls})
        self.dialogue.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls, 'environment_result': result})
        if not isinstance(result, dict):
            self.ledger.calls_seen.discard(name + json.dumps(args, sort_keys=True))
            self.messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps({'error': 'environment call failed; it will be retried automatically'})})
            return None
        calls = result.get('calls') or []
        wrap = (calls[0].get('observation') if calls and isinstance(calls[0], dict) else None) or {}
        obs = wrap.get('observation') if isinstance(wrap.get('observation'), dict) else wrap
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
            self._budget_from_message(text)
            self.brief.spoken = (self.brief.spoken + ' ' + text).strip()[:2500]
            self.shopper_updated = True
            self.brief.wait_for_shopper = False
            if not self.clock.closing(80):
                self._judge_request()
                self._judge_products(force=True)
            if _QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self._note('The shopper asked a question: answer it with message in your next turn, then continue.')
        if name in ('view', 'compare') and (not self.clock.closing(80)):
            self._judge_products()
        if any((isinstance(c, dict) and (c.get('observation') or {}).get('done') for c in calls)):
            self.done = True
        return obs
    def _post_add_recheck(self) -> None:
        (pid, sku) = self.pending_recheck
        self.pending_recheck = None
        obs = self._act('inspect_stock', {'product_id': pid, 'sku': sku})
        if not isinstance(obs, dict) or obs.get('error'):
            self.ledger.dead_keys.add(Ledger.key(pid, sku))
            if not self.done:
                self._recover_now()
            return
        stock = bool(obs.get('in_stock'))
        price = _num(obs.get('price'))
        over = self.budget is not None and price is not None and (price > self.budget + _EPS)
        if stock and (not over):
            self._note('Re-check after the cart addition: %s is in stock at %s. Ordering it now.' % (Ledger.key(pid, sku), price))
            if not self.done and self.turn < self.max_steps:
                self._act('place_test_order', self._order_args(pid, sku))
            return
        self.ledger.event_target = Ledger.key(pid, sku)
        self.ledger.dead_keys.add(Ledger.key(pid, sku))
        if not self.brief.named_only:
            self.ledger.dead_products.add(str(pid))
        self.forced_compares = 0
        self.reads_done = False
        self.searched_twins = False
        self.card_searches = 0
        self._probes.clear()
        self.verdicts.pop(Ledger.key(pid, sku), None)
        reason = 'went out of stock' if not stock else 'was repriced above budget'
        if self.brief.named_only:
            self._note('Market change: the carted item %s %s. Re-check and re-rank only the named candidate variants; do not leave that list.' % (Ledger.key(pid, sku), reason))
        else:
            self._note('Market change: the carted item %s %s. Recover using only the shopper request — same model, next acceptable option, or another listing that still fits.' % (Ledger.key(pid, sku), reason))
        if not self.clock.closing(80):
            self._judge_products(force=True)
            self._shortlist_note()
        if not self.done:
            self._recover_now()
    # -- market-event recovery --
    def _recover_now(self) -> None:
        if self.done or self.turn >= self.max_steps:
            return
        if self.shopper_question_pending and 'message' in self.tool_names and (self.turn <= self.max_steps - 4) and (not (self._pref_recovery() and self.ledger.event_target)):
            self.shopper_question_pending = False
            self.question_waits = 0
            self._act('message', {'content': self._answer_text()})
            if self.done:
                return
        if self._pref_recovery():
            if self._pref_live() is not None or self._pref_replacements() or self._target() is not None:
                self._finish_fast()
                return
            widened = self._pref_shelf_widen()
            if widened is not None:
                self._act(widened[0], widened[1])
                return
            opened = self._open_leads(self._pref_unread() or self._unseen_leads())
            if opened is not None:
                self._act(opened[0], opened[1])
                return
            measured = self._inspect_measured()
            if measured is not None:
                self._act(measured[0], measured[1])
                if self._pref_live() is not None or self._target() is not None:
                    self._finish_fast()
                return
            twin = self._twin_search()
            if twin is not None:
                self._act(twin[0], twin[1])
                if self.done:
                    return
            card = self._card_search()
            if card is not None:
                self._act(card[0], card[1])
                if self.done:
                    return
        if self._justification_mode() and self.ledger.event_target:
            if self._target() is not None:
                self._finish_fast()
                return
            opened = self._open_leads(self._unseen_leads())
            if opened is not None:
                self._act(opened[0], opened[1])
                if self.done:
                    return
            card = self._card_search()
            if card is not None:
                self._act(card[0], card[1])
                if self.done:
                    return
            narrowed = self._budget_narrow()
            if narrowed is not None:
                self._act(narrowed[0], narrowed[1])
                if self.done:
                    return
        if self._recovery_mode() and self.ledger.event_target:
            if self._target() is not None:
                self._finish_fast()
                return
            twin = self._twin_search()
            if twin is not None:
                self._act(twin[0], twin[1])
                if self.done:
                    return
            card = self._card_search()
            if card is not None:
                self._act(card[0], card[1])
                if self.done:
                    return
            if self._target() is not None:
                self._finish_fast()
                return
            return
        if self._target() is not None:
            self._finish_fast()
            return
        if self.brief.named_only or self._late() or self.turn >= self.max_steps - 8:
            self._finish_fast()
    def _finish_fast(self) -> None:
        room = self.max_steps - self.turn
        live = self._live_cart()
        if live is not None:
            (pid, sku) = live
            if not self._fresh(pid, sku) and room >= 2:
                self._act('inspect_stock', {'product_id': pid, 'sku': sku})
                if self.done:
                    return
            if not self.done and self.turn < self.max_steps:
                self._act('place_test_order', self._order_args(pid, sku))
            return
        if self.done or self.turn >= self.max_steps:
            return
        good = self._order_pool(require_ok=True)
        if self._pref_recovery():
            live = self._pref_live()
            if live is not None:
                good = [live]
        if not good and self.brief.named_only:
            good = self._order_pool(require_ok=False)
        if not good:
            opened = self._open_leads((self._pref_unread() or self._unseen_leads()) if self._pref_recovery() else self._unseen_leads())
            if opened is not None and self.turn < self.max_steps - 2:
                self._act(opened[0], opened[1])
                if not self.clock.closing(80):
                    self._judge_products()
                good = self._order_pool(require_ok=True)
                if not good and self.brief.named_only:
                    good = self._order_pool(require_ok=False)
        if not good:
            good = self._last_resort_pool()
        for cand in good[:4]:
            if self.done or self.turn >= self.max_steps:
                return
            (pid, sku) = (cand['pid'], cand['sku'])
            if (pid, sku) not in self.ledger.cart and self.turn < self.max_steps:
                obs = self._act('add_to_cart', {'product_id': pid, 'sku': sku})
                if self.done:
                    return
                if not isinstance(obs, dict) or obs.get('error'):
                    self.ledger.dead_keys.add(Ledger.key(pid, sku))
                    continue
            if self.turn < self.max_steps:
                obs = self._act('inspect_stock', {'product_id': pid, 'sku': sku})
                if self.done:
                    return
                if not isinstance(obs, dict) or not obs.get('in_stock'):
                    self.ledger.dead_keys.add(Ledger.key(pid, sku))
                    if not self.brief.named_only:
                        self.ledger.dead_products.add(str(pid))
                    continue
                price = _num(obs.get('price'))
                if self.budget is not None and price is not None and (price > self.budget + _EPS):
                    self.ledger.dead_keys.add(Ledger.key(pid, sku))
                    continue
            if not self.done and self.turn < self.max_steps:
                self._act('place_test_order', self._order_args(pid, sku))
            return
        if self.done or self.turn >= self.max_steps:
            return
        leftover = self._last_resort_pool()
        for cand in leftover[:3]:
            if self.done or self.turn >= self.max_steps:
                return
            (pid, sku) = (cand['pid'], cand['sku'])
            if (pid, sku) not in self.ledger.cart and self.turn < self.max_steps:
                obs = self._act('add_to_cart', {'product_id': pid, 'sku': sku})
                if self.done:
                    return
                if not isinstance(obs, dict) or obs.get('error'):
                    self.ledger.dead_keys.add(Ledger.key(pid, sku))
                    continue
            if self.turn < self.max_steps:
                obs = self._act('inspect_stock', {'product_id': pid, 'sku': sku})
                if self.done:
                    return
                if not isinstance(obs, dict) or not obs.get('in_stock'):
                    continue
                price = _num(obs.get('price'))
                if self.budget is not None and price is not None and (price > self.budget + _EPS):
                    continue
            if not self.done and self.turn < self.max_steps:
                self._act('place_test_order', self._order_args(pid, sku))
            return
    # -- session entry --
    def run(self) -> list:
        if not self.clock.closing(80):
            self._judge_request()
        (last_forced, repeats) = (None, 0)
        while not self.done and self.turn < self.max_steps:
            if self.clock.closing(_FINISH_RESERVE_S) or self.turn >= self.max_steps - 6:
                self._finish_fast()
                break
            if self.ledger.event_target and (not self._live_cart()) and (self.turn >= self.max_steps - 10):
                self._finish_fast()
                break
            if self.brief.ask and (not self.asked_shopper) and ('message' in self.tool_names):
                if self._hold_pref_talk():
                    pass
                else:
                    self.asked_shopper = True
                    self._act('message', {'content': self.brief.ask})
                    continue
            if self.pending_recheck and 'inspect_stock' in self.tool_names:
                self._post_add_recheck()
                continue
            plan = self._plan()
            if plan is None:
                if self.ledger.event_target and (self._late() or self.search_calls >= 8 or self.turn >= self.max_steps - 10):
                    self._finish_fast()
                    break
                plan = self._unstick()
            if plan is not None:
                (name, args) = plan
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    if name in ('inspect_stock', 'add_to_cart'):
                        self.ledger.dead_keys.add(Ledger.key(args.get('product_id'), args.get('sku')))
                    if name == 'compare':
                        self.unreadable.update((str(p) for p in args.get('product_ids') or []))
                    if name == 'message':
                        self.shopper_question_pending = False
                    (last_forced, repeats) = (None, 0)
                    self._note('That step changed nothing, so the controller moved on.')
                    continue
                obs = self._act(name, args)
                if name in ('search', 'filter'):
                    self.search_calls += 1
                if name == 'message':
                    self.shopper_question_pending = False
                    self.question_waits = 0
                if name == 'add_to_cart' and isinstance(obs, dict) and (not obs.get('error')):
                    added = obs.get('added') or {}
                    if added.get('product_id'):
                        self.pending_recheck = (str(added['product_id']), str(added['sku']))
                continue
            (last_forced, repeats) = (None, 0)
            if self.clock.closing(_FINISH_RESERVE_S):
                self._finish_fast()
                break
            proposal = self._ask_model()
            if proposal is None:
                stuck = self._unstick()
                if stuck is not None:
                    self._act(stuck[0], stuck[1])
                    continue
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
                self.dialogue.append({'role': 'assistant', 'content': '[rejected ' + name + '] ' + verdict[:200]})
                again = self._ask_model()
                if again is None:
                    break
                call = again['tool_calls'][0]
                name = (call.get('function') or {}).get('name') or ''
                args = _arguments(call)
                content = again.get('content') or ''
            if verdict == 'skip':
                stuck = self._unstick()
                if stuck is not None:
                    (name, args) = stuck
                    obs = self._act(name, args)
                    if name in ('search', 'filter'):
                        self.search_calls += 1
                    if name == 'add_to_cart' and isinstance(obs, dict) and (not obs.get('error')):
                        added = obs.get('added') or {}
                        if added.get('product_id'):
                            self.pending_recheck = (str(added['product_id']), str(added['sku']))
                continue
            self.rejected_streak = 0
            if name not in self.tool_names:
                continue
            if name == 'place_test_order':
                (pid, sku) = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
                if not pid and self.ledger.cart:
                    (pid, sku) = self.ledger.cart[-1]
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
                self.search_calls += 1
                need = self._recall_need()
                if need:
                    have = len(self.ledger.identities)
                    self._note('Distinct matching products surfaced so far: %d of %d.' % (have, need) + ('' if have >= need else ' Keep searching with different wording.'))
            if name == 'inspect_cart':
                self.cart_inspections += 1
            if name in ('view', 'compare') and (not self.clock.closing(80)):
                self._shortlist_note()
        return self.dialogue
    # -- tool guards --
    def _validate(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return 'unknown tool; choose one of ' + str(sorted((n for n in self.tool_names if n)))
        if name == 'place_test_order':
            (pid, sku) = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
            if not pid and self.ledger.cart:
                (pid, sku) = self.ledger.cart[-1]
            (ok, why) = self._valid_now(pid, sku)
            if not ok:
                return 'order refused: ' + why
            need = self._recall_need()
            if need and (not self._late()) and (len(self.ledger.identities) < need):
                return 'order refused: only %d distinct matching products have been surfaced; search until at least %d appear' % (len(self.ledger.identities), need)
            if self.brief.need_two_listings and (not self._late()) and (len(self.ledger.listing_ids) < 2):
                return 'order refused: compare at least two products from different listings first'
        if name == 'add_to_cart':
            (pid, sku) = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
            if not pid or not sku:
                return 'add_to_cart needs both product_id and sku'
            if Ledger.key(pid, sku) not in self.verdicts:
                self._judge_products()
            c = self._assess(pid, sku)
            key = Ledger.key(pid, sku)
            if not c['live']:
                return 'that variant is out of stock, over budget, or already invalidated'
            if not c['viewed']:
                return 'view or compare that product first so its raw facts are on record'
            if key not in self.ledger.inspected:
                return 'inspect_stock that variant first: current price and stock decide whether it can be bought'
            if not c['ok']:
                pref_ok = self._pref_recovery() and self._pref_keep(pid, sku)
                if pref_ok and (not self.ledger.adds) and self.brief.brand and (not self._preferred_brand_hit(pid)):
                    pref_ok = False
                if not pref_ok:
                    return 'that product is not yet supported by the evidence for this request'
            if not (self.brief.named_only and self.brief.rank_axes):
                if self._justification_mode():
                    if not self._reason_spec_hold(pid, sku):
                        return "that card's labelled facts do not support the spec the shopper asked you to justify"
                    if not self._class_hold(pid):
                        return 'that listing does not name the kind of product the shopper asked you to justify'
                elif self._pref_recovery():
                    if not self._pref_keep(pid, sku):
                        return "that card does not state the shopper's own measurement with its unit"
                    if (not self.ledger.adds) and self.brief.brand and (not self._preferred_brand_hit(pid)) and self._honor_brand_now():
                        return 'first-cart a branded listing that still meets the firm measurement'
                elif self._recovery_mode():
                    if not self._class_hold(pid):
                        return 'that listing is not the kind of product the shopper asked for'
                    if not self._measurement_grounded(pid, sku):
                        return "that card does not state the shopper's own measurement as a labelled value or variant option"
                elif not self._measurement_grounded(pid, sku):
                    return "that card does not state the shopper's own measurement as a labelled value or variant option"
                if not self._kind_hold(pid):
                    return 'that listing does not carry the same kind of labelled facts as a matching device'
            if self.brief.need_two_listings and (not self._late()) and (len(self.ledger.listing_ids) < 2):
                return 'compare two products from different listings before the first cart addition'
            if self._shopper_choice() and (not self._late()):
                hits = self._priority_hits(pid, sku)
                pool = [row for row in self._candidates() if row['ok'] and row['live']]
                best_hits = max((self._priority_hits(row['pid'], row['sku']) for row in pool), default=0)
                if best_hits > hits:
                    return "another viewed listing mentions more of the shopper's own priority words; cart that one instead of a weaker match"
            if self.brief.exact_first and (not self.ledger.adds) and c['relax'] and (not self._late()):
                return 'the first cart item must be the exact pairing the shopper named; keep searching'
            if self.brief.named_only and self.brief.candidate_keys and (not self._named_pair(pid, sku)):
                return 'stay on the named product_id::sku pairs the shopper listed'
            if self.brief.named_only and (not self.ledger.adds):
                best = self._target()
                if best and Ledger.key(best['pid'], best['sku']) != key and (not self._late()):
                    return 'the first cart item must be the current best named pairing, not a placeholder'
            if not self._ready():
                return 'hold: read one more product that also meets the request first, so there is a fallback when the market moves after the cart addition'
        if name == 'message':
            text = str(args.get('content') or '')
            if not text.strip():
                return 'message content is required'
            if self._hold_pref_talk():
                return "cart a live branded match that states the shopper's measurement first; answer after that add"
            if not self.shopper_question_pending and '?' not in text and (not self.brief.ask):
                return 'do not send status updates; use message only to answer the shopper or to ask one question'
            if not self.shopper_question_pending and self.questions_asked >= 2:
                return 'the shopper has already answered; decide from the evidence rather than asking again'
        if name == 'inspect_cart' and self.cart_inspections >= 1:
            return 'the cart contents are already known from earlier observations'
        key = name + json.dumps(args, sort_keys=True)
        if key in self.ledger.calls_seen and name in ('search', 'filter', 'view', 'compare'):
            return 'identical call already made; change the arguments'
        return None
    def _mark_call(self, name: str, args: dict) -> None:
        if name in ('search', 'filter', 'view', 'compare'):
            self.ledger.calls_seen.add(name + json.dumps(args, sort_keys=True))

# Bare names resolve here so Clock/Brief/Ledger and method bodies keep their original lookups.
_MODELS = Agent._MODELS
_PROXY = Agent._PROXY
_PROBLEM_BUDGET_S = Agent._PROBLEM_BUDGET_S
_FINISH_RESERVE_S = Agent._FINISH_RESERVE_S
_OBS_CHARS = Agent._OBS_CHARS
_MAX_REPROMPTS = Agent._MAX_REPROMPTS
_EPS = Agent._EPS
_REASON_RE = Agent._REASON_RE
_MODELISH_RE = Agent._MODELISH_RE
_PHRASE_RE = Agent._PHRASE_RE
_PREFER_RE = Agent._PREFER_RE
_SOFT_PICK_RE = Agent._SOFT_PICK_RE
_EXACT_PAIR_RE = Agent._EXACT_PAIR_RE
_PAIR_COLOR_RE = Agent._PAIR_COLOR_RE
_TRADEOFF_RE = Agent._TRADEOFF_RE
_HIDDEN_SPEC_RE = Agent._HIDDEN_SPEC_RE
_NUM = Agent._NUM
_BUDGET_RE = Agent._BUDGET_RE
_KEY_RE = Agent._KEY_RE
_QUOTED_RE = Agent._QUOTED_RE
_QUESTION_RE = Agent._QUESTION_RE
_CORE_RULES = Agent._CORE_RULES
_JUDGE_SYS = Agent._JUDGE_SYS
_COMPARE_AXIS_RE = Agent._COMPARE_AXIS_RE
_FIELD_SNAKE_RE = Agent._FIELD_SNAKE_RE
_FIELD_LIST_RE = Agent._FIELD_LIST_RE
_num = Agent._num
_tokens = Agent._tokens
_arguments = Agent._arguments
_trim = Agent._trim
_parse_json = Agent._parse_json
_budget_in = Agent._budget_in
_exact_pair_ask = Agent._exact_pair_ask
_pairing_words = Agent._pairing_words
_usable_brand = Agent._usable_brand
_norm_unit = Agent._norm_unit
_measure_mentioned = Agent._measure_mentioned
_backoff = Agent._backoff
_tool_fn = Agent._tool_fn
_fields_from_description = Agent._fields_from_description
_AXIS_STOP = Agent._AXIS_STOP
_WISH_STOP = Agent._WISH_STOP
_axes_from_text = Agent._axes_from_text
_hint_tokens = Agent._hint_tokens
_ask_only = Agent._ask_only
_field_tokens = Agent._field_tokens
_token_overlap = Agent._token_overlap
_numbers_from_text = Agent._numbers_from_text
_suffix_numbers = Agent._suffix_numbers
_claim_schema = Agent._claim_schema

class TemperatureController:
    BASE = {'judge': 0.0, 'parse': 0.0, 'act': 0.0}
    STEP = 0.0
    MAX = 0.0
    @classmethod
    def base(cls, role: str) -> float:
        return float(cls.BASE.get(role, 0.0))
    @classmethod
    def for_attempt(cls, role: str, attempt: int=0) -> float:
        return min(cls.MAX, cls.base(role) + max(0, int(attempt)) * cls.STEP)
class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget
    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)
    def closing(self, reserve: float=_FINISH_RESERVE_S) -> bool:
        return self.remaining() < reserve
class Brief:
    def __init__(self, query: str) -> None:
        self.raw = query or ''
        (goal, _, _rules) = self.raw.partition('\n\nTask rules:')
        self.goal = goal.strip()
        self.spoken = self.goal
        self.budget = _budget_in(self.goal)
        self.currency = 'PHP'
        m = re.search('\\b([A-Z]{3})\\b', self.goal)
        if m and m.group(1) in ('PHP', 'USD', 'EUR', 'SGD', 'MYR', 'IDR', 'THB', 'VND'):
            self.currency = m.group(1)
        self.candidate_keys = [(pid, sku.rstrip('.,;:')) for (pid, sku) in _KEY_RE.findall(self.goal)]
        quoted = _QUOTED_RE.findall(self.goal)
        self.theme = re.findall('[a-z0-9]+', (quoted[0] if quoted else '').lower())
        if not self.theme:
            phrase = _PHRASE_RE.search(self.goal)
            if phrase:
                self.theme = re.findall('[a-z0-9]+', phrase.group(1).lower())
        prefer: list = []
        for q in quoted:
            prefer.extend(re.findall('[a-z0-9]+', q.lower()))
        for m in _PREFER_RE.finditer(self.goal):
            prefer.extend(re.findall('[a-z0-9]+', m.group(1).lower()))
        for m in _MODELISH_RE.finditer(self.goal):
            prefer.append(m.group(0).lower())
        prefer.extend(_pairing_words(self.goal))
        budget_tok = str(int(self.budget)) if self.budget is not None else ''
        seen: set = set()
        self.prefer: list = []
        for tok in prefer:
            if tok in seen or tok in {'php', 'usd', 'eur'} or tok == budget_tok:
                continue
            seen.add(tok)
            self.prefer.append(tok)
        self.ask: str | None = None
        self.must_surface = 0
        self.named_only = bool(self.candidate_keys)
        ask = _ask_only(self.goal)
        self.need_reason = bool(_REASON_RE.search(ask))
        self.need_two_listings = bool(_TRADEOFF_RE.search(ask))
        self.wait_for_shopper = bool(_HIDDEN_SPEC_RE.search(ask))
        self.shopper_choice = self.need_two_listings
        self.exact_first = bool(self.candidate_keys) or _exact_pair_ask(ask)
        self.category = ''
        self.brand = ''
        self.search_query = ''
        self.notes = ''
        self.firm_requirements: list = []
        self.rank_axes: list = _axes_from_text(ask)
        self.claim_fields: list = []
        self.brand_is_preference = bool(re.search('\\bprefer\\b', ask, re.I) or _SOFT_PICK_RE.search(ask))
        self.fingerprint = ''
    def apply_judge(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        if self.fingerprint:
            data = dict(data)
            for key in ('need_reason', 'named_only', 'wait_for_shopper', 'exact_first', 'need_two_listings', 'shopper_choice', 'rank_axes', 'brand_is_preference', 'category'):
                data.pop(key, None)
        ask = data.get('ask')
        if isinstance(ask, str) and ask.strip() and ask.strip().lower() not in {'null', 'none', 'nil', 'n/a', 'na'}:
            self.ask = ask.strip()[:400]
        for (key, cast) in (('must_surface', int), ('named_only', bool), ('need_reason', bool), ('need_two_listings', bool), ('wait_for_shopper', bool), ('exact_first', bool)):
            if key in data and data[key] is not None:
                try:
                    setattr(self, key, cast(data[key]))
                except (TypeError, ValueError):
                    pass
        if self.candidate_keys:
            self.named_only = True
            self.exact_first = True
        ask0 = _ask_only(self.goal)
        if _TRADEOFF_RE.search(ask0):
            self.need_two_listings = True
            self.shopper_choice = True
        if _HIDDEN_SPEC_RE.search(ask0):
            self.wait_for_shopper = True
        if 'shopper_choice' in data and data['shopper_choice'] is not None:
            self.shopper_choice = bool(data['shopper_choice']) or self.shopper_choice
        for key in ('category', 'brand', 'search_query', 'notes'):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                setattr(self, key, val.strip())
        extra = data.get('match_words')
        if isinstance(extra, list) and extra and (not self.theme):
            self.theme = [str(w).lower() for w in extra if str(w).isalnum()]
        reqs = data.get('firm_requirements')
        if isinstance(reqs, list):
            cleaned = []
            seen: set = set()
            for item in reqs:
                text = ''
                if isinstance(item, str):
                    text = item.strip()
                elif isinstance(item, dict):
                    text = str(item.get('text') or item.get('requirement') or item.get('need') or '').strip()
                if not text:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(text[:160])
                if len(cleaned) >= 12:
                    break
            if cleaned:
                self.firm_requirements = cleaned
        spoken_axes = _axes_from_text(_ask_only(self.goal))
        parsed = []
        axes = data.get('rank_axes')
        if isinstance(axes, list):
            seen: set = set()
            for ax in axes:
                if not isinstance(ax, dict):
                    continue
                hint = str(ax.get('from') or ax.get('need') or ax.get('id') or '').strip()
                if not hint:
                    continue
                key = hint.lower()
                if key in seen:
                    continue
                seen.add(key)
                direction = str(ax.get('direction') or 'max').lower()
                if direction not in ('max', 'min'):
                    direction = 'min' if any((w in key for w in ('lowest', 'smallest', 'least', 'cheap'))) else 'max'
                parsed.append({'direction': direction, 'from': hint[:120]})
                if len(parsed) >= 4:
                    break
        self.rank_axes = spoken_axes or parsed or self.rank_axes
        if _TRADEOFF_RE.search(_ask_only(self.goal)) and (not spoken_axes) and (not self.candidate_keys):
            self.need_two_listings = True
            self.shopper_choice = True
            self.rank_axes = []
        cfs = data.get('claim_fields')
        if isinstance(cfs, list):
            self.claim_fields = [str(x).strip() for x in cfs if str(x).strip()][:8]
        if 'brand_is_preference' in data and data['brand_is_preference'] is not None:
            self.brand_is_preference = bool(data['brand_is_preference'])
        ask = _ask_only(self.goal)
        self.need_reason = bool(_REASON_RE.search(ask))
        self.exact_first = bool(self.candidate_keys) or _exact_pair_ask(ask)
        cleaned = _usable_brand(self.brand)
        if self.brand and not cleaned:
            self.brand = ''
        else:
            self.brand = cleaned
        if self.need_reason and self.brand and (not re.search('\\bprefer\\b', ask, re.I)):
            self.brand_is_preference = False
        self.fingerprint = json.dumps({'ask': self.ask, 'must_surface': self.must_surface, 'named_only': self.named_only, 'need_reason': self.need_reason, 'need_two_listings': self.need_two_listings, 'wait_for_shopper': self.wait_for_shopper, 'exact_first': self.exact_first, 'category': self.category, 'brand': self.brand, 'notes': self.notes, 'firm_requirements': self.firm_requirements, 'rank_axes': self.rank_axes, 'claim_fields': self.claim_fields, 'brand_is_preference': self.brand_is_preference, 'shopper_choice': self.shopper_choice}, sort_keys=True)
class Ledger:
    def __init__(self) -> None:
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
        self.titles: dict = {}
        self.seq = 0
        self.first_add_seq: int | None = None
        self.last_add_seq: int | None = None
        self.adds = 0
        self.event_target: str | None = None
        self.dead_keys: set = set()
        self.dead_products: set = set()
        self.calls_seen: set = set()
    @staticmethod
    def key(pid: Any, sku: Any) -> str:
        return str(pid) + '::' + str(sku or '')
    def record(self, name: str, args: dict, obs: dict, brief: Brief) -> None:
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
            (pid, sku) = (added.get('product_id'), added.get('sku'))
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
    def _sync_cart(self, obs: dict) -> None:
        rows = obs.get('cart')
        if not isinstance(rows, list):
            return
        pairs = [(r.get('product_id'), r.get('sku')) for r in rows if isinstance(r, dict) and r.get('product_id')]
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
    def _note_row(self, row: dict, brief: Brief) -> None:
        pid = row.get('product_id')
        if not pid:
            return
        title = str(row.get('title') or '')
        if not brief.theme or all((t in _tokens(title) for t in brief.theme)):
            self.identities.add(str(pid))
        if title:
            self.titles[str(pid)] = title
        if not row.get('sku'):
            return
        k = self.key(pid, row.get('sku'))
        self.rows[k] = row
        if row.get('price') is not None:
            self.checked.setdefault(k, {'in_stock': bool(row.get('in_stock', True)), 'price': _num(row.get('price'))})
    def _note_product(self, prod: dict, brief: Brief) -> None:
        pid = prod.get('product_id')
        if not pid:
            return
        self.products[str(pid)] = prod
        title = str(prod.get('title') or '')
        if title:
            self.titles[str(pid)] = title
        if prod.get('listing_id'):
            self.listing_ids.add(str(prod['listing_id']))
            self.listing_of[str(pid)] = str(prod['listing_id'])
        if not brief.theme or all((t in _tokens(str(prod.get('title') or '')) for t in brief.theme)):
            self.identities.add(str(pid))
        for v in prod.get('variants') or []:
            if not isinstance(v, dict) or not v.get('sku'):
                continue
            k = self.key(pid, v.get('sku'))
            self.checked[k] = {'in_stock': bool(v.get('in_stock')), 'price': _num(v.get('price'))}
            self.checked_seq[k] = self.seq
            self.options[k] = str(v.get('options') or '')
    def status(self, pid: str, sku: str) -> dict | None:
        return self.checked.get(self.key(pid, sku))
def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:
        return [{'role': 'assistant', 'content': 'agent error: ' + type(exc).__name__ + ': ' + str(exc)[:200]}]
__all__ = ['agent_main']
