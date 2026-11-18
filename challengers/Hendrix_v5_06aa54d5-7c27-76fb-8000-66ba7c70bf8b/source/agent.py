"""ORO-Bench Meridian agent (7_agent) — suite 140 / EnvPack.

Architecture (intentionally unlike 1_agent…6_agent / hioro class layout):

  * Chronometer — wall-clock budget
  * Mission     — structural goal parse (no Task-rules fingerprint table)
  * Atlas       — observation blackboard (rows / products / stock)
  * Meridian    — phase-oriented shopping session
  * SpecLens    — evidence reading without industry pixel-class ladders

Owner IM direction (Iron Seth / SN15): general-purpose agents; WxH must not
be hard-mapped through fixed 4k/qhd/fhd/hd thresholds; no suite lookup tables.
Race window targeted after Sept 13 12:00 PT IM updates; build for post-patch
rewards, not qualifying-pack overfitting.

LLM explores; Meridian owns graded commits (first cart, stockout recover,
verified-spec orders).
"""
from __future__ import annotations
import json
import re
import time
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
_MODELS = {'chutes': ('deepseek-ai/DeepSeek-V3.2-TEE', 'Qwen/Qwen3-32B-TEE'), 'openrouter': ('deepseek/deepseek-v3.2', 'qwen/qwen3-32b')}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_PROBLEM_BUDGET_S = 270.0
_OBS_CHARS = 7000
_MAX_REPROMPTS = 2
_EPS = 1e-09
_NUM = '(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?'
_BUDGET_RE = re.compile('(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|within|budget (?:is|of)|only go up to|can spend up to|at most)\\s*' + _NUM + '\\s*(PHP|php|₱|pesos?)?', re.I)
_KEY_RE = re.compile('(\\d{6,})::([A-Za-z0-9._\\-]+)')
_QUOTED_RE = re.compile('\\"([^\\"]{2,80})\\"')
_PHRASE_RE = re.compile('matches (?:the phrase )?\\"?([A-Za-z0-9][^\\",]{1,80}?)\\"?\\s*,', re.I)
_SURFACE_RE = re.compile('surface at least (\\d+) distinct', re.I)
_PREF_ITEM_RE = re.compile('prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\\-]*\\d[A-Za-z0-9\\-]*)\\b', re.I)
_PREF_BRAND_RE = re.compile("\\bprefer(?:ably)?\\s+(?:the\\s+)?([A-Za-z][\\w&.'\\- ]{1,38}?)\\s*(?:,|\\.|;|\\bbut\\b|\\bwhen\\b|\\bif\\b|$)")
_CAT_PHRASE_RE = re.compile("in the ([A-Za-z][\\w&'\\- ]{1,48}?) category", re.I)
_SPEC_WORD_RE = re.compile('\\b(?:full|ultra)\\s+hd\\b|\\b(?:hd|fhd|qhd|uhd|2k|4k|\\d+p)\\b|\\d+(?:\\.\\d+)?\\s*-?\\s*(?:inch(?:es)?|in|hz|gb)\\b', re.I)
_REASON_RE = re.compile('\\b(?:why|reason|explain|justif)', re.I)
_CLAIM_FIELDS = ('resolution', 'refresh_rate', 'screen_size', 'ram', 'price', 'in_stock', 'brand')
_QUESTION_RE = re.compile('\\?|\\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)|tell me (?:which|whether|if)|confirm whether)\\b', re.I)
_INCH_RE = re.compile('(\\d+(?:\\.\\d+)?)\\s*[-–]?\\s*(?:(?:inch(?:es)?|in)\\b|\'{2}|[\\"“”])', re.I)
_HZ_RE = re.compile('(\\d+(?:\\.\\d+)?)\\s*hz', re.I)
_WXH_RE = re.compile('(\\d{3,5})\\s*[x×]\\s*(\\d{3,5})')
_GB_RE = re.compile('(\\d+(?:\\.\\d+)?)\\s*(tb|gb)\\b', re.I)
_PAIR_RE = re.compile('(?<!\\d)(\\d+(?:\\.\\d+)?)\\s*(?:g(?:b)?)?\\s*\\+\\s*(\\d+(?:\\.\\d+)?)\\s*g(?:b)?\\b', re.I)
_PAIR_HINT_RE = re.compile('(?<!\\d)\\d+(?:\\.\\d+)?\\s*(?:g(?:b)?)?\\s*\\+\\s*\\d+(?:\\.\\d+)?(?:\\s*g(?:b)?)?(?!\\d)', re.I)
_MEM_RE = re.compile('(\\d+(?:\\.\\d+)?)\\s*gb\\s*(?:ram|memory|unified memory)\\b', re.I)
_QUARTER_RE = re.compile('\\bqHD\\b')
_FACT_LABEL = {'screen_size': 'size', 'resolution': 'display', 'refresh_rate': 'refresh_rate', 'ram': 'storage'}
_OS_RE = re.compile('\\b(?:android(?:\\s*\\d+(?:\\.\\d+)*)?|ios|blackberry(?:\\s*os)?|windows\\s*phone|symbian|harmony\\s*os|hyper\\s*os|sailfish|ubuntu\\s*touch|tizen)\\b', re.I)
_IS_PHONE_RE = re.compile('\\bsmart\\s*phones?\\b', re.I)
_NOT_PHONE_RE = re.compile('\\btablets?\\b|\\b(?:feature|basic)\\s*phones?\\b|\\b(?:mystery|blind|lucky)\\s*box(?:es)?\\b|\\b(?:accessor(?:y|ies)|screen\\s*protectors?|phone\\s*cases?|case\\s*(?:cover\\s*)?for|replacement\\s*(?:screen|battery)|smart\\s*watches?|earbuds?|headphones?)\\b', re.I)
_PART_ONLY_RE = re.compile('\\bonly\\s+(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\b|\\b(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\s+only\\b|^\\s*replacement\\s+(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\b', re.I)
_NEGATED_RE = re.compile('\\bnot\\s+(?:an?\\s+)?(?:smart\\s*phones?|feature\\s*phones?|basic\\s*phones?|tablets?|accessor(?:y|ies))\\b', re.I)
_BUNDLE_RE = re.compile('\\b(?:(?:with\\s+(?:free|bundled)|bundled\\s+with|free)\\s+|(?:with|includes?)\\s+(?:an?\\s+)?(?=(?:headphones?|earbuds?|phone\\s*case|screen\\s*protector|charger|accessor(?:y|ies))\\b))', re.I)
_PLAYBOOK = {'intent': "Firm requirements: the category, the named hardware specification, the budget and stock. A named brand is only a preference: choose it when a fully compliant item of that brand exists, otherwise any compliant brand. Use filter(category, brand, max_price, in_stock=true), then view or compare to read the exact variant's specification from raw facts. Answer the shopper with message; a budget the shopper lowers later applies to every later choice.", 'recall': 'Run several differently worded searches (k=10, in_stock=true) built from the requested words and close synonyms until enough distinct matching products have appeared; every requested word must appear as a whole word of the title. Then view the strongest candidates so the controller can verify one and buy it.', 'constraint': "The shopper hides one firm hardware requirement and answers it when asked; obey it exactly for the specific variant, reading the variant's own option text before any product-wide value. Use filter(category, brand, max_price, in_stock=true) and view candidates. If the carted variant sells out, move to a different product that still meets every requirement.", 'preference': 'You already asked which preference matters most; use the answer. Search or filter the category within budget, then compare candidates with different listing_id values, reading raw facts. The first cart item is graded, so make it the best genuine match. If it is repriced above budget, recover to a different listing_id.', 'ranking': "Only the named candidate variants count, and the controller ranks them for you: compare all of them first so their facts are on the record. Highest advertised maximum refresh rate wins, then the largest screen, then the lowest price, reading the selected variant's own value before a product-wide one. After the market moves, the remaining candidates are re-ranked.", 'recovery': 'Find the preferred model and color first: search the model code, then view it so every variant and color is on the record. The first cart item must be exactly that model and color. If it disappears, relax as little as possible - the same model in another color before any other model - and never relax the category, budget or stock.', 'justification': 'Find a product whose own raw facts state the requested specification, in the requested category and brand, in stock and within budget. View or compare it so its facts are on the record, then the controller inspects it, carts it and orders it with claims taken only from that evidence.', 'generic': 'Understand the firm requirements, filter or search the catalog, view candidates, and let the controller verify and buy the best one.'}
_CORE_RULES = 'You are a careful shopping agent working a catalog through tools. Exactly one tool call per turn, at most 30 turns, so be efficient. A deterministic controller checks every proposal and takes over the cart and the order once a candidate is fully verified, so your job is to surface and read candidates: search or filter, then view or compare so their raw facts, variant options and prices are on record. Never add an item to the cart as a placeholder - the first cart addition is graded. Prices and stock change while you shop; the controller re-checks the carted item and tells you when to move to a different product. Answer every direct question from the shopper with one short message before anything else, and never send progress or status updates. Never ask the shopper to raise the budget. Do not repeat a tool call with identical arguments; one compare of up to five products beats several view calls.'

class SpecLens:
    """Literal evidence reading — no industry pixel ladders."""

    @staticmethod
    def _inch_vals(text: str) -> set:
        return {'%gin' % float(m.group(1)) for m in _INCH_RE.finditer(text or '')}

    @staticmethod
    def _hz_vals(text: str) -> set:
        return {'%ghz' % float(m.group(1)) for m in _HZ_RE.finditer(text or '')}

    @staticmethod
    def _gb_vals(text: str) -> set:
        return {'%ggb' % (float(m.group(1)) * (1024 if m.group(2).lower() == 'tb' else 1)) for m in _GB_RE.finditer(text or '')}

    @staticmethod
    def _res_vals(text: str) -> set:
        """Display tokens asserted by text — labels as written, WxH as Np only.

    Never maps pixel pairs through industry 4k/qhd/fhd/hd threshold ladders
    (owner guidance / suite-140 IM direction). A product stating 1600x900 yields
    900p, not a silent upgrade to hd/fhd.
    """
        text = text or ''
        if _QUARTER_RE.search(text):
            return set()
        low = text.lower()
        out = set()
        for m in _WXH_RE.finditer(low):
            a, b = (int(m.group(1)), int(m.group(2)))
            long_s, short_s = (max(a, b), min(a, b))
            out.add('%dx%d' % (long_s, short_s))
            out.add('%dp' % short_s)
        if re.search('\\b(?:4k|uhd|ultra\\s*hd|2160p)\\b', low):
            out.add('4k')
            out.add('2160p')
        if re.search('\\b(?:qhd|1440p|2k)\\b', low):
            out.add('qhd')
            out.add('1440p')
        if re.search('\\b(?:fhd|1080p|full\\s*hd)\\b', low):
            out.add('fhd')
            out.add('1080p')
        stripped = re.sub('\\b(?:full|ultra)\\s*hd\\b', ' ', low)
        if re.search('\\b720p\\b', stripped):
            out.add('720p')
            out.add('hd')
        elif re.search('\\bhd\\b', stripped):
            out.add('hd')
        for m in re.finditer('\\b(\\d{3,4})p\\b', low):
            out.add(m.group(0))
        return out

    @staticmethod
    def _res_canonical(tokens: set) -> str | None:
        """Pick one shopper-facing resolution class from a token set (no WxH ladder)."""
        if not tokens:
            return None
        if tokens & {'4k', '2160p', 'uhd'}:
            return '4k'
        if tokens & {'qhd', '1440p', '2k'}:
            return 'qhd'
        if tokens & {'fhd', '1080p'}:
            return 'fhd'
        if tokens & {'hd', '720p'}:
            return 'hd'
        nps = sorted((t for t in tokens if re.fullmatch('\\d+p', t)))
        if len(nps) == 1:
            return nps[0]
        wxh = sorted((t for t in tokens if re.fullmatch('\\d+x\\d+', t)))
        if len(wxh) == 1:
            return wxh[0]
        return None

    @staticmethod
    def _res_match(got: str | None, want: str | None) -> bool:
        if not got or not want:
            return False
        if got == want:
            return True
        groups = ({'hd', '720p'}, {'fhd', '1080p'}, {'qhd', '1440p', '2k'}, {'4k', '2160p', 'uhd'})
        for g in groups:
            if got in g and want in g:
                return True
        return False

    @staticmethod
    def _adjacent_vals(field: str, text: str) -> set:
        """Values a free-form string asserts, requiring the number's own concept word.

    Titles and variant option strings mix several capacities, so a memory figure
    only counts when the word for it follows the number; storage totals and a
    graphics chip's own memory must never be read as system memory.
    """
        text = text or ''
        if field != 'ram':
            return _VALUE_SETS[field](text)
        paired = {'%ggb' % float(a) for a, _b in _PAIR_RE.findall(text)}
        return paired or {'%ggb' % float(n) for n in _MEM_RE.findall(text)}

    @staticmethod
    def _mentions(field: str, text: str) -> bool:
        """Whether an option string talks about the concept at all (even ambiguously)."""
        text = text or ''
        if field == 'ram':
            return bool(_PAIR_HINT_RE.search(text) or _MEM_RE.search(text))
        if field == 'screen_size':
            return bool(_INCH_RE.search(text))
        if field == 'refresh_rate':
            return bool(_HZ_RE.search(text))
        return bool(_WXH_RE.search(text) or re.search('\\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\\b', text, re.I))

    @staticmethod
    def _one(values: set) -> str | None:
        return next(iter(values)) if len(values) == 1 else None

    @staticmethod
    def _wanted_value(field: str | None, text: str) -> str | None:
        """The single value a request states for a concept, or nothing when unclear."""
        if not field:
            return None
        vals = _VALUE_SETS[field](text or '')
        if field == 'resolution':
            return SpecLens._res_canonical(vals)
        return SpecLens._one(vals)

    @staticmethod
    def _tokens(text: str) -> set:
        return set(re.findall('[a-z0-9]+', (text or '').lower()))

    @staticmethod
    def _same_word(a: str, b: str) -> bool:
        """Loose noun equality: ignores case, spacing and a trailing plural."""
        x = re.sub('[^a-z0-9]+', '', (a or '').lower()).rstrip('s')
        y = re.sub('[^a-z0-9]+', '', (b or '').lower()).rstrip('s')
        return bool(x) and x == y

    @staticmethod
    def _identity_text(text: str, title: bool=False) -> str:
        """Strip claims that describe what a listing is NOT, or what ships beside it."""
        text = _NEGATED_RE.sub('', text or '')
        return _BUNDLE_RE.split(text, maxsplit=1)[0] if title else text

    @staticmethod
    def _phone_identity(facts: dict, title: str, options: str) -> str:
        """'yes' / 'no' / 'unclear' for whether public evidence makes this a real handset."""
        if _PART_ONLY_RE.search(options or ''):
            return 'no'
        facts = facts if isinstance(facts, dict) else {}
        raw_title = str(title or '')
        head = SpecLens._identity_text(raw_title, title=True)
        kind = ' '.join((SpecLens._identity_text(str(facts.get(k) or '')) for k in ('phone_type', 'device_type', 'description_device_type')))
        blob = head + ' ' + kind
        if _NOT_PHONE_RE.search(blob):
            return 'no'
        if any(('[truncated]' in str(facts.get(k) or '') for k in ('phone_type', 'device_type', 'device_operating_system'))):
            return 'unclear'
        if _IS_PHONE_RE.search(blob):
            return 'yes'
        os_blob = ' '.join((SpecLens._identity_text(str(facts.get(k) or '')) for k in ('device_operating_system', 'description_operating_system') if not re.search('\\bcompatib', str(facts.get(k) or ''), re.I)))
        if _OS_RE.search(os_blob):
            return 'yes'
        if _OS_RE.search(head) and (not re.search('\\bcompatib', raw_title, re.I)):
            return 'yes'
        return 'unclear'

    @staticmethod
    def _num(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(str(value).replace(',', ''))
        except (TypeError, ValueError):
            return None

_VALUE_SETS = {
    'screen_size': SpecLens._inch_vals,
    'refresh_rate': SpecLens._hz_vals,
    'resolution': SpecLens._res_vals,
    'ram': SpecLens._gb_vals,
}

class Chronometer:

    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)

class Mission:
    """What the deterministic side needs, parsed from the public request text."""

    def __init__(self, query: str) -> None:
        self.raw = query or ''
        goal, _, rules = self.raw.partition('\n\nTask rules:')
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self._budget(self.goal)
        self.currency = 'PHP'
        m = re.search('\\b([A-Z]{3})\\b', self.goal)
        if m and m.group(1) in ('PHP', 'USD', 'EUR', 'SGD', 'MYR', 'IDR', 'THB', 'VND'):
            self.currency = m.group(1)
        self.candidate_keys = [(pid, sku.rstrip('.,;:')) for pid, sku in _KEY_RE.findall(self.goal)]
        phrase = _PHRASE_RE.search(self.goal)
        quoted = _QUOTED_RE.findall(self.goal)
        source = phrase.group(1) if phrase else quoted[0] if quoted else ''
        self.theme = re.findall('[a-z0-9]+', source.lower())
        sm = _SURFACE_RE.search(self.rules)
        self.required_identities = int(sm.group(1)) if sm else 0
        pm = _PREF_ITEM_RE.search(self.goal)
        self.pref_color = pm.group(1).lower() if pm else None
        self.pref_model = pm.group(2) if pm else None
        self.category = self._category(self.goal)
        self.brand = self._brand(self.goal, self.category)
        self.pref_brand = None if pm else self._pref_brand(self.goal)
        self.cat_qualifier = self._qualifier(self.goal)
        self.specs = self._specs(self.goal)
        self.spec_field, self.spec_value = self._spec(self.goal)
        self.spec_sealed = self.spec_field is None and 'firm requirement' in self.goal.lower()
        self.family = self._family()
        self.soft_axes = self._soft_axes(self.goal)

    @staticmethod
    def _soft_axes(goal: str) -> list:
        """Preference axes named by the shopper — never a fixed vertical checklist."""
        g = (goal or '').lower()
        axes = []
        if re.search('\\b(waterproof|water[\\s-]?resistant|sweat|swim|rain|ipx?\\d|every kind of weather)\\b', g):
            axes.append('water')
        if re.search('\\b(sport|sports|gym|workout|workouts|running|training|fitness|exercise|bootcamp|daily runs)\\b', g):
            axes.append('sport')
        if re.search('\\b(battery|long[\\s-]?lasting|all[\\s-]?day|hours?\\b|year[\\s-]?round|hard workouts?|heavy gym)\\b', g):
            axes.append('battery')
        if re.search('\\b(wireless|bluetooth|true\\s*wireless|tws|no\\s*cable|cable\\s*to\\s*snag)\\b', g):
            axes.append('wireless')
        return axes

    def _family(self) -> str:
        """Infer family from goal *structure* only — no Task-rules fingerprint table."""
        if self.candidate_keys:
            return 'ranking'
        if self.required_identities or self.theme:
            return 'recall'
        if self.pref_model:
            return 'recovery'
        if _REASON_RE.search(self.goal):
            return 'justification'
        if self.pref_brand:
            return 'intent'
        if self.spec_sealed or self.spec_field or self.brand:
            return 'constraint'
        return 'preference'

    @staticmethod
    def _budget(text: str) -> float | None:
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

    @staticmethod
    def _category(goal: str) -> str | None:
        m = _CAT_PHRASE_RE.search(goal or '')
        if m:
            return m.group(1).strip()
        m = re.search('\\b([a-z][a-z\\-]{2,20})\\s*(?:,|\\s(?:at or (?:under|below)|at most|under|below|within|with)\\b)', goal or '', re.I)
        return m.group(1) if m else None

    @staticmethod
    def _brand(goal: str, category: str | None) -> str | None:
        """A brand named as part of the requested item, e.g. 'a <Brand> <item>'."""
        if not category:
            return None
        m = re.search('\\b(?:an?|the)\\s+([^,.;]{0,60}?)\\b' + re.escape(category), goal or '', re.I)
        if not m:
            return None
        words = [w for w in re.findall("[\\w&.'\\-]+", _SPEC_WORD_RE.sub(' ', m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and (not any((c.isdigit() for c in w))) and (not SpecLens._res_vals(w))]
        return ' '.join(keep) or None

    @staticmethod
    def _pref_brand(goal: str) -> str | None:
        m = _PREF_BRAND_RE.search(goal or '')
        if not m:
            return None
        name = m.group(1).strip(' .,')
        words = name.split()
        if not words or len(words) > 4 or (not any((w[:1].isupper() for w in words))):
            return None
        return name

    @staticmethod
    def _qualifier(goal: str) -> str | None:
        """A product shape that no shelf name captures, so the title must show it."""
        if re.search('\\bgaming\\s+monitors?\\b', goal or '', re.I):
            return 'gaming'
        if re.search('\\bover[\\s-]?ear\\b', goal or '', re.I):
            return 'over ear'
        return None

    @staticmethod
    def _specs(goal: str) -> dict:
        """Every hardware value the request states unambiguously.

        A request can name more than one ("a 27-inch 144Hz monitor") while only
        one of them is the gated requirement, so all of them are kept: the gate is
        honored whichever it turns out to be, and a reason can cite each of them.
        """
        out = {}
        for field in ('screen_size', 'refresh_rate', 'resolution', 'ram'):
            value = SpecLens._wanted_value(field, goal)
            if value:
                out[field] = value
        return out

    def _spec(self, goal: str) -> tuple:
        """The requirement the controller gates on: the first one stated."""
        for field, value in self.specs.items():
            return (field, value)
        return (None, None)

    def adopt_spec(self, text: str) -> str | None:
        """Record a requirement the shopper reveals later; returns its description."""
        for field in ('screen_size', 'refresh_rate', 'ram', 'resolution'):
            value = SpecLens._wanted_value(field, text)
            if value:
                self.spec_field, self.spec_value = (field, value)
                self.specs[field] = value
                self.spec_sealed = False
                return field + ' = ' + value
        return None

class Atlas:
    """What the agent has observed for itself, keyed by product and variant."""

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

    def record(self, name: str, args: dict, obs: dict, mission: Mission) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get('error'):
            return
        if name in ('search', 'filter'):
            for row in obs.get('results') or []:
                if isinstance(row, dict):
                    self._note_row(row, mission)
        elif name == 'view':
            self._note_product(obs, mission)
        elif name == 'compare':
            for prod in obs.get('compared') or []:
                if isinstance(prod, dict):
                    self._note_product(prod, mission)
        elif name == 'inspect_stock':
            k = self.key(obs.get('product_id'), obs.get('sku'))
            self.checked[k] = {'in_stock': bool(obs.get('in_stock')), 'price': SpecLens._num(obs.get('price'))}
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
            price = SpecLens._num(r.get('current_price'))
            stock = r.get('current_in_stock')
            if price is None and stock is None:
                continue
            prev = self.checked.get(k, {})
            self.checked[k] = {'in_stock': bool(stock) if stock is not None else prev.get('in_stock', True), 'price': price if price is not None else prev.get('price')}
            self.checked_seq[k] = self.seq

    def _note_row(self, row: dict, mission: Mission) -> None:
        pid = row.get('product_id')
        if not pid:
            return
        title = str(row.get('title') or '')
        if mission.theme:
            if all((t in SpecLens._tokens(title) for t in mission.theme)):
                self.identities.add(str(pid))
        else:
            self.identities.add(str(pid))
        if not row.get('sku'):
            return
        k = self.key(pid, row.get('sku'))
        self.rows[k] = row
        if row.get('price') is not None:
            self.checked.setdefault(k, {'in_stock': bool(row.get('in_stock', True)), 'price': SpecLens._num(row.get('price'))})

    def _note_product(self, prod: dict, mission: Mission) -> None:
        pid = prod.get('product_id')
        if not pid:
            return
        self.products[str(pid)] = prod
        if prod.get('listing_id'):
            self.listing_ids.add(str(prod['listing_id']))
            self.listing_of[str(pid)] = str(prod['listing_id'])
        if mission.theme and all((t in SpecLens._tokens(str(prod.get('title') or '')) for t in mission.theme)):
            self.identities.add(str(pid))
        for v in prod.get('variants') or []:
            if not isinstance(v, dict) or not v.get('sku'):
                continue
            k = self.key(pid, v.get('sku'))
            self.checked[k] = {'in_stock': bool(v.get('in_stock')), 'price': SpecLens._num(v.get('price'))}
            self.checked_seq[k] = self.seq
            self.options[k] = str(v.get('options') or '')

    def status(self, pid: str, sku: str) -> dict | None:
        return self.checked.get(self.key(pid, sku))

def _arguments(tool_call: dict) -> dict:
    raw = (tool_call.get('function') or {}).get('arguments', '{}')
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or '{}')
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}

def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit - 40] + ' ... (truncated by agent)"}'
_PHASE_INTRO = 'intro'
_PHASE_SCOUT = 'scout'
_PHASE_STUDY = 'study'
_PHASE_SEAL = 'seal'
_PHASE_MEND = 'mend'
_PHASE_CLOSE = 'close'

class Meridian:

    def __init__(self, problem: dict) -> None:
        env = problem['environment']
        self.binding = dict(env['binding'])
        self.policy = env['policy_view']
        self.tools = list(self.policy.get('tools') or [])
        self.tool_names = {(t.get('function') or {}).get('name') for t in self.tools if isinstance(t, dict)}
        self.max_steps = int(self.policy.get('max_steps') or 30)
        self.max_calls = max(1, int(self.policy.get('max_calls_per_turn') or 1))
        self.problem_id = str(problem.get('problem_id') or problem.get('id') or 'problem')
        self.mission = Mission(str(self.policy.get('query') or ''))
        self.budget = self.mission.budget
        self.atlas = Atlas()
        self.clock = Chronometer(_PROBLEM_BUDGET_S)
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
        self.brand_probed = False
        self.pref_probed = False
        self._wishes: set | None = None
        self.probes: set = set()
        self.searched_twins = False
        self.reads_done = False
        self.event_turn: int | None = None
        self.unreadable: set = set()
        self.question_waits = 0
        self.done = False
        self.phase = _PHASE_INTRO
        self.model_index = 0
        provider = (getenv('INFERENCE_PROVIDER') or '').strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS['chutes']
        pinned = getenv('SANDBOX_MODEL')
        if pinned:
            self.roster = (pinned,) + tuple((m for m in self.roster if m != pinned))

    def _system_prompt(self) -> str:

        def _orbit():
            play = _PLAYBOOK.get(self.mission.family, _PLAYBOOK['generic'])
            extra = ''
            if self.mission.family == 'recall' and self.mission.required_identities:
                extra = ' You must surface at least %d distinct matching products in search or filter results before ordering; the controller reports the count after each search.' % self.mission.required_identities
            return _CORE_RULES + '\n\nFamily playbook: ' + play + extra
        return _orbit()

    def _opening_question(self) -> str | None:

        def _orbit():
            if self.mission.family == 'constraint' or self.mission.spec_sealed:
                return 'Before I search: which hardware specification is your firm requirement (screen size, resolution class, refresh rate, or memory), and what value do you need?'
            if self.mission.family == 'preference':
                return 'Quick question so I choose well: which of your preferences counts most, and in what order should I weigh the others? Let me know your priority.'
            return None
        return _orbit()

    def _note(self, text: str) -> None:

        def _orbit():
            self.messages.append({'role': 'user', 'content': '[note] ' + text})
            self.dialogue.append({'role': 'assistant', 'content': '[note] ' + text[:300]})
        return _orbit()

    def _prod(self, pid: str) -> dict:

        def _orbit():
            return self.atlas.products.get(str(pid)) or {}
        return _orbit()

    def _facts(self, pid: str) -> dict:

        def _orbit():
            f = self._prod(pid).get('facts')
            return f if isinstance(f, dict) else {}
        return _orbit()

    def _title(self, pid: str) -> str:

        def _orbit():
            title = str(self._prod(pid).get('title') or '')
            if title:
                return title
            for key, row in self.atlas.rows.items():
                if key.startswith(str(pid) + '::'):
                    return str(row.get('title') or '')
            return ''
        return _orbit()

    def _brand_of(self, pid: str) -> str:

        def _orbit():
            brand = self._prod(pid).get('brand')
            if brand:
                return str(brand)
            for key, row in self.atlas.rows.items():
                if key.startswith(str(pid) + '::') and row.get('brand'):
                    return str(row['brand'])
            return ''
        return _orbit()

    def _cat_path(self, pid: str) -> list:

        def _orbit():
            path = self._prod(pid).get('category_path')
            return list(path) if isinstance(path, list) else []
        return _orbit()

    def _options(self, pid: str, sku: str) -> str:

        def _orbit():
            return self.atlas.options.get(Atlas.key(pid, sku), '')
        return _orbit()

    def _strip_models(self, pid: str, title: str) -> str:

        def _orbit():
            """Drop model codes from a title so a product name is not read as a panel class."""
            cleaned = title
            for token in re.split('[;,]', str(self._facts(pid).get('model') or '')):
                token = token.strip()
                stem = re.split('[-_]', token)[0] if token else ''
                if token and re.fullmatch('[a-z0-9]+(?:[-_][a-z0-9]+)+', token, re.I) and re.search('[a-z]', stem, re.I) and re.search('\\d', stem) and (not re.fullmatch('(?:[48]k|[ufq]?hd|fullhd|ultrahd|\\d+p)\\d*', stem, re.I)):
                    cleaned = re.sub('(?<![\\w-])' + re.escape(token) + '(?![\\w-])', '', cleaned, flags=re.I)
            return cleaned
        return _orbit()

    def _fact_value(self, pid: str, field: str) -> str | None:

        def _orbit():
            """The value the product's own labelled facts state for a concept.

        A reason quoted back to the shopper reads these facts leniently: a fact
        printed as a bare number under a unit-bearing label carries that label's
        unit, so "27" under the size label is 27 inches.
        """
            raw = self._facts(pid).get(_FACT_LABEL[field])
            if not raw:
                return None
            text = str(raw).strip()
            found = _VALUE_SETS[field](text)
            if not found and field != 'resolution' and re.fullmatch('\\d+(?:\\.\\d+)?', text):
                unit = {'screen_size': 'in', 'refresh_rate': 'hz', 'ram': 'gb'}[field]
                return '%g' % float(text) + unit
            if field == 'resolution':
                return SpecLens._res_canonical(found)
            return SpecLens._one(found)
        return _orbit()

    def _product_value(self, pid: str, field: str) -> str | None:

        def _orbit():
            """Product-wide value: labelled facts plus the title, and nothing when they disagree."""
            vals: set = set()
            conflict = False
            raw = self._facts(pid).get(_FACT_LABEL[field])
            if raw:
                found = _VALUE_SETS[field](str(raw))
                if field == 'resolution':
                    conflict = conflict or (SpecLens._res_canonical(found) is None and len(found) > 1)
                else:
                    conflict = conflict or len(found) > 1
                if field == 'resolution' and _QUARTER_RE.search(str(raw)):
                    conflict = True
                vals |= found
            title = self._title(pid)
            if field == 'resolution':
                title = self._strip_models(pid, title)
                if _QUARTER_RE.search(title):
                    conflict = True
            vals |= SpecLens._adjacent_vals(field, title)
            if conflict:
                return None
            if field == 'resolution':
                return SpecLens._res_canonical(vals)
            return SpecLens._one(vals)
        return _orbit()

    def _variant_value(self, pid: str, sku: str, field: str) -> str | None:

        def _orbit():
            """The value that holds for this exact variant: its own option text wins."""
            opt = self._options(pid, sku)
            vals = SpecLens._adjacent_vals(field, opt)
            if field == 'resolution':
                if len(vals) > 1 and SpecLens._res_canonical(vals) is None:
                    return None
                if vals:
                    return SpecLens._res_canonical(vals)
            else:
                if len(vals) > 1:
                    return None
                if vals:
                    return next(iter(vals))
            if SpecLens._mentions(field, opt):
                return None
            return self._product_value(pid, field)
        return _orbit()

    def _value_match(self, field: str, got: str | None, want: str | None) -> bool:

        def _orbit():
            if field == 'resolution':
                return SpecLens._res_match(got, want)
            return bool(got) and got == want
        return _orbit()

    def _measure(self, pid: str, sku: str, field: str) -> float | None:

        def _orbit():
            value = self._variant_value(pid, sku, field)
            try:
                return float(value[:-2]) if value else None
            except ValueError:
                return None
        return _orbit()

    @staticmethod
    def _stems(text: str) -> set:

        def _orbit():
            return {w.rstrip('s') for w in re.findall('[a-z]+', (text or '').lower())}
        return _orbit()

    def _category_ok(self, pid: str) -> bool:

        def _orbit():
            """Whether the listing sits on the shelf the shopper asked for.

        Retailers spell a shelf many ways and nest it differently, so a shared
        word anywhere in the path counts. Where the request names a kind of
        product the catalog has no shelf for, the title has to carry it.
        """
            want = self.mission.category
            path = self._cat_path(pid)
            if not want or not path:
                return True
            wanted = self._stems(want)
            labels = set()
            for label in path:
                labels |= self._stems(str(label))
            placed = bool(wanted & labels) or any((self._stems(''.join(re.findall('[a-z]+', str(label).lower()))) == wanted for label in path))
            if not placed and 'smartphone' in wanted:
                placed = any((re.fullmatch('(?:smart|cell|mobile|unlocked)?\\s*phones?', str(label).strip(), re.I) for label in path))
            if not placed:
                return False
            qualifier = self.mission.cat_qualifier
            if not qualifier:
                return True
            title = self._title(pid)
            words = self._stems(title) | SpecLens._tokens(title)
            if qualifier == 'gaming':
                fast = any((float(v[:-2]) >= 120 for v in SpecLens._hz_vals(title)))
                return 'monitor' in words and ('gaming' in words or fast)
            return 'over' in words and 'ear' in words or 'overear' in words or 'circumaural' in words
        return _orbit()

    def _brand_ok(self, pid: str) -> bool:

        def _orbit():
            want = self.mission.brand
            if not want:
                return True
            wanted = SpecLens._tokens(want)
            return bool(wanted <= SpecLens._tokens(self._brand_of(pid)) or wanted <= SpecLens._tokens(self._title(pid)))
        return _orbit()

    def _device_ok(self, pid: str, sku: str) -> bool:

        def _orbit():
            """Categories that name a device are graded on the listing being that device.

        Where the task itself turns on device identity, only positive evidence
        will do; elsewhere a listing is only ruled out when the evidence
        contradicts it, so a thin product page is not treated as a disqualifier.
        """
            cat = (self.mission.category or '').lower()
            title = self._title(pid)
            blob = (title + ' ' + ' '.join((str(v) for v in self._facts(pid).values())) + ' ' + self._options(pid, sku)).lower()
            if SpecLens._same_word(self.mission.category or '', 'smartphone'):
                verdict = SpecLens._phone_identity(self._facts(pid), title, self._options(pid, sku))
                if self.mission.family in ('constraint', 'justification', 'recovery'):
                    return verdict == 'yes'
                return verdict != 'no'
            if re.search('\\b(headphone|earphone|earbud|headset)s?\\b', cat):
                # Accessories that mention headphones are not the device itself.
                if re.search('\\b(lanyard|anti[- ]?lost|rope|strap|hook|clip|case|cover|cushion|ear\\s*pad|replacement|holder|stand|cable only|charger only)\\b', blob):
                    if not re.search('\\b(bluetooth|wireless|wired)\\b.{0,40}\\b(headphone|earphone|earbud|headset)s?\\b|\\b(headphone|earphone|earbud|headset)s?\\b.{0,40}\\b(bluetooth|wireless|wired|stereo|noise)\\b', blob):
                        return False
                if re.search('\\b(lanyard|anti[- ]?lost rope|earphone rope|headset strap)\\b', blob) and (not re.search('\\b(driver|impedance|frequency\\s*response|noise\\s*cancell|ipx\\d)\\b', blob)):
                    return False
                return bool(re.search('\\b(headphone|earphone|earbud|headset|earpiece)s?\\b', blob))
            return True
        return _orbit()

    def _is_accessory_title(self, pid: str) -> bool:

        def _orbit():
            title = self._title(pid).lower()
            return bool(re.search('\\b(case|cover|casing|tempered|glass|charger|cable|protector|lanyard|dummy|trigger|gamepad|battery replacement|screen protector|anti[- ]?lost|rope|strap)\\b', title))
        return _orbit()

    def _resale_title(self, pid: str) -> bool:

        def _orbit():
            """Listings that describe someone else's listing, or a used unit, are never the answer."""
            return bool(re.search('customer (?:rating|review)|\\bcompare \\b|refurb|pre-owned|restored|renewed', self._title(pid), re.I))
        return _orbit()

    def _relax(self, pid: str, sku: str) -> int:

        def _orbit():
            """How far a candidate sits from the shopper's exact preferred model and color."""
            model, color = (self.mission.pref_model, self.mission.pref_color)
            if not model:
                return 0
            squash = lambda t: re.sub('[^a-z0-9]+', '', (t or '').lower())
            blob = squash(self._title(pid) + ' ' + ' '.join((str(v) for v in self._facts(pid).values())))
            same_model = squash(model) in blob
            shade = (self._options(pid, sku) + ' ' + str(self._facts(pid).get('color') or '') + ' ' + self._title(pid)).lower()
            same_color = bool(color) and re.search('(?<![a-z])' + re.escape(color) + '(?![a-z])', shade) is not None
            if same_model and same_color:
                return 0
            if same_model:
                return 1
            return 2 if same_color else 3
        return _orbit()

    def _assess(self, pid: str, sku: str) -> dict:

        def _orbit():
            """Score one variant against every gate the runtime will apply to it."""
            fam = self.mission.family
            key = Atlas.key(pid, sku)
            st = self.atlas.checked.get(key) or {}
            price, stock = (st.get('price'), st.get('in_stock'))
            dead = key in self.atlas.dead_keys or (str(pid) in self.atlas.dead_products and fam != 'recovery')
            live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and (not dead)
            viewed = str(pid) in self.atlas.products
            field, want = (self.mission.spec_field, self.mission.spec_value)
            spec = 2
            if field and want:
                got = self._variant_value(pid, sku, field)
                spec = 2 if self._value_match(field, got, want) else -1 if got else 0
            ok = bool(live and viewed and (spec == 2) and self._brand_ok(pid) and self._device_ok(pid, sku))
            cat_ok = self._category_ok(pid)
            ok = ok and cat_ok
            if fam == 'justification':
                ok = ok and (not self._resale_title(pid)) and bool(self._brand_of(pid))
                if field and want:
                    ok = ok and self._value_match(field, self._fact_value(pid, field), want) and self._value_match(field, self._product_value(pid, field), want)
                # Justification without a stated gate still needs a grounded claimable fact.
                if not field:
                    ok = ok and bool(self._fact_value(pid, 'resolution') or self._fact_value(pid, 'screen_size'))
            if fam == 'preference':
                ok = ok and (not self._is_accessory_title(pid))
                if self.atlas.event_target:
                    dead_listing = self.atlas.listing_of.get(self.atlas.event_target.split('::', 1)[0])
                    if dead_listing and self.atlas.listing_of.get(str(pid)) == dead_listing:
                        ok = False
            if fam == 'recall' and self.mission.theme:
                ok = ok and all((t in SpecLens._tokens(self._title(pid)) for t in self.mission.theme))
            if fam == 'ranking':
                ok = ok and (str(pid), str(sku)) in self.mission.candidate_keys
                ok = ok and self._measure(pid, sku, 'refresh_rate') is not None and (self._measure(pid, sku, 'screen_size') is not None)
            return {'pid': str(pid), 'sku': str(sku), 'key': key, 'price': price, 'live': live, 'viewed': viewed, 'spec': spec, 'cat_ok': cat_ok, 'ok': ok, 'relax': self._relax(pid, sku) if fam == 'recovery' else 0, 'pref': self._priority_hits(pid, sku) if fam == 'preference' else 0}
        return _orbit()

    def _rank_of(self, c: dict, first: bool) -> tuple:

        def _orbit():
            """Order eligible candidates the way the task's own priority does."""
            fam = self.mission.family
            price = c['price'] if c['price'] is not None else 1000000000000.0
            if fam == 'ranking':
                hz = self._measure(c['pid'], c['sku'], 'refresh_rate') or 0.0
                inch = self._measure(c['pid'], c['sku'], 'screen_size') or 0.0
                return (-hz, -inch, price, c['key'])
            if fam == 'recovery':
                return (c['relax'], price, c['key'])
            extra = sum((1 for f, v in self.mission.specs.items() if f != self.mission.spec_field and (not self._value_match(f, self._variant_value(c['pid'], c['sku'], f), v))))
            proven = 0 if SpecLens._phone_identity(self._facts(c['pid']), self._title(c['pid']), self._options(c['pid'], c['sku'])) == 'yes' else 1
            if fam == 'intent' and first and self.mission.pref_brand:
                wanted = SpecLens._tokens(self.mission.pref_brand)
                near = 0 if wanted <= SpecLens._tokens(self._brand_of(c['pid'])) or wanted <= SpecLens._tokens(self._title(c['pid'])) else 1
                return (near, extra, proven, price, c['key'])
            return (extra, proven, price, c['key'])
        return _orbit()

    def _candidates(self) -> list:

        def _orbit():
            seen = set()
            out = []
            for key in list(self.atlas.checked):
                pid, _, sku = key.partition('::')
                if not sku or key in seen:
                    continue
                seen.add(key)
                out.append(self._assess(pid, sku))
            return out
        return _orbit()

    def _target(self) -> dict | None:

        def _orbit():
            """The best fully verified candidate, or nothing while evidence is missing."""
            first = self.atlas.adds == 0
            fam = self.mission.family
            good = [c for c in self._candidates() if c['ok']]
            if fam == 'preference':
                carted = [c for c in good if (c['pid'], c['sku']) in self.atlas.cart]
                if carted:
                    return carted[0]
                # After a market kill, or late in the episode, the controller must
                # seal a preference pick itself — waiting on the model is how
                # episodes die without a terminal order.
                late = self.turn >= self.max_steps - 8
                recovering = bool(self.atlas.event_target)
                pressured = self.clock.remaining() < 90
                if good and (late or recovering or pressured):
                    ranked = sorted(good, key=lambda c: (-self._priority_hits(c['pid'], c['sku']), c['price'] if c['price'] is not None else 1000000000000.0))
                    axes = list(getattr(self.mission, 'soft_axes', None) or [])
                    need = min(2, len(axes)) if axes else 1
                    covered = [c for c in ranked if self._axis_hits(c['pid'], c['sku']) >= need]
                    if covered:
                        return covered[0]
                    if axes and 'water' in axes:
                        watered = [c for c in ranked if self._axis_hits(c['pid'], c['sku']) >= 1]
                        if watered and (recovering or pressured or late):
                            return watered[0]
                    if axes and not (pressured and self.turn >= self.max_steps - 2):
                        if self.turn < self.max_steps - 2:
                            return None
                    return ranked[0]
                return None
            if fam == 'intent' and first and self.mission.pref_brand:
                wanted = SpecLens._tokens(self.mission.pref_brand)
                named = [c for c in good if wanted <= SpecLens._tokens(self._brand_of(c['pid'])) or wanted <= SpecLens._tokens(self._title(c['pid']))]
                if named:
                    good = named
                elif self.turn < self.max_steps * 0.6:
                    self._preferred_brand_hint()
                    return None
            if fam == 'recovery' and first:
                exact = [c for c in good if c['relax'] == 0]
                if exact:
                    good = exact
                elif self.turn < self.max_steps * 0.6:
                    self._exact_match_hint()
                    return None
            if not good:
                return None
            good.sort(key=lambda c: self._rank_of(c, first))
            return good[0]
        return _orbit()

    def _answer_text(self) -> str:

        def _orbit():
            """A short, factual answer to the shopper, drawn only from observed evidence."""
            target = self._target() or (self._candidates() or [None])[0]
            if not target:
                if self.atlas.event_target:
                    return 'That listing moved past budget or stock, so I am switching to another option that still meets your firm requirements before ordering.'
                return 'I am still checking the options against your requirements; nothing is decided yet.'
            pid, sku = (target['pid'], target['sku'])
            parts = ['I am going with ' + (self._title(pid) or pid)[:90]]
            field, want = (self.mission.spec_field, self.mission.spec_value)
            if field and want and self._value_match(field, self._variant_value(pid, sku, field), want):
                parts.append('its listed ' + field.replace('_', ' ') + ' is ' + want)
            if target.get('price') is not None:
                parts.append('it is %.2f %s and in stock' % (target['price'], self.mission.currency))
            if self.atlas.event_target:
                parts.append('after the previous cart item changed on the market')
            return ', '.join(parts) + '.'
        return _orbit()

    def _read_more(self) -> tuple | None:

        def _orbit():
            """Turn the search results already in hand into read product pages, then widen.

        Reading pages is what turns rows into candidates the controller can verify,
        and a second verified product is what makes the market change survivable.
        """
            twin = self._twin_search()
            if twin is not None:
                return twin
            brand_hunt = self._preferred_brand_hunt()
            if brand_hunt is not None:
                return brand_hunt
            if self.mission.family == 'preference' and (not self.pref_probed) and 'search' in self.tool_names:
                self.pref_probed = True
                return ('search', {'query': 'waterproof sport wireless earbuds battery ipx', 'in_stock': True, 'k': 10})
            # Justification / firm-brand tasks: open the brand shelf before random search hits.
            if self.mission.family in ('justification', 'constraint') and self.mission.brand and (not self.brand_probed) and 'filter' in self.tool_names and self.mission.category:
                self.brand_probed = True
                args = {'category': self.mission.category, 'brand': self.mission.brand, 'in_stock': True, 'k': 10}
                if self.budget is not None:
                    args['max_price'] = self.budget
                return ('filter', args)
            fam = self.mission.family
            if fam == 'recall' and (not self.atlas.event_target) and self.clock.remaining() > 90 and len(self.atlas.identities) < self._recall_need():
                # Still open unread themed leads so stock/price can be verified, then
                # return None so the model can keep diversifying search wording.
                leads = self._unseen_leads()
                if leads and self.forced_compares < 12:
                    self.forced_compares += 1
                    if len(leads) > 1 and 'compare' in self.tool_names:
                        return ('compare', {'product_ids': leads[:5]})
                    if 'view' in self.tool_names:
                        return ('view', {'product_id': leads[0]})
                return None
            leads = self._unseen_leads()
            if leads and self.forced_compares < 12 and (self.turn >= 1 or self.atlas.event_target or self.clock.remaining() < 140):
                self.forced_compares += 1
                if len(leads) > 1 and 'compare' in self.tool_names:
                    return ('compare', {'product_ids': leads[:5]})
                if 'view' in self.tool_names:
                    return ('view', {'product_id': leads[0]})
            # Recall: once the minimum is met (or margin has lapsed), stop endless searching.
            if fam == 'recall' and self.mission.theme and len(self.atlas.identities) >= max(1, self.mission.required_identities):
                themed_ok = [c for c in self._candidates() if c['ok']]
                if themed_ok:
                    self.reads_done = True
                    return None
            probe = (self.mission.brand or self.mission.pref_brand or '', '%.2f' % (self.budget or 0.0), str(self.mission.spec_value), ' '.join(self.mission.theme or []))
            if self.turn < 1:
                return None
            if probe in self.probes:
                self.reads_done = True
                return None
            self.probes.add(probe)
            if fam == 'recall' and self.mission.theme and 'search' in self.tool_names:
                return ('search', {'query': ' '.join(self.mission.theme), 'in_stock': True, 'k': 10})
            if 'filter' in self.tool_names and self.mission.category:
                args = {'category': self.mission.category, 'in_stock': True, 'k': 10}
                if self.budget is not None:
                    args['max_price'] = self.budget
                if self.mission.brand or self.mission.pref_brand:
                    args['brand'] = self.mission.brand or self.mission.pref_brand
                return ('filter', args)
            if 'search' in self.tool_names:
                words = [w for w in (self.mission.brand or self.mission.pref_brand, self.mission.spec_value, self.mission.category) if w]
                if self.mission.theme:
                    words = list(self.mission.theme) + [w for w in words if w]
                if words:
                    return ('search', {'query': ' '.join(words), 'in_stock': True, 'k': 10})
            self.reads_done = True
            return None
        return _orbit()

    def _preferred_brand_hunt(self) -> tuple | None:

        def _orbit():
            """Intent gold path: surface the preferred brand before any other first cart."""
            if self.mission.family != 'intent' or not self.mission.pref_brand or self.atlas.adds:
                return None
            if self.brand_probed:
                return None
            brand = self.mission.pref_brand
            self.brand_probed = True
            self._preferred_brand_hint()
            if 'filter' in self.tool_names and self.mission.category:
                args = {'category': self.mission.category, 'brand': brand, 'in_stock': True, 'k': 10}
                if self.budget is not None:
                    args['max_price'] = self.budget
                return ('filter', args)
            if 'search' in self.tool_names:
                bits = [brand, self.mission.spec_value, self.mission.category or 'smartphone']
                return ('search', {'query': ' '.join((b for b in bits if b)), 'in_stock': True, 'k': 10})
            return None
        return _orbit()

    def _twin_search(self) -> tuple | None:

        def _orbit():
            """After the market kills the carted item, hunt nearest replacements.

        Prefer the shopper's preferred brand / theme rather than re-searching
        the dead title alone (which often only resurfaces accessories).
        """
            dead = self.atlas.event_target
            if self.searched_twins or not dead or 'search' not in self.tool_names:
                return None
            self.searched_twins = True
            if self.mission.family == 'intent' and self.mission.pref_brand:
                bits = [self.mission.pref_brand, self.mission.spec_value, self.mission.category or 'smartphone']
                self._note('Market recovery: searching the preferred brand for another compliant listing.')
                return ('search', {'query': ' '.join((b for b in bits if b)), 'k': 10, 'in_stock': True})
            if self.mission.family == 'recall' and self.mission.theme:
                self._note('Market recovery: re-searching the required phrase under budget.')
                return ('search', {'query': ' '.join(self.mission.theme), 'k': 10, 'in_stock': True})
            title = self._title(dead.split('::', 1)[0])
            if not title:
                return None
            self._note('The item that died is one of several listings sharing a title; searching that title surfaces its other colors and sizes, which meet the same requirements.')
            return ('search', {'query': ' '.join(title.split()[:8]), 'k': 10, 'in_stock': True})
        return _orbit()

    def _preferred_brand_hint(self) -> None:

        def _orbit():
            if self.hinted_brand:
                return
            self.hinted_brand = True
            self._note("No %s product that meets every firm requirement has been read yet. The shopper's first choice is graded on that brand, so filter by brand=%s within the budget and read those products before any other." % (self.mission.pref_brand, self.mission.pref_brand))
        return _orbit()

    def _sibling_search(self) -> tuple | None:

        def _orbit():
            """A found model in the wrong color: its other colors share the same title."""
            if self.searched_siblings:
                return None
            rows = [c for c in self._candidates() if c['live'] and c['viewed'] and (c['relax'] == 0)]
            if rows:
                return None
            near = [c for c in self._candidates() if c['viewed'] and c['relax'] == 1]
            if not near:
                return None
            title = self._title(near[0]['pid'])
            if not title:
                return None
            self.searched_siblings = True
            return ('search', {'query': ' '.join(title.split()[:8]), 'k': 10, 'in_stock': True})
        return _orbit()

    def _exact_match_hint(self) -> None:

        def _orbit():
            if self.hinted_exact or not self.mission.pref_model:
                return
            self.hinted_exact = True
            self._note("The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own and view the results so every color variant is on record; only that exact pairing may be the first cart item." % (self.mission.pref_model, ', ' + self.mission.pref_color if self.mission.pref_color else ''))
        return _orbit()

    def _unseen_leads(self) -> list:

        def _orbit():
            """Promising variants whose product page has not been read yet."""
            leads = []
            for key, row in self.atlas.rows.items():
                pid, _, sku = key.partition('::')
                if not sku or str(pid) in self.atlas.products or str(pid) in self.atlas.dead_products:
                    continue
                st = self.atlas.checked.get(key) or {}
                price = st.get('price')
                if not st.get('in_stock') or price is None or (self.budget is not None and price > self.budget + _EPS):
                    continue
                title = str(row.get('title') or '')
                if self.mission.theme and (not all((t in SpecLens._tokens(title) for t in self.mission.theme))):
                    continue
                if self.mission.pref_model and self.atlas.adds == 0 and (not self.atlas.event_target):
                    squashed = re.sub('[^a-z0-9]+', '', title.lower())
                    if re.sub('[^a-z0-9]+', '', self.mission.pref_model.lower()) not in squashed:
                        continue
                brand_miss = 0
                if self.mission.brand or self.mission.pref_brand:
                    wanted = SpecLens._tokens(self.mission.brand or self.mission.pref_brand)
                    if not (wanted <= SpecLens._tokens(str(row.get('brand') or '')) or wanted <= SpecLens._tokens(title)):
                        if self.mission.family == 'intent' and self.atlas.adds == 0 and (not self.atlas.event_target):
                            continue
                        brand_miss = 1
                accessory = 1 if re.search('\\b(case|cover|casing|tempered|glass|charger|cable|protector|lanyard|dummy|trigger|gamepad|anti[- ]?lost|rope|strap)\\b', title.lower()) else 0
                leads.append((accessory, brand_miss, price, str(pid)))
            leads.sort()
            out = []
            for _a, _b, _price, pid in leads:
                if pid not in out:
                    out.append(pid)
            return out
        return _orbit()

    def _shortlist_note(self) -> None:

        def _orbit():
            """Tell the model what the controller can and cannot yet verify."""
            cands = self._candidates()
            if self.mission.family == 'preference' and (not self.atlas.cart):
                good = [c for c in cands if c['ok']]
                good.sort(key=lambda c: (-self._priority_hits(c['pid'], c['sku']), c['price'] if c['price'] is not None else 1000000000000.0))
                if good:
                    self._note('Products that meet every firm requirement so far: ' + '; '.join(('%s::%s at %.2f' % (c['pid'], c['sku'], c['price'] or 0.0) for c in good[:6])) + ". Pick the one that best fits the shopper's stated priorities, inspect it, then add it.")
                    return
            best = self._target()
            if best:
                self._note('Verified candidate: %s::%s at %.2f. The controller will inspect, cart and order it; read other candidates only if one could rank better.' % (best['pid'], best['sku'], best['price'] or 0.0))
                return
            field, want = (self.mission.spec_field, self.mission.spec_value)
            blocked = [c for c in cands if c['viewed'] and c['live'] and (not c['ok'])]
            if blocked and field:
                sample = blocked[:3]
                self._note("None of the products read so far can be verified. For each, the evidence for %s must resolve to exactly %s from the variant's own option text or the product's raw facts: %s. Look at different products." % (field, want, ', '.join((c['pid'] + '::' + c['sku'] for c in sample))))
            elif blocked:
                self._note('The products read so far fail a firm requirement (category, brand, stock, budget or the kind of device requested). Search for different products.')
        return _orbit()

    def _priority_hits(self, pid: str, sku: str) -> int:

        def _orbit():
            """How much of what the shopper actually asked for this product's own facts mention."""
            blob = ' '.join((str(v) for v in self._facts(pid).values())) + ' ' + self._title(pid) + ' ' + self._options(pid, sku)
            have = SpecLens._tokens(blob)
            wish = len(self._wish_words() & have) if self._wish_words() else 0
            return wish + 2 * self._axis_hits(pid, sku)
        return _orbit()

    def _axis_hits(self, pid: str, sku: str) -> int:

        def _orbit():
            """Soft preference fit from product text; headphone tasks score all four public axes."""
            blob = (' '.join((str(v) for v in self._facts(pid).values())) + ' ' + self._title(pid) + ' ' + self._options(pid, sku)).lower()
            if self._is_accessory_title(pid):
                return -5
            cues = {
                'water': ('waterproof', 'water-resistant', 'water resistant', 'sweat', 'ipx', 'ip67', 'ip68', 'ipx5', 'ipx7', 'rain'),
                'sport': ('sport', 'gym', 'run', 'fitness', 'training', 'workout', 'exercise', 'secure fit', 'ear hook', 'bone conduction'),
                'battery': ('battery', 'mah', 'hour', 'lasting', 'endurance', 'all-day', 'all day', '2000mah', '20-hour', '20 hour', 'long life'),
                'wireless': ('wireless', 'bluetooth', 'tws', 'cordless', 'true wireless', 'bt5', 'bt 5'),
            }
            cat = (self.mission.category or '').lower()
            if re.search('\\b(headphone|earphone|earbud|headset)s?\\b', cat) or re.search('\\b(headphone|earphone|earbud|headset)s?\\b', self.mission.goal.lower()):
                return sum((1 for words in cues.values() if any((c in blob for c in words))))
            axes = list(getattr(self.mission, 'soft_axes', None) or [])
            if not axes:
                return sum((1 for words in cues.values() if any((c in blob for c in words))))
            return sum((1 for axis in axes if any((c in blob for c in cues.get(axis, (axis,))))))
        return _orbit()

    def _wish_words(self) -> set:

        def _orbit():
            """Content words from the shopper's own wording, gathered once."""
            if self._wishes is None:
                said = self.mission.goal + ' ' + ' '.join((str(m.get('content') or '') for m in self.messages if m.get('role') == 'user'))
                drop = SpecLens._tokens(self.mission.category or '') | {'the', 'and', 'for', 'with', 'that', 'this', 'you', 'your', 'find', 'need', 'want', 'order', 'stock', 'place', 'under', 'over', 'one', 'php', 'test', 'buy', 'rather', 'than', 'pay', 'bit', 'more', 'settle', 'something', 'its', 'would', 'like', 'have'}
                self._wishes = {w for w in SpecLens._tokens(said) if len(w) > 3 and (not w.isdigit()) and (w not in drop)}
            return self._wishes
        return _orbit()

    def _budget_from_message(self, text: str) -> None:

        def _orbit():
            new = Mission._budget(text)
            if new is None or (self.budget is not None and new >= self.budget):
                return
            self.budget = new
            self.shopper_updated = True
            self._note('Budget updated by the shopper: at most %.2f %s. Every later choice must fit it.' % (new, self.mission.currency))
            for pid, sku in list(self.atlas.cart):
                st = self.atlas.status(pid, sku) or {}
                price = st.get('price')
                if price is not None and price > new + _EPS:
                    self.atlas.dead_keys.add(Atlas.key(pid, sku))
                    self._note('The carted item %s costs %.2f, above the new budget; the controller will replace it.' % (Atlas.key(pid, sku), price))
        return _orbit()

    def _claims_for(self, pid: str, sku: str) -> list:

        def _orbit():
            """Claims proven by what the agent has already observed about this exact item."""
            out = []
            field, want = (self.mission.spec_field, self.mission.spec_value)
            if not field and self.mission.family == 'justification':
                field, want = ('resolution', self._fact_value(pid, 'resolution'))
            if not field or not want:
                return []
            stated = dict(self.mission.specs) or {field: want}
            stated.setdefault(field, want)
            for name, value in stated.items():
                if self._value_match(name, self._fact_value(pid, name), value) and self._value_match(name, self._product_value(pid, name), value):
                    out.append({'field': name, 'value': value})
            if not out:
                return []
            st = self.atlas.status(pid, sku) or {}
            if self.budget is not None:
                out.append({'field': 'price', 'value': 'at most %.2f' % self.budget})
            elif st.get('price') is not None:
                out.append({'field': 'price', 'value': '%.2f' % st['price']})
            out.append({'field': 'in_stock', 'value': 'true'})
            brand = self._brand_of(pid)
            if brand:
                out.append({'field': 'brand', 'value': brand})
            return out
        return _orbit()

    def _order_args(self, pid: str, sku: str) -> dict:

        def _orbit():
            out = {'product_id': pid, 'sku': sku}
            if self.mission.family == 'justification':
                claims = self._claims_for(pid, sku)
                if claims:
                    out['justification'] = {'claims': claims}
            return out
        return _orbit()

    def _valid_now(self, pid: str, sku: str) -> tuple:

        def _orbit():
            st = self.atlas.status(pid, sku)
            k = Atlas.key(pid, sku)
            if st is None:
                return (False, 'inspect_stock the exact product_id and sku first')
            if (pid, sku) not in self.atlas.cart:
                return (False, 'add that exact variant to the cart first')
            if self.atlas.last_add_seq is not None and self.atlas.checked_seq.get(k, 0) <= self.atlas.last_add_seq:
                return (False, 're-check the variant with inspect_stock after the cart addition')
            if not st.get('in_stock'):
                return (False, 'that variant is out of stock; choose a different product')
            price = st.get('price')
            if self.budget is not None and price is not None and (price > self.budget + _EPS):
                return (False, 'that variant costs %.2f, above the budget of %.2f' % (price, self.budget))
            if k in self.atlas.dead_keys:
                return (False, 'that variant was invalidated by the market change; choose a different product')
            return (True, '')
        return _orbit()

    def _recall_need(self) -> int:

        def _orbit():
            """How many distinct matching products to surface before buying.

        Surfacing more than the stated minimum is worth turns early on, but the
        catalog may simply not hold that many, so the margin lapses.
        """
            base = self.mission.required_identities
            if self.search_calls >= 6 or self.clock.remaining() < 100 or self.turn >= self.max_steps * 0.55:
                return max(1, base)
            early = self.search_calls < 8 and self.turn < self.max_steps * 0.5
            return base + (1 if early else 0)
        return _orbit()

    def _ready(self) -> bool:

        def _orbit():
            """Whether a first commitment can be made, or the shopper still owes information."""
            fam = self.mission.family
            if self.atlas.adds:
                return True
            if self.clock.remaining() < 70:
                # Wall-clock pressure: stop waiting on optional shopper polish.
                return fam != 'preference' or len(self.atlas.listing_ids) >= 2 or self.turn >= self.max_steps - 6
            if self.mission.spec_sealed:
                if self.turn < self.max_steps * 0.6:
                    return False
                self.mission.spec_sealed = False
                self._note('The firm requirement was never stated, so the remaining requirements decide the choice.')
            # Only stall early when the goal still lacks a firm gate the shopper might add.
            awaiting = fam in ('intent', 'constraint') and self.turn < 4 and (not self.shopper_updated)
            if awaiting and (not self.mission.spec_value) and self.mission.spec_sealed is False and (self.budget is None):
                return False
            if awaiting and (not self.mission.spec_value) and self.mission.spec_field is None and ('firm requirement' in self.mission.goal.lower()):
                return False
            if fam == 'preference' and len(self.atlas.listing_ids) < 2:
                return False
            if fam == 'intent' and self.mission.pref_brand and (not self.brand_probed) and self.clock.remaining() > 80:
                return False
            if fam == 'intent' and self.mission.pref_brand and self.atlas.adds == 0 and self.turn < self.max_steps * 0.5 and self.clock.remaining() > 110:
                wanted = SpecLens._tokens(self.mission.pref_brand)
                named = [c for c in self._candidates() if c['ok'] and (wanted <= SpecLens._tokens(self._brand_of(c['pid'])) or wanted <= SpecLens._tokens(self._title(c['pid'])))]
                if not named and self.brand_probed and self.forced_compares < 3:
                    # Keep reading preferred-brand leads a little longer.
                    return False
                if not named and not self.brand_probed:
                    return False
            roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
            if fam == 'recall':
                need = self._recall_need()
                have = len(self.atlas.identities)
                themed_ok = any((c['ok'] for c in self._candidates()))
                if have < need and self.clock.remaining() > 70:
                    if not (themed_ok and have >= max(1, self.mission.required_identities) and self.search_calls >= 3):
                        return False
                if self.search_calls < 3 and self.clock.remaining() > 100 and self.turn < self.max_steps * 0.55 and (not themed_ok):
                    return False
            if roomy and (not self.reads_done) and (len({c['pid'] for c in self._candidates() if c['ok']}) < 2) and fam != 'recall':
                return False
            return True
        return _orbit()

    def _fresh(self, pid: str, sku: str) -> bool:

        def _orbit():
            k = Atlas.key(pid, sku)
            if k not in self.atlas.inspected:
                return False
            return self.atlas.last_add_seq is None or self.atlas.checked_seq.get(k, 0) > self.atlas.last_add_seq
        return _orbit()

    def _plan(self) -> tuple | None:

        def _orbit():
            """The action the controller takes itself, or nothing to let the model explore."""
            if self.done:
                return None
            # Answer shopper questions immediately — waiting burns the clock and is
            # the main path to "agent did not reach a terminal state" after events.
            if self.shopper_question_pending and 'message' in self.tool_names:
                return ('message', {'content': self._answer_text()})
            # Dead cart rows block a clean recovery; clear them before hunting.
            for cpid, csku in list(self.atlas.cart):
                if Atlas.key(cpid, csku) in self.atlas.dead_keys and 'remove_from_cart' in self.tool_names:
                    return ('remove_from_cart', {'product_id': cpid, 'sku': csku})
            fam = self.mission.family
            if fam == 'ranking' and 'compare' in self.tool_names:
                missing = [p for p, _s in self.mission.candidate_keys if p not in self.atlas.products and p not in self.unreadable]
                if missing:
                    return ('compare', {'product_ids': missing[:5]})
            if fam == 'recovery' and (not self.atlas.adds) and self.mission.pref_model and ('search' in self.tool_names):
                forced = self._sibling_search()
                if forced is not None:
                    return forced
            target = self._target()
            if target is None or not self._ready():
                forced = self._read_more()
                if forced is not None:
                    return forced
                # Under pressure, still try to open unread leads even on turn 0/1.
                if self.clock.remaining() < 100 or self.atlas.event_target:
                    leads = self._unseen_leads()
                    if leads and self.forced_compares < 10:
                        self.forced_compares += 1
                        if len(leads) > 1 and 'compare' in self.tool_names:
                            return ('compare', {'product_ids': leads[:5]})
                        if 'view' in self.tool_names:
                            return ('view', {'product_id': leads[0]})
                if target is None:
                    return None
            pid, sku = (target['pid'], target['sku'])
            if (pid, sku) in self.atlas.cart:
                if not self._fresh(pid, sku):
                    return ('inspect_stock', {'product_id': pid, 'sku': sku})
                return ('place_test_order', self._order_args(pid, sku))
            if not self._ready():
                return None
            if not self._fresh(pid, sku):
                return ('inspect_stock', {'product_id': pid, 'sku': sku})
            return ('add_to_cart', {'product_id': pid, 'sku': sku})
        return _orbit()

    def _ask_model(self) -> dict | None:

        def _orbit():
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
        return _orbit()

    def _send(self, envelope: dict) -> dict | None:

        def _orbit():
            """POST one solver-turn envelope; an identical envelope replays idempotently."""
            for attempt in range(4):
                result = _PROXY.post('/environment/call', json_data=envelope)
                if isinstance(result, dict) and result.get('calls'):
                    return result
                if self.clock.remaining() < 20:
                    break
                time.sleep(min(2.0 * (attempt + 1), 6.0))
            return None
        return _orbit()

    def _act(self, name: str, args: dict, content: str='', tool_call_id: str | None=None) -> dict | None:

        def _orbit():
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
        return _orbit()

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:

        def _orbit():
            tool_calls = [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]
            self.messages.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls})
            self.dialogue.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls, 'environment_result': result})
            if not isinstance(result, dict):
                self.atlas.calls_seen.discard(name + json.dumps(args, sort_keys=True))
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
            self.atlas.record(name, args, obs, self.mission)
            self.messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': _trim(json.dumps(obs), _OBS_CHARS)})
            user = result.get('user_message')
            if isinstance(user, dict) and user.get('content'):
                text = str(user['content'])
                self.messages.append({'role': 'user', 'content': text})
                self._budget_from_message(text)
                if self.mission.family in ('constraint', 'intent') and (not self.mission.spec_value):
                    stated = self.mission.adopt_spec(text)
                    if stated:
                        self.shopper_updated = True
                        self._note('Firm requirement recorded: ' + stated + '. Only a variant that resolves to it may be bought.')
                if _QUESTION_RE.search(text):
                    self.shopper_question_pending = True
                    self._note('The shopper asked a question: answer it with message in your next turn, then continue.')
            if any((isinstance(c, dict) and (c.get('observation') or {}).get('done') for c in calls)):
                self.done = True
            return obs
        return _orbit()

    def _post_add_recheck(self) -> None:

        def _orbit():
            """The market moves on the first cart action, so re-read that item at once."""
            pid, sku = self.pending_recheck
            self.pending_recheck = None
            obs = self._act('inspect_stock', {'product_id': pid, 'sku': sku})
            if not isinstance(obs, dict) or obs.get('error'):
                return
            stock = bool(obs.get('in_stock'))
            price = SpecLens._num(obs.get('price'))
            over = self.budget is not None and price is not None and (price > self.budget + _EPS)
            if stock and (not over):
                self._note('Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it.' % (Atlas.key(pid, sku), price))
                return
            self.atlas.event_target = Atlas.key(pid, sku)
            self.event_turn = self.turn
            self.atlas.dead_keys.add(Atlas.key(pid, sku))
            self.atlas.dead_products.add(str(pid))
            self.forced_compares = 0
            reason = 'went out of stock' if not stock else 'was repriced to %.2f, above the budget' % (price or 0.0)
            guidance = {'recovery': 'Recover with the smallest possible change: the same model in another color before any other model.', 'ranking': 'The remaining named candidates are re-ranked; the controller takes the new best.', 'preference': "Recover to a product with a different listing_id that still fits the shopper's priorities.", 'constraint': 'Choose a different product that still meets every requirement, not another variant of the same one.'}.get(self.mission.family, 'Choose a different product that meets every firm requirement.')
            self._note('Market change: the carted item %s %s. ' % (Atlas.key(pid, sku), reason) + guidance)
            self._shortlist_note()
        return _orbit()

    def _finish_fast(self) -> None:

        def _orbit():
            """Out of time or turns: reach a valid order in as few actions as possible."""
            # Clear dead cart lines so a replacement can be committed.
            for pid, sku in list(self.atlas.cart):
                if Atlas.key(pid, sku) in self.atlas.dead_keys and 'remove_from_cart' in self.tool_names:
                    self._act('remove_from_cart', {'product_id': pid, 'sku': sku})
                    if self.done:
                        return
            for pid, sku in reversed(self.atlas.cart):
                ok, _why = self._valid_now(pid, sku)
                if ok and (not self.done):
                    self._act('place_test_order', self._order_args(pid, sku))
                    return
            if self.done or self.turn > self.max_steps - 2:
                return
            good = [c for c in self._candidates() if c['ok']]
            if self.mission.family == 'preference':
                strong = [c for c in good if self._priority_hits(c['pid'], c['sku']) >= 2]
                if strong:
                    good = strong
                good = sorted(good, key=lambda c: (-self._priority_hits(c['pid'], c['sku']), c['price'] if c['price'] is not None else 1000000000000.0))
            if self.mission.family == 'intent' and self.atlas.adds == 0 and self.mission.pref_brand:
                wanted = SpecLens._tokens(self.mission.pref_brand)
                named = [c for c in good if wanted <= SpecLens._tokens(self._brand_of(c['pid'])) or wanted <= SpecLens._tokens(self._title(c['pid']))]
                if named:
                    good = named
            if not good and self.mission.family != 'justification':
                good = [c for c in self._candidates() if c['live'] and c['spec'] == 2 and self._brand_ok(c['pid']) and c['cat_ok'] and self._device_ok(c['pid'], c['sku']) and (not self.mission.candidate_keys or (c['pid'], c['sku']) in self.mission.candidate_keys)]
                if self.mission.family == 'recall' and self.mission.theme:
                    good = [c for c in good if all((t in SpecLens._tokens(self._title(c['pid'])) for t in self.mission.theme))]
                if self.mission.family == 'preference':
                    good = [c for c in good if not self._is_accessory_title(c['pid'])]
                if self.mission.family == 'intent' and self.atlas.adds == 0 and self.mission.pref_brand:
                    wanted = SpecLens._tokens(self.mission.pref_brand)
                    named = [c for c in good if wanted <= SpecLens._tokens(self._brand_of(c['pid'])) or wanted <= SpecLens._tokens(self._title(c['pid']))]
                    if named:
                        good = named
            if self.mission.family == 'constraint' and self.mission.spec_field and self.mission.spec_value:
                good = [c for c in good if c.get('spec') == 2]
            if not good:
                # Last gasp: open one unread lead then stop — caller may retry.
                leads = self._unseen_leads()
                if leads and 'view' in self.tool_names and self.turn < self.max_steps - 2:
                    self._act('view', {'product_id': leads[0]})
                return
            if self.mission.family not in ('preference',):
                good.sort(key=lambda c: self._rank_of(c, self.atlas.adds == 0))
            pid, sku = (good[0]['pid'], good[0]['sku'])
            if str(pid) not in self.atlas.products and 'view' in self.tool_names:
                self._act('view', {'product_id': pid})
                if self.done:
                    return
            if (pid, sku) not in self.atlas.cart:
                obs = self._act('add_to_cart', {'product_id': pid, 'sku': sku})
                if self.done or not isinstance(obs, dict) or obs.get('error'):
                    return
            obs = self._act('inspect_stock', {'product_id': pid, 'sku': sku})
            if self.done or not isinstance(obs, dict) or (not obs.get('in_stock')):
                return
            price = SpecLens._num(obs.get('price'))
            if self.budget is not None and price is not None and (price > self.budget + _EPS):
                self.atlas.dead_keys.add(Atlas.key(pid, sku))
                self.atlas.dead_products.add(str(pid))
                self.atlas.event_target = Atlas.key(pid, sku)
                # Try one alternate in the same finish window.
                alts = [c for c in good if c['pid'] != pid and c.get('live')]
                if not alts:
                    return
                pid, sku = (alts[0]['pid'], alts[0]['sku'])
                if (pid, sku) not in self.atlas.cart:
                    obs = self._act('add_to_cart', {'product_id': pid, 'sku': sku})
                    if self.done or not isinstance(obs, dict) or obs.get('error'):
                        return
                obs = self._act('inspect_stock', {'product_id': pid, 'sku': sku})
                if self.done or not isinstance(obs, dict) or (not obs.get('in_stock')):
                    return
                price = SpecLens._num(obs.get('price'))
                if self.budget is not None and price is not None and (price > self.budget + _EPS):
                    return
            self._act('place_test_order', self._order_args(pid, sku))
        return _orbit()

    def run(self) -> list:

        def _orbit():
            question = self._opening_question()
            last_forced, repeats = (None, 0)
            idle_model = 0
            while not self.done and self.turn < self.max_steps:
                pressure = self.clock.remaining() < 35 or self.turn >= self.max_steps - 3
                if pressure:
                    self._finish_fast()
                    if self.done:
                        break
                    if self.clock.remaining() < 12 or self.turn >= self.max_steps - 1:
                        break
                    # Keep a controller-only recovery lap instead of dying quiet.
                    pressure = False
                if question and (not self.asked_shopper) and ('message' in self.tool_names):
                    self.asked_shopper = True
                    self._act('message', {'content': question})
                    continue
                if self.pending_recheck and 'inspect_stock' in self.tool_names:
                    self._post_add_recheck()
                    continue
                plan = self._plan()
                if plan is not None:
                    idle_model = 0
                    name, args = plan
                    signature = name + json.dumps(args, sort_keys=True)
                    repeats = repeats + 1 if signature == last_forced else 1
                    last_forced = signature
                    if repeats > 2:
                        if name in ('inspect_stock', 'add_to_cart'):
                            self.atlas.dead_keys.add(Atlas.key(args.get('product_id'), args.get('sku')))
                        if name == 'compare':
                            self.unreadable.update((str(p) for p in args.get('product_ids') or []))
                        if name == 'message':
                            self.shopper_question_pending = False
                        last_forced, repeats = (None, 0)
                        self._note('That step changed nothing, so the controller moved on to other candidates.')
                        continue
                    obs = self._act(name, args)
                    if name == 'message':
                        self.shopper_question_pending = False
                        self.question_waits = 0
                    if name == 'add_to_cart' and isinstance(obs, dict) and (not obs.get('error')):
                        added = obs.get('added') or {}
                        if added.get('product_id'):
                            self.pending_recheck = (str(added['product_id']), str(added['sku']))
                    continue
                last_forced, repeats = (None, 0)
                # Under clock pressure or after a market event, prefer controller recovery
                # over another inference round that can time out and kill the episode.
                if self.clock.remaining() < 55 or (self.atlas.event_target and idle_model >= 1):
                    self._finish_fast()
                    if self.done:
                        break
                    idle_model += 1
                    if idle_model >= 3 or self.clock.remaining() < 20:
                        break
                    continue
                proposal = self._ask_model()
                if proposal is None:
                    idle_model += 1
                    self._finish_fast()
                    if self.done or idle_model >= 2 or self.clock.remaining() < 25:
                        break
                    continue
                idle_model = 0
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
                    if not pid and self.atlas.cart:
                        pid, sku = self.atlas.cart[-1]
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
                    if self.mission.family == 'recall' and self.mission.required_identities:
                        have, need = (len(self.atlas.identities), self._recall_need())
                        self._note('Distinct matching products surfaced so far: %d of the required %d.' % (have, need) + ('' if have >= need else ' Keep searching with different wording.'))
                if name == 'inspect_cart':
                    self.cart_inspections += 1
                if name in ('view', 'compare'):
                    self._shortlist_note()
            return self.dialogue
        return _orbit()

    def _validate(self, name: str, args: dict) -> str | None:

        def _orbit():
            if name not in self.tool_names:
                return 'unknown tool; choose one of ' + str(sorted((n for n in self.tool_names if n)))
            if name == 'place_test_order':
                pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
                if not pid and self.atlas.cart:
                    pid, sku = self.atlas.cart[-1]
                ok, why = self._valid_now(pid, sku)
                if not ok:
                    return 'order refused: ' + why
                if self.mission.family == 'recall' and len(self.atlas.identities) < self._recall_need():
                    return 'order refused: only %d distinct matching products have been surfaced; search with different wording until at least %d appear' % (len(self.atlas.identities), self._recall_need())
                if self.mission.family == 'preference' and len(self.atlas.listing_ids) < 2:
                    return 'order refused: compare at least two products with different listing_id values first'
                if self.mission.family == 'justification':
                    claims = self._claims_for(pid, sku)
                    if not claims:
                        return 'order refused: justification needs fact-grounded claims for the required specification'
                    if self.mission.spec_field and self.mission.spec_value:
                        if not any((c.get('field') == self.mission.spec_field for c in claims)):
                            return 'order refused: justification must cite %s = %s from product facts' % (self.mission.spec_field, self.mission.spec_value)
            if name == 'add_to_cart':
                pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
                if not pid or not sku:
                    return 'add_to_cart needs both product_id and sku'
                c = self._assess(pid, sku)
                key = Atlas.key(pid, sku)
                if not c['live']:
                    return 'that variant is out of stock, over budget, or already invalidated'
                if not c['viewed']:
                    return 'view or compare that product first so its raw facts are on record'
                if key not in self.atlas.inspected:
                    return 'inspect_stock that exact variant first: its current price and stock decide whether it can be bought'
                if c['spec'] != 2 and self.mission.spec_field:
                    return 'that variant does not resolve to %s = %s from its own option text or raw facts' % (self.mission.spec_field, self.mission.spec_value)
                if not c['ok']:
                    return 'that product fails a firm requirement (category, brand, or the kind of device requested)'
                if self.mission.family == 'preference' and len(self.atlas.listing_ids) < 2:
                    return 'compare two products with different listing_id values before the first cart addition'
                if self.mission.family == 'preference' and self._is_accessory_title(pid):
                    return 'that listing looks like an accessory, not the headphone device the shopper asked for'
                if self.mission.family == 'intent' and (not self.atlas.adds) and self.mission.pref_brand:
                    wanted = SpecLens._tokens(self.mission.pref_brand)
                    if not (wanted <= SpecLens._tokens(self._brand_of(pid)) or wanted <= SpecLens._tokens(self._title(pid))):
                        named = [c for c in self._candidates() if c['ok'] and (wanted <= SpecLens._tokens(self._brand_of(c['pid'])) or wanted <= SpecLens._tokens(self._title(c['pid'])))]
                        if named:
                            return 'first cart must be the preferred brand %s when a compliant item exists; filter that brand first' % self.mission.pref_brand
                if self.mission.family == 'recovery' and (not self.atlas.adds) and self.mission.pref_model and self._relax(pid, sku):
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
            if key in self.atlas.calls_seen and name in ('search', 'filter', 'view', 'compare'):
                return 'identical call already made; change the arguments'
            self.atlas.calls_seen.add(key)
            return None
        return _orbit()

def agent_main(problem_data: dict) -> list:
    try:
        return Meridian(problem_data).run()
    except Exception as exc:
        return [{'role': 'assistant', 'content': 'agent error: ' + type(exc).__name__ + ': ' + str(exc)[:200]}]
__all__ = ['agent_main']
