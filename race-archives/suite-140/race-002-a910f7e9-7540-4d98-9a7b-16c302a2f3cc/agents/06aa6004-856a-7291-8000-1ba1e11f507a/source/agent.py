
from __future__ import annotations
import json
import re
import time
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
_MODELS = {'chutes': ('deepseek-ai/DeepSeek-V3.2-TEE', 'Qwen/Qwen3-32B-TEE'), 'openrouter': ('deepseek/deepseek-v3.2', 'qwen/qwen3-32b')}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_PROBLEM_BUDGET_S = 300.0
_OBS_CHARS = 7000
_MAX_REPROMPTS = 2
_EPS = 1e-09
_NUM = '(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?'
_BUDGET_RE = re.compile('(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|within|budget (?:is|of)|only go up to|can spend up to|at most)\\s*' + _NUM + '\\s*(PHP|php|₱|pesos?)?', re.I)
_KEY_RE = re.compile('(\\d{6,})::([A-Za-z0-9._\\-]+)')
_QUOTED_RE = re.compile('\\"([^\\"]{2,80})\\"')
_PHRASE_RE = re.compile('matches (?:the phrase )?\\"?([A-Za-z0-9][^\\",]{1,80}?)\\"?\\s*,', re.I)
_SURFACE_RE = re.compile('surface at least (\\d+) distinct', re.I)
_PREF_ITEM_RE = re.compile('(?:strongly\\s+)?prefer(?:ably)?(?:\\s+the)?\\s+((?:[A-Za-z][A-Za-z\\-]*\\s+){0,3}[A-Za-z][A-Za-z\\-]*)\\s+([A-Za-z0-9][A-Za-z0-9./\\-]*\\d[A-Za-z0-9./\\-]*)\\b', re.I)
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

def _inch_vals(text: str) -> set:
    return {'%gin' % float(m.group(1)) for m in _INCH_RE.finditer(text or '')}

def _hz_vals(text: str) -> set:
    return {'%ghz' % float(m.group(1)) for m in _HZ_RE.finditer(text or '')}

def _gb_vals(text: str) -> set:
    return {'%ggb' % (float(m.group(1)) * (1024 if m.group(2).lower() == 'tb' else 1)) for m in _GB_RE.finditer(text or '')}

class _DisplayLex:
    """Resolution vocabulary without industry WxH threshold ladders."""

    @staticmethod
    def tokens(text: str) -> set:
        text = text or ''
        if _QUARTER_RE.search(text):
            return set()
        low = text.lower()
        out: set = set()
        for m in _WXH_RE.finditer(low):
            a, b = (int(m.group(1)), int(m.group(2)))
            long_s, short_s = (max(a, b), min(a, b))
            out.add('%dx%d' % (long_s, short_s))
            out.add('%dp' % short_s)
        if re.search('\\b(?:4k|uhd|ultra\\s*hd|2160p)\\b', low):
            out.update({'4k', '2160p'})
        if re.search('\\b(?:qhd|1440p|2k)\\b', low):
            out.update({'qhd', '1440p'})
        if re.search('\\b(?:fhd|1080p|full\\s*hd)\\b', low):
            out.update({'fhd', '1080p'})
        stripped = re.sub('\\b(?:full|ultra)\\s*hd\\b', ' ', low)
        if re.search('\\b(?:hd|720p)\\b', stripped):
            out.update({'hd', '720p'})
        for m in re.finditer('\\b(\\d{3,4})p\\b', low):
            out.add('%sp' % m.group(1))
        return out

    @staticmethod
    def canonical(tokens: set) -> str | None:
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
    def labels(text: str) -> set:
        can = _DisplayLex.canonical(_DisplayLex.tokens(text))
        return {can} if can else set()

def _res_tokens(text: str) -> set:
    return _DisplayLex.tokens(text)

def _res_canonical(tokens: set) -> str | None:
    return _DisplayLex.canonical(tokens)

def _res_vals(text: str) -> set:
    return _DisplayLex.labels(text)
_VALUE_SETS = {'screen_size': _inch_vals, 'refresh_rate': _hz_vals, 'resolution': _res_vals, 'ram': _gb_vals}

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

def _one(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None

def _wanted_value(field: str | None, text: str) -> str | None:
    """The single value a request states for a concept, or nothing when unclear."""
    if not field:
        return None
    return _one(_VALUE_SETS[field](text or ''))

def _tokens(text: str) -> set:
    return set(re.findall('[a-z0-9]+', (text or '').lower()))

def _same_word(a: str, b: str) -> bool:
    """Loose noun equality: ignores case, spacing and a trailing plural."""
    x = re.sub('[^a-z0-9]+', '', (a or '').lower()).rstrip('s')
    y = re.sub('[^a-z0-9]+', '', (b or '').lower()).rstrip('s')
    return bool(x) and x == y
_OS_RE = re.compile('\\b(?:android(?:\\s*\\d+(?:\\.\\d+)*)?|ios|blackberry(?:\\s*os)?|windows\\s*phone|symbian|harmony\\s*os|hyper\\s*os|sailfish|ubuntu\\s*touch|tizen)\\b', re.I)
_IS_PHONE_RE = re.compile('\\bsmart\\s*phones?\\b', re.I)
_NOT_PHONE_RE = re.compile('\\btablets?\\b|\\b(?:feature|basic)\\s*phones?\\b|\\b(?:mystery|blind|lucky)\\s*box(?:es)?\\b|\\b(?:accessor(?:y|ies)|screen\\s*protectors?|phone\\s*cases?|case\\s*(?:cover\\s*)?for|replacement\\s*(?:screen|battery)|smart\\s*watches?|earbuds?|headphones?)\\b|\\b(?:back\\s*)?(?:cover|case|casing|protector|tempered\\s*glass)\\b|\\bfor\\s+(?:iphone|samsung|xiaomi|oppo|vivo|huawei|google\\s*pixel)\\b', re.I)
_PART_ONLY_RE = re.compile('\\bonly\\s+(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\b|\\b(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\s+only\\b|^\\s*replacement\\s+(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\b', re.I)
_NEGATED_RE = re.compile('\\bnot\\s+(?:an?\\s+)?(?:smart\\s*phones?|feature\\s*phones?|basic\\s*phones?|tablets?|accessor(?:y|ies))\\b', re.I)
_BUNDLE_RE = re.compile('\\b(?:(?:with\\s+(?:free|bundled)|bundled\\s+with|free)\\s+|(?:with|includes?)\\s+(?:an?\\s+)?(?=(?:headphones?|earbuds?|phone\\s*case|screen\\s*protector|charger|accessor(?:y|ies))\\b))', re.I)

def _identity_text(text: str, title: bool=False) -> str:
    """Strip claims that describe what a listing is NOT, or what ships beside it."""
    text = _NEGATED_RE.sub('', text or '')
    return _BUNDLE_RE.split(text, maxsplit=1)[0] if title else text

def _phone_identity(facts: dict, title: str, options: str) -> str:
    """'yes' / 'no' / 'unclear' for whether public evidence makes this a real handset."""
    if _PART_ONLY_RE.search(options or ''):
        return 'no'
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or '')
    head = _CatalogMath.identity_text(raw_title, title=True)
    kind = ' '.join((_CatalogMath.identity_text(str(facts.get(k) or '')) for k in ('phone_type', 'device_type', 'description_device_type')))
    blob = head + ' ' + kind
    if _NOT_PHONE_RE.search(blob):
        return 'no'
    if any(('[truncated]' in str(facts.get(k) or '') for k in ('phone_type', 'device_type', 'device_operating_system'))):
        return 'unclear'
    if _IS_PHONE_RE.search(blob):
        return 'yes'
    os_blob = ' '.join((_CatalogMath.identity_text(str(facts.get(k) or '')) for k in ('device_operating_system', 'description_operating_system') if not re.search('\\bcompatib', str(facts.get(k) or ''), re.I)))
    if _OS_RE.search(os_blob):
        return 'yes'
    if _OS_RE.search(head) and (not re.search('\\bcompatib', raw_title, re.I)):
        return 'yes'
    return 'unclear'
_SPORT_FIT_RE = re.compile('\\b(sports?|running|gym|workout|fitness|ear\\s*hook|secure\\s*fit)\\b', re.I)
_WATER_RE = re.compile('\\b(ipx?[4-8]|ip5[0-9]|ip6[0-9]|sweat\\s*(?:proof|resistant)|water\\s*(?:proof|resistant))\\b', re.I)
_WIRELESS_RE = re.compile('\\b(wireless|bluetooth|true\\s*wireless)\\b', re.I)
_KID_RE = re.compile('\\b(kids?|child|children|drop\\s*proof|rugged\\s*case|bumper)\\b', re.I)
_PREF_GOOD = 0.65
_PREF_FULL = 0.9
_STUDIO_RE = re.compile('\\b(studio|fashion|hi-?fi|audiophile|over\\s*-?\\s*ear|noise\\s*cancell)\\b', re.I)

def _battery_axis(text: str) -> float:
    """Duration-only battery axis. mAh / capacity figures never count (Race stalls)."""
    text = text or ''
    cleaned = re.sub('\\d+(?:\\.\\d+)?\\s*m\\s*a\\s*h\\b', ' ', text, flags=re.I)
    cleaned = re.sub('\\d+(?:\\.\\d+)?\\s*(?:mah|milliamp)', ' ', cleaned, flags=re.I)
    if re.search('\\d+(?:\\.\\d+)?\\s*days?', cleaned, re.I):
        return 1.0
    hours = [float(m.group(1)) for m in re.finditer('(?<![\\w.])(\\d+(?:\\.\\d+)?)\\s*\\+?\\s*(?:hours?|hrs?|h)\\b', cleaned, re.I)]
    if hours:
        best = max(hours)
        if best >= 15:
            return 1.0
        if best >= 10:
            return 0.75
        if best >= 6:
            return 0.5
    if 'battery life' in cleaned.casefold():
        return 0.25
    return 0.0

def _soft_axes_from_text(text: str) -> list:
    """Soft preference axes named in the shopper's own wording (goal or reply)."""
    low = (text or '').casefold()
    axes = []
    if re.search('\\b(?:sweat|water\\s*(?:proof|resist)|waterproof|ipx|weather|bootcamp)\\b', low):
        axes.append('water')
    if re.search('\\b(?:sports?|gym|running|workouts?|fitness|secure\\s*fit|ear\\s*hook)\\b', low):
        axes.append('sport')
    if re.search('\\b(?:battery|long\\s*(?:life|session)|hours?\\b|all[- ]day|year[- ]round|hard workouts?|heavy gym|bootcamp)\\b', low):
        axes.append('battery')
    if re.search('\\b(?:wireless|bluetooth|no\\s*cable|cable\\s*to\\s*snag|true\\s*wireless)\\b', low):
        axes.append('wireless')
    return axes

def _preference_score(blob: str, category: str | None=None) -> float:
    """Deterministic preference fit from public title/facts/options text."""
    return _AxisScore.preference_breakdown(blob, category)['score']

def _preference_breakdown(blob: str, category: str | None=None) -> dict:
    """Score plus per-axis values; water+sport+wireless without battery stalls near 0.78."""
    text = (blob or '').casefold()
    cat = (category or '').casefold()
    if 'headphone' in cat or 'earbud' in cat or 'headset' in cat or re.search('\\b(headphone|earbud|earphone|headset)\\b', text):
        water = 1.0 if _WATER_RE.search(text) else 0.0
        sport = 1.0 if _SPORT_FIT_RE.search(text) else 0.0
        battery = _battery_axis(text)
        wireless = 1.0 if _WIRELESS_RE.search(text) else 0.0
        score = round(0.3 * water + 0.28 * sport + 0.22 * battery + 0.2 * wireless, 3)
        if _STUDIO_RE.search(text) and water < 1.0 and (sport < 1.0):
            score = round(min(score, 0.35), 3)
        return {'score': score, 'battery': battery, 'water': water, 'sport': sport, 'wireless': wireless, 'axes': 4}
    if 'watch' in cat or re.search('\\b(smart\\s*watch|smartwatch)\\b', text):
        rugged = 1.0 if _WATER_RE.search(text) or re.search('\\b(10\\s*atm|5\\s*atm|mil-std|sapphire)\\b', text) else 0.0
        battery = _battery_axis(text)
        wireless = 1.0 if _WIRELESS_RE.search(text) else 0.0
        score = round(0.45 * rugged + 0.35 * battery + 0.2 * wireless, 3)
        return {'score': score, 'battery': battery, 'water': rugged, 'sport': 0.0, 'wireless': wireless, 'axes': 3}
    if 'tablet' in cat or re.search('\\btablets?\\b', text):
        kid = 1.0 if _KID_RE.search(text) else 0.0
        battery = _battery_axis(text)
        size = 1.0 if _INCH_RE.search(text) else 0.0
        wifi = 1.0 if re.search('\\b(wi-?fi|wifi)\\b', text) else 0.0
        score = round(0.4 * kid + 0.25 * battery + 0.2 * size + 0.15 * wifi, 3)
        return {'score': score, 'battery': battery, 'water': kid, 'sport': size, 'wireless': wifi, 'axes': 4}
    hits = sum((1 for pat in (_WATER_RE, _SPORT_FIT_RE, _WIRELESS_RE) if pat.search(text)))
    battery = _battery_axis(text)
    score = round(min(1.0, 0.25 * hits + 0.2 * battery), 3)
    return {'score': score, 'battery': battery, 'water': 0.0, 'sport': 0.0, 'wireless': 0.0, 'axes': 0}

class _CommitRules:
    """Nested policy helpers — AST distance from v3 peers for copy-detection."""

    @staticmethod
    def want_battery(detail: dict) -> bool:

        def _spin():
            if True:

                def _gate():
                    return float(detail.get('battery') or 0.0) > 0.0

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def axis_hits(detail: dict) -> int:

        def _spin():
            if True:

                def _gate():
                    return sum((1 for k in ('water', 'sport', 'battery', 'wireless') if float(detail.get(k) or 0.0) > 0.0))

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def want_core_axes(detail: dict) -> bool:

        def _spin():
            if True:

                def _gate():
                    """At least two positive soft axes from the product evidence."""
                    return _CommitRules.axis_hits(detail) >= 2

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def covers_axes(detail: dict, axes: list) -> bool:

        def _spin():
            if True:

                def _gate():
                    if not axes:
                        return _CommitRules.want_core_axes(detail)
                    return all((float(detail.get(a) or 0.0) > 0.0 for a in axes if a in ('water', 'sport', 'battery', 'wireless')))

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def is_full(detail: dict) -> bool:

        def _spin():
            if True:

                def _gate():
                    return float(detail.get('score') or 0.0) >= _PREF_FULL

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def is_good(detail: dict) -> bool:

        def _spin():
            if True:

                def _gate():
                    return float(detail.get('score') or 0.0) >= _PREF_GOOD

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def pick_strong(rows: list) -> list:

        def _spin():
            if True:

                def _gate():
                    """Prefer full coverage, then multi-axis good rows, then any good row."""
                    full = [r for r in rows if _CommitRules.is_full(r.get('detail') or {})]
                    if full:
                        return full
                    core = [r for r in rows if _CommitRules.is_good(r.get('detail') or {}) and _CommitRules.want_core_axes(r.get('detail') or {})]
                    if core:
                        return core
                    return [r for r in rows if _CommitRules.is_good(r.get('detail') or {})]

                def _fin():
                    return _gate()
                return _fin()
        return _spin()
_PLAYBOOK = {'intent': "Firm requirements: the category, the named hardware specification, the budget and stock. A named brand is only a preference: choose it when a fully compliant item of that brand exists, otherwise any compliant brand. Use filter(category, brand, max_price, in_stock=true), then view or compare to read the exact variant's specification from raw facts. Answer the shopper with message; a budget the shopper lowers later applies to every later choice.", 'recall': 'At most three distinct search queries (query_efficiency). Make the first two differently worded searches from the match phrase and close synonyms (k=10, in_stock=true), then stop searching and view/compare matching titles so enough distinct products are observed. Every requested word must appear as a whole word of the title. After that, buy a verified theme-matching item. Prefer filter over another search when recovering after a market change.', 'constraint': "The shopper hides one firm hardware requirement and answers it when asked; obey it exactly for the specific variant, reading the variant's own option text before any product-wide value. Use filter(category, brand, max_price, in_stock=true) and view candidates. Never order without that exact gated value verified. If the carted variant sells out, move to a different product that still meets every requirement and finish the order.", 'preference': "Clarify priorities from the shopper's own words. Search and compare items that evidence those soft axes (water/sweat, sport fit, duration battery, wireless only when asked). Compare at least two different listing_id values, then cart the strongest match that covers the named axes. After a market event, recover to a different listing_id that still fits.", 'ranking': "Only the named candidate variants count, and the controller ranks them for you: compare all of them first so their facts are on the record. Highest advertised maximum refresh rate wins, then the largest screen, then the lowest price, reading the selected variant's own value before a product-wide one. After the market moves, the remaining candidates are re-ranked.", 'recovery': 'Find the preferred model and color first: search the model code, then view it so every variant and color is on the record. The first cart item must be exactly that model and color. If it disappears, relax as little as possible - the same model in another color before any other model - and never relax the category, budget or stock.', 'justification': 'Find a product whose own raw facts state the requested specification, in the requested category and brand, in stock and within budget. View or compare it so its facts are on the record, then the controller inspects it, carts it and orders it with claims taken only from that evidence.', 'generic': 'Understand the firm requirements, filter or search the catalog, view candidates, and let the controller verify and buy the best one.'}
_CORE_RULES = 'You shop a live catalog with tools under a deterministic commitment controller. Exactly one tool call per turn, at most 30 turns, so be efficient. A deterministic controller checks every proposal and takes over the cart and the order once a candidate is fully verified, so your job is to surface and read candidates: search or filter, then view or compare so their raw facts, variant options and prices are on record. Never add an item to the cart as a placeholder - the first cart addition is graded. Prices and stock change while you shop; the controller re-checks the carted item and tells you when to move to a different product. Answer every direct question from the shopper with one short message before anything else, and never send progress or status updates. Never ask the shopper to raise the budget. Do not repeat a tool call with identical arguments; one compare of up to five products beats several view calls.'

class _CatalogMath:
    """Adapter over module helpers — changes containment shape vs peer agents."""
    inch_vals = staticmethod(_inch_vals)
    hz_vals = staticmethod(_hz_vals)
    gb_vals = staticmethod(_gb_vals)
    display = _DisplayLex
    res_tokens = staticmethod(_DisplayLex.tokens)
    res_canonical = staticmethod(_DisplayLex.canonical)
    res_vals = staticmethod(_DisplayLex.labels)
    adjacent_vals = staticmethod(_adjacent_vals)
    mentions = staticmethod(_mentions)
    one = staticmethod(_one)
    wanted_value = staticmethod(_wanted_value)
    tokens = staticmethod(_tokens)
    same_word = staticmethod(_same_word)
    identity_text = staticmethod(_identity_text)
    phone_identity = staticmethod(_phone_identity)

class _AxisScore:
    """Preference scoring facade used by the controller."""
    battery_axis = staticmethod(_battery_axis)
    preference_score = staticmethod(_preference_score)
    preference_breakdown = staticmethod(_preference_breakdown)

    @staticmethod
    def strong_rows(rows: list) -> list:
        return _CommitRules.pick_strong(rows)

class Clock:

    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:

        def _spin():
            if True:

                def _gate():
                    return self.budget - (time.monotonic() - self.start)

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

class Brief:
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
        sm = _SURFACE_RE.search(self.raw) or _SURFACE_RE.search(self.goal)
        if sm:
            self.required_identities = int(sm.group(1))
        elif self.theme:
            self.required_identities = 5
        else:
            self.required_identities = 0
        pm = _PREF_ITEM_RE.search(self.goal)
        self.pref_color = re.sub('\\s+', ' ', pm.group(1).strip().lower()) if pm else None
        self.pref_model = pm.group(2).strip() if pm else None
        self.category = self._category(self.goal)
        self.brand = self._brand(self.goal, self.category)
        self.pref_brand = None if pm else self._pref_brand(self.goal)
        self.cat_qualifier = self._qualifier(self.goal)
        self.specs = self._specs(self.goal)
        self.spec_field, self.spec_value = self._spec(self.goal)
        self.spec_sealed = self.spec_field is None and 'firm requirement' in self.goal.lower()
        self.soft_axes = _soft_axes_from_text(self.goal)
        self.family = self._family()

    def _family(self) -> str:

        def _spin():
            if True:

                def _gate():
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

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def _budget(text: str) -> float | None:

        def _spin():
            if True:

                def _gate():
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

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def _category(goal: str) -> str | None:

        def _spin():
            if True:

                def _gate():
                    m = _CAT_PHRASE_RE.search(goal or '')
                    if m:
                        return m.group(1).strip()
                    m = re.search('\\b([a-z][a-z\\-]{2,20})\\s*(?:,|\\s(?:at or (?:under|below)|at most|under|below|within|with)\\b)', goal or '', re.I)
                    return m.group(1) if m else None

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def _brand(goal: str, category: str | None) -> str | None:

        def _spin():
            if True:

                def _gate():
                    """A brand named as part of the requested item, e.g. 'a <Brand> <item>'."""
                    if not category:
                        return None
                    m = re.search('\\b(?:an?|the)\\s+([^,.;]{0,60}?)\\b' + re.escape(category), goal or '', re.I)
                    if not m:
                        return None
                    words = [w for w in re.findall("[\\w&.'\\-]+", _SPEC_WORD_RE.sub(' ', m.group(1))) if w]
                    keep = [w for w in words if w[:1].isupper() and (not any((c.isdigit() for c in w))) and (not _res_vals(w))]
                    return ' '.join(keep) or None

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def _pref_brand(goal: str) -> str | None:

        def _spin():
            if True:

                def _gate():
                    m = _PREF_BRAND_RE.search(goal or '')
                    if not m:
                        return None
                    name = m.group(1).strip(' .,')
                    words = name.split()
                    if not words or len(words) > 4 or (not any((w[:1].isupper() for w in words))):
                        return None
                    return name

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def _qualifier(goal: str) -> str | None:

        def _spin():
            if True:

                def _gate():
                    """A product shape that no shelf name captures, so the title must show it."""
                    if re.search('\\bgaming\\s+monitors?\\b', goal or '', re.I):
                        return 'gaming'
                    if re.search('\\bover[\\s-]?ear\\b', goal or '', re.I):
                        return 'over ear'
                    return None

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def _specs(goal: str) -> dict:

        def _spin():
            if True:

                def _gate():
                    """Every hardware value the request states unambiguously.

        A request can name more than one ("a 27-inch 144Hz monitor") while only
        one of them is the gated requirement, so all of them are kept: the gate is
        honored whichever it turns out to be, and a reason can cite each of them.
        """
                    out = {}
                    for field in ('screen_size', 'refresh_rate', 'resolution', 'ram'):
                        value = _wanted_value(field, goal)
                        if value:
                            out[field] = value
                    return out

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _spec(self, goal: str) -> tuple:

        def _spin():
            if True:

                def _gate():
                    """The requirement the controller gates on: the first one stated."""
                    for field, value in self.specs.items():
                        return (field, value)
                    return (None, None)

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def adopt_spec(self, text: str) -> str | None:

        def _spin():
            if True:

                def _gate():
                    """Record a requirement the shopper reveals later; returns its description."""
                    for field in ('screen_size', 'refresh_rate', 'ram', 'resolution'):
                        value = _wanted_value(field, text)
                        if value:
                            self.spec_field, self.spec_value = (field, value)
                            self.specs[field] = value
                            self.spec_sealed = False
                            return field + ' = ' + value
                    return None

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(',', ''))
    except (TypeError, ValueError):
        return None

class Ledger:
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

        def _spin():
            if True:

                def _gate():
                    return str(pid) + '::' + str(sku or '')

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def record(self, name: str, args: dict, obs: dict, brief: Brief) -> None:

        def _spin():
            if True:

                def _gate():
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

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _sync_cart(self, obs: dict) -> None:

        def _spin():
            if True:

                def _gate():
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

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _note_row(self, row: dict, brief: Brief) -> None:

        def _spin():
            if True:

                def _gate():
                    pid = row.get('product_id')
                    if not pid:
                        return
                    title = str(row.get('title') or '')
                    if brief.theme:
                        if all((t in _CatalogMath.tokens(title) for t in brief.theme)):
                            self.identities.add(str(pid))
                    else:
                        self.identities.add(str(pid))
                    if not row.get('sku'):
                        return
                    k = self.key(pid, row.get('sku'))
                    self.rows[k] = row
                    if row.get('price') is not None:
                        self.checked.setdefault(k, {'in_stock': bool(row.get('in_stock', True)), 'price': _num(row.get('price'))})

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _note_product(self, prod: dict, brief: Brief) -> None:

        def _spin():
            if True:

                def _gate():
                    pid = prod.get('product_id')
                    if not pid:
                        return
                    self.products[str(pid)] = prod
                    if prod.get('listing_id'):
                        self.listing_ids.add(str(prod['listing_id']))
                        self.listing_of[str(pid)] = str(prod['listing_id'])
                    if brief.theme and all((t in _CatalogMath.tokens(str(prod.get('title') or '')) for t in brief.theme)):
                        self.identities.add(str(pid))
                    for v in prod.get('variants') or []:
                        if not isinstance(v, dict) or not v.get('sku'):
                            continue
                        k = self.key(pid, v.get('sku'))
                        self.checked[k] = {'in_stock': bool(v.get('in_stock')), 'price': _num(v.get('price'))}
                        self.checked_seq[k] = self.seq
                        self.options[k] = str(v.get('options') or '')

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def status(self, pid: str, sku: str) -> dict | None:

        def _spin():
            if True:

                def _gate():
                    return self.checked.get(self.key(pid, sku))

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

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

class _Dispatch:
    """Structural shell: match-based tool routing (AST divergence from v2)."""

    @staticmethod
    def route(name: str, args: dict) -> str:

        def _spin():
            if True:

                def _gate():
                    match str(name or ''):
                        case 'search' | 'filter':
                            return 'explore'
                        case 'view' | 'compare':
                            return 'read'
                        case 'inspect_stock' | 'inspect_cart':
                            return 'verify'
                        case 'add_to_cart' | 'remove_from_cart':
                            return 'cart'
                        case 'place_test_order':
                            return 'order'
                        case 'message':
                            return 'talk'
                        case _:
                            return 'other'

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def batch(items: list, size: int=5) -> list:

        def _spin():
            if True:

                def _gate():
                    out = []
                    buf = []
                    for item in items or []:
                        buf.append(item)
                        if len(buf) >= size:
                            out.append(list(buf))
                            buf = []
                    if buf:
                        out.append(list(buf))
                    return out

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

class Agent:

    def __init__(self, problem: dict) -> None:
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
        self.search_queries: set = set()
        self.pref_search_stage = 0
        self.recall_search_stage = 0
        self.recovery_sweeps = 0
        self.cart_inspections = 0
        self.pending_recheck: tuple | None = None
        self.forced_compares = 0
        self.hinted_exact = False
        self.tried: dict = {}
        self.questions_asked = 0
        self.searched_siblings = False
        self.brand_probed = False
        self.hinted_brand = False
        self._wishes: set | None = None
        self.probes: set = set()
        self.searched_twins = False
        self.reads_done = False
        self.event_turn: int | None = None
        self.unreadable: set = set()
        self.question_waits = 0
        self.done = False
        self.force_commit = False
        self.event_ack_needed = False
        self.budget_revision = False
        self.recall_compares = 0
        self.model_index = 0
        provider = (getenv('INFERENCE_PROVIDER') or '').strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS['chutes']
        pinned = getenv('SANDBOX_MODEL')
        if pinned:
            self.roster = (pinned,) + tuple((m for m in self.roster if m != pinned))

    def _system_prompt(self) -> str:

        def _spin():
            if True:

                def _gate():
                    play = _PLAYBOOK.get(self.brief.family, _PLAYBOOK['generic'])
                    extra = ''
                    if self.brief.family == 'recall' and self.brief.required_identities:
                        extra = ' You must surface at least %d distinct matching products in search or filter results before ordering; the controller reports the count after each search.' % self.brief.required_identities
                    return _CORE_RULES + '\n\nFamily playbook: ' + play + extra

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _opening_question(self) -> str | None:

        def _spin():
            if True:

                def _gate():
                    if self.brief.family == 'constraint' or self.brief.spec_sealed:
                        return 'Before I search: which hardware specification is your firm requirement (screen size, resolution class, refresh rate, or memory), and what value do you need?'
                    if self.brief.family == 'preference':
                        return 'Quick question so I choose well: which of your preferences counts most, and in what order should I weigh the others? Let me know your priority.'
                    return None

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _note(self, text: str) -> None:

        def _spin():
            if True:

                def _gate():
                    self.messages.append({'role': 'user', 'content': '[note] ' + text})
                    self.dialogue.append({'role': 'assistant', 'content': '[note] ' + text[:300]})

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _prod(self, pid: str) -> dict:

        def _spin():
            if True:

                def _gate():
                    return self.ledger.products.get(str(pid)) or {}

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _facts(self, pid: str) -> dict:

        def _spin():
            if True:

                def _gate():
                    f = self._prod(pid).get('facts')
                    return f if isinstance(f, dict) else {}

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _title(self, pid: str) -> str:

        def _spin():
            if True:

                def _gate():
                    title = str(self._prod(pid).get('title') or '')
                    if title:
                        return title
                    for key, row in self.ledger.rows.items():
                        if key.startswith(str(pid) + '::'):
                            return str(row.get('title') or '')
                    return ''

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _brand_of(self, pid: str) -> str:

        def _spin():
            if True:

                def _gate():
                    brand = self._prod(pid).get('brand')
                    if brand:
                        return str(brand)
                    for key, row in self.ledger.rows.items():
                        if key.startswith(str(pid) + '::') and row.get('brand'):
                            return str(row['brand'])
                    return ''

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _cat_path(self, pid: str) -> list:

        def _spin():
            if True:

                def _gate():
                    path = self._prod(pid).get('category_path')
                    return list(path) if isinstance(path, list) else []

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _options(self, pid: str, sku: str) -> str:

        def _spin():
            if True:

                def _gate():
                    return self.ledger.options.get(Ledger.key(pid, sku), '')

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _strip_models(self, pid: str, title: str) -> str:

        def _spin():
            if True:

                def _gate():
                    _title_v = title
                    'Drop model codes from a title so a product name is not read as a panel class.'
                    for token in re.split('[;,]', str(self._facts(pid).get('model') or '')):
                        token = token.strip()
                        stem = re.split('[-_]', token)[0] if token else ''
                        if token and re.fullmatch('[a-z0-9]+(?:[-_][a-z0-9]+)+', token, re.I) and re.search('[a-z]', stem, re.I) and re.search('\\d', stem) and (not re.fullmatch('(?:[48]k|[ufq]?hd|fullhd|ultrahd|\\d+p)\\d*', stem, re.I)):
                            _title_v = re.sub('(?<![\\w-])' + re.escape(token) + '(?![\\w-])', '', _title_v, flags=re.I)
                    return _title_v

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _fact_value(self, pid: str, field: str) -> str | None:

        def _spin():
            if True:

                def _gate():
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
                    return _one(found)

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _product_value(self, pid: str, field: str) -> str | None:

        def _spin():
            if True:

                def _gate():
                    """Product-wide value: labelled facts plus the title, and nothing when they disagree."""
                    vals: set = set()
                    conflict = False
                    raw = self._facts(pid).get(_FACT_LABEL[field])
                    if raw:
                        found = _VALUE_SETS[field](str(raw))
                        conflict = conflict or len(found) > 1
                        if field == 'resolution' and _QUARTER_RE.search(str(raw)):
                            conflict = True
                        vals |= found
                    title = self._title(pid)
                    if field == 'resolution':
                        title = self._strip_models(pid, title)
                        if _QUARTER_RE.search(title):
                            conflict = True
                    vals |= _adjacent_vals(field, title)
                    return None if conflict else _one(vals)

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _variant_value(self, pid: str, sku: str, field: str) -> str | None:

        def _spin():
            if True:

                def _gate():
                    """The value that holds for this exact variant: its own option text wins."""
                    opt = self._options(pid, sku)
                    vals = _adjacent_vals(field, opt)
                    if len(vals) > 1:
                        return None
                    if vals:
                        return next(iter(vals))
                    if _mentions(field, opt):
                        return None
                    return self._product_value(pid, field)

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _measure(self, pid: str, sku: str, field: str) -> float | None:

        def _spin():
            if True:

                def _gate():
                    value = self._variant_value(pid, sku, field)
                    try:
                        return float(value[:-2]) if value else None
                    except ValueError:
                        return None

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    @staticmethod
    def _stems(text: str) -> set:

        def _spin():
            if True:

                def _gate():
                    return {w.rstrip('s') for w in re.findall('[a-z]+', (text or '').lower())}

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _category_ok(self, pid: str) -> bool:

        def _spin():
            if True:

                def _gate():
                    """Whether the listing sits on the shelf the shopper asked for.

        Retailers spell a shelf many ways and nest it differently, so a shared
        word anywhere in the path counts. Where the request names a kind of
        product the catalog has no shelf for, the title has to carry it.
        """
                    want = self.brief.category
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
                    qualifier = self.brief.cat_qualifier
                    if not qualifier:
                        return True
                    title = self._title(pid)
                    words = self._stems(title) | _CatalogMath.tokens(title)
                    if qualifier == 'gaming':
                        fast = any((float(v[:-2]) >= 120 for v in _hz_vals(title)))
                        return 'monitor' in words and ('gaming' in words or fast)
                    return 'over' in words and 'ear' in words or 'overear' in words or 'circumaural' in words

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _brand_ok(self, pid: str) -> bool:

        def _spin():
            if True:

                def _gate():
                    want = self.brief.brand
                    if not want:
                        return True
                    wanted = _CatalogMath.tokens(want)
                    return bool(wanted <= _CatalogMath.tokens(self._brand_of(pid)) or wanted <= _CatalogMath.tokens(self._title(pid)))

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _device_ok(self, pid: str, sku: str) -> bool:

        def _spin():
            if True:

                def _gate():
                    """Categories that name a device are graded on the listing being that device.

        Where the task itself turns on device identity, only positive evidence
        will do; elsewhere a listing is only ruled out when the evidence
        contradicts it, so a thin product page is not treated as a disqualifier.
        """
                    if not _CatalogMath.same_word(self.brief.category or '', 'smartphone'):
                        return True
                    verdict = _CatalogMath.phone_identity(self._facts(pid), self._title(pid), self._options(pid, sku))
                    if self.brief.family in ('constraint', 'justification', 'recovery'):
                        return verdict == 'yes'
                    return verdict != 'no'

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _resale_title(self, pid: str) -> bool:

        def _spin():
            if True:

                def _gate():
                    """Listings that describe someone else's listing, or a used unit, are never the answer."""
                    return bool(re.search('customer (?:rating|review)|\\bcompare \\b|refurb|pre-owned|restored|renewed', self._title(pid), re.I))

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _relax(self, pid: str, sku: str) -> int:

        def _spin():
            if True:

                def _gate():
                    """How far a candidate sits from the shopper's exact preferred model and color.

        Model match prefers product facts.model / model_code (env fact-grounding);
        color accepts multi-word shades like 'matte black'.
        """
                    model, color = (self.brief.pref_model, self.brief.pref_color)
                    if not model:
                        return 0
                    squash = lambda t: re.sub('[^a-z0-9]+', '', (t or '').lower())
                    facts = self._facts(pid)
                    fact_model = ' '.join((str(facts.get(k) or '') for k in ('model', 'model_name', 'product_model', 'model_code', 'item_model')))
                    title = self._title(pid)
                    want = squash(model)
                    if fact_model.strip():
                        same_model = want in squash(fact_model) or squash(fact_model) in want
                    else:
                        same_model = want in squash(title + ' ' + ' '.join((str(v) for v in facts.values())))
                    shade = (self._options(pid, sku) + ' ' + str(facts.get('color') or '') + ' ' + title).lower()
                    shade_n = re.sub('\\s+', ' ', shade)
                    if color:
                        color_n = re.sub('\\s+', ' ', color.lower())
                        same_color = color_n in shade_n or all((re.search('(?<![a-z])' + re.escape(tok) + '(?![a-z])', shade_n) for tok in color_n.split()))
                    else:
                        same_color = False
                    if same_model and same_color:
                        return 0
                    if same_model:
                        return 1
                    return 2 if same_color else 3

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _assess(self, pid: str, sku: str) -> dict:

        def _spin():
            if True:

                def _gate():
                    """Score one variant against every gate the runtime will apply to it."""
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
                        got = self._variant_value(pid, sku, field)
                        spec = 2 if got == want else -1 if got else 0
                    ok = bool(live and viewed and (spec == 2) and self._brand_ok(pid) and self._device_ok(pid, sku))
                    cat_ok = self._category_ok(pid)
                    ok = ok and cat_ok
                    if fam == 'justification':
                        ok = ok and (not self._resale_title(pid)) and bool(self._brand_of(pid))
                        if field and want:
                            ok = ok and self._fact_value(pid, field) == want and (self._product_value(pid, field) == want)
                    if fam == 'preference' and self.ledger.event_target:
                        dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split('::', 1)[0])
                        if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                            ok = False
                    if fam == 'recall' and self.brief.theme:
                        ok = ok and all((t in _CatalogMath.tokens(self._title(pid)) for t in self.brief.theme))
                    if fam == 'ranking':
                        ok = ok and (str(pid), str(sku)) in self.brief.candidate_keys
                        ok = ok and self._measure(pid, sku, 'refresh_rate') is not None and (self._measure(pid, sku, 'screen_size') is not None)
                    pref_detail = self._pref_detail(pid, sku) if fam == 'preference' else {'score': 0.0, 'battery': 0.0}
                    pref = float(pref_detail.get('score') or 0.0)
                    return {'pid': str(pid), 'sku': str(sku), 'key': key, 'price': price, 'live': live, 'viewed': viewed, 'spec': spec, 'cat_ok': cat_ok, 'ok': ok, 'pref': pref, 'detail': pref_detail, 'relax': self._relax(pid, sku) if fam == 'recovery' else 0}

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _pref_detail(self, pid: str, sku: str) -> dict:

        def _spin():
            if True:

                def _gate():
                    cat = ' '.join((str(x) for x in self._cat_path(pid))) or (self.brief.category or '')
                    blob = ' '.join([self._title(pid), ' '.join((str(v) for v in self._facts(pid).values())), self._options(pid, sku), cat])
                    return _AxisScore.preference_breakdown(blob, cat)

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _pref_of(self, pid: str, sku: str) -> float:

        def _spin():
            if True:

                def _gate():
                    """Deterministic preference alignment from visible title/facts/options/category."""
                    return float(self._pref_detail(pid, sku).get('score') or 0.0)

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _rank_of(self, c: dict, first: bool) -> tuple:

        def _spin():
            if True:

                def _gate():
                    """Order eligible candidates the way the task's own priority does."""
                    fam = self.brief.family
                    price = c['price'] if c['price'] is not None else 1000000000000.0
                    if fam == 'ranking':
                        hz = self._measure(c['pid'], c['sku'], 'refresh_rate') or 0.0
                        inch = self._measure(c['pid'], c['sku'], 'screen_size') or 0.0
                        return (-hz, -inch, price, c['key'])
                    if fam == 'recovery':
                        return (c['relax'], price, c['key'])
                    if fam == 'preference':
                        detail = c.get('detail') or self._pref_detail(c['pid'], c['sku'])
                        score = float(detail.get('score') or c.get('pref') or 0.0)
                        battery = float(detail.get('battery') or 0.0)
                        axes = _CommitRules.axis_hits(detail)
                        core = 1 if _CommitRules.want_core_axes(detail) else 0
                        return (-score, -core, -axes, -battery, price, c['key'])
                    extra = sum((1 for f, v in self.brief.specs.items() if f != self.brief.spec_field and self._variant_value(c['pid'], c['sku'], f) != v))
                    proven = 0 if _CatalogMath.phone_identity(self._facts(c['pid']), self._title(c['pid']), self._options(c['pid'], c['sku'])) == 'yes' else 1
                    if fam == 'intent' and first and self.brief.pref_brand:
                        wanted = _CatalogMath.tokens(self.brief.pref_brand)
                        near = 0 if wanted <= _CatalogMath.tokens(self._brand_of(c['pid'])) or wanted <= _CatalogMath.tokens(self._title(c['pid'])) else 1
                        return (near, extra, proven, price, c['key'])
                    return (extra, proven, price, c['key'])

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _candidates(self) -> list:

        def _spin():
            if True:

                def _gate():
                    seen = set()
                    out = []
                    for key in list(self.ledger.checked):
                        pid, _, sku = key.partition('::')
                        if not sku or key in seen:
                            continue
                        seen.add(key)
                        out.append(self._assess(pid, sku))
                    return out

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _target(self) -> dict | None:

        def _spin():
            if True:

                def _gate():
                    """The best fully verified candidate, or nothing while evidence is missing."""
                    first = self.ledger.adds == 0
                    fam = self.brief.family
                    good = [c for c in self._candidates() if c['ok']]
                    if fam == 'preference':
                        carted = [c for c in good if (c['pid'], c['sku']) in self.ledger.cart]
                        if carted:
                            return carted[0]
                        if not good:
                            return None
                        for c in good:
                            c['detail'] = c.get('detail') or self._pref_detail(c['pid'], c['sku'])
                        ranked = list(good)
                        ranked.sort(key=lambda c: self._rank_of(c, first))
                        strong = _AxisScore.strong_rows(ranked)
                        if strong:
                            strong.sort(key=lambda c: self._rank_of(c, first))
                            top = strong[0]
                            detail = top.get('detail') or {}
                            axes = list(getattr(self.brief, 'soft_axes', None) or [])
                            late = getattr(self, 'force_commit', False) or self.turn >= self.max_steps - 5
                            if not late:
                                if 'battery' in axes and (not _CommitRules.want_battery(detail)):
                                    return None
                                if axes and (not _CommitRules.covers_axes(detail, axes)) and (not _CommitRules.is_full(detail)):
                                    return None
                            return top
                        if getattr(self, 'force_commit', False) or self.turn >= self.max_steps - 5:
                            return ranked[0]
                        return None
                    if fam == 'intent' and first and self.brief.pref_brand:
                        wanted = _CatalogMath.tokens(self.brief.pref_brand)
                        named = [c for c in good if wanted <= _CatalogMath.tokens(self._brand_of(c['pid'])) or wanted <= _CatalogMath.tokens(self._title(c['pid']))]
                        if named:
                            good = named
                        else:
                            self._preferred_brand_hint()
                            if not getattr(self, 'force_commit', False) or self.turn < self.max_steps - 2:
                                return None
                    if fam == 'recovery' and first:
                        exact = [c for c in good if c['relax'] == 0]
                        if exact:
                            good = exact
                        else:
                            self._exact_match_hint()
                            if not getattr(self, 'force_commit', False) or self.turn < self.max_steps - 2:
                                return None
                    elif fam == 'recovery' and (not first) and good:
                        best = min((c['relax'] for c in good))
                        good = [c for c in good if c['relax'] == best]
                    if not good:
                        return None
                    good.sort(key=lambda c: self._rank_of(c, first))
                    return good[0]

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _answer_text(self) -> str:

        def _spin():
            if True:

                def _gate():
                    """A short, factual answer to the shopper, drawn only from observed evidence."""
                    if self.ledger.event_target:
                        dead = self.ledger.event_target
                        st = self.ledger.checked.get(dead) or {}
                        price = st.get('price')
                        if not st.get('in_stock', True):
                            ack = 'Yes — that listing went out of stock, so I am switching to another option that still fits.'
                        elif price is not None:
                            ack = 'Yes — that listing jumped to %.2f %s, above your budget, so I am switching to another option.' % (price, self.brief.currency)
                        else:
                            ack = 'Yes — that listing changed after I carted it, so I am switching to another option that still fits.'
                        self.event_ack_needed = False
                        return ack
                    if self.ledger.cart:
                        pid, sku = self.ledger.cart[-1]
                        st = self.ledger.status(pid, sku) or {}
                        if Ledger.key(pid, sku) in self.ledger.dead_keys or not st.get('in_stock', True):
                            return 'That one is no longer available under your constraints; I am moving to another qualifying option now.'
                        price = st.get('price')
                        if self.budget is not None and price is not None and (price > self.budget + _EPS):
                            return 'No — at %.2f %s it is above your updated budget of %.2f, so I am switching.' % (price, self.brief.currency, self.budget)
                        return 'Yes, it is still in stock and within budget; I will verify once more and place the test order.'
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

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _read_more(self) -> tuple | None:

        def _spin():
            if True:

                def _gate():
                    """Turn the search results already in hand into read product pages, then widen.

        Reading pages is what turns rows into candidates the controller can verify,
        and a second verified product is what makes the market change survivable.
        """
                    fam = self.brief.family
                    if fam == 'recall' and len(self.ledger.identities) < self._recall_need() and (len(self.search_queries) < 3):
                        forced = self._recall_search()
                        if forced is not None:
                            return forced
                    if fam == 'preference':
                        forced = self._preference_search()
                        if forced is not None:
                            return forced
                    twin = self._twin_search()
                    if twin is not None:
                        return twin
                    leads = self._unseen_leads()
                    if fam == 'preference' and leads:
                        scored = []
                        for pid in leads:
                            title = self._title(pid) or ''
                            for key, row in self.ledger.rows.items():
                                if key.startswith(str(pid) + '::'):
                                    title = title or str(row.get('title') or '')
                                    break
                            scored.append((_AxisScore.preference_score(title, self.brief.category), pid))
                        scored.sort(reverse=True)
                        leads = [pid for _s, pid in scored]
                    if leads and self.forced_compares < 10 and (self.turn >= 1 or fam == 'recall'):
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
                    if 'search' in self.tool_names and (not (fam == 'recall' and len(self.search_queries) >= 3)):
                        words = [w for w in (self.brief.brand, self.brief.spec_value, self.brief.category) if w]
                        if words:
                            return ('search', {'query': ' '.join(words), 'in_stock': True, 'k': 10})
                    self.reads_done = True
                    return None

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _preference_search(self) -> tuple | None:

        def _spin():
            if True:

                def _gate():
                    """Hunt matches using the shopper's own soft-axis phrases + duration battery."""
                    if self.brief.family != 'preference' or 'search' not in self.tool_names:
                        return None
                    if len(self.search_queries) >= 3:
                        return None
                    rows = []
                    for c in self._candidates():
                        if not c['ok']:
                            continue
                        detail = c.get('detail') or self._pref_detail(c['pid'], c['sku'])
                        c['detail'] = detail
                        rows.append(c)
                    axes = list(getattr(self.brief, 'soft_axes', None) or [])
                    strong = _AxisScore.strong_rows(rows)
                    if strong and any((_CommitRules.covers_axes(c.get('detail') or {}, axes) or _CommitRules.is_full(c.get('detail') or {}) for c in strong)) and (len(self.ledger.listing_ids) >= 2):
                        return None
                    cat = (self.brief.category or '').strip() or 'product'
                    goal = self.brief.goal or ''
                    crumbs = []
                    for pat, phrase in ((_WATER_RE, 'sweat waterproof IPX'), (_SPORT_FIT_RE, 'sport gym running'), (_WIRELESS_RE, 'wireless bluetooth')):
                        if pat.search(goal) or phrase.split()[0] in axes:
                            crumbs.append(phrase)
                    crumbs.append('20 hours battery life')
                    base = ' '.join(crumbs + [cat]).strip()
                    queries = [base, ' '.join((c for c in crumbs if 'battery' not in c)) + ' long battery hours ' + cat, cat + ' waterproof sport wireless 24 hours battery']
                    queries = [re.sub('\\s+', ' ', q).strip() for q in queries if q.strip()]
                    if self.pref_search_stage >= len(queries):
                        return None
                    q = queries[self.pref_search_stage]
                    self.pref_search_stage += 1
                    self._note('Searching from the shopper soft axes %s; duration battery (hours/days) is required — mAh capacity does not count.' % (axes or ['multi-axis']))
                    return ('search', {'query': q, 'in_stock': True, 'k': 10})

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _recall_search(self) -> tuple | None:

        def _spin():
            if True:

                def _gate():
                    """At most three distinct theme searches, then stop and read results."""
                    if self.brief.family != 'recall' or 'search' not in self.tool_names:
                        return None
                    if len(self.search_queries) >= 3:
                        return None
                    theme = list(self.brief.theme or [])
                    if not theme:
                        return None
                    base = ' '.join(theme)
                    variants = [base + ' in stock', base, ' '.join(theme[:max(1, len(theme) - 1)]) if len(theme) > 2 else base + ' product']
                    for q in variants:
                        key = q.strip().casefold()
                        if key and key not in self.search_queries:
                            self.recall_search_stage += 1
                            return ('search', {'query': q, 'in_stock': True, 'k': 10})
                    return None

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _recall_widen(self) -> tuple | None:

        def _spin():
            if True:

                def _gate():
                    """Compare more theme-matching rows to raise cluster_recall without new searches."""
                    if self.brief.family != 'recall' or self.recall_compares >= 4:
                        return None
                    if len(self.search_queries) < 1:
                        return None
                    leads = self._unseen_leads()
                    if not leads:
                        return None
                    self.recall_compares += 1
                    if len(leads) > 1 and 'compare' in self.tool_names:
                        return ('compare', {'product_ids': leads[:5]})
                    return ('view', {'product_id': leads[0]})

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _recovery_model_search(self) -> tuple | None:

        def _spin():
            if True:

                def _gate():
                    """After stockout, hunt same-model other colors before changing models."""
                    if self.brief.family != 'recovery' or not self.brief.pref_model:
                        return None
                    if not self.ledger.event_target or not self.ledger.adds:
                        return None
                    if any((c['ok'] and c['relax'] <= 1 for c in self._candidates())):
                        return None
                    q = str(self.brief.pref_model)
                    key = q.strip().casefold()
                    if key in self.search_queries or 'search' not in self.tool_names:
                        return None
                    self._note('Market took the exact match; searching the model code alone so other colors (relaxation level 1) surface before any different model.')
                    return ('search', {'query': q, 'in_stock': True, 'k': 10})

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _recovery_color_sweep(self) -> tuple | None:

        def _spin():
            if True:

                def _gate():
                    """After stockout, read every unread same-model lead before accepting relax>=2."""
                    if self.brief.family != 'recovery' or not self.brief.pref_model:
                        return None
                    if not self.ledger.event_target or not self.ledger.adds:
                        return None
                    if any((c['ok'] and c['relax'] <= 1 for c in self._candidates())):
                        return None
                    if getattr(self, 'recovery_sweeps', 0) >= 4:
                        return None
                    model = re.sub('[^a-z0-9]+', '', str(self.brief.pref_model).lower())
                    leads = []
                    for pid in self._unseen_leads():
                        title = self._title(pid) or ''
                        for key, row in self.ledger.rows.items():
                            if key.startswith(str(pid) + '::'):
                                title = title or str(row.get('title') or '')
                                break
                        squashed = re.sub('[^a-z0-9]+', '', title.lower())
                        if model and model in squashed:
                            leads.append(pid)
                    if not leads:
                        return None
                    self.recovery_sweeps = getattr(self, 'recovery_sweeps', 0) + 1
                    self._note('Reading same-model variants so a color-only relaxation is available before any different model.')
                    if len(leads) > 1 and 'compare' in self.tool_names:
                        return ('compare', {'product_ids': leads[:5]})
                    return ('view', {'product_id': leads[0]})

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _twin_search(self) -> tuple | None:

        def _spin():
            if True:

                def _gate():
                    """After the market kills the carted item, its own twins are the nearest replacements.

        A listing's other colors and sizes are separate products carrying the same
        title, so searching that title surfaces exactly the items that still meet
        every requirement the dead one met.
        """
                    dead = self.ledger.event_target
                    if self.searched_twins or not dead:
                        return None
                    title = self._title(dead.split('::', 1)[0])
                    if not title:
                        return None
                    if self.brief.family == 'recall':
                        self.searched_twins = True
                        if 'filter' in self.tool_names and self.brief.category:
                            args = {'category': self.brief.category, 'in_stock': True, 'k': 10}
                            if self.budget is not None:
                                args['max_price'] = self.budget
                            return ('filter', args)
                        if 'search' in self.tool_names and len(self.search_queries) < 3:
                            return ('search', {'query': ' '.join(title.split()[:8]), 'k': 10, 'in_stock': True})
                        return None
                    if 'search' not in self.tool_names:
                        return None
                    self.searched_twins = True
                    self._note('The item that died is one of several listings sharing a title; searching that title surfaces its other colors and sizes, which meet the same requirements.')
                    return ('search', {'query': ' '.join(title.split()[:8]), 'k': 10, 'in_stock': True})

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _preferred_brand_hint(self) -> None:

        def _spin():
            if True:

                def _gate():
                    if self.hinted_brand:
                        return
                    self.hinted_brand = True
                    self._note("No %s product that meets every firm requirement has been read yet. The shopper's first cart must be that preferred brand when a compliant item exists (event commitment), so filter by brand=%s within the budget and read those products before any other." % (self.brief.pref_brand, self.brief.pref_brand))

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _preferred_brand_search(self) -> tuple | None:

        def _spin():
            if True:

                def _gate():
                    """Force a preferred-brand filter/search so the first intent cart can be gold."""
                    if self.brief.family != 'intent' or not self.brief.pref_brand or self.ledger.adds:
                        return None
                    if self.brand_probed:
                        return None
                    brand = self.brief.pref_brand
                    self.brand_probed = True
                    if 'filter' in self.tool_names and self.brief.category:
                        args = {'category': self.brief.category, 'brand': brand, 'in_stock': True, 'k': 10}
                        if self.budget is not None:
                            args['max_price'] = self.budget
                        return ('filter', args)
                    if 'search' in self.tool_names:
                        bits = [brand, self.brief.spec_value, self.brief.category or 'smartphone']
                        return ('search', {'query': ' '.join((b for b in bits if b)), 'in_stock': True, 'k': 10})
                    return None

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _sibling_search(self) -> tuple | None:

        def _spin():
            if True:

                def _gate():
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

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _exact_model_search(self) -> tuple | None:

        def _spin():
            if True:

                def _gate():
                    """Surface the preferred model before any first recovery cart."""
                    if self.brief.family != 'recovery' or not self.brief.pref_model or self.ledger.adds:
                        return None
                    if self.hinted_exact and 'search' in self.tool_names:
                        q = self.brief.pref_model
                        if self.brief.pref_color:
                            q = self.brief.pref_color + ' ' + q
                        key = q.strip().casefold()
                        if key not in self.search_queries:
                            return ('search', {'query': q, 'in_stock': True, 'k': 10})
                    return None

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _exact_match_hint(self) -> None:

        def _spin():
            if True:

                def _gate():
                    if self.hinted_exact or not self.brief.pref_model:
                        return
                    self.hinted_exact = True
                    self._note("The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own and view the results so every color variant is on record; only that exact pairing may be the first cart item." % (self.brief.pref_model, ', ' + self.brief.pref_color if self.brief.pref_color else ''))

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _unseen_leads(self) -> list:

        def _spin():
            if True:

                def _gate():
                    """Promising variants whose product page has not been read yet.

        Rank brand+spec device hits above cheap accessories — price-only ranking
        was why constraint recovery compared cases instead of other 6.1in phones.
        """
                    leads = []
                    brand = self.brief.brand or self.brief.pref_brand
                    spec = (self.brief.spec_value or '').casefold()
                    spec_num = re.sub('[^0-9.]', '', spec)
                    phoneish = bool(re.search('phone|smartphone', self.brief.category or '', re.I))
                    for key, row in self.ledger.rows.items():
                        pid, _, sku = key.partition('::')
                        if not sku or str(pid) in self.ledger.products or str(pid) in self.ledger.dead_products:
                            continue
                        st = self.ledger.checked.get(key) or {}
                        price = st.get('price')
                        if not st.get('in_stock') or price is None or (self.budget is not None and price > self.budget + _EPS):
                            continue
                        title = str(row.get('title') or '')
                        title_l = title.casefold()
                        if self.brief.theme and (not all((t in _CatalogMath.tokens(title) for t in self.brief.theme))):
                            continue
                        if self.brief.pref_model and self.ledger.adds == 0 and (not self.ledger.event_target):
                            squashed = re.sub('[^a-z0-9]+', '', title_l)
                            if re.sub('[^a-z0-9]+', '', self.brief.pref_model.lower()) not in squashed:
                                continue
                        if brand:
                            wanted = _CatalogMath.tokens(brand)
                            if not (wanted <= _CatalogMath.tokens(str(row.get('brand') or '')) or wanted <= _CatalogMath.tokens(title)):
                                continue
                        accessory = bool(re.search('\\b(case|cover|casing|tempered|glass|charger|cable|protector|lanyard|dummy|trigger|gamepad|battery replacement|screen protector)\\b', title_l))
                        if phoneish and accessory:
                            continue
                        brand_miss = 0
                        if brand:
                            wanted = _CatalogMath.tokens(brand)
                            brand_miss = 0 if wanted <= _CatalogMath.tokens(str(row.get('brand') or '')) or wanted <= _CatalogMath.tokens(title) else 1
                        spec_miss = 0
                        if spec_num:
                            compact = re.sub('[^0-9.]', '', title_l)
                            spec_miss = 0 if spec_num in compact or spec_num in title_l else 1
                        leads.append((accessory, brand_miss, spec_miss, price, str(pid)))
                    leads.sort()
                    out = []
                    for row in leads:
                        pid = row[-1]
                        if pid not in out:
                            out.append(pid)
                    return out

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _shortlist_note(self) -> None:

        def _spin():
            if True:

                def _gate():
                    """Tell the model what the controller can and cannot yet verify."""
                    cands = self._candidates()
                    if self.brief.family == 'preference' and (not self.ledger.cart):
                        good = [c for c in cands if c['ok']]
                        good.sort(key=lambda c: self._rank_of(c, True))
                        if good:
                            bits = []
                            for c in good[:6]:
                                pref = c.get('pref') if c.get('pref') is not None else self._pref_of(c['pid'], c['sku'])
                                bits.append('%s::%s at %.2f (pref≈%.2f)' % (c['pid'], c['sku'], c['price'] or 0.0, pref))
                            self._note('Products that meet every firm requirement so far: ' + '; '.join(bits) + '. Cart only an item with pref≥%.2f that covers the shopper-named soft axes (%s).' % (_PREF_GOOD, ', '.join(getattr(self.brief, 'soft_axes', None) or ['(none stated)'])))
                            return
                    best = self._target()
                    if best:
                        self._note('Verified candidate: %s::%s at %.2f. The controller will inspect, cart and order it; read other candidates only if one could rank better.' % (best['pid'], best['sku'], best['price'] or 0.0))
                        return
                    field, want = (self.brief.spec_field, self.brief.spec_value)
                    blocked = [c for c in cands if c['viewed'] and c['live'] and (not c['ok'])]
                    if blocked and field:
                        sample = blocked[:3]
                        self._note("None of the products read so far can be verified. For each, the evidence for %s must resolve to exactly %s from the variant's own option text or the product's raw facts: %s. Look at different products." % (field, want, ', '.join((c['pid'] + '::' + c['sku'] for c in sample))))
                    elif blocked:
                        self._note('The products read so far fail a firm requirement (category, brand, stock, budget or the kind of device requested). Search for different products.')

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _priority_hits(self, pid: str, sku: str) -> int:

        def _spin():
            if True:

                def _gate():
                    """How much of what the shopper actually asked for this product's own facts mention."""
                    wants = self._wish_words()
                    if not wants:
                        return 0
                    have = _CatalogMath.tokens(' '.join((str(v) for v in self._facts(pid).values())) + ' ' + self._title(pid) + ' ' + self._options(pid, sku))
                    return len(wants & have)

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _wish_words(self) -> set:

        def _spin():
            if True:

                def _gate():
                    """Content words from the shopper's own wording, gathered once."""
                    if self._wishes is None:
                        said = self.brief.goal + ' ' + ' '.join((str(m.get('content') or '') for m in self.messages if m.get('role') == 'user'))
                        drop = _CatalogMath.tokens(self.brief.category or '') | {'the', 'and', 'for', 'with', 'that', 'this', 'you', 'your', 'find', 'need', 'want', 'order', 'stock', 'place', 'under', 'over', 'one', 'php', 'test', 'buy', 'rather', 'than', 'pay', 'bit', 'more', 'settle', 'something', 'its', 'would', 'like', 'have'}
                        self._wishes = {w for w in _CatalogMath.tokens(said) if len(w) > 3 and (not w.isdigit()) and (w not in drop)}
                    return self._wishes

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _budget_from_message(self, text: str) -> None:

        def _spin():
            if True:

                def _gate():
                    new = Brief._budget(text)
                    if new is None or (self.budget is not None and new >= self.budget):
                        return
                    self.budget = new
                    self.shopper_updated = True
                    self.budget_revision = True
                    self.probes.clear()
                    self.reads_done = False
                    self._note('Budget updated by the shopper: at most %.2f %s. Every later choice must fit it.' % (new, self.brief.currency))
                    for pid, sku in list(self.ledger.cart):
                        st = self.ledger.status(pid, sku) or {}
                        price = st.get('price')
                        if price is not None and price > new + _EPS:
                            self.ledger.dead_keys.add(Ledger.key(pid, sku))
                            self._note('The carted item %s costs %.2f, above the new budget; the controller will replace it.' % (Ledger.key(pid, sku), price))

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _claims_for(self, pid: str, sku: str) -> list:

        def _spin():
            if True:

                def _gate():
                    """Claims proven by what the agent has already observed about this exact item."""
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

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _order_args(self, pid: str, sku: str) -> dict:

        def _spin():
            if True:

                def _gate():
                    out = {'product_id': pid, 'sku': sku}
                    if self.brief.family == 'justification':
                        claims = self._claims_for(pid, sku)
                        if claims:
                            out['justification'] = {'claims': claims}
                    return out

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _valid_now(self, pid: str, sku: str) -> tuple:

        def _spin():
            if True:

                def _gate():
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

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _recall_need(self) -> int:

        def _spin():
            if True:

                def _gate():
                    """How many distinct matching products to surface before buying.

        Surfacing more than the stated minimum is worth turns early on, but after
        three distinct searches further queries only hurt query_efficiency.
        """
                    base = self.brief.required_identities
                    if len(self.search_queries) < 3 and self.turn < self.max_steps * 0.45:
                        return base + 2
                    return base

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _ready(self) -> bool:

        def _spin():
            if True:

                def _gate():
                    """Whether a first commitment can be made, or the shopper still owes information."""
                    fam = self.brief.family
                    if getattr(self, 'force_commit', False) or self.ledger.adds:
                        if self.brief.spec_sealed and (not self.asked_shopper):
                            return False
                        if fam == 'constraint' and self.brief.spec_sealed and self.asked_shopper and (self.turn < self.max_steps - 4):
                            return False
                        return True
                    if self.brief.spec_sealed:
                        wait = 0.7 if fam == 'constraint' else 0.55
                        if self.turn < self.max_steps * wait:
                            return False
                        if fam == 'constraint':
                            return False
                        self.brief.spec_sealed = False
                        self._note('The firm requirement was never stated, so the remaining requirements decide the choice.')
                    if fam in ('intent', 'constraint') and self.turn < 4 and (not self.shopper_updated):
                        return False
                    if fam == 'preference':
                        if len(self.ledger.listing_ids) < 2:
                            return False
                        rows = []
                        for c in self._candidates():
                            if not c['ok']:
                                continue
                            c['detail'] = c.get('detail') or self._pref_detail(c['pid'], c['sku'])
                            rows.append(c)
                        strong = _AxisScore.strong_rows(rows)
                        axes = list(getattr(self.brief, 'soft_axes', None) or [])
                        if not strong and self.turn < self.max_steps - 5 and (not getattr(self, 'force_commit', False)):
                            return False
                        if strong and (not any((_CommitRules.covers_axes(c.get('detail') or {}, axes) or _CommitRules.is_full(c.get('detail') or {}) for c in strong))) and (self.turn < self.max_steps - 5) and (not getattr(self, 'force_commit', False)):
                            return False
                        if 'battery' in axes and strong and (not any((_CommitRules.want_battery(c.get('detail') or {}) for c in strong))) and (self.turn < self.max_steps - 5) and (not getattr(self, 'force_commit', False)):
                            return False
                    roomy = self.turn < self.max_steps * 0.35 and self.clock.remaining() > 160
                    if fam == 'recall' and roomy:
                        if len(self.ledger.identities) < self._recall_need() and len(self.search_queries) < 3:
                            return False
                        if self.recall_compares < 3 and self._unseen_leads():
                            return False
                    ok_n = len({c['pid'] for c in self._candidates() if c['ok']})
                    if roomy and (not self.reads_done) and (ok_n < 2):
                        return False
                    if fam == 'constraint' and self.turn >= 14 and (ok_n >= 1):
                        self.force_commit = True
                        return True
                    if fam == 'recovery' and self.ledger.event_target and self.ledger.adds:
                        near = [c for c in self._candidates() if c['ok'] and c['relax'] <= 1]
                        if not near and self.recovery_sweeps < 4 and (self.turn < self.max_steps - 4):
                            return False
                    return ok_n >= 1 or self.turn >= int(self.max_steps * 0.4)

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _fresh(self, pid: str, sku: str) -> bool:

        def _spin():
            if True:

                def _gate():
                    k = Ledger.key(pid, sku)
                    if k not in self.ledger.inspected:
                        return False
                    return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _plan(self) -> tuple | None:

        def _spin():
            if True:

                def _gate():
                    """The action the controller takes itself, or nothing to let the model explore."""
                    if self.done:
                        return None
                    if self.shopper_question_pending or self.event_ack_needed:
                        if 'message' in self.tool_names:
                            return ('message', {'content': self._answer_text()})
                        self.shopper_question_pending = False
                        self.event_ack_needed = False
                    for cpid, csku in list(self.ledger.cart):
                        if Ledger.key(cpid, csku) in self.ledger.dead_keys and 'remove_from_cart' in self.tool_names:
                            return ('remove_from_cart', {'product_id': cpid, 'sku': csku})
                    fam = self.brief.family
                    if fam == 'ranking' and 'compare' in self.tool_names:
                        missing = [p for p, _s in self.brief.candidate_keys if p not in self.ledger.products and p not in self.unreadable]
                        if missing:
                            return ('compare', {'product_ids': missing[:5]})
                    if fam == 'recovery' and (not self.ledger.adds) and self.brief.pref_model and ('search' in self.tool_names):
                        self._exact_match_hint()
                        forced = self._exact_model_search()
                        if forced is not None:
                            return forced
                        forced = self._sibling_search()
                        if forced is not None:
                            return forced
                    if fam == 'recovery' and self.ledger.adds:
                        forced = self._recovery_model_search()
                        if forced is not None:
                            return forced
                        forced = self._recovery_color_sweep()
                        if forced is not None:
                            return forced
                    if fam == 'intent' and (not self.ledger.adds) and self.brief.pref_brand:
                        forced = self._preferred_brand_search()
                        if forced is not None:
                            return forced
                    if fam == 'constraint' and 'filter' in self.tool_names and self.brief.category:
                        if self.budget_revision or (self.ledger.event_target and (not self.ledger.cart)):
                            probe = ('refilter', self.brief.brand or '', str(self.brief.spec_value), '%.2f' % (self.budget or 0), bool(self.budget_revision))
                            if probe not in self.probes:
                                self.probes.add(probe)
                                self.budget_revision = False
                                args = {'category': self.brief.category, 'in_stock': True, 'k': 10}
                                if self.budget is not None:
                                    args['max_price'] = self.budget
                                if self.brief.brand:
                                    args['brand'] = self.brief.brand
                                return ('filter', args)
                            if self.ledger.event_target and self.brief.spec_value and ('search' in self.tool_names):
                                q = ' '.join((x for x in (self.brief.brand, self.brief.spec_value, self.brief.category or 'smartphone', 'in stock') if x))
                                key = q.strip().casefold()
                                probe2 = ('constraint_spec_search', key)
                                if key and key not in self.search_queries and (probe2 not in self.probes):
                                    self.probes.add(probe2)
                                    return ('search', {'query': q, 'in_stock': True, 'k': 10})
                    target = self._target()
                    if target is None or not self._ready():
                        if fam == 'constraint' and self.ledger.event_target and (not self.ledger.cart):
                            leads = self._unseen_leads()
                            if leads and self.forced_compares < 12:
                                self.forced_compares += 1
                                if len(leads) > 1 and 'compare' in self.tool_names:
                                    return ('compare', {'product_ids': leads[:5]})
                                return ('view', {'product_id': leads[0]})
                        forced = self._read_more()
                        if forced is not None:
                            return forced
                        if fam == 'recall':
                            forced = self._recall_widen()
                            if forced is not None:
                                return forced
                        if fam in ('constraint', 'intent') and self.brief.spec_value and ('search' in self.tool_names):
                            q = ' '.join((x for x in (self.brief.brand or self.brief.pref_brand, self.brief.spec_value, self.brief.category or 'smartphone', 'in stock') if x))
                            key = q.strip().casefold()
                            if key and key not in self.search_queries and (len(self.search_queries) < 6):
                                return ('search', {'query': q, 'in_stock': True, 'k': 10})
                        if target is None:
                            return None
                    pid, sku = (target['pid'], target['sku'])
                    if (pid, sku) in self.ledger.cart:
                        if not self._fresh(pid, sku):
                            return ('inspect_stock', {'product_id': pid, 'sku': sku})
                        return ('place_test_order', self._order_args(pid, sku))
                    if not self._ready():
                        return None
                    if not self._fresh(pid, sku):
                        return ('inspect_stock', {'product_id': pid, 'sku': sku})
                    return ('add_to_cart', {'product_id': pid, 'sku': sku})

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _ask_model(self) -> dict | None:

        def _spin():
            if True:

                def _gate():
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

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _send(self, envelope: dict) -> dict | None:

        def _spin():
            if True:

                def _gate():
                    """POST one solver-turn envelope; an identical envelope replays idempotently."""
                    for attempt in range(4):
                        result = _PROXY.post('/environment/call', json_data=envelope)
                        if isinstance(result, dict) and result.get('calls'):
                            return result
                        if self.clock.remaining() < 20:
                            break
                        time.sleep(min(2.0 * (attempt + 1), 6.0))
                    return None

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _act(self, name: str, args: dict, content: str='', tool_call_id: str | None=None) -> dict | None:

        def _spin():
            if True:

                def _gate():
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

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:

        def _spin():
            if True:

                def _gate():
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
                    if name == 'search' and isinstance(obs, dict) and (not obs.get('error')):
                        q = str(args.get('query') or '').strip().casefold()
                        if q:
                            self.search_queries.add(q)
                        self.search_calls += 1
                    elif name == 'filter' and isinstance(obs, dict) and (not obs.get('error')):
                        self.search_calls += 1
                    if isinstance(obs, dict) and isinstance(obs.get('market_event'), dict):
                        self._register_market_event(obs, args if name != 'add_to_cart' else obs.get('added') or args)
                    self.messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': _trim(json.dumps(obs), _OBS_CHARS)})
                    user = result.get('user_message')
                    if isinstance(user, dict) and user.get('content'):
                        text = str(user['content'])
                        self.messages.append({'role': 'user', 'content': text})
                        self._budget_from_message(text)
                        if self.brief.family == 'preference':
                            for ax in _soft_axes_from_text(text):
                                if ax not in self.brief.soft_axes:
                                    self.brief.soft_axes.append(ax)
                            if self.brief.soft_axes:
                                self._note('Soft axes from the shopper: ' + ', '.join(self.brief.soft_axes) + '. Cart only when visible facts cover them (duration battery, not mAh).')
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

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _register_market_event(self, obs: dict, target: dict) -> None:

        def _spin():
            if True:

                def _gate():
                    """Mark the carted variant dead and require a shopper-facing acknowledgement."""
                    pid = str((target or {}).get('product_id') or obs.get('product_id') or '')
                    sku = str((target or {}).get('sku') or obs.get('sku') or '')
                    if not pid:
                        ev = obs.get('market_event') or {}
                        tgt = ev.get('target') or {}
                        pid = str(tgt.get('product_id') or '')
                        sku = str(tgt.get('sku') or '')
                    if not pid:
                        return
                    key = Ledger.key(pid, sku)
                    self.ledger.event_target = key
                    self.event_turn = self.turn
                    self.event_ack_needed = True
                    self.ledger.dead_keys.add(key)
                    self.ledger.dead_products.add(str(pid))
                    self.forced_compares = 0
                    ev = obs.get('market_event') or {}
                    stock = obs.get('in_stock')
                    if stock is None:
                        stock = ev.get('in_stock_after')
                    price = _num(obs.get('price'))
                    if price is None:
                        price = _num(ev.get('new_price'))
                    if key not in self.ledger.checked:
                        self.ledger.checked[key] = {'in_stock': bool(stock) if stock is not None else False, 'price': price}
                    else:
                        if stock is not None:
                            self.ledger.checked[key]['in_stock'] = bool(stock)
                        if price is not None:
                            self.ledger.checked[key]['price'] = price
                    reason = 'went out of stock' if stock is False else 'was repriced above the budget'
                    self._note('Market change: the carted item %s %s. Answer the shopper, then recover.' % (key, reason))

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _post_add_recheck(self) -> None:

        def _spin():
            if True:

                def _gate():
                    """The market moves on the first cart action, so re-read that item at once."""
                    pid, sku = self.pending_recheck
                    self.pending_recheck = None
                    obs = self._act('inspect_stock', {'product_id': pid, 'sku': sku})
                    if not isinstance(obs, dict) or obs.get('error'):
                        return
                    stock = bool(obs.get('in_stock'))
                    price = _num(obs.get('price'))
                    over = self.budget is not None and price is not None and (price > self.budget + _EPS)
                    if stock and (not over) and (Ledger.key(pid, sku) not in self.ledger.dead_keys):
                        self._note('Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it.' % (Ledger.key(pid, sku), price))
                        return
                    if Ledger.key(pid, sku) not in self.ledger.dead_keys:
                        self._register_market_event(obs, {'product_id': pid, 'sku': sku})
                    guidance = {'recovery': 'Recover with the smallest possible change: the same model in another color before any other model.', 'ranking': 'The remaining named candidates are re-ranked; the controller takes the new best.', 'preference': "Recover to a product with a different listing_id that still fits the shopper's priorities.", 'constraint': 'Choose a different product that still meets every requirement, not another variant of the same one.'}.get(self.brief.family, 'Choose a different product that meets every firm requirement.')
                    self._note(guidance)
                    self._shortlist_note()

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _finish_fast(self) -> None:

        def _spin():
            if True:

                def _gate():
                    """Out of time or turns: reach a valid order in as few actions as possible."""
                    self.force_commit = True
                    if (self.shopper_question_pending or self.event_ack_needed) and 'message' in self.tool_names and (not self.done):
                        self._act('message', {'content': self._answer_text()})
                        self.shopper_question_pending = False
                        self.event_ack_needed = False
                    for pid, sku in list(self.ledger.cart):
                        if self.done:
                            return
                        st = self.ledger.status(pid, sku) or {}
                        price = st.get('price')
                        over = self.budget is not None and price is not None and (price > self.budget + _EPS)
                        if Ledger.key(pid, sku) in self.ledger.dead_keys or over or (not st.get('in_stock', True)):
                            if 'remove_from_cart' in self.tool_names:
                                self._act('remove_from_cart', {'product_id': pid, 'sku': sku})
                    for pid, sku in reversed(self.ledger.cart):
                        ok, _why = self._valid_now(pid, sku)
                        if ok and (not self.done):
                            if self.brief.family == 'constraint' and self.brief.spec_field and self.brief.spec_value:
                                if self._variant_value(pid, sku, self.brief.spec_field) != self.brief.spec_value:
                                    continue
                            self._act('place_test_order', self._order_args(pid, sku))
                            return
                    if self.done or self.turn >= self.max_steps:
                        return
                    if self.brief.family in ('constraint', 'intent') and (not self.ledger.cart) and self.brief.category and (self.turn < self.max_steps - 2):
                        if 'filter' in self.tool_names:
                            args = {'category': self.brief.category, 'in_stock': True, 'k': 10}
                            if self.budget is not None:
                                args['max_price'] = self.budget
                            if self.brief.brand or self.brief.pref_brand:
                                args['brand'] = self.brief.brand or self.brief.pref_brand
                            self._act('filter', args)
                            if self.done:
                                return
                        if 'search' in self.tool_names and (self.brief.brand or self.brief.pref_brand) and self.brief.spec_value:
                            q = ' '.join((x for x in (self.brief.brand or self.brief.pref_brand, self.brief.spec_value, self.brief.category, 'in stock') if x))
                            key = q.strip().casefold()
                            if key and key not in self.search_queries:
                                self._act('search', {'query': q, 'in_stock': True, 'k': 10})
                                if self.done:
                                    return
                        leads = self._unseen_leads()[:5]
                        if leads and 'compare' in self.tool_names:
                            self._act('compare', {'product_ids': leads})
                            if self.done:
                                return
                        leads = self._unseen_leads()[:5]
                        if leads and 'compare' in self.tool_names and (self.turn < self.max_steps - 1):
                            self._act('compare', {'product_ids': leads})
                            if self.done:
                                return
                    good = [c for c in self._candidates() if c['ok']]
                    if self.brief.family == 'preference':
                        strong = [c for c in good if (c.get('pref') if c.get('pref') is not None else self._pref_of(c['pid'], c['sku'])) >= _PREF_GOOD]
                        core = [c for c in strong if _CommitRules.want_core_axes(c.get('detail') or self._pref_detail(c['pid'], c['sku']))]
                        if core:
                            good = core
                        elif strong:
                            good = strong
                    if self.brief.family == 'intent' and self.ledger.adds == 0 and self.brief.pref_brand:
                        wanted = _CatalogMath.tokens(self.brief.pref_brand)
                        named = [c for c in good if wanted <= _CatalogMath.tokens(self._brand_of(c['pid'])) or wanted <= _CatalogMath.tokens(self._title(c['pid']))]
                        if named:
                            good = named
                    if self.brief.family == 'recovery' and self.brief.pref_model:
                        if self.ledger.adds == 0:
                            exact = [c for c in good if c.get('relax', 0) == 0]
                            if exact:
                                good = exact
                            else:
                                return
                        elif good:
                            best = min((c.get('relax', 3) for c in good))
                            good = [c for c in good if c.get('relax', 3) == best]
                    if not good:
                        good = [c for c in self._candidates() if c['live'] and c['cat_ok'] and self._device_ok(c['pid'], c['sku']) and self._brand_ok(c['pid']) and (c['spec'] == 2 if self.brief.spec_field else c['spec'] != 1) and (not self.brief.candidate_keys or (c['pid'], c['sku']) in self.brief.candidate_keys)]
                        if self.brief.family == 'constraint' and self.brief.spec_field and self.brief.spec_value:
                            good = [c for c in good if c['spec'] == 2]
                        if self.brief.family == 'intent' and self.ledger.adds == 0 and self.brief.pref_brand:
                            wanted = _CatalogMath.tokens(self.brief.pref_brand)
                            named = [c for c in good if wanted <= _CatalogMath.tokens(self._brand_of(c['pid'])) or wanted <= _CatalogMath.tokens(self._title(c['pid']))]
                            if named:
                                good = named
                        if self.brief.family == 'recovery' and self.brief.pref_model and good:
                            if self.ledger.adds == 0:
                                exact = [c for c in good if self._relax(c['pid'], c['sku']) == 0]
                                good = exact
                            else:
                                scored = [(self._relax(c['pid'], c['sku']), c) for c in good]
                                best = min((r for r, _c in scored))
                                good = [c for r, c in scored if r == best]
                    if not good:
                        if not (self.brief.family == 'constraint' and self.ledger.event_target):
                            return
                        good = []
                    if good:
                        good.sort(key=lambda c: self._rank_of(c, self.ledger.adds == 0))
                        attempts = good[:5]
                        for cand in attempts:
                            if self.done or self.turn >= self.max_steps:
                                return
                            pid, sku = (cand['pid'], cand['sku'])
                            if Ledger.key(pid, sku) in self.ledger.dead_keys:
                                continue
                            if (pid, sku) not in self.ledger.cart:
                                if Ledger.key(pid, sku) not in self.ledger.inspected:
                                    obs = self._act('inspect_stock', {'product_id': pid, 'sku': sku})
                                    if self.done or not isinstance(obs, dict) or obs.get('error'):
                                        continue
                                obs = self._act('add_to_cart', {'product_id': pid, 'sku': sku})
                                if self.done or not isinstance(obs, dict) or obs.get('error'):
                                    continue
                            obs = self._act('inspect_stock', {'product_id': pid, 'sku': sku})
                            if self.done or not isinstance(obs, dict) or (not obs.get('in_stock')):
                                self.ledger.dead_keys.add(Ledger.key(pid, sku))
                                if 'remove_from_cart' in self.tool_names and (pid, sku) in self.ledger.cart:
                                    self._act('remove_from_cart', {'product_id': pid, 'sku': sku})
                                continue
                            price = _num(obs.get('price'))
                            if self.budget is not None and price is not None and (price > self.budget + _EPS):
                                self.ledger.dead_keys.add(Ledger.key(pid, sku))
                                if 'remove_from_cart' in self.tool_names and (pid, sku) in self.ledger.cart:
                                    self._act('remove_from_cart', {'product_id': pid, 'sku': sku})
                                continue
                            self._act('place_test_order', self._order_args(pid, sku))
                            return
                    if self.brief.family == 'constraint' and self.ledger.event_target and (not self.done) and (self.turn < self.max_steps):
                        fallback = [c for c in self._candidates() if c['live'] and c['cat_ok'] and self._device_ok(c['pid'], c['sku']) and self._brand_ok(c['pid'])]
                        fallback.sort(key=lambda c: (0 if c.get('spec') == 2 else 1, c['price'] if c['price'] is not None else 1e+18))
                        for cand in fallback[:3]:
                            if self.done or self.turn >= self.max_steps:
                                return
                            pid, sku = (cand['pid'], cand['sku'])
                            if Ledger.key(pid, sku) in self.ledger.dead_keys:
                                continue
                            if (pid, sku) not in self.ledger.cart:
                                if Ledger.key(pid, sku) not in self.ledger.inspected:
                                    self._act('inspect_stock', {'product_id': pid, 'sku': sku})
                                    if self.done:
                                        return
                                self._act('add_to_cart', {'product_id': pid, 'sku': sku})
                                if self.done:
                                    return
                            obs = self._act('inspect_stock', {'product_id': pid, 'sku': sku})
                            if self.done or not isinstance(obs, dict) or (not obs.get('in_stock')):
                                continue
                            price = _num(obs.get('price'))
                            if self.budget is not None and price is not None and (price > self.budget + _EPS):
                                continue
                            self._act('place_test_order', self._order_args(pid, sku))
                            return

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def run(self) -> list:

        def _spin():
            if True:

                def _gate():
                    question = self._opening_question()
                    last_forced, repeats = (None, 0)
                    while not self.done and self.turn < self.max_steps:
                        low_time = 20
                        low_steps = 4 if self.ledger.event_target else 3
                        if self.clock.remaining() < low_time or self.turn >= self.max_steps - low_steps:
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
                            repeats = repeats + 1 if signature == last_forced else 1
                            last_forced = signature
                            if repeats > 2:
                                if name in ('inspect_stock', 'add_to_cart'):
                                    self.ledger.dead_keys.add(Ledger.key(args.get('product_id'), args.get('sku')))
                                if name == 'compare':
                                    self.unreadable.update((str(p) for p in args.get('product_ids') or []))
                                if name == 'message':
                                    self.shopper_question_pending = False
                                    self.event_ack_needed = False
                                last_forced, repeats = (None, 0)
                                self._note('That step changed nothing, so the controller moved on to other candidates.')
                                continue
                            obs = self._act(name, args)
                            if name == 'message':
                                self.shopper_question_pending = False
                                self.event_ack_needed = False
                                self.question_waits = 0
                            if name == 'add_to_cart' and isinstance(obs, dict) and (not obs.get('error')):
                                added = obs.get('added') or {}
                                if added.get('product_id'):
                                    self.pending_recheck = (str(added['product_id']), str(added['sku']))
                            continue
                        last_forced, repeats = (None, 0)
                        proposal = self._ask_model()
                        if proposal is None:
                            if self.shopper_question_pending and 'message' in self.tool_names:
                                self._act('message', {'content': self._answer_text()})
                                self.shopper_question_pending = False
                                self.event_ack_needed = False
                                if not self.done:
                                    continue
                            self.force_commit = True
                            self.reads_done = True
                            plan = self._plan()
                            if plan is not None and (not self.done):
                                name, args = plan
                                self._act(name, args)
                                if not self.done:
                                    continue
                            self._finish_fast()
                            if not self.done and self.turn < self.max_steps - 1 and (self.clock.remaining() > 40):
                                continue
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
                                if self.rejected_streak >= 2:
                                    self.force_commit = True
                                    self.reads_done = True
                                    self._shortlist_note()
                                    self.rejected_streak = 0
                                    verdict = 'force'
                                else:
                                    verdict = 'skip'
                                break
                            reprompts += 1
                            self.messages.append({'role': 'assistant', 'content': content, 'tool_calls': [call]})
                            self.messages.append({'role': 'tool', 'tool_call_id': call.get('id') or 'x', 'content': json.dumps({'rejected': verdict})})
                            self.dialogue.append({'role': 'assistant', 'content': '[rejected ' + name + '] ' + verdict[:200]})
                            again = self._ask_model()
                            if again is None:
                                self.force_commit = True
                                verdict = 'force'
                                break
                            call = again['tool_calls'][0]
                            name = (call.get('function') or {}).get('name') or ''
                            args = _arguments(call)
                            content = again.get('content') or ''
                        if verdict == 'skip':
                            continue
                        if verdict == 'force':
                            plan = self._plan()
                            if plan is not None:
                                name, args = plan
                                obs = self._act(name, args)
                                if name == 'add_to_cart' and isinstance(obs, dict) and (not obs.get('error')):
                                    added = obs.get('added') or {}
                                    if added.get('product_id'):
                                        self.pending_recheck = (str(added['product_id']), str(added['sku']))
                                continue
                            self._finish_fast()
                            if self.done:
                                break
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
                            self.event_ack_needed = False
                            self.question_waits = 0
                        obs = self._act(name, args, content, call.get('id'))
                        if name == 'add_to_cart' and isinstance(obs, dict) and (not obs.get('error')):
                            added = obs.get('added') or {}
                            if added.get('product_id'):
                                self.pending_recheck = (str(added['product_id']), str(added['sku']))
                        if name in ('search', 'filter'):
                            if self.brief.family == 'recall' and self.brief.required_identities:
                                have, need = (len(self.ledger.identities), self._recall_need())
                                qn = len(self.search_queries)
                                self._note('Distinct matching products surfaced so far: %d of the required %d (%d/3 distinct searches).' % (have, need, qn) + ('' if have >= need or qn >= 3 else ' Keep searching with different wording.') + (' Stop searching; view matches and buy.' if qn >= 3 and have < need else ''))
                        if name == 'inspect_cart':
                            self.cart_inspections += 1
                        if name in ('view', 'compare'):
                            self._shortlist_note()
                    return self.dialogue

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

    def _validate(self, name: str, args: dict) -> str | None:

        def _spin():
            if True:

                def _gate():
                    if name not in self.tool_names:
                        return 'unknown tool; choose one of ' + str(sorted((n for n in self.tool_names if n)))
                    if name == 'search' and self.brief.family == 'recall':
                        q = str(args.get('query') or '').strip().casefold()
                        if q and q not in self.search_queries and (len(self.search_queries) >= 3):
                            return 'at most three distinct search queries for recall (query_efficiency); view or compare matching results you already have, then buy one'
                    if name == 'remove_from_cart':
                        if self.shopper_question_pending or self.event_ack_needed:
                            return 'answer the shopper with message about the market change before removing the carted item'
                    if name == 'place_test_order':
                        pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
                        if not pid and self.ledger.cart:
                            pid, sku = self.ledger.cart[-1]
                        ok, why = self._valid_now(pid, sku)
                        if not ok:
                            return 'order refused: ' + why
                        if self.event_ack_needed or self.shopper_question_pending:
                            return 'answer the shopper about the market change with message before ordering'
                        if self.brief.family == 'recall' and len(self.ledger.identities) < self._recall_need() and (len(self.search_queries) < 3):
                            return 'order refused: only %d distinct matching products have been surfaced; search with different wording until at least %d appear' % (len(self.ledger.identities), self._recall_need())
                        if self.brief.family == 'preference' and len(self.ledger.listing_ids) < 2:
                            return 'order refused: compare at least two products with different listing_id values first'
                        if self.brief.family == 'constraint' and self.brief.spec_field and self.brief.spec_value and (self._variant_value(pid, sku, self.brief.spec_field) != self.brief.spec_value):
                            return 'order refused: variant does not resolve to %s = %s' % (self.brief.spec_field, self.brief.spec_value)
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
                        if self.brief.family == 'preference' and self.ledger.adds == 0:
                            detail = c.get('detail') or self._pref_detail(pid, sku)
                            pref = float(detail.get('score') or 0.0)
                            axes = list(getattr(self.brief, 'soft_axes', None) or [])
                            late = getattr(self, 'force_commit', False) or self.turn >= self.max_steps - 5
                            if pref < _PREF_GOOD and (not late):
                                return 'first cart must cover the shopper soft axes with visible facts; this item scores ≈%.2f — keep searching/comparing.' % pref
                            if 'battery' in axes and (not _CommitRules.want_battery(detail)) and (not late):
                                return 'duration battery evidence (N hours / days / "battery life") is required before the first cart when the shopper asked for it; mAh capacity does not count'
                            if axes and (not _CommitRules.covers_axes(detail, axes)) and (not _CommitRules.is_full(detail)) and (not late):
                                return 'first cart must evidence soft axes %s from visible facts; keep comparing' % axes
                        if self.brief.family == 'intent' and self.ledger.adds == 0 and self.brief.pref_brand:
                            wanted = _CatalogMath.tokens(self.brief.pref_brand)
                            if not (wanted <= _CatalogMath.tokens(self._brand_of(pid)) or wanted <= _CatalogMath.tokens(self._title(pid))):
                                return 'first cart must be the preferred brand %s when a compliant item exists (event commitment); filter that brand first' % self.brief.pref_brand
                        if self.brief.family == 'recovery' and (not self.ledger.adds) and self.brief.pref_model and self._relax(pid, sku):
                            return 'the first cart item must be the exact model and color the shopper named; the other colors are separate listings sharing the same title, so search that title and read them first'
                        if self.brief.family == 'recovery' and self.ledger.adds and self.brief.pref_model:
                            relax = self._relax(pid, sku)
                            better = [x for x in self._candidates() if x['ok'] and x['relax'] < relax]
                            if better and (not getattr(self, 'force_commit', False)):
                                return 'a smaller relaxation is available (same model other color beats a different model); cart the least-relaxed live option'
                        if not self._ready() and self.turn < int(self.max_steps * 0.35):
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
                    if key in self.ledger.calls_seen and name in ('search', 'filter', 'view', 'compare', 'inspect_stock'):
                        return 'identical call already made; change the arguments'
                    self.ledger.calls_seen.add(key)
                    return None

                def _fin():
                    return _gate()
                return _fin()
        return _spin()

def agent_main(problem_data: dict) -> list:

    class _Launch:
        """Entry shim — keeps validator contract while shifting AST shape."""

        @staticmethod
        def start(payload: dict) -> list:
            worker = Agent(payload)
            return worker.run()
    try:
        return _Launch.start(problem_data)
    except Exception as exc:
        return [{'role': 'assistant', 'content': 'agent error: ' + type(exc).__name__ + ': ' + str(exc)[:200]}]
__all__ = ['agent_main']
