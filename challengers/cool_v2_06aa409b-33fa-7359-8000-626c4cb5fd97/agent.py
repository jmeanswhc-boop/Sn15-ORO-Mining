"""EnvPack shopping pilot for ORO-Bench.

A model proposes exploratory tool calls; a deterministic pilot owns every
graded commitment (inspect → cart → recheck → recover → order). Evidence is
read the same way the verifier reads it: labelled facts, then selected option
text, then title.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
_SIZE_IN_RE = re.compile('(\\d+(?:\\.\\d+)?)\\s*[-–]?\\s*(?:(?:inch(?:es)?|in)\\b|\'{2}|[\\"“”])', re.I)
_RATE_HZ_RE = re.compile('(\\d+(?:\\.\\d+)?)\\s*hz', re.I)
_PIXEL_PAIR_RE = re.compile('(\\d{3,5})\\s*[x×]\\s*(\\d{3,5})')
_CAPACITY_RE = re.compile('(\\d+(?:\\.\\d+)?)\\s*(tb|gb)\\b', re.I)
_COMBO_MEM_RE = re.compile('(?<!\\d)(\\d+(?:\\.\\d+)?)\\s*(?:g(?:b)?)?\\s*\\+\\s*(\\d+(?:\\.\\d+)?)\\s*g(?:b)?\\b', re.I)
_COMBO_HINT_RE = re.compile('(?<!\\d)\\d+(?:\\.\\d+)?\\s*(?:g(?:b)?)?\\s*\\+\\s*\\d+(?:\\.\\d+)?(?:\\s*g(?:b)?)?(?!\\d)', re.I)
_RAM_LABEL_RE = re.compile('(\\d+(?:\\.\\d+)?)\\s*gb\\s*(?:ram|memory|unified memory)\\b', re.I)
_QHD_RE = re.compile('\\bqHD\\b')
_EVIDENCE_LABEL = {'screen_size': 'size', 'resolution': 'display', 'refresh_rate': 'refresh_rate', 'ram': 'storage'}
_PLATFORM_RE = re.compile('\\b(?:android(?:\\s*\\d+(?:\\.\\d+)*)?|ios|blackberry(?:\\s*os)?|windows\\s*phone|symbian|harmony\\s*os|hyper\\s*os|sailfish|ubuntu\\s*touch|tizen)\\b', re.I)
_PHONE_YES_RE = re.compile('\\bsmart\\s*phones?\\b', re.I)
_PHONE_NO_RE = re.compile('\\btablets?\\b|\\b(?:feature|basic)\\s*phones?\\b|\\b(?:mystery|blind|lucky)\\s*box(?:es)?\\b|\\b(?:accessor(?:y|ies)|screen\\s*protectors?|phone\\s*cases?|case\\s*(?:cover\\s*)?for|replacement\\s*(?:screen|battery)|smart\\s*watches?|earbuds?|headphones?)\\b', re.I)
_ACCESSORY_RE = re.compile('\\bonly\\s+(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\b|\\b(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\s+only\\b|^\\s*replacement\\s+(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\b', re.I)
_DENY_RE = re.compile('\\bnot\\s+(?:an?\\s+)?(?:smart\\s*phones?|feature\\s*phones?|basic\\s*phones?|tablets?|accessor(?:y|ies))\\b', re.I)
_KIT_RE = re.compile('\\b(?:(?:with\\s+(?:free|bundled)|bundled\\s+with|free)\\s+|(?:with|includes?)\\s+(?:an?\\s+)?(?=(?:headphones?|earbuds?|phone\\s*case|screen\\s*protector|charger|accessor(?:y|ies))\\b))', re.I)
_FAMILY_HINTS = (('preference', 'listing_id'), ('constraint', 'exact variant'), ('justification', 'trustworthy reason'), ('intent', 'nice-to-haves'))
_FAMILY_COACHING = {'intent': "Firm requirements: the category, the named hardware specification, the budget and stock. A named brand is only a preference: choose it when a fully compliant item of that brand exists, otherwise any compliant brand. Use filter(category, brand, max_price, in_stock=true), then view or compare to read the exact variant's specification from raw facts. Answer the shopper with message; a budget the shopper lowers later applies to every later choice.", 'recall': 'Run several differently worded searches (k=10, in_stock=true) built from the requested words and close synonyms until enough distinct matching products have appeared; every requested word must appear as a whole word of the title. Then view the strongest candidates so the controller can verify one and buy it.', 'constraint': "The shopper hides one firm hardware requirement and answers it when asked; obey it exactly for the specific variant, reading the variant's own option text before any product-wide value. Use filter(category, brand, max_price, in_stock=true) and view candidates. If the carted variant sells out, move to a different product that still meets every requirement.", 'preference': 'You already asked which preference matters most; use the answer. Search or filter the category within budget, then compare candidates with different listing_id values, reading raw facts. The first cart item is graded, so make it the best genuine match. If it is repriced above budget, recover to a different listing_id.', 'ranking': "Only the named candidate variants count, and the controller ranks them for you: compare all of them first so their facts are on the record. Highest advertised maximum refresh rate wins, then the largest screen, then the lowest price, reading the selected variant's own value before a product-wide one. After the market moves, the remaining candidates are re-ranked.", 'recovery': 'Find the preferred model and color first: search the model code, then view it so every variant and color is on the record. The first cart item must be exactly that model and color. If it disappears, relax as little as possible - the same model in another color before any other model - and never relax the category, budget or stock.', 'justification': 'Find a product whose own raw facts state the requested specification, in the requested category and brand, in stock and within budget. View or compare it so its facts are on the record, then the controller inspects it, carts it and orders it with claims taken only from that evidence.', 'generic': 'Understand the firm requirements, filter or search the catalog, view candidates, and let the controller verify and buy the best one.'}
_BASE_POLICY = 'You are a careful shopping agent working a catalog through tools. Exactly one tool call per turn, at most 30 turns, so be efficient. A deterministic controller checks every proposal and takes over the cart and the order once a candidate is fully verified, so your job is to surface and read candidates: search or filter, then view or compare so their raw facts, variant options and prices are on record. Never add an item to the cart as a placeholder - the first cart addition is graded. Prices and stock change while you shop; the controller re-checks the carted item and tells you when to move to a different product. Answer every direct question from the shopper with one short message before anything else, and never send progress or status updates. Never ask the shopper to raise the budget. Do not repeat a tool call with identical arguments; one compare of up to five products beats several view calls.'
_INFER_ROSTER = {'chutes': ('deepseek-ai/DeepSeek-V3.2-TEE', 'Qwen/Qwen3-32B-TEE'), 'openrouter': ('deepseek/deepseek-v3.2', 'qwen/qwen3-32b')}
_GATEWAY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_EPISODE_SECS = 270.0
_OBS_LIMIT = 7000
_REJECT_RETRIES = 2
_PRICE_EPS = 1e-09
_AMOUNT_PAT = '(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?'
_CAP_RE = re.compile('(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|within|budget (?:is|of)|only go up to|can spend up to|at most)\\s*' + _AMOUNT_PAT + '\\s*(PHP|php|₱|pesos?)?', re.I)
_VARIANT_RE = re.compile('(\\d{6,})::([A-Za-z0-9._\\-]+)')
_DQUOTE_RE = re.compile('\\"([^\\"]{2,80})\\"')
_MATCH_PHRASE_RE = re.compile('matches (?:the phrase )?\\"?([A-Za-z0-9][^\\",]{1,80}?)\\"?\\s*,', re.I)
_MIN_DISTINCT_RE = re.compile('surface at least (\\d+) distinct', re.I)
_FAVOR_ITEM_RE = re.compile('prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\\-]*\\d[A-Za-z0-9\\-]*)\\b', re.I)
_FAVOR_BRAND_RE = re.compile("\\bprefer(?:ably)?\\s+(?:the\\s+)?([A-Za-z][\\w&.'\\- ]{1,38}?)\\s*(?:,|\\.|;|\\bbut\\b|\\bwhen\\b|\\bif\\b|$)")
_CATEGORY_RE = re.compile("in the ([A-Za-z][\\w&'\\- ]{1,48}?) category", re.I)
_SPEC_NOISE_RE = re.compile('\\b(?:full|ultra)\\s+hd\\b|\\b(?:hd|fhd|qhd|uhd|2k|4k|\\d+p)\\b|\\d+(?:\\.\\d+)?\\s*-?\\s*(?:inch(?:es)?|in|hz|gb)\\b', re.I)
_JUSTIFY_RE = re.compile('\\b(?:why|reason|explain|justif)', re.I)
_PROOF_FIELDS = ('resolution', 'refresh_rate', 'screen_size', 'ram', 'price', 'in_stock', 'brand')
_ASK_RE = re.compile('\\?|\\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)|tell me (?:which|whether|if)|confirm whether)\\b', re.I)

def _field_mentioned(field: str, text: str) -> bool:
    """Whether an option string talks about the concept at all (even ambiguously)."""
    text = text or ''
    if field == 'ram':
        return bool(_COMBO_HINT_RE.search(text) or _RAM_LABEL_RE.search(text))
    if field == 'screen_size':
        return bool(_SIZE_IN_RE.search(text))
    if field == 'refresh_rate':
        return bool(_RATE_HZ_RE.search(text))
    return bool(_PIXEL_PAIR_RE.search(text) or re.search('\\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\\b', text, re.I))

def _target_from_text(field: str | None, text: str) -> str | None:
    """The single value a request states for a concept, or nothing when unclear."""
    if not field:
        return None
    return _unique_or_none(_EXTRACTORS[field](text or ''))

def _capacity_from_text(text: str) -> set:
    return {'%ggb' % (float(m.group(1)) * (1024 if m.group(2).lower() == 'tb' else 1)) for m in _CAPACITY_RE.finditer(text or '')}

def _nearby_values(field: str, text: str) -> set:
    """Values a free-form string asserts, requiring the number's own concept word.

    Titles and variant option strings mix several capacities, so a memory figure
    only counts when the word for it follows the number; storage totals and a
    graphics chip's own memory must never be read as system memory.
    """
    text = text or ''
    if field != 'ram':
        return _EXTRACTORS[field](text)
    paired = {'%ggb' % float(a) for a, _b in _COMBO_MEM_RE.findall(text)}
    return paired or {'%ggb' % float(n) for n in _RAM_LABEL_RE.findall(text)}

def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(',', ''))
    except (TypeError, ValueError):
        return None

def _normalize_identity(text: str, title: bool=False) -> str:
    """Strip claims that describe what a listing is NOT, or what ships beside it."""
    text = _DENY_RE.sub('', text or '')
    return _KIT_RE.split(text, maxsplit=1)[0] if title else text

def _phone_signature(facts: dict, title: str, options: str) -> str:
    """'yes' / 'no' / 'unclear' for whether public evidence makes this a real handset."""
    if _ACCESSORY_RE.search(options or ''):
        return 'no'
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or '')
    head = _normalize_identity(raw_title, title=True)
    kind = ' '.join((_normalize_identity(str(facts.get(k) or '')) for k in ('phone_type', 'device_type', 'description_device_type')))
    blob = head + ' ' + kind
    if _PHONE_NO_RE.search(blob):
        return 'no'
    if any(('[truncated]' in str(facts.get(k) or '') for k in ('phone_type', 'device_type', 'device_operating_system'))):
        return 'unclear'
    if _PHONE_YES_RE.search(blob):
        return 'yes'
    os_blob = ' '.join((_normalize_identity(str(facts.get(k) or '')) for k in ('device_operating_system', 'description_operating_system') if not re.search('\\bcompatib', str(facts.get(k) or ''), re.I)))
    if _PLATFORM_RE.search(os_blob):
        return 'yes'
    if _PLATFORM_RE.search(head) and (not re.search('\\bcompatib', raw_title, re.I)):
        return 'yes'
    return 'unclear'

def _words_match(a: str, b: str) -> bool:
    """Loose noun equality: ignores case, spacing and a trailing plural."""
    x = re.sub('[^a-z0-9]+', '', (a or '').lower()).rstrip('s')
    y = re.sub('[^a-z0-9]+', '', (b or '').lower()).rstrip('s')
    return bool(x) and x == y

def _clip_obs(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit - 40] + ' ... (truncated by agent)"}'

def _parse_tool_args(tool_call: dict) -> dict:
    raw = (tool_call.get('function') or {}).get('arguments', '{}')
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or '{}')
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}

def _unique_or_none(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None

def _res_from_text(text: str) -> set:
    """Display classes asserted by a text: pixel pairs and the usual shorthands."""
    text = text or ''
    if _QHD_RE.search(text):
        return set()
    low = text.lower()
    out = set()
    for m in _PIXEL_PAIR_RE.finditer(low):
        pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        cls = _res_bucket(pair[0], pair[1])
        if cls:
            out.add(cls)
    if re.search('\\b(?:4k|uhd|ultra hd|2160p?)\\b', low):
        out.add('4k')
    if re.search('\\b(?:qhd|1440p?|2k)\\b', low):
        out.add('qhd')
    if re.search('\\b(?:fhd|1080p?|full ?hd)\\b', low):
        out.add('fhd')
    if re.search('\\b(?:hd|720p?)\\b', re.sub('\\b(?:full|ultra) hd\\b', '', low)):
        out.add('hd')
    return out

def _rates_from_text(text: str) -> set:
    return {'%ghz' % float(m.group(1)) for m in _RATE_HZ_RE.finditer(text or '')}

def _sizes_from_text(text: str) -> set:
    return {'%gin' % float(m.group(1)) for m in _SIZE_IN_RE.finditer(text or '')}

def _res_bucket(long_side: int, short_side: int) -> str | None:
    if long_side >= 3840 and short_side >= 2160:
        return '4k'
    if long_side >= 2560 and short_side >= 1440:
        return 'qhd'
    if long_side >= 1920 and short_side >= 1080:
        return 'fhd'
    if long_side >= 1280 and short_side >= 720:
        return 'hd'
    return None

def _word_set(text: str) -> set:
    return set(re.findall('[a-z0-9]+', (text or '').lower()))
_EXTRACTORS = {'screen_size': _sizes_from_text, 'refresh_rate': _rates_from_text, 'resolution': _res_from_text, 'ram': _capacity_from_text}

class EvidenceBook:
    """What the agent has observed for itself, keyed by product and variant."""

    def __init__(self) -> None:
        self.cart: list = []
        self.checked: dict = {}
        self.stock_seq: dict = {}
        self.inspected_keys: set = set()
        self.products: dict = {}
        self.rows: dict = {}
        self.options: dict = {}
        self.listing_set: set = set()
        self.listing_by_pid: dict = {}
        self.identity_set: set = set()
        self.seq = 0
        self.first_cart_seq: int | None = None
        self.last_cart_seq: int | None = None
        self.adds = 0
        self.shock_key: str | None = None
        self.killed_keys: set = set()
        self.killed_pids: set = set()
        self.seen_calls: set = set()

    def status(self, pid: str, sku: str) -> dict | None:
        return self.checked.get(self.variant_key(pid, sku))

    @staticmethod
    def variant_key(pid: Any, sku: Any) -> str:
        return str(pid) + '::' + str(sku or '')

    def ingest(self, name: str, args: dict, obs: dict, brief: ShopperBrief) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get('error'):
            return
        if name in ('search', 'filter'):
            for row in obs.get('results') or []:
                if isinstance(row, dict):
                    self._capture_row(row, brief)
        elif name == 'view':
            self._capture_product(obs, brief)
        elif name == 'compare':
            for prod in obs.get('compared') or []:
                if isinstance(prod, dict):
                    self._capture_product(prod, brief)
        elif name == 'inspect_stock':
            k = self.variant_key(obs.get('product_id'), obs.get('sku'))
            self.checked[k] = {'in_stock': bool(obs.get('in_stock')), 'price': _as_float(obs.get('price'))}
            self.stock_seq[k] = self.seq
            self.inspected_keys.add(k)
        elif name == 'add_to_cart':
            added = obs.get('added') or {}
            pid, sku = (added.get('product_id'), added.get('sku'))
            if pid and sku:
                self.adds += 1
                if (pid, sku) not in self.cart:
                    self.cart.append((pid, sku))
                if self.first_cart_seq is None:
                    self.first_cart_seq = self.seq
                self.last_cart_seq = self.seq
            self._refresh_cart(obs)
        elif name == 'remove_from_cart':
            self._refresh_cart(obs)
            removed = obs.get('removed') or {}
            pair = (removed.get('product_id'), removed.get('sku'))
            if pair in self.cart:
                self.cart.remove(pair)
        elif name == 'inspect_cart':
            self._refresh_cart(obs)

    def _capture_row(self, row: dict, brief: ShopperBrief) -> None:
        pid = row.get('product_id')
        if not pid:
            return
        title = str(row.get('title') or '')
        if brief.theme:
            if all((t in _word_set(title) for t in brief.theme)):
                self.identity_set.add(str(pid))
        else:
            self.identity_set.add(str(pid))
        if not row.get('sku'):
            return
        k = self.variant_key(pid, row.get('sku'))
        self.rows[k] = row
        if row.get('price') is not None:
            self.checked.setdefault(k, {'in_stock': bool(row.get('in_stock', True)), 'price': _as_float(row.get('price'))})

    def _refresh_cart(self, obs: dict) -> None:
        rows = obs.get('cart')
        if not isinstance(rows, list):
            return
        pairs = [(r.get('product_id'), r.get('sku')) for r in rows if isinstance(r, dict) and r.get('product_id')]
        if pairs or rows == []:
            self.cart = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get('product_id'):
                continue
            k = self.variant_key(r.get('product_id'), r.get('sku'))
            price = _as_float(r.get('current_price'))
            stock = r.get('current_in_stock')
            if price is None and stock is None:
                continue
            prev = self.checked.get(k, {})
            self.checked[k] = {'in_stock': bool(stock) if stock is not None else prev.get('in_stock', True), 'price': price if price is not None else prev.get('price')}
            self.stock_seq[k] = self.seq

    def _capture_product(self, prod: dict, brief: ShopperBrief) -> None:
        pid = prod.get('product_id')
        if not pid:
            return
        self.products[str(pid)] = prod
        if prod.get('listing_id'):
            self.listing_set.add(str(prod['listing_id']))
            self.listing_by_pid[str(pid)] = str(prod['listing_id'])
        if brief.theme and all((t in _word_set(str(prod.get('title') or '')) for t in brief.theme)):
            self.identity_set.add(str(pid))
        for v in prod.get('variants') or []:
            if not isinstance(v, dict) or not v.get('sku'):
                continue
            k = self.variant_key(pid, v.get('sku'))
            self.checked[k] = {'in_stock': bool(v.get('in_stock')), 'price': _as_float(v.get('price'))}
            self.stock_seq[k] = self.seq
            self.options[k] = str(v.get('options') or '')

class Deadline:

    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def secs_left(self) -> float:
        return self.budget - (time.monotonic() - self.start)

class SessionPilot:

    def __init__(self, problem: dict) -> None:
        env = problem['environment']
        self.binding = dict(env['binding'])
        self.policy = env['policy_view']
        self.tools = list(self.policy.get('tools') or [])
        self.allowed_tools = {(t.get('function') or {}).get('name') for t in self.tools if isinstance(t, dict)}
        self.step_cap = int(self.policy.get('max_steps') or 30)
        self.calls_per_turn = max(1, int(self.policy.get('max_calls_per_turn') or 1))
        self.episode_id = str(problem.get('problem_id') or problem.get('id') or 'problem')
        self.briefing = ShopperBrief(str(self.policy.get('query') or ''))
        self.budget = self.briefing.budget
        self.book = EvidenceBook()
        self.deadline = Deadline(_EPISODE_SECS)
        self.transcript: list = []
        self.chat: list = [{'role': 'system', 'content': self._policy_text()}, {'role': 'user', 'content': self.policy.get('query') or ''}]
        self.turn = 0
        self.asked_once = False
        self.awaiting_reply = False
        self.brief_updated = False
        self.veto_streak = 0
        self.held_envelope: dict | None = None
        self.search_count = 0
        self.cart_reads = 0
        self.recheck_pair: tuple | None = None
        self.compare_boosts = 0
        self.nudged_exact = False
        self.tried: dict = {}
        self.ask_count = 0
        self.did_sibling_search = False
        self.nudged_brand = False
        self._wish_set: set | None = None
        self.probes: set = set()
        self.did_twin_search = False
        self.pages_done = False
        self.shock_step: int | None = None
        self.blocked_ids: set = set()
        self.wait_ticks = 0
        self.done = False
        self.roster_i = 0
        provider = (getenv('INFERENCE_PROVIDER') or '').strip().lower()
        self.model_roster = _INFER_ROSTER.get(provider) or _INFER_ROSTER['chutes']
        pinned = getenv('SANDBOX_MODEL')
        if pinned:
            self.model_roster = (pinned,) + tuple((m for m in self.model_roster if m != pinned))

    def _score_variant(self, pid: str, sku: str) -> dict:
        """Score one variant against every gate the runtime will apply to it."""
        fam = self.briefing.family
        variant_key = EvidenceBook.variant_key(pid, sku)
        st = self.book.checked.get(variant_key) or {}
        price, stock = (st.get('price'), st.get('in_stock'))
        dead = variant_key in self.book.killed_keys or (str(pid) in self.book.killed_pids and fam != 'recovery')
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _PRICE_EPS) and (not dead)
        viewed = str(pid) in self.book.products
        field, want = (self.briefing.need_field, self.briefing.need_value)
        spec = 2
        if field and want:
            got = self._variant_value(pid, sku, field)
            spec = 2 if got == want else -1 if got else 0
        ok = bool(live and viewed and (spec == 2) and self._brand_allowed(pid) and self._device_allowed(pid, sku))
        cat_ok = self._category_ok(pid)
        ok = ok and cat_ok
        if fam == 'justification':
            ok = ok and (not self._resale_title(pid)) and bool(self._brand_for(pid))
            if field and want:
                ok = ok and self._fact_value(pid, field) == want and (self._product_value(pid, field) == want)
        if fam == 'preference' and self.book.shock_key:
            dead_listing = self.book.listing_by_pid.get(self.book.shock_key.split('::', 1)[0])
            if dead_listing and self.book.listing_by_pid.get(str(pid)) == dead_listing:
                ok = False
        if fam == 'recall' and self.briefing.theme:
            ok = ok and all((t in _word_set(self._listing_title(pid)) for t in self.briefing.theme))
        if fam == 'ranking':
            ok = ok and (str(pid), str(sku)) in self.briefing.named_keys
            ok = ok and self._measure(pid, sku, 'refresh_rate') is not None and (self._measure(pid, sku, 'screen_size') is not None)
        return {'pid': str(pid), 'sku': str(sku), 'key': variant_key, 'price': price, 'live': live, 'viewed': viewed, 'spec': spec, 'cat_ok': cat_ok, 'ok': ok, 'relax': self._is_soft_pref(pid, sku) if fam == 'recovery' else 0}

    def _purchase_ok(self, pid: str, sku: str) -> tuple:
        st = self.book.status(pid, sku)
        k = EvidenceBook.variant_key(pid, sku)
        if st is None:
            return (False, 'inspect_stock the exact product_id and sku first')
        if (pid, sku) not in self.book.cart:
            return (False, 'add that exact variant to the cart first')
        if self.book.last_cart_seq is not None and self.book.stock_seq.get(k, 0) <= self.book.last_cart_seq:
            return (False, 're-check the variant with inspect_stock after the cart addition')
        if not st.get('in_stock'):
            return (False, 'that variant is out of stock; choose a different product')
        price = st.get('price')
        if self.budget is not None and price is not None and (price > self.budget + _PRICE_EPS):
            return (False, 'that variant costs %.2f, above the budget of %.2f' % (price, self.budget))
        if k in self.book.killed_keys:
            return (False, 'that variant was invalidated by the market change; choose a different product')
        return (True, '')

    def _viable(self) -> list:
        seen = set()
        out = []
        for variant_key in list(self.book.checked):
            pid, _, sku = variant_key.partition('::')
            if not sku or variant_key in seen:
                continue
            seen.add(variant_key)
            out.append(self._score_variant(pid, sku))
        return out

    def _ingest_result(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]
        self.chat.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls})
        self.transcript.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls, 'environment_result': result})
        if not isinstance(result, dict):
            self.book.seen_calls.discard(name + json.dumps(args, sort_keys=True))
            self.chat.append({'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps({'error': 'environment call failed; it will be retried automatically'})})
            return None
        calls = result.get('calls') or []
        wrap = (calls[0].get('observation') if calls and isinstance(calls[0], dict) else None) or {}
        obs = wrap.get('observation') if isinstance(wrap.get('observation'), dict) else wrap
        if isinstance(obs, dict) and wrap.get('error') and (not obs.get('error')):
            obs = dict(obs)
            obs['error'] = wrap.get('error')
        if wrap.get('done'):
            self.done = True
        self.book.ingest(name, args, obs, self.briefing)
        self.chat.append({'role': 'tool', 'tool_call_id': call_id, 'content': _clip_obs(json.dumps(obs), _OBS_LIMIT)})
        user = result.get('user_message')
        if isinstance(user, dict) and user.get('content'):
            text = str(user['content'])
            self.chat.append({'role': 'user', 'content': text})
            self._absorb_budget(text)
            if self.briefing.family in ('constraint', 'intent') and (not self.briefing.need_value):
                stated = self.briefing.ingest_spec(text)
                if stated:
                    self.brief_updated = True
                    self._coach('Firm requirement recorded: ' + stated + '. Only a variant that resolves to it may be bought.')
            if _ASK_RE.search(text):
                self.awaiting_reply = True
                self._coach('The shopper asked a question: answer it with message in your next turn, then continue.')
        if any((isinstance(c, dict) and (c.get('observation') or {}).get('done') for c in calls)):
            self.done = True
        return obs

    def _kickoff_question(self) -> str | None:
        if self.briefing.family == 'constraint' or self.briefing.need_locked:
            return 'Before I search: which hardware specification is your firm requirement (screen size, resolution class, refresh rate, or memory), and what value do you need?'
        if self.briefing.family == 'preference':
            return 'Quick question so I choose well: which of your preferences counts most, and in what order should I weigh the others? Let me know your priority.'
        return None

    def _strip_models(self, pid: str, title: str) -> str:
        """Drop model codes from a title so a product name is not read as a panel class."""
        for token in re.split('[;,]', str(self._raw_facts(pid).get('model') or '')):
            token = token.strip()
            stem = re.split('[-_]', token)[0] if token else ''
            if token and re.fullmatch('[a-z0-9]+(?:[-_][a-z0-9]+)+', token, re.I) and re.search('[a-z]', stem, re.I) and re.search('\\d', stem) and (not re.fullmatch('(?:[48]k|[ufq]?hd|fullhd|ultrahd|\\d+p)\\d*', stem, re.I)):
                title = re.sub('(?<![\\w-])' + re.escape(token) + '(?![\\w-])', '', title, flags=re.I)
        return title

    def _sample_tools(self) -> dict | None:
        for attempt in range(3):
            model = self.model_roster[min(self.roster_i, len(self.model_roster) - 1)]
            payload = {'model': model, 'messages': self.chat, 'tools': self.tools, 'tool_choice': 'required', 'temperature': 0}
            resp = _GATEWAY.post('/inference/chat/completions', json_data=payload)
            if resp and isinstance(resp, dict) and resp.get('choices'):
                msg = resp['choices'][0].get('message') or {}
                calls = msg.get('tool_calls') or []
                if calls:
                    return {'content': msg.get('content') or '', 'tool_calls': calls[:self.calls_per_turn]}
                self.chat.append({'role': 'assistant', 'content': msg.get('content') or ''})
                self._coach('No tool call was made. Choose exactly one of the supplied tools now.')
                continue
            if self.deadline.secs_left() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.roster_i = min(self.roster_i + 1, len(self.model_roster) - 1)
        return None

    def _has_fallback(self) -> bool:
        """Whether a first commitment can be made, or the shopper still owes information."""
        fam = self.briefing.family
        if self.book.adds:
            return True
        if self.briefing.need_locked:
            if self.turn < self.step_cap * 0.6:
                return False
            self.briefing.need_locked = False
            self._coach('The firm requirement was never stated, so the remaining requirements decide the choice.')
        if fam in ('intent', 'constraint') and self.turn < 4 and (not self.brief_updated):
            return False
        if fam == 'preference' and len(self.book.listing_set) < 2:
            return False
        roomy = self.turn < self.step_cap * 0.45 and self.deadline.secs_left() > 140
        if fam == 'recall' and roomy:
            if len(self.book.identity_set) < self._recall_quota() or self.search_count < 2:
                return False
        if roomy and (not self.pages_done) and (len({c['pid'] for c in self._viable() if c['ok']}) < 2):
            return False
        return True

    def _twin_search(self) -> tuple | None:
        """After the market kills the carted item, its own twins are the nearest replacements.

        A listing's other colors and sizes are separate products carrying the same
        title, so searching that title surfaces exactly the items that still meet
        every requirement the dead one met.
        """
        dead = self.book.shock_key
        if self.did_twin_search or not dead or 'search' not in self.allowed_tools:
            return None
        title = self._listing_title(dead.split('::', 1)[0])
        if not title:
            return None
        self.did_twin_search = True
        self._coach('The item that died is one of several listings sharing a title; searching that title surfaces its other colors and sizes, which meet the same requirements.')
        return ('search', {'query': ' '.join(title.split()[:8]), 'k': 10, 'in_stock': True})

    def _emit_shortlist(self) -> None:
        """Tell the model what the controller can and cannot yet verify."""
        cands = self._viable()
        if self.briefing.family == 'preference' and (not self.book.cart):
            good = [c for c in cands if c['ok']]
            good.sort(key=lambda c: (-self._priority_hits(c['pid'], c['sku']), c['price'] if c['price'] is not None else 1000000000000.0))
            if good:
                self._coach('Products that meet every firm requirement so far: ' + '; '.join(('%s::%s at %.2f' % (c['pid'], c['sku'], c['price'] or 0.0) for c in good[:6])) + ". Pick the one that best fits the shopper's stated priorities, inspect it, then add it.")
                return
        best = self._target()
        if best:
            self._coach('Verified candidate: %s::%s at %.2f. The controller will inspect, cart and order it; read other candidates only if one could rank better.' % (best['pid'], best['sku'], best['price'] or 0.0))
            return
        field, want = (self.briefing.need_field, self.briefing.need_value)
        blocked = [c for c in cands if c['viewed'] and c['live'] and (not c['ok'])]
        if blocked and field:
            sample = blocked[:3]
            self._coach("None of the products read so far can be verified. For each, the evidence for %s must resolve to exactly %s from the variant's own option text or the product's raw facts: %s. Look at different products." % (field, want, ', '.join((c['pid'] + '::' + c['sku'] for c in sample))))
        elif blocked:
            self._coach('The products read so far fail a firm requirement (category, brand, stock, budget or the kind of device requested). Search for different products.')

    def _target(self) -> dict | None:
        """The best fully verified candidate, or nothing while evidence is missing."""
        first = self.book.adds == 0
        fam = self.briefing.family
        good = [c for c in self._viable() if c['ok']]
        if fam == 'preference':
            carted = [c for c in good if (c['pid'], c['sku']) in self.book.cart]
            if carted:
                return carted[0]
            if good and self.turn >= self.step_cap - 8:
                good.sort(key=lambda c: (-self._priority_hits(c['pid'], c['sku']), c['price'] if c['price'] is not None else 1000000000000.0))
                return good[0]
            return None
        if fam == 'intent' and first and self.briefing.favor_brand:
            wanted = _word_set(self.briefing.favor_brand)
            named = [c for c in good if wanted <= _word_set(self._brand_for(c['pid'])) or wanted <= _word_set(self._listing_title(c['pid']))]
            if named:
                good = named
            elif self.turn < self.step_cap * 0.6:
                self._preferred_brand_hint()
                return None
        if fam == 'recovery' and first:
            exact = [c for c in good if c['relax'] == 0]
            if exact:
                good = exact
            elif self.turn < self.step_cap * 0.6:
                self._exact_match_hint()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self._rank_key(c, first))
        return good[0]

    def _stock_fresh(self, pid: str, sku: str) -> bool:
        k = EvidenceBook.variant_key(pid, sku)
        if k not in self.book.inspected_keys:
            return False
        return self.book.last_cart_seq is None or self.book.stock_seq.get(k, 0) > self.book.last_cart_seq

    def _recheck_after_cart(self) -> None:
        """The market moves on the first cart action, so re-read that item at once."""
        pid, sku = self.recheck_pair
        self.recheck_pair = None
        obs = self._execute('inspect_stock', {'product_id': pid, 'sku': sku})
        if not isinstance(obs, dict) or obs.get('error'):
            return
        stock = bool(obs.get('in_stock'))
        price = _as_float(obs.get('price'))
        over = self.budget is not None and price is not None and (price > self.budget + _PRICE_EPS)
        if stock and (not over):
            self._coach('Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it.' % (EvidenceBook.variant_key(pid, sku), price))
            return
        self.book.shock_key = EvidenceBook.variant_key(pid, sku)
        self.shock_step = self.turn
        self.book.killed_keys.add(EvidenceBook.variant_key(pid, sku))
        self.book.killed_pids.add(str(pid))
        self.compare_boosts = 0
        reason = 'went out of stock' if not stock else 'was repriced to %.2f, above the budget' % (price or 0.0)
        guidance = {'recovery': 'Recover with the smallest possible change: the same model in another color before any other model.', 'ranking': 'The remaining named candidates are re-ranked; the controller takes the new best.', 'preference': "Recover to a product with a different listing_id that still fits the shopper's priorities.", 'constraint': 'Choose a different product that still meets every requirement, not another variant of the same one.'}.get(self.briefing.family, 'Choose a different product that meets every firm requirement.')
        self._coach('Market change: the carted item %s %s. ' % (EvidenceBook.variant_key(pid, sku), reason) + guidance)
        self._emit_shortlist()

    def _rank_key(self, c: dict, first: bool) -> tuple:
        """Order eligible candidates the way the task's own priority does."""
        fam = self.briefing.family
        price = c['price'] if c['price'] is not None else 1000000000000.0
        if fam == 'ranking':
            hz = self._measure(c['pid'], c['sku'], 'refresh_rate') or 0.0
            inch = self._measure(c['pid'], c['sku'], 'screen_size') or 0.0
            return (-hz, -inch, price, c['key'])
        if fam == 'recovery':
            return (c['relax'], price, c['key'])
        extra = sum((1 for f, v in self.briefing.specs.items() if f != self.briefing.need_field and self._variant_value(c['pid'], c['sku'], f) != v))
        proven = 0 if _phone_signature(self._raw_facts(c['pid']), self._listing_title(c['pid']), self._options(c['pid'], c['sku'])) == 'yes' else 1
        if fam == 'intent' and first and self.briefing.favor_brand:
            wanted = _word_set(self.briefing.favor_brand)
            near = 0 if wanted <= _word_set(self._brand_for(c['pid'])) or wanted <= _word_set(self._listing_title(c['pid'])) else 1
            return (near, extra, proven, price, c['key'])
        return (extra, proven, price, c['key'])

    def _variant_value(self, pid: str, sku: str, field: str) -> str | None:
        """The value that holds for this exact variant: its own option text wins."""
        opt = self._options(pid, sku)
        vals = _nearby_values(field, opt)
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if _field_mentioned(field, opt):
            return None
        return self._product_value(pid, field)

    def _category_ok(self, pid: str) -> bool:
        """Whether the listing sits on the shelf the shopper asked for.

        Retailers spell a shelf many ways and nest it differently, so a shared
        word anywhere in the path counts. Where the request names a kind of
        product the catalog has no shelf for, the title has to carry it.
        """
        want = self.briefing.category
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
        qualifier = self.briefing.cat_qualifier
        if not qualifier:
            return True
        title = self._listing_title(pid)
        words = self._stems(title) | _word_set(title)
        if qualifier == 'gaming':
            fast = any((float(v[:-2]) >= 120 for v in _rates_from_text(title)))
            return 'monitor' in words and ('gaming' in words or fast)
        return 'over' in words and 'ear' in words or 'overear' in words or 'circumaural' in words

    def _post_env(self, envelope: dict) -> dict | None:
        """POST one solver-turn envelope; an identical envelope replays idempotently."""
        for attempt in range(4):
            result = _GATEWAY.post('/environment/call', json_data=envelope)
            if isinstance(result, dict) and result.get('calls'):
                return result
            if self.deadline.secs_left() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def _recall_quota(self) -> int:
        """How many distinct matching products to surface before buying.

        Surfacing more than the stated minimum is worth turns early on, but the
        catalog may simply not hold that many, so the margin lapses.
        """
        early = self.search_count < 8 and self.turn < self.step_cap * 0.5
        return self.briefing.identity_quota + (2 if early else 0)

    def _fact_value(self, pid: str, field: str) -> str | None:
        """The value the product's own labelled facts state for a concept.

        A reason quoted back to the shopper reads these facts leniently: a fact
        printed as a bare number under a unit-bearing label carries that label's
        unit, so "27" under the size label is 27 inches.
        """
        raw = self._raw_facts(pid).get(_EVIDENCE_LABEL[field])
        if not raw:
            return None
        text = str(raw).strip()
        found = _EXTRACTORS[field](text)
        if not found and field != 'resolution' and re.fullmatch('\\d+(?:\\.\\d+)?', text):
            unit = {'screen_size': 'in', 'refresh_rate': 'hz', 'ram': 'gb'}[field]
            return '%g' % float(text) + unit
        return _unique_or_none(found)

    def _absorb_budget(self, text: str) -> None:
        new = ShopperBrief._budget(text)
        if new is None or (self.budget is not None and new >= self.budget):
            return
        self.budget = new
        self.brief_updated = True
        self._coach('Budget updated by the shopper: at most %.2f %s. Every later choice must fit it.' % (new, self.briefing.currency))
        for pid, sku in list(self.book.cart):
            st = self.book.status(pid, sku) or {}
            price = st.get('price')
            if price is not None and price > new + _PRICE_EPS:
                self.book.killed_keys.add(EvidenceBook.variant_key(pid, sku))
                self._coach('The carted item %s costs %.2f, above the new budget; the controller will replace it.' % (EvidenceBook.variant_key(pid, sku), price))

    def _preferred_brand_hint(self) -> None:
        if self.nudged_brand:
            return
        self.nudged_brand = True
        self._coach("No %s product that meets every firm requirement has been read yet. The shopper's first choice is graded on that brand, so filter by brand=%s within the budget and read those products before any other." % (self.briefing.favor_brand, self.briefing.favor_brand))

    def _claims_for(self, pid: str, sku: str) -> list:
        """Claims proven by what the agent has already observed about this exact item."""
        out = []
        field, want = (self.briefing.need_field, self.briefing.need_value)
        if not field and self.briefing.family == 'justification':
            field, want = ('resolution', self._fact_value(pid, 'resolution'))
        if not field or not want:
            return []
        stated = dict(self.briefing.specs) or {field: want}
        stated.setdefault(field, want)
        for name, value in stated.items():
            if self._fact_value(pid, name) == value and self._product_value(pid, name) == value:
                out.append({'field': name, 'value': value})
        if not out:
            return []
        st = self.book.status(pid, sku) or {}
        if self.budget is not None:
            out.append({'field': 'price', 'value': 'at most %.2f' % self.budget})
        elif st.get('price') is not None:
            out.append({'field': 'price', 'value': '%.2f' % st['price']})
        out.append({'field': 'in_stock', 'value': 'true'})
        brand = self._brand_for(pid)
        if brand:
            out.append({'field': 'brand', 'value': brand})
        return out

    def drive(self) -> list:
        question = self._kickoff_question()
        last_forced, repeats = (None, 0)
        while not self.done and self.turn < self.step_cap:
            if self.deadline.secs_left() < 25 or self.turn >= self.step_cap - 3:
                self._rush_checkout()
                break
            if question and (not self.asked_once) and ('message' in self.allowed_tools):
                self.asked_once = True
                self._execute('message', {'content': question})
                continue
            if self.recheck_pair and 'inspect_stock' in self.allowed_tools:
                self._recheck_after_cart()
                continue
            plan = self._forced_step()
            if plan is not None:
                name, args = plan
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    if name in ('inspect_stock', 'add_to_cart'):
                        self.book.killed_keys.add(EvidenceBook.variant_key(args.get('product_id'), args.get('sku')))
                    if name == 'compare':
                        self.blocked_ids.update((str(p) for p in args.get('product_ids') or []))
                    if name == 'message':
                        self.awaiting_reply = False
                    last_forced, repeats = (None, 0)
                    self._coach('That step changed nothing, so the controller moved on to other candidates.')
                    continue
                obs = self._execute(name, args)
                if name == 'message':
                    self.awaiting_reply = False
                    self.wait_ticks = 0
                if name == 'add_to_cart' and isinstance(obs, dict) and (not obs.get('error')):
                    added = obs.get('added') or {}
                    if added.get('product_id'):
                        self.recheck_pair = (str(added['product_id']), str(added['sku']))
                continue
            last_forced, repeats = (None, 0)
            proposal = self._sample_tools()
            if proposal is None:
                self._rush_checkout()
                break
            call = proposal['tool_calls'][0]
            name = (call.get('function') or {}).get('name') or ''
            args = _parse_tool_args(call)
            content = proposal.get('content') or ''
            reprompts = 0
            while True:
                verdict = self._veto_reason(name, args)
                if verdict is None:
                    break
                if reprompts >= _REJECT_RETRIES:
                    self.veto_streak += 1
                    self._coach('That call was not made: ' + verdict + '. Choose a different, valid action.')
                    if self.veto_streak >= 3:
                        self._emit_shortlist()
                        self.veto_streak = 0
                    verdict = 'skip'
                    break
                reprompts += 1
                self.chat.append({'role': 'assistant', 'content': content, 'tool_calls': [call]})
                self.chat.append({'role': 'tool', 'tool_call_id': call.get('id') or 'x', 'content': json.dumps({'rejected': verdict})})
                self.transcript.append({'role': 'assistant', 'content': '[rejected ' + name + '] ' + verdict[:200]})
                again = self._sample_tools()
                if again is None:
                    break
                call = again['tool_calls'][0]
                name = (call.get('function') or {}).get('name') or ''
                args = _parse_tool_args(call)
                content = again.get('content') or ''
            if verdict == 'skip':
                continue
            self.veto_streak = 0
            if name not in self.allowed_tools:
                continue
            if name == 'place_test_order':
                pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
                if not pid and self.book.cart:
                    pid, sku = self.book.cart[-1]
                args = self._checkout_args(pid, sku)
            if name == 'message':
                if not self.awaiting_reply:
                    self.ask_count += 1
                self.awaiting_reply = False
                self.wait_ticks = 0
            obs = self._execute(name, args, content, call.get('id'))
            if name == 'add_to_cart' and isinstance(obs, dict) and (not obs.get('error')):
                added = obs.get('added') or {}
                if added.get('product_id'):
                    self.recheck_pair = (str(added['product_id']), str(added['sku']))
            if name in ('search', 'filter'):
                self.search_count += 1
                if self.briefing.family == 'recall' and self.briefing.identity_quota:
                    have, need = (len(self.book.identity_set), self._recall_quota())
                    self._coach('Distinct matching products surfaced so far: %d of the required %d.' % (have, need) + ('' if have >= need else ' Keep searching with different wording.'))
            if name == 'inspect_cart':
                self.cart_reads += 1
            if name in ('view', 'compare'):
                self._emit_shortlist()
        return self.transcript

    def _rush_checkout(self) -> None:
        """Out of time or turns: reach a valid order in as few actions as possible."""
        for pid, sku in reversed(self.book.cart):
            ok, _why = self._purchase_ok(pid, sku)
            if ok and (not self.done):
                self._execute('place_test_order', self._checkout_args(pid, sku))
                return
        if self.done or self.turn > self.step_cap - 3:
            return
        good = [c for c in self._viable() if c['ok']]
        if not good and self.briefing.family != 'justification':
            good = [c for c in self._viable() if c['live'] and c['spec'] == 2 and self._brand_allowed(c['pid']) and c['cat_ok'] and (not self.briefing.named_keys or (c['pid'], c['sku']) in self.briefing.named_keys)]
        if not good:
            return
        good.sort(key=lambda c: self._rank_key(c, self.book.adds == 0))
        pid, sku = (good[0]['pid'], good[0]['sku'])
        if (pid, sku) not in self.book.cart:
            obs = self._execute('add_to_cart', {'product_id': pid, 'sku': sku})
            if self.done or not isinstance(obs, dict) or obs.get('error'):
                return
        obs = self._execute('inspect_stock', {'product_id': pid, 'sku': sku})
        if self.done or not isinstance(obs, dict) or (not obs.get('in_stock')):
            return
        price = _as_float(obs.get('price'))
        if self.budget is not None and price is not None and (price > self.budget + _PRICE_EPS):
            return
        self._execute('place_test_order', self._checkout_args(pid, sku))

    def _exact_match_hint(self) -> None:
        if self.nudged_exact or not self.briefing.favor_model:
            return
        self.nudged_exact = True
        self._coach("The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own and view the results so every color variant is on record; only that exact pairing may be the first cart item." % (self.briefing.favor_model, ', ' + self.briefing.favor_color if self.briefing.favor_color else ''))

    def _sibling_search(self) -> tuple | None:
        """A found model in the wrong color: its other colors share the same title."""
        if self.did_sibling_search:
            return None
        rows = [c for c in self._viable() if c['live'] and c['viewed'] and (c['relax'] == 0)]
        if rows:
            return None
        near = [c for c in self._viable() if c['viewed'] and c['relax'] == 1]
        if not near:
            return None
        title = self._listing_title(near[0]['pid'])
        if not title:
            return None
        self.did_sibling_search = True
        return ('search', {'query': ' '.join(title.split()[:8]), 'k': 10, 'in_stock': True})

    def _product_row(self, pid: str) -> dict:
        return self.book.products.get(str(pid)) or {}

    def _veto_reason(self, name: str, args: dict) -> str | None:
        if name not in self.allowed_tools:
            return 'unknown tool; choose one of ' + str(sorted((n for n in self.allowed_tools if n)))
        if name == 'place_test_order':
            pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
            if not pid and self.book.cart:
                pid, sku = self.book.cart[-1]
            ok, why = self._purchase_ok(pid, sku)
            if not ok:
                return 'order refused: ' + why
            if self.briefing.family == 'recall' and len(self.book.identity_set) < self._recall_quota():
                return 'order refused: only %d distinct matching products have been surfaced; search with different wording until at least %d appear' % (len(self.book.identity_set), self._recall_quota())
            if self.briefing.family == 'preference' and len(self.book.listing_set) < 2:
                return 'order refused: compare at least two products with different listing_id values first'
        if name == 'add_to_cart':
            pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
            if not pid or not sku:
                return 'add_to_cart needs both product_id and sku'
            c = self._score_variant(pid, sku)
            variant_key = EvidenceBook.variant_key(pid, sku)
            if not c['live']:
                return 'that variant is out of stock, over budget, or already invalidated'
            if not c['viewed']:
                return 'view or compare that product first so its raw facts are on record'
            if variant_key not in self.book.inspected_keys:
                return 'inspect_stock that exact variant first: its current price and stock decide whether it can be bought'
            if c['spec'] != 2 and self.briefing.need_field:
                return 'that variant does not resolve to %s = %s from its own option text or raw facts' % (self.briefing.need_field, self.briefing.need_value)
            if not c['ok']:
                return 'that product fails a firm requirement (category, brand, or the kind of device requested)'
            if self.briefing.family == 'preference' and len(self.book.listing_set) < 2:
                return 'compare two products with different listing_id values before the first cart addition'
            if self.briefing.family == 'recovery' and (not self.book.adds) and self.briefing.favor_model and self._is_soft_pref(pid, sku):
                return 'the first cart item must be the exact model and color the shopper named; the other colors are separate listings sharing the same title, so search that title and read them first'
            if not self._has_fallback():
                return 'hold: read one more product that also meets every requirement first, so there is a fallback when the market moves right after the cart addition'
        if name == 'message':
            text = str(args.get('content') or '')
            if not text.strip():
                return 'message content is required'
            if not self.awaiting_reply and '?' not in text:
                return "do not send status updates; use message only to answer the shopper's question or to ask one"
            if not self.awaiting_reply and self.ask_count >= 2:
                return 'the shopper has already answered; decide from the evidence you have rather than asking again'
        if name == 'inspect_cart' and self.cart_reads >= 1:
            return 'the cart contents are already known from earlier observations'
        variant_key = name + json.dumps(args, sort_keys=True)
        if variant_key in self.book.seen_calls and name in ('search', 'filter', 'view', 'compare'):
            return 'identical call already made; change the arguments'
        self.book.seen_calls.add(variant_key)
        return None

    def _is_soft_pref(self, pid: str, sku: str) -> int:
        """How far a candidate sits from the shopper's exact preferred model and color."""
        model, color = (self.briefing.favor_model, self.briefing.favor_color)
        if not model:
            return 0
        squash = lambda t: re.sub('[^a-z0-9]+', '', (t or '').lower())
        blob = squash(self._listing_title(pid) + ' ' + ' '.join((str(v) for v in self._raw_facts(pid).values())))
        same_model = squash(model) in blob
        shade = (self._options(pid, sku) + ' ' + str(self._raw_facts(pid).get('color') or '') + ' ' + self._listing_title(pid)).lower()
        same_color = bool(color) and re.search('(?<![a-z])' + re.escape(color) + '(?![a-z])', shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def _wish_words(self) -> set:
        """Content words from the shopper's own wording, gathered once."""
        if self._wish_set is None:
            said = self.briefing.goal + ' ' + ' '.join((str(m.get('content') or '') for m in self.chat if m.get('role') == 'user'))
            drop = _word_set(self.briefing.category or '') | {'the', 'and', 'for', 'with', 'that', 'this', 'you', 'your', 'find', 'need', 'want', 'order', 'stock', 'place', 'under', 'over', 'one', 'php', 'test', 'buy', 'rather', 'than', 'pay', 'bit', 'more', 'settle', 'something', 'its', 'would', 'like', 'have'}
            self._wish_set = {w for w in _word_set(said) if len(w) > 3 and (not w.isdigit()) and (w not in drop)}
        return self._wish_set

    def _measure(self, pid: str, sku: str, field: str) -> float | None:
        value = self._variant_value(pid, sku, field)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    def _brand_allowed(self, pid: str) -> bool:
        want = self.briefing.brand
        if not want:
            return True
        wanted = _word_set(want)
        return bool(wanted <= _word_set(self._brand_for(pid)) or wanted <= _word_set(self._listing_title(pid)))

    def _checkout_args(self, pid: str, sku: str) -> dict:
        out = {'product_id': pid, 'sku': sku}
        if self.briefing.family == 'justification':
            claims = self._claims_for(pid, sku)
            if claims:
                out['justification'] = {'claims': claims}
        return out

    def _resale_title(self, pid: str) -> bool:
        """Listings that describe someone else's listing, or a used unit, are never the answer."""
        return bool(re.search('customer (?:rating|review)|\\bcompare \\b|refurb|pre-owned|restored|renewed', self._listing_title(pid), re.I))

    def _product_value(self, pid: str, field: str) -> str | None:
        """Product-wide value: labelled facts plus the title, and nothing when they disagree."""
        vals: set = set()
        conflict = False
        raw = self._raw_facts(pid).get(_EVIDENCE_LABEL[field])
        if raw:
            found = _EXTRACTORS[field](str(raw))
            conflict = conflict or len(found) > 1
            if field == 'resolution' and _QHD_RE.search(str(raw)):
                conflict = True
            vals |= found
        title = self._listing_title(pid)
        if field == 'resolution':
            title = self._strip_models(pid, title)
            if _QHD_RE.search(title):
                conflict = True
        vals |= _nearby_values(field, title)
        return None if conflict else _unique_or_none(vals)

    def _priority_hits(self, pid: str, sku: str) -> int:
        """How much of what the shopper actually asked for this product's own facts mention."""
        wants = self._wish_words()
        if not wants:
            return 0
        have = _word_set(' '.join((str(v) for v in self._raw_facts(pid).values())) + ' ' + self._listing_title(pid) + ' ' + self._options(pid, sku))
        return len(wants & have)

    def _device_allowed(self, pid: str, sku: str) -> bool:
        """Categories that name a device are graded on the listing being that device.

        Where the task itself turns on device identity, only positive evidence
        will do; elsewhere a listing is only ruled out when the evidence
        contradicts it, so a thin product page is not treated as a disqualifier.
        """
        if not _words_match(self.briefing.category or '', 'smartphone'):
            return True
        verdict = _phone_signature(self._raw_facts(pid), self._listing_title(pid), self._options(pid, sku))
        if self.briefing.family in ('constraint', 'justification', 'recovery'):
            return verdict == 'yes'
        return verdict != 'no'

    def _policy_text(self) -> str:
        play = _FAMILY_COACHING.get(self.briefing.family, _FAMILY_COACHING['generic'])
        extra = ''
        if self.briefing.family == 'recall' and self.briefing.identity_quota:
            extra = ' You must surface at least %d distinct matching products in search or filter results before ordering; the controller reports the count after each search.' % self.briefing.identity_quota
        return _BASE_POLICY + '\n\nFamily playbook: ' + play + extra

    def _cat_path(self, pid: str) -> list:
        path = self._product_row(pid).get('category_path')
        return list(path) if isinstance(path, list) else []

    def _read_more(self) -> tuple | None:
        """Turn the search results already in hand into read product pages, then widen.

        Reading pages is what turns rows into candidates the controller can verify,
        and a second verified product is what makes the market change survivable.
        """
        fam = self.briefing.family
        if fam == 'recall' and len(self.book.identity_set) < self._recall_quota():
            return None
        twin = self._twin_search()
        if twin is not None:
            return twin
        leads = self._unseen_leads()
        if leads and self.compare_boosts < 8 and (self.turn >= 2):
            self.compare_boosts += 1
            if len(leads) > 1 and 'compare' in self.allowed_tools:
                return ('compare', {'product_ids': leads[:5]})
            return ('view', {'product_id': leads[0]})
        probe = (self.briefing.brand or '', '%.2f' % (self.budget or 0.0), str(self.briefing.need_value))
        if self.turn < 2:
            return None
        if probe in self.probes:
            self.pages_done = True
            return None
        self.probes.add(probe)
        if 'filter' in self.allowed_tools and self.briefing.category:
            args = {'category': self.briefing.category, 'in_stock': True, 'k': 10}
            if self.budget is not None:
                args['max_price'] = self.budget
            if self.briefing.brand:
                args['brand'] = self.briefing.brand
            return ('filter', args)
        if 'search' in self.allowed_tools:
            words = [w for w in (self.briefing.brand, self.briefing.need_value, self.briefing.category) if w]
            if words:
                return ('search', {'query': ' '.join(words), 'in_stock': True, 'k': 10})
        self.pages_done = True
        return None

    def _forced_step(self) -> tuple | None:
        """The action the controller takes itself, or nothing to let the model explore."""
        if self.done:
            return None
        if self.awaiting_reply:
            if self.wait_ticks < 2 or 'message' not in self.allowed_tools:
                self.wait_ticks += 1
                return None
            return ('message', {'content': self._answer_text()})
        fam = self.briefing.family
        if fam == 'ranking' and 'compare' in self.allowed_tools:
            missing = [p for p, _s in self.briefing.named_keys if p not in self.book.products and p not in self.blocked_ids]
            if missing:
                return ('compare', {'product_ids': missing[:5]})
        if fam == 'recovery' and (not self.book.adds) and self.briefing.favor_model and ('search' in self.allowed_tools):
            forced = self._sibling_search()
            if forced is not None:
                return forced
        target = self._target()
        if target is None or not self._has_fallback():
            forced = self._read_more()
            if forced is not None:
                return forced
            if target is None:
                return None
        pid, sku = (target['pid'], target['sku'])
        if (pid, sku) in self.book.cart:
            if not self._stock_fresh(pid, sku):
                return ('inspect_stock', {'product_id': pid, 'sku': sku})
            return ('place_test_order', self._checkout_args(pid, sku))
        if not self._has_fallback():
            return None
        for cpid, csku in self.book.cart:
            if EvidenceBook.variant_key(cpid, csku) in self.book.killed_keys and 'remove_from_cart' in self.allowed_tools:
                return ('remove_from_cart', {'product_id': cpid, 'sku': csku})
        if not self._stock_fresh(pid, sku):
            return ('inspect_stock', {'product_id': pid, 'sku': sku})
        return ('add_to_cart', {'product_id': pid, 'sku': sku})

    def _raw_facts(self, pid: str) -> dict:
        f = self._product_row(pid).get('facts')
        return f if isinstance(f, dict) else {}

    def _execute(self, name: str, args: dict, content: str='', tool_call_id: str | None=None) -> dict | None:
        if self.held_envelope is not None:
            pend = self.held_envelope
            replay = self._post_env(pend['envelope'])
            if replay is None:
                self._coach('The environment is not answering; the previous action is still pending and will be retried.')
                return None
            self.held_envelope = None
            self.turn = pend['turn']
            self._ingest_result(pend['name'], pend['args'], pend['content'], pend['call_id'], replay)
            if self.done:
                return None
        turn = self.turn + 1
        group = self.episode_id + '-turn-' + str(turn)
        envelope = {**self.binding, 'call_id': group, 'idempotency_key': group, 'turn': turn, 'calls': [{'call_id': group + '-1', 'action': {'name': name, 'args': args}}]}
        result = self._post_env(envelope)
        if result is None:
            self.held_envelope = {'envelope': envelope, 'turn': turn, 'name': name, 'args': args, 'content': content, 'call_id': tool_call_id or 'agent-' + str(turn)}
        else:
            self.turn = turn
        return self._ingest_result(name, args, content, tool_call_id or 'agent-' + str(turn), result)

    def _brand_for(self, pid: str) -> str:
        brand = self._product_row(pid).get('brand')
        if brand:
            return str(brand)
        for variant_key, row in self.book.rows.items():
            if variant_key.startswith(str(pid) + '::') and row.get('brand'):
                return str(row['brand'])
        return ''

    @staticmethod
    def _stems(text: str) -> set:
        return {w.rstrip('s') for w in re.findall('[a-z]+', (text or '').lower())}

    def _unseen_leads(self) -> list:
        """Promising variants whose product page has not been read yet."""
        leads = []
        for variant_key, row in self.book.rows.items():
            pid, _, sku = variant_key.partition('::')
            if not sku or str(pid) in self.book.products or str(pid) in self.book.killed_pids:
                continue
            st = self.book.checked.get(variant_key) or {}
            price = st.get('price')
            if not st.get('in_stock') or price is None or (self.budget is not None and price > self.budget + _PRICE_EPS):
                continue
            title = str(row.get('title') or '')
            if self.briefing.theme and (not all((t in _word_set(title) for t in self.briefing.theme))):
                continue
            if self.briefing.favor_model and self.book.adds == 0 and (not self.book.shock_key):
                squashed = re.sub('[^a-z0-9]+', '', title.lower())
                if re.sub('[^a-z0-9]+', '', self.briefing.favor_model.lower()) not in squashed:
                    continue
            if self.briefing.brand:
                wanted = _word_set(self.briefing.brand)
                if not (wanted <= _word_set(str(row.get('brand') or '')) or wanted <= _word_set(title)):
                    continue
            leads.append((price, str(pid)))
        leads.sort()
        out = []
        for _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def _options(self, pid: str, sku: str) -> str:
        return self.book.options.get(EvidenceBook.variant_key(pid, sku), '')

    def _answer_text(self) -> str:
        """A short, factual answer to the shopper, drawn only from observed evidence."""
        target = self._target() or (self._viable() or [None])[0]
        if not target:
            return 'I am still checking the options against your requirements; nothing is decided yet.'
        pid, sku = (target['pid'], target['sku'])
        parts = ['I am going with ' + (self._listing_title(pid) or pid)[:90]]
        field, want = (self.briefing.need_field, self.briefing.need_value)
        if field and want and (self._variant_value(pid, sku, field) == want):
            parts.append('its listed ' + field.replace('_', ' ') + ' is ' + want)
        if target.get('price') is not None:
            parts.append('it is %.2f %s and in stock' % (target['price'], self.briefing.currency))
        return ', '.join(parts) + '.'

    def _listing_title(self, pid: str) -> str:
        title = str(self._product_row(pid).get('title') or '')
        if title:
            return title
        for variant_key, row in self.book.rows.items():
            if variant_key.startswith(str(pid) + '::'):
                return str(row.get('title') or '')
        return ''

    def _coach(self, text: str) -> None:
        self.chat.append({'role': 'user', 'content': '[note] ' + text})
        self.transcript.append({'role': 'assistant', 'content': '[note] ' + text[:300]})

class ShopperBrief:
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
        self.named_keys = [(pid, sku.rstrip('.,;:')) for pid, sku in _VARIANT_RE.findall(self.goal)]
        phrase = _MATCH_PHRASE_RE.search(self.goal)
        quoted = _DQUOTE_RE.findall(self.goal)
        source = phrase.group(1) if phrase else quoted[0] if quoted else ''
        self.theme = re.findall('[a-z0-9]+', source.lower())
        sm = _MIN_DISTINCT_RE.search(self.rules)
        self.identity_quota = int(sm.group(1)) if sm else 0
        pm = _FAVOR_ITEM_RE.search(self.goal)
        self.favor_color = pm.group(1).lower() if pm else None
        self.favor_model = pm.group(2) if pm else None
        self.category = self._category(self.goal)
        self.brand = self._brand(self.goal, self.category)
        self.favor_brand = None if pm else self._pref_brand(self.goal)
        self.cat_qualifier = self._qualifier(self.goal)
        self.specs = self._specs(self.goal)
        self.need_field, self.need_value = self._spec(self.goal)
        self.need_locked = self.need_field is None and 'firm requirement' in self.goal.lower()
        self.family = self._family()

    @staticmethod
    def _category(goal: str) -> str | None:
        m = _CATEGORY_RE.search(goal or '')
        if m:
            return m.group(1).strip()
        m = re.search('\\b([a-z][a-z\\-]{2,20})\\s*(?:,|\\s(?:at or (?:under|below)|at most|under|below|within|with)\\b)', goal or '', re.I)
        return m.group(1) if m else None

    def _family(self) -> str:
        if self.named_keys:
            return 'ranking'
        if self.identity_quota:
            return 'recall'
        if self.favor_model:
            return 'recovery'
        for name, sign in _FAMILY_HINTS:
            if sign in self.rules:
                return name
        if self.theme:
            return 'recall'
        if _JUSTIFY_RE.search(self.goal):
            return 'justification'
        if self.favor_brand:
            return 'intent'
        if self.need_locked or self.need_field or self.brand:
            return 'constraint'
        return 'preference'

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
            value = _target_from_text(field, goal)
            if value:
                out[field] = value
        return out

    @staticmethod
    def _brand(goal: str, category: str | None) -> str | None:
        """A brand named as part of the requested item, e.g. 'a <Brand> <item>'."""
        if not category:
            return None
        m = re.search('\\b(?:an?|the)\\s+([^,.;]{0,60}?)\\b' + re.escape(category), goal or '', re.I)
        if not m:
            return None
        words = [w for w in re.findall("[\\w&.'\\-]+", _SPEC_NOISE_RE.sub(' ', m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and (not any((c.isdigit() for c in w))) and (not _res_from_text(w))]
        return ' '.join(keep) or None

    @staticmethod
    def _budget(text: str) -> float | None:
        best = None
        for m in _CAP_RE.finditer(text or ''):
            whole = m.group(1).replace(',', '')
            frac = m.group(2) or '0'
            try:
                val = float(whole + '.' + frac)
            except ValueError:
                continue
            if best is None or val < best:
                best = val
        return best

    def _spec(self, goal: str) -> tuple:
        """The requirement the controller gates on: the first one stated."""
        for field, value in self.specs.items():
            return (field, value)
        return (None, None)

    def ingest_spec(self, text: str) -> str | None:
        """Record a requirement the shopper reveals later; returns its description."""
        for field in ('screen_size', 'refresh_rate', 'ram', 'resolution'):
            value = _target_from_text(field, text)
            if value:
                self.need_field, self.need_value = (field, value)
                self.specs[field] = value
                self.need_locked = False
                return field + ' = ' + value
        return None

    @staticmethod
    def _pref_brand(goal: str) -> str | None:
        m = _FAVOR_BRAND_RE.search(goal or '')
        if not m:
            return None
        name = m.group(1).strip(' .,')
        words = name.split()
        if not words or len(words) > 4 or (not any((w[:1].isupper() for w in words))):
            return None
        return name

def agent_main(problem_data: dict) -> list:
    try:
        return SessionPilot(problem_data).drive()
    except Exception as exc:
        return [{'role': 'assistant', 'content': 'agent error: ' + type(exc).__name__ + ': ' + str(exc)[:200]}]


__all__ = ["agent_main"]
